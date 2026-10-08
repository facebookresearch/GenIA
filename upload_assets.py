#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Upload the page's media (assets/) to Google Cloud Storage.

The page's media is not committed: the published page loads it from the GCS
folder named by `assetBase` in data.js (set via ASSET_BASE_URL in
collect_assets.py), in the form

    https://storage.googleapis.com/<bucket>/<prefix>

This script reads that same value, so where the page looks and where this
uploads cannot drift apart. It uploads exactly the files data.js references
(a referenced folder — the 3D gallery names a COLMAP model by its folder —
counts as every file in it), not the collector's assets/.qual/ cache, nor
anything left unreferenced. It skips files whose content is already there
(MD5 match), and sets each object's Content-Type explicitly — Python's
mimetypes does not know .webp everywhere.

    python upload_assets.py --list        # offline: what would go where
    python upload_assets.py --check       # no credentials: is the bucket complete and correct?
    python upload_assets.py --dry-run     # compare with the bucket, change nothing
    python upload_assets.py               # upload new/changed files
    python upload_assets.py --delete      # ...and delete remote files no longer referenced

Needs `pip install google-cloud-storage` and Application Default Credentials:
`gcloud auth application-default login`, or GOOGLE_APPLICATION_CREDENTIALS
pointing at a service-account key with write access to the bucket. The objects
must also be publicly readable (the script checks, but does not change IAM).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent

CONTENT_TYPES = {
    ".mp4": "video/mp4",
    ".webp": "image/webp",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}

GCS_URL = re.compile(r"^https://storage\.googleapis\.com/([^/]+)/?(.*?)/?$")


def load_manifest(path: Path) -> dict:
    text = path.read_text()
    marker = "window.PAGE_DATA = "
    if marker not in text:
        sys.exit(f"!! {path} does not look like the page manifest (no `{marker}`)")
    return json.loads(text.split(marker, 1)[1].rstrip().rstrip(";"))


def referenced_assets(node, found: set) -> set:
    """Every `assets/...` path anywhere in the manifest."""
    if isinstance(node, dict):
        for v in node.values():
            referenced_assets(v, found)
    elif isinstance(node, list):
        for v in node:
            referenced_assets(v, found)
    elif isinstance(node, str) and node.startswith("assets/"):
        found.add(node)
    return found


def content_type(path: str) -> str:
    ext = Path(path).suffix.lower()
    return CONTENT_TYPES.get(ext) or mimetypes.guess_type(path)[0] or "application/octet-stream"


def local_md5(path: Path) -> str:
    """Base64 MD5, the form GCS reports in `blob.md5_hash`."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode()


def remote_prefix_url(bucket: str, prefix: str) -> str:
    return f"gs://{bucket}/{prefix}"


def public_listing(bucket: str, prefix: str) -> dict:
    """{name: object metadata} via the anonymous JSON API — works when the
    bucket grants allUsers roles/storage.objectViewer (which includes list)."""
    out, token = {}, None
    while True:
        q = {"prefix": prefix, "maxResults": "1000", "fields": "items(name,md5Hash,contentType),nextPageToken"}
        if token:
            q["pageToken"] = token
        url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o?{urllib.parse.urlencode(q)}"
        with urllib.request.urlopen(url, timeout=30) as r:
            page = json.load(r)
        out.update({o["name"]: o for o in page.get("items", [])})
        token = page.get("nextPageToken")
        if not token:
            return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=HERE / "data.js", help="page manifest (default: data.js)")
    ap.add_argument("--list", action="store_true", help="offline: print what would be uploaded where, then stop")
    ap.add_argument("--check", action="store_true",
                    help="no credentials needed: compare the public bucket listing with data.js, change nothing")
    ap.add_argument("--dry-run", action="store_true", help="compare with the bucket but change nothing")
    ap.add_argument("--delete", action="store_true",
                    help="also delete remote objects under <prefix>/assets/ that data.js no longer references")
    ap.add_argument("--cache-control", default="public, max-age=3600",
                    help="Cache-Control for uploaded objects (default: public, max-age=3600)")
    ap.add_argument("--project", default=None, help="GCP project for the client (default: from credentials)")
    ap.add_argument("-j", "--jobs", type=int, default=8, help="parallel uploads (default 8)")
    args = ap.parse_args()

    manifest = load_manifest(args.data)
    base = manifest.get("assetBase")
    if not base:
        sys.exit("!! data.js has no `assetBase`. Set ASSET_BASE_URL in collect_assets.py "
                 "(https://storage.googleapis.com/<bucket>/<prefix>) and regenerate data.js.")
    m = GCS_URL.match(base)
    if not m:
        sys.exit(f"!! assetBase {base!r} is not a https://storage.googleapis.com/<bucket>/<prefix> URL")
    bucket_name, prefix = m.group(1), m.group(2)

    def object_name(rel: str) -> str:
        return f"{prefix}/{rel}" if prefix else rel

    refs = set()
    for r in referenced_assets(manifest, set()):
        if (HERE / r).is_dir():  # a folder counts as every file in it
            refs |= {f.relative_to(HERE).as_posix() for f in (HERE / r).rglob("*") if f.is_file()}
        else:
            refs.add(r)
    refs = sorted(refs)
    missing = [r for r in refs if not (HERE / r).is_file()]
    if missing:
        sys.exit(f"!! {len(missing)} referenced file(s) missing locally, e.g. {missing[:3]} "
                 "— run collect_assets.py first")
    total = sum((HERE / r).stat().st_size for r in refs)
    dest = f"gs://{bucket_name}/{prefix + '/' if prefix else ''}assets/"
    print(f"{len(refs)} files, {total / 1e6:.1f} MB -> {dest}")
    print(f"served as {base.rstrip('/')}/assets/...")
    if args.list:
        by_type: dict[str, int] = {}
        for r in refs:
            by_type[content_type(r)] = by_type.get(content_type(r), 0) + 1
        for ct, n in sorted(by_type.items()):
            print(f"  {n:5d}  {ct}")
        return

    if args.check:
        try:
            remote = public_listing(bucket_name, object_name("assets/"))
        except urllib.error.HTTPError as e:
            sys.exit(f"!! cannot list the bucket anonymously (HTTP {e.code}); use --dry-run with credentials")
        missing = [r for r in refs if object_name(r) not in remote]
        changed = [r for r in refs if object_name(r) in remote
                   and remote[object_name(r)].get("md5Hash") != local_md5(HERE / r)]
        wrong_ct = [r for r in refs if object_name(r) in remote
                    and remote[object_name(r)].get("contentType") != content_type(r)]
        extra = sorted(set(remote) - {object_name(r) for r in refs})
        print(f"bucket has {len(remote)} objects under {remote_prefix_url(bucket_name, object_name('assets/'))}")
        for label, items in [("missing", missing), ("content differs from local", changed),
                             ("wrong Content-Type", wrong_ct), ("not referenced by the page", extra)]:
            print(f"  {label}: {len(items)}" + (f"  e.g. {items[:3]}" if items else ""))
        ok = not (missing or changed or wrong_ct)
        print("OK: every referenced file is in the bucket, identical, with the right type" if ok
              else "!! the bucket does not match the page yet")
        sys.exit(0 if ok else 1)

    try:
        from google.cloud import storage
        from google.auth.exceptions import DefaultCredentialsError
    except ImportError:
        sys.exit("!! needs `pip install google-cloud-storage`")
    try:
        client = storage.Client(project=args.project)
    except DefaultCredentialsError as e:
        sys.exit(f"!! no Google credentials ({e}). Run `gcloud auth application-default login` "
                 "or set GOOGLE_APPLICATION_CREDENTIALS.")
    bucket = client.bucket(bucket_name)

    remote_prefix = object_name("assets/")
    remote = {b.name: b for b in client.list_blobs(bucket_name, prefix=remote_prefix)}

    to_upload, to_patch = [], []
    for rel in refs:
        blob = remote.get(object_name(rel))
        if blob is None or blob.md5_hash != local_md5(HERE / rel):
            to_upload.append(rel)
        elif blob.content_type != content_type(rel) or blob.cache_control != args.cache_control:
            to_patch.append(blob)
    wanted = {object_name(r) for r in refs}
    stale = sorted(n for n in remote if n not in wanted)

    up_bytes = sum((HERE / r).stat().st_size for r in to_upload)
    print(f"upload {len(to_upload)} ({up_bytes / 1e6:.1f} MB), fix metadata on {len(to_patch)}, "
          f"unchanged {len(refs) - len(to_upload) - len(to_patch)}, "
          f"remote-only {len(stale)}{' (will delete)' if args.delete else ''}")
    for n in stale[:10]:
        print(f"  remote-only: {n}")
    if args.dry_run:
        print("dry run — nothing changed")
        return

    def upload(rel: str) -> None:
        blob = bucket.blob(object_name(rel))
        blob.cache_control = args.cache_control
        blob.upload_from_filename(str(HERE / rel), content_type=content_type(rel))

    def patch(blob) -> None:
        blob.content_type = content_type(blob.name)
        blob.cache_control = args.cache_control
        blob.patch()

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for i, _ in enumerate(pool.map(upload, to_upload), 1):
            if i % 100 == 0 or i == len(to_upload):
                print(f"  uploaded {i}/{len(to_upload)}")
        list(pool.map(patch, to_patch))
        if args.delete and stale:
            list(pool.map(lambda n: bucket.blob(n).delete(), stale))
            print(f"  deleted {len(stale)} remote-only object(s)")

    # The page fetches anonymously, so check one object the way a visitor would.
    probe = f"{base.rstrip('/')}/{refs[0]}"
    try:
        with urllib.request.urlopen(urllib.request.Request(probe, method="HEAD"), timeout=20) as r:
            print(f"public read OK ({r.status}): {probe}")
    except urllib.error.HTTPError as e:
        print(f"!! {probe} -> HTTP {e.code}: the objects are not publicly readable. Grant it with\n"
              f"   gcloud storage buckets add-iam-policy-binding gs://{bucket_name} "
              f"--member=allUsers --role=roles/storage.objectViewer")


if __name__ == "__main__":
    main()
