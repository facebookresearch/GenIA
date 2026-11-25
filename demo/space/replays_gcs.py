#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""The recorded examples on Google Cloud Storage: published from here, fetched by the Space.

    python demo/space/replays_gcs.py push results/demo_app/replays gs://BUCKET/PREFIX
    python demo/space/replays_gcs.py pull https://storage.googleapis.com/BUCKET/PREFIX DIR

``push`` writes ``index.json`` (every file under the recordings with its size) and syncs
the folder to the bucket with ``gcloud storage rsync`` (files the folder no longer holds are
deleted there), so it needs the Cloud SDK and write access. ``pull`` needs neither: it reads
the index over HTTPS from a publicly readable bucket and downloads each file that is
missing or whose size differs, so the Space fetches only what changed. Standard library only.
"""
import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # demo/: its ui package
from ui import replay  # noqa: E402  (standard library only, like this file)

INDEX = replay.INDEX


def push(src: Path, dest: str) -> int:
    files = {p.relative_to(src).as_posix(): p.stat().st_size
             for p in sorted(src.rglob("*")) if p.is_file() and p.name != INDEX}
    (src / INDEX).write_text(json.dumps({"files": files}, indent=0))
    print(f"{len(files)} files, {sum(files.values()) / 1e6:.0f} MB -> {dest}")
    return subprocess.run(["gcloud", "storage", "rsync", "--recursive",
                           "--delete-unmatched-destination-objects", str(src), dest]).returncode


def pull(base: str, dest: Path) -> int:
    with urllib.request.urlopen(f"{base}/{INDEX}", timeout=60) as res:
        files = json.load(res)["files"]
    print(f"{len(files)} recorded files, {sum(files.values()) / 1e6:.0f} MB", flush=True)
    replay.pull(base, dest, files)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push", help="publish a recordings folder to gs://BUCKET/PREFIX")
    p.add_argument("src", type=Path)
    p.add_argument("dest")
    p = sub.add_parser("pull", help="fetch the recordings from their public HTTPS URL")
    p.add_argument("base")
    p.add_argument("dest", type=Path)
    args = parser.parse_args()
    if args.cmd == "push":
        return push(args.src, args.dest.rstrip("/"))
    return pull(args.base.rstrip("/"), args.dest)


if __name__ == "__main__":
    sys.exit(main())
