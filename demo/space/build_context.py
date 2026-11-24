#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Assemble the Hugging Face Space's tree: what ``docker build`` (and the Space) builds from.

    python demo/space/build_context.py OUT
    docker build -t genia-space OUT

``OUT`` gets genia's tracked files, its submodules' included (as checked out here, so
with SAM 3D Objects' patch applied: ``install.sh`` then skips both steps,
``GENIA_SKIP_SUBMODULES=1``), and this folder's ``Dockerfile``, ``start.sh`` and
``README.md`` (the Space's card, in place of genia's) at its root. The recorded examples
are not in it: ``start.sh`` fetches them from Google Cloud Storage (``replays_gcs.py``).
Untracked files (results, caches, ``node_modules``) never enter it. ``OUT`` must not exist.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SPACE_FILES = ("Dockerfile", "start.sh", "README.md")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"{args.out} exists")
    listed = subprocess.run(["git", "ls-files", "-z", "--recurse-submodules"], cwd=ROOT,
                            capture_output=True, check=True).stdout.decode().split("\0")
    n = 0
    for rel in filter(None, listed):
        src = ROOT / rel
        if src.is_file():   # a tracked file deleted in the working tree is skipped
            (args.out / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, args.out / rel)
            n += 1
    for name in SPACE_FILES:
        shutil.copy2(HERE / name, args.out / name)
    size = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file())
    print(f"{args.out}: {n} files, {size / 1e6:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
