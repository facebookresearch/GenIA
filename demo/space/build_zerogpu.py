#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Assemble the ZeroGPU Space's tree (a Gradio-SDK Space: pip only, no compiler, no GPU at
build time).

    python demo/space/build_zerogpu.py OUT --wheels DIR

``OUT`` gets ``zerogpu/``'s ``app.py``, ``README.md`` (the Space's card) and ``packages.txt``
at its root; genia's tracked files in ``genia/``, its submodules' included as checked out
here (so with SAM 3D Objects' patch applied) and the 3D view's built bundle
(``submodules/3DView/demo/dist``, which is not tracked); the wheels in ``DIR`` (genia's
CUDA extensions, built for the Space's GPU) in ``wheels/``; and a ``requirements.txt``:
genia's own, the torch ``install.sh`` pins, the wheels, and what ``install.sh`` installs from
PyPI or git. What ``install.sh`` installs with ``--no-deps -e`` is put on the import path by
``app.py`` instead. ``OUT`` must not exist.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

GENIA = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SPACE_FILES = ("app.py", "README.md", "packages.txt")
VIEWER_DIST = Path("submodules/3DView/demo/dist")

#: What install.sh installs besides genia's requirements.txt, for pip in one go.
REQUIREMENTS = """\
--extra-index-url https://download.pytorch.org/whl/cu128
torch==2.8.0+cu128
torchvision==0.23.0+cu128
spaces
xformers==0.0.32.post2
spconv-cu126==2.3.8
cumm-cu126==0.7.11
uniception==0.1.6
moge @ git+https://github.com/microsoft/MoGe.git@07444410f1e33f402353b99d6ccd26bd31e469e8
utils3d @ git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183
"""


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--wheels", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"{args.out} exists")
    wheels = sorted(args.wheels.glob("*.whl"))
    if not wheels:
        parser.error(f"no wheels in {args.wheels}")
    if not (GENIA / VIEWER_DIST / "webview.js").is_file():
        parser.error(f"build the 3D view first ({GENIA / VIEWER_DIST} has no webview.js)")
    listed = subprocess.run(["git", "ls-files", "-z", "--recurse-submodules"], cwd=GENIA,
                            capture_output=True, check=True).stdout.decode().split("\0")
    out = args.out / "genia"
    for rel in filter(None, listed):
        if (GENIA / rel).is_file():   # a tracked file deleted in the working tree is skipped
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(GENIA / rel, out / rel)
    shutil.copytree(GENIA / VIEWER_DIST, out / VIEWER_DIST, dirs_exist_ok=True)
    (args.out / "wheels").mkdir()
    for whl in wheels:
        shutil.copy2(whl, args.out / "wheels" / whl.name)
    for name in SPACE_FILES:
        shutil.copy2(HERE / "zerogpu" / name, args.out / name)
    own = [line for line in (GENIA / "requirements.txt").read_text().splitlines()
           if line.strip() and not line.lstrip().startswith("#")]
    (args.out / "requirements.txt").write_text(
        REQUIREMENTS + "\n".join(own) + "\n"
        + "".join(f"./wheels/{whl.name}\n" for whl in wheels))
    size = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file())
    print(f"{args.out}: {size / 1e6:.0f} MB, {len(wheels)} wheels")
    return 0


if __name__ == "__main__":
    sys.exit(main())
