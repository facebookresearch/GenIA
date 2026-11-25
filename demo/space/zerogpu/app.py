# Copyright (c) Meta Platforms, Inc. and affiliates.

"""The GenIA demo on a ZeroGPU Space: fetch the model weights, then serve ``demo/app.py``
in-process (``--in-process``: on ZeroGPU a GPU exists only inside ``spaces.GPU`` calls, in
the app's own process).

The Space's tree (``demo/space/build_zerogpu.py``): this file at its root, GenIA's checkout
in ``genia/``, its compiled extensions as wheels in ``wheels/``. GenIA and the submodules it
installs with ``--no-deps`` (map-anything, SAM3) are put on the import path here instead,
so pip never resolves their dependencies. The gated weights need the Space secret
``HF_TOKEN``. The examples' recordings are fetched when first played.
"""
import spaces  # noqa: F401 - first: ZeroGPU sets up its CUDA emulation before torch loads

import os
import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GENIA = ROOT / "genia"
PATHS = [ROOT, GENIA / "demo", GENIA / "submodules" / "map-anything",
         GENIA / "submodules" / "sam3"]
sys.path[:0] = [str(p) for p in PATHS]
os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [*map(str, PATHS),
                                                         os.environ.get("PYTHONPATH")]))

WORK = Path("/data/demo_app") if os.access("/data", os.W_OK) else Path.home() / "demo_app"
WORK.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, "-m", "genia.core.download_weights", "--no-actionmesh"],
               check=True)

sys.argv = [str(GENIA / "demo" / "app.py"), "--in-process", "--work-dir", str(WORK)]
runpy.run_path(sys.argv[0], run_name="__main__")
