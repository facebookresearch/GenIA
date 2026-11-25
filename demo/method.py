# Copyright (c) Meta Platforms, Inc. and affiliates.

"""The method the demo app runs: everything in ``app.py`` and ``ui/`` that is GenIA's.

The app runs ``ENTRY`` in a warm worker (``ui/worker.py``) with ``run_overrides``, the
recipe ``ROOT/demo/run_demo.py <mode> --data-root DIR --scene NAME [--low-vram |
--no-low-vram] [option args] RUN_ARGS output.output_dir=RUN_DIR`` would run, and shows
``ROOT/demo/data`` as its example scenes. Another method with that same ``demo/`` layout
(genia2) gets the app by copying ``app.py`` and ``ui/`` and writing its own version of
this file.
"""
import os
from pathlib import Path

import run_demo   # beside this file: the recipe
from genia.core.manifest import BLOCKS

#: The method's repository root (``demo/run_demo.py`` and ``demo/data`` live under it).
ROOT = Path(__file__).resolve().parents[1]

NAME = "GenIA"
TITLE = "GenIA: Generative Reconstruction with Test-Time Input Alignment"

#: ``run_demo.py`` settings the app offers, with their header labels. A mode other than
#: ``image`` takes several views.
MODES = {"image": "Single image", "multiview": "Multi-view"}

#: Run options shown as checkboxes: (label, info, arguments added to ``run_demo.py`` when
#: ticked, and when not). Unticked, "Full quality" runs FINETUNE 40 iterations instead of the
#: paper's 100: on the bundled scenes (RTX 4090) 35-64% less time, the same geometry (meshes
#: within a rerun's difference), 0.6-1.5 dB lower PSNR on the input views. ``++``: the key is
#: absent from ``finetuning=none``. The recorded examples (no option) are full quality.
OPTIONS = (
    ("Full quality", "The paper's settings: ~1.5x the time, sharper texture.",
     [], ["++finetuning.num_iterations=40"]),
    ("No test-time refinement", "Skips FINETUNE. Quicker, lower final quality.",
     ["finetuning=none"], []),
    ("ICP pose refinement", "Instead of the per-frame appearance + photometric first refinement.",
     ["--icp"], []),
)


def choices(mode) -> list:
    """Run choices shown as menus in ``mode``: (label, info, {choice: arguments added to
    ``run_demo.py``}), the first choice the default. None here (genia2 offers its backbones)."""
    return []


#: Added to every run. Of the blocks' own outputs (off: ``run_demo.py``'s minimal outputs)
#: only the 3D view's previews are written (voxel grids, object splats); the rest (renders,
#: debug plots, metrics) took ~40% of a multi-view run's time.
RUN_ARGS = ["output.save_previews=true"]

#: The worker's environment: GenIA then reports its loops' progress as ``@@progress`` lines
#: (``genia.core.utils.progress``), which the app shows as bars, and streams each object's
#: current placement to ``live_poses.json`` (``genia.core.utils.live_poses``), which moves the
#: previews in the 3D view. The allocator setting is ``run_demo.py --low-vram``'s: an
#: environment variable, read once per process, so the worker always has it.
ENV = {"GENIA_PROGRESS": "1", "GENIA_LIVE_POSES": "1",
       "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

#: Where the bundled examples' recordings are published (``space/replays_gcs.py push``), fetched
#: when an example is first played (``GENIA_REPLAYS_URL`` overrides it; empty: none).
REPLAYS_URL = os.environ.get("GENIA_REPLAYS_URL",
                             "https://storage.googleapis.com/geniadata/demo/replays").rstrip("/")

#: What the app's worker runs per click, as ``python <ENTRY> <overrides>``.
ENTRY = "-m genia.core"

#: Below this much free GPU memory, the app's Auto low-VRAM mode is on (``run_demo.py``'s rule).
LOW_VRAM_BELOW_GIB = run_demo.LOW_VRAM_BELOW_GIB


def run_overrides(mode, data_root, scene, low_vram: bool, args) -> list[str]:
    """``ENTRY``'s overrides for one run: ``run_demo.py``'s, for the same command line."""
    flag = "--low-vram" if low_vram else "--no-low-vram"
    overrides, _env, _stage = run_demo.plan(
        [mode, "--data-root", str(data_root), "--scene", scene, flag, *args])
    return overrides   # _env is in ENV; _stage is the dynamic mode's, which the app lacks


def worker_overrides(roomy: bool) -> list[str]:
    """Added to every run of the warm worker. On a GPU with room to spare (``roomy``),
    map-anything stays loaded between runs too; otherwise each run reloads it."""
    return ["processing.keep_recon_model=true"] if roomy else []


def warmup(overrides) -> None:
    """Load, in the worker before any click, what a run with ``overrides`` would load
    first: SAM 3D's backbone, and map-anything when it is kept loaded."""
    import genia.core.run  # noqa: F401 — the pipeline's imports, and its sys.path setup
    import torch

    from genia.core.backbone import config_path, sam3d_inference

    torch.zeros(1, device="cuda")   # CUDA's own start-up
    print("Initializing inference pipeline (warm-up)", flush=True)
    sam3d_inference(config_path(), compile=False)
    if "processing.keep_recon_model=true" in overrides:
        from genia.core.utils.model_cache import ModelCache

        ModelCache.get().map_anything_model   # prints "Loading map-anything model:"
    print("Warm: ready to reconstruct.", flush=True)

def run_in_process(overrides) -> int:
    """One run in this process, as ``python <ENTRY> <overrides>`` would do it, for the app's
    in-process runner (``ui/inprocess.py``: ZeroGPU). Returns the exit status: 0, or
    ``-SIGTERM`` when ``stop_in_process`` ended it; a failure raises."""
    import signal

    from hydra import compose, initialize_config_module

    from genia.core.run import run
    from genia.core.utils import progress

    with initialize_config_module("genia.core.configs", version_base=None):
        cfg = compose("genia", overrides=list(overrides))
    progress.clear_stop()
    try:
        run(cfg)
    except progress.Stopped:
        return -signal.SIGTERM
    return 0


def stop_in_process() -> None:
    """End ``run_in_process``'s run at the next step of one of its loops."""
    from genia.core.utils import progress

    progress.request_stop()


#: GPU seconds a run asks ZeroGPU for, by mode: measured on the bundled scenes (RTX 4090,
#: low-VRAM mode) at up to ~210 s (image) and ~350 s (multi-view) with the fast FINETUNE, and
#: ~1.6x that at full quality; with headroom for a larger upload.
GPU_SECONDS = {"image": 300, "multiview": 600}


def gpu_seconds(overrides) -> int:
    """The GPU time to request for a run with ``overrides``."""
    seconds = GPU_SECONDS["image" if "dataset=image" in overrides else "multiview"]
    if "finetuning=none" in overrides:
        return seconds // 2
    return seconds if "++finetuning.num_iterations=40" in overrides else int(seconds * 1.6)


#: ``(block, config section)`` in run order, for the step table: a block is a step when its
#: section is ``enabled`` in the run's ``config.yaml``.
STEPS = BLOCKS

#: Startup lines the run prints as it enters each piece of work, before its first block, and
#: the status shown from then on (the last one present wins).
PHASES = (
    ("Initializing inference pipeline", "Loading SAM 3D weights…"),
    ("Loading map-anything model:", "Loading map-anything + estimating depth…"),
    ("Sequence loaded:", "Preparing the scene…"),
)
