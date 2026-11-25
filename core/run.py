# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Our method's runner: ``python -m genia.core <hydra overrides>``.

Owns everything a run of our method needs around its blocks: the output directory and
config snapshot, PREPROCESSING, the block loop (per-block timing and output flags, cache
save and resume, ``exit_after``) and FINAL.
"""

import contextlib
import os
import sys
import warnings

# torch.cuda imports the deprecated pynvml package — silence its FutureWarning.
warnings.filterwarnings("ignore", message=r"The pynvml package is deprecated.*")

import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from genia.core.paths import RESULTS, SAM3D_OBJECTS_ROOT  # noqa: E402

# Ensure submodules are importable. APPEND, never insert(0): a ``sam3d_objects`` already
# placed at the head of sys.path by the host process must keep winning.
_submodules = str(SAM3D_OBJECTS_ROOT)
if _submodules not in sys.path:
    sys.path.append(_submodules)


def resolve_output_dir(cfg: DictConfig) -> None:
    """Set ``cfg.output.output_dir`` when unset, and snapshot the config into it.

    ``{RESULTS}/{experiment}/{dataset}/{scene}/{timestamp}/`` (``paths.RESULTS``, default
    ``results/`` in the checkout).  ``output.experiment_suffix`` is appended to
    ``+experiment=``, or stands in for it when there is none; with neither, the
    ``{experiment}`` segment is omitted.
    """
    if cfg.output.output_dir is None:
        from datetime import datetime

        from hydra.core.hydra_config import HydraConfig

        experiment = HydraConfig.get().runtime.choices.get("experiment")
        suffix = cfg.output.experiment_suffix
        if experiment and suffix:
            experiment = f"{experiment}_{suffix}"
        elif suffix:
            experiment = suffix
        parts = []
        if experiment:
            parts.append(experiment)
        parts += [cfg.dataset.name, cfg.dataset.scene_name,
                  datetime.now().strftime("%Y%m%d_%H%M%S")]
        cfg.output.output_dir = os.path.join(str(RESULTS), *parts)

    os.makedirs(cfg.output.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output.output_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))


def _save_timing_outputs(cfg, timer):
    """Save timing data when exiting early via exit_after."""
    from genia.core.utils.timing import plot_pipeline_timing

    final_output_dir = os.path.join(cfg.output.output_dir, "final")
    os.makedirs(final_output_dir, exist_ok=True)
    plot_pipeline_timing(timer, final_output_dir, cfg.dataset.scene_name)


def run(cfg: DictConfig) -> None:
    """One run of our method, end to end: output dir, PREPROCESSING, the blocks, FINAL.

    Separate from :func:`main` so a host that composes its own config can call it
    directly (e.g. a warm worker running it repeatedly in one process).
    """
    resolve_output_dir(cfg)
    _run_blocks(cfg)


def _run_blocks(cfg: DictConfig) -> None:
    """PREPROCESSING, the block loop, FINAL."""
    from genia.core.final import run_final

    out = run_blocks(cfg)
    if out is None:
        return
    state, sequence, pipeline, evaluator, device, timer = out

    # FINAL: Save poses, PLY, interpolated videos, tracks (always runs)
    run_final(cfg, state, sequence, pipeline.pipeline_obj, evaluator, device, timer=timer)

    print(f"\n{'='*60}")
    print("Pipeline complete!")
    print(f"{'='*60}")


def run_blocks(cfg: DictConfig, *, emit_preprocessing: bool = True):
    """PREPROCESSING and the block loop: everything but FINAL.

    Returns ``(state, sequence, pipeline, evaluator, device, timer)``, or None when a
    block's ``exit_after`` ended the run. A host with its own FINAL calls this directly,
    with ``emit_preprocessing=False`` when it writes ``preprocessing/`` itself.
    """
    import torch

    from genia.core.evaluator import Evaluator
    from genia.core.manifest import BLOCKS
    from genia.core.pipeline import Pipeline
    from genia.core.preprocessing import resolve_run_assets, run_preprocessing
    from genia.core.state import SAM3DState
    from genia.core.utils.config import (
        block_config, block_output_context, block_output_subdir, get_block_output_flag,
        print_config_table)
    from genia.core.utils.pipeline_state import (
        block_cache_keys, find_latest_pipeline_cache, get_pipeline_cache_dir,
        load_pipeline_cache, save_pipeline_cache)
    from genia.core.utils.timing import get_timer, memory_snapshot

    num_frames, asset_indices, frame_indices, view_indices = resolve_run_assets(cfg)

    print_config_table(
        OmegaConf.to_container(cfg, resolve=False),
        extra={
            "total assets": num_frames,
            "assets to load": f"{len(asset_indices)} — {asset_indices}",
            "frame_indices": frame_indices,
            "view_indices": view_indices,
        },
    )

    # The SAM3D backbone loads here, before PREPROCESSING.
    pipeline = Pipeline(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = Evaluator(device)

    # PREPROCESSING (always-runs-first stage, like FINAL): build + process the
    # Sequence for every downstream consumer, and emit processing diagnostics to
    # the unprefixed preprocessing/ folder.
    sequence = run_preprocessing(cfg, frame_indices, view_indices, asset_indices,
                                 emit=emit_preprocessing)
    pipeline.bind(sequence, evaluator, device)

    # The loop walks every block; a disabled block's runner self-skips at entry.
    active_blocks = [block for block, _section in BLOCKS]
    cache_dir = get_pipeline_cache_dir(cfg.output.output_dir)
    state = None
    resume_idx = 0
    keys = block_cache_keys(cfg, BLOCKS) if cfg.processing.block_cache_keyed else {}

    if cfg.processing.resume_from_cache:
        latest = find_latest_pipeline_cache(cache_dir, active_blocks, keys)
        if latest:
            cached_block, ckpt_path = latest
            resume_idx = active_blocks.index(cached_block) + 1
            print(f"\nPipeline cache: {ckpt_path}")
            print(f"  Resuming after '{cached_block}' — skipping blocks 0..{resume_idx - 1}")
            for j in range(resume_idx):
                bname = active_blocks[j]
                enabled = getattr(block_config(cfg, bname), "enabled", True)
                print(f"  [{bname}] skipped ({'cached' if enabled else 'disabled'})")
            state = load_pipeline_cache(ckpt_path, cls=SAM3DState)
            pipeline.resume(state)

    if state is None:
        state = pipeline.new_state()

    timer = get_timer()
    timer.reset()

    # Memory deep-dive (debug): GENIA_MEM_SNAPSHOT=<block_name> records the full
    # CUDA allocation history for that block and dumps
    # final/mem_snapshot_{block}.pickle (view at pytorch.org/memory_viz).
    mem_snapshot_block = os.environ.get("GENIA_MEM_SNAPSHOT")

    # Live object poses for a viewer following the run (GENIA_LIVE_POSES=1; see live_poses).
    from genia.core.utils import live_poses, progress
    live_poses.start(cfg, pipeline.sequence, state)

    for i, block_name in enumerate(active_blocks):
        if i < resume_idx:
            timer.mark_cached(block_name)
        else:
            timer.start_block(block_name)
            snap_ctx = (
                memory_snapshot(os.path.join(
                    cfg.output.output_dir, "final", f"mem_snapshot_{block_name}.pickle"))
                if block_name == mem_snapshot_block else contextlib.nullcontext()
            )
            progress.set_object(None)   # a block's loops are no object's until it says so
            with block_output_context(cfg, block_name), snap_ctx:
                pipeline.run_block(block_name, state)
            timer.end_block(block_name)
            live_poses.publish_state(state)   # a viewer's, so outside the block's timing
            # A block that wrote nothing (e.g. under suppress_intermediate_*) leaves no
            # folder behind; final/timing.json records which blocks ran.
            _remove_empty_dirs(os.path.join(
                cfg.output.output_dir, block_output_subdir(cfg, block_name)))
            state.check_consistency()
            was_loaded = block_name in getattr(state, "_loaded_block_caches", set())
            if get_block_output_flag(cfg, block_name, "save_cache") and not was_loaded:
                save_pipeline_cache(state, block_name, cache_dir, keys.get(block_name))

        # Check exit_after flag on the block's config (even for cached blocks)
        if getattr(block_config(cfg, block_name), "exit_after", False):
            print(f"\n[exit_after] Exiting after block '{block_name}'.")
            _save_timing_outputs(cfg, timer)
            return None

    return state, sequence, pipeline, evaluator, device, timer


def _remove_empty_dirs(path: str) -> None:
    """Delete ``path`` and every folder under it that holds no file."""
    for root, _dirs, _files in os.walk(path, topdown=False):
        if not os.listdir(root):
            os.rmdir(root)


def check_backbone() -> None:
    """Stop, saying why, when the ``sam3d_objects`` a run would import is not this checkout's
    patched copy (``submodules/sam-3d-objects`` with ``sam-3d-objects.patch``, applied by
    ``install.sh``). Another copy earlier on ``sys.path``, e.g. SAM 3D Objects installed into
    the environment by its own instructions, wins over ours (appended, see above), and an
    unpatched one fails on imports the patch fixes (``utils3d``'s renamed functions: GitHub
    issue #6). For the CLI only: a host process putting its own fork first does so on purpose."""
    import importlib.util
    import subprocess
    from pathlib import Path

    spec = importlib.util.find_spec("sam3d_objects")
    found = Path(spec.origin).resolve().parents[1] if spec and spec.origin else None
    if found != SAM3D_OBJECTS_ROOT.resolve():
        sys.exit(f"sam3d_objects would be imported from {found}, not from this checkout's "
                 f"patched copy {SAM3D_OBJECTS_ROOT}. Uninstall it from this environment "
                 "(pip uninstall sam3d_objects) or take it off PYTHONPATH, then run again.")
    patch = SAM3D_OBJECTS_ROOT.parent / "sam-3d-objects.patch"
    try:
        applied = subprocess.run(["git", "-C", str(SAM3D_OBJECTS_ROOT), "apply", "--reverse",
                                  "--check", str(patch)], capture_output=True).returncode == 0
    except FileNotFoundError:   # no git: nothing to check with
        return
    if not applied:
        sys.exit(f"{SAM3D_OBJECTS_ROOT} is not patched (or carries other changes). Restore it "
                 "and apply the patch: git -C submodules/sam-3d-objects checkout -- . && "
                 "git -C submodules/sam-3d-objects apply ../sam-3d-objects.patch "
                 "(as install.sh does).")


@hydra.main(version_base=None,
            config_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs"),
            config_name="genia")
def main(cfg: DictConfig) -> None:
    """CLI entry point: Hydra composes the config, :func:`run` does the work."""
    check_backbone()
    run(cfg)


if __name__ == "__main__":
    main()
