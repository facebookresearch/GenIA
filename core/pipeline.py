"""Our method's blocks, as the block loop in ``core/run.py`` runs them.

The runner owns the loop -- per-block timing and output flags, cache save and resume,
``exit_after``, FINAL. This class owns what is specific to the blocks: loading the SAM3D
backbone, the state the blocks start from, and what each block runs.
"""

import os

from genia.core.pose_refine import run_global_refine
from genia.core.utils.config import block_output_subdir
from genia.core.utils.timing import get_timer
from genia.core.main import (
    run_appearance_init,
    run_finetuning,
    run_gt_shapes_inversion,
    run_shape_and_poses_init,
    set_default_canon_frames,
)
from genia.core.state import SAM3DState


class Pipeline:
    """Our block pipeline, driven by :func:`genia.core.run.run`.

    Built in two steps because the backbone loads BEFORE PREPROCESSING while the rest
    needs the ``Sequence``: the constructor loads the model, and
    :meth:`bind` takes what PREPROCESSING produced.
    """

    def __init__(self, cfg):
        from loguru import logger

        from genia.core.backbone import config_path, sam3d_inference

        self.cfg = cfg
        config = config_path()
        print(f"\nInitializing inference pipeline from {config}")
        # Built once per process, so a host calling run() more than once (e.g. a warm
        # worker) doesn't re-read the checkpoint. Identical work on the
        # one-run-per-process CLI path.
        self.inference = sam3d_inference(config, compile=False)
        self.pipeline_obj = self.inference._pipeline

        # Suppress verbose per-frame loguru output from sam3d_objects submodule
        logger.disable("sam3d_objects")

    def bind(self, sequence, evaluator, device):
        """Take the processed ``Sequence`` and the run's shared objects."""
        from loguru import logger

        cfg = self.cfg
        # mv_shared_world_pose collapses the VIEWS of a timestamp onto one world placement,
        # so on monocular data (one view per timestamp) it is vacuous.  Turn it off rather
        # than erroring, which is what lets one pipeline preset run on both mono and MV
        # scenes.
        #
        # This is the ROUTINE path, not a misconfiguration: pipeline/default.yaml sets
        # it true for every pipeline, so every mono run lands here.  Hence debug, not
        # warning.
        if cfg.pipeline.mv_shared_world_pose and not sequence.is_mv:
            logger.debug(
                "pipeline.mv_shared_world_pose is vacuous on monocular data "
                "(sequence.is_mv == False); off for this run."
            )
            cfg.pipeline.mv_shared_world_pose = False

        self.sequence = sequence
        self.evaluator = evaluator
        self.device = device
        # The canonical FrameKey list. Bare time-axis labels (e.g. [0,0,...,0] for
        # MV-static after the axis lift) would silently alias every iteration to view 0
        # if used as a sequence key.
        self.frame_indices = list(sequence.frame_keys)

    def new_state(self):
        """A fresh state: one skeleton entry per (object, frame) the blocks will fill."""
        cfg, sequence = self.cfg, self.sequence
        state = SAM3DState(pipeline_obj=self.pipeline_obj)
        # When object_indices is set, only include those objects -- otherwise
        # downstream blocks see skeleton entries with empty token data for
        # unrequested objects.
        requested = (
            set(cfg.dataset.object_indices)
            if cfg.dataset.object_indices is not None
            else None
        )
        for fi in self.frame_indices:
            for obj_idx in sequence[fi].masks.keys():
                if requested is not None and obj_idx not in requested:
                    continue
                if obj_idx not in state.tokens_by_object:
                    state.tokens_by_object[obj_idx] = []
                state.tokens_by_object[obj_idx].append((fi, {}))
        set_default_canon_frames(state)
        return state

    def resume(self, state):
        """Reattach what a cached state cannot carry, and restore our invariants."""
        state.pipeline_obj = self.pipeline_obj
        # MV shared world pose: a cached state may hold per-camera independent poses.
        # Re-apply the invariant so subsequent blocks see consistent derived per-frame
        # poses.  Idempotent on already-consistent state.
        if self.sequence.is_mv and self.cfg.pipeline.mv_shared_world_pose:
            # A cache cut BEFORE pose_init (e.g. `exit_after` on appearance_init)
            # carries entries but no decoded pose; the rebase
            # returns 0 for those rather than raising, so no guard is needed here.
            for obj_idx in sorted(state.tokens_by_object.keys()):
                state.rebase_perframe_from_reference(
                    obj_idx, self.sequence, scope="timestamp")

    def run_block(self, block_name, state):
        """Run one block on ``state``. Each runner self-skips when its block is disabled."""
        cfg, sequence = self.cfg, self.sequence
        inference, pipeline_obj = self.inference, self.pipeline_obj
        evaluator, device = self.evaluator, self.device

        if block_name == "gt_shapes_inversion":
            run_gt_shapes_inversion(cfg, state, sequence, pipeline_obj, device)
        elif block_name == "shape_init":
            run_shape_and_poses_init(
                cfg, state, sequence, inference, pipeline_obj, evaluator, device,
                config_override=cfg.shape_init, block_tag="shape_init")
        elif block_name == "appearance_init":
            run_appearance_init(
                cfg, state, sequence, inference, pipeline_obj, evaluator, device)
        elif block_name == "pose_init":
            run_shape_and_poses_init(
                cfg, state, sequence, inference, pipeline_obj, evaluator, device,
                config_override=cfg.pose_init, block_tag="pose_init")
        elif block_name == "global_pose_refine_1":
            run_global_refine(
                cfg, state, sequence, pipeline_obj, evaluator, device,
                config_override=cfg.global_pose_refine_1, block_label="GLOBAL_POSE_REFINE_1")
            if cfg.output.save_renders:
                with get_timer().exclude():
                    from genia.core.utils.eval_assets_export import export_coarse_shape_wireframe
                    block_dir = os.path.join(
                        cfg.output.output_dir, block_output_subdir(cfg, block_name))
                    try:
                        export_coarse_shape_wireframe(
                            cfg, state, sequence, block_dir, pipeline_obj)
                    except Exception as e:  # noqa: BLE001 — a preview must not fail the block
                        print(f"  coarse-shape wireframe preview failed (non-fatal): {e}")
        elif block_name == "appearance_init_2":
            run_appearance_init(
                cfg, state, sequence, inference, pipeline_obj, evaluator, device,
                config_override=cfg.appearance_init_2, block_label="APPEARANCE_INIT_2")
        elif block_name == "finetune":
            run_finetuning(cfg, state, sequence, pipeline_obj, evaluator, device)
        elif block_name == "global_pose_refine_2":
            run_global_refine(
                cfg, state, sequence, pipeline_obj, evaluator, device,
                config_override=cfg.global_pose_refine_2, block_label="GLOBAL_POSE_REFINE_2")
        else:
            raise KeyError(f"no runner for block {block_name!r}")
