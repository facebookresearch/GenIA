# Copyright (c) Meta Platforms, Inc. and affiliates.

"""PREPROCESSING: build the processed ``Sequence`` every block consumes, and write ``preprocessing/``.

The always-run first stage of every run (``core/run.py``). One builder and one writer,
so every caller builds the same ``Sequence``.
"""

import os


def resolve_run_assets(cfg):
    """Which dataset assets this run consumes, and how they lift onto the
    (frame, view) axes.

    Returns ``(num_frames, asset_indices, frame_indices, view_indices)`` —
    everything ``run_preprocessing`` needs to rebuild the
    exact same ``Sequence``.  Called by ``core/run.py``.
    """
    from genia.core.utils import setup_paths
    from genia.core.utils.io_utils import MV_STATIC_DATASETS, axis_lift_indices

    _setup_kw = {}
    if getattr(cfg.dataset, "num_input_views", None) is not None:
        _setup_kw["num_input_views"] = cfg.dataset.num_input_views
    paths = setup_paths(cfg.dataset.path, cfg.dataset.scene_name, dataset_type=cfg.dataset.name, **_setup_kw)

    # Determine which assets (file positions) to load
    num_frames = len(paths['image_names'])
    if cfg.dataset.frame_index is not None:
        asset_indices = [cfg.dataset.frame_index]
    else:
        asset_indices = list(range(0, num_frames, cfg.dataset.frame_stride))
        if asset_indices and asset_indices[-1] != num_frames - 1:
            asset_indices.append(num_frames - 1)

    # Subset the pipeline-facing assets; io_utils deliberately exposes more (see
    # MV_STATIC_DATASETS for why, per dataset).
    if cfg.dataset.name in MV_STATIC_DATASETS:
        n_input = getattr(cfg.dataset, "num_input_views", None)
        if n_input is not None:
            asset_indices = asset_indices[:n_input]

    # Lift asset positions to (frame, view) coordinates per dataset shape.
    # MV-static (gso, co3d, mvcustom) → FrameKey(0, asset_idx); mono → FrameKey(asset_idx, 0).
    frame_indices, view_indices = axis_lift_indices(cfg.dataset.name, asset_indices)
    return num_frames, asset_indices, frame_indices, view_indices


def emit_preprocessing_outputs(cfg, sequence):
    """Write the ``preprocessing/`` folder for an already-built ``Sequence``.

    Called by :func:`run_preprocessing`.  ``preprocessing/colmap/`` holds the input
    cameras + depth cloud, available before FINAL lands.

    Both halves are visualization/export, so the timer is excluded HERE rather than
    at each call site: a caller that runs it inside a timed block measures the
    Sequence prep, not this.
    """
    from genia.core.utils.timing import get_timer

    out_dir = os.path.join(cfg.output.output_dir, "preprocessing")

    with get_timer().exclude():
        # Processing diagnostics -> {output_dir}/preprocessing/ (unprefixed, like
        # final/).  Respect the global intermediate-render suppression; the Sequence
        # prep always runs regardless.
        if not cfg.output.suppress_intermediate_renders:
            from genia.core.datavis import emit_sequence_diagnostics

            print("\n" + "-" * 40)
            print(f"PREPROCESSING: diagnostics -> {out_dir}")
            print("-" * 40)
            emit_sequence_diagnostics(sequence, out_dir)

        # COLMAP model of the inputs (cameras + depth point cloud) — like FINAL's
        # colmap, gated on the same save_colmap flag.
        if cfg.output.save_colmap:
            from genia.core.utils.colmap_export import export_preprocessing_colmap
            export_preprocessing_colmap(cfg, sequence, out_dir)


def run_preprocessing(cfg, frame_indices, view_indices, asset_indices):
    """PREPROCESSING: build + return the processed ``Sequence`` (depth + camera
    poses, per ``core/configs/processing``) for every downstream consumer, and emit
    the processing diagnostics.

    A special always-run stage (mirrors FINAL): NOT a manifest block,
    no per-block flags.  Writes the depth/normals/pointmap panels + summary grid
    into the unprefixed ``preprocessing/`` folder.
    """
    from genia.core.utils import Sequence

    # Declare which backend and with which checkpoint; the cache evicts the other one
    # lazily when this one is actually loaded (see `release_reconstruction_models`).
    from genia.core.utils.model_cache import ModelCache

    ModelCache.get().set_recon_model_id_from(cfg.processing)

    # Build the Sequence — loads all frame data once (images, masks, depth,
    # intrinsics) and runs depth + camera-pose processing in __init__.
    sequence = Sequence(
        cfg.dataset.path, cfg.dataset.scene_name, cfg.dataset.name,
        frame_indices,
        view_indices=view_indices,
        asset_indices=asset_indices,
        reconstruction_model=cfg.processing.reconstruction_model,
        depth_source=cfg.processing.depth_source,
        camera_poses_source=cfg.processing.camera_poses_source,
        condition_recon_model_on_gt_poses=cfg.processing.condition_recon_model_on_gt_poses,
        condition_recon_model_on_gt_depths=cfg.processing.condition_recon_model_on_gt_depths,
        downscale_factor=cfg.dataset.downscale_factor,
        median_intrinsics=cfg.processing.median_intrinsics,
        fps=float(cfg.dataset.fps),
        num_input_views=getattr(cfg.dataset, "num_input_views", None),
        world_origin_is_object=getattr(cfg.dataset, "world_origin_is_object", False),
        recon_on_test_views=getattr(
            cfg.processing, "recon_on_test_views", True
        ),
        cache_reconstruction=getattr(cfg.processing, "cache_reconstruction", False),
        recon_resolution_set=getattr(cfg.processing, "recon_resolution_set", 518),
        recon_mask_edges=getattr(cfg.processing, "recon_mask_edges", True),
        recon_confidence_min=getattr(cfg.processing, "recon_confidence_min", None),
    )

    emit_preprocessing_outputs(cfg, sequence)

    # Depth + cameras are in the Sequence, so the backend is done. Freed HERE, not left
    # to the GC: it is resident memory every later block would be charged for. See
    # ModelCache.release_reconstruction_models.
    ModelCache.get().release_reconstruction_models()
    return sequence
