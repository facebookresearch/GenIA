# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Pose and shape initialization strategies.

This module handles pose initialization (POSE_INIT block).

- **parallel**: all frames denoised in one batched ODE solve, with multi-view
  shape-velocity averaging.  The only active strategy.

Data flow
---------
All functions operate on the ``tokens_by_object`` dict:
    {obj_idx: [(frame_idx, decoder_input), ...]}
where ``decoder_input`` is a dict with keys: ``rotation``, ``translation``,
``scale``, ``decoder_input_slat``.

See :func:`run_parallel_shape_and_poses_init` for the strategy's return values.
"""

import os

import torch

from genia.core.utils.timing import get_timer


# ─── Shared: extract canonical shape latent ───────────────────────────────────


def extract_canonical_shape_latent(obj_idx, canon_frame_per_object, sequence,
                                   inference, seed):
    """Extract the canonical shape latent by running stage 1 on the canon frame.

    Runs the Sparse Structure (SS) generator on the canonical frame's image,
    mask, and pointmap to obtain the 16^3 shape latent. This latent is the
    clean end point of the ``gt_shape_trajectory`` and the fallback canonical
    shape when the batched solve returns none.

    Args:
        obj_idx: Object index.
        canon_frame_per_object: {obj_idx: frame_idx} mapping.
        sequence: Sequence object for frame data access.
        inference: SAM3D Inference callable.
        seed: Random seed for reproducibility.

    Returns:
        Shape latent tensor (1, 4096, 8) for a 16^3 voxel grid.

    Raises:
        ValueError: If no canonical frame is set.
    """
    from genia.core.utils.depth import transform_to_pytorch3d_convention

    canon_fid = canon_frame_per_object.get(obj_idx)
    if canon_fid is None:
        raise ValueError(
            f"Canonical shape extraction requires a canonical frame, "
            f"but object {obj_idx} has none"
        )

    # Run stage-1 only on the canonical frame to get the shape latent
    canon_frame = sequence[canon_fid]
    canon_pm = transform_to_pytorch3d_convention(canon_frame.pointmap)
    canon_pm_tensor = torch.from_numpy(canon_pm).float().cuda()
    canon_result = inference(
        canon_frame.image, canon_frame.masks[obj_idx],
        seed=seed, pointmap=canon_pm_tensor,
        stage1_only=True,
    )
    canonical_shape_latent = canon_result["shape"]
    print(f"    Extracted canonical shape latent from frame {canon_fid}: "
          f"{canonical_shape_latent.shape}")
    return canonical_shape_latent


# ─── Strategy: parallel ──────────────────────────────────────────────────────


def run_parallel_shape_and_poses_init(obj_idx, tokens_by_object, canon_frame_per_object,
                                      sequence, inference, seed,
                                      output_dir, scene_name,
                                      canonical_shape_latent=None,
                                      shape_velocity_averaging="none",
                                      rotation_velocity_averaging="none",
                                      inference_steps=None,
                                      gt_shape_trajectory=False,
                                      entropy_alpha=60.0,
                                      entropy_layer=9,
                                      entropy_min_weight=0.001,
                                      save_viz=True,
                                      cfg_interval_pose=None,
                                      pose_velocity_broadcast_per_frame=False):
    """Run parallel pose init: all frames denoised in one batched ODE solve.

    Cross-frame shape consensus is via ``shape_velocity_averaging`` only.

    Args:
        obj_idx: Object index.
        tokens_by_object: Full tokens dict (only obj_idx's list is read).
        canon_frame_per_object: {obj_idx: frame_idx} mapping.
        sequence: Sequence object for frame data access.
        inference: SAM3D Inference callable (must have ``stage1_batched``).
        seed: Random seed.
        output_dir: Directory for debug visualization PNGs.
        scene_name: Scene name for output filenames.
        canonical_shape_latent: Shape latent tensor ``(1 or N, 4096, 8)``.
        save_viz: The block's resolved ``save_renders``; gates every diagnostic
            this function writes.

    Returns:
        (updated_tokens_list, shape_and_poses_init_results, new_canonical_shape,
         entropy_data) where shape_and_poses_init_results is {frame_idx:
        {"rotation": ..., "translation": ..., "scale": ...}} and
        new_canonical_shape is the anchor's denoised shape latent (1, 4096, 8).
    """
    from genia.core.utils.depth import transform_to_pytorch3d_convention

    sorted_frames = sorted(tokens_by_object[obj_idx], key=lambda t: t[0])
    frames_dict = {fid: di for fid, di in sorted_frames}

    # Determine anchor frame (canonical frame)
    anchor_fid = canon_frame_per_object.get(obj_idx)
    if anchor_fid is None or anchor_fid not in frames_dict:
        anchor_fid = sorted_frames[0][0]
        print(f"    No canonical frame set, using first frame {anchor_fid} as anchor")
    else:
        print(f"    Anchor frame: {anchor_fid} (canonical)")

    # Collect per-frame data in natural sorted order
    frames_data = []
    valid_fids = []
    for fid, _ in sorted_frames:
        frame = sequence[fid]
        mask = frame.masks[obj_idx]
        if not mask.any():
            print(f"    Frame {fid}: no mask, skipping")
            continue
        pm = transform_to_pytorch3d_convention(frame.pointmap)
        frames_data.append({
            "image": frame.image,
            "mask": mask,
            "pointmap": torch.from_numpy(pm).float().cuda(),
        })
        valid_fids.append(fid)

    if not frames_data:
        # A single-frame batch whose one frame is maskless lands here.
        print("    No valid frames, returning unchanged tokens")
        return list(sorted_frames), {}, canonical_shape_latent, None

    # Validate anchor frame has a mask
    if anchor_fid not in valid_fids:
        raise ValueError(
            f"Anchor frame {anchor_fid} has no valid mask; cannot proceed"
        )

    print(f"    Parallel pose init: {len(frames_data)} frames"
          + (", gt_shape_trajectory=ON" if gt_shape_trajectory else ""))

    per_frame_results, batch_ode_histories, entropy_data, per_frame_scale, per_frame_shift = inference.stage1_batched(
        frames_data,
        canonical_shape=canonical_shape_latent,
        seed=seed,
        stage1_inference_steps=inference_steps,
        shape_velocity_averaging=shape_velocity_averaging,
        rotation_velocity_averaging=rotation_velocity_averaging,
        frame_indices=valid_fids,
        entropy_alpha=entropy_alpha,
        entropy_layer=entropy_layer,
        entropy_min_weight=entropy_min_weight,
        gt_shape_trajectory=gt_shape_trajectory,
        cfg_interval_pose=cfg_interval_pose,
        pose_velocity_broadcast_per_frame=pose_velocity_broadcast_per_frame,
    )

    # Extract per-frame poses and update tokens (including raw_ss_modalities)
    shape_and_poses_init_results = {}
    results_dict = {}
    new_canonical_shape = None
    for i, (fid, result) in enumerate(zip(valid_fids, per_frame_results)):
        decoder_input = dict(frames_dict[fid])

        # SAM3D's pose_decoder squeezes translation/rotation to 1D, but the
        # rest of the codebase expects 2D ``(1, 3)`` / ``(1, 4)``.  Re-add the
        # leading batch dim here to keep the storage convention uniform.
        _rot = result["rotation"]
        _trans = result["translation"]
        _scale = result["scale"]
        if _rot.dim() == 1:
            _rot = _rot.unsqueeze(0)
        if _trans.dim() == 1:
            _trans = _trans.unsqueeze(0)
        if _scale.dim() == 1:
            _scale = _scale.unsqueeze(0)
        decoder_input["rotation"] = _rot
        decoder_input["translation"] = _trans
        decoder_input["scale"] = _scale

        # Store new raw_ss_modalities (shape + pose tokens) for downstream use
        new_raw = result.get("raw_ss_modalities")
        if new_raw is not None:
            # Merge with existing raw_ss_modalities (preserves keys we didn't denoise)
            merged_raw = dict(decoder_input.get("raw_ss_modalities", {}))
            merged_raw.update(new_raw)
            decoder_input["raw_ss_modalities"] = merged_raw

        # Propagate SSI params into decoder_input so downstream blocks can
        # decode raw pose tokens without needing perframe_raw_modalities.
        if per_frame_scale is not None and per_frame_scale[i] is not None:
            decoder_input["pointmap_scale"] = per_frame_scale[i]
            decoder_input["pointmap_shift"] = per_frame_shift[i]

        results_dict[fid] = decoder_input
        shape_and_poses_init_results[fid] = {
            "rotation": result["rotation"],
            "translation": result["translation"],
            "scale": result["scale"],
        }

        # Anchor provides the new canonical shape
        if fid == anchor_fid and new_raw is not None and "shape" in new_raw:
            new_canonical_shape = new_raw["shape"]

    # Per-frame shape coords: each frame's raw shape token legitimately differs
    # whenever the input was per-frame, so decode each frame's own coords.
    n_decoded = 0
    for fid, decoder_input in results_dict.items():
        shape_tok = decoder_input.get("raw_ss_modalities", {}).get("shape")
        if shape_tok is None:
            continue
        set_perframe_shape_coords(decoder_input, inference, shape_tok)
        n_decoded += 1
    if n_decoded:
        print(f"    Per-frame shape coords decoded for {n_decoded} frames")

    # Debug visualization: pose trajectory + ODE trajectories.  Optional
    # diagnostics, excluded from the block's core timing.
    if save_viz:
        from genia.core.visualization import (
            plot_multi_frame_ode_trajectories,
            plot_shape_and_poses_init_trajectory,
            plot_shape_ode_trajectories,
        )
        parallel_dir = os.path.join(output_dir, "parallel")
        os.makedirs(parallel_dir, exist_ok=True)
        with get_timer().exclude():
            if shape_and_poses_init_results:
                plot_shape_and_poses_init_trajectory(
                    shape_and_poses_init_results, anchor_fid,
                    os.path.join(parallel_dir,
                                 f"{scene_name}_obj{obj_idx}_pose_trajectory.png"),
                    title=f"{scene_name} obj {obj_idx} — parallel poses",
                )
            if batch_ode_histories:
                # Map batch indices back to frame indices for ODE trajectories
                per_frame_ode_histories = {
                    valid_fids[bi]: hist
                    for bi, hist in batch_ode_histories.items()
                    if bi < len(valid_fids)
                }
                plot_multi_frame_ode_trajectories(
                    per_frame_ode_histories,
                    os.path.join(parallel_dir,
                                 f"{scene_name}_obj{obj_idx}_ode_trajectories.png"),
                    title=f"{scene_name} obj {obj_idx} — parallel ODE trajectories",
                )
                plot_shape_ode_trajectories(
                    per_frame_ode_histories,
                    os.path.join(parallel_dir,
                                 f"{scene_name}_obj{obj_idx}_shape_ode_trajectories.png"),
                    title=f"{scene_name} obj {obj_idx} — parallel shape ODE trajectories",
                )

    # Rebuild tokens list in original frame order
    updated_tokens_list = [
        (fid, results_dict.get(fid, frames_dict[fid]))
        for fid, _ in sorted_frames
    ]

    # Always return canonical shape tokens: fall back to the input latent.
    if new_canonical_shape is None and canonical_shape_latent is not None:
        new_canonical_shape = canonical_shape_latent

    # Which frames the entropy/cross-attention arrays' BATCH axis indexes: a
    # maskless frame is dropped from `valid_fids` but not from the token list.
    if entropy_data is not None:
        entropy_data["frame_keys"] = list(valid_fids)

    return updated_tokens_list, shape_and_poses_init_results, new_canonical_shape, entropy_data


# ─── Shared: per-frame shape coords ────────────────────────────────────────


def set_perframe_shape_coords(decoder_input, inference, shape_tok):
    """Decode a per-frame shape token to its 64^3 occupancy coords and attach as
    ``decoder_input["perframe_shape_coords"]`` — the post-condition shared by the parallel ODE pass (``run_parallel_shape_and_poses_init``) and
    the shape cache's hit path (``core/sam3d_shape_cache.py``), so the
    two can't diverge on how that field is derived."""
    from genia.core.utils.slat_decode import decode_shape_to_coords
    decoder_input["perframe_shape_coords"] = decode_shape_to_coords(
        inference, shape_tok)
