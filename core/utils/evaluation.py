# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Evaluation utilities for the SAM3D-Objects pipeline.

This module provides functions for evaluating reconstruction quality
including frame processing, metrics computation, and summary generation.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from rich.box import ROUNDED
from rich.table import Table

from .console import CONSOLE as _CONSOLE, fmt_path
# Pure-torch by design (no pytorch3d CUDA-extension import cost), so this is free at module
# scope and keeps it out of the per-object/per-frame overview-framing loop.
from .quaternion_ops import quaternion_to_matrix


def _fmt_frame_key(fk: Any) -> str:
    """Compact, human-readable label for a FrameKey (or bare int)."""
    if hasattr(fk, "frame") and hasattr(fk, "view"):
        return f"f{fk.frame}" if fk.view == 0 else f"f{fk.frame}/v{fk.view}"
    return str(fk)


def _metrics_table(title: str, first_col: str) -> Table:
    """Bordered table with a leading label column + PSNR/SSIM/LPIPS columns."""
    table = Table(title=title, box=ROUNDED, header_style="bold magenta")
    table.add_column(first_col, style="cyan", no_wrap=True)
    table.add_column("PSNR (dB)", justify="right")
    table.add_column("SSIM", justify="right")
    table.add_column("LPIPS", justify="right")
    return table


if TYPE_CHECKING:
    from genia.core.evaluator import Evaluator
    from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian.gaussian_model import (
        Gaussian,
    )


def process_frame_with_canonical_object(
    sequence: Any,
    frame_index,
    canonical_gaussians: Dict[int, Any],
    tokens_by_object: Dict[int, List[Tuple[Any, Dict[str, Any]]]],
    per_frame_canonical: bool = False,
    background: bool = True,
    bg_color: Optional[torch.Tensor] = None,
    canonical_mesh_verts=None,
    per_frame_mesh_verts=None,
    per_frame_mesh_rotations=None,
    canonical_mesh_faces=None,
    warp_knn_k: int = 4,
    warp_knn_eps: float = 1.0e-8,
    warp_knn_chunk_size: int = 8192,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, List[np.ndarray], List[int], "Gaussian"]:
    """
    Process a single frame using pre-decoded canonical Gaussians warped with per-frame pose.

    Parameters
    ----------
    sequence : Sequence
        Cached scene data.
    frame_index : FrameKey or int
        Frame to process. Bare ints are coerced to ``FrameKey(int, 0)``.
    canonical_gaussians : dict
        Dictionary mapping obj_idx -> decoded Gaussian object (if per_frame_canonical=False)
        OR dict[obj_idx][FrameKey] -> decoded Gaussian object (if per_frame_canonical=True).
    tokens_by_object : dict
        Dictionary mapping obj_idx -> list of (FrameKey, decoder_input) with poses.
    per_frame_canonical : bool, optional
        If True, use per-frame canonical Gaussians (standard mode).
        If False, use shared canonical Gaussians across frames (averaged-tokens mode).
    background : bool, optional
        Whether to add background Gaussians. Default: True.

    Returns
    -------
    tuple
        (rendered_image, gt_image, K_matrix, masks, object_ids, scene_gs) where images are torch tensors
        (H, W, 3), masks is a list of np arrays, object_ids is list of actual object IDs,
        and scene_gs is the Gaussian scene.
    """
    from genia.core.utils.slat_decode import make_scene

    from genia.core.utils.gaussian import attach_sh_rest
    from .frame_key import as_frame_key
    from .gaussian import create_background_gaussians, join_gaussians, transform_scene_to_r3_convention, transform_scene_to_world
    from .rendering import render_gaussians_to_image
    from genia.core.utils.deformation import _lookup_per_frame_deformation, warp_gaussians_high_res

    # Coerce bare-int frame index to FrameKey(int, 0) for back-compat.
    frame_index = as_frame_key(frame_index)

    # Get cached frame data
    frame = sequence[frame_index]
    image = frame.image
    H, W = sequence.H, sequence.W
    masks_dict = frame.masks
    K_matrix = frame.K_matrix
    pointmap_original = frame.pointmap.copy() if background else None

    # Get per-frame poses from cached tokens
    # Build outputs list with canonical gaussian + per-frame pose
    outputs = []
    # Track which objects we actually render (for mask correspondence)
    rendered_object_ids = []
    # Save canonical Gaussian state before applying per-frame appearance
    # (make_scene deep-copies, so originals can be restored after)
    saved_gs_states: Dict[int, tuple] = {}
    # Per-vertex ActionMesh deformation warp: save/restore raw means+quats
    # so the per-frame warp doesn't leak across frames (canonical_gs is
    # shared). Parallel to saved_gs_states (DC/SH).
    saved_gs_geom: Dict[int, tuple] = {}

    for obj_idx in sorted(tokens_by_object.keys()):
        # Get the canonical Gaussian for this object (and frame, if per_frame_canonical)
        if per_frame_canonical:
            if obj_idx not in canonical_gaussians or frame_index not in canonical_gaussians[obj_idx]:
                print(
                    f"    Warning: No canonical Gaussian for object {obj_idx} frame {frame_index}, skipping"
                )
                continue
            canonical_gs = canonical_gaussians[obj_idx][frame_index]
        else:
            if obj_idx not in canonical_gaussians:
                print(f"    Warning: No canonical Gaussian for object {obj_idx}, skipping")
                continue
            canonical_gs = canonical_gaussians[obj_idx]

        # Find the pose for this frame
        frame_pose = None
        for fid, decoder_input in tokens_by_object[obj_idx]:
            if fid == frame_index:
                frame_pose = decoder_input
                break

        if frame_pose is None:
            print(f"    Warning: No pose found for object {obj_idx} at frame {frame_index}, skipping")
            continue

        # Per-vertex deformation warp (actionmesh). When the GT deformation
        # field is loaded AND keyed for this (obj, frame), warp the canonical
        # Gaussian's means/quats by the per-frame field BEFORE the Stage-1
        # pose (applied downstream in make_scene). Falls through to the
        # static-canonical path when any piece is missing.
        resolved = _lookup_per_frame_deformation(
            canonical_mesh_verts,
            per_frame_mesh_verts,
            per_frame_mesh_rotations,
            obj_idx,
            int(getattr(frame_index, "frame", frame_index)),
            canonical_gs.get_xyz.device,
            canonical_mesh_faces,
        )
        if resolved is not None:
            *_warp_core, _faces = resolved
            with torch.no_grad():
                means_w, quats_w = warp_gaussians_high_res(
                    canonical_gs, *_warp_core,
                    K=warp_knn_k,
                    eps=warp_knn_eps,
                    chunk_size=warp_knn_chunk_size,
                    faces=_faces,
                )
            saved_gs_geom[obj_idx] = (canonical_gs._xyz, canonical_gs._rotation)
            canonical_gs.from_xyz(means_w)
            canonical_gs.from_rotation(quats_w)

        # Apply per-frame appearance (DC offset + SH rest) to canonical Gaussian.
        # State is saved and restored after make_scene (which deep-copies).
        has_dc = "dc_offset" in frame_pose
        has_sh = "sh_rest" in frame_pose
        if has_dc or has_sh:
            saved_gs_states[obj_idx] = (
                canonical_gs._features_dc,
                canonical_gs._features_rest,
                canonical_gs.sh_degree,
                canonical_gs.active_sh_degree,
            )
            if has_dc:
                canonical_gs._features_dc = (
                    canonical_gs._features_dc + frame_pose["dc_offset"].to(canonical_gs._features_dc.device)
                )
            if has_sh:
                attach_sh_rest(canonical_gs, frame_pose["sh_rest"].to(canonical_gs._features_dc.device))

        # Build output with canonical gaussian and per-frame pose
        output = {
            "gaussian": [canonical_gs],  # Wrap in list for make_scene
            "rotation": frame_pose["rotation"],
            "translation": frame_pose["translation"],
            "scale": frame_pose["scale"],
        }
        outputs.append(output)
        rendered_object_ids.append(obj_idx)

    # Build masks list for the objects we're actually rendering
    # MaskDict returns zeros for objects not present in the frame
    object_ids = rendered_object_ids
    masks = [masks_dict[oid] for oid in object_ids]

    if not outputs:
        # Restore any saved states before returning
        for obj_idx, (s_dc, s_rest, s_deg, s_adeg) in saved_gs_states.items():
            gs = canonical_gaussians[obj_idx] if not per_frame_canonical else canonical_gaussians[obj_idx][frame_index]
            gs._features_dc = s_dc
            gs._features_rest = s_rest
            gs.sh_degree = s_deg
            gs.active_sh_degree = s_adeg
        for obj_idx, (s_xyz, s_rot) in saved_gs_geom.items():
            gs = canonical_gaussians[obj_idx] if not per_frame_canonical else canonical_gaussians[obj_idx][frame_index]
            gs._xyz = s_xyz
            gs._rotation = s_rot
        print(f"    Warning: No objects to render for frame {frame_index}")
        return None

    # Create combined scene from all outputs (in PyTorch3D convention)
    scene_gs = make_scene(*outputs)

    # Restore canonical Gaussian state (shared across frames)
    for obj_idx, (s_dc, s_rest, s_deg, s_adeg) in saved_gs_states.items():
        gs = canonical_gaussians[obj_idx] if not per_frame_canonical else canonical_gaussians[obj_idx][frame_index]
        gs._features_dc = s_dc
        gs._features_rest = s_rest
        gs.sh_degree = s_deg
        gs.active_sh_degree = s_adeg
    for obj_idx, (s_xyz, s_rot) in saved_gs_geom.items():
        gs = canonical_gaussians[obj_idx] if not per_frame_canonical else canonical_gaussians[obj_idx][frame_index]
        gs._xyz = s_xyz
        gs._rotation = s_rot

    # Transform scene from PyTorch3D to R3 convention (camera space)
    new_scene_gs = transform_scene_to_r3_convention(scene_gs)

    # Store scene without background for point cloud export
    scene_gs_no_bg = new_scene_gs

    # Transform fg to world space for consistent rendering with bg
    frame_c2w = frame.c2w
    new_scene_gs = transform_scene_to_world(new_scene_gs, frame_c2w)

    # Add background Gaussians if requested (already in world space)
    if background and pointmap_original is not None:
        background_gs = create_background_gaussians(image, pointmap_original, masks, K_matrix, c2w=frame_c2w)
        if background_gs is not None:
            new_scene_gs = join_gaussians(background_gs, new_scene_gs)

    # Render Gaussians to image (world space → camera via c2w)
    rendered = render_gaussians_to_image(new_scene_gs, K_matrix, W, H, bg_color=bg_color, c2w=frame_c2w)

    # Convert ground truth to tensor.  Foreground-only unless the render
    # itself carries the dataset background — see ``_gt_tensor_foreground``.
    gt_image = _gt_tensor_foreground(
        image, masks, bg_color, has_background=background,
    )

    # Clamp rendered to [0, 1]
    rendered = torch.clamp(rendered.cpu(), 0.0, 1.0)

    return rendered, gt_image, K_matrix, masks, object_ids, scene_gs_no_bg


def _build_evaluation_summary(
    evaluator: "Evaluator",
    rendered_frames: List[torch.Tensor],
    gt_frames: List[torch.Tensor],
    frame_indices_processed: List[int],
    per_object_data: Dict[int, Dict[str, List[torch.Tensor]]],
    args: Any,
    suffix: str,
) -> Dict[str, Any]:
    """
    Build evaluation summary from collected frames (shared by both evaluation modes).

    Parameters
    ----------
    evaluator : Evaluator
        Evaluator instance.
    rendered_frames : list
        List of rendered frame tensors (B, C, H, W).
    gt_frames : list
        List of ground truth frame tensors (B, C, H, W).
    frame_indices_processed : list
        List of frame indices that were processed.
    per_object_data : dict
        Dictionary mapping obj_idx -> {'rendered': [], 'gt': []}.
    args : DictConfig
        Hydra configuration (GeneralConfig).
    suffix : str
        Suffix for output filenames.

    Returns
    -------
    dict
        Evaluation summary with metrics.
    """
    _CONSOLE.rule("[bold cyan]Evaluating sequence")

    # Use Evaluator's evaluate_sequence method for full-frame metrics
    seq_metrics = evaluator.evaluate_sequence(gt_frames, rendered_frames)

    def _frame_index_for_json(fk):
        """Serialize a FrameKey to JSON: int (frame value) for mono (view==0)
        to preserve byte-identical output; ``[frame, view]`` for multi-view.
        Bare ints pass through unchanged."""
        if hasattr(fk, "frame") and hasattr(fk, "view"):
            return fk.frame if fk.view == 0 else [fk.frame, fk.view]
        return fk

    # Per-frame breakdown (full frame). Skipped for a single frame, where it
    # would just duplicate the "Full frame" row of the Results summary.
    if len(frame_indices_processed) > 1:
        pf_table = _metrics_table("[bold]Per-frame metrics (full frame)[/bold]", "Frame")
        for i, frame_index in enumerate(frame_indices_processed):
            pf_table.add_row(
                _fmt_frame_key(frame_index),
                f"{seq_metrics['psnr_values'][i]:.2f}",
                f"{seq_metrics['ssim_values'][i]:.4f}",
                f"{seq_metrics['lpip_values'][i]:.4f}",
            )
        _CONSOLE.print(pf_table)

    # Build frame_metrics with nested structure: full_frame, obj_0, obj_1, ...
    frame_metrics = {
        "full_frame": [
            {
                "frame_index": _frame_index_for_json(frame_indices_processed[i]),
                "psnr": seq_metrics["psnr_values"][i],
                "ssim": seq_metrics["ssim_values"][i],
                "lpip": seq_metrics["lpip_values"][i],
            }
            for i in range(len(frame_indices_processed))
        ]
    }

    # Evaluate per-object metrics
    per_object_summary = {}
    for obj_idx in sorted(per_object_data.keys()):
        obj_key = f"obj_{obj_idx}"
        if per_object_data[obj_idx]["rendered"] and per_object_data[obj_idx]["gt"]:
            obj_metrics = evaluator.evaluate_sequence(
                per_object_data[obj_idx]["gt"], per_object_data[obj_idx]["rendered"]
            )
            # Use per-object frame indices if available, otherwise fall back to all frames
            obj_frame_indices = per_object_data[obj_idx].get("frame_indices", frame_indices_processed)
            num_obj_frames = len(obj_metrics["psnr_values"])
            frame_metrics[obj_key] = [
                {
                    "frame_index": _frame_index_for_json(
                        obj_frame_indices[i] if i < len(obj_frame_indices)
                        else frame_indices_processed[i]
                    ),
                    "psnr": obj_metrics["psnr_values"][i],
                    "ssim": obj_metrics["ssim_values"][i],
                    "lpip": obj_metrics["lpip_values"][i],
                }
                for i in range(num_obj_frames)
            ]
            per_object_summary[obj_key] = {
                "psnr_mean": float(obj_metrics["psnr_mean"]),
                "psnr_std": float(obj_metrics["psnr_std"]),
                "ssim_mean": float(obj_metrics["ssim_mean"]),
                "ssim_std": float(obj_metrics["ssim_std"]),
                "lpip_mean": float(obj_metrics["lpip_mean"]),
                "lpip_std": float(obj_metrics["lpip_std"]),
            }

    summary = {
        "num_frames": len(frame_indices_processed),
        # Per-object summary (masked region metrics)
        "per_object": per_object_summary,
        # Per-frame breakdown (full_frame, obj_1, obj_2, ...)
        "per_frame": frame_metrics,
    }

    # Save metrics to JSON if requested
    save_metrics = args.output.save_metrics
    output_dir = args.output.output_dir
    scene_name = args.dataset.scene_name
    if save_metrics:
        metrics_path = os.path.join(output_dir, f"{scene_name}{suffix}_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(summary, f, indent=2)
        _CONSOLE.print(f"[dim]Saved metrics → {fmt_path(metrics_path)}[/dim]")

    return summary


def evaluate_with_canonical_objects(
    args: Any,
    sequence: Any,
    frame_indices: List[int],
    canonical_gaussians: Dict[int, Any],
    tokens_by_object: Dict[int, List[Tuple[int, Dict[str, Any]]]],
    evaluator: "Evaluator",
    device: torch.device,
    suffix: str = "",
    save_renders: bool = True,
    per_frame_canonical: bool = False,
    before_renders: Optional[Dict[int, torch.Tensor]] = None,
    before_poses: Optional[Dict[int, Dict[int, Dict[str, Any]]]] = None,
    canonical_mesh_verts=None,
    per_frame_mesh_verts=None,
    per_frame_mesh_rotations=None,
    canonical_mesh_faces=None,
    warp_knn_k: int = 4,
    warp_knn_eps: float = 1.0e-8,
    warp_knn_chunk_size: int = 8192,
) -> Dict[str, Any]:
    """
    Evaluate frames using canonical objects with per-frame poses.

    Parameters
    ----------
    args : DictConfig
        Hydra configuration (GeneralConfig).
    sequence : Sequence
        Cached scene data.
    frame_indices : list
        List of frame indices to process.
    canonical_gaussians : dict
        Dictionary mapping obj_idx -> canonical Gaussian (if per_frame_canonical=False)
        OR dict[obj_idx][frame_idx] -> canonical Gaussian (if per_frame_canonical=True).
    tokens_by_object : dict
        Dictionary mapping obj_idx -> list of (frame_idx, decoder_input).
    evaluator : Evaluator
        Evaluator instance.
    device : torch.device
        Device to use.
    suffix : str, optional
        Suffix for output filenames.
    save_renders : bool, optional
        Whether to save rendered images.
    per_frame_canonical : bool, optional
        If True, use per-frame canonical Gaussians (standard mode).
    before_renders : dict, optional
        Dictionary mapping frame_idx -> rendered tensor (H,W,3) from before
        refinement. When provided, comparison figures show before/after.
    before_poses : dict, optional
        Pose snapshot from before refinement, in compute_pose_axes format:
        ``{obj_idx: {frame_idx: {"rotation", "translation", "scale"}}}``.
        When provided, axes overlays are drawn on comparison figures.

    Returns
    -------
    dict
        Evaluation summary with metrics.
    """
    from .frame_key import as_frame_key
    from .visualization import draw_pose_axes_on_image, save_render_comparison

    # Coerce the input to FrameKeys so set/dict lookups against tokens_by_object
    # (which are FrameKey-keyed) work uniformly. Bare ints lift to FrameKey(int, 0).
    frame_indices = [as_frame_key(f) for f in frame_indices]
    if before_renders is not None:
        before_renders = {as_frame_key(k): v for k, v in before_renders.items()}
    if before_poses is not None:
        before_poses = {
            obj_idx: {as_frame_key(k): v for k, v in obj_dict.items()}
            for obj_idx, obj_dict in before_poses.items()
        }

    rendered_frames = []
    gt_frames = []
    frame_indices_processed = []

    # Filter frame_indices to only those that have poses for ALL objects
    frame_sets_per_object = []
    for _, tokens_list in tokens_by_object.items():
        obj_frames = set(fid for fid, _ in tokens_list)
        frame_sets_per_object.append(obj_frames)

    if frame_sets_per_object:
        available_frame_indices = frame_sets_per_object[0]
        for obj_frames in frame_sets_per_object[1:]:
            available_frame_indices = available_frame_indices & obj_frames
    else:
        available_frame_indices = set()

    frame_indices_to_process = [f for f in frame_indices if f in available_frame_indices]

    if len(frame_indices_to_process) < len(frame_indices):
        print(
            f"\nNote: Only {len(frame_indices_to_process)} of {len(frame_indices)} "
            "requested frames have poses for all objects"
        )
        print(f"  Frames with complete poses: {sorted(available_frame_indices)}")
        for obj_idx, tokens_list in tokens_by_object.items():
            obj_frames = set(fid for fid, _ in tokens_list)
            missing = set(frame_indices) - obj_frames
            if missing:
                print(f"  Object {obj_idx} missing frames: {sorted(missing)}")

    per_object_data: Dict[int, Dict[str, List[torch.Tensor]]] = {}

    # Initialize per-object storage using actual object IDs from tokens_by_object
    for obj_id in tokens_by_object.keys():
        per_object_data[obj_id] = {"rendered": [], "gt": [], "frame_indices": []}

    # Initialize temporal point cloud storage
    _output_dir = args.output.output_dir
    _scene_name = args.dataset.scene_name
    _background = args.processing.add_background_gaussians
    _save_renders_flag = args.output.save_renders

    # Pre-compute 3D pose axes for overlay (before and after)
    after_axes_3d = None
    before_axes_3d = None
    if save_renders and _save_renders_flag and frame_indices_to_process:
        from .interpolation import compute_pose_axes

        after_snapshot = capture_pose_snapshot(tokens_by_object)
        after_axes_3d = compute_pose_axes(after_snapshot, frame_indices_to_process)
        if before_poses is not None:
            before_axes_3d = compute_pose_axes(before_poses, frame_indices_to_process)

    # Process each frame
    for frame_idx, frame_index in enumerate(frame_indices_to_process):
        print(f"\n  Processing frame {_fmt_frame_key(frame_index)} ({frame_idx + 1}/{len(frame_indices_to_process)})")

        _bg_color = torch.ones(3) if getattr(args.pipeline, "white_background", False) else None
        result = process_frame_with_canonical_object(
            sequence,
            frame_index,
            canonical_gaussians,
            tokens_by_object,
            per_frame_canonical=per_frame_canonical,
            background=_background,
            bg_color=_bg_color,
            canonical_mesh_verts=canonical_mesh_verts,
            per_frame_mesh_verts=per_frame_mesh_verts,
            per_frame_mesh_rotations=per_frame_mesh_rotations,
            canonical_mesh_faces=canonical_mesh_faces,
            warp_knn_k=warp_knn_k,
            warp_knn_eps=warp_knn_eps,
            warp_knn_chunk_size=warp_knn_chunk_size,
        )

        if result is None:
            print(f"    Skipping frame {frame_index} - no objects to render")
            continue

        rendered, gt_image, K_matrix, masks, object_ids, scene_gs = result

        # Convert to format expected by evaluator: (B, C, H, W)
        rendered_eval = rendered.permute(2, 0, 1).unsqueeze(0).to(device)
        gt_eval = gt_image.permute(2, 0, 1).unsqueeze(0).to(device)

        # A "before" snapshot is only required for "_after_<block>" comparison
        # renders. When upstream blocks did not populate Gaussians,
        # capture_before() returns None and the
        # comparison panel would be empty — skip writing the orphan PNG.
        _orphan_after = suffix.startswith("_after_") and before_renders is None
        if save_renders and _save_renders_flag and not _orphan_after:
            render_dir = os.path.join(_output_dir, "renders")
            os.makedirs(render_dir, exist_ok=True)
            # Use fk.frame for filename to preserve mono-run byte-identical layout.
            # MV-static post-axis-lift collides on frame=0 for all views; that case
            # is handled by the per-view subdir injection in viz_io.write_per_view,
            # which this writer does not route through.
            _name_idx = frame_index.frame if hasattr(frame_index, "frame") else int(frame_index)
            output_path = os.path.join(
                render_dir, f"{_scene_name}_frame_{_name_idx:04d}{suffix}.png"
            )
            before = before_renders.get(frame_index) if before_renders else None

            # Compute pose axes overlays for this frame.
            # Axes are always in R3 camera space (poses are camera-space),
            # so project with identity c2w (no c2w arg).
            after_axes_img = None
            before_axes_img = None
            if after_axes_3d is not None:
                after_axes_img = draw_pose_axes_on_image(
                    rendered, after_axes_3d, frame_idx, K_matrix,
                )
            if before is not None and before_axes_3d is not None:
                before_axes_img = draw_pose_axes_on_image(
                    before, before_axes_3d, frame_idx, K_matrix,
                )

            # Union of all object GT masks for foreground-only PSNR
            union_mask = torch.from_numpy(
                np.any(np.stack(masks, axis=0), axis=0)
            ).bool() if masks else None

            save_render_comparison(
                gt_image, rendered, output_path, before_render=before,
                mask=union_mask,
                before_axes_overlay=before_axes_img,
                after_axes_overlay=after_axes_img,
            )
            _CONSOLE.print(f"[dim]    Saved comparison → {fmt_path(output_path)}[/dim]")

        rendered_frames.append(rendered_eval)
        gt_frames.append(gt_eval)
        frame_indices_processed.append(frame_index)

        # Store per-object masked data for evaluation
        for obj_id, mask in zip(object_ids, masks):
            if obj_id not in per_object_data:
                per_object_data[obj_id] = {"rendered": [], "gt": [], "frame_indices": []}
            mask_tensor = torch.from_numpy(mask).float().to(device)
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)

            if _bg_color is not None:
                bg_bchw = _bg_color.view(1, 3, 1, 1).to(device)
                rendered_masked = rendered_eval * mask_tensor + bg_bchw * (1.0 - mask_tensor)
                gt_masked = gt_eval * mask_tensor + bg_bchw * (1.0 - mask_tensor)
            else:
                rendered_masked = rendered_eval * mask_tensor
                gt_masked = gt_eval * mask_tensor

            per_object_data[obj_id]["rendered"].append(rendered_masked)
            per_object_data[obj_id]["gt"].append(gt_masked)
            per_object_data[obj_id]["frame_indices"].append(frame_index)

    if not rendered_frames:
        print("\nNo frames were rendered for evaluation — skipping metrics.")
        return None

    return _build_evaluation_summary(
        evaluator,
        rendered_frames,
        gt_frames,
        frame_indices_processed,
        per_object_data,
        args,
        suffix,
    )


def print_evaluation_summary(summary: Dict[str, Any], title: str = "Evaluation Summary") -> None:
    """
    Print evaluation metrics in a formatted way.

    Parameters
    ----------
    summary : dict
        Evaluation summary dictionary.
    title : str, optional
        Title for the summary output.
    """
    n_frames = summary["num_frames"]
    single = n_frames == 1

    table = _metrics_table(
        f"[bold]{title}[/bold]  ·  {n_frames} frame{'' if single else 's'} evaluated",
        "Region",
    )

    def _cells(frames):
        """Return (psnr, ssim, lpips) cell strings for a list of per-frame dicts."""
        psnr = np.array([f["psnr"] for f in frames])
        ssim = np.array([f["ssim"] for f in frames])
        lpip = np.array([f["lpip"] for f in frames])
        if single:
            return f"{psnr[0]:.2f}", f"{ssim[0]:.4f}", f"{lpip[0]:.4f}"
        return (
            f"{psnr.mean():.2f} ± {psnr.std():.2f}",
            f"{ssim.mean():.4f} ± {ssim.std():.4f}",
            f"{lpip.mean():.4f} ± {lpip.std():.4f}",
        )

    per_frame = summary.get("per_frame", {})
    if "full_frame" in per_frame:
        table.add_row("Full frame", *_cells(per_frame["full_frame"]))
    for key in sorted(per_frame):
        if key != "full_frame":
            table.add_row(f"{key} (masked)", *_cells(per_frame[key]))

    _CONSOLE.print(table)


def capture_all_renders(
    canonical_gaussians,
    tokens_by_object,
    sequence,
    frame_indices,
    per_frame_canonical=False,
    background=True,
    bg_color=None,
    canonical_mesh_verts=None,
    per_frame_mesh_verts=None,
    per_frame_mesh_rotations=None,
    canonical_mesh_faces=None,
    warp_knn_k: int = 4,
    warp_knn_eps: float = 1.0e-8,
    warp_knn_chunk_size: int = 8192,
):
    """Render all frames with current poses, returning per-frame rendered images.

    Used to capture "before" renders prior to refinement, so that
    before/after comparison figures can be generated.

    Parameters
    ----------
    canonical_gaussians : dict
        Dictionary mapping obj_idx -> Gaussian.
    tokens_by_object : dict
        Dictionary mapping obj_idx -> list of (frame_idx, decoder_input).
    sequence : Sequence
        Cached scene data.
    frame_indices : list of int
        Frame indices to render.
    per_frame_canonical : bool, optional
        If True, use per-frame canonical Gaussians. Default: False.
    background : bool, optional
        Whether to add background Gaussians. Default: True.
    bg_color : torch.Tensor or None, optional
        Background color (3,). Defaults to black.

    Returns
    -------
    dict
        Dictionary mapping frame_idx -> rendered tensor (H, W, 3) on CPU.
    """
    renders = {}
    with torch.no_grad():
        for frame_index in frame_indices:
            result = process_frame_with_canonical_object(
                sequence, frame_index,
                canonical_gaussians, tokens_by_object,
                per_frame_canonical=per_frame_canonical,
                background=background,
                bg_color=bg_color,
                canonical_mesh_verts=canonical_mesh_verts,
                per_frame_mesh_verts=per_frame_mesh_verts,
                per_frame_mesh_rotations=per_frame_mesh_rotations,
                canonical_mesh_faces=canonical_mesh_faces,
                warp_knn_k=warp_knn_k,
                warp_knn_eps=warp_knn_eps,
                warp_knn_chunk_size=warp_knn_chunk_size,
            )
            if result is not None:
                rendered, gt_image, _, _, _, _ = result
                renders[frame_index] = rendered.cpu()
    return renders


def capture_pose_snapshot(tokens_by_object):
    """Deep-copy pose tensors from tokens_by_object.

    Returns a dict compatible with :func:`~genia.core.utils.interpolation.compute_pose_axes`
    input format::

        {obj_idx: {frame_idx: {"rotation": tensor, "translation": tensor, "scale": tensor}}}
    """
    snapshot = {}
    for obj_idx, tokens_list in tokens_by_object.items():
        snapshot[obj_idx] = {}
        for frame_idx, decoder_input in tokens_list:
            snapshot[obj_idx][frame_idx] = {
                "rotation": decoder_input["rotation"].detach().clone(),
                "translation": decoder_input["translation"].detach().clone(),
                "scale": decoder_input["scale"].detach().clone(),
            }
    return snapshot


# =====================================================================
# Pipeline block helpers
# =====================================================================


def evaluate_block(cfg, state, sequence, evaluator, device, output_dir,
                   suffix, per_frame, before_renders=None, before_poses=None,
                   save_renders=None):
    """Run evaluation for a pipeline block and print summary.

    Parameters
    ----------
    per_frame : bool
        True = evaluate per-frame Gaussians, False = evaluate canonical Gaussians.
    save_renders : bool, optional
        Override for ``cfg.output.save_renders``.  None ⇒ inherit from cfg.
    """
    from .config import output_dir_redirect

    gaussians = state.perframe_gaussians if per_frame else state.canonical_gaussians_with_fallback
    _save_renders = cfg.output.save_renders if save_renders is None else save_renders

    # Nothing to write ⇒ skip the evaluation itself, not just its writes: it costs a
    # render pass + LPIPS per frame and no caller reads the return value, so a
    # suppressed run computed a summary only to print and discard it.  Inside a block
    # ``cfg.output.*`` is already this block's resolved value (``block_output_context``);
    # FINAL runs outside it, on the global — unsuppressed — flags.
    if not _save_renders and not cfg.output.save_metrics:
        return None

    with output_dir_redirect(cfg, output_dir):
        summary = evaluate_with_canonical_objects(
            cfg, sequence, sequence.frame_indices,
            gaussians, state.tokens_by_object, evaluator, device,
            suffix=suffix, save_renders=_save_renders,
            per_frame_canonical=per_frame,
            before_renders=before_renders,
            before_poses=before_poses,
            **_deformation_warp_kwargs(state, cfg),
        )
    if summary:
        print_evaluation_summary(summary, f"Results {suffix.replace('_', ' ')}")
    return summary


def _deformation_warp_kwargs(state, cfg):
    """Per-vertex ActionMesh deformation field + warp params from state/cfg.

    Returns the kwargs accepted by ``evaluate_with_canonical_objects`` /
    ``capture_all_renders`` / ``render_keyframes`` so the per-frame
    comparison/eval/keyframe renders apply the same warp. All-``None``
    field ⇒ rigid path.

    The frame-invariant tensors (canonical mesh verts) are moved to the
    GPU here — once per block call, not once per
    frame — so ``_lookup_per_frame_deformation``'s per-frame ``.to(device)``
    no-ops on them.  Mirrors the move-once pattern in ``run_finetuning``.
    The per-frame verts/rotations stay on CPU (one intrinsic transfer per
    frame regardless).
    """
    cmv = state.canonical_mesh_verts or None
    faces = state.canonical_mesh_faces or None
    if cmv is not None and torch.cuda.is_available():
        _dev = torch.device("cuda")
        cmv = {oi: t.to(_dev) for oi, t in cmv.items()}
        if faces:
            faces = {oi: t.to(_dev) for oi, t in faces.items()}
    return dict(
        canonical_mesh_verts=cmv,
        per_frame_mesh_verts=state.canonical_mesh_per_frame_verts or None,
        per_frame_mesh_rotations=state.canonical_mesh_per_frame_rotations or None,
        canonical_mesh_faces=faces,
        warp_knn_k=int(cfg.deformation_warp.knn_k),
        warp_knn_eps=float(cfg.deformation_warp.knn_eps),
        warp_knn_chunk_size=int(cfg.deformation_warp.knn_chunk_size),
    )


def capture_before(state, sequence, cfg, per_frame):
    """Capture renders and pose snapshot before a block runs.

    Returns (before_renders, before_poses) or (None, None) if save_renders is off.
    """
    if not cfg.output.save_renders:
        return None, None

    gaussians = state.perframe_gaussians if per_frame else state.canonical_gaussians_with_fallback
    if not gaussians:
        return None, None

    # The pose snapshot + posed before-renders need per-frame pose tokens.
    # Some blocks run before pose init (e.g. APPEARANCE_INIT, which populates
    # per-frame Gaussians for the photometric gpr1 of the per-frame recipe) —
    # Gaussians exist but poses don't yet.  Skip the before-capture in that case.
    has_poses = any(
        "rotation" in di
        for toks in state.tokens_by_object.values()
        for _fid, di in toks
    )
    if not has_poses:
        return None, None

    _bg_color = torch.ones(3) if cfg.pipeline.white_background else None

    before_poses = capture_pose_snapshot(state.tokens_by_object)
    before_renders = capture_all_renders(
        gaussians, state.tokens_by_object,
        sequence=sequence, frame_indices=sequence.frame_indices,
        per_frame_canonical=per_frame, background=cfg.processing.add_background_gaussians,
        bg_color=_bg_color,
        **_deformation_warp_kwargs(state, cfg),
    )
    return before_renders, before_poses


# =====================================================================
# Voxel mesh fallback rendering
# =====================================================================

def render_voxel_keyframes(voxel_coords_by_object, tokens_by_object,
                           sequence, per_frame=False, overlay_axes=True,
                           render_space="camera",
                           voxel_colors_by_object=None,
                           frame_indices=None,
                           canonical_mesh_verts=None,
                           per_frame_mesh_verts=None,
                           per_frame_mesh_rotations=None,
                           canonical_mesh_faces=None,
                           warp_knn_k: int = 4,
                           warp_knn_eps: float = 1.0e-8,
                           warp_knn_chunk_size: int = 8192):
    """Render keyframes using mesh-based voxel grids instead of Gaussians.

    Used as a fallback when canonical Gaussians are invalidated but the
    coarse shape voxel grid is available.

    Parameters
    ----------
    voxel_coords_by_object : dict
        If ``per_frame=False``: ``{obj_idx: np.ndarray (N, 3)}`` — canonical grid.
        If ``per_frame=True``:  ``{obj_idx: {frame_idx: np.ndarray (N, 3)}}``.
    tokens_by_object : dict
        Per-object per-frame poses.
    sequence : Sequence
    per_frame : bool
        Whether voxel coords (and colors) are per-frame (True) or canonical (False).
    overlay_axes : bool
        If True (default), overlay 3D pose axes on each object.
    render_space : str
        ``"camera"`` renders from the per-frame camera.
        ``"world"`` renders from an automatically computed overview camera.
    voxel_colors_by_object : dict, optional
        Per-voxel RGB colors in [0, 1]. Same shape convention as
        ``voxel_coords_by_object`` (canonical or per-frame, see above).
        When provided, overrides the default XYZ position coloring.
    canonical_mesh_verts, per_frame_mesh_verts, per_frame_mesh_rotations : dict or None
        Per-canonical-mesh-vertex deformation field (state-shaped dicts
        matching ``state.canonical_mesh_*``).  When all three are provided
        AND the ``(obj_idx, frame_int)`` lookup hits, the canonical voxel
        coords are warped per frame via ``_warp_at_high_res`` BEFORE the
        Stage-1 pose is applied.  Falls through to the static-canonical
        rendering when any piece is missing.

    Returns
    -------
    frames : list of PIL.Image
    """
    from PIL import Image as PILImage

    from genia.core.utils.deformation import _lookup_per_frame_deformation, warp_voxel_coords_high_res
    from .visualization import render_voxel_meshes_in_camera

    if overlay_axes:
        from .interpolation import compute_pose_axes
        from .visualization import draw_pose_axes_on_image

    if frame_indices is None:
        frame_indices = sequence.frame_indices

    # Pre-compute overview camera for world-space rendering
    overview = None
    if render_space == "world":
        overview_c2w, overview_K, overview_W, overview_H = compute_scene_overview_c2w(
            sequence, frame_indices, tokens_by_object,
        )
        overview = {
            "c2w": overview_c2w, "K": overview_K,
            "W": overview_W, "H": overview_H,
        }

    frames = []

    for fi in frame_indices:
        # Collect all objects for this frame and render in a single pass
        # so the rasterizer handles depth ordering correctly.
        objects = []
        frame_poses = {}
        for obj_idx in sorted(voxel_coords_by_object.keys()):
            if per_frame:
                _per_obj = voxel_coords_by_object[obj_idx]
                if fi not in _per_obj:
                    continue
                coords = _per_obj[fi]
            else:
                coords = voxel_coords_by_object[obj_idx]

            frame_pose = None
            for fid, di in tokens_by_object[obj_idx]:
                if fid == fi:
                    frame_pose = di
                    break
            if frame_pose is None or "rotation" not in frame_pose:
                # Pre-pose blocks (the split recipe's shape_init pass, before any
                # pose block has run) carry pose-less token dicts — the start-of-run
                # skeleton entries are (frame, {}).  No pose means nothing to place:
                # skip the object like a missing entry instead of crashing.
                continue

            frame_poses[obj_idx] = frame_pose

            rot_q = frame_pose["rotation"]
            trans = frame_pose["translation"]
            scale = frame_pose["scale"]

            if isinstance(rot_q, torch.Tensor):
                rot_q = rot_q.detach().cpu()
            else:
                rot_q = torch.tensor(rot_q)
            if rot_q.dim() == 2:
                rot_q = rot_q.squeeze(0)
            rot_q = rot_q / rot_q.norm()
            R = quaternion_to_matrix(rot_q.unsqueeze(0)).squeeze(0).numpy()

            if isinstance(trans, torch.Tensor):
                trans = trans.detach().cpu().numpy().flatten()
            else:
                trans = np.asarray(trans).flatten()

            if isinstance(scale, torch.Tensor):
                scale = scale.detach().cpu().numpy().flatten()
            else:
                scale = np.atleast_1d(np.asarray(scale)).flatten()
            if scale.shape[0] == 1:
                scale = np.broadcast_to(scale, (3,))

            # Per-vertex deformation warp (actionmesh).  Warps the
            # canonical voxel coords by the per-frame deformation field
            # BEFORE the Stage-1 pose is applied below.  Falls through to
            # the static-canonical voxel rendering when the field is
            # not loaded for this (obj, frame).
            coords_for_pose = coords
            resolved = _lookup_per_frame_deformation(
                canonical_mesh_verts,
                per_frame_mesh_verts,
                per_frame_mesh_rotations,
                obj_idx,
                int(getattr(fi, "frame", fi)),
                torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                canonical_mesh_faces,
            )
            if resolved is not None:
                *_warp_core, _faces = resolved
                coords_for_pose = warp_voxel_coords_high_res(
                    coords, *_warp_core,
                    K=warp_knn_k,
                    eps=warp_knn_eps,
                    chunk_size=warp_knn_chunk_size,
                    faces=_faces,
                )

            obj_dict = {
                "voxel_coords_np": coords_for_pose.astype(np.float32),
                "obj_rotation": R,
                "obj_translation": trans,
                "obj_scale": scale,
            }
            if voxel_colors_by_object and obj_idx in voxel_colors_by_object:
                _obj_colors = voxel_colors_by_object[obj_idx]
                if per_frame:
                    if fi in _obj_colors:
                        obj_dict["_voxel_colors"] = _obj_colors[fi]
                else:
                    obj_dict["_voxel_colors"] = _obj_colors
            objects.append(obj_dict)

        # Skip frames where no object has a pose for this (frame, view) —
        # mirrors render_keyframes' ``if not object_gaussians: continue``.
        # Without it, render_voxel_meshes_in_camera returns a blank white
        # image for an empty object list, producing spurious blank per-view
        # voxel PNGs for views the object was never reconstructed in (e.g. a
        # single-view run on a multi-camera CO3D/GSO scene).
        if not objects:
            continue

        frame_data = sequence[fi]
        if render_space == "world" and overview is not None:
            composite = render_voxel_meshes_in_camera(
                objects, overview["K"], overview["W"], overview["H"],
                c2w=frame_data.c2w,
                render_c2w=overview["c2w"],
            )
        else:
            composite = render_voxel_meshes_in_camera(
                objects, frame_data.K_matrix, sequence.W, sequence.H,
            )

        if composite is not None:
            if overlay_axes and frame_poses:
                single_frame = {oi: {fi: p} for oi, p in frame_poses.items()}
                axes_data = compute_pose_axes(single_frame, [fi])
                rendered_t = torch.from_numpy(
                    np.array(composite).astype(np.float32) / 255.0
                )
                if render_space == "world" and overview is not None:
                    # Axes are in camera space; transform to world for overview projection
                    from .rendering import transform_gaussian_params_cam_to_world
                    for obj_idx in axes_data:
                        c_pts = axes_data[obj_idx]["centroids"]
                        a_pts = axes_data[obj_idx]["axes"]
                        T_ax = c_pts.shape[0]
                        all_pts = np.concatenate(
                            [c_pts, a_pts.reshape(T_ax, -1).reshape(-1, 3)], axis=0,
                        )
                        all_t = torch.from_numpy(all_pts).float()
                        dummy_q = torch.tensor([[1.0, 0, 0, 0]]).expand(all_t.shape[0], -1)
                        world_pts, _ = transform_gaussian_params_cam_to_world(
                            all_t, dummy_q, frame_data.c2w,
                        )
                        world_pts = world_pts.numpy()
                        axes_data[obj_idx]["centroids"] = world_pts[:T_ax]
                        axes_data[obj_idx]["axes"] = world_pts[T_ax:].reshape(T_ax, 3, 3)
                    rendered_np = draw_pose_axes_on_image(
                        rendered_t, axes_data, 0, overview["K"],
                        c2w=overview["c2w"],
                    )
                else:
                    rendered_np = draw_pose_axes_on_image(
                        rendered_t, axes_data, 0, frame_data.K_matrix,
                    )
                composite = PILImage.fromarray(
                    (rendered_np * 255).astype(np.uint8)
                )
            # Voxel keyframes intentionally don't concatenate GT to the side:
            # voxel colors are synthetic (XYZ position / shape PCA / SLAT PCA)
            # and have no comparable GT image. Future: if render_voxel_meshes_in_camera
            # gains a depth-output mode, GT depth from `sequence[fi].depth_map_z`
            # becomes a meaningful comparison and can be attached here.
            frames.append(composite)

    return frames


# =====================================================================
# Keyframe video rendering
# =====================================================================

def _xyz_to_sh_dc_features(xyz: torch.Tensor) -> torch.Tensor:
    """Convert XYZ positions to SH DC features for visualization.

    Normalizes XYZ to [0, 1] per dimension and encodes as 0th-order SH
    coefficients so that gsplat renders them as RGB colors.

    Parameters
    ----------
    xyz : torch.Tensor
        Gaussian positions in local/object space, shape (N, 3).

    Returns
    -------
    torch.Tensor
        SH DC features, shape (N, 1, 3).
    """
    SH_C0 = 0.28209479177387814
    lo = xyz.min(dim=0).values  # (3,)
    hi = xyz.max(dim=0).values  # (3,)
    span = hi - lo
    span = span.clamp(min=1e-6)  # avoid division by zero
    xyz_norm = (xyz - lo) / span  # [0, 1]
    sh_dc = (xyz_norm - 0.5) / SH_C0  # invert SH evaluation: color = C0 * sh + 0.5
    return sh_dc.unsqueeze(1)  # (N, 1, 3)


def _gt_tensor_foreground(image, masks, bg_color, has_background=False):
    """GT frame as a float tensor in [0, 1], background-filtered for comparison.

    Thin adapter over ``composite_on_render_bg`` (which owns the rationale) for
    the eval path, whose GT is numpy uint8 in and torch float out.
    *has_background* ⇒ the render carries the dataset background too, so GT is
    comparable as captured and passes through untouched.
    """
    from .rendering import composite_on_render_bg

    if not has_background:
        image = composite_on_render_bg(image, list(masks or []), bg_color)
    return torch.from_numpy(np.ascontiguousarray(image)).float() / 255.0


def render_keyframes(gaussians, tokens_by_object, sequence, per_frame=False,
                     bg_color=None, overlay_axes=False, background=False,
                     color_mode="default", render_space="camera",
                     frame_indices=None,
                     canonical_mesh_verts=None,
                     per_frame_mesh_verts=None,
                     per_frame_mesh_rotations=None,
                     canonical_mesh_faces=None,
                     warp_knn_k: int = 4,
                     warp_knn_eps: float = 1.0e-8,
                     warp_knn_chunk_size: int = 8192):
    """Render keyframes into PIL frames for video.

    Parameters
    ----------
    gaussians : dict
        If per_frame=False: {obj_idx: canonical_gaussian}
        If per_frame=True: {obj_idx: {frame_idx: gaussian}}
    tokens_by_object : dict
        Per-object per-frame poses.
    sequence : Sequence
    per_frame : bool
        Whether gaussians are per-frame (True) or canonical (False).
    bg_color : torch.Tensor or None
        Background color for rendering (3,). None = black.
    overlay_axes : bool
        If True, overlay 3D pose axes on each object.
    background : bool
        If True, add background Gaussians from the pointmap.
    color_mode : str
        ``"default"`` uses original SH colors; ``"xyz"`` colors each
        Gaussian by its local-space XYZ coordinate (normalized to [0, 1]).
    render_space : str
        ``"camera"`` transforms bg to camera space, renders with identity.
        ``"world"`` transforms fg to world space and renders from an
        automatically computed overview camera that sees the full scene.
    canonical_mesh_verts, per_frame_mesh_verts, per_frame_mesh_rotations : dict or None
        Per-canonical-mesh-vertex deformation field (state-shaped dicts
        matching ``state.canonical_mesh_*``).  When all three are provided
        AND the ``(obj_idx, frame_int)`` lookup hits, the canonical
        Gaussian is warped per frame via ``warp_gaussians_high_res``
        BEFORE the Stage-1 pose is applied.  Falls through to the
        static-canonical render when any piece is missing — non-actionmesh
        scenes and pre-``GT_SHAPES_INVERSION`` blocks are unaffected.

    Returns
    -------
    frames : list of PIL.Image
    """
    from PIL import Image

    from .gaussian import (
        create_background_gaussians,
        create_gaussians_object,
        join_gaussians,
        transform_scene_to_r3_convention,
        transform_scene_to_world,
    )
    from .refinement import apply_pose_to_gaussian
    from .rendering import (
        composite_on_render_bg,
        render_gaussians_to_image,
        transform_gaussian_params_world_to_cam,
    )
    from genia.core.utils.deformation import _lookup_per_frame_deformation, warp_gaussians_high_res

    if frame_indices is None:
        frame_indices = sequence.frame_indices

    if overlay_axes:
        from .interpolation import compute_pose_axes
        from .visualization import draw_pose_axes_on_image

    # Pre-compute overview camera for world-space rendering.  The canonical centroids let
    # it frame on the actual geometry rather than on the Sim(3) translation alone, which
    # is zero for an asset that carries its placement itself (see
    # ``_object_camera_position``).  Skipped on the per-frame path, where ``gaussians[obj]``
    # is a ``{frame: Gaussian}`` dict with no single canonical to take a centroid of; a
    # per-frame asset that carries its placement would need centroids keyed by
    # ``(obj, frame)``.
    overview = None
    if render_space == "world":
        centroids = None
        if not per_frame:
            centroids = {
                oi: gs.get_xyz.detach().median(dim=0).values.cpu().numpy()
                for oi, gs in gaussians.items()
            }
        overview_c2w, overview_K, overview_W, overview_H = compute_scene_overview_c2w(
            sequence, frame_indices, tokens_by_object,
            object_local_centroids=centroids,
        )
        overview = {
            "c2w": overview_c2w, "K": overview_K,
            "W": overview_W, "H": overview_H,
        }

    # World-space + background: aggregate background gaussians from all
    # available frames so the overview render shows the full reconstructed
    # scene rather than only the slice visible in the current frame's
    # camera. Camera-space renders keep the per-frame bg path because each
    # frame's bg lives in its own camera frame.
    aggregated_bg_world = None
    if render_space == "world" and background:
        from .gaussian import aggregate_background_gaussians
        aggregated_bg_world = aggregate_background_gaussians(
            sequence, sorted(gaussians.keys())
        )

    frames = []

    for fi in frame_indices:
        object_gaussians = []
        # obj_idx -> pose for the objects actually rendered this frame; also
        # the mask key set for the GT panel below (and the axes overlay).
        frame_poses = {}
        for obj_idx in sorted(gaussians.keys()):
            if per_frame:
                if fi not in gaussians[obj_idx]:
                    continue
                gs = gaussians[obj_idx][fi]
            else:
                gs = gaussians[obj_idx]

            frame_pose = None
            for fid, di in tokens_by_object[obj_idx]:
                if fid == fi:
                    frame_pose = di
                    break
            if frame_pose is None or "rotation" not in frame_pose:
                # Pre-pose blocks (e.g. APPEARANCE_INIT in the split recipe,
                # which decodes per-frame Gaussians BEFORE pose_init) carry
                # pose-less token dicts — skip like a missing entry.
                continue

            frame_poses[obj_idx] = frame_pose

            # Per-vertex deformation warp (actionmesh).  When the GT
            # deformation field is loaded AND keyed for this (obj, frame),
            # warp the canonical Gaussian's means/quats by the per-frame
            # field BEFORE applying Stage-1 pose.  Falls through to the
            # static-canonical path when any piece is missing.
            means_override = quats_override = None
            resolved = _lookup_per_frame_deformation(
                canonical_mesh_verts,
                per_frame_mesh_verts,
                per_frame_mesh_rotations,
                obj_idx,
                int(getattr(fi, "frame", fi)),
                gs.get_xyz.device,
                canonical_mesh_faces,
            )
            if resolved is not None:
                *_warp_core, _faces = resolved
                with torch.no_grad():
                    means_override, quats_override = warp_gaussians_high_res(
                        gs, *_warp_core,
                        K=warp_knn_k,
                        eps=warp_knn_eps,
                        chunk_size=warp_knn_chunk_size,
                        faces=_faces,
                    )

            xyz, rots, scs, opacities, features = apply_pose_to_gaussian(
                gs, frame_pose["rotation"], frame_pose["translation"],
                frame_pose["scale"],
                means_override=means_override,
                rotation_override=quats_override,
            )
            if color_mode == "xyz":
                features = _xyz_to_sh_dc_features(gs.get_xyz)
            obj_gs = create_gaussians_object(
                xyz=xyz, features=features, scales=scs,
                rots=rots, opacities=opacities,
            )
            object_gaussians.append(obj_gs)

        if not object_gaussians:
            continue

        if len(object_gaussians) == 1:
            scene_gs = object_gaussians[0]
        else:
            scene_gs = join_gaussians(*object_gaussians)
        scene_gs = transform_scene_to_r3_convention(scene_gs)

        frame_data = sequence[fi]
        K_matrix = frame_data.K_matrix
        frame_c2w = frame_data.c2w

        if render_space == "world":
            # Transform fg to world space; bg is already world space; render with c2w
            scene_gs = transform_scene_to_world(scene_gs, frame_c2w)
            if aggregated_bg_world is not None:
                scene_gs = join_gaussians(aggregated_bg_world, scene_gs)
            render_c2w = overview["c2w"]
            render_K = overview["K"]
            render_W = overview["W"]
            render_H = overview["H"]
        else:
            # Camera-space: bg warped to camera space; fg already camera space; render with identity
            if background:
                masks = [frame_data.masks[oid] for oid in sorted(gaussians.keys())]
                bg_gs = create_background_gaussians(
                    frame_data.image, frame_data.pointmap, masks, K_matrix,
                    c2w=frame_c2w,
                )
                if bg_gs is not None:
                    # Transform bg from world to camera space
                    bg_xyz, bg_quats = transform_gaussian_params_world_to_cam(
                        bg_gs.get_xyz, bg_gs.get_rotation, frame_c2w,
                    )
                    bg_gs = create_gaussians_object(
                        xyz=bg_xyz, features=bg_gs.get_features,
                        scales=bg_gs.get_scaling, rots=bg_quats,
                        opacities=bg_gs.get_opacity,
                    )
                    scene_gs = join_gaussians(bg_gs, scene_gs)
            render_c2w = np.eye(4, dtype=np.float32)
            render_K = K_matrix
            render_W = sequence.W
            render_H = sequence.H

        rendered = render_gaussians_to_image(
            scene_gs, render_K, render_W, render_H,
            bg_color=bg_color, c2w=render_c2w,
        )
        rendered = torch.clamp(rendered.cpu(), 0.0, 1.0)

        if overlay_axes and frame_poses:
            single_frame = {oi: {fi: p} for oi, p in frame_poses.items()}
            axes_data = compute_pose_axes(single_frame, [fi])
            if render_space == "world":
                # Axes are in camera space; transform to world for overview camera projection
                from .rendering import transform_gaussian_params_cam_to_world
                for obj_idx in axes_data:
                    c_pts = axes_data[obj_idx]["centroids"]  # (T, 3)
                    a_pts = axes_data[obj_idx]["axes"]        # (T, 3, 3)
                    T_ax = c_pts.shape[0]
                    # Stack all points, transform, unstack
                    all_pts = np.concatenate([c_pts, a_pts.reshape(T_ax, -1).reshape(-1, 3)], axis=0)
                    all_t = torch.from_numpy(all_pts).float()
                    dummy_q = torch.tensor([[1.0, 0, 0, 0]]).expand(all_t.shape[0], -1)
                    world_pts, _ = transform_gaussian_params_cam_to_world(all_t, dummy_q, frame_c2w)
                    world_pts = world_pts.numpy()
                    axes_data[obj_idx]["centroids"] = world_pts[:T_ax]
                    axes_data[obj_idx]["axes"] = world_pts[T_ax:].reshape(T_ax, 3, 3)
                rendered_np = draw_pose_axes_on_image(
                    rendered, axes_data, 0, render_K, c2w=render_c2w,
                )
            else:
                # Axes are in camera space; project with identity c2w + per-frame K
                rendered_np = draw_pose_axes_on_image(rendered, axes_data, 0, K_matrix)
            pred_uint8 = (rendered_np * 255).astype("uint8")
        else:
            pred_uint8 = (rendered.numpy() * 255).astype("uint8")

        # Concatenate ``[GT | Pred | 3*|GT-Pred|]`` when GT is comparable.
        # GT is comparable only for camera-space renders (dataset camera +
        # dataset K), where ``frame_data.image`` was captured by the same
        # camera the render simulates. World-space renders use a synthetic
        # overview camera with no GT counterpart — left pred-only.
        gt_img = getattr(frame_data, "image", None)
        if (render_space == "camera"
                and gt_img is not None
                and gt_img.ndim == 3
                and gt_img.shape[2] >= 3):
            gt_uint8 = np.ascontiguousarray(gt_img[..., :3]).astype("uint8")
            if gt_uint8.shape[:2] != pred_uint8.shape[:2]:
                # Resize GT to match render (handles rare K/W/H drift).
                gt_uint8 = np.array(
                    Image.fromarray(gt_uint8).resize(
                        (pred_uint8.shape[1], pred_uint8.shape[0]),
                        Image.BILINEAR,
                    )
                )
            # ``background=True`` ⇒ the render carries the dataset background,
            # so GT is already comparable as captured.
            if not background:
                _masks = getattr(frame_data, "masks", None) or {}
                gt_uint8 = composite_on_render_bg(
                    gt_uint8, [_masks[o] for o in frame_poses], bg_color,
                )
            diff_uint8 = np.clip(
                np.abs(gt_uint8.astype(np.int16) - pred_uint8.astype(np.int16)) * 3,
                0, 255,
            ).astype("uint8")
            composite = np.concatenate([gt_uint8, pred_uint8, diff_uint8], axis=1)
            img = Image.fromarray(composite)
        else:
            img = Image.fromarray(pred_uint8)
        frames.append(img)

    return frames


def _object_camera_position(pose: "Dict", centroid_local: "Optional[torch.Tensor]") -> np.ndarray:
    """Where the object sits in camera space: its canonical centroid under the Sim(3).

    Reduces to ``pose["translation"]`` to the extent the canonical sits at its own local
    origin: exactly when ``centroid_local`` is None, and near-exactly for a decoded SLAT
    canonical (centred near, not exactly at, the origin).  It matters for runs where that
    assumption fails outright — see
    ``PipelineState.pose_carries_placement`` — whose Sim(3) is IDENTITY with the placement
    baked into the asset: there the translation alone reports the camera origin, collapsing
    the overview framing onto the camera cluster.

    Composition order matches ``refinement.apply_pose_to_gaussian`` — scale, then rotate
    (row-vector, ``@ R`` not ``@ R.T``), then translate — including its normalization of
    the stored quaternion, which ``quaternion_to_matrix`` does not do internally.

    ``centroid_local`` is a CPU tensor, converted once per object by the caller.  The whole
    composition runs on CPU on 3- and 4-vectors: this is called per object per frame, and
    the alternative is ~35 GPU kernel launches to produce three floats we immediately
    copy back.
    """
    from .mesh_rendering import apply_pose_p3d

    t = pose["translation"].detach().float().cpu().reshape(1, 3)
    if centroid_local is None:
        return t.numpy().flatten()
    s = pose["scale"].detach().float().cpu().reshape(1, -1)
    rot = pose["rotation"].detach().float().cpu().reshape(1, 4)
    R = quaternion_to_matrix(rot / rot.norm(dim=-1, keepdim=True)).squeeze(0)
    return apply_pose_p3d(centroid_local.reshape(1, 3), R, t, s).numpy().flatten()


def overview_look_at(
    target: np.ndarray,
    ref_forward: np.ndarray,
    distance: float,
    elevation_deg: float = 30.0,
    azimuth_deg: float = 0.0,
) -> np.ndarray:
    """A c2w looking at ``target`` from ``distance`` behind ``ref_forward``, elevated.

    The viewpoint half of an overview camera, with no opinion about *what* is being
    framed -- :func:`compute_scene_overview_c2w` aims it at the mean of the camera and
    object positions, while the ``world_space`` render of
    ``genia.core.utils.render_final_results`` aims it at the foreground's box
    centre and fits its focal separately.  Both want the same placement rule, so it
    lives here once.

    The camera sits on the far side of the scene from ``ref_forward`` -- i.e. behind
    the input cameras, looking the way they look -- so the objects are seen from the
    side that was actually observed, never from behind.  Elevation and azimuth are both
    about the R3 world up (``-Y``): elevation raises the camera off the horizontal plane
    through ``target``, azimuth swings it around that plane, and the two defaults of
    ``(elevation_deg, 0.0)`` reproduce the straight-behind view.

    Parameters
    ----------
    target : np.ndarray (3,)
        World point the camera looks at.
    ref_forward : np.ndarray (3,)
        A reference camera's forward axis in R3 (its ``c2w[:3, 2]``); need not be
        normalized.
    distance : float
        How far the camera sits from ``target``.
    elevation_deg : float
        Degrees raised above the horizontal plane through ``target``.
    azimuth_deg : float
        Degrees swung around the world up from straight behind ``ref_forward``.
        Positive and negative move the camera to opposite sides of the scene; 0
        keeps the straight-behind placement.

    Returns
    -------
    np.ndarray (4, 4) float32
    """
    target = np.asarray(target, dtype=np.float64).reshape(3)
    ref_forward = np.asarray(ref_forward, dtype=np.float64).reshape(3)
    ref_forward = ref_forward / max(np.linalg.norm(ref_forward), 1e-8)

    # Offset from the target along the *negative* forward direction (behind the
    # cameras), then shifted upward (-Y in R3).
    elev = np.radians(elevation_deg)
    back_dir = -ref_forward  # direction from scene toward the cameras
    back_dir[1] = 0.0  # project onto horizontal XZ plane
    norm = np.linalg.norm(back_dir)
    if norm > 1e-6:
        back_dir /= norm
    else:
        back_dir = np.array([0.0, 0.0, -1.0])
    if azimuth_deg:
        # Rodrigues about the world up.  `back_dir` is horizontal by construction, so
        # it is perpendicular to that axis and the axis-parallel term vanishes.
        azim = np.radians(azimuth_deg)
        back_dir = (back_dir * np.cos(azim)
                    + np.cross(np.array([0.0, -1.0, 0.0]), back_dir) * np.sin(azim))
    cam_pos = target + distance * (
        np.cos(elev) * back_dir + np.sin(elev) * np.array([0.0, -1.0, 0.0])
    )

    # Look-at matrix (R3: Z-forward, Y-down, X-right)
    forward = target - cam_pos
    forward /= max(np.linalg.norm(forward), 1e-8)
    world_up = np.array([0.0, -1.0, 0.0])  # -Y is up in R3
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)  # camera Y-axis (points down)

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right.astype(np.float32)
    c2w[:3, 1] = down.astype(np.float32)
    c2w[:3, 2] = forward.astype(np.float32)
    c2w[:3, 3] = cam_pos.astype(np.float32)
    return c2w


def compute_scene_overview_c2w(
    sequence: "Any",
    frame_indices: "List[int]",
    tokens_by_object: "Dict",
    elevation_deg: float = 30.0,
    distance_scale: float = 2.5,
    object_local_centroids: "Optional[Dict[int, np.ndarray]]" = None,
) -> "Tuple[np.ndarray, np.ndarray, int, int]":
    """Compute a c2w + K for an overview camera that sees the entire world-space scene.

    Uses the first input camera's forward direction to determine the
    overview orientation — the overview camera looks at the scene from
    the same side as the input cameras, pulled back and elevated.

    Parameters
    ----------
    sequence : Sequence
    frame_indices : list of int
    tokens_by_object : dict
        Per-object per-frame poses (used to find object centroids).
    elevation_deg : float
        Camera elevation above the horizontal plane (degrees).
    distance_scale : float
        Multiplier on scene extent to set camera distance.
    object_local_centroids : dict, optional
        ``{obj_idx: (3,)}`` centroid of each object's canonical asset in its own local
        frame.  Omitted (or missing an object) means "assume origin-centred".  Supply it
        whenever the canonical may NOT be at its local origin — see
        :func:`_object_camera_position`.

    Returns
    -------
    c2w : np.ndarray (4, 4)
    K : np.ndarray (3, 3)
    W : int
    H : int
    """
    # Collect all world-space camera positions
    cam_positions = np.array([sequence[fi].c2w[:3, 3] for fi in frame_indices])

    # Collect all object positions (camera-space) transformed to world.  Note: this lift
    # omits the P3D->R3 flip; the overview framing is calibrated with it omitted, so
    # adding it would move the overview camera.
    obj_positions = []
    for obj_idx, tokens_list in tokens_by_object.items():
        centroid_local = (object_local_centroids or {}).get(obj_idx)
        if centroid_local is not None:      # once per object, not once per frame
            centroid_local = torch.as_tensor(centroid_local, dtype=torch.float32)
        for fid, di in tokens_list:
            if fid in frame_indices and "translation" in di:
                t_cam = _object_camera_position(di, centroid_local)
                c2w_frame = sequence[fid].c2w
                R, t = c2w_frame[:3, :3], c2w_frame[:3, 3]
                t_world = t_cam @ R.T + t
                obj_positions.append(t_world)

    all_positions = np.concatenate([
        cam_positions,
        np.array(obj_positions) if obj_positions else np.zeros((0, 3)),
    ], axis=0)

    center = all_positions.mean(axis=0)
    extent = max(np.linalg.norm(all_positions - center, axis=1).max(), 0.1)
    distance = extent * distance_scale

    # Use the first camera's forward direction to determine the overview
    # orientation.  This ensures we see the scene from the same side as
    # the input cameras (not from behind the objects).
    first_c2w = sequence[frame_indices[0]].c2w

    c2w = overview_look_at(center, first_c2w[:3, 2], distance, elevation_deg)

    # Simple pinhole intrinsics (use original image size, 60° FOV)
    W, H = sequence.W, sequence.H
    fov = np.radians(60.0)
    f = 0.5 * max(W, H) / np.tan(fov / 2)
    K = np.array([
        [f, 0, W / 2],
        [0, f, H / 2],
        [0, 0, 1],
    ], dtype=np.float32)

    return c2w, K, W, H


def save_keyframes_video(state, sequence, cfg, suffix="", output_dir=None,
                         background=False, color_mode="default",
                         render_space="camera",
                         voxel_color_mode="xyz"):
    """Render Gaussians at keyframe poses and save as MP4.

    When canonical Gaussians exist, renders them with per-frame poses.
    When per-frame Gaussians exist, renders each keyframe with its own
    decoded Gaussian. When both exist, saves separate videos.

    Parameters
    ----------
    output_dir : str or None
        Directory to write videos to.  Falls back to ``cfg.output.output_dir``.
    background : bool
        If True, add background Gaussians from the pointmap.
    color_mode : str
        ``"default"`` uses original SH colors; ``"xyz"`` colors each
        Gaussian by its local-space XYZ coordinate.
    render_space : str
        ``"camera"`` renders in camera space with identity c2w.
        ``"world"`` renders from an automatically computed overview camera.
    voxel_color_mode : str
        Coloring for voxel mesh keyframes: ``"xyz"`` (default),
        ``"shape_pca"`` (PCA of shape tokens), or ``"slat_pca"``
        (PCA of SLAT features).
    """
    if not cfg.output.save_renders:
        return
    # Per-frame-only runs leave the canonical stores empty, so has_canonical /
    # has_voxel_coords fall to False and only per-frame keyframes are rendered.
    canon_gs = state.canonical_gaussians
    canon_slats = state.canonical_slats
    has_canonical = bool(canon_gs)
    has_voxel_coords = bool(state.canonical_shape_coords) or bool(canon_slats)
    has_perframe = bool(state.perframe_gaussians)
    if not has_canonical and not has_perframe and not has_voxel_coords:
        return

    from .interpolation import _compute_keyframe_repeats
    from .viz_io import write_per_view

    out = output_dir or cfg.output.output_dir
    os.makedirs(out, exist_ok=True)

    frame_indices = sequence.frame_indices
    total_frames = len(sequence.paths['image_names'])
    space_postfix = "_camera_space" if render_space == "camera" else "_world_space"
    tag = f"_{suffix}{space_postfix}" if suffix else space_postfix
    # World-space renders use a single overview camera, so content depends
    # only on time -- not view. Dedupe frame_indices by .frame value so we
    # don't produce V identical renders per time. (Camera-space keeps the
    # full per-view list because each view has its own camera.)
    if render_space == "world":
        seen_frames = set()
        render_fis = []
        for fi in frame_indices:
            f_val = fi.frame if hasattr(fi, "frame") else int(fi)
            if f_val not in seen_frames:
                seen_frames.add(f_val)
                render_fis.append(fi)
    else:
        render_fis = list(frame_indices)
    # Temporal repeats only mean something for dynamic sequences. MV-static
    # camera-space runs have all keyframes at frame=0, so the helper would
    # return all-zero counts. Repeats are also single-view only -- under the
    # MV-dynamic uniform-keyframes invariant, per-view repeats would be
    # identical, and per-view MP4s are not stretched.
    n_views = len({(fi.view if hasattr(fi, "view") else 0) for fi in render_fis})
    repeats = (
        _compute_keyframe_repeats(render_fis, total_frames)
        if sequence.is_dynamic and n_views == 1 and total_frames > 0
        else None
    )

    bg_color = torch.ones(3) if cfg.pipeline.white_background else None

    def _emit(label: str, rendered_frames):
        """Group rendered frames by view and write per-view PNG (T==1) or MP4.
        For world-space, render_fis is deduped to a single view, so this
        falls into the flat-layout branch of write_per_view."""
        if not rendered_frames:
            return
        frames_by_view: Dict[int, List["Image.Image"]] = {}
        for fi, f in zip(render_fis, rendered_frames):
            v = fi.view if hasattr(fi, "view") else 0
            frames_by_view.setdefault(v, []).append(f)
        t_per_view = max(len(fs) for fs in frames_by_view.values())
        stem_word = "keyframe" if t_per_view == 1 else "keyframes"
        stem = f"{cfg.dataset.scene_name}_{stem_word}{tag}{label}"
        write_per_view(frames_by_view, out, stem, frame_repeats=repeats)

    # Per-canonical-mesh-vertex deformation forwarding.  When loaded
    # (any per-frame GT_SHAPES_INVERSION mode), ``render_keyframes`` /
    # ``render_voxel_keyframes`` warp the canonical asset per frame before
    # applying Stage-1 pose; otherwise they fall through to the
    # static-canonical path.
    _gauss_warp_kwargs = _deformation_warp_kwargs(state, cfg)
    _warp_kwargs = dict(_gauss_warp_kwargs)

    if has_canonical:
        print(f"  Rendering {len(render_fis)} canonical keyframes...")
        frames = render_keyframes(
            canon_gs, state.tokens_by_object,
            sequence, per_frame=False, bg_color=bg_color,
            overlay_axes=True, background=background,
            color_mode=color_mode, render_space=render_space,
            frame_indices=render_fis,
            **_gauss_warp_kwargs,
        )
        _emit("_canonical", frames)

    # Voxel mesh rendering with configurable coloring.
    #
    # Coords + colors must come from the same SLAT: ``compute_voxel_colors``
    # in slat_pca mode reads ``slat.feats`` directly, so its size MUST match
    # the coords passed to ``_build_voxel_surface_mesh``.  Source priority:
    #   1. Canonical SLAT (from ``canon_slats``).
    #   2. First per-frame SLAT (when no canonical SLAT was built — e.g.
    #      a run that exited before APPEARANCE_INIT decoded a canonical).
    #   3. GT-derived ``state.canonical_shape_coords`` with no SLAT —
    #      colour mode is downgraded to ``xyz`` since no features exist.
    #
    # The GT ``state.canonical_shape_coords`` must not be paired with SLAT
    # colours: the two occupancies can differ in size, which would break the
    # boolean mask in ``_build_voxel_surface_mesh``.
    voxel_coords = {}
    voxel_slats: Dict[int, Any] = {}
    for obj_idx, slat in canon_slats.items():
        voxel_coords[obj_idx] = (
            slat.coords.detach().cpu()[:, 1:].float().numpy()
        )
        voxel_slats[obj_idx] = slat
    for obj_idx, tokens in state.tokens_by_object.items():
        if obj_idx in voxel_coords:
            continue
        for _fid, _di in tokens:
            pf_slat = _di.get("decoder_input_slat")
            if pf_slat is not None:
                voxel_coords[obj_idx] = (
                    pf_slat.coords.detach().cpu()[:, 1:].float().numpy()
                )
                voxel_slats[obj_idx] = pf_slat
                break
    for obj_idx, gt_coords in state.canonical_shape_coords.items():
        if obj_idx in voxel_coords:
            continue
        voxel_coords[obj_idx] = (
            gt_coords.detach().cpu().numpy()
            if hasattr(gt_coords, "cpu") else np.asarray(gt_coords)
        )
        # voxel_slats[obj_idx] intentionally absent → xyz coloring below.
    # This is the canonical-voxels path (rendered per_frame=False, emitted
    # as ``_canonical_voxels``); skip it entirely under per-frame-only mode
    # — the dedicated per-frame voxel block below still renders.
    if voxel_coords and has_canonical:
        from .visualization import compute_voxel_colors
        voxel_colors_by_obj = {}
        for obj_idx, vc in voxel_coords.items():
            slat_obj = voxel_slats.get(obj_idx)
            canon_raw = getattr(state, "canonical_raw_modalities", {}).get(obj_idx, {})
            shape_lat = canon_raw.get("shape")
            # Downgrade colour mode when the requested data isn't
            # available for this object — same pattern as the per-frame
            # block below.  ``xyz`` is always safe; the others need a
            # SLAT or shape latent.
            mode_i = voxel_color_mode
            if mode_i == "slat_pca" and slat_obj is None:
                mode_i = "shape_pca" if shape_lat is not None else "xyz"
            elif mode_i == "shape_pca" and shape_lat is None:
                mode_i = "xyz"
            voxel_colors_by_obj[obj_idx] = compute_voxel_colors(
                mode_i, vc, grid_size=64,
                slat=slat_obj, shape_latent=shape_lat,
            )
        print(f"  Rendering {len(render_fis)} voxel-mesh keyframes...")
        frames = render_voxel_keyframes(
            voxel_coords, state.tokens_by_object, sequence,
            render_space=render_space,
            voxel_colors_by_object=voxel_colors_by_obj,
            frame_indices=render_fis,
            **_warp_kwargs,
        )
        _emit("_canonical_voxels", frames)

    # Per-frame voxel mesh rendering: each keyframe uses its own SLAT's
    # voxel coords (and matching colors), posed by that frame's decoded pose.
    # Falls back to decoder_input["perframe_shape_coords"] when SLAT isn't
    # built yet (post shape_and_poses_init, pre appearance_init).
    pf_voxel_coords = {}
    pf_voxel_colors = {}
    from .visualization import compute_voxel_colors as _compute_voxel_colors
    for obj_idx, tokens in state.tokens_by_object.items():
        for fid, di in tokens:
            pf_slat = di.get("decoder_input_slat")
            if pf_slat is not None:
                vc = pf_slat.coords.detach().cpu()[:, 1:].float().numpy()
                pf_raw = getattr(state, "perframe_raw_modalities", {}).get(obj_idx, {}).get(fid, {})
                shape_lat = pf_raw.get("raw_ss_modalities", {}).get("shape")
            else:
                pf_coords = di.get("perframe_shape_coords")
                if pf_coords is None:
                    continue
                vc = (pf_coords[:, 1:].cpu().numpy().astype(float)
                      if hasattr(pf_coords, "cpu") else pf_coords)
                shape_lat = di.get("raw_ss_modalities", {}).get("shape")
            # Downgrade color mode when the requested data isn't available
            # for this frame (per-frame SLAT may be absent in actionmesh
            # mode where APPEARANCE_INIT writes only a canonical SLAT).
            mode_i = voxel_color_mode
            if mode_i == "slat_pca" and pf_slat is None:
                mode_i = "shape_pca" if shape_lat is not None else "xyz"
            elif mode_i == "shape_pca" and shape_lat is None:
                mode_i = "xyz"
            pf_voxel_coords.setdefault(obj_idx, {})[fid] = vc
            pf_voxel_colors.setdefault(obj_idx, {})[fid] = _compute_voxel_colors(
                mode_i, vc, grid_size=64,
                slat=pf_slat, shape_latent=shape_lat,
            )
    if pf_voxel_coords:
        print(f"  Rendering {len(render_fis)} per-frame voxel-mesh keyframes...")
        frames = render_voxel_keyframes(
            pf_voxel_coords, state.tokens_by_object, sequence,
            per_frame=True,
            render_space=render_space,
            voxel_colors_by_object=pf_voxel_colors,
            frame_indices=render_fis,
        )
        _emit("_perframe_voxels", frames)

    if has_perframe:
        print(f"  Rendering {len(render_fis)} per-frame keyframes...")
        frames = render_keyframes(
            state.perframe_gaussians, state.tokens_by_object,
            sequence, per_frame=True, bg_color=bg_color,
            overlay_axes=True, background=background,
            color_mode=color_mode, render_space=render_space,
            frame_indices=render_fis,
        )
        _emit("_perframe", frames)


__all__ = [
    "process_frame_with_canonical_object",
    "evaluate_with_canonical_objects",
    "print_evaluation_summary",
    "capture_all_renders",
    "capture_pose_snapshot",
    "evaluate_block",
    "capture_before",
    "render_keyframes",
    "save_keyframes_video",
]
