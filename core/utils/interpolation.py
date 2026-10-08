"""
Pose interpolation and full-sequence rendering utilities.

This module provides functions to interpolate poses between keyframes
(selected by the dataset's frame stride) and render the full temporal
sequence using canonical Gaussian objects.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp


def _save_frames_as_video(
    frames: List["Image.Image"],
    output_path: str,
    duration: int = 100,
    frame_repeats: Optional[List[int]] = None,
) -> None:
    """Save a list of PIL Images as an H.264 MP4 video using moviepy.

    Parameters
    ----------
    frames : list of PIL.Image.Image
        RGB frames to encode.
    output_path : str
        Destination path (should end in .mp4).
    duration : int
        Per-frame duration in milliseconds (used to derive fps).
    frame_repeats : list of int, optional
        How many times to write each frame. If None, each frame is written
        once. Use this to hold keyframes for the duration they represent in
        the full sequence so that keyframe and full-sequence videos have
        the same total duration.
    """
    # Single frame: save as PNG instead of creating a 1-frame video
    if len(frames) == 1:
        png_path = os.path.splitext(output_path)[0] + ".png"
        frames[0].save(png_path)
        print(f"    Saved single frame: {png_path}")
        return

    try:
        from moviepy import ImageSequenceClip  # moviepy >= 2.0
    except ImportError:
        from moviepy.editor import ImageSequenceClip  # moviepy 1.x

    fps = max(1, 1000 // duration)

    print("\n  Creating MP4 video from frames...")

    # Build numpy array sequence, repeating frames as needed
    frame_arrays = []
    for i, frame in enumerate(frames):
        arr = np.array(frame)  # (H, W, 3) uint8 RGB
        n = frame_repeats[i] if frame_repeats else 1
        for _ in range(n):
            frame_arrays.append(arr)

    clip = ImageSequenceClip(frame_arrays, fps=fps)
    clip.write_videofile(output_path, codec="libx264", fps=fps, logger=None)

    print(f"    Created video: {output_path} ({len(frames)} frames)")


def _compute_keyframe_repeats(
    frame_indices: List[int],
    total_frames: int,
) -> List[int]:
    """Compute how many times to repeat each keyframe to match full-sequence duration.

    Each keyframe is held for the number of frames from itself to the next
    keyframe (or to the end of the sequence for the last keyframe).

    Parameters
    ----------
    frame_indices : list of int
        Sorted keyframe indices.
    total_frames : int
        Total number of frames in the full sequence.

    Returns
    -------
    list of int
        Repeat count per keyframe.
    """
    def _val(k):
        return k.frame if hasattr(k, "frame") else int(k)

    repeats = []
    for i, fi in enumerate(frame_indices):
        if i + 1 < len(frame_indices):
            repeats.append(_val(frame_indices[i + 1]) - _val(fi))
        else:
            # Last keyframe: hold for the same duration as the previous gap
            # so it's equally visible in the video. Fall back to
            # total_frames - fi when there's only one keyframe.
            if repeats:
                repeats.append(repeats[-1])
            else:
                repeats.append(max(1, total_frames - _val(fi)))
    return repeats


def interpolate_c2w(
    keyframe_c2w: Dict[int, np.ndarray],
    all_frame_indices: List[int],
) -> Dict[int, np.ndarray]:
    """Interpolate camera-to-world transforms between keyframes.

    Rotation is interpolated via SLERP, translation via linear interpolation.
    Frames outside the keyframe range are clamped to the nearest keyframe.

    Parameters
    ----------
    keyframe_c2w : dict
        ``{frame_idx: np.ndarray(4,4)}`` c2w transforms at keyframe positions.
    all_frame_indices : list of int
        Every frame index that should receive a c2w.

    Returns
    -------
    dict
        ``{frame_idx: np.ndarray(4,4)}`` interpolated c2w for all frames.
    """
    if not keyframe_c2w:
        return {fi: np.eye(4, dtype=np.float32) for fi in all_frame_indices}

    # Keys may be int or FrameKey (NamedTuple).
    def _frame_value(k):
        return k.frame if hasattr(k, "frame") else int(k)

    def _view_value(k):
        return k.view if hasattr(k, "view") else 0

    # Bucket keyframes and outputs by view so each per-view group runs through
    # time-axis interpolation independently. Mirrors `interpolate_poses` —
    # required for MV-static (e.g. gso_mv) where every keyframe sits at
    # frame=0 but on a different view, so a single global Slerp would see
    # duplicate times and crash.
    keys_by_view: Dict[int, List] = {}
    for k in keyframe_c2w.keys():
        keys_by_view.setdefault(_view_value(k), []).append(k)

    outputs_by_view: Dict[int, List] = {}
    for fi in all_frame_indices:
        outputs_by_view.setdefault(_view_value(fi), []).append(fi)

    result: Dict[int, np.ndarray] = {}

    for view_id, output_keys in outputs_by_view.items():
        if view_id not in keys_by_view:
            raise KeyError(
                f"interpolate_c2w: requested output view {view_id} has no "
                f"keyframes (available views: {sorted(keys_by_view.keys())}). "
                f"Caller contract: every output view must be represented in "
                f"keyframe_c2w."
            )
        sorted_keys = sorted(keys_by_view[view_id], key=_frame_value)

        if len(sorted_keys) == 1:
            c2w_single = keyframe_c2w[sorted_keys[0]]
            for fi in output_keys:
                result[fi] = c2w_single.copy()
            continue

        key_rots = Rotation.from_matrix(
            np.stack([keyframe_c2w[k][:3, :3] for k in sorted_keys])
        )
        key_trans = np.stack([keyframe_c2w[k][:3, 3] for k in sorted_keys])
        key_times = np.array([_frame_value(k) for k in sorted_keys], dtype=np.float64)

        slerp = Slerp(key_times, key_rots)
        t_min, t_max = key_times[0], key_times[-1]

        for fi in output_keys:
            c2w = np.eye(4, dtype=np.float32)
            t_val = float(_frame_value(fi))
            t_clamped = float(np.clip(t_val, t_min, t_max))
            c2w[:3, :3] = slerp([t_clamped]).as_matrix()[0].astype(np.float32)
            for axis in range(3):
                c2w[axis, 3] = np.interp(t_clamped, key_times, key_trans[:, axis])
            result[fi] = c2w

    return result


def interpolate_K(
    keyframe_K: Dict[int, np.ndarray],
    all_frame_indices: List[int],
) -> Dict[int, np.ndarray]:
    """Interpolate camera intrinsics between keyframes.

    All elements of the 3x3 K matrix are linearly interpolated.
    Frames outside the keyframe range are clamped to the nearest keyframe.

    Parameters
    ----------
    keyframe_K : dict
        ``{frame_idx: np.ndarray(3,3)}`` intrinsics at keyframe positions.
    all_frame_indices : list of int
        Every frame index that should receive a K matrix.

    Returns
    -------
    dict
        ``{frame_idx: np.ndarray(3,3)}`` interpolated intrinsics for all frames.
    """
    if not keyframe_K:
        return {fi: np.eye(3, dtype=np.float32) for fi in all_frame_indices}

    # Keys may be int or FrameKey (NamedTuple).
    def _frame_value(k):
        return k.frame if hasattr(k, "frame") else int(k)

    def _view_value(k):
        return k.view if hasattr(k, "view") else 0

    # Bucket by view to mirror interpolate_c2w / interpolate_poses — needed
    # for MV-static where keyframes share a frame index but vary by view.
    keys_by_view: Dict[int, List] = {}
    for k in keyframe_K.keys():
        keys_by_view.setdefault(_view_value(k), []).append(k)

    outputs_by_view: Dict[int, List] = {}
    for fi in all_frame_indices:
        outputs_by_view.setdefault(_view_value(fi), []).append(fi)

    result: Dict[int, np.ndarray] = {}

    for view_id, output_keys in outputs_by_view.items():
        if view_id not in keys_by_view:
            raise KeyError(
                f"interpolate_K: requested output view {view_id} has no "
                f"keyframes (available views: {sorted(keys_by_view.keys())}). "
                f"Caller contract: every output view must be represented in "
                f"keyframe_K."
            )
        sorted_keys = sorted(keys_by_view[view_id], key=_frame_value)

        if len(sorted_keys) == 1:
            K_single = keyframe_K[sorted_keys[0]]
            for fi in output_keys:
                result[fi] = K_single.copy()
            continue

        key_Ks = np.stack([keyframe_K[k] for k in sorted_keys])  # (N, 3, 3)
        key_times = np.array([_frame_value(k) for k in sorted_keys], dtype=np.float64)
        t_min, t_max = key_times[0], key_times[-1]

        for fi in output_keys:
            t_val = float(_frame_value(fi))
            t_clamped = float(np.clip(t_val, t_min, t_max))
            K = np.zeros((3, 3), dtype=np.float32)
            for r in range(3):
                for c in range(3):
                    K[r, c] = np.interp(t_clamped, key_times, key_Ks[:, r, c])
            result[fi] = K

    return result


def interpolate_poses(
    tokens_by_object: Dict[int, List[Tuple[int, Dict[str, Any]]]],
    all_frame_indices: List[int],
) -> Dict[int, Dict[int, Dict[str, torch.Tensor]]]:
    """
    Interpolate rotation/translation/scale for non-keyframes.

    Keyframe poses are extracted from ``tokens_by_object``.  Intermediate
    frames receive SLERP-interpolated rotations and linearly-interpolated
    translations and scales.  Frames outside the keyframe range are clamped
    to the nearest keyframe pose.

    Parameters
    ----------
    tokens_by_object : dict
        ``{obj_idx: [(FrameKey, decoder_input), ...]}`` with poses stored
        in each ``decoder_input`` dict (keys: ``rotation``, ``translation``,
        ``scale``). Tuple-element-0 may also be bare int (back-compat).
    all_frame_indices : list of int
        Every (time-axis) frame index that should receive a pose
        (e.g. ``range(N)``). Time interpolation operates on the time axis
        only; ``FrameKey.frame`` extracts that coordinate.

    Returns
    -------
    dict
        ``{obj_idx: {frame_idx: {"rotation": (4,), "translation": (3,), "scale": (3,)}}}``
        Poses in PyTorch3D wxyz quaternion convention. Output dict keyed by
        the input ``all_frame_indices`` element type (typically int).
    """
    def _frame_value(k):
        """Extract the time-axis coordinate from a FrameKey or bare int."""
        return k.frame if hasattr(k, "frame") else int(k)

    def _view_value(k):
        """Extract the view-axis coordinate (0 for bare int)."""
        return k.view if hasattr(k, "view") else 0

    # Bucket output frame keys by view so each per-view group runs through
    # the time-axis interpolator independently. For mono-dynamic / mono-static
    # there's one bucket (view=0). For MV-static every bucket has a single keyframe
    # → clamp to that view's keyframe pose. For MV-dynamic each bucket has
    # T keyframes → per-view SLERP / lerp.
    output_keys_by_view: Dict[int, List] = {}
    for fk in all_frame_indices:
        output_keys_by_view.setdefault(_view_value(fk), []).append(fk)

    result: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}

    for obj_idx, tokens_list in tokens_by_object.items():
        result[obj_idx] = {}

        # Bucket the object's keyframes by view too. For mono pipelines this
        # is a single bucket; for MV everything is partitioned cleanly.
        tokens_by_view: Dict[int, List] = {}
        for tup in tokens_list:
            tokens_by_view.setdefault(_view_value(tup[0]), []).append(tup)

        for view_id, output_keys in output_keys_by_view.items():
            view_tokens = tokens_by_view.get(view_id, [])
            if not view_tokens:
                # Object has no keyframe in this view (e.g. instance not
                # observed); skip — caller checks ``fi in interpolated_poses[obj]``.
                continue

            tokens_sorted = sorted(view_tokens, key=lambda t: _frame_value(t[0]))
            keyframe_idxs = [_frame_value(t[0]) for t in tokens_sorted]

            # Extract poses as numpy arrays
            rotations_wxyz = []
            translations = []
            scales = []
            for _, di in tokens_sorted:
                rot = di["rotation"].detach().cpu().float()
                if rot.dim() == 2:
                    rot = rot.squeeze(0)
                rotations_wxyz.append(rot.numpy())

                trans = di["translation"].detach().cpu().float()
                if trans.dim() == 2:
                    trans = trans.squeeze(0)
                translations.append(trans.numpy())

                sc = di["scale"].detach().cpu().float()
                if sc.dim() == 2:
                    sc = sc.squeeze(0)
                if sc.shape[0] == 1:
                    sc = sc.expand(3)
                scales.append(sc.numpy())

            # Per-frame DC offsets (optional, from finetuning)
            dc_offset_list = []
            for _, di in tokens_sorted:
                if "dc_offset" in di:
                    dc_offset_list.append(di["dc_offset"].detach().cpu().float())
            has_dc = len(dc_offset_list) == len(tokens_sorted) and len(dc_offset_list) > 0

            # Per-frame SH coefficients (optional, from finetuning)
            sh_rest_list = []
            for _, di in tokens_sorted:
                if "sh_rest" in di:
                    sh_rest_list.append(di["sh_rest"].detach().cpu().float())
            has_sh = len(sh_rest_list) == len(tokens_sorted) and len(sh_rest_list) > 0

            rotations_wxyz = np.array(rotations_wxyz)  # (K, 4)
            translations = np.array(translations)        # (K, 3)
            scales = np.array(scales)                     # (K, 3)

            # Align quaternion signs: ensure consecutive keyframes use the shorter SLERP path
            for i in range(1, len(rotations_wxyz)):
                if np.dot(rotations_wxyz[i], rotations_wxyz[i - 1]) < 0:
                    rotations_wxyz[i] = -rotations_wxyz[i]

            # Convert wxyz → xyzw for scipy
            rotations_xyzw = rotations_wxyz[:, [1, 2, 3, 0]]

            # Build SLERP interpolator (requires ≥2 keyframes AND strictly
            # monotonic times). Per-view bucketing already isolates each
            # bucket's keyframes to one time series; for MV-static a bucket
            # has exactly one keyframe → fall through to clamp logic below.
            kf = np.array(keyframe_idxs, dtype=np.float64)
            if len(kf) >= 2 and np.all(np.diff(kf) > 0):
                slerp = Slerp(kf, Rotation.from_quat(rotations_xyzw))
            else:
                slerp = None

            for fi in output_keys:
                # Extract the time-axis float for all numeric comparisons /
                # interpolation. FrameKey is a tuple so direct fi <= kf[0]
                # would TypeError; extract once and use the int variant.
                fi_t = float(_frame_value(fi))
                dc_interp = None
                sh_interp = None
                if fi_t <= kf[0]:
                    # Before or at first keyframe: clamp
                    rot_np = rotations_wxyz[0]
                    trans_np = translations[0]
                    scale_np = scales[0]
                    if has_dc:
                        dc_interp = dc_offset_list[0]
                    if has_sh:
                        sh_interp = sh_rest_list[0]
                elif fi_t >= kf[-1]:
                    # After or at last keyframe: clamp
                    rot_np = rotations_wxyz[-1]
                    trans_np = translations[-1]
                    scale_np = scales[-1]
                    if has_dc:
                        dc_interp = dc_offset_list[-1]
                    if has_sh:
                        sh_interp = sh_rest_list[-1]
                elif int(fi_t) in keyframe_idxs:
                    # Exact keyframe
                    ki = keyframe_idxs.index(int(fi_t))
                    rot_np = rotations_wxyz[ki]
                    trans_np = translations[ki]
                    scale_np = scales[ki]
                    if has_dc:
                        dc_interp = dc_offset_list[ki]
                    if has_sh:
                        sh_interp = sh_rest_list[ki]
                else:
                    # Interpolate
                    rot_xyzw = slerp(fi_t).as_quat()  # (4,) xyzw
                    rot_np = rot_xyzw[[3, 0, 1, 2]]         # → wxyz
                    rot_np = rot_np / np.linalg.norm(rot_np)  # re-normalize

                    trans_np = np.array([
                        np.interp(fi_t, kf, translations[:, c]) for c in range(3)
                    ])
                    scale_np = np.array([
                        np.interp(fi_t, kf, scales[:, c]) for c in range(3)
                    ])
                    # Linearly interpolate DC offsets between adjacent keyframes
                    if has_dc:
                        idx_right = np.searchsorted(kf, fi_t)
                        idx_left = idx_right - 1
                        t = (fi_t - kf[idx_left]) / (kf[idx_right] - kf[idx_left])
                        dc_interp = (1 - t) * dc_offset_list[idx_left] + t * dc_offset_list[idx_right]

                    # Linearly interpolate SH coefficients between adjacent keyframes
                    if has_sh:
                        idx_right = np.searchsorted(kf, fi_t)
                        idx_left = idx_right - 1
                        t = (fi_t - kf[idx_left]) / (kf[idx_right] - kf[idx_left])
                        sh_interp = (1 - t) * sh_rest_list[idx_left] + t * sh_rest_list[idx_right]

                pose_dict = {
                    "rotation": torch.tensor(rot_np, dtype=torch.float32),
                    "translation": torch.tensor(trans_np, dtype=torch.float32),
                    "scale": torch.tensor(scale_np, dtype=torch.float32),
                }
                if dc_interp is not None:
                    pose_dict["dc_offset"] = dc_interp if isinstance(dc_interp, torch.Tensor) else torch.tensor(dc_interp, dtype=torch.float32)
                if sh_interp is not None:
                    pose_dict["sh_rest"] = sh_interp if isinstance(sh_interp, torch.Tensor) else torch.tensor(sh_interp, dtype=torch.float32)

                # Output keyed by the full FrameKey (frame, view) so the
                # caller can index by either the original FrameKey or a
                # bare int (which back-compat-coerces via dict-ish lookup).
                result[obj_idx][fi] = pose_dict

    return result


def _sample_canonical_points(
    canonical_gs: Any,
    num_points: int,
) -> torch.Tensor:
    """
    Sample representative points from the canonical Gaussian means.

    Uses farthest-point sampling to select well-spread points across the
    object surface.  Always includes the centroid as the first point.

    Parameters
    ----------
    canonical_gs : Gaussian
        Canonical Gaussian object.
    num_points : int
        Number of points to sample (including centroid).

    Returns
    -------
    torch.Tensor
        Selected canonical positions, shape ``(num_points, 3)`` on the same device.
    """
    xyz = canonical_gs.get_xyz  # (N, 3)
    N = xyz.shape[0]

    if N <= num_points:
        return xyz

    # Start with centroid
    centroid = xyz.mean(dim=0, keepdim=True)  # (1, 3)
    selected = [centroid]

    # Distances from each point to the nearest selected point
    dists = torch.cdist(xyz.unsqueeze(0), centroid.unsqueeze(0)).squeeze(0).squeeze(-1)  # (N,)

    for _ in range(num_points - 1):
        # Pick farthest point from current selection
        idx = torch.argmax(dists).item()
        selected.append(xyz[idx:idx + 1])
        # Update distances
        new_dists = torch.cdist(xyz.unsqueeze(0), xyz[idx:idx + 1].unsqueeze(0)).squeeze(0).squeeze(-1)
        dists = torch.minimum(dists, new_dists)

    return torch.cat(selected, dim=0)  # (num_points, 3)


from .quaternion_ops import quaternion_to_matrix as _quat_to_matrix


def compute_object_tracks(
    canonical_gaussians: Dict[int, Any],
    interpolated_poses: Dict[int, Dict[int, Dict[str, torch.Tensor]]],
    all_frame_indices: List[int],
    num_points: int = 16,
    voxel_coords_by_object: Optional[Dict[int, np.ndarray]] = None,
    c2w_per_frame: Optional[Dict[int, np.ndarray]] = None,
    *,
    canonical_mesh_verts: "Optional[Dict[int, torch.Tensor]]" = None,
    per_frame_mesh_verts: "Optional[Dict[int, Dict[int, torch.Tensor]]]" = None,
    per_frame_mesh_rotations: "Optional[Dict[int, Dict[int, torch.Tensor]]]" = None,
    canonical_mesh_faces: "Optional[Dict[int, torch.Tensor]]" = None,
    warp_knn_k: int = 4,
    warp_knn_eps: float = 1.0e-8,
    warp_knn_chunk_size: int = 131072,
) -> Dict[int, np.ndarray]:
    """
    Compute per-object 3D point tracks over time.

    Samples ``num_points`` representative points from each canonical Gaussian
    (via farthest-point sampling), then transforms them through each frame's
    interpolated pose and converts to R3 convention.

    Falls back to voxel grid coordinates when canonical Gaussians are not
    available for an object but ``voxel_coords_by_object`` is provided.

    Anchor sources, in priority order: canonical Gaussians →
    ``voxel_coords_by_object`` → ``canonical_mesh_verts``.  Their points are
    pushed through the per-frame warp + Sim(3) chain below.

    Parameters
    ----------
    canonical_gaussians : dict
        ``{obj_idx: Gaussian}`` canonical Gaussian objects.
    interpolated_poses : dict
        Output of :func:`interpolate_poses`.
    all_frame_indices : list of int
        Frame indices to compute tracks for.
    num_points : int
        Number of points to track per object (default 16).
        The first point is always the centroid.
    voxel_coords_by_object : dict, optional
        ``{obj_idx: np.ndarray (N, 3)}`` integer grid positions.
        Used as fallback when canonical Gaussians are unavailable.
    c2w_per_frame : dict, optional
        ``{frame_idx: np.ndarray (4, 4)}`` camera-to-world transforms.
        When provided, tracks are transformed from camera space to world
        space.  When None, tracks remain in camera space.
    canonical_mesh_verts, per_frame_mesh_verts, per_frame_mesh_rotations : dict, optional
        Per-canonical-mesh-vertex deformation field (state-shaped dicts
        matching ``state.canonical_mesh_*``).  When supplied AND keyed
        for a given ``(obj_idx, frame_int)``, the canonical sample points
        are warped via ``_warp_at_high_res`` (KNN+IDW LBS) BEFORE the
        Stage-1 rigid pose is applied — so tracks follow the deforming
        surface, not just the rigid Sim(3) component.
        Frames missing from the dict (e.g. interpolated between
        keyframes) fall back to the rigid-only path.

    Returns
    -------
    dict
        ``{obj_idx: ndarray (T, num_points, 3)}`` tracks in R3 convention
        (world space if ``c2w_per_frame`` is provided, camera space otherwise).
        Index 0 along the points axis is the centroid.
    """
    tracks: Dict[int, np.ndarray] = {}

    for obj_idx, poses_dict in interpolated_poses.items():
        # Get canonical xyz: prefer Gaussians, fall back to voxel coords
        if obj_idx in canonical_gaussians:
            canonical_gs = canonical_gaussians[obj_idx]
            device = canonical_gs.get_xyz.device
            canonical_pts = _sample_canonical_points(canonical_gs, num_points)
        elif voxel_coords_by_object and obj_idx in voxel_coords_by_object:
            # Convert voxel grid coords [0, 63] → local space [-0.5, 0.5]
            voxel_xyz = torch.tensor(
                voxel_coords_by_object[obj_idx], dtype=torch.float32,
            )
            voxel_xyz = voxel_xyz / 63.0 - 0.5

            # Create a minimal object with get_xyz for _sample_canonical_points
            class _FakeGS:
                get_xyz = voxel_xyz.cuda()
            device = _FakeGS.get_xyz.device
            canonical_pts = _sample_canonical_points(_FakeGS(), num_points)
        elif canonical_mesh_verts and obj_idx in canonical_mesh_verts:
            # Mesh-only reconstruction: anchor on the canonical mesh
            # vertices (already canonical-norm [-0.5,0.5]).  LAST priority so gaussian
            # and voxel anchors win when present; the per-frame warp block below
            # fires on these anchors exactly as it does for gaussian anchors.
            mesh_v = canonical_mesh_verts[obj_idx]
            mesh_v = mesh_v.cuda() if not mesh_v.is_cuda else mesh_v

            class _FakeGS:
                get_xyz = mesh_v
            device = mesh_v.device
            canonical_pts = _sample_canonical_points(_FakeGS(), num_points)
        else:
            continue

        # Per-frame warp of the canonical sample points before the rigid
        # Sim(3) is applied.  Built per-frame because the deformation field
        # is keyed by frame int and the warp helper operates on one frame at
        # a time; cost is negligible for the few query points involved.
        per_frame_canon_pts: "Optional[List[torch.Tensor]]" = None
        if (
            canonical_mesh_verts is not None
            and obj_idx in canonical_mesh_verts
            and per_frame_mesh_verts is not None
            and per_frame_mesh_rotations is not None
        ):
            from genia.core.utils.deformation import _lookup_per_frame_deformation, _warp_at_high_res
            per_frame_canon_pts = []
            for fi in all_frame_indices:
                fi_int = fi.frame if hasattr(fi, "frame") else int(fi)
                resolved = _lookup_per_frame_deformation(
                    canonical_mesh_verts,
                    per_frame_mesh_verts,
                    per_frame_mesh_rotations,
                    obj_idx, fi_int, device,
                    canonical_mesh_faces,
                )
                if resolved is None:
                    per_frame_canon_pts.append(canonical_pts)
                    continue
                *_warp_core, _faces = resolved
                with torch.no_grad():
                    warped, _ = _warp_at_high_res(
                        canonical_pts, *_warp_core,
                        K=int(warp_knn_k),
                        eps=float(warp_knn_eps),
                        chunk_size=int(warp_knn_chunk_size),
                        faces=_faces,
                    )
                per_frame_canon_pts.append(warped)

        # Stack all poses into batched tensors — (T, 4), (T, 3), (T, 3)
        rots_list, trans_list, sc_list = [], [], []
        for fi in all_frame_indices:
            pose = poses_dict[fi]
            r = pose["rotation"].detach().float().squeeze()
            t = pose["translation"].detach().float().squeeze()
            s = pose["scale"].detach().float().squeeze()
            if s.dim() == 0:
                s = s.expand(3)
            elif s.shape[0] == 1:
                s = s.expand(3)
            rots_list.append(r)
            trans_list.append(t)
            sc_list.append(s)

        all_rots = torch.stack(rots_list).to(device)      # (T, 4)
        all_trans = torch.stack(trans_list).to(device)     # (T, 3)
        all_sc = torch.stack(sc_list).to(device)           # (T, 3)

        # Normalize quaternions and batch convert to rotation matrices
        all_rots = all_rots / all_rots.norm(dim=-1, keepdim=True)
        all_R = _quat_to_matrix(all_rots)  # (T, 3, 3)

        # Batched transform: scale → rotate → translate
        # canonical_pts: (P, 3) → (1, P, 3) for the rigid-only path, or
        # (T, P, 3) when we have per-frame warped points.
        if per_frame_canon_pts is not None:
            canon_TP3 = torch.stack(per_frame_canon_pts, dim=0)  # (T, P, 3)
        else:
            canon_TP3 = canonical_pts.unsqueeze(0).expand(
                len(all_frame_indices), -1, -1,
            )  # (T, P, 3) — same canonical points across frames
        scaled = canon_TP3 * all_sc.unsqueeze(1)                   # (T, P, 3)
        rotated = torch.bmm(scaled, all_R)                         # (T, P, 3)
        transformed = rotated + all_trans.unsqueeze(1)              # (T, P, 3)

        # PyTorch3D → R3 convention: negate X (left→right) and Y (up→down)
        transformed[..., :2] *= -1

        pts_np = transformed.detach().cpu().numpy()  # (T, P, 3)

        # Camera space → world space via c2w
        if c2w_per_frame is not None:
            for t_idx, fi in enumerate(all_frame_indices):
                c2w = c2w_per_frame[fi]
                if not np.allclose(c2w, np.eye(4)):
                    R = c2w[:3, :3]
                    t = c2w[:3, 3]
                    pts_np[t_idx] = pts_np[t_idx] @ R.T + t

        tracks[obj_idx] = pts_np

    return tracks


def _obj_xyz_members(xyz):
    """The ``obj_{i}_xyz`` ``(T, P, 3)`` float32 npz members.

    The ONE definition of that key convention, shared by both track artifacts
    below (``tracks_2d.npz`` dense, ``tracks_3d.npz`` sparse) — their readers
    key off the name, so the two must not drift apart.
    """
    return {f"obj_{oi}_xyz": np.asarray(xyz[oi], dtype=np.float32) for oi in xyz}


def write_tracks_3d(out_dir, tracks_3d):
    """Write the sparse 3D-track artifact ``{out_dir}/tracks_3d.npz``.

    The same world-space tracks :func:`visualize_object_tracks_3d` plots, as DATA:
    ``obj_{i}_xyz`` ``(T, P, 3)`` float32, straight out of
    :func:`compute_object_tracks`, so a 3D viewer can overlay the polylines on
    the reconstruction instead of the motion only existing inside a PNG.

    **Called from every FINAL path that computes tracks** (per-frame and
    canonical), so the file's existence follows ``output.save_tracks_3d`` alone
    rather than which rendering helper the run happened to take — an absent file
    means the flag was off.
    """
    path = os.path.join(out_dir, "tracks_3d.npz")
    np.savez_compressed(path, **_obj_xyz_members(tracks_3d))
    print(f"  Saved 3D object tracks ({len(tracks_3d)} obj) → {path}")


def write_tapvid_tracks(out_dir, all_frame_indices, uv, xyz, vis, H, W, orig_hw=None):
    """Write the dense 2D-track artifact ``{out_dir}/tracks_2d.npz``.

    **Sole owner of that file's on-disk schema.**  The pipeline writes it
    (``core/final.py::_save_tracks_2d``, anchoring on canonical Gaussians + the
    per-frame warp) and a track evaluator reads it.  A reader may be *soft* on the
    crop keys and on ``obj_*_vis`` (falling back when they are absent), so a schema
    change made in only one producer would not raise — it would quietly produce
    non-comparable tracking metrics.  Hence one writer for every caller.

    ``uv`` / ``xyz`` / ``vis`` are ``{obj_idx: array}`` with shapes (T,P,2) / (T,P,3) /
    (T,P) bool.  ``orig_hw`` is the sequence's pre-crop size; it defaults to (H, W)
    (i.e. no crop was applied).
    """
    # Record the crop transform the preprocessing ACTUALLY applied (the pipeline's
    # shared crop_resize_transform + the sequence's original pre-crop size), so an
    # evaluator un-crops the processed-pixel uv back to original-frame normalized
    # coords from STORED provenance — never re-deriving a preprocessing-specific crop
    # (survives swapping the reconstruction model).
    from genia.core.utils.sequence import crop_resize_transform

    oh, ow = (orig_hw if orig_hw is not None else (int(H), int(W)))
    scale, left, top, _, _ = crop_resize_transform((int(oh), int(ow)), (int(H), int(W)))
    frames = np.asarray([int(fi.frame) if hasattr(fi, "frame") else int(fi)
                         for fi in all_frame_indices], dtype=np.int64)
    path = os.path.join(out_dir, "tracks_2d.npz")
    np.savez(path, frames=frames, H=int(H), W=int(W),
             orig_hw=np.asarray([int(oh), int(ow)], dtype=np.int64),
             crop_scale=np.float64(scale),
             crop_offset=np.asarray([int(left), int(top)], dtype=np.int64),
             **{f"obj_{oi}_uv": uv[oi].astype(np.float32) for oi in uv},
             **_obj_xyz_members(xyz),
             **{f"obj_{oi}_vis": vis[oi] for oi in vis})
    print(f"  Saved dense 2D tracks ({len(uv)} obj) → {path}")


def render_canonical_frames_to_numpy(
    canonical_gaussians: Dict[int, Any],
    interpolated_poses: Dict[int, Dict[int, Dict[str, torch.Tensor]]],
    frame_indices: List[int],
    K_per_frame: Dict[int, np.ndarray],
    W: int,
    H: int,
    bg_color: Optional[torch.Tensor] = None,
    c2w_per_frame: Optional[Dict[int, np.ndarray]] = None,
    canonical_mesh_verts: Optional[Dict[int, torch.Tensor]] = None,
    per_frame_mesh_verts: Optional[Dict[int, Dict[int, torch.Tensor]]] = None,
    per_frame_mesh_rotations: Optional[Dict[int, Dict[int, torch.Tensor]]] = None,
    canonical_mesh_faces: Optional[Dict[int, torch.Tensor]] = None,
    warp_knn_k: int = 4,
    warp_knn_eps: float = 1.0e-8,
    warp_knn_chunk_size: int = 131072,
) -> List[np.ndarray]:
    """Foreground-only canonical-Gaussian renders returned as numpy uint8
    ``(H, W, 3)`` per frame — no MP4 / PNG side-effects.  Same per-frame
    pose + warp + gsplat path as ``render_interpolated_sequence``, but
    in-memory so consumers (e.g. side-by-side track overlay) can reuse
    the renders without round-tripping through disk.
    """
    from .gaussian import (
        create_gaussians_object,
        join_gaussians,
        transform_scene_to_r3_convention,
        transform_scene_to_world,
    )
    from .refinement import apply_pose_to_gaussian
    from .rendering import render_gaussians_to_image
    from genia.core.utils.deformation import _lookup_per_frame_deformation, warp_gaussians_high_res

    if bg_color is None:
        bg_color = torch.ones(3)
    obj_indices = sorted(canonical_gaussians.keys())

    rendered_frames: List[np.ndarray] = []
    for fi in frame_indices:
        object_gaussians = []
        for obj_idx in obj_indices:
            if obj_idx not in interpolated_poses or fi not in interpolated_poses[obj_idx]:
                continue
            pose = interpolated_poses[obj_idx][fi]
            rot = pose["rotation"].cuda()
            trans = pose["translation"].cuda()
            sc = pose["scale"].cuda()

            gs_canon = canonical_gaussians[obj_idx]
            means_override = quats_override = None
            frame_int = fi.frame if hasattr(fi, "frame") else int(fi)
            resolved = _lookup_per_frame_deformation(
                canonical_mesh_verts,
                per_frame_mesh_verts,
                per_frame_mesh_rotations,
                obj_idx, frame_int, gs_canon.get_xyz.device,
                canonical_mesh_faces,
            )
            if resolved is not None:
                *_warp_core, _faces = resolved
                with torch.no_grad():
                    means_override, quats_override = warp_gaussians_high_res(
                        gs_canon, *_warp_core,
                        K=warp_knn_k, eps=warp_knn_eps,
                        chunk_size=warp_knn_chunk_size,
                        faces=_faces,
                    )

            xyz, rots, scs, opacities, features = apply_pose_to_gaussian(
                gs_canon, rot, trans, sc,
                means_override=means_override,
                rotation_override=quats_override,
            )
            gs = create_gaussians_object(
                xyz=xyz, features=features, scales=scs,
                rots=rots, opacities=opacities,
            )
            if "dc_offset" in pose:
                gs._features_dc = gs._features_dc + pose["dc_offset"].cuda()
            if "sh_rest" in pose:
                sh = pose["sh_rest"].cuda()
                gs._features_rest = sh
                degree = int((sh.shape[1] + 1) ** 0.5) - 1
                gs.sh_degree = degree
                gs.active_sh_degree = degree
            object_gaussians.append(gs)

        if not object_gaussians:
            # Blank frame so caller's count matches frame_indices length.
            bg_uint8 = (bg_color.cpu().numpy() * 255).astype(np.uint8)
            rendered_frames.append(np.full((H, W, 3), bg_uint8, dtype=np.uint8))
            continue

        scene_gs = (
            object_gaussians[0] if len(object_gaussians) == 1
            else join_gaussians(*object_gaussians)
        )
        scene_gs = transform_scene_to_r3_convention(scene_gs)
        frame_c2w = c2w_per_frame[fi] if c2w_per_frame is not None else None
        if frame_c2w is not None:
            scene_gs = transform_scene_to_world(scene_gs, frame_c2w)
        rendered = render_gaussians_to_image(
            scene_gs, K_per_frame[fi], W, H, bg_color=bg_color, c2w=frame_c2w,
        )
        rendered = torch.clamp(rendered, 0.0, 1.0)
        rendered_frames.append(
            (rendered.detach().cpu().numpy() * 255).astype(np.uint8),
        )

    return rendered_frames


def _load_masked_gt_panel(fi, obj_indices, sequence, gt_frames_path, gt_image_names,
                          gt_masks_path, gt_mask_names, bg_color=None):
    """The left panel of a side-by-side comparison video: the GT image for frame ``fi``
    with its background replaced by ``bg_color`` — the background the render beside it
    actually uses, so the two panels agree outside the objects (``None`` ⇒ black, the
    renderers' default).

    Uses the Sequence's already-downscaled image+masks when ``fi`` is cached, else loads
    from disk via the FrameKey→asset map (raising on a miss rather than aliasing views)."""
    from .io_utils import load_image, load_masks

    if sequence is not None and fi in sequence:
        gt_img = sequence[fi].image[..., :3]
        frame_masks = sequence[fi].masks
    else:
        if sequence is None or not hasattr(sequence, "_fk_to_asset"):
            raise ValueError(
                "side-by-side GT panel requires a Sequence with `_fk_to_asset` to "
                "resolve GT assets.")
        if fi not in sequence._fk_to_asset:
            raise KeyError(
                f"FrameKey {fi} missing from sequence._fk_to_asset "
                f"(available: {sorted(sequence._fk_to_asset.keys())}).")
        asset_idx = sequence._fk_to_asset[fi]
        gt_img = load_image(os.path.join(gt_frames_path, gt_image_names[asset_idx]))[..., :3]
        d = sequence.downscale_factor
        if d > 1:
            gt_img = gt_img[::d, ::d]
        frame_masks = None
        if gt_masks_path is not None and gt_mask_names is not None and asset_idx < len(gt_mask_names):
            raw_masks = load_masks(os.path.join(gt_masks_path, gt_mask_names[asset_idx]))
            frame_masks = (type(raw_masks)({k: v[::d, ::d] for k, v in raw_masks.items()},
                                           shape=(gt_img.shape[0], gt_img.shape[1]))
                           if d > 1 else raw_masks)

    if frame_masks is None:
        return gt_img
    from .rendering import composite_on_render_bg

    return composite_on_render_bg(
        gt_img,
        [frame_masks[o] for o in obj_indices if o in frame_masks],
        bg_color,
    )


def render_interpolated_sequence(
    canonical_gaussians: Dict[int, Any],
    interpolated_poses: Dict[int, Dict[int, Dict[str, torch.Tensor]]],
    all_frame_indices: List[int],
    K_per_frame: Dict[int, np.ndarray],
    W: int,
    H: int,
    output_path: str,
    duration: int = 100,
    max_render_frames: int = 0,
    total_frames: int = 0,
    gt_frames_path: Optional[str] = None,
    gt_image_names: Optional[List[str]] = None,
    gt_masks_path: Optional[str] = None,
    gt_mask_names: Optional[List[str]] = None,
    bg_color: Optional[torch.Tensor] = None,
    c2w_per_frame: Optional[Dict[int, np.ndarray]] = None,
    sequence: Optional[Any] = None,
    canonical_mesh_verts: Optional[Dict[int, torch.Tensor]] = None,
    per_frame_mesh_verts: Optional[Dict[int, Dict[int, torch.Tensor]]] = None,
    per_frame_mesh_rotations: Optional[Dict[int, Dict[int, torch.Tensor]]] = None,
    canonical_mesh_faces: Optional[Dict[int, torch.Tensor]] = None,
    warp_knn_k: int = 4,
    warp_knn_eps: float = 1.0e-8,
    warp_knn_chunk_size: int = 8192,
) -> None:
    """
    Render frames using canonical Gaussians + interpolated poses and save as MP4.

    For each frame, applies each object's interpolated pose to its canonical
    Gaussian, joins them, transforms to R3 convention, renders, and collects
    the result.

    Parameters
    ----------
    canonical_gaussians : dict
        ``{obj_idx: Gaussian}`` canonical Gaussian objects.
    interpolated_poses : dict
        Output of :func:`interpolate_poses`.
    all_frame_indices : list of int
        Frame indices to render (typically ``range(num_frames)``).
    K_per_frame : dict
        ``{frame_idx: np.ndarray(3,3)}`` per-frame camera intrinsics.
    W, H : int
        Image width and height.
    output_path : str
        Path to save the output video.
    duration : int
        Frame duration in milliseconds (default 100 = 10 fps).
    max_render_frames : int
        If positive and ``len(all_frame_indices) > max_render_frames``,
        subsample frames uniformly (always keeping first and last).
        0 = render all frames (default).
    total_frames : int
        Total number of frames in the full sequence. When positive and
        ``len(all_frame_indices) < total_frames``, each rendered keyframe
        is repeated so the video has the same duration as a full-sequence
        video. 0 = no repetition (default).
    gt_frames_path : str, optional
        Directory containing GT images. When provided together with
        ``gt_image_names``, the video shows side-by-side [Masked GT | Rendered].
    gt_image_names : list of str, optional
        Sorted list of GT image filenames (one per frame index in the dataset).
    gt_masks_path : str, optional
        Directory containing segmentation masks.
    gt_mask_names : list of str, optional
        Sorted list of mask filenames (one per frame index in the dataset).
    """
    from PIL import Image

    from .gaussian import create_gaussians_object, join_gaussians, transform_scene_to_r3_convention, transform_scene_to_world
    from .refinement import apply_pose_to_gaussian
    from .rendering import render_gaussians_to_image
    from genia.core.utils.deformation import _lookup_per_frame_deformation, warp_gaussians_high_res

    side_by_side = (gt_frames_path is not None and gt_image_names is not None)
    obj_indices = sorted(canonical_gaussians.keys())

    # Subsample frames if requested
    render_indices = all_frame_indices
    if max_render_frames > 0 and len(all_frame_indices) > max_render_frames:
        step = max(1, len(all_frame_indices) // max_render_frames)
        render_indices = all_frame_indices[::step]
        # Ensure last frame is included
        if render_indices[-1] != all_frame_indices[-1]:
            render_indices.append(all_frame_indices[-1])
        print(f"  Subsampled to {len(render_indices)}/{len(all_frame_indices)} frames for GIF")

    frames: List[Image.Image] = []

    label = "side-by-side " if side_by_side else ""
    print(f"  Rendering {len(render_indices)} {label}interpolated frames...")
    for i, fi in enumerate(render_indices):
        # Build per-object transformed Gaussians for this frame
        object_gaussians = []
        for obj_idx in obj_indices:
            if obj_idx not in interpolated_poses:
                continue
            pose = interpolated_poses[obj_idx][fi]
            rot = pose["rotation"].cuda()
            trans = pose["translation"].cuda()
            sc = pose["scale"].cuda()

            # Per-canonical-mesh-vertex deformation warp.
            # Falls through to static-canonical when not keyed for (obj, frame).
            gs_canon = canonical_gaussians[obj_idx]
            means_override = quats_override = None
            frame_int = fi.frame if hasattr(fi, "frame") else int(fi)
            resolved = _lookup_per_frame_deformation(
                canonical_mesh_verts,
                per_frame_mesh_verts,
                per_frame_mesh_rotations,
                obj_idx, frame_int, gs_canon.get_xyz.device,
                canonical_mesh_faces,
            )
            if resolved is not None:
                *_warp_core, _faces = resolved
                with torch.no_grad():
                    means_override, quats_override = warp_gaussians_high_res(
                        gs_canon, *_warp_core,
                        K=warp_knn_k,
                        eps=warp_knn_eps,
                        chunk_size=warp_knn_chunk_size,
                        faces=_faces,
                    )

            xyz, rots, scs, opacities, features = apply_pose_to_gaussian(
                gs_canon, rot, trans, sc,
                means_override=means_override,
                rotation_override=quats_override,
            )
            gs = create_gaussians_object(
                xyz=xyz, features=features, scales=scs,
                rots=rots, opacities=opacities,
            )
            # Apply per-frame DC offset (bake into base color)
            if "dc_offset" in pose:
                dc = pose["dc_offset"].cuda()
                gs._features_dc = gs._features_dc + dc
            # Override SH rest with interpolated per-frame SH if available
            if "sh_rest" in pose:
                sh = pose["sh_rest"].cuda()
                gs._features_rest = sh
                degree = int((sh.shape[1] + 1) ** 0.5) - 1
                gs.sh_degree = degree
                gs.active_sh_degree = degree
            object_gaussians.append(gs)

        if not object_gaussians:
            continue

        # Combine all objects
        if len(object_gaussians) == 1:
            scene_gs = object_gaussians[0]
        else:
            scene_gs = join_gaussians(*object_gaussians)

        # Transform to R3 convention, then to world space for rendering with c2w
        scene_gs = transform_scene_to_r3_convention(scene_gs)
        frame_c2w = c2w_per_frame[fi] if c2w_per_frame is not None else None
        if frame_c2w is not None:
            scene_gs = transform_scene_to_world(scene_gs, frame_c2w)
        K_matrix = K_per_frame[fi]
        rendered = render_gaussians_to_image(scene_gs, K_matrix, W, H, bg_color=bg_color, c2w=frame_c2w)
        rendered = torch.clamp(rendered, 0.0, 1.0)

        # Convert rendered to numpy uint8
        render_np = (rendered.detach().cpu().numpy() * 255).astype(np.uint8)

        if side_by_side:
            gt_masked = _load_masked_gt_panel(
                fi, obj_indices, sequence, gt_frames_path, gt_image_names,
                gt_masks_path, gt_mask_names, bg_color=bg_color)
            # Compose side-by-side: [Masked GT | Rendered]
            frame_np = np.concatenate([gt_masked, render_np], axis=1)
        else:
            frame_np = render_np

        # Add frame counter
        from .visualization import draw_text_overlay
        frame_rgb = np.ascontiguousarray(frame_np)
        draw_text_overlay(frame_rgb, f"Frame {fi}", (12, 30), font_scale=0.65)
        frames.append(Image.fromarray(frame_rgb))

        if (i + 1) % 20 == 0 or i == len(render_indices) - 1:
            print(f"    {i + 1}/{len(render_indices)} frames rendered")

    if not frames:
        print("  No frames rendered, skipping video.")
        return

    # Compute per-frame repeats so keyframe videos match full-sequence duration
    repeats = None
    if total_frames > 0 and len(render_indices) < total_frames:
        repeats = _compute_keyframe_repeats(render_indices, total_frames)

    _save_frames_as_video(frames, output_path, duration, frame_repeats=repeats)


def save_canonical_renders_perframe(
    canonical_gaussians: Dict[int, Any],
    interpolated_poses: Dict[int, Dict[int, Dict[str, torch.Tensor]]],
    frame_indices: List[int],
    K_per_frame: Dict[int, np.ndarray],
    W: int,
    H: int,
    output_dir: str,
    bg_color: Optional[torch.Tensor] = None,
    c2w_per_frame: Optional[Dict[int, np.ndarray]] = None,
    canonical_mesh_verts: Optional[Dict[int, torch.Tensor]] = None,
    per_frame_mesh_verts: Optional[Dict[int, Dict[int, torch.Tensor]]] = None,
    per_frame_mesh_rotations: Optional[Dict[int, Dict[int, torch.Tensor]]] = None,
    canonical_mesh_faces: Optional[Dict[int, torch.Tensor]] = None,
    warp_knn_k: int = 4,
    warp_knn_eps: float = 1.0e-8,
    warp_knn_chunk_size: int = 8192,
) -> None:
    """Render the posed foreground scene at each frame and save one PNG per frame.

    For each frame in ``frame_indices``, applies that frame's interpolated
    pose to each canonical Gaussian, composes them, transforms to R3 / world
    space, renders from the frame's dataset camera (``K_per_frame`` +
    ``c2w_per_frame``), and writes ``{output_dir}/{frame_idx:03d}.png`` as
    foreground-only RGBA.

    No dataset background is composited — the foreground RGB is composited on
    ``bg_color`` (default white) and the rendered coverage is written to the
    alpha channel, so the empty background reads transparent.

    When the per-canonical-mesh-vertex deformation field is provided
    (``canonical_mesh_verts`` + per-frame dicts; same shape as
    ``state.canonical_mesh_*``) AND keyed for the ``(obj_idx, frame_int)``
    pair, the canonical Gaussian is warped via
    :func:`genia.core.utils.deformation.warp_gaussians_high_res` BEFORE
    the Stage-1 Sim(3) is applied — falls through to the static-canonical
    path when any piece is missing (matches the keyframes-renderer
    behaviour).
    """
    from PIL import Image

    from .gaussian import (
        create_gaussians_object,
        join_gaussians,
        transform_scene_to_r3_convention,
        transform_scene_to_world,
    )
    from .refinement import apply_pose_to_gaussian
    from .rendering import render_gaussians_to_image
    from genia.core.utils.deformation import _lookup_per_frame_deformation, warp_gaussians_high_res

    os.makedirs(output_dir, exist_ok=True)
    if bg_color is None:
        bg_color = torch.ones(3)
    obj_indices = sorted(canonical_gaussians.keys())

    # Per-view layout decision: if any FrameKey has view != 0 OR multiple
    # distinct views are present, write to view{vi:02d}/ subdirs. Single-view
    # mono runs (every fi is FrameKey(t, 0) or bare int) keep flat layout.
    distinct_views = {(fi.view if hasattr(fi, "view") else 0) for fi in frame_indices}
    use_per_view_subdirs = len(distinct_views) > 1

    print(f"  Rendering {len(frame_indices)} foreground-only scene PNGs...")
    skipped = []  # frame keys with no posed object -> no PNG (see the warning below)
    for i, fi in enumerate(frame_indices):
        object_gaussians = []
        for obj_idx in obj_indices:
            if obj_idx not in interpolated_poses or fi not in interpolated_poses[obj_idx]:
                continue
            pose = interpolated_poses[obj_idx][fi]
            rot = pose["rotation"].cuda()
            trans = pose["translation"].cuda()
            sc = pose["scale"].cuda()

            # Per-canonical-mesh-vertex deformation warp.
            # Falls through to static-canonical when the field isn't keyed
            # for this (obj, frame) — matches keyframes-renderer behaviour.
            gs_canon = canonical_gaussians[obj_idx]
            means_override = quats_override = None
            frame_int = fi.frame if hasattr(fi, "frame") else int(fi)
            resolved = _lookup_per_frame_deformation(
                canonical_mesh_verts,
                per_frame_mesh_verts,
                per_frame_mesh_rotations,
                obj_idx, frame_int, gs_canon.get_xyz.device,
                canonical_mesh_faces,
            )
            if resolved is not None:
                *_warp_core, _faces = resolved
                with torch.no_grad():
                    means_override, quats_override = warp_gaussians_high_res(
                        gs_canon, *_warp_core,
                        K=warp_knn_k,
                        eps=warp_knn_eps,
                        chunk_size=warp_knn_chunk_size,
                        faces=_faces,
                    )

            xyz, rots, scs, opacities, features = apply_pose_to_gaussian(
                gs_canon, rot, trans, sc,
                means_override=means_override,
                rotation_override=quats_override,
            )
            gs = create_gaussians_object(
                xyz=xyz, features=features, scales=scs,
                rots=rots, opacities=opacities,
            )
            if "dc_offset" in pose:
                gs._features_dc = gs._features_dc + pose["dc_offset"].cuda()
            if "sh_rest" in pose:
                sh = pose["sh_rest"].cuda()
                gs._features_rest = sh
                degree = int((sh.shape[1] + 1) ** 0.5) - 1
                gs.sh_degree = degree
                gs.active_sh_degree = degree
            object_gaussians.append(gs)

        if not object_gaussians:
            skipped.append(fi)
            continue

        if len(object_gaussians) == 1:
            scene_gs = object_gaussians[0]
        else:
            scene_gs = join_gaussians(*object_gaussians)

        scene_gs = transform_scene_to_r3_convention(scene_gs)
        frame_c2w = c2w_per_frame[fi] if c2w_per_frame is not None else None
        if frame_c2w is not None:
            scene_gs = transform_scene_to_world(scene_gs, frame_c2w)
        K_matrix = K_per_frame[fi]
        rendered, alpha = render_gaussians_to_image(
            scene_gs, K_matrix, W, H, bg_color=bg_color, c2w=frame_c2w,
            return_alpha=True,
        )
        rendered = torch.clamp(rendered, 0.0, 1.0)

        # RGBA: keep the rendered RGB (foreground on bg_color) and carry the
        # coverage in the alpha channel so the background reads transparent.
        rgb_np = (rendered.detach().cpu().numpy() * 255).astype(np.uint8)
        a_np = (torch.clamp(alpha, 0.0, 1.0).detach().cpu().numpy()
                * 255).astype(np.uint8)
        rgba_np = np.dstack([rgb_np, a_np])
        # FrameKey-aware path: per-view subdirs only when multiple views
        # exist (mono runs keep the flat layout). 3-digit frame stem.
        frame_n = fi.frame if hasattr(fi, "frame") else int(fi)
        view_n = fi.view if hasattr(fi, "view") else 0
        out_dir_for_frame = (os.path.join(output_dir, f"view{view_n:02d}")
                             if use_per_view_subdirs else output_dir)
        os.makedirs(out_dir_for_frame, exist_ok=True)
        Image.fromarray(rgba_np).save(
            os.path.join(out_dir_for_frame, f"{frame_n:03d}.png")
        )

        if (i + 1) % 20 == 0 or i == len(frame_indices) - 1:
            print(f"    {i + 1}/{len(frame_indices)} PNGs saved")

    print(f"  Saved {len(frame_indices) - len(skipped)} scene renders to {output_dir}")
    if skipped:
        # No pose for these frame keys, so nothing was written for them. Loud on
        # purpose: a silently missing view would leave train-split metrics
        # averaged over fewer views than the run has.
        # Truncated -- a long sequence can skip hundreds and bury the count.
        shown = ", ".join(str(fk) for fk in skipped[:6])
        more = f" (+{len(skipped) - 6} more)" if len(skipped) > 6 else ""
        print(f"  [warn] {len(skipped)} of {len(frame_indices)} frame(s) had no "
              f"posed object and were NOT rendered: {shown}{more}")


def _build_pose_lookup(
    tokens_by_object: Dict[int, List[Tuple[int, Dict[str, Any]]]],
) -> Dict[int, Dict[int, Dict[str, Any]]]:
    """``{obj_idx: {frame_idx: decoder_input}}`` from per-object token lists."""
    return {oi: {fid: di for fid, di in toks}
            for oi, toks in tokens_by_object.items()}


def _render_perframe_scene_rgb(
    perframe_gaussians: Dict[int, Dict[int, Any]],
    pose_lookup: Dict[int, Dict[int, Dict[str, Any]]],
    fi: Any,
    K_per_frame: Dict[int, np.ndarray],
    W: int,
    H: int,
    bg_color: Optional[torch.Tensor],
    c2w_per_frame: Optional[Dict[int, np.ndarray]],
    return_alpha: bool = False,
) -> Optional[torch.Tensor]:
    """Compose this frame's per-frame Gaussians under their pose tokens and
    render one foreground-only RGB tensor (clamped 0..1),
    or ``None`` when no object has a Gaussian+pose at ``fi``.

    Shared core of :func:`render_perframe_sequence` (video sink) and
    :func:`save_perframe_renders_perframe` (per-PNG sink). When
    ``return_alpha=True`` a ``(rgb, alpha)`` tuple is returned instead (both
    ``None`` when nothing renders), so the PNG sink can save a
    transparent-background RGBA image.
    """
    from .gaussian import (
        create_gaussians_object,
        join_gaussians,
        transform_scene_to_r3_convention,
        transform_scene_to_world,
    )
    from .refinement import apply_pose_to_gaussian
    from .rendering import render_gaussians_to_image

    object_gaussians = []
    for obj_idx in sorted(perframe_gaussians.keys()):
        if fi not in perframe_gaussians.get(obj_idx, {}):
            continue
        if fi not in pose_lookup.get(obj_idx, {}):
            continue
        di = pose_lookup[obj_idx][fi]
        rot = di["rotation"].cuda()
        trans = di["translation"].cuda()
        sc = di["scale"].cuda()
        xyz, rots, scs, opacities, features = apply_pose_to_gaussian(
            perframe_gaussians[obj_idx][fi], rot, trans, sc
        )
        object_gaussians.append(create_gaussians_object(
            xyz=xyz, features=features, scales=scs,
            rots=rots, opacities=opacities,
        ))

    if not object_gaussians:
        return (None, None) if return_alpha else None
    scene_gs = (object_gaussians[0] if len(object_gaussians) == 1
                else join_gaussians(*object_gaussians))
    scene_gs = transform_scene_to_r3_convention(scene_gs)
    frame_c2w = c2w_per_frame[fi] if c2w_per_frame is not None else None
    if frame_c2w is not None:
        scene_gs = transform_scene_to_world(scene_gs, frame_c2w)
    # Alpha comes free from the same rasterization; return it only when asked
    # (the video sink keeps rgb-only behaviour).
    rendered, alpha = render_gaussians_to_image(
        scene_gs, K_per_frame[fi], W, H, bg_color=bg_color, c2w=frame_c2w,
        return_alpha=True,
    )
    rendered = torch.clamp(rendered, 0.0, 1.0)
    return (rendered, alpha) if return_alpha else rendered


def save_perframe_renders_perframe(
    perframe_gaussians: Dict[int, Dict[int, Any]],
    tokens_by_object: Dict[int, List[Tuple[int, Dict[str, Any]]]],
    frame_indices: List[int],
    K_per_frame: Dict[int, np.ndarray],
    W: int,
    H: int,
    output_dir: str,
    bg_color: Optional[torch.Tensor] = None,
    c2w_per_frame: Optional[Dict[int, np.ndarray]] = None,
) -> None:
    """Per-frame analogue of :func:`save_canonical_renders_perframe`.

    Renders each frame's OWN per-frame Gaussian (not a frame-0 fallback)
    posed by that frame's tokens and writes one foreground-only RGBA
    ``{frame_idx:03d}.png`` (RGB composited on ``bg_color`` (default white),
    rendered coverage in the alpha channel so the background is transparent).
    Used by FINAL for per-frame-only pipelines where there is no canonical
    scene; same per-view-subdir output contract as
    :func:`save_canonical_renders_perframe`.
    """
    from PIL import Image

    os.makedirs(output_dir, exist_ok=True)
    if bg_color is None:
        bg_color = torch.ones(3)
    pose_lookup = _build_pose_lookup(tokens_by_object)
    distinct_views = {(fi.view if hasattr(fi, "view") else 0)
                      for fi in frame_indices}
    use_per_view_subdirs = len(distinct_views) > 1

    print(f"  Rendering {len(frame_indices)} per-frame foreground-only PNGs...")
    n = 0
    for fi in frame_indices:
        rendered, alpha = _render_perframe_scene_rgb(
            perframe_gaussians, pose_lookup, fi,
            K_per_frame, W, H, bg_color, c2w_per_frame,
            return_alpha=True,
        )
        if rendered is None:
            continue
        # RGBA: keep the rendered RGB (foreground on bg_color) and carry the
        # coverage in the alpha channel so the background reads transparent.
        rgb_np = (rendered.detach().cpu().numpy() * 255).astype(np.uint8)
        a_np = (torch.clamp(alpha, 0.0, 1.0).detach().cpu().numpy()
                * 255).astype(np.uint8)
        rgba_np = np.dstack([rgb_np, a_np])
        frame_n = fi.frame if hasattr(fi, "frame") else int(fi)
        view_n = fi.view if hasattr(fi, "view") else 0
        out_dir_for_frame = (os.path.join(output_dir, f"view{view_n:02d}")
                             if use_per_view_subdirs else output_dir)
        os.makedirs(out_dir_for_frame, exist_ok=True)
        Image.fromarray(rgba_np).save(
            os.path.join(out_dir_for_frame, f"{frame_n:03d}.png")
        )
        n += 1

    print(f"  Saved {n} per-frame scene renders to {output_dir}")


def render_perframe_sequence(
    perframe_gaussians: Dict[int, Dict[int, Any]],
    tokens_by_object: Dict[int, List[Tuple[int, Dict[str, Any]]]],
    frame_indices: List[int],
    K_per_frame: Dict[int, np.ndarray],
    W: int,
    H: int,
    output_path: str,
    duration: int = 100,
    bg_color: Optional[torch.Tensor] = None,
    c2w_per_frame: Optional[Dict[int, np.ndarray]] = None,
) -> None:
    """
    Render keyframes using per-frame Gaussians + per-frame poses and save as GIF.

    Unlike :func:`render_interpolated_sequence`, which uses a single canonical
    Gaussian per object, this function uses independent per-frame Gaussians —
    each frame has its own decoded Gaussian with its own refined pose.

    Parameters
    ----------
    perframe_gaussians : dict
        ``{obj_idx: {frame_idx: Gaussian}}`` per-frame decoded Gaussians.
    tokens_by_object : dict
        ``{obj_idx: [(frame_idx, decoder_input), ...]}`` with poses stored
        in each ``decoder_input`` dict.
    frame_indices : list of int
        Keyframe indices to render (e.g. ``range(0, N, stride)``).
    K_per_frame : dict
        ``{frame_idx: np.ndarray(3,3)}`` per-frame camera intrinsics.
    W, H : int
        Image width and height.
    output_path : str
        Path to save the output GIF.
    duration : int
        Frame duration in milliseconds (default 100 = 10 fps).
    bg_color : torch.Tensor or None
        Background color for rendering (3,). None = black.
    """
    from PIL import Image

    pose_lookup = _build_pose_lookup(tokens_by_object)
    frames: List[Image.Image] = []

    print(f"  Rendering {len(frame_indices)} per-frame keyframes...")
    for i, fi in enumerate(frame_indices):
        rendered = _render_perframe_scene_rgb(
            perframe_gaussians, pose_lookup, fi,
            K_per_frame, W, H, bg_color, c2w_per_frame,
        )
        if rendered is None:
            continue

        # Convert to numpy uint8 and add frame counter
        from .visualization import draw_text_overlay
        frame_np = (rendered.detach().cpu().numpy() * 255).astype(np.uint8)
        frame_rgb = np.ascontiguousarray(frame_np)
        draw_text_overlay(frame_rgb, f"Frame {fi}", (12, 30), font_scale=0.65)
        frames.append(Image.fromarray(frame_rgb))

        if (i + 1) % 20 == 0 or i == len(frame_indices) - 1:
            print(f"    {i + 1}/{len(frame_indices)} frames rendered")

    if not frames:
        print("  No frames rendered, skipping video.")
        return

    _save_frames_as_video(frames, output_path, duration)
    print(f"  Saved per-frame sequence video ({len(frames)} frames) to {output_path}")


def compute_pose_axes(
    interpolated_poses: Dict[int, Dict[int, Dict[str, torch.Tensor]]],
    all_frame_indices: List[int],
    axis_length: float = 0.15,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Compute per-object centroid position and local-frame axis endpoints.

    For each object at each frame, transforms the local origin and three
    unit-axis tips through the pose (scale, rotate, translate), then
    converts from PyTorch3D to R3 convention.

    Parameters
    ----------
    interpolated_poses : dict
        Output of :func:`interpolate_poses`.
    all_frame_indices : list of int
        Frame indices to compute axes for.
    axis_length : float
        Length of each axis arrow in canonical (local) units.

    Returns
    -------
    dict
        ``{obj_idx: {"centroids": (T, 3), "axes": (T, 3, 3)}}``
        in R3 camera-space coordinates.  ``axes[t, i, :]`` is the endpoint of
        axis *i* (0=X, 1=Y, 2=Z) at frame *t*.
    """
    result: Dict[int, Dict[str, np.ndarray]] = {}

    # Unit axis vectors in canonical (local) space, scaled
    local_axes = torch.eye(3) * axis_length  # (3, 3)

    for obj_idx, poses_dict in interpolated_poses.items():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        rots_list, trans_list, sc_list = [], [], []
        for fi in all_frame_indices:
            pose = poses_dict[fi]
            r = pose["rotation"].detach().float().squeeze()
            t = pose["translation"].detach().float().squeeze()
            s = pose["scale"].detach().float().squeeze()
            if s.dim() == 0:
                s = s.expand(3)
            elif s.shape[0] == 1:
                s = s.expand(3)
            rots_list.append(r)
            trans_list.append(t)
            sc_list.append(s)

        all_rots = torch.stack(rots_list).to(device)   # (T, 4)
        all_trans = torch.stack(trans_list).to(device)  # (T, 3)
        all_sc = torch.stack(sc_list).to(device)        # (T, 3)

        all_rots = all_rots / all_rots.norm(dim=-1, keepdim=True)
        all_R = _quat_to_matrix(all_rots)  # (T, 3, 3)

        # Centroid: origin in local space → scale @ R + trans
        # origin * scale = [0,0,0], so centroid = translation
        centroids_p3d = all_trans.clone()  # (T, 3)

        # Axis tips: axis_vec * scale @ R + trans
        axes_local = local_axes.to(device)  # (3, 3)
        # (1, 3, 3) * (T, 1, 3) → (T, 3, 3): scaled axes per frame
        scaled_axes = axes_local.unsqueeze(0) * all_sc.unsqueeze(1)
        # (T, 3, 3) @ (T, 3, 3) → (T, 3, 3): rotated axes
        rotated_axes = torch.bmm(scaled_axes, all_R)
        # Add translation: (T, 3, 3) + (T, 1, 3) → (T, 3, 3)
        axes_p3d = rotated_axes + all_trans.unsqueeze(1)

        # PyTorch3D → R3 convention: negate X (left→right) and Y (up→down)
        centroids_p3d[..., :2] *= -1
        axes_p3d[..., :2] *= -1
        centroids_r3 = centroids_p3d.detach().cpu().numpy()
        axes_r3 = axes_p3d.detach().cpu().numpy()

        result[obj_idx] = {
            "centroids": centroids_r3,  # (T, 3)
            "axes": axes_r3,            # (T, 3, 3)
        }

    return result


__all__ = [
    "interpolate_c2w",
    "interpolate_K",
    "interpolate_poses",
    "compute_object_tracks",
    "compute_pose_axes",
    "render_interpolated_sequence",
    "render_perframe_sequence",
]
