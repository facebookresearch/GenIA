# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Pose refinement utilities using differentiable rendering.

This module provides functions for refining object poses (rotation, translation,
scale) using gradient-based optimization with differentiable Gaussian rendering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Tuple

import random

import numpy as np
import torch
from tqdm import tqdm
from .quaternion_ops import (
    matrix_to_quaternion,
    p3d_to_r3_positions,
    p3d_to_r3_quaternions,
    quaternion_invert,
    quaternion_multiply,
    quaternion_to_matrix,
)

from .config import (
    CHAMFER_GT_TRIM_DEFAULT, LossConfig, PipelineConfig, resolve_chamfer_gt_trim,
    resolve_correction_granularity, resolve_correction_scale_control,
)
from .pose_params import (
    build_correction, freeze_params, natives_to_freeze,
)
from .timing import get_timer

from .rendering import bulk_keep_mask, render_gaussian_params

# Fast fused-CUDA SSIM (rahul-goel/fused-ssim), 5-10x faster than pytorch_msssim
# and numerically identical with padding="valid". Optional — falls back to
# pytorch_msssim in _compute_ssim_loss when it isn't installed.
try:
    from fused_ssim import fused_ssim as _fused_ssim
except Exception:
    _fused_ssim = None

# Keys carried forward from decoder_input when building refined output dicts.
_CARRY_FORWARD_KEYS = (
    "raw_ss_modalities", "pointmap_scale", "pointmap_shift",
    "downsample_factor", "decoder_input_slat",
)


def _carry_forward_decoder_context(
    source: Dict[str, Any], target: Dict[str, Any],
) -> None:
    """Copy non-pose context fields from *source* to *target* decoder_input dict."""
    for key in _CARRY_FORWARD_KEYS:
        if key in source:
            target[key] = source[key]


def _sync_raw_from_decoded(
    raw_mods: Dict[str, torch.Tensor],
    rotation_quat: torch.Tensor,
    translation: torch.Tensor,
    scale: torch.Tensor,
    raw_source: Dict[str, Any],
    fallback_raw_6d: torch.Tensor,
    fallback_raw_trans: torch.Tensor,
) -> None:
    """Overwrite raw layout tokens in *raw_mods* so that the SSI decode of the
    raw tokens round-trips to (``rotation_quat``, ``translation``, ``scale``).

    The global-scale optimizers in this module treat ``opt_global_scale`` as
    an independent decoded variable (decoupled from ``frozen_raw_scales``),
    so the optimized raw scale token is stale relative to the final decoded
    scale, and the optimized raw translation was paired with the frozen raw
    scale.  Inverting via ``camera_pose_to_raw_tokens`` produces raw tokens
    consistent with the decoded triplet.

    Falls back to the optimized raw rotation/translation when SSI params are
    missing on *raw_source* (no SSI snapshot).  In that case the
    raw scale stays whatever was already in *raw_mods*: no consumer can
    re-decode pose without SSI params anyway, so the contract is moot.
    """
    from genia.core.utils.pose_token_gt import camera_pose_to_raw_tokens

    ps = raw_source.get("pointmap_scale")
    psh = raw_source.get("pointmap_shift")
    if ps is None or psh is None:
        raw_mods["6drotation_normalized"] = fallback_raw_6d
        raw_mods["translation"] = fallback_raw_trans
        return

    dsf = raw_source.get("downsample_factor", 1.0)
    R_mat = quaternion_to_matrix(rotation_quat.reshape(1, 4)).squeeze(0)
    t_vec = translation.reshape(-1).float()
    s_vec = scale.reshape(-1).float()
    if s_vec.numel() == 1:
        s_vec = s_vec.expand(3)
    device = R_mat.device
    with torch.no_grad():
        new_raw = camera_pose_to_raw_tokens(
            R_mat, t_vec.to(device), s_vec.to(device),
            ps.flatten().to(device), psh.flatten().to(device),
            downsample_factor=dsf,
        )
    raw_mods["6drotation_normalized"] = new_raw["6drotation_normalized"]
    raw_mods["translation"] = new_raw["translation"]
    raw_mods["scale"] = new_raw["scale"]

if TYPE_CHECKING:
    from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian.gaussian_model import (
        Gaussian,
    )


# ---------------------------------------------------------------------------
# Background color helper
# ---------------------------------------------------------------------------

def _get_bg_color(pipeline: Optional[PipelineConfig], device: torch.device) -> torch.Tensor:
    """Return rendering background color from pipeline config (black or white)."""
    if pipeline is not None and pipeline.white_background:
        return torch.ones(3, device=device)
    return torch.zeros(3, device=device)


def _precompute_perframe_gaussian_warps(
    canonical_gaussians: Dict[int, Any],
    frames_per_obj: Dict[int, List[Any]],
    canonical_mesh_verts_per_obj: Optional[Dict[int, torch.Tensor]],
    per_frame_mesh_verts_per_obj: Optional[Dict[int, Dict[int, torch.Tensor]]],
    per_frame_mesh_rotations_per_obj: Optional[Dict[int, Dict[int, torch.Tensor]]],
    canonical_mesh_faces_per_obj: Optional[Dict[int, torch.Tensor]] = None,
    *,
    warp_knn_k: int,
    warp_knn_eps: float,
    warp_knn_chunk_size: int,
) -> Dict[int, Dict[Any, Tuple[torch.Tensor, torch.Tensor]]]:
    """Compose ``_lookup_per_frame_deformation`` + ``warp_gaussians_high_res``
    over every (obj, frame) pair, returning ``{obj: {frame: (means_w, quats_w)}}``.

    Used by global pose refinement, where canonical Gaussians are held
    fixed across the optimization loop, so the rigid-LBS warp is constant
    and can be cached once with ``no_grad``.  Returns ``{}`` when the
    deformation field is unsupplied or empty (callers should fall through
    to the static-canonical path).  Per-(obj, frame) miss inside
    ``_lookup_per_frame_deformation`` is a per-cell skip — same behaviour
    as the keyframes / evaluation / interpolation call sites.
    """
    if (
        canonical_mesh_verts_per_obj is None
        or per_frame_mesh_verts_per_obj is None
        or per_frame_mesh_rotations_per_obj is None
        or not canonical_mesh_verts_per_obj
    ):
        return {}

    from genia.core.utils.deformation import _lookup_per_frame_deformation, warp_gaussians_high_res

    cache: Dict[int, Dict[Any, Tuple[torch.Tensor, torch.Tensor]]] = {}
    for obj_idx, frame_list in frames_per_obj.items():
        if obj_idx not in canonical_mesh_verts_per_obj:
            continue
        gs_obj = canonical_gaussians.get(obj_idx)
        if gs_obj is None:
            continue
        per_obj: Dict[Any, Tuple[torch.Tensor, torch.Tensor]] = {}
        for frame_idx in frame_list:
            fi_int = (
                frame_idx.frame if hasattr(frame_idx, "frame") else int(frame_idx)
            )
            resolved = _lookup_per_frame_deformation(
                canonical_mesh_verts_per_obj,
                per_frame_mesh_verts_per_obj,
                per_frame_mesh_rotations_per_obj,
                obj_idx, fi_int, gs_obj.get_xyz.device,
                canonical_mesh_faces_per_obj,
            )
            if resolved is None:
                continue
            *_warp_core, _faces = resolved
            with torch.no_grad():
                means_w, quats_w = warp_gaussians_high_res(
                    gs_obj, *_warp_core,
                    K=int(warp_knn_k),
                    eps=float(warp_knn_eps),
                    chunk_size=int(warp_knn_chunk_size),
                    faces=_faces,
                )
            per_obj[frame_idx] = (means_w.detach(), quats_w.detach())
        if per_obj:
            cache[obj_idx] = per_obj
    return cache


def _perframe_warp_overrides(
    canonical_src: Any,
    obj_idx: int,
    frames: List[Any],
    device: torch.device,
    canonical_mesh_verts_per_obj: Optional[Dict[int, torch.Tensor]],
    per_frame_mesh_verts_per_obj: Optional[Dict[int, Dict[int, torch.Tensor]]],
    per_frame_mesh_rotations_per_obj: Optional[Dict[int, Dict[int, torch.Tensor]]],
    canonical_mesh_faces_per_obj: Optional[Dict[int, torch.Tensor]] = None,
    *,
    warp_knn_k: int,
    warp_knn_eps: float,
    warp_knn_chunk_size: int,
) -> Dict[Any, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """``{frame: (means_override, rotation_override)}`` for ONE object — what
    :class:`PosedObjectRenderer` takes per call.

    ``{}`` when no deformation field is supplied — every lookup then misses and the
    caller renders the static canonical.
    """
    warp_kw = dict(
        warp_knn_k=warp_knn_k, warp_knn_eps=warp_knn_eps,
        warp_knn_chunk_size=warp_knn_chunk_size,
    )
    field = (canonical_mesh_verts_per_obj, per_frame_mesh_verts_per_obj,
             per_frame_mesh_rotations_per_obj, canonical_mesh_faces_per_obj)
    return _precompute_perframe_gaussian_warps(
        {obj_idx: canonical_src}, {obj_idx: frames}, *field, **warp_kw,
    ).get(obj_idx, {})


# ---------------------------------------------------------------------------
# Epoch-based frame sampler
# ---------------------------------------------------------------------------

class EpochFrameSampler:
    """Epoch-based frame sampler: all frames are seen before any repeats."""

    def __init__(self, frame_indices):
        self._all = list(frame_indices)
        self._queue: list = []

    def sample(self, batch_size: int) -> list:
        batch: list = []
        while len(batch) < batch_size:
            if not self._queue:
                self._queue = list(self._all)
                random.shuffle(self._queue)
            take = min(batch_size - len(batch), len(self._queue))
            batch.extend(self._queue[:take])
            self._queue = self._queue[take:]
        return batch


def _compute_center_of_mass_loss(
    alpha: torch.Tensor,
    mask: torch.Tensor,
    occlusion_robust: bool = False,
) -> torch.Tensor:
    """
    Compute center-of-mass alignment loss between rendered alpha and GT mask.

    Standard mode: compares centroid of the full predicted alpha vs centroid
    of the GT mask. Provides useful gradients even with zero overlap.

    Occlusion-robust mode: compares centroid of the INTERSECTION (predicted
    alpha masked by GT mask) vs centroid of GT mask. This prevents the
    predicted centroid from being biased by alpha in occluded regions where
    the GT mask is absent.

    Parameters
    ----------
    alpha : torch.Tensor
        Rendered alpha/opacity map, shape (H, W), values in [0, 1].
    mask : torch.Tensor
        Ground truth binary mask, shape (H, W), boolean or float.
    occlusion_robust : bool
        If True, compute render centroid from intersection of alpha and GT mask.

    Returns
    -------
    torch.Tensor
        Scalar loss representing squared distance between centroids,
        normalized by image diagonal squared.
    """
    H, W = alpha.shape
    device = alpha.device

    # Create coordinate grids
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )

    # Compute GT mask center of mass (same in both modes)
    mask_float = mask.float()
    mask_sum = mask_float.sum() + 1e-6
    gt_cx = (mask_float * x_coords).sum() / mask_sum
    gt_cy = (mask_float * y_coords).sum() / mask_sum

    if occlusion_robust:
        # Compute render centroid from intersection only (alpha * mask)
        visible_render = alpha * mask_float
        visible_sum = visible_render.sum() + 1e-6
        render_cx = (visible_render * x_coords).sum() / visible_sum
        render_cy = (visible_render * y_coords).sum() / visible_sum
    else:
        # Standard: centroid of full predicted alpha
        alpha_sum = alpha.sum() + 1e-6
        render_cx = (alpha * x_coords).sum() / alpha_sum
        render_cy = (alpha * y_coords).sum() / alpha_sum

    # Squared distance between centers, normalized by image diagonal
    diagonal = (H**2 + W**2) ** 0.5
    com_loss = ((render_cx - gt_cx) ** 2 + (render_cy - gt_cy) ** 2) / (diagonal**2)

    return com_loss


def _compute_signed_distance_transform(
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute signed distance transform of a binary mask.

    The SDT is positive outside the mask (distance to nearest mask pixel)
    and negative inside the mask (distance to nearest non-mask pixel).
    This creates a smooth field that guides optimization toward the mask.

    Parameters
    ----------
    mask : torch.Tensor
        Binary mask, shape (H, W), boolean or float.

    Returns
    -------
    torch.Tensor
        Signed distance transform, shape (H, W). Positive outside mask,
        negative inside mask.

    Notes
    -----
    This function uses scipy.ndimage.distance_transform_edt and is not
    differentiable. The SDT should be precomputed once per frame.
    """
    from scipy.ndimage import distance_transform_edt

    mask_np = mask.detach().cpu().numpy().astype(bool)

    # Distance from outside points to nearest inside point
    dist_outside = distance_transform_edt(~mask_np)

    # Distance from inside points to nearest outside point
    dist_inside = distance_transform_edt(mask_np)

    # Signed: positive outside, negative inside
    sdt = dist_outside - dist_inside

    return torch.from_numpy(sdt).float().to(mask.device)


def _compute_sdt_loss(
    alpha: torch.Tensor,
    sdt: torch.Tensor,
    occlusion_robust: bool = False,
) -> torch.Tensor:
    """
    Compute signed distance transform loss for silhouette alignment.

    Standard mode: penalizes rendered alpha weighted by the signed distance
    to the GT mask boundary. Pixels outside the mask contribute positive loss,
    pixels inside contribute negative loss (reward).

    Occlusion-robust mode ("interior-pull"): only penalizes missing coverage
    inside the GT mask. For each pixel inside the GT mask (sdt < 0), the loss
    is ``(1 - alpha) * |sdt|``, pulling alpha toward 1 proportionally to depth
    inside the mask. Pixels outside the GT mask are ignored entirely, so
    predicted alpha extending into occluded regions is not penalized.

    Parameters
    ----------
    alpha : torch.Tensor
        Rendered alpha/opacity map, shape (H, W), values in [0, 1].
    sdt : torch.Tensor
        Precomputed signed distance transform of GT mask, shape (H, W).
        Positive outside mask, negative inside mask.
    occlusion_robust : bool
        If True, use asymmetric interior-pull loss.

    Returns
    -------
    torch.Tensor
        Scalar loss. Lower values indicate better alignment.
    """
    H, W = alpha.shape
    diagonal = (H**2 + W**2) ** 0.5
    sdt_normalized = sdt / diagonal

    if occlusion_robust:
        # Interior-pull: for pixels INSIDE the GT mask (sdt < 0),
        # penalize (1 - alpha) proportional to how deep inside the mask they are.
        # Pixels outside the GT mask are ignored entirely.
        inside_mask = (sdt < 0).float()
        interior_depth = (-sdt_normalized).clamp(min=0)  # |sdt|/diagonal, nonzero only inside
        inside_sum = inside_mask.sum() + 1e-6
        sdt_loss = ((1.0 - alpha) * interior_depth * inside_mask).sum() / inside_sum
        return sdt_loss

    # Standard: alpha-weighted average of SDT values
    alpha_sum = alpha.sum() + 1e-6
    sdt_loss = (alpha * sdt_normalized).sum() / alpha_sum
    return sdt_loss


def _compute_soft_iou_loss(
    alpha: torch.Tensor,
    mask: torch.Tensor,
    occlusion_robust: bool = False,
) -> torch.Tensor:
    """
    Compute soft intersection-over-union loss for silhouette matching.

    Soft IoU provides a differentiable approximation to the IoU metric,
    treating alpha values as soft membership. This loss is effective for
    fine-grained shape matching once the rendered object is roughly
    aligned with the target mask.

    When ``occlusion_robust=True``, uses a coverage loss instead: only
    penalizes GT mask pixels not covered by the prediction, without
    penalizing predicted alpha that extends beyond the GT mask (which may
    correspond to occluded regions of the object).

    Parameters
    ----------
    alpha : torch.Tensor
        Rendered alpha/opacity map, shape (H, W), values in [0, 1].
    mask : torch.Tensor
        Ground truth binary mask, shape (H, W), boolean or float.
    occlusion_robust : bool
        If True, use asymmetric coverage loss instead of symmetric IoU.

    Returns
    -------
    torch.Tensor
        Scalar loss in [0, 1]. 0 = perfect overlap/coverage, 1 = no overlap.
    """
    mask_float = mask.float()

    if occlusion_robust:
        # Coverage loss: penalizes GT pixels not covered by prediction.
        # Does NOT penalize prediction extending beyond GT (occluded regions).
        intersection = (alpha * mask_float).sum()
        coverage = intersection / (mask_float.sum() + 1e-6)
        return 1.0 - coverage

    # Standard symmetric soft IoU
    intersection = (alpha * mask_float).sum()
    union = alpha.sum() + mask_float.sum() - intersection + 1e-6
    iou = intersection / union
    return 1.0 - iou


def _compute_ssim_loss(
    rgb: torch.Tensor,
    gt_masked: torch.Tensor,
) -> torch.Tensor:
    """
    Compute SSIM (Structural Similarity) loss between rendered and GT images.

    SSIM captures perceptual similarity by comparing luminance, contrast, and
    structure. It's more robust to small misalignments than pixel-wise losses
    and provides better gradients for pose optimization.

    Parameters
    ----------
    rgb : torch.Tensor
        Rendered RGB image, shape (H, W, 3), values in [0, 1].
    gt_masked : torch.Tensor
        Ground truth RGB image with background masked to black,
        shape (H, W, 3), values in [0, 1].

    Returns
    -------
    torch.Tensor
        Scalar SSIM loss in [0, 1]. 0 = identical, 1 = completely different.

    Notes
    -----
    Uses the fused-CUDA SSIM kernel (fused_ssim) when available — 5-10x faster
    than pytorch_msssim and numerically identical with ``padding="valid"``
    (same 11x11 sigma=1.5 window, valid convolution). Falls back to
    pytorch_msssim if fused_ssim isn't installed.
    """
    # Convert from (H, W, C) to (B, C, H, W). fused_ssim needs contiguous input.
    rgb_bchw = rgb.permute(2, 0, 1).unsqueeze(0).contiguous()
    gt_bchw = gt_masked.permute(2, 0, 1).unsqueeze(0).contiguous()

    if _fused_ssim is not None:
        # padding="valid" matches pytorch_msssim's default (no border padding);
        # train= gates the backward buffers to when a backward can actually run.
        ssim_value = _fused_ssim(
            rgb_bchw, gt_bchw, padding="valid", train=torch.is_grad_enabled()
        )
    else:
        from pytorch_msssim import ssim

        ssim_value = ssim(rgb_bchw, gt_bchw, data_range=1.0, size_average=True)

    ssim_loss = torch.clamp(1.0 - ssim_value, 0.0, 1.0)

    return ssim_loss

def _l1_loss(rgb: torch.Tensor, gt_image: torch.Tensor) -> torch.Tensor:
    """
    Compute L1 loss between rendered and ground truth images.

    Parameters
    ----------
    rgb : torch.Tensor
        Rendered RGB image, shape (H, W, 3), values in [0, 1].
    gt_image : torch.Tensor
        Ground truth RGB image, shape (H, W, 3), values in [0, 1].

    Returns
    -------
    torch.Tensor
        Scalar L1 loss.
    """
    return (rgb - gt_image).abs().mean()


def _depth_l1_term(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    gt_mask: torch.Tensor,
    *,
    mask_only: bool = False,
    valid_mask: Optional[torch.Tensor] = None,
    return_error_map: bool = False,
):
    """
    Unweighted absolute depth L1 term.

    Zeroes GT depth outside ``gt_mask`` *before* the subtract, so
    out-of-mask pixels contribute ``|pred_depth − 0|`` (containment
    pressure).  When ``mask_only`` is ``True`` the error is additionally
    masked to in-mask pixels (bg contribution dropped entirely) and the
    reduction is a **proper masked mean** — denominator is the count of
    pixels that actually contributed, not ``H·W``.  When ``valid_mask``
    is provided, the error is further restricted to dataset-valid pixels
    (e.g. depth holes masked out) and the denominator counts only
    in-mask AND in-valid pixels.

    Caller multiplies by the loss weight.  Shape contract: all tensors
    are ``(H, W)``; ``gt_mask`` and ``valid_mask`` are bool or float
    (cast internally).

    When ``return_error_map`` is true returns ``(scalar, error)`` where
    ``error`` is the per-pixel post-masking absolute error tensor that
    was actually summed into the scalar — for diagnostic visualizations
    that must mirror the loss pixel-for-pixel.

    This is the shared primitive for the ``absolute_l1`` depth mode used
    by both ``_compute_frame_loss`` (pose refinement) and the Stage-2
    rendering guidance (``core/rendering_guidance.py``).
    """
    gt_mask_f = gt_mask.float()
    gd_masked = gt_depth * gt_mask_f
    error = (pred_depth - gd_masked).abs()
    if valid_mask is not None:
        error = error * valid_mask.float()
    if mask_only:
        error = error * gt_mask_f
        if valid_mask is not None:
            denom = (gt_mask_f * valid_mask.float()).sum().clamp(min=1.0)
        else:
            denom = gt_mask_f.sum().clamp(min=1.0)
        scalar = error.sum() / denom
    else:
        scalar = error.mean()
    if return_error_map:
        return scalar, error
    return scalar


def _match_depth_grid(
    rendered: torch.Tensor,
    gt_hw: Tuple[int, int],
    *,
    K_matrix: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
):
    """Nearest-downsample a RENDERED depth/normals channel (and its accompanying
    render-space K/mask) down to ``gt_hw``.

    GT depth/normals/valid_mask are at the reconstruction backbone's (H', W'); when
    the render happens at another resolution, the depth and normals loss terms must
    compare at the GT's own grid rather than inventing detail that was never predicted.

    Nearest, NOT area-average: averaging would blend a soft-alpha render's
    silhouette-edge depth discontinuity into its interior before the loss sees it --
    and, on the normals branch, would blend two unit vectors into a non-unit
    "average" direction. Nearest also never spreads a NaN to its neighbours the way
    bilinear/area would, so an invalid depth pixel stays exactly one pixel invalid.

    Identity when ``rendered`` already matches ``gt_hw``, so this is a safe,
    unconditional call at every site. Nearest resizes correctly either way
    (block-replication on upsample, strided sampling on downsample).

    Parameters
    ----------
    rendered : torch.Tensor
        ``(H, W)`` depth or ``(H, W, 3)`` normals, at render resolution. Bool is
        accepted too (some callers reuse this function to resize a mask via the
        ``rendered`` slot, discarding the first return value) -- cast to float for
        the interpolation and back, which is lossless since nearest never blends.
    gt_hw : (int, int)
        Target ``(H, W)`` -- the GT depth/normals grid.
    K_matrix : torch.Tensor, optional
        ``(3, 3)`` at render resolution; rescaled by the pure per-axis ratio (same
        field of view as the render, so no offset correction is needed).
    mask : torch.Tensor, optional
        ``(H, W)`` bool at render resolution; nearest-resized and re-cast to bool.

    Returns
    -------
    (rendered_at_gt_hw, K_at_gt_hw_or_None, mask_at_gt_hw_or_None)
    """
    render_hw = tuple(rendered.shape[:2])
    gt_hw = (int(gt_hw[0]), int(gt_hw[1]))
    if render_hw == gt_hw:
        return rendered, K_matrix, mask

    import torch.nn.functional as F

    was_bool = rendered.dtype == torch.bool
    src = rendered.float() if was_bool else rendered
    if src.dim() == 2:
        rendered_d = F.interpolate(
            src[None, None], size=gt_hw, mode="nearest",
        )[0, 0]
    else:
        rendered_d = F.interpolate(
            src.permute(2, 0, 1)[None], size=gt_hw, mode="nearest",
        )[0].permute(1, 2, 0)
    if was_bool:
        rendered_d = rendered_d.bool()

    K_d = None
    if K_matrix is not None:
        K_d = K_matrix.clone()
        K_d[0, :] = K_d[0, :] * (gt_hw[1] / render_hw[1])
        K_d[1, :] = K_d[1, :] * (gt_hw[0] / render_hw[0])

    mask_d = None
    if mask is not None:
        mask_d = F.interpolate(
            mask.float()[None, None], size=gt_hw, mode="nearest",
        )[0, 0].bool()

    return rendered_d, K_d, mask_d


# ---------------------------------------------------------------------------
# Chamfer (3D shape alignment vs GT depth) — shared across pose-refine paths
# ---------------------------------------------------------------------------
def trim_and_subsample(pts, trim_factor, max_points, seed=None, *,
                       return_index: bool = False):
    """Outlier trim then optional cap on an ``(M, 3)`` cloud -- shared by BOTH ICP clouds.

    Shared so the observed and the rendered cloud get the same post-processing from the
    same code, keeping the two sides of a Chamfer comparable.

    ``bulk_keep_mask`` is a median-centred radius cut at ``trim_factor * p95``, so it keeps
    at least 95% of any cloud by construction and cannot empty the target.  The empty guard
    is mandatory, not defensive: ``median`` and ``quantile`` both raise on an empty cloud,
    and an empty cloud is a REAL path here (a mask landing entirely on invalid depth, or a
    pose that renders nothing) which callers handle by SKIPPING the frame.

    ``max_points`` None / <= 0 means NO cap: every kept pixel enters the cloud.  The cap is
    a real time budget -- ``_chamfer_gt_to_model`` runs a KNN per frame per ICP iteration
    (~4800 calls a solve at 300 steps x 16 frames) -- and also a SAMPLING bias: 4000 points
    over a full-resolution silhouette is a few percent, drawn uniformly over PIXELS, so thin
    structure (legs, tails) is represented in proportion to its image area rather than its
    importance.  Uncapped removes that bias and pays for it in wall clock.

    Does NOT detach: the rendered cloud must stay differentiable w.r.t. the pose.  Both
    steps are boolean-mask/index selections, so gradients reach the surviving points and
    the GT caller detaches for itself.

    ``return_index`` additionally returns the COMPOSED index into the input, so a caller
    can carry a per-point attribute (the ICP's depth-map normals) through the identical
    selection instead of re-deriving it.  Re-deriving is not merely redundant, it is
    WRONG: ``_subsample_idx`` is a ``randperm`` prefix, so the cap PERMUTES as well as
    subsets and the surviving order is not recoverable from the masks alone.
    """
    import torch

    idx = (torch.arange(pts.shape[0], device=pts.device)
           if return_index else None)
    if trim_factor is not None and float(trim_factor) > 0 and pts.shape[0] > 0:
        keep = bulk_keep_mask(pts, float(trim_factor))
        pts = pts[keep]
        if idx is not None:
            idx = idx[keep]
    if max_points and pts.shape[0] > int(max_points):
        take = _subsample_idx(pts.shape[0], int(max_points), pts.device, seed)
        pts = pts[take]
        if idx is not None:
            idx = idx[take]
    return (pts, idx) if return_index else pts


def _chamfer_gt_points_for_object(
    pointmap: torch.Tensor,
    obj_mask: torch.Tensor,
    max_points: Optional[int],
    valid_mask: Optional[torch.Tensor] = None,
    seed=None,
    trim_factor: Optional[float] = CHAMFER_GT_TRIM_DEFAULT,
    return_index: bool = False,
):
    """Object's visible GT surface points from the camera-space depth pointmap.

    ``mask ∩ valid ∩ trimmed ∩ finite-pointmap`` pixels → ``(M, 3)``, subsampled to
    ``max_points`` and detached (a fixed alignment target, no gradient).
    ``valid_mask`` (depth-reliability) is folded in so finite-but-unreliable
    depth (dense predictors flag it separately from NaN) doesn't pollute the
    target.  ``pointmap`` is ``(H, W, 3)`` in the same camera-space R3 frame the
    model is posed into.

    ``seed`` makes the subsample REPRODUCIBLE.  The cap is a real time budget --
    ``_chamfer_gt_to_model`` runs a KNN per frame per ICP iteration -- but the randomness
    only needs to avoid aliasing with raster order, which a seeded draw does just as well.
    Unseeded, two runs of the SAME config would optimise against different targets.  Pass
    something stable per (object, frame, view); ``None`` draws unseeded for callers that
    want a fresh sample.

    ``trim_factor`` drops BACKGROUND points, which is what the two masks above cannot do:
    a segmentation mask that bleeds onto the wall behind the subject contributes pixels
    whose depth is perfectly VALID and perfectly FINITE, just wrong -- ``valid_mask``
    carries ``recon_confidence_min`` / ``recon_mask_edges``, and both are about whether a
    depth measurement is TRUSTWORTHY, not about whether it belongs to this object.  The
    Chamfer that consumes this cloud is a mean of SQUARED distances, so a stray 100x
    further than the bulk carries 10,000x, and a handful of such points can dominate the
    gradient and inflate the object.

    The cut is :func:`rendering.bulk_keep_mask` -- a median-centred radius at
    ``factor * p95`` -- reused rather than reimplemented, and the only one of
    ``trim_outliers``' four cuts that needs no opacity or scale, which a depth cloud has
    not got.  The default factor (3.0) is looser than that function's own 1.5, so clean
    GT-depth clouds pass untouched.  Cleaning the cloud rather than reweighting the loss
    keeps the objective exactly L2, runs ONCE per frame instead of once per iteration (the
    observation is loop-invariant), and makes both the `icp_clouds/` dumps and the logged
    residual show what the solver optimises.

    Assumes the strays are a MINORITY: ``p95`` is by construction blind to a blob holding
    more than 5% of the masked pixels.  This is outlier rejection, NOT segmentation
    repair.  ``None`` (or <= 0) disables the trim.
    """
    sel = obj_mask.bool() & torch.isfinite(pointmap).all(dim=-1)
    if valid_mask is not None:
        sel = sel & valid_mask.bool()
    pts = pointmap[sel]
    if return_index:
        # Row-major `sel.nonzero()` order, which is exactly the order `pointmap[sel]`
        # produced, so composing the trim/cap index below lands each surviving point on
        # the pixel it came from.  The ICP's depth-map normals ride on this.
        pix = sel.nonzero(as_tuple=False)
        pts, idx = trim_and_subsample(pts, trim_factor, max_points, seed=seed,
                                      return_index=True)
        return pts.detach(), pix[idx]
    # BEFORE the subsample, so the cap spends its whole budget on real object points --
    # and the guard is mandatory, not defensive: `bulk_keep_mask` raises on an empty cloud
    # (`median` and `quantile` both do), while an empty target is a REAL path here (a mask
    # landing entirely on invalid depth) that `icp_frame_observation` handles by SKIPPING
    # the frame.  Unguarded this turns a handled skip into a crash on exactly the
    # bad-depth scenes it exists to rescue.
    return trim_and_subsample(pts, trim_factor, max_points, seed=seed).detach()


def _chamfer_gt_points_from_depth(
    gt_depth: np.ndarray,
    K_matrix: np.ndarray,
    obj_mask: Any,
    valid_mask: Any,
    max_points: Optional[int],
    device: torch.device,
    seed=None,
    trim_factor: Optional[float] = CHAMFER_GT_TRIM_DEFAULT,
    with_normals: bool = False,
):
    """Unproject GT depth → camera-space pointmap, then the object's visible
    surface points (``_chamfer_gt_points_for_object``).  Shared by the paths
    that carry a depth map rather than a precomputed pointmap.

    ``with_normals`` returns ``(points, normals)``: the SURFACE NORMAL at each kept point,
    taken from the depth map's own grid (``depth_to_normals``) rather than by PCA over the
    scattered cloud.  The grid is far better conditioned, and the points are carried
    through the identical trim/cap selection so a normal cannot land on the wrong point.
    A point whose normal is unreliable -- the 1px border, an eroded mask edge, a
    degenerate patch -- comes back as the ZERO vector, which the ICP's normal-compatibility
    gate rejects on its own; that is the intended behaviour, not a hole to fill.
    """
    from .depth import depth_to_pointmap
    pm = torch.from_numpy(depth_to_pointmap(gt_depth, K_matrix)).float().to(device)
    om = obj_mask if isinstance(obj_mask, torch.Tensor) else torch.from_numpy(np.asarray(obj_mask)).to(device)
    vm = None
    if valid_mask is not None:
        vm = valid_mask if isinstance(valid_mask, torch.Tensor) else torch.from_numpy(np.asarray(valid_mask)).to(device)
    if not with_normals:
        return _chamfer_gt_points_for_object(pm, om, max_points, valid_mask=vm,
                                             seed=seed, trim_factor=trim_factor)
    pts, pix = _chamfer_gt_points_for_object(
        pm, om, max_points, valid_mask=vm, seed=seed, trim_factor=trim_factor,
        return_index=True)
    # `depth_to_normals` needs a DENSE FINITE map: NaN spreads to the 4-neighbours through
    # the central differences and masking does not clean it (`NaN * 0 = NaN`).  Hence
    # `nan_to_num` + a 1px erode; the erode is what drops the contaminated ring rather
    # than trusting the mask alone.
    K_t = torch.as_tensor(np.asarray(K_matrix), dtype=torch.float32, device=device)
    valid = om.bool() if vm is None else (om.bool() & vm.bool())
    nmap = depth_to_normals(
        torch.nan_to_num(torch.as_tensor(np.asarray(gt_depth), dtype=torch.float32,
                                         device=device)),
        K_t, mask=_erode_bool_mask(valid, 1))
    return pts, nmap[pix[:, 0], pix[:, 1]].detach()


def _subsample_idx(n_points: int, max_points: int, device, seed=None):
    """``max_points`` indices out of ``n_points``, reproducibly when ``seed`` is given.

    One definition for both Chamfer draws (the GT target and the model subset), because
    they have the same requirement: bound the KNN cost without aliasing against the raster
    order the points arrive in, and give the SAME answer on a re-run so two configs can be
    compared at all.
    """
    g = None
    if seed is not None:
        g = torch.Generator(device="cpu").manual_seed(int(seed) & 0x7FFFFFFF)
        return torch.randperm(n_points, generator=g)[:max_points].to(device)
    return torch.randperm(n_points, device=device)[:max_points]


def _chamfer_model_subsample_idx(
    n_points: int, max_points: int, device: torch.device, seed=None
) -> Optional[torch.Tensor]:
    """Fixed random subset of model-point indices (stable across iterations), or ``None``
    to use all points.  Precompute once per object so the Chamfer target set doesn't jitter
    every step -- and pass ``seed`` so it does not jitter between RUNS either."""
    if n_points > max_points:
        return _subsample_idx(n_points, max_points, device, seed)
    return None


def _chamfer_gt_to_model(
    gt_points: torch.Tensor,
    model_points: torch.Tensor,
    model_sub_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """One-directional Chamfer: mean squared distance from each GT point to its
    nearest posed-model point.

    GT→model only — single-view GT depth is the visible front surface, so each
    observed point pulls toward its nearest model point and the model's unseen
    back is never penalized.  Differentiable w.r.t. ``model_points`` (hence the
    pose).  Returns a scalar (0 if either set is empty).

    A mean of SQUARES, so a target point 100x further than the bulk carries 10,000x --
    which is why the outliers are removed when the target CLOUD is built
    (``_chamfer_gt_points_for_object``'s ``trim_factor``) rather than reweighted here.
    """
    if gt_points.shape[0] == 0 or model_points.shape[0] == 0:
        return torch.zeros((), device=model_points.device)
    mp = model_points if model_sub_idx is None else model_points[model_sub_idx]
    # knn_points requires matching float dtypes; GT is float32 (see builders).
    mp = mp.float()
    from pytorch3d.ops import knn_points
    knn = knn_points(gt_points.unsqueeze(0), mp.unsqueeze(0), K=1)
    # knn.dists is squared distance (B, M, 1).
    return knn.dists.squeeze(0).squeeze(-1).mean()


def rendered_surface_keep(depth, alpha, K_t, *, alpha_threshold: float = 0.5):
    """The pixels that enter the rendered cloud, as a boolean ``(H, W)`` mask.

    Split out of :func:`rendered_surface_cloud` so an INSPECTION site (the `icp_clouds/`
    depth dump) can show exactly the pixels the loss consumed without restating
    ``alpha > 0.5`` for itself.  That restatement is the specific failure this file already
    warns about -- "a second ``alpha > 0.5`` at an inspection site would let the picture
    drift from the thing being optimised" -- so the mask is shared rather than recomputed.

    See :func:`rendered_surface_cloud` for why the alpha threshold cannot be relaxed to a
    finiteness test (gsplat returns a finite ``0.0`` at uncovered pixels).
    """
    return (alpha > alpha_threshold) & torch.isfinite(depth) & (depth > 0)


def rendered_surface_cloud(
    depth: torch.Tensor,
    alpha: torch.Tensor,
    uu: torch.Tensor,
    vv: torch.Tensor,
    K_t: torch.Tensor,
    *,
    alpha_threshold: float = 0.5,
    trim_factor: Optional[float] = None,
    return_index: bool = False,
):
    """The rendered object's visible surface as a camera-space cloud, or ``None``.

    Built by the SAME steps as the observed cloud (``_chamfer_gt_points_for_object``) --
    coverage mask, one shared unprojection
    (``depth.unproject_z_depth``), then ``trim_and_subsample`` -- so the only asymmetry
    between the two sides of the ICP Chamfer is which mask the caller supplies: the object
    segmentation there, rendered coverage here.

    THE definition of "what the ICP term matches against", so the ICP solver and anything
    inspecting that match (the debug cloud dump) read one implementation.  A
    second ``alpha > 0.5`` at an inspection site would let the picture drift from the thing
    being optimised.

    ``keep = alpha > alpha_threshold`` cannot be relaxed to a finiteness test.  gsplat
    renders expected depth as ``accum_depth / alpha.clamp(min=1e-10)`` (``RGB+ED``, see
    ``rendering.render_gaussian_params``), so an UNCOVERED pixel comes back as exactly
    ``0.0`` -- finite, not NaN.  Unprojecting ``z = 0`` puts the point at the CAMERA CENTRE,
    and in a mean-of-SQUARES Chamfer a cloud of those swamps the objective -- worse the more
    wrong the pose is, since more of the frame goes uncovered.  The threshold is the only
    thing rejecting them.

    ``keep`` is a HARD mask throughout -- no gradient reaches the silhouette -- while the
    kept pixels' depth stays differentiable, hence so does the pose.

    Note the point COUNT grows with the object's rendered area: one point per kept pixel,
    not a fixed budget like the GT side's ``max_gt_points``.
    """
    from .depth import unproject_z_depth

    keep = rendered_surface_keep(depth, alpha, K_t, alpha_threshold=alpha_threshold)
    if not bool(keep.any()):
        return (None, None, keep) if return_index else None
    if return_index:
        # The cloud, its index into the KEEP-ordered unprojection, AND the keep mask --
        # so a caller holding a dense per-pixel map writes `dense[keep][idx]` and lands on
        # the right point even under a trim, without recomputing `keep`.
        pts, idx = trim_and_subsample(unproject_z_depth(depth, keep, K_t, uu, vv),
                                      trim_factor, None, return_index=True)
        return pts, idx, keep
    return trim_and_subsample(unproject_z_depth(depth, keep, K_t, uu, vv),
                              trim_factor, None)


def _compute_multiscale_rgb_loss(
    rgb: torch.Tensor,
    gt_image: torch.Tensor,
    losses: "LossConfig",
) -> torch.Tensor:
    """
    Compute multi-scale RGB loss for robust pose optimization.

    Multi-scale loss helps escape local minima by computing the loss at
    multiple resolutions. Lower resolutions capture global alignment errors
    while higher resolutions refine fine details.

    Parameters
    ----------
    rgb : torch.Tensor
        Rendered RGB image, shape (H, W, 3), values in [0, 1].
    gt_image : torch.Tensor
        Ground truth RGB image with background masked to black,
        shape (H, W, 3), values in [0, 1].
    losses : LossConfig
        Configuration with rgb_multiscale_scales and rgb_multiscale_weights.

    Returns
    -------
    torch.Tensor
        Scalar RGB loss averaged across all scales.

    Notes
    -----
    L1 per scale.  Downsampling uses area interpolation for anti-aliasing.
    """
    import torch.nn.functional as F

    device = rgb.device
    total_loss = torch.tensor(0.0, device=device)

    # Prepare tensors for interpolation: (H, W, C) -> (1, C, H, W)
    rgb_nchw = rgb.permute(2, 0, 1).unsqueeze(0)
    gt_nchw = gt_image.permute(2, 0, 1).unsqueeze(0)

    for scale, weight in zip(losses.rgb_multiscale_scales, losses.rgb_multiscale_weights):
        if scale == 1.0:
            # Full resolution - no interpolation needed
            rgb_scaled = rgb
            gt_scaled = gt_image
        else:
            # Downsample using area interpolation (anti-aliased)
            rgb_scaled = F.interpolate(
                rgb_nchw, scale_factor=scale, mode="area"
            ).squeeze(0).permute(1, 2, 0)
            gt_scaled = F.interpolate(
                gt_nchw, scale_factor=scale, mode="area"
            ).squeeze(0).permute(1, 2, 0)

        diff = (rgb_scaled - gt_scaled).abs()

        # Mean over all pixels (full image comparison)
        n_pixels = rgb_scaled.shape[0] * rgb_scaled.shape[1]
        scale_loss = diff.sum() / (n_pixels * 3)

        total_loss = total_loss + weight * scale_loss

    return total_loss


def apply_pose_to_gaussian(
    canonical_gs: "Gaussian",
    rotation: torch.Tensor,
    translation: torch.Tensor,
    scale: torch.Tensor,
    *,
    means_override: "torch.Tensor | None" = None,
    rotation_override: "torch.Tensor | None" = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply a pose transformation to Gaussian positions and rotations.

    Parameters
    ----------
    canonical_gs : Gaussian
        The canonical Gaussian object (frozen, not modified).
    rotation : torch.Tensor
        Quaternion rotation (1, 4) or (4,).
    translation : torch.Tensor
        Translation (1, 3) or (3,).
    scale : torch.Tensor
        Scale factor (1, 3) or (3,) or (1,) or scalar.
    means_override : torch.Tensor or None, keyword-only
        When provided, used in place of ``canonical_gs.get_xyz`` for the
        pose math.  Shape ``(N, 3)``.  Used by the keyframes renderer to
        inject per-frame warped Gaussian means without bypassing this
        function.  Other attributes (rotation, scaling, opacity, features)
        still come from ``canonical_gs`` unless ``rotation_override`` is
        also set.
    rotation_override : torch.Tensor or None, keyword-only
        When provided, used in place of ``canonical_gs.get_rotation``.
        Shape ``(N, 4)`` wxyz.  Companion to ``means_override``.

    Returns
    -------
    tuple
        (transformed_means, transformed_quats, scales, opacities, features)
    """
    # Get canonical Gaussian attributes (overrides applied selectively).
    xyz_local = (
        means_override if means_override is not None else canonical_gs.get_xyz
    )                                                                # (N, 3)
    rot_local = (
        rotation_override
        if rotation_override is not None
        else canonical_gs.get_rotation
    )                                                                # (N, 4)
    scales_local = canonical_gs.get_scaling  # (N, 3)
    opacities = canonical_gs.get_opacity  # (N, 1)
    features = canonical_gs.get_features  # (N, 1, 3) or (N, K, 3)

    # Ensure rotation is (4,)
    if rotation.dim() == 2:
        rotation = rotation.squeeze(0)

    # Ensure translation is (3,)
    if translation.dim() == 2:
        translation = translation.squeeze(0)

    # Ensure scale is (3,)
    if scale.dim() == 0:
        scale = scale.expand(3)
    elif scale.dim() == 1 and scale.shape[0] == 1:
        scale = scale.expand(3)
    elif scale.dim() == 2:
        scale = scale.squeeze(0)
        if scale.shape[0] == 1:
            scale = scale.expand(3)

    # Normalize quaternion
    rotation = rotation / rotation.norm()

    # Convert quaternion to rotation matrix
    # PyTorch3D convention: quaternion_to_matrix gives R such that points are transformed as p @ R
    R = quaternion_to_matrix(rotation.unsqueeze(0)).squeeze(0)  # (3, 3)

    # Transform positions following compose_transform convention:
    # tfm = Scale(scale).compose(Rotate(R)).compose(Translate(trans))
    # transformed = points * scale @ R + trans
    scaled_xyz = xyz_local * scale
    transformed_xyz = torch.mm(scaled_xyz, R) + translation  # Note: @ R, not @ R.T

    # Transform rotations: rot_world = quaternion_multiply(rotation_inv, rot_local)
    # Note: Using inverse because of the convention in make_scene
    rotation_inv = quaternion_invert(rotation.unsqueeze(0)).squeeze(0)
    transformed_rot = quaternion_multiply(
        rotation_inv.unsqueeze(0).expand(rot_local.shape[0], -1), rot_local
    )

    # Transform scales
    transformed_scales = scales_local * scale

    return transformed_xyz, transformed_rot, transformed_scales, opacities, features


def _transform_object_to_r3(
    canonical_gs: "Gaussian",
    rotation: torch.Tensor,
    translation: torch.Tensor,
    scale: torch.Tensor,
    device: torch.device,
    *,
    means_override: "torch.Tensor | None" = None,
    rotation_override: "torch.Tensor | None" = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply pose to canonical Gaussians and transform to R3 coordinate system.

    Returns raw tensor tuples ready for rendering or concatenation across objects.

    Parameters
    ----------
    canonical_gs : Gaussian
        The canonical Gaussian object.
    rotation : torch.Tensor
        Quaternion rotation.
    translation : torch.Tensor
        Translation vector.
    scale : torch.Tensor
        Scale factor.
    device : torch.device
        Device to use.
    means_override, rotation_override : torch.Tensor or None, keyword-only
        Forwarded to :func:`apply_pose_to_gaussian`.  Used by the actionmesh
        FINETUNE path to pass already-warped Gaussian means / quats from
        ``warp_gaussians_high_res``.  Identity-default leaves the canonical
        means / rotations untouched.

    Returns
    -------
    tuple
        (transformed_xyz, transformed_rot, transformed_scales, opacities, features)
    """
    # Apply pose to canonical Gaussian
    transformed_xyz, transformed_rot, transformed_scales, opacities, features = (
        apply_pose_to_gaussian(
            canonical_gs=canonical_gs, rotation=rotation,
            translation=translation, scale=scale,
            means_override=means_override,
            rotation_override=rotation_override,
        )
    )

    # Transform from PyTorch3D convention to R3 convention
    # P3D↔R3 is diag(-1,-1,1): negate X and Y for positions, apply 180° Z rotation for quaternions
    transformed_xyz = p3d_to_r3_positions(transformed_xyz)
    transformed_rot = p3d_to_r3_quaternions(transformed_rot)

    # Handle opacities
    opacities_flat = opacities.squeeze(-1) if opacities.dim() == 2 else opacities

    return transformed_xyz, transformed_rot, transformed_scales, opacities_flat, features


def _concat_fg_bg_params(
    fg_xyz, fg_rot, fg_scales, fg_opac, fg_feats, bg_params, device,
):
    """Concatenate a DETACHED background Gaussian tuple onto the foreground params
    for a full-image render, matching the foreground SH layout/degree first.

    ``bg_params`` is ``(xyz, rot, scales, opac, feats)`` in the SAME (camera)
    space as the foreground.  Returns the concatenated ``(xyz, rot, scales, opac,
    feats)``.  Mirrors the fg+bg concat in ``refine_poses_global_composite``.
    """
    bg_xyz, bg_rot, bg_scales, bg_opac, bg_feats = bg_params
    if fg_feats.dim() == 2:
        fg_feats = fg_feats.unsqueeze(1)
    if bg_feats.dim() == 2:
        bg_feats = bg_feats.unsqueeze(1)
    if fg_feats.shape[1] > bg_feats.shape[1]:
        _pad = torch.zeros(
            bg_feats.shape[0], fg_feats.shape[1] - bg_feats.shape[1], 3,
            device=device, dtype=bg_feats.dtype,
        )
        bg_feats = torch.cat([bg_feats, _pad], dim=1)
    return (
        torch.cat([fg_xyz, bg_xyz], dim=0),
        torch.cat([fg_rot, bg_rot], dim=0),
        torch.cat([fg_scales, bg_scales], dim=0),
        torch.cat([fg_opac, bg_opac], dim=0),
        torch.cat([fg_feats, bg_feats], dim=0),
    )


def _render_frame_with_pose(
    canonical_gs: "Gaussian",
    rotation: torch.Tensor,
    translation: torch.Tensor,
    scale: torch.Tensor,
    K_matrix: np.ndarray,
    W: int,
    H: int,
    device: torch.device,
    bg_color: Optional[torch.Tensor] = None,
    *,
    means_override: "torch.Tensor | None" = None,
    rotation_override: "torch.Tensor | None" = None,
    background_params: "Tuple[torch.Tensor, ...] | None" = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Render a single frame with given pose parameters.

    Assumes poses are in camera space; renders with identity c2w.

    Parameters
    ----------
    canonical_gs : Gaussian
        The canonical Gaussian object.
    rotation : torch.Tensor
        Quaternion rotation (camera space).
    translation : torch.Tensor
        Translation vector (camera space).
    scale : torch.Tensor
        Scale factor.
    K_matrix : np.ndarray
        Camera intrinsics.
    W : int
        Image width.
    H : int
        Image height.
    device : torch.device
        Device to use.
    bg_color : torch.Tensor or None, optional
        Background color (3,). Defaults to black (0,0,0).
    means_override, rotation_override : torch.Tensor or None, keyword-only
        Forwarded to :func:`_transform_object_to_r3` /
        :func:`apply_pose_to_gaussian` — used by the actionmesh FINETUNE
        path to inject warped Gaussian means + quats from
        ``warp_gaussians_high_res`` before applying Stage-1 pose.

    Returns
    -------
    tuple
        (rgb, alpha, depth) where rgb is (H, W, 3) and alpha is (H, W), depth is (H, W).
    """
    xyz, rot, scales, opac, feats = _transform_object_to_r3(
        canonical_gs, rotation, translation, scale, device,
        means_override=means_override,
        rotation_override=rotation_override,
    )

    # Optional DETACHED background (same camera space) for full-image depth:
    # concat behind the posed foreground so the render carries valid depth
    # outside the object.  The bg carries no gradient (pose grads flow only
    # through the foreground); the composite depth is fg where the object is,
    # bg elsewhere.
    if background_params is not None:
        xyz, rot, scales, opac, feats = _concat_fg_bg_params(
            xyz, rot, scales, opac, feats, background_params, device,
        )

    c2w_tensor = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)
    if bg_color is None:
        bg_color = torch.zeros(3, device=device)

    return render_gaussian_params(
        xyz, rot, scales, opac, feats,
        c2w_tensor, K_matrix, W, H, bg_color=bg_color,
    )


class PosedObjectRenderer(NamedTuple):
    """One canonical object behind a representation-agnostic render interface.

    Built by :func:`make_posed_object_renderer`. Members:

    ``render(q, t, s, K, W, H, device, bg_color=None, background_params=None, *,
    means_override=None, rotation_override=None)``
        Rasterise the object under one camera-space Sim(3) → ``(rgb (H,W,3),
        alpha (H,W), depth (H,W))``. Differentiable w.r.t. ``q``/``t``/``s``.
        The two overrides mean: *use this
        pre-warped geometry instead of the canonical* — how a DEFORMING object is
        rendered at one frame. Both ``None`` is the static canonical, bit-identical
        to a call that omits them.
    ``model_points(q, t, s, device, *, means_override=None)``
        The posed object's points in R3 camera space ``(N, 3)`` (Chamfer target).
        Takes the same override, and a caller that warps ``render`` must pass it
        here too — otherwise the Chamfer term scores the undeformed surface while
        the photometric terms score the deformed one.
    ``num_model_points()``
        ``N`` for the above, computed lazily (``Gaussian.get_xyz`` recomputes).
    """

    render: Any
    model_points: Any
    num_model_points: Any


def make_posed_object_renderer(source: Any, device: torch.device) -> PosedObjectRenderer:
    """Adapt a canonical reconstruction (a repo ``Gaussian``) to
    :class:`PosedObjectRenderer`, so pose optimisation is written once.

    Renders through gsplat via :func:`_render_frame_with_pose` (the call is forwarded
    unchanged, so numerics are identical).
    """
    def render(q, t, s, K, W, H, dev, bg_color=None, background_params=None,
               *, means_override=None, rotation_override=None):
        return _render_frame_with_pose(
            source, q, t, s, K, W, H, dev,
            bg_color=bg_color, background_params=background_params,
            means_override=means_override, rotation_override=rotation_override,
        )

    def model_points(q, t, s, dev, *, means_override=None):
        return _transform_object_to_r3(
            source, q, t, s, dev, means_override=means_override)[0]

    return PosedObjectRenderer(
        render, model_points, lambda: int(source.get_xyz.shape[0]),
    )


def depth_to_normals(
    depth: torch.Tensor,
    K_matrix: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute surface normals from a z-depth map via finite differences.

    Unprojects z-depth to 3D camera-space points using the pinhole camera
    model, then computes cross product of horizontal and vertical tangent
    vectors from central finite differences.

    Both gsplat rendered depth (``RGB+ED`` mode, which outputs expected
    z-depth ``sum(w_i * z_i) / sum(w_i)``) and MoGe/GT depth maps are
    z-depth (distance along optical axis), so no conversion is needed.

    Parameters
    ----------
    depth : torch.Tensor
        Z-depth map (H, W), distance along optical axis.
    K_matrix : torch.Tensor
        Camera intrinsics (3, 3).
    mask : torch.Tensor, optional
        Boolean mask (H, W). Normals outside mask are zeroed.

    Returns
    -------
    torch.Tensor
        Unit surface normals (H, W, 3). Zero at boundaries and outside mask.
        Normals point toward camera for front-facing surfaces (negative z
        in R3 convention where Z points into the scene).
    """
    H, W = depth.shape
    device = depth.device

    # Pixel grid
    v_coords, u_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )

    fx, fy = K_matrix[0, 0], K_matrix[1, 1]
    cx, cy = K_matrix[0, 2], K_matrix[1, 2]

    x_norm = (u_coords - cx) / fx
    y_norm = (v_coords - cy) / fy

    # Unproject z-depth to 3D camera-space points via pinhole model
    points = torch.stack([x_norm * depth, y_norm * depth, depth], dim=-1)

    # Central finite differences for tangent vectors
    du = torch.zeros_like(points)
    dv = torch.zeros_like(points)
    du[1:-1, 1:-1] = points[1:-1, 2:] - points[1:-1, :-2]
    dv[1:-1, 1:-1] = points[2:, 1:-1] - points[:-2, 1:-1]

    # Normal = dv x du (normals point toward camera for front-facing surfaces)
    normals = torch.linalg.cross(dv, du)
    normals = normals / normals.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    # Zero out boundary pixels (finite differences invalid there)
    normals[0, :] = 0
    normals[-1, :] = 0
    normals[:, 0] = 0
    normals[:, -1] = 0

    if mask is not None:
        normals = normals * mask.unsqueeze(-1).float()

    return normals


def _erode_bool_mask(mask: torch.Tensor, px: int) -> torch.Tensor:
    """Binary erosion by a ``(2*px+1)`` square.  ``mask`` is ``(H, W)``.

    A pixel survives only if every pixel in its kxk neighborhood is True.
    Implemented as dilation of the inverted mask via max_pool2d so it runs
    on GPU without pulling in scipy / skimage.  ``px <= 0`` is a no-op.
    """
    if px <= 0:
        return mask
    inv = (~mask).float()[None, None]
    k = 2 * px + 1
    dilated_inv = torch.nn.functional.max_pool2d(inv, k, stride=1, padding=px)
    return dilated_inv.squeeze(0).squeeze(0) < 0.5


def _compute_depth_discontinuity_mask(
    depth: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Compute a boolean mask that is False at depth discontinuities.

    A pixel is marked as discontinuous if the maximum absolute difference
    between it and any of its 4-connected neighbours exceeds ``threshold``
    times the local depth value. This relative criterion adapts to both
    near and far regions.

    Parameters
    ----------
    depth : torch.Tensor
        Z-depth map (H, W).
    threshold : float
        Relative depth-gradient threshold. Typical value: 0.02 (2%).

    Returns
    -------
    torch.Tensor
        Boolean mask (H, W), True where depth is smooth (no discontinuity).
    """
    # Pad depth to handle boundaries (replicate edges)
    z_pad = torch.nn.functional.pad(depth.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="replicate")
    z_pad = z_pad.squeeze(0).squeeze(0)

    # Differences to 4-connected neighbours
    local = z_pad[1:-1, 1:-1]
    local_safe = local.clamp(min=1e-6)
    diff_l = (local - z_pad[1:-1, :-2]).abs() / local_safe
    diff_r = (local - z_pad[1:-1, 2:]).abs() / local_safe
    diff_u = (local - z_pad[:-2, 1:-1]).abs() / local_safe
    diff_d = (local - z_pad[2:, 1:-1]).abs() / local_safe

    max_diff = torch.max(torch.max(diff_l, diff_r), torch.max(diff_u, diff_d))
    return max_diff <= threshold


def _compute_frame_loss(
    rgb: torch.Tensor,
    alpha: torch.Tensor,
    gt_image: torch.Tensor,
    mask: torch.Tensor,
    losses: LossConfig,
    sdt: Optional[torch.Tensor] = None,
    rendered_depth: Optional[torch.Tensor] = None,
    gt_depth: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
    bg_color: Optional[torch.Tensor] = None,
    perceptual_scale: float = 1.0,
    K_matrix: Optional[torch.Tensor] = None,
    return_pixelwise: bool = False,
    has_background: bool = False,
    depth_mask_only: Optional[bool] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute loss for a single frame.

    The loss consists of:
    - RGB loss: L1 between rendered and GT image, optionally multi-scale
    - SSIM loss: Structural similarity loss (optional, weight > 0 to enable)
    - Perceptual loss: LPIPS (VGG) between rendered and masked GT (optional)
    - Silhouette loss: Combined loss for pose-aware silhouette alignment
      - Center-of-mass: Aligns centroids (guides translation)
      - SDT: Signed distance transform (guides all params, escapes local minima)
      - IoU: Soft intersection-over-union (fine shape matching)
    - Depth loss: L1 or scale-shift invariant between rendered and GT/estimated depth (optional)

    GT background is masked to match the rendering background color for all
    appearance losses (RGB, SSIM, perceptual).

    Parameters
    ----------
    rgb : torch.Tensor
        Rendered RGB image (H, W, 3).
    alpha : torch.Tensor
        Rendered alpha/opacity (H, W).
    gt_image : torch.Tensor
        Ground truth image (H, W, 3).
    mask : torch.Tensor
        Object mask (H, W), boolean.
    losses : LossConfig
        Per-phase loss weights and RGB loss settings.
    sdt : torch.Tensor, optional
        Precomputed signed distance transform of the mask, shape (H, W).
        If None and silhouette_sdt_weight > 0, it will be computed here.
    rendered_depth : torch.Tensor, optional
        Rendered depth map (H, W). Required when depth_weight > 0.
    gt_depth : torch.Tensor, optional
        Ground truth depth map (H, W). Required when depth_weight > 0.
    K_matrix : torch.Tensor, optional
        Camera intrinsics (3, 3). Required when normals_weight > 0
        (used to unproject rendered depth to 3D for normal computation).
    valid_mask : torch.Tensor, optional
        MoGe/sensor depth validity mask (H, W), boolean.
    bg_color : torch.Tensor, optional
        Background color (3,). Used to mask GT image background.
    perceptual_scale : float
        Scale factor for perceptual loss input resolution.

    Returns
    -------
    dict
        Dictionary with loss tensors:
        - 'rgb_loss': Scalar RGB loss (multi-scale if enabled)
        - 'ssim_loss': Scalar SSIM loss
        - 'perceptual_loss': LPIPS perceptual loss (0 if disabled)
        - 'silhouette_com': Center-of-mass loss component
        - 'silhouette_sdt': SDT loss component
        - 'silhouette_iou': IoU loss component
        - 'depth_loss': Scalar depth loss (0 if disabled)
        - 'normals_loss': Scalar normals loss (0 if disabled)
    """
    device = rgb.device
    H, W = rgb.shape[:2]

    # Mask GT background to match rendering background color
    mask_f = mask.float().unsqueeze(-1)
    if bg_color is not None:
        gt_masked = gt_image * mask_f + bg_color.view(1, 1, 3) * (1.0 - mask_f)
    else:
        gt_masked = gt_image * mask_f

    # Per-pixel debug maps (populated only when return_pixelwise=True)
    px = {}
    if return_pixelwise:
        px["px_rgb_error"] = (rgb - gt_masked).abs().mean(dim=-1).detach().cpu().numpy()
        px["px_alpha"] = (alpha.squeeze(0) if alpha.dim() == 3 else alpha).detach().cpu().numpy()

    # RGB loss: L1, optionally multi-scale
    if losses.rgb_multiscale:
        # Multi-scale loss for robust optimization
        rgb_loss_value = _compute_multiscale_rgb_loss(rgb, gt_masked, losses)
    else:
        # Standard single-scale L1 loss
        rgb_loss_value = _l1_loss(rgb, gt_masked)

    # SSIM loss (optional, enabled when weight > 0)
    ssim_loss_value = torch.tensor(0.0, device=device)
    if losses.rgb_ssim_weight > 0:
        ssim_loss_value = _compute_ssim_loss(rgb, gt_masked)

    # Perceptual loss (LPIPS, optional)
    perceptual_loss_value = torch.tensor(0.0, device=device)
    if losses.perceptual_weight > 0:
        from .model_cache import ModelCache
        rgb_bchw = rgb.permute(2, 0, 1).unsqueeze(0)
        gt_bchw = gt_masked.permute(2, 0, 1).unsqueeze(0)
        # Crop to the object before any downscale: VGG then sees only the
        # object, not the (white) background.  `mask` matches the un-scaled
        # (H, W) here, so crop-then-scale keeps mask/tensor resolution aligned.
        if losses.perceptual_crop_to_mask:
            from .perceptual_loss import crop_to_mask_bbox
            rgb_bchw, gt_bchw = crop_to_mask_bbox(
                rgb_bchw, gt_bchw, mask, losses.perceptual_crop_margin)
        if perceptual_scale < 1.0:
            rgb_bchw = torch.nn.functional.interpolate(
                rgb_bchw, scale_factor=perceptual_scale,
                mode="bilinear", align_corners=False)
            gt_bchw = torch.nn.functional.interpolate(
                gt_bchw, scale_factor=perceptual_scale,
                mode="bilinear", align_corners=False)
        # Optional bf16 autocast on the LPIPS/VGG forward to halve its retained
        # activation memory; `.float()` restores fp32 for the downstream
        # weighting/sum. enabled=False is a no-op (and stays a no-op under the
        # rendering-guidance autocast(enabled=False) caller).
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=losses.perceptual_autocast_bf16):
            perceptual_loss_value = ModelCache.get().perceptual_model(
                rgb_bchw, gt_bchw).float()

    # Silhouette loss components
    alpha_squeezed = alpha.squeeze(0) if alpha.dim() == 3 else alpha

    # Initialize loss components
    com_loss_value = torch.tensor(0.0, device=device)
    sdt_loss_value = torch.tensor(0.0, device=device)
    iou_loss_value = torch.tensor(0.0, device=device)

    if losses.silhouette_weight > 0:

        # Center-of-mass loss: directly guides translation
        if losses.silhouette_com_weight > 0:
            com_loss_value = _compute_center_of_mass_loss(alpha_squeezed, mask, occlusion_robust=losses.occlusion_robust_silhouette)

        # Signed distance transform loss: guides all params, escapes local minima
        if losses.silhouette_sdt_weight > 0:
            if sdt is None:
                sdt = _compute_signed_distance_transform(mask)
            sdt_loss_value = _compute_sdt_loss(alpha_squeezed, sdt, occlusion_robust=losses.occlusion_robust_silhouette)

        # Soft IoU loss: fine-grained shape matching
        if losses.silhouette_iou_weight > 0:
            iou_loss_value = _compute_soft_iou_loss(alpha_squeezed, mask, occlusion_robust=losses.occlusion_robust_silhouette)

    # Everything below this point is depth/normals: GT depth/normals are at the
    # reconstruction backbone's resolution, so when the render is at a different
    # resolution, shadow rendered_depth/K_matrix/mask with their GT-grid counterparts
    # for the rest of this function.  A no-op whenever the shapes already match.  This
    # also keeps `rendered_normals` (below) at the SAME grid as `target_normals`, so
    # `rendered_normals[normals_valid]` indexes a tensor of the mask's shape.
    if gt_depth is not None and rendered_depth is not None:
        rendered_depth, K_matrix, mask = _match_depth_grid(
            rendered_depth, gt_depth.shape[-2:], K_matrix=K_matrix, mask=mask)

    # Depth loss (optional, enabled when weight > 0)
    # Only GT depth is masked with the object mask (background → 0).
    # valid_mask (MoGe reliability) is used to mask the error, not the inputs.
    depth_loss_value = torch.tensor(0.0, device=device)
    if losses.depth_weight > 0 and rendered_depth is not None and gt_depth is not None:
        # Partial / sparse GT depth (e.g. a mesh-rasterized depth that is NaN
        # outside the silhouette) would poison the loss: masking can't remove NaNs
        # because NaN * 0 = NaN.  Fold finiteness into valid_mask and zero the NaNs
        # so masked-out pixels contribute 0.  No-op when GT depth is already dense
        # (GSO / MoGe / map-anything cover every pixel).
        _finite = torch.isfinite(gt_depth)
        if not bool(_finite.all()):
            valid_mask = _finite if valid_mask is None else (valid_mask & _finite)
            gt_depth = torch.nan_to_num(gt_depth, nan=0.0)
        # Mask GT depth with object mask (same convention as RGB masked to black)
        gt_depth_masked = gt_depth * mask.float()
        # Per-call override wins over the config default: the shared-world MV path
        # forces masked depth on frames whose fused background came out empty (no
        # valid depth outside the object → full-image supervision would be noise).
        _depth_mask_only = (
            losses.depth_mask_only if depth_mask_only is None else depth_mask_only
        )
        # depth_mask_only=False supervises depth over the full image; that is only
        # well-defined when the render covers every GT depth pixel — i.e. a
        # background model (pointmap Gaussians or equivalent) is rendered alongside
        # the foreground. Without it, outside-mask pixels have no valid rendered
        # depth and the loss becomes noise.
        assert _depth_mask_only or has_background, (
            "losses.depth_mask_only=False requires has_background=True: "
            "full-image depth supervision needs a background model to produce "
            "valid depth outside the GT mask. Either set depth_mask_only=true, "
            "or render a background and pass has_background=True."
        )
        if losses.depth_loss_type == "scale_shift_invariant":
            # Scale-and-shift invariant depth loss for monocular depth (e.g. MoGe).
            # Fit (s, t) on reliable pixels (mask & valid_mask), then compute
            # error on mask-only or full image depending on depth_mask_only.
            fit_mask = mask & valid_mask if valid_mask is not None else mask
            rd_fit = rendered_depth[fit_mask]
            gd_fit = gt_depth[fit_mask]
            if gd_fit.numel() > 1:
                with torch.no_grad():
                    g_mean = gd_fit.mean()
                    d_mean = rd_fit.mean()
                    g_var = ((gd_fit - g_mean) ** 2).mean()
                    if g_var > 1e-8:
                        g_cov = ((gd_fit - g_mean) * (rd_fit - d_mean)).mean()
                        s = g_cov / g_var
                        t = d_mean - s * g_mean
                    else:
                        s = torch.ones(1, device=device)
                        t = d_mean - g_mean
                mask_f = mask.float()
                aligned_gt = (s * gt_depth + t) * mask_f
                error = (rendered_depth - aligned_gt).abs()
                if valid_mask is not None:
                    error = error * valid_mask.float()
                if _depth_mask_only:
                    error = error * mask_f
                    # Proper masked mean — denominator counts contributing
                    # pixels (in-mask AND in-valid), not H·W.
                    if valid_mask is not None:
                        _denom = (mask_f * valid_mask.float()).sum().clamp(min=1.0)
                    else:
                        _denom = mask_f.sum().clamp(min=1.0)
                    depth_loss_value = error.sum() / _denom
                else:
                    depth_loss_value = error.mean()
                if return_pixelwise:
                    px["px_depth_error"] = error.detach().cpu().numpy()
        else:
            # Absolute L1 depth loss (for GT depth with known scale).
            # Delegates to ``_depth_l1_term`` — the shared primitive also
            # used by the Stage-2 rendering guidance.  When
            # depth_mask_only=False, rendered depth outside
            # the mask is penalized against 0 (silhouette-like leakage
            # penalty); when True, only mask pixels contribute.
            depth_loss_value = _depth_l1_term(
                rendered_depth, gt_depth, mask,
                mask_only=_depth_mask_only,
                valid_mask=valid_mask,
            )
            if return_pixelwise:
                # Re-compute the unreduced error map for the pixelwise
                # debug plot (``_depth_l1_term`` returns the scalar mean).
                err = (rendered_depth - gt_depth_masked).abs()
                if valid_mask is not None:
                    err = err * valid_mask.float()
                if _depth_mask_only:
                    err = err * mask.float()
                px["px_depth_error"] = err.detach().cpu().numpy()

    # Surface normals loss (optional, enabled when normals_weight > 0).
    # Pseudo-GT normals are derived from gt_depth via finite differences (always
    # consistent with the pointmap).  Discontinuity masking: exclude pixels near large depth gradients where
    # finite-difference normals are unreliable.
    normals_loss_value = torch.tensor(0.0, device=device)
    if losses.normals_weight > 0 and rendered_depth is not None and K_matrix is not None:
        # Rendered normals (from gsplat z-depth, differentiable)
        rendered_normals = depth_to_normals(rendered_depth, K_matrix, mask=mask)

        # Pseudo-GT normals from gt_depth (z-depth)
        target_normals = None
        if gt_depth is not None:
            with torch.no_grad():
                target_normals = depth_to_normals(gt_depth, K_matrix, mask=mask)

        if target_normals is not None:
            # Valid region: object mask, non-zero target normals, interior pixels
            target_norm = target_normals.norm(dim=-1)
            normals_valid = mask & (target_norm > 0.5)
            normals_valid[0, :] = False
            normals_valid[-1, :] = False
            normals_valid[:, 0] = False
            normals_valid[:, -1] = False
            if valid_mask is not None:
                normals_valid = normals_valid & valid_mask

            # Discontinuity masking: exclude pixels near large depth gradients
            if losses.normals_discontinuity_threshold > 0 and gt_depth is not None:
                disc_mask = _compute_depth_discontinuity_mask(
                    gt_depth, losses.normals_discontinuity_threshold,
                )
                normals_valid = normals_valid & disc_mask

            if normals_valid.sum() > 0:
                rn = rendered_normals[normals_valid]  # (M, 3)
                gn = target_normals[normals_valid]    # (M, 3)
                # Sign-invariant cosine: 1 - |cos(angle)|
                cos_sim = (rn * gn).sum(dim=-1)
                normals_loss_value = (1.0 - cos_sim.abs()).mean()
                if return_pixelwise:
                    # normals_valid/mask may be at the GT-depth grid (render
                    # resampled by _match_depth_grid above), not (H, W).
                    px_normals = torch.zeros(mask.shape, device=device)
                    px_normals[normals_valid] = (1.0 - cos_sim.abs()).detach()
                    px["px_normals_error"] = px_normals.cpu().numpy()

    result = {
        "rgb_loss": rgb_loss_value,  # Scalar RGB loss
        "ssim_loss": ssim_loss_value,  # Scalar SSIM loss (0 if disabled)
        "perceptual_loss": perceptual_loss_value,  # LPIPS perceptual loss (0 if disabled)
        "silhouette_com": com_loss_value,  # Center-of-mass loss component (0 if disabled)
        "silhouette_sdt": sdt_loss_value,  # SDT loss component (0 if disabled)
        "silhouette_iou": iou_loss_value,  # IoU loss component (0 if disabled)
        "depth_loss": depth_loss_value,   # Scalar depth loss (0 if disabled)
        "normals_loss": normals_loss_value,  # Surface normals loss (0 if disabled)
    }
    if return_pixelwise:
        result.update(px)
    return result


def _prepare_frame_data_for_refinement(
    sequence: Any,
    frame_idx: int,
    obj_idx: int,
) -> Optional[Dict[str, Any]]:
    """
    Prepare frame data for refinement from a cached Sequence.

    Parameters
    ----------
    sequence : Sequence
        Cached scene data.
    frame_idx : int
        Frame index.
    obj_idx : int
        Object index.

    Returns
    -------
    dict or None
        Dictionary with 'image', 'mask', 'K_matrix', 'gt_depth',
        'valid_mask', 'H', 'W', or None if the object is not present
        in this frame.
    """
    frame = sequence[frame_idx]

    image, render_masks, K_matrix = frame.image, frame.masks, frame.K_matrix
    mask = render_masks[obj_idx]
    if not mask.any():
        return None  # Object not present in this frame

    return {
        "image": image,
        "mask": mask,
        "K_matrix": K_matrix,
        "gt_depth": frame.depth_map_z,
        "valid_mask": frame.valid_mask,
        "H": image.shape[0],
        "W": image.shape[1],
    }


# ---------------------------------------------------------------------------
# Differentiable pose decoder: raw Stage 1 tokens → final pose
# ---------------------------------------------------------------------------

# Constants from the MM-DiT Stage 1 for 6D rotation denormalization.
# Imported from sam3d_objects.pipeline.inference_utils but duplicated here
# to avoid heavy submodule imports at module level.
_ROTATION_6D_MEAN = torch.tensor([
    -0.06366084883674913, 0.008438224692279752, 0.00017084786438302483,
    0.0007126610473540038, -0.0030916726538816417, 0.5166093753457688,
])
_ROTATION_6D_STD = torch.tensor([
    0.6656971967514863, 0.6787012271867754, 0.30345010594844524,
    0.4394504420678794, 0.39817973931717104, 0.6176286868761914,
])


def differentiable_rotation_decode(
    raw_6drot_normalized: torch.Tensor,
) -> torch.Tensor:
    """Decode raw normalized 6D rotation tokens to rotation matrices.

    Steps: denormalize → Gram-Schmidt orthogonalization.
    Matches SAM3D's pose_decoder rotation chain exactly.

    Parameters
    ----------
    raw_6drot_normalized : torch.Tensor
        Normalized 6D rotation from Stage 1, shape ``(..., 6)``.

    Returns
    -------
    torch.Tensor
        Rotation matrix/matrices ``(..., 3, 3)``.
    """
    device = raw_6drot_normalized.device
    mean = _ROTATION_6D_MEAN.to(device)
    std = _ROTATION_6D_STD.to(device)
    rot_6d = raw_6drot_normalized * std + mean
    a1 = rot_6d[..., 0:3]
    a2 = rot_6d[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)  # (..., 3, 3)


def differentiable_pose_decode(
    raw_6drot_normalized: torch.Tensor,
    raw_translation: torch.Tensor,
    raw_scale: torch.Tensor,
    scene_scale: torch.Tensor,
    scene_shift: torch.Tensor,
    downsample_factor: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert raw Stage 1 pose tokens to final pose via differentiable chain.

    Uses ``ScaleShiftInvariant.to_instance_pose`` directly.

    Accepts either single-token ``(D,)`` or multi-token ``(T, D)`` inputs.
    When multi-token, each token is decoded independently through the full
    SSI chain (matching SAM3D's ``_broadcast_postcompose``).

    Returns a camera-space pose (no c2w composition).

    Parameters
    ----------
    raw_6drot_normalized : torch.Tensor
        Normalized 6D rotation, shape ``(6,)`` or ``(T, 6)``.
    raw_translation : torch.Tensor
        Raw translation, shape ``(3,)`` or ``(T, 3)``.
    raw_scale : torch.Tensor
        Raw log-scale, shape ``(1,)``, ``(3,)``, ``(T, 1)``, or ``(T, 3)``.
    scene_scale : torch.Tensor
        Pointmap scale, shape ``(3,)`` or ``(1, 3)``.
    scene_shift : torch.Tensor
        Pointmap shift, shape ``(3,)`` or ``(1, 3)``.
    downsample_factor : float
        Post-multiplier for scale (from sparse-structure downsampling).

    Returns
    -------
    rotation_matrix : torch.Tensor
        Rotation matrix ``(1, 3, 3)`` or ``(T, 3, 3)``.
    quaternion : torch.Tensor
        Quaternion ``(1, 4)`` or ``(T, 4)`` in wxyz format.
    translation : torch.Tensor
        Translation ``(1, 3)`` or ``(T, 3)``.
    scale : torch.Tensor
        Isotropic scale ``(1, 3)`` or ``(T, 3)``.
    """
    from pytorch3d.transforms import quaternion_to_matrix as _q2m
    from sam3d_objects.data.dataset.tdfy.pose_target import (
        PoseTarget, ScaleShiftInvariant,
    )

    device = raw_6drot_normalized.device

    # Raw modalities stored by inference carry a (batch=1, T, D) shape
    # (see elem_raw in core/inference.py). Squeeze the leading batch dim so
    # downstream logic treats the remaining leading dim as the token count T.
    def _squeeze_leading_batch(x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(0) if x.dim() == 3 and x.shape[0] == 1 else x

    raw_6drot_normalized = _squeeze_leading_batch(raw_6drot_normalized)
    raw_translation = _squeeze_leading_batch(raw_translation)
    raw_scale = _squeeze_leading_batch(raw_scale)

    # Rotation: denormalize + Gram-Schmidt → matrix → quaternion
    rotation_matrix = differentiable_rotation_decode(raw_6drot_normalized)
    # Ensure 3D: (T, 3, 3) — unsqueeze if single-token (3, 3)
    if rotation_matrix.dim() == 2:
        rotation_matrix = rotation_matrix.unsqueeze(0)
    quaternion = matrix_to_quaternion(rotation_matrix)  # (T, 4)

    # Scale: exp
    raw_scale_2d = raw_scale if raw_scale.dim() >= 2 else raw_scale.unsqueeze(0)
    instance_scale = torch.exp(raw_scale_2d)  # (T, D_scale)

    # Translation: ensure 2D
    translation = raw_translation if raw_translation.dim() >= 2 else raw_translation.unsqueeze(0)

    # SSI decode via ScaleShiftInvariant.to_instance_pose
    # (same path as training's decode_poses_pipeline).
    # Loop per-token because _broadcast_postcompose treats the leading dim
    # as batch and the SSI transform is per-frame (shared across tokens).
    T = quaternion.shape[0]
    ssi_scale = scene_scale.flatten().to(device)
    ssi_shift = scene_shift.flatten().to(device)

    dec_Rs, dec_qs, dec_ts, dec_ss = [], [], [], []
    for i in range(T):
        pose_target = PoseTarget(
            x_instance_scale=instance_scale[i:i+1],
            x_instance_rotation=quaternion[i:i+1],
            x_instance_translation=translation[i:i+1],
            x_scene_scale=ssi_scale,
            x_scene_center=ssi_shift,
            x_translation_scale=torch.ones(1, 1, device=device),
            pose_target_convention=ScaleShiftInvariant.pose_target_convention,
        )
        ip = ScaleShiftInvariant.to_instance_pose(pose_target)
        q = ip.instance_quaternion_l2c.squeeze(0)   # (4,)
        dec_qs.append(q)
        dec_Rs.append(_q2m(q.unsqueeze(0)).squeeze(0))  # (3, 3)
        dec_ts.append(ip.instance_position_l2c.squeeze(0))  # (3,)
        s = ip.instance_scale_l2c.squeeze(0)  # (D,)
        dec_ss.append(s)

    dec_R = torch.stack(dec_Rs)  # (T, 3, 3)
    dec_q = torch.stack(dec_qs)  # (T, 4)
    dec_t = torch.stack(dec_ts)  # (T, 3)
    dec_s = torch.stack(dec_ss)  # (T, D)

    # Apply downsample_factor + make isotropic (always return 3-component scale)
    final_scale = dec_s * downsample_factor
    final_scale = final_scale.mean(dim=-1, keepdim=True).expand(*dec_s.shape[:-1], 3)

    return dec_R, dec_q, dec_t, final_scale


def refine_pose_for_frame(
    canonical_gs: "Gaussian | Dict[str, Any]",
    initial_rotation: torch.Tensor,
    initial_translation: torch.Tensor,
    initial_scale: torch.Tensor,
    gt_image: torch.Tensor,
    mask: torch.Tensor | np.ndarray,
    K_matrix: np.ndarray,
    losses: LossConfig,
    pipeline: PipelineConfig,
    refine_scale: str = "perframe",
    gt_depth: Optional[np.ndarray] = None,
    valid_mask: Optional[np.ndarray] = None,
    raw_modalities: Optional[Dict[str, torch.Tensor]] = None,
    scene_scale: Optional[torch.Tensor] = None,
    scene_shift: Optional[torch.Tensor] = None,
    downsample_factor: float = 1.0,
    optimize_pose_tokens: bool = False,
    output_dir: Optional[str] = None,
    save_renders: bool = True,
    obj_idx: int = 0,
    frame_idx: int = 0,
    *,
    renderer: Optional[PosedObjectRenderer] = None,
) -> Dict[str, Any]:
    """
    Refine pose parameters using differentiable rendering.

    Parameters
    ----------
    canonical_gs : Gaussian
        The canonical object (frozen), rendered through
        :func:`make_posed_object_renderer`.
    initial_rotation : torch.Tensor
        Initial quaternion rotation (1, 4).
    initial_translation : torch.Tensor
        Initial translation (1, 3).
    initial_scale : torch.Tensor
        Initial scale (1, 3) or (1,).
    gt_image : torch.Tensor
        Ground truth image (H, W, 3) in [0, 1].
    mask : np.ndarray or torch.Tensor
        Object mask (H, W), boolean.
    K_matrix : np.ndarray
        Camera intrinsics (3, 3).
    losses : LossConfig
        Per-phase loss weights and iteration count.
    pipeline : PipelineConfig
        Cross-phase pipeline control flags.
    refine_scale : str
        Scale refinement mode: "perframe" (optimize per-frame) or "none" (freeze).
    gt_depth : np.ndarray, optional
        Ground truth z-depth map (H, W). When provided with depth_weight > 0,
        depth loss is computed in the masked region.
    valid_mask : np.ndarray, optional
        Boolean mask of valid depth pixels (H, W). When provided, intersected
        with the object mask for depth loss (excludes unreliable MoGe pixels).
    raw_modalities : dict, optional
        Raw Stage 1 pose tokens: ``6drotation_normalized``, ``translation``,
        ``scale``. When provided with ``optimize_pose_tokens=True``, these are
        optimized instead of the final pose, with gradients flowing through
        the differentiable pose decoder.
    scene_scale : torch.Tensor, optional
        Pointmap scale for PoseTargetConverter, shape ``(3,)``.
    scene_shift : torch.Tensor, optional
        Pointmap shift for PoseTargetConverter, shape ``(3,)``.
    downsample_factor : float
        Scale post-multiplier from sparse-structure downsampling.
    optimize_pose_tokens : bool
        When True and raw_modalities are available, optimize raw Stage 1 pose
        tokens instead of final pose parameters.
    renderer : PosedObjectRenderer or None, keyword-only
        Prebuilt adapter for ``canonical_gs``; built here when omitted. Callers
        looping over frames should build it ONCE per object and pass it.

    Returns
    -------
    dict
        Refined pose parameters with keys:
        - rotation: refined quaternion (1, 4)
        - translation: refined translation (1, 3)
        - scale: refined uniform scale (1, 3)
        - loss_history: list of loss values at each iteration
        - best_iteration: iteration index with lowest loss
    """
    # Handle num_iterations == 0 (pose init search only, no optimization)
    if losses.num_iterations == 0:
        initial_scale_flat = initial_scale.view(-1)
        # Collapse to uniform scale
        if initial_scale_flat.shape[0] == 3:
            s = initial_scale_flat.mean().item()
        else:
            s = initial_scale_flat[0].item()
        scale_out = torch.tensor([[s, s, s]],
                                  device=initial_rotation.device, dtype=initial_rotation.dtype)
        return {
            "rotation": initial_rotation,
            "translation": initial_translation,
            "scale": scale_out,
            "loss_history": [],
            "best_iteration": 0,
        }

    device = initial_rotation.device
    H, W = gt_image.shape[:2]
    if renderer is None:
        renderer = make_posed_object_renderer(canonical_gs, device)

    # Prepare ground truth
    if isinstance(gt_image, np.ndarray):
        gt_image = torch.from_numpy(gt_image).float().to(device)
    else:
        gt_image = gt_image.float().to(device)

    # Prepare mask
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask).bool().to(device)
    else:
        mask = mask.bool().to(device)

    # Prepare GT depth tensor and valid mask (for depth loss or depth-derived normals)
    gt_depth_tensor = None
    valid_mask_tensor = None
    need_depth = losses.depth_weight > 0 or losses.normals_weight > 0
    if gt_depth is not None and need_depth:
        gt_depth_tensor = torch.from_numpy(gt_depth).float().to(device)
        if valid_mask is not None:
            valid_mask_tensor = torch.from_numpy(valid_mask).bool().to(device)

    # K_matrix tensor (for normals loss)
    K_tensor = None
    if losses.normals_weight > 0:
        K_tensor = torch.from_numpy(K_matrix).float().to(device)
        if valid_mask is not None and valid_mask_tensor is None:
            valid_mask_tensor = torch.from_numpy(valid_mask).bool().to(device)

    # Precompute signed distance transform for silhouette loss (expensive, do once)
    sdt = None
    if losses.silhouette_weight > 0 and losses.silhouette_sdt_weight > 0:
        sdt = _compute_signed_distance_transform(mask)

    # Determine whether to use pose token optimization
    # Only enabled for per-frame refinement (raw modalities may be stale otherwise)
    use_pose_tokens = (
        optimize_pose_tokens
        and refine_scale == "perframe"
        and raw_modalities is not None
        and "6drotation_normalized" in raw_modalities
        and scene_scale is not None
        and scene_shift is not None
    )
    if optimize_pose_tokens and not use_pose_tokens:
        if pipeline.verbose:
            print("      [Pose tokens] Raw modalities or scene context unavailable (or non-perframe mode), falling back to direct pose optimization")
    # Also check that raw_modalities has the expected "translation" key (not confuse
    # with the final translation from the token dict which has the same name)
    if use_pose_tokens and "translation" not in raw_modalities:
        use_pose_tokens = False
        if pipeline.verbose:
            print("      [Pose tokens] Missing raw translation modality, falling back to direct pose optimization")

    if use_pose_tokens:
        # --- Pose token optimization path ---
        # Optimize raw Stage 1 outputs and forward through differentiable pose decoder
        _rot_lr = float(losses.lr_rotation)
        _trans_lr = float(losses.lr_translation)
        _scale_lr = float(losses.lr_scale)
        opt_6drot = raw_modalities["6drotation_normalized"].clone().detach().float().requires_grad_(_rot_lr > 0)
        opt_raw_trans = raw_modalities["translation"].clone().detach().float().requires_grad_(_trans_lr > 0)
        opt_raw_scale = raw_modalities["scale"].clone().detach().float().requires_grad_(
            refine_scale == "perframe" and _scale_lr > 0)

        # Keep initial values for regularization
        initial_6drot = opt_6drot.clone().detach()
        initial_raw_trans = opt_raw_trans.clone().detach()
        initial_raw_scale = opt_raw_scale.clone().detach()

        param_groups = []
        if _rot_lr > 0:
            param_groups.append({"params": [opt_6drot], "lr": _rot_lr})
        if _trans_lr > 0:
            param_groups.append({"params": [opt_raw_trans], "lr": _trans_lr})
        if refine_scale == "perframe" and _scale_lr > 0:
            param_groups.append({"params": [opt_raw_scale], "lr": _scale_lr})

        # Dummy direct-opt variables (not used but referenced in shared code paths)
        opt_rotation = None
        opt_translation = None
        opt_scale = None
        initial_scale_param = None

        if pipeline.verbose:
            print("      [Pose tokens] Optimizing raw Stage 1 modalities (6D rotation, translation, log-scale)")
    else:
        # --- Direct pose optimization path ---
        opt_6drot = None
        opt_raw_trans = None
        opt_raw_scale = None
        initial_6drot = None
        initial_raw_trans = None
        initial_raw_scale = None

        _rot_lr = float(losses.lr_rotation)
        _trans_lr = float(losses.lr_translation)
        _scale_lr = float(losses.lr_scale)
        opt_rotation = initial_rotation.clone().detach().requires_grad_(_rot_lr > 0)
        opt_translation = initial_translation.clone().detach().requires_grad_(_trans_lr > 0)

        # Isotropic scale parameter (scalar)
        initial_scale_flat = initial_scale.view(-1)
        if initial_scale_flat.shape[0] == 3:
            initial_scale_param = initial_scale_flat.mean().view(1)
        else:
            initial_scale_param = initial_scale_flat[:1]

        _scale_grad = refine_scale == "perframe" and _scale_lr > 0
        opt_scale = initial_scale_param.clone().detach().requires_grad_(_scale_grad)

        param_groups = []
        if _rot_lr > 0:
            param_groups.append({"params": [opt_rotation], "lr": _rot_lr})
        if _trans_lr > 0:
            param_groups.append({"params": [opt_translation], "lr": _trans_lr})
        if refine_scale == "perframe" and _scale_lr > 0:
            param_groups.append({"params": [opt_scale], "lr": _scale_lr})

    optimizer = torch.optim.Adam(param_groups)

    bg_color = _get_bg_color(pipeline, device)

    best_loss = float("inf")
    best_params = {}
    best_iteration = 0
    loss_history = []

    # Optional Chamfer (GT-depth → posed-model) — precompute the object's
    # visible GT surface points (unproject GT depth to a camera-space pointmap,
    # keep mask ∩ valid ∩ finite) + a fixed model-point subsample, once.
    gt_chamfer_pts = None
    chamfer_model_idx = None
    # Resolved ONCE, outside the loop: a property of the config, not of the frame.
    _chamfer_trim = resolve_chamfer_gt_trim(losses, "global_pose_refine")
    if getattr(losses, "chamfer_weight", 0.0) > 0 and gt_depth is not None:
        # mask (already a torch tensor by this point) and K_matrix (still the
        # function's numpy param) may be render-resolution; gt_depth never
        # exceeds the backbone's resolution -- downsample to match before use.
        _, _K_chamfer, _mask_chamfer = _match_depth_grid(
            mask, gt_depth.shape,
            K_matrix=torch.as_tensor(K_matrix, dtype=torch.float32, device=device),
            mask=mask)
        gt_chamfer_pts = _chamfer_gt_points_from_depth(
            gt_depth, _K_chamfer.cpu().numpy(), _mask_chamfer, valid_mask,
            int(getattr(losses, "chamfer_max_gt_points", 2048)), device,
            trim_factor=_chamfer_trim)
        chamfer_model_idx = _chamfer_model_subsample_idx(
            renderer.num_model_points(),
            int(getattr(losses, "chamfer_max_model_points", 8192)), device)

    for iteration in range(losses.num_iterations):
        optimizer.zero_grad()

        # Derive current pose (either from raw tokens or direct params)
        if use_pose_tokens:
            _, cur_rotation, cur_translation, cur_scale = differentiable_pose_decode(
                opt_6drot, opt_raw_trans, opt_raw_scale,
                scene_scale, scene_shift, downsample_factor,
            )
        else:
            cur_rotation = opt_rotation
            cur_translation = opt_translation
            cur_scale = opt_scale

        # Render frame with current pose
        rgb, alpha, depth = renderer.render(
            cur_rotation, cur_translation, cur_scale, K_matrix, W, H, device,
            bg_color=bg_color,
        )

        # One-directional Chamfer (GT-depth → posed model means), optional.
        chamfer_loss_value = torch.tensor(0.0, device=device)
        if gt_chamfer_pts is not None:
            _model_xyz = renderer.model_points(
                cur_rotation, cur_translation, cur_scale, device)
            chamfer_loss_value = _chamfer_gt_to_model(
                gt_chamfer_pts, _model_xyz, model_sub_idx=chamfer_model_idx)

        # Compute RGB, SSIM, silhouette, depth, and normals loss
        want_pixelwise = (
            save_renders
            and output_dir is not None
            and iteration % 50 == 0
        )
        losses_dict = _compute_frame_loss(
            rgb, alpha, gt_image, mask, losses, sdt,
            rendered_depth=depth, gt_depth=gt_depth_tensor,
            valid_mask=valid_mask_tensor,
            bg_color=bg_color,
            K_matrix=K_tensor,
            return_pixelwise=want_pixelwise,
            # This path renders the object alone (no background model), so depth
            # outside the GT mask has no valid rendered counterpart -- force masked
            # depth regardless of the config default.  Same auto-fall-back the
            # shared-world composite path applies on background-less frames; lets a
            # global-refine LossConfig (depth_mask_only=false + render_with_background)
            # drive per-frame refinement.
            depth_mask_only=True,
        )

        # Save per-pixel debug plot every 50 iterations
        if want_pixelwise:
            _K = K_tensor
            if _K is None:
                _K = torch.from_numpy(K_matrix).float().to(device)
            mask_f = mask.float().unsqueeze(-1)
            gt_masked = gt_image * mask_f + bg_color.view(1, 1, 3) * (1.0 - mask_f)
            frame_px: Dict[str, Any] = {
                "rendered_rgb": rgb.detach().cpu().numpy(),
                "gt_rgb": gt_masked.detach().cpu().numpy(),
                "mask": mask.cpu().numpy(),
            }
            if depth is not None:
                depth_sq = depth.squeeze(0) if depth.dim() == 3 else depth
                frame_px["rendered_depth"] = depth_sq.detach().cpu().numpy()
                frame_px["rendered_normals"] = depth_to_normals(depth_sq, _K).detach().cpu().numpy()
            if gt_depth_tensor is not None:
                # mask/_K are render-resolution; gt_depth_tensor never exceeds the
                # backbone's resolution -- downsample to match before use.
                _, _K_gt, _mask_gt = _match_depth_grid(
                    mask, gt_depth_tensor.shape[-2:], K_matrix=_K, mask=mask)
                gt_depth_masked = gt_depth_tensor * _mask_gt.float()
                frame_px["gt_depth"] = gt_depth_masked.cpu().numpy()
                frame_px["gt_normals"] = depth_to_normals(gt_depth_tensor, _K_gt, mask=_mask_gt).detach().cpu().numpy()
            for k, v in losses_dict.items():
                if k.startswith("px_"):
                    frame_px[k] = v
            from .visualization import save_pixelwise_loss_plot
            with get_timer().exclude():
                save_pixelwise_loss_plot({frame_idx: frame_px}, iteration, obj_idx, output_dir)

        # RGB and SSIM losses are separate scalars for logging
        rgb_loss_value = losses_dict["rgb_loss"]  # Excludes SSIM
        ssim_loss_value = losses_dict["ssim_loss"]  #  (0 if disabled)
        depth_loss_value = losses_dict["depth_loss"]  #  (0 if disabled)
        normals_loss_value = losses_dict["normals_loss"]  #  (0 if disabled)

        # Extract individual silhouette components for logging
        silh_com = losses_dict["silhouette_com"]
        silh_sdt = losses_dict["silhouette_sdt"]
        silh_iou = losses_dict["silhouette_iou"]

        # Combined silhouette loss
        silhouette_loss_value = (
            losses.silhouette_com_weight * silh_com
            + losses.silhouette_sdt_weight * silh_sdt
            + losses.silhouette_iou_weight * silh_iou
        )

        # Combine weighted losses
        weighted_rgb_loss = losses.rgb_weight * rgb_loss_value
        weighted_ssim_loss_value = losses.rgb_ssim_weight * ssim_loss_value
        weighted_silhouette_loss_value = losses.silhouette_weight * silhouette_loss_value
        weighted_depth_loss = losses.depth_weight * depth_loss_value
        weighted_normals_loss = losses.normals_weight * normals_loss_value
        weighted_perceptual_loss = losses.perceptual_weight * losses_dict["perceptual_loss"]

        base_loss = weighted_rgb_loss + weighted_ssim_loss_value + weighted_silhouette_loss_value + weighted_perceptual_loss + weighted_depth_loss + weighted_normals_loss + getattr(losses, "chamfer_weight", 0.0) * chamfer_loss_value

        # Debug: Check if rendered object overlaps with mask
        if iteration == 0 and pipeline.verbose:
            with torch.no_grad():
                rendered_in_mask = rgb[mask]
                bg_in_mask = bg_color.expand(rendered_in_mask.shape[0], -1)
                non_bg_mask = (rendered_in_mask - bg_in_mask).abs().sum(dim=-1) > 0.01
                print(
                    f"      [Overlap check] Mask pixels: {mask.sum().item()}, "
                    f"Non-background in mask: {non_bg_mask.sum().item()} "
                    f"({100*non_bg_mask.float().mean().item():.1f}%)"
                )

        # Optional: add regularization to prevent large deviations from initial pose
        if losses.regularization_weight > 0:
            rot_w = losses.regularization_rotation_weight
            trans_w = losses.regularization_translation_weight
            scale_w = losses.regularization_scale_weight
            if use_pose_tokens:
                # Regularize in raw Stage 1 space
                reg_rot = ((opt_6drot - initial_6drot) ** 2).sum()
                reg_trans = ((opt_raw_trans - initial_raw_trans) ** 2).sum()
                if refine_scale == "perframe":
                    reg_scale = ((opt_raw_scale - initial_raw_scale) ** 2).sum()
                    weighted_reg_loss = losses.regularization_weight * (rot_w * reg_rot + trans_w * reg_trans + scale_w * reg_scale)
                else:
                    weighted_reg_loss = losses.regularization_weight * (rot_w * reg_rot + trans_w * reg_trans)
            else:
                opt_rotation_normalized = opt_rotation / opt_rotation.norm()
                reg_rot = (
                    (opt_rotation_normalized - initial_rotation / initial_rotation.norm()) ** 2
                ).sum()
                reg_trans = ((opt_translation - initial_translation) ** 2).sum()
                if refine_scale == "perframe":
                    reg_scale = ((opt_scale - initial_scale_param) ** 2).sum()
                    weighted_reg_loss = losses.regularization_weight * (rot_w * reg_rot + trans_w * reg_trans + scale_w * reg_scale)
                else:
                    weighted_reg_loss = losses.regularization_weight * (rot_w * reg_rot + trans_w * reg_trans)
        else:
            weighted_reg_loss = torch.tensor(0.0, device=device)

        # Total loss
        loss = base_loss + weighted_reg_loss

        # Record loss for this iteration (all values are weighted)
        loss_history.append(
            {
                "total": loss.item(),
                "rgb": weighted_rgb_loss.item(),
                "ssim": weighted_ssim_loss_value.item(),
                "silhouette": weighted_silhouette_loss_value.item(),
                "silhouette_com": silh_com.item(),
                "silhouette_sdt": silh_sdt.item(),
                "silhouette_iou": silh_iou.item(),
                "depth": weighted_depth_loss.item(),
                "normals": weighted_normals_loss.item(),
                "perceptual": weighted_perceptual_loss.item() if isinstance(weighted_perceptual_loss, torch.Tensor) else weighted_perceptual_loss,
                "chamfer": (getattr(losses, "chamfer_weight", 0.0) * chamfer_loss_value).item(),
                "regularization": weighted_reg_loss.item(),
            }
        )

        # Track best BEFORE backward/step so saved params match the evaluated loss
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_iteration = iteration
            if use_pose_tokens:
                # Derive final pose from current raw tokens
                with torch.no_grad():
                    _, best_rot, best_trans, best_scl = differentiable_pose_decode(
                        opt_6drot, opt_raw_trans, opt_raw_scale,
                        scene_scale, scene_shift, downsample_factor,
                    )
                best_params = {
                    "rotation": best_rot.detach(),
                    "translation": best_trans.detach(),
                    "scale": best_scl.detach(),
                    "raw_modalities": {
                        "6drotation_normalized": opt_6drot.clone().detach(),
                        "translation": opt_raw_trans.clone().detach(),
                        "scale": opt_raw_scale.clone().detach(),
                    },
                }
            else:
                # Store scale as 3D tensor (1, 3) — expand scalar if isotropic
                scale_3d = opt_scale.clone().detach().expand(3).reshape(1, 3)
                best_params = {
                    "rotation": opt_rotation.clone().detach(),
                    "translation": opt_translation.clone().detach(),
                    "scale": scale_3d,
                }

        # Backprop
        loss.backward()

        # Step optimizer
        optimizer.step()

        # Project scale back to isotropic (uniform) (direct path only)
        if not use_pose_tokens and refine_scale == "perframe":
            with torch.no_grad():
                opt_scale.fill_(opt_scale.mean())

        # Early stopping: break if no improvement for patience iterations
        if losses.early_stop_patience > 0 and (iteration - best_iteration) >= losses.early_stop_patience:
            if pipeline.verbose:
                print(
                    f"      Early stopping at iteration {iteration} "
                    f"(no improvement for {losses.early_stop_patience} iters, best at {best_iteration})"
                )
            break

        if pipeline.verbose and (
            iteration % pipeline.log_interval == 0 or iteration == losses.num_iterations - 1
        ):
            silh_details = ""
            if losses.silhouette_weight > 0:
                silh_details = (
                    f" (com={silh_com.item():.4f}, sdt={silh_sdt.item():.4f}, "
                    f"iou={silh_iou.item():.4f})"
                )
            percept_str = ""
            if losses.perceptual_weight > 0:
                percept_val = weighted_perceptual_loss.item() if isinstance(weighted_perceptual_loss, torch.Tensor) else weighted_perceptual_loss
                percept_str = f"percept={percept_val:.6f} "
            depth_str = ""
            if losses.depth_weight > 0:
                depth_str = f"depth={weighted_depth_loss.item():.6f} "
            print(
                f"      Iteration {iteration}: total={loss.item():.6f} "
                f"rgb={weighted_rgb_loss.item():.6f} "
                f"ssim={weighted_ssim_loss_value.item():.6f} "
                f"{percept_str}"
                f"{depth_str}"
                f"silh={weighted_silhouette_loss_value.item():.6f}{silh_details} "
                f"reg={weighted_reg_loss.item():.8f} "
            )

    if pipeline.verbose:
        print(f"      Best loss: {best_loss:.6f} (iteration {best_iteration})")

    # Normalize the rotation in best_params
    best_params["rotation"] = best_params["rotation"] / best_params["rotation"].norm()

    # Add loss history and best iteration to the result
    best_params["loss_history"] = loss_history
    best_params["best_iteration"] = best_iteration

    return best_params


def refine_poses_global_composite(
    canonical_gaussians: Dict[int, Any],
    tokens_by_object: Dict[int, List[Tuple[int, Dict[str, Any]]]],
    sequence: Any,
    losses: LossConfig,
    pipeline: PipelineConfig,
    perframe_raw_modalities: Optional[Dict[int, Dict[int, Dict[str, Any]]]] = None,
    output_dir: Optional[str] = None,
    save_renders: bool = True,
    # Per-canonical-mesh-vertex deformation field (actionmesh mono-dynamic).
    # See ``refine_poses_for_sequence`` for full kwarg semantics.  Warps
    # are evaluated once per (obj, frame) under ``no_grad`` and cached on
    # ``frame_data[frame_idx]["overrides_per_obj"]``; the composite render
    # injects them via ``_transform_object_to_r3``'s ``means_override`` /
    # ``rotation_override`` kwargs.
    canonical_mesh_verts_per_obj: Optional[Dict[int, torch.Tensor]] = None,
    per_frame_mesh_verts_per_obj: Optional[Dict[int, Dict[int, torch.Tensor]]] = None,
    per_frame_mesh_rotations_per_obj: Optional[Dict[int, Dict[int, torch.Tensor]]] = None,
    canonical_mesh_faces_per_obj: Optional[Dict[int, torch.Tensor]] = None,
    warp_knn_k: int = 4,
    warp_knn_eps: float = 1.0e-8,
    warp_knn_chunk_size: int = 8192,
) -> Dict[int, List[Tuple[int, Dict[str, Any]]]]:
    """
    Refine poses for ALL objects jointly with composite rendering.

    Renders all objects together in a single gsplat pass per frame.
    Losses are computed on the full composite image vs the full GT image
    (union mask for silhouette, union-masked GT for appearance).

    Each object has its own global scale, per-frame rotation, and per-frame
    translation.  All are optimized jointly in a single optimizer.

    Parameters
    ----------
    canonical_gaussians : dict
        {obj_idx: Gaussian} — canonical Gaussian per object.
    tokens_by_object : dict
        {obj_idx: [(frame_idx, decoder_input), ...]} per object.
    sequence : Sequence
        Cached scene data.
    losses : LossConfig
        Per-phase loss weights and iteration count.
    pipeline : PipelineConfig
        Cross-phase pipeline control flags.

    Returns
    -------
    dict
        Refined tokens_by_object with updated poses for each object.
    """
    device = torch.device("cuda")
    all_obj_indices = sorted(tokens_by_object.keys())
    num_objects = len(all_obj_indices)

    print(f"\n    Composite rendering: {num_objects} objects jointly")

    # ── Determine valid frames (union of all objects' frame sets) ──
    frame_to_obj_inputs = {}  # {frame_idx: {obj_idx: decoder_input}}
    for obj_idx in all_obj_indices:
        for frame_idx, decoder_input in tokens_by_object[obj_idx]:
            if frame_idx not in frame_to_obj_inputs:
                frame_to_obj_inputs[frame_idx] = {}
            frame_to_obj_inputs[frame_idx][obj_idx] = decoder_input

    # ── Load frame data with union masks ──
    frame_data = {}
    valid_frame_indices = []

    for frame_idx in sorted(frame_to_obj_inputs.keys()):
        frame = sequence[frame_idx]
        render_image, render_masks, render_K = frame.image, frame.masks, frame.K_matrix
        gt_image = torch.from_numpy(render_image).float().cuda() / 255.0

        # Compute union mask and per-object masks
        H, W = render_image.shape[0], render_image.shape[1]
        union_mask = torch.zeros(H, W, dtype=torch.bool, device=device)
        per_obj_masks = {}
        present_objects = []

        for obj_idx in all_obj_indices:
            m = torch.from_numpy(render_masks[obj_idx]).bool().cuda()
            per_obj_masks[obj_idx] = m
            if m.any():
                union_mask = union_mask | m
                present_objects.append(obj_idx)

        if not union_mask.any():
            print(f"      Warning: No objects visible in frame {frame_idx}, skipping")
            continue

        # GT depth and valid mask (needed for depth loss or depth-derived normals)
        gt_depth_tensor = None
        valid_mask_tensor = None
        need_depth = losses.depth_weight > 0 or losses.normals_weight > 0
        if frame.depth_map_z is not None and need_depth:
            gt_depth_tensor = torch.from_numpy(frame.depth_map_z).float().cuda()
            if frame.valid_mask is not None:
                valid_mask_tensor = torch.from_numpy(frame.valid_mask).bool().cuda()

        # K_matrix tensor (for normals loss)
        K_tensor = None
        if losses.normals_weight > 0:
            K_tensor = torch.from_numpy(render_K).float().cuda()
            if frame.valid_mask is not None and valid_mask_tensor is None:
                valid_mask_tensor = torch.from_numpy(frame.valid_mask).bool().cuda()

        frame_data[frame_idx] = {
            "gt_image": gt_image,
            "union_mask": union_mask,
            "per_obj_masks": per_obj_masks,
            "present_objects": present_objects,
            "gt_depth": gt_depth_tensor,
            "valid_mask": valid_mask_tensor,
            "K_tensor": K_tensor,
            "K_matrix": render_K,
            "H": H,
            "W": W,
            "overrides_per_obj": {},
        }
        valid_frame_indices.append(frame_idx)

    if len(valid_frame_indices) == 0:
        print("      No valid frames for composite refinement")
        return tokens_by_object

    print(f"      {len(valid_frame_indices)} valid frames, "
          f"{num_objects} objects")

    # ── Optional background Gaussians for full-image depth supervision ──
    # depth_mask_only=False scores the depth loss over the WHOLE image, which
    # is only well-defined if the render has valid depth OUTSIDE the object
    # union.  Build a pointmap background per frame and append it to the
    # composite render so has_background=True is truthful.  The background is a
    # FIXED depth reference: every param is DETACHED, so no gradient ever flows
    # into it (pose grads come only through the foreground objects).
    # NOTE: full-image mode passes an all-ones loss mask (so GT depth is not
    # zeroed outside the objects), which makes the silhouette term inert —
    # coverage is meaningless once the whole background is rendered.  Drop
    # silhouette_weight to 0 if you don't want that near-constant term.
    background_params_per_frame: Optional[Dict[int, Tuple[torch.Tensor, ...]]] = None
    if getattr(losses, "render_with_background", False):
        from .gaussian import create_background_gaussians
        from .rendering import transform_gaussian_params_world_to_cam
        background_params_per_frame = {}
        print(f"      render_with_background=True → building DETACHED background "
              f"Gaussians for {len(valid_frame_indices)} frames "
              f"(full-image depth; silhouette term goes inert)...")
        for frame_idx in valid_frame_indices:
            frame = sequence[frame_idx]
            # Background = everything NOT covered by any present object.
            obj_masks = [frame.masks[oi] for oi in frame_data[frame_idx]["present_objects"]]
            bg_gs = create_background_gaussians(
                frame.image, frame.pointmap, obj_masks, frame.K_matrix, c2w=frame.c2w,
            )
            if bg_gs is None:
                continue
            # World→camera (composite renders camera-space with identity c2w).
            # Detach BEFORE and AFTER the transform so the background is a pure
            # fixed reference — no autograd graph is retained on it.
            bg_xyz = bg_gs.get_xyz.detach().to(device)
            bg_rot = bg_gs.get_rotation.detach().to(device)
            bg_xyz, bg_rot = transform_gaussian_params_world_to_cam(bg_xyz, bg_rot, frame.c2w)
            background_params_per_frame[frame_idx] = (
                bg_xyz.detach(),
                bg_rot.detach(),
                bg_gs.get_scaling.detach().to(device),
                bg_gs.get_opacity.squeeze(-1).detach().to(device),
                bg_gs.get_features.detach().to(device),
            )
        print(f"      Background Gaussians ready "
              f"({len(background_params_per_frame)}/{len(valid_frame_indices)} frames)")

    # ── Optional Chamfer (GT-depth → posed-model) alignment targets ──
    # Per (frame, obj): the object's visible GT surface points from the depth
    # pointmap.  Per obj: a fixed model-point subsample.  Both cheap to reuse
    # every iter.  camera-space R3 on both sides (composite renders identity c2w).
    gt_chamfer_points = None
    chamfer_model_idx = None
    _chamfer_trim = resolve_chamfer_gt_trim(losses, "global_pose_refine")
    if getattr(losses, "chamfer_weight", 0.0) > 0:
        _max_gt = int(getattr(losses, "chamfer_max_gt_points", 2048))
        _max_model = int(getattr(losses, "chamfer_max_model_points", 8192))
        gt_chamfer_points = {}
        for frame_idx in valid_frame_indices:
            pm = torch.from_numpy(sequence[frame_idx].pointmap).float().to(device)
            _vm = frame_data[frame_idx].get("valid_mask")
            # sequence[frame_idx].masks (NOT frame_data[...]["per_obj_masks"],
            # which may be render-resolution) -- pm is always at the backbone's
            # (H', W'), and pointmap-aligned masks must match it exactly.
            gt_chamfer_points[frame_idx] = {
                oi: _chamfer_gt_points_for_object(
                    pm, torch.from_numpy(
                        np.asarray(sequence[frame_idx].masks[oi])).bool().to(device),
                    _max_gt, valid_mask=_vm, trim_factor=_chamfer_trim)
                for oi in frame_data[frame_idx]["present_objects"]
            }
        chamfer_model_idx = {
            oi: _chamfer_model_subsample_idx(
                canonical_gaussians[oi].get_xyz.shape[0], _max_model, device)
            for oi in all_obj_indices
        }
        print(f"      Chamfer alignment ON (GT→model, weight={losses.chamfer_weight}, "
              f"GT cap={_max_gt}, model cap={_max_model})")

    # ── Pre-compute per-(obj, frame) deformation warps (actionmesh) ──
    # Identity-default no-op when the deformation field is unsupplied.
    # Composite render iterates frame-major, so we invert the (frame ->
    # objects) layout into (obj -> frames) for the helper, then re-key
    # the result back into ``frame_data[frame_idx]["overrides_per_obj"]``.
    _frames_per_obj_for_warp: Dict[int, List[Any]] = {}
    for frame_idx in valid_frame_indices:
        for obj_idx in frame_data[frame_idx]["present_objects"]:
            _frames_per_obj_for_warp.setdefault(obj_idx, []).append(frame_idx)
    warp_cache = _precompute_perframe_gaussian_warps(
        canonical_gaussians, _frames_per_obj_for_warp,
        canonical_mesh_verts_per_obj,
        per_frame_mesh_verts_per_obj,
        per_frame_mesh_rotations_per_obj,
        canonical_mesh_faces_per_obj,
        warp_knn_k=warp_knn_k, warp_knn_eps=warp_knn_eps,
        warp_knn_chunk_size=warp_knn_chunk_size,
    )
    for obj_idx, per_frame in warp_cache.items():
        for frame_idx, override in per_frame.items():
            frame_data[frame_idx]["overrides_per_obj"][obj_idx] = override
    _n_warped = sum(len(v) for v in warp_cache.values())
    if _n_warped:
        print(f"      Deformation warp: cached {_n_warped} (obj, frame) overrides")

    # ── Determine per-object pose-token availability ──
    use_pose_tokens_per_obj = {}  # {obj_idx: bool}
    if pipeline.optimize_pose_tokens and perframe_raw_modalities:
        for obj_idx in all_obj_indices:
            obj_raw = perframe_raw_modalities.get(obj_idx, {})
            ok = True
            for fi in valid_frame_indices:
                if obj_idx not in frame_to_obj_inputs.get(fi, {}):
                    continue
                raw = obj_raw.get(fi)
                if (raw is None
                        or "raw_ss_modalities" not in raw
                        or "6drotation_normalized" not in raw["raw_ss_modalities"]
                        or "translation" not in raw["raw_ss_modalities"]
                        or "scale" not in raw["raw_ss_modalities"]
                        or raw.get("pointmap_scale") is None
                        or raw.get("pointmap_shift") is None):
                    ok = False
                    break
            use_pose_tokens_per_obj[obj_idx] = ok
    else:
        for obj_idx in all_obj_indices:
            use_pose_tokens_per_obj[obj_idx] = False

    any_pose_tokens = any(use_pose_tokens_per_obj.values())
    if any_pose_tokens:
        pt_objs = [oi for oi, v in use_pose_tokens_per_obj.items() if v]
        print(f"      [Pose tokens] Enabled for objects: {pt_objs}")

    # ── Initialize optimizable parameters ──
    opt_rotations = {}      # {obj_idx: {frame_idx: tensor}}
    opt_translations = {}   # {obj_idx: {frame_idx: tensor}}
    opt_global_scale = {}   # {obj_idx: tensor}
    initial_rotations = {}
    initial_translations = {}
    initial_global_scales = {}

    # Pose-token specific dicts
    opt_6drots = {}         # {obj_idx: {frame_idx: tensor}}
    opt_raw_trans = {}      # {obj_idx: {frame_idx: tensor}}
    initial_6drots = {}
    initial_raw_trans = {}
    frozen_raw_scales = {}  # {obj_idx: {frame_idx: tensor}}
    scene_scales_all = {}   # {obj_idx: {frame_idx: tensor/value}}
    scene_shifts_all = {}
    downsample_factors_all = {}

    param_groups = []

    for obj_idx in all_obj_indices:
        opt_rotations[obj_idx] = {}
        opt_translations[obj_idx] = {}
        initial_rotations[obj_idx] = {}
        initial_translations[obj_idx] = {}
        opt_6drots[obj_idx] = {}
        opt_raw_trans[obj_idx] = {}
        initial_6drots[obj_idx] = {}
        initial_raw_trans[obj_idx] = {}
        frozen_raw_scales[obj_idx] = {}
        scene_scales_all[obj_idx] = {}
        scene_shifts_all[obj_idx] = {}
        downsample_factors_all[obj_idx] = {}

        # Initialize per-object global scale from first available frame
        first_di = None
        for fi in valid_frame_indices:
            if obj_idx in frame_to_obj_inputs.get(fi, {}):
                first_di = frame_to_obj_inputs[fi][obj_idx]
                break
        if first_di is None:
            continue

        scale_flat = first_di["scale"].view(-1)
        init_scale = scale_flat.mean().view(1) if scale_flat.shape[0] >= 3 else scale_flat[:1].clone()
        _scale_lr = float(losses.lr_scale)
        opt_global_scale[obj_idx] = init_scale.detach().to(device).requires_grad_(_scale_lr > 0)
        initial_global_scales[obj_idx] = opt_global_scale[obj_idx].clone().detach()
        if _scale_lr > 0:
            param_groups.append({"params": [opt_global_scale[obj_idx]], "lr": _scale_lr})

        scale_str = f"{opt_global_scale[obj_idx].item():.6f}"
        print(f"      Object {obj_idx}: initial scale = {scale_str}")

        # Per-frame rotation and translation
        obj_raw = (perframe_raw_modalities or {}).get(obj_idx, {})
        for frame_idx in valid_frame_indices:
            if obj_idx not in frame_to_obj_inputs.get(frame_idx, {}):
                continue

            _rot_lr = float(losses.lr_rotation)
            _trans_lr = float(losses.lr_translation)
            if use_pose_tokens_per_obj[obj_idx]:
                raw = obj_raw[frame_idx]
                mods = raw["raw_ss_modalities"]
                opt_6drots[obj_idx][frame_idx] = mods["6drotation_normalized"].clone().detach().float().to(device).requires_grad_(_rot_lr > 0)
                opt_raw_trans[obj_idx][frame_idx] = mods["translation"].clone().detach().float().to(device).requires_grad_(_trans_lr > 0)
                frozen_raw_scales[obj_idx][frame_idx] = mods["scale"].clone().detach().float().to(device)
                scene_scales_all[obj_idx][frame_idx] = raw["pointmap_scale"]
                scene_shifts_all[obj_idx][frame_idx] = raw["pointmap_shift"]
                downsample_factors_all[obj_idx][frame_idx] = raw.get("downsample_factor", 1.0)
                initial_6drots[obj_idx][frame_idx] = opt_6drots[obj_idx][frame_idx].clone().detach()
                initial_raw_trans[obj_idx][frame_idx] = opt_raw_trans[obj_idx][frame_idx].clone().detach()
                if _rot_lr > 0:
                    param_groups.append({"params": [opt_6drots[obj_idx][frame_idx]], "lr": _rot_lr})
                if _trans_lr > 0:
                    param_groups.append({"params": [opt_raw_trans[obj_idx][frame_idx]], "lr": _trans_lr})
            else:
                di = frame_to_obj_inputs[frame_idx][obj_idx]
                initial_rotations[obj_idx][frame_idx] = di["rotation"].clone().detach().to(device)
                initial_translations[obj_idx][frame_idx] = di["translation"].clone().detach().to(device)
                opt_rotations[obj_idx][frame_idx] = (
                    di["rotation"].clone().detach().to(device).requires_grad_(_rot_lr > 0)
                )
                opt_translations[obj_idx][frame_idx] = (
                    di["translation"].clone().detach().to(device).requires_grad_(_trans_lr > 0)
                )
                if _rot_lr > 0:
                    param_groups.append({"params": [opt_rotations[obj_idx][frame_idx]], "lr": _rot_lr})
                if _trans_lr > 0:
                    param_groups.append({"params": [opt_translations[obj_idx][frame_idx]], "lr": _trans_lr})

    # ---- Shared correction: ONE Sim(3) per object, composed onto every frame's pose ----
    # `pose_i o dT` in the object's CANONICAL frame, so per-frame root MOTION survives --
    # unlike the shared-world path, which DERIVES each frame's pose from one reference and
    # thereby replaces it.  Orthogonal to `strategy`: this is the photometric loss driving
    # the shared correction.
    # On MV data this solver is reached ONLY with mv_shared_world_pose off (
    # `refine_poses_for_sequence` short-circuits to the shared-world path first), where its
    # per-FrameKey natives are per-VIEW -- which is what the resolver refuses.
    correction_granularity_mode = resolve_correction_granularity(
        losses, "global_pose_refine", is_mv=sequence.is_mv,
        mv_shared_world_pose=pipeline.mv_shared_world_pose)
    opt_delta_aa, opt_delta_logs, opt_delta_t = {}, {}, {}
    if correction_granularity_mode != "per_frame":
        for obj_idx in all_obj_indices:
            (opt_delta_aa[obj_idx], opt_delta_logs[obj_idx], opt_delta_t[obj_idx],
             _dg) = build_correction(
                float(losses.lr_rotation), float(losses.lr_scale),
                float(losses.lr_translation), device)
            param_groups.extend(_dg)
        _freeze = natives_to_freeze(
            correction_granularity_mode,
            resolve_correction_scale_control(losses, "global_pose_refine"))
        if _freeze == "all":
            # Freeze the native params so the delta is the ONLY thing that moves.  That
            # is what makes the fit well-posed, and what lets the result be READ as "the
            # systematic error": with both free (`correction_granularity: both`, which skips
            # this freeze) any delta is absorbable into the per-frame poses, so their
            # split is a gauge choice rather than a result.
            _delta_ids = {id(p) for d in (opt_delta_aa, opt_delta_logs, opt_delta_t)
                          for p in d.values()}
            _is_delta = [any(id(p) in _delta_ids for p in g["params"])
                         for g in param_groups]
            for g, keep in zip(param_groups, _is_delta):
                if not keep:
                    for prm in g["params"]:
                        prm.requires_grad_(False)
            param_groups = [g for g, keep in zip(param_groups, _is_delta) if keep]
        elif _freeze == "scale":
            # The object has ONE size, so the shared correction owns it and the native
            # scale is held.  Dropped from the groups rather than given lr=0, because
            # `losses.lr_scale` also drives the correction's scale and would freeze that.
            param_groups = freeze_params(param_groups, opt_global_scale.values())
        print(f"      Pose correction: {correction_granularity_mode} "
              f"({len(opt_delta_aa)} object(s), one Sim(3) each)"
              + ("; native scale FROZEN" if _freeze == "scale" else ""))

    optimizer = torch.optim.Adam(param_groups)

    # Batch size
    num_frames = len(valid_frame_indices)
    batch_size = losses.batch_size if losses.batch_size > 0 else num_frames
    batch_size = min(batch_size, num_frames)
    effective_num_iterations = losses.num_iterations

    print(f"      Batch size: {batch_size} (out of {num_frames} frames)")

    # ── Optimization loop ──
    best_loss = float("inf")
    best_params = {}
    best_iteration = 0
    loss_history = []

    if effective_num_iterations == 0:
        print("      Skipping optimization (num_iterations=0)")
        return tokens_by_object

    frame_sampler = EpochFrameSampler(valid_frame_indices)

    pbar = tqdm(
        range(effective_num_iterations),
        desc="      Composite opt",
        leave=True,
    )
    for iteration in pbar:
        optimizer.zero_grad()

        # Sample batch of frames (epoch-based: all frames seen before any repeats)
        if batch_size >= len(valid_frame_indices):
            batch_frame_indices = valid_frame_indices
        else:
            batch_frame_indices = frame_sampler.sample(batch_size)

        total_rgb_loss = 0.0
        total_ssim_loss = 0.0
        total_silhouette_loss = 0.0
        total_depth_loss = 0.0
        total_normals_loss = 0.0
        total_perceptual_loss = 0.0
        total_reg_loss = 0.0
        total_chamfer_loss = 0.0
        total_chamfer_weight = 0.0   # per-(frame,object) contributions → per-object mean, not object-count-scaled
        total_weight = 0.0

        want_pixelwise = (
            save_renders
            and output_dir is not None
            and iteration % 50 == 0
        )
        px_data: Optional[Dict[int, Dict[str, Any]]] = (
            {} if want_pixelwise else None
        )

        for frame_idx in batch_frame_indices:
            data = frame_data[frame_idx]
            frame_weight = 1.0

            # Transform and concatenate all present objects' Gaussians
            all_xyz, all_rot, all_scales, all_opac, all_feats = [], [], [], [], []

            for obj_idx in data["present_objects"]:
                # Determine current rotation and translation
                if use_pose_tokens_per_obj.get(obj_idx, False) and frame_idx in opt_6drots.get(obj_idx, {}):
                    _, cur_rot, cur_trans, _ = differentiable_pose_decode(
                        opt_6drots[obj_idx][frame_idx],
                        opt_raw_trans[obj_idx][frame_idx],
                        frozen_raw_scales[obj_idx][frame_idx],
                        scene_scales_all[obj_idx][frame_idx],
                        scene_shifts_all[obj_idx][frame_idx],
                        downsample_factors_all[obj_idx][frame_idx],
                    )
                elif frame_idx in opt_rotations.get(obj_idx, {}):
                    cur_rot = opt_rotations[obj_idx][frame_idx]
                    cur_trans = opt_translations[obj_idx][frame_idx]
                else:
                    continue
                # Shared correction, in the object's CANONICAL frame.  Identity when this
                # object has no delta (`per_frame`), so no branch is needed.
                cur_rot, cur_trans, cur_scale = _apply_gs_correction(
                    opt_delta_aa.get(obj_idx), opt_delta_logs.get(obj_idx),
                    opt_delta_t.get(obj_idx),
                    cur_rot, cur_trans, opt_global_scale[obj_idx])
                _override = data["overrides_per_obj"].get(obj_idx)
                _means_ovr, _rot_ovr = _override if _override is not None else (None, None)
                xyz, rot, scales, opac, feats = _transform_object_to_r3(
                    canonical_gaussians[obj_idx],
                    cur_rot,
                    cur_trans,
                    cur_scale,
                    device,
                    means_override=_means_ovr,
                    rotation_override=_rot_ovr,
                )
                all_xyz.append(xyz)
                all_rot.append(rot)
                all_scales.append(scales)
                all_opac.append(opac)
                all_feats.append(feats)

                # One-directional Chamfer (GT-depth → this object's posed means).
                if gt_chamfer_points is not None:
                    _gt_pts = gt_chamfer_points.get(frame_idx, {}).get(obj_idx)
                    if _gt_pts is not None:
                        total_chamfer_loss = total_chamfer_loss + frame_weight * _chamfer_gt_to_model(
                            _gt_pts, xyz, model_sub_idx=chamfer_model_idx.get(obj_idx))
                        total_chamfer_weight = total_chamfer_weight + frame_weight

            if not all_xyz:
                continue

            cat_xyz = torch.cat(all_xyz, dim=0)
            cat_rot = torch.cat(all_rot, dim=0)
            cat_scales = torch.cat(all_scales, dim=0)
            cat_opac = torch.cat(all_opac, dim=0)
            cat_feats = torch.cat(all_feats, dim=0)

            # Append the (detached) background so the render has valid depth
            # outside the object union (full-image depth mode).  All bg params
            # are detached → they add geometry to the render but carry no
            # gradient.  ``_num_fg`` marks the fg/bg split for foreground-weight.
            _num_fg = cat_xyz.shape[0]
            has_bg = (
                background_params_per_frame is not None
                and frame_idx in background_params_per_frame
            )
            if has_bg:
                cat_xyz, cat_rot, cat_scales, cat_opac, cat_feats = _concat_fg_bg_params(
                    cat_xyz, cat_rot, cat_scales, cat_opac, cat_feats,
                    background_params_per_frame[frame_idx], device,
                )

            # Single composite render (camera-space poses → identity c2w)
            _c2w = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)
            bg_color = _get_bg_color(pipeline, device)

            rgb, alpha, depth = render_gaussian_params(
                cat_xyz, cat_rot, cat_scales, cat_opac, cat_feats,
                _c2w, data["K_matrix"], data["W"], data["H"],
                bg_color=bg_color,
            )

            # Compute losses on composite.  Foreground-only mode masks to the
            # union of per-object masks (silhouette / depth see the object
            # footprint).  Full-image mode (has_bg) passes an all-ones mask so
            # GT depth is NOT zeroed outside the objects — the background render
            # supplies valid depth there, which is what makes has_background=True
            # meaningful (at the cost of neutralising the silhouette term).
            assert "union_mask" in data, (
                "refine_poses_global_composite: composite rendering requires "
                "frame_data['union_mask'] (union of all per-object masks)."
            )
            _loss_mask = (
                torch.ones(data["H"], data["W"], device=device, dtype=torch.bool)
                if has_bg else data["union_mask"]
            )
            losses_dict = _compute_frame_loss(
                rgb, alpha, data["gt_image"], _loss_mask,
                losses,
                rendered_depth=depth, gt_depth=data.get("gt_depth"),
                valid_mask=data.get("valid_mask"),
                bg_color=bg_color,
                K_matrix=data.get("K_tensor"),
                return_pixelwise=want_pixelwise,
                has_background=has_bg,
                # Frames without a fused background (e.g. GSO/OAB foreground-only GT
                # depth) fall back to masked depth so empty-bg pixels aren't
                # supervised as noise.
                depth_mask_only=losses.depth_mask_only or not has_bg,
            )

            # Collect per-pixel data for debug visualization
            if want_pixelwise:
                _K = data.get("K_tensor")
                if _K is None:
                    _K = torch.from_numpy(data["K_matrix"]).float().to(device)
                # Mask the GT panels with the SAME mask the loss used, so the
                # visualization matches what was actually supervised: object-only
                # in foreground mode, full-image (background included) in
                # full-image / has_bg mode.
                _mask = _loss_mask
                _mask_f = _mask.float().unsqueeze(-1)
                gt_masked = data["gt_image"] * _mask_f + bg_color.view(1, 1, 3) * (1.0 - _mask_f)
                frame_px: Dict[str, Any] = {
                    "rendered_rgb": rgb.detach().cpu().numpy(),
                    "gt_rgb": gt_masked.detach().cpu().numpy(),
                    "mask": _mask.cpu().numpy(),
                }
                if depth is not None:
                    depth_sq = depth.squeeze(0) if depth.dim() == 3 else depth
                    frame_px["rendered_depth"] = depth_sq.detach().cpu().numpy()
                    frame_px["rendered_normals"] = depth_to_normals(depth_sq, _K).detach().cpu().numpy()
                gt_d = data.get("gt_depth")
                if gt_d is not None:
                    _, _K_gt, _mask_gt = _match_depth_grid(
                        _mask, gt_d.shape[-2:], K_matrix=_K, mask=_mask)
                    gt_depth_masked = gt_d * _mask_gt.float()
                    frame_px["gt_depth"] = gt_depth_masked.cpu().numpy()
                    frame_px["gt_normals"] = depth_to_normals(gt_d, _K_gt, mask=_mask_gt).detach().cpu().numpy()
                for k, v in losses_dict.items():
                    if k.startswith("px_"):
                        frame_px[k] = v
                px_data[frame_idx] = frame_px

            rgb_loss_value = losses_dict["rgb_loss"]
            ssim_loss_value = losses_dict["ssim_loss"]
            depth_loss_value = losses_dict["depth_loss"]
            normals_loss_value = losses_dict["normals_loss"]

            # Accumulate weighted losses
            total_rgb_loss += frame_weight * losses.rgb_weight * rgb_loss_value
            total_ssim_loss += frame_weight * losses.rgb_ssim_weight * ssim_loss_value
            total_depth_loss += frame_weight * losses.depth_weight * depth_loss_value
            total_normals_loss += frame_weight * losses.normals_weight * normals_loss_value
            total_perceptual_loss += frame_weight * losses.perceptual_weight * losses_dict["perceptual_loss"]

            # Silhouette loss (composite alpha vs union mask)
            total_silhouette_loss += frame_weight * losses.silhouette_weight * (
                losses.silhouette_com_weight * losses_dict["silhouette_com"]
                + losses.silhouette_sdt_weight * losses_dict["silhouette_sdt"]
                + losses.silhouette_iou_weight * losses_dict["silhouette_iou"]
            )

            total_weight += frame_weight

            # Per-object regularization
            if losses.regularization_weight > 0:
                for obj_idx in data["present_objects"]:
                    if use_pose_tokens_per_obj.get(obj_idx, False) and frame_idx in opt_6drots.get(obj_idx, {}):
                        reg_rot = ((opt_6drots[obj_idx][frame_idx] - initial_6drots[obj_idx][frame_idx]) ** 2).sum()
                        reg_trans = ((opt_raw_trans[obj_idx][frame_idx] - initial_raw_trans[obj_idx][frame_idx]) ** 2).sum()
                    elif frame_idx in opt_rotations.get(obj_idx, {}):
                        opt_rot_n = opt_rotations[obj_idx][frame_idx] / opt_rotations[obj_idx][frame_idx].norm()
                        init_rot_n = initial_rotations[obj_idx][frame_idx] / initial_rotations[obj_idx][frame_idx].norm()
                        reg_rot = ((opt_rot_n - init_rot_n) ** 2).sum()
                        reg_trans = ((opt_translations[obj_idx][frame_idx] - initial_translations[obj_idx][frame_idx]) ** 2).sum()
                    else:
                        continue
                    total_reg_loss += losses.regularization_weight * (
                        losses.regularization_rotation_weight * reg_rot
                        + losses.regularization_translation_weight * reg_trans
                    )

        if total_weight == 0:
            continue

        # Normalize by total weight
        batch_rgb_loss = total_rgb_loss / total_weight
        batch_ssim_loss = total_ssim_loss / total_weight
        batch_silhouette_loss = total_silhouette_loss / total_weight
        batch_depth_loss = total_depth_loss / total_weight
        batch_normals_loss = total_normals_loss / total_weight
        batch_perceptual_loss = total_perceptual_loss / total_weight
        batch_reg_loss = (
            total_reg_loss / len(batch_frame_indices)
            if losses.regularization_weight > 0
            else torch.tensor(0.0, device=device)
        )

        # Global scale regularization (per-object)
        if losses.regularization_weight > 0:
            for obj_idx in all_obj_indices:
                scale_reg = ((opt_global_scale[obj_idx] - initial_global_scales[obj_idx]) ** 2).sum()
                batch_reg_loss = batch_reg_loss + losses.regularization_weight * losses.regularization_scale_weight * scale_reg

        batch_chamfer_loss = (
            total_chamfer_loss / total_chamfer_weight
            if isinstance(total_chamfer_loss, torch.Tensor) and total_chamfer_weight > 0
            else torch.tensor(0.0, device=device)
        )

        total_loss = (
            batch_rgb_loss
            + batch_ssim_loss
            + batch_silhouette_loss
            + batch_depth_loss
            + batch_normals_loss
            + batch_perceptual_loss
            + batch_reg_loss
            + getattr(losses, "chamfer_weight", 0.0) * batch_chamfer_loss
        )

        # Record loss
        loss_history.append({
            "total": total_loss.item(),
            "rgb": batch_rgb_loss.item(),
            "ssim": batch_ssim_loss.item(),
            "silhouette": batch_silhouette_loss.item(),
            "depth": batch_depth_loss.item() if isinstance(batch_depth_loss, torch.Tensor) else batch_depth_loss,
            "normals": batch_normals_loss.item() if isinstance(batch_normals_loss, torch.Tensor) else batch_normals_loss,
            "perceptual": batch_perceptual_loss.item() if isinstance(batch_perceptual_loss, torch.Tensor) else batch_perceptual_loss,
            "regularization": batch_reg_loss.item() if isinstance(batch_reg_loss, torch.Tensor) else batch_reg_loss,
            "chamfer": batch_chamfer_loss.item(),
        })

        # Track best BEFORE backward/step so saved params match the evaluated loss
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_iteration = iteration
            bp_rotations = {}
            bp_translations = {}
            bp_raw_6drots = {}
            bp_raw_translations = {}
            for oi in all_obj_indices:
                bp_rotations[oi] = {}
                bp_translations[oi] = {}
                if use_pose_tokens_per_obj.get(oi, False):
                    bp_raw_6drots[oi] = {fi: opt_6drots[oi][fi].clone().detach() for fi in opt_6drots[oi]}
                    bp_raw_translations[oi] = {fi: opt_raw_trans[oi][fi].clone().detach() for fi in opt_raw_trans[oi]}
                    with torch.no_grad():
                        for fi in opt_6drots[oi]:
                            _, dec_rot, dec_trans, _ = differentiable_pose_decode(
                                bp_raw_6drots[oi][fi], bp_raw_translations[oi][fi],
                                frozen_raw_scales[oi][fi],
                                scene_scales_all[oi][fi], scene_shifts_all[oi][fi],
                                downsample_factors_all[oi][fi],
                            )
                            bp_rotations[oi][fi] = dec_rot.detach()
                            bp_translations[oi][fi] = dec_trans.detach()
                else:
                    bp_rotations[oi] = {fi: opt_rotations[oi][fi].clone().detach()
                                        for fi in opt_rotations[oi]}
                    bp_translations[oi] = {fi: opt_translations[oi][fi].clone().detach()
                                           for fi in opt_translations[oi]}
            best_params = {
                "global_scales": {oi: opt_global_scale[oi].clone().detach() for oi in all_obj_indices},
                "rotations": bp_rotations,
                "translations": bp_translations,
            }
            # The shared correction rides along with the params it was evaluated
            # against.  Snapshotted HERE, inside the best-tracking branch, so it cannot
            # come from a later iteration than the poses it is composed onto.
            if opt_delta_aa:
                best_params["delta"] = {
                    oi: (opt_delta_aa[oi].clone().detach(),
                         opt_delta_logs[oi].clone().detach(),
                         opt_delta_t[oi].clone().detach())
                    for oi in opt_delta_aa
                }
            if bp_raw_6drots:
                best_params["raw_6drots"] = bp_raw_6drots
                best_params["raw_translations"] = bp_raw_translations

        # Backprop
        total_loss.backward()
        optimizer.step()

        # Save per-pixel debug plot every 50 iterations
        if want_pixelwise and px_data:
            from .visualization import save_pixelwise_loss_plot
            with get_timer().exclude():
                save_pixelwise_loss_plot(px_data, iteration, 0, output_dir)

        # Progress bar
        postfix = {
            "loss": f"{total_loss.item():.4f}",
            "rgb": f"{batch_rgb_loss.item():.4f}",
        }
        if losses.perceptual_weight > 0:
            pval = batch_perceptual_loss.item() if isinstance(batch_perceptual_loss, torch.Tensor) else batch_perceptual_loss
            postfix["lpips"] = f"{pval:.4f}"
        if losses.silhouette_weight > 0:
            postfix["sil"] = f"{batch_silhouette_loss.item():.4f}"
        pbar.set_postfix(postfix)

        # Early stopping: break if no improvement for patience iterations
        if losses.early_stop_patience > 0 and (iteration - best_iteration) >= losses.early_stop_patience:
            if pipeline.verbose:
                print(
                    f"      Early stopping at iteration {iteration} "
                    f"(no improvement for {losses.early_stop_patience} iters, best at {best_iteration})"
                )
            break

    if pipeline.verbose:
        print(f"      Best loss: {best_loss:.6f} (iteration {best_iteration})")
        for oi in all_obj_indices:
            s = best_params["global_scales"][oi]
            s_str = f"{s.item():.6f}"
            print(f"      Object {oi} final scale: {s_str}")

    # ── Build refined tokens ──
    refined_tokens = {}
    for obj_idx in all_obj_indices:
        refined_tokens[obj_idx] = []
        global_scale_3d = best_params["global_scales"][obj_idx].expand(3).reshape(1, 3)

        for frame_idx, decoder_input in tokens_by_object[obj_idx]:
            has_frame = (
                (use_pose_tokens_per_obj.get(obj_idx, False) and frame_idx in opt_6drots.get(obj_idx, {}))
                or frame_idx in opt_rotations.get(obj_idx, {})
            )
            if has_frame and frame_idx in best_params["rotations"].get(obj_idx, {}):
                refined_rotation = best_params["rotations"][obj_idx][frame_idx]
                refined_rotation = refined_rotation / refined_rotation.norm()
                refined_translation = best_params["translations"][obj_idx][frame_idx]
                refined_scale_3d = global_scale_3d

                # Compose the shared correction IN.  Applied per frame inside the render
                # loop but onto a copy, so without this the fitted correction never
                # reaches the tokens: under `correction_granularity: shared` the native params
                # are frozen, and the block would return its input poses unchanged.
                # Before `_sync_raw_from_decoded` below, so raw and decoded agree.
                refined_rotation, refined_translation, refined_scale_3d = _apply_gs_correction(
                    *(best_params.get("delta", {}).get(obj_idx) or (None, None, None)),
                    refined_rotation, refined_translation, refined_scale_3d)

                refined_decoder_input = {
                    "rotation": refined_rotation,
                    "translation": refined_translation,
                    "scale": refined_scale_3d.clone(),
                    "refinement_loss_history": loss_history,
                    "refinement_batch_loss_history": loss_history,
                    "refinement_best_iteration": best_iteration,
                }

                # Carry forward raw modalities — re-derived via the SSI
                # inverse of the optimized decoded pose so the round-trip
                # ``decode(raw) == (rotation, translation, scale)`` holds
                # exactly.  The optimizer here treats global_scale as an
                # independent decoded variable (decoupled from
                # ``frozen_raw_scales``), so the optimized raw scale is
                # stale relative to the final decoded scale, and the
                # optimized raw translation was paired with the frozen raw
                # scale.  ``camera_pose_to_raw_tokens`` produces raw tokens
                # consistent with the final decoded (R, t, global_scale_3d).
                if (use_pose_tokens_per_obj.get(obj_idx, False)
                        and "raw_6drots" in best_params
                        and obj_idx in best_params.get("raw_6drots", {})):
                    obj_raw = (perframe_raw_modalities or {}).get(obj_idx, {})
                    raw_source = obj_raw.get(frame_idx, {})
                    original_raw = dict(raw_source.get("raw_ss_modalities", {}))
                    _sync_raw_from_decoded(
                        original_raw,
                        refined_decoder_input["rotation"],
                        refined_decoder_input["translation"],
                        refined_decoder_input["scale"],
                        raw_source,
                        fallback_raw_6d=best_params["raw_6drots"][obj_idx][frame_idx],
                        fallback_raw_trans=best_params["raw_translations"][obj_idx][frame_idx],
                    )
                    refined_decoder_input["raw_ss_modalities"] = original_raw
                    if raw_source.get("pointmap_scale") is not None:
                        refined_decoder_input["pointmap_scale"] = raw_source["pointmap_scale"]
                        refined_decoder_input["pointmap_shift"] = raw_source["pointmap_shift"]
                    refined_decoder_input["downsample_factor"] = raw_source.get("downsample_factor", 1.0)
                else:
                    _carry_forward_decoder_context(decoder_input, refined_decoder_input)

                refined_tokens[obj_idx].append((frame_idx, refined_decoder_input))
            else:
                refined_tokens[obj_idx].append((frame_idx, decoder_input))

    return refined_tokens


# ---------------------------------------------------------------------------
# MV shared-world-pose refinement
# ---------------------------------------------------------------------------


_NONFINITE_TERM_REPORTED: set = set()


def _term_groups(losses: LossConfig) -> "list[tuple[str, float, list]]":
    """``(metric key, outer weight, [(losses_dict key, inner weight), ...])``.

    THE single place naming which loss terms exist and what each is worth: the
    summed expression, the drop list, and the per-term breakdown all read it.

    A GROUP is exactly one entry of FINETUNE's ``metrics`` dict, which is what
    makes the two granularities fall out of one table:

    * the SUM uses the nesting ``outer * (Σ inner_w * val)``.  Do NOT
      "simplify" this by folding ``outer`` into the inner weights: float
      addition is not associative, so with a non-trivial master weight AND
      non-trivial sub-weights folding changes the number.
    * the DROP test is per inner term on its full contribution
      ``outer * inner_w * val``, so one bad sub-term leaves its siblings intact
      and a zero weight (outer OR inner) cannot smuggle a NaN in via ``0.0*nan``.
    """
    return [
        ("rgb",        1.0, [("rgb_loss",        losses.rgb_weight)]),
        ("ssim",       1.0, [("ssim_loss",       losses.rgb_ssim_weight)]),
        ("silhouette", losses.silhouette_weight, [
            ("silhouette_com", losses.silhouette_com_weight),
            ("silhouette_sdt", losses.silhouette_sdt_weight),
            ("silhouette_iou", losses.silhouette_iou_weight)]),
        ("perceptual", 1.0, [("perceptual_loss", losses.perceptual_weight)]),
        ("depth",      1.0, [("depth_loss",      losses.depth_weight)]),
        ("normals",    1.0, [("normals_loss",    losses.normals_weight)]),
    ]


def _group_contribution(losses_dict, outer, inner, drop=()):
    """``outer * (Σ inner_w * val)`` over the members not in ``drop``.

    Returns ``None`` when every member was dropped, so the caller can leave the
    group out of the sum entirely rather than adding a zero.  ``outer == 1.0``
    skips the multiply: exact either way, but it avoids adding a node to the
    autograd graph per single-member group.
    """
    acc = None
    for key, w in inner:
        if key in drop:
            continue
        term = w * losses_dict[key]
        acc = term if acc is None else acc + term
    if acc is None:
        return None
    return acc if outer == 1.0 else outer * acc


def _aggregate_frame_loss(
    losses_dict: Dict[str, torch.Tensor],
    losses: LossConfig,
    report: "list | None" = None,
    terms_out: "dict | None" = None,
) -> torch.Tensor:
    """Sum the weighted loss terms from ``_compute_frame_loss`` output.

    Mirrors the scalar aggregation used in ``refine_pose_for_frame`` so that
    MV shared-pose optimization uses the same loss surface per frame.

    A NON-FINITE TERM IS DROPPED, NOT PROPAGATED.  The terms are summed, so one
    bad term would make the frame total NaN, and from there the whole chunk and
    the whole step: callers gate their backward on ``loss.item() >= 1e-10``,
    which is False for NaN, so ``torch.autograd.grad`` would never be called and
    EVERY term of EVERY frame in that step would silently lose its gradient.
    FINETUNE calls ``.backward()`` with no finiteness check at all, so a NaN
    would reach ``.grad`` and from there Adam's moments, which never recover.

    Parameters
    ----------
    report
        If given, receives the names of dropped terms, so a caller can attach
        richer context (inputs, ranges) to its own diagnostic.
    terms_out
        If given, receives the per-term weighted tensors at METRIC granularity
        (one entry per group in :func:`_term_groups`, silhouette pre-combined).
        A dropped term reports ``0.0`` here, never NaN: FINETUNE sums
        ``metrics`` to rebuild its returned total, so a NaN would re-poison the
        return value even though the gradient was saved -- and ``0.0`` is the
        honest figure, being what the term actually contributed.
    """
    groups = _term_groups(losses)

    def _assemble(drop=()):
        contribs, total = {}, None
        for key, outer, inner in groups:
            c = _group_contribution(losses_dict, outer, inner, drop)
            contribs[key] = c
            if c is not None:
                total = c if total is None else total + c
        return contribs, total

    contribs, total = _assemble()
    if total is None:                       # no terms at all; nothing to sum
        total = torch.zeros((), device=_any_device(losses_dict))

    def _fill(contribs):
        if terms_out is None:
            return
        zero = torch.zeros((), device=total.device, dtype=total.dtype)
        terms_out.update({k: (zero if c is None else c)
                          for k, c in contribs.items()})

    # Populated BEFORE the fast-path return: consumers need the breakdown on
    # every call, not only when something is non-finite.  Returning early
    # without it would empty FINETUNE's metrics on exactly the HEALTHY path.
    _fill(contribs)

    # ONE sync on the healthy path.  Only when it fails do we pay a test per
    # term to find out which one -- ten `__bool__` calls per frame is not a
    # price worth paying when nothing is wrong.
    if torch.isfinite(total):
        return total

    dropped = [k for _, outer, inner in groups for k, w in inner
               if not torch.isfinite(outer * w * losses_dict[k]).all()]
    if report is not None:
        report.extend(dropped)
    for name in dropped:
        if name not in _NONFINITE_TERM_REPORTED:
            _NONFINITE_TERM_REPORTED.add(name)
            print(f"    [loss] NON-FINITE term '{name}' DROPPED from the frame "
                  f"total (reported once per term per process). The remaining "
                  f"terms keep their gradient.")
    contribs, kept_total = _assemble(drop=set(dropped))
    _fill(contribs)
    if kept_total is None:
        return torch.zeros((), device=total.device, dtype=total.dtype)
    return kept_total


def _any_device(losses_dict: Dict[str, torch.Tensor]) -> torch.device:
    for v in losses_dict.values():
        if torch.is_tensor(v):
            return v.device
    return torch.device("cpu")


def _prepare_shared_world_frame_data(
    sequence: Any,
    obj_idx: int,
    frame_indices: List[int],
    losses: LossConfig,
    device: torch.device,
) -> Dict[int, Dict[str, Any]]:
    """Per-frame GT tensors needed by ``_compute_frame_loss``, keyed by frame idx.

    Skips frames where the object is absent (no mask coverage). Moves arrays
    onto ``device`` once to avoid per-iteration transfer cost.
    """
    frame_data: Dict[int, Dict[str, Any]] = {}
    need_depth = losses.depth_weight > 0 or losses.normals_weight > 0
    for fi in frame_indices:
        data = _prepare_frame_data_for_refinement(sequence, fi, obj_idx)
        if data is None:
            continue
        mask = torch.from_numpy(np.asarray(data["mask"])).bool().to(device)
        gt_image = torch.from_numpy(data["image"]).float().to(device) / 255.0
        sdt = None
        if losses.silhouette_weight > 0 and losses.silhouette_sdt_weight > 0:
            sdt = _compute_signed_distance_transform(mask)
        gt_depth_t = None
        valid_mask_t = None
        if data["gt_depth"] is not None and need_depth:
            gt_depth_t = torch.from_numpy(data["gt_depth"]).float().to(device)
            if data["valid_mask"] is not None:
                valid_mask_t = torch.from_numpy(data["valid_mask"]).bool().to(device)
        K_tensor = None
        if losses.normals_weight > 0:
            K_tensor = torch.from_numpy(data["K_matrix"]).float().to(device)
            if data["valid_mask"] is not None and valid_mask_t is None:
                valid_mask_t = torch.from_numpy(data["valid_mask"]).bool().to(device)
        # Optional Chamfer: object's visible GT surface points (unproject GT
        # depth → camera-space pointmap, keep mask ∩ valid ∩ finite), once.
        chamfer_gt_pts = None
        if getattr(losses, "chamfer_weight", 0.0) > 0 and data["gt_depth"] is not None:
            # sequence[fi].masks/.K_matrix (NOT data[...], which may be
            # render-resolution) -- data["gt_depth"] is always at the backbone's
            # (H', W'), and depth-unprojected points must match it exactly.
            chamfer_gt_pts = _chamfer_gt_points_from_depth(
                data["gt_depth"], sequence[fi].K_matrix,
                sequence[fi].masks[obj_idx], data["valid_mask"],
                int(getattr(losses, "chamfer_max_gt_points", 2048)), device,
                trim_factor=resolve_chamfer_gt_trim(losses, "global_pose_refine"))
        frame_data[fi] = {
            "gt_image": gt_image,
            "mask": mask,
            "sdt": sdt,
            "gt_depth": gt_depth_t,
            "valid_mask": valid_mask_t,
            "K_tensor": K_tensor,
            "K_matrix": data["K_matrix"],
            "H": data["H"],
            "W": data["W"],
            "chamfer_gt_pts": chamfer_gt_pts,
        }
    return frame_data


# Cap on the fused multi-view background point cloud (V views × full-image
# pointmap) the shared-world path re-projects into each view for full-image
# depth.  Bounds per-frame render cost + memory regardless of view count.
_SHARED_WORLD_BG_MAX_POINTS = 150000


def _build_fused_mv_background(
    sequence: Any, frame_data: Dict[int, Dict[str, Any]], obj_idx: int,
    device: torch.device,
) -> Dict[int, Tuple[torch.Tensor, ...]]:
    """Fuse every view's pointmap background (this object excluded) into one
    shared-world cloud, then re-project it into each view's camera.

    The shared-world path renders ONE object foreground-only, so (unlike the
    composite path) it has no background — full-image depth (depth_mask_only=
    False) needs one.  Building it from the union of all views fills the holes a
    single view can't see; all params are DETACHED (fixed depth reference, no
    pose gradient).  Capped at ``_SHARED_WORLD_BG_MAX_POINTS``.

    Returns ``{frame_idx: (xyz, rot, scale, opac, feat)}`` in each view's camera
    space, or ``{}`` when no view has background geometry (→ masked-depth
    fallback in the caller).
    """
    from .gaussian import create_background_gaussians
    from .rendering import transform_gaussian_params_world_to_cam
    w_xyz, w_rot, w_scale, w_opac, w_feat = [], [], [], [], []
    for fi in frame_data:
        frame = sequence[fi]
        bg_gs = create_background_gaussians(
            frame.image, frame.pointmap, [frame.masks[obj_idx]],
            frame.K_matrix, c2w=frame.c2w,
        )
        if bg_gs is None:
            continue
        w_xyz.append(bg_gs.get_xyz.detach().to(device))
        w_rot.append(bg_gs.get_rotation.detach().to(device))
        w_scale.append(bg_gs.get_scaling.detach().to(device))
        w_opac.append(bg_gs.get_opacity.squeeze(-1).detach().to(device))
        w_feat.append(bg_gs.get_features.detach().to(device))
    if not w_xyz:
        print(f"      obj {obj_idx}: no background geometry in any view "
              f"→ masked-depth fallback")
        return {}
    f_xyz, f_rot = torch.cat(w_xyz), torch.cat(w_rot)
    f_scale, f_opac, f_feat = torch.cat(w_scale), torch.cat(w_opac), torch.cat(w_feat)
    if f_xyz.shape[0] > _SHARED_WORLD_BG_MAX_POINTS:
        sel = torch.randperm(f_xyz.shape[0], device=device)[:_SHARED_WORLD_BG_MAX_POINTS]
        f_xyz, f_rot, f_scale, f_opac, f_feat = (
            f_xyz[sel], f_rot[sel], f_scale[sel], f_opac[sel], f_feat[sel])
    # Re-project the shared-world union into each view's camera once (fixed → no
    # per-iteration transform); the shared-world loop renders in camera space.
    bg_cam_per_frame: Dict[int, Tuple[torch.Tensor, ...]] = {}
    for fi in frame_data:
        c_xyz, c_rot = transform_gaussian_params_world_to_cam(
            f_xyz, f_rot, sequence[fi].c2w)
        bg_cam_per_frame[fi] = (c_xyz.detach(), c_rot.detach(), f_scale, f_opac, f_feat)
    print(f"      obj {obj_idx}: fused MV background {f_xyz.shape[0]} pts "
          f"→ full-image depth on {len(bg_cam_per_frame)} view(s)")
    return bg_cam_per_frame


def _apply_shared_correction(delta, R_ref, t_ref, s_ref):
    """Compose the shared Sim(3) onto a reference pose, in the object's CANONICAL frame.

    ``delta`` is the ``(aa, log_ds, dt)`` triple, or a triple of ``None`` when
    ``correction_granularity: per_frame`` -- in which case this is the identity and returns its
    inputs unchanged, so the caller needs no branch.

    Applied to the REFERENCE rather than to each derived frame because an object-frame
    correction COMMUTES with the c2w derivation: ``derive_perframe_pose_from_shared_decoded``
    is ``R_i = R_ref @ M33.T`` / ``t_i = M33 @ t_ref + M3``, and ``v @ A.T == A @ v`` for a
    1-D ``v``, so composing before or after the derive gives the same ``(R, t, s)``.  That
    is also why the correction preserves the shared-world invariant: every frame receives
    the identical object-frame transform.

    Row-convention throughout and NO transposes -- ``R_ref`` here is already the matrix the
    renderer right-multiplies by, which is what ``compose_root_delta`` documents.
    """
    aa, log_ds, dt = delta if delta is not None else (None, None, None)
    if aa is None:
        return R_ref, t_ref, s_ref
    # Imported here rather than at module scope: pose_refit reaches back into this module
    # (for the renderer and the rendered-chamfer), so a top-level import would make the
    # pair mutually importable and order-dependent.
    from pytorch3d.transforms import axis_angle_to_matrix

    from .pose_refit import compose_root_delta

    R_new, s_new, t_new = compose_root_delta(
        axis_angle_to_matrix(aa), torch.exp(log_ds).reshape(()), dt,
        R_ref, s_ref.reshape(-1).mean(), t_ref.reshape(3),
    )
    return R_new, t_new, s_new.reshape(1).expand(3).contiguous()


def _apply_gs_correction(aa, log_ds, dt, q, t, s):
    """Quaternion-flavoured wrapper around :func:`_apply_shared_correction`.

    The composition lives in ONE place; this only adapts the shapes the per-frame solvers
    carry (wxyz quaternion + ``(1,3)``/``(3,)`` translation and scale) to the row-matrix
    form ``compose_root_delta`` takes, and back.  Identity when ``aa is None``
    (``correction_granularity: per_frame``), so callers need no branch.
    """
    if aa is None:
        return q, t, s
    R_new, t_new, s_new = _apply_shared_correction(
        (aa, log_ds, dt),
        quaternion_to_matrix(q.reshape(1, 4))[0], t.reshape(3), s.reshape(-1),
    )
    return (matrix_to_quaternion(R_new.contiguous().unsqueeze(0)).reshape(1, 4),
            t_new.reshape(1, 3), s_new.reshape(1, -1))


def prefer_present(primary, fallback):
    """First of `primary`, `fallback` that is present -- WITHOUT a truth test.

    `primary or fallback` forces `bool(primary)`, which RAISES on a multi-element
    tensor. The SSI params (`pointmap_scale` / `pointmap_shift`) are exactly that
    on an MV run once APPEARANCE_INIT has populated `perframe_raw_modalities`
    (one entry per view).

    An empty dict still falls through, preserving the `or` behaviour the
    `raw_ss_modalities` lookup relied on.
    """
    if primary is None:
        return fallback
    if isinstance(primary, dict) and not primary:
        return fallback
    return primary


def refine_poses_shared_world_for_sequence(
    canonical_gaussians: Dict[int, Any],
    tokens_by_object: Dict[int, List[Tuple[int, Dict[str, Any]]]],
    sequence: Any,
    losses: LossConfig,
    pipeline: PipelineConfig,
    perframe_raw_modalities: Dict[int, Dict[int, Dict[str, Any]]],
    canon_frame_per_object: Optional[Dict[int, int]] = None,
    per_frame_canonical: bool = False,
    output_dir: Optional[str] = None,
    save_renders: bool = True,
    canonical_mesh_verts_per_obj: Optional[Dict[int, torch.Tensor]] = None,
    per_frame_mesh_verts_per_obj: Optional[Dict[int, Dict[int, torch.Tensor]]] = None,
    per_frame_mesh_rotations_per_obj: Optional[Dict[int, Dict[int, torch.Tensor]]] = None,
    canonical_mesh_faces_per_obj: Optional[Dict[int, torch.Tensor]] = None,
    warp_knn_k: int = 4,
    warp_knn_eps: float = 1.0e-8,
    warp_knn_chunk_size: int = 8192,
) -> Dict[int, List[Tuple[int, Dict[str, Any]]]]:
    """MV shared-world-pose refinement: one shared pose per object, all frames
    contribute aggregated loss per iteration.

    Each object is optimized independently. Per iteration: resolve the shared
    reference cam_ref-space pose, derive every frame's cam_i-space pose via a
    fixed c2w rebase, render each frame, sum the per-frame losses, and backprop
    into the shared pose. At the end, best-loss state is propagated to all
    frames via :func:`rebase_object_poses_from_reference`.

    Two modes, by ``pipeline.optimize_pose_tokens``:

    - ``True`` (raw-token chain): the shared pose is the reference frame's raw
      Stage-1 tokens, decoded each step via SSI. Requires
      ``raw_ss_modalities`` + SSI params on the ref-frame decoder_input.
    - ``False`` (decoded): the shared pose is the reference frame's decoded
      ``rotation`` / ``translation`` / ``scale``, optimized directly (no raw
      tokens / SSI).

    An object is rendered from its canonical Gaussian through
    :func:`make_posed_object_renderer`.

    DEFORMING assets are supported.  When the per-canonical-mesh-vertex
    field is supplied, each frame's warped geometry is precomputed ONCE per (obj, frame)
    under ``no_grad`` — the canonical shape is fixed across the solve, so the warp is
    constant — and handed to the renderer per call as ``means_override`` /
    ``rotation_override`` (``_precompute_perframe_gaussian_warps``).  This is what lets a
    deforming object's ONE shared correction be fit from EVERY frame instead of from the
    canonical timestamp alone.

    ``losses.correction_granularity`` is honoured here as on every other solver:
    ``per_frame`` optimises the per-timestamp reference poses, ``shared`` freezes them and
    fits ONE Sim(3) per object in its canonical frame.  The correction is composed onto the
    REFERENCE pose -- it commutes with the c2w derivation, so that is identical to
    composing it on every derived frame, and it preserves the shared-world invariant by
    construction.

    Preconditions: ``pipeline.mv_shared_world_pose=True``. (``sequence.is_mv``
    is enforced at the call site; single-view data degenerates to an identity
    rebase.)
    """
    from .pipeline_state import rebase_object_poses_from_reference, resolve_reference_frame
    from .pose_params import (
        build_decoded_shared_world_pose_params,
        build_shared_world_pose_params,
        derive_perframe_pose_from_shared_decoded,
        precompute_c2w_rebases,
    )

    if not pipeline.mv_shared_world_pose:
        raise RuntimeError(
            "refine_poses_shared_world_for_sequence called with "
            "mv_shared_world_pose=False; dispatch logic is inconsistent."
        )
    # Two modes: raw-token chain (optimize_pose_tokens=True; the reference frame
    # must carry raw_ss_modalities + SSI params) and decoded-pose (False; the
    # shared pose is the reference frame's decoded rotation/translation/scale,
    # no raw tokens / SSI needed).
    decoded_mode = not pipeline.optimize_pose_tokens

    canon_frame_per_object = canon_frame_per_object or {}
    device = torch.device("cuda")
    bg_color = _get_bg_color(pipeline, device)

    all_objs = sorted(tokens_by_object.keys())

    if losses.num_iterations == 0:
        # No optimization: return tokens untouched.
        return {o: list(tokens_by_object[o]) for o in all_objs}

    # ── Per-object setup ──────────────────────────────────────────────
    obj_setup: Dict[int, Dict[str, Any]] = {}
    for obj_idx in all_objs:
        entries = tokens_by_object[obj_idx]
        if not entries:
            continue

        ref_frame = resolve_reference_frame(entries, canon_frame_per_object, obj_idx)

        # Source ref-frame raw tokens + SSI params. Priority: the snapshot in
        # perframe_raw_modalities; otherwise the live decoder_input on the ref
        # frame itself (populated by Stage-1 pose init).
        _pf_map = perframe_raw_modalities if perframe_raw_modalities is not None else {}
        ref_raw_info = _pf_map.get(obj_idx, {}).get(ref_frame)
        ref_di_tmp = next((di for fi, di in entries if fi == ref_frame), None)

        def _pull(source, key):
            return source.get(key) if source is not None else None

        ref_raw_mods = prefer_present(
            _pull(ref_raw_info, "raw_ss_modalities"),
            _pull(ref_di_tmp, "raw_ss_modalities"),
        )
        ref_ps = prefer_present(_pull(ref_raw_info, "pointmap_scale"),
                               _pull(ref_di_tmp, "pointmap_scale"))
        ref_psh = prefer_present(_pull(ref_raw_info, "pointmap_shift"),
                                _pull(ref_di_tmp, "pointmap_shift"))
        ref_dsf = (
            ref_raw_info.get("downsample_factor", 1.0) if ref_raw_info is not None
            else (ref_di_tmp.get("downsample_factor", 1.0) if ref_di_tmp is not None else 1.0)
        )

        # Decoded mode reads the always-present decoded pose off ref_di_tmp; only
        # the raw-token path needs raw_ss_modalities + SSI params validated.
        if not decoded_mode and (ref_raw_mods is None or ref_ps is None or ref_psh is None
                or "6drotation_normalized" not in ref_raw_mods
                or "translation" not in ref_raw_mods
                or "scale" not in ref_raw_mods):
            raise ValueError(
                f"obj {obj_idx}: reference frame {ref_frame} is missing raw_ss_modalities "
                f"(6drotation_normalized / translation / scale) or SSI params "
                f"(pointmap_scale / pointmap_shift). Ensure pose init "
                f"populates raw tokens on the ref-frame decoder_input, or set "
                f"pipeline.optimize_pose_tokens=false."
            )

        # Canonical source selection (shared mode: always use a single canonical,
        # the reference frame's when per_frame_canonical is set).
        if per_frame_canonical:
            per_fr = canonical_gaussians.get(obj_idx, {})
            canonical_gs = per_fr.get(ref_frame) if isinstance(per_fr, dict) else None
        else:
            canonical_gs = canonical_gaussians.get(obj_idx)
        canonical_src = canonical_gs
        if canonical_src is None:
            print(f"      Skipping obj {obj_idx}: no canonical Gaussian "
                  f"for ref frame {ref_frame}")
            continue
        renderer = make_posed_object_renderer(canonical_src, device)

        # Deformation warp, precomputed ONCE per frame: the canonical shape is fixed for
        # the whole solve, so the warp is constant -- the same reasoning (and the same
        # helpers) as the global-scale / composite paths.  Warping inside the render
        # would instead run a kNN per frame per iteration.
        frame_indices = [fi for fi, _ in entries]
        warp_overrides = _perframe_warp_overrides(
            canonical_src, obj_idx, frame_indices, device,
            canonical_mesh_verts_per_obj,
            per_frame_mesh_verts_per_obj,
            per_frame_mesh_rotations_per_obj,
            canonical_mesh_faces_per_obj,
            warp_knn_k=warp_knn_k, warp_knn_eps=warp_knn_eps,
            warp_knn_chunk_size=warp_knn_chunk_size,
        )
        if warp_overrides:
            print(f"      obj {obj_idx}: deformation warp cached for "
                  f"{len(warp_overrides)}/{len(frame_indices)} frame(s)")

        # ``opt_6d`` holds the 6D rotation token (raw mode) or the wxyz quaternion
        # (decoded mode); the loop / write-back branch on ``decoded_mode``.
        if decoded_mode:
            opt_6d, opt_t, opt_s, param_groups = build_decoded_shared_world_pose_params(
                ref_di_tmp,
                float(losses.lr_rotation), float(losses.lr_translation), float(losses.lr_scale),
                device,
            )
        else:
            opt_6d, opt_t, opt_s, param_groups = build_shared_world_pose_params(
                ref_raw_mods,
                float(losses.lr_rotation), float(losses.lr_translation), float(losses.lr_scale),
                device,
            )
        init_6d = opt_6d.clone().detach()
        init_t = opt_t.clone().detach()
        init_s = opt_s.clone().detach()

        # ---- Shared correction: ONE Sim(3) per object, in its CANONICAL frame ----------
        # Same knob, same three values and the same composer as every other solver.  It is
        # applied to the REFERENCE pose, once, rather than to each derived frame: an
        # object-frame correction COMMUTES with the c2w derivation
        # (`derive_perframe_pose_from_shared_decoded` is `R_i = R_ref @ M33.T`,
        # `t_i = M33 @ t_ref + M3`, and `v @ A.T == A @ v` for a 1-D v), so the two orders
        # are identical -- which is also why it preserves the shared-world invariant by
        # construction.  The scale is isotropic (the decoded builder means-collapses
        # it, the raw branch goes through `differentiable_pose_decode`).
        # Note: `opt_6d`/`opt_t`/`opt_s` are ONE pose per object for the whole
        # SEQUENCE (a single `ref_frame`), so on MV-DYNAMIC data `per_frame` here means
        # per-sequence, not per-timestamp; on MV-STATIC data the two coincide.  That is
        # the TEMPORAL axis; the view axis is handled by
        # `pose_refit._timestamp_delta_index`.
        # (The guard cannot fire here: mv_shared_world_pose=True is a precondition.)
        _pc_mode = resolve_correction_granularity(
            losses, "global_pose_refine", is_mv=sequence.is_mv,
            mv_shared_world_pose=pipeline.mv_shared_world_pose)
        delta_aa = delta_logs = delta_t = None
        if _pc_mode != "per_frame":
            delta_aa, delta_logs, delta_t, _delta_groups = build_correction(
                float(losses.lr_rotation), float(losses.lr_scale),
                float(losses.lr_translation), device)
            _freeze = natives_to_freeze(
                _pc_mode,
                resolve_correction_scale_control(losses, "global_pose_refine"))
            if _freeze == "all":
                # Freeze the natives so the correction is the only thing that moves --
                # what makes the fit well-posed and readable as "the systematic error".
                for _p in (opt_6d, opt_t, opt_s):
                    _p.requires_grad_(False)
                param_groups = _delta_groups
            else:
                if _freeze == "scale":
                    # The shared correction owns SIZE; hold the native scale.
                    param_groups = freeze_params(param_groups, [opt_s])
                param_groups = param_groups + _delta_groups
            print(f"      obj {obj_idx}: pose correction {_pc_mode} (one Sim(3))")

        c2w_rebase = precompute_c2w_rebases(sequence, frame_indices, ref_frame, device)
        frame_data = _prepare_shared_world_frame_data(
            sequence, obj_idx, frame_indices, losses, device,
        )
        if not frame_data:
            print(f"      Skipping obj {obj_idx}: no frames with mask coverage")
            continue

        # Optional fused multi-view background so the (foreground-only) shared-
        # world render carries valid depth outside the object → full-image depth
        # (depth_mask_only=False).  Empty → masked-depth fallback in the loop.
        _want_bg = bool(getattr(losses, "render_with_background", False))
        bg_cam_per_frame = (
            _build_fused_mv_background(sequence, frame_data, obj_idx, device)
            if _want_bg else {}
        )

        optimizer = torch.optim.Adam(param_groups) if param_groups else None

        obj_setup[obj_idx] = {
            "ref_frame": ref_frame,
            "renderer": renderer,
            "opt_6d": opt_6d, "opt_t": opt_t, "opt_s": opt_s,
            "delta": (delta_aa, delta_logs, delta_t),
            "init_6d": init_6d, "init_t": init_t, "init_s": init_s,
            "ref_ps": ref_ps, "ref_psh": ref_psh, "ref_dsf": ref_dsf,
            "c2w_rebase": c2w_rebase,
            "warp_overrides": warp_overrides,
            "frame_data": frame_data,
            "bg_cam_per_frame": bg_cam_per_frame,
            "optimizer": optimizer,
            "best_loss": float("inf"),
            "best_iteration": 0,
            "best_state": None,
            "loss_history": [],
        }

    if not obj_setup:
        return {o: list(tokens_by_object[o]) for o in all_objs}

    print(f"\n  Shared-world-pose refinement: {len(obj_setup)} object(s), "
          f"{losses.num_iterations} iters/obj")

    # ── Optimization loop (per-object, independent) ──────────────────
    total_steps = losses.num_iterations * len(obj_setup)
    pbar = tqdm(total=total_steps, desc="  Shared-world refine", leave=True, ncols=120)
    for obj_idx, setup in obj_setup.items():
        pbar.set_description(f"  obj {obj_idx}")
        chamfer_model_idx = (
            _chamfer_model_subsample_idx(
                setup["renderer"].num_model_points(),
                int(getattr(losses, "chamfer_max_model_points", 8192)), device)
            if getattr(losses, "chamfer_weight", 0.0) > 0 else None
        )
        for iteration in range(losses.num_iterations):
            if setup["optimizer"] is not None:
                setup["optimizer"].zero_grad()

            # Shared reference cam-space pose (row-convention matrix). Raw mode
            # decodes the tokens via SSI; decoded mode uses the optimized
            # quaternion/translation/scale directly.
            if decoded_mode:
                q_ref = torch.nn.functional.normalize(setup["opt_6d"].reshape(1, 4), dim=-1)
                R_ref = quaternion_to_matrix(q_ref).squeeze(0)
                t_ref = setup["opt_t"].reshape(3)
                s_ref = setup["opt_s"].reshape(1).expand(3).contiguous()  # isotropic
            else:
                matrix_ref, _quat_ref, trans_ref, scale_ref = differentiable_pose_decode(
                    setup["opt_6d"], setup["opt_t"], setup["opt_s"],
                    setup["ref_ps"], setup["ref_psh"], setup["ref_dsf"],
                )
                R_ref = matrix_ref.squeeze(0)
                t_ref = trans_ref.squeeze(0)
                s_ref = scale_ref.squeeze(0)

            # ONE site for the correction, covering both branches: composing it here and
            # deriving is identical to deriving and composing on every frame (see the
            # commuting note at the parameter setup).
            R_ref, t_ref, s_ref = _apply_shared_correction(
                setup["delta"], R_ref, t_ref, s_ref)

            sum_rgb = torch.zeros((), device=device)
            sum_ssim = torch.zeros((), device=device)
            sum_silh = torch.zeros((), device=device)
            sum_depth = torch.zeros((), device=device)
            sum_normals = torch.zeros((), device=device)
            sum_perceptual = torch.zeros((), device=device)
            sum_chamfer = torch.zeros((), device=device)

            want_pixelwise = (save_renders and output_dir is not None
                              and iteration % 50 == 0)
            px_data: Optional[Dict[int, Dict[str, Any]]] = (
                {} if want_pixelwise else None
            )

            total_loss = torch.zeros((), device=device)
            n_frames = 0
            for fi, data in setup["frame_data"].items():
                R_i, t_i, s_i = derive_perframe_pose_from_shared_decoded(
                    R_ref, t_ref, s_ref, setup["c2w_rebase"][fi],
                )
                q_i = matrix_to_quaternion(R_i.unsqueeze(0))  # (1, 4)

                _bg = setup["bg_cam_per_frame"].get(fi)
                # This frame's pre-warped geometry when the object deforms; (None, None)
                # otherwise, which is the static canonical and bit-identical to omitting
                # the arguments.  Precomputed once per frame -- see the setup above.
                _ovr = setup["warp_overrides"].get(fi)
                _means_ovr, _rot_ovr = _ovr if _ovr is not None else (None, None)
                rgb, alpha, depth = setup["renderer"].render(
                    q_i, t_i.unsqueeze(0), s_i.unsqueeze(0),
                    data["K_matrix"], data["W"], data["H"], device, bg_color=bg_color,
                    background_params=_bg,
                    means_override=_means_ovr, rotation_override=_rot_ovr,
                )
                # Full-image depth only where a fused background exists this frame:
                # all-ones loss mask (GT depth not zeroed outside the object) +
                # has_background=True.  Without a background, fall back to masked
                # depth (object mask + depth_mask_only=True) so empty-bg pixels
                # aren't supervised as noise.  Silhouette is inert either way in
                # full-image mode (config sets silhouette_weight=0).
                _has_bg = _bg is not None
                _loss_mask = (
                    torch.ones(data["H"], data["W"], device=device, dtype=torch.bool)
                    if _has_bg else data["mask"]
                )
                losses_dict = _compute_frame_loss(
                    rgb, alpha, data["gt_image"], _loss_mask, losses,
                    data["sdt"], rendered_depth=depth, gt_depth=data["gt_depth"],
                    valid_mask=data["valid_mask"], bg_color=bg_color,
                    K_matrix=data["K_tensor"],
                    return_pixelwise=want_pixelwise,
                    has_background=_has_bg,
                    depth_mask_only=not _has_bg,
                )
                total_loss = total_loss + _aggregate_frame_loss(losses_dict, losses)
                n_frames += 1

                # One-directional Chamfer (GT-depth → posed model means), optional.
                if getattr(losses, "chamfer_weight", 0.0) > 0 and data.get("chamfer_gt_pts") is not None:
                    # The SAME override the render used: a Chamfer term measuring the
                    # undeformed canonical while the photometric terms measure the
                    # deformed frame would be two halves of one loss disagreeing about
                    # the object's shape.
                    _model_xyz = setup["renderer"].model_points(
                        q_i, t_i.unsqueeze(0), s_i.unsqueeze(0), device,
                        means_override=_means_ovr)
                    _cham = getattr(losses, "chamfer_weight", 0.0) * _chamfer_gt_to_model(
                        data["chamfer_gt_pts"], _model_xyz, model_sub_idx=chamfer_model_idx)
                    total_loss = total_loss + _cham
                    sum_chamfer = sum_chamfer + _cham

                if want_pixelwise:
                    _K = data["K_tensor"]
                    if _K is None:
                        _K = torch.from_numpy(data["K_matrix"]).float().to(device)
                    # Mask the GT panels with the SAME mask the loss used, so the
                    # viz matches what was supervised: object-only in masked mode,
                    # full-image (background included) in full-image / has_bg mode.
                    _mask_f = _loss_mask.float().unsqueeze(-1)
                    gt_masked = (
                        data["gt_image"] * _mask_f
                        + bg_color.view(1, 1, 3) * (1.0 - _mask_f)
                    )
                    frame_px: Dict[str, Any] = {
                        "rendered_rgb": rgb.detach().cpu().numpy(),
                        "gt_rgb": gt_masked.detach().cpu().numpy(),
                        "mask": _loss_mask.cpu().numpy(),
                    }
                    if depth is not None:
                        depth_sq = depth.squeeze(0) if depth.dim() == 3 else depth
                        frame_px["rendered_depth"] = depth_sq.detach().cpu().numpy()
                        frame_px["rendered_normals"] = depth_to_normals(
                            depth_sq, _K,
                        ).detach().cpu().numpy()
                    gt_d = data["gt_depth"]
                    if gt_d is not None:
                        _, _K_gt, _mask_gt = _match_depth_grid(
                            _loss_mask, gt_d.shape[-2:], K_matrix=_K, mask=_loss_mask)
                        frame_px["gt_depth"] = (gt_d * _mask_gt.float()).cpu().numpy()
                        frame_px["gt_normals"] = depth_to_normals(
                            gt_d, _K_gt, mask=_mask_gt,
                        ).detach().cpu().numpy()
                    for k, v in losses_dict.items():
                        if k.startswith("px_"):
                            frame_px[k] = v
                    px_data[fi] = frame_px

                sum_rgb = sum_rgb + losses.rgb_weight * losses_dict["rgb_loss"]
                sum_ssim = sum_ssim + losses.rgb_ssim_weight * losses_dict["ssim_loss"]
                sum_silh = sum_silh + losses.silhouette_weight * (
                    losses.silhouette_com_weight * losses_dict["silhouette_com"]
                    + losses.silhouette_sdt_weight * losses_dict["silhouette_sdt"]
                    + losses.silhouette_iou_weight * losses_dict["silhouette_iou"]
                )
                sum_depth = sum_depth + losses.depth_weight * losses_dict["depth_loss"]
                sum_normals = sum_normals + losses.normals_weight * losses_dict["normals_loss"]
                sum_perceptual = sum_perceptual + losses.perceptual_weight * losses_dict["perceptual_loss"]

            denom = max(n_frames, 1)
            total_loss = total_loss / denom

            reg_val = torch.zeros((), device=device)
            if losses.regularization_weight > 0:
                reg = (
                    losses.regularization_rotation_weight * ((setup["opt_6d"] - setup["init_6d"]) ** 2).sum()
                    + losses.regularization_translation_weight * ((setup["opt_t"] - setup["init_t"]) ** 2).sum()
                    + losses.regularization_scale_weight * ((setup["opt_s"] - setup["init_s"]) ** 2).sum()
                )
                reg_val = losses.regularization_weight * reg
                total_loss = total_loss + reg_val

            if setup["optimizer"] is not None:
                total_loss.backward()
                setup["optimizer"].step()

            if want_pixelwise and px_data:
                from .visualization import save_pixelwise_loss_plot
                with get_timer().exclude():
                    save_pixelwise_loss_plot(px_data, iteration, obj_idx, output_dir)

            loss_val = float(total_loss.detach())
            _to_float = lambda v: v.item() if isinstance(v, torch.Tensor) else float(v)
            setup["loss_history"].append({
                "iteration": iteration,
                "total": loss_val,
                "rgb": _to_float(sum_rgb) / denom,
                "ssim": _to_float(sum_ssim) / denom,
                "silhouette": _to_float(sum_silh) / denom,
                "depth": _to_float(sum_depth) / denom,
                "normals": _to_float(sum_normals) / denom,
                "perceptual": _to_float(sum_perceptual) / denom,
                "chamfer": _to_float(sum_chamfer) / denom,
                "regularization": _to_float(reg_val),
            })
            if loss_val < setup["best_loss"]:
                setup["best_loss"] = loss_val
                setup["best_iteration"] = iteration
                # The correction rides along with the params it was evaluated against,
                # so it can never come from a later iteration than the pose it composes
                # onto.  ``None`` entries under `per_frame` keep the shape uniform.
                _d = setup["delta"]
                setup["best_state"] = (
                    setup["opt_6d"].detach().clone(),
                    setup["opt_t"].detach().clone(),
                    setup["opt_s"].detach().clone(),
                    tuple(None if _x is None else _x.detach().clone() for _x in _d),
                )
            pbar.update(1)
    pbar.close()

    # ── Write-back: apply best state, propagate via rebase ──────────
    refined_tokens: Dict[int, List[Tuple[int, Dict[str, Any]]]] = {}
    for obj_idx in all_objs:
        if obj_idx not in obj_setup:
            refined_tokens[obj_idx] = list(tokens_by_object[obj_idx])
            continue
        setup = obj_setup[obj_idx]
        # A 4-tuple; the fallback fires when every iteration scored NaN
        # (``NaN < inf`` is False), so it must carry the live correction too.
        best_6d, best_t, best_s, best_delta = setup["best_state"] or (
            setup["opt_6d"].detach(), setup["opt_t"].detach(), setup["opt_s"].detach(),
            setup["delta"],
        )
        ref_frame = setup["ref_frame"]

        entries = list(tokens_by_object[obj_idx])
        ref_di = next(di for fi, di in entries if fi == ref_frame)

        pf_raw_obj = perframe_raw_modalities.setdefault(obj_idx, {}) if perframe_raw_modalities is not None else {}
        if decoded_mode:
            # best_6d is the wxyz quaternion; write the decoded ref pose straight
            # into the ref entry (no raw tokens / SSI). rebase_object_poses_from_
            # reference reads it and refreshes raw only where SSI is present.
            #
            # BAKE the correction in.  It was composed onto the reference inside the loop
            # but onto a local, so without this the block returns its INPUT pose under
            # `shared` -- where the natives are frozen -- while its loss history still
            # shows improvement.  Composing on the reference is enough for every frame:
            # the rebase below derives them all from it.
            _q_ref = torch.nn.functional.normalize(best_6d.reshape(1, 4), dim=-1).detach()
            _R_b, _t_b, _s_b = _apply_shared_correction(
                best_delta, quaternion_to_matrix(_q_ref)[0], best_t.reshape(3),
                best_s.reshape(-1))
            ref_di["rotation"] = matrix_to_quaternion(
                _R_b.contiguous().unsqueeze(0)).detach()
            ref_di["translation"] = _t_b.reshape(1, 3).detach()
            ref_di["scale"] = _s_b.reshape(1, -1).detach()
        else:
            # Update reference raw tokens in both tokens_by_object and perframe_raw_modalities.
            ref_raw_mods = ref_di.setdefault("raw_ss_modalities", {})
            ref_raw_mods["6drotation_normalized"] = best_6d
            ref_raw_mods["translation"] = best_t
            ref_raw_mods["scale"] = best_s
            pf_raw_obj.setdefault(ref_frame, {
                "pointmap_scale": setup["ref_ps"],
                "pointmap_shift": setup["ref_psh"],
                "downsample_factor": setup["ref_dsf"],
                "raw_ss_modalities": {},
            })
            pf_ref_raw = pf_raw_obj[ref_frame].setdefault("raw_ss_modalities", {})
            pf_ref_raw["6drotation_normalized"] = best_6d
            pf_ref_raw["translation"] = best_t
            pf_ref_raw["scale"] = best_s

            # Decode ref pose into the ref entry (rebase helper will read this).
            with torch.no_grad():
                _, q_final, t_final, s_final = differentiable_pose_decode(
                    best_6d, best_t, best_s,
                    setup["ref_ps"], setup["ref_psh"], setup["ref_dsf"],
                )
            # Same bake, on the decoded pose.  The raw tokens written above stay
            # UNcorrected, which is harmless: the rebase re-derives them from this decoded
            # pose via `camera_pose_to_raw_tokens`, so raw and decoded end up consistent.
            _R_b, _t_b, _s_b = _apply_shared_correction(
                best_delta, quaternion_to_matrix(q_final.reshape(1, 4))[0],
                t_final.reshape(3), s_final.reshape(-1))
            ref_di["rotation"] = matrix_to_quaternion(
                _R_b.contiguous().unsqueeze(0)).detach()
            ref_di["translation"] = _t_b.reshape(1, 3).detach()
            ref_di["scale"] = _s_b.reshape(1, -1).detach()

        # Propagate reference pose to every frame (updates decoded + raw tokens).
        # scope="all": this solve produces ONE Sim(3) for the whole sequence -- that is what
        # "shared world" means here -- so it reaches every frame, not just the reference's
        # timestamp.  pose_init's rebase is the other axis (motion in the per-frame Sim(3)).
        rebase_object_poses_from_reference(
            entries, pf_raw_obj, sequence, ref_frame, scope="all")

        # Annotate entries with loss history for save_and_plot_loss_history.
        # All frames share one optimization, so every frame points at the same
        # per-object history; setting batch_loss_history selects one-row-per-object
        # plotting in plot_refinement_history.
        for _, di in entries:
            di["refinement_loss_history"] = setup["loss_history"]
            di["refinement_batch_loss_history"] = setup["loss_history"]
            di["refinement_best_iteration"] = setup["best_iteration"]

        refined_tokens[obj_idx] = entries

    return refined_tokens


def refine_poses_for_sequence(
    object_gaussians: Dict[int, Any],
    tokens_by_object: Dict[int, List[Tuple[int, Dict[str, Any]]]],
    sequence: Any,
    losses: LossConfig,
    pipeline: PipelineConfig,
    refine_scale: str = "perframe",
    per_frame_canonical: bool = False,
    perframe_raw_modalities: Optional[Dict[int, Dict[int, Dict[str, Any]]]] = None,
    output_dir: Optional[str] = None,
    save_renders: bool = True,
    canon_frame_per_object: Optional[Dict[int, int]] = None,
    # Per-canonical-mesh-vertex deformation field (actionmesh mono-dynamic).
    # State-shaped dicts: canonical_mesh_verts_per_obj[obj_idx] -> (V, 3),
    # per_frame_mesh_verts_per_obj[obj_idx][frame_int] -> (V, 3),
    # per_frame_mesh_rotations_per_obj[obj_idx][frame_int] -> (V, 3, 3).
    # When supplied, the canonical Gaussian is warped per (obj, frame)
    # via ``warp_gaussians_high_res`` BEFORE the Stage-1 rigid pose is
    # applied — consumed by the global-scale, composite and shared-world
    # paths.  Identity-default ``None`` collapses to the rigid render.
    canonical_mesh_verts_per_obj: Optional[Dict[int, torch.Tensor]] = None,
    per_frame_mesh_verts_per_obj: Optional[Dict[int, Dict[int, torch.Tensor]]] = None,
    per_frame_mesh_rotations_per_obj: Optional[Dict[int, Dict[int, torch.Tensor]]] = None,
    canonical_mesh_faces_per_obj: Optional[Dict[int, torch.Tensor]] = None,
    warp_knn_k: int = 4,
    warp_knn_eps: float = 1.0e-8,
    warp_knn_chunk_size: int = 8192,
) -> Dict[int, List[Tuple[int, Dict[str, Any]]]]:
    """
    Refine per-frame poses for all objects using differentiable rendering.

    Parameters
    ----------
    object_gaussians : dict
        The Gaussians rendered to drive pose refinement. NOT necessarily
        canonical: with ``per_frame_canonical=True`` (the per-frame
        ``refine_geometry: own_frame`` scope) this is the per-frame predictions
        ``{obj_idx: {frame_idx: Gaussian}}``; with
        ``per_frame_canonical=False`` it is one shared Gaussian per object
        ``{obj_idx: Gaussian}``.
    tokens_by_object : dict
        Dictionary mapping object_index -> list of (frame_index, decoder_input).
    sequence : Sequence
        Cached scene data.
    losses : LossConfig
        Per-phase loss weights and iteration count.
    pipeline : PipelineConfig
        Cross-phase pipeline control flags.
    refine_scale : str
        Scale refinement mode: "perframe" (per-frame scale optimization) or
        "global" (shared scale across all frames via batch optimization).
    per_frame_canonical : bool, optional
        If True, use per-frame canonical Gaussians (standard mode).
        If False, use shared canonical Gaussians across frames (averaged-tokens mode).
    perframe_raw_modalities : dict, optional
        Raw Stage 1 modalities per frame, structured as
        ``{obj_idx: {frame_idx: {"raw_ss_modalities": {...}, "pointmap_scale": ...,
        "pointmap_shift": ..., "downsample_factor": ...}}}``.
        Used for pose token optimization in global refinement mode.
    save_renders : bool
        False skips the per-iteration ``debug_pixelwise/`` PNGs (and the CPU copies
        feeding them) in every branch.  Pass the block's RESOLVED flag,
        ``get_block_output_flag(cfg, block, "save_renders")`` — raw
        ``cfg.output.save_renders`` ignores ``suppress_intermediate_renders``.

    Returns
    -------
    dict
        Refined tokens_by_object with updated poses.
    """
    if object_gaussians is None:
        raise RuntimeError(
            "refine_poses_for_sequence: no render source is available "
            "(object_gaussians is None). "
            "Per-frame refinement renders decoded per-frame Gaussians, but none could be "
            "decoded."
        )

    # ── Deformation-field validation ──
    # The three deformation dicts must be supplied together (matches the
    # FINETUNE / Stage-2 rendering-guidance contract).  Asymmetric supply
    # would silently fall back to rigid rendering inside
    # ``_lookup_per_frame_deformation`` — raise here so misuse surfaces
    # at the entry point.
    _has_canon_mesh = canonical_mesh_verts_per_obj is not None
    _has_pf_verts = per_frame_mesh_verts_per_obj is not None
    _has_pf_R = per_frame_mesh_rotations_per_obj is not None
    if not (_has_canon_mesh == _has_pf_verts == _has_pf_R):
        raise ValueError(
            "refine_poses_for_sequence: canonical_mesh_verts_per_obj, "
            "per_frame_mesh_verts_per_obj and per_frame_mesh_rotations_per_obj "
            "must be supplied together "
            f"(got canonical_mesh_verts_per_obj={'set' if _has_canon_mesh else 'None'}, "
            f"per_frame_mesh_verts_per_obj={'set' if _has_pf_verts else 'None'}, "
            f"per_frame_mesh_rotations_per_obj={'set' if _has_pf_R else 'None'})."
        )
    # The global-scale, composite AND shared-world paths consume the deformation field.
    # The per-frame path does not, so reject silently-dropped supply there.
    _deformation_supplied = _has_canon_mesh and bool(canonical_mesh_verts_per_obj)
    if _deformation_supplied:
        if refine_scale != "global" and not pipeline.mv_shared_world_pose:
            raise NotImplementedError(
                "refine_poses_for_sequence: per-canonical-mesh-vertex "
                "deformation field is consumed by refine_scale='global' and by the "
                f"shared-world path, but not by the per-frame one (got "
                f"refine_scale={refine_scale!r}, mv_shared_world_pose=False).  The "
                "per-frame path uses per-frame canonical Gaussians directly and does "
                "not warp."
            )

    # MV shared-world-pose: collapse per-camera params to one shared pose per
    # object; aggregated per-frame loss drives a single optimizer. Short-circuits
    # before the per-frame / global dispatch since the shared path supersedes both.
    # (is_mv is asserted at the sequence-construction call site; not redundantly
    # checked here.)
    if pipeline.mv_shared_world_pose:
        # NOTE: ``or {}`` would drop an empty-but-present dict (falsy),
        # breaking the write-back's mutation back into state. Use ``is None``.
        pf_raw = perframe_raw_modalities if perframe_raw_modalities is not None else {}
        return refine_poses_shared_world_for_sequence(
            object_gaussians, tokens_by_object, sequence, losses, pipeline,
            save_renders=save_renders,
            perframe_raw_modalities=pf_raw,
            canon_frame_per_object=canon_frame_per_object,
            per_frame_canonical=per_frame_canonical,
            output_dir=output_dir,
            canonical_mesh_verts_per_obj=canonical_mesh_verts_per_obj,
            per_frame_mesh_verts_per_obj=per_frame_mesh_verts_per_obj,
            per_frame_mesh_rotations_per_obj=per_frame_mesh_rotations_per_obj,
            canonical_mesh_faces_per_obj=canonical_mesh_faces_per_obj,
            warp_knn_k=warp_knn_k,
            warp_knn_eps=warp_knn_eps,
            warp_knn_chunk_size=warp_knn_chunk_size,
        )

    # Global scale: all objects rendered jointly (composite)
    if refine_scale == "global":
        print("\n  Refining poses with COMPOSITE rendering (all objects jointly)...")
        return refine_poses_global_composite(
            object_gaussians, tokens_by_object, sequence, losses, pipeline,
            save_renders=save_renders,
            perframe_raw_modalities=perframe_raw_modalities,
            output_dir=output_dir,
            canonical_mesh_verts_per_obj=canonical_mesh_verts_per_obj,
            per_frame_mesh_verts_per_obj=per_frame_mesh_verts_per_obj,
            per_frame_mesh_rotations_per_obj=per_frame_mesh_rotations_per_obj,
            canonical_mesh_faces_per_obj=canonical_mesh_faces_per_obj,
            warp_knn_k=warp_knn_k,
            warp_knn_eps=warp_knn_eps,
            warp_knn_chunk_size=warp_knn_chunk_size,
        )

    # Per-frame refinement
    print("\n  Refining per-frame poses with differentiable rendering...")

    refined_tokens = {}
    total_frames = sum(len(tl) for tl in tokens_by_object.values())
    n_objects = len(tokens_by_object)
    pbar = tqdm(
        total=total_frames,
        desc="  Per-frame refine",
        leave=True,
        ncols=120,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} frames [{elapsed}<{remaining}]",
    )

    # One adapter per distinct render source, not per frame: a single canonical
    # Gaussian is shared by every frame.  Keying on `id` is safe here because every
    # source belongs to a caller-owned dict that outlives the loop, so none can be freed
    # and have its id reused -- unlike `_block_icp_frames`, which splats its own.
    _renderers: Dict[int, PosedObjectRenderer] = {}

    def _renderer_for(source, device):
        if id(source) not in _renderers:
            _renderers[id(source)] = make_posed_object_renderer(source, device)
        return _renderers[id(source)]

    for obj_idx in sorted(tokens_by_object.keys()):
        refined_tokens[obj_idx] = []

        # An object with no Gaussians passes its tokens through unrefined; one with
        # Gaussians for most frames SKIPS the frames that failed to decode.
        # ``obj_gaussians`` is a {frame: Gaussian} dict under `per_frame_canonical` and a
        # single Gaussian otherwise; test emptiness only on the dict, never with a bare
        # truth test on a Gaussian.
        obj_gaussians = (object_gaussians or {}).get(obj_idx)
        has_gaussians = obj_gaussians is not None and (
            bool(obj_gaussians) if isinstance(obj_gaussians, dict) else True)
        if not has_gaussians:
            refined_tokens[obj_idx].extend(tokens_by_object[obj_idx])
            pbar.update(len(tokens_by_object[obj_idx]))
            continue

        for frame_idx, decoder_input in tokens_by_object[obj_idx]:
            desc = (f"  obj {obj_idx} f{frame_idx}" if n_objects > 1
                    else f"  frame {frame_idx}")
            pbar.set_description(desc)

            # Resolve the render source for this (obj, frame)
            if per_frame_canonical:
                if frame_idx not in obj_gaussians:
                    refined_tokens[obj_idx].append((frame_idx, decoder_input))
                    pbar.update(1)
                    continue
                canonical_gs = obj_gaussians[frame_idx]
            else:
                canonical_gs = obj_gaussians

            frame = sequence[frame_idx]
            render_image, render_masks, K_matrix = frame.image, frame.masks, frame.K_matrix
            mask = render_masks[obj_idx]
            if not mask.any():
                refined_tokens[obj_idx].append((frame_idx, decoder_input))
                pbar.update(1)
                continue

            depth_map_z = frame.depth_map_z

            # Ground truth image
            gt_image = torch.from_numpy(render_image).float().cuda() / 255.0

            # Initial pose
            initial_rotation = decoder_input["rotation"]
            initial_translation = decoder_input["translation"]
            initial_scale = decoder_input["scale"]

            # Extract raw modalities and scene context for pose token optimization
            raw_modalities = decoder_input.get("raw_ss_modalities")
            di_scene_scale = decoder_input.get("pointmap_scale")
            di_scene_shift = decoder_input.get("pointmap_shift")
            di_downsample_factor = decoder_input.get("downsample_factor", 1.0)

            # Refine pose
            refined_pose = refine_pose_for_frame(
                canonical_gs,
                initial_rotation,
                initial_translation,
                initial_scale,
                gt_image,
                mask,
                K_matrix,
                losses=losses,
                pipeline=pipeline,
                refine_scale=refine_scale,
                gt_depth=depth_map_z,
                valid_mask=frame.valid_mask,
                raw_modalities=raw_modalities,
                scene_scale=di_scene_scale,
                scene_shift=di_scene_shift,
                downsample_factor=di_downsample_factor,
                optimize_pose_tokens=pipeline.optimize_pose_tokens,
                output_dir=output_dir,
                save_renders=save_renders,
                obj_idx=obj_idx,
                frame_idx=frame_idx,
                renderer=_renderer_for(canonical_gs, initial_rotation.device),
            )

            # Create refined decoder input
            refined_decoder_input = {
                "rotation": refined_pose["rotation"],
                "translation": refined_pose["translation"],
                "scale": refined_pose["scale"],
                "refinement_loss_history": refined_pose["loss_history"],
                "refinement_best_iteration": refined_pose["best_iteration"],
            }
            # Preserve raw modalities (updated if pose token optimization was used)
            if "raw_modalities" in refined_pose:
                # Merge updated pose keys back into original dict (keeps shape, translation_scale)
                original_raw = decoder_input.get("raw_ss_modalities", {})
                merged = dict(original_raw)
                merged.update(refined_pose["raw_modalities"])
                refined_decoder_input["raw_ss_modalities"] = merged
            elif "raw_ss_modalities" in decoder_input:
                refined_decoder_input["raw_ss_modalities"] = decoder_input["raw_ss_modalities"]
            # Carry forward scene context + slat
            for _cfk in ("pointmap_scale", "pointmap_shift",
                         "downsample_factor", "decoder_input_slat"):
                if _cfk in decoder_input:
                    refined_decoder_input[_cfk] = decoder_input[_cfk]

            refined_tokens[obj_idx].append((frame_idx, refined_decoder_input))
            pbar.update(1)

    pbar.close()
    return refined_tokens


__all__ = [
    "LossConfig",
    "PipelineConfig",
    "apply_pose_to_gaussian",
    "refine_pose_for_frame",
    "refine_poses_global_composite",
    "refine_poses_for_sequence",
]
