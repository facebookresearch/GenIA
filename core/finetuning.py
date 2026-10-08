"""
Canonical token fine-tuning via differentiable rendering.

This module provides the core optimization loop for fine-tuning SLAT tokens
(and optionally a LoRA-adapted decoder) after pose refinement. It backs the
FINETUNE block (``finetuning.enabled``).

Gradient flow::

    opt_feats (requires_grad) -> SparseTensor -> Decoder (frozen / LoRA)
    -> Gaussian attrs -> apply_pose -> gsplat render -> loss -> backward
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from genia.core.utils.config import (
    LossConfig, PipelineConfig,
    resolve_correction_granularity, resolve_correction_scale_control,
)
from genia.core.config import FinetuningConfig
from genia.core.utils.model_cache import ModelCache
from genia.core.utils.timing import get_timer
from genia.core.utils.slat_decode import (
    _get_or_build_wrapped_decoder,
    decode_tokens,
)
from genia.core.utils.gaussian import attach_sh_rest


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


# ---------------------------------------------------------------------------
# Guarded access to optional FinetuningConfig fields
# ---------------------------------------------------------------------------
# OmegaConf DictConfigs in struct mode raise ConfigAttributeError when a
# key is missing (e.g. a YAML that omits the field).  ``getattr`` with a
# default suppresses the error (ConfigAttributeError ⊂ AttributeError) and
# falls back to a default.

def _ft_lora_init(ft_config) -> str:
    return str(getattr(ft_config, "lora_init", "default"))


def _ft_use_dora(ft_config) -> bool:
    return bool(getattr(ft_config, "use_dora", False))


def _ft_rs_scaling(ft_config) -> bool:
    return bool(getattr(ft_config, "lora_rs_scaling", False))


def _ft_lora_targets(ft_config) -> str:
    return str(getattr(ft_config, "lora_targets", "all"))


def _ft_random_bg_seed(ft_config):
    return getattr(ft_config, "random_background_seed", None)


def _ft_decoder_autocast_bf16(ft_config) -> bool:
    # Fallback mirrors the documented default (ON).
    return bool(getattr(ft_config, "decoder_autocast_bf16", True))


def _ft_decoder_checkpoint(ft_config) -> bool:
    return bool(getattr(ft_config, "decoder_checkpoint", False))


def _ft_decoder_checkpoint_min_voxels(ft_config) -> int:
    return int(getattr(ft_config, "decoder_checkpoint_min_voxels", 0))


def _enable_decoder_checkpoint(decoder: "torch.nn.Module | None") -> None:
    """Set ``use_checkpoint=True`` on every submodule that exposes it.

    The transformer-torso blocks captured ``use_checkpoint`` at construction
    and read their own copy live in forward, so a top-level flag won't reach
    them — set it on every submodule.  Frees the torso's retained activation
    graph during the under-grad FINETUNE decode (the OOM term).  no_grad
    decodes (initial/best/export) are ~unaffected (checkpoint is a near-noop
    without grad)."""
    if decoder is None:
        return
    for _m in decoder.modules():
        if hasattr(_m, "use_checkpoint"):
            _m.use_checkpoint = True


def setup_shared_lora_decoder(
    base_decoder: torch.nn.Module,
    rank: int,
    alpha: float,
    init: str = "default",
    use_dora: bool = False,
    rs_scaling: bool = False,
    lora_targets: str = "all",
) -> torch.nn.Module:
    """Deep-copy + freeze + LoRA-wrap a decoder **once** (cache-aware).

    Returns the wrapped decoder, which hosts :class:`LoRALayer` wrappers
    around every matching ``nn.Linear``.  Every Parameter from the
    original layers is frozen.  Per-object :class:`LoRAAdapter` instances
    allocate their own ``(A, B)`` (and optional DoRA ``magnitude``)
    Parameters and bind them via :meth:`LoRAAdapter.activate` — so a
    multi-object run holds **one** decoder copy + N small adapters
    instead of N full decoder copies.

    ``use_dora=True`` enables the DoRA magnitude/direction decomposition
    on every wrapped layer (Liu et al. 2024, arXiv:2402.09353).
    ``rs_scaling=True`` switches per-layer scaling from ``alpha/rank``
    to rsLoRA's ``alpha/√rank`` (Kalajdzievski 2023).

    Within a single process, repeated calls with the same configuration
    reuse a cached wrapped decoder (see ``_get_or_build_wrapped_decoder``).
    Safe to share because ``LoRAAdapter.activate`` rebinds the layers'
    ``lora_A/B/magnitude`` attributes — the shared decoder's "default"
    Parameters from ``apply_lora_to_decoder`` are never read after that.
    """
    return _get_or_build_wrapped_decoder(
        base_decoder, rank, alpha, init, use_dora, rs_scaling, lora_targets,
    )


# ---------------------------------------------------------------------------
# Eval helper: averaged masked PSNR / SSIM for a single decoded Gaussian
# ---------------------------------------------------------------------------

def _compute_mean_psnr_ssim(
    gs: Any,
    poses: Dict[Any, Dict[str, torch.Tensor]],
    frame_indices: List[Any],
    sequence: Any,
    obj_idx: int,
    device: torch.device,
    *,
    bg_color: Optional[torch.Tensor] = None,
    # actionmesh deformation (same contract as ``compute_total_loss``)
    canonical_mesh_verts: "torch.Tensor | None" = None,
    per_frame_mesh_verts: "Dict[int, torch.Tensor] | None" = None,
    per_frame_mesh_rotations: "Dict[int, torch.Tensor] | None" = None,
    canonical_mesh_faces: "torch.Tensor | None" = None,
    warp_knn_k: int = 4,
    warp_knn_eps: float = 1.0e-8,
    warp_knn_chunk_size: int = 131072,
) -> Tuple[Optional[float], Optional[float]]:
    """Mean masked PSNR / SSIM of ``gs`` across all frames.

    Renders ``gs`` per-frame with the supplied poses (optionally warped
    via the per-canonical-mesh-vertex deformation field), then computes
    PSNR / SSIM against the masked GT image — same masking rule the
    FINETUNE loss uses. Returns ``(None, None)`` if no frame contributes.
    """
    from genia.core.utils.refinement import (
        _compute_ssim_loss,
        _prepare_frame_data_for_refinement,
        _render_frame_with_pose,
    )
    use_warp = (
        canonical_mesh_verts is not None
        and per_frame_mesh_verts is not None
        and per_frame_mesh_rotations is not None
    )
    if use_warp:
        from genia.core.utils.deformation import warp_gaussians_high_res

    psnrs: List[float] = []
    ssims: List[float] = []
    with torch.no_grad():
        for fi in frame_indices:
            if fi not in poses:
                continue
            data = _prepare_frame_data_for_refinement(sequence, fi, obj_idx)
            if data is None:
                continue

            means_override = rotation_override = None
            if use_warp:
                fi_int = fi.frame if hasattr(fi, "frame") else int(fi)
                if fi_int in per_frame_mesh_verts:
                    means_override, rotation_override = warp_gaussians_high_res(
                        gs,
                        canonical_mesh_verts,
                        per_frame_mesh_verts[fi_int],
                        per_frame_mesh_rotations[fi_int],
                        K=int(warp_knn_k),
                        eps=float(warp_knn_eps),
                        chunk_size=int(warp_knn_chunk_size),
                        faces=canonical_mesh_faces,
                    )

            rgb, _alpha, _depth = _render_frame_with_pose(
                gs,
                poses[fi]["rotation"], poses[fi]["translation"], poses[fi]["scale"],
                data["K_matrix"], data["W"], data["H"], device,
                bg_color=bg_color,
                means_override=means_override,
                rotation_override=rotation_override,
            )

            # _prepare_frame_data_for_refinement returns numpy ndarrays
            # (uint8 image in [0, 255]; bool mask) — convert + normalize.
            gt = torch.from_numpy(data["image"]).float().to(device) / 255.0
            mask = torch.from_numpy(np.asarray(data["mask"])).bool().to(device)
            mask3 = mask.unsqueeze(-1).float()
            bg = (
                bg_color.view(1, 1, 3)
                if bg_color is not None
                else torch.zeros(1, 1, 3, device=device)
            )
            gt_masked = gt * mask3 + bg * (1.0 - mask3)
            rgb_masked = rgb * mask3 + bg * (1.0 - mask3)

            mse = ((rgb_masked - gt_masked) ** 2).mean()
            psnr = -10.0 * torch.log10(mse + 1e-8)
            ssim = 1.0 - _compute_ssim_loss(rgb_masked, gt_masked)
            psnrs.append(psnr.item())
            ssims.append(ssim.item())

    if not psnrs:
        return None, None
    return sum(psnrs) / len(psnrs), sum(ssims) / len(ssims)


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def _sample_random_bg_color(
    device: torch.device,
    *,
    seed_base: Optional[int],
    iter_idx: int,
    frame_idx: Any,
) -> torch.Tensor:
    """Per-(iter, frame) uniform RGB used for ``random_background``.

    With ``seed_base=None`` (default) we draw from the **global** torch
    RNG.  When ``seed_base`` is set
    (typically the run's master seed), the colour is derived from
    ``hash((seed_base, iter_idx, frame))``: identical across reruns of
    the same config, but still varies per (iter, frame) so the augmentation
    intent — preventing fixed-BG bake-in — is preserved.

    ``frame_idx`` may be a :class:`FrameKey`, an int, or any hashable.
    """
    if seed_base is None:
        return torch.rand(3, device=device)
    fi_key: Any
    if hasattr(frame_idx, "frame") and hasattr(frame_idx, "view"):
        fi_key = (int(frame_idx.frame), int(frame_idx.view))
    else:
        fi_key = int(frame_idx)
    # Stable across PYTHONHASHSEED runs because tuple/int hashing is
    # itself deterministic; modulo keeps the seed inside a Generator's
    # accepted range.
    seed = (hash((int(seed_base), int(iter_idx), fi_key)) & 0xFFFFFFFF)
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.rand(3, generator=g).to(device)


def compute_total_loss(
    canonical_gs: Any,
    poses: Dict[int, Dict[str, torch.Tensor]],
    frame_indices: List[int],
    sequence: Any,
    obj_idx: int,
    losses: LossConfig,
    device: torch.device,
    perceptual_scale: float = 0.5,
    sh_rest_per_frame: Optional[Dict[int, torch.Tensor]] = None,
    dc_offset_per_frame: Optional[Dict[int, torch.Tensor]] = None,
    bg_color: Optional[torch.Tensor] = None,
    return_pixelwise: bool = False,
    microbatch_size: int = 0,
    root_delta: Optional[Dict[str, torch.Tensor]] = None,
    # ── actionmesh: high-res per-canonical-mesh-vertex deformation field ──
    # All three must be supplied together; ``None`` falls back to the
    # rigid-Sim(3)-only render path.  Identical contract to
    # ``build_appearance_rendering_guidance_transform``.
    canonical_mesh_verts: "torch.Tensor | None" = None,
    per_frame_mesh_verts: "Dict[int, torch.Tensor] | None" = None,
    per_frame_mesh_rotations: "Dict[int, torch.Tensor] | None" = None,
    canonical_mesh_faces: "torch.Tensor | None" = None,
    warp_knn_k: int = 4,
    warp_knn_eps: float = 1.0e-8,
    warp_knn_chunk_size: int = 131072,
    # ── per-render BG randomization ──
    random_background: bool = False,
    random_background_seed: Optional[int] = None,
    random_background_iter: int = 0,
) -> Tuple[torch.Tensor, Dict[str, float], Optional[Dict[int, Dict[str, np.ndarray]]]]:
    """Render each frame and accumulate loss.

    Uses the shared functions from ``refinement.py``:
    - ``_prepare_frame_data_for_refinement`` for GT data loading
    - ``_render_frame_with_pose`` for differentiable rendering
    - ``_compute_frame_loss`` for per-frame loss computation

    Parameters
    ----------
    return_pixelwise : bool
        When True, also return per-frame per-pixel error maps (as numpy
        arrays) for debug visualization.

    Returns
    -------
    tuple
        ``(total_loss, metrics, pixelwise_data)`` where *pixelwise_data* is
        ``None`` when *return_pixelwise* is False, or a dict
        ``{frame_idx: {"rendered_rgb": (H,W,3), "gt_rgb": (H,W,3),
        "mask": (H,W), "px_rgb_error": (H,W), ...}}`` when True.
    """
    from genia.core.utils.refinement import (
        _aggregate_frame_loss,
        _compute_frame_loss,
        _prepare_frame_data_for_refinement,
        _render_frame_with_pose,
    )

    # Validate deformation kwargs (asymmetric supply raises early so a
    # half-wired call site doesn't silently fall back to rigid-only).
    _has_canon_mesh = canonical_mesh_verts is not None
    _has_pf_verts = per_frame_mesh_verts is not None
    _has_pf_R = per_frame_mesh_rotations is not None
    if not (_has_canon_mesh == _has_pf_verts == _has_pf_R):
        raise ValueError(
            "compute_total_loss: canonical_mesh_verts, per_frame_mesh_verts "
            "and per_frame_mesh_rotations must be supplied together "
            f"(got canonical_mesh_verts={'set' if _has_canon_mesh else 'None'}, "
            f"per_frame_mesh_verts={'set' if _has_pf_verts else 'None'}, "
            f"per_frame_mesh_rotations={'set' if _has_pf_R else 'None'})."
        )
    use_warp = _has_canon_mesh
    if use_warp:
        from genia.core.utils.deformation import warp_gaussians_high_res

    # Gradient accumulation: process frames in micro-batches, backward per
    # chunk so only one chunk's rendering+VGG graph is in memory at a time.
    # The decode_tokens graph (shared across frames) is retained until the
    # last chunk via retain_graph=True.
    valid_frames = [fi for fi in frame_indices if fi in poses]
    n_total_valid = len(valid_frames)
    use_grad_accum = microbatch_size > 0 and n_total_valid > microbatch_size

    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    metrics = {
        "rgb": 0.0, "ssim": 0.0, "silhouette": 0.0, "depth": 0.0,
        "normals": 0.0, "perceptual": 0.0,
        "total": 0.0,
    }
    n_frames = 0
    pixelwise_data: Optional[Dict[int, Dict[str, np.ndarray]]] = (
        {} if return_pixelwise else None
    )

    _chunk_loss: Optional[torch.Tensor] = None
    _frames_in_chunk = 0

    iter_frames = valid_frames if use_grad_accum else frame_indices
    for vi, frame_idx in enumerate(iter_frames):
        if not use_grad_accum and frame_idx not in poses:
            continue

        # Start a new micro-batch
        if use_grad_accum and _frames_in_chunk == 0:
            _chunk_loss = torch.tensor(0.0, device=device, requires_grad=True)

        frame_data = _prepare_frame_data_for_refinement(sequence, frame_idx, obj_idx)
        if frame_data is None:
            continue

        pose = poses[frame_idx]
        if root_delta is not None:
            # ONE Sim(3) per object composed onto this frame's own pose, so the
            # per-frame motion is preserved rather than replaced.
            pose = compose_pose_with_root_delta(pose, root_delta)
        gt_image = torch.from_numpy(frame_data["image"]).float().to(device) / 255.0
        mask_tensor = torch.from_numpy(frame_data["mask"]).bool().to(device)

        # Apply per-frame DC offset (base color varies per frame)
        _saved_dc = None
        if dc_offset_per_frame is not None and frame_idx in dc_offset_per_frame:
            _saved_dc = canonical_gs._features_dc
            canonical_gs._features_dc = _saved_dc + dc_offset_per_frame[frame_idx]

        # Attach per-frame SH coefficients before rendering — save state
        # so it can be restored after.  The mutation otherwise leaks into
        # the next frame's render in this same iter, which silently
        # corrupts the loss when SH coverage is partial across frames.
        _saved_sh: Optional[Tuple[Optional[torch.Tensor], int, int]] = None
        if sh_rest_per_frame is not None and frame_idx in sh_rest_per_frame:
            _saved_sh = (
                getattr(canonical_gs, "_features_rest", None),
                getattr(canonical_gs, "sh_degree", 0),
                getattr(canonical_gs, "active_sh_degree", 0),
            )
            attach_sh_rest(canonical_gs, sh_rest_per_frame[frame_idx])

        # actionmesh: warp Gaussians via per-canonical-mesh-vertex Φ + R
        # BEFORE applying Stage-1 rigid pose.  Per-frame DC / SH mutations
        # above stay valid — only means / rotation are overridden in
        # apply_pose_to_gaussian; scales, opacity, features pass through
        # unchanged.  Identity Φ collapses to canonical-bit-identical.
        # The deformation-field dicts are keyed by plain ``int`` frame
        # indices (built by ``compute_canonical_mesh_correspondence``);
        # ``frame_idx`` may be a FrameKey for multi-view actionmesh, so
        # extract ``.frame`` before lookup — same idiom used in the
        # rendering-guidance / DDA / viz call sites.
        means_override_i = None
        rotation_override_i = None
        if use_warp:
            fi_int = frame_idx.frame if hasattr(frame_idx, "frame") else int(frame_idx)
            if fi_int in per_frame_mesh_verts:
                means_override_i, rotation_override_i = warp_gaussians_high_res(
                    canonical_gs,
                    canonical_mesh_verts,
                    per_frame_mesh_verts[fi_int],
                    per_frame_mesh_rotations[fi_int],
                    K=int(warp_knn_k),
                    eps=float(warp_knn_eps),
                    chunk_size=int(warp_knn_chunk_size),
                    faces=canonical_mesh_faces,
                )

        # Per-render BG color: optionally sample a fresh uniform RGB so the
        # model can't bake a fixed BG color into the Gaussians.  The same color
        # must be used for both the rasterizer AND the GT-image masking inside
        # ``_compute_frame_loss`` — otherwise foreground gradients pick up a
        # spurious "match-background" signal.
        bg_color_frame = bg_color
        if random_background:
            bg_color_frame = _sample_random_bg_color(
                device,
                seed_base=random_background_seed,
                iter_idx=random_background_iter,
                frame_idx=frame_idx,
            )
        rgb, alpha, depth = _render_frame_with_pose(
            canonical_gs,
            pose["rotation"], pose["translation"], pose["scale"],
            frame_data["K_matrix"],
            frame_data["W"], frame_data["H"],
            device,
            bg_color=bg_color_frame,
            means_override=means_override_i,
            rotation_override=rotation_override_i,
        )
        # Restore DC after rendering (DC offset mode)
        if _saved_dc is not None:
            canonical_gs._features_dc = _saved_dc

        # Mirror DC: restore SH state so the next frame's render starts
        # from the canonical, not the previous frame's mutated, state.
        if _saved_sh is not None:
            prev_rest, prev_degree, prev_active = _saved_sh
            if prev_rest is None:
                # Original Gaussian had no _features_rest — drop the
                # attribute we just set in attach_sh_rest.
                if hasattr(canonical_gs, "_features_rest"):
                    delattr(canonical_gs, "_features_rest")
            else:
                canonical_gs._features_rest = prev_rest
            canonical_gs.sh_degree = prev_degree
            canonical_gs.active_sh_degree = prev_active

        gt_depth_tensor = None
        valid_mask_tensor = None
        need_depth = losses.depth_weight > 0 or losses.normals_weight > 0
        if need_depth and frame_data["gt_depth"] is not None:
            gt_depth_tensor = torch.from_numpy(frame_data["gt_depth"]).float().to(device)
            if frame_data["valid_mask"] is not None:
                valid_mask_tensor = torch.from_numpy(frame_data["valid_mask"]).bool().to(device)

        K_tensor = None
        if losses.normals_weight > 0:
            K_tensor = torch.from_numpy(frame_data["K_matrix"]).float().to(device)
            if valid_mask_tensor is None and frame_data["valid_mask"] is not None:
                valid_mask_tensor = torch.from_numpy(frame_data["valid_mask"]).bool().to(device)

        losses_dict = _compute_frame_loss(
            rgb, alpha, gt_image, mask_tensor, losses,
            rendered_depth=depth, gt_depth=gt_depth_tensor,
            valid_mask=valid_mask_tensor,
            bg_color=bg_color_frame,
            perceptual_scale=perceptual_scale,
            K_matrix=K_tensor,
            return_pixelwise=return_pixelwise,
            has_background=False,
        )

        # Collect per-pixel data for debug visualization
        if return_pixelwise:
            from genia.core.utils.refinement import depth_to_normals as _d2n

            frame_px = {
                "rendered_rgb": rgb.detach().cpu().numpy(),
                "gt_rgb": gt_image.cpu().numpy(),
                "mask": mask_tensor.cpu().numpy(),
            }
            _K = torch.from_numpy(frame_data["K_matrix"]).float().to(device)

            # Rendered depth + depth-derived normals
            if depth is not None:
                depth_sq = depth.squeeze(0) if depth.dim() == 3 else depth
                frame_px["rendered_depth"] = depth_sq.detach().cpu().numpy()
                normals = _d2n(depth_sq, _K)  # (H, W, 3)
                frame_px["rendered_normals"] = normals.detach().cpu().numpy()

            # GT depth + depth-derived GT normals
            if frame_data["gt_depth"] is not None:
                gt_d = frame_data["gt_depth"]  # numpy (H, W)
                frame_px["gt_depth"] = gt_d
                gt_d_t = torch.from_numpy(gt_d).float().to(device)
                frame_px["gt_normals"] = _d2n(gt_d_t, _K).cpu().numpy()

            for k, v in losses_dict.items():
                if k.startswith("px_"):
                    frame_px[k] = v
            pixelwise_data[frame_idx] = frame_px

        # The same aggregator rendering guidance uses; `terms_out` supplies the
        # per-term breakdown for `metrics`.  It also isolates non-finite terms,
        # which matters here: the backward below has no finiteness gate, so a
        # NaN term would reach Adam's moments and never wash out.
        _g_terms: dict = {}
        frame_loss = _aggregate_frame_loss(
            losses_dict, losses, terms_out=_g_terms,
        )

        if use_grad_accum:
            _chunk_loss = _chunk_loss + frame_loss
            _frames_in_chunk += 1
        else:
            total_loss = total_loss + frame_loss
        n_frames += 1

        # Per-term metrics; their sum is what the grad-accum path below
        # reconstructs as the reported total.
        for _k in ("rgb", "ssim", "silhouette", "depth", "normals",
                   "perceptual"):
            metrics[_k] += _g_terms[_k].item()

        # Backward at micro-batch boundary (frees chunk's rendering graph)
        if use_grad_accum and _frames_in_chunk >= microbatch_size:
            more_frames = (vi < len(iter_frames) - 1)
            (_chunk_loss / max(n_total_valid, 1)).backward(
                retain_graph=more_frames,
            )
            _chunk_loss = None
            _frames_in_chunk = 0

    # Backward remaining partial micro-batch (last chunk, release graph)
    if use_grad_accum and _chunk_loss is not None and _frames_in_chunk > 0:
        (_chunk_loss / max(n_total_valid, 1)).backward(retain_graph=False)
        _chunk_loss = None

    if n_frames > 0:
        if not use_grad_accum:
            total_loss = total_loss / n_frames
        for k in metrics:
            metrics[k] /= n_frames

    if use_grad_accum:
        # Backward already done per chunk — return detached scalar
        total_val = sum(metrics[k] for k in metrics if k != "total")
        total_loss = torch.tensor(total_val, device=device)

    metrics["total"] = total_loss.item()
    return total_loss, metrics, pixelwise_data

def _flag_visible_by_depth(
    xyz: torch.Tensor,
    alpha_map: torch.Tensor,
    depth_map: torch.Tensor,
    K_matrix: Any,
    W: int,
    H: int,
    margin: float,
    alpha_eps: float = 0.01,
) -> torch.Tensor:
    """Single-view per-Gaussian visibility by depth occlusion.

    Projects each Gaussian centre with *K_matrix* (identity-c2w camera space) and
    compares its camera z to the rendered expected-depth at that pixel.  A
    Gaussian is occluded (invisible) iff a surface is present there
    (``alpha > alpha_eps``) AND it sits behind it by more than *margin* (relative:
    ``z > z_surface * (1 + margin)``).  Out-of-frame / behind-camera Gaussians and
    pixels with no surface default to not-occluded-this-view; cross-view
    OR-accumulation in the caller handles "seen somewhere".

    Returns (n_gauss,) bool on ``xyz.device``.
    """
    K = torch.as_tensor(K_matrix, dtype=torch.float32, device=xyz.device).reshape(3, 3)
    depth_map = depth_map.reshape(H, W)
    alpha_map = alpha_map.reshape(H, W)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    eps = 1e-6
    zc = z.clamp_min(eps)
    u = (fx * x / zc + cx).round().long()
    v = (fy * y / zc + cy).round().long()
    in_view = (z > eps) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    uc = u.clamp(0, W - 1)
    vc = v.clamp(0, H - 1)
    z_surf = depth_map[vc, uc]
    has_surface = alpha_map[vc, uc] > alpha_eps
    occluded = has_surface & (z > z_surf * (1.0 + margin))
    return in_view & ~occluded


def _compute_gaussian_visibility(
    canonical_gs: Any,
    poses: Dict[int, Dict[str, torch.Tensor]],
    frames: List[int],
    sequence: Any,
    obj_idx: int,
    device: torch.device,
    ft_config: FinetuningConfig,
    warp: Optional[Dict[str, Any]] = None,
    per_frame_out: Optional[list] = None,
) -> Optional[torch.Tensor]:
    """Per-Gaussian train-view visibility, OR-accumulated over *frames*.

    Occlusion test: render the object's expected-depth map and flag a Gaussian
    invisible iff its projected centre lies behind the surface by more than
    ``visibility_depth_margin`` (relative) in every frame (see
    :func:`_flag_visible_by_depth`).

    Self-occlusion only (no inter-object compositing).  When *warp* is given
    (per-frame mesh deformation bundle from :func:`_visibility_warp_bundle`), each
    frame's Gaussians are warped to that frame's deformed shape BEFORE the rigid
    pose, so the OR over frames means "occluded across the whole DEFORMING
    sequence" — a voxel whose back rotates/articulates into view at any frame is
    NOT flagged invisible.  Without *warp* (static objects) the rigid canonical is
    used.  Cross-object occlusion is not modelled.

    Returns
    -------
    (n_gauss,) bool tensor on *device*, or ``None`` when no frame had a valid
    pose/frame_data (caller should then skip whatever consumes this).
    """
    from genia.core.utils.refinement import _prepare_frame_data_for_refinement, _transform_object_to_r3
    from genia.core.utils.rendering import render_gaussian_params
    if warp is not None:
        from genia.core.utils.deformation import warp_gaussians_high_res

    c2w = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)

    visible_gauss: Optional[torch.Tensor] = None  # (n_gauss,) bool, OR-accumulated
    for frame_idx in frames:
        if frame_idx not in poses:
            continue
        frame_data = _prepare_frame_data_for_refinement(sequence, frame_idx, obj_idx)
        if frame_data is None:
            continue

        pose = poses[frame_idx]
        # Per-frame deformation warp (dynamic objects): warp the canonical to this
        # frame's deformed shape BEFORE the rigid pose, so occlusion is tested on
        # the geometry as it actually appears this frame.  None ⇒ rigid canonical.
        means_override = rotation_override = None
        if warp is not None:
            fi_int = frame_idx.frame if hasattr(frame_idx, "frame") else int(frame_idx)
            if fi_int in warp["per_frame_mesh_verts"]:
                means_override, rotation_override = warp_gaussians_high_res(
                    canonical_gs, warp["canonical_mesh_verts"],
                    warp["per_frame_mesh_verts"][fi_int],
                    warp["per_frame_mesh_rotations"][fi_int],
                    K=int(warp["knn_k"]), eps=float(warp["knn_eps"]),
                    chunk_size=int(warp["knn_chunk"]),
                    faces=warp["canonical_mesh_faces"],
                )
        with torch.no_grad():
            xyz, rot, scales, opac, feats = _transform_object_to_r3(
                canonical_gs, pose["rotation"], pose["translation"], pose["scale"], device,
                means_override=means_override, rotation_override=rotation_override,
            )

        # Forward-only occlusion test against the rendered expected-depth — no
        # backward, so it all runs under no_grad (pass-through attrs may still
        # carry the freed decode graph; no_grad keeps it off).
        with torch.no_grad():
            _, alpha, depth = render_gaussian_params(
                xyz, rot, scales, opac, feats,
                c2w, frame_data["K_matrix"], frame_data["W"], frame_data["H"],
            )
            vis = _flag_visible_by_depth(
                xyz, alpha, depth, frame_data["K_matrix"],
                frame_data["W"], frame_data["H"], ft_config.visibility_depth_margin,
            )

        if per_frame_out is not None:
            per_frame_out.append((frame_idx, vis))
        visible_gauss = vis if visible_gauss is None else (visible_gauss | vis)

    return visible_gauss


_GAUSSIAN_RAW_ATTRS = ("_xyz", "_features_dc", "_features_rest",
                       "_scaling", "_rotation", "_opacity")


def _visibility_warp_bundle(
    canonical_mesh_verts: Any,
    per_frame_mesh_verts: Any,
    per_frame_mesh_rotations: Any,
    canonical_mesh_faces: Any,
    knn_k: int,
    knn_eps: float,
    knn_chunk: int,
) -> Optional[Dict[str, Any]]:
    """Bundle the per-frame deformation-warp inputs for the visibility mask, or
    ``None`` when no deformation field is loaded (static objects → rigid canonical).
    With it, the visibility OR-over-frames means 'never visible across the DEFORMING
    sequence' (see :func:`_compute_gaussian_visibility`)."""
    if per_frame_mesh_verts is None or canonical_mesh_verts is None:
        return None
    return {
        "canonical_mesh_verts": canonical_mesh_verts,
        "per_frame_mesh_verts": per_frame_mesh_verts,
        "per_frame_mesh_rotations": per_frame_mesh_rotations,
        "canonical_mesh_faces": canonical_mesh_faces,
        "knn_k": knn_k, "knn_eps": knn_eps, "knn_chunk": knn_chunk,
    }


def _setup_visibility_mask(
    initial_gs: Any,
    poses: Dict[int, Dict[str, torch.Tensor]],
    frames: List[int],
    sequence: Any,
    obj_idx: int,
    n_tokens: int,
    device: torch.device,
    ft_config: FinetuningConfig,
    warp: Optional[Dict[str, Any]] = None,
    coords: Optional[torch.Tensor] = None,
    output_dir: Optional[str] = None,
    scene_name: Optional[str] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """One-time (stable) visibility for the FINETUNE gradient mask.

    Computes per-Gaussian train-view visibility ONCE (depth occlusion test;
    recomputing per-iter would let boundary voxels flicker in and accumulate gradient)
    and returns two stable masks:

    - ``invisible_gauss`` (n_gauss,): Gaussians occluded in EVERY train view.  Their
      decoded attributes are stop-gradient'd before each render
      (:func:`_detach_invisible_gaussians`) — closes the decoder leak and kills
      their own wrong gradient, to neither the tokens nor the shared LoRA decoder.
    - ``frozen_tokens`` (n_tokens,): voxels whose 32 Gaussians are ALL occluded.
      Their ``opt_feats.grad`` row is zeroed each iter so the token itself stays at
      the appearance-init prior.  The detach alone cannot freeze a token — it is
      SHARED by 32 spread Gaussians, so one visible sibling drags the whole voxel
      (incl. its occluded Gaussians), and ``token_drift`` nudges it too.

    A voxel with even one visible Gaussian is NOT frozen (it carries visible
    content); its occluded Gaussians still follow the token — the irreducible
    shared-token cost.  Returns ``(None, None)`` when no train frame had a valid
    pose.  Self-occlusion only (see :func:`_compute_gaussian_visibility`).
    """
    pf = [] if (output_dir is not None and coords is not None) else None
    vis = _compute_gaussian_visibility(
        initial_gs, poses, [f for f in frames if f in poses],
        sequence, obj_idx, device, ft_config, warp=warp,
        per_frame_out=pf,
    )
    if vis is None:
        return None, None
    invisible = ~vis
    frozen_tokens = ~vis.view(n_tokens, -1).any(dim=1)
    print(f"  Visibility mask (obj {obj_idx}): "
          f"{int(invisible.sum())}/{invisible.numel()} Gaussians occluded, "
          f"{int(frozen_tokens.sum())}/{n_tokens} voxels frozen "
          f"({100.0 * frozen_tokens.float().mean().item():.1f}%)")
    if output_dir is not None and coords is not None and pf:
        try:
            from genia.core.visualization import save_perframe_voxel_visibility_plot
            with get_timer().exclude():
                coords_np = (coords[:, 1:] if coords.shape[1] == 4 else coords).detach().cpu().numpy()
                vis_per_frame = np.stack([
                    v.view(n_tokens, -1).any(dim=1).float().cpu().numpy() for _f, v in pf])
                frame_poses = [(f, poses[f]) for f, _v in pf]
                save_perframe_voxel_visibility_plot(
                    coords_np, vis_per_frame, frame_poses, sequence, obj_idx,
                    output_path=os.path.join(output_dir, f"{scene_name}_obj{obj_idx}_visibility.png"),
                    title=f"{scene_name} obj {obj_idx} — FINETUNE visibility (occluded across the whole sequence)")
            print(f"  Visibility mask (obj {obj_idx}): saved per-frame visibility plot")
        except Exception as e:
            print(f"  Visibility mask (obj {obj_idx}): plot skipped ({e})")
    return invisible, frozen_tokens


def compose_pose_with_root_delta(pose, root_delta):
    """``pose_i o dT`` for one frame, returning a new pose dict.

    The delta is ONE Sim(3) shared by every frame, applied in the object's CANONICAL
    frame, so per-frame root MOTION survives -- on a dynamic scene this is the
    canonical-frame correction propagating outward.

    NO TRANSPOSES.  The renderer poses points as ``x @ quaternion_to_matrix(q)``
    (``apply_pose_to_gaussian``), so ``quaternion_to_matrix(q)`` IS the row-vector matrix
    ``compose_root_delta`` documents, and ``matrix_to_quaternion`` takes it straight back.
    Transposing on the way in and out would compose the delta in CAMERA space instead --
    ``Q_new = Q @ dR`` rather than ``dR @ Q`` -- which is a different transform per frame
    the moment two frames disagree in rotation, cannot express a shared canonical-frame
    correction at all when they do, and breaks the shared-world invariant.
    """
    from pytorch3d.transforms import axis_angle_to_matrix

    from genia.core.utils.pose_refit import compose_root_delta
    from genia.core.utils.quaternion_ops import matrix_to_quaternion, quaternion_to_matrix

    R_row = quaternion_to_matrix(pose["rotation"].reshape(1, 4))[0]
    dR_row = axis_angle_to_matrix(root_delta["aa"])
    s_i = pose["scale"].reshape(-1)
    R_new, s_new, t_new = compose_root_delta(
        dR_row, torch.exp(root_delta["log_ds"]).reshape(()), root_delta["dt"],
        R_row, s_i.mean(), pose["translation"].reshape(3))
    out = dict(pose)
    out["rotation"] = matrix_to_quaternion(R_new.contiguous().unsqueeze(0)).squeeze(0)
    out["translation"] = t_new
    out["scale"] = s_new.reshape(1).expand(s_i.shape[0]) if s_i.shape[0] > 1 else s_new.reshape(1)
    return out


def derive_shared_world_poses(poses, ts_ref, groups, rebase, sw_q, sw_t, scale_for_render):
    """Rewrite every frame's pose from its TIMESTAMP's shared (q, t), in place.

    One shared pose per (object, timestamp); each view of that timestamp is derived by the
    fixed c2w rebase ``M_i``, so gradients from every view land on the one shared leaf and
    the object keeps a single world placement per instant.

    Must run INSIDE the optimization loop, once per iteration.  The existing per-frame
    scale alias (``expand(3).unsqueeze(0)``) may be built once before the loop only
    because ``expand``/``unsqueeze`` save no tensors for backward; the quaternion
    normalize and the matmuls here do, so a graph built once would raise "backward through
    the graph a second time" on iteration 2.

    The normalize is load-bearing.  ``quaternion_ops.quaternion_to_matrix`` does NOT
    normalize, while ``apply_pose_to_gaussian`` (the per-frame path) does -- so an
    Adam-drifted ``|q| != 1`` would silently scale the derived rotation matrix by
    ``|q|^2``, i.e. become an extra scale factor fighting ``opt_global_scale``.

    Rebinding ``poses[fk]`` is safe: every in-loop consumer reads it at call time, and the
    dict-splat preserves the scale alias.
    """
    from genia.core.utils.pose_params import derive_perframe_pose_from_shared_decoded
    from genia.core.utils.quaternion_ops import matrix_to_quaternion, quaternion_to_matrix

    for ts in ts_ref:
        q = torch.nn.functional.normalize(sw_q[ts].reshape(1, 4), dim=-1)
        R_ref = quaternion_to_matrix(q)[0]
        t_ref = sw_t[ts].reshape(3)
        for fk in groups[ts]:
            R_i, t_i, _ = derive_perframe_pose_from_shared_decoded(
                R_ref, t_ref, scale_for_render, rebase[fk])
            poses[fk] = {
                **poses[fk],
                "rotation": matrix_to_quaternion(R_i.unsqueeze(0)).reshape(1, 4),
                "translation": t_i.reshape(1, 3),
            }


def snapshot_poses_with_correction(poses, root_delta=None):
    """Detached copy of ``poses``, with the shared correction COMPOSED IN when there is one.

    The correction is applied per frame at loss time (``compute_total_loss``) but onto a
    COPY, so without this it never reaches ``tokens_by_object``: under
    ``correction_granularity: shared`` the native per-frame params are frozen, so the block would
    return its INPUT poses unchanged and its own before/after PSNR would show nothing
    happened.  ``pose_refit.apply_icp_refine`` is the path this mirrors.

    Restores the STORED shapes.  ``compose_pose_with_root_delta`` returns
    ``(4,)/(3,)/(1,|3,)`` while every consumer downstream expects the
    ``(1,4)/(1,3)/(1,3)`` that ``decode_perframe_poses_from_raw`` writes -- bare 1-D
    survives everything until ``make_scene`` hands the translation to PyTorch3D's
    ``Translate``, which reads it as the X coordinate with Y and Z missing.
    """
    out = {}
    for fi, pose in poses.items():
        if root_delta is None:
            out[fi] = {k: v.clone().detach() for k, v in pose.items()}
            continue
        composed = compose_pose_with_root_delta(pose, root_delta)
        snap = {k: v.clone().detach() for k, v in composed.items()}
        snap["rotation"] = snap["rotation"].reshape(1, 4)
        snap["translation"] = snap["translation"].reshape(1, 3)
        snap["scale"] = snap["scale"].reshape(-1).expand(3).reshape(1, 3)
        out[fi] = snap
    return out


def _detach_invisible_gaussians(gs: Any, invisible: torch.Tensor) -> None:
    """Stop-gradient the occluded Gaussians' decoded attributes IN PLACE, before the
    render, so backward sends no gradient through them to the tokens OR the shared
    (LoRA'd) decoder.  *invisible* is the stable per-Gaussian mask from
    :func:`_setup_visibility_mask`.  Visible rows keep full gradient."""
    for name in _GAUSSIAN_RAW_ATTRS:
        attr = getattr(gs, name, None)
        if attr is None or not attr.requires_grad:  # _features_rest may be None
            continue
        m = invisible.view(-1, *([1] * (attr.dim() - 1)))
        setattr(gs, name, torch.where(m, attr.detach(), attr))


def _setup_opacity_anchor(
    initial_gs: Any,
    poses: Dict[int, Dict[str, torch.Tensor]],
    frames: List[int],
    sequence: Any,
    obj_idx: int,
    device: torch.device,
    ft_config: FinetuningConfig,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """One-time setup for the invisible-Gaussian opacity anchor.

    Returns ``(init_opacity, invisible)``: the iter-0 per-Gaussian opacity (always
    captured) and the boolean train-invisible mask (``None`` when no train frame
    had a valid pose, leaving the anchor disabled).  Both are stable for the whole
    run — recomputing per-iter would re-flag already-collapsed Gaussians.  See
    ``FinetuningConfig.invisible_opacity_anchor_weight``.
    """
    init_opacity = initial_gs.get_opacity.detach().reshape(-1).clone()
    vis = _compute_gaussian_visibility(
        initial_gs, poses, [fi for fi in frames if fi in poses],
        sequence, obj_idx, device, ft_config,
    )
    if vis is None:
        print(f"  Invisible-opacity anchor (obj {obj_idx}): no valid train "
              f"frame for visibility — anchor disabled this run")
        return init_opacity, None
    invisible = ~vis
    print(f"  Invisible-opacity anchor (obj {obj_idx}): "
          f"weight={ft_config.invisible_opacity_anchor_weight}, "
          f"{int(invisible.sum())}/{invisible.numel()} Gaussians train-invisible")
    return init_opacity, invisible


def _apply_decoder_anchors(
    decoder: Any,
    opt_feats: torch.Tensor,
    coords: torch.Tensor,
    init_opacity: Optional[torch.Tensor],
    invisible: Optional[torch.Tensor],
    opacity_weight: float,
    init_rotation: Optional[torch.Tensor],
    rotation_weight: float,
    autocast_bf16: bool,
    metrics: Dict[str, float],
) -> None:
    """Opacity + rotation anchors — gradient routed to the decoder's LoRA ONLY.

    Both regularize a DECODED Gaussian attribute back toward its iter-0 value, so
    they adapt the (LoRA'd) decoder's token→attr mapping rather than rewrite the
    SLAT tokens — perturbing the tokens corrupts the shape representation (the
    opacity anchor demonstrably backfires that way, driving back-side opacity
    DOWN).  A single ``opt_feats.detach()`` re-decode serves both; ``backward()``
    runs on this independent graph (detached tokens) and accumulates into the
    LoRA params' ``.grad`` on top of the main loss's, leaving ``opt_feats.grad``
    untouched.

    - opacity (``invisible_opacity_anchor_weight``): one-sided ``relu(init - cur)``
      on the train-invisible Gaussians.
    - rotation (``gauss_rotation_anchor_weight``): L2 ``(cur - init)`` on every
      Gaussian's raw ``_rotation``.

    No-op when both are off, and also when the decoder is FROZEN
    (``lora_decoder``/``optimize_decoder`` off): the tokens are detached, so a
    frozen decoder leaves no grad-requiring parameter in this graph — the decode
    carries no ``grad_fn`` and ``backward()`` would raise.  Anchoring a decoder
    that cannot move is meaningless by construction, so skip it.  Call AFTER the
    main ``total_loss.backward()``, before ``optimizer.step()``.
    """
    op_on = (opacity_weight > 0 and invisible is not None
             and init_opacity is not None and bool(invisible.any()))
    rot_on = rotation_weight > 0 and init_rotation is not None
    if not (op_on or rot_on):
        return
    # Checked before the decode so a frozen decoder costs nothing here.
    if not any(p.requires_grad for p in decoder.parameters()):
        return
    gs = decode_tokens(decoder, opt_feats.detach(), coords, autocast_bf16=autocast_bf16)
    loss = None
    if op_on:
        deficit = torch.relu(init_opacity - gs.get_opacity.reshape(-1))[invisible]
        op_term = opacity_weight * deficit.pow(2).mean()
        metrics["opacity_anchor"] = op_term.item()
        loss = op_term
    if rot_on:
        rot_term = rotation_weight * (gs._rotation - init_rotation).pow(2).mean()
        metrics["rotation_anchor"] = rot_term.item()
        loss = rot_term if loss is None else loss + rot_term
    loss.backward()


# ---------------------------------------------------------------------------
# Core finetuning — single object
# ---------------------------------------------------------------------------

def _finetune_single_object(
    canonical_feats: torch.Tensor,
    canonical_coords: torch.Tensor,
    frame_indices: List[int],
    poses: Dict[int, Dict[str, torch.Tensor]],
    obj_idx: int,
    decoder: torch.nn.Module,
    sequence: Any,
    losses: LossConfig,
    pipeline: PipelineConfig,
    ft_config: FinetuningConfig,
    device: torch.device,
    output_dir: str,
    scene_name: str,
    save_renders: bool = True,
    save_metrics: bool = True,
    shared_world_ref_key: Any = None,
    *,
    canonical_mesh_verts: "torch.Tensor | None" = None,
    per_frame_mesh_verts: "Dict[int, torch.Tensor] | None" = None,
    per_frame_mesh_rotations: "Dict[int, torch.Tensor] | None" = None,
    canonical_mesh_faces: "torch.Tensor | None" = None,
    warp_knn_k: int = 4,
    warp_knn_eps: float = 1.0e-8,
    warp_knn_chunk_size: int = 131072,
) -> Dict[str, Any]:
    """Run token fine-tuning optimization for a single object.

    Returns a dict with keys: ``gaussian``, ``feats``, ``coords``, ``poses``,
    ``loss_history``, ``best_iter``, ``decoder``,
    ``gs_adapter``.
    """
    from genia.core.utils.lora import LoRAAdapter

    # ------------------------------------------------------------------
    # 1. Optionally LoRA-adapt the decoder via a single shared copy +
    #    per-object adapter.
    # ------------------------------------------------------------------
    lora_params: List[torch.nn.Parameter] = []
    obj_decoder = decoder            # default: shared frozen Gaussian decoder
    gs_adapter: Optional[LoRAAdapter] = None
    if ft_config.lora_decoder and ft_config.optimize_decoder:
        obj_decoder = setup_shared_lora_decoder(
            decoder,
            ft_config.lora_rank,
            ft_config.lora_alpha,
            init=_ft_lora_init(ft_config),
            use_dora=_ft_use_dora(ft_config),
            rs_scaling=_ft_rs_scaling(ft_config),
            lora_targets=_ft_lora_targets(ft_config),
        )
        gs_adapter = LoRAAdapter(
            obj_decoder,
            ft_config.lora_rank,
            ft_config.lora_alpha,
            init=_ft_lora_init(ft_config),
            use_dora=_ft_use_dora(ft_config),
            rs_scaling=_ft_rs_scaling(ft_config),
        )
        gs_adapter.activate()
        lora_params = gs_adapter.parameters
        lora_trainable = sum(p.numel() for p in lora_params)
        total_params = sum(p.numel() for p in obj_decoder.parameters())
        method = "DoRA" if _ft_use_dora(ft_config) else "LoRA"
        print(f"  {method} gauss-decoder (obj {obj_idx}): rank={ft_config.lora_rank} "
              f"alpha={ft_config.lora_alpha} init={_ft_lora_init(ft_config)} — "
              f"{lora_trainable} params "
              f"({lora_trainable / total_params:.2%} of decoder)")

    # Gradient-checkpoint the decoder torso to free the retained activation
    # graph on very large objects.  Gated by voxel count so small
    # objects (which already fit) skip the ~2x recompute.
    _n_voxels = canonical_coords.shape[0]
    if _ft_decoder_checkpoint(ft_config) and _n_voxels >= _ft_decoder_checkpoint_min_voxels(ft_config):
        _enable_decoder_checkpoint(obj_decoder)
        print(f"  Decoder checkpointing (obj {obj_idx}): ON ({_n_voxels} voxels)")

    # ------------------------------------------------------------------
    # 2. Decode initial tokens (for before/after comparison)
    # ------------------------------------------------------------------
    with torch.no_grad():
        initial_gs = decode_tokens(obj_decoder, canonical_feats.float(), canonical_coords)
    n_points = initial_gs.get_xyz.shape[0]
    print(f"  Initial Gaussian (obj {obj_idx}): {n_points} points")

    # Snapshot of iter-0 raw _rotation for the soft anchor regulariser.
    # Detached + cloned so it can't get rebound when the per-iter decode
    # mutates canonical_gs.
    rotation_anchor_init: Optional[torch.Tensor] = None
    if ft_config.gauss_rotation_anchor_weight > 0:
        rotation_anchor_init = initial_gs._rotation.detach().clone()

    # Invisible-Gaussian opacity anchor: snapshot iter-0 opacity + the set of
    # Gaussians not visible from any train view, both computed ONCE here (the
    # invisible set is a stable geometric property; recomputing per-iter would
    # also re-flag already-collapsed Gaussians).  Reused every iter to penalise
    # opacity drops below init on the back-side.  See FinetuningConfig.
    opacity_anchor_init: Optional[torch.Tensor] = None
    opacity_anchor_invisible: Optional[torch.Tensor] = None
    if ft_config.invisible_opacity_anchor_weight > 0:
        opacity_anchor_init, opacity_anchor_invisible = _setup_opacity_anchor(
            initial_gs, poses, frame_indices, sequence, obj_idx, device, ft_config,
        )

    # Visibility gradient mask (stable, computed once): per-Gaussian occluded set
    # (detached before each render — leak-free) + per-token frozen set
    # (opt_feats.grad zeroed each iter so fully-occluded voxels stay at the prior).
    vis_invisible_gauss: Optional[torch.Tensor] = None
    vis_frozen_tokens: Optional[torch.Tensor] = None
    if ft_config.token_grad_visibility_mask:
        vis_invisible_gauss, vis_frozen_tokens = _setup_visibility_mask(
            initial_gs, poses, frame_indices, sequence, obj_idx,
            canonical_coords.shape[0], device, ft_config,
            warp=_visibility_warp_bundle(
                canonical_mesh_verts, per_frame_mesh_verts, per_frame_mesh_rotations,
                canonical_mesh_faces, warp_knn_k, warp_knn_eps,
                warp_knn_chunk_size),
            coords=canonical_coords, output_dir=(output_dir if save_renders else None),
            scene_name=scene_name,
        )

    # Per-frame appearance parameters:
    #   sh_degree=None → disabled (no per-frame appearance)
    #   sh_degree=0    → DC offsets only (base color varies per frame)
    #   sh_degree>0    → DC offsets + SH rest (base color + view-dependent vary per frame)
    dc_offset_per_frame: Optional[Dict[int, torch.nn.Parameter]] = None
    sh_rest_per_frame: Optional[Dict[int, torch.nn.Parameter]] = None

    if ft_config.sh_degree is not None:
        # Per-frame DC offsets (always when sh_degree is not None)
        dc_offset_per_frame = {}
        for fi in frame_indices:
            if fi in poses:
                dc_offset_per_frame[fi] = torch.nn.Parameter(
                    torch.zeros(n_points, 1, 3, device=device, dtype=torch.float32)
                )
        total_dc_params = sum(p.numel() for p in dc_offset_per_frame.values())
        print(f"  Per-frame DC offsets: {len(dc_offset_per_frame)} frames, "
              f"{total_dc_params} params (lr={ft_config.sh_lr})")

        # Per-frame SH rest (only when sh_degree > 0)
        if ft_config.sh_degree > 0:
            extra_sh = (ft_config.sh_degree + 1) ** 2 - 1
            sh_rest_per_frame = {}
            for fi in frame_indices:
                if fi in poses:
                    sh_rest_per_frame[fi] = torch.nn.Parameter(
                        torch.zeros(n_points, extra_sh, 3, device=device, dtype=torch.float32)
                    )
            total_sh_params = sum(p.numel() for p in sh_rest_per_frame.values())
            print(f"  Per-frame SH degree {ft_config.sh_degree}: "
                  f"{len(sh_rest_per_frame)} frames, {total_sh_params} params "
                  f"(lr={ft_config.sh_lr_rest})")

    # Warm up perceptual models if needed (lazy-loaded via ModelCache)
    if losses.perceptual_weight > 0:
        print("  Loading LPIPS perceptual model...")
        _ = ModelCache.get().perceptual_model

    # ------------------------------------------------------------------
    # 3. Setup optimizable parameters
    # ------------------------------------------------------------------
    _token_lr = float(ft_config.token_lr)
    _optimize_tokens = bool(ft_config.optimize_tokens) and _token_lr > 0
    opt_feats = canonical_feats.clone().detach().float().requires_grad_(_optimize_tokens)

    original_feats = None
    if ft_config.token_drift_weight > 0:
        original_feats = canonical_feats.clone().detach().float()

    # Resolved HERE, before the shared-world gate below reads it, and reused by the
    # correction setup further down -- it is a pure config read with no dependencies.
    # With mv_shared_world_pose ON, the gate below already frees one pose per
    # (object, TIMESTAMP) and derives each view by a c2w rebase -- exactly what the
    # per-frame axis means.  With it OFF the natives are per-FrameKey, i.e. per VIEW, so
    # the resolver refuses per_frame/both there.
    _correction_granularity = resolve_correction_granularity(
        ft_config, "finetuning", is_mv=sequence.is_mv,
        mv_shared_world_pose=pipeline.mv_shared_world_pose)

    # ---- MV shared world pose ------------------------------------------------------
    from genia.core.utils.frame_key import as_frame_key, frame_key_sort_key, group_by_frame
    from genia.core.state import timestamp_reference_keys
    from genia.core.utils.pose_params import (
        build_decoded_shared_world_pose_params, build_correction,
        freeze_params, natives_to_freeze, precompute_c2w_rebases,
    )

    # ONE pose per (object, TIMESTAMP); every view of that timestamp is DERIVED from it
    # by a fixed c2w rebase, so the object has a single world placement per instant
    # instead of V independent camera-space ones.  Gated on the pipeline flag alone --
    # no block knob -- exactly as `refine_poses_for_sequence` dispatches, so
    # `pipeline=mv_ours` gets it and monocular runs (where the flag is forced
    # off) are untouched.  FINETUNE is the LAST block to touch pose, so without this it
    # undoes the invariant every upstream block maintained.
    _shared_world = bool(getattr(pipeline, "mv_shared_world_pose", False)) and \
        bool(ft_config.refine_poses)
    _sw_ts_ref: Dict[Any, Any] = {}      # timestamp -> reference FrameKey
    _sw_groups: Dict[Any, list] = {}     # timestamp -> that timestamp's keys, view-sorted
    _sw_rebase: Dict[Any, torch.Tensor] = {}   # frame key -> constant 4x4 M_i
    _sw_q: Dict[Any, torch.Tensor] = {}        # timestamp -> shared quaternion leaf
    _sw_t: Dict[Any, torch.Tensor] = {}        # timestamp -> shared translation leaf
    if _shared_world:
        if shared_world_ref_key is None or shared_world_ref_key not in poses:
            raise ValueError(
                "finetuning: mv_shared_world_pose is on but no usable reference frame "
                f"was supplied for object {obj_idx} (got {shared_world_ref_key!r}). "
                "The caller must resolve it with `resolve_reference_frame`.")
        for fk in poses:
            if sequence[fk].c2w is None:
                raise ValueError(
                    f"finetuning: mv_shared_world_pose needs every frame's c2w to derive "
                    f"the per-view poses; frame {fk} has none.")
        _sw_ts_ref = timestamp_reference_keys(list(poses.keys()), shared_world_ref_key)
        _sw_groups = group_by_frame({as_frame_key(fk): None for fk in poses})
        for _ts, _ref_fk in _sw_ts_ref.items():
            _sw_rebase.update(
                precompute_c2w_rebases(sequence, _sw_groups[_ts], _ref_fk, device))
        print(f"  MV shared world pose: {len(_sw_ts_ref)} timestamp(s) x "
              f"{len(poses)} frame(s), reference {shared_world_ref_key} "
              f"(per-timestamp refs {sorted(_sw_ts_ref.values(), key=frame_key_sort_key)})")

    # Save original poses for before/after comparison
    original_poses = {}
    rot_params, trans_params = [], []

    # Create a single global scale shared across all frames (shape: (1, 3) or (3,))
    # Use the first available frame's scale as the initial value.
    opt_global_scale: Optional[torch.Tensor] = None
    initial_global_scale: Optional[torch.Tensor] = None

    for fi in frame_indices:
        if fi not in poses:
            continue
        original_poses[fi] = {
            "rotation": poses[fi]["rotation"].clone().detach(),
            "translation": poses[fi]["translation"].clone().detach(),
            "scale": poses[fi]["scale"].clone().detach(),
        }
        if ft_config.refine_poses:
            _rot_lr = float(losses.lr_rotation)
            _trans_lr = float(losses.lr_translation)
            if not _shared_world:
                poses[fi]["rotation"] = poses[fi]["rotation"].clone().detach().float().requires_grad_(_rot_lr > 0)
                poses[fi]["translation"] = poses[fi]["translation"].clone().detach().float().requires_grad_(_trans_lr > 0)
                rot_params.append(poses[fi]["rotation"])
                trans_params.append(poses[fi]["translation"])
            # Initialize global scale from first frame.  Isotropic mode
            # holds a single scalar Parameter (Adam tracks one moment);
            # we expand to (3,) only when handing the value to the
            # renderer below.
            if opt_global_scale is None:
                scale_val = poses[fi]["scale"].clone().detach().float().view(-1)
                if scale_val.shape[0] == 1:
                    scale_val = scale_val.expand(3).clone()
                _scale_lr = float(losses.lr_scale)
                opt_global_scale = (
                    scale_val.mean().reshape(1)
                    .clone().to(device).requires_grad_(_scale_lr > 0)
                )
                initial_global_scale = opt_global_scale.clone().detach()

    # Shared-world leaves: one (q, t) per TIMESTAMP, cloned off that timestamp's
    # reference view.  `build_decoded_shared_world_pose_params` is the decoded-mode
    # builder (FINETUNE carries no raw tokens and no SSI, i.e. the
    # `optimize_pose_tokens=False` path).  Its scale leaf is deliberately DISCARDED:
    # FINETUNE keeps its own single `opt_global_scale` per object, which is stricter than
    # the refine block's per-timestamp scale and is what the rebase's `s_i = s_ref`
    # already assumes.  The param groups are rebuilt here rather than reused because
    # FINETUNE's carry `tag`/`name`, which drive the pose freeze and grad logging.
    if _shared_world:
        for _ts, _ref_fk in _sw_ts_ref.items():
            _q, _t, _s_unused, _ = build_decoded_shared_world_pose_params(
                poses[_ref_fk], _rot_lr, _trans_lr, 0.0, device)
            _sw_q[_ts], _sw_t[_ts] = _q, _t
        rot_params = list(_sw_q.values())
        trans_params = list(_sw_t.values())

    # Point all frames' scale to the shared global scale tensor.
    # Isotropic case expands the scalar to (3,) for the renderer; the
    # expanded view shares storage with the underlying Parameter so
    # gradients flow back to the (1,) Adam-tracked leaf.
    scale_for_render: Optional[torch.Tensor] = None
    if ft_config.refine_poses and opt_global_scale is not None:
        scale_for_render = opt_global_scale.expand(3)
        for fi in frame_indices:
            if fi in poses:
                # Store as (1, 3) to match expected shape in compute_total_loss
                poses[fi]["scale"] = scale_for_render.unsqueeze(0)

    # Build optimizer param groups — skip groups with lr=0 (their params
    # already have requires_grad=False, so no gradients are wasted).
    param_groups = []
    if _optimize_tokens:
        param_groups.append({"params": [opt_feats], "lr": _token_lr, "tag": "appearance", "name": "tokens"})
    if ft_config.refine_poses:
        if _rot_lr > 0 and rot_params:
            param_groups.append({"params": rot_params, "lr": _rot_lr, "tag": "pose", "name": "rotation"})
        if _trans_lr > 0 and trans_params:
            param_groups.append({"params": trans_params, "lr": _trans_lr, "tag": "pose", "name": "translation"})
        if _scale_lr > 0 and opt_global_scale is not None:
            param_groups.append({"params": [opt_global_scale], "lr": _scale_lr, "tag": "pose", "name": "scale"})
    if lora_params:
        _lora_lr = float(ft_config.lora_lr)
        if _lora_lr > 0:
            lora_group = {"params": lora_params, "lr": _lora_lr, "tag": "appearance", "name": "lora"}
            if ft_config.lora_weight_decay > 0:
                lora_group["weight_decay"] = float(ft_config.lora_weight_decay)
            param_groups.append(lora_group)
    if dc_offset_per_frame is not None:
        _dc_lr = float(ft_config.sh_lr)
        if _dc_lr > 0:
            param_groups.append({
                "params": list(dc_offset_per_frame.values()),
                "lr": _dc_lr, "tag": "appearance", "name": "dc_offset",
            })
    if sh_rest_per_frame is not None:
        _sh_lr = float(ft_config.sh_lr_rest)
        if _sh_lr > 0:
            param_groups.append({
                "params": list(sh_rest_per_frame.values()),
                "lr": _sh_lr, "tag": "appearance", "name": "sh_rest",
            })

    # ---- Shared correction: ONE Sim(3) per object, composed onto every frame ----
    # In the object's CANONICAL frame, so per-frame root MOTION survives.
    _root_delta = None
    if _correction_granularity != "per_frame":
        if not ft_config.refine_poses:
            raise ValueError(
                "finetuning: correction_granularity requires refine_poses=true — the "
                "correction IS a pose parameter, and pose optimisation is off.")
        _rd_aa, _rd_logs, _rd_dt, _rd_groups = build_correction(
            float(losses.lr_rotation), float(losses.lr_scale),
            float(losses.lr_translation), device,
            # `tag` drives the pose freeze; `name` is per component, which is why
            # this is a callable rather than one dict of extras.
            group_extra=lambda k: {"tag": "pose", "name": f"correction_granularity_{k}"})
        _root_delta = {"aa": _rd_aa, "log_ds": _rd_logs, "dt": _rd_dt}
        param_groups.extend(_rd_groups)
        _freeze = natives_to_freeze(
            _correction_granularity,
            resolve_correction_scale_control(ft_config, "finetuning"))
        if _freeze == "all":
            # Freeze the native per-frame pose params so the correction is the only
            # thing that moves.  `correction_granularity: both` skips this freeze and leaves
            # them live alongside the correction; the split between the two is then a
            # gauge choice rather than a result, which is harmless only because
            # `snapshot_poses_with_correction` composes them before anything reads them.
            _delta_ids = {id(v) for v in _root_delta.values()}
            _keep = [any(id(p) in _delta_ids for p in g["params"])
                     or g.get("tag") != "pose"
                     for g in param_groups]
            for g, keep in zip(param_groups, _keep):
                if not keep:
                    for prm in g["params"]:
                        prm.requires_grad_(False)
            param_groups = [g for g, keep in zip(param_groups, _keep) if keep]
        elif _freeze == "scale" and opt_global_scale is not None:
            # The object has ONE size, so the shared correction owns it and the native
            # scale is held.  Dropped from the groups rather than given lr=0, because
            # `losses.lr_scale` also drives the correction's `log_ds` above.
            param_groups = freeze_params(param_groups, [opt_global_scale])
        print(f"  Pose correction: {_correction_granularity} (one Sim(3) for obj {obj_idx})"
              + ("; native scale FROZEN" if _freeze == "scale" else ""))

    if not param_groups:
        # All LRs are zero ⇒ nothing to optimise.  Return the same dict
        # contract as the optimised path: caller (``finetune_canonical_tokens``)
        # consumes ``result["gaussian"]`` / ``result["feats"]`` / ... — the
        # optimised values are simply the inputs unchanged.
        print("  WARNING: all LRs are 0.0 — nothing to optimize, skipping finetuning")
        with torch.no_grad():
            unchanged_gs = decode_tokens(
                obj_decoder, canonical_feats.float().to(device), canonical_coords,
            )
        return {
            "gaussian": unchanged_gs,
            "initial_gaussian": initial_gs,
            "feats": canonical_feats,
            "coords": canonical_coords,
            "poses": poses,
            "original_poses": original_poses,
            "dc_per_frame": dc_offset_per_frame,
            "sh_per_frame": sh_rest_per_frame,
            "loss_history": [],
            "best_iter": 0,
            "decoder": obj_decoder,
            "gs_adapter": gs_adapter,
        }

    optimizer = torch.optim.AdamW(param_groups)

    # Batch size
    num_frames = len(frame_indices)
    batch_size = ft_config.batch_size if ft_config.batch_size > 0 else num_frames
    batch_size = min(batch_size, num_frames)
    effective_num_iterations = ft_config.num_iterations

    # Linear LR warmup (multiplier min(step / W, 1)) on every param group.
    scheduler = None
    warmup_steps = max(0, int(ft_config.lr_warmup_steps))
    if warmup_steps > 0:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step, _W=warmup_steps: min(step / _W, 1.0),
        )

    parts = []
    if _optimize_tokens:
        parts.append("token")
    if ft_config.lora_decoder and ft_config.optimize_decoder:
        parts.append("lora")
    if ft_config.sh_degree is not None:
        parts.append("dc_offset")
        if ft_config.sh_degree > 0:
            parts.append(f"sh{ft_config.sh_degree}")
    if ft_config.refine_poses:
        parts.append("pose")
    mode_str = " + ".join(parts) if parts else "no-op"

    if ft_config.refine_poses and opt_global_scale is not None:
        print(f"  Global scale (isotropic): {opt_global_scale.data.tolist()}")

    print(f"\n  Starting {mode_str} fine-tuning (obj {obj_idx})")
    print(f"    Iterations: {effective_num_iterations}  Batch: {batch_size}/{num_frames} frames")
    print(f"    Token LR: {ft_config.token_lr}  warmup={warmup_steps}")
    if lora_params:
        print(f"    LoRA LR: {ft_config.lora_lr}  rank={ft_config.lora_rank}"
              f"  weight_decay={ft_config.lora_weight_decay}")
    if dc_offset_per_frame is not None:
        print(f"    DC offset LR: {ft_config.sh_lr}")
    if sh_rest_per_frame is not None:
        print(f"    SH rest LR: {ft_config.sh_lr_rest}  degree={ft_config.sh_degree}")
    if ft_config.refine_poses:
        print(f"    Pose LR: rot={losses.lr_rotation}  trans={losses.lr_translation}"
              f"  scale={losses.lr_scale}")

    # ------------------------------------------------------------------
    # 4. Optimization loop
    # ------------------------------------------------------------------
    loss_history: List[Dict[str, float]] = []
    # No best-tracking: the final iter's state is what we save.  The
    # ``best_*`` variables below carry the iter-0 snapshot for the zero-iter
    # edge case; they are overwritten in-place with the final state after
    # the loop exits.  ``final_loss`` carries the last reported total loss.
    final_loss = float("nan")
    best_feats = opt_feats.clone().detach()
    # Derive once before the loop as well, so the ZERO-iteration path (and the iter-0
    # snapshot it falls back to) writes derived poses rather than the per-view input.  A
    # no-op in practice on input the upstream rebase already made consistent, and a
    # correction when it did not.
    if _shared_world:
        with torch.no_grad():
            derive_shared_world_poses(poses, _sw_ts_ref, _sw_groups, _sw_rebase,
                                      _sw_q, _sw_t, scale_for_render)
    best_poses = snapshot_poses_with_correction(poses, _root_delta)
    best_dc_per_frame: Optional[Dict[int, torch.Tensor]] = None
    if dc_offset_per_frame is not None:
        best_dc_per_frame = {fi: p.clone().detach() for fi, p in dc_offset_per_frame.items()}
    best_sh_per_frame: Optional[Dict[int, torch.Tensor]] = None
    if sh_rest_per_frame is not None:
        best_sh_per_frame = {fi: p.clone().detach() for fi, p in sh_rest_per_frame.items()}
    best_iter = 0
    frame_sampler = EpochFrameSampler(frame_indices)

    pbar = tqdm(
        range(effective_num_iterations),
        desc=f"  Finetune obj {obj_idx}",
        leave=True,
        disable=effective_num_iterations == 0,
    )
    _autocast_bf16 = _ft_decoder_autocast_bf16(ft_config)
    for iteration in pbar:
        optimizer.zero_grad()

        # Every frame's pose re-derived from its timestamp's shared (q, t).  Inside the
        # loop by necessity -- see `derive_shared_world_poses`.  All frames, not just the
        # batch, and the cost is a few small matmuls.
        if _shared_world:
            derive_shared_world_poses(poses, _sw_ts_ref, _sw_groups, _sw_rebase,
                                      _sw_q, _sw_t, scale_for_render)

        canonical_gs = decode_tokens(obj_decoder, opt_feats, canonical_coords,
                                     autocast_bf16=_autocast_bf16)

        # Sample frame batch (epoch-based: all frames seen before any repeats)
        if batch_size >= len(frame_indices):
            batch_frames = frame_indices
        else:
            batch_frames = frame_sampler.sample(batch_size)

        # Stop-gradient the occluded Gaussians before the render (stable mask) —
        # closes the decoder leak + kills their own wrong gradient.
        if vis_invisible_gauss is not None:
            _detach_invisible_gaussians(canonical_gs, vis_invisible_gauss)

        # save_renders is the block's resolved flag: no debug_pixelwise/ PNGs (nor
        # the CPU copies feeding them) when the run asked for no renders.
        want_pixelwise = save_renders and iteration % 50 == 0
        total_loss, metrics, px_data = compute_total_loss(
            canonical_gs, poses, batch_frames, sequence, obj_idx, losses, device,
            perceptual_scale=ft_config.perceptual_scale,
            sh_rest_per_frame=sh_rest_per_frame,
            dc_offset_per_frame=dc_offset_per_frame,
            bg_color=torch.ones(3, device=device) if pipeline.white_background else None,
            return_pixelwise=want_pixelwise,
            microbatch_size=ft_config.microbatch_size,
            canonical_mesh_verts=canonical_mesh_verts,
            per_frame_mesh_verts=per_frame_mesh_verts,
            per_frame_mesh_rotations=per_frame_mesh_rotations,
            canonical_mesh_faces=canonical_mesh_faces,
            warp_knn_k=warp_knn_k,
            warp_knn_eps=warp_knn_eps,
            warp_knn_chunk_size=warp_knn_chunk_size,
            random_background=ft_config.random_background,
            random_background_seed=_ft_random_bg_seed(ft_config),
            random_background_iter=iteration,
            root_delta=_root_delta,
        )

        # Token drift regularization
        drift_loss_val = 0.0
        if ft_config.token_drift_weight > 0 and original_feats is not None:
            drift_loss = ft_config.token_drift_weight * torch.nn.functional.mse_loss(opt_feats, original_feats)
            total_loss = total_loss + drift_loss
            drift_loss_val = drift_loss.item()

        # DC offset regularization: L2 toward zero
        dc_reg_val = 0.0
        if dc_offset_per_frame is not None and ft_config.sh_reg_weight > 0:
            batch_dc = [dc_offset_per_frame[fi] for fi in batch_frames if fi in dc_offset_per_frame]
            if batch_dc:
                dc_reg = ft_config.sh_reg_weight * torch.stack(batch_dc).pow(2).mean()
                total_loss = total_loss + dc_reg
                dc_reg_val = dc_reg.item()

        # DC offset cross-frame consistency: penalize deviation from mean
        dc_consistency_val = 0.0
        if dc_offset_per_frame is not None and ft_config.sh_consistency_weight > 0:
            batch_dc = [dc_offset_per_frame[fi] for fi in batch_frames if fi in dc_offset_per_frame]
            if len(batch_dc) > 1:
                dc_stack = torch.stack(batch_dc)
                dc_mean = dc_stack.mean(dim=0)
                dc_cons = ft_config.sh_consistency_weight * (dc_stack - dc_mean.unsqueeze(0)).pow(2).mean()
                total_loss = total_loss + dc_cons
                dc_consistency_val = dc_cons.item()

        # SH regularization: L2 toward zero (per-frame, protects base color)
        sh_reg_val = 0.0
        if sh_rest_per_frame is not None and ft_config.sh_reg_weight > 0:
            batch_sh = [sh_rest_per_frame[fi] for fi in batch_frames if fi in sh_rest_per_frame]
            if batch_sh:
                sh_reg = ft_config.sh_reg_weight * torch.stack(batch_sh).pow(2).mean()
                total_loss = total_loss + sh_reg
                sh_reg_val = sh_reg.item()

        # SH cross-frame consistency: penalize deviation from mean
        sh_consistency_val = 0.0
        if sh_rest_per_frame is not None and ft_config.sh_consistency_weight > 0:
            batch_sh = [sh_rest_per_frame[fi] for fi in batch_frames if fi in sh_rest_per_frame]
            if len(batch_sh) > 1:
                sh_stack = torch.stack(batch_sh)
                sh_mean = sh_stack.mean(dim=0)
                consistency_loss = ft_config.sh_consistency_weight * (sh_stack - sh_mean.unsqueeze(0)).pow(2).mean()
                total_loss = total_loss + consistency_loss
                sh_consistency_val = consistency_loss.item()

        metrics["drift"] = drift_loss_val
        metrics["dc_reg"] = dc_reg_val
        metrics["dc_consistency"] = dc_consistency_val
        metrics["sh_reg"] = sh_reg_val
        metrics["sh_consistency"] = sh_consistency_val
        if vis_frozen_tokens is not None:
            metrics["vis_masked_frac"] = vis_frozen_tokens.float().mean().item()
        metrics["total"] = total_loss.item()
        final_loss = metrics["total"]

        # Backward: when grad_accum was used, rendering gradients are already
        # accumulated and total_loss is detached.  Adding reg losses re-enables
        # requires_grad only through the reg terms, so backward() propagates
        # only the regularization gradients.  If no reg losses were added,
        # total_loss stays detached and we skip backward entirely.
        if total_loss.requires_grad:
            total_loss.backward()

        # Freeze fully-occluded voxels: zero their token grad — keeps the back-side
        # at the prior, which the per-Gaussian detach alone cannot (shared token).
        if vis_frozen_tokens is not None and opt_feats.grad is not None:
            opt_feats.grad[vis_frozen_tokens] = 0.0

        # Opacity + rotation anchors — gradient to the decoder's LoRA only
        # (tokens detached), accumulated on top of the main loss's LoRA grad.
        _apply_decoder_anchors(
            obj_decoder, opt_feats, canonical_coords,
            opacity_anchor_init, opacity_anchor_invisible,
            ft_config.invisible_opacity_anchor_weight,
            rotation_anchor_init, ft_config.gauss_rotation_anchor_weight,
            _autocast_bf16, metrics,
        )

        # Per-group gradient norms
        grad_norms = {}
        for g in optimizer.param_groups:
            gname = g.get("name", "?")
            gnorm = 0.0
            for p in g["params"]:
                if p.grad is not None:
                    gnorm += p.grad.data.norm(2).item() ** 2
            grad_norms[gname] = gnorm ** 0.5

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Save per-pixel debug plot every 50 iterations
        if want_pixelwise and px_data:
            from genia.core.utils.visualization import save_pixelwise_loss_plot
            with get_timer().exclude():
                save_pixelwise_loss_plot(px_data, iteration, obj_idx, output_dir)

        # Track parameter drift (after step — measures actual drift from initial)
        with torch.no_grad():
            token_l2 = (opt_feats - canonical_feats.float().to(device)).norm().item()
            token_rel = token_l2 / max(canonical_feats.float().norm().item(), 1e-8)
            metrics["token_l2"] = token_l2
            metrics["token_rel"] = token_rel
            if ft_config.refine_poses:
                rot_deltas, trans_deltas = [], []
                for fi in frame_indices:
                    if fi not in poses or fi not in original_poses:
                        continue
                    rot_deltas.append((poses[fi]["rotation"] - original_poses[fi]["rotation"].to(device)).norm().item())
                    trans_deltas.append((poses[fi]["translation"] - original_poses[fi]["translation"].to(device)).norm().item())
                metrics["pose_rot_l2"] = sum(rot_deltas) / max(len(rot_deltas), 1)
                metrics["pose_trans_l2"] = sum(trans_deltas) / max(len(trans_deltas), 1)
                # Global scale delta (single shared scale vs first frame's original)
                if opt_global_scale is not None and initial_global_scale is not None:
                    metrics["pose_scale_l2"] = (opt_global_scale - initial_global_scale.to(device)).norm().item()
                else:
                    metrics["pose_scale_l2"] = 0.0

        # Store per-group gradient norms
        for gname, gnorm in grad_norms.items():
            metrics[f"grad_{gname}"] = gnorm

        loss_history.append(metrics)

        # Update tqdm postfix (match global refinement display)
        postfix = {
            "loss": f"{metrics['total']:.4f}",
        }
        if losses.rgb_weight > 0:
            postfix["rgb"] = f"{metrics['rgb']:.4f}"
        if losses.rgb_ssim_weight > 0:
            postfix["ssim"] = f"{metrics['ssim']:.4f}"
        if losses.silhouette_weight > 0:
            postfix["silh"] = f"{metrics['silhouette']:.4f}"
        if losses.perceptual_weight > 0:
            postfix["lpips"] = f"{metrics['perceptual']:.4f}"
        if losses.depth_weight > 0:
            postfix["depth"] = f"{metrics['depth']:.4f}"
        if drift_loss_val > 0:
            postfix["drift"] = f"{drift_loss_val:.4f}"
        if dc_reg_val > 0:
            postfix["dc_reg"] = f"{dc_reg_val:.4f}"
        if dc_consistency_val > 0:
            postfix["dc_cons"] = f"{dc_consistency_val:.4f}"
        if sh_reg_val > 0:
            postfix["sh_reg"] = f"{sh_reg_val:.4f}"
        if sh_consistency_val > 0:
            postfix["sh_cons"] = f"{sh_consistency_val:.4f}"
        if "vis_masked_frac" in metrics:
            postfix["vmask"] = f"{metrics['vis_masked_frac']:.2f}"
        if metrics.get("opacity_anchor", 0) > 0:
            postfix["op_anch"] = f"{metrics['opacity_anchor']:.4f}"
        for gname, gnorm in grad_norms.items():
            if gnorm > 0:
                postfix[f"∇{gname}"] = f"{gnorm:.1e}"
        pbar.set_postfix(postfix)

    # ------------------------------------------------------------------
    # 5. Snapshot FINAL-iter state + save outputs
    # ------------------------------------------------------------------
    # No best-tracking: take the final iter's state.  ``best_*``
    # variables keep their names downstream (history JSON, save paths,
    # PSNR/SSIM eval) but hold the FINAL-iter values.
    if effective_num_iterations > 0:
        best_feats = opt_feats.clone().detach()
        if ft_config.refine_poses:
            best_poses = snapshot_poses_with_correction(poses, _root_delta)
        if dc_offset_per_frame is not None:
            best_dc_per_frame = {fi: dc_offset_per_frame[fi].clone().detach() for fi in dc_offset_per_frame}
        if sh_rest_per_frame is not None:
            best_sh_per_frame = {fi: p.clone().detach() for fi, p in sh_rest_per_frame.items()}
        best_iter = effective_num_iterations - 1

    print(f"  Final loss: {final_loss:.4f} at iteration {best_iter+1}")

    # Report the fitted correction, not just that one was requested: "correction_granularity:
    # shared" alone cannot distinguish "ran and moved the object 3 degrees" from "ran and
    # did nothing".  Same three numbers `run_icp_refine_block` prints.
    if _root_delta is not None:
        with torch.no_grad():
            _aa = _root_delta["aa"].reshape(3)
            _ang = float(torch.rad2deg(_aa.norm()))
            print(f"  Pose correction fitted: ΔR {_ang:.2f}°, "
                  f"Δs {float(torch.exp(_root_delta['log_ds']).reshape(())):.4f}, "
                  f"|Δt| {float(_root_delta['dt'].reshape(3).norm()):.4f}")

    # Decode final Gaussian (pure decoded tokens, no per-frame appearance baked in)
    with torch.no_grad():
        best_gs = decode_tokens(obj_decoder, best_feats.float().to(device), canonical_coords)

    # Before/after PSNR + SSIM headline (mean over all frames, masked to obj).
    _eval_bg = torch.ones(3, device=device) if pipeline.white_background else None
    _psnr_before, _ssim_before = _compute_mean_psnr_ssim(
        initial_gs, original_poses, frame_indices, sequence, obj_idx, device,
        bg_color=_eval_bg,
        canonical_mesh_verts=canonical_mesh_verts,
        per_frame_mesh_verts=per_frame_mesh_verts,
        per_frame_mesh_rotations=per_frame_mesh_rotations,
        canonical_mesh_faces=canonical_mesh_faces,
        warp_knn_k=warp_knn_k, warp_knn_eps=warp_knn_eps,
        warp_knn_chunk_size=warp_knn_chunk_size,
    )
    _psnr_after, _ssim_after = _compute_mean_psnr_ssim(
        best_gs, best_poses, frame_indices, sequence, obj_idx, device,
        bg_color=_eval_bg,
        canonical_mesh_verts=canonical_mesh_verts,
        per_frame_mesh_verts=per_frame_mesh_verts,
        per_frame_mesh_rotations=per_frame_mesh_rotations,
        canonical_mesh_faces=canonical_mesh_faces,
        warp_knn_k=warp_knn_k, warp_knn_eps=warp_knn_eps,
        warp_knn_chunk_size=warp_knn_chunk_size,
    )
    if _psnr_before is not None and _psnr_after is not None:
        print(
            f"  FINETUNE obj {obj_idx}: PSNR {_psnr_before:.2f} → {_psnr_after:.2f} "
            f"(Δ {_psnr_after - _psnr_before:+.2f})  "
            f"SSIM {_ssim_before:.4f} → {_ssim_after:.4f} "
            f"(Δ {_ssim_after - _ssim_before:+.4f})"
        )

    from genia.core.visualization import (
        plot_finetune_losses,
        plot_gradient_norms,
        plot_parameter_drift,
        plot_sh_magnitudes,
        render_color_shift_debug,
        visualize_slat_voxels_before_after,
    )

    os.makedirs(output_dir, exist_ok=True)

    # The finetuned token, LoRA adapter and loss curve: a record of this pass that
    # nothing reads back (final/ holds the result), so gated like the block's metrics.
    if save_metrics:
        # Save finetuned token
        finetuned_path = os.path.join(output_dir, f"{scene_name}_obj{obj_idx}_finetuned_token.npz")
        save_dict = dict(
            slat_feats=best_feats.cpu().numpy(),
            slat_coords=canonical_coords.cpu().numpy(),
        )
        if best_dc_per_frame:
            save_dict["dc_frame_indices"] = np.array(sorted(best_dc_per_frame.keys()))
            for fi, dc_tensor in best_dc_per_frame.items():
                save_dict[f"dc_offset_frame_{fi}"] = dc_tensor.cpu().numpy()
        if best_sh_per_frame:
            save_dict["sh_degree"] = np.array(ft_config.sh_degree)
            save_dict["sh_frame_indices"] = np.array(sorted(best_sh_per_frame.keys()))
            for fi, sh_tensor in best_sh_per_frame.items():
                save_dict[f"sh_rest_frame_{fi}"] = sh_tensor.cpu().numpy()
        np.savez(finetuned_path, **save_dict)
        print(f"  Saved fine-tuned token: {finetuned_path}")

        # Save LoRA weights (read from the adapter directly — independent of
        # which adapter is currently activated on the shared decoder)
        if gs_adapter is not None:
            lora_path = os.path.join(output_dir, f"{scene_name}_obj{obj_idx}_lora_weights.pt")
            torch.save({
                "lora_rank": ft_config.lora_rank,
                "lora_alpha": ft_config.lora_alpha,
                "use_dora": _ft_use_dora(ft_config),
                "rs_scaling": _ft_rs_scaling(ft_config),
                "lora_targets": _ft_lora_targets(ft_config),
                "state_dict": gs_adapter.state_dict(),
            }, lora_path)
            print(f"  Saved LoRA weights: {lora_path}")

        # Save loss history
        history_path = os.path.join(output_dir, f"{scene_name}_obj{obj_idx}_loss_history.json")
        history_data = {
            "num_iterations": ft_config.num_iterations,
            "lr_tokens": ft_config.token_lr,
            "refine_poses": ft_config.refine_poses,
            "batch_size": batch_size,
            "best_iteration": best_iter + 1,
            "best_loss": final_loss,
            "history": loss_history,
        }
        if ft_config.lora_decoder and ft_config.optimize_decoder:
            history_data.update({
                "lora_decoder": True,
                "lora_rank": ft_config.lora_rank,
                "lora_alpha": ft_config.lora_alpha,
                "lora_lr": ft_config.lora_lr,
            })
        if ft_config.sh_degree is not None:
            history_data.update({"sh_degree": ft_config.sh_degree, "sh_lr": ft_config.sh_lr,
                                 "sh_reg_weight": ft_config.sh_reg_weight,
                                 "sh_consistency_weight": ft_config.sh_consistency_weight})
            if ft_config.sh_degree > 0:
                history_data["sh_lr_rest"] = ft_config.sh_lr_rest
        with open(history_path, "w") as f:
            json.dump(history_data, f, indent=2)
        print(f"  Saved loss history: {history_path}")

    # Plot loss curves and parameter drift.  Diagnostics, so gated on the block's
    # resolved save_renders — 4-6 PNGs PER OBJECT otherwise, in a run that asked for
    # none.
    with get_timer().exclude():
        if save_renders:
            plot_path = os.path.join(output_dir, f"{scene_name}_obj{obj_idx}_loss_plot.png")
            plot_finetune_losses(loss_history, best_iter, plot_path, config=losses)

            drift_plot_path = os.path.join(output_dir, f"{scene_name}_obj{obj_idx}_param_drift.png")
            plot_parameter_drift(loss_history, best_iter, drift_plot_path, has_poses=ft_config.refine_poses)

            grad_plot_path = os.path.join(output_dir, f"{scene_name}_obj{obj_idx}_grad_norms.png")
            plot_gradient_norms(loss_history, best_iter, grad_plot_path)

            # SH magnitude plot (per-frame appearance parameters at each degree)
            if best_dc_per_frame or best_sh_per_frame:
                sh_mag_path = os.path.join(output_dir, f"{scene_name}_obj{obj_idx}_sh_magnitudes.png")
                plot_sh_magnitudes(
                    dc_per_frame=best_dc_per_frame,
                    sh_per_frame=best_sh_per_frame,
                    sh_degree=ft_config.sh_degree if ft_config.sh_degree is not None else 0,
                    output_path=sh_mag_path,
                )

                # Color shift debug: amplified diff + offset-only renders
                shift_debug_path = os.path.join(output_dir, f"{scene_name}_obj{obj_idx}_color_shift_debug.png")
                render_color_shift_debug(
                    canonical_gs=best_gs,
                    dc_per_frame=best_dc_per_frame or {},
                    sh_per_frame=best_sh_per_frame,
                    poses=best_poses,
                    sequence=sequence,
                    obj_idx=obj_idx,
                    output_path=shift_debug_path,
                    white_background=pipeline.white_background,
                )

            # SLAT voxel before/after visualization (shared PCA)
            voxel_path = os.path.join(output_dir, f"{scene_name}_obj{obj_idx}_slat_before_after.png")
            visualize_slat_voxels_before_after(
                feats_before=canonical_feats,
                feats_after=best_feats,
                coords=canonical_coords,
                output_path=voxel_path,
                title=f"SLAT Tokens — Object {obj_idx} (before vs after fine-tuning)",
            )

    # Feature change summary
    delta = (best_feats.cpu() - canonical_feats.float().cpu()).norm().item()
    rel_delta = delta / max(canonical_feats.float().cpu().norm().item(), 1e-8)
    print(f"  Feature L2 change: {delta:.4f} (relative: {rel_delta:.4%})")

    return {
        "gaussian": best_gs,
        "initial_gaussian": initial_gs,
        "feats": best_feats,
        "coords": canonical_coords,
        "poses": best_poses,
        "original_poses": original_poses,
        "dc_per_frame": best_dc_per_frame,
        "sh_per_frame": best_sh_per_frame,
        "loss_history": loss_history,
        "best_iter": best_iter,
        "decoder": obj_decoder,
        # The adapter carries the trained (A, B) Parameters; the caller reads
        # its state dict into ``PipelineState.lora_state_dicts``.  ``None``
        # when LoRA is off.
        "gs_adapter": gs_adapter,
    }


# ---------------------------------------------------------------------------
# Per-frame entry point — finetune each frame's own SLAT
# ---------------------------------------------------------------------------

def finetune_perframe_tokens(
    tokens_by_object: Dict[int, List[Tuple[int, Dict]]],
    decoder: torch.nn.Module,
    sequence: Any,
    losses: "LossConfig",
    pipeline: "PipelineConfig",
    ft_config: "FinetuningConfig",
    device: torch.device,
    output_dir: str,
    scene_name: str,
    save_renders: bool = True,
    save_metrics: bool = True,
) -> Tuple[Dict[int, List[Tuple[int, Dict]]], Dict[int, Dict]]:
    """Fine-tune each frame's OWN SLAT against that frame's GT.

    The per-frame counterpart of :func:`finetune_canonical_tokens`, for runs
    that never built a canonical object — a dynamic sequence reconstructed from
    SAM3D's per-frame shapes (``PipelineState.has_canonical`` is False, so the
    canonical entry point has nothing to optimize).

    Each ``(object, frame)`` is a **one-frame instance of the same optimizer**:
    ``_finetune_single_object`` receives that frame's SLAT plus a single-element
    ``frame_indices``, so it supervises exactly that frame's image / mask /
    depth and refines that frame's pose.  Nothing about the loss, LoRA, or
    visibility machinery is duplicated here.

    No deformation warp is threaded through — a per-frame SLAT already **is**
    the deformed geometry at its own timestamp, so the ``canonical_mesh_*``
    kwargs the canonical path uses have no meaning.

    ``num_iterations`` is spent **per frame**, so wall-clock scales with the
    frame count.

    Returns
    -------
    tuple
        ``(tokens_by_object, perframe_gaussians)`` — each entry's finetuned
        SLAT rides along in its ``decoder_input_slat``, and the Gaussians
        (``{obj_idx: {frame_idx: Gaussian}}``) are decoded with the per-frame
        LoRA decoder when one was used.  The LoRA weights themselves are
        **not** returned — only ``set_canonical_slat`` consumes
        ``PipelineState.lora_state_dicts``, and there is no canonical here, so
        the decoded Gaussians are what carries the adaptation.
    """
    from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp

    from genia.core.utils.frame_key import frame_key_stem

    if (
        ft_config.sh_degree is not None and ft_config.sh_degree > 0
        and not (ft_config.lora_decoder and ft_config.optimize_decoder)
    ):
        print("WARNING: sh_degree > 0 requires lora_decoder + optimize_decoder. Setting sh_degree=0 (DC only).")
        ft_config.sh_degree = 0

    new_perframe_gaussians: Dict[int, Dict] = {}
    new_tokens_by_object: Dict[int, List[Tuple[int, Dict]]] = dict(tokens_by_object)

    for obj_idx in sorted(tokens_by_object.keys()):
        updated_tokens: List[Tuple[int, Dict]] = []
        entries = tokens_by_object[obj_idx]
        n_slats = sum(1 for _, di in entries if di.get("decoder_input_slat") is not None)
        print(f"\n{'='*60}")
        print(f"Fine-tuning object {obj_idx} per-frame ({n_slats} frame(s))")
        print(f"{'='*60}")

        for frame_idx, di in entries:
            slat = di.get("decoder_input_slat")
            if slat is None:
                # No per-frame SLAT for this frame (never decoded, or purged) —
                # nothing to optimize; carry the entry through untouched.
                updated_tokens.append((frame_idx, di))
                continue

            poses = {
                frame_idx: {
                    "rotation": di["rotation"].clone().to(device),
                    "translation": di["translation"].clone().to(device),
                    "scale": di["scale"].clone().to(device),
                }
            }

            # Per-frame artifact subdir: every file _finetune_single_object
            # writes is named {scene}_obj{obj}_* , so sibling frames would
            # otherwise overwrite each other.
            frame_output_dir = os.path.join(
                output_dir, "perframe", frame_key_stem(frame_idx),
            )

            print(f"\n  --- object {obj_idx}, frame {frame_idx} ---")
            result = _finetune_single_object(
                canonical_feats=slat.feats.clone().to(device),
                canonical_coords=slat.coords.clone().to(device),
                frame_indices=[frame_idx],
                poses=poses,
                obj_idx=obj_idx,
                decoder=decoder,
                sequence=sequence,
                losses=losses,
                pipeline=pipeline,
                ft_config=ft_config,
                device=device,
                output_dir=frame_output_dir,
                scene_name=scene_name,
                save_renders=save_renders,
                save_metrics=save_metrics,
            )

            new_di = dict(di)
            new_di["decoder_input_slat"] = sp.SparseTensor(
                feats=result["feats"].to(device), coords=result["coords"].to(device),
            )
            if frame_idx in result["poses"]:
                bp = result["poses"][frame_idx]
                new_di["rotation"] = bp["rotation"]
                new_di["translation"] = bp["translation"]
                new_di["scale"] = bp["scale"]
            if result["dc_per_frame"] and frame_idx in result["dc_per_frame"]:
                new_di["dc_offset"] = result["dc_per_frame"][frame_idx]
            if result["sh_per_frame"] and frame_idx in result["sh_per_frame"]:
                new_di["sh_rest"] = result["sh_per_frame"][frame_idx]
            new_di["refinement_loss_history"] = result["loss_history"]
            new_di["refinement_batch_loss_history"] = result["loss_history"]
            new_di["refinement_best_iteration"] = result["best_iter"]
            updated_tokens.append((frame_idx, new_di))
            new_perframe_gaussians.setdefault(obj_idx, {})[frame_idx] = result["gaussian"]

        new_tokens_by_object[obj_idx] = updated_tokens

    return new_tokens_by_object, new_perframe_gaussians


# ---------------------------------------------------------------------------
# Main entry point — finetune all objects
# ---------------------------------------------------------------------------

def finetune_canonical_tokens(
    canonical_slats: Dict[int, Any],
    tokens_by_object: Dict[int, List[Tuple[int, Dict]]],
    decoder: torch.nn.Module,
    sequence: Any,
    losses: "LossConfig",
    pipeline: "PipelineConfig",
    ft_config: "FinetuningConfig",
    device: torch.device,
    output_dir: str,
    scene_name: str,
    save_renders: bool = True,
    save_metrics: bool = True,
    canon_frame_per_object: "Dict[int, Any] | None" = None,
    *,
    # ── actionmesh: per-obj per-canonical-mesh-vertex deformation field ──
    # When supplied, the per-frame render path warps decoded Gaussians via
    # ``warp_gaussians_high_res`` BEFORE applying Stage-1 rigid pose, so
    # FINETUNE supervises against deformed-frame GT images.  ``None``
    # falls back to the rigid-Sim(3)-only path.  Identical contract
    # to ``stage2_mv``'s ``canonical_mesh_verts`` kwargs.
    canonical_mesh_verts_per_obj: "Dict[int, torch.Tensor] | None" = None,
    per_frame_mesh_verts_per_obj: "Dict[int, Dict[int, torch.Tensor]] | None" = None,
    per_frame_mesh_rotations_per_obj: "Dict[int, Dict[int, torch.Tensor]] | None" = None,
    canonical_mesh_faces_per_obj: "Dict[int, torch.Tensor] | None" = None,
    warp_knn_k: int = 4,
    warp_knn_eps: float = 1.0e-8,
    warp_knn_chunk_size: int = 131072,
) -> Tuple[Dict[int, Any], Dict[int, List[Tuple[int, Dict]]], Dict[int, Any],
           Dict[int, dict]]:
    """Fine-tune canonical tokens for all objects.

    Parameters
    ----------
    canonical_slats : dict
        ``{obj_idx: SparseTensor}`` — canonical SLAT tokens per object.
    tokens_by_object : dict
        ``{obj_idx: [(frame_idx, decoder_input_dict), ...]}`` with poses.
    decoder : nn.Module
        Frozen Gaussian decoder (LoRA-wrapped once and shared across objects,
        each with its own adapter, if LoRA is used).
    sequence : Sequence
        Loaded scene data (GT images, masks, depth).
    losses : LossConfig
        Phase-specific loss weights and learning rates.
    pipeline : PipelineConfig
        Cross-phase pipeline control flags.
    ft_config : FinetuningConfig
        Token fine-tuning specific parameters.
    device : torch.device
    output_dir : str
        Directory for output files.
    scene_name : str
    save_renders : bool

    Returns
    -------
    tuple
        ``(canonical_slats, tokens_by_object, canonical_gaussians, lora_info)``
        — updated.  ``canonical_gaussians`` are decoded with the LoRA
        decoder (when used).  ``lora_info`` is ``{obj_idx: {"lora_rank",
        "lora_alpha", "use_dora", "rs_scaling", "lora_targets",
        "state_dict"}}`` for cache persistence.
    """
    from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp

    if (
        ft_config.sh_degree is not None and ft_config.sh_degree > 0
        and not (ft_config.lora_decoder and ft_config.optimize_decoder)
    ):
        print("WARNING: sh_degree > 0 requires lora_decoder + optimize_decoder. Setting sh_degree=0 (DC only).")
        ft_config.sh_degree = 0

    new_canonical_gaussians: Dict[int, Any] = {}
    new_canonical_slats: Dict[int, Any] = {}
    new_tokens_by_object: Dict[int, List[Tuple[int, Dict]]] = dict(tokens_by_object)
    lora_info: Dict[int, dict] = {}

    for obj_idx in sorted(canonical_slats.keys()):
        slat = canonical_slats[obj_idx]
        feats = slat.feats.clone().to(device)
        coords = slat.coords.clone().to(device)

        # Extract frame indices and poses from tokens_by_object
        frame_indices = []
        poses: Dict[int, Dict[str, torch.Tensor]] = {}
        for frame_idx, di in tokens_by_object[obj_idx]:
            frame_indices.append(frame_idx)
            poses[frame_idx] = {
                "rotation": di["rotation"].clone().to(device),
                "translation": di["translation"].clone().to(device),
                "scale": di["scale"].clone().to(device),
            }

        print(f"\n{'='*60}")
        print(f"Fine-tuning object {obj_idx} ({len(frame_indices)} frames)")
        print(f"{'='*60}")

        # Per-obj deformation lookup (None ⇒ rigid-only for this obj).
        _cm_obj = (
            canonical_mesh_verts_per_obj.get(obj_idx)
            if canonical_mesh_verts_per_obj is not None else None
        )
        _pfv_obj = (
            per_frame_mesh_verts_per_obj.get(obj_idx)
            if per_frame_mesh_verts_per_obj is not None else None
        )
        _pfR_obj = (
            per_frame_mesh_rotations_per_obj.get(obj_idx)
            if per_frame_mesh_rotations_per_obj is not None else None
        )
        _faces_obj = (
            canonical_mesh_faces_per_obj.get(obj_idx)
            if canonical_mesh_faces_per_obj is not None else None
        )
        # The object's shared-world reference, resolved the ONE way the rebase resolves
        # it (`canon_frame_per_object`, else the lowest FrameKey).  Two rules for one
        # reference silently produce a wrong rebase.
        #
        # Gated on the flag rather than resolved unconditionally: `resolve_reference_frame`
        # sorts with `frame_key_sort_key`, which needs real FrameKeys, and a per-frame
        # caller may legitimately carry bare ints.  MV data always has
        # FrameKeys, so the resolution is only reachable where it is well defined.
        _sw_ref = None
        if bool(getattr(pipeline, "mv_shared_world_pose", False)):
            from genia.core.utils.pipeline_state import resolve_reference_frame
            _sw_ref = resolve_reference_frame(
                tokens_by_object[obj_idx], canon_frame_per_object or {}, obj_idx)

        result = _finetune_single_object(
            canonical_feats=feats,
            canonical_coords=coords,
            frame_indices=frame_indices,
            poses=poses,
            obj_idx=obj_idx,
            shared_world_ref_key=_sw_ref,
            decoder=decoder,
            sequence=sequence,
            losses=losses,
            pipeline=pipeline,
            ft_config=ft_config,
            device=device,
            output_dir=output_dir,
            scene_name=scene_name,
            save_renders=save_renders,
            save_metrics=save_metrics,
            canonical_mesh_verts=_cm_obj,
            per_frame_mesh_verts=_pfv_obj,
            per_frame_mesh_rotations=_pfR_obj,
            canonical_mesh_faces=_faces_obj,
            warp_knn_k=warp_knn_k,
            warp_knn_eps=warp_knn_eps,
            warp_knn_chunk_size=warp_knn_chunk_size,
        )

        new_canonical_gaussians[obj_idx] = result["gaussian"]
        new_canonical_slats[obj_idx] = sp.SparseTensor(
            feats=result["feats"].to(device), coords=result["coords"].to(device),
        )

        # LoRA state dict of the per-object Gaussian-decoder adapter.
        gs_adapter_i = result.get("gs_adapter")
        if gs_adapter_i is not None:
            lora_info[obj_idx] = {
                "lora_rank": ft_config.lora_rank,
                "lora_alpha": ft_config.lora_alpha,
                "use_dora": _ft_use_dora(ft_config),
                "rs_scaling": _ft_rs_scaling(ft_config),
                "lora_targets": _ft_lora_targets(ft_config),
                "state_dict": gs_adapter_i.state_dict(),
            }

        updated_tokens = []
        for frame_idx, di in tokens_by_object[obj_idx]:
            new_di = dict(di)
            if frame_idx in result["poses"]:
                bp = result["poses"][frame_idx]
                new_di["rotation"] = bp["rotation"]
                new_di["translation"] = bp["translation"]
                new_di["scale"] = bp["scale"]
            if result["dc_per_frame"] and frame_idx in result["dc_per_frame"]:
                new_di["dc_offset"] = result["dc_per_frame"][frame_idx]
            if result["sh_per_frame"] and frame_idx in result["sh_per_frame"]:
                new_di["sh_rest"] = result["sh_per_frame"][frame_idx]
            new_di["refinement_loss_history"] = result["loss_history"]
            new_di["refinement_batch_loss_history"] = result["loss_history"]
            new_di["refinement_best_iteration"] = result["best_iter"]
            updated_tokens.append((frame_idx, new_di))
        new_tokens_by_object[obj_idx] = updated_tokens

    return (new_canonical_slats, new_tokens_by_object, new_canonical_gaussians,
            lora_info)
