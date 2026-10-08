# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Differentiable in-ODE rendering guidance for Stage-2 appearance (SLAT).

At every active ODE step the one-step Tweedie estimate of the SLAT
(``x̂₁ = x_t + (1-t)·v``) is decoded to Gaussians, rendered from each
frame's camera under the frozen per-frame pose, scored against GT with
the standard ``LossConfig`` terms, and the weighted gradient is
subtracted from the velocity.  Deformed objects are warped with the
high-res deformation helpers from ``core/utils/deformation.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from genia.core.utils.refinement import (
    _compute_signed_distance_transform,
)
from genia.core.utils.deformation import warp_gaussians_high_res


_LPIPS_WEIGHTS_REPORTED = [False]


def _perceptual_weights_report():
    """Are the SHARED LPIPS weights still finite?

    The perceptual model is a process-wide singleton
    (``ModelCache.get().perceptual_model``), so if a stray write ever
    lands on its parameters it NaNs for every frame and every
    subsequent step -- exactly the 'starts at one t, then always'
    pattern.  This distinguishes that from a degenerate INPUT, which
    would come and go with the data.
    """
    try:
        from genia.core.utils.model_cache import ModelCache
        m = ModelCache.get().perceptual_model
    except Exception as e:            # not built yet / no cache
        return f"unavailable ({type(e).__name__})"
    bad = [n for n, prm in m.named_parameters()
           if not torch.isfinite(prm).all()]
    buf_bad = [n for n, b in m.named_buffers()
               if b.is_floating_point() and not torch.isfinite(b).all()]
    if not bad and not buf_bad:
        return "all finite"
    return (f"NON-FINITE: {len(bad)} param(s) {bad[:4]}, "
            f"{len(buf_bad)} buffer(s) {buf_bad[:4]}")


def _tstats(name, x):
    if x is None:
        return f"{name}=None"
    if not torch.is_tensor(x):
        try:
            x = torch.as_tensor(x)
        except Exception:
            return f"{name}=<unconvertible>"
    if x.numel() == 0:
        return f"{name}: shape={tuple(x.shape)} EMPTY"
    xf = x.detach().float()
    fin = torch.isfinite(xf)
    n_fin = int(fin.sum())
    if n_fin == 0:
        return f"{name}: shape={tuple(x.shape)} ALL non-finite"
    v = xf[fin]
    return (f"{name}: shape={tuple(x.shape)} "
            f"min={v.min().item():.4g} max={v.max().item():.4g} "
            f"mean={v.mean().item():.4g} std={v.std().item():.4g} "
            f"finite={n_fin}/{x.numel()}")


def _warn_non_finite_loss(loss, losses, branch, frame_idx, t, rgb, alpha,
                          gt=None, mask=None, frame_label=None,
                          dropped=None):
    """Report a non-finite per-frame guidance loss, loudly.

    A NaN loss disables guidance for the WHOLE step and does so SILENTLY: the
    gradient is taken under ``chunk_loss.item() >= 1e-10`` and the velocity
    update under ``total_loss_val >= 1e-10``, and both comparisons are False
    for NaN.  So there is no exception, no ``skip_reason`` and no change to the
    velocity -- the step simply loses its guidance and the solve carries on
    looking healthy.  That silence is why this prints rather than passes.

    Also names the first quantity in the chain that went bad, so the report
    distinguishes "the geometry decoded badly" from "the render blew up" from
    "a specific loss term is ill-conditioned on a fine render" -- three very
    different bugs that look identical from the loss value alone.
    """
    # Fire when the AGGREGATE is bad, or when _aggregate_frame_loss dropped a
    # term to keep it good -- the drop is the thing worth seeing, and after it
    # the aggregate is finite, so testing the aggregate alone would go silent.
    if torch.isfinite(loss) and not dropped:
        return
    stage = "loss-only (render finite)"
    if rgb is not None and alpha is not None and not (
            torch.isfinite(rgb).all() and torch.isfinite(alpha).all()):
        stage = "RENDER (rgb/alpha)"
    terms = [k for k, v in losses.items()
             if torch.is_tensor(v) and not torch.isfinite(v).all()]
    fid = frame_idx if frame_label is None else frame_label
    if dropped:
        outcome = (f"terms {sorted(set(dropped))} DROPPED; the remaining terms "
                   f"keep their gradient")
    else:
        outcome = "guidance SKIPPED for this step"
    print(f"    [rg] NON-FINITE {branch} loss, frame {fid}, t={t:.3f}: "
          f"first bad = {stage}; non-finite loss terms = {terms} "
          f"-> {outcome}")
    # FINITE IS NOT SANE.  A render that is finite but blank, constant or far
    # outside [0, 1] blows up inside VGG and NaNs LPIPS with no bad value ever
    # appearing in the render itself -- so report the RANGE, not just isfinite.
    print(f"      {_tstats('rgb', rgb)}")
    print(f"      {_tstats('alpha', alpha)}")
    if gt is not None:
        print(f"      {_tstats('gt', gt)}")
    if mask is not None:
        _m = mask if torch.is_tensor(mask) else torch.as_tensor(mask)
        print(f"      mask: shape={tuple(_m.shape)} "
              f"foreground_px={int((_m > 0).sum())} of {_m.numel()}")
    if any("perceptual" in k for k in terms) and not _LPIPS_WEIGHTS_REPORTED[0]:
        _LPIPS_WEIGHTS_REPORTED[0] = True
        print(f"      LPIPS shared weights -> {_perceptual_weights_report()} "
              f"(reported once; the model is a process-wide singleton)")


@dataclass
class GuidanceSchedule:
    active_from: float
    active_until: float


def _prepare_frame_targets(
    frames: list,
    device: torch.device,
    resolution_scale: int = 1,
) -> List[dict]:
    """Build per-frame GT targets for rendering guidance.

    ``resolution_scale > 1`` divides each frame's H, W by that factor
    (nearest for masks/depth, area for RGB) and scales the intrinsics
    (``K[:2]``) accordingly, so downstream render-loop calls work in the
    reduced resolution end-to-end.  Bounded at ``max(1, H // scale)``.
    """
    rs = max(1, int(resolution_scale))
    out = []
    for f in frames:
        m = f["mask"]
        if isinstance(m, np.ndarray):
            m = torch.from_numpy(m).float()
        m = m.float().to(device)
        d = f.get("depth_map_z")
        if d is not None:
            if isinstance(d, np.ndarray):
                d = torch.from_numpy(d).float()
            d = d.float().to(device)
        img = f.get("image")
        img_t = None
        if img is not None:
            if isinstance(img, np.ndarray):
                img_t = torch.from_numpy(img).float()
            else:
                img_t = img.float()
            if img_t.dtype == torch.uint8 or img_t.max() > 1.5:
                img_t = img_t / 255.0
            img_t = img_t.to(device)

        H0, W0 = int(m.shape[0]), int(m.shape[1])
        H, W = max(1, H0 // rs), max(1, W0 // rs)
        if rs > 1 and (H, W) != (H0, W0):
            # mask + depth: nearest (preserve sharp edges / NaNs).
            m = torch.nn.functional.interpolate(
                m[None, None], size=(H, W), mode="nearest"
            )[0, 0]
            if d is not None:
                d = torch.nn.functional.interpolate(
                    d[None, None], size=(H, W), mode="nearest"
                )[0, 0]
            # rgb: area downsample (anti-alias).  Channel-first for interp,
            # restore the original layout (assumed H,W,3 or H,W,4).
            if img_t is not None:
                ch_last = img_t.dim() == 3 and img_t.shape[-1] in (3, 4)
                if ch_last:
                    img_t = img_t.permute(2, 0, 1)[None]
                    img_t = torch.nn.functional.interpolate(
                        img_t, size=(H, W), mode="area"
                    )[0].permute(1, 2, 0).contiguous()
                else:
                    img_t = torch.nn.functional.interpolate(
                        img_t[None] if img_t.dim() == 3 else img_t[None, None],
                        size=(H, W), mode="area",
                    ).squeeze(0)

        m_bool = m > 0.5

        K = f.get("K_matrix")
        if K is not None and rs > 1:
            K_arr = K.detach().cpu().numpy() if isinstance(K, torch.Tensor) else np.asarray(K)
            K_arr = K_arr.copy()
            K_arr[0, 0] /= rs
            K_arr[1, 1] /= rs
            K_arr[0, 2] /= rs
            K_arr[1, 2] /= rs
            K = torch.as_tensor(K_arr) if isinstance(K, torch.Tensor) else K_arr

        c2w = f.get("c2w")
        if c2w is not None:
            c2w = torch.as_tensor(c2w, device=device, dtype=torch.float32)

        out.append({
            "mask": m_bool,
            "sdt": _compute_signed_distance_transform(m_bool),
            "K": K,
            "depth_z": d,
            "image_rgb": img_t,
            "H": H,
            "W": W,
            # MV shared-world pose (gsplat): per-frame extrinsics, used to derive
            # off-reference views' poses from the reference. None when unused.
            "c2w": c2w,
        })
    return out


# ════════════════════════════════════════════════════════════════════════════
# Stage-2 appearance (SLAT) rendering guidance
# ════════════════════════════════════════════════════════════════════════════
#
# Operates on the canonical SLAT.  At each ODE step, computes the one-step Tweedie estimate of the SLAT, decodes
# to Gaussians (gsplat), renders per-frame
# under each frame's camera with the (frozen) per-frame foreground pose, and
# subtracts the autograd gradient of an RGB + silhouette + depth + LPIPS loss
# from the velocity.  Reuses ``_compute_frame_loss`` / ``_aggregate_frame_loss``
# so loss-weight tuning happens via the standard ``LossConfig``.


def _rgb_err_map(rendered_rgb, gt_rgb, mask):
    """Per-pixel mean-abs RGB error masked to the object (for diagnostic PNGs).

    ``rendered_rgb`` / ``gt_rgb`` are ``(H, W, 3)`` in [0, 1]; the error is
    averaged over channels and zeroed outside the object mask (so the random
    background never shows up).  Returns an ``(H, W)`` numpy array.
    """
    err = (rendered_rgb.detach() - gt_rgb).abs().mean(dim=-1)
    return (err * mask.float()).cpu().numpy()


def _rgb_to_np(rgb, max_hw: int = 256):
    """``(H, W, 3)`` rendered/GT RGB in [0, 1] → numpy thumbnail for diagnostic
    PNGs.  Stride-downsampled so the per-step storage stays bounded regardless
    of render resolution (the viz cells are ~240px anyway)."""
    t = rgb.detach().clamp(0.0, 1.0)
    s = max(1, max(t.shape[0], t.shape[1]) // max_hw)
    if s > 1:
        t = t[::s, ::s]
    return t.cpu().numpy()


def _gt_on_bg_np(gt_rgb, mask, bg_color, max_hw: int = 256):
    """GT composited onto ``bg_color`` via the object mask, exactly as
    ``_compute_frame_loss`` (``gt*m + bg*(1-m)``) — so the GT thumbnail matches
    what the photometric loss actually compares the render against (incl. the
    per-step random background)."""
    m = mask.float().unsqueeze(-1)
    comp = gt_rgb * m + bg_color.view(1, 1, 3) * (1.0 - m)
    return _rgb_to_np(comp, max_hw=max_hw)


def build_appearance_rendering_guidance_transform(
    *,
    pipeline,
    frames: list,
    frame_labels: "Sequence[int] | None" = None,
    pose_decoder,
    canonical_coords: torch.Tensor,
    slat_mean: torch.Tensor,
    slat_std: torch.Tensor,
    losses_cfg,
    velocity_weight: float,
    schedule: GuidanceSchedule,
    bg_color: torch.Tensor,
    device: torch.device,
    microbatch_size: int = 8,
    resolution_scale: int = 1,
    # ``(‖v‖/‖g‖)·g`` grad normalization: rescales the rendering grad to ‖v‖ so ``velocity_weight`` becomes a
    # fraction-of-velocity step, decoupled from the raw grad magnitude.  Off →
    # raw ``v -= velocity_weight·g``.
    normalize_grad: bool = False,
    # bf16 autocast on the gaussian decoder forward (outputs cast back to
    # fp32) — ~halves the per-ODE-step decode-activation memory; mirrors
    # FINETUNE's ``decoder_autocast_bf16``.
    decoder_autocast_bf16: bool = False,
    # Fresh uniform RGB background per (ODE step, frame); mirrors FINETUNE.
    random_background: bool = False,
    random_background_seed: Optional[int] = None,
    # Occlusion stop-gradient for the gaussian branch (FINETUNE parity —
    # mirrors ``token_grad_visibility_mask``'s ``_detach_invisible_gaussians``):
    # at each active ODE step, flag Gaussians whose projected centre is
    # depth-occluded in EVERY frame (``_flag_visible_by_depth``, OR over
    # frames, self-occlusion only) and stop-gradient their decoded
    # attributes, so backward sends no gradient from unseen Gaussians into
    # the SLAT velocity.  Costs one extra no-grad render per frame per step.
    visibility_detach: bool = False,
    visibility_depth_margin: float = 0.02,
    # Per-canonical-mesh-vertex deformation field (high-res Φ + R).  All
    # three are required together for the actionmesh path; ``None``
    # disables the warp (static behaviour, used by canonical when no
    # deformation field is present, e.g. global-mode GSO).
    canonical_mesh_verts: "torch.Tensor | None" = None,
    per_frame_mesh_verts: "list | None" = None,
    per_frame_mesh_rotations: "list | None" = None,
    canonical_mesh_faces: "torch.Tensor | None" = None,
    # KNN/blend knobs forwarded from the global ``deformation_warp`` config.
    warp_knn_k: int = 4,
    warp_knn_eps: float = 1.0e-8,
    warp_knn_chunk_size: int = 8192,
) -> Tuple[Callable, list]:
    """Return ``(transform_fn, loss_history)`` for Stage-2 SLAT guidance.

    ``transform_fn(t_raw, v_fused, x_t_one_view) -> v_fused'`` runs at every
    ODE step.  Decodes the one-step Tweedie SLAT estimate to Gaussians,
    renders every frame, computes ``_compute_frame_loss`` against GT,
    and subtracts the weighted autograd gradient from ``v_fused`` (shape
    ``(1, L, 8)``).

    Parameters
    ----------
    pose_decoder : Callable[[], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        Closure returning per-frame ``(quat[N,4], trans[N,3], scale[N,3])``,
        all ``no_grad``.  Re-invoked every ODE step (poses are constant in
        APPEARANCE_INIT).
    canonical_coords : torch.Tensor
        Sparse coords ``(L, 4)`` for the SLAT SparseTensor wrap.  Must be in
        batch=0 form.
    slat_mean, slat_std : torch.Tensor
        Decode-side normalisation: SLAT ODE runs in normalised space, so we
        un-normalise via ``feats * slat_std + slat_mean`` before decoding,
        mirroring ``stage2_mv``'s post-loop wrap.
    losses_cfg : LossConfig
        Standard per-phase loss weights — only loss-weight fields are read.
    velocity_weight : float
        Strength multiplier applied to the autograd gradient before the
        velocity update: ``v -= velocity_weight * grad``.  With
        ``normalize_grad=True`` the grad is first rescaled to ‖v‖, so the step
        is ``velocity_weight`` *as a fraction of* ‖v‖ — calibrate to
        ~0.01–0.1 instead of the thousands that the raw-grad path needs.
    bg_color : torch.Tensor
        ``(3,)`` background colour on ``device``.  Foreground-only rendering
        composites the prediction onto this colour; ``_compute_frame_loss``
        composites GT onto the same colour for matched RGB / LPIPS.
    """
    from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp
    from genia.core.utils.refinement import (
        _aggregate_frame_loss, _compute_frame_loss, _depth_l1_term,
        _match_depth_grid,
        _transform_object_to_r3,
    )
    from genia.core.utils.slat_decode import _GAUSSIAN_FP_ATTRS, _cast_attrs_float32
    from genia.core.finetuning import (
        _detach_invisible_gaussians,
        _flag_visible_by_depth,
        _sample_random_bg_color,
    )
    from genia.core.utils.rendering import render_gaussian_params

    frame_targets = _prepare_frame_targets(
        frames, device, resolution_scale=resolution_scale
    )
    # Normalize intrinsics to numpy once — the per-frame loops below otherwise
    # re-convert (device→host sync) every frame at every ODE step.
    for _fd in frame_targets:
        if isinstance(_fd["K"], torch.Tensor):
            _fd["K"] = _fd["K"].detach().cpu().numpy()
    canonical_coords = canonical_coords.to(device)
    slat_mean_d = slat_mean.to(device)
    slat_std_d = slat_std.to(device)
    bg_color_d = bg_color.to(device).float()

    # Per-canonical-mesh-vertex deformation field (high-res Φ + R).  All
    # three kwargs must be supplied together for the actionmesh path, or
    # all three None for the static / canonical path (no warp).
    _N_frames = len(frames)
    _has_mesh = canonical_mesh_verts is not None
    _has_pf_verts = per_frame_mesh_verts is not None
    _has_pf_R = per_frame_mesh_rotations is not None
    if not (_has_mesh == _has_pf_verts == _has_pf_R):
        raise ValueError(
            "build_appearance_rendering_guidance_transform: "
            "canonical_mesh_verts, per_frame_mesh_verts and per_frame_mesh_rotations "
            "must be supplied together (got "
            f"canonical_mesh_verts={'set' if _has_mesh else 'None'}, "
            f"per_frame_mesh_verts={'set' if _has_pf_verts else 'None'}, "
            f"per_frame_mesh_rotations={'set' if _has_pf_R else 'None'})."
        )
    _canonical_verts_d: "torch.Tensor | None" = None
    _per_frame_verts_d: "list | None" = None
    _per_frame_rotations_d: "list | None" = None
    if _has_mesh:
        _canonical_verts_d = canonical_mesh_verts.detach().to(device)
        if _canonical_verts_d.dim() != 2 or _canonical_verts_d.shape[1] != 3:
            raise ValueError(
                f"canonical_mesh_verts must be (V, 3); got "
                f"{tuple(_canonical_verts_d.shape)}"
            )
        V = _canonical_verts_d.shape[0]
        if len(per_frame_mesh_verts) != _N_frames:
            raise ValueError(
                f"per_frame_mesh_verts has length {len(per_frame_mesh_verts)}, "
                f"expected N={_N_frames}"
            )
        if len(per_frame_mesh_rotations) != _N_frames:
            raise ValueError(
                f"per_frame_mesh_rotations has length {len(per_frame_mesh_rotations)}, "
                f"expected N={_N_frames}"
            )
        _per_frame_verts_d = [t.detach().to(device) for t in per_frame_mesh_verts]
        _per_frame_rotations_d = [t.detach().to(device) for t in per_frame_mesh_rotations]
        _faces_d = (
            canonical_mesh_faces.detach().to(device).long()
            if canonical_mesh_faces is not None else None
        )
        for i, (pv, pr) in enumerate(zip(_per_frame_verts_d, _per_frame_rotations_d)):
            if pv.shape != (V, 3) or pr.shape != (V, 3, 3):
                raise ValueError(
                    f"per-frame mesh shape mismatch at slot {i}: "
                    f"verts={tuple(pv.shape)}, R={tuple(pr.shape)}, "
                    f"expected ({V}, 3) / ({V}, 3, 3)"
                )

    gs_decoder = pipeline.models["slat_decoder_gs"]

    _eye_c2w = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)

    def _gauss_cam_params_r3(gs_obj, i, quat_i, trans_i, scale_i):
        """Camera-space (R3) render params for frame *i* — the exact inputs the
        gaussian branch renders with, shared by the loss render and the
        no-grad visibility pass.  With the per-vertex deformation field, warp
        decoded Gaussians at native resolution (``warp_gaussians_high_res``:
        KNN+IDW translation + SO(3) blend onto the Gaussian quat) BEFORE the
        Stage-1 pose; either way pose + P3D→R3 go through
        ``_transform_object_to_r3``, whose ``means_override`` /
        ``rotation_override`` exist for exactly this warp injection."""
        means_w = quats_w = None
        if _canonical_verts_d is not None:
            means_w, quats_w = warp_gaussians_high_res(
                gs_obj,
                _canonical_verts_d,
                _per_frame_verts_d[i],
                _per_frame_rotations_d[i],
                K=warp_knn_k,
                eps=warp_knn_eps,
                chunk_size=warp_knn_chunk_size,
                faces=_faces_d,
            )
        return _transform_object_to_r3(
            gs_obj, quat_i, trans_i, scale_i, device,
            means_override=means_w, rotation_override=quats_w,
        )

    print(
        f"    Appearance rendering guidance: "
        f"velocity_w={velocity_weight}, normalize_grad={normalize_grad}, "
        f"visibility_detach={visibility_detach}, "
        f"rgb_w={losses_cfg.rgb_weight}, ssim_w={losses_cfg.rgb_ssim_weight}, "
        f"silh_w={losses_cfg.silhouette_weight}, "
        f"depth_w={losses_cfg.depth_weight}, lpips_w={losses_cfg.perceptual_weight}, "
        f"active=[{schedule.active_from}, {schedule.active_until}]"
    )

    loss_history: list = []
    _vis_logged: list = []  # one-shot print guard for the visibility mask

    def _transform(t_raw, v_fused, x_t_one):
        """Return a velocity ``(1,L,8)`` updated by the guidance gradient."""
        t = float(t_raw)
        # Monotonic per-active-step index for random-background seeding (each
        # active step appends exactly once to loss_history below).
        step_idx = len(loss_history)
        if t < schedule.active_from or t > schedule.active_until:
            return v_fused

        w_v = velocity_weight
        monitor_only = w_v <= 0.0

        rem = 1.0 - t
        x1 = x_t_one.detach() + rem * v_fused.detach()  # (1, L, 8)
        x1 = x1.requires_grad_(not monitor_only)

        grad_ctx = torch.enable_grad() if not monitor_only else torch.no_grad()
        with grad_ctx, torch.amp.autocast("cuda", enabled=False):
            with torch.no_grad():
                quat_all, trans_all, scale_all = pose_decoder()
            quat_all = quat_all.to(device)
            trans_all = trans_all.to(device)
            scale_all = scale_all.to(device)

            # Per-frame depth- and RGB-error maps for diagnostic PNGs.
            # Computed alongside the frame loss when the matching loss term is
            # active (``depth_weight``/``rgb_weight`` > 0) and the GT signal is
            # present; ``None`` otherwise so the plot leaves blank cells.  Cheap
            # (subtract + abs); depth reuses ``_depth_l1_term`` so masking
            # matches the loss exactly, RGB uses ``_rgb_err_map``.
            want_depth_err = (
                float(losses_cfg.depth_weight) > 0.0
                and any(fd["depth_z"] is not None for fd in frame_targets)
            )
            want_rgb_err = (
                float(losses_cfg.rgb_weight) > 0.0
                and any(fd["image_rgb"] is not None for fd in frame_targets)
            )
            n_frames = len(frame_targets)
            valid_K = sum(1 for fd in frame_targets if fd["K"] is not None)
            n_g_total = valid_K
            frame_depth_errors_g: list = [None] * n_frames
            frame_rgb_errors_g: list = [None] * n_frames
            # Rendered + GT RGB thumbnails shown beside the error maps.
            frame_rendered_g: list = [None] * n_frames
            frame_gt_rgb: list = [None] * n_frames

            # Microbatch the per-frame render loop.  ``mb >= n_frames`` (or 0)
            # falls through to single-pass.  Decoder + un-norm run per
            # chunk so the chunk's graph fully releases after
            # ``autograd.grad`` — bounds peak memory at one chunk's
            # decoder+render activations instead of holding the decoder
            # graph alive across chunks via ``retain_graph``.
            mb = max(1, min(int(microbatch_size) if microbatch_size > 0 else n_frames, n_frames))
            chunks = list(range(0, n_frames, mb))

            acc_grad = (
                torch.zeros_like(x1) if not monitor_only else None
            )
            total_loss_val = 0.0
            L_g_total_val = 0.0
            # Per-step occlusion mask (visibility_detach): (n_gauss,) bool,
            # True = occluded in EVERY frame → stop-gradient'd below.  The
            # decode is deterministic in x1, so the chunk-0 mask holds for
            # every chunk of this step.
            invisible_gauss = None
            n_occluded = 0

            for ci, chunk_start in enumerate(chunks):
                chunk_end = min(chunk_start + mb, n_frames)

                # Un-norm must stay inside the loop so its edge to ``x1``
                # is chunk-local — otherwise chunk 1+ backward hits a
                # freed graph edge.
                feats = x1[0] * slat_std_d + slat_mean_d  # (L, 8)
                slat = sp.SparseTensor(coords=canonical_coords, feats=feats)
                # bf16 decoder forward (activations stored in bf16 for the
                # chunk backward), outputs cast back to fp32 below so the
                # gsplat render + loss stay full-precision.
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=decoder_autocast_bf16):
                    gs_list = gs_decoder(slat)
                gs_obj = gs_list[0] if isinstance(gs_list, list) else gs_list
                if gs_obj is not None and decoder_autocast_bf16:
                    _cast_attrs_float32(gs_obj, _GAUSSIAN_FP_ATTRS)
                if gs_obj is None or gs_obj.get_xyz.shape[0] == 0:
                    gs_obj = None

                # SLAT is constant within a step → chunk 0 alone decides
                # whether the decode is empty for the whole step.
                if ci == 0 and gs_obj is None:
                    loss_history.append({
                        "t": t,
                        "total": 0.0,
                        "gaussian": 0.0,
                        "n_g": 0,
                        "skip_reason": "empty_decode",
                    })
                    return v_fused

                # Visibility pass (FINETUNE parity): render each frame once
                # without grad, flag Gaussians whose projected centre sits
                # behind the rendered surface, OR over frames.  Forward-only —
                # loss values are unchanged, only gradients are masked.
                if ci == 0 and visibility_detach and not monitor_only:
                    with torch.no_grad():
                        vis_any = None
                        for vi, vfd in enumerate(frame_targets):
                            K_v = vfd["K"]
                            if K_v is None:
                                continue
                            params_v = _gauss_cam_params_r3(
                                gs_obj, vi,
                                quat_all[vi], trans_all[vi], scale_all[vi],
                            )
                            _, alpha_v, depth_v = render_gaussian_params(
                                *params_v, _eye_c2w, K_v, vfd["W"], vfd["H"],
                            )
                            vis_v = _flag_visible_by_depth(
                                params_v[0], alpha_v, depth_v, K_v,
                                vfd["W"], vfd["H"], visibility_depth_margin,
                            )
                            vis_any = vis_v if vis_any is None else (vis_any | vis_v)
                        if vis_any is not None:
                            invisible_gauss = ~vis_any
                            n_occluded = int(invisible_gauss.sum())
                            if not _vis_logged:
                                _vis_logged.append(True)
                                print(
                                    f"    rg visibility_detach: {n_occluded}/"
                                    f"{invisible_gauss.numel()} Gaussians occluded "
                                    f"in every frame → stop-gradient'd"
                                )

                # Stop-gradient the occluded rows' decoded attributes in place
                # (shared FINETUNE recipe) before this chunk's renders, so
                # backward sends no gradient through them to x1.
                if invisible_gauss is not None:
                    _detach_invisible_gaussians(gs_obj, invisible_gauss)

                L_g_partial = torch.tensor(0.0, device=device, dtype=torch.float32)

                for i in range(chunk_start, chunk_end):
                    fd = frame_targets[i]
                    K_np = fd["K"]
                    if K_np is None:
                        continue
                    H_i, W_i = fd["H"], fd["W"]
                    quat_i = quat_all[i]
                    trans_i = trans_all[i]
                    scale_i = scale_all[i]

                    # Per-(step, frame) background: the render(s) and the GT
                    # compositing in _compute_frame_loss share this colour so
                    # the photometric loss stays background-agnostic.
                    bg_i = bg_color_d
                    if random_background:
                        bg_i = _sample_random_bg_color(
                            device,
                            seed_base=random_background_seed,
                            iter_idx=step_idx,
                            frame_idx=i,
                        )

                    # GT-on-loss-bg thumbnail: branch-independent (depends only
                    # on bg_i/fd), so compute once per frame rather than in each
                    # render branch.
                    if want_rgb_err and fd["image_rgb"] is not None:
                        frame_gt_rgb[i] = _gt_on_bg_np(
                            fd["image_rgb"], fd["mask"], bg_i
                        )

                    # Gaussian branch — posed (and, with the per-vertex
                    # deformation field, warped) camera-space params via
                    # ``_gauss_cam_params_r3``, rendered with identity c2w.
                    if gs_obj is not None:
                        rgb_g, alpha_g, depth_g = render_gaussian_params(
                            *_gauss_cam_params_r3(gs_obj, i, quat_i, trans_i, scale_i),
                            _eye_c2w, K_np, W_i, H_i, bg_color=bg_i,
                        )
                        losses_g = _compute_frame_loss(
                            rgb_g, alpha_g, fd["image_rgb"], fd["mask"],
                            losses=losses_cfg,
                            sdt=fd["sdt"],
                            rendered_depth=depth_g,
                            gt_depth=fd["depth_z"],
                            bg_color=bg_i,
                            has_background=False,
                        )
                        _drop_g = []
                        _Lg_i = _aggregate_frame_loss(
                            losses_g, losses_cfg, report=_drop_g)
                        _warn_non_finite_loss(
                            _Lg_i, losses_g, "gaussian", i, float(t),
                            rgb_g, alpha_g, gt=fd["image_rgb"], mask=fd["mask"],
                            frame_label=(frame_labels[i] if frame_labels
                                         else None),
                            dropped=_drop_g,
                        )
                        L_g_partial = L_g_partial + _Lg_i
                        if want_depth_err and fd["depth_z"] is not None:
                            # depth_g/fd["mask"] may be render-resolution;
                            # fd["depth_z"] never exceeds the backbone's res.
                            _depth_g_d, _, _mask_g_d = _match_depth_grid(
                                depth_g, fd["depth_z"].shape[-2:], mask=fd["mask"])
                            _, derr_g = _depth_l1_term(
                                _depth_g_d, fd["depth_z"], _mask_g_d,
                                mask_only=bool(losses_cfg.depth_mask_only),
                                return_error_map=True,
                            )
                            frame_depth_errors_g[i] = derr_g.detach().cpu().numpy()
                        if want_rgb_err and fd["image_rgb"] is not None:
                            frame_rgb_errors_g[i] = _rgb_err_map(
                                rgb_g, fd["image_rgb"], fd["mask"]
                            )
                            frame_rendered_g[i] = _rgb_to_np(rgb_g)

                # Match single-pass: divide by the total frame count up front,
                # so summing chunk_loss across chunks equals the frame mean.
                if n_g_total == 0:
                    continue
                chunk_loss = L_g_partial / n_g_total
                L_g_total_val += L_g_partial.item()
                total_loss_val += chunk_loss.item()

                if not monitor_only and chunk_loss.item() >= 1e-10:
                    g = torch.autograd.grad(chunk_loss, x1)[0]
                    acc_grad = acc_grad + g

            L_g_avg = L_g_total_val / n_g_total if n_g_total > 0 else 0.0

            grad_norm_val = 0.0
            guidance_norm_val = 0.0
            # ‖v‖ from the backbone, captured before any guidance is subtracted.
            backbone_velocity_norm = float(v_fused.norm().item())
            if not monitor_only and total_loss_val >= 1e-10:
                grad_norm_val = acc_grad.norm().item()
                if normalize_grad:
                    # Rescale grad to ‖v‖ so w_v is a fraction-of-velocity step;
                    # skip when grad ≈ 0.
                    if grad_norm_val >= 1e-10:
                        acc_grad = acc_grad * (backbone_velocity_norm / grad_norm_val)
                        v_fused = v_fused - w_v * acc_grad
                        guidance_norm_val = w_v * backbone_velocity_norm
                else:
                    v_fused = v_fused - w_v * acc_grad
                    guidance_norm_val = w_v * grad_norm_val

        loss_history.append({
            "t": t,
            "total": total_loss_val,
            "gaussian": L_g_avg,
            "n_g": n_g_total,
            "rg_occluded_gauss": n_occluded,
            "grad_norm": grad_norm_val,
            # Decomposition  v_post = v_backbone − w_v·g  (w_v = velocity_weight;
            # g = raw grad, or grad rescaled to ‖v‖ when normalize_grad),
            # recorded as norms for the velocity-decomposition plot.
            "backbone_velocity_norm": backbone_velocity_norm,
            "guidance_velocity_norm": float(guidance_norm_val),
            "velocity_norm": float(v_fused.norm().item()),
            "per_frame_rendered_gaussian": frame_rendered_g,
            "per_frame_gt_rgb": frame_gt_rgb,
            "per_frame_depth_error_gaussian": frame_depth_errors_g,
            "per_frame_rgb_error_gaussian": frame_rgb_errors_g,
        })
        return v_fused

    return _transform, loss_history
