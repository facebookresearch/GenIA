# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Depth-grounded object placement: an object's centre and extent read from masked depth.

A pose-free observation: the (t, s) refit (``pose_refit``) measures an
object through :class:`DepthObservation` / :func:`depth_grounded_center_extent`, and
fuse the per-view estimates of one timestamp with
:func:`fuse_p3d_positions_by_timestamp`.

Also the posed-shape overlay diagnostic (:func:`save_pose_overlay_viz`) that a pose
block writes around itself.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


def _depth_sample_mask(mask, pm_p3d, disc_threshold=0.0, erode_px=0):
    """Which masked pixels the depth STATISTICS may be read from: finite pointmap,
    minus depth discontinuities, minus an eroded rim.  ``(H, W)`` bool, or ``None``
    when nothing survives even before the optional filters.

    The plain ``mask & isfinite`` intersection keeps every pixel the reconstruction
    happened to emit a number for, and a predicted depth is *finite but unreliable*
    exactly where the object's silhouette meets the background: a mask a pixel or
    two too generous samples the background plane, and background is always BEHIND,
    so the contamination is one-sided and biases the depth median far.  Under
    ``scale_mode="metric_mask"`` that single median is the only depth measurement
    entering the placement, and it scales both the position and the size
    (``extent = bbox_px / f * d_med``), so the whole object slides down its
    projection ray.

    ``recon_mask_edges`` already removes the hard jumps on predicted depth
    (map-anything NaNs them, and ``isfinite`` drops them here), but it
    is inert on ``processing=ground_truth`` -- where ``valid_mask`` is only
    ``depth > 0`` -- and it never touches the finite-but-wrong ring just inside a
    dilated mask.  These two filters are that second line, reusing the pipeline's
    existing kernels rather than a second copy of the test.

    Both are OFF by default (``0``).  If the filtered set comes back empty the
    unfiltered one is returned instead: a thin or articulated subject can erode away
    entirely, and a biased estimate still beats none at all (the frame would otherwise
    be dropped).

    NOTE the returned mask governs ``pts`` ONLY.  The ``metric_mask`` silhouette
    bbox is read from the FULL ``mask`` -- eroding it would shrink the measured
    extent by ``2*erode_px`` pixels, which is the very under-size that mode exists
    to fix.
    """
    valid = mask & np.isfinite(pm_p3d).all(axis=-1)
    if not valid.any():
        return None
    if disc_threshold <= 0 and erode_px <= 0:
        return valid

    # The pipeline's own kernels (torch, CPU round-trip on one HxW frame) rather
    # than a numpy re-derivation, so `disc_threshold` means here what
    # `depth_discontinuity_threshold` means in the refinement losses.
    import torch

    from genia.core.utils.refinement import (
        _compute_depth_discontinuity_mask, _erode_bool_mask,
    )

    # .clone() is load-bearing: as_tensor SHARES the numpy buffer, so the in-place
    # `&=` below would rewrite `valid` itself and take the empty-set fallback down
    # with it (an over-eroded frame would return an empty mask, not the unfiltered one).
    strict = torch.as_tensor(np.ascontiguousarray(valid), dtype=torch.bool).clone()
    if disc_threshold > 0:
        # z of a non-finite pixel is NaN, so every comparison against it is False and
        # its neighbours are dropped too -- a 1px dilation of the invalid region,
        # which is precisely the straddling pixels an edge mask leaves behind.
        # The shared kernel is 4-CONNECTED, so a pixel that meets the object only
        # diagonally (a mask corner) reads as locally smooth and survives; a handful
        # of corner samples is not worth forking the pipeline's discontinuity test.
        z = torch.as_tensor(np.ascontiguousarray(pm_p3d[..., 2]), dtype=torch.float32)
        strict &= _compute_depth_discontinuity_mask(z, float(disc_threshold))
    if erode_px > 0:
        strict &= _erode_bool_mask(
            torch.as_tensor(np.ascontiguousarray(mask), dtype=torch.bool),
            int(erode_px))
    strict = strict.numpy()
    return strict if strict.any() else valid


class DepthObservation(NamedTuple):
    """HOW a frame's depth-grounded ``(center, extent)`` is measured.

    Read by the post-ODE (t, s) refit, which measures the object once the rotation
    exists.  The three knobs travel as ONE value, built by :meth:`from_pose_init_config`
    and passed down, so every stage that measures the object reads the same pixels the
    configuration selected rather than inheriting :func:`depth_grounded_center_extent`'s
    defaults.
    """

    scale_mode: str = "metric_mask"
    disc_threshold: float = 0.0
    erode_px: int = 0

    @classmethod
    def from_pose_init_config(cls, sp_cfg) -> "DepthObservation":
        """The pose-init block's three knobs.  ``getattr`` defaults match the dataclass,
        so a config lacking a key reads as that key being off."""
        return cls(
            disc_threshold=float(getattr(
                sp_cfg, "post_rotation_refit_disc_threshold", 0.0)),
            erode_px=int(getattr(sp_cfg, "post_rotation_refit_erode_px", 0)),
        )

    def read(self, mask, pm_p3d, K):
        """This frame's ``(center, extent)``, or None.  ``pm_p3d`` is PyTorch3D, as the
        name says: under ``metric_mask`` only z is read and the flip is a no-op, but every
        other mode takes the center from ``median(pts)`` and would come back mirrored."""
        return depth_grounded_center_extent(
            mask, pm_p3d, K, self.scale_mode, self.disc_threshold, self.erode_px)


def depth_grounded_center_extent(mask, pm_p3d, K, scale_mode="metric_mask",
                                 disc_threshold=0.0, erode_px=0):
    """One frame's depth-grounded object ``(center, extent)`` in PyTorch3D camera
    space, or ``None`` when the mask covers no finite pointmap sample.

    ``disc_threshold`` / ``erode_px`` restrict which masked pixels the depth
    STATISTICS are read from (:func:`_depth_sample_mask`); both default to off.  The
    ``metric_mask`` silhouette bbox is taken from the unfiltered ``mask`` either way.
    """
    valid = _depth_sample_mask(mask, pm_p3d, disc_threshold, erode_px)
    if valid is None:
        return None
    pts = pm_p3d[valid]
    extent = np.percentile(pts, 97.5, axis=0) - np.percentile(pts, 2.5, axis=0)
    center = np.median(pts, axis=0).astype(np.float32)
    scale = np.float32(extent.max())
    if scale_mode == "metric_mask":
        center, scale = _mask_center_scale(mask, np.median(pts[:, 2]), K, center, scale)
    return center, scale


def _mask_center_scale(mask, d_med, K, center_fallback, scale_fallback):
    """Object center + isotropic scale from the 2D mask **silhouette** (its bbox)
    back-projected at the median masked depth ``d_med``, in PyTorch3D camera space.

    Used by ``scale_mode="metric_mask"``.  The mask spans the full subject even
    where the pointmap is sparse or its depth statistics are biased (thin/
    articulated subjects, non-uniform pixel density), so it avoids the 3D-point
    under-estimate that shrinks e.g. a backpacked hiker to its dense torso.  Only
    a single robust scalar (median depth) enters, so per-pixel depth noise does
    not distort the size.  Falls back to the point-based estimates if K is absent.
    """
    if K is None:
        return center_fallback, scale_fallback
    K = np.asarray(K, dtype=np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    rows, cols = np.nonzero(mask)
    if rows.size == 0 or fx <= 0 or fy <= 0:
        return center_fallback, scale_fallback
    rmin, rmax, cmin, cmax = rows.min(), rows.max(), cols.min(), cols.max()
    d = float(d_med)
    # Metric silhouette size at depth d; scale = the larger lateral extent.
    w = (cmax - cmin) / fx * d
    h = (rmax - rmin) / fy * d
    scale = np.float32(max(w, h))
    # bbox-center pixel → P3D (X-left, Y-up: negate the R3 x,y offsets).
    uc, vc = 0.5 * (cmin + cmax), 0.5 * (rmin + rmax)
    center = np.array([-(uc - cx) / fx * d, -(vc - cy) / fy * d, d],
                      dtype=np.float32)
    return center, scale


# Reductions the per-timestamp multi-view consensus can fuse with — numpy
# function names, called with axis=0 (per-axis on the world centers, over the
# scalar list for scale).  The center is a POSITION, so a per-axis min/max would
# fabricate a bbox corner no view observed: only the averaging pair is offered.
_MV_CENTER_REDUCERS = ("median", "mean")


def group_records_by_timestamp(records):
    """``{FrameKey.frame: [records]}`` -- the per-timestamp bucket, singletons INCLUDED.

    One owner for the grouping half of a multi-view consensus, because the two stages that
    need it want different slices of it: position fusion skips singleton timestamps (a lone
    view has nothing to fuse with), while a per-timestamp SCALE reduce must still visit them
    -- a mono sequence is all singletons, and dropping them would silently disable it.

    Each record is a dict carrying a ``"fk"`` (FrameKey, or anything ``as_frame_key`` takes).
    """
    from .frame_key import as_frame_key

    by_frame = {}
    for r in records:
        by_frame.setdefault(as_frame_key(r["fk"]).frame, []).append(r)
    return by_frame


def fuse_p3d_positions_by_timestamp(records, sequence, key: str, reduce: str = "median"):
    """In place, collapse each timestamp's per-view P3D positions to one world placement.

    Groups ``records`` by ``FrameKey.frame``, and for every timestamp holding more than one
    view: lift each view's position to world through ITS OWN ``c2w``, reduce per axis, and
    project the single world position back into every view's camera.  Each record is a dict
    carrying a ``"fk"`` (FrameKey) and the position under ``key``.

    Owns the whole operation -- grouping, the singleton skip, and the reduce whitelist -- so
    every caller shares one notion of consensus (the post-ODE refit,
    ``core/pose_refit.py::apply_post_rotation_refit``, fuses on ``"t"``).

    Each position is written back at the dtype it arrived as.  A singleton timestamp (mono,
    or a lone view) is left untouched, and an identity ``c2w`` makes the round-trip a no-op,
    so mono callers are unaffected.

    Returns the groups it fused, so a caller can reduce its own per-timestamp quantities
    over the same grouping without rebuilding it.
    """
    # diag(-1,-1,1) flip between P3D and R3 positions; self-inverse, so the one
    # numpy-safe helper does both directions (P3D->R3 here, R3->P3D on the way back).
    from genia.core.utils.quaternion_ops import r3_to_p3d_positions as flip_p3d_r3

    if reduce not in _MV_CENTER_REDUCERS:
        raise ValueError(
            f"reduce must be one of "
            f"{_MV_CENTER_REDUCERS} (a position has no meaningful per-axis min/max -- it "
            f"would name an unobserved bbox corner), got {reduce!r}")

    groups = [g for g in group_records_by_timestamp(records).values() if len(g) > 1]

    for group in groups:
        c2ws = [np.asarray(sequence[r["fk"]].c2w, dtype=np.float64) for r in group]
        world = [(c2w @ np.append(flip_p3d_r3(np.asarray(r[key], dtype=np.float64)), 1.0))[:3]
                 for r, c2w in zip(group, c2ws)]
        p_world = getattr(np, reduce)(np.stack(world, axis=0), axis=0)
        for r, c2w in zip(group, c2ws):
            p_r3 = (np.linalg.inv(c2w) @ np.append(p_world, 1.0))[:3]
            # Written back at the dtype it came in as: the fusion runs in float64, but a
            # caller should not silently change precision by opting into consensus.
            r[key] = flip_p3d_r3(p_r3).astype(np.asarray(r[key]).dtype)
    return groups


def _decoded_pose_records(sequence, obj_idx: int, tokens_list):
    """Per-frame DECODED pose records for the overlay panels below.

    Keys ``fk``/``frame``/``mask``/``center``/``scale`` feed the panel/plot code below,
    plus the rotation and the full scale vector.  ``center``/``scale`` carry the decoded
    translation and mean scale so the trailing z/s plot stays comparable between start
    and end.
    """
    from genia.core.utils.quaternion_ops import quaternion_to_matrix

    recs = []
    for fk, di in tokens_list:
        if di.get("rotation") is None or di.get("translation") is None \
                or di.get("scale") is None:
            continue
        frame = sequence[fk]
        m = frame.masks.get(obj_idx)
        m = (np.zeros(frame.image.shape[:2], dtype=bool) if m is None
             else np.asarray(m, dtype=bool))
        t_vec = di["translation"].reshape(3).detach().float().cpu()
        s_vec = di["scale"].reshape(-1).detach().float().cpu()
        if s_vec.numel() == 1:
            s_vec = s_vec.repeat(3)
        recs.append({
            "fk": fk,
            "frame": fk.frame if hasattr(fk, "frame") else int(fk),
            "mask": m,
            "center": t_vec.numpy(),
            "scale": float(s_vec.mean()),
            "R": quaternion_to_matrix(
                di["rotation"].reshape(1, 4).detach().float().cpu())[0],
            "t": t_vec,
            "s_vec": s_vec,
        })
    return recs


def _flat_series_ylim(values, rel_eps: float = 1e-4, pad: float = 0.01):
    """Y-limits for a series that may be CONSTANT, or ``None`` to keep autoscale.

    matplotlib scales an axis to the data range, so a series whose only variation is
    float32 rounding gets an axis spanning ~1e-7 plus an offset label -- drawing
    quantisation noise as a square wave, so a scale held frozen for the whole ODE
    would read as wildly varying.

    Below ``rel_eps`` relative spread the series is called constant and given a
    +-``pad`` window around its mean, so flat reads as flat.  Above it, ``None``:
    real variation deserves the real autoscale.
    """
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return None
    mid = float(v.mean())
    span = float(v.max() - v.min())
    if span > rel_eps * max(abs(mid), 1e-12):
        return None
    half = pad * max(abs(mid), 1.0)
    return mid - half, mid + half


def save_pose_overlay_viz(
    state, sequence, obj_idx: int, device, out_path, block_label: str = "pose_init",
) -> bool:
    """Render the GT canonical shape at this object's CURRENT decoded pose, overlaid on
    the image.

    Reads whatever pose is in ``state.tokens_by_object`` right now (rotation included,
    any post-ODE refit applied), so a block that calls it before and after shows what it
    changed -- e.g. a refine that inflates the object's apparent size shows up as a
    picture rather than as a PSNR drop.  ``block_label`` only names the block in the
    title.

    WHICH SHAPE each panel draws: this frame's entry in
    ``state.canonical_mesh_per_frame_verts`` when the deformation field keys it, and
    ``state.canonical_mesh_verts`` (the canonical timestamp) otherwise -- on a deforming
    object the shape that generated the silhouette is THAT FRAME's.  Both dicts are
    canonical-normalised and share ``faces`` (fixed topology), so this is a bare vertex
    swap; per-frame SIZE stays with the Sim(3) scale.

    Per frame: rasterise the posed shape's silhouette (nvdiffrast) and overlay it
    (green) on the input RGB with the GT object mask outline (red).  A trailing panel
    plots the decoded z and scale over frames.  Returns False (no render) when the GT
    mesh is unavailable.
    """
    from pathlib import Path

    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from genia.core.utils.mesh_rendering import apply_pose_p3d_to_r3, render_depth_and_alpha

    verts = state.canonical_mesh_verts.get(obj_idx)
    faces = state.canonical_mesh_faces.get(obj_idx)
    # This object's per-frame deformed vertices, or {} when the run has no
    # deformation field.  getattr, not attribute access: a caller may pass a lightweight
    # shim carrying only the three fields above.
    pf_verts = (getattr(state, "canonical_mesh_per_frame_verts", None)
                or {}).get(obj_idx) or {}
    tokens_list = state.tokens_by_object.get(obj_idx, [])
    if verts is None or faces is None or not tokens_list:
        return False
    verts = verts.to(device).float()
    faces = faces.to(device).to(torch.int32)
    recs = _decoded_pose_records(sequence, obj_idx, tokens_list)
    if not recs:
        return False
    deformed = any(r["frame"] in pf_verts for r in recs)

    # Center the canonical mesh + measure its max-extent so the decoded metric
    # scale maps to a true metric size.
    vmin, vmax = verts.min(0).values, verts.max(0).values
    canon_center = (vmin + vmax) * 0.5
    verts_local = verts - canon_center
    canon_extent = float((vmax - vmin).max())

    import nvdiffrast.torch as dr
    glctx = dr.RasterizeCudaContext(device=device)

    # Render the GT shape at each frame's decoded pose.
    panels = []
    for r in recs:
        with torch.no_grad():
            R = r["R"].to(device)
            t = r["t"].to(device)
            s = r["s_vec"].to(device) / max(canon_extent, 1e-8)
            # THIS frame's deformed shape when the field keys it; the canonical
            # timestamp otherwise (rigid runs, or a frame the field skips).
            _v = pf_verts.get(r["frame"])
            v_local = (verts_local if _v is None
                       else _v.to(device).float() - canon_center)
            verts_r3 = apply_pose_p3d_to_r3(v_local, R, t, s)
            frame = sequence[r["fk"]]
            H, W = frame.image.shape[:2]
            K = frame.K_matrix
            _, alpha = render_depth_and_alpha(
                verts_r3, faces, glctx,
                float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]), H, W,
            )
        panels.append({"frame": frame, "fid": r["frame"],
                       "alpha": alpha.detach().cpu().numpy(),
                       "m": r["mask"],
                       "z": float(r["center"][2]), "s": float(r["scale"])})
    if not panels:
        return False
    panels.sort(key=lambda p: p["fid"])

    n = len(panels)
    ncol = min(n, 6)
    nrow = (n + ncol - 1) // ncol
    fig = plt.figure(figsize=(3.0 * ncol, 3.0 * nrow + 3.2))
    gs = fig.add_gridspec(nrow + 1, ncol)
    for i, p in enumerate(panels):
        ax = fig.add_subplot(gs[i // ncol, i % ncol])
        overlay = p["frame"].image.astype(np.float32) / 255.0
        fgr = p["alpha"] > 0.5
        overlay[fgr] = overlay[fgr] * 0.45 + np.array([0.1, 1.0, 0.1]) * 0.55
        ax.imshow(np.clip(overlay, 0.0, 1.0))
        if p["m"].any():
            ax.contour(p["m"].astype(np.float32), levels=[0.5],
                       colors="red", linewidths=1.0)
        ax.set_title(f"f{p['fid']}  z={p['z']:.2f} s={p['s']:.2f}", fontsize=8)
        ax.axis("off")

    # Name the SHAPE: a silhouette that never changes form is the expected picture
    # on a rigid object and a BUG on a deforming one.
    _shape = ("per-frame deformed shape" if deformed
              else "canonical shape (no deformation field)")
    _z_lab = "decoded z (Sim(3) translation, P3D cam)"
    _s_lab = "decoded scale (Sim(3))"
    _head = (f"{block_label} RESULT @ decoded pose — obj {obj_idx}  "
             f"(green = posed GT shape, red = GT mask; {_shape})")

    axt = fig.add_subplot(gs[nrow, :])
    fr = [p["fid"] for p in panels]
    z_vals = [p["z"] for p in panels]
    s_vals = [p["s"] for p in panels]
    axt.plot(fr, z_vals, "-o", ms=3, color="tab:green", label="z (depth)")
    axt.set_xlabel("frame")
    axt.set_ylabel(_z_lab)
    axt.grid(True, alpha=0.3)
    axs = axt.twinx()
    axs.plot(fr, s_vals, "--ks", ms=3, alpha=0.7, label="scale")
    axs.set_ylabel(_s_lab)
    # A constant series gets a window instead of an axis zoomed onto float noise,
    # and no offset notation on either axis: an absolute tick label is what makes
    # "this line is flat" readable at a glance.
    for _ax, _vals in ((axt, z_vals), (axs, s_vals)):
        _lim = _flat_series_ylim(_vals)
        if _lim is not None:
            _ax.set_ylim(*_lim)
        _ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    h1, l1 = axt.get_legend_handles_labels()
    h2, l2 = axs.get_legend_handles_labels()
    axt.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    axt.set_title(_head, fontsize=10)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return True
