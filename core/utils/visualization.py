# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Visualization utilities for the SAM3D-Objects pipeline.

This module provides functions for plotting refinement histories
and other diagnostic visualizations.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def _rounded_rect(img, p0, p1, color, radius: int, thickness: int = -1):
    """Filled (thickness=-1) or stroked rounded rectangle, in place."""
    import cv2

    x0, y0 = p0
    x1, y1 = p1
    r = max(0, min(radius, (x1 - x0) // 2, (y1 - y0) // 2))
    if r == 0:
        cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA)
        return
    if thickness < 0:
        cv2.rectangle(img, (x0 + r, y0), (x1 - r, y1), color, -1)
        cv2.rectangle(img, (x0, y0 + r), (x1, y1 - r), color, -1)
    else:
        cv2.line(img, (x0 + r, y0), (x1 - r, y0), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x0 + r, y1), (x1 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x0, y0 + r), (x0, y1 - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y0 + r), (x1, y1 - r), color, thickness, cv2.LINE_AA)
    for cx, cy, a0, a1 in (
        (x0 + r, y0 + r, 180, 270), (x1 - r, y0 + r, 270, 360),
        (x1 - r, y1 - r, 0, 90), (x0 + r, y1 - r, 90, 180),
    ):
        cv2.ellipse(img, (cx, cy), (r, r), 0, a0, a1, color,
                    thickness, cv2.LINE_AA)


def draw_text_overlay(
    img,
    text: str,
    org,
    *,
    font_scale: float = 0.6,
    color=(255, 255, 255),
    panel: bool = True,
    panel_color=(0, 0, 0),
    alpha: float = 0.5,
    pad: int = 8,
    radius: int = 10,
    thickness: int = 1,
):
    """Draw a readable text label onto ``img`` (BGR or RGB) in place.

    ``org`` is the cv2 baseline-left ``(x, y)`` of the text (drop-in for
    ``cv2.putText``).  ``panel=True`` draws a rounded, semi-transparent
    ``panel_color`` box behind the text so it stays legible on any
    background.  ``panel=False``
    falls back to a contrast outline, which keeps small/coloured labels
    readable without occluding the image.  The text box is clamped to the
    image bounds.  Returns ``img``.
    """
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = int(org[0]), int(org[1])
    h, w = img.shape[:2]

    if panel:
        x0 = max(0, x - pad)
        y0 = max(0, y - th - pad)
        x1 = min(w - 1, x + tw + pad)
        y1 = min(h - 1, y + baseline + pad)
        overlay = img.copy()
        _rounded_rect(overlay, (x0, y0), (x1, y1), panel_color, radius, -1)
        cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0, dst=img)
        cv2.putText(img, text, (x, y), font, font_scale, color,
                    thickness, cv2.LINE_AA)
    else:
        cv2.putText(img, text, (x, y), font, font_scale, panel_color,
                    thickness + 2, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), font, font_scale, color,
                    thickness, cv2.LINE_AA)
    return img


def save_figure(
    fig,
    path: str,
    *,
    dpi: int = 150,
    bbox_inches: Optional[str] = "tight",
    close: bool = True,
    **savefig_kwargs,
) -> None:
    """Save a matplotlib figure, creating parent directories as needed.

    Consolidates the ``os.makedirs(dirname, exist_ok=True) +
    fig.savefig(...) + plt.close(fig)`` trio used throughout the codebase.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    path : str
        Output path.  Parent directory is created if missing.
    dpi : int
        Resolution (default 150 — the codebase-wide norm; some plots
        pass 120).
    bbox_inches : str or None
        Forwarded to ``savefig``.  Pass ``None`` to keep the raw figure
        extent.
    close : bool
        Call ``plt.close(fig)`` after saving.  Set False only if the
        caller needs the figure to stay alive.
    **savefig_kwargs
        Forwarded to ``fig.savefig`` (``facecolor``, ``transparent``,
        ``pad_inches``, ``metadata``, etc.).
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches, **savefig_kwargs)
    if close:
        plt.close(fig)


def _plot_loss_rows(rows, loss_types, loss_titles, output_path):
    """Plot a grid of loss curves.

    Parameters
    ----------
    rows : list of dict
        Each dict has 'label' (str for y-axis), 'loss_history' (list of dicts),
        'best_iteration' (int).
    loss_types : list of str
    loss_titles : list of str
    output_path : str
    """
    n_rows = len(rows)
    n_cols = len(loss_types)

    fig_height = max(3, 1.5 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, fig_height), squeeze=False)

    for row_idx, row_info in enumerate(rows):
        loss_history = row_info["loss_history"]
        best_iter = row_info["best_iteration"]

        # Use explicit iteration numbers if available (global opt with batching),
        # otherwise fall back to sequential indices (per-frame opt)
        if loss_history and "iteration" in loss_history[0]:
            iterations = [h["iteration"] for h in loss_history]
        else:
            iterations = list(range(len(loss_history)))

        for col_idx, (loss_type, title) in enumerate(zip(loss_types, loss_titles)):
            ax = axes[row_idx, col_idx]

            # Extract loss values for this type (handle missing keys for backward compat)
            if loss_type in loss_history[0]:
                values = [h[loss_type] for h in loss_history]
            else:
                values = [0.0] * len(loss_history)
                ax.text(
                    0.5, 0.5, "N/A", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="gray"
                )

            ax.plot(iterations, values, "b-", linewidth=1)

            # Mark best iteration
            if best_iter in iterations:
                best_idx = iterations.index(best_iter)
                ax.axvline(x=best_iter, color="r", linestyle="--", alpha=0.7, linewidth=0.8)
                ax.scatter([best_iter], [values[best_idx]], color="r", s=20, zorder=5)
            elif best_iter < len(values) and iterations == list(range(len(values))):
                ax.axvline(x=best_iter, color="r", linestyle="--", alpha=0.7, linewidth=0.8)
                ax.scatter([best_iter], [values[best_iter]], color="r", s=20, zorder=5)

            if row_idx == 0:
                ax.set_title(title, fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(row_info["label"], fontsize=8)
            if row_idx == n_rows - 1:
                ax.set_xlabel("Iteration", fontsize=8)

            ax.tick_params(axis="both", labelsize=7)
            ax.grid(True, alpha=0.3)

            if loss_type == "regularization":
                ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

    plt.tight_layout()
    save_figure(fig, output_path)
    print(f"Saved refinement loss plot to {output_path}")


# Curated titles for known loss keys; dynamic discovery falls back to the
# raw key name (title-cased) for anything unknown so new losses surface
# automatically without a code change here.
_LOSS_TITLES: Dict[str, str] = {
    "total": "Total Loss",
    "rgb": "RGB Loss",
    "ssim": "SSIM Loss",
    "silhouette": "Silhouette Loss",
    "depth": "Depth Loss",
    "normals": "Normals Loss",
    "perceptual": "LPIPS Loss",
    "regularization": "Regularization",
    "drift": "Token Drift",
    "dc_reg": "DC L2 Reg",
    "dc_consistency": "DC Consistency",
    "sh_reg": "SH L2 Reg",
    "sh_consistency": "SH Consistency",
    "rotation_anchor": "Rotation Anchor",
}

# Display order for known keys; unknown keys are appended alphabetically.
_LOSS_ORDER: List[str] = [
    "total", "rgb", "ssim", "silhouette", "depth", "normals",
    "perceptual", "regularization",
    "drift", "dc_reg", "dc_consistency",
    "sh_reg", "sh_consistency", "rotation_anchor",
]

# Bookkeeping metrics, not losses, that share the ``loss_history`` dicts;
# excluded to keep the loss grid focused.
#
# *Future-loss contract*: any NEW key added to a caller's loss_history dict
# is auto-discovered and plotted.  To suppress one as bookkeeping, add its
# exact name to ``_NON_LOSS_KEYS``.  Prefix matching is restricted to keys
# whose suffix is data-dependent (``grad_<param_group_name>``); do not
# expand it without a similar reason, or you'll silently swallow real
# losses whose names happen to start with the prefix.
_NON_LOSS_KEYS: set = {
    "iteration", "best_iteration",
    # Parameter drift (bookkeeping metrics, not losses)
    "token_l2", "token_rel",
    "pose_rot_l2", "pose_trans_l2", "pose_scale_l2",
    # Silhouette sub-components — already aggregated into ``silhouette``
    "silhouette_com", "silhouette_sdt", "silhouette_iou",
    # Shape-refinement occupancy diagnostics (not losses)
    "iou", "mean_empty_occ", "mean_occupied_occ",
}
# Only truly dynamic prefixes belong here.  ``grad_<group>`` is set by
# ``finetuning.py`` with per-param-group names that aren't known up front.
_NON_LOSS_PREFIXES: Tuple[str, ...] = ("grad_",)


def _discover_loss_keys(rows: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """Union of loss-like keys across all rows' loss histories.

    Drops bookkeeping fields (see ``_NON_LOSS_KEYS`` / ``_NON_LOSS_PREFIXES``)
    and orders the result by ``_LOSS_ORDER`` with unknown keys appended
    alphabetically.  A new loss key — added by any caller in the future —
    is picked up automatically: it survives the deny-list filter, falls
    into the "extra" bucket, and gets a title auto-derived from its name.
    Add it to ``_LOSS_ORDER`` / ``_LOSS_TITLES`` only to control its
    column position or display label.
    """
    present: set = set()
    for row in rows:
        for entry in row["loss_history"]:
            for k, v in entry.items():
                if k in _NON_LOSS_KEYS:
                    continue
                if any(k.startswith(p) for p in _NON_LOSS_PREFIXES):
                    continue
                # bool is a subclass of int — exclude explicitly so flag
                # fields don't get plotted as zero-valued loss curves.
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    continue
                present.add(k)

    known = [k for k in _LOSS_ORDER if k in present]
    extra = sorted(present - set(known))
    ordered = known + extra
    titles = [_LOSS_TITLES.get(k, k.replace("_", " ").title()) for k in ordered]
    return ordered, titles


def plot_refinement_history(
    refinement_data: Dict[str, Any],
    output_path: str,
) -> None:
    """
    Plot refinement loss history.

    For global refinement (batch_loss_history present): one row per object
    showing the batch-level loss curve.
    For per-frame refinement: one row per (object, frame) pair.

    Loss columns are discovered from the data so every loss recorded by the
    caller (refinement, shape-and-poses, finetuning, ...) shows up without
    a hardcoded list here.

    Parameters
    ----------
    refinement_data : dict
        Refinement history data.
    output_path : str
        Path to save the output figure.
    """
    # Detect if batch-level history is available (global refinement mode)
    has_batch = False
    for obj_idx, obj_data in refinement_data["objects"].items():
        for frame_idx, frame_data in obj_data.items():
            if "batch_loss_history" in frame_data:
                has_batch = True
            break
        break

    if has_batch:
        # Global refinement: one row per object using the batch-level loss curve
        rows = []
        for obj_idx in sorted(refinement_data["objects"].keys(), key=lambda x: int(x)):
            obj_data = refinement_data["objects"][obj_idx]
            # All frames share the same batch history; take from the first frame
            for frame_idx, frame_data in obj_data.items():
                if "batch_loss_history" in frame_data:
                    rows.append({
                        "label": f"Obj {obj_idx}",
                        "loss_history": frame_data["batch_loss_history"],
                        "best_iteration": frame_data["best_iteration"],
                    })
                    break
    else:
        # Per-frame refinement: one row per (object, frame)
        rows = []
        for obj_idx, obj_data in refinement_data["objects"].items():
            for frame_idx, frame_data in obj_data.items():
                rows.append({
                    "label": f"Obj {obj_idx}, F{frame_idx}",
                    "loss_history": frame_data["loss_history"],
                    "best_iteration": frame_data["best_iteration"],
                })
        rows.sort(key=lambda x: x["label"])

    if not rows:
        print("No refinement data to plot")
        return

    loss_types, loss_titles = _discover_loss_keys(rows)
    if not loss_types:
        print("No loss keys discovered in refinement data")
        return

    _plot_loss_rows(rows, loss_types, loss_titles, output_path)


def _compute_pca_colors(feats: np.ndarray) -> np.ndarray:
    """Reduce (N, C) features to (N, 3) PCA-colored RGB in [0, 1].

    Parameters
    ----------
    feats : np.ndarray (N, C)
        Per-voxel feature vectors.

    Returns
    -------
    np.ndarray (N, 3)
        Normalized PCA colors suitable for vertex coloring.
    """
    import torch as _torch

    F = _torch.from_numpy(feats).float() if isinstance(feats, np.ndarray) else feats.float()
    if F.shape[0] == 0:
        return np.zeros((0, 3), np.float32)
    F_centered = F - F.mean(dim=0, keepdim=True)
    _, _, Vh = _torch.linalg.svd(F_centered, full_matrices=False)
    k = min(3, Vh.shape[0])  # fewer than 3 voxels -> fewer PCA components
    pca3 = np.full((F.shape[0], 3), 0.5, dtype=np.float32)
    pca3[:, :k] = (F_centered @ Vh[:k].T).numpy()
    for c in range(3):
        lo, hi = pca3[:, c].min(), pca3[:, c].max()
        if hi - lo > 1e-8:
            pca3[:, c] = (pca3[:, c] - lo) / (hi - lo)
        else:
            pca3[:, c] = 0.5
    return pca3.astype(np.float32)


def _upsample_shape_features_to_voxels(
    shape_latent: "torch.Tensor",
    voxel_coords: np.ndarray,
    grid_size: int = 64,
) -> np.ndarray:
    """Upsample shape latent features from 16³ to occupied 64³ voxel positions.

    The shape latent lives on a 16³ grid (4096 positions × 8 features).
    This function uses trilinear interpolation to sample features at the
    occupied voxel positions in the 64³ grid.

    Parameters
    ----------
    shape_latent : torch.Tensor
        Shape latent of shape ``(1, 4096, 8)`` or ``(4096, 8)``.
    voxel_coords : np.ndarray (N, 3)
        Integer grid positions of occupied voxels in the 64³ grid.
    grid_size : int
        Target voxel grid resolution (default 64).

    Returns
    -------
    np.ndarray (N, 8)
        Interpolated features at each occupied voxel position.
    """
    import torch
    import torch.nn.functional as F

    lat = shape_latent.detach().cpu().float()
    if lat.dim() == 3:
        lat = lat.squeeze(0)  # (4096, 8)
    latent_res = round(lat.shape[0] ** (1.0 / 3))  # 16

    # Reshape to (1, 8, 16, 16, 16) matching ss_decoder convention:
    # shape_latent.permute(0, 2, 1).view(1, 8, 16, 16, 16)
    feat_vol = lat.reshape(latent_res, latent_res, latent_res, 8)
    feat_vol = feat_vol.permute(3, 0, 1, 2).unsqueeze(0)  # (1, 8, D, H, W)

    # Normalize voxel coords to [-1, 1] for grid_sample
    # Voxel center at integer position p → (p + 0.5) / grid_size * 2 - 1
    coords = torch.from_numpy(voxel_coords).float()
    norm = (coords + 0.5) / grid_size * 2 - 1  # (N, 3) in (x, y, z)

    # grid_sample 5D: grid[..., 0]=W, grid[..., 1]=H, grid[..., 2]=D
    # Our spatial ordering: x=D, y=H, z=W → grid = (z, y, x)
    grid = norm[:, [2, 1, 0]].reshape(1, 1, 1, -1, 3)  # (1, 1, 1, N, 3)

    # For 5D input, mode="bilinear" performs trilinear interpolation
    sampled = F.grid_sample(
        feat_vol, grid, mode="bilinear", align_corners=False, padding_mode="border",
    )  # (1, 8, 1, 1, N)

    return sampled.squeeze(0).squeeze(1).squeeze(1).T.numpy()  # (N, 8)


def compute_voxel_colors(
    color_mode: str,
    voxel_coords: np.ndarray,
    grid_size: int = 64,
    slat: Any = None,
    shape_latent: "Optional[torch.Tensor]" = None,
) -> np.ndarray:
    """Compute per-voxel (N, 3) RGB colors for the given color mode.

    Parameters
    ----------
    color_mode : str
        ``"xyz"`` — normalized grid position (R=X, G=Y, B=Z).
        ``"shape_pca"`` — PCA of shape tokens upsampled to 64³.
        ``"slat_pca"`` — PCA of SLAT token features.
    voxel_coords : np.ndarray (N, 3)
        Integer grid positions of occupied voxels.
    grid_size : int
        Voxel grid resolution (default 64).
    slat : SparseTensor, optional
        Required for ``"slat_pca"`` mode.
    shape_latent : torch.Tensor, optional
        Required for ``"shape_pca"`` mode.  Shape ``(1, 4096, 8)``.

    Returns
    -------
    np.ndarray (N, 3)
        Per-voxel colors in [0, 1].

    Raises
    ------
    ValueError
        If required data is missing for the requested color mode.
    """
    if color_mode == "xyz":
        return ((voxel_coords + 0.5) / grid_size).astype(np.float32)

    if color_mode == "shape_pca":
        if shape_latent is None:
            raise ValueError(
                "color_mode='shape_pca' requires shape_latent but it is None"
            )
        feats = _upsample_shape_features_to_voxels(
            shape_latent, voxel_coords, grid_size,
        )
        return _compute_pca_colors(feats)

    if color_mode == "slat_pca":
        if slat is None:
            raise ValueError(
                "color_mode='slat_pca' requires slat but it is None"
            )
        feats = slat.feats.detach().cpu().float().numpy()
        return _compute_pca_colors(feats)

    raise ValueError(f"Unknown voxel color_mode: {color_mode!r}")


def visualize_slat_voxels(
    slat: Any = None,
    output_path: str = "",
    title: str = "",
    *,
    xyz: "Optional[np.ndarray]" = None,
    feats: "Optional[torch.Tensor]" = None,
) -> None:
    """
    Visualize a voxel grid colored by PCA of features.

    Accepts either a SparseTensor (the ``slat`` argument) or raw
    ``xyz`` / ``feats`` arrays.  Features are reduced to 3 components
    via PCA and mapped to RGB.

    Parameters
    ----------
    slat : SparseTensor, optional
        SLAT tokens with ``.coords`` (N, 4) and ``.feats`` (N, C).
        coords[:, 0] is batch index, cols 1:4 are spatial xyz.
    output_path : str
        Path to save the output PNG.
    title : str, optional
        Title shown on the figure.
    xyz : np.ndarray, optional
        (N, 3) spatial positions.  Used instead of *slat* when provided.
    feats : torch.Tensor, optional
        (N, C) feature vectors.  Used instead of *slat* when provided.
    """
    import torch

    if xyz is None or feats is None:
        coords = slat.coords.detach().cpu()   # (N, 4)
        xyz = coords[:, 1:].float().numpy()   # (N, 3)
        feats = slat.feats.detach().cpu()      # (N, C)

    F = feats.float() if isinstance(feats, torch.Tensor) else torch.from_numpy(feats).float()

    # PCA → 3 components
    F_centered = F - F.mean(dim=0, keepdim=True)
    _, _, Vh = torch.linalg.svd(F_centered, full_matrices=False)
    pca3 = (F_centered @ Vh[:3].T).numpy()  # (N, 3)

    # Normalize each component independently to [0, 1] for RGB
    for c in range(3):
        lo, hi = pca3[:, c].min(), pca3[:, c].max()
        if hi - lo > 1e-8:
            pca3[:, c] = (pca3[:, c] - lo) / (hi - lo)
        else:
            pca3[:, c] = 0.5

    viewpoints = [
        (30, 45, "Front-right"),
        (30, 135, "Back-left"),
        (75, 45, "Top-down"),
    ]

    fig = plt.figure(figsize=(5 * len(viewpoints), 5))
    for i, (elev, azim, vp_label) in enumerate(viewpoints):
        ax = fig.add_subplot(1, len(viewpoints), i + 1, projection="3d")
        ax.scatter(
            xyz[:, 0], xyz[:, 1], xyz[:, 2],
            c=pca3, s=40, marker="s", edgecolors="none", alpha=0.9,
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(0, 64)
        ax.set_ylim(0, 64)
        ax.set_zlim(0, 64)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"{vp_label}  ({xyz.shape[0]} voxels)", fontsize=9)

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    save_figure(fig, output_path)
    print(f"  Saved voxel PCA visualization to {output_path}")


# Unit cube (side 1, centered at origin) — one instance per occupied voxel.
_VOXEL_CUBE_V = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
], dtype=np.float32) - 0.5
_VOXEL_CUBE_F = np.array([  # 12 triangles (2 per face, CCW from outside)
    [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
    [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
    [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
], dtype=np.int64)


# Axis-gizmo overlay for the voxel-mesh composite.  Local +X/+Y/+Z are drawn in
# each panel so a reader can tell which way the object's own frame points --
# without it the six views are unlabelled and calibrating
# ``gt_shapes_inversion.local_rotation`` is guesswork.
_AXIS_GIZMO_COLORS = ((0.90, 0.10, 0.10), (0.10, 0.65, 0.10), (0.15, 0.35, 0.95))
_AXIS_GIZMO_LABELS = ("+X", "+Y", "+Z")


def _axis_gizmo_dirs(cameras, image_size):
    """Screen-space direction of each local +axis, as the panel displays it.

    Returns one ``(ux, uy, inplane, toward)`` per axis: ``(ux, uy)`` is a unit
    direction in DISPLAY pixels (y down), ``inplane`` is the projected length as
    a fraction of the image (small ⇒ the axis points along the view direction),
    and ``toward`` is True when the axis points out of the screen.

    Projected through the caller's own ``cameras`` and then flipped exactly as
    ``render_slat_voxel_mesh`` flips the rasterized image, so the gizmo cannot
    disagree with the picture it annotates.
    """
    import torch
    L = 0.5  # half the [-0.5, 0.5] voxel cube — direction only, length unused
    pts = torch.tensor(
        [[0.0, 0.0, 0.0], [L, 0.0, 0.0], [0.0, L, 0.0], [0.0, 0.0, L]],
        dtype=torch.float32, device=cameras.device,
    )
    scr = cameras.transform_points_screen(
        pts[None], image_size=((image_size, image_size),))[0]
    # The render does torch.flip(img, dims=[0, 1]) -- a 180 degree image
    # rotation.  Apply it to the projected points too.
    sx = (image_size - 1) - scr[:, 0]
    sy = (image_size - 1) - scr[:, 1]
    view_z = cameras.get_world_to_view_transform().transform_points(
        pts[None])[0][:, 2]
    out = []
    for i in range(3):
        dx = float(sx[i + 1] - sx[0])
        dy = float(sy[i + 1] - sy[0])
        n = (dx * dx + dy * dy) ** 0.5
        u = (dx / n, dy / n) if n > 1e-6 else (0.0, 0.0)
        out.append((u[0], u[1], n / float(image_size),
                    bool(view_z[i + 1] < view_z[0])))
    return out


def _draw_axis_gizmo(ax, dirs, anchor=(0.15, 0.15), length=0.11):
    """Draw the triad from :func:`_axis_gizmo_dirs` in a panel corner.

    An axis pointing (nearly) along the view direction has no usable 2D
    direction, so it becomes a marker instead of an arrow: filled = toward the
    viewer, hollow = away.
    """
    ax0, ay0 = anchor
    for (ux, uy, inplane, toward), col, lab in zip(
            dirs, _AXIS_GIZMO_COLORS, _AXIS_GIZMO_LABELS):
        if inplane < 0.08:
            ax.plot([ax0], [ay0], transform=ax.transAxes, marker="o",
                    mfc=(col if toward else "none"), mec=col, ms=10, mew=2.0,
                    clip_on=False)
            ax.text(ax0 + 0.07, ay0 - 0.07, lab, transform=ax.transAxes,
                    color=col, fontsize=9, fontweight="bold",
                    ha="center", va="center")
            continue
        # axes-fraction y is up, display y is down -> negate uy
        ax.annotate(
            "", xy=(ax0 + ux * length, ay0 - uy * length), xytext=(ax0, ay0),
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0,
                            shrinkA=0, shrinkB=0),
        )
        ax.text(ax0 + ux * length * 1.5, ay0 - uy * length * 1.5, lab,
                transform=ax.transAxes, color=col, fontsize=9,
                fontweight="bold", ha="center", va="center")


def build_voxel_cube_mesh(xyz, voxel_colors, grid_size=64):
    """Sparse occupied voxels → a triangle mesh of per-voxel cubes.

    Each integer grid index in *xyz* becomes an axis-aligned cube (8 verts, 12
    triangles) centered at ``(xyz + 0.5) / grid_size - 0.5`` with side
    ``1 / grid_size``, tinted by the row-aligned RGB in *voxel_colors*.

    Returns ``(verts (N*8, 3) float32, faces (N*12, 3) int64,
    colors (N*8, 3) float32)``.
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    N = xyz.shape[0]
    voxel_size = 1.0 / grid_size
    centers = (xyz + 0.5) / grid_size - 0.5  # (N, 3), voxel center in [-0.5, 0.5]
    verts = (_VOXEL_CUBE_V * voxel_size + centers[:, None, :]).reshape(-1, 3)
    faces = _VOXEL_CUBE_F + (np.arange(N) * 8)[:, None, None]
    colors = np.repeat(np.asarray(voxel_colors, dtype=np.float32), 8, axis=0)
    return verts.astype(np.float32), faces.reshape(-1, 3), colors


def render_slat_voxel_mesh(
    slat: Any = None,
    output_path: str = "",
    title: str = "",
    *,
    xyz: "Optional[np.ndarray]" = None,
    image_size: int = 512,
    distance: float = 2.0,
    fov: float = 40.0,
    grid_size: int = 64,
    save_obj: bool = True,
    shading: bool = False,
    color_mode: str = "xyz",
    shape_latent: "Optional[Any]" = None,
    voxel_colors: "Optional[np.ndarray]" = None,
) -> None:
    """Render sparse voxel grid as a cube mesh with configurable coloring.

    Each occupied voxel becomes a small cube whose vertex color is
    determined by *color_mode*.

    Parameters
    ----------
    slat : SparseTensor, optional
        SLAT tokens with ``.coords`` (N, 4) and ``.feats`` (N, C).
        coords[:, 0] is batch index, cols 1:4 are spatial xyz.
    output_path : str
        Path to save the output PNG (multi-view composite).
    title : str, optional
        Title shown on the figure.
    xyz : np.ndarray, optional
        (N, 3) integer grid positions.  Used instead of *slat* when provided.
    image_size : int
        Rendered image size per view (square).
    distance : float
        Camera distance from the mesh center.
    fov : float
        Field of view in degrees.
    grid_size : int
        SLAT grid resolution (default 64 for the 64³ voxel grid).
    save_obj : bool
        If True, save an OBJ file with vertex colors next to the PNG.
    shading : bool
        If True, apply Phong shading with a directional light.
        If False (default), render vertex colors directly.
    color_mode : str
        ``"xyz"`` — normalized grid position (R=X, G=Y, B=Z).
        ``"shape_pca"`` — PCA of shape tokens (upsampled from 16³).
        ``"slat_pca"`` — PCA of SLAT token features.
        Ignored when *voxel_colors* is given.
    shape_latent : torch.Tensor, optional
        Shape latent ``(1, 4096, 8)``.  Required for ``color_mode="shape_pca"``.
    voxel_colors : np.ndarray, optional
        Explicit per-voxel RGB ``(N, 3)`` in ``[0, 1]`` (row-aligned to *xyz*).
        When provided, used directly and *color_mode* is bypassed — e.g. for
        correspondence colouring (each voxel tinted by the canonical voxel it
        maps to).
    """
    import torch

    if xyz is None:
        coords = slat.coords.detach().cpu()
        xyz = coords[:, 1:].float().numpy()  # (N, 3) integer grid positions

    N = xyz.shape[0]
    if N == 0:
        print(f"  Skipping voxel mesh render: no occupied voxels")
        return

    # Per-voxel colors: explicit override (e.g. correspondence colouring) or
    # computed from color_mode.
    if voxel_colors is not None:
        voxel_colors = np.asarray(voxel_colors, dtype=np.float32)
        if voxel_colors.shape != (N, 3):
            raise ValueError(
                f"voxel_colors must be ({N}, 3) to match xyz; "
                f"got {voxel_colors.shape}"
            )
    else:
        voxel_colors = compute_voxel_colors(
            color_mode, xyz, grid_size=grid_size,
            slat=slat, shape_latent=shape_latent,
        )  # (N, 3)

    # 8 verts + 12 faces per voxel (vectorized; see build_voxel_cube_mesh).
    all_verts, all_faces, all_colors = build_voxel_cube_mesh(
        xyz, voxel_colors, grid_size=grid_size)

    # Optionally save OBJ with vertex colors (v x y z r g b)
    if save_obj:
        obj_path = output_path.rsplit(".", 1)[0] + ".obj"
        with open(obj_path, "w") as f:
            f.write(f"# Voxel mesh: {N} cubes, grid_size={grid_size}\n")
            for vi in range(all_verts.shape[0]):
                v = all_verts[vi]
                c = all_colors[vi]
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} "
                        f"{c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n")
            for fi in range(all_faces.shape[0]):
                face = all_faces[fi] + 1  # OBJ is 1-indexed
                f.write(f"f {face[0]} {face[1]} {face[2]}\n")
        print(f"  Saved voxel mesh OBJ to {obj_path}")

    # Render multi-view composite using PyTorch3D
    try:
        from pytorch3d.renderer import (
            BlendParams,
            FoVPerspectiveCameras,
            MeshRasterizer,
            RasterizationSettings,
            TexturesVertex,
            look_at_view_transform,
        )
        from pytorch3d.renderer.blending import hard_rgb_blend
        from pytorch3d.structures import Meshes
    except ImportError:
        print("  PyTorch3D not available — skipping voxel mesh rendering")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    verts_t = torch.from_numpy(all_verts).float().to(device)
    faces_t = torch.from_numpy(all_faces).long().to(device)
    colors_t = torch.from_numpy(all_colors).float().to(device)

    mesh = Meshes(
        verts=[verts_t], faces=[faces_t],
        textures=TexturesVertex(verts_features=[colors_t]),
    )

    viewpoints = [
        ("front", 0, 0), ("back", 0, 180),
        ("left", 0, -90), ("right", 0, 90),
        ("top", 90, 0), ("bottom", -90, 0),
    ]

    blend_params = BlendParams(
        sigma=1e-4, gamma=1e-4, background_color=(1.0, 1.0, 1.0),
    )
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
        bin_size=0,  # naive rasterization — avoids bin overflow for dense meshes
    )

    renders = {}
    gizmos = {}
    for name, elev, azim in viewpoints:
        R, T = look_at_view_transform(dist=distance, elev=elev, azim=azim)
        cameras = FoVPerspectiveCameras(
            device=device, R=R.to(device), T=T.to(device), fov=fov,
        )
        rasterizer = MeshRasterizer(
            cameras=cameras, raster_settings=raster_settings,
        )
        with torch.no_grad():
            fragments = rasterizer(mesh)
            if shading:
                from pytorch3d.renderer import (
                    PointLights, HardFlatShader,
                )
                # Place light at camera position for consistent illumination
                cam_pos = cameras.get_camera_center()
                lights = PointLights(
                    device=device,
                    location=cam_pos,
                    ambient_color=((0.6, 0.6, 0.6),),
                    diffuse_color=((0.4, 0.4, 0.4),),
                    specular_color=((0.1, 0.1, 0.1),),
                )
                shader = HardFlatShader(
                    device=device, cameras=cameras, lights=lights,
                    blend_params=blend_params,
                )
                img = shader(fragments, mesh)
                img = img[0, ..., :3]
            else:
                texels = mesh.sample_textures(fragments)
                img = hard_rgb_blend(texels, fragments, blend_params)
                img = img[0, ..., :3]
        img = torch.flip(img, dims=[0, 1])
        renders[name] = torch.clamp(img, 0.0, 1.0).cpu().numpy()
        gizmos[name] = _axis_gizmo_dirs(cameras, image_size)

    # Composite: 2 rows × 3 cols
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    view_order = ["front", "left", "top", "back", "right", "bottom"]
    for idx, name in enumerate(view_order):
        ax = axes[idx // 3][idx % 3]
        ax.imshow(renders[name])
        ax.axis("off")
        # No view-name title: "front"/"top"/... name the CAMERA, which says
        # nothing about the object's own frame.  The gizmo does, so it is the
        # only label.
        _draw_axis_gizmo(ax, gizmos[name])
    if title:
        fig.suptitle(f"{title}  ({N} voxels)", fontsize=12)
    plt.tight_layout()
    save_figure(fig, output_path)
    print(f"  Saved voxel mesh rendering to {output_path}")


def _compose_depth_alpha_image(depth, alpha, cmap_name="turbo", vrange=None):
    """Compose a (depth, alpha) pair into an RGB PIL image.

    Depth is normalized to [0, 1] using the alpha-masked min/max so the
    colormap covers the rendered surface, then rendered through ``cmap_name``
    and alpha-composited over a white background.  Pixels with alpha ≈ 0
    show as pure white.

    ``vrange=(lo, hi)`` overrides that per-image normalisation with a SHARED one.  Two
    images normalised each to its own range cannot be compared by colour -- which is the
    whole point when the pair is an observed and a rendered depth being aligned to each
    other (`icp_clouds/renders/`).  ``None`` keeps the per-image normalisation.
    """
    from PIL import Image as PILImage

    a = alpha.detach().cpu().numpy() if hasattr(alpha, "detach") else np.asarray(alpha)
    d = depth.detach().cpu().numpy() if hasattr(depth, "detach") else np.asarray(depth)
    a = a.astype(np.float64)
    valid = a > 0.05
    if vrange is not None:
        d_lo, d_hi = float(vrange[0]), float(vrange[1])
    elif valid.any():
        d_lo, d_hi = float(d[valid].min()), float(d[valid].max())
    else:
        d_lo, d_hi = 0.0, 1.0
    d_norm = np.clip((d - d_lo) / max(d_hi - d_lo, 1e-6), 0.0, 1.0)
    rgb = matplotlib.colormaps[cmap_name](d_norm)[..., :3]
    a3 = a[..., None]
    out = a3 * rgb + (1.0 - a3) * 1.0
    return PILImage.fromarray((out * 255).astype(np.uint8))


def _compute_psnr_and_l1(img, gt, mask_np):
    """Return (l1_heatmap, psnr) for a single image vs GT, optionally masked."""
    import torch

    l1 = torch.abs(img - gt).mean(dim=-1).numpy()
    if mask_np is not None:
        l1 = np.where(mask_np, l1, 0.0)
        mask_t = torch.from_numpy(mask_np).unsqueeze(-1)
        masked_sq = ((img - gt) * mask_t) ** 2
        n_pixels = mask_t.sum().item() * img.shape[-1]  # masked pixels × channels
        mse = masked_sq.sum().item() / max(n_pixels, 1.0)
    else:
        mse = ((img - gt) ** 2).mean().item()
    psnr = -10 * torch.log10(torch.tensor(mse + 1e-8)).item()
    return l1, psnr


def _fill_comparison_row(axes_row, before, after, l1_before, l1_after,
                         psnr_before, psnr_after, vmax, show_titles=True):
    """Fill columns 1-4 of a row: Before | After | L1 Before | L1 After.

    Column 0 (GT or label) is handled by the caller.
    """
    axes_row[1].imshow(before.numpy())
    axes_row[1].set_title(f"PSNR {psnr_before:.1f}" if not show_titles
                          else f"Before (PSNR {psnr_before:.1f})", fontsize=11)

    axes_row[2].imshow(after.numpy())
    axes_row[2].set_title(f"PSNR {psnr_after:.1f}" if not show_titles
                          else f"After (PSNR {psnr_after:.1f})", fontsize=11)

    im3 = axes_row[3].imshow(l1_before, cmap="hot", vmin=0, vmax=vmax)
    if show_titles:
        axes_row[3].set_title("L1 Error Before", fontsize=11)
    plt.colorbar(im3, ax=axes_row[3], fraction=0.046, pad=0.04)

    im4 = axes_row[4].imshow(l1_after, cmap="hot", vmin=0, vmax=vmax)
    if show_titles:
        axes_row[4].set_title("L1 Error After", fontsize=11)
    plt.colorbar(im4, ax=axes_row[4], fraction=0.046, pad=0.04)


def save_render_comparison(
    gt_image: "torch.Tensor",
    rendered: "torch.Tensor",
    output_path: str,
    before_render: "torch.Tensor | None" = None,
    mask: "torch.Tensor | None" = None,
    title: str = "",
    before_axes_overlay: "np.ndarray | None" = None,
    after_axes_overlay: "np.ndarray | None" = None,
) -> None:
    """
    Save a render comparison figure with L1 error heatmaps.

    Without ``before_render``, creates a 3-column figure:
    ``GT | Rendered (PSNR) | L1 Error``.

    With ``before_render``, creates a 5-column figure:
    ``GT | Before (PSNR) | After (PSNR) | L1 Before | L1 After``.

    When ``after_axes_overlay`` is provided, a "Pose Axes" row is appended
    at the bottom showing rendered images with projected 2D coordinate axes.

    Parameters
    ----------
    gt_image : torch.Tensor
        Ground truth image (H, W, 3) in [0, 1].
    rendered : torch.Tensor
        Current rendered image (H, W, 3) in [0, 1].
    output_path : str
        Path to save the output PNG.
    before_render : torch.Tensor, optional
        Rendered image before refinement (H, W, 3) in [0, 1].
        When provided, the figure shows before/after comparison.
    mask : torch.Tensor, optional
        Boolean mask (H, W).  When provided, L1 error heatmaps and PSNR
        are computed only within the masked region.
    title : str, optional
        Title shown on the figure.
    before_axes_overlay : np.ndarray, optional
        Before-render with pose axes overlay (H, W, 3) float in [0, 1].
    after_axes_overlay : np.ndarray, optional
        After-render with pose axes overlay (H, W, 3) float in [0, 1].
    """
    import torch

    gt = gt_image.detach().cpu().float()
    after = rendered.detach().cpu().float()

    # Prepare mask (H, W) boolean numpy array
    if mask is not None:
        mask_np = mask.detach().cpu().bool().numpy()
    else:
        mask_np = None

    has_axes = after_axes_overlay is not None

    if before_render is not None:
        # 5-column layout: GT | Before | After | L1 Before | L1 After
        before = before_render.detach().cpu().float()

        l1_before, psnr_before = _compute_psnr_and_l1(before, gt, mask_np)
        l1_after, psnr_after = _compute_psnr_and_l1(after, gt, mask_np)
        vmax = max(l1_before.max(), l1_after.max(), 0.05)

        n_rows = 1 + (1 if has_axes else 0)
        fig, axes = plt.subplots(n_rows, 5, figsize=(25, 5 * n_rows), squeeze=False)

        # Row 0: GT in col 0, renders in cols 1-4
        axes[0, 0].imshow(gt.numpy())
        axes[0, 0].set_title("GT", fontsize=11)
        _fill_comparison_row(
            axes[0], before, after,
            l1_before, l1_after, psnr_before, psnr_after, vmax,
            show_titles=True,
        )
        if has_axes:
            ar = n_rows - 1  # axes row is always last
            axes[ar, 0].set_visible(False)
            if before_axes_overlay is not None:
                axes[ar, 1].imshow(before_axes_overlay)
            else:
                axes[ar, 1].set_visible(False)
            axes[ar, 2].imshow(after_axes_overlay)
            axes[ar, 3].set_visible(False)
            axes[ar, 4].set_visible(False)
            axes_row_y = 1.0 - (ar + 0.5) / n_rows
            fig.text(
                0.01, axes_row_y, "Pose Axes",
                fontsize=13, fontweight="bold", rotation=90,
                va="center", ha="left",
            )
    else:
        # 3-column: GT | Rendered | L1 Error
        l1 = torch.abs(after - gt).mean(dim=-1).numpy()

        if mask_np is not None:
            l1 = np.where(mask_np, l1, 0.0)
            mask_t = torch.from_numpy(mask_np).unsqueeze(-1)
            mse = (((after - gt) * mask_t) ** 2).mean().item()
        else:
            mse = ((after - gt) ** 2).mean().item()

        vmax = max(l1.max(), 0.05)
        psnr = -10 * torch.log10(torch.tensor(mse + 1e-8)).item()

        n_rows = 1 + (1 if has_axes else 0)
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows), squeeze=False)

        axes[0, 0].imshow(gt.numpy())
        axes[0, 0].set_title("GT", fontsize=11)

        axes[0, 1].imshow(after.numpy())
        axes[0, 1].set_title(f"Rendered (PSNR {psnr:.1f})", fontsize=11)

        im2 = axes[0, 2].imshow(l1, cmap="hot", vmin=0, vmax=vmax)
        axes[0, 2].set_title("L1 Error", fontsize=11)
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

        if has_axes:
            ar = n_rows - 1
            axes[ar, 0].set_visible(False)
            axes[ar, 1].imshow(after_axes_overlay)
            axes[ar, 2].set_visible(False)
            axes_row_y = 1.0 - (ar + 0.5) / n_rows
            fig.text(
                0.01, axes_row_y, "Pose Axes",
                fontsize=13, fontweight="bold", rotation=90,
                va="center", ha="left",
            )

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    if title:
        fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    save_figure(fig, output_path)


def visualize_object_tracks_2d(
    tracks_2d: "Dict[int, np.ndarray]",
    all_frame_indices: "List[int]",
    W: int,
    H: int,
    output_path: str,
    rendered_frames: "Optional[List[np.ndarray]]" = None,
    keyframe_indices: "Optional[List[int]]" = None,
    trail_length: int = 16,
    duration: int = 100,
    *,
    gt_frames: "Optional[List[np.ndarray]]" = None,
) -> None:
    """
    Create a GIF with 2D object point tracks overlaid on frames.

    Each object's tracked points are drawn with a shared base hue but
    varying brightness so individual points are distinguishable.
    Point 0 (the centroid) is drawn larger than the others.

    Parameters
    ----------
    tracks_2d : dict
        ``{obj_idx: ndarray (T, P, 2)}`` pixel coordinates for each frame
        and tracked point.  ``P`` is the number of tracked points; index 0
        is the centroid.
    all_frame_indices : list of int
        Frame indices corresponding to the T dimension.
    W, H : int
        Image dimensions.
    output_path : str
        Path to save the output GIF.
    rendered_frames : list of ndarray, optional
        Background frames (H, W, 3) uint8.  If None and ``gt_frames`` is
        also None, uses dark gray background.
    keyframe_indices : list of int, optional
        Keyframe indices to mark with ring markers.
    trail_length : int
        Number of past frames to draw trailing lines for.
    duration : int
        Frame duration in milliseconds.
    gt_frames : list of ndarray, optional
        When provided **together with** ``rendered_frames``, the output
        becomes a single side-by-side video ``[Rendered | GT]``: the same
        track overlay is drawn on both panels with identical pixel
        coordinates, then the panels are concatenated horizontally
        (output width = ``2 * W``).  Used by the FINAL block as one
        unified rendered-vs-GT visualisation.
    """
    import cv2
    import numpy as np
    from PIL import Image

    side_by_side = (rendered_frames is not None and gt_frames is not None)
    keyframe_set = set(keyframe_indices) if keyframe_indices else set()
    obj_indices = sorted(tracks_2d.keys())
    n_objects = max(len(obj_indices), 1)

    # Generate per-object base hues, then per-point color variations
    def _make_colors(obj_i: int, n_pts: int):
        """Return list of RGB tuples for each point of an object."""
        base_hue = int(180 * obj_i / n_objects)
        colors = []
        for p in range(n_pts):
            # Vary saturation/value slightly per point
            sat = max(120, 255 - p * 10)
            val = max(150, 255 - p * 8)
            hsv = np.array([[[base_hue, sat, val]]], dtype=np.uint8)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
            colors.append((int(rgb[0]), int(rgb[1]), int(rgb[2])))
        return colors

    # Pre-compute colors: {obj_idx: [color_p0, color_p1, ...]}
    obj_colors = {}
    for i, obj_idx in enumerate(obj_indices):
        n_pts = tracks_2d[obj_idx].shape[1]
        obj_colors[obj_idx] = _make_colors(i, n_pts)

    # Draw trail lines with faded colors (blended toward background gray)
    # instead of per-segment canvas copy + alpha blend.
    bg_val = 40

    def _faded_color(color: tuple, alpha: float) -> tuple:
        """Blend *color* toward bg_val by *alpha* (0 = bg, 1 = full color)."""
        a = alpha * 0.6  # max trail opacity
        return tuple(int(bg_val * (1 - a) + c * a) for c in color)

    def _draw_overlay(canvas: np.ndarray, t_idx: int, fi) -> None:
        """In-place overlay of trail + per-point markers + frame counter
        on a single (H, W, 3) canvas."""
        for obj_idx in obj_indices:
            pts = tracks_2d[obj_idx]  # (T, P, 2)
            colors = obj_colors[obj_idx]
            n_pts = pts.shape[1]

            for p_idx in range(n_pts):
                color = colors[p_idx]
                is_centroid = (p_idx == 0)
                thickness = 2 if is_centroid else 1

                # Trail with faded colors
                start = max(0, t_idx - trail_length)
                trail_len = t_idx - start
                for j in range(start, t_idx):
                    alpha = (j - start + 1) / (trail_len + 1)
                    pt1 = (int(pts[j, p_idx, 0]), int(pts[j, p_idx, 1]))
                    pt2 = (int(pts[j + 1, p_idx, 0]), int(pts[j + 1, p_idx, 1]))
                    faded = _faded_color(color, alpha)
                    cv2.line(canvas, pt1, pt2, faded, thickness=thickness, lineType=cv2.LINE_AA)

                cx, cy = int(pts[t_idx, p_idx, 0]), int(pts[t_idx, p_idx, 1])
                is_keyframe = fi in keyframe_set
                if is_centroid:
                    radius = 7 if is_keyframe else 5
                    cv2.circle(canvas, (cx, cy), radius, color, -1, lineType=cv2.LINE_AA)
                    if is_keyframe:
                        cv2.circle(canvas, (cx, cy), radius + 2, (255, 255, 255), 1, lineType=cv2.LINE_AA)
                else:
                    radius = 4 if is_keyframe else 3
                    cv2.circle(canvas, (cx, cy), radius, color, -1, lineType=cv2.LINE_AA)

        draw_text_overlay(canvas, f"Frame {fi}", (12, 30), font_scale=0.65)

    frames = []
    for t_idx, fi in enumerate(all_frame_indices):
        if side_by_side:
            # [Rendered | GT] panel.  Draw the same overlay on both panels
            # so the user sees the predicted track at the same screen
            # position on each side — left shows where it landed on the
            # rendered Gaussians, right shows where it lands on the GT.
            left = (
                rendered_frames[t_idx].copy()
                if t_idx < len(rendered_frames)
                else np.full((H, W, 3), 40, dtype=np.uint8)
            )
            right = (
                gt_frames[t_idx].copy()
                if t_idx < len(gt_frames)
                else np.full((H, W, 3), 40, dtype=np.uint8)
            )
            _draw_overlay(left, t_idx, fi)
            _draw_overlay(right, t_idx, fi)
            # Add panel labels under the frame counter
            for canvas, label in ((left, "Rendered"), (right, "GT")):
                draw_text_overlay(
                    canvas, label, (W - 120, 30), font_scale=0.6,
                )
            combined = np.concatenate([left, right], axis=1)
            frames.append(Image.fromarray(combined))
        else:
            # Single panel.
            if rendered_frames is not None and t_idx < len(rendered_frames):
                canvas = rendered_frames[t_idx].copy()
            else:
                canvas = np.full((H, W, 3), 40, dtype=np.uint8)
            _draw_overlay(canvas, t_idx, fi)
            frames.append(Image.fromarray(canvas))

    if not frames:
        return

    from .interpolation import _save_frames_as_video
    _save_frames_as_video(frames, output_path, duration=duration)


def visualize_object_tracks_3d(
    tracks_3d: "Dict[int, np.ndarray]",
    all_frame_indices: "List[int]",
    output_path: str,
    keyframe_indices: "Optional[List[int]]" = None,
) -> None:
    """
    Save a 3D trajectory plot showing object point tracks over time.

    Creates a figure with three viewpoints (front-right, top-down, side).
    Each object gets a distinct base color; the centroid (point 0) is drawn
    with thicker lines than the sampled surface points.

    Parameters
    ----------
    tracks_3d : dict
        ``{obj_idx: ndarray (T, P, 3)}`` in R3 world coordinates.
        Point index 0 is the centroid.
    all_frame_indices : list of int
        Frame indices (for labeling).
    output_path : str
        Path to save the output PNG.
    keyframe_indices : list of int, optional
        Keyframe indices to highlight.
    """
    import numpy as np

    keyframe_set = set(keyframe_indices) if keyframe_indices else set()
    obj_indices = sorted(tracks_3d.keys())

    cmap = matplotlib.colormaps["tab10"]
    viewpoints = [
        (25, 45, "Front-right"),
        (80, 0, "Top-down"),
        (15, 90, "Side"),
    ]

    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    fig = plt.figure(figsize=(6 * len(viewpoints), 5))

    for vi, (elev, azim, vp_label) in enumerate(viewpoints):
        ax = fig.add_subplot(1, len(viewpoints), vi + 1, projection="3d")

        for ci, obj_idx in enumerate(obj_indices):
            data = tracks_3d[obj_idx]  # (T, P, 3)
            base_color = cmap(ci % 10)
            T, P = data.shape[0], data.shape[1]

            for p_idx in range(P):
                pts = data[:, p_idx, :]  # (T, 3)
                is_centroid = (p_idx == 0)

                line_alpha_base = 1.0 if is_centroid else 0.4
                lw = 2.0 if is_centroid else 0.8

                # Build all segments + colors at once, add as Line3DCollection
                if T > 1:
                    segments = [pts[t:t + 2] for t in range(T - 1)]
                    time_fracs = np.arange(T - 1) / max(T - 2, 1)
                    alphas = line_alpha_base * (0.3 + 0.7 * time_fracs)
                    seg_colors = [(*base_color[:3], a) for a in alphas]
                    lc = Line3DCollection(segments, colors=seg_colors, linewidths=lw)
                    ax.add_collection3d(lc)

                # Keyframe markers (centroid only to avoid clutter)
                if is_centroid:
                    kf_mask = np.array([fi in keyframe_set for fi in all_frame_indices[:T]])
                    if kf_mask.any():
                        kf_pts = pts[kf_mask]
                        ax.scatter(
                            kf_pts[:, 0], kf_pts[:, 1], kf_pts[:, 2],
                            c=[base_color], s=50, marker="o",
                            edgecolors="white", linewidths=0.5,
                            label=f"Obj {obj_idx}" if vi == 0 else None,
                            zorder=5,
                        )

                    # Start and end markers
                    ax.scatter(
                        *pts[0], c=[base_color], s=80, marker="^",
                        edgecolors="black", linewidths=0.5, zorder=6,
                    )
                    ax.scatter(
                        *pts[-1], c=[base_color], s=80, marker="s",
                        edgecolors="black", linewidths=0.5, zorder=6,
                    )

        # Auto-scale axes from data (Line3DCollection doesn't auto-update limits)
        all_pts = np.concatenate([tracks_3d[oi] for oi in obj_indices if oi in tracks_3d])
        for setter, dim in [(ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)]:
            lo, hi = all_pts[..., dim].min(), all_pts[..., dim].max()
            margin = max(0.05 * (hi - lo), 1e-3)
            setter(lo - margin, hi + margin)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(vp_label, fontsize=10)

    if obj_indices:
        fig.axes[0].legend(fontsize=8, loc="upper left")

    fig.suptitle(
        f"3D Object Trajectories ({len(all_frame_indices)} frames)",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    save_figure(fig, output_path)
    print(f"  Saved 3D trajectory plot to {output_path}")


def visualize_pose_axes_3d(
    pose_data: "Dict[int, Dict[str, np.ndarray]]",
    all_frame_indices: "List[int]",
    output_path: str,
    keyframe_indices: "Optional[List[int]]" = None,
) -> None:
    """
    Save a 3D plot showing per-object coordinate axes and centroid trajectory.

    At each keyframe, draws RGB arrows (X=red, Y=green, Z=blue) from the
    object centroid showing the local frame orientation.  The centroid
    trajectory is drawn as a time-colored line.

    Parameters
    ----------
    pose_data : dict
        ``{obj_idx: {"centroids": (T, 3), "axes": (T, 3, 3)}}`` in R3 coords.
        Output of :func:`compute_pose_axes`.
    all_frame_indices : list of int
        Frame indices (T dimension).
    output_path : str
        Path to save PNG.
    keyframe_indices : list of int, optional
        Frames at which to draw the axes arrows.  If None, draws at every frame.
    """
    keyframe_set = set(keyframe_indices) if keyframe_indices else set(all_frame_indices)
    obj_indices = sorted(pose_data.keys())
    axis_colors = [(1, 0, 0), (0, 0.7, 0), (0, 0, 1)]  # R, G, B
    axis_labels = ["X", "Y", "Z"]

    cmap = matplotlib.colormaps["tab10"]
    viewpoints = [
        (25, 45, "Front-right"),
        (80, 0, "Top-down"),
        (15, 90, "Side"),
    ]

    fig = plt.figure(figsize=(6 * len(viewpoints), 5))

    for vi, (elev, azim, vp_label) in enumerate(viewpoints):
        ax = fig.add_subplot(1, len(viewpoints), vi + 1, projection="3d")

        for ci, obj_idx in enumerate(obj_indices):
            centroids = pose_data[obj_idx]["centroids"]  # (T, 3)
            axes_tips = pose_data[obj_idx]["axes"]        # (T, 3, 3)
            base_color = cmap(ci % 10)
            T = centroids.shape[0]

            # Centroid trajectory (time-colored)
            for t in range(T - 1):
                frac = t / max(T - 1, 1)
                alpha = 0.3 + 0.7 * frac
                ax.plot(
                    centroids[t:t + 2, 0],
                    centroids[t:t + 2, 1],
                    centroids[t:t + 2, 2],
                    color=(*base_color[:3], alpha), linewidth=2.0,
                )

            # Draw axes arrows at keyframes
            for t, fi in enumerate(all_frame_indices):
                if fi not in keyframe_set:
                    continue
                c = centroids[t]
                for axis_i in range(3):
                    tip = axes_tips[t, axis_i]
                    dx, dy, dz = tip - c
                    ax.quiver(
                        c[0], c[1], c[2], dx, dy, dz,
                        color=axis_colors[axis_i], arrow_length_ratio=0.2,
                        linewidth=1.5,
                    )

            # Start / end markers on centroid
            ax.scatter(
                *centroids[0], c=[base_color], s=80, marker="^",
                edgecolors="black", linewidths=0.5, zorder=6,
            )
            ax.scatter(
                *centroids[-1], c=[base_color], s=80, marker="s",
                edgecolors="black", linewidths=0.5, zorder=6,
            )

            if vi == 0:
                ax.plot([], [], color=base_color, linewidth=2, label=f"Obj {obj_idx}")

        # Legend for axis colors (only on first viewpoint)
        if vi == 0:
            for axis_i in range(3):
                ax.plot([], [], color=axis_colors[axis_i], linewidth=2,
                        label=f"{axis_labels[axis_i]} axis")
            ax.legend(fontsize=7, loc="upper left")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(vp_label, fontsize=10)

    fig.suptitle(
        f"Object Reference Frames ({len(all_frame_indices)} frames, "
        f"axes at {len(keyframe_set)} keyframes)",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    save_figure(fig, output_path)
    print(f"  Saved 3D pose axes plot to {output_path}")


def draw_pose_axes_on_image(
    image: "torch.Tensor",
    pose_axes_data: "Dict[int, Dict[str, np.ndarray]]",
    t_idx: int,
    K_matrix: np.ndarray,
    c2w: "Optional[np.ndarray]" = None,
) -> np.ndarray:
    """Overlay projected 2D pose axes on a rendered image.

    Draws per-object centroid circles and RGB axis arrows
    (X=red, Y=green, Z=blue) on the image.

    Parameters
    ----------
    image : torch.Tensor
        Rendered image (H, W, 3) in [0, 1].
    pose_axes_data : dict
        ``{obj_idx: {"centroids": (T, 3), "axes": (T, 3, 3)}}`` from
        :func:`~genia.core.utils.interpolation.compute_pose_axes`.
        Points are in R3 **camera-space** coordinates.
    t_idx : int
        Index into the T dimension for this frame.
    K_matrix : np.ndarray
        Camera intrinsics (3, 3).
    c2w : np.ndarray (4, 4), optional
        Camera-to-world transform.  When provided, world-space points
        are transformed to camera space before projection.
        ``None`` = identity (camera at world origin).

    Returns
    -------
    np.ndarray
        Image with axes overlay (H, W, 3) float in [0, 1].
    """
    import cv2

    img_np = image.detach().cpu().float().numpy()
    canvas = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    # Pre-compute world-to-camera rotation and translation
    if c2w is not None:
        w2c = np.linalg.inv(c2w.astype(np.float64)).astype(np.float32)
        R_w2c = w2c[:3, :3]  # (3, 3)
        t_w2c = w2c[:3, 3]   # (3,)
    else:
        R_w2c = None

    # BGR axis colors: X=red, Y=green, Z=blue
    axis_colors_bgr = [(0, 0, 255), (0, 180, 0), (255, 0, 0)]

    # Per-object colors (HSV → BGR)
    obj_indices = sorted(pose_axes_data.keys())
    n_objects = max(len(obj_indices), 1)
    obj_colors_bgr = {}
    for i, obj_idx in enumerate(obj_indices):
        hue = int(180 * i / n_objects)
        hsv = np.array([[[hue, 220, 255]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        obj_colors_bgr[obj_idx] = (int(bgr[0]), int(bgr[1]), int(bgr[2]))

    def _project(pts_3d):
        """Project world-space points to pixel coords via w2c + K."""
        pts = pts_3d
        if R_w2c is not None:
            pts = pts @ R_w2c.T + t_w2c
        proj = (K_matrix @ pts.T).T
        z = proj[:, 2:3]
        z = np.where(np.abs(z) < 1e-6, 1e-6, z)
        return proj[:, :2] / z

    for obj_idx in obj_indices:
        centroids_3d = pose_axes_data[obj_idx]["centroids"]  # (T, 3)
        axes_3d = pose_axes_data[obj_idx]["axes"]              # (T, 3, 3)

        c2d = _project(centroids_3d[t_idx:t_idx+1, :])[0]   # (2,)
        cx, cy = int(c2d[0]), int(c2d[1])

        # Centroid circle
        color = obj_colors_bgr[obj_idx]
        cv2.circle(canvas, (cx, cy), 4, color, -1, lineType=cv2.LINE_AA)

        # Axis arrows
        tips_3d = axes_3d[t_idx]  # (3, 3)
        tips_2d = _project(tips_3d)  # (3, 2)
        for axis_i in range(3):
            tip_x, tip_y = int(tips_2d[axis_i, 0]), int(tips_2d[axis_i, 1])
            cv2.arrowedLine(
                canvas, (cx, cy), (tip_x, tip_y),
                axis_colors_bgr[axis_i], thickness=2, tipLength=0.25,
                line_type=cv2.LINE_AA,
            )

    result = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return result


def visualize_pose_axes_2d(
    pose_data: "Dict[int, Dict[str, np.ndarray]]",
    all_frame_indices: "List[int]",
    K_per_frame: "Dict[int, np.ndarray]",
    W: int,
    H: int,
    output_path: str,
    rendered_frames: "Optional[List[np.ndarray]]" = None,
    keyframe_indices: "Optional[List[int]]" = None,
    trail_length: int = 16,
    duration: int = 100,
    c2w_per_frame: "Optional[Dict[int, np.ndarray]]" = None,
) -> None:
    """
    Create a video with projected 2D coordinate axes and centroid trajectory.

    Each frame shows per-object RGB arrows (X=red, Y=green, Z=blue)
    projected from 3D, plus a fading centroid trail.

    Parameters
    ----------
    pose_data : dict
        ``{obj_idx: {"centroids": (T, 3), "axes": (T, 3, 3)}}`` in R3 coords.
    all_frame_indices : list of int
        Frame indices (T dimension).
    K_per_frame : dict
        ``{frame_idx: np.ndarray(3,3)}`` per-frame camera intrinsics.
    W, H : int
        Image dimensions.
    output_path : str
        Path to save video.
    rendered_frames : list of ndarray, optional
        Background frames (H, W, 3) uint8 BGR.  If None, uses dark gray.
    keyframe_indices : list of int, optional
        Frames at which to highlight with ring markers.
    trail_length : int
        Number of past frames for the centroid trail.
    duration : int
        Frame duration in milliseconds.
    """
    import cv2
    from PIL import Image

    keyframe_set = set(keyframe_indices) if keyframe_indices else set()
    obj_indices = sorted(pose_data.keys())
    # RGB colors: X=red, Y=green, Z=blue
    axis_colors = [(255, 0, 0), (0, 180, 0), (0, 0, 255)]

    # Per-object base colors (HSV → RGB)
    n_objects = max(len(obj_indices), 1)
    obj_colors = {}
    for i, obj_idx in enumerate(obj_indices):
        hue = int(180 * i / n_objects)
        hsv = np.array([[[hue, 220, 255]]], dtype=np.uint8)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
        obj_colors[obj_idx] = (int(rgb[0]), int(rgb[1]), int(rgb[2]))

    def _project(pts_3d: np.ndarray, K: np.ndarray,
                 c2w: "Optional[np.ndarray]" = None) -> np.ndarray:
        """Project (N, 3) R3 world-space points to (N, 2) pixel coords."""
        pts = pts_3d
        if c2w is not None:
            w2c = np.linalg.inv(c2w.astype(np.float64)).astype(np.float32)
            pts = pts @ w2c[:3, :3].T + w2c[:3, 3]
        proj = (K @ pts.T).T  # (N, 3)
        z = proj[:, 2:3]
        z = np.where(np.abs(z) < 1e-6, 1e-6, z)
        return proj[:, :2] / z  # (N, 2)

    # Pre-project all centroids and axis tips per frame
    projected: Dict[int, Dict[str, np.ndarray]] = {}
    for obj_idx in obj_indices:
        centroids_3d = pose_data[obj_idx]["centroids"]   # (T, 3)
        axes_3d = pose_data[obj_idx]["axes"]              # (T, 3, 3)
        T = centroids_3d.shape[0]

        centroids_2d = np.zeros((T, 2), dtype=np.float64)
        axes_2d = np.zeros((T, 3, 2), dtype=np.float64)
        for t_idx, fi in enumerate(all_frame_indices):
            K = K_per_frame[fi]
            _c2w = c2w_per_frame[fi] if c2w_per_frame else None
            centroids_2d[t_idx] = _project(centroids_3d[t_idx:t_idx+1], K, _c2w)[0]
            axes_2d[t_idx] = _project(axes_3d[t_idx], K, _c2w)  # (3, 3) → (3, 2)

        projected[obj_idx] = {"centroids": centroids_2d, "axes": axes_2d}

    frames = []
    for t_idx, fi in enumerate(all_frame_indices):
        if rendered_frames is not None and t_idx < len(rendered_frames):
            canvas = rendered_frames[t_idx].copy()
        else:
            canvas = np.full((H, W, 3), 40, dtype=np.uint8)
        is_keyframe = fi in keyframe_set

        for obj_idx in obj_indices:
            c2d = projected[obj_idx]["centroids"]  # (T, 2)
            a2d = projected[obj_idx]["axes"]        # (T, 3, 2)
            color = obj_colors[obj_idx]

            # Centroid trail
            start = max(0, t_idx - trail_length)
            for j in range(start, t_idx):
                alpha = (j - start + 1) / (t_idx - start + 1)
                pt1 = (int(c2d[j, 0]), int(c2d[j, 1]))
                pt2 = (int(c2d[j + 1, 0]), int(c2d[j + 1, 1]))
                overlay = canvas.copy()
                cv2.line(overlay, pt1, pt2, color, thickness=2, lineType=cv2.LINE_AA)
                cv2.addWeighted(overlay, alpha * 0.6, canvas, 1 - alpha * 0.6, 0, canvas)

            # Current centroid
            cx, cy = int(c2d[t_idx, 0]), int(c2d[t_idx, 1])
            radius = 6 if is_keyframe else 4
            cv2.circle(canvas, (cx, cy), radius, color, -1, lineType=cv2.LINE_AA)
            if is_keyframe:
                cv2.circle(canvas, (cx, cy), radius + 2, (255, 255, 255), 1, lineType=cv2.LINE_AA)

            # Axis arrows (always drawn)
            for axis_i in range(3):
                tip_x, tip_y = int(a2d[t_idx, axis_i, 0]), int(a2d[t_idx, axis_i, 1])
                cv2.arrowedLine(
                    canvas, (cx, cy), (tip_x, tip_y),
                    axis_colors[axis_i], thickness=2, tipLength=0.25,
                    line_type=cv2.LINE_AA,
                )

        draw_text_overlay(canvas, f"Frame {fi}", (12, 30), font_scale=0.65)
        frames.append(Image.fromarray(canvas))

    if not frames:
        return

    from .interpolation import _save_frames_as_video
    _save_frames_as_video(frames, output_path, duration=duration)


def save_and_plot_loss_history(
    tokens_by_object,
    suffix,
    output_dir,
    scene_name,
    save_json=True,
    save_plot=True,
):
    """Extract refinement loss history from tokens, save as JSON, and plot as PNG.

    Parameters
    ----------
    tokens_by_object : dict
        Dictionary mapping obj_idx -> list of (frame_idx, decoder_input).
    suffix : str
        Label for the refinement stage (e.g., "perframe", "global").
    output_dir : str
        Directory to write JSON and PNG files.
    scene_name : str
        Scene name for filenames.
    save_json : bool, optional
        Whether to save the JSON file. Default: True.
    save_plot : bool, optional
        Whether to save the PNG plot. Default: True.
    """
    if not save_json and not save_plot:
        return
    refinement_data = {
        'objects': {}
    }
    has_history = False
    for obj_idx, tokens_list in tokens_by_object.items():
        # JSON keys must be str/int — FrameKey namedtuple isn't serialisable.
        refinement_data['objects'][str(obj_idx)] = {}
        obj_data = refinement_data['objects'][str(obj_idx)]
        for frame_idx, decoder_input in tokens_list:
            if 'refinement_loss_history' in decoder_input:
                has_history = True
                frame_entry = {
                    'loss_history': decoder_input['refinement_loss_history'],
                    'best_iteration': decoder_input['refinement_best_iteration'],
                }
                if 'refinement_batch_loss_history' in decoder_input:
                    frame_entry['batch_loss_history'] = decoder_input['refinement_batch_loss_history']
                obj_data[str(frame_idx)] = frame_entry
    if not has_history:
        return

    if save_json:
        json_path = os.path.join(
            output_dir,
            f"{scene_name}_{suffix}_refinement_history.json"
        )
        with open(json_path, 'w') as f:
            json.dump(refinement_data, f, indent=2)
        print(f"\nSaved {suffix} refinement loss history to {json_path}")

    if save_plot:
        plot_path = os.path.join(
            output_dir,
            f"{scene_name}_{suffix}_refinement_history.png"
        )
        plot_refinement_history(refinement_data, plot_path)


def _save_pixelwise_grid(
    pixelwise_data: Dict[int, Dict[str, np.ndarray]],
    frame_indices: List[int],
    columns: List[Tuple[str, str, str]],
    title: str,
    save_path: str,
) -> None:
    """Save a rows-by-columns image grid.

    Parameters
    ----------
    columns : list of (data_key, title, mode)
        *mode* is ``"heatmap"``, ``"gray"``, or ``"rgb"``.
    """
    n_rows = len(frame_indices)
    n_cols = len(columns)
    if n_rows == 0 or n_cols == 0:
        return

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.5 * n_cols, 3.0 * n_rows),
        squeeze=False,
    )

    for row, fi in enumerate(frame_indices):
        fdata = pixelwise_data[fi]
        for col, (key, col_title, mode) in enumerate(columns):
            ax = axes[row, col]
            data = fdata.get(key)

            if data is None:
                ax.axis("off")
                if row == 0:
                    ax.set_title(col_title, fontsize=9)
                continue

            if mode == "heatmap":
                im = ax.imshow(data, cmap="viridis")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            elif mode == "gray":
                ax.imshow(data, cmap="gray", vmin=0, vmax=1)
            elif mode == "normals":
                # Map normal xyz ∈ [-1, 1] to RGB ∈ [0, 1]
                ax.imshow(np.clip(data * 0.5 + 0.5, 0, 1))
            else:  # rgb
                ax.imshow(np.clip(data, 0, 1))

            if row == 0:
                ax.set_title(col_title, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"Frame {fi}", fontsize=9)
            h, w = data.shape[:2]
            ax.set_xlabel(f"{w}×{h}", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, save_path, dpi=120)


def save_pixelwise_loss_plot(
    pixelwise_data: Dict[int, Dict[str, np.ndarray]],
    iteration: int,
    obj_idx: int,
    output_dir: str,
) -> None:
    """Save per-pixel debug grids for finetuning.

    Saves three separate plots into ``debug_pixelwise/``:

    - **errors** — RGB error, depth error, normals error, alpha heatmaps
    - **rendered** — rendered RGB per frame
    - **gt** — ground-truth RGB per frame

    Columns whose data is all-None across frames are skipped.

    Parameters
    ----------
    pixelwise_data : dict
        ``{frame_idx: {"rendered_rgb": (H,W,3), "gt_rgb": (H,W,3),
        "mask": (H,W), "px_rgb_error": (H,W), ...}}``
    iteration : int
        Current optimization iteration number.
    obj_idx : int
        Object index.
    output_dir : str
        Root output directory.
    """
    import matplotlib
    matplotlib.use("Agg")

    if not pixelwise_data:
        return

    frame_indices = sorted(pixelwise_data.keys())
    save_dir = os.path.join(output_dir, "debug_pixelwise")
    os.makedirs(save_dir, exist_ok=True)
    prefix = f"iter_{iteration:04d}_obj_{obj_idx:03d}"

    def _has(key: str) -> bool:
        return any(pixelwise_data[fi].get(key) is not None for fi in frame_indices)

    # --- 1. Error maps (gauss + mesh side-by-side when both present) ---
    error_columns = []
    for key, title, mode in [
        ("px_rgb_error", "Gauss RGB Err (L1)", "heatmap"),
        ("mesh_px_rgb_error", "Mesh RGB Err (L1)", "heatmap"),
        ("px_depth_error", "Gauss Depth Err", "heatmap"),
        ("mesh_px_depth_error", "Mesh Depth Err", "heatmap"),
        ("px_normals_error", "Gauss Normals Err", "heatmap"),
        ("mesh_px_normals_error", "Mesh Normals Err", "heatmap"),
    ]:
        if _has(key):
            error_columns.append((key, title, mode))

    if error_columns:
        _save_pixelwise_grid(
            pixelwise_data, frame_indices, error_columns,
            f"Per-pixel losses — iter {iteration}, obj {obj_idx}",
            os.path.join(save_dir, f"{prefix}_errors.png"),
        )

    # --- 2. Rendered RGB + Alpha + Depth + Normals (gauss + mesh) ---
    render_columns = [("rendered_rgb", "Gauss RGB", "rgb")]
    if _has("mesh_rendered_rgb"):
        render_columns.append(("mesh_rendered_rgb", "Mesh RGB", "rgb"))
    if _has("px_alpha"):
        render_columns.append(("px_alpha", "Gauss Alpha", "gray"))
    if _has("mesh_px_alpha"):
        render_columns.append(("mesh_px_alpha", "Mesh Alpha", "gray"))
    if _has("rendered_depth"):
        render_columns.append(("rendered_depth", "Gauss Depth", "heatmap"))
    if _has("mesh_rendered_depth"):
        render_columns.append(("mesh_rendered_depth", "Mesh Depth", "heatmap"))
    if _has("rendered_normals"):
        render_columns.append(("rendered_normals", "Gauss Normals", "normals"))
    if _has("mesh_rendered_normals"):
        render_columns.append(("mesh_rendered_normals", "Mesh Normals", "normals"))
    if _has("rendered_rgb"):
        _save_pixelwise_grid(
            pixelwise_data, frame_indices, render_columns,
            f"Rendered — iter {iteration}, obj {obj_idx}",
            os.path.join(save_dir, f"{prefix}_rendered.png"),
        )

    # --- 3. GT RGB + Depth + Normals ---
    gt_columns = [("gt_rgb", "GT RGB", "rgb")]
    if _has("gt_depth"):
        gt_columns.append(("gt_depth", "GT Depth", "heatmap"))
    if _has("gt_normals"):
        gt_columns.append(("gt_normals", "GT Normals", "normals"))
    if _has("gt_rgb"):
        _save_pixelwise_grid(
            pixelwise_data, frame_indices, gt_columns,
            f"Ground Truth — iter {iteration}, obj {obj_idx}",
            os.path.join(save_dir, f"{prefix}_gt.png"),
        )


# =====================================================================
# Voxel-mesh rendering through a camera
# =====================================================================


def render_voxel_mesh_in_camera(
    voxel_coords_np,
    K_matrix,
    W,
    H,
    obj_rotation,
    obj_translation,
    obj_scale,
    c2w=None,
    grid_size=64,
    voxel_colors=None,
    shading=False,
):
    """Render voxel cubes in world space through a pinhole camera.

    Transforms voxel cubes by the object pose (PyTorch3D convention:
    ``world = local * scale @ R + t``), then renders through the camera.

    Parameters
    ----------
    voxel_coords_np : np.ndarray (N, 3)
        Integer grid coords of occupied voxels (values in [0, grid_size-1]).
    K_matrix : np.ndarray (3, 3)
        Camera intrinsics.
    W, H : int
        Image width, height.
    obj_rotation : np.ndarray (3, 3)
        Rotation matrix (PyTorch3D convention: ``points @ R``).
    obj_translation : np.ndarray (3,)
        Translation vector.
    obj_scale : np.ndarray (3,) or float
        Scale factor(s).
    c2w : np.ndarray (4, 4) or None
        Camera-to-world transform.  None = identity.
    grid_size : int
        Voxel grid resolution (default 64).

    Returns
    -------
    PIL.Image
        Rendered RGB image.
    """
    import torch
    from PIL import Image as PILImage

    from pytorch3d.renderer import (
        BlendParams, MeshRasterizer, PerspectiveCameras,
        RasterizationSettings, TexturesVertex,
    )
    from pytorch3d.renderer.blending import hard_rgb_blend
    from pytorch3d.structures import Meshes

    N = voxel_coords_np.shape[0]
    if N == 0:
        # Return white image
        return PILImage.new("RGB", (W, H), (255, 255, 255))

    voxel_size = 1.0 / grid_size
    obj_scale = np.atleast_1d(np.asarray(obj_scale, dtype=np.float32))
    if obj_scale.shape[0] == 1:
        obj_scale = np.broadcast_to(obj_scale, (3,))

    # Unit cube vertices centered at origin
    cube_v = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float32) - 0.5

    cube_f = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
        [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
    ], dtype=np.int64)

    # Vectorized: build all N*8 vertices and N*12 faces at once
    # local_pos: (N, 3) — voxel center in local space [-0.5, 0.5]
    local_pos = (voxel_coords_np + 0.5) / grid_size - 0.5  # (N, 3)

    # Tile cube template for all voxels: (N, 8, 3)
    local_verts = cube_v[np.newaxis, :, :] * voxel_size + local_pos[:, np.newaxis, :]

    # PyTorch3D convention: posed = local * scale @ R + t  (P3D camera space)
    posed_verts = (local_verts * obj_scale) @ obj_rotation + obj_translation

    # P3D → R3: negate X and Y
    posed_verts[..., :2] *= -1
    all_verts = posed_verts.reshape(N * 8, 3)

    # Faces: offset each cube's indices by its vertex base
    offsets = (np.arange(N) * 8)[:, np.newaxis, np.newaxis]  # (N, 1, 1)
    all_faces = (cube_f[np.newaxis, :, :] + offsets).reshape(N * 12, 3)

    # Colors: custom or XYZ from voxel position [0, 1], broadcast to 8 verts
    if voxel_colors is not None:
        color_per_voxel = np.asarray(voxel_colors, dtype=np.float32)[:N]
    else:
        color_per_voxel = (voxel_coords_np + 0.5) / grid_size  # (N, 3)
    all_colors = np.repeat(color_per_voxel, 8, axis=0)  # (N*8, 3)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Verts are now in R3 camera space. Apply c2w to get world space,
    # then w2c to get rendering camera space.
    if c2w is None:
        c2w = np.eye(4, dtype=np.float32)
    # cam→world: p_world = p_cam @ R_c2w^T + t_c2w
    R_c2w = c2w[:3, :3].astype(np.float32)
    t_c2w = c2w[:3, 3].astype(np.float32)
    world_verts = all_verts @ R_c2w.T + t_c2w
    # world→render cam: w2c
    w2c = np.linalg.inv(c2w).astype(np.float32)
    R_w2c = w2c[:3, :3]
    T_w2c = w2c[:3, 3]
    cam_verts = world_verts @ R_w2c.T + T_w2c

    fx, fy = K_matrix[0, 0], K_matrix[1, 1]
    cx, cy = K_matrix[0, 2], K_matrix[1, 2]

    # Verts are in R3 camera space (X-right, Y-down, Z-forward).
    # PyTorch3D PerspectiveCameras with in_ndc=False uses
    # x_screen = -fx * X/Z + cx (internal X/Y negation for P3D convention).
    # Negate focal lengths to cancel this, giving standard R3 projection:
    # x_screen = fx * X/Z + cx.

    verts_t = torch.from_numpy(cam_verts).float().to(device)
    faces_t = torch.from_numpy(all_faces).long().to(device)
    colors_t = torch.from_numpy(all_colors).float().to(device)

    mesh = Meshes(
        verts=[verts_t], faces=[faces_t],
        textures=TexturesVertex(verts_features=[colors_t]),
    )

    cameras = PerspectiveCameras(
        focal_length=torch.tensor([[-fx, -fy]]).float(),
        principal_point=torch.tensor([[cx, cy]]).float(),
        R=torch.eye(3).unsqueeze(0).float(),
        T=torch.zeros(1, 3).float(),
        image_size=torch.tensor([[H, W]]),
        in_ndc=False,
        device=device,
    )

    blend_params = BlendParams(
        sigma=1e-4, gamma=1e-4, background_color=(1.0, 1.0, 1.0),
    )
    raster_settings = RasterizationSettings(
        image_size=(H, W), blur_radius=0.0, faces_per_pixel=1, bin_size=0,
    )

    rasterizer = MeshRasterizer(
        cameras=cameras, raster_settings=raster_settings,
    )

    with torch.no_grad():
        fragments = rasterizer(mesh)
        if shading:
            # Flat per-face Phong shading with a camera-collocated light, so the
            # voxel cubes read as 3-D (top faces brighter, sides darker) instead
            # of flat colour.  Mirrors render_slat_voxel_mesh(shading=True).
            from pytorch3d.renderer import HardFlatShader, PointLights
            lights = PointLights(
                device=device, location=cameras.get_camera_center(),
                ambient_color=((0.6, 0.6, 0.6),),
                diffuse_color=((0.4, 0.4, 0.4),),
                specular_color=((0.1, 0.1, 0.1),),
            )
            shader = HardFlatShader(
                device=device, cameras=cameras, lights=lights,
                blend_params=blend_params,
            )
            img = shader(fragments, mesh)[0, ..., :3]
        else:
            texels = mesh.sample_textures(fragments)
            img = hard_rgb_blend(texels, fragments, blend_params)
            img = img[0, ..., :3]

    img_np = torch.clamp(img, 0.0, 1.0).cpu().numpy()
    return PILImage.fromarray((img_np * 255).astype(np.uint8))


def _build_voxel_surface_mesh(voxel_coords_np, grid_size=64,
                              voxel_colors=None,
                              return_face_to_voxel=False):
    """Build a surface-only mesh from a voxel grid (vectorized).

    Only emits quad faces that border an empty voxel, dramatically reducing
    triangle count for solid objects (~95% fewer faces than full cubes).

    Parameters
    ----------
    voxel_coords_np : np.ndarray (N, 3)
        Integer voxel coordinates in the grid.
    grid_size : int
        Voxel grid resolution (default 64).
    voxel_colors : np.ndarray (N, 3), optional
        Per-voxel colors in [0, 1].  When ``None``, defaults to
        normalized XYZ grid position (R=X, G=Y, B=Z).
    return_face_to_voxel : bool
        If True, return a 4th element: ``face_to_voxel`` array ``(F,)``
        mapping each triangle index to its source voxel index (into the
        input ``voxel_coords_np``).

    Returns
    -------
    local_verts : np.ndarray (V, 3)
        Vertex positions in normalized local space ([-0.5, 0.5]).
    faces : np.ndarray (F, 3)
        Triangle indices.
    vert_colors : np.ndarray (V, 3)
        Per-vertex colors.
    face_to_voxel : np.ndarray (F,) int64  *(only when return_face_to_voxel=True)*
        Source voxel index for each triangle.
    """
    N = len(voxel_coords_np)
    _empty = (
        np.zeros((0, 3), dtype=np.float32),
        np.zeros((0, 3), dtype=np.int64),
        np.zeros((0, 3), dtype=np.float32),
    )
    if N == 0:
        return (*_empty, np.zeros(0, dtype=np.int64)) if return_face_to_voxel else _empty

    voxel_size = 1.0 / grid_size

    # Build 3D occupancy grid for O(1) neighbor lookups.  Inputs may be
    # warped float coords (per-vertex KNN-IDW deformation field) that drift
    # past [0, grid_size-1] after large rotations/translations — clip after
    # int cast so the +1 padding offset + ±1 neighbor lookup stays in bounds.
    occ_grid = np.zeros((grid_size + 2, grid_size + 2, grid_size + 2), dtype=bool)
    coords_i = np.clip(
        voxel_coords_np.astype(np.intp), 0, grid_size - 1,
    )
    occ_grid[coords_i[:, 0] + 1, coords_i[:, 1] + 1, coords_i[:, 2] + 1] = True

    # 6 face directions with cube corner indices for each face quad
    #   Cube corners:  0=(0,0,0) 1=(1,0,0) 2=(1,1,0) 3=(0,1,0)
    #                  4=(0,0,1) 5=(1,0,1) 6=(1,1,1) 7=(0,1,1)
    neighbor_offsets = np.array([
        [-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1],
    ], dtype=np.intp)
    face_quad_verts = np.array([
        [0, 4, 7, 3],  # -X
        [1, 2, 6, 5],  # +X
        [0, 1, 5, 4],  # -Y
        [2, 3, 7, 6],  # +Y
        [0, 3, 2, 1],  # -Z
        [4, 5, 6, 7],  # +Z
    ], dtype=np.intp)

    cube_corners = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float32) - 0.5

    # Voxel centers in normalized local space
    centers = (coords_i.astype(np.float32) + 0.5) / grid_size - 0.5  # (N, 3)
    if voxel_colors is not None:
        colors_per_voxel = np.asarray(voxel_colors, dtype=np.float32)
    else:
        colors_per_voxel = (coords_i.astype(np.float32) + 0.5) / grid_size  # (N, 3)

    all_verts = []
    all_faces = []
    all_colors = []
    all_face_to_voxel = []
    vert_offset = 0

    for face_idx in range(6):
        dx, dy, dz = neighbor_offsets[face_idx]
        # Check which voxels have an empty neighbor in this direction
        # (+1 offset for the padding in occ_grid)
        neighbor_occ = occ_grid[
            coords_i[:, 0] + 1 + dx,
            coords_i[:, 1] + 1 + dy,
            coords_i[:, 2] + 1 + dz,
        ]
        exposed = ~neighbor_occ  # (N,) bool mask
        n_exposed = exposed.sum()
        if n_exposed == 0:
            continue

        # Get the 4 corner offsets for this face
        quad = face_quad_verts[face_idx]  # (4,)
        corners = cube_corners[quad]  # (4, 3)

        # Build vertices: (n_exposed, 4, 3)
        fv = corners[np.newaxis, :, :] * voxel_size + centers[exposed, np.newaxis, :]
        all_verts.append(fv.reshape(n_exposed * 4, 3))

        # Build faces: 2 triangles per quad
        base = np.arange(n_exposed, dtype=np.int64) * 4 + vert_offset
        tri0 = np.stack([base, base + 1, base + 2], axis=1)
        tri1 = np.stack([base, base + 2, base + 3], axis=1)
        all_faces.append(np.concatenate([tri0, tri1], axis=0))

        # Colors: repeat per-voxel color to 4 vertices
        c = colors_per_voxel[exposed]  # (n_exposed, 3)
        all_colors.append(np.repeat(c, 4, axis=0))

        # Face → voxel mapping: 2 triangles per exposed voxel.
        # Faces are ordered [all tri0, all tri1] via concat([tri0, tri1]),
        # so f2v must also be [exposed_idx, exposed_idx] (tile, not repeat).
        if return_face_to_voxel:
            exposed_idx = np.where(exposed)[0]  # (n_exposed,)
            all_face_to_voxel.append(np.tile(exposed_idx, 2))

        vert_offset += n_exposed * 4

    if not all_verts:
        return (*_empty, np.zeros(0, dtype=np.int64)) if return_face_to_voxel else _empty

    result = (
        np.concatenate(all_verts, axis=0),
        np.concatenate(all_faces, axis=0),
        np.concatenate(all_colors, axis=0),
    )
    if return_face_to_voxel:
        return (*result, np.concatenate(all_face_to_voxel, axis=0))
    return result


def render_voxel_meshes_in_camera(
    objects,
    K_matrix,
    W,
    H,
    c2w=None,
    render_c2w=None,
    grid_size=64,
):
    """Render multiple objects' voxel cubes in a single pass with correct depth.

    Uses surface-only meshing: only faces bordering empty voxels are emitted,
    reducing triangle count by ~80-90% for solid objects.  Callers rendering
    many frames should pre-compute the local mesh via
    ``_build_voxel_surface_mesh`` and pass it through ``_local_mesh`` in the
    object dict to avoid rebuilding it every frame.

    Parameters
    ----------
    objects : list of dict
        Each dict has keys: ``voxel_coords_np`` (N, 3), ``obj_rotation`` (3, 3),
        ``obj_translation`` (3,), ``obj_scale`` (3,) or float.
        Optional: ``_local_mesh`` tuple of (local_verts, faces, vert_colors)
        from ``_build_voxel_surface_mesh`` to skip recomputation.
    K_matrix : np.ndarray (3, 3)
        Camera intrinsics.
    W, H : int
        Image width, height.
    c2w : np.ndarray (4, 4) or None
        Camera-to-world transform for the frame whose camera space the
        object poses are defined in.  None = identity.
    render_c2w : np.ndarray (4, 4) or None
        Camera-to-world transform for the camera to render from.
        None = same as ``c2w`` (render from the frame's own camera).
    grid_size : int
        Voxel grid resolution (default 64).

    Returns
    -------
    PIL.Image
        Rendered RGB image with all objects depth-ordered correctly.
    """
    import torch
    from PIL import Image as PILImage

    from pytorch3d.renderer import (
        BlendParams, MeshRasterizer, PerspectiveCameras,
        RasterizationSettings, TexturesVertex,
    )
    from pytorch3d.renderer.blending import hard_rgb_blend
    from pytorch3d.structures import Meshes

    if not objects:
        return PILImage.new("RGB", (W, H), (255, 255, 255))

    if c2w is None:
        c2w = np.eye(4, dtype=np.float32)
    R_c2w = c2w[:3, :3].astype(np.float32)
    t_c2w = c2w[:3, 3].astype(np.float32)
    # Render camera: defaults to the same camera as the frame
    if render_c2w is None:
        render_c2w = c2w
    w2c_render = np.linalg.inv(render_c2w.astype(np.float64)).astype(np.float32)
    R_w2c = w2c_render[:3, :3]
    T_w2c = w2c_render[:3, 3]

    all_cam_verts_list = []
    all_faces_list = []
    all_colors_list = []
    vert_offset = 0

    for obj in objects:
        obj_rotation = obj["obj_rotation"]
        obj_translation = obj["obj_translation"]
        obj_scale = np.atleast_1d(np.asarray(obj["obj_scale"], dtype=np.float32))
        if obj_scale.shape[0] == 1:
            obj_scale = np.broadcast_to(obj_scale, (3,))

        # Use pre-computed local mesh if available, otherwise build it
        if "_local_mesh" in obj:
            local_verts, faces_local, vert_colors = obj["_local_mesh"]
        else:
            local_verts, faces_local, vert_colors = _build_voxel_surface_mesh(
                obj["voxel_coords_np"], grid_size,
                voxel_colors=obj.get("_voxel_colors"),
            )

        if len(local_verts) == 0:
            continue

        # Apply pose: v_posed = (v * scale) @ R + t, then P3D → R3
        posed_verts = (local_verts * obj_scale) @ obj_rotation + obj_translation
        posed_verts[..., :2] *= -1  # P3D → R3

        # cam→world→render cam
        world_verts = posed_verts @ R_c2w.T + t_c2w
        cam_verts = world_verts @ R_w2c.T + T_w2c

        all_cam_verts_list.append(cam_verts)
        all_faces_list.append(faces_local + vert_offset)
        all_colors_list.append(vert_colors)
        vert_offset += len(local_verts)

    if not all_cam_verts_list:
        return PILImage.new("RGB", (W, H), (255, 255, 255))

    all_cam_verts = np.concatenate(all_cam_verts_list, axis=0)
    all_faces = np.concatenate(all_faces_list, axis=0)
    all_colors = np.concatenate(all_colors_list, axis=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fx, fy = K_matrix[0, 0], K_matrix[1, 1]
    cx, cy = K_matrix[0, 2], K_matrix[1, 2]

    verts_t = torch.from_numpy(all_cam_verts).float().to(device)
    faces_t = torch.from_numpy(all_faces).long().to(device)
    colors_t = torch.from_numpy(all_colors).float().to(device)

    mesh = Meshes(
        verts=[verts_t], faces=[faces_t],
        textures=TexturesVertex(verts_features=[colors_t]),
    )

    cameras = PerspectiveCameras(
        focal_length=torch.tensor([[-fx, -fy]]).float(),
        principal_point=torch.tensor([[cx, cy]]).float(),
        R=torch.eye(3).unsqueeze(0).float(),
        T=torch.zeros(1, 3).float(),
        image_size=torch.tensor([[H, W]]),
        in_ndc=False,
        device=device,
    )

    blend_params = BlendParams(
        sigma=1e-4, gamma=1e-4, background_color=(1.0, 1.0, 1.0),
    )
    raster_settings = RasterizationSettings(
        image_size=(H, W), blur_radius=0.0, faces_per_pixel=1, bin_size=0,
    )

    rasterizer = MeshRasterizer(
        cameras=cameras, raster_settings=raster_settings,
    )

    with torch.no_grad():
        fragments = rasterizer(mesh)
        texels = mesh.sample_textures(fragments)
        img = hard_rgb_blend(texels, fragments, blend_params)
        img = img[0, ..., :3]

    img_np = torch.nan_to_num(img, nan=1.0).clamp(0.0, 1.0).cpu().numpy()
    return PILImage.fromarray((img_np * 255).astype(np.uint8))


# ── Visibility category color constants ──────────────────────────────
# 2-entry palette: matches the per-(voxel, view) visibility actually used
# by Stage-2 velocity fusion (the DDA visibility that fed Stage-2 velocity
# fusion).  The plot reflects only the source that fed fusion, never
# side-channel visibility computations.
_VISIBILITY_CATEGORY_COLORS = {
    0: np.array([0.84, 0.15, 0.16]),  # Red  — invisible
    1: np.array([0.12, 0.47, 0.71]),  # Blue — visible
}


def render_voxel_binary_visibility(
    voxel_coords_np: np.ndarray,
    visible: np.ndarray,
    K_matrix: np.ndarray,
    W: int,
    H: int,
    obj_rotation: np.ndarray,
    obj_translation: np.ndarray,
    obj_scale,
    c2w=None,
    grid_size: int = 64,
) -> np.ndarray:
    """Render voxels colored by binary visibility from the camera viewpoint.

    Uses the shared 2-entry palette in :data:`_VISIBILITY_CATEGORY_COLORS`
    (red=invisible, blue=visible).  ``visible`` is a per-voxel bool array
    -- typically a column of the per-view visibility matrix that fed
    Stage-2 velocity fusion (the DDA visibility that fed Stage-2 velocity fusion).

    Returns
    -------
    img : (H, W, 3) uint8
    """
    cats = visible.astype(int)
    colors = np.zeros((len(cats), 3), dtype=np.float32)
    for cat, rgb in _VISIBILITY_CATEGORY_COLORS.items():
        colors[cats == cat] = rgb

    pil_img = render_voxel_mesh_in_camera(
        voxel_coords_np, K_matrix, W, H,
        obj_rotation, obj_translation, obj_scale,
        c2w=c2w, grid_size=grid_size,
        voxel_colors=colors,
    )
    return np.asarray(pil_img)


__all__ = [
    "plot_refinement_history",
    "visualize_slat_voxels",
    "save_render_comparison",
    "visualize_object_tracks_2d",
    "visualize_object_tracks_3d",
    "save_and_plot_loss_history",
    "save_pixelwise_loss_plot",
    "render_voxel_mesh_in_camera",
    "render_voxel_meshes_in_camera",
    "compute_voxel_colors",
    "render_voxel_binary_visibility",
]
