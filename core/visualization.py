"""Plots and debug renders specific to our method: ODE-step renders, attention and entropy
plots, visibility snapshots, FINETUNE and rendering-guidance diagnostics.

Kept out of the shared ``core/utils/visualization.py``, whose helpers these build on.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from genia.core.utils.visualization import (
    _VISIBILITY_CATEGORY_COLORS,
    draw_pose_axes_on_image,
    render_voxel_binary_visibility,
    save_figure,
)


def _crop_around_mask(image: np.ndarray, mask: np.ndarray,
                      box_size_factor: float = 1.0,
                      padding_factor: float = 0.1) -> np.ndarray:
    """Replicate SAM3D's ``crop_around_mask_with_padding`` for visualization.

    Crops *image* around the bounding box of *mask*, pads to square,
    and adds symmetric padding — matching the preprocessing that produces
    the "cropped" DINOv2 conditioning inputs.

    The bbox centering uses ``size // 2`` (integer division) to match
    SAM3D's ``compute_mask_bbox`` exactly.  For odd ``size`` this
    produces a ``(size-1) × (size-1)`` crop that is then padded to
    square — identical to ``torchvision.transforms.functional.crop``
    followed by ``F.pad`` in the original code.

    Parameters
    ----------
    image : np.ndarray
        ``(H, W)`` or ``(H, W, C)`` array.
    mask : np.ndarray
        ``(H, W)`` boolean-like mask.
    """
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return image
    min_y, max_y = int(ys.min()), int(ys.max())
    min_x, max_x = int(xs.min()), int(xs.max())
    bbox_h = max_y - min_y
    bbox_w = max_x - min_x
    size = max(bbox_h, bbox_w, 2)
    size = int(size * box_size_factor)
    cy = (min_y + max_y) / 2
    cx = (min_x + max_x) / 2

    # Match SAM3D compute_mask_bbox: integer division for centering
    y1 = int(cy - size // 2)
    x1 = int(cx - size // 2)
    y2 = int(cy + size // 2)
    x2 = int(cx + size // 2)
    crop_h = y2 - y1
    crop_w = x2 - x1

    H, W = image.shape[:2]
    extra = image.shape[2:] if image.ndim > 2 else ()

    # Crop with zero-fill for out-of-bounds pixels
    crop = np.zeros((crop_h, crop_w) + extra, dtype=image.dtype)
    sy1, sy2 = max(0, y1), min(H, y2)
    sx1, sx2 = max(0, x1), min(W, x2)
    dy1, dx1 = sy1 - y1, sx1 - x1
    crop[dy1:dy1 + sy2 - sy1, dx1:dx1 + sx2 - sx1] = image[sy1:sy2, sx1:sx2]

    # Pad to square (matching SAM3D's F.pad step)
    max_dim = max(crop_h, crop_w)
    if crop_h != max_dim or crop_w != max_dim:
        pad_h = (max_dim - crop_h) // 2
        pad_w = (max_dim - crop_w) // 2
        square = np.zeros((max_dim, max_dim) + extra, dtype=image.dtype)
        square[pad_h:pad_h + crop_h, pad_w:pad_w + crop_w] = crop
        crop = square

    # Symmetric extension (padding_factor)
    extend = int(max_dim * padding_factor)
    if extend > 0:
        padded = np.zeros((max_dim + 2 * extend, max_dim + 2 * extend) + extra,
                          dtype=image.dtype)
        padded[extend:extend + max_dim, extend:extend + max_dim] = crop
        return padded
    return crop


def _pad_to_square(image: np.ndarray) -> np.ndarray:
    """Center-pad an image to a square, matching SAM3D's ``pad_to_square_centered``."""
    H, W = image.shape[:2]
    if H == W:
        return image
    extra = image.shape[2:] if image.ndim > 2 else ()
    max_dim = max(H, W)
    padded = np.zeros((max_dim, max_dim) + extra, dtype=image.dtype)
    ph = (max_dim - H) // 2
    pw = (max_dim - W) // 2
    padded[ph:ph + H, pw:pw + W] = image
    return padded


def _overlay_heatmap_on_bg(
    hm_grid: np.ndarray,
    bg_img: np.ndarray,
    grid_h: int,
    grid_w: int,
    cmap,
    alpha: float = 0.6,
) -> "tuple[np.ndarray, float]":
    """Overlay a patch-grid heatmap on a background image.

    Each cell is independently normalised to [0, 1] so patterns are
    visible regardless of absolute magnitude (matches the approach used
    by ``plot_attn_bias_debug``).

    Returns ``(blended_uint8, raw_max)`` where *raw_max* is the
    un-normalised peak attention value (for text annotation).
    """
    from scipy.ndimage import zoom as ndimage_zoom

    H, W = bg_img.shape[:2]
    hm_up = ndimage_zoom(hm_grid, (H / grid_h, W / grid_w), order=1)
    raw_max = float(hm_up.max())
    hm_up = hm_up / max(raw_max, 1e-10)
    hm_rgb = cmap(np.clip(hm_up, 0, 1))[:, :, :3]
    img_f = bg_img.astype(np.float32) / 255.0
    blended = (1 - alpha) * img_f + alpha * hm_rgb
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8), raw_max


def _mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Convert a boolean mask to ``(H, W, 3)`` uint8 (white on black)."""
    m = mask.astype(np.uint8) * 255
    return np.stack([m, m, m], axis=-1)


def _prepare_stream_backgrounds(
    image: np.ndarray,
    mask: np.ndarray,
    target_h: int,
    target_w: int,
) -> "dict[str, np.ndarray]":
    """Build per-stream background images for attention overlay.

    Replicates the SAM3D DINOv2 preprocessing chain (crop / pad / rembg)
    so that each stream's heatmap is overlaid on the image the model
    actually saw.

    Parameters
    ----------
    image : np.ndarray
        ``(H, W, 3)`` uint8 full-scene image.
    mask : np.ndarray
        ``(H, W)`` boolean mask for the target object.
    target_h, target_w : int
        All backgrounds are resized to this size for uniform figure cells.

    Returns
    -------
    dict
        ``{stream_name: (target_h, target_w, 3) uint8}``
    """
    from PIL import Image as _PILImage

    bool_mask = mask.astype(bool) if mask.dtype != bool else mask

    def _resize(arr, h, w, interp=_PILImage.BILINEAR):
        pil = _PILImage.fromarray(arr)
        return np.asarray(pil.resize((w, h), interp))

    # --- Cropped image (rembg: background zeroed) ---
    cropped_img = _crop_around_mask(image, bool_mask)
    cropped_mask = _crop_around_mask(mask.astype(np.uint8) * 255, bool_mask)
    # rembg: zero background pixels
    bg_mask_3ch = np.stack([cropped_mask > 127] * 3, axis=-1)
    cropped_img_rembg = cropped_img * bg_mask_3ch.astype(np.uint8)

    # --- Full image (pad to square) ---
    full_img = _pad_to_square(image)

    # --- Cropped mask as RGB ---
    cropped_mask_rgb = _mask_to_rgb(cropped_mask > 127)

    # --- Full mask as RGB ---
    full_mask_rgb = _pad_to_square(_mask_to_rgb(bool_mask))

    backgrounds = {
        "cropped_image":    _resize(cropped_img_rembg, target_h, target_w),
        "full_image":       _resize(full_img, target_h, target_w),
        "cropped_mask":     _resize(cropped_mask_rgb, target_h, target_w),
        "full_mask":        _resize(full_mask_rgb, target_h, target_w),
    }
    return backgrounds


def render_appearance_ode_steps(
    slat_snapshots,
    pipeline,
    canonical_coords,
    tokens_list,
    sequence,
    output_dir,
    scene_name,
    obj_idx,
    canonical_mesh_verts=None,
    per_frame_mesh_verts=None,
    per_frame_mesh_rotations=None,
    canonical_mesh_faces=None,
    warp_knn_k: int = 4,
    warp_knn_eps: float = 1.0e-8,
    warp_knn_chunk_size: int = 8192,
    bg_color=None,
):
    """Decode SLAT at each captured Stage-2 ODE step and render it.

    At every captured step, the un-normalised SLAT is decoded to Gaussians,
    the optional per-canonical-mesh-vertex deformation is applied via
    ``warp_gaussians_high_res``, Stage-1's per-frame ``(R, t, s)`` is
    composed, and the per-frame renders are concatenated into one row per
    step.  The rows are stacked into one grid PNG.

    Mirrors the rendering-guidance code path so the viz shows exactly what
    APPEARANCE_INIT renders during the in-ODE guidance loop.

    Parameters
    ----------
    slat_snapshots : list of (float, torch.Tensor)
        ``[(t, feats_(L, 8))]`` from
        ``stage2_mv(capture_slat_snapshots=True)``.  Feats are the
        normalised latent — this function applies ``slat_std/slat_mean``
        before decoding.
    pipeline
        SAM3D pipeline.  Must expose ``models["slat_decoder_gs"]`` plus
        the ``slat_mean`` / ``slat_std`` tensors.
    canonical_coords : torch.Tensor (L, 4) int32
        Sparse-tensor coords with batch column — same coords used by
        ``stage2_mv``.
    tokens_list : list of (FrameKey, decoder_input)
        Per-frame Stage-1 results; each ``decoder_input`` carries
        ``rotation`` (4 / 1×4), ``translation`` (3 / 1×3), ``scale``
        (3 / 1×3 / 1).
    sequence : Sequence
        Frame data accessor — provides ``H``, ``W`` and per-frame
        ``K_matrix``.
    output_dir : str
        Directory the grid PNG is written to.
    scene_name : str
    obj_idx : int
    canonical_mesh_verts : torch.Tensor (V, 3) or None
        Canonical mesh vertices in canonical-norm.  Required for the
        actionmesh warp; pass alongside the two per-frame fields.
    per_frame_mesh_verts : list[torch.Tensor (V, 3)] or None
        Per-slot deformed mesh vertices in frame-i self-norm.
    per_frame_mesh_rotations : list[torch.Tensor (V, 3, 3)] or None
        Per-slot per-vertex SO(3) rotation from kNN-Kabsch, blended onto
        the Gaussian quaternions.
    warp_knn_k, warp_knn_eps, warp_knn_chunk_size
        Knobs forwarded to ``warp_gaussians_high_res``.
    bg_color : torch.Tensor (3,) or None
        Background colour (default white).
    """
    import torch
    from PIL import Image as PILImage, ImageDraw

    if not slat_snapshots or not tokens_list:
        return
    if not torch.cuda.is_available():
        print("    [render_appearance_ode_steps] requires CUDA; skipping")
        return

    _has_mesh = canonical_mesh_verts is not None
    _has_pf_v = per_frame_mesh_verts is not None
    _has_pf_R = per_frame_mesh_rotations is not None
    if not (_has_mesh == _has_pf_v == _has_pf_R):
        raise ValueError(
            "render_appearance_ode_steps: canonical_mesh_verts, "
            "per_frame_mesh_verts and per_frame_mesh_rotations must be "
            "supplied together"
        )
    use_warp = _has_mesh

    from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp
    from pytorch3d.transforms import (
        quaternion_invert, quaternion_multiply, quaternion_to_matrix,
    )
    from genia.core.utils.rendering import render_gaussian_params
    from genia.core.utils.refinement import _render_frame_with_pose
    from genia.core.utils.quaternion_ops import (
        p3d_to_r3_positions, p3d_to_r3_quaternions,
    )
    if use_warp:
        from genia.core.utils.deformation import warp_gaussians_high_res

    device = torch.device("cuda")
    n_frames = len(tokens_list)
    H, W = int(sequence.H), int(sequence.W)

    if use_warp:
        if (len(per_frame_mesh_verts) != n_frames
                or len(per_frame_mesh_rotations) != n_frames):
            raise ValueError(
                f"per-frame mesh field length mismatch with tokens_list "
                f"(got verts={len(per_frame_mesh_verts)}, "
                f"R={len(per_frame_mesh_rotations)}, expected {n_frames})"
            )
        canonical_verts_d = canonical_mesh_verts.to(device).float()
        if canonical_verts_d.dim() != 2 or canonical_verts_d.shape[1] != 3:
            raise ValueError(
                f"canonical_mesh_verts must be (V, 3); got "
                f"{tuple(canonical_verts_d.shape)}"
            )
        V_mesh = canonical_verts_d.shape[0]
        per_frame_verts_d = [p.to(device).float() for p in per_frame_mesh_verts]
        per_frame_rotations_d = [r.to(device).float() for r in per_frame_mesh_rotations]
        faces_d = (
            canonical_mesh_faces.to(device).long()
            if canonical_mesh_faces is not None else None
        )
        for i, (pv, pr) in enumerate(zip(per_frame_verts_d, per_frame_rotations_d)):
            if pv.shape != (V_mesh, 3) or pr.shape != (V_mesh, 3, 3):
                raise ValueError(
                    f"slot {i}: verts={tuple(pv.shape)}, "
                    f"R={tuple(pr.shape)} "
                    f"(expected ({V_mesh}, 3) / ({V_mesh}, 3, 3))"
                )
    else:
        canonical_verts_d = per_frame_verts_d = per_frame_rotations_d = None
        faces_d = None

    bg = bg_color if bg_color is not None else torch.ones(
        3, device=device, dtype=torch.float32,
    )
    bg = bg.to(device).float()

    gs_decoder = pipeline.models["slat_decoder_gs"]
    slat_mean_d = pipeline.slat_mean.to(device)
    slat_std_d = pipeline.slat_std.to(device)
    canonical_coords_dev = canonical_coords.to(device)

    # One row per captured ODE step (all steps).
    n_snaps = len(slat_snapshots)
    snap_idxs = list(range(n_snaps))

    # Pre-resolve per-frame poses + intrinsics.
    quat_list, trans_list, scale_list, K_list = [], [], [], []
    for fk, di in tokens_list:
        rot = di["rotation"].to(device).float()
        tr = di["translation"].to(device).float()
        sc = di["scale"].to(device).float()
        if rot.dim() == 2:
            rot = rot.squeeze(0)
        if tr.dim() == 2:
            tr = tr.squeeze(0)
        if sc.dim() == 2:
            sc = sc.squeeze(0)
        quat_list.append(rot)
        trans_list.append(tr)
        scale_list.append(sc)
        K_list.append(sequence[fk].K_matrix)

    os.makedirs(output_dir, exist_ok=True)

    print(f"    Rendering appearance ODE steps: {len(snap_idxs)} steps × "
          f"{n_frames} frames"
          f"{' (deformation warp)' if use_warp else ''}")

    def _render_gs_row(gs_obj):
        """Render one row image (all frames concatenated) for the Gaussian branch."""
        if gs_obj is None or gs_obj.get_xyz.shape[0] == 0:
            return PILImage.new("RGB", (W * n_frames, H), (255, 255, 255)), 0
        n_g = gs_obj.get_xyz.shape[0]
        frame_imgs = []
        for i in range(n_frames):
            K_np = K_list[i]
            if hasattr(K_np, "cpu"):
                K_np = K_np.detach().cpu().numpy()
            quat_i = quat_list[i]
            trans_i = trans_list[i]
            scale_i = scale_list[i]
            with torch.no_grad():
                if use_warp:
                    means_w, quats_w = warp_gaussians_high_res(
                        gs_obj,
                        canonical_verts_d,
                        per_frame_verts_d[i],
                        per_frame_rotations_d[i],
                        K=warp_knn_k,
                        eps=warp_knn_eps,
                        chunk_size=warp_knn_chunk_size,
                        faces=faces_d,
                    )
                    rot = quat_i / quat_i.norm()
                    R_pose = quaternion_to_matrix(rot.unsqueeze(0)).squeeze(0)
                    if scale_i.dim() == 0:
                        sc_v = scale_i.expand(3)
                    elif scale_i.dim() == 1 and scale_i.shape[0] == 1:
                        sc_v = scale_i.expand(3)
                    else:
                        sc_v = scale_i
                    means_cam = torch.mm(means_w * sc_v, R_pose) + trans_i
                    rot_inv = quaternion_invert(rot.unsqueeze(0)).squeeze(0)
                    quats_cam = quaternion_multiply(
                        rot_inv.unsqueeze(0).expand(quats_w.shape[0], -1),
                        quats_w,
                    )
                    scales_cam = gs_obj.get_scaling * sc_v
                    # P3D → R3 conversion (matches _transform_object_to_r3
                    # / the canonical path through _render_frame_with_pose).
                    means_cam = p3d_to_r3_positions(means_cam)
                    quats_cam = p3d_to_r3_quaternions(quats_cam)
                    rgb, _alpha, _depth = render_gaussian_params(
                        means_cam, quats_cam, scales_cam,
                        gs_obj.get_opacity, gs_obj.get_features,
                        torch.eye(4, device=device).unsqueeze(0),
                        K_np, W, H, bg_color=bg,
                    )
                else:
                    rgb, _alpha, _depth = _render_frame_with_pose(
                        gs_obj, quat_i, trans_i, scale_i,
                        K_np, W, H, device, bg_color=bg,
                    )
            arr = (rgb.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
            frame_imgs.append(PILImage.fromarray(arr))
        total_w = sum(im.width for im in frame_imgs)
        row_img = PILImage.new(
            "RGB", (total_w, frame_imgs[0].height), (255, 255, 255),
        )
        xoff = 0
        for im in frame_imgs:
            row_img.paste(im, (xoff, 0))
            xoff += im.width
        return row_img, n_g

    imgs = []
    for step_pos, snap_i in enumerate(snap_idxs):
        t_val, feats_norm = slat_snapshots[snap_i]
        feats = feats_norm.to(device).float() * slat_std_d + slat_mean_d
        slat = sp.SparseTensor(coords=canonical_coords_dev, feats=feats)

        with torch.no_grad():
            gs_list = gs_decoder(slat)
        gs_obj = gs_list[0] if isinstance(gs_list, list) else gs_list
        row_g, n_g = _render_gs_row(gs_obj)
        draw = ImageDraw.Draw(row_g)
        label = (f"step {step_pos}/{len(snap_idxs)-1}  "
                 f"t={t_val:.3f}  ng={n_g}")
        draw.text((5, 5), label, fill=(255, 255, 255))
        draw.text((4, 4), label, fill=(0, 0, 0))
        imgs.append(row_g)

        print(f"    ODE step {step_pos:2d} (t={t_val:.3f}): "
              f"n_g={n_g} ({n_frames} frames)")

    # Vertically-stacked grid PNG.
    grid_w, cell_h = imgs[0].size
    grid = PILImage.new(
        "RGB", (grid_w, len(imgs) * cell_h), (255, 255, 255),
    )
    for row, img in enumerate(imgs):
        grid.paste(img, (0, row * cell_h))
    grid_path = os.path.join(
        output_dir,
        f"{scene_name}_obj{obj_idx}_slat_ode_steps_gaussian.png",
    )
    grid.save(grid_path)
    print(f"    Saved appearance ODE steps grid: {grid_path}")


def plot_finetune_losses(
    loss_history: List[Dict[str, float]],
    best_iter: int,
    output_path: str,
    config: Any = None,
) -> None:
    """Plot token fine-tuning loss curves and save to disk.

    Creates a single-row grid of subplots (one per loss type).
    When *config* is provided the subplot titles include the weight.
    """
    import matplotlib
    matplotlib.use("Agg")

    iterations = list(range(1, len(loss_history) + 1))

    columns = [
        ("total", "Total Loss", None),
        ("rgb", "RGB Loss", "rgb_weight"),
        ("ssim", "SSIM Loss", "rgb_ssim_weight"),
        ("silhouette", "Silhouette Loss", "silhouette_weight"),
        ("depth", "Depth Loss", "depth_weight"),
        ("normals", "Normals Loss", "normals_weight"),
        ("perceptual", "LPIPS Loss", "perceptual_weight"),
        ("drift", "Token Drift", None),
    ]

    n_cols = len(columns)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 3.5))

    for col_idx, (key, title, weight_attr) in enumerate(columns):
        ax = axes[col_idx]
        values = [m.get(key, 0.0) for m in loss_history]

        if weight_attr and config:
            w = getattr(config, weight_attr, None)
            if w is not None:
                title = f"{title} (w={w})"

        ax.plot(iterations, values, "b-", linewidth=1)

        if best_iter < len(values):
            ax.axvline(x=best_iter + 1, color="r", linestyle="--", alpha=0.7, linewidth=0.8)
            ax.scatter([best_iter + 1], [values[best_iter]], color="r", s=20, zorder=5)

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Iteration", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_path)
    print(f"  Saved loss plot: {output_path}")


def plot_parameter_drift(
    loss_history: List[Dict[str, float]],
    best_iter: int,
    output_path: str,
    has_poses: bool = False,
) -> None:
    """Plot token and pose parameter drift from their original values."""
    import matplotlib
    matplotlib.use("Agg")

    iterations = list(range(1, len(loss_history) + 1))

    columns = [
        ("token_l2", "Token L2 Drift", "b"),
        ("token_rel", "Token Relative Drift", "b"),
    ]
    if has_poses:
        columns.extend([
            ("pose_rot_l2", "Rotation L2 Drift (avg)", "g"),
            ("pose_trans_l2", "Translation L2 Drift (avg)", "orange"),
            ("pose_scale_l2", "Scale L2 Drift (avg)", "purple"),
        ])

    n_cols = len(columns)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 3.5))
    if n_cols == 1:
        axes = [axes]

    for col_idx, (key, title, color) in enumerate(columns):
        ax = axes[col_idx]
        values = [m.get(key, 0.0) for m in loss_history]

        ax.plot(iterations, values, color=color, linewidth=1)

        if best_iter < len(values):
            ax.axvline(x=best_iter + 1, color="r", linestyle="--", alpha=0.7, linewidth=0.8)
            ax.scatter([best_iter + 1], [values[best_iter]], color="r", s=20, zorder=5)

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Iteration", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(True, alpha=0.3)

        if "rel" in key:
            from matplotlib.ticker import PercentFormatter
            ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))

    plt.tight_layout()
    save_figure(fig, output_path)
    print(f"  Saved parameter drift plot: {output_path}")


def plot_gradient_norms(
    loss_history: List[Dict[str, float]],
    best_iter: int,
    output_path: str,
) -> None:
    """Plot per-group gradient norms over fine-tuning iterations.

    One subplot per parameter group (matching ``plot_finetune_losses`` style).
    Expects ``loss_history`` entries to contain ``grad_<name>`` keys
    (populated by ``_finetune_single_object``).
    """
    import matplotlib
    matplotlib.use("Agg")

    if not loss_history:
        return

    # Discover all grad_* keys present in the history
    grad_keys = sorted({
        k for m in loss_history for k in m if k.startswith("grad_")
    })
    if not grad_keys:
        return

    iterations = list(range(1, len(loss_history) + 1))

    n_cols = len(grad_keys)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 3.5))
    if n_cols == 1:
        axes = [axes]

    for col_idx, key in enumerate(grad_keys):
        ax = axes[col_idx]
        label = key.removeprefix("grad_")
        values = [m.get(key, 0.0) for m in loss_history]

        ax.plot(iterations, values, "b-", linewidth=1)

        if best_iter < len(values):
            ax.axvline(x=best_iter + 1, color="r", linestyle="--",
                       alpha=0.7, linewidth=0.8)
            ax.scatter([best_iter + 1], [values[best_iter]],
                       color="r", s=20, zorder=5)

        ax.set_title(f"{label}", fontsize=10)
        ax.set_xlabel("Iteration", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_path)
    print(f"  Saved gradient norms plot: {output_path}")


def plot_sh_magnitudes(
    dc_per_frame: Optional[Dict[int, "torch.Tensor"]] = None,
    sh_per_frame: Optional[Dict[int, "torch.Tensor"]] = None,
    sh_degree: int = 0,
    output_path: str = "",
) -> None:
    """Plot per-frame magnitude of appearance parameters at each SH degree.

    Creates a bar/line chart with frame index on the x-axis and mean per-Gaussian
    L2 norm on the y-axis, with one series per SH degree band (DC offset,
    degree-1, degree-2, degree-3).

    Parameters
    ----------
    dc_per_frame : dict, optional
        {frame_idx: tensor (N, 1, 3)} DC offsets per frame.
    sh_per_frame : dict, optional
        {frame_idx: tensor (N, K, 3)} combined SH rest per frame.
    sh_degree : int
        SH degree (0 = DC only, 1-3 = DC + SH rest).
    output_path : str
        Path to save the plot.
    """
    import matplotlib
    matplotlib.use("Agg")
    import torch

    has_dc = dc_per_frame is not None and len(dc_per_frame) > 0
    has_sh = sh_per_frame is not None and len(sh_per_frame) > 0

    if not has_dc and not has_sh:
        return

    # Collect sorted frame indices (union of both dicts)
    frame_set: set = set()
    if has_dc:
        frame_set.update(dc_per_frame.keys())
    if has_sh:
        frame_set.update(sh_per_frame.keys())
    frames = sorted(frame_set)

    # Build per-degree magnitude arrays
    # DC offset: shape (N, 1, 3) — always degree 0
    # SH rest: shape (N, K, 3) where K = (sh_degree+1)^2 - 1
    #   degree 1: bands 0-2 (3 bands)
    #   degree 2: bands 3-7 (5 bands)
    #   degree 3: bands 8-14 (7 bands)
    series = {}  # label -> list of magnitudes per frame

    if has_dc:
        dc_mags = []
        for fi in frames:
            if fi in dc_per_frame:
                t = dc_per_frame[fi].detach().float()
                if isinstance(t, torch.Tensor):
                    # Mean per-Gaussian L2 norm: sqrt(sum over channels) averaged over Gaussians
                    dc_mags.append(t.pow(2).sum(dim=-1).sqrt().mean().item())
                else:
                    dc_mags.append(0.0)
            else:
                dc_mags.append(0.0)
        series["DC offset (deg 0)"] = dc_mags

    if has_sh:
        # Split SH rest into per-degree bands
        # degree 1: 3 bands (indices 0:3)
        # degree 2: 5 bands (indices 3:8)
        # degree 3: 7 bands (indices 8:15); which bands exist depends on sh_degree
        degree_bands = []
        if sh_degree >= 1:
            degree_bands.append(("SH deg 1 (3 bands)", 0, 3))
        if sh_degree >= 2:
            degree_bands.append(("SH deg 2 (5 bands)", 3, 8))
        if sh_degree >= 3:
            degree_bands.append(("SH deg 3 (7 bands)", 8, 15))

        for label, start, end in degree_bands:
            mags = []
            for fi in frames:
                if fi in sh_per_frame:
                    t = sh_per_frame[fi].detach().float()
                    if isinstance(t, torch.Tensor) and t.shape[1] > start:
                        actual_end = min(end, t.shape[1])
                        band = t[:, start:actual_end, :]
                        mags.append(band.pow(2).sum(dim=-1).sqrt().mean().item())
                    else:
                        mags.append(0.0)
                else:
                    mags.append(0.0)
            series[label] = mags

    # Plot
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    n_series = len(series)
    fig, ax = plt.subplots(1, 1, figsize=(max(6, len(frames) * 0.3), 4))

    x = list(range(len(frames)))
    bar_width = 0.8 / max(n_series, 1)

    for i, (label, mags) in enumerate(series.items()):
        offsets = [xi + (i - n_series / 2 + 0.5) * bar_width for xi in x]
        ax.bar(offsets, mags, width=bar_width, label=label,
               color=colors[i % len(colors)], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([str(f) for f in frames], fontsize=7, rotation=45 if len(frames) > 20 else 0)
    ax.set_xlabel("Frame Index", fontsize=9)
    ax.set_ylabel("Mean per-Gaussian L2 norm", fontsize=9)
    ax.set_title("Per-frame Appearance Parameter Magnitudes", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="both", labelsize=7)

    plt.tight_layout()
    save_figure(fig, output_path)
    print(f"  Saved SH magnitude plot: {output_path}")


def plot_shape_and_poses_init_trajectory(
    shape_and_poses_init_results: Dict[int, Dict[str, "torch.Tensor"]],
    anchor_fid: int,
    output_path: str,
    title: str = "",
) -> None:
    """Plot decoded pose values (rotation, translation, scale) vs frame index.

    Creates a 3-row figure showing Euler angles, translation xyz, and scale
    across all frames in the pose init results. The anchor frame is marked
    with a vertical dashed line.

    Args:
        shape_and_poses_init_results: {frame_idx: {"rotation": (1,4), "translation": (1,3),
            "scale": (1,3)}} — decoded poses from pose init.
        anchor_fid: Frame index of the anchor (canonical) frame.
        output_path: Path to save the PNG figure.
        title: Optional title prefix.
    """
    import torch
    from genia.core.utils.quaternion_ops import matrix_to_euler_angles, quaternion_to_matrix

    import matplotlib
    matplotlib.use("Agg")

    sorted_fids = sorted(shape_and_poses_init_results.keys())
    if len(sorted_fids) < 2:
        return

    # Extract decoded values
    euler_angles = []
    translations = []
    scales = []
    for fid in sorted_fids:
        r = shape_and_poses_init_results[fid]
        quat = r["rotation"].detach().cpu().float()
        if quat.dim() == 1:
            quat = quat.unsqueeze(0)
        R_mat = quaternion_to_matrix(quat)
        euler = torch.rad2deg(matrix_to_euler_angles(R_mat, "XYZ")).squeeze(0)
        euler_angles.append(euler.numpy())

        t = r["translation"].detach().cpu().float().squeeze()
        translations.append(t.numpy())

        s = r["scale"].detach().cpu().float().squeeze()
        scales.append(s.numpy())

    euler_angles = np.array(euler_angles)  # (N, 3)
    translations = np.array(translations)  # (N, 3)
    scales = np.array(scales)  # (N,) or (N, 3)
    if scales.ndim == 1:
        scales = scales[:, None]

    # FrameKey-aware X axis: use sequential index so MV mode (where
    # sorted_fids is a list of FrameKey 2-tuples) doesn't trip
    # matplotlib's axvline / array-axis logic.
    x = list(range(len(sorted_fids)))
    try:
        anchor_x = sorted_fids.index(anchor_fid)
    except ValueError:
        anchor_x = None
    xtick_labels = [str(fid) for fid in sorted_fids]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Row 1: Euler angles
    ax = axes[0]
    for i, (label, color) in enumerate(zip(["rx", "ry", "rz"], ["r", "g", "b"])):
        ax.plot(x, euler_angles[:, i], f"{color}o-", markersize=3, linewidth=1, label=label)
    if anchor_x is not None:
        ax.axvline(anchor_x, color="gray", linestyle="--", alpha=0.7, label="anchor")
    ax.set_ylabel("Euler angle (deg)")
    ax.legend(fontsize=7, ncol=4)
    ax.grid(True, alpha=0.3)

    # Row 2: Translation
    ax = axes[1]
    for i, (label, color) in enumerate(zip(["tx", "ty", "tz"], ["r", "g", "b"])):
        ax.plot(x, translations[:, i], f"{color}o-", markersize=3, linewidth=1, label=label)
    if anchor_x is not None:
        ax.axvline(anchor_x, color="gray", linestyle="--", alpha=0.7)
    ax.set_ylabel("Translation")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)

    # Row 3: Scale
    ax = axes[2]
    if scales.shape[1] == 1:
        ax.plot(x, scales[:, 0], "ko-", markersize=3, linewidth=1, label="scale")
    else:
        for i, (label, color) in enumerate(zip(["sx", "sy", "sz"], ["r", "g", "b"])):
            ax.plot(x, scales[:, i], f"{color}o-", markersize=3, linewidth=1, label=label)
    if anchor_x is not None:
        ax.axvline(anchor_x, color="gray", linestyle="--", alpha=0.7)
    ax.set_ylabel("Scale")
    ax.set_xlabel("Frame index")
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, fontsize=7, rotation=30, ha="right")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)

    fig.suptitle(title or "Pose init trajectory", fontsize=11)
    plt.tight_layout()
    save_figure(fig, output_path)
    print(f"  Saved pose init trajectory: {output_path}")


def plot_multi_frame_ode_trajectories(
    per_frame_histories: Dict[int, List[Tuple[float, Dict[str, "torch.Tensor"]]]],
    output_path: str,
    title: str = "",
) -> None:
    """Plot ODE trajectories for all frames overlaid on the same axes.

    Each frame is a different color. Shows whether frames converge to
    similar/smooth final values or diverge during denoising.

    Args:
        per_frame_histories: {frame_idx: ode_history} where ode_history is
            a list of (timestep, {modality: tensor}) from step_callback.
        output_path: Path to save the PNG figure.
        title: Optional title.
    """
    import matplotlib
    matplotlib.use("Agg")

    if not per_frame_histories:
        return

    sorted_fids = sorted(per_frame_histories.keys())
    sample_history = per_frame_histories[sorted_fids[0]]
    if not sample_history:
        return

    modalities = ["6drotation_normalized", "translation", "scale", "translation_scale"]
    present = [k for k in modalities if k in sample_history[0][1]]
    if not present:
        return

    # Determine dims per modality
    dims_per_mod = {}
    for mod_name in present:
        v = sample_history[0][1][mod_name].squeeze()
        dims_per_mod[mod_name] = 1 if v.dim() == 0 else v.shape[0]

    total_dims = sum(dims_per_mod.values())
    fig, axes = plt.subplots(1, total_dims, figsize=(3 * total_dims, 3.5))
    if total_dims == 1:
        axes = [axes]

    frame_cmap = plt.cm.viridis(np.linspace(0, 1, len(sorted_fids)))
    ax_idx = 0
    for mod_name in present:
        n_dims = dims_per_mod[mod_name]
        for d in range(n_dims):
            ax = axes[ax_idx]
            for fi, fid in enumerate(sorted_fids):
                history = per_frame_histories[fid]
                t_values = np.array([h[0] for h in history])
                vals = [h[1][mod_name].squeeze().numpy() for h in history]
                vals = np.array(vals)
                if vals.ndim == 1:
                    vals = vals[:, None]
                ax.plot(t_values, vals[:, d], color=frame_cmap[fi],
                        linewidth=0.6, alpha=0.7)
            dim_label = f"d{d}" if n_dims > 1 else ""
            ax.set_title(f"{mod_name} {dim_label}".strip(), fontsize=7)
            ax.set_xlabel("ODE time $t$", fontsize=7)
            ax.tick_params(axis="both", labelsize=6)
            ax.grid(True, alpha=0.3)
            ax_idx += 1

    fig.suptitle(title or "Multi-frame ODE trajectories", fontsize=10)
    plt.tight_layout()
    save_figure(fig, output_path)
    print(f"  Saved multi-frame ODE trajectories: {output_path}")


def plot_shape_ode_trajectories(
    per_frame_histories: Dict[int, List[Tuple[float, Dict[str, "torch.Tensor"]]]],
    output_path: str,
    title: str = "",
) -> None:
    """Plot shape token ODE trajectories aggregated to one line per frame.

    Shape tokens are high-dimensional (4096 tokens x 8 features).  To make this
    readable, the step callback logs two compact summaries per frame per ODE step:

    - ``shape_mean`` — mean feature vector across all tokens (8 dims)
    - ``shape_norm_std`` — std of per-token L2 norms (1 scalar)

    The plot has **3 panels**:

    1. **PCA of mean features**: projects the (steps, 8) mean-feature trajectory
       into 3 principal components, showing one curve per frame (viridis colormap).
       Reveals whether frames converge to the same canonical shape.
    2. **Mean-feature L2 norm**: the magnitude of the mean feature vector at each
       ODE step — shows convergence speed per frame.
    3. **Token norm spread**: std of per-token norms — tracks how much per-token
       diversity remains as denoising progresses.

    Args:
        per_frame_histories: ``{frame_idx: [(t, {modality: tensor}), ...]}``
            Must include ``shape_mean`` and ``shape_norm_std`` keys (added by
            ``_step_callback`` in ``stage1_batched``).
        output_path: PNG save path.
        title: Optional super-title.
    """
    import matplotlib
    matplotlib.use("Agg")

    if not per_frame_histories:
        return

    sorted_fids = sorted(per_frame_histories.keys())
    sample_history = per_frame_histories[sorted_fids[0]]
    if not sample_history or "shape_mean" not in sample_history[0][1]:
        return

    n_steps = len(sample_history)
    n_frames = len(sorted_fids)
    t_values = np.array([h[0] for h in sample_history])
    frame_cmap = plt.cm.viridis(np.linspace(0, 1, n_frames))

    # Gather per-frame trajectories: mean features (F, S, D) and norm std (F, S)
    mean_feats = []  # (F, S, D)
    norm_stds = []   # (F, S)
    for fid in sorted_fids:
        hist = per_frame_histories[fid]
        mf = np.array([h[1]["shape_mean"].numpy() for h in hist])  # (S, D)
        ns = np.array([h[1]["shape_norm_std"].item() for h in hist])  # (S,)
        mean_feats.append(mf)
        norm_stds.append(ns)
    mean_feats = np.stack(mean_feats)  # (F, S, D)
    norm_stds = np.stack(norm_stds)    # (F, S)

    # PCA: flatten (F*S, D) → project to 3 components
    flat = mean_feats.reshape(-1, mean_feats.shape[-1])  # (F*S, D)
    flat_centered = flat - flat.mean(axis=0, keepdims=True)
    try:
        _, _, Vt = np.linalg.svd(flat_centered, full_matrices=False)
        n_pcs = min(3, Vt.shape[0])
        proj = flat_centered @ Vt[:n_pcs].T  # (F*S, n_pcs)
        proj = proj.reshape(n_frames, n_steps, n_pcs)
    except np.linalg.LinAlgError:
        n_pcs = 0
        proj = None

    # Mean-feature L2 norm per frame per step
    mean_norms = np.linalg.norm(mean_feats, axis=-1)  # (F, S)

    n_cols = n_pcs + 2  # PCs + mean norm + norm std
    fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 3.5))
    if n_cols == 1:
        axes = [axes]

    # Panel 1..n_pcs: PCA components
    if proj is not None:
        for pc in range(n_pcs):
            ax = axes[pc]
            for fi in range(n_frames):
                ax.plot(t_values, proj[fi, :, pc], color=frame_cmap[fi],
                        linewidth=0.7, alpha=0.8)
            ax.set_title(f"shape PC{pc + 1}", fontsize=8)
            ax.set_xlabel("ODE time $t$", fontsize=7)
            ax.tick_params(axis="both", labelsize=6)
            ax.grid(True, alpha=0.3)

    # Panel: mean-feature L2 norm
    ax_norm = axes[n_pcs]
    for fi in range(n_frames):
        ax_norm.plot(t_values, mean_norms[fi], color=frame_cmap[fi],
                     linewidth=0.7, alpha=0.8)
    ax_norm.set_title("mean-feat ‖·‖", fontsize=8)
    ax_norm.set_xlabel("ODE time $t$", fontsize=7)
    ax_norm.tick_params(axis="both", labelsize=6)
    ax_norm.grid(True, alpha=0.3)

    # Panel: token norm spread
    ax_std = axes[n_pcs + 1]
    for fi in range(n_frames):
        ax_std.plot(t_values, norm_stds[fi], color=frame_cmap[fi],
                    linewidth=0.7, alpha=0.8)
    ax_std.set_title("token norm σ", fontsize=8)
    ax_std.set_xlabel("ODE time $t$", fontsize=7)
    ax_std.tick_params(axis="both", labelsize=6)
    ax_std.grid(True, alpha=0.3)

    fig.suptitle(title or "Shape ODE trajectories", fontsize=10)
    plt.tight_layout()
    save_figure(fig, output_path)
    print(f"  Saved shape ODE trajectories: {output_path}")


def plot_slat_ode_trajectory(
    ode_history: List[Tuple[float, Dict[str, "torch.Tensor"]]],
    output_path: str,
    title: str = "",
) -> None:
    """Plot SLAT feature evolution across ODE timesteps (Stage 2).

    Shows how the 8-dim SLAT features evolve from noise (t~0) to clean (t~1).
    Two panels: mean feature per dimension (8 lines) and per-voxel norm spread.

    Args:
        ode_history: List of (timestep, {"slat_mean": (8,), "slat_norm_std": (1,)})
            snapshots collected at each ODE step.
        output_path: Path to save the PNG figure.
        title: Optional title.
    """
    import matplotlib
    matplotlib.use("Agg")

    if not ode_history or "slat_mean" not in ode_history[0][1]:
        return

    t_values = np.array([h[0] for h in ode_history])
    mean_feats = np.array([h[1]["slat_mean"].numpy() for h in ode_history])  # (S, 8)
    norm_stds = np.array([h[1]["slat_norm_std"].item() for h in ode_history])  # (S,)

    dim_colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    # Panel 1: mean feature per dimension
    ax = axes[0]
    for d in range(mean_feats.shape[1]):
        ax.plot(t_values, mean_feats[:, d], color=dim_colors[d % 10],
                linewidth=0.8, alpha=0.8, label=f"d{d}")
    ax.set_title("SLAT mean feature", fontsize=9)
    ax.set_xlabel("ODE time $t$", fontsize=8)
    ax.legend(fontsize=6, ncol=4, loc="best")
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 2: per-voxel norm spread
    ax = axes[1]
    ax.plot(t_values, norm_stds, "k-", linewidth=0.8)
    ax.set_title("voxel norm σ", fontsize=9)
    ax.set_xlabel("ODE time $t$", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle(title or "Stage 2 SLAT ODE trajectory", fontsize=10)
    plt.tight_layout()
    save_figure(fig, output_path)
    print(f"  Saved SLAT ODE trajectory: {output_path}")


def visualize_slat_voxels_before_after(
    feats_before: "torch.Tensor",
    feats_after: "torch.Tensor",
    coords: "torch.Tensor",
    output_path: str,
    title: str = "",
) -> None:
    """Visualize SLAT voxels before and after with shared PCA basis.

    Both feature sets are projected using a single PCA computed on the
    concatenation so that colors are directly comparable.

    Parameters
    ----------
    feats_before : torch.Tensor
        (N, C) features before the update.
    feats_after : torch.Tensor
        (N, C) features after the update.
    coords : torch.Tensor
        (N, 4) sparse tensor coords shared by both feature sets
        (batch_idx, x, y, z).
    output_path : str
        Path to save the output PNG.
    title : str, optional
        Figure suptitle.
    """
    import torch

    xyz_after = coords[:, 1:].detach().cpu().float().numpy()  # (N, 3)

    F_before = feats_before.detach().cpu().float()
    F_after = feats_after.detach().cpu().float()

    # Shared PCA basis from concatenated features
    F_cat = torch.cat([F_before, F_after], dim=0)
    mean = F_cat.mean(dim=0, keepdim=True)
    _, _, Vh = torch.linalg.svd(F_cat - mean, full_matrices=False)
    basis = Vh[:3]  # (3, C)

    def _project_and_normalize(F: torch.Tensor) -> np.ndarray:
        pca3 = ((F - mean) @ basis.T).numpy()
        for c in range(3):
            lo, hi = pca3[:, c].min(), pca3[:, c].max()
            if hi - lo > 1e-8:
                pca3[:, c] = (pca3[:, c] - lo) / (hi - lo)
            else:
                pca3[:, c] = 0.5
        return pca3

    rgb_before = _project_and_normalize(F_before)
    rgb_after = _project_and_normalize(F_after)

    # Per-voxel difference (same grid)
    diff = (F_after - F_before).norm(dim=1).numpy()  # (N,)
    diff_normalized = diff / max(diff.max(), 1e-8)

    viewpoints = [
        (30, 45, "Front-right"),
        (30, 135, "Back-left"),
        (75, 45, "Top-down"),
    ]

    n_views = len(viewpoints)
    n_rows = 3
    fig = plt.figure(figsize=(5 * n_views, 5 * n_rows))

    rows = [
        ("Before", rgb_before, xyz_after),
        ("After", rgb_after, xyz_after),
    ]
    for row_idx, (label, rgb, xyz) in enumerate(rows):
        for col_idx, (elev, azim, vp_label) in enumerate(viewpoints):
            ax = fig.add_subplot(n_rows, n_views, row_idx * n_views + col_idx + 1, projection="3d")
            ax.scatter(
                xyz[:, 0], xyz[:, 1], xyz[:, 2],
                c=rgb, s=40, marker="s", edgecolors="none", alpha=0.9,
            )
            ax.set_xlim(0, 64)
            ax.set_ylim(0, 64)
            ax.set_zlim(0, 64)
            ax.set_box_aspect([1, 1, 1])
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f"{label} — {vp_label}", fontsize=9)
            ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    # Difference row: heatmap coloring
    import matplotlib.cm as cm
    diff_colors = cm.hot(diff_normalized)[:, :3]  # (N, 3)
    for col_idx, (elev, azim, vp_label) in enumerate(viewpoints):
        ax = fig.add_subplot(n_rows, n_views, 2 * n_views + col_idx + 1, projection="3d")
        ax.scatter(
            xyz_after[:, 0], xyz_after[:, 1], xyz_after[:, 2],
            c=diff_colors, s=40, marker="s", edgecolors="none", alpha=0.9,
        )
        ax.set_xlim(0, 64)
        ax.set_ylim(0, 64)
        ax.set_zlim(0, 64)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"Difference — {vp_label} (max={diff.max():.3f})", fontsize=9)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    if title:
        fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    save_figure(fig, output_path)
    print(f"  Saved SLAT before/after visualization to {output_path}")


def render_color_shift_debug(
    canonical_gs: Any,
    dc_per_frame: Dict[int, "torch.Tensor"],
    sh_per_frame: Optional[Dict[int, "torch.Tensor"]],
    poses: Dict[int, Dict[str, "torch.Tensor"]],
    sequence: Any,
    obj_idx: int,
    output_path: str,
    amplification: float = 10.0,
    white_background: bool = False,
) -> None:
    """Render per-frame color shift debug visualization.

    For each keyframe produces two images:

    1. **Amplified pixel difference** — ``|render(canon + offset) -
       render(canon)| * amplification``, showing the pixel-space impact of
       the per-frame DC/SH offsets after splatting and alpha-blending.
    2. **Offset-only render** — Gaussians rendered with
       ``_features_dc = dc_offset * amplification`` (no canonical base, no
       SH rest).  The 0th-order SH evaluation maps this to
       ``offset * amp * C0 + 0.5``, producing a gray midpoint with colored
       deviations that show offset direction and magnitude per Gaussian.

    Parameters
    ----------
    canonical_gs : Gaussian
        Decoded canonical Gaussian (e.g. ``best_gs``).
    dc_per_frame : dict
        ``{frame_idx: Tensor (N, 1, 3)}`` per-frame DC offsets.
    sh_per_frame : dict or None
        ``{frame_idx: Tensor (N, K, 3)}`` per-frame SH rest, or None.
    poses : dict
        ``{frame_idx: {"rotation": ..., "translation": ..., "scale": ...}}``.
    sequence : Sequence
        Scene data (provides ``K_matrix``, ``H``, ``W`` per frame).
    obj_idx : int
        Object index (for frame data loading).
    output_path : str
        Where to save the grid PNG.
    amplification : float
        Multiplier for difference and offset-only renders.
    white_background : bool
        If True, use white background for difference renders (matching
        the finetuning rendering convention).
    """
    import torch
    from genia.core.utils.refinement import _render_frame_with_pose

    frame_indices = sorted(dc_per_frame.keys())
    if not frame_indices:
        return

    device = canonical_gs.get_xyz.device
    H, W = sequence.H, sequence.W

    bg_color = torch.ones(3, device=device) if white_background else None
    gray_bg = torch.full((3,), 0.5, device=device)

    diff_images = []
    offset_images = []

    with torch.no_grad():
        for fi in frame_indices:
            if fi not in poses:
                continue

            pose = poses[fi]
            rot = pose["rotation"]
            trans = pose["translation"]
            scale = pose["scale"]
            K_matrix = sequence[fi].K_matrix

            # Save canonical state
            saved_dc = canonical_gs._features_dc
            saved_rest = canonical_gs._features_rest
            saved_deg = canonical_gs.sh_degree
            saved_adeg = canonical_gs.active_sh_degree

            # --- Render WITH offset ---
            canonical_gs._features_dc = saved_dc + dc_per_frame[fi]
            if sh_per_frame is not None and fi in sh_per_frame:
                from genia.core.utils.gaussian import attach_sh_rest
                attach_sh_rest(canonical_gs, sh_per_frame[fi])

            rgb_with, _, _ = _render_frame_with_pose(
                canonical_gs, rot, trans, scale, K_matrix, W, H, device,
                bg_color=bg_color,
            )

            # Restore
            canonical_gs._features_dc = saved_dc
            canonical_gs._features_rest = saved_rest
            canonical_gs.sh_degree = saved_deg
            canonical_gs.active_sh_degree = saved_adeg

            # --- Render WITHOUT offset ---
            rgb_without, _, _ = _render_frame_with_pose(
                canonical_gs, rot, trans, scale, K_matrix, W, H, device,
                bg_color=bg_color,
            )

            # --- Amplified pixel difference ---
            diff = (rgb_with - rgb_without).abs() * amplification
            diff = diff.clamp(0.0, 1.0)
            diff_images.append(diff.cpu().numpy())

            # --- Offset-only render ---
            canonical_gs._features_dc = dc_per_frame[fi] * amplification
            canonical_gs._features_rest = None
            canonical_gs.sh_degree = 0
            canonical_gs.active_sh_degree = 0

            rgb_offset, _, _ = _render_frame_with_pose(
                canonical_gs, rot, trans, scale, K_matrix, W, H, device,
                bg_color=gray_bg,
            )
            rgb_offset = rgb_offset.clamp(0.0, 1.0)
            offset_images.append(rgb_offset.cpu().numpy())

            # Restore
            canonical_gs._features_dc = saved_dc
            canonical_gs._features_rest = saved_rest
            canonical_gs.sh_degree = saved_deg
            canonical_gs.active_sh_degree = saved_adeg

    if not diff_images:
        return

    # --- Assemble grid ---
    n_frames = len(diff_images)
    fig, axes = plt.subplots(n_frames, 2, figsize=(8, 3 * n_frames))
    if n_frames == 1:
        axes = axes[np.newaxis, :]

    amp_str = f"\u00d7{amplification:g}"
    axes[0, 0].set_title(f"Pixel Diff ({amp_str})", fontsize=10)
    axes[0, 1].set_title(f"Offset Only ({amp_str})", fontsize=10)

    rendered_frames = [fi for fi in frame_indices if fi in poses]
    for row, fi in enumerate(rendered_frames):
        axes[row, 0].imshow(diff_images[row])
        axes[row, 0].set_ylabel(f"f{fi}", fontsize=9)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        axes[row, 1].imshow(offset_images[row])
        axes[row, 1].set_xticks([])
        axes[row, 1].set_yticks([])

    fig.suptitle(f"Color Shift Debug — Object {obj_idx}", fontsize=12, y=1.0)
    fig.tight_layout()
    save_figure(fig, output_path)
    print(f"  Saved color shift debug: {output_path}")


def prepare_entropy_viz_data(
    entropy: np.ndarray,
    voxel_xyz: "np.ndarray | None",
    decoded_grid: int = 64,
    sparse_coords: "np.ndarray | None" = None,
) -> "Tuple[np.ndarray, np.ndarray]":
    """Build 3D coordinates and mask entropy to occupied voxels.

    Parameters
    ----------
    entropy : np.ndarray (N_views, L)
        Per-view per-latent entropy (or any per-point values).
    voxel_xyz : np.ndarray (M, 3) or None
        Decoded voxel coordinates (64^3 grid), used to mask empty latents.
    decoded_grid : int
        Decoded voxel grid resolution (default 64).
    sparse_coords : np.ndarray (L, 3) or None
        Actual sparse tensor coordinates at the resolution matching entropy.
        When provided, used directly as visualization coordinates (bypasses
        the regular-grid assumption that fails for non-cubic L).

    Returns
    -------
    viz_xyz : np.ndarray (K, 3)
        3D coordinates for occupied latent points.
    viz_entropy : np.ndarray (N_views, K)
        Entropy values for occupied latent points.
    """
    L = entropy.shape[1]

    # Direct coordinates provided (e.g. from SparseEntropyHook) — use as-is
    if sparse_coords is not None and sparse_coords.shape[0] == L:
        return sparse_coords.astype(np.float32), entropy

    if voxel_xyz is not None and L == voxel_xyz.shape[0]:
        return voxel_xyz.astype(np.float32), entropy

    # Build regular latent grid (e.g. 16^3 = 4096)
    latent_grid = round(L ** (1.0 / 3))
    # Guard: only use regular grid when L is actually a perfect cube
    if latent_grid ** 3 != L:
        # Non-cubic L (e.g. sparse downsampled tokens) — use flat index coords
        idx = np.arange(L)
        side = int(np.ceil(L ** (1.0 / 3)))
        viz_xyz = np.stack([
            idx % side,
            (idx // side) % side,
            idx // (side ** 2),
        ], axis=-1).astype(np.float32)
        return viz_xyz, entropy

    idx = np.arange(L)
    viz_xyz = np.stack([
        idx % latent_grid,
        (idx // latent_grid) % latent_grid,
        idx // (latent_grid ** 2),
    ], axis=-1).astype(np.float32)

    # Mask to occupied voxels
    if voxel_xyz is not None and latent_grid > 0:
        ratio = max(decoded_grid // latent_grid, 1)
        latent_ijk = np.clip(
            (voxel_xyz / ratio).astype(int), 0, latent_grid - 1
        )
        flat = (latent_ijk[:, 0]
                + latent_ijk[:, 1] * latent_grid
                + latent_ijk[:, 2] * latent_grid ** 2)
        mask = np.zeros(L, dtype=bool)
        mask[np.unique(flat)] = True
        viz_xyz = viz_xyz[mask]
        entropy = entropy[:, mask]

    return viz_xyz, entropy


_ATTN_MODE_SETTINGS = {
    "entropy": {
        "cmap": "RdBu_r",  # blue=low entropy (confident), red=high
        "cbar_label": "Entropy (Blue = low, Red = high)",
    },
}


_ATTN_CAMERA_ANGLES = {
    "Right":       (15, -60),
    "Left":        (15, 120),
    "Front right": (15, -30),
    "Back left":   (15, 150),
    "Top":         (75,   0),
}


_VISIBILITY_CATEGORY_LABELS = {
    0: "Invisible",
    1: "Visible",
}


def save_perframe_voxel_visibility_plot(coords_np, vis_per_frame, frame_poses, sequence, obj_idx, output_path, title):
    """Per-frame voxel-visibility plot (3D scatter + per-frame input image with pose
    axes + GT mask + rendered voxel visibility), via plot_attention_weights(mode=
    "visibility").  Shared by appearance_init canonical and FINETUNE.

    coords_np: (N,3) int voxel grid coords.  vis_per_frame: (V,N) per-frame
    per-voxel visibility (float/bool).  frame_poses: list of (frame_key, pose_dict),
    V entries; pose_dict has 'rotation'/'translation'/'scale' tensors."""
    import torch
    from genia.core.utils.interpolation import compute_pose_axes
    from pytorch3d.transforms import quaternion_to_matrix
    pose_dict = {obj_idx: {fk: di for fk, di in frame_poses}}
    fks = [fk for fk, _ in frame_poses]
    axes_data = compute_pose_axes(pose_dict, fks)
    input_imgs, input_masks, renders = [], [], []
    for vi, (fk, di) in enumerate(frame_poses):
        img = sequence[fk].image
        K = sequence[fk].K_matrix
        img_t = torch.from_numpy(img).float() / 255.0
        input_imgs.append((draw_pose_axes_on_image(img_t, axes_data, vi, K) * 255).astype(np.uint8))
        input_masks.append(sequence[fk].masks[obj_idx])
        R = quaternion_to_matrix(di["rotation"].detach().cpu().float().reshape(1, 4)).squeeze(0).numpy()
        renders.append(render_voxel_binary_visibility(
            coords_np.astype(np.int32), visible=vis_per_frame[vi].astype(bool),
            K_matrix=K, W=sequence.W, H=sequence.H, obj_rotation=R,
            obj_translation=di["translation"].detach().cpu().numpy().flatten()[:3],
            obj_scale=di["scale"].detach().cpu().numpy().flatten(),
            c2w=getattr(sequence[fk], "c2w", None)))
    viz_xyz, viz_vals = prepare_entropy_viz_data(vis_per_frame, coords_np, sparse_coords=coords_np)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plot_attention_weights(
        viz_xyz, viz_vals, [f"frame {fk.frame}" for fk, _ in frame_poses],
        output_path=output_path, mode="visibility",
        input_images=input_imgs, input_masks=input_masks, render_images=renders,
        title=title)


def plot_attention_weights(
    xyz: np.ndarray,
    values: np.ndarray,
    view_labels: "List[str]",
    output_path: str,
    mode: str = "entropy",
    input_images: "list | None" = None,
    input_masks: "list | None" = None,
    render_images: "list | None" = None,
    title: str = "",
    point_size: float = 1.5,
    figscale: float = 1.0,
    dpi: int = 200,
) -> None:
    """Paper-style multi-angle 3D scatter visualization of per-view attention.

    Replicates Figure 3 from MV-SAM3D: rows = input viewpoints, columns =
    canonical 3D viewing angles.  Supports both entropy and visibility modes.

    Parameters
    ----------
    xyz : np.ndarray (K, 3)
        3D coordinates (voxel or latent grid).
    values : np.ndarray (N_views, K)
        Per-view per-point values to colorize.
    view_labels : list of str
        Label for each view row (e.g. ``["frame 0", "frame 10", ...]``).
    output_path : str
        Where to save the figure (pdf/png/svg).
    mode : str
        ``"entropy"`` — blue=low (confident), red=high (uncertain).
        ``"visibility"`` — blue=high (visible), red=low (occluded).
    input_images : list of np.ndarray, optional
        If provided, prepends an "Input" column with each view's image.
        Each element should be (H, W, 3) in [0, 1] or [0, 255].
    input_masks : list of np.ndarray, optional
        Per-view boolean masks (H, W).  When provided alongside
        ``input_images``, pixels outside the mask are darkened to
        highlight the object.
    render_images : list of np.ndarray, optional
        Per-view composite Gaussian renders (one per view), shown in a
        column after scene images.  Each element is (H, W, 3) uint8.
    title : str
        Optional suptitle.
    point_size : float
        Scatter marker size.
    figscale : float
        Global scale factor for figure dimensions.
    dpi : int
        Output resolution.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors

    # ── Mode settings ────────────────────────────────────────────────
    discrete_mode = (mode == "visibility")
    if not discrete_mode:
        if mode not in _ATTN_MODE_SETTINGS:
            raise ValueError(
                f"Unknown mode {mode!r}, expected one of "
                f"{list(_ATTN_MODE_SETTINGS) + ['visibility']}"
            )
        settings = _ATTN_MODE_SETTINGS[mode]
        cmap = plt.get_cmap(settings["cmap"])
        cbar_label = settings["cbar_label"]

    # ── Normalize xyz to [-0.5, 0.5] ────────────────────────────────
    xyz_norm = np.array(xyz, dtype=np.float32)
    lo = xyz_norm.min(axis=0)
    hi = xyz_norm.max(axis=0)
    span = (hi - lo).max()
    if span > 0:
        xyz_norm = (xyz_norm - (lo + hi) / 2) / span

    N_views = values.shape[0]
    angle_names = list(_ATTN_CAMERA_ANGLES.keys())
    n_angle_cols = len(angle_names)

    # Decide whether to show input images / render images columns
    show_images = input_images is not None and len(input_images) == N_views
    show_render = render_images is not None and len(render_images) == N_views
    n_img_cols = (1 if show_images else 0) + (1 if show_render else 0)
    n_cols = n_img_cols + n_angle_cols + 1  # +1 for colorbar/legend

    # Global value range across all views (percentile-based) — continuous only
    norm = None
    if not discrete_mode:
        all_vals = values.ravel()
        vmin = float(np.percentile(all_vals, 1))
        vmax = float(np.percentile(all_vals, 99))
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # ── Figure layout ────────────────────────────────────────────────
    col_w = 2.8 * figscale
    row_h = 2.8 * figscale
    img_col_w = 2.2 * figscale
    cbar_w = 0.35 * figscale

    width_ratios = []
    if show_images:
        width_ratios.append(img_col_w)
    if show_render:
        width_ratios.append(img_col_w)
    width_ratios.extend([col_w] * n_angle_cols)
    width_ratios.append(cbar_w)

    fig, axes = plt.subplots(
        N_views, n_cols,
        figsize=(sum(width_ratios), row_h * N_views),
        gridspec_kw={"width_ratios": width_ratios},
        squeeze=False,
    )

    # ── Column titles ────────────────────────────────────────────────
    col = 0
    if show_images:
        axes[0, col].set_title("Input", fontsize=10, fontweight="bold")
        col += 1
    if show_render:
        axes[0, col].set_title("Render", fontsize=10, fontweight="bold")
        col += 1
    for name in angle_names:
        axes[0, col].set_title(name, fontsize=10, fontweight="bold")
        col += 1

    # ── Draw each row ────────────────────────────────────────────────
    for row in range(N_views):
        vals_np = values[row]
        col = 0

        # Optional input image
        if show_images:
            ax_img = axes[row, col]
            ax_img.axis("off")
            img = input_images[row]
            if hasattr(img, "numpy"):
                img = img.numpy()
            if img.dtype != np.uint8 and img.max() <= 1.0:
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            # Darken pixels outside the object mask to highlight the object
            if input_masks is not None and row < len(input_masks):
                mask = input_masks[row]
                if hasattr(mask, "numpy"):
                    mask = mask.numpy()
                img = img.copy()
                img[~mask.astype(bool)] = (img[~mask.astype(bool)] * 0.4).astype(np.uint8)
            ax_img.imshow(img)
            ax_img.set_ylabel(view_labels[row], fontsize=8,
                              rotation=0, ha="right", va="center",
                              labelpad=10)
            col += 1

        # Optional composite Gaussian render
        if show_render:
            ax_rn = axes[row, col]
            ax_rn.axis("off")
            rn_img = render_images[row]
            if hasattr(rn_img, "numpy"):
                rn_img = rn_img.numpy()
            ax_rn.imshow(rn_img)
            col += 1

        # 3D scatter from each canonical angle
        if discrete_mode:
            # Map binary visibility (0/1) to fixed RGBA colors -- palette
            # is the 2-entry _VISIBILITY_CATEGORY_COLORS (red/blue).
            cat_ints = np.clip(vals_np.astype(int), 0, 1)
            colors = np.zeros((len(cat_ints), 4), dtype=np.float32)
            for cat, rgb in _VISIBILITY_CATEGORY_COLORS.items():
                mask = cat_ints == cat
                colors[mask, :3] = rgb
                colors[mask, 3] = 1.0
        else:
            colors = cmap(norm(vals_np))  # (K, 4) — same for all angles
        for angle_name in angle_names:
            ax = axes[row, col]
            pos = ax.get_position()
            ax.remove()
            ax3 = fig.add_axes(pos, projection="3d")
            axes[row, col] = ax3

            elev, azim = _ATTN_CAMERA_ANGLES[angle_name]

            ax3.scatter(
                xyz_norm[:, 0], xyz_norm[:, 1], xyz_norm[:, 2],
                c=colors, s=point_size, alpha=0.85,
                depthshade=True, edgecolors="none", rasterized=True,
            )
            ax3.view_init(elev=elev, azim=azim)
            ax3.set_xlim(-0.55, 0.55)
            ax3.set_ylim(-0.55, 0.55)
            ax3.set_zlim(-0.55, 0.55)
            ax3.set_box_aspect([1, 1, 1])

            ax3.set_xlabel("X", fontsize=6, labelpad=-2)
            ax3.set_ylabel("Y", fontsize=6, labelpad=-2)
            ax3.set_zlabel("Z", fontsize=6, labelpad=-2)
            ax3.tick_params(labelsize=5, pad=0)
            ax3.xaxis.pane.fill = False
            ax3.yaxis.pane.fill = False
            ax3.zaxis.pane.fill = False
            ax3.xaxis.pane.set_edgecolor("lightgray")
            ax3.yaxis.pane.set_edgecolor("lightgray")
            ax3.zaxis.pane.set_edgecolor("lightgray")
            ax3.grid(True, linewidth=0.3, alpha=0.5)

            # Row label on first 3D column (when no image column)
            if not show_images and col == 0:
                ax3.text2D(-0.15, 0.5, view_labels[row], fontsize=8,
                           transform=ax3.transAxes, rotation=90,
                           ha="center", va="center")

            col += 1

        # Colorbar / legend in last column
        ax_cb = axes[row, col]
        ax_cb.axis("off")
        if discrete_mode:
            # Draw legend only on the first row
            if row == 0:
                from matplotlib.patches import Patch
                handles = [
                    Patch(facecolor=_VISIBILITY_CATEGORY_COLORS[cat],
                          label=_VISIBILITY_CATEGORY_LABELS[cat])
                    for cat in sorted(_VISIBILITY_CATEGORY_COLORS)
                ]
                ax_cb.legend(
                    handles=handles, loc="center", fontsize=6,
                    frameon=True, framealpha=0.8, edgecolor="lightgray",
                    handlelength=1.2, handleheight=1.0,
                )
        else:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cb = fig.colorbar(sm, ax=ax_cb, fraction=0.9, pad=0.05, aspect=20)
            cb.set_label(cbar_label, fontsize=6)
            cb.ax.tick_params(labelsize=5)

    if title:
        fig.suptitle(title, fontsize=11, y=1.01)

    plt.subplots_adjust(wspace=0.05, hspace=0.12)
    save_figure(fig, output_path, dpi=dpi)
    print(f"  Saved {mode} weights visualization to {output_path}")


def plot_attn_bias_debug(
    vis_bias_debug: list,
    input_images: list,
    input_masks: list,
    obj_idx: int,
    output_dir: str,
    scene_name: str,
    n_sample_voxels: int = 8,
) -> None:
    """Plot attention bias debug visualizations for ``canonical``.

    Produces ``{scene}_obj{i}_attn_bias_voxels.png`` — per-voxel attention
    maps for a spread of sample voxels (visible + occluded), with attention
    heatmaps overlaid on the input image using the ``hot`` colormap.

    Parameters
    ----------
    vis_bias_debug : list
        Per-hook dicts from ``stage2_mv`` with keys ``attn_before``,
        ``attn_after``, ``mask_down``, ``mask_down_full``, ``coords``.
    input_images : list
        Per-view ``(H, W, 3)`` uint8 images.
    input_masks : list
        Per-view ``(H, W)`` boolean masks.
    obj_idx : int
        Object index (for file naming).
    output_dir : str
        Directory to save PNGs (e.g. ``canonical/``).
    scene_name : str
        Scene name (for file naming / titles).
    n_sample_voxels : int
        Number of voxels to show in the per-voxel plot.
    """
    import matplotlib
    matplotlib.use("Agg")
    from scipy.ndimage import zoom as ndimage_zoom

    if not vis_bias_debug:
        return
    hook_data = vis_bias_debug[0]  # use first hooked layer

    attn_before = hook_data.get("attn_before")
    attn_after = hook_data.get("attn_after")
    mask_down_raw = hook_data.get("mask_down")      # list[Tensor] or Tensor or None
    mask_down_full_raw = hook_data.get("mask_down_full")  # list[Tensor] or Tensor or None
    coords = hook_data.get("coords")
    if attn_before is None or attn_after is None:
        return

    # Move to numpy
    def _to_np(x):
        if x is None:
            return None
        return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)

    attn_before = _to_np(attn_before)
    attn_after = _to_np(attn_after)
    # mask_down_raw may be list[Tensor] (per-view) or a single Tensor
    def _masks_to_np_list(raw):
        if raw is None:
            return None
        if isinstance(raw, list):
            return [_to_np(m) for m in raw]
        return [_to_np(raw)]  # wrap single tensor as 1-element list
    mask_down_list = _masks_to_np_list(mask_down_raw)
    mask_down_full_list = _masks_to_np_list(mask_down_full_raw)
    coords = _to_np(coords)
    _, L_down, P_total = attn_before.shape
    NUM_P = 1369  # 37 * 37
    GRID = 37

    # Reference image size
    _ref_img = input_images[0] if input_images else np.zeros((256, 256, 3), dtype=np.uint8)
    _H, _W = _ref_img.shape[:2]
    view_idx = 0

    # Select view-specific masks for visualization
    mask_down = mask_down_list[min(view_idx, len(mask_down_list) - 1)] if mask_down_list else None
    mask_down_full = (mask_down_full_list[min(view_idx, len(mask_down_full_list) - 1)]
                      if mask_down_full_list else None)

    # Each stream: (name, patch_start, patch_end, mask)
    # Cropped streams use mask_down, full-image streams use mask_down_full.
    _STREAMS_RAW = [
        ("Cropped Obj",  5,    5 + NUM_P,    mask_down),
        ("Full Scene",   1379, 1379 + NUM_P, mask_down_full),
        ("Cropped Mask", 2753, 2753 + NUM_P, mask_down),
        ("Full Mask",    4127, 4127 + NUM_P, mask_down_full),
    ]
    _STREAMS = [
        (n, s, e, m)
        for n, s, e, m in _STREAMS_RAW
        if e <= P_total
    ]

    # Extract per-stream attention: {name: (before, after)} each (N, L, 1369)
    stream_attn = {}
    for name, s, e, _ in _STREAMS:
        stream_attn[name] = (
            attn_before[:, :, s:e],
            attn_after[:, :, s:e],
        )

    # --- Prepare per-stream background images ---
    img_raw = (input_images[view_idx] if input_images
               else np.zeros((_H, _W, 3), dtype=np.uint8))
    if hasattr(img_raw, "numpy"):
        img_raw = img_raw.numpy()
    if img_raw.dtype != np.uint8 and img_raw.max() <= 1.0:
        img_raw = (np.clip(img_raw, 0, 1) * 255).astype(np.uint8)
    _obj_mask = None
    if input_masks and view_idx < len(input_masks):
        _obj_mask = input_masks[view_idx]
        if hasattr(_obj_mask, "numpy"):
            _obj_mask = _obj_mask.numpy()
        _obj_mask = _obj_mask.astype(bool)

    _bg_map = _prepare_stream_backgrounds(
        img_raw, _obj_mask if _obj_mask is not None else np.ones((_H, _W), dtype=bool),
        _H, _W,
    )
    # Map Stage 2 stream names to background keys
    _stream_to_bg = {
        "Cropped Obj":  "cropped_image",
        "Full Scene":   "full_image",
        "Cropped Mask": "cropped_mask",
        "Full Mask":    "full_mask",
    }

    cmap_hot = plt.cm.hot

    def _overlay(heatmap_grid, bg_img):
        blended, _ = _overlay_heatmap_on_bg(
            heatmap_grid, bg_img, GRID, GRID, cmap_hot)
        return blended

    os.makedirs(output_dir, exist_ok=True)

    # ===================================================================
    # PER-VOXEL PLOT: sample voxels with FPS
    # ===================================================================
    if coords is None or L_down == 0:
        return

    def _fps(pts, k):
        n = pts.shape[0]
        if k >= n:
            return np.arange(n)
        selected = [0]
        dists = np.full(n, np.inf)
        for _ in range(k - 1):
            last = pts[selected[-1]]
            d = np.linalg.norm(pts - last, axis=1)
            dists = np.minimum(dists, d)
            selected.append(int(np.argmax(dists)))
        return np.array(selected)

    # Occluded = no visible patches in ANY stream (union of both masks)
    has_cropped = mask_down.any(axis=1) if mask_down is not None else np.zeros(L_down, dtype=bool)
    has_full = mask_down_full.any(axis=1) if mask_down_full is not None else np.zeros(L_down, dtype=bool)
    is_occluded = ~(has_cropped | has_full)
    visible_idx = np.where(~is_occluded)[0]
    occluded_idx = np.where(is_occluded)[0]

    n_vis_pick = min(n_sample_voxels - min(1, len(occluded_idx)), len(visible_idx))
    sample_ids = []
    if n_vis_pick > 0 and len(visible_idx) > 0:
        fps_local = _fps(coords[visible_idx].astype(np.float64), n_vis_pick)
        sample_ids = visible_idx[fps_local].tolist()
    if len(occluded_idx) > 0:
        sample_ids.append(int(occluded_idx[0]))
    if not sample_ids:
        return

    n_vox = len(sample_ids)

    # Columns: Label | [Geom | Before | After] per stream
    n_cols_v = 1 + 3 * len(_STREAMS)
    fig, axes = plt.subplots(
        n_vox, n_cols_v, figsize=(2.5 * n_cols_v, 2.5 * n_vox),
        squeeze=False,
    )
    fig.suptitle(
        f"{scene_name} obj {obj_idx} — Per-Voxel Attention Bias "
        f"(view {view_idx})",
        fontsize=10,
    )

    for row, vid in enumerate(sample_ids):
        occ_str = "occluded" if is_occluded[vid] else "visible"
        xyz_str = f"({coords[vid, 0]},{coords[vid, 1]},{coords[vid, 2]})"

        # Col 0: label
        ax = axes[row, 0]
        ax.text(0.5, 0.5, f"voxel {vid}\n{xyz_str}\n{occ_str}",
                transform=ax.transAxes, fontsize=7, ha="center", va="center")
        ax.axis("off")
        if row == 0:
            ax.set_title("Voxel", fontsize=8)

        # Per-stream columns: [Geom | Before | After]
        for si, (sname, _, _, smask) in enumerate(_STREAMS):
            sb_s, sa_s = stream_attn[sname]
            vb = sb_s[view_idx, vid].reshape(GRID, GRID)
            va = sa_s[view_idx, vid].reshape(GRID, GRID)
            col_base = 1 + si * 3

            bg_key = _stream_to_bg.get(sname, "full_image")
            bg = _bg_map.get(bg_key, np.full((_H, _W, 3), 128, dtype=np.uint8))

            # Geometric mask: green overlay on stream background
            ax = axes[row, col_base]
            if smask is not None:
                mask_37 = smask[vid].reshape(GRID, GRID).astype(np.float32)
                mask_up = ndimage_zoom(mask_37, (_H / GRID, _W / GRID), order=0)
                bg_f = bg.astype(np.float32) / 255.0
                green = np.array([0, 1, 0])
                overlay = np.where(
                    mask_up[..., None] > 0.5,
                    0.5 * bg_f + 0.5 * green,
                    bg_f,
                )
                ax.imshow((np.clip(overlay, 0, 1) * 255).astype(np.uint8))
            else:
                ax.imshow(bg)
            if row == 0:
                ax.set_title(f"{sname}\nGeom", fontsize=6)
            ax.set_xticks([])
            ax.set_yticks([])

            # Before (attention without bias)
            ax = axes[row, col_base + 1]
            ax.imshow(_overlay(vb, bg))
            if row == 0:
                ax.set_title(f"{sname}\nBefore", fontsize=6)
            ax.set_xticks([])
            ax.set_yticks([])

            # After (attention with bias)
            ax = axes[row, col_base + 2]
            ax.imshow(_overlay(va, bg))
            if row == 0:
                ax.set_title(f"{sname}\nAfter", fontsize=6)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    voxel_path = os.path.join(
        output_dir, f"{scene_name}_obj{obj_idx}_attn_bias_voxels.png",
    )
    save_figure(fig, voxel_path)
    print(f"    Saved attention bias per-voxel: {voxel_path}")


def _plot_guidance_loss_components(ax, ts, components, title="Loss components"):
    """Loss-components-over-ODE-t panel.  ``components`` = list of
    ``(label, values, marker)``; the first entry is drawn bold (the total)."""
    for i, (label, values, marker) in enumerate(components):
        ax.plot(ts, values, f"{marker}-", label=label,
                linewidth=(2 if i == 0 else 1),
                alpha=(1.0 if i == 0 else 0.7), markersize=3)
    ax.set_xlabel("ODE time t")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_guidance_velocity_comparison(ax, ts, curves, title,
                                       ylabel="Velocity norm", legend_ncol=1):
    """Backbone-velocity-vs-guidance panel.  ``curves`` = list of dicts with
    keys ``label`` + ``values`` and optional ``linestyle`` ('-'/'--'),
    ``color``, ``alpha``, ``linewidth``.  Solid = backbone/applied velocity,
    dashed = guidance step, by convention."""
    for c in curves:
        ax.plot(ts, c["values"], c.get("linestyle", "-"),
                color=c.get("color"), label=c["label"],
                alpha=c.get("alpha", 1.0), linewidth=c.get("linewidth", 1.5),
                markersize=3)
    ax.set_xlabel("ODE time t")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if curves:
        ax.legend(fontsize=(6 if legend_ncol > 1 else 8), ncol=legend_ncol)
    ax.grid(True, alpha=0.3)


def plot_rendering_guidance_depth_error(
    loss_history: list,
    output_path: str,
    frame_indices: "list[int] | None" = None,
    error_key: str = "per_frame_depth_error",
    title: str = "Rendering guidance — per-pixel depth error",
    metrics: "list | None" = None,
) -> None:
    """Plot per-step per-frame pixel-wise error heatmaps.

    Produces a grid with ODE steps as rows and frames as columns.
    Each cell shows the absolute depth error masked to the object region.

    Parameters
    ----------
    loss_history : list of dict
        From the rendering-guidance ``loss_history``.  Each entry must
        have ``error_key`` (list of (H,W) numpy arrays or None), one per
        rendered frame.
    output_path : str
        Path to save the PNG.
    frame_indices : list of int, optional
        Frame labels for column headers.  If None, uses 0-based indices.
    error_key : str
        Key in each ``loss_history`` entry holding the per-frame error
        list.  Stage-2 calls this with ``per_frame_depth_error_gaussian``.
    title : str
        Figure suptitle.
    metrics : list of (key, label, cmap), optional
        When given, render one frames-wide column block per metric.  A string
        ``cmap`` makes an error block (scalar heatmap + shared colorbar); a
        ``None`` ``cmap`` makes an RGB-image block (e.g. rendered / GT views,
        shown as-is with no colorbar).  Defaults to a single depth block from
        ``error_key``.
    """
    import matplotlib
    matplotlib.use("Agg")

    if not loss_history:
        return

    # One or more metrics, each a frames-wide column block with its own
    # colorbar (depth and RGB live on different scales).  Default = depth only.
    if metrics is None:
        metrics = [(error_key, "Abs depth error", "magma")]

    # Keep steps where any metric has any non-None frame.
    steps = [
        e for e in loss_history
        if any(any(d is not None for d in (e.get(k) or [])) for k, _, _ in metrics)
    ]
    if not steps:
        return

    n_frames = 0
    for k, _, _ in metrics:
        lst = steps[0].get(k)
        if lst:
            n_frames = len(lst)
            break
    if n_frames == 0:
        return
    n_steps = len(steps)
    n_metrics = len(metrics)

    if frame_indices is None:
        frame_indices = list(range(n_frames))

    # Per-metric block layout.  Error metrics (cmap is a colormap name) get a
    # trailing colorbar column; image metrics (cmap is None → RGB thumbnail,
    # e.g. the rendered / GT views) are shown as-is with no colorbar.
    has_cbar = [cmap is not None for _, _, cmap in metrics]
    col_starts, width_ratios, acc = [], [], 0
    for cb in has_cbar:
        col_starts.append(acc)
        width_ratios += [1] * n_frames
        acc += n_frames
        if cb:
            width_ratios += [0.05]
            acc += 1
    total_cols = acc

    cell_h, cell_w = 2.0, 2.0
    fig, all_axes = plt.subplots(
        n_steps, total_cols,
        figsize=(cell_w * n_frames * n_metrics + 1.0 * sum(has_cbar) + 0.5,
                 cell_h * n_steps + 1.0),
        squeeze=False,
        gridspec_kw={"width_ratios": width_ratios},
    )

    for m, (key, label, cmap) in enumerate(metrics):
        col0 = col_starts[m]
        is_image = cmap is None
        # Per-metric global vmax for a consistent colorbar (error blocks only).
        vmax = 1.0
        if not is_image:
            vmax = 0.0
            for e in steps:
                for d in (e.get(key) or []):
                    if d is not None:
                        vmax = max(vmax, float(d.max()))
            if vmax < 1e-8:
                vmax = 1.0

        im = None
        for row, entry in enumerate(steps):
            vals = entry.get(key) or [None] * n_frames
            for f in range(n_frames):
                ax = all_axes[row, col0 + f]
                v = vals[f] if f < len(vals) else None
                if v is not None:
                    if is_image:
                        ax.imshow(v, interpolation="nearest")
                    else:
                        im = ax.imshow(v, cmap=cmap, vmin=0, vmax=vmax,
                                       interpolation="nearest")
                else:
                    ax.set_facecolor("0.9")
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(
                        f"{label}\nf{frame_indices[f]}" if f == 0
                        else f"f{frame_indices[f]}",
                        fontsize=8,
                    )
                if col0 + f == 0:
                    ax.set_ylabel(f"t={entry['t']:.2f}", fontsize=8)
        if has_cbar[m]:
            cb_axes = all_axes[:, col0 + n_frames].tolist()
            for a in cb_axes:
                a.axis("off")
            if im is not None:
                fig.colorbar(im, ax=cb_axes, shrink=0.9, label=label)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=120, bbox_inches=None)


def plot_appearance_rendering_guidance_loss(
    loss_history: list,
    output_path: str,
) -> None:
    """Plot Stage-2 appearance rendering-guidance diagnostics over ODE steps.

    Parameters
    ----------
    loss_history : list of dict
        From ``build_appearance_rendering_guidance_transform``'s
        ``loss_history``.  Each entry has keys: ``t``, ``total``,
        ``gaussian``, ``n_g``, ``grad_norm``, ``backbone_velocity_norm``,
        ``guidance_velocity_norm``, ``velocity_norm`` (and optionally
        ``skip_reason``).
    output_path : str
        Path to save the PNG.
    """
    import matplotlib
    matplotlib.use("Agg")

    if not loss_history:
        return

    ts = [e["t"] for e in loss_history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Loss (total = per-frame mean) + steps skipped on an empty decode.
    _plot_guidance_loss_components(
        axes[0], ts, [("Total", [e.get("total", 0.0) for e in loss_history], "o")])
    skipped_ts = [e["t"] for e in loss_history if "skip_reason" in e]
    for st in skipped_ts:
        axes[0].axvline(st, color="tab:red", alpha=0.3, linewidth=0.8)
    if skipped_ts:
        axes[0].plot([], [], color="tab:red", alpha=0.3,
                     label=f"empty decode (n={len(skipped_ts)})")
        axes[0].legend(fontsize=8)

    # Velocity decomposition:  v_post = v_backbone − wᵥ·∇ (wᵥ =
    # velocity_weight).  All three are velocity-space norms, so the gap between
    # the backbone and applied curves is the visible guidance effect and the
    # guidance-step curve is its magnitude.
    _curves = []
    if all("backbone_velocity_norm" in e for e in loss_history):
        _curves.append({"label": "Backbone velocity ‖v‖", "linestyle": "-",
                        "color": "tab:blue",
                        "values": [e["backbone_velocity_norm"] for e in loss_history]})
    _curves.append({"label": "Applied velocity ‖v − wᵥ∇‖ (post-guidance)",
                    "linestyle": "-", "color": "tab:orange", "linewidth": 2,
                    "values": [e.get("velocity_norm", 0.0) for e in loss_history]})
    _curves.append({"label": "Guidance step ‖wᵥ∇‖", "linestyle": "--",
                    "color": "tab:green",
                    "values": [e.get("guidance_velocity_norm", 0.0) for e in loss_history]})
    _plot_guidance_velocity_comparison(axes[1], ts, _curves, "Velocity decomposition")

    fig.suptitle("Appearance rendering guidance diagnostics", fontsize=12)
    fig.tight_layout()
    save_figure(fig, output_path, bbox_inches=None)


def save_decoded_visualizations(state, cfg, pipeline_obj, output_dir, tag,
                                frame_indices, canonical_only=False,
                                perframe_only=False,
                                voxel_color_mode="xyz"):
    """Decode and visualize per-frame and canonical Gaussians.

    Renders turntable views, PLY files, meshes, and SLAT voxel grids.
    Called after blocks that modify SLAT tokens to visualize the updated Gaussians.
    Gated by cfg.output flags: render_decoded, save_decoded_ply, save_decoded_meshes,
    save_slat_voxels.

    When ``canonical_only=True``, skips per-frame decoding/rendering and only
    renders canonical Gaussians + canonical SLAT voxels.

    When ``perframe_only=True``, skips canonical Gaussian rendering and only
    renders per-frame Gaussians + per-frame SLAT voxels.

    Parameters
    ----------
    voxel_color_mode : str
        Coloring for ``save_slat_voxel_mesh`` renders: ``"xyz"`` (default),
        ``"shape_pca"`` (PCA of shape tokens), or ``"slat_pca"`` (PCA of SLAT features).
    """
    from genia.core.utils.rendering import render_perframe_decoded, render_multiview_comparison
    from genia.core.utils.io_utils import save_perframe_ply, save_perframe_meshes
    from genia.core.utils.visualization import render_slat_voxel_mesh, visualize_slat_voxels

    if not (cfg.output.render_decoded or cfg.output.save_decoded_ply
            or cfg.output.save_decoded_meshes or cfg.output.save_slat_voxels
            or cfg.output.save_slat_voxel_mesh):
        return

    print(f"\nDecoding Gaussians for visualization ({tag})...")

    if not canonical_only:
        state.ensure_perframe_gaussians()

    renders_dir = os.path.join(output_dir, "decoded_renders")

    if cfg.output.render_decoded and not cfg.output.suppress_intermediate_renders:
        if not canonical_only and state.perframe_gaussians is not None:
            render_perframe_decoded(
                state.perframe_gaussians,
                output_dir=renders_dir,
                scene_name=cfg.dataset.scene_name, frame_indices=frame_indices,
                tag=tag, image_size=cfg.output.render_size,
                distance=cfg.output.render_distance, fov=cfg.output.render_fov,
            )
        # Per-frame-only runs leave the canonical store empty, so this skips
        # every canonical visualization.
        valid_gs = state.canonical_gaussians
        if valid_gs and not perframe_only:
            print(f"\n  Rendering canonical Gaussians ({tag})...")
            os.makedirs(renders_dir, exist_ok=True)
            for obj_idx in sorted(valid_gs.keys()):
                render_path = os.path.join(
                    renders_dir,
                    f"{cfg.dataset.scene_name}_obj{obj_idx}_canonical_{tag}.png",
                )
                print(f"  Rendering canonical object {obj_idx}...")
                render_multiview_comparison(
                    valid_gs[obj_idx],
                    mesh_path=None,
                    output_path=render_path,
                    image_size=cfg.output.render_size,
                    distance=cfg.output.render_distance,
                    fov=cfg.output.render_fov,
                )

        # Fallback: render voxel mesh for objects whose canonical
        # Gaussians were invalidated (None) by a shape update (shape
        # coords available but Gaussians not yet rebuilt).
        if not perframe_only and state.canonical_shape_coords:
            missing = (
                set(state.canonical_shape_coords.keys())
                - set(valid_gs.keys())
            )
            if missing:
                os.makedirs(renders_dir, exist_ok=True)
                for obj_idx in sorted(missing):
                    xyz = state.canonical_shape_coords[obj_idx]
                    canon_raw = state.canonical_raw_modalities.get(obj_idx, {})
                    shape_lat = canon_raw.get("shape")
                    canon_slat = state.canonical_slats.get(obj_idx)
                    # Downgrade colour mode when the requested data isn't
                    # available for this object — same pattern as the
                    # keyframes branch and per-frame loop below.
                    mode_i = voxel_color_mode
                    if mode_i == "slat_pca" and canon_slat is None:
                        mode_i = "shape_pca" if shape_lat is not None else "xyz"
                    elif mode_i == "shape_pca" and shape_lat is None:
                        mode_i = "xyz"
                    out_path = os.path.join(
                        renders_dir,
                        f"{cfg.dataset.scene_name}_obj{obj_idx}"
                        f"_canonical_{tag}_shape_voxels.png",
                    )
                    print(f"  Rendering shape voxels for object {obj_idx} "
                          f"({xyz.shape[0]} voxels)...")
                    render_slat_voxel_mesh(
                        xyz=xyz, output_path=out_path,
                        title=(f"{cfg.dataset.scene_name} obj {obj_idx} "
                               f"canonical shape"),
                        image_size=cfg.output.render_size,
                        distance=cfg.output.render_distance,
                        fov=cfg.output.render_fov,
                        save_obj=False,
                        color_mode=mode_i,
                        slat=canon_slat,
                        shape_latent=shape_lat,
                    )

        # Per-frame counterpart: render decoder_input["perframe_shape_coords"]
        # (populated by the parallel pass or per-frame GT shape injection).
        # Only when no per-frame SLAT is available — once Stage 2 produces SLATs,
        # the slat-based rendering further below takes over.
        if not canonical_only:
            for obj_idx in sorted(state.tokens_by_object.keys()):
                pf_items = [
                    (fid, di) for fid, di in state.tokens_by_object[obj_idx]
                    if di.get("perframe_shape_coords") is not None
                    and di.get("decoder_input_slat") is None
                ]
                if not pf_items:
                    continue
                os.makedirs(renders_dir, exist_ok=True)
                print(f"  Rendering per-frame shape voxels for object "
                      f"{obj_idx} ({len(pf_items)} frames)...")
                for fid, di in pf_items:
                    xyz = di["perframe_shape_coords"]
                    # decode_shape_to_coords returns (N, 4) int [batch, x, y, z];
                    # render_slat_voxel_mesh expects (N, 3) float numpy (matching
                    # canonical_shape_coords' storage form).
                    if hasattr(xyz, "cpu"):
                        xyz = xyz[:, 1:].cpu().numpy().astype(float)
                    shape_lat = di.get("raw_ss_modalities", {}).get("shape")
                    # Downgrade colour mode when the requested data isn't
                    # available for this frame (e.g. actionmesh's
                    # ``reset_shape_to_gt`` drops per-frame shape latents,
                    # and per-frame SLATs aren't built in actionmesh mode).
                    # Same pattern as the keyframes-rendering branch.
                    mode_i = voxel_color_mode
                    if mode_i == "slat_pca":
                        mode_i = "shape_pca" if shape_lat is not None else "xyz"
                    elif mode_i == "shape_pca" and shape_lat is None:
                        mode_i = "xyz"
                    _name_idx = fid.frame if hasattr(fid, "frame") else int(fid)
                    out_path = os.path.join(
                        renders_dir,
                        f"{cfg.dataset.scene_name}_obj{obj_idx}"
                        f"_f{_name_idx:04d}_{tag}_shape_voxels.png",
                    )
                    render_slat_voxel_mesh(
                        xyz=xyz, output_path=out_path,
                        title=(f"{cfg.dataset.scene_name} obj {obj_idx} "
                               f"frame {_name_idx} shape"),
                        image_size=cfg.output.render_size,
                        distance=cfg.output.render_distance,
                        fov=cfg.output.render_fov,
                        save_obj=False,
                        color_mode=mode_i,
                        shape_latent=shape_lat,
                    )

    if cfg.output.save_decoded_ply and not canonical_only:
        save_perframe_ply(
            state.perframe_gaussians, state.tokens_by_object,
            output_dir=os.path.join(output_dir, "decoded_plys"),
            scene_name=cfg.dataset.scene_name, tag=tag,
            compressed=cfg.output.save_compressed_ply,
        )
    if cfg.output.save_decoded_meshes and not canonical_only:
        save_perframe_meshes(
            state.tokens_by_object, pipeline=pipeline_obj,
            output_dir=os.path.join(output_dir, "decoded_meshes"),
            scene_name=cfg.dataset.scene_name, tag=tag,
        )
    if cfg.output.save_slat_voxels:
        voxels_dir = os.path.join(output_dir, "slat_voxels")
        os.makedirs(voxels_dir, exist_ok=True)
        if not canonical_only:
            for obj_idx in sorted(state.tokens_by_object.keys()):
                for frame_idx, decoder_input in state.tokens_by_object[obj_idx]:
                    slat = decoder_input["decoder_input_slat"]
                    out_path = os.path.join(
                        voxels_dir,
                        f"{cfg.dataset.scene_name}_obj{obj_idx}_f{frame_idx}_slat_pca.png",
                    )
                    visualize_slat_voxels(
                        slat, out_path,
                        title=f"{cfg.dataset.scene_name} obj {obj_idx} frame {frame_idx}",
                    )
        valid_slats = state.canonical_slats
        if valid_slats and not perframe_only:
            for obj_idx in sorted(valid_slats.keys()):
                slat = valid_slats[obj_idx]
                out_path = os.path.join(
                    voxels_dir,
                    f"{cfg.dataset.scene_name}_obj{obj_idx}_canonical_slat_pca.png",
                )
                visualize_slat_voxels(
                    slat, out_path,
                    title=f"{cfg.dataset.scene_name} obj {obj_idx} canonical",
                )
    if cfg.output.save_slat_voxel_mesh:
        voxels_dir = os.path.join(output_dir, "slat_voxels")
        os.makedirs(voxels_dir, exist_ok=True)
        if not canonical_only:
            for obj_idx in sorted(state.tokens_by_object.keys()):
                # Per-frame shape latent for shape_pca coloring
                pf_raw_obj = state.perframe_raw_modalities.get(obj_idx, {})
                for frame_idx, decoder_input in state.tokens_by_object[obj_idx]:
                    slat = decoder_input.get("decoder_input_slat")
                    if slat is None:
                        continue
                    pf_shape = None
                    if voxel_color_mode == "shape_pca":
                        pf_raw = pf_raw_obj.get(frame_idx, {})
                        raw_mods = pf_raw.get("raw_ss_modalities", {})
                        pf_shape = raw_mods.get("shape")
                    out_path = os.path.join(
                        voxels_dir,
                        f"{cfg.dataset.scene_name}_obj{obj_idx}_f{frame_idx}_slat_mesh.png",
                    )
                    render_slat_voxel_mesh(
                        slat, out_path,
                        title=f"{cfg.dataset.scene_name} obj {obj_idx} frame {frame_idx}",
                        image_size=cfg.output.render_size,
                        distance=cfg.output.render_distance,
                        fov=cfg.output.render_fov,
                        color_mode=voxel_color_mode,
                        shape_latent=pf_shape,
                    )
        valid_slats = state.canonical_slats
        if valid_slats and not perframe_only:
            for obj_idx in sorted(valid_slats.keys()):
                slat = valid_slats[obj_idx]
                canon_raw = state.canonical_raw_modalities.get(obj_idx, {})
                canon_shape = canon_raw.get("shape")
                out_path = os.path.join(
                    voxels_dir,
                    f"{cfg.dataset.scene_name}_obj{obj_idx}_canonical_slat_mesh.png",
                )
                render_slat_voxel_mesh(
                    slat, out_path,
                    title=f"{cfg.dataset.scene_name} obj {obj_idx} canonical",
                    image_size=cfg.output.render_size,
                    distance=cfg.output.render_distance,
                    fov=cfg.output.render_fov,
                    color_mode=voxel_color_mode,
                    shape_latent=canon_shape,
                )
