# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Gaussian splatting rendering utilities.

This module provides functions for rendering 3D Gaussian scenes to images
using differentiable rendering via gsplat, including multi-view rendering
and comparison visualizations.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import torch
from gsplat.rendering import rasterization

from .visualization import draw_text_overlay, save_figure

_IDENTITY_4x4 = np.eye(4, dtype=np.float32)


def bg_rgb_float(bg_color) -> np.ndarray:
    """Render background as a float ``(3,)`` in [0, 1].  ``None`` ⇒ black,
    matching every renderer in this module."""
    if bg_color is None:
        return np.zeros(3, dtype=np.float32)
    arr = (bg_color.detach().cpu().numpy()
           if hasattr(bg_color, "detach") else bg_color)
    return np.asarray(arr, dtype=np.float32).reshape(-1)[:3].clip(0.0, 1.0)


def foreground_union(masks, H: int, W: int):
    """Boolean ``(H, W)`` union of *masks*, nearest-resized when needed.

    *masks* is an iterable of per-object mask arrays, tolerating ``None``
    entries and a trailing channel axis.  Returns ``None`` when nothing usable
    is present, so callers can fall back rather than blank the frame.
    """
    from PIL import Image

    fg = np.zeros((H, W), dtype=bool)
    for m in masks:
        if m is None:
            continue
        m = np.asarray(m)
        if m.ndim == 3:
            m = m[..., 0]
        if m.shape[:2] != (H, W):
            m = np.array(
                Image.fromarray(m.astype("uint8") * 255).resize(
                    (W, H), Image.NEAREST,
                )
            ) > 127
        fg |= m.astype(bool)
    return fg if fg.any() else None


def composite_on_render_bg(image, masks, bg_color):
    """Replace *image*'s background with the flat background a render uses.

    Pipeline renders contain only the reconstructed objects on ``bg_color``.
    Comparing or displaying them against an image that still has its dataset
    background charges every background pixel to the error — swamping the
    object error in block metrics, and lighting up the whole ``|GT - Pred|``
    panel in the comparison videos.  Compositing the image onto the *same*
    background makes the two agree pixel-for-pixel outside the objects.

    Dataset-agnostic: the foreground is the union of *masks* (the objects
    actually rendered).  *image* may be uint8 or float in [0, 1]; the same
    dtype comes back.  Returns *image* unchanged when no usable mask is
    present — an unmasked panel beats a blank one.
    """
    fg = foreground_union(masks, image.shape[0], image.shape[1])
    if fg is None:
        return image

    bg = bg_rgb_float(bg_color)
    if np.issubdtype(np.asarray(image).dtype, np.integer):
        bg = (bg * 255.0).clip(0, 255).astype(image.dtype)
    out = np.broadcast_to(bg.astype(image.dtype), image.shape).copy()
    out[fg] = image[fg]
    return out


if TYPE_CHECKING:
    from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian.gaussian_model import (
        Gaussian,
    )


def render_gaussian_params(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    features: torch.Tensor,
    c2w: torch.Tensor,
    K_matrix: torch.Tensor | np.ndarray,
    W: int,
    H: int,
    bg_color: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Differentiable Gaussian rendering using gsplat.

    Parameters
    ----------
    means : torch.Tensor
        Gaussian positions, shape (N, 3).
    quats : torch.Tensor
        Gaussian rotations as quaternions (wxyz format), shape (N, 4).
    scales : torch.Tensor
        Gaussian scales, shape (N, 3).
    opacities : torch.Tensor
        Gaussian opacities, shape (N,) or (N, 1).
    features : torch.Tensor
        Gaussian colors/features, shape (N, 3) or (N, 1, 3).
    c2w : torch.Tensor
        Camera-to-world transformation matrix, shape (4, 4) or (1, 4, 4).
    K_matrix : torch.Tensor or np.ndarray
        Camera intrinsics, shape (3, 3) or (1, 3, 3).
    W : int
        Image width.
    H : int
        Image height.
    bg_color : torch.Tensor, optional
        Background color, shape (3,). Defaults to black [0, 0, 0].

    Returns
    -------
    tuple
        (rgb, alpha, depth) where:
        - rgb: Rendered image, shape (H, W, 3)
        - alpha: Alpha/opacity map, shape (H, W)
        - depth: Depth map, shape (H, W)

    Examples
    --------
    >>> rgb, alpha, depth = render_gaussian_params(
    ...     means, quats, scales, opacities, features,
    ...     c2w=torch.eye(4), K_matrix=K, W=640, H=480
    ... )
    >>> rgb.shape
    torch.Size([480, 640, 3])
    """
    device = means.device

    # View matrix: from camera to world
    if isinstance(c2w, np.ndarray):
        c2w_torch = torch.from_numpy(c2w).float().to(device)
    else:
        c2w_torch = c2w.float().to(device)
    w2c = torch.inverse(c2w_torch)

    if w2c.dim() == 2:
        w2c = w2c.unsqueeze(0)  # [1, 4, 4]

    # Intrinsics
    if isinstance(K_matrix, np.ndarray):
        K = torch.from_numpy(K_matrix).float().to(device)
    else:
        K = K_matrix.float().to(device)

    if K.dim() == 2:
        K = K.unsqueeze(0)  # [1, 3, 3]

    # Handle opacity shape
    if opacities.dim() == 2:
        opacities = opacities.squeeze(-1)  # (N,)

    # Handle features shape - gsplat expects (N, K, 3) where K = (sh_degree+1)^2
    if features.dim() == 2:
        features = features.unsqueeze(1)  # (N, 3) -> (N, 1, 3)

    # Auto-detect SH degree from features: K = (degree+1)^2
    sh_k = features.shape[1]
    sh_degree = int(sh_k**0.5) - 1

    # Default to black background
    if bg_color is None:
        bg_color = torch.zeros(3, device=device)
    else:
        bg_color = bg_color.to(device)

    # Expected depth: alpha-weighted mean ray depth, normalised by accumulated alpha.
    render_mode = "RGB+ED"

    # Render using gsplat
    rgbd, alpha, info = rasterization(
        means=means,  # (N, 3)
        quats=quats,  # (N, 4)
        scales=scales,  # (N, 3)
        opacities=opacities,  # (N,)
        colors=features,  # (N, 1, 3)
        viewmats=w2c,  # (1, 4, 4)
        Ks=K,  # (1, 3, 3)
        width=W,
        height=H,
        near_plane=0.1,
        far_plane=100000.0,
        render_mode=render_mode,
        sh_degree=sh_degree,
        rasterize_mode="classic",
        distributed=False,
        camera_model="pinhole",
        packed=False,
        backgrounds=bg_color[None, ...],
    )

    rgb = rgbd[0, ..., :3]  # (H, W, 3)
    depth = rgbd[0, ..., 3]  # (H, W)
    alpha = alpha[0, ..., 0]  # (H, W)

    return rgb, alpha, depth


def transform_gaussian_params_cam_to_world(
    xyz: torch.Tensor,
    quats: torch.Tensor,
    c2w: "torch.Tensor | np.ndarray",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Transform Gaussian positions and quaternions from R3 camera space to R3 world space.

    Parameters
    ----------
    xyz : torch.Tensor
        Gaussian positions in R3 camera space, shape ``(N, 3)``.
    quats : torch.Tensor
        Gaussian quaternions (wxyz) in R3 camera space, shape ``(N, 4)``.
    c2w : torch.Tensor or np.ndarray
        Camera-to-world transform, shape ``(4, 4)``.

    Returns
    -------
    xyz_world : torch.Tensor
        Positions in R3 world space, shape ``(N, 3)``.
    quats_world : torch.Tensor
        Quaternions in R3 world space, shape ``(N, 4)``.
    """
    from pytorch3d.transforms import matrix_to_quaternion, quaternion_multiply

    if isinstance(c2w, np.ndarray):
        if np.allclose(c2w, np.eye(4)):
            return xyz, quats
        c2w = torch.from_numpy(c2w).float().to(xyz.device)
    else:
        if torch.allclose(c2w.float(), torch.eye(4, device=c2w.device)):
            return xyz, quats
        c2w = c2w.float().to(xyz.device)

    R = c2w[:3, :3]  # (3, 3)
    t = c2w[:3, 3]   # (3,)

    # Positions: p_world = p_cam @ R^T + t  (row-vector convention, matches create_background_gaussians)
    xyz_world = xyz @ R.T + t

    # Quaternions: compose c2w rotation with per-Gaussian orientation.
    # matrix_to_quaternion returns q such that quaternion_to_matrix(q) = M.
    # gsplat interprets quaternions as column-vector rotations (R @ v), so we
    # pass R directly to get the quaternion for column-vector rotation R_c2w.
    c2w_quat = matrix_to_quaternion(R.unsqueeze(0)).squeeze(0)  # (4,)
    quats_world = quaternion_multiply(c2w_quat.unsqueeze(0).expand(quats.shape[0], -1), quats)

    return xyz_world, quats_world


def transform_gaussian_params_world_to_cam(
    xyz: torch.Tensor,
    quats: torch.Tensor,
    c2w: "torch.Tensor | np.ndarray",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Transform Gaussian positions and quaternions from R3 world space to R3 camera space.

    Inverse of :func:`transform_gaussian_params_cam_to_world`.

    Parameters
    ----------
    xyz : torch.Tensor
        Gaussian positions in R3 world space, shape ``(N, 3)``.
    quats : torch.Tensor
        Gaussian quaternions (wxyz) in R3 world space, shape ``(N, 4)``.
    c2w : torch.Tensor or np.ndarray
        Camera-to-world transform, shape ``(4, 4)``.

    Returns
    -------
    xyz_cam : torch.Tensor
        Positions in R3 camera space, shape ``(N, 3)``.
    quats_cam : torch.Tensor
        Quaternions in R3 camera space, shape ``(N, 4)``.
    """
    from pytorch3d.transforms import matrix_to_quaternion, quaternion_invert, quaternion_multiply

    if isinstance(c2w, np.ndarray):
        if np.allclose(c2w, np.eye(4)):
            return xyz, quats
        c2w = torch.from_numpy(c2w).float().to(xyz.device)
    else:
        if torch.allclose(c2w.float(), torch.eye(4, device=c2w.device)):
            return xyz, quats
        c2w = c2w.float().to(xyz.device)

    R = c2w[:3, :3]  # (3, 3)
    t = c2w[:3, 3]   # (3,)

    # Positions: p_cam = (p_world - t) @ R  (inverse of p_world = p_cam @ R^T + t)
    xyz_cam = (xyz - t) @ R

    # Quaternions: undo c2w rotation.
    # matrix_to_quaternion(R) gives the quaternion for column-vector R_c2w.
    # Invert it to get w2c rotation quaternion.
    c2w_quat = matrix_to_quaternion(R.unsqueeze(0)).squeeze(0)  # (4,)
    c2w_quat_inv = quaternion_invert(c2w_quat)
    quats_cam = quaternion_multiply(c2w_quat_inv.unsqueeze(0).expand(quats.shape[0], -1), quats)

    return xyz_cam, quats_cam


def render_gaussians_scene(
    scene_gs: "Gaussian",
    c2w: torch.Tensor,
    K: torch.Tensor,
    w: int,
    h: int,
    bg_color: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Render a single frame from the Gaussian scene using given camera parameters.

    Parameters
    ----------
    scene_gs : Gaussian
        Gaussian scene object.
    c2w : torch.Tensor
        Camera-to-world transformation matrix, shape (4, 4).
    K : torch.Tensor
        Camera intrinsics matrix, shape (3, 3).
    w : int
        Image width.
    h : int
        Image height.
    bg_color : torch.Tensor, optional
        Background color, shape (3,). Defaults to black.

    Returns
    -------
    tuple
        (rgb, alpha, depth) where:
        - rgb: Rendered image, shape (H, W, 3), in [0, 1] range
        - alpha: Alpha/opacity map, shape (H, W)
        - depth: Depth map, shape (H, W)
    """
    # Ensure tensors are on CUDA
    c2w = c2w.cuda() if not c2w.is_cuda else c2w
    Ks = K.cuda() if not K.is_cuda else K

    if c2w.dim() == 2:
        c2w = c2w.unsqueeze(0)  # [1, 4, 4]

    if Ks.dim() == 2:
        Ks = Ks.unsqueeze(0)  # [1, 3, 3]

    means = scene_gs.get_xyz  # [N, 3]
    rotations = scene_gs.get_rotation  # [N, 4]
    scales = scene_gs.get_scaling  # [N, 3]
    opacity = scene_gs.get_opacity  # [N, 1]
    features = scene_gs.get_features  # [N, 1, 3]
    width = w
    height = h

    # Set background color (default to black if not provided)
    if bg_color is None:
        bg_color = torch.zeros(3, device=c2w.device)
    else:
        bg_color = bg_color.to(c2w.device)

    rgb, alpha, depth = render_gaussian_params(
        means=means,
        quats=rotations,
        scales=scales,
        opacities=opacity,
        features=features,
        c2w=c2w,
        K_matrix=Ks,
        W=width,
        H=height,
        bg_color=bg_color,
    )

    return rgb, alpha, depth


def render_gaussians_to_image(
    scene_gs: "Gaussian",
    K_matrix: np.ndarray,
    W: int,
    H: int,
    bg_color: Optional[torch.Tensor] = None,
    c2w: np.ndarray = _IDENTITY_4x4,
    return_alpha: bool = False,
) -> "torch.Tensor | Tuple[torch.Tensor, torch.Tensor]":
    """
    Render Gaussian scene to an image.

    Parameters
    ----------
    scene_gs : Gaussian
        Gaussian scene object.
    K_matrix : np.ndarray
        Camera intrinsics matrix, shape (3, 3).
    W : int
        Image width.
    H : int
        Image height.
    bg_color : torch.Tensor, optional
        Background color, shape (3,). Defaults to black.
    c2w : np.ndarray, optional
        Camera-to-world transform, shape (4, 4). Defaults to identity
        (camera at origin).
    return_alpha : bool, optional
        When True, also return the accumulated coverage map so callers can
        save foreground renders with a transparent background (RGBA PNG)
        instead of relying on ``bg_color`` to fill empty pixels.

    Returns
    -------
    torch.Tensor
        Rendered image, shape (H, W, 3), in [0, 1] range. When
        ``return_alpha=True``, a ``(rgb, alpha)`` tuple is returned instead,
        with ``alpha`` of shape (H, W) in [0, 1].

    Examples
    --------
    >>> rendered = render_gaussians_to_image(scene_gs, K_matrix, 640, 480)
    >>> rendered.shape
    torch.Size([480, 640, 3])
    """
    c2w_tensor = torch.from_numpy(c2w).float()
    K = torch.from_numpy(K_matrix).float()

    # Default to black background for evaluation
    if bg_color is None:
        bg_color = torch.zeros(3)

    rendered_frame, alpha, _ = render_gaussians_scene(
        scene_gs, c2w=c2w_tensor, K=K, w=W, h=H, bg_color=bg_color
    )

    if return_alpha:
        return rendered_frame, alpha
    return rendered_frame


def render_gaussian_from_view(
    gaussian: "Gaussian",
    R: torch.Tensor,
    T: torch.Tensor,
    image_size: int = 512,
    fov: float = 60.0,
) -> np.ndarray:
    """
    Render Gaussian from a specific viewpoint using PyTorch3D camera convention.

    Parameters
    ----------
    gaussian : Gaussian
        Gaussian splatting model.
    R : torch.Tensor
        Rotation matrix from look_at_view_transform, shape (1, 3, 3).
    T : torch.Tensor
        Translation vector from look_at_view_transform, shape (1, 3).
    image_size : int
        Output image size (square).
    fov : float
        Field of view in degrees.

    Returns
    -------
    np.ndarray
        Rendered image, shape (H, W, 3), values in [0, 1].
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create camera-to-world matrix
    # PyTorch3D uses row-major convention
    w2c = torch.eye(4, device=device)
    w2c[:3, :3] = R[0].T  # Transpose because PyTorch3D convention
    w2c[:3, 3] = T[0]
    c2w = torch.inverse(w2c)

    # Create intrinsics (simple perspective)
    focal_length = image_size / (2 * np.tan(np.radians(fov) / 2))
    K = torch.eye(3, device=device)
    K[0, 0] = focal_length
    K[1, 1] = focal_length
    K[0, 2] = image_size / 2
    K[1, 2] = image_size / 2

    # Render using the utility function with white background
    white_bg = torch.ones(3, device=device)
    rendered, alpha, _ = render_gaussians_scene(
        gaussian,
        c2w=c2w,
        K=K,
        w=image_size,
        h=image_size,
        bg_color=white_bg
    )

    return rendered.cpu().numpy()


def create_comparison_grid(
    gaussian_renders: Dict[str, np.ndarray],
    mesh_renders: Dict[str, np.ndarray],
    view_names: List[str],
) -> plt.Figure:
    """
    Create a grid showing Gaussian and Mesh renders side by side.

    Parameters
    ----------
    gaussian_renders : dict
        Maps view_name -> rendered image as np.ndarray.
    mesh_renders : dict
        Maps view_name -> rendered image as np.ndarray.
    view_names : list
        List of view names to display.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with the comparison grid.
    """
    n_views = len(view_names)
    has_mesh = bool(mesh_renders)
    n_rows = 2 if has_mesh else 1

    fig = plt.figure(figsize=(4 * n_views, 4 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_views, figure=fig, hspace=0.05, wspace=0.05)

    for col, view_name in enumerate(view_names):
        # Gaussian render
        ax_gaussian = fig.add_subplot(gs[0, col])
        im = gaussian_renders[view_name]
        # clip values to [0, 1] for display
        im = np.clip(im, 0.0, 1.0)
        ax_gaussian.imshow(im)
        ax_gaussian.axis('off')
        if col == 0 and has_mesh:
            ax_gaussian.set_ylabel('Gaussians', fontsize=16, rotation=0, labelpad=60, va='center')
        ax_gaussian.set_title(view_name.capitalize(), fontsize=14)

        # Mesh render
        if has_mesh:
            ax_mesh = fig.add_subplot(gs[1, col])
            im = mesh_renders[view_name]
            # clip values to [0, 1] for display
            im = np.clip(im, 0.0, 1.0)
            ax_mesh.imshow(im)
            ax_mesh.axis('off')
            if col == 0:
                ax_mesh.set_ylabel('Mesh', fontsize=16, rotation=0, labelpad=60, va='center')

    return fig


def bulk_keep_mask(pts: torch.Tensor, factor: float = 1.5) -> torch.Tensor:
    """Boolean mask keeping a point cloud's bulk and dropping the stray tail.

    Everything past ``factor`` times the 95th-percentile radius about the cloud's MEDIAN.
    A depth-unprojected reconstruction can strand splats well behind the object, which
    are harmless from the input view but wreck anything that measures the cloud's centre
    or extent.  Keying the cutoff on p95 is what stops it eating real geometry: coherent
    reconstructions sit at ``max/p95 <= 1.25``, so nothing is dropped, while a second
    cluster holding more than 5% of the cloud drags p95 out to enclose itself.

    At least 95% of points always survive, so the result is never empty for finite input.
    """
    d = (pts - pts.median(dim=0).values).norm(dim=-1)
    return d <= torch.quantile(d, 0.95) * factor


def _weighted_quantiles(x: torch.Tensor, w: torch.Tensor, qs) -> torch.Tensor:
    """Values below which each fraction in ``qs`` of the WEIGHT of ``x`` lies.

    Nearest-rank, and all of ``qs`` off ONE sort -- the sort is the whole cost, and
    the caller always wants a matched pair.
    """
    order = torch.argsort(x)
    xs = x[order]
    c = torch.cumsum(w[order], dim=0)
    c = c / c[-1].clamp_min(1e-12)
    idx = torch.searchsorted(c, torch.as_tensor(qs, device=c.device, dtype=c.dtype))
    return xs[idx.clamp(max=len(xs) - 1)]


def visible_bulk_mask(means: torch.Tensor, opacities: torch.Tensor,
                      scales: torch.Tensor, q: float = 0.999) -> torch.Tensor:
    """Keep the splats holding fraction ``q`` of the cloud's VISIBLE mass, per axis.

    The companion to :func:`bulk_keep_mask`, for the failure it cannot see.  That
    one cuts on radius about the median, so it fires on a second CLUSTER but not on
    a diffuse HAZE of faint stray splats, which pushes p95 out with it while
    inflating the object's box and dragging its centre off the object.

    What separates such a haze is not distance but VISIBILITY: its splats are far
    fainter than the bulk.  A splat you cannot see should not decide where the
    object is or how large it is, so each splat is weighted by
    ``opacity x footprint`` (the product of its two largest axes, i.e. its on-screen
    area at best orientation) and the box is the weighted 1-q..q quantile on each
    axis.  Scale-free, so it needs no absolute opacity threshold.

    ``q=0.999`` is a near no-op on a coherent reconstruction and decisive on an
    incoherent one; a lower ``q`` starts cropping real geometry.
    """
    w = opacities.reshape(-1).clamp_min(0)
    # A splat's footprint, not its volume: the smallest axis is its thickness, which
    # does not affect how much of the frame it covers.
    w = w * scales.sort(dim=1).values[:, 1:].prod(dim=1)
    keep = torch.ones(len(means), dtype=torch.bool, device=means.device)
    if float(w.sum()) <= 0:            # no opacity recorded -- nothing to weigh by
        return keep
    for a in range(3):
        lo, hi = _weighted_quantiles(means[:, a], w, (1.0 - q, q))
        keep &= (means[:, a] >= lo) & (means[:, a] <= hi)
    return keep


def oversize_splat_mask(means: torch.Tensor, scales: torch.Tensor,
                        factor: float = 0.06) -> torch.Tensor:
    """Drop splats whose own radius is a large fraction of the whole OBJECT.

    The failure neither :func:`bulk_keep_mask` nor :func:`visible_bulk_mask` can
    see, because it is not about position or visibility: a few fully opaque splats
    at the object's own depth whose radius is many times the median.  They render
    as big ellipses pasted over the object, and their radius enters the framing fit
    and pushes the object smaller as well.

    Judged as a fraction of the cloud's EXTENT rather than a multiple of the median
    radius, because the multiple does not separate: a clean reconstruction can reach
    over 10x its own median splat radius.

    ``factor=0.06`` is a trade rather than a clean split -- the largest splats of a
    coherent reconstruction can approach this fraction too -- chosen so that it
    drops only a handful of splats out of a clean cloud while catching most
    oversized ones.

    The extent is taken over splat CENTRES, not centres plus radii: the giant splats
    are centred on the object, so letting their radii into the measurement would
    inflate the very scale that judges them.  It should also be measured on a cloud
    the other two cuts have already cleaned -- a haze still in it inflates the
    extent and raises this bar.
    """
    lo, hi = means.min(dim=0).values, means.max(dim=0).values
    extent = float((hi - lo).norm())
    if extent <= 0:
        return torch.ones(len(means), dtype=torch.bool, device=means.device)
    return 3.0 * scales.max(dim=1).values <= factor * extent


def connected_bulk_mask(means: torch.Tensor, opacities: torch.Tensor,
                        scales: torch.Tensor, voxel: float = 0.01,
                        min_mass: float = 0.01) -> torch.Tensor:
    """Keep the connected components holding at least ``min_mass`` of visible mass.

    The failure the other three cuts cannot touch: a compact CLUMP of ordinary
    splats stranded off the body.  Every splat in it is normally sized, normally
    opaque and near the object, so it is not far, not faint and not oversized --
    but together they render as a solid blob beside the reconstruction, and they
    drag the framing box out with them.

    Points are voxelised at ``voxel`` times the cloud's extent and labelled
    26-connected; a component is kept when its share of ``opacity x footprint``
    reaches ``min_mass``.  Mass rather than count, so a dense speck cannot outvote a
    diffuse limb.  A coherent reconstruction is ONE component, so the cut cannot
    fire on it, and the threshold is not delicate.

    ``min_mass`` is a floor rather than "keep the largest" so a genuinely two-part
    object keeps both parts.
    """
    from scipy import ndimage

    m = means.detach().cpu().numpy().astype(np.float64)
    lo, hi = m.min(axis=0), m.max(axis=0)
    extent = float(np.linalg.norm(hi - lo))
    keep_all = torch.ones(len(means), dtype=torch.bool, device=means.device)
    if extent <= 0:
        return keep_all
    idx = np.floor((m - lo) / (voxel * extent)).astype(np.int64)
    grid = np.zeros(idx.max(axis=0) + 1, dtype=bool)
    grid[tuple(idx.T)] = True
    labels, n = ndimage.label(grid, structure=np.ones((3, 3, 3)))
    if n <= 1:
        return keep_all
    comp = labels[tuple(idx.T)]
    w = (opacities.reshape(-1).clamp_min(0)
         * scales.sort(dim=1).values[:, 1:].prod(dim=1)).detach().cpu().numpy()
    total = w.sum()
    if total <= 0:
        return keep_all
    share = np.bincount(comp, weights=w, minlength=n + 1) / total
    return torch.as_tensor(share[comp] >= min_mass, device=means.device)


def auto_frame_canonical(xyz: torch.Tensor, fov: float) -> "Tuple[np.ndarray, float]":
    """``(look_at, distance)`` framing a canonical asset wherever it happens to sit.

    The six canonical views orbit their look-at point at a fixed distance, which frames the
    object only when the object is at that point at ~unit scale.  That holds for a decoded
    SLAT canonical and not for a canonical that carries the object's metric placement (see
    ``PipelineState.pose_carries_placement``).  This is the framing such assets get
    instead; shared by the Gaussian and mesh
    ``decoded_renders/`` grids, which write the same filename and are meant to be directly
    comparable.

    ``radius / tan(0.7 * fov/2)`` uses ~70% of the half-FoV so a narrow-FoV config does not
    clip, and matches the fixed default distance: a radius-0.5 canonical (the
    normalized, object-local case) maps to distance 2.005 against ``render_distance=2.0``.

    Centre is the MEDIAN with a stray-splat trim, not the mean: a depth-unprojected
    asset can carry splats stranded behind the object, which drag a mean centroid and
    blow up the radius.  The cut is :func:`bulk_keep_mask`, shared with
    ``render_final_results.trim_outliers``.
    """
    pts = xyz.detach().reshape(-1, 3).float()
    pts = pts[bulk_keep_mask(pts)]
    centre = pts.median(dim=0).values
    # A degenerate asset (one splat, or all-coincident centres) would put the camera AT the
    # object; keep it outside so `look_at_view_transform` gets a real baseline.
    radius = max(float((pts - centre).norm(dim=-1).max()), 1e-6)
    distance = radius / float(np.tan(0.7 * np.radians(fov) * 0.5))
    return centre.cpu().numpy(), distance


def render_multiview_comparison(
    gaussian: "Gaussian",
    mesh_path: Optional[str],
    output_path: str,
    image_size: int = 512,
    distance: Optional[float] = 2.0,
    fov: float = 60.0,
) -> None:
    """
    Render Gaussian and Mesh from multiple viewpoints and create comparison grid.

    Parameters
    ----------
    gaussian : Gaussian
        Decoded Gaussian model.
    mesh_path : str or None
        Path to saved mesh .obj file, or None to skip mesh rendering.
    output_path : str
        Path to save the comparison image.
    image_size : int
        Size of rendered images (square).
    distance : float or None
        Camera distance, with the cameras orbiting the ORIGIN.  ``None`` auto-frames on the
        asset instead (:func:`auto_frame_canonical`) — for a reconstruction that bakes its
        placement into the canonical rather than into the Sim(3), which the caller detects
        with ``PipelineState.pose_carries_placement``.
    fov : float
        Field of view in degrees.
    """
    from .mesh_rendering import PyTorch3DMeshRenderer

    # Create mesh renderer wrapper
    mesh_renderer = PyTorch3DMeshRenderer()
    view_names = ['front', 'back', 'left', 'right', 'top', 'bottom']

    at = (0.0, 0.0, 0.0)
    if distance is None:
        at, distance = auto_frame_canonical(gaussian.get_xyz, fov)

    # Get camera viewpoints using the wrapper
    views = mesh_renderer.get_camera_positions(distance=distance, at=at)

    # Render Gaussian from all views
    print("    Rendering Gaussian from multiple viewpoints...")
    gaussian_renders = {}
    for view_name in view_names:
        R, T = views[view_name]
        rendered = render_gaussian_from_view(
            gaussian, R, T,
            image_size=image_size,
            fov=fov
        )
        gaussian_renders[view_name] = rendered

    # Render mesh from all views if available
    mesh_renders = {}
    if mesh_path and os.path.exists(mesh_path):
        print("    Rendering Mesh from multiple viewpoints...")
        try:
            # Load mesh with vertex colors using the wrapper
            mesh = mesh_renderer.load_mesh_with_vertex_colors(mesh_path)

            # Render all views using the wrapper
            # Same (at, distance) as the Gaussian row above, or the two rows of the
            # comparison grid frame the object differently.
            mesh_renders = mesh_renderer.render_all_views(
                mesh, view_names,
                distance=distance,
                image_size=image_size,
                fov=fov,
                at=at,
            )
        except Exception as e:
            print(f"    Error rendering mesh: {e}")
            # Create blank renders
            for view_name in view_names:
                mesh_renders[view_name] = np.ones((image_size, image_size, 3))

    # Create comparison grid and save
    print("    Creating comparison visualization...")
    fig = create_comparison_grid(gaussian_renders, mesh_renders, view_names)
    save_figure(fig, output_path)
    print(f"    Saved render: {output_path}")


def render_canonical_motion_video(
    canonical_gs: "Gaussian",
    output_path: str,
    frame_indices: "List[int]",
    canonical_mesh_verts: torch.Tensor,
    per_frame_mesh_verts: "Dict[int, torch.Tensor]",
    per_frame_mesh_rotations: "Dict[int, torch.Tensor]",
    canonical_mesh_faces: "Optional[torch.Tensor]" = None,
    *,
    image_size: int = 512,
    distance: "Optional[float]" = 2.0,
    fov: float = 60.0,
    duration: int = 100,
    K_knn: int = 4,
    knn_eps: float = 1.0e-8,
    knn_chunk_size: int = 131072,
) -> None:
    """6-view turntable animation of the canonical Gaussians warped per frame.

    Stitched as MP4.  Used by the FINAL block on actionmesh scenes where
    ``state.canonical_mesh_per_frame_verts`` is populated; static scenes
    use the turntable PNG from :func:`render_multiview_comparison`
    instead.  Cameras match that PNG's six standard viewpoints
    (front / back / left / right / top / bottom) so the static and
    dynamic outputs are directly comparable; per-frame layout is a 2×3
    grid (top row: front / back / left, bottom row: right / top / bottom).
    Frames missing from the deformation field fall back to canonical
    (rest-pose) Gaussians.

    ``distance=None`` auto-frames on the asset (:func:`auto_frame_canonical`) instead of
    orbiting the origin — for a reconstruction whose placement lives in the canonical
    rather than in the Sim(3); the caller decides with
    ``PipelineState.pose_carries_placement``.  The framing is computed ONCE from the
    rest-pose canonical and reused for every frame: re-deriving it per frame would make the
    camera chase the object, cancelling the deformation's residual translation and
    shimmering on centroid noise.
    """
    from PIL import Image
    from .gaussian import create_gaussians_object
    from .interpolation import _save_frames_as_video
    from .mesh_rendering import PyTorch3DMeshRenderer
    from genia.core.utils.deformation import warp_gaussians_high_res

    mesh_renderer = PyTorch3DMeshRenderer()
    view_names = ["front", "back", "left", "right", "top", "bottom"]
    canon_xyz = canonical_gs.get_xyz          # computed property: materialise once
    at = (0.0, 0.0, 0.0)
    if distance is None:
        at, distance = auto_frame_canonical(canon_xyz, fov)
    cameras = mesh_renderer.get_camera_positions(distance=distance, at=at)
    device = canon_xyz.device
    canon_verts_d = canonical_mesh_verts.to(device)
    faces_d = (
        canonical_mesh_faces.to(device).long()
        if canonical_mesh_faces is not None else None
    )

    print(
        f"    Rendering canonical motion video ({len(frame_indices)} frames, "
        f"6-view 2x3 grid)..."
    )
    frames: "List[Image.Image]" = []
    for fi in frame_indices:
        fi_int = fi.frame if hasattr(fi, "frame") else int(fi)
        pf_v = per_frame_mesh_verts.get(fi_int)
        pf_R = per_frame_mesh_rotations.get(fi_int)
        if pf_v is None or pf_R is None:
            warped_gs = canonical_gs
        else:
            with torch.no_grad():
                means_w, quats_w = warp_gaussians_high_res(
                    canonical_gs, canon_verts_d,
                    pf_v.to(device), pf_R.to(device),
                    K=int(K_knn), eps=float(knn_eps),
                    chunk_size=int(knn_chunk_size),
                    faces=faces_d,
                )
            warped_gs = create_gaussians_object(
                xyz=means_w, features=canonical_gs.get_features,
                scales=canonical_gs.get_scaling, rots=quats_w,
                opacities=canonical_gs.get_opacity,
            )

        # Render all 6 views as uint8 with a per-view label tag.
        view_imgs: "List[np.ndarray]" = []
        for vn in view_names:
            R_v, T_v = cameras[vn]
            rendered = render_gaussian_from_view(
                warped_gs, R_v, T_v, image_size=image_size, fov=fov,
            )
            img = (np.clip(rendered, 0.0, 1.0) * 255).astype(np.uint8)
            draw_text_overlay(img, vn.capitalize(), (10, 28), font_scale=0.6)
            view_imgs.append(img)

        # 2×3 grid via numpy concat (faster than matplotlib per frame).
        top = np.concatenate(view_imgs[:3], axis=1)             # (H, 3W, 3)
        bottom = np.concatenate(view_imgs[3:], axis=1)          # (H, 3W, 3)
        grid = np.concatenate([top, bottom], axis=0)            # (2H, 3W, 3)
        draw_text_overlay(
            grid, f"Frame {fi}", (12, 2 * image_size - 14),
            font_scale=0.7, thickness=2,
        )
        frames.append(Image.fromarray(grid))

    _save_frames_as_video(frames, output_path, duration=duration)
    print(f"    Saved canonical motion video: {output_path}")


def render_perframe_decoded(
    perframe_gaussians,
    output_dir,
    scene_name,
    frame_indices,
    tag="perframe",
    image_size=512,
    distance=2.0,
    fov=60.0,
):
    """Render decoded per-frame Gaussians from multiple viewpoints and create MP4 videos.

    For each object and frame, renders a multi-view comparison grid image.
    Then creates an MP4 video per object animating across frames.

    Parameters
    ----------
    perframe_gaussians : dict
        Nested dict: ``{obj_idx: {FrameKey or int: Gaussian}}``.
    output_dir : str
        Directory to save rendered PNGs and MP4 videos.
    scene_name : str
        Scene name for filenames.
    frame_indices : list of int or FrameKey
        Frames to include in the MP4 sequence (for time ordering).
    tag : str, optional
        Tag for filenames (e.g., "initial", "perframe"). Default: "perframe".
    image_size : int, optional
        Render resolution (square). Default: 512.
    distance : float, optional
        Camera distance from object. Default: 2.0.
    fov : float, optional
        Field of view in degrees. Default: 60.0.
    """
    print("\n" + "-" * 40)
    print(f"Rendering decoded per-frame Gaussians ({tag})...")
    print("-" * 40)

    def _frame_value(k):
        """Time-axis coordinate (FrameKey.frame or bare int)."""
        return k.frame if hasattr(k, "frame") else int(k)

    def _view_value(k):
        """View-axis coordinate (FrameKey.view or 0 for bare int)."""
        return k.view if hasattr(k, "view") else 0

    # Per-view subdir injection rule: only when multiple distinct views
    # appear in `frame_indices`. Single-view (mono pipelines) preserves
    # byte-identical PNG/MP4 filenames.
    distinct_views = {_view_value(fk) for fk in frame_indices}
    use_per_view = len(distinct_views) > 1

    os.makedirs(output_dir, exist_ok=True)
    render_paths_by_object = {}

    for obj_idx in sorted(perframe_gaussians.keys()):
        render_paths_by_object[obj_idx] = {}
        frame_gaussians = perframe_gaussians[obj_idx]
        for frame_key, gaussian in frame_gaussians.items():
            fi = _frame_value(frame_key)
            vi = _view_value(frame_key)
            # PNG filename: subdir per view when multi-view (avoids the
            # filename collision on MV-static, where every key has frame=0).
            png_dir = (os.path.join(output_dir, f"view{vi:02d}")
                       if use_per_view else output_dir)
            os.makedirs(png_dir, exist_ok=True)
            render_path = os.path.join(
                png_dir,
                f"{scene_name}_obj{obj_idx}_f{fi}_{tag}.png"
            )
            print(f"  Rendering object {obj_idx} frame_key {frame_key}...")
            render_multiview_comparison(
                gaussian,
                mesh_path=None,
                output_path=render_path,
                image_size=image_size,
                distance=distance,
                fov=fov,
            )
            render_paths_by_object[obj_idx][frame_key] = render_path

    # Create MP4 videos for each object (skip for single frame — PNGs suffice).
    # MV-static (multi-view, single timestep) → emit one PNG per view (already
    # written above), no MP4 since there's nothing to animate per view.
    # Mono-dynamic (single view, multi-frame) → one MP4 per object across time.
    # MV-dynamic (multi-view, multi-frame) → one MP4 per (object, view).
    if len(frame_indices) > 1:
        from genia.core.utils.interpolation import _save_frames_as_video
        from PIL import Image

        # Bucket FrameKeys by view. Each bucket plays as one MP4 (per object
        # per view); frames within a bucket are ordered by time.
        from genia.core.utils.frame_key import group_by_view
        keys_by_view = group_by_view(frame_indices)

        for obj_idx, render_paths in render_paths_by_object.items():
            for view_id, view_keys in keys_by_view.items():
                if len(view_keys) <= 1:
                    # Single timestep in this view — PNG already written;
                    # skip MP4 (would be a 1-frame video).
                    continue

                frames = []
                img_size = None
                for frame_key in view_keys:
                    path = render_paths.get(frame_key)
                    if path and os.path.exists(path):
                        with Image.open(path) as img:
                            if img_size is None:
                                img_size = img.size
                            frame = img.convert("RGB").copy()
                    else:
                        if img_size is not None:
                            frame = Image.new("RGB", img_size, (255, 255, 255))
                        else:
                            continue
                    fi = _frame_value(frame_key)
                    frame_np = np.ascontiguousarray(np.array(frame))
                    draw_text_overlay(
                        frame_np, f"Frame {fi}", (12, 30), font_scale=0.65,
                    )
                    frames.append(Image.fromarray(frame_np))

                if not frames:
                    continue
                # Path layout: subdir per view when multi-view, flat for mono.
                if use_per_view:
                    mp4_dir = os.path.join(output_dir, f"view{view_id:02d}")
                    os.makedirs(mp4_dir, exist_ok=True)
                    video_path = os.path.join(
                        mp4_dir, f"{scene_name}_obj{obj_idx}_{tag}.mp4",
                    )
                else:
                    video_path = os.path.join(
                        output_dir, f"{scene_name}_obj{obj_idx}_{tag}.mp4",
                    )
                _save_frames_as_video(frames, video_path)
                


__all__ = [
    "render_gaussian_params",
    "render_gaussians_scene",
    "render_gaussians_to_image",
    "render_gaussian_from_view",
    "create_comparison_grid",
    "render_multiview_comparison",
    "render_perframe_decoded",
]
