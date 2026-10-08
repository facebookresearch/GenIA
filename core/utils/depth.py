# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Depth processing utilities for the SAM3D-Objects pipeline.

This module provides functions for depth map processing, pointmap generation,
and coordinate system transformations between R3 and PyTorch3D conventions.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
from .quaternion_ops import r3_to_p3d_positions


def pinhole_xy(u, v, z, K):
    """Camera-space ``x, y`` from pixel coordinates and z-depth -- the ONE pinhole
    unprojection in this codebase.

    Elementwise, so it takes numpy or torch and stays differentiable w.r.t. ``z`` -- which
    is what lets a RENDERED depth map be unprojected inside a pose-optimisation loop by the
    same code that unprojects an observed one.

    Pixel coordinates are INTEGER (no +0.5 half-pixel offset).  It is the convention the
    GT clouds, the rendered clouds and every stored pointmap share, and changing it would
    shift all of them by half a pixel at once.
    """
    return (u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1]


def unproject_z_depth(depth, keep, K, uu, vv):
    """``(M, 3)`` camera-space points from a z-depth map over the ``keep`` mask.

    The sparse counterpart of :func:`depth_to_pointmap`, sharing its arithmetic through
    :func:`pinhole_xy` so the two cannot drift.  Equal to the dense route to the last bit
    in float64.

    ``uu``/``vv`` are the pixel-centre grids, passed in rather than built here because they
    are loop-invariant and the pose-optimisation callers hoist them out of a
    several-hundred-iteration loop.

    R3 camera convention, matching ``FrameData.pointmap``, so observed and rendered clouds
    land in the same frame with no further transform.
    """
    z = depth[keep]
    x, y = pinhole_xy(uu[keep], vv[keep], z, K)
    if isinstance(z, np.ndarray):
        return np.stack([x, y, z], axis=-1)
    return torch.stack([x, y, z], dim=-1)


def depth_to_pointmap(
    depth_map: np.ndarray,
    K: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Convert depth map to 3D pointmap using camera intrinsics.

    Parameters
    ----------
    depth_map : np.ndarray
        Depth map as a NumPy array of shape (H, W).
    K : np.ndarray
        Camera intrinsics matrix of shape (3, 3).
    valid_mask : np.ndarray, optional
        Boolean mask indicating valid depth values, shape (H, W).
        Invalid pixels are set to NaN.

    Returns
    -------
    np.ndarray
        Pointmap as a NumPy array of shape (H, W, 3) where each pixel
        contains its 3D (x, y, z) coordinates in camera space.

    Examples
    --------
    >>> depth = np.random.rand(480, 640) * 10  # Random depth 0-10m
    >>> K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])
    >>> pointmap = depth_to_pointmap(depth, K)
    >>> pointmap.shape
    (480, 640, 3)
    """
    H, W = depth_map.shape[:2]

    if valid_mask is not None:
        depth_map = depth_map.copy()
        depth_map[~valid_mask] = np.nan

    # Generate 3D point cloud from z-depth
    # Create pixel coordinate grids (u, v)
    v_coords, u_coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    # Convert to 3D coordinates using pinhole camera model
    z = depth_map
    x, y = pinhole_xy(u_coords, v_coords, z, K)

    pointmap = np.stack((x, y, z), axis=-1)  # (H, W, 3)

    return pointmap


def transform_to_pytorch3d_convention(pointmap: np.ndarray) -> np.ndarray:
    """
    Transform pointmap from R3 to PyTorch3D camera convention.

    R3 convention: X-right, Y-down, Z-forward
    PyTorch3D convention: X-left, Y-up, Z-forward

    Parameters
    ----------
    pointmap : np.ndarray
        Pointmap in R3 convention, shape (H, W, 3) or (N, 3).

    Returns
    -------
    np.ndarray
        Pointmap in PyTorch3D convention, same shape as input.

    Notes
    -----
    This transformation is applied before running SAM3D inference,
    as the model internally uses PyTorch3D conventions.
    """
    # R3→PyTorch3D is diag(-1,-1,1): negate X and Y
    return r3_to_p3d_positions(pointmap)


def load_and_process_depth(
    frames_path: str,
    depth_names: list[str],
    W: int,
    H: int,
    use_dataset_depth: bool = False,
    image: Optional[np.ndarray] = None,
    dataset_type: str = "gso",
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
    """
    Load and process depth maps to generate pointmap.

    Supports two modes:
    1. MoGe estimated depth (default): Run monocular depth estimation model
    2. Dataset ground truth depth: z-depth TIFFs (GSO, OursActionBench)

    Parameters
    ----------
    frames_path : str
        Path to the frames directory (for dataset GT depth).
    depth_names : list[str]
        List of depth file names (empty for MoGe mode).
    W : int
        Image width.
    H : int
        Image height.
    use_dataset_depth : bool, optional
        Whether to use dataset-provided depth instead of MoGe. Default: False.
        Raises FileNotFoundError if True but no depth files are available.
    image : np.ndarray, optional
        Input image (required for MoGe mode).
    dataset_type : str, optional
        Dataset type (``"gso"`` or ``"oursactionbench"``). Controls intrinsics
        when ``use_dataset_depth=True``. Default: ``"gso"``.

    Returns
    -------
    tuple
        (pointmap, K_matrix, valid_mask, depth_map_z, normals_map) where:
        - pointmap: 3D coordinates array of shape (H, W, 3)
        - K_matrix: Camera intrinsics matrix of shape (3, 3)
        - valid_mask: Boolean mask of valid depth values, or None for GT depth
        - depth_map_z: Z-depth map of shape (H, W), float32
        - normals_map: Surface normals of shape (H, W, 3), or None if unavailable

    Raises
    ------
    FileNotFoundError
        If use_dataset_depth is True but no depth files are available.
    ValueError
        If MoGe mode is used but image is not provided.
    """
    import os

    from .io_utils import load_image

    valid_mask = None

    if use_dataset_depth and not depth_names:
        raise FileNotFoundError(
            "depth_source='gt' but no depth files are available for this scene. "
            "Render them or use "
            "depth_source='pred'."
        )

    if use_dataset_depth:
        # Load depth map from file
        depth_path = os.path.join(frames_path, depth_names[0])
        depth_map = load_image(depth_path, to_uint8=False)

        if dataset_type == "gso":
            # GSO: float32 TIFF z-depth from PyTorch3D rasterization.
            # Background pixels are 0.
            # Intrinsics match the EscherNet protocol (lens 35mm, sensor 32mm),
            # see core/utils/eval_assets_export.gso_blender_intrinsics.
            fx = fy = 35.0 * W / 32.0
            cx = W / 2.0
            cy = H / 2.0

            depth_map_z = depth_map.astype(np.float32)
            valid_mask = depth_map_z > 0
            depth_map_z[~valid_mask] = 0.0
        elif dataset_type == "oursactionbench":
            # OAB: float32 TIFF z-depth from PyTorch3D rasterization.
            # Background pixels are 0.
            # Intrinsics come from the per-scene camera.json (via
            # actionbench_intrinsics — same K used by the rasterizer that
            # produced the tiff, so depth + K are consistent).
            from pathlib import Path as _Path
            from genia.core.utils.gt_data import actionbench_intrinsics
            scene_dir = _Path(frames_path).parent
            data_root = str(scene_dir.parent)
            scene_name = scene_dir.name
            K_ab = actionbench_intrinsics(H, W, scene_name=scene_name, data_root=data_root)
            fx = float(K_ab[0, 0]); fy = float(K_ab[1, 1])
            cx = float(K_ab[0, 2]); cy = float(K_ab[1, 2])

            depth_map_z = depth_map.astype(np.float32)
            valid_mask = depth_map_z > 0
            depth_map_z[~valid_mask] = 0.0
        else:
            raise ValueError(f"No GT depth loader for dataset {dataset_type!r}.")

        normals_map = None  # No normals from GT depth
    else:
        # MoGe monocular depth
        if image is None:
            raise ValueError("MoGe mode requires an image")

        from .model_cache import ModelCache

        # The same call the SAM3D pipeline's bundled MoGe makes (`image_to_float`, then
        # `MoGe.__call__`'s `infer(..., force_projection=False)`), so the depth matches
        # it exactly.
        loaded_image = (np.array(image) / 255).astype(np.float32)
        loaded_image = torch.from_numpy(loaded_image)
        loaded_image_rgb = loaded_image.permute(2, 0, 1).contiguous()[:3]

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                depth_output = ModelCache.get().moge_model.infer(
                    loaded_image_rgb.to(torch.device("cuda")), force_projection=False
                )

        depth_map_z = depth_output["depth"].cpu().numpy()
        valid_mask = depth_output["mask"].cpu().numpy()
        depth_map_z[~valid_mask] = 0.0

        # Extract surface normals from MoGe's normal_head (if available)
        normals_map = None
        if "normal" in depth_output and depth_output["normal"] is not None:
            normals_raw = depth_output["normal"]
            if normals_raw.dim() == 4:
                normals_raw = normals_raw[0]  # remove batch dim
            normals_map = normals_raw.cpu().numpy().astype(np.float32)  # (H, W, 3)

        if dataset_type == "oursactionbench":
            # K comes from the per-scene fit at
            # `{dataset}/{scene}/camera.json` (raises if missing).
            # MoGe's depth is still used (no GT depth available).
            from pathlib import Path as _Path
            from genia.core.utils.gt_data import actionbench_intrinsics
            # frames_path = {dataset_root}/{scene}/imgs/
            scene_dir = _Path(frames_path).parent
            data_root = str(scene_dir.parent)
            scene_name = scene_dir.name
            K_ab = actionbench_intrinsics(H, W, scene_name=scene_name, data_root=data_root)
            fx = float(K_ab[0, 0]); fy = float(K_ab[1, 1])
            cx = float(K_ab[0, 2]); cy = float(K_ab[1, 2])
        else:
            intrinsics = depth_output["intrinsics"].cpu().numpy()
            # MoGe returns intrinsics normalised to [0,1] image coords, so each
            # axis denormalises by its own extent (as cx/cy below already do).
            fx = intrinsics[0, 0] * W
            fy = intrinsics[1, 1] * H
            cx = intrinsics[0, 2] * W
            cy = intrinsics[1, 2] * H

    # Create intrinsics matrix
    K_matrix = np.eye(3)
    K_matrix[0, 0] = fx
    K_matrix[1, 1] = fy
    K_matrix[0, 2] = cx
    K_matrix[1, 2] = cy

    # Generate pointmap from depth
    pointmap = depth_to_pointmap(depth_map_z, K_matrix, valid_mask=valid_mask)
    # Mark invalid pixels as NaN so SAM3D's SSI normalizer ignores them
    if valid_mask is not None:
        pointmap[~valid_mask] = np.nan

    return pointmap, K_matrix, valid_mask, depth_map_z, normals_map


def compute_conegs_scaling(
    points_3d_camera: torch.Tensor,
    points_depth: torch.Tensor,
    K_inv: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Gaussian scaling based on pixel footprint.

    This function calculates the appropriate Gaussian standard deviation
    for each 3D point based on its depth and the camera intrinsics.

    Parameters
    ----------
    points_3d_camera : torch.Tensor
        Camera-space 3D points for each pixel, shape (N, 3).
    points_depth : torch.Tensor
        Z-depth for each pixel, shape (N,).
    K_inv : torch.Tensor
        Inverse intrinsics matrix, shape (3, 3).

    Returns
    -------
    torch.Tensor
        Isotropic Gaussian standard deviation per pixel, shape (N, 1).

    Notes
    -----
    The scaling is based on the pixel footprint at each depth,
    ensuring consistent Gaussian sizes relative to the projected area.
    """
    eps = 1e-6

    # Unnormalized ray direction for each pixel:
    # p_cam = z * d  =>  d = p_cam / z
    z = points_3d_camera[:, 2].clamp_min(eps)  # (N,)
    d = points_3d_camera / z[:, None]  # (N,3)
    d_norm = torch.linalg.norm(d, dim=1).clamp_min(eps)  # (N,)

    # Metric distance from camera origin to the 3D point (along the ray)
    s = points_depth  # (N,)

    # Constant pixel footprint (no distortion)
    col0 = K_inv[:, 0]
    col1 = K_inv[:, 1]
    pixel_width = 0.5 * (torch.linalg.norm(col0) + torch.linalg.norm(col1))

    pixel_width = pixel_width * (2.0 / math.sqrt(12.0))

    sigma = pixel_width * (s / d_norm)  # (N,)
    return sigma[:, None]


__all__ = [
    "depth_to_pointmap",
    "transform_to_pytorch3d_convention",
    "load_and_process_depth",
    "compute_conegs_scaling",
]
