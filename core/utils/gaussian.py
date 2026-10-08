"""
Gaussian splatting operations and utilities.

This module provides functions for creating, manipulating, and transforming
3D Gaussian splat representations, including coordinate system conversions.
"""

from __future__ import annotations

import sys
from typing import Any, List, Optional

import numpy as np
import torch
from .quaternion_ops import (
    p3d_to_r3_positions,
    p3d_to_r3_quaternions,
)

_IDENTITY_4x4 = np.eye(4, dtype=np.float32)

# Add submodules directory to path to import sam3d_objects.
#
# APPEND, never insert(0): a host process may put its own ``sam3d_objects``
# fork at the head of sys.path before importing this module, and that fork must
# keep winning; prepending here would shadow it and cache our copy in
# ``sys.modules`` for the whole process.
from genia.core.paths import SAM3D_OBJECTS_ROOT  # noqa: E402

_SUBMODULES = str(SAM3D_OBJECTS_ROOT)
if _SUBMODULES not in sys.path:
    sys.path.append(_SUBMODULES)

from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian.gaussian_model import Gaussian

# Spherical harmonics constant (degree 0)
C0: float = 0.28209479177387814


def RGB2SH(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB colors to spherical harmonics coefficients (degree 0).

    Parameters
    ----------
    rgb : np.ndarray
        RGB colors in [0, 1] range, shape (..., 3).

    Returns
    -------
    np.ndarray
        SH coefficients, same shape as input.

    Examples
    --------
    >>> rgb = np.array([0.5, 0.5, 0.5])
    >>> sh = RGB2SH(rgb)
    >>> sh
    array([0., 0., 0.])
    """
    return (rgb - 0.5) / C0


def SH2RGB(sh: np.ndarray) -> np.ndarray:
    """
    Convert spherical harmonics coefficients (degree 0) to RGB colors.

    Parameters
    ----------
    sh : np.ndarray
        SH coefficients, shape (..., 3).

    Returns
    -------
    np.ndarray
        RGB colors in [0, 1] range, same shape as input.

    Examples
    --------
    >>> sh = np.array([0., 0., 0.])
    >>> rgb = SH2RGB(sh)
    >>> rgb
    array([0.5, 0.5, 0.5])
    """
    return sh * C0 + 0.5


def create_gaussians_object(
    xyz: torch.Tensor,
    features: torch.Tensor,
    scales: torch.Tensor,
    rots: torch.Tensor,
    opacities: torch.Tensor,
    *,
    aabb: "Optional[List[float]]" = None,
) -> Gaussian:
    """
    Create a Gaussian model from raw parameters.

    This function handles the normalization and internal representation
    required by the Gaussian model, including computing the axis-aligned
    bounding box (AABB) for coordinate normalization.

    Parameters
    ----------
    xyz : torch.Tensor
        3D positions, shape (N, 3).
    features : torch.Tensor
        SH features (color), shape (N, 1, 3) or (N, K, 3).
    scales : torch.Tensor
        Gaussian scales, shape (N, 3).
    rots : torch.Tensor
        Quaternion rotations (wxyz format), shape (N, 4).
    opacities : torch.Tensor
        Opacities in [0, 1], shape (N, 1).
    aabb : sequence of 6 floats, optional, keyword-only
        ``[min_x, min_y, min_z, size_x, size_y, size_z]``.  Default ``None``
        derives it from ``xyz``, which is right for an arbitrary cloud but
        DEGENERATE when the points are coplanar or there is only one: that axis
        gets ``size == 0`` and the normalization below divides by zero, NaNing
        every coordinate.  Pass the known box when the caller has one (a voxel
        grid always does -- see ``VOXEL_GRID_AABB``).

    Returns
    -------
    Gaussian
        Initialized Gaussian model with all parameters set.

    Notes
    -----
    The Gaussian model internally stores normalized coordinates and applies
    activation functions to scales and opacities. This function handles
    the necessary transformations.
    """
    # AABB (axis-aligned bounding box), format
    # [min_x, min_y, min_z, size_x, size_y, size_z] -- supplied, or derived from
    # the cloud when the caller has no box of its own.
    if aabb is None:
        xyz_min = xyz.min(dim=0)[0]
        xyz_max = xyz.max(dim=0)[0]
        xyz_size = xyz_max - xyz_min
        aabb = torch.cat([xyz_min, xyz_size]).tolist()
    else:
        aabb = [float(v) for v in aabb]
        if len(aabb) != 6:
            raise ValueError(f"aabb must be 6 values [min xyz, size xyz], got {len(aabb)}")
        xyz_min = torch.as_tensor(aabb[:3], dtype=xyz.dtype, device=xyz.device)
        xyz_size = torch.as_tensor(aabb[3:], dtype=xyz.dtype, device=xyz.device)

    # Normalize xyz to [0, 1] range for internal storage
    # The Gaussian model expects normalized coordinates and will denormalize using AABB
    xyz_normalized = (xyz - xyz_min) / xyz_size

    # Create Gaussian model with computed AABB
    gaussians = Gaussian(aabb=aabb, scaling_bias=0.0, opacity_bias=0.0)

    # Move all tensors to CUDA
    device = "cuda"
    xyz_normalized = xyz_normalized.to(device)
    features = features.to(device)
    scales = scales.to(device)
    rots = rots.to(device)
    opacities = opacities.to(device)

    # Initialize gaussians with the computed values
    gaussians._xyz = xyz_normalized  # Use normalized coordinates!
    if features.shape[1] > 1:
        # Split into DC (first band) and SH rest (higher-order bands)
        gaussians._features_dc = features[:, :1, :]
        gaussians._features_rest = features[:, 1:, :]
        degree = int(features.shape[1] ** 0.5) - 1
        gaussians.sh_degree = degree
        gaussians.active_sh_degree = degree
    else:
        gaussians._features_dc = features

    # Disable scale_bias and opacity_bias, move to correct device
    gaussians.scale_bias = torch.tensor(0.0, device=gaussians._xyz.device)
    gaussians.opacity_bias = torch.tensor(0.0, device=gaussians._xyz.device)

    # Apply inverse activation to convert external scales to internal representation
    scales_internal = gaussians.inverse_scaling_activation(scales)

    gaussians._scaling = scales_internal
    gaussians._rotation = rots - gaussians.rots_bias[None, :]

    # Clamp opacities to avoid numerical issues with inverse_sigmoid at exactly 0 or 1
    opacities_clamped = torch.clamp(opacities, 1e-6, 1.0 - 1e-6)
    opacities_internal = gaussians.inverse_opacity_activation(opacities_clamped)

    gaussians._opacity = opacities_internal

    return gaussians


# Sigma of a voxel splat as a multiple of the voxel edge.  The single most important
# setting of the voxel renderer: too small and holes open between splats, so the loss
# landscape goes spiky and the gradient noisy; too large and the surface blurs and the
# gradient flattens.  0.4 balances the two.
VOXEL_SPLAT_SIGMA_MULT = 0.4

# The occupancy grid's own box, in the object-local frame every consumer uses
# (``(idx + 0.5) / G - 0.5`` maps voxel centres into it).  Passed explicitly so a
# planar or single-voxel occupancy cannot produce a zero-size AABB axis.
VOXEL_GRID_AABB = [-0.5, -0.5, -0.5, 1.0, 1.0, 1.0]


def point_splat_gaussians(
    xyz: torch.Tensor,
    sigma: float,
    device: "torch.device | str" = "cuda",
) -> "Optional[Gaussian]":
    """One isotropic, near-opaque, colourless Gaussian at each of ``xyz`` (N, 3).

    The shared core of the voxel renderer: a caller supplies voxel centres or, for pose
    alignment, the canonical shape's own points (mesh vertices
    on a deforming object, voxel-derived points otherwise) at the equivalent sigma.  Both
    want the same thing -- a differentiable surface proxy whose depth and silhouette can be
    rasterised -- and neither scores colour.

    ``sigma`` is an ABSOLUTE standard deviation in the same units as ``xyz``, so callers
    working from a grid pass ``sigma_mult / grid``.  The AABB is pinned to the canonical
    ``[-0.5, 0.5]`` box rather than derived from the points, which would be degenerate for
    a coplanar or single-point cloud.
    """
    m = xyz.shape[0]
    if m == 0:
        return None
    scales = torch.full((m, 3), float(sigma), device=device)
    rots = torch.zeros(m, 4, device=device)
    rots[:, 0] = 1.0                                   # identity, wxyz
    opac = torch.full((m, 1), 0.99, device=device)
    rgb = np.full((m, 1, 3), 0.5, dtype=np.float32)    # never scored
    feats = torch.from_numpy(RGB2SH(rgb)).float().to(device)
    return create_gaussians_object(
        xyz.to(device), feats, scales, rots, opac, aabb=VOXEL_GRID_AABB)


def join_gaussians(*gaussian_objects: Gaussian) -> Gaussian:
    """
    Join multiple Gaussian objects into a single combined Gaussian object.

    Parameters
    ----------
    *gaussian_objects : Gaussian
        Variable number of Gaussian objects to combine.

    Returns
    -------
    Gaussian
        Combined Gaussian object containing all gaussians from input objects.

    Raises
    ------
    ValueError
        If no Gaussian objects are provided.

    Examples
    --------
    >>> combined = join_gaussians(gs1, gs2, gs3)
    >>> combined.get_xyz.shape[0]
    30000  # Sum of points from gs1, gs2, gs3
    """
    if len(gaussian_objects) == 0:
        raise ValueError("At least one Gaussian object must be provided")

    if len(gaussian_objects) == 1:
        return gaussian_objects[0]

    # Collect all properties from each Gaussian object
    all_xyz = []
    all_features = []
    all_scales = []
    all_rots = []
    all_opacities = []

    for gs in gaussian_objects:
        all_xyz.append(gs.get_xyz)
        all_features.append(gs.get_features)
        all_scales.append(gs.get_scaling)
        all_rots.append(gs.get_rotation)
        all_opacities.append(gs.get_opacity)

    # Pad features to the maximum SH degree before concatenating.
    # Different Gaussians may have different SH band counts (e.g. background
    # has DC only with shape (N,1,3) while finetuned objects may have degree-1
    # SH with shape (N,4,3)).  Pad shorter features with zeros on dim=1.
    max_sh = max(f.shape[1] for f in all_features)
    for i, f in enumerate(all_features):
        if f.shape[1] < max_sh:
            pad = torch.zeros(
                f.shape[0], max_sh - f.shape[1], f.shape[2],
                device=f.device, dtype=f.dtype,
            )
            all_features[i] = torch.cat([f, pad], dim=1)

    # Concatenate all properties
    combined_xyz = torch.cat(all_xyz, dim=0)
    combined_features = torch.cat(all_features, dim=0)
    combined_scales = torch.cat(all_scales, dim=0)
    combined_rots = torch.cat(all_rots, dim=0)
    combined_opacities = torch.cat(all_opacities, dim=0)

    # Create new combined Gaussian object
    combined_gs = create_gaussians_object(
        xyz=combined_xyz,
        features=combined_features,
        scales=combined_scales,
        rots=combined_rots,
        opacities=combined_opacities,
    )

    return combined_gs


def create_gaussians_from_pointmap(
    image: np.ndarray,
    pointmap: np.ndarray,
    K: np.ndarray,
    scale_pointmap: Optional[np.ndarray] = None,
) -> Gaussian:
    """
    Create Gaussian splats from pointmap and RGB image.

    Parameters
    ----------
    image : np.ndarray
        RGB image as a NumPy array, shape (N, 3) for flattened or (H, W, 3).
    pointmap : np.ndarray
        Pointmap as a NumPy array of shape (N, 3) for flattened or (H, W, 3).
    K : np.ndarray
        Camera intrinsics matrix, shape (3, 3).
    scale_pointmap : np.ndarray, optional
        CAMERA-space points to size the Gaussians from, when ``pointmap`` has
        already been lifted out of camera space.  The pixel-footprint scale is a
        function of camera depth (``sigma ~ z / |ray|``), so feeding it world
        coordinates is wrong twice over: the ray direction is meaningless, and a
        world ``z`` may be NEGATIVE, giving ``sigma < 0`` whose ``log`` is NaN
        (e.g. background behind the world origin).  A rigid ``c2w``
        does not rescale, so the camera-space sigma is already the world sigma.

    Returns
    -------
    Gaussian
        The created Gaussian model.
    """
    from .depth import compute_conegs_scaling

    # Create Gaussians from pointmap
    # Reshape pointmap to (N, 3)
    xyz = pointmap.reshape(-1, 3)
    xyz = torch.from_numpy(xyz).float()  # (N, 3)

    # Convert RGB to SH degree 0
    # SH0 = (RGB - 0.5) / C0, where C0 = 0.28209479177387814
    rgb = image.reshape(-1, 3).astype(np.float32) / 255.0  # Normalize to [0, 1]
    features = RGB2SH(rgb)
    features = torch.from_numpy(features).float().unsqueeze(1)  # (N, 1, 3) for SH degree 0

    # Compute scales using compute_conegs_scaling
    K_torch = torch.from_numpy(K).float()
    K_inv = torch.inverse(K_torch)

    # Footprint geometry must be CAMERA-space (see ``scale_pointmap``); the
    # positions in ``xyz`` may already have been lifted to world.
    cam_xyz = (xyz if scale_pointmap is None
               else torch.from_numpy(scale_pointmap.reshape(-1, 3)).float())
    points_depth = cam_xyz[:, 2]  # (N,) camera z-depth

    # Isotropic sigma; invariant under the rigid camera->world transform.
    scales_sigma_world = compute_conegs_scaling(cam_xyz, points_depth, K_inv)  # (N, 1)

    # Make it isotropic (same scale in all 3 dimensions)
    scales = scales_sigma_world.repeat(1, 3)  # (N, 3)

    # Rotation irrelevant: isotropic scales
    rots = torch.zeros((xyz.shape[0], 4), dtype=torch.float32)
    rots[:, -1] = 1

    # All opacities should be 1.0
    opacities = torch.ones((xyz.shape[0], 1), dtype=torch.float32)

    # Create Gaussian model
    gaussians = create_gaussians_object(
        xyz=xyz,
        features=features,
        scales=scales,
        rots=rots,
        opacities=opacities,
    )

    return gaussians


def create_background_gaussians(
    image: np.ndarray,
    pointmap: np.ndarray,
    masks: List[np.ndarray],
    K_matrix: np.ndarray,
    c2w: np.ndarray = _IDENTITY_4x4,
) -> Optional[Gaussian]:
    """
    Create Gaussian splats for the background (non-masked regions).

    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W, 3).
    pointmap : np.ndarray
        3D pointmap in R3 camera-space convention, shape (H, W, 3).
    masks : list of np.ndarray
        List of object masks, each shape (H, W).
    K_matrix : np.ndarray
        Camera intrinsics matrix, shape (3, 3).
    c2w : np.ndarray
        Camera-to-world transform, shape (4, 4).  The camera-space
        pointmap is transformed to world space so that background
        Gaussians live in the same coordinate frame as the foreground
        scene.  Defaults to identity.

    Returns
    -------
    Gaussian or None
        Background Gaussian object, or ``None`` if no valid background
        pixels remain after combining the object masks with the
        finite-depth mask (e.g. GSO, where the GT depth tiff covers only
        the foreground object).

    Notes
    -----
    The background is defined as all pixels that are NOT covered by any
    of the provided object masks.
    """
    # Create combined mask of all objects
    background_mask = ~np.any(np.stack(masks, axis=0), axis=0)

    # Also exclude pixels with non-finite pointmap values (NaN from invalid depth)
    valid = np.isfinite(pointmap).all(axis=-1)
    background_mask = background_mask & valid

    if not background_mask.any():
        return None

    bg_points_cam = pointmap[background_mask]  # (N, 3) camera-space

    # Transform to world space using c2w
    R = c2w[:3, :3].astype(np.float32)
    t = c2w[:3, 3].astype(np.float32)
    bg_points = bg_points_cam @ R.T + t

    # Create background Gaussians from the non-masked region.  The positions are
    # world-space, but the pixel-footprint scale has to be measured in CAMERA
    # space -- a world z can be negative, and log(negative sigma) is NaN.
    gaussians_bg = create_gaussians_from_pointmap(
        image=image[background_mask],
        pointmap=bg_points,
        K=K_matrix,
        scale_pointmap=bg_points_cam,
    )

    return gaussians_bg


def aggregate_background_gaussians(sequence, object_ids) -> Optional[Gaussian]:
    """Every frame's background, each lifted to world by its OWN camera, joined.

    One frame's background only covers the slice its camera saw; unioning all
    of them gives the full reconstructed backdrop.  Each piece is placed with
    that frame's ``(K, c2w)``, so the result is a single world-space cloud --
    correct exactly when the c2w's are a real trajectory (``map_anything``);
    with identity c2w every frame's "world" is its own camera space and the
    pieces overlay, which is the same convention the rest of the pipeline uses.

    ``object_ids`` are the foreground ids to carve out of the background.
    Returns ``None`` when no frame has any valid background pixel.
    """
    pieces = []
    for fk in sequence.frame_keys:
        fd = sequence[fk]
        piece = create_background_gaussians(
            fd.image, fd.pointmap, [fd.masks[oid] for oid in object_ids],
            fd.K_matrix, c2w=fd.c2w,
        )
        if piece is not None:
            pieces.append(piece)
    if not pieces:
        return None
    return pieces[0] if len(pieces) == 1 else join_gaussians(*pieces)


def match_sh_bands(gaussian: Gaussian, n_bands: int) -> Gaussian:
    """Zero-pad ``gaussian``'s SH features out to ``n_bands`` coefficient bands.

    Background Gaussians carry DC only (``(N, 1, 3)``) while a decoded
    reconstruction carries higher-order bands, so a PLY written from each would
    declare a different ``f_rest_*`` count and a viewer could not load them as
    one scene.  Same zero-pad :func:`join_gaussians` applies internally, minus
    the join.  A no-op when the bands already match (or already exceed).
    """
    features = gaussian.get_features
    if features.shape[1] >= n_bands:
        return gaussian
    pad = torch.zeros(
        features.shape[0], n_bands - features.shape[1], features.shape[2],
        device=features.device, dtype=features.dtype,
    )
    return create_gaussians_object(
        xyz=gaussian.get_xyz,
        features=torch.cat([features, pad], dim=1),
        scales=gaussian.get_scaling,
        rots=gaussian.get_rotation,
        opacities=gaussian.get_opacity,
    )


def transform_scene_to_r3_convention(scene_gs: Gaussian) -> Gaussian:
    """
    Transform combined scene from PyTorch3D convention back to R3 convention.

    This should be done AFTER make_scene() on the combined scene.

    PyTorch3D convention: X-left, Y-up, Z-forward
    R3 convention: X-right, Y-down, Z-forward

    Parameters
    ----------
    scene_gs : Gaussian
        Scene Gaussian object in PyTorch3D convention.

    Returns
    -------
    Gaussian
        Scene Gaussian object in R3 convention.

    Notes
    -----
    This transformation is the inverse of transform_to_pytorch3d_convention
    and should be applied to the output of make_scene() before rendering.
    """
    # Get the denormalized xyz coordinates
    xyz_unnormalized = scene_gs.get_xyz  # This applies: xyz * aabb[3:] + aabb[:3]

    # Transform from PyTorch3D convention to R3 convention
    # P3D↔R3 is diag(-1,-1,1): negate X and Y for positions, apply 180° Z rotation for quaternions
    xyz = p3d_to_r3_positions(xyz_unnormalized)

    # Transform rotations (quaternions)
    original_rots = scene_gs.get_rotation  # (N, 4) in wxyz format
    transformed_rots = p3d_to_r3_quaternions(original_rots)

    # Create new Gaussians object
    new_scene_gs = create_gaussians_object(
        xyz=xyz,
        features=scene_gs.get_features,
        scales=scene_gs.get_scaling,
        rots=transformed_rots,
        opacities=scene_gs.get_opacity,
    )

    return new_scene_gs


def transform_scene_to_world(scene_gs: Gaussian, c2w: np.ndarray) -> Gaussian:
    """Transform Gaussian scene from R3 camera space to R3 world space.

    No-op when *c2w* is identity.  Mirrors
    :func:`transform_scene_to_r3_convention` but applies a rigid c2w
    transform instead of the P3D↔R3 convention flip.

    Parameters
    ----------
    scene_gs : Gaussian
        Scene Gaussian object in R3 camera space (e.g. after
        ``transform_scene_to_r3_convention``).
    c2w : np.ndarray
        Camera-to-world transform, shape ``(4, 4)``.

    Returns
    -------
    Gaussian
        Scene Gaussian object in R3 world space.
    """
    if np.allclose(c2w, np.eye(4)):
        return scene_gs

    from .rendering import transform_gaussian_params_cam_to_world

    xyz = scene_gs.get_xyz          # (N, 3)
    quats = scene_gs.get_rotation   # (N, 4) wxyz

    xyz_world, quats_world = transform_gaussian_params_cam_to_world(xyz, quats, c2w)

    return create_gaussians_object(
        xyz=xyz_world,
        features=scene_gs.get_features,
        scales=scene_gs.get_scaling,
        rots=quats_world,
        opacities=scene_gs.get_opacity,
    )


__all__ = [
    "C0",
    "RGB2SH",
    "SH2RGB",
    "create_gaussians_object",
    "join_gaussians",
    "create_gaussians_from_pointmap",
    "create_background_gaussians",
    "transform_scene_to_r3_convention",
    "attach_sh_rest",
]


def attach_sh_rest(gs: Any, sh_rest: torch.Tensor) -> None:
    """Attach extra SH coefficients to a decoded Gaussian (in-place)."""
    gs._features_rest = sh_rest
    degree = int((sh_rest.shape[1] + 1) ** 0.5) - 1
    gs.sh_degree = degree
    gs.active_sh_degree = degree
