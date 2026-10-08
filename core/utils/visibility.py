"""Per-voxel per-view visibility weights (DDA self-occlusion + rasterized cross-object).

**Self-occlusion** is computed per-object via Numba JIT DDA ray tracing through
the object's own 64³ occupancy grid.  ``neighbor_tolerance`` prevents
grazing-angle false positives at surface boundaries.

**Cross-object occlusion** (multiple objects in scene) rasterizes all objects'
voxel surface meshes in one pass with PyTorch3D ``MeshRasterizer`` to produce a
scene depth buffer.  A ``minimum_filter`` fills sub-pixel gaps.  A voxel is
cross-occluded when the nearest rasterized surface depth at its projected pixel
is significantly closer than the voxel's own depth.

A voxel is visible iff it is:
  1. not self-occluded (DDA),
  2. inside the image and in front of the camera,
  3. not cross-occluded (rasterized depth test).

Coordinate conventions::

  Pose: transformed = xyz_local * scale @ R + translation  (PyTorch3D right-multiply)
  Inverse: cam_canonical = -translation @ R^T / scale
  Canonical → voxel: cam_voxel = (cam_canonical + 0.5) * 64
  Screen: u = fx * (−cam_x) / cam_z + cx   (P3D→R3 negate X, Y)
"""
from __future__ import annotations

import math
from typing import Tuple

import numba
import numpy as np


# =====================================================================
# JIT-compiled DDA core
# =====================================================================

@numba.njit(cache=True)
def _ray_box_intersection(
    origin: np.ndarray,
    direction: np.ndarray,
    grid_size: int,
) -> Tuple[float, float, bool]:
    """Ray-AABB intersection. Returns (t_enter, t_exit, hit)."""
    t_min = -1e30
    t_max = 1e30

    for i in range(3):
        if abs(direction[i]) < 1e-10:
            if origin[i] < 0.0 or origin[i] > grid_size:
                return 0.0, 0.0, False
        else:
            t1 = (0.0 - origin[i]) / direction[i]
            t2 = (float(grid_size) - origin[i]) / direction[i]
            if t1 > t2:
                t1, t2 = t2, t1
            if t1 > t_min:
                t_min = t1
            if t2 < t_max:
                t_max = t2
            if t_min > t_max:
                return 0.0, 0.0, False

    return t_min, t_max, True


@numba.njit(cache=True)
def _check_ray_occluded(
    start: np.ndarray,
    target_x: int, target_y: int, target_z: int,
    occupancy: np.ndarray,
    grid_size: int,
    tolerance_sq: float,
) -> bool:
    """Trace a single ray from start to target voxel center. Return True if occluded."""
    end_x = target_x + 0.5
    end_y = target_y + 0.5
    end_z = target_z + 0.5

    dx = end_x - start[0]
    dy = end_y - start[1]
    dz = end_z - start[2]
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length < 1e-8:
        return False
    inv_len = 1.0 / length
    dx *= inv_len
    dy *= inv_len
    dz *= inv_len

    sx = start[0]
    sy = start[1]
    sz = start[2]

    # If start is outside grid, advance to entry point
    inside = (sx >= 0.0 and sx < grid_size and
              sy >= 0.0 and sy < grid_size and
              sz >= 0.0 and sz < grid_size)
    if not inside:
        t_enter, _, hit = _ray_box_intersection(start, np.array([dx, dy, dz]), grid_size)
        if not hit or t_enter > length:
            return False
        if t_enter > 0.0:
            sx = start[0] + dx * (t_enter + 0.001)
            sy = start[1] + dy * (t_enter + 0.001)
            sz = start[2] + dz * (t_enter + 0.001)

    # Current voxel
    vx = int(math.floor(sx))
    vy = int(math.floor(sy))
    vz = int(math.floor(sz))
    if vx < 0: vx = 0
    if vy < 0: vy = 0
    if vz < 0: vz = 0
    if vx >= grid_size: vx = grid_size - 1
    if vy >= grid_size: vy = grid_size - 1
    if vz >= grid_size: vz = grid_size - 1

    # End voxel
    ex = min(max(target_x, 0), grid_size - 1)
    ey = min(max(target_y, 0), grid_size - 1)
    ez = min(max(target_z, 0), grid_size - 1)

    # Step direction
    step_x = 1 if dx >= 0.0 else -1
    step_y = 1 if dy >= 0.0 else -1
    step_z = 1 if dz >= 0.0 else -1

    # tmax and tdelta
    if abs(dx) < 1e-10:
        tmax_x = 1e30
        tdelta_x = 1e30
    else:
        if dx > 0:
            tmax_x = (vx + 1 - sx) / dx
        else:
            tmax_x = (vx - sx) / dx
        tdelta_x = abs(1.0 / dx)

    if abs(dy) < 1e-10:
        tmax_y = 1e30
        tdelta_y = 1e30
    else:
        if dy > 0:
            tmax_y = (vy + 1 - sy) / dy
        else:
            tmax_y = (vy - sy) / dy
        tdelta_y = abs(1.0 / dy)

    if abs(dz) < 1e-10:
        tmax_z = 1e30
        tdelta_z = 1e30
    else:
        if dz > 0:
            tmax_z = (vz + 1 - sz) / dz
        else:
            tmax_z = (vz - sz) / dz
        tdelta_z = abs(1.0 / dz)

    max_steps = grid_size * 3
    for _ in range(max_steps):
        # Bounds check
        if vx < 0 or vx >= grid_size or vy < 0 or vy >= grid_size or vz < 0 or vz >= grid_size:
            break

        # Check occupancy (skip if near target)
        if occupancy[vx, vy, vz]:
            dist_sq = float((vx - target_x) ** 2 + (vy - target_y) ** 2 + (vz - target_z) ** 2)
            if dist_sq > tolerance_sq:
                return True  # Occluded

        # Reached target
        if vx == ex and vy == ey and vz == ez:
            break

        # Step to next voxel (<=  gives lowest-axis-wins on ties,
        # matching np.argmin behaviour in the reference implementation)
        if tmax_x <= tmax_y:
            if tmax_x <= tmax_z:
                vx += step_x
                tmax_x += tdelta_x
            else:
                vz += step_z
                tmax_z += tdelta_z
        else:
            if tmax_y <= tmax_z:
                vy += step_y
                tmax_y += tdelta_y
            else:
                vz += step_z
                tmax_z += tdelta_z

    return False


@numba.njit(cache=True, parallel=True)
def _compute_self_occlusion_jit(
    coords_int: np.ndarray,
    camera_pos_voxel: np.ndarray,
    occupancy: np.ndarray,
    grid_size: int,
    tolerance_sq: float,
) -> np.ndarray:
    """JIT-compiled self-occlusion for all voxels (parallelized over voxels)."""
    N = coords_int.shape[0]
    visibility = np.ones(N, dtype=np.float32)

    for i in numba.prange(N):
        tx = coords_int[i, 0]
        ty = coords_int[i, 1]
        tz = coords_int[i, 2]
        if _check_ray_occluded(camera_pos_voxel, tx, ty, tz,
                               occupancy, grid_size, tolerance_sq):
            visibility[i] = 0.0

    return visibility


# =====================================================================
# Single-object DDA visibility
# =====================================================================


# =====================================================================
# Single-object multi-view (DDA only, no cross-object)
# =====================================================================


# =====================================================================
# Multi-object visibility (DDA self-occlusion + rasterized cross-object)
# =====================================================================


__all__ = [
]
