# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Pure-torch quaternion and rotation utilities.

Drop-in replacements for pytorch3d.transforms functions, avoiding the heavy
CUDA extension loading that pytorch3d triggers on import.

All quaternions use wxyz convention (w, x, y, z).
All rotation matrices use the PyTorch3D right-multiply convention:
    p_rotated = p @ R
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# quaternion_to_matrix
# ---------------------------------------------------------------------------


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert wxyz quaternions to rotation matrices.

    Equivalent to ``pytorch3d.transforms.quaternion_to_matrix``.

    Parameters
    ----------
    quaternions : torch.Tensor
        Quaternions (..., 4) in wxyz order.

    Returns
    -------
    torch.Tensor
        Rotation matrices (..., 3, 3).
    """
    w, x, y, z = quaternions.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return torch.stack(
        [
            1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
            2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
            2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
        ],
        dim=-1,
    ).reshape(*quaternions.shape[:-1], 3, 3)


# ---------------------------------------------------------------------------
# matrix_to_quaternion
# ---------------------------------------------------------------------------


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    # min=1e-6 (not 0.0): at degenerate rotations (e.g. identity) the discarded
    # Shepperd candidates are sqrt(0), whose backward is +inf; argmax zeroes
    # their upstream grad but 0*inf=NaN poisons the shared matrix-entry grads.
    # The eps floor makes clamp's gradient 0 there, killing the inf. The
    # selected candidate always has q_abs >= 1 > eps, so it is unaffected.
    return torch.sqrt(torch.clamp(x, min=1e-6))


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to wxyz quaternions (Shepperd's method).

    Equivalent to ``pytorch3d.transforms.matrix_to_quaternion``.

    Parameters
    ----------
    matrix : torch.Tensor
        Rotation matrices (..., 3, 3).

    Returns
    -------
    torch.Tensor
        Quaternions (..., 4) in wxyz order.
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]

    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # Build four candidate quaternions, each correct when its component is
    # the largest (numerically best-conditioned).
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m01 + m10, m20 + m02], dim=-1),
            torch.stack([m02 - m20, m01 + m10, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m12 + m21, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )  # (..., 4, 4)

    # Normalize each candidate by its corresponding q_abs (clamped for safety).
    flr = torch.tensor(0.1, dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].clamp(min=flr))

    # Pick the best-conditioned candidate (largest |q_i|).
    flat_q = q_abs.reshape(-1, 4)
    best_idx = flat_q.argmax(dim=-1)  # (B,)
    flat_candidates = quat_candidates.reshape(-1, 4, 4)
    result = flat_candidates[
        torch.arange(flat_candidates.shape[0], device=matrix.device), best_idx
    ]

    return result.reshape(*batch_dim, 4)


# ---------------------------------------------------------------------------
# quaternion_multiply
# ---------------------------------------------------------------------------


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two wxyz quaternions.

    Equivalent to ``pytorch3d.transforms.quaternion_multiply``.

    Parameters
    ----------
    a, b : torch.Tensor
        Quaternions (..., 4) in wxyz order.

    Returns
    -------
    torch.Tensor
        Product quaternion (..., 4) in wxyz order.
    """
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


# ---------------------------------------------------------------------------
# quaternion_invert
# ---------------------------------------------------------------------------


def quaternion_invert(quaternion: torch.Tensor) -> torch.Tensor:
    """Invert a unit quaternion (conjugate).

    Equivalent to ``pytorch3d.transforms.quaternion_invert``.

    Parameters
    ----------
    quaternion : torch.Tensor
        Unit quaternions (..., 4) in wxyz order.

    Returns
    -------
    torch.Tensor
        Inverted quaternions (..., 4).
    """
    scaling = torch.tensor([1, -1, -1, -1], dtype=quaternion.dtype, device=quaternion.device)
    return quaternion * scaling


# ---------------------------------------------------------------------------
# matrix_to_euler_angles
# ---------------------------------------------------------------------------


def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError(f"Invalid axis letter: {letter}")


def matrix_to_euler_angles(matrix: torch.Tensor, convention: str) -> torch.Tensor:
    """Convert rotation matrices to Euler angles.

    Equivalent to ``pytorch3d.transforms.matrix_to_euler_angles``.

    Parameters
    ----------
    matrix : torch.Tensor
        Rotation matrices (..., 3, 3).
    convention : str
        Axis convention string, e.g. ``"XYZ"``.

    Returns
    -------
    torch.Tensor
        Euler angles (..., 3) in radians.
    """
    # Use the general intrinsic rotation decomposition.
    # R = R_i(a) @ R_j(b) @ R_k(c)
    # For Tait-Bryan (i != k): extract b from element (i, k) and neighbours.
    i = _index_from_letter(convention[0])
    j = _index_from_letter(convention[1])
    k = _index_from_letter(convention[2])

    if i == k:
        raise NotImplementedError("Only Tait-Bryan (i!=k) conventions supported")

    # Determine sign: even permutation (XYZ, YZX, ZXY) → +1, odd → -1
    even_perm = (i, j, k) in ((0, 1, 2), (1, 2, 0), (2, 0, 1))
    sign = 1.0 if even_perm else -1.0

    # b = asin(sign * m[i, k])
    b = torch.asin(torch.clamp(sign * matrix[..., i, k], -1.0, 1.0))

    # a = atan2(-sign * m[j, k], m[k, k])
    a = torch.atan2(-sign * matrix[..., j, k], matrix[..., k, k])

    # c = atan2(-sign * m[i, j], m[i, i])
    c = torch.atan2(-sign * matrix[..., i, j], matrix[..., i, i])

    return torch.stack([a, b, c], dim=-1)


# ---------------------------------------------------------------------------
# PyTorch3D <-> R3 coordinate conversion helpers
# ---------------------------------------------------------------------------

# The P3D↔R3 conversion is a 180° rotation around Z: diag(-1, -1, 1).
# This is its own inverse (symmetric, diag(-1,-1,1)² = I).
_P3D_TO_R3_QUAT_WXYZ = (0.0, 0.0, 0.0, 1.0)  # 180° around Z in wxyz


def p3d_to_r3_positions(xyz: torch.Tensor) -> torch.Tensor:
    """Convert positions from PyTorch3D to R3 convention (negate X and Y)."""
    out = xyz.clone()
    out[..., :2] *= -1
    return out


def r3_to_p3d_positions(xyz):
    """Convert positions from R3 to PyTorch3D convention (negate X and Y).

    Same operation as p3d_to_r3_positions since diag(-1,-1,1) is self-inverse.
    Works with both torch.Tensor and numpy arrays.
    """
    if isinstance(xyz, torch.Tensor):
        out = xyz.clone()
    else:
        out = xyz.copy()
    out[..., :2] *= -1
    return out


def p3d_to_r3_quaternions(quats: torch.Tensor) -> torch.Tensor:
    """Convert quaternions from PyTorch3D to R3 convention.

    Applies left-multiplication by the P3D→R3 quaternion (180° around Z).
    Result: (w,x,y,z) → (-z, -y, x, w).
    """
    p3d_to_r3_q = torch.tensor(
        _P3D_TO_R3_QUAT_WXYZ, dtype=quats.dtype, device=quats.device,
    )
    return quaternion_multiply(p3d_to_r3_q.expand_as(quats), quats)


__all__ = [
    "quaternion_to_matrix",
    "matrix_to_quaternion",
    "quaternion_multiply",
    "quaternion_invert",
    "matrix_to_euler_angles",
    "p3d_to_r3_positions",
    "r3_to_p3d_positions",
    "p3d_to_r3_quaternions",
]
