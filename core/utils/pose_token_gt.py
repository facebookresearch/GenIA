# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Inverse SSI mapping: camera-space poses → raw Stage 1 pose tokens.

Reverses the ``differentiable_pose_decode`` chain in ``refinement.py``
to convert camera-space poses into the SSI-normalized raw token space that
the flow-matching model predicts.
"""

from typing import Dict

import torch

# Import normalization constants from refinement.py
from genia.core.utils.refinement import _ROTATION_6D_MEAN, _ROTATION_6D_STD


# =====================================================================
# Camera-space pose → raw SSI-normalized tokens
# =====================================================================

def camera_pose_to_raw_tokens(
    R_pt3d: torch.Tensor,
    t_pt3d: torch.Tensor,
    s_pt3d: torch.Tensor,
    pointmap_scale: torch.Tensor,
    pointmap_shift: torch.Tensor,
    downsample_factor: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Convert camera-space pose to raw Stage 1 tokens.

    Uses SAM3D's native ``ScaleShiftInvariant.from_instance_pose`` for the
    SSI inversion (camera-space → SSI-normalized space), then converts the
    resulting quaternion/scale to the flow-matching raw token format
    (normalized 6D rotation, log-scale).

    Parameters
    ----------
    R_pt3d : (3, 3) — rotation matrix in PyTorch3D camera space (row-vector).
    t_pt3d : (3,) — translation in PyTorch3D camera space.
    s_pt3d : (3,) — scale in PyTorch3D camera space.
    pointmap_scale : (3,) — SSI scale from pointmap normalization.
    pointmap_shift : (3,) — SSI shift from pointmap normalization.
    downsample_factor : float — from sparse-structure model (usually 1.0).

    Returns
    -------
    dict with keys:
        ``6drotation_normalized`` : (6,) — normalized 6D rotation
        ``translation`` : (3,) — raw translation in SSI space
        ``scale`` : (3,) — raw log-scale in SSI space
        ``translation_scale`` : (1,) — always 1.0 (SAM3D training convention:
            ``x_translation_scale = ones_like(...)`` — see
            ``sam3d_objects/data/dataset/tdfy/pose_target.py``)
    """
    from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
    from sam3d_objects.data.dataset.tdfy.pose_target import (
        InstancePose, ScaleShiftInvariant,
    )

    device = R_pt3d.device

    # Undo downsample_factor on scale
    scale_undone = s_pt3d.unsqueeze(0) / downsample_factor  # (1, 3)

    # Build InstancePose — rotation as quaternion.
    # matrix_to_quaternion is a pure bijection (3x3 ↔ quaternion); pass R_pt3d
    # directly so the round-trip through from_instance_pose → to_instance_pose
    # is exact (SSI is rotation-free, so compose/decompose preserves R).
    quat = matrix_to_quaternion(R_pt3d.unsqueeze(0))  # (1, 4)
    # No sign flip — matches original SAM3D (facebookresearch/sam-3d-objects).

    instance_pose = InstancePose(
        instance_scale_l2c=scale_undone,
        instance_position_l2c=t_pt3d.unsqueeze(0),  # (1, 3)
        instance_quaternion_l2c=quat,
        scene_scale=pointmap_scale.flatten().to(device),
        scene_shift=pointmap_shift.flatten().to(device),
    )

    # SAM3D native SSI inversion: camera-space → SSI-normalized space
    pose_target = ScaleShiftInvariant.from_instance_pose(instance_pose)

    # Extract SSI-space components
    ssi_scale = pose_target.x_instance_scale       # (1, 3)
    ssi_rotation = pose_target.x_instance_rotation  # (1, 4) quaternion
    ssi_translation = pose_target.x_instance_translation  # (1, 3)

    # Quaternion → rotation matrix → 6D (first two columns)
    R_ssi = quaternion_to_matrix(ssi_rotation).squeeze(0)  # (3, 3)
    rot_6d = torch.cat([R_ssi[:, 0], R_ssi[:, 1]])  # (6,)

    # Normalize 6D rotation
    mean = _ROTATION_6D_MEAN.to(device)
    std = _ROTATION_6D_STD.to(device)
    raw_6drot_normalized = (rot_6d - mean) / std

    # Log-scale: keep all 3 components to match backbone in_channels=3.
    # For isotropic GT scale, all 3 are identical log(s).
    raw_log_scale = torch.log(ssi_scale.clamp(min=1e-8))  # (1, 3)

    return {
        "6drotation_normalized": raw_6drot_normalized,        # (6,)
        "translation": ssi_translation.squeeze(0),             # (3,)
        "scale": raw_log_scale.squeeze(0),                     # (3,)
        "translation_scale": torch.ones(1, device=device),     # always 1.0
    }
