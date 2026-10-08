# Copyright (c) Meta Platforms, Inc. and affiliates.

"""SAM3D voxel-grid normalisation helper.

The SAM3D bbox-centre + max-extent isotropic-scale normalisation
(``compute_sam3d_normalization``), used by
``shape_inversion.py:_voxelise_with_normalization`` (the Stage-1 mesh
voxeliser).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def compute_sam3d_normalization(
    points: np.ndarray,
    *,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, float]:
    """Bbox-centre + max-extent isotropic-scale normalisation (SAM3D convention).

    Returns ``(center, scale)`` such that the transformed points
    ``(points - center) / scale`` fit in ``[-0.5, 0.5]^3`` along the
    longest axis; the other two axes get a smaller extent.

    Centre = bbox midpoint; scale = max axis extent.  Used for mesh
    voxelisation (``shape_inversion.py``).

    Parameters
    ----------
    points : np.ndarray, shape ``(N, 3)``
        Points to normalise.  At least 1 row.
    eps : float
        Lower bound on ``scale`` to avoid division by zero on degenerate
        inputs (all points coincident).

    Returns
    -------
    center : np.ndarray, shape ``(3,)``, dtype matching input.
    scale  : float
    """
    points = np.asarray(points)
    vmin = points.min(axis=0)
    vmax = points.max(axis=0)
    center = (vmax + vmin) / 2.0
    scale = max(float((vmax - vmin).max()), eps)
    return center, scale
