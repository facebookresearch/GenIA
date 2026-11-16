"""MVSynth (``dataset=mvsynth``): synthetic multi-object scenes from GSO meshes.

Laid out as ``{path}/{scene}/``: the ``mvcustom`` layout
(``rgbs/``, ``segmentations/``) plus ``gt/`` -- ``depth/{NNNNN}.tiff`` (float32 z-depth, 0 =
background), ``cameras.json`` (one K, one camera-to-world per view, held-out views included)
and ``objects.json``.  This module is the reader for what ``Sequence`` consumes from ``gt/``:
the GT depth, K and camera poses of the INPUT views.  The cameras are already R3
(X right, Y down = gravity, Z forward) and c2w, so nothing is converted on load.
"""

import json
import os
from typing import List

import numpy as np


def load_mvsynth_cameras(scene_dir: str) -> dict:
    """Parse ``{scene_dir}/gt/cameras.json``, raising if it is missing or has no train views."""
    path = os.path.join(scene_dir, "gt", "cameras.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found: not an MVSynth scene directory.")
    with open(path) as f:
        cams = json.load(f)
    if not any(c["split"] == "train" for c in cams["cameras"]):
        raise ValueError(f"{path}: no 'train' cameras.")
    return cams


def mvsynth_c2w(scene_dir: str) -> List[np.ndarray]:
    """Camera-to-world (4x4 float32, R3/OpenCV) of every INPUT view, in view order."""
    return [np.asarray(c["c2w"], dtype=np.float32)
            for c in load_mvsynth_cameras(scene_dir)["cameras"] if c["split"] == "train"]


def mvsynth_k(scene_dir: str, H: int, W: int) -> np.ndarray:
    """The scene's pinhole K (3x3 float32) SCALED to an ``(H, W)`` image: the stored K belongs
    to the stored render size, so a run that downscales must scale fx, fy, cx, cy with it."""
    cams = load_mvsynth_cameras(scene_dir)
    sx, sy = W / cams["size"][0], H / cams["size"][1]
    K = np.asarray(cams["K"], dtype=np.float64)
    return np.array([[K[0, 0] * sx, 0, K[0, 2] * sx],
                     [0, K[1, 1] * sy, K[1, 2] * sy],
                     [0, 0, 1]], dtype=np.float32)
