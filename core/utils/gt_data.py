# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Dataset ground-truth geometry: what a dataset ships about its cameras and meshes.

Read by data loading (``Sequence``, depth), FINAL's exports and evaluation:

* OursActionBench (``camera.json``): the fitted camera (:func:`load_actionbench_camera_fit`),
  its intrinsics at a given resolution, and the object-local rotation presets that align a
  predicted mesh with the benchmark's frame (:func:`actionbench_local_rotation_matrix`).
* Mesh loading with that orientation applied (:func:`_load_and_orient_mesh`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


# Named local-rotation presets (3x3 row-vector matrices; ``p_new = p_old @ R^T``,
# equivalently ``p_new_col = R @ p_old_col``).  Add new entries here when
# experimenting; keep them descriptive so YAML config stays readable.
_ACTIONBENCH_ROTATION_PRESETS: "dict[str, np.ndarray]" = {
    "identity":   np.eye(3, dtype=np.float32),
    "flip_x_180": np.diag([ 1.0, -1.0, -1.0]).astype(np.float32),
    "flip_y_180": np.diag([-1.0,  1.0, -1.0]).astype(np.float32),
    "flip_z_180": np.diag([-1.0, -1.0,  1.0]).astype(np.float32),
    "rot_x_+90":  np.array([[1, 0, 0], [0, 0, -1], [0, 1,  0]], dtype=np.float32),
    "rot_x_-90":  np.array([[1, 0, 0], [0, 0,  1], [0, -1, 0]], dtype=np.float32),
    "rot_y_+90":  np.array([[0, 0, 1], [0, 1,  0], [-1, 0, 0]], dtype=np.float32),
    "rot_y_-90":  np.array([[0, 0, -1], [0, 1, 0], [1, 0,  0]], dtype=np.float32),
    "rot_z_+90":  np.array([[0, -1, 0], [1, 0, 0], [0, 0,  1]], dtype=np.float32),
    "rot_z_-90":  np.array([[0,  1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32),
}


def actionbench_local_rotation_matrix(preset_or_matrix) -> np.ndarray:
    """Resolve ``local_rotation`` to a 3x3 ``np.float32`` rotation matrix.

    Accepts:

    - ``None`` → identity.
    - A preset name string (see :data:`_ACTIONBENCH_ROTATION_PRESETS`).
    - A list/tuple of preset names — **composed left-to-right**, i.e. the
      first preset is applied first.  Mathematically the result is
      ``R = R_n @ ... @ R_1`` (column-vector convention) so that
      ``p_final = R @ p_raw``.
    - A 3x3 array-like — used directly.
    """
    if preset_or_matrix is None:
        return _ACTIONBENCH_ROTATION_PRESETS["identity"].copy()
    if isinstance(preset_or_matrix, str):
        if preset_or_matrix not in _ACTIONBENCH_ROTATION_PRESETS:
            raise ValueError(
                f"Unknown actionbench local_rotation preset {preset_or_matrix!r}; "
                f"available: {sorted(_ACTIONBENCH_ROTATION_PRESETS)}"
            )
        return _ACTIONBENCH_ROTATION_PRESETS[preset_or_matrix].copy()
    # Sequence of preset names (or omegaconf ListConfig, which is iterable
    # but not a list).  Treated as composition in iteration order.
    if not isinstance(preset_or_matrix, np.ndarray):
        try:
            items = list(preset_or_matrix)
        except TypeError:
            items = None
        if items is not None and items and all(isinstance(x, str) for x in items):
            R = _ACTIONBENCH_ROTATION_PRESETS["identity"].copy()
            for name in items:
                if name not in _ACTIONBENCH_ROTATION_PRESETS:
                    raise ValueError(
                        f"Unknown actionbench local_rotation preset {name!r} in "
                        f"sequence {items!r}; available: "
                        f"{sorted(_ACTIONBENCH_ROTATION_PRESETS)}"
                    )
                R = _ACTIONBENCH_ROTATION_PRESETS[name] @ R
            return R
    R = np.asarray(preset_or_matrix, dtype=np.float32)
    if R.shape != (3, 3):
        raise ValueError(
            f"actionbench local_rotation must be a preset name, a list of "
            f"preset names, or a 3x3 matrix; got shape {R.shape}"
        )
    return R


def load_actionbench_camera_fit(scene_name: str,
                                  data_root: "str | Path" = "data/oursactionbench"
                                  ) -> dict:
    """Load the per-scene fitted camera (intrinsics + per-frame extrinsics).

    Shipped with each OursActionBench scene as ``{data_root}/{scene}/camera.json``.
    Returns the parsed dict with keys ``focal_px``, ``image_hw``,
    ``fov_deg``, and ``per_frame_cameras`` (list of T entries each with
    ``c2w``, ``w2c``, ``R_w2c``, ``t_w2c``, ``az_deg``, ``el_deg``,
    ``distance``, ``K``).

    OursActionBench camera.json (``camera_json_version`` >= 2) additionally
    carries ``per_frame_test_cameras`` (same per-entry schema): one held-out
    upper-hemisphere NVS view per timestamp. The key is normalised to
    ``None`` for a json without it, so callers can rely on it always being
    present.

    Raises ``FileNotFoundError`` if the per-scene camera.json is missing —
    every scene must have a fitted camera before pipeline use.
    """
    import json
    path = Path(data_root) / scene_name / "camera.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Scene {scene_name!r} has no fitted camera at {path}."
        )
    cam = json.loads(path.read_text())
    cam.setdefault("per_frame_test_cameras", None)
    return cam


def actionbench_intrinsics(H: int, W: int, scene_name: str,
                            data_root: "str | Path" = "data/oursactionbench") -> np.ndarray:
    """Return the per-scene fitted camera intrinsics scaled to (H, W).

    Square pixels, principal point at image center, fov constant — so focal
    scales linearly with resolution from the saved reference. Reads the
    focal length from ``{data_root}/{scene}/camera.json`` (raises if
    missing).
    """
    cam = load_actionbench_camera_fit(scene_name, data_root)
    focal_ref = float(cam["focal_px"])
    res_ref = float(cam["image_hw"][0])
    s = float(H) / res_ref
    fx = fy = focal_ref * s
    cx = (W - 1) * 0.5
    cy = (H - 1) * 0.5
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                    dtype=np.float32)


def _load_and_orient_mesh(
    mesh_path: str, local_rotation = "identity",
) -> Tuple[np.ndarray, np.ndarray]:
    """Read a triangle mesh and apply ``local_rotation`` to its vertices.

    Both ActionMesh ``.glb`` and GSO ``.obj`` files are stored Z-up (floor
    at z=0), while the SLAT decoder expects Y-up.  Pick the rotation that
    fits your data via the preset names in :data:`_ACTIONBENCH_ROTATION_PRESETS`
    (``identity``, ``flip_{x,y,z}_180``, ``rot_{x,y,z}_{+,-}90``) — or
    pass a 3x3 matrix.  Default is ``"identity"`` (no rotation); ``"rot_x_+90"``
    maps Z-up to Y-up.

    Returns
    -------
    verts : np.ndarray ``(M, 3)`` float64
    faces : np.ndarray ``(F, 3)`` int32, triangle indices
    """
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if not mesh.has_triangles():
        raise ValueError(f"Mesh file has no triangles: {mesh_path}")

    verts = np.asarray(mesh.vertices, dtype=np.float64).copy()
    faces = np.asarray(mesh.triangles, dtype=np.int32).copy()

    R = actionbench_local_rotation_matrix(local_rotation)
    if not np.allclose(R, np.eye(3, dtype=np.float32)):
        # Same row-vec convention used by the actionbench loader:
        # ``v_new_row = v_old_row @ R.T``  (equivalently  v_new_col = R @ v_old_col).
        verts = verts @ R.T.astype(np.float64)
    return verts, faces
