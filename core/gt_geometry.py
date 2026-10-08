# Copyright (c) Meta Platforms, Inc. and affiliates.

"""GT geometry / pose injection primitives for the GT_SHAPES_INVERSION block.

The block runner in :mod:`genia.core.main` composes these helpers:

* :func:`_resolve_gso_mesh_path` / :func:`_resolve_temporal_surfaces_path` —
  per-mode mesh path resolvers.
* :func:`voxelize_mesh_to_canonical_coords` — voxelise GT mesh into the
  ``canonical_shape_coords`` layout (used in ``mode="global"``).
* :func:`gt_shape_latent_from_mesh` — invert a mesh into a Stage-1 shape
  latent (used in both modes).
* :func:`populate_gt_object_poses_in_state` — derive per-view object poses
  from each frame's c2w (used in ``mode="global"`` + ``load_gt_poses=true``,
  GSO render setup).
* :func:`populate_gt_raw_modalities_in_state` — synthesise raw Stage-1 pose
  modalities + SSI constants so a raw-to-decoded round-trip reproduces the
  GT pose (paired with the previous helper).

Wraps :func:`genia.core.shape_inversion.voxelize_mesh` and
:func:`genia.core.shape_inversion.mesh_to_shape_tokens` plus
``PipelineState`` mutation methods.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from genia.core.utils.gt_data import (
    _ACTIONBENCH_ROTATION_PRESETS,
    actionbench_local_rotation_matrix,
    load_actionbench_camera_fit,
)


def _resolve_gso_mesh_path(
    scene_name: str, gt_data_root: str = "data/gso30",
) -> Optional[Path]:
    """Return the GSO GT mesh path (``{root}/{scene}/meshes/model.obj``).

    Returns ``None`` if the path does not exist (caller should fall back to
    the pipeline-produced canonical shape rather than crash).
    """
    p = Path(gt_data_root) / scene_name / "meshes" / "model.obj"
    return p if p.exists() else None


def _resolve_temporal_surfaces_path(
    scene_name: str, obj_idx: int, frame_idx: int,
    gt_data_root: str = "data/davis_actionmesh",
    scene_subdir: str = "",
) -> Path:
    """Return the per-frame GT mesh path for a "temporal surfaces" dataset.

    Layout: ``{root}/{scene}/{scene_subdir}/obj_NNN/mesh_NN.glb`` where
    ``NNN = obj_idx-1`` (background ``obj_idx=0`` has no GT mesh; foreground
    starts at 1).

    Used by both ActionMesh (``data/davis_actionmesh``) and OursActionBench
    (``data/oursactionbench``) — same layout, different ``gt_data_root``.
    ``scene_subdir`` is empty for both (a no-op in the join): it exists for
    layouts that keep the meshes in a named subdir of the scene rather than at
    its root, e.g. ``dyncustom``'s ``{scene}/actionmesh/`` beside the frames.
    """
    return (
        Path(gt_data_root) / scene_name / scene_subdir
        / f"obj_{obj_idx - 1:03d}" / f"mesh_{frame_idx:02d}.glb"
    )


def voxelize_mesh_to_canonical_coords(
    mesh_path: str,
    grid_size: int = 64,
    dilate: int = 0,
    *,
    local_rotation = "identity",
) -> np.ndarray:
    """Voxelise a triangle mesh into ``(N, 3)`` int32 canonical voxel indices.

    Thin adapter over :func:`genia.core.shape_inversion.voxelize_mesh`,
    which produces a dense ``(1, 1, G, G, G)`` occupancy tensor; this helper
    extracts the occupied indices and returns them in the
    ``state.canonical_shape_coords[obj_idx]`` layout.  ``local_rotation``
    is forwarded to the underlying loader.
    """
    from genia.core.shape_inversion import voxelize_mesh

    occ = voxelize_mesh(
        str(mesh_path), grid_size=grid_size, dilate=dilate,
        local_rotation=local_rotation,
    )
    occ_np = occ[0, 0].cpu().numpy() > 0.5
    if not occ_np.any():
        raise RuntimeError(f"Voxelisation of {mesh_path} produced 0 voxels")
    ix, iy, iz = np.where(occ_np)
    return np.stack([ix, iy, iz], axis=-1).astype(np.int32)


def _load_and_normalize_temporal_surfaces(
    scene_name: str,
    obj_idx: int,
    frame_indices: "list[int]",
    canonical_frame_idx: int,
    gt_data_root: str = "data/davis_actionmesh",
    *,
    scene_subdir: str = "",
    local_rotation = "identity",
    static_mesh_path: "Optional[Path]" = None,
    subdivide_passes: int = 0,
) -> dict:
    """Shared provider helper for the "temporal surfaces" GT-shape layout
    (per-frame ``obj_NNN/mesh_NN.glb`` files at ``gt_data_root``): load +
    Y-up-orient + bbox-normalise the meshes for one object.

    Used by both ActionMesh and OursActionBench (same layout, different
    ``gt_data_root``).

    Mesh source: per-frame GLBs by default, but the un-welded
    ``deformations_vertices.npy (T,V,3)`` + ``deformations_faces.npy`` arrays
    are used instead **iff** the canonical GLB welds in open3d (loaded vertex
    count < the .npy's).  open3d welds spatially-coincident GLB vertices at
    load, and on self-contacting meshes (e.g. OAB) the weld set drifts per
    frame, breaking fixed-topology.  Substituting the .npy is only valid when
    it shares the GLB's load frame, which holds exactly in the welding case;
    non-welding GLBs (ActionMesh, already
    fixed-topology, .npy in a different raw frame) keep the GLB path.

    Each frame is independently bbox-normalised to its own ``[-0.5, 0.5]³`` cube
    (frame-i self-norm), matching what Stage-1 ``(R, t, s)`` expects to operate
    on; the canonical-frame normalisation defines the canonical-norm reference
    frame.  This divides out the object's per-frame size, which the per-frame
    Sim(3) scale then carries.

    Topology check: every per-frame mesh must have the same vertex count and
    face count as the canonical mesh (fixed-topology requirement; the .npy
    source satisfies this by construction).

    Returns
    -------
    dict with keys:
      canonical_verts_raw : np.ndarray ``(M, 3)`` float64
        Y-up-oriented canonical mesh vertices, raw (no bbox-norm).
      canonical_verts     : np.ndarray ``(M, 3)`` float64
        Canonical mesh in canonical-norm ``[-0.5, 0.5]³``.
      faces               : np.ndarray ``(F, 3)`` int32
        Triangle indices, shared across all frames.
      canonical_center    : np.ndarray ``(3,)`` float64
        Bbox center used for canonical normalisation.
      canonical_scale     : float
        Bbox max-extent used for canonical normalisation.
      per_frame_verts_raw : dict[int, np.ndarray ``(M, 3)`` float64]
        Y-up-oriented per-frame mesh vertices, raw.
      per_frame_verts     : dict[int, np.ndarray ``(M, 3)`` float64]
        Per-frame mesh in frame-i self-norm.
      bbox_scale_ratios   : dict[int, float]
        ``scale_i / canonical_scale`` per frame (diagnostic; returned, not logged).

    Raises
    ------
    ValueError
        ``canonical_frame_idx`` not in ``frame_indices``, or any mesh has a
        degenerate bbox (all verts collinear).
    FileNotFoundError
        Missing per-frame mesh file under ``gt_data_root``.
    RuntimeError
        Vertex / face count mismatch across frames (fixed-topology violated).
    """
    from genia.core.utils.gt_data import _load_and_orient_mesh

    if canonical_frame_idx not in frame_indices:
        raise ValueError(
            f"canonical_frame_idx={canonical_frame_idx} not in "
            f"frame_indices={list(frame_indices)} (obj_idx={obj_idx})"
        )

    canonical_path = (
        static_mesh_path if static_mesh_path is not None
        else _resolve_temporal_surfaces_path(
            scene_name, obj_idx, canonical_frame_idx, gt_data_root=str(gt_data_root),
            scene_subdir=str(scene_subdir),
        )
    )
    if not canonical_path.exists():
        raise FileNotFoundError(
            f"Canonical mesh not found: {canonical_path} "
            f"(scene={scene_name}, obj_idx={obj_idx}, frame={canonical_frame_idx})"
        )

    # ── Mesh source: per-frame ``deformations_*.npy`` vs per-frame GLBs ────────
    # open3d welds spatially-coincident GLB vertices at load; on self-contacting
    # meshes (e.g. OAB) the weld set drifts per frame, breaking fixed-topology.
    # The ``deformations_{vertices,faces}.npy`` arrays are the un-welded,
    # fixed-topology source.  Use them ONLY when the GLB actually welds (open3d
    # V < npy V) — which is exactly when the .npy shares the GLB's load frame.
    # Non-welding GLBs (ActionMesh: already fixed-topology, .npy in a
    # *different* raw frame) and GSO single-mesh keep the GLB path.
    #
    # Note: the .npy could serve as the single fixed-topology source for every
    # dataset that ships it once each scene's .npy→GLB-oriented transform is
    # resolved (e.g. Procrustes on the shared-topology canonical verts), since
    # the canonical/pipeline frame is the GLB-oriented one.
    npy_v_path = canonical_path.parent / "deformations_vertices.npy"
    npy_f_path = canonical_path.parent / "deformations_faces.npy"
    npy_verts = npy_faces = npy_rotation = None
    use_npy = False
    if static_mesh_path is None and npy_v_path.exists() and npy_f_path.exists():
        V_can_glb = _load_and_orient_mesh(
            str(canonical_path), local_rotation=local_rotation,
        )[0]
        if V_can_glb.shape[0] < int(np.load(npy_v_path, mmap_mode="r").shape[1]):
            use_npy = True                          # GLB welds → use the .npy source
            npy_verts = np.load(npy_v_path)                  # (T, V, 3)
            npy_faces = np.load(npy_f_path).astype(np.int32)
            R = actionbench_local_rotation_matrix(local_rotation)
            if not np.allclose(R, np.eye(3, dtype=R.dtype)):
                npy_rotation = R.T.astype(np.float64)        # row-vec: v @ Rᵀ

    def _load_oriented(frame_int: int):
        """Oriented (verts, faces) for one frame from the chosen source."""
        if use_npy:
            if frame_int >= npy_verts.shape[0]:
                raise IndexError(
                    f"frame {frame_int} out of range for deformations_vertices.npy "
                    f"T={npy_verts.shape[0]} (scene={scene_name}, obj_idx={obj_idx})"
                )
            v = npy_verts[frame_int].astype(np.float64)
            return (v if npy_rotation is None else v @ npy_rotation), npy_faces
        path = (
            static_mesh_path if static_mesh_path is not None
            else _resolve_temporal_surfaces_path(
                scene_name, obj_idx, frame_int, gt_data_root=str(gt_data_root),
                scene_subdir=str(scene_subdir),
            )
        )
        if not path.exists():
            raise FileNotFoundError(
                f"Frame {frame_int} mesh not found: {path} "
                f"(scene={scene_name}, obj_idx={obj_idx})"
            )
        return _load_and_orient_mesh(str(path), local_rotation=local_rotation)

    V_can, F_can = _load_oriented(canonical_frame_idx)
    M = V_can.shape[0]

    vmin = V_can.min(axis=0)
    vmax = V_can.max(axis=0)
    center_can = (vmax + vmin) / 2.0
    scale_can = float((vmax - vmin).max())
    if scale_can <= 0:
        raise ValueError(
            f"Canonical mesh has degenerate bbox (scale={scale_can}): {canonical_path}"
        )

    per_frame_verts_raw: "dict[int, np.ndarray]" = {}
    per_frame_verts: "dict[int, np.ndarray]" = {}
    bbox_scale_ratios: "dict[int, float]" = {}

    _bbox: "dict[int, tuple]" = {}          # frame -> (center_i, scale_i), measured once
    for frame_int in frame_indices:
        V_i, F_i = _load_oriented(frame_int)
        if V_i.shape[0] != M:
            raise RuntimeError(
                f"Topology mismatch at frame {frame_int}: vertex count "
                f"{V_i.shape[0]} differs from canonical {M} "
                f"(scene={scene_name}, obj_idx={obj_idx}). Fixed-topology "
                f"requirement violated."
            )
        if F_i.shape[0] != F_can.shape[0]:
            raise RuntimeError(
                f"Topology mismatch at frame {frame_int}: face count "
                f"{F_i.shape[0]} differs from canonical {F_can.shape[0]} "
                f"(scene={scene_name}, obj_idx={obj_idx})."
            )

        vmin_i = V_i.min(axis=0)
        vmax_i = V_i.max(axis=0)
        scale_i = float((vmax_i - vmin_i).max())
        if scale_i <= 0:
            raise ValueError(
                f"Frame {frame_int} mesh has degenerate bbox (scale={scale_i}) "
                f"(scene={scene_name}, obj_idx={obj_idx}, "
                f"source={'npy' if use_npy else 'glb'})"
            )
        bbox_scale_ratios[frame_int] = scale_i / scale_can
        _bbox[frame_int] = ((vmax_i + vmin_i) / 2.0, scale_i)
        per_frame_verts_raw[frame_int] = V_i

    V_can_norm = (V_can - center_can) / scale_can  # (M, 3) canonical-norm
    for frame_int, V_i in per_frame_verts_raw.items():
        center_i, scale_i = _bbox[frame_int]
        per_frame_verts[frame_int] = (V_i - center_i) / scale_i  # frame-i self-norm

    out = {
        "canonical_verts_raw": V_can,
        "canonical_verts": V_can_norm,
        "faces": F_can,
        "canonical_center": center_can,
        "canonical_scale": scale_can,
        "per_frame_verts_raw": per_frame_verts_raw,
        "per_frame_verts": per_frame_verts,
        "bbox_scale_ratios": bbox_scale_ratios,
    }
    if subdivide_passes > 0:
        # 1-to-4 midpoint subdivision per frame; deterministic given the
        # shared input faces, so the new mid-edge verts map 1:1 across
        # frames (fixed-topology correspondence preserved).
        import trimesh
        def _subdiv(v, f, n):
            m = trimesh.Trimesh(vertices=v, faces=f, process=False)
            for _ in range(n):
                m = m.subdivide()
            return (np.asarray(m.vertices, dtype=v.dtype),
                    np.asarray(m.faces, dtype=f.dtype))
        V_can_d, F_d = _subdiv(out["canonical_verts"], out["faces"], subdivide_passes)
        V_can_raw_d, _ = _subdiv(out["canonical_verts_raw"], out["faces"], subdivide_passes)
        pf_d, pf_raw_d = {}, {}
        for fi, v in out["per_frame_verts"].items():
            v_d, f_d = _subdiv(v, out["faces"], subdivide_passes)
            assert np.array_equal(f_d, F_d), (
                f"Subdivision produced inconsistent topology for frame {fi}"
            )
            pf_d[fi] = v_d
            pf_raw_d[fi], _ = _subdiv(out["per_frame_verts_raw"][fi], out["faces"], subdivide_passes)
        out["canonical_verts"] = V_can_d
        out["canonical_verts_raw"] = V_can_raw_d
        out["faces"] = F_d
        out["per_frame_verts"] = pf_d
        out["per_frame_verts_raw"] = pf_raw_d
    return out


def _bin_vertices_to_voxels(V_norm, grid_size: int = 64):
    """Bin (M, 3) normalised verts into a voxel grid.

    Returns ``(coords (L, 3) int32, vox_per_vertex (M,) int64, first_vertex_idx
    (L,) int64)`` — the unique occupied cells, each vertex's cell row, and the
    index of the first vertex that landed in each cell.
    """
    vox = np.clip(
        np.floor((np.clip(V_norm, -0.5 + 1e-6, 0.5 - 1e-6) + 0.5) * grid_size),
        0, grid_size - 1,
    ).astype(np.int64)                                  # (M, 3)
    keys = vox[:, 0] * grid_size * grid_size + vox[:, 1] * grid_size + vox[:, 2]
    unique_keys, first_idx, inv = np.unique(
        keys, return_index=True, return_inverse=True,
    )
    coords = np.stack([
        unique_keys // (grid_size * grid_size),
        (unique_keys // grid_size) % grid_size,
        unique_keys % grid_size,
    ], axis=-1).astype(np.int32)                        # (L, 3)
    return coords, inv.astype(np.int64), first_idx.astype(np.int64)


# =====================================================================
# Per-frame voxel correspondence (mesh-derived; two methods)
#
# Both methods discretise the SAME fixed-topology GT meshes: the canonical
# mesh defines the canonical voxel grid + a per-(sample|vertex)→canonical-voxel
# map, and each frame's mesh is discretised the same way so per-frame voxels map
# back to canonical rows.  Selected by ``perframe_voxelization`` ("surface" |
# "vertex").  Consumed by ``appearance_init=canonical_unified``.
# =====================================================================

def _n_per_tri_from_edges(V_norm, F, sample_edge, grid_size, n_max=48):
    """Per-triangle barycentric subdivision count from the canonical edge lengths.

    ``V_norm`` is in canonical-norm (longest bbox axis spans 1.0), so an edge of
    length ``e`` spans ``e·grid_size`` voxels; ``n = ceil(edge_voxels / sample_edge)``
    places lattice samples ~``sample_edge`` voxels apart.  Clamped to ``[1, n_max]``
    (``n=1`` ⇒ just the 3 vertices).
    """
    V = np.asarray(V_norm, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    e = np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1)
    e = np.maximum(e, np.linalg.norm(V[F[:, 1]] - V[F[:, 2]], axis=1))
    e = np.maximum(e, np.linalg.norm(V[F[:, 2]] - V[F[:, 0]], axis=1))
    n = np.ceil(e * grid_size / max(float(sample_edge), 1e-6))
    return np.clip(n.astype(np.int64), 1, int(n_max))


def _barycentric_lattice(V_norm, F, n_per_tri):
    """Sample every triangle on a barycentric lattice ``(i,j,k)/n``.

    Returns ``(S, 3)`` sample positions (area-adaptive: a triangle with subdivision
    ``n`` contributes ``(n+1)(n+2)/2`` samples).  Vectorised by grouping faces with
    equal ``n``.  ``n=1`` reproduces the triangle's 3 vertices.  The sample ORDER is
    deterministic given ``(F, n_per_tri)`` — so the same ``n_per_tri`` on a different
    pose yields the *same-indexed* samples (fixed-barycentric correspondence).
    """
    V = np.asarray(V_norm, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    n_per_tri = np.asarray(n_per_tri, dtype=np.int64)
    out = []
    for n in np.unique(n_per_tri):
        ij = [(i, j, n - i - j) for i in range(n + 1) for j in range(n + 1 - i)]
        bary = np.asarray(ij, dtype=np.float64) / float(n)        # (P, 3)
        Vt = V[F[n_per_tri == n]]                                 # (Tg, 3, 3)
        out.append(np.einsum("pb,tbc->tpc", bary, Vt).reshape(-1, 3))
    return np.concatenate(out, axis=0) if out else np.zeros((0, 3))


def canonical_voxel_grid_from_mesh(
    V_can_norm, F, sample_edge: float = 0.5, grid_size: int = 64, n_max: int = 48,
):
    """Canonical voxel grid + per-sample→canonical-voxel map (barycentric method).

    Returns ``(canonical_coords (L_canon, 3) int32, canon_vox_per_sample (S,) int64,
    n_per_tri (T,) int64)`` — the occupied canonical cells, the canonical voxel row
    of each barycentric sample, and the fixed per-triangle subdivision pattern (to
    be reused on every frame).
    """
    n_per_tri = _n_per_tri_from_edges(V_can_norm, F, sample_edge, grid_size, n_max)
    samples = _barycentric_lattice(V_can_norm, F, n_per_tri)
    coords, vox_per_sample, _ = _bin_vertices_to_voxels(samples, grid_size)
    return coords, vox_per_sample, n_per_tri


def perframe_voxel_grid_from_mesh(
    V_i_norm, F, canon_vox_per_sample, n_per_tri, grid_size: int = 64,
):
    """Per-frame voxel grid + per-frame-voxel→canonical map (barycentric method).

    Applies the SAME ``n_per_tri`` lattice to the frame-i verts (so sample ``k`` is
    the same surface point as canonical sample ``k`` — fixed-topology barycentric),
    bins, and reads each per-frame cell's first sample's canonical row.  No open3d,
    no nearest-neighbour search.

    Returns ``(coords (L_pf, 3) int32, pf_to_canon (L_pf,) int64)``.
    """
    samples = _barycentric_lattice(V_i_norm, F, n_per_tri)
    coords, _, first_idx = _bin_vertices_to_voxels(samples, grid_size)
    pf_to_canon = np.asarray(canon_vox_per_sample, dtype=np.int64)[first_idx]
    return coords, pf_to_canon.astype(np.int64)


def canonical_voxel_grid_from_surface(
    V_can_norm, F, grid_size: int = 64, dilate: int = 0,
):
    """Canonical voxel grid (open3d surface voxelisation) + per-vertex→row map.

    The grid is the exact triangle-surface rasterisation (matches
    ``gt_canonical_shape_coords``).  Each canonical vertex is mapped to its row;
    the rare vertex whose floored cell isn't occupied snaps to the nearest occupied
    cell centre.  Returns ``(coords (L_canon, 3) int32, canon_row_per_vert (M,) int64)``.
    """
    from scipy.spatial import cKDTree
    from genia.core.shape_inversion import _voxelise_with_normalization

    V = np.asarray(V_can_norm, dtype=np.float64)
    occ = _voxelise_with_normalization(
        V, np.asarray(F, dtype=np.int32), grid_size=grid_size, dilate=dilate,
        center=None, scale=None,
    )
    occ_np = occ[0, 0].cpu().numpy() > 0.5
    if not occ_np.any():
        raise RuntimeError("canonical surface voxelisation produced 0 voxels")
    ix, iy, iz = np.where(occ_np)
    coords = np.stack([ix, iy, iz], axis=-1).astype(np.int32)
    key_to_row = {tuple(c.tolist()): r for r, c in enumerate(coords)}
    vidx = np.clip(
        np.floor((np.clip(V, -0.5 + 1e-6, 0.5 - 1e-6) + 0.5) * grid_size),
        0, grid_size - 1,
    ).astype(np.int64)
    canon_row = np.array(
        [key_to_row.get(tuple(v.tolist()), -1) for v in vidx], dtype=np.int64,
    )
    miss = canon_row < 0
    if miss.any():
        centers = (coords.astype(np.float64) + 0.5) / grid_size - 0.5
        _, nn = cKDTree(centers).query(V[miss], k=1)
        canon_row[miss] = np.atleast_1d(nn).astype(np.int64)
    return coords, canon_row


def perframe_voxel_grid_from_surface(
    V_i_norm, F, canon_row_per_vert, grid_size: int = 64, dilate: int = 0,
):
    """Per-frame voxel grid (surface voxelisation) + per-frame-voxel→canonical map.

    Surface-voxelises the frame mesh, then maps each cell to canonical via its
    nearest frame-i vertex's row (cKDTree — exact where a vertex sits in the cell,
    nearest-vertex for the grazing remainder).

    Returns ``(coords (L_pf, 3) int32, pf_to_canon (L_pf,) int64)``.
    """
    from scipy.spatial import cKDTree
    from genia.core.shape_inversion import _voxelise_with_normalization

    V = np.asarray(V_i_norm, dtype=np.float64)
    occ = _voxelise_with_normalization(
        V, np.asarray(F, dtype=np.int32), grid_size=grid_size, dilate=dilate,
        center=None, scale=None,
    )
    occ_np = occ[0, 0].cpu().numpy() > 0.5
    if not occ_np.any():
        raise RuntimeError("per-frame surface voxelisation produced 0 voxels")
    ix, iy, iz = np.where(occ_np)
    coords = np.stack([ix, iy, iz], axis=-1).astype(np.int32)
    centers = (coords.astype(np.float64) + 0.5) / grid_size - 0.5
    _, nn = cKDTree(V).query(centers, k=1)
    nn = np.atleast_1d(np.asarray(nn, dtype=np.int64))
    pf_to_canon = np.asarray(canon_row_per_vert, dtype=np.int64)[nn]
    return coords, pf_to_canon.astype(np.int64)


def compute_perframe_voxel_correspondence(
    canonical_verts, per_frame_verts, faces,
    method: str = "surface", sample_edge: float = 0.5, grid_size: int = 64,
    dilate: int = 0,
):
    """Canonical voxel grid + per-frame voxel→canonical correspondence, both methods.

    Inputs are the fixed-topology GT meshes (canonical-norm canonical verts, frame-i
    self-norm per-frame verts, shared faces) — i.e. the per-vertex mesh field.

    - ``method="surface"``: open3d surface voxelisation (dense, exact grid) +
      nearest-vertex mapping (cKDTree for grazing voxels).
    - ``method="vertex"``: area-adaptive barycentric-lattice binning (sparser grid,
      exact cKDTree-free correspondence via fixed barycentric samples).

    Returns ``(canonical_coords (L_canon, 3) int32, {frame_int: (coords_pf
    (L_pf, 3) int32, pf_to_canon (L_pf,) int64)})``.
    """
    V_can = np.asarray(canonical_verts, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    per_frame: "dict[int, tuple]" = {}
    if method == "vertex":
        coords, canon_map, n_per_tri = canonical_voxel_grid_from_mesh(
            V_can, F, sample_edge=sample_edge, grid_size=grid_size,
        )
        for fi, V_i in per_frame_verts.items():
            per_frame[int(fi)] = perframe_voxel_grid_from_mesh(
                np.asarray(V_i, dtype=np.float64), F, canon_map, n_per_tri, grid_size,
            )
    elif method == "surface":
        coords, canon_map = canonical_voxel_grid_from_surface(
            V_can, F, grid_size=grid_size, dilate=dilate,
        )
        for fi, V_i in per_frame_verts.items():
            per_frame[int(fi)] = perframe_voxel_grid_from_surface(
                np.asarray(V_i, dtype=np.float64), F, canon_map, grid_size, dilate,
            )
    else:
        raise ValueError(
            f"compute_perframe_voxel_correspondence: method={method!r} "
            f"(expected 'surface' or 'vertex')"
        )
    return coords, per_frame


def collapse_perframe_to_canonical(
    canonical_coords, per_frame, *,
    visibility_alpha: float = 30.0, visibility_min_weight: float = 0.001,
):
    """Collapse per-frame voxel SLAT features onto the shared canonical grid.

    Single source of truth for the ``appearance_init=canonical_unified`` dynamic
    fusion.  Each per-frame voxel carries a feature + a visibility weight and is
    scattered onto its canonical row ``pf_to_canon`` (the GT correspondence).

    Two-level fusion, mirroring the per-view velocity fusion of
    ``canonical_unified``'s static path (``stage2_mv``):

    * **within a frame** several per-frame voxels pool onto a canonical row as a
      visibility-weighted mean (visibility floored to ``visibility_min_weight`` so
      a row seen by only occluded voxels falls back to a plain mean, never zero);
    * **across frames** a canonical row is a ``softmax(visibility_alpha · v̄)``
      weighted mean over the frames that reach it, where ``v̄`` is that frame's
      mean floored-visibility at the row — the SAME relative
      softmax-over-contributors recipe ``stage2_mv`` applies over views.  A row
      occluded in every reaching frame gets a uniform softmax → plain frame-mean
      (no zero hole); a row visible in some frames up-weights those by the
      temperature ``visibility_alpha``.

    With **unit weights** (``visibility_weighting=False``) ``v̄≡1`` so the
    across-frame softmax is uniform and the result is the plain
    per-frame-mean-then-frame-average (one frame, one vote).  Rows reached by no
    voxel at all stay zero.

    Parameters
    ----------
    canonical_coords : (L_canon, 3) or (L_canon, 4) int tensor.
    per_frame : list of ``(feats (L_pf, C), weights (L_pf,), pf_to_canon (L_pf,))``
        torch tensors on a common device; ``weights`` are per-voxel visibility
        (DDA 0/1, or all-ones when off).
    visibility_alpha : softmax temperature for the across-frame fusion (matches
        ``stage2_mv``'s ``visibility_alpha``).
    visibility_min_weight : floor on the (per-voxel and fused) visibility weights.

    Returns
    -------
    (canon_feats (L_canon, C), frame_count (L_canon,), landing_mass (L_canon,))
    torch tensors.  ``frame_count`` is the number of frames with any voxel
    reaching the row (visibility-independent).  ``landing_mass`` is the integer
    landing count each canonical voxel receives, summed over frames
    (``bincount(pf_to_canon)``).
    """
    import torch

    feats0 = per_frame[0][0]
    device, dtype = feats0.device, feats0.dtype
    L = int(canonical_coords.shape[0])
    C = int(feats0.shape[1])
    F = len(per_frame)
    eps = torch.finfo(dtype).eps

    # Per-(frame, canonical-row) aggregates.
    #   gsum  : Σ v̂·feat  (floored-vis-weighted feature sum)
    #   gmass : Σ v̂       (floored-vis-weighted mass → within-frame mean denom)
    #   tmass : Σ 1        (landing count, visibility-independent)
    gsum = torch.zeros(F, L, C, device=device, dtype=dtype)
    gmass = torch.zeros(F, L, device=device, dtype=dtype)
    tmass = torch.zeros(F, L, device=device, dtype=dtype)
    for fi, (feats, weights, pf_to_canon) in enumerate(per_frame):
        vfloor = weights.clamp(min=visibility_min_weight)   # occluded keep a tiny weight
        gsum[fi].index_add_(0, pf_to_canon, vfloor.unsqueeze(1) * feats)
        gmass[fi].index_add_(0, pf_to_canon, vfloor)
        tmass[fi].index_add_(0, pf_to_canon, weights.new_ones(weights.shape[0]))

    reached = tmass > 0                                     # (F, L) any voxel here
    # Within-frame visibility-weighted mean (floored denom → plain mean when all
    # occluded; never zero where reached).
    per_frame_mean = gsum / gmass.clamp(min=eps).unsqueeze(-1)   # (F, L, C)
    # Per-frame mean floored-visibility at the row ∈ [floor, 1]; the across-frame
    # softmax logit.  Unit weights → 1 everywhere → uniform softmax.
    vbar = gmass / tmass.clamp(min=eps)                          # (F, L)

    logits = (visibility_alpha * vbar).masked_fill(~reached, float("-inf"))
    w = torch.softmax(logits, dim=0)                            # over frames; 0 at unreached
    w = torch.where(reached, w.clamp(min=visibility_min_weight), torch.zeros_like(w))
    # Landing mass is an integer >= 1 wherever reached, so capping it at 1 keeps
    # one-frame-one-vote.
    w = w * tmass.clamp(max=1.0)
    w = w / w.sum(dim=0, keepdim=True).clamp(min=eps)           # rows w/ no frame → 0
    canon_feats = (w.unsqueeze(-1) * per_frame_mean).sum(dim=0)  # (L, C); unreached → 0

    frame_count = reached.sum(dim=0).to(dtype)
    landing_mass = tmass.sum(dim=0)
    return canon_feats, frame_count, landing_mass


def _per_vertex_kabsch(
    V_can_norm: np.ndarray,
    V_i_norm: np.ndarray,
    nn_idx: np.ndarray,
) -> "tuple[np.ndarray, np.ndarray]":
    """Vectorised per-vertex Kabsch: best-fit rigid rotation per vertex.

    For each vertex ``v``, fit ``R_v`` aligning the kNN-canonical positions
    (centred at v) to the same neighbour indices in the deformed frame
    (also centred at v).  Reflection-fixed (det > 0).

    Parameters
    ----------
    V_can_norm : np.ndarray ``(V, 3)`` float64
        Canonical mesh vertices in canonical-norm.
    V_i_norm   : np.ndarray ``(V, 3)`` float64
        Frame-i mesh vertices in frame-i self-norm (vertex-aligned with
        V_can_norm by the fixed-topology assumption).
    nn_idx : np.ndarray ``(V, k)`` int64
        Per-vertex spatial-kNN indices into ``[0, V)``.  Must NOT include
        the vertex itself (caller drops the self column).

    Returns
    -------
    R_v          : np.ndarray ``(V, 3, 3)`` float32  SO(3) per-vertex rotation.
    min_singular : np.ndarray ``(V,)``     float32  smallest singular value
                                                    per vertex (rank-deficiency
                                                    diagnostic).
    """
    V = V_can_norm.shape[0]

    # Canonical centred neighbour positions.
    P_can = V_can_norm[nn_idx] - V_can_norm[:, None, :]   # (V, k, 3)
    # Frame-i centred neighbour positions.
    P_i = V_i_norm[nn_idx] - V_i_norm[:, None, :]         # (V, k, 3)

    # H = P_can^T @ P_i per vertex: (V, 3, k) @ (V, k, 3) → (V, 3, 3).
    H = np.einsum("vki,vkj->vij", P_can, P_i)             # (V, 3, 3)

    U, S, Vt = np.linalg.svd(H)                            # (V, 3, 3) each, S: (V, 3)

    # R = V @ U^T.  ``Vt.transpose(0, 2, 1)`` is V; ``U.transpose(0, 2, 1)`` is U^T.
    R_v = np.einsum(
        "vij,vjk->vik", Vt.transpose(0, 2, 1), U.transpose(0, 2, 1),
    )
    dets = np.linalg.det(R_v)
    flip = dets < 0
    if flip.any():
        D = np.broadcast_to(np.eye(3), (V, 3, 3)).copy()
        D[flip, 2, 2] = -1.0
        R_v = np.einsum(
            "vij,vjk,vkl->vil",
            Vt.transpose(0, 2, 1), D, U.transpose(0, 2, 1),
        )

    return R_v.astype(np.float32), S[:, 2].astype(np.float32)


def compute_canonical_mesh_correspondence(
    scene_name: str,
    obj_idx: int,
    frame_indices: "list[int]",
    canonical_frame_idx: int,
    gt_data_root: str = "data/davis_actionmesh",
    *,
    scene_subdir: str = "",
    k_kabsch: int = 8,
    loaded: Optional[dict] = None,
    local_rotation = "identity",
    static_mesh_path: "Optional[Path]" = None,
    subdivide_passes: int = 0,
) -> dict:
    """Provider: persist per-canonical-mesh-vertex correspondences (Φ + R)
    at the natural high resolution of the GT signal.

    Persists the raw per-vertex deformation signal so Stage-2 rendering guidance
    can sample it at decoded primitive positions.

    Algorithm
    ---------
    1. Load + Y-up-orient + per-frame bbox-norm via
       :func:`_load_and_normalize_temporal_surfaces` (shared with the voxel
       provider).  Topology check inside.
    2. Canonical kNN: spatial ``k_kabsch`` nearest neighbours of every
       canonical vertex (excluding self).  Topology-agnostic — avoids the
       1-ring assumption.
    3. Per frame ``i``: for each vertex ``v``, vectorised Kabsch on the
       neighbours' canonical vs frame-i positions, both centered at the
       vertex's own position::

           P_can = V_can[N(v)] − V_can[v]                # (k, 3)
           P_i   = V_i[N(v)]   − V_i[v]                  # (k, 3)
           H     = P_can^T @ P_i                          # (3, 3)
           U,S,Vt = svd(H);  d = sign(det(Vt^T @ U^T))
           R_v[i] = Vt^T @ diag(1, 1, d) @ U^T            # SO(3)

    Known limitation — flat-region rank deficiency
    ----------------------------------------------
    On locally-planar mesh regions the ``k_kabsch`` neighbours all lie in
    the surface tangent plane → ``H`` is rank-2 (smallest singular value
    ≈ 0), so the rotation about the surface-normal axis is numerically
    ill-defined.  In-plane components remain well-defined; the Stage-2
    K=4 IDW blend averages out the noise.  Logged via the
    ``min_singular_value_histogram`` diagnostic.

    Returns
    -------
    dict with keys:
      canonical_verts     : torch.Tensor ``(V, 3)`` float32
        Canonical mesh in canonical-norm ``[-0.5, 0.5]³``.  Identity at the
        canonical frame: ``per_frame_verts[canonical_frame] == canonical_verts``
        bit-for-bit.
      faces               : torch.Tensor ``(F, 3)`` int64
        Triangle indices, fixed across frames.
      per_frame_verts     : dict[int, torch.Tensor ``(V, 3)`` float32]
        Per-frame deformed verts in frame-i self-norm.  Vertex-aligned
        with ``canonical_verts``.
      per_frame_rotations    : dict[int, torch.Tensor ``(V, 3, 3)`` float32]
        Per-vertex SO(3) rigid rotation via kNN-Kabsch.  Identity at the
        canonical frame.
      bbox_scale_ratios   : dict[int, float]
        ``scale_i / scale_canonical`` per frame (diagnostic; returned, not logged).
      diagnostics         : dict
        ``{V, F, k_kabsch, min_singular_value_histogram, min_singular_value_bins}``.

    Raises
    ------
    ValueError
        ``canonical_frame_idx`` not in ``frame_indices``; degenerate bbox;
        ``V <= k_kabsch`` (mesh too small for the kNN window).
    FileNotFoundError
        Missing per-frame mesh file.
    RuntimeError
        Vertex / face count mismatch across frames.
    """
    import torch
    from scipy.spatial import cKDTree

    if loaded is None:
        loaded = _load_and_normalize_temporal_surfaces(
            scene_name, obj_idx, frame_indices, canonical_frame_idx, gt_data_root,
            scene_subdir=scene_subdir,
            local_rotation=local_rotation, static_mesh_path=static_mesh_path,
            subdivide_passes=subdivide_passes,
        )
    V_can_norm = loaded["canonical_verts"].astype(np.float64)  # (V, 3)
    F_can = loaded["faces"]
    V = V_can_norm.shape[0]

    if V <= k_kabsch:
        raise ValueError(
            f"Mesh has V={V} vertices but k_kabsch={k_kabsch} requires "
            f"V > k_kabsch (need k_kabsch+1 entries to drop self after kNN). "
            f"(scene={scene_name}, obj_idx={obj_idx})"
        )

    # 1. Canonical kNN once: for each vertex, its nearest *other* vertices
    #    via cKDTree (Euclidean).  The runtime warp uses closest-face snap
    #    when faces are available (robust to points straddling component
    #    boundaries on coarse multi-component meshes) and falls back to this
    #    Euclidean kNN when there are no faces.
    tree = cKDTree(V_can_norm)
    _, nn_idx_full = tree.query(V_can_norm, k=k_kabsch + 1)  # (V, k+1)
    nn_idx = nn_idx_full[:, 1:].astype(np.int64)             # drop self

    # 2. Per-frame Kabsch.  Vectorised SVD on (V, 3, 3) batch.
    per_frame_rotations: "dict[int, torch.Tensor]" = {}
    per_frame_verts: "dict[int, torch.Tensor]" = {}
    min_sing_per_frame: "list[np.ndarray]" = []

    for frame_int in frame_indices:
        V_i_norm = loaded["per_frame_verts"][frame_int].astype(np.float64)  # (V, 3)

        R_v, min_sing = _per_vertex_kabsch(V_can_norm, V_i_norm, nn_idx)

        per_frame_rotations[frame_int] = torch.from_numpy(R_v).contiguous()
        per_frame_verts[frame_int] = torch.from_numpy(
            V_i_norm.astype(np.float32),
        ).contiguous()
        min_sing_per_frame.append(min_sing)

    canonical_verts_t = torch.from_numpy(
        V_can_norm.astype(np.float32),
    ).contiguous()
    faces_t = torch.from_numpy(F_can.astype(np.int64)).contiguous()

    # Histogram of smallest singular value across (V × N_frames) Kabsch fits
    # — flags how many vertices live in flat regions where R is ill-defined.
    bins = [0.0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, np.inf]
    all_min_sing = np.concatenate(min_sing_per_frame) if min_sing_per_frame else np.zeros(0, dtype=np.float32)
    sing_hist = np.histogram(all_min_sing, bins=bins)[0].tolist()

    return {
        "canonical_verts": canonical_verts_t,
        "faces": faces_t,
        "per_frame_verts": per_frame_verts,
        "per_frame_rotations": per_frame_rotations,
        "bbox_scale_ratios": dict(loaded["bbox_scale_ratios"]),
        "diagnostics": {
            "V": V,
            "F": int(F_can.shape[0]),
            "k_kabsch": int(k_kabsch),
            "min_singular_value_histogram": sing_hist,
            "min_singular_value_bins": [
                "0", "1e-6", "1e-4", "1e-2", "1e-1", "1.0", "inf",
            ],
        },
    }


def populate_gt_object_poses_in_state(
    state,
    sequence,
    mesh_path: str,
    obj_idx: int,
    device,
) -> int:
    """Fill ``state.tokens_by_object[obj_idx][i][1]`` with GT-derived per-frame
    ``rotation``, ``translation``, ``scale`` so downstream consumers (evaluators,
    viz, rg pose decoder) see a populated state — as if Stage 1 had run, but
    with GT poses instead of predicted ones.

    Static-GSO assumption: object at world origin in canonical orientation,
    pose derived from each frame's c2w via :func:`gt_object_poses_from_c2w`.

    Returns the number of frames populated.
    """

    tokens_list = state.tokens_by_object.get(obj_idx)
    if not tokens_list:
        return 0

    c2ws = []
    for fk, _ in tokens_list:
        frame_data = sequence[fk]
        c2ws.append(frame_data.c2w)

    quats, transes, scales = gt_object_poses_from_c2w(mesh_path, c2ws, device)

    new_tokens = []
    for i, (fk, decoder_input) in enumerate(tokens_list):
        di = dict(decoder_input)
        # Match Stage 1's output shapes (see inference_utils.pose_decoder):
        #   rotation: (1, 4) wxyz  ·  translation: (1, 3)  ·  scale: (1, 3)
        # 2D shapes are required by pytorch3d Translate inside make_scene.
        di["rotation"] = quats[i].unsqueeze(0)
        di["translation"] = transes[i].unsqueeze(0)
        di["scale"] = scales[i].unsqueeze(0)
        new_tokens.append((fk, di))
    state.tokens_by_object[obj_idx] = new_tokens
    return len(new_tokens)


def gt_object_poses_from_c2w(
    mesh_path: str,
    c2w_list: list,
    device,
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Derive per-view P3D camera-space object poses from frame c2w.

    Assumes the EscherNet/GSO static-render setup (EscherNet's Blender
    ``normalize_scene``):
    every object is scaled to **unit-cube extent (1.0) and centered at world
    origin** before rendering.  So the canonical Gaussians (also in
    [-0.5, 0.5]³ P3D) need ``scale = 1.0`` and the world-space object
    position is just the origin (no center offset).

    The ``mesh_path`` argument is currently unused (kept for API symmetry
    with :func:`populate_gt_object_poses_in_state` which loads the mesh
    elsewhere) — the rendered world is normalized away from the OBJ-file's
    raw bounds.

    Pose derivation: invert the standard render chain
    canonical_y → canonical_z → world_blender → world_r3 → cam_r3 → cam_p3d:

        canonical_y * 1.0 @ R_pose + trans = cam_p3d
        canonical_y = canonical_z @ R_z2y    # R_z2y applied in voxelize_mesh
                                              # (so canonical_z = canonical_y @ R_z2y.T)
        cam_p3d = cam_r3 @ D                 # P3D ↔ R3, D = diag(-1,-1,1)
        cam_r3  = (world_r3 - t_c2w) @ R_c2w
        world_r3 = world_blender @ R_W.T     # R_W = BLENDER_TO_R3_WORLD
        world_blender = canonical_z * 1.0    # render-world = OBJ Z-up unit cube

    Solving for (R_pose, trans) with the canonical-voxel-grid Y-up rotation
    (introduced in ``shape_inversion.voxelize_mesh`` so the inverted SLAT
    latent decodes a Y-up shape, matching the SLAT decoder's training):
        R_pose = R_z2y.T @ R_W.T @ R_c2w @ D
        trans  = (-t_c2w @ R_c2w) @ D

    Returns
    -------
    quats : torch.Tensor ``(N, 4)`` wxyz
    transes : torch.Tensor ``(N, 3)``
    scales : torch.Tensor ``(N, 3)``  (always 1.0 — render-world unit cube)
    """
    import torch

    from genia.core.utils.eval_assets_export import BLENDER_TO_R3_WORLD as R_W_np
    from genia.core.utils.quaternion_ops import matrix_to_quaternion

    R_W = torch.from_numpy(np.asarray(R_W_np, dtype=np.float32)).to(device)
    D = torch.diag(torch.tensor([-1.0, -1.0, 1.0], device=device))
    # Z-up → Y-up row-vector rotation applied in shape_inversion.voxelize_mesh.
    # Must match SAM3D's training convention.
    R_z2y = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], device=device,
    )
    # Local rotation baked into the GT pose: the canonical mesh stays in its
    # native Z-up frame at load time (mesh loader uses local_rotation=identity
    # for GSO).  Compose rot_x_+90 (= R_z2y) on the canonical's left in the
    # pose chain so the Z-up canonical lands in the Y-up render frame the
    # rest of the formula assumes.  Left-composed with the existing R_z2y.T
    # this cancels mathematically (kept explicit here so the intent is clear).
    R_local = R_z2y  # rot_x_+90

    quats, transes = [], []
    for c2w in c2w_list:
        if isinstance(c2w, np.ndarray):
            c2w_t = torch.from_numpy(c2w.astype(np.float32)).to(device)
        else:
            c2w_t = c2w.to(device).float()
        R_c2w = c2w_t[:3, :3]
        t_c2w = c2w_t[:3, 3]
        R_pose = R_local @ R_z2y.T @ R_W.T @ R_c2w @ D
        trans = (-t_c2w @ R_c2w) @ D
        quats.append(matrix_to_quaternion(R_pose.unsqueeze(0)).squeeze(0))
        transes.append(trans)

    quats_t = torch.stack(quats)
    trans_t = torch.stack(transes)
    scales_t = torch.ones((len(c2w_list), 3), device=device)
    _ = mesh_path  # (unused; kept in signature for API symmetry)
    return quats_t, trans_t, scales_t


def gt_shape_latent_from_mesh(
    mesh_path: str, ss_decoder, device,
    *, num_steps: int = 100, prior_weight: float = 0.01,
    prior_target_var: float = 1.0,
    l2_weight: float = 0.0,
    local_rotation = "identity",
    autocast_dtype = None,
):
    """Invert a GT mesh into a Stage-1 shape latent ``(1, 4096, 8)``.

    Returns ``(latent, round_trip_iou)``.  Wraps
    :func:`genia.core.shape_inversion.mesh_to_shape_tokens` and reshapes
    its ``(4096, 8)`` output to add the batch dim that downstream
    consumers (``stage1_batched``) expect.

    Consumed by the GT_SHAPES_INVERSION block runner — the parallel ODE
    needs the GT shape latent (not a Stage-1 inference result) for
    ``gt_shape_trajectory`` to actually follow the GT trajectory.
    100 Adam steps is typically enough for IoU > 0.9 on GSO meshes; no
    caching — re-runs each call.
    """
    from genia.core.shape_inversion import mesh_to_shape_tokens
    tokens, iou = mesh_to_shape_tokens(
        mesh_path, ss_decoder, num_steps=num_steps,
        prior_weight=prior_weight, prior_target_var=prior_target_var,
        l2_weight=l2_weight,
        device=str(device), local_rotation=local_rotation,
        autocast_dtype=autocast_dtype,
    )
    return tokens.unsqueeze(0).to(device).detach(), iou


def gt_shape_latents_from_meshes(
    mesh_paths, ss_decoder, device,
    *, num_steps: int = 100, prior_weight: float = 0.01,
    prior_target_var: float = 1.0,
    l2_weight: float = 0.0,
    local_rotation = "identity",
    chunk_size: int = 16,
    autocast_dtype = None,
):
    """Batched sibling of :func:`gt_shape_latent_from_mesh`.

    Voxelises every mesh on the CPU first (open3d, ~0.33 s/mesh), stacks them
    into one ``(T, 1, 64, 64, 64)`` tensor, and runs a SINGLE batched Adam
    inversion — chunked at ``chunk_size`` frames for memory, which never
    changes the result.

    Returns ``([latent (1, 4096, 8) on device], [iou])``: the same per-frame
    objects ``gt_shape_latent_from_mesh`` returns one at a time, so
    ``_print_inversion_stats`` and ``set_all_perframe_shape_tokens`` consume
    them unchanged.

    Equivalent to inverting each mesh on its own *in exact arithmetic*: the loss
    is a sum of per-sample terms, Adam is element-wise, and every sample shares
    the seed-42 ``z0`` the one-at-a-time path re-draws on every call.  On GPU
    the latents still differ — so does the unbatched path from itself — while
    round-trip IoU agrees to <1e-3; see :func:`invert_decoder`.
    """
    import torch

    from genia.core.shape_inversion import (
        occupancies_to_shape_tokens, voxelize_mesh,
    )

    occ = torch.cat(
        [voxelize_mesh(str(p), local_rotation=local_rotation) for p in mesh_paths],
        dim=0,
    ).to(device)  # (T, 1, 64, 64, 64)
    tokens, ious = occupancies_to_shape_tokens(
        occ, ss_decoder, num_steps=num_steps,
        prior_weight=prior_weight, prior_target_var=prior_target_var,
        l2_weight=l2_weight,
        chunk_size=chunk_size, autocast_dtype=autocast_dtype,
    )
    latents = [tokens[i:i + 1].detach() for i in range(tokens.shape[0])]
    return latents, ious


def populate_gt_raw_modalities_in_state(
    state, sequence, obj_idx: int, device, pipeline_obj,
) -> int:
    """Synthesize raw Stage-1 modalities + SSI constants from the GT decoded
    poses (``rotation``/``translation``/``scale``) already on each frame's
    decoder_input dict.

    Inverse of ``differentiable_pose_decode``: given camera-space GT pose
    plus the SAM3D preprocessor's per-frame SSI ``(scale, shift)``, fills in
    ``raw_ss_modalities``, ``pointmap_scale``, ``pointmap_shift``, and
    ``downsample_factor`` so the round-trip ``raw → decoded`` reproduces
    the GT pose exactly.

    The (scale, shift) MUST come from the same normalizer Stage 1 uses at
    decode time (``pipeline.ss_preprocessor.pointmap_normalizer``), not from
    the generic ``ScaleShiftInvariant.get_scale_and_shift`` formula on the
    masked pointmap — those use different formulas (``scale = nanmean(|pm|)``
    vs ``scale = median_z * scale_factor``) and disagree, biasing the round
    trip and shifting the rendered object in pose-pinned debug runs.

    Must be called after :func:`populate_gt_object_poses_in_state`.
    Returns the number of frames populated.
    """
    import torch
    from pytorch3d.transforms import quaternion_to_matrix

    from genia.core.utils.depth import transform_to_pytorch3d_convention
    from genia.core.utils.pose_token_gt import camera_pose_to_raw_tokens

    tokens_list = state.tokens_by_object.get(obj_idx)
    if not tokens_list:
        return 0

    pointmap_normalizer = pipeline_obj.ss_preprocessor.pointmap_normalizer

    new_tokens = []
    n_filled = 0
    for fk, decoder_input in tokens_list:
        di = dict(decoder_input)
        if "rotation" not in di:
            new_tokens.append((fk, di))
            continue

        # GT decoded pose in PyTorch3D camera space (set by
        # populate_gt_object_poses_in_state).
        R_pt3d = quaternion_to_matrix(di["rotation"]).squeeze(0).to(device)
        t_pt3d = di["translation"].squeeze(0).to(device)
        s_pt3d = di["scale"].squeeze(0).to(device)

        # SSI scale/shift via the same normalizer Stage 1 uses at decode time.
        # CRITICAL: Stage 1 normalizes the *PyTorch3D-converted* pointmap
        # (pose_init.py applies transform_to_pytorch3d_convention before
        # stage1_batched), so we must convert here too.  Without this, the
        # X/Y shift components have flipped signs (R3 ↔ P3D differ by
        # D = diag(-1, -1, 1)), biasing the SSI roundtrip and shifting
        # the rendered object in pose-pinned debug runs.
        frame = sequence[fk]
        pm_p3d_np = transform_to_pytorch3d_convention(frame.pointmap)
        pm_3hw = (
            torch.as_tensor(pm_p3d_np, dtype=torch.float32, device=device)
            .permute(2, 0, 1).contiguous()
        )
        mask_1hw = torch.as_tensor(
            frame.masks[obj_idx], dtype=torch.float32, device=device,
        ).unsqueeze(0)
        ssi_result = pointmap_normalizer.normalize(pm_3hw, mask_1hw)
        scale = ssi_result.scale.to(device)
        shift = ssi_result.shift.to(device)

        raw = camera_pose_to_raw_tokens(
            R_pt3d, t_pt3d, s_pt3d, scale, shift, downsample_factor=1.0,
        )
        # Merge — pose modalities (6drotation_normalized/translation/scale/
        # translation_scale) overwrite, but preserve 'shape' set earlier by
        # the inversion loop (camera_pose_to_raw_tokens has no 'shape' key).
        existing = di.get("raw_ss_modalities", {})
        existing.update(raw)
        di["raw_ss_modalities"] = existing
        di["pointmap_scale"] = scale
        di["pointmap_shift"] = shift
        di["downsample_factor"] = 1.0
        new_tokens.append((fk, di))
        n_filled += 1
    state.tokens_by_object[obj_idx] = new_tokens
    return n_filled


# =====================================================================
# OursActionBench (per-frame surface points with row-index correspondence)
# =====================================================================

# OursActionBench camera: per-scene per-frame cameras live in
# ``{data_root}/{scene}/camera.json``. Load via
# :func:`load_actionbench_camera_fit`; use :func:`actionbench_intrinsics` for
# K and :func:`populate_gt_object_poses_in_state_actionbench` to fold the
# per-frame extrinsics into per-frame object poses. Runtime ``c2w`` stays
# identity — the rig motion is captured entirely by the object poses.
#
# The runtime ``local_rotation`` (a 3x3 ``R`` resolved by
# :func:`actionbench_local_rotation_matrix`) is applied analytically on top
# of each per-frame c2w via :func:`apply_local_rotation_to_c2w`, so you can
# pick any rotation by changing only the ``local_rotation`` config.


# Short aliases — a "flip" is unambiguously 180° around the named axis.
for _ax in ("x", "y", "z"):
    _ACTIONBENCH_ROTATION_PRESETS[f"flip_{_ax}"] = (
        _ACTIONBENCH_ROTATION_PRESETS[f"flip_{_ax}_180"]
    )
del _ax


def apply_local_rotation_to_c2w(c2w: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Transform a c2w matrix expressed in the *raw* surface-point frame to
    the runtime frame implied by applying ``R`` to all points.

    Derivation: if ``p_runtime = R @ p_raw`` (column-vector form), the camera
    that projects ``p_runtime`` is ``c2w_runtime = R_4x4 @ c2w_raw`` (rotation
    block premultiplied by ``R``, position rotated by ``R``).
    """
    R = np.asarray(R, dtype=np.float32)
    R4 = np.eye(4, dtype=c2w.dtype)
    R4[:3, :3] = R
    return R4 @ np.asarray(c2w)


def gt_object_poses_from_actionbench_surfaces(
    per_frame_points: "list[np.ndarray]",
    c2w: "np.ndarray | list[np.ndarray] | np.ndarray",
    device,
    *,
    local_rotation = "identity",
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Derive per-frame P3D camera-space object poses for OursActionBench.

    The inverted Stage-1 latent for frame ``i`` is the canonical_y unit cube
    obtained by self-normalising ``points_i`` (per-frame bbox).  So in
    world frame:

        p_world = canonical_y * scale_i + center_i  (Y-up R3, points already Y-up)

    With a per-frame fitted ``c2w[i]`` and the standard R3 → P3D flip
    ``D = diag(-1, -1, 1)``:

        cam_r3   = (p_world - t_c2w[i]) @ R_c2w[i]
        cam_p3d  = cam_r3 @ D

    Match to the SAM3D row-vec convention ``cam_p3d = canonical_y * scale @ R_pose + trans``:

        R_pose[i] = R_c2w[i] @ D                            (per-frame)
        trans[i]  = (center_i - t_c2w[i]) @ R_c2w[i] @ D    (per-frame)
        scale[i]  = scale_i (broadcast to (3,))

    ``c2w`` may be:
      - A single (4, 4) array — broadcast across all frames (shared camera).
      - A (T, 4, 4) array or list of T (4, 4) arrays — per-frame camera, used
        for the dynamic-camera case where the rig animates per frame.

    No R_z2y or BLENDER_TO_R3_WORLD here (unlike the GSO path) — the
    surface points are already R3 Y-up.
    """
    import torch
    from genia.core.utils.quaternion_ops import matrix_to_quaternion

    T = len(per_frame_points)
    c2w_arr = np.asarray(c2w, dtype=np.float32)
    if c2w_arr.ndim == 2:
        c2w_arr = np.broadcast_to(c2w_arr, (T, 4, 4)).copy()
    if c2w_arr.shape != (T, 4, 4):
        raise ValueError(
            f"gt_object_poses_from_actionbench_surfaces: expected c2w shape "
            f"(4, 4) or ({T}, 4, 4), got {c2w_arr.shape}"
        )

    # Apply the user's local_rotation to each per-frame c2w (raw surface-point
    # frame → runtime frame).  Identity → no-op.
    R_local = actionbench_local_rotation_matrix(local_rotation)
    c2w_runtime = np.stack(
        [apply_local_rotation_to_c2w(c2w_arr[i], R_local) for i in range(T)]
    )
    c2w_t = torch.as_tensor(c2w_runtime, dtype=torch.float32, device=device)
    R_c2w = c2w_t[:, :3, :3]  # (T, 3, 3)
    t_c2w = c2w_t[:, :3, 3]   # (T, 3)
    D = torch.diag(torch.tensor([-1.0, -1.0, 1.0], device=device))

    quats, transes, scales = [], [], []
    for i, points in enumerate(per_frame_points):
        R_i = R_c2w[i]; t_i = t_c2w[i]
        R_pose_i = R_i @ D  # row-vec rotation
        quats.append(matrix_to_quaternion(R_pose_i.unsqueeze(0)).squeeze(0))
        # ``center_i``/``scale_i`` must be measured in the SAME frame as the
        # shape this pose will be applied to.  That shape is the mesh AFTER
        # ``local_rotation`` (``_load_and_orient_mesh`` does ``verts @ R.T``,
        # i.e. the column-vector form ``R @ v``, and the per-frame voxelisation
        # inherits it), so the points must be rotated the same way here.
        #
        # Measuring on the RAW points instead would leave the centre rotated
        # the WRONG way: the composed translation would carry
        # ``c_raw @ R_local`` where it needs ``c_raw @ R_local.T``, displacing
        # the posed object in the image while leaving its size and orientation
        # right.  A no-op when local_rotation is identity.
        pts = np.asarray(points, dtype=np.float32) @ R_local.T.astype(np.float32)
        vmin = pts.min(axis=0)
        vmax = pts.max(axis=0)
        center_i = torch.as_tensor((vmin + vmax) * 0.5, dtype=torch.float32,
                                   device=device)
        scale_i = float((vmax - vmin).max())
        if scale_i <= 0:
            raise ValueError(f"Degenerate per-frame bbox: scale={scale_i}")
        transes.append((center_i - t_i) @ R_i @ D)
        scales.append(torch.full((3,), scale_i, device=device))

    return torch.stack(quats), torch.stack(transes), torch.stack(scales)


def populate_gt_object_poses_in_state_actionbench(
    state, obj_idx: int,
    surfaces_cache: "dict[int, np.ndarray]",
    scene_name: str, device,
    *,
    local_rotation = "identity",
    data_root: "str | Path" = "data/actionbench",
) -> int:
    """Fill ``state.tokens_by_object[obj_idx]`` with GT camera-space object
    poses derived from OursActionBench surface points + the per-scene camera.

    Per-frame camera ``c2w`` is loaded from ``{data_root}/{scene}/camera.json``.
    The runtime camera stays identity; the rig motion is folded into the
    per-frame object poses written here.

    Raises ``FileNotFoundError`` if the per-scene camera.json is missing.

    Returns the number of frames populated.
    """
    tokens_list = state.tokens_by_object.get(obj_idx)
    if not tokens_list:
        return 0
    per_frame_points = []
    frame_indices = []
    for fk, _ in tokens_list:
        fi = int(fk.frame)
        if fi not in surfaces_cache:
            raise RuntimeError(
                f"populate_gt_object_poses_in_state_actionbench: frame {fi} "
                f"missing from surfaces cache (obj_idx={obj_idx})"
            )
        per_frame_points.append(surfaces_cache[fi])
        frame_indices.append(fi)

    cam_fit = load_actionbench_camera_fit(scene_name, data_root)
    pf_list = cam_fit.get("per_frame_cameras")
    if not pf_list:
        raise RuntimeError(
            f"populate_gt_object_poses_in_state_actionbench: per-scene "
            f"camera.json for {scene_name!r} has no per_frame_cameras"
        )
    try:
        c2w_per_frame = np.stack(
            [np.asarray(pf_list[fi]["c2w"], dtype=np.float32)
             for fi in frame_indices],
            axis=0,
        )
    except (IndexError, KeyError) as e:
        raise RuntimeError(
            f"populate_gt_object_poses_in_state_actionbench: per-scene "
            f"camera.json for {scene_name!r} missing frame {e}"
        )

    quats, transes, scales = gt_object_poses_from_actionbench_surfaces(
        per_frame_points, c2w_per_frame, device, local_rotation=local_rotation,
    )
    new_tokens = []
    for i, (fk, di) in enumerate(tokens_list):
        di = dict(di)
        di["rotation"] = quats[i].unsqueeze(0)
        di["translation"] = transes[i].unsqueeze(0)
        di["scale"] = scales[i].unsqueeze(0)
        new_tokens.append((fk, di))
    state.tokens_by_object[obj_idx] = new_tokens
    return len(new_tokens)

