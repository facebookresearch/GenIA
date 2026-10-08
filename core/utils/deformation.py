"""Per-vertex non-rigid deformation of a canonical asset, applied at native mesh resolution.

When a state carries a deformation field (``state.canonical_mesh_verts`` / ``_faces`` /
``_per_frame_verts`` / ``_per_frame_rotations``), every consumer that places the canonical
asset at a timestamp warps it through these helpers BEFORE the Sim(3) pose: FINAL's exports,
renders and tracks, evaluation, refinement and our own guidance. Identity at the canonical
frame, so the warp is a bit-equivalent no-op there.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


def _warp_at_high_res(
    p_canon: torch.Tensor,
    verts_canon: torch.Tensor,
    verts_frame_i: torch.Tensor,
    R_vert_i: torch.Tensor,
    *,
    K: int = 4,
    eps: float = 1.0e-8,
    chunk_size: int = 131072,
    compute_rotation_blend: bool = False,
    faces: "Optional[torch.Tensor]" = None,
) -> "Tuple[torch.Tensor, Optional[torch.Tensor]]":
    """High-resolution rigid-LBS warp at arbitrary primitive positions.

    Evaluates the per-canonical-mesh-vertex deformation field
    ``(verts_canon → verts_frame_i, R_vert_i)`` at arbitrary points
    ``p_canon`` (decoded Gaussian / FlexiCubes vertex positions) via
    KNN + inverse-distance weighting (translation) and optional SO(3)
    blending (rotation, for the Gaussian-quat update).

    Math (per primitive, K nearest mesh verts indexed ``k``)::

        d²_k = ‖p − verts_canon[k]‖²;  w_k = (1/(d²_k + ε)) / Σ
        disp_k(p) = R_v[k] @ (p − verts_canon[k]) + verts_frame_i[k] − p
        p_warp    = p + Σ_k w_k · disp_k(p)

    Identity invariance: when ``verts_frame_i == verts_canon`` and
    ``R_vert_i == I``, ``disp_k(p) ≡ 0`` for every k, so ``p_warp == p``
    regardless of K, ε, weights.  Uniform translation passes through
    exactly (Σ w_k = 1).

    Differentiability: KNN selection runs under ``torch.no_grad()`` on
    detached inputs (selection is piecewise-constant in p; differentiating
    the discrete topk argmax is meaningless).  IDW weights and the
    rigid-LBS displacement are recomputed WITH grad on ``p_canon`` only —
    GT inputs are detached defensively.

    Memory: kNN selection uses ``pytorch3d.ops.knn_points`` (CUDA-tiled,
    no full P×V matrix), so peak memory is independent of V.  We still
    chunk over P with ``chunk_size`` rows at a time to bound the IDW +
    rigid-LBS gather (B × K × 3 × 3) on extreme P.  Default 131072 keeps
    typical Stage-2 calls (P ≈ 100–200k) at 1–2 chunks — small chunks
    are slower because pytorch3d's per-call launch overhead dominates.

    Parameters
    ----------
    p_canon : torch.Tensor ``(P, 3)``
        Decoded primitive positions in canonical-norm.  May require grad.
    verts_canon : torch.Tensor ``(V, 3)``
        Canonical mesh vertex positions in canonical-norm.  Detached
        internally.
    verts_frame_i : torch.Tensor ``(V, 3)``
        Frame-i deformed mesh verts in frame-i self-norm.  Detached.
    R_vert_i : torch.Tensor ``(V, 3, 3)``
        Per-vertex SO(3) rigid rotation from kNN-Kabsch.  Detached.
    K : int = 4
        Number of nearest mesh vertices used for IDW (clamped to V if
        smaller).
    eps : float = 1e-8
        Stabilises the IDW denominator when a primitive coincides with a
        mesh vertex (d² = 0).
    chunk_size : int = 131072
        Number of primitives processed per chunk.  Bounds the IDW +
        rigid-LBS gather memory; smaller values pay per-chunk launch
        overhead (pytorch3d kNN handles its own memory internally).
    compute_rotation_blend : bool = False
        When True, also returns a per-primitive blended rotation
        ``R_p_blend (P, 3, 3)`` for the Gaussian-quat update.  Skipped
        for the mesh branch (vertex-color rendering is rotation-invariant).
    faces : ``(F, 3)`` int64 or None
        When provided, switches the snap step to **closest-face snap**:
        each query is snapped to the canonical face whose centroid is
        Euclidean-nearest, and the IDW support set is fixed to that face's
        own 3 vertices (so ``K`` is effectively 3).  Designed for
        multi-component meshes where the K-nearest verts span across
        disconnected sub-parts — the face-snap forces the neighbourhood
        to stay on the sub-part the query geometrically belongs to.
        For a query lying exactly on a face, the IDW weights collapse to
        barycentric weights and the warp is exact.  When ``None`` (or empty),
        the snap stays on the Euclidean-kNN vertex path.

    Returns
    -------
    p_warp : torch.Tensor ``(P, 3)``
        Warped primitive positions.  Differentiable through ``p_canon``.
    R_p_blend : torch.Tensor ``(P, 3, 3)`` or None
        Per-primitive blended rotation.  ``None`` when
        ``compute_rotation_blend=False``.
    """
    if p_canon.dim() != 2 or p_canon.shape[1] != 3:
        raise ValueError(
            f"p_canon must be (P, 3); got {tuple(p_canon.shape)}"
        )
    if verts_canon.dim() != 2 or verts_canon.shape[1] != 3:
        raise ValueError(
            f"verts_canon must be (V, 3); got {tuple(verts_canon.shape)}"
        )
    V = verts_canon.shape[0]
    if V == 0:
        raise ValueError("verts_canon must have V > 0")
    if verts_frame_i.shape != (V, 3):
        raise ValueError(
            f"verts_frame_i shape {tuple(verts_frame_i.shape)} != ({V}, 3)"
        )
    if R_vert_i.shape != (V, 3, 3):
        raise ValueError(
            f"R_vert_i shape {tuple(R_vert_i.shape)} != ({V}, 3, 3)"
        )
    # Defensive detach — caller can't accidentally leak grad through GT.
    verts_canon_d = verts_canon.detach()
    verts_frame_d = verts_frame_i.detach()
    R_vert_d = R_vert_i.detach()

    K_eff = min(int(K), V)
    if K_eff < 1:
        raise ValueError(f"K must be >= 1; got {K}")

    # Closest-face snap when faces are supplied (its whole purpose is to keep
    # the IDW support on one connected sub-part of multi-component meshes).
    # Empty (0, 3) faces mean no mesh connectivity → fall through to the vertex path.
    use_face_snap = faces is not None and faces.shape[0] > 0
    face_centroids_d: "Optional[torch.Tensor]" = None
    faces_d: "Optional[torch.Tensor]" = None
    if use_face_snap:
        if faces.dim() != 2 or faces.shape[1] != 3:
            raise ValueError(
                f"faces must be (F, 3); got {tuple(faces.shape)}"
            )
        faces_d = faces.detach().to(device=verts_canon.device).long()
        face_centroids_d = verts_canon_d[faces_d].mean(dim=1)         # (F, 3)
        K_eff = 3                                                      # face has 3 verts

    # pytorch3d's optimised kNN (CUDA kernel + tiling) — ~12× faster than
    # brute-force ``cdist+topk`` on P≈100k, V≈19k (146 ms → 12 ms) and
    # avoids materialising the full B × V distance matrix, so peak memory
    # is independent of V.  Imported here (not at module top) to follow
    # the file's pattern of lazy pytorch3d imports.
    from pytorch3d.ops import knn_points

    P = p_canon.shape[0]
    chunk = max(1, int(chunk_size))

    p_warp_parts: "list[torch.Tensor]" = []
    R_blend_parts: "list[torch.Tensor]" = []

    for start in range(0, P, chunk):
        end = min(start + chunk, P)
        p_chunk = p_canon[start:end]                                  # (B, 3)
        B = p_chunk.shape[0]

        # 1. No-grad KNN selection on detached inputs.  Selection is
        #    discrete and non-differentiable; the gradient path comes
        #    later via the IDW weights and rigid-LBS displacement.
        with torch.no_grad():
            if use_face_snap:
                # Snap query → nearest face centroid; IDW support set is
                # that face's own 3 verts.  Multi-component-safe: the face
                # carries a single connected sub-part by construction, so
                # the K=3 neighbours never straddle a shell boundary.
                _, nf_b, _ = knn_points(
                    p_chunk.detach()[None], face_centroids_d[None],
                    K=1, return_sorted=False,
                )
                idx_k = faces_d[nf_b[0, :, 0]]                        # (B, 3)
            else:
                _, idx_k_batched, _ = knn_points(
                    p_chunk.detach()[None],     # (1, B, 3)
                    verts_canon_d[None],        # (1, V, 3)
                    K=K_eff,
                    return_sorted=False,
                )
                idx_k = idx_k_batched[0]                              # (B, K)

        # 2. Gather neighbour data.
        canon_k = verts_canon_d[idx_k]                                # (B, K, 3)
        frame_k = verts_frame_d[idx_k]                                # (B, K, 3)
        R_k = R_vert_d[idx_k]                                         # (B, K, 3, 3)

        # 3. Grad-flowing displacement & weights.
        delta = p_chunk[:, None, :] - canon_k                          # (B, K, 3)
        d2_grad = delta.pow(2).sum(-1)                                 # (B, K)
        w = 1.0 / (d2_grad + eps)
        w = w / w.sum(-1, keepdim=True)                                # (B, K)

        # 4. Rigid-LBS displacement per neighbour (column-vector convention):
        #        disp_k = R_v_k @ (p − vert_canon[k]) + verts_frame_i[k] − p.
        rotated_delta = torch.einsum("bkij,bkj->bki", R_k, delta)      # (B, K, 3)
        disp_per_k = rotated_delta + frame_k - p_chunk[:, None, :]     # (B, K, 3)

        # 5. IDW blend.
        disp_blended = (w.unsqueeze(-1) * disp_per_k).sum(dim=1)       # (B, 3)
        p_warp_parts.append(p_chunk + disp_blended)

        # 6. Optional SO(3) blend for the Gaussian-quat update: quaternion
        # linear blend + renormalise (nlerp).
        if compute_rotation_blend:
            from pytorch3d.transforms import (
                matrix_to_quaternion, quaternion_to_matrix,
            )
            q_k = matrix_to_quaternion(R_k.reshape(-1, 3, 3)).reshape(
                B, K_eff, 4,
            )                                                       # wxyz
            # Sign-align each neighbour's quat to the first neighbour's
            # so the linear blend doesn't average q vs −q (same R).
            signs = torch.sign(
                (q_k * q_k[:, :1]).sum(-1, keepdim=True)
            )
            signs = torch.where(
                signs == 0, torch.ones_like(signs), signs,
            )
            q_aligned = signs * q_k                                 # (B, K, 4)
            q_avg = (w.unsqueeze(-1) * q_aligned).sum(dim=1)        # (B, 4)
            q_avg = q_avg / q_avg.norm(
                dim=-1, keepdim=True,
            ).clamp_min(1.0e-12)
            R_p = quaternion_to_matrix(q_avg)                        # (B, 3, 3)
            R_blend_parts.append(R_p)

    p_warp = torch.cat(p_warp_parts, dim=0)
    R_p_blend = torch.cat(R_blend_parts, dim=0) if R_blend_parts else None
    return p_warp, R_p_blend


def _lookup_per_frame_deformation(
    canonical_mesh_verts: "dict | None",
    per_frame_mesh_verts: "dict | None",
    per_frame_mesh_rotations: "dict | None",
    obj_idx: int,
    frame_int: int,
    device: torch.device,
    canonical_mesh_faces: "dict | None" = None,
) -> "Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]] | None":
    """Resolve ``(canonical_verts, frame_verts, frame_rotations, faces)``
    tensors on ``device`` for one ``(obj_idx, frame_int)``, or ``None`` if
    any required piece is missing.  ``faces`` is ``None`` when no
    ``canonical_mesh_faces`` dict was passed or the obj has no faces — the
    warp then stays on the Euclidean vertex-snap path; when faces are
    provided, the warp switches to closest-face snap (component-safe on
    multi-shell meshes).  All callers splat the tuple straight into a
    ``warp_*`` helper.

    Single source of truth for the keyframes renderers' "is per-vertex
    deformation available for this object at this frame?" check.  Returns
    ``None`` (no warp) when:
      * ``canonical_mesh_verts is None`` (no actionmesh deformation loaded);
      * ``obj_idx`` not in ``canonical_mesh_verts``;
      * ``per_frame_mesh_verts`` / ``per_frame_mesh_rotations`` missing the
        ``obj_idx`` outer key or the ``frame_int`` inner key.

    The dicts are the state-shaped fields populated by ``GT_SHAPES_INVERSION``:
      ``state.canonical_mesh_verts``, ``state.canonical_mesh_per_frame_verts``,
      ``state.canonical_mesh_per_frame_rotations``, ``state.canonical_mesh_faces``.
    """
    if canonical_mesh_verts is None or obj_idx not in canonical_mesh_verts:
        return None
    pf_verts = (per_frame_mesh_verts or {}).get(obj_idx, {}).get(frame_int)
    pf_rotations = (per_frame_mesh_rotations or {}).get(obj_idx, {}).get(frame_int)
    if pf_verts is None or pf_rotations is None:
        return None
    faces = (canonical_mesh_faces or {}).get(obj_idx)
    return (
        canonical_mesh_verts[obj_idx].to(device),
        pf_verts.to(device),
        pf_rotations.to(device),
        faces.to(device) if faces is not None else None,
    )


def warp_voxel_coords_high_res(
    voxel_coords: "np.ndarray",
    canonical_mesh_verts: torch.Tensor,
    per_frame_mesh_verts_i: torch.Tensor,
    per_frame_mesh_rotations_i: torch.Tensor,
    *,
    grid_size: int = 64,
    K: int = 4,
    eps: float = 1.0e-8,
    chunk_size: int = 8192,
    faces: "Optional[torch.Tensor]" = None,
) -> "np.ndarray":
    """High-resolution per-vertex warp of canonical voxel-grid coords.

    Parallel to :func:`warp_gaussians_high_res` for the voxel-mesh
    rendering path.  Maps integer voxel indices ``∈ [0, grid_size)``
    through canonical-norm cell-centers, warps via :func:`_warp_at_high_res`
    (translation only), and maps back to continuous coord-space floats
    suitable for the existing voxel-mesh builder.

    Identity invariance: when ``per_frame_mesh_verts_i == canonical_mesh_verts``
    and ``per_frame_mesh_rotations_i == I``, the round-trip is a no-op
    (warp is identity by construction; coord ↔ canonical-norm is exact
    inverse arithmetic).

    Parameters
    ----------
    voxel_coords : np.ndarray ``(N, 3)``
        Integer (or float) voxel indices in ``[0, grid_size)``.
    canonical_mesh_verts, per_frame_mesh_verts_i, per_frame_mesh_rotations_i
        Per-vertex deformation field for one frame; same convention as
        :func:`warp_gaussians_high_res`.  Caller is responsible for moving
        these to the desired device (the helper picks compute device from
        ``canonical_mesh_verts.device``).
    grid_size : int = 64
    K, eps, chunk_size, faces
        Forwarded to :func:`_warp_at_high_res`.  When ``faces`` is provided
        the snap switches to closest-face-centroid (component-safe on
        multi-shell meshes); otherwise the vertex-snap path is used.

    Returns
    -------
    np.ndarray ``(N, 3)`` float32
        Warped continuous voxel-grid coords, same coordinate space as the
        input (downstream consumers accept floats).
    """
    device = canonical_mesh_verts.device
    G = int(grid_size)
    # Voxel index → canonical-norm cell-center: c_norm = (i + 0.5)/G − 0.5.
    canon_centers = torch.from_numpy(
        (voxel_coords.astype(np.float64) + 0.5) / G - 0.5,
    ).float().to(device)
    with torch.no_grad():
        warped_centers, _ = _warp_at_high_res(
            canon_centers,
            canonical_mesh_verts,
            per_frame_mesh_verts_i,
            per_frame_mesh_rotations_i,
            K=int(K), eps=float(eps), chunk_size=int(chunk_size),
            compute_rotation_blend=False,
            faces=faces,
        )
    # Inverse map: c_coord = (c_norm + 0.5)·G − 0.5.
    return ((warped_centers.cpu().numpy() + 0.5) * G - 0.5).astype(np.float32)


def warp_gaussians_high_res(
    gs_obj,
    canonical_mesh_verts: torch.Tensor,
    per_frame_mesh_verts_i: torch.Tensor,
    per_frame_mesh_R_vert_i: torch.Tensor,
    *,
    K: int = 4,
    eps: float = 1.0e-8,
    chunk_size: int = 8192,
    faces: "Optional[torch.Tensor]" = None,
) -> "Tuple[torch.Tensor, torch.Tensor]":
    """High-resolution rigid-LBS warp of decoded Gaussians for one frame.

    Wraps :func:`_warp_at_high_res` (translation) plus the SO(3)-blended
    Gaussian-quat update (left-compose, normalise) — shared by the
    rendering-guidance builder, the ODE-steps debug viz in
    ``core/visualization.py`` and the FINAL exports, so every call site
    applies the same warp.

    Parameters
    ----------
    gs_obj
        Decoded canonical Gaussian — must expose ``get_xyz`` ``(P, 3)`` and
        ``get_rotation`` ``(P, 4)`` wxyz.
    canonical_mesh_verts : ``(V, 3)``
        Canonical mesh vertex positions in canonical-norm.
    per_frame_mesh_verts_i : ``(V, 3)``
        Frame-i deformed mesh vertices in frame-i self-norm (single frame).
    per_frame_mesh_R_vert_i : ``(V, 3, 3)``
        Per-vertex SO(3) rotations from kNN-Kabsch (single frame).
    K, eps, chunk_size, faces
        Forwarded to :func:`_warp_at_high_res`.  When ``faces`` is provided
        the snap switches to closest-face-centroid (component-safe on
        multi-shell meshes); otherwise the vertex-snap path is used.

    Returns
    -------
    means_warped : ``(P, 3)``
        Differentiable warped Gaussian means (grad flows back to
        ``gs_obj.get_xyz``).
    quats_warped : ``(P, 4)`` wxyz, normalised
        Composed quaternion: ``quat(R_p_blend) ⊗ gs_obj.get_rotation``.
        Differentiable through both inputs.
    """
    from pytorch3d.transforms import matrix_to_quaternion, quaternion_multiply

    means_w, R_p_blend = _warp_at_high_res(
        gs_obj.get_xyz,
        canonical_mesh_verts,
        per_frame_mesh_verts_i,
        per_frame_mesh_R_vert_i,
        K=int(K),
        eps=float(eps),
        chunk_size=int(chunk_size),
        compute_rotation_blend=True,
        faces=faces,
    )
    q_R_blend = matrix_to_quaternion(R_p_blend)              # (P, 4) wxyz
    quats_w = quaternion_multiply(q_R_blend, gs_obj.get_rotation)
    quats_w = quats_w / quats_w.norm(dim=-1, keepdim=True).clamp_min(1.0e-10)
    return means_w, quats_w
