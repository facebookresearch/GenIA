# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Per-voxel visibility for our Stage-2 conditioning.

Self-occlusion by DDA ray tracing, plus cross-object occlusion by rasterization.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from genia.core.utils.visibility import _compute_self_occlusion_jit


def compute_self_occlusion(
    voxel_coords: np.ndarray,
    camera_pos_voxel: np.ndarray,
    grid_size: int = 64,
    neighbor_tolerance: float = 4.0,
) -> np.ndarray:
    """Detect self-occlusion via DDA ray tracing for a single camera.

    Parameters
    ----------
    voxel_coords : (N, 3) int
        Integer voxel coordinates of occupied latent points.
    camera_pos_voxel : (3,) float
        Camera position in voxel space.
    grid_size : int
        Voxel grid resolution.
    neighbor_tolerance : float
        Ignore occluding voxels within this distance (in voxel units)
        of the target.  Handles grazing-angle false positives.

    Returns
    -------
    visibility : (N,) float32
        1.0 = visible, 0.0 = self-occluded.
    """
    coords_int = np.ascontiguousarray(voxel_coords.astype(np.int64))
    cam = np.ascontiguousarray(camera_pos_voxel.astype(np.float64))

    # Build occupancy grid (vectorized)
    occupancy = np.zeros((grid_size, grid_size, grid_size), dtype=np.bool_)
    valid = np.all((coords_int >= 0) & (coords_int < grid_size), axis=1)
    vc = coords_int[valid]
    occupancy[vc[:, 0], vc[:, 1], vc[:, 2]] = True

    tolerance_sq = neighbor_tolerance ** 2

    return _compute_self_occlusion_jit(
        coords_int, cam, occupancy, grid_size, tolerance_sq,
    )


def _camera_to_voxel(
    rotation: np.ndarray,
    translation: np.ndarray,
    scale: np.ndarray,
    grid_size: int = 64,
) -> np.ndarray:
    """Transform camera origin [0,0,0] from camera space to voxel space.

    Inverse of ``apply_pose_to_gaussian``:
        transformed = xyz_local * scale @ R + translation
    So:
        xyz_local = (cam_camera - translation) @ R^T / scale
    Camera is at [0,0,0]:
        cam_canonical = -translation @ R^T / scale
    Canonical → voxel:
        cam_voxel = (cam_canonical + 0.5) * grid_size

    Parameters
    ----------
    rotation : (4,) or (1, 4)
        Quaternion (wxyz) — PyTorch3D convention.
    translation : (3,) or (1, 3)
        Translation vector.
    scale : (3,) or (1, 3) or (1,) or scalar
        Scale factor(s).
    grid_size : int
        Voxel grid resolution (default 64).

    Returns
    -------
    cam_voxel : (3,) float
        Camera position in voxel space.
    """
    from pytorch3d.transforms import quaternion_to_matrix

    import torch

    # Ensure flat numpy
    rot = np.atleast_1d(rotation).flatten()
    trans = np.atleast_1d(translation).flatten()[:3]
    sc = np.atleast_1d(scale).flatten()
    if len(sc) == 1:
        sc = np.repeat(sc, 3)

    # Quaternion → rotation matrix (PyTorch3D: p_new = p @ R)
    R = quaternion_to_matrix(
        torch.tensor(rot, dtype=torch.float32).unsqueeze(0)
    ).squeeze(0).numpy()  # (3, 3)

    # Inverse pose: cam_canonical = -trans @ R^T / scale
    cam_canonical = (-trans @ R.T) / sc

    # Canonical [-0.5, 0.5] → voxel [0, grid_size)
    cam_voxel = (cam_canonical + 0.5) * grid_size

    return cam_voxel


def compute_visibility_for_all_views(
    voxel_coords: np.ndarray,
    rotations: list,
    translations: list,
    scales: list,
    grid_size: int = 64,
    neighbor_tolerance: float = 4.0,
) -> np.ndarray:
    """Compute per-voxel per-view visibility via DDA ray tracing.

    Parameters
    ----------
    voxel_coords : (N, 3) int
        Integer voxel coordinates of occupied latent points in the 64^3 grid.
    rotations : list of array-like
        Per-view quaternions (wxyz), each shape ``(4,)`` or ``(1, 4)``.
    translations : list of array-like
        Per-view translations, each shape ``(3,)`` or ``(1, 3)``.
    scales : list of array-like
        Per-view scales, each shape ``(3,)`` or ``(1, 3)`` or ``(1,)`` or scalar.
    grid_size : int
        Voxel grid resolution (default 64).
    neighbor_tolerance : float
        DDA neighbor tolerance in voxel units.

    Returns
    -------
    visibility_matrix : (N_views, N_voxels) float32
        1.0 = visible, 0.0 = self-occluded.
    """
    N_views = len(rotations)
    N_voxels = len(voxel_coords)

    visibility_matrix = np.zeros((N_views, N_voxels), dtype=np.float32)

    for view_idx in range(N_views):
        cam_voxel = _camera_to_voxel(
            rotations[view_idx], translations[view_idx], scales[view_idx],
            grid_size=grid_size,
        )
        print(f"    View {view_idx}: camera voxel pos = "
              f"[{cam_voxel[0]:.1f}, {cam_voxel[1]:.1f}, {cam_voxel[2]:.1f}]")

        vis = compute_self_occlusion(
            voxel_coords, cam_voxel,
            grid_size=grid_size,
            neighbor_tolerance=neighbor_tolerance,
        )
        visibility_matrix[view_idx] = vis

        n_visible = int(vis.sum())
        print(f"      {n_visible}/{N_voxels} visible "
              f"({100.0 * n_visible / N_voxels:.1f}%)")

    return visibility_matrix


#: PyTorch3D's coarse rasterizer refuses more bins than this along an image side.
_MAX_BINS_PER_SIDE = 22


def _coarse_raster_budget(verts: np.ndarray, faces: np.ndarray, K: np.ndarray,
                          H: int, W: int, bin_size: int = 64) -> Tuple[int, int]:
    """``(bin_size, max_faces_per_bin)`` for PyTorch3D's coarse-to-fine rasterizer that no
    bin can overflow.  An overflow drops faces SILENTLY (the kernel only prints), so the
    budget is an upper bound on every bin's face count rather than a guess: each face's
    pixel bounding box (R3 camera ``verts``, intrinsics ``K``) widened by one bin on every
    side, plus the principal point's offset from the centre (the camera is built with negated
    focals, so the kernel's grid may be this one mirrored about ``2 c - size``), faces reaching
    ``z <= 0`` counted in every bin, and the maximum taken over the four mirrorings of the
    grid (whatever pixel orientation the kernel bins in).  The budget sets only how much the
    kernel allocates and scans per bin, so the fragments are those of an unbounded budget, at a
    fraction of its cost.  ``bin_size`` grows
    past 64 only to respect :data:`_MAX_BINS_PER_SIDE`."""
    while -(-max(H, W) // bin_size) >= _MAX_BINS_PER_SIDE:
        bin_size *= 2
    nby, nbx = -(-H // bin_size), -(-W // bin_size)
    mgx = 1 + int(np.ceil(abs(2 * K[0, 2] - W) / bin_size))
    mgy = 1 + int(np.ceil(abs(2 * K[1, 2] - H) / bin_size))
    z = verts[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * verts[:, 0] / z + K[0, 2]
        v = K[1, 1] * verts[:, 1] / z + K[1, 2]
    fu, fv, fz = u[faces], v[faces], z[faces]                 # (F, 3) each
    behind = (fz <= 1e-6).any(1)
    best = 1
    for mx in (False, True):
        for my in (False, True):
            uu = (W - fu) if mx else fu
            vv = (H - fv) if my else fv
            x0 = np.where(behind, 0, np.floor(uu.min(1) / bin_size) - mgx)
            x1 = np.where(behind, nbx - 1, np.floor(uu.max(1) / bin_size) + mgx)
            y0 = np.where(behind, 0, np.floor(vv.min(1) / bin_size) - mgy)
            y1 = np.where(behind, nby - 1, np.floor(vv.max(1) / bin_size) + mgy)
            keep = (x1 >= 0) & (x0 <= nbx - 1) & (y1 >= 0) & (y0 <= nby - 1)
            x0, x1 = (np.clip(a[keep], 0, nbx - 1).astype(int) for a in (x0, x1))
            y0, y1 = (np.clip(a[keep], 0, nby - 1).astype(int) for a in (y0, y1))
            diff = np.zeros((nby + 1, nbx + 1), np.int64)     # 2D difference array of the boxes
            np.add.at(diff, (y0, x0), 1)
            np.add.at(diff, (y0, x1 + 1), -1)
            np.add.at(diff, (y1 + 1, x0), -1)
            np.add.at(diff, (y1 + 1, x1 + 1), 1)
            best = max(best, int(diff.cumsum(0).cumsum(1)[:nby, :nbx].max(initial=0)))
    return bin_size, best


def _rasterize_scene(
    objects_data: dict,
    R_cache: dict,
    trans_cache: dict,
    scale_cache: dict,
    view_idx: int,
    K: np.ndarray,
    H: int,
    W: int,
    grid_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize all objects' voxel cubes and return per-pixel information.

    Builds surface meshes for all objects (with face→voxel tracking),
    applies per-view poses, and rasterizes in a single pass with
    PyTorch3D ``MeshRasterizer``.

    Returns
    -------
    zbuf : (H, W) float32
        Closest surface depth at each pixel (camera-space z).
        ``-1`` where no surface is visible (background).
    obj_map : (H, W) int64
        Object index at each pixel (``-1`` = background).
    voxel_map : (H, W) int64
        Voxel index (within its object) at each pixel (``-1`` = background).
    """
    import torch
    from pytorch3d.renderer import (
        MeshRasterizer, PerspectiveCameras, RasterizationSettings,
        TexturesVertex,
    )
    from pytorch3d.structures import Meshes
    from genia.core.utils.visualization import _build_voxel_surface_mesh

    obj_indices = sorted(objects_data.keys())
    _empty = (
        np.full((H, W), -1.0, dtype=np.float32),
        np.full((H, W), -1, dtype=np.int64),
        np.full((H, W), -1, dtype=np.int64),
    )

    all_verts_list: list = []
    all_faces_list: list = []
    all_colors_list: list = []
    # Global face → (obj_idx, voxel_idx) mapping
    all_face_obj: list = []
    all_face_vox: list = []
    vert_offset = 0

    for oi in obj_indices:
        coords = objects_data[oi]["voxel_coords"]
        R = R_cache[(oi, view_idx)]
        trans = trans_cache[(oi, view_idx)]
        sc = scale_cache[(oi, view_idx)]

        local_verts, faces, vert_colors, f2v = _build_voxel_surface_mesh(
            coords.astype(np.float32), grid_size,
            return_face_to_voxel=True,
        )
        if len(local_verts) == 0:
            continue

        # Pose: posed = (local * scale) @ R + trans, then P3D → R3
        posed = (local_verts * sc) @ R + trans
        posed[..., :2] *= -1

        all_verts_list.append(posed)
        all_faces_list.append(faces + vert_offset)
        all_colors_list.append(vert_colors)
        all_face_obj.append(np.full(len(faces), oi, dtype=np.int64))
        all_face_vox.append(f2v.astype(np.int64))
        vert_offset += len(local_verts)

    if not all_verts_list:
        return _empty

    all_verts = np.concatenate(all_verts_list, axis=0)
    all_faces = np.concatenate(all_faces_list, axis=0)
    all_colors = np.concatenate(all_colors_list, axis=0)
    face_obj = np.concatenate(all_face_obj, axis=0)
    face_vox = np.concatenate(all_face_vox, axis=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    mesh = Meshes(
        verts=[torch.from_numpy(all_verts).float().to(device)],
        faces=[torch.from_numpy(all_faces).long().to(device)],
        textures=TexturesVertex(
            verts_features=[torch.from_numpy(all_colors).float().to(device)],
        ),
    )
    cameras = PerspectiveCameras(
        focal_length=torch.tensor([[-fx, -fy]]).float(),
        principal_point=torch.tensor([[cx, cy]]).float(),
        R=torch.eye(3).unsqueeze(0).float(),
        T=torch.zeros(1, 3).float(),
        image_size=torch.tensor([[H, W]]),
        in_ndc=False,
        device=device,
    )
    # Coarse-to-fine (binned) rasterization with a budget no bin can overflow (an overflow
    # silently drops faces): the same fragments as the naive `bin_size=0` kernel, faster.
    bin_size, max_faces_per_bin = _coarse_raster_budget(all_verts, all_faces, K, H, W)
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=RasterizationSettings(
            image_size=(H, W), blur_radius=0.0,
            faces_per_pixel=1, bin_size=bin_size, max_faces_per_bin=max_faces_per_bin,
        ),
    )

    with torch.no_grad():
        fragments = rasterizer(mesh)

    zbuf = fragments.zbuf[0, :, :, 0].cpu().numpy()
    pix_to_face = fragments.pix_to_face[0, :, :, 0].cpu().numpy()

    obj_map = np.full((H, W), -1, dtype=np.int64)
    voxel_map = np.full((H, W), -1, dtype=np.int64)
    valid = pix_to_face >= 0
    obj_map[valid] = face_obj[pix_to_face[valid]]
    voxel_map[valid] = face_vox[pix_to_face[valid]]

    return zbuf, obj_map, voxel_map


def compute_visibility_multi_object(
    objects_data: dict,
    K_matrices: list,
    image_height: int,
    image_width: int,
    grid_size: int = 64,
    neighbor_tolerance: float = 4.0,
) -> tuple:
    """Hybrid DDA + z-buffer visibility for multiple objects.

    Self-occlusion is computed per-object via DDA ray tracing through the
    object's own 64³ occupancy grid (with ``neighbor_tolerance`` to avoid
    grazing-angle artifacts).  Cross-object occlusion is determined by a
    screen-space z-buffer: all objects' voxels are projected and the
    closest object at each pixel wins.

    A voxel is visible iff:
      1. DDA says it is not self-occluded within its own object,
      2. it projects within the image and in front of the camera,
      3. no other object's rasterized surface is significantly closer at
         that pixel (depth-buffer test with ``neighbor_tolerance``-based
         tolerance).

    Parameters
    ----------
    objects_data : dict[int, dict]
        ``{obj_idx: {"voxel_coords": np.ndarray (L, 3) int, "rotations": list,
        "translations": list, "scales": list}}`` — a single canonical voxel
        coord set per object, shared across every view.
    K_matrices : list of (3, 3) ndarray
        Per-view camera intrinsics.
    image_height, image_width : int
        Image dimensions for the z-buffer.
    grid_size : int
        Voxel grid resolution (default 64).
    neighbor_tolerance : float
        DDA neighbor tolerance in voxel units for self-occlusion.

    Returns
    -------
    visibility : dict[int, np.ndarray]
        ``{obj_idx: (N_views, N_voxels) float32}`` with 1.0 = visible.
    pixel_coords : dict[int, list[dict[int, np.ndarray]]]
        ``{obj_idx: [view_0_map, view_1_map, ...]}``.
        Each ``view_map`` is ``{voxel_idx: (M, 2) int}`` — the set of
        ``(row, col)`` pixel coordinates where that voxel is visible
        (rasterized surface pixels whose face belongs to this voxel), or
        at least its projected centre pixel when rasterization misses it.
    """
    from pytorch3d.transforms import quaternion_to_matrix

    import torch

    obj_indices = sorted(objects_data.keys())
    if not obj_indices:
        return {}, {}

    N_views = len(objects_data[obj_indices[0]]["rotations"])

    # Pre-compute rotation matrices + parsed pose vectors
    R_cache: dict = {}
    trans_cache: dict = {}
    scale_cache: dict = {}

    for oi in obj_indices:
        d = objects_data[oi]
        for vi in range(N_views):
            rot = np.atleast_1d(d["rotations"][vi]).flatten()
            R = quaternion_to_matrix(
                torch.tensor(rot, dtype=torch.float32).unsqueeze(0)
            ).squeeze(0).numpy()
            R_cache[(oi, vi)] = R

            trans = np.atleast_1d(d["translations"][vi]).flatten()[:3]
            trans_cache[(oi, vi)] = trans

            sc = np.atleast_1d(d["scales"][vi]).flatten()
            if len(sc) == 1:
                sc = np.repeat(sc, 3)
            scale_cache[(oi, vi)] = sc

    tolerance_sq = neighbor_tolerance ** 2
    voxel_counts = {oi: len(objects_data[oi]["voxel_coords"]) for oi in obj_indices}

    result: dict = {}
    pixel_coords: dict = {}
    for oi in obj_indices:
        result[oi] = np.zeros((N_views, voxel_counts[oi]), dtype=np.float32)
        pixel_coords[oi] = [{} for _ in range(N_views)]

    H, W = image_height, image_width

    for view_idx in range(N_views):
        K = np.asarray(K_matrices[view_idx], dtype=np.float64)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # ── Per-object DDA self-occlusion ────────────────────────────
        dda_vis: dict = {}
        proj_u: dict = {}
        proj_v: dict = {}
        proj_z: dict = {}

        for oi in obj_indices:
            coords = objects_data[oi]["voxel_coords"]
            coords_int = np.ascontiguousarray(coords.astype(np.int64))
            R = R_cache[(oi, view_idx)]
            trans = trans_cache[(oi, view_idx)]
            sc = scale_cache[(oi, view_idx)]

            # DDA self-occlusion in object's own voxel grid
            cam_canonical = (-trans @ R.T) / sc
            cam_voxel = (cam_canonical + 0.5) * grid_size

            occupancy = np.zeros(
                (grid_size, grid_size, grid_size), dtype=np.bool_,
            )
            valid = np.all(
                (coords_int >= 0) & (coords_int < grid_size), axis=1,
            )
            vc = coords_int[valid]
            occupancy[vc[:, 0], vc[:, 1], vc[:, 2]] = True

            cam_arr = np.ascontiguousarray(
                cam_voxel.astype(np.float64),
            )
            dda_vis[oi] = _compute_self_occlusion_jit(
                coords_int, cam_arr, occupancy, grid_size, tolerance_sq,
            )

            # Screen projection (centre pixel, depth-test lookups).
            canonical = (coords.astype(np.float64) + 0.5) / grid_size - 0.5
            cam = canonical * sc @ R + trans
            z = cam[:, 2].copy()
            safe_z = np.maximum(z, 1e-8)
            proj_u[oi] = np.round(
                fx * (-cam[:, 0]) / safe_z + cx,
            ).astype(np.int64)
            proj_v[oi] = np.round(
                fy * (-cam[:, 1]) / safe_z + cy,
            ).astype(np.int64)
            proj_z[oi] = z

        # ── Cross-object occlusion ──────────────────────────────────
        # Rasterize all objects' voxel meshes → scene depth + maps (a single object
        # too, so it behaves exactly as several: footprint pixels, depth-buffer test)
        zbuf, obj_map, voxel_map = _rasterize_scene(
            objects_data, R_cache, trans_cache, scale_cache,
            view_idx, K, H, W, grid_size,
        )

        # Min-filter the depth buffer to fill sub-pixel gaps
        from scipy.ndimage import minimum_filter
        zbuf_valid = np.where(zbuf > 0, zbuf, np.inf)
        zbuf_min = minimum_filter(zbuf_valid, size=3)

        # Combine: DDA visible AND in-bounds AND not cross-occluded
        for oi in obj_indices:
            u_arr = proj_u[oi]
            v_arr = proj_v[oi]
            z_arr = proj_z[oi]
            sc = scale_cache[(oi, view_idx)]

            in_bounds = (
                (u_arr >= 0) & (u_arr < W)
                & (v_arr >= 0) & (v_arr < H)
                & (z_arr > 0)
            )
            idx = np.where(in_bounds & (dda_vis[oi] > 0.5))[0]

            if len(idx) > 0:
                depth_tol = (
                    np.max(sc) / grid_size * neighbor_tolerance
                )
                nearest_z = zbuf_min[v_arr[idx], u_arr[idx]]
                not_occluded = (
                    (nearest_z == np.inf)
                    | (z_arr[idx] <= nearest_z + depth_tol)
                )
                vis_voxels = np.unique(idx[not_occluded])
                result[oi][view_idx, vis_voxels] = 1.0

        # Build voxel → pixel mapping from rasterized obj_map/voxel_map
        for oi in obj_indices:
            rast_mask = obj_map == oi
            if not rast_mask.any():
                continue
            rows, cols = np.where(rast_mask)
            vox_ids = voxel_map[rast_mask]
            # Only include pixels whose voxel is marked visible
            vis_mask = result[oi][view_idx, vox_ids] > 0.5
            rows, cols, vox_ids = rows[vis_mask], cols[vis_mask], vox_ids[vis_mask]
            # One stable sort, then split per voxel: each voxel keeps its pixels in
            # raster order, keys ascend -- what a per-voxel `vox_ids == vid` scan
            # gives, without its (voxels x pixels) cost.
            order = np.argsort(vox_ids, kind="stable")
            vids, starts = np.unique(vox_ids[order], return_index=True)
            rc = np.stack([rows[order], cols[order]], axis=1)
            pix_map = pixel_coords[oi][view_idx]
            for vid, chunk in zip(vids.tolist(), np.split(rc, starts[1:])):
                pix_map[vid] = chunk

        # ── Pixel coords fallback ──────────────────────────────────
        # Ensure all visible voxels have at least their projected center
        # in pixel_coords: rasterization can miss voxels whose surface faces
        # are sub-pixel.
        for oi in obj_indices:
            vis_voxels = np.where(result[oi][view_idx] > 0.5)[0]
            pix_map = pixel_coords[oi][view_idx]
            missing = vis_voxels[~np.isin(vis_voxels, list(pix_map))]
            u_arr, v_arr = proj_u[oi][missing], proj_v[oi][missing]
            ok = (u_arr >= 0) & (u_arr < W) & (v_arr >= 0) & (v_arr < H)
            for vid, u, v in zip(missing[ok].tolist(), u_arr[ok].tolist(), v_arr[ok].tolist()):
                pix_map[vid] = np.array([[v, u]], dtype=np.int64)

        for oi in obj_indices:
            n_vis = int(result[oi][view_idx].sum())
            n_tot = voxel_counts[oi]
            n_self = int((dda_vis[oi] == 0).sum())
            print(
                f"    View {view_idx} obj {oi}: "
                f"{n_vis}/{n_tot} visible "
                f"({100.0 * n_vis / n_tot:.1f}%), "
                f"self-occluded: {n_self}"
            )

    return result, pixel_coords
