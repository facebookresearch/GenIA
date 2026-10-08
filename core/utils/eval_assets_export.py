"""Export held-out NVS test renders (+ per-frame-only meshes) from FINAL.

Writes to ``{run_dir}/final/``:

    renders_test/
    └── {010..024}.png         # 15 held-out NVS renders, RGBA transparent bg
                                # (RGB composited on white, coverage in alpha),
                                # at the train-view resolution (sequence.H/W --
                                # the GSO 256² protocol is NOT enforced; test
                                # views match train views in renders_train/)
    meshes/{obj:03d}/{stem}.ply  # per-frame-only runs only (no canonical
                                # object): object-local mesh
                                # decoded from each frame's SLAT. Canonical
                                # pipelines get meshes/{obj:03d}.glb from
                                # save_per_object_mesh instead.

An evaluator loads the canonical ``meshes/{obj:03d}.glb`` or, for per-frame-only runs,
the per-frame ``meshes/{obj:03d}/{stem}.ply`` matched to ``poses.json``.

Design notes
------------
* NVS renders use a hardcoded GT intrinsic derived from EscherNet's Blender
  render script (``lens=35mm``, ``sensor_width=32mm``, square pixels):
  ``fx = fy = 35 * W / 32``, ``cx = W/2``, ``cy = H/2``.
* Object is placed in the world (Blender GT) frame by composing frame 0's
  per-frame pose and its c2w. For static GSO scenes any frame is equivalent.
* Foreground-only render: background Gaussians are skipped entirely. The RGB
  channels are composited on white (GT NVS has a pure white bg, so an RGB-only
  evaluator that drops alpha sees matching white-background pixels), and the rendered coverage is saved in the alpha
  channel so the PNG background is transparent.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from genia.core.utils.frame_key import FrameKey


def gso_blender_intrinsics(H: int = 256, W: int = 256) -> np.ndarray:
    """GSO GT camera intrinsics.

    Derived from EscherNet's Blender render script (``blender_script_mvs.py``):

        cam.data.lens = 35         # mm
        cam.data.sensor_width = 32 # mm

    Blender auto-computes sensor height from the output aspect ratio. For
    square 1:1 renders with ``lens=35, sensor_width=32``::

        fx = fy = f_mm * W / sensor_width = 35 * W / 32
        cx, cy = W/2, H/2
    """
    fx = fy = 35.0 * W / 32.0
    cx, cy = W / 2.0, H / 2.0
    K = np.array(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return K


BLENDER_TO_R3_WORLD = np.array([
    [1.0,  0.0,  0.0],
    [0.0,  0.0, -1.0],
    [0.0,  1.0,  0.0],
], dtype=np.float32)
"""World-axis rotation mapping Blender +Z-up to pipeline R3 -Y-up.

Used by (a) the GSO loader in ``sequence.py::_load_dataset_camera_poses``
to convert per-view ``.npy`` w2c matrices, (b) this module's
``blender_w2c_to_opencv_c2w`` for held-out test-view cameras, and (c) GT
pose derivation in ``gt_geometry``.  All three sites must use the *same* rotation -- this is the
single source of truth."""


CO3D_LARA_TO_R3 = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
"""diag(1,-1,-1,1): a 180-deg world rotation about X mapping LaRa's CO3D world
(+Y up) to pipeline R3 (-Y up).  Apply as a LEFT multiply ``M @ c2w`` via
:func:`load_co3d_lara_c2w`.

The LaRa c2w *camera* axes are already R3/OpenCV-compatible (X-right, Y-down,
+Z into the scene -- the optical axis points at the object), so only the world
frame is rotated; the camera-local axes are left untouched (right-multiplying
M as well would flip the camera's Y/Z and reverse the optical axis, so every
camera would face away from the object).

Single source of truth for the CO3D convention, mirroring
:data:`BLENDER_TO_R3_WORLD` for GSO.  Used by every CO3D c2w consumer
(input/train + held-out test cameras) so they share one world frame as the
reconstructed Gaussians."""


def load_co3d_lara_c2w(c2w_path: str) -> np.ndarray:
    """Load a LaRa-format CO3D c2w ``.npy`` and return it in pipeline R3.

    The single entry point for reading CO3D camera poses: left-multiplies the
    :data:`CO3D_LARA_TO_R3` world rotation so input/train and held-out test
    cameras share one world frame (-Y up, optical axes facing the object).
    """
    c2w = np.load(c2w_path).astype(np.float32)
    return CO3D_LARA_TO_R3 @ c2w


def co3d_k_from_fov(fov_xy: np.ndarray, H: int, W: int) -> np.ndarray:
    """Pinhole K from a CO3D LaRa per-view ``(fov_x, fov_y)`` pair (radians).

    Shared single-source helper — called by the on-disk loader
    (``Sequence._load_from_co3d_flat``), map-anything's CO3D GT-K conditioning
    (``Sequence._load_with_map_anything``), and the two GT-camera branches in
    this module (``co3d_test_cameras``, ``export_co3d_eval_assets``).
    """
    focal_x = 0.5 * W / np.tan(0.5 * float(fov_xy[0]))
    focal_y = 0.5 * H / np.tan(0.5 * float(fov_xy[1]))
    return np.array([
        [focal_x, 0.0,     W / 2.0],
        [0.0,     focal_y, H / 2.0],
        [0.0,     0.0,     1.0    ],
    ], dtype=np.float32)


def blender_w2c_to_opencv_c2w(w2c_3x4: np.ndarray) -> np.ndarray:
    """Convert a Blender-convention 3x4 w2c to a pipeline-R3 4x4 c2w.

    Applies two rotations: camera-local axis flip ``diag(1, -1, -1)``
    (Blender cam Y-up -> R3 cam Y-down) and world-axis rotation ``R_W``
    (Blender +Z-up -> R3 -Y-up).  Mirrors
    :func:`genia.core.utils.sequence.Sequence._load_dataset_camera_poses` for
    GSO so held-out test-view cameras match pipeline FrameData.c2w.
    """
    M_cam = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    R_W = BLENDER_TO_R3_WORLD
    R_blender = w2c_3x4[:3, :3].astype(np.float32)
    t = w2c_3x4[:3, 3].astype(np.float32)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R_W @ R_blender.T @ M_cam
    c2w[:3, 3] = R_W @ (-R_blender.T @ t)
    return c2w


def _load_heldout_c2w(
    scene_dir: str, indices: range
) -> dict[int, np.ndarray]:
    """Load held-out GSO cameras from render_mvs_25/model/{idx:03d}.npy.

    Returns a dict ``{idx: c2w (4,4)}`` in OpenCV convention.
    """
    render_dir = os.path.join(scene_dir, "render_mvs_25", "model")
    result = {}
    for idx in indices:
        npy_path = os.path.join(render_dir, f"{idx:03d}.npy")
        if not os.path.isfile(npy_path):
            continue
        w2c_3x4 = np.load(npy_path).astype(np.float32)
        result[idx] = blender_w2c_to_opencv_c2w(w2c_3x4)
    return result


def _fr(fk) -> int:
    """Temporal frame index of a ``FrameKey`` (bare ints pass through)."""
    return fk.frame if hasattr(fk, "frame") else int(fk)


def _compose_world_space_foreground(
    state, sequence, frame=None, deformation_warp=None,
) -> Any:
    """Place all decoded foreground Gaussians into the world frame.

    Lifts canonical object-local Gaussians to Blender world space via a frame's
    per-frame pose and that frame's c2w.  ``frame=None`` (default) uses the
    smallest FrameKey (frame 0 / view 0) — static object, any frame equivalent.
    Passing a temporal ``frame`` selects that timestamp's pose token + c2w, so
    a moving object is placed at its world position for that frame (used by the
    per-frame COLMAP export).

    ``deformation_warp`` (``cfg.deformation_warp``) additionally applies the
    per-frame **non-rigid** warp from ``state.canonical_mesh_*`` before the
    Sim(3) — the same canonical-Gaussian → per-frame geometry path as
    ``renders_train/`` and ``gaussians/{obj:03d}/{frame:03d}.ply``.  Without it
    (or on frames with no keyed deformation) the canonical geometry is placed
    rigidly, which is correct only for static scenes.
    """
    from .gaussian import (
        create_gaussians_object,
        join_gaussians,
        transform_scene_to_r3_convention,
        transform_scene_to_world,
    )
    from .refinement import apply_pose_to_gaussian
    from genia.core.utils.deformation import _lookup_per_frame_deformation, warp_gaussians_high_res

    canonical_gaussians = state.canonical_gaussians_with_fallback

    # Both canonical and per-frame Gaussians live in object-local P3D space;
    # the per-frame pose token lifts them into camera space before the
    # R3/world transforms.
    object_gs = []
    for obj_idx, per_frame_tokens in state.tokens_by_object.items():
        if not per_frame_tokens:
            continue
        if frame is None:
            _fk, di = per_frame_tokens[0]
        else:
            match = [(fk, d) for (fk, d) in per_frame_tokens if _fr(fk) == frame]
            if not match:
                continue
            _fk, di = match[0]
        gs_local = canonical_gaussians.get(obj_idx)
        if gs_local is None:
            continue

        rot = di["rotation"].cuda()
        trans = di["translation"].cuda()
        sc = di["scale"].cuda()

        means_override = quats_override = None
        resolved = (
            None if (deformation_warp is None or frame is None) else
            _lookup_per_frame_deformation(
                state.canonical_mesh_verts,
                state.canonical_mesh_per_frame_verts,
                state.canonical_mesh_per_frame_rotations,
                obj_idx, int(frame), gs_local.get_xyz.device,
                state.canonical_mesh_faces,
            )
        )
        if resolved is not None:
            *_warp_core, _faces = resolved
            with torch.no_grad():
                means_override, quats_override = warp_gaussians_high_res(
                    gs_local, *_warp_core,
                    K=int(deformation_warp.knn_k),
                    eps=float(deformation_warp.knn_eps),
                    chunk_size=int(deformation_warp.knn_chunk_size),
                    faces=_faces,
                )

        xyz, rots, scs, opacities, features = apply_pose_to_gaussian(
            gs_local, rot, trans, sc,
            means_override=means_override, rotation_override=quats_override,
        )
        object_gs.append(create_gaussians_object(
            xyz=xyz, features=features, scales=scs,
            rots=rots, opacities=opacities,
        ))

    if not object_gs:
        raise RuntimeError(
            "No canonical or per-frame Gaussians available; cannot compose "
            "a world-space scene for GSO export."
        )

    scene_gs = object_gs[0] if len(object_gs) == 1 else join_gaussians(*object_gs)
    scene_gs = transform_scene_to_r3_convention(scene_gs)

    c2w = sequence[static_anchor_frame_key(sequence, frame)].c2w
    scene_gs = transform_scene_to_world(scene_gs, c2w)
    return scene_gs


def static_anchor_frame_key(sequence, frame=None):
    """The frame key whose c2w lifts a static reconstruction to world.

    Smallest FrameKey by ``(view, frame)`` -- for mono runs the lowest frame
    index, for MV-static (lifted) view 0.  The single definition of the world
    frame that Gaussians, meshes and GT co-visibility masks are all compared in;
    ``frame`` restricts the candidates to one timestamp (falling back to all
    keys when that timestamp has none).
    """
    from genia.core.utils.frame_key import frame_key_sort_key
    candidates = sequence.frame_keys
    if frame is not None:
        candidates = [fk for fk in candidates if _fr(fk) == frame] or sequence.frame_keys
    return min(candidates, key=frame_key_sort_key)


def render_static_eval_views(state, sequence, cameras, H: int, W: int) -> dict:
    """Render a camera list of a STATIC scene -> ``{stem: RGBA uint8 (H, W, 4)}``.

    The shared render core behind the static benchmark exporters (GSO, CO3D).
    ``cameras`` is ``[(stem, c2w, K), ...]`` -- note ``c2w`` before ``K``, the
    opposite order from this module's ``*_test_cameras`` providers.  Producers
    filter their own missing cameras; every entry passed here is rendered.  The
    foreground is composited on white with the rendered coverage carried in the
    alpha band, so an RGB-only consumer that drops alpha sees exactly the
    pre-alpha pixels.

    The Gaussians (``state.canonical_gaussians``) are rendered with gsplat, from
    the scene composited once by ``_compose_world_space_foreground``.

    ``static`` is the real constraint, and it is the abstraction's rather than
    any dataset's: ONE scene is composited and reused across every camera.
    OursActionBench is therefore not routed here -- it rebuilds its scene per
    timestamp (per-frame warp + per-frame Sim(3) + per-frame lift).
    """
    from .rendering import render_gaussians_to_image

    world_scene_gs = _compose_world_space_foreground(state, sequence)
    bg_white = torch.ones(3)
    return {
        stem: _rgba_uint8(*render_gaussians_to_image(
            world_scene_gs, K, W, H, bg_color=bg_white, c2w=c2w,
            return_alpha=True,
        ))
        for stem, c2w, K in cameras
    }


def _rgba_uint8(rgb, alpha) -> np.ndarray:
    """``(rgb (H,W,3), alpha (H,W))`` CUDA tensors -> ``(H,W,4)`` uint8 RGBA."""
    rgb_u8 = (torch.clamp(rgb, 0.0, 1.0).detach().cpu().numpy()
              * 255).astype(np.uint8)
    a_u8 = (torch.clamp(alpha, 0.0, 1.0).detach().cpu().numpy()
            * 255).astype(np.uint8)
    return np.dstack([rgb_u8, a_u8])


def gso_heldout_render_cameras(scene_dir: str, indices: range, H: int, W: int) -> list:
    """EscherNet held-out GSO test cameras as ``[(idx, c2w, K), ...]``.

    The ``render_static_eval_views`` tuple shape (see its note on ordering);
    ``gso_test_cameras`` serves COLMAP export from the same two primitives in its
    own shape.  Indices without a ``.npy`` are dropped.
    """
    c2w_map = _load_heldout_c2w(scene_dir, indices)
    K = gso_blender_intrinsics(H=H, W=W)
    return [(idx, c2w_map[idx], K) for idx in indices if idx in c2w_map]


# =====================================================================
# GSO co-visibility masks (GT-shape-injected runs only)
# =====================================================================
#
# A test-view pixel is *co-visible* when the GT surface point it observes is
# also seen (unoccluded) by at least one input/train view.  Pixels that are
# foreground in a test view but were never observed during reconstruction are
# unconstrained -- the generative decoder hallucinates them -- so restricting
# pixel metrics to the co-visible set isolates *reconstruction* fidelity from
# *completion* quality.
#
# The mask is derived purely from GT geometry: the injected canonical mesh
# (decoded from the GT-inverted shape latent) placed at the GT object pose,
# rendered against the GT input + test cameras.  It is therefore independent of
# the reconstruction's pose/appearance quality, and only well-defined when GT
# shapes were injected (``state.gt_canonical_shape_coords`` populated -- gated at the call
# site).  Written as ``renders_test/{idx:03d}_covis.png`` (uint8 {0, 255}) for a
# co-visible-only (co-PSNR/co-SSIM/co-LPIPS) metric panel.

# A reprojected test point counts as "seen" by an input view when its depth
# matches that view's GT-mesh z-buffer within this fraction of the object's
# world-space bounding-box diagonal.
_COVIS_DEPTH_TOL_FRAC = 0.01


def _render_gt_mesh_depth(verts_world, faces_i32, glctx, c2w, K, H, W):
    """GT-mesh z-depth + silhouette ``(H, W)`` from one R3 camera ``(c2w, K)``."""
    from .mesh_rendering import render_depth_and_alpha

    device = verts_world.device
    w2c = np.linalg.inv(np.asarray(c2w, dtype=np.float32))
    R = torch.as_tensor(w2c[:3, :3], dtype=torch.float32, device=device)
    t = torch.as_tensor(w2c[:3, 3], dtype=torch.float32, device=device)
    verts_cam = verts_world @ R.T + t  # R3 world -> R3 camera space
    return render_depth_and_alpha(
        verts_cam, faces_i32, glctx,
        float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]), H, W,
    )


def _covisible_mask_from_depths(z_test, a_test, c2w_test, K_test, input_buffers, tau):
    """Boolean co-visibility mask ``(H, W)`` uint8 {0, 255}.

    A test-view foreground pixel is co-visible when the 3D point it unprojects
    to reprojects -- unoccluded (input z-buffer match within ``tau``) -- into at
    least one input view.  Pure geometry (device-agnostic): the depth/silhouette
    buffers can come from any renderer.

    Parameters
    ----------
    z_test, a_test : (H, W) tensors -- test-view z-depth + silhouette alpha.
    c2w_test : (4, 4) ; K_test : (3, 3) -- test camera (R3 cam->world, pinhole).
    input_buffers : list of ``(z (H,W), a (H,W), c2w (4,4), K (3,3))`` per input view.
    tau : float -- depth-match tolerance in world units.
    """
    device = z_test.device
    H, W = z_test.shape
    mask = torch.zeros((H, W), dtype=torch.uint8, device=device)
    ys, xs = torch.nonzero(a_test > 0.5, as_tuple=True)
    if ys.numel() == 0:
        return mask

    def _t(a):
        return torch.as_tensor(
            np.asarray(a, dtype=np.float32), dtype=torch.float32, device=device
        )

    Kt = _t(K_test)
    fx, fy, cx, cy = Kt[0, 0], Kt[1, 1], Kt[0, 2], Kt[1, 2]
    zt = z_test[ys, xs]
    pc = torch.stack(
        [(xs.float() - cx) / fx * zt, (ys.float() - cy) / fy * zt, zt], dim=-1
    )
    c2w = _t(c2w_test)
    pw = pc @ c2w[:3, :3].T + c2w[:3, 3]  # test foreground -> R3 world

    covis = torch.zeros(ys.numel(), dtype=torch.bool, device=device)
    for z_in, a_in, c2w_in, K_in in input_buffers:
        Hi, Wi = z_in.shape
        w2c = torch.linalg.inv(_t(c2w_in))[:3]
        Ki = _t(K_in)
        fxi, fyi, cxi, cyi = Ki[0, 0], Ki[1, 1], Ki[0, 2], Ki[1, 2]
        p = pw @ w2c[:, :3].T + w2c[:, 3]
        zc = p[:, 2]
        u = (fxi * p[:, 0] / zc + cxi).round().long()
        v = (fyi * p[:, 1] / zc + cyi).round().long()
        inb = (zc > 1e-6) & (u >= 0) & (u < Wi) & (v >= 0) & (v < Hi)
        uc, vc = u.clamp(0, Wi - 1), v.clamp(0, Hi - 1)
        covis |= inb & (a_in[vc, uc] > 0.5) & ((zc - z_in[vc, uc]).abs() < tau)

    mask[ys[covis], xs[covis]] = 255
    return mask


def _export_gso_covisibility_masks(
    cfg, state, sequence, scene_dir, out_dir, pipeline_obj
) -> None:
    """Write per-test-view co-visibility masks for a GT-shape-injected GSO run.

    For each EscherNet held-out test view, marks the foreground pixels whose GT
    surface point is unoccluded in >=1 input view, saving
    ``{out_dir}/{idx:03d}_covis.png`` (uint8 {0, 255}, at ``sequence.H/W`` so it
    pixel-aligns with ``renders_test/{idx:03d}.png``).  No-op for per-frame-only
    runs (no canonical SLAT to decode the GT mesh from).
    """
    import nvdiffrast.torch as dr
    from PIL import Image

    from genia.core.utils.slat_decode import redecode_slat

    if not state.canonical_slats:
        print("  co-visibility masks: skipped (no canonical SLAT)")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H, W = sequence.H, sequence.W
    K = gso_blender_intrinsics(H=H, W=W)  # GT analytic Blender intrinsics

    # ---- Combined GT mesh in R3 world (all foreground objects) ----
    # Decode each object's GT-injected canonical SLAT to a mesh and place it
    # with the exact (pose token, anchor c2w) the predicted Gaussians use (see
    # ``_compose_world_space_foreground``), so the GT silhouette + the test
    # cameras share the world frame the rendered test views are produced in
    # (both anchor via ``static_anchor_frame_key``).
    anchor_c2w = sequence[static_anchor_frame_key(sequence)].c2w
    world_verts, faces_list, voff = [], [], 0
    for obj_idx, per_frame_tokens in state.tokens_by_object.items():
        if obj_idx == 0 or not per_frame_tokens:
            continue  # background has no GT mesh
        slat = state.canonical_slats.get(obj_idx)
        if slat is None:
            continue
        md = redecode_slat(pipeline_obj, slat, formats=["mesh"])["mesh"][0]
        verts = md.vertices.to(device).float()  # object-local P3D
        di = per_frame_tokens[0][1]
        world = _pose_mesh_verts_to_world(
            verts, di["rotation"].to(device), di["translation"].to(device),
            di["scale"].to(device), anchor_c2w,
        )
        world_verts.append(world)
        faces_list.append(md.faces.to(device).to(torch.int32) + voff)
        voff += verts.shape[0]

    if not world_verts:
        print("  co-visibility masks: skipped (no foreground GT mesh)")
        return

    verts_world = torch.cat(world_verts, dim=0)
    faces_i32 = torch.cat(faces_list, dim=0).contiguous()
    diag = float((verts_world.amax(0) - verts_world.amin(0)).norm())
    tau = _COVIS_DEPTH_TOL_FRAC * diag

    glctx = dr.RasterizeCudaContext(device=device)

    # ---- Input/train-view GT-mesh depth buffers (occlusion oracle) ----
    input_buffers = []  # (z (H,W), alpha (H,W), c2w (4,4), K (3,3))
    for fk in sequence.frame_keys:
        c2w = np.asarray(sequence[fk].c2w, dtype=np.float32)
        z, a = _render_gt_mesh_depth(verts_world, faces_i32, glctx, c2w, K, H, W)
        input_buffers.append((z, a, c2w, K))

    # ---- Per test view: render depth, mark co-visible foreground ----
    c2w_test = _load_heldout_c2w(scene_dir, range(10, 25))
    n_saved, covis_fracs = 0, []
    for idx in range(10, 25):
        c2w = c2w_test.get(idx)
        if c2w is None:
            continue
        c2w = np.asarray(c2w, dtype=np.float32)
        z_test, a_test = _render_gt_mesh_depth(
            verts_world, faces_i32, glctx, c2w, K, H, W
        )
        mask = _covisible_mask_from_depths(
            z_test, a_test, c2w, K, input_buffers, tau
        )
        fg = int((a_test > 0.5).sum())
        if fg:
            covis_fracs.append(float((mask > 0).sum()) / fg)
        Image.fromarray(mask.cpu().numpy()).save(
            os.path.join(out_dir, f"{idx:03d}_covis.png")
        )
        n_saved += 1

    if n_saved:
        frac = float(np.mean(covis_fracs)) if covis_fracs else 0.0
        print(
            f"  Saved {n_saved} co-visibility masks -> "
            f"renders_test/{{idx:03d}}_covis.png "
            f"(mean co-visible foreground {frac:.0%})"
        )


def export_gso_eval_assets(
    cfg, state, sequence, final_output_dir: str, pipeline_obj
) -> None:
    """Export EscherNet-protocol GSO held-out NVS renders for the current run.

    Writes ``{final_output_dir}/renders_test/{idx:03d}.png`` for idx in
    10..24, at the **train-view resolution** (``sequence.H``/``sequence.W``)
    -- the GSO 256² eval protocol is intentionally not enforced here so
    test views match train views (``renders_train/``). Guarded by
    ``cfg.dataset.name == 'gso'`` at the call site.

    Canonical pipelines export the mesh via ``save_per_object_mesh``
    (``meshes/{obj:03d}.glb``, gated on ``has_canonical`` in the FINAL
    block). Per-frame-only runs have ``has_canonical=False`` so that writer
    is skipped -- :func:`_export_gso_perframe_meshes` (called below) fills the
    gap with object-local ``meshes/{obj:03d}/{stem}.ply``. Either way the
    evaluator applies GSO pose composition at eval time.

    Exceptions propagate -- a failure here should not silently hide behind
    a successful FINAL block.
    """
    from PIL import Image

    out_dir = os.path.join(final_output_dir, "renders_test")
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "-" * 40)
    print("Exporting GSO held-out NVS renders ...")
    print("-" * 40)

    scene_dir = os.path.join(cfg.dataset.path, cfg.dataset.scene_name)
    cameras = gso_heldout_render_cameras(
        scene_dir, range(10, 25), sequence.H, sequence.W
    )
    renders = render_static_eval_views(
        state, sequence, cameras, sequence.H, sequence.W
    )
    for idx, rgb in renders.items():
        Image.fromarray(rgb).save(os.path.join(out_dir, f"{idx:03d}.png"))
    if renders:
        rendered_keys = sorted(renders.keys())
        print(
            f"  Saved {len(renders)} NVS renders ({rendered_keys[0]:03d}.."
            f"{rendered_keys[-1]:03d}.png) -> renders_test/ "
            f"at {sequence.W}x{sequence.H} RGBA (transparent bg)"
        )
    else:
        print("  WARNING: no NVS renders produced (no .npy test cameras found)")

    _export_gso_perframe_meshes(cfg, state, final_output_dir, pipeline_obj)

    # Co-visibility masks for the co-PSNR/co-SSIM/co-LPIPS diagnostic -- only
    # meaningful when the GT mesh was injected.
    if state.gt_canonical_shape_coords:
        _export_gso_covisibility_masks(
            cfg, state, sequence, scene_dir, out_dir, pipeline_obj
        )


def _export_gso_perframe_meshes(
    cfg, state, final_output_dir: str, pipeline_obj
) -> None:
    """Per-object object-local meshes for per-frame-only GSO runs.

    Canonical GSO pipelines already get ``meshes/{obj:03d}.glb`` from
    ``save_per_object_mesh`` (gated on ``has_canonical``) in the FINAL
    block.  Per-frame-only runs have ``has_canonical=False`` so that writer
    is skipped — mirror the
    OursActionBench per-frame-only branch (``_resolve_perframe_mesh_verts``)
    and decode each frame's own SLAT to an **object-local** mesh PLY at
    ``meshes/{obj:03d}/{stem}.ply`` (same per-object/per-frame layout as
    ``gaussians/{obj:03d}/{stem}.ply`` from ``save_perframe_per_object_ply``;
    colourless geometry-only PLY — Chamfer/IoU need geometry only and the
    per-frame encoding cost stays low).  ``final/poses.json`` carries the
    per-frame Sim(3) an evaluator composes the object-local mesh into Blender
    world with.

    No-op when the run is canonical or ``output.save_output_mesh`` is off.
    """
    if state.has_canonical or not cfg.output.save_output_mesh:
        return

    import trimesh

    from .frame_key import frame_key_stem
    from genia.core.utils.slat_decode import redecode_slat

    meshes_root = os.path.join(final_output_dir, "meshes")
    n_mesh = 0
    for obj_idx, toks in sorted(state.tokens_by_object.items()):
        if not toks:
            continue
        for fk, di in toks:
            slat = di.get("decoder_input_slat")
            if slat is None:
                continue
            md = redecode_slat(pipeline_obj, slat, formats=["mesh"])["mesh"][0]
            obj_dir = os.path.join(meshes_root, f"{obj_idx:03d}")
            os.makedirs(obj_dir, exist_ok=True)
            trimesh.Trimesh(
                vertices=md.vertices.detach().cpu().numpy(),
                faces=md.faces.detach().cpu().numpy(),
                process=False,
            ).export(os.path.join(obj_dir, f"{frame_key_stem(fk)}.ply"))
            n_mesh += 1
    if n_mesh:
        print(
            f"  Saved {n_mesh} per-object per-frame meshes -> "
            f"meshes/{{obj:03d}}/{{stem}}.ply (object-local, colourless)"
        )


# =====================================================================
# OursActionBench: per-timestamp held-out hemisphere NVS + per-frame mesh
# =====================================================================
#
# Dynamic analogue of the GSO export above.  OursActionBench is a deforming
# scene reconstructed against per-frame GT poses + GT shapes (the fitted
# input camera is baked into the per-frame object pose; FrameData.c2w stays
# identity).  For each timestamp we render the prediction from a held-out
# upper-hemisphere camera and export the per-frame predicted mesh, so an
# evaluator can score per-frame PSNR/SSIM/LPIPS + Chamfer/IoU with the same
# metric core as GSO.
#
# Coordinate path (per frame i), mirroring the keyframes renderer
# ``save_canonical_renders_perframe`` but with a *different* c2w for the
# world-lift vs the render camera:
#
#   canonical (object-local P3D)
#     -> warp_gaussians_high_res / _warp_at_high_res   (per-frame deform)
#     -> apply_pose (rot_i,trans_i,scale_i)            (-> input-cam P3D)
#     -> transform_scene_to_r3_convention              (-> input-cam R3)
#     -> transform_scene_to_world(input_c2w[i])        (-> oursactionbench world)
#     -> render at (K_test[i], test_c2w[i])
#
# Both input_c2w[i] and test_c2w[i] come from camera.json.


def _pose_mesh_verts_to_world(verts, rotation, translation, scale, input_c2w):
    """Apply the per-frame Sim(3) + P3D->R3 + input-c2w lift to raw mesh
    vertices, reusing the *exact* convention helpers the Gaussian render
    path uses (``apply_pose_to_gaussian`` position math +
    ``p3d_to_r3_positions`` + ``transform_gaussian_params_cam_to_world``).
    Returns world-space vertices ``(V, 3)`` matching the rendered Gaussians.
    """
    from pytorch3d.transforms import quaternion_to_matrix

    from .quaternion_ops import p3d_to_r3_positions
    from .rendering import transform_gaussian_params_cam_to_world

    rot = rotation.squeeze(0) if rotation.dim() == 2 else rotation
    rot = rot / rot.norm()
    trans = translation.squeeze(0) if translation.dim() == 2 else translation
    if scale.dim() == 0:
        scale = scale.expand(3)
    elif scale.dim() == 1 and scale.shape[0] == 1:
        scale = scale.expand(3)
    elif scale.dim() == 2:
        scale = scale.squeeze(0)
        if scale.shape[0] == 1:
            scale = scale.expand(3)

    R = quaternion_to_matrix(rot.unsqueeze(0)).squeeze(0)  # (3, 3)
    posed = torch.mm(verts * scale, R) + trans             # apply_pose math
    r3 = p3d_to_r3_positions(posed)                        # P3D -> R3 cam
    dummy_q = torch.zeros((verts.shape[0], 4), device=verts.device)
    dummy_q[:, 0] = 1.0
    world, _ = transform_gaussian_params_cam_to_world(r3, dummy_q, input_c2w)
    return world


def _resolve_perframe_mesh_verts(
    per_frame_only, di, obj_idx, device, meshes_root,
    resolved, warp_kw, dw, pipeline_obj,
):
    """Object-local ``(verts_tensor, faces_np)`` for one obj/frame, or None.

    Per-frame-only decodes this frame's own SLAT (no canonical GLB exists);
    the per-frame token already encodes the deformed shape, so no warp is
    applied. Canonical reloads the GLB ``save_per_object_mesh`` wrote and
    applies the per-vertex deformation warp when keyed for this frame.
    """
    import trimesh

    from genia.core.utils.deformation import _warp_at_high_res
    from genia.core.utils.slat_decode import redecode_slat

    if per_frame_only:
        slat = di.get("decoder_input_slat")
        if slat is None:
            return None
        md = redecode_slat(pipeline_obj, slat, formats=["mesh"])["mesh"][0]
        return md.vertices.to(device).float(), md.faces.detach().cpu().numpy()

    glb_path = os.path.join(meshes_root, f"{obj_idx:03d}.glb")
    if not os.path.isfile(glb_path):
        return None
    m = trimesh.load(glb_path, force="mesh", process=False)
    mv = torch.as_tensor(
        np.asarray(m.vertices), dtype=torch.float32, device=device
    )
    faces_np = np.asarray(m.faces)
    if resolved is not None:
        *_warp_core, _faces = resolved
        with torch.no_grad():
            mv, _ = _warp_at_high_res(
                mv, *_warp_core, K=warp_kw["K"], eps=warp_kw["eps"],
                chunk_size=warp_kw["chunk_size"],
                compute_rotation_blend=False,
                faces=_faces,
            )
    return mv, faces_np


# ---------------------------------------------------------------------------
# World-space composites: the whole scene in one file per frame, in the SAME
# world frame as final/colmap/ -- so world + background + the COLMAP points
# open on top of each other in a viewer.
# ---------------------------------------------------------------------------

def _world_asset_path(root: str, stem: str, frame, ext: str) -> str:
    """``{root}/{stem}.{ext}`` (static) or ``{root}/{stem}/{frame:03d}.{ext}``
    (dynamic) -- the per-frame split ``colmap/sparse/{frame:03d}`` already uses."""
    if frame is None:
        return os.path.join(root, f"{stem}.{ext}")
    os.makedirs(os.path.join(root, stem), exist_ok=True)
    return os.path.join(root, stem, f"{int(frame):03d}.{ext}")


def _world_anchor_c2w(cfg, sequence, frame):
    """The c2w that lifts ``frame``'s reconstruction into world space.

    OAB's ``FrameData.c2w`` is identity -- its real cameras live in
    ``camera.json`` -- so the Gaussian path routes through
    ``_oursactionbench_world_scene``, which lifts by that file's per-frame input
    c2w.  The mesh path has to read the SAME source or the two world files land
    in different frames.  Every other dataset lifts by the anchor FrameKey's c2w.
    """
    if str(cfg.dataset.name).lower() == "oursactionbench":
        from genia.core.utils.gt_data import load_actionbench_camera_fit

        cam = load_actionbench_camera_fit(cfg.dataset.scene_name, cfg.dataset.path)
        per_frame_in = cam.get("per_frame_cameras") or []
        idx = 0 if frame is None else int(frame)
        if idx < len(per_frame_in):
            return np.array(per_frame_in[idx]["c2w"], dtype=np.float32)
    return sequence[static_anchor_frame_key(sequence, frame)].c2w


def _save_world_gaussian(gaussian, root: str, stem: str, frame, compressed: bool) -> str:
    """Write one world-space Gaussian file, honouring ``save_compressed_ply``
    (same ``.compressed.ply`` convention as the per-object assets)."""
    from .io_utils import save_gaussian_compressed_ply, save_gaussian_ply

    path = _world_asset_path(
        root, stem, frame, "compressed.ply" if compressed else "ply"
    )
    (save_gaussian_compressed_ply if compressed else save_gaussian_ply)(gaussian, path)
    return path


def _export_world_meshes(cfg, state, sequence, meshes_root, frames, pipeline_obj) -> int:
    """All objects joined into one world-space mesh per frame -> ``meshes/world*.glb``.

    Per object: the canonical ``meshes/{obj:03d}.glb`` (or, on per-frame-only
    runs, that frame's own SLAT) -> per-frame deformation warp -> the SAME
    Sim(3) + P3D->R3 + c2w chain the Gaussians take, so mesh and Gaussian world
    files coincide.  Vertex colours are reloaded from the per-object GLB rather
    than decoded a second time.
    """
    import trimesh

    from genia.core.utils.deformation import _lookup_per_frame_deformation

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dw = cfg.deformation_warp
    warp_kw = {"K": int(dw.knn_k), "eps": float(dw.knn_eps),
               "chunk_size": int(dw.knn_chunk_size)}
    per_frame_only = not state.has_canonical
    colors_by_obj = {}   # the per-object GLB's colours are frame-independent
    n = 0
    for frame in frames:
        anchor_c2w = _world_anchor_c2w(cfg, sequence, frame)
        verts_all, faces_all, colors_all, voff = [], [], [], 0
        for obj_idx, per_frame_tokens in state.tokens_by_object.items():
            if obj_idx == 0 or not per_frame_tokens:
                continue
            match = (per_frame_tokens[:1] if frame is None else
                     [(fk, d) for (fk, d) in per_frame_tokens if _fr(fk) == frame])
            if not match:
                continue
            di = match[0][1]
            resolved = None if frame is None else _lookup_per_frame_deformation(
                state.canonical_mesh_verts, state.canonical_mesh_per_frame_verts,
                state.canonical_mesh_per_frame_rotations,
                obj_idx, int(frame), device, state.canonical_mesh_faces,
            )
            mesh = _resolve_perframe_mesh_verts(
                per_frame_only, di, obj_idx, device, meshes_root,
                resolved, warp_kw, dw, pipeline_obj,
            )
            if mesh is None:
                continue
            mv, faces_np = mesh
            with torch.no_grad():
                mw = _pose_mesh_verts_to_world(
                    mv, di["rotation"].to(device), di["translation"].to(device),
                    di["scale"].to(device), anchor_c2w,
                )
            if obj_idx not in colors_by_obj:
                colors_by_obj[obj_idx] = _glb_vertex_colors(
                    meshes_root, obj_idx, mv.shape[0]
                )
            verts_all.append(mw.detach().cpu().numpy())
            faces_all.append(np.asarray(faces_np) + voff)
            colors_all.append(colors_by_obj[obj_idx])
            voff += int(mv.shape[0])

        if not verts_all:
            continue
        trimesh.Trimesh(
            vertices=np.concatenate(verts_all, axis=0),
            faces=np.concatenate(faces_all, axis=0),
            vertex_colors=(np.concatenate(colors_all, axis=0)
                           if all(c is not None for c in colors_all) else None),
            process=False,
        ).export(_world_asset_path(meshes_root, "world", frame, "glb"))
        n += 1
    return n


def _glb_vertex_colors(meshes_root: str, obj_idx: int, n_verts: int):
    """Per-vertex RGBA of ``meshes/{obj:03d}.glb``, or None when unavailable or
    a different vertex count (one None makes the whole join colourless)."""
    import trimesh

    glb_path = os.path.join(meshes_root, f"{obj_idx:03d}.glb")
    if not os.path.isfile(glb_path):
        return None
    visual = trimesh.load(glb_path, force="mesh", process=False).visual
    colors = getattr(visual, "vertex_colors", None)
    if colors is None:
        return None
    colors = np.asarray(colors)
    return colors if colors.shape[0] == n_verts else None


def export_world_space_assets(cfg, state, sequence, final_output_dir, pipeline_obj):
    """Whole-scene world-space composites, aligned with ``final/colmap/``::

        gaussians/world.ply        all foreground objects, posed, in world space
        gaussians/background.ply   every frame's background, each lifted by its
                                   own (K, c2w), unioned
        meshes/world.glb           the same foreground objects as one mesh

    Dynamic scenes get ``{stem}/{frame:03d}.{ext}`` per temporal frame instead
    of a single file, 1:1 with ``colmap/sparse/{frame:03d}``.  The background
    stays ONE file: it already spans every frame's camera.

    **Alignment invariant.** The foreground comes from
    ``colmap_export.world_scene_gaussians`` -- the very function the COLMAP
    point cloud is built from -- so ``world.*`` and ``colmap/sparse/{f}`` are
    the same geometry by construction, not by convention.  The background is
    placed by each frame's own c2w, the same lift ``create_background_gaussians``
    performs for every render.  Background SH bands are padded to match the
    foreground so both PLYs declare the same ``f_rest_*`` count and a viewer
    loads them as one scene.

    Note these files are R3-**world**, unlike the sibling per-object
    ``{obj:03d}.ply`` / ``.glb``, which stay object-local PyTorch3D.

    Export artifact -- wrap the call in ``_TIMER.exclude()``.
    """
    from .colmap_export import world_scene_gaussians
    from .gaussian import aggregate_background_gaussians, match_sh_bands

    print("\n" + "-" * 40)
    print("Exporting world-space composites ...")
    print("-" * 40)

    # Dynamic -> one file per temporal frame (1:1 with colmap/sparse/{frame:03d});
    # static -> a single flat file, written under the sentinel frame None.
    frames = (sorted({_fr(fk) for fk in sequence.frame_keys})
              if sequence.is_dynamic else [None])
    per_frame_suffix = "/{frame:03d}" if sequence.is_dynamic else ""
    gaussians_root = os.path.join(final_output_dir, "gaussians")
    os.makedirs(gaussians_root, exist_ok=True)
    compressed = bool(cfg.output.save_compressed_ply)

    n_fg, sh_bands = 0, 1
    for frame in frames:
        try:
            scene_gs = world_scene_gaussians(cfg, state, sequence, frame)
        except Exception as e:  # noqa: BLE001 — an export must not fail FINAL
            print(f"  world Gaussians: skipped for frame {frame} ({e})")
            continue
        sh_bands = max(sh_bands, int(scene_gs.get_features.shape[1]))
        _save_world_gaussian(scene_gs, gaussians_root, "world", frame, compressed)
        n_fg += 1
    if n_fg:
        print(f"  {n_fg} world Gaussian file(s) -> gaussians/world"
              f"{per_frame_suffix} ({sh_bands} SH band(s))")

    # Background: already world-space (create_background_gaussians applies each
    # frame's c2w itself), so it needs no convention transform of its own.
    object_ids = sorted(oi for oi in state.tokens_by_object if oi != 0)
    bg_gs = aggregate_background_gaussians(sequence, object_ids)
    if bg_gs is None:
        print("  background: skipped (no valid background pixels)")
    else:
        bg_gs = match_sh_bands(bg_gs, sh_bands)
        path = _save_world_gaussian(
            bg_gs, gaussians_root, "background", None, compressed
        )
        print(f"  background -> gaussians/{os.path.basename(path)} "
              f"({bg_gs.get_xyz.shape[0]} points, {len(sequence.frame_keys)} frames)")

    if cfg.output.save_output_mesh:
        meshes_root = os.path.join(final_output_dir, "meshes")
        os.makedirs(meshes_root, exist_ok=True)
        n_mesh = _export_world_meshes(
            cfg, state, sequence, meshes_root, frames, pipeline_obj
        )
        if n_mesh:
            print(f"  {n_mesh} world mesh file(s) -> "
                  f"meshes/world{per_frame_suffix}.glb")
        else:
            print("  world mesh: skipped (no per-object mesh available)")


def export_coarse_shape_wireframe(cfg, state, sequence, block_dir, pipeline_obj) -> int:
    """Posed coarse-shape wireframe preview -> ``coarse_shape/world[.glb|/{f:03d}.glb]``.

    A mid-pipeline preview for an interactive 3D viewer, NOT a FINAL export: called right after the
    ``global_pose_refine_1`` block, when the Stage-1 shape latent (a ``(1,4096,8)``
    tensor decoding to a 64**3 occupancy grid) and per-frame poses already exist,
    long before Stage-2 appearance refinement. ``block_dir`` is the caller-resolved
    ``NN_global_pose_refine_1`` output dir (see ``block_output_subdir``); this
    function only owns the ``coarse_shape/`` sub-path under it, mirroring
    ``final/meshes/world*.glb``'s naming convention one level down so the two are
    never confused for each other.

    Exported as literal ``GL_LINES`` geometry (a ``trimesh.path.Path3D`` GLB), not a
    normal triangle mesh, so "wireframe" is baked into the file itself and renders
    as one in any glTF viewer, with no per-asset material override needed.

    Edge topology (``edges_unique``) is decoded once per object, not once per
    (object, frame): posing only moves vertices, it never changes which vertices
    share a face. Per frame, each object's posed segment array is built
    independently and concatenated -- since only line segments are needed (not a
    watertight combined mesh), there is no cross-object vertex-index offset to get
    right, unlike ``_export_world_meshes``.

    A ``.complete`` sentinel is written once every frame has been exported, so a
    concurrent reader (e.g. a viewer polling the directory) never sees a
    directory -- or, for a static run, a lone ``world.glb`` -- mid-write.

    Callers must wrap this in ``_TIMER.exclude()`` (its call site sits inside a
    timed pipeline block, unlike every other export in this file) and in a
    ``try/except`` (a preview must never fail the block it's attached to --
    mirrors ``export_orbit_viz``'s own call-site precedent, not handled in here).
    """
    import trimesh

    from .mesh_rendering import diffmc_mesh
    from genia.core.utils.slat_decode import decode_shape_to_occ

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    canonical_by_obj = {}  # obj_idx -> (verts (V,3) canonical, edges (E,2) long)
    for obj_idx, raw in state.canonical_raw_modalities.items():
        if obj_idx == 0:
            continue
        occ = decode_shape_to_occ(pipeline_obj, raw["shape"].to(device))
        verts, faces = diffmc_mesh(occ, device)
        if verts.shape[0] == 0 or faces.shape[0] == 0:
            continue  # iso-surface didn't form -- same guard rendering_guidance.py uses
        edges = trimesh.Trimesh(
            vertices=verts.detach().cpu().numpy(), faces=faces.cpu().numpy(),
            process=False,
        ).edges_unique
        canonical_by_obj[obj_idx] = (verts, torch.as_tensor(edges, device=device))
    if not canonical_by_obj:
        return 0

    frames = (sorted({_fr(fk) for fk in sequence.frame_keys})
              if sequence.is_dynamic else [None])
    coarse_root = os.path.join(block_dir, "coarse_shape")
    os.makedirs(coarse_root, exist_ok=True)
    n = 0
    for frame in frames:
        anchor_c2w = _world_anchor_c2w(cfg, sequence, frame)
        segments = []
        for obj_idx, (verts, edges) in canonical_by_obj.items():
            per_frame_tokens = state.tokens_by_object.get(obj_idx, [])
            match = (per_frame_tokens[:1] if frame is None else
                     [(fk, d) for (fk, d) in per_frame_tokens if _fr(fk) == frame])
            if not match:
                continue
            di = match[0][1]
            world_verts = _pose_mesh_verts_to_world(
                verts, di["rotation"].to(device), di["translation"].to(device),
                di["scale"].to(device), anchor_c2w,
            )
            segments.append(world_verts[edges].detach().cpu().numpy())  # (E,2,3)
        if not segments:
            continue
        path = _world_asset_path(coarse_root, "world", frame, "glb")
        # Path3D.export() only knows path formats (dxf/svg/...), not glb -- .scene()
        # wraps it so the general glTF exporter handles it, writing GL_LINES verbatim
        # (verified: trimesh has no single-call Path3D-to-glb shortcut for this).
        trimesh.load_path(np.concatenate(segments, axis=0)).scene().export(path)
        n += 1

    if n:
        open(os.path.join(coarse_root, ".complete"), "w").close()
    return n


def _test_intrinsics_scaled(cam_entry, json_image_hw, out_h, out_w):
    """Scale a camera.json K to the ``out_h`` x ``out_w`` render resolution
    (focal scales with resolution per axis; principal point recentred).
    Test views render at the train resolution -- no fixed 256 protocol."""
    h_json = float(json_image_hw[0])
    w_json = float(json_image_hw[1] if len(json_image_hw) > 1 else json_image_hw[0])
    K = np.array(cam_entry["K"], dtype=np.float32).copy()
    K[0, 0] *= float(out_w) / w_json
    K[1, 1] *= float(out_h) / h_json
    K[0, 2] = (out_w - 1) * 0.5
    K[1, 2] = (out_h - 1) * 0.5
    return K


# =====================================================================
# Held-out test-camera derivation, decoupled from rendering.
#
# Each helper returns a list of ``(K (3,3), c2w (4,4), src_render_path,
# colmap_name)`` for the held-out views of one dataset, where
# ``src_render_path`` is the render PNG this module already writes under
# ``final/`` and ``colmap_name`` is the flat, globally-unique basename used by
# the COLMAP exporter (``core/utils/colmap_export.py``).  These reuse the
# SAME camera primitives the corresponding ``export_*_eval_assets`` renderers
# use (``_load_heldout_c2w`` / ``gso_blender_intrinsics`` /
# ``load_actionbench_camera_fit`` / ``_test_intrinsics_scaled`` / the CO3D
# meta+fovs reads), so the COLMAP cameras match the rendered test views.
# =====================================================================


def gso_test_cameras(cfg, sequence, indices: range = range(10, 25)):
    """GSO held-out NVS cameras (R3/OpenCV c2w), one per available view."""
    scene_dir = os.path.join(cfg.dataset.path, cfg.dataset.scene_name)
    c2w_map = _load_heldout_c2w(scene_dir, indices)
    K = gso_blender_intrinsics(H=sequence.H, W=sequence.W)
    return [
        (K, c2w_map[idx], f"renders_test/{idx:03d}.png", f"test_{idx:03d}.png")
        for idx in indices if idx in c2w_map
    ]


def co3d_test_cameras(cfg, sequence):
    """CO3D target-view cameras (R3/OpenCV c2w); GT (override) or MA-predicted,
    matching ``export_co3d_eval_assets``."""
    import json as _json

    scene_dir = os.path.join(cfg.dataset.path, cfg.dataset.scene_name)
    meta_path = os.path.join(scene_dir, "meta.json")
    if not os.path.isfile(meta_path):
        return []
    with open(meta_path) as f:
        meta = _json.load(f)
    target_indices = [
        i for i, v in enumerate(meta.get("views") or [])
        if v.get("role") == "target"
    ]
    if not target_indices:
        return []
    H, W = sequence.H, sequence.W
    use_gt = cfg.processing.camera_poses_source == "gt"
    fovs = None
    if use_gt:
        fovs_path = os.path.join(scene_dir, "fovs.npy")
        if not os.path.isfile(fovs_path):
            return []
        fovs = np.load(fovs_path).astype(np.float32)
    out = []
    for idx in target_indices:
        if use_gt:
            c2w_path = os.path.join(scene_dir, f"{idx:03d}.npy")
            if not os.path.isfile(c2w_path):
                continue
            c2w = load_co3d_lara_c2w(c2w_path)
            K = co3d_k_from_fov(fovs[idx], H, W)
        else:
            cam = sequence.predicted_camera(idx)
            if cam is None:
                continue
            c2w, K = cam
        out.append((np.asarray(K, np.float32), np.asarray(c2w, np.float32),
                    f"renders_test/{idx:03d}.png", f"test_{idx:03d}.png"))
    return out


def oursactionbench_test_cameras(cfg, sequence):
    """OAB per-frame held-out test cameras (R3 c2w + scaled K) from camera.json,
    matching the test views ``export_oursactionbench_eval_assets`` renders."""
    from genia.core.utils.gt_data import load_actionbench_camera_fit

    cam = load_actionbench_camera_fit(cfg.dataset.scene_name, cfg.dataset.path)
    per_frame_test = cam.get("per_frame_test_cameras")
    json_hw = cam.get("image_hw", [512, 512])
    if not per_frame_test:
        return []
    H, W = sequence.H, sequence.W
    out, seen = [], set()
    for fk in sequence.frame_keys:
        f = fk.frame if hasattr(fk, "frame") else int(fk)
        if f in seen or f >= len(per_frame_test):
            continue
        seen.add(f)
        K = _test_intrinsics_scaled(per_frame_test[f], json_hw, H, W)
        c2w = np.array(per_frame_test[f]["c2w"], dtype=np.float32)
        out.append((K, c2w, f"renders_test/{f:02d}.png", f"test_{f:02d}.png"))
    return out


def oursactionbench_input_cameras(cfg, sequence):
    """OAB per-frame INPUT (train) cameras from camera.json.  FrameData.c2w is
    identity for OAB (the fitted input camera is baked into the per-frame object
    pose), so the meaningful train cameras live in camera.json per_frame_cameras
    — in the same OAB world frame as the test cameras + the point cloud."""
    from genia.core.utils.gt_data import load_actionbench_camera_fit

    cam = load_actionbench_camera_fit(cfg.dataset.scene_name, cfg.dataset.path)
    per_frame_in = cam.get("per_frame_cameras")
    json_hw = cam.get("image_hw", [512, 512])
    if not per_frame_in:
        return []
    H, W = sequence.H, sequence.W
    out, seen = [], set()
    for fk in sequence.frame_keys:
        f = fk.frame if hasattr(fk, "frame") else int(fk)
        if f in seen or f >= len(per_frame_in):
            continue
        seen.add(f)
        K = _test_intrinsics_scaled(per_frame_in[f], json_hw, H, W)
        c2w = np.array(per_frame_in[f]["c2w"], dtype=np.float32)
        out.append((K, c2w, f"renders_train/{f:03d}.png", f"train_{f:03d}.png"))
    return out


def export_co3d_eval_assets(
    cfg, state, sequence, final_output_dir: str, pipeline_obj
) -> None:
    """Export LaRa-protocol CO3D held-out NVS renders for the current run.

    LaRa's CO3D preprocessing materializes
    exactly 8 views per scene -- 4 input (asset indices 0..3) + 4 target
    (4..7) -- with roles recorded in ``meta.json``.  The pipeline reconstructs
    from the input views; this function renders the same scene from every
    labelled view and writes flat ``{NNN:03d}.png`` files at the train-view
    resolution (``sequence.H``/``sequence.W``) as foreground-only RGBA with a
    transparent background (foreground composited on white, rendered coverage
    in the alpha channel):

    * *target* views -> ``{final}/renders_test/{NNN:03d}.png`` (the LaRa NVS
      headline metric).
    * *input* views -> ``{final}/renders_train/{NNN:03d}.png`` (input-view
      sanity check).  CO3D owns its renders_train so the layout stays flat per
      view -- the generic ``save_canonical_renders_perframe`` writer (skipped
      for co3d in ``run_final``) would otherwise emit a frame-keyed mv-static
      ``view{vv}/`` subdir.

    An evaluator then compares each against the GT mask-composited image for
    PSNR / SSIM / LPIPS.

    Target-view camera source follows ``cfg.processing.camera_poses_source``:

    * ``"gt"``: GT c2w (``{NNN}.npy``) + GT K from ``fovs.npy`` -- world
      Gaussians are in GT-world (input-view c2w was overridden to GT), so
      the test cameras must match that frame.
    * ``"pred"``: MA-predicted (c2w, K) via
      :meth:`Sequence.predicted_camera`. World Gaussians are in MA-world
      (input-view c2w left as MA's prediction); test views from MA stay in
      the same frame. Requires ``recon_on_test_views=true``
      so MA was actually run on those views.

    Guarded by ``cfg.dataset.name == 'co3d'`` at the call site.  Exceptions
    propagate -- a failure here should not silently hide behind a successful
    FINAL block.
    """
    import json as _json

    from PIL import Image

    scene_dir = os.path.join(cfg.dataset.path, cfg.dataset.scene_name)
    with open(os.path.join(scene_dir, "meta.json")) as f:
        meta = _json.load(f)
    views = meta.get("views") or []
    target_indices = [i for i, v in enumerate(views) if v.get("role") == "target"]
    input_indices = [i for i, v in enumerate(views) if v.get("role") == "input"]
    # meta.json labels all 4 views ``input``, but the run may have consumed FEWER
    # (``num_input_views``, e.g. 1 for a single-view run).
    # Subset to the views actually reconstructed from, as ``resolve_run_assets`` slices
    # ``asset_indices[:n_input]`` — else renders_train holds views the model never saw and
    # the input-view metric silently mixes seen with unseen ones.
    n_input = getattr(cfg.dataset, "num_input_views", None)
    if n_input is not None:
        input_indices = input_indices[:n_input]
    if not target_indices and not input_indices:
        print("  WARNING: no labelled views in meta.json; "
              "no CO3D NVS renders exported.")
        return

    use_gt_cameras = cfg.processing.camera_poses_source == "gt"
    cam_source = "GT" if use_gt_cameras else "MA-predicted"

    print("\n" + "-" * 40)
    print(f"Exporting CO3D NVS renders ({cam_source} cameras) ...")
    print("-" * 40)

    H, W = sequence.H, sequence.W
    fovs = (np.load(os.path.join(scene_dir, "fovs.npy")).astype(np.float32)
            if use_gt_cameras else None)

    def _camera(idx):
        """(c2w, K) for asset ``idx`` from the configured source, or None."""
        if use_gt_cameras:
            c2w_path = os.path.join(scene_dir, f"{idx:03d}.npy")
            if not os.path.isfile(c2w_path):
                print(f"  [skip] {idx:03d}: c2w .npy missing")
                return None
            return load_co3d_lara_c2w(c2w_path), co3d_k_from_fov(fovs[idx], H, W)
        cam = sequence.predicted_camera(idx)
        if cam is None:
            print(f"  [skip] {idx:03d}: no MA-predicted camera (set "
                  f"processing.recon_on_test_views=true)")
            return None
        return cam

    # Both splits in ONE render pass so the world scene is composited once, then
    # each asset is written to the dir its meta.json role selects.
    cameras = [(idx, *cam) for idx in [*target_indices, *input_indices]
               if (cam := _camera(idx)) is not None]
    rendered = render_static_eval_views(state, sequence, cameras, H, W)

    def _write_split(indices, sub):
        """Write the already-rendered assets in ``indices`` flat to
        ``{final}/{sub}/{NNN}.png``; returns how many landed."""
        out_dir = os.path.join(final_output_dir, sub)
        os.makedirs(out_dir, exist_ok=True)
        n = 0
        for idx in indices:
            if (rgba := rendered.get(idx)) is not None:
                Image.fromarray(rgba).save(os.path.join(out_dir, f"{idx:03d}.png"))
                n += 1
        return n

    n_test = _write_split(target_indices, "renders_test")
    n_train = _write_split(input_indices, "renders_train")
    print(
        f"  Saved {n_test} test (targets {target_indices}) + {n_train} train "
        f"(inputs {input_indices}) NVS renders at {W}x{H} RGBA (transparent bg), "
        f"{cam_source} cameras"
    )


def _oab_covis_mask_for_frame(
    state, frame_int, frame_objs, input_c2w, test_c2w, K_input, K_test, glctx, H, W
):
    """Per-timestamp OAB co-visibility mask + co-visible fraction.

    OAB pairs one test view with each train timestamp, so co-visibility for the
    test view at ``frame_int`` is computed against the GT mesh deformed to *that*
    timestamp, observed by the **paired input view at the same timestamp**.
    Builds the deformed GT mesh (combined over objects), placed in OAB world
    exactly like the rendered prediction -- per-frame Sim(3) + the input-view c2w
    lift -- then reuses the static co-visibility core.  Returns
    ``(mask (H,W) uint8, covis_fraction)`` or ``(None, None)`` when no per-frame
    GT mesh is keyed for this frame.
    """
    device = torch.device("cuda")
    world_verts, faces_list, voff = [], [], 0
    for obj_idx, _fk, di in frame_objs:
        pf = state.canonical_mesh_per_frame_verts.get(obj_idx)
        faces = state.canonical_mesh_faces.get(obj_idx)
        if not pf or faces is None or frame_int not in pf:
            continue
        verts = pf[frame_int].to(device).float()  # canonical mesh deformed to t
        world = _pose_mesh_verts_to_world(
            verts, di["rotation"].to(device), di["translation"].to(device),
            di["scale"].to(device), input_c2w,
        )
        world_verts.append(world)
        faces_list.append(faces.to(device).to(torch.int32) + voff)
        voff += verts.shape[0]
    if not world_verts:
        return None, None

    verts_world = torch.cat(world_verts, dim=0)
    faces_i32 = torch.cat(faces_list, dim=0).contiguous()
    tau = _COVIS_DEPTH_TOL_FRAC * float(
        (verts_world.amax(0) - verts_world.amin(0)).norm()
    )
    z_in, a_in = _render_gt_mesh_depth(
        verts_world, faces_i32, glctx, input_c2w, K_input, H, W
    )
    z_te, a_te = _render_gt_mesh_depth(
        verts_world, faces_i32, glctx, test_c2w, K_test, H, W
    )
    mask = _covisible_mask_from_depths(
        z_te, a_te, test_c2w, K_test, [(z_in, a_in, input_c2w, K_input)], tau
    )
    fg = int((a_te > 0.5).sum())
    return mask, (float((mask > 0).sum()) / fg if fg else 0.0)


def export_oursactionbench_eval_assets(
    cfg, state, sequence, final_output_dir: str, pipeline_obj
) -> None:
    """Export per-timestamp held-out NVS renders + per-frame predicted
    meshes for an OursActionBench run.

    Writes (for each timestamp ``f``):
      * ``{final}/renders_test/{f:02d}.png`` -- foreground-only RGBA render
        (transparent background; foreground composited on white, coverage in
        the alpha channel) from that frame's held-out hemisphere camera, at the
        train-view resolution (``sequence.H``/``sequence.W``; no fixed 256
        protocol -- matches ``renders_train/``).
      * ``{final}/meshes/{obj:03d}/{f:03d}.ply`` -- per-frame predicted
        mesh in oursactionbench world Y-up (canonical mesh from
        ``final/meshes/{obj:03d}.glb`` warped by the per-frame deformation
        + posed + lifted), colourless (Chamfer/IoU only need geometry).
        Same per-object / per-frame layout as
        ``final/gaussians/{obj:03d}/{f:03d}.ply``; PLY keeps the per-frame
        encoding cost low (no GLB).

    Renders the Gaussians (canonical or per-frame) with gsplat.

    Guarded by ``cfg.dataset.name == 'oursactionbench'`` at the call site.
    Exceptions propagate -- a failure here should not hide behind a
    successful FINAL block.
    """
    import trimesh
    from PIL import Image

    from .gaussian import (
        create_gaussians_object,
        join_gaussians,
        transform_scene_to_r3_convention,
        transform_scene_to_world,
    )
    from genia.core.utils.gt_data import load_actionbench_camera_fit
    from .refinement import apply_pose_to_gaussian
    from .rendering import render_gaussians_to_image
    from genia.core.utils.deformation import (
        _lookup_per_frame_deformation,
        warp_gaussians_high_res,
    )

    print("\n" + "-" * 40)
    print("Exporting OursActionBench per-frame NVS + meshes ...")
    print("-" * 40)

    # Test views render at the train-view resolution (no fixed 256).
    H, W = sequence.H, sequence.W
    bg_white = torch.ones(3)
    dw = cfg.deformation_warp
    warp_kw = dict(
        K=int(dw.knn_k), eps=float(dw.knn_eps),
        chunk_size=int(dw.knn_chunk_size),
    )

    cam = load_actionbench_camera_fit(cfg.dataset.scene_name, cfg.dataset.path)
    per_frame_in = cam.get("per_frame_cameras")
    per_frame_test = cam.get("per_frame_test_cameras")
    json_hw = cam.get("image_hw", [512, 512])
    if not per_frame_in:
        raise RuntimeError(
            f"camera.json for {cfg.dataset.scene_name!r} has no "
            f"per_frame_cameras; cannot lift prediction to world."
        )
    if not per_frame_test:
        raise RuntimeError(
            f"camera.json for {cfg.dataset.scene_name!r} has no "
            f"per_frame_test_cameras. Re-render the scene with "
            f"camera_json_version >= 2."
        )

    # OAB cannot be routed through render_static_eval_views -- that helper
    # composites ONE static scene and reuses it for every camera, whereas OAB
    # rebuilds per timestamp (per-frame warp + per-frame Sim(3) + per-frame lift).
    # Per-frame-only: the frame-0 canonical fallback would freeze every test
    # view to the frame-0 prediction, so source each timestamp's own Gaussian.
    per_frame_only = not state.has_canonical
    perframe_gaussians, canonical_gaussians = {}, {}
    if per_frame_only:
        state.ensure_perframe_gaussians()
        if state.perframe_gaussians is None:
            print("  [skip] per-frame-only run with no decoded per-frame "
                  "Gaussians — skipping OursActionBench eval export.")
            return
        perframe_gaussians = state.perframe_gaussians
    else:
        canonical_gaussians = state.canonical_gaussians_with_fallback

    # Carry the FrameKey so the per-frame Gaussian resolves per timestamp.
    frames: dict[int, list[tuple[int, FrameKey, dict]]] = {}
    for obj_idx, toks in state.tokens_by_object.items():
        if not toks:
            continue
        if not per_frame_only and canonical_gaussians.get(obj_idx) is None:
            continue
        for fk, di in toks:
            if per_frame_only and fk not in perframe_gaussians.get(obj_idx, {}):
                continue
            frame_int = fk.frame if hasattr(fk, "frame") else int(fk)
            frames.setdefault(frame_int, []).append((obj_idx, fk, di))
    if not frames:
        raise RuntimeError(
            "No canonical Gaussians / pose tokens available; "
            "cannot export OursActionBench eval assets."
        )

    renders_dir = os.path.join(final_output_dir, "renders_test")
    meshes_root = os.path.join(final_output_dir, "meshes")
    os.makedirs(renders_dir, exist_ok=True)
    os.makedirs(meshes_root, exist_ok=True)

    # Per-timestamp co-visibility masks (GT-shape-injected runs): each test
    # view's co-visibility vs the GT mesh deformed to its own timestamp,
    # observed by the paired input view -> renders_test/{f:02d}_covis.png.
    gen_covis = (bool(state.gt_canonical_shape_coords)
                 and bool(getattr(state, "canonical_mesh_per_frame_verts", None)))
    covis_glctx = None
    n_covis, covis_fracs = 0, []
    if gen_covis:
        import nvdiffrast.torch as dr
        covis_glctx = dr.RasterizeCudaContext(device=torch.device("cuda"))

    # Per-object per-frame Gaussian PLYs are written by run_final
    # (dataset-agnostic) — deliberately not duplicated here.
    def _select_gaussian(obj_idx, fk):
        return (perframe_gaussians[obj_idx][fk] if per_frame_only
                else canonical_gaussians[obj_idx])

    n_render = n_mesh = 0
    for frame_int in sorted(frames):
        if frame_int >= len(per_frame_in) or frame_int >= len(per_frame_test):
            print(f"  [skip f{frame_int:02d}] no camera entry in camera.json")
            continue
        input_c2w = np.array(per_frame_in[frame_int]["c2w"], dtype=np.float32)
        test_c2w = np.array(per_frame_test[frame_int]["c2w"], dtype=np.float32)
        K_test = _test_intrinsics_scaled(
            per_frame_test[frame_int], json_hw, H, W
        )

        object_gs = []
        for obj_idx, fk, di in frames[frame_int]:
            gs_canon = _select_gaussian(obj_idx, fk)
            device = gs_canon.get_xyz.device
            rot = di["rotation"].to(device)
            trans = di["translation"].to(device)
            sc = di["scale"].to(device)

            resolved = _lookup_per_frame_deformation(
                state.canonical_mesh_verts,
                state.canonical_mesh_per_frame_verts,
                state.canonical_mesh_per_frame_rotations,
                obj_idx, frame_int, device,
                state.canonical_mesh_faces,
            )

            means_override = quats_override = None
            if resolved is not None:
                *_warp_core, _faces = resolved
                with torch.no_grad():
                    means_override, quats_override = warp_gaussians_high_res(
                        gs_canon, *_warp_core,
                        K=warp_kw["K"], eps=warp_kw["eps"],
                        chunk_size=warp_kw["chunk_size"],
                        faces=_faces,
                    )

            xyz, rots, scs, opac, feats = apply_pose_to_gaussian(
                gs_canon, rot, trans, sc,
                means_override=means_override,
                rotation_override=quats_override,
            )
            object_gs.append(create_gaussians_object(
                xyz=xyz, features=feats, scales=scs,
                rots=rots, opacities=opac,
            ))

            # Per-frame predicted mesh PLY (geometry only) ->
            # meshes/{obj:03d}/{frame:03d}.ply.
            if cfg.output.save_output_mesh:
                mesh = _resolve_perframe_mesh_verts(
                    per_frame_only, di, obj_idx, device, meshes_root,
                    resolved, warp_kw, dw, pipeline_obj,
                )
                if mesh is not None:
                    mv, mfaces = mesh
                    with torch.no_grad():
                        mw = _pose_mesh_verts_to_world(
                            mv, rot, trans, sc, input_c2w
                        )
                    obj_mesh_dir = os.path.join(meshes_root, f"{obj_idx:03d}")
                    os.makedirs(obj_mesh_dir, exist_ok=True)
                    trimesh.Trimesh(
                        vertices=mw.detach().cpu().numpy(),
                        faces=mfaces, process=False,
                    ).export(os.path.join(obj_mesh_dir, f"{frame_int:03d}.ply"))
                    n_mesh += 1

        scene_gs = object_gs[0] if len(object_gs) == 1 else join_gaussians(*object_gs)
        scene_gs = transform_scene_to_r3_convention(scene_gs)
        scene_gs = transform_scene_to_world(scene_gs, input_c2w)
        rgb, alpha = render_gaussians_to_image(
            scene_gs, K_test, W, H, bg_color=bg_white, c2w=test_c2w,
            return_alpha=True,
        )
        Image.fromarray(_rgba_uint8(rgb, alpha)).save(
            os.path.join(renders_dir, f"{frame_int:02d}.png")
        )
        n_render += 1

        if gen_covis:
            K_input = _test_intrinsics_scaled(per_frame_in[frame_int], json_hw, H, W)
            cmask, cfrac = _oab_covis_mask_for_frame(
                state, frame_int, frames[frame_int], input_c2w, test_c2w,
                K_input, K_test, covis_glctx, H, W,
            )
            if cmask is not None:
                Image.fromarray(cmask.cpu().numpy()).save(
                    os.path.join(renders_dir, f"{frame_int:02d}_covis.png")
                )
                n_covis += 1
                covis_fracs.append(cfrac)

    print(
        f"  Saved {n_render} NVS renders -> renders_test/ and "
        f"{n_mesh} per-frame meshes -> meshes/{{obj:03d}}/{{frame:03d}}.ply "
        f"({W}x{H}, {len(frames)} timestamps)"
    )
    if n_covis:
        print(
            f"  Saved {n_covis} per-timestamp co-visibility masks -> "
            f"renders_test/{{f:02d}}_covis.png "
            f"(mean co-visible foreground {float(np.mean(covis_fracs)):.0%})"
        )
    if n_mesh == 0:
        print(
            "  WARNING: no per-frame meshes (final/meshes/ absent -- set "
            "output.save_output_mesh=true for 3D eval); 2D NVS still exported."
        )


# =====================================================================
# Synthesized held-out NVS (datasets with no GT test views, e.g. davis_actionmesh)
# =====================================================================

#: The four synthesized viewpoints, as ``(tag, azimuth_sign, elevation_sign)``.
#: Signs scale ``output.synth_nvs_{azimuth,elevation}``, so the set is a
#: horizontal pair and a vertical pair around each train camera.  The tag is the
#: filename suffix (``{frame:03d}_{tag}.png``) BOTH the pipeline export and the
#: ``synth_nvs`` renderer of ``genia.core.utils.render_final_results``
#: write, and what an evaluator globs for the consistency diagnostic — one
#: table, so the three cannot drift apart.
SYNTH_NVS_OFFSETS = (
    ("azpos", +1.0, 0.0),
    ("azneg", -1.0, 0.0),
    ("elpos", 0.0, +1.0),
    ("elneg", 0.0, -1.0),
)


def _synth_orbit_c2w(input_c2w, centroid, azimuth_deg, elevation_deg, up=None,
                     radius=None):
    """Synthesize an R3/OpenCV camera-to-world for a novel view orbiting the
    scene ``centroid`` at the *same radius* the training camera observes it.

    The training camera's own up/right axes define the orbit frame — no world
    gravity direction is assumed (map-anything's world has no known up).  The
    eye is the training eye rotated ``azimuth_deg`` about the camera up-axis and
    ``elevation_deg`` about the camera right-axis, both around ``centroid``; the
    new camera looks back at ``centroid`` with the training up kept roughly
    vertical.  At ``(0, 0)`` the eye is the training eye exactly and the object
    is framed at the same scale — the radius is self-calibrated from the train
    camera, not a scene-scale guess.  The ROTATION is not the training one even
    there: it is rebuilt to aim at ``centroid``, which the training camera does
    only when the object happens to sit on its optical axis.

    ``radius`` overrides the self-calibrated ``|eye0 - centroid|`` with an
    explicit world-unit distance, keeping the orientation logic unchanged — for
    a caller whose training camera sits too close to (or inside) the geometry
    it is framing for its own distance to serve as an orbit radius (see
    ``render_orbit``'s radius fallback).

    ``up`` replaces ``cam_up`` as both the azimuth rotation axis and the output
    camera's up, for a caller that knows the world vertical; a degenerate
    override (parallel to the view ray) falls back to ``cam_up``.  ``None`` — the
    default, and what ``export_synth_nvs_assets`` passes, since it follows each
    train camera rather than orbiting one — keeps the camera's own up.  A
    turntable needs the override: spinning about an elevated camera's own up
    walks the eye below the object for half the turn.
    """
    R = np.asarray(input_c2w[:3, :3], dtype=np.float64)
    eye0 = np.asarray(input_c2w[:3, 3], dtype=np.float64)
    c = np.asarray(centroid, dtype=np.float64)
    cam_up = -R[:, 1]      # OpenCV +Y is down → camera up is -Y
    cam_right = R[:, 0]
    if up is not None:
        up = np.asarray(up, dtype=np.float64)
        up = up / (np.linalg.norm(up) + 1e-12)
        # Keep the hemisphere the training camera implies, so an override given
        # as a bare axis never flips the object upside down.
        up = up if up @ cam_up >= 0 else -up
        fwd = (c - eye0) / (np.linalg.norm(c - eye0) + 1e-12)
        right = np.cross(fwd, up)                 # OpenCV x = z × up, as below
        if np.linalg.norm(right) > 1e-6:          # `up` is not the view ray itself
            cam_up, cam_right = up, right / np.linalg.norm(right)

    def _rodrigues(v, k, deg):
        k = k / (np.linalg.norm(k) + 1e-12)
        th = np.deg2rad(deg)
        return (v * np.cos(th)
                + np.cross(k, v) * np.sin(th)
                + k * (k @ v) * (1.0 - np.cos(th)))

    v = eye0 - c                              # object → training eye (radius = |v|)
    if radius is not None:
        v = v / (np.linalg.norm(v) + 1e-12) * radius
    v = _rodrigues(v, cam_up, azimuth_deg)
    v = _rodrigues(v, cam_right, elevation_deg)
    eye = c + v

    z = c - eye
    z /= (np.linalg.norm(z) + 1e-12)          # forward (+Z, toward centroid)
    x = np.cross(z, cam_up)                    # right (+X); OpenCV x×y=z handedness
    if np.linalg.norm(x) < 1e-6:              # up ∥ forward → pick any right
        x = np.cross(z, cam_right)
    x /= (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)                         # down (+Y)

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = x
    c2w[:3, 1] = y
    c2w[:3, 2] = z
    c2w[:3, 3] = eye
    return c2w


def _synth_frame_groups(tokens_by_object, per_frame_only,
                        canonical_gaussians, perframe_gaussians):
    """``{frame_int: [(obj_idx, FrameKey, decoder_input), ...]}`` for the export.

    ONE entry per (timestamp, object).  The grouping keys on ``FrameKey.frame``,
    so on an MV-static dataset (mvcustom — the asset axis is VIEWS, every one at
    frame 0) all V views would otherwise land in the same bucket and the same
    object would be composed V times, each posed into a different view's camera,
    smearing V copies through one render.  The first entry wins, which is view 0
    — the same one the caller takes the camera from.

    Objects with no Gaussian to render are dropped: no canonical one on a
    canonical run, or no Gaussian keyed at that timestamp on a per-frame one.
    """
    frames: dict[int, list[tuple[int, "FrameKey", dict]]] = {}
    seen: set[tuple[int, int]] = set()
    for obj_idx, toks in tokens_by_object.items():
        if not toks:
            continue
        if not per_frame_only and canonical_gaussians.get(obj_idx) is None:
            continue
        for fk, di in toks:
            if per_frame_only and fk not in perframe_gaussians.get(obj_idx, {}):
                continue
            frame_int = fk.frame if hasattr(fk, "frame") else int(fk)
            if (frame_int, obj_idx) in seen:
                continue
            seen.add((frame_int, obj_idx))
            frames.setdefault(frame_int, []).append((obj_idx, fk, di))
    return frames


def export_synth_nvs_assets(
    cfg, state, sequence, final_output_dir: str, pipeline_obj
) -> None:
    """Render FOUR synthesized held-out novel views per timestamp for datasets
    that ship no GT test views (e.g. davis_actionmesh).

    For each timestamp ``f`` the foreground (per-frame deformation warp, when a
    per-frame mesh field is loaded, + per-frame Sim(3), lifted to world via that
    frame's train c2w — the exact OAB path) is rendered from four cameras orbiting
    the object centroid at the training-camera radius, reusing the train
    intrinsics/resolution.  The four are the signed pairs of
    :data:`SYNTH_NVS_OFFSETS`, scaled by ``output.synth_nvs_azimuth`` (horizontal
    magnitude) and ``output.synth_nvs_elevation`` (vertical), with
    ``f*synth_nvs_azimuth_step`` added to the azimuth of each.  Writes
    ``{final}/renders_synth_nvs/{f:03d}_{tag}.png`` (foreground RGBA, transparent
    bg — RGB composited on white, coverage in the alpha channel).

    Four rather than one because these renders also feed a **view-consistency
    diagnostic**: an evaluator scores CLIP-I between each of them and the train
    image they were generated from.  That is NOT an NVS reconstruction
    metric — both images derive from the same input, so it cannot show
    correctness, and it is maximised by a billboard-flat reconstruction whose
    appearance never changes with viewpoint.  What it measures is whether the
    appearance survives a viewpoint change, and four spread directions measure it
    far better than one.

    Gated by ``config.synth_nvs_enabled(cfg)`` at the call site — ON by default for
    every dataset with no GT held-out split, which is the case this exists for.  Written to a
    dedicated folder (not ``renders_test/``) so it is never mistaken for scored
    held-out views.
    """
    from PIL import Image

    from .gaussian import (
        create_gaussians_object,
        join_gaussians,
        transform_scene_to_r3_convention,
        transform_scene_to_world,
    )
    from .refinement import apply_pose_to_gaussian
    from .rendering import render_gaussians_to_image
    from genia.core.utils.deformation import _lookup_per_frame_deformation, warp_gaussians_high_res

    print("\n" + "-" * 40)
    print("Exporting synthesized per-timestamp NVS ...")
    print("-" * 40)

    # Renders at the train-view resolution (same K/H/W as renders_train).
    H, W = sequence.H, sequence.W
    bg_white = torch.ones(3)
    dw = cfg.deformation_warp
    warp_kw = dict(
        K=int(dw.knn_k), eps=float(dw.knn_eps),
        chunk_size=int(dw.knn_chunk_size),
    )
    az0 = float(cfg.output.synth_nvs_azimuth)
    az_step = float(cfg.output.synth_nvs_azimuth_step)
    el = float(cfg.output.synth_nvs_elevation)

    # Per-frame-only: the frame-0 canonical fallback would freeze every view to
    # the frame-0 prediction, so source each timestamp's own Gaussian.
    per_frame_only = not state.has_canonical
    if per_frame_only:
        state.ensure_perframe_gaussians()
        if state.perframe_gaussians is None:
            print("  [skip] per-frame-only run with no decoded per-frame "
                  "Gaussians — skipping synthesized NVS export.")
            return
        perframe_gaussians = state.perframe_gaussians
        canonical_gaussians = {}
    else:
        perframe_gaussians = {}
        canonical_gaussians = state.canonical_gaussians_with_fallback

    frames = _synth_frame_groups(
        state.tokens_by_object, per_frame_only,
        canonical_gaussians, perframe_gaussians,
    )
    if not frames:
        print("  [skip] no posed foreground Gaussians — nothing to render.")
        return

    out_dir = os.path.join(final_output_dir, "renders_synth_nvs")
    os.makedirs(out_dir, exist_ok=True)

    def _select_gaussian(obj_idx, fk):
        return (perframe_gaussians[obj_idx][fk] if per_frame_only
                else canonical_gaussians[obj_idx])

    n_render = 0
    for frame_int in sorted(frames):
        # This timestamp's train camera (map-anything predicted c2w + K); all
        # objects at a mono timestamp share it, matching the OAB per-frame lift.
        fk_ref = frames[frame_int][0][1]
        input_c2w = np.asarray(sequence[fk_ref].c2w, dtype=np.float32)
        K = np.asarray(sequence[fk_ref].K_matrix, dtype=np.float32)

        # Per-object warp+pose mirrors export_oursactionbench_eval_assets; kept
        # separate rather than shared because that loop interleaves per-frame
        # mesh/covis export this qualitative render has no need for.
        object_gs = []
        for obj_idx, fk, di in frames[frame_int]:
            gs_canon = _select_gaussian(obj_idx, fk)
            device = gs_canon.get_xyz.device
            rot = di["rotation"].to(device)
            trans = di["translation"].to(device)
            sc = di["scale"].to(device)

            resolved = _lookup_per_frame_deformation(
                state.canonical_mesh_verts,
                state.canonical_mesh_per_frame_verts,
                state.canonical_mesh_per_frame_rotations,
                obj_idx, frame_int, device,
                state.canonical_mesh_faces,
            )
            means_override = quats_override = None
            if resolved is not None:
                *_warp_core, _faces = resolved
                with torch.no_grad():
                    means_override, quats_override = warp_gaussians_high_res(
                        gs_canon, *_warp_core,
                        K=warp_kw["K"], eps=warp_kw["eps"],
                        chunk_size=warp_kw["chunk_size"],
                        faces=_faces,
                    )

            xyz, rots, scs, opac, feats = apply_pose_to_gaussian(
                gs_canon, rot, trans, sc,
                means_override=means_override,
                rotation_override=quats_override,
            )
            object_gs.append(create_gaussians_object(
                xyz=xyz, features=feats, scales=scs,
                rots=rots, opacities=opac,
            ))

        scene_gs = object_gs[0] if len(object_gs) == 1 else join_gaussians(*object_gs)
        scene_gs = transform_scene_to_r3_convention(scene_gs)
        scene_gs = transform_scene_to_world(scene_gs, input_c2w)

        # Orbit the object centroid (median world Gaussian mean — outlier-robust)
        # at the train-camera radius; azimuth may drift per frame for a slow orbit.
        centroid = torch.median(
            scene_gs.get_xyz.detach(), dim=0
        ).values.cpu().numpy()
        for tag, az_sign, el_sign in SYNTH_NVS_OFFSETS:
            test_c2w = _synth_orbit_c2w(
                input_c2w, centroid,
                az_sign * az0 + az_step * frame_int, el_sign * el,
            )
            rgb, alpha = render_gaussians_to_image(
                scene_gs, K, W, H, bg_color=bg_white, c2w=test_c2w,
                return_alpha=True,
            )
            Image.fromarray(_rgba_uint8(rgb, alpha)).save(
                os.path.join(out_dir, f"{frame_int:03d}_{tag}.png")
            )
            n_render += 1

    print(
        f"  Saved {n_render} synthesized NVS renders → renders_synth_nvs/ "
        f"({len(frames)} timestamps x {len(SYNTH_NVS_OFFSETS)} views, {W}x{H}, "
        f"az=±{az0:g}+{az_step:g}/frame, el=±{el:g}; no GT — "
        f"view-consistency diagnostic only)"
    )
