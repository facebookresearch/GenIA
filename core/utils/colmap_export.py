"""COLMAP sparse-reconstruction export.

Two exporters share the writer core (:func:`write_colmap_model` + the
bin/text writers):

* :func:`export_colmap_reconstruction` — FINAL block: train + held-out test
  cameras (with rendered images) + a point cloud from the reconstructed
  Gaussian means/SH-DC colours.
* :func:`export_preprocessing_colmap` — PREPROCESSING stage (before any
  inference, so no Gaussians): train cameras with the INPUT RGB images +
  held-out test cameras (pose-only) + a full-scene point cloud unprojected
  from the INPUT depth and coloured by RGB.

The FINAL exporter writes a portable, self-contained COLMAP project to
``{final}/colmap/``::

    images/                 # flat, globally-unique basenames, copied from
        train_*.png         #   renders_train/  (train_v{vv:02d}_{fr:03d}.png when MV)
        test_*.png          #   renders_test/   (held-out NVS views)
    sparse/
        000/                # one model per temporal frame (FrameKey.frame)
            cameras.bin  + cameras.txt
            images.bin   + images.txt
            points3D.bin + points3D.txt
        001/
        ...

so it opens directly in the COLMAP GUI / pycolmap / a 3DGS trainer with
``image_path = {final}/colmap/images``.

A COLMAP model has no time axis, so the temporal dimension is split across
**one model per frame** (``sparse/{frame:03d}``) rather than collapsed into a
single static model.  Static scenes (GSO, frame 0 only) yield a single
``sparse/000``; dynamic scenes (OAB) yield one model per timestamp.

Contents (per frame):
* **That frame's train + held-out test cameras.**  The pipeline's R3 convention
  (X-right, Y-down, Z-forward) IS the OpenCV/COLMAP camera convention, so a
  camera's world->cam is simply ``w2c = inv(c2w)``; ``qvec = rotmat2qvec(w2c R)``
  (COLMAP wxyz), ``tvec = w2c t``.  Test cameras come from the shared
  derivation helpers in :mod:`genia.core.utils.eval_assets_export` (same primitives
  the test-view renderer uses, so cameras match the rendered ``renders_test/``).
  Per-frame test cameras (OAB) go to their own frame; globally-static test
  cameras (GSO/CO3D) are replicated into every frame's model.
* **A colored point cloud** for that frame, from the world-space foreground
  Gaussian means (``get_xyz``) tinted by their DC spherical-harmonic colour
  (:func:`genia.core.utils.gaussian.SH2RGB`).  Dynamic scenes (OAB, actionmesh)
  place the per-frame geometry — the non-rigid deformation warp from
  ``state.canonical_mesh_*`` followed by that frame's Sim(3), the same path
  ``renders_train/`` and ``gaussians/{obj:03d}/{frame:03d}.ply`` take; static
  scenes (GSO/CO3D) reuse the exact static geometry across their single (or
  replicated) frame model(s).

The binary formats mirror COLMAP's ``read_write_model.py`` (verified
field-by-field): ``cameras.bin`` ``<iiQQ`` + ``<d``*params, ``images.bin``
``<i``/``<dddd``/``<ddd``/``<i``/name+``\\0``/``<Q``, ``points3D.bin``
``<Q``/``<ddd``/``<BBB``/``<d``/``<Q``.  No pycolmap dependency.
"""

from __future__ import annotations

import os
import shutil
import struct

import numpy as np

# COLMAP camera model id for PINHOLE (fx, fy, cx, cy); see CAMERA_MODEL_IDS.
_PINHOLE_MODEL_ID = 1


# ---------------------------------------------------------------------------
# Rotation -> COLMAP quaternion (wxyz), and c2w -> (qvec, tvec) world->cam.
# ---------------------------------------------------------------------------

def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> COLMAP quaternion ``(qw, qx, qy, qz)``.

    Eigen-decomposition form from COLMAP's ``read_write_model.py`` (with the
    ``qw >= 0`` sign normalisation).  Assumes ``R`` is orthonormal.
    """
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
    ]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec = -qvec
    return qvec


def _c2w_to_qvec_tvec(c2w: np.ndarray):
    """Camera-to-world (R3/OpenCV) -> COLMAP world->cam ``(qvec wxyz, tvec)``."""
    w2c = np.linalg.inv(np.asarray(c2w, dtype=np.float64))
    return rotmat2qvec(w2c[:3, :3]), w2c[:3, 3]


# ---------------------------------------------------------------------------
# COLMAP binary writers (little-endian struct, mirroring read_write_model.py).
# ---------------------------------------------------------------------------

def _write_cameras_bin(path, cameras):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for c in cameras:
            f.write(struct.pack("<iiQQ", c["id"], _PINHOLE_MODEL_ID, c["w"], c["h"]))
            f.write(struct.pack("<dddd", *c["params"]))  # fx, fy, cx, cy


def _write_images_bin(path, images):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for im in images:
            f.write(struct.pack("<i", im["id"]))
            f.write(struct.pack("<dddd", *[float(v) for v in im["qvec"]]))
            f.write(struct.pack("<ddd", *[float(v) for v in im["tvec"]]))
            f.write(struct.pack("<i", im["cam_id"]))
            f.write(im["name"].encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", 0))  # num_points2D (no 2D observations)


def _write_points3d_bin(path, xyz, rgb):
    buf = bytearray()
    buf += struct.pack("<Q", int(xyz.shape[0]))
    for i in range(xyz.shape[0]):
        buf += struct.pack("<Q", i + 1)                       # point id (1-based)
        buf += struct.pack("<ddd", float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2]))
        buf += struct.pack("<BBB", int(rgb[i, 0]), int(rgb[i, 1]), int(rgb[i, 2]))
        buf += struct.pack("<d", 0.0)                         # reprojection error
        buf += struct.pack("<Q", 0)                           # empty track
    with open(path, "wb") as f:
        f.write(buf)


# ---------------------------------------------------------------------------
# COLMAP text writers (human-readable; readers prefer .bin when both present).
# ---------------------------------------------------------------------------

def _write_cameras_txt(path, cameras):
    with open(path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras)}\n")
        for c in cameras:
            params = " ".join(repr(float(p)) for p in c["params"])
            f.write(f"{c['id']} PINHOLE {c['w']} {c['h']} {params}\n")


def _write_images_txt(path, images):
    with open(path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(images)}, mean observations per image: 0\n")
        for im in images:
            q = " ".join(repr(float(v)) for v in im["qvec"])
            t = " ".join(repr(float(v)) for v in im["tvec"])
            f.write(f"{im['id']} {q} {t} {im['cam_id']} {im['name']}\n")
            f.write("\n")  # mandatory (empty) POINTS2D line


def _write_points3d_txt(path, xyz, rgb):
    with open(path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {int(xyz.shape[0])}, mean track length: 0\n")
        for i in range(xyz.shape[0]):
            f.write(
                f"{i + 1} {xyz[i, 0]!r} {xyz[i, 1]!r} {xyz[i, 2]!r} "
                f"{int(rgb[i, 0])} {int(rgb[i, 1])} {int(rgb[i, 2])} 0\n"
            )


# ---------------------------------------------------------------------------
# Camera + point-cloud assembly.
# ---------------------------------------------------------------------------

def _train_image_name(fk, is_mv: bool) -> str:
    """Flat, globally-unique COLMAP basename for an input/train view — shared by
    the FINAL and PREPROCESSING colmap exporters so the naming can't drift."""
    fr = fk.frame if hasattr(fk, "frame") else int(fk)
    vw = fk.view if hasattr(fk, "view") else 0
    return f"train_v{vw:02d}_{fr:03d}.png" if is_mv else f"train_{fr:03d}.png"


def _train_cameras(cfg, sequence):
    """``[(K, c2w, src_render_path, colmap_name)]`` for every input/train view."""
    name = str(cfg.dataset.name).lower()
    if name == "oursactionbench":
        # FrameData.c2w is identity for OAB (camera baked into the pose); the
        # real per-frame input cameras live in camera.json.
        from .eval_assets_export import oursactionbench_input_cameras
        return oursactionbench_input_cameras(cfg, sequence)

    is_mv = sequence.is_mv
    out = []
    for fk in sequence.frame_keys:
        fd = sequence[fk]
        K = np.asarray(fd.K_matrix, dtype=np.float64)
        c2w = np.asarray(fd.c2w, dtype=np.float64)
        fr = fk.frame if hasattr(fk, "frame") else int(fk)
        vw = fk.view if hasattr(fk, "view") else 0
        src = (f"renders_train/view{vw:02d}/{fr:03d}.png" if is_mv
               else f"renders_train/{fr:03d}.png")
        out.append((K, c2w, src, _train_image_name(fk, is_mv)))
    return out


def _test_cameras(cfg, sequence):
    """``[(K, c2w, src_render_path, colmap_name)]`` for held-out test views
    (empty for datasets without held-out cameras)."""
    name = str(cfg.dataset.name).lower()
    from . import eval_assets_export as eae
    if name == "gso":
        return eae.gso_test_cameras(cfg, sequence)
    if name == "oursactionbench":
        return eae.oursactionbench_test_cameras(cfg, sequence)
    if name == "co3d":
        return eae.co3d_test_cameras(cfg, sequence)
    return []


def _frame_of_name(name: str) -> int:
    """Temporal frame index encoded in a flat COLMAP basename.

    ``train_{fr:03d}.png`` / ``train_v{vv:02d}_{fr:03d}.png`` / ``test_{f:02d}.png``
    all carry the frame as the trailing integer (the names are *built* from it),
    so the last ``_``-delimited field is the frame.
    """
    return int(name[:-len(".png")].split("_")[-1])


def train_records(cfg, sequence):
    """``[(frame_int, K, c2w, src_render_path, colmap_name)]`` for every
    input/train view, tagged with its temporal ``FrameKey.frame``.

    Public because it is the one place that knows where a dataset's real input
    cameras come from — notably that OursActionBench's live in ``camera.json``
    rather than its identity ``FrameData.c2w`` — so consumers outside the COLMAP
    export share the rule instead of restating it."""
    return [(_frame_of_name(nm), K, c2w, src, nm)
            for (K, c2w, src, nm) in _train_cameras(cfg, sequence)]


def _test_records(cfg, sequence):
    """``[(frame_or_None, K, c2w, src_render_path, colmap_name)]`` for held-out
    test views.  OAB test cameras are per-timestamp (tagged with their frame);
    GSO/CO3D test cameras are globally-static novel views (``frame=None`` →
    replicated into every per-frame model)."""
    per_frame = str(cfg.dataset.name).lower() == "oursactionbench"
    return [((_frame_of_name(nm) if per_frame else None), K, c2w, src, nm)
            for (K, c2w, src, nm) in _test_cameras(cfg, sequence)]


def _scene_gs_to_point_cloud(scene_gs, max_points=None):
    """World-space Gaussian scene -> ``(xyz (N,3) float64, rgb (N,3) uint8)``."""
    from .gaussian import SH2RGB

    xyz = scene_gs.get_xyz.detach().cpu().numpy().astype(np.float64)
    f_dc = scene_gs._features_dc[:, 0, :].detach().cpu().numpy()  # (N, 3) DC band
    rgb = np.clip(SH2RGB(f_dc), 0.0, 1.0)
    rgb = (rgb * 255.0).round().astype(np.uint8)
    if max_points is not None and xyz.shape[0] > int(max_points):
        rng = np.random.default_rng(0)  # deterministic subsample
        sel = rng.choice(xyz.shape[0], int(max_points), replace=False)
        xyz, rgb = xyz[sel], rgb[sel]
    return xyz, rgb


def _oursactionbench_world_scene(cfg, state, frame):
    """Build the OAB world-space foreground Gaussian scene for ``frame``.

    Mirrors the per-frame world-lift in
    ``eval_assets_export.export_oursactionbench_eval_assets`` (per-frame
    deformation warp -> Sim(3) -> R3 -> lift via that frame's input c2w), so
    the point cloud shares the OAB world frame with the train/test cameras of
    the same timestamp.  Raises if ``frame`` has no Gaussians / pose tokens.
    """
    import torch

    from .gaussian import (
        create_gaussians_object, join_gaussians,
        transform_scene_to_r3_convention, transform_scene_to_world,
    )
    from genia.core.utils.gt_data import load_actionbench_camera_fit
    from .refinement import apply_pose_to_gaussian
    from genia.core.utils.deformation import _lookup_per_frame_deformation, warp_gaussians_high_res

    cam = load_actionbench_camera_fit(cfg.dataset.scene_name, cfg.dataset.path)
    per_frame_in = cam.get("per_frame_cameras")
    if not per_frame_in:
        raise RuntimeError("OAB camera.json has no per_frame_cameras")

    per_frame_only = not state.has_canonical
    if per_frame_only:
        state.ensure_perframe_gaussians()
        perframe_gaussians = state.perframe_gaussians or {}
        canonical_gaussians = {}
    else:
        perframe_gaussians = {}
        canonical_gaussians = state.canonical_gaussians_with_fallback

    by_frame = {}
    for obj_idx, toks in state.tokens_by_object.items():
        if not toks:
            continue
        if not per_frame_only and canonical_gaussians.get(obj_idx) is None:
            continue
        for fk, di in toks:
            fint = fk.frame if hasattr(fk, "frame") else int(fk)
            if per_frame_only and fk not in perframe_gaussians.get(obj_idx, {}):
                continue
            by_frame.setdefault(fint, []).append((obj_idx, fk, di))
    if not by_frame:
        raise RuntimeError("no OAB Gaussians / pose tokens for the point cloud")

    fint = int(frame)
    if fint not in by_frame:
        raise RuntimeError(f"OAB frame {fint} has no Gaussians / pose tokens")
    if fint >= len(per_frame_in):
        raise RuntimeError(f"OAB frame {fint} has no camera.json entry")
    input_c2w = np.array(per_frame_in[fint]["c2w"], dtype=np.float32)

    dw = cfg.deformation_warp
    warp_kw = dict(K=int(dw.knn_k), eps=float(dw.knn_eps),
                   chunk_size=int(dw.knn_chunk_size))

    object_gs = []
    for obj_idx, fk, di in by_frame[fint]:
        gs_canon = (perframe_gaussians[obj_idx][fk] if per_frame_only
                    else canonical_gaussians[obj_idx])
        device = gs_canon.get_xyz.device
        rot = di["rotation"].to(device)
        trans = di["translation"].to(device)
        sc = di["scale"].to(device)
        resolved = _lookup_per_frame_deformation(
            state.canonical_mesh_verts, state.canonical_mesh_per_frame_verts,
            state.canonical_mesh_per_frame_rotations,
            obj_idx, fint, device, state.canonical_mesh_faces,
        )
        means_override = quats_override = None
        if resolved is not None:
            *_warp_core, _faces = resolved
            with torch.no_grad():
                means_override, quats_override = warp_gaussians_high_res(
                    gs_canon, *_warp_core, K=warp_kw["K"], eps=warp_kw["eps"],
                    chunk_size=warp_kw["chunk_size"], faces=_faces,
                )
        xyz, rots, scs, opac, feats = apply_pose_to_gaussian(
            gs_canon, rot, trans, sc,
            means_override=means_override, rotation_override=quats_override,
        )
        object_gs.append(create_gaussians_object(
            xyz=xyz, features=feats, scales=scs, rots=rots, opacities=opac,
        ))

    scene_gs = object_gs[0] if len(object_gs) == 1 else join_gaussians(*object_gs)
    scene_gs = transform_scene_to_r3_convention(scene_gs)
    scene_gs = transform_scene_to_world(scene_gs, input_c2w)
    return scene_gs


def world_scene_gaussians(cfg, state, sequence, frame):
    """Foreground Gaussians for ``frame``, in the same world frame as that
    timestamp's cameras.  Raises when no Gaussians exist for ``frame``.

    The single definition of "the world scene": this module turns it into the
    COLMAP point cloud, and ``eval_assets_export.export_world_space_assets``
    writes it as ``gaussians/world.ply`` -- so the two cannot drift apart.
    OAB lifts via ``camera.json``'s per-frame input c2w (its ``FrameData.c2w``
    is identity); everything else via the anchor FrameKey's c2w.
    """
    if str(cfg.dataset.name).lower() == "oursactionbench":
        return _oursactionbench_world_scene(cfg, state, frame)
    from .eval_assets_export import _compose_world_space_foreground
    return _compose_world_space_foreground(
        state, sequence, frame=frame, deformation_warp=cfg.deformation_warp,
    )


def _world_point_cloud(cfg, state, sequence, frame, max_points):
    """``(xyz, rgb)`` foreground point cloud for ``frame``, in the same world
    frame as that timestamp's cameras; ``(0,3)`` arrays (cameras-only model) if
    no Gaussians exist.  Static scenes ignore ``frame`` (geometry is exact);
    dynamic ones get that frame's non-rigid deformation warp + Sim(3)."""
    try:
        scene_gs = world_scene_gaussians(cfg, state, sequence, frame)
        return _scene_gs_to_point_cloud(scene_gs, max_points)
    except Exception as e:  # noqa: BLE001 — never fail FINAL over the point cloud
        print(f"  [colmap] no point cloud for frame {frame} ({e}); "
              "writing a cameras-only model")
        return np.zeros((0, 3), np.float64), np.zeros((0, 3), np.uint8)


def write_colmap_model(model_dir, camera_records, width, height, xyz, rgb):
    """Write a COLMAP sparse model (binary + text) from camera records + a point
    cloud.  ``camera_records = [(K (3x3), c2w (4x4), colmap_name)]``; dedups
    cameras by rounded intrinsics.  Does NO image I/O (callers place the PNGs).
    Returns ``(n_cameras, n_images)``.  Shared by the FINAL and PREPROCESSING
    colmap exporters.
    """
    os.makedirs(model_dir, exist_ok=True)
    W, H = int(round(width)), int(round(height))
    cameras, images, cam_id_by_key = [], [], {}
    for (K, c2w, name) in camera_records:
        K = np.asarray(K, dtype=np.float64)
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        key = (round(fx, 4), round(fy, 4), round(cx, 4), round(cy, 4), W, H)
        if key not in cam_id_by_key:
            cam_id_by_key[key] = len(cameras) + 1
            cameras.append({"id": cam_id_by_key[key], "w": W, "h": H,
                            "params": (fx, fy, cx, cy)})
        qvec, tvec = _c2w_to_qvec_tvec(c2w)
        images.append({"id": len(images) + 1, "qvec": qvec, "tvec": tvec,
                       "cam_id": cam_id_by_key[key], "name": name})
    _write_cameras_bin(os.path.join(model_dir, "cameras.bin"), cameras)
    _write_images_bin(os.path.join(model_dir, "images.bin"), images)
    _write_points3d_bin(os.path.join(model_dir, "points3D.bin"), xyz, rgb)
    _write_cameras_txt(os.path.join(model_dir, "cameras.txt"), cameras)
    _write_images_txt(os.path.join(model_dir, "images.txt"), images)
    _write_points3d_txt(os.path.join(model_dir, "points3D.txt"), xyz, rgb)
    return len(cameras), len(images)


def _write_per_frame_models(colmap_dir, train, test, width, height, cloud_fn):
    """Write one ``sparse/{frame:03d}`` model per temporal frame, shared by the
    FINAL and PREPROCESSING exporters.

    ``train`` / ``test`` are ``[(frame_or_None, K, c2w, _src, name)]``; frames
    are taken from ``train``.  Test cameras tagged with a frame go to that
    frame's model; ``frame=None`` test cameras (globally-static GSO/CO3D) are
    replicated into every model.  ``cloud_fn(frame) -> (xyz, rgb)`` builds that
    frame's point cloud.  K/c2w are coerced to float64 by ``write_colmap_model``.
    Returns ``(n_models, total_cameras, total_points)``.
    """
    global_test = [(K, c2w, nm) for (fr, K, c2w, _s, nm) in test if fr is None]
    frames = sorted({fr for (fr, *_rest) in train}) or [0]
    total_cam = total_pts = 0
    for fr in frames:
        recs = (
            [(K, c2w, nm) for (f, K, c2w, _s, nm) in train if f == fr]
            + [(K, c2w, nm) for (f, K, c2w, _s, nm) in test if f == fr]
            + global_test
        )
        xyz, rgb = cloud_fn(fr)
        n_cam, _ = write_colmap_model(
            os.path.join(colmap_dir, "sparse", f"{fr:03d}"),
            recs, width, height, xyz, rgb,
        )
        total_cam += n_cam
        total_pts += int(xyz.shape[0])
    return len(frames), total_cam, total_pts


def export_colmap_reconstruction(
    cfg, state, sequence, final_output_dir: str, pipeline_obj, max_points=None,
) -> None:
    """Write per-frame COLMAP sparse models (binary + text) + image copies under
    ``{final_output_dir}/colmap/`` from the reconstructed Gaussians + cameras
    (one ``sparse/{frame:03d}`` per temporal frame).  See the module docstring."""
    colmap_dir = os.path.join(final_output_dir, "colmap")
    images_dir = os.path.join(colmap_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    print("\n" + "-" * 40)
    print("Exporting COLMAP reconstruction ...")
    print("-" * 40)
    if sequence.is_dynamic:
        print("  [colmap] dynamic scene: one COLMAP model per temporal frame "
              "(sparse/{frame:03d}); each model's point cloud is that frame's "
              "geometry.")

    train = train_records(cfg, sequence)   # [(frame, K, c2w, src, name)]
    test = _test_records(cfg, sequence)      # [(frame_or_None, K, c2w, src, name)]

    # Place images once: copy each render PNG into the flat images/ dir (shared
    # across all per-frame models; pose entries are kept regardless of whether
    # the source render exists).
    n_copied = n_missing = 0
    for (_fr, _K, _c2w, src, name) in train + test:
        dst = os.path.join(images_dir, name)
        if os.path.isfile(dst):
            continue
        src_abs = os.path.join(final_output_dir, src)
        if os.path.isfile(src_abs):
            shutil.copyfile(src_abs, dst)
            n_copied += 1
        else:
            n_missing += 1

    # One model per temporal frame, with that frame's posed Gaussian cloud.
    n_models, total_cam, total_pts = _write_per_frame_models(
        colmap_dir, train, test, sequence.W, sequence.H,
        lambda fr: _world_point_cloud(cfg, state, sequence, fr, max_points),
    )

    print(
        f"  {n_models} COLMAP model(s) -> colmap/sparse/{{frame:03d}} "
        f"({total_cam} cameras, {n_copied} PNGs copied"
        + (f", {n_missing} missing" if n_missing else "")
        + f", {total_pts} points total); image_path = colmap/images"
    )


# ---------------------------------------------------------------------------
# PREPROCESSING colmap: input cameras + a colored point cloud unprojected from
# the input depth (no Gaussians exist yet).
# ---------------------------------------------------------------------------

def _save_input_image(images_dir, name, image):
    """Write a FrameData ``image`` (uint8 RGB, or float [0,1], maybe RGBA) as a
    PNG into ``images_dir/name``."""
    from PIL import Image

    img = np.asarray(image)
    if img.ndim == 3 and img.shape[-1] >= 3:
        img = img[..., :3]
    if not np.issubdtype(img.dtype, np.uint8):
        img = (np.clip(img, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    Image.fromarray(img).save(os.path.join(images_dir, name))


def _load_test_gt_image(cfg, sequence, name):
    """GT held-out test image for a test camera ``name`` (``test_{idx}.png``),
    composited on white and resized to the test-camera resolution
    (``sequence.H``; these datasets render square) so it matches the camera K.
    Returns a uint8 ``(H, W, 3)`` array, or ``None`` if the GT file is absent
    (→ that view stays pose-only).  Same GT layout/protocol as the
    evaluator."""
    ds = str(cfg.dataset.name).lower()
    idx = _frame_of_name(name)  # trailing int: GSO/CO3D view idx, OAB frame
    scene_dir = os.path.join(cfg.dataset.path, cfg.dataset.scene_name)
    size = int(sequence.H)

    if ds in ("gso", "oursactionbench"):
        from .eschernet_metrics import load_gso_gt_image  # RGBA-on-white
        gt = (os.path.join(scene_dir, "render_mvs_25", "model", f"{idx:03d}.png")
              if ds == "gso"
              else os.path.join(scene_dir, "test_imgs", f"{idx:02d}.png"))
        if not os.path.isfile(gt):
            return None
        return load_gso_gt_image(gt, size=size)
    if ds == "co3d":
        # CO3D GT = 3-channel RGB + binary mask sidecar, composited on white
        # (the evaluator's own loader).
        from .eschernet_metrics import _load_co3d_gt_image
        rgb_p = os.path.join(scene_dir, f"{idx:03d}.png")
        mask_p = os.path.join(scene_dir, f"{idx:03d}_mask.png")
        if not (os.path.isfile(rgb_p) and os.path.isfile(mask_p)):
            return None
        return _load_co3d_gt_image(rgb_p, mask_p, size=size)
    return None


def _save_test_images(cfg, sequence, test, images_dir):
    """Save each held-out test view's GT image into ``images_dir`` (so the
    preprocessing COLMAP test cameras carry images, not just poses).  Returns
    ``(n_saved, n_missing)``; views whose GT image is absent stay pose-only."""
    saved = missing = 0
    for (_fr, _K, _c2w, _src, name) in test:
        img = _load_test_gt_image(cfg, sequence, name)
        if img is None:
            missing += 1
            continue
        _save_input_image(images_dir, name, img)  # uint8 (H,W,3) -> PNG
        saved += 1
    return saved, missing


def _depth_world_point_cloud(sequence, c2w_by_key, frames=None, max_points=500_000):
    """Colored point cloud from the INPUT depth: per frame, unproject the
    camera-space ``pointmap`` to world via ``c2w_by_key[fk]`` and colour by the
    input RGB.  ``frames`` (a set of allowed ``FrameKey.frame`` ints) restricts
    the cloud to one temporal frame; ``None`` uses every frame.  Returns
    ``(xyz (N,3) float64, rgb (N,3) uint8)`` — empty arrays if no finite depth
    (→ cameras-only model).
    """
    xyz_list, rgb_list = [], []
    for fk in sequence.frame_keys:
        if fk not in c2w_by_key:
            continue
        if frames is not None and (
            fk.frame if hasattr(fk, "frame") else int(fk)
        ) not in frames:
            continue
        fd = sequence[fk]
        pm = np.asarray(fd.pointmap, dtype=np.float64)          # (H, W, 3) R3 cam-space
        valid = np.isfinite(pm).all(axis=-1)
        vm = getattr(fd, "valid_mask", None)
        if vm is not None:
            valid &= np.asarray(vm, dtype=bool)
        if not valid.any():
            continue
        c2w = np.asarray(c2w_by_key[fk], dtype=np.float64)
        xyz_list.append(pm[valid] @ c2w[:3, :3].T + c2w[:3, 3])  # -> world
        img = np.asarray(fd.image)
        img = img[..., :3] if img.ndim == 3 and img.shape[-1] >= 3 else img
        if not np.issubdtype(img.dtype, np.uint8):
            img = (np.clip(img, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        rgb_list.append(img[valid])
    if not xyz_list:
        return np.zeros((0, 3), np.float64), np.zeros((0, 3), np.uint8)
    xyz = np.concatenate(xyz_list, axis=0)
    rgb = np.concatenate(rgb_list, axis=0).astype(np.uint8)
    if max_points is not None and xyz.shape[0] > int(max_points):
        rng = np.random.default_rng(0)  # deterministic subsample
        sel = rng.choice(xyz.shape[0], int(max_points), replace=False)
        xyz, rgb = xyz[sel], rgb[sel]
    return xyz, rgb


def export_preprocessing_colmap(cfg, sequence, colmap_root, max_points=500_000):
    """Write COLMAP models under ``{colmap_root}/colmap/`` from PREPROCESSING
    inputs — before any inference, so there are no Gaussians.  One model per
    temporal frame (``sparse/{frame:03d}``); each contains:

    * **train cameras** for that frame, each with its input RGB image copied
      into the shared ``images/`` (uses ``sequence[fk]`` K/c2w; for OAB the real
      input cameras come from camera.json since ``FrameData.c2w`` is identity);
    * **held-out test cameras**: per-frame (OAB) for that frame, or
      globally-static (GSO/CO3D) replicated into every frame's model — each with
      its **GT** held-out image copied into ``images/`` (composited on white at
      the camera resolution; pose-only only if that GT image is missing);
    * **point cloud**: that frame's depth unprojected to world + coloured by RGB.

    The cloud and the cameras of a frame share one world frame (the same
    per-frame c2w is used for both the train camera and the depth unprojection).
    """
    name_l = str(cfg.dataset.name).lower()
    is_mv = sequence.is_mv
    colmap_dir = os.path.join(colmap_root, "colmap")
    images_dir = os.path.join(colmap_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    print("\n" + "-" * 40)
    print("PREPROCESSING: COLMAP (input cameras + depth point cloud) ...")
    print("-" * 40)

    # Train records (frame, K, c2w, _src, name) + the per-frame c2w used for
    # BOTH the camera and that frame's depth unprojection (so cloud + cameras
    # share one frame).  ``c2w_by_key`` keys the depth unprojection by FrameKey.
    train, c2w_by_key = [], {}
    if name_l == "oursactionbench":
        # OAB FrameData.c2w is identity (camera baked into the object pose); the
        # real per-frame input cameras live in camera.json.
        from .eval_assets_export import oursactionbench_input_cameras
        oab = {_frame_of_name(nm): (K, c2w, nm)
               for (K, c2w, _src, nm) in oursactionbench_input_cameras(cfg, sequence)}
        for fk in sequence.frame_keys:
            fr = fk.frame if hasattr(fk, "frame") else int(fk)
            if fr not in oab:
                continue
            K, c2w, nm = oab[fr]
            _save_input_image(images_dir, nm, sequence[fk].image)
            train.append((fr, K, c2w, None, nm))
            c2w_by_key[fk] = c2w
    else:
        for fk in sequence.frame_keys:
            fd = sequence[fk]
            fr = fk.frame if hasattr(fk, "frame") else int(fk)
            nm = _train_image_name(fk, is_mv)
            _save_input_image(images_dir, nm, fd.image)
            train.append((fr, fd.K_matrix, fd.c2w, None, nm))
            c2w_by_key[fk] = fd.c2w

    # Held-out test cameras + their GT held-out images (saved into images/).
    test = _test_records(cfg, sequence)  # [(frame_or_None, K, c2w, src, name)]
    n_test_img, n_test_missing = _save_test_images(cfg, sequence, test, images_dir)

    # One model per temporal frame, with that frame's input-depth point cloud.
    def cloud(fr):
        return _depth_world_point_cloud(
            sequence, c2w_by_key, frames={fr}, max_points=max_points,
        )
    n_models, total_cam, total_pts = _write_per_frame_models(
        colmap_dir, train, test, sequence.W, sequence.H, cloud,
    )

    print(
        f"  {n_models} preprocessing COLMAP model(s) -> "
        f"colmap/sparse/{{frame:03d}} ({len(train)} train + {len(test)} test "
        f"cameras [{n_test_img} GT test images"
        + (f", {n_test_missing} pose-only" if n_test_missing else "")
        + f"], {total_cam} total cam entries, {total_pts} depth points total); "
        "image_path = colmap/images"
    )
