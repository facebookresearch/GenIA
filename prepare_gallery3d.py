#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Stage one run's 3D results for the main page's 3D gallery (#gallery3d-sec).

    python prepare_gallery3d.py <run dir> <scene>

copies, from a pipeline run's output (results/.../<scene>/<timestamp>/),

    final/gaussians/world.compressed.ply  -> assets/gallery3d/<scene>/world.compressed.ply
    preprocessing/colmap/sparse/000/*.bin -> assets/gallery3d/<scene>/colmap/sparse/000/
    preprocessing/colmap/images/*         -> assets/gallery3d/<scene>/colmap/images/

writes the gallery's preview of the scene, one thumbnail per input view, as
assets/gallery3d/<scene>/views/<image>.webp (drawn like the main page's
"Input views" tiles, and encoded like their pools), and prints the scene's
entry: add it to GALLERY3D in collect_assets.py and, unless you regenerate
data.js, to its "gallery3d" list too (the line is valid Python and JSON).

A dynamic run (final/gaussians/world/<frame>.compressed.ply, one per frame)
ports only its first, middle and last frame, as one temporal item:

    final/gaussians/world/<frame>.compressed.ply -> assets/gallery3d/<scene>/world/<frame>.compressed.ply
    final/tracks_3d.npz                          -> assets/gallery3d/<scene>/tracks_3d.npz

with the thumbnails of those three frames. The COLMAP model is frame 0's
(sparse/000; the viewer shows one), with only the images it names.

The entry also carries the viewer's starting viewpoint, `view`: looking at
the centre of the splat from behind and a little above the first input camera,
so the scene opens close to the first input photo, with that camera's frustum
in the foreground (3DView's `setView`, in the COLMAP frame).

The gallery hides the COLMAP points (only the cameras are shown), but 3DView
still sizes the frustums from them (and frames the view, for an entry without
`view`), and they are most of the download. So points3D.bin is cropped to the
box spanning every camera centre and the splat (a 1-99th percentile box of its
chunk centres, padded by 5%), keeping every STRIDE-th point: the framing then
covers the cameras and the objects, not the whole room.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData

HERE = Path(__file__).resolve().parent
STRIDE = 20
DYN_FRAMES = 3  # a dynamic run's ported frames: first, middle, last
THUMB_SIZE = 320  # long side, px; as collect_assets.py's input-view pools
# Starting viewpoint, relative to the first input camera: BACK times its distance
# to the objects' centre, raised by RISE times that distance along its image up.
BACK, RISE = 1.4, 0.3


def cameras(images_bin: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Image names, camera centres C = -R^T t (N, 3) and world-to-camera
    rotations R (N, 3, 3) from a COLMAP images.bin, in file order."""
    b = images_bin.read_bytes()
    (n,) = struct.unpack_from("<Q", b, 0)
    off, names, centres, rots = 8, [], [], []
    for _ in range(n):
        _, qw, qx, qy, qz, tx, ty, tz, _ = struct.unpack_from("<I7dI", b, off)
        off += 4 + 56 + 4
        end = b.index(b"\0", off)
        names.append(b[off:end].decode())
        off = end + 1
        (n2d,) = struct.unpack_from("<Q", b, off)
        off += 8 + n2d * 24  # (x, y, point3D_id) per 2D point
        w, x, y, z = np.array([qw, qx, qy, qz]) / np.linalg.norm([qw, qx, qy, qz])
        r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        centres.append(-r.T @ np.array([tx, ty, tz]))
        rots.append(r)
    return names, np.array(centres), np.array(rots)


def home_view(centre: np.ndarray, r: np.ndarray, target: np.ndarray) -> dict:
    """The viewer's starting pose: behind and above camera (centre, r), at target."""
    back = centre - target
    up = -r[1]  # the camera's -y axis in the world: image up (COLMAP is +y down)
    position = target + BACK * back + RISE * np.linalg.norm(back) * up
    return {"position": [round(float(v), 4) for v in position],
            "target": [round(float(v), 4) for v in target]}


def splat_box(plys: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Robust bounds of compressed PLYs, from their per-chunk min/max."""
    chunks = [PlyData.read(str(ply))["chunk"].data for ply in plys]
    lo = np.concatenate([np.stack([ch["min_x"], ch["min_y"], ch["min_z"]], 1) for ch in chunks])
    hi = np.concatenate([np.stack([ch["max_x"], ch["max_y"], ch["max_z"]], 1) for ch in chunks])
    centres = (lo + hi) / 2
    lo, hi = np.percentile(centres, 1, 0), np.percentile(centres, 99, 0)
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


def crop_points(src: Path, dst: Path, lo: np.ndarray, hi: np.ndarray) -> tuple[int, int]:
    """Write the points of COLMAP points3D.bin `src` inside [lo, hi], every STRIDE-th."""
    b = src.read_bytes()
    (n,) = struct.unpack_from("<Q", b, 0)
    off, inside = 8, []
    for _ in range(n):
        start = off
        _, x, y, z = struct.unpack_from("<Qddd", b, off)
        off += 8 + 24 + 3 + 8  # id, xyz, rgb, error
        (track,) = struct.unpack_from("<Q", b, off)
        off += 8 + 8 * track
        if lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1] and lo[2] <= z <= hi[2]:
            inside.append(b[start:off])
    assert off == len(b), f"{src}: trailing bytes"
    kept = inside[::STRIDE]
    dst.write_bytes(struct.pack("<Q", len(kept)) + b"".join(kept))
    return n, len(kept)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path, help="run dir (holds final/ and preprocessing/)")
    ap.add_argument("scene", help="name of the folder under assets/gallery3d/")
    args = ap.parse_args()

    colmap = args.run / "preprocessing" / "colmap"
    gaussians = args.run / "final" / "gaussians"
    out = HERE / "assets" / "gallery3d" / args.scene
    base = f"assets/gallery3d/{args.scene}"
    model = out / "colmap" / "sparse" / "000"
    model.mkdir(parents=True, exist_ok=True)
    (out / "colmap" / "images").mkdir(exist_ok=True)
    (out / "views").mkdir(exist_ok=True)

    per_frame = sorted((gaussians / "world").glob("*.compressed.ply"))
    if per_frame:  # dynamic: first, middle and last frame
        picks = sorted({round(i * (len(per_frame) - 1) / (DYN_FRAMES - 1)) for i in range(DYN_FRAMES)})
        splats = [per_frame[i] for i in picks]
        (out / "world").mkdir(exist_ok=True)
        assets = [f"world/{s.name}" for s in splats]
        # The thumbnails: each picked frame's input views, from its own model.
        thumbs = [n for s in splats
                  for n in sorted(cameras(colmap / "sparse" / s.name.split(".")[0] / "images.bin")[0])]
    else:
        splats = [gaussians / "world.compressed.ply"]
        assets = [splats[0].name]
        thumbs = None  # every input view
    for splat, rel in zip(splats, assets):
        shutil.copyfile(splat, out / rel)
    tracks = args.run / "final" / "tracks_3d.npz"
    if per_frame and tracks.is_file():
        shutil.copyfile(tracks, out / tracks.name)

    for f in ("cameras.bin", "images.bin"):
        shutil.copyfile(colmap / "sparse" / "000" / f, model / f)
    names, cams, rots = cameras(colmap / "sparse" / "000" / "images.bin")
    for name in names:
        shutil.copyfile(colmap / "images" / name, out / "colmap" / "images" / name)
    views = []
    for name in thumbs or sorted(names):
        img = colmap / "images" / name
        thumb = Image.open(img).convert("RGB")
        thumb.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
        thumb.save(out / "views" / f"{img.stem}.webp", quality=80, method=6)
        views.append(f"{base}/views/{img.stem}.webp")

    first = names.index(min(names))  # the first input view, as the tile's first thumbnail
    lo, hi = splat_box(splats)
    view = home_view(cams[first], rots[first], (lo + hi) / 2)
    lo, hi = np.minimum(lo, cams.min(0)), np.maximum(hi, cams.max(0))
    total, kept = crop_points(colmap / "sparse" / "000" / "points3D.bin", model / "points3D.bin", lo, hi)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"{args.scene}: {len(cams)} camera(s), points {total} -> {kept}, {size / 1e6:.1f} MB in {out}")
    entry = {"title": args.scene, "assets": [f"{base}/{a}" for a in assets]}
    if per_frame:
        entry["temporal"] = True
        if tracks.is_file():
            entry["tracks"] = f"{base}/{tracks.name}"
    entry.update({"colmap": f"{base}/colmap", "views": views, "view": view})
    print(json.dumps(entry))


if __name__ == "__main__":
    main()
