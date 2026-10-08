# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Render visualizations from a pipeline ``final/`` directory.

FINAL renders its orbit, world-space and track-overlay views through
:func:`process_run`; :func:`main` is the command-line entry point.

Point this at any run's ``final/`` folder -- ``results/[{experiment}/]{dataset}/
{scene}/{timestamp}/final`` -- and it re-renders the exported assets
(``poses.json`` + ``gaussians/`` + ``meshes/``) into ``final/viz/``.  Nothing
from the pipeline runtime is needed: the run is reconstructed purely from the
files ``FINAL`` wrote, so any run with a ``final/`` folder can be rendered.
Renders are plug-ins registered with :func:`renderer` and selected on the CLI;
run with ``--list`` to see them.
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
import math
import os
import re
import sys
import traceback
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any, Callable, Collection, Iterable, Iterator, NamedTuple

import numpy as np

# Skip sam3d_objects heavyweight init (must be set before importing it).
os.environ.setdefault("LIDRA_SKIP_INIT", "1")

from genia.core.paths import SAM3D_OBJECTS_ROOT  # noqa: E402

# APPENDED, not inserted: FINAL imports this module from inside a live pipeline
# process, where another `sam3d_objects` may already sit earlier on sys.path —
# an appended entry can never outrank it, while still making ours importable for
# the standalone CLI.
if str(SAM3D_OBJECTS_ROOT) not in sys.path:
    sys.path.append(str(SAM3D_OBJECTS_ROOT))

# The repo's frame/view primitives -- `frame_key_stem` in particular owns the
# per-frame output naming convention, so viz filenames track the pipeline's.
from genia.core.utils.frame_key import (  # noqa: E402
    FrameKey,
    frame_key_sort_key,
    frame_key_stem,
    group_by_view,
)


# ---------------------------------------------------------------------------
# The GPU stack (torch + gsplat + the pipeline renderer, ~20s) is deferred until
# after argparse, so --list / --dry-run / contact_sheet never pay for it.
# ---------------------------------------------------------------------------

torch = None
Image = None
render_gaussian_params = None
p3d_to_r3_positions = None
p3d_to_r3_quaternions = None
quaternion_to_matrix = None
quaternion_invert = None
quaternion_multiply = None


def _lazy_imports(gpu: bool = True) -> None:
    """Import PIL (+ torch, the pipeline render and convention helpers on GPU)."""
    global torch, Image
    global render_gaussian_params, bulk_keep_mask, visible_bulk_mask
    global oversize_splat_mask, connected_bulk_mask
    global p3d_to_r3_positions, p3d_to_r3_quaternions
    global quaternion_to_matrix, quaternion_invert, quaternion_multiply

    if Image is None:
        from PIL import Image as _Image
        Image = _Image
    if not gpu or torch is not None:
        return

    import torch as _torch

    from genia.core.utils.quaternion_ops import (
        p3d_to_r3_positions as _p2r_pos,
        p3d_to_r3_quaternions as _p2r_quat,
        quaternion_invert as _qinv,
        quaternion_multiply as _qmul,
        quaternion_to_matrix as _q2m,
    )
    from genia.core.utils.rendering import (
        bulk_keep_mask as _bulk_keep,
        connected_bulk_mask as _connected,
        oversize_splat_mask as _oversize,
        render_gaussian_params as _render,
        visible_bulk_mask as _visible_bulk,
    )

    torch = _torch
    quaternion_invert, quaternion_multiply, quaternion_to_matrix = _qinv, _qmul, _q2m
    p3d_to_r3_positions, p3d_to_r3_quaternions = _p2r_pos, _p2r_quat
    render_gaussian_params = _render
    bulk_keep_mask = _bulk_keep
    visible_bulk_mask = _visible_bulk
    oversize_splat_mask = _oversize
    connected_bulk_mask = _connected


def log(msg: str) -> None:
    print(msg, flush=True)


# ===========================================================================
# Run discovery: results/[{experiment}/]{dataset}/{scene}/{ts}/final
# ===========================================================================

_TIMESTAMP_RE = re.compile(r"^\d{8}_\d{6}$")
_UNKNOWN = "unknown"


@dataclass(frozen=True)
class RunPath:
    """Identity of a run, parsed from its ``final/`` path."""

    final_dir: Path
    experiment: str | None
    dataset: str
    scene: str
    timestamp: str

    @property
    def label(self) -> str:
        parts = [self.experiment, self.dataset, self.scene, self.timestamp]
        return "/".join(p for p in parts if p)


def parse_run_path(final_dir: Path) -> RunPath:
    """Split a ``final/`` path into its ``results/`` identity components.

    Anchors on the ``results`` component when present (the ``{experiment}``
    segment is optional, so counting from the right alone is ambiguous); falls
    back to positional parsing for results trees rooted elsewhere.  Identity is
    metadata: it labels the log lines and names the ``--out`` subdirs.
    """
    final_dir = final_dir.resolve()
    parts = list(final_dir.parts)

    anchors = [i for i, p in enumerate(parts) if p == "results"]
    if anchors:
        rel = parts[anchors[-1] + 1 : -1]  # drop the trailing "final"
        if len(rel) == 4:
            experiment, dataset, scene, timestamp = rel
            return RunPath(final_dir, experiment, dataset, scene, timestamp)
        if len(rel) == 3:
            dataset, scene, timestamp = rel
            return RunPath(final_dir, None, dataset, scene, timestamp)

    # Fallback: read what we can off the tail, tolerate a shallow tree.
    dataset, scene, timestamp = ([_UNKNOWN] * 3 + parts[:-1])[-3:]
    return RunPath(final_dir, None, dataset, scene,
                   timestamp if _TIMESTAMP_RE.match(timestamp) else _UNKNOWN)


# ===========================================================================
# Exported assets
# ===========================================================================


@dataclass
class Transform:
    """One object's Sim(3) at one frame key, plus its per-frame asset."""

    key: FrameKey
    translation: np.ndarray  # (3,)
    rotation: np.ndarray     # (4,) wxyz
    scale: np.ndarray        # (3,)
    ply_perframe: Path | None = None
    mesh_perframe: Path | None = None   # per-frame deformed mesh, object-local P3D
    index: int = 0           # row into the object's dc_offsets / sh_rest arrays


@dataclass
class ObjectAssets:
    """Everything ``final/`` holds for one reconstructed object."""

    obj_idx: int
    ply: Path | None                       # canonical Gaussian, object-local P3D
    mesh: Path | None                      # canonical GLB, object-local P3D
    transforms: dict[FrameKey, Transform]
    dc_offsets: np.ndarray | None = None   # (T, N, 3)
    sh_rest: np.ndarray | None = None      # (T, N, K, 3)


@dataclass
class Camera:
    c2w: np.ndarray            # (4, 4)
    K: np.ndarray | None       # (3, 3)


@dataclass
class Tracks:
    """3D object tracks, from ``final/tracks_2d.npz`` or ``final/tracks_3d.npz``.

    ``xyz[obj]`` is ``(T, P, 3)`` in the same R3 space ``compose(space="world")``
    produces: canonical anchors warped by the per-frame deformation, then placed
    by that frame's Sim(3).  ``frames[t]`` is the temporal index of row ``t``.
    """

    frames: list[int]
    xyz: dict[int, np.ndarray]
    vis: dict[int, np.ndarray]

    def row_for(self, frame: int) -> int | None:
        try:
            return self.frames.index(frame)
        except ValueError:
            return None


def _obj_xyz_members(data) -> dict[int, np.ndarray]:
    """The ``obj_{i}_xyz`` ``(T, P, 3)`` members of an open npz, by object index.

    The reader half of ``interpolation._obj_xyz_members``, which is the single
    writer of that key convention across BOTH track artifacts -- so the two
    readers here (dense ``tracks_2d.npz``, sparse ``tracks_3d.npz``) cannot drift
    apart from each other or from it.
    """
    return {int(m.group(1)): np.asarray(data[name], dtype=np.float32)
            for name in data.files
            if (m := re.fullmatch(r"obj_(\d+)_xyz", name))}


class FinalRun:
    """A ``final/`` directory, loaded from disk.

    Owns the parsed ``poses.json`` (objects + cameras), the discovered asset
    paths, and the render-space composition helpers that turn them into a posed
    Gaussian cloud for any frame key.
    """

    def __init__(self, final_dir: Path):
        self.path = parse_run_path(Path(final_dir))
        self.dir = self.path.final_dir
        poses_path = self.dir / "poses.json"
        if not poses_path.is_file():
            raise FileNotFoundError(f"no poses.json in {self.dir}")
        with open(poses_path) as f:
            raw = json.load(f)

        self.objects: dict[int, ObjectAssets] = {}
        for entry in raw.get("objects", []):
            obj = self._parse_object(entry)
            self.objects[obj.obj_idx] = obj

        self.cameras: dict[FrameKey, Camera] = {}
        for entry in raw.get("cameras", []):
            key = FrameKey(int(entry.get("frame", entry.get("frame_idx", 0))),
                           int(entry.get("view", 0)))
            K = entry.get("K")
            self.cameras[key] = Camera(
                c2w=np.asarray(entry["c2w"], dtype=np.float32),
                K=None if K is None else np.asarray(K, dtype=np.float32),
            )
        # MV runs may key cameras on one view only; first camera per timestamp.
        self._camera_by_frame: dict[int, Camera] = {}
        for key in sorted(self.cameras, key=frame_key_sort_key):
            self._camera_by_frame.setdefault(key.frame, self.cameras[key])

        self._cloud_cache: dict[Path, Any] = {}

    # -- construction helpers ------------------------------------------------

    def _parse_object(self, entry: dict) -> ObjectAssets:
        obj_idx = int(entry["obj_idx"])
        transforms: dict[FrameKey, Transform] = {}
        for i, x in enumerate(entry.get("transforms", [])):
            scale = np.asarray(x["scale"], dtype=np.float32).reshape(-1)
            if scale.size == 1:  # isotropic runs write a 1-vector
                scale = np.repeat(scale, 3)
            pf = x.get("ply_perframe")
            t = Transform(
                key=FrameKey(int(x.get("frame", 0)), int(x.get("view", 0))),
                translation=np.asarray(x["translation"], dtype=np.float32).reshape(3),
                rotation=np.asarray(x["rotation"], dtype=np.float32).reshape(4),
                scale=scale,
                ply_perframe=(self.dir / pf) if pf else None,
                index=i,
            )
            transforms[t.key] = t

        # Per-frame deformed meshes follow the eval convention on disk
        # (`meshes/{obj:03d}/{frame:03d}.ply`, written by the per-frame mesh
        # export), matched to each transform by timestamp — poses.json need not
        # name them, mirroring the canonical-GLB fallback below.
        pf_mesh_dir = self.dir / "meshes" / f"{obj_idx:03d}"
        if pf_mesh_dir.is_dir():
            for t in transforms.values():
                cand = pf_mesh_dir / f"{t.key.frame:03d}.ply"
                if cand.is_file():
                    t.mesh_perframe = cand

        def _rel(key: str) -> Path | None:
            """Resolve an asset reference, ignoring stale or non-file entries.

            ``is_file`` matters: ``save_perframe_poses_json`` writes a
            *directory* under ``"ply"`` for per-frame-only pipelines, and a
            mesh-only run can name a GLB it never wrote.
            """
            v = entry.get(key)
            p = self.dir / v if v else None
            return p if p is not None and p.is_file() else None

        # `save_per_object_mesh` writes meshes/{obj:03d}.glb whether or not
        # poses.json names it, so fall back to the conventional path.
        mesh = _rel("mesh")
        if mesh is None:
            conventional = self.dir / "meshes" / f"{obj_idx:03d}.glb"
            mesh = conventional if conventional.is_file() else None

        obj = ObjectAssets(
            obj_idx=obj_idx,
            ply=_rel("ply"),
            mesh=mesh,
            transforms=transforms,
        )
        obj.dc_offsets = self._load_bin(entry, "dc_offsets", len(transforms))
        obj.sh_rest = self._load_bin(entry, "sh_rest", len(transforms))
        return obj

    def _load_bin(self, entry: dict, key: str, n_transforms: int) -> np.ndarray | None:
        """Read a flat Float32 side-car (``dc_offsets`` / ``sh_rest``).

        Rows are positional -- the writer co-iterates them with ``transforms``
        -- so a row count that disagrees means the two are no longer aligned
        and the offsets cannot be attributed to a frame.
        """
        rel, shape = entry.get(f"{key}_file"), entry.get(f"{key}_shape")
        if not rel or not shape:
            return None
        path = self.dir / rel
        if not path.is_file():
            return None
        arr = np.fromfile(path, dtype=np.float32).reshape(*shape)
        if arr.shape[0] != n_transforms:
            log(f"  ! {path.name}: {arr.shape[0]} rows vs {n_transforms} "
                f"transforms -- ignoring (cannot align to frames)")
            return None
        return arr

    # -- identity + shape ----------------------------------------------------

    @cached_property
    def frame_keys(self) -> list[FrameKey]:
        """All ``(frame, view)`` keys, sorted by ``(view, frame)``."""
        keys = set(self.cameras)
        for obj in self.objects.values():
            keys |= set(obj.transforms)
        return sorted(keys, key=frame_key_sort_key)

    @cached_property
    def frames(self) -> list[int]:
        return sorted({k.frame for k in self.frame_keys})

    @cached_property
    def views(self) -> list[int]:
        return sorted({k.view for k in self.frame_keys})

    @property
    def is_dynamic(self) -> bool:
        return len(self.frames) > 1

    @property
    def is_mv(self) -> bool:
        return len(self.views) > 1

    @property
    def has_gaussians(self) -> bool:
        return any(o.ply is not None for o in self.objects.values())

    @property
    def has_perframe_gaussians(self) -> bool:
        return any(
            t.ply_perframe is not None
            for o in self.objects.values()
            for t in o.transforms.values()
        )

    @property
    def has_meshes(self) -> bool:
        return any(o.mesh is not None for o in self.objects.values())

    @property
    def has_perframe_meshes(self) -> bool:
        return any(
            t.mesh_perframe is not None
            for o in self.objects.values()
            for t in o.transforms.values()
        )

    @property
    def has_renderable_geometry(self) -> bool:
        """Anything a GPU render can pose — gaussians (canonical or per-frame) or
        a mesh.  When false, `train_views`/`orbit`/`synth_nvs` have nothing to do."""
        return self.has_gaussians or self.has_perframe_gaussians or self.has_meshes

    @property
    def prefers_mesh(self) -> bool:
        """Render the mesh only when there are no gaussians of either kind —
        gaussians carry appearance the mesh does not, so they always win."""
        return (not (self.has_gaussians or self.has_perframe_gaussians)
                and self.has_meshes)

    @property
    def poses_vary(self) -> bool:
        """Does any object's Sim(3) actually change across its frame keys?

        Distinguishes a genuinely moving object from a run that is *labelled*
        dynamic but stored one static pose per frame (a run whose deformation
        lives only in its pre-rendered ``renders_train/``).
        """
        for obj in self.objects.values():
            xs = list(obj.transforms.values())
            for x in xs[1:]:
                if not (np.allclose(x.translation, xs[0].translation)
                        and np.allclose(x.rotation, xs[0].rotation)
                        and np.allclose(x.scale, xs[0].scale)):
                    return True
        return False

    @cached_property
    def resolution(self) -> tuple[int, int]:
        """(H, W) of the run's renders -- from an exported PNG, else from K.

        ``poses.json`` carries ``K`` but no image size, so an exported PNG is
        the only record of the resolution the run actually rendered at.
        """
        for sub in ("renders_train", "renders_test"):
            hits = sorted((self.dir / sub).glob("**/*.png"))
            if hits:
                _lazy_imports(gpu=False)
                with Image.open(hits[0]) as im:
                    return im.size[1], im.size[0]
        for cam in self.cameras.values():
            if cam.K is not None:
                return int(round(cam.K[1, 2] * 2)), int(round(cam.K[0, 2] * 2))
        return 512, 512

    def camera(self, key: FrameKey) -> Camera | None:
        """That key's camera, or the timestamp's when views are keyed on one."""
        return self.cameras.get(key) or self._camera_by_frame.get(key.frame)

    @cached_property
    def tracks(self) -> Tracks | None:
        """The run's 3D tracks: ``final/tracks_2d.npz``'s ``obj_*_xyz``, falling
        back to ``final/tracks_3d.npz`` for the objects it does not carry.

        Two artifacts because a run whose ``tracks_2d.npz`` carries its own
        PIXEL correspondence has its ``obj_*_xyz`` dropped from that file on
        purpose: its ``uv`` anchors are a different, denser set, and pairing two
        anchor counts under one object is the schema hazard
        ``interpolation.write_tapvid_tracks`` exists to prevent.  Such runs still
        write ``tracks_3d.npz`` out of the SAME world-space
        ``compute_object_tracks`` call, so it is the 3D source here.
        """
        # tracks_3d.npz carries no frame axis of its own -- it is written from the
        # same `all_frame_indices` tracks_2d.npz records, so those rows win when
        # both exist, and the run's own timestamps stand in when neither does.
        frames, xyz, vis = self.frames, {}, {}
        dense = self.dir / "tracks_2d.npz"
        if dense.is_file():
            with np.load(dense, allow_pickle=False) as data:
                frames = [int(f) for f in data["frames"]]
                xyz = _obj_xyz_members(data)
                vis = {obj: np.asarray(data[f"obj_{obj}_vis"], dtype=bool)
                       for obj in xyz if f"obj_{obj}_vis" in data.files}
        sparse = self.dir / "tracks_3d.npz"
        if sparse.is_file():
            with np.load(sparse, allow_pickle=False) as data:
                extra = _obj_xyz_members(data)
            for obj, pts in extra.items():
                if obj in xyz:
                    continue
                if pts.shape[0] != len(frames):
                    log(f"  ! tracks_3d.npz obj {obj}: {pts.shape[0]} rows vs "
                        f"{len(frames)} frames -- ignoring")
                    continue
                xyz[obj] = pts
        return Tracks(frames, xyz, vis) if xyz else None

    def summary(self) -> str:
        kind = "dynamic" if self.is_dynamic else "static"
        kind += "/mv" if self.is_mv else "/mono"
        H, W = self.resolution
        bits = [f"{len(self.objects)} object(s)", f"{len(self.frames)} frame(s)",
                f"{len(self.views)} view(s)", kind, f"{W}x{H}"]
        if self.has_perframe_gaussians:
            bits.append("per-frame warped gaussians")
        if not self.has_gaussians:
            bits.append("no canonical gaussians")
        return ", ".join(bits)

    # -- scene composition ---------------------------------------------------

    def _cloud(self, path: Path, device: str, cache: bool) -> "GaussianCloud":
        """Load a PLY, caching only assets that get reused across frames.

        Canonical clouds are posed once per frame, so caching pays for itself;
        a ``ply_perframe`` asset is used exactly once, so caching it would just
        pin every frame's splats in VRAM.
        """
        cached = self._cloud_cache.get(path)
        if cached is None:
            cached = load_gaussian_ply(path, device)
            if cache:
                self._cloud_cache[path] = cached
        return cached

    def resolve_transform(self, obj: ObjectAssets, key: FrameKey) -> Transform | None:
        """The pose to place ``obj`` with at ``key`` -- exact, or its anchor.

        Poses are camera-space, so an object keyed at a different frame key is
        placed via *that* key's ``c2w`` and still lands in the same world spot.
        Two fallbacks: the same timestamp under another view (MV runs key the
        pose on one view), then -- on a static run only -- a lone transform (an
        object reconstructed once but observed by several cameras, e.g. CO3D).
        """
        exact = obj.transforms.get(key)
        if exact is not None:
            return exact
        same_frame = sorted(k for k in obj.transforms if k.frame == key.frame)
        if same_frame:
            return obj.transforms[same_frame[0]]
        if not self.is_dynamic and len(obj.transforms) == 1:
            return next(iter(obj.transforms.values()))
        return None

    def compose(
        self,
        key: FrameKey,
        *,
        device: str = "cuda",
        space: str = "world",
        objects: Iterable[int] | None = None,
        use_perframe: bool = True,
        pose_from: FrameKey | None = None,
        geometry: "Callable[[int, FrameKey], GaussianCloud | None] | None" = None,
    ) -> "GaussianCloud | None":
        """Posed Gaussian cloud for one frame key, or None if nothing is posed.

        ``space``: ``object`` (no pose), ``camera`` (Sim(3) + P3D->R3) or
        ``world`` (+ the ``c2w`` of the key each object's pose is anchored on).
        ``camera`` space is only meaningful when every object has a pose at
        ``key`` itself -- otherwise the anchors disagree on what "camera" means.

        ``pose_from`` splits *which frame's shape* from *where it is placed*:
        geometry and colour still come from ``key`` (so the per-frame warped
        deformation is kept) while the Sim(3) + ``c2w`` come from that other
        key.  Pinning it to one frame strips the object's root motion, leaving
        only the deformation -- what a turntable of a moving object wants.

        ``geometry`` overrides where the object-local cloud COMES from, without
        touching where it is put: a callback returning None for an object falls
        through to the exported PLYs, and everything after -- the colour shift, the
        Sim(3), P3D->R3 and the ``c2w`` -- runs identically either way, so a
        provider cloud lands in exactly the same world spot as the PLY it replaces.
        :class:`DeformationSource` is the one caller.
        """
        wanted = set(self.objects if objects is None else objects)
        clouds: list[GaussianCloud] = []
        for obj_idx in sorted(set(self.objects) & wanted):
            obj = self.objects[obj_idx]
            xform = self.resolve_transform(obj, key)
            if xform is None:
                continue
            pose = xform if pose_from is None else (
                self.resolve_transform(obj, pose_from) or xform
            )
            cloud = None if geometry is None else geometry(obj_idx, key)
            if cloud is None:
                perframe = xform.ply_perframe if use_perframe else None
                src = perframe or obj.ply
                if src is None:
                    continue
                cloud = self._cloud(src, device, cache=perframe is None)
            # Cloned before the colour shift writes into `sh` -- the cache above and
            # a provider both hand out tensors they still own.
            cloud = apply_color_shift(cloud.clone(), obj, xform)
            if space == "object":
                clouds.append(cloud)
                continue
            cloud = p3d_to_r3(
                apply_sim3(cloud, pose.rotation, pose.translation, pose.scale)
            )
            if space == "world":
                cam = self.camera(pose.key)
                if cam is not None:
                    cloud = cam_to_world(cloud, cam.c2w)
            clouds.append(cloud)

        return concat_clouds(clouds) if clouds else None

    def compose_mesh(
        self,
        key: FrameKey,
        *,
        device: str = "cuda",
        objects: Iterable[int] | None = None,
        pose_from: FrameKey | None = None,
    ) -> "MeshData | None":
        """Posed world-space mesh for one frame key -- the mesh twin of :meth:`compose`.

        The per-frame deformed mesh (``mesh_perframe``) when present, else the
        canonical GLB -- both live in the same object-local PyTorch3D space as
        the canonical Gaussians, so they go through the identical Sim(3) ->
        P3D->R3 -> ``c2w`` chain, and ``pose_from`` means the same thing: keep
        ``key``'s deformed geometry while borrowing another frame's Sim(3).
        """
        wanted = set(self.objects if objects is None else objects)
        parts: list[MeshData] = []
        for obj_idx in sorted(set(self.objects) & wanted):
            obj = self.objects[obj_idx]
            xform = self.resolve_transform(obj, key)
            if xform is None:
                continue
            src = xform.mesh_perframe or obj.mesh   # per-frame deformation wins
            if src is None:
                continue
            pose = xform if pose_from is None else (
                self.resolve_transform(obj, pose_from) or xform
            )
            mesh = load_mesh(src, device)
            verts = self._place_points(mesh.verts.detach().cpu().numpy(), pose)
            parts.append(MeshData(
                verts=torch.as_tensor(verts, dtype=torch.float32, device=device),
                faces=mesh.faces, colors=mesh.colors,
            ))
        return merge_meshes(parts) if parts else None

    def _place_points(self, local: np.ndarray, pose: Transform) -> np.ndarray:
        """Object-local P3D points -> posed R3 world, the plain-point chain."""
        pts = apply_sim3_points(local, pose.rotation, pose.translation, pose.scale)
        return _cam_to_world_points(r3_to_p3d_points(pts), self.camera(pose.key))

    def _unplace_points(self, world: np.ndarray, pose: Transform) -> np.ndarray:
        """Inverse of :meth:`_place_points`."""
        pts = r3_to_p3d_points(_world_to_cam_points(world, self.camera(pose.key)))
        return unapply_sim3_points(pts, pose.rotation, pose.translation, pose.scale)

    def rebase_points(self, obj: ObjectAssets, pts: np.ndarray,
                      src_key: FrameKey, dst_key: FrameKey) -> np.ndarray:
        """Move posed R3 points from ``src_key``'s placement to ``dst_key``'s.

        The inverse of the Gaussian chain followed by the forward one: undo
        ``src``'s c2w + P3D->R3 + Sim(3) back to object-local, then re-apply
        ``dst``'s.  Used to strip root motion from ``final/tracks_2d.npz``
        tracks so they line up with a pose-pinned render.
        """
        src = self.resolve_transform(obj, src_key)
        dst = self.resolve_transform(obj, dst_key)
        if src is None or dst is None:
            return pts
        return self._place_points(self._unplace_points(pts, src), dst)


def discover_final_dirs(patterns: Iterable[str]) -> list[Path]:
    """Expand CLI paths (literal, glob, or a run dir holding ``final/``)."""
    out: list[Path] = []
    for pattern in patterns:
        hits = [Path(p) for p in sorted(_glob.glob(pattern))] or [Path(pattern)]
        for hit in hits:
            if (hit / "poses.json").is_file():
                out.append(hit)
            elif (hit / "final" / "poses.json").is_file():
                out.append(hit / "final")
            else:
                log(f"  ! skipping {hit}: no poses.json (or final/poses.json)")
    return list(dict.fromkeys(p.resolve() for p in out))


# ===========================================================================
# Gaussian assets: load + the pipeline's pose/space conventions
# ===========================================================================


@dataclass
class GaussianCloud:
    """Raw gsplat-ready Gaussian parameters (activations already applied)."""

    means: Any       # (N, 3)
    quats: Any       # (N, 4) wxyz, normalized
    scales: Any      # (N, 3)
    opacities: Any   # (N,)
    sh: Any          # (N, K, 3), K = (degree + 1)^2

    def __len__(self) -> int:
        return int(self.means.shape[0])

    def clone(self) -> "GaussianCloud":
        return GaussianCloud(self.means.clone(), self.quats.clone(), self.scales.clone(),
                             self.opacities.clone(), self.sh.clone())

    def centroid(self) -> np.ndarray:
        """Outlier-robust center (median of means)."""
        return self.means.median(dim=0).values.detach().cpu().numpy()


def load_gaussian_ply(path: Path, device: str = "cuda") -> GaussianCloud:
    """Read ``.ply`` / ``.compressed.ply`` into activated Gaussian parameters.

    Inverts what ``core/utils/io_utils.py::save_gaussian_ply`` wrote: ``exp``
    on log-scales, ``sigmoid`` on inverse-sigmoid opacities, normalize on the
    biased quaternion.  Both formats carry the full SH (``.compressed.ply``
    round-trips ``f_rest_*`` through gsconverter's quantized ``sh`` element).
    """
    path = Path(path)
    if path.name.endswith(".compressed.ply"):
        from gsconverter.formats.compressed_ply import CompressedPlyFormat
        data = CompressedPlyFormat().read(str(path))
    else:
        from plyfile import PlyData
        data = PlyData.read(str(path))["vertex"].data

    n = len(data)
    xyz = np.stack([data["x"], data["y"], data["z"]], -1).astype(np.float32)
    rot = np.stack([data[f"rot_{i}"] for i in range(4)], -1).astype(np.float32)
    scale = np.stack([data[f"scale_{i}"] for i in range(3)], -1).astype(np.float32)
    opacity = np.asarray(data["opacity"], dtype=np.float32)

    sh = np.stack([data[f"f_dc_{i}"] for i in range(3)], -1)[:, None, :].astype(np.float32)
    n_rest = sum(1 for k in data.dtype.names if k.startswith("f_rest_"))
    if n_rest:
        rest = np.stack([data[f"f_rest_{i}"] for i in range(n_rest)], -1).astype(np.float32)
        # save_gaussian_ply flattens channel-major: (N, 3, K-1) -> (N, 3*(K-1)).
        rest = rest.reshape(n, 3, n_rest // 3).transpose(0, 2, 1)
        sh = np.concatenate([sh, rest], axis=1)

    t = torch.from_numpy
    quats = t(rot).to(device)
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return GaussianCloud(
        means=t(xyz).to(device),
        quats=quats,
        scales=t(scale).to(device).exp(),
        opacities=torch.sigmoid(t(opacity).to(device)),
        sh=t(np.ascontiguousarray(sh)).to(device),
    )


@dataclass
class MeshData:
    """A triangle mesh with per-vertex colour, ready to rasterize."""

    verts: Any    # (V, 3)
    faces: Any    # (F, 3) int32
    colors: Any   # (V, 3) in [0, 1]

    def __len__(self) -> int:
        return int(self.verts.shape[0])

    def centroid(self) -> np.ndarray:
        """Outlier-robust center, matching :meth:`GaussianCloud.centroid`."""
        return self.verts.median(dim=0).values.detach().cpu().numpy()


_MESH_CACHE: dict[Path, MeshData] = {}
_GLCTX = None


def load_mesh(path: Path, device: str = "cuda") -> MeshData:
    """Read a GLB/PLY/OBJ into object-local vertices, faces and vertex colours.

    Vertex colours are what ``save_per_object_mesh`` embeds in the GLB (the SLAT
    mesh decoder's ``vertex_attrs[:, :3]``); meshes without them render neutral
    grey rather than failing.
    """
    path = Path(path)
    cached = _MESH_CACHE.get(path)
    if cached is not None:
        return cached
    import trimesh

    m = trimesh.load(str(path), force="mesh", process=False)
    verts = torch.as_tensor(np.asarray(m.vertices), dtype=torch.float32, device=device)
    faces = torch.as_tensor(np.asarray(m.faces), dtype=torch.int32, device=device)
    rgb = getattr(getattr(m, "visual", None), "vertex_colors", None)
    if rgb is not None and len(rgb) == len(verts):
        colors = torch.as_tensor(
            np.asarray(rgb)[:, :3] / 255.0, dtype=torch.float32, device=device)
    else:
        colors = torch.full((len(verts), 3), 0.75, dtype=torch.float32, device=device)
    out = MeshData(verts=verts, faces=faces, colors=colors)
    _MESH_CACHE[path] = out
    return out


def merge_meshes(parts: list[MeshData]) -> MeshData:
    """Concatenate meshes into one, offsetting each part's face indices."""
    if len(parts) == 1:
        return parts[0]
    faces, offset = [], 0
    for p in parts:
        faces.append(p.faces + offset)
        offset += len(p)
    return MeshData(
        verts=torch.cat([p.verts for p in parts]),
        faces=torch.cat(faces).contiguous(),
        colors=torch.cat([p.colors for p in parts]),
    )


def rows_correspond(a, b) -> bool:
    """Do two per-frame geometries index the same splats (vertices) row for row?

    **Equal length does not settle it.**  ``save_compressed_ply`` MORTON-SORTS what
    it writes (gsconverter's ``_sort_morton_order``: splats are reordered by a
    spatial curve so neighbours compress together), so every frame of a deformation
    lands in ITS OWN order and row `i` is a different splat in each file -- same
    count, no correspondence.  Blending those rows would mix unrelated Gaussians.

    So test a quantity the deformation does NOT touch.  ``warp_gaussians_high_res``
    warps means and quats only, leaving opacity and scale alone -- so a per-splat
    value that no longer lines up row for row IS the reorder.  For a mesh it is the
    face array: one canonical mesh warped keeps its tessellation, and a per-frame
    reconstruction does not.

    A CONSTANT witness proves nothing: a cloud that writes one opacity for every
    splat passes the opacity comparison whatever the order is.  Fall through to the
    scales, and refuse outright when neither varies: correspondence cannot be
    established, and interpolating on an unverified one would blend unrelated splats.

    A per-frame pipeline that legitimately varies opacity per frame is refused too;
    that is the conservative direction, and the caller says so.
    """
    if len(a) != len(b):
        return False
    if isinstance(a, MeshData):
        return a.faces.shape == b.faces.shape and bool(torch.equal(a.faces, b.faces))
    for witness in (a.opacities, a.scales):
        if float(witness.std()) < 1e-6:      # carries no order information
            continue
        other = b.opacities if witness is a.opacities else b.scales
        return bool(torch.allclose(witness, other, atol=1e-4))
    return False


@dataclass
class _WarpShim:
    """The two attributes ``warp_gaussians_high_res`` reads off a ``Gaussian``."""

    get_xyz: Any
    get_rotation: Any


class DeformationSource:
    """Canonical Gaussians + the deformation field, read from ``final/final.pt``.

    The turntable's geometry source, in place of ``gaussians/{obj}/{frame}.ply``.
    Those exported clouds are this same warp, but ``save_compressed_ply`` (the
    default) MORTON-SORTS every file it writes, so row `i` is a different splat in
    each frame and they cannot be interpolated -- see :func:`rows_correspond`.  The
    cache is order-free and unquantized, so re-running the warp here restores the
    correspondence.  It is also faster: one load, instead of one
    :func:`load_gaussian_ply` per frame per pass.

    Serves a run with no deformation field too, returning the canonical cloud
    unwarped -- static runs, where it is simply the better-conditioned read.
    """

    def __init__(self, canonical, field, warp_kwargs, device):
        self._canonical = canonical      # {obj: GaussianCloud}, object-local
        self._field = field              # the four state-shaped dicts
        self._warp = warp_kwargs
        self._device = device
        self._failed = False

    # -- opening --------------------------------------------------------------

    @classmethod
    def open(cls, final_dir: Path, *, device: str, objects, frames,
             required: bool = False) -> "DeformationSource | None":
        """Load the source, or None (with ONE logged reason) if it cannot serve.

        ALL-OR-NOTHING per run: a partially served scene would compose provider rows
        beside PLY rows, and the correspondence this exists for would be gone again.
        Declines on a missing/broken ``final.pt``, absent canonical Gaussians, an
        object it does not carry, or a field that does not key every requested
        object at every requested frame.  Never raises -- the orbit is cosmetic
        output closing a run that may have cost hours.
        """
        try:
            return cls._open(final_dir, device, objects, frames)
        except Exception as exc:   # noqa: BLE001 — any failure means "use the PLYs"
            log(f"    (final.pt: {type(exc).__name__}: {exc} — falling back to the "
                f"exported per-frame PLYs)")
            if required:
                raise
            return None

    @classmethod
    def _open(cls, final_dir, device, objects, frames):
        from genia.core.utils.pipeline_state import (
            gaussian_blob_to_render_tensors, load_pipeline_cache_raw,
        )
        from genia.core.utils.deformation import _lookup_per_frame_deformation

        path = Path(final_dir) / "final.pt"
        if not path.is_file():
            log("    (no final.pt — reading geometry from the exported PLYs)")
            return None
        blob = load_pipeline_cache_raw(str(path))
        raw = blob.get("canonical_gaussians") or {}
        # Keep the four fields we need and drop the rest NOW: the slats, tokens and
        # voxel correspondences in there are most of the file.
        field = {k: blob.get(k) for k in (
            "canonical_mesh_verts", "canonical_mesh_faces",
            "canonical_mesh_per_frame_verts", "canonical_mesh_per_frame_rotations")}
        del blob

        canonical = {}
        for obj_idx in sorted(objects):
            g = raw.get(obj_idx)
            if g is None:
                log(f"    (final.pt carries no canonical gaussian for object "
                    f"{obj_idx} — reading geometry from the exported PLYs)")
                return None   # a canonical run missing ONE object; per-frame-only
                              # runs are caught before the load
            canonical[obj_idx] = GaussianCloud(
                **gaussian_blob_to_render_tensors(g, device=device))

        src = cls(canonical, field, _warp_kwargs(final_dir), device)
        # Every (object, frame) or none: `_lookup_per_frame_deformation` is the
        # export's own "is the field keyed here?" test, so the orbit and the PLY
        # writer agree by construction rather than by convention.
        # ... on the CPU the tensors already sit on: this asks only whether the
        # field is KEYED, and the lookup moves what it resolves, which would copy
        # every frame's tensors to the device to answer a question about dict keys.
        covered = [_lookup_per_frame_deformation(
            field["canonical_mesh_verts"], field["canonical_mesh_per_frame_verts"],
            field["canonical_mesh_per_frame_rotations"], o, f,
            torch.device("cpu"), field["canonical_mesh_faces"]) is not None
            for o in canonical for f in sorted(set(frames))]
        if any(covered) and not all(covered):
            log("    (final.pt's deformation field does not cover every frame — "
                "reading geometry from the exported PLYs)")
            return None
        if any(covered):
            # Force the pytorch3d import and its kernel while there is still a clean
            # fallback, rather than mid-clip on the first real warp.
            obj = next(iter(canonical))
            probe = canonical[obj]
            src._warp_one(_WarpShim(probe.means[:8], probe.quats[:8]), obj,
                          sorted(set(frames))[0])
            log(f"    geometry from final.pt: canonical gaussians + the deformation "
                f"field ({len(canonical)} object(s), no Morton-sorted PLYs)")
        else:
            log("    geometry from final.pt: canonical gaussians (no deformation "
                "field — nothing to interpolate)")
        return src

    # -- serving --------------------------------------------------------------

    def cloud(self, obj_idx: int, frame: int) -> "GaussianCloud | None":
        """Object-local Gaussians for one exported frame — what
        ``load_gaussian_ply(gaussians/{obj}/{frame}.ply)`` would give before the
        Morton sort.  The canonical cloud when the object has no field."""
        canon = self._canonical.get(obj_idx)
        if canon is None or self._failed:
            return None
        try:
            warped = self._warp_one(_WarpShim(canon.means, canon.quats),
                                    obj_idx, frame)
        except Exception as exc:   # noqa: BLE001
            log(f"    ! warp failed ({type(exc).__name__}: {exc}) — falling back to "
                f"the exported PLYs for the rest of this render")
            self._failed = True
            return None
        if warped is None:
            return canon
        means, quats = warped
        # scales/opacities/SH are the canonical tensors, SHARED not copied: the warp
        # does not touch them.  That is what makes two frames of this source
        # correspond row for row by construction (`compose` clones before mutating).
        return GaussianCloud(means=means, quats=quats, scales=canon.scales,
                             opacities=canon.opacities, sh=canon.sh)

    def _warp_one(self, shim, obj_idx: int, frame: int):
        from genia.core.utils.deformation import _lookup_per_frame_deformation, warp_gaussians_high_res

        resolved = _lookup_per_frame_deformation(
            self._field["canonical_mesh_verts"],
            self._field["canonical_mesh_per_frame_verts"],
            self._field["canonical_mesh_per_frame_rotations"],
            obj_idx, int(frame), torch.device(self._device),
            self._field["canonical_mesh_faces"])
        if resolved is None:
            return None
        canon_verts, frame_verts, frame_rots, faces = resolved
        with torch.no_grad():
            return warp_gaussians_high_res(shim, canon_verts, frame_verts,
                                           frame_rots, faces=faces, **self._warp)

    def provider(self):
        """The callback :meth:`FinalRun.compose` takes as ``geometry``."""
        return lambda obj_idx, key: self.cloud(obj_idx, key.frame)


def _warp_kwargs(final_dir: Path) -> dict:
    """The run's OWN ``deformation_warp`` knobs, so re-warping here reproduces the
    warp its per-frame PLYs were exported with.  Defaults mirror
    ``core/configs/deformation_warp/default.yaml``."""
    cfg = (load_run_config(Path(final_dir)) or {}).get("deformation_warp") or {}
    return {
        "K": int(cfg.get("knn_k", 4)),
        "eps": float(cfg.get("knn_eps", 1e-8)),
        "chunk_size": int(cfg.get("knn_chunk_size", 131072)),
    }


def blend_geometry(a, b, w: float):
    """Linearly blend two per-frame geometries that share a correspondence.

    What a deformation field warps per frame is ONE canonical asset, so row `i` is
    the same splat -- the same vertex -- in every frame.  Blending two of them
    therefore samples the motion field BETWEEN the timestamps it was written at,
    which is what ``orbit.slowmo`` renders.

    Falls back to ``a`` (holding the earlier frame) whenever :func:`rows_correspond`
    says the rows do not line up -- a per-frame reconstruction, or the Morton-sorted
    output of the compressed PLY writer.  Row `i` means nothing across those, so
    there is nothing to interpolate.
    """
    if not rows_correspond(a, b):
        return a
    if isinstance(a, MeshData):
        return MeshData(verts=torch.lerp(a.verts, b.verts, w), faces=a.faces,
                        colors=torch.lerp(a.colors, b.colors, w))
    # Rotations nlerp'd on the shorter arc (hence the sign align): consecutive
    # frames of a deformation are a small rotation apart, where nlerp and slerp
    # differ by far less than a pixel.
    quats = torch.where((a.quats * b.quats).sum(-1, keepdim=True) < 0,
                        -b.quats, b.quats)
    quats = torch.lerp(a.quats, quats, w)
    return GaussianCloud(
        means=torch.lerp(a.means, b.means, w),
        quats=quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-12),
        scales=torch.lerp(a.scales, b.scales, w),
        opacities=torch.lerp(a.opacities, b.opacities, w),
        sh=torch.lerp(a.sh, b.sh, w),
    )


def render_mesh(mesh: MeshData, c2w: np.ndarray, K: np.ndarray, H: int, W: int,
                bg: np.ndarray):
    """nvdiffrast render -> ``(rgb, alpha, depth)``, matching :func:`render_cloud`.

    Reuses the pipeline's own ``render_rgba_and_depth``, so the projection and
    the y-flip to image convention are the ones Stage-2 mesh guidance uses.
    """
    global _GLCTX
    import nvdiffrast.torch as dr

    from genia.core.utils.mesh_rendering import render_rgba_and_depth

    if _GLCTX is None:
        _GLCTX = dr.RasterizeCudaContext()
    c2w = np.asarray(c2w, dtype=np.float64)
    R = torch.as_tensor(c2w[:3, :3], dtype=torch.float32, device=mesh.verts.device)
    t = torch.as_tensor(c2w[:3, 3], dtype=torch.float32, device=mesh.verts.device)
    verts_cam = (mesh.verts - t) @ R          # world -> R3 camera (row-vector c2w)

    rgb, alpha, depth = render_rgba_and_depth(
        verts_cam, mesh.faces, mesh.colors, _GLCTX,
        float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]), H, W,
    )
    # render_rgba_and_depth leaves the background at zero; composite on `bg` so
    # the output matches the Gaussian path (foreground on bg, coverage in alpha).
    bg_t = torch.as_tensor(bg, dtype=torch.float32, device=rgb.device)
    a = alpha.clamp(0, 1)[..., None]
    return (rgb.clamp(0, 1) * a + bg_t * (1 - a)), alpha.clamp(0, 1), depth


def apply_color_shift(cloud: GaussianCloud, obj: ObjectAssets,
                      xform: Transform) -> GaussianCloud:
    """Fold this frame's ``dc_offset`` / ``sh_rest`` into the cloud's SH."""
    if obj.dc_offsets is not None:
        delta = torch.from_numpy(obj.dc_offsets[xform.index]).to(cloud.sh.device)
        cloud.sh[:, 0, :] = cloud.sh[:, 0, :] + delta               # (N, 3)
    if obj.sh_rest is not None:
        rest = torch.from_numpy(obj.sh_rest[xform.index]).to(cloud.sh.device)
        cloud.sh = torch.cat([cloud.sh[:, :1], rest], dim=1)        # (N, K-1, 3)
    return cloud


def _sim3_matrix(rotation: np.ndarray) -> np.ndarray:
    """Row-vector rotation matrix R for a wxyz quaternion (P3D convention)."""
    w, x, y, z = np.asarray(rotation, dtype=np.float64) / (
        np.linalg.norm(rotation) + 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)],
        [2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)],
        [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64).T  # quaternion_to_matrix is column-vector; row-vector = its T


def apply_sim3_points(pts: np.ndarray, rotation, translation, scale) -> np.ndarray:
    """Object-local -> camera-space Sim(3) on plain points: ``x * s @ R + t``."""
    R = _sim3_matrix(rotation)
    return (np.asarray(pts, np.float64) * np.asarray(scale, np.float64)) @ R \
        + np.asarray(translation, np.float64)


def unapply_sim3_points(pts: np.ndarray, rotation, translation, scale) -> np.ndarray:
    """Inverse of :func:`apply_sim3_points`: ``((p - t) @ R.T) / s``."""
    R = _sim3_matrix(rotation)
    return ((np.asarray(pts, np.float64) - np.asarray(translation, np.float64)) @ R.T) \
        / np.asarray(scale, np.float64)


def r3_to_p3d_points(pts: np.ndarray) -> np.ndarray:
    """R3 <-> P3D positions (negate X and Y -- the flip is self-inverse)."""
    out = np.asarray(pts, np.float64).copy()
    out[..., :2] *= -1
    return out


def _world_to_cam_points(pts: np.ndarray, cam: "Camera | None") -> np.ndarray:
    if cam is None or np.allclose(cam.c2w, np.eye(4)):
        return np.asarray(pts, np.float64)
    R, t = cam.c2w[:3, :3], cam.c2w[:3, 3]
    return (np.asarray(pts, np.float64) - t) @ R


def _cam_to_world_points(pts: np.ndarray, cam: "Camera | None") -> np.ndarray:
    if cam is None or np.allclose(cam.c2w, np.eye(4)):
        return np.asarray(pts, np.float64)
    R, t = cam.c2w[:3, :3], cam.c2w[:3, 3]
    return np.asarray(pts, np.float64) @ R.T + t


def apply_sim3(cloud: GaussianCloud, rotation: np.ndarray, translation: np.ndarray,
               scale: np.ndarray) -> GaussianCloud:
    """Object-local -> camera-space Sim(3), matching ``apply_pose_to_gaussian``.

    Row-vector convention: ``x * s @ R + t`` (the torch twin of
    :func:`apply_sim3_points`); the per-Gaussian orientation is pre-multiplied
    by the *inverse* object quaternion (the ``make_scene`` convention), and
    scales pick up the object scale element-wise.
    """
    device = cloud.means.device
    q = torch.as_tensor(rotation, dtype=torch.float32, device=device)
    q = q / q.norm().clamp_min(1e-12)
    t = torch.as_tensor(translation, dtype=torch.float32, device=device)
    s = torch.as_tensor(scale, dtype=torch.float32, device=device)

    R = quaternion_to_matrix(q.unsqueeze(0)).squeeze(0)         # (3, 3)
    cloud.means = torch.mm(cloud.means * s, R) + t              # @ R, not @ R.T
    q_inv = quaternion_invert(q.unsqueeze(0)).squeeze(0)
    cloud.quats = quaternion_multiply(
        q_inv.unsqueeze(0).expand(cloud.quats.shape[0], -1), cloud.quats
    )
    cloud.scales = cloud.scales * s
    return cloud


def p3d_to_r3(cloud: GaussianCloud) -> GaussianCloud:
    """PyTorch3D (X-left/Y-up) -> R3 (X-right/Y-down) camera space."""
    cloud.means = p3d_to_r3_positions(cloud.means)
    cloud.quats = p3d_to_r3_quaternions(cloud.quats)
    return cloud


def cam_to_world(cloud: GaussianCloud, c2w: np.ndarray) -> GaussianCloud:
    """R3 camera space -> R3 world space (no-op for an identity ``c2w``)."""
    from genia.core.utils.rendering import transform_gaussian_params_cam_to_world
    cloud.means, cloud.quats = transform_gaussian_params_cam_to_world(
        cloud.means, cloud.quats, np.asarray(c2w, dtype=np.float32)
    )
    return cloud


def concat_clouds(clouds: list[GaussianCloud]) -> GaussianCloud:
    """Merge object clouds into one scene cloud (SH bands zero-padded to max K)."""
    if len(clouds) == 1:
        return clouds[0]
    k = max(c.sh.shape[1] for c in clouds)
    shs = []
    for c in clouds:
        if c.sh.shape[1] < k:
            pad = torch.zeros(c.sh.shape[0], k - c.sh.shape[1], 3, device=c.sh.device)
            shs.append(torch.cat([c.sh, pad], dim=1))
        else:
            shs.append(c.sh)
    return GaussianCloud(
        means=torch.cat([c.means for c in clouds]),
        quats=torch.cat([c.quats for c in clouds]),
        scales=torch.cat([c.scales for c in clouds]),
        opacities=torch.cat([c.opacities for c in clouds]),
        sh=torch.cat(shs),
    )


# ===========================================================================
# Rendering + image helpers
# ===========================================================================


def render_cloud(cloud: GaussianCloud, c2w: np.ndarray, K: np.ndarray,
                 H: int, W: int, bg: np.ndarray):
    """gsplat render -> ``(rgb (H,W,3), alpha (H,W), depth (H,W))`` tensors."""
    bg_t = torch.as_tensor(bg, dtype=torch.float32, device=cloud.means.device)
    rgb, alpha, depth = render_gaussian_params(
        means=cloud.means, quats=cloud.quats, scales=cloud.scales,
        opacities=cloud.opacities, features=cloud.sh,
        c2w=np.asarray(c2w, dtype=np.float32), K_matrix=np.asarray(K, dtype=np.float32),
        W=W, H=H, bg_color=bg_t,
    )
    return rgb.clamp(0, 1), alpha.clamp(0, 1), depth


def to_rgba(rgb, alpha) -> np.ndarray:
    """``(H,W,3)+(H,W)`` float tensors -> RGBA uint8 (coverage in alpha).

    Packed and cast on device: one uint8 transfer instead of two float32 ones.
    Truncating (not rounding) keeps this bit-identical to the pipeline's own
    ``(x * 255).astype(uint8)`` in ``save_canonical_renders_perframe``.
    """
    rgba = torch.cat([rgb, alpha[..., None]], dim=-1)
    return rgba.mul(255).to(torch.uint8).cpu().numpy()


def flatten_alpha(arr: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """RGBA uint8 -> RGB uint8, compositing the transparent part onto ``bg``."""
    bg_u8 = np.asarray(bg, np.float32) * 255.0
    a = arr[..., 3:4].astype(np.float32) / 255.0
    return (arr[..., :3].astype(np.float32) * a + bg_u8 * (1.0 - a)).astype(np.uint8)


def project_r3(pts: np.ndarray, c2w: np.ndarray, K: np.ndarray) -> np.ndarray:
    """R3 world points -> ``(N, 3)`` of ``(u, v, z_cam)`` for a pinhole camera."""
    w2c = np.linalg.inv(np.asarray(c2w, np.float64))
    cam = np.asarray(pts, np.float64) @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam[:, 2]
    safe = np.where(np.abs(z) < 1e-6, 1e-6, z)
    u = K[0, 0] * cam[:, 0] / safe + K[0, 2]
    v = K[1, 1] * cam[:, 1] / safe + K[1, 2]
    return np.stack([u, v, z], axis=-1)


def track_colors(n: int) -> list[tuple[int, int, int]]:
    """``n`` visually distinct RGB triples (golden-angle hue walk)."""
    import colorsys
    return [
        tuple(int(255 * c) for c in colorsys.hsv_to_rgb((i * 0.618033988) % 1.0, 0.85, 1.0))
        for i in range(n)
    ]


def draw_tracks(rgba: np.ndarray, trails: np.ndarray, c2w: np.ndarray, K: np.ndarray,
                depth: np.ndarray | None, colors: list[tuple[int, int, int]],
                width: int = 2, head: int = 3, opacity: float = 1.0) -> np.ndarray:
    """Draw 3D point trails onto an RGBA frame, in place of a 3D line renderer.

    ``trails`` is ``(P, L, 3)`` -- one polyline per tracked point, oldest first,
    in the same R3 space the frame was rendered from.  When a ``depth`` buffer
    is given, vertices behind the rendered surface are dropped so trails go
    *behind* the object instead of always floating on top.  The trail fades
    towards its tail and ends in a filled dot at the current position.
    ``opacity`` scales the whole overlay (0.5 = half-transparent tracks).
    """
    from PIL import ImageDraw

    P, L, _ = trails.shape
    if P == 0 or L == 0:
        return rgba
    H, W = rgba.shape[:2]
    flat = project_r3(trails.reshape(-1, 3), c2w, K).reshape(P, L, 3)

    ui = np.clip(flat[..., 0].astype(int), 0, W - 1)
    vi = np.clip(flat[..., 1].astype(int), 0, H - 1)
    on_screen = (flat[..., 0] >= 0) & (flat[..., 0] < W) & \
                (flat[..., 1] >= 0) & (flat[..., 1] < H) & (flat[..., 2] > 0)
    if depth is not None:
        surf = depth[vi, ui]
        # depth == 0 is background (nothing rendered there) -> always visible.
        on_screen &= (surf <= 0) | (flat[..., 2] <= surf + 0.02 * np.abs(surf) + 1e-3)

    img = Image.fromarray(rgba).convert("RGBA")
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for p in range(P):
        rgb = colors[p % len(colors)]
        for l in range(1, L):
            if not (on_screen[p, l] and on_screen[p, l - 1]):
                continue
            alpha = int(60 + 195 * (l / max(1, L - 1)))  # fade towards the tail
            draw.line(
                [(flat[p, l - 1, 0], flat[p, l - 1, 1]), (flat[p, l, 0], flat[p, l, 1])],
                fill=(*rgb, alpha), width=width,
            )
        if on_screen[p, L - 1] and head > 0:
            x, y = flat[p, L - 1, 0], flat[p, L - 1, 1]
            draw.ellipse([x - head, y - head, x + head, y + head], fill=(*rgb, 255))
    if opacity < 1.0:  # scale the whole overlay's alpha, keeping the fade shape
        r, g, b, a = layer.split()
        layer = Image.merge("RGBA", (r, g, b, a.point(lambda v: int(v * opacity))))
    return np.asarray(Image.alpha_composite(img, layer))


# Radius of the dot a trail ends in. Not an option: it is `draw_tracks`'s own default,
# named here only so the supersample path can scale it with the frame.
_TRACK_HEAD_PX = 3


def overlay_trails(frame: np.ndarray, trails: np.ndarray | None, upto: int,
                   c2w: np.ndarray, K: np.ndarray, depth: Any,
                   colors: list, opts: dict) -> np.ndarray:
    """Draw track trails up to frame index ``upto`` onto ``frame`` — the shared
    body of the ``orbit`` / ``synth_nvs`` overlay.  ``trails is None`` (overlay
    off) returns the frame untouched, so the depth read-back is skipped too."""
    if trails is None:
        return frame
    lo = 0 if not opts["track_trail"] else max(0, upto + 1 - opts["track_trail"])
    return draw_tracks(
        frame, trails[lo : upto + 1].transpose(1, 0, 2), c2w, K,
        depth.detach().cpu().numpy(), colors, width=opts["track_width"],
        head=opts.get("track_head", _TRACK_HEAD_PX), opacity=opts["track_opacity"],
    )


def scale_intrinsics(K: np.ndarray, src_hw: tuple[int, int],
                     dst_hw: tuple[int, int]) -> np.ndarray:
    """Rescale a 3x3 K between resolutions (pixel-center convention)."""
    if src_hw == dst_hw:
        return K
    sy, sx = dst_hw[0] / src_hw[0], dst_hw[1] / src_hw[1]
    out = np.asarray(K, dtype=np.float32).copy()
    out[0, 0] *= sx
    out[1, 1] *= sy
    out[0, 2] = (out[0, 2] + 0.5) * sx - 0.5
    out[1, 2] = (out[1, 2] + 0.5) * sy - 0.5
    return out


def centred_intrinsics(focal: float, W: int, H: int) -> np.ndarray:
    """K for a ``W x H`` canvas: one focal on both axes (square pixels), principal
    point at the centre.  Sole owner of that centring convention, so the fitted
    renders cannot disagree with :func:`fit_focal` about where the axis is."""
    return np.array([[focal, 0.0, (W - 1) * 0.5],
                     [0.0, focal, (H - 1) * 0.5],
                     [0.0, 0.0, 1.0]], np.float32)


def square_intrinsics(focal: float, side: int) -> np.ndarray:
    """:func:`centred_intrinsics` on a square canvas, so a turntable shows the
    object the same way at every azimuth."""
    return centred_intrinsics(focal, side, side)


def bounding_points(geom: "GaussianCloud | MeshData", max_points: int = 20000):
    """``(P, 3)`` positions + ``(P,)`` radii bounding the geometry's visible extent.

    Subsampled by stride for the framing fit -- the extremes that decide the
    focal survive subsampling, and the fit's safety margin absorbs the rest.
    For Gaussians the radius is 3 sigma of the widest axis, i.e. roughly where
    gsplat stops drawing a splat, so the fit bounds *rendered* pixels rather
    than centres; mesh vertices are exact and need no radius.
    """
    step = max(1, len(geom) // max_points)
    if isinstance(geom, MeshData):
        # Gaussians reach here already trimmed (`trim_outliers`); a mesh cannot be,
        # since dropping a vertex would tear its faces.  So the same radius cut is
        # applied to the FRAMING only -- a stray vertex still renders, it just does
        # not decide where the camera looks or how far away it sits.
        verts = geom.verts[::step]
        verts = verts[bulk_keep_mask(verts)].detach().cpu().numpy()
        return verts, np.zeros(len(verts), np.float32)
    means = geom.means[::step].detach().cpu().numpy()
    radii = 3.0 * geom.scales[::step].max(dim=1).values.detach().cpu().numpy()
    return means, radii


def trim_outliers(geom: "GaussianCloud",
                  factor: float = 1.5) -> tuple["GaussianCloud", int]:
    """Drop stray splats outside the cloud's bulk; returns ``(cloud, dropped)``.

    Four cuts, one per kind of stray a depth-unprojected cloud can carry, and all
    four rules live in ``core/utils/rendering.py`` so the turntable's framing and
    its render cannot desync.  They are independent -- each catches strays the
    others do not, which is why all four run:

    * ``bulk_keep_mask`` -- radius about the median, which catches a second
      CLUSTER stranded behind the object.
    * ``visible_bulk_mask`` -- the opacity-weighted per-axis box, which catches a
      diffuse HAZE the radius rule is blind to: a haze drags the radius percentile
      out with it, so the radius cut alone can drop nothing while the object's box
      is inflated and its centre pulled off the object.
    * ``oversize_splat_mask`` -- splats that are a large fraction of the object
      themselves.  Neither of the others can see these: they sit at the object's
      own depth and are opaque, so they are neither far nor faint, just far too
      big.
    * ``connected_bulk_mask`` -- compact CLUMPS stranded off the body, whose splats
      are individually unremarkable on all three counts above and only objectionable
      together.  A coherent reconstruction is ONE component, so this cannot fire on
      one.

    Why a turntable in particular needs this: it swings the camera right past the
    stranded splats, where they collapse the framing fit (``x / z`` with
    ``z -> 0``), pull the framed box centre off the object, and smear the frame
    with splats the size of the object.
    """
    keep = (bulk_keep_mask(geom.means, factor)
            & visible_bulk_mask(geom.means, geom.opacities, geom.scales))
    # The last two run on the SURVIVORS, because both measure against the object's
    # extent: a haze still in the cloud inflates that extent, raising the size bar
    # and coarsening the voxel grid the components are labelled on.
    idx = torch.nonzero(keep, as_tuple=True)[0]
    inner = (oversize_splat_mask(geom.means[idx], geom.scales[idx])
             & connected_bulk_mask(geom.means[idx], geom.opacities[idx],
                                   geom.scales[idx]))
    keep[idx[~inner]] = False
    dropped = int(len(geom) - keep.sum())
    if not dropped:
        return geom, 0
    return GaussianCloud(geom.means[keep], geom.quats[keep], geom.scales[keep],
                         geom.opacities[keep], geom.sh[keep]), dropped


def bounds_aabb(pts, radii):
    """``(lo, hi)`` of the axis-aligned box around one ``bounding_points`` entry,
    radii included so it bounds rendered pixels rather than centres, or None."""
    pts = np.asarray(pts, np.float64)
    if not len(pts):
        return None
    r = np.asarray(radii, np.float64).reshape(-1, 1)
    return (pts - r).min(axis=0), (pts + r).max(axis=0)


def fit_focal(bounds, cameras, W: int, H: int, margin: float = 0.04) -> float | None:
    """Largest focal that keeps every point inside a ``W x H`` frame, for every camera.

    Projection offset from the principal point is ``f * x / z``, i.e. linear in
    ``f``, so the tightest framing is just the available half-span divided by
    the worst ``(|x| + r) / z`` over all points and all cameras.  ``bounds`` is
    a list of ``(points, radii)`` -- one entry per distinct geometry the render
    will show, so a deforming object is framed for its whole motion and never
    changes apparent size mid-turn.  Returns None when nothing is in front of
    the camera.

    The two axes are fitted SEPARATELY and the tighter one wins: square pixels
    mean one focal serves both, but a non-square frame has a different half-span
    on each, so folding them into a single worst ratio (what a square-only fit
    can do) would overflow the short side.
    """
    half_x = (W * 0.5) * (1.0 - margin)
    half_y = (H * 0.5) * (1.0 - margin)
    worst_x = worst_y = 0.0
    for c2w in cameras:
        w2c = np.linalg.inv(np.asarray(c2w, np.float64))
        for pts, radii in bounds:
            cam = np.asarray(pts, np.float64) @ w2c[:3, :3].T + w2c[:3, 3]
            front = cam[:, 2] > 1e-6
            if not front.any():
                continue
            z, r = cam[front, 2], np.asarray(radii, np.float64)[front]
            worst_x = max(worst_x, float(((np.abs(cam[front, 0]) + r) / z).max()))
            worst_y = max(worst_y, float(((np.abs(cam[front, 1]) + r) / z).max()))
    fits = [h / w for h, w in ((half_x, worst_x), (half_y, worst_y)) if w > 0]
    return min(fits) if fits else None


def fit_square_focal(bounds, cameras, side: int, margin: float = 0.04) -> float | None:
    """:func:`fit_focal` on a square canvas -- what a turntable always renders into."""
    return fit_focal(bounds, cameras, side, side, margin)


# The R3 world up (OpenCV: +Y is down, so up is -Y), and the datasets whose world
# frame is known to have it.  Deliberately short: the axis cannot be recovered
# from a run's own cameras, so each entry is a claim about the dataset.
_WORLD_UP = np.array([0.0, -1.0, 0.0])
_WORLD_UP_DATASETS = {"gso"}


def load_run_config(final_dir: Path) -> dict:
    """The Hydra config snapshot beside ``final/``, or ``{}``.

    A run is otherwise rebuilt from ``final/`` alone; this is the one thing on
    disk that records what the run was OF -- its dataset, and whether it kept
    that dataset's cameras.  A hand-assembled ``final/`` with no snapshot
    answers ``{}``, and every caller has to work from there.
    """
    import yaml

    cfg_path = Path(final_dir).parent / "config.yaml"
    if not cfg_path.is_file():
        return {}
    try:
        return yaml.safe_load(cfg_path.read_text()) or {}
    except yaml.YAMLError:
        return {}


def has_dataset_world_up(final_dir: Path) -> bool:
    """Does THIS run's world frame have a known vertical?

    Two conditions, both recorded in the run's own ``config.yaml`` beside
    ``final/``: the dataset's world frame has a known up, and the run kept that
    dataset's cameras — predicted poses (map-anything) put the object in an
    arbitrary frame, where the dataset's up means nothing.  A run with no
    config (a hand-assembled ``final/``) answers False, the safe direction.
    """
    cfg = load_run_config(final_dir)
    return (cfg.get("dataset", {}).get("name") in _WORLD_UP_DATASETS
            and cfg.get("processing", {}).get("camera_poses_source") == "gt")


def resolve_up(spec: str, final_dir: Path) -> np.ndarray | None:
    """``-O orbit.up=...`` -> the orbit's vertical, or None for the camera's own."""
    spec = (spec or "auto").strip().lower()
    if spec == "camera":
        return None
    if spec == "auto":
        return _WORLD_UP if has_dataset_world_up(final_dir) else None
    if spec == "world":
        return _WORLD_UP
    # ValueError, not SystemExit: this also runs inside FINAL, where killing the
    # process would skip the timing summary at the end of a multi-hour run.
    parts = spec.split(",")
    try:
        if len(parts) != 3:
            raise ValueError
        return np.array([float(p) for p in parts], dtype=np.float64)
    except ValueError:
        raise ValueError(
            f"orbit.up wants auto|camera|world|'x,y,z', got {spec!r}") from None


# ===========================================================================
# Render registry
# ===========================================================================


@dataclass
class Option:
    """One renderer-specific knob, settable via ``-O [render.]key=value``."""

    type: str          # "int" | "float" | "bool" | "str" | "int?"
    default: Any
    help: str = ""


@dataclass
class Renderer:
    name: str
    help: str
    fn: Callable[[FinalRun, "RenderContext"], None]
    options: dict[str, Option]
    needs_gpu: bool
    # Where under {final}/viz/ this render writes. Defaults to its name; the
    # track-overlay family sets a nested path so the annotated variants group
    # under one folder ("track_overlays/train_views") instead of scattering
    # top-level names that read like the clean renders they are derived from.
    out_sub: str = ""


RENDERERS: dict[str, Renderer] = {}


def renderer(name: str, help: str, options: dict[str, Option] | None = None,
             needs_gpu: bool = True, out_sub: str = ""):
    """Register a render under ``name``; the function gets ``(run, ctx)``.

    Outputs go through ``ctx.write_png`` / ``ctx.write_video``, which own the
    ``--dry-run`` / ``--overwrite`` protocol and the tally.  Options declared
    here are exposed as ``-O <name>.<key>=<value>`` and shown by ``--list``.
    """
    def deco(fn):
        RENDERERS[name] = Renderer(name, help, fn, dict(options or {}), needs_gpu,
                                   out_sub or name)
        return fn
    return deco


@dataclass
class RenderContext:
    """Everything a renderer needs besides the run itself, plus its output sinks."""

    out_dir: Path                 # {viz_root}/{render.out_sub}
    device: str
    frame_keys: list[FrameKey]    # after --frames / --views filtering
    objects: list[int]            # after --objects filtering
    H: int
    W: int
    src_hw: tuple[int, int]       # the run's own resolution, for scale_intrinsics
    bg: np.ndarray                # (3,) float RGB the foreground composites on
    fps: int
    overwrite: bool
    dry_run: bool
    opts: dict[str, Any]
    written: list[Path] = field(default_factory=list)

    def path(self, *parts: str) -> Path:
        return self.out_dir.joinpath(*parts)

    def wants(self, path: Path) -> bool:
        """False when this output is already done, or we are only planning.

        Either way the path counts as an output, so ``--dry-run`` reports the
        full set and a re-run reports what is already there.
        """
        if self.dry_run or (path.exists() and not self.overwrite):
            self.written.append(path)
            return False
        return True

    def write_png(self, path: Path, arr: np.ndarray) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(path)
        self.written.append(path)
        return path

    def write_video(self, path: Path, frames: Iterable[np.ndarray | Path]) -> Path | None:
        """Encode RGBA frames (arrays, or PNGs to stream off disk) to H.264 mp4.

        Frames are flattened onto ``ctx.bg`` and fed to the encoder one at a
        time, so a long orbit at high resolution never holds the whole clip.
        """
        if not self.wants(path):
            return None
        import imageio.v2 as imageio

        stream = self._rgb_frames(frames)
        first = next(stream, None)
        if first is None:
            return None
        h, w = first.shape[0] - first.shape[0] % 2, first.shape[1] - first.shape[1] % 2
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(path), fps=self.fps, codec="libx264",
                                    quality=8, macro_block_size=None)
        n = 0
        try:
            for frame in (first, *stream):
                if frame.shape[:2] != first.shape[:2]:
                    continue  # a stale PNG at another resolution
                writer.append_data(frame[:h, :w])
                n += 1
        finally:
            writer.close()
        if n < 2:  # a 1-frame clip is not worth a video file
            path.unlink(missing_ok=True)
            return None
        self.written.append(path)
        return path

    def _rgb_frames(self, frames: Iterable[np.ndarray | Path]) -> Iterator[np.ndarray]:
        for frame in frames:
            if isinstance(frame, Path):
                if not frame.is_file():
                    continue
                with Image.open(frame) as im:
                    frame = np.asarray(im.convert("RGBA"))
            yield flatten_alpha(frame, self.bg)

    def write_per_view_videos(self, prefix: str, keys: list[FrameKey]) -> None:
        """One mp4 per view from the per-frame PNGs a render just wrote:
        ``{prefix}.mp4`` for view 0, ``{prefix}_vNN.mp4`` for the rest.  PNGs are
        streamed back off disk so frames skipped this run still make it in."""
        for view, vkeys in group_by_view(keys).items():
            suffix = "" if view == 0 else f"_v{view:02d}"
            self.write_video(self.path(f"{prefix}{suffix}.mp4"),
                             [self.path(f"{frame_key_stem(k)}.png") for k in vkeys])


def _coerce(value: str, kind: str) -> Any:
    if kind.endswith("?") and value.lower() in ("none", "null", ""):
        return None
    base = kind.rstrip("?")
    if base == "bool":
        if value.lower() in ("1", "true", "yes", "on"):
            return True
        if value.lower() in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"expected a boolean, got {value!r}")
    return {"int": int, "float": float, "str": str}[base](value)


def _split_opt(item: str) -> tuple[str, str, str]:
    """``"[render.]key=value"`` -> ``(render_or_empty, key, value)``."""
    key, sep, value = item.partition("=")
    if not sep:
        raise SystemExit(f"--opt needs KEY=VALUE, got {item!r}")
    target, _, name = key.rpartition(".")
    return target, name, value


def resolve_options(rdr: Renderer, raw: list[str]) -> dict[str, Any]:
    """Merge ``-O`` assignments over the defaults; scoped beats global."""
    opts = {k: o.default for k, o in rdr.options.items()}
    scoped: dict[str, Any] = {}
    for item in raw:
        target, name, value = _split_opt(item)
        if target and target != rdr.name:
            continue                              # another render's knob
        if name not in rdr.options:
            if not target:
                continue                          # a global knob owned elsewhere
            raise SystemExit(f"unknown option {name!r} for render {rdr.name!r}; "
                             f"available: {sorted(rdr.options) or 'none'}")
        (scoped if target else opts)[name] = _coerce(value, rdr.options[name].type)
    return {**opts, **scoped}


def check_options_are_claimed(selected: list[Renderer], raw: list[str]) -> None:
    """Reject a ``-O`` no selected render owns, instead of silently dropping it."""
    names = {r.name for r in selected}
    for item in raw:
        target, name, _value = _split_opt(item)
        if target and target not in names:
            raise SystemExit(f"-O {item}: no selected render named {target!r} "
                             f"(selected: {sorted(names)})")
        if not target and not any(name in r.options for r in selected):
            raise SystemExit(f"-O {item}: no selected render has an option "
                             f"{name!r} (see --list)")


# ===========================================================================
# Built-in renders
# ===========================================================================


def alpha_bbox(a: np.ndarray) -> tuple[int, int, int, int] | None:
    """``(x0, y0, x1, y1)`` of what is opaque enough to count as foreground, or None."""
    ys, xs = np.where(a > 8)
    return (xs.min(), ys.min(), xs.max(), ys.max()) if len(xs) else None


def sequence_crop_box(boxes, pad: float, H: int, W: int) -> tuple[int, int, int, int]:
    """``(x0, y0, x1, y1)``: one crop covering every box in ``boxes``, at the frame's
    aspect ratio.

    ``boxes`` are ``(x0, y0, x1, y1)`` foreground extents (see :func:`alpha_bbox`), one
    per frame of a sequence, or one per CELL of a figure row when the caller wants a box
    every run in the row shares. Their union, padded by ``pad`` x its longer side,
    then GROWN (never squeezed) to ``W/H`` so the crop rescales without distorting, and
    shifted -- not shrunk -- back inside the frame when that growth overhangs an edge.
    One box for every frame is the point: a per-frame box would make the object jitter
    in scale across a row of tiles meant to differ only in timestep.

    An empty ``boxes`` (nothing rendered at all) gives the full frame back.
    """
    boxes = list(boxes)
    if not boxes:
        return 0, 0, W - 1, H - 1
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes); y1 = max(b[3] for b in boxes)
    cw, ch = x1 - x0 + 1, y1 - y0 + 1
    m = pad * max(cw, ch)
    cw, ch = cw + 2 * m, ch + 2 * m
    ar = W / H
    if cw / ch < ar:                      # too tall -> widen
        cw = ch * ar
    else:                                 # too wide -> heighten
        ch = cw / ar
    s = min(1.0, W / cw, H / ch)          # a box bigger than the frame scales down whole
    cw, ch = cw * s, ch * s
    cx, cy = (x0 + x1 + 1) / 2, (y0 + y1 + 1) / 2
    nx = min(max(cx - cw / 2, 0), W - cw)
    ny = min(max(cy - ch / 2, 0), H - ch)
    return int(round(nx)), int(round(ny)), int(round(nx + cw)) - 1, int(round(ny + ch)) - 1


def fit_to_native(frame: np.ndarray, W: int, H: int) -> np.ndarray:
    """Downscale a cropped frame back to the run's own ``W x H``. NEVER upscales.

    Supersampling exists to feed the crop, not to ship a bigger file: rendering at 2x
    and resolving back down is ordinary supersampling, so the tile lands at the run's
    resolution with the aliasing of a 2x render rather than a 1x one. Output is
    therefore always <= 1.0x native, and a crop already at or below it (a small object,
    where the box is tighter than the supersample covers) is passed through untouched
    rather than blown up to fill the size.

    The crop carries the frame's aspect ratio by construction, so resizing to exactly
    ``(W, H)`` is not a distortion -- and it removes the sub-pixel aspect differences
    that integer rounding leaves between one sequence's box and another's, which would
    otherwise print as slightly unequal tile heights across a figure row.
    """
    if frame.shape[1] <= W:
        return frame
    return np.asarray(Image.fromarray(frame).resize((W, H), Image.LANCZOS))


def _supersampled_size(ctx: RenderContext, trails) -> tuple[int, int, dict]:
    """``(H, W, draw_opts)`` to render at, for this render's ``supersample`` factor.

    Rendering above native and resolving back down in :func:`fit_to_native` is ordinary
    supersampling: the tile ships at the run's own resolution with the aliasing of a 2x
    render.  Track width and head scale with the factor, or the trails would come out
    proportionally thinner once the frame is resolved down.
    """
    ss = float(ctx.opts.get("supersample", 1.0))
    if ss == 1.0:
        return ctx.H, ctx.W, ctx.opts
    H, W = max(1, round(ctx.H * ss)), max(1, round(ctx.W * ss))
    log(f"    supersample {ss:g}x: rendering {W}x{H} (native {ctx.W}x{ctx.H})")
    if trails is None:
        return H, W, ctx.opts
    return H, W, {**ctx.opts,
                  "track_width": max(1, round(ctx.opts["track_width"] * ss)),
                  "track_head": max(1, round(_TRACK_HEAD_PX * ss))}


def _write_cropped(ctx: RenderContext, pending: list, H: int, W: int) -> None:
    """Fit ONE box to every held frame, crop to it, resolve to native, write.

    The box is fitted to the alpha AS DRAWN, trails included -- fitting it to the bare
    silhouette would slice the tails off the very overlay these renders exist for.
    """
    if not pending:
        return
    x0, y0, x1, y1 = sequence_crop_box(
        [b for b in (alpha_bbox(f[..., 3]) for _, f in pending) if b],
        ctx.opts["crop_pad"], H, W)
    cw, ch = x1 - x0 + 1, y1 - y0 + 1
    out_w, out_h = (ctx.W, ctx.H) if cw > ctx.W else (cw, ch)
    log(f"    crop: {cw}x{ch} of {W}x{H} at ({x0},{y0}) — "
        f"{cw * ch / (W * H):.0%} of the frame, {cw / ctx.W:.2f}x native "
        f"-> written at {out_w}x{out_h}")
    for out, frame in pending:
        ctx.write_png(out, fit_to_native(frame[y0:y1 + 1, x0:x1 + 1], ctx.W, ctx.H))


def _render_from_train_cameras(run: FinalRun, ctx: RenderContext, *,
                               tracks: bool) -> None:
    """Re-render the posed scene from each dataset (input/train) camera.

    Shared body of the two renders below.  ``tracks`` is the ONLY difference, and it
    is a property of the render, not a knob on a shared one: the output dir is
    ``viz/<render name>``, so an annotated frame must come from a differently-named
    render or it lands in the clean one's folder.
    """
    if not run.has_renderable_geometry:
        log("    ! run has no gaussians or meshes — nothing to re-render")
        return
    use_mesh = run.prefers_mesh
    space = ctx.opts["space"]
    if use_mesh and space != "world":
        log("    (mesh renders are world-space only; ignoring space=camera)")
        space = "world"

    # A dynamic run whose stored geometry never changes (constant poses, no
    # per-frame asset) would emit identical frames -- the deformation is only in
    # the run's own renders_train/. Say so instead of implying it is static.
    per_frame_asset = run.has_perframe_meshes if use_mesh else run.has_perframe_gaussians
    per_frame = run.poses_vary or per_frame_asset
    if run.is_dynamic and not per_frame:
        log("    ! stored geometry is static across frames — every re-render "
            "is identical; see the method's own output with "
            "`-r contact_sheet -O source=renders_train`")

    # Tracks live in `compose(space="world")` space, so a camera-space render has
    # nothing to project them against -- the same reason `orbit` rebases them.
    # Otherwise draw them only when the run shipped a tracks_2d.npz.  A per-frame
    # prediction with no cross-frame correspondence writes none and gets a bare
    # render -- still written, and cropped identically, so a figure row can mix
    # the two.
    show_tracks = tracks and ctx.opts["tracks"] and space == "world" and run.tracks is not None
    if tracks and ctx.opts["tracks"] and not show_tracks:
        log("    (skipping the 3D track overlay: tracks are world-space)" if run.tracks
            is not None else "    (no 3D tracks on disk — rendering without them)")
    trails = _track_trails(run, ctx, ctx.frame_keys) if show_tracks else None
    colors = track_colors(trails.shape[1]) if trails is not None else []

    # Cropping needs the whole sequence before it can write ANY frame, so frames are
    # held in memory (~10 MB for 16 at 518x294) and `wants` is ignored: a partial
    # re-render would fit the box to a subset and silently reframe the rest.
    crop = ctx.opts.get("crop", False) and not ctx.dry_run
    H, W, draw_opts = _supersampled_size(ctx, trails)
    pending: list = []

    for i, key in enumerate(ctx.frame_keys):
        cam = run.camera(key)
        if cam is None or cam.K is None:
            log(f"    ! no camera with intrinsics for {key}, skipping")
            continue
        out = ctx.path(f"{frame_key_stem(key)}.png")
        if not ctx.wants(out) and not crop:
            continue
        if use_mesh:
            geom = run.compose_mesh(key, device=ctx.device, objects=ctx.objects)
        else:
            geom = run.compose(key, device=ctx.device, space=space,
                               objects=ctx.objects, use_perframe=ctx.opts["perframe"])
        if geom is None:
            continue
        c2w = cam.c2w if space == "world" else np.eye(4, dtype=np.float32)
        K = scale_intrinsics(cam.K, ctx.src_hw, (H, W))
        render = render_mesh if use_mesh else render_cloud
        rgb, alpha, depth = render(geom, c2w, K, H, W, ctx.bg)
        frame = to_rgba(rgb, alpha)
        if trails is not None:
            frame = overlay_trails(frame, trails, i, c2w, K, depth, colors, draw_opts)
        if crop:
            pending.append((out, frame))   # held: the box needs every frame first
        else:
            ctx.write_png(out, frame)

    _write_cropped(ctx, pending, H, W)

    if ctx.opts["video"]:
        # Named for the render, which is also the folder -- so the clean and the
        # annotated clip cannot collide.
        ctx.write_per_view_videos(ctx.out_dir.name, ctx.frame_keys)


# Every render that can draw a trail takes the same five knobs, and the figure
# renders take the same three crop knobs on top. Defined once, so `track_overlay` and
# `track_overlay_nvs` always share their defaults: a default drifting between two
# copies would reframe one subrow of a figure row against the other.
_TRACK_DRAW_OPTIONS = {
    "track_points": Option("int", 40, "cap on tracked points drawn"),
    "track_subsample": Option("float", 0.5, "fraction of the capped points to "
                              "keep, spread evenly through space (1 = all)"),
    "track_trail": Option("int", 0, "trail length in frames (0 = whole history)"),
    "track_width": Option("int", 2, "trail line width in px"),
    "track_opacity": Option("float", 1.0, "overlay opacity (1 = opaque)"),
}
# `track_overlay_nvs` takes the draw knobs WITHOUT this toggle: it exists to draw the
# overlay and its body never consults one, so offering it would be a dead option.
_TRACK_OPTIONS = {
    "tracks": Option("bool", True, "overlay final/tracks_2d.npz 3D tracks "
                     "when the run has them"),
    **_TRACK_DRAW_OPTIONS,
}
_FIGURE_CROP_OPTIONS = {
    "crop": Option("bool", False, "trim every frame to ONE box covering the object "
                   "across the whole sequence, at the frame's aspect ratio. OFF by "
                   "default: figures are cropped later, to a box shared by the "
                   "whole row; a per-run box here would pre-trim the pixels that "
                   "box needs"),
    "crop_pad": Option("float", 0.05, "padding around that box, as a fraction of "
                       "its longer side"),
    "supersample": Option("float", 2.0, "render at this multiple of the run's "
                          "resolution, so whoever crops has the pixels to do it with"),
}

_TRAIN_CAMERA_OPTIONS = {
    "video": Option("bool", True, "also encode the frames to mp4"),
    "space": Option("str", "world", "pose space: world | camera"),
    "perframe": Option("bool", True, "use per-frame warped PLYs when present"),
}


@renderer(
    "train_views",
    "Re-render the posed scene from each dataset (input/train) camera. Clean "
    "pixels, nothing drawn on top -- see `track_overlay` for the annotated twin.",
    options=dict(_TRAIN_CAMERA_OPTIONS),
)
def render_train_views(run: FinalRun, ctx: RenderContext) -> None:
    _render_from_train_cameras(run, ctx, tracks=False)


@renderer(
    "track_overlay",
    "The train-camera render with the run's 3D tracks drawn over it, into "
    "viz/track_overlays/train_views/. A FIGURE asset: annotated and supersampled, so "
    "it is deliberately kept out of `train_views` and away from every dir an evaluation "
    "reads. A run with no tracks_2d.npz still renders -- same viewpoint and resolution, "
    "just nothing drawn on top, so the pair can be cropped to one box.",
    options={
        **_TRAIN_CAMERA_OPTIONS,
        **_TRACK_OPTIONS,
        **_FIGURE_CROP_OPTIONS,
    },
    out_sub="track_overlays/train_views",
)
def render_track_overlay(run: FinalRun, ctx: RenderContext) -> None:
    _render_from_train_cameras(run, ctx, tracks=True)


@renderer(
    "synth_nvs",
    "Per-timestamp synthesized novel views: the world-placed scene (root motion "
    "KEPT) seen from FOUR fixed offsets off each frame's train camera (\u00b1azimuth, "
    "\u00b1elevation), at the train intrinsics. Reproduces the pipeline's "
    "renders_synth_nvs/ off disk, filenames included.",
    options={
        "azimuth": Option("float", 30.0, "horizontal magnitude, deg (two views at \u00b1this)"),
        "azimuth_step": Option("float", 0.0, "extra azimuth per timestamp (0 = fixed viewpoint)"),
        "elevation": Option("float", 30.0, "vertical magnitude, deg (two views at \u00b1this)"),
        "video": Option("bool", True, "also encode the per-timestamp frames to mp4"),
        **_TRACK_OPTIONS,
    },
)
def render_synth_nvs(run: FinalRun, ctx: RenderContext) -> None:
    # Unlike `orbit` (one pinned timestamp swept through azimuth), this steps the
    # timestamp and keeps root motion — the object moves through the scene while
    # the viewpoint holds a fixed offset off the train camera, at the train K.
    if not run.has_renderable_geometry:
        log("    ! run has no gaussians or meshes — nothing to re-render")
        return
    from genia.core.utils.eval_assets_export import (
        SYNTH_NVS_OFFSETS, _synth_orbit_c2w,
    )

    opts = ctx.opts
    az0, az_step, el = opts["azimuth"], opts["azimuth_step"], opts["elevation"]
    use_mesh = run.prefers_mesh
    render = render_mesh if use_mesh else render_cloud
    keys = ctx.frame_keys

    # Tracks are stored posed (root motion kept), which is exactly this render's
    # space — so no rebasing (ref_key=None), unlike `orbit`. Draw them only when
    # the run actually shipped a tracks_2d.npz: per-frame predictions with no
    # cross-frame correspondence don't write one — their per-frame gaussians are
    # independent shapes, not tracks.
    show_tracks = opts["tracks"] and run.tracks is not None
    trails = _track_trails(run, ctx, keys) if show_tracks else None
    colors = track_colors(trails.shape[1]) if trails is not None else []

    # This render REPRODUCES the pipeline's renders_synth_nvs/ off disk: native
    # resolution, uncropped, all four tags. `track_overlay_nvs` is the figure variant.

    for i, key in enumerate(keys):
        cam = run.camera(key)
        if cam is None or cam.K is None:
            log(f"    ! no camera with intrinsics for {key}, skipping")
            continue
        # `wants` RECORDS a skipped path, so probe each output exactly once —
        # asking twice would double-count it in the run summary, and an `any()`
        # short-circuit would leave later tags unrecorded.
        wanted = {tag: ctx.wants(ctx.path(f"{frame_key_stem(key)}_{tag}.png"))
                  for tag, _, _ in SYNTH_NVS_OFFSETS}
        if not any(wanted.values()):
            continue
        if use_mesh:
            geom = run.compose_mesh(key, device=ctx.device, objects=ctx.objects)
        else:
            geom = run.compose(key, device=ctx.device, space="world",
                               objects=ctx.objects)
        if geom is None:
            continue
        centroid = geom.centroid()
        K = scale_intrinsics(cam.K, ctx.src_hw, (ctx.H, ctx.W))
        # The train camera orbited about the world-placed object centroid at the
        # SAME distance (radius = |eye − centroid|), once per offset: the eye at
        # (0, 0) would be the train eye itself, so the magnitudes set how novel
        # each of the four viewpoints is.
        for tag, az_sign, el_sign in SYNTH_NVS_OFFSETS:
            if not wanted[tag]:
                continue
            c2w = _synth_orbit_c2w(cam.c2w, centroid,
                                   az_sign * az0 + az_step * key.frame,
                                   el_sign * el)
            rgb, alpha, depth = render(geom, c2w, K, ctx.H, ctx.W, ctx.bg)
            frame = overlay_trails(to_rgba(rgb, alpha), trails, i, c2w, K, depth,
                                   colors, opts)
            ctx.write_png(ctx.path(f"{frame_key_stem(key)}_{tag}.png"), frame)

    if opts["video"]:
        # One mp4 per offset — `write_per_view_videos` names PNGs by frame key
        # alone, which cannot see the tag.
        for tag, _, _ in SYNTH_NVS_OFFSETS:
            ctx.write_video(
                ctx.path(f"synth_nvs_{tag}.mp4"),
                [ctx.path(f"{frame_key_stem(k)}_{tag}.png") for k in keys],
            )


@renderer(
    "track_overlay_nvs",
    "ONE synthesized novel view per timestamp with the run's 3D tracks drawn over it, "
    "into viz/track_overlays/synth_nvs/. The figure twin of `track_overlay`, seen from "
    "off-axis instead of from the train camera. A FIGURE asset -- annotated and "
    "supersampled -- which is why it is NOT `synth_nvs`: that one stays a faithful, "
    "native-resolution reproduction of the scored renders_synth_nvs/. A run with no "
    "tracks_2d.npz still renders, just with nothing drawn on top.",
    options={
        # Chosen for legibility, NOT to match one of the four scored offsets: at 40/0
        # this is none of renders_synth_nvs/'s tags (azpos is the nearest, at +30/0).
        "azimuth": Option("float", 40.0, "horizontal offset off the train camera, deg"),
        "azimuth_step": Option("float", 0.0, "extra azimuth per timestamp"),
        "elevation": Option("float", 0.0, "vertical offset off the train camera, deg"),
        "video": Option("bool", True, "also encode the frames to mp4"),
        **_TRACK_DRAW_OPTIONS,
        **_FIGURE_CROP_OPTIONS,
    },
    out_sub="track_overlays/synth_nvs",
)
def render_track_overlay_nvs(run: FinalRun, ctx: RenderContext) -> None:
    """One synthesized view per timestamp, offset from that frame's train camera.

    ONE offset, not the four `synth_nvs` writes: a figure shows a single novel view
    per run, always the same one so the row stays a comparison, and rendering the
    other three would be three quarters of the cost thrown away.  Root motion is kept,
    as in `synth_nvs` -- the object moves through the scene while the viewpoint holds a
    fixed offset -- so the tracks need no rebasing.

    The offset follows each frame's OWN train camera rather than orbiting a fixed one,
    so on a moving-camera sequence the viewpoint moves with it.  That is `synth_nvs`'s
    convention, and it is what lets the tracks be drawn without rebasing.
    """
    if not run.has_renderable_geometry:
        log("    ! run has no gaussians or meshes — nothing to re-render")
        return
    from genia.core.utils.eval_assets_export import _synth_orbit_c2w

    opts = ctx.opts
    az0, az_step, el = opts["azimuth"], opts["azimuth_step"], opts["elevation"]
    use_mesh = run.prefers_mesh
    render = render_mesh if use_mesh else render_cloud
    keys = ctx.frame_keys

    trails = _track_trails(run, ctx, keys)
    colors = track_colors(trails.shape[1]) if trails is not None else []

    crop = opts["crop"] and not ctx.dry_run
    H, W, draw_opts = _supersampled_size(ctx, trails)
    pending: list = []

    for i, key in enumerate(keys):
        cam = run.camera(key)
        if cam is None or cam.K is None:
            log(f"    ! no camera with intrinsics for {key}, skipping")
            continue
        out = ctx.path(f"{frame_key_stem(key)}.png")
        if not ctx.wants(out) and not crop:
            continue
        geom = (run.compose_mesh(key, device=ctx.device, objects=ctx.objects) if use_mesh
                else run.compose(key, device=ctx.device, space="world",
                                 objects=ctx.objects))
        if geom is None:
            continue
        K = scale_intrinsics(cam.K, ctx.src_hw, (H, W))
        c2w = _synth_orbit_c2w(cam.c2w, geom.centroid(),
                               az0 + az_step * key.frame, el)
        rgb, alpha, depth = render(geom, c2w, K, H, W, ctx.bg)
        frame = overlay_trails(to_rgba(rgb, alpha), trails, i, c2w, K, depth,
                               colors, draw_opts)
        if crop:
            pending.append((out, frame))
        else:
            ctx.write_png(out, frame)

    _write_cropped(ctx, pending, H, W)

    if opts["video"]:
        ctx.write_per_view_videos("track_overlay_nvs", ctx.frame_keys)


@renderer(
    "track_overlay_test",
    "The run's 3D tracks drawn over a render from each HELD-OUT TEST camera, into "
    "viz/track_overlays/test_views/. The counterpart of `track_overlay_nvs` for the "
    "datasets that ship a GT test split (gso, oursactionbench, co3d): the novel view "
    "is a real one the method never saw, so there is no reason to synthesize one. "
    "A FIGURE asset -- annotated and supersampled -- which is why it is not "
    "`renders_test/`, the dir the NVS metrics are scored from.",
    options={
        "video": Option("bool", True, "also encode the frames to mp4"),
        **_TRACK_DRAW_OPTIONS,
        **_FIGURE_CROP_OPTIONS,
    },
    out_sub="track_overlays/test_views",
)
def render_track_overlay_test(run: FinalRun, ctx: RenderContext) -> None:
    """One render per held-out test camera, with the tracks drawn on.

    The cameras come from the dataset (see :func:`resolve_test_cameras`), not from
    ``poses.json`` -- a run records only the cameras it reconstructed from.  Root
    motion is kept, as in ``synth_nvs``, so the tracks need no rebasing; on
    OursActionBench each timestamp has its OWN test camera, so the viewpoint steps
    with the sequence exactly as ``renders_test/`` does.
    """
    if not run.has_renderable_geometry:
        log("    ! run has no gaussians or meshes — nothing to re-render")
        return
    # Resolved over the WHOLE run (the per-frame resolvers pair a camera with each
    # of the dataset's timestamps), then filtered to the keys this context renders,
    # so `--frames` narrows the overlay the way it narrows every other render.
    selected = set(ctx.frame_keys)
    views = [v for v in resolve_test_cameras(run, ctx.src_hw) if v.key in selected]
    if not views:
        return

    opts = ctx.opts
    use_mesh = run.prefers_mesh
    render = render_mesh if use_mesh else render_cloud

    # `_track_trails` indexes the trail by position in the keys it was built over,
    # so build it over the keys these cameras look at, in the same order.
    trails = _track_trails(run, ctx, [v.key for v in views])
    colors = track_colors(trails.shape[1]) if trails is not None else []

    crop = opts["crop"] and not ctx.dry_run
    H, W, draw_opts = _supersampled_size(ctx, trails)
    pending: list = []

    for i, view in enumerate(views):
        out = ctx.path(f"{view.stem}.png")
        if not ctx.wants(out) and not crop:
            continue
        key = view.key
        geom = (run.compose_mesh(key, device=ctx.device, objects=ctx.objects) if use_mesh
                else run.compose(key, device=ctx.device, space="world",
                                 objects=ctx.objects))
        if geom is None:
            continue
        K = scale_intrinsics(np.asarray(view.K, dtype=np.float32), ctx.src_hw, (H, W))
        c2w = np.asarray(view.c2w, dtype=np.float32)
        rgb, alpha, depth = render(geom, c2w, K, H, W, ctx.bg)
        frame = overlay_trails(to_rgba(rgb, alpha), trails, i, c2w, K, depth,
                               colors, draw_opts)
        if crop:
            pending.append((out, frame))
        else:
            ctx.write_png(out, frame)

    _write_cropped(ctx, pending, H, W)

    if opts["video"]:
        # Named by the render, and ordered by the views rather than by FrameKey:
        # a static split's "frames" are viewpoints, not timestamps.
        ctx.write_video(ctx.path("track_overlay_test.mp4"),
                        [ctx.path(f"{v.stem}.png") for v in views])


@dataclass
class _CameraSequence:
    """The three fields the test-camera resolvers read off a ``Sequence``.

    They want a live pipeline object; a finished run has only ``final/`` plus its
    config snapshot.  Everything they need from it is the render size and the
    frame keys, both of which the run knows -- so a shim is enough, and the
    resolvers stay the single definition of where each dataset's test cameras
    come from rather than being re-derived here.
    """

    H: int
    W: int
    frame_keys: list[FrameKey]


def _rebase_c2w(run: FinalRun, key: FrameKey, input_c2w, test_c2w) -> np.ndarray:
    """A dataset-world camera, expressed in the space ``compose(space="world")`` builds.

    The two are the same frame for every dataset but OursActionBench, whose
    ``FrameData.c2w`` is identity: its fitted input camera is baked into the
    per-frame OBJECT pose instead, so a run's "world" is that input camera's own
    space while ``camera.json``'s test cameras live in the OAB world.  Composing
    the run's own c2w with the input camera's inverse carries one to the other --
    which is the read-only half of what ``export_oursactionbench_eval_assets``
    does when it lifts the geometry the other way before rendering.
    """
    cam = run.camera(key)
    run_c2w = (np.eye(4) if cam is None else np.asarray(cam.c2w)).astype(np.float64)
    return (run_c2w @ np.linalg.inv(np.asarray(input_c2w, np.float64))
            @ np.asarray(test_c2w, np.float64)).astype(np.float32)


def _src_frame(src: str) -> int:
    """The timestamp a resolver's ``renders_{train,test}/{f}.png`` names.

    The resolvers pad it differently per dataset (OAB test is 2-digit, train
    3-digit), which is exactly why the pairing below goes through this rather
    than through position in the returned list.
    """
    return int(Path(src).stem)


class TestView(NamedTuple):
    """One held-out test camera, ready to render: where to write it, which
    timestamp's geometry it sees, and its (K, c2w) in the run's own world."""

    stem: str
    key: FrameKey
    K: np.ndarray
    c2w: np.ndarray


def resolve_test_cameras(run: FinalRun, hw: tuple[int, int]) -> list[TestView]:
    """This run's held-out test views, with intrinsics scaled to ``hw``.

    Empty for a dataset with no GT test split, when the run has no config snapshot
    to name that dataset, or when the dataset itself is not on disk (the
    cameras live beside the data, not in ``final/``) -- every one of which is a
    reason to render nothing rather than to guess a viewpoint.
    """
    if not run.frame_keys:
        return []
    from genia.core.utils import eval_assets_export as eae

    # The three datasets whose FINAL writes a SCORED `renders_test/`
    # (`genia.core.utils.config.GT_TEST_VIEW_DATASETS`), through the same resolvers
    # `core/utils/colmap_export.py` uses -- so the overlay looks through exactly
    # the cameras the NVS metrics were computed from.  The flag says whether the
    # dataset gives ONE test camera per timestamp (OAB) or a fixed set of views
    # for the whole run (GSO / CO3D).
    sources = {
        "gso": (eae.gso_test_cameras, False),
        "co3d": (eae.co3d_test_cameras, False),
        "oursactionbench": (eae.oursactionbench_test_cameras, True),
    }
    cfg_raw = load_run_config(run.dir)
    name = (cfg_raw.get("dataset") or {}).get("name")
    if name not in sources:
        log(f"    ! {name or 'this run'} has no GT test split — no test cameras")
        return []
    resolver, per_frame = sources[name]
    if name == "co3d" and (cfg_raw.get("processing") or {}).get(
            "camera_poses_source") != "gt":
        # The predicted-pose branch reads the target cameras back off a live
        # Sequence, which a finished run cannot rebuild.
        log("    ! CO3D with predicted cameras — test cameras need the live run")
        return []
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(cfg_raw)
    shim = _CameraSequence(*hw, run.frame_keys)
    try:
        cams = resolver(cfg, shim)
    except (FileNotFoundError, OSError, KeyError) as exc:
        log(f"    ! could not read {name} test cameras ({exc}) — skipping")
        return []
    if not cams:
        log(f"    ! {name} exposes no test cameras for this scene")
        return []
    if not per_frame:
        # A static split: every view sees the run's single timestamp, so the stem
        # comes from the render each camera belongs to (`renders_test/010.png`).
        key = run.frame_keys[0]
        return [TestView(Path(src).stem, key, K, c2w) for K, c2w, src, _ in cams]
    # Per-frame: pair by TIMESTAMP, not by position in the two lists. Both
    # resolvers silently skip a frame camera.json does not cover, and a positional
    # zip would then hand one frame's test camera another frame's input camera.
    keys = {k.frame: k for k in run.frame_keys if k.view == run.views[0]}
    inputs = {_src_frame(src): c2w
              for _, c2w, src, _ in eae.oursactionbench_input_cameras(cfg, shim)}
    out = []
    for K, c2w, src, _ in cams:
        frame = _src_frame(src)
        key, input_c2w = keys.get(frame), inputs.get(frame)
        if key is None or input_c2w is None:
            continue
        out.append(TestView(frame_key_stem(key), key, K,
                            _rebase_c2w(run, key, input_c2w, c2w)))
    return out


def farthest_point_indices(pts: np.ndarray, k: int) -> np.ndarray:
    """``k`` indices of ``pts`` (N,3) spread evenly through space by greedy
    farthest-point sampling.  Deterministic (seeded at index 0), so the same
    tracks are kept across frames and reruns.  Returns sorted indices."""
    n = len(pts)
    if k >= n:
        return np.arange(n)
    pts = np.asarray(pts, dtype=np.float64)
    chosen = [0]
    dist = np.linalg.norm(pts - pts[0], axis=1)
    for _ in range(1, k):
        i = int(dist.argmax())
        chosen.append(i)
        dist = np.minimum(dist, np.linalg.norm(pts - pts[i], axis=1))
    return np.array(sorted(chosen))


# An anchor counts as ON the object when the radius holding its `_ANCHOR_K` nearest
# drawn points is within `_ANCHOR_SPARSITY` times the MEDIAN such radius over all
# anchors.  Self-calibrating, so it needs no length scale and no per-run tuning: it
# asks whether an anchor sits somewhere as dense as the object typically is.
#
# Distance to the NEAREST drawn point is not enough: it answers "is there geometry
# here", and on a hazy reconstruction there is -- one faint, isolated splat.  Density
# asks the question that matters.  The threshold is a no-op on a coherent
# reconstruction (every anchor sits near the median density) and decisive on an
# incoherent one, where farthest-point sampling picks anchors from the sparse tail.
_ANCHOR_K = 64
_ANCHOR_SPARSITY = 2.5


def _anchor_keep_mask(traj: np.ndarray, pose_bounds) -> np.ndarray:
    """Which anchors stay ON the object, as a boolean mask over ``traj`` ``(T, P, 3)``.

    ``pose_bounds`` maps a trajectory row to the ``bounding_points`` of the geometry
    the caller will actually DRAW at that row -- an anchor sits on a canonical Gaussian
    or voxel, so a stray one lands wherever that asset has stray parts, and the
    anchors' own spread is no evidence about where the object is.

    An anchor must pass at EVERY row that has bounds, not just the reference one.  A
    noisy trajectory can start on the object and wander off, and the drawn dot is the
    trail's HEAD at the current frame -- so filtering on the reference frame alone
    would leave exactly those strays on screen.

    Falls back to the anchors' own bulk at the reference row
    (``rendering.bulk_keep_mask``) when the caller has no bounds to offer -- the
    renders that draw over a real camera, where the geometry is untrimmed and a stray
    dot is cosmetic rather than a framing error.
    """
    if not pose_bounds:
        # Imported here, not through `_lazy_imports`: this branch is the only part of
        # the track path that needs the rule, and a `--dry-run` never binds the GPU
        # globals.  Same reason as the scipy import below.
        import torch as _torch

        from genia.core.utils.rendering import bulk_keep_mask as _bulk_keep

        return _bulk_keep(
            _torch.as_tensor(np.ascontiguousarray(traj[0]))).numpy()
    from scipy.spatial import cKDTree

    keep = np.ones(traj.shape[1], dtype=bool)
    for t, bounds in pose_bounds.items():
        if bounds is None or t >= len(traj) or not keep.any():
            continue
        pts = np.asarray(bounds[0], np.float64)
        if not len(pts):
            continue
        # `k=[k]` asks for the k-th neighbour alone, and the whole anchor set is
        # queried rather than just the survivors: the threshold is the median over
        # all of them, so narrowing the set would move the bar.
        k = [min(_ANCHOR_K, len(pts))]
        d = cKDTree(pts).query(traj[t], k=k)[0][:, 0]
        keep &= d < _ANCHOR_SPARSITY * float(np.median(d))
    return keep if keep.any() else np.ones(traj.shape[1], dtype=bool)


def _track_trails(run: FinalRun, ctx: RenderContext, keys: list[FrameKey],
                  ref_key: FrameKey | None = None,
                  pose_bounds=None) -> np.ndarray | None:
    """``(len(keys), P, 3)`` track positions in ``compose(space="world")`` space.

    Reads :attr:`FinalRun.tracks` (posed R3 points).  ``ref_key=None`` keeps
    each frame's own posed position — for renders that keep root motion
    (``synth_nvs``).  A ``ref_key`` rebases every frame onto that key's pose,
    stripping root motion the same way the pinned Gaussians have it stripped
    (``orbit``).  Returns None when the run has no tracks, or none for the
    objects being rendered.
    """
    tracks = run.tracks
    if tracks is None:
        log("    (no 3D tracks on disk — rendering without them)")
        return None
    wanted = set(run.objects if ctx.objects is None else ctx.objects)
    per_obj: list[np.ndarray] = []
    for obj_idx in sorted(set(tracks.xyz) & wanted):
        obj = run.objects.get(obj_idx)
        if obj is None:
            continue
        xyz = tracks.xyz[obj_idx]                      # (T, P, 3)
        # EVERY anchor's whole trajectory first, then filter, then sample.  In that
        # order because both later steps depend on it: an anchor is judged by where it
        # goes, not only by where it starts, and farthest-point sampling MAXIMISES
        # spread -- it seeks out the extremes, which is exactly where a stray sits, so
        # sampling before filtering makes an outlier nearly certain to be among the
        # few drawn rather than merely possible.
        rows = []
        for k in keys:
            t = tracks.row_for(k.frame)
            if t is None:
                return None                            # tracks don't cover these frames
            rows.append(xyz[t] if ref_key is None
                        else run.rebase_points(obj, xyz[t], k, ref_key))
        traj = np.stack(rows)                          # (len(keys), P, 3)
        inside = np.flatnonzero(_anchor_keep_mask(traj, pose_bounds))
        if not len(inside):                            # degenerate — keep them all
            inside = np.arange(traj.shape[1])
        # How many to draw: the track_points cap, thinned by track_subsample.
        # farthest_point_indices clamps to the available count, so no upper bound
        # here even if track_subsample is set above 1.
        base = min(max(1, ctx.opts["track_points"]), len(inside))
        n_pts = max(1, round(base * ctx.opts["track_subsample"]))
        # Sampled on the reference frame's positions, so the kept tracks are spread
        # evenly through space rather than by anchor index.
        sel = inside[farthest_point_indices(traj[0][inside], n_pts)]
        per_obj.append(traj[:, sel])                   # (len(keys), n_pts, 3)
    if not per_obj:
        return None
    out = np.concatenate(per_obj, axis=1)
    log(f"    3D tracks: {out.shape[1]} point(s) over {out.shape[0]} frame(s)")
    return out


def motion_sample(m: int, slowmo: int, n_keys: int) -> tuple[int, int, float]:
    """Orbit step ``m`` -> the two frames it sits between, and the blend weight.

    ``m / slowmo`` is a continuous position in the sequence, so the frames land on
    the steps that are multiples of ``slowmo`` and the steps between them ask for a
    blend.  ``slowmo=1`` is the plain one-frame-per-step mapping, weight 0.

    The last interval CLAMPS: the final frame is HELD through the loop's seam
    rather than morphing back into the first.  The seam stays the hard cut it
    already was, and no motion is invented that the sequence does not contain.
    """
    pos = m / slowmo
    t0 = min(int(pos), n_keys - 1)
    return t0, min(t0 + 1, n_keys - 1), pos - t0


def sample_of(m: int, slowmo: int, n_keys: int,
              interpolate: bool) -> tuple[int, int, float]:
    """Which motion sample step ``m`` SHOWS -- :func:`motion_sample`, collapsed onto
    the held frame when the geometry cannot be interpolated.

    Holding the FLOOR frame (not the nearest) keeps the held clip a strict
    sub-sampling of the interpolated one: step ``t * slowmo`` shows frame ``t`` in
    both, so the two sit side by side in a gallery without a phase shift.

    Two steps with the same sample show the same picture, which is why this tuple --
    not the step index -- is what the trim, the bounds and the framing are keyed on.
    """
    t0, t1, w = motion_sample(m, slowmo, n_keys)
    return (t0, t1, w) if interpolate else (t0, t0, 0.0)


@renderer(
    "orbit",
    "Square turntable. ONE camera path per run: the target and FoV are fitted over "
    "every pose at once, so a deforming object does not make the camera chase it. "
    "Gaussians when the run has them, else the mesh. "
    "Dynamic runs deform in place (root motion removed), with their 3D tracks overlaid.",
    options={
        "n_frames": Option("int", 60, "orbit steps; on an animated FULL turn a "
                           "MINIMUM, rounded up to whole sequence loops "
                           "(0 = one per frame key)"),
        "azimuth_start": Option("float", 0.0, "first azimuth, degrees"),
        "azimuth_span": Option("float", 360.0, "total azimuth swept, degrees"),
        "elevation": Option("float", 0.0, "elevation offset, degrees"),
        "up": Option("str", "auto", "the vertical the turntable spins about: "
                     "auto (the dataset's world up where one is known, else "
                     "camera) | camera (the train camera's own up) | world "
                     "(R3 -Y) | 'x,y,z'"),
        "margin": Option("float", 0.04, "padding kept around the object when "
                                        "fitting the FoV (fraction of the frame)"),
        "frame": Option("int?", None, "freeze on this timestamp (implies animate=false)"),
        "animate": Option("bool", True, "step the timestamp along the orbit "
                                        "(dynamic runs; no-op when static)"),
        "slowmo": Option("int", 10, "play the motion this many times slower: each "
                         "1-frame interval is drawn over N steps with the "
                         "deformation LERPed between the two frames (needs "
                         "per-frame geometry, else it renders at 1)"),
        "source": Option("str", "auto", "geometry source: auto (final.pt when it "
                         "can serve the run, else the exported PLYs) | final_pt | plys"),
        "video": Option("bool", True, "encode the orbit to mp4"),
        "keep_frames": Option("bool", True, "also write the individual PNGs"),
        "distance_scale": Option("float", 2.5, "FALLBACK camera distance as a "
                                 "multiple of the framed box's half-diagonal, used "
                                 "only when the self-calibrated radius (the training "
                                 "camera's own distance to the centroid) would put "
                                 "the orbit camera inside the box -- e.g. a "
                                 "whole-scene reconstruction, whose 'object' is the "
                                 "whole scene rather than what one training camera "
                                 "was aimed at. Same convention as world_space's "
                                 "distance_scale."),
        **_TRACK_OPTIONS,
    },
)
def render_orbit(run: FinalRun, ctx: RenderContext) -> None:
    if not ctx.frame_keys:
        return
    opts = ctx.opts
    # Resolved up front so a bad `up` is rejected by --dry-run too, not first
    # discovered after the geometry has been loaded.
    up = resolve_up(opts["up"], run.dir)
    keys = ctx.frame_keys
    if opts["frame"] is not None:
        keys = [k for k in keys if k.frame == opts["frame"]] or keys
    # Step through the sequence unless pinned to one timestamp, taking each
    # frame's *deformation* but keeping the reference frame's Sim(3): a
    # turntable should show the object changing shape in place, not flying
    # across the frame under its root motion. No-op on a static run (one key).
    animate = bool(opts["animate"]) and opts["frame"] is None and len(keys) > 1
    # Slow motion lengthens the clip on EVERY animated run, whether or not its
    # geometry can be interpolated: a held clip keeps the same duration and the same
    # camera path as an interpolated one, so the two sit side by side in a gallery.
    # `interpolate` below decides only whether the object moves between frames.
    #
    # It slows TIME, so it keys on distinct timestamps, not on `keys` -- those are
    # (frame, view) pairs, and on an MV STATIC run they are all one timestamp seen
    # from several sides.  Multiplying by the VIEW count there would make an MV clip
    # longer than the mono clip of the same scene, breaking the side-by-side
    # comparison the length is supposed to protect.
    slowmo = (max(1, int(opts["slowmo"]))
              if animate and len({k.frame for k in keys}) > 1 else 1)
    ref_key = keys[0]
    ref_cam = run.camera(ref_key)
    if ref_cam is None or ref_cam.K is None:
        log("    ! no reference camera with intrinsics, skipping orbit")
        return

    use_mesh = run.prefers_mesh

    # The geometry SOURCE, decided once.  `final.pt` carries the canonical gaussians
    # and the deformation field, so re-warping here beats reading the exported
    # per-frame PLYs on both correctness (they are Morton-sorted, see
    # `rows_correspond`) and speed.  Mesh runs keep the PLY path: their per-frame
    # meshes are written by a plain writer and already correspond.
    want_source = opts["source"] if opts["source"] in ("auto", "final_pt") else None
    # A PER-FRAME RECONSTRUCTION is recognised here rather than discovered object by
    # object inside `open`: it reconstructs each timestamp independently and never
    # holds a canonical asset, so the cache has no canonical gaussians to serve and
    # there is nothing to interpolate between its frames.  Skipping it up front saves
    # loading a large final.pt to learn that.
    per_frame_only = run.has_perframe_gaussians and not run.has_gaussians
    if per_frame_only and want_source:
        log("    (per-frame reconstruction: no canonical asset, so the geometry "
            "comes from the exported per-frame PLYs and the frames cannot be "
            "interpolated)")
    source = None
    if want_source and not use_mesh and not ctx.dry_run and not per_frame_only:
        source = DeformationSource.open(
            run.dir, device=ctx.device, objects=sorted(
                set(run.objects) if ctx.objects is None else set(ctx.objects)),
            frames={k.frame for k in keys},
            required=opts["source"] == "final_pt")
    provider = None if source is None else source.provider()

    def compose_frame(t: int):
        return (run.compose_mesh(keys[t], device=ctx.device, objects=ctx.objects,
                                 pose_from=ref_key) if use_mesh else
                run.compose(keys[t], device=ctx.device, objects=ctx.objects,
                            pose_from=ref_key, geometry=provider))

    # Whether the frames can be interpolated at all, settled once on the first pair
    # rather than per sample.  It does NOT change the clip's length: a run that
    # cannot interpolate holds each frame for `slowmo` steps, so its orbit stays
    # comparable, frame for frame and camera for camera, with one that can.
    interpolate = slowmo > 1
    if interpolate and not ctx.dry_run and source is None:
        # Only the PLY path can be non-corresponding: `DeformationSource` warps one
        # canonical asset, so its frames correspond by construction.
        first, second = compose_frame(0), compose_frame(1)
        if first is None or second is None or not rows_correspond(first, second):
            log(f"    ! the per-frame assets do not correspond row for row — a "
                f"per-frame reconstruction, or (usually) gaussians written by the "
                f"compressed PLY writer, which Morton-sorts each frame into its own "
                f"order. Holding each frame for {slowmo} steps instead of "
                f"interpolating: the clip keeps its length and its camera path, so "
                f"it still sits beside an interpolated one — the motion steps rather "
                f"than flows. Keep final/final.pt, or re-export with "
                f"output.save_compressed_ply=false, to interpolate")
            interpolate = False
        del first, second

    # One pass through the sequence takes `loop` steps: one per frame, or `slowmo`
    # per frame when the motion is slowed.  Everything below indexes MOTION SAMPLES
    # in [0, loop), not frames -- the two coincide at slowmo=1.
    loop = len(keys) * slowmo
    n = max(1, opts["n_frames"] or loop)
    closed = opts["azimuth_span"] % 360.0 == 0.0
    if animate and closed and n % loop:
        # `n_frames` is a MINIMUM on a dynamic run: a turntable that stops
        # part-way through the sequence jumps when the clip loops, and the
        # motion is the point of animating it.  Round up to whole loops.
        n += loop - n % loop
        log(f"    {opts['n_frames']} steps is not a whole number of "
            f"{loop}-step loops — rounding up to {n}")
    if slowmo > 1:
        # "where they correspond", not "lerped": a run can pass the first-pair check
        # and still hold later pairs, and the per-sample note below says when it did.
        log(f"    slow motion x{slowmo}: {len(keys)} frame(s) over {loop} steps, "
            + ("the deformation interpolated where they correspond" if interpolate
               else f"each HELD for {slowmo} steps (nothing to interpolate)"))
    out_video = ctx.path("orbit.mp4") if opts["video"] else None
    frame_paths = [ctx.path("frames", f"{i:03d}.png") for i in range(n)]
    if ctx.dry_run:
        if opts["keep_frames"]:
            ctx.written.extend(frame_paths)
        if out_video is not None:
            ctx.written.append(out_video)
        return
    # The orbit is one artifact -- an existing video means there is nothing to do.
    if out_video is not None and not ctx.wants(out_video):
        log(f"    {out_video.name} exists (pass --overwrite to redo it)")
        return

    dropped: dict[tuple, int] = {}
    held: set[tuple] = set()    # samples whose blend fell back to holding a frame

    # UNtrimmed geometry per frame.  A blend reads two frames and re-reads them for
    # every step between, so those two are worth keeping.  Nothing else re-reads:
    # at slowmo=1, and when holding (every sample is one frame, visited once), each
    # frame is composed exactly once and `maxsize=0` keeps the orbit holding a
    # single cloud at a time.
    compose_at = lru_cache(maxsize=2 if interpolate else 0)(compose_frame)

    def geometry(sample: tuple[int, int, float]):
        """Renderable geometry for one motion sample ``(t0, t1, w)``.

        Every frame is composed with the REFERENCE pose, so blending two of them
        blends their deformations alone -- the Sim(3) they share is affine in the
        points, which makes lerping after posing the same as lerping before it.

        Trimming comes AFTER the blend, deliberately: it drops a different set of
        splats in each frame, and the blend needs the two clouds to still
        correspond row for row.
        """
        t0, t1, w = sample
        geom = compose_at(t0)
        if w > 0 and t1 != t0 and geom is not None:
            other = compose_at(t1)
            blended = geom if other is None else blend_geometry(geom, other, w)
            if blended is geom:
                # `blend_geometry` held t0 rather than interpolating: the frames do
                # not correspond.  Print it once -- the clip is a stutter, not a
                # slow motion, and nothing else in the output would say why.
                if not held:
                    log("    ! two frames stopped corresponding part-way through "
                        "— holding the earlier one for those steps")
                held.add(sample)
            geom = blended
        if geom is None or use_mesh:
            return geom
        # A turntable is the one render that gets close to a stray splat behind
        # the object, so it is the one render that has to reject them.
        geom, n_out = trim_outliers(geom)
        dropped[sample] = n_out
        return geom

    ref_sample = sample_of(0, slowmo, len(keys), interpolate)
    ref_geom = geometry(ref_sample)
    if ref_geom is None:
        log("    ! no gaussians or meshes posed at the reference frame, skipping")
        return
    if dropped.get(ref_sample):
        log(f"    dropped {dropped[ref_sample]} outlier splat(s) outside the "
            f"object's bulk ({len(ref_geom)} kept)")
    if use_mesh:
        log(f"    rendering the mesh ({len(ref_geom)} verts) — run has no gaussians")

    from genia.core.utils.eval_assets_export import _synth_orbit_c2w

    # Which steps show which pose, defined ONCE: the bounds below, the per-pose
    # framing and the render loop all key off it, so they cannot disagree.
    # Keyed by the SAMPLE, not the step: steps showing the same picture collapse
    # into one group, so a held clip trims, bounds and frames each distinct pose
    # once instead of `slowmo` times.  (The renders cannot collapse -- every step
    # is a different azimuth, which is the whole point of keeping the length.)
    steps_of: dict[tuple, list[int]] = {}
    for i in range(n):
        steps_of.setdefault(
            sample_of(i % loop if animate else 0, slowmo, len(keys), interpolate),
            []).append(i)

    # Bounds of every pose, gathered BEFORE the cameras: they decide both where the
    # orbit looks and how large the object comes out.
    pose_bounds = {}
    for t in steps_of:
        g = ref_geom if t == ref_sample else geometry(t)
        if g is not None:
            pose_bounds[t] = bounding_points(g)
    if not pose_bounds:
        log("    ! no geometry to frame, skipping")
        return

    # Trails only mean something against geometry that actually deforms. With a
    # canonical-only asset (a gaussian/mesh run whose motion is pure Sim(3)) the
    # render is static once the pose is pinned, so moving trails would imply
    # motion the picture does not show -- require a per-frame deforming asset.
    show_tracks = (opts["tracks"] and animate
                   and (run.has_perframe_gaussians or run.has_perframe_meshes))
    if opts["tracks"] and not show_tracks and run.tracks is not None:
        log("    (skipping the 3D track overlay: no per-frame geometry to match it)")
    # `pose_bounds` is keyed by motion SAMPLE; the anchor filter indexes each
    # trajectory by FRAME, so collapse it (keeping the exact frame, w=0, which
    # sorts first) before handing it over.
    frame_bounds: dict[int, Any] = {}
    for (t0, _t1, _w), b in pose_bounds.items():
        frame_bounds.setdefault(t0, b)
    trails = (_track_trails(run, ctx, keys, ref_key, frame_bounds)
              if show_tracks else None)
    trail_opts = opts
    if trails is not None and slowmo > 1:
        # Resampled through the SAME mapping the geometry is, so a trail head stays
        # on the splat it belongs to; `track_trail` is scaled with it, so its unit
        # stays FRAMES rather than silently becoming steps.
        trails = np.stack([trails[t0] * (1.0 - w) + trails[t1] * w
                           for t0, t1, w in
                           (sample_of(m, slowmo, len(keys), interpolate)
                            for m in range(loop))])
        trail_opts = {**opts, "track_trail": opts["track_trail"] * slowmo}
    colors = track_colors(trails.shape[1]) if trails is not None else []

    # (0, 0) reproduces the reference camera, so a full turn must not repeat it.
    # Always square: a turntable sweeps the object through every azimuth, so a
    # frame shaped like the source sequence would clip it side-on.
    side = max(ctx.H, ctx.W)
    azimuths = [
        opts["azimuth_start"] + opts["azimuth_span"] * (
            i / n if closed else i / max(1, n - 1))
        for i in range(n)
    ]
    log(f"    spinning about {'the train camera up' if up is None else np.round(up, 3)}"
        f" (up={opts['up']})")

    # ONE framing for the whole clip: the target is the centre of the box covering
    # EVERY pose, and the focal is fitted over every pose from every azimuth (which
    # is what `fit_square_focal` takes a LIST of bounds for).  So the camera path is
    # a plain circle and nothing about it depends on the timestamp.
    #
    # Framing each pose on its OWN extent instead would make the camera chase the
    # object's axis-aligned box, and that box moves every frame on anything that
    # deforms: the object would jitter, and the per-pose focal would make it breathe.
    #
    # The cost: the object fills somewhat less of the frame than a per-pose fit
    # would give it, and a deformation's drift and growth stay visible rather than
    # being normalized away.  For a SHAPE viewer that is the honest picture, and runs
    # stay comparable in a gallery since each is still fitted to its own extent.
    boxes = [box for box in (bounds_aabb(*b) for b in pose_bounds.values())
             if box is not None]
    if not boxes:
        log("    ! no geometry to frame, skipping")
        return
    lo = np.min([b[0] for b in boxes], axis=0)
    hi = np.max([b[1] for b in boxes], axis=0)
    centroid = (lo + hi) * 0.5
    # The self-calibrated radius (`_synth_orbit_c2w`'s default: the training camera's
    # own distance to the centroid) is only a sane orbit radius when the geometry is
    # roughly the size of what that camera was aimed at. On a whole-scene
    # reconstruction the "object" is the whole scene, whose box can dwarf that
    # distance -- the camera would then sit INSIDE the box, and fitting a focal that
    # keeps everything in frame from in there collapses toward zero (a huge FoV that
    # renders the object as impossibly small and far away). Falling back to a
    # world_space-style radius, fitted to the box itself, is the same self-fit
    # `world_space` uses.
    eye0 = np.asarray(ref_cam.c2w[:3, 3], dtype=np.float64)
    self_radius = float(np.linalg.norm(centroid - eye0))
    box_half_diag = float(np.linalg.norm(hi - lo)) * 0.5
    radius = None
    if self_radius < box_half_diag:
        radius = box_half_diag * opts["distance_scale"]
        log(f"    training camera sits inside the framed box (self-calibrated "
            f"radius {self_radius:.2f} < half-diagonal {box_half_diag:.2f}) — "
            f"using a self-fitted radius {radius:.2f} instead")
    cams = [_synth_orbit_c2w(ref_cam.c2w, centroid, az, opts["elevation"], up=up,
                             radius=radius) for az in azimuths]
    focal = fit_square_focal(list(pose_bounds.values()), cams, side,
                             margin=max(0.0, opts["margin"]))
    if focal is None:
        log("    ! nothing in front of the orbit cameras, skipping")
        return
    Ks = [square_intrinsics(focal, side)] * n
    log(f"    framing: {side}x{side}, ONE camera path — target and focal "
        f"({focal:.1f}px) fitted over all {len(pose_bounds)} pose(s) from "
        f"{len(azimuths)} view(s), {opts['margin']:.0%} margin")

    render = render_mesh if use_mesh else render_cloud
    # Frame-major, not step-major: per-frame clouds are deliberately not cached,
    # so stepping i in order would re-decode each one on every loop.  The steps
    # sharing a motion sample are rendered together instead, which decodes each
    # cloud once and still holds only one.
    frames: list[np.ndarray | None] = [None] * n
    for t, idxs in steps_of.items():                      # every step, grouped
        geom = ref_geom if t == ref_sample else geometry(t)
        if geom is None:
            continue
        for i in idxs:
            c2w, K = cams[i], Ks[i]
            if c2w is None:
                continue
            rgb, alpha, depth = render(geom, c2w, K, side, side, ctx.bg)
            # The group key is a sample, not an index; the trails carry one row
            # per step of the loop, so the step's own position indexes them.
            frames[i] = overlay_trails(to_rgba(rgb, alpha), trails,
                                       i % loop if animate else 0, c2w, K,
                                       depth, colors, trail_opts)
            if opts["keep_frames"]:
                ctx.write_png(frame_paths[i], frames[i])

    if out_video is not None:
        ctx.write_video(out_video, [f for f in frames if f is not None])


# X=red, Y=green, Z=blue -- the convention of
# `core/utils/visualization.py::draw_pose_axes_on_image`, whose output this render
# reproduces from disk.
_AXIS_COLORS = ((255, 60, 60), (60, 220, 60), (60, 120, 255))


def load_background_cloud(run: FinalRun, device: str) -> "GaussianCloud | None":
    """``final/gaussians/background[.compressed].ply``, or None if the run wrote none.

    Already R3 WORLD space (every frame's background lifted by its own ``c2w`` and
    unioned, per ``gaussian.aggregate_background_gaussians``), so it composes with
    ``FinalRun.compose(space="world")`` by concatenation and nothing else.  Absent for
    every white-background dataset (GSO, OAB) and for any run whose
    ``output.save_world_assets`` was off -- foreground-only is a normal outcome here,
    not an error.
    """
    for name in ("background.compressed.ply", "background.ply"):
        path = run.dir / "gaussians" / name
        if path.is_file():
            return load_gaussian_ply(path, device)
    return None


def overview_cameras(run: FinalRun, keys_by_view: dict[int, list[FrameKey]],
                     target: np.ndarray, distance: float,
                     elevation: float, azimuth: float) -> dict[int, np.ndarray]:
    """One overview c2w per view, each placed behind THAT view's own input camera.

    A single shared camera cannot tell the views of one timestamp apart: an MV run that
    honours ``pipeline.mv_shared_world_pose`` puts every object at the same WORLD point
    from every view, so V renders from one viewpoint are V copies of one picture.  Giving each view the
    viewpoint of the camera that actually observed it is what makes the set worth
    rendering -- the scene layout seen from each side it was captured from.

    Each view keeps its own ELEVATION, which is the half of its vantage that
    ``overview_look_at`` would otherwise throw away: that function flattens the reference
    forward onto the horizontal plane before applying ``elevation``, so two cameras
    differing mainly in height collapse onto the same viewpoint.  ``azimuth`` needs no such treatment -- it is already relative, rotating a ``back_dir``
    that carries the view's own azimuth.

    Elevations are CENTRED ON THEIR MEAN rather than used absolutely, so ``elevation``
    keeps meaning "the centre of the fan sits this far above horizontal".  That also
    makes a mono run a bit-exact no-op: with one view ``own - mean`` is ``x - x``, exactly
    0.0, so ``overview_look_at`` sees identical arguments; with absolute elevations
    every mono run with a tilted input camera would move.

    ``target`` and ``distance`` are SHARED, and the caller fits one focal over all the
    returned cameras, so the views stay directly comparable in position and scale.

    NOTE the world up is R3 ``-Y`` throughout, which is only KNOWN to be up on a run that
    kept the dataset's cameras (GSO + ``camera_poses_source=gt``); a predicted-pose run
    sits in an arbitrary frame, the caveat ``viz_orbit_up=auto`` encodes for the orbit.
    The per-view SEPARATION is unaffected -- relative geometry between views holds
    whichever axis is up -- only the absolute tilt and the image roll rest on the guess.
    """
    # Local, like the orbit's `_synth_orbit_c2w`: `genia.core.utils.evaluation` imports torch,
    # and this file defers that to `_lazy_imports` so --list/--dry-run stay fast.
    from genia.core.utils.evaluation import overview_look_at

    # No camera in poses.json -> look down +Z, the identity camera such a run's poses are
    # already expressed against.
    forwards = {}
    for view, vkeys in keys_by_view.items():
        cam = run.camera(vkeys[0])
        forwards[view] = (cam.c2w[:3, 2] if cam is not None
                          else np.array([0.0, 0.0, 1.0]))

    # Up is -Y in R3 and the camera sits along -forward from the target, so this view's
    # own elevation above horizontal is asin(forward . -up) = asin(forward[1]).
    own = {view: float(np.degrees(np.arcsin(np.clip(
              (f / max(np.linalg.norm(f), 1e-12))[1], -1.0, 1.0))))
           for view, f in forwards.items()}
    mean_own = float(np.mean(list(own.values())))

    # No clamp on the total: a straight-down camera gives own = 90, and overview_look_at
    # at 110 still returns a finite, orthonormal c2w at the right distance (checked).
    # `own - mean` FIRST, then add: grouped the other way, `(elevation + own) - mean`
    # loses the low bits and a mono run drifts by a float32 LSB instead of being the
    # exact no-op the whole centring exists to guarantee.
    return {view: overview_look_at(target, f, distance,
                                   elevation + (own[view] - mean_own), azimuth)
            for view, f in forwards.items()}


def draw_pose_axes(rgba: np.ndarray, run: FinalRun, key: FrameKey, c2w: np.ndarray,
                   K: np.ndarray, objects: Collection[int], axis_length: float,
                   width: int = 2) -> np.ndarray:
    """Draw each object's Sim(3) axes at ``key`` onto an RGBA frame.

    The object's local origin and three axis tips go through ``_place_points`` -- the
    same object-local-P3D -> posed-R3 chain the Gaussians take -- so the axes land on
    the geometry by construction rather than by a re-derivation that could drift from
    it.  ``axis_length`` is in object-local units and is therefore scaled by the pose,
    matching ``interpolation.compute_pose_axes``.

    Drawn on top, with no depth test: the axes originate at the object's own centre,
    so testing them against the rendered surface would bury every one of them.
    """
    from PIL import ImageDraw

    local = np.vstack([np.zeros((1, 3)), np.eye(3) * float(axis_length)])
    img = Image.fromarray(rgba).convert("RGBA")
    draw = ImageDraw.Draw(img)
    for obj_idx in sorted(set(run.objects) & set(objects)):
        pose = run.resolve_transform(run.objects[obj_idx], key)
        if pose is None:
            continue
        uv = project_r3(run._place_points(local, pose), c2w, K)
        # All four points strictly in front, or skip the object: a vertex approaching
        # the image plane sends `x / z` to infinity, and PIL would try to rasterize it.
        if not (uv[:, 2] > 1e-6).all():
            continue
        ox, oy = uv[0, :2]
        for axis in range(3):
            draw.line([(ox, oy), tuple(uv[axis + 1, :2])],
                      fill=(*_AXIS_COLORS[axis], 255), width=width)
        draw.ellipse([ox - width, oy - width, ox + width, oy + width],
                     fill=(255, 255, 255, 255))
    return np.array(img)


def view_tag(view: int, multi: bool) -> str:
    """``_vNN`` when more than one view is being rendered, else empty.

    Deliberately NOT ``frame_key_stem``, which leaves view 0 unsuffixed: that reads
    fine where views are separate folders or separate assets, but here the views of one
    timestamp are a SERIES of the same picture from different sides, and a bare
    ``000.png`` beside ``000_v01.png`` does not look like a member of it.  A mono run
    still gets the plain ``{frame:03d}`` every other per-frame writer uses.
    """
    return f"_v{view:02d}" if multi else ""



@renderer(
    "world_space",
    "Every object posed into world space, seen from a fixed overview camera PER VIEW -- "
    "each placed behind the input camera that observed that view, so an MV run shows its "
    "layout from every side it was captured from (one shared camera would render V "
    "copies of one picture). Framed on the FOREGROUND -- the camera aims at the objects' "
    "box centre and ONE focal is fitted to them across every view and frame, so the "
    "objects fill the frame, the camera stays static as they move, and the views stay "
    "comparable in scale. Root motion is KEPT (unlike `orbit`), so a dynamic run shows "
    "its objects travelling through the scene, and a multi-object run shows their real "
    "relative layout. Foreground only by default; add "
    "-O world_space.background=true for the run's background point cloud, "
    "-O world_space.axes=true for each object's Sim(3) pose axes.",
    options={
        "elevation": Option("float", 20.0, "degrees above horizontal for the CENTRE of "
                            "the views' fan — each view keeps its own camera's elevation "
                            "relative to that, so an MV run's views stay distinct"),
        "azimuth": Option("float", 10.0, "camera swing around the world up, degrees "
                          "off straight-behind-the-train-camera (0 = straight behind)"),
        "distance_scale": Option("float", 2.5, "camera distance as a multiple of the "
                                 "foreground's bounding radius"),
        "margin": Option("float", 0.08, "padding kept around the foreground when "
                                        "fitting the focal (fraction of the frame)"),
        "background": Option("bool", False, "include gaussians/background*.ply when the "
                             "run wrote one. OFF by default: it is an unprojected depth "
                             "map, so it arrives full of holes and occluding fragments "
                             "that bury the reconstruction this render exists to show"),
        "axes": Option("bool", False, "draw each object's Sim(3) pose axes. OFF by "
                       "default: this render exists to show the scene, and the axes "
                       "are a pose DEBUG overlay that clutters it"),
        "axis_length": Option("float", 0.15, "axis arm length, in object-local units "
                                             "(so the pose scales it)"),
        "perframe": Option("bool", True, "use per-frame warped PLYs when present"),
        "video": Option("bool", True, "also encode one mp4 per view, when that view has "
                        "more than one frame"),
    },
)
def render_world_space(run: FinalRun, ctx: RenderContext) -> None:
    if not run.has_renderable_geometry:
        log("    ! run has no gaussians or meshes — nothing to render")
        return
    opts = ctx.opts

    # One render per (frame, view), each view from its OWN overview camera -- see
    # `overview_cameras` for why a shared one makes the views indistinguishable.
    keys = list(ctx.frame_keys)
    if not keys:
        return

    use_mesh = run.prefers_mesh
    by_view = group_by_view(keys)
    multi = len(by_view) > 1        # from the FILTERED keys, so `--views 0` reads as mono
    frame_paths = [ctx.path(f"{k.frame:03d}{view_tag(k.view, multi)}.png") for k in keys]
    # A video per VIEW, gated on THAT view having more than one frame: mono-dynamic gets
    # one clip, MV-static gets V stills and no clip (there is no time axis to play), and
    # MV-dynamic gets V clips.  Per-view rather than a global max, so a view with a lone
    # frame is not handed to the encoder on a run whose views are unequal in length.
    videos = {v: ctx.path(f"world_space{view_tag(v, multi)}.mp4")
              for v, vkeys in by_view.items() if opts["video"] and len(vkeys) > 1}
    if ctx.dry_run:
        ctx.written.extend(frame_paths)
        ctx.written.extend(videos.values())
        return
    # One artifact: the camera is fitted over EVERY key, so re-rendering a subset would
    # have to redo that fit anyway -- there is no cheap partial resume to offer.
    primary = next(iter(videos.values()), frame_paths[0])
    if not ctx.wants(primary):
        log(f"    {primary.name} exists (pass --overwrite to redo it)")
        return

    def foreground(key: FrameKey):
        if use_mesh:
            return run.compose_mesh(key, device=ctx.device, objects=ctx.objects)
        return run.compose(key, device=ctx.device, space="world",
                           objects=ctx.objects, use_perframe=opts["perframe"])

    # Bounds over EVERY key, unioned -- what makes the camera STATIC: the objects never
    # leave the frame as they move, and never breathe as their silhouettes narrow and
    # widen.  Measured on the TRIMMED cloud (one stray splat collapses an `x / z` fit)
    # while the render below keeps every splat: a scene view should not silently delete
    # geometry, it should show what the run produced.  A separate pass from the render,
    # like the orbit's, so only one composed cloud is ever held at a time.
    bounds = []
    for key in keys:
        geom = foreground(key)
        if geom is not None:
            bounds.append(bounding_points(
                geom if use_mesh else trim_outliers(geom)[0]))
    if not bounds:
        log("    ! nothing posed at any frame key, skipping")
        return

    pts = np.concatenate([b[0] for b in bounds])
    radii = np.concatenate([b[1] for b in bounds])
    box = bounds_aabb(pts, radii)
    if box is None:
        log("    ! no foreground extent to frame, skipping")
        return
    lo, hi = box
    # The box CENTRE, not a mass centroid: a mass centroid follows
    # wherever the splats are dense and puts an asymmetric scene off-centre.  Distance
    # comes from the foreground's own size, so perspective is scaled to the objects
    # rather than to how far apart the input cameras happened to be.
    target = (lo + hi) * 0.5
    radius = max(float(np.linalg.norm(hi - lo)) * 0.5, 1e-3)

    cams = overview_cameras(run, by_view, target, radius * opts["distance_scale"],
                            opts["elevation"], opts["azimuth"])
    # ONE focal over EVERY view, so the views are comparable in scale -- `fit_focal`
    # already takes a list of cameras and returns the worst case, which is exactly that.
    # Fitted to the foreground ALONE: the background is rendered but does not vote, and
    # letting it in is what leaves the objects a quarter of the frame.
    focal = fit_focal(bounds, list(cams.values()), ctx.W, ctx.H, opts["margin"])
    if focal is None:
        log("    ! nothing in front of the overview camera, skipping")
        return
    K = centred_intrinsics(focal, ctx.W, ctx.H)

    # How much of the frame the foreground won in its TIGHTEST view -- the one number
    # that says whether one shared fit was good enough for all of them, so it belongs in
    # the log rather than only in the PNGs.
    covers = []
    for c2w in cams.values():
        uv = project_r3(pts, c2w, K)
        seen = uv[uv[:, 2] > 0, :2]
        covers.append(float(np.prod(seen.max(0) - seen.min(0))) / (ctx.W * ctx.H)
                      if len(seen) else 0.0)
    log(f"    framing: {ctx.W}x{ctx.H}, focal {focal:.1f}px fitted to the foreground "
        f"over {len(bounds)} frame key(s) from {len(cams)} view(s) "
        f"({opts['margin']:.0%} margin) — its box covers {min(covers):.0%} of the frame")

    bg_cloud = None
    if opts["background"] and not use_mesh:
        bg_cloud = load_background_cloud(run, ctx.device)
        log(f"    + background point cloud ({len(bg_cloud)} splats)"
            if bg_cloud is not None else
            "    (no gaussians/background*.ply — foreground only)")

    render = render_mesh if use_mesh else render_cloud
    paths = dict(zip(keys, frame_paths))
    for key in keys:
        geom = foreground(key)
        if geom is None:
            continue
        if bg_cloud is not None:
            geom = concat_clouds([geom, bg_cloud])
        c2w = cams[key.view]
        rgb, alpha, _ = render(geom, c2w, K, ctx.H, ctx.W, ctx.bg)
        frame = to_rgba(rgb, alpha)
        if opts["axes"]:
            frame = draw_pose_axes(frame, run, key, c2w, K, ctx.objects,
                                   opts["axis_length"])
        ctx.write_png(paths[key], frame)

    # Not `ctx.write_per_view_videos`: that owns two naming decisions this render makes
    # differently (frame_key_stem for the PNGs, an unsuffixed view 0 for the mp4), and
    # three other renders depend on it meaning what it means.  Streaming the PNGs back
    # off disk is its one behaviour worth keeping, and `write_video` already does that.
    for view, out in videos.items():
        ctx.write_video(out, [paths[k] for k in by_view[view]])


@renderer(
    "contact_sheet",
    "Tile PNGs the run already exported (renders_train/renders_test/...) into a grid.",
    options={
        "source": Option("str", "renders_train", "subfolder of final/ to tile"),
        "cols": Option("int", 0, "columns (0 = square-ish)"),
        "cell": Option("int", 256, "cell size in pixels (longest side; aspect preserved)"),
        "limit": Option("int", 64, "max images (0 = all)"),
    },
    needs_gpu=False,
)
def render_contact_sheet(run: FinalRun, ctx: RenderContext) -> None:
    out = ctx.path(f"{ctx.opts['source'].replace('/', '_')}.png")
    if not ctx.wants(out):
        if not ctx.dry_run:
            log(f"    {out.name} exists (pass --overwrite to redo it)")
        return
    src = run.dir / ctx.opts["source"]
    paths = sorted(src.glob("**/*.png"))
    if ctx.opts["limit"]:
        paths = paths[: ctx.opts["limit"]]
    if not paths:
        log(f"    ! no PNGs under {src}, skipping")
        return

    cell = ctx.opts["cell"]
    cols = ctx.opts["cols"] or max(1, math.ceil(math.sqrt(len(paths))))
    rows = math.ceil(len(paths) / cols)
    # Cell box follows the source aspect (a run's renders share a resolution),
    # so nothing is squished; `cell` is the longest side.
    with Image.open(paths[0]) as im0:
        w0, h0 = im0.size
    scale = cell / max(w0, h0)
    cw, ch = max(1, round(w0 * scale)), max(1, round(h0 * scale))
    bg = tuple((ctx.bg * 255).astype(np.uint8).tolist())
    sheet = Image.new("RGB", (cols * cw, rows * ch), bg)
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            tile = Image.fromarray(flatten_alpha(np.asarray(im.convert("RGBA")), ctx.bg))
        # Fit within the cell box preserving aspect, centred (letterboxes a
        # stray odd-sized image rather than distorting it).
        tile.thumbnail((cw, ch), Image.LANCZOS)
        ox = (i % cols) * cw + (cw - tile.width) // 2
        oy = (i // cols) * ch + (ch - tile.height) // 2
        sheet.paste(tile, (ox, oy))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    ctx.written.append(out)


# ===========================================================================
# CLI
# ===========================================================================


def parse_index_spec(spec: str, available: list[int]) -> list[int]:
    """``"0,4"`` / ``"0-15"`` / ``"0-30:5"`` / ``"all"`` -> a list of indices."""
    if spec.strip().lower() in ("", "all"):
        return list(available)
    picked: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        step = 1
        if ":" in part:
            part, step_s = part.split(":", 1)
            step = int(step_s)
        if "-" in part:
            lo, hi = part.split("-", 1)
            picked.extend(range(int(lo), int(hi) + 1, step))
        else:
            picked.append(int(part))
    keep = set(available)
    return [i for i in picked if i in keep]


def parse_bg(spec: str) -> np.ndarray:
    named = {"white": (1.0, 1.0, 1.0), "black": (0.0, 0.0, 0.0), "gray": (0.5, 0.5, 0.5)}
    if spec in named:
        return np.asarray(named[spec], dtype=np.float32)
    vals = [float(v) for v in spec.split(",")]
    if len(vals) != 3:
        raise SystemExit(f"--bg wants a name or 'R,G,B', got {spec!r}")
    scale = 255.0 if max(vals) > 1.0 else 1.0
    return np.asarray([v / scale for v in vals], dtype=np.float32)


def parse_resolution(spec: str | None, native: tuple[int, int]) -> tuple[int, int]:
    if not spec:
        return native
    if "x" in spec.lower():
        w, h = spec.lower().split("x", 1)
        return int(h), int(w)
    return int(spec), int(spec)


def render_menu(show_options: bool = False) -> str:
    """The registered renders as text: a ``name — help`` menu, plus each
    render's ``-O`` options when ``show_options`` (what ``--list`` prints; the
    compact form is the ``--help`` epilog)."""
    lines = ["available renders (select with -r NAME, repeatable; omit -r or "
             "pass -r all for every one):", ""]
    for name in sorted(RENDERERS):
        r = RENDERERS[name]
        gpu = "" if r.needs_gpu else "  [no GPU needed]"
        lines.append(f"  {name}{gpu}\n      {r.help}")
        if show_options:
            for key, opt in r.options.items():
                assign = f"-O {name}.{key}={opt.default!r}"
                lines.append(f"      {assign:<44} ({opt.type}) {opt.help}")
        lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=render_menu(),
    )
    p.add_argument("final_dirs", nargs="*",
                   help="one or more final/ dirs (globs and run dirs also work)")
    p.add_argument("-r", "--render", action="append", default=[], dest="renders",
                   help="render name (repeatable); omit or 'all' runs every render")
    p.add_argument("-O", "--opt", action="append", default=[], dest="opts",
                   metavar="KEY=VALUE",
                   help="render option; scope it as RENDER.KEY=VALUE")
    p.add_argument("--list", action="store_true", help="list renders and exit")
    p.add_argument("--out", default=None,
                   help="output root (default: {final}/viz); several runs get "
                        "one subdir each under it")
    p.add_argument("--frames", default="all",
                   help="frame filter: 'all', '0,4,8', '0-15', '0-30:5'")
    p.add_argument("--views", default="all", help="view filter, same syntax")
    p.add_argument("--objects", default="all", help="object-index filter, same syntax")
    p.add_argument("--resolution", default=None,
                   help="render size: 'N' or 'WxH' (default: the run's own)")
    p.add_argument("--bg", default="white",
                   help="background the foreground composites on (white/black/gray/R,G,B)")
    p.add_argument("--fps", type=int, default=24, help="video frame rate")
    p.add_argument("--device", default="cuda", help="torch device")
    p.add_argument("--overwrite", action="store_true", help="re-render existing files")
    p.add_argument("--dry-run", action="store_true",
                   help="print what would be written, render nothing")
    return p


def viz_roots(final_dirs: list[Path], out: str | None) -> dict[Path, Path]:
    """Where each run writes: its own ``{final}/viz``, or a subdir of ``--out``.

    A shared ``--out`` over several runs gets one subdir per run, so their
    identically-named outputs (``orbit.mp4``, ...) cannot overwrite each other.
    """
    if out is None:
        return {d: d / "viz" for d in final_dirs}
    root = Path(out)
    if len(final_dirs) == 1:
        return {final_dirs[0]: root}
    runs = {d: parse_run_path(d) for d in final_dirs}
    scenes = [r.scene for r in runs.values()]
    unique = len(set(scenes)) == len(scenes)
    return {d: root / (r.scene if unique else f"{r.scene}_{r.timestamp}")
            for d, r in runs.items()}


def process_run(final_dir: Path, selected: list[Renderer], args,
                viz_root: Path) -> None:
    run = FinalRun(final_dir)
    log(f"\n=== {run.path.label}")
    log(f"    {run.summary()}")

    frames = set(parse_index_spec(args.frames, run.frames))
    views = set(parse_index_spec(args.views, run.views))
    keys = [k for k in run.frame_keys if k.frame in frames and k.view in views]
    objects = parse_index_spec(args.objects, sorted(run.objects))
    src_hw = run.resolution
    H, W = parse_resolution(args.resolution, src_hw)
    bg = parse_bg(args.bg)

    for rdr in selected:
        ctx = RenderContext(
            out_dir=viz_root / rdr.out_sub,
            device=args.device,
            frame_keys=keys,
            objects=objects,
            H=H, W=W, src_hw=src_hw,
            bg=bg,
            fps=args.fps,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            opts=resolve_options(rdr, args.opts),
        )
        log(f"  [{rdr.name}] -> {ctx.out_dir}")
        rdr.fn(run, ctx)
        verb = "would write" if args.dry_run else "wrote/kept"
        log(f"  [{rdr.name}] {verb} {len(ctx.written)} file(s)")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.list:
        log(render_menu(show_options=True))
        return 0
    if not args.final_dirs:
        parser.print_usage()
        log("error: no final/ directory given (use --list to see the renders)")
        return 2

    # No -r means "everything", same as -r all -- the useful default for a
    # first look at a run.
    names = (sorted(RENDERERS) if not args.renders or "all" in args.renders
             else args.renders)
    unknown = [n for n in names if n not in RENDERERS]
    if unknown:
        log(f"error: unknown render(s) {unknown}; available: {sorted(RENDERERS)}")
        return 2
    selected = [RENDERERS[n] for n in names]
    check_options_are_claimed(selected, args.opts)

    final_dirs = discover_final_dirs(args.final_dirs)
    if not final_dirs:
        log("error: no final/ directory matched")
        return 2

    _lazy_imports(gpu=any(r.needs_gpu for r in selected) and not args.dry_run)

    roots = viz_roots(final_dirs, args.out)
    failures = 0
    for final_dir in final_dirs:
        try:
            process_run(final_dir, selected, args, roots[final_dir])
        except Exception as exc:
            failures += 1
            log(f"  ! {final_dir}: {type(exc).__name__}: {exc}")
            if len(final_dirs) == 1:  # nothing else to salvage -- show where
                traceback.print_exc()
    if failures:
        log(f"\n{failures}/{len(final_dirs)} run(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
