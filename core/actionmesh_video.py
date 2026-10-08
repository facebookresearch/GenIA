# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Run ActionMesh on a dynamic sequence: one animated mesh per object.

GenIA's dynamic shape injection (``gt_shapes_inversion=actionmesh``) reads per-frame
meshes ``obj_NNN/mesh_NN.glb``; this script produces them from a clip with the
ActionMesh model (``submodules/actionmesh``, with TripoSG from its ``third_party/``).
It runs in the ``genia`` environment and reads the weights from the Hugging Face cache
(``python -m genia.core.download_weights`` fetches them; otherwise they download on first use).

On top of the actionmesh package it adds:
- ``load_per_object_from_rgb_seg_dirs``: one ActionMeshInput per palette id of a
  segmented clip, cropped like ActionMesh's curated examples (512x512 RGBA, one
  clip-constant scale, per-frame bbox-centred BICUBIC crop).
- Input-layout detection, stride/length validation against the model's [16, 31]
  frame budget, a per-object run loop, and a ``metadata.json`` (selected frames +
  per-object palette grouping) with the ``processed_frames/`` the model saw.
- A ``timing.json`` whose headline number is the inference pass alone (frame
  loading, mesh writing and the optional GLB/render passes are measured but held
  out; the weight load happens before the clock starts).

Layouts
-------
Any dynamic sequence, in whichever shape it is already on disk.  The first two
carry a palette segmentation, so they split the scene per object; the last two
carry the mask in an alpha channel and are single-object.

    davis:    <input_path>/JPEGImages/Full-Resolution/<scene>/*.jpg
              <input_path>/Annotations/Full-Resolution/<scene>/*.png
    rgb_seg:  <input_path>/<scene>/rgbs/*.png
              <input_path>/<scene>/segmentations/*.png   (the dataset=dyncustom layout)
    curated:  <input_path>/<scene>/*.png   (RGBA, alpha = mask)
    video:    <input_path>/<scene>.mp4     (white-background clip, e.g. the V2M4
              benchmark — decoded + matted in memory, nothing staged to disk)

Usage
-----
    python core/actionmesh_video.py \\
        --input_path demo/data/dynamic --input_scene dinosaur \\
        --output_dir demo/data/dynamic/dinosaur/actionmesh

then run GenIA with ``gt_shapes_inversion=actionmesh
gt_shapes_inversion.gt_data_root=demo/data/dynamic gt_shapes_inversion.scene_subdir=actionmesh``
(``demo/run_demo.py dynamic`` does both).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# --- make the clean (pinned) actionmesh package importable ----------------
_REPO = Path(__file__).resolve().parents[1]   # the genia checkout
_ACTIONMESH = _REPO / "submodules" / "actionmesh"
_TRIPOSG = _ACTIONMESH / "third_party" / "TripoSG"
for _p in (_ACTIONMESH, _TRIPOSG):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from actionmesh.io.glb_export import create_animated_glb  # noqa: E402
from actionmesh.io.mesh_io import save_deformation, save_meshes  # noqa: E402
from actionmesh.io.video_input import (  # noqa: E402
    IMAGE_EXTENSIONS,
    ActionMeshInput,
    load_from_image_dir,
)
import actionmesh.pipeline  # noqa: E402
from actionmesh.pipeline import ActionMeshPipeline  # noqa: E402
from natsort import natsorted  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Curated-asset emulation (matches assets/examples/davis_camel/*.png):
# 512x512 RGBA, shared scale across the clip, per-frame center crop, BICUBIC.
CURATED_TARGET_SIZE = 512

def _bbox_of_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return (x0, y0, w, h) of the tight bbox of a binary mask, or None if empty."""
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    if not rows.any() or not cols.any():
        return None
    r = np.nonzero(rows)[0]
    c = np.nonzero(cols)[0]
    return int(c[0]), int(r[0]), int(c[-1] - c[0] + 1), int(r[-1] - r[0] + 1)


def _curated_crop_and_resize(
    rgb: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    target_side: int,
    target_size: int = CURATED_TARGET_SIZE,
) -> Image.Image:
    """Center-crop a `target_side` x `target_side` window on the bbox, resize to target_size.

    A single shared `target_side` is used for every frame (so the on-canvas
    scale is constant across the clip), each frame is centered on its own
    per-frame bbox, and any portion of the mask that exceeds `target_side` is
    clipped at the canvas boundary. BICUBIC downsampling produces soft-alpha
    edges close to the curated assets while keeping bbox geometry within ~2 px.

    Args:
        rgb: (H, W, 3) uint8 RGB array.
        mask: (H, W) uint8 binary mask (0/255).
        bbox: (x, y, w, h) tight bbox of `mask` (precomputed).
        target_side: side length of the crop window in *original* pixels;
            shared across all frames of the clip.
        target_size: output side length in pixels.

    Returns:
        RGBA PIL image of shape (target_size, target_size).
    """
    x, y, w, h = bbox
    cx = x + w / 2.0
    cy = y + h / 2.0

    half = target_side / 2.0
    x_start = int(round(cx - half))
    y_start = int(round(cy - half))
    x_end = x_start + target_side
    y_end = y_start + target_side

    H, W = mask.shape
    pad_top = max(0, -y_start)
    pad_bottom = max(0, y_end - H)
    pad_left = max(0, -x_start)
    pad_right = max(0, x_end - W)

    rgb_padded = np.pad(
        rgb,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        constant_values=0,
    )
    mask_padded = np.pad(
        mask,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        constant_values=0,
    )
    ys = y_start + pad_top
    xs = x_start + pad_left
    rgb_crop = rgb_padded[ys : ys + target_side, xs : xs + target_side]
    mask_crop = mask_padded[ys : ys + target_side, xs : xs + target_side]

    rgba = np.concatenate([rgb_crop, mask_crop[..., None]], axis=-1)
    # BICUBIC: more soft-alpha values than BILINEAR while keeping bbox geometry
    # within ~2 px of curated (LANCZOS spreads ~6 px and was rejected).
    return Image.fromarray(rgba, mode="RGBA").resize(
        (target_size, target_size), Image.BICUBIC
    )


def load_per_object_from_rgb_seg_dirs(
    rgb_dir: str | Path,
    seg_dir: str | Path,
    max_frames: int | None = None,
    stride: int = 1,
    background_id: int = 0,
    groups: list[list[int]] | None = None,
    start: int = 0,
) -> tuple[dict[str, ActionMeshInput], list[str]]:
    """
    Load a frame sequence from explicit RGB and segmentation directories,
    splitting one ActionMeshInput per non-background object id present in the
    palette segmentation masks.

    Frames are emitted in 512x512 RGBA, similar to the curated example assets
    (e.g. assets/examples/davis_camel/*.png). A single shared crop side equal
    to `max per-frame max_dim` (computed per object across the clip) is used
    for every frame, each frame is centered on its own bbox, then resized to
    512x512 with BICUBIC. Because the scale is constant across the clip, the
    on-canvas object size grows/shrinks naturally as the bbox does. The
    "shared = max per-frame max_dim" choice guarantees the object always fits
    inside the canvas (no clipping).

    Args:
        rgb_dir: Path to directory of RGB images.
        seg_dir: Path to directory of palette PNG segmentation masks.
        max_frames: Maximum number of frames to load. None for all frames.
        stride: Take every nth frame (default=1).
        background_id: Palette index treated as background (default=0).
        start: First frame index to start from (default=0), applied before stride.
        groups: Optional partition of palette ids into objects. Each inner list
            is merged into a single ActionMeshInput by OR-ing the masks of its
            ids (e.g. ``[[1, 2], [3]]`` reconstructs ids 1+2 as one object and 3
            as another). Ids not present in the masks are dropped; ``None`` (the
            default) puts each id in its own group (one object per id).

    Returns:
        ({group_label: ActionMeshInput}, [rgb_filename, ...]) — the dict is keyed
        by the group's ``'+'``-joined ids (e.g. ``'1+2'``, or just ``'1'`` when
        ungrouped); the second element lists the original RGB filenames that were
        selected after applying stride and max_frames.
    """
    rgb_dir = Path(rgb_dir)
    seg_dir = Path(seg_dir)
    if not rgb_dir.is_dir():
        raise ValueError(f"RGB directory not found: '{rgb_dir}'")
    if not seg_dir.is_dir():
        raise ValueError(f"Segmentation directory not found: '{seg_dir}'")

    rgb_paths = natsorted(
        p for p in rgb_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    seg_paths = natsorted(
        p for p in seg_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not rgb_paths:
        raise ValueError(f"No images found in '{rgb_dir}'")
    if len(rgb_paths) != len(seg_paths):
        raise ValueError(
            f"Mismatched frame counts: rgb={len(rgb_paths)}, seg={len(seg_paths)}"
        )

    rgb_paths = rgb_paths[start::stride]
    seg_paths = seg_paths[start::stride]
    if max_frames is not None:
        rgb_paths = rgb_paths[:max_frames]
        seg_paths = seg_paths[:max_frames]

    rgbs = [Image.open(p).convert("RGB") for p in rgb_paths]
    segs: list[np.ndarray] = []
    for sp, rgb in zip(seg_paths, rgbs):
        s = Image.open(sp)
        if s.mode not in ("P", "L"):
            s = s.convert("L")
        if s.size != rgb.size:
            s = s.resize(rgb.size, Image.NEAREST)
        segs.append(np.array(s))

    object_ids = sorted(
        {int(v) for s in segs for v in np.unique(s) if int(v) != background_id}
    )
    if not object_ids:
        raise ValueError(f"No non-background ids found in '{seg_dir}'")

    # Resolve groups: default = one object per id. Drop ids absent from the masks
    # and any group left empty; fall back to per-id if nothing valid remains.
    if groups is None:
        resolved_groups = [[oid] for oid in object_ids]
    else:
        valid_ids = set(object_ids)
        resolved_groups = [[i for i in g if i in valid_ids] for g in groups]
        resolved_groups = [g for g in resolved_groups if g]
        if not resolved_groups:
            resolved_groups = [[oid] for oid in object_ids]

    timesteps = torch.arange(len(rgbs), dtype=torch.float32)

    rgbs_np = [np.asarray(rgb, dtype=np.uint8) for rgb in rgbs]

    result: dict[str, ActionMeshInput] = {}
    for group in resolved_groups:
        label = "+".join(str(i) for i in group)
        # Union the group's ids into one mask per frame.
        masks = [np.isin(seg, group).astype(np.uint8) * 255 for seg in segs]
        bboxes = [_bbox_of_mask(m) for m in masks]
        valid = [b for b in bboxes if b is not None]
        if not valid:
            raise ValueError(f"Object group {label} has no mask in any frame")
        # Shared crop side = max per-frame max_dim (across the clip), so the
        # largest bbox just fits the canvas and no frame ever clips the object.
        # Frames where the object is absent fall back to the first valid bbox
        # to keep a consistent canvas placement (their mask is fully zero, so
        # the output is a transparent square regardless of position).
        max_max_dim = max(max(b[2], b[3]) for b in valid)
        target_side = max_max_dim
        fallback_bbox = valid[0]

        frames = []
        for rgb_arr, mask, bbox in zip(rgbs_np, masks, bboxes):
            frames.append(
                _curated_crop_and_resize(
                    rgb_arr, mask, bbox or fallback_bbox, target_side
                )
            )
        result[label] = ActionMeshInput(frames=frames, timesteps=timesteps.clone())
        logger.info(
            f"group {label}: max_max_dim={max_max_dim} -> target_side={target_side} "
            f"(scale={CURATED_TARGET_SIZE / target_side:.3f})"
        )

    logger.info(
        f"Loaded {len(rgbs)} frames from rgb='{rgb_dir}', seg='{seg_dir}' "
        f"split into {len(result)} objects (groups: {resolved_groups})"
    )
    selected_frames = [p.name for p in rgb_paths]
    return result, selected_frames


def check_pytorch3d_installed() -> bool:
    """Check if pytorch3d is installed."""
    try:
        import pytorch3d  # noqa: F401

        return True
    except ImportError:
        logger.warning(
            "PyTorch3D is not installed. Video rendering will be skipped. "
            "See https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md"
        )
        return False


def check_blender_available(blender_path: str | None = None) -> bool:
    """Check if Blender is available and return the path to the executable."""
    if blender_path is None:
        logger.warning(
            "No Blender path provided. animated_mesh.glb will not be saved. "
            "Use --blender_path to specify your Blender 3.5.1 executable."
        )
        return False

    if os.path.isfile(blender_path) and os.access(blender_path, os.X_OK):
        return True
    else:
        logger.warning(
            f"Provided Blender path '{blender_path}' is not a valid executable. "
            "animated_mesh.glb will not be saved."
        )
        return False


@torch.no_grad()
def run_actionmesh(
    pipeline: ActionMeshPipeline,
    input: ActionMeshInput,
    output_dir: str,
    seed: int,
    blender_path: str | None = None,
    # -- Pipeline parameters
    stage_0_steps: int | None = None,
    face_decimation: int | None = None,
    floaters_threshold: float | None = None,
    stage_1_steps: int | None = None,
    guidance_scales: list[float] | None = None,
    anchor_idx: int | None = None,
):
    # -- Save loader output (RGBA, transparent bg) — directly comparable to
    #    assets/examples/davis_camel/*.png. Done BEFORE the pipeline runs
    #    because the pipeline's ImagePreprocessor mutates input.frames in place,
    #    compositing onto white and returning RGB.
    processed_dir = Path(output_dir) / "processed_frames"
    with _TIMER.phase("save_processed_frames"):
        processed_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(input.frames):
            frame.save(processed_dir / f"{i:02d}.png")

    # -- Run inference (mutates input.frames in place: bg-removed + cropped+padded)
    with _TIMER.phase("inference"):
        meshes = pipeline(
            input=input,
            seed=seed,
            stage_0_steps=stage_0_steps,
            face_decimation=face_decimation,
            floaters_threshold=floaters_threshold,
            stage_1_steps=stage_1_steps,
            guidance_scales=guidance_scales,
            anchor_idx=anchor_idx,
        )

    # -- Save meshes + T vertices + faces
    with _TIMER.phase("save_output"):
        save_meshes(meshes, output_dir=output_dir)
        vertices_path, faces_path = save_deformation(
            meshes, path=f"{output_dir}/deformations"
        )

    # -- [Optional] Create animated GLB (requires Blender 3.5.1)
    if check_blender_available(blender_path):
        animated_glb_path = f"{output_dir}/animated_mesh.glb"
        with _TIMER.phase("animated_glb"):
            create_animated_glb(
                blender_path=blender_path,
                vertices_npy=vertices_path,
                faces_npy=faces_path,
                output_glb=animated_glb_path,
                fps=8,
            )

    # -- [Optional] Render output (automatically if pytorch3d is installed)
    if check_pytorch3d_installed():
        from actionmesh.render.visualizer import ActionMeshVisualizer

        with _TIMER.phase("render"):
            visualizer = ActionMeshVisualizer(image_size=256)
            visualizer.render(
                meshes,
                input_frames=input.frames,
                device=pipeline.device,
                output_dir=output_dir,
            )


def _davis_dirs(root: str | Path, scene: str) -> tuple[Path, Path]:
    """``(rgb_dir, seg_dir)`` for a DAVIS scene — the single place the DAVIS
    ``JPEGImages``/``Annotations`` ``Full-Resolution`` layout is spelled."""
    base = Path(root)
    return (
        base / "JPEGImages" / "Full-Resolution" / scene,
        base / "Annotations" / "Full-Resolution" / scene,
    )


def _scene_rgb_seg_dirs(root: str | Path, scene: str) -> tuple[Path, Path]:
    """``(rgb_dir, seg_dir)`` for the per-scene ``rgbs/`` + ``segmentations/``
    layout that ``dataset=dyncustom`` reads."""
    base = Path(root) / scene
    return base / "rgbs", base / "segmentations"


def _rgb_and_seg_dirs(root: str | Path, scene: str):
    """``(rgb_dir, seg_dir, layout)`` for a scene whose masks are separate palette
    PNGs, or ``None`` when neither convention applies (the ``curated`` and
    ``video`` inputs carry their mask in an alpha channel instead).

    The single place "where are this scene's frames and masks" is answered.
    """
    davis_rgb, davis_seg = _davis_dirs(root, scene)
    if davis_rgb.is_dir():
        return davis_rgb, davis_seg, "davis"
    scene_rgb, scene_seg = _scene_rgb_seg_dirs(root, scene)
    if scene_rgb.is_dir() and scene_seg.is_dir():
        return scene_rgb, scene_seg, "rgb_seg"
    return None


# White-matte constants for the `video` layout.  244 is well below the ~255 the
# benchmark clips render their background at, and the 0.05%-of-frame floor clears
# the ~34 H.264 ringing specks per frame that survive the flood-fill (measured
# over the 20 V2M4 `simple` clips; every dropped component was < 0.04%).
_WHITE_THRESHOLD = 244
_MIN_COMPONENT_FRAC = 0.0005


def _load_white_bg_video(video_path: Path) -> list[Image.Image]:
    """Decode a white-background clip to RGBA frames in memory (alpha = silhouette).

    The silhouette is the complement of the near-white components that TOUCH THE
    FRAME BORDER — not a global white threshold, so white *inside* the object (an
    astronaut's suit, a skull, minion eyes) stays foreground.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames: list[Image.Image] = []
    fg_fracs = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        white = (rgb >= _WHITE_THRESHOLD).all(axis=-1).astype(np.uint8)
        _, lbl = cv2.connectedComponents(white, connectivity=4)
        border = np.concatenate([lbl[0], lbl[-1], lbl[:, 0], lbl[:, -1]])
        fg = ~np.isin(lbl, np.unique(border[border > 0]))

        n_fg, fg_lbl, stats, _ = cv2.connectedComponentsWithStats(
            fg.astype(np.uint8), connectivity=8
        )
        min_area = _MIN_COMPONENT_FRAC * rgb.shape[0] * rgb.shape[1]
        keep = [i for i in range(1, n_fg) if stats[i, cv2.CC_STAT_AREA] >= min_area]
        fg = np.isin(fg_lbl, keep)

        fg_fracs.append(float(fg.mean()))
        frames.append(Image.fromarray(np.dstack([rgb, (fg * 255).astype(np.uint8)]), "RGBA"))
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    lo, hi = min(fg_fracs), max(fg_fracs)
    logger.info(f"Decoded {len(frames)} frames from {video_path.name}, "
                f"foreground {lo:.3f}-{hi:.3f} of the image.")
    if lo < 0.01 or hi > 0.99:
        logger.warning("Degenerate silhouette — is this clip really on a white background?")
    return frames


def _repo_rel(path: str | Path) -> str:
    """`path` as a string relative to the repo root (portable metadata paths)."""
    return os.path.relpath(Path(path).resolve(), _REPO)


# -- Timing -----------------------------------------------------------------
# The headline number is the model's forward pass and nothing else.  Every other
# phase() is still measured and written out, just held out of it: disk I/O at
# both ends (reading the frames, writing the meshes), the processed_frames dump
# and the optional GLB/render passes all vary with the filesystem and the
# visualization flags rather than with the reconstruction.  Weight loading is
# excluded structurally rather than by name — build_pipeline() runs before
# run_job(), which is where the timer starts.
CORE_PHASES = ("inference",)


class RunTimer:
    """Wall-clock per phase for one run_job call, plus peak CUDA memory.

    Measures input -> output only.  Process startup, imports and the pipeline
    build (weights -> GPU) all happen before ``reset()``, so none of them are in
    ``total_core_seconds``, so a process that reuses one pipeline across jobs
    reports the same number as a cold CLI run.

    ``torch.cuda.synchronize()`` at every boundary, or an async kernel launched
    inside a phase is billed to whichever later phase happens to wait on it.
    """

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}
        self.per_object: dict[str, dict[str, float]] = {}
        self._object: str | None = None

    def reset(self) -> None:
        self.seconds.clear()
        self.per_object.clear()
        self._object = None
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @contextlib.contextmanager
    def object(self, name: str):
        """Attribute the phases inside to ``name`` as well as to the total."""
        self._object = name
        try:
            yield
        finally:
            self._object = None

    @contextlib.contextmanager
    def phase(self, name: str):
        self._sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            dt = time.perf_counter() - t0
            self.seconds[name] = self.seconds.get(name, 0.0) + dt
            if self._object is not None:
                obj = self.per_object.setdefault(self._object, {})
                obj[name] = obj.get(name, 0.0) + dt

    @staticmethod
    def _sync() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @property
    def total_core_seconds(self) -> float:
        return sum(self.seconds.get(k, 0.0) for k in CORE_PHASES)

    def to_dict(self) -> dict:
        r3 = lambda d: {k: round(v, 3) for k, v in d.items()}  # noqa: E731
        excluded = {k: v for k, v in self.seconds.items() if k not in CORE_PHASES}
        return {
            "total_core_seconds": round(self.total_core_seconds, 3),
            "core_phases": r3({
                k: self.seconds[k] for k in CORE_PHASES if k in self.seconds
            }),
            "excluded_phases": r3(excluded),
            "per_object": {k: r3(v) for k, v in self.per_object.items()},
            "peak_alloc_mb": round(
                torch.cuda.max_memory_allocated() / 1024**2, 1
            ) if torch.cuda.is_available() else 0.0,
            "peak_reserved_mb": round(
                torch.cuda.max_memory_reserved() / 1024**2, 1
            ) if torch.cuda.is_available() else 0.0,
        }


_TIMER = RunTimer()


def _parse_object_groups(specs: list[str] | None) -> list[list[int]] | None:
    """``["1,2", "3"]`` -> ``[[1, 2], [3]]`` (the CLI form of ``--object_groups``).

    It ends up as ``load_per_object_from_rgb_seg_dirs(groups=...)``, which unions
    each group's palette ids into one mask per frame.
    """
    if not specs:
        return None
    groups = []
    for spec in specs:
        try:
            ids = sorted({int(tok) for tok in spec.split(",") if tok.strip()})
        except ValueError:
            raise ValueError(
                f"--object_groups takes comma-separated integers, got '{spec}'"
            ) from None
        if not ids:
            raise ValueError(f"--object_groups got an empty group: '{spec}'")
        if 0 in ids:
            raise ValueError(
                f"--object_groups got palette id 0 in '{spec}', which is the "
                "background — grouping it would merge the background into an object."
            )
        groups.append(ids)
    return groups


def build_pipeline(fast: bool, low_ram: bool, dtype: str) -> ActionMeshPipeline:
    """Construct the ActionMesh pipeline for the given preset/precision flags.

    The fast/low_ram flags select the config name (and lazy loading); dtype picks
    bfloat16 vs float16. Pulled out of ``main`` so a caller can build it once and
    reuse it across jobs.
    """
    if fast and low_ram:
        config_name = "actionmesh_fast_lowram.yaml"
        logger.info("Fast + Low RAM mode enabled.")
    elif fast:
        config_name = "actionmesh_fast.yaml"
        logger.info("Fast mode enabled: quality might be slightly reduced.")
    elif low_ram:
        config_name = "actionmesh_lowram.yaml"
        logger.info("Low RAM mode enabled.")
    else:
        config_name = "actionmesh.yaml"

    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    config_dir = _ACTIONMESH / "actionmesh" / "configs"
    # Upstream reads its weights from ./pretrained_weights/<name>, downloading them there
    # when missing. Read them from the Hugging Face cache instead, like every other model:
    # skip its download, then repoint the folders it stored before any model loads.
    from huggingface_hub import snapshot_download

    from genia.core.download_weights import ACTIONMESH_WEIGHTS

    weights = {name: snapshot_download(repo) for repo, name in ACTIONMESH_WEIGHTS}
    actionmesh.pipeline.download_if_missing = lambda repo_id, local_dir: local_dir
    pipeline: ActionMeshPipeline = ActionMeshPipeline(
        config_name=config_name,
        config_dir=str(config_dir),
        dtype=torch_dtype,
        lazy_loading=low_ram,
    )
    for name, path in weights.items():  # _triposg_weights_dir, _dinov2_weights_dir, ...
        setattr(pipeline, f"_{name.lower()}_weights_dir", path)
    encoder = pipeline.cfg.model.image_encoder
    encoder.pretrained_dino_feature_extractor = encoder.pretrained_dino_model = weights["dinov2"]
    pipeline.to("cuda")
    return pipeline


def run_job(
    pipeline: ActionMeshPipeline,
    *,
    input_path: str,
    input_scene: str,
    input_stride: int = 1,
    input_length: int = 31,
    input_start: int = 0,
    output_dir: str | None = None,
    seed: int = 44,
    blender_path: str | None = None,
    stage_0_steps: int | None = None,
    face_decimation: int | None = None,
    floaters_threshold: float | None = None,
    stage_1_steps: int | None = None,
    guidance_scales: list[float] | None = None,
    anchor_idx: int | None = None,
    object_groups: list[list[int]] | None = None,
) -> str:
    """Detect the layout, load the scene, run ActionMesh per object.

    Returns the resolved output directory. Takes a built ``pipeline`` so it can be
    reused across jobs.
    """
    # -- Detect input layout. Four are supported, in two families:
    #    Separate palette masks (resolved by `_rgb_and_seg_dirs`), which split
    #    the scene per object:
    #      davis:   <input_path>/JPEGImages/Full-Resolution/<scene>/*.jpg
    #               <input_path>/Annotations/Full-Resolution/<scene>/*.png
    #      rgb_seg: <input_path>/<scene>/rgbs/*.png
    #               <input_path>/<scene>/segmentations/*.png  (the dataset=dyncustom
    #               layout)
    #    Mask carried in an alpha channel, single object:
    #      curated: <input_path>/<scene>/*.png  (flat dir of pre-processed RGBA
    #               PNGs, alpha channel is the mask — e.g. assets/examples/*).
    #      video:   <input_path>/<scene>.mp4  (white-background clip; decoded and
    #               matted in memory, so no extracted-frames copy hits disk).
    #
    # The mask-dir family is resolved FIRST: an rgb_seg scene dir holds no PNGs of
    # its own, so the curated branch would match it and then find zero frames.
    # Start the clock here: build_pipeline() (weights -> GPU) and every import
    # already ran, so what follows is input -> output and nothing else.
    _TIMER.reset()

    input_root = Path(input_path)
    curated_dir = input_root / input_scene
    video_path = input_root / f"{input_scene}.mp4"
    video_frames: list[Image.Image] = []
    # `seg_dir` is what the layouts differ in downstream: set for the mask-dir
    # family (where `scan_dir` is the rgb dir), None for the alpha-channel one.
    seg_dir: Path | None = None
    mask_dirs = _rgb_and_seg_dirs(input_root, input_scene)
    if mask_dirs is not None:
        scan_dir, seg_dir, layout = mask_dirs
    elif curated_dir.is_dir():
        layout = "curated"
        scan_dir = curated_dir
    elif video_path.is_file():
        layout = "video"
        scan_dir = video_path
        with _TIMER.phase("load_input"):  # decode+matte is this layout's input read
            video_frames = _load_white_bg_video(video_path)
    else:
        raise FileNotFoundError(
            f"No input found for scene '{input_scene}'. Tried the davis layout "
            f"'{_davis_dirs(input_root, input_scene)[0]}', the rgb_seg layout "
            f"'{_scene_rgb_seg_dirs(input_root, input_scene)[0]}', the curated "
            f"layout '{curated_dir}' and the video '{video_path}'."
        )
    logger.info(f"Detected {layout} layout at {scan_dir}")

    # -- Validate stride/length against the model's frame budget [16, 31]
    assert 16 <= input_length <= 31, (
        f"--input_length must be in [16, 31], got {input_length}"
    )
    assert input_stride >= 1, f"--input_stride must be >= 1, got {input_stride}"
    assert input_start >= 0, f"--input_start must be >= 0, got {input_start}"
    n_available = len(video_frames) if layout == "video" else sum(
        1 for p in scan_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    n_after_start = max(0, n_available - input_start)
    n_after_stride = (n_after_start + input_stride - 1) // input_stride
    assert n_after_stride >= input_length, (
        f"Not enough frames after start={input_start}, stride={input_stride}: "
        f"{n_after_stride} available from {n_available} in '{scan_dir}', "
        f"need at least --input_length={input_length}."
    )

    # -- Set default output directory if not provided
    if output_dir is None:
        output_dir = (
            f"outputs/{input_scene}_stride_{input_stride}_length_{input_length}"
        )
        logger.info(f"Output directory not specified, using: {output_dir}")

    # -- Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with _TIMER.phase("load_input"):
        if seg_dir is not None:      # davis / rgb_seg: palette masks beside the rgbs
            objects, selected_frames = load_per_object_from_rgb_seg_dirs(
                rgb_dir=scan_dir,
                seg_dir=seg_dir,
                max_frames=input_length,
                stride=input_stride,
                groups=object_groups,
                start=input_start,
            )
        elif layout == "video":
            # Video: already RGBA in memory, no per-object split. selected_frames
            # records the SOURCE frame numbers, so mesh_NN traces back to the frame
            # it came from (stride 2 → mesh_00 is frame 0, mesh_01 is frame 2, ...).
            idxs = list(range(input_start, n_available, input_stride))[:input_length]
            frames = [video_frames[i] for i in idxs]
            objects = {"0": ActionMeshInput(
                frames=frames, timesteps=torch.arange(len(frames), dtype=torch.float32)
            )}
            selected_frames = [f"{i:05d}.png" for i in idxs]
        else:
            # Curated: pre-processed RGBA PNGs, no per-object split. The vendored
            # load_from_image_dir has no start offset, so input_start is unsupported.
            if input_start > 0:
                raise ValueError(
                    "input_start (>0) is unsupported for the curated layout "
                    "(the vendored load_from_image_dir has no start offset)."
                )
            am_input = load_from_image_dir(
                curated_dir / "*.png",
                max_frames=input_length,
                stride=input_stride,
            )
            objects = {"0": am_input}  # str key like the palette group labels
            png_paths = natsorted(
                p for p in curated_dir.iterdir() if p.suffix.lower() == ".png"
            )
            png_paths = png_paths[::input_stride][:input_length]
            selected_frames = [p.name for p in png_paths]

    # -- Persist run metadata: the selected frames + the per-object palette
    #    grouping (which segmentation ids were merged into each obj_NNN).
    object_meta = [
        {
            "obj_idx": obj_idx,
            "dir": f"obj_{obj_idx:03d}",
            "palette_ids": [int(x) for x in key.split("+")],
            "label": key,
        }
        for obj_idx, key in enumerate(objects.keys())
    ]
    metadata = {
        "scene": input_scene,
        "images_dir": _repo_rel(scan_dir),
        "segmentations_dir": _repo_rel(seg_dir) if seg_dir else None,
        "input_start": input_start,
        "input_stride": input_stride,
        "input_length": input_length,
        "selected_frames": selected_frames,
        "objects": object_meta,
    }
    metadata_path = Path(output_dir) / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    logger.info(f"Wrote run metadata to {metadata_path}")
    for obj_idx, (palette_id, obj_input) in enumerate(objects.items()):
        obj_out_dir = f"{output_dir}/obj_{obj_idx:03d}"
        Path(obj_out_dir).mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Running ActionMesh for obj_{obj_idx:03d} "
            f"(palette id={palette_id}) → {obj_out_dir}"
        )
        with _TIMER.object(f"obj_{obj_idx:03d}"):
            run_actionmesh(
                pipeline,
                input=obj_input,
                output_dir=obj_out_dir,
                seed=seed,
                blender_path=blender_path,
                stage_0_steps=stage_0_steps,
                face_decimation=face_decimation,
                floaters_threshold=floaters_threshold,
                stage_1_steps=stage_1_steps,
                guidance_scales=guidance_scales,
                anchor_idx=anchor_idx,
            )

    # -- Persist timing. Written last, so it exists only for a run that finished
    #    — a killed run leaves no timing.json to be read as a complete one.
    timing = _TIMER.to_dict() | {
        "scene": input_scene,
        "objects": len(objects),
        "frames": input_length,
    }
    (Path(output_dir) / "timing.json").write_text(json.dumps(timing, indent=2))
    logger.info(
        f"Inference took {timing['total_core_seconds']:.1f}s "
        f"({len(objects)} object(s) x {input_length} frames), excluding startup, "
        f"weight loading and disk I/O — wrote {output_dir}/timing.json"
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="Root holding the scene, in any supported layout (davis / rgb_seg / curated / video — see the module docstring).",
    )
    parser.add_argument(
        "--input_scene",
        type=str,
        default=None,
        help="Scene name within --input_path (e.g. 'camel').",
    )
    parser.add_argument(
        "--input_start",
        type=int,
        default=0,
        help="First frame index to start from (applied before stride). Default: 0.",
    )
    parser.add_argument(
        "--input_stride",
        type=int,
        default=1,
        help="Frame stride: keep every Nth frame (e.g. 2 -> 0, 2, 4, ...). Default: 1.",
    )
    parser.add_argument(
        "--input_length",
        type=int,
        default=None,
        help="Number of frames to keep after striding. Must be in [16, 31]. Default: 31.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Output directory for generated meshes. "
            "Default: outputs/<input_scene>_stride_<S>_length_<L>"
        ),
    )
    parser.add_argument(
        "--object_groups",
        action="append",
        metavar="IDS",
        help=(
            "Comma-separated palette ids to UNION into one object (their masks "
            "are merged before the reconstruction runs, e.g. a horse and its "
            "rider as one mesh). Repeat for several objects: "
            "--object_groups 1,2 --object_groups 3. Only the davis/rgb_seg "
            "layouts carry a palette; the alpha-channel ones are single-object "
            "and ignore this. Passing it is EXHAUSTIVE — an id in no group is "
            "dropped, which is how an object is skipped. Default: one object "
            "per id."
        ),
    )
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument(
        "--blender_path",
        type=str,
        default=None,
        help="Path to Blender executable.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast preset (stage_0_steps=50, stage_1_steps=15).",
    )
    parser.add_argument(
        "--low_ram",
        action="store_true",
        help="Use low RAM preset (split_cfg_batch=true, clear_autocast=true).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bfloat16", "float16"],
        default="bfloat16",
        help="Data type for mixed precision inference. Default: bfloat16",
    )
    # -- Pipeline parameters
    parser.add_argument(
        "--stage_0_steps",
        type=int,
        default=None,
        help="Number of inference steps for image-to-3D (TripoSG). Default: 100. Fast: 50",
    )
    parser.add_argument(
        "--face_decimation",
        type=int,
        default=None,
        help="Target number of faces for mesh decimation. Default: 40000",
    )
    parser.add_argument(
        "--floaters_threshold",
        type=float,
        default=None,
        help="Threshold for removing floaters (0.0-1.0). Default: 0.02",
    )
    parser.add_argument(
        "--stage_1_steps",
        type=int,
        default=None,
        help="Number of flow-matching denoising steps in ActionMesh temporal 3D denoiser (Stage I). Default: 30. Fast: 15",
    )
    parser.add_argument(
        "--guidance_scales",
        type=float,
        nargs="+",
        default=None,
        help="Classifier-free guidance scales in ActionMesh temporal 3D denoiser. Default: [7.5]",
    )
    parser.add_argument(
        "--anchor_idx",
        type=int,
        default=None,
        help="Index of the anchor frame (fixing the topology). Default: 0",
    )
    args = parser.parse_args()

    if not args.input_path or not args.input_scene:
        parser.error("--input_path and --input_scene are required.")

    pipeline = build_pipeline(args.fast, args.low_ram, args.dtype)
    run_job(
        pipeline,
        input_path=args.input_path,
        input_scene=args.input_scene,
        input_stride=args.input_stride,
        input_length=args.input_length if args.input_length is not None else 31,
        input_start=args.input_start,
        output_dir=args.output_dir,
        seed=args.seed,
        blender_path=args.blender_path,
        stage_0_steps=args.stage_0_steps,
        face_decimation=args.face_decimation,
        floaters_threshold=args.floaters_threshold,
        stage_1_steps=args.stage_1_steps,
        guidance_scales=args.guidance_scales,
        anchor_idx=args.anchor_idx,
        object_groups=_parse_object_groups(args.object_groups),
    )


if __name__ == "__main__":
    main()
