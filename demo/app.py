#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""The demo app: pick or upload a scene, mask its objects with SAM3, reconstruct it.

    python demo/app.py                    # http://127.0.0.1:7860
    python demo/app.py --dev              # also the step table, the log and the run's files
    python demo/app.py --share            # also a public Gradio link
    python demo/app.py --work-dir DIR     # uploads and runs (default: results/demo_app)
    python demo/app.py --in-process       # the method in this process (ZeroGPU), no worker

The method it runs is ``demo/method.py``, the one file here that names it (GenIA): its
modes (``run_demo.py`` settings), run options and steps. Each mode starts from the bundled
scenes in ``demo/data``; "Add your own" masks an upload with SAM3, by naming the objects or
by clicking them on the first view (SAM3 carries the clicks to the other views), and adds it
to the scenes. **Reconstruct** runs ``demo/run_demo.py``'s recipe on the selected scene,
``--low-vram`` included, in a warm worker (``ui/worker.py``): one process that loads the
method's models as the app starts and keeps them between runs, so a click starts
reconstructing at once. The 3D view shows the input cameras and depth cloud as soon as
preprocessing has run, then the reconstruction. For a bundled example with a recorded run
(``demo/record_examples.py``, into ``<work-dir>/replays``) Reconstruct plays that recording
instead, in ~15 s and without a GPU, unless **Play recorded runs** is unticked.

The layout and the 3D view (3DView, ``submodules/3DView``, built by ``install.sh``) follow the
GenerativeReconstruction framework's demo app; the look is 3DView's own (``demo/ui/style.py``).

SAM3 runs in a child process, started on the first segmentation. On a GPU under
``ROOMY_GPU_GIB`` it is ended before each reconstruction (the two would not fit a 24 GB GPU
together); a larger one keeps it, and the method keeps more of its own models loaded too.
"""
import argparse
import functools
import json
import multiprocessing
import os
import queue
import re
import shlex
import threading
import time
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

import method
from ui import common as gc
from ui import replay as replay_mod
from ui import viewer as gv
from ui.inprocess import InProcess, gpu
from ui.worker import Worker

DATA = method.ROOT / "demo" / "data"
#: From this much GPU memory, SAM3 and every model the method can keep stay loaded between
#: runs; below it only what fits next to a run does (a 24 GB card).
ROOMY_GPU_GIB = 40
#: A ZeroGPU Space's GPU (its default size, half an RTX Pro 6000 Blackwell), which
#: ``nvidia-smi`` cannot see there outside a run.
ZEROGPU_GIB = 48

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
#: Uploads are downscaled to this long side (the bundled scenes are at most 1024 px).
MAX_SIDE = 1024
_PREVIEW_MAX = 512     # longest side shipped to the browser
_THUMB_MAX = 320       # longest side of an example-grid thumbnail
_GRID_COLS = 4
_CARD_COLS = 6       # example-scene cards per row: narrow enough to crop a photo only a little
_CARD_PX = 170
LOG_LINES = 200
#: Where the run's output splits into lines: after "\n", or after a "\r" not followed by "\n".
_LINE_END = re.compile(r"(?<=\n)|(?<=\r)(?!\n)")
#: Outline thickness of a mask preview in DISPLAYED pixels, in a `_TILE_PX` tile.
_OUTLINE_PX = 2
#: Click marker radius, as a fraction of the image's short side.
_MARKER_FRAC = 0.012
_POSITIVE_RGB, _NEGATIVE_RGB = (60, 220, 90), (240, 70, 70)

# Gallery tiles at a fixed size (Gradio's grid makes one image fill a wide column).
_TILE_PX = 150
_GALLERY_PX = 170
_CSS = f"""
.scene-tiles {{ padding-top: 40px; }}
.scene-summary, .scene-summary p {{ text-align: center; }}
.scene-tiles .grid-wrap {{ height: {_GALLERY_PX}px !important; min-height: 0 !important;
                           max-height: none !important; }}
.scene-tiles .grid-container {{
  grid-template-columns: repeat(auto-fill, minmax(0, {_TILE_PX}px)) !important;
  grid-template-rows: none !important;
  grid-auto-rows: minmax(0, {_TILE_PX}px) !important;
  justify-content: start;
}}
.scene-tiles .thumbnail-item {{ max-width: {_TILE_PX}px; max-height: {_TILE_PX}px; }}
.scene-card button, .scene-card .image-frame {{ width: 100%; height: 100%; }}
.scene-card img {{ object-fit: cover !important; width: 100%; height: 100%; }}
{gc.CSS}
"""


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


# --- images and masks ------------------------------------------------------------------

def palette():
    """The DAVIS palette: instance id -> colour, as the bundled masks use."""
    pal = []
    for i in range(256):
        c, cid = [0, 0, 0], i
        for j in range(8):
            for k in range(3):
                c[k] |= ((cid >> k) & 1) << (7 - j)
            cid >>= 3
        pal += c
    return pal


def load_view(path):
    """An RGB image, downscaled to ``MAX_SIDE`` on its long side."""
    img = Image.open(path).convert("RGB")
    scale = MAX_SIDE / max(img.size)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    return np.array(img)


def save_mask(mask, path):
    img = Image.fromarray(mask.astype(np.uint8), mode="P")
    img.putpalette(palette())
    img.save(path)


def _fit(im: Image.Image, size=None, resample=Image.BILINEAR, max_side=_PREVIEW_MAX):
    """Downscale to ``size``, or to fit ``max_side``, for the browser."""
    if size is None:
        if max(im.size) <= max_side:
            return im
        s = max_side / max(im.size)
        size = tuple(max(1, round(d * s)) for d in im.size)
    return im if im.size == size else im.resize(size, resample)


def _instance_borders(ids: np.ndarray) -> np.ndarray:
    """True on the object side of every instance boundary: an outline per instance."""
    from scipy.ndimage import grey_dilation, grey_erosion
    width = max(1, round(_OUTLINE_PX * max(ids.shape) / _TILE_PX))
    size = 2 * width + 1
    return (grey_dilation(ids, size=size) != grey_erosion(ids, size=size)) & (ids > 0)


def _preview(rgb_path: Path, mask_path: Path):
    """``(overlay, ids)``: the RGB outlined in its instances' palette colours, and the ids."""
    rgb_im = _fit(Image.open(rgb_path).convert("RGB"))
    mask_im = _fit(Image.open(mask_path), rgb_im.size, Image.NEAREST)
    rgb = np.array(rgb_im, dtype=np.float32)
    ids = np.array(mask_im)
    colours = np.zeros((256, 3), np.uint8)
    pal = mask_im.getpalette()
    if pal:
        flat = np.array(pal, np.uint8).reshape(-1, 3)
        colours[:len(flat)] = flat
    else:
        colours[1:] = (255, 0, 0)
    overlay = np.where(_instance_borders(ids)[..., None], colours[ids].astype(np.float32), rgb)
    return overlay.astype(np.uint8), {int(i) for i in np.unique(ids) if i}


# --- point prompts (the clicked objects, a plain list; the last one is being clicked) ---

def _upload_list(uploads):
    return [uploads] if isinstance(uploads, str) else list(uploads or [])


def _first_upload_frame(uploads):
    paths = sorted(_upload_list(uploads), key=lambda p: _natural_key(Path(p).name))
    if not paths:
        return None
    try:
        return np.asarray(_fit(Image.open(paths[0]).convert("RGB")))
    except OSError:
        return None


def _pt_add(objects, rel_xy, positive: bool, name: str):
    objects = [dict(o, points=list(o["points"]), labels=list(o["labels"])) for o in objects or []]
    if not objects:
        objects = [{"name": name or "object", "points": [], "labels": []}]
    objects[-1]["points"].append([float(rel_xy[0]), float(rel_xy[1])])
    objects[-1]["labels"].append(1 if positive else 0)
    return objects


def _pt_next_object(objects, name: str):
    objects = list(objects or [])
    if objects and not objects[-1]["points"]:
        return objects
    return objects + [{"name": name or "object", "points": [], "labels": []}]


def _pt_undo(objects):
    objects = [dict(o, points=list(o["points"]), labels=list(o["labels"])) for o in objects or []]
    while objects and not objects[-1]["points"]:
        objects.pop()
    if objects:
        objects[-1]["points"].pop()
        objects[-1]["labels"].pop()
        if not objects[-1]["points"]:
            objects.pop()
    return objects


def _pt_summary(objects) -> str:
    objects = [o for o in objects or [] if o["points"]]
    if not objects:
        return "_Click the object to segment. Positive adds, negative carves away._"
    return " · ".join(
        f"**{i}. {o['name']}** — {sum(o['labels'])}+ / {len(o['labels']) - sum(o['labels'])}−"
        for i, o in enumerate(objects, start=1))


def _pt_draw(base, objects):
    """``base`` with every click marked: filled for positive, ringed for negative."""
    if base is None:
        return None
    im = Image.fromarray(np.asarray(base)).convert("RGB")
    draw = ImageDraw.Draw(im)
    r = max(3, round(min(im.size) * _MARKER_FRAC))
    for o in objects or []:
        for (rx, ry), label in zip(o["points"], o["labels"]):
            x, y = rx * im.width, ry * im.height
            colour = _POSITIVE_RGB if label == 1 else _NEGATIVE_RGB
            draw.ellipse((x - r, y - r, x + r, y + r), fill=colour if label == 1 else None,
                         outline=(255, 255, 255) if label == 1 else colour, width=max(2, r // 3))
    return np.asarray(im)


# --- SAM3 ------------------------------------------------------------------------------

def _propagate(predictor, session_id, start):
    from sam3.visualization_utils import prepare_masks_for_visualization
    out = {r["frame_index"]: r["outputs"] for r in predictor.handle_stream_request(
        request=dict(type="propagate_in_video", session_id=session_id, start_frame_index=start))}
    return prepare_masks_for_visualization(out)   # {frame: {obj_id: bool mask}}


def _segment(predictor, frames_dir, n_frames, shape, names, objects):
    """Instance-id masks (one per view) and the ``classes.json`` index.

    ``names``: open-vocabulary prompts, each finding every instance it matches (prompted on
    the middle view and propagated both ways, so an object first detected late still reaches
    the first view). ``objects``: clicked objects, ``{"name", "points" (relative x, y on
    view 0), "labels" (1 positive, 0 negative)}``, one id each.
    """
    masks = [np.zeros(shape, np.uint8) for _ in range(n_frames)]
    classes = {}
    sid = predictor.handle_request(
        request=dict(type="start_session", resource_path=frames_dir))["session_id"]
    try:
        if names:
            mid = n_frames // 2
            for name in names:
                predictor.handle_request(request=dict(
                    type="add_prompt", session_id=sid, frame_index=mid, text=name))
                found = _propagate(predictor, sid, mid)
                for sam_id in sorted({i for objs in found.values() for i in objs}):
                    new_id = len(classes) + 1
                    n_same = sum(c["class"] == name for c in classes.values())
                    classes[str(new_id)] = {"class": name, "instance": n_same + 1}
                    for f, objs in found.items():
                        if sam_id in objs:
                            masks[f][objs[sam_id]] = new_id
        else:
            for obj_id, obj in enumerate(objects, start=1):
                predictor.handle_request(request=dict(
                    type="add_prompt", session_id=sid, frame_index=0, points=obj["points"],
                    point_labels=obj["labels"], obj_id=obj_id, rel_coordinates=True))
            found = _propagate(predictor, sid, 0)
            for obj_id, obj in enumerate(objects, start=1):
                n_same = sum(c["class"] == obj["name"] for c in classes.values())
                classes[str(obj_id)] = {"class": obj["name"], "instance": n_same + 1}
                for f, objs in found.items():
                    if obj_id in objs:
                        masks[f][objs[obj_id]] = obj_id
    finally:
        predictor.handle_request(request=dict(type="close_session", session_id=sid))
    return masks, classes


def _sam3_worker(conn):
    """SAM3 in its own process: answers ``_segment`` requests until it is killed."""
    import torch
    from sam3.model_builder import build_sam3_video_predictor
    with torch.autocast("cuda", dtype=torch.bfloat16):
        predictor = build_sam3_video_predictor()
        while True:
            args = conn.recv()
            try:
                conn.send(("ok", _segment(predictor, *args)))
            except Exception as exc:
                conn.send(("error", f"{type(exc).__name__}: {exc}"))


class Segmenter:
    """SAM3's video predictor over a folder of views, in a child process started on first use.

    ``release`` ends that process: the only way to get all of SAM3's GPU memory back, since
    its ``shutdown()`` leaves ~1.2 GB behind per load, which a demo session would pile up.
    """

    def __init__(self):
        self.proc = self.conn = None

    def segment(self, frames_dir, n_frames, shape, names=None, objects=None):
        if self.proc is None:
            ctx = multiprocessing.get_context("spawn")
            self.conn, child = ctx.Pipe()
            self.proc = ctx.Process(target=_sam3_worker, args=(child,), daemon=True)
            self.proc.start()
            child.close()   # else a dead worker's end stays open here and recv() never returns
        try:
            self.conn.send((frames_dir, n_frames, shape, names, objects))
            status, out = self.conn.recv()
        except (EOFError, OSError):   # the worker died, e.g. out of memory
            self.release()
            raise gr.Error("SAM3 stopped unexpectedly; see the server log.")
        if status == "error":
            raise gr.Error(f"SAM3 failed: {out}")
        return out

    def release(self):
        if self.proc is not None:
            self.proc.kill()
            self.proc.join()
            self.proc = self.conn = None


#: SAM3, loaded in this process by ``InProcessSegmenter``.
_predictor = None


@gpu(lambda frames_dir, n_frames, *_: 20 + 4 * n_frames)   # seconds: a few per view
def _segment_on_gpu(frames_dir, n_frames, shape, names, objects):
    import torch

    with torch.autocast("cuda", dtype=torch.bfloat16):
        return _segment(_predictor, frames_dir, n_frames, shape, names, objects)


class InProcessSegmenter:
    """``Segmenter`` for the in-process app (ZeroGPU): SAM3 loads into this process as the app
    starts (placed on the GPU at start-up, as ZeroGPU wants) and stays, and each segmentation is
    one short GPU call (``_segment_on_gpu``). Nothing else here holds a GPU."""

    def __init__(self):
        global _predictor
        import torch
        from sam3.model_builder import build_sam3_video_predictor

        with torch.autocast("cuda", dtype=torch.bfloat16):
            _predictor = build_sam3_video_predictor()

    def segment(self, frames_dir, n_frames, shape, names=None, objects=None):
        try:
            return _segment_on_gpu(frames_dir, n_frames, shape, names, objects)
        except Exception as exc:   # noqa: BLE001 — said in the app, as the worker's are
            raise gr.Error(f"SAM3 failed: {type(exc).__name__}: {exc}")

    def release(self):
        pass   # kept loaded: ZeroGPU's GPU has room for it beside the method's models


# --- scenes on disk ----------------------------------------------------------------------

def stage(root, setting, scene, views, masks, classes):
    """Write a scene where ``run_demo.py --data-root root`` reads it, laid out as demo/data."""
    scene_dir = Path(root) / setting / scene
    if setting == "image":
        scene_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(views[0]).save(scene_dir / "0000.png")
        save_mask(masks[0], scene_dir / "0000_mask.png")
        classes_path = scene_dir / "0000_classes.json"
    else:
        for sub in ("rgbs", "segmentations"):
            (scene_dir / sub).mkdir(parents=True, exist_ok=True)
        for i, (view, mask) in enumerate(zip(views, masks)):
            Image.fromarray(view).save(scene_dir / "rgbs" / f"{i:05d}.png")
            save_mask(mask, scene_dir / "segmentations" / f"{i:05d}.png")
        classes_path = scene_dir / "segmentations" / "classes.json"
    classes_path.write_text(json.dumps(classes, indent=2))
    return scene_dir


def _scene_pairs(mode: str, scene_dir: Path):
    """``([(rgb, mask), …], classes_json)`` of a staged scene (bundled or uploaded)."""
    scene_dir = Path(scene_dir)
    if mode == "image":
        rgb = next((f for f in sorted(scene_dir.iterdir()) if f.suffix.lower() in _IMAGE_EXTS
                    and not f.stem.endswith("_mask")), None) if scene_dir.is_dir() else None
        if rgb is None:
            return [], None
        return [(rgb, scene_dir / f"{rgb.stem}_mask.png")], scene_dir / f"{rgb.stem}_classes.json"
    rgbs, seg = scene_dir / "rgbs", scene_dir / "segmentations"
    if not rgbs.is_dir():
        return [], None
    views = sorted(rgbs.iterdir(), key=lambda f: _natural_key(f.name))
    return ([(f, seg / f"{f.stem}.png") for f in views if f.suffix.lower() in _IMAGE_EXTS],
            seg / "classes.json")


def describe(mode: str, scene_dir):
    """``(gallery, summary)`` for a staged scene: its views outlined by their masks."""
    if not scene_dir:
        return [], ""
    scene = Path(scene_dir).name
    pairs, index_path = _scene_pairs(mode, scene_dir)
    if not pairs:
        return [], f"❌ `{scene}` has no input images."
    if missing := [m for _, m in pairs if not m.is_file()]:
        return [], f"❌ `{scene}` has no mask for {missing[0].name}."
    gallery, ids = [], set()
    for rgb, mask in pairs:
        overlay, view_ids = _preview(rgb, mask)
        gallery.append(overlay)
        ids |= view_ids
    if not ids:
        return gallery, f"❌ `{scene}` has empty masks: nothing to reconstruct."
    names = json.loads(index_path.read_text()) if index_path.is_file() else {}
    listed = ", ".join(f"{i}:{names[str(i)]['class']}" if str(i) in names else str(i)
                       for i in sorted(ids))
    return gallery, (f"✅ `{scene}` — {_count(len(pairs), 'view')}, "
                     f"{_count(len(ids), 'object')} [{listed}].")


# --- the app -----------------------------------------------------------------------------

def gpu_policy(in_process: bool = False) -> tuple[bool, bool]:
    """``(roomy, auto_low_vram)`` for this machine's GPU, read before the runner takes any:
    whether models stay loaded between runs (``ROOMY_GPU_GIB``), and the Auto low-VRAM mode.
    In process with no GPU to see, it is ZeroGPU's (``ZEROGPU_GIB``)."""
    memory = gc.gpu_memory_gib() or ((ZEROGPU_GIB,) * 2 if in_process else None)   # (total, free)
    return (bool(memory) and memory[0] >= ROOMY_GPU_GIB,
            bool(memory) and memory[1] < method.LOW_VRAM_BELOW_GIB)


def new_runner(roomy: bool, in_process: bool = False):
    """What runs the method: a warm worker process (``ui/worker.py``), or, for ZeroGPU, this
    process itself (``ui/inprocess.py``)."""
    if in_process:
        os.environ.update(method.ENV)   # read once per process: before the models load
        return InProcess(warm=method.worker_overrides(roomy))
    return Worker(method.ENTRY, {**os.environ, **method.ENV},
                  warm=method.worker_overrides(roomy), cwd=method.ROOT)


def build_demo(work_dir: Path, dev: bool = False, in_process: bool = False) -> gr.Blocks:
    """``dev`` adds the step table, the log and the run's files (and toggles to hide them);
    ``in_process`` runs the method in this process (ZeroGPU), not in a worker process."""
    segmenter = InProcessSegmenter() if in_process else Segmenter()
    roomy, auto_low_vram = gpu_policy(in_process)
    runner = new_runner(roomy, in_process)
    # On ZeroGPU there is no GPU to see outside a run: it comes with each run.
    gpu_error = (lambda: None) if in_process else gc.gpu_error
    if not gpu_error():
        runner.start()   # loads while the visitor picks a scene (in process: before launch)
    running: list = []   # the live run in progress, if any, as the callable that stops it
    replaying = dict.fromkeys(method.MODES)   # each mode panel's replay in progress, if any
    upload_root, run_root, thumb_root = work_dir / "data", work_dir / "runs", work_dir / "thumbs"
    replay_root = work_dir / "replays"   # demo/record_examples.py writes them

    def scenes(mode):
        """Bundled scenes, then this work dir's uploads, as scene-dir paths."""
        out = []
        for root in (DATA, upload_root):
            d = root / mode
            if d.is_dir():
                out += [str(p) for p in sorted(d.iterdir()) if p.is_dir()]
        return out

    def scene_cards(mode, names):
        cards = []
        for scene_dir in names:
            pairs, _ = _scene_pairs(mode, scene_dir)
            if not pairs:
                continue
            rgb = pairs[0][0]
            thumb = thumb_root / mode / f"{Path(scene_dir).name}.png"
            if not thumb.is_file() or thumb.stat().st_mtime < rgb.stat().st_mtime:
                thumb.parent.mkdir(parents=True, exist_ok=True)
                _fit(Image.open(rgb).convert("RGB"), max_side=_THUMB_MAX).save(thumb)
            cards.append({"dir": scene_dir, "name": Path(scene_dir).name, "thumb": str(thumb),
                          "views": len(pairs)})
        return cards

    def prepare(mode, uploads, classes, prompt_mode, points, scene_list):
        """Segment an upload with SAM3 and add it to the scenes, selected."""
        files = sorted(_upload_list(uploads), key=lambda p: _natural_key(Path(p).name))
        if not files:
            raise gr.Error("Upload the image first." if mode == "image"
                           else "Upload the views first.")
        if mode != "image" and len(files) < 2:
            raise gr.Error(f"{method.MODES[mode]} needs at least two views.")
        names = objects = None
        if prompt_mode == "Text":
            names = [n.strip() for n in classes.split(",") if n.strip()]
            if not names:
                raise gr.Error("Name at least one object, e.g. 'mug, plush toy'.")
        else:
            objects = [o for o in points or [] if o["points"]]
            if not objects or any(1 not in o["labels"] for o in objects):
                raise gr.Error("Every object needs at least one Positive click.")
        views = [load_view(f) for f in files]
        if len({v.shape for v in views}) > 1:
            raise gr.Error("All views must have the same size.")
        yield gr.skip(), gr.skip(), gr.skip(), "⏳ Segmenting with SAM3…"
        scene = time.strftime("upload_%Y%m%d_%H%M%S")
        frames_dir = work_dir / "sam3" / scene
        frames_dir.mkdir(parents=True, exist_ok=True)
        for i, view in enumerate(views):
            Image.fromarray(view).save(frames_dir / f"{i:05d}.png")
        try:
            masks, index = segmenter.segment(str(frames_dir), len(views), views[0].shape[:2],
                                             names, objects)
        except gr.Error as exc:   # said in the caption too, which still reads "Segmenting…"
            yield gr.skip(), gr.skip(), gr.skip(), f"❌ {exc.message}"
            return
        if not index:
            yield gr.skip(), gr.skip(), gr.skip(), "❌ SAM3 found nothing to segment."
            return
        scene_dir = str(stage(upload_root, mode, scene, views, masks, index))
        yield scene_dir, scene_list + [scene_dir], *describe(mode, scene_dir)

    def recorded(mode, scene_dir):
        """Whether a bundled example scene has a recording: here, or published (``REMOTE``,
        fetched when first played)."""
        if not scene_dir or Path(scene_dir).parent != DATA / mode:
            return False
        scene = Path(scene_dir).name
        return bool(replay_mod.recording(replay_root, mode, scene)
                    or replay_mod.is_published(mode, scene))

    def select_scene(mode, scene_dir):
        return scene_dir, *describe(mode, scene_dir), recorded_box(mode, scene_dir)

    def recorded_box(mode, scene_dir):
        """The "Play recorded runs" box shows only for a scene that has a recording."""
        return gr.update(visible=bool(recorded(mode, scene_dir)))

    def halt():
        if not running and not any(replaying.values()):
            return "Nothing is running."
        for stop in list(running):
            stop()   # a live run: the worker's process group, or the in-process run
        for playback in replaying.values():
            if playback is not None:
                playback.stop()   # still its panel's: it ends on "Stopped"
        return "⏹ Stopping…"

    def new_run_dir(scene_dir, prefix=""):
        run_dir = run_root / Path(scene_dir).name / (prefix + time.strftime("%Y%m%d_%H%M%S"))
        run_dir.parent.mkdir(parents=True, exist_ok=True)   # the log and manifest sit beside it
        return run_dir

    def run(mode, scene_dir, play, low_vram, extra, *ticked):
        """Reconstruct the selected scene, streaming ``[log, status, table, files, viewer, task
        bars]``: a bundled example with a recorded run plays it when ``play`` (the panel's
        "Play recorded runs" box) is ticked, without a GPU; anything else runs live."""
        if not scene_dir:
            raise gr.Error("Select or add a scene first.")
        if play and recorded(mode, scene_dir):
            yield from replay(mode, scene_dir)
            return
        if gpu := gpu_error():
            raise gr.Error(gpu)
        scene_dir = Path(scene_dir)
        run_dir = new_run_dir(scene_dir)
        if not roomy:
            segmenter.release()
        try:
            overrides = method.run_overrides(
                mode, scene_dir.parents[1], scene_dir.name,
                {"auto": auto_low_vram, "on": True, "off": False}[low_vram],
                [*(arg for on, (_, _, yes, no) in zip(ticked, method.OPTIONS)
                   for arg in (yes if on else no)),
                 *method.RUN_ARGS, *runner.warm, f"output.output_dir={run_dir}",
                 *shlex.split(extra or "")])
        except ValueError as exc:
            raise gr.Error(str(exc))
        running.append(runner.stop)
        try:
            yield from stream(run_dir, runner.run(overrides), lambda: runner.returncode,
                              abandon=runner.stop)
        finally:
            running.remove(runner.stop)
            runner.start()     # warm again for the next click (a no-op if it still is)

    def replay(mode, scene_dir):
        """Play the selected example's recorded run (Stop reaches it through ``replaying``),
        fetching a published one first if it is not here yet."""
        scene = Path(scene_dir).name
        root = replay_mod.recording(replay_root, mode, scene)
        if root is None:
            yield gr.skip(), "⬇ Fetching this example's recorded run…", *[gr.skip()] * 4
            try:
                root = replay_mod.fetch(replay_root, mode, scene)
            except OSError as exc:
                raise gr.Error(f"Could not fetch the recorded run ({exc}): untick "
                               "**Play recorded runs** to reconstruct it live.") from None
        playback = replaying[mode] = replay_mod.Replay(root)
        note = (f"▶ A recorded run of this example (it took {gc.mmss(playback.live_seconds)} "
                "live) · untick **Play recorded runs** to reconstruct it again")
        try:
            yield from stream(new_run_dir(scene_dir, "replay_"),
                              playback.run, lambda: playback.returncode, abandon=playback.stop,
                              done_status=note)
        finally:
            replaying[mode] = None

    def stream(run_dir, source, returncode, abandon, done_status=None):
        """Follow one run in ``run_dir``: ``source``'s raw output (or ``source(run_dir)``'s,
        a replay writing the run dir itself) into ``[log, status, table, files, viewer,
        task bars]``, once a second. ``abandon`` ends the run if the page goes away;
        ``done_status`` replaces the status line once it ended successfully."""
        chunks: queue.Queue = queue.Queue()

        def read():   # one line per item, ending kept: tqdm's "\r" means it is overwritten
            # The whole log also goes beside the run dir, so a failed run can be read later.
            pending = ""
            try:
                with open(run_dir.with_name(run_dir.name + ".log"), "w") as saved:
                    for chunk in (source(run_dir) if callable(source) else source):
                        saved.write(chunk)   # raw reads: any number of lines
                        *done, pending = _LINE_END.split(pending + chunk)
                        if done and not pending and done[-1].endswith("\r"):
                            pending = done.pop()   # its "\n" may be in the next read
                        for line in done:
                            chunks.put(line)
                if pending:
                    chunks.put(pending)
            finally:
                chunks.put(None)   # always: without it the page would wait forever

        threading.Thread(target=read, daemon=True).start()
        lines, overwrite, log, ended, t0 = [], False, "", False, time.time()
        tasks = {}   # the running block's reported loops: {task: (n, total, stopped)}
        viewer = gv.viewer_html(run_dir, dev=dev)   # emitted once; content arrives by manifest
        try:
            while True:
                deadline = time.time() + 1.0   # one page update a second, output or not
                while not ended and (left := deadline - time.time()) > 0:
                    try:
                        chunk = chunks.get(timeout=left)
                    except queue.Empty:
                        break
                    if chunk is None:
                        ended = True
                        break
                    log += chunk
                    text = chunk.rstrip("\r\n")
                    if gc.starts_block(text):
                        tasks = {}
                    # A progress line goes to the bars, not the log. It can arrive glued to
                    # the end of a tqdm fragment (stderr's "\r" pieces and stdout share a pipe).
                    if (cut := text.find(gc.PROGRESS_PREFIX)) >= 0:
                        if reported := gc.read_progress(text[cut:]):
                            tasks[reported[0]] = reported[1:]
                        text = text[:cut]
                        if not text:
                            continue
                    if overwrite and lines:
                        lines[-1] = text
                    else:
                        lines = (lines + [text])[-LOG_LINES:]
                    overwrite = chunk.endswith("\r")
                rc = returncode() if ended else None
                gv.write_manifest(run_dir, done=ended)
                status = gc.status_text(ended, rc, log, time.time() - t0, method.PHASES)
                if done_status and ended and rc == 0:
                    status = done_status
                files = [str(p) for p in (
                    run_dir / "final" / "meshes" / "world.glb",
                    *sorted((run_dir / "final" / "gaussians").glob("world*.ply")),
                    run_dir / "final" / "poses.json",
                    run_dir / "final" / "viz" / "orbit" / "orbit.mp4") if p.is_file()]
                yield ("\n".join(lines), status,
                       gc.block_progress(run_dir, log, method.NAME, method.STEPS),
                       gr.update(value=files, visible=bool(files)), viewer,
                       "" if ended else gc.task_bars(tasks))
                viewer = gr.skip()
                if ended:
                    return
        finally:
            if not ended:      # abandoned (the tab closed, or a newer run took over)
                abandon()

    def mode_panel(mode, visible):
        staged = scenes(mode)
        first = staged[0] if staged else None
        init_gallery, init_status = describe(mode, first)
        with gr.Column(visible=visible) as panel:
            selected = gr.State(first)
            scene_list = gr.State(staged)
            with gr.Row():
                with gr.Column(scale=1):
                    gallery = gr.Gallery(
                        label="Selected scene — mask overlay", value=init_gallery,
                        columns=_GRID_COLS, object_fit="contain", allow_preview=False,
                        show_share_button=False, elem_classes=["scene-tiles"])
                    status = gr.Markdown(init_status, elem_classes=["scene-summary"])
                    with gr.Accordion("Add your own", open=not staged):
                        if mode != "image":
                            upload = gr.File(
                                label="Views of one static scene (sorted by filename)",
                                file_count="multiple", file_types=["image"], type="filepath")
                        else:
                            upload = gr.Image(label="Image", type="filepath", height=260)
                        prompt_mode = gr.Radio(["Text", "Points"], value="Text",
                                               label="Prompt SAM3 with",
                                               info="Name the objects, or click them.")
                        classes = gr.Textbox(label="What to reconstruct (comma-separated)",
                                             value="object", placeholder="e.g. mug, plush toy")
                        with gr.Column(visible=False) as point_box:
                            canvas = gr.Image(label="Click the object (first view)",
                                              type="numpy", interactive=False, height=260,
                                              show_download_button=False, show_share_button=False)
                            pt_note = gr.Markdown(_pt_summary([]))
                            with gr.Row():
                                point_kind = gr.Radio(["Positive", "Negative"], value="Positive",
                                                      label="A click adds", scale=2)
                                obj_name = gr.Textbox(label="Object name", value="object", scale=1)
                            with gr.Row():
                                next_btn = gr.Button("Next object", size="sm")
                                undo_btn = gr.Button("Undo point", size="sm")
                                clear_btn = gr.Button("Clear points", size="sm")
                        pts_state = gr.State([])
                        base_state = gr.State(None)
                        prep_btn = gr.Button("Segment and add", variant="secondary")
                    with gr.Column(elem_classes=["controls-tile"]):
                        options = [gr.Checkbox(label=label, value=False, info=info)
                                   for label, info, *_ in method.OPTIONS]
                        play_recorded = gr.Checkbox(
                            label="Play recorded runs", value=True,
                            visible=bool(recorded(mode, first)),
                            info="This example has a recorded run: Reconstruct plays it (~15 s, "
                                 "no GPU) instead of running it; untick to run it live.")
                        low_vram = gr.Dropdown(
                            [("Auto (from the free GPU memory)", "auto"), ("On", "on"),
                             ("Off", "off")],
                            value="auto", label="Low-VRAM mode",
                            info="Slower, to fit a 24 GB GPU (run_demo.py --low-vram).")
                        extra = gr.Textbox(label="Extra Hydra overrides", value="", visible=dev,
                                           placeholder="key=value …")
                    run_btn, stop_btn = gc.run_stop_buttons("Reconstruct")
                with gr.Column(scale=2):
                    viewer = gr.HTML(gv.viewer_html(None, dev=dev), padding=False)
                    run_status = gr.Markdown("", elem_classes=["run-status"])
                    tasks_view = gr.HTML("", padding=False)
                    with gr.Row(visible=dev):
                        toggle = gr.Checkbox(label="Show run details", value=dev, container=False)
                        viewer_ui = gr.Checkbox(label="Show 3D controls", value=gv.UI_DEFAULT,
                                                container=False)
                        viewer_ui.change(None, inputs=viewer_ui, js=gv.UI_TOGGLE_JS)
                    table = gr.Markdown(visible=dev)
                    files = gr.Files(label="Results", visible=False)
                    log = gr.Textbox(label="Log", lines=20, max_lines=20, autoscroll=True,
                                     visible=dev)
                    toggle.change(lambda show: (gr.update(visible=show), gr.update(visible=show)),
                                  inputs=toggle, outputs=[table, log])
            run_outputs = [log, run_status, table, files, viewer, tasks_view]

            gr.HTML('<div class="section-title">Example scenes '
                    '<span>· click one to select it</span></div>', padding=False)

            @gr.render(inputs=[scene_list, selected])
            def _grid(names, current):
                cards = scene_cards(mode, names)
                for row in range(0, len(cards), _CARD_COLS):
                    with gr.Row():
                        for card in cards[row:row + _CARD_COLS]:
                            picked = ["selected"] if card["dir"] == current else []
                            with gr.Column(scale=1, min_width=120):
                                thumb = gr.Image(value=card["thumb"], height=_CARD_PX,
                                                 interactive=False, show_label=False,
                                                 show_download_button=False,
                                                 show_fullscreen_button=False, sources=[],
                                                 elem_classes=["scene-card", *picked])
                                gr.Markdown(f"**{card['name']}**  \n"
                                            f"{_count(card['views'], 'view')}")
                            thumb.select(functools.partial(select_scene, mode, card["dir"]),
                                         outputs=[selected, gallery, status, play_recorded])
                        for _ in range(_CARD_COLS - len(cards[row:row + _CARD_COLS])):
                            gr.Column(scale=1, min_width=120)   # a short last row keeps widths

        prompt_mode.change(
            lambda m: (gr.update(visible=m == "Text"), gr.update(visible=m != "Text")),
            inputs=prompt_mode, outputs=[classes, point_box])

        def load_canvas(ups):
            base = _first_upload_frame(ups)
            return base, base, [], _pt_summary([])

        upload.change(load_canvas, inputs=upload, outputs=[canvas, base_state, pts_state, pt_note])

        def redraw(objects, base):
            return _pt_draw(base, objects), objects, _pt_summary(objects)

        def click(objects, base, kind, name, evt: gr.SelectData):
            if base is None:
                return gr.skip(), gr.skip(), gr.skip()
            h, w = np.asarray(base).shape[:2]
            return redraw(_pt_add(objects, (evt.index[0] / w, evt.index[1] / h),
                                  kind == "Positive", name), base)

        canvas.select(click, inputs=[pts_state, base_state, point_kind, obj_name],
                      outputs=[canvas, pts_state, pt_note])
        next_btn.click(lambda o, b, n: redraw(_pt_next_object(o, n), b),
                       inputs=[pts_state, base_state, obj_name],
                       outputs=[canvas, pts_state, pt_note])
        undo_btn.click(lambda o, b: redraw(_pt_undo(o), b), inputs=[pts_state, base_state],
                       outputs=[canvas, pts_state, pt_note])
        clear_btn.click(lambda b: redraw([], b), inputs=base_state,
                        outputs=[canvas, pts_state, pt_note])
        prep_btn.click(functools.partial(prepare, mode),
                       inputs=[upload, classes, prompt_mode, pts_state, scene_list],
                       outputs=[selected, scene_list, gallery, status]).then(
            functools.partial(recorded_box, mode), inputs=selected, outputs=play_recorded)
        return panel, dict(run=run_btn, stop=stop_btn, handler=functools.partial(run, mode),
                           inputs=[selected, play_recorded, low_vram, extra, *options],
                           outputs=run_outputs)

    first_mode = next(iter(method.MODES))
    with gr.Blocks(title=method.NAME, theme=gc.theme(), css=_CSS, js=gc.startup_js()) as demo:
        with gc.header(method.TITLE):
            buttons = [gr.Button(label, size="sm",
                                 variant="primary" if mode == first_mode else "secondary")
                       for mode, label in method.MODES.items()]
        if gpu := gpu_error():
            gr.Markdown(f"> ⚠️ **{gpu}**")
        built = [mode_panel(mode, mode == first_mode) for mode in method.MODES]
        panels = [panel for panel, _ in built]

        def switch(target):
            return ([gr.update(visible=m == target) for m in method.MODES]
                    + [gr.update(variant="primary" if m == target else "secondary")
                       for m in method.MODES])

        for mode, button in zip(method.MODES, buttons):
            button.click(functools.partial(switch, mode), outputs=panels + buttons, queue=False)
        for _, w in built:
            gc.wire_run(**w, halt=halt, lock=buttons)
        gc.credits()
    return demo


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", default=str(method.ROOT / "results" / "demo_app"))
    parser.add_argument("--dev", action="store_true",
                        help="also show the step table, the log and the extra-overrides box")
    # Left to Gradio when not given: it reads GRADIO_SERVER_NAME / GRADIO_SERVER_PORT, else
    # 127.0.0.1:7860 — so a container sets its address (0.0.0.0) in its environment.
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--in-process", action="store_true",
                        help="run the method in this process, not a worker (ZeroGPU)")
    args = parser.parse_args()
    work_dir = Path(args.work_dir).resolve()
    build_demo(work_dir, dev=args.dev, in_process=args.in_process).queue(
        default_concurrency_limit=1).launch(
        server_name=args.host, server_port=args.port, share=args.share,
        allowed_paths=[str(work_dir), str(gv.VIEWER_PAGE.parent), gv.bundle_root()])


if __name__ == "__main__":
    main()
