"""Content-keyed disk cache for the SAM3D shape pre-pass (``parallel_shape`` /
the ``shape_init`` block of the mono_ours / mv_ours / ours split).

The shape pre-pass runs the full joint shape+pose ODE but keeps only the
per-frame **shape** latents — ``pose_init`` re-denoises pose afterwards, so the
pass's pose output is discarded (see
``core/configs/shape_and_poses_init/parallel_shape.yaml``). This cache stores those
shape latents so a rerun skips the (costly, ~ODE-solve) ``run_parallel_shape_
and_poses_init`` call. It is **cross-experiment** (unlike the block-level
``cache/{block}.pt`` PipelineState cache, which is keyed on the results-path and
so recomputes under a new experiment dir): runs that vary only later blocks
reuse one shape per object.

Gate: enabled only when the shape-pass sub-config carries ``shape_cache_dir``
(present in ``parallel_shape.yaml``, absent from its ``parallel`` base), so
only the pose-discarding pre-pass caches. Keyed on everything that determines the joint solve's shape output: the
seed, the resolved shape sub-config, the conditioning determinants (resolved
``processing`` + ``dataset`` config and the per-frame masked images), and the
input canonical shape latent. Any change misses to a fresh entry; the checkpoint
weights / backbone code are NOT hashed (bump ``_CACHE_KEY_VERSION`` or delete
``cached_shapes/sam3d/`` after changing them).

Artifact: one ``.pt`` per object holding ``{"perframe_shape": {fk_str: tensor},
"canonical_shape": tensor}`` (+ a ``.json`` provenance sidecar).
"""
import datetime
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

# Atomic write + corrupt-degrades-to-a-miss read.
from genia.core.paths import GENIA_ROOT
from genia.core.utils.disk_cache import (GENERATION_SECONDS, PEAK_ALLOC_MB,
                                     load_blob, store_blob)

_CACHE_KEY_VERSION = "v1"


def fk_str(fk) -> str:
    """Stable string id for a FrameKey (or bare int) — the per-frame dict key."""
    return str(tuple(fk)) if hasattr(fk, "__iter__") else str(fk)


def _masked_rgba(frame_data, obj_idx):
    """RGBA PIL image of one object in a frame (RGB + alpha = its GT mask), or
    ``None`` when the object's mask is empty in this frame."""
    from PIL import Image

    mask = np.asarray(frame_data.masks[obj_idx]).astype(bool)
    if not mask.any():
        return None
    rgb = np.asarray(frame_data.image)[..., :3].astype(np.uint8)
    return Image.fromarray(np.dstack([rgb, (mask * 255).astype(np.uint8)]), mode="RGBA")


def _cond_images(sequence, obj_idx, frame_keys):
    """Per-frame masked RGBA (the pass's image conditioning), None where empty."""
    return [_masked_rgba(sequence[fk], obj_idx) for fk in frame_keys]


def _cache_key(sp_cfg, cfg, seed, frame_keys, cond_images,
               canonical_shape_latent) -> str:
    """16-hex key over the joint solve's shape determinants."""
    h = hashlib.sha256()
    h.update(f"{_CACHE_KEY_VERSION}|seed={int(seed)}".encode())
    # Resolved config: the shape sub-config (all joint-solve knobs) + the
    # conditioning determinants (depth/pose source, downscale, num_views, ...).
    for node in (sp_cfg, cfg.processing, cfg.dataset):
        blob = json.dumps(OmegaConf.to_container(node, resolve=True),
                          sort_keys=True, default=str)
        h.update(b"|cfg|")
        h.update(blob.encode())
    # Per-frame conditioning content (order-sensitive), belt-and-suspenders
    # over the config dicts.
    for fk, im in zip(frame_keys, cond_images):
        h.update(f"|f:{fk_str(fk)}:".encode())
        if im is not None:
            h.update(f"{im.mode}:{im.size[0]}x{im.size[1]}:".encode())
            h.update(np.asarray(im).tobytes())
    # Input canonical shape latent (folds in any upstream shape state, e.g. GT injection).
    if canonical_shape_latent is not None:
        t = canonical_shape_latent.detach().to("cpu", torch.float32).contiguous()
        h.update(b"|z|")
        h.update(t.numpy().tobytes())
    return h.hexdigest()[:16]


def cache_path(cache_dir, sequence, obj_idx, frame_keys,
               sp_cfg, cfg, canonical_shape_latent, seed):
    """Cache file for one object's shape-pass output, or ``None`` when caching
    is off. Relative dirs resolve against the genia checkout."""
    if not cache_dir:
        return None
    root = Path(cache_dir)
    if not root.is_absolute():
        root = GENIA_ROOT / root
    key = _cache_key(sp_cfg, cfg, seed, frame_keys,
                     _cond_images(sequence, obj_idx, frame_keys),
                     canonical_shape_latent)
    return (root / "sam3d" / str(sequence.dataset_type)
            / str(sequence.scene_name).replace(os.sep, "_")
            / f"obj{obj_idx:03d}_{len(frame_keys)}f_seed{int(seed)}_{key}.pt")


def load(path, device):
    """Cached ``{"perframe_shape": {fk_str: tensor}, "canonical_shape": tensor}``
    with tensors moved to ``device``, or ``None`` on miss / corrupt file."""
    blob = load_blob(path, device, log_prefix="    [shape-cache]")
    if blob is None:
        return None
    return {
        "perframe_shape": {k: v.to(device)
                           for k, v in blob["perframe_shape"].items()},
        "canonical_shape": blob["canonical_shape"].to(device),
    }


def store_shape_tokens(path, perframe, canonical_shape, meta=None,
                       log_prefix="    [shape-cache] cached shape_init shapes") -> None:
    """Write the on-disk shape-token artifact — the
    ``{"perframe_shape": {fk_str: tensor}, "canonical_shape": tensor}`` blob
    (+ optional ``.json`` sidecar), atomically (tmp + ``os.replace``, safe
    under concurrent writers), from an already-built ``perframe`` dict."""
    store_blob(path,
               {"perframe_shape": perframe,
                "canonical_shape": canonical_shape.detach().cpu()},
               meta=meta, log=log_prefix)


def store(path, tokens_list, canonical_shape, meta=None) -> None:
    """Atomically write the per-frame shape latents + canonical shape, plus a
    provenance sidecar. Delegates the schema + atomic write to
    ``store_shape_tokens``; builds ``perframe`` from the pass's raw modalities."""
    if path is None:
        return
    perframe = {
        fk_str(fk): di["raw_ss_modalities"]["shape"].detach().cpu()
        for fk, di in tokens_list
        if di.get("raw_ss_modalities", {}).get("shape") is not None
    }
    store_shape_tokens(path, perframe, canonical_shape, meta=meta)


def cache_meta(sequence, obj_idx, frame_keys, seed,
               generation_seconds=None, peak_alloc_mb=None) -> dict:
    """Provenance sidecar content (advisory — never read by the lookup).

    ``generation_seconds`` is how long the shape pass took, so a later run that
    HITS this entry can charge itself that time
    (:func:`~genia.core.utils.disk_cache.credit_cached_work`) instead of reporting
    the fraction of a second the load took.  ``peak_alloc_mb`` is the same idea for
    memory: without it a hit reports the LOAD's footprint as the shape pass's."""
    return {
        "dataset": sequence.dataset_type,
        "scene": sequence.scene_name,
        "obj_idx": obj_idx,
        "n_frames": len(frame_keys),
        "frame_keys": [fk_str(fk) for fk in frame_keys],
        "is_dynamic": sequence.is_dynamic,
        "seed": int(seed),
        "cache_key_version": _CACHE_KEY_VERSION,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        **({GENERATION_SECONDS: round(float(generation_seconds), 3)}
           if generation_seconds is not None else {}),
        **({PEAK_ALLOC_MB: round(float(peak_alloc_mb), 1)} if peak_alloc_mb else {}),
    }
