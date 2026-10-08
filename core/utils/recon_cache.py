"""Per-scene disk cache for the reconstruction backend (map-anything).

A reconstruction model's output is a pure function of its inputs -- the images, the
three optional conditioning signals, and which checkpoint is loaded. It does NOT depend
on the pipeline recipe, which is what makes caching it safe where caching a pipeline
BLOCK is not: ``output.save_cache=false`` / ``processing.resume_from_cache=false`` exist
so a stale block hit cannot skip work the config asked for, and a content-keyed
depth cache cannot go stale that way.

Useful for demos, where the same example scene is re-run repeatedly. The **model id is
part of the key**, so a result from one checkpoint is never served to another.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

CACHE_VERSION = 1          # bump to invalidate every cache on a format change


@dataclass
class ReconResult:
    """Per-frame output from a reconstruction backend (map-anything)."""

    depth_map_z: np.ndarray
    """Z-depth map, shape (H, W), float32."""

    K_matrix: np.ndarray
    """Camera intrinsics, shape (3, 3), float32."""

    valid_mask: np.ndarray
    """Boolean validity mask, shape (H, W)."""

    c2w: np.ndarray
    """Camera-to-world transform, shape (4, 4), float32."""

    cropped_image: Optional[np.ndarray] = None
    """RGB image cropped/resized to match the depth map, shape (H, W, 3), uint8.
    Set when the model predicted depth (not dataset depth).  The caller should use this
    instead of the original image so that RGB and depth are pixel-aligned."""


def cache_key(images, frame_indices, dataset_depths, dataset_intrinsics,
              dataset_camera_poses, model_id: str) -> str:
    """SHA-256 over every input the model sees, plus the loaded checkpoint id."""
    h = hashlib.sha256()
    h.update(f"v{CACHE_VERSION}|{model_id}".encode())
    h.update(repr(list(frame_indices)).encode())
    for img in images:
        h.update(f"|{img.shape}{img.dtype}".encode())
        h.update(img.tobytes())          # tobytes() is C-order regardless of layout
    # Named, so a value moving between conditioning slots changes the key.
    for name, table in (("depths", dataset_depths),
                        ("intrinsics", dataset_intrinsics),
                        ("poses", dataset_camera_poses)):
        h.update(f"|{name}:".encode())
        if not table:
            continue
        for idx in sorted(table):
            arr = np.asarray(table[idx], dtype=np.float32)
            h.update(f"|{idx}{arr.shape}".encode())
            # NaN marks invalid GT-depth pixels and never equals itself, so it is
            # replaced by a sentinel rather than hashed raw.
            h.update(np.nan_to_num(arr, nan=-12345.0).tobytes())
    return h.hexdigest()


def masking_key_suffix(mask_edges: bool, confidence_min: Optional[float]) -> str:
    """The output-filtering options as a key fragment, EMPTY when they are at defaults.

    map-anything filters its own output (`recon_mask_edges`, `recon_confidence_min`)
    and must key on it: a filtered result must not be served to an unfiltered request.
    The empty-when-default rule keeps a default-filtered entry's key independent of
    which off-by-default filtering options exist.
    """
    if mask_edges and confidence_min is None:
        return ""
    return f"|edges={mask_edges}|confmin={confidence_min}"


_FIELDS = ("depth_map_z", "K_matrix", "valid_mask", "c2w", "cropped_image")


def _flatten(results: Dict[int, ReconResult]) -> Dict[str, np.ndarray]:
    """`{idx: ReconResult}` as one flat npz payload."""
    flat: Dict[str, np.ndarray] = {"__frames__": np.array(sorted(results), dtype=np.int64)}
    for idx, res in results.items():
        for name in _FIELDS:
            arr = getattr(res, name)
            if arr is not None:   # cropped_image is None when depth came from the dataset
                flat[f"{idx}|{name}"] = arr
    return flat


def _load_results(path, label: str) -> Optional[Dict[int, ReconResult]]:
    """The cached results, or None on a miss -- a corrupt or partial file is a miss
    too (recomputed and overwritten), never an exception mid-run."""
    from .timing import get_timer

    if not path.is_file():
        return None
    with get_timer().exclude():
        try:
            with np.load(path) as data:
                out = {}
                for idx in data["__frames__"].tolist():
                    fields = {name: data[f"{idx}|{name}"] for name in _FIELDS
                              if f"{idx}|{name}" in data}
                    out[int(idx)] = ReconResult(**fields)
            print(f"  {label}: loaded cached depth from {path}")
            return out
        except Exception as exc:   # noqa: BLE001 — any load failure is just a miss
            print(f"  {label}: ignoring unreadable cache ({exc}): {path}")
            return None


def run_cached(
    run_fn: Callable[[List[np.ndarray], List[int]], Dict[int, ReconResult]],
    images: List[np.ndarray],
    frame_indices: List[int],
    dataset_depths: Optional[Dict[int, np.ndarray]] = None,
    dataset_intrinsics: Optional[Dict[int, np.ndarray]] = None,
    dataset_camera_poses: Optional[Dict[int, np.ndarray]] = None,
    cache_dir: Optional[str] = None,
    *,
    model_id: str,
    label: str,
    tag: str,
) -> Dict[int, ReconResult]:
    """``run_fn``, reusing a previous result for identical inputs.

    ``cache_dir=None`` (the default) delegates straight through, so this is a no-op
    unless a caller opts in.

    ``run_fn`` is called as ``run_fn(images, frame_indices)``; a backend that consumes
    conditioning binds it beforehand (``functools.partial``). The three conditioning
    tables reach this function for the KEY alone, so a backend that ignores them does
    not have to declare them.

    The key hashes the images' bytes, the conditioning arrays and the model id --
    everything the model sees. Hashing content rather than paths/mtimes costs
    milliseconds against a multi-second forward pass and cannot be fooled by a scene
    dir rewritten in place.
    """
    from pathlib import Path

    from . import disk_cache
    from .timing import get_timer, measure_core_seconds

    if not cache_dir:
        return run_fn(images, frame_indices)

    key = cache_key(images, frame_indices, dataset_depths, dataset_intrinsics,
                    dataset_camera_poses, model_id)
    path = Path(cache_dir) / f"{key}.npz"
    cached = _load_results(path, label)
    if cached is not None:
        # Charge the hit what the miss cost, so a cached run's PREPROCESSING seconds
        # stay comparable with a cold one's instead of reporting ~0s for the depth pass.
        disk_cache.credit_cached_work(path, log_prefix=f"  [{tag}-cache]")
        return cached

    with measure_core_seconds() as span:
        results = run_fn(images, frame_indices)
    try:
        with get_timer().exclude():
            flat = _flatten(results)
        # Compressed: depth float32 and the bool mask pack several times smaller, for a
        # small cost on a write that only follows a (much slower) miss. `np.load` reads
        # compressed and uncompressed entries alike.
        disk_cache.atomic_write(
            path, lambda tmp: np.savez_compressed(tmp, **flat), tmp_suffix=".npz",
            meta={disk_cache.GENERATION_SECONDS: span.seconds,
                  "frames": len(frame_indices)},
            log=f"  {label}: cached depth",
        )
    except OSError as exc:
        # A read-only scene dir (a shipped example on a share) must cost the cache,
        # not the run -- we already have the results.
        print(f"  {label}: could not write cache ({exc})")
    return results


__all__ = ["ReconResult", "CACHE_VERSION", "cache_key", "run_cached"]


#: Datasets whose true intrinsics ``Sequence`` feeds map-anything as conditioning.
GT_K_DATASETS = ("gso", "co3d", "oursactionbench")
