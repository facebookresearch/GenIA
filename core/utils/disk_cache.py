# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Shared disk-cache I/O primitives for the content-keyed ``cached_shapes/``
caches (``sam3d_shape_cache.py`` shape tokens).

Only the I/O policy lives here — atomic writes that are safe under concurrent
jobs on the same scene, the advisory ``.json`` provenance sidecar,
and "a corrupt/partial file degrades to a miss". Each cache keeps its own key /
path / schema. All of it is timer-excluded: cache I/O is not pipeline compute.
"""
import json
import os
import tempfile

import torch

from genia.core.utils.timing import get_timer


def atomic_write(path, write_fn, *, tmp_suffix, meta=None, log=None) -> None:
    """Write ``path`` via ``write_fn(tmp_path)`` atomically (tmp + ``os.replace``),
    plus a ``<name>.json`` sidecar when ``meta`` is given. The sidecar is
    advisory — never read back by a cache lookup — so it gets a plain write, no
    atomicity ceremony. ``path=None`` (caching off) is a no-op."""
    if path is None:
        return
    with get_timer().exclude():
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=tmp_suffix)
        os.close(fd)
        try:
            write_fn(tmp)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        if meta is not None:
            path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        if log:
            print(f"{log} -> {path}")


#: Sidecar field holding how long the cached artefact took to compute, and the
#: block that paid for it on the run that filled the cache.
GENERATION_SECONDS = "generation_seconds"

#: Sidecar field holding the peak CUDA allocation the filling run reached. Only a
#: producer that ran in a CHILD process needs it -- an in-process one is measured by
#: the block itself -- but a hit reads it back either way, so a cached child-produced
#: artefact still reports the producer's footprint rather than the load's. When the
#: field is absent the hit credits no memory, which leaves the block reporting its own
#: (honest, if incomplete) measurement.
PEAK_ALLOC_MB = "peak_alloc_mb"

#: Charged for a hit on an entry that does NOT record its generation time (e.g. a
#: lost or corrupt sidecar). Absurd on purpose: the run
#: is reporting a number nobody measured, so it should be impossible to miss in a
#: timing table or a cost plot rather than plausible enough to publish. Regenerate
#: the entry cold to replace it with the real cost.
UNKNOWN_GENERATION_SECONDS = 10000.0


def credit_cached_work(path, *, block=None, log_prefix="  [cache]") -> float:
    """Charge a cache HIT the compute time the filling run recorded, and return it.

    A cache hit costs a fraction of a second, so without this the run reports a
    near-zero cost for work it genuinely needed -- and nothing in ``timing.json``
    marks the hit, so the number reads as a real measurement. The seconds
    come from the ``.json`` sidecar :func:`atomic_write` wrote beside the artefact.

    An entry that records NO generation time is charged
    :data:`UNKNOWN_GENERATION_SECONDS` instead of nothing: crediting zero would
    reproduce the exact failure this exists to prevent -- a silently too-fast
    number -- whereas the sentinel is impossible to miss and says "regenerate me".

    Credited to ``block``, or to whichever block is running (so the same call works
    from any block).
    """
    if path is None:
        return 0.0
    with get_timer().exclude():
        timer = get_timer()
        name = block or timer.active_block
        if name is None:
            return 0.0
        meta = read_meta(path)
        # Memory first: it is credited whether or not the seconds are recoverable, and
        # a max is harmless when the block already measured a larger one itself.
        peak_mb = float((meta or {}).get(PEAK_ALLOC_MB) or 0.0)
        if peak_mb > 0.0:
            timer.credit_peak_alloc(name, peak_mb * 1024**2)
        seconds = float((meta or {}).get(GENERATION_SECONDS) or 0.0)
        if seconds > 0.0:
            timer.credit_seconds(name, seconds)
            print(f"{log_prefix} hit: crediting {seconds:.1f}s of cached compute "
                  f"to {name}")
        else:
            timer.credit_seconds(name, UNKNOWN_GENERATION_SECONDS)
            print(f"{log_prefix} !! hit on an entry with no recorded generation time; "
                  f"charging {name} the {UNKNOWN_GENERATION_SECONDS:.0f}s sentinel -- "
                  f"this run's timing is NOT a measurement. Regenerate cold: {path}")
            return UNKNOWN_GENERATION_SECONDS
        return seconds


def read_meta(path):
    """The ``.json`` provenance sidecar beside ``path``, or None when absent or
    unreadable -- the sidecar is advisory, so a bad one never breaks a run."""
    if path is None:
        return None
    side = path.with_suffix(".json")
    try:
        return json.loads(side.read_text())
    except (OSError, ValueError):
        return None


def store_blob(path, blob, *, meta=None, log=None) -> None:
    """:func:`atomic_write` for a torch blob (``torch.save``)."""
    atomic_write(path, lambda tmp: torch.save(blob, tmp), tmp_suffix=".pt.tmp",
                 meta=meta, log=log)


def load_blob(path, device, *, log_prefix):
    """The cached torch blob mapped to ``device``, or ``None`` on miss — a
    corrupt/partial file is a miss too (recomputed + overwritten), never a
    raised exception mid-run."""
    if path is None or not path.is_file():
        return None
    with get_timer().exclude():
        try:
            return torch.load(path, map_location=device, weights_only=True)
        except Exception as e:  # noqa: BLE001 — any load failure is just a miss
            print(f"{log_prefix} read failed ({e}) — recomputing: {path}")
            return None
