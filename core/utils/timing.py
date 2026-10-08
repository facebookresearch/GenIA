# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Pipeline timing + peak-memory instrumentation.

Provides ``PipelineTimer`` for measuring per-block core computation time
(excluding visualization, evaluation, and I/O overhead) plus per-block peak
CUDA memory (allocated + reserved), and ``plot_pipeline_timing`` for generating
a summary bar chart + JSON.  The peak-memory readout is what surfaces the effect
of memory knobs like ``rendering_guidance_decoder_autocast_bf16``.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

from .visualization import save_figure


@dataclass
class BlockTiming:
    """Timing + peak-CUDA-memory data for a single pipeline block."""

    total_seconds: float = 0.0
    excluded_seconds: float = 0.0
    # Work this block needed but did not execute, timed by whoever did (a cached
    # external stage).  Kept apart from total_seconds, which end_block overwrites.
    credited_seconds: float = 0.0
    status: str = "ran"  # "ran" | "skipped" | "cached"
    # Peak CUDA memory during the block (0 if no GPU): allocated = live tensors
    # (what bf16 shrinks), reserved = allocator high-water.  Reset at start_block,
    # so each is the block's own footprint (resident state + transient allocs).
    peak_alloc_bytes: float = 0.0
    peak_reserved_bytes: float = 0.0
    # Peak reported by a CHILD PROCESS, which this process's counter cannot see.
    # Kept apart from peak_alloc_bytes, which end_block overwrites, for the same
    # reason credited_seconds is kept apart from total_seconds.
    external_peak_alloc_bytes: float = 0.0
    # CUDA memory ALREADY allocated when the block opened. Not a peak -- the level.
    # reset_peak_memory_stats() rebases the peak to whatever is live at that moment, so
    # peak_alloc_bytes is "this block's own allocations ON TOP OF everything still
    # resident". Recording the floor makes that inherited part visible instead of
    # silent; nothing subtracts it (see PipelineTimer.credit_peak_alloc's note on what
    # "required memory" has to mean).
    baseline_alloc_bytes: float = 0.0
    # This block's work ran in a child process whose peak cannot be recovered, so
    # the figure this process measured is not the block's. Reported as null rather
    # than as a number nobody can act on.
    peak_unmeasured: bool = False

    @property
    def core_seconds(self) -> float:
        return max(0.0, self.total_seconds - self.excluded_seconds) + self.credited_seconds

    @property
    def peak_alloc(self) -> float:
        """The block's peak allocation: this process's, or a child's if that is larger.

        A max, never a sum -- the two processes' peaks are points in time, and the
        block's peak is whichever is higher, not their total."""
        return max(self.peak_alloc_bytes, self.external_peak_alloc_bytes)


class PipelineTimer:
    """Accumulates per-block wall-clock timing with exclude regions.

    Usage::

        timer = PipelineTimer()

        timer.start_block("appearance_init")
        # ... core computation ...
        with timer.exclude():
            save_keyframes_video(...)  # not counted
        # ... more core computation ...
        timer.end_block("appearance_init")

    ``torch.cuda.synchronize()`` is called at every timing boundary so
    that GPU-bound work is captured accurately.
    """

    def __init__(self) -> None:
        self._blocks: dict[str, BlockTiming] = {}
        self._active_block: Optional[str] = None
        self._block_start: float = 0.0
        self._exclude_depth: int = 0

    # ------------------------------------------------------------------
    # Block-level API (called from the main dispatch loop)
    # ------------------------------------------------------------------

    def start_block(self, name: str) -> None:
        """Begin timing a block."""
        _sync_cuda()
        self._active_block = name
        if name not in self._blocks:
            self._blocks[name] = BlockTiming()
        self._blocks[name].status = "ran"
        self._blocks[name].baseline_alloc_bytes = _allocated_memory()
        _reset_peak_memory()
        self._block_start = time.perf_counter()

    def end_block(self, name: str) -> None:
        """End timing a block."""
        _sync_cuda()
        bt = self._blocks[name]
        bt.total_seconds = time.perf_counter() - self._block_start
        bt.peak_alloc_bytes, bt.peak_reserved_bytes = _peak_memory()
        self._active_block = None

    def mark_cached(self, name: str) -> None:
        """Record that a block was loaded from cache (not executed)."""
        if name not in self._blocks:
            self._blocks[name] = BlockTiming()
        self._blocks[name].status = "cached"

    @property
    def active_block(self) -> Optional[str]:
        """The block being timed right now, or None outside any block.

        Lets code deep inside a block credit work to it without being told which
        block it is in, so the same helper serves every block.
        """
        return self._active_block

    def credit_seconds(self, name: str, seconds: float) -> None:
        """Add externally-measured work to a block's core time.

        For a stage this run legitimately needed but did not execute, because an
        earlier run cached its result and recorded how long it took.  Without this
        the cache would quietly make every run after the first look faster.  Safe to
        call from inside
        the block — it lands in its own field, which ``end_block`` does not overwrite.
        """
        if name not in self._blocks:
            self._blocks[name] = BlockTiming()
        self._blocks[name].credited_seconds += max(0.0, float(seconds))

    def credit_peak_alloc(self, name: str, peak_bytes: float) -> None:
        """Record a peak allocation this process could not observe.

        ``torch.cuda.max_memory_allocated`` is per PROCESS, so work run in a child
        process (or skipped on a cache hit) is invisible to the parent's counter,
        which then measures only its own footprint.

        Kept, like :meth:`credit_seconds`, in a field ``end_block`` does not overwrite:
        it re-reads the process counter and would discard anything written to
        ``peak_alloc_bytes`` from inside the block.  Takes the MAX rather than
        accumulating -- see :attr:`BlockTiming.peak_alloc`.
        """
        if name not in self._blocks:
            self._blocks[name] = BlockTiming()
        bt = self._blocks[name]
        bt.external_peak_alloc_bytes = max(bt.external_peak_alloc_bytes,
                                           max(0.0, float(peak_bytes)))

    # ------------------------------------------------------------------
    # Exclude context manager (called inside block functions)
    # ------------------------------------------------------------------

    @contextmanager
    def exclude(self):
        """Context manager: time spent inside is subtracted from core time.

        Safe to call when no block is active (becomes a no-op) and safe to
        nest (only the outermost layer measures).
        """
        if self._active_block is None:
            yield
            return
        self._exclude_depth += 1
        if self._exclude_depth == 1:
            _sync_cuda()
            t0 = time.perf_counter()
        try:
            yield
        finally:
            if self._exclude_depth == 1:
                _sync_cuda()
                dt = time.perf_counter() - t0
                block = self._blocks.get(self._active_block)
                if block is not None:
                    block.excluded_seconds += dt
            self._exclude_depth -= 1

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def blocks(self) -> dict[str, BlockTiming]:
        return dict(self._blocks)

    @property
    def total_core_seconds(self) -> float:
        return sum(
            b.core_seconds for b in self._blocks.values() if b.status == "ran"
        )

    @property
    def peak_alloc_bytes(self) -> float:
        """Pipeline-wide peak allocated CUDA memory (max over ran blocks).

        ``peak_alloc`` per block, not ``peak_alloc_bytes``: work this process did not
        run itself reports its footprint through :meth:`credit_peak_alloc`, and the
        pipeline peak has to see it."""
        ran = [b.peak_alloc for b in self._blocks.values()
               if b.status == "ran" and not b.peak_unmeasured]
        return max(ran) if ran else 0.0

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict."""
        # null, not a number, when any block that RAN could not be measured: the
        # pipeline peak is then a lower bound on this process alone rather than the run's
        # footprint, and a consumer cannot tell the two apart from a float.
        unmeasured = any(b.peak_unmeasured for b in self._blocks.values()
                         if b.status == "ran")
        result: dict = {
            "blocks": {},
            "total_core_seconds": round(self.total_core_seconds, 3),
            "peak_alloc_mb": (None if unmeasured
                              else round(self.peak_alloc_bytes / 1024**2, 1)),
        }
        for name, bt in self._blocks.items():
            result["blocks"][name] = {
                "status": bt.status,
                "total_seconds": round(bt.total_seconds, 3),
                "excluded_seconds": round(bt.excluded_seconds, 3),
                "credited_seconds": round(bt.credited_seconds, 3),
                "core_seconds": round(bt.core_seconds, 3),
                "peak_alloc_mb": (None if bt.peak_unmeasured
                                  else round(bt.peak_alloc / 1024**2, 1)),
                "baseline_alloc_mb": round(bt.baseline_alloc_bytes / 1024**2, 1),
                "peak_reserved_mb": round(bt.peak_reserved_bytes / 1024**2, 1),
            }
        return result

    def reset(self) -> None:
        """Clear all accumulated data (for module-level reuse)."""
        self._blocks.clear()
        self._active_block = None
        self._block_start = 0.0
        self._exclude_depth = 0


# ------------------------------------------------------------------
# Process-wide shared instance
# ------------------------------------------------------------------
# Single timer shared by the main dispatch loop (which owns block
# start/end) and by helpers that run inside a block, so they can
# ``get_timer().exclude()`` without importing ``genia.core.main`` (circular).
_TIMER: PipelineTimer = PipelineTimer()


def get_timer() -> PipelineTimer:
    """Return the process-wide :class:`PipelineTimer` singleton."""
    return _TIMER


@dataclass
class _CoreSpan:
    """Handle yielded by :func:`measure_core_seconds`; ``seconds`` is set on exit."""

    seconds: float = 0.0


@contextmanager
def measure_core_seconds():
    """Measure a region the way ``core_seconds`` measures a block: minus exclusions.

    For recording how long a cacheable stage took, so a later run that hits the cache
    can be credited it (:func:`~genia.core.utils.disk_cache.credit_cached_work`). Raw
    wall-clock is the wrong number here: the region may contain ``exclude()``d viz or a
    model load, which the live run does not charge to the block -- crediting the raw
    span would make a cache hit cost *more* than running the stage.
    """
    timer = get_timer()
    block = timer.blocks.get(timer.active_block)
    excluded_before = block.excluded_seconds if block is not None else 0.0
    span = _CoreSpan()
    t0 = time.perf_counter()
    try:
        yield span
    finally:
        # Re-read: end_block never runs mid-region, but exclude() mutates in place.
        block = timer.blocks.get(timer.active_block)
        excluded = (block.excluded_seconds - excluded_before) if block is not None else 0.0
        span.seconds = max(0.0, time.perf_counter() - t0 - excluded)


@contextmanager
def exclude_model_load(what: str):
    """Keep a one-off model-weight load out of the enclosing block's core time.

    Reading a multi-GB checkpoint is process startup, amortised over everything the run
    then does -- not the per-object compute a block is meant to measure. ``core/run.py``
    builds the backbone *before* the timer starts, so this is what puts lazily-loaded
    backends on that same footing.

    Prints the cost either way -- excluded, but never invisible. Safe outside a block
    (:meth:`PipelineTimer.exclude` is then a no-op), which the lazy getters need; the
    log line says so rather than claiming an exclusion that did not happen.
    """
    timer = get_timer()
    active = timer.active_block
    t0 = time.perf_counter()
    with timer.exclude():
        yield
    where = f"excluded from {active}" if active else "no block active, nothing to exclude"
    print(f"  [startup] {what}: {time.perf_counter() - t0:.1f}s model load ({where})")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _cuda():
    """Return ``torch.cuda`` when a GPU is usable, else None (CPU / no torch)."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda
    except ImportError:
        pass
    return None


def _sync_cuda() -> None:
    """Synchronize CUDA (if available) so GPU work is captured by wall-clock."""
    if (c := _cuda()) is not None:
        c.synchronize()


def _reset_peak_memory() -> None:
    """Reset the CUDA peak-memory counters at a block boundary.

    NOT enough on its own to make the next block's peak "its own": the reset rebases the
    peak to whatever is currently ALLOCATED, so a block starts at the level everything
    still resident occupies and measures its own work on top. Time has no equivalent --
    which is why excluding a block works for seconds and quietly fails for memory.
    ``start_block`` records that floor as ``baseline_alloc_bytes``, and the pipeline
    frees the reconstruction backend once the Sequence is built so preprocessing's
    weights are not part of it."""
    if (c := _cuda()) is not None:
        c.reset_peak_memory_stats()


def _allocated_memory() -> float:
    """CUDA bytes allocated RIGHT NOW (0 if no GPU) -- the level, not the peak."""
    return float(c.memory_allocated()) if (c := _cuda()) is not None else 0.0


def _peak_memory() -> tuple[float, float]:
    """Peak (allocated, reserved) CUDA bytes since the last reset; (0, 0) if no GPU."""
    if (c := _cuda()) is None:
        return 0.0, 0.0
    return float(c.max_memory_allocated()), float(c.max_memory_reserved())


@contextmanager
def memory_snapshot(snapshot_path: str, max_entries: int = 200_000):
    """Record the full CUDA allocation history inside the block and dump it.

    Debugging tool for "what forms the peak?" questions the scalar per-block
    peaks can't answer: wraps ``torch.cuda.memory._record_memory_history()``
    (every alloc/free with its Python stack) and writes a pickle viewable at
    https://pytorch.org/memory_viz on exit.  Driven by ``GENIA_MEM_SNAPSHOT``
    (see ``core/run.py``); no-op without a GPU.  Recording adds noticeable
    CPU overhead — debug runs only.
    """
    if _cuda() is None:
        yield
        return
    import torch
    torch.cuda.memory._record_memory_history(max_entries=max_entries)
    try:
        yield
    finally:
        os.makedirs(os.path.dirname(snapshot_path) or ".", exist_ok=True)
        torch.cuda.memory._dump_snapshot(snapshot_path)
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f"  [mem-snapshot] wrote {snapshot_path}")


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def plot_pipeline_timing(
    timer: PipelineTimer,
    output_dir: str,
    scene_name: str,
) -> None:
    """Generate per-block horizontal bar charts of core time + peak CUDA memory.

    Saves:
    - ``{output_dir}/{scene_name}_timing.png`` — two panels (core time | peak
      memory), one bar per block
    - ``{output_dir}/timing.json`` — full timing + memory data

    Only blocks with ``status == "ran"`` and ``core_seconds > 0.1`` appear
    on the chart.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # ── Save JSON ──
    timing_data = timer.to_dict()
    json_path = os.path.join(output_dir, "timing.json")
    with open(json_path, "w") as f:
        json.dump(timing_data, f, indent=2)
    print(f"  Saved timing data to {json_path}")

    # ── Filter to blocks that actually ran with meaningful time ──
    ran_blocks = [
        (name, bt)
        for name, bt in timer.blocks.items()
        if bt.status == "ran" and bt.core_seconds > 0.1
    ]
    if not ran_blocks:
        print("  No blocks with significant core time — skipping timing plot")
        return

    names = [name for name, _ in ran_blocks]
    core_times = [bt.core_seconds for _, bt in ran_blocks]
    excluded_times = [bt.excluded_seconds for _, bt in ran_blocks]
    peak_alloc_gb = [bt.peak_alloc_bytes / 1024**3 for _, bt in ran_blocks]
    peak_reserved_gb = [bt.peak_reserved_bytes / 1024**3 for _, bt in ran_blocks]
    y_pos = list(range(len(names)))

    # ── Two panels sharing block names: core time | peak CUDA memory ──
    fig, (ax_t, ax_m) = plt.subplots(
        1, 2, sharey=True, figsize=(13, max(3, 0.5 * len(names))),
        gridspec_kw={"width_ratios": [3, 2]},
    )

    # Left: core computation + excluded (viz/eval) time, stacked.
    ax_t.barh(y_pos, core_times, color="#2196F3", edgecolor="white",
              label="Core computation")
    ax_t.barh(y_pos, excluded_times, left=core_times, color="#E0E0E0",
              edgecolor="white", alpha=0.7, label="Excluded (viz/eval)")
    ax_t.set_yticks(y_pos)
    ax_t.set_yticklabels(names)
    ax_t.invert_yaxis()  # first block at top
    ax_t.set_xlabel("Time (seconds)")
    ax_t.legend(loc="lower right")
    for i, (ct, et) in enumerate(zip(core_times, excluded_times)):
        ax_t.text(ct + et + 0.5, i, f"{ct:.1f}s", va="center", fontsize=8)

    # Right: peak memory — reserved (allocator high-water) with allocated on top.
    ax_m.barh(y_pos, peak_reserved_gb, color="#E0E0E0", edgecolor="white",
              alpha=0.7, label="Reserved")
    ax_m.barh(y_pos, peak_alloc_gb, color="#4CAF50", edgecolor="white",
              label="Allocated")
    ax_m.set_xlabel("Peak CUDA memory (GB)")
    ax_m.legend(loc="lower right")
    for i, gb in enumerate(peak_alloc_gb):
        ax_m.text(max(peak_reserved_gb[i], gb) + 0.1, i, f"{gb:.1f}GB",
                  va="center", fontsize=8)

    fig.suptitle(
        f"Pipeline Timing & Memory — {scene_name}    "
        f"(total core {timer.total_core_seconds:.1f}s · "
        f"peak {timer.peak_alloc_bytes / 1024**3:.1f}GB)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    png_path = os.path.join(output_dir, f"{scene_name}_timing.png")
    save_figure(fig, png_path)
    print(f"  Saved timing chart to {png_path}")
