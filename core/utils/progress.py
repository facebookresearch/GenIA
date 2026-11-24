# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Progress of a block's iterative work, as lines a host application can read.

``tqdm`` here is tqdm, plus: with ``GENIA_PROGRESS=1`` in the environment (the demo app sets
it), every loop it wraps also prints machine-readable lines to stdout::

    @@progress {"task": "Stage 2 (2 frames)", "n": 12, "total": 25}

at most twice a second, and once more when the loop ends; a loop that ends short of its
total (stopped early, e.g. converged) says so with ``"stopped": true``. The app shows them as
progress bars. Without the variable nothing extra is printed, so a command-line run looks as before.

A loop working on one object is that object's: a per-object loop calls
``set_object(obj_idx)`` first, and every loop started until the next call is described as
``Object {obj_idx} · {description}``, on the console and in its lines, so the app shows one
bar per object.

A host running the pipeline in its own process (the demo app on ZeroGPU), where a run
cannot be killed, stops it with ``request_stop()``: the next step of any loop here raises
``Stopped``, which ends the run (``clear_stop()`` before the next one).
"""
import json
import os
import threading
import time

from tqdm import tqdm as _tqdm

PREFIX = "@@progress "
#: Seconds between two lines of one loop (the last step is always reported).
INTERVAL = 0.5

_object = None   # the object the loops started now work on (``set_object``)
_stop = threading.Event()   # set by ``request_stop``


class Stopped(Exception):
    """Raised by a loop's next step once ``request_stop`` was called."""


def request_stop() -> None:
    """End the running pipeline at the next step of any of its loops."""
    _stop.set()


def clear_stop() -> None:
    _stop.clear()


def set_object(obj_idx) -> None:
    """Report the loops started from now on as ``obj_idx``'s (None: as no object's)."""
    global _object
    _object = obj_idx


class tqdm(_tqdm):  # noqa: N801 — a drop-in for tqdm's own name
    def __init__(self, *args, **kwargs):
        if _object is not None and kwargs.get("desc"):
            desc = kwargs["desc"]
            indent = desc[:len(desc) - len(desc.lstrip())]
            kwargs["desc"] = f"{indent}Object {_object} · {desc.lstrip()}"
        super().__init__(*args, **kwargs)
        self._report_on = os.environ.get("GENIA_PROGRESS") == "1" and not self.disable
        self._reported_at, self._reported_n = 0.0, None

    def _emit(self, n, **extra):
        task = (self.desc or "steps").strip().rstrip(":")
        print(PREFIX + json.dumps({"task": task, "n": n, "total": self.total, **extra}),
              flush=True)

    def _report(self, n, force=False):
        now = time.monotonic()
        if (not self._report_on or n == self._reported_n
                or (not force and now - self._reported_at < INTERVAL)):
            return
        self._reported_at, self._reported_n = now, n
        self._emit(n)

    def __iter__(self):
        # tqdm only syncs ``self.n`` at its display intervals while iterating (through
        # ``update``, hence the repeat check in ``_report``), so count here.
        n = self.n
        for item in super().__iter__():
            yield item
            n += 1
            self._report(n, force=n == self.total)
            if _stop.is_set():
                raise Stopped

    def close(self):
        if getattr(self, "_report_on", False) and self.total and self.n < self.total:
            self._report_on = False   # once: tqdm may close again (e.g. on deletion)
            self._emit(self.n, stopped=True)
        super().close()

    def update(self, n=1):
        displayed = super().update(n)
        self._report(self.n, force=self.n == self.total)
        if _stop.is_set():
            raise Stopped
        return displayed
