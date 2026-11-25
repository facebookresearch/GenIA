# Copyright (c) Meta Platforms, Inc. and affiliates.

"""The method run in the app's own process: the runner for ZeroGPU.

On a ZeroGPU Space the GPU exists only inside a function decorated with ``spaces.GPU``,
in the app's process; a worker process of its own (``ui/worker.py``) would never get one.
So here the models load into the app's process as it starts (``method.warmup``: on
ZeroGPU they are placed on the GPU at import, as it asks), and each run is one call of the
decorated ``_gpu_run``, for the GPU time ``method.gpu_seconds`` asks for. Anything else, a
recorded example's replay included, never requests a GPU.

``InProcess`` has ``Worker``'s interface (``warm``, ``start``, ``run``, ``returncode``,
``stop``), so the app follows either the same way. Outside ZeroGPU (no ``spaces`` package)
the decorator does nothing and the run simply happens in the app's process. A run here
cannot be killed: ``stop`` asks the method to end it (``method.stop_in_process``), which
GenIA does at the next step of any of its loops.
"""
from __future__ import annotations

import contextlib
import gc
import queue
import sys
import threading
import traceback

import method


def gpu(duration):
    """``spaces.GPU(duration=duration)`` on ZeroGPU: the decorated function gets a GPU for
    ``duration`` seconds (a number, or a callable of its arguments). Elsewhere, nothing.

    ZeroGPU is told by ``spaces.GPU`` existing, not by the import working: a ``spaces/`` folder
    on the path (a checkout's Space files) imports as an empty namespace package."""
    try:
        import spaces
    except ImportError:   # not on ZeroGPU
        return lambda fn: fn
    if not hasattr(spaces, "GPU"):   # someone's spaces/ folder, not Hugging Face's package
        return lambda fn: fn
    return spaces.GPU(duration=duration)


class _Pipe:
    """A text stream whose writes go to a queue (the run's captured output)."""

    def __init__(self, q: queue.Queue) -> None:
        self.q = q

    def write(self, text: str) -> int:
        if text:
            self.q.put(text)
        return len(text)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


@gpu(method.gpu_seconds)   # called with the run's overrides
def _gpu_run(overrides):
    """One run of the method, on a thread, yielding its output as it is printed, then
    ``{"returncode": ...}``. A generator, so the output streams out of the GPU call."""
    out: queue.Queue = queue.Queue()
    result = {}

    def target():
        try:
            with contextlib.redirect_stdout(_Pipe(out)), contextlib.redirect_stderr(_Pipe(out)):
                result["returncode"] = method.run_in_process(overrides)
        except BaseException:   # noqa: BLE001 — the traceback is the run's report
            out.put(traceback.format_exc())
            result["returncode"] = 1
        finally:
            _release()
            out.put(None)

    threading.Thread(target=target, daemon=True).start()
    while (chunk := out.get()) is not None:
        yield chunk
    yield {"returncode": result.get("returncode", 1)}


def _release() -> None:
    """Hand back what the run held, keeping the loaded models."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


class InProcess:
    """``Worker``'s interface over runs in this process (see the module docstring)."""

    def __init__(self, warm: list[str] = ()) -> None:
        self.warm = list(warm)
        self.returncode: int | None = None
        self._loaded = False

    def start(self) -> None:
        """Load what the method keeps loaded (once; blocking: on ZeroGPU at start-up)."""
        if not self._loaded:
            self._loaded = True
            method.warmup(self.warm)

    def stop(self) -> None:
        method.stop_in_process()

    def run(self, overrides):
        """Run once, yielding the output; ``returncode`` is set at the end."""
        self.returncode = None
        self.start()
        for chunk in _gpu_run(list(overrides)):
            if isinstance(chunk, dict):
                self.returncode = chunk["returncode"]
            else:
                yield chunk
