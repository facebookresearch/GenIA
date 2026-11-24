# Copyright (c) Meta Platforms, Inc. and affiliates.

"""A warm method runner: one child process, many runs.

The app used to start ``run_demo.py`` per click, so every click paid for the process
again: importing torch and the method, CUDA's start-up, and the models' loads (SAM 3D's
backbone alone ~18 s, map-anything more) — about 40 s before the first block ran. This
keeps one child alive across clicks instead. As soon as it starts it loads what
``method.warmup`` names (before any click), then runs the method's Hydra entry once per
request, exactly as ``python <entry> <overrides>`` would; the modules it imports and the
models they cache stay loaded for the next run.

It is a separate process, not a thread of the app, so Stop still works (a process group
takes a signal; a CUDA kernel in a thread cannot be interrupted), an out-of-memory kills
the worker and not the app, and the run's output is a real pipe.

Adapted from the GenerativeReconstruction framework's demo (``gradio/_gradio_worker.py``),
whose protocol it keeps, plus the warm-up request. The child side is ``_serve``, reached
by running this file; the parent side is ``Worker``.
"""
from __future__ import annotations

import codecs
import json
import os
import queue
import signal
import subprocess
import sys
import threading
from pathlib import Path

#: ``demo/``: where ``method.py`` is, the module the child warms up through.
DEMO = Path(__file__).resolve().parents[1]

# Framing, each printed by the child on its own line and never shown to a visitor.
READY = "__DEMO_WORKER_READY__"     # warm-up done: what follows belongs to a run
RUN_END = "__DEMO_RUN_END__"        # a run ended; its exit code follows on the line


# ======================================================================
# Child side
# ======================================================================

def _serve(entry: str) -> None:
    """Serve requests from stdin, one JSON object per line.

    ``{"warmup": [overrides]}`` calls ``method.warmup(overrides)`` (a method without one
    has nothing to load ahead), then prints ``READY``. ``{"overrides": [...]}`` runs
    ``entry`` as ``python <entry> <overrides>`` (``-m <package>`` or a script path)
    through ``runpy``, so each run is the command-line run itself, then prints
    ``RUN_END <exit code>``.
    """
    import runpy
    import traceback

    sys.path.insert(0, str(DEMO))
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        if "warmup" in request:
            import method

            try:
                if hasattr(method, "warmup"):
                    method.warmup(request["warmup"])
            except Exception:   # noqa: BLE001 — a cold start still runs; the log says why
                traceback.print_exc()
            print(READY, flush=True)
            continue
        from hydra.core.global_hydra import GlobalHydra

        rc = 0
        try:
            GlobalHydra.instance().clear()   # one instance per process; hydra.main makes it
            sys.argv = [entry, *request["overrides"]]
            if entry.startswith("-m "):
                runpy.run_module(entry[3:].strip(), run_name="__main__", alter_sys=True)
            else:
                runpy.run_path(entry, run_name="__main__")
        except SystemExit as exc:            # hydra.main exits 1 on a failed run
            rc = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        except Exception:                    # noqa: BLE001 — the traceback is the report
            traceback.print_exc()
            rc = 1
        finally:
            sys.stderr.flush()
            _release()
        print(f"{RUN_END} {rc}", flush=True)


def _release() -> None:
    """Hand back what the run held, keeping the cached models."""
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:   # noqa: BLE001 — a worker that cannot free still serves
        pass


# ======================================================================
# Parent side
# ======================================================================

class Worker:
    """The method's runner in a child process, reused across runs; one run at a time.

    ``start`` spawns it (if it is not running) and sends the warm-up, returning at once:
    the child loads while the visitor is still choosing a scene. ``run`` streams one
    run's output. A Stop or a failed run ends the process (a failed CUDA run can leave
    its context unusable); the caller ``start``s a fresh one, which warms up again.
    """

    def __init__(self, entry: str, env: dict, warm: list[str] = (), cwd=None,
                 python: str = sys.executable) -> None:
        self.entry, self.env, self.warm, self.cwd, self.python = entry, env, list(warm), cwd, python
        self.returncode: int | None = None
        self._proc: subprocess.Popen | None = None
        self._reader: Reader | None = None
        self._ready = False

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> None:
        """Spawn the worker and send its warm-up, unless one is already running."""
        if self.alive:
            return
        # Its own session, so ``stop`` reaches everything the run started.
        self._proc = subprocess.Popen(
            [self.python, "-u", str(Path(__file__).resolve()), self.entry], cwd=self.cwd,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env={**self.env, "PYTHONUNBUFFERED": "1"}, start_new_session=True)
        self._reader, self._ready = Reader(self._proc.stdout), False
        self._send({"warmup": self.warm})

    def stop(self) -> bool:
        """Kill the worker's process group. True if one was running."""
        proc, self._proc, self._reader = self._proc, None, None
        if proc is None or proc.poll() is not None:
            return False
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        return True

    def _send(self, request: dict) -> None:
        self._proc.stdin.write((json.dumps(request) + "\n").encode())
        self._proc.stdin.flush()

    def run(self, overrides):
        """Run once, yielding the run's output as it arrives; ``returncode`` is set at the
        end (the process's own exit status if it died, e.g. ``-SIGKILL`` after ``stop``).
        A worker still warming up streams its warm-up first (it is the wait)."""
        self.returncode = None
        self.start()
        proc, reader = self._proc, self._reader
        try:
            self._send({"overrides": list(overrides)})
        except (BrokenPipeError, OSError):
            self.stop()
            yield "The demo's worker exited before the run started.\n"
            self.returncode = 1
            return
        pending = ""
        while True:
            text = pending + reader.drain(timeout=0.5)
            pending = ""
            if not self._ready and READY in text:   # warm-up output is the app's, not the run's
                self._ready = True
                text = text.split(READY, 1)[1].lstrip("\n")
            if RUN_END in text:
                body, _, tail = text.partition(RUN_END)
                while "\n" not in tail and proc.poll() is None:
                    tail += reader.drain(timeout=0.5)
                if body:
                    yield body
                self.returncode = _rc_of(tail)
                if self.returncode:
                    self.stop()
                return
            # Hold back a trailing partial line that may be the start of a marker.
            head, sep, last = text.rpartition("\n")
            if last and (RUN_END.startswith(last) or READY.startswith(last)):
                text, pending = head + sep, last
            if text:
                yield text
            if proc.poll() is not None and not reader.alive:
                self.returncode = proc.returncode or 1
                if self._proc is proc:
                    self.stop()
                return


class Reader:
    """Raw reads of a pipe on a thread, so tqdm's ``\\r`` updates arrive as they happen."""

    def __init__(self, pipe) -> None:
        self._pipe, self._q = pipe, queue.Queue()
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        fd = self._pipe.fileno()
        decode = codecs.getincrementaldecoder("utf-8")("replace").decode   # chars split by reads
        while True:
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                break
            if not chunk:
                break
            self._q.put(decode(chunk))

    @property
    def alive(self) -> bool:
        return self._thread.is_alive() or not self._q.empty()

    def drain(self, timeout: float = 0.0) -> str:
        """Everything buffered, waiting up to ``timeout`` for the first chunk."""
        out = ""
        try:
            out = self._q.get(timeout=timeout) if timeout else ""
        except queue.Empty:
            return out
        while not self._q.empty():
            out += self._q.get_nowait()
        return out


def _rc_of(tail: str) -> int:
    """The exit code after a ``RUN_END`` marker; 1 if missing or unreadable."""
    try:
        return int(tail.split()[0])
    except (IndexError, ValueError):
        return 1


if __name__ == "__main__":
    _serve(sys.argv[1])
