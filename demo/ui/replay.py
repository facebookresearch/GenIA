# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Recorded runs of the example scenes, played back instead of run.

An example scene always reconstructs the same way, so the app plays a recording of it at
once instead of making a visitor wait for the GPU ("Run live" still runs it). A recording
is what a run showed and when: its output as it arrived, every ``live_poses.json`` the
3D view moved the previews by, and each file the app reads (``KEEP``) with the time it
appeared::

    <replays>/<mode>/<scene>/timeline.json   # {"seconds", "returncode", "log", "poses", "files"}
    <replays>/<mode>/<scene>/run/...         # those files, at their paths in the run dir

``record`` makes one from a live run; ``Replay.run`` writes it into a fresh run dir on the
same schedule, compressed to ``SECONDS``, yielding the output as ``Worker.run`` does, so the
app follows a replay exactly as it follows a run. ``demo/record_examples.py`` records the
bundled scenes.
"""
from __future__ import annotations

import fnmatch
import json
import os
import shutil
import signal
import threading
import time
from pathlib import Path

#: What the app reads from a run dir (``fnmatch`` patterns on its relative paths, ``*``
#: crossing ``/``): the step table's config, the 3D view's sources, the downloads.
KEEP = (
    "config.yaml",
    "preprocessing/colmap/sparse/*.bin",   # the binary flavour the 3D view loads, not the .txt
    "preprocessing/colmap/images/*",
    "[0-9][0-9]_*/voxel_grid/*",
    "[0-9][0-9]_*/splats/*",
    "final/gaussians/world*.ply",
    "final/gaussians/background*.ply",
    "final/meshes/world.glb",
    "final/poses.json",
    "final/viz/orbit/orbit.mp4",
    "pose_cubes.glb",
)
POSES = "live_poses.json"
#: A replay's length, however long the run took.
SECONDS = 15.0


def recording(replays: Path, mode: str, scene: str) -> Path | None:
    """The recording of ``mode``/``scene`` under ``replays``, if there is one."""
    root = replays / mode / scene
    return root if (root / "timeline.json").is_file() else None


def record(chunks, run_dir: Path, out: Path, returncode) -> None:
    """Record the live run streaming ``chunks`` into ``run_dir``, as ``out``.

    ``returncode()`` is the run's exit code once ``chunks`` is exhausted. The output and
    the live poses are timed as they arrive; the files by their modification times.
    """
    t0, log, poses, done = time.time(), [], [], threading.Event()

    def watch():   # live_poses.json is rewritten in place: keep every version
        last = None
        while not done.wait(0.1):
            try:
                text = (run_dir / POSES).read_text().rstrip()   # without live_poses' padding
            except OSError:
                continue
            if text != last:
                poses.append((time.time() - t0, text))
                last = text

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        for chunk in chunks:
            log.append((time.time() - t0, chunk))
    finally:
        done.set()
        watcher.join()
    seconds = time.time() - t0
    if out.exists():
        shutil.rmtree(out)
    files = []
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(run_dir).as_posix()
        if any(fnmatch.fnmatch(rel, pattern) for pattern in KEEP):
            (out / "run" / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, out / "run" / rel)
            files.append((path.stat().st_mtime - t0, rel))
    # A sentinel last among files of the same instant: it says the others are complete.
    files.sort(key=lambda f: (f[0], f[1].endswith(".complete")))
    (out / "timeline.json").write_text(json.dumps({
        "seconds": seconds, "returncode": returncode(),
        "log": log, "poses": poses, "files": files}))


class Replay:
    """A recording, played into a run dir like ``Worker`` runs one."""

    def __init__(self, root: Path, seconds: float = SECONDS) -> None:
        self.root, self.seconds = root, seconds
        self.timeline = json.loads((root / "timeline.json").read_text())
        self.returncode: int | None = None
        self._stop = threading.Event()

    @property
    def live_seconds(self) -> float:
        """How long the recorded run took."""
        return self.timeline["seconds"]

    def stop(self) -> None:
        self._stop.set()

    def run(self, run_dir: Path):
        """Write the recording into ``run_dir`` on its schedule, compressed to ``seconds``,
        yielding its output; ``returncode`` is the recorded run's (``-SIGKILL`` if stopped)."""
        tl = self.timeline
        scale = min(1.0, self.seconds / max(tl["seconds"], 1e-6))
        events = sorted([(t, 0, "log", c) for t, c in tl["log"]]
                        + [(t, 1, "file", rel) for t, rel in tl["files"]]
                        + [(t, 2, "poses", text) for t, text in tl["poses"]],
                        key=lambda e: e[:2])
        self.returncode, start = None, time.time()
        for t, _, kind, payload in events:
            if self._stop.wait(max(0.0, start + t * scale - time.time())):
                self.returncode = -signal.SIGKILL
                return
            if kind == "log":
                yield payload
            elif kind == "file":
                src, dst = self.root / "run" / payload, run_dir / payload
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(src, dst)   # same disk: no copy
                except OSError:
                    shutil.copy2(src, dst)
            else:
                tmp = run_dir / (POSES + ".tmp")
                run_dir.mkdir(parents=True, exist_ok=True)
                tmp.write_text(payload + " " * (-len(payload) % 16384))   # as live_poses.PAD
                tmp.replace(run_dir / POSES)
        self.returncode = tl["returncode"]
