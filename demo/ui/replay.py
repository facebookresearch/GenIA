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
bundled scenes. Published recordings (``demo/space/replays_gcs.py push``; where, the
method's ``REPLAYS_URL``) are fetched a scene at a time when the app first plays one
(``fetch``).
"""
from __future__ import annotations

import concurrent.futures
import fnmatch
import functools
import json
import os
import shutil
import signal
import threading
import time
import urllib.request
from pathlib import Path

#: The published recordings' index: ``{"files": {relative path: size}}``.
INDEX = "index.json"

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


@functools.lru_cache(maxsize=None)
def published(base: str) -> dict:
    """The recordings published at ``base``, ``{relative path: size}``; empty when ``base`` is,
    or holds none, or cannot be reached (asked once per process)."""
    if not base:
        return {}
    try:
        with urllib.request.urlopen(f"{base}/{INDEX}", timeout=10) as res:
            return json.load(res)["files"]
    except (OSError, ValueError, KeyError):
        return {}


def is_published(mode: str, scene: str, base: str) -> bool:
    return f"{mode}/{scene}/timeline.json" in published(base)


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with urllib.request.urlopen(url, timeout=60) as res, open(tmp, "wb") as f:
        shutil.copyfileobj(res, f, 1 << 20)
    os.replace(tmp, path)   # a half-downloaded file is never taken for a whole one


def pull(base: str, dest: Path, files: dict, prefix: str = "", workers: int = 16) -> int:
    """Download the published ``files`` under ``prefix`` that ``dest`` lacks or holds at another
    size; the number fetched. The timeline last, so a recording is whole once it exists."""
    todo = [rel for rel, size in files.items() if rel.startswith(prefix)
            and not ((dest / rel).is_file() and (dest / rel).stat().st_size == size)]
    timelines = [rel for rel in todo if rel.endswith("timeline.json")]
    rest = [rel for rel in todo if rel not in timelines]
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        list(pool.map(lambda rel: _download(f"{base}/{rel}", dest / rel), rest))
    for rel in timelines:
        _download(f"{base}/{rel}", dest / rel)
    return len(todo)


def fetch(replays: Path, mode: str, scene: str, base: str) -> Path | None:
    """``mode``/``scene``'s published recording, downloaded into ``replays``; its local
    recording (None if none is published)."""
    if not is_published(mode, scene, base):
        return None
    pull(base, replays, published(base), prefix=f"{mode}/{scene}/")
    return recording(replays, mode, scene)


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
