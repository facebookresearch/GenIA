#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Record the bundled example scenes, for the demo app to replay (``demo/ui/replay.py``).

    python demo/record_examples.py                       # every bundled scene of every mode
    python demo/record_examples.py --mode image --scene sloth
    python demo/record_examples.py --work-dir DIR --force

Each recording is one run of the app's default recipe (no run option ticked) on this
machine's GPU, through the app's warm worker, written to ``<work-dir>/replays/`` (the app's
work dir, default ``results/demo_app``). An existing recording is kept unless ``--force``.
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import app
import method
from ui import replay


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", default=str(method.ROOT / "results" / "demo_app"))
    parser.add_argument("--mode", choices=sorted(method.MODES), help="default: every mode")
    parser.add_argument("--scene", help="default: every bundled scene of the mode(s)")
    parser.add_argument("--force", action="store_true", help="redo existing recordings")
    args = parser.parse_args()
    replays = Path(args.work_dir).resolve() / "replays"
    todo = [(mode, d.name) for mode in ([args.mode] if args.mode else method.MODES)
            for d in sorted((app.DATA / mode).iterdir())
            if d.is_dir() and args.scene in (None, d.name)]
    if not todo:
        parser.error("no such bundled scene")
    roomy, low_vram = app.gpu_policy()
    worker = app.new_runner(roomy)
    failed = 0
    for mode, scene in todo:
        if replay.recording(replays, mode, scene) and not args.force:
            print(f"{mode}/{scene}: recorded already (--force redoes it)")
            continue
        print(f"{mode}/{scene}: recording…", flush=True)
        with tempfile.TemporaryDirectory(dir=replays.parent if replays.parent.is_dir() else None) as tmp:
            run_dir = Path(tmp) / "run"
            overrides = method.run_overrides(
                mode, app.DATA, scene, low_vram,
                [*method.RUN_ARGS, *worker.warm, f"output.output_dir={run_dir}"])
            replay.record(worker.run(overrides), run_dir, replays / mode / scene,
                          lambda: worker.returncode)
        if worker.returncode:
            failed += 1
            shutil.rmtree(replays / mode / scene)   # a failed run is not worth replaying
            print(f"{mode}/{scene}: the run failed (exit code {worker.returncode})")
            worker.start()
        else:
            print(f"{mode}/{scene}: done", flush=True)
    worker.stop()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
