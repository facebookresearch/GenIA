# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Each object's pose while it is being predicted and optimised, for a viewer following the run.

With ``GENIA_LIVE_POSES=1`` in the environment (the demo app sets it), the pose-optimising
loops, whenever ``due()``, ``record`` each object's current pose and ``flush``: the run's
``live_poses.json`` is rewritten (atomically) with every object's world placement::

    {"objects": {"obj_1": [16 numbers, the 4x4 row-major], ...}}

It is ``object_world_matrix``, the placement the ``obj_{i}`` nodes of the voxel grid and the
pose cubes (and the final meshes and splats) get, so a viewer moving those nodes shows each
object exactly where it is.
After every block the poses the run's state holds are published too (``publish_state``).

Static scenes only (one timestamp; a dynamic scene's objects move per frame). Without the
variable ``due()`` is always False, so the loops never do any of it.
"""
import json
import os
import time

from genia.core.utils.frame_key import as_frame_key

#: Seconds between two writes from the loops (a block's end always writes).
INTERVAL = 0.5
#: The file is padded with spaces to a multiple of this many bytes, so its successive versions
#: keep one size: a web server reading the size before the content (Starlette does) would
#: otherwise fail on a version replaced in between ("Too much data for declared Content-Length").
PAD = 16384

_run = None  # the run being followed, set by ``start``


def start(cfg, sequence, state):
    """Follow this run (a no-op unless ``GENIA_LIVE_POSES=1`` and the scene is static)."""
    global _run
    _run = None
    if os.environ.get("GENIA_LIVE_POSES") != "1" or sequence.is_dynamic:
        return
    from genia.core.utils.eval_assets_export import _world_anchor_c2w

    _run = {"path": os.path.join(cfg.output.output_dir, "live_poses.json"),
            "anchor_c2w": _world_anchor_c2w(cfg, sequence, None),
            "state": state, "poses": {}, "written": 0.0}


def due() -> bool:
    """Whether a loop should record and flush now: following a run, and ``INTERVAL`` since the
    last write. Loops check it first, so between writes they do no extra work."""
    return _run is not None and time.monotonic() - _run["written"] >= INTERVAL


def anchor(obj_idx):
    """The frame key whose pose places ``obj_idx`` in the world (its first token's, the one
    ``export_voxel_grid`` uses), or None."""
    tokens = _run["state"].tokens_by_object.get(obj_idx) if _run is not None else None
    return as_frame_key(tokens[0][0]) if tokens else None


def record(obj_idx, frame_key, rotation, translation, scale):
    """Note ``obj_idx``'s current pose; only its ``anchor`` frame's counts. ``rotation`` is the
    object-pose quaternion (wxyz) or its row-vector 3x3 matrix; ``scale`` a scalar or (3,)."""
    import torch
    from pytorch3d.transforms import matrix_to_quaternion

    if _run is None or frame_key is None or as_frame_key(frame_key) != anchor(obj_idx):
        return
    rotation = rotation.detach()
    if rotation.shape[-2:] == (3, 3):
        rotation = matrix_to_quaternion(rotation.reshape(1, 3, 3))
    rotation = rotation.reshape(1, 4) / rotation.norm()
    scale = torch.as_tensor(scale, dtype=torch.float32, device=rotation.device).detach()
    _run["poses"][obj_idx] = (rotation, translation.detach().reshape(1, 3), scale.reshape(-1))


def flush():
    """Write every recorded object's world placement to ``live_poses.json``."""
    if _run is None or not _run["poses"]:
        return
    from genia.core.utils.eval_assets_export import object_world_matrix
    from genia.core.utils.timing import get_timer

    _run["written"] = time.monotonic()
    with get_timer().exclude():   # a viewer's, not the block's work
        objects = {f"obj_{i}": object_world_matrix(r, t, s, _run["anchor_c2w"], r.device)
                   .ravel().tolist() for i, (r, t, s) in _run["poses"].items()}
        text = json.dumps({"objects": objects})
        tmp = _run["path"] + ".tmp"
        with open(tmp, "w") as f:
            f.write(text + " " * (-len(text) % PAD))   # see PAD
        os.replace(tmp, _run["path"])


def publish_state(state):
    """Publish the poses ``state`` holds, e.g. once a block has written its own."""
    if _run is None:
        return
    for obj_idx, tokens in state.tokens_by_object.items():
        if obj_idx != 0 and tokens and "rotation" in tokens[0][1]:
            fk, d = tokens[0]
            record(obj_idx, fk, d["rotation"], d["translation"], d["scale"])
    flush()
