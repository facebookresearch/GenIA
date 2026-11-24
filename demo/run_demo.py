#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Run GenIA on the bundled demo inputs.

    python demo/run_demo.py image                    # single image (default scene: sloth)
    python demo/run_demo.py image --scene lab_duo
    python demo/run_demo.py image --icp              # ICP pose refinement instead of photometric
    python demo/run_demo.py multiview                # 2 views of one static scene (pablo)
    python demo/run_demo.py dynamic                  # 16-frame clip, static camera (dinosaur)
    python demo/run_demo.py dynamic --per-frame-shapes   # SAM3D's per-frame shapes, no ActionMesh
    python demo/run_demo.py image output.save_viz_orbit=false   # extra Hydra overrides pass through
    python demo/run_demo.py image --dry-run          # print the command instead of running it
    python demo/run_demo.py image --no-minimal-outputs   # also write every block's renders and metrics
    python demo/run_demo.py image --data-root DIR --scene NAME   # your own scene: DIR/image/NAME/
    python demo/run_demo.py image --low-vram         # fit a 24 GB GPU (default: auto)

Each setting is one ``python -m genia.core`` call. None of the inputs carry depth or
cameras, so map-anything predicts both. Results are written to
``results/{experiment}/{dataset}/{scene}/{timestamp}/`` in the checkout
(``GENIA_RESULTS`` overrides it). Every run starts from scratch: the pipeline cache, the
Stage-1 shape cache and the map-anything depth cache are off (re-enable one by passing
its override, e.g. ``processing.resume_from_cache=true``). Only ``final/`` and
``preprocessing/``'s COLMAP export are written by default: the per-block renders, videos and
metrics are skipped (``--no-minimal-outputs`` keeps them).

Dynamic scenes take their shapes from ActionMesh: per-frame meshes of one deforming
object, fitted into ``{scene}/actionmesh/`` by ``core/actionmesh_video.py`` on first use,
are injected in place of SAM3D's shape prediction. ``--per-frame-shapes`` uses SAM3D's
independent per-frame shapes instead (no canonical object, so no tracks or interpolation).

``--low-vram`` gradient-checkpoints FINETUNE's decoder on every object, not only the large
ones, at about twice FINETUNE's time: the image and multi-view demos then peak at ~16 GB
instead of running out of memory on a 24 GB GPU. By default it is on when the GPU has
less than ``LOW_VRAM_BELOW_GIB`` free.
"""
import argparse
import glob
import os
import shlex
import subprocess
import sys

GENIA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(GENIA, "demo", "data")

# No caching: no resume from (or writing of) the pipeline cache, no Stage-1 shape cache.
# The map-anything depth cache is off by default (processing.cache_reconstruction).
NO_CACHE = [
    "processing.resume_from_cache=false",
    "output.save_cache=false",
]
# Only where shape_init runs: its group carries the key.
NO_SHAPE_CACHE = ["shape_init.shape_cache_dir=null"]

# Skip the per-block renders, videos and metrics; FINAL still writes all of final/.
MINIMAL_OUTPUTS = [
    "output.suppress_intermediate_renders=true",
    "output.suppress_intermediate_metrics=true",
]

# Checkpoint FINETUNE's decoder whatever the object's size (the default only does from
# 30k voxels, sized for 40 GB), and let the allocator grow segments instead of fragmenting.
# `++` (add or override): `finetuning=none` has no such key, and must still compose.
LOW_VRAM = ["++finetuning.decoder_checkpoint_min_voxels=0"]
LOW_VRAM_ENV = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
#: ``--low-vram`` defaults on below this much FREE GPU memory: an idle 24, 32 or 40 GB card,
#: or a larger one that other processes already use.
LOW_VRAM_BELOW_GIB = 44

SETTINGS = {
    # One RGB image + instance mask: {scene}/0000.png + 0000_mask.png.
    "image": {
        "default_scene": "sloth",
        "overrides": [
            "+experiment=mono_static",
            "dataset=image",
            "pipeline=mono_ours",
            "processing=map_anything",
        ],
        "static": True,
    },
    # Several stills of one static scene: {scene}/rgbs/NNNNN.png + segmentations/NNNNN.png.
    "multiview": {
        "default_scene": "pablo",
        "overrides": [
            "+experiment=mv_static",
            "dataset=mvcustom",
            "pipeline=mv_ours",
            "processing=map_anything",
        ],
        "static": True,
    },
    # A monocular clip, one frame per timestep: {scene}/rgbs/ + segmentations/, plus an
    # optional poses.json with the camera per frame (all identity here: a static camera).
    # The shape source (SHAPE_INJECTION or PER_FRAME_SHAPES below) is added on top.
    "dynamic": {
        "default_scene": "dinosaur",
        "overrides": [
            "+experiment=mono_dyn",
            "dataset=dyncustom",
            "pipeline=mono_ours",
            "processing=map_anything",
            "global_pose_refine@global_pose_refine_1=none",
        ],
        "static": False,
    },
}

# Static scenes (image, multiview) by default run a per-frame appearance pass before the
# first pose refinement and make that refinement render-based: it renders those per-frame
# Gaussians, so the two go together. Its RGB terms are down-weighted to a tenth
# (global_pose_refine/low_rgb.yaml), so depth and chamfer lead. --icp keeps the pipeline
# default instead: no early appearance pass and an ICP refinement against the shape.
PHOTOMETRIC_REFINE = [
    "appearance_init=perframe",
    "global_pose_refine@global_pose_refine_1=low_rgb",
]


# Dynamic shape source (default): ActionMesh meshes staged at
# {scene}/actionmesh/obj_NNN/mesh_NN.glb instead of SAM3D's shape prediction. With one
# deforming shape per object, appearance fuses onto a canonical object (canonical_unified)
# and FINETUNE runs its canonical strategy.
SHAPE_INJECTION = [
    "appearance_init@appearance_init_2=canonical_unified",
    "gt_shapes_inversion=actionmesh",
    "gt_shapes_inversion.scene_subdir=actionmesh",
    "shape_and_poses_init@shape_init=none",
]

# Dynamic shape source with --per-frame-shapes: SAM3D predicts every frame's shape
# independently, so appearance stays per frame and FINETUNE runs its per-frame strategy.
# No canonical object is built, which the final pose refinement (canonical scope) needs;
# FINETUNE still refines each frame's pose.
PER_FRAME_SHAPES = [
    "appearance_init@appearance_init_2=perframe_guided",
    "global_pose_refine@global_pose_refine_2=none",
    "finetuning=perframe",
    *NO_SHAPE_CACHE,
    "output.experiment_suffix=per_frame_shapes",
]

#: ActionMesh's frame budget per clip.
ACTIONMESH_FRAMES = (16, 31)


def scenes(setting, data_root=DATA):
    root = os.path.join(data_root, setting)
    if not os.path.isdir(root):
        return []
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))


def default_low_vram():
    """Whether ``--low-vram`` is on when not given: under ``LOW_VRAM_BELOW_GIB`` free on the GPU.

    Asked in a child process: a CUDA query in this one would keep a CUDA context (~400 MB)
    on the GPU for the whole run.
    """
    query = "import torch; print(torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else -1)"
    out = subprocess.run([sys.executable, "-c", query], capture_output=True, text=True)
    free = int(out.stdout.split()[-1]) if out.returncode == 0 else -1
    return 0 <= free < LOW_VRAM_BELOW_GIB * 2**30


def build_overrides(setting, scene=None, icp=False, extra=(), per_frame_shapes=False,
                    minimal_outputs=True, data_root=DATA, low_vram=False):
    """The Hydra overrides of one demo run, on ``data_root/<setting>/<scene>/``."""
    spec = SETTINGS[setting]
    scene = scene or spec["default_scene"]
    root = os.path.join(os.path.abspath(data_root), setting)
    if scene not in scenes(setting, data_root):
        raise ValueError(f"unknown {setting} scene {scene!r} in {root}; "
                         f"available: {scenes(setting, data_root)}")
    if icp and not spec["static"]:
        raise ValueError(f"--icp applies to the static settings (image, multiview), not {setting}")
    if per_frame_shapes and setting != "dynamic":
        raise ValueError(f"--per-frame-shapes applies to the dynamic setting, not {setting}")
    overrides = [
        *spec["overrides"],
        *NO_CACHE,
        *(MINIMAL_OUTPUTS if minimal_outputs else []),
        f"dataset.path={root}",
        f"dataset.scene_name={scene}",
    ]
    if setting != "dynamic":
        overrides += NO_SHAPE_CACHE
    elif per_frame_shapes:
        overrides += PER_FRAME_SHAPES
    else:
        overrides += SHAPE_INJECTION + [f"gt_shapes_inversion.gt_data_root={root}"]
    if spec["static"]:
        overrides += ["output.experiment_suffix=icp"] if icp else PHOTOMETRIC_REFINE
    if low_vram:
        overrides += LOW_VRAM
    return overrides + list(extra)


def actionmesh_command(scene, data_root=DATA):
    """The command that fits ``scene``'s ActionMesh meshes, or None when they exist."""
    scene_dir = os.path.join(os.path.abspath(data_root), "dynamic", scene)
    out_dir = os.path.join(scene_dir, "actionmesh")
    if glob.glob(os.path.join(out_dir, "obj_*", "mesh_*.glb")):
        return None
    n_frames = len(glob.glob(os.path.join(scene_dir, "rgbs", "*")))
    lo, hi = ACTIONMESH_FRAMES
    if not lo <= n_frames <= hi:
        raise ValueError(
            f"ActionMesh fits clips of {lo}-{hi} frames; {scene} has {n_frames}. "
            f"Use --per-frame-shapes, or trim the clip.")
    return [sys.executable, os.path.join(GENIA, "core", "actionmesh_video.py"),
            "--input_path", os.path.dirname(scene_dir), "--input_scene", scene,
            "--input_stride", "1", "--input_length", str(n_frames),
            "--output_dir", out_dir]


def _parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("setting", choices=sorted(SETTINGS))
    parser.add_argument("--scene", help="scene under <data-root>/<setting>/ (default: per setting)")
    parser.add_argument("--data-root", default=DATA,
                        help="directory holding <setting>/<scene>/, laid out like demo/data "
                             "(default: demo/data)")
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=None,
                        help="checkpoint FINETUNE's decoder on every object, ~2x its time, to "
                             f"fit a 24 GB GPU (default: on below {LOW_VRAM_BELOW_GIB} GiB free)")
    parser.add_argument("--icp", action="store_true",
                        help="static settings only: ICP pose refinement instead of the "
                             "per-frame appearance + photometric one")
    parser.add_argument("--per-frame-shapes", action="store_true",
                        help="dynamic only: SAM3D's per-frame shapes instead of ActionMesh meshes")
    parser.add_argument("--minimal-outputs", action=argparse.BooleanOptionalAction, default=True,
                        help="skip the per-block renders, videos and metrics; final/ is always "
                             "written (default: on)")
    parser.add_argument("--dry-run", action="store_true", help="print the command(s) and exit")
    return parser


def plan(argv=None):
    """One run of this script's command line ``argv``, without running it: ``(overrides,
    env, stage)``, the ``python -m genia.core`` overrides, the environment they need and
    the command staging the scene first (or None). For a host that runs the pipeline
    itself, e.g. the demo app's warm worker. Raises ValueError on an unusable request."""
    args, extra = _parser().parse_known_args(argv)
    low_vram = default_low_vram() if args.low_vram is None else args.low_vram
    overrides = build_overrides(args.setting, args.scene, args.icp, extra,
                                args.per_frame_shapes, args.minimal_outputs,
                                args.data_root, low_vram)
    stage = (actionmesh_command(args.scene or SETTINGS["dynamic"]["default_scene"],
                                args.data_root)
             if args.setting == "dynamic" and not args.per_frame_shapes else None)
    return overrides, (LOW_VRAM_ENV if low_vram else {}), stage


def main():
    parser = _parser()
    args, _ = parser.parse_known_args()
    try:
        overrides, env, stage = plan()
    except ValueError as exc:
        parser.error(str(exc))
    cmds = ([stage] if stage else []) + [[sys.executable, "-m", "genia.core", *overrides]]
    for cmd in cmds:
        print(shlex.join([f"{k}={v}" for k, v in env.items()] + cmd), flush=True)
        if not args.dry_run:
            rc = subprocess.run(cmd, env={**os.environ, **env}).returncode
            if rc:
                return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
