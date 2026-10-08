#!/usr/bin/env python3
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


def scenes(setting):
    root = os.path.join(DATA, setting)
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))


def build_overrides(setting, scene=None, icp=False, extra=(), per_frame_shapes=False,
                    minimal_outputs=True):
    """The Hydra overrides of one demo run."""
    spec = SETTINGS[setting]
    scene = scene or spec["default_scene"]
    if scene not in scenes(setting):
        raise ValueError(f"unknown {setting} scene {scene!r}; available: {scenes(setting)}")
    if icp and not spec["static"]:
        raise ValueError(f"--icp applies to the static settings (image, multiview), not {setting}")
    if per_frame_shapes and setting != "dynamic":
        raise ValueError(f"--per-frame-shapes applies to the dynamic setting, not {setting}")
    overrides = [
        *spec["overrides"],
        *NO_CACHE,
        *(MINIMAL_OUTPUTS if minimal_outputs else []),
        f"dataset.path={os.path.join(DATA, setting)}",
        f"dataset.scene_name={scene}",
    ]
    if setting != "dynamic":
        overrides += NO_SHAPE_CACHE
    elif per_frame_shapes:
        overrides += PER_FRAME_SHAPES
    else:
        overrides += SHAPE_INJECTION + [
            f"gt_shapes_inversion.gt_data_root={os.path.join(DATA, setting)}"]
    if spec["static"]:
        overrides += ["output.experiment_suffix=icp"] if icp else PHOTOMETRIC_REFINE
    return overrides + list(extra)


def actionmesh_command(scene):
    """The command that fits ``scene``'s ActionMesh meshes, or None when they exist."""
    scene_dir = os.path.join(DATA, "dynamic", scene)
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
            "--input_path", os.path.join(DATA, "dynamic"), "--input_scene", scene,
            "--input_stride", "1", "--input_length", str(n_frames),
            "--output_dir", out_dir]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("setting", choices=sorted(SETTINGS))
    parser.add_argument("--scene", help="scene under demo/data/<setting>/ (default: per setting)")
    parser.add_argument("--icp", action="store_true",
                        help="static settings only: ICP pose refinement instead of the "
                             "per-frame appearance + photometric one")
    parser.add_argument("--per-frame-shapes", action="store_true",
                        help="dynamic only: SAM3D's per-frame shapes instead of ActionMesh meshes")
    parser.add_argument("--minimal-outputs", action=argparse.BooleanOptionalAction, default=True,
                        help="skip the per-block renders, videos and metrics; final/ is always "
                             "written (default: on)")
    parser.add_argument("--dry-run", action="store_true", help="print the command(s) and exit")
    args, extra = parser.parse_known_args()

    try:
        overrides = build_overrides(args.setting, args.scene, args.icp, extra,
                                    args.per_frame_shapes, args.minimal_outputs)
        stage = (actionmesh_command(args.scene or SETTINGS["dynamic"]["default_scene"])
                 if args.setting == "dynamic" and not args.per_frame_shapes else None)
    except ValueError as exc:
        parser.error(str(exc))
    cmds = ([stage] if stage else []) + [[sys.executable, "-m", "genia.core", *overrides]]
    for cmd in cmds:
        print(shlex.join(cmd), flush=True)
        if not args.dry_run:
            rc = subprocess.run(cmd).returncode
            if rc:
                return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
