# Running the Pipeline (CLI Reference)

This document describes how to run GenIA with `python -m genia.core` ([`core/run.py`](../core/run.py)), which composes the Hydra config rooted at [`core/configs/genia.yaml`](../core/configs/genia.yaml). Overrides use Hydra's `key=value` syntax (not `--key value`).

## Demo options

`demo/run_demo.py <setting>` prints and runs the full `python -m genia.core` command for
each demo setting (`--dry-run` only prints it); extra `key=value` overrides are passed
through. Demo runs use no caches, so each starts from scratch: pass
`processing.resume_from_cache=true output.save_cache=true` (pipeline cache),
`processing.cache_reconstruction=true` (map-anything depth) or
`shape_init.shape_cache_dir=cached_shapes` (Stage-1 shapes; every setting that runs
`shape_init`, i.e. not the default dynamic one) to turn one back on.

Dynamic scenes take their shapes from ActionMesh by default: its per-frame meshes of one
deforming object are fitted on first use into `{scene}/actionmesh/`
(`core/actionmesh_video.py`) and injected in place of SAM 3D Objects' shape prediction
(`gt_shapes_inversion=actionmesh`). ActionMesh reads its weights from the Hugging Face cache,
which `python -m genia.core.download_weights` fills. It fits clips of 16-31 frames.

Further flags:

| Flag | Settings | Effect |
|---|---|---|
| `--scene <name>` | all | a scene folder under `demo/data/<setting>/` |
| `--icp` | `image`, `multiview` | keeps the pipeline's ICP pose refinement. By default the static demos add a per-frame appearance pass before the first pose refinement and make that refinement render-based with the RGB terms at a tenth of their weight (`appearance_init=perframe`, `global_pose_refine@global_pose_refine_1=low_rgb`) |
| `--no-minimal-outputs` | all | also writes every block's renders, videos and metrics. By default the demo sets `output.suppress_intermediate_renders=true` and `output.suppress_intermediate_metrics=true`, so only `final/` and `preprocessing/colmap/` are written |
| `--per-frame-shapes` | `dynamic` | uses SAM 3D Objects' independent per-frame shapes instead of ActionMesh's. No canonical object is built, so the run writes per-frame outputs without tracks or interpolation |

## Overview

A run:
1. **PREPROCESSING** builds the `Sequence`: images, masks, depth and cameras (`processing=`), and writes diagnostics to `preprocessing/`.
2. Runs the pipeline blocks in order. Each block's runner skips itself when its section has `enabled: false`; the `pipeline=` preset selects which blocks run and how.
3. Each block evaluates itself (before/after renders, PSNR/SSIM/LPIPS, keyframe videos) when its output flags allow.
4. **FINAL** exports poses, Gaussians, meshes, renders, tracks and timing to `final/`.

## Pipeline Blocks

Authoritative order: `BLOCKS` in [`core/manifest.py`](../core/manifest.py).

```
PREPROCESSING         Build the Sequence (depth + cameras); always runs
GT_SHAPES_INVERSION   Optional: invert GT meshes into Stage-1 shape latents (and poses)
SHAPE_INIT            Stage-1 batched ODE: settle each frame's shape (pose kept as a fallback)
APPEARANCE_INIT       Optional: Stage-2 appearance before the pose pass (off by default)
POSE_INIT             Stage-1 batched ODE: denoise pose over the settled shape, then refit (t, s);
                      a frame rotated past pose_flip_guard_deg keeps SHAPE_INIT's pose
GLOBAL_POSE_REFINE_1  Pose refinement (ICP by default)
APPEARANCE_INIT_2     Stage-2 appearance (canonical_unified by default)
FINETUNE              Token + LoRA fine-tuning, jointly with pose
GLOBAL_POSE_REFINE_2  Final photometric pose refinement
FINAL                 Export; always runs
```

## Basic Usage

```bash
# One image
python -m genia.core +experiment=mono_static dataset=image dataset.path=<dir> dataset.scene_name=<scene>

# Monocular video, with SAM 3D Objects' per-frame shapes (the demo's --per-frame-shapes)
python -m genia.core +experiment=mono_dyn dataset=dyncustom dataset.path=<dir> dataset.scene_name=<scene> \
    global_pose_refine@global_pose_refine_1=none appearance_init@appearance_init_2=perframe_guided \
    finetuning=perframe global_pose_refine@global_pose_refine_2=none

# Multi-view static capture
python -m genia.core +experiment=mv_static dataset=mvcustom dataset.path=<dir> dataset.scene_name=<scene>

# GSO single view, ground-truth depth and cameras
python -m genia.core +experiment=mono_static dataset=gso dataset.scene_name=elephant \
    dataset.num_input_views=1 processing=ground_truth
```

`+experiment=` presets ([`core/configs/experiment/`](../core/configs/experiment/)) bundle the recipe (`ours.yaml`); `mv_static` also selects `pipeline=mv_ours`. The dataset is chosen separately with `dataset=`. The benchmark datasets (`gso`, `co3d`, `davis_actionmesh`, `oursactionbench`) are not shipped; their configs in [`core/configs/dataset/`](../core/configs/dataset/) expect them under `data/`.

## Configuration

### Dataset (`dataset=`)

| Config key | Description |
|------------|-------------|
| `dataset=` | `image`, `mvcustom`, `dyncustom`, `gso`, `co3d`, `oursactionbench`, `davis_actionmesh` |
| `dataset.path` | Dataset root (each dataset config sets a default under `data/`) |
| `dataset.scene_name` | Scene to process |
| `dataset.frame_index` | Process a single frame (null = all frames at `frame_stride`) |
| `dataset.frame_stride` | Stride for iterating frames |
| `dataset.object_indices` | Only process these object ids, e.g. `dataset.object_indices='[1]'` |
| `dataset.num_input_views` | Multi-view static datasets (`gso`, `co3d`, `mvcustom`): keep only the first N of the loaded views as input (null = all loaded views). The held-out split is fixed by the dataset, not by this key: GSO loads input candidates 0-9 (N <= 10) and always tests on views 10-24; CO3D loads its 4 input + 4 target views, so N=4 (the default) keeps the targets out; mvcustom has no held-out split |
| `dataset.downscale_factor` | Integer downscale of images, masks, depth and pointmaps by stride slicing; intrinsics are adjusted |

`is_dynamic` / `is_mv` are not config fields: they are properties of the loaded `Sequence`, derived from its `FrameKey` set.

### Processing (`processing=`)

| Option | Depth | Cameras |
|--------|-------|---------|
| `map_anything` (root default) | predicted by map-anything | predicted by map-anything |
| `moge` | MoGe per-frame monocular depth | identity |
| `ground_truth` | dataset GT | dataset GT |

`depth_source` / `camera_poses_source` (`pred` | `gt`; `gt` raises if the dataset lacks the signal) are set by each option's file (`core/configs/processing/{map_anything,moge,ground_truth}.yaml`). Related keys in [`core/configs/processing/default.yaml`](../core/configs/processing/default.yaml): `condition_recon_model_on_gt_poses` / `condition_recon_model_on_gt_depths` (feed GT signals to map-anything as a soft prior), `recon_resolution_set` (518 or 504), `recon_mask_edges`, `recon_confidence_min`, `cache_reconstruction`, `resume_from_cache`.

### Block configuration

| Config key | Options | Description |
|------------|---------|-------------|
| `gt_shapes_inversion=` | `none`, `gso`, `actionmesh`, `oursactionbench` | Invert GT meshes into shape latents; the per-frame modes also build a deformation field |
| `shape_and_poses_init@shape_init=` | `parallel_shape`, `none` | Stage-1 shape pass |
| `shape_and_poses_init@pose_init=` | `parallel_pose`, `none` | Stage-1 pose pass |
| `appearance_init=` / `appearance_init@appearance_init_2=` | `canonical_unified`, `perframe`, `perframe_guided`, `none` | Stage-2 appearance |
| `global_pose_refine@global_pose_refine_1=` / `@global_pose_refine_2=` | `default`, `low_rgb`, `icp`, `none` | Render-based (`default`: RGB + depth + chamfer; `low_rgb`: RGB at a tenth of its weight) or ICP pose refinement |
| `finetuning=` | `canonical`, `perframe`, `none` | `canonical` optimizes one shared SLAT per object against every frame; `perframe` optimizes each frame's own SLAT against that frame alone (for runs without a canonical object), spending `num_iterations` per frame |

Individual keys are overridden through the block's section, e.g. `pose_init.inference_steps=25`, `appearance_init_2.inference_steps=10`, `global_pose_refine_2.num_iterations=200`, `finetuning.losses.perceptual_weight=0.1`.

Every block section also has per-block output flags `save_renders`, `save_metrics`, `save_cache` (null = inherit `output.*`) and `exit_after` (stop the pipeline after this block).

### Pipeline flags (`pipeline.*`)

| Config key | Default | Description |
|------------|---------|-------------|
| `pipeline.mv_shared_world_pose` | `true` | Multi-view: one shared object placement per timestamp; every view's camera-space pose is derived from the reference view's through the camera rebase. Turned off automatically on monocular data, where it is vacuous |
| `pipeline.optimize_pose_tokens` | `true` | Optimize raw Stage-1 pose tokens (through the differentiable pose decoder) instead of final pose parameters |
| `pipeline.white_background` | `true` | Render Gaussians on white instead of black |
| `pipeline.verbose`, `pipeline.log_interval` | `true`, `20` | Logging during refinement |

### Output flags (`output.*`)

Selected keys from [`core/configs/output/default.yaml`](../core/configs/output/default.yaml):

| Config key | Default | Description |
|------------|---------|-------------|
| `output.output_dir` | null | Output directory; auto-generated when null |
| `output.experiment_suffix` | null | Appended to the experiment path segment |
| `output.save_renders` / `save_metrics` / `save_cache` | true | Global defaults for the per-block flags |
| `output.suppress_intermediate_renders` / `_metrics` | false | Force the per-block render/metric flags off (FINAL unaffected) |
| `output.save_output_mesh` | true | Per-object canonical meshes in `final/meshes/` |
| `output.save_output_renders` | true | Foreground renders from each input view in `final/renders_train/` |
| `output.save_colmap` | true | COLMAP sparse models in `final/colmap/` (and `preprocessing/colmap/`) |
| `output.save_world_assets` | true | World-space composites `gaussians/world*`, `gaussians/background*`, `meshes/world*` |
| `output.save_tracks_2d` / `save_tracks_3d` | true | Track data `final/tracks_2d.npz` / `final/tracks_3d.npz` |
| `output.save_synth_nvs` | null (auto) | Four synthesized novel views per timestamp in `final/renders_synth_nvs/`; auto = on for datasets without a GT held-out split |
| `output.save_viz_orbit` | true | Turntable `final/viz/orbit/orbit.mp4` |
| `output.save_viz_world_space` | true | Scene overview renders in `final/viz/world_space/` |

## Output Files

Results are written to `results/{experiment}/{dataset}/{scene}/{timestamp}/` in the checkout; set `GENIA_RESULTS` to write them elsewhere (model weights are read from the usual Hugging Face and torch hub caches, which `HF_HUB_CACHE` / `TORCH_HOME` move; `GENIA_SAM3D_WEIGHTS` points at a SAM 3D Objects `checkpoints/` folder kept elsewhere). `{experiment}` is the `+experiment=` name with `_<output.experiment_suffix>` appended (or the suffix alone), and is omitted when neither is set. The pipeline cache is a sibling of the timestamp directory, shared by runs of the same scene:

```
results/{experiment}/{dataset}/{scene}/
├── cache/                      # Pipeline state checkpoints, one {block}.pt per block
└── {timestamp}/
    ├── config.yaml             # Full Hydra config snapshot
    ├── preprocessing/          # Depth/normals/pointmap panels, summary grid, colmap/ (input cameras + depth cloud)
    ├── 00_shape_init/          # One folder per ENABLED block that wrote something: renders, metrics, keyframe videos
    ├── 01_pose_init/
    ├── 02_global_pose_refine_1/
    ├── 03_appearance_init_2/
    ├── 04_finetune/
    ├── 05_global_pose_refine_2/
    └── final/
        ├── poses.json          # Per-object per-frame Sim(3) transforms + per-frame cameras
        ├── gaussians/          # Per-object canonical Gaussians + world-space composites
        ├── meshes/             # Per-object canonical meshes (GLB) + world-space composites
        ├── renders_train/      # Foreground renders from each input view (RGBA)
        ├── renders_test/       # Held-out test views (gso, co3d, oursactionbench)
        ├── renders_synth_nvs/  # Synthesized novel views (datasets without a test split)
        ├── colmap/             # COLMAP sparse model(s), one per timestamp
        ├── tracks_2d.npz, tracks_3d.npz
        ├── viz/                # orbit/, world_space/, track_overlays/
        ├── final.pt            # Full pipeline state
        └── timing.json         # Per-block timing
```

Block folders are prefixed with a two-digit run-order index over the **enabled** blocks, so the numbering is contiguous and depends on the configuration (the tree above is the default recipe, where `gt_shapes_inversion` and `appearance_init` are off). `preprocessing/` and `final/` are always unprefixed.

`final/` can be re-rendered post hoc with `python -m genia.core.utils.render_final_results`.

## Evaluation Metrics

Each block reports, on the input views:

| Metric | Range | Better | Description |
|--------|-------|--------|-------------|
| **PSNR** | 0 to inf dB | Higher | Peak Signal-to-Noise Ratio |
| **SSIM** | 0 to 1 | Higher | Structural Similarity Index |
| **LPIPS** | 0 to 1 | Lower | Learned Perceptual Image Patch Similarity |

## Examples

### Faster appearance pass

```bash
python -m genia.core +experiment=mono_static dataset=image dataset.scene_name=<scene> \
    appearance_init_2.inference_steps=10
```

### GT shape injection on a mono-dynamic benchmark

```bash
python -m genia.core +experiment=mono_dyn dataset=davis_actionmesh dataset.scene_name=camel \
    gt_shapes_inversion=actionmesh shape_and_poses_init@shape_init=none
```

### Phase-specific loss overrides

```bash
python -m genia.core +experiment=mono_static dataset=image dataset.scene_name=<scene> \
    global_pose_refine_2.num_iterations=200 \
    global_pose_refine_2.perceptual_weight=0.1 \
    finetuning.num_iterations=200
```

### Stop after a block

```bash
python -m genia.core +experiment=mono_dyn dataset=dyncustom dataset.scene_name=<scene> \
    pose_init.exit_after=true
```

## Troubleshooting

### Memory

Reduce the number of frames or the input resolution:

```bash
python -m genia.core ... dataset.frame_stride=2        # fewer frames
python -m genia.core ... dataset.downscale_factor=2    # half resolution
```

`finetuning.decoder_checkpoint` (on by default) checkpoints the in-loop decoder for objects above `finetuning.decoder_checkpoint_min_voxels`; lower that threshold to extend it to smaller objects.

### Datasets without depth or cameras

`image`, `dyncustom`, `mvcustom`, `davis_actionmesh` and `co3d` have no GT depth; use `processing=map_anything` (the default) or `processing=moge`. `processing=ground_truth` requires a dataset that ships depth and cameras.
