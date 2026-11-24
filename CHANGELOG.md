<!-- Copyright (c) Meta Platforms, Inc. and affiliates. -->

Changes since the initial release (October 8, 2026), newest first.

November 24, 2026

### Demo app: reconstruct in the browser, and watch it happen

`python demo/app.py` is a Gradio app over `demo/run_demo.py`'s recipe (`--dev` adds the log).

* Single image and Multi-view modes, from the bundled scenes or an upload masked with SAM3
  (name the objects, or click them on the first view).
* A live 3D view (3DView): the input cameras and depth, each object's voxel grid and pose cube
  moving as the poses are optimised, its Gaussians as soon as they exist (on pablo the first
  at ~1 min), then the final mesh, Gaussians and background. One progress bar per loop and per
  object.
* Fast by default: FINETUNE 40 iterations instead of 100 (35-64% less time on the bundled
  scenes, meshes within a rerun's difference, 0.6-1.5 dB lower input-view PSNR) unless
  **Full quality** is ticked; **No test-time refinement**, **ICP pose refinement** and
  **Low-VRAM mode** (auto below 44 GiB free) are options.
* A warm worker keeps the models loaded between runs (preprocessing ~1 s after a click
  instead of ~37 s). Recorded examples (`demo/record_examples.py`) replay a full-quality run
  in ~15 s without a GPU; untick **Play recorded runs** to run one live.
* `--in-process` runs the method in the app's process for a ZeroGPU Space, one GPU call per
  live run. A Space for the app is in `demo/space/` (not published yet).
* Everything method-specific is in `demo/method.py`.

### Demo command line

* `demo/run_demo.py --data-root <dir>` reads scenes outside the checkout.
* `--low-vram` fits a 24 GB GPU (peak ~16 GB: sloth 221 s, pablo 521 s on an RTX 4090); on
  by itself below 44 GiB free, `--no-low-vram` keeps the previous behaviour.

### Pipeline

* `GENIA_PROGRESS=1` prints machine-readable progress (`@@progress {"task", "n", "total"}`,
  per object; `core/utils/progress.py`), and `progress.request_stop()` ends a run in the same
  process. `GENIA_LIVE_POSES=1` streams each object's world placement to `live_poses.json`
  while poses are optimised (static scenes). Without the variables nothing changes.
* `output.save_previews` writes only the 3D-view previews (voxel grids, per-object Gaussians,
  pose cubes): a multi-view run ~40% faster (pablo 412 s → 240 s). The voxel grid replaces
  the marching-cubes `coarse_shape/` preview, and with it the `diso` dependency.
* `processing.keep_recon_model` keeps map-anything loaded for the next run in the process.
* `processing.block_cache_keyed` names each block's cache after the config it depends on, so
  a resumed run re-runs only the blocks whose settings changed (a new FINETUNE iteration
  count re-runs FINETUNE and GLOBAL_POSE_REFINE_2). Both off by default.

### Python 3.12 and torch 2.8

* `install.sh` builds on Python 3.12, torch 2.8.0 / torchvision 0.23.0 (CUDA 12.8; was 3.11,
  2.7.1), xformers 0.0.32.post2, Open3D 0.19.0. Same results within run-to-run spread, same
  speed (sloth, lab_duo, pablo at the paper settings: meshes 0.2-0.3% Chamfer from the torch
  2.7 runs, where another seed moves pablo's by 0.8-0.9%). Runs are not bit-identical even
  with the same seed. Rebuild an existing environment to update.

### New submodules

* SAM3 (`submodules/sam3`, gated weights via `download_weights`, `--no-sam3` skips them) and
  3DView (`submodules/3DView`, built by `install.sh` with Node >= 22); both in
  `THIRD_PARTY_LICENSES.md`.

November 17, 2026

### Pixel alignment: half-pixel offsets fixed

The camera intrinsics follow gsplat's convention (pixel `i` spans `[i, i + 1)`, its centre at
`i + 0.5`; a centred camera has `cx = W / 2`), but several steps treated pixel `i` as sitting
at `i`. Each put the geometry or the targets up to half a pixel off the image they belong to.
Now:

* Pointmaps unproject each pixel through its centre (`depth.pinhole_xy`), and projected
  points are looked up in the pixel they fall in (`floor`, not `round`): FINAL's track
  occlusion test, the co-visibility masks, the visibility voxel lookup and FINETUNE's
  occlusion test.
* Map-anything recovers its intrinsics on the integer pixel grid: its predicted K is moved
  half a pixel into the convention above, and GT intrinsics handed to it are moved back.
* `dataset.downscale_factor`: the strided K is `cx' = (cx - 0.5) / d + 0.5` instead of
  `cx / d` (a third of a pixel off at `d = 3`).
* Resized masks and GT depth use the centre-aligned nearest resize (`cv2.INTER_NEAREST_EXACT`,
  torch `nearest-exact`), as the image's area/linear resize is; plain nearest sat half a pixel
  off it. Same in rendering guidance at `rendering_guidance_resolution_scale > 1` and in the
  depth-to-render grid match.
* The crop to the reconstruction backend's input scales K per axis by the size the image was
  actually resized to (its sides are rounded down), and the native-resolution K is derived
  through the stride, crop and cap that made its pixels.
* OursActionBench: its images were rendered with the principal point at the image centre,
  so its K now has `cx = W / 2` (`camera.json` stores `(W - 1) / 2`, the same point in the
  integer convention).
* Exported 2D tracks record the per-axis crop scale (`crop_scale_xy`).

The map-anything result cache and the shape cache are keyed anew, so cached results from
before this change are recomputed.

November 12, 2026

### Faster appearance stage, same results

Speed-ups that compute the same thing as before: bit-identical, identical, or equal up to
fp16 rounding (no larger than the difference between two runs of the previous code). On a
4-view, five-object scene (A100) the appearance block (`appearance_init_2`) takes about 90 s
instead of 190-240 s, and reconstruction compute drops by about 15%; scores (PSNR, LPIPS,
DreamSim, F-score) and peak GPU memory are unchanged.

* Visibility attention bias: the bias is no longer materialized as a dense
  `(views, 1, voxels, 5496)` tensor, which kept xformers off FlashAttention. The cross-attention
  now runs as two FlashAttention calls plus one Triton kernel that applies the same bias in
  closed form (PyTorch's memory-efficient kernel below sm80). Needs the updated
  `submodules/sam-3d-objects.patch`; an existing checkout re-applies it with
  `git -C submodules/sam-3d-objects checkout -- . && git -C submodules/sam-3d-objects apply ../sam-3d-objects.patch`.
* Attention-bias masks are built from flat arrays on the GPU, without per-voxel Python loops.
* Multi-view appearance stage: the classifier-free-guidance unconditional pass runs once per
  step instead of once per view (its input is the same for every view).
* Rendering guidance decodes the Gaussians once per ODE step at every
  `rendering_guidance_microbatch_size`; the chunks only split the renders and losses. Each
  view's camera-space Gaussians are computed once per step and shared by the visibility pass
  and the loss render. A mask's signed distance transform is computed only when
  `silhouette_sdt_weight > 0`.
* Visibility: PyTorch3D's binned rasterizer with a per-bin face budget that cannot overflow,
  and voxel-to-pixel maps from one sort instead of a scan per voxel.
* Shape inversion (one sample at a time): the decoder runs channels-last.
* Image loading: mask ids found by histogram; RGB/RGBA/gray PNGs decoded with OpenCV
  (palette masks stay on PIL).
* Diagnostic plots (attention-bias panels, guidance error maps) are drawn from thumbnails.

November 3, 2026

### Lower GPU memory for the SAM 3D backbone

* The SAM 3D backbone now loads with the Linear/Conv weights of its two generators and image
  encoders stored in fp16 (they already computed in fp16 autocast), one shared DINOv2 instead of
  four identical copies, and SAM 3D's bundled MoGe depth model kept on the CPU until it is called.
  Memory after loading drops from 13.0 to 5.9 GB; a 4-view, five-object run peaks at
  25.2 GB instead of 32.2 GB, with the same results (shape stage bit-identical, appearance stage
  within run-to-run noise) and the same run time.

November 2, 2026

### mvsynth at 1280x960

* `dataset=mvsynth` loads at 1280x960 by default (`downscale_factor: 3`); earlier runs used the
  3840x2880 masters.

October 30–31, 2026

### Native-resolution conditioning and supervision (opt-in)

Both are off by default, so results match the initial release unless they are turned on.

* `processing.native_conditioning=true`: SAM 3D's shape and appearance stages are conditioned on
  the frame's native pixels over the same field of view as the depth backend's crop, instead of
  the ~518 px crop. A no-op where nothing was cropped (`processing=ground_truth`).
* `processing.native_supervision=true`: the RGB, mask and intrinsics targets of refinement,
  FINETUNE and Stage-2 rendering guidance at the same native pixels; depth and normal terms stay
  on the crop's grid.

October 28, 2026

### Fixes

* `sam3d_objects` imports without `LIDRA_SKIP_INIT`: the patch drops an upstream startup import
  this repository does not ship.

October 27, 2026

### mvsynth dataset, denser multi-view background

* `dataset=mvsynth`: a loader for synthetic multi-view, multi-object scenes with ground-truth
  depth, intrinsics and cameras (the `mvcustom` layout plus a `gt/` folder; the scenes are not
  distributed).
* Multi-view fused background: when the views' backgrounds exceed the point budget, each view is
  now subsampled by a pixel stride, with each kept splat covering the pixels it stands for,
  instead of a random subset of single-pixel splats, which at high resolution left a
  near-transparent speckle.

October 25, 2026

### Precomputed geometry, host integration, FINETUNE pose fix

* `processing=precomputed`: read per-frame depth, intrinsics, valid mask, camera and cropped image
  from an `.npz` written by another tool, so reconstruction backends this repository does not ship
  can feed the pipeline (`processing.precomputed_geometry`, `processing.precomputed_model`).
* The shape cache is off by default (`shape_cache_dir: null`; set it to `cached_shapes` to enable).
* `run_blocks`: preprocessing and the block loop without the final export, for a host that runs
  its own.
* Fix: on multi-view runs, the pose update from FINETUNE was discarded by the following global pose
  refinement, which restarted from raw pose tokens FINETUNE had left unchanged; the tokens are now
  re-synced.

October 8, 2026

### Initial code upload

* Uploaded initial codebase after arxiv submission.
