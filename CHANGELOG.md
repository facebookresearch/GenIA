<!-- Copyright (c) Meta Platforms, Inc. and affiliates. -->

Changes since the initial release (October 8, 2026), newest first.

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
