<!-- Copyright (c) Meta Platforms, Inc. and affiliates. -->

# Third-party components

This repository includes third-party code as git submodules.
Each keeps its own licence; nothing here is relicensed by this project's `LICENSE`
(CC BY-NC 4.0, noncommercial). **Check this table before any commercial use.**

## Included in this repository

| Path | Upstream | Licence |
|---|---|---|
| `submodules/sam-3d-objects` (git submodule at `f91db41`; `submodules/sam-3d-objects.patch` holds our modifications) | [facebookresearch/sam-3d-objects](https://github.com/facebookresearch/sam-3d-objects) | **SAM License** (Meta), including its Acceptable Use Policy |
| `core/inference.py` | Derived from SAM 3D Objects' inference wrapper | **SAM License** (Meta) |
| `submodules/sam-3d-objects/sam3d_objects/model/backbone/tdfy_dit/renderers/gaussian_render.py`, `.../representations/gaussian/general_utils.py` | [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting), via SAM 3D Objects | **Gaussian-Splatting License** (Inria): non-commercial research and evaluation use only |
| `submodules/map-anything` (git submodule) | [facebookresearch/map-anything](https://github.com/facebookresearch/map-anything) | Apache-2.0 |
| `submodules/actionmesh` (git submodule: the shapes of dynamic scenes) | [facebookresearch/ActionMesh](https://github.com/facebookresearch/ActionMesh) | **FAIR Noncommercial Research License** — noncommercial research only |
| `submodules/actionmesh/third_party/TripoSG` (nested submodule) | [VAST-AI-Research/TripoSG](https://github.com/VAST-AI-Research/TripoSG) | MIT |

The multi-view entropy weighting in `core/entropy.py` follows the formulation of
[MV-SAM3D](https://github.com/devinli123/MV-SAM3D) (a SAM 3D Objects derivative,
SAM License).

## Installed dependencies

Installed by `install.sh`, not redistributed here; each under its own licence, e.g.
[MoGe](https://github.com/microsoft/MoGe) (MIT),
[gsplat](https://github.com/nerfstudio-project/gsplat) (Apache-2.0),
[PyTorch3D](https://github.com/facebookresearch/pytorch3d) (BSD),
[nvdiffrast](https://github.com/NVlabs/nvdiffrast) (NVIDIA Source Code License,
non-commercial).

## Model weights

Weights are downloaded at setup time and are **not** covered by this repository's
licence. The SAM 3D Objects checkpoint requires accepting its licence on Hugging Face
before download; map-anything, MoGe and the ActionMesh / TripoSG / DINOv2 / RMBG weights
fetched by ActionMesh carry their own terms.

## Demo inputs

Every scene under `demo/data/` has a `LICENSE.md` with its source and terms.

| Scene | Source | Licence |
|---|---|---|
| `image/lab_duo`, `image/lab_turtle`, `image/office_buddies`, `image/stutty`, `multiview/lab_duck`, `multiview/pablo` | Captured by the authors | **CC BY-NC 4.0**, as this project's code |
| `dynamic/camel` | [DAVIS dataset](https://davischallenge.org/) (the `camel` sequence, every second frame of frames 0-30, downscaled to 640x360) | **CC BY-NC 4.0** per the DAVIS website (the original repository states BSD) |
| `dynamic/dinosaur` | [ActionMesh Hugging Face Space](https://huggingface.co/spaces/facebook/ActionMesh) demo assets (original source undocumented) | **FAIR Noncommercial Research License** (Meta) |
| `multiview/stuffed_toy` | [MV-SAM3D repository](https://github.com/devinli123/MV-SAM3D) example images (downscaled to 1024x768) | **SAM License** (Meta), the repository's licence |
| `image/sloth` | [4DPM](https://makezur.github.io/4DPM/), provided to us directly by a 4DPM author | **No public licence**; property of the 4DPM authors, shared with us for this project |
