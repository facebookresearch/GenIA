#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

# Build the `genia` conda environment: Python 3.12, torch 2.8 + CUDA 12.8, with everything the
# pipeline, the demos and the ActionMesh runner (core/actionmesh_video.py) need.
#
#   bash install.sh && conda activate genia
#
# The CUDA extensions are compiled against a LOCAL CUDA toolkit matching the torch wheels,
# never a conda one. Requirements: conda, git, and a system CUDA 12.8 toolkit.
#
# Knobs:
#   GENIA_CUDA_ROOT             parent of the cuda-<ver> dirs              (default /usr/local)
#   GENIA_CUDA_VERSION          toolkit version                            (default 12.8)
#   GENIA_TORCH_CUDA_ARCH_LIST  target GPU arch(s)                         (default 8.0 = A100)
#                               e.g. 8.6 (A6000/3090), 8.9 (L40S/4090), 9.0 (H100)
#   GENIA_ENV_NAME              environment name                           (default genia)
#   MAX_JOBS                    parallel compile jobs for the extensions   (default 1)
#   GENIA_SKIP_SUBMODULES=1     the submodules are already checked out and SAM 3D Objects'
#                               patch applied (e.g. a Docker build from demo/space/'s tree,
#                               which has no .git): neither is fetched nor patched here
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
ENV_NAME=${GENIA_ENV_NAME:-genia}
export MAX_JOBS=${MAX_JOBS:-1}
export TORCH_CUDA_ARCH_LIST="${GENIA_TORCH_CUDA_ARCH_LIST:-8.0}"
eval "$(conda shell.bash hook)"

use_cuda() {    # use_cuda <version>: build against <GENIA_CUDA_ROOT>/cuda-<version>
    export CUDA_HOME=${GENIA_CUDA_ROOT:-/usr/local}/cuda-$1
    export CUDA_PATH=$CUDA_HOME
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH
    [ -d "$CUDA_HOME" ] || { echo "No CUDA toolkit at $CUDA_HOME: set GENIA_CUDA_ROOT / GENIA_CUDA_VERSION."; exit 1; }
}

# SAM 3D Objects (the backbone) + our patch, map-anything, and ActionMesh with its
# TripoSG submodule (pinned with an SSH URL: fetched over HTTPS).
if [ "${GENIA_SKIP_SUBMODULES:-0}" != 1 ]; then
    git submodule update --init submodules/sam-3d-objects submodules/map-anything submodules/sam3 \
        submodules/3DView
    if git -C submodules/sam-3d-objects apply --reverse --check ../sam-3d-objects.patch 2>/dev/null; then
        echo "submodules/sam-3d-objects: patch already applied"
    else
        git -C submodules/sam-3d-objects apply ../sam-3d-objects.patch
    fi
    git -c url."https://github.com/".insteadOf="git@github.com:" \
        submodule update --init --recursive submodules/actionmesh
fi
use_cuda "${GENIA_CUDA_VERSION:-12.8}"
conda env create -n "$ENV_NAME" -f environment.yml
conda activate "$ENV_NAME"

TORCH=(torch==2.8.0+cu128 torchvision==0.23.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128)
pip install "${TORCH[@]}"
# torch pinned on the same line, so a requirement that wanted another torch fails here
# instead of silently replacing the build the extensions are compiled against.
pip install -r requirements.txt "${TORCH[@]}"

# CUDA extensions built against the env's torch (--no-build-isolation). --no-deps on the
# prebuilt wheels so pip cannot replace the pinned torch.
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9"
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@v1.5.3
pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git@v0.4.0
pip install --no-build-isolation --no-deps git+https://github.com/rahul-goel/fused-ssim.git
pip install --no-deps xformers==0.0.32.post2   # the build for torch 2.8
pip install --no-deps spconv-cu126==2.3.8 cumm-cu126==0.7.11
pip install --no-deps git+https://github.com/microsoft/MoGe.git@07444410f1e33f402353b99d6ccd26bd31e469e8
pip install --no-deps git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183

# map-anything (the default depth + camera backend) and ActionMesh. TripoSG is not
# installed: core/actionmesh_video.py puts it on sys.path. genia.core.download_weights
# fetches its weights (TripoSG, DINOv2, RMBG, ActionMesh).
pip install --no-deps -e submodules/map-anything
# uniception's metadata pulls torchaudio (a second torch) and rerun-sdk (numpy>=2); the
# modules map-anything uses import neither.
pip install --no-deps uniception==0.1.6
pip install --no-deps -e submodules/actionmesh
# SAM3: the demo app's masks (demo/app.py); its dependencies are in requirements.txt.
pip install --no-deps -e submodules/sam3

# Register the `genia.core` import path (dependencies are already installed above).
pip install --no-deps -e .

# The demo app's 3D view (3DView, submodules/3DView): a Three.js bundle built with Node >= 22.
VIEWER_BUILD="cd submodules/3DView && npm ci --no-audit --no-fund && npm run build -w @3dview/core && npm run build -w 3dview-demo"
if command -v npm >/dev/null && [ "$(node -p 'process.versions.node.split(".")[0]')" -ge 22 ]; then
    (eval "$VIEWER_BUILD")
else
    echo "NOTE: no Node >= 22, so the demo app's 3D view is not built. Later: ($VIEWER_BUILD)"
fi

echo
echo "Done. Next: conda activate $ENV_NAME && python -m genia.core.download_weights"
