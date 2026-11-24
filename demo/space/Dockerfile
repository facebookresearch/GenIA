# Copyright (c) Meta Platforms, Inc. and affiliates.

# The GenIA demo app (demo/app.py) as a Hugging Face Docker Space on a GPU.
#
# Built from the tree demo/space/build_context.py assembles (genia's tracked files with
# their submodules checked out and SAM 3D Objects' patch applied, this folder's files at
# its root), never from the repository itself:
#
#   python demo/space/build_context.py /tmp/genia-space
#   docker build -t genia-space /tmp/genia-space
#   docker run --gpus all -p 7860:7860 -e HF_TOKEN=... genia-space
#
# install.sh builds the environment as for a local install; the CUDA extensions are
# compiled here, with no GPU, for every architecture the Space may run on.
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git libgl1 libglib2.0-0 xz-utils \
    && rm -rf /var/lib/apt/lists/*

# Node >= 22 for the 3D view's bundle (install.sh builds submodules/3DView with it).
ARG NODE_VERSION=22.13.1
RUN curl -fsSL https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz \
        | tar -xJ -C /opt \
    && ln -s /opt/node-v${NODE_VERSION}-linux-x64/bin/node /opt/node-v${NODE_VERSION}-linux-x64/bin/npm \
        /opt/node-v${NODE_VERSION}-linux-x64/bin/npx /usr/local/bin/

# Hugging Face runs a Space's container as uid 1000; everything below is that user's.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user
WORKDIR /home/user

RUN curl -fsSL -o /tmp/miniforge.sh \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash /tmp/miniforge.sh -b -p $HOME/conda && rm /tmp/miniforge.sh
ENV PATH=$HOME/conda/bin:$PATH

COPY --chown=user . genia
WORKDIR /home/user/genia
# No GPU at build time: the extensions are built for the listed architectures (A100, A10G,
# L4/L40S, H100), and pytorch3d is told to build its CUDA kernels anyway.
ENV GENIA_SKIP_SUBMODULES=1 \
    GENIA_CUDA_ROOT=/usr/local \
    GENIA_TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    FORCE_CUDA=1 \
    MAX_JOBS=4
RUN bash install.sh && conda clean -afy && rm -rf $HOME/.cache/pip

ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860
EXPOSE 7860
CMD ["bash", "start.sh"]
