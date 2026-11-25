---
# Copyright (c) Meta Platforms, Inc. and affiliates.
title: GenIA
emoji: 🧸
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 5.49.0
python_version: 3.12
app_file: app.py
# The first start downloads the model weights (~21 GB).
startup_duration_timeout: 1h
pinned: false
license: cc-by-nc-4.0
short_description: Generative 3D reconstruction from one or a few photos
---

# GenIA: Generative Reconstruction with Test-Time Input Alignment

Reconstruct the objects in one photo, or in a few photos of one static scene: what the
photos see is reconstructed, what they don't is generated. Pick an example (it plays a
recorded run, no GPU needed) or upload your own photos and name the objects to reconstruct.

[Project page](https://facebookresearch.github.io/GenIA/) ·
[Paper](https://arxiv.org/abs/2610.12388) · [Code](https://github.com/facebookresearch/GenIA)

A live run takes a few minutes of GPU time, which counts against your daily ZeroGPU quota.
