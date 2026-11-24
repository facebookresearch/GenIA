---
# Copyright (c) Meta Platforms, Inc. and affiliates.
title: GenIA
emoji: 🧸
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
# The first start fetches the model weights (several GB) and loads the models.
startup_duration_timeout: 1h
# A GPU is required; the app adapts to 24 GB (low-VRAM mode) and keeps more loaded from 40 GB.
suggested_hardware: l40sx1
pinned: false
---

# GenIA: Generative Reconstruction with Test-Time Input Alignment

Upload an image or a few views of a static scene, mask its objects with SAM3, and
reconstruct them in 3D with GenIA. The example scenes play a recorded run at once;
**Run live** reconstructs them on the Space's GPU.
