# Copyright (c) Meta Platforms, Inc. and affiliates.

"""GenIA: generative reconstruction with test-time input alignment -- our method.

The package holds the block pipeline (``core/pipeline.py``) and its block runners
(``core/main.py``), the SAM3D inference wrapper with our in-ODE procedures (guidance,
velocity averaging, visibility attention), token-space shape/pose initialisation, GT-shape
injection and FINETUNE, plus the shared library under ``core/utils/`` (data loading, the
base config, pipeline state, rendering, refinement, evaluation and the ``final/`` writer).
"""
