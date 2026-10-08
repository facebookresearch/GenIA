# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Filesystem anchors: the checkout root, the model weights, the results tree and the
vendored submodules. ``GENIA_SAM3D_WEIGHTS`` / ``GENIA_RESULTS`` override the defaults.

Model weights live in the Hugging Face and torch hub caches, which
``python -m genia.core.download_weights`` fills."""
import os
from pathlib import Path

#: The genia checkout (this file is ``core/paths.py``).
GENIA_ROOT = Path(__file__).resolve().parents[1]

#: The SAM 3D Objects checkpoint repository; its files sit in ``checkpoints/``.
SAM3D_REPO = "facebook/sam-3d-objects"


def sam3d_weights() -> Path:
    """The SAM 3D Objects checkpoint folder (``pipeline.yaml`` and the files it names):
    ``GENIA_SAM3D_WEIGHTS`` if set, else its snapshot in the Hugging Face cache."""
    if "GENIA_SAM3D_WEIGHTS" in os.environ:
        return Path(os.environ["GENIA_SAM3D_WEIGHTS"])
    from huggingface_hub import try_to_load_from_cache

    config = try_to_load_from_cache(SAM3D_REPO, "checkpoints/pipeline.yaml")
    if not isinstance(config, str):
        raise FileNotFoundError(f"{SAM3D_REPO} is not in the Hugging Face cache: run "
                                "`python -m genia.core.download_weights`")
    return Path(config).parent

#: Root of the ``{experiment}/{dataset}/{scene}/{timestamp}/`` run tree.
RESULTS = Path(os.environ.get("GENIA_RESULTS", GENIA_ROOT / "results"))

#: Third-party code: git submodules (map-anything, ActionMesh, SAM 3D Objects).
GENIA_SUBMODULES = GENIA_ROOT / "submodules"

#: The SAM 3D Objects checkout (``sam3d_objects`` package, patched by install.sh with
#: ``submodules/sam-3d-objects.patch``). APPEND it to ``sys.path``, never insert(0): an
#: already-importable ``sam3d_objects`` must keep winning.
SAM3D_OBJECTS_ROOT = GENIA_SUBMODULES / "sam-3d-objects"
