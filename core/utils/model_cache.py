# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Singleton cache for lazy-loaded models used during optimization.

Keeps runtime model state (perceptual loss network, reconstruction
backends) out of the configuration, so that config dataclasses stay
serialisable and copyable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch

from genia.core.paths import GENIA_SUBMODULES

from .timing import exclude_model_load

if TYPE_CHECKING:
    from .perceptual_loss import PerceptualLoss


class ModelCache:
    """Lazy-loaded, singleton model cache for optimization losses."""

    _instance: Optional["ModelCache"] = None

    def __init__(self) -> None:
        self._perceptual_model: Optional["PerceptualLoss"] = None
        self._map_anything_model: Optional[Any] = None
        self._map_anything_model_id: str = "facebook/map-anything"
        self._moge_model: Optional[Any] = None

    #: Reconstruction backends, by the attribute holding each one. Only one is kept
    #: resident at a time: loading one evicts the other.
    _RECON_SLOTS = ("_map_anything_model", "_moge_model")

    def release_reconstruction_models(self, keep: Optional[str] = None) -> None:
        """Free every resident reconstruction backend except ``keep``'s slot.

        Lazy by construction: it frees only what is actually loaded, so selecting a
        backend whose depth comes from cache does not drop the other one's weights for
        a forward pass that never happens.

        Called once the Sequence is built (``preprocessing.run_preprocessing``), because
        weights left resident cost more than VRAM: ``reset_peak_memory_stats`` rebases
        each block's peak to what is LIVE, so every later block would be charged for
        preprocessing's weights -- but only on datasets whose depth is predicted. Timing
        excludes preprocessing by dropping its block; memory cannot be excluded that way
        because it persists, so it has to be released. A later access reloads through the
        lazy getter.
        """
        freed = False
        for slot in self._RECON_SLOTS:
            if slot == keep or getattr(self, slot) is None:
                continue
            setattr(self, slot, None)
            freed = True
        if freed:
            torch.cuda.empty_cache()

    @classmethod
    def get(cls) -> "ModelCache":
        """Return the global singleton, creating it on first call."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Perceptual model: LPIPS (VGG)
    # ------------------------------------------------------------------

    @property
    def perceptual_model(self) -> "PerceptualLoss":
        if self._perceptual_model is None:
            self._perceptual_model = self._load_perceptual_model()
        return self._perceptual_model

    @staticmethod
    def _load_perceptual_model() -> "PerceptualLoss":
        with exclude_model_load("VGG19 perceptual loss"):
            from .perceptual_loss import PerceptualLoss

            model = PerceptualLoss()
            model = model.to(torch.device("cuda"))
            model.eval()
            return model

    #: Which setter declares each backend's checkpoint (backends with no id are absent).
    _RECON_MODEL_ID_SETTERS = {
        "map_anything": ("set_map_anything_model_id", "map_anything_model_id"),
    }

    def set_recon_model_id_from(self, processing) -> None:
        """Declare the checkpoint for whichever backend ``processing`` selects.

        A no-op for the backends that have no id (moge / null).  Must run before the
        first access, because the id also keys the on-disk depth cache.
        """
        entry = self._RECON_MODEL_ID_SETTERS.get(processing.reconstruction_model)
        if entry is not None:
            setter, key = entry
            getattr(self, setter)(getattr(processing, key))

    # ------------------------------------------------------------------
    # Map-Anything (multi-view depth + pose estimation)
    # ------------------------------------------------------------------

    def set_map_anything_model_id(self, model_id: str) -> None:
        """Set the HuggingFace model ID before first access.

        Re-setting the SAME id after the model is loaded is a no-op rather than an
        error: a host that calls ``main()`` repeatedly in one process runs this line
        once per run, and asking again for the model already resident is not a change.
        A DIFFERENT id still raises — the loaded model would not be the one asked for.
        """
        if self._map_anything_model is not None and model_id != self._map_anything_model_id:
            raise RuntimeError(
                f"Cannot change map-anything model ID to {model_id!r} — "
                f"{self._map_anything_model_id!r} is already loaded"
            )
        self._map_anything_model_id = model_id

    @property
    def map_anything_model_id(self) -> str:
        """Which checkpoint `map_anything_model` would load. Read by the depth
        cache's key, so a model-id change cannot hit another model's cached depth."""
        return self._map_anything_model_id

    @property
    def map_anything_model(self) -> Any:
        # Deliberately NOT wrapped in exclude_model_load, unlike the loss nets above:
        # this one loads inside depth preprocessing, which runs before the block timer
        # starts. Excluding it here would make any recorded PREPROCESSING seconds stop
        # meaning "what the depth pass cost".
        if self._map_anything_model is None:
            self._map_anything_model = self._load_map_anything_model()
        return self._map_anything_model

    def _load_map_anything_model(self) -> Any:
        # Evict the other backend before allocating this one, not when the
        # config was read: a cache hit never gets here, so it never pays.
        self.release_reconstruction_models(keep="_map_anything_model")

        import sys

        map_anything_path = GENIA_SUBMODULES / "map-anything"
        if str(map_anything_path) not in sys.path:
            sys.path.insert(0, str(map_anything_path))

        try:
            from mapanything.models.mapanything import MapAnything
        except ImportError:
            raise ImportError(
                "map-anything is required when processing.reconstruction_model='map_anything'. "
                "Install it with: pip install -e submodules/map-anything"
            )

        print(f"Loading map-anything model: {self._map_anything_model_id}")
        model = MapAnything.from_pretrained(self._map_anything_model_id)
        model = model.to(torch.device("cuda"))
        model.eval()
        return model

    # ------------------------------------------------------------------
    # MoGe (per-frame monocular depth + intrinsics, no poses)
    # ------------------------------------------------------------------

    #: The checkpoint the SAM3D pipeline bundles as its `depth_model`
    #: (its `pipeline.yaml`).
    _MOGE_MODEL_ID = "Ruicheng/moge-vitl"

    @property
    def moge_model(self) -> Any:
        # Not wrapped in exclude_model_load, for the same reason as map-anything above.
        if self._moge_model is None:
            self._moge_model = self._load_moge_model()
        return self._moge_model

    def _load_moge_model(self) -> Any:
        # Evict the other backends before allocating, not when the config was read.
        self.release_reconstruction_models(keep="_moge_model")

        from moge.model.v1 import MoGeModel

        print(f"Loading MoGe model: {self._MOGE_MODEL_ID}")
        model = MoGeModel.from_pretrained(self._MOGE_MODEL_ID)
        return model.to(torch.device("cuda")).eval()
