"""The SAM3D backbone: our :class:`~genia.core.inference.Inference` wrapper, built once per process.

``core/pipeline.py`` gets it from :func:`sam3d_inference`.
"""

from typing import Any, Optional

from genia.core.paths import sam3d_weights
from genia.core.utils.timing import exclude_model_load


def config_path() -> str:
    """The pipeline config our method builds the backbone from."""
    return str(sam3d_weights() / "pipeline.yaml")

_inference: Optional[Any] = None
_spec: Optional[tuple] = None


def sam3d_inference(config_file: str, compile: bool = False) -> Any:
    """The :class:`Inference` wrapper for ``config_file``, built once per process.

    One process runs the pipeline once, so this changes nothing on the CLI path; it
    exists for a host that calls ``run()`` repeatedly (e.g. a warm demo worker),
    where rebuilding the backbone costs ~60s of checkpoint reads per run.

    A single slot: the config path is composed from one constant at each call site, so a
    second one means something is wrong, and a silent second backbone would double VRAM
    with no way to release it.
    """
    global _inference, _spec
    spec = (str(config_file), bool(compile))
    if _inference is not None and spec != _spec:
        raise RuntimeError(
            f"Cannot build the SAM3D backbone for {spec} — {_spec} is already loaded"
        )
    if _inference is None:
        from genia.core.inference import Inference

        with exclude_model_load("SAM3D backbone"):
            _inference = Inference(str(config_file), compile=compile)
        _spec = spec
    return _inference


