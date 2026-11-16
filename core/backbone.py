# Copyright (c) Meta Platforms, Inc. and affiliates.

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
            _slim(_inference._pipeline)
        _spec = spec
    return _inference


class _OnDemandGPU:
    """SAM3D's MoGe depth model, kept on the CPU until a call needs it (GenIA passes its own
    pointmap, so normally never)."""

    def __init__(self, depth_model):            # sam3d_objects' DepthModel: the network is .model
        self._depth_model = depth_model
        depth_model.model.cpu()
        self._on_gpu = False

    def __call__(self, *args, **kwargs):
        if not self._on_gpu:
            self._depth_model.model.to(self._depth_model.device)
            self._on_gpu = True
        return self._depth_model(*args, **kwargs)


def _slim(pipeline) -> None:
    """Same outputs, less memory (13.0 -> 5.9 GB after loading): the weights SAM3D's fp16
    autocast casts anyway (Linear/ConvNd of both generators and the image encoders) stored in
    fp16, so the matmuls see the same numbers; the four DINOv2 copies in the two image encoders
    (one network, checked equal) as one; MoGe off the GPU until used."""
    import gc

    import torch

    convs = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)
    encoders = list(pipeline.condition_embedders.values())
    for module in [pipeline.models["ss_generator"], pipeline.models["slat_generator"], *encoders]:
        for m in module.modules():
            if isinstance(m, convs):
                for p in m.parameters(recurse=False):
                    if p.dtype == torch.float32:
                        p.data = p.data.half()
    slots = [(e, i) for e in encoders for i, s in enumerate(e.module_list) if hasattr(s, "backbone")]
    ref = slots[0][0].module_list[slots[0][1]].backbone
    ref_state = ref.state_dict()
    for e, i in slots[1:]:
        state = e.module_list[i].backbone.state_dict()
        if state.keys() != ref_state.keys() or not all(torch.equal(v, ref_state[k]) for k, v in state.items()):
            raise RuntimeError("SAM3D image encoders' DINOv2 copies differ; cannot share them")
        e.module_list[i].backbone = ref
        del state                    # else the last replaced copy stays alive
    pipeline.depth_model = _OnDemandGPU(pipeline.depth_model)
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[sam3d] slim backbone: {torch.cuda.memory_allocated() / 2**20:.0f} MB allocated")
