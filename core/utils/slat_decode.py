"""Decoding SAM3D latents (shape tokens and SLATs) into geometry and Gaussians.

Representation support, not a pipeline step: any state holding SAM3D latents is decoded
through here, whichever code path produced them. FINAL, evaluation, refinement and the pipeline state all
read decoded outputs through these helpers, including the LoRA-adapted decoder a FINETUNE
pass leaves behind.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import torch

from genia.core.utils.quaternion_ops import quaternion_invert, quaternion_multiply

if TYPE_CHECKING:
    from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp


def decode_shape_to_occ(
    pipeline_or_inference: Any,
    shape_latent: torch.Tensor,
) -> torch.Tensor:
    """Decode a shape latent into the dense 64³ sigmoid occupancy grid.

    Sibling of :func:`decode_shape_to_coords` that returns the continuous
    occupancy grid (``sigmoid(logits)``) instead of the binarized
    occupied-voxel coordinates.  Its caller, the coarse-shape wireframe
    preview (:func:`eval_assets_export.export_coarse_shape_wireframe`), extracts
    an iso-surface from it with :func:`mesh_rendering.diffmc_mesh`.

    Parameters
    ----------
    pipeline_or_inference : Pipeline or Inference
        Either the SAM3D pipeline directly (has ``.models``) or the
        Inference wrapper (has ``._pipeline``).
    shape_latent : torch.Tensor
        Shape latent of shape ``(1, 4096, 8)``.

    Returns
    -------
    torch.Tensor
        Sigmoid occupancy ``(64, 64, 64)`` in ``[0, 1]`` on the latent's
        device.
    """
    if hasattr(pipeline_or_inference, '_pipeline'):
        pipeline = pipeline_or_inference._pipeline
    else:
        pipeline = pipeline_or_inference
    ss_decoder = pipeline.models["ss_decoder"]
    with torch.no_grad():
        shape_cube = (
            shape_latent
            .permute(0, 2, 1)
            .contiguous()
            .view(1, 8, 16, 16, 16)
        )
        logits = ss_decoder(shape_cube.to(dtype=ss_decoder.dtype))
        return torch.sigmoid(logits[0, 0].float())


def decode_shape_to_coords(
    pipeline_or_inference: Any,
    shape_latent: torch.Tensor,
) -> torch.Tensor:
    """Decode a shape latent into sparse voxel coordinates.

    Runs the ``ss_decoder`` (Sparse Structure Decoder) to produce the
    64³ occupancy grid and extracts occupied-voxel coordinates.

    Parameters
    ----------
    pipeline_or_inference : Pipeline or Inference
        Either the SAM3D pipeline directly (has ``.models``) or the
        Inference wrapper (has ``._pipeline``).
    shape_latent : torch.Tensor
        Shape latent of shape ``(1, 4096, 8)``.

    Returns
    -------
    torch.Tensor
        Integer coordinates ``(N, 4)`` where columns are
        ``[batch_idx, x, y, z]`` with values in ``[0, 63]``.
    """
    if hasattr(pipeline_or_inference, '_pipeline'):
        pipeline = pipeline_or_inference._pipeline
    else:
        pipeline = pipeline_or_inference
    ss_decoder = pipeline.models["ss_decoder"]
    with torch.no_grad():
        shape_cube = (
            shape_latent
            .permute(0, 2, 1)
            .contiguous()
            .view(1, 8, 16, 16, 16)
        )
        ss = ss_decoder(shape_cube.to(dtype=ss_decoder.dtype))
        coords = torch.argwhere(ss > 0)[:, [0, 2, 3, 4]].int()
    return coords


def redecode_slat(
    pipeline: Any,
    slat: "sp.SparseTensor",
    formats: List[str] = ["gaussian", "mesh"],
) -> Dict[str, Any]:
    """
    Re-run the decoder forward pass using saved SLAT tokens.

    Parameters
    ----------
    pipeline : Pipeline
        The SAM3D pipeline with decode_slat method.
    slat : sp.SparseTensor
        SLAT tokens to decode.
    formats : list of str, optional
        Output formats to decode. Default: ["gaussian", "mesh"].

    Returns
    -------
    dict
        Decoded outputs with keys for each requested format.

    Examples
    --------
    >>> outputs = redecode_slat(pipeline, slat, formats=["gaussian"])
    >>> gs = outputs["gaussian"][0]
    >>> gs.get_xyz.shape
    torch.Size([10000, 3])
    """
    print(f"Re-decoding SLAT tokens to formats: {formats}")
    print(f"  SLAT features shape: {slat.feats.shape}")
    print(f"  SLAT coords shape: {slat.coords.shape}")

    with torch.no_grad():
        decoded_outputs = pipeline.decode_slat(slat, formats=formats)

    # Print info about decoded outputs
    if "gaussian" in decoded_outputs:
        gs = decoded_outputs["gaussian"][0]
        print(f"  Decoded Gaussians: {gs.get_xyz.shape[0]} points")
        print(f"    xyz range: [{gs.get_xyz.min().item():.3f}, {gs.get_xyz.max().item():.3f}]")

    if "mesh" in decoded_outputs:
        mesh = decoded_outputs["mesh"][0]
        print(f"  Decoded Mesh: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")

    return decoded_outputs


def make_scene(*outputs, in_place=False):
    # Local, not module-level: a host process may put its own sam3d_objects fork on
    # sys.path before first use, and this import must resolve to that one.
    from sam3d_objects.utils.visualization import SceneVisualizer

    if not in_place:
        outputs = [copy.deepcopy(output) for output in outputs]

    all_outs = []
    minimum_kernel_size = float("inf")
    for output in outputs:
        # move gaussians to scene frame of reference
        PC = SceneVisualizer.object_pointcloud(
            points_local=output["gaussian"][0].get_xyz.unsqueeze(0),
            quat_l2c=output["rotation"],
            trans_l2c=output["translation"],
            scale_l2c=output["scale"],
        )
        output["gaussian"][0].from_xyz(PC.points_list()[0])
        # must ... ROTATE
        output["gaussian"][0].from_rotation(
            quaternion_multiply(
                quaternion_invert(output["rotation"]),
                output["gaussian"][0].get_rotation,
            )
        )
        scale = output["gaussian"][0].get_scaling
        adjusted_scale = scale * output["scale"]
        # Use the mean of the scale components for the minimum kernel size
        scale_mean = output["scale"][0].mean().item()
        output["gaussian"][0].mininum_kernel_size *= scale_mean
        adjusted_scale = torch.maximum(
            adjusted_scale,
            torch.tensor(
                output["gaussian"][0].mininum_kernel_size * 1.1,
                device=adjusted_scale.device,
            ),
        )
        output["gaussian"][0].from_scaling(adjusted_scale)
        minimum_kernel_size = min(
            minimum_kernel_size,
            output["gaussian"][0].mininum_kernel_size,
        )
        all_outs.append(output)

    # merge gaussians
    scene_gs = all_outs[0]["gaussian"][0]
    # Re-encode the scene object's scaling under the scene-wide (min) kernel
    # size: raw _scaling only decodes correctly under the kernel size it was
    # encoded with.  No-op (skipped for bit-exactness) when the first object
    # already owns the min.
    if scene_gs.mininum_kernel_size != minimum_kernel_size:
        _scene_scaling = scene_gs.get_scaling
        scene_gs.mininum_kernel_size = minimum_kernel_size
        scene_gs.from_scaling(_scene_scaling)
    for out in all_outs[1:]:
        out_gs = out["gaussian"][0]
        # Containers may differ (aabb / scale_bias / opacity_bias / kernel
        # size / scaling activation), so raw internal tensors are NOT
        # interchangeable: re-encode this object's decoded values into the
        # scene container before concatenating raw storage.
        xyz_b = (out_gs.get_xyz - scene_gs.aabb[None, :3]) / scene_gs.aabb[None, 3:]
        scaling_b = scene_gs.inverse_scaling_activation(
            torch.sqrt(
                torch.square(out_gs.get_scaling)
                - scene_gs.mininum_kernel_size**2
            )
        ) - scene_gs.scale_bias
        # Raw-space bias shift, NOT from_rotation(get_rotation) /
        # from_opacity(get_opacity): exact, and the opacity round trip would
        # produce inf logits for opacities saturating to 1.0 in fp32.
        rotation_b = (
            out_gs._rotation + out_gs.rots_bias[None, :] - scene_gs.rots_bias[None, :]
        )
        opacity_b = out_gs._opacity + out_gs.opacity_bias - scene_gs.opacity_bias
        # Row count of the scene BEFORE this concat — the SH-rest zero-pad
        # below must match the pre-merge Gaussian count, not the merged one.
        n_a = scene_gs._features_dc.shape[0]
        scene_gs._xyz = torch.cat([scene_gs._xyz, xyz_b], dim=0)
        scene_gs._features_dc = torch.cat(
            [scene_gs._features_dc, out_gs._features_dc], dim=0
        )
        scene_gs._scaling = torch.cat([scene_gs._scaling, scaling_b], dim=0)
        scene_gs._rotation = torch.cat([scene_gs._rotation, rotation_b], dim=0)
        scene_gs._opacity = torch.cat([scene_gs._opacity, opacity_b], dim=0)

        # Merge SH rest (higher-order bands), padding if needed
        rest_a = scene_gs._features_rest
        rest_b = out_gs._features_rest
        if rest_a is not None or rest_b is not None:
            device = scene_gs._features_dc.device
            n_b = out_gs._features_dc.shape[0]
            if rest_a is None:
                rest_a = torch.zeros(n_a, rest_b.shape[1], 3, device=device)
            if rest_b is None:
                rest_b = torch.zeros(n_b, rest_a.shape[1], 3, device=device)
            # Pad to max SH bands if different degrees
            if rest_a.shape[1] != rest_b.shape[1]:
                max_bands = max(rest_a.shape[1], rest_b.shape[1])
                if rest_a.shape[1] < max_bands:
                    rest_a = torch.cat([rest_a, torch.zeros(n_a, max_bands - rest_a.shape[1], 3, device=device)], dim=1)
                if rest_b.shape[1] < max_bands:
                    rest_b = torch.cat([rest_b, torch.zeros(n_b, max_bands - rest_b.shape[1], 3, device=device)], dim=1)
            scene_gs._features_rest = torch.cat([rest_a, rest_b], dim=0)
            degree = int((scene_gs._features_rest.shape[1] + 1) ** 0.5) - 1
            scene_gs.sh_degree = degree
            scene_gs.active_sh_degree = degree

    return scene_gs


# Decoder-output float tensor attributes cast back to fp32 after a bf16
# autocast decode, so downstream gsplat + losses stay full-precision.
# Shared with the rendering-guidance decode sites.
_GAUSSIAN_FP_ATTRS = ("_xyz", "_features_dc", "_features_rest",
                      "_scaling", "_rotation", "_opacity")


def _cast_attrs_float32(obj: Any, attrs: tuple) -> Any:
    """Cast the named floating-point tensor attributes of ``obj`` to fp32
    in-place (differentiable ``.float()`` — gradients still flow back into
    the bf16 decoder graph).  Non-tensor / integer / absent attrs skipped."""
    for a in attrs:
        t = getattr(obj, a, None)
        if torch.is_tensor(t) and t.is_floating_point() and t.dtype != torch.float32:
            setattr(obj, a, t.float())
    return obj


def decode_tokens(
    decoder: torch.nn.Module,
    opt_feats: torch.Tensor,
    coords: torch.Tensor,
    autocast_bf16: bool = False,
) -> Any:
    """Decode SLAT tokens to Gaussian via the decoder (with gradients).

    Parameters
    ----------
    decoder : SLatGaussianDecoder
        Frozen (or LoRA-adapted) decoder model.
    opt_feats : torch.Tensor
        Optimizable features, shape (N, C), requires_grad=True.
    coords : torch.Tensor
        Fixed sparse coordinates, shape (N, 4).
    autocast_bf16 : bool
        Run the decoder forward under ``torch.autocast(bfloat16)`` — decoder
        activations are stored in bf16 for backward (~halves the dominant
        peak-memory term).  Outputs are cast back to fp32 so downstream
        gsplat rendering + losses keep full precision.

    Returns
    -------
    Gaussian
        Decoded Gaussian object (single batch item).
    """
    from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp

    slat = sp.SparseTensor(feats=opt_feats, coords=coords)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_bf16):
        gaussians_list = decoder(slat)
    g = gaussians_list[0]
    if autocast_bf16:
        _cast_attrs_float32(g, _GAUSSIAN_FP_ATTRS)
    return g


# ────────────────────────────────────────────────────────────────────────────
# Wrapped-decoder cache (step 13)
# ────────────────────────────────────────────────────────────────────────────
# Both ``setup_shared_lora_decoder`` (FINETUNE-block) and ``decode_with_lora``
# (FINAL-block, called once per object via ``PipelineState.set_canonical_slat``)
# build a deepcopy + LoRA-wrapped decoder.  In a multi-object scene the cost
# pays N times per process; in a single-object scene it pays twice (FINETUNE
# + FINAL).  This cache shares one wrapped decoder per (base_id, rank, alpha,
# init, use_dora, rs_scaling, lora_targets) tuple within a process.
#
# Bit-identity: the un-cached ``apply_lora_to_decoder`` consumes RNG via
# ``_init_lora_A`` once per matching layer.  Skipping that on cache hits
# would drift downstream RNG state (LoRAAdapter init, random_background
# sampling, …).  To stay bit-identical, cache hits "replay" the same draws
# by calling ``_init_lora_A`` again and discarding the result.
#
# Cache values store ``(wrapped_decoder, [(rank, in_features, init,
# device), …])`` — the second element is the per-layer spec list needed
# for replay, in named_modules iteration order (same order apply_lora used).
_LORA_DECODER_CACHE: Dict[Tuple[int, int, float, str, bool, bool, str],
                          Tuple[torch.nn.Module, List[Tuple]]] = {}


def _get_or_build_wrapped_decoder(
    base_decoder: torch.nn.Module,
    rank: int,
    alpha: float,
    init: str,
    use_dora: bool,
    rs_scaling: bool,
    lora_targets: str,
) -> torch.nn.Module:
    """Cache-aware LoRA-wrapping. Returns the wrapped decoder.

    Cache miss: deepcopy + freeze + ``apply_lora_to_decoder`` (consumes RNG
    via per-layer ``_init_lora_A`` draws); record per-layer specs; cache.
    Cache hit: return the cached wrapped decoder + replay the same RNG
    draws so downstream state matches the un-cached path.

    Callers that mutate the returned decoder (e.g. ``decode_with_lora``'s
    merge) MUST ``copy.deepcopy`` it first — the cache holds the canonical
    wrapped instance and assumes its module structure is immutable.
    """
    from .lora import apply_lora_to_decoder, lora_layers, _init_lora_A

    key = (id(base_decoder), int(rank), float(alpha), str(init),
           bool(use_dora), bool(rs_scaling), str(lora_targets))
    cached = _LORA_DECODER_CACHE.get(key)
    if cached is not None:
        wrapped, specs = cached
        # RNG replay — discard the result; identical # draws as the
        # cache-miss ``apply_lora_to_decoder`` path.
        for (_r, _in_f, _init, _device) in specs:
            _init_lora_A(_r, _in_f, _init, _device)
        return wrapped

    decoder = copy.deepcopy(base_decoder)
    for p in decoder.parameters():
        p.requires_grad_(False)
    apply_lora_to_decoder(
        decoder, rank, alpha, init=init, use_dora=use_dora,
        rs_scaling=rs_scaling, lora_targets=lora_targets,
    )
    # Record per-layer specs for future RNG replay, in named_modules order
    # (matches the order apply_lora_to_decoder iterated).
    specs: List[Tuple] = []
    for name, layer in lora_layers(decoder):
        specs.append((rank, layer.original.in_features, init,
                      layer.original.weight.device))
    _LORA_DECODER_CACHE[key] = (decoder, specs)
    return decoder


def decode_with_lora(
    pipeline_obj: Any,
    slat: Any,
    lora_info: dict,
) -> Any:
    """Decode a SLAT using a temporarily reconstructed LoRA / DoRA decoder.

    Parameters
    ----------
    pipeline_obj : Pipeline
        SAM3D pipeline (provides the frozen decoder to clone).
    slat : SparseTensor
        SLAT tokens to decode.
    lora_info : dict
        ``{"lora_rank": int, "lora_alpha": float, "state_dict": {str: Tensor}
        [, "use_dora": bool, "rs_scaling": bool, "lora_targets": str]}``.

        ``use_dora`` defaults to ``False``; when omitted we auto-detect
        from the presence of ``.magnitude`` keys in ``state_dict``
        (back-compat with caches written before DoRA support).
        ``rs_scaling`` defaults to ``False`` (back-compat with caches
        written before rsLoRA support); set ``True`` to read state
        trained with ``alpha/√rank`` scaling.
        ``lora_targets`` defaults to ``"all"`` (back-compat with caches
        written before the target-filter knob existed); other values
        select which decoder linears were LoRA-wrapped at training.

    Returns
    -------
    Gaussian
        Single decoded Gaussian object.
    """
    import copy
    from .lora import LoRALayer, merge_lora_into_decoder

    state = lora_info["state_dict"]
    use_dora = lora_info.get(
        "use_dora",
        any(k.endswith(".magnitude") for k in state),
    )
    rs_scaling = bool(lora_info.get("rs_scaling", False))
    lora_targets = str(lora_info.get("lora_targets", "all"))

    # Cache-aware build: returns the canonical wrapped decoder.  We deepcopy
    # below because the merge mutates module structure (replaces LoRALayer
    # with stock Linear) and the cache must stay clean for the next caller.
    wrapped_template = _get_or_build_wrapped_decoder(
        pipeline_obj.models["slat_decoder_gs"],
        rank=lora_info["lora_rank"],
        alpha=lora_info["lora_alpha"],
        init="default",  # not stored in lora_info; apply_lora_to_decoder default
        use_dora=use_dora,
        rs_scaling=rs_scaling,
        lora_targets=lora_targets,
    )
    decoder = copy.deepcopy(wrapped_template)
    for param in decoder.parameters():
        param.requires_grad_(False)

    # Load saved adapter weights (A, B always; magnitude when DoRA)
    for name, module in decoder.named_modules():
        if isinstance(module, LoRALayer):
            a_key = f"{name}.lora_A"
            b_key = f"{name}.lora_B"
            m_key = f"{name}.magnitude"
            if a_key in state:
                module.lora_A.data.copy_(state[a_key].to(module.lora_A.device))
            if b_key in state:
                module.lora_B.data.copy_(state[b_key].to(module.lora_B.device))
            if module.magnitude is not None and m_key in state:
                module.magnitude.data.copy_(state[m_key].to(module.magnitude.device))

    # Fold the LoRA / DoRA delta into the frozen base weights so the
    # forward becomes one matmul per layer instead of three (LoRA) or
    # three + a row-norm (DoRA).  Bit-identical to the LoRA forward
    # modulo fp precision; merge is computed in float32 and cast back
    # to the layer's original dtype.
    merge_lora_into_decoder(decoder)

    with torch.no_grad():
        gs = decode_tokens(decoder, slat.feats.float(), slat.coords)

    del decoder
    return gs
