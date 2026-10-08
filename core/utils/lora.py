# Copyright (c) Meta Platforms, Inc. and affiliates.

"""LoRA / DoRA utilities for decoder fine-tuning.

A two-layer design separates *where the adapter hooks live* from *whose
parameters are currently active*:

* :class:`LoRALayer` wraps an `nn.Linear` (or `SparseLinear`) and adds a
  low-rank delta computed from a pair of adapter matrices ``(A, B)``.
  When :attr:`magnitude` is set, the layer additionally applies the
  *DoRA* magnitude/direction decomposition (Liu et al. 2024).  The
  wrapped original weights remain frozen.
* :class:`LoRAAdapter` owns the per-instance ``(A, B)`` Parameters (and
  optional DoRA ``magnitude``) for every `LoRALayer` in a decoder.
  ``activate()`` rebinds the layer's adapter attributes to point at
  *this* adapter's Parameters, enabling many adapters to share a single
  decoder copy (e.g. multi-object fine-tuning).

Both LoRA initialisations supported by the codebase live in
:func:`_init_lora_A`:

* ``"default"``  — ``randn(rank, in_f) * (1 / rank)`` (project default).
* ``"kaiming"``  — ``nn.init.kaiming_uniform_(A, a=sqrt(5))``, matching
  the reference implementation in *Hu et al.* `LoRA: Low-Rank Adaptation
  of Large Language Models` (arXiv:2106.09685) and HuggingFace PEFT.

DoRA (``use_dora=True``):

* *Liu et al.* `DoRA: Weight-Decomposed Low-Rank Adaptation`
  (ICML 2024 Oral, arXiv:2402.09353).
* Decomposes the pretrained ``W`` (shape ``[out_f, in_f]``) into a
  per-output magnitude vector ``m = ‖W‖_row`` and a unit-direction
  matrix ``W / ‖W‖_row``.  LoRA's low-rank delta updates the direction;
  the magnitude trains as a separate scalar-per-row Parameter.
* Forward (per layer): ``y = (m / ‖W + ΔV‖_row) · (x @ (W + ΔV)^T)``
  where ``ΔV = scaling · B @ A``.  The norm is detached from autograd
  (PEFT default), so gradient on ``A`` / ``B`` only updates direction
  while ``m`` carries magnitude — the explicit decomposition that
  motivates DoRA.
* At iter-0, ``B = 0`` ⇒ ``ΔV = 0`` ⇒ ``‖W + ΔV‖_row = ‖W‖_row = m``,
  so the layer is bit-identical to the frozen base.
* Cost: ~one extra ``out_f``-sized vector per layer + the row-norm
  computation per forward.  Negligible for our backbone.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Adapter parameter initialisation
# ---------------------------------------------------------------------------

LORA_INIT_STRATEGIES = ("default", "kaiming")

# Which decoder ``nn.Linear`` leaves to wrap.  Matched against the
# last segment of each module's ``named_modules`` path.
LORA_TARGET_STRATEGIES = ("all", "attn", "qkv")
_ATTN_LEAVES = frozenset({"to_q", "to_k", "to_v", "to_qkv", "to_kv", "to_out"})
_QKV_LEAVES = _ATTN_LEAVES - {"to_out"}


def _matches_lora_target(name: str, target: str) -> bool:
    """Return True when an ``nn.Linear`` named *name* should be wrapped
    under the given ``lora_targets`` strategy.

    *name* is the dotted ``named_modules`` path; we match the leaf segment
    (everything after the last ``.``).  Unknown ``target`` raises.
    """
    if target == "all":
        return True
    leaf = name.rsplit(".", 1)[-1] if "." in name else name
    if target == "attn":
        return leaf in _ATTN_LEAVES
    if target == "qkv":
        return leaf in _QKV_LEAVES
    raise ValueError(
        f"unknown lora_targets {target!r}; expected one of {LORA_TARGET_STRATEGIES}"
    )


def _compute_scaling(alpha: float, rank: int, rs_scaling: bool) -> float:
    """Per-layer ``scaling`` factor used in the LoRA forward.

    * ``rs_scaling=False`` — classic LoRA: ``alpha / rank``.
    * ``rs_scaling=True``  — rsLoRA (Kalajdzievski 2023, arXiv:2312.03732):
      ``alpha / √rank``.  Decouples the effective delta magnitude from
      ``rank`` so ``lora_lr`` transfers across ranks.

    Bit-equivalent to classic at ``rank=1``; ``√rank`` factor at higher ranks.
    """
    denom = math.sqrt(rank) if rs_scaling else float(rank)
    return float(alpha) / denom


def _init_lora_A(
    rank: int, in_features: int, init: str, device, dtype=torch.float32,
) -> torch.Tensor:
    """Allocate the ``A`` matrix of a LoRA pair under the chosen strategy.

    ``B`` is always initialised to zeros so the adapter starts as a no-op
    at iter 0 — see :class:`LoRAAdapter`.
    """
    if init == "default":
        return torch.randn(rank, in_features, device=device, dtype=dtype) * (1.0 / rank)
    if init == "kaiming":
        # Matches Hu et al. 2021 / HuggingFace PEFT: A ~ Kaiming-uniform.
        tensor = torch.empty(rank, in_features, device=device, dtype=dtype)
        nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))
        return tensor
    raise ValueError(
        f"unknown LoRA init {init!r}; expected one of {LORA_INIT_STRATEGIES}"
    )


def _init_dora_magnitude(layer: "LoRALayer") -> torch.Tensor:
    """Per-output row-norm of the frozen weight, ``‖W[i, :]‖_2``.

    This is the iter-0 magnitude vector for a DoRA adapter: combined
    with ``B = 0`` it makes the layer bit-identical to the frozen base.
    """
    with torch.no_grad():
        W = layer.original.weight.data.float()  # (out_f, in_f), frozen
        return torch.linalg.norm(W, dim=1)      # (out_f,)


# ---------------------------------------------------------------------------
# LoRALayer — frozen original + active (A, B [, magnitude]) adapter
# ---------------------------------------------------------------------------

class LoRALayer(nn.Module):
    """LoRA-wrapped linear layer with optional DoRA magnitude correction.

    Forward (LoRA path, ``magnitude is None``)::

        y = original(x) + (x @ A^T @ B^T) * scaling

    Forward (DoRA path, ``magnitude is set``)::

        ΔV = scaling * (B @ A)             # (out_f, in_f)
        y_lin = (x @ W^T + b) + (x @ ΔV^T) # standard LoRA-augmented linear
        scale = magnitude / ‖W + ΔV‖_row   # (out_f,), DETACHED from autograd
        y     = y_lin * scale              # broadcast on output dim

    The original layer's weights are frozen (set externally, see
    :func:`apply_lora_to_decoder`).  ``A`` / ``B`` / ``magnitude`` are
    the *active* adapter parameters; they can be rebound by
    :class:`LoRAAdapter` to swap in another instance's weights without
    rebuilding the module.

    When ``enabled`` is False both the LoRA delta and the DoRA
    magnitude/norm rescaling are skipped; only the frozen original
    output is returned.  This is used during the CFG unconditional pass
    so that the unconditional velocity is exactly the base model's.
    Callers toggle ``enabled`` via duck typing
    (``getattr(m, "lora_A", None) is not None``) to avoid import cycles.

    Handles both regular Tensor and SparseTensor inputs: when the original
    module is a SparseLinear, returns a SparseTensor; when called via the
    attention ``_linear`` helper (which extracts ``.feats`` first), both
    input and output are plain Tensors.
    """

    def __init__(
        self,
        original: nn.Linear,
        rank: int,
        alpha: float = 1.0,
        init: str = "default",
        use_dora: bool = False,
        rs_scaling: bool = False,
    ):
        super().__init__()
        self.original = original
        self.enabled = True
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.init = str(init)
        self.rs_scaling = bool(rs_scaling)
        in_f = original.in_features
        out_f = original.out_features
        device = original.weight.device
        self.lora_A = nn.Parameter(_init_lora_A(rank, in_f, init, device))
        self.lora_B = nn.Parameter(
            torch.zeros(out_f, rank, device=device, dtype=torch.float32)
        )
        self.scaling = _compute_scaling(alpha, rank, self.rs_scaling)
        # DoRA magnitude — allocated only when ``use_dora=True``.  An
        # adapter's :meth:`LoRAAdapter.activate` may also set this
        # attribute later to swap in a per-object magnitude.
        self.magnitude: Optional[nn.Parameter] = (
            nn.Parameter(_init_dora_magnitude(self)) if use_dora else None
        )

    def _dora_scale(self) -> torch.Tensor:
        """Per-output rescaling factor ``magnitude / ‖W + ΔV‖_row``.

        Returns a ``(1, out_f)`` tensor ready to broadcast against the
        output of the LoRA-augmented linear forward.  The norm is
        ``.detach()``-ed (PEFT default) so gradient on this scale flows
        only through ``magnitude``, not through ``A`` / ``B``.
        """
        W = self.original.weight.float()                                 # (out_f, in_f)
        delta_W = (self.lora_B @ self.lora_A) * self.scaling             # (out_f, in_f)
        norm = torch.linalg.norm(W + delta_W, dim=1).detach()            # (out_f,)
        return (self.magnitude / norm).view(1, -1)                       # (1, out_f)

    def forward(self, x):
        orig_dtype = self.original.weight.dtype
        # Internally we always compute in float32 for LoRA-param gradient
        # stability, but the OUTPUT is cast back to ``orig_dtype`` so the
        # wrapper is dtype-transparent.  This matters when only a subset
        # of the decoder's linears are LoRA-wrapped (``lora_targets`` !=
        # "all"): an upcast-only output would feed float32 into the next
        # unwrapped fp16 layer and crash ("mat1/mat2 dtype mismatch").
        # Cast input to original layer's dtype (decoder may be float16).
        #
        # DoRA path: per Liu et al. 2024 the bias is added AFTER the
        # magnitude scaling — ``y = m · (V/‖V‖_row) · x + b`` — so we
        # subtract the bias from ``orig_out``, scale the (Wx + ΔVx)
        # term, then add the bias back unscaled.  Without this, the
        # bias drifts with ``m`` (small effect at iter-0 where m=‖W‖
        # makes scale=1, but real once ``m`` trains).
        bias = self.original.bias if self.magnitude is not None else None
        if hasattr(x, "feats"):
            x_cast = x.replace(x.feats.to(orig_dtype)) if x.feats.dtype != orig_dtype else x
            orig_out = self.original(x_cast)
            if not self.enabled:
                return orig_out
            feats = x.feats.float()
            delta = nn.functional.linear(
                nn.functional.linear(feats, self.lora_A), self.lora_B
            ) * self.scaling
            out_feats = orig_out.feats.float() + delta
            if self.magnitude is not None:
                if bias is not None:
                    out_feats = out_feats - bias.float()
                out_feats = out_feats * self._dora_scale()
                if bias is not None:
                    out_feats = out_feats + bias.float()
            return orig_out.replace(out_feats.to(orig_dtype))
        else:
            x_cast = x.to(orig_dtype) if x.dtype != orig_dtype else x
            orig_out = self.original(x_cast)
            if not self.enabled:
                return orig_out
            feats = x.float()
            delta = nn.functional.linear(
                nn.functional.linear(feats, self.lora_A), self.lora_B
            ) * self.scaling
            out = orig_out.float() + delta
            if self.magnitude is not None:
                if bias is not None:
                    out = out - bias.float()
                out = out * self._dora_scale()
                if bias is not None:
                    out = out + bias.float()
            return out.to(orig_dtype)


# ---------------------------------------------------------------------------
# Wiring LoRA into a decoder
# ---------------------------------------------------------------------------

def apply_lora_to_decoder(
    decoder: nn.Module, rank: int, alpha: float = 1.0,
    init: str = "default", use_dora: bool = False,
    rs_scaling: bool = False, lora_targets: str = "all",
) -> List[nn.Parameter]:
    """Replace every matching `nn.Linear` in *decoder* with a :class:`LoRALayer`.

    ``lora_targets`` filters which linears get wrapped — see
    :func:`_matches_lora_target` for the rules ("all" / "attn" / "qkv").

    Returns the list of trainable adapter Parameters in iteration order
    (``A`` and ``B`` per layer; plus ``magnitude`` per layer when
    ``use_dora=True``).  When using :class:`LoRAAdapter`, the returned
    list is a "default" adapter — typically discarded once an explicit
    adapter is activated.
    """
    targets = [
        (name, module)
        for name, module in decoder.named_modules()
        if isinstance(module, nn.Linear)
        and _matches_lora_target(name, lora_targets)
    ]

    lora_params: List[nn.Parameter] = []
    for name, module in targets:
        lora_layer = LoRALayer(
            module, rank, alpha, init=init, use_dora=use_dora,
            rs_scaling=rs_scaling,
        )
        # Navigate to the parent and replace the child
        parts = name.split(".")
        parent = decoder
        for part in parts[:-1]:
            parent = getattr(parent, part) if not part.isdigit() else parent[int(part)]
        setattr(parent, parts[-1], lora_layer)
        lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])
        if lora_layer.magnitude is not None:
            lora_params.append(lora_layer.magnitude)

    return lora_params


# ---------------------------------------------------------------------------
# Merge LoRA delta into base weights (post-FINETUNE inference speedup)
# ---------------------------------------------------------------------------

def _merged_weight(layer: "LoRALayer") -> torch.Tensor:
    """Fold a :class:`LoRALayer`'s ``(A, B[, magnitude])`` into a single weight
    matrix that is mathematically equivalent to its forward.

    For plain LoRA:    W' = W + scaling · B @ A
    For DoRA:          W' = (magnitude / ‖W + scaling·B@A‖_row) · (W + scaling·B@A)

    Returned tensor matches ``layer.original.weight.dtype`` so the merged
    module is a drop-in replacement.
    """
    orig_dtype = layer.original.weight.dtype
    W = layer.original.weight.data.float()                                # (out_f, in_f)
    delta = (layer.lora_B.data @ layer.lora_A.data) * layer.scaling        # (out_f, in_f)
    W_merged = W + delta
    if layer.magnitude is not None:
        norm = torch.linalg.norm(W_merged, dim=1)                          # (out_f,)
        scale = (layer.magnitude.data / norm).view(-1, 1)                  # (out_f, 1)
        W_merged = W_merged * scale
    return W_merged.to(orig_dtype)


def merge_lora_into_decoder(decoder: nn.Module) -> nn.Module:
    """In-place replace every :class:`LoRALayer` in *decoder* with a stock
    frozen copy of its original ``nn.Linear`` (or compatible subclass), with
    the LoRA / DoRA delta folded into the weights.

    Forward becomes one matmul per layer instead of three; bias is unchanged
    (DoRA's magnitude scaling is bias-transparent — see
    :meth:`LoRALayer.forward` and the paper :math:`y = m · V/‖V‖ · x + b`).

    Bit-identical to the LoRA forward modulo fp precision: the merge is
    computed in float32 and cast back to the layer's original dtype.

    Idempotent: layers that are already merged (plain ``nn.Linear``) are
    skipped.  Returns the same *decoder* object for chaining.
    """
    targets = [(n, m) for n, m in decoder.named_modules() if isinstance(m, LoRALayer)]
    for name, layer in targets:
        W_merged = _merged_weight(layer)
        original = layer.original
        # Stock copy: same class (handles SparseLinear etc.), same shape,
        # same bias presence.  Frozen.
        merged = original.__class__(
            original.in_features, original.out_features,
            bias=(original.bias is not None),
        ).to(device=original.weight.device, dtype=original.weight.dtype)
        merged.weight.data.copy_(W_merged)
        if original.bias is not None:
            merged.bias.data.copy_(original.bias.data)
        for p in merged.parameters():
            p.requires_grad_(False)

        parts = name.split(".")
        parent = decoder
        for part in parts[:-1]:
            parent = getattr(parent, part) if not part.isdigit() else parent[int(part)]
        setattr(parent, parts[-1], merged)
    return decoder


def lora_layers(decoder: nn.Module) -> List[Tuple[str, "LoRALayer"]]:
    """Return ``[(name, layer)]`` for every :class:`LoRALayer` in *decoder*,
    in `named_modules` order.  Used by :class:`LoRAAdapter`."""
    return [
        (name, module)
        for name, module in decoder.named_modules()
        if isinstance(module, LoRALayer)
    ]


# ---------------------------------------------------------------------------
# LoRAAdapter — per-instance Parameters bound into a shared decoder
# ---------------------------------------------------------------------------

class LoRAAdapter:
    """Per-instance adapter weights for a shared LoRA-wrapped decoder.

    Allocates a fresh ``(A, B)`` Parameter pair (and an optional DoRA
    ``magnitude`` Parameter) for every :class:`LoRALayer` in
    *decoder_with_lora* and stores the layer reference + name alongside.
    ``activate()`` rebinds each layer's ``lora_A`` / ``lora_B`` /
    ``magnitude`` attributes to point at this adapter's Parameters;
    subsequent forward passes use them, and gradients flow back to
    *this adapter's* Parameters only.

    Multi-object fine-tuning thus needs **one** decoder copy + **N** small
    adapters (≈ rank·(Σ in_f + out_f) Parameters each, plus Σ out_f when
    DoRA is on), instead of N full decoder copies.  The optimiser is
    built per-adapter on :attr:`parameters`, so per-object Adam state is
    naturally isolated.
    """

    def __init__(
        self,
        decoder_with_lora: nn.Module,
        rank: int,
        alpha: float,
        init: str = "default",
        use_dora: bool = False,
        rs_scaling: bool = False,
    ):
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.init = str(init)
        self.use_dora = bool(use_dora)
        self.rs_scaling = bool(rs_scaling)
        # Parallel lists (name, layer-ref, A, B, magnitude); fastest to
        # iterate, and ``name`` doubles as the state-dict key prefix.
        # ``magnitude`` is ``None`` per entry when ``use_dora=False``.
        self._names: List[str] = []
        self._layers: List[LoRALayer] = []
        self._A: List[nn.Parameter] = []
        self._B: List[nn.Parameter] = []
        self._mag: List[Optional[nn.Parameter]] = []
        for name, layer in lora_layers(decoder_with_lora):
            in_f = layer.original.in_features
            out_f = layer.original.out_features
            device = layer.original.weight.device
            self._names.append(name)
            self._layers.append(layer)
            self._A.append(nn.Parameter(_init_lora_A(rank, in_f, init, device)))
            self._B.append(nn.Parameter(
                torch.zeros(out_f, rank, device=device, dtype=torch.float32)
            ))
            if use_dora:
                self._mag.append(nn.Parameter(_init_dora_magnitude(layer)))
            else:
                self._mag.append(None)

    @property
    def parameters(self) -> List[nn.Parameter]:
        """Flat list of trainable adapter Parameters (A, B [, magnitude]
        per layer).  ``magnitude`` entries are appended only when
        ``use_dora=True``."""
        out: List[nn.Parameter] = []
        for A, B, m in zip(self._A, self._B, self._mag):
            out.append(A)
            out.append(B)
            if m is not None:
                out.append(m)
        return out

    def activate(self) -> None:
        """Bind this adapter's Parameters into the shared LoRA layers.

        Idempotent: calling twice is a no-op.  After this returns, the
        decoder's forward pass uses *this* adapter's parameters and
        backward fills *this* adapter's grads.

        Always sets ``layer.magnitude`` (to this adapter's per-layer
        ``magnitude`` Parameter, or ``None``), so the activated state is
        fully determined by the active adapter — switching from a
        DoRA-on adapter to a DoRA-off adapter correctly disables DoRA.
        """
        for layer, A, B, m in zip(self._layers, self._A, self._B, self._mag):
            layer.lora_A = A
            layer.lora_B = B
            layer.magnitude = m
            # ``scaling`` is rank/alpha/rs-scaling-derived and shared across
            # adapters, but we set it here defensively so swapping rank,
            # alpha, or rs_scaling between adapters on the same decoder
            # Just Works.
            layer.rs_scaling = self.rs_scaling
            layer.scaling = _compute_scaling(self.alpha, self.rank, self.rs_scaling)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Serialisable ``{f"{name}.lora_A": tensor, f"{name}.lora_B":
        tensor [, f"{name}.magnitude": tensor]}``.

        ``.magnitude`` keys are present only when ``use_dora=True``.
        Format is what ``slat_decode.decode_with_lora`` consumes.
        Tensors are detached and moved to CPU.
        """
        out: Dict[str, torch.Tensor] = {}
        for name, A, B, m in zip(self._names, self._A, self._B, self._mag):
            out[f"{name}.lora_A"] = A.data.detach().cpu()
            out[f"{name}.lora_B"] = B.data.detach().cpu()
            if m is not None:
                out[f"{name}.magnitude"] = m.data.detach().cpu()
        return out
