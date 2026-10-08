# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Cross-attention entropy weighting for multi-view velocity averaging.

Implements per-point, per-view observation confidence via normalized Shannon
entropy of the cross-attention distribution.  Low entropy = attention
concentrated on specific image patches = visible region = high weight.
High entropy = diffuse attention = occluded region = low weight.

Reference: MV-SAM3D (https://github.com/devinli123/MV-SAM3D)

Design: Our batched ODE processes all N frames in a single backbone forward
pass.  A persistent forward hook on the designated cross-attention layer
captures q and k for all N views simultaneously.  Entropy and fusion weights
are computed from the **first ODE step only** (step 0, where the latent is
pure noise and view-discriminative attention is strongest) and **frozen** for
all subsequent steps — matching MV-SAM3D's approach without requiring a
separate warmup pass.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


# =====================================================================
# Entropy computation
# =====================================================================

def compute_entropy_weights(
    attn: torch.Tensor,
    alpha: float = 30.0,
    min_weight: float = 0.01,
    patch_start: int = 0,
    patch_end: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-point per-view fusion weights from cross-attention.

    Equivalent to MV-SAM3D ``compute_ss_entropy_weights``.

    Parameters
    ----------
    attn : torch.Tensor
        Head-averaged attention weights ``(N, L, P)`` where N=views,
        L=latent points, P=condition tokens.  Already softmax-normalized
        over the P dimension.
    alpha : float
        Temperature for ``softmax(-alpha * H, dim=0)``.
    min_weight : float
        Floor for per-view weights (prevents complete zeroing).
    patch_start : int
        Start index of patch tokens in the condition sequence.
        Used to extract only image patch tokens for entropy (skip CLS,
        register, mask, pointmap tokens).  Default 0 = use all tokens.
    patch_end : int or None
        End index of patch tokens.  None = use all tokens from patch_start.

    Returns
    -------
    entropy : torch.Tensor
        Normalized Shannon entropy ``(N, L)`` in [0, 1].
    weights : torch.Tensor
        Fusion weights ``(N, L)`` summing to 1 across views (dim 0).
    """
    N = attn.shape[0]

    # Extract patch tokens subset for entropy computation
    if patch_end is not None:
        attn = attn[:, :, patch_start:patch_end]
    elif patch_start > 0:
        attn = attn[:, :, patch_start:]
    P = attn.shape[-1]

    # Normalize attention to sum to 1 over patch tokens
    attn_sum = attn.sum(dim=-1, keepdim=True).clamp(min=1e-10)
    p = attn / attn_sum  # (N, L, P)

    # Normalized Shannon entropy: H = -sum(p * log(p)) / log(P)
    log_p = torch.log(p + 1e-10)
    entropy = -(p * log_p).sum(dim=-1)  # (N, L)
    max_entropy = math.log(P)
    if max_entropy > 0:
        entropy = entropy / max_entropy  # normalize to [0, 1]

    # Fusion weights: softmax(-alpha * H) over views
    if N == 1:
        weights = torch.ones_like(entropy)
    else:
        logits = -alpha * entropy  # (N, L)
        weights = F.softmax(logits, dim=0)  # (N, L)

        # Clamp + renormalize
        if min_weight > 0:
            weights = weights.clamp(min=min_weight)
            weights = weights / weights.sum(dim=0, keepdim=True)

    return entropy, weights


# =====================================================================
# Forward hook for cross-attention entropy extraction
# =====================================================================

class ShapeEntropyHook:
    """Persistent forward hook on a shape cross-attention module.

    Captures cross-attention at the first ODE step (step 0) only, computes
    per-view entropy and fusion weights, then freezes them for all subsequent
    steps (matching MV-SAM3D's warmup-then-freeze approach).  The hook
    returns immediately on steps 1+ without any computation.

    Parameters
    ----------
    cross_attn_module : nn.Module
        A ``MultiHeadAttention`` instance (Stage 1 shape cross-attention).
    alpha : float
        Entropy temperature.
    min_weight : float
        Minimum per-view weight.
    """

    # Default SS condition layout (DINOv2 with prenorm_features=False):
    #   [0]       CLS token
    #   [1:1370]  1369 image patch tokens (37x37 for 518px input, 14px patches)
    # Registers are stripped by prenorm_features=False, so no gap at [1:5].
    # We use [1:1370] — patches only, matching MV-SAM3D Stage 1.
    SS_PATCH_START = 1
    SS_PATCH_END = 1370
    SS_MIN_WEIGHT = 0.001  # MV-SAM3D Stage 1 uses 0.001 (vs 0.01 for Stage 2)

    def __init__(
        self,
        cross_attn_module,
        alpha: float = 30.0,
        min_weight: float = SS_MIN_WEIGHT,
        patch_start: int = SS_PATCH_START,
        patch_end: int = SS_PATCH_END,
    ):
        self.alpha = alpha
        self.min_weight = min_weight
        self.patch_start = patch_start
        self.patch_end = patch_end
        self._captured_this_step = False
        self._weights: Optional[torch.Tensor] = None  # (N, L) — frozen after step 0
        self._entropy: Optional[torch.Tensor] = None  # (N, L) — step 0 entropy
        self._handle = cross_attn_module.register_forward_hook(self._hook_fn)

    # ------------------------------------------------------------------ hook
    def _hook_fn(self, module, inputs, output):
        if self._weights is not None:
            return  # already captured at step 0 — skip all subsequent steps
        if self._captured_this_step:
            return
        self._captured_this_step = True

        x, context = inputs[0], inputs[1]
        with torch.no_grad():
            attn_avg = self._compute_head_avg_attention(module, x, context)
            entropy, weights = compute_entropy_weights(
                attn_avg, alpha=self.alpha, min_weight=self.min_weight,
                patch_start=self.patch_start, patch_end=self.patch_end,
            )
        self._entropy = entropy.detach()
        self._weights = weights.detach()

    def _compute_head_avg_attention(self, module, x, context):
        """Compute head-averaged attention (N, L, P) from q and k.

        Loops over heads to keep memory at (N, L, P) per iteration
        (~150 MB for N=9, L=4096, P=1024).
        """
        N, L, _C = x.shape
        P = context.shape[1]
        num_heads = module.num_heads
        head_dim = module.head_dim

        # Re-derive q, k from module's linear layers
        q = module.to_q(x).reshape(N, L, num_heads, head_dim)
        kv = module.to_kv(context).reshape(N, P, 2, num_heads, head_dim)
        k = kv[:, :, 0]  # (N, P, H, d)

        # qk_rms_norm_cross is False in our model configs — no norm needed.
        # If it were True, we'd apply module.q_rms_norm / module.k_rms_norm.
        if getattr(module, "qk_rms_norm", False):
            q = module.q_rms_norm(q)
            k = module.k_rms_norm(k)

        scale = 1.0 / math.sqrt(head_dim)

        # Per-head loop for memory safety
        attn_sum = torch.zeros(N, L, P, device=x.device, dtype=torch.float32)
        for h in range(num_heads):
            # q_h: (N, L, d), k_h: (N, P, d)
            scores = torch.bmm(
                q[:, :, h].float(),
                k[:, :, h].float().transpose(1, 2),
            ) * scale  # (N, L, P)
            attn_h = F.softmax(scores, dim=-1)
            attn_sum += attn_h

        return attn_sum / num_heads  # (N, L, P)

    # ------------------------------------------------------------------ API
    def get_weights(self) -> Optional[torch.Tensor]:
        """Return step-0 frozen ``(N, L)`` weights."""
        return self._weights

    def get_entropy(self) -> Optional[torch.Tensor]:
        """Return step-0 entropy ``(N, L)`` for visualization."""
        return self._entropy

    def reset_step(self):
        """Reset capture flag for next ODE step."""
        self._captured_this_step = False

    def remove(self):
        """Remove the forward hook."""
        self._handle.remove()


