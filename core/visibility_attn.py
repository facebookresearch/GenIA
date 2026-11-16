# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Visibility-guided cross-attention bias for Stage 2 SLAT prediction.

Injects geometric visibility information (voxel→pixel mapping from DDA
ray tracing) as an additive attention bias in the DiT backbone's
cross-attention layers.  Visible voxels are steered to attend to their
geometrically correct image patches; occluded voxels fall back to
unbiased (original) attention.

Passive-stream compensation
---------------------------
When the bias is enabled on a subset of streams ("active"), the global
softmax over the concatenated 4-stream context shifts the partition
function, indirectly suppressing the un-biased ("passive") streams'
aggregate share.  ``compensate_passive_streams="approx"`` subtracts a
per-voxel scalar ``c = log(v·exp(α) + (1-v))`` from every active-stream
patch token (``v`` = fraction of active patches visible to the voxel).
Under uniform pre-softmax logits this exactly preserves both passive AND
global (CLS/region) tokens' aggregate post-softmax shares while keeping
the visible:invisible steering ratio at ``exp(α):1``, at zero extra cost.

This mechanism is **orthogonal** to the velocity-level
``visibility_weighting`` (which weights per-view velocities *after*
the backbone forward).  This operates *inside* the backbone, biasing
cross-attention logits *during* the forward pass.  Both can be used
independently or together.

Works with any batch size N.  When per-view masks are provided (one mask
per view), each view gets its own bias in the block-diagonal layout.
When a single mask is provided, it is shared across all views.

Usage::

    from genia.core.visibility_attn import (
        SparseVisibilityBiasHook, build_voxel_patch_mask,
    )

    # Build mask from voxel→pixel coords (from compute_visibility_multi_object),
    # mapped into the 518x518 DINOv2 input first
    vids, pixels = flatten_pixel_coords(pixel_coords_view0)
    mask = build_voxel_patch_mask(vids, transform_pixels_to_518(pixels, obj_mask),
                                  L_original)

    # Hook into all cross-attention blocks
    hooks = []
    for block in wrapper.blocks:
        hook = SparseVisibilityBiasHook(
            block.cross_attn, mask, alpha=5.0, N=1, L=L_original,
        )
        hooks.append(hook)

    # ... run ODE ...

    # Cleanup
    for h in hooks:
        h.remove()
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import xformers.ops as xops


# =====================================================================
# Patch layout constants (DINOv2 with prenorm_features=True for Stage 2)
# =====================================================================

VIT_PATCH_SIZE = 14       # pixels per ViT patch
VIT_GRID_H = 37           # 518 / 14 = 37
VIT_GRID_W = 37
NUM_PATCHES = VIT_GRID_H * VIT_GRID_W  # 1369

# Condition token layout (per DINO stream): [CLS, reg0..reg3, patch0..patch1368]
SLAT_PATCH_START = 5
SLAT_PATCH_END = SLAT_PATCH_START + NUM_PATCHES  # 1374

# Dual-stream token layout: the EmbedderFuser concatenates four DINO streams:
#   [0:1374]     DINO(image = cropped object)
#   [1374:2748]  DINO(rgb_image = full scene)
#   [2748:4122]  DINO(mask = cropped)
#   [4122:5496]  DINO(rgb_image_mask = full)
FULL_IMAGE_OFFSET = SLAT_PATCH_END                            # 1374
FULL_IMAGE_PATCH_START = FULL_IMAGE_OFFSET + SLAT_PATCH_START  # 1379
FULL_IMAGE_PATCH_END = FULL_IMAGE_OFFSET + SLAT_PATCH_END      # 2748

CROPPED_MASK_OFFSET = 2 * SLAT_PATCH_END                       # 2748
CROPPED_MASK_PATCH_START = CROPPED_MASK_OFFSET + SLAT_PATCH_START  # 2753
CROPPED_MASK_PATCH_END = CROPPED_MASK_OFFSET + SLAT_PATCH_END     # 4122

FULL_MASK_OFFSET = 3 * SLAT_PATCH_END                          # 4122
FULL_MASK_PATCH_START = FULL_MASK_OFFSET + SLAT_PATCH_START     # 4127
FULL_MASK_PATCH_END = FULL_MASK_OFFSET + SLAT_PATCH_END         # 5496

# SparseDownsample factor for the SLAT backbone
_DOWNSAMPLE_FACTOR = (2, 2, 2)


# =====================================================================
# Coordinate transform: original image → 518×518 DINOv2 input space
# =====================================================================

_DINOV2_SIZE = 518  # DINOv2 input resolution


def flatten_pixel_coords(
    pixel_coords: Dict[int, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """``{voxel_idx: (M, 2) (row, col)}`` -> ``(vids (K,) int64, pixels (K, 2))``: every
    voxel's pixels concatenated in dict order, each row tagged with its voxel.  Voxels with
    no pixel contribute nothing.  The flat form the transforms and the mask builder take,
    so none of them loops over voxels in Python."""
    items = [(v, p) for v, p in pixel_coords.items() if len(p) > 0]
    if not items:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 2), dtype=np.int64)
    vids = np.repeat(np.fromiter((v for v, _ in items), dtype=np.int64, count=len(items)),
                     [len(p) for _, p in items])
    return vids, np.concatenate([p for _, p in items])


def transform_pixels_to_518(
    pixels: np.ndarray,
    mask: np.ndarray,
    box_size_factor: float = 1.0,
    padding_factor: float = 0.1,
) -> np.ndarray:
    """Transform pixel coordinates from original image space to 518×518
    DINOv2 input space.

    Replicates the SLAT preprocessor's crop-pad-resize chain:
    ``crop_around_mask_with_padding(box_size_factor=1.0, padding_factor=0.1)``
    → ``pad_to_square_centered`` (no-op) → ``Resize(518)``.

    Parameters
    ----------
    pixels : ndarray (K, 2)
        Pixel coords ``(row, col)`` in original image space
        (:func:`flatten_pixel_coords`).
    mask : ndarray (H, W)
        Binary object mask in original image space (used to compute bbox).
    box_size_factor : float
        Bounding box scale factor (matches preprocessor, default 1.0).
    padding_factor : float
        Extension factor on each side (matches preprocessor, default 0.1).

    Returns
    -------
    transformed : ndarray (K, 2) int32
        The same pixels in 518×518 space ``(row_518, col_518)``; returned
        unchanged when the mask is empty (no transform possible).
    """
    # Replicate compute_mask_bbox logic (from img_and_mask_transforms.py:335)
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return pixels  # empty mask — no transform possible
    min_x, max_x = int(xs.min()), int(xs.max())
    min_y, max_y = int(ys.min()), int(ys.max())

    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    bbox_w = max_x - min_x
    bbox_h = max_y - min_y
    size = max(bbox_w, bbox_h, 2)
    size = int(size * box_size_factor)

    x1 = int(center_x - size // 2)
    y1 = int(center_y - size // 2)
    x2 = int(center_x + size // 2)
    y2 = int(center_y + size // 2)

    # After crop
    H_crop = y2 - y1
    W_crop = x2 - x1

    # Pad to square
    S = max(H_crop, W_crop)
    pad_h = (S - H_crop) // 2
    pad_w = (S - W_crop) // 2

    # Extend by padding_factor on each side
    ext = int(S * padding_factor)
    S_ext = S + 2 * ext

    # Scale to 518
    scale = _DINOV2_SIZE / S_ext

    rows = pixels[:, 0].astype(np.float64)
    cols = pixels[:, 1].astype(np.float64)
    # Apply: crop → pad → extend → scale
    # Use pixel-center convention: (r + 0.5) * scale - 0.5
    r_pre = (rows - y1) + pad_h + ext
    c_pre = (cols - x1) + pad_w + ext
    r_518 = (r_pre + 0.5) * scale - 0.5
    c_518 = (c_pre + 0.5) * scale - 0.5
    return np.stack([np.floor(r_518).astype(np.int32),
                     np.floor(c_518).astype(np.int32)], axis=1)


# =====================================================================
# Coordinate transform: original image → 518×518 for FULL-IMAGE stream
# =====================================================================
# The full-image stream ("rgb_image") bypasses the mask-based crop.
# Its preprocessing is: pad_to_square_centered → Resize(518).
# See img_processing.py:pad_to_square_centered (lines 110-136).


def transform_pixels_to_518_full_image(
    pixels: np.ndarray,
    H: int,
    W: int,
) -> np.ndarray:
    """Transform pixel coordinates from original image space to 518×518
    DINOv2 input space for the **full-image** conditioning stream.

    Replicates ``pad_to_square_centered`` → ``Resize(518)`` (no mask crop).

    Parameters
    ----------
    pixels : ndarray (K, 2)
        Pixel coords ``(row, col)`` in original image space.
    H, W : int
        Original image dimensions.

    Returns
    -------
    transformed : ndarray (K, 2) int32
        The same pixels in 518×518 space ``(row_518, col_518)``.
    """
    S = max(H, W)
    diff = abs(H - W)
    pad1 = diff // 2
    # pad_to_square_centered: if H > W → pad width (left=pad1), else pad height (top=pad1)
    if H > W:
        pad_top = 0
        pad_left = pad1
    else:
        pad_top = pad1
        pad_left = 0
    scale = _DINOV2_SIZE / S

    rows = pixels[:, 0].astype(np.float64)
    cols = pixels[:, 1].astype(np.float64)
    # pad_to_square_centered shifts then resize scales
    r_518 = (rows + pad_top + 0.5) * scale - 0.5
    c_518 = (cols + pad_left + 0.5) * scale - 0.5
    return np.stack([np.floor(r_518).astype(np.int32),
                     np.floor(c_518).astype(np.int32)], axis=1)


# =====================================================================
# Voxel→patch mask construction
# =====================================================================

def build_voxel_patch_mask(
    vids: np.ndarray,
    pixels: np.ndarray,
    L_original: int,
    patch_size: int = VIT_PATCH_SIZE,
    grid_h: int = VIT_GRID_H,
    grid_w: int = VIT_GRID_W,
    dilate_patches: int = 1,
    device: "torch.device | str" = "cpu",
) -> torch.Tensor:
    """Build a boolean voxel→patch mapping from pixel coordinates.

    **Important**: ``pixels`` must be in 518×518 DINOv2 input space
    (use :func:`transform_pixels_to_518` first if they are in original
    image space).

    Parameters
    ----------
    vids, pixels : ndarray (K,), (K, 2)
        Each pixel's voxel and its ``(row, col)`` in **518×518 space**
        (:func:`flatten_pixel_coords`, then a transform).
    L_original : int
        Total number of voxels at original resolution.
    dilate_patches : int
        Dilate each voxel's patch mask by this many patches (box kernel).
        Default 1 adds a 1-patch border around every occupied patch,
        compensating for projection quantization and patch-boundary
        effects.  Set to 0 to disable.
    device : the device the mask is built (and dilated) on; the hook moves it to
        the backbone's anyway, and a GPU dilates the ``(L, 37, 37)`` grid ~100x faster.

    Returns
    -------
    mask : torch.Tensor
        Boolean ``(L_original, NUM_PATCHES)`` — True where voxel ``i``
        has at least one pixel in patch ``j`` (or a neighboring patch
        within ``dilate_patches``).  Occluded voxels (no pixels) have
        all-False rows.
    """
    num_patches = grid_h * grid_w
    img_size = grid_h * patch_size  # 518
    mask = torch.zeros(L_original, num_patches, dtype=torch.bool, device=device)
    rows = pixels[:, 0].astype(np.int64)
    cols = pixels[:, 1].astype(np.int64)
    # Drop out-of-range voxels and out-of-bounds pixels
    valid = ((vids < L_original) & (rows >= 0) & (rows < img_size)
             & (cols >= 0) & (cols < img_size))
    if not valid.any():
        return mask
    vids, rows, cols = vids[valid], rows[valid], cols[valid]
    # Repeated (voxel, patch) pairs set the same entry: no dedup needed
    patch_idx = (rows // patch_size) * grid_w + (cols // patch_size)
    mask[torch.from_numpy(vids).to(device), torch.from_numpy(patch_idx).to(device)] = True
    if dilate_patches > 0:
        mask = _dilate_patch_grid(mask, grid_h, grid_w, dilate_patches)
    return mask


def _downsample_mask(
    mask: torch.Tensor,
    upsample_idx: torch.Tensor,
    L_original: int,
) -> torch.Tensor:
    """Downsample voxel→patch mask from L_original to L_down.

    ``upsample_idx`` maps each original voxel to its downsampled parent:
    ``upsample_idx[i]`` = index into L_down for original voxel ``i``.
    Multiple original voxels merge into one downsampled voxel — take
    the union (OR) of their patch masks.

    Parameters
    ----------
    mask : torch.Tensor
        Boolean ``(L_original, P)`` mask.
    upsample_idx : torch.Tensor
        ``(N * L_original,)`` — indices into packed ``(N * L_down,)`` space.
        All N batch elements share the same mapping pattern.
    L_original : int
        Number of voxels at original resolution.

    Returns
    -------
    mask_down : torch.Tensor
        Boolean ``(L_down, P)`` mask at downsampled resolution.
    """
    # upsample_idx is (N*L_original,) — extract batch-0 mapping
    idx_b0 = upsample_idx[:L_original]  # (L_original,) → values in [0, L_down)
    return _downsample_mask_local(mask, idx_b0, int(idx_b0.max().item()) + 1)


def _downsample_mask_local(
    mask: torch.Tensor,
    idx_local: torch.Tensor,
    L_down: int,
) -> torch.Tensor:
    """OR-union a voxel→patch mask onto ``L_down`` parents.

    The frame-local core of :func:`_downsample_mask`, split out so a RAGGED
    batch can call it per frame.  ``_downsample_mask`` derives the parent map
    from batch 0 and reuses it for every view — correct only when all views
    share one grid.  With per-frame grids each frame has its own map and its own
    ``L_down``, so the caller slices ``upsample_idx`` to that frame's rows and
    rebases the values by the frame's downsampled offset before calling here.

    Parameters
    ----------
    mask : ``(L_original, P)`` bool.
    idx_local : ``(L_original,)`` parent index per row, in ``[0, L_down)``.
    L_down : number of parents for this frame.
    """
    P = mask.shape[1]
    mask_float = mask.float().to(device=idx_local.device)
    mask_down_float = torch.zeros(L_down, P, device=idx_local.device)
    idx_expanded = idx_local.unsqueeze(1).expand(-1, P)
    mask_down_float.scatter_reduce_(
        0, idx_expanded, mask_float, reduce="amax",
    )
    return mask_down_float > 0.5


# =====================================================================
# Structured bias: the same additive bias, never materialized
# =====================================================================

@triton.jit
def _merge_correct_kernel(
    Q, K, V, O1, L1, O2, L2, OUT, C, PTR, KEYS, FIX, VKV, gain, scale,
    stride_qr, stride_qh, stride_kr, stride_kh, stride_vr, stride_vh,
    stride_o1r, stride_o1h, stride_o2r, stride_o2h, stride_or, stride_oh, stride_lh,
    N_FIX: tl.constexpr, TWO: tl.constexpr, HAS_C: tl.constexpr, H: tl.constexpr,
    D: tl.constexpr,
):
    """One row, all heads, of :meth:`FootprintBias.attend`: merge the FlashAttention call(s),
    then apply the footprint's ``+alpha`` and undo ``-c_i`` where it does not belong.

    ``O1``/``L1``: output and log-sum-exp over keys ``[0, E)`` of the row's view, ``O2``/``L2``
    over ``[E, kv_len)`` (``TWO``).  ``-c_i`` (``HAS_C``) applies to every key of the first
    call, so it shifts ``L1`` by ``-c_i``; the passive keys inside ``[0, E)`` (``FIX``, local
    offsets) must not have it, so their weight is multiplied back by ``e^c_i``, and the
    footprint keys' (``KEYS[PTR[row]:PTR[row+1]]``, packed) by ``e^alpha``.  A key's weight in
    the merged attention is ``exp(logit - c_i - lse)``; multiplying it by ``e^delta`` adds
    ``r = expm1(delta) * exp(logit - c_i - lse)`` to the normaliser and ``r v`` to the output,
    so the biased softmax is ``(out + sum r v) / (1 + sum r)``.  ``logit - c_i - lse <= 0``,
    so nothing overflows.  Each program holds the row's (H, D) tile, so every load is one
    contiguous row of all heads."""
    row = tl.program_id(0)
    hs = tl.arange(0, H)
    d = tl.arange(0, D)[None, :]
    o = tl.load(O1 + row * stride_o1r + hs[:, None] * stride_o1h + d).to(tl.float32)
    lse = tl.load(L1 + hs * stride_lh + row)
    c = 0.0
    if HAS_C:
        c = tl.load(C + row)
        lse -= c
    if TWO:
        o2 = tl.load(O2 + row * stride_o2r + hs[:, None] * stride_o2h + d).to(tl.float32)
        lse2 = tl.load(L2 + hs * stride_lh + row)
        m = tl.maximum(lse, lse2)
        w1 = tl.exp(lse - m)
        w2 = tl.exp(lse2 - m)
        o = (w1[:, None] * o + w2[:, None] * o2) / (w1 + w2)[:, None]
        lse = m + tl.log(w1 + w2)
    start = tl.load(PTR + row)
    end = tl.load(PTR + row + 1)
    if end > start:  # a visible row: the only ones with a footprint, and with c_i > 0
        q = tl.load(Q + row * stride_qr + hs[:, None] * stride_qh + d).to(tl.float32)
        acc = tl.zeros([H, D], dtype=tl.float32)
        den = tl.zeros([H], dtype=tl.float32)
        for f in range(start, end):
            key = tl.load(KEYS + f)
            k = tl.load(K + key * stride_kr + hs[:, None] * stride_kh + d).to(tl.float32)
            v = tl.load(V + key * stride_vr + hs[:, None] * stride_vh + d).to(tl.float32)
            r = gain * tl.exp(tl.sum(q * k, axis=1) * scale - c - lse)
            acc += r[:, None] * v
            den += r
        if HAS_C:
            base = tl.load(VKV + row)
            gain_fix = tl.exp(c) - 1.0
            for f in range(N_FIX):
                key = base + tl.load(FIX + f)
                k = tl.load(K + key * stride_kr + hs[:, None] * stride_kh + d).to(tl.float32)
                v = tl.load(V + key * stride_vr + hs[:, None] * stride_vh + d).to(tl.float32)
                r = gain_fix * tl.exp(tl.sum(q * k, axis=1) * scale - c - lse)
                acc += r[:, None] * v
                den += r
        o = (o + acc) / (1.0 + den)[:, None]
    tl.store(OUT + row * stride_or + hs[:, None] * stride_oh + d, o.to(OUT.dtype.element_ty))


def _flash_available(device) -> bool:
    """xformers' FlashAttention (gappy keys, log-sum-exp) runs on sm80+ only; below,
    :meth:`FootprintBias.attend` uses PyTorch's memory-efficient kernel per view."""
    return torch.cuda.get_device_capability(device) >= (8, 0)


class FootprintBias:
    """The hook's additive bias in the form it has, not as a dense ``(N, 1, L, P)`` tensor.

    Per query row ``i`` (packed over views) the bias is ``alpha`` on the keys of its
    footprint ``F_i`` (the patches the voxel projects onto, on the enabled streams) and
    ``-c_i`` on every active-stream key (the ``approx`` compensation; ``c`` is None when it is
    off).  :meth:`attend` computes exactly the biased attention with FlashAttention at the
    model's own head dimension, on the K/V as they are: the keys of each view split at ``E``
    (the end of the last active stream), one call per side (one call when ``c`` is None or
    nothing lies past ``E``), each returning its log-sum-exp; :func:`_merge_correct_kernel`
    merges them with ``-c_i`` on the first and corrects the few keys whose bias differs from
    their side's: the footprint (``+alpha``) and the passive keys before ``E`` (``FIX``: the
    CLS/register tokens of the active streams, no ``-c_i``).  A dense bias would keep xformers
    off FlashAttention and cost a per-call fp16 cast + 16-head copy of it.

    ``ptr`` (T+1,) slices ``keys`` per packed query row (packed key indices, view ``i``'s
    keys offset by ``i * kv_len``); ``fix`` (local offsets) and ``view_kv`` (T,), each row's
    view offset ``i * kv_len``, are set when ``c`` is.
    """

    NUM_WARPS = 4  # of the merge kernel (fastest on an A100)

    def __init__(self, alpha, q_seqlen, kv_len, ptr, keys, c=None, split=None, fix=None,
                 view_kv=None):
        self.alpha = float(alpha)
        self.q_seqlen = [int(x) for x in q_seqlen]
        self.kv_len = int(kv_len)
        self.ptr, self.keys = ptr, keys
        self.c, self.fix, self.view_kv = c, fix, view_kv
        self.split = int(split) if split is not None else self.kv_len

    def attend(self, q, k, v, q_seqlen, kv_seqlen):
        """``q`` (1, sum_q, H, D), ``k``/``v`` (1, sum_kv, H, D) -> (sum_q, H, D)."""
        N, P, E = len(self.q_seqlen), self.kv_len, self.split
        if list(q_seqlen) != self.q_seqlen or list(kv_seqlen) != [P] * N:
            raise ValueError(
                f"FootprintBias built for q {self.q_seqlen} / kv {[P] * N}, "
                f"called with q {list(q_seqlen)} / kv {list(kv_seqlen)}")
        T, H, D = q.shape[1], q.shape[2], q.shape[3]
        scale = 1.0 / math.sqrt(D)
        two = E < P
        if _flash_available(q.device):
            gappy = xops.fmha.attn_bias.BlockDiagonalGappyKeysMask.from_seqlens
            starts = [i * P for i in range(N)]
            if two:
                o1, l1 = xops.fmha.memory_efficient_attention_partial(
                    q, k, v, gappy(self.q_seqlen, starts + [N * P], [E] * N), scale=scale)
                o2, l2 = xops.fmha.memory_efficient_attention_partial(
                    q, k, v, gappy(self.q_seqlen, [s + E for s in starts] + [N * P],
                                   [P - E] * N), scale=scale)
                o2, l2 = o2[0], l2.reshape(H, T)
            else:
                o1, l1 = xops.fmha.memory_efficient_attention_partial(
                    q, k, v, xops.fmha.BlockDiagonalMask.from_seqlens(self.q_seqlen, [P] * N),
                    scale=scale)
            o1, l1 = o1[0], l1.reshape(H, T)
        else:
            outs = self._per_view_attention(q, k, v, scale, [(0, E), (E, P)] if two else [(0, P)])
            (o1, l1), (o2, l2) = outs[0], outs[-1]
        if not two:
            o2, l2 = o1, l1  # unread when TWO is False
        q3, k3, v3 = q[0], k[0], v[0]
        if q3.stride(-1) != 1 or k3.stride(-1) != 1 or v3.stride(-1) != 1:
            q3, k3, v3 = q3.contiguous(), k3.contiguous(), v3.contiguous()
        out = torch.empty(T, H, D, dtype=q.dtype, device=q.device)
        has_c = self.c is not None
        _merge_correct_kernel[(T,)](
            q3, k3, v3, o1, l1, o2, l2, out,
            self.c if has_c else l1, self.ptr, self.keys,
            self.fix if has_c else self.ptr, self.view_kv if has_c else self.ptr,
            math.expm1(self.alpha), scale,
            q3.stride(0), q3.stride(1), k3.stride(0), k3.stride(1), v3.stride(0), v3.stride(1),
            o1.stride(0), o1.stride(1), o2.stride(0), o2.stride(1), out.stride(0), out.stride(1),
            l1.stride(0),
            N_FIX=self.fix.numel() if has_c else 0, TWO=two, HAS_C=has_c, H=H, D=D,
            num_warps=self.NUM_WARPS,
        )
        return out

    def _per_view_attention(self, q, k, v, scale, ranges):
        """Below sm80 (2080 Ti), where xformers has no FlashAttention and its cutlass kernel no
        log-sum-exp: PyTorch's memory-efficient kernel (sm50+), one call per view and key range
        ``ranges`` (slices of the K/V, no copies).  -> per range ``(out (T, H, D), lse (H, T))``,
        the layout of the xformers path."""
        P = self.kv_len
        outs = [([], []) for _ in ranges]
        qo = 0
        for i, L in enumerate(self.q_seqlen):
            qi = q[:, qo:qo + L].transpose(1, 2)
            for (o_list, l_list), (s, e) in zip(outs, ranges):
                o, lse = torch.ops.aten._scaled_dot_product_efficient_attention(
                    qi, k[:, i * P + s:i * P + e].transpose(1, 2),
                    v[:, i * P + s:i * P + e].transpose(1, 2), None, True, scale=scale)[:2]
                o_list.append(o[0].transpose(0, 1))
                l_list.append(lse[0, :, :L])  # the kernel pads its lse rows to 32
            qo += L
        return [(torch.cat(o), torch.cat(l, 1)) for o, l in outs]

    def to_dense(self) -> torch.Tensor:
        """The ``(N, 1, max L, kv_len)`` fp32 bias it stands for (debug capture, tests)."""
        N, P = len(self.q_seqlen), self.kv_len
        dev = self.keys.device
        offs = torch.tensor([0] + self.q_seqlen, device=dev).cumsum(0)
        dense = torch.zeros(N, 1, max(self.q_seqlen), P, device=dev)
        counts = (self.ptr[1:] - self.ptr[:-1]).long()
        rows = torch.arange(int(offs[-1]), device=dev).repeat_interleave(counts)
        view = torch.searchsorted(offs[1:], rows, right=True)
        keys = self.keys.long()
        dense[view, 0, rows - offs[view], keys - view * P] = self.alpha
        if self.c is not None:
            act = torch.ones(P, dtype=torch.bool, device=dev)
            act[self.split:] = False
            act[self.fix.long()] = False
            for n in range(N):
                block = dense[n, 0, :self.q_seqlen[n]]
                block[:, act] -= self.c[offs[n]:offs[n + 1], None]
        return dense


def build_footprint_bias(
    streams: Sequence[Tuple[int, int, "Optional[List[torch.Tensor]]", bool]],
    L_down_list: Sequence[int],
    P_total: int,
    alpha: float,
    compensate: bool,
) -> FootprintBias:
    """The :class:`FootprintBias` of the hook's per-view, per-stream masks.

    ``streams``: ``(start, end, masks, enabled)`` per DINO stream, ``masks`` a per-view list of
    ``(L_down_i, end - start)`` bool or None.  A stream contributes footprint keys when it is
    enabled, has masks and lies inside the context (``end <= P_total``), the hook's rule;
    the active streams are those, the passive ones the rest in range.  ``compensate``:
    ``approx`` compensation, applied (as in the hook) only when both kinds exist;
    ``c_i = log(v e^alpha + 1 - v)`` with ``v`` = the row's footprint keys / active keys."""
    N = len(L_down_list)
    device = next(m[0].device for _, _, m, _ in streams if m)
    q_offs = [0]
    for L in L_down_list:
        q_offs.append(q_offs[-1] + int(L))
    active_ranges, n_passive = [], 0
    rows, keys = [], []
    for start, end, masks, enabled in streams:
        if end > P_total:
            continue
        if not (enabled and masks):
            n_passive += 1
            continue
        active_ranges.append((start, end))
        for i in range(N):
            r, col = masks[i].nonzero(as_tuple=True)
            rows.append(r + q_offs[i])
            keys.append(col + start + i * P_total)
    T = q_offs[-1]
    rows = torch.cat(rows) if rows else torch.zeros(0, dtype=torch.long, device=device)
    keys = torch.cat(keys) if keys else torch.zeros(0, dtype=torch.long, device=device)
    order = torch.argsort(rows, stable=True)
    rows, keys = rows[order], keys[order]
    counts = torch.bincount(rows, minlength=T)
    ptr = torch.zeros(T + 1, dtype=torch.long, device=device)
    ptr[1:] = counts.cumsum(0)
    c = split = fix = view_kv = None
    if compensate and active_ranges and n_passive:
        n_active = sum(e - s for s, e in active_ranges)
        c = torch.log1p(counts.float() / n_active * math.expm1(alpha))
        split = max(e for _, e in active_ranges)
        act = torch.zeros(split, dtype=torch.bool, device=device)
        for s, e in active_ranges:
            act[s:e] = True
        fix = (~act).nonzero().squeeze(1).int()
        view_kv = torch.cat([torch.full((int(L),), i * P_total, dtype=torch.int32, device=device)
                             for i, L in enumerate(L_down_list)])
    return FootprintBias(alpha, L_down_list, P_total, ptr.int(), keys.int(),
                         c=c, split=split, fix=fix, view_kv=view_kv)


# =====================================================================
# Forward hook for visibility-biased cross-attention
# =====================================================================

class SparseVisibilityBiasHook:
    """Forward pre-hook that injects visibility-based attention bias into
    sparse cross-attention in the Stage 2 SLAT DiT backbone.

    Registers on a ``SparseMultiHeadAttention`` module.  Before each
    forward call, injects an ``attn_bias`` keyword argument so the
    module's native cross-attention path adds ``+alpha`` to logits for
    (voxel, patch) pairs where the voxel is geometrically visible in
    that patch.

    Occluded voxels (all-False mask rows) receive zero bias — their
    attention is unchanged from the original (unbiased fallback).

    The bias is static for the entire ODE solve (geometry doesn't change).

    Parameters
    ----------
    cross_attn_module : nn.Module
        A ``SparseMultiHeadAttention`` (cross-attention type).
    voxel_patch_mask : torch.Tensor or list[torch.Tensor]
        Boolean ``(L_original, NUM_PATCHES)`` from :func:`build_voxel_patch_mask`.
        Pass a list of N tensors for per-view masks, or a single tensor
        to share across all views.
    alpha : float
        Additive bias temperature.  0 = no effect; larger = stronger
        steering toward visible patches.
    N : int
        Number of views (batch elements).  When per-view masks are
        provided, each view gets its own bias on the block diagonal.
    L : int
        Number of voxels at original resolution.  Shared by every view; with
        per-view grids pass ``per_view_L`` instead (``L`` is then unused).
    per_view_L : Sequence[int], optional
        RAGGED mode: per-view voxel counts, one per batch element, when the N
        batch elements sit on DIFFERENT grids (``stage2_dyn``'s per-frame
        deformed grids).  ``None`` (the default) keeps the rectangular
        behaviour exactly, so ``stage2_mv`` is unaffected by construction.

        The bias then packs each view's rows at its own length.
    compensate_passive_streams : str
        ``"off"`` or ``"approx"`` (see the module docstring).
    debug : bool
        When True, capture head-0 attention before/after on first call
        (expensive — only enable for visualization).
    """

    def __init__(
        self,
        cross_attn_module,
        voxel_patch_mask: "torch.Tensor | list[torch.Tensor]",
        alpha: float = 5.0,
        N: int = 1,
        L: int = 0,
        per_view_L: Optional[Sequence[int]] = None,
        patch_start: int = SLAT_PATCH_START,
        patch_end: int = SLAT_PATCH_END,
        voxel_patch_mask_full: "Optional[torch.Tensor | list[torch.Tensor]]" = None,
        full_patch_start: int = FULL_IMAGE_PATCH_START,
        full_patch_end: int = FULL_IMAGE_PATCH_END,
        stream_enables: Optional[Dict[str, bool]] = None,
        compensate_passive_streams: str = "off",
        debug: bool = False,
        shared_state: Optional[Dict[str, "torch.Tensor"]] = None,
    ):
        self.alpha = alpha
        self.N = N
        self.L = L  # L_original (shared); unused when per_view_L is set
        self._per_view_L = (
            [int(x) for x in per_view_L] if per_view_L is not None else None
        )
        if self._per_view_L is not None and len(self._per_view_L) != N:
            raise ValueError(
                f"per_view_L has {len(self._per_view_L)} entries but N={N}"
            )
        self.patch_start = patch_start
        self.patch_end = patch_end
        self.full_patch_start = full_patch_start
        self.full_patch_end = full_patch_end
        self._debug = debug

        # Per-stream enable flags (default all True)
        se = stream_enables or {}
        self._enable_cropped_image = se.get("cropped_image", True)
        self._enable_full_image = se.get("full_image", True)
        self._enable_cropped_mask = se.get("cropped_mask", True)
        self._enable_full_mask = se.get("full_mask", True)

        # Normalize to list for uniform handling
        if isinstance(voxel_patch_mask, torch.Tensor):
            voxel_patch_mask = [voxel_patch_mask]
        if isinstance(voxel_patch_mask_full, torch.Tensor):
            voxel_patch_mask_full = [voxel_patch_mask_full]

        # Passive-stream partition-function compensation: "off" | "approx"
        if compensate_passive_streams not in ("off", "approx"):
            raise ValueError(
                f"compensate_passive_streams must be 'off' or 'approx', "
                f"got {compensate_passive_streams!r}"
            )
        self._compensate_passive_streams = compensate_passive_streams

        # The debug capture reshapes the packed queries to (N, L_down, ...),
        # which only exists when every view has the same length.  Refuse
        # rather than silently mis-splitting them.
        if self._per_view_L is not None:
            if debug:
                raise ValueError(
                    "debug=True is rectangular-only "
                    "(_capture_debug_attention reshapes to (N, L_down, ...)); "
                    "it cannot be combined with per_view_L."
                )

        # Store masks; will be downsampled lazily on first pre-hook call
        self._masks_original = voxel_patch_mask  # list[(L_original, P_patches)] bool
        self._masks_original_full = voxel_patch_mask_full  # list or None
        self._bias: Optional[FootprintBias] = None
        self._masks_downsampled: Optional[List[torch.Tensor]] = None
        self._masks_downsampled_full: Optional[List[torch.Tensor]] = None
        self._L_down_list: Optional[List[int]] = None
        # Optional shared cache: hooks created together (same masks/N/alpha)
        # can share a single bias instead of holding one copy per layer.
        self._shared_state = shared_state

        # CFG-aware activation: only bias the conditional forward pass
        self._active = True

        # Debug: capture attention before/after and coords on first call
        self._debug_attn_before: Optional[torch.Tensor] = None
        self._debug_attn_after: Optional[torch.Tensor] = None
        self._debug_coords: Optional[torch.Tensor] = None  # (L_down, 3)
        self._debug_captured = False

        self._handle = cross_attn_module.register_forward_pre_hook(
            self._pre_hook_fn, with_kwargs=True,
        )

    def _build_bias(self, query_sparse, context) -> FootprintBias:
        """Build the attention bias on first call: a :class:`FootprintBias`.

        Downsamples each view's voxel→patch mask from L_original to
        L_down, then turns the per-view, per-stream masks into the structured
        bias (footprint keys per packed row, compensation per row).

        RECTANGULAR (``per_view_L is None``): every view has the same
        ``L_down = total_points // N`` and the parent map is taken from batch 0
        and reused.

        RAGGED (``per_view_L`` set): each view has its OWN length and its OWN
        parent map, both derived from the layouts rather than from a division;
        the rows are packed at those lengths, so there is no padding.
        """
        P_total = context.shape[1]

        # Get downsample idx to map mask from L_original → L_down
        factor = _DOWNSAMPLE_FACTOR
        upsample_idx = query_sparse.get_spatial_cache(
            f"upsample_{factor}_idx"
        )

        device = query_sparse.device
        N = self.N
        ragged = self._per_view_L is not None

        if ragged:
            # Downsampled slices, one per view — the lengths a `// N` cannot
            # express.  SparseDownsample keys its unique() on the batch column
            # as the most-significant term, so these stay grouped and ascending.
            q_layout = query_sparse.layout
            L_down_list = [sl.stop - sl.start for sl in q_layout]
            # The ORIGINAL (pre-downsample) layout, cached by SparseDownsample
            # alongside the idx it belongs to.
            up_layout = query_sparse.get_spatial_cache(
                f"upsample_{factor}_layout"
            )
            if upsample_idx is not None and up_layout is None:
                raise RuntimeError(
                    "ragged bias needs the upsample layout cache "
                    f"('upsample_{factor}_layout') alongside the idx cache"
                )
            # The masks were built at `per_view_L[i]` rows, but the parent map
            # comes from the tensor being forwarded.  If those disagree the
            # driver is biasing a grid it is not running, so say so here rather
            # than let scatter_reduce fail with a shape complaint 3 frames deep.
            _src = up_layout if up_layout is not None else q_layout
            _actual = [sl.stop - sl.start for sl in _src]
            if _actual != self._per_view_L:
                raise ValueError(
                    f"per_view_L {self._per_view_L} does not match the "
                    f"queried grid's per-view lengths {_actual}"
                )
        else:
            L_down = query_sparse.feats.shape[0] // N
            L_down_list = [L_down] * N
        self._L_down_list = L_down_list

        # --- Downsample per-view masks ---
        masks_down = []
        masks_down_full = []
        for i in range(N):
            # Pick mask for this view (reuse last if fewer masks than views)
            idx = min(i, len(self._masks_original) - 1)
            mask = self._masks_original[idx].to(device)
            if upsample_idx is not None:
                # RAGGED: this view's own parent map — its slice of the packed
                # idx, rebased to [0, L_down_i).  RECTANGULAR: batch 0's map,
                # reused for every view.
                if ragged:
                    idx_local = upsample_idx[up_layout[i]] - q_layout[i].start
                    mask_d = _downsample_mask_local(
                        mask, idx_local, L_down_list[i])
                else:
                    mask_d = _downsample_mask(mask, upsample_idx, self.L)
            else:
                mask_d = mask
            masks_down.append(mask_d)

            if self._masks_original_full is not None:
                idx_f = min(i, len(self._masks_original_full) - 1)
                mask_f = self._masks_original_full[idx_f].to(device)
                if upsample_idx is not None:
                    # The full-image stream needs the SAME per-frame map; using
                    # frame 0's here would mis-bias it silently, since the
                    # shapes still line up.
                    if ragged:
                        mask_df = _downsample_mask_local(
                            mask_f, idx_local, L_down_list[i])
                    else:
                        mask_df = _downsample_mask(
                            mask_f, upsample_idx, self.L)
                else:
                    mask_df = mask_f
                masks_down_full.append(mask_df)

        self._masks_downsampled = masks_down
        self._masks_downsampled_full = masks_down_full or None

        # --- The bias, structured: footprint keys per row + compensation ---
        full = masks_down_full or None
        return build_footprint_bias(
            [(self.patch_start, self.patch_end, masks_down, self._enable_cropped_image),
             (self.full_patch_start, self.full_patch_end, full, self._enable_full_image),
             (CROPPED_MASK_PATCH_START, CROPPED_MASK_PATCH_END, masks_down,
              self._enable_cropped_mask),
             (FULL_MASK_PATCH_START, FULL_MASK_PATCH_END, full, self._enable_full_mask)],
            L_down_list, P_total, self.alpha,
            compensate=self._compensate_passive_streams == "approx",
        )

    def set_active(self, active: bool) -> None:
        """Enable/disable the bias.  Use to skip unconditional CFG passes."""
        self._active = active

    def _pre_hook_fn(self, module, args, kwargs):
        """Inject ``attn_bias`` into the module's forward kwargs."""
        if not self._active or self.alpha == 0.0:
            return args, kwargs

        from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.basic import (
            SparseTensor,
        )

        query_sparse = args[0] if args else kwargs.get("x")
        context = args[1] if len(args) > 1 else kwargs.get("context")

        if query_sparse is None or not isinstance(query_sparse, SparseTensor):
            return args, kwargs

        # Build bias on first call (lazy — needs query_sparse for downsample
        # idx). With a shared_state cache, only the first hook to fire builds
        # the bias; subsequent hooks reuse the same tensor.
        if self._bias is None:
            cached = self._shared_state.get("bias") if self._shared_state is not None else None
            if cached is not None:
                self._bias = cached
                # Restore the per-hook derived state that _build_bias would
                # otherwise set.  It is identical across hooks (same
                # masks/N/alpha), but only the bias-building hook computes it.
                st = self._shared_state
                self._masks_downsampled = st.get("masks_down")
                self._masks_downsampled_full = st.get("masks_down_full")
                self._L_down_list = st.get("L_down_list")
            else:
                self._bias = self._build_bias(query_sparse, context)
                if self._shared_state is not None:
                    self._shared_state["bias"] = self._bias
                    self._shared_state["masks_down"] = self._masks_downsampled
                    self._shared_state["masks_down_full"] = self._masks_downsampled_full
                    self._shared_state["L_down_list"] = self._L_down_list
            # Capture coords for debug viz (batch 0, xyz only).  layout[0]
            # is view 0's rows: slice(0, L_down) when rectangular, and the
            # right thing when each view has its own length.
            coords = query_sparse.coords  # (sum_L, 4): [batch, x, y, z]
            self._debug_coords = (
                coords[query_sparse.layout[0], 1:4].detach().cpu()
            )

        # Debug: capture head-0 attention before/after on first call
        if self._debug and not self._debug_captured:
            self._capture_debug_attention(module, query_sparse, context)

        kwargs = dict(kwargs)  # don't mutate the original
        kwargs["attn_bias"] = self._bias  # built on the query's device
        return args, kwargs

    def _capture_debug_attention(self, module, query_sparse, context):
        """Compute head-0 attention with and without bias for visualization."""
        N = self.N
        L_down = query_sparse.feats.shape[0] // N
        P = context.shape[1]

        with torch.no_grad():
            q = module._linear(module.to_q, query_sparse)
            q = module._reshape_chs(q, (module.num_heads, -1))
            q_feats = q.feats if hasattr(q, "feats") else q
            q_4d = q_feats.reshape(N, L_down, module.num_heads, -1)

            kv = module.to_kv(context)
            kv = kv.reshape(N, P, 2, module.num_heads, -1)
            k = kv[:, :, 0]

            if getattr(module, "qk_rms_norm", False):
                q_4d = module.q_rms_norm(q_4d)
                k = module.k_rms_norm(k)

            head_dim = q_4d.shape[-1]
            # (N, L_down, P) scores for head 0
            scores = torch.bmm(
                q_4d[:, :, 0], k[:, :, 0].transpose(1, 2),
            ) / math.sqrt(head_dim)

            self._debug_attn_before = F.softmax(
                scores.float(), dim=-1,
            ).detach().cpu()

            dense = self._bias.to_dense()  # (N, 1, L_down, P_total)
            biased_scores = scores.clone()
            for i in range(N):
                biased_scores[i] += dense[i, 0].to(scores.dtype)
            self._debug_attn_after = F.softmax(
                biased_scores.float(), dim=-1,
            ).detach().cpu()

        self._debug_captured = True

    # ------------------------------------------------------------------ API
    def get_debug_attention(
        self,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return (attn_before, attn_after) from first hook call.

        Both are ``(N, L_down, P)`` on CPU, head 0 only.
        Returns ``(None, None)`` if ``debug=False``.
        """
        return self._debug_attn_before, self._debug_attn_after

    def get_mask_downsampled(self) -> Optional[List[torch.Tensor]]:
        """Return per-view ``(L_down, P_patches)`` cropped-stream masks."""
        return self._masks_downsampled

    def get_mask_downsampled_full(self) -> Optional[List[torch.Tensor]]:
        """Return per-view ``(L_down, P_patches)`` full-image-stream masks."""
        return self._masks_downsampled_full

    def get_bias(self) -> Optional[FootprintBias]:
        """Return the :class:`FootprintBias` (None before first call); ``.to_dense()`` gives
        the ``(N, 1, L_down, P_total)`` per-view stacked tensor."""
        return self._bias

    def get_coords(self) -> Optional[torch.Tensor]:
        """Return ``(L_down, 3)`` xyz voxel coords at hooked resolution."""
        return self._debug_coords

    def remove(self):
        """Remove the forward hook."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# =====================================================================
# DINOv2 feature capture hook
# =====================================================================

# Stream names in EmbedderFuser call order (matches embedder_list iteration).
DINO_STREAM_NAMES = ["Cropped Obj", "Full Scene", "Cropped Mask", "Full Mask"]


class DinoFeatureCaptureHook:
    """Capture raw DINOv2 patch tokens from EmbedderFuser's Dino calls.

    Registers forward hooks on each unique Dino embedder inside the
    EmbedderFuser.  Hooks fire in the sequential call order of
    ``EmbedderFuser.forward()``::

        embedder_0(image)  → Cropped Obj
        embedder_0(rgb_image) → Full Scene
        embedder_1(mask)  → Cropped Mask
        embedder_1(rgb_image_mask) → Full Mask

    Across sub-batch chunks, each chunk produces 4 calls.
    Use :meth:`get_per_stream_features` to reassemble into
    ``{stream_name: (N_views, 1374, D)}`` tensors.

    Parameters
    ----------
    embedder_fuser : EmbedderFuser
        The condition embedder module (``pipeline.condition_embedders[...]``).
    """

    def __init__(self, embedder_fuser):
        self._outputs: List[torch.Tensor] = []
        self._handles: List = []
        self._n_streams = sum(
            len(kwargs_info) for _, kwargs_info in embedder_fuser.embedder_list
        )
        seen: set = set()
        for embedder, _ in embedder_fuser.embedder_list:
            if id(embedder) not in seen:
                seen.add(id(embedder))
                h = embedder.register_forward_hook(self._hook_fn)
                self._handles.append(h)

    def _hook_fn(self, module, input, output):
        self._outputs.append(output.detach().cpu())

    def get_per_stream_features(self) -> Dict[str, torch.Tensor]:
        """Reassemble captured outputs into per-stream tensors.

        Returns
        -------
        dict
            ``{stream_name: (N_views, 1374, D)}`` for each DINO stream.
        """
        if not self._outputs:
            return {}
        ns = self._n_streams
        n_chunks = len(self._outputs) // ns
        result = {}
        for j, name in enumerate(DINO_STREAM_NAMES[:ns]):
            chunks = [self._outputs[i * ns + j] for i in range(n_chunks)]
            result[name] = torch.cat(chunks, dim=0)  # (N_views, 1374, D)
        return result

    def remove(self):
        """Remove all forward hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._outputs.clear()


# =====================================================================
# Helper: parse layer selection string
# =====================================================================

def parse_layer_selection(layers_str: str, num_blocks: int) -> List[int]:
    """Parse a layer selection string into a list of block indices.

    Parameters
    ----------
    layers_str : str
        ``"all"`` or comma-separated indices (supports negative indexing).
        Examples: ``"all"``, ``"0,1,2"``, ``"-1,-2"``, ``"0,1,-1"``.
    num_blocks : int
        Total number of transformer blocks.

    Returns
    -------
    indices : list of int
        Non-negative block indices in ascending order.
    """
    if layers_str.strip().lower() == "all":
        return list(range(num_blocks))

    indices = []
    for s in layers_str.split(","):
        s = s.strip()
        if not s:
            continue
        idx = int(s)
        if idx < 0:
            idx = num_blocks + idx
        if 0 <= idx < num_blocks:
            indices.append(idx)
    return sorted(set(indices))


# =====================================================================
# Patch-grid helpers
# =====================================================================

def _dilate_patch_grid(grid: torch.Tensor, grid_h: int, grid_w: int,
                       dilate: int = 1) -> torch.Tensor:
    """Dilate a boolean patch grid by ``dilate`` patches in each direction.

    Accepts ``(num_patches,)`` or ``(N, num_patches)`` input.  Uses max-pooling
    on the 2D grid to expand True regions, then re-flattens.
    """
    if dilate <= 0:
        return grid
    k = 2 * dilate + 1
    orig_shape = grid.shape
    if grid.ndim == 1:
        grid_4d = grid.float().reshape(1, 1, grid_h, grid_w)
    else:
        grid_4d = grid.float().reshape(grid.shape[0], 1, grid_h, grid_w)
    dilated = torch.nn.functional.max_pool2d(
        grid_4d, kernel_size=k, stride=1, padding=dilate,
    )
    return (dilated.reshape(orig_shape) > 0.5)

