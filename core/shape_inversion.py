"""Shape token inversion: mesh → voxelize → decoder inversion → shape tokens.

Converts a 3D mesh into the SLAT shape token format ``(4096, 8)`` that SAM3D
uses internally.  The decoder inversion optimizes a latent ``z`` such that
``ss_decoder(z) ≈ gt_occupancy``, keeping ``z`` near the VAE prior ``N(0, I)``
via a moment-matching penalty ``mean(z)² + (var(z) - prior_target_var)²`` —
controls the empirical first/second moments together so individual entries
cannot grow unbounded the way pure ``mean(z²)`` L2 allows.

Functions
---------
- ``voxelize_mesh`` : mesh file → ``(1, 1, 64, 64, 64)`` binary occupancy
- ``invert_decoder`` : occupancy + decoder → optimized latent ``(1, 8, 16, 16, 16)``
- ``latent_to_shape_tokens`` : latent → ``(1, 4096, 8)``
- ``shape_tokens_to_latent`` : ``(1, 4096, 8)`` → latent
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from genia.core.utils.gt_data import _load_and_orient_mesh

logger = logging.getLogger(__name__)


def _voxelise_with_normalization(
    verts: np.ndarray,
    faces: np.ndarray,
    grid_size: int = 64,
    dilate: int = 0,
    center: Optional[np.ndarray] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Voxelise oriented vertices+faces into a binary occupancy grid.

    Per-mesh normalisation when ``(center, scale)`` is ``None`` (centre +
    uniform scale to ``[-0.5, 0.5]``).  Shared normalisation when both
    are provided — used by the actionmesh deformation provider so all
    frames share the canonical-frame's normalisation.

    Returns
    -------
    occupancy : torch.Tensor, shape ``(1, 1, grid_size, grid_size, grid_size)``
    """
    import open3d as o3d

    verts = np.asarray(verts, dtype=np.float64).copy()

    if center is None or scale is None:
        from genia.core.utils.per_frame_grid import compute_sam3d_normalization
        center, scale = compute_sam3d_normalization(verts)

    if scale > 0:
        verts = (verts - center) / scale
    else:
        verts = verts - center
    verts = np.clip(verts, -0.5 + 1e-6, 0.5 - 1e-6)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32))

    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh,
        voxel_size=1 / 64,
        min_bound=(-0.5, -0.5, -0.5),
        max_bound=(0.5, 0.5, 0.5),
    )

    grid_indices = np.array([v.grid_index for v in voxel_grid.get_voxels()])
    normalized = (grid_indices + 0.5) / 64 - 0.5
    coords = ((normalized + 0.5) * grid_size).astype(int)
    coords = np.clip(coords, 0, grid_size - 1)

    occupancy = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    occupancy[coords[:, 0], coords[:, 1], coords[:, 2]] = 1.0

    if dilate > 0:
        from scipy.ndimage import binary_dilation
        occupancy = binary_dilation(occupancy > 0.5, iterations=dilate).astype(np.float32)

    return torch.from_numpy(occupancy).unsqueeze(0).unsqueeze(0)


def voxelize_mesh(
    mesh_path: str, grid_size: int = 64, dilate: int = 0,
    *, local_rotation = "identity",
) -> torch.Tensor:
    """Voxelize a mesh into a binary occupancy grid.

    Follows the exact SAM3D convention:
    1. Normalize vertices to ``[-0.5, 0.5]`` (center + uniform scale)
    2. Surface voxelization via Open3D at ``voxel_size=1/64``
    3. No interior fill (surface only)

    Parameters
    ----------
    mesh_path : str
        Path to mesh file (OBJ, PLY, or any Open3D-supported format).
    grid_size : int
        Resolution of output grid (default 64).
    dilate : int
        Morphological dilation iterations (default 0).

    Returns
    -------
    occupancy : torch.Tensor, shape ``(1, 1, grid_size, grid_size, grid_size)``
    """
    verts, faces = _load_and_orient_mesh(mesh_path, local_rotation=local_rotation)
    return _voxelise_with_normalization(
        verts, faces, grid_size=grid_size, dilate=dilate,
    )


_AUTOCAST_DTYPES = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}


def resolve_autocast_dtype(name: str):
    """Map a ``gt_shapes_inversion.inversion_precision`` value to a torch dtype.

    ``"fp32"`` -> ``None`` (autocast disabled).  ``"auto"`` (the default) is
    ``bf16`` on Ampere or newer and ``fp32`` everywhere else.

    Note: **bf16 is only a speedup on sm_80+.**  Pre-Ampere GPUs (e.g. sm_75,
    sm_70) have no bf16 tensor cores and emulate it, running markedly SLOWER
    than fp32.  A flat ``bf16`` default would therefore regress every
    pre-Ampere GPU, which is what ``"auto"`` exists to avoid.  Round-trip
    quality is essentially the same either way.

    Raises on an unknown value, so a typo in the YAML fails at block entry
    rather than silently running fp32.
    """
    if name == "auto":
        if not torch.cuda.is_available():
            return None
        return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else None
    if name not in _AUTOCAST_DTYPES:
        raise ValueError(
            f"inversion_precision={name!r} — must be 'auto' or one of "
            f"{sorted(_AUTOCAST_DTYPES)}"
        )
    return _AUTOCAST_DTYPES[name]


def invert_decoder(
    decoder: torch.nn.Module,
    gt_occupancy: torch.Tensor,
    num_steps: int = 500,
    lr: float = 0.2,
    prior_weight: float = 0.01,
    prior_target_var: float = 1.0,
    l2_weight: float = 0.0,
    seed: Optional[int] = 42,
    show_progress: bool = True,
    *,
    autocast_dtype: Optional[torch.dtype] = None,
    progress_desc: str = "Shape inversion",
) -> Tuple[torch.Tensor, List[Tuple[float, float]]]:
    """Optimize latent z to reconstruct gt_occupancy via the decoder.

    Batched over the leading dim: ``B`` grids are inverted jointly.  The loss
    is a SUM of per-sample terms and Adam is element-wise, so each row takes
    exactly the gradient it would take alone — the batch is equivalent to ``B``
    independent ``B=1`` calls *in exact arithmetic*.

    Note: **on GPU it is not bit-equal, but neither is the unbatched path.**
    100 Adam steps at lr=0.2 chaotically amplify float-level conv
    nondeterminism, so two identical serial runs already differ in ``z``.
    What is stable is the thing downstream consumes: the round-trip IoU.  Do
    not expect to reproduce a stored latent bit-for-bit.

    Parameters
    ----------
    decoder : nn.Module
        Frozen SparseStructureDecoder.  Must not couple samples (the SAM3D
        ``ss_decoder`` is dense conv3d + Channel/GroupNorm — no BatchNorm).
    gt_occupancy : torch.Tensor
        Binary occupancy ``(B, 1, 64, 64, 64)`` on the same device.
    num_steps : int
        Optimization iterations.
    lr : float
        Adam learning rate.
    prior_weight : float
        Weight on the VAE prior moment-match penalty
        ``mean(z)² + (var(z) - prior_target_var)²``.
    prior_target_var : float
        Variance the prior pushes ``z`` toward.  ``1.0`` matches the
        VAE prior exactly; lower values (e.g. ``0.9``) match the
        empirical Stage-1 distribution better.
    l2_weight : float
        Weight on a per-element L2 penalty ``mean(z²)``.  Note that
        with ``prior_weight > 0`` pinning ``var(z) ≈ prior_target_var``
        this term is essentially constant — leave at ``0`` unless you
        disable the prior.
    seed : int, optional
        Random seed for reproducibility.
    show_progress : bool
        Show a tqdm progress bar.  The postfix is refreshed every
        ``num_steps // 10`` steps, not every step — each refresh is a device
        sync.
    progress_desc : str
        Label for that progress bar (used to name the chunk when called from
        :func:`occupancies_to_shape_tokens`).
    autocast_dtype : torch.dtype, optional
        When set (and ``gt_occupancy`` is on CUDA), run the decoder
        forward + BCE under ``torch.autocast`` at this dtype.  ``bfloat16``
        is faster and lighter on Ampere+ with negligible quality cost, but
        is NOT bit-identical to the fp32 path.  ``None``
        (default) keeps full fp32.

    Returns
    -------
    z : torch.Tensor
        Optimized latent ``(B, 8, 16, 16, 16)``.
    losses : list of (batch-mean bce, batch-mean prior) per step.
    """
    if gt_occupancy.dim() != 5 or gt_occupancy.shape[1] != 1:
        raise ValueError(
            f"invert_decoder: gt_occupancy must be (B, 1, G, G, G), got "
            f"{tuple(gt_occupancy.shape)}"
        )
    device = gt_occupancy.device
    B = gt_occupancy.shape[0]

    if seed is not None:
        # Bit-parity with the one-at-a-time path: it re-seeds on EVERY call, so
        # every frame genuinely starts from the same z0.  Broadcast one draw --
        # ``randn(B, ...)`` would give each row a different start.
        torch.manual_seed(seed)
        z = torch.randn(1, 8, 16, 16, 16, device=device)
        z = z.expand(B, -1, -1, -1, -1).clone().requires_grad_(True)
    else:
        # Unseeded: B independent draws.  NOTE this is not the same Philox
        # sequence B separate ``randn(1, ...)`` calls would produce.
        z = torch.randn(B, 8, 16, 16, 16, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([z], lr=lr)

    # Stash detached device scalars and sync ONCE after the loop: ``.item()``
    # per step would force 2 device syncs x num_steps x n_frames.
    bce_hist, prior_hist = [], []
    steps = range(num_steps)
    if show_progress:
        try:
            from tqdm import tqdm
            steps = tqdm(steps, desc=progress_desc, leave=False)
        except ImportError:
            pass
    postfix_every = max(1, num_steps // 10)

    for i in steps:
        optimizer.zero_grad()
        with torch.autocast(
            "cuda", dtype=autocast_dtype,
            enabled=autocast_dtype is not None and z.is_cuda,
        ):
            logits = decoder(z)
            # BCE in fp32 regardless: the loss is a mean over 262144 voxels,
            # where bf16's 8-bit mantissa would quantise the accumulation.
            bce = F.binary_cross_entropy_with_logits(
                logits.float(), gt_occupancy,
                reduction="none",
            ).mean(dim=(1, 2, 3, 4))                              # (B,)
        prior = prior_weight * (
            z.mean(dim=(1, 2, 3, 4)).pow(2)
            + (z.var(dim=(1, 2, 3, 4)) - prior_target_var) ** 2
        )                                                          # (B,)
        l2 = (l2_weight * z.pow(2).mean(dim=(1, 2, 3, 4))
              if l2_weight > 0 else z.new_zeros(B))
        # SUM over the batch: d(sum_b L_b)/dz_b == dL_b/dz_b, so every row's
        # gradient is exactly the gradient it would get on its own, and Adam is
        # element-wise -- the batch is numerically equivalent to B lone runs.
        (bce + prior + l2).sum().backward()
        optimizer.step()
        bce_hist.append(bce.detach().mean())
        prior_hist.append(prior.detach().mean())
        if (show_progress and hasattr(steps, "set_postfix")
                and (i % postfix_every == 0 or i == num_steps - 1)):
            steps.set_postfix(bce=f"{bce_hist[-1].item():.4f}",
                              prior=f"{prior_hist[-1].item():.4f}")

    if not bce_hist:  # num_steps=0
        return z.detach(), []
    losses = list(zip(torch.stack(bce_hist).cpu().tolist(),
                      torch.stack(prior_hist).cpu().tolist()))
    return z.detach(), losses


def latent_to_shape_tokens(z: torch.Tensor) -> torch.Tensor:
    """Convert decoder latent ``(B, 8, 16, 16, 16)`` to shape tokens ``(B, 4096, 8)``."""
    return z.permute(0, 2, 3, 4, 1).reshape(-1, 4096, 8)


def shape_tokens_to_latent(tokens: torch.Tensor) -> torch.Tensor:
    """Convert shape tokens ``(B, 4096, 8)`` to decoder latent ``(B, 8, 16, 16, 16)``."""
    return tokens.reshape(-1, 16, 16, 16, 8).permute(0, 4, 1, 2, 3).contiguous()


def compute_iou_per_sample(
    pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.0,
) -> torch.Tensor:
    """Per-sample IoU over a batch of logits/targets ``(B, 1, G, G, G)`` -> ``(B,)``.

    IoU is computed per batch element, so a batch of frames yields one
    per-frame number each rather than a union-over-frames IoU.
    """
    pred_bin = (pred > threshold).float()
    dims = tuple(range(1, pred.dim()))
    inter = (pred_bin * target).sum(dim=dims)
    union = ((pred_bin + target) > 0).float().sum(dim=dims)
    return inter / union.clamp(min=1e-8)


def occupancies_to_shape_tokens(
    occupancy: torch.Tensor,
    decoder: torch.nn.Module,
    num_steps: int = 500,
    lr: float = 0.2,
    prior_weight: float = 0.01,
    prior_target_var: float = 1.0,
    l2_weight: float = 0.0,
    *,
    chunk_size: int = 0,
    autocast_dtype: Optional[torch.dtype] = None,
    show_progress: bool = True,
) -> Tuple[torch.Tensor, List[float]]:
    """Batched occupancy -> shape tokens ``(B, 4096, 8)`` + per-sample IoU.

    The GPU half of :func:`mesh_to_shape_tokens`, split out so a caller can do
    all the (CPU, open3d) voxelisation up front and then run ONE batched Adam
    inversion over every frame.  Tokens come back on ``occupancy.device``.

    ``chunk_size`` bounds peak memory — the decoder backward holds ~1.0 GB of
    activations per sample in fp32.  ``0`` = one batch.  Every sample starts
    from the same seeded ``z0`` and the per-sample losses are independent, so
    chunking does not change the optimisation; see the note in
    :func:`invert_decoder` for why the GPU result still is not bit-equal
    between two chunkings (or between two identical unbatched runs).

    An OOM halves the chunk size and retries — peak memory is reached in the
    first step's backward, so the retry throws away ~100 ms.

    Parameters
    ----------
    occupancy : torch.Tensor
        Binary occupancy ``(B, 1, 64, 64, 64)``, already on the target device.

    Returns
    -------
    tokens : torch.Tensor, shape ``(B, 4096, 8)``
    ious : list of float, per-sample round-trip IoU.
    """
    B = occupancy.shape[0]
    cs = B if chunk_size <= 0 else min(int(chunk_size), B)
    zs: List[torch.Tensor] = []
    ious: List[torch.Tensor] = []

    a = 0
    while a < B:
        occ_c = occupancy[a:a + cs]
        try:
            z_c, _ = invert_decoder(
                decoder, occ_c, num_steps=num_steps, lr=lr,
                prior_weight=prior_weight, prior_target_var=prior_target_var,
                l2_weight=l2_weight,
                show_progress=show_progress, autocast_dtype=autocast_dtype,
                progress_desc=f"Shape inversion [{a}:{a + occ_c.shape[0]}/{B}]",
            )
        except torch.cuda.OutOfMemoryError:
            if cs == 1:
                raise
            n_failed, cs = occ_c.shape[0], max(1, cs // 2)
            # Drop the failed slice's reference BEFORE reclaiming, or its
            # activations stay pinned and the retry OOMs at half the size too.
            del occ_c
            torch.cuda.empty_cache()
            print(f"  [shape-inversion] OOM at chunk_size={n_failed}, "
                  f"retrying at {cs} (~1.0 GB activations/frame)")
            continue  # every remaining chunk shrinks too, not just this one
        with torch.no_grad():
            ious.append(compute_iou_per_sample(decoder(z_c), occ_c))
        zs.append(z_c)
        a += occ_c.shape[0]
        if occupancy.is_cuda:
            torch.cuda.empty_cache()  # hand the peak back before the next chunk

    z = torch.cat(zs, dim=0)
    return latent_to_shape_tokens(z), torch.cat(ious).cpu().tolist()


def mesh_to_shape_tokens(
    mesh_path: str,
    decoder: torch.nn.Module,
    num_steps: int = 500,
    lr: float = 0.2,
    prior_weight: float = 0.01,
    prior_target_var: float = 1.0,
    l2_weight: float = 0.0,
    device: str = "cuda",
    local_rotation = "identity",
    *,
    autocast_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, float]:
    """End-to-end: mesh file → shape tokens ``(4096, 8)`` + IoU.

    Parameters
    ----------
    mesh_path : str
        Path to mesh (OBJ/PLY).
    decoder : nn.Module
        Frozen SparseStructureDecoder (on device).
    num_steps, lr, prior_weight, prior_target_var, l2_weight :
        optimization hyperparams (see :func:`invert_decoder`).
    device : str

    Returns
    -------
    shape_tokens : torch.Tensor, shape ``(4096, 8)``
    iou : float
    """
    occupancy = voxelize_mesh(mesh_path, local_rotation=local_rotation).to(device)
    tokens, ious = occupancies_to_shape_tokens(
        occupancy, decoder, num_steps=num_steps, lr=lr,
        prior_weight=prior_weight, prior_target_var=prior_target_var,
        l2_weight=l2_weight,
        autocast_dtype=autocast_dtype,
    )
    return tokens[0].cpu(), ious[0]  # (4096, 8), float
