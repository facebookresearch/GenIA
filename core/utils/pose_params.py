"""Composable helpers for shared-world-pose pose-parameter management.

Used by the MV shared-world-pose optimization path: one shared set of raw
Stage 1 pose tokens (per object, taken from a reference frame) drives all
per-frame rendered poses via a fixed c2w rebase.

These helpers are kept small and pure so the optimization loop remains
readable. Row-convention rebase:

    M_r3 = c2w_i^{-1} @ c2w_ref                     # R3 convention
    M    = S @ M_r3 @ S   with S=diag(-1,-1,1,1)    # conjugate to P3D
    R_i  = R_ref @ M[:3,:3].T
    t_i  = M[:3,:3] @ t_ref + M[:3,3]
    s_i  = s_ref

Conjugation by ``S`` is needed because ``sequence[fi].c2w`` is stored in R3
convention while the pose ``(R_ref, t_ref)`` lives in the PyTorch3D camera
space used by ``apply_pose_to_gaussian``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch


def build_shared_world_pose_params(
    ref_raw_modalities: Dict[str, torch.Tensor],
    rot_lr: float,
    trans_lr: float,
    scale_lr: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    """Clone the reference frame's raw pose tokens as leaf ``nn.Parameter``s.

    Reads ``6drotation_normalized``, ``translation``, ``scale`` from
    ``ref_raw_modalities`` and returns them as fresh tensors with
    ``requires_grad`` set per learning rate. ``param_groups`` is the list
    suitable for ``torch.optim.Adam``.

    Components with LR ≤ 0 are frozen (``requires_grad=False``, not included
    in ``param_groups``).
    """
    opt_6drot = (
        ref_raw_modalities["6drotation_normalized"]
        .clone().detach().float().to(device)
        .requires_grad_(rot_lr > 0)
    )
    opt_raw_trans = (
        ref_raw_modalities["translation"]
        .clone().detach().float().to(device)
        .requires_grad_(trans_lr > 0)
    )
    opt_raw_scale = (
        ref_raw_modalities["scale"]
        .clone().detach().float().to(device)
        .requires_grad_(scale_lr > 0)
    )
    param_groups: List[Dict[str, Any]] = []
    if rot_lr > 0:
        param_groups.append({"params": [opt_6drot], "lr": rot_lr})
    if trans_lr > 0:
        param_groups.append({"params": [opt_raw_trans], "lr": trans_lr})
    if scale_lr > 0:
        param_groups.append({"params": [opt_raw_scale], "lr": scale_lr})
    return opt_6drot, opt_raw_trans, opt_raw_scale, param_groups


def build_decoded_shared_world_pose_params(
    ref_decoder_input: Dict[str, torch.Tensor],
    rot_lr: float,
    trans_lr: float,
    scale_lr: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    """Clone the reference frame's DECODED pose as leaf ``nn.Parameter``s.

    The decoded-pose counterpart of :func:`build_shared_world_pose_params` for
    the ``optimize_pose_tokens=False`` path: reads the wxyz quaternion
    ``rotation`` and ``translation`` straight from the reference decoder_input
    (no raw tokens / SSI needed), and an isotropic scalar ``scale``. Returns
    ``(opt_q, opt_t, opt_s, param_groups)``; components with LR ≤ 0 are frozen.
    """
    opt_q = (
        ref_decoder_input["rotation"]
        .clone().detach().float().to(device).requires_grad_(rot_lr > 0)
    )
    opt_t = (
        ref_decoder_input["translation"]
        .clone().detach().float().to(device).requires_grad_(trans_lr > 0)
    )
    # Isotropic scalar scale: the raw-token counterpart decodes scale through
    # ``differentiable_pose_decode`` (always isotropic), and downstream consumers
    # assert sx == sy == sz. Optimizing a 3-vector would let the components drift
    # apart and break that invariant.
    opt_s = (
        ref_decoder_input["scale"].clone().detach().float()
        .reshape(-1).mean().reshape(1).to(device).requires_grad_(scale_lr > 0)
    )
    param_groups: List[Dict[str, Any]] = []
    if rot_lr > 0:
        param_groups.append({"params": [opt_q], "lr": rot_lr})
    if trans_lr > 0:
        param_groups.append({"params": [opt_t], "lr": trans_lr})
    if scale_lr > 0:
        param_groups.append({"params": [opt_s], "lr": scale_lr})
    return opt_q, opt_t, opt_s, param_groups


def precompute_c2w_rebases(
    sequence: Any,
    frame_indices: List[int],
    ref_frame: int,
    device: torch.device,
) -> Dict[int, torch.Tensor]:
    """Precompute ``M_i = S @ inv(c2w_i) @ c2w_ref @ S`` as constant 4x4 buffers.

    The ``S = diag(-1, -1, 1, 1)`` conjugation converts the R3-convention c2w
    matrices (as stored in ``sequence[fi].c2w``) into the PyTorch3D convention
    used by the pose tokens (see module docstring).

    Returns ``{frame_idx: M_i (4, 4) float32}``. ``M_ref`` is the identity up
    to numerical noise. Compute once per optimization (c2w is static).
    """
    c2w_ref = torch.as_tensor(sequence[ref_frame].c2w, device=device, dtype=torch.float32)
    S = torch.diag(torch.tensor([-1.0, -1.0, 1.0, 1.0], device=device, dtype=torch.float32))
    out: Dict[int, torch.Tensor] = {}
    for fi in frame_indices:
        c2w_i = torch.as_tensor(sequence[fi].c2w, device=device, dtype=torch.float32)
        M_r3 = torch.linalg.inv(c2w_i) @ c2w_ref
        out[fi] = S @ M_r3 @ S
    return out


def derive_perframe_pose_from_shared_decoded(
    R_ref: torch.Tensor,   # (3, 3) row-convention rotation
    t_ref: torch.Tensor,   # (3,) translation
    s_ref: torch.Tensor,   # (3,) per-axis scale
    M: torch.Tensor,       # (4, 4) = inv(c2w_i) @ c2w_ref
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable rebase: ref pose + fixed c2w transform -> cam_i pose.

    Row-convention, matching ``differentiable_pose_decode`` outputs:

        R_i = R_ref @ M[:3,:3].T
        t_i = M[:3,:3] @ t_ref + M[:3,3]
        s_i = s_ref            # scale is camera-invariant (local space)

    Gradients flow from ``R_i`` / ``t_i`` back into ``R_ref`` / ``t_ref``;
    ``M`` is a constant buffer. Scale is returned unchanged (rigid c2w doesn't
    alter local-space scale).
    """
    M33 = M[:3, :3]
    M3 = M[:3, 3]
    R_i = R_ref @ M33.T
    t_i = M33 @ t_ref + M3
    return R_i, t_i, s_ref


def build_correction(
    rot_lr: float,
    scale_lr: float,
    trans_lr: float,
    device: torch.device,
    *,
    group_extra: Any = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    """The ONE Sim(3) a ``correction_granularity`` rides on: ``(aa, log_ds, dt, param_groups)``.

    Axis-angle, log-scale and translation, all zero (identity) and each free only if its
    learning rate is positive.  Groups are appended **rotation, scale, translation**.

    ``group_extra`` is an optional ``component -> dict`` callable whose result is merged
    into that component's group; the component is ``"aa"``, ``"log_ds"`` or ``"dt"``.  A
    callable rather than one dict because FINETUNE needs a DIFFERENT ``name`` per
    component (``correction_granularity_aa`` ...) alongside a shared ``tag``, and those keys
    drive the pose freeze and gradient logging.

    Shared by the photometric and FINETUNE solvers that read ``correction_granularity``.
    ICP builds its own: its optimiser takes parallel ``params``/``lrs`` lists rather than
    group dicts, and its per-timestamp leaves are batched ``(n, 3)``.
    """
    aa = torch.zeros(3, device=device, requires_grad=rot_lr > 0)
    log_ds = torch.zeros(1, device=device, requires_grad=scale_lr > 0)
    dt = torch.zeros(3, device=device, requires_grad=trans_lr > 0)
    param_groups: List[Dict[str, Any]] = []
    for _name, _p, _lr in (("aa", aa, rot_lr), ("log_ds", log_ds, scale_lr),
                           ("dt", dt, trans_lr)):
        if _lr > 0:
            param_groups.append({"params": [_p], "lr": _lr,
                                 **(group_extra(_name) if group_extra else {})})
    return aa, log_ds, dt, param_groups


def freeze_params(param_groups: List[Dict[str, Any]],
                  params: Any) -> List[Dict[str, Any]]:
    """Freeze ``params`` and return ``param_groups`` without the groups that held them.

    Both halves matter: clearing ``requires_grad`` stops the gradient, and dropping the
    group stops the optimiser carrying dead state.

    This is the freeze MECHANISM, which is uniform; :func:`natives_to_freeze` is the
    freeze POLICY, which is what the five solvers must agree on.  They are separate
    because the solvers hold their natives in different shapes but all express "held" the
    same way.
    """
    ids = {id(p) for p in params}
    for g in param_groups:
        for p in g["params"]:
            if id(p) in ids:
                p.requires_grad_(False)
    return [g for g in param_groups if not any(id(p) in ids for p in g["params"])]


def natives_to_freeze(mode: str, scale_control: str = "perframe") -> str:
    """Which NATIVE pose params a mode freezes: ``"all"`` | ``"scale"`` | ``"none"``.

    * ``per_frame`` — no correction exists; the natives are the whole model. ``"none"``.
    * ``shared``    — the correction is the only thing that moves, which is what makes the
      fit well-posed and lets the result be read as "the systematic error". ``"all"``.
    * ``both``      — natives AND correction free at once, so ``scale_control`` decides:
      ``"shared"`` (the shipped default) freezes the native SCALE, leaving size to the one
      shared transform; ``"perframe"`` leaves everything free.

    **Why scale is singled out.** ``both`` is gauge-ambiguous by construction and that is
    tolerated for rotation and translation — the composite is what gets written back, so
    the split never leaves the solver.  Scale is the case where it should not be: the
    object has ONE size, and a per-frame transform that models it lets the optimiser
    breathe the object frame-to-frame while the shared transform chases the residual.

    ``scale_control`` defaults to ``"perframe"`` HERE while the CONFIG defaults to
    ``"shared"``: a caller that does not pass the knob leaves the natives free, and only a
    resolved config freezes the native scale.

    A pure policy function, deliberately: the solvers hold their natives in genuinely
    different shapes (dicts keyed by object, a bare triple, a ``tag``-carrying group list),
    so routing them through one *freezing* signature would be a worse abstraction than the
    duplication it removes.  What must not diverge between five solvers is the DECISION,
    so that is what lives here; each site applies it to its own parameters.
    """
    if mode == "shared":
        return "all"
    if mode == "both" and scale_control == "shared":
        return "scale"
    return "none"


__all__ = [
    "build_correction",
    "freeze_params",
    "natives_to_freeze",
    "build_shared_world_pose_params",
    "build_decoded_shared_world_pose_params",
    "precompute_c2w_rebases",
    "derive_perframe_pose_from_shared_decoded",
]
