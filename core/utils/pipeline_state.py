# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Pipeline state management and caching.

Provides the :class:`PipelineState` dataclass that carries mutable state
between pipeline blocks, plus serialization / cache helpers for resuming
mid-pipeline.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

from .frame_key import (
    FrameKey, as_frame_key, frame_key_sort_key, group_by_frame,
)


# =====================================================================
# Pose rebase (shared-world-pose invariant helper)
# =====================================================================


def derive_camera_space_pose_from_reference(R_ref, t_ref, c2w_ref, c2w_i):
    """Derive frame ``i``'s PyTorch3D camera-space ``(R_i, t_i)`` from the
    reference frame's pose ``(R_ref, t_ref)`` via the known camera extrinsics::

        M    = S @ (c2w_i^{-1} @ c2w_ref) @ S      with S=diag(-1,-1,1,1)
        R_i  = R_ref @ M[:3,:3].T                  # row convention, P3D
        t_i  = M[:3,:3] @ t_ref + M[:3,3]

    The ``S`` conjugation maps the R3 ``c2w`` convention into the P3D camera
    space used by the stored / decoded pose (see
    :func:`rebase_object_poses_from_reference` for the rationale). ``S`` is
    self-inverse.

    Pure tensor ops — **differentiable** in ``R_ref``/``t_ref`` (the ``c2w_*``
    are constants), so this is shared by both the post-ODE rebase (detached
    inputs) and the in-ODE gsplat rendering-guidance shared-world path (grad
    flows back to the reference pose token).
    """
    import torch

    device, dtype = R_ref.device, R_ref.dtype
    c2w_ref = torch.as_tensor(c2w_ref, device=device, dtype=dtype)
    c2w_i = torch.as_tensor(c2w_i, device=device, dtype=dtype)
    S = torch.diag(torch.tensor([-1.0, -1.0, 1.0, 1.0], device=device, dtype=dtype))
    M = S @ (torch.linalg.inv(c2w_i) @ c2w_ref) @ S
    M33, M3 = M[:3, :3], M[:3, 3]
    R_i = R_ref @ M33.T
    t_i = M33 @ t_ref + M3
    return R_i, t_i


def rebase_object_poses_from_reference(
    entries: list,
    perframe_raw_modalities_obj: dict,
    sequence: Any,
    ref_key: FrameKey,
    *,
    scope: str = "all",
) -> int:
    """Collapse per-frame poses onto a shared local->world transform, along ``scope``'s axis.

    ``scope`` defaults to ``"all"``; the per-timestamp mode is an opt-in.  Every call site
    states it explicitly, because the two answers are both silently plausible and the wrong
    one produces a well-formed pose that is simply wrong:

    - ``"timestamp"`` -- group by ``FrameKey.frame``: one object at one timestamp has a
      single world placement seen by its views, while two timestamps may legitimately
      DIFFER.  This is pose_init's shared-world rebase, where the object's motion lives in
      the per-frame Sim(3), so reducing across time would erase it.  Each timestamp's
      reference is the resolved ``ref_key``'s VIEW at that timestamp (the pose-velocity
      broadcast lead in ``core/inference.py``), falling back to the lowest view
      present when that view is absent there.
    - ``"all"`` -- one group over every entry.  This is
      ``refinement.refine_poses_shared_world_for_sequence``'s write-back, where the motion
      lives in the WARP (or there is one timestamp) and the Sim(3) genuinely IS shared
      across the sequence, so the one fitted pose must reach every frame.

    On a single timestamp the two are identical; they differ only on dynamic data.

    Within a timestamp, the reference's pose ``(R_ref, t_ref, s_ref)`` derives every other
    view's camera-space pose via the row-convention rebase formula::

        M_r3  = c2w_i^{-1} @ c2w_ref               # 4x4 in R3 convention
        M     = S @ M_r3 @ S   with S=diag(-1,-1,1,1)   # conjugate into P3D
        R_i   = R_ref @ M[:3,:3].T                 # row convention, P3D
        t_i   = M[:3,:3] @ t_ref + M[:3,3]
        s_i   = s_ref                              # within the timestamp only

    ``s_i = s_ref`` is deliberately per timestamp: forcing one timestamp's size onto another
    is the same error as forcing its position.  Unifying size across time is the refit's
    ``global_scale`` consensus, later and by agreement.

    The conjugation by ``S`` is required because ``sequence[fk].c2w`` is in R3
    convention (standard computer-vision c2w acting on R3 camera-space points)
    while ``(R_ref, t_ref)`` live in the PyTorch3D camera space used by
    ``apply_pose_to_gaussian``. Without it, the rebase is only correct when
    every c2w is identity.

    Parameters
    ----------
    entries : list of (FrameKey, decoder_input_dict)
        From ``tokens_by_object[obj_idx]`` — mutated in place.
    perframe_raw_modalities_obj : dict
        From ``perframe_raw_modalities[obj_idx]`` — may be empty. Updated in
        place when entries carry raw_ss_modalities. Keyed by FrameKey.
    sequence : Sequence
        Provides ``sequence[fk].c2w`` per FrameKey.
    ref_key : FrameKey
        Reference key; its VIEW selects each timestamp's reference, whose pose defines that
        timestamp's world placement as ``c2w_ref @ P_ref``.

    Returns
    -------
    int
        Number of non-reference frames rebased, summed over timestamps.
    """
    ref_key = as_frame_key(ref_key)
    by_key = {as_frame_key(fk): di for fk, di in entries}
    if ref_key not in by_key:
        raise ValueError(f"ref_key={ref_key} not found in entries")

    if scope == "all":
        groups = {None: sorted(by_key, key=frame_key_sort_key)}
    elif scope == "timestamp":
        groups = group_by_frame(by_key)
    else:
        raise ValueError(f"scope must be 'timestamp' or 'all', got {scope!r}")

    n_updated = 0
    for _f, group in groups.items():
        # For "all" the reference is the caller's, verbatim.  Per timestamp it comes from
        # the one shared rule below, so a second consumer of that rule (FINETUNE's
        # shared-world mode) cannot drift from this one.
        grp_ref = ref_key if _f is None else timestamp_reference_key(by_key, ref_key, _f, group)
        n_updated += _rebase_one_timestamp(
            grp_ref, group, by_key, perframe_raw_modalities_obj, sequence)
    return n_updated


def timestamp_reference_key(by_key, ref_key, frame, group=None):
    """Which view stands for one TIMESTAMP, given the object's resolved reference.

    The resolved reference's own VIEW where that view is present at this timestamp, else
    the lowest view present.  Named and public because two callers must agree on it:
    :func:`rebase_object_poses_from_reference` (``scope="timestamp"``) and FINETUNE's
    shared-world parameterization, which derives each timestamp's views from exactly this
    frame.  ``group`` is that timestamp's keys already view-sorted, if the caller has it.
    """
    ref_key = as_frame_key(ref_key)
    same_view = FrameKey(frame, ref_key.view)
    if same_view in by_key:
        return same_view
    if group:
        return group[0]
    return min((k for k in by_key if as_frame_key(k).frame == frame),
               key=frame_key_sort_key)


def _rebase_one_timestamp(ref_key, group, by_key, perframe_raw_modalities_obj, sequence) -> int:
    """One timestamp's views, derived from ``ref_key``.  Returns non-reference frames done.

    Split out so the per-timestamp loop above reads as the grouping decision it is, and this
    stays the plain row-convention rebase.
    """
    import torch
    from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix

    from genia.core.utils.pose_token_gt import camera_pose_to_raw_tokens

    ref_di = by_key[ref_key]

    # Nothing has DECODED a pose into the reference yet: there is no pose to derive the
    # others from, so this is "nothing to do" (the same 0 `rebase_perframe_from_reference`
    # returns for an empty entry list), not an error.  Reached whenever the rebase runs
    # before pose_init -- a cache cut at appearance_init, or a shape-cache hit -- where
    # the entries exist (masks, Gaussians) but carry no pose.  Deliberately checked on the
    # REFERENCE rather than "any frame": `resolve_reference_frame` picks the canonical
    # frame or `min(entries)` without consulting pose content, so a mix of posed and
    # pose-less frames can resolve to a pose-less reference.
    # NOT merged into the ValueError above: a MISSING reference key is a mis-resolved
    # reference and must stay loud.
    # ``.get(...) is None``, not ``in``: a producer may leave the key present and None
    # (``pose_refit`` guards for exactly that), and ``in`` would pass it through to
    # ``.reshape`` and raise AttributeError.
    if any(ref_di.get(k) is None for k in ("rotation", "translation", "scale")):
        # This TIMESTAMP has nothing to derive from; the others may still.  Per group, not
        # per object, or one pose-less timestamp in a mixed set would silently abandon the
        # rest.
        return 0

    # Reference pose: stored rotation is a wxyz quaternion; t/s are (3,)-like.
    q_ref = ref_di["rotation"].reshape(-1)[:4].float()
    t_ref = ref_di["translation"].reshape(-1)[:3].float()
    s_ref = ref_di["scale"].reshape(-1).float()
    if s_ref.numel() == 1:
        s_ref = s_ref.expand(3).contiguous()
    device = q_ref.device

    # (3,3) row-convention rotation via PyTorch3D (matches differentiable_pose_decode).
    R_ref = quaternion_to_matrix(q_ref.unsqueeze(0)).squeeze(0)

    c2w_ref = torch.as_tensor(sequence[ref_key].c2w, device=device, dtype=torch.float32)

    n_updated = 0
    for fk in group:
        di = by_key[fk]
        c2w_i = torch.as_tensor(sequence[fk].c2w, device=device, dtype=torch.float32)
        # Row-convention rebase via the shared differentiable helper (here with
        # detached inputs, so the result is detached too).
        R_i, t_i = derive_camera_space_pose_from_reference(R_ref, t_ref, c2w_ref, c2w_i)
        s_i = s_ref.clone()
        q_i = matrix_to_quaternion(R_i.unsqueeze(0)).squeeze(0)

        # Match storage shapes from decode_perframe_poses_from_raw: (1, 4) / (1, 3) / (1, 3).
        di["rotation"] = q_i.unsqueeze(0).detach()
        di["translation"] = t_i.unsqueeze(0).detach()
        di["scale"] = s_i.unsqueeze(0).detach()

        # Refresh raw layout tokens via SSI inversion so future decodes round-trip.
        pf_raw = perframe_raw_modalities_obj.get(fk)
        ps = pf_raw.get("pointmap_scale") if pf_raw else di.get("pointmap_scale")
        psh = pf_raw.get("pointmap_shift") if pf_raw else di.get("pointmap_shift")
        dsf = (pf_raw.get("downsample_factor", 1.0) if pf_raw
               else di.get("downsample_factor", 1.0))
        if ps is not None and psh is not None:
            raw_tokens = camera_pose_to_raw_tokens(R_i, t_i, s_i, ps, psh, dsf)
            if pf_raw is not None:
                raw_mods = pf_raw.setdefault("raw_ss_modalities", {})
                for k, v in raw_tokens.items():
                    raw_mods[k] = v.detach()
            if "raw_ss_modalities" in di:
                for k, v in raw_tokens.items():
                    di["raw_ss_modalities"][k] = v.detach()

        if fk != ref_key:
            n_updated += 1

    return n_updated


def resolve_reference_frame(
    entries: list,
    canon_frame_per_object: dict,
    obj_idx: int,
    ref_key: Optional[FrameKey] = None,
) -> FrameKey:
    """Determine the reference FrameKey for shared-world-pose rebase.

    Priority: explicit ``ref_key`` arg → ``canon_frame_per_object[obj_idx]`` →
    smallest frame_key (by ``(view, frame)``) in ``entries``.
    """
    if ref_key is not None:
        return as_frame_key(ref_key)
    fallback = canon_frame_per_object.get(obj_idx)
    if fallback is not None:
        return as_frame_key(fallback)
    return min((fk for fk, _ in entries), key=frame_key_sort_key)


# =====================================================================
# Pipeline state
# =====================================================================

class _NoneMissingDict(dict):
    """Canonical store ``{obj_idx: value}`` whose real-only invariant is
    self-enforcing: subscripting an absent object returns ``None`` (instead of
    raising ``KeyError``), and assigning ``None`` drops the key.

    The store therefore never holds ``None`` placeholders, so iteration,
    truthiness, ``in`` and ``len`` all reflect exactly the objects that have a
    real canonical representation, while ``state.canonical_gaussians[obj_idx]``
    is ``None`` when that object has none.  This is what lets a pipeline be
    "per-frame only" with no flag: a pipeline whose active blocks never populate
    these stores simply leaves them empty (see ``PipelineState.has_canonical``).
    """

    def __missing__(self, key):
        return None

    def __setitem__(self, key, value):
        # Assigning None drops the key — keeps the store real-only without
        # any caller having to special-case invalidation / "no canonical".
        if value is None:
            self.pop(key, None)
        else:
            super().__setitem__(key, value)

    @classmethod
    def of(cls, d=None) -> "_NoneMissingDict":
        """Wrap *d* into a real-only store, dropping any ``None`` values (the
        ``dict`` constructor bypasses ``__setitem__``, so strip here).  Used for
        bulk construction: ``__post_init__`` (fresh + cache-loaded state) and
        ``SAM3DState.apply_finetuning_results`` reassignment."""
        return cls({k: v for k, v in (d or {}).items() if v is not None})


@dataclass
class PipelineState:
    """Mutable state flowing between pipeline blocks.

    Cached decoded values
    ---------------------
    Three fields are **cached decoded values** derived from SLAT tokens:

    - ``canonical_gaussians`` — decoded from ``canonical_slats``
    - ``perframe_gaussians`` — decoded from per-frame SLATs in ``tokens_by_object``
    - ``canonical_shape_coords`` — voxel coords from ``canonical_slats``

    These fields must **never be set directly from outside this class**.
    Use the mutation methods (``set_canonical_slat``, ``set_all_perframe_slats``,
    etc.) which automatically re-decode the cached values when the source
    tokens change.  If tokens are removed, cached values are removed too.

    Pose-only changes (rotation, translation, scale) do NOT require
    re-decoding because the Gaussian decoder depends only on SLAT
    features/coords, not poses — poses are applied at render time.
    """

    tokens_by_object: dict = field(default_factory=dict)
    """Per-object per-frame tokens: {obj_idx: [(FrameKey, decoder_input), ...]}.
    The list is sorted by ``frame_key_sort_key`` (i.e. ``(view, frame)``)."""

    perframe_gaussians: Optional[dict] = None
    """Per-frame decoded Gaussians: {obj_idx: {FrameKey: Gaussian}}.
    Cached — populated by mutation methods, never set externally."""

    canonical_gaussians: dict = field(default_factory=_NoneMissingDict)
    """Per-object canonical Gaussian: {obj_idx: Gaussian}.
    A :class:`_NoneMissingDict`: holds only real entries (invalidation pops),
    and ``canonical_gaussians[obj_idx]`` is ``None`` when absent.
    Cached — populated by mutation methods, never set externally."""

    canonical_slats: dict = field(default_factory=_NoneMissingDict)
    """Per-object canonical SLAT: {obj_idx: SparseTensor}.
    A :class:`_NoneMissingDict`: holds only real entries (invalidation pops),
    and ``canonical_slats[obj_idx]`` is ``None`` when absent."""

    canon_frame_per_object: dict = field(default_factory=dict)
    """Per-object canonical FrameKey: {obj_idx: FrameKey | None}"""

    canonical_shape_coords: dict = field(default_factory=dict)
    """Transient per-object shape voxel coords for fallback rendering.
    Cached — decoded from the shape tokens (``SAM3DState.canonical_raw_modalities``).

    Structure: {obj_idx: np.ndarray (N, 3)} — integer xyz grid positions.
    Present only between POSE_INIT (shape update invalidates canonical
    Gaussians) and APPEARANCE_INIT canonical (which rebuilds
    the canonical SLAT/Gaussians).
    Used by visualization to render XYZ-colored voxel mesh when canonical
    Gaussians are unavailable.
    """

    canonical_mesh_verts: dict = field(default_factory=dict)
    """Per-object canonical mesh vertices in canonical-norm ``[-0.5, 0.5]³``.

    Structure: ``{obj_idx: torch.Tensor (V, 3) float32}``.

    The raw per-vertex GT signal — same source as the per-frame voxel
    correspondence but at native mesh resolution instead of the voxel-resolution
    binning.  Populated by ``GT_SHAPES_INVERSION`` in any per-frame mode,
    alongside the per-voxel field.

    Consumed by the deformation warp (KNN+IDW + SO(3) blend at decoded primitive
    positions).

    Lifecycle: cleared together with ``canonical_mesh_faces``,
    ``canonical_mesh_per_frame_verts``, and ``canonical_mesh_per_frame_rotations``
    in ``invalidate_canonical`` (when the GT canonical voxel set changes —
    the mesh source for the new shape may differ).
    Crucially **preserved** across ``SAM3DState.apply_finetuning_results``: FINETUNE
    only mutates SLAT features (and optionally LoRA / DC offsets / poses),
    never the underlying GT mesh, so the per-vertex Φ + R remain valid
    for the post-FINETUNE keyframes-video / decoded-viz warp paths.
    """

    canonical_mesh_faces: dict = field(default_factory=dict)
    """Per-object canonical mesh face indices.

    Structure: ``{obj_idx: torch.Tensor (F, 3) int64}``.  Topology is fixed
    across frames (the ActionMesh contract), so a single per-object face
    array suffices.  Populated alongside ``canonical_mesh_verts``.
    """

    canonical_mesh_per_frame_verts: dict = field(default_factory=dict)
    """Per-object per-frame deformed mesh vertices in frame-i self-norm.

    Structure: ``{obj_idx: {frame_int: torch.Tensor (V, 3) float32}}``.
    Vertex-aligned with ``canonical_mesh_verts[obj_idx]`` (fixed topology).
    At the canonical frame, this equals ``canonical_mesh_verts[obj_idx]``
    bit-for-bit (identity invariance is by construction).

    Stored in **frame-i SELF-normalisation** (each frame's mesh independently
    centered + scaled to ``[-0.5, 0.5]``), matching what Stage-1
    ``(R_i, t_i, s_i)`` operates on.  Consumed by Stage-2 rendering
    guidance as the per-frame anchor source.

    Lifecycle: identical to ``canonical_mesh_verts``.
    """

    canonical_mesh_per_frame_rotations: dict = field(default_factory=dict)
    """Per-object per-frame **per-vertex local rigid rotation** ``R_v[i]``.

    Structure: ``{obj_idx: {frame_int: torch.Tensor (V, 3, 3) float32}}``.
    Computed at ``GT_SHAPES_INVERSION`` time via vertex-kNN Kabsch on the
    fixed-topology vertex correspondences: for each canonical vertex ``v``,
    the 3×3 ``R_v[i]`` is the best rigid rotation aligning the vertex's
    spatially-nearest neighbours' canonical positions to the same neighbour
    indices in frame-i (both centred at ``v``).  Identity at the canonical
    frame.  All entries are guaranteed SO(3) (reflection-fix applied).

    Consumed by Stage-2 ``appearance_init=canonical_unified`` rendering guidance
    (Gaussian-quat update via SO(3)-blended ``R_p`` over the K-NN of each
    decoded primitive).

    Lifecycle: identical to ``canonical_mesh_verts``.
    """

    gt_canonical_shape_coords: dict = field(default_factory=dict)
    """Per-object GT-mesh canonical voxel grid (NOT a cached decoded value).

    Structure: ``{obj_idx: np.ndarray (L, 3)}`` — integer xyz grid positions
    of the voxelised GT mesh.

    Populated ONCE by ``GT_SHAPES_INVERSION``:

    - ``mode='global'``: GT mesh voxelisation.
    - any per-frame mode: canonical-frame voxel grid (the same grid that
      Φ/R are row-aligned to).

    Stable across the rest of the pipeline — never popped by
    ``set_canonical_slat`` or ``SAM3DState.set_canonical_raw_modalities``.  Distinct
    from ``canonical_shape_coords`` (transient cache decoded from current
    shape tokens).
    """

    lora_state_dicts: dict = field(default_factory=dict)
    """Per-object LoRA state dicts from FINETUNE.

    Structure: ``{obj_idx: {"lora_rank": int, "lora_alpha": float,
    "state_dict": {str: Tensor}}}``

    Present only after FINETUNE when ``lora_decoder=True``.  Used by
    ``set_canonical_slat`` for re-decoding with the LoRA decoder.
    Serialized automatically via ``torch.save`` (nested dicts of Tensors).
    """

    pipeline_obj: Any = field(default=None, repr=False)
    """SAM3D pipeline for decoding SLATs → Gaussians. Set once after
    construction.  Excluded from serialization (transient)."""

    def __post_init__(self):
        # Normalize the canonical stores into the real-only `_NoneMissingDict`
        # on every construction path — fresh state and cache-loaded state
        # alike (`load_pipeline_cache` builds via `PipelineState(**kwargs)`,
        # and a cache may hold plain dicts carrying `None` placeholders).
        self.canonical_gaussians = _NoneMissingDict.of(self.canonical_gaussians)
        self.canonical_slats = _NoneMissingDict.of(self.canonical_slats)

    @property
    def has_canonical(self) -> bool:
        """True iff some active block has built a real canonical object.

        There is no configuration flag for this: a pipeline is "per-frame
        only" precisely when its active blocks never populate the canonical
        stores (e.g. with every canonical block disabled).  Canonical
        consumers (FINAL outputs, eval, exports) branch on this."""
        return bool(self.canonical_slats) or bool(self.canonical_gaussians)

    def pose_carries_placement(self, obj_idx: int) -> bool:
        """Does this object's Sim(3) place it, or did the reconstruction bake placement in?

        Usually the two are decoupled: an object-local canonical (origin-centred, roughly
        unit scale, straight out of the SLAT decoder) plus a real per-frame Sim(3) that
        places it.  A world-space reconstruction instead lives in the anchor camera's frame
        at metric scale and carries an IDENTITY Sim(3), so the object's position AND size
        live in the asset itself.

        Derived from the poses rather than declared by a flag, like :attr:`has_canonical`.

        Consumers: the canonical renderers, which orbit the ORIGIN at a fixed distance and
        so only frame an object that is actually there (see ``rendering.auto_frame_canonical``
        for the alternative framing).

        Note: this reads the emitted pose, so a world-space reconstruction that is put
        through a pose-refine block perturbs the identity and reports True again.

        An object with no transforms reports True, so it keeps the fixed framing rather
        than a guess derived from its geometry.
        """
        import torch

        entries = self.tokens_by_object.get(obj_idx)
        if not entries:
            return True
        for _fk, pose in entries:
            if not {"rotation", "translation", "scale"} <= pose.keys():
                return True                      # skeleton entry — main.py's ``(fi, {})``
            rot = pose["rotation"].detach().reshape(-1)
            identity = torch.zeros_like(rot)
            identity[0] = 1.0
            if not (torch.allclose(rot, identity)
                    and torch.allclose(pose["translation"].detach(),
                                       torch.zeros_like(pose["translation"]))
                    and torch.allclose(pose["scale"].detach(),
                                       torch.ones_like(pose["scale"]))):
                return True
        return False

    # -----------------------------------------------------------------
    # Per-frame degradation views (canonical, else frame-0 per-frame)
    # -----------------------------------------------------------------

    @property
    def canonical_gaussians_with_fallback(self) -> dict:
        """Canonical Gaussians, falling back to each object's earliest
        per-frame Gaussian when no canonical is populated.

        For per-frame-only pipelines with no active canonical blocks, this
        exposes a frame-0 "trivial canonicalization" so
        downstream final outputs (mesh / scene renders / poses.json) can be
        produced uniformly.  Callers that need to distinguish real from
        fallback canonical should test ``has_canonical`` / read
        ``canonical_gaussians``.
        """
        if self.canonical_gaussians:
            return dict(self.canonical_gaussians)
        if not self.perframe_gaussians:
            return {}
        result = {}
        for oi, per_frame in self.perframe_gaussians.items():
            if not per_frame:
                continue
            result[oi] = per_frame[min(per_frame.keys())]
        return result

    # NB: there is no ``canonical_slats_with_fallback``.  Unlike Gaussians,
    # the SLAT store needs no frame-0 fallback — the
    # ``canonical_gaussians.keys() <= canonical_slats.keys()`` invariant
    # guarantees it is populated whenever any canonical exists, so consumers
    # read ``canonical_slats`` directly.

    # -----------------------------------------------------------------
    # Canonical SLAT mutation methods
    # -----------------------------------------------------------------

    def set_canonical_slat(self, obj_idx: int, slat) -> None:
        """Set canonical SLAT and re-decode cached Gaussian.

        Uses the LoRA decoder when ``lora_state_dicts`` has an entry for
        this object.  Clears ``canonical_shape_coords`` for this object.
        """
        self.canonical_slats[obj_idx] = slat

        lora_info = self.lora_state_dicts.get(obj_idx)
        _has_lora = bool(lora_info and lora_info.get("state_dict"))
        if _has_lora and self.pipeline_obj is not None:
            from genia.core.utils.slat_decode import decode_with_lora
            self.canonical_gaussians[obj_idx] = decode_with_lora(
                self.pipeline_obj, slat, lora_info,
            )
        else:
            from genia.core.utils.slat_decode import redecode_slat
            decoded = redecode_slat(self.pipeline_obj, slat, formats=["gaussian"])
            self.canonical_gaussians[obj_idx] = decoded["gaussian"][0]

        self.canonical_shape_coords.pop(obj_idx, None)

    def invalidate_canonical(self, obj_idx: int,
                             shape_coords=None) -> None:
        """Drop the canonical SLAT and Gaussian for an object (shape changed).

        Assigning ``None`` removes the keys (the ``_NoneMissingDict`` store is
        real-only), so a subsequent ``canonical_slats[obj_idx]`` reads ``None``.

        Does NOT clear ``SAM3DState.canonical_poses`` — the pose remains valid even
        when the SLAT is dropped (shape changed but pose is still
        meaningful).

        Clears the per-canonical-mesh-vertex deformation fields
        (``canonical_mesh_*``) **iff the voxel-grid row order changes** (i.e., the
        new ``shape_coords`` differ from the current, or coords are popped
        entirely).  Preserves them when the new ``shape_coords`` are element-wise
        equal to the existing entry — this keeps the GT-anchored fields alive
        across same-grid canonical updates.

        Parameters
        ----------
        shape_coords : np.ndarray (N, 3), optional
            Voxel grid coordinates for fallback visualization.
        """
        import numpy as np

        self.canonical_slats[obj_idx] = None
        self.canonical_gaussians[obj_idx] = None

        old_coords = self.canonical_shape_coords.get(obj_idx)
        if shape_coords is not None:
            coords_changed = (
                old_coords is None
                or old_coords.shape != shape_coords.shape
                or not np.array_equal(old_coords, shape_coords)
            )
            self.canonical_shape_coords[obj_idx] = shape_coords
        else:
            coords_changed = old_coords is not None
            self.canonical_shape_coords.pop(obj_idx, None)

        if coords_changed:
            self.canonical_mesh_verts.pop(obj_idx, None)
            self.canonical_mesh_faces.pop(obj_idx, None)
            self.canonical_mesh_per_frame_verts.pop(obj_idx, None)
            self.canonical_mesh_per_frame_rotations.pop(obj_idx, None)

    # -----------------------------------------------------------------
    # Per-frame SLAT mutation methods
    # -----------------------------------------------------------------

    def set_all_perframe_slats(self, obj_idx: int, slats_by_frame: dict) -> None:
        """Replace per-frame SLATs for all frames of *obj_idx* and re-decode
        all cached Gaussians.

        Parameters
        ----------
        slats_by_frame : dict
            ``{FrameKey: SparseTensor}`` — new SLAT for each frame. Bare-int
            keys are coerced to ``FrameKey(int, 0)``.
        """
        from genia.core.utils.slat_decode import redecode_slat

        # Coerce keys to FrameKey for consistent lookup.
        slats_by_frame = {as_frame_key(k): v for k, v in slats_by_frame.items()}
        for entry_fk, di in self.tokens_by_object[obj_idx]:
            if entry_fk in slats_by_frame:
                di["decoder_input_slat"] = slats_by_frame[entry_fk]
        if self.perframe_gaussians is None:
            self.perframe_gaussians = {}
        self.perframe_gaussians[obj_idx] = {}
        for entry_fk, di in self.tokens_by_object[obj_idx]:
            decoded = redecode_slat(
                self.pipeline_obj, di["decoder_input_slat"], formats=["gaussian"],
            )
            self.perframe_gaussians[obj_idx][entry_fk] = decoded["gaussian"][0]

    def rebase_perframe_from_reference(
        self,
        obj_idx: int,
        sequence: Any,
        ref_key: Optional[FrameKey] = None,
        *,
        scope: str = "all",
    ) -> int:
        """Collapse per-frame poses of ``obj_idx`` onto one shared local->world
        transform defined by the reference frame's pose composed with its c2w.

        Thin wrapper around :func:`rebase_object_poses_from_reference`; see it for what
        ``scope`` selects and why every caller states it anyway. When
        ``ref_key`` is ``None``, falls back to ``canon_frame_per_object[obj_idx]``
        and then to the smallest FrameKey (by ``(view, frame)``) in
        ``tokens_by_object[obj_idx]``.

        Returns the number of non-reference frames rebased. Mutates both
        ``tokens_by_object[obj_idx][fk]`` and, on a :class:`genia.core.state.SAM3DState`,
        ``perframe_raw_modalities[obj_idx][fk]`` in place.
        """
        entries = self.tokens_by_object.get(obj_idx, [])
        if not entries:
            return 0
        resolved_ref = resolve_reference_frame(
            entries, self.canon_frame_per_object, obj_idx, ref_key,
        )
        # ``.get`` (not ``setdefault``) so that calling rebase on an object with
        # no prior ``perframe_raw_modalities`` entry does not seed an empty dict
        # in state.  The base state has no snapshot.
        pf_raw_obj = getattr(self, "perframe_raw_modalities", {}).get(obj_idx, {})
        return rebase_object_poses_from_reference(
            entries, pf_raw_obj, sequence, resolved_ref, scope=scope,
        )

    def invalidate_perframe_slats(self, obj_idx: int) -> None:
        """Remove cached per-frame SLATs and Gaussians for *obj_idx*.

        Clears ``decoder_input_slat`` from each frame's decoder_input dict
        and drops the whole per-frame Gaussian cache.  Call before
        re-running Stage 2 on new shape coords.

        The Gaussian cache is dropped to ``None`` (not popped per-object):
        ``ensure_perframe_gaussians`` early-returns on any non-``None``
        cache, so a partial dict would never re-decode the popped object
        AND would violate ``check_consistency``'s "non-None ⇒
        complete" invariant when other objects' SLATs were invalidated.
        """
        if obj_idx in self.tokens_by_object:
            for _, di in self.tokens_by_object[obj_idx]:
                di.pop("decoder_input_slat", None)
        self.perframe_gaussians = None

    def ensure_perframe_gaussians(self) -> None:
        """Decode per-frame Gaussians from current SLATs if not already cached."""
        if self.perframe_gaussians is not None:
            return
        from genia.core.utils.slat_decode import redecode_slat

        self.perframe_gaussians = {}
        any_decoded = False
        for obj_idx, tokens_list in self.tokens_by_object.items():
            obj_gs = {}
            for fid, di in tokens_list:
                slat = di.get("decoder_input_slat")
                if slat is None:
                    continue  # No SLAT for this frame
                decoded = redecode_slat(
                    self.pipeline_obj, slat,
                    formats=["gaussian"],
                )
                obj_gs[fid] = decoded["gaussian"][0]
            if obj_gs:
                self.perframe_gaussians[obj_idx] = obj_gs
                any_decoded = True
        if not any_decoded:
            self.perframe_gaussians = None

    def check_consistency(self) -> None:
        """Assert PipelineState Gaussians match their token sources.

        Checks structural consistency: Gaussian dicts have the same FrameKey
        keys as the token dicts they were decoded from. Also asserts every
        per-frame key is a ``FrameKey`` (catches bare-int keys).  A state
        subclass extends it with its own invariants.
        """
        if self.perframe_gaussians is not None:
            for obj_idx in self.tokens_by_object:
                frame_keys = {fk for fk, _ in self.tokens_by_object[obj_idx]}
                for fk in frame_keys:
                    assert isinstance(fk, FrameKey), (
                        f"tokens_by_object[{obj_idx}] has non-FrameKey key: "
                        f"{fk!r} (type {type(fk).__name__})"
                    )
                assert obj_idx in self.perframe_gaussians, \
                    f"perframe_gaussians missing obj {obj_idx}"
                assert set(self.perframe_gaussians[obj_idx].keys()) == frame_keys, \
                    f"perframe_gaussians frame mismatch for obj {obj_idx}"
        # Each canonical Gaussian must have a corresponding canonical SLAT.  Both
        # stores are real-only (no None placeholders); for a per-frame-only run no
        # canonical object exists so both are empty and this is a no-op.
        if self.canonical_gaussians:
            assert set(self.canonical_gaussians.keys()) <= set(self.canonical_slats.keys()), \
                "canonical_gaussians has keys not in canonical_slats"
        # canon_frame_per_object values must be FrameKey or None.
        for obj_idx, fk in self.canon_frame_per_object.items():
            if fk is not None:
                assert isinstance(fk, FrameKey), (
                    f"canon_frame_per_object[{obj_idx}] is not a FrameKey: "
                    f"{fk!r} (type {type(fk).__name__})"
                )


# =====================================================================
# Serialization (private)
# =====================================================================

_TAG_SPARSE_TENSOR = "__SparseTensor__"
_TAG_GAUSSIAN = "__Gaussian__"


def _serialize_value(v):
    """Recursively convert SparseTensor / Gaussian to torch.save-friendly dicts."""
    from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp
    from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian import Gaussian

    if isinstance(v, sp.SparseTensor):
        return {_TAG_SPARSE_TENSOR: True, "feats": v.feats.cpu(), "coords": v.coords.cpu()}
    if isinstance(v, Gaussian):
        return {
            _TAG_GAUSSIAN: True,
            "init_params": v.init_params,
            # Attributes producers set AFTER construction, which `init_params` therefore
            # does not describe.  Without these the reload silently reverts them to whatever
            # `Gaussian.__init__` re-derives: `create_gaussians_object` zeroes both biases
            # (gaussian.py) but records the constructor args that produced -inf, and it
            # raises `sh_degree` for multi-band features while `init_params` keeps 0.
            "derived": {
                "scale_bias": v.scale_bias.cpu(),
                "opacity_bias": v.opacity_bias.cpu(),
                "sh_degree": v.sh_degree,
                "active_sh_degree": v.active_sh_degree,
                "mininum_kernel_size": v.mininum_kernel_size,
            },
            "_xyz": v._xyz.cpu() if v._xyz is not None else None,
            "_features_dc": v._features_dc.cpu() if v._features_dc is not None else None,
            "_features_rest": v._features_rest.cpu() if v._features_rest is not None else None,
            "_scaling": v._scaling.cpu() if v._scaling is not None else None,
            "_rotation": v._rotation.cpu() if v._rotation is not None else None,
            "_opacity": v._opacity.cpu() if v._opacity is not None else None,
        }
    if isinstance(v, dict):
        return {k: _serialize_value(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_serialize_value(item) for item in v]
    if isinstance(v, tuple):
        items = [_serialize_value(item) for item in v]
        # Preserve NamedTuple subclass (e.g. FrameKey) — `tuple(items)` would
        # collapse it into a plain tuple, stripping the type.
        if hasattr(type(v), "_make"):
            return type(v)._make(items)
        return tuple(items)
    return v


def _deserialize_value(v):
    """Inverse of _serialize_value — reconstruct SparseTensor / Gaussian."""
    import torch

    from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp
    from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian import Gaussian

    if isinstance(v, dict):
        if _TAG_SPARSE_TENSOR in v:
            return sp.SparseTensor(coords=v["coords"].cuda(), feats=v["feats"].cuda())
        if _TAG_GAUSSIAN in v:
            g = Gaussian(**v["init_params"])
            for attr in ("_xyz", "_features_dc", "_features_rest",
                         "_scaling", "_rotation", "_opacity"):
                val = v[attr]
                if val is not None:
                    setattr(g, attr, val.cuda())
            device = g._xyz.device if g._xyz is not None else torch.device("cuda")
            for name, val in (v.get("derived") or {}).items():
                setattr(g, name, val.to(device) if torch.is_tensor(val) else val)
            # A cache without ``derived`` carries no record of the biases, and
            # `Gaussian.__init__` maps its bias ARGUMENTS through the inverse activations
            # (`scale_bias = log(scaling_bias)`, `opacity_bias = logit(opacity_bias)`) — so
            # the `0.0` that `gaussian.create_gaussians_object` passes reconstructs as -inf,
            # where it zeroes both on the live object.  Left alone that yields
            # `exp(x - inf) = 0` scales and `sigmoid(x - inf) = 0` opacities: an INVISIBLE
            # scene.  Reset a non-finite bias to 0, which reproduces the producer exactly
            # since it wrote `_scaling`/`_opacity` under a zero bias.  The SLAT decoder's own
            # biases are finite, so its caches are untouched either way.
            for bias in ("scale_bias", "opacity_bias"):
                if not torch.isfinite(getattr(g, bias)):
                    setattr(g, bias, torch.tensor(0.0, device=device))
            return g
        return {k: _deserialize_value(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_deserialize_value(item) for item in v]
    if isinstance(v, tuple):
        items = [_deserialize_value(item) for item in v]
        if hasattr(type(v), "_make"):
            return type(v)._make(items)
        return tuple(items)
    return v


def gaussian_blob_to_render_tensors(blob: dict, *, device="cpu") -> dict:
    """A ``__Gaussian__`` blob -> ``{means, quats, scales, opacities, sh}``.

    The activations :class:`Gaussian`'s getters apply, applied here WITHOUT building
    one — its ``setup_functions`` hardcodes ``.cuda()`` on the biases, so a consumer
    that only wants renderable tensors would otherwise need a GPU and
    ``sam3d_objects``.  Lives beside :func:`_serialize_value` so it cannot drift from
    the tag / ``derived`` contract that writes the blob.

    Written for the turntable, which reads its geometry from ``final/final.pt``
    rather than from ``gaussians/*.ply``: the compressed PLY writer Morton-sorts
    every file it writes, so the exported per-frame clouds do not correspond row
    for row and cannot be interpolated.  The cache is order-free and unquantized.

    Returns the same values ``render_final_results.load_gaussian_ply`` recovers from
    a PLY (opacity squeezed to ``(N,)``, SH ``(N, K, 3)`` with the DC band first),
    up to the compression's own quantization.
    """
    import torch

    if blob.get("_xyz") is None:
        raise ValueError("Gaussian blob carries no geometry (_xyz is None)")
    init = blob.get("init_params") or {}
    derived = blob.get("derived") or {}

    softplus = init.get("scaling_activation", "exp") == "softplus"

    def _bias(name: str, fallback) -> "torch.Tensor":
        """The stored bias, else re-derived through the same inverse activation
        ``Gaussian.__init__`` puts its bias ARGUMENT through."""
        v = derived.get(name)
        v = torch.as_tensor(fallback() if v is None else v, dtype=torch.float32)
        # Same healing as `_deserialize_value`: a cache without `derived`
        # reconstructs `create_gaussians_object`'s zeroed bias as -inf, which
        # renders a geometrically correct but completely INVISIBLE scene.
        return v if torch.isfinite(v) else torch.tensor(0.0)

    def _get(name: str):
        t = blob[name]
        return None if t is None else t.to(device=device, dtype=torch.float32)

    def _inverse_scaling() -> "torch.Tensor":
        s = torch.tensor(float(init.get("scaling_bias", 1.0)))
        return torch.log(torch.expm1(s)) if softplus else torch.log(s)

    def _inverse_sigmoid() -> "torch.Tensor":
        p = torch.tensor(float(init.get("opacity_bias", 0.5)))
        return torch.log(p / (1.0 - p))

    scale_bias = _bias("scale_bias", _inverse_scaling)
    opacity_bias = _bias("opacity_bias", _inverse_sigmoid)
    kernel = float(derived.get("mininum_kernel_size",
                               init.get("mininum_kernel_size", 0.0)))

    aabb = torch.as_tensor(init.get("aabb", [0, 0, 0, 1, 1, 1]),
                           dtype=torch.float32, device=device)
    quats = _get("_rotation").clone()
    quats[:, 0] += 1.0                                   # `rots_bias`, wxyz
    act = torch.nn.functional.softplus if softplus else torch.exp
    scaling = act(_get("_scaling") + scale_bias.to(device))
    dc, rest = _get("_features_dc"), _get("_features_rest")
    return {
        "means": _get("_xyz") * aabb[None, 3:] + aabb[None, :3],
        "quats": torch.nn.functional.normalize(quats, dim=-1),
        "scales": torch.sqrt(torch.square(scaling) + kernel ** 2),
        "opacities": torch.sigmoid(_get("_opacity") + opacity_bias.to(device)).squeeze(-1),
        "sh": dc if rest is None else torch.cat((dc, rest), dim=1),
    }


# =====================================================================
# Cache I/O
# =====================================================================

def get_pipeline_cache_dir(output_dir: str) -> str:
    """Return the pipeline cache directory as a sibling of the timestamp dir.

    Given ``results/{experiment}/{dataset}/{scene}/{timestamp}`` returns
    ``results/{experiment}/{dataset}/{scene}/cache``. The cache sits one level above
    the timestamp dir, so it is shared across runs that match every preceding path
    segment (same experiment/dataset/scene).
    """
    return os.path.join(os.path.dirname(output_dir), "cache")


def save_pipeline_cache(state: PipelineState, block_name: str, cache_dir: str) -> None:
    """Serialize *state* after *block_name* completes."""
    import torch

    os.makedirs(cache_dir, exist_ok=True)
    data = {k: _serialize_value(v) for k, v in state.__dict__.items()
            if k != "pipeline_obj"}
    path = os.path.join(cache_dir, f"{block_name}.pt")
    torch.save(data, path)
    print(f"  Pipeline cache saved: {path}")


def _torch_load(path: str, **kwargs):
    """``torch.load`` of a pipeline cache (it pickles ``FrameKey``, so not weights-only)."""
    import torch

    return torch.load(path, weights_only=False, **kwargs)


def load_pipeline_cache_raw(path: str) -> dict:
    """The cache AS WRITTEN — no PipelineState, no Gaussian, no CUDA.

    :func:`load_pipeline_cache` reconstructs the live objects, which imports
    ``sam3d_objects`` and forces every tensor onto the GPU.  A consumer that only
    wants tensors (the turntable's geometry source) pays neither: everything comes
    back as the plain nested dicts ``_serialize_value`` produced, with the Gaussians
    still tagged (see :func:`gaussian_blob_to_render_tensors`).
    """
    return _torch_load(path, map_location="cpu")


def load_pipeline_cache(path: str, cls: type = PipelineState) -> PipelineState:
    """Reconstruct a state of class *cls* from a cached ``.pt`` file.

    *cls* is the run's state class (e.g. ``genia.core.state.SAM3DState``); no class
    is pickled, so any cache loads into any of them.  Keys *cls* does not declare
    (a newer cache, or another state class's fields) are silently dropped so that
    removing a field does not break existing caches.
    """
    data = _torch_load(path)

    valid_fields = set(cls.__dataclass_fields__)
    kwargs = {
        k: _deserialize_value(v)
        for k, v in data.items()
        if k in valid_fields
    }
    return cls(**kwargs)


def find_latest_pipeline_cache(cache_dir: str, active_blocks: list) -> tuple | None:
    """Scan *active_blocks* from last→first, return ``(block_name, path)`` or ``None``."""
    for block_name in reversed(active_blocks):
        path = os.path.join(cache_dir, f"{block_name}.pt")
        if os.path.isfile(path):
            return (block_name, path)
    return None


__all__ = [
    "PipelineState",
    "get_pipeline_cache_dir",
    "save_pipeline_cache",
    "load_pipeline_cache",
    "load_pipeline_cache_raw",
    "gaussian_blob_to_render_tensors",
    "find_latest_pipeline_cache",
]
