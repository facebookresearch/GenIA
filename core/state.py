# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Our pipeline's state: the base :class:`PipelineState` plus SAM3D's token stores.

The base state carries what every method produces (poses, Gaussians, SLATs, meshes,
tracks).  SAM3D additionally keeps its raw Stage-1 modalities (canonical and a per-frame
snapshot), the poses decoded from them, and the per-frame GT voxel correspondence.  They
live here so the shared code (FINAL, pose refine) never depends on them; the few readers
there that use them when present go through ``getattr``.

No class is pickled into a cache (``save_pipeline_cache`` dumps ``__dict__``), so every
cache loads into either class: ``load_pipeline_cache(path, cls=SAM3DState)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from genia.core.utils.frame_key import FrameKey, as_frame_key, group_by_frame
from genia.core.utils.pipeline_state import (
    PipelineState, _NoneMissingDict, timestamp_reference_key,
)


@dataclass
class SAM3DState(PipelineState):
    """:class:`PipelineState` plus the SAM3D raw-token stores our blocks read and write."""

    canonical_poses: dict = field(default_factory=dict)
    """Per-object canonical pose: {obj_idx: {rotation, translation, scale}}.
    Cached — auto-decoded from ``canonical_raw_modalities`` layout tokens
    via ``differentiable_pose_decode``.  Never set externally."""

    perframe_raw_modalities: dict = field(default_factory=dict)
    """Per-frame snapshot of raw Stage 1 tokens, with the SSI constants that decode them.

    Structure: {obj_idx: {FrameKey: {
        'raw_ss_modalities': {key: Tensor, ...},
        'pointmap_scale': Tensor,
        'pointmap_shift': Tensor,
        'downsample_factor': float,
    }}}

    Preserved independently from tokens_by_object so later stages can
    access these pose tokens even after a Stage-1 block or a global pose
    refine overwrites the poses.
    """

    canonical_raw_modalities: dict = field(default_factory=dict)
    """Per-object canonical raw Stage 1 modalities (source field).

    Structure: {obj_idx: {key: Tensor, ...}}
    Keys: ``shape``, ``6drotation_normalized``, ``translation``, ``scale``,
    ``translation_scale``.

    Set from the canonical frame and updated by the Stage-1 blocks.
    Auto-decodes to:

    - ``canonical_poses`` — from layout tokens (6D rot, translation, scale)
    - ``canonical_slats`` coords — from shape tokens (occupancy grid)
    - ``canonical_shape_coords`` — numpy coords for visualization

    Pointmap scale/shift/downsample_factor needed for pose decoding are
    read from ``perframe_raw_modalities`` (canonical frame).
    """

    gt_perframe_voxel_correspondence: dict = field(default_factory=dict)
    """Per-frame GT voxel grids + per-frame-voxel→canonical correspondence.

    Structure: ``{obj_idx: {frame_int: {"coords": Tensor (L_pf, 3) int64,
    "pf_to_canon": Tensor (L_pf,) int64}}}``.  Each frame's mesh is voxelised on its OWN (frame-i self-norm)
    grid — the real GT shape at that timestamp, independent of the canonical grid
    (so ``L_pf`` may exceed ``L_canon`` and several per-frame voxels collapse onto
    one canonical voxel).  ``pf_to_canon`` indexes rows of
    ``gt_canonical_shape_coords[obj_idx]`` (via the fixed-topology shared vertex
    index; ``vertex`` method exact, ``surface`` method nearest-vertex).  NO
    voxel-grid warping is involved.

    Populated ONCE by ``GT_SHAPES_INVERSION`` (``mode=per_frame_meshes``) via
    ``gt_geometry.compute_perframe_voxel_correspondence`` (derived from the
    per-vertex mesh field; ``perframe_voxelization`` selects the method).  Consumed by
    the dynamic path of the ``appearance_init=canonical_unified`` Stage-2 strategy.  Lifecycle: GT-stable like
    ``gt_canonical_shape_coords``.
    """

    # -----------------------------------------------------------------
    # Canonical raw modalities mutation methods
    # -----------------------------------------------------------------

    def set_canonical_raw_modalities(self, obj_idx: int, raw_modalities: dict,
                                     decode_shape: bool = True) -> None:
        """Set canonical raw Stage 1 modalities and auto-decode poses.

        Always decodes layout tokens → ``canonical_poses[obj_idx]``.

        When ``decode_shape=True`` (default), also decodes shape tokens →
        updates ``canonical_slats[obj_idx]`` coords and
        ``canonical_shape_coords``.  If coords changed, invalidates
        ``canonical_gaussians`` (stale features).  Shape decoding runs the
        ``ss_decoder`` on GPU.  Pass ``decode_shape=False`` when
        ``set_canonical_slat`` already set the coords, or when only the
        layout tokens changed.

        Parameters
        ----------
        raw_modalities : dict
            Raw Stage 1 modality tensors: ``shape``,
            ``6drotation_normalized``, ``translation``, ``scale``, etc.
        decode_shape : bool
            If True (default), decode shape tokens to update SLAT coords.
        """
        import torch
        from genia.core.utils.refinement import differentiable_pose_decode

        self.canonical_raw_modalities[obj_idx] = raw_modalities
        ss = raw_modalities

        # --- Decode layout tokens → canonical_poses ---
        canon_fid = self.canon_frame_per_object.get(obj_idx)
        pf_raw = None
        if obj_idx in self.perframe_raw_modalities and canon_fid is not None:
            pf_raw = self.perframe_raw_modalities[obj_idx].get(canon_fid)
        if pf_raw is not None:
            with torch.no_grad():
                _, rot, trans, scale = differentiable_pose_decode(
                    ss["6drotation_normalized"],
                    ss["translation"],
                    ss["scale"],
                    pf_raw["pointmap_scale"],
                    pf_raw["pointmap_shift"],
                    pf_raw.get("downsample_factor", 1.0),
                )
            self.canonical_poses[obj_idx] = {
                "rotation": rot.detach(),
                "translation": trans.detach(),
                "scale": scale.detach(),
            }

        # --- Optionally decode shape tokens → canonical SLAT coords ---
        if decode_shape and "shape" in ss:
            from genia.core.utils.slat_decode import decode_shape_to_coords

            new_coords = decode_shape_to_coords(self.pipeline_obj, ss["shape"])
            new_shape_coords = new_coords[:, 1:].cpu().numpy().astype(float)

            shape_changed = True
            old_slat = self.canonical_slats.get(obj_idx)
            if old_slat is not None:
                old_coords = old_slat.coords
                shape_changed = (
                    old_coords.shape != new_coords.shape
                    or not torch.equal(old_coords, new_coords)
                )

            if shape_changed:
                self.invalidate_canonical(obj_idx, shape_coords=new_shape_coords)
            else:
                # Shape unchanged — just update coords for visualization
                self.canonical_shape_coords[obj_idx] = new_shape_coords

    def set_all_perframe_shape_tokens(
        self, obj_idx: int, shape_by_frame: dict,
        *, decode_coords: bool = True,
    ) -> None:
        """Replace per-frame raw *shape* tokens for *obj_idx* and keep all
        derived state consistent.

        Per-frame analog of :meth:`set_canonical_raw_modalities`'s shape
        branch.  Changing a per-frame shape token makes three things stale;
        they are repaired here so callers never reach into ``decoder_input``
        directly:

        1. ``raw_ss_modalities['shape']`` — overwritten with the new token.
        2. Cached per-frame SLATs / Gaussians — decoded from the OLD shape;
           dropped via :meth:`invalidate_perframe_slats` (pops
           ``decoder_input_slat``; clears the per-frame Gaussian cache).
        3. ``perframe_shape_coords`` — the voxel grid the new shape
           occupies; re-decoded here (``decode_coords=True``) so per-frame
           consumers (APPEARANCE_INIT=perframe) find a grid without depending
           on the parallel Stage-1 solve (the only other producer).  Pass ``decode_coords=False`` when a single
           shared canonical grid already covers every frame (the
           GT_SHAPES_INVERSION ``mode=global`` broadcast — ``raw_ss_modalities
           ['shape']`` is identical across frames and the grid lives in
           ``canonical_shape_coords``).

        Parameters
        ----------
        shape_by_frame : dict
            ``{FrameKey: shape_token}``.  Bare-int keys are coerced to
            ``FrameKey(int, 0)``.
        decode_coords : bool
            If True (default), decode each new token to its voxel grid and
            store it on the frame's decoder_input as ``perframe_shape_coords``.
        """
        from genia.core.utils.slat_decode import decode_shape_to_coords

        shape_by_frame = {as_frame_key(k): v for k, v in shape_by_frame.items()}
        for entry_fk, di in self.tokens_by_object[obj_idx]:
            if entry_fk in shape_by_frame:
                di.setdefault("raw_ss_modalities", {})["shape"] = (
                    shape_by_frame[entry_fk]
                )

        # Per-frame SLATs/Gaussians were decoded from the old shape.
        self.invalidate_perframe_slats(obj_idx)

        if not decode_coords:
            return
        for entry_fk, di in self.tokens_by_object[obj_idx]:
            if entry_fk in shape_by_frame:
                di["perframe_shape_coords"] = decode_shape_to_coords(
                    self.pipeline_obj, shape_by_frame[entry_fk],
                )

    # Raw Stage 1 layout token keys (pose modalities, excluding shape).
    _LAYOUT_KEYS = ("6drotation_normalized", "translation",
                    "scale", "translation_scale")

    # Per-frame keys that live BOTH in tokens_by_object[obj][i][1] (decoder_input)
    # AND in perframe_raw_modalities[obj][fk]. Two-place storage is intentional:
    # `perframe_raw_modalities` is a SNAPSHOT, kept so later stages (the Stage-1
    # blocks, GLOBAL_POSE_REFINE) can access those tokens even after they
    # overwrite the live decoder_input.
    # Of these keys:
    #  - "pointmap_scale", "pointmap_shift", "downsample_factor" are SSI
    #    *constants* (frame-level intrinsics). They are fixed when the frame's
    #    tokens are first built, and the two copies must always agree.
    #  - "raw_ss_modalities" is the layout-token dict that *does* change with
    #    optimization. The two copies may legitimately differ (snapshot vs live).
    _PERFRAME_SSI_CONSTANT_KEYS = (
        "pointmap_scale", "pointmap_shift", "downsample_factor",
    )

    def decode_perframe_poses_from_raw(
        self,
        obj_idx: int,
        source: str = "perframe_raw_modalities",
    ) -> int:
        """Re-decode per-frame poses from raw layout tokens into ``tokens_by_object``.

        For each frame of *obj_idx*, reads the raw layout tokens
        (``6drotation_normalized``, ``translation``, ``scale``) and
        scene context (``pointmap_scale``, ``pointmap_shift``,
        ``downsample_factor``), runs ``differentiable_pose_decode``, and
        writes the decoded ``rotation`` / ``translation`` / ``scale``
        back into the corresponding ``tokens_by_object`` decoder_input dict.

        Parameters
        ----------
        source : str
            Where to read raw modalities from:
            ``"perframe_raw_modalities"`` — the per-frame snapshot (default).
            ``"tokens_by_object"`` — the ``di["raw_ss_modalities"]`` dict
            on each frame's decoder_input (current live state, may have
            been modified by a Stage-1 block).

        Returns the number of frames successfully decoded.
        """
        import torch
        from genia.core.utils.refinement import differentiable_pose_decode

        if source == "perframe_raw_modalities":
            pf_raw_obj = self.perframe_raw_modalities.get(obj_idx, {})
        elif source != "tokens_by_object":
            raise ValueError(f"Unknown source: {source!r}")

        n_updated = 0
        for fk, di in self.tokens_by_object.get(obj_idx, []):
            if source == "perframe_raw_modalities":
                raw_info = pf_raw_obj.get(fk)
                if raw_info is None:
                    continue
                raw_mods = raw_info.get("raw_ss_modalities", {})
                ps = raw_info.get("pointmap_scale")
                psh = raw_info.get("pointmap_shift")
                dsf = raw_info.get("downsample_factor", 1.0)
            else:  # tokens_by_object
                raw_mods = di.get("raw_ss_modalities", {})
                ps = di.get("pointmap_scale")
                psh = di.get("pointmap_shift")
                dsf = di.get("downsample_factor", 1.0)

            if ("6drotation_normalized" in raw_mods
                    and "translation" in raw_mods
                    and "scale" in raw_mods
                    and ps is not None and psh is not None):
                with torch.no_grad():
                    _, rot, trans, scale = differentiable_pose_decode(
                        raw_mods["6drotation_normalized"],
                        raw_mods["translation"],
                        raw_mods["scale"],
                        ps, psh, dsf,
                    )
                di["rotation"] = rot.detach()
                di["translation"] = trans.detach()
                di["scale"] = scale.detach()
                n_updated += 1
        return n_updated

    def update_canonical_layout_from_perframe(self, obj_idx: int) -> bool:
        """Replace canonical layout tokens with the canonical frame's
        refined per-frame layout tokens and re-decode canonical poses.

        Reads ``6drotation_normalized``, ``translation``, ``scale``, and
        ``translation_scale`` from the canonical frame's entry in
        ``perframe_raw_modalities`` and writes them into
        ``canonical_raw_modalities``.  Then re-decodes
        ``canonical_poses`` via ``set_canonical_raw_modalities``
        (with ``decode_shape=False`` since shape hasn't changed).

        Returns True if the update was performed, False if data was missing.
        """
        canon_fk = self.canon_frame_per_object.get(obj_idx)
        if canon_fk is None:
            return False
        canon_fk = as_frame_key(canon_fk)
        canon_raw = self.canonical_raw_modalities.get(obj_idx)
        if canon_raw is None:
            return False
        pf_raw_obj = self.perframe_raw_modalities.get(obj_idx, {})
        pf_raw_info = pf_raw_obj.get(canon_fk)
        if pf_raw_info is None:
            return False
        pf_mods = pf_raw_info.get("raw_ss_modalities", {})

        layout_keys = self._LAYOUT_KEYS
        for key in layout_keys:
            if key in pf_mods:
                canon_raw[key] = pf_mods[key].clone().detach()

        # Re-decode canonical poses (shape unchanged)
        self.set_canonical_raw_modalities(obj_idx, canon_raw, decode_shape=False)
        return True

    # -----------------------------------------------------------------
    # SSI-param consistency between tokens_by_object and perframe_raw_modalities
    # -----------------------------------------------------------------

    def assert_perframe_ssi_consistent(
        self,
        obj_idx: Optional[int] = None,
        frame_key: Optional[FrameKey] = None,
    ) -> None:
        """Assert that ``tokens_by_object`` and ``perframe_raw_modalities`` agree
        on the SSI *constants* (``pointmap_scale``, ``pointmap_shift``,
        ``downsample_factor``) wherever both have an entry.

        Diagnostic helper. Skips entries where the snapshot has no value yet
        (nothing has populated this frame). Does NOT compare
        ``raw_ss_modalities`` because those legitimately diverge (the snapshot
        is a frozen copy).

        Raises ``AssertionError`` on mismatch with a description of the offending
        ``(obj_idx, frame_key, field)`` triple.
        """
        import torch

        target_obj = obj_idx
        target_fk = as_frame_key(frame_key) if frame_key is not None else None

        for oi, entries in self.tokens_by_object.items():
            if target_obj is not None and oi != target_obj:
                continue
            pf_obj = self.perframe_raw_modalities.get(oi, {})
            if not pf_obj:
                continue
            for fk, di in entries:
                if target_fk is not None and fk != target_fk:
                    continue
                slot = pf_obj.get(fk)
                if slot is None:
                    continue  # snapshot not populated for this frame yet
                for k in self._PERFRAME_SSI_CONSTANT_KEYS:
                    di_v = di.get(k)
                    pf_v = slot.get(k)
                    if di_v is None or pf_v is None:
                        continue
                    if isinstance(di_v, torch.Tensor) and isinstance(pf_v, torch.Tensor):
                        if not torch.equal(di_v.detach().cpu(), pf_v.detach().cpu()):
                            raise AssertionError(
                                f"SSI desync: obj_idx={oi} frame_key={fk} "
                                f"field={k!r}: tokens_by_object has {di_v.flatten()[:4]} "
                                f"but perframe_raw_modalities has {pf_v.flatten()[:4]}"
                            )
                    else:
                        if di_v != pf_v:
                            raise AssertionError(
                                f"SSI desync: obj_idx={oi} frame_key={fk} "
                                f"field={k!r}: tokens_by_object={di_v!r} "
                                f"perframe_raw_modalities={pf_v!r}"
                            )

    # -----------------------------------------------------------------
    # Bulk state update
    # -----------------------------------------------------------------

    def apply_finetuning_results(
        self,
        canonical_slats: dict,
        tokens_by_object: dict,
        canonical_gaussians: dict = None,
        lora_state_dicts: dict = None,
    ) -> None:
        """Apply finetuning results atomically.

        When *canonical_gaussians* is provided (decoded with the LoRA
        decoder inside the finetuning loop), uses them directly.  When
        not provided, falls back to re-decoding with the frozen decoder.

        *lora_state_dicts* stores per-object LoRA weights for cache
        persistence and future re-decoding.
        """
        self.canonical_slats = _NoneMissingDict.of(canonical_slats)
        self.tokens_by_object = tokens_by_object
        self.canonical_shape_coords.clear()
        # Mesh-level deformation fields (canonical_mesh_verts /
        # canonical_mesh_per_frame_verts / canonical_mesh_per_frame_rotations
        # / canonical_mesh_faces) are row-aligned with the V GT mesh
        # vertices — independent of any SLAT / voxel-grid state.  They
        # MUST be preserved across FINETUNE so the post-block keyframes
        # video, decoded-viz, and any subsequent block can still apply
        # the per-frame Φ + R warp.  Clearing them silently degrades
        # actionmesh runs to rigid-only rendering after FINETUNE.

        if lora_state_dicts is not None:
            self.lora_state_dicts = lora_state_dicts

        if canonical_gaussians is not None:
            self.canonical_gaussians = _NoneMissingDict.of(canonical_gaussians)
        else:
            # Fallback: re-decode with frozen decoder (no LoRA)
            from genia.core.utils.slat_decode import redecode_slat
            self.canonical_gaussians = _NoneMissingDict()
            for obj_idx, slat in self.canonical_slats.items():
                decoded = redecode_slat(self.pipeline_obj, slat, formats=["gaussian"])
                self.canonical_gaussians[obj_idx] = decoded["gaussian"][0]

        # Invalidate per-frame cache
        self.perframe_gaussians = None

    def apply_perframe_finetuning_results(
        self,
        tokens_by_object: dict,
        perframe_gaussians: dict = None,
    ) -> None:
        """Apply per-frame finetuning results atomically.

        The per-frame counterpart of :meth:`apply_finetuning_results`, for runs
        with no canonical object.  *tokens_by_object* already carries each
        frame's finetuned ``decoder_input_slat``; *perframe_gaussians*
        (``{obj_idx: {FrameKey: Gaussian}}``) are the Gaussians decoded inside
        the finetuning loop — with the LoRA decoder when one was used, which
        ``ensure_perframe_gaussians``' frozen-decoder re-decode could not
        reproduce.  Omit them to re-decode from the SLATs instead.

        No canonical store is touched: there is none, and creating one here
        would flip ``has_canonical`` and reroute FINAL.
        """
        self.tokens_by_object = tokens_by_object
        if perframe_gaussians is not None:
            self.perframe_gaussians = {
                obj_idx: {as_frame_key(fk): gs for fk, gs in by_frame.items()}
                for obj_idx, by_frame in perframe_gaussians.items()
            }
        else:
            self.perframe_gaussians = None
            self.ensure_perframe_gaussians()

    # -----------------------------------------------------------------
    # Core lifecycle, extended to the SAM3D stores
    # -----------------------------------------------------------------

    def check_consistency(self) -> None:
        super().check_consistency()
        # SSI two-place invariant: live tokens_by_object SSI constants must match
        # the snapshot in perframe_raw_modalities, where both stores are populated.
        # Permissive: skips frames whose snapshot hasn't been populated yet.
        # Raises AssertionError on actual desync.
        self.assert_perframe_ssi_consistent()


def timestamp_reference_keys(keys, ref_key):
    """``{timestamp: reference FrameKey}`` over ``keys``, via :func:`timestamp_reference_key`."""
    by_key = {as_frame_key(k): None for k in keys}
    return {f: timestamp_reference_key(by_key, ref_key, f, group)
            for f, group in group_by_frame(by_key).items()}
