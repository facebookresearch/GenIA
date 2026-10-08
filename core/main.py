"""
The block runners of the GenIA pipeline. ``core/pipeline.py`` maps block names to
them; the block loop in ``core/run.py`` runs them.

Lifts 2D video frames to 3D Gaussians via SLAT tokens, then optionally refines
poses and fine-tunes tokens through differentiable rendering.

The blocks, their order and their config sections are ``core/manifest.py``'s
``BLOCKS``.

Examples
--------
# Single image
python -m genia.core +experiment=mono_static dataset=image dataset.scene_name=<scene>

# Multi-view static
python -m genia.core +experiment=mv_static dataset=mvcustom dataset.scene_name=<scene>
"""
import os
import sys
import warnings

# torch.cuda imports the deprecated pynvml package — silence its FutureWarning.
warnings.filterwarnings("ignore", message=r"The pynvml package is deprecated.*")

# Skip sam3d_objects heavyweight initialization (must be set before any utils import)
os.environ['LIDRA_SKIP_INIT'] = '1'

from genia.core.paths import SAM3D_OBJECTS_ROOT

# Ensure submodules are importable.
# APPEND, never insert(0): a caller may have put its own vendored ``sam3d_objects``
# at the head of sys.path before importing this module, and that copy must win.
_submodules = str(SAM3D_OBJECTS_ROOT)
if _submodules not in sys.path:
    sys.path.append(_submodules)

import numpy as np

from genia.core.utils.config import (
    get_block_output_flag, block_output_subdir,
)
from genia.core.utils.pipeline_state import get_pipeline_cache_dir, load_pipeline_cache
from genia.core.utils.console import block_header
from genia.core.utils.evaluation import (
    capture_before,
    evaluate_block,
    save_keyframes_video,
)
from genia.core.visualization import save_decoded_visualizations
from genia.core.utils.timing import PipelineTimer, get_timer, measure_core_seconds
from genia.core.final import (
    _build_perobj_mesh_warp_kwargs,
)
from genia.core.pose_refine import _emit_pose_overlay_viz

# Process-wide singleton (shared with helpers that run inside a block).
_TIMER: PipelineTimer = get_timer()


def _build_stage2_mesh_warp_kwargs(state, obj_idx, tokens_list, device, dw_cfg) -> dict:
    """Build the per-vertex-deformation kwargs for ``stage2_mv`` /
    ``render_appearance_ode_steps`` (actionmesh appearance-init path).

    Returns the full bundle: per-frame mesh verts and R (sliced + ordered
    to match ``tokens_list``), the shared canonical mesh verts, and the
    KNN/blend knobs from the global ``deformation_warp`` config.  Caller
    passes the dict to the consumer via ``**kwargs``.

    Returns ``{}`` (empty dict) when actionmesh is not the strategy or the
    mesh fields aren't populated — caller can splat unconditionally.
    """
    if obj_idx not in state.canonical_mesh_verts:
        return {}
    canon_verts = state.canonical_mesh_verts[obj_idx].to(device)
    pf_verts = [
        state.canonical_mesh_per_frame_verts[obj_idx][_fk.frame].to(device)
        for _fk, _ in tokens_list
    ]
    pf_R_vert = [
        state.canonical_mesh_per_frame_rotations[obj_idx][_fk.frame].to(device)
        for _fk, _ in tokens_list
    ]
    faces = state.canonical_mesh_faces.get(obj_idx)
    return {
        "canonical_mesh_verts": canon_verts,
        "per_frame_mesh_verts": pf_verts,
        "per_frame_mesh_rotations": pf_R_vert,
        "canonical_mesh_faces": (
            faces.to(device).long() if faces is not None else None
        ),
        "warp_knn_k": int(dw_cfg.knn_k),
        "warp_knn_eps": float(dw_cfg.knn_eps),
        "warp_knn_chunk_size": int(dw_cfg.knn_chunk_size),
    }


# Knobs that make the ODE batch INTERDEPENDENT across views.  Restricting the batch to one
# view is only equivalent when none of them is active: each either averages a velocity over
# the batch or draws its signal from more than one frame.  `rotation_velocity_averaging` is
# deliberately absent -- it groups by view, so on MV-static every group is a singleton and it
# is already a no-op; refusing on it would mean this never fires on the shipped recipe.
_ODE_BATCH_COUPLINGS = (
    ("shape_velocity_averaging", ("none", "", None)),
)


def _reference_only_ode_reference(cfg, sp_cfg, state, sequence, obj_idx):
    """The frame to run the Stage-1 ODE on alone, or None to run the full batch.

    The per-object precondition below checks that the non-reference frames carry the raw
    pose tokens and SSI (``pointmap_scale`` / ``pointmap_shift``) that the rebase and the
    refit need.  This block writes none of them before the ODE: they come from an earlier
    Stage-1 pass (the ``shape_init`` ODE) or from GT pose injection.

    Refuses rather than raises, always: the fallback is the full-batch ODE, so a
    wrong "no" costs time and a wrong "yes" costs correctness.
    """
    from genia.core.utils.frame_key import as_frame_key
    from genia.core.pose_refit import frame_has_refit_tokens
    from genia.core.utils.pipeline_state import resolve_reference_frame

    flag = getattr(sp_cfg, "mv_reference_only_ode", False)
    # Strict, because the silent failure direction here is ENABLING the optimisation: the
    # YAML is untyped (the dataclasses are never registered as a schema), so a stored config carrying the
    # string "false" would be truthy.
    if not isinstance(flag, bool):
        raise TypeError(
            f"mv_reference_only_ode must be a bool, got {flag!r} "
            f"(note {flag!r} is TRUTHY, so this would have silently ENABLED it)")
    if not flag:
        return None

    def _no(reason):
        print(f"    Reference-only ODE: OFF for obj {obj_idx} — {reason}")
        return None

    if not sequence.is_mv:
        # The only silent refusal: this ships ON by default and mono data has one view per
        # timestamp, so there is nothing to restrict.  Printing it would add a line per
        # object to every mono run.  Every other refusal below reports a config or data
        # fact that disables an optimisation the user could plausibly have expected.
        return None
    if not bool(cfg.pipeline.mv_shared_world_pose):
        return _no("mv_shared_world_pose is off, so nothing would derive the other views")
    if sequence.is_dynamic:
        # NOT because the rebase would erase motion -- it groups by timestamp and does not.
        # Because ONE view cannot stand in for T timestamps: the restriction would have to
        # keep one view PER timestamp (a batch of T, not of 1), which is a different change.
        return _no("dynamic data: one view cannot stand in for every timestamp")
    if getattr(sp_cfg, "shape_cache_dir", None):
        return _no("shape_cache_dir is keyed on the whole frame set")
    for knob, off_values in _ODE_BATCH_COUPLINGS:
        # Absent reads as OFF (the first listed off-value), not as coupled: a config
        # predating a knob does not have that feature, and treating the gap as "on" would
        # refuse on every recipe that never grew the key.
        val = getattr(sp_cfg, knob, off_values[0])
        if val not in off_values:
            return _no(f"{knob}={val!r} couples the ODE batch across views")

    entries = state.tokens_by_object.get(obj_idx) or []
    if len(entries) < 2:
        return _no("fewer than two frames: nothing to save")
    ref_fk = as_frame_key(
        resolve_reference_frame(entries, state.canon_frame_per_object, obj_idx))
    by_key = {as_frame_key(fk): di for fk, di in entries}
    if ref_fk not in by_key:
        return _no(f"reference {ref_fk} is not among this object's frames")

    # The reference must be the pose-velocity broadcast LEAD, or the restricted run would
    # reproduce a trajectory driven by a different view.  True by construction; asserted so
    # the assumption is visible rather than load-bearing and invisible.
    if ref_fk.view != min(as_frame_key(fk).view for fk in by_key):
        return _no(f"reference {ref_fk} is not the lowest-view (broadcast lead)")

    mask = (sequence[ref_fk].masks or {}).get(obj_idx)
    if mask is None or not bool(mask.any()):
        return _no(f"reference {ref_fk} has no mask for this object")

    # THE precondition.  A non-reference frame that loses these is skipped by the refit
    # SILENTLY -- the block would report the frames it did refit and say nothing about the
    # rest, turning a V-view consensus into a one-view fit.
    pf_raw_obj = (state.perframe_raw_modalities or {}).get(obj_idx, {})
    for fk, di in entries:
        fk = as_frame_key(fk)
        if fk == ref_fk:
            continue
        if not isinstance(di.get("raw_ss_modalities"), dict):
            return _no(f"{fk} has no raw_ss_modalities, so the rebase cannot write its "
                       f"pose tokens and the refit would skip it")
        if not frame_has_refit_tokens(di):
            return _no(f"{fk} lacks the raw translation/scale the refit reads")
        pf_raw = pf_raw_obj.get(fk)
        ps = pf_raw.get("pointmap_scale") if pf_raw else di.get("pointmap_scale")
        psh = pf_raw.get("pointmap_shift") if pf_raw else di.get("pointmap_shift")
        if ps is None or psh is None:
            return _no(f"{fk} has no pointmap_scale/shift, so the rebase would drop its "
                       f"raw tokens")
        if pf_raw is not None and not _ssi_matches(pf_raw, di):
            # The rebase encodes the derived tokens against one SSI while the refit reads
            # the other -- a well-formed but wrong pose.
            return _no(f"{fk}: perframe_raw_modalities SSI disagrees with the frame's")
    return ref_fk


def _ssi_matches(pf_raw, di) -> bool:
    """Do the two SSI sources agree for this frame?  Missing on either side counts as
    agreement -- the caller has already established that the resolved pair is present."""
    import torch

    for key in ("pointmap_scale", "pointmap_shift"):
        a, b = pf_raw.get(key), di.get(key)
        if a is None or b is None:
            continue
        if not torch.allclose(torch.as_tensor(a).float(), torch.as_tensor(b).float()):
            return False
    return True


def _snapshot_pose(tokens_list):
    """``{fk: pose}`` of every frame that has one: decoded fields + raw pose tokens."""
    snap = {}
    for fk, di in tokens_list:
        if di.get("rotation") is None:
            continue
        raw = di.get("raw_ss_modalities", {})
        snap[fk] = {
            "decoded": {k: di[k].clone() for k in ("rotation", "translation", "scale")},
            "raw": {k: raw[k].clone() for k in
                    ("6drotation_normalized", "translation", "scale", "translation_scale")
                    if k in raw},
        }
    return snap


def _guard_pose_flips(tokens_list, prior, max_deg):
    """Restore a frame's prior pose when this pass rotated it by more than ``max_deg``.

    A pose re-derived from noise can land in another mode of a near-symmetric object (a
    flat square pillow turned half a turn fits silhouette and depth alike), which no
    later refinement undoes.  A jump that large from the prior pose is that failure, not
    a refinement.
    """
    import math
    for fk, di in tokens_list:
        before = prior.get(fk)
        if before is None or di.get("rotation") is None:
            continue
        q0 = before["decoded"]["rotation"].reshape(-1).float()
        q1 = di["rotation"].reshape(-1).float().to(q0.device)
        dot = (q0 @ q1).abs() / (q0.norm() * q1.norm())
        deg = math.degrees(2 * math.acos(min(1.0, float(dot))))
        if deg <= max_deg:
            print(f"    Pose flip guard: {fk} rotated {deg:.1f} deg -> kept")
            continue
        print(f"    Pose flip guard: {fk} rotated {deg:.1f} deg > {max_deg} "
              "-> prior pose restored")
        di.update(before["decoded"])
        di.setdefault("raw_ss_modalities", {}).update(before["raw"])


def _seat_reference_only_pose_init(state, obj_idx, ode_tokens, ref_fk):
    """Seat a reference-only ODE result on the FULL token list.

    REPLACES the reference's whole dict rather than copying its Sim(3) the way
    a shared-world write-back does: the ODE returns a fresh dict carrying the merged
    raw modalities (including the denoised shape), its own pointmap SSI and
    ``perframe_shape_coords``.  Copying three pose fields would leave the reference's RAW
    tokens at their pre-ODE seed values while its DECODED pose came from the solve -- an
    internally inconsistent frame that ``decode_perframe_poses_from_raw`` would later undo.

    Propagation is NOT this function's job: the shared-world rebase runs at the call site
    for BOTH branches, because deriving the other views from the reference is right whether
    or not the ODE was restricted to it -- the refit needs each view's own rotation either
    way.

    Returns the full tokens list.
    """
    from genia.core.utils.frame_key import as_frame_key

    ode_by_key = {as_frame_key(fk): di for fk, di in ode_tokens}
    ref_di = ode_by_key.get(as_frame_key(ref_fk))
    if ref_di is None:
        raise ValueError(
            f"reference-only ODE returned no entry for its own reference {ref_fk}; "
            f"got {sorted(ode_by_key)}")

    entries = [(fk, ref_di if as_frame_key(fk) == as_frame_key(ref_fk) else di)
               for fk, di in state.tokens_by_object[obj_idx]]
    state.tokens_by_object[obj_idx] = entries
    return entries


def _entropy_plot_frames(tokens_list, obj_entropy, n_rows):
    """The ``(fk, di)`` entries the entropy / cross-attention arrays' BATCH axis indexes.

    Those arrays are one row per frame the ODE actually VISITED, which is not always the
    object's whole token list: a maskless frame is dropped from the batch, and the
    reference-only path restricts it deliberately.  Zipping them against the full list
    silently mislabels every view after the first gap -- and pairs an entropy row with
    another view's image, which is worse than no plot.

    ``frame_keys`` is recorded by the ODE itself so this cannot be re-derived wrongly.
    Raises on a count it cannot explain, rather than truncating to the shorter of the two.
    """
    from genia.core.utils.frame_key import as_frame_key

    keys = obj_entropy.get("frame_keys") if isinstance(obj_entropy, dict) else None
    if keys:
        wanted = {as_frame_key(k) for k in keys}
        rows = [(fk, di) for fk, di in tokens_list if as_frame_key(fk) in wanted]
    else:
        rows = list(tokens_list)
    if len(rows) != n_rows:
        raise ValueError(
            f"entropy/cross-attention arrays have {n_rows} row(s) but {len(rows)} frame(s) "
            f"resolve to them (of {len(tokens_list)} in the object); the plot would "
            f"mislabel views. frame_keys={keys!r}")
    return rows


# =====================================================================
# Shape-token stats probe
# =====================================================================

def _print_inversion_stats(
    shapes, ss_decoder, obj_idx, label="inverted", indent="    ",
):
    """Print z + decoder-logit stats for a list of shape tokens (each
    ``(1, 4096, 8)``).  Used for both the GT-inversion path (label=
    ``"inverted"``) and the Stage-1 prediction probe (label=
    ``"predicted"``) so the two can be compared 1:1."""
    import torch
    from genia.core.shape_inversion import shape_tokens_to_latent

    if not shapes:
        return
    agg = torch.cat([s.reshape(-1) for s in shapes])
    pf_mean = torch.tensor([s.mean().item() for s in shapes])
    pf_var = torch.tensor([s.var().item() for s in shapes])
    pf_min = torch.tensor([s.min().item() for s in shapes])
    pf_max = torch.tensor([s.max().item() for s in shapes])
    print(
        f"{indent}{label} shape tokens: obj {obj_idx}, {len(shapes)} frames\n"
        f"{indent}  aggregate: mean={agg.mean():.3f} var={agg.var():.3f} "
        f"min={agg.min():.2f} max={agg.max():.2f}\n"
        f"{indent}  per-frame mean: {pf_mean.mean():.3f}±{pf_mean.std():.3f} "
        f"(range=[{pf_mean.min():.3f}, {pf_mean.max():.3f}])\n"
        f"{indent}  per-frame var:  {pf_var.mean():.3f}±{pf_var.std():.3f} "
        f"(range=[{pf_var.min():.3f}, {pf_var.max():.3f}])\n"
        f"{indent}  per-frame min:  range=[{pf_min.min():.2f}, {pf_min.max():.2f}]\n"
        f"{indent}  per-frame max:  range=[{pf_max.min():.2f}, {pf_max.max():.2f}]"
    )

    logits_chunks = []
    with torch.no_grad():
        for s in shapes:
            z = shape_tokens_to_latent(s).to(s.device)
            logits_chunks.append(ss_decoder(z).reshape(-1))
    logits_all = torch.cat(logits_chunks)
    occ = logits_all > 0
    n_occ = int(occ.sum().item())
    n_emp = int((~occ).sum().item())
    msg = (
        f"{indent}  decoder logits (over {len(shapes)} frames × 64^3): "
        f"mean={logits_all.mean():.3f} min={logits_all.min():.2f} "
        f"max={logits_all.max():.2f}"
    )
    if n_occ > 0 and n_emp > 0:
        occ_l = logits_all[occ]
        emp_l = logits_all[~occ]
        frac_occ = n_occ / logits_all.numel()
        msg += (
            f"\n{indent}    occupied (logit>0): frac={frac_occ*100:.2f}%  "
            f"mean={occ_l.mean():.2f} max={occ_l.max():.2f}"
            f"\n{indent}    empty   (logit<=0): frac={(1-frac_occ)*100:.2f}%  "
            f"mean={emp_l.mean():.2f} min={emp_l.min():.2f}"
        )
    print(msg)


# =====================================================================
# GT_SHAPES_INVERSION: Per-frame GT mesh → Stage-1 shape latents
# =====================================================================

def _correspondence_voxel_colors(state, obj_idx, frame_int, gt_coords_int,
                                 grid_size=64):
    """Per-voxel RGB for per-frame GT voxels, coloured by the CANONICAL position
    each maps to through the fixed-topology mesh vertices — so a correct
    correspondence keeps every body part a constant colour across frames.

    Each voxel centre (frame-i self-norm) → nearest frame-i mesh vertex → that vertex's
    canonical position → ``clip(pos + 0.5)`` RGB.  Returns ``(N, 3)`` float32
    row-aligned to ``gt_coords_int``, or ``None`` when no deformation field is
    on state (e.g. mode=global) so the caller can fall back to xyz colouring.
    """
    canon_verts = state.canonical_mesh_verts.get(obj_idx)
    per_frame_verts = state.canonical_mesh_per_frame_verts.get(obj_idx)
    if canon_verts is None or per_frame_verts is None:
        return None
    frame_verts = per_frame_verts.get(int(frame_int))
    if frame_verts is None:
        return None
    from scipy.spatial import cKDTree

    def _np(t):
        return (t.detach().cpu().numpy() if hasattr(t, "detach")
                else np.asarray(t)).astype(np.float64)

    cv = _np(canon_verts)                                    # canonical-norm
    fv = _np(frame_verts)                                    # frame-i self-norm
    centers = (gt_coords_int.astype(np.float64) + 0.5) / grid_size - 0.5
    vstar = cKDTree(fv).query(centers, k=1)[1]               # nearest frame-i vertex
    return np.clip(cv[vstar] + 0.5, 0.0, 1.0).astype(np.float32)


def _peak_alloc_mb_now() -> float:
    """Peak CUDA allocation so far in the running block, in MB.

    Read WITHOUT resetting -- ``start_block`` owns that counter and ``end_block`` reads
    it back. Recorded beside ``generation_seconds`` so a run that HITS this shape cache
    reports what generating it needed, not what loading it needed."""
    import torch          # local: main.py imports torch lazily
    return (torch.cuda.max_memory_allocated() / 1024 ** 2
            if torch.cuda.is_available() else 0.0)


def run_gt_shapes_inversion(cfg, state, sequence, pipeline_obj, device):
    """GT_SHAPES_INVERSION: Invert GT meshes into Stage-1 shape latents and
    write into per-frame ``raw_ss_modalities``.  Optionally derive per-frame
    object poses from GT cameras (``load_gt_poses``).

    The first block.  Single source of truth for GT
    injection across both per-frame ActionMesh dynamic scenes and
    single-mesh GSO static scenes.  Branches on ``mode``:

    - ``per_frame_meshes``: invert one mesh per frame
      (``{root}/{scene}/obj_NNN/mesh_NN.glb``).  ActionMesh / OursActionBench
      layout.  ``load_gt_poses=true`` is rejected on ``davis_actionmesh`` (no
      GT camera/object poses); on OursActionBench it derives the poses from
      the fitted cameras in ``camera.json``.
    - ``global``: invert one mesh per scene
      (``{root}/{scene}/meshes/model.obj``).  Broadcasts the inverted
      latent across all frames AND populates
      ``state.canonical_shape_coords[obj_idx]`` for Stage-2 canonical
      consumers.  When ``load_gt_poses=true`` derives per-frame
      rotation/translation/scale + raw modalities from each frame's c2w
      (assumes static object at world origin in canonical orientation —
      EscherNet/GSO render setup).
    """
    if not cfg.gt_shapes_inversion.enabled:
        print("\n  GT_SHAPES_INVERSION: skipped (gt_shapes_inversion.enabled=false)")
        return

    cfg_block = cfg.gt_shapes_inversion
    # Run-order-indexed output subfolder, e.g. "01_gt_shapes_inversion".
    gt_block_subdir = block_output_subdir(cfg, "gt_shapes_inversion")
    os.makedirs(os.path.join(cfg.output.output_dir, gt_block_subdir),
                exist_ok=True)

    # Per-block cache: GT-mesh inversion is deterministic given the meshes,
    # so always reuse an existing cache.  No meta-based invalidation — delete
    # the cache dir (or set cfg.processing.resume_from_cache=false) to force a
    # rebuild after changing deformation knobs.
    cache_path = os.path.join(
        get_pipeline_cache_dir(cfg.output.output_dir),
        "gt_shapes_inversion.pt",
    )
    if cfg.processing.resume_from_cache and os.path.isfile(cache_path):
        cached = load_pipeline_cache(cache_path, cls=type(state))
        for f in cached.__dataclass_fields__:
            if f == "pipeline_obj":
                continue
            setattr(state, f, getattr(cached, f))
        # Mark the block as cache-loaded so the block loop in core/run.py skips the
        # redundant re-serialize.
        loaded = getattr(state, "_loaded_block_caches", set())
        loaded.add("gt_shapes_inversion")
        state._loaded_block_caches = loaded
        print(f"\nGT_SHAPES_INVERSION: skipped (loaded {cache_path})")
        return

    from genia.core.shape_inversion import resolve_autocast_dtype
    from genia.core.gt_geometry import (
        _resolve_temporal_surfaces_path, _resolve_gso_mesh_path,
        gt_shape_latent_from_mesh,
        gt_shape_latents_from_meshes,
        voxelize_mesh_to_canonical_coords,
        populate_gt_object_poses_in_state, populate_gt_raw_modalities_in_state,
    )

    mode = str(cfg_block.mode)
    if mode not in ("per_frame_meshes", "global"):
        raise ValueError(
            f"gt_shapes_inversion.mode={mode!r} - must be "
            f"'per_frame_meshes' or 'global'"
        )
    dataset_name = str(cfg.dataset.name).lower()
    if (cfg_block.load_gt_poses and mode == "per_frame_meshes"
            and dataset_name == "davis_actionmesh"):
        raise RuntimeError(
            f"gt_shapes_inversion.load_gt_poses=true is unsupported in "
            f"mode={mode!r} on dataset={dataset_name!r} "
            f"(ActionMesh has no GT camera/object poses)"
        )
    block_header(f"GT_SHAPES_INVERSION: mode={mode}, "
          f"load_gt_poses={bool(cfg_block.load_gt_poses)}")

    # By design we overwrite per-frame `raw_ss_modalities['shape']` and
    # invalidate dependent caches.  Warn whenever any decoder_input dict
    # already has content.
    state_has_content = any(
        bool(di) for tl in state.tokens_by_object.values() for _, di in tl
    )
    if state_has_content:
        warnings.warn(
            "GT_SHAPES_INVERSION: pipeline state is non-empty.  By design "
            "this block overwrites per-frame raw_ss_modalities['shape'] and "
            "invalidates cached per-frame SLATs / per-frame Gaussians / "
            "canonical SLAT / canonical Gaussian / canonical pose.  Pose "
            "modalities remain in raw_ss_modalities but were derived under "
            "the OLD shape; SHAPE_INIT / POSE_INIT will re-derive them."
        )

    ss_decoder = pipeline_obj.models["ss_decoder"]
    scene = str(cfg.dataset.scene_name)
    root = str(cfg_block.gt_data_root)
    subdir = str(cfg_block.scene_subdir)
    n_steps = int(cfg_block.num_inversion_steps)
    prior_w = float(cfg_block.prior_weight)
    prior_var = float(getattr(cfg_block, "prior_target_var", 1.0))
    l2_w = float(getattr(cfg_block, "l2_weight", 0.0))
    inv_dtype = resolve_autocast_dtype(
        str(getattr(cfg_block, "inversion_precision", "auto"))
    )
    chunk_sz = int(getattr(cfg_block, "inversion_chunk_size", 16))
    # Keep ``local_rotation`` as-is (str OR ListConfig of preset names);
    # ``actionbench_local_rotation_matrix`` handles both shapes.
    local_rotation = getattr(cfg_block, "local_rotation", "identity")
    n_inverted = 0

    for obj_idx, tokens_list in state.tokens_by_object.items():
        if obj_idx == 0:
            continue  # background has no GT mesh

        if mode == "per_frame_meshes":
            # GSO: single static mesh reused across all frames (1-frame
            # mono-dynamic test).  ActionMesh: per-frame meshes.
            gso_static_mesh_path = (
                _resolve_gso_mesh_path(scene, gt_data_root=root)
                if dataset_name == "gso" else None
            )
            if dataset_name == "gso" and gso_static_mesh_path is None:
                raise RuntimeError(
                    f"GT_SHAPES_INVERSION: GSO mesh not found for scene={scene} "
                    f"under {root} (mode=per_frame, dataset=gso)"
                )
            # Resolve every frame's mesh path up front (fail fast, before any
            # GPU work), then voxelise + invert ALL frames as ONE batched Adam
            # run.  GSO routed through per_frame_meshes resolves every frame to
            # the SAME model.obj and the inversion is seeded, so the N results
            # are bit-identical -- invert once and broadcast, as mode="global"
            # already does.
            frame_keys = [fk for fk, _ in tokens_list]
            is_static = gso_static_mesh_path is not None
            if is_static:
                mesh_paths = [gso_static_mesh_path]
            else:
                mesh_paths = [
                    _resolve_temporal_surfaces_path(
                        scene, int(obj_idx), int(fk.frame), gt_data_root=root,
                        scene_subdir=subdir,
                    )
                    for fk in frame_keys
                ]
            for mp in mesh_paths:
                if not mp.exists():
                    raise RuntimeError(
                        f"GT_SHAPES_INVERSION: mesh not found at {mp} "
                        f"(obj_idx={obj_idx})"
                    )

            latents, ious = gt_shape_latents_from_meshes(
                [str(mp) for mp in mesh_paths], ss_decoder, device,
                num_steps=n_steps, prior_weight=prior_w,
                prior_target_var=prior_var,
                l2_weight=l2_w,
                local_rotation=local_rotation,
                chunk_size=chunk_sz, autocast_dtype=inv_dtype,
            )
            if is_static:  # broadcast the single inversion to every frame
                n_fk = len(frame_keys)
                latents, ious = latents * n_fk, ious * n_fk
                mesh_paths = mesh_paths * n_fk

            inverted_latents = []
            shape_by_frame = {}
            for fk, mesh_path, latent, iou in zip(
                frame_keys, mesh_paths, latents, ious,
            ):
                shape_by_frame[fk] = latent.detach()
                inverted_latents.append(latent.detach())
                z = latent.squeeze(0)
                print(f"  obj {obj_idx} frame {fk.frame}: "
                      f"{mesh_path.name} -> IoU={iou:.3f}  "
                      f"z[mean={z.mean():.3f} var={z.var():.3f} "
                      f"min={z.min():.2f} max={z.max():.2f}]")
                n_inverted += 1
            _print_inversion_stats(
                inverted_latents, ss_decoder, obj_idx,
            )
            state.set_all_perframe_shape_tokens(obj_idx, shape_by_frame)
            state.invalidate_canonical(obj_idx)
            # GSO routed through per_frame_meshes: derive per-frame poses from c2w
            # exactly like the global branch (single static object at world
            # origin).  ActionMesh has no GT camera poses → guard rejected earlier.
            if cfg_block.load_gt_poses and dataset_name == "gso":
                assert gso_static_mesh_path is not None
                n_poses = populate_gt_object_poses_in_state(
                    state, sequence, str(gso_static_mesh_path), obj_idx, device,
                )
                n_raw = populate_gt_raw_modalities_in_state(
                    state, sequence, obj_idx, device, pipeline_obj,
                )
                print(f"  obj {obj_idx}: {n_poses} GT poses from c2w; "
                      f"{n_raw} raw Stage-1 modalities + SSI constants")
            # OursActionBench: known per-frame cameras (camera.json) baked into
            # per-frame object poses via populate_gt_object_poses_in_state_actionbench,
            # using per-frame mesh vertices (loaded from
            # obj_NNN/deformations_vertices.npy) as the surface points.
            elif cfg_block.load_gt_poses and dataset_name == "oursactionbench":
                from genia.core.gt_geometry import (
                    populate_gt_object_poses_in_state_actionbench,
                )
                from pathlib import Path as _Path
                deformations_path = (
                    _Path(root) / scene / f"obj_{int(obj_idx) - 1:03d}"
                    / "deformations_vertices.npy"
                )
                if not deformations_path.exists():
                    raise FileNotFoundError(
                        f"GT_SHAPES_INVERSION: load_gt_poses=true on "
                        f"oursactionbench requires {deformations_path}"
                    )
                verts_per_frame = np.load(deformations_path)  # (T, V, 3) Y-up
                surfaces_cache = {
                    fi: verts_per_frame[fi] for fi in range(verts_per_frame.shape[0])
                }
                n_poses = populate_gt_object_poses_in_state_actionbench(
                    state, obj_idx, surfaces_cache,
                    scene, device,
                    local_rotation=local_rotation,
                    data_root=root,
                )
                n_raw = populate_gt_raw_modalities_in_state(
                    state, sequence, obj_idx, device, pipeline_obj,
                )
                print(f"  obj {obj_idx}: {n_poses} GT poses from per-frame "
                      f"fitted cameras; {n_raw} raw Stage-1 modalities + "
                      f"SSI constants")

        else:  # mode == "global"
            # The dataset's single static GT mesh (GSO layout).
            mesh_path = _resolve_gso_mesh_path(scene, gt_data_root=root)
            if mesh_path is None:
                raise RuntimeError(
                    f"GT_SHAPES_INVERSION: shape mesh not found for "
                    f"scene={scene} under {root} (mode=global)"
                )
            # Voxelise once -> canonical_shape_coords (drops cached canonical
            # SLAT/Gaussian; the new coords feed Stage-2 canonical).
            coords = voxelize_mesh_to_canonical_coords(
                str(mesh_path), grid_size=64,
                dilate=int(cfg_block.gt_mesh_dilate),
                local_rotation=local_rotation,
            )
            state.invalidate_canonical(obj_idx, shape_coords=coords)
            state.gt_canonical_shape_coords[obj_idx] = coords
            print(f"  obj {obj_idx}: voxelised {mesh_path.name} -> "
                  f"{coords.shape[0]} canonical voxels "
                  f"(bbox=[{coords.min(0)} .. {coords.max(0)}])")
            # Invert once -> broadcast to every frame's raw_ss_modalities.
            latent, iou = gt_shape_latent_from_mesh(
                str(mesh_path), ss_decoder, device,
                num_steps=n_steps, prior_weight=prior_w,
                prior_target_var=prior_var,
                l2_weight=l2_w,
                local_rotation=local_rotation,
                autocast_dtype=inv_dtype,
            )
            z = latent.squeeze(0)
            print(f"  obj {obj_idx}: GT-mesh inversion IoU={iou:.3f} "
                  f"({n_steps} Adam steps)  "
                  f"z[mean={z.mean():.3f} var={z.var():.3f} "
                  f"min={z.min():.2f} max={z.max():.2f}]")
            _print_inversion_stats(
                [latent.detach()] * len(tokens_list), ss_decoder, obj_idx,
            )
            # Broadcast the single inverted latent to every frame.  A shared
            # canonical grid (state.canonical_shape_coords, set above) already
            # covers all frames, so skip per-frame coord decoding.
            state.set_all_perframe_shape_tokens(
                obj_idx,
                {fk: latent.detach() for fk, _ in tokens_list},
                decode_coords=False,
            )
            n_inverted += len(tokens_list)
            # Optional: object poses from each frame's c2w + raw Stage-1 modalities.
            if bool(getattr(cfg_block, "load_gt_poses", False)):
                n_poses = populate_gt_object_poses_in_state(
                    state, sequence, str(mesh_path), obj_idx, device,
                )
                populate_gt_raw_modalities_in_state(
                    state, sequence, obj_idx, device, pipeline_obj,
                )
                print(f"  obj {obj_idx}: {n_poses} GT poses + raw Stage-1 "
                      f"modalities + SSI constants")

        # Stale per-frame SLATs/Gaussians and the re-decoded
        # perframe_shape_coords are handled per-mode by
        # set_all_perframe_shape_tokens (called in both branches above).
        state.canonical_poses.pop(obj_idx, None)

    print(f"\nGT_SHAPES_INVERSION: wrote {n_inverted} per-frame shape latents")

    # ------------------------------------------------------------------
    # Derive per-object GT geometry from the same per-frame meshes.  Computed
    # automatically in ``mode=per_frame_meshes`` (temporal meshes available);
    # skipped in ``mode=global`` (single static mesh).
    #
    # Decoupling: this reads raw GT meshes off disk — it does NOT consume the
    # per-frame inverted SLAT latents from the loop above.  The per-canonical-
    # mesh-vertex field (``compute_canonical_mesh_correspondence``, natural mesh
    # resolution) is the single source: it feeds the rendering-guidance / keyframe
    # warp helpers AND carries the fixed-topology vertices from which the
    # per-frame voxel correspondence is DERIVED (no second disk read).  That
    # derivation (``compute_perframe_voxel_correspondence``) defines the canonical
    # voxel grid (``state.canonical_shape_coords`` + ``state.gt_canonical_shape_coords``)
    # and the per-frame GT voxel grids + per-frame-voxel→canonical correspondence
    # (``state.gt_perframe_voxel_correspondence``) feeding the
    # ``appearance_init=canonical_unified`` consumer.  Per-frame
    # ``raw_ss_modalities['shape']`` is left unchanged (Stage-1 still sees the
    # per-frame inversions).
    # ------------------------------------------------------------------
    if mode == "per_frame_meshes":
        import torch
        from genia.core.gt_geometry import (
            compute_perframe_voxel_correspondence,
            compute_canonical_mesh_correspondence,
        )

        canonical_frame_idx = int(getattr(cfg_block, "canonical_frame_idx", 0))
        k_kabsch = int(getattr(cfg_block, "k_kabsch", 8))
        subdivide_passes = int(getattr(cfg_block, "subdivide_passes", 0))
        pf_method = str(getattr(cfg_block, "perframe_voxelization", "surface"))
        pf_sample_edge = float(
            getattr(cfg_block, "perframe_voxel_sample_edge", 0.5)
        )

        print("\n" + "-" * 60)
        print(f"GT_SHAPES_INVERSION: computing deformation field "
              f"(canonical_frame_idx={canonical_frame_idx}, k_kabsch={k_kabsch}, "
              f"subdivide_passes={subdivide_passes}, "
              f"perframe_voxelization={pf_method})")
        print("-" * 60)

        n_provider_objs = 0
        for obj_idx, tokens_list in state.tokens_by_object.items():
            if obj_idx == 0:
                continue  # background has no GT mesh

            # Sorted unique temporal frame ints from this object's tokens_list.
            # Mono-dynamic: every slot has a distinct fk.frame.  MV-dynamic
            # (future): multiple slots share the same fk.frame.
            frame_indices = sorted({int(fk.frame) for fk, _ in tokens_list})
            if not frame_indices:
                continue
            if canonical_frame_idx not in frame_indices:
                raise ValueError(
                    f"gt_shapes_inversion.canonical_frame_idx={canonical_frame_idx} "
                    f"not present in tokens_list for obj_idx={obj_idx} "
                    f"(temporal frames: {frame_indices})"
                )

            static_mesh_path = (
                _resolve_gso_mesh_path(scene, gt_data_root=root)
                if dataset_name == "gso" else None
            )

            # Per-canonical-mesh-vertex correspondence (the high-res raw GT
            # signal — natural mesh resolution).  Consumed by Stage-2
            # appearance-init rendering guidance + the keyframe warp helpers;
            # also the fixed-topology
            # vertices the per-frame voxel correspondence is derived from.
            mesh_corr = compute_canonical_mesh_correspondence(
                scene_name=scene,
                obj_idx=int(obj_idx),
                frame_indices=frame_indices,
                canonical_frame_idx=canonical_frame_idx,
                gt_data_root=root,
                scene_subdir=subdir,
                k_kabsch=k_kabsch,
                local_rotation=local_rotation,
                static_mesh_path=static_mesh_path,
                subdivide_passes=subdivide_passes,
            )
            state.canonical_mesh_verts[obj_idx] = mesh_corr["canonical_verts"]
            state.canonical_mesh_faces[obj_idx] = mesh_corr["faces"]
            state.canonical_mesh_per_frame_verts[obj_idx] = mesh_corr["per_frame_verts"]
            state.canonical_mesh_per_frame_rotations[obj_idx] = mesh_corr["per_frame_rotations"]

            mdiag = mesh_corr["diagnostics"]
            print(
                f"  obj {obj_idx}: mesh-vertex correspondence V={mdiag['V']}, "
                f"F={mdiag['F']}, k_kabsch={mdiag['k_kabsch']}; "
                f"min_singular_value histogram (bins "
                f"{mdiag['min_singular_value_bins']}): "
                f"{mdiag['min_singular_value_histogram']}"
            )

            # Warp-free per-frame voxel correspondence, DERIVED from the fixed-
            # topology vertices above (no mesh re-load).  Direct assignment (NOT
            # invalidate_canonical) avoids popping pre-existing canonical_* state
            # from the per-frame inversion loop.
            canon_coords, pf_voxel_bundle = compute_perframe_voxel_correspondence(
                mesh_corr["canonical_verts"],
                mesh_corr["per_frame_verts"],
                mesh_corr["faces"],
                method=pf_method,
                sample_edge=pf_sample_edge,
                grid_size=64,
                dilate=int(getattr(cfg_block, "gt_mesh_dilate", 0)),
            )
            state.canonical_shape_coords[obj_idx] = canon_coords
            state.gt_canonical_shape_coords[obj_idx] = canon_coords
            state.gt_perframe_voxel_correspondence[obj_idx] = {
                fi: {
                    "coords": torch.from_numpy(coords_pf).to(torch.int64),
                    "pf_to_canon": torch.from_numpy(pf_to_canon).to(torch.int64),
                }
                for fi, (coords_pf, pf_to_canon) in pf_voxel_bundle.items()
            }
            _pf_sizes = [c.shape[0] for c, _ in pf_voxel_bundle.values()]
            print(
                f"  obj {obj_idx}: canonical voxels L_canon={canon_coords.shape[0]}; "
                f"per-frame GT voxel grids L_pf ∈ "
                f"[{min(_pf_sizes)}, {max(_pf_sizes)}] over "
                f"{len(frame_indices)} frames"
            )

            n_provider_objs += 1

        print(f"\nGT_SHAPES_INVERSION: deformation field for {n_provider_objs} object(s)")

        # per_frame_meshes explicitly requests a deformation field; if none
        # materialised the canonical would render with a frozen shape on every
        # frame.  Fail loudly here rather than emit degraded output at FINAL.
        # (Rigid-motion scenes need no field — they don't select this mode.)
        if not state.canonical_mesh_per_frame_verts:
            raise RuntimeError(
                "gt_shapes_inversion.mode='per_frame_meshes' requested a "
                "per-frame deformation field, but none was built "
                "(state.canonical_mesh_per_frame_verts is empty) — every frame "
                "would render with the frozen canonical shape. Expected "
                f"foreground object meshes under '{root}/{scene}/obj_NNN/'."
            )

    with _TIMER.exclude():
        if cfg.output.render_decoded and not cfg.output.suppress_intermediate_renders:
            from genia.core.utils.slat_decode import decode_shape_to_coords
            from genia.core.utils.visualization import render_slat_voxel_mesh

            renders_dir = os.path.join(
                cfg.output.output_dir, gt_block_subdir, "decoded_renders",
            )
            os.makedirs(renders_dir, exist_ok=True)

            # Cache once per (scene, root) for the global (GSO) mode — the
            # mesh is shared across frames + objects.
            gso_gt_voxels_cache = {}

            def _gt_voxels_for_frame(obj_idx_, fk_):
                """Return (L, 3) int voxel coords for the GT input used to
                invert the latent at this (obj, frame). Returns None if
                unavailable for this mode."""
                if mode == "per_frame_meshes":
                    mp = _resolve_temporal_surfaces_path(
                        scene, int(obj_idx_), int(fk_.frame), gt_data_root=root,
                        scene_subdir=subdir,
                    )
                    return voxelize_mesh_to_canonical_coords(
                        str(mp), grid_size=64,
                        dilate=int(cfg_block.gt_mesh_dilate),
                        local_rotation=local_rotation,
                    ) if mp.exists() else None
                if mode == "global":
                    if "_p" not in gso_gt_voxels_cache:
                        mp = _resolve_gso_mesh_path(scene, gt_data_root=root)
                        gso_gt_voxels_cache["_p"] = (
                            voxelize_mesh_to_canonical_coords(
                                str(mp), grid_size=64,
                                dilate=int(cfg_block.gt_mesh_dilate),
                                local_rotation=local_rotation,
                            ) if mp is not None else None
                        )
                    return gso_gt_voxels_cache["_p"]
                return None

            for obj_idx, tokens_list in state.tokens_by_object.items():
                if obj_idx == 0:
                    continue
                for fk, di in tokens_list:
                    shape_lat = di.get("raw_ss_modalities", {}).get("shape")
                    if shape_lat is None:
                        continue
                    coords = decode_shape_to_coords(pipeline_obj, shape_lat)
                    xyz = coords[:, 1:].cpu().numpy().astype(float)
                    stem = (f"{cfg.dataset.scene_name}_obj{obj_idx}"
                            f"_perframe_{fk.frame:03d}")
                    decoded_path = os.path.join(
                        renders_dir, f"{stem}_decoded_voxels.png",
                    )
                    render_slat_voxel_mesh(
                        xyz=xyz, output_path=decoded_path,
                        title=(f"{cfg.dataset.scene_name} obj {obj_idx} "
                               f"frame {fk.frame} decoded(z)"),
                        image_size=cfg.output.render_size,
                        distance=cfg.output.render_distance,
                        fov=cfg.output.render_fov,
                        save_obj=False,
                        color_mode="shape_pca",
                        shape_latent=shape_lat,
                    )

                    gt_xyz = _gt_voxels_for_frame(obj_idx, fk)
                    if gt_xyz is not None:
                        gt_path = os.path.join(
                            renders_dir, f"{stem}_gt_voxels.png",
                        )
                        # Colour each GT voxel by the canonical voxel it
                        # corresponds to (constant colour per body part across
                        # frames) when the deformation field is on state; else
                        # fall back to own-position colouring (e.g. mode=global).
                        corr_colors = _correspondence_voxel_colors(
                            state, obj_idx, int(fk.frame), gt_xyz, grid_size=64,
                        )
                        gt_title = (
                            f"{cfg.dataset.scene_name} obj {obj_idx} "
                            f"frame {fk.frame} GT target voxels "
                            + ("(canonical-correspondence colour)"
                               if corr_colors is not None
                               else "(input to inversion)")
                        )
                        render_slat_voxel_mesh(
                            xyz=gt_xyz.astype(float), output_path=gt_path,
                            title=gt_title,
                            image_size=cfg.output.render_size,
                            distance=cfg.output.render_distance,
                            fov=cfg.output.render_fov,
                            save_obj=False,
                            color_mode="xyz",  # used only when corr_colors is None
                            voxel_colors=corr_colors,
                        )

                    # GT-pose alignment: the canonical voxels posed by the
                    # GT pose just written to state, over the GT frame.
                    # Reuses the keyframes voxel renderer; identity c2w = the
                    # dataset's camera is baked into the per-frame pose.
                    if gt_xyz is not None and di.get("rotation") is not None:
                        import torch as _torch
                        from genia.core.utils.quaternion_ops import (
                            quaternion_to_matrix as _q2m,
                        )
                        from genia.core.utils.visualization import (
                            render_voxel_meshes_in_camera as _rvm,
                        )
                        _fd = sequence[fk]
                        _img = np.asarray(_fd.image)
                        _Hh, _Ww = _img.shape[:2]
                        _K = np.asarray(_fd.K_matrix, dtype=np.float32)
                        _rq = di["rotation"]
                        _rq = (_rq.detach().cpu() if isinstance(_rq, _torch.Tensor)
                               else _torch.tensor(_rq)).squeeze()
                        _rq = _rq / _rq.norm()
                        _R = _q2m(_rq.unsqueeze(0)).squeeze(0).numpy()

                        def _np3(v):
                            a = (v.detach().cpu().numpy()
                                 if isinstance(v, _torch.Tensor)
                                 else np.asarray(v)).reshape(-1)
                            return a
                        _tr = _np3(di["translation"])[:3]
                        _sc = _np3(di["scale"])
                        _vr = np.asarray(_rvm(
                            [{"voxel_coords_np": gt_xyz.astype(np.float32),
                              "obj_rotation": _R, "obj_translation": _tr,
                              "obj_scale": _sc}],
                            _K, int(_Ww), int(_Hh), c2w=None, grid_size=64,
                        ).convert("RGB"))
                        _fg = (_vr < 250).any(axis=2)
                        try:
                            _gm = np.asarray(_fd.masks[obj_idx]) > 0
                        except Exception:               # noqa: BLE001
                            _gm = None
                        _iou = (
                            float((_fg & _gm).sum() / max(int((_fg | _gm).sum()), 1))
                            if _gm is not None and _gm.shape == _fg.shape
                            else float("nan")
                        )
                        import matplotlib.pyplot as _plt
                        _disp = (_img if _img.dtype == np.uint8
                                 else np.clip(_img, 0.0, 1.0))
                        _f, _ax = _plt.subplots(
                            1, 3, figsize=(9, 3.2), constrained_layout=True,
                        )
                        _ax[0].imshow(_disp); _ax[0].set_title("GT", fontsize=8)
                        _ax[1].imshow(_vr)
                        _ax[1].set_title("posed voxels", fontsize=8)
                        _ax[2].imshow(_disp)
                        _ov = np.zeros((_Hh, _Ww, 4), np.float32)
                        _ov[_fg] = (1.0, 0.0, 0.0, 0.5)
                        _ax[2].imshow(_ov)
                        _ax[2].set_title(f"IoU={_iou:.3f}", fontsize=8)
                        for _a in _ax:
                            _a.set_xticks([]); _a.set_yticks([])
                        _f.suptitle(
                            f"{cfg.dataset.scene_name} obj {obj_idx} "
                            f"frame {fk.frame} GT-pose alignment", fontsize=9,
                        )
                        _f.savefig(os.path.join(
                            renders_dir, f"{stem}_alignment.png"), dpi=120)
                        _plt.close(_f)


# =====================================================================
# Canonical frame selection
# =====================================================================

def set_default_canon_frames(state):
    """Default every object's canonical frame to its first per-frame
    reconstruction (the ``FrameKey`` of ``tokens_list[0]``).

    Called by ``Pipeline.new_state`` on the fresh state skeleton.
    """
    for obj_idx, tokens_list in state.tokens_by_object.items():
        if tokens_list:
            first_fid = tokens_list[0][0]  # FrameKey of the first per-frame reconstruction
            state.canon_frame_per_object[obj_idx] = first_fid

# =====================================================================
# POSE_INIT: Pose Initialization
# =====================================================================


def run_shape_and_poses_init(cfg, state, sequence, inference, pipeline_obj, evaluator, device,
                             config_override, block_tag):
    """SHAPE_INIT / POSE_INIT: the batched Stage-1 ODE over every frame.

    Runs twice per pipeline: ``shape_init`` settles the shape (its pose output is kept
    only as a fallback), then ``pose_init`` denoises the pose against that shape and
    falls back to ``shape_init``'s pose for a frame it would flip
    (``pose_flip_guard_deg``).
    ``config_override`` is the block's config node, ``block_tag`` its name (output
    dirs / eval).  Skipped if ``enabled=False``.  Strategy ``parallel`` is the only
    one.
    """
    import torch

    sp_cfg = config_override

    block_title = block_tag.replace("_", " ").title()   # shape_init -> "Shape Init"

    if not getattr(sp_cfg, "enabled", True):
        block_header(f"{block_tag.upper()}: {block_title} — SKIPPED ({block_tag}.enabled=false)")
        return

    from genia.core.utils.depth_grounding import DepthObservation

    # How every stage in this block measures the object's depth-grounded (center, extent).
    # Built ONCE and passed down, because the seed and the refit re-measure the SAME
    # observation and the refit's answer replaces the seed's: read separately they could
    # drift (e.g. `disc_threshold` filtering the seed's pixels and not the refit's).
    # See depth_grounding.DepthObservation.
    _depth_obs = DepthObservation.from_pose_init_config(sp_cfg)

    if sp_cfg.strategy in ("none", None):
        raise ValueError(
            f"{block_tag}.strategy='none' is not a valid value. "
            f"Set {block_tag}.enabled=false to skip the block, "
            "or pick a valid strategy (parallel)."
        )

    block_output_dir = os.path.join(cfg.output.output_dir, block_output_subdir(cfg, block_tag))
    os.makedirs(block_output_dir, exist_ok=True)

    frame_indices = sequence.frame_indices

    block_header(f"{block_tag.upper()}: {block_title} (strategy={sp_cfg.strategy})")

    from genia.core.pose_init import (
        extract_canonical_shape_latent,
        run_parallel_shape_and_poses_init,
    )

    # SAM3D denoises ONE SHAPE PER FRAME, and on a dynamic sequence nothing
    # merges them: shape velocity averaging runs per timestamp (a no-op on
    # mono).  Those per-frame shapes ARE the reconstruction, so the canonical
    # collapse below is skipped and the run stays per-frame
    # (``has_canonical=False`` → FINAL's per-frame path).  Every other case
    # keeps building the canonical: static scenes (one timestamp) and
    # GT-injected shape (canonical grid + deformation field).
    _perframe_shape = (
        sequence.is_dynamic
        and not getattr(cfg.gt_shapes_inversion, "enabled", False)
    )

    # Pose velocity averaging collapses the frames' velocity into one temporal
    # mean per view — defensible only when the frames share ONE shape whose
    # orientation / size is a single common unknown (GT-injected canonical +
    # deformation field).  Under per-frame shape
    # predictions each frame denoises its own shape, so each frame's pose is
    # its own unknown and the coupling only drags them toward a common pose.
    def _vel_avg(name):
        value = getattr(sp_cfg, name)
        if _perframe_shape and value != "none":
            print(f"  {name} disabled (config: {value!r}) — per-frame shapes "
                  "denoise pose independently")
            return "none"
        return value

    _rot_vel_avg = _vel_avg("rotation_velocity_averaging")

    # ── Pose init search ──
    print("\n" + "-" * 40)
    print("Pose initialization search")
    print("-" * 40)

    strategy = sp_cfg.strategy

    if strategy == "parallel":
        _eff_shape_vel_avg = sp_cfg.shape_velocity_averaging
        print("\nParallel pose init: all frames in one batched ODE solve")
        if _eff_shape_vel_avg in ("none", "", None):
            print("  shape velocity averaging: OFF (each frame keeps its own velocity)")
        else:
            print(f"  shape velocity averaging: {_eff_shape_vel_avg} (per timestamp)")
    else:
        raise ValueError(f"Unknown pose init strategy: {strategy}")

    entropy_data_per_object = {}  # {obj_idx: entropy_data} for visualization

    for obj_idx in sorted(state.tokens_by_object.keys()):
        print(f"\n  Object {obj_idx}:")

        if strategy == "parallel":
            # Get canonical shape latent.  Priority order:
            #   0. GT shapes (when ``gt_shapes_inversion.enabled=True``) —
            #      stacks the per-frame latents from
            #      ``tokens_by_object[obj][i][1]['raw_ss_modalities']['shape']``
            #      (populated by the GT_SHAPES_INVERSION block) into
            #      ``(N, 4096, 8)``.  The downstream
            #      ``canonical_shape.expand(N, -1, -1)`` is a no-op when the
            #      tensor already has shape ``(N, P, F)`` — Stage-1's
            #      ``gt_shape_trajectory`` callback then sees per-frame
            #      ``_clean`` and converges each frame to its own GT shape.
            #      In ``mode=global`` the per-frame entries are all the same
            #      broadcast latent (single-mesh GSO path).
            #   1. Cached ``raw_ss_modalities['shape']`` from the canonical
            #      frame, when present.
            #   2. Fallback: ``extract_canonical_shape_latent`` runs a fresh
            #      Stage-1 pass on the canonical frame.
            parallel_shape_latent = None
            _gt_stack = getattr(cfg.gt_shapes_inversion, "enabled", False)
            # The per-frame stack is also the right anchor for a pass that
            # HOLDS the shape along the trajectory (``gt_shape_trajectory``): in
            # the mono_ours split, shape_init leaves each frame with its OWN
            # settled shape, so pose_init must hold that frame's shape, not the
            # anchor frame's.  Falls back to the cached canonical-frame latent
            # below when some frame has no shape yet.
            _traj_stack = getattr(sp_cfg, "gt_shape_trajectory", False)
            if _gt_stack or _traj_stack:
                _perframe = [
                    di.get("raw_ss_modalities", {}).get("shape")
                    for _, di in state.tokens_by_object[obj_idx]
                ]
                _complete = all(s is not None for s in _perframe)
                if _gt_stack and not _complete:
                    _missing = [
                        fk for (fk, _), s in
                        zip(state.tokens_by_object[obj_idx], _perframe) if s is None
                    ]
                    raise RuntimeError(
                        f"gt_shapes_inversion.enabled=true but obj {obj_idx} "
                        f"frame(s) {_missing} have no shape latent in "
                        f"raw_ss_modalities - did GT_SHAPES_INVERSION run?"
                    )
                if _complete:
                    parallel_shape_latent = torch.cat(_perframe, dim=0)  # (N, 4096, 8)
                    _src = "gt_shapes_inversion" if _gt_stack else "per-frame shapes"
                    print(f"    Using shape latents stack from {_src} (shape="
                          f"{tuple(parallel_shape_latent.shape)})")
            if parallel_shape_latent is None:
                canon_fid = state.canon_frame_per_object.get(obj_idx)
                if canon_fid is not None:
                    for fid, di in state.tokens_by_object[obj_idx]:
                        if fid == canon_fid and "raw_ss_modalities" in di:
                            parallel_shape_latent = di["raw_ss_modalities"].get("shape")
                            if parallel_shape_latent is not None:
                                print(f"    Using cached shape latent from "
                                      f"frame {canon_fid}")
                            break
            if parallel_shape_latent is None:
                print("    Cached shape not available, extracting via inference")
                parallel_shape_latent = extract_canonical_shape_latent(
                    obj_idx, state.canon_frame_per_object, sequence,
                    inference, cfg.processing.seed,
                )
            # Stage-1 ODE kwargs shared by the live run and the shape cache key.
            _temporal_kwargs = dict(
                inference_steps=getattr(sp_cfg, "inference_steps", None),
                entropy_alpha=getattr(sp_cfg, "entropy_alpha", 30.0),
                entropy_layer=getattr(sp_cfg, "entropy_layer", -1),
                entropy_min_weight=getattr(sp_cfg, "entropy_min_weight", 0.001),
                rotation_velocity_averaging=_rot_vel_avg,
                # Every diagnostic this block writes (trajectory plots) rides on the
                # block's resolved render flag, so a run with
                # output.suppress_intermediate_renders writes none of them.
                save_viz=get_block_output_flag(cfg, block_tag, "save_renders"),
            )

            if strategy == "parallel":
                # SAM3D shape pre-pass cache (parallel_shape / shape_init): the
                # joint solve's per-frame SHAPE output is cacheable because
                # pose_init re-denoises pose (a hit leaves its flip guard no fallback).
                # Gated on shape_cache_dir, which lives only in
                # parallel_shape.yaml.  See core/sam3d_shape_cache.py.
                from genia.core import sam3d_shape_cache as _ssc
                from genia.core.utils.disk_cache import credit_cached_work
                _shape_cache_path = None
                _shape_cached = None
                # Defined before the branch: the shape-cache hit below skips the ODE, and
                # the seat call after the branch reads this either way.
                _ref_only_fk = None
                _prior_pose = None
                if getattr(sp_cfg, "shape_cache_dir", None):
                    # Keying + lookup are cache bookkeeping, not pipeline
                    # compute (the key hashes every conditioning image).
                    with _TIMER.exclude():
                        _fks = [fk for fk, _ in state.tokens_by_object[obj_idx]]
                        _shape_cache_path = _ssc.cache_path(
                            sp_cfg.shape_cache_dir, sequence, obj_idx, _fks,
                            sp_cfg, cfg, parallel_shape_latent, cfg.processing.seed)
                        _shape_cached = _ssc.load(_shape_cache_path, device)

                if _shape_cached is not None:
                    print(f"    loaded cached shape_init shapes <- {_shape_cache_path}")
                    # The run still needed this shape: charge the block what the
                    # filling run measured, else a warm SHAPE_INIT reports ~0s.
                    credit_cached_work(_shape_cache_path,
                                       log_prefix="    [shape-cache]")
                    # Reproduce the per-frame post-conditions of the ODE that
                    # downstream blocks read: the shape token AND
                    # its decoded coords (via the shared pose_init helper, so this
                    # can't drift from the ODE).  Pose modalities stay as-is (a hit
                    # produces no shape_init pose, so pose_init's flip guard has no
                    # fallback).
                    from genia.core.pose_init import set_perframe_shape_coords
                    new_tokens_list = state.tokens_by_object[obj_idx]
                    for fk, di in new_tokens_list:
                        # A frame the pass skipped (no mask) carries no shape on
                        # the ODE path either, so it is absent from the blob.
                        _tok = _shape_cached["perframe_shape"].get(_ssc.fk_str(fk))
                        if _tok is None:
                            continue
                        di.setdefault("raw_ss_modalities", {})["shape"] = _tok
                        set_perframe_shape_coords(di, inference, _tok)
                    new_shape = _shape_cached["canonical_shape"]
                    obj_entropy_data = None
                else:
                    # ── Reference-only ODE (MV-static) ──────────────────────────
                    # Decided at the ODE call: the precondition inspects the raw pose
                    # tokens on the non-reference frames (written by an earlier Stage-1
                    # pass or GT pose injection).  Everything batch-aligned was already
                    # built full, so the restriction is one index applied at the call --
                    # which is also why the ordering question does not arise: a one-element
                    # batch cannot disagree with the sort inside the ODE.
                    _ref_only_fk = _reference_only_ode_reference(
                        cfg, sp_cfg, state, sequence, obj_idx)
                    _ode_tokens_in = state.tokens_by_object
                    _ode_shape_latent = parallel_shape_latent
                    if _ref_only_fk is not None:
                        _fks_now = [fk for fk, _ in state.tokens_by_object[obj_idx]]
                        _ref_pos = _fks_now.index(_ref_only_fk)
                        # Looked up fresh from the state at the call, not from an
                        # earlier capture of the token list.
                        _ode_tokens_in = {
                            obj_idx: [state.tokens_by_object[obj_idx][_ref_pos]]}
                        if _ode_shape_latent is not None and _ode_shape_latent.shape[0] > 1:
                            # Mandatory: the generator does `canonical_shape.expand(N,...)`,
                            # which raises rather than broadcasting on a mismatch.
                            _ode_shape_latent = _ode_shape_latent[_ref_pos:_ref_pos + 1]
                        print(f"    Reference-only ODE: {_ref_only_fk} stands in for "
                              f"{len(_fks_now)} view(s)")

                    # Each frame's pose before this pass, for the flip guard below.
                    if getattr(sp_cfg, "pose_flip_guard_deg", None) is not None:
                        _prior_pose = _snapshot_pose(state.tokens_by_object[obj_idx])
                    # Exclusion-aware: the pass saves its ODE-trajectory plots inside
                    # this call under _TIMER.exclude(), and a credited hit has to mean
                    # the same thing the live block charged -- not that plus the viz.
                    with measure_core_seconds() as _shape_span:
                        new_tokens_list, _, new_shape, obj_entropy_data = run_parallel_shape_and_poses_init(
                            obj_idx, _ode_tokens_in, state.canon_frame_per_object,
                            sequence, inference, cfg.processing.seed,
                            block_output_dir, cfg.dataset.scene_name,
                            canonical_shape_latent=_ode_shape_latent,
                            shape_velocity_averaging=sp_cfg.shape_velocity_averaging,
                            gt_shape_trajectory=getattr(sp_cfg, "gt_shape_trajectory", False),
                            cfg_interval_pose=getattr(sp_cfg, "cfg_interval_pose", None),
                            pose_velocity_broadcast_per_frame=getattr(sp_cfg, "pose_velocity_broadcast_per_frame", False),
                            **_temporal_kwargs,
                        )
                    if _shape_cache_path is not None:
                        _ssc.store(_shape_cache_path, new_tokens_list, new_shape,
                                   meta=_ssc.cache_meta(
                                       sequence, obj_idx,
                                       [fk for fk, _ in new_tokens_list],
                                       cfg.processing.seed,
                                       generation_seconds=_shape_span.seconds,
                                       peak_alloc_mb=_peak_alloc_mb_now()))
            if _ref_only_fk is not None and _shape_cached is None:
                # Splice the reference's whole dict in.  After this, in BOTH branches,
                # `new_tokens_list is state.tokens_by_object[obj_idx]` and it is the FULL
                # list -- which is what leaves the refit, the ICP, `_print_inversion_stats`
                # and the canonical bookkeeping below untouched.
                new_tokens_list = _seat_reference_only_pose_init(
                    state, obj_idx, new_tokens_list, _ref_only_fk)
            else:
                state.tokens_by_object[obj_idx] = new_tokens_list
            if _prior_pose:
                _guard_pose_flips(new_tokens_list, _prior_pose, sp_cfg.pose_flip_guard_deg)

            # ── Shared-world rebase, BEFORE the refit ────────────────────────────────
            # Per timestamp, each view's pose derived from that timestamp's reference.  It
            # belongs HERE and not only at the end of the block: the refit fits each view's
            # (t, s) against `di["rotation"]`, and the ODE hands every view of a timestamp
            # the SAME camera-space rotation (`pose_velocity_broadcast_per_frame` overwrites
            # their velocity from the lowest-view lead), which is correct for at most one
            # camera.  Refitting before the rebase would therefore take a consensus over V
            # views that disagree, biasing the fitted scale.
            # Runs on BOTH ODE branches; `mv_reference_only_ode` only decides how many views
            # the ODE visited, not how the answer is propagated.
            if (_shape_cached is None and sequence.is_mv
                    and bool(cfg.pipeline.mv_shared_world_pose)
                    and any("rotation" in di for _, di in new_tokens_list)):
                # The reference passed EXPLICITLY when the restriction picked one, so the two
                # cannot resolve differently; else resolved from the token list.
                from genia.core.utils.pipeline_state import resolve_reference_frame
                _rb_ref = _ref_only_fk if _ref_only_fk is not None else resolve_reference_frame(
                    new_tokens_list, state.canon_frame_per_object, obj_idx)
                _n_derived = state.rebase_perframe_from_reference(
                    obj_idx, sequence, _rb_ref, scope="timestamp")
                print(f"    Shared world pose: derived {_n_derived} non-reference "
                      f"view(s) from {_rb_ref}")
            if obj_entropy_data is not None:
                entropy_data_per_object[obj_idx] = obj_entropy_data

            # Post-ODE (translation, scale) refit against the observed depth.  It needs
            # the rotation the ODE has just produced: foreshortening and the
            # front-surface depth depend on it.  Rewrites the raw translation/scale
            # tokens only.
            _refit_mode = sp_cfg.post_rotation_refit
            if _refit_mode and _refit_mode != "none":
                # BEFORE shot, so the `end` twin below brackets the refit rather than
                # just ending the block.  The refit rewrites (t, s) -- and with
                # `mv_consensus` on it also fuses them across views -- which is exactly
                # the kind of change that is
                # invisible in a scalar residual and only shows up as the object sitting
                # at the wrong size or depth.  Emitted only when the refit will actually
                # run: otherwise it would duplicate `end` for every object.
                if not cfg.output.suppress_intermediate_renders:
                    _emit_pose_overlay_viz(
                        cfg, state, sequence, obj_idx, device, block_output_dir,
                        "pre_refit",
                    )
                from genia.core.pose_refit import apply_post_rotation_refit
                _n_refit = apply_post_rotation_refit(
                    state, sequence, obj_idx, device, pipeline_obj, _refit_mode,
                    mv_consensus=sp_cfg.post_rotation_refit_mv_consensus,
                    obs=_depth_obs,
                )
                print(f"    Post-rotation refit ({_refit_mode}): "
                      f"{_n_refit} frame(s) for obj {obj_idx}"
                      + ("" if _n_refit else " — no canonical shape yet, skipped"))

            # End-of-block posed-shape overlay, after the refit so it shows what the
            # block produced; paired with `_pre_refit.png` above, the difference
            # between the two IS the refit.
            if not cfg.output.suppress_intermediate_renders:
                _emit_pose_overlay_viz(
                    cfg, state, sequence, obj_idx, device, block_output_dir, "end",
                )

            # Stats on predicted shape tokens — calibrate against the GT
            # inversion to see the natural distribution Stage-1 produces.
            with _TIMER.exclude():
                _shapes = [
                    di["raw_ss_modalities"]["shape"].detach()
                    for _, di in new_tokens_list
                    if di.get("raw_ss_modalities", {}).get("shape") is not None
                ]
                _print_inversion_stats(
                    _shapes, pipeline_obj.models["ss_decoder"], obj_idx,
                    label="predicted",
                )

            # Update canonical raw modalities with new shape and poses.
            # set_canonical_raw_modalities auto-decodes:
            #   - layout tokens → canonical_poses
            #   - shape tokens → canonical_slats coords + canonical_shape_coords
            #   - invalidates canonical_gaussians if shape coords changed
            assert new_shape is not None, (
                "shape_and_poses_init must always return shape tokens"
            )

            # Per-frame SAM3D shape on a dynamic sequence: keep every frame's
            # own shape.  Writing the anchor frame's shape into the canonical
            # store (and over every frame below) would freeze the object at the
            # canonical timestamp and flip ``has_canonical``.  An object that
            # ALREADY has a canonical keeps it updated, so the later pose pass
            # must not orphan it.  Both stores are checked: ``set_canonical_slat``
            # pops ``canonical_shape_coords`` while filling ``canonical_slats``.
            _has_canonical_obj = (
                obj_idx in state.canonical_shape_coords
                or obj_idx in state.canonical_slats
            )
            if _perframe_shape and not _has_canonical_obj:
                print(f"    Per-frame shape kept for obj {obj_idx} "
                      f"(dynamic + per-frame SAM3D shape; canonical skipped)")
                continue

            # Build updated canonical raw modalities from new shape + anchor pose.
            canon_fid = state.canon_frame_per_object.get(obj_idx)
            updated_raw = dict(state.canonical_raw_modalities.get(obj_idx, {}))
            updated_raw["shape"] = new_shape.clone().detach()
            # Update layout tokens from anchor frame's new results
            if canon_fid is not None:
                for fid, di in new_tokens_list:
                    if fid == canon_fid and "raw_ss_modalities" in di:
                        for key in ("6drotation_normalized", "translation",
                                    "scale", "translation_scale"):
                            if key in di["raw_ss_modalities"]:
                                updated_raw[key] = (
                                    di["raw_ss_modalities"][key].clone().detach()
                                )
                        break
            # Actionmesh: when the GT-anchored canonical voxel grid exists
            # for this object, it is fixed by the provider — don't let the
            # Stage-1-averaged shape's decoded coords replace it (the
            # canonical_unified dynamic path relies on the GT grid being
            # preserved across the Stage-1 blocks).
            _keep_gt_canonical_grid = (
                state.gt_canonical_shape_coords.get(obj_idx) is not None
            )
            state.set_canonical_raw_modalities(
                obj_idx, updated_raw,
                decode_shape=not _keep_gt_canonical_grid,
            )
            print(f"    Updated canonical raw modalities for obj {obj_idx}")

            # Also update shape in perframe_raw_modalities for ALL frames.
            if obj_idx in state.perframe_raw_modalities:
                n_updated = 0
                for fid, raw in state.perframe_raw_modalities[obj_idx].items():
                    if "raw_ss_modalities" in raw:
                        raw["raw_ss_modalities"]["shape"] = new_shape.clone().detach()
                        n_updated += 1
                if n_updated:
                    print(f"    Updated shape in perframe_raw_modalities "
                          f"({n_updated} frames)")

    # MV shared world pose: collapse per-camera predictions to one shared local->world
    # pose derived from the reference frame. See `PipelineState.rebase_perframe_from_reference`.
    # Activates iff the data is multi-view AND the user opted in via mv_shared_world_pose.
    if sequence.is_mv and cfg.pipeline.mv_shared_world_pose:
        from genia.core.utils.frame_key import frame_key_sort_key
        print("\n" + "-" * 40)
        print("Shared world pose: rebasing per-frame poses from reference")
        print("-" * 40)
        for obj_idx in sorted(state.tokens_by_object.keys()):
            # Shape-only pass served from the SAM3D shape cache: nothing ever
            # decoded a pose into the entries (the ODE writes one as a side
            # effect; the cache hit skips the ODE), and pose_init rebases the
            # real poses.
            if not any("rotation" in di for _, di in state.tokens_by_object[obj_idx]):
                print(f"    obj {obj_idx}: no decoded pose — rebase skipped")
                continue
            n = state.rebase_perframe_from_reference(
                obj_idx, sequence, scope="timestamp")
            ref_frame = state.canon_frame_per_object.get(
                obj_idx,
                min((fk for fk, _ in state.tokens_by_object[obj_idx]),
                    key=frame_key_sort_key),
            )
            print(f"    obj {obj_idx}: rebased {n} non-reference frame(s) from {ref_frame}")

    with _TIMER.exclude():
        # Entropy visualization (if entropy weighting was used)
        if cfg.output.save_renders and entropy_data_per_object:
            for obj_idx, obj_entropy in entropy_data_per_object.items():
                if "entropy" not in obj_entropy:
                    continue
                from genia.core.visualization import (
                    plot_attention_weights,
                    prepare_entropy_viz_data,
                )

                entropy_np = obj_entropy["entropy"].numpy()
                N_views = entropy_np.shape[0]

                _stat_rows = _entropy_plot_frames(
                    state.tokens_by_object[obj_idx], obj_entropy, N_views)
                print(f"\n    Entropy statistics (obj {obj_idx}):")
                for v in range(N_views):
                    fid = _stat_rows[v][0]
                    h = entropy_np[v]
                    print(f"        frame {fid}: mean={h.mean():.4f}, "
                          f"std={h.std():.4f}, "
                          f"min={h.min():.4f}, max={h.max():.4f}")

                voxel_xyz = state.canonical_shape_coords.get(obj_idx)
                viz_xyz, entropy_np = prepare_entropy_viz_data(entropy_np, voxel_xyz)

                # The frames the entropy rows actually correspond to -- not necessarily
                # every frame of the object (see _entropy_plot_frames).
                tokens_list = _entropy_plot_frames(
                    state.tokens_by_object[obj_idx], obj_entropy, N_views)
                view_labels = [f"frame {fid}" for fid, _ in tokens_list]

                # Collect input images with pose overlay and masks
                from genia.core.utils.interpolation import compute_pose_axes
                from genia.core.utils.visualization import draw_pose_axes_on_image
                _pose_dict = {obj_idx: {fid: di for fid, di in tokens_list}}
                _fids = [fid for fid, _ in tokens_list]
                _axes_data = compute_pose_axes(_pose_dict, _fids)
                input_images = []
                for _vi, (fid, _) in enumerate(tokens_list):
                    img = sequence[fid].image
                    K = sequence[fid].K_matrix
                    img_t = torch.from_numpy(img).float() / 255.0
                    img_np = draw_pose_axes_on_image(img_t, _axes_data, _vi, K)
                    input_images.append((img_np * 255).astype(np.uint8))
                input_masks = [sequence[fid].masks[obj_idx] for fid, _ in tokens_list]

                # Save alongside ODE trajectory plots in the strategy subfolder
                _entropy_subdir = "parallel"
                entropy_path = os.path.join(
                    block_output_dir, _entropy_subdir,
                    f"{cfg.dataset.scene_name}_obj{obj_idx}_entropy.png",
                )
                plot_attention_weights(
                    viz_xyz, entropy_np, view_labels,
                    output_path=entropy_path,
                    mode="entropy",
                    input_images=input_images,
                    input_masks=input_masks,
                    title=f"{cfg.dataset.scene_name} obj {obj_idx} — Stage 1 cross-attention entropy",
                )

        # The pose_init pass only refines poses — shape/SLAT is untouched from
        # shape_init — so skip the decoded_renders/ Gaussian decode+render here.
        # shape_init still emits it.
        if block_tag != "pose_init":
            save_decoded_visualizations(
                state, cfg, pipeline_obj,
                output_dir=block_output_dir,
                tag=block_tag, frame_indices=sequence.frame_indices,
                canonical_only=False,
                voxel_color_mode="shape_pca",
            )

        # Skip evaluation after this block — poses are not yet refined enough
        # for meaningful metrics (evaluated after later refinement blocks instead)
        save_keyframes_video(
            state, sequence, cfg, suffix=block_tag,
            output_dir=block_output_dir,
        )
        save_keyframes_video(
            state, sequence, cfg, suffix=block_tag,
            output_dir=block_output_dir, color_mode="xyz",
            render_space="camera",
            voxel_color_mode="shape_pca",
        )
        has_c2w = any(
            not np.allclose(sequence[fi].c2w, np.eye(4))
            for fi in frame_indices
        )
        if has_c2w:
            save_keyframes_video(
                state, sequence, cfg, suffix=block_tag,
                output_dir=block_output_dir, color_mode="xyz",
                render_space="world",
                voxel_color_mode="shape_pca",
            )


# =====================================================================
# APPEARANCE_INIT
# =====================================================================

def _perframe_stage2_kwargs(cfg, ai_cfg, inference_steps, device):
    """``stage2_mv`` kwargs for a SINGLE-frame Stage-2 denoise, or ``None``.

    Shared by the ``perframe`` strategy and ``canonical_unified``'s dynamic
    path: both denoise one frame at a time, so cross-view fusion is trivial
    (1 view ⇒ visibility velocity weighting is a no-op and stays off)
    and the only things ``stage2_mv`` adds over the bare
    ``pipeline.sample_slat`` are the in-ODE hooks — the
    ``SparseVisibilityBiasHook`` and the rendering-guidance transform.

    Returns ``None`` when neither hook is active, which is the caller's signal
    to take the cheaper bare-sampler path.
    """
    import torch

    attn_bias_on = ai_cfg.visibility_attn_bias
    rg_on = ai_cfg.rendering_guidance_active
    if not (attn_bias_on or rg_on):
        return None

    return dict(
        inference_steps=inference_steps, seed=cfg.processing.seed,
        visibility_min_weight=ai_cfg.visibility_min_weight,
        visibility_attn_bias_alpha=ai_cfg.visibility_attn_bias_alpha,
        visibility_attn_bias_layers=ai_cfg.visibility_attn_bias_layers,
        visibility_attn_bias_streams={
            "cropped_image": ai_cfg.visibility_attn_bias_cropped_image,
            "full_image": ai_cfg.visibility_attn_bias_full_image,
            "cropped_mask": ai_cfg.visibility_attn_bias_cropped_mask,
            "full_mask": ai_cfg.visibility_attn_bias_full_mask,
        },
        visibility_attn_bias_compensate_passive_streams=
            ai_cfg.visibility_attn_bias_compensate_passive_streams,
        rendering_guidance_active=ai_cfg.rendering_guidance_active,
        rendering_guidance_velocity_weight=ai_cfg.rendering_guidance_velocity_weight,
        rendering_guidance_active_from=ai_cfg.rendering_guidance_active_from,
        rendering_guidance_active_until=ai_cfg.rendering_guidance_active_until,
        rendering_guidance_losses_cfg=(
            ai_cfg.rendering_guidance_losses if rg_on else None),
        rendering_guidance_bg_color=(
            torch.ones(3, device=device, dtype=torch.float32) if rg_on else None),
        rendering_guidance_microbatch_size=ai_cfg.rendering_guidance_microbatch_size,
        rendering_guidance_resolution_scale=ai_cfg.rendering_guidance_resolution_scale,
        rendering_guidance_normalize_grad=ai_cfg.rendering_guidance_normalize_grad,
        rendering_guidance_decoder_autocast_bf16=ai_cfg.rendering_guidance_decoder_autocast_bf16,
        rendering_guidance_random_background=ai_cfg.rendering_guidance_random_background,
        rendering_guidance_random_background_seed=ai_cfg.rendering_guidance_random_background_seed,
        rendering_guidance_visibility_detach=ai_cfg.rendering_guidance_visibility_detach,
        rendering_guidance_visibility_depth_margin=ai_cfg.rendering_guidance_visibility_depth_margin,
    )


def _stage2_frame_data(sequence, frame_idx, obj_idx):
    """Per-frame conditioning payload for ``stage2_mv``.

    ``K_matrix`` / ``c2w`` / ``depth_map_z`` are read only by in-ODE rendering
    guidance; cheap to always include.
    """
    frame_data = sequence[frame_idx]
    return {
        "image": frame_data.image,
        "mask": frame_data.masks[obj_idx],
        "K_matrix": frame_data.K_matrix,
        "c2w": getattr(frame_data, "c2w", None),
        "depth_map_z": getattr(frame_data, "depth_map_z", None),
    }


def _perframe_attn_bias_pixel_coords(obj_idx, coords_int, decoder_input,
                                     sequence, frame_idx, ai_cfg):
    """Voxel→patch projection for ONE frame's grid + pose (attn-bias input).

    ``compute_visibility_multi_object`` with a single object and a single view;
    returns the per-view pixel coords ``stage2_mv`` feeds to the
    ``SparseVisibilityBiasHook``.
    """
    from genia.core.visibility import compute_visibility_multi_object

    _, pixel_coords = compute_visibility_multi_object(
        {obj_idx: {
            "voxel_coords": coords_int,
            "rotations": [decoder_input["rotation"].detach().cpu().numpy().flatten()],
            "translations": [decoder_input["translation"].detach().cpu().numpy().flatten()],
            "scales": [decoder_input["scale"].detach().cpu().numpy().flatten()],
        }},
        K_matrices=[sequence[frame_idx].K_matrix],
        image_height=sequence.H, image_width=sequence.W,
        neighbor_tolerance=ai_cfg.visibility_neighbor_tolerance,
    )
    return pixel_coords.get(obj_idx)


def reset_shape_to_gt(state) -> int:
    """Replace predicted canonical shape with the GT voxelisation. Returns the
    number of objects reset -- ``0`` meaning there was no GT to inject.

    OPPORTUNISTIC, not mandatory.  With no GT voxelisation there is nothing to
    reset, so the predicted shape is kept and the run continues; the flag can
    therefore live in a shared config.

    Note: returning 0 does NOT mean consensus fusion will work on a predicted
    shape.  The dynamic ``canonical_unified`` path indexes ``pf_to_canon`` into
    the ROWS of the GT grid, so it still refuses a predicted grid in the
    strategy dispatch -- deliberately, because scattering per-frame velocities
    onto the wrong canonical rows is silently wrong output.
    """
    target_objs = sorted(state.gt_canonical_shape_coords.keys())
    if not target_objs:
        print("  reset_shape_to_gt: no GT voxelisation available "
              "(GT_SHAPES_INVERSION did not run) -> keeping the predicted "
              "shape")
        return 0
    for _oi in target_objs:
        _gt_coords = state.gt_canonical_shape_coords[_oi]
        # Drop canonical raw shape token
        if _oi in state.canonical_raw_modalities:
            state.canonical_raw_modalities[_oi].pop("shape", None)
        # Drop per-frame raw shape tokens (snapshot)
        for _raw in state.perframe_raw_modalities.get(_oi, {}).values():
            _raw.get("raw_ss_modalities", {}).pop("shape", None)
        # Drop per-frame raw shape tokens (live decoder_input)
        for _, _di in state.tokens_by_object.get(_oi, []):
            _di.get("raw_ss_modalities", {}).pop("shape", None)
        # Inject GT coords into the transient canonical_shape_coords cache and
        # invalidate canonical SLAT/Gaussians.  Φ/R are preserved when the new
        # coords match the existing entry (np.array_equal in
        # invalidate_canonical short-circuits the row-order check).
        state.invalidate_canonical(_oi, shape_coords=_gt_coords)
    print(f"  reset_shape_to_gt: dropped predicted shape tokens and injected "
          f"GT voxelisation from gt_canonical_shape_coords "
          f"({len(target_objs)} object(s))")
    return len(target_objs)


def _appearance_unified_is_static(tokens_by_object) -> bool:
    """True when the scene carries a single timestamp (no temporal axis).

    ``canonical_unified`` fuses over OBSERVATIONS: views in the static case,
    frames in the dynamic one.  With one timestamp the two coincide, so the
    strategy degenerates to ``canonical`` — which is also what lets a static
    run take the unified config without any GT correspondence.

    Static MV data (GSO/CO3D/mvcustom: every asset at frame 0) is static here;
    a mono- or MV-dynamic sequence is not.
    """
    frames = {fk.frame for tl in tokens_by_object.values() for fk, _ in tl}
    return len(frames) <= 1


def run_appearance_init(cfg, state, sequence, inference, pipeline_obj, evaluator, device,
                        config_override=None, block_label="APPEARANCE_INIT"):
    """APPEARANCE_INIT: Re-run Stage 2 Diffusion to predict appearance features.

    Per-frame mode:
        Re-run Stage 2 diffusion for each frame using that frame's own
        per-frame voxel coordinates.
        Overwrites ``decoder_input_slat`` in ``state.tokens_by_object`` and
        re-decodes per-frame Gaussians.  Poses and Stage 1 shape tokens are
        left untouched.

    Canonical mode (``canonical_unified``):
        Run Stage 2 once on a single canonical SLAT, fusing per-view
        velocities (static scene) or per-frame velocities on each frame's own
        GT grid (dynamic scene) at every ODE step.
    """
    ai_cfg = config_override if config_override is not None else cfg.appearance_init
    if not ai_cfg.enabled:
        return

    import torch
    from tqdm import tqdm

    strategy = ai_cfg.strategy

    block_header(f"{block_label}: Per-frame Appearance Update (strategy={strategy})")

    if strategy not in ("perframe", "canonical_unified"):
        raise ValueError(f"Unknown appearance_init.strategy: {strategy!r}")

    if strategy == "canonical_unified" and _appearance_unified_is_static(
            state.tokens_by_object):
        # Single-timestamp scene (static MV / GSO / CO3D / mvcustom): there is
        # no temporal axis to fuse over, so run the multi-view ``canonical``
        # path, which fuses the per-view velocities in one ODE.
        print("  canonical_unified: single-timestamp scene -> multi-view "
              "canonical path")
        strategy = "canonical"

    if strategy == "canonical_unified":
        if (not state.gt_canonical_shape_coords
                or not state.gt_perframe_voxel_correspondence):
            raise ValueError(
                f"appearance_init.strategy={strategy!r} requires the GT "
                "voxelized canonical grid (state.gt_canonical_shape_coords) and "
                "the per-frame GT voxel grids + correspondences "
                "(state.gt_perframe_voxel_correspondence) on the pipeline state."
                + "  This scene is DYNAMIC (>1 timestamp), so the unified "
                  "strategy needs the frame->canonical correspondence that a "
                  "per-frame GT_SHAPES_INVERSION mode populates; it runs "
                  "without injection only on static scenes."
            )

    block_dir_name = block_label.lower()
    block_output_dir = os.path.join(cfg.output.output_dir, block_output_subdir(cfg, block_dir_name))
    os.makedirs(block_output_dir, exist_ok=True)

    frame_indices = sequence.frame_indices

    inference_steps = ai_cfg.inference_steps

    # Capture renders BEFORE appearance update for before/after comparison.
    # perframe updates per-frame SLATs; canonical updates canonical SLAT.
    _eval_per_frame = (strategy == "perframe")
    with _TIMER.exclude():
        before_renders, before_poses = capture_before(
            state, sequence, cfg, per_frame=_eval_per_frame,
        )
        if before_renders:
            print(f"  Captured renders before {block_label}")

    # Optional reset to GT-voxelised shape: drop predicted shape tokens and
    # inject the stable GT mesh voxelisation from gt_canonical_shape_coords
    # (written once by GT_SHAPES_INVERSION, immune to appearance updates).
    if ai_cfg.reset_shape_to_gt:
        reset_shape_to_gt(state)

    # Determine which objects have canonical shape coords available
    # (either from valid SLAT or from canonical_shape_coords fallback).
    canon_obj_indices = set(state.canonical_slats.keys()) | set(state.canonical_shape_coords.keys())

    if strategy == "canonical_unified" and len(canon_obj_indices) > 1:
        # The dynamic path is single-object (no cross-object occlusion under
        # variable per-frame grids).
        raise NotImplementedError(
            f"{strategy} on a dynamic scene is single-object only; got "
            f"{len(canon_obj_indices)} canonical objects.  Use "
            f"appearance_init=perframe for multi-object dynamic scenes."
        )

    # Perframe strategy doesn't need canonical coords — iterate all objects
    # with per-frame tokens instead.
    target_obj_indices = (
        set(state.tokens_by_object.keys()) if strategy == "perframe"
        else canon_obj_indices
    )

    # Pre-compute multi-object DDA visibility (accounts for cross-object
    # occlusion) on the canonical grid + Stage-1 poses.  Feeds the velocity
    # weighting and the cross-attention bias of the static canonical path.
    _multi_obj_visibility: dict = {}
    _multi_obj_pixel_coords: dict = {}
    if strategy == "canonical" and (
            ai_cfg.visibility_weighting or ai_cfg.visibility_attn_bias):

        def _to_numpy(x):
            return x.cpu().numpy() if hasattr(x, "cpu") else np.array(x)

        _all_objects_data = {}
        for _oi in sorted(canon_obj_indices):
            _slat = state.canonical_slats.get(_oi)
            if _slat is not None:
                _coords = _slat.coords[:, 1:4].cpu().numpy()
            elif _oi in state.canonical_shape_coords:
                _coords = state.canonical_shape_coords[_oi].astype(np.int32)
            else:
                continue
            _tl = state.tokens_by_object[_oi]
            _all_objects_data[_oi] = {
                "voxel_coords": _coords,
                "rotations": [_to_numpy(di["rotation"]) for _, di in _tl],
                "translations": [_to_numpy(di["translation"]) for _, di in _tl],
                "scales": [_to_numpy(di["scale"]) for _, di in _tl],
            }

        # Per-view K matrices from the first object's frame list
        _first_tl = state.tokens_by_object[sorted(_all_objects_data.keys())[0]]
        _K_matrices = [sequence[fid].K_matrix for fid, _ in _first_tl]

        print(f"\n  Computing DDA visibility "
              f"({len(_all_objects_data)} objects, "
              f"{sequence.H}x{sequence.W})...")
        from genia.core.visibility import compute_visibility_multi_object
        _multi_obj_visibility, _multi_obj_pixel_coords = \
            compute_visibility_multi_object(
                _all_objects_data,
                K_matrices=_K_matrices,
                image_height=sequence.H,
                image_width=sequence.W,
                neighbor_tolerance=ai_cfg.visibility_neighbor_tolerance,
            )

    for obj_idx in sorted(target_obj_indices):
        tokens_list = state.tokens_by_object[obj_idx]

        if strategy == "perframe":
            # Re-run Stage 2 per-frame. Each frame's own existing SLAT coords
            # are used when available; otherwise fall back to canonical coords
            # (when per-frame SLATs were never populated). Poses and shape
            # tokens are unchanged.
            canon_slat = state.canonical_slats.get(obj_idx)
            canon_coords_fallback = None
            if canon_slat is not None:
                canon_coords_fallback = canon_slat.coords
            elif obj_idx in state.canonical_shape_coords:
                xyz_np = state.canonical_shape_coords[obj_idx]
                batch_col = np.zeros((xyz_np.shape[0], 1), dtype=np.int32)
                coords_np = np.hstack([batch_col, xyz_np.astype(np.int32)])
                canon_coords_fallback = torch.tensor(
                    coords_np, dtype=torch.int32, device=device,
                )

            # attn_bias + rendering_guidance act on each per-frame Stage-2
            # prediction.  When either is active, route the denoise through
            # stage2_mv (single view) — the existing function that wires the
            # SparseVisibilityBiasHook + rendering-guidance transform — instead
            # of the bare sample_slat; params mirror `canonical`.
            _s2_kwargs = _perframe_stage2_kwargs(
                cfg, ai_cfg, inference_steps, device,
            )
            _use_stage2 = _s2_kwargs is not None
            _attn_bias_on = ai_cfg.visibility_attn_bias

            print(f"\n  Object {obj_idx}: re-running Stage 2 on per-frame "
                  f"coords ({len(tokens_list)} frames, "
                  f"{inference_steps} steps"
                  + (f", via stage2_mv: attn_bias={_attn_bias_on}, "
                     f"rendering_guidance={ai_cfg.rendering_guidance_active}"
                     if _use_stage2 else "") + ")")
            perframe_slats = []
            for frame_idx, decoder_input in tqdm(
                tokens_list, desc=f"  Obj {obj_idx} perframe appearance"
            ):
                # Coord-source priority:
                #   1. ``perframe_shape_coords`` — set by the parallel pass
                #      (and its shape-cache hit path), OR by
                #      GT_SHAPES_INVERSION (per-frame modes).  Gives each
                #      frame its own voxel grid decoded from its per-frame
                #      shape token.  Required for true per-frame appearance
                #      under non-rigid (mono-dynamic, MV-dynamic) scenes.
                #   2. ``decoder_input_slat`` — set by an earlier appearance
                #      pass.  Falls through to its coords when present.
                #   3. canonical coords fallback — single grid for all frames.
                pf_coords = decoder_input.get("perframe_shape_coords")
                existing_slat = decoder_input.get("decoder_input_slat")
                if pf_coords is not None:
                    coords = pf_coords
                elif existing_slat is not None:
                    coords = existing_slat.coords
                elif canon_coords_fallback is not None:
                    coords = canon_coords_fallback
                else:
                    raise RuntimeError(
                        f"appearance_init=perframe requires voxel coords "
                        f"for obj {obj_idx} frame {frame_idx}, but neither "
                        f"a per-frame SLAT nor a canonical shape is "
                        f"available."
                    )
                frame_data = sequence[frame_idx]
                if _use_stage2:
                    # attn_bias needs the voxel→patch projection on THIS
                    # frame's grid + pose; rendering_guidance needs only this
                    # frame's pose decoder.  stage2_mv preprocesses the frame
                    # internally (no manual merge_mask_to_rgba).
                    _pf_pc = None
                    if _attn_bias_on:
                        _coords_int = coords.detach().cpu().numpy().astype(np.int32)
                        if _coords_int.shape[1] == 4:
                            _coords_int = _coords_int[:, 1:]
                        _pf_pc = _perframe_attn_bias_pixel_coords(
                            obj_idx, _coords_int, decoder_input,
                            sequence, frame_idx, ai_cfg,
                        )

                    # Single-frame pose decoder: this frame's (R, t, s) as
                    # (1, …) batches, matching the one-element frames list.
                    def _pf_pose_decoder(_d=decoder_input):
                        return (_d["rotation"].reshape(1, 4),
                                _d["translation"].reshape(1, 3),
                                _d["scale"].reshape(1, -1))

                    slat, _, _ = inference.stage2_mv(
                        [_stage2_frame_data(sequence, frame_idx, obj_idx)],
                        coords,
                        visibility_pixel_coords=_pf_pc,
                        rendering_guidance_pose_decoder=_pf_pose_decoder,
                        **_s2_kwargs,
                    )
                else:
                    rgba = inference.merge_mask_to_rgba(
                        frame_data.image, frame_data.masks[obj_idx],
                    )
                    slat_input_dict = pipeline_obj.preprocess_image(
                        rgba, pipeline_obj.slat_preprocessor,
                    )
                    with torch.no_grad():
                        slat = pipeline_obj.sample_slat(
                            slat_input_dict, coords,
                            inference_steps=inference_steps,
                        )
                perframe_slats.append(slat)
            slats_by_frame = {tokens_list[i][0]: perframe_slats[i]
                              for i in range(len(tokens_list))}
            state.set_all_perframe_slats(obj_idx, slats_by_frame)
            continue

        slat = state.canonical_slats.get(obj_idx)
        if slat is not None:
            canonical_coords = slat.coords
        elif obj_idx in state.canonical_shape_coords:
            # Reconstruct integer coords tensor from numpy fallback
            xyz_np = state.canonical_shape_coords[obj_idx]
            batch_col = np.zeros((xyz_np.shape[0], 1), dtype=np.int32)
            coords_np = np.hstack([batch_col, xyz_np.astype(np.int32)])
            canonical_coords = torch.tensor(coords_np, dtype=torch.int32,
                                            device=device)
        else:
            continue

        print(f"\n  Object {obj_idx}: re-predicting appearance on "
              f"{canonical_coords.shape[0]} canonical voxels")

        if strategy in ("canonical", "canonical_unified"):
            # Update canonical layout tokens from refined per-frame poses
            if state.update_canonical_layout_from_perframe(obj_idx):
                print(f"    Updated canonical layout tokens from refined "
                      f"per-frame poses (frame {state.canon_frame_per_object.get(obj_idx)})")

            print(f"    Running Stage 2 diffusion ({inference_steps} steps, "
                  f"{len(tokens_list)} frames, strategy={strategy})")

            frames_data = [
                _stage2_frame_data(sequence, frame_idx, obj_idx)
                for frame_idx, _ in tokens_list
            ]

            # Pre-computed DDA visibility (binary 0/1) for velocity weighting
            # and the voxel→pixel coords for the cross-attention bias.
            visibility_weights = None
            if ai_cfg.visibility_weighting and obj_idx in _multi_obj_visibility:
                visibility_weights = torch.from_numpy(
                    _multi_obj_visibility[obj_idx]).float().to(device)
            _pixel_coords_for_bias = None
            if ai_cfg.visibility_attn_bias:
                _pixel_coords_for_bias = _multi_obj_pixel_coords.get(obj_idx)

            # Build pose_decoder closure for in-ODE rendering guidance.
            # Returns per-frame (quat[N,4], trans[N,3], scale[N,3]) tensors,
            # in tokens_list order to match frames_data.  Reads the decoded
            # pose from each decoder_input (see below).
            _rg_pose_decoder = None
            _rg_losses_cfg = None
            if ai_cfg.rendering_guidance_active:
                _rg_tokens_list = list(tokens_list)

                # Reads decoded (rotation, translation, scale) from
                # decoder_input — populated by either Stage 1's pose_decoder
                # or gt_shapes_inversion.load_gt_poses.  Re-decoding from raw_ss_modalities
                # is unnecessary during APPEARANCE_INIT (raw modalities are
                # static; cached decoded values are equivalent).
                def _rg_pose_decoder(_tokens=_rg_tokens_list):
                    quats, transes, scales = [], [], []
                    for _fk, _di in _tokens:
                        quats.append(_di["rotation"])
                        transes.append(_di["translation"])
                        scales.append(_di["scale"])
                    return (
                        torch.stack([q.squeeze(0) if q.dim() == 2 else q for q in quats]),
                        torch.stack([t.squeeze(0) if t.dim() == 2 else t for t in transes]),
                        torch.stack([s.squeeze(0) if s.dim() == 2 else s for s in scales]),
                    )

                _rg_losses_cfg = ai_cfg.rendering_guidance_losses
                _rg_bg_color = torch.ones(3, device=device, dtype=torch.float32)
            else:
                _rg_bg_color = None

            # Which token entries the Stage-2 rendering-guidance history covers,
            # for the loss/depth-error plots in the shared tail.  Every branch
            # must set it: `canonical` guides ONE transform over its whole
            # conditioning set, while the per-frame paths build one transform
            # per frame and surface frame 0's trajectory.
            _s2_rg_frames = tokens_list[:1]

            if strategy == "canonical_unified":
                # Consensus over frames: each frame is denoised on its OWN GT
                # voxelisation (state.gt_perframe_voxel_correspondence) and the
                # per-frame velocities are reduced onto the canonical grid at every
                # ODE step (collapse_perframe_to_canonical, visibility-weighted).
                # Per-frame visibility — dda only (single object ⇒ pure
                # self-occlusion).  When enabled it weights the
                # per-frame→canonical collapse below.
                _vis_enabled = ai_cfg.visibility_weighting
                _perframe_vis = []  # (L_pf,) per frame, aligned with tokens_list

                L_canon = canonical_coords.shape[0]
                _pf_bundle = state.gt_perframe_voxel_correspondence[obj_idx]

                # Per-frame DDA self-occlusion on each frame's own GT grid, in a
                # separate pass so the DDA logs don't interleave with the
                # sampling progress bar.  Stored aligned with tokens_list.
                if _vis_enabled:
                    from genia.core.visibility import compute_visibility_for_all_views
                    for _fk, _di in tokens_list:
                        _vc = _pf_bundle[_fk.frame]["coords"].cpu().numpy()
                        _vis_pf = compute_visibility_for_all_views(
                            _vc.astype(np.int32),
                            [_di["rotation"].detach().cpu().numpy().flatten()],
                            [_di["translation"].detach().cpu().numpy().flatten()],
                            [_di["scale"].detach().cpu().numpy().flatten()],
                            grid_size=64,
                            neighbor_tolerance=ai_cfg.visibility_neighbor_tolerance,
                        )[0]                                      # (L_pf,)
                        _perframe_vis.append(_vis_pf)
                    _fracs = [float(v.mean()) for v in _perframe_vis]
                    print(
                        f"    obj {obj_idx}: per-frame DDA visibility computed "
                        f"for {len(_perframe_vis)} frames (mean visible fraction "
                        f"{sum(_fracs) / max(len(_fracs), 1):.3f}); weighting the "
                        f"canonical collapse"
                    )
                # attn_bias + rendering_guidance act on each per-frame Stage-2
                # prediction (before the collapse).  When either is active, route
                # the per-frame denoise through stage2_mv (single view) — the
                # existing function that wires the SparseVisibilityBiasHook +
                # rendering-guidance transform — instead of the bare sample_slat.
                # Cross-view fusion is trivial (1 view); cross-FRAME visibility
                # stays in collapse_perframe_to_canonical.
                _attn_bias_on = ai_cfg.visibility_attn_bias
                _s2_kwargs = _perframe_stage2_kwargs(
                    cfg, ai_cfg, inference_steps, device,
                )
                _use_stage2 = _s2_kwargs is not None
                if _use_stage2:
                    print(f"    obj {obj_idx}: per-frame Stage-2 via stage2_mv "
                          f"(attn_bias={_attn_bias_on}, rendering_guidance="
                          f"{ai_cfg.rendering_guidance_active})")

                # pf_to_canon addresses rows of the GT voxelisation, so the
                # canonical grid MUST be that same grid.  It normally is
                # because canonical_unified.yaml sets reset_shape_to_gt; check the coords themselves
                # rather than the flag, so a grid that already matches is
                # accepted and a predicted one is refused loudly instead of
                # scattering velocities onto the wrong rows.
                if obj_idx not in state.gt_canonical_shape_coords:
                    # Reachable because reset_shape_to_gt degrades on a
                    # missing GT voxelisation, so a predicted-shape run gets
                    # this far.  Say what is wrong rather than KeyError-ing on
                    # the dict lookup below.
                    raise ValueError(
                        f"canonical_unified (dynamic) obj {obj_idx}: no GT "
                        f"voxelisation is available, but this strategy "
                        f"fuses per-frame velocities through "
                        f"`pf_to_canon`, which indexes ROWS of the GT grid "
                        f"-- there is nothing for it to index into. Run "
                        f"GT_SHAPES_INVERSION in a per-frame mode "
                        f"(gt_shapes_inversion=actionmesh / oursactionbench)."
                    )
                _gt_xyz = np.asarray(
                    state.gt_canonical_shape_coords[obj_idx])
                _cur_xyz = (canonical_coords[:, 1:]
                            if canonical_coords.shape[1] == 4
                            else canonical_coords).cpu().numpy()
                if not np.array_equal(
                        _cur_xyz.astype(np.int64), _gt_xyz.astype(np.int64)):
                    raise ValueError(
                        f"canonical_unified (dynamic) obj {obj_idx}: the "
                        f"canonical grid ({_cur_xyz.shape[0]} voxels) is not "
                        f"the GT voxelisation ({_gt_xyz.shape[0]} voxels) "
                        f"that gt_perframe_voxel_correspondence's "
                        f"pf_to_canon indexes into, so per-frame velocities "
                        f"would land on the wrong canonical rows.  Set "
                        f"appearance_init.reset_shape_to_gt=true (the "
                        f"canonical_unified config group sets it)."
                    )

                # CONSENSUS-CANONICAL: ONE canonical latent, fused across
                # frames at EVERY ODE step — the dynamic analogue of the
                # `canonical` strategy's per-view velocity fusion, rather
                # than N independent ODEs collapsed once at the end.  Each
                # step gathers the canonical latent onto every frame's own
                # deformed grid through pf_to_canon, evaluates that frame's
                # CFG velocity (with its own attn-bias + rendering
                # guidance, applied PRE-fusion), collapses the velocities
                # back with the same reducer used below, and Euler-steps
                # the canonical alone.  Same 2*N forwards per step as the
                # per-frame path, so no extra cost.
                _entries = []
                for _vi, (_fk, _di) in enumerate(tokens_list):
                    _entry = _pf_bundle[_fk.frame]
                    coords_int = _entry["coords"].cpu().numpy().astype(np.int32)
                    _pf_pc = None
                    if _attn_bias_on:
                        _pf_pc = _perframe_attn_bias_pixel_coords(
                            obj_idx, coords_int, _di, sequence, _fk, ai_cfg,
                        )

                    def _pf_pose_decoder(_d=_di):
                        return (_d["rotation"].reshape(1, 4),
                                _d["translation"].reshape(1, 3),
                                _d["scale"].reshape(1, -1))

                    _entries.append({
                        "coords": torch.tensor(
                            np.pad(coords_int, ((0, 0), (1, 0))),
                            dtype=torch.int32, device=device,
                        ),
                        "pf_to_canon": _entry["pf_to_canon"],
                        "visibility": (
                            _perframe_vis[_vi] if _vis_enabled else None),
                        "pixel_coords": _pf_pc,
                        "pose_decoder": _pf_pose_decoder,
                    })

                _cons_kwargs = dict(_s2_kwargs) if _s2_kwargs else dict(
                    inference_steps=inference_steps,
                    seed=cfg.processing.seed,
                )
                # Batching is the driver's own knob, deliberately NOT part
                # of _perframe_stage2_kwargs: that builder is shared with
                # the per-frame stage2_mv path, which takes no frame_chunk.
                _cons_kwargs.update(
                    frame_chunk=ai_cfg.consensus_frame_chunk,
                    visibility_alpha=ai_cfg.visibility_alpha,
                    visibility_min_weight=ai_cfg.visibility_min_weight,
                    capture_slat_snapshots=(
                        bool(getattr(ai_cfg, "save_ode_steps_viz", False))
                        and get_block_output_flag(
                            cfg, block_dir_name, "save_renders")
                    ),
                )
                mv_slat, s2_entropy_data, s2_ode_history = (
                    inference.stage2_dyn(
                        frames_data, canonical_coords, _entries,
                        **_cons_kwargs,
                    )
                )

                # Per-frame visibility plot (same as the `canonical` strategy's
                # Stage-2 visibility plot, for visual debugging of the collapse):
                # which canonical voxels each frame sees, scattered from the
                # per-frame DDA visibility onto the canonical grid via pf_to_canon.
                # Only when visibility was computed (weighting on).
                if _vis_enabled and cfg.output.save_renders:
                    with _TIMER.exclude():
                        from genia.core.visualization import save_perframe_voxel_visibility_plot

                        _coords_np = canonical_coords[:, 1:].cpu().numpy() \
                            if canonical_coords.shape[1] == 4 \
                            else canonical_coords.cpu().numpy()
                        # Per-frame visibility on the canonical grid: a canonical
                        # voxel is "seen" by a frame if any per-frame voxel landing
                        # on it is visible (scatter-max via pf_to_canon).
                        _vis_canon = np.zeros((len(tokens_list), L_canon),
                                              dtype=np.float32)
                        for _vi, (_fk, _) in enumerate(tokens_list):
                            _p2c = _pf_bundle[_fk.frame]["pf_to_canon"].cpu().numpy()
                            np.maximum.at(_vis_canon[_vi], _p2c, _perframe_vis[_vi])

                        _canon_dir = os.path.join(block_output_dir, "canonical")
                        save_perframe_voxel_visibility_plot(
                            _coords_np, _vis_canon, tokens_list, sequence, obj_idx,
                            output_path=os.path.join(_canon_dir, f"{cfg.dataset.scene_name}_obj{obj_idx}_visibility.png"),
                            title=f"{cfg.dataset.scene_name} obj {obj_idx} — {strategy} per-frame visibility")
                        print(f"    obj {obj_idx}: saved per-frame visibility plot")
            else:
                # Conditioning set: collapse the TEMPORAL axis onto the canonical
                # timestamp, keep the VIEW axis.  For dynamic scenes the
                # off-canonical frames show deformed geometry, so averaging them
                # onto the canonical (frame-0) grid blurs appearance; the warp is
                # identity at the canonical timestamp, so the conditioning grid
                # and image share geometry (attn-bias/visibility projection exact).
                # Static multi-view (GSO/CO3D: all views at frame 0) is unaffected
                # — every view stays, so multi-view aggregation is preserved.
                _canon_fk = state.canon_frame_per_object.get(obj_idx)
                _canon_t = (_canon_fk.frame if _canon_fk is not None
                            else tokens_list[0][0].frame)
                _cond_pos = [i for i, (fk, _) in enumerate(tokens_list)
                             if fk.frame == _canon_t]
                _cond_tokens_list = [tokens_list[i] for i in _cond_pos]
                _cond_frames_data = [frames_data[i] for i in _cond_pos]
                _cond_vis_weights = (visibility_weights[_cond_pos]
                                     if visibility_weights is not None else None)
                _cond_pixel_coords = (
                    [_pixel_coords_for_bias[i] for i in _cond_pos]
                    if _pixel_coords_for_bias is not None else None)
                # Rendering-guidance pose decoder restricted to the conditioning
                # frames — reuse the shared closure with the sliced token list
                # (it takes the token list as its first arg; stage2_mv calls the
                # decoder with no args, so wrap it).
                _cond_rg_pose_decoder = (
                    (lambda: _rg_pose_decoder(_cond_tokens_list))
                    if _rg_pose_decoder is not None else None)
                # One guidance transform over the whole conditioning set here.
                _s2_rg_frames = _cond_tokens_list
                if len(_cond_pos) < len(tokens_list):
                    print(f"    canonical: conditioning on {len(_cond_pos)} view(s) "
                          f"at canonical timestamp {_canon_t} (temporal axis "
                          f"collapsed; {len(tokens_list) - len(_cond_pos)} "
                          f"off-canonical frame(s) discarded)")

                # Run multi-view Stage 2 — produces single averaged SLAT
                mv_slat, s2_entropy_data, s2_ode_history = inference.stage2_mv(
                    _cond_frames_data,
                    canonical_coords,
                    inference_steps=inference_steps,
                    seed=cfg.processing.seed,
                    visibility_min_weight=ai_cfg.visibility_min_weight,
                    fused_min_weight=ai_cfg.fused_min_weight,
                    visibility_weights=_cond_vis_weights,
                    visibility_alpha=ai_cfg.visibility_alpha,
                    visibility_pixel_coords=_cond_pixel_coords,
                    visibility_attn_bias_alpha=ai_cfg.visibility_attn_bias_alpha,
                    visibility_attn_bias_layers=ai_cfg.visibility_attn_bias_layers,
                    visibility_attn_bias_streams={
                        "cropped_image": ai_cfg.visibility_attn_bias_cropped_image,
                        "full_image": ai_cfg.visibility_attn_bias_full_image,
                        "cropped_mask": ai_cfg.visibility_attn_bias_cropped_mask,
                        "full_mask": ai_cfg.visibility_attn_bias_full_mask,
                    },
                    visibility_attn_bias_compensate_passive_streams=
                        ai_cfg.visibility_attn_bias_compensate_passive_streams,
                    visibility_attn_debug=cfg.output.save_renders,
                    capture_dino_features=(
                        cfg.output.save_renders and ai_cfg.visibility_attn_bias
                    ),
                    rendering_guidance_active=ai_cfg.rendering_guidance_active,
                    rendering_guidance_velocity_weight=ai_cfg.rendering_guidance_velocity_weight,
                    rendering_guidance_active_from=ai_cfg.rendering_guidance_active_from,
                    rendering_guidance_active_until=ai_cfg.rendering_guidance_active_until,
                    rendering_guidance_losses_cfg=_rg_losses_cfg,
                    rendering_guidance_pose_decoder=_cond_rg_pose_decoder,
                    rendering_guidance_bg_color=_rg_bg_color,
                    rendering_guidance_microbatch_size=ai_cfg.rendering_guidance_microbatch_size,
                    rendering_guidance_resolution_scale=ai_cfg.rendering_guidance_resolution_scale,
                    rendering_guidance_normalize_grad=ai_cfg.rendering_guidance_normalize_grad,
                    rendering_guidance_decoder_autocast_bf16=ai_cfg.rendering_guidance_decoder_autocast_bf16,
                    rendering_guidance_random_background=ai_cfg.rendering_guidance_random_background,
                    rendering_guidance_random_background_seed=ai_cfg.rendering_guidance_random_background_seed,
                    rendering_guidance_visibility_detach=ai_cfg.rendering_guidance_visibility_detach,
                    rendering_guidance_visibility_depth_margin=ai_cfg.rendering_guidance_visibility_depth_margin,
                    # actionmesh: high-res per-canonical-mesh-vertex deformation
                    # field for the rendering-guidance builder (rigid-LBS warp
                    # at decoded primitive positions via _warp_at_high_res).
                    # ``_build_stage2_mesh_warp_kwargs`` returns ``{}`` for non-actionmesh,
                    # so the splat is a no-op outside the actionmesh path.
                    **_build_stage2_mesh_warp_kwargs(
                        state, obj_idx, _cond_tokens_list, device, cfg.deformation_warp
                    ),
                    # Per-ODE-step SLAT capture for the appearance-ode-steps
                    # debug viz.  These snapshots have no consumer besides the viz
                    # (Stage-2 rendering guidance decodes the Tweedie SLAT itself),
                    # so the viz knob IS the capture switch — times the block's
                    # RESOLVED render flag, which honours
                    # output.suppress_intermediate_renders and the per-block
                    # override.
                    capture_slat_snapshots=(
                        bool(getattr(ai_cfg, "save_ode_steps_viz", False))
                        and get_block_output_flag(cfg, block_dir_name, "save_renders")
                    ),
                )

            with _TIMER.exclude():
                # Conditioning frames as pose-overlaid RGB + their masks for
                # the attention-bias debug plot (canonical-only, so
                # `_cond_tokens_list` is in scope here).
                _want_attn_bias_viz = bool(
                    s2_entropy_data and s2_entropy_data.get("vis_bias_debug"))
                _input_imgs = _input_masks = None
                if cfg.output.save_renders and _want_attn_bias_viz:
                    from genia.core.utils.interpolation import compute_pose_axes
                    from genia.core.utils.visualization import draw_pose_axes_on_image
                    _axes_data = compute_pose_axes(
                        {obj_idx: {fid: di for fid, di in _cond_tokens_list}},
                        [fid for fid, _ in _cond_tokens_list],
                    )
                    _input_imgs = []
                    for _vi, (fid, _) in enumerate(_cond_tokens_list):
                        img_t = torch.from_numpy(sequence[fid].image).float() / 255.0
                        img_np = draw_pose_axes_on_image(
                            img_t, _axes_data, _vi, sequence[fid].K_matrix,
                        )
                        _input_imgs.append((img_np * 255).astype(np.uint8))
                    _input_masks = [sequence[fid].masks[obj_idx]
                                    for fid, _ in _cond_tokens_list]

                # Stage 2 ODE trajectory plot
                if cfg.output.save_renders and s2_ode_history:
                    from genia.core.visualization import plot_slat_ode_trajectory
                    canonical_dir = os.path.join(block_output_dir, "canonical")
                    os.makedirs(canonical_dir, exist_ok=True)
                    plot_slat_ode_trajectory(
                        s2_ode_history,
                        output_path=os.path.join(
                            canonical_dir,
                            f"{cfg.dataset.scene_name}_obj{obj_idx}_slat_ode.png",
                        ),
                        title=f"{cfg.dataset.scene_name} obj {obj_idx} — Stage 2 SLAT denoising",
                    )

                # Rendering guidance loss trajectory + per-pixel depth error
                # + numeric dump (when active).  Depth-error grid is only
                # produced when the depth term contributes (depth_weight>0 &
                # GT depth present).
                _ai_rg_history = (
                    s2_entropy_data.get("rendering_guidance_loss_history")
                    if s2_entropy_data is not None else None
                )
                if _ai_rg_history and get_block_output_flag(
                        cfg, block_dir_name, "save_renders"):
                    from genia.core.visualization import (
                        plot_appearance_rendering_guidance_loss,
                        plot_rendering_guidance_depth_error,
                    )
                    canonical_dir = os.path.join(block_output_dir, "canonical")
                    os.makedirs(canonical_dir, exist_ok=True)
                    _ai_scene = cfg.dataset.scene_name
                    plot_appearance_rendering_guidance_loss(
                        _ai_rg_history,
                        os.path.join(
                            canonical_dir,
                            f"{_ai_scene}_obj{obj_idx}_rendering_guidance.png",
                        ),
                    )
                    _ai_fids = [fid for fid, _ in _s2_rg_frames]
                    plot_rendering_guidance_depth_error(
                        _ai_rg_history,
                        os.path.join(
                            canonical_dir,
                            f"{_ai_scene}_obj{obj_idx}"
                            f"_rendering_guidance_error_gaussian.png",
                        ),
                        frame_indices=_ai_fids,
                        metrics=[
                            # RGB image blocks (cmap=None) — rendered + GT
                            # shown beside the error maps for context.
                            ("per_frame_rendered_gaussian", "Rendered", None),
                            ("per_frame_gt_rgb", "GT", None),
                            ("per_frame_depth_error_gaussian",
                             "Abs depth error", "magma"),
                            ("per_frame_rgb_error_gaussian",
                             "Abs RGB error", "viridis"),
                        ],
                        title="Appearance rendering guidance — "
                              "rendered / GT / depth & RGB error",
                    )
                    # JSON dump drops the per-pixel arrays (they're already
                    # rendered into the PNGs above and would bloat the file).
                    import json as _json
                    _ai_rg_serializable = [
                        {k: v for k, v in entry.items()
                         if not k.startswith("per_frame_")}
                        for entry in _ai_rg_history
                    ]
                    with open(os.path.join(
                        canonical_dir,
                        f"{_ai_scene}_obj{obj_idx}_rendering_guidance.json",
                    ), "w") as _f:
                        _json.dump(_ai_rg_serializable, _f, indent=2)

                # Per-ODE-step appearance render.  Snapshots were captured
                # inside stage2_mv when capture_slat_snapshots=True.
                _slat_snaps = (
                    s2_entropy_data.get("slat_ode_snapshots")
                    if s2_entropy_data is not None else None
                )
                if _slat_snaps:
                    from genia.core.visualization import render_appearance_ode_steps
                    canonical_dir = os.path.join(block_output_dir, "canonical")
                    os.makedirs(canonical_dir, exist_ok=True)
                    # State-driven: returns {} for plain canonical,
                    # the warp bundle when a per-frame GT_SHAPES_INVERSION
                    # mode populated the deformation field.
                    _ode_mesh_kwargs = _build_stage2_mesh_warp_kwargs(
                        state, obj_idx, tokens_list, device, cfg.deformation_warp
                    )
                    render_appearance_ode_steps(
                        slat_snapshots=_slat_snaps,
                        pipeline=pipeline_obj,
                        canonical_coords=canonical_coords,
                        tokens_list=tokens_list,
                        sequence=sequence,
                        output_dir=canonical_dir,
                        scene_name=cfg.dataset.scene_name,
                        obj_idx=obj_idx,
                        **_ode_mesh_kwargs,
                    )

                # Attention bias debug plots
                if cfg.output.save_renders and _want_attn_bias_viz:
                    from genia.core.visualization import plot_attn_bias_debug
                    canonical_dir = os.path.join(block_output_dir, "canonical")
                    os.makedirs(canonical_dir, exist_ok=True)
                    plot_attn_bias_debug(
                        s2_entropy_data["vis_bias_debug"],
                        input_images=_input_imgs,
                        input_masks=_input_masks,
                        obj_idx=obj_idx,
                        output_dir=canonical_dir,
                        scene_name=cfg.dataset.scene_name,
                    )

            # Update canonical representation with MV-averaged SLAT.
            # Per-frame SLATs are preserved — only the canonical is updated.
            state.set_canonical_slat(obj_idx, mv_slat)
            print(f"    Updated canonical SLAT + Gaussians "
                  f"({state.canonical_gaussians[obj_idx].get_xyz.shape[0]} points)")

        # Override per-frame Stage 1 shape latent with canonical shape.
        # After appearance init, per-frame SLATs use canonical voxel coords,
        # so the raw shape modality must match the canonical object's shape.
        canon_fid = state.canon_frame_per_object.get(obj_idx)
        if canon_fid is not None and obj_idx in state.perframe_raw_modalities:
            canon_raw = state.perframe_raw_modalities[obj_idx].get(canon_fid, {})
            canon_shape = canon_raw.get("raw_ss_modalities", {}).get("shape")
            if canon_shape is not None:
                n_updated = 0
                for fid, raw in state.perframe_raw_modalities[obj_idx].items():
                    if "raw_ss_modalities" in raw and "shape" in raw["raw_ss_modalities"]:
                        raw["raw_ss_modalities"]["shape"] = canon_shape.clone()
                        n_updated += 1
                print(f"    Updated Stage 1 shape latent to canonical for {n_updated} frames")

    # Per-frame Gaussians already re-decoded by set_all_perframe_slats above.

    # Pose tokens are required for the posed renders + pose-delta metrics in
    # evaluate_block / the keyframes videos.  When this block runs BEFORE pose
    # init (APPEARANCE_INIT, which only produces per-frame Gaussians), no pose
    # tokens exist yet, so the pose-dependent eval/viz is both meaningless and
    # crashes (KeyError on 'rotation').  Skip it in that case; the Gaussians
    # are already in state.
    _has_poses = any(
        "rotation" in di
        for toks in state.tokens_by_object.values()
        for _fid, di in toks
    )
    if not _has_poses:
        print(f"  {block_label}: no pose tokens yet — skipping pose-dependent "
              f"eval/viz (per-frame Gaussians populated in state).")
        return

    with _TIMER.exclude():
        evaluate_block(
            cfg, state, sequence, evaluator, device, block_output_dir,
            suffix=f"_after_{block_dir_name}", per_frame=_eval_per_frame,
            before_renders=before_renders, before_poses=before_poses,
        )

        save_decoded_visualizations(
            state, cfg, pipeline_obj,
            output_dir=block_output_dir,
            tag="appearance_init", frame_indices=frame_indices,
            perframe_only=(strategy == "perframe"),
            canonical_only=(strategy in ("canonical", "canonical_unified")),
            voxel_color_mode="slat_pca",
        )
        save_keyframes_video(
            state, sequence, cfg, suffix="appearance_init",
            output_dir=block_output_dir,
            voxel_color_mode="slat_pca",
        )
        save_keyframes_video(
            state, sequence, cfg, suffix="appearance_init",
            output_dir=block_output_dir,
            render_space="world",
            voxel_color_mode="slat_pca",
        )


# =====================================================================
# FINETUNE: Token Fine-tuning
# =====================================================================


def run_finetuning(cfg, state, sequence, pipeline_obj, evaluator, device):
    """FINETUNE: canonical (or per-frame) token + LoRA-decoder fine-tuning."""
    block_header("FINETUNE: Token Fine-tuning")

    block_output_dir = os.path.join(cfg.output.output_dir, block_output_subdir(cfg, "finetune"))

    if not cfg.finetuning.enabled:
        print("  Finetuning SKIPPED (finetuning.enabled=false)")
        return
    os.makedirs(block_output_dir, exist_ok=True)

    if cfg.finetuning.strategy not in ("canonical", "perframe"):
        raise NotImplementedError(
            f"finetuning.strategy={cfg.finetuning.strategy!r} is not implemented; "
            "only 'canonical' and 'perframe' are supported."
        )

    # ``perframe`` optimizes each frame's OWN SLAT against that frame's GT —
    # the strategy for runs with no canonical object (dynamic sequences
    # reconstructed from SAM3D's per-frame shapes).
    per_frame = cfg.finetuning.strategy == "perframe"

    from genia.core.utils import save_and_plot_loss_history
    from genia.core.finetuning import finetune_perframe_tokens

    # Capture renders BEFORE finetuning
    with _TIMER.exclude():
        before_renders, before_poses = capture_before(
            state, sequence, cfg, per_frame=per_frame,
        )
        if before_renders:
            print("  Captured renders before finetuning")

    decoder = pipeline_obj.models["slat_decoder_gs"]
    decoder.eval()
    for param in decoder.parameters():
        param.requires_grad_(False)

    if per_frame:
        # Per-frame SLATs already carry each timestamp's own geometry, so the
        # deformation warp (a canonical-only concern) does not apply.
        if not any(di.get("decoder_input_slat") is not None
                   for toks in state.tokens_by_object.values() for _, di in toks):
            print("  Finetuning SKIPPED (strategy='perframe' but no per-frame "
                  "SLATs to optimize)")
            return
        ft_tokens, ft_gaussians = finetune_perframe_tokens(
            tokens_by_object=state.tokens_by_object,
            decoder=decoder,
            sequence=sequence,
            losses=cfg.finetuning.losses,
            pipeline=cfg.pipeline,
            ft_config=cfg.finetuning,
            device=device,
            output_dir=block_output_dir,
            scene_name=cfg.dataset.scene_name,
            # Resolved, not raw: FINETUNE's per-iteration debug_pixelwise/
            # PNGs must go away under output.suppress_intermediate_renders.
            save_renders=get_block_output_flag(cfg, "finetune", "save_renders"),
            save_metrics=get_block_output_flag(cfg, "finetune", "save_metrics"),
        )
        state.apply_perframe_finetuning_results(
            ft_tokens, perframe_gaussians=ft_gaussians,
        )
    else:
        _run_canonical_finetuning(
            cfg, state, sequence, pipeline_obj, device, decoder,
            block_output_dir,
        )

    with _TIMER.exclude():
        block_header("EVALUATION: After token fine-tuning")

        evaluate_block(
            cfg, state, sequence, evaluator, device, block_output_dir,
            suffix="_finetuned", per_frame=per_frame,
            before_renders=before_renders, before_poses=before_poses,
        )

        save_and_plot_loss_history(
            state.tokens_by_object, suffix="finetune",
            output_dir=block_output_dir, scene_name=cfg.dataset.scene_name,
            save_json=cfg.output.save_metrics,
            save_plot=cfg.output.save_renders,
        )

        # Per-frame comparison PNGs were already written by ``evaluate_block``
        # above (finetune/renders/), and ``run_final`` redoes the canonical
        # turntable / PLY / mesh exports under ``final/``.  Skip
        # ``save_decoded_visualizations`` here so FINETUNE doesn't write a
        # near-duplicate ``finetune/decoded_renders/`` subdirectory.
        save_keyframes_video(
            state, sequence, cfg, suffix="finetune",
            output_dir=block_output_dir,
            voxel_color_mode="slat_pca",
        )


def _run_canonical_finetuning(cfg, state, sequence, pipeline_obj, device,
                              decoder, block_output_dir):
    """Canonical FINETUNE body — optimize one shared SLAT per object."""
    from genia.core.finetuning import finetune_canonical_tokens

    # actionmesh deformation: when GT_SHAPES_INVERSION (a per-frame mode)
    # populated state.canonical_mesh_verts, FINETUNE warps Gaussians via Φ + R
    # BEFORE applying Stage-1 pose.  State tensors live on CPU; moved once to
    # the FINETUNE device so the warp's pytorch3d.ops.knn_points (CUDA-only)
    # doesn't have to re-transfer per frame per iteration.  All-None ⇒
    # rigid-only path (bit-identical regression on rest-pose / canonical).
    ft_warp_kwargs = _build_perobj_mesh_warp_kwargs(
        state, cfg, device, label="FINETUNE deformation warp"
    )

    ft_slats, ft_tokens, ft_gaussians, ft_lora = finetune_canonical_tokens(
        canonical_slats=state.canonical_slats,
        tokens_by_object=state.tokens_by_object,
        decoder=decoder,
        sequence=sequence,
        losses=cfg.finetuning.losses,
        pipeline=cfg.pipeline,
        ft_config=cfg.finetuning,
        device=device,
        output_dir=block_output_dir,
        scene_name=cfg.dataset.scene_name,
        save_renders=get_block_output_flag(cfg, "finetune", "save_renders"),
        save_metrics=get_block_output_flag(cfg, "finetune", "save_metrics"),
        # Needed by the MV shared-world parameterization to resolve each object's
        # reference frame the same way the rebase does.
        canon_frame_per_object=state.canon_frame_per_object,
        **ft_warp_kwargs,
    )
    state.apply_finetuning_results(
        ft_slats, ft_tokens,
        canonical_gaussians=ft_gaussians,
        lora_state_dicts=ft_lora,
    )
