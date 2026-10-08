# Copyright (c) Meta Platforms, Inc. and affiliates.

"""GLOBAL_POSE_REFINE: the one gradient-based (or ICP) pose-refinement block.

Runs as blocks ``global_pose_refine_1``/``_2``. ``losses.refine_geometry`` picks what each
frame is refined against.
"""

import os

import numpy as np

from genia.core.final import _build_perobj_mesh_warp_kwargs
from genia.core.utils.config import (
    VALID_REFINE_GEOMETRY,
    block_output_subdir,
    get_block_output_flag,
    print_loss_config,
    resolve_chamfer_gt_trim,
    resolve_correction_granularity,
    resolve_correction_scale_control,
    resolve_refine_geometry,
)
from genia.core.utils.console import block_header
from genia.core.utils.evaluation import capture_before, evaluate_block, save_keyframes_video
from genia.core.utils.timing import PipelineTimer, get_timer

_TIMER: PipelineTimer = get_timer()

_ICP_INERT_WEIGHTS = ("rgb_weight", "silhouette_weight", "depth_weight",
                      "normals_weight", "perceptual_weight", "chamfer_weight")


def refine_strategy(blk_cfg) -> str:
    """The block's pose-refinement strategy, read defensively.

    ``getattr`` with a default is not optional here: the dataclasses are never registered as a schema
    and ``main.py`` never merges the structured schema, so a block composed from
    ``none.yaml`` -- which carries only the five enable/output keys -- has no
    ``strategy`` attribute at all.
    """
    return getattr(blk_cfg, "strategy", "default") or "default"


def _warn_inert_pixel_weights(blk_cfg, label: str) -> None:
    """Say so when a config sets a knob this strategy cannot read.

    ICP replaces the pixel-aligned terms rather than supplementing them -- its
    ``alpha > 0.5`` mask is hard, so no gradient reaches the silhouette, and mixing the
    two reintroduces exactly the misregistration ICP exists to avoid.  A weight left
    non-zero here is therefore not merely ignored, it signals a misunderstanding, so it
    is reported rather than silently dropped.
    """
    live = [w for w in _ICP_INERT_WEIGHTS if float(getattr(blk_cfg, w, 0.0) or 0.0) > 0]
    if live:
        print(f"    [warn] {label}: strategy=icp ignores {', '.join(live)} "
              f"(the solver has one loss of its own); set them to 0 to silence this")


def icp_history_to_token_fields(history) -> dict:
    """One ICP history -> the token-dict fields `save_and_plot_loss_history` harvests.

    Both keys or neither: that writer reads ``refinement_best_iteration``
    UNCONDITIONALLY once a frame carries a ``refinement_loss_history``, so writing one
    without the other is a KeyError raised at the END of the block, after the solve is
    already paid for.

    Note: ``best_iteration`` means something weaker here than on the photometric path.
    Those loops RESTORE the best iterate; ICP keeps its FINAL one, so this marks the
    lowest residual REACHED, not the delta that was applied.  A best that is not the last
    index therefore says the solver moved past its own optimum -- informative, and
    invisible if this reported ``len(history) - 1`` instead.
    """
    residuals = [h["residual"] for h in history]
    # The closed-form solver's own diagnostics, carried through when present.  They are
    # what a run is READ with -- `gate_kept` says whether the correspondences survived,
    # `nu` where the schedule had reached, `perframe_residual_norm` how the shared and
    # per-timestamp transforms split the error.
    _extra = ("nu", "n_rows", "gate_kept", "perframe_residual_norm")
    return {
        # Rename onto the plotter's vocabulary; `total` is the series it always draws.
        "refinement_loss_history": [
            {"total": h["residual"], "icp_chamfer": h["residual"],
             "n_pairs": h["n_pairs"],
             **{k: h[k] for k in _extra if k in h}} for h in history
        ],
        # NaN residuals (a frame that rendered nothing) sort last rather than winning.
        "refinement_best_iteration": min(
            range(len(residuals)),
            key=lambda i: (np.isnan(residuals[i]), residuals[i])),
    }


def run_icp_refine_block(blk_cfg, state, sequence, device, gaussians,
                         *, per_frame_deltas: bool, shared_correction: bool = False,
                         scale_control: str = "perframe",
                         label: str, debug_dir=None) -> None:
    """Drive :func:`pose_refit.apply_icp_refine` over every object of one block.

    Mirrors what the photometric path leaves behind, so a block's OUTPUTS do not depend
    on its strategy: the per-iteration history is stashed under
    ``refinement_loss_history`` on each frame's token dict, which is where
    ``save_and_plot_loss_history`` harvests it from.
    """
    from genia.core.utils.pose_refit import apply_icp_refine

    _warn_inert_pixel_weights(blk_cfg, label)
    steps = int(getattr(blk_cfg, "num_iterations", 100))
    # Names the third mode: under `both` the deltas ARE per-frame, so a binary label
    # would print "per-frame" and a run log could not be told from a `per_frame` one.
    granularity = ("per-frame + one shared correction (both)" if shared_correction
                   else "per-frame" if per_frame_deltas
                   else "root (one per object)")
    force_splat = bool(getattr(blk_cfg, "icp_force_voxel_splat", True))
    _chamfer_trim = resolve_chamfer_gt_trim(blk_cfg, label)
    print(f"  {label}: ICP strategy, {granularity}, closed-form p2pl, {steps} steps max"
          + (", voxel-splat source (preferred)" if force_splat else "")
          + (f", GT cloud trimmed at {_chamfer_trim:g}x p95"
             if _chamfer_trim else ", GT cloud untrimmed"))

    for obj_idx in sorted(state.tokens_by_object or {}):
        n, history = apply_icp_refine(
            state, sequence, obj_idx, device, gaussians,
            per_frame_deltas=per_frame_deltas,
            shared_correction=shared_correction, scale_control=scale_control,
            steps=steps,
            # `null` / 0 = uncapped: every masked pixel enters the ICP target.
            max_gt_points=(lambda v: int(v) if v else None)(
                getattr(blk_cfg, "icp_max_gt_points", None)),
            force_voxel_splat=force_splat,
            fit_scale=bool(getattr(blk_cfg, "icp_fit_scale", True)),
            chamfer_reverse_weight=float(
                getattr(blk_cfg, "icp_chamfer_reverse_weight", 0.0)),
            # NOT an `icp_*` knob: it governs the shared Chamfer TARGET, so the ICP
            # block and the photometric `chamfer_weight` term read the same field.
            chamfer_gt_trim_factor=_chamfer_trim,
            pred_trim_factor=float(getattr(blk_cfg, "icp_pred_trim_factor", 0.0)),
            target_seed=getattr(blk_cfg, "icp_target_seed", 0),
            p2pl_trim=float(getattr(blk_cfg, "icp_p2pl_trim", 0.8)),
            p2pl_gnc_from=float(getattr(blk_cfg, "icp_p2pl_gnc_from", 8.0)),
            p2pl_gnc_to=float(getattr(blk_cfg, "icp_p2pl_gnc_to", 0.8)),
            p2pl_normal_threshold=float(
                getattr(blk_cfg, "icp_p2pl_normal_threshold", 0.5)),
            perframe_prior_weight=float(
                getattr(blk_cfg, "icp_perframe_prior_weight", 1.0)),
            max_scale_step=float(getattr(blk_cfg, "icp_max_scale_step", 0.1)),
            p2pl_normals=str(getattr(blk_cfg, "icp_p2pl_normals", "pca")),
            keep_best_iterate=bool(getattr(blk_cfg, "icp_keep_best_iterate", False)),
            debug_dir=debug_dir,
        )
        last = history[-1] if history else None
        # Report the DELTA, not just the frame count: "16 frame(s)" cannot distinguish
        # "ran and corrected 3 degrees" from "ran and did nothing".
        print(f"    obj {obj_idx}: {n} frame(s)"
              + (f" — ΔR {last['angle_deg']:.2f}°, Δs {last['scale']:.4f}, "
                 f"|Δt| {last['trans_norm']:.4f}" if last else "")
              + ("" if n else " — no frame had both a decoded pose, a render source "
                              "(Gaussian or mesh) and usable depth, skipped"))
        if not (n and history):
            continue
        for _fk, di in state.tokens_by_object.get(obj_idx) or []:
            di.update(icp_history_to_token_fields(history))


def _emit_pose_overlay_viz(cfg, state, sequence, obj_idx, device, out_dir, tag,
                           block: str = "pose_init"):
    """Write one posed-shape overlay diagnostic (``_{block}_{tag}.png``).

    Green = the GT canonical shape at the object's CURRENT pose, red = the GT mask, one
    panel per frame, with decoded z and scale plotted underneath.  ``tag`` selects the
    filename and the log label.

    Not pose_init's alone: it reads whatever pose is in ``state``, so a REFINE block
    brackets itself with a start/end pair and the picture shows what it changed.  That
    matters because a typical refine failure -- the object growing in image space while
    translation stays put -- is invisible in a scalar loss that is going down.

    Non-fatal by design: a render hiccup must not kill a run that may have cost hours,
    and it can't be pre-tested on CPU (nvdiffrast needs CUDA).
    """
    from genia.core.utils.depth_grounding import save_pose_overlay_viz

    path = os.path.join(
        out_dir, f"{cfg.dataset.scene_name}_obj{obj_idx}_{block}_{tag}.png")
    with _TIMER.exclude():
        try:
            ok = save_pose_overlay_viz(
                state, sequence, obj_idx, device, path, block_label=block)
            print(f"    {block} {tag} viz -> {path}" if ok else
                  f"    {block} {tag} viz skipped (no GT mesh / pose)")
        except Exception as _e:
            print(f"    [warn] {block} {tag} viz failed: {_e}")


def _emit_refine_overlay_viz(cfg, state, sequence, device, out_dir, block_tag, tag):
    """The start/end overlay pair for a REFINE block, over every object it holds.

    Gated on the block's RESOLVED render flag, so a suppressed run writes none of them.
    """
    if not get_block_output_flag(cfg, block_tag, "save_renders"):
        return
    for obj_idx in sorted(state.tokens_by_object.keys()):
        _emit_pose_overlay_viz(cfg, state, sequence, obj_idx, device, out_dir, tag,
                               block=block_tag)


def icp_renders_voxel_splat(gr_cfg) -> bool:
    """Does this block's ICP render the canonical SHAPE splat instead of a Gaussian?

    ``strategy: icp`` with ``icp_force_voxel_splat`` skips ``_gaussian_for`` entirely in
    ``pose_refit._block_icp_frames`` and renders ``canonical_points_for_frame`` -- the
    canonical occupancy grid -- so the ``object_gaussians`` argument
    ``refine_geometry_render_plan`` hands it is never read.

    Exists because the canonical-scope refusal below asks "would this collapse to the
    EARLIEST frame's Gaussian?", and under this combination there is no Gaussian in play to
    collapse.  A real function rather than an inline condition, like
    :func:`refine_geometry_render_plan`, so it can be tested directly.

    Deliberately NOT true for ``icp_force_voxel_splat: false``: that path DOES resolve a
    Gaussian per frame and the collapse is real there.

    The ``icp_force_voxel_splat`` fallback is ``True``, matching the dataclass default and
    every YAML -- an absent key means the splat IS rendered, so defaulting to False
    here would refuse a run whose ICP never reads a Gaussian.
    ``strategy`` is what carries the non-ICP case, and it has no such default.
    """
    return (str(getattr(gr_cfg, "strategy", "")) == "icp"
            and bool(getattr(gr_cfg, "icp_force_voxel_splat", True)))


def refine_geometry_render_plan(state, scope: str):
    """``(object_gaussians, per_frame_canonical, refine_scale)`` for a resolved ``refine_geometry``.

    THE definition of what the knob means, extracted so it can be called and tested.
    These three arguments are the entire difference between the canonical and per-frame
    refinement scopes (plus, on the ICP branch, ``per_frame_deltas``, which
    ``correction_granularity: per_frame`` expresses).

    ``own_frame`` renders each frame against ITS OWN reconstruction.  Deliberately NOT
    ``canonical_gaussians_with_fallback``, which collapses a per-frame state to the
    EARLIEST frame's Gaussian -- i.e. it would refine frame 37's pose against frame 0's
    geometry.

    Mutates ``state`` only to decode per-frame Gaussians when they are needed and absent
    (``ensure_perframe_gaussians`` is itself a no-op when they exist), which the canonical
    branch also does for a per-frame-only state.
    """
    if scope not in VALID_REFINE_GEOMETRY:
        raise ValueError(f"refine_geometry must be one of {VALID_REFINE_GEOMETRY}, got {scope!r}")
    per_frame = scope == "own_frame"
    if per_frame or not state.has_canonical:
        state.ensure_perframe_gaussians()
    if per_frame:
        return _perframe_sources(state), True, "perframe"
    return state.canonical_gaussians_with_fallback, False, "global"


def _perframe_sources(state):
    """Per-frame Gaussians, or the canonical one broadcast when there are none to decode.

    The per-frame loop resolves its render source from the per-frame Gaussian, else SKIPS
    THE OBJECT SILENTLY.  A state holding canonical Gaussians but no per-frame SLAT for
    ``ensure_perframe_gaussians`` to decode would therefore pass its poses through
    unrefined.  This adds the missing Gaussian source.

    Legitimate only for SINGLE-FRAME objects: with one frame per object the canonical
    reconstruction IS that frame's own reconstruction, so the broadcast is exact rather
    than an approximation.  With two or more frames it would quietly re-enact
    ``canonical`` scope under the ``own_frame`` name -- the collapse this scope exists to
    avoid -- so that case RAISES instead.
    """
    if state.perframe_gaussians:
        return state.perframe_gaussians
    if not state.canonical_gaussians:
        return state.perframe_gaussians      # nothing to broadcast
    multi = {o: len(t) for o, t in state.tokens_by_object.items() if len(t) > 1}
    if multi:
        raise ValueError(
            "refine_geometry='own_frame': no per-frame Gaussians could be decoded (the "
            "state holds no per-frame SLAT), and the canonical Gaussian cannot stand in "
            f"because these objects have more than one frame: {multi}. Broadcasting it would "
            "refine every frame against ONE shared reconstruction, which is "
            "refine_geometry='canonical' under another name. Use refine_geometry='canonical' "
            "if that is what you want.")
    return {o: {fk: state.canonical_gaussians[o] for fk, _ in t}
            for o, t in state.tokens_by_object.items() if o in state.canonical_gaussians}


def degrade_canonical_to_own_frame(cfg, gr_cfg, state, sequence, block_label: str) -> str:
    """``"own_frame"`` when the canonical scope has no canonical to refine against.

    Called from the collapse guard in :func:`run_global_refine`: `refine_geometry:
    canonical` on a state with no canonical object is served by the OTHER geometry --
    every frame already carries its own reconstruction, and each is refined against that.
    The degrade is PRINTED; what the guard exists to prevent is the SILENT collapse onto
    the earliest frame's Gaussian, not the fallback.

    Raises when ``own_frame`` cannot serve either:

    * ``correction_granularity != per_frame`` under ``strategy: default``, which
      dispatches to the per-frame loop -- the one solver of six that never reads that
      knob, so the fallback would perform one delta per frame while the config asked for
      one for the sequence.  This is the silent downgrade :func:`run_global_refine`
      already refuses on a DECLARED ``own_frame``.  Exempt when every object has a single
      frame (the two granularities are then the same quantity), and never reached under
      ``strategy: icp``, which implements both at either geometry.
    * Nothing to render: no per-frame Gaussians, no canonical to broadcast
      (:func:`_perframe_sources` refuses a multi-frame broadcast for the same reason this
      guard fires) -- the loop would skip every object.

    Availability is checked by BUILDING the plan the block is about to use, so the check
    cannot disagree with what runs.  :func:`run_global_refine` builds it again on the next
    line; both halves are idempotent (``ensure_perframe_gaussians`` is a no-op once they
    are decoded).
    """
    why = (f"{block_label}: refine_geometry='canonical' has no canonical object to refine "
           "against (has_canonical=False), and the fallback to refine_geometry='own_frame' ")
    off = ("Turn this block off "
           f"(`global_pose_refine@{block_label.lower()}=none`).")
    multi = {o: len(t) for o, t in state.tokens_by_object.items() if len(t) > 1}
    if multi and refine_strategy(gr_cfg) != "icp":
        cg = resolve_correction_granularity(
            gr_cfg, block_label, is_mv=sequence.is_mv,
            mv_shared_world_pose=cfg.pipeline.mv_shared_world_pose)
        if cg != "per_frame":
            raise ValueError(
                f"{why}cannot serve either: correction_granularity={cg!r} is not implemented "
                "under 'own_frame' with strategy='default' (the per-frame loop has no "
                "shared-correction path, so it would be silently downgraded to one delta per "
                f"frame across {multi}). Use correction_granularity='per_frame', or "
                f"strategy='icp', which fits a shared correction at either geometry. Or: {off}")
    try:
        sources, _, _ = refine_geometry_render_plan(state, "own_frame")
    except ValueError as exc:
        raise ValueError(f"{why}cannot serve either: {exc}") from exc
    if not sources:
        raise ValueError(
            f"{why}has nothing to render either -- this run decoded no per-frame Gaussians, "
            f"so every object would be skipped. {off}")
    print(f"  [{block_label}] no canonical object (has_canonical=False) -> refining each "
          "frame against its OWN reconstruction (refine_geometry='own_frame'); 'canonical' "
          "would collapse onto the EARLIEST frame's Gaussian")
    return "own_frame"


def run_global_refine(cfg, state, sequence, pipeline_obj, evaluator, device,
                      config_override=None, block_label="GLOBAL_POSE_REFINE_2"):
    """Pose refinement for a whole sequence — the repo's ONE gradient-based pose block.

    Drives `global_pose_refine_{1,2}`, including per-frame refinement:
    `losses.refine_geometry` picks what each frame is refined against and whether scale is
    shared (see :func:`refine_geometry_render_plan`), so "global scale" is the `canonical`
    default rather than a property of this function.
    """
    gr_cfg = config_override if config_override is not None else cfg.global_pose_refine_2
    if not gr_cfg.enabled:
        print(f"\n  {block_label}: skipped (enabled=false)")
        return

    from genia.core.utils import (
        refine_poses_for_sequence,
        save_and_plot_loss_history,
    )

    block_tag = block_label.lower()

    block_header(f"{block_label}: Pose Refinement")

    block_output_dir = os.path.join(cfg.output.output_dir, block_output_subdir(cfg, block_tag))
    os.makedirs(block_output_dir, exist_ok=True)

    # ── refine_geometry: WHAT each frame is refined against, and whether scale is shared ──
    # The choice sets three arguments -- the Gaussian dict, `per_frame_canonical` and
    # `refine_scale`.  `canonical` is what an absent key resolves to.
    # Resolved HERE, above the before-capture, because the render/eval passes are
    # per-frame under `own_frame` too -- capturing them canonically and scoring them
    # per-frame would compare two different pictures.
    _scope = resolve_refine_geometry(gr_cfg, block_label)
    _per_frame = _scope == "own_frame"
    if _per_frame and cfg.pipeline.mv_shared_world_pose:
        # Not downgraded silently: the two are contradictory statements about how many
        # poses EXIST.  `refine_poses_for_sequence` short-circuits to the shared-world
        # solver BEFORE the refine_scale dispatch, so the scope would be accepted and
        # then ignored -- a run that reports a per-frame refine and performs a
        # one-Sim(3)-for-the-sequence one.
        raise ValueError(
            f"{block_label}: refine_geometry='own_frame' refines each frame against its own "
            "reconstruction with a per-frame pose, but "
            "pipeline.mv_shared_world_pose=true DERIVES every frame's pose from one "
            "reference — there is no per-frame pose left to refine. Use "
            "refine_geometry='canonical', or turn pipeline.mv_shared_world_pose off for this "
            "run.")
    if _per_frame and refine_strategy(gr_cfg) != "icp":
        # The OTHER cell `refine_geometry: own_frame` cannot serve.  Under the photometric
        # strategy this scope dispatches to the per-frame loop (`refine_pose_for_frame`),
        # which is the ONE solver of the six that never reads `correction_granularity` --
        # so a `shared`/`both` here would ask for one Sim(3) and get one delta per frame,
        # with nothing in the log saying so.  ICP is
        # exempt: it reads the knob at either geometry (`per_frame_deltas` /
        # `shared_correction`), which is why the test asserts it does NOT raise there.
        _cg = resolve_correction_granularity(
            gr_cfg, block_label, is_mv=sequence.is_mv,
            mv_shared_world_pose=cfg.pipeline.mv_shared_world_pose)
        if _cg != "per_frame":
            raise ValueError(
                f"{block_label}: correction_granularity={_cg!r} is not implemented under "
                "refine_geometry='own_frame' with strategy='default'. That scope renders each "
                "frame against its own reconstruction through the per-frame loop, which has "
                "no shared-correction path, so the setting would be silently downgraded to "
                "one delta per frame. Use correction_granularity='per_frame', or "
                "strategy='icp', which fits a shared correction at either geometry.")
    if (not _per_frame and not state.has_canonical
            and not cfg.pipeline.mv_shared_world_pose
            and not icp_renders_voxel_splat(gr_cfg)):
        # The mirror of the `own_frame` refusals above, for the CANONICAL scope: with no
        # canonical object and no shared-world pose, `canonical_gaussians_with_fallback`
        # collapses to the EARLIEST frame's Gaussian, so frame 37's pose would be refined
        # against frame 0's geometry -- silently, and worst on exactly the deforming
        # sequences where a per-frame reconstruction is used in the first place.
        #
        # DEGRADED to `own_frame`, not refused: what must not happen is the SILENT
        # collapse.  A photometric gpr1 (`refine_geometry: canonical`) run straight after
        # a per-frame APPEARANCE_INIT sees per-frame Gaussians and no canonical yet, and a
        # mono-DYNAMIC run with no shape injection builds no canonical at all.  The fallback prints, and raises when `own_frame` cannot
        # serve either.
        #
        # Not reached at all by: `strategy: icp` rendering the voxel splat
        # (`icp_renders_voxel_splat`) -- that path never reads a Gaussian, it renders the
        # canonical shape, so a run with `appearance_init=none` arrives here with
        # has_canonical=False and nothing wrong.
        _scope = degrade_canonical_to_own_frame(cfg, gr_cfg, state, sequence, block_label)
        _per_frame = _scope == "own_frame"
    gaussians, per_frame_canonical, _refine_scale = refine_geometry_render_plan(state, _scope)

    with _TIMER.exclude():
        # Capture renders BEFORE refinement
        before_renders, before_poses = capture_before(
            state, sequence, cfg, per_frame=_per_frame,
        )
        if before_renders:
            print(f"  Captured renders before {block_tag}")
    # The posed-shape overlay, BEFORE the refine.  Its end twin is written below, and the
    # pair is the only place a refine's effect on APPARENT SIZE is legible: an ICP that
    # grows the object while leaving translation put drives its own loss DOWN, so neither
    # the loss history nor the per-object crop metrics show it -- it surfaces as a
    # full-frame PSNR drop in a later block, far from the cause.
    _emit_refine_overlay_viz(cfg, state, sequence, device, block_output_dir,
                             block_tag, "start")

    print(f"\n  Pose optimization (refine_geometry={_scope}, "
          f"{'per-frame' if _per_frame else 'global'} scale)")
    print_loss_config(gr_cfg, cfg.pipeline,
                      stage="per-frame" if _per_frame else "global")

    # perframe_raw_modalities is required for the shared-world path; the global path
    # receives None here (no pose-token optimization on that path).
    _pf_raw = (
        getattr(state, "perframe_raw_modalities", {})
        if cfg.pipeline.mv_shared_world_pose
        else None
    )

    # actionmesh deformation: when GT_SHAPES_INVERSION populated
    # ``state.canonical_mesh_*`` (per-canonical-mesh-vertex Φ + R), the
    # global-scale + composite refinement paths warp the canonical Gaussian
    # per (obj, frame) BEFORE applying the optimised Sim(3) pose (all-None ⇒
    # rigid-only).  Mirrors the FINETUNE block runner.
    # Not under `own_frame`: each frame already renders its OWN decoded Gaussian, which
    # carries that frame's geometry, so there is no canonical shape to warp onto it.
    # `refine_poses_for_sequence` refuses the combination anyway (the per-frame path does
    # not consume the field).
    _warp_kwargs = ({} if _per_frame else _build_perobj_mesh_warp_kwargs(
        state, cfg, device, label="Deformation warp"))

    # ONE Sim(3) IS THE POINT, not a limitation to be lifted.  For a dynamic-scene
    # method the object's motion already lives in its deformation field; the only
    # thing a refine may correct is the single systematic mis-seating of the whole
    # reconstruction — the ROOT pose at the anchor frame.  The alternative the
    # global/composite path offers ("its own global scale, per-frame rotation, and
    # per-frame translation") hands every frame a free rigid pose, which lets the
    # optimiser absorb — or manufacture — the very per-frame motion being reconstructed.
    #
    # Note: "a deformation field exists" is NOT "the asset deforms".  A STATIC multi-view run
    # (GSO/CO3D at one timestamp) with GT shape injection populates the field too, but its
    # frame axis collapses to a single index, so the field is a one-entry IDENTITY map.
    # Restricting there would be pointless at best: the frames are CAMERAS of one instant,
    # and aggregating over them is the whole purpose of a multi-view refine.  So the third
    # conjunct is the data shape, read the way the rest of the pipeline reads it.
    _deforms = bool(_warp_kwargs.get("canonical_mesh_verts_per_obj"))
    # "this asset deforms and its motion belongs to the WARP" -- the fact the refusals
    # below key off, whatever the strategy.  Its per-frame motion is the deformation
    # field's, so the only thing a refine may correct is the single systematic
    # mis-seating of the whole reconstruction: ONE Sim(3) shared by the sequence.
    #
    # That correction is fit from EVERY frame on BOTH strategies.  The photometric
    # shared-world solver takes the deformation field and precomputes each frame's warped
    # geometry once, so the warp kwargs are passed through untouched.
    #
    # The ICP solver renders THIS frame's injected voxel grid
    # (`canonical_points_for_frame` -> `gt_perframe_voxel_correspondence`), so it too can
    # render a deforming object at any frame.  The per-frame GRID gives a uniform surface
    # density (deformed mesh VERTICES are not a surface sampling and would render as a
    # speckled wireframe).
    _deforming_dynamic = (bool(cfg.pipeline.mv_shared_world_pose) and _deforms
                          and bool(sequence.is_dynamic))
    _tokens_in = state.tokens_by_object

    if refine_strategy(gr_cfg) == "icp":
        # `correction_granularity: shared` solves ONE Sim(3) delta for the object;
        # `per_frame` solves one delta per frame, and `both` solves both at once (the
        # per-frame deltas are this solver's "natives").  All three values are honoured
        # by every solver -- the only refusal left is the deforming one below, which is
        # about the DATA, not about a solver that cannot express the mode.
        _rd = resolve_correction_granularity(
            gr_cfg, "global_pose_refine", is_mv=sequence.is_mv,
            mv_shared_world_pose=cfg.pipeline.mv_shared_world_pose)
        if _deforming_dynamic and _rd in ("per_frame", "both"):
            # Not downgraded silently: a DEFORMING object's per-frame motion is the
            # deformation's job, and a per-frame delta hands the optimiser the freedom
            # to absorb or invent exactly that motion.  Same
            # refusal the photometric path makes by construction (it fits one pose).
            raise ValueError(
                f"global_pose_refine: correction_granularity={_rd!r} is not available for a "
                "DEFORMING object — it frees a correction PER FRAME ('both' adds a shared "
                "one on top, but keeps the per-frame deltas), and that per-frame motion "
                "belongs to the deformation field, not to a pose optimiser. Use "
                "correction_granularity='shared'.")
        # A DEFORMING asset is fit from EVERY frame, not just the canonical timestamp.
        # The correction is ONE Sim(3) shared by the whole sequence, so every frame has a
        # say in it.  The ICP solver can render every frame because
        # `canonical_points_for_frame` resolves THIS frame's injected voxel grid, falling
        # back to the canonical only where the injection built none.
        #
        # Only on the splat path.  With `icp_force_voxel_splat: false` the solver
        # renders the canonical Gaussians instead, which are UNDEFORMED at every frame, so
        # that combination is refused rather than silently fit against the wrong shape.
        _icp_splat = bool(getattr(gr_cfg, "icp_force_voxel_splat", True))
        if _deforming_dynamic and not _icp_splat:
            raise ValueError(
                "global_pose_refine: strategy=icp with icp_force_voxel_splat=false on a "
                "DEFORMING asset would match every frame against the UNDEFORMED canonical "
                "reconstruction. Set icp_force_voxel_splat=true (the shipped default), "
                "which renders each frame's own deformed geometry.")
        if _deforming_dynamic:
            print("  Deforming asset + shared-world pose: fitting the ONE root "
                  "correction from EVERY frame (each matched against its own deformed "
                  "geometry), not from the canonical timestamp alone")
        run_icp_refine_block(
            gr_cfg, state, sequence, device, gaussians,
            per_frame_deltas=(_rd != "shared"), shared_correction=(_rd == "both"),
            # Inert unless `_rd == "both"`.  Under `per_frame`, freezing the per-timestamp
            # scale would leave no scale control at all.
            scale_control=resolve_correction_scale_control(
                gr_cfg, "global_pose_refine"),
            label=block_label,
            debug_dir=block_output_dir if cfg.output.save_renders else None)
        # apply_icp_refine mutates the token dicts in place, and every frame is in that
        # set, so the state already holds the refined poses — no broadcast needed.
        # The shared-world invariant survives all three modes, for two reasons that are
        # easy to conflate: the correction is composed in the OBJECT frame, so it lands
        # identically on frames that differ in rotation (`compose_root_delta`); and the
        # per-frame axis is per TIMESTAMP, so the views OF a timestamp receive the same
        # delta rather than one each (`pose_refit._timestamp_delta_index`).
        _refined = state.tokens_by_object
    else:
        _refined = refine_poses_for_sequence(
            gaussians, _tokens_in,
            sequence,
            losses=gr_cfg,
            pipeline=cfg.pipeline,
            refine_scale=_refine_scale,
            per_frame_canonical=per_frame_canonical,
            perframe_raw_modalities=_pf_raw,
            output_dir=block_output_dir,
            save_renders=get_block_output_flag(cfg, block_tag, "save_renders"),
            canon_frame_per_object=state.canon_frame_per_object,
            **_warp_kwargs,
        )
    # No broadcast on either branch.  Both fit from every frame and write every
    # frame -- the photometric solver through its own `scope="all"` rebase, ICP in place
    # -- so pushing one frame's pose onto the rest would REPLACE the motion they just
    # preserved rather than fill a gap.
    state.tokens_by_object = _refined

    # AFTER the refine and after the shared-world write-back, so the overlay shows the
    # pose the block actually leaves behind rather than a pre-broadcast intermediate.
    _emit_refine_overlay_viz(cfg, state, sequence, device, block_output_dir,
                             block_tag, "end")

    with _TIMER.exclude():
        save_and_plot_loss_history(
            state.tokens_by_object, suffix=block_tag,
            output_dir=block_output_dir, scene_name=cfg.dataset.scene_name,
            save_json=cfg.output.save_metrics,
            save_plot=cfg.output.save_renders,
        )

        # Evaluate after global refinement
        block_header(f"EVALUATION: After {block_tag}")
        evaluate_block(
            cfg, state, sequence, evaluator, device, block_output_dir,
            suffix=f"_after_{block_tag}", per_frame=_per_frame,
            before_renders=before_renders, before_poses=before_poses,
        )
        save_keyframes_video(
            state, sequence, cfg, suffix=block_tag,
            output_dir=block_output_dir,
        )
