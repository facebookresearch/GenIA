"""
Configuration classes for the GenIA pipeline.

Hydra-based configuration with structured dataclasses.
Uses config groups for polymorphic strategies (pose init, appearance init).

Hierarchy
---------
GeneralConfig                   (the base sections; the block sections live in
│                                ``genia.core.config.GeniaConfig``)
├── dataset: DatasetConfig
├── processing: ProcessingConfig
├── deformation_warp: DeformationWarpConfig
├── global_pose_refine_2: GlobalPoseRefineConfig (extends LossConfig)
│          Global pose refinement losses + block output flags (GLOBAL_POSE_REFINE_2)
├── pipeline: PipelineConfig
│          Cross-phase flags (optimize_pose_tokens, mv_shared_world_pose, ...)
└── output: OutputConfig
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional


# =====================================================================
# Dataset
# =====================================================================

@dataclass
class DatasetConfig:
    """Dataset selection, scene, and frame sampling."""

    name: str = "image"
    """Dataset type: ``image``, ``mvcustom``, ``dyncustom``, ``gso``, ``co3d``,
    ``oursactionbench`` or ``davis_actionmesh``."""

    path: Optional[str] = None
    """Path to dataset root. Defaults to standard paths per dataset type."""

    scene_name: Optional[str] = None
    """Name of the scene to process."""

    frame_index: Optional[int] = None
    """Single frame to process (0-based). If None, uses frame_stride."""

    frame_stride: int = 1
    """Stride for iterating over frames."""

    object_indices: Optional[List[int]] = None
    """Only process these object indices (0-based). If None, processes all.

    CLI examples::

        dataset.object_indices='[1]'        # single object
        dataset.object_indices='[1,2,3]'    # multiple objects
    """

    fps: int = 24
    """Native frame rate of the dataset."""

    downscale_factor: int = 1
    """Integer downscale factor for input images, masks, and depth.

    All spatial data (images, masks, depth, pointmaps) are downscaled by
    stride slicing (nearest-neighbor) and camera intrinsics are adjusted
    accordingly.  A value of 1 means no downscaling.
    """

    num_input_views: Optional[int] = None
    """Number of input views for benchmark datasets (e.g. GSO).

    Controls how many views are exposed to the pipeline. Remaining views
    are reserved for evaluation only. For GSO: views 0..(N-1) are input;
    views 10..24 are test views for novel view synthesis evaluation.
    When None, all available views are used.
    """

    world_origin_is_object: bool = False
    """World origin coincides with the reconstructed object.

    When true, camera-to-origin distance is a valid scale anchor.
    ``_align_depth_scale_to_gt_poses`` uses it as a fallback for the
    single-view (``N=1``) case where no pairwise inter-camera baselines
    exist.  True for GSO (EscherNet convention: cameras orbit around
    the object at world origin) and CO3D (LaRa) (its normalization puts
    the object at world origin); false for the other datasets, where
    world origin is typically the scene root or camera 0.
    """


# =====================================================================
# Processing
# =====================================================================

@dataclass
class ProcessingConfig:
    """Processing behavior flags."""

    seed: int = 42
    """Random seed for inference."""

    add_background_gaussians: bool = True
    """Add background Gaussians from non-masked regions."""

    resume_from_cache: bool = True
    """Resume from ``.pt`` pipeline state checkpoints when available.

    When ``True`` (default), :func:`find_latest_pipeline_cache` scans the
    per-experiment ``cache/`` directory for the most recent
    ``{block_name}.pt`` and the pipeline skips blocks up to and including
    that point.  When ``False``, every block runs from scratch.

    Affects only the ``.pt`` checkpoint resume path — nothing else.
    """

    median_intrinsics: bool = False
    """Use median camera intrinsics across keyframes (MoGe depth only).

    MoGe predicts slightly different intrinsics per frame.  For fixed-camera
    datasets the true intrinsics are constant; medianing reduces noise.
    Ignored when ``depth_source='gt'`` (GT depth has fixed intrinsics).
    Ignored when ``reconstruction_model='map_anything'`` (multi-view inference
    produces consistent intrinsics).
    """

    reconstruction_model: Optional[str] = "moge"
    """Sequence depth/pose reconstruction model: ``'moge'`` (per-frame depth,
    no pose prediction → identity poses), ``'map_anything'`` (multi-view depth **+
    predicted camera poses**; can also consume a GT prior), or ``None`` (no model
    loaded).

    ``None`` is valid only when ``depth_source='gt'`` (you cannot predict depth
    with no model); ``camera_poses_source='pred'`` then yields identity poses.
    Orthogonal to the SAM3D pipeline's own internal MoGe used by the Stage-1
    forward pass; this selects only the *Sequence* depth/pose source.
    """

    map_anything_model_id: str = "facebook/map-anything"
    """HuggingFace model ID for map-anything.

    Also supports ``'facebook/map-anything-apache'`` (Apache 2.0 license).
    Only used when ``reconstruction_model='map_anything'``.
    """

    recon_resolution_set: int = 518
    """Resolution the reconstruction backend is fed at: 518 (default) or 504.

    map-anything's ``fixed_mapping`` rule, applied to both map_anything and moge so
    both backends see the same pixels. The LONG
    side is this value; the short side is snapped to the nearest of ten fixed aspect
    ratios and the image center-cropped to it (``sequence.fixed_mapping_size``).
    Not purely a downscale: an input smaller than this on its long side is scaled UP.
    """

    recon_mask_edges: bool = True
    """Apply map-anything's normals+depth edge mask to its predicted depth map.

    ON is map-anything's default.  ``apply_mask`` is inert on ordinary indoor scenes,
    where MA's non-ambiguous mask comes back all-ones, so this is the only filter unless
    ``recon_confidence_min`` is set.  It is the wrong tool for MA's characteristic
    failure, which is smooth rather than noisy: on a dark textureless backdrop MA can
    hallucinate one coherent, badly wrong surface, and the edge test fires along that
    surface's internal cliffs -- shredding it into disconnected islands that keep their
    wrong depth and become background-Gaussian floaters.  ``recon_confidence_min``
    rejects such a region outright instead.
    """

    recon_confidence_min: Optional[float] = None
    """Drop predicted-depth pixels whose ``conf`` is at or below this.

    ``None`` (default) leaves MA's confidence unused.
    The conf head is ``expp1``-activated, so **1.0 is its FLOOR**, not its middle, and
    the distribution is bimodal: a spike at the floor, then a jump to ~5-10 on resolved
    geometry.  The floor means "I am guessing" and reliably marks hallucinated
    geometry such as the surfaces described under ``recon_mask_edges``.

    Threshold the floor, NOT a percentile: MA's own ``apply_confidence_mask`` is
    quantile-based, which tracks how much of a given image is unresolved rather than
    whether a given pixel is.

    Use **1.01, not 1.0.**  The floor is approached asymptotically and only some pixels
    round all the way onto it, so ``> 1.0`` selects on float representation rather than
    on geometry.  Past the cliff the kept fraction is flat, so 1.01 is a plateau value,
    not a tuned one.

    Both filtering knobs are offered only by map-anything; MoGe/None raise. Both join
    the recon cache key, but only when non-default.
    """

    cache_reconstruction: bool = False
    """Cache map-anything's depth per scene at ``{dataset.path}/{scene}/ma_cache``.

    Keyed by a content hash of the images + conditioning + model id, so it can only
    hit for identical inputs (unlike ``resume_from_cache``, which resumes whole
    pipeline blocks). Off by default so benchmark runs always recompute their own
    depth; useful when the same scene is re-run many times.
    """

    depth_source: str = "pred"
    """Final per-frame depth source: ``'pred'`` (predicted by
    ``reconstruction_model``) or ``'gt'`` (dataset / ground-truth depth).

    ``'gt'`` always raises if the dataset has no depth (no fallback).
    Orthogonal to conditioning: feeding GT depth to map-anything as an *input*
    is controlled separately by ``condition_recon_model_on_gt_depths``.
    """

    camera_poses_source: str = "pred"
    """Final per-frame camera-pose (c2w) source: ``'pred'`` (from
    ``reconstruction_model`` — ``moge``/``None`` → identity, ``map_anything`` →
    MA-predicted) or ``'gt'`` (dataset / ground-truth poses).

    ``'gt'`` always raises if the dataset has no poses (no fallback).
    Orthogonal to conditioning: feeding GT poses to map-anything as an *input*
    is controlled separately by ``condition_recon_model_on_gt_poses``.
    """

    condition_recon_model_on_gt_poses: bool = False
    """Feed GT camera poses to the reconstruction backend as a conditioning input.

    ``True`` requires ``reconstruction_model='map_anything'`` (else ``ValueError`` at
    ``Sequence`` init) and a dataset with GT poses (else raises). A no-op for the final c2w when
    ``camera_poses_source='gt'`` (the prediction is discarded either way); its purpose is
    improving the prediction when ``camera_poses_source='pred'``.

    A SOFT prior: map-anything encodes the poses view-0-relative into its image tokens
    and then regresses its own. This primes the model; it does not pin the cameras.
    """

    condition_recon_model_on_gt_depths: bool = False
    """Feed GT depth to the reconstruction backend as a conditioning input.

    Same gate as ``condition_recon_model_on_gt_poses`` (``map_anything`` only), and
    the dataset must have GT depth (else raises). A no-op for the final depth when
    ``depth_source='gt'``; its purpose is improving the prediction when
    ``depth_source='pred'``.

    map-anything returns the conditioning depth and K VERBATIM for a conditioned frame
    (its `cropped_image` is None and the original-resolution GT is kept). Do not read a
    conditioned MA depth as a prediction.
    """

    recon_on_test_views: bool = True
    """Run map-anything on held-out test views too (not only pipeline-facing
    input views).

    Applies to datasets that materialize a held-out test set alongside the
    input views (``gso``, ``co3d``). With the default ``True``, MA processes
    the full view set so multi-view
    inference has a robust pairwise-distance anchor, and downstream consumers
    (e.g. CO3D ``renders_test/`` via :meth:`Sequence.predicted_camera`) can
    use MA's predicted camera for the held-out views. Set ``False`` to
    restrict MA to the pipeline-facing input views only.

    Ignored when ``reconstruction_model != 'map_anything'``.
    """


# =====================================================================
# Deformation-field warp evaluation (config group: deformation_warp/)
# =====================================================================

@dataclass
class DeformationWarpConfig:
    """Sampling-side knobs for the per-canonical-mesh-vertex deformation
    field built by ``GT_SHAPES_INVERSION`` in any per-frame mode.

    The field lives on canonical mesh vertices, but consumers evaluate it at
    arbitrary off-vertex points (decoded Gaussian centers, voxel-grid
    centers, mesh primitive positions).  These knobs control the KNN+IDW
    interpolation + SO(3) blend done by ``_warp_at_high_res`` and its two
    wrappers (``warp_gaussians_high_res``, ``warp_voxel_coords_high_res``)
    in ``core/utils/deformation.py``.

    Consumers across the pipeline that share these knobs:

    - Stage-2 in-ODE rendering guidance (the single-timestamp ``canonical``
      branch of ``appearance_init=canonical_unified``)
    - Stage-2 ODE-steps debug viz (``render_appearance_ode_steps``)
    - Keyframes-video rendering and before/after evaluation across **all**
      pipeline blocks (``render_keyframes`` / ``render_voxel_keyframes`` in
      ``core/utils/evaluation.py``)
    - FINAL's exports and the post-hoc renders (``core/final.py``,
      ``core/utils/eval_assets_export.py``, ``core/utils/colmap_export.py``,
      ``core/utils/render_final_results.py``)

    Defaults are calibrated for V_mesh ~ 5-10k canonical vertices (typical
    actionmesh range).
    """

    knn_k: int = 4
    """K nearest canonical mesh vertices per evaluation point in the
    rigid-LBS warp (KNN+IDW).  Larger = smoother blending across the
    canonical surface, marginally higher per-step cost.  K=4 is sufficient
    for typical V_mesh ~ 5-10k.  Ignored when the field carries faces: the
    support is then the 3 vertices of the snapped canonical face."""

    knn_eps: float = 1.0e-8
    """IDW denominator floor — stabilises weight computation when an
    evaluation point coincides with a canonical mesh vertex (d² → 0)."""

    knn_chunk_size: int = 8192
    """Evaluation-axis chunking for the warp's IDW + rigid-LBS gather
    (``B × K × 3 × 3``).  ``pytorch3d.ops.knn_points`` does its own internal
    tiling, so peak memory is independent of ``V_mesh`` — this only bounds
    the per-chunk gather; smaller values pay per-chunk launch overhead.
    The ``deformation_warp`` YAML sets 131072."""


# =====================================================================
# Per-phase loss weights & learning rates
# =====================================================================

@dataclass
class LossConfig:
    """Per-phase loss weights, learning rates, and iteration count.

    One instance per pipeline phase (global_pose_refine, appearance_init rendering guidance, finetune).
    Controls WHAT gets optimized and HOW MUCH each loss contributes.
    """

    # Optimization
    num_iterations: int = 100
    early_stop_patience: int = 0      # 0 = disabled; stop after N iterations without improvement
    lr_rotation: float = 0.01
    lr_translation: float = 0.001
    lr_scale: float = 0.001

    # Global-mode batching (only consulted on the global refinement path,
    # refine_poses_global_composite)
    batch_size: int = 0               # 0 = all frames

    # RGB loss (L1)
    rgb_weight: float = 1.0
    rgb_multiscale: bool = True
    rgb_multiscale_scales: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.25])
    rgb_multiscale_weights: List[float] = field(default_factory=lambda: [0.5, 0.3, 0.2])
    rgb_ssim_weight: float = 0.0

    # Silhouette loss
    silhouette_weight: float = 0.0
    silhouette_com_weight: float = 1.0
    silhouette_sdt_weight: float = 0.1
    silhouette_iou_weight: float = 1.0
    occlusion_robust_silhouette: bool = True

    # Regularization (anchor to initial pose). regularization_weight is the
    # master; the three relative weights scale each component independently
    # (all 1.0 = uniform). Set translation/scale to 0 to
    # regularize rotation only.
    regularization_weight: float = 0.0
    regularization_rotation_weight: float = 1.0     # relative weight, rotation component
    regularization_translation_weight: float = 1.0  # relative weight, translation component
    regularization_scale_weight: float = 1.0        # relative weight, scale component

    # Depth
    depth_weight: float = 0.0
    depth_loss_type: str = "l1"       # "l1" or "scale_shift_invariant"
    depth_mask_only: bool = True
    """Restrict depth loss to pixels inside the GT object mask.
    When True, rendered depth outside the mask is ignored (no silhouette-like
    penalty from depth).  Should be True for single-object rendering and False
    for composite rendering where the depth buffer already handles occlusions."""

    render_with_background: bool = False
    """Composite pose refinement only: append DETACHED pointmap-background
    Gaussians to the render so it has valid depth OUTSIDE the object union.
    This is what makes ``has_background=True`` truthful — pair it with
    ``depth_mask_only=False`` for full-image depth supervision (the
    ``has_background`` assertion otherwise rejects that combination).  Mirrors
    ``FinetuningConfig.render_with_background``; the background is a fixed
    reference (all params detached, no pose gradient).  Full-image mode passes
    an all-ones loss mask, so the silhouette term goes inert.  Off by default."""

    # Chamfer (3D shape alignment vs GT depth) — pose refinement
    chamfer_weight: float = 0.0
    """Optional one-directional Chamfer loss: GT-depth object points → nearest
    posed-model (Gaussian-mean) point.  0 = off.  Gives a wide-basin 3D
    alignment signal that the pixel/silhouette losses lack — robust to a poor
    init, and supplies direct translation + scale gradients.  **One-directional
    (GT→model)** because single-view GT depth is the visible front surface only:
    each observed GT point pulls toward its nearest model point, and the model's
    unseen back is never penalized.  Not pixel-aligned.  Wired into the
    render-based pose-refine paths (per-frame, global-composite, shared-world)."""

    chamfer_max_gt_points: int = 2048
    """Max GT-depth points sampled per object/frame for the Chamfer (the term is
    O(M·N); GT points = object mask ∩ finite-pointmap pixels)."""

    chamfer_max_model_points: int = 8192
    """Max posed-model points (Gaussian means) per object for the Chamfer
    (a fixed random subset, stable across iterations)."""

    chamfer_gt_trim_factor: float = 3.0
    """Drop BACKGROUND points from the Chamfer GT target: a median-centred radius cut at
    ``factor x p95``, applied once per frame when the cloud is built (before the
    ``chamfer_max_gt_points`` subsample, so the cap is spent on real object points).
    ``0`` = off.

    A segmentation mask bleeding onto the wall behind the subject contributes pixels
    whose depth is valid and finite -- so neither ``recon_confidence_min`` nor
    ``recon_mask_edges`` rejects them, both being about whether a measurement is
    trustworthy rather than whose it is -- and the Chamfer is a mean of SQUARES, so a
    handful of such points can dominate the gradient.

    Uses ``rendering.bulk_keep_mask``; keep the two in step if either moves.  The
    factor is NOT that function's default of 1.5, which is too tight for this use.
    3.0 is the tightest cut that keeps every point of clean GT-depth clouds (so it is a
    no-op on clean data); below it the cut starts eating the object, and above it
    cleaning degrades while buying nothing back.

    Note: ``p95`` is blind to a blob holding more than 5% of the masked pixels.  Outlier
    rejection, not segmentation repair."""

    # Surface normals
    normals_weight: float = 0.0        # sign-invariant cosine vs depth-derived normals
    normals_discontinuity_threshold: float = 0.02  # mask depth-disc pixels; 0=disabled

    # Perceptual
    perceptual_weight: float = 0.0
    # Crop LPIPS inputs to the GT mask bbox (+ margin) before the
    # perceptual model — VGG activation memory then scales with the
    # object, not the full (mostly-background) frame.  Default on: objects
    # usually occupy only part of the frame, so the crop is a near-free memory
    # win wherever a perceptual loss is active.
    perceptual_crop_to_mask: bool = True
    perceptual_crop_margin: float = 0.1  # padding as fraction of bbox extent, per side
    # Run the LPIPS/VGG forward under bf16 autocast — halves its retained
    # activation memory (only the pred-path activations are stored; the gt
    # branch holds no grad).  Off by default: bf16 loses precision in LPIPS's
    # channel-wise feature normalisation + squared difference, which bites most
    # near convergence.  Loss-path only — never the eval metric.
    perceptual_autocast_bf16: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to dictionary."""
        d = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, (list, tuple)):
                val = list(val)
            d[f.name] = val
        return d


# =====================================================================
# Global pose refinement  (config group: global_pose_refine/)
# =====================================================================

@dataclass
class GlobalPoseRefineConfig(LossConfig):
    """Global pose refinement (GLOBAL_POSE_REFINE_1 / GLOBAL_POSE_REFINE_2 blocks).

    Extends LossConfig with per-block output flags.  Consumer functions
    that accept ``LossConfig`` work unchanged (IS-A relationship).
    """

    enabled: bool = True
    """Whether to run this block. False = skip global refinement."""

    strategy: str = "default"
    """``"default"`` | ``"icp"`` -- what the pose is optimised AGAINST.

    ``default`` is the photometric loop.  ``icp`` hands control to ``pose_refit``'s
    closed-form solver, which matches the RENDERED surface to the observed one by nearest
    neighbour instead of per pixel, so a few pixels of misregistration compare
    CORRESPONDING surface rather than unrelated surface.

    Under ``icp`` most of this config is INERT: every pixel-loss weight, the learning
    rates, ``early_stop_patience``.  ``num_iterations`` IS used, as the solver's step
    budget.

    Orthogonal to ``correction_granularity``, which chooses the PARAMETERS rather than the loss."""

    correction_scale_control: str = "shared"
    """``"shared"`` (default) | ``"perframe"`` -- which transform owns SCALE under
    ``correction_granularity: both``.

    ``shared`` freezes the NATIVE scale, so the one shared correction is all that sets the
    object's size.  ``both`` is gauge-ambiguous by construction and that is tolerated for
    rotation and translation (the composite is what gets written back, so the split never
    leaves the solver); scale is the case where it should not be, because the object has
    ONE size and a per-frame transform modelling it lets the optimiser breathe the object
    frame-to-frame while the correction chases the residual.  ``perframe`` leaves the
    native scale free beside the correction.

    INERT outside ``both`` -- under ``per_frame`` there is no shared transform to hold the
    scale, and under ``shared`` the natives are frozen already.  Read it through
    :func:`resolve_correction_scale_control`.

    NOT expressible with ``losses.lr_scale``: that drives the native AND the
    correction's scale at every call site, so zeroing it freezes both."""

    correction_granularity: str = "per_frame"
    """``"per_frame"`` | ``"shared"`` | ``"both"`` -- the GRANULARITY of the pose
    correction this block optimises.  Orthogonal to ``strategy``: a shared correction is
    available under the photometric loss too.

    - ``per_frame`` -- one correction per frame.  Under the ICP strategy, which has no
      native per-frame params, this is one delta per frame.
    - ``shared``    -- ONE Sim(3) per object, in the object's CANONICAL frame, composed
      onto every frame's existing pose (``pose_i o dT``), so per-frame root MOTION is
      preserved.  Per-frame params frozen.  Well-posed: the correction IS the systematic
      error and can be read as such.
    - ``both``      -- natives AND the correction free, SIMULTANEOUSLY (both live from
      step 0).  This is ``per_frame`` plus 7 REDUNDANT dof: it cannot reach any pose set
      ``per_frame`` cannot, since ``native_i = target_i o dT^-1`` reproduces any target.
      What it changes is the OPTIMISER TRAJECTORY -- the shared parameter supplies one
      coherent all-frames-at-once descent direction that per-frame params can only
      approximate by moving in lockstep, which under Adam (per-parameter normalisation)
      is a real preconditioning effect.  An optimiser device, NOT a modelling choice.
      The ICP solver instead estimates the two hierarchically (``icp_perframe_prior_weight``).

    Every solver honours all three values identically: the ICP solver, the composite and
    global-scale photometric paths, the shared-world path, and FINETUNE.

    .. warning::
       ``both`` carries a GAUGE FREEDOM: for any dT', ``pose_i -> pose_i o dT o dT'^-1``
       paired with dT' composes to bit-identical poses, so a 7-dimensional family of
       settings has exactly equal loss.  The COMPOSED pose stays identifiable and still
       converges; the SPLIT between the correction and the natives is decided by
       initialisation and the optimiser path, not by the data.  So under ``both`` the
       fitted correction must NOT be read as "the systematic error" -- that reading is
       valid only under ``shared``.  It is harmless in practice because the correction is
       ABSORBED into the poses at write-back and nothing downstream reads the split.

       Pinning it is possible but NOT done: ``regularization_weight`` already anchors the
       natives and never the delta in all three photometric solvers, which is the right
       sign for a gauge fix -- but no preset ships enough of it to close all 7 dof, and
       FINETUNE has no pose regulariser at all.

    A DIFFERENT axis from ``pipeline.mv_shared_world_pose``, which chooses how many pose
    parameters EXIST (one per (object, timestamp), every view DERIVED from it by a
    camera rebase) rather than what rides on top of them.  The two compose: a correction
    in the object's canonical frame is applied identically to every frame, so it
    preserves the shared-world invariant by construction."""

    refine_geometry: str = "canonical"
    """``"canonical"`` | ``"own_frame"`` -- WHAT GEOMETRY each frame is refined against,
    and with it whether SCALE is shared.

    - ``canonical`` (default) -- every frame renders ONE shared object
      (``canonical_gaussians_with_fallback``, which on a per-frame-only state collapses to
      the EARLIEST frame's Gaussian) and scale is optimised globally.
    - ``own_frame`` -- every frame renders ITS OWN reconstruction
      (``state.perframe_gaussians[obj][frame]``) and scale is per-frame.

    This one value picks three call arguments of the refine runner: the Gaussian dict,
    ``per_frame_canonical`` and ``refine_scale``.  The ICP branch needs nothing extra:
    ``correction_granularity: per_frame`` is one delta per frame there.

    ORTHOGONAL to ``correction_granularity``, which chooses what rides ON TOP of the per-frame
    poses, and to ``strategy``, which chooses the solver."""

    # ---- ICP strategy (strategy="icp") ----------------------------------
    # A closed-form point-to-plane solver: trimmed, normal-gated, Welsch-weighted
    # correspondences on a graduated-non-convexity schedule, solved in the object's
    # canonical frame.  ``num_iterations`` is its step budget.
    icp_fit_scale: bool = True
    """Let the ICP delta change the object's size.  False holds the size (the scale
    column is dropped from the least-squares system)."""

    icp_p2pl_trim: float = 0.8
    """Keep this quantile of the surviving correspondences, closest first (Chetverikov's
    trimmed ICP).  The third and last rejection, after the distance gate and the normal
    test.  1.0 = keep everything."""

    icp_p2pl_gnc_from: float = 8.0
    """Graduated non-convexity: the correspondence kernel's half-width at iteration 0, in
    units of the observed cloud's own median point spacing.  Also sets the solver's REACH,
    matches being gated at ``3 x nu x resolution`` -- so this is the first knob to raise if
    a run reports a low ``gate_kept`` and does nothing."""

    icp_p2pl_gnc_to: float = 0.8
    """The kernel half-width at the LAST iteration (geometric interpolation from
    ``icp_p2pl_gnc_from``).  Narrowing makes the late iterations reject outliers the early
    ones needed in order to find the basin."""

    icp_p2pl_normal_threshold: float = 0.5
    """Reject a correspondence whose two normals disagree by more than this cosine -- it
    has crossed to the far side of a thin structure.  -1.0 disables the test."""

    icp_perframe_prior_weight: float = 1.0
    """The per-timestamp residual's ridge toward the shared transform, as a multiple of
    ONE frame's evidence (each frame's block is normalised by its own total correspondence
    weight).  Larger = more of the error forced onto the shared Sim(3).

    REQUIRED > 0 under ``correction_granularity: both``, and the solver raises otherwise:
    with both transforms free the system is rank-deficient along ``(shared + v, per_frame
    - v)``, since moving error between them changes no residual.  Inert under the other
    granularities."""

    icp_p2pl_normals: str = "pca"
    """``"pca"`` | ``"depth"`` -- where the surface normals come from.

    - ``pca`` -- local PCA over the scattered cloud.
    - ``depth`` -- a central difference on the depth map's own GRID (``depth_to_normals``),
      which is far better conditioned and needs no KNN.  Both ICP clouds are unprojected
      depth maps, so the grid is available on each side.

    Under ``depth`` a pixel whose normal is unreliable (the 1px border, an eroded mask
    edge, a degenerate patch) yields the ZERO vector and is rejected by the normal gate on
    its own.  A frame whose observation carries no grid normals falls back to PCA."""

    icp_keep_best_iterate: bool = False
    """Return the LOWEST-residual iterate rather than the last one.  The residual is not
    monotone: the GNC schedule narrows the gate as it runs, so late iterations fit a
    shrinking correspondence set and can drift back up.  It selects on the one-directional
    chamfer, which growing the object lowers, so it is a within-run tie-break rather than
    a scale criterion."""

    icp_max_scale_step: float = 0.1
    """Clamp on the fractional size change ONE iteration may take.  The Chamfer has
    nothing opposing growth, so an early iteration solving against bad correspondences
    could otherwise leap; this bounds the leap without changing the objective's minimum."""

    icp_max_gt_points: Optional[int] = None
    """Observed-surface points per frame, subsampled ONCE (not per iteration).  ``null`` /
    0 = every masked pixel.  The "model" side of the match is the RENDERED depth map
    unprojected to a cloud, so its size is the render resolution."""

    icp_target_seed: Optional[int] = 0
    """Base seed for the observed-cloud subsample (only drawn when ``icp_max_gt_points``
    caps it); ``null`` = draw fresh every run.  The per-frame seed is
    ``base + f(obj, frame, view)`` (``pose_refit.icp_target_seed``), so views do not share
    one draw."""

    icp_pred_trim_factor: float = 0.0
    """Outlier trim on the RENDERED cloud, as a multiple of its p95 radius.  0 = off.
    The mirror of ``chamfer_gt_trim_factor``, which trims the observed cloud; same
    primitive (``rendering.bulk_keep_mask``), so it keeps at least 95% of the cloud."""

    icp_chamfer_reverse_weight: float = 0.0
    """Add ``w * chamfer(rendered -> observed)``, penalising rendered surface with no
    observation near it.  0 = off.  The forward term is one-directional by design (the GT
    is one view's visible surface, so occluded model surface has no GT to match), which
    also leaves it nothing opposing GROWTH; this term trades some occlusion tolerance for
    that."""

    icp_force_voxel_splat: bool = True
    """Render the canonical SHAPE, one Gaussian per voxel, rather than the block's own
    reconstruction.  The splat is the only source present on EVERY frame: Gaussians exist
    only after a Stage-2 pass, and a GT-injected or Stage-1 shape is occupancy, not a SLAT.
    Set false to align the reconstruction instead.  Either way an object with no voxel
    grid falls back to its Gaussians, and is skipped only when it has neither."""

    # Per-block output flags (None = inherit from global OutputConfig)
    save_renders: Optional[bool] = None
    save_metrics: Optional[bool] = None
    save_cache: Optional[bool] = None

    exit_after: bool = False
    """Exit pipeline after this block (useful for debugging)."""

    _VALID_CORRECTION_GRANULARITY = ("per_frame", "shared", "both")

    _VALID_REFINE_STRATEGY = ("default", "icp")
    # Mirrors ``pose_refit.ICP_P2PL_NORMALS``.
    _VALID_ICP_P2PL_NORMALS = ("pca", "depth")

    def _validate_refine_strategy(self) -> None:
        if self.strategy not in self._VALID_REFINE_STRATEGY:
            raise ValueError(
                f"{type(self).__name__}.strategy must be one of "
                f"{self._VALID_REFINE_STRATEGY}, got {self.strategy!r}")
        if self.icp_p2pl_normals not in self._VALID_ICP_P2PL_NORMALS:
            raise ValueError(
                f"{type(self).__name__}.icp_p2pl_normals must be one of "
                f"{self._VALID_ICP_P2PL_NORMALS}, got {self.icp_p2pl_normals!r}")

    def __post_init__(self):
        self._validate_refine_strategy()
        if self.correction_granularity not in self._VALID_CORRECTION_GRANULARITY:
            raise ValueError(
                f"GlobalPoseRefineConfig.correction_granularity must be one of "
                f"{self._VALID_CORRECTION_GRANULARITY}, got {self.correction_granularity!r}")
        # The module-level resolver, not a second copy of the value list: it is the guard
        # a real (Hydra) run gets, and two lists would drift.  Defined below this class --
        # fine, the lookup happens at instantiation.
        resolve_refine_geometry(self, "GlobalPoseRefineConfig")


# =====================================================================
# Pipeline control (cross-phase)
# =====================================================================

@dataclass
class PipelineConfig:
    """Cross-phase pipeline control flags.

    Controls WHICH stages run and HOW they behave globally.
    Not specific to any single optimization phase.
    """

    # NOTE: is_dynamic and is_mv are not config flags: they are properties of
    # `Sequence`, derived from the FrameKey set carried by the loaded data.
    # Pipeline configs are algorithmic recipes, not data-shape declarations.
    # Read these via `sequence.is_dynamic` / `sequence.is_mv`.

    optimize_pose_tokens: bool = True
    """Optimize raw Stage 1 pose tokens instead of final pose parameters.
    When True, backpropagates through the differentiable pose decoder chain
    (6D rotation, log-scale, translation). Falls back to direct pose optimization
    when raw modalities are unavailable (e.g. a cache without them)."""

    white_background: bool = False
    """Use white background (instead of black) when rendering Gaussians.
    Applies to all rendering: optimization losses, evaluation, keyframe videos."""

    verbose: bool = True
    """Verbose logging during refinement."""

    log_interval: int = 20
    """Log progress every N iterations."""

    mv_shared_world_pose: bool = True
    """Multi-view: collapse all per-camera object poses to a single shared
    local->world transform derived from the reference frame's prediction.
    Other frames' predicted camera-space poses are discarded; per-frame
    local->cam poses are derived via c2w_i^{-1} @ c2w_ref @ P_ref.

    True by default: per-view object poses are not a quantity a single rigid
    object has.  MONOCULAR data ignores it -- after ``Sequence`` construction
    ``Pipeline.bind`` turns it off with a ``logger.debug`` when ``sequence.is_mv`` is
    False, one view per timestamp making the collapse vacuous, so this being true
    costs mono runs nothing.
    """

    # NOTE: data-shape checks involving mv_shared_world_pose run after Sequence
    # construction, since data shape is derived from the FrameKey set, not
    # declared in config.


# =====================================================================
# Output
# =====================================================================

@dataclass
class OutputConfig:
    """Output directories, rendering, and export options."""
    
    output_dir: Optional[str] = None
    """Directory to save outputs. Auto-generated if omitted."""

    experiment_suffix: Optional[str] = None
    """Appended (underscore-joined) to the experiment segment of the auto-generated
    output path. Ignored when output_dir is set or no +experiment= is active."""

    save_renders: bool = True
    """Save rendered images and comparisons."""

    save_metrics: bool = True
    """Save metrics to JSON file."""

    save_cache: bool = True
    """Save pipeline state cache (.pt) after each block."""

    suppress_intermediate_renders: bool = False
    """Force save_renders=False for every pipeline block (and any other tag). FINAL unaffected."""

    suppress_intermediate_metrics: bool = False
    """Force save_metrics=False for every pipeline block (and any other tag). FINAL unaffected."""

    render_decoded: bool = False
    """Render decoded objects from multiple viewpoints right after decoding."""

    render_size: int = 512
    """Size of multi-view rendered images (square)."""

    render_distance: float = 2.0
    """Camera distance from object for multi-view rendering."""

    render_fov: float = 40.0
    """Field of view in degrees for multi-view rendering."""

    save_decoded_ply: bool = False
    """Save decoded Gaussian objects as PLY files."""

    save_compressed_ply: bool = True
    """Use compressed PLY format when saving."""

    save_decoded_meshes: bool = False
    """Save decoded mesh objects as OBJ files."""

    save_output_ply: bool = False
    """Save full posed scene as per-frame PLY files."""

    save_output_mesh: bool = True
    """Save per-object canonical meshes (decoded from SLAT) to {final}/meshes/."""

    save_output_renders: bool = True
    """Render the full foreground scene (no background) from each dataset
    camera view and save as PNGs in {final}/renders_train/. Canonical path only."""

    save_colmap: bool = True
    """Write portable COLMAP sparse models to {final}/colmap/ (shared images/ +
    one sparse/{frame:03d}/{cameras,images,points3D}.{bin,txt} per temporal
    frame): each frame's train + held-out test cameras and a colored foreground
    point cloud from that frame's posed Gaussian means."""

    save_world_assets: bool = True
    """Write world-space composites of the whole scene next to the per-object
    assets: gaussians/world.ply (all foreground objects posed),
    gaussians/background.ply (every frame's background lifted by its own camera,
    unioned) and meshes/world.glb.  Dynamic scenes get world/{frame:03d}.{ply,glb},
    1:1 with colmap/sparse/.  Built from the same source as the COLMAP cloud, so
    all of them open aligned in a viewer."""

    save_slat_voxels: bool = False
    """Visualize SLAT tokens as PCA-colored voxel grid."""

    save_slat_voxel_mesh: bool = False
    """Render SLAT voxel grid as a cube mesh colored by XYZ position."""

    save_tracks_2d: bool = False
    """Write the dense per-object 2D point-track artifact ``final/tracks_2d.npz``.

    Symmetric with ``save_tracks_3d``: these two gate the two track DATA exports,
    while ``save_viz_tracks_{2d,3d}`` gate the two track PICTURES. Read by
    point-tracking evaluators and by the track overlays of
    ``render_final_results.py``. Convention-correct (reuses compute_object_tracks +
    _project_tracks_to_2d). Eval-only: no GT enters the pipeline."""

    save_tracks_3d: bool = False
    """Write the per-object 3D world track artifact ``final/tracks_3d.npz``
    (``interpolation.write_tracks_3d``).

    Data only. The ``*_tracks_3d_world_space.png`` picture drawn from the same tracks
    is gated by ``save_viz_tracks_3d``."""

    save_viz_tracks_2d: bool = False
    """Draw the 2D track overlay video ``{scene}_tracks_2d_world_space.mp4``
    (reconstruction | GT side by side). A PICTURE; the data is ``save_tracks_2d``."""

    save_viz_tracks_3d: bool = False
    """Draw the 3D track plot ``{scene}_tracks_3d_world_space.png``.
    A PICTURE; the data behind it is ``save_tracks_3d``."""

    tapvid_track_points: int = 16384
    """How many anchors ``final/tracks_2d.npz`` carries per object.

    An evaluator cannot ask the reconstruction "where did pixel (u,v) go" -- it
    substitutes the nearest exported anchor at the query frame, and that substitution is
    a FLOOR on the reported error (~0.5*sqrt(object_area/N) px@256, so 1/sqrt(N)). Too few
    anchors make the tightest TAP-Vid thresholds (1/2 px) report anchor density rather
    than tracking. Only the tracks artifact reads this; the track VISUALIZATIONS use
    their own 16 anchors."""

    save_pose_axes_3d: bool = False
    """Save 3D pose axes of objects over time."""

    save_pose_axes_2d: bool = False
    """Save 2D pose axes of objects over time."""

    save_synth_nvs: Optional[bool] = None
    """Render FOUR *synthesized* held-out novel views per timestamp — no GT, so
    nothing is scored against truth; they feed the CLIP-I view-consistency
    diagnostic instead. Foreground is orbited around the object centroid at the
    train camera's radius; writes
    {final}/renders_synth_nvs/{f:03d}_{tag}.png (tags in
    eval_assets_export.SYNTH_NVS_OFFSETS).

    None (default) = AUTO: on for every dataset that ships no GT held-out split,
    off for the ones that do. Resolve it with `synth_nvs_enabled(cfg)`, never by
    reading this field — a bare truth test would read auto as off. true/false
    force it either way."""

    synth_nvs_azimuth: float = 30.0
    """HORIZONTAL magnitude (deg) of the synthesized views, about the train
    camera's up-axis: two of the four sit at ±this. Only read when the
    synthesized NVS export runs."""

    synth_nvs_azimuth_step: float = 0.0
    """Extra azimuth (deg) added per timestamp (frame_int * step) to all four
    views. 0 ⇒ the viewpoints hold a fixed offset off each train camera, which
    is what the consistency diagnostic wants; nonzero ⇒ a slow orbit as time
    advances. Only read when the synthesized NVS export runs."""

    synth_nvs_elevation: float = 30.0
    """VERTICAL magnitude (deg) of the synthesized views, about the train
    camera's right-axis: the other two sit at ±this. Only read when the
    synthesized NVS export runs."""

    save_viz_orbit: bool = True
    """Render the turntable into {final}/viz/orbit/orbit.mp4 at the end of
    FINAL, via the `orbit` renderer of
    genia.core.utils.render_final_results (same code as the post-hoc
    `-r orbit` CLI, same output path). Dataset-agnostic, qualitative only; a
    no-op for a run that exported no gaussians and no meshes."""

    viz_orbit_frames: int = 60
    """Orbit steps in one full turn (0 ⇒ one per frame key). Only read when
    save_viz_orbit=true."""

    viz_orbit_fps: int = 24
    """Frame rate of orbit.mp4 — with viz_orbit_frames it sets the clip length.
    Only read when save_viz_orbit=true."""

    viz_orbit_slowmo: int = 10
    """Play the motion this many times slower in the turntable: each 1-frame
    interval is drawn over N orbit steps with the DEFORMATION interpolated
    between the two frames (so one sequence loop takes frames x this many
    steps, and viz_orbit_frames rounds up to whole loops). EVERY animated run
    gets that length, whether or not its geometry can be interpolated — same
    duration and same camera path, so two runs on one scene are comparable
    frame for frame; a run that cannot interpolate holds each frame instead.
    The geometry comes from final/final.pt (canonical gaussians + the
    deformation field), whose frames correspond by construction, falling back
    to the exported PLYs. 1 = off, one step per frame. Only read when
    save_viz_orbit=true."""

    viz_orbit_up: str = "auto"
    """Vertical the turntable spins about: auto (the dataset's world up where
    one is known — GSO — else the train camera's own up) | camera | world (R3
    -Y) | "x,y,z". Spinning about an elevated train camera's own up walks the eye
    over the object and out the other side, so the world up is preferred where
    known. Only read when save_viz_orbit=true."""

    save_viz_world_space: bool = True
    """Render the whole scene from one overview camera into
    {final}/viz/world_space/ at the end of FINAL — via the `world_space` renderer
    of genia.core.utils.render_final_results (same code as the post-hoc
    CLI, same output paths). Every object posed into WORLD space, foreground only
    (the background point cloud and the Sim(3) pose axes are both opt-in overlays,
    off by default). Root motion is KEPT, unlike the orbit, so this is the render
    that shows the scene's LAYOUT — where the objects sit relative to each other,
    and where they travel.

    ONE OVERVIEW CAMERA PER VIEW, each behind the input camera that observed it —
    an MV run whose views agree in world space (the mv_shared_world_pose invariant)
    would otherwise render V copies of one picture. Each view KEEPS ITS OWN
    ELEVATION too, centred on the mean across views: two cameras differing mainly
    in height would otherwise collapse onto one viewpoint, and centring is what
    leaves a mono run unchanged. Target,
    distance and focal are shared, so the views stay comparable in position and
    scale. World up is assumed to be R3 -Y, which is only KNOWN on a run that kept
    the dataset's cameras — it moves the absolute tilt and roll, never the per-view
    separation.

    ONE PNG PER (FRAME, VIEW) plus one mp4 per view that has more than one frame:
    mono-static a single PNG, mono-dynamic N PNGs + one clip, MV-static V PNGs and
    no clip (no time axis to play), MV-dynamic V*N PNGs + V clips. A multi-view run
    suffixes EVERY view including view 0 ({frame:03d}_v{view:02d}.png), departing
    from frame_key_stem: these are one scene from several sides, and a bare 000.png
    beside 000_v01.png does not read as a member of that series. Mono keeps the
    plain {frame:03d}.png.

    Framed on the FOREGROUND: the camera aims at the objects' box centre and its
    focal is fitted to them over the whole sequence, so the background fills the
    frame and spills off the edges rather than dictating a zoom-out. (The
    world-space keyframe render {scene}_keyframe_final_world_space_*.png, also
    written by run_final, instead aims at the mean of the camera and object
    positions at a fixed 60 degree FoV.)

    Written for EVERY run, static and dynamic, including a run whose cameras are
    all identity (e.g. OursActionBench), which the keyframe render skips via its
    has_c2w guard."""

    save_viz_track_overlays: bool = True
    """Re-render with the run's 3D tracks overlaid, into
    {final}/viz/track_overlays/ at the end of FINAL — via the `track_overlay`
    renderer of genia.core.utils.render_final_results plus ONE off-axis
    renderer (same code as the post-hoc CLI, same output paths). WHICH off-axis
    view is the dataset's call, taken by the same synth_nvs_enabled() that gates
    renders_synth_nvs/: a dataset with a GT held-out split gets
    `track_overlay_test` -> test_views/ (the real cameras its NVS numbers were
    scored from — no reason to synthesize a view the dataset already supplies),
    everything else gets `track_overlay_nvs` -> synth_nvs/.
    FIGURE assets with ANNOTATED, 2x-supersampled, uncropped pixels, which is why
    they get their own folder rather than sharing renders_train/,
    viz/train_views/ or viz/synth_nvs/: anything named after a render an
    evaluation reads stays clean enough to be scored. Written for every DYNAMIC
    run: a run with no 3D tracks gets the same viewpoint and resolution with
    nothing drawn on top, so it can be framed alike beside an annotated one. A
    STATIC run (one timestamp — GSO, CO3D, mvcustom) is skipped even with the flag
    on: with no trajectory to draw the overlay is only a supersampled
    train_views. The post-hoc CLI still renders it on an explicit
    `-r track_overlay`."""


# =====================================================================
# Top-level config
# =====================================================================

@dataclass
class GeneralConfig:
    """The base config sections.

    The block sections are :class:`genia.core.config.GeniaConfig`.
    """

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    deformation_warp: DeformationWarpConfig = field(default_factory=DeformationWarpConfig)
    global_pose_refine_2: GlobalPoseRefineConfig = field(default_factory=GlobalPoseRefineConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


# =====================================================================
# Config table printing
# =====================================================================

def print_config_table(
    config,
    title: str = "GenIA Pipeline Configuration",
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Render a config mapping as a sectioned rich table.

    Top-level entries whose value is a mapping become sections; their immediate
    (non-None) fields are listed. Used on a plain OmegaConf container
    (pipeline startup).
    """
    from collections.abc import Mapping

    from rich.box import ROUNDED
    from rich.table import Table

    from .console import CONSOLE

    table = Table(
        title=f"[bold]{title}[/bold]",
        box=ROUNDED,
        header_style="bold magenta",
    )
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value")

    def _add_section(name: str, items) -> None:
        rows = [(k, v) for k, v in items if v is not None]
        if not rows:
            return
        table.add_section()
        table.add_row(f"[bold yellow]{name}[/bold yellow]", "")
        for k, v in rows:
            table.add_row(f"  {str(k).replace('_', ' ')}", str(v))

    for name, val in config.items():
        if isinstance(val, Mapping):
            _add_section(name, list(val.items()))

    if extra:
        _add_section("runtime", list(extra.items()))

    CONSOLE.print(table)


# =====================================================================
# Utility: print loss config
# =====================================================================

def print_loss_config(
    losses: LossConfig,
    pipeline: PipelineConfig,
    stage: str = "",
) -> None:
    """Print loss config details to console.

    Parameters
    ----------
    losses : LossConfig
        The phase-specific loss configuration.
    pipeline : PipelineConfig
        Cross-phase pipeline settings.
    stage : str
        Stage label (e.g., "per-frame", "global").
    """
    from rich.box import ROUNDED
    from rich.table import Table

    from .console import CONSOLE

    table = Table(
        title=f"[bold]{stage.capitalize()} loss config[/bold]",
        box=ROUNDED,
        header_style="bold magenta",
    )
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value")

    table.add_row("Iterations", str(losses.num_iterations))
    table.add_row("LR rot / trans / scale",
                  f"{losses.lr_rotation} / {losses.lr_translation} / {losses.lr_scale}")
    table.add_row("RGB",
                  f"l1  w={losses.rgb_weight}  "
                  f"ssim={losses.rgb_ssim_weight}  multiscale={losses.rgb_multiscale}")
    table.add_row("Silhouette", str(losses.silhouette_weight))
    table.add_row("Regularization",
                  f"w={losses.regularization_weight}  "
                  f"rot={losses.regularization_rotation_weight}  "
                  f"trans={losses.regularization_translation_weight}  "
                  f"scale={losses.regularization_scale_weight}")
    if losses.depth_weight > 0:
        table.add_row("Depth", f"w={losses.depth_weight}, type={losses.depth_loss_type}")
    if losses.normals_weight > 0:
        disc = f", disc_thresh={losses.normals_discontinuity_threshold}" if losses.normals_discontinuity_threshold > 0 else ""
        table.add_row("Normals", f"w={losses.normals_weight}{disc}")
    if "global" in stage.lower():
        table.add_row("Batch size",
                      str(losses.batch_size) if losses.batch_size > 0 else "all frames")

    CONSOLE.print(table)


# =====================================================================
# Per-block output flag resolution
# =====================================================================

from contextlib import contextmanager

def _blocks_of(cfg) -> tuple:
    """The block manifest (``core/manifest.py``)."""
    from genia.core.manifest import BLOCKS
    return BLOCKS


def block_config(cfg, block_name: str):
    """``block_name``'s config section, or ``None`` when the run has no such block.

    The blocks and their sections come from the manifest (:func:`_blocks_of`).
    """
    section = dict(_blocks_of(cfg)).get(block_name)
    return None if section is None else getattr(cfg, section)


def get_block_output_flag(cfg, block_name: str, flag_name: str) -> bool:
    """Resolve a per-block output flag, falling back to global default.

    Checks ``block_config.<flag_name>`` first; if *None* (not overridden)
    or unavailable, falls back to ``cfg.output.<flag_name>``.

    If the block config has ``enabled=False``, all output flags are forced
    to False (the block did no work, so there is nothing to save/render).
    """
    block_cfg = block_config(cfg, block_name)

    # Blocks with enabled=False produce no output
    if block_cfg is not None and not getattr(block_cfg, "enabled", True):
        return False

    # Checked for an UNRECOGNISED tag too: suppression is a property of the output,
    # not of whether this name happens to be a manifest block.
    # FINAL and PREPROCESSING cannot be caught: neither passes its name here.
    if flag_name == "save_renders" and cfg.output.suppress_intermediate_renders:
        return False
    if flag_name == "save_metrics" and cfg.output.suppress_intermediate_metrics:
        return False

    if block_cfg is not None:
        val = getattr(block_cfg, flag_name, None)
        if val is not None:
            return bool(val)
    return bool(getattr(cfg.output, flag_name))


GT_TEST_VIEW_DATASETS = frozenset({"gso", "oursactionbench", "co3d"})
"""Datasets that ship a GT held-out split, i.e. whose FINAL writes a SCORED
``renders_test/`` through one of the ``export_*_eval_assets`` branches.  Keep in
sync with that if/elif chain in ``run_final`` (``core/final.py``); a dataset
missing here only gets a redundant qualitative render, never a wrong metric."""


VALID_CORRECTION_GRANULARITY = ("per_frame", "shared", "both")
"""The granularity a block's ``correction_granularity`` may select.  Mirrored on both
dataclasses; kept here too so the runtime resolver below does not have to pick one."""


VALID_CORRECTION_SCALE_CONTROL = ("perframe", "shared")
"""Which transform owns SCALE under ``correction_granularity: both``; read by the
runtime resolver below."""


CHAMFER_GT_TRIM_DEFAULT = 3.0


def resolve_chamfer_gt_trim(node, block_label: str):
    """The block's ``chamfer_gt_trim_factor``, from the config it ACTUALLY sees.

    An absent key resolves to the shipped default rather than to "off".
    ``0`` means off explicitly.
    """
    value = getattr(node, "chamfer_gt_trim_factor", None)
    if value is None:
        return CHAMFER_GT_TRIM_DEFAULT
    value = float(value)
    if value == 0.0:
        return None
    if not value >= 1.0:
        # Below 1.0 the cut passes INSIDE the p95 bulk and starts eating the object.
        raise ValueError(
            f"{block_label}: chamfer_gt_trim_factor must be 0 (off) or >= 1.0 (it "
            f"multiplies the cloud's 95th-percentile radius), got {value!r}")
    return value

def resolve_correction_scale_control(node, block_label: str) -> str:
    """The block's ``correction_scale_control``, read from the config it ACTUALLY sees.

    ``shared`` (the default) freezes the NATIVE scale under ``correction_granularity: both``, so
    the one shared correction is all that sets the object's size.  ``perframe`` leaves the
    native scale free beside it.

    **Inert outside ``both``**, and deliberately not enforced: under ``per_frame`` there is
    no shared transform to hold the scale, and under ``shared`` the natives are frozen
    already.  Leaving it inert lets one config carry a value across runs that vary
    ``correction_granularity``.

    Same absent-key contract as :func:`resolve_correction_granularity`: a block composed from a
    ``none.yaml`` carries only the enable/output keys, and those blocks return before
    using the value.
    """
    value = getattr(node, "correction_scale_control", None)
    if value is None:
        return "shared"
    if value not in VALID_CORRECTION_SCALE_CONTROL:
        raise ValueError(
            f"{block_label}: correction_scale_control must be one of "
            f"{VALID_CORRECTION_SCALE_CONTROL}, got {value!r}")
    return str(value)


def resolve_correction_granularity(node, block_label: str, *,
                            is_mv: bool = False,
                            mv_shared_world_pose: bool = False) -> str:
    """The block's ``correction_granularity``, read from the config a block ACTUALLY sees.

    The dataclass ``__post_init__`` validators cannot do this job: the YAML is the schema
    and the structured dataclasses are never merged into it, so they never run on the
    Hydra config.  Without this, a bad value is accepted silently.

    **The per-frame axis is per TIMESTAMP, never per view.**  ``per_frame`` frees one pose
    per timestamp regardless of how many views observe it; ``shared`` frees one for the
    sequence; ``both`` frees both.  On MULTI-VIEW data that
    meaning depends on ``pipeline.mv_shared_world_pose``: with it on, the views of a
    timestamp share one world placement and the modes mean what they say; with it OFF each
    camera keeps its own object pose, so ``per_frame``/``both`` could only mean a pose PER
    VIEW -- a quantity that does not exist for one rigid object -- and this raises instead.

    Pass ``is_mv`` / ``mv_shared_world_pose`` to get that check.  Both default to False so
    that an unguarded call never raises: the absent-key fallback below is ``per_frame``, so
    a block composed from a ``none.yaml`` resolves to ``per_frame``, and an unconditional
    raise would fire on blocks that return before using the value at all.

    An ABSENT key falls back to ``per_frame``, matching ``refine_strategy``'s defensive
    read: a block composed from a ``none.yaml`` carries only the enable/output keys, and
    those blocks return early anyway.
    """
    value = getattr(node, "correction_granularity", None)
    if value is None:
        return "per_frame"
    if value not in VALID_CORRECTION_GRANULARITY:
        raise ValueError(
            f"{block_label}: correction_granularity must be one of "
            f"{VALID_CORRECTION_GRANULARITY}, got {value!r}")
    if is_mv and not mv_shared_world_pose and value in ("per_frame", "both"):
        raise ValueError(
            f"{block_label}: correction_granularity={value!r} is not available on MULTI-VIEW data "
            "with pipeline.mv_shared_world_pose=false. The per-frame axis is per TIMESTAMP, "
            "and with the shared-world pose off each camera carries its own object pose, so "
            "there is no per-timestamp pose to free -- only a per-VIEW one, which is not a "
            "quantity a single rigid object has. Use correction_granularity='shared', or turn "
            "pipeline.mv_shared_world_pose on so the views of a timestamp share a placement.")
    return str(value)


VALID_REFINE_GEOMETRY = ("canonical", "own_frame")
"""The geometry a refine block may be scoped to.  Mirrored on
:class:`GlobalPoseRefineConfig`; kept here too so the runtime resolver below does not
have to reach into the dataclass."""


def resolve_refine_geometry(node, block_label: str) -> str:
    """The block's ``refine_geometry``, read from the config a block ACTUALLY sees.

    Same reason :func:`resolve_correction_granularity` exists: the structured dataclasses are
    never merged into the Hydra config, so the dataclass ``__post_init__``
    validator never runs on the Hydra config and a bad value would be accepted silently.

    ``canonical`` -- one shared object per frame, global scale.  ``own_frame`` -- each
    frame against its OWN per-frame Gaussian, per-frame scale.  See
    :attr:`GlobalPoseRefineConfig.refine_geometry`.

    An ABSENT key falls back to ``canonical``, matching :func:`refine_strategy`'s and
    :func:`resolve_correction_granularity`'s defensive reads: a block composed from a
    ``none.yaml`` carries only the enable/output keys and returns before using the value.
    """
    value = getattr(node, "refine_geometry", None)
    if value is None:
        return "canonical"
    if value not in VALID_REFINE_GEOMETRY:
        raise ValueError(
            f"{block_label}: refine_geometry must be one of {VALID_REFINE_GEOMETRY}, got {value!r}")
    return str(value)


def synth_nvs_enabled(cfg) -> bool:
    """Whether FINAL renders the synthesized off-train view (renders_synth_nvs/).

    ``output.save_synth_nvs`` is tri-state, like the per-block flags above: an
    explicit ``true``/``false`` wins, and ``None`` (the default) resolves from the
    dataset — ON wherever there is no GT test split to render instead, so every
    run on a no-GT dataset gets a novel view without having to set the flag.
    """
    flag = cfg.output.save_synth_nvs
    if flag is not None:
        return bool(flag)
    return cfg.dataset.name not in GT_TEST_VIEW_DATASETS


@contextmanager
def output_dir_redirect(cfg, output_dir):
    """Temporarily redirect cfg.output.output_dir and restore on exit.

    Used to control where evaluate_with_canonical_objects() saves its
    metrics JSON and render images.
    """
    from omegaconf import OmegaConf

    original = cfg.output.output_dir
    OmegaConf.update(cfg, "output.output_dir", output_dir)
    try:
        yield
    finally:
        OmegaConf.update(cfg, "output.output_dir", original)


@contextmanager
def block_output_context(cfg, block_name: str):
    """Temporarily override ``cfg.output.save_renders`` and ``save_metrics``
    with per-block values for the duration of a block execution.

    All existing helpers (``capture_before``, ``evaluate_block``,
    ``save_keyframes_video``, ``save_and_plot_loss_history``) read these
    global flags, so the context manager ensures they see the correct
    per-block override without requiring signature changes.
    """
    from omegaconf import OmegaConf

    orig_renders = cfg.output.save_renders
    orig_metrics = cfg.output.save_metrics
    OmegaConf.update(cfg, "output.save_renders",
                     get_block_output_flag(cfg, block_name, "save_renders"))
    OmegaConf.update(cfg, "output.save_metrics",
                     get_block_output_flag(cfg, block_name, "save_metrics"))
    try:
        yield
    finally:
        OmegaConf.update(cfg, "output.save_renders", orig_renders)
        OmegaConf.update(cfg, "output.save_metrics", orig_metrics)


def block_output_subdir(cfg, block_name: str, base: "Optional[str]" = None) -> str:
    """Folder name for a pipeline block's per-run outputs, prefixed with its
    zero-padded run-order index over ENABLED blocks — e.g.
    ``"00_shape_init"``.  Makes the order blocks ran in explicit and
    lexically sortable under the run's timestamp dir.

    The index counts the enabled manifest blocks *preceding* ``block_name``
    (so it is contiguous, with no gaps from disabled blocks) and is derived
    purely from ``cfg`` — i.e. stable across cache/resume runs of the same
    config.  ``base`` defaults to ``block_name`` (every block's folder name
    already equals its block name).  FINAL is not a manifest block
    and is never indexed (its ``final/`` folder stays unprefixed).  An unknown
    ``block_name`` is returned unprefixed (defensive).
    """
    base = base or block_name
    idx = 0
    for bn, section in _blocks_of(cfg):
        if bn == block_name:
            return f"{idx:02d}_{base}"
        if getattr(getattr(cfg, section), "enabled", True):
            idx += 1
    return base


# =====================================================================
# Exports
# =====================================================================

__all__ = [
    # Leaf configs
    "DatasetConfig",
    "ProcessingConfig",
    "OutputConfig",
    # Loss config
    "LossConfig",
    # Per-frame / global pose refinement
    "GlobalPoseRefineConfig",
    # Pipeline
    "PipelineConfig",
    # Top-level
    "GeneralConfig",
    # Utilities
    "print_loss_config",
    "print_config_table",
    # Per-block output flag resolution
    "block_config",
    "get_block_output_flag",
    "resolve_correction_granularity",
    "resolve_refine_geometry",
    "VALID_REFINE_GEOMETRY",
    "resolve_correction_scale_control",
    "VALID_CORRECTION_GRANULARITY",
    "output_dir_redirect",
    "block_output_context",
    "block_output_subdir",
]
