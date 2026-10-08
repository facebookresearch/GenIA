"""Our method's config sections: the blocks GenIA runs, beside the base ``GeneralConfig``.

These dataclasses document the schema of our sections in the Hydra tree. They are not
registered with Hydra (no ConfigStore): the runtime schema is the YAML under
``core/configs/``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from genia.core.utils.config import GlobalPoseRefineConfig, LossConfig

# =====================================================================
# GT shapes inversion  (config group: gt_shapes_inversion/)
# =====================================================================

@dataclass
class GtShapesInversionConfig:
    """GT_SHAPES_INVERSION block — invert GT meshes into Stage-1 shape
    latents (and optionally derive per-frame poses from c2w), writing
    into per-frame ``raw_ss_modalities`` plus, in global mode,
    ``state.canonical_shape_coords``.

    The first block of the pipeline.  Single source of truth for GT
    injection across both per-frame ActionMesh dynamic scenes and
    single-mesh GSO static scenes.
    """

    enabled: bool = False
    """Enable GT injection.  When false, the block is a no-op."""

    mode: str = "per_frame_meshes"
    """``"per_frame_meshes"``: per-frame mesh inversion (ActionMesh / OursActionBench
    layout ``{root}/{scene}/obj_NNN/mesh_NN.glb`` with NNN = obj_idx-1).  Per-frame
    ``raw_ss_modalities['shape']`` is populated.

    ``"global"``: single mesh per scene (GSO layout
    ``{root}/{scene}/meshes/model.obj``).  One inversion broadcast across
    all frames AND ``state.canonical_shape_coords[obj_idx]`` populated for
    Stage-2 canonical consumers.  When ``load_gt_poses=true`` also
    derives per-frame object poses from c2w (assumes static object at
    world origin in canonical orientation, EscherNet/GSO render setup).

    The ``per_frame_meshes`` mode additionally builds, from the temporal meshes
    (grounded on ``canonical_frame_idx``), the per-frame-voxel→canonical
    correspondence (``perframe_voxelization`` method) plus the
    per-canonical-mesh-vertex spatial-kNN Kabsch field.  ``"global"`` is a single
    static mesh, so no deformation is computed there.
    """

    gt_data_root: str = "data/davis_actionmesh"
    """Root directory containing per-scene GT geometry.  Layout depends on
    ``mode``:
      - ``per_frame_meshes``:      ``{root}/{scene}/{scene_subdir}/obj_NNN/mesh_NN.glb``
      - ``global``:                ``{root}/{scene}/meshes/model.obj``"""

    scene_subdir: str = ""
    """Extra path segment between ``{scene}`` and ``obj_NNN/`` in
    ``per_frame_meshes`` mode (empty = meshes at the scene root, which is the
    ActionMesh / OursActionBench layout).  Set it for scenes that
    keep the meshes in a named subdir beside their frames — ``dyncustom`` stages
    them at ``{scene}/actionmesh/``.  Ignored in ``global`` mode."""

    num_inversion_steps: int = 100
    """Number of Adam steps for per-mesh shape token inversion (passed to
    ``mesh_to_shape_tokens``).  100 is typically sufficient for IoU > 0.9."""

    # Inversion hyperparameters below are tuned to match the natural
    # Stage-1 shape-token distribution (a property of the SAM3D backbone,
    # not the dataset), so the same defaults apply to per-frame
    # ActionMesh and global GSO inversion alike.

    prior_weight: float = 1.0
    """Weight on the VAE prior moment-match penalty applied during decoder
    inversion: ``prior_weight * (mean(z)² + (var(z) - prior_target_var)²)``.
    Drives the empirical mean → 0 and variance → ``prior_target_var``,
    bounding the magnitude of the optimised shape tokens (so they stay in
    the SAM3D backbone's training distribution).  Increase if individual
    entries of the inverted latent grow too large; decrease if it hurts
    reconstruction IoU."""

    prior_target_var: float = 0.9
    """Target variance for the moment-match prior.  Matches the empirical
    Stage-1 shape-token variance (~0.9).
    Set to ``1.0`` to match the canonical VAE prior ``N(0, I)`` exactly.
    Only consumed when ``prior_weight > 0``."""

    l2_weight: float = 0.0
    """Weight on a per-element L2 penalty ``mean(z²)`` during decoder
    inversion.  Note: with ``prior_weight > 0`` already pinning
    ``var(z) ≈ prior_target_var`` (and ``mean(z²) ≈ var + mean²``), this
    term is essentially a constant offset and contributes ~zero gradient
    — leave at 0.0 unless you disable the moment-match prior."""


    inversion_precision: str = "auto"
    """Autocast dtype for the decoder-inversion forward/backward:
    ``"auto"`` | ``"fp32"`` | ``"bf16"`` | ``"fp16"``.

    ``auto`` (default) = **bf16 on Ampere or newer, fp32 elsewhere** — because
    bf16 is a speedup only where the tensor cores support it; pre-Ampere GPUs
    emulate bf16 and run markedly SLOWER than fp32.  Round-trip quality is
    essentially unchanged; hardware support is the trade-off.

    Reproducibility is not a reason to pin fp32: the GPU inversion is not
    run-to-run reproducible even in fp32 (100 Adam steps amplify float-level
    conv nondeterminism)."""

    inversion_chunk_size: int = 16
    """Frames inverted per GPU batch in ``mode="per_frame_meshes"``.  The
    dense ``ss_decoder`` backward holds ~1.0 GB of activations per frame
    (fp32), so this bounds peak VRAM while the SAM3D models are resident;
    an OOM halves it and retries.  Results are chunk-INVARIANT — every
    frame starts from the same seeded z0 and the per-frame losses are
    independent — so this is a pure memory lever.  ``0`` = all frames in
    one batch.  Batching is only a modest speed win (the decoder already
    saturates the GPU at B=1)."""

    gt_mesh_dilate: int = 0
    """Optional morphological dilation in voxels applied to the voxelised
    GT mesh before writing canonical_shape_coords.  Only used in
    ``mode="global"``.  Set to 1-2 if voxel count is suspiciously low."""

    load_gt_poses: bool = False
    """Also populate per-frame ``rotation``/``translation``/``scale``
    (and the SSI-roundtripped raw modalities) from GT cameras.  On GSO (either
    mode) they come from each frame's c2w, assuming a static object at world
    origin (GSO render setup); on OursActionBench (``mode="per_frame_meshes"``,
    shipped ``true`` in ``oursactionbench.yaml``) from the per-scene fitted
    cameras in ``camera.json`` and the per-frame mesh vertices.  Raises on
    ``dataset=davis_actionmesh`` in ``mode="per_frame_meshes"``, since
    ActionMesh has no GT camera/object poses."""

    canonical_frame_idx: int = 0
    """Frame that **grounds** the per-object deformation field: its mesh
    defines the canonical voxel grid, and all per-frame motion + voxel↔canonical
    correspondences are measured against this rest pose.  Indexes the temporal
    ``FrameKey.frame`` axis of ``tokens_list``; default 0 = view 0, frame 0.
    Override per-scene when frame 0 is not near rest pose.  Consumed whenever
    the deformation field is built (i.e. any per-frame ``mode``)."""

    k_kabsch: int = 8
    """kNN neighbourhood size for per-vertex Kabsch in the per-canonical-mesh-vertex
    correspondence provider (``compute_canonical_mesh_correspondence``).  For each
    canonical vertex, the local rigid rotation R_v[i] is fit to the k_kabsch
    spatially-nearest canonical neighbours' frame-i positions.  Default 8 always
    gives ≥3 non-collinear neighbours on non-degenerate meshes."""

    subdivide_passes: int = 0
    """Apply N passes of 1-to-4 midpoint triangle subdivision to the GT meshes
    before fitting the per-vertex/per-voxel deformation field.  Reduces the
    per-vertex IDW+Kabsch local-rigid approximation error on coarse meshes
    (residual is O(edge²); each pass roughly halves the mean barycenter L2 of
    the warp).  Memory grows ~4× per pass (V and F quadruple), so N=2 is the
    practical sweet spot for coarse meshes.  Default 0 (no subdivision); the
    oursactionbench config enables it for its coarse meshes — dense meshes
    don't need it."""



    perframe_voxelization: str = "surface"
    """How the per-frame GT voxel grids + per-frame-voxel→canonical
    correspondence (``state.gt_perframe_voxel_correspondence``, consumed by
    ``appearance_init=canonical_unified``) are discretised from the
    fixed-topology temporal meshes.  Both methods derive correspondence from the
    shared vertex topology — only the grid discretisation differs.

    ``"surface"`` (default): open3d surface voxelisation of each mesh → dense
    grid matching ``gt_canonical_shape_coords``; map each voxel to canonical via
    its nearest frame-i vertex (cKDTree for the grazing voxels no vertex lands
    in).  Robust across all meshes (coarse GSO/OAB included).

    ``"vertex"``: bin a fixed area-adaptive **barycentric lattice** over the
    triangles (density set by ``perframe_voxel_sample_edge``) → sparser grid with
    exact, cKDTree-free correspondence (the same barycentric samples are the same
    surface point across frames).  Opt-in for dense meshes; on coarse meshes the
    grid undercovers the surface voxelisation (see ``perframe_voxel_sample_edge``)."""

    perframe_voxel_sample_edge: float = 0.5
    """Barycentric-lattice target spacing in **voxel units** for
    ``perframe_voxelization="vertex"`` (ignored for ``"surface"``).  Each triangle
    is subdivided to ``ceil(edge_voxels / sample_edge)`` so samples sit ~this many
    voxels apart; ``0.5`` recovers ~87–96% of the surface-voxelised cells at
    ≤~290k transient samples.  Larger → sparser (``→∞`` ⇒ pure mesh-vertex
    binning, which catastrophically undercovers coarse meshes)."""

    local_rotation: Any = "identity"
    """3D rotation applied to GT geometry at load time (and matched
    analytically on each per-frame fitted OursActionBench c2w so the GT pose
    stays consistent).  Accepts a preset name, a list of preset names
    (composed left-to-right, first applied first), or a 3x3 matrix.

    - ``mode="per_frame_meshes"`` (ActionMesh / OursActionBench): rotation passed
      to :func:`gt_data._load_and_orient_mesh` for every per-frame
      ``.glb`` (and matched on each OAB fitted c2w so the GT object pose stays
      consistent).  ``rot_x_+90`` rotates Z-up → Y-up.
    - ``mode="global"`` (GSO): rotation applied to the single mesh.

    Preset name from
    :data:`genia.core.utils.gt_data._ACTIONBENCH_ROTATION_PRESETS` —
    ``identity`` (default), ``flip_x_180``, ``flip_y_180``, ``flip_z_180``,
    ``rot_{x,y,z}_{+,-}90``."""

    # Per-block output flags (None = inherit from global OutputConfig)
    save_renders: Optional[bool] = None
    save_metrics: Optional[bool] = None
    save_cache: Optional[bool] = None

    exit_after: bool = False
    """Exit pipeline after this block (useful for debugging the inversion)."""

# =====================================================================
# Pose initialization hierarchy  (config group: shape_and_poses_init/)
# =====================================================================

@dataclass
class ShapeAndPosesInitConfig:
    """The Stage-1 ODE blocks ``shape_init`` / ``pose_init`` (config group
    ``shape_and_poses_init/``): every frame's shape and pose tokens denoised in one
    batched ODE solve.  ``shape_init`` settles the shape (its pose is kept only as a
    fallback); ``pose_init`` then denoises the pose along ``gt_shape_trajectory``.
    """

    enabled: bool = True
    """Run the block.  ``False`` skips it (``strategy='none'`` is rejected)."""

    strategy: str = "parallel"
    """The only strategy: ``'parallel'`` (all frames in one batched ODE solve)."""

    inference_steps: Optional[int] = None
    """Override the Stage-1 ODE step count.  ``None`` uses the ss_generator's
    built-in default (25)."""

    gt_shape_trajectory: bool = False
    """Drive the shape token along ``(1-t)*noise + t*clean`` at every ODE step, so
    only the pose modalities are denoised while the backbone still sees
    in-distribution noisy-shape context.  ``clean`` is each frame's shape in
    ``raw_ss_modalities['shape']`` (injected, or settled by ``shape_init``)."""

    shape_velocity_averaging: str = "none"
    """Cross-frame shape consensus: ``"none"`` or ``"entropy"`` (the views of each
    timestamp fuse their shape velocity, weighted by cross-attention entropy).
    Different timestamps stay independent."""

    entropy_alpha: float = 60.0
    """Softmax temperature of the entropy weighting."""

    entropy_layer: int = 9
    """Backbone block whose shape cross-attention the entropy is read from."""

    entropy_min_weight: float = 0.001
    """Floor on the entropy weights."""

    rotation_velocity_averaging: str = "none"
    """``"none"`` or ``"median"``: per view, replace each frame's rotation velocity by
    the median across that view's timestamps (robust to one badly-conditioned
    frame).  Disabled automatically when every frame has its own predicted shape."""

    pose_velocity_broadcast_per_frame: bool = False
    """Give every view of a timestamp the lowest-view element's pose velocity."""

    pose_flip_guard_deg: Optional[float] = None
    """Restore a frame's prior pose (``shape_init``'s) when this pass rotates it by more
    than this many degrees: a mode flip of a near-symmetric object, not a refinement.
    ``None`` disables the guard."""

    cfg_interval_pose: Optional[List[int]] = None
    """Classifier-free-guidance interval for the pose modalities; an empty list
    disables CFG on pose.  ``None`` keeps the model default."""

    mv_reference_only_ode: bool = False
    """Multi-view static data: run the ODE on the reference view only and derive the
    other views through the shared-world rebase (their ODE output is discarded
    anyway).  A cost knob; refuses itself, per object and with a printed reason,
    whenever the restriction would not be equivalent."""

    post_rotation_refit: str = "none"
    """Refit (translation, scale) AFTER the ODE from the rotation it produced:
    ``"none"``, ``"translation"``, ``"scale"`` or ``"both"``.  Projects the rotated
    canonical shape and matches the observed silhouette size, silhouette centre and
    visible depth (see ``core/utils/pose_refit.py``).  Requires a canonical
    shape at ODE exit; a per-frame run has none and the refit no-ops."""

    post_rotation_refit_mv_consensus: bool = True
    """Per timestamp, fuse the refit translation of the views into one world
    placement and their scales into one size (median).  Inert on mono data."""

    post_rotation_refit_disc_threshold: float = 0.0
    """Drop depth discontinuities (relative 4-neighbour z-gradient) from the pixels
    the refit's depth statistics are read from.  0 disables."""

    post_rotation_refit_erode_px: int = 0
    """Erode the object mask by this many pixels before reading depth statistics."""

    # Per-block output flags (None = inherit from global OutputConfig)
    save_renders: Optional[bool] = None
    save_metrics: Optional[bool] = None
    save_cache: Optional[bool] = None

    exit_after: bool = False
    """Exit pipeline after this block (useful for debugging)."""

    def __post_init__(self):
        if self.strategy in ("none", None):
            raise ValueError(
                "shape_and_poses_init.strategy='none' is not a valid strategy. "
                "Use enabled=False to skip the block."
            )

# =====================================================================
# Appearance initialization (APPEARANCE_INIT)
# =====================================================================

@dataclass
class AppearanceInitConfig:
    """Appearance initialization parameters (APPEARANCE_INIT).

    Re-runs Stage 2 Diffusion (SLAT sampling) to predict appearance features.

    Per-frame mode:   Re-run Stage 2 diffusion for each frame on that
                      frame's own per-frame voxel coordinates; overwrites
                      the per-frame SLATs.  Poses + Stage 1 shape tokens
                      left untouched.
    Canonical mode:   Runs Stage 2 once on a single canonical SLAT, fusing
                      the per-view (static) or per-frame (dynamic)
                      velocities at every ODE step.
    """

    enabled: bool = False
    """Enable appearance initialization block."""

    strategy: str = "perframe"
    """Stage-2 SLAT denoising strategy.

    * ``"perframe"``: independent per-frame SLAT denoising (one ODE per frame).
    * ``"canonical_unified"``: ONE strategy for static and dynamic scenes — the
      observation axis is views in the static case and frames in the dynamic
      one, fused by the same visibility-weighted rule at every ODE step.
      A **single-timestamp** scene (static MV / GSO / CO3D / mvcustom) runs
      one multi-view ODE over its views (``Inference.stage2_mv``) and needs no
      shape injection.  A **multi-timestamp** scene runs the consensus-canonical
      ODE (``Inference.stage2_dyn``): one canonical latent and one
      noise draw, per step gathered onto each frame's own GT grid through
      ``pf_to_canon``, evaluated there with that frame's attention bias and
      rendering guidance (applied to the per-frame velocity BEFORE fusion),
      collapsed back with ``collapse_perframe_to_canonical``, then a single
      Euler step on the canonical.  The per-frame states are re-gathered from
      the canonical every step, so they cannot drift.  Canonical rows no
      frame reaches take the mean fused velocity of the reached rows (a zero
      would leave them at their initial noise).  Dynamic path requires
      ``gt_perframe_voxel_correspondence`` AND a GT-anchored canonical grid
      (``reset_shape_to_gt=true``, since ``pf_to_canon`` indexes GT rows) and is
      single-object.  See
      ``core/configs/appearance_init/canonical_unified.yaml``."""

    visibility_min_weight: float = 0.001
    """Minimum per-view visibility weight after softmax."""

    fused_min_weight: float = 0.001
    """Minimum per-view fused weight (clamp after the visibility renormalisation)."""


    inference_steps: int = 25
    """Number of flow-matching steps for Stage 2 diffusion."""

    reset_shape_to_gt: bool = False
    """At block entry, drop the predicted shape tokens (canonical +
    per-frame ``raw_ss_modalities['shape']``) and reset the canonical
    voxel grid to the GT mesh voxelisation stored in
    ``state.canonical_shape_coords`` by ``GT_SHAPES_INVERSION``.  Forces
    Stage 2 to denoise appearance on the GT-anchored grid regardless of
    what the Stage-1 blocks produced.  Strategy-agnostic.
    Requires ``state.canonical_shape_coords`` populated for every
    target object (i.e. a ``GT_SHAPES_INVERSION`` mode that writes
    the canonical grid: ``gso``, or any per-frame mode such as
    ``actionmesh`` / ``oursactionbench``).  When actionmesh Φ/R are present,
    they are preserved (the reset uses ``invalidate_canonical`` with
    unchanged ``shape_coords``)."""

    consensus_frame_chunk: Optional[int] = None
    """``canonical_unified`` dynamic path only: how many frames share ONE
    batched backbone forward in the consensus ODE.

    The frames of a step are independent given the canonical latent, so they
    can ride one ragged sparse batch instead of N sequential calls: a step
    costs ``2*ceil(N/K)`` forwards rather than ``2*N``.

    ``1`` runs one frame per forward (the unbatched path);
    ``None`` puts every frame in one group.  Larger K is not free — the bias
    is right-padded to the group's longest frame — but attention stays
    block-diagonal, so K never changes the math, only float accumulation
    order and peak memory."""

    # Visibility weighting
    visibility_weighting: bool = False
    """Enable per-view visibility weighting for velocity averaging."""

    visibility_alpha: float = 30.0
    """Temperature for ``softmax(alpha * visibility)``.  Higher = sharper."""

    visibility_neighbor_tolerance: float = 4.0
    """DDA neighbor tolerance in voxel units.  Ignores occluders within this
    distance of the target to handle grazing-angle false positives."""

    # Cross-attention bias (independent of velocity-level visibility weighting)
    visibility_attn_bias: bool = False
    """Enable visibility-guided cross-attention bias.  Adds ``+alpha`` to
    attention logits for (voxel, patch) pairs where the voxel projects onto
    that image patch.  Applied per view, so it works with one view or with
    the N views of a multi-view conditioning set (each view's bias uses its
    own voxel projection)."""

    visibility_attn_bias_alpha: float = 5.0
    """Additive bias temperature.  0 = no effect; higher = stronger steering
    toward geometrically visible patches.  Typical range: 1-20."""

    visibility_attn_bias_layers: str = "all"
    """Which cross-attention layers to apply the bias to.
    ``'all'``: every block.  Comma-separated indices: ``'0,1,2'``,
    ``'-1,-2'`` (negative = from end).  Start with ``'all'``, then ablate."""

    visibility_attn_bias_cropped_image: bool = True
    """Apply visibility bias to cropped-object image patches [5:1374]."""

    visibility_attn_bias_full_image: bool = True
    """Apply visibility bias to full-scene image patches [1379:2748]."""

    visibility_attn_bias_cropped_mask: bool = True
    """Apply visibility bias to cropped mask patches [2753:4122]."""

    visibility_attn_bias_full_mask: bool = True
    """Apply visibility bias to full mask patches [4127:5496]."""

    visibility_attn_bias_compensate_passive_streams: str = "off"
    """Compensate passive (un-biased) streams for the partition-function shift
    induced by the +alpha bias on active streams:
    ``"off" | "approx"``.

    * ``"off"``: no compensation.  Passive streams' aggregate post-softmax
      share is silently suppressed when active streams get +alpha on
      visible patches.
    * ``"approx"`` (``-c`` on active): subtract a per-voxel
      scalar ``c_i = log(v_i · exp(α) + (1 − v_i))`` from every active-
      stream patch token (``v_i`` = fraction of active patches visible to
      voxel ``i``).  Under uniform pre-softmax logits, exactly preserves
      passive AND CLS/region tokens' post-softmax shares; visible:invisible
      steering ratio remains ``exp(α):1``.  Zero perf cost.

    Opt-in (default ``"off"``) — the no-compensation behavior matches the
    attention semantics SAM3D was trained with."""

    # ── Rendering-based velocity guidance (Stage-2 SLAT denoising) ────
    # At each ODE step, decodes the one-step Tweedie SLAT estimate to
    # Gaussians (gsplat), renders per frame, computes losses against GT
    # image/mask/depth, and subtracts the autograd gradient from the
    # velocity.  Off by default.
    rendering_guidance_active: bool = False
    """Master switch for in-ODE rendering guidance during SLAT denoising."""

    rendering_guidance_velocity_weight: float = 1.0
    """Strength multiplier applied to the autograd gradient before the
    velocity update: ``v -= velocity_weight * grad``."""

    rendering_guidance_active_from: float = 0.0
    """Skip rendering guidance for t below this threshold."""

    rendering_guidance_active_until: float = 1.0
    """Hard cutoff: disable rendering guidance for t above this threshold."""

    rendering_guidance_microbatch_size: int = 8
    """Microbatch size for the per-frame render loop.  ``0`` = single-pass
    (build the graph for all N frames, one backward).  ``k > 0`` = process
    frames in chunks of ``k`` with per-chunk backward + leaf-tensor gradient
    accumulation; the resulting gradient is identical to the single-pass
    one but caps activation memory at ``min(k, N)`` frames.  Default 8
    keeps activation memory bounded on many-view scenes (especially with
    LPIPS); raise (or set to 0) when N is small."""

    rendering_guidance_resolution_scale: int = 1
    """Integer downscale factor applied to GT mask / depth / RGB and the
    intrinsics ``K[:2]`` before the render loop.  ``1`` (default) =
    native resolution; ``2`` = render and compare at H//2 × W//2 (~4×
    activation memory reduction)."""

    rendering_guidance_decoder_autocast_bf16: bool = False
    """Run the in-guidance SLAT->Gaussian decoder forward under
    ``torch.autocast(bfloat16)``.  The Tweedie SLAT is decoded under grad at
    EVERY ODE step (re-decoded per frame-chunk), so decoder activations are
    the dominant guidance memory term — bf16 ~halves it.  Decoder outputs are
    cast back to fp32 so gsplat rendering, losses, and the autograd grad stay
    full-precision (see ``FinetuningConfig.decoder_autocast_bf16``)."""

    rendering_guidance_normalize_grad: bool = False
    """Rescale the rendering-loss gradient to ‖v‖ before the velocity update
    (``v -= velocity_weight · g·‖v‖/‖g‖``).  Makes ``rendering_guidance_velocity_weight`` an absolute
    *fraction of the backbone velocity* per step (calibrate to ~0.01–0.1),
    decoupled from the raw, t-varying grad magnitude — so the guidance stays
    subordinate to the backbone instead of front-loaded and dominant.  ``False``
    (default) = raw ``v -= velocity_weight · g``."""

    rendering_guidance_random_background: bool = False
    """Sample a fresh uniform RGB background per (ODE step, frame) for the
    render-guidance loss — both the render(s) and the GT compositing use the
    same colour, so the photometric loss can't bake a fixed background into the
    decoded appearance (mirrors FINETUNE's ``random_background``).  ``False`` =
    fixed pipeline background colour."""

    rendering_guidance_random_background_seed: Optional[int] = None
    """Master seed for ``rendering_guidance_random_background``.  ``None``
    (default) draws from the global RNG; an int derives the colour from
    ``hash((seed, step_idx, frame_idx))`` — reproducible across reruns yet still
    varying per (step, frame)."""

    rendering_guidance_visibility_detach: bool = False
    """Stop-gradient occluded Gaussians in the gaussian guidance branch,
    mirroring FINETUNE's ``token_grad_visibility_mask`` detach: at each active
    ODE step, flag Gaussians whose projected centre is depth-occluded in EVERY
    frame (``_flag_visible_by_depth`` OR-accumulated over frames, self-occlusion
    only) and detach their decoded attributes (shared
    ``_detach_invisible_gaussians`` recipe), so backward sends no gradient
    from unseen Gaussians into the SLAT velocity.  Costs one extra no-grad
    render per frame per active step.  Dataclass default off; the shipped
    ``canonical_unified`` / ``perframe_guided`` YAMLs enable it."""

    rendering_guidance_visibility_depth_margin: float = 0.02
    """Relative depth tolerance for ``rendering_guidance_visibility_detach``:
    a Gaussian is occluded in a frame when ``z > z_surface * (1 + margin)`` at
    its projected pixel (twin of ``FinetuningConfig.visibility_depth_margin``)."""

    rendering_guidance_losses: LossConfig = field(default_factory=LossConfig)
    """Loss weights for rendering guidance (only loss-weight fields are read:
    ``rgb_weight``, ``silhouette_*``, ``depth_*``, ``normals_*``,
    ``perceptual_*``, ``rgb_multiscale*``, ``occlusion_robust_silhouette``).
    Iteration / LR fields are ignored — guidance is classifier-style, not an
    inner optimisation loop."""

    save_ode_steps_viz: bool = True
    """Capture per-ODE-step SLAT snapshots inside ``stage2_mv`` and write
    a per-ODE-step viz: per-step
    Gaussian renders for every frame, stitched into a row per step and
    stacked into one grid PNG.  Honours actionmesh Φ/R when active.  Gated by the
    block's RESOLVED render flag (``get_block_output_flag(..., "save_renders")``,
    so ``output.suppress_intermediate_renders`` and a per-block override both
    apply) — runs that disable renders pay no cost.  Adds ~L·8·4 B
    per ODE step in CPU memory and one decode + N gsplat renders per
    captured step."""

    # Per-block output flags (None = inherit from global OutputConfig)
    save_renders: Optional[bool] = None
    save_metrics: Optional[bool] = None
    save_cache: Optional[bool] = None

    exit_after: bool = False
    """Exit pipeline after this block (useful for debugging)."""

    def __post_init__(self):
        if self.visibility_attn_bias_compensate_passive_streams not in ("off", "approx"):
            raise ValueError(
                "visibility_attn_bias_compensate_passive_streams must be 'off' or "
                f"'approx', got {self.visibility_attn_bias_compensate_passive_streams!r}"
            )


# =====================================================================
# Token fine-tuning (FINETUNE)
# =====================================================================

@dataclass
class FinetuningConfig:
    """Token fine-tuning parameters."""

    enabled: bool = False
    """Enable token fine-tuning."""

    strategy: str = "canonical"
    """Which SLAT the FINETUNE block optimises.

    - ``'canonical'``: one shared canonical token set per object, supervised
      against every frame's render (joint canonical-token + LoRA optimisation).
    - ``'perframe'``: each frame's OWN SLAT, supervised against that frame
      alone.  For runs with no canonical object (``has_canonical=False`` — a
      dynamic sequence reconstructed from SAM3D's per-frame shapes), where the
      canonical strategy has nothing to optimise.  ``num_iterations`` is then
      spent per frame, so wall-clock scales with the frame count."""

    num_iterations: int = 100
    """Number of fine-tuning iterations."""

    token_lr: float = 0.0001
    """Learning rate for SLAT token features."""

    optimize_tokens: bool = True
    """Gate token optimization.  When ``False``, SLAT token features are
    frozen (no gradient, no Adam param group) regardless of ``token_lr``."""

    batch_size: int = 0
    """Frames sampled per iteration (0 = all frames)."""

    microbatch_size: int = 0
    """Micro-batch size for gradient accumulation (0 = disabled).
    When > 0 and batch_size > microbatch_size, frames are processed in
    micro-batches with backward() per micro-batch, freeing rendering+VGG
    graphs between micro-batches to reduce peak GPU memory."""

    decoder_autocast_bf16: bool = True
    """Run the in-loop SLAT->Gaussian decoder forward under
    ``torch.autocast(bfloat16)``.  Decoder activations are held in bf16 for
    the backward pass (~halves the dominant peak-memory term); decoder outputs
    are cast back to fp32 so gsplat rendering, losses, and optimizer states
    keep full precision.  Set False for a pure-fp32 decode."""

    decoder_checkpoint: bool = False
    """Gradient-checkpoint the per-object SLAT->Gaussian decoder torso during
    the FINETUNE optimization loop.  The under-grad decode of a very large
    object (>~1.5M Gaussians, e.g. indoor image scenes) retains the full
    transformer-torso activation graph and can OOM a 40 GB GPU even with
    ``decoder_autocast_bf16``; checkpointing frees that retained graph.
    no_grad decodes (initial/best/export) are ~unaffected."""

    decoder_checkpoint_min_voxels: int = 0
    """Only checkpoint objects whose canonical SLAT has at least this many
    voxels (tokens).  ``0`` (default) checkpoints every object when
    ``decoder_checkpoint`` is on.  Checkpointing costs ~2x decode compute, so
    small objects that already fit pay it for nothing — set a threshold (e.g.
    30000) to restrict the slow path to the few giant objects that actually
    OOM.  No effect when
    ``decoder_checkpoint`` is off."""

    # LoRA decoder adaptation
    lora_decoder: bool = True
    """Apply LoRA adapters to a per-object decoder copy."""

    optimize_decoder: bool = True
    """Gate LoRA decoder optimization.  When ``False``, LoRA adapters are
    not set up and no decoder parameters are optimized (equivalent to
    skipping ``lora_decoder``).  Combined with ``lora_decoder`` via AND."""

    lora_rank: int = 2
    """LoRA rank (low-rank dimension)."""

    lora_lr: float = 0.001
    """Learning rate for LoRA A/B parameters."""

    lora_alpha: float = 0.5
    """LoRA scaling factor.  Combined with ``lora_rank`` via either
    ``alpha / rank`` (classic LoRA) or ``alpha / √rank`` (rsLoRA, on by
    default — see ``lora_rs_scaling``)."""

    lora_rs_scaling: bool = True
    """Use rank-stable LoRA scaling ``alpha / √rank`` instead of the
    classic ``alpha / rank`` — *Kalajdzievski 2023* (arXiv:2312.03732).

    The classic formula shrinks the effective adapter delta as ``rank``
    grows, so ``lora_lr`` has to be retuned every time ``lora_rank``
    changes.  rsLoRA decouples them, so increasing rank only adds
    capacity without quietly damping the update magnitude.  Bit-equivalent
    to classic LoRA at ``rank=1``; ~1.4× larger delta at ``rank=2``;
    ~2× at ``rank=4``."""

    lora_targets: str = "all"
    """Which ``nn.Linear`` layers in the decoder are wrapped with LoRA.

    * ``"all"`` (default) — every ``nn.Linear``.  At low rank the FFN +
      adaLN modulation linears contribute on this decoder, so wrapping them
      beats ``"attn"``.
    * ``"attn"`` — only the linears inside the attention modules:
      ``to_q``, ``to_k``, ``to_v``, ``to_qkv``, ``to_kv``, ``to_out``.
      Smaller adapter (~110k params at rank=2 vs ~620k for ``"all"``).
    * ``"qkv"`` — same as ``"attn"`` minus ``to_out``.  Hu et al.'s
      original "Q + V" recipe in our fused-projection codebase
      collapses to "projections that produce Q/K/V", since ``to_qkv``
      and ``to_kv`` are fused so K can't be split out without surgery.

    Matched against the *leaf* name of every ``nn.Linear`` module
    (i.e. the last segment of its dotted ``named_modules`` path)."""

    lora_init: str = "default"
    """Initialisation strategy for the LoRA ``A`` matrix (``B`` is always
    zero).  Options:

    * ``"default"`` (default) — ``A ~ N(0, 1/rank²)``.
      Has noticeably more init magnitude than Kaiming at low ranks and
      in turn helps the adapter ramp up faster within the modest number
      of FINETUNE iterations.  Variance shrinks as rank grows, so this
      collapses at high rank — switch to ``"kaiming"`` once ``lora_rank``
      is raised.
    * ``"kaiming"`` — ``kaiming_uniform_(A, a=√5)``; matches the reference
      LoRA implementation in *Hu et al. 2021* (arXiv:2106.09685) and
      HuggingFace PEFT.  Variance is decoupled from ``rank``, but at
      ``rank=2`` the init magnitude is much smaller than ``"default"``
      and the adapter trains more slowly under our current step count.

    Switch to ``"kaiming"`` if you raise ``lora_rank`` past 2."""

    use_dora: bool = True
    """Use DoRA (Weight-Decomposed Low-Rank Adaptation) instead of plain
    LoRA — *Liu et al. 2024* (ICML Oral, arXiv:2402.09353).

    Each wrapped layer gains a per-output ``magnitude`` Parameter
    initialised to ``‖W‖_row``; LoRA's low-rank delta then updates the
    *direction* while ``magnitude`` carries the *scale*.  Forward becomes
    ``y = (m / ‖W + ΔV‖_row) · (x @ (W + ΔV)^T)`` with the norm detached
    (PEFT default), so gradient on ``A`` / ``B`` only updates direction.

    Identity at iter-0 (``B = 0`` ⇒ ``ΔV = 0`` ⇒ ``m / ‖W‖_row = 1``).
    Cost over plain LoRA: one ``out_f``-vector per layer + a row-norm
    computation per forward.  Pairs well with ``lora_init="kaiming"``
    and higher ``lora_rank``."""

    lora_weight_decay: float = 0.0
    """L2 regularization for LoRA parameters.  Default ``0`` follows the
    standard LoRA recipe (Hu et al. 2021): the zero-init ``B`` matrix
    already pins the adapter to the no-op anchor at iter-0, and AdamW WD
    on both ``A`` and ``B`` works against that prior.  Set ``>0`` only
    when you observe LoRA params growing without bound."""

    gauss_rotation_anchor_weight: float = 0.0
    """Soft L2 pull on the canonical Gaussian's raw ``_rotation`` toward
    its FINETUNE iter-0 value: ``weight · mean‖_rotation - _rotation_init‖²``.

    Per-iter (not per-frame), so the weight is dimensionally comparable
    across batch sizes — independent of ``batch_size`` / ``n_frames``.
    ``0.0`` (default) leaves rotation free to move with the photometric
    loss.  Larger values dampen rotation drift;
    very large values approximate a hard freeze.

    Useful when the per-frame deformation field already supplies the
    geometric signal (actionmesh) and the photometric loss should not
    repurpose Gaussian rotations to compensate."""

    # View-dependent SH colors
    sh_degree: Optional[int] = None
    """SH degree for per-frame appearance (None=disabled, 0=DC offsets only, 1-3=DC+SH)."""

    sh_lr: float = 0.01
    """Learning rate for DC offsets (degree 0)."""

    sh_lr_rest: float = 0.001
    """Learning rate for SH rest coefficients (degree 1+)."""

    sh_reg_weight: float = 0.01
    """L2 penalty on SH rest coefficients (toward zero).
    Prevents higher-order SH bands from absorbing base/DC color."""

    sh_consistency_weight: float = 0.01
    """L2 penalty on each frame's SH deviation from cross-frame mean.
    Encourages consistent view-dependent appearance across frames."""

    # Regularization
    token_drift_weight: float = 0.001
    """L2 penalty pulling token features toward their original values."""

    # Visibility gradient masking (upstream / stop-gradient form)
    token_grad_visibility_mask: bool = False
    """Stop-gradient the decoded Gaussians occluded from every batch view of the
    current iteration (back-side / self-occluded), BEFORE the render — so its
    backward sends no gradient through them, to neither the SLAT tokens NOR the
    shared (LoRA'd) decoder.  Supervision then reaches only Gaussians visible from
    some train view, leaving the unseen back at its appearance-init prior.
    Per-Gaussian (depth-occlusion test, see ``visibility_depth_margin``):
    an occluded Gaussian stops contributing gradient but still *follows* its token
    if visible siblings move it, so only its own wrong signal is removed.
    Detaching the invisible Gaussian attributes — vs. post-hoc zeroing of
    ``opt_feats.grad`` — also blocks the unseen-region leak into the shared
    decoder, and needs no token-contiguity assumption.  Self-occlusion only
    (target object; each frame warped by the deformation field when one is
    loaded).  Off by default."""

    visibility_depth_margin: float = 0.02
    """Relative depth tolerance of the train-view visibility test (shared by
    ``token_grad_visibility_mask`` and ``invisible_opacity_anchor_weight``): a
    Gaussian is occluded when ``z_gaussian > z_surface * (1 + visibility_depth_margin)``
    at its projected pixel (only where a surface is rendered).  A fraction of the
    surface depth, so it is scene-scale invariant."""

    invisible_opacity_anchor_weight: float = 0.0
    """Soft anchor that resists FINETUNE driving down the opacity of Gaussians
    not visible from any train view (back-side / occluded) — otherwise the
    shared LoRA/decoder update (gradient-fed by the visible Gaussians) hollows
    them out and the held-out novel views go transparent.  Train-view visibility
    is computed ONCE at finetune start (depth occlusion test, over all train
    frames); each
    iteration the invisible Gaussians pay ``weight * mean(relu(opacity_init -
    opacity_cur)**2)`` — one-sided, so only opacity DROPS below the appearance-
    init value are penalised (legitimate increases are free).  The loss
    backprops through the frozen decoder to the tokens + LoRA, so it counters the
    collapse at its source.  0 = off.  Complementary to
    ``token_grad_visibility_mask``: the mask stop-gradients the invisible
    Gaussians (freezing them at the prior), whereas the anchor actively pushes
    their opacity back up — relevant for invisible Gaussians that still drift
    because a visible token-sibling moves them."""

    # Perceptual computation
    perceptual_scale: float = 0.5
    """Resolution scale for LPIPS/DINO computation (0.5=half, 1.0=full)."""

    # Joint pose refinement during finetuning
    refine_poses: bool = True
    """Jointly optimize per-frame poses alongside token features."""

    correction_scale_control: str = "shared"
    """``"shared"`` (default) | ``"perframe"`` -- which transform owns SCALE under
    ``correction_granularity: both``.

    ``shared`` freezes the NATIVE scale, so the one shared correction is all that sets the
    object's size.  ``both`` is gauge-ambiguous by construction and that is tolerated for
    rotation and translation (the composite is what gets written back, so the split never
    leaves the solver); scale is the case where it should not be, because the object has
    ONE size and a per-frame transform modelling it lets the optimiser breathe the object
    frame-to-frame while the correction chases the residual.  ``perframe`` leaves the
    per-frame scale free.

    INERT outside ``both`` -- under ``per_frame`` there is no shared transform to hold the
    scale, and under ``shared`` the natives are frozen already.  Read it through
    :func:`resolve_correction_scale_control`.

    Note: NOT expressible with ``losses.lr_scale``: that drives the native AND the
    correction's scale at every call site, so zeroing it freezes both."""

    correction_granularity: str = "per_frame"
    """``"per_frame"`` | ``"shared"`` | ``"both"`` -- the GRANULARITY of the pose
    correction.  ``shared`` is ONE Sim(3) per object, in the object's CANONICAL frame,
    composed onto every frame's pose, so per-frame root MOTION survives; ``both`` leaves
    the natives free alongside it (an optimiser device, gauge-ambiguous -- read the
    warning there).  Same meaning, same three values and the same composer as every
    refine solver -- see :class:`GlobalPoseRefineConfig.correction_granularity`.

    .. note::
       The gauge is entirely OPEN here: FINETUNE has no pose regulariser at all
       (``regularization_weight`` is not read on this path), so ``both``'s split is
       decided purely by the optimiser.  Harmless because the correction is ABSORBED into
       the poses at snapshot time (:func:`snapshot_poses_with_correction`).

    On a DYNAMIC scene ``shared`` is the canonical-frame correction that propagates to
    every frame, fit from EVERY frame; on a static one it is simply a shared pose
    correction."""

    random_background: bool = True
    """Randomize the rendering background color (uniform RGB in [0,1]) for
    every per-frame render in the foreground-only path. The same color is
    used for both the rasterizer background and the GT image masking inside
    ``_compute_frame_loss``, so foreground gradients are unchanged while the
    background color the model is implicitly asked to reproduce changes every
    rendering — discouraging Gaussians from baking the background color into
    their appearance."""

    random_background_seed: Optional[int] = None
    """Master seed for ``random_background``.  When ``None`` (default), the
    background colour is drawn from the global torch RNG (per-run state,
    not reproducible across reruns).  When set to an int, each render's
    BG colour is derived deterministically from
    ``hash((random_background_seed, iter_idx, frame_idx))`` — bit-identical
    across reruns of the same config while still varying per (iter, frame)
    so the augmentation intent is preserved.  Set this for reproducible
    runs."""

    lr_warmup_steps: int = 10
    """Linear LR warmup over the first ``lr_warmup_steps`` iterations
    (0 = disabled).  Multiplier ``min(step / warmup, 1)`` applied to every
    param group.  Smooths AdamW's first
    ~50 steps where moment estimates are unstable, especially with the
    zero-init LoRA ``B`` matrix and small ``token_lr``."""

    losses: LossConfig = field(default_factory=LossConfig)
    """Loss config for the fine-tuning phase (inline in finetuning/canonical.yaml)."""

    # Per-block output flags (None = inherit from global OutputConfig)
    save_renders: Optional[bool] = None
    save_metrics: Optional[bool] = None
    save_cache: Optional[bool] = None

    exit_after: bool = False
    """Exit pipeline after this block (useful for debugging)."""

    _VALID_CORRECTION_GRANULARITY = ("per_frame", "shared", "both")

    def __post_init__(self):
        if self.correction_granularity not in self._VALID_CORRECTION_GRANULARITY:
            raise ValueError(
                f"FinetuningConfig.correction_granularity must be one of "
                f"{self._VALID_CORRECTION_GRANULARITY}, got {self.correction_granularity!r}")

# =====================================================================
# Our sections
# =====================================================================

@dataclass
class GeniaConfig:
    """The config sections of our blocks (``core/manifest.py`` names each block's section)."""

    gt_shapes_inversion: GtShapesInversionConfig = field(default_factory=GtShapesInversionConfig)
    # The Stage-1 split: shape_init settles the shape (parallel_shape), then
    # pose_init denoises the pose against it (parallel_pose).  Both use the
    # shape_and_poses_init config group.
    shape_init: Any = field(
        default_factory=lambda: ShapeAndPosesInitConfig(enabled=False),
    )
    pose_init: Any = field(
        default_factory=lambda: ShapeAndPosesInitConfig(enabled=False),
    )
    appearance_init: AppearanceInitConfig = field(default_factory=AppearanceInitConfig)
    appearance_init_2: AppearanceInitConfig = field(default_factory=AppearanceInitConfig)
    global_pose_refine_1: GlobalPoseRefineConfig = field(default_factory=GlobalPoseRefineConfig)
    finetuning: FinetuningConfig = field(default_factory=FinetuningConfig)
