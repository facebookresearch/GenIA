# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import sys

from genia.core.paths import SAM3D_OBJECTS_ROOT

# Make the vendored sam3d_objects importable (appended, never prepended).
if str(SAM3D_OBJECTS_ROOT) not in sys.path:
    sys.path.append(str(SAM3D_OBJECTS_ROOT))

os.environ["LIDRA_SKIP_INIT"] = "true"
from typing import Union, Optional, List, Callable
import numpy as np
from PIL import Image
from omegaconf import OmegaConf, DictConfig, ListConfig
from hydra.utils import instantiate, get_method
import torch
import torch.nn.functional as F
import shutil
import subprocess
import builtins

import sam3d_objects  # noqa: F401  (initialises the vendored package; do not remove)
from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap

__all__ = ["Inference"]

WHITELIST_FILTERS = [
    lambda target: target.split(".", 1)[0] in {"sam3d_objects", "torch", "torchvision", "moge"},
]

BLACKLIST_FILTERS = [
    lambda target: get_method(target)
    in {
        builtins.exec,
        builtins.eval,
        builtins.__import__,
        os.kill,
        os.system,
        os.putenv,
        os.remove,
        os.removedirs,
        os.rmdir,
        os.fchdir,
        os.setuid,
        os.fork,
        os.forkpty,
        os.killpg,
        os.rename,
        os.renames,
        os.truncate,
        os.replace,
        os.unlink,
        os.fchmod,
        os.fchown,
        os.chmod,
        os.chown,
        os.chroot,
        os.fchdir,
        os.lchown,
        os.getcwd,
        os.chdir,
        shutil.rmtree,
        shutil.move,
        shutil.chown,
        subprocess.Popen,
        builtins.help,
    },
]


def _install_vis_bias_hooks(
    wrapper, frames, visibility_pixel_coords, L, N, *,
    alpha, layers, streams, compensate_passive_streams,
    debug, verbose=True,
):
    """Build per-observation voxel->patch masks and register the bias hooks.

    Shared by ``stage2_mv`` (N views on one canonical grid) and
    ``stage2_dyn`` (a GROUP of frames, each on its own deformed grid).
    Returns ``hooks``; the caller owns removal.  The hooked layers share one
    per-call bias cache (``shared_state``), built fresh here — it MUST NOT be
    reused across observations with different grids, or the first
    observation's bias is silently served to all of them.

    ``L`` is either one int (every observation on the SAME grid: stage2_mv's N
    views, or a single frame) or a per-observation sequence of N ints (a
    stage2_dyn group, whose frames have DIFFERENT voxel counts).  The sequence
    form is forwarded to the hook as ``per_view_L``, which switches it to the
    ragged bias; the int form leaves the rectangular path untouched.
    """
    from genia.core.visibility_attn import (
        SparseVisibilityBiasHook, build_voxel_patch_mask,
        parse_layer_selection, transform_pixels_to_518,
        transform_pixels_to_518_full_image,
    )

    # int -> one shared grid (rectangular); sequence -> one grid per
    # observation (ragged).  `_L_of` is the only place the two differ below.
    _L_list = [int(x) for x in L] if isinstance(L, (list, tuple)) else None
    if _L_list is not None and len(_L_list) != N:
        raise ValueError(f"L has {len(_L_list)} entries but N={N}")
    _L_of = (lambda i: _L_list[i]) if _L_list is not None else (lambda i: L)

    voxel_patch_masks = []
    voxel_patch_masks_full = []
    for _vi in range(N):
        _raw_pc = visibility_pixel_coords[_vi]
        _n_with_pixels = sum(1 for v in _raw_pc.values() if len(v) > 0)
        pixel_coords_518 = transform_pixels_to_518(
            _raw_pc, frames[_vi]["mask"],
        )
        _n_in_bounds = 0
        _n_out_bounds = 0
        for _vid, _px in pixel_coords_518.items():
            if len(_px) == 0:
                continue
            _r, _c = _px[:, 0], _px[:, 1]
            _valid = (_r >= 0) & (_r < 518) & (_c >= 0) & (_c < 518)
            _n_in_bounds += int(_valid.sum())
            _n_out_bounds += int((~_valid).sum())

        vpm = build_voxel_patch_mask(pixel_coords_518, _L_of(_vi))
        voxel_patch_masks.append(vpm)

        _H_orig, _W_orig = frames[_vi]["mask"].shape[:2]
        pixel_coords_518_full = transform_pixels_to_518_full_image(
            _raw_pc, _H_orig, _W_orig,
        )
        vpm_full = build_voxel_patch_mask(pixel_coords_518_full, _L_of(_vi))
        voxel_patch_masks_full.append(vpm_full)

        if verbose:
            _n_visible = int(vpm.any(dim=1).sum())
            _n_visible_full = int(vpm_full.any(dim=1).sum())
            if N > 1:
                print(f"    View {_vi}: {_n_with_pixels} voxels with pixels, "
                      f"cropped={_n_visible}/{_L_of(_vi)}, "
                      f"full={_n_visible_full}/{_L_of(_vi)}")
            else:
                print(f"    Pixel coords: {_n_with_pixels} voxels with pixels, "
                      f"{_n_in_bounds} pixels in-bounds, "
                      f"{_n_out_bounds} out-of-bounds after 518 transform")

    blocks = wrapper.blocks
    layer_indices = parse_layer_selection(layers, len(blocks))
    # Single shared bias cache across all hooked layers — the bias only
    # depends on masks/N/alpha (constant across layers), so 24 identical
    # (N,1,L_down,P_total) tensors would be wasted GPU memory.
    shared_state: dict = {}
    hooks = []
    for li in layer_indices:
        hooks.append(SparseVisibilityBiasHook(
            blocks[li].cross_attn,
            voxel_patch_masks,
            alpha=alpha,
            N=N, L=_L_of(0), per_view_L=_L_list,
            voxel_patch_mask_full=voxel_patch_masks_full,
            stream_enables=streams,
            compensate_passive_streams=compensate_passive_streams,
            debug=debug,
            shared_state=shared_state,
        ))
    if verbose:
        _agg_L = sum(_L_list) if _L_list is not None else _L_of(0)
        if _L_list is not None:
            _agg_visible = sum(int(m.any(dim=1).sum())
                               for m in voxel_patch_masks)
            _agg_visible_full = sum(int(m.any(dim=1).sum())
                                    for m in voxel_patch_masks_full)
        else:
            _agg_visible = int(voxel_patch_masks[0].any(dim=1).sum())
            _agg_visible_full = int(voxel_patch_masks_full[0].any(dim=1).sum())
        _agg_occluded = _agg_L - _agg_visible
        print(f"    Visibility attn bias: alpha={alpha}, "
              f"layers={layer_indices}, "
              f"cropped={_agg_visible}/{_agg_L}, "
              f"full={_agg_visible_full}/{_agg_L} voxels "
              f"({_agg_occluded} occluded, unbiased fallback)"
              f"{f', {N} per-view masks' if N > 1 else ''}")
    return hooks


def _frame_groups(n: int, chunk: "int | None") -> "list[list[int]]":
    """Partition ``range(n)`` into contiguous ascending groups of <= ``chunk``.

    One group is one batched backbone forward in ``stage2_dyn``.  ``chunk=None``
    (or >= n) gives a single group; ``chunk=1`` gives n singletons (one forward
    per frame).

    Contiguity is load-bearing twice over: it lets the group's conditioning be a
    VIEW of ``cond_embedded`` rather than a gather, and it keeps each group's
    membership fixed for the whole solve — the attention-bias hook builds its
    bias lazily on the first forward and caches it, so a group whose membership
    drifted between steps would be served a stale bias.
    """
    if n <= 0:
        return []
    if chunk is None or chunk <= 0 or chunk >= n:
        return [list(range(n))]
    return [list(range(s, min(s + chunk, n))) for s in range(0, n, chunk)]


def _ragged_sparse_batch(coords_per_frame):
    """Concatenate per-frame ``(L_i, 4)`` voxel coords into ONE ragged batch.

    Column 0 is renumbered to the frame's position WITHIN the batch, ascending
    and contiguous.  That is the ``SparseTensor`` invariant: its ``layout`` is
    derived purely from ``bincount(coords[:, 0]).cumsum()``
    (``sparse/basic.py:__cal_layout``), so rows of one batch element must be
    contiguous and elements must appear in ascending order.  Per-element ROW
    COUNTS may differ freely — that is what makes the ragged batch legal.

    A violation is SILENT.  The library's own check
    (``basic.py`` "data of batch i is not contiguous") runs only under
    ``SPARSE_DEBUG``, which is off by default and cannot be turned on here
    (``full_attn.py``'s debug assert compares all four coord columns to the
    batch index and fails on any valid input).  So this function is the single
    point of truth, and the caller asserts the built tensor's ``layout`` against
    the offsets returned here.

    Returns ``(coords (sum L_i, 4) int32, offsets)`` where ``offsets`` is the
    cumulative ``[0, L_0, L_0+L_1, ...]`` that slices any ``(sum L_i, C)`` row
    tensor back into per-frame blocks — and equals the resulting layout.
    """
    import torch

    if not coords_per_frame:
        raise ValueError("_ragged_sparse_batch: no frames given")
    out, offsets, running = [], [0], 0
    for i, c in enumerate(coords_per_frame):
        c = torch.as_tensor(c)
        if c.ndim != 2 or c.shape[1] != 4:
            raise ValueError(
                f"_ragged_sparse_batch: frame {i} coords must be (L, 4); "
                f"got {tuple(c.shape)}"
            )
        c = c.to(torch.int32).clone()
        c[:, 0] = i                      # within-batch index, ascending
        out.append(c)
        running += int(c.shape[0])
        offsets.append(running)
    return torch.cat(out, dim=0), offsets


def _embed_slat_conditions(pipeline, batched_input, N, chunk=8):
    """Chunked Stage-2 condition embedding -> ``(N, P, C)``.

    Shared by ``stage2_mv`` and ``stage2_dyn``.  Each frame is embedded
    independently (no cross-frame interaction in the embedder), so this is
    sub-batched purely to bound peak memory on large frame counts.
    """
    import torch

    chunks = []
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        args, _ = pipeline.get_condition_input(
            pipeline.condition_embedders["slat_condition_embedder"],
            {k: v[s:e] for k, v in batched_input.items()},
            pipeline.slat_condition_input_mapping,
        )
        chunks.append(args[0])
    return torch.cat(chunks, dim=0)


def _appearance_guidance_schedule(active_from, active_until):
    """``GuidanceSchedule`` for a Stage-2 appearance guidance transform
    (shared by ``stage2_mv`` and ``stage2_dyn``)."""
    from genia.core.rendering_guidance import GuidanceSchedule

    return GuidanceSchedule(
        active_from=float(active_from),
        active_until=float(active_until),
    )


def _consensus_fuse_step(
    x, dt, vel_frames, canonical_coords, *,
    visibility_alpha=30.0, visibility_min_weight=0.001,
):
    """One consensus-canonical Euler step: fuse per-frame velocities, advance.

    ``vel_frames`` is the ``collapse_perframe_to_canonical`` per-frame tuple
    list ``(velocity (L_pf, C), weights (L_pf,), pf_to_canon (L_pf,))`` — velocities, not final features, so
    the fusion happens at EVERY ODE step rather than once at the end.  The
    reduction is that function verbatim (visibility-weighted within a frame,
    ``softmax(visibility_alpha * v_bar)`` across frames), which is the same
    recipe ``stage2_mv`` applies over views.

    Canonical rows no frame reaches get the mean fused velocity of the reached
    rows: a zero there would leave the row at its initial NOISE for the whole
    solve (unlike the post-hoc collapse, where an unreached row is merely a
    zero feature).

    Returns ``(x_next (1, L_canon, C), unreached (L_canon,) bool)``.  The mask is
    constant across steps (it follows the fixed per-frame geometry); it is
    returned so the caller can report the count.
    """
    from genia.core.gt_geometry import collapse_perframe_to_canonical

    v_canon, frame_count, _ = collapse_perframe_to_canonical(
        canonical_coords, vel_frames,
        visibility_alpha=visibility_alpha,
        visibility_min_weight=visibility_min_weight,
    )
    unreached = frame_count == 0
    if bool(unreached.any()):
        reached = ~unreached
        if bool(reached.any()):
            v_canon = v_canon.clone()
            v_canon[unreached] = v_canon[reached].mean(dim=0)
    return x + v_canon.unsqueeze(0) * dt, unreached


class Inference:
    # public facing inference API
    # only put publicly exposed arguments here
    def __init__(self, config_file: str, compile: bool = False):
        # load inference pipeline
        config = OmegaConf.load(config_file)
        config.rendering_engine = "pytorch3d"  # overwrite to disable nvdiffrast
        config.compile_model = compile
        config.workspace_dir = os.path.dirname(config_file)
        check_hydra_safety(config, WHITELIST_FILTERS, BLACKLIST_FILTERS)
        self._pipeline: InferencePipelinePointMap = instantiate(config)

    def merge_mask_to_rgba(self, image, mask):
        mask = mask.astype(np.uint8) * 255
        mask = mask[..., None]
        # embed mask in alpha channel
        rgba_image = np.concatenate([image[..., :3], mask], axis=-1)
        return rgba_image

    def stage1_batched(
        self,
        frames: List[dict],
        canonical_shape: torch.Tensor,
        seed: Optional[int] = None,
        stage1_inference_steps: Optional[int] = None,
        shape_velocity_averaging: str = "none",
        rotation_velocity_averaging: str = "none",
        frame_indices: Optional[List[int]] = None,
        entropy_alpha: float = 60.0,
        entropy_layer: int = 9,
        entropy_min_weight: float = 0.001,
        user_step_callback=None,
        gt_shape_trajectory: bool = False,
        cfg_interval_pose: Optional[List[int]] = None,
        pose_velocity_broadcast_per_frame: bool = False,
    ) -> tuple:
        """Run Stage 1 (sparse structure) on multiple frames in one batched call.

        Every frame's shape and pose modalities are denoised together in one ODE
        solve.  Cross-frame coupling is limited to the velocity transforms below:
        shape velocity averaging within each timestamp, the pose-velocity
        broadcast within each timestamp, and a per-view rotation consensus.

        Args:
            frames: List of dicts, each with keys ``image`` (HxWx3 uint8 numpy),
                ``mask`` (HxW bool numpy), ``pointmap`` (HxWx3 torch tensor, PyTorch3D
                convention).
            canonical_shape: Shape latent, ``(1, 4096, 8)`` or per-frame ``(N, 4096, 8)``;
                the clean end point of the ``gt_shape_trajectory``.
            seed: Random seed for reproducibility.
            shape_velocity_averaging: ``"none"`` or ``"entropy"`` (entropy-weighted
                consensus over the views of each timestamp).
            rotation_velocity_averaging: ``"none"`` or ``"median"`` (per-view median
                of the rotation velocity across that view's timestamps).
            gt_shape_trajectory: Drive ``x_t["shape"]`` along ``(1-t)*noise +
                t*canonical_shape`` so only the pose modalities are denoised.
            cfg_interval_pose: CFG interval for the pose modalities; an empty list
                disables CFG on pose.  ``None`` keeps the model default.

        Returns:
            ``(results, ode_histories, entropy_data, per_frame_scale, per_frame_shift)``:
            - results: per-frame dicts with ``rotation``, ``translation``, ``scale``,
              ``raw_ss_modalities``.
            - ode_histories: ``{batch_idx: [(t, {modality: tensor}), ...]}``.
        """
        pipeline = self._pipeline
        N = len(frames)

        if shape_velocity_averaging not in ("none", "entropy"):
            raise ValueError(
                f"shape_velocity_averaging must be 'none' or 'entropy', got "
                f"{shape_velocity_averaging!r}")
        if rotation_velocity_averaging not in ("none", "median"):
            raise ValueError(
                f"rotation_velocity_averaging must be 'none' or 'median', got "
                f"{rotation_velocity_averaging!r}")
        _do_shape_vel_avg = shape_velocity_averaging == "entropy"
        _do_rot_vel_avg = rotation_velocity_averaging == "median"
        if gt_shape_trajectory and _do_shape_vel_avg:
            raise ValueError(
                "stage1_batched: gt_shape_trajectory=True with shape velocity "
                "averaging is ambiguous — averaging would silently skip the "
                "trajectory callback. Disable one of them."
            )

        # 1. Preprocess each frame individually (preprocess_image asserts ndim==3)
        ss_inputs = []
        for f in frames:
            rgba = self.merge_mask_to_rgba(f["image"], f["mask"])
            pm = f["pointmap"]
            # compute_pointmap just passes through when pointmap is provided
            pm_dict = pipeline.compute_pointmap(rgba, pointmap=pm)
            ss_input = pipeline.preprocess_image(
                rgba, pipeline.ss_preprocessor, pointmap=pm_dict["pointmap"],
            )
            ss_inputs.append(ss_input)

        # Save per-frame pointmap_scale/shift for pose decoding
        per_frame_scale = [d.get("pointmap_scale") for d in ss_inputs]
        per_frame_shift = [d.get("pointmap_shift") for d in ss_inputs]

        # 2. Stack into batch (concatenate along dim=0)
        batched_ss_input = {}
        for key in ss_inputs[0]:
            vals = [d[key] for d in ss_inputs]
            if isinstance(vals[0], torch.Tensor):
                batched_ss_input[key] = torch.cat(vals, dim=0)
            else:
                batched_ss_input[key] = vals[0]  # non-tensor: take first

        # 3. Run generator with batch=N
        ss_generator = pipeline.models["ss_generator"]
        ss_generator.no_shortcut = True
        ss_generator.reverse_fn.strength = pipeline.ss_cfg_strength
        ss_generator.reverse_fn.strength_pm = pipeline.ss_cfg_strength_pm

        # Optional ODE step-count override, restored before return so later
        # stage1 calls in the same process see the original value.
        _ss_prev_inference_steps = ss_generator.inference_steps
        if stage1_inference_steps is not None:
            ss_generator.inference_steps = int(stage1_inference_steps)
            print(f"    Stage-1 ODE steps: {_ss_prev_inference_steps} -> "
                  f"{ss_generator.inference_steps} (override)")

        # Pose CFG interval override (shape keeps the model default).
        _cfg_overridden = cfg_interval_pose is not None
        if _cfg_overridden:
            _default_iv = tuple(ss_generator.reverse_fn.interval)
            _s = float(ss_generator.reverse_fn.strength)
            _s_pm = float(ss_generator.reverse_fn.strength_pm)
            if len(cfg_interval_pose) > 0:
                _pose_iv, _pose_s, _pose_s_pm = tuple(cfg_interval_pose), _s, _s_pm
            else:  # empty list → disable CFG on pose
                _pose_iv, _pose_s, _pose_s_pm = _default_iv, 0.0, 0.0

            _iv_dict = {"shape": _default_iv}
            _s_dict = {"shape": _s}
            _s_pm_dict = {"shape": _s_pm}
            for k in ("6drotation_normalized", "translation",
                       "scale", "translation_scale"):
                _iv_dict[k] = _pose_iv
                _s_dict[k] = _pose_s
                _s_pm_dict[k] = _pose_s_pm

            _orig_iv = ss_generator.reverse_fn.interval
            _orig_s = ss_generator.reverse_fn.strength
            _orig_s_pm = ss_generator.reverse_fn.strength_pm
            ss_generator.reverse_fn.interval = _iv_dict
            ss_generator.reverse_fn.strength = _s_dict
            ss_generator.reverse_fn.strength_pm = _s_pm_dict

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=pipeline.shape_model_dtype):
                # Build latent shapes with batch=N
                latent_shape_dict = {
                    k: (N,) + (v.pos_emb.shape[0], v.input_layer.in_features)
                    for k, v in ss_generator.reverse_fn.backbone.latent_mapping.items()
                }

                # Condition embedder (DINOv2 + pointmap) — sub-batched to
                # avoid OOM on large frame counts (each frame is independently
                # embedded; no cross-frame interaction in the embedder).
                _COND_CHUNK = 8
                embedded_chunks = []
                for _s in range(0, N, _COND_CHUNK):
                    _e = min(_s + _COND_CHUNK, N)
                    chunk_input = {
                        k: v[_s:_e] if isinstance(v, torch.Tensor)
                           and v.shape[0] == N else v
                        for k, v in batched_ss_input.items()
                    }
                    chunk_args, _ = pipeline.get_condition_input(
                        pipeline.condition_embedders["ss_condition_embedder"],
                        chunk_input,
                        pipeline.ss_condition_input_mapping,
                    )
                    embedded_chunks.append(chunk_args[0])
                cond_args = (torch.cat(embedded_chunks, dim=0),)
                cond_kwargs = {}

                if seed is not None:
                    torch.manual_seed(seed)

                # Initial noise: one sample replicated to all N frames, so results
                # do not depend on batch composition.  Shape noise is replicated
                # too: at t=0 the backbone jointly processes all modalities, so
                # differing shape noise would change the pose velocities.
                noise_init = {}
                for k, v in ss_generator.reverse_fn.backbone.latent_mapping.items():
                    single = torch.randn(
                        (1, v.pos_emb.shape[0], v.input_layer.in_features),
                        device=batched_ss_input["image"].device,
                    )
                    noise_init[k] = single.expand(N, -1, -1).contiguous()
                cond_kwargs["noise_init_override"] = noise_init

                # GT shape trajectory: x_t["shape"] = (1-t)*noise + t*clean at every
                # ODE step, so the backbone sees realistic noisy-shape context while
                # the shape converges to the given canonical at t=1.
                if gt_shape_trajectory:
                    _shape_noise = noise_init["shape"]
                    _shape_clean = canonical_shape.expand(N, -1, -1)

                    def _gt_shape_cb(t_val, x_t, velocity,
                                     _noise=_shape_noise, _clean=_shape_clean):
                        t = t_val.item() if hasattr(t_val, "item") else float(t_val)
                        x_t["shape"] = (1 - t) * _noise + t * _clean

                    if user_step_callback is not None:
                        _prev_user = user_step_callback

                        def _composed_user(t_val, x_t, velocity,
                                           _p=_prev_user, _g=_gt_shape_cb):
                            _g(t_val, x_t, velocity)
                            _p(t_val, x_t, velocity)
                        user_step_callback = _composed_user
                    else:
                        user_step_callback = _gt_shape_cb

                # -- Per-batch view/timestamp groups --
                # Each batch element is a (frame, view) pair.  Builds:
                #   _view_ids[i]:         view of batch element i
                #   _frame_group_lead[i]: lowest-view element sharing element i's
                #                         timestamp (pose-velocity broadcast)
                #   _shape_avg_groups[i]: dense timestamp id (shape averaging)
                if frame_indices is not None and frame_indices \
                        and hasattr(frame_indices[0], "view"):
                    _view_ids = [int(fk.view) for fk in frame_indices]
                    _lead_per_frame = {}
                    for bi, fk in enumerate(frame_indices):
                        cur = _lead_per_frame.get(fk.frame)
                        if cur is None or _view_ids[bi] < _view_ids[cur]:
                            _lead_per_frame[fk.frame] = bi
                    _frame_group_lead = [
                        _lead_per_frame[fk.frame] for fk in frame_indices
                    ]
                    _frame_group_dense_id = {}
                    _shape_avg_groups = []
                    for fk in frame_indices:
                        gid = _frame_group_dense_id.setdefault(
                            fk.frame, len(_frame_group_dense_id),
                        )
                        _shape_avg_groups.append(gid)
                else:
                    # Bare ints / None: one view, every element its own timestamp.
                    _view_ids = [0] * N
                    _frame_group_lead = list(range(N))
                    _shape_avg_groups = list(range(N))

                # Per-frame ODE history: {batch_idx: [(t, {mod: tensor}), ...]}
                ode_histories = {i: [] for i in range(N)}
                _log_keys = ["6drotation_normalized", "translation",
                             "scale", "translation_scale"]

                def _step_callback(t, x_t, velocity):
                    t_val = float(t)
                    for bi in range(N):
                        snapshot = {}
                        for k in _log_keys:
                            if k in x_t:
                                snapshot[k] = x_t[k][bi].detach().cpu().clone()
                        # Compact shape summary: mean feature (D,) + norm std (1,)
                        if "shape" in x_t:
                            _s = x_t["shape"][bi]  # (tokens, D)
                            snapshot["shape_mean"] = _s.mean(dim=0).detach().cpu().clone()
                            snapshot["shape_norm_std"] = (
                                _s.norm(dim=-1).std().detach().cpu().unsqueeze(0).clone()
                            )
                        ode_histories[bi].append((t_val, snapshot))

                if user_step_callback is not None:
                    def _composed_callback(t_val, x_t, velocity,
                                           _inner=_step_callback,
                                           _user=user_step_callback):
                        _user(t_val, x_t, velocity)
                        _inner(t_val, x_t, velocity)
                    cond_kwargs["step_callback"] = _composed_callback
                else:
                    cond_kwargs["step_callback"] = _step_callback

                backbone = ss_generator.reverse_fn.backbone
                blocks = backbone.blocks

                # --- Shape velocity averaging (entropy-weighted, per timestamp) ---
                # Views at the same timestamp consense; different timestamps stay
                # independent (a no-op on mono-dynamic, where each group has size 1).
                _entropy_hook = None
                if _do_shape_vel_avg:
                    from genia.core.entropy import ShapeEntropyHook

                    _shape_avg_groups_t = torch.tensor(_shape_avg_groups, dtype=torch.long)
                    _shape_avg_n_groups = int(_shape_avg_groups_t.max().item()) + 1
                    block_idx = (
                        entropy_layer if entropy_layer >= 0
                        else len(blocks) + entropy_layer
                    )
                    _entropy_hook = ShapeEntropyHook(
                        blocks[block_idx].cross_attn["shape"],
                        alpha=entropy_alpha,
                        min_weight=entropy_min_weight,
                    )
                    print(f"    Entropy weighting: layer={block_idx}/{len(blocks)}, "
                          f"alpha={entropy_alpha}, min_weight={entropy_min_weight}")

                    def _shape_velocity_avg(t_raw, velocity, x_t=None):
                        """Entropy-weighted shape velocity averaging."""
                        if "shape" not in velocity:
                            return velocity
                        v = velocity["shape"]                # (N, L, D)
                        w = _entropy_hook.get_weights()      # (N, L) or None
                        grp = _shape_avg_groups_t.to(v.device)
                        G = _shape_avg_n_groups
                        if w is not None:
                            # weighted promotes to w.dtype (float32);
                            # sums_v must match for index_add_.
                            weighted = w.unsqueeze(-1) * v   # (N, L, D)
                            sums_v = torch.zeros(
                                (G, v.shape[1], v.shape[2]),
                                device=v.device, dtype=weighted.dtype,
                            )
                            sums_v.index_add_(0, grp, weighted)
                            sums_w = torch.zeros(
                                (G, w.shape[1]), device=w.device, dtype=w.dtype,
                            )
                            sums_w.index_add_(0, grp.to(w.device), w)
                            # Renormalize: entropy weights summed to 1 across N
                            # globally; after scattering they no longer sum to 1
                            # within a group.
                            means = sums_v / sums_w.clamp_min(1e-12).unsqueeze(-1)
                        else:
                            sums_v = torch.zeros(
                                (G, v.shape[1], v.shape[2]),
                                device=v.device, dtype=v.dtype,
                            )
                            sums_v.index_add_(0, grp, v)
                            counts = torch.bincount(
                                grp, minlength=G,
                            ).clamp_min(1).to(v.dtype).view(-1, 1, 1)
                            means = sums_v / counts
                        velocity["shape"] = means.index_select(0, grp)
                        _entropy_hook.reset_step()
                        return velocity

                # --- Pose-velocity broadcast within timestamp groups ---
                # Every view of a timestamp takes the lowest-view element's pose
                # velocity.
                _POSE_BROADCAST_KEYS = (
                    "6drotation_normalized", "translation",
                    "scale", "translation_scale",
                )
                if pose_velocity_broadcast_per_frame:
                    _lead_idx = torch.tensor(_frame_group_lead, dtype=torch.long)
                    _is_noop = all(_frame_group_lead[i] == i for i in range(N))

                    def _pose_velocity_broadcast(t_raw, velocity, x_t=None):
                        if _is_noop:
                            return velocity
                        for k in _POSE_BROADCAST_KEYS:
                            if k in velocity:
                                _idx = _lead_idx.to(velocity[k].device)
                                velocity[k] = velocity[k].index_select(0, _idx)
                        return velocity

                # --- Rotation velocity consensus (per-view median) ---
                # Temporal only: grouped by view and reduced across that view's
                # timestamps.  Cross-view averaging is never valid (camera-space
                # rotations differ per view).  The median is robust to one
                # badly-conditioned frame, which moves a mean at every step.
                if _do_rot_vel_avg:
                    _rot_key = "6drotation_normalized"
                    _rot_groups_t = torch.tensor(
                        _view_ids, dtype=torch.long,
                    ).to(canonical_shape.device)
                    print("    Rotation velocity averaging: median")

                    def _rotation_velocity_avg(t_raw, velocity, x_t=None):
                        if _rot_key in velocity:
                            v = velocity[_rot_key]
                            out = v.clone()
                            ids = _rot_groups_t.to(v.device)
                            for gid in ids.unique().tolist():
                                m = ids == gid
                                out[m] = v[m].median(dim=0, keepdim=True).values
                            velocity[_rot_key] = out
                        return velocity

                # --- Compose velocity transforms ---
                _transforms = []
                if _do_shape_vel_avg:
                    _transforms.append(_shape_velocity_avg)
                if pose_velocity_broadcast_per_frame:
                    _transforms.append(_pose_velocity_broadcast)
                if _do_rot_vel_avg:
                    _transforms.append(_rotation_velocity_avg)

                if len(_transforms) == 1:
                    ss_generator._velocity_transform = _transforms[0]
                elif len(_transforms) > 1:
                    def _composed_velocity_transform(t_raw, velocity, x_t):
                        for _fn in _transforms:
                            velocity = _fn(t_raw, velocity, x_t)
                        return velocity
                    ss_generator._velocity_transform = (
                        _composed_velocity_transform
                    )

                # Run the generator with progress bar
                # (skip ss_decoder / coord processing)
                from tqdm import tqdm
                pbar = tqdm(
                    ss_generator.generate_iter(
                        latent_shape_dict,
                        batched_ss_input["image"].device,
                        *cond_args,
                        **cond_kwargs,
                    ),
                    total=ss_generator.inference_steps,
                    desc=f"  Parallel ODE ({N} frames)",
                )
                for _, return_dict, _ in pbar:
                    pass
                pbar.close()

                # Restore the CFG override
                if _cfg_overridden:
                    ss_generator.reverse_fn.interval = _orig_iv
                    ss_generator.reverse_fn.strength = _orig_s
                    ss_generator.reverse_fn.strength_pm = _orig_s_pm

                # Clean up velocity transform + hooks
                ss_generator._velocity_transform = None
                _entropy = None
                if _entropy_hook is not None:
                    _entropy = _entropy_hook.get_entropy()
                    _entropy_hook.remove()

        # 4. Split per element and run pose decoder
        raw_modality_keys = [
            "shape", "6drotation_normalized", "translation",
            "scale", "translation_scale",
        ]
        results = []
        for i in range(N):
            elem = {k: v[i:i+1] for k, v in return_dict.items()
                    if isinstance(v, torch.Tensor)}
            elem_raw = {k: elem[k].clone() for k in raw_modality_keys if k in elem}
            # Pose decoder (per-element, since it uses squeeze(0))
            decoded = pipeline.pose_decoder(
                elem, scene_scale=per_frame_scale[i], scene_shift=per_frame_shift[i],
            )
            results.append({**decoded, "raw_ss_modalities": elem_raw})

        entropy_data = None
        if _entropy is not None:
            entropy_data = {"entropy": _entropy.cpu()}

        # Restore previous ss_generator.inference_steps after the override
        # block above (no-op when the kwarg was None).
        ss_generator.inference_steps = _ss_prev_inference_steps

        return results, ode_histories, entropy_data, per_frame_scale, per_frame_shift

    def stage2_mv(
        self,
        frames: List[dict],
        canonical_coords: torch.Tensor,
        inference_steps: int = 25,
        seed: "int | None" = None,
        visibility_min_weight: float = 0.001,
        fused_min_weight: float = 0.001,
        visibility_weights: "torch.Tensor | None" = None,
        visibility_alpha: float = 30.0,
        visibility_pixel_coords: "list[dict[int, np.ndarray]] | None" = None,
        visibility_attn_bias_alpha: float = 0.0,
        visibility_attn_bias_layers: str = "all",
        visibility_attn_bias_streams: "dict[str, bool] | None" = None,
        visibility_attn_bias_compensate_passive_streams: str = "off",
        visibility_attn_debug: bool = False,
        capture_dino_features: bool = False,
        # Rendering-based velocity guidance (in-ODE)
        rendering_guidance_active: bool = False,
        rendering_guidance_velocity_weight: float = 1.0,
        rendering_guidance_active_from: float = 0.0,
        rendering_guidance_active_until: float = 1.0,
        rendering_guidance_losses_cfg=None,
        rendering_guidance_pose_decoder=None,
        rendering_guidance_bg_color: "torch.Tensor | None" = None,
        rendering_guidance_microbatch_size: int = 8,
        rendering_guidance_resolution_scale: int = 1,
        rendering_guidance_normalize_grad: bool = False,
        rendering_guidance_decoder_autocast_bf16: bool = False,
        rendering_guidance_random_background: bool = False,
        rendering_guidance_random_background_seed: "int | None" = None,
        rendering_guidance_visibility_detach: bool = False,
        rendering_guidance_visibility_depth_margin: float = 0.02,
        # Per-canonical-mesh-vertex correspondence (high-resolution Φ + R)
        # — consumed by the rendering-guidance builder for the rigid-LBS
        # warp at decoded primitive positions.  Independent of the voxel
        # field above; populated by GT_SHAPES_INVERSION in any per-frame mode.
        canonical_mesh_verts: "torch.Tensor | None" = None,
        per_frame_mesh_verts: "list | None" = None,
        per_frame_mesh_rotations: "list | None" = None,
        canonical_mesh_faces: "torch.Tensor | None" = None,
        # Warp KNN/blend knobs (forwarded from the global ``deformation_warp`` config).
        warp_knn_k: int = 4,
        warp_knn_eps: float = 1.0e-8,
        warp_knn_chunk_size: int = 8192,
        # Per-step debug snapshot capture: when True, surfaces every ODE
        # step's full ``(L, 8)`` SLAT feats via
        # ``entropy_data["slat_ode_snapshots"]`` for the appearance ODE-
        # step viz.  Off by default — captures cost ~L·8·4 B per step in CPU memory.
        capture_slat_snapshots: bool = False,
    ):
        """Run Stage 2 (SLAT appearance) with multi-view velocity averaging.

        Produces N per-view velocity predictions per ODE step by batching
        all N views into a single SparseTensor with N batch elements (2
        backbone calls per step for CFG, not 2N).  Velocities are fused
        with visibility weights (or uniformly) before the Euler step.

        Args:
            frames: List of dicts with ``image`` (HxWx3 uint8 numpy) and
                ``mask`` (HxW bool numpy), one per frame.
            canonical_coords: Voxel coordinates (num_voxels, 4) tensor
                with batch column.
            inference_steps: Number of flow-matching ODE steps.
            visibility_weights: Pre-computed ``(N, L_original)`` visibility
                matrix (1=visible, 0=occluded) from DDA ray tracing.
                ``None`` = uniform fusion.
            visibility_alpha: Temperature for visibility softmax.
            visibility_pixel_coords: Per-view list of ``{voxel_idx: (M,2)}``
                pixel coords from ``compute_visibility_multi_object``.
                Used for cross-attention bias.
            visibility_attn_bias_alpha: Additive bias temperature for
                cross-attention.  0 = disabled.
            visibility_attn_bias_layers: ``"all"`` or comma-separated layer
                indices for cross-attention bias.
            visibility_attn_debug: Enable debug attention capture in hooks
                (expensive — only for visualization).

        Returns:
            (slat, entropy_data, ode_history): SparseTensor with canonical
            coords and multi-view-averaged appearance features, a diagnostics
            dict (or None), and ODE trajectory history
            ``[(t, {"slat_mean": (8,), "slat_norm_std": (1,)}), ...]``.
        """
        from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp
        from sam3d_objects.model.backbone.tdfy_dit.models.structured_latent_flow import (
            SLatFlowModel,
        )

        pipeline = self._pipeline
        N = len(frames)

        # 1. Preprocess all frames
        preprocessed = []
        for f in frames:
            rgba = self.merge_mask_to_rgba(f["image"], f["mask"])
            slat_input = pipeline.preprocess_image(
                rgba, pipeline.slat_preprocessor
            )
            preprocessed.append(slat_input)

        # 2. Stack into batch
        batched_input = {
            k: torch.cat([p[k] for p in preprocessed], dim=0)
            for k in preprocessed[0].keys()
        }
        device = batched_input["image"].device

        # 3. Configure generator
        slat_generator = pipeline.models["slat_generator"]
        L = canonical_coords.shape[0]
        latent_shape = (N, L, 8)
        prev_steps = slat_generator.inference_steps
        prev_no_shortcut = getattr(slat_generator, "no_shortcut", None)
        prev_dynamics = slat_generator._generate_dynamics
        if inference_steps:
            slat_generator.inference_steps = inference_steps
        if prev_no_shortcut is not None:
            slat_generator.no_shortcut = True
        slat_generator.reverse_fn.strength = pipeline.slat_cfg_strength

        with torch.autocast(device_type="cuda", dtype=pipeline.dtype):
            with torch.no_grad():
                # 5. Sub-batch condition embedding (DINOv2) to avoid OOM
                _dino_hook = None
                if capture_dino_features:
                    from genia.core.visibility_attn import DinoFeatureCaptureHook
                    _embedder_fuser = pipeline.condition_embedders[
                        "slat_condition_embedder"
                    ]
                    _dino_hook = DinoFeatureCaptureHook(_embedder_fuser)

                cond_embedded = _embed_slat_conditions(
                    pipeline, batched_input, N)          # (N, P, C)

                _dino_features = None
                if _dino_hook is not None:
                    _dino_features = _dino_hook.get_per_stream_features()
                    _dino_hook.remove()

                # 6. Build batched coords: N copies of canonical_coords
                # with the batch column set to the view index.  All N views
                # share the canonical voxel layout (static).  Deformation-
                # aware processing lives in stage2_dyn (canonical_unified,
                # run_appearance_init in core/main.py).
                batched_coords_list: "list[torch.Tensor]" = []
                for i in range(N):
                    c = canonical_coords.clone()
                    c[:, 0] = i
                    batched_coords_list.append(c)
                batched_coords = torch.cat(batched_coords_list, dim=0)  # (N*L, 4)

                # 7. Access underlying SLatFlowModel (bypass wrapper's x[0])
                wrapper = slat_generator.reverse_fn.backbone
                cfg_strength = float(pipeline.slat_cfg_strength)
                time_scale = slat_generator.time_scale

                _fused_weights = [None]   # visibility fusion weights (frozen)

                # 8. Set up visibility cross-attention bias hooks
                _vis_bias_hooks = []
                if (visibility_pixel_coords is not None
                        and visibility_attn_bias_alpha > 0):
                    _vis_bias_hooks = _install_vis_bias_hooks(
                        wrapper, frames, visibility_pixel_coords, L, N,
                        alpha=visibility_attn_bias_alpha,
                        layers=visibility_attn_bias_layers,
                        streams=visibility_attn_bias_streams,
                        compensate_passive_streams=
                            visibility_attn_bias_compensate_passive_streams,
                        debug=visibility_attn_debug,
                    )

                # 8d. Build in-ODE rendering guidance transform (off by default)
                _guidance_transform = None
                _guidance_loss_history: list = []
                if rendering_guidance_active:
                    from genia.core.rendering_guidance import (
                        build_appearance_rendering_guidance_transform,
                    )
                    if rendering_guidance_pose_decoder is None:
                        raise ValueError(
                            "rendering_guidance_active=True requires "
                            "rendering_guidance_pose_decoder closure."
                        )
                    if rendering_guidance_losses_cfg is None:
                        raise ValueError(
                            "rendering_guidance_active=True requires "
                            "rendering_guidance_losses_cfg (LossConfig)."
                        )
                    _bg = rendering_guidance_bg_color
                    if _bg is None:
                        _bg = torch.ones(3, device=device, dtype=torch.float32)
                    _schedule = _appearance_guidance_schedule(
                        rendering_guidance_active_from,
                        rendering_guidance_active_until,
                    )
                    _guidance_transform, _guidance_loss_history = (
                        build_appearance_rendering_guidance_transform(
                            pipeline=pipeline,
                            frames=frames,
                            pose_decoder=rendering_guidance_pose_decoder,
                            canonical_coords=canonical_coords,
                            slat_mean=pipeline.slat_mean,
                            slat_std=pipeline.slat_std,
                            losses_cfg=rendering_guidance_losses_cfg,
                            velocity_weight=float(rendering_guidance_velocity_weight),
                            schedule=_schedule,
                            bg_color=_bg,
                            device=device,
                            microbatch_size=rendering_guidance_microbatch_size,
                            resolution_scale=rendering_guidance_resolution_scale,
                            normalize_grad=rendering_guidance_normalize_grad,
                            decoder_autocast_bf16=rendering_guidance_decoder_autocast_bf16,
                            random_background=rendering_guidance_random_background,
                            random_background_seed=rendering_guidance_random_background_seed,
                            visibility_detach=rendering_guidance_visibility_detach,
                            visibility_depth_margin=rendering_guidance_visibility_depth_margin,
                            canonical_mesh_verts=canonical_mesh_verts,
                            per_frame_mesh_verts=per_frame_mesh_verts,
                            per_frame_mesh_rotations=per_frame_mesh_rotations,
                            canonical_mesh_faces=canonical_mesh_faces,
                            warp_knn_k=int(warp_knn_k),
                            warp_knn_eps=float(warp_knn_eps),
                            warp_knn_chunk_size=int(warp_knn_chunk_size),
                        )
                    )

                # 9. Monkey-patch _generate_dynamics for per-view batched calls.
                # FlowMatching._generate_dynamics signature: (self, x_t, t, *args, **kwargs)
                # The SLAT generator is always FlowMatching (not ShortCut),
                # so there is no `d` positional arg.

                def _per_view_dynamics(x_t, t, *args_cond, **kwargs_cond):
                    """Batched per-view dynamics with manual CFG."""
                    t_val = t.item() if isinstance(t, torch.Tensor) else t
                    t_tensor = torch.tensor(
                        [t_val * time_scale], device=device, dtype=torch.float32,
                    )
                    d_tensor = None

                    # x_t: (N, L, D) — all N elements identical
                    # Build batched SparseTensor with N batch elements
                    x_sparse = sp.SparseTensor(
                        feats=x_t.reshape(N * L, -1),
                        coords=batched_coords,
                    )

                    # Conditional forward: each batch element sees its own cond
                    for _h in _vis_bias_hooks:
                        _h.set_active(True)
                    y_cond = SLatFlowModel.forward(
                        wrapper, x_sparse, t_tensor, cond_embedded, d_tensor,
                    )

                    # Unconditional forward: zero conditions (no visibility bias)
                    for _h in _vis_bias_hooks:
                        _h.set_active(False)
                    y_uncond = SLatFlowModel.forward(
                        wrapper, x_sparse, t_tensor,
                        torch.zeros_like(cond_embedded), d_tensor,
                    )

                    # CFG combination: (1 + s) * cond - s * uncond
                    v = (
                        (1 + cfg_strength) * y_cond.feats
                        - cfg_strength * y_uncond.feats
                    )
                    v = v.reshape(N, L, -1)  # (N, L, D)

                    # Visibility fusion weights (computed once, frozen): binary
                    # 0/1 visibility -> softmax over views -> clamp + renorm.
                    if _fused_weights[0] is None and visibility_weights is not None:
                        w_v = F.softmax(
                            visibility_alpha * visibility_weights, dim=0,
                        )
                        w_v = w_v.clamp(min=visibility_min_weight)
                        w_v = w_v / w_v.sum(dim=0, keepdim=True).clamp(min=1e-10)
                        w_v = w_v.clamp(min=fused_min_weight)
                        _fused_weights[0] = w_v / w_v.sum(
                            dim=0, keepdim=True).clamp(min=1e-10)

                    # Fuse per-view velocities: visibility-weighted, else the
                    # per-canonical-voxel uniform mean across views.
                    w = _fused_weights[0]
                    if w is not None:
                        v_fused = (w.unsqueeze(-1) * v).sum(0, keepdim=True)
                    else:
                        v_fused = v.mean(dim=0, keepdim=True)

                    # In-ODE rendering guidance (classifier-style velocity update)
                    if _guidance_transform is not None:
                        v_fused = _guidance_transform(t_val, v_fused, x_t[0:1])

                    return v_fused.expand(N, -1, -1)

                slat_generator._generate_dynamics = _per_view_dynamics

                # 10. Replicate initial noise
                cond_kwargs = {}
                coords_numpy = canonical_coords.cpu().numpy()
                # Seed right before the initial-noise draw so the SLAT flow is
                # reproducible across runs (mirrors stage1_batched).  Makes
                # rg-off vs rg-on a paired A/B: identical starting noise, so the
                # only difference is the rendering-guidance perturbation.
                if seed is not None:
                    torch.manual_seed(seed)
                single_noise = torch.randn(
                    (1, L, 8), device=device,
                )
                cond_kwargs["noise_init_override"] = (
                    single_noise.expand(N, -1, -1).contiguous()
                )

                # 11. Run generator (dynamics is monkey-patched above)
                from tqdm import tqdm
                num_steps = slat_generator.inference_steps
                ode_history = []  # [(t, {slat_mean: (8,), slat_norm_std: (1,)})]
                slat_snapshots: list = []  # only populated when capture_slat_snapshots
                pbar = tqdm(
                    slat_generator.generate_iter(
                        latent_shape, device,
                        cond_embedded, coords_numpy,
                        **cond_kwargs,
                    ),
                    total=num_steps,
                    desc=f"  Stage 2 ({N} frames)",
                )
                for t_val, slat_out, _ in pbar:
                    t_float = t_val.item() if isinstance(t_val, torch.Tensor) else float(t_val)
                    # All N views are identical after fusion — snapshot element 0
                    _s = slat_out[0]  # (L, 8)
                    ode_history.append((t_float, {
                        "slat_mean": _s.mean(dim=0).detach().cpu().clone(),
                        "slat_norm_std": _s.norm(dim=-1).std().detach().cpu().unsqueeze(0).clone(),
                    }))
                    if capture_slat_snapshots:
                        slat_snapshots.append((t_float, _s.detach().cpu().clone()))
                    if _guidance_loss_history:
                        _e = _guidance_loss_history[-1]
                        # Show the guidance loss + diagnostics (grad norm,
                        # frame count).  Empty-decode steps surface
                        # skip_reason instead.
                        if _e.get("skip_reason"):
                            pbar.set_postfix(
                                t=f"{_e['t']:.2f}",
                                skip=_e["skip_reason"],
                            )
                        else:
                            _post = {
                                "t": f"{_e['t']:.2f}",
                                "rg": f"{_e['total']:.3f}",
                                "gn": f"{_e.get('grad_norm', 0.0):.2e}",
                            }
                            if _e.get("n_g"):
                                _post["ng"] = str(_e["n_g"])
                            pbar.set_postfix(**_post)
                pbar.close()

                # 12. Cleanup
                slat_generator._generate_dynamics = prev_dynamics
                # Collect debug attention from visibility bias hooks
                _vis_bias_debug = []
                for h in _vis_bias_hooks:
                    before, after = h.get_debug_attention()
                    _vis_bias_debug.append({
                        "attn_before": before,
                        "attn_after": after,
                        "bias": h.get_bias(),
                        "mask_down": h.get_mask_downsampled(),
                        "mask_down_full": h.get_mask_downsampled_full(),
                        "coords": h.get_coords(),
                    })
                    h.remove()

                # 13. Extract element 0 (all identical), wrap in SparseTensor.
                # Final SLAT lives in canonical coords (downstream decoders
                # read these via canonical_coords from the outer scope).
                slat = sp.SparseTensor(
                    coords=canonical_coords,
                    feats=slat_out[0],  # (num_voxels, 8)
                ).to(device)
                slat = (
                    slat * pipeline.slat_std.to(device)
                    + pipeline.slat_mean.to(device)
                )

        slat_generator.inference_steps = prev_steps
        if prev_no_shortcut is not None:
            slat_generator.no_shortcut = prev_no_shortcut

        entropy_data = {}
        if visibility_weights is not None:
            entropy_data["visibility"] = visibility_weights.cpu()  # (N, L_original)
        if _vis_bias_debug:
            entropy_data["vis_bias_debug"] = _vis_bias_debug
        if _dino_features:
            entropy_data["dino_patch_features"] = _dino_features
        if _guidance_loss_history:
            entropy_data["rendering_guidance_loss_history"] = _guidance_loss_history
        if slat_snapshots:
            entropy_data["slat_ode_snapshots"] = slat_snapshots
        return slat, entropy_data or None, ode_history

    def stage2_dyn(
        self,
        frames: List[dict],
        canonical_coords: torch.Tensor,
        perframe_entries: list,
        inference_steps: int = 25,
        seed: "int | None" = None,
        visibility_alpha: float = 30.0,
        visibility_min_weight: float = 0.001,
        # Cross-attention visibility bias (per frame, N=1 each)
        visibility_attn_bias_alpha: float = 0.0,
        visibility_attn_bias_layers: str = "all",
        visibility_attn_bias_streams: "dict[str, bool] | None" = None,
        visibility_attn_bias_compensate_passive_streams: str = "off",
        visibility_attn_debug: bool = False,
        # In-ODE rendering guidance (per frame, applied BEFORE fusion)
        rendering_guidance_active: bool = False,
        rendering_guidance_velocity_weight: float = 1.0,
        rendering_guidance_active_from: float = 0.0,
        rendering_guidance_active_until: float = 1.0,
        rendering_guidance_losses_cfg=None,
        rendering_guidance_bg_color: "torch.Tensor | None" = None,
        rendering_guidance_microbatch_size: int = 8,
        rendering_guidance_resolution_scale: int = 1,
        rendering_guidance_normalize_grad: bool = False,
        rendering_guidance_decoder_autocast_bf16: bool = False,
        rendering_guidance_random_background: bool = False,
        rendering_guidance_random_background_seed: "int | None" = None,
        rendering_guidance_visibility_detach: bool = False,
        rendering_guidance_visibility_depth_margin: float = 0.02,
        capture_slat_snapshots: bool = False,
        frame_chunk: "int | None" = 8,
    ):
        """Stage 2 with CONSENSUS-CANONICAL in-ODE fusion across frames.

        The dynamic analogue of :meth:`stage2_mv`.  ``stage2_mv`` can average
        per-view velocities directly because every view conditions the SAME
        canonical grid; a deforming object gives each frame its own voxel
        grid, so there are no aligned rows to average.  This driver keeps ONE
        canonical latent and, at every ODE step, gathers it out to each
        frame's own grid through the GT correspondence, evaluates that frame's
        CFG velocity there, maps the velocities back to canonical, fuses them
        with the visibility-weighted rule, and Euler-steps the canonical
        latent alone.  Per-frame states are re-gathered from the canonical
        every step, so the trajectories cannot drift apart.

        Cost is ``2*ceil(N/K)`` backbone forwards per step (``K =
        frame_chunk``): the frames of a step are independent given the
        canonical latent, so they ride ONE ragged sparse batch instead of N
        sequential calls.

        Rendering guidance is applied PER FRAME, to that frame's velocity,
        BEFORE fusion: each frame's grid is its own deformed geometry, so
        there is no single geometry at which a fused canonical velocity could
        be rendered.  ``stage2_mv`` guides the already-fused velocity instead,
        which is equivalent there because all views share one geometry.

        Args:
            frames: N frame payloads (``image``/``mask``), tokens_list order.
            canonical_coords: ``(L_canon, 4)`` canonical voxel coords.  MUST be
                the GT voxelisation the correspondence indexes into (i.e. the
                caller ran ``reset_shape_to_gt``) — ``pf_to_canon`` addresses
                its rows.
            perframe_entries: N dicts, tokens_list order, each with ``coords``
                ``(L_pf, 4)``, ``pf_to_canon`` ``(L_pf,)``, optional ``visibility``
                ``(L_pf,)`` (DDA 0/1; ``None`` -> ones), optional
                ``pixel_coords`` (attn-bias input on THAT frame's grid) and
                optional ``pose_decoder`` (rendering guidance).
            frame_chunk: how many frames share one batched backbone forward.
                Frames are partitioned into contiguous groups of this size, so a
                step costs ``2*ceil(N/K)`` forwards instead of ``2*N`` — the
                FLOPs are identical, what shrinks is per-call Python, kernel
                launches and per-step ``SparseTensor`` construction.  ``None``
                = one group (2 forwards/step); ``1`` evaluates frames one
                at a time.
                Attention is block-diagonal throughout, so K never changes the
                math — only float accumulation order and peak memory.  Works
                with the attention bias (hooks are keyed per GROUP and get a
                ragged, right-padded bias); only ``visibility_attn_debug``,
                whose capture is rectangular-only, degrades it to 1.

        Returns:
            ``(slat, entropy_data, ode_history)`` — same shape as
            ``stage2_mv`` so callers share the downstream path;
            ``entropy_data`` carries only guidance/snapshot payloads.
        """
        from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp
        from sam3d_objects.model.backbone.tdfy_dit.models.structured_latent_flow import (
            SLatFlowModel,
        )
        from tqdm import tqdm

        pipeline = self._pipeline
        N = len(frames)
        if N != len(perframe_entries):
            raise ValueError(
                f"stage2_dyn: {N} frames but "
                f"{len(perframe_entries)} per-frame entries"
            )

        # 1. Preprocess every frame, stack, embed conditions (as stage2_mv).
        preprocessed = []
        for f in frames:
            rgba = self.merge_mask_to_rgba(f["image"], f["mask"])
            preprocessed.append(
                pipeline.preprocess_image(rgba, pipeline.slat_preprocessor)
            )
        batched_input = {
            k: torch.cat([p[k] for p in preprocessed], dim=0)
            for k in preprocessed[0].keys()
        }
        device = batched_input["image"].device

        slat_generator = pipeline.models["slat_generator"]
        L_canon = canonical_coords.shape[0]
        prev_steps = slat_generator.inference_steps
        if inference_steps:
            slat_generator.inference_steps = inference_steps

        rg_on = rendering_guidance_active

        # Frames are evaluated in batched GROUPS: one ragged sparse batch per
        # group, so a step costs 2*ceil(N/K) backbone forwards instead of 2*N.
        # Groups are fixed for the whole solve — each group's bias hook builds
        # its bias lazily on the first forward and caches it, so drifting
        # membership would serve a stale bias.  The hooks are keyed per GROUP
        # (one live hook set per batched forward) and `per_view_L` gives that
        # set a ragged, right-padded bias.
        #
        # The bias `debug` capture reshapes the packed queries to
        # (N, L_down, ...), which has no ragged meaning, so the hook refuses it
        # with per_view_L set.  Degrade rather than raise — it is a viz aid.
        if (visibility_attn_bias_alpha > 0 and visibility_attn_debug
                and (frame_chunk is None or frame_chunk > 1)):
            print("    stage2_dyn: attn-bias debug capture is "
                  "rectangular-only -> frame_chunk=1 (batching off)")
            frame_chunk = 1
        groups = _frame_groups(N, frame_chunk)

        hooks_by_group: "list[list]" = [[] for _ in groups]
        rg_transforms: "list" = [None] * N
        rg_histories: "list" = [None] * N
        ode_history: list = []
        slat_snapshots: list = []

        try:
            with torch.autocast(device_type="cuda", dtype=pipeline.dtype):
                with torch.no_grad():
                    cond_embedded = _embed_slat_conditions(
                        pipeline, batched_input, N)      # (N, P, C)

                    wrapper = slat_generator.reverse_fn.backbone
                    cfg_strength = float(pipeline.slat_cfg_strength)
                    time_scale = slat_generator.time_scale

                    # 2. Per-frame coords / correspondence / visibility.
                    pf_coords, pf_to_canon, pf_vis = [], [], []
                    for b, entry in enumerate(perframe_entries):
                        _c = entry["coords"]
                        if not torch.is_tensor(_c):
                            _c = torch.as_tensor(_c)
                        _c = _c.to(device)
                        if _c.shape[1] == 3:  # add batch column
                            _c = torch.cat(
                                [torch.zeros((_c.shape[0], 1), dtype=_c.dtype,
                                             device=device), _c], dim=1)
                        else:
                            _c = _c.clone()
                            _c[:, 0] = 0
                        pf_coords.append(_c.to(torch.int32))

                        _m = torch.as_tensor(entry["pf_to_canon"]).to(
                            device=device, dtype=torch.long)
                        if int(_m.max()) >= L_canon or int(_m.min()) < 0:
                            raise ValueError(
                                f"stage2_dyn: frame {b} "
                                f"pf_to_canon indexes rows outside the "
                                f"canonical grid (max={int(_m.max())}, "
                                f"L_canon={L_canon}).  The canonical grid "
                                f"must be the GT voxelisation the "
                                f"correspondence was built against."
                            )
                        pf_to_canon.append(_m)

                        _v = entry.get("visibility")
                        _v = (torch.ones(_c.shape[0], device=device,
                                         dtype=torch.float32)
                              if _v is None
                              else torch.as_tensor(_v).to(
                                  device=device, dtype=torch.float32))
                        pf_vis.append(_v)

                    # 3. Ragged batch geometry per GROUP, built once — only the
                    # feats change per step.  `_ragged_sparse_batch` is the sole
                    # place the batch column is renumbered, and its offsets must
                    # equal the SparseTensor's own derived layout (asserted at
                    # step 0 below).
                    group_coords, group_offsets, group_cond = [], [], []
                    for members in groups:
                        _gc, _off = _ragged_sparse_batch(
                            [pf_coords[b] for b in members])
                        _gc = _gc.to(device)
                        group_coords.append(_gc)
                        group_offsets.append(_off)
                        # Contiguous members -> a VIEW, not a gather.
                        group_cond.append(
                            cond_embedded[members[0]:members[-1] + 1])
                    group_uncond = [torch.zeros_like(c) for c in group_cond]

                    if len(groups) < N:
                        print(f"    Consensus batching: {N} frames in "
                              f"{len(groups)} group(s) -> {2 * len(groups)} "
                              f"forwards/step (was {2 * N})")

                    # 4. Attention-bias hooks, one set per GROUP, installed ONCE
                    # and toggled per step.  Each group gets its OWN
                    # shared_state: that cache is unkeyed, so one dict across
                    # groups would silently serve the first group's bias to all.
                    if visibility_attn_bias_alpha > 0:
                        for g, members in enumerate(groups):
                            # Each entry is already a PER-VIEW LIST of
                            # {voxel: pixels} dicts (compute_visibility_multi_object
                            # builds ``pixel_coords[obj] = [{} per view]``) with
                            # exactly one view, so the group's masks are those
                            # lists concatenated in member order — NOT re-wrapped.
                            _pcs = [perframe_entries[b].get("pixel_coords")
                                    for b in members]
                            if not all(_pcs):
                                continue
                            _flat = [pc[0] for pc in _pcs]
                            _Ls = [int(pf_coords[b].shape[0]) for b in members]
                            _hooks = _install_vis_bias_hooks(
                                wrapper, [frames[b] for b in members], _flat,
                                _Ls[0] if len(members) == 1 else _Ls,
                                len(members),
                                alpha=visibility_attn_bias_alpha,
                                layers=visibility_attn_bias_layers,
                                streams=visibility_attn_bias_streams,
                                compensate_passive_streams=
                                    visibility_attn_bias_compensate_passive_streams,
                                debug=visibility_attn_debug,
                                verbose=(g == 0),
                            )
                            for _h in _hooks:
                                _h.set_active(False)
                            hooks_by_group[g] = _hooks
                        # Each group's bias is (K, 1, L_down_max, 5496) fp32,
                        # shared across that group's layers.  If it ever OOMs,
                        # the hook already does .to(device) on every call, so
                        # moving each _bias to CPU after the first step costs
                        # one H2D copy per forward and nothing else.
                        _n_hooked = sum(1 for h in hooks_by_group if h)
                        print(f"    Consensus attn bias: {_n_hooked}/"
                              f"{len(groups)} group(s) hooked, alpha="
                              f"{visibility_attn_bias_alpha}")

                    # 4. Per-frame rendering-guidance transforms, built once
                    # on each frame's OWN grid (already the deformed geometry,
                    # so no warp kwargs are needed here).
                    if rg_on:
                        from genia.core.rendering_guidance import (
                            build_appearance_rendering_guidance_transform,
                        )
                        if rendering_guidance_losses_cfg is None:
                            raise ValueError(
                                "rendering_guidance_active=True requires "
                                "rendering_guidance_losses_cfg (LossConfig)."
                            )
                        _bg = rendering_guidance_bg_color
                        if _bg is None:
                            _bg = torch.ones(3, device=device,
                                             dtype=torch.float32)
                        _schedule = _appearance_guidance_schedule(
                            rendering_guidance_active_from,
                            rendering_guidance_active_until,
                        )
                        # Constant across frames — hoisted so the loop body
                        # shows only what actually varies per frame: the
                        # conditioning image, its pose decoder, and its own
                        # (already-deformed) voxel grid.  No deformation-warp
                        # kwargs here, unlike stage2_mv: each frame's grid IS
                        # the deformed geometry, so there is nothing to warp.
                        _rg_common = dict(
                            pipeline=pipeline,
                            slat_mean=pipeline.slat_mean,
                            slat_std=pipeline.slat_std,
                            losses_cfg=rendering_guidance_losses_cfg,
                            velocity_weight=float(
                                rendering_guidance_velocity_weight),
                            schedule=_schedule,
                            bg_color=_bg,
                            device=device,
                            microbatch_size=rendering_guidance_microbatch_size,
                            resolution_scale=rendering_guidance_resolution_scale,
                            normalize_grad=rendering_guidance_normalize_grad,
                            decoder_autocast_bf16=
                                rendering_guidance_decoder_autocast_bf16,
                            random_background=
                                rendering_guidance_random_background,
                            random_background_seed=
                                rendering_guidance_random_background_seed,
                            visibility_detach=
                                rendering_guidance_visibility_detach,
                            visibility_depth_margin=
                                rendering_guidance_visibility_depth_margin,
                        )
                        for b, entry in enumerate(perframe_entries):
                            _pd = entry.get("pose_decoder")
                            if _pd is None:
                                raise ValueError(
                                    f"rendering_guidance_active=True requires a "
                                    f"pose_decoder for every frame; frame {b} "
                                    f"has none."
                                )
                            rg_transforms[b], rg_histories[b] = (
                                build_appearance_rendering_guidance_transform(
                                    frames=[frames[b]],
                                    # one frame per transform, so the loop index
                                    # inside is always 0 -- pass the REAL id or
                                    # every frame reports itself as "frame 0".
                                    frame_labels=[b],
                                    pose_decoder=_pd,
                                    canonical_coords=pf_coords[b],
                                    **_rg_common,
                                )
                            )
                        print(f"    Consensus rendering guidance: {N} "
                              f"per-frame transforms (applied pre-fusion)")

                    # 5. Consensus ODE.  One canonical latent, one noise draw
                    # (per-frame states are pure gathers of it, so the whole
                    # solve is determined by this seed).
                    t_seq = slat_generator._prepare_t().to(device)
                    if seed is not None:
                        torch.manual_seed(seed)
                    x = torch.randn((1, L_canon, 8), device=device)
                    unreached = None

                    pbar = tqdm(
                        list(zip(t_seq[:-1], t_seq[1:])),
                        desc=f"  Stage 2 consensus ({N} frames)",
                    )
                    # Progress readout follows one frame's guidance history
                    # (each is a complete trajectory over the steps).
                    _hist = next((h for h in rg_histories if h is not None), None)
                    for _step, (t0, t1) in enumerate(pbar):
                        t_val = float(t0)
                        dt = t1 - t0
                        t_tensor = torch.tensor(
                            [t_val * time_scale], device=device,
                            dtype=torch.float32,
                        )
                        vel_frames = []
                        for g, members in enumerate(groups):
                            # Gather every member's state out of the ONE
                            # canonical latent, then concatenate in member order
                            # — the same ascending order `group_coords[g]` was
                            # built in, which is what makes the offset split
                            # below address the right rows.
                            x_pf_list = [x[0, pf_to_canon[b]] for b in members]
                            feats = torch.cat(x_pf_list, dim=0)
                            _off = group_offsets[g]

                            for _h in hooks_by_group[g]:
                                _h.set_active(True)
                            x_sp = sp.SparseTensor(
                                feats=feats, coords=group_coords[g],
                            )
                            # Shape agreement, EVERY step: the batch element
                            # count of the latent, its conditioning and the
                            # group must be the same number, or attention pairs
                            # a frame with another frame's cond.  Three shape
                            # reads, no CPU sync — cheap enough not to gate on
                            # the step, and the failure it catches otherwise
                            # surfaces as a bare "Batch size mismatch" from
                            # inside the backbone with no way back to a group.
                            if not (x_sp.shape[0] == group_cond[g].shape[0]
                                    == len(members)):
                                raise RuntimeError(
                                    f"stage2_dyn step {_step} group {g}: batch "
                                    f"disagreement — latent {x_sp.shape[0]}, "
                                    f"cond {group_cond[g].shape[0]}, members "
                                    f"{len(members)} ({members}); feats "
                                    f"{tuple(feats.shape)}, coords "
                                    f"{tuple(group_coords[g].shape)}, offsets "
                                    f"{_off}"
                                )
                            if _step == 0:
                                # The batch-order invariant is only checked by
                                # the library under DEBUG (unusable here), so
                                # this is where it becomes loud: a silent
                                # violation would mis-split every velocity.
                                _want = [slice(a, b_) for a, b_
                                         in zip(_off, _off[1:])]
                                assert list(x_sp.layout) == _want, (
                                    f"stage2_dyn group {g}: SparseTensor layout "
                                    f"{list(x_sp.layout)} != offset split "
                                    f"{_want} — batch column is not ascending/"
                                    f"contiguous and the split is wrong."
                                )
                            y_cond = SLatFlowModel.forward(
                                wrapper, x_sp, t_tensor, group_cond[g], None,
                            )
                            for _h in hooks_by_group[g]:
                                _h.set_active(False)
                            # REBUILD rather than reuse: `_spatial_cache` is
                            # shared BY REFERENCE with every derived tensor and
                            # the input blocks write `upsample_*` into it.
                            x_sp_u = sp.SparseTensor(
                                feats=feats, coords=group_coords[g],
                            )
                            y_uncond = SLatFlowModel.forward(
                                wrapper, x_sp_u, t_tensor, group_uncond[g], None,
                            )
                            v_all = (
                                (1 + cfg_strength) * y_cond.feats
                                - cfg_strength * y_uncond.feats
                            )
                            for j, b in enumerate(members):
                                v_pf = v_all[_off[j]:_off[j + 1]]
                                if rg_transforms[b] is not None:
                                    # Per-observation guidance, BEFORE fusion.
                                    v_pf = rg_transforms[b](
                                        t_val, v_pf.unsqueeze(0),
                                        x_pf_list[j].unsqueeze(0),
                                    )[0]
                                # fp32 before the collapse: it sizes eps from
                                # the feature dtype, and bf16's eps (~7.8e-3)
                                # would distort the mass normalisation.
                                vel_frames.append((
                                    v_pf.float(), pf_vis[b],
                                    pf_to_canon[b],
                                ))

                        x, unreached = _consensus_fuse_step(
                            x, dt, vel_frames, canonical_coords,
                            visibility_alpha=visibility_alpha,
                            visibility_min_weight=visibility_min_weight,
                        )
                        _t1 = float(t1)
                        _s = x[0]
                        ode_history.append((_t1, {
                            "slat_mean": _s.mean(dim=0).detach().cpu().clone(),
                            "slat_norm_std": _s.norm(dim=-1).std()
                                .detach().cpu().unsqueeze(0).clone(),
                        }))
                        if capture_slat_snapshots:
                            slat_snapshots.append(
                                (_t1, _s.detach().cpu().clone()))
                        _post = {"t": f"{_t1:.2f}"}
                        if _hist:
                            _e = _hist[-1]
                            if _e.get("skip_reason"):
                                _post["skip"] = _e["skip_reason"]
                            else:
                                _post["rg"] = f"{_e['total']:.3f}"
                                _post["gn"] = f"{_e.get('grad_norm', 0.0):.2e}"
                        pbar.set_postfix(**_post)
                    pbar.close()

                    if unreached is not None:
                        _n_unreached = int(unreached.sum())
                        if _n_unreached:
                            print(f"    Consensus: {_n_unreached}/{L_canon} "
                                  f"canonical voxels reached by no frame "
                                  f"(filled with the mean fused velocity)")

                    slat = sp.SparseTensor(
                        coords=canonical_coords, feats=x[0],
                    ).to(device)
                    slat = (
                        slat * pipeline.slat_std.to(device)
                        + pipeline.slat_mean.to(device)
                    )
        finally:
            for _hooks in hooks_by_group:
                for _h in _hooks:
                    _h.remove()
            slat_generator.inference_steps = prev_steps

        entropy_data = None
        _rg_hist = [h for h in rg_histories if h]
        if _rg_hist:
            # One history per frame; the shared loss plot takes a single ODE
            # trajectory, so hand it frame 0's (each is complete over the steps).
            entropy_data = {"rendering_guidance_loss_history": _rg_hist[0]}
        if slat_snapshots:
            entropy_data = entropy_data or {}
            entropy_data["slat_ode_snapshots"] = slat_snapshots
        return slat, entropy_data, ode_history

    def __call__(
        self,
        image: Union[Image.Image, np.ndarray],
        mask: Optional[Union[None, Image.Image, np.ndarray]],
        seed: Optional[int] = None,
        pointmap=None,
        stage1_only: bool = False,
    ) -> dict:
        image = self.merge_mask_to_rgba(image, mask)
        return self._pipeline.run(
            image,
            None,
            seed,
            stage1_only=stage1_only,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            with_layout_postprocess=True,
            use_vertex_color=True,
            stage1_inference_steps=None,
            pointmap=pointmap,
        )


def check_target(
    target: str,
    whitelist_filters: List[Callable],
    blacklist_filters: List[Callable],
):
    if any(filt(target) for filt in whitelist_filters):
        if not any(filt(target) for filt in blacklist_filters):
            return
    raise RuntimeError(
        f"target '{target}' is not allowed to be hydra instantiated, if this is a mistake, please do modify the whitelist_filters / blacklist_filters"
    )


def check_hydra_safety(
    config: DictConfig,
    whitelist_filters: List[Callable],
    blacklist_filters: List[Callable],
):
    to_check = [config]
    while len(to_check) > 0:
        node = to_check.pop()
        if isinstance(node, DictConfig):
            to_check.extend(list(node.values()))
            if "_target_" in node:
                check_target(node["_target_"], whitelist_filters, blacklist_filters)
        elif isinstance(node, ListConfig):
            to_check.extend(list(node))


