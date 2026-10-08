"""
Sequence data loading and caching for the SAM3D-Objects pipeline.

This module provides the Sequence class that loads all frame data (images,
masks, depth, pointmaps, intrinsics) once at construction time and provides
cached access throughout the pipeline, eliminating redundant disk I/O.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .depth import depth_to_pointmap, load_and_process_depth
from .frame_key import FrameKey, as_frame_key
from .io_utils import MaskDict, group_palette_masks, load_image, load_masks, setup_paths
from .recon_cache import GT_K_DATASETS


MA_ALL_TRAIN_VIEWS_DATASETS = ("gso", "mvcustom", "co3d")
"""MV-static datasets where map-anything additionally sees the views the pipeline
does NOT consume, for a robust pairwise scale anchor."""


@dataclass
class FrameData:
    """Cached data for a single frame.

    Attributes
    ----------
    image : np.ndarray
        RGB image, shape (H, W, 3), dtype uint8.
    masks : MaskDict
        Segmentation masks mapping object ID -> bool mask (H, W).
        Returns zero mask for missing object IDs.
    K_matrix : np.ndarray
        Camera intrinsics matrix, shape (3, 3).
    pointmap : np.ndarray
        3D pointmap in R3 convention (X-right, Y-down, Z-forward),
        shape (H, W, 3).
    depth_map_z : np.ndarray
        Z-depth map, shape (H, W), dtype float32.
    valid_mask : np.ndarray or None
        Boolean mask of valid depth pixels, shape (H, W).
        None for GT depth, set for predicted depth.
    normals_map : np.ndarray or None
        Surface normals from MoGe, shape (H, W, 3), float32.
        None when GT depth is used or MoGe lacks a normal head.
    c2w : np.ndarray
        Camera-to-world transform, shape (4, 4), dtype float32.
        Defaults to identity (camera at origin).
        When non-identity, object poses are in world space and rendering
        uses this transform to project world-space Gaussians into the camera.
    """

    image: np.ndarray
    masks: MaskDict
    K_matrix: np.ndarray
    pointmap: np.ndarray
    depth_map_z: np.ndarray
    valid_mask: Optional[np.ndarray]
    normals_map: Optional[np.ndarray] = None
    c2w: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32))


def crop_resize_transform(orig_shape, target_shape):
    """The spatial transform map-anything's ``crop_resize_if_necessary`` applies —
    scale-to-cover the target, then center-crop — returned as explicit params.

    Single source of truth for mapping between original-image and processed-image
    pixel coords (used both to crop the masks here AND, recorded into the artifact,
    to un-crop the exported 2D tracks during evaluation).  Returns
    ``(scale, left, top, scaled_w, scaled_h)``; a processed pixel ``(u, v)`` maps
    back to the original image as ``((u + left) / scale, (v + top) / scale)``.
    """
    orig_h, orig_w = orig_shape
    target_h, target_w = target_shape
    scale = max(target_w / orig_w, target_h / orig_h) + 1e-8
    scaled_w = int(np.floor(orig_w * scale))
    scaled_h = int(np.floor(orig_h * scale))
    left = (scaled_w - target_w) // 2
    top = (scaled_h - target_h) // 2
    return scale, left, top, scaled_w, scaled_h


#: map-anything's `fixed_mapping` resolution tables, VENDORED from
#: `mapanything/utils/image.py`.  Copied rather than imported because reaching that
#: module pulls in `uniception.models.encoders.image_normalizations`, which sets
#: `torch.backends.cuda.matmul.allow_tf32 = True` process-wide at import -- the leak
#: `Sequence._load_with_map_anything` scopes to its own call.  Every backend sizes its input through
#: this table, so importing it would put that leak on the MoGe path too.
_RESOLUTION_MAPPINGS = {
    518: {1.000: (518, 518), 1.321: (518, 392), 1.542: (518, 336), 1.762: (518, 294),
          2.056: (518, 252), 3.083: (518, 168), 0.757: (392, 518), 0.649: (336, 518),
          0.567: (294, 518), 0.486: (252, 518)},
    512: {1.000: (512, 512), 1.333: (512, 384), 1.524: (512, 336), 1.778: (512, 288),
          2.000: (512, 256), 3.200: (512, 160), 0.750: (384, 512), 0.656: (336, 512),
          0.562: (288, 512), 0.500: (256, 512)},
    504: {1.000: (504, 504), 1.333: (504, 378), 1.565: (504, 322), 1.800: (504, 280),
          2.118: (504, 238), 3.273: (504, 154), 0.750: (378, 504), 0.639: (322, 504),
          0.556: (280, 504), 0.472: (238, 504)},
}


def fixed_mapping_size(aspect_ratio: float, resolution_set: int = 518):
    """``(width, height)`` the model input is snapped to, map-anything's rule.

    The long side is always ``resolution_set``; the short side comes from whichever of
    ten fixed aspect ratios is nearest, so the image is center-cropped to that ratio
    rather than squeezed.  Every entry is a multiple of 14 (patch-aligned).

    Note this does not only downscale: an input smaller than ``resolution_set`` on its
    long side is scaled UP.
    """
    table = _RESOLUTION_MAPPINGS[resolution_set]
    return table[min(table, key=lambda k: abs(k - aspect_ratio))]


def _crop_resize_image(image: np.ndarray, target_shape: tuple, interp=None) -> np.ndarray:
    """Scale-to-cover + center-crop ``image`` to ``(target_h, target_w)``.

    The image half of :func:`_crop_masks_to_match`; both read their geometry from
    :func:`crop_resize_transform`, so they cannot disagree about where the crop fell.
    ``interp`` overrides the scale-chosen interpolator for data that must not be
    blended -- GT depth passes ``cv2.INTER_NEAREST``, because averaging across a depth
    discontinuity fabricates a surface that exists at no pixel.
    """
    import cv2

    target_h, target_w = target_shape
    scale, left, top, scaled_w, scaled_h = crop_resize_transform(
        image.shape[:2], target_shape)
    if interp is None:
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (scaled_w, scaled_h), interpolation=interp)
    return resized[top:top + target_h, left:left + target_w]


def crop_resize_intrinsics(K: np.ndarray, orig_shape: tuple,
                           target_shape: tuple) -> np.ndarray:
    """``K`` moved from the original image frame into the cropped/resized one.

    :func:`crop_resize_transform` maps a processed pixel back as ``(u + left) / scale``,
    so forward::

        fx' = s*fx      cx' = s*cx - left
        fy' = s*fy      cy' = s*cy - top

    Used for map-anything's GT-K override.
    """
    scale, left, top, _sw, _sh = crop_resize_transform(orig_shape, target_shape)
    out = np.asarray(K, dtype=np.float32).copy()
    out[0, 0] *= scale
    out[1, 1] *= scale
    out[0, 2] = out[0, 2] * scale - left
    out[1, 2] = out[1, 2] * scale - top
    return out


#: Backends that predict camera poses as well as depth, and so share every gate that
#: exists because a pose prediction has to be captured, anchored or pruned. MoGe and the
#: GT path leave c2w at identity.
POSE_PREDICTING_MODELS = ("map_anything",)

#: Backends that CONSUME a GT signal (poses / depth / intrinsics) as a prior, i.e. for
#: which `condition_recon_model_on_gt_{poses,depths}` mean anything.
CONDITIONING_CAPABLE_MODELS = ("map_anything",)


def _crop_masks_to_match(
    masks: "MaskDict",
    orig_shape: tuple,
    target_shape: tuple,
) -> "MaskDict":
    """Apply the same center-crop + resize that map-anything uses to masks.

    Replicates the spatial transform from ``crop_resize_if_necessary`` (scale
    to cover target, then center-crop) so that masks align with the
    cropped/resized image and depth.
    """
    import cv2

    target_h, target_w = target_shape
    scale, left, top, scaled_w, scaled_h = crop_resize_transform(
        orig_shape, target_shape)

    cropped_data = {}
    for obj_id, mask in masks.items():
        # Resize to scaled size (nearest to keep binary)
        resized = cv2.resize(
            mask.astype(np.uint8), (scaled_w, scaled_h),
            interpolation=cv2.INTER_NEAREST,
        )
        # Center-crop
        cropped = resized[top:top + target_h, left:left + target_w]
        cropped_data[obj_id] = cropped.astype(bool)
    return MaskDict(cropped_data, shape=(target_h, target_w))


class Sequence:
    """Scene data wrapper that loads all frames once and caches them.

    This class eliminates redundant disk I/O by loading all frame data
    (images, masks, depth maps, pointmaps, intrinsics) at construction
    time. Downstream functions access cached data via ``sequence[frame_idx]``.

    Parameters
    ----------
    dataset_path : str
        Root path to the dataset.
    scene_name : str
        Name of the scene to process.
    dataset_type : str
        Dataset name (``dataset.name``), e.g. ``"gso"`` or ``"dyncustom"``.
    frame_indices : list of int
        Frame indices to load.
    reconstruction_model : str or None, optional
        Depth/pose model: ``"moge"`` (per-frame depth, identity poses),
        ``"map_anything"`` (multi-view depth + predicted poses), or ``None``
        (no model; only valid with ``depth_source="gt"``). Default: ``"moge"``.
    depth_source : str, optional
        ``"gt"`` = dataset-provided depth, ``"pred"`` = predicted by
        ``reconstruction_model``. Default: ``"pred"``.
    camera_poses_source : str, optional
        ``"gt"`` = dataset poses (raises if absent), ``"pred"`` = from
        ``reconstruction_model`` (moge/None→identity, map_anything→predicted).
        Default: ``"pred"``.

    Examples
    --------
    >>> seq = Sequence("data/davis_actionmesh", "camel", "davis_actionmesh", [0, 10, 20])
    >>> frame = seq[10]
    >>> frame.image.shape
    (480, 854, 3)
    >>> frame.K_matrix.shape
    (3, 3)
    >>> 10 in seq
    True
    >>> len(seq)
    3
    """

    def __init__(
        self,
        dataset_path: str,
        scene_name: str,
        dataset_type: str,
        frame_indices: List[int],
        reconstruction_model: Optional[str] = "moge",
        depth_source: str = "pred",
        camera_poses_source: str = "pred",
        condition_recon_model_on_gt_poses: bool = False,
        condition_recon_model_on_gt_depths: bool = False,
        downscale_factor: int = 1,
        median_intrinsics: bool = False,
        fps: float = 24.0,
        num_input_views: Optional[int] = None,
        world_origin_is_object: bool = False,
        view_indices: Optional[List[int]] = None,
        asset_indices: Optional[List[int]] = None,
        recon_on_test_views: bool = True,
        cache_reconstruction: bool = False,
        recon_resolution_set: int = 518,
        recon_mask_edges: bool = True,
        recon_confidence_min: Optional[float] = None,
    ):
        # Phase-A validation: type/consistency of the sourcing config. All
        # raise (always-strict). GT-availability (Phase B) is checked at load
        # time below, once dataset paths are known.
        if depth_source not in ("gt", "pred"):
            raise ValueError(
                f"depth_source must be 'gt' or 'pred'; got {depth_source!r}")
        if camera_poses_source not in ("gt", "pred"):
            raise ValueError(
                f"camera_poses_source must be 'gt' or 'pred'; got {camera_poses_source!r}"
            )
        if recon_resolution_set not in _RESOLUTION_MAPPINGS:
            raise ValueError(
                f"recon_resolution_set must be one of "
                f"{sorted(_RESOLUTION_MAPPINGS)}; got {recon_resolution_set!r}."
            )
        if recon_resolution_set == 512:
            raise ValueError(
                "recon_resolution_set=512 is not selectable: 512 is not a multiple of "
                "the 14-px patch size. Use 518 (default) or 504."
            )
        if reconstruction_model not in ("moge", "map_anything", None):
            raise ValueError(
                f"reconstruction_model must be 'moge', 'map_anything', or None; "
                f"got {reconstruction_model!r}"
            )
        if depth_source == "pred" and reconstruction_model is None:
            raise ValueError(
                "depth_source='pred' requires a reconstruction_model (got None) -- "
                "cannot predict depth with no model. Use depth_source='gt'."
            )
        if depth_source == "gt" and reconstruction_model in POSE_PREDICTING_MODELS:
            raise ValueError(
                f"depth_source='gt' with reconstruction_model="
                f"{reconstruction_model!r} is not supported (it predicts depth; its "
                f"output path would discard the GT depth). Use "
                f"reconstruction_model='moge'/None for GT depth, or "
                f"depth_source='pred' to use the model's depth."
            )
        # Which backends can be TOLD a signal, which is not the same question as which
        # ones predict one -- see `CONDITIONING_CAPABLE_MODELS`.
        _cond_opts = [name for name, on in
                      (("condition_recon_model_on_gt_poses",
                        condition_recon_model_on_gt_poses),
                       ("condition_recon_model_on_gt_depths",
                        condition_recon_model_on_gt_depths)) if on]
        if _cond_opts and reconstruction_model not in CONDITIONING_CAPABLE_MODELS:
            raise ValueError(
                f"{' and '.join(_cond_opts)}=True requires one of "
                f"{' / '.join(CONDITIONING_CAPABLE_MODELS)} (the backends that consume "
                f"a conditioning input); got {reconstruction_model!r}."
            )
        # Output filtering is a property of predicting depth at all, so it is offered by
        # exactly the backends that do -- and MoGe/None, which load or estimate per
        # frame, emit neither an edge test nor a confidence to threshold.
        _filter_opts = [name for name, changed in
                        (("recon_confidence_min", recon_confidence_min is not None),
                         ("recon_mask_edges=False", not recon_mask_edges)) if changed]
        if _filter_opts and reconstruction_model not in POSE_PREDICTING_MODELS:
            raise ValueError(
                f"{', '.join(_filter_opts)} requires one of "
                f"{POSE_PREDICTING_MODELS}; got {reconstruction_model!r}, which emits "
                f"no confidence to threshold and no geometry to run an edge test over."
            )

        # Internal bool derived from depth_source; only ever True with moge/None
        # (a pose-predicting backend + gt is rejected above), i.e. the _load_frame path. The
        # depth.py leaf keys on this.
        use_dataset_depth = depth_source == "gt"

        self.dataset_path = dataset_path
        self.scene_name = scene_name
        self.dataset_type = dataset_type
        self.cache_reconstruction = cache_reconstruction
        self.recon_resolution_set = recon_resolution_set
        self.recon_mask_edges = recon_mask_edges
        self.recon_confidence_min = recon_confidence_min
        # Two-axis indexing: frame_indices and view_indices are parallel lists
        # of the same length; their pair (frame, view) becomes a FrameKey.
        # asset_indices is the file-position into self.paths[*_names] arrays;
        # for mono datasets it equals frame_indices. For
        # MV-static datasets (post-axis-lift) it equals view_indices.
        if view_indices is None:
            view_indices = [0] * len(frame_indices)
        if asset_indices is None:
            asset_indices = list(frame_indices)
        if len(view_indices) != len(frame_indices):
            raise ValueError(
                f"view_indices length {len(view_indices)} != "
                f"frame_indices length {len(frame_indices)}"
            )
        if len(asset_indices) != len(frame_indices):
            raise ValueError(
                f"asset_indices length {len(asset_indices)} != "
                f"frame_indices length {len(frame_indices)}"
            )
        # Sort triples by (view, frame) so per-view contiguity is preserved
        # (mono data, view=0 throughout, stays frame-sorted).
        _triples = sorted(
            zip(frame_indices, view_indices, asset_indices),
            key=lambda t: (t[1], t[0]),
        )
        self._frame_keys: List[FrameKey] = [FrameKey(f, v) for f, v, _ in _triples]
        self._asset_indices: List[int] = [a for _, _, a in _triples]
        self._fk_to_asset: Dict[FrameKey, int] = {
            fk: a for fk, a in zip(self._frame_keys, self._asset_indices)
        }
        self._asset_to_fk: Dict[int, FrameKey] = {
            a: fk for fk, a in zip(self._frame_keys, self._asset_indices)
        }
        # `frame_indices` holds FrameKeys: callers iterating
        # `for fi in sequence.frame_indices` index `sequence[fk]` natively.
        # For MV-static data it yields distinct FrameKey(0, vi) entries, where
        # bare ints would silently alias every view to view 0.
        self.frame_indices = list(self._frame_keys)
        self.downscale_factor = downscale_factor
        self.fps = fps
        self.reconstruction_model = reconstruction_model
        self.camera_poses_source = camera_poses_source
        self.condition_recon_model_on_gt_poses = condition_recon_model_on_gt_poses
        self.condition_recon_model_on_gt_depths = condition_recon_model_on_gt_depths
        self.world_origin_is_object = world_origin_is_object
        _setup_kw = {}
        if num_input_views is not None:
            _setup_kw["num_input_views"] = num_input_views
        self.paths = setup_paths(dataset_path, scene_name, dataset_type, **_setup_kw)

        # Frames to process with map-anything.  Normally equals frame_indices,
        # but for MV-static benchmarks (gso, mvcustom, co3d) we always run map-anything
        # on the full available view set so that pairwise inter-camera
        # distances give a robust depth-scale anchor -- even in the
        # single-view setting where frame_indices has only 1 entry.  Extra
        # frames are pruned from self._frames after GT-pose alignment.
        # _ma_frame_indices stores ASSET indices (positions into paths arrays).
        # _ma_frame_keys is the parallel FrameKey list used for self._frames
        # dict access. For asset indices not in user's _asset_to_fk (extras
        # for MA), synthesize FrameKey(asset, 0); this is collision-free for
        # both mono (user keys are FrameKey(t, 0), so asset must already be in
        # the user's set) and MV-static (user keys are FrameKey(0, view), so
        # extras at asset≠0 don't collide and the asset==0 extra only exists
        # when view 0 is not in the user's subset).
        # Note: when both axes vary, MA-extra FrameKey synthesis would need the
        # loader to provide an explicit mapping.
        if (dataset_type in MA_ALL_TRAIN_VIEWS_DATASETS
                and reconstruction_model in POSE_PREDICTING_MODELS
                and recon_on_test_views):
            n_train = len(self.paths["image_names"])
            self._ma_frame_indices = sorted(
                set(range(n_train)) | set(self._asset_indices)
            )
        else:
            self._ma_frame_indices = list(self._asset_indices)
        self._ma_frame_keys: List[FrameKey] = [
            self._asset_to_fk.get(a, FrameKey(a, 0)) for a in self._ma_frame_indices
        ]

        # Determine depth source: MoGe when depth_source='pred' with a moge
        # reconstruction model. GT depth (depth_source='gt') only reaches the
        # _load_frame path (map_anything+gt is rejected in Phase A).
        has_gt_depth = bool(self.paths.get("depth_names"))
        # Phase-B GT-depth existence (always-strict, no fallback).
        if use_dataset_depth and not has_gt_depth:
            raise FileNotFoundError(
                f"depth_source='gt' but dataset {dataset_type!r} scene "
                f"{scene_name!r} has no GT depth files on disk. Render them "
                f"or use depth_source='pred'."
            )
        if self.condition_recon_model_on_gt_depths and not has_gt_depth:
            raise FileNotFoundError(
                f"condition_recon_model_on_gt_depths=True but dataset "
                f"{dataset_type!r} scene {scene_name!r} has no GT depth files to "
                f"condition map-anything on."
            )
        if reconstruction_model in POSE_PREDICTING_MODELS:
            # provides its own depth; uses_moge_depth is False.
            self.uses_moge_depth = False
        else:
            self.uses_moge_depth: bool = not (use_dataset_depth and has_gt_depth)

        # Load all frames eagerly
        self._frames: Dict[FrameKey, FrameData] = {}
        # MA-predicted (c2w, K) per ASSET index, populated for every view MA
        # was run on (including any extras pruned from self._frames). Lets
        # downstream code (e.g. CO3D NVS export) recover the predicted camera
        # of held-out target views that don't live in self._frames.
        # Always present (empty dict when MA not run).
        self._ma_predicted_cameras: Dict[int, Dict[str, np.ndarray]] = {}
        ds_tag = f", downscale={downscale_factor}x" if downscale_factor > 1 else ""
        depth_tag = f", model={reconstruction_model}" if reconstruction_model != "moge" else ""
        ma_tag = (
            f" ({reconstruction_model} on {len(self._ma_frame_indices)})"
            if len(self._ma_frame_indices) != len(self._frame_keys)
            else ""
        )
        print(f"\nLoading {len(self._frame_keys)} frames into Sequence{ds_tag}{depth_tag}{ma_tag}...")

        if (dataset_type == "co3d"
              and reconstruction_model not in POSE_PREDICTING_MODELS):
            # Depth-free path: load image / c2w / K from the flat per-view
            # files, leave depth=0. A pose-predicting backend falls through to its
            # own loader below — it uses the same flat files via
            # _load_image_and_masks / _load_dataset_camera_poses.
            self._load_from_co3d_flat()
        elif reconstruction_model == "map_anything":
            self._load_with_map_anything()
        else:
            for fk, asset_idx in zip(self._frame_keys, self._asset_indices):
                self._frames[fk] = self._load_frame(asset_idx, use_dataset_depth)

        # Optionally replace per-frame intrinsics with median across keyframes.
        # Both MoGe and map-anything predict per-frame K that drifts slightly for
        # a physically fixed camera (map-anything's joint inference correlates the
        # per-view K but does not force them identical); medianing collapses that
        # to a shared K. Skipped only when depth_source='gt' (K already fixed).
        if median_intrinsics:
            if use_dataset_depth:
                import warnings
                warnings.warn(
                    "median_intrinsics is ignored when depth_source='gt' "
                    "(GT depth already has fixed intrinsics).",
                    stacklevel=2,
                )
            else:
                self._apply_median_intrinsics()

        # camera_poses_source='gt': write dataset/GT poses into FrameData.c2w
        # (Phase B: raises if the dataset has none -- always strict).
        #
        # For map-anything (whose depth is necessarily 'pred' here), capture the
        # predicted poses *before* overwriting so _align_depth_scale_to_gt_poses
        # can anchor MA's depth to the GT camera metric: map-anything outputs
        # cam_trans and depth scaled by a shared `scale_final_output` (see
        # mapanything/model.py:1789-1790), so the ratio of GT to predicted
        # inter-camera distances is the correct depth multiplier. moge/None have
        # no predicted c2w, so there's nothing to anchor -- realign is MA-only.
        # The realign keys on the *unconditioned* MA poses; conditioning MA on
        # GT poses (condition_recon_model_on_gt_poses) is a soft prior that still
        # leaves a usable scale ratio.
        if camera_poses_source == "gt":
            ma_poses = None
            if reconstruction_model in POSE_PREDICTING_MODELS:
                # ma_poses keyed by ASSET index (used as the dict key in
                # _align_depth_scale_to_gt_poses + _ma_frame_indices iteration).
                # Read from _ma_predicted_cameras, not FrameData.c2w: under 'gt'
                # _finalize_recon_frame leaves the frame at identity, and identity
                # baselines would make the realign a silent no-op.
                ma_poses = {
                    asset_idx: self._ma_predicted_cameras[asset_idx]["c2w"].copy()
                    for asset_idx in self._ma_frame_indices
                }
            self._apply_dataset_camera_poses()
            if ma_poses is not None:
                self._align_depth_scale_to_gt_poses(ma_poses)

        # Prune loaded-but-unused frames (MV-static benchmarks run
        # map-anything on the full view set for a robust pairwise scale
        # anchor, but only keep the user-requested subset in self._frames).
        _user_fk_set = set(self._frame_keys)
        if set(self._ma_frame_keys) != _user_fk_set:
            self._frames = {fk: self._frames[fk] for fk in self._frame_keys}

        # Cache dimensions from first frame
        first = next(iter(self._frames.values()))
        self.H, self.W = first.image.shape[:2]
        # Original pre-crop image size — set by _load_with_map_anything when it
        # crops; identity (== processed size) otherwise.  Recorded into the
        # 2D-track artifact so evaluation un-crops from stored provenance rather
        # than re-deriving a preprocessing-specific crop.
        if getattr(self, "orig_hw", None) is None:
            self.orig_hw = (int(self.H), int(self.W))

        ds_info = f", downscaled {self.downscale_factor}x" if self.downscale_factor > 1 else ""
        print(f"Sequence loaded: {len(self._frames)} frames cached, "
              f"image size={self.W}x{self.H}{ds_info}")

    # ------------------------------------------------------------------
    # Flat-file loading (co3d LaRa)
    # ------------------------------------------------------------------

    def _load_from_co3d_flat(self) -> None:
        """Load views from a flat LaRa-format CO3D scene dir (per-view PNG +
        c2w .npy, plus a single ``fovs.npy`` and ``groups.json`` per scene).

        Depth-free path: depth/pointmap stay zero placeholders so downstream
        consumers that need depth (background Gaussians, render-init) must
        run map-anything / MoGe on top. For map-anything end-to-end, set
        ``processing=map_anything`` and the dispatch routes through
        ``_load_with_map_anything`` instead of this method.
        """
        frames_path = self.paths["frames_path"]
        camera_pose_names = self.paths["camera_pose_names"]
        fovs = self._co3d_fovs()
        print(f"  Loading from flat layout: {frames_path}")

        # LaRa c2w → R3 (world +Y-up → -Y-up; camera axes already R3); same
        # conversion the held-out test cameras get (single entry point).
        from .eval_assets_export import co3d_k_from_fov, load_co3d_lara_c2w
        for fk, asset_idx in zip(self._frame_keys, self._asset_indices):
            image, masks = self._load_image_and_masks(asset_idx)
            H, W = image.shape[:2]
            K = co3d_k_from_fov(fovs[asset_idx], H, W)
            c2w_path = os.path.join(frames_path, camera_pose_names[asset_idx])
            c2w = load_co3d_lara_c2w(c2w_path)

            depth_z = np.zeros((H, W), dtype=np.float32)
            pointmap = np.zeros((H, W, 3), dtype=np.float32)
            if self.downscale_factor > 1:
                image, masks, K, depth_z, pointmap, _, _ = (
                    self._apply_downscaling(
                        image, masks, K, depth_z, pointmap, None, None,
                    )
                )
            self._frames[fk] = FrameData(
                image=image, masks=masks, K_matrix=K,
                pointmap=pointmap, depth_map_z=depth_z, valid_mask=None,
                c2w=c2w,
            )

    def _model_input_size(self, aspect_ratio: float) -> Tuple[int, int]:
        """``(H, W)`` the reconstruction backend is fed, for ``aspect_ratio``.

        One rule for both backends (map-anything and MoGe), so a run can be attributed
        to the model rather than to the resolution it happened to see. map-anything is
        where the rule comes from (`preprocess_inputs` snaps internally).
        """
        tw, th = fixed_mapping_size(aspect_ratio, self.recon_resolution_set)
        return th, tw

    def _crop_resize_to_model_input(
        self, image: np.ndarray, masks: MaskDict, target_hw: Tuple[int, int],
    ) -> Tuple[np.ndarray, MaskDict]:
        """``image`` and ``masks`` center-cropped + resized to ``target_hw`` together.

        A no-op when they are already that size, which is the common case once
        map-anything's own preprocessing has run.
        """
        orig_shape = image.shape[:2]
        if orig_shape == tuple(target_hw):
            return image, masks
        return (_crop_resize_image(image, target_hw),
                _crop_masks_to_match(masks, orig_shape, target_hw))

    # ------------------------------------------------------------------
    # Image + mask loading (shared by both depth paths)
    # ------------------------------------------------------------------

    def _co3d_fovs(self) -> np.ndarray:
        """Lazy-load + cache the per-view FoV table (``fovs.npy``, shape (N, 2),
        radians) of a LaRa-format CO3D scene."""
        if not hasattr(self, "_co3d_fovs_cached"):
            self._co3d_fovs_cached = np.load(self.paths["fovs_path"]).astype(np.float32)
        return self._co3d_fovs_cached

    def _load_image_and_masks(
        self, frame_idx: int
    ) -> Tuple[np.ndarray, MaskDict]:
        """Load image and segmentation masks for a single frame."""
        paths = self.paths

        image_path = os.path.join(paths["frames_path"], paths["image_names"][frame_idx])
        mask_path = os.path.join(paths["masks_path"], paths["mask_names"][frame_idx])

        image = load_image(image_path, to_rgb=True)
        # RGBA images: composite on white background before stripping alpha
        if image.ndim == 3 and image.shape[2] == 4:
            alpha = image[:, :, 3:4].astype(np.float32) / 255.0
            rgb = image[:, :, :3].astype(np.float32)
            image = (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
        else:
            image = image[..., :3]
        masks = load_masks(mask_path)
        # ActionMesh: collapse raw palette IDs into per-object union masks per
        # the scene's metadata.json (e.g. DAVIS labels [1,2] -> one object).
        palette_groups = self.paths.get("palette_groups")
        if palette_groups is not None:
            masks = group_palette_masks(masks, palette_groups)
        return image, masks

    # ------------------------------------------------------------------
    # Dataset depth loading (for map-anything conditioning)
    # ------------------------------------------------------------------

    def _load_dataset_depth_for_frame(
        self, frame_idx: int, W: int, H: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Load dataset GT depth and intrinsics for a frame.

        Returns ``(depth_map_z, K_matrix)`` or ``None`` if unavailable.
        """
        paths = self.paths
        if self.dataset_type not in ("gso", "oursactionbench") or not paths.get("depth_names"):
            return None

        depth_frames_path = paths["frames_path"]
        depth_names = [paths["depth_names"][frame_idx]]

        # Reuse existing GT depth loading (returns pointmap, K, mask, depth, normals)
        _, K_matrix, _, depth_map_z, _ = load_and_process_depth(
            depth_frames_path,
            depth_names,
            W, H,
            use_dataset_depth=True,
            dataset_type=self.dataset_type,
        )
        return depth_map_z, K_matrix

    # ------------------------------------------------------------------
    # Conditioning tables, shared by every backend that can consume them
    # ------------------------------------------------------------------
    # All three are gathered at ORIGINAL image resolution, because that is the frame
    # the dataset states them in. map-anything is handed native pixels and snaps
    # internally, so it uses them as-is.

    def _gt_depth_table(self, orig_h: int, orig_w: int):
        """``({idx: depth_z}, {idx: K})`` from the dataset's own depth, or two Nones.

        The pair travels together because a depth map is only usable as a conditioning
        input alongside the K it was measured with.
        """
        depths, intrinsics = {}, {}
        for idx in self._ma_frame_indices:
            result = self._load_dataset_depth_for_frame(idx, orig_w, orig_h)
            if result is not None:
                depths[idx], intrinsics[idx] = result
        if not depths:
            raise FileNotFoundError(
                "condition_recon_model_on_gt_depths=True but no per-frame GT "
                f"depth could be loaded for {self.dataset_type!r} scene "
                f"{self.scene_name!r}."
            )
        return depths, intrinsics

    def _gt_intrinsics_table(self, orig_h: int, orig_w: int):
        """``{idx: K}`` for the datasets whose true K is known analytically, else None.

        A fixed, required signal rather than a toggle: a predicted FoV drifts noticeably
        on GSO renders, and every GT object pose downstream was derived for the true K,
        so a render of a GT-posed prediction would land at the wrong size. GSO is a
        Blender pinhole (fx=fy=35*W/32); CO3D LaRa bakes a per-view focal into a square
        512² frame (`fovs.npy`); OursActionBench fits one per scene (`camera.json`).

        Which datasets those are is `recon_cache.GT_K_DATASETS`, read rather than
        restated.
        """
        if self.dataset_type not in GT_K_DATASETS:
            return None
        if self.dataset_type == "gso":
            from .eval_assets_export import gso_blender_intrinsics
            K_gt = gso_blender_intrinsics(orig_h, orig_w)
            return {idx: K_gt for idx in self._ma_frame_indices}
        if self.dataset_type == "co3d":
            from .eval_assets_export import co3d_k_from_fov
            fovs = self._co3d_fovs()
            return {idx: co3d_k_from_fov(fovs[idx], orig_h, orig_w)
                    for idx in self._ma_frame_indices}
        if self.dataset_type == "oursactionbench":
            from genia.core.utils.gt_data import actionbench_intrinsics
            K_gt = actionbench_intrinsics(
                orig_h, orig_w, scene_name=self.scene_name,
                data_root=self.dataset_path,
            )
            return {idx: K_gt for idx in self._ma_frame_indices}
        raise AssertionError(f"{self.dataset_type} is in GT_K_DATASETS with no builder")

    def _conditioning_tables(self, orig_h: int, orig_w: int):
        """``(depths, intrinsics, poses)`` at ORIGINAL resolution, or Nones.

        The precedence rule: GT depth supplies the K it was measured with, and the
        analytic K fills in only when it did not. map-anything uses these as-is (it
        snaps internally).
        """
        depths = intrinsics = None
        if self.condition_recon_model_on_gt_depths:
            depths, intrinsics = self._gt_depth_table(orig_h, orig_w)
        if intrinsics is None:
            intrinsics = self._gt_intrinsics_table(orig_h, orig_w)
        return depths, intrinsics, self._conditioning_camera_poses()

    # ------------------------------------------------------------------
    # Map-anything path (multi-view)
    # ------------------------------------------------------------------

    def _recon_cache_dir(self, subdir: str) -> Optional[str]:
        """Where this scene's cached map-anything depth lives, or None when off.

        Beside the scene it describes, so it is deleted with the scene and never
        outlives the images it was computed from. The root comes from
        ``paths["data_path"]``, the per-scene root; were it ever a shared dataset root,
        the content-keyed filenames would still keep entries apart.
        """
        if not self.cache_reconstruction:
            return None
        return os.path.join(self.paths["data_path"], subdir)

    def _load_with_map_anything(self) -> None:
        """Load all frames using map-anything for depth + camera poses.

        Phase 1: Load images and masks for all frames.
        Phase 2: Run map-anything on all images jointly.
        Phase 3: Populate FrameData with depth, intrinsics, c2w, pointmap.
        """
        from .map_anything_depth import run_map_anything_cached

        # Phase 1: Load images and masks
        images = []
        masks_by_idx: Dict[int, MaskDict] = {}
        orig_h = orig_w = None
        for idx in self._ma_frame_indices:
            image, masks = self._load_image_and_masks(idx)
            images.append(image)
            masks_by_idx[idx] = masks
            if orig_h is None:
                orig_h, orig_w = image.shape[:2]
        # Original (pre-crop) size for the 2D-track un-crop provenance.
        self.orig_hw = (int(orig_h), int(orig_w))

        # Optionally feed GT depth to map-anything as a conditioning input
        # (final depth stays MA's prediction; depth_source='gt'+map_anything is
        # rejected in Phase A). Existence was checked in Phase B.
        dataset_depths, dataset_intrinsics, dataset_camera_poses = self._conditioning_tables(
            orig_h, orig_w)

        # Phase 2: Run map-anything on all frames.
        #
        # Scoped so map-anything's matmul precision cannot escape into the pipeline:
        # ``mapanything/models/mapanything/model.py`` sets
        # ``torch.backends.cuda.matmul.allow_tf32 = True`` at IMPORT time, which is
        # process-wide and permanent.  Unrestored, every matmul AFTER preprocessing
        # (pose decode, refinement, FINETUNE) runs at TF32's ~1e-3 relative on a
        # map_anything run while a ground_truth run stays at fp32's ~1e-7 -- so datasets
        # would be computed at different precisions depending on the backend.  Wrapped
        # here rather than at the import because the model load inside
        # ``run_map_anything`` pulls ``mapanything.models`` by a second path.
        # map-anything's own forward is unaffected: it runs under bf16 autocast.
        import torch

        _tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            results = run_map_anything_cached(
                images,
                self._ma_frame_indices,
                dataset_depths=dataset_depths,
                dataset_intrinsics=dataset_intrinsics,
                dataset_camera_poses=dataset_camera_poses,
                cache_dir=self._recon_cache_dir("ma_cache"),
                resolution_set=self.recon_resolution_set,
                mask_edges=self.recon_mask_edges,
                confidence_min=self.recon_confidence_min,
            )
        finally:
            torch.backends.cuda.matmul.allow_tf32 = _tf32

        # Phase 3: Build FrameData with depth, intrinsics, c2w, pointmap.
        # idx = asset index into paths arrays; fk = corresponding FrameKey
        # for self._frames dict assignment.
        for i, (idx, fk) in enumerate(zip(self._ma_frame_indices, self._ma_frame_keys)):
            image = images[i]
            masks = masks_by_idx[idx]
            ma_result = results[idx]

            depth_map_z = ma_result.depth_map_z   # read below for the model resolution
            K_matrix = ma_result.K_matrix         # compared against GT K below

            # When map-anything predicted depth, use the cropped/resized
            # image so that RGB and depth are pixel-aligned.
            if ma_result.cropped_image is not None:
                image = ma_result.cropped_image
                # Crop masks to match (same spatial transform)
                model_h, model_w = depth_map_z.shape
                masks = _crop_masks_to_match(masks, images[i].shape[:2], (model_h, model_w))

            # When MA's output c2w is overridden with GT c2w (camera_poses_source
            # == "gt") and we passed GT intrinsics as conditioning input, also
            # override K with the GT K so the (c2w, K) pair stays internally
            # consistent. Without this, MA's predicted K -- even with
            # conditioning -- can drift a few percent, leaving a half-MA/half-GT
            # camera spec.
            if (self.camera_poses_source == "gt"
                    and dataset_intrinsics is not None
                    and idx in dataset_intrinsics):
                model_h, model_w = depth_map_z.shape
                # MA scales uniformly then centre-crops; a per-axis rescale skews fx
                # whenever the input AR differs from the snapped table entry.
                K_gt = crop_resize_intrinsics(
                    dataset_intrinsics[idx], (orig_h, orig_w), (model_h, model_w)
                ).astype(K_matrix.dtype)
                # Sanity check: with K conditioning + override, MA's K should
                # track the conditioning K closely. Loud failure here means
                # something silently broke in the conditioning pipeline.
                fx_ratio = K_matrix[0, 0] / K_gt[0, 0]
                fy_ratio = K_matrix[1, 1] / K_gt[1, 1]
                assert 0.9 < fx_ratio < 1.1 and 0.9 < fy_ratio < 1.1, (
                    f"frame {idx}: MA-predicted K ({K_matrix[0,0]:.1f}, "
                    f"{K_matrix[1,1]:.1f}) differs from GT K "
                    f"({K_gt[0,0]:.1f}, {K_gt[1,1]:.1f}) by more than 10% -- "
                    f"GT-K conditioning may have failed"
                )
                ma_result = dataclasses.replace(ma_result, K_matrix=K_gt)

            self._finalize_recon_frame(fk, idx, image, masks, ma_result)

    def _finalize_recon_frame(
        self,
        fk: FrameKey,
        idx: int,
        image: np.ndarray,
        masks: MaskDict,
        res: "ReconResult",
    ) -> None:
        """Turn one reconstruction backend's per-frame output into a :class:`FrameData`.

        Backend-agnostic: the NaN fill, the pointmap, downscaling and the
        predicted-camera record.

        Takes the whole ``ReconResult`` rather than its four arrays unpacked: they are
        all same-shaped ndarrays, which is exactly the argument list that transposes
        silently at a future call site.
        """
        K_matrix, depth_map_z = res.K_matrix, res.depth_map_z
        valid_mask, c2w = res.valid_mask, res.c2w
        # Mark invalid depth pixels as NaN in depth map. The COPY matters: the raw
        # depth stays in FrameData.depth_map_z, which the depth losses read.
        depth_for_pointmap = depth_map_z.copy()
        if valid_mask is not None:
            depth_for_pointmap[~valid_mask] = np.nan

        # Generate pointmap from depth + K (camera-space R3 convention).
        # Kept in camera space for SAM3D conditioning.  Background
        # Gaussians are transformed to world space at creation time
        # using c2w (see create_background_gaussians).
        pointmap = depth_to_pointmap(depth_for_pointmap, K_matrix)
        if valid_mask is not None:
            pointmap[~valid_mask] = np.nan

        # Apply downscaling
        if self.downscale_factor > 1:
            image, masks, K_matrix, depth_map_z, pointmap, valid_mask, _ = (
                self._apply_downscaling(
                    image, masks, K_matrix, depth_map_z, pointmap, valid_mask
                )
            )

        # camera_poses_source='pred' uses the model's predicted c2w; 'gt' starts
        # from identity here and is overwritten by _apply_dataset_camera_poses()
        # after the caller's loop.
        frame_c2w = c2w if self.camera_poses_source == "pred" else np.eye(4, dtype=np.float32)

        # Persist the predicted (c2w, K) for every asset the model ran on. K
        # here is at render resolution (post-downscale; the K-override
        # branch in the MA loop only fires under camera_poses_source='gt', so under
        # 'pred' this stays the prediction).
        self._ma_predicted_cameras[idx] = {
            "c2w": c2w.copy(),
            "K": K_matrix.copy(),
        }

        self._frames[fk] = FrameData(
            image=image,
            masks=masks,
            K_matrix=K_matrix,
            pointmap=pointmap,
            depth_map_z=depth_map_z,
            valid_mask=valid_mask,
            normals_map=None,  # map-anything outputs no normals
            c2w=frame_c2w,
        )

    # ------------------------------------------------------------------
    # MoGe / GT depth path (per-frame)
    # ------------------------------------------------------------------

    def _load_frame(
        self,
        frame_idx: int,
        use_dataset_depth: bool,
    ) -> FrameData:
        """Load all data for a single frame from disk."""
        image, masks = self._load_image_and_masks(frame_idx)
        if self.reconstruction_model == "moge" and not use_dataset_depth:
            # Same input rule as map-anything, so a MoGe run differs from a
            # map-anything one by the MODEL and not by the resolution it happened to
            # see. MoGe's normalized intrinsics denormalize by whatever (W, H) it is
            # handed, so K stays correct for the crop. Gated on predicting: dataset
            # depth is stored at the native size, and cropping the RGB alone would
            # misalign it.
            #
            # The target is decided ONCE, by the first frame, not per frame. `_frames`
            # is filled lazily so there is no batch to average over here -- but
            # `self.H/W` is taken from the first frame and treated as the sequence's
            # size, so per-frame snapping would silently mislabel a sequence whose
            # frames differ in aspect ratio.
            if getattr(self, "_moge_target_hw", None) is None:
                # Pre-crop size, for the 2D-track un-crop. Without this it would
                # default to the POST-crop size and the tracks would never un-crop.
                self.orig_hw = (int(image.shape[0]), int(image.shape[1]))
                self._moge_target_hw = self._model_input_size(
                    image.shape[1] / image.shape[0])
            image, masks = self._crop_resize_to_model_input(
                image, masks, self._moge_target_hw)
        H, W = image.shape[:2]

        depth_names_for_frame = []
        depth_frames_path = self.paths["frames_path"]
        if self.dataset_type in ("gso", "oursactionbench") and self.paths["depth_names"]:
            depth_names_for_frame = [self.paths["depth_names"][frame_idx]]

        pointmap, K_matrix, valid_mask, depth_map_z, normals_map = load_and_process_depth(
            depth_frames_path,
            depth_names_for_frame,
            W,
            H,
            use_dataset_depth=use_dataset_depth,
            image=image,
            dataset_type=self.dataset_type,
        )

        # Downscale all spatial data by integer stride (nearest-neighbor).
        # Applied after depth processing so both MoGe and GT depth paths
        # are handled uniformly.
        if self.downscale_factor > 1:
            image, masks, K_matrix, depth_map_z, pointmap, valid_mask, normals_map = (
                self._apply_downscaling(
                    image, masks, K_matrix, depth_map_z, pointmap, valid_mask, normals_map
                )
            )

        return FrameData(
            image=image,
            masks=masks,
            K_matrix=K_matrix,
            pointmap=pointmap,
            depth_map_z=depth_map_z,
            valid_mask=valid_mask,
            normals_map=normals_map,
        )

    # ------------------------------------------------------------------
    # Downscaling
    # ------------------------------------------------------------------

    def _apply_downscaling(
        self,
        image: np.ndarray,
        masks: MaskDict,
        K_matrix: np.ndarray,
        depth_map_z: np.ndarray,
        pointmap: np.ndarray,
        valid_mask: Optional[np.ndarray],
        normals_map: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, MaskDict, np.ndarray, np.ndarray, np.ndarray,
               Optional[np.ndarray], Optional[np.ndarray]]:
        """Downscale all spatial data by ``self.downscale_factor``.

        Always returns 7 values: (image, masks, K_matrix, depth_map_z,
        pointmap, valid_mask, normals_map).  ``normals_map`` may be None.
        """
        d = self.downscale_factor
        image = image[::d, ::d]
        depth_map_z = depth_map_z[::d, ::d]
        pointmap = pointmap[::d, ::d]
        if valid_mask is not None:
            valid_mask = valid_mask[::d, ::d]
        if normals_map is not None:
            normals_map = normals_map[::d, ::d]
        masks = MaskDict(
            {k: v[::d, ::d] for k, v in masks.items()},
            shape=(image.shape[0], image.shape[1]),
        )
        K_matrix = K_matrix.copy()
        K_matrix[0, :] /= d  # fx, skew, cx
        K_matrix[1, :] /= d  # fy, cy

        return image, masks, K_matrix, depth_map_z, pointmap, valid_mask, normals_map

    # ------------------------------------------------------------------
    # Median intrinsics (MoGe only)
    # ------------------------------------------------------------------

    def _apply_median_intrinsics(self) -> None:
        """Replace per-frame intrinsics with element-wise median and recompute pointmaps.

        MoGe predicts per-frame intrinsics that vary slightly between frames.
        For fixed-camera datasets, the true intrinsics are constant.  This method
        computes the element-wise median K across all loaded frames, then
        recomputes each frame's pointmap using the shared K matrix.
        """
        if len(self._frames) <= 1:
            return

        K_stack = np.stack([self._frames[fk].K_matrix for fk in self._frame_keys])
        median_K = np.median(K_stack, axis=0)

        print(
            f"  Median intrinsics: fx={median_K[0, 0]:.2f}, fy={median_K[1, 1]:.2f}, "
            f"cx={median_K[0, 2]:.2f}, cy={median_K[1, 2]:.2f}"
        )

        for fk in self._frame_keys:
            frame = self._frames[fk]
            frame.pointmap = depth_to_pointmap(
                frame.depth_map_z, median_K, valid_mask=frame.valid_mask
            )
            frame.K_matrix = median_K.copy()

    # ------------------------------------------------------------------
    # Dataset camera poses
    # ------------------------------------------------------------------

    def _load_dataset_camera_poses(self) -> Optional[Dict[int, np.ndarray]]:
        """Load dataset camera poses as c2w matrices in OpenCV convention.

        Returns ``{frame_idx: c2w_4x4}`` or ``None`` if the dataset has
        no camera poses.  Used both for map-anything conditioning and for
        overriding ``FrameData.c2w`` with GT poses.

        Supports:
        - **GSO**: per-view ``.npy`` w2c matrices (Blender convention,
          converted to OpenCV via ``diag(1,-1,-1)`` rotation flip).
        - **CO3D LaRa**: per-view ``{NNN}.npy`` c2w.
        - **DynCustom**: optional ``{scene}/poses.json``.
        - **OursActionBench**: identity per frame.  The dataset's per-scene
          per-frame fitted cameras (``data/oursactionbench/{scene}/camera.json``,
          loaded by :func:`gt_data.load_actionbench_camera_fit`) are
          baked into the per-frame object poses by ``GT_SHAPES_INVERSION``;
          the dataset's own camera path stays identity, and we expose that
          here so ``camera_poses_source='gt'`` (or
          ``condition_recon_model_on_gt_poses``) can pin MA's predicted c2w to
          identity rather than letting it drift.
        """
        # --- OursActionBench: identity per frame
        # (camera baked into object poses by GT_SHAPES_INVERSION) ---
        if self.dataset_type == "oursactionbench":
            ident = np.eye(4, dtype=np.float32)
            return {idx: ident.copy() for idx in self._ma_frame_indices}

        # --- DynCustom: OPTIONAL per-frame cameras from {scene}/poses.json ---
        # The only dataset whose poses are optional: an uploaded clip usually has
        # none, but one staged from a tracked capture can ship them.  Absent ->
        # None, and MA predicts the cameras.
        if self.dataset_type == "dyncustom":
            return self._load_poses_json(
                os.path.join(self.paths["data_path"], "poses.json")
            )

        # --- CO3D LaRa: per-view c2w (4x4 float32) stored as ``{NNN}.npy``
        # in the flat scene dir. The camera
        # axes are already R3/OpenCV (+X right, Y-down, +Z into the scene); only
        # the world differs (+Y up vs R3's -Y up), so load_co3d_lara_c2w left-
        # multiplies the diag(1,-1,-1,1) world rotation (see CO3D_LARA_TO_R3).
        if self.dataset_type == "co3d":
            from .eval_assets_export import load_co3d_lara_c2w
            frames_path = self.paths["frames_path"]
            camera_pose_names = self.paths["camera_pose_names"]
            return {
                idx: load_co3d_lara_c2w(
                    os.path.join(frames_path, camera_pose_names[idx])
                )
                for idx in self._ma_frame_indices
            }

        # --- GSO: per-view .npy files (3x4 w2c, Blender convention) ---
        # Blender is Y-up in camera frame + Z-up in world frame (EscherNet's
        # render script imports OBJ with axis_up='Z', axis_forward='Y').
        # Pipeline code assumes R3 everywhere: Y-down in camera frame, -Y-up
        # in world frame.  Two rotations convert:
        #   M_cam = diag(1, -1, -1)   flips camera local Y and Z
        #   R_W                       maps Blender +Z-up -> R3 -Y-up
        # R_W is the single source of truth -- imported from eval_assets_export
        # so the same rotation applies at load time, for test-view cameras,
        # and for the exported mesh un-rotation.
        camera_pose_names = self.paths.get("camera_pose_names")
        if camera_pose_names:
            from .eval_assets_export import BLENDER_TO_R3_WORLD as R_W
            M_cam = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
            render_dir = self.paths["frames_path"]
            result = {}
            for idx in self._ma_frame_indices:
                if idx >= len(camera_pose_names):
                    continue
                npy_path = os.path.join(render_dir, camera_pose_names[idx])
                if not os.path.isfile(npy_path):
                    continue
                w2c_3x4 = np.load(npy_path).astype(np.float32)
                R_blender = w2c_3x4[:3, :3]
                t = w2c_3x4[:3, 3]
                c2w = np.eye(4, dtype=np.float32)
                c2w[:3, :3] = R_W @ R_blender.T @ M_cam
                c2w[:3, 3] = R_W @ (-R_blender.T @ t)
                result[idx] = c2w
            return result if result else None

        return None

    def _conditioning_camera_poses(self) -> Optional[Dict[int, np.ndarray]]:
        """Camera poses to hand map-anything as a conditioning input, or ``None``
        to let it predict them from the images alone.

        Independent of ``camera_poses_source``, which decides what the pipeline
        CONSUMES; this only decides what the model is told.

        ``condition_recon_model_on_gt_poses`` is always-strict -- true on a
        dataset with no poses raises.  That is the right contract for a benchmark
        (a run must not quietly lose conditioning it was configured for), but it
        cannot express ``dyncustom``, whose cameras are genuinely optional: an
        uploaded clip has none, while one staged from a tracked capture ships a
        ``poses.json``.  So that dataset opts in by the file's presence instead.
        """
        if self.condition_recon_model_on_gt_poses:
            return self._load_gt_camera_poses_or_raise()
        if self.dataset_type == "dyncustom":
            poses = self._load_dataset_camera_poses()
            if poses:
                print("  Conditioning map-anything on the scene's poses.json cameras")
            return poses
        return None

    def _load_poses_json(self, path: str) -> Optional[Dict[int, np.ndarray]]:
        """Per-frame cameras from a ``poses.json``, or ``None`` if there is none.

        The schema is the ``cameras`` block this pipeline's own FINAL block
        writes (:func:`io_utils.save_perframe_poses_json`), so a previous run's
        cameras -- or any tool that emits that shape -- feed straight back in::

            {"cameras": [{"frame": 0, "view": 0, "c2w": [[..4x4..]]}, ...]}

        ``c2w`` is **camera-to-world** in R3/OpenCV, exactly like
        ``FrameData.c2w``; nothing is inverted on load.  ``frame`` indexes the
        scene's sorted frame listing.  Other keys (``K``, ``frame_idx``, the
        ``objects`` block) are ignored -- only the cameras are read.

        Absent is fine; PRESENT-BUT-BROKEN is not.  A file that exists is a
        deliberate act, so a malformed one raises rather than silently falling
        back to predicted cameras -- the failure would otherwise be a quietly
        worse reconstruction rather than an error.
        """
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            return None

        import json

        with open(path) as f:
            data = json.load(f)
        cameras = data.get("cameras") if isinstance(data, dict) else None
        if not cameras:
            raise ValueError(
                f"{path}: no 'cameras' entries. Expected the schema FINAL writes: "
                f'{{"cameras": [{{"frame": 0, "view": 0, "c2w": [[..4x4..]]}}, ...]}}'
            )

        poses: Dict[int, np.ndarray] = {}
        for entry in cameras:
            frame = int(entry["frame"])
            c2w = np.asarray(entry["c2w"], dtype=np.float32)
            if c2w.shape != (4, 4):
                raise ValueError(
                    f"{path}: frame {frame} c2w has shape {c2w.shape}, expected (4, 4)."
                )
            if frame in poses:
                # Mono-dynamic: one camera per timestamp. A repeat means a
                # multi-view file, whose extra views would be silently dropped.
                raise ValueError(
                    f"{path}: frame {frame} appears more than once — this dataset "
                    f"is mono (one camera per timestamp)."
                )
            poses[frame] = c2w

        missing = [i for i in self._ma_frame_indices if i not in poses]
        if missing:
            raise ValueError(
                f"{path}: no camera for frame(s) {missing}. A partial file would "
                f"condition only some frames and leave the rest to drift; give "
                f"every frame a camera or remove the file."
            )
        print(f"  Loaded {len(poses)} camera poses from {path}")
        return poses

    def _load_gt_camera_poses_or_raise(self) -> Dict[int, np.ndarray]:
        """Load dataset camera poses, always raising if absent.

        Called only when GT poses are requested (``camera_poses_source='gt'``
        or ``condition_recon_model_on_gt_poses``), so a missing dataset is a
        misconfiguration -- always-strict, no fallback.
        """
        poses = self._load_dataset_camera_poses()
        if poses is None:
            raise FileNotFoundError(
                f"GT camera poses requested (camera_poses_source='gt' or "
                f"condition_recon_model_on_gt_poses=True) but dataset "
                f"{self.dataset_type!r} scene {self.scene_name!r} has no camera "
                f"poses to load."
            )
        return poses

    def _apply_dataset_camera_poses(self) -> None:
        """Load dataset camera poses and write them to FrameData.c2w.

        Raises if the dataset has no camera poses (always-strict -- see
        ``_load_gt_camera_poses_or_raise``).
        """
        poses = self._load_gt_camera_poses_or_raise()
        # poses is asset-indexed (Dict[int, c2w]); map each asset_idx to its
        # FrameKey via _asset_to_fk (covers user-visible frames). For asset
        # indices not in _asset_to_fk (MA-extras), use the synthesized
        # FrameKey from _ma_frame_keys.
        asset_to_any_fk = dict(self._asset_to_fk)
        for asset_idx, fk in zip(self._ma_frame_indices, self._ma_frame_keys):
            asset_to_any_fk.setdefault(asset_idx, fk)
        for asset_idx, c2w in poses.items():
            fk = asset_to_any_fk.get(asset_idx)
            if fk is not None and fk in self._frames:
                self._frames[fk].c2w = c2w
        print(f"  Dataset camera poses applied to {len(poses)} frames")

    def _align_depth_scale_to_gt_poses(
        self, ma_poses: Dict[int, np.ndarray]
    ) -> None:
        """Scale depth maps so their metric scale matches GT camera distances.

        Why the ratio works: map-anything outputs ``cam_trans`` and depth
        multiplied by a shared learned ``scale_final_output`` (see
        ``mapanything/model.py:1789-1790``), so translations and depth are
        always in the same metric.  If GT-metric inter-camera distances are
        ``D_gt`` and map-anything's are ``D_ma``, then multiplying depth by
        ``D_gt / D_ma`` restores the GT metric.

        Multi-view (``N >= 2``): use pairwise distances -- translation and
        rotation invariant, so map-anything's arbitrary world frame doesn't
        matter.  Median ratio is robust to per-view noise.

        Single-view (``N = 1``) fallback: no pairs exist.  When the dataset
        declares ``world_origin_is_object`` (GSO: object at world origin by
        EscherNet convention), ``||t_gt||`` and ``||t_ma||`` both measure
        camera-to-object distance, so the same ratio works.  Asserts loud
        on ``||t_ma|| ~= 0`` -- that means map-anything put its only camera
        at identity (typical without pose conditioning), which would make
        the ratio undefined; fixing that is upstream (enable
        ``condition_recon_model_on_gt_poses`` so MA receives a non-identity pose
        anchor).  Datasets without the flag skip silently: there's no valid
        anchor for single-view.
        """
        # ma_poses is asset-indexed; map asset → FrameKey for self._frames.
        asset_to_any_fk = dict(self._asset_to_fk)
        for asset_idx, fk in zip(self._ma_frame_indices, self._ma_frame_keys):
            asset_to_any_fk.setdefault(asset_idx, fk)

        indices = [idx for idx in self._ma_frame_indices if idx in ma_poses]
        if not indices:
            return

        if len(indices) < 2:
            if not self.world_origin_is_object:
                return  # no anchor: silent skip matches multi-view < 2 behavior
            idx = indices[0]
            gt_t = float(np.linalg.norm(self._frames[asset_to_any_fk[idx]].c2w[:3, 3]))
            ma_t = float(np.linalg.norm(ma_poses[idx][:3, 3]))
            # gt_t ~= 0 means _apply_dataset_camera_poses was a no-op (dataset
            # has no GT poses) despite world_origin_is_object=true -- a config
            # error.  Skip silently instead of dividing by ~zero and zeroing
            # all depth, like the multi-view zero-baseline guard below.
            if gt_t < 1e-6:
                return
            # Fail loud if map-anything placed its only camera at the world
            # origin (identity c2w).  Ratio scale=gt_t/ma_t is undefined,
            # and silently no-op'ing would mask whatever's upstream (pose
            # conditioning not being passed, model ignoring it, etc.).
            assert ma_t >= 1e-6, (
                f"map-anything predicted camera 0 at identity (||t_ma||={ma_t:.2e}); "
                f"single-view scale anchor undefined.  Check that "
                f"condition_recon_model_on_gt_poses=True and the dataset provides "
                f"a non-identity GT pose for frame 0."
            )
            scale = gt_t / ma_t
            if abs(scale - 1.0) < 1e-4:
                print(f"  Depth scale alignment (N=1): scale={scale:.4f} (no correction needed)")
                return
            for asset_i in self._ma_frame_indices:
                frame = self._frames[asset_to_any_fk[asset_i]]
                frame.depth_map_z = frame.depth_map_z * scale
                frame.pointmap = frame.pointmap * scale
            print(f"  Depth scale aligned to GT (N=1, camera-to-origin): scale={scale:.4f}")
            return

        # Collect camera positions
        gt_positions = np.array([self._frames[asset_to_any_fk[i]].c2w[:3, 3] for i in indices])
        ma_positions = np.array([ma_poses[i][:3, 3] for i in indices])

        # Pairwise distances (upper triangle)
        from itertools import combinations
        gt_dists = []
        ma_dists = []
        for i, j in combinations(range(len(indices)), 2):
            gt_dists.append(np.linalg.norm(gt_positions[i] - gt_positions[j]))
            ma_dists.append(np.linalg.norm(ma_positions[i] - ma_positions[j]))

        gt_dists = np.array(gt_dists)
        ma_dists = np.array(ma_dists)

        # No GT camera baseline: every GT camera is at the same position
        # (e.g. OursActionBench / any static-camera dataset under
        # camera_poses_source='gt' -- dataset cameras are identity, the real
        # camera is baked into the per-frame object poses).  gt_dists are
        # then all ~0, so scale=median(0/ma_dists)=0 would multiply ALL
        # depth and pointmaps by zero.  No camera motion means no metric
        # anchor exists here, so no-op instead of zeroing depth -- mirrors
        # the single-view ``gt_t < 1e-6`` guard above.
        if not (gt_dists > 1e-6).any():
            print("  Depth scale alignment: GT camera baselines ~0 "
                  "(static-camera dataset); no metric anchor, skipping.")
            return

        # Robust scale: median ratio (skip near-zero baselines)
        valid = ma_dists > 1e-6
        if not valid.any():
            return
        ratios = gt_dists[valid] / ma_dists[valid]
        scale = float(np.median(ratios))
        iqr = float(np.percentile(ratios, 75) - np.percentile(ratios, 25))
        n_pairs = int(valid.sum())
        if abs(scale - 1.0) < 1e-4:
            print(f"  Depth scale alignment: scale={scale:.4f} "
                  f"(IQR={iqr:.4f}, n_pairs={n_pairs}) (no correction needed)")
            return

        # Apply scale to depth and pointmap for every frame
        for asset_idx in self._ma_frame_indices:
            frame = self._frames[asset_to_any_fk[asset_idx]]
            frame.depth_map_z = frame.depth_map_z * scale
            frame.pointmap = frame.pointmap * scale

        print(f"  Depth scale aligned to GT poses: scale={scale:.4f} "
              f"(IQR={iqr:.4f}, n_pairs={n_pairs})")

    def __getitem__(self, key) -> FrameData:
        """Get cached frame data by FrameKey or int (coerced to FrameKey(int, 0))."""
        return self._frames[as_frame_key(key)]

    def __contains__(self, key) -> bool:
        """Check if a FrameKey (or coerced int) is loaded."""
        return as_frame_key(key) in self._frames

    def __len__(self) -> int:
        """Number of loaded frames."""
        return len(self._frames)

    def __iter__(self):
        """Iterate over FrameKeys in sorted (view, frame) order."""
        yield from self._frame_keys

    @property
    def frame_keys(self) -> List[FrameKey]:
        """List of FrameKeys in sorted (view, frame) order."""
        return list(self._frame_keys)

    @property
    def is_dynamic(self) -> bool:
        """True if more than one distinct frame value is present (time axis populated)."""
        return len({fk.frame for fk in self._frame_keys}) > 1

    @property
    def is_mv(self) -> bool:
        """True if more than one distinct view value is present (view axis populated)."""
        return len({fk.view for fk in self._frame_keys}) > 1

    def predicted_camera(
        self, asset_idx: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return MA-predicted ``(c2w, K)`` for ``asset_idx``, or ``None``
        if MA was not run on this asset.

        Covers held-out views that MA processed for joint pose inference but
        that were pruned from :attr:`_frames` post-MA (CO3D NVS targets).
        K is at render resolution (post-downscale).
        """
        cam = self._ma_predicted_cameras.get(asset_idx)
        if cam is None:
            return None
        return cam["c2w"], cam["K"]

    def __repr__(self) -> str:
        return (
            f"Sequence(scene={self.scene_name!r}, type={self.dataset_type!r}, "
            f"frames={len(self)}, size={self.H}x{self.W})"
        )


__all__ = [
    "FrameData",
    "Sequence",
]
