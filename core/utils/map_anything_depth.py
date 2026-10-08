"""
Map-Anything multi-view depth and camera pose estimation.

Wraps the map-anything model to provide depth maps, intrinsics, validity
masks, and camera-to-world (c2w) transforms for all keyframes jointly.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .recon_cache import ReconResult, masking_key_suffix, run_cached


#: The per-frame result type lives in `recon_cache`, beside the cache that stores it;
#: aliased here under the map-anything name.
MapAnythingResult = ReconResult


def run_map_anything(
    images: List[np.ndarray],
    frame_indices: List[int],
    dataset_depths: Optional[Dict[int, np.ndarray]] = None,
    dataset_intrinsics: Optional[Dict[int, np.ndarray]] = None,
    dataset_camera_poses: Optional[Dict[int, np.ndarray]] = None,
    resolution_set: int = 518,
    mask_edges: bool = True,
    confidence_min: Optional[float] = None,
) -> Dict[int, MapAnythingResult]:
    """Run map-anything multi-view inference on all frames.

    Parameters
    ----------
    images : list of np.ndarray
        Per-frame RGB images, each (H, W, 3) uint8.
    frame_indices : list of int
        Corresponding frame indices (same length as *images*).
    dataset_depths : dict, optional
        ``{frame_idx: depth_map_z}`` where depth_map_z is (H, W) float32.
        When provided, map-anything is conditioned on this depth to improve
        camera pose predictions.  Requires *dataset_intrinsics* for the same
        frames.
    dataset_intrinsics : dict, optional
        ``{frame_idx: K_matrix}`` where K_matrix is (3, 3) float32.
        Required when *dataset_depths* is provided.
    dataset_camera_poses : dict, optional
        ``{frame_idx: c2w}`` where c2w is (4, 4) float32 camera-to-world.
        When provided, map-anything is conditioned on these poses to improve
        depth predictions.
    mask_edges : bool, optional
        Apply map-anything's normals+depth edge mask.  Default True (its own
        default).  See ``ProcessingConfig.recon_mask_edges``.
    confidence_min : float, optional
        Reject pixels with ``conf <= confidence_min``.  ``None`` (default) leaves
        the confidence unused.  See ``ProcessingConfig.recon_confidence_min``.

    Returns
    -------
    dict
        ``{frame_idx: MapAnythingResult}`` with depth, intrinsics, mask, c2w
        for every frame.  When *dataset_depths* was provided, the returned
        depth and intrinsics are the original dataset values (map-anything
        only contributes c2w).
    """
    from mapanything.utils.image import preprocess_inputs
    from mapanything.utils.cropping import crop_resize_if_necessary

    from .model_cache import ModelCache
    # Our vendored copy of map-anything's own table, not theirs: every backend sizes
    # its input through one function, so map-anything's replicated crop here and the
    # crop `Sequence` applies for MoGe cannot drift onto different targets.
    from .sequence import fixed_mapping_size

    assert len(images) == len(frame_indices)
    orig_h, orig_w = images[0].shape[:2]

    # ------------------------------------------------------------------
    # 0. Compute the crop+resize that preprocess_inputs will apply
    # ------------------------------------------------------------------
    # Replicate target-size logic from preprocess_inputs (fixed_mapping mode) so we
    # can apply the same spatial transform to RGB and masks later. `resolution_set` is
    # passed to preprocess_inputs below as well -- the two MUST agree, or the RGB is
    # cropped to one size and the depth predicted at another.
    avg_ar = np.mean([img.shape[1] / img.shape[0] for img in images])
    target_w, target_h = fixed_mapping_size(avg_ar, resolution_set)
    target_size = (target_w, target_h)
    print(f"  Map-anything: center-crop+resize {orig_w}x{orig_h} -> "
          f"{target_w}x{target_h} (patch-14 aligned, model resolution)")

    # ------------------------------------------------------------------
    # 1. Build input views
    # ------------------------------------------------------------------
    has_dataset_depth = dataset_depths is not None and len(dataset_depths) > 0
    has_dataset_intrinsics = dataset_intrinsics is not None and len(dataset_intrinsics) > 0
    has_dataset_poses = dataset_camera_poses is not None and len(dataset_camera_poses) > 0
    input_views = []
    for img, fi in zip(images, frame_indices):
        view: dict = {"img": img}
        if has_dataset_depth and fi in dataset_depths:
            assert dataset_intrinsics is not None and fi in dataset_intrinsics, (
                "dataset_intrinsics required for every frame with dataset_depths"
            )
            view["depth_z"] = dataset_depths[fi]
        if has_dataset_intrinsics and fi in dataset_intrinsics:
            view["intrinsics"] = dataset_intrinsics[fi]
        if has_dataset_poses and fi in dataset_camera_poses:
            view["camera_poses"] = dataset_camera_poses[fi]
        input_views.append(view)

    # ------------------------------------------------------------------
    # 2. Preprocess (resize + normalize)
    # ------------------------------------------------------------------
    processed_views = preprocess_inputs(
        input_views, verbose=False, resolution_set=resolution_set)

    # ------------------------------------------------------------------
    # 3. Run inference
    # ------------------------------------------------------------------
    model = ModelCache.get().map_anything_model
    predictions = model.infer(
        processed_views,
        memory_efficient_inference=True,
        minibatch_size=None,
        use_amp=True,
        amp_dtype="bf16",
        apply_mask=True,
        mask_edges=mask_edges,
        # Left off even when `confidence_min` is set: MA's own confidence mask is a
        # QUANTILE (`conf > percentile(conf, 10)`), which tracks how much of this
        # particular image is unresolved rather than whether a given pixel is.
        apply_confidence_mask=False,
        use_multiview_confidence=False,
    )

    # ------------------------------------------------------------------
    # 4. Extract outputs (keep at model resolution, crop RGB to match)
    # ------------------------------------------------------------------
    results: Dict[int, MapAnythingResult] = {}
    for i, fi in enumerate(frame_indices):
        pred = predictions[i]

        # c2w — resolution-independent
        c2w = pred["camera_poses"][0].cpu().numpy().astype(np.float32)  # (4, 4)

        if has_dataset_depth and fi in dataset_depths:
            # Use original dataset depth + intrinsics; only take c2w
            depth_z = dataset_depths[fi].astype(np.float32)
            K = dataset_intrinsics[fi].astype(np.float32)
            # Build valid_mask from finite depth (dataset depth may use NaN
            # for invalid pixels or have all valid)
            valid_mask = np.isfinite(depth_z) & (depth_z > 0)
            cropped_image = None  # depth is at original resolution
        else:
            # Use map-anything predictions at model resolution.
            # Apply the same crop+resize to the original RGB so that
            # depth and image are pixel-aligned.
            depth_z = pred["depth_z"][0].squeeze(-1).cpu().numpy()  # (H_m, W_m)
            valid_mask = pred["mask"][0].squeeze(-1).cpu().numpy() > 0.5  # (H_m, W_m)
            K = pred["intrinsics"][0].cpu().numpy()  # (3, 3)

            # Reject the pixels MA itself rates as guesses -- see
            # ``ProcessingConfig.recon_confidence_min``. Narrowing `valid_mask` is the
            # whole implementation: the caller NaNs depth by it, and `apply_mask=True`
            # already zeroed depth outside MA's own mask, which this only shrinks.
            if confidence_min is not None:
                conf = pred["conf"][0].cpu().numpy()  # (H_m, W_m)
                valid_mask = valid_mask & (conf > confidence_min)

            # Crop+resize original image with the same transform
            import PIL.Image
            pil_img = PIL.Image.fromarray(images[i])
            result_tuple = crop_resize_if_necessary(
                image=pil_img, resolution=target_size,
            )
            cropped_pil = result_tuple[0]
            cropped_image = np.array(cropped_pil)  # (H_m, W_m, 3) uint8

        results[fi] = MapAnythingResult(
            depth_map_z=depth_z,
            K_matrix=K,
            valid_mask=valid_mask,
            c2w=c2w,
            cropped_image=cropped_image,
        )

    n_with_depth = sum(1 for fi in frame_indices if has_dataset_depth and fi in (dataset_depths or {}))
    n_with_poses = sum(1 for fi in frame_indices if has_dataset_poses and fi in (dataset_camera_poses or {}))
    cond_parts = []
    if n_with_depth:
        cond_parts.append(f"{n_with_depth} conditioned on dataset depth")
    if n_with_poses:
        cond_parts.append(f"{n_with_poses} conditioned on dataset camera poses")
    cond_str = f" ({', '.join(cond_parts)})" if cond_parts else ""
    print(f"  Map-anything: {len(results)} frames processed{cond_str}")

    return results


__all__ = [
    "MapAnythingResult",
    "run_map_anything",
]


def run_map_anything_cached(
    images: List[np.ndarray],
    frame_indices: List[int],
    dataset_depths: Optional[Dict[int, np.ndarray]] = None,
    dataset_intrinsics: Optional[Dict[int, np.ndarray]] = None,
    dataset_camera_poses: Optional[Dict[int, np.ndarray]] = None,
    cache_dir: Optional[str] = None,
    resolution_set: int = 518,
    mask_edges: bool = True,
    confidence_min: Optional[float] = None,
) -> Dict[int, MapAnythingResult]:
    """:func:`run_map_anything`, reusing a previous result for identical inputs.

    See :func:`recon_cache.run_cached` — keyed on the model id (and resolution and
    filtering options), so results from different settings can never be crossed.
    """
    import functools

    from .model_cache import ModelCache

    # Bound for the CALL, passed again below for the KEY -- two different jobs.
    runner = functools.partial(
        run_map_anything, dataset_depths=dataset_depths,
        dataset_intrinsics=dataset_intrinsics,
        dataset_camera_poses=dataset_camera_poses,
        resolution_set=resolution_set, mask_edges=mask_edges,
        confidence_min=confidence_min)
    opts = masking_key_suffix(mask_edges, confidence_min)
    return run_cached(
        runner, images, frame_indices, dataset_depths, dataset_intrinsics,
        dataset_camera_poses, cache_dir,
        model_id=f"{ModelCache.get().map_anything_model_id}@{resolution_set}{opts}",
        label="Map-anything", tag="ma",
    )
