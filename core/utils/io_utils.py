# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
File I/O utilities for the SAM3D-Objects pipeline.

This module provides functions for loading images, masks, and setting up
dataset paths, as well as saving various output formats.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    import torch

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


class MaskDict(dict):
    """
    A dict subclass that returns an all-zeros mask for missing object IDs.

    This ensures consistent object indexing across frames - if an object
    doesn't appear in a frame, accessing its mask returns zeros instead
    of raising KeyError.

    Attributes
    ----------
    shape : tuple
        The (H, W) shape of masks, used to create zero masks for missing keys.

    Examples
    --------
    >>> masks = MaskDict({1: mask1, 3: mask3}, shape=(480, 640))
    >>> masks[1].any()
    True  # Object 1 is present
    >>> masks[2].any()
    False  # Object 2 not present, returns zeros mask
    >>> masks.keys()
    dict_keys([1, 3])  # Only actually present objects
    """

    def __init__(self, data: Dict[int, np.ndarray], shape: Tuple[int, int]):
        super().__init__(data)
        self.shape = shape

    def __getitem__(self, key: int) -> np.ndarray:
        if key in self:
            return super().__getitem__(key)
        # Return all-zeros mask for missing object IDs
        return np.zeros(self.shape, dtype=bool)

    def get(self, key: int, default=None) -> np.ndarray:
        if key in self:
            return super().__getitem__(key)
        if default is not None:
            return default
        return np.zeros(self.shape, dtype=bool)


def load_image(path: str, to_uint8: bool = True, to_rgb: bool = False) -> np.ndarray:
    """
    Load an image from disk.

    Parameters
    ----------
    path : str
        Path to the image file. Supports PNG, JPG, TIFF, etc.
    to_uint8 : bool, optional
        Whether to convert the image to uint8 dtype. Default: True.
        Set to False for depth maps or floating-point images.
    to_rgb : bool, optional
        Expand palette ("P") / grayscale ("L") images to 3-channel RGB.
        Default: False returns the raw stored values, which :func:`load_masks`
        needs (a mask's palette indices ARE its object IDs). Set True for
        photos that merely happen to be palettized (the ``mvcustom``
        captures), which otherwise load as a 2D index map.

    Returns
    -------
    np.ndarray
        Loaded image as a NumPy array.

    Examples
    --------
    >>> img = load_image("image.png")
    >>> img.dtype
    dtype('uint8')
    >>> depth = load_image("depth.tiff", to_uint8=False)
    >>> depth.dtype
    dtype('float32')
    """
    image = Image.open(path)
    if to_rgb and image.mode in ("P", "L"):
        image = image.convert("RGB")
    image = np.array(image)
    if to_uint8:
        image = image.astype(np.uint8)
    return image


def load_masks(mask_path: str) -> MaskDict:
    """
    Load segmentation masks from a file.

    Parses a segmentation mask image where each unique pixel value
    represents a different object instance (0 = background).

    Parameters
    ----------
    mask_path : str
        Path to the segmentation mask image file.

    Returns
    -------
    MaskDict
        A dict mapping object IDs (pixel values) to boolean mask arrays.
        Accessing a missing object ID returns an all-zeros mask instead
        of raising KeyError, ensuring consistent behavior across frames.

    Notes
    -----
    - Pixel value 0 is always treated as background and skipped
    - The returned MaskDict automatically returns zeros for missing keys,
      so objects that don't appear in a frame still have valid (empty) masks

    Examples
    --------
    >>> masks = load_masks("segmentation.png")
    >>> masks.keys()
    dict_keys([1, 2, 3])  # Objects present in this frame
    >>> masks[1].shape
    (480, 640)
    >>> masks[1].any()
    True  # Object 1 is present

    >>> # Object 5 not in this frame - returns zeros mask automatically
    >>> masks[5].any()
    False
    """
    mask = load_image(mask_path)
    H, W = mask.shape[:2]

    # Handle RGBA masks: use alpha channel as binary foreground (object_id=1)
    if mask.ndim == 3 and mask.shape[2] == 4:
        alpha = mask[:, :, 3]
        masks_dict = {}
        if np.any(alpha > 0):
            masks_dict[1] = alpha > 0
        return MaskDict(masks_dict, shape=(H, W))

    # Build dict of object_id -> mask
    masks_dict = {}
    for object_id in np.unique(mask):
        if object_id == 0:
            continue  # skip background
        masks_dict[int(object_id)] = (mask == object_id)

    return MaskDict(masks_dict, shape=(H, W))


def group_palette_masks(
    masks: "MaskDict", palette_groups: Dict[int, List[int]]
) -> "MaskDict":
    """Collapse raw palette-id masks into per-object union masks.

    ``palette_groups`` maps an output object ID to the segmentation palette IDs
    that compose it (ActionMesh metadata's ``objects`` joining, e.g.
    ``{1: [1, 2]}`` merges DAVIS labels 1 and 2 into object 1). Palette IDs
    absent from the frame contribute nothing (``MaskDict`` yields zeros for
    missing keys); an object with no pixels in the frame is dropped.
    """
    grouped: Dict[int, np.ndarray] = {}
    for obj_id, palette_ids in palette_groups.items():
        union = np.zeros(masks.shape, dtype=bool)
        for pid in palette_ids:
            union |= masks[pid]
        if union.any():
            grouped[obj_id] = union
    return MaskDict(grouped, shape=masks.shape)


MV_STATIC_DATASETS = ("mvcustom", "gso", "co3d")
"""Datasets whose asset position is a VIEW at one timestamp, not a timestep.

Two things follow from that one fact, which is why the set is named rather than
restated: :func:`axis_lift_indices` maps the asset to ``FrameKey(0, asset)``, and
every consumer must subset to ``dataset.num_input_views`` — this loader
deliberately exposes MORE assets than the pipeline should consume (GSO's 10 train
views, every mvcustom image, CO3D's 8 LaRa views) so map-anything
keeps a robust pairwise scale anchor.  For CO3D that subsetting is load-bearing:
its assets are {000..003} input + {004..007} HELD-OUT NVS targets, so skipping it
reconstructs from the very views the run is then scored on.

Read by ``preprocessing.resolve_run_assets``.
"""


def _natural_key(s: str):
    """Sort key for natural ordering of strings with embedded numbers."""
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def axis_lift_indices(
    dataset_type: str,
    asset_indices: List[int],
) -> Tuple[List[int], List[int]]:
    """Map asset positions (file-index) to (frame_indices, view_indices).

    Each dataset's true axis structure decides where the asset position lands:

    - ``mvcustom``, ``gso``, ``co3d``: MV-static. asset position == view label,
      frame=0. A single selected asset → ``FrameKey(0, asset_idx)``
      (preserves the view label even when only one view is selected).
    - everything else (``image``, ``davis_actionmesh``, ``dyncustom``,
      ``oursactionbench``): mono. asset position == time index, view=0.

    Returns
    -------
    frame_indices : List[int]
        Time axis labels per asset position.
    view_indices : List[int]
        View axis labels per asset position.
    """
    n = len(asset_indices)
    if dataset_type in MV_STATIC_DATASETS:
        # MV-static: asset position is the view label. frame is constant=0.
        return [0] * n, list(asset_indices)
    # mono-dynamic and mono-static datasets: time axis, view=0
    return list(asset_indices), [0] * n


def setup_paths(
    dataset_path: str,
    scene_name: str,
    dataset_type: str,
    **kwargs,
) -> dict:
    """
    Setup and validate all necessary paths for a dataset.

    Parameters
    ----------
    dataset_path : str
        Root path to the dataset.
    scene_name : str
        Name of the scene to process.
    dataset_type : str
        Dataset name (``dataset.name``).

    Returns
    -------
    dict
        Dictionary containing all paths and file lists:
        - data_path: Root path to scene data
        - frames_path: Path to frame images
        - masks_path: Path to segmentation masks
        - image_names: Sorted list of image filenames
        - mask_names: Sorted list of mask filenames
        - depth_names: Sorted list of depth filenames (empty when the dataset has none)
        - dataset_type: The dataset type string

    Raises
    ------
    ValueError
        If dataset_type is not a supported dataset.
    """
    if dataset_type == "davis_actionmesh":
        # Metadata-driven layout: data/davis_actionmesh/{scene}/metadata.json points
        # back to the original RGB + segmentation source dirs (e.g. DAVIS),
        # selects the input frame subset, and defines each object as a union of
        # one or more segmentation palette IDs.
        data_path = os.path.join(dataset_path, scene_name)
        with open(os.path.join(data_path, "metadata.json")) as f:
            meta = json.load(f)

        # images_dir / segmentations_dir are repo-root-relative (Hydra chdir is
        # off), same convention as dataset.path itself.
        frames_path = meta["images_dir"]
        masks_path = meta["segmentations_dir"]

        # selected_frames lists the input RGB filenames in order; the matching
        # segmentation shares the stem with a .png extension (DAVIS convention).
        image_names = list(meta["selected_frames"])
        mask_names = [os.path.splitext(f)[0] + ".png" for f in image_names]

        # Object ID = obj_idx + 1 (background=0, foreground from 1; matches the
        # obj_NNN <-> obj_idx=NNN+1 GT-mesh convention). Each maps to the
        # palette IDs to union together at mask-load time.
        palette_groups = {
            int(obj["obj_idx"]) + 1: [int(p) for p in obj["palette_ids"]]
            for obj in meta["objects"]
        }

        return {
            "data_path": data_path,
            "frames_path": frames_path,
            "masks_path": masks_path,
            "image_names": image_names,
            "mask_names": mask_names,
            "depth_names": [],  # No GT depth, uses map-anything (or MoGe)
            "palette_groups": palette_groups,
            "dataset_type": "davis_actionmesh",
        }

    elif dataset_type == "dyncustom":
        # Own / uploaded mono-DYNAMIC clips: {path}/{scene}/rgbs/{NNNNN}.png +
        # segmentations/{NNNNN}.png -- the mvcustom layout, but each file is a
        # TIMESTEP rather than a view (see axis_lift_indices). The segmentation
        # palette value IS the object ID (0 = background), the DAVIS convention
        # load_masks already speaks; segmentations/classes.json only documents
        # the class/instance behind each ID, so no palette grouping is needed.
        # Per-frame GT meshes, when a scene has them, sit in a sibling
        # actionmesh/ dir -- read by gt_shapes_inversion, never by Sequence.
        data_path = os.path.join(dataset_path, scene_name)
        frames_path = os.path.join(data_path, "rgbs")
        masks_path = os.path.join(data_path, "segmentations")

        image_names = sorted(
            [f for f in os.listdir(frames_path)
             if os.path.splitext(f)[1].lower() in _IMAGE_EXTS],
            key=_natural_key,
        )
        mask_names = [os.path.splitext(f)[0] + ".png" for f in image_names]

        return {
            "data_path": data_path,
            "frames_path": frames_path,
            "masks_path": masks_path,
            "image_names": image_names,
            "mask_names": mask_names,
            "depth_names": [],  # No GT depth, uses map-anything
            "dataset_type": "dyncustom",
        }

    elif dataset_type == "oursactionbench":
        # Single-object scenes: imgs/{NN}.png are RGBA (alpha = silhouette).
        # GT geometry (obj_NNN/mesh_NN.glb) and the per-scene camera.json live at
        # the scene root — consumed by gt_shapes_inversion, not by Sequence.
        data_path = os.path.join(dataset_path, scene_name)
        frames_path = os.path.join(data_path, "imgs")

        # Filter RGB image candidates: ``.png`` only, no ``depth_`` prefix.
        # GT depth tiffs (``depth_*.tiff``) share the dir and would otherwise
        # be picked up here because ``_IMAGE_EXTS`` includes ``.tiff``.
        image_names = sorted(
            [
                f for f in os.listdir(frames_path)
                if os.path.splitext(f)[1].lower() in _IMAGE_EXTS
                and not f.startswith("depth_")
            ],
            key=_natural_key,
        )

        # GT z-depth tiffs (``depth_{stem}.tiff``, optional).
        # All-or-nothing: only populate ``depth_names`` if every input view has
        # one; otherwise the loader falls back to MoGe/map-anything (same as
        # the GSO branch below).
        depth_candidates = [
            f"depth_{os.path.splitext(f)[0]}.tiff" for f in image_names
        ]
        if all(
            os.path.isfile(os.path.join(frames_path, d)) for d in depth_candidates
        ):
            depth_names = depth_candidates
        else:
            depth_names = []

        return {
            "data_path": data_path,
            "frames_path": frames_path,
            "masks_path": frames_path,  # alpha lives in the same RGBA file
            "image_names": image_names,
            "mask_names": image_names,  # load_masks handles RGBA alpha
            "depth_names": depth_names,
            "dataset_type": dataset_type,
        }

    elif dataset_type == "image":
        # Structure: {path}/{scene_name}/{image}.{ext} + {stem}_mask.png
        scene_dir = os.path.join(dataset_path, scene_name)

        # Find the RGB image (first non-mask image file)
        candidates = [
            f for f in os.listdir(scene_dir)
            if os.path.splitext(f)[1].lower() in _IMAGE_EXTS
            and not os.path.splitext(f)[0].endswith("_mask")
        ]
        if not candidates:
            raise FileNotFoundError(
                f"No image file found in {scene_dir}. "
                f"Expected a file with extension: {', '.join(sorted(_IMAGE_EXTS))}"
            )
        image_file = sorted(candidates)[0]
        stem = os.path.splitext(image_file)[0]
        mask_file = f"{stem}_mask.png"
        mask_path = os.path.join(scene_dir, mask_file)
        if not os.path.isfile(mask_path):
            raise FileNotFoundError(
                f"Mask file not found: {mask_path}. "
                f"Expected '{mask_file}' alongside '{image_file}'."
            )

        return {
            "data_path": scene_dir,
            "frames_path": scene_dir,
            "masks_path": scene_dir,
            "image_names": [image_file],
            "mask_names": [mask_file],
            "depth_names": [],
            "dataset_type": "image",
        }

    elif dataset_type == "mvcustom":
        # Own multi-view static captures: {path}/{scene}/rgbs/{NNNNN}.png +
        # segmentations/{NNNNN}.png -- the dyncustom layout above, but each file is a
        # VIEW rather than a timestep (see axis_lift_indices). The segmentation
        # palette value IS the object ID (0 = background), the DAVIS convention
        # load_masks already speaks; segmentations/classes.json only documents
        # the class/instance behind each ID, so no palette grouping is needed.
        data_path = os.path.join(dataset_path, scene_name)
        frames_path = os.path.join(data_path, "rgbs")
        masks_path = os.path.join(data_path, "segmentations")

        image_names = sorted(
            [f for f in os.listdir(frames_path)
             if os.path.splitext(f)[1].lower() in _IMAGE_EXTS],
            key=_natural_key,
        )
        mask_names = [os.path.splitext(f)[0] + ".png" for f in image_names]

        return {
            "data_path": data_path,
            "frames_path": frames_path,
            "masks_path": masks_path,
            "image_names": image_names,
            "mask_names": mask_names,
            "depth_names": [],
            "dataset_type": "mvcustom",
        }

    elif dataset_type == "gso":
        # GSO-30 benchmark: {path}/{scene}/render_mvs_25/model/{idx:03d}.{png,npy}
        # RGBA images (alpha = object mask). 25 views total; views 0-9 are
        # training-input candidates, views 10-24 are held-out test views.
        # We always expose the full 10 training views here; num_input_views
        # subsets the pipeline-facing frames downstream (see
        # preprocessing.resolve_run_assets).
        num_input_views = kwargs.get("num_input_views")
        assert num_input_views is None or num_input_views <= 10, (
            f"GSO dataset: num_input_views must be <= 10 (views 0-9 are input "
            f"candidates; views 10-24 are held-out test views). Got {num_input_views}."
        )
        render_dir = os.path.join(dataset_path, scene_name, "render_mvs_25", "model")
        if not os.path.isdir(render_dir):
            raise FileNotFoundError(
                f"GSO render directory not found: {render_dir}. "
                f"Expected structure: {{path}}/{{scene}}/render_mvs_25/model/"
            )

        all_image_names = sorted(f for f in os.listdir(render_dir) if f.endswith(".png"))
        image_names = all_image_names[:10]

        # Per-view camera extrinsic matrices (.npy, 3x4 w2c)
        camera_pose_names = [f.replace(".png", ".npy") for f in image_names]

        # GT z-depth tiffs (``depth_{stem}.tiff``, optional).
        # All-or-nothing: only populate depth_names if every input view has one;
        # otherwise the loader falls back to MoGe with the standard warning.
        depth_candidates = [
            f"depth_{os.path.splitext(f)[0]}.tiff" for f in image_names
        ]
        if all(
            os.path.isfile(os.path.join(render_dir, d)) for d in depth_candidates
        ):
            depth_names = depth_candidates
        else:
            depth_names = []

        # Same RGBA file serves as both image (RGB via white-bg composite)
        # and mask (alpha channel)
        return {
            "data_path": os.path.join(dataset_path, scene_name),
            "frames_path": render_dir,
            "masks_path": render_dir,
            "image_names": image_names,
            "mask_names": image_names,  # load_masks handles RGBA alpha
            "depth_names": depth_names,
            "camera_pose_names": camera_pose_names,
            "dataset_type": "gso",
        }

    elif dataset_type == "co3d":
        # Flat LaRa-format CO3D scene:
        # only the 8 LaRa-selected views are materialized — ``{000..003}.png``
        # are input views (one per K-means cluster), ``{004..007}.png`` are
        # held-out NVS target views. Each view has a 3-channel RGB ``{NNN}.png``
        # (background kept, so map-anything sees scene context), a binary
        # foreground mask ``{NNN}_mask.png``, and a c2w 4x4 ``{NNN}.npy``.
        # Per-view FoVs live in ``fovs.npy`` (8, 2 radians); per-view metadata
        # (role / cluster / original frame index) in ``meta.json``. MV-static
        # (frame=0, view=asset).
        import re

        scene_dir = os.path.join(dataset_path, scene_name)
        if not os.path.isdir(scene_dir):
            raise FileNotFoundError(
                f"CO3D scene dir not found: {scene_dir}. "
                f"Prepare the LaRa-format scene first."
            )
        # RGB image PNGs are bare 3-digit names; mask PNGs share the prefix
        # plus ``_mask`` suffix (diagnostic PNGs have descriptive names).
        rx = re.compile(r"^\d{3}\.png$")
        image_names = sorted(f for f in os.listdir(scene_dir) if rx.match(f))
        if not image_names:
            raise FileNotFoundError(
                f"No {{NNN}}.png view files found in {scene_dir}. "
                f"Re-prepare the LaRa-format scene."
            )
        mask_names = [f"{os.path.splitext(f)[0]}_mask.png" for f in image_names]
        camera_pose_names = [f"{os.path.splitext(f)[0]}.npy" for f in image_names]
        for stem_list, label in [(mask_names, "_mask.png"), (camera_pose_names, ".npy")]:
            if not all(os.path.isfile(os.path.join(scene_dir, n)) for n in stem_list):
                raise FileNotFoundError(
                    f"Missing per-view {label} files in {scene_dir}."
                )
        for required in ("fovs.npy", "meta.json"):
            if not os.path.isfile(os.path.join(scene_dir, required)):
                raise FileNotFoundError(f"{scene_dir}/{required} missing.")

        return {
            "data_path": scene_dir,
            "scene_name": scene_name,
            "dataset_type": "co3d",
            "frames_path": scene_dir,
            "masks_path": scene_dir,
            "image_names": image_names,
            "mask_names": mask_names,
            "depth_names": [],
            "camera_pose_names": camera_pose_names,
            "fovs_path": os.path.join(scene_dir, "fovs.npy"),
            "meta_path": os.path.join(scene_dir, "meta.json"),
        }

    else:
        raise ValueError(
            f"Unknown dataset type: {dataset_type}. Supported: 'davis_actionmesh', "
            f"'dyncustom', 'oursactionbench', 'image', 'mvcustom', 'gso', 'co3d'"
        )


def save_mesh_to_obj(mesh: "torch.Tensor", output_path: str) -> None:
    """
    Save a mesh object to an OBJ file.

    Parameters
    ----------
    mesh : MeshExtractResult or similar
        Mesh object with the following attributes:
        - vertices or verts: Tensor of shape (N, 3)
        - faces: Tensor of shape (M, 3)
        - vertex_attrs (optional): Can be:
          - A tensor of shape (N, C) where C >= 3 (first 3 channels are RGB color)
          - A dict with 'color' key
          - None
        - vertex_colors (optional): Alternative to vertex_attrs
    output_path : str
        Path to save the OBJ file.

    Notes
    -----
    - OBJ files use 1-indexed vertices
    - Vertex colors are clamped to [0, 1] range
    - Colors are written in the "v x y z r g b" format

    Examples
    --------
    >>> save_mesh_to_obj(mesh, "output/model.obj")
    Saved mesh to output/model.obj (10000 vertices, 20000 faces)
    """
    # Handle both 'vertices' and 'verts' attribute names
    if hasattr(mesh, "vertices"):
        verts = mesh.vertices.cpu().numpy() if hasattr(mesh.vertices, "cpu") else mesh.vertices
    elif hasattr(mesh, "verts"):
        verts = mesh.verts.cpu().numpy() if hasattr(mesh.verts, "cpu") else mesh.verts
    else:
        raise AttributeError("Mesh object has no 'vertices' or 'verts' attribute")

    faces = mesh.faces.cpu().numpy() if hasattr(mesh.faces, "cpu") else mesh.faces

    # Check for vertex colors
    vertex_colors = None
    if hasattr(mesh, "vertex_attrs") and mesh.vertex_attrs is not None:
        va = mesh.vertex_attrs
        # vertex_attrs can be a tensor directly or a dict
        if isinstance(va, dict):
            if "color" in va:
                vc = va["color"]
                vertex_colors = vc.cpu().numpy() if hasattr(vc, "cpu") else vc
        elif hasattr(va, "cpu"):
            # It's a tensor - assume first 3 channels are RGB
            va_np = va.cpu().numpy()
            if va_np.shape[-1] >= 3:
                vertex_colors = va_np[..., :3]
        elif isinstance(va, np.ndarray):
            if va.shape[-1] >= 3:
                vertex_colors = va[..., :3]
    elif hasattr(mesh, "vertex_colors") and mesh.vertex_colors is not None:
        vertex_colors = (
            mesh.vertex_colors.cpu().numpy()
            if hasattr(mesh.vertex_colors, "cpu")
            else mesh.vertex_colors
        )

    with open(output_path, "w") as f:
        f.write(f"# OBJ file with {len(verts)} vertices and {len(faces)} faces\n")

        # Write vertices (with colors if available)
        for i, v in enumerate(verts):
            if vertex_colors is not None:
                c = vertex_colors[i]
                # Clamp colors to [0, 1]
                c = np.clip(c, 0, 1)
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")
            else:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        # Write faces (OBJ uses 1-indexed vertices)
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    print(f"Saved mesh to {output_path} ({len(verts)} vertices, {len(faces)} faces)")


def _reject_non_finite_gaussian(values, path: str) -> None:
    """Raise if a Gaussian carries NaN anywhere, or a non-finite POSITION.

    A non-finite mean is not merely bad data downstream: ``gsconverter``'s Morton
    sort clips the coordinates into buckets that a NaN can never fall into, so the
    partition never shrinks and ``recursive_sort`` dies with a ``RecursionError``
    thousands of frames deep, naming neither the file nor the real cause.  Fail here
    instead, where the offending asset is known.

    ``inf`` is fatal only in the positions: both writers store opacity as
    ``logit(sigmoid(x))``, which legitimately saturates to +-inf in float32 for a
    fully opaque or fully transparent Gaussian, and rejecting that would turn
    exports that have always worked into hard failures.  NaN is never legitimate
    anywhere, so it is rejected across every attribute.
    """
    bad = np.isnan(values)
    bad[:, :3] |= ~np.isfinite(values[:, :3])      # both writers lay x,y,z out first
    if bad.any():
        rows = int(bad.any(axis=1).sum())
        raise ValueError(
            f"Refusing to write {path}: {rows}/{values.shape[0]} Gaussians carry "
            f"NaN attributes or non-finite positions.  The geometry feeding this "
            f"export diverged — fix that upstream rather than writing an unusable "
            f"asset.")


def save_gaussian_ply(gaussian, path: str) -> None:
    """Save a Gaussian to PLY, including SH rest bands if present.

    Unlike the submodule's ``Gaussian.save_ply`` which only saves DC features,
    this function also writes ``_features_rest`` (higher-order SH coefficients)
    so that view-dependent appearance is preserved in the output file.
    """
    import torch
    from plyfile import PlyData, PlyElement

    inverse_sigmoid = lambda x: torch.log(x / (1 - x))

    xyz = gaussian.get_xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = (gaussian._features_dc.detach()
            .transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy())
    opacity = gaussian.get_opacity.detach()
    opacities = inverse_sigmoid(opacity).cpu().numpy()
    scale = torch.log(gaussian.get_scaling).detach().cpu().numpy()
    rotation = (gaussian._rotation + gaussian.rots_bias[None, :]).detach().cpu().numpy()

    # Build attribute list
    attrs = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(f_dc.shape[1]):
        attrs.append(f"f_dc_{i}")

    # SH rest features
    f_rest = None
    if gaussian._features_rest is not None:
        f_rest = (gaussian._features_rest.detach()
                  .transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy())
        for i in range(f_rest.shape[1]):
            attrs.append(f"f_rest_{i}")

    attrs.append("opacity")
    for i in range(scale.shape[1]):
        attrs.append(f"scale_{i}")
    for i in range(rotation.shape[1]):
        attrs.append(f"rot_{i}")

    dtype_full = [(a, "f4") for a in attrs]
    data = np.empty(xyz.shape[0], dtype=dtype_full)
    arrays = [xyz, normals, f_dc]
    if f_rest is not None:
        arrays.append(f_rest)
    arrays.extend([opacities, scale, rotation])
    values = np.concatenate(arrays, axis=1)
    _reject_non_finite_gaussian(values, path)
    data[:] = list(map(tuple, values))

    el = PlyElement.describe(data, "vertex")
    PlyData([el]).write(path)


def save_gaussian_compressed_ply(gaussian, path: str) -> None:
    """Save a Gaussian as compressed PLY using gsconverter (no intermediate file).

    Builds the structured numpy array in-memory from the Gaussian model's
    properties and writes it directly via ``CompressedPlyFormat``.

    Parameters
    ----------
    gaussian : Gaussian
        A decoded Gaussian splatting model.
    path : str
        Output file path (.ply).
    """
    import torch
    from gsconverter.formats.compressed_ply import CompressedPlyFormat

    xyz = gaussian.get_xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = (gaussian._features_dc.detach()
            .transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy())
    opacity = gaussian.get_opacity.detach()
    opacities = torch.log(opacity / (1 - opacity)).cpu().numpy()
    scale = torch.log(gaussian.get_scaling).detach().cpu().numpy()
    rotation = (gaussian._rotation + gaussian.rots_bias[None, :]).detach().cpu().numpy()

    # Build attribute list (include SH rest if present)
    attrs = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(f_dc.shape[1]):
        attrs.append(f"f_dc_{i}")
    f_rest = None
    if gaussian._features_rest is not None:
        f_rest = (gaussian._features_rest.detach()
                  .transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy())
        for i in range(f_rest.shape[1]):
            attrs.append(f"f_rest_{i}")
    attrs.append("opacity")
    for i in range(scale.shape[1]):
        attrs.append(f"scale_{i}")
    for i in range(rotation.shape[1]):
        attrs.append(f"rot_{i}")

    dtype_full = [(a, "f4") for a in attrs]
    data = np.empty(xyz.shape[0], dtype=dtype_full)
    arrays = [xyz, normals, f_dc]
    if f_rest is not None:
        arrays.append(f_rest)
    arrays.extend([opacities, scale, rotation])
    values = np.concatenate(arrays, axis=1)
    _reject_non_finite_gaussian(values, path)
    data[:] = list(map(tuple, values))

    CompressedPlyFormat().write(data, path)


def save_perframe_ply(perframe_gaussians, tokens_by_object, output_dir, scene_name,
                      tag="perframe", compressed=False):
    """Save decoded per-frame Gaussians as PLY files.

    Parameters
    ----------
    perframe_gaussians : dict
        Nested dict: {obj_idx: {frame_idx: Gaussian}}.
    tokens_by_object : dict
        Dictionary mapping obj_idx -> list of (frame_idx, decoder_input).
        Used to determine which objects/frames to save.
    output_dir : str
        Directory to save PLY files.
    scene_name : str
        Scene name for filenames.
    tag : str, optional
        Tag for filenames (e.g., "initial", "perframe"). Default: "perframe".
    compressed : bool, optional
        If True, save as compressed PLY instead of standard PLY. Default: False.
    """
    fmt = "compressed PLY" if compressed else "PLY"
    print(f"\n  Saving decoded Gaussians as {fmt} files ({tag})...")
    os.makedirs(output_dir, exist_ok=True)

    for obj_idx in sorted(tokens_by_object.keys()):
        for frame_idx, _ in tokens_by_object[obj_idx]:
            gaussian = perframe_gaussians[obj_idx][frame_idx]
            ext = ".compressed.ply" if compressed else ".ply"
            ply_path = os.path.join(
                output_dir,
                f"{scene_name}_obj{obj_idx}_f{frame_idx}_{tag}{ext}"
            )
            if compressed:
                save_gaussian_compressed_ply(gaussian, ply_path)
            else:
                save_gaussian_ply(gaussian, ply_path)
            print(f"    Saved {fmt}: object {obj_idx} frame {frame_idx}")


def save_perframe_meshes(tokens_by_object, pipeline, output_dir, scene_name, tag="perframe"):
    """Save decoded per-frame meshes as OBJ files.

    Re-decodes SLAT tokens to meshes and saves each as an OBJ file.

    Parameters
    ----------
    tokens_by_object : dict
        Dictionary mapping obj_idx -> list of (frame_idx, decoder_input).
        Each decoder_input must contain 'decoder_input_slat'.
    pipeline : Pipeline
        The inference pipeline (used for re-decoding SLAT -> mesh).
    output_dir : str
        Directory to save OBJ files.
    scene_name : str
        Scene name for filenames.
    tag : str, optional
        Tag for filenames (e.g., "initial", "perframe"). Default: "perframe".
    """
    from genia.core.utils.slat_decode import redecode_slat

    print(f"\n  Saving decoded meshes as OBJ files ({tag})...")
    os.makedirs(output_dir, exist_ok=True)

    for obj_idx in sorted(tokens_by_object.keys()):
        for frame_idx, decoder_input in tokens_by_object[obj_idx]:
            slat = decoder_input["decoder_input_slat"]
            decoded_mesh = redecode_slat(pipeline, slat, formats=["mesh"])
            if "mesh" in decoded_mesh and decoded_mesh["mesh"]:
                mesh = decoded_mesh["mesh"][0]
                mesh_path = os.path.join(
                    output_dir,
                    f"{scene_name}_obj{obj_idx}_f{frame_idx}_{tag}.obj"
                )
                save_mesh_to_obj(mesh, mesh_path)
                print(f"    Saved mesh: object {obj_idx} frame {frame_idx}")


def save_per_object_ply_with_poses(
    canonical_gaussians,
    interpolated_poses,
    all_frame_indices,
    output_dir,
    scene_name="scene",
    compressed=False,
    c2w_per_frame=None,
    K_per_frame=None,
    canonical_mesh_verts=None,
    per_frame_mesh_verts=None,
    per_frame_mesh_rotations=None,
    canonical_mesh_faces=None,
    warp_knn_k: int = 4,
    warp_knn_eps: float = 1.0e-8,
    warp_knn_chunk_size: int = 8192,
    mesh_objs=None,
):
    """Save per-object canonical Gaussians as individual PLY files with a poses.json.

    Mesh-only reconstructions (``canonical_gaussians`` empty) still get a full
    ``poses.json``: the object loop runs over ``canonical_gaussians ∪ interpolated_poses``
    and the per-object ``"ply"`` / warped-PLY writes are skipped where an object has no
    Gaussian; ``mesh_objs`` (a set of obj_idx that have a ``meshes/{obj:03d}.glb``) adds a
    ``"mesh"`` reference to those entries.  For runs with Gaussians the union equals the
    gaussian keys.

    Each object's canonical Gaussian is saved in its local (PyTorch3D) space
    under ``{output_dir}/gaussians/{obj_idx:03d}.ply``.  A single
    ``{output_dir}/poses.json`` maps each object to its per-frame transforms
    (translation, rotation, scale) -- these transforms also apply to the
    per-object meshes saved under ``{output_dir}/meshes/``, so ``poses.json``
    lives at the run-final level rather than inside ``gaussians/``.  Per-frame
    color shifts (``dc_offset``, ``sh_rest``) are saved as flat Float32 binary
    files alongside the PLY.

    When the per-canonical-mesh-vertex deformation field is provided
    (``canonical_mesh_verts`` + the two per-frame dicts; same shape as
    ``state.canonical_mesh_*``), additionally writes a per-frame **warped**
    PLY for each keyframe at which the field is keyed:

        ``{output_dir}/gaussians/{obj_idx:03d}/{frame_idx:03d}.ply``

    The Sim(3) transform in ``poses.json`` still applies on top — the per-frame
    PLY has the *non-rigid deformation* baked in (the part that can't be
    encoded in a Sim(3)).  Each transform entry in ``poses.json`` gains an
    optional ``"ply_perframe"`` field referencing the warped asset; consumers
    that don't understand it can keep using the canonical PLY + Sim(3).

    Parameters
    ----------
    canonical_gaussians : dict
        ``{obj_idx: Gaussian}`` canonical Gaussian objects.
    interpolated_poses : dict
        Output of ``interpolate_poses()``:
        ``{obj_idx: {frame_idx: {"rotation": (4,), "translation": (3,), "scale": (3,)}}}``.
    all_frame_indices : list of int
        Every frame index to include in poses (e.g. ``range(num_frames)``).
    output_dir : str
        Run final directory.  PLY files go to ``{output_dir}/gaussians/``;
        ``poses.json`` goes to ``{output_dir}/poses.json``.
    scene_name : str, optional
        Scene name (metadata only). Default: "scene".
    compressed : bool, optional
        If True, save as compressed PLY. Default: False.
    c2w_per_frame : dict, optional
        ``{frame_idx: np.ndarray(4,4)}`` camera-to-world transforms per frame.
        Included in ``poses.json`` under ``"cameras"`` key.
        Defaults to identity for all frames if not provided.
    """
    import torch

    # Only create gaussians/ when there is at least one Gaussian to write — a run that
    # withholds its Gaussians (canonical_gaussians empty) gets poses.json but no gaussians/ dir.
    gaussians_dir = os.path.join(output_dir, "gaussians")
    if canonical_gaussians:
        os.makedirs(gaussians_dir, exist_ok=True)

    ext = ".compressed.ply" if compressed else ".ply"
    fmt = "compressed PLY" if compressed else "PLY"

    def _to_list(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().flatten().tolist()
        elif isinstance(x, np.ndarray):
            return x.flatten().tolist()
        return [float(x)]

    def _to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    objects_json = []

    # Lazy import — avoids pulling rendering_guidance at module load time.
    if canonical_mesh_verts is not None:
        from .gaussian import create_gaussians_object
        from genia.core.utils.deformation import _lookup_per_frame_deformation, warp_gaussians_high_res

    mesh_objs = set(mesh_objs or ())
    for obj_idx in sorted(set(canonical_gaussians) | set(interpolated_poses)):
        has_gaussian = obj_idx in canonical_gaussians
        ply_filename = None
        if not has_gaussian:
            # Mesh-only object: no PLY, transforms (+ optional mesh ref) only.
            pass
        else:
            ply_filename = f"{obj_idx:03d}{ext}"
            ply_path = os.path.join(gaussians_dir, ply_filename)
            gs = canonical_gaussians[obj_idx]
            if compressed:
                save_gaussian_compressed_ply(gs, ply_path)
            else:
                save_gaussian_ply(gs, ply_path)
            print(f"  Saved object {obj_idx} {fmt}: {ply_path}")

        # Collect per-frame transforms and color shifts
        transforms = []
        dc_offset_frames = []
        sh_rest_frames = []
        obj_poses = interpolated_poses.get(obj_idx, {})

        # Per-frame warped PLY writes go under gaussians/{obj_idx:03d}/.
        # Only created when the deformation field is keyed for this object AND a
        # Gaussian exists to warp (objects with no Gaussian skip the warped-PLY write).
        _gs_canon = canonical_gaussians[obj_idx] if has_gaussian else None
        _has_deformation = (
            has_gaussian
            and canonical_mesh_verts is not None
            and obj_idx in (canonical_mesh_verts or {})
        )
        perframe_dir = None
        if _has_deformation:
            perframe_dir = os.path.join(gaussians_dir, f"{obj_idx:03d}")
            os.makedirs(perframe_dir, exist_ok=True)

        for fi in all_frame_indices:
            if fi not in obj_poses:
                continue
            pose = obj_poses[fi]
            # FrameKey-aware: emit explicit "frame" and "view" int fields so
            # readers can disambiguate (frame, view) without parsing the key
            # representation. Default view=0 for a bare-int key.
            frame_n = fi.frame if hasattr(fi, "frame") else int(fi)
            view_n = fi.view if hasattr(fi, "view") else 0
            xform = {
                "frame": int(frame_n),
                "view": int(view_n),
                "translation": _to_list(pose["translation"]),
                "rotation": _to_list(pose["rotation"]),
                "scale": _to_list(pose["scale"]),
            }

            # Per-frame warped PLY (actionmesh).  Bake the non-rigid
            # deformation into the canonical-space asset so downstream
            # consumers can apply ``transforms`` (Sim(3)) on top.
            if perframe_dir is not None:
                resolved = _lookup_per_frame_deformation(
                    canonical_mesh_verts,
                    per_frame_mesh_verts,
                    per_frame_mesh_rotations,
                    obj_idx, int(frame_n), _gs_canon.get_xyz.device,
                    canonical_mesh_faces,
                )
                if resolved is not None:
                    *_warp_core, _faces = resolved
                    with torch.no_grad():
                        means_w, quats_w = warp_gaussians_high_res(
                            _gs_canon, *_warp_core,
                            K=warp_knn_k,
                            eps=warp_knn_eps,
                            chunk_size=warp_knn_chunk_size,
                            faces=_faces,
                        )
                        gs_warped = create_gaussians_object(
                            xyz=means_w,
                            features=_gs_canon.get_features,
                            scales=_gs_canon.get_scaling,
                            rots=quats_w,
                            opacities=_gs_canon.get_opacity,
                        )
                    pf_filename = f"{int(frame_n):03d}{ext}"
                    pf_path = os.path.join(perframe_dir, pf_filename)
                    if compressed:
                        save_gaussian_compressed_ply(gs_warped, pf_path)
                    else:
                        save_gaussian_ply(gs_warped, pf_path)
                    xform["ply_perframe"] = (
                        f"gaussians/{obj_idx:03d}/{pf_filename}"
                    )

            transforms.append(xform)
            if "dc_offset" in pose:
                dc_offset_frames.append(_to_numpy(pose["dc_offset"]))
            if "sh_rest" in pose:
                sh_rest_frames.append(_to_numpy(pose["sh_rest"]))

        obj_entry = {
            "obj_idx": int(obj_idx),
            "transforms": transforms,
        }
        if ply_filename is not None:
            obj_entry["ply"] = f"gaussians/{ply_filename}"
        if obj_idx in mesh_objs:
            obj_entry["mesh"] = f"meshes/{obj_idx:03d}.glb"

        # Save per-frame dc_offset as binary Float32: shape (T, N, 3)
        if dc_offset_frames:
            dc_arr = np.stack(dc_offset_frames, axis=0).squeeze(2).astype(np.float32)  # (T, N, 1, 3) -> (T, N, 3)
            dc_filename = f"{obj_idx:03d}_dc_offsets.bin"
            dc_arr.tofile(os.path.join(gaussians_dir, dc_filename))
            obj_entry["dc_offsets_file"] = f"gaussians/{dc_filename}"
            obj_entry["dc_offsets_shape"] = list(dc_arr.shape)
            print(f"  Saved dc_offsets for object {obj_idx}: {dc_filename} {list(dc_arr.shape)}")

        # Save per-frame sh_rest as binary Float32: shape (T, N, K, 3)
        if sh_rest_frames:
            sh_arr = np.stack(sh_rest_frames, axis=0).astype(np.float32)  # (T, N, K, 3)
            sh_filename = f"{obj_idx:03d}_sh_rest.bin"
            sh_arr.tofile(os.path.join(gaussians_dir, sh_filename))
            obj_entry["sh_rest_file"] = f"gaussians/{sh_filename}"
            obj_entry["sh_rest_shape"] = list(sh_arr.shape)
            print(f"  Saved sh_rest for object {obj_idx}: {sh_filename} {list(sh_arr.shape)}")

        objects_json.append(obj_entry)

    # Build per-frame camera entries
    cameras_json = []
    for fi in all_frame_indices:
        if c2w_per_frame is not None and fi in c2w_per_frame:
            c2w_matrix = c2w_per_frame[fi]
        else:
            c2w_matrix = np.eye(4, dtype=np.float32)
        frame_n = fi.frame if hasattr(fi, "frame") else int(fi)
        view_n = fi.view if hasattr(fi, "view") else 0
        cam_entry = {
            "frame": int(frame_n),
            "view": int(view_n),
            "frame_idx": int(frame_n),  # same as "frame", for readers keyed on frame_idx
            "c2w": c2w_matrix.tolist(),
        }
        # Per-frame intrinsics (3x3) when available — required to project the
        # 3D object tracks to 2D (e.g. for point-tracking evaluation); omitted
        # otherwise.
        if K_per_frame is not None and fi in K_per_frame:
            K_mat = K_per_frame[fi]
            cam_entry["K"] = (K_mat.tolist() if hasattr(K_mat, "tolist") else K_mat)
        cameras_json.append(cam_entry)

    # Write poses.json at the run-final level: per-frame object transforms +
    # cameras apply to every per-object asset (gaussians, meshes), not just
    # the gaussian PLYs.  Asset file paths are stored relative to the final
    # directory (e.g. ``gaussians/000.ply``).
    poses_json = {"objects": objects_json, "cameras": cameras_json}
    poses_path = os.path.join(output_dir, "poses.json")
    with open(poses_path, "w") as f:
        json.dump(poses_json, f, indent=2)
    print(f"  Saved poses: {poses_path}")


def save_per_object_mesh(
    canonical_slats,
    pipeline_obj,
    output_dir,
):
    """Save per-object canonical meshes (decoded from SLAT) to ``{output_dir}/meshes/``.

    Each mesh is written as ``{obj_idx:03d}.glb`` in the object's local
    PyTorch3D space — the same convention used by
    ``save_per_object_ply_with_poses`` for canonical Gaussians — so the
    shared ``{output_dir}/poses.json`` describes per-frame transforms for
    these meshes equally.  When available, per-vertex RGB (from the mesh
    decoder's ``vertex_attrs``) is written into the GLB.

    Per-frame *warped* meshes are not written: GLB encoding is expensive,
    and the per-frame Gaussians under ``gaussians/{NNN}/`` carry the
    deformation at much lower I/O cost.

    Parameters
    ----------
    canonical_slats : dict
        ``{obj_idx: SparseTensor}`` canonical SLAT tokens per object.
    pipeline_obj : Pipeline
        SAM3D pipeline used to decode SLAT -> mesh via ``redecode_slat``.
    output_dir : str
        Run output directory. Files are written to ``{output_dir}/meshes/``.
    """
    import trimesh

    from genia.core.utils.slat_decode import redecode_slat

    meshes_dir = os.path.join(output_dir, "meshes")
    os.makedirs(meshes_dir, exist_ok=True)

    for obj_idx in sorted(canonical_slats.keys()):
        decoded = redecode_slat(pipeline_obj, canonical_slats[obj_idx], formats=["mesh"])
        mesh_res = decoded["mesh"][0]
        verts = mesh_res.vertices.detach().cpu().numpy()
        faces = mesh_res.faces.detach().cpu().numpy()
        vertex_colors = _extract_mesh_vertex_colors(mesh_res, verts.shape[0])

        mesh_filename = f"{obj_idx:03d}.glb"
        mesh_path = os.path.join(meshes_dir, mesh_filename)
        trimesh.Trimesh(
            vertices=verts, faces=faces,
            vertex_colors=vertex_colors, process=False,
        ).export(mesh_path)
        print(
            f"  Saved object {obj_idx} mesh: {mesh_path} "
            f"({verts.shape[0]} verts, {faces.shape[0]} faces"
            f"{', with colors' if vertex_colors is not None else ''})"
        )


def _extract_mesh_vertex_colors(mesh_res, n_verts: int):
    """Extract per-vertex RGB (uint8, shape (N, 3)) from a MeshExtractResult.

    Returns None if no colors are available. SAM3D's SLatMeshDecoder stores
    colors in ``vertex_attrs`` (first 3 channels = RGB in [0, 1]); some
    versions use ``vertex_colors``. Values are clamped to [0, 1] and
    converted to uint8.
    """
    import numpy as _np

    arr = None
    va = getattr(mesh_res, "vertex_attrs", None)
    if va is not None:
        if isinstance(va, dict):
            c = va.get("color")
            if c is not None:
                arr = c.detach().cpu().numpy() if hasattr(c, "detach") else _np.asarray(c)
        elif hasattr(va, "detach"):
            arr = va.detach().cpu().numpy()
        else:
            arr = _np.asarray(va)
    if arr is None:
        vc = getattr(mesh_res, "vertex_colors", None)
        if vc is not None:
            arr = vc.detach().cpu().numpy() if hasattr(vc, "detach") else _np.asarray(vc)
    if arr is None:
        return None

    if arr.ndim != 2 or arr.shape[0] != n_verts or arr.shape[1] < 3:
        return None
    rgb = _np.clip(arr[:, :3], 0.0, 1.0)
    return (rgb * 255.0).astype(_np.uint8)


def save_perframe_per_object_ply(
    perframe_gaussians,
    output_dir,
    compressed=False,
):
    """Save per-frame Gaussians per object: ``gaussians/{obj:03d}/{frame:03d}.ply``.

    Object-local (no pose baked) — ``final/poses.json`` carries the per-frame
    Sim(3), matching the canonical :func:`save_per_object_ply_with_poses`
    per-frame layout. The dataset-agnostic per-frame-only Gaussian output.

    Parameters
    ----------
    perframe_gaussians : dict
        ``{obj_idx: {frame_key: Gaussian}}`` per-frame decoded Gaussians.
    output_dir : str
        Run output directory; PLYs go to ``{output_dir}/gaussians/``.
    compressed : bool
        Save ``.compressed.ply`` instead of ``.ply``.
    """
    from .frame_key import frame_key_stem

    gaussians_dir = os.path.join(output_dir, "gaussians")
    ext = ".compressed.ply" if compressed else ".ply"
    save = save_gaussian_compressed_ply if compressed else save_gaussian_ply

    n = 0
    for obj_idx, per_frame in sorted(perframe_gaussians.items()):
        if not per_frame:
            continue
        obj_dir = os.path.join(gaussians_dir, f"{obj_idx:03d}")
        os.makedirs(obj_dir, exist_ok=True)
        for fk, gs in per_frame.items():
            save(gs, os.path.join(obj_dir, f"{frame_key_stem(fk)}{ext}"))
            n += 1
    print(f"  Saved {n} per-object per-frame PLYs to {gaussians_dir}")


def save_perframe_poses_json(
    perframe_gaussians,
    interpolated_poses,
    all_frame_indices,
    output_dir,
    c2w_per_frame=None,
    K_per_frame=None,
    compressed=False,
):
    """Write ``final/poses.json`` for per-frame-only pipelines.

    Per-frame-only runs have no canonical Gaussian/mesh — each frame owns an
    object-local ``gaussians/{obj:03d}/{frame:03d}.ply`` (from
    :func:`save_perframe_per_object_ply`) plus, for OursActionBench, a
    world-placed ``meshes/{obj:03d}/{frame:03d}.ply``.  ``poses.json``
    carries the per-frame Sim(3) (so a consumer can compose the object-local
    Gaussian into world) and the per-frame cameras, matching the canonical
    :func:`save_per_object_ply_with_poses` schema.  A 3D evaluator needs this
    file to resolve ``obj_idx`` for per-frame-only runs.

    Parameters mirror :func:`save_per_object_ply_with_poses`:
    ``interpolated_poses`` is ``{obj_idx: {frame_key: {"rotation",
    "translation", "scale"}}}`` (from ``interpolate_poses``);
    ``all_frame_indices`` the ordered frame keys; ``c2w_per_frame`` the
    per-frame camera-to-world (identity when absent).
    """
    import torch

    def _to_list(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().flatten().tolist()
        if isinstance(x, np.ndarray):
            return x.flatten().tolist()
        return [float(x)]

    ext = ".compressed.ply" if compressed else ".ply"
    objects_json = []
    for obj_idx in sorted(perframe_gaussians.keys()):
        if not perframe_gaussians[obj_idx]:
            continue
        obj_poses = interpolated_poses.get(obj_idx, {})
        transforms = []
        for fi in all_frame_indices:
            if fi not in obj_poses:
                continue
            pose = obj_poses[fi]
            frame_n = fi.frame if hasattr(fi, "frame") else int(fi)
            view_n = fi.view if hasattr(fi, "view") else 0
            transforms.append({
                "frame": int(frame_n),
                "view": int(view_n),
                "translation": _to_list(pose["translation"]),
                "rotation": _to_list(pose["rotation"]),
                "scale": _to_list(pose["scale"]),
                "ply_perframe": (
                    f"gaussians/{obj_idx:03d}/{int(frame_n):03d}{ext}"
                ),
            })
        if not transforms:
            continue
        objects_json.append({
            "obj_idx": int(obj_idx),
            "ply": f"gaussians/{obj_idx:03d}",
            "transforms": transforms,
        })

    cameras_json = []
    for fi in all_frame_indices:
        if c2w_per_frame is not None and fi in c2w_per_frame:
            c2w_matrix = c2w_per_frame[fi]
        else:
            c2w_matrix = np.eye(4, dtype=np.float32)
        frame_n = fi.frame if hasattr(fi, "frame") else int(fi)
        view_n = fi.view if hasattr(fi, "view") else 0
        cam_entry = {
            "frame": int(frame_n),
            "view": int(view_n),
            "frame_idx": int(frame_n),  # same as "frame"
            "c2w": (c2w_matrix.tolist() if hasattr(c2w_matrix, "tolist")
                    else c2w_matrix),
        }
        if K_per_frame is not None and fi in K_per_frame:
            K_mat = K_per_frame[fi]
            cam_entry["K"] = (K_mat.tolist() if hasattr(K_mat, "tolist") else K_mat)
        cameras_json.append(cam_entry)

    poses_json = {"objects": objects_json, "cameras": cameras_json}
    poses_path = os.path.join(output_dir, "poses.json")
    with open(poses_path, "w") as f:
        json.dump(poses_json, f, indent=2)
    print(f"  Saved poses: {poses_path}")


__all__ = [
    "load_image",
    "load_masks",
    "setup_paths",
    "save_mesh_to_obj",
    "save_perframe_ply",
    "save_perframe_meshes",
    "save_perframe_per_object_ply",
    "save_perframe_poses_json",
    "save_gaussian_ply",
    "save_gaussian_compressed_ply",
    "save_per_object_mesh",
]
