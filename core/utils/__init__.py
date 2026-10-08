# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
SAM3D-Objects Utilities Package.

This package provides utilities for the SAM3D-Objects 3D Gaussian splatting
pipeline, including depth processing, Gaussian operations, rendering,
pose refinement, and evaluation.

Imports are lazy: only the lightweight ``config`` and ``pipeline_state``
modules are loaded eagerly.  All other modules (rendering, refinement,
gaussian, ...) are imported on first access, keeping ``--help`` and
config-only paths fast.

Module Structure
----------------
- config: Configuration dataclasses and output flag resolution
- pipeline_state: PipelineState dataclass and cache I/O
- model_cache: Singleton cache for lazy-loaded models (perceptual, reconstruction backends)
- depth: Depth processing and pointmap generation
- io_utils: File I/O operations (load/save images, masks, meshes)
- slat_decode: decoding SAM3D shape tokens / SLATs into geometry and Gaussians
- gaussian: Gaussian splatting operations
- rendering: Differentiable rendering with gsplat
- refinement: Pose refinement using differentiable rendering
- evaluation: Metrics computation, keyframe rendering, block evaluation helpers
- visualization: Plotting and visualization utilities
"""

import importlib as _importlib

# ---------------------------------------------------------------------------
# Config — lightweight (stdlib + hydra/omegaconf only), safe to import eagerly
# ---------------------------------------------------------------------------
from .config import (
    # Leaf configs
    DatasetConfig,
    ProcessingConfig,
    OutputConfig,
    # Loss config
    LossConfig,
    # Per-block refinement configs
    GlobalPoseRefineConfig,
    # Pipeline
    PipelineConfig,
    # Top-level
    GeneralConfig,
    # Utilities
    print_loss_config,
    # Per-block output flag resolution
    block_config,
    get_block_output_flag,
    output_dir_redirect,
    block_output_context,
    block_output_subdir,
)

# Pipeline state — lightweight (stdlib + dataclasses only)
from .pipeline_state import (
    PipelineState,
    get_pipeline_cache_dir,
    save_pipeline_cache,
    load_pipeline_cache,
    find_latest_pipeline_cache,
)

# ModelCache — lightweight singleton (stdlib + torch only)
from .model_cache import ModelCache

# ---------------------------------------------------------------------------
# Lazy-import registry: attribute name -> (submodule, attribute)
# ---------------------------------------------------------------------------
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Depth processing
    "compute_conegs_scaling":              (".depth", "compute_conegs_scaling"),
    "depth_to_pointmap":                   (".depth", "depth_to_pointmap"),
    "load_and_process_depth":              (".depth", "load_and_process_depth"),
    "transform_to_pytorch3d_convention":   (".depth", "transform_to_pytorch3d_convention"),
    # Gaussian operations
    "C0":                                  (".gaussian", "C0"),
    "RGB2SH":                              (".gaussian", "RGB2SH"),
    "SH2RGB":                              (".gaussian", "SH2RGB"),
    "create_background_gaussians":         (".gaussian", "create_background_gaussians"),
    "create_gaussians_from_pointmap":      (".gaussian", "create_gaussians_from_pointmap"),
    "create_gaussians_object":             (".gaussian", "create_gaussians_object"),
    "join_gaussians":                      (".gaussian", "join_gaussians"),
    "transform_scene_to_r3_convention":    (".gaussian", "transform_scene_to_r3_convention"),
    # I/O utilities
    "load_image":                          (".io_utils", "load_image"),
    "load_masks":                          (".io_utils", "load_masks"),
    "MaskDict":                            (".io_utils", "MaskDict"),
    "save_mesh_to_obj":                    (".io_utils", "save_mesh_to_obj"),
    "save_perframe_ply":                   (".io_utils", "save_perframe_ply"),
    "save_perframe_meshes":                (".io_utils", "save_perframe_meshes"),
    "save_per_object_ply_with_poses":      (".io_utils", "save_per_object_ply_with_poses"),
    "save_per_object_mesh":                (".io_utils", "save_per_object_mesh"),
    "save_perframe_per_object_ply":        (".io_utils", "save_perframe_per_object_ply"),
    "save_perframe_poses_json":            (".io_utils", "save_perframe_poses_json"),
    "setup_paths":                         (".io_utils", "setup_paths"),
    # Refinement
    "apply_pose_to_gaussian":              (".refinement", "apply_pose_to_gaussian"),
    "refine_pose_for_frame":               (".refinement", "refine_pose_for_frame"),
    "refine_poses_for_sequence":           (".refinement", "refine_poses_for_sequence"),
    # Rendering
    "create_comparison_grid":              (".rendering", "create_comparison_grid"),
    "render_gaussian_from_view":           (".rendering", "render_gaussian_from_view"),
    "render_gaussian_params":              (".rendering", "render_gaussian_params"),
    "render_gaussians_scene":              (".rendering", "render_gaussians_scene"),
    "render_gaussians_to_image":           (".rendering", "render_gaussians_to_image"),
    "render_multiview_comparison":         (".rendering", "render_multiview_comparison"),
    "render_perframe_decoded":             (".rendering", "render_perframe_decoded"),
    # Mesh rendering (PyTorch3D)
    "PyTorch3DMeshRenderer":               (".mesh_rendering", "PyTorch3DMeshRenderer"),
    # SLAT decoding
    "redecode_slat":                       (".slat_decode", "redecode_slat"),
    # Evaluation
    "capture_all_renders":                 (".evaluation", "capture_all_renders"),
    "capture_pose_snapshot":               (".evaluation", "capture_pose_snapshot"),
    "evaluate_with_canonical_objects":     (".evaluation", "evaluate_with_canonical_objects"),
    "print_evaluation_summary":            (".evaluation", "print_evaluation_summary"),
    "process_frame_with_canonical_object": (".evaluation", "process_frame_with_canonical_object"),
    "evaluate_block":                      (".evaluation", "evaluate_block"),
    "capture_before":                      (".evaluation", "capture_before"),
    "render_keyframes":                    (".evaluation", "render_keyframes"),
    "save_keyframes_video":                (".evaluation", "save_keyframes_video"),
    # Sequence data loading
    "FrameData":                           (".sequence", "FrameData"),
    "Sequence":                            (".sequence", "Sequence"),
    # Interpolation
    "compute_object_tracks":               (".interpolation", "compute_object_tracks"),
    "interpolate_poses":                   (".interpolation", "interpolate_poses"),
    "render_interpolated_sequence":        (".interpolation", "render_interpolated_sequence"),
    "render_perframe_sequence":            (".interpolation", "render_perframe_sequence"),
    # Visualization
    "plot_refinement_history":             (".visualization", "plot_refinement_history"),
    "save_and_plot_loss_history":          (".visualization", "save_and_plot_loss_history"),
    "save_render_comparison":              (".visualization", "save_render_comparison"),
    "visualize_object_tracks_2d":          (".visualization", "visualize_object_tracks_2d"),
    "visualize_object_tracks_3d":          (".visualization", "visualize_object_tracks_3d"),
    "visualize_slat_voxels":               (".visualization", "visualize_slat_voxels"),
    # Timing instrumentation
    "PipelineTimer":                       (".timing", "PipelineTimer"),
    "plot_pipeline_timing":                (".timing", "plot_pipeline_timing"),
    # Quaternion / rotation ops (pure-torch, no pytorch3d)
    "quaternion_to_matrix":                (".quaternion_ops", "quaternion_to_matrix"),
    "matrix_to_quaternion":                (".quaternion_ops", "matrix_to_quaternion"),
    "quaternion_multiply":                 (".quaternion_ops", "quaternion_multiply"),
    "quaternion_invert":                   (".quaternion_ops", "quaternion_invert"),
    "matrix_to_euler_angles":              (".quaternion_ops", "matrix_to_euler_angles"),
    "p3d_to_r3_positions":                 (".quaternion_ops", "p3d_to_r3_positions"),
    "r3_to_p3d_positions":                 (".quaternion_ops", "r3_to_p3d_positions"),
    "p3d_to_r3_quaternions":               (".quaternion_ops", "p3d_to_r3_quaternions"),
    # LoRA
    "LoRALayer":                           (".lora", "LoRALayer"),
    "apply_lora_to_decoder":               (".lora", "apply_lora_to_decoder"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = _importlib.import_module(module_path, __package__)
        value = getattr(module, attr_name)
        # Cache in module namespace so __getattr__ is only called once per name
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Config (eagerly loaded)
    "DatasetConfig",
    "ProcessingConfig",
    "OutputConfig",
    "LossConfig",
    "GlobalPoseRefineConfig",
    "PipelineConfig",
    "GeneralConfig",
    "print_loss_config",
    "block_config",
    "get_block_output_flag",
    "output_dir_redirect",
    "block_output_context",
    "block_output_subdir",
    # Pipeline state (eagerly loaded)
    "PipelineState",
    "get_pipeline_cache_dir",
    "save_pipeline_cache",
    "load_pipeline_cache",
    "find_latest_pipeline_cache",
    # Model cache
    "ModelCache",
    # Everything below is lazy-loaded
    *_LAZY_IMPORTS.keys(),
]
