"""
Data debug visualization for the PREPROCESSING stage: per-frame depth / normals /
pointmap panels and multi-frame summary grids (``emit_sequence_diagnostics``).
"""
import os
import sys

os.environ['LIDRA_SKIP_INIT'] = '1'

# APPEND, never insert(0): a ``sam3d_objects`` the host process already placed at
# the head of sys.path must keep winning, and this module can be imported after
# that. Prepending would jump ours back in front for every later
# ``import sam3d_objects``.
from genia.core.paths import SAM3D_OBJECTS_ROOT  # noqa: E402

_submodules = str(SAM3D_OBJECTS_ROOT)
if _submodules not in sys.path:
    sys.path.append(_submodules)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from genia.core.utils.sequence import FrameData
from genia.core.utils.visualization import save_figure


# Fixed color palette for mask overlays
_MASK_COLORS = [
    (1.0, 0.2, 0.2),  # red
    (0.2, 0.6, 1.0),  # blue
    (0.2, 0.9, 0.3),  # green
    (1.0, 0.8, 0.1),  # yellow
    (0.8, 0.3, 0.9),  # purple
    (1.0, 0.5, 0.1),  # orange
    (0.1, 0.9, 0.9),  # cyan
    (0.9, 0.4, 0.6),  # pink
]


def _colorize_depth(depth: np.ndarray, valid_mask=None) -> np.ndarray:
    """Colorize depth map with viridis, marking invalid pixels as red."""
    mask = np.isfinite(depth)
    if valid_mask is not None:
        mask = mask & valid_mask
    vmin = np.nanmin(depth[mask]) if mask.any() else 0.0
    vmax = np.nanmax(depth[mask]) if mask.any() else 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.viridis
    colored = cmap(norm(np.where(mask, depth, vmin)))[:, :, :3]  # (H,W,3)
    # Mark invalid as red
    colored[~mask] = [1.0, 0.0, 0.0]
    return colored


def _depth_edges_on_rgb(image: np.ndarray, depth: np.ndarray, valid_mask=None) -> np.ndarray:
    """Overlay depth gradient edges on RGB for alignment checking."""
    import cv2
    mask = np.isfinite(depth)
    if valid_mask is not None:
        mask = mask & valid_mask
    d = np.where(mask, depth, 0.0).astype(np.float32)
    # Normalize depth to 0-255 for Sobel
    dmin, dmax = d[mask].min() if mask.any() else 0.0, d[mask].max() if mask.any() else 1.0
    if dmax - dmin > 1e-6:
        d_norm = ((d - dmin) / (dmax - dmin) * 255).astype(np.uint8)
    else:
        d_norm = np.zeros_like(d, dtype=np.uint8)
    sx = cv2.Sobel(d_norm, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(d_norm, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(sx ** 2 + sy ** 2)
    edges = (edges / (edges.max() + 1e-8) * 255).astype(np.uint8)
    # Threshold to binary edges
    edge_mask = edges > 30
    # Overlay green edges on RGB
    overlay = image.copy().astype(np.float32) / 255.0
    overlay[edge_mask] = [0.0, 1.0, 0.0]
    return overlay


def _colorize_normals(normals: np.ndarray) -> np.ndarray:
    """Map normal vectors to RGB: (nx,ny,nz) -> ((nx+1)/2, (ny+1)/2, (nz+1)/2)."""
    return np.clip((normals + 1.0) / 2.0, 0.0, 1.0)


def _colorize_pointmap(pointmap: np.ndarray) -> np.ndarray:
    """Per-channel normalize pointmap XYZ to [0,1] for RGB visualization."""
    mask = np.isfinite(pointmap).all(axis=-1)
    result = np.zeros_like(pointmap)
    for c in range(3):
        ch = pointmap[:, :, c]
        vals = ch[mask]
        if vals.size == 0:
            continue
        vmin, vmax = vals.min(), vals.max()
        if vmax - vmin > 1e-8:
            result[:, :, c] = np.where(mask, (ch - vmin) / (vmax - vmin), 0.0)
    return result


def plot_frame_panel(
    frame: FrameData,
    frame_idx: int,
    output_dir: str,
) -> None:
    """Save a multi-panel debug figure for one frame."""
    obj_ids = sorted(frame.masks.keys())
    has_normals = frame.normals_map is not None
    has_valid = frame.valid_mask is not None

    # Layout: 2 rows x 3 cols (always), panels may be blank
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f"Frame {frame_idx}  —  {frame.image.shape[1]}x{frame.image.shape[0]}  "
                 f"K: fx={frame.K_matrix[0,0]:.1f} fy={frame.K_matrix[1,1]:.1f} "
                 f"cx={frame.K_matrix[0,2]:.1f} cy={frame.K_matrix[1,2]:.1f}",
                 fontsize=13)

    # (0,0) RGB + mask overlay
    ax = axes[0, 0]
    overlay = frame.image.astype(np.float32) / 255.0
    for i, oid in enumerate(obj_ids):
        color = np.array(_MASK_COLORS[i % len(_MASK_COLORS)])
        m = frame.masks[oid]
        overlay[m] = overlay[m] * 0.5 + color * 0.5
    ax.imshow(overlay)
    legend_labels = [f"obj {oid}" for oid in obj_ids]
    for i, label in enumerate(legend_labels):
        ax.plot([], [], 's', color=_MASK_COLORS[i % len(_MASK_COLORS)], label=label)
    if legend_labels:
        ax.legend(fontsize=8, loc='upper right')
    ax.set_title("RGB + masks")
    ax.axis('off')

    # (0,1) Depth
    ax = axes[0, 1]
    depth_vis = _colorize_depth(frame.depth_map_z, frame.valid_mask)
    ax.imshow(depth_vis)
    d_valid = frame.depth_map_z[np.isfinite(frame.depth_map_z)]
    if frame.valid_mask is not None:
        d_valid = frame.depth_map_z[frame.valid_mask & np.isfinite(frame.depth_map_z)]
    if d_valid.size > 0:
        ax.set_title(f"Depth z  [{d_valid.min():.2f}, {d_valid.max():.2f}]")
    else:
        ax.set_title("Depth z (no valid)")
    ax.axis('off')

    # (0,2) Depth edges on RGB (alignment check)
    ax = axes[0, 2]
    alignment = _depth_edges_on_rgb(frame.image, frame.depth_map_z, frame.valid_mask)
    ax.imshow(alignment)
    ax.set_title("Depth edges on RGB (alignment)")
    ax.axis('off')

    # (1,0) Pointmap XYZ as RGB
    ax = axes[1, 0]
    pm_vis = _colorize_pointmap(frame.pointmap)
    ax.imshow(pm_vis)
    ax.set_title("Pointmap XYZ→RGB")
    ax.axis('off')

    # (1,1) Valid mask or normals
    ax = axes[1, 1]
    if has_valid:
        ax.imshow(frame.valid_mask.astype(np.float32), cmap='gray', vmin=0, vmax=1)
        frac = frame.valid_mask.mean() * 100
        ax.set_title(f"Valid mask ({frac:.1f}% valid)")
    elif has_normals:
        ax.imshow(_colorize_normals(frame.normals_map))
        ax.set_title("Normals")
    else:
        ax.text(0.5, 0.5, "No valid_mask\nor normals", ha='center', va='center',
                transform=ax.transAxes, fontsize=14, color='gray')
        ax.set_title("—")
    ax.axis('off')

    # (1,2) Normals (if valid_mask took slot 1,1) or c2w info
    ax = axes[1, 2]
    if has_valid and has_normals:
        ax.imshow(_colorize_normals(frame.normals_map))
        ax.set_title("Normals")
    else:
        # Show c2w matrix
        c2w = frame.c2w
        is_identity = np.allclose(c2w, np.eye(4))
        txt = "c2w = identity" if is_identity else f"c2w =\n{np.array2string(c2w, precision=3, suppress_small=True)}"
        ax.text(0.5, 0.5, txt, ha='center', va='center',
                transform=ax.transAxes, fontsize=10, family='monospace')
        ax.set_title("Camera pose")
    ax.axis('off')

    plt.tight_layout()
    path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
    save_figure(fig, path)


def plot_summary_grid(
    sequence,
    frame_indices: list,
    output_dir: str,
) -> None:
    """Save RGB / depth / mask summary strips across all frames."""
    n = len(frame_indices)
    if n == 0:
        return

    # Determine max 20 frames per strip to keep images reasonable
    stride = max(1, n // 20)
    subset = frame_indices[::stride]
    m = len(subset)

    fig, axes = plt.subplots(3, m, figsize=(2.5 * m, 7.5))
    if m == 1:
        axes = axes[:, np.newaxis]
    fig.suptitle(f"Summary — {n} frames (showing every {stride})", fontsize=13)

    for col, fi in enumerate(subset):
        frame = sequence[fi]

        # Row 0: RGB
        axes[0, col].imshow(frame.image)
        axes[0, col].set_title(f"f{fi}", fontsize=8)
        axes[0, col].axis('off')

        # Row 1: Depth
        axes[1, col].imshow(_colorize_depth(frame.depth_map_z, frame.valid_mask))
        axes[1, col].axis('off')

        # Row 2: Mask overlay
        overlay = frame.image.astype(np.float32) / 255.0
        for i, oid in enumerate(sorted(frame.masks.keys())):
            color = np.array(_MASK_COLORS[i % len(_MASK_COLORS)])
            mask = frame.masks[oid]
            overlay[mask] = overlay[mask] * 0.4 + color * 0.6
        axes[2, col].imshow(overlay)
        axes[2, col].axis('off')

    # Row labels
    axes[0, 0].set_ylabel("RGB", fontsize=10)
    axes[1, 0].set_ylabel("Depth", fontsize=10)
    axes[2, 0].set_ylabel("Masks", fontsize=10)

    plt.tight_layout()
    path = os.path.join(output_dir, "summary_grid.png")
    save_figure(fig, path, dpi=120)
    print(f"  Summary grid → {path}")


def emit_sequence_diagnostics(sequence, output_dir: str) -> None:
    """Write per-frame depth/normals/pointmap panels + a summary grid for a
    processed ``Sequence`` into ``output_dir`` (per-view subdir when MV).

    Called by the PREPROCESSING stage (``core/preprocessing.py``).
    """
    os.makedirs(output_dir, exist_ok=True)
    frame_keys = list(sequence.frame_keys)
    n_views = len({fk.view for fk in frame_keys})
    print(f"\nGenerating per-frame panels...")
    for fk in frame_keys:
        # Per-view subdir when N_views > 1; flat layout for mono.
        panel_dir = (os.path.join(output_dir, f"view{fk.view:02d}")
                     if n_views > 1 else output_dir)
        os.makedirs(panel_dir, exist_ok=True)
        plot_frame_panel(sequence[fk], fk.frame, panel_dir)
    print(f"\nGenerating summary grid...")
    plot_summary_grid(sequence, frame_keys, output_dir)
