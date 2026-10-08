"""GT image loaders, following the EscherNet-protocol GSO-30 evaluation.

Sources (EscherNet repository, Kong et al., ECCV 2024,
https://github.com/kxhit/EscherNet):
    load_gso_gt_image          eval_2D_NVS.py composite rule (alpha==0 -> white)
"""

import numpy as np
from PIL import Image


def load_gso_gt_image(
    gt_path: str, size: int = 256, linear_to_srgb: bool = False
) -> np.ndarray:
    """Load a GSO GT RGBA image, composite on white, resize to (size, size).

    EscherNet protocol (eval_2D_NVS.py:96): replace only fully-transparent
    pixels (alpha == 0) with white; keep the raw RGB of anti-aliased silhouette
    edges. This differs from a general alpha-blend and is what EscherNet's
    published tables measure -- we match it exactly for apples-to-apples.

    When linear_to_srgb is True, treats the image as Blender output with
    premultiplied alpha in linear color space: un-premultiplies and applies
    sRGB gamma first.
    """
    raw = np.array(Image.open(gt_path).convert("RGBA")).astype(np.float32)
    alpha = raw[:, :, 3:4] / 255.0
    rgb = raw[:, :, :3]

    if linear_to_srgb:
        safe_alpha = np.where(alpha > 0, alpha, 1.0)
        rgb = np.clip(rgb / safe_alpha, 0, 255)
        rgb = np.clip(rgb / 255.0, 0, 1) ** (1.0 / 2.2) * 255.0

    rgba = np.concatenate([rgb, alpha * 255.0], axis=-1)
    fully_transparent = (rgba[:, :, 3] == 0.0)
    rgba[fully_transparent] = [255.0, 255.0, 255.0, 255.0]
    composited = rgba[:, :, :3].astype(np.uint8)
    return np.array(Image.fromarray(composited).resize((size, size)))


def _load_co3d_gt_image(
    rgb_path: str, mask_path: str, size: int,
) -> np.ndarray:
    """Composite a CO3D GT RGB + binary mask onto white at ``(size, size)``.

    LaRa-format CO3D scenes store the
    GT as a 3-channel ``{NNN}.png`` (full background) + a binary 0/1
    ``{NNN}_mask.png`` (foreground).  The export's predicted render is
    foreground-on-white at the train-view resolution, so for an
    apples-to-apples comparison we composite the GT through the binary
    mask onto white before scoring.  No alpha-blending of edges (the mask
    is hard binary).
    """
    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L"))
    fg = (mask > 0)[..., None].astype(np.float32)
    composited = (rgb.astype(np.float32) * fg + 255.0 * (1.0 - fg)).astype(np.uint8)
    if composited.shape[0] != size or composited.shape[1] != size:
        composited = np.array(Image.fromarray(composited).resize((size, size)))
    return composited
