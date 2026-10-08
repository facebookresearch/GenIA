import torch
import torch.nn as nn
import lpips


def mask_bbox(mask, margin: float = 0.1):
    """The mask's bounding box + ``margin``, as ``(y0, x0, y1, x1)`` ints.

    ``margin`` pads by that fraction of the box's extent on each side, clamped to the
    image.  Returns None for an empty mask.  Split out of :func:`crop_to_mask_bbox` so a
    caller that must map coordinates INTO the crop (the rotation search draws its
    correspondences on a full-frame tile) can ask for the box instead of re-deriving it
    and hoping the two conventions stay in step.
    """
    nz = torch.nonzero(mask > 0.5, as_tuple=False)
    if nz.numel() == 0:
        return None
    y0, x0 = nz.min(dim=0).values.tolist()
    y1, x1 = (nz.max(dim=0).values + 1).tolist()
    H, W = mask.shape[-2:]
    py, px = round((y1 - y0) * margin), round((x1 - x0) * margin)
    return (max(0, y0 - py), max(0, x0 - px), min(H, y1 + py), min(W, x1 + px))


def crop_to_mask_bbox(rgb_bchw, gt_bchw, mask, margin: float = 0.1):
    """Crop both ``(1, 3, H, W)`` tensors to the mask's bounding box + margin.

    ``mask`` is ``(H, W)`` (bool or float) at the SAME spatial resolution as the
    tensors.  ``margin`` pads the box by that fraction of its extent on each
    side, clamped to the image.  An empty mask returns the inputs unchanged
    (full-frame fallback).  The slice is differentiable and the bounds are
    Python ints, so the crop shape is static for a given call.
    """
    box = mask_bbox(mask, margin)
    if box is None:
        return rgb_bchw, gt_bchw
    y0, x0, y1, x1 = box
    return rgb_bchw[..., y0:y1, x0:x1], gt_bchw[..., y0:y1, x0:x1]


class PerceptualLoss(nn.Module):
    """LPIPS perceptual loss (Zhang et al., 2018).

    Thin wrapper around the ``lpips`` package so the rest of the codebase
    keeps the same interface: ``loss = model(pred, target)`` where both
    inputs are ``(B, 3, H, W)`` in ``[0, 1]``.
    """

    def __init__(self, net: str = "vgg"):
        super().__init__()
        import warnings

        # lpips.LPIPS expects images in [-1, 1].  verbose=False silences the
        # "Loading model from: ..." print; catch_warnings suppresses the
        # torchvision `pretrained`/`weights` deprecation UserWarnings emitted
        # while lpips builds the VGG backbone internally.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = lpips.LPIPS(net=net, verbose=False)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, pred_img, target_img, **kwargs):
        """Compute LPIPS between prediction and target.

        Parameters
        ----------
        pred_img, target_img : torch.Tensor
            Shape ``(B, 3, H, W)`` in ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            Scalar LPIPS distance (mean over batch).
        """
        # Map [0, 1] -> [-1, 1]
        pred_scaled = pred_img * 2.0 - 1.0
        target_scaled = target_img * 2.0 - 1.0
        return self.model(pred_scaled, target_scaled).mean()
