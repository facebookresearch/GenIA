# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Fetch the model weights GenIA needs into the Hugging Face and torch hub caches.

    python -m genia.core.download_weights                  # everything
    python -m genia.core.download_weights --no-actionmesh  # static scenes only
    python -m genia.core.download_weights --dry-run

Every model is read from the usual caches (``HF_HUB_CACHE`` / ``TORCH_HOME`` move them), so
weights another project already fetched are reused, and fetching them here lets later runs
work offline. Files already cached are left alone.

- SAM 3D Objects: its ``checkpoints/`` folder (``genia.core.paths.sam3d_weights``;
  ``GENIA_SAM3D_WEIGHTS`` points at a copy kept elsewhere instead). The repository is gated:
  accept its licence at https://huggingface.co/facebook/sam-3d-objects, then log in with
  ``hf auth login``.
- ActionMesh, which supplies the shapes of dynamic scenes, with TripoSG, DINOv2 and RMBG
  (~17 GB; ``core/actionmesh_video.py`` points ActionMesh at them).
- map-anything and MoGe (depth), SAM 3D Objects' DINOv2 image encoder and the LPIPS backbone.
"""
import argparse
import sys

from genia.core.paths import SAM3D_REPO

#: (repo, the ./pretrained_weights/ folder actionmesh.pipeline would read it from).
ACTIONMESH_WEIGHTS = [
    ("VAST-AI/TripoSG", "TripoSG"),
    ("facebook/dinov2-large", "dinov2"),
    ("briaai/RMBG-1.4", "RMBG"),
    ("facebook/ActionMesh", "ActionMesh"),
]

#: Hugging Face models loaded at run time: map-anything (depth + cameras) and MoGe (depth;
#: also SAM 3D Objects' depth model).
RUNTIME_HF = ["facebook/map-anything", "Ruicheng/moge-vitl"]
#: SAM 3D Objects' image encoder, loaded through torch hub.
DINO_HUB = ("facebookresearch/dinov2", "dinov2_vitl14_reg")
#: LPIPS backbone: VGG for the perceptual loss.
LPIPS_NETS = ["vgg"]
#: The files the loaders read: skips the TensorFlow / Flax / .bin copies some repos ship.
HF_PATTERNS = ["*.json", "*.safetensors", "*.pt"]


def fetch_hf(repo, dry_run: bool, patterns=None) -> int:
    """Fill the Hugging Face cache with ``repo`` (only the files matching ``patterns``)."""
    from huggingface_hub import constants, snapshot_download
    from huggingface_hub.errors import GatedRepoError

    if dry_run:
        print(f"would fetch {repo} -> {constants.HF_HUB_CACHE}")
        return 0
    try:
        snapshot_download(repo_id=repo, allow_patterns=patterns)
    except GatedRepoError:
        print(f"{repo} is gated: accept the licence at https://huggingface.co/{repo}, "
              "run `hf auth login`, and re-run this script.")
        return 1
    print(f"ok: {repo}")
    return 0


def fetch_torch_hub(dry_run: bool) -> None:
    """Fill the torch hub cache: SAM 3D Objects' DINOv2 and the LPIPS backbone."""
    import torch

    if dry_run:
        print(f"would fetch {DINO_HUB[1]} and LPIPS {'/'.join(LPIPS_NETS)} "
              f"-> {torch.hub.get_dir()}")
        return
    import warnings

    import lpips

    torch.hub.load(*DINO_HUB, source="github", verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # torchvision's `pretrained` deprecation
        for net in LPIPS_NETS:
            lpips.LPIPS(net=net, verbose=False)
    print(f"ok: {DINO_HUB[1]}, LPIPS {'/'.join(LPIPS_NETS)}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-actionmesh", action="store_true",
                    help="skip the ActionMesh weights (only needed for dynamic scenes)")
    ap.add_argument("--dry-run", action="store_true", help="print what would be downloaded")
    args = ap.parse_args()

    rc = fetch_hf(SAM3D_REPO, args.dry_run, ["checkpoints/*"])
    if not args.no_actionmesh:
        for repo, _ in ACTIONMESH_WEIGHTS:
            rc = fetch_hf(repo, args.dry_run) or rc
    for repo in RUNTIME_HF:
        rc = fetch_hf(repo, args.dry_run, HF_PATTERNS) or rc
    fetch_torch_hub(args.dry_run)
    return rc


if __name__ == "__main__":
    sys.exit(main())
