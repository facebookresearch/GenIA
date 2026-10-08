# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Our pose-token refit.

The post-rotation (t, s) refit, which writes refined poses back into the raw Stage-1
tokens.
"""

import numpy as np

from genia.core.utils.pose_refit import (
    _refit_one,
    canonical_points_for_frame,
)


def frame_has_refit_tokens(di) -> bool:
    """Does this frame carry the RAW pose tokens the refit needs?

    The refit inverts through the SSI chain, so it reads the raw layout tokens rather than
    the decoded pose; a frame without them is skipped, silently and by design (a per-frame
    SAM3D run legitimately has none).

    Named and public because more than one caller has to agree on it.  A caller that wants
    to know IN ADVANCE whether a frame will be refit -- e.g. anything that arranges for
    some frames to get their pose derived rather than predicted -- must ask the same
    question the refit asks, or the two drift and the refit starts skipping frames the
    caller believed it would visit.  That skip is silent: the block reports the frames it
    DID refit and nothing reports the ones it did not.
    """
    raw = di.get("raw_ss_modalities") or {}
    return "translation" in raw and "scale" in raw


def _frame_refit_inputs(state, sequence, obj_idx: int, fk, di, device, normalizer,
                        *, obs=None):
    """Everything one frame's refit needs.

    ``obs`` is the :class:`~genia.core.utils.depth_grounding.DepthObservation` spec (None =
    its defaults).  The pointmap is read in PyTorch3D convention.

    Returns a dict, or None when this frame cannot be refit.
    """
    import torch

    from genia.core.utils.depth import transform_to_pytorch3d_convention
    from genia.core.utils.depth_grounding import DepthObservation


    if not frame_has_refit_tokens(di):
        return None
    frame_int = fk.frame if hasattr(fk, "frame") else int(fk)
    pts = canonical_points_for_frame(state, obj_idx, frame_int)
    if pts is None:
        return None

    frame_data = sequence[fk]
    mask = frame_data.masks.get(obj_idx)
    if mask is None:
        return None
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return None
    pm = np.asarray(frame_data.pointmap, dtype=np.float64)
    K = np.asarray(frame_data.K_matrix, dtype=np.float64)
    obs = obs if obs is not None else DepthObservation()
    got = obs.read(mask, transform_to_pytorch3d_convention(pm), K)
    if got is None:
        return None
    center, extent = got

    # Encode with the SSI this frame's tokens were DECODED against, not a fresh one:
    # `decode_perframe_poses_from_raw` reads di["pointmap_scale"/"shift"], so
    # re-normalising here would break the round trip silently (the pose would come back
    # in a different metric).  Fall back to a fresh normalise only when the frame
    # carries none.
    ps, psh = di.get("pointmap_scale"), di.get("pointmap_shift")
    if ps is None or psh is None:
        pm_3hw = (torch.as_tensor(pm, dtype=torch.float32, device=device)
                  .permute(2, 0, 1).contiguous())
        mask_1hw = torch.as_tensor(mask, dtype=torch.float32,
                                   device=device).unsqueeze(0)
        ssi = normalizer.normalize(pm_3hw, mask_1hw)
        ps, psh = ssi.scale, ssi.shift
        di["pointmap_scale"], di["pointmap_shift"] = ps, psh
    return {
        "pts": pts,
        "center": np.asarray(center, dtype=np.float64),
        "extent": float(extent),
        "K": K,
        "pointmap_scale": ps,
        "pointmap_shift": psh,
        "downsample_factor": di.get("downsample_factor", 1.0),
    }


def apply_post_rotation_refit(
    state, sequence, obj_idx: int, device, pipeline_obj, mode: str,
    *, iters: int = 5, mv_consensus: bool = True, obs=None,
) -> int:
    """Refit each frame's (translation, scale) using its decoded rotation, in place.

    Runs AFTER the Stage-1 ODE, which is the whole point: the seed had to guess without
    a rotation, and now one exists.  Writes the RAW tokens (via the same
    ``camera_pose_to_raw_tokens`` SSI inversion the seed uses) and re-decodes, because
    downstream consumers read the raw layout tokens rather than the decoded fields.

    ``mode`` selects which half is applied: ``translation``, ``scale``, ``both``.  They
    are separate because they do not tolerate rotation error equally.  Returns the number
    of frames refit.

    ``mv_consensus`` fuses, per timestamp, the refit translation of that timestamp's views
    into one world placement and reduces their scales to one size (median).  Each fit
    reads a single view, and on a shared-world MV run the per-view disagreement is then
    re-derived away from one reference frame -- so without this the refit is decided by
    one view and the rest are discarded.  Inert on mono data (one view per timestamp).
    """
    import torch

    from genia.core.utils.pose_token_gt import camera_pose_to_raw_tokens
    from genia.core.utils.quaternion_ops import quaternion_to_matrix

    from genia.core.utils.frame_key import as_frame_key

    if mode not in ("translation", "scale", "both"):
        raise ValueError(
            f"post_rotation_refit must be one of 'none', 'translation', 'scale', "
            f"'both', got {mode!r}")
    tokens_list = state.tokens_by_object.get(obj_idx)
    if not tokens_list:
        return 0
    normalizer = pipeline_obj.ss_preprocessor.pointmap_normalizer

    # Gather every frame's fit FIRST: the view consensus needs every view of a timestamp
    # before anything is written.
    fits = []
    for fk, di in tokens_list:
        if di.get("rotation") is None:
            continue
        inp = _frame_refit_inputs(state, sequence, obj_idx, fk, di, device, normalizer,
                                  obs=obs)
        if inp is None:
            continue          # no canonical, no mask, or a degenerate observation
        R = quaternion_to_matrix(di["rotation"].reshape(1, 4))[0].detach().cpu().numpy()
        got = _refit_one(
            inp, R.astype(np.float64),
            di["translation"].reshape(3).detach().cpu().numpy().astype(np.float64),
            di["scale"].reshape(-1).detach().cpu().numpy().astype(np.float64),
            mode, iters)
        if got is None:
            continue
        t_out, s_out, s_raw = got
        fits.append({"di": di, "inp": inp, "R": R, "t": t_out, "s": s_out,
                     "s_raw": s_raw, "fk": as_frame_key(fk)})

    from genia.core.utils.depth_grounding import (
        fuse_p3d_positions_by_timestamp,
        group_records_by_timestamp,
    )

    if mv_consensus and fits and mode in ("scale", "both"):
        # View reduce of SCALE, per timestamp, from each frame's own fitted size.
        # Translation is left as fitted.
        for group in group_records_by_timestamp(fits).values():
            s_ts = float(np.median(np.asarray([f["s_raw"] for f in group], dtype=np.float64)))
            for f in group:
                f["s"] = np.full(3, s_ts, dtype=np.float64)

    if mv_consensus and fits and mode in ("translation", "both"):
        # The fused translation names one world point for a timestamp.
        fused = fuse_p3d_positions_by_timestamp(fits, sequence, "t")
        if fused:
            print(f"      mv consensus: fused the refit translation of "
                  f"{sum(len(g) for g in fused)} view(s) across {len(fused)} timestamp(s)")

    n = 0
    for f in fits:
        inp, R = f["inp"], f["R"]
        ps, psh = inp["pointmap_scale"], inp["pointmap_shift"]
        tok = camera_pose_to_raw_tokens(
            torch.as_tensor(R, dtype=torch.float32, device=device),
            torch.as_tensor(f["t"], dtype=torch.float32, device=device),
            torch.as_tensor(f["s"], dtype=torch.float32, device=device),
            ps.to(device), psh.to(device),
            downsample_factor=inp["downsample_factor"],
        )
        # Rotation is NOT rewritten: the ODE produced it and the refit consumes it.
        raw = f["di"].setdefault("raw_ss_modalities", {})
        raw["translation"] = tok["translation"].detach()
        raw["scale"] = tok["scale"].detach()
        n += 1

    if n:
        state.decode_perframe_poses_from_raw(obj_idx, source="tokens_by_object")
    return n


