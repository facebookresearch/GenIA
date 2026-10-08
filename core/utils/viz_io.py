# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Per-view visualization writer.

Writes a per-frame visualization as either a single PNG (when T_per_view == 1)
or an MP4 video (when T_per_view >= 2), with a `view{vi:02d}/` subdir injected
only when `n_views_total > 1`. Mono runs (N_views == 1) preserve byte-identical
flat layout; MV-static and MV-dynamic runs use per-view subdirs.

Reuses `interpolation._save_frames_as_video` for the MP4 path so encoder
behavior stays consistent across the codebase.
"""

import os
from typing import Dict, List, Optional

from PIL import Image

from genia.core.utils.interpolation import _save_frames_as_video


def write_per_view(
    frames_by_view: Dict[int, List["Image.Image"]],
    base_dir: str,
    stem: str,
    *,
    duration: int = 100,
    frame_repeats: Optional[List[int]] = None,
    n_views_total: Optional[int] = None,
) -> List[str]:
    """Write per-view PNG (T==1) or MP4 (T>=2) artifacts.

    Parameters
    ----------
    frames_by_view : dict[int, list[PIL.Image.Image]]
        For each view index, the ordered list of frames to encode.
    base_dir : str
        Directory under which to write. Subdir `view{vi:02d}/` is injected only
        if `n_views_total > 1`.
    stem : str
        Filename stem without extension (e.g. "keyframes", "{scene}_perframe").
    duration, frame_repeats : passed through to `_save_frames_as_video` for
        MP4 encoding.
    n_views_total : optional
        The total view count for the run (defaults to len(frames_by_view)).
        Pass explicitly when this helper is called per-view but the caller
        knows the global view count (e.g., when only one view's frames are
        ready at call time).
    """
    n_views = n_views_total if n_views_total is not None else len(frames_by_view)
    written: List[str] = []
    for view_idx, frames in frames_by_view.items():
        if not frames:
            continue
        if n_views > 1:
            out_dir = os.path.join(base_dir, f"view{view_idx:02d}")
        else:
            out_dir = base_dir
        os.makedirs(out_dir, exist_ok=True)
        if len(frames) == 1:
            path = os.path.join(out_dir, f"{stem}.png")
            frames[0].save(path)
        else:
            path = os.path.join(out_dir, f"{stem}.mp4")
            _save_frames_as_video(frames, path, duration=duration, frame_repeats=frame_repeats)
        written.append(path)
    return written
