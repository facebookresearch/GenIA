"""Two-axis frame-view indexing primitives.

`FrameKey(frame, view)` is the per-frame state key. The data shape
(mono/mv x static/dynamic) is a property of the FrameKey set carried by
`Sequence`, not a manually declared config flag.

Helpers:
- `as_frame_key(x)`: coerce int -> FrameKey(x, 0); pass FrameKey through.
- `frame_key_sort_key(fk)`: sort by (view, frame) so per-view contiguity
  is preserved. Use as `sorted(frame_keys, key=frame_key_sort_key)`.
- `group_by_view(keys)`: bucket FrameKeys by view; each bucket sorted by frame.
"""

from typing import Dict, Iterable, List, NamedTuple, Tuple


class FrameKey(NamedTuple):
    frame: int
    view: int


def as_frame_key(x) -> FrameKey:
    """Coerce int -> FrameKey(x, 0). Pass FrameKey through unchanged.

    Accepts a 2-tuple as `(frame, view)`. Anything else castable to int via
    `int(x)` is treated as a frame index in view 0.
    """
    if isinstance(x, FrameKey):
        return x
    if isinstance(x, tuple) and len(x) == 2:
        return FrameKey(int(x[0]), int(x[1]))
    return FrameKey(int(x), 0)


def frame_key_sort_key(fk: FrameKey) -> Tuple[int, int]:
    """Sort by (view, frame). Per-view contiguity in iteration order."""
    return (fk.view, fk.frame)


def frame_key_stem(fk) -> str:
    """Per-frame output filename stem: ``'{frame:03d}'`` (view 0) or
    ``'{frame:03d}_v{view:02d}'``. Single owner of the PLY/render naming
    convention used by the per-frame output writers."""
    fk = as_frame_key(fk)
    return (f"{fk.frame:03d}" if fk.view == 0
            else f"{fk.frame:03d}_v{fk.view:02d}")


def group_by_view(keys: Iterable[FrameKey]) -> Dict[int, List[FrameKey]]:
    """Bucket FrameKeys by view. Each bucket is sorted by frame ascending."""
    out: Dict[int, List[FrameKey]] = {}
    for fk in keys:
        out.setdefault(fk.view, []).append(fk)
    for v in out:
        out[v].sort(key=lambda fk: fk.frame)
    return out


def group_by_frame(keys: Iterable[FrameKey]) -> Dict[int, List[FrameKey]]:
    """Bucket FrameKeys by frame (timestamp). Each bucket is sorted by view ascending.

    The twin of :func:`group_by_view`, and the axis a shared-world pose collapses along: one
    object at one TIMESTAMP has a single world placement seen by its views, while the same
    object at two timestamps may legitimately have moved.  Reducing along the wrong axis is
    how a rebase erases motion.
    """
    out: Dict[int, List[FrameKey]] = {}
    for fk in keys:
        out.setdefault(fk.frame, []).append(fk)
    for f in out:
        out[f].sort(key=lambda fk: fk.view)
    return out


