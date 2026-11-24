# Copyright (c) Meta Platforms, Inc. and affiliates.

"""3D viewer for a run's COLMAP reconstruction, 3DGS splats and tracks.

A copy of the GenerativeReconstruction framework's ``gradio/_gradio_viewer.py`` (the demo
app there looks and works the same way); only where the bundle is found differs.

Renders the 3DView (``submodules/3DView``) Three.js bundle in an ``<iframe>``: a colored
point cloud with clickable camera frustums, 3D Gaussian Splatting PLYs (including
the compressed PlayCanvas/SuperSplat flavour the pipeline writes), and ``.npz``
point tracks — all in one scene, so several sources overlay in the same frame.

Five stages, all discovered from the run dir rather than announced by the pipeline:

* **preprocessing** — ``preprocessing/colmap/`` exists as soon as PREPROCESSING has
  run, long before the pipeline finishes: the input cameras + the depth-unprojected
  point cloud. Written whenever ``output.save_colmap`` is set (default true) and
  NOT suppressed by ``output.suppress_intermediate_renders``.
* **voxel grid** — ``NN_{block}/voxel_grid/`` (see :func:`_voxel_grid_source`), the
  objects' occupied Stage-1 voxels as cubes, posed, written after each block that moves
  the objects; the newest replaces the previous one, so it follows the poses as they are
  refined — well before the pipeline's own "final" stage below. An early, deliberately
  provisional-looking preview, not a scored artifact. Superseded, once the splats below
  land, the same way the depth cloud is (``hideVoxelGrid``, see below) — but it is a
  *Scene-panel item*, not a 3DView checkbox, so it takes its own hide path (see
  :func:`_voxel_grid_source` and ``viewer/index.html``).
* **pose cubes** — ``pose_cubes.glb``, each object's pose as its frame's cube with that
  frame's axes, one ``obj_{i}`` node per object that the page moves by the live poses as
  it does the voxel grid's: the pose itself (rotation, translation, scale), as it changes.
* **object splats** — ``NN_{block}/splats/`` (see :func:`_preview_splat_sources`), each
  object's Gaussians after each block that changes how it looks, one item per object
  placed by the live poses: the reconstruction as it stands, before FINAL. They count as
  the splats below until those land, and are then replaced by them.
* **final** — ``final/gaussians/world[.compressed].ply`` (all foreground objects,
  posed) + ``background[.compressed].ply``, added to the *same* scene as 3DGS
  splats. Both are R3 world space, in the frame the preprocessing cameras are
  already in, so they land where the cameras look. ``final/colmap/`` is only
  reached for when preprocessing wrote none — its cameras are the same input
  cameras, so loading it too would double every frustum.

The depth cloud is the *input* the splats are reconstructed from, so once they land
it is only clutter: the manifest's ``hidePoints`` then asks the page to switch the
cloud off. A display toggle, not a content decision — the points stay loaded, so
the scene's bounds (hence fit-to-view) and the panel's own Points control are
unaffected, and the user can switch them back on. ``hideVoxelGrid`` answers the
same question for the voxel-grid preview, on the same trigger (the splats
landing) — also a display toggle, not a removal: the item stays in the Scene
panel's list, one click away from being shown again.

Each source is keyed by an id the viewer page posts exactly once
(:func:`write_manifest`), so the finished reconstruction is *added* to the live scene
rather than replacing it; only the previews (voxel grid, object splats) leave the list,
when newer ones replace them.

Dynamic runs get one COLMAP model per temporal frame (``sparse/{frame:03d}``); we
surface the first and last so the two ends of the motion can be compared, plus a
3D-track ``.npz`` when one was exported. Their SPLATS take the opposite treatment —
every frame, as one temporal item the viewer can play (see :func:`_splat_sources`).

Nothing here depends on Gradio internals beyond the file-serving URL shape, and
nothing depends on the pipeline being importable.
"""

from __future__ import annotations

import json
import os
import urllib.parse
from pathlib import Path

VIEWER_PAGE = Path(__file__).resolve().parent / "viewer" / "index.html"


def _viewer_dist() -> Path:
    """Where the prebuilt 3DView bundle is: BUILT in the ``submodules/3DView`` checkout
    by ``install.sh`` (4.5 MB of JS is not worth committing), or ``DEMO_VIEWER_DIST``."""
    if override := os.environ.get("DEMO_VIEWER_DIST"):
        return Path(override)
    return VIEWER_PAGE.parents[3] / "submodules" / "3DView" / "demo" / "dist"


VIEWER_DIST = _viewer_dist()

def post_js(payload: str) -> str:
    """A JS statement posting ``payload`` into every embedded viewer.

    For a Gradio control's ``js=`` (no Python round-trip). Posted INTO the iframe
    rather than rebuilt into its URL, so the loaded scene and the user's viewpoint
    survive; the page acts on it (see viewer/index.html). Every viewer on the page
    is addressed at once — the apps run one pipeline at a time, so the others are
    placeholders anyway, and this needs no per-panel element id. ``?.`` because a
    frame that has not loaded yet has no contentWindow.
    """
    return ("document.querySelectorAll('iframe.viewer3d').forEach("
            f"(f) => f.contentWindow?.postMessage({payload}, '*'));")


# Show/hide the viewer's own "3DView" panel (its scene-wide display controls).
# The "Scene" panel — the item list — is never hidden: it is how a visitor sees
# what got loaded and toggles individual assets.
UI_TOGGLE_JS = "(show) => { %s }" % post_js("{ type: 'hostUi', show }")

# Is that panel shown before anyone touches the checkbox?  Off: the scene is what
# a visitor came for, and the display controls mostly re-decide what the pipeline
# already decided.  Read by BOTH the host checkbox's initial `value` and the
# `?ui=` the page loads with, so the two cannot drift out of sync (see
# viewer/index.html).
UI_DEFAULT = False

# Gradio serves any file under `allowed_paths` at this route.
_FILE_ROUTE = "/gradio_api/file="

# The voxel-grid Scene-panel item's label. A module constant, not a literal
# inline in _voxel_grid_source, because viewer/index.html has to match this
# exact string too (see hideVoxelGrid in write_manifest) -- one name for both
# sides of that split, the same role UI_TOGGLE_JS/UI_DEFAULT play above.
VOXEL_GRID_LABEL = "voxel grid"
#: The pose cubes' item, which the page moves by node like the voxel grid (index.html
#: matches it the same way).
POSE_CUBES_LABEL = "pose cubes"


def bundle_root() -> str:
    """The extra root ``launch(allowed_paths=…)`` needs, so the iframe can fetch
    the viewer bundle (everything else it loads lives under the repo)."""
    return str(VIEWER_DIST)


def _url(path: Path) -> str:
    return _FILE_ROUTE + urllib.parse.quote(str(path))


def _bundle_url() -> str:
    """The viewer bundle's URL, carrying a cache key so a REBUILT one is fetched.

    Without it the URL is stable across builds, so a browser that already has
    ``webview.js`` serves its cached copy — and a viewer feature added by a rebuild
    silently does not exist for that visitor, with no error anywhere to say so. That
    is not hypothetical: the temporal-grouping hint (:func:`_temporal_source`) landed
    in a rebuild, and a page holding the previous bundle went on showing the modal it
    was meant to answer.

    Keyed on mtime: one ``stat``, and it changes exactly when the build does. A
    rebuild producing identical bytes re-fetches needlessly, which costs one 5.6 MB
    transfer on a rebuild that already took longer than that.
    """
    js = VIEWER_DIST / "webview.js"
    try:
        return f"{_url(js)}?v={int(js.stat().st_mtime)}"
    except OSError:
        # No bundle: the page says so itself (and `dev` names the path it wanted).
        # Failing here instead would take the whole panel down for a missing file.
        return _url(js)


def _colmap_source(colmap_dir: Path, sparse: Path, label: str) -> dict:
    """A ``loadColmap`` message: the webview fetches + parses these three files.

    ``format="bin"`` — ``write_colmap_model`` emits both flavours; the binary one
    is smaller over the wire and is what the loader prefers.
    """
    images = colmap_dir / "images"
    msg = {
        "type": "loadColmap",
        # Stage-qualified ("final" / "preprocessing"), and unique in the manifest:
        # the viewer page posts each id exactly once, so two models must never
        # collide under one id.
        "id": f"colmap-{colmap_dir.parent.name}-{label}-{sparse.name}",
        "label": label,
        "format": "bin",
        "urls": {
            "cameras": _url(sparse / "cameras.bin"),
            "images": _url(sparse / "images.bin"),
            "points3d": _url(sparse / "points3D.bin"),
        },
        "source": str(sparse),
    }
    if images.is_dir():
        # Trailing slash: the loader joins each COLMAP image name onto this. A
        # plain "/" (not %2F) — `_url` leaves path separators unescaped, so the
        # whole thing stays one consistently-formed path after `file=`.
        msg["imageBaseUrl"] = _url(images) + "/"
    return msg


def _asset_source(path: Path, label: str) -> dict:
    """An ``addAsset`` message — mesh, 3DGS splat or tracks, picked by extension."""
    return {
        "type": "addAsset",
        "id": f"asset-{label}",
        "label": label,
        "asset": {"uri": _url(path), "name": path.name},
    }


def _sparse_models(colmap_dir: Path) -> list[Path]:
    """Per-frame COLMAP models, in frame order (static runs collapse to one)."""
    sparse = colmap_dir / "sparse"
    if not sparse.is_dir():
        return []
    return sorted(d for d in sparse.iterdir() if (d / "cameras.bin").is_file())


def _first_and_last(paths: list[Path]) -> list[Path]:
    """Both ends of a per-frame sequence, or the single static entry unchanged.

    Only the two ends: one frustum set — or one whole reconstruction — per frame
    over a 16-frame clip buries the geometry, and the ends are what shows how far
    the scene moved. What fills the gap is the motion drawn some other way: the 3D
    tracks as polylines (:func:`_tracks`), and the splats as a temporal item that
    plays (:func:`_splat_sources`) — which is why THOSE are not reduced to two.
    """
    return paths if len(paths) <= 1 else [paths[0], paths[-1]]


def _frame_label(path: Path) -> str:
    """``frame 007`` from a ``007`` model dir or an ``007.compressed.ply`` file."""
    return f"frame {path.name.split('.')[0]}"


def _plys(d: Path, pat: str) -> list[Path]:
    """PLYs in ``d`` matching ``pat``, compressed flavour preferred.

    Serves both shapes the pipeline writes from one rule: ``pat="*"`` on a
    dynamic run's per-frame directory, ``pat="world"`` on the flat static file.

    Both flavours are matched separately rather than with one ``*.ply`` glob:
    ``.compressed.ply`` also ends in ``.ply``, so a single glob would load a
    frame twice if a run ever wrote both. ``or`` short-circuits, so the second
    glob only runs when the first found nothing.
    """
    if not d.is_dir():
        return []
    return sorted(d.glob(f"{pat}.compressed.ply")) or sorted(d.glob(f"{pat}.ply"))


def _temporal_source(paths: list[Path], label: str) -> dict:
    """An ``addGroup`` carrying a per-frame sequence as ONE scrubbable item.

    ``grouping="temporal"`` answers the viewer's grouping modal in advance. It asks
    that question because a multi-file *pick* is ambiguous — but we walked the
    directory ourselves, so we already know these are timesteps of one capture, and
    a dialog in front of content the visitor never assembled is just an obstacle.
    Hosts that omit the field still get asked, so this changes nobody else.

    Members are ordinary ``addAsset`` messages — a group member takes the same path
    it would alone — so each keeps a unique id and the timeline orders them by label.
    """
    return {
        "type": "addGroup",
        "id": f"group-{label}",
        "label": label,
        "grouping": "temporal",
        "members": [_asset_source(p, f"{label} {_frame_label(p)}") for p in paths],
    }


def _splat_sources(gaussians: Path, complete: bool) -> list[dict]:
    """World + background splat messages, compressed flavour preferred.

    A **dynamic** run writes the world scene as a DIRECTORY of per-frame files
    (``world/{frame:03d}.ply``, 1:1 with ``colmap/sparse/{frame:03d}``) instead of
    one flat file. Every frame is loaded, as a single temporal item: the pipeline
    warps ONE canonical Gaussian per frame (means and quats move; count, scales,
    opacities and SH do not), so the frames share a packed layout and the viewer
    can scrub them by writing into one mesh's buffer instead of re-sorting per
    frame — a video rather than a slideshow. That is also why this does NOT take
    ``_first_and_last`` the way ``_cameras`` does: N frustum sets pile up in one
    frame, but N splat frames are the item.

    A static run has the flat file and is unaffected. ``background`` already spans
    every camera in one file, so it only ever takes the flat path.

    ``complete`` withholds the temporal item until the run has stopped writing, and
    it is load-bearing rather than tidy. The page posts each source ONCE, keyed by
    id, so whatever a group holds when it is first published is what it holds
    forever. This directory is filled one frame at a time, and ticks are driven by
    subprocess output — so a tick landing mid-write published a two-frame group and
    silently suppressed the sixteen-frame one behind it as a duplicate id. A group
    may only be announced when it can no longer grow.
    """
    out = []
    for stem, label in (("world", "objects"), ("background", "background")):
        if per_frame := (_plys(gaussians / stem, "*") if complete else []):
            out.append(_temporal_source(per_frame, label))
        elif flat := _plys(gaussians, stem):
            out.append(_asset_source(flat[0], label))
    return out


#: Where a block's voxel-grid preview is written: ``voxel_grid/``, and ``coarse_shape/``
#: (its marching-cubes wireframe) for GenIA up to 83d9728, a run of which still shows one.
_VOXEL_GRID_DIRS = ("voxel_grid", "coarse_shape")


def _voxel_grid_source(run_dir: Path) -> list[dict]:
    """The NEWEST posed voxel-grid preview, once complete.

    A run writes one after each block that moves the objects (``NN_{block}/voxel_grid/``,
    from pose initialisation through every pose refinement), so the newest sits at the
    current poses. Its id names the block it came from: a newer preview is a new source,
    and the viewer page swaps it in for the previous one (``viewer/index.html``), so the
    Scene panel keeps a single voxel-grid item that moves with the poses.

    Gated on each export's own ``.complete`` sentinel (written once every frame is on
    disk), not on the run's ``done`` flag: these land long before FINAL. The ``NN_``
    prefix is run order, so the last match is the newest; it is globbed rather than
    imported, as nothing here depends on the pipeline being importable.
    """
    done = sorted(p for d in _VOXEL_GRID_DIRS
                  for p in run_dir.glob(f"[0-9][0-9]_*/{d}/.complete"))
    if not done:
        return []
    root = done[-1].parent
    if (flat := root / "world.glb").is_file():
        source = _asset_source(flat, VOXEL_GRID_LABEL)
    elif per_frame := sorted((root / "world").glob("*.glb")):
        source = _temporal_source(per_frame, VOXEL_GRID_LABEL)
    else:
        return []
    source["id"] = f"voxels-{root.parent.name}"
    return [source]


def _preview_splat_sources(run_dir: Path) -> list[dict]:
    """The NEWEST block's per-object splats, once complete (``NN_{block}/splats/``).

    A run writes them after each block that changes how the objects look, one file per
    object in its own canonical frame (``obj_{i}.compressed.ply``). Each becomes an item
    labelled ``object {i}`` whose ``livePose`` names its entry in the run's live poses,
    which the page places it by, as it does the voxel grid's nodes: the splats show, and
    move with the poses, long before FINAL. A newer block's replace the older ones the
    way voxel grids do, and FINAL's own splats replace them all (``viewer/index.html``).
    """
    done = sorted(run_dir.glob("[0-9][0-9]_*/splats/.complete"))
    if not done:
        return []
    root, out = done[-1].parent, []
    for path in sorted(root.glob("obj_*.ply")):
        key = path.name.split(".")[0]   # obj_{i}, as live_poses names it
        source = _asset_source(path, f"object {key[4:]}")
        source.update(id=f"splats-{root.parent.name}-{key}", livePose=key)
        out.append(source)
    return out


def _tracks(final_dir: Path) -> list[tuple[Path, str]]:
    """3D point tracks, if a run exported them in the viewer's ``.npz`` format.

    Written by ``interpolation.write_tracks_3d`` under ``output.save_tracks_3d``:
    one ``obj_{i}_xyz`` array of ``(steps, points, 3)`` world positions. On a
    dynamic run they draw that motion as polylines standing still in one frame —
    the complement of the temporal splat item, which shows the same motion by
    playing it.

    NOTE ``final/tracks_2d.npz`` is deliberately NOT used, though it carries
    the same ``obj_{i}_xyz`` world arrays (one writer owns that schema — see
    ``interpolation._obj_xyz_members``) alongside its 2D ``obj_{i}_uv`` ones. It is
    the DENSE set: 2048 anchors per object against these 16, i.e. thousands of
    polylines to overlay. It is also gated on a different flag and needs a
    deformation field, so its absence would say nothing about this one's.
    """
    p = final_dir / "tracks_3d.npz"
    return [(p, "3D tracks")] if p.is_file() else []


def _cameras(colmap_dir: Path) -> list[dict]:
    """``loadColmap`` messages for a colmap dir's per-frame models."""
    models = _first_and_last(_sparse_models(colmap_dir))
    single = len(models) == 1
    return [_colmap_source(colmap_dir, m, "cameras" if single else _frame_label(m))
            for m in models]


def _sources_for(run_dir: Path, complete: bool) -> tuple[list[dict], bool]:
    """``(sources, has_splats)`` — every source the run has produced so far.

    An add-only list. Cameras + depth cloud come from ``preprocessing/``, and the
    finished reconstruction adds only its world + background splats on top.
    ``final/colmap/`` is a fallback for a run whose preprocessing wrote none — its
    cameras are the same input cameras, so loading both would double the frustums.

    Until FINAL's splats exist, the newest per-object previews stand in for them
    (:func:`_preview_splat_sources`), and count as splats for ``has_splats``.

    ``has_splats`` rides along because :func:`write_manifest` needs it for
    ``hidePoints``; asking :func:`_splat_sources` again there would repeat every
    probe it just did (a per-frame directory scan on a dynamic run), on every tick.
    """
    final_dir = Path(run_dir) / "final"
    msgs = _cameras(Path(run_dir) / "preprocessing" / "colmap") or _cameras(final_dir / "colmap")
    splats = _splat_sources(final_dir / "gaussians", complete)
    previews = [] if splats else _preview_splat_sources(Path(run_dir))
    voxel_grid = _voxel_grid_source(Path(run_dir))
    cubes = Path(run_dir) / "pose_cubes.glb"   # written once, atomically: no sentinel needed
    pose_cubes = [_asset_source(cubes, POSE_CUBES_LABEL)] if cubes.is_file() else []
    tracks = [_asset_source(p, label) for p, label in _tracks(final_dir)]
    return (msgs + splats + previews + voxel_grid + pose_cubes + tracks,
            bool(splats or previews))


# Last payload published per manifest path, so an unchanged one is not rewritten.
# Keyed by path and never evicted: one short JSON string per run of this process.
_PUBLISHED: dict[Path, str] = {}


def _manifest_path(run_dir: Path) -> Path:
    """Where the run's source list is published for its viewer page to poll.

    A *sibling* of the run dir, not a file inside it: this is viewer plumbing, and
    the run dir is the run's outputs (which the apps list back to the user).
    """
    run_dir = Path(run_dir)
    return run_dir.with_name(run_dir.name + ".viewer.json")


def write_manifest(run_dir: Path, done: bool = False) -> bool:
    """Publish what the run has produced; ``True`` once there is anything to show.

    Called every tick of a run: the page polls this file and posts whatever is new
    to it into the live scene, so this — not :func:`viewer_html` — is how content
    reaches a viewer that is already up. ``done`` tells the page to stop polling.

    Ticks are driven by subprocess output, so they can run several times a second,
    while the payload changes a handful of times in a whole run (each source as it
    lands, then ``done``) and the page polls at a fixed 1.5 s. So an unchanged
    payload is not rewritten: the content is already on disk and re-publishing it
    is pure churn on the event-loop thread.
    """
    sources, has_splats = _sources_for(run_dir, complete=done)
    payload = json.dumps({
        "sources": sources,
        "done": done,
        # True from the tick the splats exist — see this module's docstring for why
        # this is a display request and not a decision about what to load.
        "hidePoints": has_splats,
        # Same trigger, a separate flag: the voxel-grid preview is superseded
        # (not replaced — it stays loaded) once the real reconstruction lands.
        "hideVoxelGrid": has_splats,
        # Where the run streams each object's current placement while optimising its pose
        # (GenIA's live_poses); the page moves the voxel grid's per-object nodes by it.
        "livePoses": _url(Path(run_dir) / "live_poses.json"),
    }, separators=(",", ":"))
    path = _manifest_path(run_dir)
    if _PUBLISHED.get(path) != payload:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        # Padded to a fixed size (trailing spaces, which JSON ignores): the file server reads
        # the size before the content, so a version replaced in between must not differ in it.
        tmp.write_text(payload + " " * (-len(payload) % 16384))
        tmp.replace(path)  # atomic: a poll landing mid-write would not parse
        _PUBLISHED[path] = payload
    return bool(sources)


#: The frame around the 3D view: one of the page's panels (``style.py``, 3DView's glass) on the
#: viewer's own ground colour, so the iframe edge does not show.
_FRAME = ("border:1px solid var(--block-border-color);border-radius:12px;"
          "box-shadow:var(--glass-shadow);background:#1b1b1d")


def viewer_html(run_dir: Path | None, height: int = 720, dev: bool = False) -> str:
    """The ``<iframe>`` markup for a run's 3D scene, or a placeholder.

    The markup depends on nothing but the run dir, so it is emitted **once** per
    run: everything that lands later reaches the live scene through the manifest
    the page polls, and the iframe (bundle, loaded content, viewpoint) survives.

    ``dev`` adds the expected bundle path to the not-found message. Off, that path is
    withheld like every other: no disk path reaches the deployed UI.
    """
    def shell(body: str) -> str:
        return (f'<div style="height:{height}px;{_FRAME};overflow:hidden;display:flex;'
                f'align-items:center;justify-content:center">{body}</div>')

    if not VIEWER_DIST.joinpath("webview.js").is_file():
        where = (f'<br><code style="font-size:12px">{VIEWER_DIST / "webview.js"}'
                 '</code><br>Build it (<code>npm run build</code>) or set '
                 '<code>DEMO_VIEWER_DIST</code>.') if dev else ''
        return shell(
            '<div style="color:#9aa0aa;font-size:13px;padding:16px;text-align:center">'
            f'3D view unavailable.{where}</div>')

    if run_dir is None:
        return shell('<div style="color:#9aa0aa;font-size:13px">3D view appears once '
                     'preprocessing has run.</div>')

    query = urllib.parse.urlencode({
        "bundle": _bundle_url(),
        "manifest": _url(_manifest_path(run_dir)),
        "ui": "1" if UI_DEFAULT else "0",
    })
    src = f"{_url(VIEWER_PAGE)}?{query}"
    return (
        # `viewer3d` is what UI_TOGGLE_JS addresses the frame by.
        f'<iframe class="viewer3d" src="{src}" style="width:100%;height:{height}px;'
        f'{_FRAME}" '
        f'allow="fullscreen"></iframe>'
    )
