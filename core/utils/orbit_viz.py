"""FINAL's in-process renders, from the exported ``final/`` folder.

Nothing about the renders lives here: this drives the renderers of
:mod:`genia.core.utils.render_final_results` in-process, through the same
``process_run`` its CLI uses, so the pipeline and the CLI are one render with one
set of defaults, writing the same paths the CLI writes.  Both rebuild the run
from disk, so they go after everything else FINAL writes.

- :func:`export_orbit_viz`  -> ``{final}/viz/orbit/orbit.mp4``
- :func:`export_track_viz`  -> ``{final}/viz/track_overlays/{train_views,synth_nvs|test_views}/``
- :func:`export_world_space_viz` -> ``{final}/viz/world_space/``
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path


def export_orbit_viz(cfg, final_output_dir: str, device=None) -> None:
    """Render ``{final_output_dir}/viz/orbit/orbit.mp4``, gated by ``output.save_viz_orbit``.

    A no-op for a run with nothing to pose (no ``poses.json`` — e.g. a
    tracks-only run — or no gaussians/meshes because the run's
    ``save_output_ply``/``save_output_mesh`` were off), and idempotent: an
    existing ``orbit.mp4`` is kept.

    Failures are caught and reported rather than raised.  This is cosmetic
    output written at the very end of a run that may have cost hours, so it
    must never be what destroys it.
    """
    if not cfg.output.save_viz_orbit:
        return

    print("\n" + "-" * 40)
    print("Rendering the orbit turntable ...")
    print("-" * 40)
    final_dir = Path(final_output_dir)
    try:
        from genia.core.utils import render_final_results as rfr

        try:
            run = rfr.FinalRun(final_dir)
        except FileNotFoundError as exc:   # no poses.json — nothing to pose
            print(f"  [orbit] skipped: {exc}")
            return
        if not run.has_renderable_geometry:
            print("  [orbit] skipped: the run exported no gaussians and no meshes")
            return
        rfr._lazy_imports(gpu=True)
        # process_run builds its own FinalRun (a cheap JSON re-read); ours above
        # is only for the two skip checks.
        rfr.process_run(final_dir, [rfr.RENDERERS["orbit"]],
                        _cli_args(cfg, device), final_dir / "viz")
    except Exception as exc:  # noqa: BLE001 — cosmetic output must not fail the run
        print(f"  [orbit] FAILED ({type(exc).__name__}: {exc}) — the run's "
              f"exports are unaffected; re-render with "
              f"`python -m genia.core.utils.render_final_results "
              f"{final_output_dir} -r orbit`")
        traceback.print_exc()


def export_track_viz(cfg, final_output_dir: str, device=None) -> None:
    """Render ``{final_output_dir}/viz/track_overlays/``, gated by ``output.save_viz_track_overlays``.

    Two figure assets with the run's 3D tracks drawn on: ``train_views/`` from each
    dataset camera, and a second view off the training axis.  Their own folder, NOT
    ``viz/train_views`` or ``viz/synth_nvs``: these pixels are annotated, and every dir
    named after a render an evaluation reads has to stay clean enough to be scored.

    **Which off-axis view is the dataset's decision, on exactly the terms that decide
    ``renders_synth_nvs/``**: :func:`config.synth_nvs_enabled`.  A dataset with a GT
    held-out split (OursActionBench here; GSO/CO3D are static and skipped below) gets
    ``test_views/`` -- a real camera the run never saw, which is strictly the better
    figure and the one its NVS numbers were scored from.  Only a dataset with no such
    split gets ``synth_nvs/``, a synthesized view.  Reading the same resolver
    is what keeps the overlay from synthesizing a novel view for a run whose
    ``renders_synth_nvs/`` was, correctly, never written.

    Written for every DYNAMIC run, not only those with a ``tracks_2d.npz``: a run with
    no cross-frame correspondence gets the same viewpoint and resolution with nothing
    drawn on top, so it can be cropped to the same box as an annotated run beside it.

    A **static** run (one timestamp — GSO, CO3D, mvcustom) is skipped: there is no
    trajectory to draw, so the overlay is a supersampled copy of ``train_views`` and
    the figures it feeds are the dynamic ones.  The post-hoc CLI still renders it on
    an explicit ``-r track_overlay``.

    Idempotent and failure-tolerant on the same terms as :func:`export_orbit_viz`.
    """
    if not cfg.output.save_viz_track_overlays:
        return
    final_dir = Path(final_output_dir)

    print("\n" + "-" * 40)
    print("Rendering the 3D track overlays ...")
    print("-" * 40)
    try:
        from genia.core.utils import render_final_results as rfr

        try:
            run = rfr.FinalRun(final_dir)
        except FileNotFoundError as exc:   # no poses.json — nothing to pose
            print(f"  [tracks] skipped: {exc}")
            return
        if not run.has_renderable_geometry:
            print("  [tracks] skipped: the run exported no gaussians and no meshes")
            return
        if not run.is_dynamic:
            print("  [tracks] skipped: static run (a single timestamp) — the track "
                  "overlays are a dynamic-only figure asset")
            return
        from genia.core.utils.config import synth_nvs_enabled

        off_axis = ("track_overlay_nvs" if synth_nvs_enabled(cfg)
                    else "track_overlay_test")
        rfr._lazy_imports(gpu=True)
        rfr.process_run(final_dir,
                        [rfr.RENDERERS["track_overlay"], rfr.RENDERERS[off_axis]],
                        _cli_args(cfg, device), final_dir / "viz")
    except Exception as exc:  # noqa: BLE001 — cosmetic output must not fail the run
        print(f"  [tracks] FAILED ({type(exc).__name__}: {exc}) — the run's "
              f"exports are unaffected; re-render with "
              f"`python -m genia.core.utils.render_final_results "
              f"{final_output_dir} -r track_overlay -r track_overlay_nvs "
              f"(or -r track_overlay_test)`")
        traceback.print_exc()


def export_world_space_viz(cfg, final_output_dir: str, device=None) -> None:
    """Render ``{final_output_dir}/viz/world_space/``, gated by ``output.save_viz_world_space``.

    The whole scene from an overview camera per view: every object posed into WORLD
    space (the Sim(3) axes and the background point cloud are opt-in renderer
    options, off by default).  Root motion is KEPT
    (unlike the orbit, which pins the pose), so this is the render that shows the scene's
    LAYOUT -- where the objects sit relative to each other and to the background, and
    where they travel.  Framed on the FOREGROUND, so the objects fill the frame instead of
    the background dictating a zoom-out.

    Written for EVERY run: no ``is_dynamic`` gate (a static scene still has a layout) and,
    deliberately, no ``has_c2w`` gate.  ``run_final``'s own world-space keyframe render
    skips a run whose cameras are all identity (e.g. every OursActionBench run), so this
    is the only scene view such a run gets.  With identity cameras world space IS camera
    space, and an elevated off-axis look at the objects and their background is still
    worth having.

    Idempotent and failure-tolerant on the same terms as :func:`export_orbit_viz`.
    """
    if not cfg.output.save_viz_world_space:
        return

    print("\n" + "-" * 40)
    print("Rendering the world-space scene overview ...")
    print("-" * 40)
    final_dir = Path(final_output_dir)
    try:
        from genia.core.utils import render_final_results as rfr

        try:
            run = rfr.FinalRun(final_dir)
        except FileNotFoundError as exc:   # no poses.json — nothing to pose
            print(f"  [world] skipped: {exc}")
            return
        if not run.has_renderable_geometry:
            print("  [world] skipped: the run exported no gaussians and no meshes")
            return
        rfr._lazy_imports(gpu=True)
        rfr.process_run(final_dir, [rfr.RENDERERS["world_space"]],
                        _cli_args(cfg, device), final_dir / "viz")
    except Exception as exc:  # noqa: BLE001 — cosmetic output must not fail the run
        print(f"  [world] FAILED ({type(exc).__name__}: {exc}) — the run's "
              f"exports are unaffected; re-render with "
              f"`python -m genia.core.utils.render_final_results "
              f"{final_output_dir} -r world_space`")
        traceback.print_exc()


def _cli_args(cfg, device) -> argparse.Namespace:
    """The ``output.viz_orbit_*`` knobs, as the CLI arguments ``process_run`` takes.

    Built by the CLI's own parser, so FINAL inherits every default it does not
    deliberately set — including `up=auto`, which the renderer resolves from the
    run's `config.yaml`, identically for both entry points.

    Shared with :func:`export_track_viz` and :func:`export_world_space_viz`, which want
    exactly that "every default" behaviour and set nothing of their own: the three `-O`
    below are `orbit.`-scoped, and `resolve_options` drops another render's scoped knob,
    so they are inert there.  `--fps` is shared rather than duplicated per render.
    """
    from genia.core.utils import render_final_results as rfr

    out = cfg.output
    return rfr.build_parser().parse_args([
        "--fps", str(int(out.viz_orbit_fps)),
        "--device", str(device) if device is not None else "cuda",
        "-O", f"orbit.n_frames={int(out.viz_orbit_frames)}",
        "-O", f"orbit.slowmo={int(out.viz_orbit_slowmo)}",
        "-O", f"orbit.up={out.viz_orbit_up}",
        "-O", "orbit.keep_frames=false",   # the mp4 is ~10x smaller than the PNGs
    ])
