"""FINAL: evaluate the finished state and write the method-agnostic ``final/`` layout.

Called by the runner (``core/run.py``) after the last block.  This is the one writer of
``final/`` (poses.json, gaussians/, meshes/, renders, tracks, colmap/, viz/), so every run
is scored and rendered from the same files.
"""

import os

import numpy as np

from genia.core.utils.config import synth_nvs_enabled
from genia.core.utils.console import block_header
from genia.core.utils.evaluation import evaluate_block, save_keyframes_video
from genia.core.utils.pipeline_state import save_pipeline_cache
from genia.core.utils.timing import PipelineTimer, get_timer, plot_pipeline_timing

_TIMER: PipelineTimer = get_timer()


def _build_perobj_mesh_warp_kwargs(state, cfg, device, *, label: str) -> dict:
    """Device-resident per-object mesh-deformation warp kwargs for the
    pose-refinement (``refine_poses_for_sequence``) and FINETUNE
    (``finetune_canonical_tokens``) entry points.

    When GT_SHAPES_INVERSION populated ``state.canonical_mesh_*`` (per-
    canonical-mesh-vertex Φ + R), moves the per-object dicts to ``device``
    once so each path warps the canonical Gaussian per (obj, frame) BEFORE
    applying the Sim(3) pose.  All-None (rigid-only path) when no mesh field
    is loaded.  Caller splats the result via ``**kwargs``.
    """
    dw = cfg.deformation_warp
    knobs = {
        "warp_knn_k": int(dw.knn_k),
        "warp_knn_eps": float(dw.knn_eps),
        "warp_knn_chunk_size": int(dw.knn_chunk_size),
    }
    if not state.canonical_mesh_verts:
        return {
            "canonical_mesh_verts_per_obj": None,
            "per_frame_mesh_verts_per_obj": None,
            "per_frame_mesh_rotations_per_obj": None,
            "canonical_mesh_faces_per_obj": None,
            **knobs,
        }
    print(
        f"  {label}: enabled — "
        f"{len(state.canonical_mesh_verts)} object(s) with per-frame Φ + R"
    )
    return {
        "canonical_mesh_verts_per_obj": {
            oi: t.to(device) for oi, t in state.canonical_mesh_verts.items()
        },
        "per_frame_mesh_verts_per_obj": {
            oi: {fi: t.to(device) for fi, t in pf.items()}
            for oi, pf in state.canonical_mesh_per_frame_verts.items()
        },
        "per_frame_mesh_rotations_per_obj": {
            oi: {fi: t.to(device) for fi, t in pf.items()}
            for oi, pf in state.canonical_mesh_per_frame_rotations.items()
        },
        "canonical_mesh_faces_per_obj": {
            oi: t.to(device) for oi, t in state.canonical_mesh_faces.items()
        },
        **knobs,
    }


def _project_tracks_to_2d(tracks_3d, all_frame_indices, K_per_frame, c2w_per_frame):
    """Project 3D tracks (R3 world space) to 2D pixel coords, accounting for c2w."""
    tracks_2d = {}
    for obj_idx, pts_3d in tracks_3d.items():
        uv_list = []
        for t_idx, fi_t in enumerate(all_frame_indices):
            K_t = K_per_frame[fi_t]
            pts_t = pts_3d[t_idx]  # (P, 3)
            c2w = c2w_per_frame[fi_t] if c2w_per_frame else None
            if c2w is not None:
                w2c = np.linalg.inv(c2w.astype(np.float64)).astype(np.float32)
                pts_t = pts_t @ w2c[:3, :3].T + w2c[:3, 3]
            proj = (K_t @ pts_t.T).T  # (P, 3)
            z = proj[:, 2:3]
            z = np.where(np.abs(z) < 1e-6, 1e-6, z)
            uv_list.append(proj[:, :2] / z)
        tracks_2d[obj_idx] = np.stack(uv_list, axis=0)  # (T, P, 2)
    return tracks_2d


def _deformation_warp_kwargs(cfg, state):
    """Per-canonical-mesh-vertex deformation-field kwargs (actionmesh-only;
    empty dicts collapse to None).  Shared by the FINAL warp consumers
    (save_per_object_ply, canonical renders, tapvid tracks)."""
    _dw = cfg.deformation_warp
    return dict(
        canonical_mesh_verts=(state.canonical_mesh_verts or None),
        per_frame_mesh_verts=(state.canonical_mesh_per_frame_verts or None),
        per_frame_mesh_rotations=(state.canonical_mesh_per_frame_rotations or None),
        canonical_mesh_faces=(state.canonical_mesh_faces or None),
        warp_knn_k=int(_dw.knn_k), warp_knn_eps=float(_dw.knn_eps),
        warp_knn_chunk_size=int(_dw.knn_chunk_size),
    )


def _anchor_mesh_visibility(
    xyz, uv, all_frame_indices, K_per_frame, c2w_per_frame,
    interpolated_poses, per_frame_mesh_verts, canonical_mesh_faces, H, W,
    *, tol_frac=0.02,
):
    """Per-anchor per-frame visibility via an nvdiffrast depth buffer of the
    deformed, per-frame-POSED mesh (the occlusion oracle — a surface, resolving
    thin parts a 64^3 voxel grid would not).  An anchor is OCCLUDED (self-
    occlusion) only when it projects onto the object silhouette AND its
    camera-space depth is behind the rasterised surface by more than ``tol_frac``
    of the median anchor depth.  Everything else stays visible.  Returns
    ``{obj: (T, N) bool}``; all-visible when the mesh is unavailable or anything
    fails — the visibility test must NEVER break the artifact."""
    out = {oi: np.ones(xyz[oi].shape[:2], dtype=bool) for oi in xyz}
    if not per_frame_mesh_verts or not canonical_mesh_faces or not interpolated_poses:
        return out
    try:
        import torch
        import nvdiffrast.torch as dr
        from genia.core.utils.interpolation import _quat_to_matrix
        from genia.core.utils.mesh_rendering import render_depth_and_alpha
        dev = "cuda"
        glctx = dr.RasterizeCudaContext()

        def _np(a):
            return a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)

        for oi, pts in xyz.items():                       # pts (T, N, 3) world R3
            faces = canonical_mesh_faces.get(oi)
            pfv = per_frame_mesh_verts.get(oi)
            poses = interpolated_poses.get(oi)
            if faces is None or pfv is None or poses is None:
                continue
            faces_t = torch.as_tensor(_np(faces), dtype=torch.int32, device=dev)
            vis = np.ones(pts.shape[:2], dtype=bool)
            for t, fi in enumerate(all_frame_indices):
                fint = int(fi.frame) if hasattr(fi, "frame") else int(fi)
                mv_local = pfv.get(fint)
                pose = poses.get(fi)
                if mv_local is None or pose is None:
                    continue                              # no per-frame mesh → leave visible
                # Pose object-local (PyTorch3D) mesh verts → world R3 with the
                # SAME transform compute_object_tracks applies to the anchors:
                # scale → p@R → +t → negate XY → c2w.
                v = torch.as_tensor(_np(mv_local), dtype=torch.float32, device=dev)
                q = pose["rotation"].detach().float().reshape(-1).to(dev)
                q = q / q.norm()
                Rm = _quat_to_matrix(q.unsqueeze(0))[0]   # (3,3), row-vector p@R
                s = pose["scale"].detach().float().reshape(-1).to(dev)
                s = s.expand(3) if s.numel() == 1 else s
                tt = pose["translation"].detach().float().reshape(-1).to(dev)
                vw = (v * s) @ Rm + tt
                vw[:, :2] *= -1                            # PyTorch3D → R3 world
                vw = vw.detach().cpu().numpy()
                K = np.asarray(K_per_frame[fi], np.float64)
                c2w = c2w_per_frame[fi] if c2w_per_frame else None
                if c2w is not None and not np.allclose(c2w, np.eye(4)):
                    w2c = np.linalg.inv(np.asarray(c2w, np.float64))
                    vc = vw @ w2c[:3, :3].T + w2c[:3, 3]  # world R3 → cam R3
                    ac = pts[t] @ w2c[:3, :3].T + w2c[:3, 3]
                else:
                    vc, ac = vw, pts[t]
                depth, alpha = render_depth_and_alpha(
                    torch.as_tensor(vc, dtype=torch.float32, device=dev), faces_t,
                    glctx, float(K[0, 0]), float(K[1, 1]), float(K[0, 2]),
                    float(K[1, 2]), int(H), int(W))
                D = _np(depth).squeeze()                  # (H, W) camera z-depth
                A = _np(alpha).squeeze() > 0.5
                u = np.round(uv[oi][t, :, 0]).astype(int)
                w = np.round(uv[oi][t, :, 1]).astype(int)
                inb = (u >= 0) & (u < W) & (w >= 0) & (w < H)
                uc, wc = np.clip(u, 0, W - 1), np.clip(w, 0, H - 1)
                zc = ac[:, 2]
                med = np.median(zc[np.isfinite(zc)]) if np.isfinite(zc).any() else 1.0
                tol = tol_frac * max(1e-6, abs(float(med)))
                surf = D[wc, uc]
                # OCCLUDED only when confident: in-bounds, on the silhouette, and
                # clearly behind the front surface.  Everything else → visible.
                occ = inb & A[wc, uc] & np.isfinite(surf) & (zc > surf + tol)
                vis[t] = ~occ
            out[oi] = vis
    except Exception as e:  # noqa: BLE001 — visibility must never break the artifact
        print(f"  [tapvid] mesh visibility skipped ({type(e).__name__}: {e})")
        return {oi: np.ones(xyz[oi].shape[:2], dtype=bool) for oi in xyz}
    return out


def _wants_track_outputs(cfg) -> bool:
    """Does this run want ANY of the three per-object track outputs?

    The gate for entering the track pass at all.  ``save_tracks_2d`` is deliberately
    absent: ``tracks_2d.npz`` is written by ``_save_tracks_2d``, which anchors on
    canonical points and runs on its own, not out of this pass.
    """
    o = cfg.output
    return bool(o.save_tracks_3d or o.save_viz_tracks_3d or o.save_viz_tracks_2d)


def _save_object_track_viz(
    cfg, sequence, output_dir, *, frame_indices, all_frame_indices,
    interpolated_poses, K_per_frame, c2w_per_frame, paths, warp_kwargs,
    canonical_gaussians=None,
):
    """Write the ``tracks_3d_world_space.png`` / ``tracks_2d_world_space.mp4``
    object-track visualizations.

    ``compute_object_tracks`` picks the highest-priority anchor source
    available (canonical Gaussians → voxels → mesh verts); the left video
    panel is the canonical render.
    """
    import torch

    from genia.core.utils.interpolation import compute_object_tracks, write_tracks_3d
    from genia.core.utils.visualization import (
        visualize_object_tracks_2d,
        visualize_object_tracks_3d,
    )

    canonical_gaussians = canonical_gaussians or {}
    print("\n  Computing object tracks...")
    tracks_3d = compute_object_tracks(
        canonical_gaussians, interpolated_poses,
        all_frame_indices, num_points=16,
        c2w_per_frame=c2w_per_frame,
        **warp_kwargs,
    )
    if not tracks_3d:
        print("  No trackable objects — skipping track visualizations.")
        return

    if cfg.output.save_viz_tracks_3d:
        tracks_3d_path = os.path.join(
            output_dir,
            f"{cfg.dataset.scene_name}_tracks_3d_world_space.png",
        )
        visualize_object_tracks_3d(
            tracks_3d, all_frame_indices,
            output_path=tracks_3d_path,
            keyframe_indices=frame_indices,
        )
    if cfg.output.save_tracks_3d:
        write_tracks_3d(output_dir, tracks_3d)   # the same tracks as data

    if not cfg.output.save_viz_tracks_2d:
        return

    tracks_2d = _project_tracks_to_2d(
        tracks_3d, all_frame_indices, K_per_frame, c2w_per_frame,
    )

    # Side-by-side track overlay: [reconstruction | GT].  One unified MP4
    # replaces the two previous separate ``_world_space.mp4`` /
    # ``_gt_world_space.mp4`` videos.
    from genia.core.utils.interpolation import render_canonical_frames_to_numpy
    from genia.core.utils.io_utils import load_image

    rendered_track_frames = render_canonical_frames_to_numpy(
        canonical_gaussians, interpolated_poses,
        all_frame_indices, K_per_frame, sequence.W, sequence.H,
        bg_color=torch.ones(3),
        c2w_per_frame=c2w_per_frame,
        **warp_kwargs,
    )

    gt_frames = []
    for fi in all_frame_indices:
        if fi in sequence:
            gt_frames.append(sequence[fi].image)
        else:
            img = load_image(
                os.path.join(paths['frames_path'], paths['image_names'][fi])
            )
            d = cfg.dataset.downscale_factor
            if d > 1:
                img = img[::d, ::d]
            gt_frames.append(img)

    tracks_2d_path = os.path.join(
        output_dir,
        f"{cfg.dataset.scene_name}_tracks_2d_world_space.mp4",
    )
    visualize_object_tracks_2d(
        tracks_2d, all_frame_indices,
        sequence.W, sequence.H,
        output_path=tracks_2d_path,
        rendered_frames=rendered_track_frames,
        gt_frames=gt_frames,
        keyframe_indices=frame_indices,
    )


def _save_tracks_2d(
    out_dir, interpolated_poses, all_frame_indices, K_per_frame, c2w_per_frame,
    warp_kwargs, H, W, *, canonical_gaussians=None, voxel_coords_by_object=None,
    orig_hw=None, num_points=2048,
):
    """Write the dense per-object 2D-track artifact (``final/tracks_2d.npz``)
    for the TAP-Vid-DAVIS eval.  Eval-only: no GT enters here.

    Reuses the convention-correct ``compute_object_tracks`` →
    ``_project_tracks_to_2d`` chain.  Anchors, in priority order: canonical
    Gaussians when an appearance pass produced them, else the canonical voxel
    grid from shape inversion (so the tracks — which depend only on pose + mesh
    deformation, not appearance — are available even on the perframe path).

    No-ops without the per-vertex deformation field in ``warp_kwargs`` (per-frame
    SAM3D shapes, or any rigid canonical): the anchors would merely be transported by
    the per-frame Sim(3) and the artifact would report the pose rather than the
    motion — so nothing is written and the TAP-Vid eval skips the scene.

    Computes the anchors only; the on-disk schema is owned by
    ``interpolation.write_tapvid_tracks``, so every producer of this artifact
    shares one schema."""
    from genia.core.utils.interpolation import compute_object_tracks, write_tapvid_tracks
    warp = dict(warp_kwargs or {})
    has_deformation = bool(
        warp.get("canonical_mesh_verts") and warp.get("per_frame_mesh_verts"))
    if not has_deformation:
        print("  [tapvid] no deformation field — tracks skipped")
        return
    xyz = compute_object_tracks(
        canonical_gaussians or {}, interpolated_poses, all_frame_indices,
        num_points=num_points, voxel_coords_by_object=voxel_coords_by_object,
        c2w_per_frame=c2w_per_frame, **warp,
    )
    uv = _project_tracks_to_2d(xyz, all_frame_indices, K_per_frame, c2w_per_frame)
    if not uv:
        return
    # Per-anchor per-frame visibility (mesh depth-buffer self-occlusion test) so the
    # eval matches only against anchors actually seen at the query frame.
    vis = _anchor_mesh_visibility(
        xyz, uv, all_frame_indices, K_per_frame, c2w_per_frame, interpolated_poses,
        warp.get("per_frame_mesh_verts"), warp.get("canonical_mesh_faces"), H, W)
    write_tapvid_tracks(out_dir, all_frame_indices, uv, xyz, vis, H, W, orig_hw=orig_hw)


def run_final(cfg, state, sequence, pipeline_obj, evaluator=None, device=None,
              timer=None):
    """Evaluate the final state, then save poses, PLY, videos, tracks.

    Runs ``evaluate_block`` on the final canonical Gaussians (same path as
    every other block — writes ``{scene}_final_metrics.json`` into
    ``final/``, gated by ``output.save_metrics``), then saves final outputs:
    poses JSON, PLY files, per-object canonical meshes (gated by
    ``output.save_output_mesh``), per-frame foreground-only dataset-view
    renders (gated by ``output.save_output_renders``), full-sequence
    interpolated videos, and object tracks.  No before/after delta — FINAL
    is the terminal state, so it evaluates the final Gaussians directly.
    """
    import torch
    from genia.core.utils import (
        save_per_object_mesh,
        save_per_object_ply_with_poses,
        save_perframe_per_object_ply,
        save_perframe_poses_json,
    )

    frame_indices = sequence.frame_indices
    paths = sequence.paths
    # Canonical-ness is emergent: a canonical object exists iff some active
    # block built one (see PipelineState.has_canonical).
    has_canonical = state.has_canonical
    if has_canonical:
        # SLATs are guaranteed present here (gaussians.keys() ⊆ slats.keys());
        # Gaussians may be momentarily invalidated, so they get the frame-0
        # per-frame fallback.
        canonical_gaussians = state.canonical_gaussians_with_fallback
        canonical_slats = dict(state.canonical_slats)
        if not state.canonical_gaussians:
            print(
                "  [final] Canonical SLATs present but Gaussians invalidated — "
                "using frame-0 per-frame fallback for canonical outputs."
            )
    else:
        # Per-frame-only run: no active block built a canonical object.  FINAL
        # emits per-frame outputs only (perframe video + perframe scene PLY);
        # canonical mesh / interpolated video / tracks / decoded-canonical viz
        # are skipped.  GSO/OAB eval-asset export still runs.
        canonical_gaussians = {}
        canonical_slats = {}
        print(
            "  [final] per-frame outputs only (no canonical object built) — "
            "skipping canonical mesh / interpolated video / tracks / "
            "decoded-canonical viz."
        )
    is_dynamic = sequence.is_dynamic

    final_output_dir = os.path.join(cfg.output.output_dir, "final")
    os.makedirs(final_output_dir, exist_ok=True)

    # ── Evaluate the final state (same path as every other block) ──
    # No before/after delta: FINAL is the terminal state, so evaluate the
    # final canonical Gaussians directly.  ``evaluate_block`` internally
    # honours ``output.save_metrics``; render writes are forced off here
    # (the comparison panels under ``final/renders/`` duplicate the per-block
    # renders and add no signal — the benchmark uses ``renders_{train,test}/``).
    if evaluator is not None and has_canonical:
        with _TIMER.exclude():
            block_header("EVALUATION: Final state")
            evaluate_block(
                cfg, state, sequence, evaluator, device, final_output_dir,
                suffix="_final", per_frame=False, save_renders=False,
            )

    save_pipeline_cache(state, "final", final_output_dir)

    # ── Per-frame video ──
    if (not has_canonical
            and cfg.output.save_renders and state.perframe_gaussians is not None):
        from genia.core.utils.interpolation import render_perframe_sequence

        perframe_video_path = os.path.join(
            final_output_dir, f"{cfg.dataset.scene_name}_perframe.mp4"
        )
        _bg = torch.ones(3) if cfg.pipeline.white_background else None
        perframe_c2w = {fi: sequence[fi].c2w for fi in frame_indices}
        perframe_K = {fi: sequence[fi].K_matrix for fi in frame_indices}
        render_perframe_sequence(
            state.perframe_gaussians, state.tokens_by_object,
            frame_indices, perframe_K, sequence.W, sequence.H,
            output_path=perframe_video_path,
            bg_color=_bg,
            c2w_per_frame=perframe_c2w,
        )

    # ── Per-frame per-object PLY (per-frame-only pipelines) ──
    # Object-local gaussians/{obj:03d}/{frame:03d}.ply — the same layout the
    # canonical path emits; poses.json carries the per-frame Sim(3). Dataset
    # -agnostic: every per-frame-only pipeline gets the identical structure.
    if not has_canonical and cfg.output.save_output_ply:
        state.ensure_perframe_gaussians()
        if state.perframe_gaussians is not None:
            save_perframe_per_object_ply(
                state.perframe_gaussians,
                final_output_dir,
                compressed=cfg.output.save_compressed_ply,
            )
            # poses.json: per-frame Sim(3) + cameras so an evaluator can
            # resolve obj_idx + compose the object-local per-frame Gaussians
            # into world space.
            from genia.core.utils.interpolation import interpolate_poses

            pf_keys = list(frame_indices)
            save_perframe_poses_json(
                state.perframe_gaussians,
                interpolate_poses(state.tokens_by_object, pf_keys),
                pf_keys,
                final_output_dir,
                c2w_per_frame={fi: sequence[fi].c2w for fi in pf_keys},
                K_per_frame={fi: sequence[fi].K_matrix for fi in pf_keys},
                compressed=cfg.output.save_compressed_ply,
            )

    # Dense 2D tracks (eval-only artifact) for the perframe path: the
    # anchors are the canonical voxel grid from GT shape inversion, so no
    # canonical Gaussian is needed.  A per-frame SAM3D *shape* run has neither
    # the grid nor the deformation field `_save_tracks_2d` requires.
    if (not has_canonical and is_dynamic
            and cfg.output.save_tracks_2d):
        with _TIMER.exclude():
            from genia.core.utils.interpolation import (
                interpolate_poses, interpolate_c2w, interpolate_K,
            )
            _tv = list(frame_indices)  # actionmesh keyframes == all frames
            _save_tracks_2d(
                final_output_dir,
                interpolate_poses(state.tokens_by_object, _tv), _tv,
                interpolate_K({fi: sequence[fi].K_matrix for fi in _tv}, _tv),
                interpolate_c2w({fi: sequence[fi].c2w for fi in _tv}, _tv),
                _deformation_warp_kwargs(cfg, state), sequence.H, sequence.W,
                voxel_coords_by_object=(
                    state.gt_canonical_shape_coords or state.canonical_shape_coords
                ),
                orig_hw=sequence.orig_hw,
                num_points=cfg.output.tapvid_track_points,
            )

    # ── Per-frame renders_train (per-frame-only pipelines) ──
    # The canonical renders_train path below is gated on has_canonical, so a
    # per-frame-only pipeline would otherwise emit no renders_train/.
    # CO3D writes its own flat per-input-view renders_train via
    # export_co3d_eval_assets (the generic writer's frame-keyed/mv-static layout
    # doesn't match co3d's flat {view:03d}.png convention), so skip it here.
    if (not has_canonical and cfg.output.save_output_renders
            and state.perframe_gaussians is not None
            and cfg.dataset.name != "co3d"):
        from genia.core.utils.interpolation import save_perframe_renders_perframe

        perframe_c2w = {fi: sequence[fi].c2w for fi in frame_indices}
        perframe_K = {fi: sequence[fi].K_matrix for fi in frame_indices}
        save_perframe_renders_perframe(
            state.perframe_gaussians, state.tokens_by_object,
            frame_indices, perframe_K, sequence.W, sequence.H,
            output_dir=os.path.join(final_output_dir, "renders_train"),
            bg_color=torch.ones(3),
            c2w_per_frame=perframe_c2w,
        )

    # ── Per-object canonical meshes ──
    if has_canonical and cfg.output.save_output_mesh:
        save_per_object_mesh(
            canonical_slats,
            pipeline_obj,
            final_output_dir,
        )

    # ── Canonical interpolation + visualization ──
    if has_canonical and (
        cfg.output.save_renders
        or cfg.output.save_output_ply
        or cfg.output.save_output_mesh
        or cfg.output.save_output_renders
    ):
        from genia.core.utils.interpolation import (
            compute_pose_axes,
            interpolate_poses,
            render_interpolated_sequence,
            save_canonical_renders_perframe,
        )

        block_header("Full-sequence interpolation + visualization")

        # Dynamic sequences render a dense interpolated video — fill in
        # every frame between sparse keyframes.  Static scenes treat each
        # listed frame as a distinct view; there are no phantom frames to
        # interpolate into, so the "all frames" set stays equal to the
        # actual keyframes.  (Without this gate, a static run would inflate
        # all_frame_indices to every entry in paths['image_names'] and
        # interpolate_c2w would duplicate the single real c2w across all
        # of them, producing N identical renders.)
        if is_dynamic:
            all_frame_indices = list(range(len(paths['image_names'])))
        else:
            all_frame_indices = list(frame_indices)

        print("\n  Interpolating poses for all frames...")
        interpolated_poses = interpolate_poses(state.tokens_by_object, all_frame_indices)
        print(f"  Interpolated poses for {len(all_frame_indices)} frames "
              f"({len(frame_indices)} keyframes)")

        # Interpolate per-frame c2w and intrinsics for all frames
        from genia.core.utils.interpolation import interpolate_c2w, interpolate_K
        keyframe_c2w = {fi: sequence[fi].c2w for fi in frame_indices}
        c2w_per_frame = interpolate_c2w(keyframe_c2w, all_frame_indices)
        keyframe_K = {fi: sequence[fi].K_matrix for fi in frame_indices}
        K_per_frame = interpolate_K(keyframe_K, all_frame_indices)

        # Per-canonical-mesh-vertex deformation field (actionmesh-only).  Same
        # source consumed by Stage-2 rendering guidance + DDA + keyframes.
        _warp_kwargs = _deformation_warp_kwargs(cfg, state)

        # Gated on EITHER export: poses.json places `meshes/` as much as
        # `gaussians/`.  With PLYs off the Gaussians are
        # withheld from the writer (it skips the per-object PLY where an object has
        # none) and `mesh_objs` puts the meshes/{obj}.glb reference in their place.
        if cfg.output.save_output_ply or cfg.output.save_output_mesh:
            _ply_on = cfg.output.save_output_ply
            save_per_object_ply_with_poses(
                canonical_gaussians if _ply_on else {}, interpolated_poses,
                all_frame_indices, final_output_dir,
                scene_name=cfg.dataset.scene_name,
                compressed=cfg.output.save_compressed_ply,
                c2w_per_frame=c2w_per_frame,
                K_per_frame=K_per_frame,
                mesh_objs=None if _ply_on else set(canonical_slats),
                **_warp_kwargs,
            )

        # TAP-Vid dense 2D tracks (eval-only artifact; no GT enters here).
        # Canonical path: anchors come from the canonical Gaussians.
        if is_dynamic and cfg.output.save_tracks_2d:
            with _TIMER.exclude():
                _save_tracks_2d(
                    final_output_dir, interpolated_poses, all_frame_indices,
                    K_per_frame, c2w_per_frame, _warp_kwargs,
                    sequence.H, sequence.W,
                    canonical_gaussians=canonical_gaussians,
                    orig_hw=sequence.orig_hw,
                    num_points=cfg.output.tapvid_track_points,
                )

        # CO3D writes its own flat per-input-view renders_train via
        # export_co3d_eval_assets; the generic frame-keyed/mv-static layout here
        # would emit a view{vv}/ subdir that the co3d eval does not expect.
        if cfg.output.save_output_renders and cfg.dataset.name != "co3d":
            save_canonical_renders_perframe(
                canonical_gaussians, interpolated_poses,
                all_frame_indices, K_per_frame,
                sequence.W, sequence.H,
                output_dir=os.path.join(final_output_dir, "renders_train"),
                bg_color=torch.ones(3),
                c2w_per_frame=c2w_per_frame,
                **_warp_kwargs,
            )

        if cfg.output.save_renders:
            from genia.core.utils.visualization import (
                visualize_pose_axes_2d,
                visualize_pose_axes_3d,
            )

            _bg = torch.ones(3) if cfg.pipeline.white_background else None

            if is_dynamic:
                video_path = os.path.join(final_output_dir, f"{cfg.dataset.scene_name}.mp4")
                render_interpolated_sequence(
                    canonical_gaussians, interpolated_poses,
                    all_frame_indices, K_per_frame, sequence.W, sequence.H,
                    output_path=video_path,
                    gt_frames_path=paths['frames_path'],
                    gt_image_names=paths['image_names'],
                    gt_masks_path=paths['masks_path'],
                    gt_mask_names=paths['mask_names'],
                    bg_color=_bg,
                    c2w_per_frame=c2w_per_frame,
                    sequence=sequence,
                    **_warp_kwargs,
                )

            if is_dynamic and _wants_track_outputs(cfg):
                # Forwards the per-vertex deformation field so actionmesh
                # tracks follow the deforming surface, not just the rigid Sim(3).
                _save_object_track_viz(
                    cfg, sequence, final_output_dir,
                    frame_indices=frame_indices,
                    all_frame_indices=all_frame_indices,
                    interpolated_poses=interpolated_poses,
                    K_per_frame=K_per_frame,
                    c2w_per_frame=c2w_per_frame,
                    paths=paths,
                    warp_kwargs=_warp_kwargs,
                    canonical_gaussians=canonical_gaussians,
                )

            if is_dynamic and (cfg.output.save_pose_axes_3d or cfg.output.save_pose_axes_2d):
                print("\n  Computing pose axes for reference frame visualization...")
                pose_data = compute_pose_axes(interpolated_poses, all_frame_indices)

                if cfg.output.save_pose_axes_3d:
                    axes_3d_path = os.path.join(final_output_dir, f"{cfg.dataset.scene_name}_pose_axes_3d.png")
                    visualize_pose_axes_3d(
                        pose_data, all_frame_indices,
                        output_path=axes_3d_path,
                        keyframe_indices=frame_indices,
                    )

                if cfg.output.save_pose_axes_2d:
                    axes_2d_path = os.path.join(final_output_dir, f"{cfg.dataset.scene_name}_pose_axes_2d.mp4")
                    visualize_pose_axes_2d(
                        pose_data, all_frame_indices,
                        K_per_frame, sequence.W, sequence.H,
                        output_path=axes_2d_path,
                        keyframe_indices=frame_indices,
                        c2w_per_frame=c2w_per_frame,
                    )

    # ── Final keyframe renders: camera-space + world-space ──
    if cfg.output.save_renders:
        # Camera-space keyframes are skipped on multi-view runs: they would
        # land in one final/view{VV}/ dir per view, duplicating renders_train/.
        if not sequence.is_mv:
            save_keyframes_video(
                state, sequence, cfg, suffix="final",
                output_dir=final_output_dir, background=True,
            )
        has_c2w = any(
            not np.allclose(sequence[fi].c2w, np.eye(4))
            for fi in frame_indices
        )
        if has_c2w:
            save_keyframes_video(
                state, sequence, cfg, suffix="final",
                output_dir=final_output_dir, background=True,
                render_space="world",
            )

        # ── Canonical decoded viz: turntable PNG (static) or motion MP4
        # (when state.canonical_mesh_per_frame_verts is populated — actionmesh).
        # Empty (so skipped) for per-frame-only runs.
        canon_gs = state.canonical_gaussians
        if canon_gs:
            from genia.core.utils.rendering import (
                render_canonical_motion_video, render_multiview_comparison,
            )
            canon_renders_dir = os.path.join(
                final_output_dir, "decoded_renders",
            )
            os.makedirs(canon_renders_dir, exist_ok=True)
            for obj_idx in sorted(canon_gs.keys()):
                has_motion = (
                    obj_idx in state.canonical_mesh_verts
                    and obj_idx in state.canonical_mesh_per_frame_verts
                    and obj_idx in state.canonical_mesh_per_frame_rotations
                )
                # A world-space reconstruction (identity Sim(3)) puts the object's
                # placement and metric size in the canonical itself, where cameras orbiting the
                # origin at a fixed distance miss it — auto-frame those instead.
                canon_distance = (
                    cfg.output.render_distance
                    if state.pose_carries_placement(obj_idx)
                    else None  # auto-frame
                )
                if has_motion:
                    out_path = os.path.join(
                        canon_renders_dir,
                        f"{cfg.dataset.scene_name}_obj{obj_idx}_canonical_final.mp4",
                    )
                    render_canonical_motion_video(
                        canon_gs[obj_idx],
                        output_path=out_path,
                        frame_indices=frame_indices,
                        canonical_mesh_verts=state.canonical_mesh_verts[obj_idx],
                        per_frame_mesh_verts=state.canonical_mesh_per_frame_verts[obj_idx],
                        per_frame_mesh_rotations=state.canonical_mesh_per_frame_rotations[obj_idx],
                        canonical_mesh_faces=state.canonical_mesh_faces.get(obj_idx),
                        image_size=cfg.output.render_size,
                        distance=canon_distance,
                        fov=cfg.output.render_fov,
                        K_knn=int(cfg.deformation_warp.knn_k),
                        knn_eps=float(cfg.deformation_warp.knn_eps),
                        knn_chunk_size=int(cfg.deformation_warp.knn_chunk_size),
                    )
                else:
                    out_path = os.path.join(
                        canon_renders_dir,
                        f"{cfg.dataset.scene_name}_obj{obj_idx}_canonical_final.png",
                    )
                    print(f"  Rendering canonical object {obj_idx}...")
                    render_multiview_comparison(
                        canon_gs[obj_idx],
                        mesh_path=None,
                        output_path=out_path,
                        image_size=cfg.output.render_size,
                        distance=canon_distance,
                        fov=cfg.output.render_fov,
                    )

    # ── GSO-30 eval assets (held-out NVS renders + per-frame meshes) ──
    if cfg.dataset.name == "gso":
        with _TIMER.exclude():
            from genia.core.utils.eval_assets_export import export_gso_eval_assets
            export_gso_eval_assets(
                cfg, state, sequence, final_output_dir, pipeline_obj
            )
    # ── OursActionBench eval assets (per-frame NVS + per-frame meshes) ──
    elif cfg.dataset.name == "oursactionbench":
        with _TIMER.exclude():
            from genia.core.utils.eval_assets_export import (
                export_oursactionbench_eval_assets,
            )
            export_oursactionbench_eval_assets(
                cfg, state, sequence, final_output_dir, pipeline_obj
            )
    # ── CO3D LaRa eval assets (held-out 2D NVS renders only) ──
    elif cfg.dataset.name == "co3d":
        with _TIMER.exclude():
            from genia.core.utils.eval_assets_export import export_co3d_eval_assets
            export_co3d_eval_assets(
                cfg, state, sequence, final_output_dir, pipeline_obj
            )

    # ── Synthesized per-timestamp NVS (every dataset with no GT test split) ──
    # ON by default for exactly the datasets the branches above did NOT handle, so
    # every run has SOME novel view.
    # Qualitative only, and deliberately not renders_test/ — nothing scores it.
    if synth_nvs_enabled(cfg):
        with _TIMER.exclude():
            from genia.core.utils.eval_assets_export import export_synth_nvs_assets
            export_synth_nvs_assets(
                cfg, state, sequence, final_output_dir, pipeline_obj
            )

    # ── World-space composites (gaussians/world*, background*, meshes/world*) ──
    # Dataset-agnostic; runs after the per-object assets exist (the mesh join
    # reloads meshes/{obj:03d}.glb) and shares its foreground with the COLMAP
    # cloud below, so the two land in the same world frame.
    if cfg.output.save_world_assets:
        with _TIMER.exclude():
            from genia.core.utils.eval_assets_export import export_world_space_assets
            export_world_space_assets(
                cfg, state, sequence, final_output_dir, pipeline_obj
            )

    # ── COLMAP sparse reconstruction (train+test cameras + colored point cloud) ──
    # Dataset-agnostic; runs after renders_train/renders_test exist so the
    # image copies + NAMEs line up.  Generic datasets get a train-only model.
    if cfg.output.save_colmap:
        with _TIMER.exclude():
            from genia.core.utils.colmap_export import export_colmap_reconstruction
            export_colmap_reconstruction(
                cfg, state, sequence, final_output_dir, pipeline_obj
            )

    # ── Timing summary ──
    if timer is not None:
        plot_pipeline_timing(timer, final_output_dir, cfg.dataset.scene_name)

    # ── Post-hoc renders (final/viz/) ──
    # All three rebuild the run from what everything above wrote to disk, so they go
    # last.  The track overlay needs the tracks_2d.npz written earlier in this block,
    # and the world-space overview needs gaussians/background*.ply.
    with _TIMER.exclude():
        from genia.core.utils.orbit_viz import (
            export_orbit_viz,
            export_track_viz,
            export_world_space_viz,
        )
        export_orbit_viz(cfg, final_output_dir, device)
        export_track_viz(cfg, final_output_dir, device)
        export_world_space_viz(cfg, final_output_dir, device)
