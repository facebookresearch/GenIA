# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
PyTorch3D mesh rendering utilities.

This module provides a wrapper class for rendering meshes using PyTorch3D's
renderer components, including camera positioning, mesh loading with vertex
colors, and multi-view rendering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    BlendParams,
    FoVPerspectiveCameras,
    HardFlatShader,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    TexturesVertex,
    look_at_view_transform,
)

if TYPE_CHECKING:
    from pytorch3d.structures import Meshes


class PyTorch3DMeshRenderer:
    """
    Wrapper class for PyTorch3D mesh rendering functionality.

    Encapsulates camera positioning, mesh loading, and rendering operations
    using PyTorch3D's renderer components.

    Parameters
    ----------
    device : torch.device, optional
        Device to use for rendering. Defaults to CUDA if available.

    Examples
    --------
    >>> renderer = PyTorch3DMeshRenderer()
    >>> views = renderer.get_camera_positions(distance=2.0)
    >>> R, T = views['front']
    >>> mesh = renderer.load_mesh_with_vertex_colors("model.obj")
    >>> image = renderer.render_from_view(mesh, R, T)
    """

    def __init__(self, device: torch.device = None):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    # (elev, azim, up) per view.  Only the POLES differ, and they have to: at elev=+/-90 the
    # usual up is PARALLEL to the view axis, so `look_at_rotation`'s cross product vanishes
    # and its degenerate fallback returns a SINGULAR matrix -- which
    # `render_gaussian_from_view` then tries to invert.  Looking at the origin that stays
    # latent by luck (the eye keeps a ~4e-08 fp32 residue in z that rescues the cross product);
    # any real look-at point rounds the residue away.  The poles' ups are OPPOSITE because
    # that is what reproduces the degenerate fallback's own choice bit-for-bit at the
    # origin -- +Z for top, -Z for bottom; using +Z for both rolls the bottom view 180 degrees.
    _UP, _TOP_UP, _BOTTOM_UP = (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)
    _VIEWS = {'top': (90, 0, _TOP_UP), 'bottom': (-90, 0, _BOTTOM_UP),
              'front': (0, 0, _UP), 'back': (0, 180, _UP),
              'left': (0, -90, _UP), 'right': (0, 90, _UP)}

    def get_camera_positions(
        self,
        distance: float = 2.0,
        at: "Sequence[float]" = (0.0, 0.0, 0.0),
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get camera positions for 6 standard viewpoints.

        Parameters
        ----------
        distance : float
            Camera distance from the look-at point.
        at : sequence of 3 floats
            Point the six cameras orbit and look at; defaults to the origin.  An asset
            that does not sit there needs ``rendering.auto_frame_canonical`` to supply it.

        Returns
        -------
        dict
            Maps view name -> (R, T) tuple for camera transform.
            View names: 'top', 'bottom', 'front', 'back', 'left', 'right'.
        """
        at_pt = (tuple(float(v) for v in at),)
        return {
            name: look_at_view_transform(
                dist=distance, elev=elev, azim=azim, at=at_pt, up=(up,))
            for name, (elev, azim, up) in self._VIEWS.items()
        }

    def load_mesh_with_vertex_colors(self, mesh_path: str) -> "Meshes":
        """
        Load a mesh from OBJ file with vertex colors.

        Parameters
        ----------
        mesh_path : str
            Path to the OBJ file.

        Returns
        -------
        Meshes
            PyTorch3D Meshes object with vertex colors as texture.
        """
        # Load vertex colors from OBJ file manually
        verts_rgb = None
        with open(mesh_path, 'r') as f:
            vertex_colors = []
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    # Check if vertex has color (format: v x y z r g b)
                    if len(parts) >= 7:
                        r, g, b = float(parts[4]), float(parts[5]), float(parts[6])
                        vertex_colors.append([r, g, b])
                    else:
                        # No color, use white
                        vertex_colors.append([1.0, 1.0, 1.0])

            if vertex_colors:
                verts_rgb = torch.tensor(vertex_colors, dtype=torch.float32, device=self.device)[None]

        # Load mesh using PyTorch3D
        mesh = load_objs_as_meshes([mesh_path], device=self.device)

        # Apply vertex colors as texture
        if verts_rgb is not None:
            mesh.textures = TexturesVertex(verts_features=verts_rgb)
        elif mesh.textures is None:
            # Fallback: Add simple white texture if none exists
            verts_rgb = torch.ones_like(mesh.verts_packed())[None]
            mesh.textures = TexturesVertex(verts_features=verts_rgb.to(self.device))

        return mesh

    def render_from_view(
        self,
        mesh: "Meshes",
        R: torch.Tensor,
        T: torch.Tensor,
        image_size: int = 512,
        fov: float = 60.0,
    ) -> np.ndarray:
        """
        Render mesh from a specific viewpoint.

        Parameters
        ----------
        mesh : Meshes
            PyTorch3D mesh object.
        R : torch.Tensor
            Rotation matrix, shape (1, 3, 3).
        T : torch.Tensor
            Translation vector, shape (1, 3).
        image_size : int
            Output image size (square).
        fov : float
            Field of view in degrees.

        Returns
        -------
        np.ndarray
            Rendered image, shape (H, W, 3), values in [0, 1].
        """
        # Move mesh to device
        mesh = mesh.to(self.device)

        # Create cameras
        cameras = FoVPerspectiveCameras(
            device=self.device,
            R=R.to(self.device),
            T=T.to(self.device),
            fov=fov
        )

        # Create rasterizer
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        # Create blend params that don't attenuate colors
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(1.0, 1.0, 1.0))

        # Create renderer with flat shader (no lighting, just albedo/vertex colors)
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=HardFlatShader(
                device=self.device,
                cameras=cameras,
                blend_params=blend_params
            )
        )

        # Render
        with torch.no_grad():
            images = renderer(mesh)

        # Extract RGB (drop alpha channel)
        image = images[0, ..., :3]

        # HardFlatShader seems to darken colors even without lighting
        # Apply a brightness boost to compensate (empirical adjustment)
        image = torch.clamp(image * 1.8, 0.0, 1.0)

        # Flip vertically and horizontally to match coordinate system
        image = torch.flip(image, dims=[0, 1])

        return image.cpu().numpy()

    def render_all_views(
        self,
        mesh: "Meshes",
        view_names: List[str],
        distance: float = 2.0,
        image_size: int = 512,
        fov: float = 60.0,
        at: "Sequence[float]" = (0.0, 0.0, 0.0),
    ) -> Dict[str, np.ndarray]:
        """
        Render mesh from all specified viewpoints.

        Parameters
        ----------
        mesh : Meshes
            PyTorch3D mesh object.
        view_names : list of str
            List of view names to render (e.g., ['front', 'back', 'left', 'right', 'top', 'bottom']).
        distance : float
            Camera distance from the look-at point.
        image_size : int
            Output image size (square).
        fov : float
            Field of view in degrees.
        at : sequence of 3 floats
            Look-at point; defaults to the origin.  ``render_multiview_comparison``
            must pass the SAME value it used for its Gaussian row, or the two rows
            of the comparison grid end up framed differently.

        Returns
        -------
        dict
            Maps view_name -> rendered image as np.ndarray (H, W, 3).
        """
        views = self.get_camera_positions(distance=distance, at=at)
        renders = {}

        for view_name in view_names:
            R, T = views[view_name]
            renders[view_name] = self.render_from_view(
                mesh, R, T,
                image_size=image_size,
                fov=fov
            )

        return renders


# ════════════════════════════════════════════════════════════════════════════
# DiffMC + nvdiffrast differentiable rendering
# ════════════════════════════════════════════════════════════════════════════
#
# Differentiable pipeline: occupancy grid → DiffMC mesh → nvdiffrast render.


def diffmc_mesh(occ_probs: torch.Tensor, device: torch.device):
    """Extract a differentiable mesh from occupancy probabilities via DiffMC.

    Pads the grid with a 1-voxel border of zeros and filters out the
    boundary faces that DiffMC always generates.

    Parameters
    ----------
    occ_probs : torch.Tensor
        Occupancy probabilities ``(D, H, W)`` in ``[0, 1]``.
    device : torch.device
        Device for the DiffMC module.

    Returns
    -------
    verts : torch.Tensor
        Vertices ``(V, 3)`` in ``[-0.5, 0.5]³``.
    faces : torch.Tensor
        Triangle indices ``(F, 3)`` as int32 (nvdiffrast requirement).
    """
    from diso import DiffMC

    padded = torch.nn.functional.pad(occ_probs, (1, 1, 1, 1, 1, 1), value=0.0)
    dmc = DiffMC(dtype=torch.float32).to(device)
    verts, faces = dmc(padded, isovalue=0.5)

    orig_size = occ_probs.shape[0]
    padded_size = padded.shape[0]
    verts = (verts * padded_size - 1) / orig_size - 0.5

    margin = 0.5 / orig_size
    lo, hi = -0.5 - margin + 0.01, 0.5 + margin - 0.01
    vert_inside = ((verts > lo) & (verts < hi)).all(dim=1)
    face_inside = vert_inside[faces].all(dim=1)

    kept_faces = faces[face_inside]
    used_verts = torch.unique(kept_faces)
    remap = torch.full((verts.shape[0],), -1, dtype=torch.long, device=device)
    remap[used_verts] = torch.arange(used_verts.shape[0], device=device)
    verts = verts[used_verts]
    faces = remap[kept_faces].int()

    return verts, faces


def project_to_clip(
    verts_r3: torch.Tensor,
    fx: float, fy: float, cx: float, cy: float,
    H: int, W: int,
    near: float = 0.1, far: float = 10.0,
) -> torch.Tensor:
    """Project R3 camera-space vertices to OpenGL clip space for nvdiffrast.

    Parameters
    ----------
    verts_r3 : torch.Tensor
        Vertices ``(V, 3)`` in R3 camera space (X-right, Y-down, Z-forward).
    fx, fy : float
        Focal lengths in pixels.
    cx, cy : float
        Principal point in pixels.
    H, W : int
        Image height and width.
    near, far : float
        Near/far clip planes.

    Returns
    -------
    torch.Tensor
        Clip-space vertices ``(V, 4)`` as ``[x, y, z, w]``.
    """
    x, y, z = verts_r3[:, 0], verts_r3[:, 1], verts_r3[:, 2]

    x_ndc = 2.0 * fx * x / (W * z) + (2.0 * cx / W - 1.0)
    y_ndc = -(2.0 * fy * y / (H * z) + (2.0 * cy / H - 1.0))
    z_ndc = (far + near) / (far - near) - 2.0 * far * near / ((far - near) * z)

    clip = torch.stack([x_ndc * z, y_ndc * z, z_ndc * z, z], dim=-1)
    return clip


def render_depth_and_alpha(
    verts_r3: torch.Tensor,
    faces_int32: torch.Tensor,
    glctx,
    fx: float, fy: float, cx: float, cy: float,
    H: int, W: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Render depth + alpha via nvdiffrast.

    Uses hard rasterization + antialiased edges.  The antialias pass gives
    silhouette gradients at every edge pixel without blurring the image.

    Parameters
    ----------
    verts_r3 : torch.Tensor
        Posed vertices ``(V, 3)`` in R3 camera space.
    faces_int32 : torch.Tensor
        Triangle indices ``(F, 3)`` as int32.
    glctx : nvdiffrast.torch.RasterizeCudaContext
        Rasterization context (stateless, reusable).
    fx, fy, cx, cy : float
        Camera intrinsics.
    H, W : int
        Image resolution.

    Returns
    -------
    depth : torch.Tensor
        Camera-space z-depth ``(H, W)``.
    alpha : torch.Tensor
        Silhouette alpha ``(H, W)`` in ``[0, 1]``.
    """
    import nvdiffrast.torch as dr

    verts_clip = project_to_clip(verts_r3, fx, fy, cx, cy, H, W)

    rast_out, _ = dr.rasterize(glctx, verts_clip[None], faces_int32, resolution=[H, W])

    z_attr = verts_r3[:, 2:3].contiguous()
    depth, _ = dr.interpolate(z_attr[None], rast_out, faces_int32)
    depth = dr.antialias(depth, rast_out, verts_clip[None], faces_int32)

    alpha = (rast_out[..., 3:4] > 0).float()
    alpha = dr.antialias(alpha, rast_out, verts_clip[None], faces_int32)

    # nvdiffrast outputs OpenGL convention (row 0 = bottom); flip to
    # image convention (row 0 = top) to match GT depth/mask.
    return depth[0, :, :, 0].flip(0), alpha[0, :, :, 0].flip(0)


def render_rgba_and_depth(
    verts_r3: torch.Tensor,
    faces_int32: torch.Tensor,
    vert_rgb: torch.Tensor,
    glctx,
    fx: float, fy: float, cx: float, cy: float,
    H: int, W: int,
    uv: "Optional[torch.Tensor]" = None,
    tex: "Optional[torch.Tensor]" = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render colored mesh → (rgb, alpha, depth) via nvdiffrast.

    Sibling of :func:`render_depth_and_alpha` that additionally interpolates
    per-vertex colors.  Used by Stage-2 in-ODE rendering guidance, where the
    SLAT mesh decoder yields ``vertex_attrs[:, :3]`` RGB.

    Supplying ``uv`` + ``tex`` switches the colour source from interpolated
    ``vert_rgb`` to per-pixel sampling of a UV texture, for a textured mesh
    whose renders should show the same appearance as its exported GLB rather
    than a baked-down approximation of it.  Geometry, depth, alpha and antialiasing are
    identical either way; only the RGB fetch differs.

    Parameters
    ----------
    verts_r3 : torch.Tensor
        Posed vertices ``(V, 3)`` in R3 camera space.
    faces_int32 : torch.Tensor
        Triangle indices ``(F, 3)`` as int32.
    vert_rgb : torch.Tensor
        Per-vertex RGB ``(V, 3)`` in ``[0, 1]``.  Ignored when ``tex`` is given.
    glctx : nvdiffrast.torch.RasterizeCudaContext
        Rasterization context (stateless, reusable).
    fx, fy, cx, cy : float
        Camera intrinsics.
    H, W : int
        Image resolution.
    uv : torch.Tensor, optional
        Per-vertex UVs ``(V, 2)`` — indexed by ``faces_int32``, i.e. one UV per
        vertex (as trimesh gives).  Required with ``tex``.
    tex : torch.Tensor, optional
        UV texture ``(Ht, Wt, 3)`` float in ``[0, 1]``, already flipped to
        nvdiffrast's bottom-up row order (glTF's UV origin is top-left).
        Sampled ``linear-mipmap-linear``: minifying a 2048² texture into a ~200 px
        object aliases badly otherwise, and the mip level is driven by the
        screen-space UV derivatives, which is why the rasteriser's ``rast_db``
        is threaded through here.

    Returns
    -------
    rgb : torch.Tensor
        Foreground RGB ``(H, W, 3)``; pixels outside silhouette are zero.
    alpha : torch.Tensor
        Silhouette alpha ``(H, W)`` in ``[0, 1]``.
    depth : torch.Tensor
        Camera-space z-depth ``(H, W)``.
    """
    import nvdiffrast.torch as dr

    verts_clip = project_to_clip(verts_r3, fx, fy, cx, cy, H, W)
    rast_out, rast_db = dr.rasterize(glctx, verts_clip[None], faces_int32, resolution=[H, W])

    if tex is not None:
        uvi, uv_da = dr.interpolate(uv[None].contiguous(), rast_out, faces_int32,
                                    rast_db=rast_db, diff_attrs="all")
        rgb = dr.texture(tex[None].contiguous(), uvi, uv_da,
                         filter_mode="linear-mipmap-linear")
        # dr.texture samples the texture everywhere, including the background (where the
        # interpolated UV is 0); zero it so the caller's "foreground RGB, 0 outside the
        # silhouette" contract holds exactly as on the vert_rgb path.
        rgb = torch.where(rast_out[..., 3:] > 0, rgb, torch.zeros_like(rgb))
    else:
        rgb, _ = dr.interpolate(vert_rgb[None].contiguous(), rast_out, faces_int32)
    rgb = dr.antialias(rgb, rast_out, verts_clip[None], faces_int32)

    z_attr = verts_r3[:, 2:3].contiguous()
    depth, _ = dr.interpolate(z_attr[None], rast_out, faces_int32)
    depth = dr.antialias(depth, rast_out, verts_clip[None], faces_int32)

    alpha = (rast_out[..., 3:4] > 0).float()
    alpha = dr.antialias(alpha, rast_out, verts_clip[None], faces_int32)

    # Y-flip to image convention (matches render_depth_and_alpha).
    return (
        rgb[0].flip(0),
        alpha[0, :, :, 0].flip(0),
        depth[0, :, :, 0].flip(0),
    )


def apply_pose_p3d(
    local_verts: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    s: torch.Tensor,
) -> torch.Tensor:
    """Apply a Sim(3) to points, staying in PyTorch3D space.

    Row-vector convention — ``posed = (verts * s) @ R + t``, ``@ R`` and NOT ``@ R.T``,
    matching ``refinement.apply_pose_to_gaussian``.  A transposed rotation is an easy
    mistake that barely moves IoU/Chamfer, so the composition is written once here and reused rather than
    re-transcribed: :func:`apply_pose_p3d_to_r3` is this plus the axis flip, and
    ``evaluation._object_camera_position`` is this without it.
    """
    return (local_verts * s) @ R + t


def apply_pose_p3d_to_r3(
    local_verts: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    s: torch.Tensor,
) -> torch.Tensor:
    """Apply pose (scale, rotate, translate) and P3D→R3 coordinate flip.

    Uses PyTorch3D row-vector convention: ``posed = (verts * s) @ R + t``,
    matching ``apply_pose_to_gaussian`` in ``refinement.py``.

    Parameters
    ----------
    local_verts : torch.Tensor
        Vertices ``(V, 3)`` in local object space ``[-0.5, 0.5]³``.
    R : torch.Tensor
        Rotation matrix ``(3, 3)``, row-vector convention (``p @ R``).
    t : torch.Tensor
        Translation ``(3,)``.
    s : torch.Tensor
        Scale ``(3,)`` or ``(1,)``.

    Returns
    -------
    torch.Tensor
        Posed vertices ``(V, 3)`` in R3 camera space.
    """
    posed_r3 = apply_pose_p3d(local_verts, R, t, s).clone()
    posed_r3[..., :2] *= -1
    return posed_r3


__all__ = [
    "PyTorch3DMeshRenderer",
    "diffmc_mesh",
    "project_to_clip",
    "render_depth_and_alpha",
    "apply_pose_p3d_to_r3",
]
