#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import importlib
import math
import sys
import warnings
from pathlib import Path

import torch
from src.scene.gaussian_model import GaussianModel
from src.utils.sh_utils import RGB2SH

_raster_backend_name = None
_GaussianRasterizationSettings = None
_GaussianRasterizer = None
_repo_root = Path(__file__).resolve().parents[2]
_local_rasterizer_roots = {
    "diff_gaussian_rasterization": _repo_root / "src" / "submodules" / "gaussian-rasterization",
    "diff_gaussian_rasterization_sb": _repo_root / "src" / "submodules" / "gaussian-rasterization-sb",
}


def _ensure_local_rasterizer_path(module_name: str):
    local_root = _local_rasterizer_roots.get(module_name)
    if local_root is None:
        return
    if local_root.exists():
        local_root_s = str(local_root)
        if local_root_s not in sys.path:
            sys.path.insert(0, local_root_s)


def _load_rasterizer_backend(backend: str):
    global _raster_backend_name, _GaussianRasterizationSettings, _GaussianRasterizer
    backend = str(backend or "default").lower()
    if backend in ("default", "base", "gaussian-rasterization"):
        candidates = [("default", "diff_gaussian_rasterization")]
    elif backend in ("sb", "gaussian-rasterization-sb", "visibility_mask"):
        candidates = [
            ("sb", "diff_gaussian_rasterization_sb"),
            ("default", "diff_gaussian_rasterization"),
        ]
    else:
        warnings.warn(f"Unknown rasterizer backend '{backend}', fallback to default.")
        candidates = [("default", "diff_gaussian_rasterization")]

    import_errors = []
    for backend_name, module_name in candidates:
        try:
            _ensure_local_rasterizer_path(module_name)
            module = importlib.import_module(module_name)
            _GaussianRasterizationSettings = module.GaussianRasterizationSettings
            _GaussianRasterizer = module.GaussianRasterizer
            _raster_backend_name = backend_name
            if backend in ("sb", "gaussian-rasterization-sb", "visibility_mask") and backend_name != "sb":
                warnings.warn(
                    "Requested rasterizer backend 'sb' is unavailable, fallback to default backend."
                )
            return _raster_backend_name
        except Exception as exc:
            import_errors.append(f"{module_name}: {exc}")

    raise ImportError("Failed to import rasterizer backend. " + " | ".join(import_errors))


def set_rasterizer_backend(backend: str):
    backend = str(backend or "default").lower()
    if _GaussianRasterizer is not None and backend == _raster_backend_name:
        return _raster_backend_name
    return _load_rasterizer_backend(backend)


def _get_rasterizer_api():
    if _GaussianRasterizer is None or _GaussianRasterizationSettings is None:
        _load_rasterizer_backend("default")
    return _GaussianRasterizationSettings, _GaussianRasterizer


def _empty_semantics_like(rendered_image: torch.Tensor, semantics):
    sem_channels = int(semantics.shape[1]) if (semantics is not None and semantics.ndim == 2) else 1
    return torch.zeros(
        (sem_channels, rendered_image.shape[1], rendered_image.shape[2]),
        device=rendered_image.device,
        dtype=rendered_image.dtype,
    )


def _ensure_map_hw(x: torch.Tensor, rendered_image: torch.Tensor, fill_value: float = 0.0):
    if x is not None:
        if x.ndim == 2:
            return x.unsqueeze(0)
        if x.ndim == 3:
            return x
    return torch.full(
        (1, rendered_image.shape[1], rendered_image.shape[2]),
        fill_value=fill_value,
        device=rendered_image.device,
        dtype=rendered_image.dtype,
    )


def _rasterize_with_compat(rasterizer, raster_inputs, semantics, visibility_mask):
    attempts = []
    full_kwargs = dict(raster_inputs)
    full_kwargs["semantics"] = semantics
    if visibility_mask is not None:
        full_kwargs["visibility_mask"] = visibility_mask
    attempts.append(full_kwargs)

    no_vis_kwargs = dict(raster_inputs)
    no_vis_kwargs["semantics"] = semantics
    attempts.append(no_vis_kwargs)
    attempts.append(dict(raster_inputs))

    tried = set()
    last_type_error = None
    for kwargs in attempts:
        signature = tuple(sorted(kwargs.keys()))
        if signature in tried:
            continue
        tried.add(signature)
        try:
            outputs = rasterizer(**kwargs)
            if isinstance(outputs, list):
                outputs = tuple(outputs)
            if hasattr(outputs, "_fields") and hasattr(outputs, "__iter__"):
                outputs = tuple(outputs)
            if isinstance(outputs, tuple):
                if len(outputs) >= 5:
                    rendered_image, radii, depth, alpha, rendered_semantics = outputs[:5]
                    depth = _ensure_map_hw(depth, rendered_image, fill_value=0.0)
                    alpha = _ensure_map_hw(alpha, rendered_image, fill_value=1.0)
                    return rendered_image, radii, depth, alpha, rendered_semantics
                if len(outputs) == 4:
                    rendered_image, radii, depth, alpha = outputs
                    depth = _ensure_map_hw(depth, rendered_image, fill_value=0.0)
                    alpha = _ensure_map_hw(alpha, rendered_image, fill_value=1.0)
                    rendered_semantics = _empty_semantics_like(rendered_image, semantics)
                    return rendered_image, radii, depth, alpha, rendered_semantics
                if len(outputs) == 3:
                    rendered_image, radii, depth = outputs
                    depth = _ensure_map_hw(depth, rendered_image, fill_value=0.0)
                    alpha = _ensure_map_hw(None, rendered_image, fill_value=1.0)
                    rendered_semantics = _empty_semantics_like(rendered_image, semantics)
                    return rendered_image, radii, depth, alpha, rendered_semantics
                if len(outputs) == 2:
                    rendered_image, radii = outputs
                    depth = _ensure_map_hw(None, rendered_image, fill_value=0.0)
                    alpha = _ensure_map_hw(None, rendered_image, fill_value=1.0)
                    rendered_semantics = _empty_semantics_like(rendered_image, semantics)
                    return rendered_image, radii, depth, alpha, rendered_semantics
            raise RuntimeError("Unexpected rasterizer outputs format.")
        except TypeError as exc:
            last_type_error = exc
            continue
    if last_type_error is not None:
        raise last_type_error
    raise RuntimeError("Rasterizer invocation failed.")


def render(
    viewpoint_camera,
    pc: GaussianModel,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    deform=True,
    render_deformation=False,
    visibility_mask=None,
):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    GaussianRasterizationSettings, GaussianRasterizer = _get_rasterizer_api()
    raster_kwargs = dict(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(), # seems like a 4x4 matrix
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
    )
    if hasattr(GaussianRasterizationSettings, "_fields") and "debug" in GaussianRasterizationSettings._fields:
        raster_kwargs["debug"] = False
    raster_settings = GaussianRasterizationSettings(**raster_kwargs)

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    render_time = viewpoint_camera.time if viewpoint_camera.time is not None else getattr(pc, "current_time", None)
    try:
        xyz, scales, rots, opacity, shs, semantics = pc(deform, time=render_time)
    except TypeError:
        xyz, scales, rots, opacity, shs, semantics = pc(deform)
    if render_deformation:
        # set deformation as color
        mean_def = pc._deformation.get_mean_def(pc.get_xyz).abs()
        mean_def = mean_def / (torch.quantile(mean_def, 0.99)+1e-12)
        shs = RGB2SH(mean_def[:, None])
        opacity = opacity.clamp(0, 0.9)
    means2D = screenspace_points

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    raster_inputs = dict(
        means3D=xyz,
        means2D=means2D,
        shs=shs,
        colors_precomp=None,
        opacities=opacity,
        scales=scales,
        rotations=rots,
        cov3D_precomp=None,
    )
    vis_mask = None
    if visibility_mask is not None:
        vis_mask = visibility_mask.reshape(-1).to(device=xyz.device, dtype=torch.bool)
        if vis_mask.shape[0] != xyz.shape[0]:
            warnings.warn("visibility_mask size mismatch, ignoring visibility mask.")
            vis_mask = None

    rendered_image, radii, depth, alpha, rendered_semantics = _rasterize_with_compat(
        rasterizer=rasterizer,
        raster_inputs=raster_inputs,
        semantics=semantics,
        visibility_mask=vis_mask,
    )
    rendered_image = rendered_image.permute(1, 2, 0)
    depth = depth.squeeze(0)
    # spotlight light source model
    rendered_image = viewpoint_camera.spotlight_render(rendered_image, depth.detach())
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth":depth,
            "alpha": alpha.squeeze(0),
            "semantics": rendered_semantics}
