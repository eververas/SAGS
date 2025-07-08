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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel

from einops import repeat


def generate_neural_gaussians(gaussians : GaussianModel, viewpoint_camera, visible_mask = None):

    # view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(gaussians.get_xyz.shape[0], dtype=torch.bool, device = gaussians.get_xyz.device)

    if torch.isnan(gaussians.get_feats).sum() > 0:
        print('Found nan feats')  

    # init attributes
    xyz = gaussians.get_xyz[visible_mask]
    feats = gaussians.aggregate_feats_for_subgraph(visible_mask)
    scales = gaussians.get_scaling[visible_mask]

    # camera properties for xyz
    ob_view = xyz - viewpoint_camera.camera_center
    ob_dist = ob_view.norm(dim=1, keepdim=True) 
    ob_view = ob_view / ob_dist

    # camera-dependent feats
    cat_feats_view = torch.cat([feats, ob_view, ob_dist], dim=1) # [N, c+3]

    # predict attributes
    opacity = gaussians.get_opacity_mlp(cat_feats_view)
    mask = opacity[:, 0] > 0
    opacity = opacity[mask]
    cat_feats_view = cat_feats_view[mask]
    feats = feats[mask]
    xyz = xyz[mask]
    scales = scales[mask]

    color = gaussians.get_color_mlp(cat_feats_view).reshape([-1, 3])
    scale_rot = gaussians.get_cov_mlp(cat_feats_view).reshape([-1, 7])
    offsets = gaussians.get_offsets_mlp(feats).view([-1, 3])

    # mask + float conversion fo tcnn outputs
    concatenated = torch.cat([scales, opacity, color, scale_rot], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=gaussians.n_offsets)
    concatenated_all = concatenated_repeated
    _, opacity, color, scale_rot = concatenated_all.split([6, 1, 3, 7], dim=-1)

    scales_pos = scales[:, :3]
    scales_cov = scales[:, 3:]

    # post-process cov
    scaling = scales_cov * torch.sigmoid(scale_rot[:, :3])
    rotation = gaussians.rotation_activation(scale_rot[:, 3:7])

    # post-process offsets 
    offsets = offsets * scales_pos
    xyz = xyz + offsets

    # combined visibility + opacity mask
    combined_mask = visible_mask.clone()
    combined_mask[visible_mask] = mask

    # update neural attributes
    if gaussians.is_training:
        gaussians.update_gaussians_neural_attributes(opacity, scaling, rotation, xyz, combined_mask)

    return xyz, opacity, color, scaling, rotation, combined_mask

def render(viewpoint_camera, gaussians : GaussianModel, pipe, bg_color : torch.Tensor, visible_mask = None, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    Background tensor (bg_color) must be on GPU!
    """
 
    # Get neural gaussian attributes
    xyz, opacity, color, scaling, rotation, combined_mask = generate_neural_gaussians(gaussians, viewpoint_camera, visible_mask)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = xyz
    means2D = screenspace_points
    shs = None
    colors_precomp = color
    opacities = opacity
    scales = scaling
    rotations = rotation
    cov3D_precomp = None

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacities,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "combined_mask": combined_mask,
            "radii": radii,
            "scaling": scaling}


def prefilter_voxel(viewpoint_camera, gaussians : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(gaussians.get_xyz, dtype=gaussians.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D = gaussians.get_xyz

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = gaussians.get_covariance(scaling_modifier)
    else:
        scales = gaussians.get_scaling
        rotations = gaussians.get_rotation

    radii_pure = rasterizer.visible_filter(means3D = means3D,
        scales = scales[:,:3],
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    return radii_pure > 0
