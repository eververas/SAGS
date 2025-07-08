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

import os
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
from functools import reduce
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

from pytorch3d.ops import knn_points as KNN
from pytorch3d.ops import knn_gather

from torch_geometric.utils import subgraph
from torch_scatter import scatter_mean, scatter_sum, scatter_max

from .SimpleConv import DiffAggregator

import tinycudann as tcnn


def NormalEstimation(pos, lengths=None, gauss=False, k=10, max_dist= None):
    if lengths!=None:
        mask = (torch.arange(pos.shape[1], device=pos.device)[None] >= lengths[:, None])
    
    dists, idx, nn = KNN(pos, pos, lengths1=lengths, lengths2=lengths, K=k, return_nn=True) #[B,N,K,3]
    u, s, v = torch.pca_lowrank(nn.cpu(), q=3, center=True)
    normals = v[:,:,:,-1].to(pos.device)
    curv  = (s[:,:,-1]/s.sum(-1)).unsqueeze(-1).to(pos.device)
    if lengths!= None:
        normals[mask]= 0.0
        curv[mask] =0.0
    if gauss:
        if max_dist!=None:
            gauss_weights = torch.exp(-2*dists/(max_dist+1e-10))
        else:
            gauss_weights = torch.exp(-2*dists/(dists[...,-1:]+1e-10))
        gauss_weights = gauss_weights/(gauss_weights.sum(-1,keepdim=True)+1e-10)
        if lengths!= None:
            gauss_weights[mask]= 0.0
        mean_curv = (gauss_weights.unsqueeze(-1)*knn_gather(curv, idx)).sum(-2)
        roughness = torch.abs(curv - mean_curv)
        corr = ((knn_gather(curv, idx)) - curv.unsqueeze(-1).repeat(1,1,k,1))
        dev = torch.sqrt((gauss_weights.unsqueeze(-1)*(corr**2)).sum(-2))
            
    return normals, curv, roughness

def adaptive_midpoint_interpolate(sparse_pts, mask, low_k=10, high_k=2):
    # sparse_pts: (n, 3)
    up_rate = 2 
    pts_num = sparse_pts.shape[0]
    up_pts_num = int(pts_num * 2)
    k = int(2 * 2)
    # (n, k, 3)
    # Low Curv Points 
    self_pairs = torch.stack([torch.arange(sparse_pts.shape[0]), torch.arange(sparse_pts.shape[0])],-1).to(sparse_pts.device)
    knn_idx = KNN(sparse_pts[None], sparse_pts[None] , K = max(10, low_k)).idx
    mid_pts_pairs = torch.stack([knn_idx[:, mask, 0:1].repeat(1,1, low_k) , knn_idx[:, mask]], -1).view(-1, 2)
    all_pts_pairs = torch.stack([knn_idx[:, ~mask, 0:1].repeat(1,1, high_k) , knn_idx[:, ~mask, 1:1+high_k]], -1).view(-1, 2)
    mid_pts_pairs = torch.cat([self_pairs, mid_pts_pairs, all_pts_pairs], 0)
    # Discard repeated pairs
    mid_pts_pairs = mid_pts_pairs.sort(dim=-1).values.unique(dim=0)                        
    mid_pts = (sparse_pts[mid_pts_pairs[:, 0]] + sparse_pts[mid_pts_pairs[:, 1]]) / 2.0
    return mid_pts, mid_pts_pairs

def midpoint_interpolate(sparse_pts, k = 4):
    # sparse_pts: (b, 3, n)
    k = int(k)
    # (b, 3, n, k)
    knn_idx = KNN(sparse_pts[None], sparse_pts[None], K=k).idx
    mid_pts_pairs = torch.stack([knn_idx[...,0:1].repeat(1,1,k) , knn_idx], -1).view(-1, 2)
    mid_pts_pairs = mid_pts_pairs.sort(dim=-1).values.unique(dim=0)
    mid_pts = (sparse_pts[mid_pts_pairs[:, 0]] + sparse_pts[mid_pts_pairs[:, 1]])/ 2.0 
    return mid_pts, mid_pts_pairs


class AffineLayer(nn.Module):
    def __init__(self, in_dim=32):
        super().__init__()
        self.affine_alpha = nn.Parameter(torch.ones([1, in_dim]))
        self.affine_beta  = nn.Parameter(torch.zeros([1, in_dim]))

    def forward(self, feats):
        return self.affine_alpha * feats + self.affine_beta

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, 
                 feat_dim: int=32, 
                 knn_neighbors : int=10,
                 voxel_size: float=0.01,
                 update_depth: int=3, 
                 update_init_factor: int=16,
                 update_hierachy_factor: int=4,
                 upsampling_factors: list[int]=None):
        
        self._xyz = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom_grow = torch.empty(0)
        self.denom_prune = torch.empty(0)
        self.opacity_accum = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self.feat_dim = feat_dim
        self.voxel_size = voxel_size
        self.knn_neighbors = knn_neighbors
        self.n_offsets = 1
        self.upsampling_factors = upsampling_factors or [10, 2] 
        
        # for voxel-based densification 
        self.update_depth = update_depth
        self.update_init_factor = update_init_factor
        self.update_hierachy_factor = update_hierachy_factor

        self._feats = torch.empty(0)
        self._neural_scaling = torch.empty(0)
        self._neural_rotation = torch.empty(0)
        self._neural_xyz = torch.empty(0)
        self._neural_opacity = torch.empty(0)
        self._feats_aggr = torch.empty(0)

        in_dim = self.feat_dim
        hidden_dim = self.feat_dim

        pos_enc_n_levels = 16
        pos_enc_n_feats_per_level = 2
        pos_enc_dim = pos_enc_n_levels * pos_enc_n_feats_per_level

        global_feat_dim = in_dim
        agg_in_dim = in_dim + pos_enc_dim + global_feat_dim

        self.mlp_opacity = tcnn.Network(
            n_input_dims=(agg_in_dim + 3 + 1),
            n_output_dims=self.n_offsets,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Tanh",
                "n_neurons": hidden_dim,
                "n_hidden_layers": 2,
            },
        ).cuda()
        self.mlp_color = tcnn.Network(
            n_input_dims=(agg_in_dim + 3 + 1),
            n_output_dims=self.n_offsets * 3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": hidden_dim,
                "n_hidden_layers": 2,
            },
        ).cuda()
        self.mlp_cov = tcnn.Network(
            n_input_dims=(agg_in_dim + 3 + 1),
            n_output_dims=self.n_offsets * 7,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": 2,
            },
        ).cuda()
        self.hash_grid = tcnn.Encoding(
            n_input_dims = 3,
            encoding_config = {
                "otype": "HashGrid",
                "n_levels": pos_enc_n_levels,
                "n_features_per_level": pos_enc_n_feats_per_level,
                "log2_hashmap_size": 15,
                "base_resolution": 16,
                "per_level_scale": 1.447,
            },
        ).cuda()
        self.mlp_offsets = tcnn.Network(
            n_input_dims = agg_in_dim,
            n_output_dims = self.n_offsets * 3,
            network_config = {
                "otype": "FullyFusedMLP",
                "activation": "None",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        ).cuda()
        self.affine_layer = AffineLayer(in_dim=agg_in_dim).cuda()
        self.message_aggr = DiffAggregator(aggr='max').cuda()

        self.setup_functions()

    def create_knn_graph(self):
        self.neighbors = KNN(self.get_xyz[None], self.get_xyz[None], K=self.knn_neighbors).idx[0]
        self._edge_index = torch.stack([
            self.neighbors[:, :1].repeat(1, self.knn_neighbors).reshape(-1),
            self.neighbors[:, ].reshape(-1)
        ], 0)
        
    def get_edge_index_for_subgraph(self, edge_index, mask):
        edge_index_subgraph, _ = subgraph(mask, edge_index, relabel_nodes=True)
        return edge_index_subgraph
    
    def aggreagate_feats(self, xyz, feats, edge_index):
        
        # difference aggreagation
        agg_feats = self.message_aggr(feats, edge_index)
        # distance-based mean aggreagation
        with torch.no_grad():
            dists = torch.sqrt(torch.sum((xyz[edge_index[1]] - xyz[edge_index[0]])**2, dim=1))
            dists[dists==0] = torch.inf
            dists_sum = scatter_sum(1 / (dists + 1e-8), edge_index[0], dim=0)
            weights = (1 / (dists + 1e-8)) / (dists_sum[edge_index[0]] + 1e-8)
            weights[torch.isnan(weights)] = 0
        agg_feats = scatter_mean(agg_feats[edge_index[1]] * weights[:, None], edge_index[0], dim=0)
        agg_feats = feats + 0.5 * agg_feats
        # global feature
        global_feat = torch.max(feats, dim=0)[0].repeat(agg_feats.shape[0], 1)
        # positional encoding feature
        xyz_enc = self.hash_grid(xyz)
        # cat features
        cat_feats = torch.cat([xyz_enc, agg_feats, global_feat], -1)
        # apply affine transformation on features
        out_feats = self.affine_layer(cat_feats)

        return out_feats

    def aggregate_feats_for_subgraph(self, mask):
        xyz = self.get_xyz[mask]
        feats = self.get_feats[mask]
        edge_index = self.get_edge_index 
        edge_index_subg = self.get_edge_index_for_subgraph(edge_index, mask) 
        out_feats = self.aggreagate_feats(xyz, feats, edge_index_subg)
        return out_feats

    def aggregate_feats_for_scene(self):
        xyz = self.get_xyz
        feats = self.get_feats
        edge_index = self.get_edge_index 
        out_feats = self.aggreagate_feats(xyz, feats, edge_index)
        self._feats_aggr = out_feats

    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        self.mlp_offsets.eval()
        self.hash_grid.eval()
        self.affine_layer.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        self.mlp_offsets.train()
        self.hash_grid.train()
        self.affine_layer.train()

    def capture(self):
        return (
            self._xyz,
            self._feats,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom_grow,
            self.denom_prune,
            self.opacity_accum,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self._xyz, 
        self._feats,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom_grow,
        denom_prune,
        opacity_accum,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom_grow = denom_grow
        self.denom_prune = denom_prune
        self.opacity_accum = opacity_accum
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_feats(self):
        return self._feats
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_neural_opacity(self):
        return self._neural_opacity
    
    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity

    @property
    def get_color_mlp(self):
        return self.mlp_color
    
    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_offsets_mlp(self):
        return self.mlp_offsets
    
    @property
    def get_neural_scaling(self):
        return self._neural_scaling
    
    @property
    def get_neural_rotation(self):
        return self._neural_rotation
    
    @property
    def get_neural_xyz(self):
        return self._neural_xyz
    
    @property
    def get_edge_index(self):
        return self._edge_index
    
    @property
    def get_feats_aggr(self):
        return self._feats_aggr

    @property
    def is_training(self):
        return self.mlp_cov.training

    def update_gaussians_neural_attributes(self, opacity, scaling, rotation, xyz, visible_mask):
        self._neural_opacity[visible_mask] = opacity.clone().detach()
        self._neural_scaling[visible_mask] = scaling.clone().detach()
        self._neural_rotation[visible_mask] = rotation.clone().detach()
        self._neural_xyz[visible_mask] = xyz.clone().detach()

    def compute_neural_xyz(self):
        xyz = self.get_xyz
        self.aggregate_feats_for_scene()
        xyz_feat = self._feats_aggr 
        offsets = self.get_offsets_mlp(xyz_feat).view([-1, 3])
        scales_pos = self.get_scaling[:, :3]
        offsets = offsets * scales_pos
        xyz = xyz + offsets
        return xyz
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def voxelize_sample(self, data = None, voxel_size = 0.01):
        np.random.shuffle(data)
        data = np.unique(np.round(data / voxel_size), axis=0) * voxel_size
        return data

    def curvature_based_pointcloud_upsampling(self, points, low_k=10, high_k=2):
        points = torch.from_numpy(points).cuda()
        # estimate normals
        _, curv, _  = NormalEstimation(points[None], lengths=None, gauss=True, k=10, max_dist=None)
        # create low curvature mask
        curv = curv.squeeze()
        curv_norm = (curv - curv.mean()) / (curv.std())
        mask = curv_norm < (curv_norm.mean() - curv_norm.std())
        print('Upsampling Total:' , mask.sum())
        # upsample
        points, mid_pts_pairs = adaptive_midpoint_interpolate(points, mask, low_k=low_k, high_k=high_k)
        # points, mid_pts_pairs = midpoint_interpolate(points, k = 4)
        points = points.cpu().numpy()
        return points

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        points = pcd.points

        if self.voxel_size <= 0:
            init_points = torch.tensor(points).float().cuda()
            init_dist = distCUDA2(init_points).float().cuda()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            del init_points
            torch.cuda.empty_cache()
        print(f'Initial voxel_size: {self.voxel_size}')

        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())

        # voxelize pointcloud
        points = self.voxelize_sample(points, voxel_size = self.voxel_size)
        # upsample pointcloud based on estimated curvature
        low_curv_ups_factor = self.upsampling_factors[0]
        high_curv_ups_factor = self.upsampling_factors[1]
        points = self.curvature_based_pointcloud_upsampling(points, low_k=low_curv_ups_factor, high_k=high_curv_ups_factor)

        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        feats = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 6)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._feats = nn.Parameter(feats.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        device = self.max_radii2D.device
        self._neural_scaling = torch.zeros((self.get_xyz.shape[0], 3)).float().cuda() 
        self._neural_rotation = torch.zeros((self.get_xyz.shape[0], 4)).float().cuda() 
        self._neural_xyz = torch.zeros((self.get_xyz.shape[0], 3)).float().cuda() 
        self._neural_opacity = torch.zeros((self.get_xyz.shape[0], 1)).float().cuda() 

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom_grow = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom_prune = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.opacity_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            # ---------------------------------------------------------------------------------
            {'params': [self._feats], 'lr': training_args.feats_lr, "name": "feats"},
            {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
            {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
            {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
            {'params': self.mlp_offsets.parameters(), 'lr': training_args.mlp_offsets_lr_init, "name": "mlp_offsets"},
            {'params': self.hash_grid.parameters(), 'lr': training_args.hash_grid_lr_init, "name": "hash_grid"},
            {'params': self.affine_layer.parameters(), 'lr': training_args.affine_layer_lr_init, "name": "affine_layer"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args           = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.mlp_opacity_scheduler_args   = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)
        self.mlp_color_scheduler_args     = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
        self.mlp_cov_scheduler_args       = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)
        self.mlp_offsets_scheduler_args  = get_expon_lr_func(lr_init=training_args.mlp_offsets_lr_init,
                                                    lr_final=training_args.mlp_offsets_lr_final,
                                                    lr_delay_mult=training_args.mlp_offsets_lr_delay_mult,
                                                    max_steps=training_args.mlp_offsets_lr_max_steps)
        self.hash_grid_scheduler_args    = get_expon_lr_func(lr_init=training_args.hash_grid_lr_init,
                                                    lr_final=training_args.hash_grid_lr_final,
                                                    lr_delay_mult=training_args.hash_grid_lr_delay_mult,
                                                    max_steps=training_args.hash_grid_lr_max_steps)
        self.affine_layer_scheduler_args = get_expon_lr_func(lr_init=training_args.affine_layer_lr_init,
                                                    lr_final=training_args.affine_layer_lr_final,
                                                    lr_delay_mult=training_args.affine_layer_lr_delay_mult,
                                                    max_steps=training_args.affine_layer_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_offsets":
                lr = self.mlp_offsets_scheduler_args(iteration)
                param_group['lr'] = lr  
            if param_group["name"] == "hash_grid":
                lr = self.hash_grid_scheduler_args(iteration)
                param_group['lr'] = lr  
            if param_group["name"] == "affine_layer":
                lr = self.affine_layer_scheduler_args(iteration)
                param_group['lr'] = lr 

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._feats.shape[1]):
            l.append('feats_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        feats = self._feats.detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, feats, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        # xyz_feats
        xyz_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("feats")]
        xyz_feat_names = sorted(xyz_feat_names, key = lambda x: int(x.split('_')[-1]))
        xyz_feats = np.zeros((xyz.shape[0], len(xyz_feat_names)))
        for idx, attr_name in enumerate(xyz_feat_names):
            xyz_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._feats = nn.Parameter(torch.tensor(xyz_feats, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group['name'] or 'conv' in group['name'] or 'hash' in group['name'] or 'affine' in group['name']:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group['name'] or 'conv' in group['name'] or 'hash' in group['name'] or 'affine' in group['name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def mask_neural_attributes(self, valid_points_mask):
        self._neural_scaling = self._neural_scaling[valid_points_mask]
        self._neural_rotation = self._neural_rotation[valid_points_mask]
        self._neural_xyz = self._neural_xyz[valid_points_mask]
        self._neural_opacity = self._neural_opacity[valid_points_mask]

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._feats = optimizable_tensors["feats"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom_grow = self.denom_grow[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.denom_prune = self.denom_prune[valid_points_mask]
        self.opacity_accum = self.opacity_accum[valid_points_mask]

        self.mask_neural_attributes(valid_points_mask)

    def densification_postfix(self, new_xyz, new_feats, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "feats": new_feats,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._feats = optimizable_tensors["feats"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom_grow = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.denom_prune = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.opacity_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

    def densification_postfix_neural_attributes(self, selected_pts_mask=None, N=1):
        shape_dif = self.get_xyz.shape[0] - self._neural_opacity.shape[0]
        new_neural_opacity = torch.zeros([shape_dif, 1], dtype=torch.int32, device="cuda")
        new_neural_scaling = torch.zeros([shape_dif, 3], dtype=torch.int32, device="cuda")
        new_neural_rotation = torch.zeros([shape_dif, 4], dtype=torch.int32, device="cuda")
        new_neural_xyz = torch.zeros([shape_dif, 3], dtype=torch.int32, device="cuda")
        if selected_pts_mask is not None:
            new_neural_opacity = self._neural_opacity[selected_pts_mask].repeat(N,1)
            new_neural_scaling = self._neural_scaling[selected_pts_mask].repeat(N,1)
            new_neural_rotation = self._neural_rotation[selected_pts_mask].repeat(N,1)
            new_neural_xyz = self._neural_xyz[selected_pts_mask].repeat(N,1)

        self._neural_opacity = torch.cat([self._neural_opacity, new_neural_opacity], dim=0)
        self._neural_scaling = torch.cat([self._neural_scaling, new_neural_scaling], dim=0)
        self._neural_rotation = torch.cat([self._neural_rotation, new_neural_rotation], dim=0)
        self._neural_xyz = torch.cat([self._neural_xyz, new_neural_xyz], dim=0)

    def xyz_growing(self, grads, threshold, xyz_mask):
        ## 
        init_length = self.get_xyz.shape[0] * self.n_offsets
        for i in range(self.update_depth):
            # update threshold
            cur_threshold = threshold * ((self.update_hierachy_factor // 2)**i)
            # mask from grad threshold
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, xyz_mask)
            # random pick
            rand_mask = torch.rand_like(candidate_mask.float()) > (0.5**(i+1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)
            
            length_inc = self.get_xyz.shape[0]*self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)

            # all_xyz = self.get_neural_xyz
            all_xyz = self.compute_neural_xyz()
            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            cur_size = self.voxel_size * size_factor
            grid_coords = torch.round(self.get_xyz / cur_size).int()
            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()
            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)

            ## split data for reducing peak memory calling
            use_chunk = True
            if use_chunk:
                chunk_size = 4096
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                remove_duplicates_list = []
                for i in range(max_iters):
                    cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                    remove_duplicates_list.append(cur_remove_duplicates)                
                remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else:
                remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)
            remove_duplicates = ~remove_duplicates
            candidate_xyz = selected_grid_coords_unique[remove_duplicates]*cur_size

            if candidate_xyz.shape[0] > 0:
                new_scaling = torch.ones_like(candidate_xyz).repeat([1,2]).float().cuda() * cur_size 
                new_scaling = torch.log(new_scaling)
                new_rotation = torch.zeros([candidate_xyz.shape[0], 4], device=candidate_xyz.device).float()
                new_rotation[:,0] = 1.0
                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_xyz.shape[0], 1), dtype=torch.float, device="cuda"))
                new_feats = self.get_feats.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]
                new_feats = scatter_max(new_feats, inverse_indices.unsqueeze(1).expand(-1, new_feats.size(1)), dim=0)[0][remove_duplicates]
                d = {
                    "xyz": candidate_xyz,
                    "feats": new_feats,
                    "opacity": new_opacities,
                    "scaling" : new_scaling,
                    "rotation" : new_rotation
                }
                
                self.denom_prune = torch.cat([self.denom_prune, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                self.opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)

                torch.cuda.empty_cache()
                
                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._xyz = optimizable_tensors["xyz"]
                self._feats = optimizable_tensors["feats"]
                self._opacity = optimizable_tensors["opacity"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]

            self.create_knn_graph()

    def densify_and_prune_voxel_based(self, max_grad, min_opacity, success_threshold, check_interval, grow=True, prune=True):
        if grow:
            # # adding anchors
            grads = self.xyz_gradient_accum / self.denom_grow 
            grads[grads.isnan()] = 0.0
            grads_norm = torch.norm(grads, dim=-1)
            xyz_mask = (self.denom_grow > check_interval*success_threshold*0.5).squeeze(dim=1)
            
            self.xyz_growing(grads_norm, max_grad, xyz_mask)
            
            # update denom_grow
            shape_dif = self.get_xyz.shape[0] - self.denom_grow.shape[0]
            self.denom_grow[xyz_mask] = 0
            padding_offset_demon = torch.zeros([shape_dif, 1], dtype=torch.int32, device=self.denom_grow.device)
            self.denom_grow = torch.cat([self.denom_grow, padding_offset_demon], dim=0)
            self.xyz_gradient_accum[xyz_mask] = 0
            padding_xyz_gradient_accum = torch.zeros([shape_dif, 1], dtype=torch.int32, device=self.denom_grow.device)
            self.xyz_gradient_accum = torch.cat([self.xyz_gradient_accum, padding_xyz_gradient_accum], dim=0)

            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

            self.densification_postfix_neural_attributes()

        if prune:
            # prune anchors
            prune_mask = (self.opacity_accum < min_opacity * self.denom_prune).squeeze(dim=1)
            xyz_mask = (self.denom_prune > check_interval * success_threshold).squeeze(dim=1) 
            prune_mask = torch.logical_and(prune_mask, xyz_mask) 
            
            if xyz_mask.sum() > 0:
                self.opacity_accum[xyz_mask] = torch.zeros([xyz_mask.sum(), 1], device='cuda').float()
                self.denom_prune[xyz_mask] = torch.zeros([xyz_mask.sum(), 1], device='cuda').float() 
            if prune_mask.shape[0] > 0:
                self.prune_points(prune_mask)

            self.create_knn_graph()
        
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def add_densification_stats(self, viewspace_point_tensor, update_filter, visibility_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[visibility_filter, :2], dim=-1, keepdim=True)
        self.denom_grow[update_filter] += 1
        self.denom_prune[update_filter] += 1
        # update opacity stats
        temp_opacity = self.get_neural_opacity.clone().detach()
        temp_opacity[temp_opacity < 0] = 0
        self.opacity_accum[update_filter] += temp_opacity[update_filter]

    def save_mlp_checkpoints(self, path):
        mkdir_p(os.path.dirname(path))        
        torch.save({
            'mlp_opacity' : self.mlp_opacity.state_dict(),
            'mlp_color'   : self.mlp_color.state_dict(),
            'mlp_cov'     : self.mlp_cov.state_dict(),
            'mlp_offsets' : self.mlp_offsets.state_dict(),
            'hash_grid'   : self.hash_grid.state_dict(),
            'affine_layer': self.affine_layer.state_dict(),
        }, path)

    def load_mlp_checkpoints(self,path):
        checkpoint = torch.load(path)
        self.mlp_opacity.load_state_dict(checkpoint['mlp_opacity'])
        self.mlp_color.load_state_dict(checkpoint['mlp_color'])
        self.mlp_cov.load_state_dict(checkpoint['mlp_cov'])
        self.mlp_offsets.load_state_dict(checkpoint['mlp_offsets'])
        self.hash_grid.load_state_dict(checkpoint['hash_grid'])
        self.affine_layer.load_state_dict(checkpoint['affine_layer'])