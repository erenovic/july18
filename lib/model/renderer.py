

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# from nerf import NeRF
from .sdf_network import SDFNetwork
from .variance_net import SingleVarianceNetwork
from .color_net import ColorNetwork
from .feature_net import ConvBnReLU, FeatureNet
from .sparse_conv_net import SparseCostRegNet
from .sparseconvnet import SparseConvNetTensor

from .utils import generate_grid, back_project_sparse_type


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nansa
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class Renderer(nn.Module):
    '''
    Taken from official NeuS repository with small modifications
    '''
    def __init__(self, config, device=torch.device('cuda')):
        super(Renderer, self).__init__()

        self.n_samples = config.renderer["n_samples"]
        self.n_importance = config.renderer["n_importance"]
        self.n_outside = config.renderer["n_outside"]
        self.up_sample_steps = config.renderer["up_sample_steps"] 
        self.perturb = config.renderer["perturb"]
        
        self.nerf = None

        self.feat_extractor = FeatureNet()
        self.sdf_net = SDFNetwork(config.sdf_net)
        self.var_net = SingleVarianceNetwork(config.var_net)
        self.color_net = ColorNetwork(config.color_net)

        # Use fused pyramid feature maps are very important
        self.compress_layer = ConvBnReLU(
            ch_in=32, ch_out=16, kernel=3, stride=1, padding=1
        )
        self.sparse_costreg_net = SparseCostRegNet(d_in=16, d_out=8)

        self.register_buffer(
            'vol_dims', 
            torch.tensor([96, 96, 96]).to(device), 
            persistent=False
        )
        self.voxel_size = 2 / (self.vol_dims[0] - 1)

    def forward(self, pts, rays_o, rays_d, z_vals, sample_dist, imgs=None,
                bg_rgb=None, ref_poses=None, cos_anneal_ratio=0.0):
        '''
        Rendering along the rays provided.
        :rays_o: origin point of the rays,
        :rays_d: direction vector of rays,
        '''
        
        features = self.compute_features(imgs)

        img_size = imgs.shape[-2:]
        conditional_volume = self.compute_volume(
            features, ref_poses['partial_vol_origin'].to(features.device), 
            ref_poses['proj_mats'].to(features.device), img_size
        )
        
        breakpoint()

        B = len(rays_o)

        bg_alpha, bg_sampled_color= None, None

        # Up sample
        if self.n_importance > 0:
            with torch.no_grad():
                # Calculate the SDF values for each point along the rays
                sdf = self.sdf_net.sdf(pts.reshape(-1, 3)).reshape(B, self.n_samples)

                # Iterative importance sampling steps
                for i in range(self.up_sample_steps):
                    # At each iteration we have additional samples.
                    new_z_vals = self.up_sample(
                        rays_o, rays_d, z_vals, sdf, self.n_importance // self.up_sample_steps, 64 * 2**i
                    )
                    z_vals, sdf = self.cat_z_vals(
                        rays_o, rays_d, z_vals, new_z_vals, sdf, last=(i + 1 == self.up_sample_steps)
                    )

            n_samples = self.n_samples + self.n_importance

        # # I think this model is not necessary since we always have masks!
        # # Background model
        # if self.n_outside > 0:
        #     z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1)
        #     z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)
        #     ret_outside = self.render_core_outside(rays_o, rays_d, z_vals_feed, sample_dist, self.nerf)

        #     background_sampled_color = ret_outside['sampled_color']
        #     background_alpha = ret_outside['alpha']

        # Render core
        ret_fine = self.render_core(
            rays_o, rays_d, z_vals, sample_dist,
            bg_rgb=bg_rgb, bg_alpha=bg_alpha, bg_sampled_color=bg_sampled_color,
            cos_anneal_ratio=cos_anneal_ratio
        )

        weights = ret_fine['weights']
        weights_sum = weights.sum(dim=-1, keepdim=True)
        weight_max = torch.max(weights, dim=-1, keepdim=True)[0]
        
        return {
            'color_fine': ret_fine['color'],
            's_val': ret_fine['s_val'].reshape(B, n_samples).mean(dim=-1, keepdim=True),
            'cdf_fine': ret_fine['cdf'],
            'weight_sum': weights_sum,
            'weight_max': weight_max,
            'gradients': ret_fine['gradients'],
            'weights': weights,
            'gradient_error': ret_fine['gradient_error'],
            'inside_sphere': ret_fine['inside_sphere']
        }
    

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        # Calculate the norm of points 
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        # Check if the points are inside the unit sphere
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        
        # Previous sample's SDF value and next sample's SDF value
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        # Previous sample's distance along the ray and next sample's distance along the ray
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples
    

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            new_sdf = self.sdf_net.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf
    

    def render_core(self, rays_o, rays_d, z_vals, sample_dist,
                    bg_alpha=None, bg_sampled_color=None, bg_rgb=None,
                    cos_anneal_ratio=0.0):
        
        B, N = z_vals.shape

        # Section length for each consequtive distance along rays
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints are the points to consider for SDF
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        # Calculate the SDF for points
        sdf_nn_output = self.sdf_net(pts)
        sdf = sdf_nn_output[:, :1]
        feature_vector = sdf_nn_output[:, 1:]

        gradients = self.sdf_net.gradient(pts).squeeze()

        # Get color values for points
        sampled_color = self.color_net(pts, gradients, dirs, feature_vector).reshape(B, N, 3)

        inv_s = self.var_net(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter
        inv_s = inv_s.expand(B * N, 1)

        true_cos = (dirs * gradients).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(B, N).clip(0.0, 1.0)

        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(B, N)
        inside_sphere = (pts_norm < 1.0).float().detach()
        relax_inside_sphere = (pts_norm < 1.2).float().detach()

        # I think we don't need this part, bg_alpha is always None for us!
        # Render with background
        # if bg_alpha is not None:
        #     alpha = alpha * inside_sphere + background_alpha[:, :n_samples] * (1.0 - inside_sphere)
        #     alpha = torch.cat([alpha, background_alpha[:, n_samples:]], dim=-1)
        #     sampled_color = sampled_color * inside_sphere[:, :, None] +\
        #                     background_sampled_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None]
        #     sampled_color = torch.cat([sampled_color, background_sampled_color[:, n_samples:]], dim=1)

        weights = alpha * torch.cumprod(torch.cat([torch.ones([B, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        weights_sum = weights.sum(dim=-1, keepdim=True)

        color = (sampled_color * weights[:, :, None]).sum(dim=1)
        if bg_rgb is not None:    # Fixed background, usually black
            color = color + bg_rgb * (1.0 - weights_sum)

        # Eikonal loss
        gradient_error = (torch.linalg.norm(gradients.reshape(B, N, 3), ord=2,
                                            dim=-1) - 1.0) ** 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

        return {
            'color': color,
            'sdf': sdf,
            'dists': dists,
            'gradients': gradients.reshape(B, N, 3),
            's_val': 1.0 / inv_s,
            'mid_z_vals': mid_z_vals,
            'weights': weights,
            'cdf': c.reshape(B, N),
            'gradient_error': gradient_error,
            'inside_sphere': inside_sphere
        }
    

    def compute_features(self, imgs, lod=0):
        """
        get feature maps of all conditional images
        :param imgs:
        :return:
        """
        pyramid_feature_maps = self.feat_extractor(imgs[0])

        # * the pyramid features are very important, if only use the coarst features, hard to optimize
        fused_feature_maps = torch.cat([
            F.interpolate(pyramid_feature_maps[0], scale_factor=4, mode='bilinear', align_corners=True),
            F.interpolate(pyramid_feature_maps[1], scale_factor=2, mode='bilinear', align_corners=True),
            pyramid_feature_maps[2]
        ], dim=1)

        return fused_feature_maps
    

    def aggregate_multiview_features(self, multiview_features, multiview_masks):
        """
        aggregate mutli-view features by compute their cost variance
        :param multiview_features: (num of voxels, num_of_views, c)
        :param multiview_masks: (num of voxels, num_of_views)
        :return:
        """
        num_pts, n_views, C = multiview_features.shape

        counts = torch.sum(multiview_masks, dim=1, keepdim=False)  # [num_pts]

        assert torch.all(counts > 0)  # the point is visible for at least 1 view

        volume_sum = torch.sum(multiview_features, dim=1, keepdim=False)  # [num_pts, C]
        volume_sq_sum = torch.sum(multiview_features ** 2, dim=1, keepdim=False)
        del multiview_features

        counts = 1. / (counts + 1e-5)
        costvar = volume_sq_sum * counts[:, None] - (volume_sum * counts[:, None]) ** 2
        costvar_mean = torch.cat([costvar, volume_sum * counts[:, None]], dim=1)
        del volume_sum, volume_sq_sum, counts

        return costvar_mean


    def compute_volume(self, feature_maps, partial_vol_origin, 
                       proj_mats, img_size
                       ):
        """
        From SparseNeuS (https://github.com/xxlong0/SparseNeuS/blob/main/models/sparse_sdf_network.py)

        :param feature_maps: pyramid features (B,V,C0+C1+C2,H,W) fused pyramid features
        :param partial_vol_origin: [B, 3]  the world coordinates of the volume origin (0,0,0)
        :param proj_mats: projection matrix transform world pts into image space [B,V,4,4] suitable for original image size
        :param sizeH: the H of original image size
        :param sizeW: the W of original image size
        :param pre_coords: the coordinates of sparse volume from the prior lod
        :param pre_feats: the features of sparse volume from the prior lod
        :return:
        """
        sizeH, sizeW = img_size
        device = proj_mats.device
        bs = feature_maps.shape[0]
        N_views = feature_maps.shape[1]
        minimum_visible_views = np.min([1, N_views - 1])

        outputs = {}
        pts_samples = []

        # * use fused pyramid feature maps are very important
        if self.compress_layer is not None:
            feats = self.compress_layer(feature_maps)
        else:
            feats = feature_maps[0]
        feats = feats[:, None, :, :, :]  # [V, B, C, H, W]

        KRcam = proj_mats.permute(1, 0, 2, 3).contiguous()  # [V, B, 4, 4]
        interval = 1

        # ----generate new coords----
        coords = generate_grid(self.vol_dims, 1)[0]
        coords = coords.view(3, -1).to(device)  # [3, num_pts]
        up_coords = []
        for b in range(bs):
            up_coords.append(torch.cat([torch.ones(1, coords.shape[-1]).to(coords.device) * b, coords]))
        up_coords = torch.cat(up_coords, dim=1).permute(1, 0).contiguous()

        # * since we only estimate the geometry of input reference image at one time;
        # * mask the outside of the camera frustum
        frustum_mask = back_project_sparse_type(
            up_coords, partial_vol_origin, self.voxel_size,
            feats, KRcam, sizeH=sizeH, sizeW=sizeW, only_mask=True)  # [num_pts, n_views]
        frustum_mask = torch.sum(frustum_mask, dim=-1) > minimum_visible_views  # ! here should be large
        up_coords = up_coords[frustum_mask]  # [num_pts_valid, 4]

        # ----back project----
        multiview_features, multiview_masks = back_project_sparse_type(
            up_coords, partial_vol_origin, self.voxel_size, feats,
            KRcam, sizeH=sizeH, sizeW=sizeW)  # (num of voxels, num_of_views, c), (num of voxels, num_of_views)

        volume = self.aggregate_multiview_features(multiview_features, multiview_masks)

        del multiview_features, multiview_masks

        feat = volume

        # batch index is in the last position
        r_coords = up_coords[:, [1, 2, 3, 0]]

        print(r_coords.shape)
        print('HERE!')
        breakpoint()

        # sparse_feat = SparseTensor(feat, r_coords.to(
        #     torch.int32))  # - directly use sparse tensor to avoid point2voxel operations
        sparse_feat = SparseConvNetTensor(feat, metadata=None, spatial_size=None)

        feat = self.sparse_costreg_net(sparse_feat)

        dense_volume, valid_mask_volume = self.sparse_to_dense_volume(up_coords[:, 1:], feat, self.vol_dims, interval,
                                                                      device=None)  # [1, C/1, X, Y, Z]

        outputs['dense_volume_scale%d' % self.lod] = dense_volume
        outputs['valid_mask_volume_scale%d' % self.lod] = valid_mask_volume
        outputs['visible_mask_scale%d' % self.lod] = valid_mask_volume
        outputs['coords_scale%d' % self.lod] = generate_grid(self.vol_dims, interval).to(device)

        return outputs