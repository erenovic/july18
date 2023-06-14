
import torch
import torch.nn as nn
import numpy as np

from .positional_encoding import PositionalEncoding
from .utils import grid_sample_3d


class LatentSDFLayer(nn.Module):
    def __init__(self, d_in=3, d_out=129, d_hidden=128, n_layers=4,
                 skip_in=(4,), multires=0, bias=0.5,
                 d_conditional_feature=8):
        super(LatentSDFLayer, self).__init__()

        self.d_conditional_feature = d_conditional_feature

        # concat latent code for ench layer input excepting the first layer and the last layer
        dims_in = [d_in] + [d_hidden + d_conditional_feature for _ in range(n_layers - 2)] + [d_hidden]
        dims_out = [d_hidden for _ in range(n_layers - 1)] + [d_out]

        self.num_layers = n_layers
        self.skip_in = skip_in

        for l in range(0, self.num_layers):
            if l in self.skip_in:
                in_dim = dims_in[l] + dims_in[0]
            else:
                in_dim = dims_in[l]

            out_dim = dims_out[l]

            lin = nn.Linear(in_dim, out_dim)

            if l == self.num_layers - 2:
                torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(in_dim), std=0.0001)
                torch.nn.init.constant_(lin.bias, -bias)
                # the channels for latent codes are set to 0
                torch.nn.init.constant_(lin.weight[:, -d_conditional_feature:], 0.0)
                torch.nn.init.constant_(lin.bias[-d_conditional_feature:], 0.0)

            elif multires > 0 and l == 0:  # the first layer
                torch.nn.init.constant_(lin.bias, 0.0)
                # * the channels for position embeddings are set to 0
                torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                # * the channels for the xyz coordinate (3 channels) for initialized by normal distribution
                torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
            elif multires > 0 and l in self.skip_in:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                # * the channels for position embeddings (and conditional_feature) are initialized to 0
                torch.nn.init.constant_(lin.weight[:, -(dims_in[0] - 3 + d_conditional_feature):], 0.0)
            else:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                # the channels for latent code are initialized to 0
                torch.nn.init.constant_(lin.weight[:, -d_conditional_feature:], 0.0)

            lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)
        

    def forward(self, inputs, latent):
        
        x = inputs
        for l in range(0, self.num_layers):
            lin = getattr(self, "lin" + str(l))

            # * due to the conditional bias, different from original neus version
            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            if 0 < l < self.num_layers - 1:
                x = torch.cat([x, latent], 1)

            x = lin(x)

            if l < self.num_layers - 1:
                x = self.activation(x)

        return x


class SDFNetwork(nn.Module):
    def __init__(self, config):
        super(SDFNetwork, self).__init__()

        self.d_in = config["d_in"]
        self.d_out = config["d_out"]
        self.d_hidden = config["d_hidden"]
        self.d_feature = config["d_feature"]
        self.n_layers = config["n_layers"]
        self.skip_in = config["skip_in"]
        self.multires = config["multires"]
        self.bias = config["bias"]
        self.scale = config["scale"]
        self.weight_norm = config["weight_norm"]
        
        dims = [self.d_in] + \
            [self.d_hidden for _ in range(self.n_layers)] + \
                [self.d_out]

        self.embed_fn = None

        if self.multires > 0:
            self.embed_fn = PositionalEncoding(**config["pos_encode"])
            input_ch = self.embed_fn.out_dim
            dims[0] = input_ch

        self.num_layers = len(dims)

        self.sdf_layer = LatentSDFLayer(
            d_in=3, d_out=self.d_hidden + 1,
            d_hidden=self.d_hidden,
            n_layers=self.n_layers,
            multires=self.multires,
            d_conditional_feature=self.d_feature,  # self.regnet_d_out
            skip_in=self.skip_in
        )

        self.activation = nn.Softplus(beta=100)


    def sdf(self, pts, conditional_volume):
        num_pts = pts.shape[0]
        device = pts.device
        pts_ = pts.clone()
        pts = pts.view(1, 1, 1, num_pts, 3)  # - should be in range (-1, 1)

        pts = torch.flip(pts, dims=[-1])

        sampled_feature = grid_sample_3d(conditional_volume, pts)  # [1, c, 1, 1, num_pts]
        sampled_feature = sampled_feature.view(-1, num_pts).permute(1, 0).contiguous().to(device)

        sdf_pts = self.sdf_layer(pts_, sampled_feature)

        outputs = {}
        outputs['sdf_pts'] = sdf_pts[:, :1]
        outputs['sdf_features_pts'] = sdf_pts[:, 1:]
        outputs['sampled_latent'] = sampled_feature

        return outputs

    def gradient(self, x, conditional_volume):
        """
        return the gradient of specific lod
        :param x:
        :param lod:
        :return:
        """
        x.requires_grad_(True)
        output = self.sdf(x, conditional_volume)
        y = output['sdf_pts']

        d_output = torch.ones_like(y, requires_grad=False, device=y.device)

        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)