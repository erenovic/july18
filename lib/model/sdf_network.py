
import torch
import torch.nn as nn
import numpy as np

from .positional_encoding import PositionalEncoding


class SDFNetwork(nn.Module):
    def __init__(self, config):
        super(SDFNetwork, self).__init__()

        self.d_in = config["d_in"]
        self.d_out = config["d_out"]
        self.d_hidden = config["d_hidden"]
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
        
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if l == self.num_layers - 2:
                torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                torch.nn.init.constant_(lin.bias, -self.bias)
            elif self.multires > 0 and l == 0:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
            elif self.multires > 0 and l in self.skip_in:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
            else:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if self.weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs):
        inputs = inputs * self.scale
        if self.embed_fn is not None:
            inputs = self.embed_fn(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def sdf(self, x):
        return self.forward(x)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)