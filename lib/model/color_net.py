
import torch
import torch.nn as nn

from .positional_encoding import PositionalEncoding


class ColorNetwork(nn.Module):
    def __init__(self, config):
        super(ColorNetwork, self).__init__()

        self.d_feature = config["d_feature"]
        self.d_in = config["d_in"]
        self.d_out = config["d_out"]
        self.d_hidden = config["d_hidden"]
        self.n_layers = config["n_layers"]
        self.weight_norm = config["weight_norm"]
        self.multires_view = config["multires_view"]
        self.squeeze_out = config["squeeze_out"]

        dims = [self.d_in + self.d_feature] + \
            [self.d_hidden for _ in range(self.n_layers)] + \
                [self.d_out]

        self.embedview_fn = None
        if self.multires_view > 0:
            self.embedview_fn = PositionalEncoding(**config["pos_encode"])
            input_ch = self.embedview_fn.out_dim
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if self.weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        x = torch.cat(
            [points, view_dirs, normals, feature_vectors], dim=-1
        )
       
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x