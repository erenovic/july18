
import torch
import torch.nn as nn


class SingleVarianceNetwork(nn.Module):
    '''
    Taken from NeuS official repository
    '''
    def __init__(self, config):
        super(SingleVarianceNetwork, self).__init__()
        self.variance = nn.Parameter(torch.tensor(config["init_val"]))

    def forward(self, x):
        return torch.ones([len(x), 1]).to(x.device) * torch.exp(self.variance * 10.0)