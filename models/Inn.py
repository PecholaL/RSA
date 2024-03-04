""" Modified from DeepMIH
    configuration is at RSAno/models/config.yaml
"""

import torch.nn as nn

from models.InnBlock import InnBlock


class Mihnet_s1(nn.Module):
    def __init__(self, config_path, num_inn):
        super(Mihnet_s1, self).__init__()
        self.innbs = nn.ModuleList([InnBlock(config_path) for _ in range(num_inn)])

    def forward(self, a, m, rev=False):
        if not rev:
            for innb in self.innbs:
                a, m = innb(a, m)
        else:
            for innb in reversed(self.innbs):
                a, m = innb(a, m, rev=True)
        return a, m


class Mihnet_s2(nn.Module):
    def __init__(self, config_path, num_inn):
        super(Mihnet_s2, self).__init__()
        self.innbs = nn.ModuleList([InnBlock(config_path) for _ in range(num_inn)])

    def forward(self, a, m, rev=False):
        if not rev:
            for innb in self.innbs:
                a, m = innb(a, m)
        else:
            for innb in reversed(self.innbs):
                a, m = innb(a, m, rev=True)
        return a, m
