""" Modified from DeepHIN
    * define invertible block InnBlock for Mihnet
"""

import torch
import torch.nn as nn

from models.Subnet import Subnet


class InnBlock(nn.Module):
    def __init__(self, config):
        super(InnBlock, self).__init__()
        self.channel = config["RSA"]["struct"]["channel"]
        self.clamp = config["RSA"]["struct"]["clamp"]

        # ψ
        self.psi = Subnet(self.channel, self.channel)
        # φ
        self.phi = Subnet(self.channel, self.channel)
        # ρ
        self.rho = Subnet(self.channel, self.channel)
        # η
        self.eta = Subnet(self.channel, self.channel)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x1, x2, cond, rev=False):
        x1 = x1.to(torch.float32)
        x2 = x2.to(torch.float32)
        cond = cond.to(torch.float32)
        if not rev:
            t2 = self.phi(x2)
            s2 = self.psi(x2)
            y1 = self.e(s2) * x1 + t2
            s1, t1 = self.rho(y1), self.eta(y1)
            y2 = self.e(s1) * x2 + t1
        else:
            s1, t1 = self.rho(x1), self.eta(x1)
            y2 = (x2 - t1) / self.e(s1)
            t2 = self.phi(y2)
            s2 = self.psi(y2)
            y1 = (x1 - t2) / self.e(s2)
        return y1, y2
