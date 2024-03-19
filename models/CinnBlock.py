""" Modified from DeepHIN
    * define invertible block InnBlock for Mihnet
"""

import torch
import torch.nn as nn

from models.Subnet import Subnet


class CinnBlock(nn.Module):
    def __init__(self, config):
        super(CinnBlock, self).__init__()
        self.channel = config["RSA"]["struct"]["cinnblock"]["channel"]
        self.clamp = config["RSA"]["struct"]["cinnblock"]["clamp"]
        self.cond_len = config["RSA"]["struct"]["cinnblock"]["cond_len"]
        self.cond_trans = config["RSA"]["struct"]["cinnblock"]["cond_trans"]

        # ψ
        self.psi = Subnet(self.channel, self.channel - 1)
        # φ
        self.phi = Subnet(self.channel, self.channel - 1)
        # ρ
        self.rho = Subnet(self.channel, self.channel - 1)
        # η
        self.eta = Subnet(self.channel, self.channel - 1)

        self.cond_layer = nn.Linear(self.cond_len, self.cond_trans)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x1, x2, cond, rev=False):
        x1 = x1.to(torch.float32)
        x2 = x2.to(torch.float32)
        cond = cond.to(torch.float32)
        # transform shape of condition: [B,1,192]->[B,1,cond_trans]->[B,C,cond_trans]
        cond_t = self.cond_layer(cond)
        cond_t = nn.functional.pad(cond_t, (0, x1.shape[2] - cond_t.shape[2]), value=0)
        print("cond_t_padding.shape: ", cond_t.shape)
        if not rev:
            t2 = self.phi(x2)
            t2i = torch.cat((t2, cond_t), dim=1)
            s2 = self.e(self.psi(x2))
            s2i = torch.cat((s2, cond_t), dim=1)
            y1 = s2i * x1 + t2i
            s1 = self.e(self.rho(y1))
            t1 = self.eta(y1)
            s1i = torch.cat((s1, cond_t), dim=1)
            t1i = torch.cat((t1, cond_t), dim=1)
            y2 = s1i * x2 + t1i
        else:
            s1 = self.e(self.rho(x1))
            s1i = torch.cat((s1, cond_t), dim=1)
            t1 = self.eta(x1)
            t1i = torch.cat((t1, cond_t), dim=1)
            y2 = (x2 - t1i) / s1i
            t2 = self.phi(y2)
            t2i = torch.cat((t2, cond_t), dim=1)
            s2 = self.e(self.psi(y2))
            s2i = torch.cat((s2, cond_t), dim=1)
            y1 = (x1 - t2i) / s2i
        return y1, y2
