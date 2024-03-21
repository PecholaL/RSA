""" Main Module
        Anonymizer / Restorer
"""

import random
import torch
import torch.nn as nn
from models.CinnBlock import CinnBlock
from models.SpkEnc import SpkEnc


class RSA(nn.Module):
    def __init__(self, config) -> None:
        super(RSA, self).__init__()
        self.cinnb_nums = int(config["RSA"]["struct"]["basic"]["cinnb_nums"])
        self.cinnbs = nn.ModuleList([CinnBlock(config) for _ in range(self.cinnb_nums)])

    def forward(self, x, cond, rev=False):
        x1, x2 = torch.split(x, int(x.shape[1] / 2), dim=1)

        if not rev:
            for cinnb in self.cinnbs:
                x1, x2 = cinnb(x1, x2, cond)
        else:
            for cinnb in reversed(self.cinnbs):
                x1, x2 = cinnb(x1, x2, cond, True)
        x_out = torch.cat((x1, x2), dim=1)

        return x_out


class RSA_v0(nn.Module):
    def __init__(self, config) -> None:
        super(RSA, self).__init__()
        self.cinnb_nums = int(config["RSA"]["struct"]["basic"]["cinnb_nums"])
        self.cinnbs = nn.ModuleList([CinnBlock(config) for _ in range(self.cinnb_nums)])
        self.enc = SpkEnc(**config["RSA"]["struct"]["SpkEnc"])

    def forward(self, x, x_, cond, rev=False):
        x1, x2 = torch.split(x, int(x.shape[1] / 2), dim=1)

        if not rev:
            for cinnb in self.cinnbs:
                x1, x2 = cinnb(x1, x2, cond)
        else:
            for cinnb in reversed(self.cinnbs):
                x1, x2 = cinnb(x1, x2, cond, True)
        x_out = torch.cat((x1, x2), dim=1)

        seg_list = list(torch.split(x, 5, dim=2))
        random.shuffle(seg_list)
        x_ts = torch.cat(seg_list, dim=2)
        anchor_emb = self.enc(x)
        positive_emb = self.enc(x_ts)
        negative_emb = self.enc(x_out)

        return x_out, anchor_emb, positive_emb, negative_emb
