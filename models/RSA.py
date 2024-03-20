""" Main Module
        Anonymizer / Restorer
"""

import torch
import torch.nn as nn
from models.CinnBlock import CinnBlock


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
