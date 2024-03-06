""" Anonymous Speaker Condition Generator
"""

import torch
import torch.nn as nn

import FrEIA.framework as Ff
import FrEIA.modules as Fm


class ASCG(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cinn = self.build_inn()
        self.trainable_parameters = [
            p for p in self.cinn.parameters() if p.requires_grad
        ]
        for p in self.trainable_parameters:
            p.data = 0.01 * torch.randn_like(p)

        self.optim = torch.optim.Adam(
            self.trainable_parameters, lr=lr, weight_decay=weight_decay
        )
