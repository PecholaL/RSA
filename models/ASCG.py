""" Anonymous Speaker Condition Generator
"""

import torch
import torch.nn as nn

import FrEIA.framework as Ff
import FrEIA.modules as Fm

import yaml


class ASCG(nn.Module):
    def __init__(self, config_path) -> None:
        super().__init__()
        with open(config_path) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.cinn = self.build_inn()
        self.trainable_parameters = [
            p for p in self.cinn.parameters() if p.requires_grad
        ]
        for p in self.trainable_parameters:
            p.data = 0.01 * torch.randn_like(p)

        self.optim = torch.optim.Adam(
            self.trainable_parameters,
            lr=self.config["ASCG"]["training"]["lr"],
            weight_decay=["ASCG"]["training"]["weight_decay"],
        )
