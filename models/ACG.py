""" Anonymization Condition Generator
"""

import torch
import torch.nn as nn

import FrEIA.framework as Ff
import FrEIA.modules as Fm

import yaml


class ACG(nn.Module):
    def __init__(self, config_path) -> None:
        super().__init__()
        with open(config_path) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self.inn = self.build_inn()

        self.trainable_parameters = [
            p for p in self.inn.parameters() if p.requires_grad
        ]
        for p in self.trainable_parameters:
            p.data = 0.01 * torch.randn_like(p)

        self.optim = torch.optim.Adam(
            self.trainable_parameters,
            lr=float(self.config["ACG"]["training"]["lr"]),
            weight_decay=float(self.config["ACG"]["training"]["weight_decay"]),
        )

    """ build the INN
            input: key, 
            output: condition for RSA.anonymizer
    """

    def build_inn(self):
        # full connected subnet (i.e. φ,ψ,ρ,η) for the INN block
        def subnet(ch_in, ch_out):
            # print("subnettttt", ch_in, ch_out)
            return nn.Sequential(
                nn.Linear(
                    ch_in,
                    int(self.config["ACG"]["struct"]["f_mid"]),
                ),
                nn.ReLU(),
                nn.Linear(
                    int(self.config["ACG"]["struct"]["f_mid"]),
                    ch_out,
                ),
            )

        cond = Ff.ConditionNode(1)
        nodes = [
            Ff.InputNode(
                1,
                1,
                int(self.config["ACG"]["struct"]["input_size"]),
            )
        ]

        nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}))  # nodes: [node0, node1]

        for k in range(int(self.config["ACG"]["struct"]["layers"])):
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {"seed": k}))
            nodes.append(
                Ff.Node(
                    nodes[-1],
                    Fm.GLOWCouplingBlock,
                    {"subnet_constructor": subnet, "clamp": 1.0},
                    conditions=cond,
                )
            )

        return Ff.ReversibleGraphNet(
            nodes + [cond, Ff.OutputNode(nodes[-1])], verbose=False
        )

    def forward(self, x, cond):
        return self.inn(x, cond)

    def reverse_sample(self, z, cond):
        return self.inn(z, cond, rev=True)
