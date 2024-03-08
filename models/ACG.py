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
            lr=self.config["ACG"]["training"]["lr"],
            weight_decay=self.config["ACG"]["training"]["weight_decay"],
        )

    """ build the INN
            input: key, 
            output: condition for RSA.anonymizer
    """

    def build_inn(self):
        # full connected subnet (i.e. φ,ψ,ρ,η) for the INN block
        def subnet(f_in, f_out):
            return nn.Sequential(
                nn.Linear(
                    self.config["ACG"]["struct"]["f_in"],
                    self.config["ACG"]["struct"]["f_mid"],
                ),
                nn.ReLU(),
                nn.Linear(
                    self.config["ACG"]["struct"]["f_mid"],
                    self.config["ACG"]["struct"]["f_out"],
                ),
            )

        inn = Ff.SequenceINN(2)
        for k in range(8):
            inn.append(Fm.AllInOneBlock, subnet_constructor=subnet, permute_soft=True)

        nodes = [Ff.InputNode(self.config["ACG"]["struct"]["input_size"])]
        nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}))  # nodes: [node0, node1]

        for k in range(self.config["ACG"]["struct"]["layers"]):
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {"seed": k}))

        return Ff.ReversibleGraphNet(nodes, verbose=False)

    def forward(self, x):
        z = self.inn(x)
        jac = self.inn.log_jacobian(run_forward=False)
        return z, jac

    def reverse_sample(self, z):
        return self.inn(z, rev=True)
