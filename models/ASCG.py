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

        self.cinn = self.build_cinn(self.config)
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

    """ build the conditional INN
            input: noise, 
            condition: key
            output: condition for RSA
    """

    def build_cinn(config):
        def subnet():
            return nn.Sequential(
                nn.Linear(config["ASCG"]["struct"]["ch_in"], 512),
                nn.ReLU(),
                nn.Linear(512, config["ASCG"]["struct"]["ch_out"]),
            )

        cond = Ff.ConditionNode(config["ASCG"]["struct"]["cond_node_size"])
        nodes = [Ff.InputNode(1, 1, config["ASCG"]["struct"]["input_size"])]
        nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}))  # nodes: [node0, node1]

        for k in range(config["ASCG"]["struct"]["layers"]):
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
        z = self.cinn(x, cond)
        jac = self.cinn.log_jacobian(run_forward=False)
        return z, jac

    def reverse_sample(self, z, cond):
        return self.cinn(z, cond, rev=True)
