""" TBD/deprecated 
    found some limitations of FrEIA when building anonymizer
"""

""" Main Module
        Anonymizer / Restorer
"""

import torch.optim
import torch.nn as nn

from FrEIA.framework import *
from FrEIA.modules import *
import yaml


""" HYPER PARAMETERS
    # load from config.yaml
"""
# config_path = "./models/config.yaml"  # excecute train_ACG.py in the base directory
config_path = "../models/config.yaml"  # for test in _TEST_ONLY_ directory
with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

    """ hyper params for cond coupling block
    """
    # condition length
    cond_len = config["RSA"]["struct"]["cond_len"]
    # for subnet
    level = config["RSA"]["struct"]["level"]
    c_in = config["RSA"]["struct"]["c_in"]
    c_mid = config["RSA"]["struct"]["c_mid"]
    c_out = config["RSA"]["struct"]["c_out"]
    cc_layers = config["RSA"]["struct"]["cc_layers"]
    # for cc
    n_blocks = config["RSA"]["struct"]["n_blocks"]
    # for training
    clamp = config["RSA"]["training"]["clamp"]  # for RNVP/GLOW coupling block
    init_scale = config["RSA"]["training"]["init_scale"]
    lr = float(config["RSA"]["training"]["lr"])
    weight_decay = float(config["RSA"]["training"]["weight_decay"])
    decay_by = float(config["RSA"]["training"]["decay_by"])
    n_iterations = int(config["RSA"]["training"]["n_iterations"])


""" Anonymizer / Restorer
    ################################################################################
"""


# cond_subnet is composed of a set of conv layers
# level: num of conv layers, level<=4
def cond_subnet(in_dim, out_dim):
    modules = []
    print("subnet", in_dim, out_dim)
    for i in range(level):
        if i == 0:
            modules.extend(
                [
                    nn.Conv1d(in_dim, out_dim, 3, stride=2, padding=1),
                    nn.LeakyReLU(),
                ]
            )
        else:
            modules.extend(
                [
                    nn.Conv1d(out_dim, out_dim, 3, stride=2, padding=1),
                    nn.LeakyReLU(),
                ]
            )
    modules.append(nn.BatchNorm1d(2 * out_dim))
    return nn.Sequential(*modules)


def build_inn(nodes):
    nodes.append(Node([nodes[-1].out0], Flatten, {}, name="flatten"))
    for i in range(n_blocks):
        nodes.append(
            Node([nodes[-1].out0], PermuteRandom, {"seed": i}, name=f"permute_{i}")
        )
        nodes.append(
            Node(
                [nodes[-1].out0],
                GLOWCouplingBlock,
                {"subnet_constructor": cond_subnet, "clamp": clamp},
                conditions=[cond_node],
                name=f"GLOW_CC_{i}",
            )
        )
    nodes.append(OutputNode([nodes[-1].out0], name="out"))
    nodes.append(cond_node)
    return


def init_model(mod):
    for key, param in mod.named_parameters():
        split = key.split(".")
        if param.requires_grad:
            param.data = init_scale * torch.randn(param.data.shape)  # .cuda()
            if split[3][-1] == "3":  # last convolution in the coeff func
                param.data.fill_(0.0)


def save_model(ckpt_path):
    torch.save(RSA_cINN.state_dict(), ckpt_path)
    return


def load_model(ckpt_path):
    RSA_cINN.load_state_dict(ckpt_path)
    return


""" BUILD MODEL
"""
cond_node = ConditionNode(cond_len)
nodes = [InputNode(1, c_in, name="in")]
build_inn(nodes)
RSA_cINN = ReversibleGraphNet(nodes, verbose=False)
init_model(RSA_cINN)
params_trainable = list(filter(lambda p: p.requires_grad, RSA_cINN.parameters()))

gamma = (decay_by) ** (1.0 / 1500)
optim = torch.optim.Adam(
    params_trainable, lr=lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=weight_decay
)
weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=100, gamma=gamma)
