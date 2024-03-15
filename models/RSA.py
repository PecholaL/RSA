""" Main Module
        Anonymizer (Restorer)
"""

import warnings

import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from FrEIA.framework import *
from FrEIA.modules import *
from models.subnet_coupling import *
import data
import yaml

config_path = ""

with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    feature_channels = config["RSA"]["struct"]["feature_channels"]

    """ hyper params for cond coupling block
    """
    # condition length
    cond_len = config["RSA"]["struct"]["cond_len"]
    # for subnet
    c_in = config["RSA"]["struct"]["c_in"]
    c_mid = config["RSA"]["struct"]["c_mid"]
    c_out = config["RSA"]["struct"]["c_out"]
    cc_layers = config["RSA"]["struct"]["cc_layers"]
    # for training
    clamp = config["RSA"]["struct"]["clamp"]  # for RNVP/GLOW coupling block


conditions = [
    ConditionNode(cond_len),
]


def random_orthog(n):
    w = np.random.randn(n, n)
    w = w + w.T
    w, _, _ = np.linalg.svd(w)
    return torch.FloatTensor(w)


# cond_subnet is composed of a set of conv layers
# level: num of conv layers, level<=4
def cond_subnet(level, c_in, c_mid, c_out):
    c_intern = [feature_channels, c_mid, c_mid, c_mid]
    modules = []

    for i in range(level):
        modules.extend(
            [
                nn.Conv2d(c_intern[i], c_intern[i + 1], 3, stride=2, padding=1),
                nn.LeakyReLU(),
            ]
        )

    modules.append(nn.BatchNorm2d(2 * c_out))

    return nn.Sequential(*modules)


fc_cond_net = nn.Sequential(
    *[
        nn.Conv2d(feature_channels, c_mid, 3, stride=2, padding=1),  # 32 x 32
        nn.LeakyReLU(),
        nn.Conv2d(c_mid, 2 * c_mid, 3, stride=2, padding=1),  # 16 x 16
        nn.LeakyReLU(),
        nn.Conv2d(2 * c_mid, 2 * c_mid, 3, stride=2, padding=1),  # 8 x 8
        nn.LeakyReLU(),
        nn.Conv2d(2 * c_mid, cond_len, 3, stride=2, padding=1),  # 4 x 4
        nn.AvgPool2d(4),
        nn.BatchNorm2d(cond_len),
    ]
)


def _add_conditioned_section(nodes, depth, channels_in, channels, cond_level):

    for k in range(depth):
        nodes.append(
            Node(
                [nodes[-1].out0],
                subnet_coupling_layer,
                {
                    "clamp": clamp,
                    "F_class": F_conv,
                    "subnet": cond_subnet(cond_level, channels // 2),
                    "sub_len": channels,
                    "F_args": {"leaky_slope": 5e-2, "channels_hidden": channels},
                },
                conditions=[conditions[0]],
                name=f"conv_{k}",
            )
        )
        nodes.append(
            Node([nodes[-1].out0], conv_1x1, {"M": random_orthog(channels_in)})
        )
