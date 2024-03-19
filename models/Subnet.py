""" Subnet for cINN Block
"""

import torch
import torch.nn as nn

from models.utils import initialize_weights


class Subnet(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True):
        super(Subnet, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channel, 64, kernel_size=3, padding=2, dilation=2, bias=bias
        )
        self.conv2 = nn.Conv1d(
            in_channel + 64, 64, kernel_size=3, padding=2, dilation=2, bias=bias
        )
        self.conv3 = nn.Conv1d(
            in_channel + 2 * 64, 64, kernel_size=3, padding=1, dilation=1, bias=bias
        )
        self.conv4 = nn.Conv1d(
            in_channel + 3 * 64,
            out_channel,
            kernel_size=3,
            padding=2,
            dilation=2,
            bias=bias,
        )
        self.lrelu = nn.LeakyReLU(inplace=True)
        # initialization
        initialize_weights([self.conv4], 0.0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        return x4


""" @Test
"""
subnet = Subnet(40, 40)
