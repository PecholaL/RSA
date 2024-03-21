""" Speaker Encoder for Contrastive Learning
    * Encoder from AdaIN-VC(https://github.com/jjery2243542/adaptive_voice_conversion)
    * APC from MAIN-VC(https://github.com/PecholaL/MAIN-VC)
"""

import torch
import torch.nn as nn


import yaml  # for test (build model)


class SpkEnc(nn.Module):
    def __init__(
        self,
        c_in,
        c_h,
        c_out,
        kernel_size,
        c_bank,
        n_conv_blocks,
        n_dense_blocks,
        subsample,
        act,
        dropout_rate,
    ):
        super(SpkEnc, self).__init__()

        self.c_in = c_in
        self.c_h = c_h
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.c_bank = c_bank
        self.n_conv_blocks = n_conv_blocks
        self.n_dense_blocks = n_dense_blocks
        self.subsample = subsample
        self.act = get_act_func(act)

        # build spk. encoder
        self.APC_module = nn.ModuleList(
            [
                nn.Conv1d(
                    c_in,
                    c_bank,
                    kernel_size=3,
                    padding=1,
                    dilation=1,
                    padding_mode="reflect",
                ),
                nn.Conv1d(
                    c_in,
                    c_bank,
                    kernel_size=3,
                    padding=2,
                    dilation=2,
                    padding_mode="reflect",
                ),
                nn.Conv1d(
                    c_in,
                    c_bank,
                    kernel_size=3,
                    padding=4,
                    dilation=4,
                    padding_mode="reflect",
                ),
                nn.Conv1d(
                    c_in,
                    c_bank,
                    kernel_size=3,
                    padding=6,
                    dilation=6,
                    padding_mode="reflect",
                ),
                nn.Conv1d(
                    c_in,
                    c_bank,
                    kernel_size=3,
                    padding=8,
                    dilation=8,
                    padding_mode="reflect",
                ),
            ]
        )

        in_channels = self.c_in + self.c_bank * 5
        self.in_conv_layer = nn.Conv1d(in_channels, c_h, kernel_size=1)

        self.first_conv_layers = nn.ModuleList(
            [nn.Conv1d(c_h, c_h, kernel_size=kernel_size) for _ in range(n_conv_blocks)]
        )

        self.second_conv_layers = nn.ModuleList(
            [
                nn.Conv1d(c_h, c_h, kernel_size=kernel_size, stride=sub)
                for sub, _ in zip(subsample, range(n_conv_blocks))
            ]
        )

        self.pooling_layer = nn.AdaptiveAvgPool1d(1)

        self.first_dense_layers = nn.ModuleList(
            [nn.Linear(c_h, c_h) for _ in range(n_dense_blocks)]
        )
        self.second_dense_layers = nn.ModuleList(
            [nn.Linear(c_h, c_h) for _ in range(n_dense_blocks)]
        )

        self.output_layer = nn.Linear(c_h, c_out)
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def conv_blocks(self, inData):
        outData = inData
        for l in range(self.n_conv_blocks):
            y = pad_layer(outData, self.first_conv_layers[l])
            y = self.act(y)
            y = self.dropout_layer(y)
            y = pad_layer(y, self.second_conv_layers[l])
            y = self.act(y)
            y = self.dropout_layer(y)
            if self.subsample[l] > 1:
                outData = nn.functional.avg_pool1d(
                    outData, kernel_size=self.subsample[l], ceil_mode=True
                )
            outData = y + outData
        return outData

    def dense_blocks(self, inp):
        out = inp
        for l in range(self.n_dense_blocks):
            y = self.first_dense_layers[l](out)
            y = self.act(y)
            y = self.dropout_layer(y)
            y = self.second_dense_layers[l](y)
            y = self.act(y)
            y = self.dropout_layer(y)
            out = y + out
        return out

    def APC(self, inp, act):
        out_list = []
        for layer in self.APC_module:
            out_list.append(act(layer(inp)))
        outData = torch.cat(out_list + [inp], dim=1)
        return outData

    def forward(self, x):
        # APC
        out = self.APC(x, act=self.act)
        # dimension reduction
        out = pad_layer(out, self.in_conv_layer)
        out = self.act(out)
        # conv blocks
        out = self.conv_blocks(out)
        # avg pooling
        out = self.pooling_layer(out).squeeze(2)
        # dense blocks
        out = self.dense_blocks(out)
        out = self.output_layer(out)
        return out


def pad_layer(inData, layer, pad_mode="reflect"):
    kernel_size = layer.kernel_size[0]
    if kernel_size % 2 == 0:
        pad = (kernel_size // 2, kernel_size // 2 - 1)
    else:
        pad = (kernel_size // 2, kernel_size // 2)
    inData = nn.functional.pad(inData, pad=pad, mode=pad_mode)
    outData = layer(inData)
    return outData


def get_act_func(func_name):
    if func_name == "lrelu":
        return nn.LeakyReLU()
    return nn.ReLU()


# test
# with open("./models/config.yaml") as f:
#     config = yaml.load(f, Loader=yaml.FullLoader)
# Es = SpkEnc(**config["RSA"]["struct"]["SpkEnc"])
# x = torch.randn(4, 80, 100)
# emb = Es(x)  # torch.Size([4, 64])
# print(emb.shape)
