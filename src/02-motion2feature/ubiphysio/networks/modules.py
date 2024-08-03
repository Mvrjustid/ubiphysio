import torch
import torch.nn as nn
import numpy as np
import time
import math
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from networks.layers import *

def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        # m.bias.data.fill_(0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class VQEncoderV3(nn.Module):
    def __init__(self, input_size, channels, n_down):
        super(VQEncoderV3, self).__init__()
        assert len(channels) == n_down
        layers = [
            nn.Conv1d(input_size, channels[0], 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(channels[0]),
        ]

        for i in range(1, n_down):
            layers += [
                nn.Conv1d(channels[i-1], channels[i], 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                ResBlock(channels[i]),
            ]
        self.main = nn.Sequential(*layers)
        # self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(init_weight)
        # self.out_net.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        # print(outputs.shape)
        return outputs
    
class VQDecoderV3(nn.Module):
    def __init__(self, input_size, channels, n_resblk, n_up):
        super(VQDecoderV3, self).__init__()
        assert len(channels) == n_up + 1
        if input_size == channels[0]:
            layers = []
        else:
            layers = [nn.Conv1d(input_size, channels[0], kernel_size=3, stride=1, padding=1)]

        for i in range(n_resblk):
            layers += [ResBlock(channels[0])]
        # channels = channels
        for i in range(n_up):
            layers += [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(channels[i], channels[i+1], kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        layers += [nn.Conv1d(channels[-1], channels[-1], kernel_size=3, stride=1, padding=1)]
        self.main = nn.Sequential(*layers)
        self.main.apply(init_weight)
        # self.out_net.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return outputs
    
class VQDiscriminator(nn.Module):
    def __init__(self, input_size, hidden_size, n_layer):
        super(VQDiscriminator, self).__init__()
        sequence = [nn.Conv1d(input_size, hidden_size, 4, 2, 1),
                    nn.BatchNorm1d(hidden_size),
                    nn.LeakyReLU(0.2, inplace=True)
                    ]
        layer_size = hidden_size
        for i in range(n_layer-1):
            sequence += [
                    nn.Conv1d(layer_size, layer_size//2, 4, 2, 1),
                    nn.BatchNorm1d(layer_size//2),
                    nn.LeakyReLU(0.2, inplace=True)
            ]
            layer_size = layer_size // 2

        self.out_net = nn.Conv1d(layer_size, 1, 3, 1, 1)
        self.main = nn.Sequential(*sequence)

        self.out_net.apply(init_weight)
        self.main.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        feats = self.main(inputs)
        outs = self.out_net(feats)
        return feats.permute(0, 2, 1), outs.permute(0, 2, 1)