from __future__ import print_function
import torch
import re
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.utils.checkpoint as cp
# from .utils import load_state_dict_from_url
import torchvision.models as models
from torch.hub import load_state_dict_from_url
from collections import OrderedDict
from torch import Tensor
from collections import OrderedDict
import requests
import io
from efficientnet_pytorch import EfficientNet


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.mp = nn.AdaptiveMaxPool2d(output_size=(1, 1))

    def forward(self, x):
        return torch.cat([self.ap(x), self.mp(x)], dim=1)

class final_model(nn.Module):

    def __init__(self, c_out=3, n_tiles=30, tile_size=70):
        super().__init__()
        self.c_out = c_out
        self.n_tiles = n_tiles
        self.tile_size = tile_size
        self.reduced_cnn = EfficientModel(n_tiles, tile_size)
        self.lstm = LstmModel()

    def forward(self, x):
        # b, timepoints, patches, 3, tile_size, tile_size, where b=1
        x = x.view(-1, self.n_tiles, 3, self.tile_size, self.tile_size)
        h = self.reduced_cnn(x)  # b*timepoints, feature_vector
        h = h.view(-1, h.shape[0], h.shape[1])  # b, timepoints, feature_vector

        h = self.lstm(h)  # b, timepoints, 1

        return h


class LstmModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return None

class EfficientModel(nn.Module):

    def __init__(self, n_tiles=30, tile_size=70, name='efficientnet-b0'):
        super().__init__()

        m = EfficientNet.from_pretrained(name, advprop=True, in_channels=1)
        c_feature = m._fc.in_features
        m._fc = nn.Identity()
        self.feature_extractor = m
        self.n_tiles = n_tiles
        self.tile_size = tile_size
        self.head = AttentionHead(c_feature, n_tiles)

    def forward(self, x):
        # b*timepoints, patches, 3, tile_size, tile_size, where b=1
        x = x.view(-1, 3, self.tile_size, self.tile_size)
        h = self.feature_extractor(x)
        h, w = self.head(h)
        return h

class AttentionHead(nn.Module):

    def __init__(self, c_in, n_tiles):
        self.n_tiles = n_tiles
        super().__init__()
        self.attention_pool = AttentionPool(c_in, c_in//2)

    def forward(self, x):

        bn, c = x.shape
        h = x.view(-1, self.n_tiles, c)
        h, w = self.attention_pool(h)

        return h, w

class AttentionPool(nn.Module):

    def __init__(self, c_in, d):
        super().__init__()
        self.lin_V = nn.Linear(c_in, d)
        self.lin_w = nn.Linear(d, 1)

    def compute_weights(self, x):
        key = self.lin_V(x)  # b, n, d
        weights = self.lin_w(torch.tanh(key))  # b, n, 1
        weights = torch.softmax(weights, dim=1)
        return weights

    def forward(self, x):
        weights = self.compute_weights(x)
        pooled = torch.matmul(x.transpose(1, 2), weights).squeeze(2)   # b, c, n x b, n, 1 => b, c, 1
        return pooled, weights


