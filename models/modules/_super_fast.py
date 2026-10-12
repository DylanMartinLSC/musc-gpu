"""Approximate layer5/r3/28x28 scoring profile, validated on bottle83 only."""
import math
import torch
from torch.nn import functional as F
from models.modules import _encoder_gpu as deployed

PATCH_GRID = 28


class SuperFastAggregation:
    def __init__(self):
        self.base = deployed.new_aggregation()

    def clear(self):
        self.base.clear()

    def __call__(self, layer, tokens, return_cpu=True):
        features = self.base(layer, tokens, return_cpu=False)
        b, p, layers, c = features.shape
        side = math.isqrt(p)
        if side != 37 or side*side != p or layers != 1 or layer.r != 3:
            raise ValueError('super_fast requires one layer, r=3 and a37x37 source patch grid')
        with torch.autocast('cuda', enabled=False):
            spatial = features.permute(0, 2, 3, 1).reshape(b, layers*c, side, side)
            pooled = F.adaptive_avg_pool2d(spatial.float(), (PATCH_GRID, PATCH_GRID)).to(features.dtype)
            features = pooled.reshape(b, layers, c, PATCH_GRID*PATCH_GRID).permute(0, 3, 1, 2).contiguous()
        return features.cpu() if return_cpu else features


def encode_image(model, image, out_layers):
    return deployed.encode_image(model, image, out_layers)


def new_aggregation():
    return SuperFastAggregation()


encode_image.aggregate_features = deployed.aggregate_features
encode_image.new_aggregation = new_aggregation
encode_image.begin_inference = deployed.begin_inference
