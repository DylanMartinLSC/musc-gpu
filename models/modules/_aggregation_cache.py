"""Reuse spatial normalization within one inference, never across calls."""
import math
import torch
from torch.nn import functional as F
from models.modules import _encoder_gpu as deployed


class RadiusAggregationCache:
    def __init__(self):
        self._spatial = {}

    def clear(self):
        self._spatial.clear()

    def __call__(self, layer, features, return_cpu=True):
        first = features[0]
        batch, length, channels = first.shape
        side = math.isqrt(length-1)
        if (side*side != length-1 or layer.patch_maker.stride != 1
                or layer.r not in (1, 3, 5) or layer.LNA.output_dim != channels
                or len(features) != len(layer.LNA.preprocessing_modules)
                or any(t.shape != first.shape or t.dtype != first.dtype or t.device != first.device for t in features)):
            return layer._embed(features, return_cpu=return_cpu)
        shape = (channels, side, side)
        affine_key = (first.device, shape, torch.get_default_dtype())
        if affine_key not in deployed._AFFINE:
            if len(deployed._AFFINE) >= 4:
                deployed._AFFINE.clear()
            weight = torch.ones(shape, device=first.device, dtype=torch.get_default_dtype())
            deployed._AFFINE[affine_key] = (weight, torch.zeros_like(weight))
        weight, bias = deployed._AFFINE[affine_key]
        values = []
        for token in features:
            try:
                version = token._version
            except RuntimeError:
                # Inference-mode tensors have no mutation counter. Keep their
                # original behavior by recomputing instead of memoizing them.
                version = None
            key = ((id(token), version, token.data_ptr(), tuple(token.shape),
                    tuple(token.stride()), token.dtype, token.device)
                   if version is not None else None)
            entry = self._spatial.get(key) if key is not None else None
            if entry is not None and entry[0] is token:
                spatial = entry[1]
            else:
                spatial = token[:, 1:].reshape(batch, side, side, channels).permute(0, 3, 1, 2)
                spatial = F.layer_norm(spatial, shape, weight, bias)
                if key is not None:
                    self._spatial[key] = (token, spatial)
            if layer.r != 1:
                spatial = F.avg_pool2d(spatial, layer.r, stride=1,
                                       padding=layer.r//2, count_include_pad=True)
                deployed._STATS['direct_pool_calls'] += 1
            values.append(spatial.flatten(2).transpose(1, 2))
        deployed._STATS['aggregation_calls'] += 1
        result = torch.stack(values, dim=2).detach()
        return result.cpu() if return_cpu else result
