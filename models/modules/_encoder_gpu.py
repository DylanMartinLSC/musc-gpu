"""Eliminate Unfold neighbourhood expansion using its exact channel-mean identity.

Only immutable all-one/all-zero LayerNorm affine buffers are reused. Every input
image is encoded normally and every feature is normalized/pooled afresh.
"""
import math
import torch
from torch.nn import functional as F

_AFFINE = {}
_STATS = {'aggregation_calls': 0, 'direct_pool_calls': 0}


def aggregate_features(layer, features, return_cpu=True):
    first = features[0]
    batch, length, channels = first.shape
    side = math.isqrt(length-1)
    if (side*side != length-1 or layer.patch_maker.stride != 1
            or layer.r not in (1, 3, 5) or layer.LNA.output_dim != channels
            or len(features) != len(layer.LNA.preprocessing_modules)
            or any(t.shape != first.shape or t.dtype != first.dtype or t.device != first.device for t in features)):
        return layer._embed(features, return_cpu=return_cpu)
    shape = (channels, side, side)
    key = (first.device, shape, torch.get_default_dtype())
    if key not in _AFFINE:
        if len(_AFFINE) >= 4:
            _AFFINE.clear()
        weight = torch.ones(shape, device=first.device, dtype=torch.get_default_dtype())
        _AFFINE[key] = (weight, torch.zeros_like(weight))
    weight, bias = _AFFINE[key]
    values = []
    for token in features:
        spatial = token[:, 1:].reshape(batch, side, side, channels).permute(0, 3, 1, 2)
        spatial = F.layer_norm(spatial, shape, weight, bias)
        if layer.r != 1:
            spatial = F.avg_pool2d(spatial, layer.r, stride=1,
                                   padding=layer.r//2, count_include_pad=True)
            _STATS['direct_pool_calls'] += 1
        values.append(spatial.flatten(2).transpose(1, 2))
    _STATS['aggregation_calls'] += 1
    result = torch.stack(values, dim=2).detach()
    return result.cpu() if return_cpu else result


def encode_image(model, image, out_layers):
    from models.modules._attention_gpu import encode_image as accelerated_encode
    return accelerated_encode(model, image, out_layers)


def begin_inference():
    from models.modules._attention_gpu import begin_inference as begin_attention
    begin_attention()


def new_aggregation():
    from models.modules._aggregation_cache import RadiusAggregationCache
    return RadiusAggregationCache()


# MuSc uses this optional hook only on its GPU candidate path. Original and
# identity deployed encoder functions have no hook and use unchanged LNAMD.
encode_image.aggregate_features = aggregate_features
encode_image.new_aggregation = new_aggregation
encode_image.begin_inference = begin_inference


def research_diagnostics():
    from models.modules._attention_gpu import research_diagnostics as attention_diagnostics
    return dict(_STATS, immutable_affine_buffers=len(_AFFINE), **attention_diagnostics())
