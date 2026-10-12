"""Backend selection; no model loading or CUDA calls."""
import os
from copy import deepcopy

def select_backend(requested, device_type, platform=None):
    if requested not in ('original', 'gpu', 'super_fast'):
        raise ValueError('inference_backend must be original, gpu or super_fast')
    if requested == 'original':
        return 'original', None
    if device_type != 'cuda':
        return 'original', 'CUDA is unavailable'
    if (os.name if platform is None else platform) != 'nt':
        return 'original', 'the accelerated NVRTC runtime currently supports native Windows only'
    return requested, None


def configure_backend(cfg, device_type, platform=None):
    """Apply the opt-in fixed preset only when its accelerated path is available."""
    requested = cfg['models'].get('inference_backend', 'original')
    selected, _ = select_backend(requested, device_type, platform)
    if selected != 'super_fast':
        return cfg
    models = cfg['models']
    datasets = cfg['datasets']
    if (models['backbone_name'] != 'ViT-L-14-336' or models['pretrained'] != 'openai'
            or datasets['img_resize'] != 518 or models['batch_size'] != 4
            or datasets['divide_num'] != 1):
        raise ValueError('super_fast requires ViT-L-14-336/openai, img_resize=518, '
                         'batch_size=4 and divide_num=1; use gpu for other configurations')
    result = deepcopy(cfg)
    result['models']['feature_layers'] = [5]
    result['models']['r_list'] = [3]
    return result
