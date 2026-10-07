"""Backend selection; no model loading or CUDA calls."""
import os

def select_backend(requested, device_type, platform=None):
    if requested not in ('original', 'gpu'):
        raise ValueError('inference_backend must be original or gpu')
    if requested == 'original':
        return 'original', None
    if device_type != 'cuda':
        return 'original', 'CUDA is unavailable'
    if (os.name if platform is None else platform) != 'nt':
        return 'original', 'the accelerated NVRTC runtime currently supports native Windows only'
    return 'gpu', None
