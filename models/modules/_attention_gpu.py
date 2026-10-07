"""Native alignment-aware block softmax with dual probability output stores."""
import torch
import time
from torch.nn import functional as F

from models.modules._cuda_runtime import compile_kernels, launch

# Preserve native 1024 logical lanes using 256 physical threads and aligned tails.
SOURCE = r'''
__device__ float read_half(unsigned short bits) {
  float result; asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits)); return result;
}
__device__ unsigned short write_half(float value) {
  unsigned short result; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(result) : "f"(value)); return result;
}
__device__ float warp_reduce(float value, bool maximum) {
  #pragma unroll
  for (int step=16; step; step>>=1) {
    float peer=__shfl_down_sync(0xffffffff,value,step);
    value=maximum ? (value<peer ? peer : value) : value+peer;
  }
  return value;
}
__device__ float virtual_reduce(float* values, float* shared, bool maximum) {
  int lane=threadIdx.x&31, warp=threadIdx.x>>5;
  float partial[4];
  #pragma unroll
  for (int g=0;g<4;g++) partial[g]=warp_reduce(values[g],maximum);
  __syncthreads();
  if (lane==0) {
    #pragma unroll
    for (int g=0;g<4;g++) shared[warp+8*g]=partial[g];
  }
  __syncthreads();
  float value=threadIdx.x<32 ? shared[lane] : (maximum ? -3.402823466e+38F : 0.0f);
  if (warp==0) value=warp_reduce(value,maximum);
  if (threadIdx.x==0) shared[0]=value;
  __syncthreads();
  return shared[0];
}
extern "C" __global__ void aligned_dual(unsigned short* logits, float* probabilities, int rows, int retain) {
  int row=blockIdx.x;
  if (row>=rows) return;
  long long origin=(long long)row*1370;
  int shift=origin&7;
  float a[4],b[4],values[4];
  int first[4],second[4];
  bool va[4],vb[4];
  #pragma unroll
  for (int g=0;g<4;g++) {
    first[g]=threadIdx.x+256*g-shift;
    second[g]=first[g]+1024;
    va[g]=first[g]>=0;
    vb[g]=second[g]<1370;
    a[g]=va[g] ? read_half(logits[origin+first[g]]) : -3.402823466e+38F;
    b[g]=vb[g] ? read_half(logits[origin+second[g]]) : -3.402823466e+38F;
    values[g]=a[g]<b[g] ? b[g] : a[g];
  }
  __shared__ float shared[32];
  float maximum=virtual_reduce(values,shared,true);
  #pragma unroll
  for (int g=0;g<4;g++) {
    values[g]=0.0f;
    if (va[g]) values[g]=values[g]+expf(a[g]-maximum);
    if (vb[g]) values[g]=values[g]+expf(b[g]-maximum);
  }
  float total=virtual_reduce(values,shared,false);
  #pragma unroll
  for (int g=0;g<4;g++) {
    if (va[g]) {
      float value=expf(a[g]-maximum)/total;
      if (retain) probabilities[origin+first[g]]=value;
      logits[origin+first[g]]=write_half(value);
    }
    if (vb[g]) {
      float value=expf(b[g]-maximum)/total;
      if (retain) probabilities[origin+second[g]]=value;
      logits[origin+second[g]]=write_half(value);
    }
  }
}
'''
_STATS = {}
_COMPILED = set()
_SETUP_SECONDS = 0.0

def begin_inference():
    _STATS.clear()
    _STATS.update(fused_softmax_calls=0, reused_casts=0, softmax_calls=0, half_only_calls=0)

def encode_image(model, image, out_layers):
    original_softmax, original_bmm = F.softmax, torch.bmm
    pending = {}
    active = [None]
    restored = []
    blocks = getattr(getattr(getattr(model, 'visual', None), 'transformer', None), 'resblocks', [])
    # Transformer.forward retains only block12 weights. Scope to its actual MHA.
    if len(blocks) == 24:
        for index, block in enumerate(blocks):
            attention = block.attn
            if not isinstance(attention, torch.nn.MultiheadAttention):
                continue
            present = 'forward' in attention.__dict__
            previous = attention.__dict__.get('forward')
            original = attention.forward
            def wrapped(*args, _original=original, _index=index, **kwargs):
                previous_active = active[0]
                active[0] = _index
                try:
                    return _original(*args, **kwargs)
                finally:
                    active[0] = previous_active
            restored.append((attention, present, previous))
            attention.forward = wrapped
    def softmax(input, dim=None, _stacklevel=3, dtype=None):
        global _SETUP_SECONDS
        _STATS['softmax_calls'] += 1
        if (input.is_cuda and input.dtype == torch.float16 and input.ndim == 3
                and input.shape[-2:] == (1370,1370) and input.is_contiguous()
                and dim in (-1,2) and dtype is None and not torch.is_grad_enabled()
                and torch.is_autocast_enabled('cuda')
                and torch.get_autocast_dtype('cuda') == torch.float16):
            device = input.device.index
            start = time.perf_counter()
            kernel = compile_kernels(SOURCE,['aligned_dual'],device)['aligned_dual']
            if device not in _COMPILED:
                _SETUP_SECONDS += time.perf_counter()-start
                _COMPILED.add(device)
            retain = active[0] is None or active[0] == 11
            output = torch.empty_like(input,dtype=torch.float32) if retain else input
            if not retain:
                _STATS['half_only_calls'] += 1
            rows = input.numel()//1370
            launch(kernel,(rows,1,1),(256,1,1),[input,output],[rows,int(retain)],device)
            pending[id(output)] = (output,input)
            _STATS['fused_softmax_calls'] += 1
            _STATS['dispatch'] = dict(rows=rows, columns=1370, block=[256,1,1],
                                      output_dtype=str(output.dtype), input_stride=list(input.stride()))
            return output
        return original_softmax(input,dim=dim,_stacklevel=_stacklevel,dtype=dtype)
    def bmm(left,right,*,out=None):
        entry = pending.pop(id(left),None)
        if entry is not None and right.dtype==torch.float16 and out is None:
            _STATS['reused_casts'] += 1
            return original_bmm(entry[1],right)
        return original_bmm(left,right) if out is None else original_bmm(left,right,out=out)
    try:
        F.softmax,torch.bmm=softmax,bmm
        return model.encode_image(image,out_layers)
    finally:
        F.softmax,torch.bmm=original_softmax,original_bmm
        pending.clear()
        for attention, present, previous in reversed(restored):
            if present:
                attention.forward = previous
            else:
                attention.__dict__.pop('forward', None)

def research_diagnostics():
    return dict(_STATS, first_use_setup_seconds=_SETUP_SECONDS)



begin_inference()
