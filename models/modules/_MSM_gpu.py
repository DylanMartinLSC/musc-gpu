"""Exact mutual scoring using bounded GEMM tiles for full MuSc image groups.

Every patch pair is evaluated. Dot-product scratch stays within 256 MiB;
near-zero pairs use direct differences to preserve accuracy.
"""
import torch
from models.modules._cuda_runtime import compile_kernels, launch
MAX_DOT_BYTES = 256*1024*1024

CUDA_SOURCE = r'''
__device__ __forceinline__ float half_float(unsigned short bits) {
    float value;
    asm("cvt.f32.f16 %0, %1;" : "=f"(value) : "h"(bits));
    return value;
}

__device__ __forceinline__ float warp_distance(
    const float* z, int query, int reference, int c, int lane) {
    float sum = 0.0f;
    for (int channel=lane; channel<c; channel+=32) {
        float difference = z[query*c+channel]-z[reference*c+channel];
        sum = fmaf(difference,difference,sum);
    }
    for (int offset=16; offset>0; offset/=2)
        sum += __shfl_down_sync(0xffffffff,sum,offset);
    return __shfl_sync(0xffffffff,sum,0);
}

extern "C" __global__ void half_feature_bounds(
    const float* z, const unsigned short* h, const float* center,
    float* norms, float* errors, int total, int c) {
    int row=blockIdx.x*8+threadIdx.y;
    if(row>=total) return;
    int lane=threadIdx.x;
    double norm=0.0, residual=0.0;
    for(int channel=lane; channel<c; channel+=32) {
        // Difference of FP32 inputs evaluated in FP64; h may include both
        // FP32 subtraction and half conversion error, measured together here.
        double value=(double)z[row*c+channel]-(double)center[channel];
        double difference=value-(double)half_float(h[row*c+channel]);
        norm+=value*value;
        residual+=difference*difference;
    }
    for(int offset=16;offset>0;offset/=2) {
        norm+=__shfl_down_sync(0xffffffff,norm,offset);
        residual+=__shfl_down_sync(0xffffffff,residual,offset);
    }
    if(lane==0) {
        norms[row]=(float)norm;
        errors[row]=(float)sqrt(residual)*(1.0f+4.0f*1.192092896e-7f*c);
    }
}

extern "C" __global__ void screened_half_min(
    const unsigned short* dot, const float* norms, const float* errors,
    const float* z, float* nearest, int n, int p, int c,
    int query_start, int query_count, int ref_start, int ref_count) {
    int row=blockIdx.x*4+threadIdx.y;
    if (row>=query_count) return;
    int query=query_start+row, image=ref_start+blockIdx.y, lane=threadIdx.x;
    if (query/p==image) {
        if (lane==0) nearest[query*n+image]=3.402823466e+38F;
        return;
    }
    int columns=ref_count*p;
    float approx_best=3.402823466e+38F;
    int selected=0;
    for (int patch=lane; patch<p; patch+=32) {
        float dp=half_float(dot[row*columns+blockIdx.y*p+patch]);
        float d=isfinite(dp) ? norms[query]+norms[image*p+patch]-2.0f*dp : 0.0f;
        if (d<approx_best) { approx_best=d; selected=patch; }
    }
    for (int offset=16; offset>0; offset/=2) {
        float other=__shfl_down_sync(0xffffffff,approx_best,offset);
        int index=__shfl_down_sync(0xffffffff,selected,offset);
        if (lane+offset<32 && other<approx_best) { approx_best=other; selected=index; }
    }
    selected=__shfl_sync(0xffffffff,selected,0);
    float best=warp_distance(z,query,image*p+selected,c,lane);
    float gamma=(c*1.192092896e-7f)/(1.0f-c*1.192092896e-7f);
    float qnorm=sqrtf(norms[query])*(1.0f+gamma), qerror=errors[query];
    for (int base=0; base<p; base+=32) {
        int patch=base+lane, reference=image*p+patch;
        bool eligible=false;
        if (patch<p && patch!=selected) {
            float dp=half_float(dot[row*columns+blockIdx.y*p+patch]);
            float rnorm=sqrtf(norms[reference])*(1.0f+gamma), rerror=errors[reference];
            // Cauchy-Schwarz bounds feature quantization, gamma bounds FP32
            // accumulation, and half rounding includes subnormal output error.
            float dot_error=qerror*rnorm+rerror*qnorm+qerror*rerror
                +gamma*(qnorm+qerror)*(rnorm+rerror)
                +0.000489f*fabsf(dp)+3.0e-8f;
            float lower=norms[query]+norms[reference]-2.0f*dp
                -2.0f*dot_error-gamma*(norms[query]+norms[reference]);
            eligible=!isfinite(dp) || lower<=best*(1.0f+2.0f*gamma);
        }
        unsigned mask=__ballot_sync(0xffffffff,eligible);
        while (mask) {
            int winner=__ffs(mask)-1;
            best=fminf(best,warp_distance(z,query,image*p+base+winner,c,lane));
            mask &= mask-1;
        }
    }
    if (lane==0) nearest[query*n+image]=sqrtf(best);
}

extern "C" __global__ void screened_half_min_transposed(
    const unsigned short* dot, const float* norms, const float* errors,
    const float* z, float* nearest, int n, int p, int c,
    int query_start, int query_count, int ref_start, int ref_count) {
    int row=blockIdx.x*4+threadIdx.y;
    if (row>=query_count) return;
    int query=query_start+row, image=ref_start+blockIdx.y, lane=threadIdx.x;
    if (query/p==image) {
        if (lane==0) nearest[query*n+image]=3.402823466e+38F;
        return;
    }
    int columns=ref_count*p;
    float approx_best=3.402823466e+38F;
    int selected=0;
    for (int patch=lane; patch<p; patch+=32) {
        float dp=half_float(dot[(blockIdx.y*p+patch)*query_count+row]);
        float d=isfinite(dp) ? norms[query]+norms[image*p+patch]-2.0f*dp : 0.0f;
        if (d<approx_best) { approx_best=d; selected=patch; }
    }
    for (int offset=16; offset>0; offset/=2) {
        float other=__shfl_down_sync(0xffffffff,approx_best,offset);
        int index=__shfl_down_sync(0xffffffff,selected,offset);
        if (lane+offset<32 && other<approx_best) { approx_best=other; selected=index; }
    }
    selected=__shfl_sync(0xffffffff,selected,0);
    float best=warp_distance(z,query,image*p+selected,c,lane);
    float gamma=(c*1.192092896e-7f)/(1.0f-c*1.192092896e-7f);
    float qnorm=sqrtf(norms[query])*(1.0f+gamma), qerror=errors[query];
    for (int base=0; base<p; base+=32) {
        int patch=base+lane, reference=image*p+patch;
        bool eligible=false;
        if (patch<p && patch!=selected) {
            float dp=half_float(dot[(blockIdx.y*p+patch)*query_count+row]);
            float rnorm=sqrtf(norms[reference])*(1.0f+gamma), rerror=errors[reference];
            // Cauchy-Schwarz bounds feature quantization, gamma bounds FP32
            // accumulation, and half rounding includes subnormal output error.
            float dot_error=qerror*rnorm+rerror*qnorm+qerror*rerror
                +gamma*(qnorm+qerror)*(rnorm+rerror)
                +0.000489f*fabsf(dp)+3.0e-8f;
            float lower=norms[query]+norms[reference]-2.0f*dp
                -2.0f*dot_error-gamma*(norms[query]+norms[reference]);
            eligible=!isfinite(dp) || lower<=best*(1.0f+2.0f*gamma);
        }
        unsigned mask=__ballot_sync(0xffffffff,eligible);
        while (mask) {
            int winner=__ffs(mask)-1;
            best=fminf(best,warp_distance(z,query,image*p+base+winner,c,lane));
            mask &= mask-1;
        }
    }
    if (lane==0) nearest[query*n+image]=sqrtf(best);
}

extern "C" __global__ void nearest_image(
    const float* z, float* nearest, int n, int p, int c) {
    const int lane = threadIdx.x;
    const int row = threadIdx.y;
    const int tid = row * 32 + lane;
    const int query = blockIdx.x * 8 + row;
    const int ref_image = blockIdx.y;
    const int query_image = query / p;
    const bool valid = query < n*p;
    if (blockIdx.x * 8 >= n*p) return;
    __shared__ float qs[8*32];
    __shared__ float rs[32*32];
    float best = 3.402823466e+38F;
    for (int base = 0; base < p; base += 32) {
        float squared = 0.0f;
        for (int channel = 0; channel < c; channel += 32) {
            const int qc = channel + lane;
            qs[tid] = (valid && qc < c) ? z[query*c + qc] : 0.0f;
            for (int k = tid; k < 32*32; k += 256) {
                int patch = base + k/32;
                int rc = channel + k%32;
                rs[k] = (patch < p && rc < c) ? z[(ref_image*p + patch)*c + rc] : 0.0f;
            }
            __syncthreads();
            #pragma unroll
            for (int k = 0; k < 32; ++k) {
                float difference = qs[row*32 + k] - rs[lane*32 + k];
                squared = fmaf(difference, difference, squared);
            }
            __syncthreads();
        }
        if (base + lane < p) best = fminf(best, squared);
    }
    for (int offset = 16; offset > 0; offset /= 2)
        best = fminf(best, __shfl_down_sync(0xffffffff, best, offset));
    if (lane == 0 && valid)
        nearest[query*n + ref_image] = (query_image == ref_image) ? 3.402823466e+38F : sqrtf(best);
}

extern "C" __global__ void reduce_dot(
    const float* dot, const float* norms, const float* z, float* nearest, int n, int p, int c) {
    int query = blockIdx.x * 4 + threadIdx.y;
    int ref_image = blockIdx.y;
    int lane = threadIdx.x;
    int total = n*p;
    // Warp-uniform exit: no shared-memory barriers in this kernel.
    if (query >= total) return;
    float best = 3.402823466e+38F;
    if (query/p != ref_image) {
        for (int q=lane; q<p; q+=32) {
            int reference = ref_image*p + q;
            float d = norms[query] + norms[reference] - 2.0f*dot[query*total+reference];
            // Avoid cancellation around duplicate/near-zero distances. The
            // conservative threshold routes close pairs through direct differences.
            float threshold = 32.0f*1.192092896e-7f*c*(norms[query]+norms[reference]);
            if (d <= threshold) {
                d = 0.0f;
                for (int channel=0; channel<c; ++channel) {
                    float difference = z[query*c+channel]-z[reference*c+channel];
                    d = fmaf(difference,difference,d);
                }
            }
            best = fminf(best, fmaxf(0.0f,d));
        }
    }
    for (int offset=16; offset>0; offset/=2)
        best = fminf(best,__shfl_down_sync(0xffffffff,best,offset));
    if (lane==0)
        nearest[query*n+ref_image] = (query/p==ref_image) ? 3.402823466e+38F : sqrtf(best);
}

extern "C" __global__ void trimmed_score(
    const float* nearest, double* scores, int n, int total, int a, int b) {
    int query = blockIdx.x * blockDim.x + threadIdx.x;
    if (query >= total) return;
    float values[32];
    for (int j = 0; j < n; ++j) {
        float v = nearest[query*n + j];
        int k = j;
        while (k > 0 && values[k-1] > v) { values[k] = values[k-1]; --k; }
        values[k] = v;
    }
    float sum = 0.0f;
    for (int j = a; j < b; ++j) sum += values[j];
    scores[query] = (double)(sum / (b-a));
}
extern "C" __global__ void reduce_dot_tile(
    const float* dot, const float* norms, const float* z, float* nearest,
    int n, int p, int c, int query_start, int query_count, int ref_start, int ref_count) {
    int row = blockIdx.x * 4 + threadIdx.y;
    if (row >= query_count) return;
    int query = query_start + row;
    int ref_image = ref_start + blockIdx.y;
    int lane = threadIdx.x;
    int columns = ref_count*p;
    float best = 3.402823466e+38F;
    if (query/p != ref_image) {
        for (int base=0; base<p; base+=32) {
            int patch = base + lane;
            int reference = ref_image*p + patch;
            float d = 3.402823466e+38F;
            float threshold = 0.0f;
            if (patch<p) {
                d = norms[query] + norms[reference]
                    - 2.0f*dot[row*columns + blockIdx.y*p + patch];
                threshold = 32.0f*1.192092896e-7f*c*(norms[query]+norms[reference]);
            }
            // Cooperative channel loads avoid the 1024-channel scalar loop's
            // strided memory traffic. Every suspect pair is still recomputed.
            unsigned mask = __ballot_sync(0xffffffff,patch<p && d<=threshold);
            while (mask) {
                int selected = __ffs(mask)-1;
                int selected_ref = ref_image*p+base+selected;
                float sum = 0.0f;
                for (int channel=lane; channel<c; channel+=32) {
                    float difference = z[query*c+channel]-z[selected_ref*c+channel];
                    sum = fmaf(difference,difference,sum);
                }
                for (int offset=16; offset>0; offset/=2)
                    sum += __shfl_down_sync(0xffffffff,sum,offset);
                float exact = __shfl_sync(0xffffffff,sum,0);
                if (lane==selected) d = exact;
                mask &= mask-1;
            }
            best = fminf(best,fmaxf(0.0f,d));
        }
    }
    for (int offset=16; offset>0; offset/=2)
        best = fminf(best,__shfl_down_sync(0xffffffff,best,offset));
    if (lane==0)
        nearest[query*n+ref_image] = (query/p==ref_image) ? 3.402823466e+38F : sqrtf(best);
}
'''


def MSM(Z, device=None, topmin_min=0, topmin_max=0.3):
    if Z.ndim != 3 or not Z.is_cuda or Z.dtype != torch.float32:
        raise ValueError('Expected CUDA FP32 features with shape (N,P,C).')
    n,p,c = Z.shape
    if not (n >= 2 and p > 0 and c > 0):
        raise ValueError('Requires N>=2, P>0, C>0.')
    a = int((n-1)*topmin_min) if topmin_min < 1 else int(topmin_min)
    b = int((n-1)*topmin_max) if topmin_max < 1 else int(topmin_max)
    a,b = sorted((a,b))
    if not 0 <= a < b <= n-1:
        raise ValueError('Require valid nonempty trimmed ranks 0<=a<b<=N-1.')
    with torch.cuda.device(Z.device):
        z = Z.contiguous()
        kernels = compile_kernels(CUDA_SOURCE, ['nearest_image','reduce_dot','reduce_dot_tile','trimmed_score',
                                  'half_feature_bounds','screened_half_min','screened_half_min_transposed'], Z.device.index)
        nearest = torch.empty((n*p,n),dtype=torch.float32,device=Z.device)
        scores = torch.empty((n,p),dtype=torch.float64,device=Z.device)
        # Real 1369x1024 features always use GEMM. Query/reference tiling
        # bounds scratch without subdividing the shared reference image pool.
        if p>=64:
            flat = z.reshape(n*p,c)
            screened = 512<=c<=4096
            if screened:
                center = flat.mean(dim=0)
                half = (flat-center).half()
                norms = torch.empty(n*p,dtype=torch.float32,device=Z.device)
                errors = torch.empty_like(norms)
                launch(kernels['half_feature_bounds'],((n*p+7)//8,1,1),(32,8,1),
                       [flat,half,center,norms,errors],[n*p,c],Z.device)
                query_tile = min(4096,n*p,max(1,MAX_DOT_BYTES//(4*p)))
                ref_tile = min(n,max(1,MAX_DOT_BYTES//(4*p*query_tile)))
                old_reduction = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
                try:
                    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
                    # Image-aligned square tiles cover the upper triangle once.
                    # Both directions retain per-reference-IMAGE minima; diagonal
                    # tiles are evaluated normally, including self exclusion.
                    image_block = int((MAX_DOT_BYTES // 2)**0.5) // p
                    if image_block >= 1 and n > 8:
                        image_block = min(n, image_block)
                        for left in range(0,n,image_block):
                            left_images = min(image_block,n-left)
                            left_count = left_images*p
                            for right in range(left,n,image_block):
                                right_images = min(image_block,n-right)
                                right_count = right_images*p
                                dot = torch.mm(half[left*p:(left+left_images)*p],
                                               half[right*p:(right+right_images)*p].T)
                                launch(kernels['screened_half_min'],((left_count+3)//4,right_images,1),(32,4,1),
                                       [dot,norms,errors,flat,nearest],
                                       [n,p,c,left*p,left_count,right,right_images],Z.device)
                                if right != left:
                                    launch(kernels['screened_half_min_transposed'],((right_count+3)//4,left_images,1),(32,4,1),
                                           [dot,norms,errors,flat,nearest],
                                           [n,p,c,right*p,right_count,left,left_images],Z.device)
                                del dot  # Release this tile before allocating the next; scratch <=256 MiB.
                    else:
                        for start in range(0,n*p,query_tile):
                            count = min(query_tile,n*p-start)
                            for ref in range(0,n,ref_tile):
                                refs = min(ref_tile,n-ref)
                                dot = torch.mm(half[start:start+count],half[ref*p:(ref+refs)*p].T)
                                launch(kernels['screened_half_min'],((count+3)//4,refs,1),(32,4,1),
                                       [dot,norms,errors,z,nearest],[n,p,c,start,count,ref,refs],Z.device)
                                del dot
                finally:
                    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = old_reduction
                if n<=32:
                    launch(kernels['trimmed_score'],((n*p+127)//128,1,1),(128,1,1),
                           [nearest,scores],[n,n*p,a,b],Z.device)
                else:
                    scores = nearest.topk(b,dim=1,largest=False,sorted=True).values[:,a:b].mean(1).double().view(n,p)
                return scores
            norms = (flat*flat).sum(1)
            if n*p<=8192 and n>=12:
                dot = torch.mm(flat,flat.T)
                launch(kernels['reduce_dot'],((n*p+3)//4,n,1),(32,4,1),
                       [dot,norms,z,nearest],[n,p,c],Z.device)
            else:
                scratch_bytes = MAX_DOT_BYTES
                query_tile = min(4096,n*p,max(1,scratch_bytes//(4*p)))
                ref_tile = min(n,max(1,scratch_bytes//(4*p*query_tile)))
                for start in range(0,n*p,query_tile):
                    count = min(query_tile,n*p-start)
                    for ref in range(0,n,ref_tile):
                        refs = min(ref_tile,n-ref)
                        dot = torch.mm(flat[start:start+count],flat[ref*p:(ref+refs)*p].T)
                        launch(kernels['reduce_dot_tile'],((count+3)//4,refs,1),(32,4,1),
                               [dot,norms,z,nearest],[n,p,c,start,count,ref,refs],Z.device)
                        del dot
        else:
            launch(kernels['nearest_image'],((n*p+7)//8,n,1),(32,8,1),
                   [z,nearest],[n,p,c],Z.device)
        if n<=32:
            launch(kernels['trimmed_score'],((n*p+127)//128,1,1),(128,1,1),
                   [nearest,scores],[n,n*p,a,b],Z.device)
        else:
            scores = nearest.topk(b,dim=1,largest=False,sorted=True).values[:,a:b].mean(1).double().view(n,p)
        return scores
