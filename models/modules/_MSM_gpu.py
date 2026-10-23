"""Exact mutual scoring using bounded GEMM tiles for full MuSc image groups.

Candidate (claude_rankskip_20261021): v2 defers all exact work; anchors and
candidate survivors run only for pairs that can reach the trimmed ranks.

Every patch pair is evaluated. Dot-product scratch stays within 256 MiB;
near-zero pairs use direct differences to preserve accuracy.
"""
import torch
from models.modules._cuda_runtime import compile_kernels, launch
MAX_DOT_BYTES = 256*1024*1024
RECORDS_PER_PAIR = 4
# Diagnostic only: set to a list to record per-call candidate counts (synchronizes).
STATS = None

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

// Rank-deferred screening (v2). Anchor selection and certificate expressions as in
// screened_half_min, but exact work is normally deferred. Each pair records an upper
// bound on its evaluated anchor distance, a lower bound over all its patches, and
// appends every non-anchor patch whose lower bound does not exceed the anchor
// upper bound: a superset of the original eligible survivors, because the
// evaluated best never exceeds that upper bound. A pair whose records do not fit
// (or whose anchor bound is nonfinite) is refined in place exactly as in
// screened_half_min while its dot tile is resident, and is marked resolved (-2).
#define DEFERRED_BODY(DOT) \
    int row=blockIdx.x*4+threadIdx.y; \
    if (row>=query_count) return; \
    int query=query_start+row, image=ref_start+blockIdx.y, lane=threadIdx.x; \
    long slot=(long)query*n+image; \
    if (query/p==image) { \
        if (lane==0) { nearest[slot]=3.402823466e+38F; upper[slot]=3.402823466e+38F; anchor[slot]=-1; } \
        return; \
    } \
    int columns=ref_count*p; (void)columns; \
    float approx_best=3.402823466e+38F; \
    int selected=0; \
    for (int patch=lane; patch<p; patch+=32) { \
        float dp=half_float(DOT(patch)); \
        float d=isfinite(dp) ? norms[query]+norms[image*p+patch]-2.0f*dp : 0.0f; \
        if (d<approx_best) { approx_best=d; selected=patch; } \
    } \
    for (int offset=16; offset>0; offset/=2) { \
        float other=__shfl_down_sync(0xffffffff,approx_best,offset); \
        int index=__shfl_down_sync(0xffffffff,selected,offset); \
        if (lane+offset<32 && other<approx_best) { approx_best=other; selected=index; } \
    } \
    selected=__shfl_sync(0xffffffff,selected,0); \
    float gamma=(c*1.192092896e-7f)/(1.0f-c*1.192092896e-7f); \
    float qnorm=sqrtf(norms[query])*(1.0f+gamma), qerror=errors[query]; \
    float anchor_upper=3.402823466e+38F, anchor_lower=-3.402823466e+38F; \
    { \
        int reference=image*p+selected; \
        float dp=half_float(DOT(selected)); \
        float rnorm=sqrtf(norms[reference])*(1.0f+gamma), rerror=errors[reference]; \
        float dot_error=qerror*rnorm+rerror*qnorm+qerror*rerror \
            +gamma*(qnorm+qerror)*(rnorm+rerror)+0.000489f*fabsf(dp)+3.0e-8f; \
        float centre=norms[query]+norms[reference]-2.0f*dp; \
        float slack=2.0f*dot_error+gamma*(norms[query]+norms[reference]); \
        if (isfinite(dp)) { anchor_upper=centre+slack; anchor_lower=centre-slack; } \
    } \
    /* Upper bound on the FP32-evaluated anchor distance, hence on the evaluated best. */ \
    float computed_upper=anchor_upper*(1.0f+4.0f*gamma); \
    float cutoff=computed_upper*(1.0f+4.0f*gamma); \
    float pair_lower=3.402823466e+38F; \
    bool overflow=!isfinite(cutoff); \
    for (int base=0; base<p; base+=32) { \
        int patch=base+lane, reference=image*p+patch; \
        bool candidate=false; \
        float lower=0.0f; \
        if (patch<p) { \
            float dp=half_float(DOT(patch)); \
            float rnorm=sqrtf(norms[reference])*(1.0f+gamma), rerror=errors[reference]; \
            float dot_error=qerror*rnorm+rerror*qnorm+qerror*rerror \
                +gamma*(qnorm+qerror)*(rnorm+rerror) \
                +0.000489f*fabsf(dp)+3.0e-8f; \
            lower=norms[query]+norms[reference]-2.0f*dp \
                -2.0f*dot_error-gamma*(norms[query]+norms[reference]); \
            if (!isfinite(dp)) lower=-3.402823466e+38F; \
            pair_lower=fminf(pair_lower,lower); \
            candidate=patch!=selected && !(lower>cutoff); \
        } \
        unsigned mask=__ballot_sync(0xffffffff,candidate); \
        if (mask && !overflow) { \
            unsigned position=0; \
            if (lane==0) position=atomicAdd(count,(unsigned)__popc(mask)); \
            position=__shfl_sync(0xffffffff,position,0); \
            unsigned rank=__popc(mask&((1u<<lane)-1u)); \
            if (position+__popc(mask)>(unsigned)capacity) { \
                overflow=true; \
                /* Reserved slots below capacity must never hold stale records. */ \
                if (candidate && position+rank<(unsigned)capacity) records[3*(position+rank)]=-1; \
            } else if (candidate) { \
                records[3*(position+rank)]=(int)slot; \
                records[3*(position+rank)+1]=reference; \
                records[3*(position+rank)+2]=__float_as_int(lower); \
            } \
        } \
    } \
    if (!overflow) { \
        for (int offset=16; offset>0; offset/=2) \
            pair_lower=fminf(pair_lower,__shfl_xor_sync(0xffffffff,pair_lower,offset)); \
        if (lane==0) { upper[slot]=computed_upper; lower_out[slot]=pair_lower; anchor_lower_out[slot]=anchor_lower; anchor[slot]=selected; } \
        return; \
    } \
    float best=warp_distance(z,query,image*p+selected,c,lane); \
    for (int base=0; base<p; base+=32) { \
        int patch=base+lane, reference=image*p+patch; \
        bool eligible=false; \
        if (patch<p && patch!=selected) { \
            float dp=half_float(DOT(patch)); \
            float rnorm=sqrtf(norms[reference])*(1.0f+gamma), rerror=errors[reference]; \
            float dot_error=qerror*rnorm+rerror*qnorm+qerror*rerror \
                +gamma*(qnorm+qerror)*(rnorm+rerror) \
                +0.000489f*fabsf(dp)+3.0e-8f; \
            float lower=norms[query]+norms[reference]-2.0f*dp \
                -2.0f*dot_error-gamma*(norms[query]+norms[reference]); \
            eligible=!isfinite(dp) || lower<=best*(1.0f+2.0f*gamma); \
        } \
        unsigned mask=__ballot_sync(0xffffffff,eligible); \
        while (mask) { \
            int winner=__ffs(mask)-1; \
            best=fminf(best,warp_distance(z,query,image*p+base+winner,c,lane)); \
            mask &= mask-1; \
        } \
    } \
    if (lane==0) { nearest[slot]=sqrtf(best); upper[slot]=best; anchor[slot]=-2; }

#define FORWARD_DOT(patch) dot[row*columns+blockIdx.y*p+(patch)]
#define REVERSE_DOT(patch) dot[(blockIdx.y*p+(patch))*query_count+row]

extern "C" __global__ void deferred_half_min(
    const unsigned short* dot, const float* norms, const float* errors,
    const float* z, float* nearest, float* upper, float* lower_out, float* anchor_lower_out, int* anchor,
    unsigned* count, int* records, int capacity,
    int n, int p, int c, int query_start, int query_count, int ref_start, int ref_count) {
    DEFERRED_BODY(FORWARD_DOT)
}

extern "C" __global__ void deferred_half_min_transposed(
    const unsigned short* dot, const float* norms, const float* errors,
    const float* z, float* nearest, float* upper, float* lower_out, float* anchor_lower_out, int* anchor,
    unsigned* count, int* records, int capacity,
    int n, int p, int c, int query_start, int query_count, int ref_start, int ref_count) {
    DEFERRED_BODY(REVERSE_DOT)
}

// threshold[q] >= the b-th smallest evaluated minimum of query q. A pair whose lower
// bound exceeds it (with rounding margin) is strictly outside the b smallest minima;
// it stores sqrt(lower), also above every selected value, so the trimmed ranks see
// the same selected values. In a kept pair, a patch whose lower bound exceeds the
// threshold can only be its minimum if the pair lies outside the selected ranks,
// where any value >= the true minimum preserves them; such patches (anchor included)
// are not evaluated. A pair inside the ranks always evaluates its minimizing patch.
// Resolved and self pairs (negative codes) are final.
extern "C" __global__ void rank_anchor(
    const float* z, const float* threshold, const float* lower_in, const float* anchor_lower,
    int* anchor, float* best, float* nearest, int n, int p, int c) {
    int query=blockIdx.x*4+threadIdx.y;
    if (query>=n*p) return;
    int image=blockIdx.y, lane=threadIdx.x;
    long slot=(long)query*n+image;
    int code=anchor[slot];
    if (code<0) return;
    float gamma=(c*1.192092896e-7f)/(1.0f-c*1.192092896e-7f);
    float limit=threshold[query]*(1.0f+4.0f*gamma);
    float lower=lower_in[slot];
    if (lower>limit) {
        if (lane==0) { nearest[slot]=sqrtf(lower); anchor[slot]=-1; }
        return;
    }
    if (code & 0x40000000) {
        // Record overflow in the lane-per-query reverse consumer: exhaustive exact scan.
        float value=3.402823466e+38F;
        for (int patch=0; patch<p; ++patch)
            value=fminf(value,warp_distance(z,query,image*p+patch,c,lane));
        if (lane==0) best[slot]=value;
        return;
    }
    if (anchor_lower[slot]>limit) {
        if (lane==0) best[slot]=3.402823466e+38F;
        return;
    }
    float value=warp_distance(z,query,image*p+code,c,lane);
    if (lane==0) best[slot]=value;
}

// Candidates also apply the original survivor rule against the current best, which
// only decreases, so the evaluated set contains every original survivor.
extern "C" __global__ void rank_candidates(
    const float* z, const float* threshold, const int* anchor, const int* records,
    const unsigned* count, float* best, int capacity, int n, int p, int c) {
    unsigned total=min(*count,(unsigned)capacity), lane=threadIdx.x;
    float gamma=(c*1.192092896e-7f)/(1.0f-c*1.192092896e-7f);
    for (unsigned record=blockIdx.x*4+threadIdx.y; record<total; record+=gridDim.x*4) {
        int slot=records[3*record];
        if (slot<0 || anchor[slot]<0) continue;   // hole, rank-skipped or resolved (warp-uniform)
        int query=slot/n, reference=records[3*record+1];
        float lower=__int_as_float(records[3*record+2]);
        if (lower>threshold[query]*(1.0f+4.0f*gamma)) continue;
        if (lower>best[slot]*(1.0f+2.0f*gamma)) continue;
        float value=warp_distance(z,query,reference,c,lane);
        // Nonnegative floats order like their integer bit patterns.
        if (lane==0) atomicMin((int*)&best[slot],__float_as_int(value));
    }
}

extern "C" __global__ void rank_finish(
    const int* anchor, const float* best, float* nearest, int total) {
    long slot=(long)blockIdx.x*blockDim.x+threadIdx.x;
    if (slot>=total) return;
    if (anchor[slot]>=0) nearest[slot]=sqrtf(best[slot]);
}

// Reverse-direction deferred screening with one lane per query. The reverse tile is
// patch-major over query rows, so each patch iteration reads 32 adjacent queries'
// dots as one coalesced segment. Arithmetic per (query, patch) uses the original
// certificate expressions; no exact distance is evaluated here. Pairs whose records
// overflow (or whose anchor bound is nonfinite) are flagged for an exhaustive exact
// scan in rank_anchor, which yields the same minimum as any valid evaluated set.
extern "C" __global__ void deferred_reverse_lanes(
    const unsigned short* dot, const float* norms, const float* errors,
    float* nearest, float* upper, float* lower_out, float* anchor_lower_out, int* anchor,
    unsigned* count, int* records, int capacity,
    int n, int p, int c, int query_start, int query_count, int ref_start, int ref_count) {
    int lane=threadIdx.x, row=blockIdx.x*128+threadIdx.y*32+lane;
    if (blockIdx.x*128+threadIdx.y*32>=query_count) return;   // warp-uniform
    int image=ref_start+blockIdx.y, query=query_start+row;
    bool valid=row<query_count;
    bool active=valid && query/p!=image;
    long slot=(long)query*n+image;
    if (valid && !active) {
        nearest[slot]=3.402823466e+38F; upper[slot]=3.402823466e+38F; anchor[slot]=-1;
    }
    const unsigned short* column=dot+(long)blockIdx.y*p*query_count+(valid ? row : 0);
    float nq=active ? norms[query] : 0.0f;
    float approx_best=3.402823466e+38F;
    int selected=0;
    #pragma unroll 4
    for (int patch=0; patch<p; ++patch) {
        float dp=half_float(column[(long)patch*query_count]);
        float d=isfinite(dp) ? nq+norms[image*p+patch]-2.0f*dp : 0.0f;
        if (d<approx_best) { approx_best=d; selected=patch; }
    }
    float gamma=(c*1.192092896e-7f)/(1.0f-c*1.192092896e-7f);
    float qnorm=sqrtf(nq)*(1.0f+gamma), qerror=active ? errors[query] : 0.0f;
    float anchor_upper=3.402823466e+38F, anchor_lower=-3.402823466e+38F;
    {
        int reference=image*p+selected;
        float dp=half_float(column[(long)selected*query_count]);
        float rnorm=sqrtf(norms[reference])*(1.0f+gamma), rerror=errors[reference];
        float dot_error=qerror*rnorm+rerror*qnorm+qerror*rerror
            +gamma*(qnorm+qerror)*(rnorm+rerror)+0.000489f*fabsf(dp)+3.0e-8f;
        float centre=nq+norms[reference]-2.0f*dp;
        float slack=2.0f*dot_error+gamma*(nq+norms[reference]);
        if (isfinite(dp)) { anchor_upper=centre+slack; anchor_lower=centre-slack; }
    }
    float computed_upper=anchor_upper*(1.0f+4.0f*gamma);
    float cutoff=computed_upper*(1.0f+4.0f*gamma);
    float pair_lower=3.402823466e+38F;
    bool overflow=!isfinite(cutoff);
    for (int patch=0; patch<p; ++patch) {
        int reference=image*p+patch;
        float dp=half_float(column[(long)patch*query_count]);
        float rnorm=sqrtf(norms[reference])*(1.0f+gamma), rerror=errors[reference];
        float dot_error=qerror*rnorm+rerror*qnorm+qerror*rerror
            +gamma*(qnorm+qerror)*(rnorm+rerror)
            +0.000489f*fabsf(dp)+3.0e-8f;
        float lower=nq+norms[reference]-2.0f*dp
            -2.0f*dot_error-gamma*(nq+norms[reference]);
        if (!isfinite(dp)) lower=-3.402823466e+38F;
        pair_lower=fminf(pair_lower,lower);
        bool candidate=active && !overflow && patch!=selected && !(lower>cutoff);
        unsigned mask=__ballot_sync(0xffffffff,candidate);
        if (mask) {
            unsigned position=0;
            if (lane==__ffs(mask)-1) position=atomicAdd(count,(unsigned)__popc(mask));
            position=__shfl_sync(0xffffffff,position,__ffs(mask)-1);
            unsigned rank=__popc(mask&((1u<<lane)-1u));
            if (position+__popc(mask)>(unsigned)capacity) {
                if (candidate) {
                    overflow=true;
                    if (position+rank<(unsigned)capacity) records[3*(position+rank)]=-1;
                }
            } else if (candidate) {
                records[3*(position+rank)]=(int)slot;
                records[3*(position+rank)+1]=reference;
                records[3*(position+rank)+2]=__float_as_int(lower);
            }
        }
    }
    if (!active) return;
    upper[slot]=isfinite(computed_upper) ? computed_upper : 3.402823466e+38F;
    lower_out[slot]=pair_lower;
    anchor_lower_out[slot]=anchor_lower;
    anchor[slot]=selected | (overflow ? 0x40000000 : 0);
}

// rank_anchor with one warp per query looping over all reference images. The query
// row stays in registers (C<=1024, C%32==0), so each kept anchor reads only its
// reference row. Per-lane channel order (lane, lane+32, ...), fmaf accumulation and
// the shuffle-down tree are those of warp_distance, so evaluated values are bitwise
// identical. Decisions per pair are exactly rank_anchor's.
extern "C" __global__ void rank_anchor_rows(
    const float* z, const float* threshold, const float* lower_in, const float* anchor_lower,
    int* anchor, float* best, float* nearest, int n, int p, int c) {
    int query=blockIdx.x*4+threadIdx.y;
    if (query>=n*p) return;
    int lane=threadIdx.x;
    float row[32];
    #pragma unroll
    for (int k=0; k<32; ++k) row[k]=(32*k<c) ? z[(long)query*c+lane+32*k] : 0.0f;
    float gamma=(c*1.192092896e-7f)/(1.0f-c*1.192092896e-7f);
    float limit=threshold[query]*(1.0f+4.0f*gamma);
    for (int image=0; image<n; ++image) {
        long slot=(long)query*n+image;
        int code=anchor[slot];
        if (code<0) continue;
        float lower=lower_in[slot];
        if (lower>limit) {
            if (lane==0) { nearest[slot]=sqrtf(lower); anchor[slot]=-1; }
            continue;
        }
        if (code & 0x40000000) {
            float value=3.402823466e+38F;
            for (int patch=0; patch<p; ++patch)
                value=fminf(value,warp_distance(z,query,image*p+patch,c,lane));
            if (lane==0) best[slot]=value;
            continue;
        }
        if (anchor_lower[slot]>limit) {
            if (lane==0) best[slot]=3.402823466e+38F;
            continue;
        }
        const float* reference=z+(long)(image*p+code)*c+lane;
        float sum=0.0f;
        #pragma unroll
        for (int k=0; k<32; ++k) {
            if (32*k<c) {
                float difference=row[k]-reference[32*k];
                sum=fmaf(difference,difference,sum);
            }
        }
        for (int offset=16; offset>0; offset/=2)
            sum += __shfl_down_sync(0xffffffff,sum,offset);
        if (lane==0) best[slot]=sum;
    }
}

// Runs right after deferred_reverse_lanes on the same resident reverse tile. Pairs it
// flagged (record overflow or nonfinite anchor bound) are refined in place exactly as
// screened_half_min_transposed: original anchor distance, then every patch passing the
// original survivor certificate against the running best. They become resolved (-2).
// Unflagged pairs exit after one load.
extern "C" __global__ void reverse_overflow_refine(
    const unsigned short* dot, const float* norms, const float* errors,
    const float* z, float* nearest, float* upper, int* anchor,
    int n, int p, int c, int query_start, int query_count, int ref_start, int ref_count) {
    int row=blockIdx.x*4+threadIdx.y;
    if (row>=query_count) return;
    int query=query_start+row, image=ref_start+blockIdx.y, lane=threadIdx.x;
    long slot=(long)query*n+image;
    int code=anchor[slot];
    if (code<0 || !(code & 0x40000000)) return;
    int selected=code & 0x3fffffff;
    float gamma=(c*1.192092896e-7f)/(1.0f-c*1.192092896e-7f);
    float qnorm=sqrtf(norms[query])*(1.0f+gamma), qerror=errors[query];
    float best=warp_distance(z,query,image*p+selected,c,lane);
    for (int base=0; base<p; base+=32) {
        int patch=base+lane, reference=image*p+patch;
        bool eligible=false;
        if (patch<p && patch!=selected) {
            float dp=half_float(dot[(blockIdx.y*p+patch)*query_count+row]);
            float rnorm=sqrtf(norms[reference])*(1.0f+gamma), rerror=errors[reference];
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
    if (lane==0) { nearest[slot]=sqrtf(best); upper[slot]=best; anchor[slot]=-2; }
}
'''


def _strip_kernels(source, keep):
    """Remove extern "C" kernel functions not named in keep (brace matched); helpers stay."""
    import re
    out, position = [], 0
    for match in re.finditer(r'extern "C" __global__ void (\w+)', source):
        if match.start() < position or match.group(1) in keep:
            continue
        depth, i = 0, source.index('{', match.end())
        while True:
            depth += {'{': 1, '}': -1}.get(source[i], 0)
            i += 1
            if depth == 0:
                break
        out.append(source[position:match.start()]); position = i
    return ''.join(out) + source[position:]


ALL_KERNELS = ['nearest_image','reduce_dot','reduce_dot_tile','trimmed_score',
               'half_feature_bounds','screened_half_min','screened_half_min_transposed',
               'deferred_half_min','deferred_half_min_transposed','deferred_reverse_lanes','reverse_overflow_refine',
               'rank_anchor','rank_anchor_rows','rank_candidates','rank_finish']
CORE_KERNELS = ['trimmed_score','half_feature_bounds','deferred_half_min','deferred_reverse_lanes','reverse_overflow_refine',
                'rank_anchor','rank_anchor_rows','rank_candidates','rank_finish']
CORE_SOURCE = _strip_kernels(CUDA_SOURCE, set(CORE_KERNELS))


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
        core = p>=64 and 512<=c<=4096 and n>8 and int((MAX_DOT_BYTES // 2)**0.5)//p >= 1
        kernels = (compile_kernels(CORE_SOURCE, CORE_KERNELS, Z.device.index) if core else
                   compile_kernels(CUDA_SOURCE, ALL_KERNELS, Z.device.index))
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
                        upper = torch.empty((n*p,n),dtype=torch.float32,device=Z.device)
                        pair_lower = torch.empty_like(upper)
                        anchor_lower = torch.empty_like(upper)
                        anchor = torch.empty((n*p,n),dtype=torch.int32,device=Z.device)
                        capacity = int(RECORDS_PER_PAIR*n*p*n)
                        records = torch.empty(3*capacity,dtype=torch.int32,device=Z.device)
                        count = torch.zeros(1,dtype=torch.int32,device=Z.device)
                        for left in range(0,n,image_block):
                            left_images = min(image_block,n-left)
                            left_count = left_images*p
                            for right in range(left,n,image_block):
                                right_images = min(image_block,n-right)
                                right_count = right_images*p
                                dot = torch.mm(half[left*p:(left+left_images)*p],
                                               half[right*p:(right+right_images)*p].T)
                                launch(kernels['deferred_half_min'],((left_count+3)//4,right_images,1),(32,4,1),
                                       [dot,norms,errors,flat,nearest,upper,pair_lower,anchor_lower,anchor,count,records],
                                       [capacity,n,p,c,left*p,left_count,right,right_images],Z.device)
                                if right != left:
                                    launch(kernels['deferred_reverse_lanes'],((right_count+127)//128,left_images,1),(32,4,1),
                                           [dot,norms,errors,nearest,upper,pair_lower,anchor_lower,anchor,count,records],
                                           [capacity,n,p,c,right*p,right_count,left,left_images],Z.device)
                                    launch(kernels['reverse_overflow_refine'],((right_count+3)//4,left_images,1),(32,4,1),
                                           [dot,norms,errors,flat,nearest,upper,anchor],
                                           [n,p,c,right*p,right_count,left,left_images],Z.device)
                                del dot  # Release this tile before allocating the next; scratch <=256 MiB.
                        # Every pair now holds an upper bound on its evaluated minimum,
                        # so the b-th smallest bounds the b-th smallest exact minimum.
                        threshold = upper.kthvalue(b,dim=1).values
                        best = upper  # Storage reuse: rank_anchor overwrites every unskipped pair.
                        if STATS is not None:
                            resolved = int((anchor == -2).sum())
                        if c%32==0 and c<=1024:
                            launch(kernels['rank_anchor_rows'],((n*p+3)//4,1,1),(32,4,1),
                                   [flat,threshold,pair_lower,anchor_lower,anchor,best,nearest],[n,p,c],Z.device)
                        else:
                            launch(kernels['rank_anchor'],((n*p+3)//4,n,1),(32,4,1),
                                   [flat,threshold,pair_lower,anchor_lower,anchor,best,nearest],[n,p,c],Z.device)
                        launch(kernels['rank_candidates'],(82*16,1,1),(32,4,1),
                               [flat,threshold,anchor,records,count,best],[capacity,n,p,c],Z.device)
                        launch(kernels['rank_finish'],((n*p*n+255)//256,1,1),(256,1,1),
                               [anchor,best,nearest],[n*p*n],Z.device)
                        if STATS is not None:
                            STATS.append(dict(n=n, pairs=n*p*(n-1), capacity=capacity, records=int(count.item()),
                                              resolved_in_place=resolved, rank_skipped=int((anchor == -1).sum())-n*p))
                        del pair_lower, anchor_lower, anchor, threshold, records, best, upper
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
