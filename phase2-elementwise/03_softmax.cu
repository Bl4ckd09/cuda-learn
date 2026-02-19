/*
 * 03_softmax.cu — Row-wise softmax: standard and online algorithms
 *
 * softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
 *
 * Three implementations:
 *   v1: Three-pass (find max → compute exp sum → normalize)
 *   v2: Two-pass (find max & exp sum in parallel → normalize)
 *   v3: ONLINE softmax — single pass, the key Flash Attention insight!
 *
 * Online softmax (Milakov & Gimelshein, 2018):
 *   Maintains a running max `m` and running denominator `d`.
 *   When processing new element x_i:
 *     m_new = max(m, x_i)
 *     d = d * exp(m_old - m_new) + exp(x_i - m_new)
 *   Final: softmax[i] = exp(x_i - m_final) / d
 *
 *   This is THE key insight for Flash Attention: you never need to
 *   see the entire row to compute softmax. You can process TILES
 *   and update the running statistics as you go.
 *
 * Compile: nvcc -arch=sm_89 -O2 -o 03_softmax 03_softmax.cu
 */

#include "common.h"

#define WARP_SIZE 32

/* ========== CPU reference ========== */
void softmax_cpu(const float *x, float *out, int N, int D) {
    for (int row = 0; row < N; row++) {
        const float *xr = x + row * D;
        float *or_ = out + row * D;

        /* Pass 1: find max */
        float max_val = -FLT_MAX;
        for (int j = 0; j < D; j++)
            max_val = fmaxf(max_val, xr[j]);

        /* Pass 2: compute exp sum */
        float sum = 0.0f;
        for (int j = 0; j < D; j++) {
            or_[j] = expf(xr[j] - max_val);
            sum += or_[j];
        }

        /* Pass 3: normalize */
        for (int j = 0; j < D; j++)
            or_[j] /= sum;
    }
}

/* ========== Warp helpers ========== */
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

/* Block-level reduction using warps + shared memory */
__device__ float block_reduce_max(float val, float *smem) {
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    val = warp_reduce_max(val);
    if (lane == 0) smem[warp_id] = val;
    __syncthreads();

    if (warp_id == 0) {
        val = (threadIdx.x < num_warps) ? smem[threadIdx.x] : -FLT_MAX;
        val = warp_reduce_max(val);
    }
    /* Broadcast to all threads */
    if (threadIdx.x == 0) smem[0] = val;
    __syncthreads();
    return smem[0];
}

__device__ float block_reduce_sum(float val, float *smem) {
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    val = warp_reduce_sum(val);
    if (lane == 0) smem[warp_id] = val;
    __syncthreads();

    if (warp_id == 0) {
        val = (threadIdx.x < num_warps) ? smem[threadIdx.x] : 0.0f;
        val = warp_reduce_sum(val);
    }
    if (threadIdx.x == 0) smem[0] = val;
    __syncthreads();
    return smem[0];
}

/* ========== v1: Three-pass softmax ========== */
__global__ void softmax_v1(const float *x, float *out, int D) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float *xr = x + row * D;
    float *or_ = out + row * D;

    /* Pass 1: find max */
    float local_max = -FLT_MAX;
    for (int j = tid; j < D; j += blockDim.x)
        local_max = fmaxf(local_max, xr[j]);
    float max_val = block_reduce_max(local_max, smem);

    /* Pass 2: exp and sum */
    float local_sum = 0.0f;
    for (int j = tid; j < D; j += blockDim.x) {
        float e = expf(xr[j] - max_val);
        or_[j] = e;
        local_sum += e;
    }
    float sum_val = block_reduce_sum(local_sum, smem);

    /* Pass 3: normalize */
    for (int j = tid; j < D; j += blockDim.x)
        or_[j] /= sum_val;
}

/* ========== v2: Two-pass softmax (fused max+exp) ========== */
__global__ void softmax_v2(const float *x, float *out, int D) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float *xr = x + row * D;
    float *or_ = out + row * D;

    /* Pass 1: find max */
    float local_max = -FLT_MAX;
    for (int j = tid; j < D; j += blockDim.x)
        local_max = fmaxf(local_max, xr[j]);
    float max_val = block_reduce_max(local_max, smem);

    /* Pass 2: exp, sum, and normalize in one pass */
    float local_sum = 0.0f;
    for (int j = tid; j < D; j += blockDim.x) {
        float e = expf(xr[j] - max_val);
        or_[j] = e;
        local_sum += e;
    }
    float sum_val = block_reduce_sum(local_sum, smem);

    float inv_sum = 1.0f / sum_val;
    for (int j = tid; j < D; j += blockDim.x)
        or_[j] *= inv_sum;
}

/* ========== v3: Online softmax (single-pass over data!) ========== */
/*
 * THE KEY INSIGHT FOR FLASH ATTENTION:
 *
 * Standard softmax needs to see ALL elements to find max first.
 * Online softmax updates incrementally:
 *
 *   For each new element x_i:
 *     m_new = max(m_old, x_i)
 *     d_new = d_old * exp(m_old - m_new) + exp(x_i - m_new)
 *
 *   The rescaling factor exp(m_old - m_new) corrects all previous
 *   contributions when the max changes.
 *
 * In Flash Attention, each "element" is actually a TILE of scores,
 * and the output O is also rescaled: O_new = O_old * exp(m_old - m_new) / d_new
 */
__global__ void softmax_v3_online(const float *x, float *out, int D) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float *xr = x + row * D;
    float *or_ = out + row * D;

    /* Each thread maintains its own running max and sum */
    float local_max = -FLT_MAX;
    float local_sum = 0.0f;

    /* Single pass: update max and sum together */
    for (int j = tid; j < D; j += blockDim.x) {
        float val = xr[j];
        float old_max = local_max;
        local_max = fmaxf(local_max, val);
        /* Rescale previous sum when max changes */
        local_sum = local_sum * expf(old_max - local_max) + expf(val - local_max);
    }

    /* Now we need to combine across threads.
     * Each thread has (local_max, local_sum).
     * To combine thread A (m_a, d_a) and thread B (m_b, d_b):
     *   m_new = max(m_a, m_b)
     *   d_new = d_a * exp(m_a - m_new) + d_b * exp(m_b - m_new)
     */

    /* Warp-level combination */
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other_max = __shfl_down_sync(0xffffffff, local_max, offset);
        float other_sum = __shfl_down_sync(0xffffffff, local_sum, offset);
        float new_max = fmaxf(local_max, other_max);
        local_sum = local_sum * expf(local_max - new_max)
                  + other_sum * expf(other_max - new_max);
        local_max = new_max;
    }

    /* Cross-warp combination via shared memory */
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    /* Two arrays in shared memory: one for max, one for sum */
    float *smem_max = smem;
    float *smem_sum = smem + num_warps;

    if (lane == 0) {
        smem_max[warp_id] = local_max;
        smem_sum[warp_id] = local_sum;
    }
    __syncthreads();

    /* First warp reduces across all warps.
     * ALL 32 lanes must participate in __shfl_down_sync(0xffffffff, ...),
     * so lanes >= num_warps use neutral values (max=-inf, sum=0). */
    if (warp_id == 0) {
        local_max = (tid < num_warps) ? smem_max[tid] : -FLT_MAX;
        local_sum = (tid < num_warps) ? smem_sum[tid] : 0.0f;
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            float other_max = __shfl_down_sync(0xffffffff, local_max, offset);
            float other_sum = __shfl_down_sync(0xffffffff, local_sum, offset);
            float new_max = fmaxf(local_max, other_max);
            local_sum = local_sum * expf(local_max - new_max)
                      + other_sum * expf(other_max - new_max);
            local_max = new_max;
        }
    }

    /* Broadcast final max and sum */
    if (tid == 0) {
        smem_max[0] = local_max;
        smem_sum[0] = local_sum;
    }
    __syncthreads();

    float final_max = smem_max[0];
    float final_sum = smem_sum[0];

    /* Second pass: compute final softmax values */
    for (int j = tid; j < D; j += blockDim.x)
        or_[j] = expf(xr[j] - final_max) / final_sum;
}

int main() {
    printf("=== Phase 2.3: Softmax (Standard and Online) ===\n\n");

    int N = 2048;
    int D = 512;  // sequence length for attention softmax
    size_t bytes = (size_t)N * D * sizeof(float);

    float *h_x = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);
    float *h_gpu = (float *)malloc(bytes);
    fill_random(h_x, N * D);

    softmax_cpu(h_x, h_ref, N, D);

    float *d_x, *d_out;
    CHECK_CUDA(cudaMalloc(&d_x, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));

    int block_size = 256;
    int num_warps = block_size / WARP_SIZE;
    int smem_basic = num_warps * sizeof(float);
    int smem_online = 2 * num_warps * sizeof(float);

    printf("Correctness tests (N=%d, D=%d):\n", N, D);

    /* v1: Three-pass */
    softmax_v1<<<N, block_size, smem_basic>>>(d_x, d_out, D);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_gpu, d_out, bytes, cudaMemcpyDeviceToHost));
    verify(h_gpu, h_ref, N * D, 1e-5f, "softmax-v1 (3-pass)");

    /* v2: Two-pass */
    softmax_v2<<<N, block_size, smem_basic>>>(d_x, d_out, D);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_gpu, d_out, bytes, cudaMemcpyDeviceToHost));
    verify(h_gpu, h_ref, N * D, 1e-5f, "softmax-v2 (2-pass)");

    /* v3: Online */
    softmax_v3_online<<<N, block_size, smem_online>>>(d_x, d_out, D);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_gpu, d_out, bytes, cudaMemcpyDeviceToHost));
    verify(h_gpu, h_ref, N * D, 1e-5f, "softmax-v3 (online)");

    /* Benchmark */
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    printf("\nBenchmark (N=%d, D=%d):\n", N, D);
    int runs = 20;

    /* v1 benchmark */
    softmax_v1<<<N, block_size, smem_basic>>>(d_x, d_out, D);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(start));
    for (int r = 0; r < runs; r++)
        softmax_v1<<<N, block_size, smem_basic>>>(d_x, d_out, D);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    { float ms; CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop)); ms /= runs;
      printf("  v1 (3-pass):         %.3f ms, %.1f GB/s\n", ms, 2.0f * bytes / ms / 1e6); }

    /* v2 benchmark */
    softmax_v2<<<N, block_size, smem_basic>>>(d_x, d_out, D);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(start));
    for (int r = 0; r < runs; r++)
        softmax_v2<<<N, block_size, smem_basic>>>(d_x, d_out, D);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    { float ms; CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop)); ms /= runs;
      printf("  v2 (2-pass):         %.3f ms, %.1f GB/s\n", ms, 2.0f * bytes / ms / 1e6); }

    /* v3 benchmark */
    softmax_v3_online<<<N, block_size, smem_online>>>(d_x, d_out, D);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(start));
    for (int r = 0; r < runs; r++)
        softmax_v3_online<<<N, block_size, smem_online>>>(d_x, d_out, D);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    { float ms; CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop)); ms /= runs;
      printf("  v3 (online):         %.3f ms, %.1f GB/s\n", ms, 2.0f * bytes / ms / 1e6); }

    printf("\nOnline softmax reads the input data only ONCE.\n");
    printf("This is the principle behind Flash Attention's tiled approach.\n");

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_out));
    free(h_x); free(h_ref); free(h_gpu);

    printf("\n=== Softmax complete! ===\n");
    return 0;
}
