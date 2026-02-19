/*
 * 02_rmsnorm.cu — RMSNorm with warp-level reductions
 *
 * RMSNorm (Root Mean Square Normalization) is used in nanochat/LLaMA instead of LayerNorm:
 *   RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
 *
 * Where x is one row (of dimension D), and gamma is a learnable scale vector.
 *
 * Unlike LayerNorm, RMSNorm doesn't subtract the mean — simpler and equally effective.
 *
 * This kernel introduces WARP-LEVEL REDUCTIONS using __shfl_down_sync:
 *   - Instead of shared memory + __syncthreads for reductions,
 *   - Threads within a warp (32 threads) can directly read each other's registers
 *   - __shfl_down_sync shuffles a value from a higher-lane thread to a lower-lane
 *   - This builds a tree reduction within the warp in 5 steps (log2(32) = 5)
 *   - Much faster than shared memory because: no shared mem bank conflicts,
 *     no __syncthreads, and register-to-register is fastest possible
 *
 * Compile: nvcc -arch=sm_89 -O2 -o 02_rmsnorm 02_rmsnorm.cu
 */

#include "common.h"

#define WARP_SIZE 32

/* ========== CPU reference ========== */
void rmsnorm_cpu(const float *x, const float *gamma, float *out,
                 int N, int D, float eps) {
    for (int row = 0; row < N; row++) {
        const float *xr = x + row * D;
        float *or_ = out + row * D;

        /* Compute mean of squares */
        float sum_sq = 0.0f;
        for (int j = 0; j < D; j++)
            sum_sq += xr[j] * xr[j];
        float rms = sqrtf(sum_sq / D + eps);

        /* Normalize and scale */
        for (int j = 0; j < D; j++)
            or_[j] = xr[j] / rms * gamma[j];
    }
}

/* ========== Warp reduction helper ========== */
/*
 * __shfl_down_sync(mask, val, delta):
 *   - Within a warp, thread `lane` gets the value from thread `lane + delta`
 *   - mask = 0xffffffff means all 32 lanes participate
 *
 * Tree reduction for sum:
 *   Step 1 (delta=16): lanes 0-15 add lanes 16-31
 *   Step 2 (delta=8):  lanes 0-7 add lanes 8-15
 *   Step 3 (delta=4):  lanes 0-3 add lanes 4-7
 *   Step 4 (delta=2):  lanes 0-1 add lanes 2-3
 *   Step 5 (delta=1):  lane 0 adds lane 1
 *   Result: lane 0 has the sum of all 32 values
 */
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;  // only lane 0 has the correct result
}

/* ========== Kernel 1: Simple (one block per row, shared memory reduction) ========== */
__global__ void rmsnorm_v1(const float *x, const float *gamma, float *out,
                           int D, float eps) {
    extern __shared__ float smem[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    const float *xr = x + row * D;
    float *or_ = out + row * D;

    /* Phase 1: Each thread computes partial sum of squares */
    float partial_sq = 0.0f;
    for (int j = tid; j < D; j += blockDim.x)
        partial_sq += xr[j] * xr[j];

    smem[tid] = partial_sq;
    __syncthreads();

    /* Phase 2: Block-level reduction in shared memory */
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    float rms = sqrtf(smem[0] / D + eps);

    /* Phase 3: Normalize and scale */
    for (int j = tid; j < D; j += blockDim.x)
        or_[j] = xr[j] / rms * gamma[j];
}

/* ========== Kernel 2: Warp shuffle reduction (faster) ========== */
__global__ void rmsnorm_v2(const float *x, const float *gamma, float *out,
                           int D, float eps) {
    /*
     * Strategy: One block per row. Each thread processes D/blockDim.x elements.
     * Reduction uses warp shuffles + shared memory for cross-warp communication.
     */
    extern __shared__ float smem[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    const float *xr = x + row * D;
    float *or_ = out + row * D;

    /* Phase 1: Each thread accumulates partial sum of squares */
    float partial_sq = 0.0f;
    for (int j = tid; j < D; j += blockDim.x)
        partial_sq += xr[j] * xr[j];

    /* Phase 2: Warp-level reduction */
    partial_sq = warp_reduce_sum(partial_sq);

    /* Phase 3: First thread of each warp writes to shared memory */
    if (lane == 0) smem[warp_id] = partial_sq;
    __syncthreads();

    /* Phase 4: First warp reduces across warps */
    if (warp_id == 0) {
        partial_sq = (tid < num_warps) ? smem[tid] : 0.0f;
        partial_sq = warp_reduce_sum(partial_sq);
    }

    /* Broadcast RMS to all threads via shared memory */
    if (tid == 0) smem[0] = sqrtf(partial_sq / D + eps);
    __syncthreads();
    float rms = smem[0];

    /* Phase 5: Normalize and scale */
    for (int j = tid; j < D; j += blockDim.x)
        or_[j] = xr[j] / rms * gamma[j];
}

int main() {
    printf("=== Phase 2.2: RMSNorm with Warp Reductions ===\n\n");

    int N = 4096;    // batch size (number of rows)
    int D = 1024;    // hidden dim (nanochat-like)
    float eps = 1e-6f;

    size_t x_bytes = (size_t)N * D * sizeof(float);
    size_t gamma_bytes = D * sizeof(float);

    float *h_x = (float *)malloc(x_bytes);
    float *h_gamma = (float *)malloc(gamma_bytes);
    float *h_ref = (float *)malloc(x_bytes);
    float *h_gpu = (float *)malloc(x_bytes);
    fill_random(h_x, N * D);
    fill_random_positive(h_gamma, D);

    /* CPU reference */
    rmsnorm_cpu(h_x, h_gamma, h_ref, N, D, eps);

    /* GPU */
    float *d_x, *d_gamma, *d_out;
    CHECK_CUDA(cudaMalloc(&d_x, x_bytes));
    CHECK_CUDA(cudaMalloc(&d_gamma, gamma_bytes));
    CHECK_CUDA(cudaMalloc(&d_out, x_bytes));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, x_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_gamma, h_gamma, gamma_bytes, cudaMemcpyHostToDevice));

    int block_size = 256;

    /* --- v1: Shared memory reduction --- */
    rmsnorm_v1<<<N, block_size, block_size * sizeof(float)>>>(d_x, d_gamma, d_out, D, eps);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_gpu, d_out, x_bytes, cudaMemcpyDeviceToHost));
    verify(h_gpu, h_ref, N * D, 1e-4f, "RMSNorm-v1 (smem)");

    /* --- v2: Warp shuffle reduction --- */
    int smem_size = (block_size / WARP_SIZE) * sizeof(float);
    rmsnorm_v2<<<N, block_size, smem_size>>>(d_x, d_gamma, d_out, D, eps);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_gpu, d_out, x_bytes, cudaMemcpyDeviceToHost));
    verify(h_gpu, h_ref, N * D, 1e-4f, "RMSNorm-v2 (warp)");

    /* --- Benchmark --- */
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    printf("\nBenchmark (N=%d, D=%d):\n", N, D);

    /* v1 */
    rmsnorm_v1<<<N, block_size, block_size * sizeof(float)>>>(d_x, d_gamma, d_out, D, eps);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(start));
    for (int r = 0; r < 100; r++)
        rmsnorm_v1<<<N, block_size, block_size * sizeof(float)>>>(d_x, d_gamma, d_out, D, eps);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float v1_ms;
    CHECK_CUDA(cudaEventElapsedTime(&v1_ms, start, stop));
    v1_ms /= 100;

    /* v2 */
    rmsnorm_v2<<<N, block_size, smem_size>>>(d_x, d_gamma, d_out, D, eps);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(start));
    for (int r = 0; r < 100; r++)
        rmsnorm_v2<<<N, block_size, smem_size>>>(d_x, d_gamma, d_out, D, eps);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float v2_ms;
    CHECK_CUDA(cudaEventElapsedTime(&v2_ms, start, stop));
    v2_ms /= 100;

    float total_bytes_rw = (2.0f * N * D + D) * sizeof(float);  // read x+gamma, write out
    printf("  v1 (shared mem): %.3f ms, %.1f GB/s\n", v1_ms, total_bytes_rw / v1_ms / 1e6);
    printf("  v2 (warp shfl):  %.3f ms, %.1f GB/s\n", v2_ms, total_bytes_rw / v2_ms / 1e6);
    printf("  Speedup: %.2fx\n", v1_ms / v2_ms);

    /* Cleanup */
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_gamma));
    CHECK_CUDA(cudaFree(d_out));
    free(h_x); free(h_gamma); free(h_ref); free(h_gpu);

    printf("\n=== RMSNorm complete! ===\n");
    return 0;
}
