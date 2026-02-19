/*
 * 01_naive_attention.cu — Naive scaled dot-product attention
 *
 * Three separate steps, fully materializing the N×N attention matrix:
 *   1. S = Q @ K^T / sqrt(d)     — matmul kernel
 *   2. P = softmax(S, dim=-1)    — row-wise softmax kernel
 *   3. O = P @ V                 — matmul kernel
 *
 * This is what you'd get if you wrote attention "the obvious way."
 * It works, but uses O(N^2) memory for the attention matrix.
 *
 * At N=2048, d=64: S is 2048^2 * 4 bytes = 16 MB per head.
 * With 12 heads and batch 32, that's 6 GB — half your VRAM!
 *
 * Compile: nvcc -arch=sm_89 -O2 -lcublas -o 01_naive_attention 01_naive_attention.cu
 */

#include "common.h"
#include <cublas_v2.h>

#define WARP_SIZE 32

/* ═══════════ Kernel 1: S = Q @ K^T * scale ═══════════
 *
 * This is a standard matmul: S[i][j] = sum_k Q[i][k] * K[j][k] * scale
 * Note: K is NOT transposed in memory. We access K[j][k] which gives us K^T.
 * Each thread computes one element of the N×N score matrix.
 */
__global__ void matmul_QKT(const float *Q, const float *K, float *S,
                           int N, int d, float scale) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  /* query index */
    int col = blockIdx.x * blockDim.x + threadIdx.x;  /* key index */

    if (row < N && col < N) {
        float dot = 0.0f;
        for (int k = 0; k < d; k++)
            dot += Q[row * d + k] * K[col * d + k];
        S[row * N + col] = dot * scale;
    }
}

/* ═══════════ Kernel 2: P = softmax(S, dim=-1) ═══════════
 *
 * Row-wise softmax over the N×N score matrix.
 * One block per row, using warp+shared memory reductions.
 */
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

__global__ void softmax_rows(float *S, float *P, int N) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;

    /* Pass 1: find max */
    float local_max = -FLT_MAX;
    for (int j = tid; j < N; j += blockDim.x)
        local_max = fmaxf(local_max, S[row * N + j]);
    float max_val = block_reduce_max(local_max, smem);

    /* Pass 2: exp and sum */
    float local_sum = 0.0f;
    for (int j = tid; j < N; j += blockDim.x) {
        float e = expf(S[row * N + j] - max_val);
        P[row * N + j] = e;
        local_sum += e;
    }
    float sum_val = block_reduce_sum(local_sum, smem);

    /* Pass 3: normalize */
    float inv_sum = 1.0f / sum_val;
    for (int j = tid; j < N; j += blockDim.x)
        P[row * N + j] *= inv_sum;
}

/* ═══════════ Kernel 3: O = P @ V ═══════════
 *
 * Standard matmul: O[i][k] = sum_j P[i][j] * V[j][k]
 * P is [N, N], V is [N, d], O is [N, d]
 */
__global__ void matmul_PV(const float *P, const float *V, float *O,
                          int N, int d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < d) {
        float acc = 0.0f;
        for (int j = 0; j < N; j++)
            acc += P[row * N + j] * V[j * d + col];
        O[row * d + col] = acc;
    }
}

/* ═══════════ Main ═══════════ */
int main() {
    printf("=== Phase 3.1: Naive Attention (Materialized N×N) ===\n\n");

    /* Dimensions — single head, single batch */
    int N = 1024;   /* sequence length */
    int d = 64;     /* head dimension */
    float scale = 1.0f / sqrtf((float)d);

    size_t qkv_bytes = (size_t)N * d * sizeof(float);  /* Q, K, V, O each */
    size_t attn_bytes = (size_t)N * N * sizeof(float);  /* S and P */

    printf("Config: N=%d (seq len), d=%d (head dim)\n", N, d);
    printf("Memory: Q,K,V,O = %.1f KB each, S,P = %.1f MB each\n",
           qkv_bytes / 1024.0f, attn_bytes / (1024.0f * 1024.0f));
    printf("Total attention matrix: %.1f MB (this is what Flash Attention eliminates)\n\n",
           2.0f * attn_bytes / (1024.0f * 1024.0f));

    /* Allocate host */
    float *h_Q = (float *)malloc(qkv_bytes);
    float *h_K = (float *)malloc(qkv_bytes);
    float *h_V = (float *)malloc(qkv_bytes);
    float *h_O_ref = (float *)malloc(qkv_bytes);
    float *h_O_gpu = (float *)malloc(qkv_bytes);

    srand(42);
    fill_random(h_Q, N * d);
    fill_random(h_K, N * d);
    fill_random(h_V, N * d);

    /* CPU reference */
    attention_cpu(h_Q, h_K, h_V, h_O_ref, N, d);

    /* Allocate device */
    float *d_Q, *d_K, *d_V, *d_O, *d_S, *d_P;
    CHECK_CUDA(cudaMalloc(&d_Q, qkv_bytes));
    CHECK_CUDA(cudaMalloc(&d_K, qkv_bytes));
    CHECK_CUDA(cudaMalloc(&d_V, qkv_bytes));
    CHECK_CUDA(cudaMalloc(&d_O, qkv_bytes));
    CHECK_CUDA(cudaMalloc(&d_S, attn_bytes));
    CHECK_CUDA(cudaMalloc(&d_P, attn_bytes));

    CHECK_CUDA(cudaMemcpy(d_Q, h_Q, qkv_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K, qkv_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V, qkv_bytes, cudaMemcpyHostToDevice));

    /* ── Method 1: Custom naive kernels ── */
    printf("Correctness:\n");

    /* Kernel 1: S = Q @ K^T * scale */
    dim3 block_mm(16, 16);
    dim3 grid_qkt(CEIL_DIV(N, 16), CEIL_DIV(N, 16));
    matmul_QKT<<<grid_qkt, block_mm>>>(d_Q, d_K, d_S, N, d, scale);

    /* Kernel 2: P = softmax(S) */
    int softmax_bs = 256;
    int smem_size = (softmax_bs / WARP_SIZE) * sizeof(float);
    softmax_rows<<<N, softmax_bs, smem_size>>>(d_S, d_P, N);

    /* Kernel 3: O = P @ V */
    dim3 grid_pv(CEIL_DIV(d, 16), CEIL_DIV(N, 16));
    matmul_PV<<<grid_pv, block_mm>>>(d_P, d_V, d_O, N, d);

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_O_gpu, d_O, qkv_bytes, cudaMemcpyDeviceToHost));
    verify(h_O_gpu, h_O_ref, N * d, 1e-3f, "naive-attention (custom)");

    /* ── Method 2: cuBLAS matmuls + custom softmax ── */
    cublasHandle_t handle;
    cublasCreate(&handle);

    /* S = Q @ K^T * scale using cuBLAS
     * Row-major trick: compute S^T = K @ Q^T, which cuBLAS sees as column-major S.
     * cublasSgemm(handle, op_A, op_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
     * We want: S[N,N] = Q[N,d] @ K[N,d]^T * scale
     * Row-major trick: S = (K^T)^T @ Q^T viewed as col-major
     *   => cublasSgemm(N, N, d, scale, K, d, Q, d, 0, S, N)
     *   But simpler: cublasSgemm(CUBLAS_OP_T, CUBLAS_OP_N, N, N, d, scale, K, d, Q, d, 0, S, N)
     */
    float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                N, N, d, &scale,
                d_K, d,    /* K^T: op_A=T, so A=K[N,d] row-major, lda=d */
                d_Q, d,    /* Q:   op_B=N, so B=Q[N,d] row-major, lda=d */
                &beta, d_S, N);

    /* Softmax (reuse custom kernel) */
    softmax_rows<<<N, softmax_bs, smem_size>>>(d_S, d_P, N);

    /* O = P @ V using cuBLAS
     * We want: O[N,d] = P[N,N] @ V[N,d]
     * Row-major trick: cublasSgemm with swapped args
     *   => cublasSgemm(CUBLAS_OP_N, CUBLAS_OP_N, d, N, N, 1, V, d, P, N, 0, O, d)
     */
    float alpha = 1.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                d, N, N, &alpha,
                d_V, d,    /* V[N,d] row-major, lda=d */
                d_P, N,    /* P[N,N] row-major, lda=N */
                &beta, d_O, d);

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_O_gpu, d_O, qkv_bytes, cudaMemcpyDeviceToHost));
    verify(h_O_gpu, h_O_ref, N * d, 1e-3f, "naive-attention (cuBLAS)");

    /* ── Benchmark ── */
    printf("\nBenchmark (N=%d, d=%d):\n", N, d);
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    int runs = 50;

    /* Custom naive */
    matmul_QKT<<<grid_qkt, block_mm>>>(d_Q, d_K, d_S, N, d, scale);
    softmax_rows<<<N, softmax_bs, smem_size>>>(d_S, d_P, N);
    matmul_PV<<<grid_pv, block_mm>>>(d_P, d_V, d_O, N, d);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int r = 0; r < runs; r++) {
        matmul_QKT<<<grid_qkt, block_mm>>>(d_Q, d_K, d_S, N, d, scale);
        softmax_rows<<<N, softmax_bs, smem_size>>>(d_S, d_P, N);
        matmul_PV<<<grid_pv, block_mm>>>(d_P, d_V, d_O, N, d);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_custom;
    CHECK_CUDA(cudaEventElapsedTime(&ms_custom, start, stop));
    ms_custom /= runs;

    /* cuBLAS */
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, N, d, &scale, d_K, d, d_Q, d, &beta, d_S, N);
    softmax_rows<<<N, softmax_bs, smem_size>>>(d_S, d_P, N);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, d, N, N, &alpha, d_V, d, d_P, N, &beta, d_O, d);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int r = 0; r < runs; r++) {
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, N, d, &scale, d_K, d, d_Q, d, &beta, d_S, N);
        softmax_rows<<<N, softmax_bs, smem_size>>>(d_S, d_P, N);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, d, N, N, &alpha, d_V, d, d_P, N, &beta, d_O, d);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_cublas;
    CHECK_CUDA(cudaEventElapsedTime(&ms_cublas, start, stop));
    ms_cublas /= runs;

    printf("  Custom naive:     %.3f ms\n", ms_custom);
    printf("  cuBLAS + softmax: %.3f ms  (%.1fx speedup)\n",
           ms_cublas, ms_custom / ms_cublas);

    /* Memory analysis */
    size_t total_attn_mem = 2 * attn_bytes;  /* S + P */
    printf("\nMemory analysis:\n");
    printf("  Attention matrices (S+P): %.1f MB\n", total_attn_mem / (1024.0f * 1024.0f));
    printf("  QKV+O:                    %.1f MB\n", 4.0f * qkv_bytes / (1024.0f * 1024.0f));
    printf("  Ratio (attn/qkv):         %.0fx\n",
           (float)total_attn_mem / (4.0f * qkv_bytes));
    printf("\n  At N=2048, 12 heads, batch 32:\n");
    printf("    S+P = %d * %d * 12 * 32 * 4 = %.1f GB\n",
           2048, 2048, 2.0f * 2048.0f * 2048.0f * 12.0f * 32.0f * 4.0f / (1024.0f * 1024.0f * 1024.0f));
    printf("  This is why Flash Attention exists!\n");

    /* Cleanup */
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    cublasDestroy(handle);
    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_O));
    CHECK_CUDA(cudaFree(d_S));
    CHECK_CUDA(cudaFree(d_P));
    free(h_Q); free(h_K); free(h_V);
    free(h_O_ref); free(h_O_gpu);

    printf("\n=== Naive attention complete! ===\n");
    return 0;
}
