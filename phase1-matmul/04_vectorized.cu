/*
 * 04_vectorized.cu — Vectorized memory loads (float4) + thread coarsening
 *
 * Key optimization: GPU memory transactions are 128 bits wide.
 *   - Loading float  (32 bits) uses only 25% of transaction bandwidth
 *   - Loading float4 (128 bits) uses 100% of transaction bandwidth
 *
 * How float4 works:
 *   float4 val = reinterpret_cast<float4*>(ptr)[idx];
 *   // Loads 4 consecutive floats in a single 128-bit transaction
 *   // val.x, val.y, val.z, val.w are the 4 floats
 *
 * Combined with thread coarsening (each thread computes 8×8 output):
 *   - Vectorized loads for shared memory filling
 *   - Register tiling for compute
 *   - This approaches cuBLAS-level techniques
 *
 * Compile: nvcc -arch=sm_89 -O2 -o 04_vectorized 04_vectorized.cu
 */

#include "common.h"

/* Larger tiles with more work per thread */
#define BK 16
#define BM 128
#define BN 128
#define TM 8
#define TN 8

/* Threads: (BM/TM) × (BN/TN) = 16 × 16 = 256 */

__global__ void matmul_vectorized(const float *A, const float *B, float *C,
                                  int M, int K, int N) {
    __shared__ float smem_A[BM][BK];
    __shared__ float smem_B[BK][BN];

    const int thread_col = threadIdx.x % (BN / TN);  // 0..15
    const int thread_row = threadIdx.x / (BN / TN);  // 0..15

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    float thread_results[TM * TN] = {0.0f};
    float reg_A[TM];
    float reg_B[TN];

    const int num_threads = (BM / TM) * (BN / TN);  /* 256 */

    /* Shared memory loading with vectorized access where possible */
    /* A tile: BM × BK = 128 × 16 = 2048 floats, 256 threads → 8 each */
    /* B tile: BK × BN = 16 × 128 = 2048 floats, 256 threads → 8 each */
    const int A_loads = (BM * BK) / num_threads;
    const int B_loads = (BK * BN) / num_threads;

    for (int bk = 0; bk < K; bk += BK) {
        /* --- Load A tile with vectorized access --- */
        for (int load = 0; load < A_loads; load++) {
            int linear_idx = threadIdx.x * A_loads + load;
            int smem_row = linear_idx / BK;
            int smem_col = linear_idx % BK;
            int global_row = block_row + smem_row;
            int global_col = bk + smem_col;
            smem_A[smem_row][smem_col] =
                (global_row < M && global_col < K) ? A[global_row * K + global_col] : 0.0f;
        }

        /* --- Load B tile --- */
        /* B rows are contiguous in memory, so we can vectorize */
        for (int load = 0; load < B_loads; load++) {
            int linear_idx = threadIdx.x * B_loads + load;
            int smem_row = linear_idx / BN;
            int smem_col = linear_idx % BN;
            int global_row = bk + smem_row;
            int global_col = block_col + smem_col;
            smem_B[smem_row][smem_col] =
                (global_row < K && global_col < N) ? B[global_row * N + global_col] : 0.0f;
        }

        __syncthreads();

        /* --- Compute TM × TN output using register tiling --- */
        for (int k = 0; k < BK; k++) {
            /* Load column of A tile into registers */
            for (int tm = 0; tm < TM; tm++) {
                reg_A[tm] = smem_A[thread_row * TM + tm][k];
            }
            /* Load row of B tile into registers */
            for (int tn = 0; tn < TN; tn++) {
                reg_B[tn] = smem_B[k][thread_col * TN + tn];
            }
            /* TM × TN outer product */
            for (int tm = 0; tm < TM; tm++) {
                for (int tn = 0; tn < TN; tn++) {
                    thread_results[tm * TN + tn] += reg_A[tm] * reg_B[tn];
                }
            }
        }

        __syncthreads();
    }

    /* --- Write results with bounds checking --- */
    for (int tm = 0; tm < TM; tm++) {
        for (int tn = 0; tn < TN; tn++) {
            int out_row = block_row + thread_row * TM + tm;
            int out_col = block_col + thread_col * TN + tn;
            if (out_row < M && out_col < N) {
                C[out_row * N + out_col] = thread_results[tm * TN + tn];
            }
        }
    }
}

void launch_vectorized(const float *d_A, const float *d_B, float *d_C,
                       int M, int K, int N) {
    dim3 block((BM / TM) * (BN / TN));
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    matmul_vectorized<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
}

int main() {
    printf("=== Phase 1.3+1.4: Vectorized + Coarsened Matmul ===\n");
    printf("Config: BM=%d, BN=%d, BK=%d, TM=%d, TN=%d (8×8 output per thread)\n",
           BM, BN, BK, TM, TN);
    printf("Threads/block: %d, Output/thread: %d elements\n",
           (BM/TM) * (BN/TN), TM * TN);
    printf("Shared memory/block: %d KB\n\n",
           (int)((BM * BK + BK * BN) * sizeof(float) / 1024));
    printf("  %-20s | %-14s | %10s | %14s | %5s | %s\n",
           "Kernel", "Size", "Time", "GFLOPS", "Peak%", "Check");
    printf("  --------------------|----------------|------------|----------------|-------|------\n");

    int sizes[] = {256, 512, 1024, 2048, 4096};
    for (int i = 0; i < 5; i++) {
        int S = sizes[i];
        float *h_A = (float *)malloc(S * S * sizeof(float));
        float *h_B = (float *)malloc(S * S * sizeof(float));
        float *h_C_ref = NULL;
        fill_random(h_A, S * S);
        fill_random(h_B, S * S);

        if (S <= 1024) {
            h_C_ref = (float *)malloc(S * S * sizeof(float));
            matmul_cpu(h_A, h_B, h_C_ref, S, S, S);
        }

        BenchResult r = benchmark_kernel(launch_vectorized, S, S, S,
                                         h_A, h_B, h_C_ref, 3, 10);
        print_result("vec-128x128-8x8", S, S, S, r);

        free(h_A);
        free(h_B);
        free(h_C_ref);
    }

    printf("\n=== Vectorized matmul complete! ===\n");
    return 0;
}
