/*
 * 03_coarsened.cu — Thread coarsening: each thread computes multiple output elements
 *
 * Instead of 1 thread → 1 output element, each thread computes a TM×TN tile.
 * This is sometimes called "register tiling" or "thread tiling".
 *
 * Why it's faster than basic tiling:
 *   1. More compute per thread → better hides memory latency
 *   2. Values accumulate in registers (fastest memory)
 *   3. Shared memory reads are reused across the TM×TN tile
 *   4. Fewer blocks needed → less scheduling overhead
 *
 * Memory access pattern:
 *   - Each thread loads TM values from smem_A (one column of its tile)
 *   - Each thread loads TN values from smem_B (one row of its tile)
 *   - Computes TM * TN output elements using these TM + TN values
 *   - Arithmetic intensity: TM * TN FMADs per (TM + TN) shared mem reads
 *
 * With TM=TN=8: 64 FMADs per 16 smem reads = 4 FLOP/read
 * This is the key pattern used in cuBLAS-level kernels.
 *
 * Compile: nvcc -arch=sm_89 -O2 -o 03_coarsened 03_coarsened.cu
 */

#include "common.h"

/*
 * Parameters (tunable):
 *   BK = tile size along K dimension (shared memory tile width)
 *   BM = block tile height in M dimension
 *   BN = block tile width in N dimension
 *   TM = number of rows each thread computes
 *   TN = number of cols each thread computes
 *
 * Block has (BN/TN) × (BM/TM) threads.
 * Each thread computes TM × TN output elements.
 * BM × BN elements are computed per block.
 */

#define BK 16
#define BM 64
#define BN 64
#define TM 4
#define TN 4

/* Threads per block: (BN/TN) × (BM/TM) = 16 × 16 = 256 */

__global__ void matmul_coarsened(const float *A, const float *B, float *C,
                                 int M, int K, int N) {
    __shared__ float smem_A[BM][BK];
    __shared__ float smem_B[BK][BN];

    /* Thread position within the block */
    const int thread_col = threadIdx.x % (BN / TN);  // 0..15
    const int thread_row = threadIdx.x / (BN / TN);  // 0..15

    /* Block's starting position in the output matrix */
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    /* Register storage for the thread's output tile */
    float thread_results[TM * TN] = {0.0f};

    /* Registers for the A and B values used in computation */
    float reg_A[TM];
    float reg_B[TN];

    /* Total threads in the block */
    const int num_threads = (BM / TM) * (BN / TN);  /* 256 */

    /* Precompute loading indices for shared memory */
    /* Each thread loads BM*BK/num_threads elements of A */
    /* and BK*BN/num_threads elements of B */
    const int A_loads_per_thread = (BM * BK) / num_threads;
    const int B_loads_per_thread = (BK * BN) / num_threads;

    /* Iterate over tiles along K */
    for (int bk = 0; bk < K; bk += BK) {
        /* --- Load A tile (BM × BK) into shared memory --- */
        for (int load = 0; load < A_loads_per_thread; load++) {
            int linear_idx = threadIdx.x * A_loads_per_thread + load;
            int smem_row = linear_idx / BK;
            int smem_col = linear_idx % BK;
            int global_row = block_row + smem_row;
            int global_col = bk + smem_col;
            smem_A[smem_row][smem_col] =
                (global_row < M && global_col < K) ? A[global_row * K + global_col] : 0.0f;
        }

        /* --- Load B tile (BK × BN) into shared memory --- */
        for (int load = 0; load < B_loads_per_thread; load++) {
            int linear_idx = threadIdx.x * B_loads_per_thread + load;
            int smem_row = linear_idx / BN;
            int smem_col = linear_idx % BN;
            int global_row = bk + smem_row;
            int global_col = block_col + smem_col;
            smem_B[smem_row][smem_col] =
                (global_row < K && global_col < N) ? B[global_row * N + global_col] : 0.0f;
        }

        __syncthreads();

        /* --- Compute: each thread computes TM × TN outputs --- */
        for (int k = 0; k < BK; k++) {
            /* Load TM values from smem_A column k into registers */
            for (int tm = 0; tm < TM; tm++) {
                reg_A[tm] = smem_A[thread_row * TM + tm][k];
            }
            /* Load TN values from smem_B row k into registers */
            for (int tn = 0; tn < TN; tn++) {
                reg_B[tn] = smem_B[k][thread_col * TN + tn];
            }
            /* Outer product: TM × TN multiply-adds */
            for (int tm = 0; tm < TM; tm++) {
                for (int tn = 0; tn < TN; tn++) {
                    thread_results[tm * TN + tn] += reg_A[tm] * reg_B[tn];
                }
            }
        }

        __syncthreads();
    }

    /* --- Write TM × TN results to global memory --- */
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

void launch_coarsened(const float *d_A, const float *d_B, float *d_C,
                      int M, int K, int N) {
    dim3 block((BM / TM) * (BN / TN));  /* 256 threads in 1D */
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    matmul_coarsened<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
}

int main() {
    printf("=== Phase 1.3-1.4: Coarsened Matmul (Thread Tiling) ===\n");
    printf("Config: BM=%d, BN=%d, BK=%d, TM=%d, TN=%d\n", BM, BN, BK, TM, TN);
    printf("Threads/block: %d, Output/thread: %dx%d=%d elements\n",
           (BM/TM) * (BN/TN), TM, TN, TM*TN);
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

        BenchResult r = benchmark_kernel(launch_coarsened, S, S, S,
                                         h_A, h_B, h_C_ref, 3, 10);
        print_result("coarsened-4x4", S, S, S, r);

        free(h_A);
        free(h_B);
        free(h_C_ref);
    }

    printf("\n=== Coarsened matmul complete! ===\n");
    return 0;
}
