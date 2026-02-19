/*
 * 02_tiled.cu — Tiled matmul with shared memory
 *
 * THE fundamental GPU optimization. Instead of each thread reading
 * an entire row/column from global memory, threads COOPERATE:
 *
 *   1. All threads in a block load one TILE_SIZE x TILE_SIZE tile
 *      of A and B into shared memory (__shared__)
 *   2. __syncthreads() — wait for all threads to finish loading
 *   3. Each thread computes partial dot product from shared memory
 *   4. Repeat for the next tile along K dimension
 *
 * Why it's faster:
 *   - Shared memory is ~100x faster than global memory (on-chip SRAM)
 *   - Each global memory value is read once but used TILE_SIZE times
 *   - Reduces global memory reads by factor of TILE_SIZE
 *   - For TILE_SIZE=16: 16x fewer global reads → huge bandwidth savings
 *
 * Visual (TILE_SIZE=4 for illustration):
 *
 *   A:          K →          B:        N →
 *   ┌────┬────┬───┐         ┌────┬────┐
 *   │tile│tile│...│    M    │tile│tile│
 *   │ 0  │ 1  │   │    ↓   │ 0  │ 1  │
 *   ├────┼────┤   │         ├────┼────┤
 *   │    │    │   │         │tile│tile│
 *   └────┴────┴───┘         │ 2  │ 3  │
 *                            └────┴────┘
 *
 *   For each (tile_A, tile_B) pair along K:
 *     - Load tile_A into smem_A[TILE][TILE]
 *     - Load tile_B into smem_B[TILE][TILE]
 *     - Each thread accumulates: sum += smem_A[ty][k] * smem_B[k][tx]
 *
 * Compile: nvcc -arch=sm_89 -O2 -o 02_tiled 02_tiled.cu
 */

#include "common.h"

#define TILE_SIZE 16

__global__ void matmul_tiled(const float *A, const float *B, float *C,
                             int M, int K, int N) {
    /*
     * Shared memory: fast on-chip SRAM, shared by all threads in a block.
     * Two tiles: one for the A sub-matrix, one for the B sub-matrix.
     * Each is TILE_SIZE x TILE_SIZE = 16x16 = 256 floats = 1 KB each.
     */
    __shared__ float smem_A[TILE_SIZE][TILE_SIZE];
    __shared__ float smem_B[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;  // column within the block/tile
    int ty = threadIdx.y;  // row within the block/tile

    int row = blockIdx.y * TILE_SIZE + ty;  // global row in C
    int col = blockIdx.x * TILE_SIZE + tx;  // global col in C

    float sum = 0.0f;

    /* Iterate over tiles along the K dimension */
    int num_tiles = CEIL_DIV(K, TILE_SIZE);
    for (int t = 0; t < num_tiles; t++) {
        /* --- Load tile from global to shared memory --- */
        // Each thread loads one element of tile_A and one element of tile_B
        int a_col = t * TILE_SIZE + tx;
        int b_row = t * TILE_SIZE + ty;

        // Bounds checking: load 0 if out of bounds
        smem_A[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        smem_B[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        /* --- Synchronize: all threads must finish loading before computing --- */
        __syncthreads();

        /* --- Compute partial dot product from shared memory --- */
        // This is the fast part: shared memory reads instead of global
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += smem_A[ty][k] * smem_B[k][tx];
        }

        /* --- Synchronize again before loading next tile --- */
        // Without this, fast threads might start loading the next tile
        // while slow threads are still reading the current tile
        __syncthreads();
    }

    /* Write result to global memory */
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void launch_tiled(const float *d_A, const float *d_B, float *d_C,
                  int M, int K, int N) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(CEIL_DIV(N, TILE_SIZE), CEIL_DIV(M, TILE_SIZE));
    matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
}

/* Also test with TILE_SIZE=32 */
#define TILE_SIZE_32 32

__global__ void matmul_tiled_32(const float *A, const float *B, float *C,
                                int M, int K, int N) {
    __shared__ float smem_A[TILE_SIZE_32][TILE_SIZE_32];
    __shared__ float smem_B[TILE_SIZE_32][TILE_SIZE_32];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE_32 + ty;
    int col = blockIdx.x * TILE_SIZE_32 + tx;

    float sum = 0.0f;
    int num_tiles = CEIL_DIV(K, TILE_SIZE_32);

    for (int t = 0; t < num_tiles; t++) {
        int a_col = t * TILE_SIZE_32 + tx;
        int b_row = t * TILE_SIZE_32 + ty;

        smem_A[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        smem_B[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE_32; k++) {
            sum += smem_A[ty][k] * smem_B[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void launch_tiled_32(const float *d_A, const float *d_B, float *d_C,
                     int M, int K, int N) {
    dim3 block(TILE_SIZE_32, TILE_SIZE_32);
    dim3 grid(CEIL_DIV(N, TILE_SIZE_32), CEIL_DIV(M, TILE_SIZE_32));
    matmul_tiled_32<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
}

int main() {
    printf("=== Phase 1.2: Tiled Matmul (Shared Memory) ===\n");
    printf("RTX 4070 FP32 peak: %.1f TFLOPS\n\n", RTX4070_PEAK_TFLOPS);
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

        BenchResult r16 = benchmark_kernel(launch_tiled, S, S, S,
                                           h_A, h_B, h_C_ref, 3, 10);
        print_result("tiled-16x16", S, S, S, r16);

        BenchResult r32 = benchmark_kernel(launch_tiled_32, S, S, S,
                                           h_A, h_B, h_C_ref, 3, 10);
        print_result("tiled-32x32", S, S, S, r32);

        free(h_A);
        free(h_B);
        free(h_C_ref);
    }

    printf("\nShared memory per block:\n");
    printf("  16x16 tiles: 2 * 16*16*4 = 2048 bytes (2 KB)\n");
    printf("  32x32 tiles: 2 * 32*32*4 = 8192 bytes (8 KB)\n");
    printf("  RTX 4070 has 48 KB shared memory per block → plenty\n");

    printf("\n=== Tiled matmul complete! ===\n");
    return 0;
}
