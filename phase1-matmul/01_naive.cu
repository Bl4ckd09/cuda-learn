/*
 * 01_naive.cu — Naive matmul: one thread per output element
 *
 * C[M,N] = A[M,K] @ B[K,N]
 *
 * Each thread computes: C[row][col] = dot(A[row,:], B[:,col])
 *
 * Why it's slow:
 *   - Each thread reads K values from A (row) + K values from B (column)
 *   - B column access has stride N → poor memory coalescing
 *   - No data reuse between threads (adjacent threads share row of A
 *     but both read it separately from global memory)
 *   - Arithmetic intensity: 2K FLOPs / (2K * 4 bytes) = 0.5 FLOP/byte
 *     RTX 4070 needs ~58 FLOP/byte for compute-bound → we're bandwidth-bound
 *
 * Compile: nvcc -arch=sm_89 -O2 -o 01_naive 01_naive.cu
 */

#include "common.h"

#define BLOCK_SIZE 16

__global__ void matmul_naive(const float *A, const float *B, float *C,
                             int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void launch_naive(const float *d_A, const float *d_B, float *d_C,
                  int M, int K, int N) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(CEIL_DIV(N, BLOCK_SIZE), CEIL_DIV(M, BLOCK_SIZE));
    matmul_naive<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
}

int main() {
    printf("=== Phase 1.1: Naive Matmul ===\n");
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

        BenchResult r = benchmark_kernel(launch_naive, S, S, S,
                                         h_A, h_B, h_C_ref, 3, 10);
        print_result("naive", S, S, S, r);

        free(h_A);
        free(h_B);
        free(h_C_ref);
    }

    printf("\n=== Naive matmul complete! ===\n");
    return 0;
}
