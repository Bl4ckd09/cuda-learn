/*
 * 03_memory_timing.cu — Understanding GPU memory transfer costs
 *
 * Key insight: For many simple operations, the bottleneck is NOT
 * the kernel computation — it's the data transfer between CPU and GPU.
 *
 * Timeline of a typical GPU operation:
 *   1. cudaMalloc    (allocate GPU memory — one-time cost)
 *   2. H2D transfer  (copy data from CPU to GPU — bandwidth limited)
 *   3. Kernel exec   (actual GPU computation — usually fast)
 *   4. D2H transfer  (copy results back to CPU — bandwidth limited)
 *
 * PCIe bandwidth: ~16 GB/s (PCIe 4.0 x16)
 * GPU memory bandwidth: ~504 GB/s (RTX 4070 GDDR6X)
 * Ratio: GPU internal memory is ~30x faster than PCIe transfer!
 *
 * Lesson: Minimize CPU↔GPU transfers. Keep data on GPU as long as possible.
 * This is why training loops keep tensors on GPU for the entire epoch.
 *
 * Compile: nvcc -arch=sm_89 -o 03_memory_timing 03_memory_timing.cu
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do {                                   \
    cudaError_t err = call;                                     \
    if (err != cudaSuccess) {                                   \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",           \
                __FILE__, __LINE__, cudaGetErrorString(err));   \
        exit(EXIT_FAILURE);                                     \
    }                                                           \
} while(0)

/* Simple kernel: multiply each element by 2 */
__global__ void scale_kernel(float *data, int N) {
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += stride) {
        data[i] *= 2.0f;
    }
}

/* Time a specific phase using CUDA events */
float time_phase(const char *name, void (*fn)(void *), void *ctx) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    fn(ctx);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return ms;
}

typedef struct {
    float *h_data;
    float *d_data;
    size_t bytes;
    int N;
} TimingCtx;

void do_h2d(void *arg) {
    TimingCtx *ctx = (TimingCtx *)arg;
    CHECK_CUDA(cudaMemcpy(ctx->d_data, ctx->h_data, ctx->bytes, cudaMemcpyHostToDevice));
}

void do_kernel(void *arg) {
    TimingCtx *ctx = (TimingCtx *)arg;
    int block_size = 256;
    int num_blocks = (ctx->N + block_size - 1) / block_size;
    scale_kernel<<<num_blocks, block_size>>>(ctx->d_data, ctx->N);
}

void do_d2h(void *arg) {
    TimingCtx *ctx = (TimingCtx *)arg;
    CHECK_CUDA(cudaMemcpy(ctx->h_data, ctx->d_data, ctx->bytes, cudaMemcpyDeviceToHost));
}

void benchmark_size(int N) {
    size_t bytes = N * sizeof(float);
    float MB = bytes / (1024.0f * 1024.0f);

    /* Allocate */
    float *h_data = (float *)malloc(bytes);
    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, bytes));

    /* Initialize host data */
    for (int i = 0; i < N; i++)
        h_data[i] = (float)i;

    TimingCtx ctx = {h_data, d_data, bytes, N};

    /* Warmup */
    do_h2d(&ctx);
    do_kernel(&ctx);
    do_d2h(&ctx);

    /* Benchmark (average of 5 runs) */
    float h2d_ms = 0, kernel_ms = 0, d2h_ms = 0;
    int runs = 5;
    for (int r = 0; r < runs; r++) {
        /* Re-init for each run */
        for (int i = 0; i < N; i++) h_data[i] = (float)i;

        h2d_ms += time_phase("H2D", do_h2d, &ctx);
        kernel_ms += time_phase("Kernel", do_kernel, &ctx);
        d2h_ms += time_phase("D2H", do_d2h, &ctx);
    }
    h2d_ms /= runs;
    kernel_ms /= runs;
    d2h_ms /= runs;

    float total_ms = h2d_ms + kernel_ms + d2h_ms;
    float h2d_bw = MB / h2d_ms * 1000;  // MB/s → convert ms to s
    float d2h_bw = MB / d2h_ms * 1000;
    float kernel_bw = (MB * 2) / kernel_ms * 1000;  // read + write

    printf("  N=%9d (%7.1f MB) | H2D %7.3f ms (%6.1f GB/s) | "
           "Kernel %7.3f ms (%6.1f GB/s) | D2H %7.3f ms (%6.1f GB/s) | "
           "Total %7.3f ms | Kernel = %.0f%%\n",
           N, MB, h2d_ms, h2d_bw / 1024, kernel_ms, kernel_bw / 1024,
           d2h_ms, d2h_bw / 1024, total_ms,
           kernel_ms / total_ms * 100);

    CHECK_CUDA(cudaFree(d_data));
    free(h_data);
}

/* Compare pageable vs pinned memory transfers */
void pinned_vs_pageable() {
    printf("\n=== Pinned vs Pageable Memory ===\n");
    printf("Pinned memory (cudaMallocHost) is page-locked — DMA can transfer\n");
    printf("directly without staging through a CPU buffer. Much faster for PCIe.\n\n");

    int N = 10000000;  // 10M
    size_t bytes = N * sizeof(float);
    float MB = bytes / (1024.0f * 1024.0f);

    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, bytes));

    /* Pageable (regular malloc) */
    float *h_pageable = (float *)malloc(bytes);
    for (int i = 0; i < N; i++) h_pageable[i] = (float)i;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    /* Warmup */
    CHECK_CUDA(cudaMemcpy(d_data, h_pageable, bytes, cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDA(cudaMemcpy(d_data, h_pageable, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float pageable_ms;
    CHECK_CUDA(cudaEventElapsedTime(&pageable_ms, start, stop));

    /* Pinned (cudaMallocHost) */
    float *h_pinned;
    CHECK_CUDA(cudaMallocHost(&h_pinned, bytes));
    for (int i = 0; i < N; i++) h_pinned[i] = (float)i;

    /* Warmup */
    CHECK_CUDA(cudaMemcpy(d_data, h_pinned, bytes, cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDA(cudaMemcpy(d_data, h_pinned, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float pinned_ms;
    CHECK_CUDA(cudaEventElapsedTime(&pinned_ms, start, stop));

    printf("Transfer %.1f MB to GPU:\n", MB);
    printf("  Pageable: %7.3f ms (%5.1f GB/s)\n", pageable_ms, MB / pageable_ms * 1000 / 1024);
    printf("  Pinned:   %7.3f ms (%5.1f GB/s)\n", pinned_ms, MB / pinned_ms * 1000 / 1024);
    printf("  Speedup:  %.1fx\n", pageable_ms / pinned_ms);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFreeHost(h_pinned));
    CHECK_CUDA(cudaFree(d_data));
    free(h_pageable);
}

int main() {
    printf("=== GPU Memory Transfer Timing ===\n\n");
    printf("Measuring: cudaMemcpy (H2D) → Kernel → cudaMemcpy (D2H)\n");
    printf("The kernel just scales each float by 2.0.\n\n");

    /* Benchmark various sizes */
    int sizes[] = {1000, 10000, 100000, 1000000, 10000000, 50000000};
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < n_sizes; i++) {
        benchmark_size(sizes[i]);
    }

    printf("\nKey observation: The kernel is a tiny fraction of total time!\n");
    printf("Memory transfers dominate. This is why PyTorch keeps tensors on GPU.\n");

    pinned_vs_pageable();

    printf("\n=== Memory timing complete! ===\n");
    return 0;
}
