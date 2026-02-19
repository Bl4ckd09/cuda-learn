/*
 * 06_atomics.cu — Race conditions and atomic operations
 *
 * The problem: When 1M threads all try to increment the same counter,
 * they read-modify-write the same memory location simultaneously.
 * Without atomics, most increments are LOST (read stale value).
 *
 * The race condition:
 *   Thread A reads counter = 5
 *   Thread B reads counter = 5   (stale! A's write hasn't happened)
 *   Thread A writes counter = 6
 *   Thread B writes counter = 6  (overwrites A's result!)
 *   → Lost one increment!
 *
 * atomicAdd guarantees read-modify-write is ONE indivisible operation.
 * No other thread can see the value between read and write.
 *
 * When you need atomics in real code:
 *   - Histogram computation
 *   - Finding max/min across threads
 *   - Reduction (sum, count) across blocks
 *   - Argmax/argmin for inference
 *
 * Compile: nvcc -arch=sm_89 -o 06_atomics 06_atomics.cu
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do {                                   \
    cudaError_t err = call;                                     \
    if (err != cudaSuccess) {                                   \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",           \
                __FILE__, __LINE__, cudaGetErrorString(err));   \
        exit(EXIT_FAILURE);                                     \
    }                                                           \
} while(0)

/* ========== Non-atomic increment (BROKEN) ========== */
__global__ void increment_non_atomic(int *counter) {
    // Race condition: many threads read the same old value,
    // all add 1, all write back the same "old + 1" value.
    // Result: most increments are lost.
    *counter += 1;
}

/* ========== Atomic increment (CORRECT) ========== */
__global__ void increment_atomic(int *counter) {
    // atomicAdd reads the current value, adds 1, writes back,
    // all as a single indivisible operation.
    atomicAdd(counter, 1);
}

/* ========== Atomic histogram ========== */
/*
 * A practical example: building a histogram of values.
 * Each thread categorizes its element and atomically increments
 * the corresponding bin.
 */
__global__ void histogram_kernel(const int *data, int *bins, int N, int num_bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride) {
        int bin = data[i] % num_bins;
        atomicAdd(&bins[bin], 1);
    }
}

/* ========== Atomic max/min ========== */
__global__ void find_max_atomic(const float *data, float *result, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride) {
        // atomicMax doesn't exist for float in older CUDA, but does for int.
        // For float, we use the int-based CAS trick (or __float_as_int).
        // On sm_89 (Ada), we can use atomicMax with unsigned int reinterpretation
        // for positive floats (IEEE 754 ordering matches int ordering for positives).
        //
        // Simple approach: use a shared memory reduction first, then atomic.
        // For now, just demonstrate the concept with int.
        int int_val = __float_as_int(data[i]);
        atomicMax((int *)result, int_val);
    }
}

int main() {
    printf("=== Atomic Operations ===\n\n");

    /* ===== Demo 1: Race condition vs Atomic ===== */
    int NUM_BLOCKS = 1000;
    int NUM_THREADS = 1000;
    int expected = NUM_BLOCKS * NUM_THREADS;

    printf("Launching %d blocks x %d threads = %d total threads\n",
           NUM_BLOCKS, NUM_THREADS, expected);
    printf("Each thread increments a counter by 1.\n\n");

    int *d_counter;
    int h_counter;
    CHECK_CUDA(cudaMalloc(&d_counter, sizeof(int)));

    /* Non-atomic (broken) */
    CHECK_CUDA(cudaMemset(d_counter, 0, sizeof(int)));
    increment_non_atomic<<<NUM_BLOCKS, NUM_THREADS>>>(d_counter);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Non-atomic result: %d (expected %d, lost %d increments = %.1f%%)\n",
           h_counter, expected, expected - h_counter,
           (float)(expected - h_counter) / expected * 100);

    /* Atomic (correct) */
    CHECK_CUDA(cudaMemset(d_counter, 0, sizeof(int)));
    increment_atomic<<<NUM_BLOCKS, NUM_THREADS>>>(d_counter);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Atomic result:     %d (expected %d) — %s\n\n",
           h_counter, expected, h_counter == expected ? "CORRECT" : "WRONG");

    CHECK_CUDA(cudaFree(d_counter));

    /* ===== Demo 2: Atomic histogram ===== */
    printf("--- Histogram Example ---\n");
    int N = 10000000;  // 10M elements
    int num_bins = 10;

    int *h_data = (int *)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) h_data[i] = i;  // uniform distribution mod num_bins

    int *d_data, *d_bins;
    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_bins, num_bins * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_bins, 0, num_bins * sizeof(int)));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    histogram_kernel<<<256, 256>>>(d_data, d_bins, N, num_bins);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    int h_bins[10];
    CHECK_CUDA(cudaMemcpy(h_bins, d_bins, num_bins * sizeof(int), cudaMemcpyDeviceToHost));

    printf("Histogram of %d elements into %d bins (%.3f ms):\n", N, num_bins, ms);
    int total = 0;
    for (int i = 0; i < num_bins; i++) {
        printf("  Bin %d: %d\n", i, h_bins[i]);
        total += h_bins[i];
    }
    printf("  Total: %d (expected %d) — %s\n\n",
           total, N, total == N ? "CORRECT" : "WRONG");

    /* Cleanup */
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_bins));
    free(h_data);

    printf("Key takeaway: Atomics serialize access → slower than reductions.\n");
    printf("For performance-critical code (like softmax denominators), prefer\n");
    printf("warp shuffles + shared memory reductions (Phase 2).\n");

    printf("\n=== Atomics demo complete! ===\n");
    return 0;
}
