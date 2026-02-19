/*
 * 07_streams.cu — CUDA streams for asynchronous execution
 *
 * By default, CUDA operations execute on the "default stream" (stream 0)
 * and are serialized:
 *   H2D → Kernel → D2H → H2D → Kernel → D2H → ...
 *
 * With multiple streams, independent operations can OVERLAP:
 *   Stream 1: H2D → Kernel → D2H
 *   Stream 2:       H2D → Kernel → D2H
 *                   ↑ overlaps with stream 1's kernel!
 *
 * Why this matters for LLM training:
 *   - PyTorch uses streams to overlap data loading with computation
 *   - cudaMemcpyAsync + compute overlapping = free bandwidth
 *   - Multi-GPU training uses streams per GPU
 *
 * Requirements for async operations:
 *   1. Use pinned (page-locked) host memory: cudaMallocHost()
 *   2. Use cudaMemcpyAsync() instead of cudaMemcpy()
 *   3. Launch kernels on specific streams: kernel<<<grid, block, 0, stream>>>
 *
 * Compile: nvcc -arch=sm_89 -o 07_streams 07_streams.cu
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

/* A kernel that does enough work to be visible in timeline */
__global__ void process_chunk(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stride) {
        float x = data[i];
        // Do some non-trivial work to make the kernel take measurable time
        for (int j = 0; j < 100; j++) {
            x = sinf(x) * cosf(x) + 0.001f;
        }
        data[i] = x;
    }
}

/* ========== Demo 1: Sequential vs Overlapped ========== */
void demo_sequential_vs_overlapped() {
    printf("=== Demo 1: Sequential vs Overlapped Streams ===\n\n");

    int N = 4000000;  // 4M elements per chunk
    int num_chunks = 4;
    size_t chunk_bytes = N * sizeof(float);

    /* Pinned host memory (required for async transfers) */
    float *h_data;
    CHECK_CUDA(cudaMallocHost(&h_data, num_chunks * chunk_bytes));
    for (int i = 0; i < num_chunks * N; i++)
        h_data[i] = 0.5f;

    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, num_chunks * chunk_bytes));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int block_size = 256;
    int num_blocks = (N + block_size - 1) / block_size;

    /* ---- Sequential: one chunk at a time on default stream ---- */
    CHECK_CUDA(cudaEventRecord(start));
    for (int c = 0; c < num_chunks; c++) {
        float *h_chunk = h_data + c * N;
        float *d_chunk = d_data + c * N;
        CHECK_CUDA(cudaMemcpy(d_chunk, h_chunk, chunk_bytes, cudaMemcpyHostToDevice));
        process_chunk<<<num_blocks, block_size>>>(d_chunk, N);
        CHECK_CUDA(cudaMemcpy(h_chunk, d_chunk, chunk_bytes, cudaMemcpyDeviceToHost));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float seq_ms;
    CHECK_CUDA(cudaEventElapsedTime(&seq_ms, start, stop));

    /* ---- Overlapped: each chunk on its own stream ---- */
    cudaStream_t streams[4];
    for (int i = 0; i < num_chunks; i++)
        CHECK_CUDA(cudaStreamCreate(&streams[i]));

    /* Re-initialize */
    for (int i = 0; i < num_chunks * N; i++)
        h_data[i] = 0.5f;

    CHECK_CUDA(cudaEventRecord(start));
    for (int c = 0; c < num_chunks; c++) {
        float *h_chunk = h_data + c * N;
        float *d_chunk = d_data + c * N;
        /* Each operation goes on its own stream */
        CHECK_CUDA(cudaMemcpyAsync(d_chunk, h_chunk, chunk_bytes,
                                   cudaMemcpyHostToDevice, streams[c]));
        /* The 0 after block_size is shared memory size (we use none) */
        process_chunk<<<num_blocks, block_size, 0, streams[c]>>>(d_chunk, N);
        CHECK_CUDA(cudaMemcpyAsync(h_chunk, d_chunk, chunk_bytes,
                                   cudaMemcpyDeviceToHost, streams[c]));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float overlap_ms;
    CHECK_CUDA(cudaEventElapsedTime(&overlap_ms, start, stop));

    printf("Sequential (%d chunks): %.2f ms\n", num_chunks, seq_ms);
    printf("Overlapped (%d streams): %.2f ms\n", num_chunks, overlap_ms);
    printf("Speedup: %.2fx\n\n", seq_ms / overlap_ms);
    printf("With overlapping, H2D of chunk N+1 happens during kernel of chunk N.\n");

    /* Cleanup */
    for (int i = 0; i < num_chunks; i++)
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFreeHost(h_data));
}

/* ========== Demo 2: Stream events for synchronization ========== */
void demo_stream_events() {
    printf("=== Demo 2: Stream Events ===\n\n");

    int N = 2000000;
    size_t bytes = N * sizeof(float);

    float *h_a, *h_b;
    CHECK_CUDA(cudaMallocHost(&h_a, bytes));
    CHECK_CUDA(cudaMallocHost(&h_b, bytes));
    for (int i = 0; i < N; i++) { h_a[i] = 1.0f; h_b[i] = 2.0f; }

    float *d_a, *d_b;
    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));

    cudaStream_t stream1, stream2;
    CHECK_CUDA(cudaStreamCreate(&stream1));
    CHECK_CUDA(cudaStreamCreate(&stream2));

    /*
     * Events let one stream wait for another:
     *   Stream 1: H2D(A) → Kernel(A) → [record event]
     *   Stream 2: H2D(B) → [wait for stream1's event] → Kernel(B)
     *
     * This ensures stream2's kernel doesn't start until
     * stream1's kernel is done, even though they're on different streams.
     */
    cudaEvent_t event_a_done;
    CHECK_CUDA(cudaEventCreate(&event_a_done));

    /* Stream 1: process A */
    CHECK_CUDA(cudaMemcpyAsync(d_a, h_a, bytes, cudaMemcpyHostToDevice, stream1));
    process_chunk<<<(N + 255) / 256, 256, 0, stream1>>>(d_a, N);
    CHECK_CUDA(cudaEventRecord(event_a_done, stream1));  // signal: A is done

    /* Stream 2: process B, but wait for A first */
    CHECK_CUDA(cudaMemcpyAsync(d_b, h_b, bytes, cudaMemcpyHostToDevice, stream2));
    CHECK_CUDA(cudaStreamWaitEvent(stream2, event_a_done));  // wait for signal
    process_chunk<<<(N + 255) / 256, 256, 0, stream2>>>(d_b, N);

    /* Copy results back */
    CHECK_CUDA(cudaMemcpyAsync(h_a, d_a, bytes, cudaMemcpyDeviceToHost, stream1));
    CHECK_CUDA(cudaMemcpyAsync(h_b, d_b, bytes, cudaMemcpyDeviceToHost, stream2));

    /* Wait for both */
    CHECK_CUDA(cudaStreamSynchronize(stream1));
    CHECK_CUDA(cudaStreamSynchronize(stream2));

    printf("Stream event synchronization completed.\n");
    printf("Stream 2's kernel started only after Stream 1's kernel finished.\n");
    printf("But H2D transfers could still overlap.\n\n");

    /* Cleanup */
    CHECK_CUDA(cudaEventDestroy(event_a_done));
    CHECK_CUDA(cudaStreamDestroy(stream1));
    CHECK_CUDA(cudaStreamDestroy(stream2));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFreeHost(h_a));
    CHECK_CUDA(cudaFreeHost(h_b));
}

/* ========== Demo 3: Stream priorities ========== */
void demo_stream_priorities() {
    printf("=== Demo 3: Stream Priorities ===\n\n");

    int least_priority, greatest_priority;
    CHECK_CUDA(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
    printf("Priority range: [%d (highest), %d (lowest)]\n",
           greatest_priority, least_priority);
    printf("(Lower number = higher priority)\n\n");

    cudaStream_t high_priority_stream, low_priority_stream;
    CHECK_CUDA(cudaStreamCreateWithPriority(&high_priority_stream,
                                            cudaStreamNonBlocking, greatest_priority));
    CHECK_CUDA(cudaStreamCreateWithPriority(&low_priority_stream,
                                            cudaStreamNonBlocking, least_priority));

    printf("High priority stream created (priority %d)\n", greatest_priority);
    printf("Low priority stream created (priority %d)\n", least_priority);
    printf("Use case: prioritize attention computation over gradient sync.\n\n");

    CHECK_CUDA(cudaStreamDestroy(high_priority_stream));
    CHECK_CUDA(cudaStreamDestroy(low_priority_stream));
}

int main() {
    printf("=== CUDA Streams and Asynchronous Execution ===\n\n");

    demo_sequential_vs_overlapped();
    demo_stream_events();
    demo_stream_priorities();

    printf("=== Streams demo complete! ===\n");
    printf("\nTo see stream overlap visually:\n");
    printf("  nsys profile -o streams_profile ./07_streams\n");
    printf("  nsys-ui streams_profile.qdrep\n");
    return 0;
}
