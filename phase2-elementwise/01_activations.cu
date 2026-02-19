/*
 * 01_activations.cu — ReLU and GELU activation functions
 *
 * Activations are the simplest elementwise kernels:
 *   output[i] = f(input[i])
 *
 * They're pure memory-bandwidth-bound (almost no arithmetic).
 * Bandwidth = 2 * N * sizeof(float) / time  (read input + write output)
 *
 * GELU (Gaussian Error Linear Unit) is what nanochat/GPT models use:
 *   GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
 *
 * This is the "fast approximation" used by PyTorch's F.gelu.
 *
 * Compile: nvcc -arch=sm_89 -O2 -o 01_activations 01_activations.cu
 */

#include "common.h"

/* ========== CPU references ========== */
void relu_cpu(const float *in, float *out, int N) {
    for (int i = 0; i < N; i++)
        out[i] = fmaxf(in[i], 0.0f);
}

void gelu_cpu(const float *in, float *out, int N) {
    const float sqrt_2_over_pi = 0.7978845608f;  // sqrt(2/π)
    for (int i = 0; i < N; i++) {
        float x = in[i];
        float inner = sqrt_2_over_pi * (x + 0.044715f * x * x * x);
        out[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

/* ========== GPU kernels ========== */
__global__ void relu_kernel(const float *in, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stride) {
        out[i] = fmaxf(in[i], 0.0f);
    }
}

__global__ void gelu_kernel(const float *in, float *out, int N) {
    const float sqrt_2_over_pi = 0.7978845608f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stride) {
        float x = in[i];
        float inner = sqrt_2_over_pi * (x + 0.044715f * x * x * x);
        out[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

/* ========== Fused bias + GELU ========== */
/*
 * In a transformer MLP:
 *   hidden = GELU(x @ W + bias)
 *
 * Unfused: bias_add kernel → GELU kernel (2 reads + 2 writes)
 * Fused: one kernel (1 read + 1 write) → 2x bandwidth savings
 */
void bias_gelu_cpu(const float *in, const float *bias, float *out, int N, int D) {
    const float sqrt_2_over_pi = 0.7978845608f;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
            float x = in[i * D + j] + bias[j];
            float inner = sqrt_2_over_pi * (x + 0.044715f * x * x * x);
            out[i * D + j] = 0.5f * x * (1.0f + tanhf(inner));
        }
    }
}

__global__ void bias_gelu_kernel(const float *in, const float *bias, float *out,
                                  int total, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    const float sqrt_2_over_pi = 0.7978845608f;
    for (int i = idx; i < total; i += stride) {
        float x = in[i] + bias[i % D];
        float inner = sqrt_2_over_pi * (x + 0.044715f * x * x * x);
        out[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

int main() {
    printf("=== Phase 2.1: Activation Functions ===\n\n");

    int N = 10000000;  // 10M elements
    size_t bytes = N * sizeof(float);

    float *h_in = (float *)malloc(bytes);
    float *h_out_cpu = (float *)malloc(bytes);
    float *h_out_gpu = (float *)malloc(bytes);
    fill_random(h_in, N);

    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));
    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    int bs = 256;
    int nb = CEIL_DIV(N, bs);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    /* --- ReLU --- */
    relu_cpu(h_in, h_out_cpu, N);
    relu_kernel<<<nb, bs>>>(d_in, d_out, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_out_gpu, d_out, bytes, cudaMemcpyDeviceToHost));
    verify(h_out_gpu, h_out_cpu, N, 1e-6f, "ReLU");

    /* ReLU bandwidth */
    relu_kernel<<<nb, bs>>>(d_in, d_out, N);  // warmup
    CHECK_CUDA(cudaEventRecord(start));
    for (int r = 0; r < 100; r++)
        relu_kernel<<<nb, bs>>>(d_in, d_out, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float relu_ms;
    CHECK_CUDA(cudaEventElapsedTime(&relu_ms, start, stop));
    relu_ms /= 100;
    float relu_bw = 2.0f * bytes / relu_ms / 1e6;  // GB/s
    printf("  ReLU: %.3f ms, %.1f GB/s (%.1f%% of 504 GB/s peak)\n\n",
           relu_ms, relu_bw, relu_bw / 504 * 100);

    /* --- GELU --- */
    gelu_cpu(h_in, h_out_cpu, N);
    gelu_kernel<<<nb, bs>>>(d_in, d_out, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_out_gpu, d_out, bytes, cudaMemcpyDeviceToHost));
    verify(h_out_gpu, h_out_cpu, N, 1e-5f, "GELU");

    gelu_kernel<<<nb, bs>>>(d_in, d_out, N);
    CHECK_CUDA(cudaEventRecord(start));
    for (int r = 0; r < 100; r++)
        gelu_kernel<<<nb, bs>>>(d_in, d_out, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float gelu_ms;
    CHECK_CUDA(cudaEventElapsedTime(&gelu_ms, start, stop));
    gelu_ms /= 100;
    float gelu_bw = 2.0f * bytes / gelu_ms / 1e6;
    printf("  GELU: %.3f ms, %.1f GB/s (%.1f%% of peak)\n\n",
           gelu_ms, gelu_bw, gelu_bw / 504 * 100);

    /* --- Fused bias + GELU --- */
    printf("--- Fused Bias + GELU ---\n");
    int rows = 4096, D = 1024;
    int total = rows * D;
    size_t total_bytes = total * sizeof(float);
    size_t bias_bytes = D * sizeof(float);

    float *h_in2 = (float *)malloc(total_bytes);
    float *h_bias = (float *)malloc(bias_bytes);
    float *h_ref2 = (float *)malloc(total_bytes);
    float *h_gpu2 = (float *)malloc(total_bytes);
    fill_random(h_in2, total);
    fill_random(h_bias, D);

    bias_gelu_cpu(h_in2, h_bias, h_ref2, rows, D);

    float *d_in2, *d_bias, *d_out2;
    CHECK_CUDA(cudaMalloc(&d_in2, total_bytes));
    CHECK_CUDA(cudaMalloc(&d_bias, bias_bytes));
    CHECK_CUDA(cudaMalloc(&d_out2, total_bytes));
    CHECK_CUDA(cudaMemcpy(d_in2, h_in2, total_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bias, h_bias, bias_bytes, cudaMemcpyHostToDevice));

    bias_gelu_kernel<<<CEIL_DIV(total, bs), bs>>>(d_in2, d_bias, d_out2, total, D);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_gpu2, d_out2, total_bytes, cudaMemcpyDeviceToHost));
    verify(h_gpu2, h_ref2, total, 1e-5f, "Fused Bias+GELU");

    /* Compare fused vs unfused bandwidth */
    /* Unfused: bias_add reads in+bias, writes tmp; gelu reads tmp, writes out
       = 4 * total_bytes + 2 * bias_bytes of memory traffic
       Fused: reads in+bias, writes out = 2 * total_bytes + bias_bytes */
    printf("  Unfused memory traffic: %.1f MB (4 passes)\n",
           (4.0f * total_bytes + 2 * bias_bytes) / 1e6);
    printf("  Fused memory traffic:   %.1f MB (2 passes)\n\n",
           (2.0f * total_bytes + bias_bytes) / 1e6);

    /* Cleanup */
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_in2));
    CHECK_CUDA(cudaFree(d_bias));
    CHECK_CUDA(cudaFree(d_out2));
    free(h_in); free(h_out_cpu); free(h_out_gpu);
    free(h_in2); free(h_bias); free(h_ref2); free(h_gpu2);

    printf("=== Activations complete! ===\n");
    return 0;
}
