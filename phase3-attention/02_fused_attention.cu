/*
 * 02_fused_attention.cu — Fused single-tile attention
 *
 * For short sequences that fit entirely in shared memory:
 *   Load Q, K, V tiles → compute S = QK^T/√d → softmax(S) → O = P@V
 *   ALL in one kernel, no global memory for S or P!
 *
 * This is NOT Flash Attention yet — it only works when N fits in shared mem.
 * But it teaches the shared memory management patterns needed for the real thing.
 *
 * Shared memory layout:
 *   Q_tile  [Br, d]     — query rows assigned to this block
 *   K_tile  [N,  d]     — ALL keys (must fit in smem)
 *   V_tile  [N,  d]     — ALL values (must fit in smem)
 *   S_row   [Br, N]     — attention scores for this block's rows
 *
 * Constraint: N * d * 4 bytes * 2 (K+V) + Br * N * 4 + Br * d * 4 <= 48 KB
 *   For d=64, N=128: K+V = 64KB → too big for 48KB!
 *   So we process K,V in tiles too — a stepping stone to Flash Attention.
 *
 * Actually, let's use a simpler approach: one block per query row.
 * Each block processes one query row against ALL K,V (loaded in tiles).
 * This naturally leads to the Flash Attention tiling pattern.
 *
 * Compile: nvcc -arch=sm_89 -O2 -o 02_fused_attention 02_fused_attention.cu
 */

#include "common.h"

#define WARP_SIZE 32

/*
 * Fused attention: one block per query row.
 *
 * For query row i:
 *   1. Load q[i] into registers (d values per thread, striped)
 *   2. Loop over K/V in tiles of Bc columns:
 *      a. Load K tile [Bc, d] into shared memory
 *      b. Compute s[j] = dot(q, k[j]) / sqrt(d) for j in tile
 *      c. Online softmax: update running max m and denominator l
 *      d. Load V tile [Bc, d] into shared memory
 *      e. Update output: o = o * correction + P_tile @ V_tile
 *   3. Final normalization: o = o / l
 *
 * This IS the Flash Attention forward pass pattern!
 * The only difference from "real" Flash Attention is:
 *   - We process one query row per block (Br=1) instead of a tile of Br rows
 *   - We don't handle multi-head / batched dimensions
 */

/* Block size for KV tiles — how many key/value rows we load at once */
#define Bc 32

__global__ void fused_attention(const float *Q, const float *K, const float *V,
                                float *O, int N, int d, float scale) {
    /* One block per query row */
    int qi = blockIdx.x;  /* which query row */
    int tid = threadIdx.x;

    /*
     * Shared memory layout:
     *   kv_tile[Bc][d]  — current K or V tile
     *   scores[Bc]      — attention scores for current tile
     */
    extern __shared__ float smem[];
    float *kv_tile = smem;              /* [Bc * d] */
    float *scores  = smem + Bc * d;     /* [Bc] */

    /* Load query row into registers — each thread holds d/blockDim.x values */
    /* For simplicity: thread tid handles dimensions tid, tid+blockDim.x, ... */
    const float *q_row = Q + qi * d;

    /* Running online softmax state */
    float m = -FLT_MAX;   /* running max */
    float l = 0.0f;        /* running sum of exp */

    /* Output accumulator — each thread accumulates its d-dimensions */
    /* We need d output values. With blockDim.x threads, each handles d/blockDim.x dims. */
    /* For d=64 and blockDim.x=64, each thread handles 1 dim. Simple! */
    /* For larger d, each thread handles multiple dims via striding. */
    float o_acc[4] = {0};  /* max 4 dims per thread (supports d up to 4*blockDim.x) */
    int dims_per_thread = CEIL_DIV(d, blockDim.x);

    /* Process K,V in tiles of Bc rows */
    for (int kv_start = 0; kv_start < N; kv_start += Bc) {
        int kv_end = min(kv_start + Bc, N);
        int tile_len = kv_end - kv_start;

        /* ── Load K tile into shared memory ── */
        /* kv_tile[j][k] = K[(kv_start + j) * d + k] */
        for (int idx = tid; idx < tile_len * d; idx += blockDim.x) {
            int j = idx / d;
            int k = idx % d;
            kv_tile[j * d + k] = K[(kv_start + j) * d + k];
        }
        __syncthreads();

        /* ── Compute scores: s[j] = dot(q, k[j]) * scale ── */
        for (int j = tid; j < tile_len; j += blockDim.x) {
            float dot = 0.0f;
            for (int k = 0; k < d; k++)
                dot += q_row[k] * kv_tile[j * d + k];
            scores[j] = dot * scale;
        }
        __syncthreads();

        /* ── Online softmax update ── */

        /* All threads read scores (read-only) to find tile max — no race */
        float tile_max = -FLT_MAX;
        for (int j = 0; j < tile_len; j++)
            tile_max = fmaxf(tile_max, scores[j]);

        float m_new = fmaxf(m, tile_max);
        float correction = expf(m - m_new);

        /* Convert scores to exp(s - m_new) — ONLY assigned threads write to avoid race! */
        for (int j = tid; j < tile_len; j += blockDim.x)
            scores[j] = expf(scores[j] - m_new);
        __syncthreads();

        /* All threads read exp-scores (read-only) to compute tile sum — no race */
        float tile_sum = 0.0f;
        for (int j = 0; j < tile_len; j++)
            tile_sum += scores[j];
        float l_new = l * correction + tile_sum;

        /* ── Load V tile into shared memory (reuse kv_tile) ── */
        /* scores[] in a separate region, so safe to overwrite kv_tile */
        __syncthreads();

        for (int idx = tid; idx < tile_len * d; idx += blockDim.x) {
            int j = idx / d;
            int k = idx % d;
            kv_tile[j * d + k] = V[(kv_start + j) * d + k];
        }
        __syncthreads();

        /* ── Update output: o = o * correction + P_tile @ V_tile ── */
        for (int di = 0; di < dims_per_thread; di++) {
            int dim = tid + di * blockDim.x;
            if (dim >= d) break;

            /* Rescale previous accumulation */
            o_acc[di] *= correction;

            /* Add contribution: sum_j scores[j] * V[j][dim] */
            for (int j = 0; j < tile_len; j++)
                o_acc[di] += scores[j] * kv_tile[j * d + dim];
        }

        m = m_new;
        l = l_new;
        __syncthreads();
    }

    /* ── Final normalization: o = o / l ── */
    float inv_l = 1.0f / l;
    for (int di = 0; di < dims_per_thread; di++) {
        int dim = tid + di * blockDim.x;
        if (dim >= d) break;
        O[qi * d + dim] = o_acc[di] * inv_l;
    }
}

int main() {
    printf("=== Phase 3.2: Fused Attention (Tiled KV, Online Softmax) ===\n\n");

    int N = 1024;
    int d = 64;
    float scale = 1.0f / sqrtf((float)d);

    size_t qkv_bytes = (size_t)N * d * sizeof(float);

    printf("Config: N=%d, d=%d, Bc=%d (KV tile size)\n", N, d, Bc);
    printf("No materialized N×N attention matrix!\n\n");

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

    attention_cpu(h_Q, h_K, h_V, h_O_ref, N, d);

    /* Allocate device */
    float *d_Q, *d_K, *d_V, *d_O;
    CHECK_CUDA(cudaMalloc(&d_Q, qkv_bytes));
    CHECK_CUDA(cudaMalloc(&d_K, qkv_bytes));
    CHECK_CUDA(cudaMalloc(&d_V, qkv_bytes));
    CHECK_CUDA(cudaMalloc(&d_O, qkv_bytes));

    CHECK_CUDA(cudaMemcpy(d_Q, h_Q, qkv_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K, qkv_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V, qkv_bytes, cudaMemcpyHostToDevice));

    /* Launch: one block per query row, blockDim.x = 64 (one thread per d-dim) */
    int block_size = d;  /* d=64: one thread per dimension */
    /* Shared memory: kv_tile[Bc*d] + scores[Bc] */
    int smem_size = (Bc * d + Bc) * sizeof(float);
    printf("Shared memory per block: %d bytes (%.1f KB)\n", smem_size, smem_size / 1024.0f);

    printf("\nCorrectness:\n");
    fused_attention<<<N, block_size, smem_size>>>(d_Q, d_K, d_V, d_O, N, d, scale);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_O_gpu, d_O, qkv_bytes, cudaMemcpyDeviceToHost));
    verify(h_O_gpu, h_O_ref, N * d, 1e-3f, "fused-attn (Br=1, tiled KV)");

    /* Benchmark */
    printf("\nBenchmark (N=%d, d=%d):\n", N, d);
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    int runs = 50;

    /* Warmup */
    fused_attention<<<N, block_size, smem_size>>>(d_Q, d_K, d_V, d_O, N, d, scale);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int r = 0; r < runs; r++)
        fused_attention<<<N, block_size, smem_size>>>(d_Q, d_K, d_V, d_O, N, d, scale);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= runs;
    printf("  Fused (Br=1): %.3f ms\n", ms);
    printf("  Extra memory: 0 bytes (no N×N matrix!)\n");

    /* Cleanup */
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_O));
    free(h_Q); free(h_K); free(h_V);
    free(h_O_ref); free(h_O_gpu);

    printf("\n=== Fused attention complete! ===\n");
    return 0;
}
