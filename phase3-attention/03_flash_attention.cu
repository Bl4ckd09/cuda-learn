/*
 * 03_flash_attention.cu — Flash Attention Forward Pass (FP32)
 *
 * The real Flash Attention algorithm (Dao et al., 2022):
 *   - Outer loop over Q tiles (Br query rows per block)
 *   - Inner loop over KV tiles (Bc key/value rows per iteration)
 *   - Online softmax maintains running max and denominator per row
 *   - Never materializes the N×N attention matrix!
 *
 * Memory: O(N*d) instead of O(N^2) — the entire point.
 *
 * Design: blockDim.x = D_HEAD (64) threads.
 *   Each thread "owns" one dimension of the output across ALL Br rows.
 *   Per-row softmax state (m, l) is in shared memory so ALL threads
 *   can read correction factors.
 *
 * Compile: nvcc -arch=sm_89 -O2 -lcublas -o 03_flash_attention 03_flash_attention.cu
 */

#include "common.h"
#include <cublas_v2.h>

#define Br 32
#define Bc 32
#define D_HEAD 64

/*
 * Shared memory layout:
 *   q_tile   [Br * D_HEAD]  — query rows for this block
 *   kv_tile  [Bc * D_HEAD]  — current K or V tile
 *   scores   [Br * Bc]      — attention scores for current tile
 *   row_m    [Br]           — running max per query row
 *   row_l    [Br]           — running sum per query row
 *   row_corr [Br]           — correction factor for current tile
 *
 * Total with Br=32, Bc=32, d=64:
 *   32*64 + 32*64 + 32*32 + 32 + 32 + 32 = 2048+2048+1024+96 = 5216 floats
 *   = 20864 bytes ≈ 20.4 KB ✓ (well within 48 KB)
 */

__global__ void flash_attention_fwd(const float *Q, const float *K, const float *V,
                                    float *O, int N, float scale) {
    int q_start = blockIdx.x * Br;
    int tid = threadIdx.x;  /* 0..D_HEAD-1 */

    extern __shared__ float smem[];
    float *q_tile  = smem;
    float *kv_tile = q_tile + Br * D_HEAD;
    float *scores  = kv_tile + Bc * D_HEAD;
    float *row_m   = scores + Br * Bc;
    float *row_l   = row_m + Br;
    float *row_corr = row_l + Br;

    /* Load Q tile into shared memory */
    for (int idx = tid; idx < Br * D_HEAD; idx += blockDim.x) {
        int r = idx / D_HEAD;
        int global_row = q_start + r;
        q_tile[idx] = (global_row < N) ? Q[global_row * D_HEAD + (idx % D_HEAD)] : 0.0f;
    }

    /* Initialize per-row state in shared memory */
    for (int i = tid; i < Br; i += blockDim.x) {
        row_m[i] = -FLT_MAX;
        row_l[i] = 0.0f;
    }
    __syncthreads();

    /* Per-thread output accumulators: o_acc[i] for row i, dimension tid */
    float o_acc[Br];
    for (int i = 0; i < Br; i++) o_acc[i] = 0.0f;

    /* ── Inner loop: iterate over KV tiles ── */
    for (int kv_start = 0; kv_start < N; kv_start += Bc) {
        int kv_end = min(kv_start + Bc, N);
        int tile_len = kv_end - kv_start;

        /* Load K tile */
        for (int idx = tid; idx < Bc * D_HEAD; idx += blockDim.x) {
            int r = idx / D_HEAD;
            kv_tile[idx] = (r < tile_len) ? K[(kv_start + r) * D_HEAD + (idx % D_HEAD)] : 0.0f;
        }
        __syncthreads();

        /* Compute scores: S[i][j] = dot(q[i], k[j]) * scale */
        for (int sij = tid; sij < Br * Bc; sij += blockDim.x) {
            int i = sij / Bc;
            int j = sij % Bc;
            if (j < tile_len && (q_start + i) < N) {
                float dot = 0.0f;
                for (int k = 0; k < D_HEAD; k++)
                    dot += q_tile[i * D_HEAD + k] * kv_tile[j * D_HEAD + k];
                scores[sij] = dot * scale;
            } else {
                scores[sij] = -FLT_MAX;  /* padding: will become 0 after softmax */
            }
        }
        __syncthreads();

        /* Online softmax update — one thread per row (threads 0..Br-1) */
        for (int i = tid; i < Br; i += blockDim.x) {
            if (q_start + i >= N) {
                row_corr[i] = 1.0f;
                continue;
            }

            float old_m = row_m[i];

            /* Find tile max */
            float tile_max = -FLT_MAX;
            for (int j = 0; j < tile_len; j++)
                tile_max = fmaxf(tile_max, scores[i * Bc + j]);

            float m_new = fmaxf(old_m, tile_max);

            /* Correction factor: guard against -inf - (-inf) = NaN */
            float corr = (old_m == -FLT_MAX) ? 0.0f : expf(old_m - m_new);

            /* Compute exp(s - m_new) in-place and accumulate sum */
            float tile_sum = 0.0f;
            for (int j = 0; j < tile_len; j++) {
                float e = expf(scores[i * Bc + j] - m_new);
                scores[i * Bc + j] = e;
                tile_sum += e;
            }

            row_m[i] = m_new;
            row_l[i] = row_l[i] * corr + tile_sum;
            row_corr[i] = corr;
        }
        __syncthreads();

        /* ALL threads apply correction and load V for output update */

        /* Apply correction to ALL output accumulators */
        for (int i = 0; i < Br; i++)
            o_acc[i] *= row_corr[i];

        /* Load V tile (reuse kv_tile) */
        /* But we still need scores! They're in a separate region, so safe. */
        __syncthreads();
        for (int idx = tid; idx < Bc * D_HEAD; idx += blockDim.x) {
            int r = idx / D_HEAD;
            kv_tile[idx] = (r < tile_len) ? V[(kv_start + r) * D_HEAD + (idx % D_HEAD)] : 0.0f;
        }
        __syncthreads();

        /* Accumulate output: o[i][tid] += sum_j scores[i][j] * V[j][tid] */
        for (int i = 0; i < Br; i++) {
            if (q_start + i >= N) continue;
            for (int j = 0; j < tile_len; j++)
                o_acc[i] += scores[i * Bc + j] * kv_tile[j * D_HEAD + tid];
        }
        __syncthreads();
    }

    /* Final normalization: o = o / l */
    for (int i = 0; i < Br; i++) {
        int global_row = q_start + i;
        if (global_row < N) {
            float inv_l = (row_l[i] > 0.0f) ? (1.0f / row_l[i]) : 0.0f;
            O[global_row * D_HEAD + tid] = o_acc[i] * inv_l;
        }
    }
}

int main() {
    printf("=== Phase 3.3: Flash Attention Forward (FP32) ===\n\n");

    int test_Ns[] = {256, 512, 1024, 2048};
    int num_tests = 4;
    int d = D_HEAD;
    float scale = 1.0f / sqrtf((float)d);

    /* Shared memory: q_tile + kv_tile + scores + row_m + row_l + row_corr */
    int smem_size = (Br * d + Bc * d + Br * Bc + 3 * Br) * sizeof(float);
    printf("Config: Br=%d, Bc=%d, d=%d\n", Br, Bc, d);
    printf("Shared memory per block: %d bytes (%.1f KB)\n\n", smem_size, smem_size / 1024.0f);

    cublasHandle_t handle;
    cublasCreate(&handle);

    printf("%-6s  %-12s  %-12s  %-12s  %-12s  %-12s\n",
           "N", "Flash (ms)", "Naive (ms)", "Speedup", "Flash KB", "Naive KB");
    printf("------  ----------  ----------  ----------  ----------  ----------\n");

    for (int t = 0; t < num_tests; t++) {
        int N = test_Ns[t];
        size_t qkv_bytes = (size_t)N * d * sizeof(float);
        size_t attn_bytes = (size_t)N * N * sizeof(float);

        float *h_Q = (float *)malloc(qkv_bytes);
        float *h_K = (float *)malloc(qkv_bytes);
        float *h_V = (float *)malloc(qkv_bytes);
        float *h_O_ref = (float *)malloc(qkv_bytes);
        float *h_O_flash = (float *)malloc(qkv_bytes);

        srand(42);
        fill_random(h_Q, N * d);
        fill_random(h_K, N * d);
        fill_random(h_V, N * d);

        attention_cpu(h_Q, h_K, h_V, h_O_ref, N, d);

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

        /* ── Flash Attention ── */
        int grid_flash = CEIL_DIV(N, Br);
        flash_attention_fwd<<<grid_flash, D_HEAD, smem_size>>>(
            d_Q, d_K, d_V, d_O, N, scale);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_O_flash, d_O, qkv_bytes, cudaMemcpyDeviceToHost));

        if (t == 0) {
            printf("\nCorrectness (N=%d):\n", N);
            verify(h_O_flash, h_O_ref, N * d, 5e-3f, "flash-attention-fwd");
            printf("\n");
        }

        /* ── Benchmark: Flash Attention ── */
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        int runs = 50;

        flash_attention_fwd<<<grid_flash, D_HEAD, smem_size>>>(
            d_Q, d_K, d_V, d_O, N, scale);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventRecord(start));
        for (int r = 0; r < runs; r++)
            flash_attention_fwd<<<grid_flash, D_HEAD, smem_size>>>(
                d_Q, d_K, d_V, d_O, N, scale);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms_flash;
        CHECK_CUDA(cudaEventElapsedTime(&ms_flash, start, stop));
        ms_flash /= runs;

        /* ── Benchmark: Naive (cuBLAS matmuls only, no softmax) ── */
        float beta_zero = 0.0f, alpha_one = 1.0f;
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    N, N, d, &scale, d_K, d, d_Q, d, &beta_zero, d_S, N);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    d, N, N, &alpha_one, d_V, d, d_S, N, &beta_zero, d_O, d);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventRecord(start));
        for (int r = 0; r < runs; r++) {
            cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        N, N, d, &scale, d_K, d, d_Q, d, &beta_zero, d_S, N);
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        d, N, N, &alpha_one, d_V, d, d_S, N, &beta_zero, d_O, d);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms_naive;
        CHECK_CUDA(cudaEventElapsedTime(&ms_naive, start, stop));
        ms_naive /= runs;

        size_t flash_mem = 4 * qkv_bytes;
        size_t naive_mem = 4 * qkv_bytes + 2 * attn_bytes;

        printf("%-6d  %-12.3f  %-12.3f  %-12.2fx  %-12.1f  %-12.1f\n",
               N, ms_flash, ms_naive, ms_naive / ms_flash,
               flash_mem / 1024.0f, naive_mem / 1024.0f);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaFree(d_Q));
        CHECK_CUDA(cudaFree(d_K));
        CHECK_CUDA(cudaFree(d_V));
        CHECK_CUDA(cudaFree(d_O));
        CHECK_CUDA(cudaFree(d_S));
        CHECK_CUDA(cudaFree(d_P));
        free(h_Q); free(h_K); free(h_V);
        free(h_O_ref); free(h_O_flash);
    }

    printf("\nFlash mem = Q+K+V+O only (O(N*d))\n");
    printf("Naive mem = Q+K+V+O + S+P (O(N^2) extra)\n");
    printf("Note: Our Flash kernel uses only %d threads/block — a production\n", D_HEAD);
    printf("kernel would use 128-256 threads with register tiling for speed.\n");
    printf("The point here is CORRECTNESS and understanding the algorithm.\n");

    cublasDestroy(handle);
    printf("\n=== Flash Attention complete! ===\n");
    return 0;
}
