/*
 * 05_multihead_fp16.cu — Multi-head Batched Flash Attention (FP16/mixed precision)
 *
 * Extends Flash Attention to handle real transformer dimensions:
 *   - Batch dimension (B)
 *   - Multiple heads (H)
 *   - FP16 storage with FP32 accumulation (mixed precision)
 *
 * Layout: Q, K, V, O are [B, H, N, d] in row-major (contiguous in d)
 * Each block processes one (batch, head, q_tile) triple.
 *
 * Mixed precision strategy:
 *   - Q, K, V, O stored as __half (FP16) — halves memory, doubles bandwidth
 *   - Dot products and softmax accumulated in float (FP32) — numerical stability
 *   - This matches what production Flash Attention and tensor cores do
 *
 * Compile: nvcc -arch=sm_89 -O2 -lcublas -o 05_multihead_fp16 05_multihead_fp16.cu
 */

#include "common.h"
#include <cuda_fp16.h>
#include <cublas_v2.h>

#define Br 32
#define Bc 32
#define D_HEAD 64

/* ═══════════ CPU reference (FP32, multi-head batched) ═══════════ */
/*
 * Q, K, V: [B, H, N, d] in FP32
 * O:       [B, H, N, d] in FP32
 */
void mha_attention_cpu(const float *Q, const float *K, const float *V,
                       float *O, int B, int H, int N, int d) {
    float scale = 1.0f / sqrtf((float)d);
    size_t head_stride = (size_t)N * d;
    size_t batch_stride = (size_t)H * head_stride;

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            const float *Qbh = Q + b * batch_stride + h * head_stride;
            const float *Kbh = K + b * batch_stride + h * head_stride;
            const float *Vbh = V + b * batch_stride + h * head_stride;
            float *Obh       = O + b * batch_stride + h * head_stride;
            attention_cpu(Qbh, Kbh, Vbh, Obh, N, d);
        }
    }
}

/* ═══════════ FP16 conversion kernels ═══════════ */
__global__ void float_to_half_kernel(const float *in, __half *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = __float2half(in[idx]);
}

__global__ void half_to_float_kernel(const __half *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = __half2float(in[idx]);
}

/* ═══════════ Multi-head Flash Attention Forward (FP16 in, FP32 accum) ═══════════ */
/*
 * Grid: (ceil(N/Br), H, B) — one block per (batch, head, q_tile)
 * Block: D_HEAD threads — one per output dimension
 *
 * Q, K, V: [B, H, N, d] in __half
 * O:       [B, H, N, d] in __half
 *
 * Shared memory:
 *   q_tile   [Br * D_HEAD] float  — query rows (converted to FP32 for accumulation)
 *   kv_tile  [Bc * D_HEAD] float  — K or V tile (converted to FP32)
 *   scores   [Br * Bc]     float  — attention scores
 *   row_m    [Br]          float  — running max per query row
 *   row_l    [Br]          float  — running sum per query row
 *   row_corr [Br]          float  — correction factor
 */
__global__ void flash_attention_mh_fp16(
    const __half *Q, const __half *K, const __half *V,
    __half *O, int B, int H, int N, float scale)
{
    int q_tile_idx = blockIdx.x;  /* which Q tile within this head */
    int h = blockIdx.y;           /* head index */
    int b = blockIdx.z;           /* batch index */
    int tid = threadIdx.x;        /* 0..D_HEAD-1 */

    int q_start = q_tile_idx * Br;
    if (q_start >= N) return;

    /* Pointer to this (batch, head) slice */
    size_t head_stride = (size_t)N * D_HEAD;
    size_t batch_stride = (size_t)H * head_stride;
    size_t offset = b * batch_stride + h * head_stride;

    const __half *Q_bh = Q + offset;
    const __half *K_bh = K + offset;
    const __half *V_bh = V + offset;
    __half *O_bh       = O + offset;

    /* Shared memory */
    extern __shared__ float smem[];
    float *q_tile   = smem;
    float *kv_tile  = q_tile + Br * D_HEAD;
    float *scores   = kv_tile + Bc * D_HEAD;
    float *row_m    = scores + Br * Bc;
    float *row_l    = row_m + Br;
    float *row_corr = row_l + Br;

    /* Load Q tile: FP16 → FP32 */
    for (int idx = tid; idx < Br * D_HEAD; idx += blockDim.x) {
        int r = idx / D_HEAD;
        int global_row = q_start + r;
        q_tile[idx] = (global_row < N)
            ? __half2float(Q_bh[global_row * D_HEAD + (idx % D_HEAD)])
            : 0.0f;
    }

    /* Initialize per-row state */
    for (int i = tid; i < Br; i += blockDim.x) {
        row_m[i] = -FLT_MAX;
        row_l[i] = 0.0f;
    }
    __syncthreads();

    /* Per-thread output accumulators (FP32) */
    float o_acc[Br];
    for (int i = 0; i < Br; i++) o_acc[i] = 0.0f;

    /* Inner loop over KV tiles */
    for (int kv_start = 0; kv_start < N; kv_start += Bc) {
        int kv_end = min(kv_start + Bc, N);
        int tile_len = kv_end - kv_start;

        /* Load K tile: FP16 → FP32 */
        for (int idx = tid; idx < Bc * D_HEAD; idx += blockDim.x) {
            int r = idx / D_HEAD;
            kv_tile[idx] = (r < tile_len)
                ? __half2float(K_bh[(kv_start + r) * D_HEAD + (idx % D_HEAD)])
                : 0.0f;
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
                scores[sij] = -FLT_MAX;
            }
        }
        __syncthreads();

        /* Online softmax — one thread per row */
        for (int i = tid; i < Br; i += blockDim.x) {
            if (q_start + i >= N) {
                row_corr[i] = 1.0f;
                continue;
            }

            float old_m = row_m[i];
            float tile_max = -FLT_MAX;
            for (int j = 0; j < tile_len; j++)
                tile_max = fmaxf(tile_max, scores[i * Bc + j]);

            float m_new = fmaxf(old_m, tile_max);
            float corr = (old_m == -FLT_MAX) ? 0.0f : expf(old_m - m_new);

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

        /* Apply correction */
        for (int i = 0; i < Br; i++)
            o_acc[i] *= row_corr[i];

        /* Load V tile: FP16 → FP32 */
        __syncthreads();
        for (int idx = tid; idx < Bc * D_HEAD; idx += blockDim.x) {
            int r = idx / D_HEAD;
            kv_tile[idx] = (r < tile_len)
                ? __half2float(V_bh[(kv_start + r) * D_HEAD + (idx % D_HEAD)])
                : 0.0f;
        }
        __syncthreads();

        /* Accumulate output (FP32) */
        for (int i = 0; i < Br; i++) {
            if (q_start + i >= N) continue;
            for (int j = 0; j < tile_len; j++)
                o_acc[i] += scores[i * Bc + j] * kv_tile[j * D_HEAD + tid];
        }
        __syncthreads();
    }

    /* Write output: FP32 → FP16 */
    for (int i = 0; i < Br; i++) {
        int global_row = q_start + i;
        if (global_row < N) {
            float inv_l = (row_l[i] > 0.0f) ? (1.0f / row_l[i]) : 0.0f;
            O_bh[global_row * D_HEAD + tid] = __float2half(o_acc[i] * inv_l);
        }
    }
}

/* ═══════════ Single-head FP32 Flash Attention (for comparison) ═══════════ */
/* Reuse the 03_flash_attention kernel logic inline for benchmarking */
__global__ void flash_attention_fp32(const float *Q, const float *K, const float *V,
                                     float *O, int N, float scale) {
    int q_start = blockIdx.x * Br;
    int tid = threadIdx.x;

    extern __shared__ float smem[];
    float *q_tile  = smem;
    float *kv_tile = q_tile + Br * D_HEAD;
    float *scores  = kv_tile + Bc * D_HEAD;
    float *row_m   = scores + Br * Bc;
    float *row_l   = row_m + Br;
    float *row_corr = row_l + Br;

    for (int idx = tid; idx < Br * D_HEAD; idx += blockDim.x) {
        int r = idx / D_HEAD;
        int global_row = q_start + r;
        q_tile[idx] = (global_row < N) ? Q[global_row * D_HEAD + (idx % D_HEAD)] : 0.0f;
    }
    for (int i = tid; i < Br; i += blockDim.x) {
        row_m[i] = -FLT_MAX;
        row_l[i] = 0.0f;
    }
    __syncthreads();

    float o_acc[Br];
    for (int i = 0; i < Br; i++) o_acc[i] = 0.0f;

    for (int kv_start = 0; kv_start < N; kv_start += Bc) {
        int kv_end = min(kv_start + Bc, N);
        int tile_len = kv_end - kv_start;

        for (int idx = tid; idx < Bc * D_HEAD; idx += blockDim.x) {
            int r = idx / D_HEAD;
            kv_tile[idx] = (r < tile_len) ? K[(kv_start + r) * D_HEAD + (idx % D_HEAD)] : 0.0f;
        }
        __syncthreads();

        for (int sij = tid; sij < Br * Bc; sij += blockDim.x) {
            int i = sij / Bc;
            int j = sij % Bc;
            if (j < tile_len && (q_start + i) < N) {
                float dot = 0.0f;
                for (int k = 0; k < D_HEAD; k++)
                    dot += q_tile[i * D_HEAD + k] * kv_tile[j * D_HEAD + k];
                scores[sij] = dot * scale;
            } else {
                scores[sij] = -FLT_MAX;
            }
        }
        __syncthreads();

        for (int i = tid; i < Br; i += blockDim.x) {
            if (q_start + i >= N) { row_corr[i] = 1.0f; continue; }
            float old_m = row_m[i];
            float tile_max = -FLT_MAX;
            for (int j = 0; j < tile_len; j++)
                tile_max = fmaxf(tile_max, scores[i * Bc + j]);
            float m_new = fmaxf(old_m, tile_max);
            float corr = (old_m == -FLT_MAX) ? 0.0f : expf(old_m - m_new);
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

        for (int i = 0; i < Br; i++) o_acc[i] *= row_corr[i];

        __syncthreads();
        for (int idx = tid; idx < Bc * D_HEAD; idx += blockDim.x) {
            int r = idx / D_HEAD;
            kv_tile[idx] = (r < tile_len) ? V[(kv_start + r) * D_HEAD + (idx % D_HEAD)] : 0.0f;
        }
        __syncthreads();

        for (int i = 0; i < Br; i++) {
            if (q_start + i >= N) continue;
            for (int j = 0; j < tile_len; j++)
                o_acc[i] += scores[i * Bc + j] * kv_tile[j * D_HEAD + tid];
        }
        __syncthreads();
    }

    for (int i = 0; i < Br; i++) {
        int global_row = q_start + i;
        if (global_row < N) {
            float inv_l = (row_l[i] > 0.0f) ? (1.0f / row_l[i]) : 0.0f;
            O[global_row * D_HEAD + tid] = o_acc[i] * inv_l;
        }
    }
}

int main() {
    printf("=== Phase 3.5: Multi-Head Batched Flash Attention (FP16) ===\n\n");

    int B = 4;     /* batch size */
    int H = 12;    /* number of heads (matches typical transformer) */
    int N = 512;   /* sequence length */
    int d = D_HEAD; /* head dimension */
    float scale = 1.0f / sqrtf((float)d);

    size_t total_elems = (size_t)B * H * N * d;
    size_t fp32_bytes = total_elems * sizeof(float);
    size_t fp16_bytes = total_elems * sizeof(__half);

    printf("Config: B=%d, H=%d, N=%d, d=%d\n", B, H, N, d);
    printf("Total elements: %zu (%.1f M)\n", total_elems, total_elems / 1e6);
    printf("FP32 size: %.1f MB, FP16 size: %.1f MB (2x savings)\n\n",
           fp32_bytes / (1024.0f * 1024.0f), fp16_bytes / (1024.0f * 1024.0f));

    /* Host allocation (FP32 for generation and verification) */
    float *h_Q = (float *)malloc(fp32_bytes);
    float *h_K = (float *)malloc(fp32_bytes);
    float *h_V = (float *)malloc(fp32_bytes);
    float *h_O_ref = (float *)malloc(fp32_bytes);
    float *h_O_fp32 = (float *)malloc(fp32_bytes);
    float *h_O_fp16 = (float *)malloc(fp32_bytes);

    srand(42);
    fill_random(h_Q, total_elems);
    fill_random(h_K, total_elems);
    fill_random(h_V, total_elems);

    /* CPU reference (FP32, multi-head batched) */
    mha_attention_cpu(h_Q, h_K, h_V, h_O_ref, B, H, N, d);

    /* ── Device allocation ── */
    /* FP32 buffers */
    float *d_Q_f32, *d_K_f32, *d_V_f32, *d_O_f32;
    CHECK_CUDA(cudaMalloc(&d_Q_f32, fp32_bytes));
    CHECK_CUDA(cudaMalloc(&d_K_f32, fp32_bytes));
    CHECK_CUDA(cudaMalloc(&d_V_f32, fp32_bytes));
    CHECK_CUDA(cudaMalloc(&d_O_f32, fp32_bytes));

    /* FP16 buffers */
    __half *d_Q_f16, *d_K_f16, *d_V_f16, *d_O_f16;
    CHECK_CUDA(cudaMalloc(&d_Q_f16, fp16_bytes));
    CHECK_CUDA(cudaMalloc(&d_K_f16, fp16_bytes));
    CHECK_CUDA(cudaMalloc(&d_V_f16, fp16_bytes));
    CHECK_CUDA(cudaMalloc(&d_O_f16, fp16_bytes));

    CHECK_CUDA(cudaMemcpy(d_Q_f32, h_Q, fp32_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K_f32, h_K, fp32_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V_f32, h_V, fp32_bytes, cudaMemcpyHostToDevice));

    /* Convert FP32 → FP16 on device */
    int conv_bs = 256;
    int conv_grid = CEIL_DIV(total_elems, conv_bs);
    float_to_half_kernel<<<conv_grid, conv_bs>>>(d_Q_f32, d_Q_f16, total_elems);
    float_to_half_kernel<<<conv_grid, conv_bs>>>(d_K_f32, d_K_f16, total_elems);
    float_to_half_kernel<<<conv_grid, conv_bs>>>(d_V_f32, d_V_f16, total_elems);
    CHECK_CUDA(cudaDeviceSynchronize());

    int smem_size = (Br * d + Bc * d + Br * Bc + 3 * Br) * sizeof(float);

    /* ═══ Test 1: Multi-head FP32 Flash Attention ═══ */
    printf("Correctness tests:\n");

    /* Run per-head FP32 Flash Attention (loop over B*H heads) */
    size_t head_stride = (size_t)N * d;
    size_t batch_stride = (size_t)H * head_stride;
    int grid_per_head = CEIL_DIV(N, Br);

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            size_t off = b * batch_stride + h * head_stride;
            flash_attention_fp32<<<grid_per_head, D_HEAD, smem_size>>>(
                d_Q_f32 + off, d_K_f32 + off, d_V_f32 + off,
                d_O_f32 + off, N, scale);
        }
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_O_fp32, d_O_f32, fp32_bytes, cudaMemcpyDeviceToHost));
    verify(h_O_fp32, h_O_ref, total_elems, 5e-3f, "MH Flash FP32 (loop)");

    /* ═══ Test 2: Multi-head FP16 Flash Attention (3D grid) ═══ */
    dim3 grid_mh(grid_per_head, H, B);
    flash_attention_mh_fp16<<<grid_mh, D_HEAD, smem_size>>>(
        d_Q_f16, d_K_f16, d_V_f16, d_O_f16, B, H, N, scale);
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Convert FP16 output back to FP32 for verification */
    half_to_float_kernel<<<conv_grid, conv_bs>>>(d_O_f16, d_O_f32, total_elems);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_O_fp16, d_O_f32, fp32_bytes, cudaMemcpyDeviceToHost));

    /* FP16 has lower precision — use larger tolerance */
    verify(h_O_fp16, h_O_ref, total_elems, 5e-2f, "MH Flash FP16 (3D grid)");

    /* Also verify FP16 vs FP32 Flash (both GPU) */
    verify(h_O_fp16, h_O_fp32, total_elems, 5e-2f, "FP16 vs FP32 Flash");

    /* ═══ Benchmark ═══ */
    printf("\nBenchmark (B=%d, H=%d, N=%d, d=%d):\n", B, H, N, d);
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    int runs = 50;

    /* FP32 multi-head (loop over heads) */
    for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++) {
            size_t off = b * batch_stride + h * head_stride;
            flash_attention_fp32<<<grid_per_head, D_HEAD, smem_size>>>(
                d_Q_f32 + off, d_K_f32 + off, d_V_f32 + off,
                d_O_f32 + off, N, scale);
        }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int r = 0; r < runs; r++) {
        for (int b = 0; b < B; b++)
            for (int h = 0; h < H; h++) {
                size_t off = b * batch_stride + h * head_stride;
                flash_attention_fp32<<<grid_per_head, D_HEAD, smem_size>>>(
                    d_Q_f32 + off, d_K_f32 + off, d_V_f32 + off,
                    d_O_f32 + off, N, scale);
            }
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_fp32_loop;
    CHECK_CUDA(cudaEventElapsedTime(&ms_fp32_loop, start, stop));
    ms_fp32_loop /= runs;

    /* FP16 multi-head (3D grid — single launch!) */
    flash_attention_mh_fp16<<<grid_mh, D_HEAD, smem_size>>>(
        d_Q_f16, d_K_f16, d_V_f16, d_O_f16, B, H, N, scale);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int r = 0; r < runs; r++) {
        flash_attention_mh_fp16<<<grid_mh, D_HEAD, smem_size>>>(
            d_Q_f16, d_K_f16, d_V_f16, d_O_f16, B, H, N, scale);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_fp16_3d;
    CHECK_CUDA(cudaEventElapsedTime(&ms_fp16_3d, start, stop));
    ms_fp16_3d /= runs;

    printf("  FP32 (loop B*H):     %.3f ms\n", ms_fp32_loop);
    printf("  FP16 (3D grid):      %.3f ms  (%.2fx speedup)\n",
           ms_fp16_3d, ms_fp32_loop / ms_fp16_3d);

    /* Memory comparison */
    printf("\nMemory comparison (Q+K+V+O):\n");
    printf("  FP32: %.1f MB\n", 4.0f * fp32_bytes / (1024.0f * 1024.0f));
    printf("  FP16: %.1f MB  (2x savings)\n", 4.0f * fp16_bytes / (1024.0f * 1024.0f));

    /* Scale analysis — what real models look like */
    printf("\nAt production scale (B=32, H=12, N=2048, d=64):\n");
    size_t prod_fp32 = 4ULL * 32 * 12 * 2048 * 64 * sizeof(float);
    size_t prod_fp16 = 4ULL * 32 * 12 * 2048 * 64 * sizeof(__half);
    size_t prod_naive_attn = 2ULL * 32 * 12 * 2048 * 2048 * sizeof(float);
    printf("  Flash FP32 (Q+K+V+O): %.1f MB\n", prod_fp32 / (1024.0f * 1024.0f));
    printf("  Flash FP16 (Q+K+V+O): %.1f MB\n", prod_fp16 / (1024.0f * 1024.0f));
    printf("  Naive attn matrices:   %.1f MB  (Flash saves %.0fx!)\n",
           prod_naive_attn / (1024.0f * 1024.0f),
           (float)prod_naive_attn / prod_fp16);

    /* Cleanup */
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_Q_f32));
    CHECK_CUDA(cudaFree(d_K_f32));
    CHECK_CUDA(cudaFree(d_V_f32));
    CHECK_CUDA(cudaFree(d_O_f32));
    CHECK_CUDA(cudaFree(d_Q_f16));
    CHECK_CUDA(cudaFree(d_K_f16));
    CHECK_CUDA(cudaFree(d_V_f16));
    CHECK_CUDA(cudaFree(d_O_f16));
    free(h_Q); free(h_K); free(h_V);
    free(h_O_ref); free(h_O_fp32); free(h_O_fp16);

    printf("\n=== Multi-head FP16 Flash Attention complete! ===\n");
    return 0;
}
