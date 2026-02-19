/*
 * 04_flash_attention_backward.cu — Flash Attention Backward Pass (FP32)
 *
 * The backward pass of Flash Attention recomputes the attention matrix
 * tile-by-tile rather than storing it from the forward pass.
 *
 * Given: Q, K, V, O (forward output), dO (upstream gradient)
 * Compute: dQ, dK, dV
 *
 * Standard attention backward (with materialized P):
 *   dV = P^T @ dO              [N,d]
 *   dP = dO @ V^T              [N,N]
 *   D_i = rowsum(dO * O)       [N] (the "D" vector, per-row dot product)
 *   dS = P * (dP - D)          [N,N]  (softmax backward)
 *   dQ = dS @ K                [N,d]
 *   dK = dS^T @ Q              [N,d]
 *
 * Flash Attention backward: same math, but computed tile-by-tile.
 * For each (Q_tile, KV_tile):
 *   1. Recompute S_tile = Q_tile @ K_tile^T * scale
 *   2. Recompute P_tile = exp(S_tile - m) / l   (using saved m, l from forward)
 *   3. Compute local dV_tile += P_tile^T @ dO_tile
 *   4. Compute local dP_tile = dO_tile @ V_tile^T
 *   5. Compute dS_tile = P_tile * (dP_tile - D_i)
 *   6. Accumulate dQ_tile += dS_tile @ K_tile
 *   7. Accumulate dK_tile += dS_tile^T @ Q_tile
 *
 * For this learning implementation, we implement a CPU-based backward
 * to verify gradients, then a CUDA backward using the tiled approach.
 *
 * Compile: nvcc -arch=sm_89 -O2 -lcublas -o 04_flash_attention_backward 04_flash_attention_backward.cu
 */

#include "common.h"

/* ════════════════════════════════════════════════════════════
 * CPU reference: standard attention backward (materialized)
 * ════════════════════════════════════════════════════════════ */
void attention_backward_cpu(const float *Q, const float *K, const float *V,
                            const float *O, const float *dO,
                            float *dQ, float *dK, float *dV,
                            int N, int d) {
    float scale = 1.0f / sqrtf((float)d);
    size_t nn = (size_t)N * N;

    float *S = (float *)calloc(nn, sizeof(float));
    float *P = (float *)calloc(nn, sizeof(float));
    float *dP = (float *)calloc(nn, sizeof(float));
    float *dS = (float *)calloc(nn, sizeof(float));
    float *D  = (float *)calloc(N,  sizeof(float));

    /* Forward: S = Q @ K^T * scale */
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float dot = 0.0f;
            for (int k = 0; k < d; k++)
                dot += Q[i * d + k] * K[j * d + k];
            S[i * N + j] = dot * scale;
        }

    /* Forward: P = softmax(S) */
    for (int i = 0; i < N; i++) {
        float mx = -FLT_MAX;
        for (int j = 0; j < N; j++) mx = fmaxf(mx, S[i * N + j]);
        float sm = 0.0f;
        for (int j = 0; j < N; j++) {
            P[i * N + j] = expf(S[i * N + j] - mx);
            sm += P[i * N + j];
        }
        for (int j = 0; j < N; j++) P[i * N + j] /= sm;
    }

    /* D[i] = rowsum(dO[i] * O[i]) */
    for (int i = 0; i < N; i++) {
        float sum = 0.0f;
        for (int k = 0; k < d; k++)
            sum += dO[i * d + k] * O[i * d + k];
        D[i] = sum;
    }

    /* dV = P^T @ dO */
    memset(dV, 0, (size_t)N * d * sizeof(float));
    for (int j = 0; j < N; j++)
        for (int k = 0; k < d; k++) {
            float acc = 0.0f;
            for (int i = 0; i < N; i++)
                acc += P[i * N + j] * dO[i * d + k];
            dV[j * d + k] = acc;
        }

    /* dP = dO @ V^T */
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float dot = 0.0f;
            for (int k = 0; k < d; k++)
                dot += dO[i * d + k] * V[j * d + k];
            dP[i * N + j] = dot;
        }

    /* dS = P * (dP - D) */
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            dS[i * N + j] = P[i * N + j] * (dP[i * N + j] - D[i]);

    /* dS = dS * scale (chain rule from S = QK^T * scale) */
    /* Actually: dQ = dS @ K * scale, dK = dS^T @ Q * scale
     * But dS already comes from softmax backward, and the scale
     * was applied during forward. So: dQ = dS @ K, dK = dS^T @ Q.
     * Wait — let me be more careful.
     *
     * Forward: S = QK^T * scale, P = softmax(S), O = PV
     * Backward:
     *   dS = softmax_backward(dP, P) = P * (dP - D)
     *   dQ = dS @ K * scale   (because dS/dQ = K^T * scale)
     *   dK = dS^T @ Q * scale (because dS/dK = Q * scale, transposed)
     *
     * Actually, more precisely: S_ij = scale * sum_k Q_ik * K_jk
     *   dQ_ik = scale * sum_j dS_ij * K_jk
     *   dK_jk = scale * sum_i dS_ij * Q_ik
     */

    /* dQ = dS @ K * scale */
    memset(dQ, 0, (size_t)N * d * sizeof(float));
    for (int i = 0; i < N; i++)
        for (int k = 0; k < d; k++) {
            float acc = 0.0f;
            for (int j = 0; j < N; j++)
                acc += dS[i * N + j] * K[j * d + k];
            dQ[i * d + k] = acc * scale;
        }

    /* dK = dS^T @ Q * scale */
    memset(dK, 0, (size_t)N * d * sizeof(float));
    for (int j = 0; j < N; j++)
        for (int k = 0; k < d; k++) {
            float acc = 0.0f;
            for (int i = 0; i < N; i++)
                acc += dS[i * N + j] * Q[i * d + k];
            dK[j * d + k] = acc * scale;
        }

    free(S); free(P); free(dP); free(dS); free(D);
}

/* ════════════════════════════════════════════════════════════
 * CUDA Flash Attention Backward (tiled, recomputing P)
 * ════════════════════════════════════════════════════════════
 *
 * Design: Two-kernel approach
 *   Kernel 1 (dV kernel): Loop over Q tiles as outer, KV tile as the block.
 *     Each block "owns" a KV tile and accumulates dK, dV.
 *   Kernel 2 (dQ kernel): Loop over KV tiles as outer, Q tile as the block.
 *     Each block "owns" a Q tile and accumulates dQ.
 *
 * For simplicity in this learning implementation, we use a single kernel
 * that processes one query row at a time (Br=1), looping over KV tiles.
 * This mirrors our 02_fused_attention approach.
 */

#define Bc 32
#define D_HEAD 64

/*
 * Kernel: Compute dQ for one query row at a time.
 * Also atomically accumulate dK, dV.
 *
 * For query row qi:
 *   1. Compute D[qi] = dot(dO[qi], O[qi])
 *   2. Loop over KV tiles:
 *      a. Recompute s[j] = dot(q, k[j]) * scale
 *      b. Recompute p[j] = exp(s[j] - m) / l   (m, l saved from forward)
 *      c. dv[j] += p[j] * dO[qi]  (accumulate dV)
 *      d. dp[j] = dot(dO[qi], v[j])
 *      e. ds[j] = p[j] * (dp[j] - D[qi])
 *      f. dq[qi] += ds[j] * k[j] * scale  (accumulate dQ)
 *      g. dk[j] += ds[j] * q[qi] * scale  (accumulate dK)
 */
__global__ void flash_attention_bwd(
    const float *Q, const float *K, const float *V,
    const float *O, const float *dO,
    const float *row_m, const float *row_l,  /* saved from forward */
    float *dQ, float *dK, float *dV,
    int N, float scale)
{
    int qi = blockIdx.x;  /* one block per query row */
    int tid = threadIdx.x; /* 0..D_HEAD-1 */

    if (qi >= N) return;

    extern __shared__ float smem[];
    float *kv_tile = smem;           /* [Bc * D_HEAD] */
    float *scores  = kv_tile + Bc * D_HEAD;  /* [Bc] */
    float *dp_tile = scores + Bc;    /* [Bc] */

    /* Load query row and dO row into registers */
    float q_val = Q[qi * D_HEAD + tid];
    float do_val = dO[qi * D_HEAD + tid];
    float o_val = O[qi * D_HEAD + tid];

    /* Compute D[qi] = dot(dO[qi], O[qi]) — need reduction across threads */
    /* Use shared memory for reduction */
    float *reduction = dp_tile + Bc;  /* reuse space for reduction */
    reduction[tid] = do_val * o_val;
    __syncthreads();
    /* Simple reduction (d=64, 2 warps) */
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) reduction[tid] += reduction[tid + s];
        __syncthreads();
    }
    float D_qi = reduction[0];

    float my_m = row_m[qi];
    float my_l = row_l[qi];
    float inv_l = 1.0f / my_l;

    /* Accumulate dQ in register */
    float dq_acc = 0.0f;

    /* Loop over KV tiles */
    for (int kv_start = 0; kv_start < N; kv_start += Bc) {
        int kv_end = min(kv_start + Bc, N);
        int tile_len = kv_end - kv_start;

        /* Load K tile */
        for (int idx = tid; idx < Bc * D_HEAD; idx += blockDim.x) {
            int r = idx / D_HEAD;
            kv_tile[idx] = (r < tile_len) ? K[(kv_start + r) * D_HEAD + (idx % D_HEAD)] : 0.0f;
        }
        __syncthreads();

        /* Recompute scores and probabilities */
        /* Each thread computes partial dot product for one j, reduces */
        for (int j = 0; j < tile_len; j++) {
            /* dot(q, k[j]) — each thread has one dimension */
            reduction[tid] = q_val * kv_tile[j * D_HEAD + tid];
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) reduction[tid] += reduction[tid + s];
                __syncthreads();
            }
            if (tid == 0) {
                float s_val = reduction[0] * scale;
                float p_val = expf(s_val - my_m) * inv_l;
                scores[j] = p_val;
            }
            __syncthreads();
        }

        /* dV[j] += p[j] * dO[qi] — each thread handles its dimension */
        for (int j = 0; j < tile_len; j++) {
            atomicAdd(&dV[(kv_start + j) * D_HEAD + tid], scores[j] * do_val);
        }

        /* Compute dp[j] = dot(dO[qi], v[j]) */
        /* Load V tile */
        for (int idx = tid; idx < Bc * D_HEAD; idx += blockDim.x) {
            int r = idx / D_HEAD;
            kv_tile[idx] = (r < tile_len) ? V[(kv_start + r) * D_HEAD + (idx % D_HEAD)] : 0.0f;
        }
        __syncthreads();

        for (int j = 0; j < tile_len; j++) {
            reduction[tid] = do_val * kv_tile[j * D_HEAD + tid];
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) reduction[tid] += reduction[tid + s];
                __syncthreads();
            }
            if (tid == 0) {
                dp_tile[j] = reduction[0];
            }
            __syncthreads();
        }

        /* ds[j] = p[j] * (dp[j] - D_qi) */
        /* dQ[qi] += ds[j] * K[j] * scale */
        /* dK[j] += ds[j] * Q[qi] * scale */

        /* Reload K tile for dQ/dK computation */
        for (int idx = tid; idx < Bc * D_HEAD; idx += blockDim.x) {
            int r = idx / D_HEAD;
            kv_tile[idx] = (r < tile_len) ? K[(kv_start + r) * D_HEAD + (idx % D_HEAD)] : 0.0f;
        }
        __syncthreads();

        for (int j = 0; j < tile_len; j++) {
            float ds_j = scores[j] * (dp_tile[j] - D_qi);

            /* dQ[qi][tid] += ds_j * K[j][tid] * scale */
            dq_acc += ds_j * kv_tile[j * D_HEAD + tid] * scale;

            /* dK[j][tid] += ds_j * Q[qi][tid] * scale */
            atomicAdd(&dK[(kv_start + j) * D_HEAD + tid], ds_j * q_val * scale);
        }
        __syncthreads();
    }

    /* Write dQ */
    dQ[qi * D_HEAD + tid] = dq_acc;
}

/* ═══════════ Forward pass to get O, m, l ═══════════ */
/* Simplified forward that also saves per-row m and l */
__global__ void flash_attention_fwd_save(
    const float *Q, const float *K, const float *V,
    float *O, float *row_m_out, float *row_l_out,
    int N, float scale)
{
    int qi = blockIdx.x;
    int tid = threadIdx.x;

    if (qi >= N) return;

    /*
     * Shared memory layout:
     *   kv_tile  [Bc * D_HEAD]
     *   scores   [Bc]           — softmax probabilities for current tile
     *   reduction[D_HEAD]       — for dot product reductions
     *   sm_state [3]            — {m, l, correction} for cross-thread communication
     */
    extern __shared__ float smem[];
    float *kv_tile  = smem;
    float *scores   = kv_tile + Bc * D_HEAD;
    float *reduction = scores + Bc;
    float *sm_state  = reduction + D_HEAD;  /* [0]=m, [1]=l, [2]=correction */

    float q_val = Q[qi * D_HEAD + tid];

    /* Initialize shared softmax state */
    if (tid == 0) {
        sm_state[0] = -FLT_MAX;  /* m */
        sm_state[1] = 0.0f;       /* l */
    }
    __syncthreads();

    float o_acc = 0.0f;

    for (int kv_start = 0; kv_start < N; kv_start += Bc) {
        int kv_end = min(kv_start + Bc, N);
        int tile_len = kv_end - kv_start;

        /* Load K tile */
        for (int idx = tid; idx < Bc * D_HEAD; idx += blockDim.x) {
            int r = idx / D_HEAD;
            kv_tile[idx] = (r < tile_len) ? K[(kv_start + r) * D_HEAD + (idx % D_HEAD)] : 0.0f;
        }
        __syncthreads();

        /* Compute scores via cross-thread reduction (tid==0 writes) */
        for (int j = 0; j < tile_len; j++) {
            reduction[tid] = q_val * kv_tile[j * D_HEAD + tid];
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) reduction[tid] += reduction[tid + s];
                __syncthreads();
            }
            if (tid == 0) scores[j] = reduction[0] * scale;
            __syncthreads();
        }

        /* Online softmax — ONLY thread 0 modifies shared scores and state */
        if (tid == 0) {
            float old_m = sm_state[0];
            float old_l = sm_state[1];

            float tile_max = -FLT_MAX;
            for (int j = 0; j < tile_len; j++)
                tile_max = fmaxf(tile_max, scores[j]);
            float m_new = fmaxf(old_m, tile_max);
            float corr = (old_m == -FLT_MAX) ? 0.0f : expf(old_m - m_new);

            float tile_sum = 0.0f;
            for (int j = 0; j < tile_len; j++) {
                scores[j] = expf(scores[j] - m_new);
                tile_sum += scores[j];
            }

            sm_state[0] = m_new;
            sm_state[1] = old_l * corr + tile_sum;
            sm_state[2] = corr;
        }
        __syncthreads();

        /* ALL threads apply correction to their accumulator */
        o_acc *= sm_state[2];

        /* Load V tile */
        __syncthreads();
        for (int idx = tid; idx < Bc * D_HEAD; idx += blockDim.x) {
            int r = idx / D_HEAD;
            kv_tile[idx] = (r < tile_len) ? V[(kv_start + r) * D_HEAD + (idx % D_HEAD)] : 0.0f;
        }
        __syncthreads();

        /* Accumulate output */
        for (int j = 0; j < tile_len; j++)
            o_acc += scores[j] * kv_tile[j * D_HEAD + tid];
        __syncthreads();
    }

    float inv_l = (sm_state[1] > 0.0f) ? (1.0f / sm_state[1]) : 0.0f;
    O[qi * D_HEAD + tid] = o_acc * inv_l;

    if (tid == 0) {
        row_m_out[qi] = sm_state[0];
        row_l_out[qi] = sm_state[1];
    }
}

int main() {
    printf("=== Phase 3.4: Flash Attention Backward ===\n\n");

    int N = 256;  /* smaller for backward testing (O(N^2) CPU reference) */
    int d = D_HEAD;
    float scale = 1.0f / sqrtf((float)d);

    size_t qkv_bytes = (size_t)N * d * sizeof(float);

    printf("Config: N=%d, d=%d, Bc=%d\n\n", N, d, Bc);

    /* Allocate host */
    float *h_Q = (float *)malloc(qkv_bytes);
    float *h_K = (float *)malloc(qkv_bytes);
    float *h_V = (float *)malloc(qkv_bytes);
    float *h_dO = (float *)malloc(qkv_bytes);

    float *h_O = (float *)malloc(qkv_bytes);
    float *h_dQ_ref = (float *)malloc(qkv_bytes);
    float *h_dK_ref = (float *)malloc(qkv_bytes);
    float *h_dV_ref = (float *)malloc(qkv_bytes);

    float *h_dQ_gpu = (float *)malloc(qkv_bytes);
    float *h_dK_gpu = (float *)malloc(qkv_bytes);
    float *h_dV_gpu = (float *)malloc(qkv_bytes);

    srand(42);
    fill_random(h_Q, N * d);
    fill_random(h_K, N * d);
    fill_random(h_V, N * d);
    fill_random(h_dO, N * d);

    /* CPU forward to get O */
    attention_cpu(h_Q, h_K, h_V, h_O, N, d);

    /* CPU backward reference */
    attention_backward_cpu(h_Q, h_K, h_V, h_O, h_dO,
                           h_dQ_ref, h_dK_ref, h_dV_ref, N, d);

    /* Device allocation */
    float *d_Q, *d_K, *d_V, *d_O, *d_dO;
    float *d_dQ, *d_dK, *d_dV;
    float *d_row_m, *d_row_l;

    CHECK_CUDA(cudaMalloc(&d_Q, qkv_bytes));
    CHECK_CUDA(cudaMalloc(&d_K, qkv_bytes));
    CHECK_CUDA(cudaMalloc(&d_V, qkv_bytes));
    CHECK_CUDA(cudaMalloc(&d_O, qkv_bytes));
    CHECK_CUDA(cudaMalloc(&d_dO, qkv_bytes));
    CHECK_CUDA(cudaMalloc(&d_dQ, qkv_bytes));
    CHECK_CUDA(cudaMalloc(&d_dK, qkv_bytes));
    CHECK_CUDA(cudaMalloc(&d_dV, qkv_bytes));
    CHECK_CUDA(cudaMalloc(&d_row_m, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_row_l, N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_Q, h_Q, qkv_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K, qkv_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V, qkv_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_dO, h_dO, qkv_bytes, cudaMemcpyHostToDevice));

    /* Forward pass to get O, m, l */
    int smem_fwd = (Bc * d + Bc + D_HEAD + 3) * sizeof(float);
    flash_attention_fwd_save<<<N, D_HEAD, smem_fwd>>>(
        d_Q, d_K, d_V, d_O, d_row_m, d_row_l, N, scale);
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Verify forward pass */
    float *h_O_gpu = (float *)malloc(qkv_bytes);
    CHECK_CUDA(cudaMemcpy(h_O_gpu, d_O, qkv_bytes, cudaMemcpyDeviceToHost));
    printf("Forward pass:\n");
    verify(h_O_gpu, h_O, N * d, 1e-3f, "flash-fwd (Br=1)");

    /* Backward pass */
    CHECK_CUDA(cudaMemset(d_dQ, 0, qkv_bytes));
    CHECK_CUDA(cudaMemset(d_dK, 0, qkv_bytes));
    CHECK_CUDA(cudaMemset(d_dV, 0, qkv_bytes));

    int smem_bwd = (Bc * d + Bc + Bc + D_HEAD) * sizeof(float);
    flash_attention_bwd<<<N, D_HEAD, smem_bwd>>>(
        d_Q, d_K, d_V, d_O, d_dO,
        d_row_m, d_row_l,
        d_dQ, d_dK, d_dV,
        N, scale);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_dQ_gpu, d_dQ, qkv_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_dK_gpu, d_dK, qkv_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_dV_gpu, d_dV, qkv_bytes, cudaMemcpyDeviceToHost));

    printf("\nBackward pass:\n");
    verify(h_dQ_gpu, h_dQ_ref, N * d, 5e-3f, "dQ");
    verify(h_dK_gpu, h_dK_ref, N * d, 5e-3f, "dK");
    verify(h_dV_gpu, h_dV_ref, N * d, 5e-3f, "dV");

    /* Benchmark */
    printf("\nBenchmark (N=%d, d=%d):\n", N, d);
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    int runs = 20;

    /* Forward */
    flash_attention_fwd_save<<<N, D_HEAD, smem_fwd>>>(
        d_Q, d_K, d_V, d_O, d_row_m, d_row_l, N, scale);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(start));
    for (int r = 0; r < runs; r++)
        flash_attention_fwd_save<<<N, D_HEAD, smem_fwd>>>(
            d_Q, d_K, d_V, d_O, d_row_m, d_row_l, N, scale);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_fwd;
    CHECK_CUDA(cudaEventElapsedTime(&ms_fwd, start, stop));
    ms_fwd /= runs;

    /* Backward */
    CHECK_CUDA(cudaMemset(d_dQ, 0, qkv_bytes));
    CHECK_CUDA(cudaMemset(d_dK, 0, qkv_bytes));
    CHECK_CUDA(cudaMemset(d_dV, 0, qkv_bytes));
    flash_attention_bwd<<<N, D_HEAD, smem_bwd>>>(
        d_Q, d_K, d_V, d_O, d_dO, d_row_m, d_row_l,
        d_dQ, d_dK, d_dV, N, scale);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(start));
    for (int r = 0; r < runs; r++) {
        CHECK_CUDA(cudaMemset(d_dQ, 0, qkv_bytes));
        CHECK_CUDA(cudaMemset(d_dK, 0, qkv_bytes));
        CHECK_CUDA(cudaMemset(d_dV, 0, qkv_bytes));
        flash_attention_bwd<<<N, D_HEAD, smem_bwd>>>(
            d_Q, d_K, d_V, d_O, d_dO, d_row_m, d_row_l,
            d_dQ, d_dK, d_dV, N, scale);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_bwd;
    CHECK_CUDA(cudaEventElapsedTime(&ms_bwd, start, stop));
    ms_bwd /= runs;

    printf("  Forward:  %.3f ms\n", ms_fwd);
    printf("  Backward: %.3f ms (%.1fx forward)\n", ms_bwd, ms_bwd / ms_fwd);
    printf("  Backward recomputes attention tiles — trades compute for memory.\n");

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_O));
    CHECK_CUDA(cudaFree(d_dO));
    CHECK_CUDA(cudaFree(d_dQ));
    CHECK_CUDA(cudaFree(d_dK));
    CHECK_CUDA(cudaFree(d_dV));
    CHECK_CUDA(cudaFree(d_row_m));
    CHECK_CUDA(cudaFree(d_row_l));
    free(h_Q); free(h_K); free(h_V); free(h_dO); free(h_O);
    free(h_O_gpu);
    free(h_dQ_ref); free(h_dK_ref); free(h_dV_ref);
    free(h_dQ_gpu); free(h_dK_gpu); free(h_dV_gpu);

    printf("\n=== Flash Attention Backward complete! ===\n");
    return 0;
}
