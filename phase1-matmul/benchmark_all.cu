/*
 * benchmark_all.cu — Head-to-head comparison of all matmul implementations
 *
 * Runs every kernel at each matrix size and produces a comparison table.
 * This is the Phase 1 deliverable: proving each optimization level's impact.
 *
 * Compile: nvcc -arch=sm_89 -O2 -lcublas -o benchmark_all benchmark_all.cu
 */

#include "common.h"
#include <cublas_v2.h>
#include <string.h>

#define CHECK_CUBLAS(call) do {                                     \
    cublasStatus_t status = call;                                   \
    if (status != CUBLAS_STATUS_SUCCESS) {                          \
        fprintf(stderr, "cuBLAS error at %s:%d: status=%d\n",      \
                __FILE__, __LINE__, status);                        \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
} while(0)

/* ========== Kernel 1: Naive ========== */
#define NAIVE_BS 16

__global__ void kern_naive(const float *A, const float *B, float *C,
                           int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
            sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

void launch_naive(const float *d_A, const float *d_B, float *d_C,
                  int M, int K, int N) {
    dim3 b(NAIVE_BS, NAIVE_BS);
    dim3 g(CEIL_DIV(N, NAIVE_BS), CEIL_DIV(M, NAIVE_BS));
    kern_naive<<<g, b>>>(d_A, d_B, d_C, M, K, N);
}

/* ========== Kernel 2: Tiled (shared memory) ========== */
#define TILE 32

__global__ void kern_tiled(const float *A, const float *B, float *C,
                           int M, int K, int N) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    float sum = 0.0f;
    for (int t = 0; t < CEIL_DIV(K, TILE); t++) {
        int ac = t * TILE + tx, br = t * TILE + ty;
        sA[ty][tx] = (row < M && ac < K) ? A[row * K + ac] : 0.0f;
        sB[ty][tx] = (br < K && col < N) ? B[br * N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; k++)
            sum += sA[ty][k] * sB[k][tx];
        __syncthreads();
    }
    if (row < M && col < N)
        C[row * N + col] = sum;
}

void launch_tiled(const float *d_A, const float *d_B, float *d_C,
                  int M, int K, int N) {
    dim3 b(TILE, TILE);
    dim3 g(CEIL_DIV(N, TILE), CEIL_DIV(M, TILE));
    kern_tiled<<<g, b>>>(d_A, d_B, d_C, M, K, N);
}

/* ========== Kernel 3: Coarsened (4×4 per thread) ========== */
#define C_BK 16
#define C_BM 64
#define C_BN 64
#define C_TM 4
#define C_TN 4

__global__ void kern_coarsened(const float *A, const float *B, float *C,
                               int M, int K, int N) {
    __shared__ float sA[C_BM][C_BK];
    __shared__ float sB[C_BK][C_BN];
    const int tcol = threadIdx.x % (C_BN / C_TN);
    const int trow = threadIdx.x / (C_BN / C_TN);
    const int brow = blockIdx.y * C_BM;
    const int bcol = blockIdx.x * C_BN;
    float res[C_TM * C_TN] = {0};
    float rA[C_TM], rB[C_TN];
    const int nt = (C_BM / C_TM) * (C_BN / C_TN);
    const int al = (C_BM * C_BK) / nt;
    const int bl = (C_BK * C_BN) / nt;

    for (int bk = 0; bk < K; bk += C_BK) {
        for (int l = 0; l < al; l++) {
            int li = threadIdx.x * al + l;
            int sr = li / C_BK, sc = li % C_BK;
            int gr = brow + sr, gc = bk + sc;
            sA[sr][sc] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (int l = 0; l < bl; l++) {
            int li = threadIdx.x * bl + l;
            int sr = li / C_BN, sc = li % C_BN;
            int gr = bk + sr, gc = bcol + sc;
            sB[sr][sc] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        __syncthreads();
        for (int k = 0; k < C_BK; k++) {
            for (int tm = 0; tm < C_TM; tm++)
                rA[tm] = sA[trow * C_TM + tm][k];
            for (int tn = 0; tn < C_TN; tn++)
                rB[tn] = sB[k][tcol * C_TN + tn];
            for (int tm = 0; tm < C_TM; tm++)
                for (int tn = 0; tn < C_TN; tn++)
                    res[tm * C_TN + tn] += rA[tm] * rB[tn];
        }
        __syncthreads();
    }
    for (int tm = 0; tm < C_TM; tm++)
        for (int tn = 0; tn < C_TN; tn++) {
            int or_ = brow + trow * C_TM + tm;
            int oc = bcol + tcol * C_TN + tn;
            if (or_ < M && oc < N) C[or_ * N + oc] = res[tm * C_TN + tn];
        }
}

void launch_coarsened(const float *d_A, const float *d_B, float *d_C,
                      int M, int K, int N) {
    dim3 b((C_BM / C_TM) * (C_BN / C_TN));
    dim3 g(CEIL_DIV(N, C_BN), CEIL_DIV(M, C_BM));
    kern_coarsened<<<g, b>>>(d_A, d_B, d_C, M, K, N);
}

/* ========== Kernel 4: Coarsened 8×8 (larger register tile) ========== */
#define V_BK 16
#define V_BM 128
#define V_BN 128
#define V_TM 8
#define V_TN 8

__global__ void kern_vec(const float *A, const float *B, float *C,
                         int M, int K, int N) {
    __shared__ float sA[V_BM][V_BK];
    __shared__ float sB[V_BK][V_BN];
    const int tcol = threadIdx.x % (V_BN / V_TN);
    const int trow = threadIdx.x / (V_BN / V_TN);
    const int brow = blockIdx.y * V_BM;
    const int bcol = blockIdx.x * V_BN;
    float res[V_TM * V_TN] = {0};
    float rA[V_TM], rB[V_TN];
    const int nt = (V_BM / V_TM) * (V_BN / V_TN);
    const int al = (V_BM * V_BK) / nt;
    const int bl = (V_BK * V_BN) / nt;

    for (int bk = 0; bk < K; bk += V_BK) {
        for (int l = 0; l < al; l++) {
            int li = threadIdx.x * al + l;
            int sr = li / V_BK, sc = li % V_BK;
            int gr = brow + sr, gc = bk + sc;
            sA[sr][sc] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (int l = 0; l < bl; l++) {
            int li = threadIdx.x * bl + l;
            int sr = li / V_BN, sc = li % V_BN;
            int gr = bk + sr, gc = bcol + sc;
            sB[sr][sc] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        __syncthreads();
        for (int k = 0; k < V_BK; k++) {
            for (int tm = 0; tm < V_TM; tm++)
                rA[tm] = sA[trow * V_TM + tm][k];
            for (int tn = 0; tn < V_TN; tn++)
                rB[tn] = sB[k][tcol * V_TN + tn];
            for (int tm = 0; tm < V_TM; tm++)
                for (int tn = 0; tn < V_TN; tn++)
                    res[tm * V_TN + tn] += rA[tm] * rB[tn];
        }
        __syncthreads();
    }
    for (int tm = 0; tm < V_TM; tm++)
        for (int tn = 0; tn < V_TN; tn++) {
            int or_ = brow + trow * V_TM + tm;
            int oc = bcol + tcol * V_TN + tn;
            if (or_ < M && oc < N) C[or_ * N + oc] = res[tm * V_TN + tn];
        }
}

void launch_vec(const float *d_A, const float *d_B, float *d_C,
                int M, int K, int N) {
    dim3 b((V_BM / V_TM) * (V_BN / V_TN));
    dim3 g(CEIL_DIV(N, V_BN), CEIL_DIV(M, V_BM));
    kern_vec<<<g, b>>>(d_A, d_B, d_C, M, K, N);
}

/* ========== Kernel 5: cuBLAS ========== */
static cublasHandle_t handle = NULL;

void launch_cublas(const float *d_A, const float *d_B, float *d_C,
                   int M, int K, int N) {
    if (!handle) CHECK_CUBLAS(cublasCreate(&handle));
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
}

int main() {
    printf("╔══════════════════════════════════════════════════════════════════════════╗\n");
    printf("║         Phase 1: Matmul Optimization Ladder — Full Benchmark           ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════════╝\n\n");
    printf("RTX 4070 (sm_89) | FP32 peak: %.1f TFLOPS | Memory BW: ~504 GB/s\n\n", RTX4070_PEAK_TFLOPS);

    typedef struct {
        const char *name;
        KernelLauncher fn;
    } KernelEntry;

    KernelEntry kernels[] = {
        {"1-naive-16x16",    launch_naive},
        {"2-tiled-32x32",    launch_tiled},
        {"3-coarse-4x4",     launch_coarsened},
        {"4-coarse-8x8",     launch_vec},
        {"5-cuBLAS-FP32",    launch_cublas},
    };
    int num_kernels = sizeof(kernels) / sizeof(kernels[0]);

    int sizes[] = {256, 512, 1024, 2048, 4096};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int si = 0; si < num_sizes; si++) {
        int S = sizes[si];
        size_t n = (size_t)S * S;

        float *h_A = (float *)malloc(n * sizeof(float));
        float *h_B = (float *)malloc(n * sizeof(float));
        float *h_C_ref = NULL;
        fill_random(h_A, n);
        fill_random(h_B, n);

        if (S <= 1024) {
            h_C_ref = (float *)malloc(n * sizeof(float));
            matmul_cpu(h_A, h_B, h_C_ref, S, S, S);
        }

        printf("── Matrix size: %dx%d ─────────────────────────────────────────\n", S, S);
        printf("  %-20s | %10s | %14s | %5s | %6s | %s\n",
               "Kernel", "Time", "GFLOPS", "Peak%", "vs #1", "Check");
        printf("  --------------------|------------|----------------|-------|--------|------\n");

        double baseline_ms = 0;
        for (int ki = 0; ki < num_kernels; ki++) {
            BenchResult r = benchmark_kernel(kernels[ki].fn, S, S, S,
                                             h_A, h_B, h_C_ref, 3, 10);
            if (ki == 0) baseline_ms = r.avg_ms;

            const char *status = r.pass == 1 ? "PASS" : (r.pass == 0 ? "FAIL" : "N/A");
            printf("  %-20s | %8.3f ms | %8.1f GFLOPS | %5.1f%% | %5.1fx | %s\n",
                   kernels[ki].name, r.avg_ms, r.gflops, r.pct_peak,
                   baseline_ms / r.avg_ms, status);
        }
        printf("\n");

        free(h_A);
        free(h_B);
        free(h_C_ref);
    }

    if (handle) cublasDestroy(handle);

    printf("═══════════════════════════════════════════════════════════════════════════\n");
    printf("Summary: Each optimization level should show clear improvement.\n");
    printf("  naive → tiled:     ~3-10x (shared memory eliminates redundant reads)\n");
    printf("  tiled → coarsened: ~1.5-3x (register tiling, more work per thread)\n");
    printf("  coarsened → cuBLAS: ~1.5-3x (hand-tuned assembly, auto-tuning)\n");
    printf("═══════════════════════════════════════════════════════════════════════════\n");

    return 0;
}
