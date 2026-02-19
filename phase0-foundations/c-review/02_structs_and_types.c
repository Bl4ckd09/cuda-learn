/*
 * 02_structs_and_types.c — Custom types and sizeof for CUDA programming
 *
 * Why this matters for CUDA:
 *   - sizeof() is used EVERYWHERE: cudaMalloc(&d_A, N * sizeof(float))
 *   - Structs can be passed to kernels (but prefer flat arrays for perf)
 *   - Understanding type sizes matters for memory coalescing and alignment
 *   - float (4B) vs double (8B) vs __half (2B, FP16) — each has different
 *     bandwidth characteristics on GPU
 *
 * Compile: gcc -Wall -o 02_structs 02_structs_and_types.c
 * Run:     ./02_structs
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------- Exercise 1: sizeof — know your type sizes ---------- */
void type_sizes() {
    printf("=== Exercise 1: Type Sizes ===\n");
    printf("  char:        %zu bytes\n", sizeof(char));
    printf("  short:       %zu bytes\n", sizeof(short));
    printf("  int:         %zu bytes\n", sizeof(int));
    printf("  long:        %zu bytes\n", sizeof(long));
    printf("  long long:   %zu bytes\n", sizeof(long long));
    printf("  float:       %zu bytes   <-- most common in GPU computing\n", sizeof(float));
    printf("  double:      %zu bytes   <-- 2x bandwidth cost on GPU\n", sizeof(double));
    printf("  size_t:      %zu bytes   <-- unsigned, used for sizes\n", sizeof(size_t));
    printf("  void*:       %zu bytes   <-- pointer size (64-bit system)\n", sizeof(void *));

    // Array sizeof
    float arr[256];
    printf("  float[256]:  %zu bytes = %zu * %zu\n\n",
           sizeof(arr), sizeof(arr) / sizeof(arr[0]), sizeof(float));
}

/* ---------- Exercise 2: Structs — data organization ---------- */
/*
 * In CUDA, you might define a config struct:
 *   struct KernelConfig { int N; int block_size; float learning_rate; };
 * and pass it to a kernel. But for hot-path data (like matrices),
 * always use flat arrays for memory coalescing.
 */

typedef struct {
    float x, y, z;
} Vec3;

typedef struct {
    int rows;
    int cols;
    float *data;   // pointer to flat array — THE pattern for CUDA matrices
} Matrix;

Matrix matrix_create(int rows, int cols) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.data = (float *)malloc(rows * cols * sizeof(float));
    return m;
}

void matrix_fill(Matrix *m, float val) {
    for (int i = 0; i < m->rows * m->cols; i++)
        m->data[i] = val;
}

float matrix_get(const Matrix *m, int row, int col) {
    return m->data[row * m->cols + col];  // row-major indexing
}

void matrix_set(Matrix *m, int row, int col, float val) {
    m->data[row * m->cols + col] = val;
}

void matrix_print(const Matrix *m, const char *name) {
    printf("  %s (%dx%d):\n", name, m->rows, m->cols);
    for (int i = 0; i < m->rows && i < 4; i++) {
        printf("    ");
        for (int j = 0; j < m->cols && j < 6; j++)
            printf("%6.1f ", matrix_get(m, i, j));
        if (m->cols > 6) printf("...");
        printf("\n");
    }
    if (m->rows > 4) printf("    ...\n");
}

void matrix_free(Matrix *m) {
    free(m->data);
    m->data = NULL;
}

void structs_exercise() {
    printf("=== Exercise 2: Structs ===\n");

    // Simple struct
    Vec3 v = {1.0f, 2.0f, 3.0f};
    printf("  Vec3: (%.1f, %.1f, %.1f), sizeof = %zu bytes\n",
           v.x, v.y, v.z, sizeof(Vec3));

    // Matrix struct — the pattern you'll use in CUDA
    Matrix A = matrix_create(4, 4);
    matrix_fill(&A, 0.0f);
    for (int i = 0; i < 4; i++)
        matrix_set(&A, i, i, 1.0f);  // identity matrix

    matrix_print(&A, "Identity A");
    printf("  A[2][2] = %.1f (accessed via flat index %d)\n\n",
           matrix_get(&A, 2, 2), 2 * A.cols + 2);

    matrix_free(&A);
}

/* ---------- Exercise 3: typedef and type aliasing ---------- */
/*
 * CUDA uses typedefs heavily:
 *   cudaError_t   = enum for error codes
 *   cudaStream_t  = pointer to opaque stream struct
 *   dim3          = struct { unsigned int x, y, z; }
 */
typedef unsigned int uint;
typedef float *FloatPtr;

void typedef_exercise() {
    printf("=== Exercise 3: typedef ===\n");

    uint block_size = 256;
    printf("  block_size (uint): %u\n", block_size);

    // Simulating CUDA's dim3 struct
    typedef struct {
        unsigned int x, y, z;
    } dim3_sim;

    dim3_sim grid = {16, 16, 1};
    dim3_sim block = {32, 32, 1};
    unsigned int total_threads = grid.x * grid.y * grid.z * block.x * block.y * block.z;
    printf("  grid(%u,%u,%u) x block(%u,%u,%u) = %u total threads\n\n",
           grid.x, grid.y, grid.z, block.x, block.y, block.z, total_threads);
}

/* ---------- Exercise 4: Memory layout and alignment ---------- */
/*
 * GPU memory coalescing requires aligned, contiguous access.
 * Struct padding can waste bandwidth.
 */
struct Unpadded {
    char a;     // 1 byte
    int b;      // 4 bytes — but starts at offset 4 (padded!)
    char c;     // 1 byte — at offset 8, padded to 12 total
};

struct __attribute__((packed)) Packed {
    char a;
    int b;
    char c;
};

void alignment_exercise() {
    printf("=== Exercise 4: Memory Alignment ===\n");

    printf("  struct Unpadded: sizeof = %zu (with padding)\n", sizeof(struct Unpadded));
    printf("  struct Packed:   sizeof = %zu (no padding, but slower access!)\n",
           sizeof(struct Packed));
    printf("  Lesson: GPU prefers aligned access. Use flat float arrays\n");
    printf("  for matrices, not arrays of structs.\n\n");
}

int main() {
    type_sizes();
    structs_exercise();
    typedef_exercise();
    alignment_exercise();

    printf("=== All struct/type exercises complete! ===\n");
    return 0;
}
