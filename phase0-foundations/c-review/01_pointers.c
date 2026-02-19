/*
 * 01_pointers.c — Pointer fundamentals for CUDA programming
 *
 * Why this matters for CUDA:
 *   - cudaMalloc() returns a pointer to GPU memory
 *   - cudaMemcpy() takes src/dst pointers (just like memcpy)
 *   - Kernel arguments are passed as pointers to device memory
 *   - Understanding pointer arithmetic is essential for 2D array indexing
 *
 * Compile: gcc -Wall -o 01_pointers 01_pointers.c
 * Run:     ./01_pointers
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------- Exercise 1: Basics — address-of and dereference ---------- */
void basics() {
    printf("=== Exercise 1: Pointer Basics ===\n");

    int x = 42;
    int *p = &x;       // p stores the ADDRESS of x

    printf("Value of x:          %d\n", x);
    printf("Address of x (&x):   %p\n", (void *)&x);
    printf("Value of p:          %p   (same as &x)\n", (void *)p);
    printf("Dereference *p:      %d   (same as x)\n", *p);

    // Modify x through the pointer
    *p = 99;
    printf("After *p = 99, x is: %d\n\n", x);
}

/* ---------- Exercise 2: Double pointers (pointer-to-pointer) ---------- */
/*
 * Why this matters: In CUDA, you often see:
 *   float *d_A;
 *   cudaMalloc((void**)&d_A, size);
 * This passes a pointer-to-pointer so cudaMalloc can modify d_A itself.
 */
void double_pointers() {
    printf("=== Exercise 2: Double Pointers ===\n");

    int val = 10;
    int *p = &val;
    int **pp = &p;      // pointer to pointer

    printf("val = %d, *p = %d, **pp = %d\n", val, *p, **pp);
    printf("&val = %p, p = %p, *pp = %p\n", (void *)&val, (void *)p, (void *)*pp);

    // Simulating what cudaMalloc does: modifying a pointer through **
    // cudaMalloc((void**)&d_ptr, size) sets d_ptr to point at GPU memory
    int *simulated_device_ptr = NULL;
    int **pp2 = &simulated_device_ptr;
    *pp2 = (int *)malloc(sizeof(int));  // like cudaMalloc setting the pointer
    **pp2 = 777;
    printf("After simulated 'cudaMalloc': *simulated_device_ptr = %d\n\n",
           *simulated_device_ptr);
    free(simulated_device_ptr);
}

/* ---------- Exercise 3: Pointer arithmetic with arrays ---------- */
/*
 * Why this matters: CUDA kernels receive flat 1D pointers for 2D matrices.
 * You index into them as: matrix[row * width + col]
 * This is just pointer arithmetic: *(matrix + row * width + col)
 */
void pointer_arithmetic() {
    printf("=== Exercise 3: Pointer Arithmetic ===\n");

    int arr[] = {10, 20, 30, 40, 50};
    int *p = arr;   // array name decays to pointer to first element

    printf("Array traversal via pointer arithmetic:\n");
    for (int i = 0; i < 5; i++) {
        printf("  *(p + %d) = %d   (at address %p)\n", i, *(p + i), (void *)(p + i));
    }
    // Note: each step advances by sizeof(int) = 4 bytes

    // 2D array as flat 1D — THE pattern used in CUDA matmul
    int rows = 3, cols = 4;
    int *matrix = (int *)malloc(rows * cols * sizeof(int));

    // Fill: matrix[i][j] = i * 10 + j
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            matrix[i * cols + j] = i * 10 + j;

    printf("\n2D matrix stored as flat 1D (row-major):\n");
    for (int i = 0; i < rows; i++) {
        printf("  Row %d: ", i);
        for (int j = 0; j < cols; j++)
            printf("%3d ", matrix[i * cols + j]);
        printf("\n");
    }
    printf("Key: matrix[row][col] = *(matrix + row * cols + col)\n\n");

    free(matrix);
}

/* ---------- Exercise 4: malloc/free — CPU memory management ---------- */
/*
 * Pattern comparison:
 *   CPU:  float *h_A = (float*)malloc(N * sizeof(float));  free(h_A);
 *   GPU:  float *d_A; cudaMalloc(&d_A, N * sizeof(float)); cudaFree(d_A);
 */
void heap_allocation() {
    printf("=== Exercise 4: Heap Allocation (malloc/free) ===\n");

    int N = 1000000;  // 1M elements
    float *data = (float *)malloc(N * sizeof(float));

    if (data == NULL) {
        fprintf(stderr, "malloc failed!\n");
        return;
    }

    // Initialize
    for (int i = 0; i < N; i++)
        data[i] = (float)i * 0.001f;

    // Verify
    printf("Allocated %d floats (%zu bytes)\n", N, N * sizeof(float));
    printf("data[0] = %.3f, data[%d] = %.3f\n", data[0], N - 1, data[N - 1]);

    free(data);
    printf("Memory freed.\n\n");
}

/* ---------- Exercise 5: void pointers and casting ---------- */
/*
 * Why this matters: cudaMalloc takes (void**) because it's type-agnostic.
 * You cast the result to your actual type (float*, int*, etc.)
 */
void void_pointers() {
    printf("=== Exercise 5: Void Pointers ===\n");

    float f = 3.14f;
    int i = 42;

    void *vp;

    vp = &f;
    printf("void* pointing to float: %.2f\n", *(float *)vp);

    vp = &i;
    printf("void* pointing to int:   %d\n", *(int *)vp);

    // Simulating cudaMalloc's signature: cudaError_t cudaMalloc(void **devPtr, size_t size)
    void *generic_ptr = malloc(256);
    float *typed_ptr = (float *)generic_ptr;  // cast to actual type
    typed_ptr[0] = 1.5f;
    typed_ptr[1] = 2.5f;
    printf("After cast: typed_ptr[0]=%.1f, typed_ptr[1]=%.1f\n\n", typed_ptr[0], typed_ptr[1]);
    free(generic_ptr);
}

/* ---------- Exercise 6: Function pointers ---------- */
/*
 * Less directly CUDA-related, but useful for understanding callbacks
 * (e.g., cudaStreamAddCallback).
 */
typedef float (*BinaryOp)(float, float);

float add(float a, float b) { return a + b; }
float mul(float a, float b) { return a * b; }

void function_pointers() {
    printf("=== Exercise 6: Function Pointers ===\n");

    BinaryOp ops[] = {add, mul};
    const char *names[] = {"add", "mul"};

    for (int i = 0; i < 2; i++) {
        printf("  %s(3.0, 4.0) = %.1f\n", names[i], ops[i](3.0f, 4.0f));
    }
    printf("\n");
}

int main() {
    basics();
    double_pointers();
    pointer_arithmetic();
    heap_allocation();
    void_pointers();
    function_pointers();

    printf("=== All pointer exercises complete! ===\n");
    return 0;
}
