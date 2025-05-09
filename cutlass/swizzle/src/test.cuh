#pragma once
#include <mma.h>
#include "utils.cuh"

#define CHECK_TEST(test)                                                       \
    if (!test) {                                                               \
        std::cout << "\033[1;31m" << #test << " failed\033[0m\n";              \
        return 1;                                                              \
    } else {                                                                   \
        std::cout << "\033[1;32m" << #test << " passed\033[0m\n";              \
    }

/**
 * \brief 16x16 Matrix Multiplication of C = A * B^T
 */
void matmul_16x16(half *c, half *a, half *b, int stride) {
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            half sum = 0.0f;
            for (int k = 0; k < 16; k++) {
                sum += a[i * stride + k] * b[j * stride + k];
            }
            c[i * stride + j] = sum;
        }
    }
}

/**
 * \brief Test 16x16 Matrix Multiplication C = A * B^T
 */
bool test_mma_16x16(void (*kernel)(half *, half *, half *)) {
    const int n = 16 * 16;
    half *a_h = (half *)malloc(n * sizeof(half));
    half *b_h = (half *)malloc(n * sizeof(half));
    half *c_h = (half *)malloc(n * sizeof(half));

    fill_data(a_h, n);
    fill_data(b_h, n);

    half *a_d, *b_d, *c_d;
    cudaMalloc(&a_d, n * sizeof(half));
    cudaMalloc(&b_d, n * sizeof(half));
    cudaMalloc(&c_d, n * sizeof(half));

    cudaMemcpy(a_d, a_h, n * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, n * sizeof(half), cudaMemcpyHostToDevice);

    kernel<<<1, 32>>>(c_d, a_d, b_d);

    cudaMemcpy(c_h, c_d, n * sizeof(half), cudaMemcpyDeviceToHost);

    half *c_ref = (half *)malloc(n * sizeof(half));
    matmul_16x16(c_ref, a_h, b_h, 16);

    bool ret = !diff(c_h, c_ref, n);
    free(a_h);
    free(b_h);
    free(c_h);
    free(c_ref);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    return ret;
}

/**
 * \brief 4 blocks of 16x16 Matrix Multiplication of C = A * B^T, each block C =
 * block(A) * block(B)^T
 * \note this test launches 4 warps in a block, each warp computes a 16x16 mma
 */
bool test_mma_16x16_x4(void (*kernel)(half *, half *, half *)) {
    const int n = 16 * 16 * 4;
    half *a_h = (half *)malloc(n * sizeof(half));
    half *b_h = (half *)malloc(n * sizeof(half));
    half *c_h = (half *)malloc(n * sizeof(half));

    fill_data(a_h, n);
    fill_data(b_h, n);

    half *a_d, *b_d, *c_d;
    cudaMalloc(&a_d, n * sizeof(half));
    cudaMalloc(&b_d, n * sizeof(half));
    cudaMalloc(&c_d, n * sizeof(half));

    cudaMemcpy(a_d, a_h, n * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, n * sizeof(half), cudaMemcpyHostToDevice);

    kernel<<<1, dim3(32, 4)>>>(c_d, a_d, b_d);

    cudaDeviceSynchronize();
    cudaMemcpy(c_h, c_d, n * sizeof(half), cudaMemcpyDeviceToHost);

    half *c_ref = (half *)malloc(n * sizeof(half));
#pragma omp parallel for
    for (int i = 0; i < 4; i++) {
        matmul_16x16(c_ref + i * 16, a_h + i * 16, b_h + i * 16, 16 * 4);
    }

    bool ret = !diff(c_h, c_ref, n);
    free(a_h);
    free(b_h);
    free(c_h);
    free(c_ref);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    return ret;
}

/**
 * \brief Test 16x256 2 phase operations:
 *
 * Phase 1: for each 1x256 row, regard it as a 16x16 matrix, compute C = A * B^T
 *
 * Phase 2: tile 16x256 into 16 blocks of 16x16, compute C = A * B^T on each
 * block
 *
 * \note this test aim to achieve complete bank conflict free by applying 2
 * layer of swizzle
 */
bool test_mma_multi_pattern(void (*kernel)(half *, half *, half *)) {
    const int n = 16 * 256;
    half *a_h = (half *)malloc(n * sizeof(half));
    half *b_h = (half *)malloc(n * sizeof(half));
    half *c_h = (half *)malloc(n * sizeof(half));

    fill_data(a_h, n);
    fill_data(b_h, n);

    half *a_d, *b_d, *c_d;
    cudaMalloc(&a_d, n * sizeof(half));
    cudaMalloc(&b_d, n * sizeof(half));
    cudaMalloc(&c_d, n * sizeof(half));

    cudaMemcpy(a_d, a_h, n * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, n * sizeof(half), cudaMemcpyHostToDevice);

    kernel<<<1, dim3(32, 16)>>>(c_d, a_d, b_d);

    cudaDeviceSynchronize();
    cudaMemcpy(c_h, c_d, n * sizeof(half), cudaMemcpyDeviceToHost);
    // compute reference result
    half *c_ref = (half *)malloc(n * sizeof(half));
#pragma omp parallel for
    for (int i = 0; i < 16; i++) {
        matmul_16x16(c_ref + i * 256, a_h + i * 256, b_h + i * 256, 16);
    }
#pragma omp parallel for
    for (int i = 0; i < 16; i++) {
        matmul_16x16(c_ref + i * 16, a_h + i * 16, b_h + i * 16, 256);
    }

    // compare results
    bool ret = !diff(c_h, c_ref, n, 1.5e-2);

    free(a_h);
    free(b_h);
    free(c_h);
    free(c_ref);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    return ret;
}
