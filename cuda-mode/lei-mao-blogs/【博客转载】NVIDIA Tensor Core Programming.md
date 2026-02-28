> 博客来源：https://leimao.github.io/blog/NVIDIA-Tensor-Core-Programming/ ，来自Lei Mao，已获得作者转载授权。

# NVIDIA Tensor Core 编程

## 简介

NVIDIA Tensor Core 是自Volta架构以来NVIDIA GPU上专门用于通用矩阵乘法（GEMM）运算的专用加速器。由于人工智能计算通常由GEMM运算主导，NVIDIA Tensor Core对于加速人工智能应用至关重要。

![NVIDIA Tensor Core GEMM Math](https://files.mdnice.com/user/59/2ea087bb-9da0-4f01-986d-f53ebcfca03a.png)

由于NVIDIA Tensor Core专门为GEMM设计，使用NVIDIA Tensor Core的GEMM吞吐量比使用更适合通用并行编程的NVIDIA CUDA Core能够实现的吞吐量要高得多。

![NVIDIA GEMM Throughput Tuning Tensor Core VS Pascal CUDA Core](https://files.mdnice.com/user/59/f0090b29-08cb-4a00-b27e-495f80b9df24.jpg)

对于NVIDIA Ampere架构，每个SM有4个Tensor Core。特别是，NVIDIA A100 GPU(https://www.nvidia.com/en-us/data-center/a100/) 有108个流多处理器（SM），总共有432个Tensor Core。

![NVIDIA GA100 Full GPU with 128 SMs](https://files.mdnice.com/user/59/0402c26b-f6ab-40d0-bd42-222d6e3d51d7.jpg)

![Each NVIDIA Ampere SM Has 4 Tensor Cores](https://files.mdnice.com/user/59/5cd7136e-574c-40ec-b657-2fa441f0e7a5.png)

NVIDIA Tensor Core是完全可编程的。Warp级别的Tensor Core编程API已在`nvcuda::wmma`命名空间下的`mma.h`头文件中声明。

## NVIDIA Tensor Core 编程

### 矩阵乘法分解

NVIDIA CUDA允许用户在warp级别编程Tensor Core GEMM运算。虽然每个Tensor Core只能执行某些特定小尺寸的矩阵乘法，针对不同的数据类型(https://docs.nvidia.com/cuda/archive/12.0.1/cuda-c-programming-guide/#element-types-and-matrix-sizes)，如我之前的文章"CUDA矩阵乘法"(https://leimao.github.io/blog/CUDA-Matrix-Multiplication/)中所讨论的，大型GEMM可以分解为多个小型GEMM和累加运算。

给定一个GEMM运算 $D = AB + C$，其中 $D \in \mathbb{R}^{m \times n}$, $A \in \mathbb{R}^{m \times k}$, $B \in \mathbb{R}^{k \times n}$, $C \in \mathbb{R}^{m \times n}$，矩阵可以分解为更小的矩阵。

$$A = \begin{bmatrix}
A_{1,1}^{d \times d} & A_{1,2}^{d \times d} & \cdots & A_{1,k/d}^{d \times d} \\
A_{2,1}^{d \times d} & A_{2,2}^{d \times d} & \cdots & A_{2,k/d}^{d \times d} \\
\vdots & \vdots & \ddots & \vdots \\
A_{m/d,1}^{d \times d} & A_{m/d,2}^{d \times d} & \cdots & A_{m/d,k/d}^{d \times d}
\end{bmatrix}$$

$$B = \begin{bmatrix}
B_{1,1}^{d \times d} & B_{1,2}^{d \times d} & \cdots & B_{1,n/d}^{d \times d} \\
B_{2,1}^{d \times d} & B_{2,2}^{d \times d} & \cdots & B_{2,n/d}^{d \times d} \\
\vdots & \vdots & \ddots & \vdots \\
B_{k/d,1}^{d \times d} & B_{k/d,2}^{d \times d} & \cdots & B_{k/d,n/d}^{d \times d}
\end{bmatrix}$$

$$C = \begin{bmatrix}
C_{1,1}^{d \times d} & C_{1,2}^{d \times d} & \cdots & C_{1,n/d}^{d \times d} \\
C_{2,1}^{d \times d} & C_{2,2}^{d \times d} & \cdots & C_{2,n/d}^{d \times d} \\
\vdots & \vdots & \ddots & \vdots \\
C_{m/d,1}^{d \times d} & C_{m/d,2}^{d \times d} & \cdots & C_{m/d,n/d}^{d \times d}
\end{bmatrix}$$

$$D = \begin{bmatrix}
D_{1,1}^{d \times d} & D_{1,2}^{d \times d} & \cdots & D_{1,n/d}^{d \times d} \\
D_{2,1}^{d \times d} & D_{2,2}^{d \times d} & \cdots & D_{2,n/d}^{d \times d} \\
\vdots & \vdots & \ddots & \vdots \\
D_{m/d,1}^{d \times d} & D_{m/d,2}^{d \times d} & \cdots & D_{m/d,n/d}^{d \times d}
\end{bmatrix}$$

$D$中的每个小矩阵通过多个小型GEMM和累加运算计算得出。

$$D_{i_m,i_n}^{d \times d} = \sum_{i_k=1}^{k/d} A_{i_m,i_k}^{d \times d} B_{i_k,i_n}^{d \times d}$$

在我之前的文章"CUDA矩阵乘法"中，我使用CUDA Core和CUDA共享内存来执行上述数学运算，每个线程块计算一个$D_{i_m,i_n}^{d \times d}$。这次我将使用Tensor Core来计算完全相同的数学运算，其中每个warp计算一个$D_{i_m,i_n}^{d \times d}$。更具体地说，每个warp计算一个$16 \times 16 \times 16$的GEMM，得到$D$矩阵中的一个$16 \times 16$瓦片，即$d = 16$。

### 使用NVIDIA Tensor Core的矩阵乘法实现

在这个实现中，我们将使用Tensor Core执行GEMM运算，使用HMMA（半精度矩阵乘法和累加）和IMMA（整数矩阵乘法和累加）指令。此外，已经实现并验证了涉及转置矩阵乘法的四种不同类型的GEMM。

• $D = AB + C$，其中$D \in \mathbb{R}^{m \times n}$, $A \in \mathbb{R}^{m \times k}$, $B \in \mathbb{R}^{k \times n}$, $C \in \mathbb{R}^{m \times n}$

• $D = A^T B + C$，其中$D \in \mathbb{R}^{m \times n}$, $A \in \mathbb{R}^{k \times m}$, $B \in \mathbb{R}^{k \times n}$, $C \in \mathbb{R}^{m \times n}$

• $D = AB^T + C$，其中$D \in \mathbb{R}^{m \times n}$, $A \in \mathbb{R}^{m \times k}$, $B \in \mathbb{R}^{n \times k}$, $C \in \mathbb{R}^{m \times n}$

• $D = A^T B^T + C$，其中$D \in \mathbb{R}^{m \times n}$, $A \in \mathbb{R}^{k \times m}$, $B \in \mathbb{R}^{n \times k}$, $C \in \mathbb{R}^{m \times n}$

在这个实现中，我们将主要关注GEMM运算中的矩阵乘法部分，通过设置$C = 0$。

```c++
#include <cassert>
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <utility>
#include <vector>
// CUDA运行时和MMA库头文件
#include <cuda_runtime.h>
#include <mma.h>

// CUDA错误检查宏定义
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           int const line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// 检查最后一个CUDA错误的宏定义
#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, int const line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// 性能测量函数模板
template <class T>
float measure_performance(std::function<T(cudaStream_t)> bound_function,
                          cudaStream_t stream, int num_repeats = 100,
                          int num_warmups = 100)
{
    cudaEvent_t start, stop;
    float time;

    // 创建CUDA事件用于计时
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // 预热运行
    for (int i{0}; i < num_warmups; ++i)
    {
        bound_function(stream);
    }

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // 开始计时
    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    for (int i{0}; i < num_repeats; ++i)
    {
        bound_function(stream);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_LAST_CUDA_ERROR();
    
    // 计算平均延迟
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    float const latency{time / num_repeats};

    return latency;
}

// 使用WMMA进行GEMM运算的CUDA核函数
// 所有矩阵数据都以列主序存储，与大多数cuBLAS GEMM API一致
// 对于形状为M x N的矩阵A，前导维度是M
// 对于转置的矩阵A（形状为N x M），前导维度是N
// 矩阵A: M x K，或K x N（如果转置）
// 矩阵B: K x M，或M x K（如果转置）
// 矩阵C: M x N
// WMMA_FRAG_LAYOUT_A: 如果A转置则为nvcuda::wmma::row_major，否则为nvcuda::wmma::col_major
// WMMA_FRAG_LAYOUT_B: 如果B转置则为nvcuda::wmma::row_major，否则为nvcuda::wmma::col_major
template <typename T1, typename T2, int WMMA_M, int WMMA_N, int WMMA_K,
          typename WMMA_FRAG_LAYOUT_A, typename WMMA_FRAG_LAYOUT_B>
__global__ void wmma_gemm_a_col_major_b_col_major(
    T1 const* A, T1 const* B, T2* C, uint32_t m, uint32_t n, uint32_t k,
    uint32_t lda, uint32_t ldb, uint32_t ldc, bool is_A_transpose,
    bool is_B_transpose, float alpha, float beta)
{
    // 使用2D网格进行分块
    // 确定warp的2D索引
    uint32_t const warpM{(blockIdx.x * blockDim.x + threadIdx.x) / warpSize};
    uint32_t const warpN{blockIdx.y * blockDim.y + threadIdx.y};

    // 声明WMMA片段
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, T1,
                           WMMA_FRAG_LAYOUT_A>
        a_frag{};
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, T1,
                           WMMA_FRAG_LAYOUT_B>
        b_frag{};
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                           T2>
        acc_frag{};
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                           T2>
        c_frag{};

    // 确保累加器从0开始
    nvcuda::wmma::fill_fragment(acc_frag, static_cast<T2>(0));

    // 在K维度上循环
    for (uint32_t ki{0}; ki < k; ki += WMMA_K)
    {
        // 确定线性内存中MMA矩阵的第一个元素
        // 矩阵A的MMA矩阵
        uint32_t const matrix_mma_a_row_idx{is_A_transpose ? ki
                                                           : warpM * WMMA_M};
        uint32_t const matrix_mma_a_col_idx{is_A_transpose ? warpM * WMMA_M
                                                           : ki};
        // 矩阵B的MMA矩阵
        uint32_t const matrix_mma_b_row_idx{is_B_transpose ? warpN * WMMA_N
                                                           : ki};
        uint32_t const matrix_mma_b_col_idx{is_B_transpose ? ki
                                                           : warpN * WMMA_N};

        // 边界检查
        if (matrix_mma_a_row_idx < (is_A_transpose ? k : m) &&
            matrix_mma_a_col_idx < (is_A_transpose ? m : k) &&
            matrix_mma_b_row_idx < (is_B_transpose ? n : k) &&
            matrix_mma_b_col_idx < (is_B_transpose ? k : n))
        {
            // 确定MMA矩阵第一个元素的内存地址
            // 注意所有矩阵都假设为列主序，因此索引与常见的行主序索引不同
            T1 const* matrix_mma_a_mptr{A + matrix_mma_a_row_idx +
                                        matrix_mma_a_col_idx * lda};
            T1 const* matrix_mma_b_mptr{B + matrix_mma_b_row_idx +
                                        matrix_mma_b_col_idx * ldb};
            // 加载MMA矩阵输入
            nvcuda::wmma::load_matrix_sync(a_frag, matrix_mma_a_mptr, lda);
            nvcuda::wmma::load_matrix_sync(b_frag, matrix_mma_b_mptr, ldb);

            // 执行矩阵乘法
            nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // 加载C的当前值，按beta缩放，并加上按alpha缩放的结果
    uint32_t const matrix_mma_c_row_idx{warpM * WMMA_M};
    uint32_t const matrix_mma_c_col_idx{warpN * WMMA_N};

    if (matrix_mma_c_row_idx < m && matrix_mma_c_col_idx < n)
    {
        T2* matrix_mma_c_mptr{C + matrix_mma_c_row_idx +
                              matrix_mma_c_col_idx * ldc};
        nvcuda::wmma::load_matrix_sync(c_frag, matrix_mma_c_mptr, ldc,
                                       nvcuda::wmma::mem_col_major);
        // 让编译器决定如何进行逐元素操作
        // 这种逐元素操作可以是缩放、累加、量化等
        // https://docs.nvidia.com/cuda/archive/12.0.1/cuda-c-programming-guide/#id40
        // 处理整数类型时要小心
        for (uint32_t i = 0; i < c_frag.num_elements; i++)
        {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }
        // 存储输出
        nvcuda::wmma::store_matrix_sync(matrix_mma_c_mptr, c_frag, ldc,
                                        nvcuda::wmma::mem_col_major);
    }
}

// 启动WMMA矩阵乘法的函数模板
template <typename T1, typename T2>
void launch_wmma_mm(T1 const* A, T1 const* B, T2* C, uint32_t m, uint32_t n,
                    uint32_t k, bool is_A_transpose, bool is_B_transpose,
                    cudaStream_t stream)
{
    // 假设数据中没有填充
    uint32_t const lda{is_A_transpose ? k : m};
    uint32_t const ldb{is_B_transpose ? n : k};
    uint32_t const ldc{m};
    float const alpha{1.0f};
    float const beta{0.0f};

    // WMMA矩阵块大小常量
    constexpr int WMMA_M{16};
    constexpr int WMMA_N{16};
    constexpr int WMMA_K{16};

    constexpr int WARP_SIZE{32};

    dim3 gridDim;
    dim3 blockDim;

    // blockDim.x必须是warpSize的倍数
    // 块大小128x4意味着我们有16个(4x4)warp
    // 每个warp计算一个16x16输出块
    // 一个块计算一个64x64输出块
    // 每个块有4x4个warp，总共4x4x32个线程
    int const num_warps_x = 4;
    int const num_warps_y = 4;
    blockDim.x = num_warps_x * WARP_SIZE;
    blockDim.y = num_warps_y;
    // 向上取整
    gridDim.x = (m + (WMMA_M * num_warps_x - 1)) / (WMMA_M * num_warps_x);
    gridDim.y = (n + WMMA_N * num_warps_y - 1) / (WMMA_N * num_warps_y);

    // C = A * B
    if ((!is_A_transpose) && (!is_B_transpose))
    {
        wmma_gemm_a_col_major_b_col_major<T1, T2, WMMA_M, WMMA_N, WMMA_K,
                                          nvcuda::wmma::col_major,
                                          nvcuda::wmma::col_major>
            <<<gridDim, blockDim, 0, stream>>>(A, B, C, m, n, k, lda, ldb, ldc,
                                               is_A_transpose, is_B_transpose,
                                               alpha, beta);
    }
    // C = A^T * B
    else if ((is_A_transpose) && (!is_B_transpose))
    {
        wmma_gemm_a_col_major_b_col_major<T1, T2, WMMA_M, WMMA_N, WMMA_K,
                                          nvcuda::wmma::row_major,
                                          nvcuda::wmma::col_major>
            <<<gridDim, blockDim, 0, stream>>>(A, B, C, m, n, k, lda, ldb, ldc,
                                               is_A_transpose, is_B_transpose,
                                               alpha, beta);
    }
    // C = A * B^T
    else if ((!is_A_transpose) && (is_B_transpose))
    {
        wmma_gemm_a_col_major_b_col_major<T1, T2, WMMA_M, WMMA_N, WMMA_K,
                                          nvcuda::wmma::col_major,
                                          nvcuda::wmma::row_major>
            <<<gridDim, blockDim, 0, stream>>>(A, B, C, m, n, k, lda, ldb, ldc,
                                               is_A_transpose, is_B_transpose,
                                               alpha, beta);
    }
    // C = A^T * B^T
    else
    {
        wmma_gemm_a_col_major_b_col_major<T1, T2, WMMA_M, WMMA_N, WMMA_K,
                                          nvcuda::wmma::row_major,
                                          nvcuda::wmma::row_major>
            <<<gridDim, blockDim, 0, stream>>>(A, B, C, m, n, k, lda, ldb, ldc,
                                               is_A_transpose, is_B_transpose,
                                               alpha, beta);
    }
    CHECK_LAST_CUDA_ERROR();
}

// CPU参考实现：A和B是列主序矩阵
template <typename T1, typename T2>
void mm_a_col_major_b_col_major(T1 const* A, T1 const* B, T2* C, uint32_t m,
                                uint32_t n, uint32_t k, uint32_t lda,
                                uint32_t ldb, uint32_t ldc, bool is_A_transpose,
                                bool is_B_transpose)
{
    for (uint32_t ni{0}; ni < n; ++ni)
    {
        for (uint32_t mi{0}; mi < m; ++mi)
        {
            // 计算C[mi, ni]
            T2 accum{0};
            // C = A * B
            if ((!is_A_transpose) && (!is_B_transpose))
            {
                for (uint32_t ki{0}; ki < k; ++ki)
                {
                    // A[mi, ki] * B[ki, ni]
                    accum += A[ki * lda + mi] * B[ni * ldb + ki];
                }
            }
            // C = A^T * B
            else if ((is_A_transpose) && (!is_B_transpose))
            {
                for (uint32_t ki{0}; ki < k; ++ki)
                {
                    // A[ki, mi] * B[ki, ni]
                    accum += A[mi * lda + ki] * B[ni * ldb + ki];
                }
            }
            // C = A * B^T
            else if ((!is_A_transpose) && (is_B_transpose))
            {
                for (uint32_t ki{0}; ki < k; ++ki)
                {
                    // A[mi, ki] * B[ni, ki]
                    accum += A[ki * lda + mi] * B[ki * ldb + ni];
                }
            }
            // C = A^T * B^T
            else
            {
                for (uint32_t ki{0}; ki < k; ++ki)
                {
                    // A[ki, mi] * B[ni, ki]
                    accum += A[mi * lda + ki] * B[ki * ldb + ni];
                }
            }
            C[ni * ldc + mi] = accum;
        }
    }
}

// 启动CPU矩阵乘法的函数模板
template <typename T1, typename T2>
void launch_mm(T1 const* A, T1 const* B, T2* C, uint32_t m, uint32_t n,
               uint32_t k, bool is_A_transpose, bool is_B_transpose)
{
    // 假设数据中没有填充
    uint32_t const lda{is_A_transpose ? k : m};
    uint32_t const ldb{is_B_transpose ? n : k};
    uint32_t const ldc{m};
    mm_a_col_major_b_col_major(A, B, C, m, n, k, lda, ldb, ldc, is_A_transpose,
                               is_B_transpose);
}

// 填充随机浮点值
void fill_random_float_values(float* arr, size_t n,
                              std::default_random_engine& e)
{
    std::uniform_real_distribution<float> uniform_dist(-256, 256);
    for (size_t i{0}; i < n; ++i)
    {
        arr[i] = uniform_dist(e);
    }
}

// 填充随机int8值
void fill_random_int8_values(int8_t* arr, size_t n,
                             std::default_random_engine& e)
{
    std::uniform_int_distribution<int8_t> uniform_dist(-128, 127);
    for (size_t i{0}; i < n; ++i)
    {
        arr[i] = uniform_dist(e);
    }
}

// 填充随机int32值
void fill_random_int32_values(int32_t* arr, size_t n,
                              std::default_random_engine& e)
{
    std::uniform_int_distribution<int32_t> uniform_dist(-128, 127);
    for (size_t i{0}; i < n; ++i)
    {
        arr[i] = uniform_dist(e);
    }
}

// 将float数组转换为half数组
void float2half(__half* half_arr, float const* float_arr, size_t n)
{
    for (size_t i{0}; i < n; ++i)
    {
        half_arr[i] = __float2half(float_arr[i]);
    }
}

// 计算两个数组的平均绝对差异比率
template <typename T>
float get_avg_abs_diff_ratio(T const* arr_1, T const* arr_2, size_t n)
{
    float sum_abs_diff_ratio{0};
    for (size_t i{0}; i < n; ++i)
    {
        sum_abs_diff_ratio += std::abs(static_cast<float>(arr_1[i]) -
                                       static_cast<float>(arr_2[i])) /
                              std::abs(static_cast<float>(arr_1[i]) +
                                       static_cast<float>(arr_2[i]));
    }
    return sum_abs_diff_ratio / n;
}

// 检查两个数组是否相等
template <typename T>
bool array_equal(T const* arr_1, T const* arr_2, size_t n)
{
    for (size_t i{0}; i < n; ++i)
    {
        if (arr_1[i] != arr_2[i])
        {
            return false;
        }
    }
    return true;
}

// 打印测试头部信息
void print_test_header(bool is_A_transpose, bool is_B_transpose)
{
    // C = A * B
    if ((!is_A_transpose) && (!is_B_transpose))
    {
        std::cout << "C = A * B" << std::endl;
    }
    // C = A^T * B
    else if ((is_A_transpose) && (!is_B_transpose))
    {
        std::cout << "C = A^T * B" << std::endl;
    }
    // C = A * B^T
    else if ((!is_A_transpose) && (is_B_transpose))
    {
        std::cout << "C = A * B^T" << std::endl;
    }
    // C = A^T * B^T
    else
    {
        std::cout << "C = A^T * B^T" << std::endl;
    }
}

int main()
{
    // 性能测试参数
    constexpr int num_repeats{10};
    constexpr int num_warmups{10};

    // 矩阵大小定义
    uint32_t const matrix_size_m{1024};
    uint32_t const matrix_size_n{1024};
    uint32_t const matrix_size_k{1024};
    std::cout << "Matrix Sizes" << std::endl;
    std::cout << "M: " << matrix_size_m << std::endl;
    std::cout << "N: " << matrix_size_n << std::endl;
    std::cout << "K: " << matrix_size_k << std::endl;

    // 随机数生成器
    std::default_random_engine random_engine(0);

    // 创建CUDA流
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    // HMMA测试（半精度矩阵乘法和累加）
    std::cout << "FP16 HMMA" << std::endl;
    std::vector<float> matrix_a_float(matrix_size_m * matrix_size_k);
    std::vector<float> matrix_b_float(matrix_size_k * matrix_size_n);
    std::vector<__half> matrix_a_half(matrix_size_m * matrix_size_k);
    std::vector<__half> matrix_b_half(matrix_size_k * matrix_size_n);
    std::vector<float> matrix_c_float(matrix_size_m * matrix_size_n);
    std::vector<float> matrix_c_float_reference(matrix_size_m * matrix_size_n);

    // 获取主机内存指针
    float* h_matrix_a_float{matrix_a_float.data()};
    float* h_matrix_b_float{matrix_b_float.data()};
    __half* h_matrix_a_half{matrix_a_half.data()};
    __half* h_matrix_b_half{matrix_b_half.data()};
    float* h_matrix_c_float{matrix_c_float.data()};
    float* h_matrix_c_float_reference{matrix_c_float_reference.data()};

    // 填充随机数据
    fill_random_float_values(h_matrix_a_float, matrix_a_float.size(),
                             random_engine);
    fill_random_float_values(h_matrix_b_float, matrix_b_float.size(),
                             random_engine);
    fill_random_float_values(h_matrix_c_float, matrix_c_float.size(),
                             random_engine);
    fill_random_float_values(h_matrix_c_float_reference,
                             matrix_c_float_reference.size(), random_engine);
    // 转换为半精度
    float2half(h_matrix_a_half, h_matrix_a_float, matrix_a_float.size());
    float2half(h_matrix_b_half, h_matrix_b_float, matrix_b_float.size());

    // 分配设备内存
    half *d_matrix_a_half, *d_matrix_b_half;
    float* d_matrix_c_float;

    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_a_half,
                                matrix_size_m * matrix_size_k * sizeof(half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_b_half,
                                matrix_size_k * matrix_size_n * sizeof(half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_c_float,
                                matrix_size_m * matrix_size_n * sizeof(float)));

    // 将数据从主机复制到设备
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix_a_half, h_matrix_a_half,
                                matrix_a_float.size() * sizeof(__half),
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix_b_half, h_matrix_b_half,
                                matrix_b_float.size() * sizeof(__half),
                                cudaMemcpyHostToDevice));

    // 测试所有转置组合
    for (bool is_A_transpose : {true, false})
    {
        for (bool is_B_transpose : {true, false})
        {
            print_test_header(is_A_transpose, is_B_transpose);
            // 使用CPU计算参考输出
            launch_mm(h_matrix_a_float, h_matrix_b_float,
                      h_matrix_c_float_reference, matrix_size_m, matrix_size_n,
                      matrix_size_k, is_A_transpose, is_B_transpose);
            // 使用CUDA WMMA计算输出
            launch_wmma_mm(d_matrix_a_half, d_matrix_b_half, d_matrix_c_float,
                           matrix_size_m, matrix_size_n, matrix_size_k,
                           is_A_transpose, is_B_transpose, stream);
            CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

            // 将结果从设备复制到主机
            CHECK_CUDA_ERROR(cudaMemcpy(h_matrix_c_float, d_matrix_c_float,
                                        matrix_c_float.size() * sizeof(float),
                                        cudaMemcpyDeviceToHost));

            // 验证结果准确性
            float const avg_abs_diff_ratio{get_avg_abs_diff_ratio(
                h_matrix_c_float, h_matrix_c_float_reference,
                matrix_c_float.size())};
            if (avg_abs_diff_ratio > 0.01)
            {
                std::cout << "Got high average absolute diff ratio: "
                          << avg_abs_diff_ratio << std::endl;
            }

            // 性能测量
            std::function<void(cudaStream_t)> const function_hmma{std::bind(
                launch_wmma_mm<__half, float>, d_matrix_a_half, d_matrix_b_half,
                d_matrix_c_float, matrix_size_m, matrix_size_n, matrix_size_k,
                is_A_transpose, is_B_transpose, std::placeholders::_1)};
            float const latency_hmma{measure_performance(
                function_hmma, stream, num_repeats, num_warmups)};
            std::cout << std::fixed << std::setprecision(3)
                      << "HMMA Latency: " << latency_hmma << " ms" << std::endl;
        }
    }

    // 释放HMMA测试的设备内存
    CHECK_CUDA_ERROR(cudaFree(d_matrix_a_half));
    CHECK_CUDA_ERROR(cudaFree(d_matrix_b_half));
    CHECK_CUDA_ERROR(cudaFree(d_matrix_c_float));

    // IMMA测试（整数矩阵乘法和累加）
    std::cout << "INT8 IMMA" << std::endl;
    std::vector<int8_t> matrix_a_int8(matrix_size_m * matrix_size_k);
    std::vector<int8_t> matrix_b_int8(matrix_size_k * matrix_size_n);
    std::vector<int32_t> matrix_c_int32(matrix_size_m * matrix_size_n);
    std::vector<int32_t> matrix_c_int32_reference(matrix_size_m *
                                                  matrix_size_n);

    // 获取主机内存指针
    int8_t* h_matrix_a_int8{matrix_a_int8.data()};
    int8_t* h_matrix_b_int8{matrix_b_int8.data()};
    int32_t* h_matrix_c_int32{matrix_c_int32.data()};
    int32_t* h_matrix_c_int32_reference{matrix_c_int32_reference.data()};

    // 填充随机整数数据
    fill_random_int8_values(h_matrix_a_int8, matrix_a_int8.size(),
                            random_engine);
    fill_random_int8_values(h_matrix_b_int8, matrix_b_int8.size(),
                            random_engine);
    fill_random_int32_values(h_matrix_c_int32, matrix_c_int32.size(),
                             random_engine);
    fill_random_int32_values(h_matrix_c_int32_reference,
                             matrix_c_int32_reference.size(), random_engine);

    // 分配设备内存用于INT8 IMMA测试
    int8_t *d_matrix_a_int8, *d_matrix_b_int8;
    int32_t* d_matrix_c_int32;

    CHECK_CUDA_ERROR(cudaMalloc(
        &d_matrix_a_int8, matrix_size_m * matrix_size_k * sizeof(int8_t)));
    CHECK_CUDA_ERROR(cudaMalloc(
        &d_matrix_b_int8, matrix_size_k * matrix_size_n * sizeof(int8_t)));
    CHECK_CUDA_ERROR(cudaMalloc(
        &d_matrix_c_int32, matrix_size_m * matrix_size_n * sizeof(int32_t)));

    // 将数据从主机复制到设备
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix_a_int8, h_matrix_a_int8,
                                matrix_a_int8.size() * sizeof(int8_t),
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix_b_int8, h_matrix_b_int8,
                                matrix_b_int8.size() * sizeof(int8_t),
                                cudaMemcpyHostToDevice));

    // 测试所有转置组合
    for (bool is_A_transpose : {true, false})
    {
        for (bool is_B_transpose : {true, false})
        {
            print_test_header(is_A_transpose, is_B_transpose);
            // 使用CPU计算参考输出
            launch_mm(h_matrix_a_int8, h_matrix_b_int8,
                      h_matrix_c_int32_reference, matrix_size_m, matrix_size_n,
                      matrix_size_k, is_A_transpose, is_B_transpose);
            // 使用CUDA WMMA计算输出
            launch_wmma_mm(d_matrix_a_int8, d_matrix_b_int8, d_matrix_c_int32,
                           matrix_size_m, matrix_size_n, matrix_size_k,
                           is_A_transpose, is_B_transpose, stream);
            CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
            
            // 将结果从设备复制到主机
            CHECK_CUDA_ERROR(cudaMemcpy(h_matrix_c_int32, d_matrix_c_int32,
                                        matrix_c_int32.size() * sizeof(int32_t),
                                        cudaMemcpyDeviceToHost));
            // 整数矩阵乘法的CPU和CUDA结果应该按位相同
            assert(array_equal(h_matrix_c_int32, h_matrix_c_int32_reference,
                               matrix_c_int32.size()));

            // 性能测量
            std::function<void(cudaStream_t)> const function_imma{
                std::bind(launch_wmma_mm<int8_t, int32_t>, d_matrix_a_int8,
                          d_matrix_b_int8, d_matrix_c_int32, matrix_size_m,
                          matrix_size_n, matrix_size_k, is_A_transpose,
                          is_B_transpose, std::placeholders::_1)};
            float const latency_imma{measure_performance(
                function_imma, stream, num_repeats, num_warmups)};
            std::cout << std::fixed << std::setprecision(3)
                      << "IMMA Latency: " << latency_imma << " ms" << std::endl;
        }
    }

    // 释放IMMA测试的设备内存
    CHECK_CUDA_ERROR(cudaFree(d_matrix_a_int8));
    CHECK_CUDA_ERROR(cudaFree(d_matrix_b_int8));
    CHECK_CUDA_ERROR(cudaFree(d_matrix_c_int32));

    // 销毁CUDA流
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}
```

所有转置矩阵乘法实现实际上并没有转置矩阵。相反，我们使用了在我之前的文章"行主序 VS 列主序"(https://leimao.github.io/blog/Row-Major-VS-Column-Major/)中介绍的行主序和列主序技巧。

我们还观察到，对于以列主序存储的矩阵的矩阵乘法，$C = A^T B$是最快的，而$C = AB^T$是最慢的，这是在NVIDIA RTX 3090 GPU上使用HMMA和IMMA指令的GEMM实现。

```shell
$ nvcc mma.cu -o mma --gpu-architecture=compute_86
$ ./mma
Matrix Sizes
M: 1024
N: 1024
K: 1024
FP16 HMMA
C = A^T * B^T
HMMA Latency: 0.177 ms
C = A^T * B
HMMA Latency: 0.169 ms
C = A * B^T
HMMA Latency: 0.189 ms
C = A * B
HMMA Latency: 0.177 ms
INT8 IMMA
C = A^T * B^T
IMMA Latency: 0.129 ms
C = A^T * B
IMMA Latency: 0.090 ms
C = A * B^T
IMMA Latency: 0.170 ms
C = A * B
IMMA Latency: 0.129 ms
```

## 结论

NVIDIA Tensor Core是可编程的，可用于加速由GEMM运算主导的计算。

## 参考文献

- Programming Tensor Cores in CUDA 9(https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/)
- CUDA C Programming Guide - Warp Matrix Functions(https://docs.nvidia.com/cuda/archive/12.0.1/cuda-c-programming-guide/#warp-matrix-functions)
- CUDA Matrix Multiplication(https://leimao.github.io/blog/CUDA-Matrix-Multiplication/)
- Row-Major VS Column-Major(https://leimao.github.io/blog/Row-Major-VS-Column-Major/)



