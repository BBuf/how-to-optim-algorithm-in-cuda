> 博客来源：https://leimao.github.io/blog/CUDA-Coalesced-Memory-Access/ ，来自Lei Mao，已获得作者转载授权。

# CUDA 合并内存访问

## 简介

在CUDA编程中，从CUDA kernel访问GPU全局内存通常是影响CUDA kernel性能的一个因素。为了减少全局内存IO，我们希望通过合并全局内存访问来减少全局内存访问次数，并将可重用的数据缓存在快速的共享内存中。

在这篇博客文章中，我想讨论如何合并GPU全局内存的读写访问，并使用一个例子来展示通过合并全局内存读写访问带来的性能提升。

## CUDA 矩阵转置

### 实现

在下面的例子中，我实现了三个用于（异地）矩阵转置的CUDA kernel。

- 全局内存读访问是合并的，而全局内存写访问是非合并的。
- 全局内存写访问是合并的，而全局内存读访问是非合并的。
- 全局内存读写访问都是合并的。这是通过使用共享内存实现的。

```c++
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <cuda_runtime.h>
// CUDA错误检查宏定义，用于检查CUDA API调用的返回值
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, char const* func, char const* file, int line)
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
#define CHECK_LAST_CUDA_ERROR() check_last(__FILE__, __LINE__)
void check_last(char const* file, int line)
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

// 性能测量函数模板，用于测量CUDA kernel的执行时间
template <class T>
float measure_performance(std::function<T(cudaStream_t)> bound_function,
                          cudaStream_t stream, size_t num_repeats = 100,
                          size_t num_warmups = 100)
{
    cudaEvent_t start, stop;
    float time;

    // 创建CUDA事件用于计时
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // 预热阶段，避免首次执行的开销影响测量结果
    for (size_t i{0}; i < num_warmups; ++i)
    {
        bound_function(stream);
    }

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // 开始计时并执行多次测量
    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    for (size_t i{0}; i < num_repeats; ++i)
    {
        bound_function(stream);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    // 计算平均延迟
    float const latency{time / num_repeats};

    return latency;
}

// 向上取整除法的辅助函数
constexpr size_t div_up(size_t a, size_t b) { return (a + b - 1) / b; }

// 矩阵转置kernel：全局内存读取合并，写入非合并
// 线程按行主序访问输入矩阵（合并读取），但按列主序写入输出矩阵（非合并写入）
template <typename T>
__global__ void transpose_read_coalesced(T* output_matrix,
                                         T const* input_matrix, size_t M,
                                         size_t N)
{
    // 计算当前线程对应的矩阵坐标
    size_t const j{threadIdx.x + blockIdx.x * blockDim.x};  // 列索引
    size_t const i{threadIdx.y + blockIdx.y * blockDim.y};  // 行索引
    size_t const from_idx{i * N + j};  // 输入矩阵的线性索引（行主序）
    if ((i < M) && (j < N))
    {
        size_t const to_idx{j * M + i};  // 输出矩阵的线性索引（转置后的位置）
        output_matrix[to_idx] = input_matrix[from_idx];
    }
}

// 矩阵转置kernel：全局内存写入合并，读取非合并
// 线程按列主序访问输入矩阵（非合并读取），但按行主序写入输出矩阵（合并写入）
template <typename T>
__global__ void transpose_write_coalesced(T* output_matrix,
                                          T const* input_matrix, size_t M,
                                          size_t N)
{
    // 注意：这里的坐标映射与read_coalesced不同
    size_t const j{threadIdx.x + blockIdx.x * blockDim.x};  // 输出矩阵的列索引
    size_t const i{threadIdx.y + blockIdx.y * blockDim.y};  // 输出矩阵的行索引
    size_t const to_idx{i * M + j};  // 输出矩阵的线性索引（行主序，合并写入）
    if ((i < N) && (j < M))
    {
        size_t const from_idx{j * N + i};  // 输入矩阵的线性索引（非合并读取）
        output_matrix[to_idx] = input_matrix[from_idx];
    }
}

// 启动读取合并的矩阵转置kernel
template <typename T>
void launch_transpose_read_coalesced(T* output_matrix, T const* input_matrix,
                                     size_t M, size_t N, cudaStream_t stream)
{
    constexpr size_t const warp_size{32};
    // 使用32x32的线程块，与warp大小匹配以优化内存访问
    dim3 const threads_per_block{warp_size, warp_size};
    dim3 const blocks_per_grid{static_cast<unsigned int>(div_up(N, warp_size)),
                               static_cast<unsigned int>(div_up(M, warp_size))};
    transpose_read_coalesced<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        output_matrix, input_matrix, M, N);
    CHECK_LAST_CUDA_ERROR();
}

// 启动写入合并的矩阵转置kernel
template <typename T>
void launch_transpose_write_coalesced(T* output_matrix, T const* input_matrix,
                                      size_t M, size_t N, cudaStream_t stream)
{
    constexpr size_t const warp_size{32};
    dim3 const threads_per_block{warp_size, warp_size};
    // 注意：网格维度与read_coalesced不同，因为坐标映射不同
    dim3 const blocks_per_grid{static_cast<unsigned int>(div_up(M, warp_size)),
                               static_cast<unsigned int>(div_up(N, warp_size))};
    transpose_write_coalesced<<<blocks_per_grid, threads_per_block, 0,
                                stream>>>(output_matrix, input_matrix, M, N);
    CHECK_LAST_CUDA_ERROR();
}

// 矩阵转置kernel：使用共享内存实现读写都合并
// 通过共享内存作为中间缓冲区，实现全局内存的合并读取和合并写入
template <typename T, size_t BLOCK_SIZE = 32>
__global__ void transpose_read_write_coalesced(T* output_matrix,
                                               T const* input_matrix, size_t M,
                                               size_t N)
{
    // 使用BLOCK_SIZE + 1避免共享内存bank冲突
    // https://leimao.github.io/blog/CUDA-Shared-Memory-Bank/
    // 可以尝试将其设置为BLOCK_SIZE而不是BLOCK_SIZE + 1来观察性能下降
    __shared__ T buffer[BLOCK_SIZE][BLOCK_SIZE + 1];

    // 计算输入矩阵的坐标（用于合并读取）
    size_t const matrix_j{threadIdx.x + blockIdx.x * blockDim.x};
    size_t const matrix_i{threadIdx.y + blockIdx.y * blockDim.y};
    size_t const matrix_from_idx{matrix_i * N + matrix_j};

    // 有两种方式将矩阵数据写入共享内存：
    // 1. 将转置后的矩阵数据从DRAM写入共享内存，然后将非转置的矩阵数据从共享内存写入DRAM
    // 2. 将非转置的矩阵数据从DRAM写入共享内存，然后将转置后的矩阵数据从共享内存写入DRAM
    // 两种方法应该产生相同的性能，即使存在共享内存访问bank冲突

    if ((matrix_i < M) && (matrix_j < N))
    {
        // 第一种方法：将数据以转置形式存储在共享内存中
        buffer[threadIdx.x][threadIdx.y] = input_matrix[matrix_from_idx];
        // 第二种方法：将数据以原始形式存储在共享内存中
        // buffer[threadIdx.y][threadIdx.x] = input_matrix[matrix_from_idx];
    }

    // 确保块内的缓冲区已填充完毕
    __syncthreads();

    // 计算输出矩阵的坐标（用于合并写入）
    size_t const matrix_transposed_j{threadIdx.x + blockIdx.y * blockDim.y};
    size_t const matrix_transposed_i{threadIdx.y + blockIdx.x * blockDim.x};

    if ((matrix_transposed_i < N) && (matrix_transposed_j < M))
    {
        size_t const to_idx{matrix_transposed_i * M + matrix_transposed_j};
        // 第一种方法：从转置存储的共享内存中读取
        output_matrix[to_idx] = buffer[threadIdx.y][threadIdx.x];
        // 第二种方法：从原始存储的共享内存中读取
        // output_matrix[to_idx] = buffer[threadIdx.x][threadIdx.y];
    }
}

// 启动读写都合并的矩阵转置kernel
template <typename T>
void launch_transpose_read_write_coalesced(T* output_matrix,
                                           T const* input_matrix, size_t M,
                                           size_t N, cudaStream_t stream)
{
    constexpr size_t const warp_size{32};
    dim3 const threads_per_block{warp_size, warp_size};
    dim3 const blocks_per_grid{static_cast<unsigned int>(div_up(N, warp_size)),
                               static_cast<unsigned int>(div_up(M, warp_size))};
    transpose_read_write_coalesced<T, warp_size>
        <<<blocks_per_grid, threads_per_block, 0, stream>>>(output_matrix,
                                                            input_matrix, M, N);
    CHECK_LAST_CUDA_ERROR();
}

// 比较两个数组是否相等的辅助函数
template <typename T>
bool is_equal(T const* data_1, T const* data_2, size_t size)
{
    for (size_t i{0}; i < size; ++i)
    {
        if (data_1[i] != data_2[i])
        {
            return false;
        }
    }
    return true;
}

// 验证矩阵转置实现正确性的函数
template <typename T>
bool verify_transpose_implementation(
    std::function<void(T*, T const*, size_t, size_t, cudaStream_t)>
        transpose_function,
    size_t M, size_t N)
{
    // 固定随机种子以确保可重现性
    std::mt19937 gen{0};
    cudaStream_t stream;
    size_t const matrix_size{M * N};
    std::vector<T> matrix(matrix_size, 0.0f);
    std::vector<T> matrix_transposed(matrix_size, 1.0f);
    std::vector<T> matrix_transposed_reference(matrix_size, 2.0f);
    std::uniform_real_distribution<T> uniform_dist(-256, 256);
    
    // 生成随机输入矩阵
    for (size_t i{0}; i < matrix_size; ++i)
    {
        matrix[i] = uniform_dist(gen);
    }
    
    // 使用CPU创建参考转置矩阵
    for (size_t i{0}; i < M; ++i)
    {
        for (size_t j{0}; j < N; ++j)
        {
            size_t const from_idx{i * N + j};
            size_t const to_idx{j * M + i};
            matrix_transposed_reference[to_idx] = matrix[from_idx];
        }
    }
    
    // 分配GPU内存并执行转置操作
    T* d_matrix;
    T* d_matrix_transposed;
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix, matrix_size * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_transposed, matrix_size * sizeof(T)));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix, matrix.data(),
                                matrix_size * sizeof(T),
                                cudaMemcpyHostToDevice));
    transpose_function(d_matrix_transposed, d_matrix, M, N, stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_CUDA_ERROR(cudaMemcpy(matrix_transposed.data(), d_matrix_transposed,
                                matrix_size * sizeof(T),
                                cudaMemcpyDeviceToHost));
    
    // 验证结果正确性
    bool const correctness{is_equal(matrix_transposed.data(),
                                    matrix_transposed_reference.data(),
                                    matrix_size)};
    
    // 清理资源
    CHECK_CUDA_ERROR(cudaFree(d_matrix));
    CHECK_CUDA_ERROR(cudaFree(d_matrix_transposed));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    return correctness;
}

// 性能测试函数
template <typename T>
void profile_transpose_implementation(
    std::function<void(T*, T const*, size_t, size_t, cudaStream_t)>
        transpose_function,
    size_t M, size_t N)
{
    constexpr int num_repeats{100};   // 重复执行次数
    constexpr int num_warmups{10};    // 预热次数
    cudaStream_t stream;
    size_t const matrix_size{M * N};
    
    // 分配GPU内存
    T* d_matrix;
    T* d_matrix_transposed;
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix, matrix_size * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_transposed, matrix_size * sizeof(T)));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    // 包装转置函数以便于性能测量
    std::function<void(cudaStream_t)> const transpose_function_wrapped{
        std::bind(transpose_function, d_matrix_transposed, d_matrix, M, N,
                  std::placeholders::_1)};
    
    // 测量性能并输出结果
    float const transpose_function_latency{measure_performance(
        transpose_function_wrapped, stream, num_repeats, num_warmups)};
    std::cout << std::fixed << std::setprecision(3)
              << "Latency: " << transpose_function_latency << " ms"
              << std::endl;
    
    // 清理资源
    CHECK_CUDA_ERROR(cudaFree(d_matrix));
    CHECK_CUDA_ERROR(cudaFree(d_matrix_transposed));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}

int main()
{
    // 单元测试：验证所有实现的正确性
    for (size_t m{1}; m <= 64; ++m)
    {
        for (size_t n{1}; n <= 64; ++n)
        {
            assert(verify_transpose_implementation<float>(
                &launch_transpose_write_coalesced<float>, m, n));
            assert(verify_transpose_implementation<float>(
                &launch_transpose_read_coalesced<float>, m, n));
            assert(verify_transpose_implementation<float>(
                &launch_transpose_read_write_coalesced<float>, m, n));
        }
    }

    // 性能测试
    // M: 行数
    size_t const M{12800};
    // N: 列数
    size_t const N{12800};
    std::cout << M << " x " << N << " Matrix" << std::endl;
    
    std::cout << "Transpose Write Coalesced" << std::endl;
    profile_transpose_implementation<float>(
        &launch_transpose_write_coalesced<float>, M, N);
        
    std::cout << "Transpose Read Coalesced" << std::endl;
    profile_transpose_implementation<float>(
        &launch_transpose_read_coalesced<float>, M, N);
        
    std::cout << "Transpose Read and Write Coalesced" << std::endl;
    profile_transpose_implementation<float>(
        &launch_transpose_read_write_coalesced<float>, M, N);
}

```

### 性能表现

使用 `12800 x 12800` 矩阵测量了三个CUDA kernel的性能。我们使用 `12800 x 12800` 方阵进行性能测量的原因是，我们希望尽可能公平地比较全局内存合并读取和合并写入的性能。

使用 `-Xptxas -O0`，我们可以禁用CUDA kernel的所有NVCC编译器优化。我们可以看到，具有全局内存合并写入的 kernel比具有全局内存合并读取的 kernel快得多，至少在这个使用案例中是这样。通过在 kernel中同时启用全局内存合并读取和写入，该 kernel的性能在所有三个 kernel中是最好的。

```shell
$ nvcc transpose.cu -o transpose -Xptxas -O0
$ ./transpose
12800 x 12800 Matrix
Transpose Write Coalesced
Latency: 5.220 ms
Transpose Read Coalesced
Latency: 7.624 ms
Transpose Read and Write Coalesced
Latency: 4.804 ms
```

使用 `-Xptxas -O3`（这是编译器默认选项），我们可以启用CUDA kernel的所有NVCC编译器优化。在这种情况下，三个 kernel的CUDA kernel性能顺序保持不变。

```shell
$ nvcc transpose.cu -o transpose -Xptxas -O3
$ ./transpose
12800 x 12800 Matrix
Transpose Write Coalesced
Latency: 2.924 ms
Transpose Read Coalesced
Latency: 5.337 ms
Transpose Read and Write Coalesced
Latency: 2.345 ms
```

所有测量都是在具有Intel i9-9900K CPU和NVIDIA RTX 3090 GPU的平台上进行的。

## 结论

在CUDA kernel实现中，我们应该尽可能地尝试合并全局内存的读取和写入操作。

