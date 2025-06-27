> 博客来源：https://leimao.github.io/blog/CUDA-Reduction/ ，来自Lei Mao，已获得作者转载授权。后续会转载一些Lei Mao的CUDA相关Blog，也是一个完整的专栏，Blog会从稍早一些的CUDA架构到当前最新的CUDA架构，也会包含实用工程技巧，底层指令分析，Cutlass分析等等多个课题，是一个时间线十分明确的专栏。

# CUDA 归约操作

## 简介

归约操作是并行计算中的常见操作。通常归约操作用于计算一系列元素的和、最大值、最小值或乘积。

在这篇博客文章中，我们将讨论并行归约算法及其在CUDA中的实现。

## 批量归约求和

在这个例子中，我们在CUDA中实现了两个批量归约求和kernel 。批量归约求和kernel 计算一批数组中每个数组元素的和。

归约算法的思想很简单。对于批次中的每个数组，我们将分配一个由固定数量线程组成的线程块来计算数组中元素的和。每个线程将从全局内存访问数组中的多个元素，并将部分和存储在寄存器文件中。在所有线程计算完部分和后，我们有两种方法将部分和进一步归约为最终和。一种方法是使用共享内存存储部分和，并在共享内存中归约部分和。另一种方法是使用warp级原语(developer.nvidia.com/blog/using-cuda-warp-level-primitives/)在warp中的寄存器文件中归约部分和，然后在共享内存中进行较小规模的归约。

```c++
#include <cassert>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

// CUDA错误检查宏定义
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

// 检查最后一个CUDA错误
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

// 性能测量模板函数
template <class T>
float measure_performance(std::function<T(cudaStream_t)> bound_function,
                          cudaStream_t stream, size_t num_repeats = 10,
                          size_t num_warmups = 10)
{
    cudaEvent_t start, stop;
    float time;

    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // 预热阶段
    for (size_t i{0}; i < num_warmups; ++i)
    {
        bound_function(stream);
    }

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // 正式测量阶段
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

    float const latency{time / num_repeats};

    return latency;
}

// 字符串居中对齐函数
std::string std_string_centered(std::string const& s, size_t width,
                                char pad = ' ')
{
    size_t const l{s.length()};
    // 如果宽度太小则抛出异常
    if (width < l)
    {
        throw std::runtime_error("Width is too small.");
    }
    size_t const left_pad{(width - l) / 2};
    size_t const right_pad{width - l - left_pad};
    std::string const s_centered{std::string(left_pad, pad) + s +
                                 std::string(right_pad, pad)};
    return s_centered;
}

// 共享内存归约求和版本1 - 使用线程间同步
template <size_t NUM_THREADS>
__device__ float shared_data_reduce_sum_v1(float shared_data[NUM_THREADS])
{
    static_assert(NUM_THREADS % 32 == 0,
                  "NUM_THREADS must be a multiple of 32");
    size_t const thread_idx{threadIdx.x};
    
    // 使用树状归约模式，每次步长减半
#pragma unroll
    for (size_t stride{NUM_THREADS / 2}; stride > 0; stride /= 2)
    {
        if (thread_idx < stride)
        {
            // 每个线程将自己的值与stride距离的线程值相加
            shared_data[thread_idx] += shared_data[thread_idx + stride];
        }
        // 同步所有线程，确保上一轮计算完成
        __syncthreads();
    }
    return shared_data[0];
}

// 共享内存归约求和版本2 - 用于warp级归约后的最终归约
template <size_t NUM_WARPS>
__device__ float shared_data_reduce_sum_v2(float shared_data[NUM_WARPS])
{
    float sum{0.0f};
    
    // 由于warp内所有线程访问相同的共享内存位置，会产生广播，无Bank冲突
#pragma unroll
    for (size_t i{0}; i < NUM_WARPS; ++i)
    {
        // 这里不会有共享内存Bank冲突
        // 因为warp中的多个线程访问相同的共享内存位置，结果是广播
        sum += shared_data[i];
    }
    return sum;
}

// warp内归约求和 - 使用shuffle指令
__device__ float warp_reduce_sum(float val)
{
    constexpr unsigned int FULL_MASK{0xffffffff};
    
    // 使用butterfly模式进行warp内归约
#pragma unroll
    for (size_t offset{16}; offset > 0; offset /= 2)
    {
        // __shfl_down_sync将值从更高索引的线程传递到当前线程
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    // 只有warp中的第一个线程会返回正确的结果
    return val;
}

// 块级归约求和版本1 - 使用共享内存
template <size_t NUM_THREADS>
__device__ float block_reduce_sum_v1(float const* __restrict__ input_data,
                                     float shared_data[NUM_THREADS],
                                     size_t num_elements)
{
    static_assert(NUM_THREADS % 32 == 0,
                  "NUM_THREADS must be a multiple of 32");
    
    // 计算每个线程需要处理的元素数量
    size_t const num_elements_per_thread{(num_elements + NUM_THREADS - 1) /
                                         NUM_THREADS};
    size_t const thread_idx{threadIdx.x};
    float sum{0.0f};
    
    // 每个线程处理多个元素
    for (size_t i{0}; i < num_elements_per_thread; ++i)
    {
        size_t const offset{thread_idx + i * NUM_THREADS};
        if (offset < num_elements)
        {
            sum += input_data[offset];
        }
    }
    
    // 将部分和存储到共享内存
    shared_data[thread_idx] = sum;
    __syncthreads();
    
    // 在共享内存中进行最终归约
    float const block_sum{shared_data_reduce_sum_v1<NUM_THREADS>(shared_data)};
    return block_sum;
}

// 块级归约求和版本2 - 结合warp级原语和共享内存
template <size_t NUM_THREADS, size_t NUM_WARPS = NUM_THREADS / 32>
__device__ float block_reduce_sum_v2(float const* __restrict__ input_data,
                                     float shared_data[NUM_WARPS],
                                     size_t num_elements)
{
    // 计算每个线程需要处理的元素数量
    size_t const num_elements_per_thread{(num_elements + NUM_THREADS - 1) /
                                         NUM_THREADS};
    size_t const thread_idx{threadIdx.x};
    float sum{0.0f};
    
    // 每个线程处理多个元素
    for (size_t i{0}; i < num_elements_per_thread; ++i)
    {
        size_t const offset{thread_idx + i * NUM_THREADS};
        if (offset < num_elements)
        {
            sum += input_data[offset];
        }
    }
    
    // 首先在warp内进行归约
    sum = warp_reduce_sum(sum);
    
    // 每个warp的第一个线程将warp的归约结果存储到共享内存
    if (threadIdx.x % 32 == 0)
    {
        shared_data[threadIdx.x / 32] = sum;
    }
    __syncthreads();
    
    // 在warp归约结果之间进行最终归约
    float const block_sum{shared_data_reduce_sum_v2<NUM_WARPS>(shared_data)};
    return block_sum;
}

// 批量归约求和kernel 版本1
template <size_t NUM_THREADS>
__global__ void batched_reduce_sum_v1(float* __restrict__ output_data,
                                      float const* __restrict__ input_data,
                                      size_t num_elements_per_batch)
{
    static_assert(NUM_THREADS % 32 == 0,
                  "NUM_THREADS must be a multiple of 32");
    size_t const block_idx{blockIdx.x};
    size_t const thread_idx{threadIdx.x};
    
    // 每个块使用的共享内存
    __shared__ float shared_data[NUM_THREADS];
    
    // 计算当前块对应的数据的归约和
    float const block_sum{block_reduce_sum_v1<NUM_THREADS>(
        input_data + block_idx * num_elements_per_batch, shared_data,
        num_elements_per_batch)};
    
    // 只有第一个线程写入结果
    if (thread_idx == 0)
    {
        output_data[block_idx] = block_sum;
    }
}

// 批量归约求和kernel 版本2
template <size_t NUM_THREADS>
__global__ void batched_reduce_sum_v2(float* __restrict__ output_data,
                                      float const* __restrict__ input_data,
                                      size_t num_elements_per_batch)
{
    static_assert(NUM_THREADS % 32 == 0,
                  "NUM_THREADS must be a multiple of 32");
    constexpr size_t NUM_WARPS{NUM_THREADS / 32};
    size_t const block_idx{blockIdx.x};
    size_t const thread_idx{threadIdx.x};
    
    // 每个块使用的共享内存，只需要存储每个warp的结果
    __shared__ float shared_data[NUM_WARPS];
    
    // 计算当前块对应的数据的归约和
    float const block_sum{block_reduce_sum_v2<NUM_THREADS, NUM_WARPS>(
        input_data + block_idx * num_elements_per_batch, shared_data,
        num_elements_per_batch)};
    
    // 只有第一个线程写入结果
    if (thread_idx == 0)
    {
        output_data[block_idx] = block_sum;
    }
}

// 启动批量归约求和kernel 版本1
template <size_t NUM_THREADS>
void launch_batched_reduce_sum_v1(float* output_data, float const* input_data,
                                  size_t batch_size,
                                  size_t num_elements_per_batch,
                                  cudaStream_t stream)
{
    size_t const num_blocks{batch_size};
    batched_reduce_sum_v1<NUM_THREADS><<<num_blocks, NUM_THREADS, 0, stream>>>(
        output_data, input_data, num_elements_per_batch);
    CHECK_LAST_CUDA_ERROR();
}

// 启动批量归约求和kernel 版本2
template <size_t NUM_THREADS>
void launch_batched_reduce_sum_v2(float* output_data, float const* input_data,
                                  size_t batch_size,
                                  size_t num_elements_per_batch,
                                  cudaStream_t stream)
{
    size_t const num_blocks{batch_size};
    batched_reduce_sum_v2<NUM_THREADS><<<num_blocks, NUM_THREADS, 0, stream>>>(
        output_data, input_data, num_elements_per_batch);
    CHECK_LAST_CUDA_ERROR();
}

// 性能分析函数
float profile_batched_reduce_sum(
    std::function<void(float*, float const*, size_t, size_t, cudaStream_t)>
        batched_reduce_sum_launch_function,
    size_t batch_size, size_t num_elements_per_batch)
{
    size_t const num_elements{batch_size * num_elements_per_batch};

    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    // 初始化测试数据
    constexpr float element_value{1.0f};
    std::vector<float> input_data(num_elements, element_value);
    std::vector<float> output_data(batch_size, 0.0f);

    // 分配GPU内存
    float* d_input_data;
    float* d_output_data;

    CHECK_CUDA_ERROR(cudaMalloc(&d_input_data, num_elements * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output_data, batch_size * sizeof(float)));

    // 将数据复制到GPU
    CHECK_CUDA_ERROR(cudaMemcpy(d_input_data, input_data.data(),
                                num_elements * sizeof(float),
                                cudaMemcpyHostToDevice));

    // 执行kernel 函数
    batched_reduce_sum_launch_function(d_output_data, d_input_data, batch_size,
                                       num_elements_per_batch, stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // 验证kernel 的正确性
    CHECK_CUDA_ERROR(cudaMemcpy(output_data.data(), d_output_data,
                                batch_size * sizeof(float),
                                cudaMemcpyDeviceToHost));
    for (size_t i{0}; i < batch_size; ++i)
    {
        if (output_data.at(i) != num_elements_per_batch * element_value)
        {
            std::cout << "Expected: " << num_elements_per_batch * element_value
                      << " but got: " << output_data.at(i) << std::endl;
            throw std::runtime_error("Error: incorrect sum");
        }
    }
    
    // 绑定函数以进行性能测量
    std::function<void(cudaStream_t)> const bound_function{std::bind(
        batched_reduce_sum_launch_function, d_output_data, d_input_data,
        batch_size, num_elements_per_batch, std::placeholders::_1)};
    float const latency{measure_performance<void>(bound_function, stream)};
    std::cout << "延迟: " << latency << " ms" << std::endl;

    // 计算有效带宽
    size_t num_bytes{num_elements * sizeof(float) + batch_size * sizeof(float)};
    float const bandwidth{(num_bytes * 1e-6f) / latency};
    std::cout << "有效带宽: " << bandwidth << " GB/s" << std::endl;

    // 清理GPU内存
    CHECK_CUDA_ERROR(cudaFree(d_input_data));
    CHECK_CUDA_ERROR(cudaFree(d_output_data));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

    return latency;
}

int main()
{
    size_t const batch_size{2048};                 // 批次大小
    size_t const num_elements_per_batch{1024 * 256}; // 每批次元素数量

    constexpr size_t string_width{50U};
    std::cout << std_string_centered("", string_width, '~') << std::endl;
    std::cout << std_string_centered("NVIDIA GPU 设备信息", string_width,
                                     ' ')
              << std::endl;
    std::cout << std_string_centered("", string_width, '~') << std::endl;

    // 查询设备名称和峰值内存带宽
    int device_id{0};
    cudaGetDevice(&device_id);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);
    std::cout << "设备名称: " << device_prop.name << std::endl;
    float const memory_size{static_cast<float>(device_prop.totalGlobalMem) /
                            (1 << 30)};
    std::cout << "内存大小: " << memory_size << " GB" << std::endl;
    float const peak_bandwidth{
        static_cast<float>(2.0f * device_prop.memoryClockRate *
                           (device_prop.memoryBusWidth / 8) / 1.0e6)};
    std::cout << "峰值带宽: " << peak_bandwidth << " GB/s" << std::endl;

    std::cout << std_string_centered("", string_width, '~') << std::endl;
    std::cout << std_string_centered("归约求和性能分析", string_width, ' ')
              << std::endl;
    std::cout << std_string_centered("", string_width, '~') << std::endl;

    std::cout << std_string_centered("", string_width, '=') << std::endl;
    std::cout << "批次大小: " << batch_size << std::endl;
    std::cout << "每批次元素数量: " << num_elements_per_batch
              << std::endl;
    std::cout << std_string_centered("", string_width, '=') << std::endl;

    constexpr size_t NUM_THREADS_PER_BATCH{256}; // 每批次线程数
    static_assert(NUM_THREADS_PER_BATCH % 32 == 0,
                  "NUM_THREADS_PER_BATCH must be a multiple of 32");
    static_assert(NUM_THREADS_PER_BATCH <= 1024,
                  "NUM_THREADS_PER_BATCH must be less than or equal to 1024");

    std::cout << "批量归约求和 V1" << std::endl;
    float const latency_v1{profile_batched_reduce_sum(
        launch_batched_reduce_sum_v1<NUM_THREADS_PER_BATCH>, batch_size,
        num_elements_per_batch)};
    std::cout << std_string_centered("", string_width, '-') << std::endl;

    std::cout << "批量归约求和 V2" << std::endl;
    float const latency_v2{profile_batched_reduce_sum(
        launch_batched_reduce_sum_v2<NUM_THREADS_PER_BATCH>, batch_size,
        num_elements_per_batch)};
    std::cout << std_string_centered("", string_width, '-') << std::endl;
}
```

要构建和运行归约求和示例，请运行以下命令。

```shell
$ nvcc reduce_sum.cu -o reduce_sum
$ ./reduce_sum
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
              NVIDIA GPU 设备信息
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
设备名称: NVIDIA GeForce RTX 3090
内存大小: 23.6694 GB
峰值带宽: 936.096 GB/s
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
               归约求和性能分析
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
==================================================
批次大小: 2048
每批次元素数量: 262144
==================================================
批量归约求和 V1
延迟: 2.42976 ms
有效带宽: 883.83 GB/s
--------------------------------------------------
批量归约求和 V2
延迟: 2.44303 ms
有效带宽: 879.028 GB/s
--------------------------------------------------
```

结果表明，两个批量归约求和kernel 具有相似的性能。有效带宽约为GPU峰值带宽的94%。需要注意的是，在我的系统上，有效带宽可能会在一天中的不同时间从运行到运行有所变化，从750 GB/s到900 GB/s。

## 大数组归约求和

如果我们有更大的数组和更小的批次大小怎么办？线程块中的最大线程数是1024。如果只分配一个线程块来计算更大数组中元素的和，而批次大小很小，那么GPU利用率和有效带宽将会很低。

在这种情况下，我们需要将大数组拆分为多个较小的数组，就好像每个大数组是一批数组一样。我们将分配多个线程块来计算每个较小数组中元素的和。一旦计算出每个较小数组中元素的和，我们将使用批量归约求和kernel 再次将部分和进一步归约为最终和。

具体来说，假设一批数据的形状为`(batch_size, num_elements_per_batch)`，如果`num_elements_per_batch`非常大而`batch_size`非常小，我们总是可以将数据重新塑形为`(batch_size * inner_batch_size, inner_num_elements_per_batch)`的形状并运行批量归约求和kernel 。结果归约求和的形状将为`(batch_size * inner_batch_size, 1)`。我们可以进一步将归约求和重新塑形为`(batch_size, inner_batch_size)`（让我们再次称之为`(batch_size, num_elements_per_batch)`）并运行批量归约求和kernel 。这个过程可以重复，直到`num_elements_per_batch`不是太大。

当然，除了多次运行批量归约求和kernel 和同步之外，我们也可以尝试使用原子操作将每个较小数组的部分和添加到全局内存中的最终和。然而，与多次运行批量归约求和kernel 和同步相比，这可能会有也可能不会有性能下降。

## 参考文献

- Optimizing Parallel Reduction in CUDA - Mark Harris(https://leimao.github.io/downloads/blog/2024-07-30-CUDA-Reduction/s7622-Kyrylo-perelygin-robust-and-scalable-cuda.pdf)
- Parallel Computation Patterns (Reduction)(https://www.cs.ucr.edu/~mchow009/teaching/cs147/winter20/slides/5-Reduction.pdf)
- Using CUDA Warp-Level Primitives(https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)



