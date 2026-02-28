> 博客来源：leimao.github.io/blog/CUDA-Cooperative-Groups/ ，来自Lei Mao，已获得作者转载授权。后续会转载一些Lei Mao的CUDA相关Blog，也是一个完整的专栏，Blog会从稍早一些的CUDA架构到当前最新的CUDA架构，也会包含实用工程技巧，底层指令分析，Cutlass分析等等多个课题，是一个时间线十分明确的专栏。

# CUDA Cooperative Groups

## 简介

CUDA cooperative groups（协作组）是一个功能特性，允许开发者创建和管理能够相互同步和通信的线程组。与传统的CUDA编程模型相比，cooperative groups为在GPU上编写并行算法提供了更加灵活和高效的方式。

在这篇博客文章中，我们将讨论并行归约算法以及其在CUDA中使用cooperative groups的实现。

## 使用Cooperative Groups实现批量归约求和与完整归约求和

在这个例子中，我们修改了之前博客文章"CUDA Reduction"中实现的两个批量归约求和kernel ，使其使用cooperative groups进行线程间的同步和通信。归约算法保持完全相同，只是用于同步线程组的API不同。我们还实现了一个完整归约求和kernel ，该kernel 使用cooperative groups通过单次kernel 启动将元素数组归约为单个值。

```c++
#include <cassert>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <cooperative_groups.h>
#include <cuda_runtime.h>

// CUDA错误检查宏定义，用于自动检查CUDA API调用的返回值
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

// 性能测量函数模板，用于测量CUDA操作的执行时间
template <class T>
float measure_performance(std::function<T(cudaStream_t)> bound_function,
                          cudaStream_t stream, size_t num_repeats = 10,
                          size_t num_warmups = 10)
{
    cudaEvent_t start, stop;
    float time;

    // 创建CUDA事件用于计时
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // 预热运行，确保GPU处于稳定状态
    for (size_t i{0}; i < num_warmups; ++i)
    {
        bound_function(stream);
    }

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // 开始计时并重复执行指定次数
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

// 字符串居中对齐的工具函数
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

// 使用cooperative groups的线程块归约求和（模板版本）
template <size_t NUM_THREADS>
__device__ float thread_block_reduce_sum(
    cooperative_groups::thread_block_tile<NUM_THREADS> group,
    float shared_data[NUM_THREADS], float val)
{
    static_assert(NUM_THREADS % 32 == 0,
                  "NUM_THREADS must be a multiple of 32");
    size_t thread_idx{group.thread_rank()};
    shared_data[thread_idx] = val;
    group.sync(); // 同步线程组
#pragma unroll
    // 使用二分归约算法进行求和
    for (size_t offset{group.size() / 2}; offset > 0; offset /= 2)
    {
        if (thread_idx < offset)
        {
            shared_data[thread_idx] += shared_data[thread_idx + offset];
        }
        group.sync(); // 每次迭代后同步
    }
    // 这里不会有共享内存bank冲突
    // 因为warp中的多个线程访问同一个共享内存位置，导致广播
    return shared_data[0];
}

// 使用cooperative groups的线程块归约求和（动态版本）
__device__ float thread_block_reduce_sum(cooperative_groups::thread_block group,
                                         float* shared_data, float val)
{
    size_t const thread_idx{group.thread_rank()};
    shared_data[thread_idx] = val;
    group.sync(); // 同步整个线程块
    // 使用二分归约算法
    for (size_t stride{group.size() / 2}; stride > 0; stride /= 2)
    {
        if (thread_idx < stride)
        {
            shared_data[thread_idx] += shared_data[thread_idx + stride];
        }
        group.sync(); // 每次迭代后同步
    }
    return shared_data[0];
}

// 最终线程块归约求和（处理多个warp的结果）
template <size_t NUM_WARPS>
__device__ float thread_block_reduce_sum(float shared_data[NUM_WARPS])
{
    float sum{0.0f};
#pragma unroll
    for (size_t i{0}; i < NUM_WARPS; ++i)
    {
        // 这里不会有共享内存bank冲突
        // 因为warp中的多个线程访问同一个共享内存位置，导致广播
        sum += shared_data[i];
    }
    return sum;
}

// 单个线程的归约求和
__device__ float thread_reduce_sum(float const* __restrict__ input_data,
                                   size_t start_offset, size_t num_elements,
                                   size_t stride)
{
    float sum{0.0f};
    // 按步长遍历并累加元素
    for (size_t i{start_offset}; i < num_elements; i += stride)
    {
        sum += input_data[i];
    }
    return sum;
}

// warp级别的归约求和
__device__ float
warp_reduce_sum(cooperative_groups::thread_block_tile<32> group, float val)
{
#pragma unroll
    for (size_t offset{group.size() / 2}; offset > 0; offset /= 2)
    {
        // shfl_down函数是warp shuffle操作，只对大小为32的线程块tile存在
        val += group.shfl_down(val, offset);
    }
    // 只有warp中的第一个线程会返回正确的结果
    return val;
}

// 线程块归约求和版本1：使用所有线程进行归约
template <size_t NUM_THREADS>
__device__ float
thread_block_reduce_sum_v1(float const* __restrict__ input_data,
                           size_t num_elements)
{
    static_assert(NUM_THREADS % 32 == 0,
                  "NUM_THREADS must be a multiple of 32");
    __shared__ float shared_data[NUM_THREADS];
    size_t const thread_idx{
        cooperative_groups::this_thread_block().thread_index().x};
    // 每个线程计算自己负责的元素的和
    float sum{
        thread_reduce_sum(input_data, thread_idx, num_elements, NUM_THREADS)};
    shared_data[thread_idx] = sum;
    // 注意：静态线程块cooperative groups仍然不被支持
    // 这种方式不工作：
    // cooperative_groups::thread_block_tile<NUM_THREADS> const
    // thread_block{cooperative_groups::tiled_partition<NUM_THREADS>(cooperative_groups::this_thread_block())};
    // float const block_sum{thread_block_reduce_sum<NUM_THREADS>(thread_block,
    // shared_data, sum)};
    // 这种方式可以工作：
    float const block_sum{thread_block_reduce_sum(
        cooperative_groups::this_thread_block(), shared_data, sum)};
    return block_sum;
}

// 线程块归约求和版本2：先在warp内归约，再在warp间归约
template <size_t NUM_THREADS, size_t NUM_WARPS = NUM_THREADS / 32>
__device__ float
thread_block_reduce_sum_v2(float const* __restrict__ input_data,
                           size_t num_elements)
{
    static_assert(NUM_THREADS % 32 == 0,
                  "NUM_THREADS must be a multiple of 32");
    __shared__ float shared_data[NUM_WARPS];
    size_t const thread_idx{
        cooperative_groups::this_thread_block().thread_index().x};
    // 每个线程计算自己负责的元素的和
    float sum{
        thread_reduce_sum(input_data, thread_idx, num_elements, NUM_THREADS)};
    // 创建warp级别的cooperative group
    cooperative_groups::thread_block_tile<32> const warp{
        cooperative_groups::tiled_partition<32>(
            cooperative_groups::this_thread_block())};
    // 在warp内进行归约
    sum = warp_reduce_sum(warp, sum);
    // 只有每个warp的第一个线程将结果存储到共享内存
    if (warp.thread_rank() == 0)
    {
        shared_data[cooperative_groups::this_thread_block().thread_rank() /
                    32] = sum;
    }
    cooperative_groups::this_thread_block().sync();
    // 最终归约多个warp的结果
    float const block_sum{thread_block_reduce_sum<NUM_WARPS>(shared_data)};
    return block_sum;
}

// 批量归约求和kernel版本1
template <size_t NUM_THREADS>
__global__ void batched_reduce_sum_v1(float* __restrict__ output_data,
                                      float const* __restrict__ input_data,
                                      size_t num_elements_per_batch)
{
    static_assert(NUM_THREADS % 32 == 0,
                  "NUM_THREADS must be a multiple of 32");
    size_t const block_idx{cooperative_groups::this_grid().block_rank()};
    size_t const thread_idx{
        cooperative_groups::this_thread_block().thread_rank()};
    // 每个block处理一个batch的数据
    float const block_sum{thread_block_reduce_sum_v1<NUM_THREADS>(
        input_data + block_idx * num_elements_per_batch,
        num_elements_per_batch)};
    // 只有每个block的第一个线程写入结果
    if (thread_idx == 0)
    {
        output_data[block_idx] = block_sum;
    }
}

// 批量归约求和kernel版本2
template <size_t NUM_THREADS>
__global__ void batched_reduce_sum_v2(float* __restrict__ output_data,
                                      float const* __restrict__ input_data,
                                      size_t num_elements_per_batch)
{
    static_assert(NUM_THREADS % 32 == 0,
                  "NUM_THREADS must be a multiple of 32");
    constexpr size_t NUM_WARPS{NUM_THREADS / 32};
    size_t const block_idx{cooperative_groups::this_grid().block_rank()};
    size_t const thread_idx{
        cooperative_groups::this_thread_block().thread_rank()};
    // 每个block处理一个batch的数据，使用版本2的归约算法
    float const block_sum{thread_block_reduce_sum_v2<NUM_THREADS, NUM_WARPS>(
        input_data + block_idx * num_elements_per_batch,
        num_elements_per_batch)};
    // 只有每个block的第一个线程写入结果
    if (thread_idx == 0)
    {
        output_data[block_idx] = block_sum;
    }
}

// 完整归约求和kernel：将整个数组归约为单个值
template <size_t NUM_THREADS, size_t NUM_BLOCK_ELEMENTS>
__global__ void full_reduce_sum(float* output,
                                float const* __restrict__ input_data,
                                size_t num_elements, float* workspace)
{
    static_assert(NUM_THREADS % 32 == 0,
                  "NUM_THREADS must be a multiple of 32");
    static_assert(NUM_BLOCK_ELEMENTS % NUM_THREADS == 0,
                  "NUM_BLOCK_ELEMENTS must be a multiple of NUM_THREADS");
    // 工作空间大小：num_elements
    size_t const num_grid_elements{
        NUM_BLOCK_ELEMENTS * cooperative_groups::this_grid().num_blocks()};
    float* const workspace_ptr_1{workspace};
    float* const workspace_ptr_2{workspace + num_elements / 2};
    size_t remaining_elements{num_elements};

    // 第一次归约迭代
    float* workspace_output_data{workspace_ptr_1};
    size_t const num_grid_iterations{
        (remaining_elements + num_grid_elements - 1) / num_grid_elements};
    for (size_t i{0}; i < num_grid_iterations; ++i)
    {
        size_t const grid_offset{i * num_grid_elements};
        size_t const block_offset{grid_offset +
                                  cooperative_groups::this_grid().block_rank() *
                                      NUM_BLOCK_ELEMENTS};
        size_t const num_actual_elements_to_reduce_per_block{
            remaining_elements >= block_offset
                ? min(NUM_BLOCK_ELEMENTS, remaining_elements - block_offset)
                : 0};
        // 每个block归约其负责的元素
        float const block_sum{thread_block_reduce_sum_v1<NUM_THREADS>(
            input_data + block_offset,
            num_actual_elements_to_reduce_per_block)};
        if (cooperative_groups::this_thread_block().thread_rank() == 0)
        {
            workspace_output_data
                [i * cooperative_groups::this_grid().num_blocks() +
                 cooperative_groups::this_grid().block_rank()] = block_sum;
        }
    }
    // 网格级别同步：等待所有block完成
    cooperative_groups::this_grid().sync();
    remaining_elements =
        (remaining_elements + NUM_BLOCK_ELEMENTS - 1) / NUM_BLOCK_ELEMENTS;

    // 后续的归约迭代
    float* workspace_input_data{workspace_output_data};
    workspace_output_data = workspace_ptr_2;
    while (remaining_elements > 1)
    {
        size_t const num_grid_iterations{
            (remaining_elements + num_grid_elements - 1) / num_grid_elements};
        for (size_t i{0}; i < num_grid_iterations; ++i)
        {
            size_t const grid_offset{i * num_grid_elements};
            size_t const block_offset{
                grid_offset + cooperative_groups::this_grid().block_rank() *
                                  NUM_BLOCK_ELEMENTS};
            size_t const num_actual_elements_to_reduce_per_block{
                remaining_elements >= block_offset
                    ? min(NUM_BLOCK_ELEMENTS, remaining_elements - block_offset)
                    : 0};
            // 归约工作空间中的数据
            float const block_sum{thread_block_reduce_sum_v1<NUM_THREADS>(
                workspace_input_data + block_offset,
                num_actual_elements_to_reduce_per_block)};
            if (cooperative_groups::this_thread_block().thread_rank() == 0)
            {
                workspace_output_data
                    [i * cooperative_groups::this_grid().num_blocks() +
                     cooperative_groups::this_grid().block_rank()] = block_sum;
            }
        }
        // 网格级别同步
        cooperative_groups::this_grid().sync();
        remaining_elements =
            (remaining_elements + NUM_BLOCK_ELEMENTS - 1) / NUM_BLOCK_ELEMENTS;

        // 交换输入和输出数据指针
        float* const temp{workspace_input_data};
        workspace_input_data = workspace_output_data;
        workspace_output_data = temp;
    }

    // 将最终结果复制到输出
    workspace_output_data = workspace_input_data;
    if (cooperative_groups::this_grid().thread_rank() == 0)
    {
        *output = workspace_output_data[0];
    }
}

// 启动批量归约求和版本1的包装函数
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

// 启动批量归约求和版本2的包装函数
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

// 启动完整归约求和的包装函数
template <size_t NUM_THREADS, size_t NUM_BLOCK_ELEMENTS>
void launch_full_reduce_sum(float* output, float const* input_data,
                            size_t num_elements, float* workspace,
                            cudaStream_t stream)
{
    // 参考：https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-programming-guide/index.html#grid-synchronization
    void const* func{reinterpret_cast<void const*>(
        full_reduce_sum<NUM_THREADS, NUM_BLOCK_ELEMENTS>)};
    int dev{0};
    cudaDeviceProp deviceProp;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, dev));
    // 使用设备的多处理器数量作为网格大小
    dim3 const grid_dim{
        static_cast<unsigned int>(deviceProp.multiProcessorCount)};
    dim3 const block_dim{NUM_THREADS};

    // 这将启动一个能够最大程度填充GPU的网格
    // 在实践中，这并不总是最好的选择
    // void const* func{reinterpret_cast<void const*>(
    //     full_reduce_sum<NUM_THREADS, NUM_BLOCK_ELEMENTS>)};
    // int dev{0};
    // dim3 const block_dim{NUM_THREADS};
    // int num_blocks_per_sm{0};
    // cudaDeviceProp deviceProp;
    // cudaGetDeviceProperties(&deviceProp, dev);
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, func,
    //                                               NUM_THREADS, 0);
    // dim3 const grid_dim{static_cast<unsigned int>(num_blocks_per_sm)};

    void* args[]{static_cast<void*>(&output), static_cast<void*>(&input_data),
                 static_cast<void*>(&num_elements),
                 static_cast<void*>(&workspace)};
    // 启动cooperative kernel
    CHECK_CUDA_ERROR(cudaLaunchCooperativeKernel(func, grid_dim, block_dim,
                                                 args, 0, stream));
    CHECK_LAST_CUDA_ERROR();
}

// 完整归约求和性能分析函数
float profile_full_reduce_sum(
    std::function<void(float*, float const*, size_t, float*, cudaStream_t)>
        full_reduce_sum_launch_function,
    size_t num_elements)
{
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    constexpr float element_value{1.0f};
    std::vector<float> input_data(num_elements, element_value);
    float output{0.0f};

    float* d_input_data;
    float* d_workspace;
    float* d_output;

    // 分配GPU内存
    CHECK_CUDA_ERROR(cudaMalloc(&d_input_data, num_elements * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_workspace, num_elements * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, sizeof(float)));

    // 将数据从主机复制到设备
    CHECK_CUDA_ERROR(cudaMemcpy(d_input_data, input_data.data(),
                                num_elements * sizeof(float),
                                cudaMemcpyHostToDevice));

    // 执行归约操作
    full_reduce_sum_launch_function(d_output, d_input_data, num_elements,
                                    d_workspace, stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // 验证kernel的正确性
    CHECK_CUDA_ERROR(
        cudaMemcpy(&output, d_output, sizeof(float), cudaMemcpyDeviceToHost));
    if (output != num_elements * element_value)
    {
        std::cout << "Expected: " << num_elements * element_value
                  << " but got: " << output << std::endl;
        throw std::runtime_error("Error: incorrect sum");
    }
    // 测量性能
    std::function<void(cudaStream_t)> const bound_function{
        std::bind(full_reduce_sum_launch_function, d_output, d_input_data,
                  num_elements, d_workspace, std::placeholders::_1)};
    float const latency{measure_performance<void>(bound_function, stream)};
    std::cout << "Latency: " << latency << " ms" << std::endl;

    // 计算有效带宽
    size_t num_bytes{num_elements * sizeof(float) + sizeof(float)};
    float const bandwidth{(num_bytes * 1e-6f) / latency};
    std::cout << "Effective Bandwidth: " << bandwidth << " GB/s" << std::endl;

    // 清理GPU内存
    CHECK_CUDA_ERROR(cudaFree(d_input_data));
    CHECK_CUDA_ERROR(cudaFree(d_workspace));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

    return latency;
}

// 批量归约求和性能分析函数
float profile_batched_reduce_sum(
    std::function<void(float*, float const*, size_t, size_t, cudaStream_t)>
        batched_reduce_sum_launch_function,
    size_t batch_size, size_t num_elements_per_batch)
{
    size_t const num_elements{batch_size * num_elements_per_batch};

    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    constexpr float element_value{1.0f};
    std::vector<float> input_data(num_elements, element_value);
    std::vector<float> output_data(batch_size, 0.0f);

    float* d_input_data;
    float* d_output_data;

    // 分配GPU内存
    CHECK_CUDA_ERROR(cudaMalloc(&d_input_data, num_elements * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output_data, batch_size * sizeof(float)));

    // 将数据从主机复制到设备
    CHECK_CUDA_ERROR(cudaMemcpy(d_input_data, input_data.data(),
                                num_elements * sizeof(float),
                                cudaMemcpyHostToDevice));

    // 执行批量归约操作
    batched_reduce_sum_launch_function(d_output_data, d_input_data, batch_size,
                                       num_elements_per_batch, stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // 验证kernel的正确性
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
    // 测量性能
    std::function<void(cudaStream_t)> const bound_function{std::bind(
        batched_reduce_sum_launch_function, d_output_data, d_input_data,
        batch_size, num_elements_per_batch, std::placeholders::_1)};
    float const latency{measure_performance<void>(bound_function, stream)};
    std::cout << "Latency: " << latency << " ms" << std::endl;

    // 计算有效带宽
    size_t num_bytes{num_elements * sizeof(float) + batch_size * sizeof(float)};
    float const bandwidth{(num_bytes * 1e-6f) / latency};
    std::cout << "Effective Bandwidth: " << bandwidth << " GB/s" << std::endl;

    // 清理GPU内存
    CHECK_CUDA_ERROR(cudaFree(d_input_data));
    CHECK_CUDA_ERROR(cudaFree(d_output_data));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

    return latency;
}

// 主函数：演示和测试不同的归约算法
int main()
{
    size_t const batch_size{2048};
    size_t const num_elements_per_batch{1024 * 256};

    constexpr size_t string_width{50U};
    std::cout << std_string_centered("", string_width, '~') << std::endl;
    std::cout << std_string_centered("NVIDIA GPU Device Info", string_width,
                                     ' ')
              << std::endl;
    std::cout << std_string_centered("", string_width, '~') << std::endl;

    // 查询设备名称和峰值内存带宽
    int device_id{0};
    cudaGetDevice(&device_id);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);
    std::cout << "Device Name: " << device_prop.name << std::endl;
    float const memory_size{static_cast<float>(device_prop.totalGlobalMem) /
                            (1 << 30)};
    std::cout << "Memory Size: " << memory_size << " GB" << std::endl;
    float const peak_bandwidth{
        static_cast<float>(2.0f * device_prop.memoryClockRate *
                           (device_prop.memoryBusWidth / 8) / 1.0e6)};
    std::cout << "Peak Bandwitdh: " << peak_bandwidth << " GB/s" << std::endl;

    std::cout << std_string_centered("", string_width, '~') << std::endl;
    std::cout << std_string_centered("Reduce Sum Profiling", string_width, ' ')
              << std::endl;
    std::cout << std_string_centered("", string_width, '~') << std::endl;

    std::cout << std_string_centered("", string_width, '=') << std::endl;
    std::cout << "Batch Size: " << batch_size << std::endl;
    std::cout << "Number of Elements Per Batch: " << num_elements_per_batch
              << std::endl;
    std::cout << std_string_centered("", string_width, '=') << std::endl;

    constexpr size_t NUM_THREADS_PER_BATCH{256};
    static_assert(NUM_THREADS_PER_BATCH % 32 == 0,
                  "NUM_THREADS_PER_BATCH must be a multiple of 32");
    static_assert(NUM_THREADS_PER_BATCH <= 1024,
                  "NUM_THREADS_PER_BATCH must be less than or equal to 1024");

    // 测试批量归约求和版本1
    std::cout << "Batched Reduce Sum V1" << std::endl;
    float const latency_v1{profile_batched_reduce_sum(
        launch_batched_reduce_sum_v1<NUM_THREADS_PER_BATCH>, batch_size,
        num_elements_per_batch)};
    std::cout << std_string_centered("", string_width, '-') << std::endl;

    // 测试批量归约求和版本2
    std::cout << "Batched Reduce Sum V2" << std::endl;
    float const latency_v2{profile_batched_reduce_sum(
        launch_batched_reduce_sum_v2<NUM_THREADS_PER_BATCH>, batch_size,
        num_elements_per_batch)};
    std::cout << std_string_centered("", string_width, '-') << std::endl;

    // 测试完整归约求和
    std::cout << "Full Reduce Sum" << std::endl;
    constexpr size_t NUM_THREADS{256};
    constexpr size_t NUM_BLOCK_ELEMENTS{NUM_THREADS * 1024};
    float const latency_v3{profile_full_reduce_sum(
        launch_full_reduce_sum<NUM_THREADS, NUM_BLOCK_ELEMENTS>,
        batch_size * num_elements_per_batch)};
    std::cout << std_string_centered("", string_width, '-') << std::endl;
}
```

要构建和运行这个归约求和示例，请运行以下命令：

```shell
$ nvcc reduce_sum_cooperative_groups.cu -o reduce_sum_cooperative_groups
$ ./reduce_sum_cooperative_groups
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
              NVIDIA GPU Device Info
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Device Name: NVIDIA GeForce RTX 3090
Memory Size: 23.6694 GB
Peak Bandwitdh: 936.096 GB/s
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
               Reduce Sum Profiling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
==================================================
Batch Size: 2048
Number of Elements Per Batch: 262144
==================================================
Batched Reduce Sum V1
Latency: 2.43301 ms
Effective Bandwidth: 882.649 GB/s
--------------------------------------------------
Batched Reduce Sum V2
Latency: 2.43445 ms
Effective Bandwidth: 882.126 GB/s
--------------------------------------------------
Full Reduce Sum
Latency: 2.47788 ms
Effective Bandwidth: 866.663 GB/s
--------------------------------------------------
```

使用cooperative groups的批量归约求和kernel 的性能与使用传统CUDA编程模型的批量归约求和kernel 性能相似。

## 大数组归约求和

可以有三种方法来实现大数组归约求和kernel ：

- 使用多次批量归约求和kernel 启动迭代地归约数组。
- 使用一次完整归约求和kernel 启动迭代地归约数组，其中kernel 由网格协作组管理。

在不使用cooperative groups的情况下，我们只能在线程块内同步线程，这导致了第一种方法。但由于多次kernel 启动，会有额外的kernel 启动开销。

使用cooperative groups，我们可以跨线程块同步线程，这导致了第二种方法。然而，与第一种方法相比，第二种方法也有缺点，即在归约的后期阶段，实际使用的网格数量要少得多，因为归约问题的规模变小了，这是对计算资源的浪费。

## 参考文献

- Cooperative Groups: Flexible CUDA Thread Programming(https://developer.nvidia.com/blog/cooperative-groups/)
- Cooperative Groups - NVIDIA GTC 2017(https://leimao.github.io/downloads/blog/2024-08-06-CUDA-Cooperative-Groups/s7622-Kyrylo-perelygin-robust-and-scalable-cuda.pdf)

