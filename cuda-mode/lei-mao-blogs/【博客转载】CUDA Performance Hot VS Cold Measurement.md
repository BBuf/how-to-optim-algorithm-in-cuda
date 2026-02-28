> 博客来源：https://leimao.github.io/blog/CUDA-Performance-Hot-Cold-Measurement/ ，来自Lei Mao，已获得作者转载授权。后续会转载一些Lei Mao的CUDA相关Blog，也是一个完整的专栏，Blog会从稍早一些的CUDA架构到当前最新的CUDA架构，也会包含实用工程技巧，底层指令分析，Cutlass分析等等多个课题，是一个时间线十分明确的专栏。

# CUDA性能热测量与冷测量

## 简介

为了测量CUDA kernel的性能，通常用户会多次运行 kernel并取执行时间的平均值。然而，CUDA kernel的性能可能会受到缓存效应的影响，从而导致测量到的性能与实际性能有所不同。

例如，在性能测量期间，每次CUDA kernel调用中，CUDA kernel都会访问相同的输入数据，导致从L2缓存读取而不访问DRAM，而在实际应用中，每次 kernel调用的输入数据可能不同，导致 kernel从DRAM读取。为了在某些特定用例的性能测量中消除缓存效应，用户可以在每次运行 kernel之前刷新GPU L2缓存。因此， kernel将始终在"冷"状态下运行。

在这篇博客文章中，我将讨论如何在"热"状态和"冷"状态下测量CUDA kernel的性能。

## CUDA性能热测量与冷测量

在我之前的博客文章"函数绑定和性能测量"(https://leimao.github.io/blog/Function-Binding-Performance-Measurement/)中，我讨论了如何使用函数绑定来测量CUDA kernel的性能。该性能测量实现实际上只能测量CUDA kernel在"热"状态下的性能。为了测量CUDA kernel在"冷"状态下的性能，我们可以稍微修改实现，使得在每次运行 kernel之前刷新L2缓存。

### L2缓存刷新

在CUDA中没有直接刷新GPU L2缓存的API。但是，我们可以在GPU内存中分配一个与L2缓存相同大小的缓冲区，并向其写入一些值。这将导致L2缓存中之前缓存的所有值被逐出。以下示例展示了如何在"热"状态和"冷"状态下测量CUDA kernel的性能。

```c++
#include <functional>
#include <iomanip>
#include <iostream>

#include <cuda_runtime.h>
// 检查CUDA错误的宏定义
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
// CUDA错误检查函数，如果出现错误则打印错误信息并退出程序
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

// 性能测量函数模板
// bound_function: 绑定的函数对象
// stream: CUDA流
// num_repeats: 重复测量次数
// num_warmups: 预热次数
// flush_l2_cache: 是否在每次测量前刷新L2缓存
template <class T>
float measure_performance(std::function<T(cudaStream_t)> bound_function,
                          cudaStream_t stream, size_t num_repeats = 100,
                          size_t num_warmups = 100, bool flush_l2_cache = false)
{
    int device_id{0};
    int l2_cache_size{0};
    // 获取当前设备ID
    CHECK_CUDA_ERROR(cudaGetDevice(&device_id));
    // 获取L2缓存大小
    CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&l2_cache_size,
                                            cudaDevAttrL2CacheSize, device_id));

    // 分配与L2缓存同样大小的缓冲区，用于刷新L2缓存
    void* l2_flush_buffer{nullptr};
    CHECK_CUDA_ERROR(
        cudaMalloc(&l2_flush_buffer, static_cast<size_t>(l2_cache_size)));

    // 创建CUDA事件用于计时
    cudaEvent_t start, stop;
    float time{0.0f};
    float call_time{0.0f};

    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // 预热阶段：运行kernel多次以稳定性能
    for (size_t i{0}; i < num_warmups; ++i)
    {
        bound_function(stream);
    }

    // 等待所有预热操作完成
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // 正式测量阶段
    for (size_t i{0}; i < num_repeats; ++i)
    {
        // 如果需要冷测量，则在每次运行前刷新L2缓存
        if (flush_l2_cache)
        {
            // 通过向L2缓存大小的缓冲区写入数据来刷新L2缓存
            CHECK_CUDA_ERROR(cudaMemsetAsync(l2_flush_buffer, 0,
                                             static_cast<size_t>(l2_cache_size),
                                             stream));
            CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
        }
        // 记录开始时间
        CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
        // 执行被测量的函数
        CHECK_CUDA_ERROR(bound_function(stream));
        // 记录结束时间
        CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
        // 等待stop事件完成
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        // 计算本次执行时间
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&call_time, start, stop));
        // 累加总时间
        time += call_time;
    }
    // 销毁CUDA事件
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    // 释放L2缓存刷新缓冲区
    CHECK_CUDA_ERROR(cudaFree(l2_flush_buffer));

    // 计算平均延迟
    float const latency{time / num_repeats};

    return latency;
}

// 简单的数组拷贝kernel
__global__ void copy(float* output, float const* input, size_t n)
{
    // 计算当前线程的全局索引
    size_t const idx{blockDim.x * blockIdx.x + threadIdx.x};
    // 计算网格步长（总线程数）
    size_t const stride{blockDim.x * gridDim.x};
    // 使用网格步长循环处理数组元素
    for (size_t i{idx}; i < n; i += stride)
    {
        output[i] = input[i];
    }
}

// 启动拷贝kernel的包装函数
cudaError_t launch_copy(float* output, float const* input, size_t n,
                        cudaStream_t stream)
{
    // 设置线程块大小
    dim3 const threads_per_block{1024};
    // 设置网格大小
    dim3 const blocks_per_grid{32};
    // 启动kernel
    copy<<<blocks_per_grid, threads_per_block, 0, stream>>>(output, input, n);
    // 返回最后的CUDA错误状态
    return cudaGetLastError();
}

int main()
{
    int device_id{0};
    // 获取当前设备ID
    CHECK_CUDA_ERROR(cudaGetDevice(&device_id));
    cudaDeviceProp device_prop;
    // 获取设备属性
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&device_prop, device_id));
    // 打印设备名称
    std::cout << "Device Name: " << device_prop.name << std::endl;
    // 计算并打印DRAM大小（GB）
    float const memory_size{static_cast<float>(device_prop.totalGlobalMem) /
                            (1 << 30)};
    std::cout << "DRAM Size: " << memory_size << " GB" << std::endl;
    // 计算并打印DRAM峰值带宽（GB/s）
    float const peak_bandwidth{
        static_cast<float>(2.0f * device_prop.memoryClockRate *
                           (device_prop.memoryBusWidth / 8) / 1.0e6)};
    std::cout << "DRAM Peak Bandwitdh: " << peak_bandwidth << " GB/s"
              << std::endl;
    // 获取L2缓存大小
    int const l2_cache_size{device_prop.l2CacheSize};
    // 计算并打印L2缓存大小（MB）
    float const l2_cache_size_mb{static_cast<float>(l2_cache_size) / (1 << 20)};
    std::cout << "L2 Cache Size: " << l2_cache_size_mb << " MB" << std::endl;

    // 设置测量参数
    constexpr size_t num_repeats{10000};  // 重复测量次数
    constexpr size_t num_warmups{1000};   // 预热次数

    // 计算数组大小（L2缓存大小的一半，以float为单位）
    size_t const n{l2_cache_size / 2 / sizeof(float)};
    cudaStream_t stream;

    float *d_input, *d_output;

    // 分配GPU内存
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, n * sizeof(float)));

    // 创建CUDA流
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    // 创建绑定函数对象
    std::function<cudaError_t(cudaStream_t)> function{
        std::bind(launch_copy, d_output, d_input, n, std::placeholders::_1)};

    // 测量热状态下的性能（不刷新L2缓存）
    float const hot_latency{
        measure_performance(function, stream, num_repeats, num_warmups, false)};
    std::cout << std::fixed << std::setprecision(4)
              << "Hot Latency: " << hot_latency << " ms" << std::endl;

    // 测量冷状态下的性能（每次都刷新L2缓存）
    float const cold_latency{
        measure_performance(function, stream, num_repeats, num_warmups, true)};
    std::cout << std::fixed << std::setprecision(4)
              << "Cold Latency: " << cold_latency << " ms" << std::endl;

    // 清理资源
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}
```

要构建和运行这个示例，请执行以下命令。

```shell
$ nvcc measure_performance.cu -o measure_performance -std=c++14
$ ./measure_performance
Device Name: NVIDIA GeForce RTX 3090
DRAM Size: 23.4365 GB
DRAM Peak Bandwitdh: 936.096 GB/s
L2 Cache Size: 6 MB
Hot Latency: 0.0095 ms
Cold Latency: 0.0141 ms
```

我们可以看到"热"状态和"冷"状态之间存在性能差异，这种性能差异是由于缓存效应造成的。然而，如果 kernel不是内存密集型的，或者缓存大小太小而对问题没有帮助，那么"热"状态和"冷"状态之间的性能差异可能可以忽略不计。

### Nsight Compute

使用NVIDIA Nsight Compute来测量CUDA kernel的性能也是很常见的。

为了使硬件性能计数器值更加确定性，NVIDIA Nsight Compute默认使用`--cache-control all`在每次重放通道之前刷新所有GPU缓存。因此，在每个通道中， kernel将访问干净的缓存，行为就像 kernel在完全隔离的环境中执行一样。

这种行为对于性能分析可能是不希望的，特别是如果测量专注于更大应用程序执行中的 kernel，并且收集的数据针对以缓存为中心的指标。在这种情况下，您可以使用`--cache-control none`来禁用工具对任何硬件缓存的刷新。

```shell
$ ncu --help
  --cache-control arg (=all)            Control the behavior of the GPU caches during profiling. Allowed values:
                                          all
                                          none
```

## 参考资料

- Measure Cold - NVBench(https://github.com/NVIDIA/nvbench/blob/c03033b50e46748207b27685b1cdfcbe4a2fec59/nvbench/detail/measure_cold.cuh)
- L2 Flush - NVBench(https://github.com/NVIDIA/nvbench/blob/c03033b50e46748207b27685b1cdfcbe4a2fec59/nvbench/detail/l2flush.cuh)
- Cache Control - Nsight Compute(https://docs.nvidia.com/nsight-compute/2025.1/ProfilingGuide/index.html#cache-control)








