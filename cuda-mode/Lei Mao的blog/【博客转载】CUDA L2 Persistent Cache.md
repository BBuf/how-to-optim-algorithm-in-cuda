> 博客来源：https://leimao.github.io/blog/CUDA-L2-Persistent-Cache/ ，来自Lei Mao，已获得作者转载授权。

# CUDA L2持久缓存

## 简介

从CUDA 11.0开始，计算能力8.0及以上的设备具有影响L2缓存中数据持久性的能力。由于L2缓存位于芯片上，它可能为全局内存提供更高的带宽和更低的延迟访问。

在这篇博客文章中，我创建了一个CUDA示例来演示如何使用L2持久缓存来加速数据传输。

## CUDA L2持久缓存

在这个示例中，我将有一个包含特定值的小常量缓冲区，该缓冲区将用于重置一个大的流式缓冲区。例如，如果常量缓冲区的大小为4，值为`[5, 2, 1, 4]`，要重置的大流式缓冲区大小为100，重置后大流式缓冲区将具有值`[5, 2, 1, 4, 5, 2, 1, 4, ...]`，即重复常量缓冲区的值。

由于流式缓冲区比常量缓冲区大得多，常量缓冲区中的每个元素被访问的频率比流式缓冲区更高。从全局内存访问缓冲区是非常昂贵的。如果我们能够将频繁访问的常量缓冲区缓存在L2缓存中，对频繁访问的常量缓冲区的访问就可以被加速。

### CUDA数据重置

对于数据重置CUDA内核，我创建了一个基准版本，它在不使用持久L2缓存的情况下启动内核；一个变体版本，它使用3MB持久L2缓存启动内核，但当常量缓冲区大小超过3MB时会发生数据抖动；以及一个优化版本，它使用3MB持久L2缓存启动内核，但消除了数据抖动。

```c++
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
// CUDA错误检查宏，用于检查CUDA API调用的返回值
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, char const* const func, char const* const file,
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

// 检查最后一个CUDA错误的宏
#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(char const* const file, int const line)
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

// 性能测量函数模板，用于测量CUDA内核的执行时间
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

    // 预热阶段，让GPU进入稳定状态
    for (int i{0}; i < num_warmups; ++i)
    {
        bound_function(stream);
    }

    // 等待所有预热操作完成
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // 开始计时
    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    // 执行多次测试以获得平均性能
    for (int i{0}; i < num_repeats; ++i)
    {
        bound_function(stream);
    }
    // 结束计时
    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_LAST_CUDA_ERROR();
    
    // 计算总执行时间
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    
    // 清理事件资源
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    // 返回平均延迟
    float const latency{time / num_repeats};

    return latency;
}

// CUDA内核：使用查找表重置数据流
// data_streaming: 要重置的大数据流
// lut_persistent: 持久查找表（小数据，频繁访问）
// data_streaming_size: 数据流大小
// lut_persistent_size: 查找表大小
__global__ void reset_data(int* data_streaming, int const* lut_persistent,
                           size_t data_streaming_size,
                           size_t lut_persistent_size)
{
    // 计算当前线程的全局索引
    size_t const idx{blockDim.x * blockIdx.x + threadIdx.x};
    // 计算网格步长（所有线程的总数）
    size_t const stride{blockDim.x * gridDim.x};
    
    // 使用网格步长循环处理数据，确保所有数据都被处理
    for (size_t i{idx}; i < data_streaming_size; i += stride)
    {
        // 使用查找表的值循环填充数据流
        data_streaming[i] = lut_persistent[i % lut_persistent_size];
    }
}

/**
 * @brief 使用lut_persistent重置data_streaming，使data_streaming重复lut_persistent的内容
 *
 * @param data_streaming 要重置的数据
 * @param lut_persistent 用于重置data_streaming的值
 * @param data_streaming_size data_streaming的大小
 * @param lut_persistent_size lut_persistent的大小
 * @param stream CUDA流
 */
void launch_reset_data(int* data_streaming, int const* lut_persistent,
                       size_t data_streaming_size, size_t lut_persistent_size,
                       cudaStream_t stream)
{
    // 设置线程块大小
    dim3 const threads_per_block{1024};
    // 设置网格大小
    dim3 const blocks_per_grid{32};
    
    // 启动CUDA内核
    reset_data<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        data_streaming, lut_persistent, data_streaming_size,
        lut_persistent_size);
    CHECK_LAST_CUDA_ERROR();
}

// 验证数据是否正确重置
bool verify_data(int* data, int n, size_t size)
{
    for (size_t i{0}; i < size; ++i)
    {
        if (data[i] != i % n)
        {
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[])
{
    // 默认持久数据大小为3MB
    size_t num_megabytes_persistent_data{3};
    if (argc == 2)
    {
        num_megabytes_persistent_data = std::atoi(argv[1]);
    }

    // 性能测试参数
    constexpr int const num_repeats{100};    // 重复次数
    constexpr int const num_warmups{10};     // 预热次数

    // 获取GPU设备属性
    cudaDeviceProp device_prop{};
    int current_device{0};
    CHECK_CUDA_ERROR(cudaGetDevice(&current_device));
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&device_prop, current_device));
    
    // 打印GPU信息
    std::cout << "GPU: " << device_prop.name << std::endl;
    std::cout << "L2 Cache Size: " << device_prop.l2CacheSize / 1024 / 1024
              << " MB" << std::endl;
    std::cout << "Max Persistent L2 Cache Size: "
              << device_prop.persistingL2CacheMaxSize / 1024 / 1024 << " MB"
              << std::endl;

    // 设置数据大小
    size_t const num_megabytes_streaming_data{1024};  // 流式数据1GB
    if (num_megabytes_persistent_data > num_megabytes_streaming_data)
    {
        std::runtime_error(
            "Try setting persistent data size smaller than 1024 MB.");
    }
    
    // 计算数组元素个数
    size_t const size_persistent(num_megabytes_persistent_data * 1024 * 1024 /
                                 sizeof(int));
    size_t const size_streaming(num_megabytes_streaming_data * 1024 * 1024 /
                                sizeof(int));
    
    std::cout << "Persistent Data Size: " << num_megabytes_persistent_data
              << " MB" << std::endl;
    std::cout << "Steaming Data Size: " << num_megabytes_streaming_data << " MB"
              << std::endl;
    cudaStream_t stream;

    // 创建主机端数据
    std::vector<int> lut_persistent_vec(size_persistent, 0);
    // 初始化查找表，值为0, 1, 2, ...
    for (size_t i{0}; i < lut_persistent_vec.size(); ++i)
    {
        lut_persistent_vec[i] = i;
    }
    std::vector<int> data_streaming_vec(size_streaming, 0);

    // 设备端指针
    int* d_lut_persistent;
    int* d_data_streaming;
    // 主机端指针
    int* h_lut_persistent = lut_persistent_vec.data();
    int* h_data_streaming = data_streaming_vec.data();

    // 分配GPU内存
    CHECK_CUDA_ERROR(
        cudaMalloc(&d_lut_persistent, size_persistent * sizeof(int)));
    CHECK_CUDA_ERROR(
        cudaMalloc(&d_data_streaming, size_streaming * sizeof(int)));
    
    // 创建CUDA流
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    
    // 将查找表数据从主机复制到设备
    CHECK_CUDA_ERROR(cudaMemcpy(d_lut_persistent, h_lut_persistent,
                                size_persistent * sizeof(int),
                                cudaMemcpyHostToDevice));

    // 测试内核正确性
    launch_reset_data(d_data_streaming, d_lut_persistent, size_streaming,
                      size_persistent, stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_CUDA_ERROR(cudaMemcpy(h_data_streaming, d_data_streaming,
                                size_streaming * sizeof(int),
                                cudaMemcpyDeviceToHost));
    assert(verify_data(h_data_streaming, size_persistent, size_streaming));

    // 基准测试：不使用持久L2缓存
    std::function<void(cudaStream_t)> const function{
        std::bind(launch_reset_data, d_data_streaming, d_lut_persistent,
                  size_streaming, size_persistent, std::placeholders::_1)};
    float const latency{
        measure_performance(function, stream, num_repeats, num_warmups)};
    std::cout << std::fixed << std::setprecision(3)
              << "Latency Without Using Persistent L2 Cache: " << latency
              << " ms" << std::endl;

    // 开始使用持久缓存
    cudaStream_t stream_persistent_cache;
    size_t const num_megabytes_persistent_cache{3};  // 持久L2缓存大小3MB
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream_persistent_cache));

    // 设置持久L2缓存大小限制
    CHECK_CUDA_ERROR(
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
                           num_megabytes_persistent_cache * 1024 * 1024));

    // 配置访问策略窗口（可能导致抖动的版本）
    cudaStreamAttrValue stream_attribute_thrashing;
    stream_attribute_thrashing.accessPolicyWindow.base_ptr =
        reinterpret_cast<void*>(d_lut_persistent);  // 持久数据的基地址
    stream_attribute_thrashing.accessPolicyWindow.num_bytes =
        num_megabytes_persistent_data * 1024 * 1024;  // 持久数据的字节数
    stream_attribute_thrashing.accessPolicyWindow.hitRatio = 1.0;  // 命中率100%
    stream_attribute_thrashing.accessPolicyWindow.hitProp =
        cudaAccessPropertyPersisting;  // 命中时使用持久属性
    stream_attribute_thrashing.accessPolicyWindow.missProp =
        cudaAccessPropertyStreaming;   // 未命中时使用流式属性

    // 为流设置访问策略
    CHECK_CUDA_ERROR(cudaStreamSetAttribute(
        stream_persistent_cache, cudaStreamAttributeAccessPolicyWindow,
        &stream_attribute_thrashing));

    // 测试可能导致抖动的持久L2缓存性能
    float const latency_persistent_cache_thrashing{measure_performance(
        function, stream_persistent_cache, num_repeats, num_warmups)};
    std::cout << std::fixed << std::setprecision(3) << "Latency With Using "
              << num_megabytes_persistent_cache
              << " MB Persistent L2 Cache (Potentially Thrashing): "
              << latency_persistent_cache_thrashing << " ms" << std::endl;

    // 配置访问策略窗口（避免抖动的版本）
    cudaStreamAttrValue stream_attribute_non_thrashing{
        stream_attribute_thrashing};
    // 调整命中率以避免抖动：持久缓存大小 / 持久数据大小
    stream_attribute_non_thrashing.accessPolicyWindow.hitRatio =
        std::min(static_cast<double>(num_megabytes_persistent_cache) /
                     num_megabytes_persistent_data,
                 1.0);
    
    // 更新流的访问策略
    CHECK_CUDA_ERROR(cudaStreamSetAttribute(
        stream_persistent_cache, cudaStreamAttributeAccessPolicyWindow,
        &stream_attribute_non_thrashing));

    // 测试避免抖动的持久L2缓存性能
    float const latency_persistent_cache_non_thrashing{measure_performance(
        function, stream_persistent_cache, num_repeats, num_warmups)};
    std::cout << std::fixed << std::setprecision(3) << "Latency With Using "
              << num_megabytes_persistent_cache
              << " MB Persistent L2 Cache (Non-Thrashing): "
              << latency_persistent_cache_non_thrashing << " ms" << std::endl;

    // 清理资源
    CHECK_CUDA_ERROR(cudaFree(d_lut_persistent));
    CHECK_CUDA_ERROR(cudaFree(d_data_streaming));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream_persistent_cache));
}
```

为了避免数据抖动，`accessPolicyWindow.hitRatio`和`accessPolicyWindow.num_bytes`的乘积应该小于或等于`cudaLimitPersistingL2CacheSize`。`accessPolicyWindow.hitRatio`参数可以用来指定接收`accessPolicyWindow.hitProp`属性（通常是`cudaAccessPropertyPersisting`）的访问比例。`accessPolicyWindow.num_bytes`参数可以用来指定访问策略窗口覆盖的字节数，通常是持久数据的大小。

在实践中，我们可以将`accessPolicyWindow.hitRatio`设置为持久L2缓存大小与持久数据大小的比例。例如，如果持久L2缓存大小为3MB，持久数据大小为4MB，我们可以将`accessPolicyWindow.hitRatio`设置为3/4 = 0.75。

### 运行CUDA数据重置

我们可以在NVIDIA Ampere GPU上构建和运行这个示例。在我的情况下，我使用了NVIDIA RTX 3090 GPU。

```shell
$ nvcc l2-persistent.cu -o l2-persistent -std=c++14 --gpu-architecture=compute_80
$ ./l2-persistent
GPU: NVIDIA GeForce RTX 3090
L2 Cache Size: 6 MB
Max Persistent L2 Cache Size: 4 MB
Persistent Data Size: 3 MB
Steaming Data Size: 1024 MB
Latency Without Using Persistent L2 Cache: 3.071 ms
Latency With Using 3 MB Persistent L2 Cache (Potentially Thrashing): 2.436 ms
Latency With Using 3 MB Persistent L2 Cache (Non-Thrashing): 2.443 ms
```

我们可以看到，当持久数据大小为3MB且持久L2缓存为3MB时，应用程序的性能提高了大约20%。

### 基准测试

我们还可以通过改变持久数据大小来运行一些小型基准测试。

```shell
$ ./l2-persistent 1
GPU: NVIDIA GeForce RTX 3090
L2 Cache Size: 6 MB
Max Persistent L2 Cache Size: 4 MB
Persistent Data Size: 1 MB
Steaming Data Size: 1024 MB
Latency Without Using Persistent L2 Cache: 1.754 ms
Latency With Using 3 MB Persistent L2 Cache (Potentially Thrashing): 1.685 ms
Latency With Using 3 MB Persistent L2 Cache (Non-Thrashing): 1.674 ms
$ ./l2-persistent 2
GPU: NVIDIA GeForce RTX 3090
L2 Cache Size: 6 MB
Max Persistent L2 Cache Size: 4 MB
Persistent Data Size: 2 MB
Steaming Data Size: 1024 MB
Latency Without Using Persistent L2 Cache: 2.158 ms
Latency With Using 3 MB Persistent L2 Cache (Potentially Thrashing): 1.997 ms
Latency With Using 3 MB Persistent L2 Cache (Non-Thrashing): 2.002 ms
$ ./l2-persistent 3
GPU: NVIDIA GeForce RTX 3090
L2 Cache Size: 6 MB
Max Persistent L2 Cache Size: 4 MB
Persistent Data Size: 3 MB
Steaming Data Size: 1024 MB
Latency Without Using Persistent L2 Cache: 3.095 ms
Latency With Using 3 MB Persistent L2 Cache (Potentially Thrashing): 2.510 ms
Latency With Using 3 MB Persistent L2 Cache (Non-Thrashing): 2.533 ms
$ ./l2-persistent 4
GPU: NVIDIA GeForce RTX 3090
L2 Cache Size: 6 MB
Max Persistent L2 Cache Size: 4 MB
Persistent Data Size: 4 MB
Steaming Data Size: 1024 MB
Latency Without Using Persistent L2 Cache: 3.906 ms
Latency With Using 3 MB Persistent L2 Cache (Potentially Thrashing): 3.632 ms
Latency With Using 3 MB Persistent L2 Cache (Non-Thrashing): 3.706 ms
$ ./l2-persistent 5
GPU: NVIDIA GeForce RTX 3090
L2 Cache Size: 6 MB
Max Persistent L2 Cache Size: 4 MB
Persistent Data Size: 5 MB
Steaming Data Size: 1024 MB
Latency Without Using Persistent L2 Cache: 4.120 ms
Latency With Using 3 MB Persistent L2 Cache (Potentially Thrashing): 4.554 ms
Latency With Using 3 MB Persistent L2 Cache (Non-Thrashing): 3.920 ms
$ ./l2-persistent 6
GPU: NVIDIA GeForce RTX 3090
L2 Cache Size: 6 MB
Max Persistent L2 Cache Size: 4 MB
Persistent Data Size: 6 MB
Steaming Data Size: 1024 MB
Latency Without Using Persistent L2 Cache: 4.194 ms
Latency With Using 3 MB Persistent L2 Cache (Potentially Thrashing): 4.583 ms
Latency With Using 3 MB Persistent L2 Cache (Non-Thrashing): 4.255 ms

```

我们可以看到，即使当持久数据大小大于持久L2缓存时，使用无抖动持久L2缓存的延迟通常不会比基准性能更差。

## 常见问题

### 持久缓存 VS 共享内存？

持久缓存与共享内存不同。持久缓存对GPU中的所有线程可见，而共享内存只对同一块中的线程可见。

对于小尺寸的频繁访问数据，我们也可以使用共享内存来加速数据访问。然而，共享内存在每个线程块中被限制为48到96KB（取决于GPU），而持久缓存在每个GPU中被限制为几MB。

## 参考资料

- L2 Cache Access Window(https://docs.nvidia.com/cuda/archive/11.7.0/cuda-c-best-practices-guide/index.html#L2-cache-window)
- Function Binding and Performance Measurement(https://leimao.github.io/blog/Function-Binding-Performance-Measurement/)
