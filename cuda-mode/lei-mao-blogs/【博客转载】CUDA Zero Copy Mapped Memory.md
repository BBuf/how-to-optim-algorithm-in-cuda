> > 博客来源：https://leimao.github.io/blog/CUDA-Zero-Copy-Mapped-Memory/ ，来自Lei Mao，已获得作者转载授权。

# CUDA零拷贝映射内存

## 简介

统一内存在NVIDIA嵌入式平台上使用，如NVIDIA Drive系列和NVIDIA Jetson系列。由于CPU和集成GPU使用相同的内存，因此可以消除通常在使用独立GPU系统上发生的主机和设备之间的CUDA内存拷贝，使GPU可以直接访问CPU的输出，CPU也可以直接访问GPU的输出。这样，在某些使用场景下系统性能可以得到显著提升。

在这篇博客中，我想讨论CUDA mapped pinned内存与CUDA non-mapped pinned内存，并比较它们在内存受限kernel上的性能。

## CUDA Pinned Mapped内存

CUDA pinned mapped内存使GPU线程能够直接访问主机内存。为此，它需要mapped pinned（non-pageable, page-locked）内存(https://leimao.github.io/blog/Page-Locked-Host-Memory-Data-Transfer/)。在集成GPU上（即CUDA设备属性结构的integrated字段设置为1的GPU），mapped pinned内存总是性能收益，因为它避免了多余的拷贝，集成GPU和CPU内存在物理上是相同的。在独立GPU上，mapped pinned内存只在某些情况下是有优势的。因为数据不会缓存在GPU上，mapped pinned内存应该只读取或写入一次，并且读写内存的全局加载和存储应该是合并的。零拷贝可以代替流使用，因为kernel发起的数据传输会自动与kernel执行重叠，而无需设置和确定最佳流数量的开销。

## CUDA Pinned内存：Non-Mapped VS Mapped

以下实现比较了内存受限kernel的延迟以及必要时主机和设备之间的内存拷贝。

CUDA mapped内存也使用pinned内存。对于CUDA pinned内存，我们仍然需要分配设备内存并在主机内存和设备内存之间传输数据，而对于CUDA mapped内存，设备内存分配和内存传输（如果有的话）是抽象的。

```c++
#include <cassert>
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
// CUDA错误检查宏，用于检查CUDA API调用的返回值
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file,
           const int line)
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
void checkLast(const char* const file, const int line)
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

// 性能测量模板函数，用于测量CUDA kernel的执行时间
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

    // 预热阶段，避免首次执行的开销影响测量结果
    for (int i{0}; i < num_warmups; ++i)
    {
        bound_function(stream);
    }

    // 等待流中所有操作完成
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

// CUDA kernel：执行浮点数加法运算
__global__ void float_addition(float* output, float const* input_1,
                               float const* input_2, uint32_t n)
{
    // 计算当前线程的全局索引
    const uint32_t idx{blockDim.x * blockIdx.x + threadIdx.x};
    // 计算网格步长（所有线程块中的总线程数）
    const uint32_t stride{blockDim.x * gridDim.x};
    
    // 使用网格步长循环处理数组元素，确保所有元素都被处理
    for (uint32_t i{idx}; i < n; i += stride)
    {
        output[i] = input_1[i] + input_2[i];
    }
}

// 使用non-mapped pinned内存启动浮点加法kernel
void launch_float_addition_non_mapped_pinned_memory(
    float* h_output, float const* h_input_1, float const* h_input_2,
    float* d_output, float* d_input_1, float* d_input_2, uint32_t n,
    cudaStream_t stream)
{
    // 异步将输入数据从主机内存拷贝到设备内存
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_input_1, h_input_1, n * sizeof(float),
                                     cudaMemcpyHostToDevice, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_input_2, h_input_2, n * sizeof(float),
                                     cudaMemcpyHostToDevice, stream));
    
    // 配置kernel启动参数
    dim3 const threads_per_block{1024};  // 每个线程块1024个线程
    dim3 const blocks_per_grid{32};      // 网格中32个线程块
    
    // 启动kernel执行浮点加法
    float_addition<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        d_output, d_input_1, d_input_2, n);
    CHECK_LAST_CUDA_ERROR();
    
    // 异步将结果从设备内存拷贝回主机内存
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_output, d_output, n * sizeof(float),
                                     cudaMemcpyDeviceToHost, stream));
}

// 使用mapped pinned内存启动浮点加法kernel
void launch_float_addition_mapped_pinned_memory(float* d_output,
                                                float* d_input_1,
                                                float* d_input_2, uint32_t n,
                                                cudaStream_t stream)
{
    // 配置kernel启动参数
    dim3 const threads_per_block{1024};  // 每个线程块1024个线程
    dim3 const blocks_per_grid{32};      // 网格中32个线程块
    
    // 直接启动kernel，无需显式内存拷贝（零拷贝）
    float_addition<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        d_output, d_input_1, d_input_2, n);
    CHECK_LAST_CUDA_ERROR();
}

// 初始化主机内存，将所有元素设置为指定值
void initialize_host_memory(float* h_buffer, uint32_t n, float value)
{
    for (int i{0}; i < n; ++i)
    {
        h_buffer[i] = value;
    }
}

// 验证主机内存中的所有元素是否等于期望值
bool verify_host_memory(float* h_buffer, uint32_t n, float value)
{
    for (int i{0}; i < n; ++i)
    {
        if (h_buffer[i] != value)
        {
            return false;
        }
    }
    return true;
}

int main()
{
    // 性能测试参数
    constexpr int const num_repeats{10};   // 重复测试次数
    constexpr int const num_warmups{10};   // 预热次数

    constexpr int const n{1000000};        // 数组大小
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    // 测试数据的初始值
    float const v_input_1{1.0f};
    float const v_input_2{1.0f};
    float const v_output{0.0f};
    float const v_output_reference{v_input_1 + v_input_2};  // 期望的输出值

    // 检查设备是否支持mapped内存
    cudaDeviceProp prop;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, 0));
    if (!prop.canMapHostMemory)
    {
        throw std::runtime_error{"Device does not supported mapped memory."};
    }

    // 声明各种类型的内存指针
    float *h_input_1, *h_input_2, *h_output;    // 普通pinned内存（主机端）
    float *d_input_1, *d_input_2, *d_output;    // 设备内存

    float *a_input_1, *a_input_2, *a_output;    // mapped pinned内存（主机端）
    float *m_input_1, *m_input_2, *m_output;    // mapped pinned内存（设备端指针）

    // 分配普通pinned内存
    CHECK_CUDA_ERROR(cudaMallocHost(&h_input_1, n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMallocHost(&h_input_2, n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMallocHost(&h_output, n * sizeof(float)));

    // 分配设备内存
    CHECK_CUDA_ERROR(cudaMalloc(&d_input_1, n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_input_2, n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, n * sizeof(float)));

    // 分配mapped pinned内存（可被GPU直接访问的主机内存）
    CHECK_CUDA_ERROR(
        cudaHostAlloc(&a_input_1, n * sizeof(float), cudaHostAllocMapped));
    CHECK_CUDA_ERROR(
        cudaHostAlloc(&a_input_2, n * sizeof(float), cudaHostAllocMapped));
    CHECK_CUDA_ERROR(
        cudaHostAlloc(&a_output, n * sizeof(float), cudaHostAllocMapped));

    // 获取mapped pinned内存的设备端指针
    CHECK_CUDA_ERROR(cudaHostGetDevicePointer(&m_input_1, a_input_1, 0));
    CHECK_CUDA_ERROR(cudaHostGetDevicePointer(&m_input_2, a_input_2, 0));
    CHECK_CUDA_ERROR(cudaHostGetDevicePointer(&m_output, a_output, 0));

    // 验证non-mapped pinned内存实现的正确性
    initialize_host_memory(h_input_1, n, v_input_1);
    initialize_host_memory(h_input_2, n, v_input_2);
    initialize_host_memory(h_output, n, v_output);
    launch_float_addition_non_mapped_pinned_memory(
        h_output, h_input_1, h_input_2, d_output, d_input_1, d_input_2, n,
        stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    assert(verify_host_memory(h_output, n, v_output_reference));

    // 验证mapped pinned内存实现的正确性
    initialize_host_memory(a_input_1, n, v_input_1);
    initialize_host_memory(a_input_2, n, v_input_2);
    initialize_host_memory(a_output, n, v_output);
    launch_float_addition_mapped_pinned_memory(m_output, m_input_1, m_input_2,
                                               n, stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    assert(verify_host_memory(a_output, n, v_output_reference));

    // 测量两种方法的延迟性能
    // 绑定non-mapped pinned内存函数
    std::function<void(cudaStream_t)> function_non_mapped_pinned_memory{
        std::bind(launch_float_addition_non_mapped_pinned_memory, h_output,
                  h_input_1, h_input_2, d_output, d_input_1, d_input_2, n,
                  std::placeholders::_1)};
    // 绑定mapped pinned内存函数
    std::function<void(cudaStream_t)> function_mapped_pinned_memory{
        std::bind(launch_float_addition_mapped_pinned_memory, m_output,
                  m_input_1, m_input_2, n, std::placeholders::_1)};
    
    // 测量non-mapped pinned内存的性能
    float const latency_non_mapped_pinned_memory{measure_performance(
        function_non_mapped_pinned_memory, stream, num_repeats, num_warmups)};
    // 测量mapped pinned内存的性能
    float const latency_mapped_pinned_memory{measure_performance(
        function_mapped_pinned_memory, stream, num_repeats, num_warmups)};
    
    // 输出性能测试结果
    std::cout << std::fixed << std::setprecision(3)
              << "CUDA Kernel With Non-Mapped Pinned Memory Latency: "
              << latency_non_mapped_pinned_memory << " ms" << std::endl;
    std::cout << std::fixed << std::setprecision(3)
              << "CUDA Kernel With Mapped Pinned Memory Latency: "
              << latency_mapped_pinned_memory << " ms" << std::endl;

    // 清理所有分配的内存资源
    CHECK_CUDA_ERROR(cudaFree(d_input_1));
    CHECK_CUDA_ERROR(cudaFree(d_input_2));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaFreeHost(h_input_1));
    CHECK_CUDA_ERROR(cudaFreeHost(h_input_2));
    CHECK_CUDA_ERROR(cudaFreeHost(h_output));
    CHECK_CUDA_ERROR(cudaFreeHost(a_input_1));
    CHECK_CUDA_ERROR(cudaFreeHost(a_input_2));
    CHECK_CUDA_ERROR(cudaFreeHost(a_output));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}

```

### 独立GPU

这是在拥有Intel Core i9-9900K CPU和NVIDIA RTX 3090 GPU的桌面系统上的延迟性能分析。

```shell
$ nvcc mapped_memory.cu -o mapped_memory -std=c++14
$ ./mapped_memory
CUDA Kernel With Non-Mapped Pinned Memory Latency: 0.964 ms
CUDA Kernel With Mapped Pinned Memory Latency: 0.631 ms
```

我们可以看到，对于内存受限的kernel，在使用独立GPU、独立主机内存和设备内存的平台上，使用mapped pinned内存比使用non-mapped pinned内存快近30%。

### 集成GPU

这是在NVIDIA Jetson Xavier上的延迟性能分析。

```shell
$ nvcc mapped_memory.cu -o mapped_memory -std=c++14
$ ./mapped_memory
CUDA Kernel With Non-Mapped Pinned Memory Latency: 2.343 ms
CUDA Kernel With Mapped Pinned Memory Latency: 0.431 ms
```

我们可以看到，对于内存受限的kernel，在使用集成GPU和统一内存的平台上，使用mapped pinned内存比使用non-mapped pinned内存快近6倍。这是因为使用mapped内存真正消除了统一内存上主机和设备之间的内存拷贝。

### 注意事项

CUDA零拷贝内存禁用了GPU上的数据缓存，对于计算受限的kernel可能会有性能下降。

### 参考资料

- Function Binding and Performance Measurement(https://leimao.github.io/blog/Function-Binding-Performance-Measurement/)
- NVIDIA CUDA Memory Management(https://developer.ridgerun.com/wiki/index.php/NVIDIA_CUDA_Memory_Management)
- Zero Copy Memory - CUDA Best Practice Guide(https://docs.nvidia.com/cuda/archive/11.7.0/cuda-c-best-practices-guide/index.html#zero-copy)

