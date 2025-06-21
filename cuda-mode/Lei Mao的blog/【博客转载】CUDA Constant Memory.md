> 博客来源：https://leimao.github.io/blog/CUDA-Constant-Memory/ ，来自Lei Mao，已获得作者转载授权。

# CUDA 常量内存

## 简介

CUDA 常量内存是设备上的一个特殊内存空间。它被缓存且只读。

使用常量内存时有一些注意事项。在这篇文章中，我们将讨论常量内存的使用方法和注意事项。

## 常量内存

设备上总共有 64 KB 的常量内存。常量内存空间是被缓存的。因此，从常量内存读取只在缓存未命中时需要从设备内存读取一次；否则，只需要从常量缓存读取一次。在一个 warp 内的线程对不同地址的访问是串行化的，因此成本与 warp 内所有线程读取的唯一地址数量成线性关系。因此，当同一 warp 中的线程只访问少数几个不同位置时，常量缓存效果最佳。如果一个 warp 的所有线程访问同一位置，那么常量内存可以和寄存器访问一样快。

## 常量内存使用和性能

在下面的示例中，我们对数组执行加法运算。其中一个常量输入数组存储在全局内存中，另一个常量输入数组存储在全局内存或常量内存中。我们比较在不同访问模式下访问常量内存和全局内存的性能。

```c++
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

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

template <class T>
float measure_performance(std::function<T(cudaStream_t)> bound_function,
                          cudaStream_t stream, unsigned int num_repeats = 100,
                          unsigned int num_warmups = 100)
{
    cudaEvent_t start, stop;
    float time;

    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    for (unsigned int i{0}; i < num_warmups; ++i)
    {
        bound_function(stream);
    }

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    for (unsigned int i{0}; i < num_repeats; ++i)
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

// 使用所有的常量内存空间
// 常量内存大小为64KB，除以int的大小得到可存储的int数量
constexpr unsigned int N{64U * 1024U / sizeof(int)};
// 声明常量内存数组，存储在GPU的常量内存中
__constant__ int const_values[N];

// 用于生成伪随机访问模式的魔数
constexpr unsigned int magic_number{1357U};

// 定义访问模式的枚举类型
enum struct AccessPattern
{
    OneAccessPerBlock,    // 每个块一次访问
    OneAccessPerWarp,     // 每个warp一次访问
    OneAccessPerThread,   // 每个线程一次访问
    PseudoRandom         // 伪随机访问
};

// CPU版本的常量加法函数，用于生成参考结果
void add_constant_cpu(int* sums, int const* inputs, int const* values,
                      unsigned int num_sums, unsigned int num_values,
                      unsigned int block_size, AccessPattern access_pattern)
{
    // 遍历所有需要计算的和
    for (unsigned int i{0U}; i < num_sums; ++i)
    {
        // 计算当前元素所属的块ID
        unsigned int const block_id{i / block_size};
        // 计算当前元素在块内的线程ID
        unsigned int const thread_id{i % block_size};
        // 计算当前线程所属的warp ID（每个warp有32个线程）
        unsigned int const warp_id{thread_id / 32U};
        unsigned int index{0U};

        // 根据访问模式确定要访问的常量数组索引
        switch (access_pattern)
        {
            case AccessPattern::OneAccessPerBlock:
                // 每个块访问同一个常量值
                index = block_id % num_values;
                break;
            case AccessPattern::OneAccessPerWarp:
                // 每个warp访问同一个常量值
                index = warp_id % num_values;
                break;
            case AccessPattern::OneAccessPerThread:
                // 每个线程访问不同的常量值
                index = thread_id % num_values;
                break;
            case AccessPattern::PseudoRandom:
                // 使用魔数生成伪随机访问模式
                index = (thread_id * magic_number) % num_values;
                break;
        }

        // 执行加法运算：输入值 + 常量值
        sums[i] = inputs[i] + values[index];
    }
}

// 使用全局内存的CUDA kernel
__global__ void add_constant_global_memory(
    int* sums, int const* inputs, int const* values, unsigned int num_sums,
    unsigned int num_values,
    AccessPattern access_pattern = AccessPattern::OneAccessPerBlock)
{
    // 计算当前线程的全局索引
    unsigned int const i{blockIdx.x * blockDim.x + threadIdx.x};
    // 获取块ID
    unsigned int const block_id{blockIdx.x};
    // 获取块内线程ID
    unsigned int const thread_id{threadIdx.x};
    // 计算warp ID
    unsigned int const warp_id{threadIdx.x / warpSize};
    unsigned int index{0U};

    // 根据访问模式确定要访问的全局内存索引
    switch (access_pattern)
    {
        case AccessPattern::OneAccessPerBlock:
            // 每个块访问同一个全局内存位置
            index = block_id % num_values;
            break;
        case AccessPattern::OneAccessPerWarp:
            // 每个warp访问同一个全局内存位置
            index = warp_id % num_values;
            break;
        case AccessPattern::OneAccessPerThread:
            // 每个线程访问不同的全局内存位置
            index = thread_id % num_values;
            break;
        case AccessPattern::PseudoRandom:
            // 使用魔数生成伪随机访问模式
            index = (thread_id * magic_number) % num_values;
            break;
    }

    // 边界检查，确保不越界
    if (i < num_sums)
    {
        // 从全局内存读取常量值并执行加法运算
        sums[i] = inputs[i] + values[index];
    }
}

// 启动使用全局内存的kernel的包装函数
void launch_add_constant_global_memory(int* sums, int const* inputs,
                                       int const* values, unsigned int num_sums,
                                       unsigned int num_values,
                                       unsigned int block_size,
                                       AccessPattern access_pattern,
                                       cudaStream_t stream)
{
    // 计算网格大小，确保能处理所有元素
    add_constant_global_memory<<<(num_sums + block_size - 1) / block_size,
                                 block_size, 0, stream>>>(
        sums, inputs, values, num_sums, num_values, access_pattern);
    // 检查kernel启动是否成功
    CHECK_LAST_CUDA_ERROR();
}

// 使用常量内存的CUDA kernel
__global__ void add_constant_constant_memory(int* sums, int const* inputs,
                                             unsigned int num_sums,
                                             AccessPattern access_pattern)
{
    // 计算当前线程的全局索引
    unsigned int const i{blockIdx.x * blockDim.x + threadIdx.x};
    // 获取块ID
    unsigned int const block_id{blockIdx.x};
    // 获取块内线程ID
    unsigned int const thread_id{threadIdx.x};
    // 计算warp ID
    unsigned int const warp_id{threadIdx.x / warpSize};
    unsigned int index{0U};

    // 根据访问模式确定要访问的常量内存索引
    switch (access_pattern)
    {
        case AccessPattern::OneAccessPerBlock:
            // 每个块访问同一个常量内存位置
            index = block_id % N;
            break;
        case AccessPattern::OneAccessPerWarp:
            // 每个warp访问同一个常量内存位置
            index = warp_id % N;
            break;
        case AccessPattern::OneAccessPerThread:
            // 每个线程访问不同的常量内存位置
            index = thread_id % N;
            break;
        case AccessPattern::PseudoRandom:
            // 使用魔数生成伪随机访问模式
            index = (thread_id * magic_number) % N;
            break;
    }

    // 边界检查，确保不越界
    if (i < num_sums)
    {
        // 从常量内存读取常量值并执行加法运算
        sums[i] = inputs[i] + const_values[index];
    }
}

// 启动使用常量内存的kernel的包装函数
void launch_add_constant_constant_memory(int* sums, int const* inputs,
                                         unsigned int num_sums,
                                         unsigned int block_size,
                                         AccessPattern access_pattern,
                                         cudaStream_t stream)
{
    // 计算网格大小，确保能处理所有元素
    add_constant_constant_memory<<<(num_sums + block_size - 1) / block_size,
                                   block_size, 0, stream>>>(
        sums, inputs, num_sums, access_pattern);
    // 检查kernel启动是否成功
    CHECK_LAST_CUDA_ERROR();
}

// 解析命令行参数的函数
void parse_args(int argc, char** argv, AccessPattern& access_pattern,
                unsigned int& block_size, unsigned int& num_sums)
{
    // 检查参数数量是否足够
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <access pattern> <block size> <number of sums>"
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // 解析访问模式参数
    std::string const access_pattern_str{argv[1]};
    if (access_pattern_str == "one_access_per_block")
    {
        access_pattern = AccessPattern::OneAccessPerBlock;
    }
    else if (access_pattern_str == "one_access_per_warp")
    {
        access_pattern = AccessPattern::OneAccessPerWarp;
    }
    else if (access_pattern_str == "one_access_per_thread")
    {
        access_pattern = AccessPattern::OneAccessPerThread;
    }
    else if (access_pattern_str == "pseudo_random")
    {
        access_pattern = AccessPattern::PseudoRandom;
    }
    else
    {
        std::cerr << "Invalid access pattern: " << access_pattern_str
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // 解析块大小和求和数量参数
    block_size = std::stoi(argv[2]);
    num_sums = std::stoi(argv[3]);
}

int main(int argc, char** argv)
{
    // 定义性能测试的预热次数和重复次数
    constexpr unsigned int num_warmups{100U};
    constexpr unsigned int num_repeats{100U};

    // 设置默认参数值
    AccessPattern access_pattern{AccessPattern::OneAccessPerBlock};
    unsigned int block_size{1024U};
    unsigned int num_sums{12800000U};
    // 从命令行修改访问模式、块大小和求和数量
    parse_args(argc, argv, access_pattern, block_size, num_sums);

    // 创建CUDA流
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    // 在主机内存中初始化常量值数组
    int h_values[N];
    // 初始化主机内存中的常量值
    for (unsigned int i{0U}; i < N; ++i)
    {
        h_values[i] = i;
    }
    // 在全局内存中初始化常量值
    int* d_values;
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_values, N * sizeof(int), stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_values, h_values, N * sizeof(int),
                                     cudaMemcpyHostToDevice, stream));
    // 在常量内存中初始化常量值
    CHECK_CUDA_ERROR(cudaMemcpyToSymbolAsync(const_values, h_values,
                                             N * sizeof(int), 0,
                                             cudaMemcpyHostToDevice, stream));

    // 创建输入数组并初始化为0
    std::vector<int> inputs(num_sums, 0);
    int* h_inputs{inputs.data()};
    // 为常量内存测试分配设备输入数组
    int* d_inputs_for_constant;
    // 为全局内存测试分配设备输入数组
    int* d_inputs_for_global;
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_inputs_for_constant,
                                     num_sums * sizeof(int), stream));
    CHECK_CUDA_ERROR(
        cudaMallocAsync(&d_inputs_for_global, num_sums * sizeof(int), stream));
    // 将输入数据复制到设备
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_inputs_for_constant, h_inputs,
                                     num_sums * sizeof(int),
                                     cudaMemcpyHostToDevice, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_inputs_for_global, h_inputs,
                                     num_sums * sizeof(int),
                                     cudaMemcpyHostToDevice, stream));

    // 创建结果数组
    std::vector<int> reference_sums(num_sums, 0);      // CPU参考结果
    std::vector<int> sums_from_constant(num_sums, 1);  // 常量内存结果
    std::vector<int> sums_from_global(num_sums, 2);    // 全局内存结果

    // 获取主机数组指针
    int* h_reference_sums{reference_sums.data()};
    int* h_sums_from_constant{sums_from_constant.data()};
    int* h_sums_from_global{sums_from_global.data()};

    // 分配设备结果数组
    int* d_sums_from_constant;
    int* d_sums_from_global;
    CHECK_CUDA_ERROR(
        cudaMallocAsync(&d_sums_from_constant, num_sums * sizeof(int), stream));
    CHECK_CUDA_ERROR(
        cudaMallocAsync(&d_sums_from_global, num_sums * sizeof(int), stream));

    // 同步流，确保所有异步操作完成
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // 在CPU上计算参考结果
    add_constant_cpu(h_reference_sums, h_inputs, h_values, num_sums, N,
                     block_size, access_pattern);
    // 在GPU上使用全局内存计算结果
    launch_add_constant_global_memory(d_sums_from_global, d_inputs_for_global,
                                      d_values, num_sums, N, block_size,
                                      access_pattern, stream);
    // 在GPU上使用常量内存计算结果
    launch_add_constant_constant_memory(d_sums_from_constant,
                                        d_inputs_for_constant, num_sums,
                                        block_size, access_pattern, stream);

    // 将结果从设备复制到主机
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_sums_from_constant, d_sums_from_constant,
                                     num_sums * sizeof(int),
                                     cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_sums_from_global, d_sums_from_global,
                                     num_sums * sizeof(int),
                                     cudaMemcpyDeviceToHost, stream));

    // 同步流，确保所有数据传输完成
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // 验证结果的正确性
    for (unsigned int i{0U}; i < num_sums; ++i)
    {
        // 检查常量内存结果是否与参考结果一致
        if (h_reference_sums[i] != h_sums_from_constant[i])
        {
            std::cerr << "Error at index " << i << " for constant memory."
                      << std::endl;
            std::exit(EXIT_FAILURE);
        }
        // 检查全局内存结果是否与参考结果一致
        if (h_reference_sums[i] != h_sums_from_global[i])
        {
            std::cerr << "Error at index " << i << " for global memory."
                      << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    // 测量性能
    // 创建常量内存kernel的绑定函数
    std::function<void(cudaStream_t)> bound_function_constant_memory{
        std::bind(launch_add_constant_constant_memory, d_sums_from_constant,
                  d_inputs_for_constant, num_sums, block_size, access_pattern,
                  std::placeholders::_1)};
    // 创建全局内存kernel的绑定函数
    std::function<void(cudaStream_t)> bound_function_global_memory{
        std::bind(launch_add_constant_global_memory, d_sums_from_global,
                  d_inputs_for_global, d_values, num_sums, N, block_size,
                  access_pattern, std::placeholders::_1)};
    // 测量常量内存的性能
    float const latency_constant_memory{measure_performance(
        bound_function_constant_memory, stream, num_repeats, num_warmups)};
    // 测量全局内存的性能
    float const latency_global_memory{measure_performance(
        bound_function_global_memory, stream, num_repeats, num_warmups)};
    // 输出性能测试结果
    std::cout << "Latency for Add using constant memory: "
              << latency_constant_memory << " ms" << std::endl;
    std::cout << "Latency for Add using global memory: "
              << latency_global_memory << " ms" << std::endl;

    // 清理资源
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    CHECK_CUDA_ERROR(cudaFree(d_values));
    CHECK_CUDA_ERROR(cudaFree(d_inputs_for_constant));
    CHECK_CUDA_ERROR(cudaFree(d_inputs_for_global));
    CHECK_CUDA_ERROR(cudaFree(d_sums_from_constant));
    CHECK_CUDA_ERROR(cudaFree(d_sums_from_global));

    return 0;
}
```

该程序在 NVIDIA RTX 3090 GPU 上编译和执行。

```shell
$ nvcc add_constant.cu -o add_constant
```

如果我们使用每个块 1024 个线程执行 12800000 次加法运算。

```shell
$ ./add_constant one_access_per_block 1024 12800000
Latency for Add using constant memory: 0.151798 ms
Latency for Add using global memory: 0.171404 ms
$ ./add_constant one_access_per_warp 1024 12800000
Latency for Add using constant memory: 0.164012 ms
Latency for Add using global memory: 0.189501 ms
$ ./add_constant one_access_per_thread 1024 12800000
Latency for Add using constant memory: 0.281967 ms
Latency for Add using global memory: 0.164649 ms
$ ./add_constant pseudo_random 1024 12800000
Latency for Add using constant memory: 1.2925 ms
Latency for Add using global memory: 0.159621 ms
```

如果我们使用每个块 1024 个线程执行 128000 次加法运算。

```shell
$ ./add_constant one_access_per_block 1024 128000
Latency for Add using constant memory: 0.00289792 ms
Latency for Add using global memory: 0.00323584 ms
$ ./add_constant one_access_per_warp 1024 128000
Latency for Add using constant memory: 0.00315392 ms
Latency for Add using global memory: 0.00359392 ms
$ ./add_constant one_access_per_thread 1024 128000
Latency for Add using constant memory: 0.00596992 ms
Latency for Add using global memory: 0.00383264 ms
$ ./add_constant pseudo_random 1024 128000
Latency for Add using constant memory: 0.0215347 ms
Latency for Add using global memory: 0.00482304 ms
```

在这两种情况下，我们可以看到，如果是每个块一次访问或每个 warp 一次访问，访问常量内存比访问全局内存快约 10%。如果是每个线程一次访问，那么访问常量内存比访问全局内存慢约 70%。如果是伪随机访问，那么访问常量内存比访问全局内存慢约 800%。

## 结论

要使用常量内存，了解访问模式是很重要的。如果访问模式是每个块一次访问或每个 warp 一次访问（通常用于广播），那么常量内存是一个不错的选择。如果访问模式是每个线程一次访问或者甚至是伪随机访问，那么常量内存是一个非常糟糕的选择。

## 参考资料

- Device Memory Spaces(https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#device-memory-spaces)
- Constant Memory(https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#constant-memory)
- Constant Specifier(https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#constant)

