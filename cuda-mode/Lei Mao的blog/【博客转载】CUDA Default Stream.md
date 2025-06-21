> 博客来源：https://leimao.github.io/blog/CUDA-Default-Stream/ ，来自Lei Mao，已获得作者转载授权。

# CUDA 默认流

## 引言

CUDA默认流在不同场景下可能有不同的同步行为。有时，即使我们在为不同 kernel分配CUDA流时犯了一些错误，它也能帮助程序正确运行。

在这篇博客文章中，我想介绍两种类型的CUDA默认流：默认传统流和默认每线程流，并讨论它们在不同场景下的同步行为。

## 默认流与非默认阻塞流

在下面的例子中，我使用`cudaStreamCreate`创建了一个非默认阻塞流。对于一系列应该在同一个非默认阻塞CUDA流上按顺序运行的CUDA kernel，我犯了一个错误，意外地为其中一个 kernel使用了默认流。

如果默认流是默认传统流，当在传统流中执行操作（如 kernel启动或`cudaStreamWaitEvent()`）时，传统流首先等待所有阻塞流，然后将操作排队到传统流中，然后所有阻塞流等待传统流。因此，即使我犯了错误，CUDA kernel仍然按顺序运行，应用程序的正确性不受影响。

如果默认流是默认每线程流，它是非阻塞的，不会与其他CUDA流同步。因此，我的错误会导致应用程序运行不正确。

```c++
#include <cassert>
#include <iostream>
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

// CUDA kernel函数：对数组中的每个元素加上指定值
__global__ void add_val_in_place(int32_t* data, int32_t val, uint32_t n)
{
    // 计算当前线程的全局索引
    uint32_t const idx{blockDim.x * blockIdx.x + threadIdx.x};
    // 计算网格步长（总线程数）
    uint32_t const stride{blockDim.x * gridDim.x};
    // 使用网格步长循环处理数组元素，确保所有元素都被处理
    for (uint32_t i{idx}; i < n; i += stride)
    {
        data[i] += val;
    }
}

// 启动CUDA kernel的包装函数
void launch_add_val_in_place(int32_t* data, int32_t val, uint32_t n,
                             cudaStream_t stream)
{
    // 定义每个线程块的线程数
    dim3 const threads_per_block{1024};
    // 定义网格中的线程块数
    dim3 const blocks_per_grid{32};
    // 在指定流上启动kernel
    add_val_in_place<<<blocks_per_grid, threads_per_block, 0, stream>>>(data,
                                                                        val, n);
    // 检查kernel启动是否成功
    CHECK_LAST_CUDA_ERROR();
}

// 检查数组中所有元素是否都等于指定值
bool check_array_value(int32_t const* data, uint32_t n, int32_t val)
{
    for (uint32_t i{0}; i < n; ++i)
    {
        if (data[i] != val)
        {
            return false;
        }
    }
    return true;
}

int main()
{
    // 定义常量：数组大小和要添加的值
    constexpr uint32_t const n{1000000};
    constexpr int32_t const val_1{1};
    constexpr int32_t const val_2{2};
    constexpr int32_t const val_3{3};
    
    // 创建多流应用程序
    cudaStream_t stream_1{0};
    cudaStream_t stream_2{0};
    // stream_1是一个非默认阻塞流
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream_1));

    // 在主机上创建并初始化数组
    std::vector<int32_t> vec(n, 0);
    int32_t* d_data{nullptr};
    // 在设备上分配内存
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, n * sizeof(int32_t)));
    // 将数据从主机复制到设备
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, vec.data(), n * sizeof(int32_t),
                                cudaMemcpyHostToDevice));
    
    // 在同一个CUDA流上按顺序运行一系列CUDA kernel
    launch_add_val_in_place(d_data, val_1, n, stream_1);
    // 第二个kernel启动本应在stream_1上运行
    // 但是实现中有一个错误，导致kernel启动在默认流stream_2上运行
    launch_add_val_in_place(d_data, val_2, n, stream_2);
    launch_add_val_in_place(d_data, val_3, n, stream_1);

    // 等待stream_1上的所有操作完成
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_1));
    // 将结果从设备复制回主机
    CHECK_CUDA_ERROR(cudaMemcpy(vec.data(), d_data, n * sizeof(int32_t),
                                cudaMemcpyDeviceToHost));

    // 检查应用程序的正确性
    // 如果默认流stream_2是传统默认流，结果仍然会是正确的
    assert(check_array_value(vec.data(), n, val_1 + val_2 + val_3));

    // 清理资源
    CHECK_CUDA_ERROR(cudaFree(d_data));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream_1));
}
```

我们在实现中使三个 kernel没有在同一个CUDA流中运行，但结果仍然是正确的。

```shell
$ nvcc add.cu -o add -std=c++14
$ ./add
```

这与运行以下命令相同，因为`--default-stream`的默认值是`legacy`。

```shell
$ nvcc add.cu -o add -std=c++14 --default-stream=legacy
$ ./add
```

根据使用情况，这种错误有时可能会影响应用程序性能。通常可以使用CUDA性能分析软件来识别，比如Nsight Systems(https://leimao.github.io/blog/Docker-Nsight-Systems/)。

但是，如果默认流变成`per-thread`，结果不再正确，因为 kernel启动不再按顺序发出。

```shell
$ nvcc add.cu -o add -std=c++14 --default-stream=per-thread
$ ./add
add: add.cu:98: int main(): Assertion `check_array_value(vec.data(), n, val_1 + val_2 + val_3)' failed.
Aborted (core dumped)
```

## 默认流与非默认非阻塞流

在某些应用程序中，可以使用`cudaStreamCreateWithFlags`创建非默认流，创建的非默认流变为非阻塞的。在这种情况下，默认流，即使是默认传统流，也无法与非默认非阻塞流同步。因此，我的错误会导致应用程序运行不正确，无论非默认流是传统流还是每线程流。

```c++
#include <cassert>
#include <iostream>
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

__global__ void add_val_in_place(int32_t* data, int32_t val, uint32_t n)
{
    uint32_t const idx{blockDim.x * blockIdx.x + threadIdx.x};
    uint32_t const stride{blockDim.x * gridDim.x};
    for (uint32_t i{idx}; i < n; i += stride)
    {
        data[i] += val;
    }
}

void launch_add_val_in_place(int32_t* data, int32_t val, uint32_t n,
                             cudaStream_t stream)
{
    dim3 const threads_per_block{1024};
    dim3 const blocks_per_grid{32};
    add_val_in_place<<<blocks_per_grid, threads_per_block, 0, stream>>>(data,
                                                                        val, n);
    CHECK_LAST_CUDA_ERROR();
}

bool check_array_value(int32_t const* data, uint32_t n, int32_t val)
{
    for (uint32_t i{0}; i < n; ++i)
    {
        if (data[i] != val)
        {
            return false;
        }
    }
    return true;
}
int main()
{
    // 定义常量：数组大小和三个要添加的值
    constexpr uint32_t const n{1000000};
    constexpr int32_t const val_1{1};
    constexpr int32_t const val_2{2};
    constexpr int32_t const val_3{3};
    
    // 创建多流应用程序
    cudaStream_t stream_1{0};  // 非默认流
    cudaStream_t stream_2{0};  // 默认流（值为0表示默认流）
    
    // stream_1 是一个非默认的非阻塞流
    CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&stream_1, cudaStreamNonBlocking));

    // 在主机端创建并初始化数组，所有元素初始化为0
    std::vector<int32_t> vec(n, 0);
    
    // 在设备端分配内存
    int32_t* d_data{nullptr};
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, n * sizeof(int32_t)));
    
    // 将主机数据复制到设备
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, vec.data(), n * sizeof(int32_t),
                                cudaMemcpyHostToDevice));
    
    // 在同一个CUDA流上按顺序运行一系列CUDA kernel
    // 第一个kernel：在stream_1上执行，给数组每个元素加val_1
    launch_add_val_in_place(d_data, val_1, n, stream_1);
    
    // 第二个kernel启动本应该在stream_1上运行
    // 但是，实现中有一个错误，导致kernel启动在默认流stream_2上运行
    // 这里故意使用stream_2来演示默认流的同步行为
    launch_add_val_in_place(d_data, val_2, n, stream_2);
    
    // 第三个kernel：在stream_1上执行，给数组每个元素加val_3
    launch_add_val_in_place(d_data, val_3, n, stream_1);

    // 等待stream_1上的所有操作完成
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_1));
    
    // 将结果从设备复制回主机
    CHECK_CUDA_ERROR(cudaMemcpy(vec.data(), d_data, n * sizeof(int32_t),
                                cudaMemcpyDeviceToHost));

    // 检查应用程序的正确性
    // 如果默认流stream_2是传统默认流，结果仍然会是正确的
    // 因为传统默认流会与其他流同步，确保kernel按顺序执行
    // 如果默认流是每线程流，则可能导致执行顺序错误
    assert(check_array_value(vec.data(), n, val_1 + val_2 + val_3));

    // 清理资源：释放设备内存和销毁流
    CHECK_CUDA_ERROR(cudaFree(d_data));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream_1));
}
```

```shell
$ nvcc add.cu -o add -std=c++14 --default-stream=legacy
$ ./add
add: add.cu:98: int main(): Assertion `check_array_value(vec.data(), n, val_1 + val_2 + val_3)' failed.
Aborted (core dumped)
```

```shell
$ nvcc add.cu -o add -std=c++14 --default-stream=per-thread
$ ./add
add: add.cu:98: int main(): Assertion `check_array_value(vec.data(), n, val_1 + val_2 + val_3)' failed.
Aborted (core dumped)
```

## 参考资料

- Stream synchronization behavior(https://docs.nvidia.com/cuda/archive/11.7.1/cuda-runtime-api/stream-sync-behavior.html)

