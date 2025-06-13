> 博客来源：https://leimao.github.io/blog/CUDA-Shared-Memory-Capacity/ ，来自Lei Mao，已获得作者转载授权。

# CUDA 共享内存容量

## 介绍

CUDA 共享内存是 CUDA  kernel实现和优化中极其强大的功能。由于 CUDA 共享内存位于芯片上，其内存带宽比位于芯片外的全局内存要大得多。因此，通过在共享内存上缓存内存访问来优化 CUDA  kernel可以显著提高某些操作的性能，特别是对于那些内存受限的操作。

然而，CUDA 共享内存对每个线程块都有大小限制，默认为 48 KB。有时，我们希望在实现中使用稍多一些的共享内存。在这篇博客文章中，我想讨论如何分配静态共享内存、动态共享内存，以及如何请求超过 48 KB 的动态共享内存。

## 模板 kernel

我们实现了一个模板 kernel来演示 CUDA 共享内存的分配。模板在数学上几乎等价于卷积的特殊情况，其权重恰好为 1，使用有效填充。

例如，给定一个一维数组 ${1, 1, 1, 1, 1, 1, 1}$
和一个半径为2的模板 kernel，我们将得到输出一维数组 ${1, 1, 5, 5, 5, 1, 1}$。

模板操作会从输入张量中产生许多冗余的内存读取，因此是一个内存受限的操作。如果内存读取未被缓存且程序从全局内存读取，性能将会很差。因此，我们将利用片上共享内存来缓存内存读取并提高性能。

### 静态共享内存

在这个实现中，我们分配了静态共享内存，其大小必须在编译时已知。该实现还支持任意的"有效"数组大小、半径和 CUDA 线程块大小。还要注意，当我们实现 kernel时，必须特别注意半径大于 CUDA 线程块大小且"有效"数组大小不能被 CUDA 线程块大小整除的情况，因为正确实现它们并不容易。

```c++
#include <cassert>
#include <iostream>
#include <vector>

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

template <int BLOCK_SIZE = 1024, int RADIUS = 5>
__global__ void stencil_1d_kernel(int const* d_in, int* d_out,
                                  int valid_array_size)
{
    __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];

    // This has to be int because we will use negative indices.
    int const gindex{static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x)};
    int const lindex{static_cast<int>(threadIdx.x) + RADIUS};

    int const valid_block_size{
        min(BLOCK_SIZE,
            valid_array_size - static_cast<int>(blockIdx.x * blockDim.x))};

    // Read input elements into shared memory
    if (gindex < valid_array_size)
    {
        temp[lindex] = d_in[gindex];
        if (RADIUS <= valid_block_size)
        {
            if (threadIdx.x < RADIUS)
            {
                temp[lindex - RADIUS] = d_in[gindex - RADIUS];
                temp[lindex + valid_block_size] =
                    d_in[gindex + valid_block_size];
            }
        }
        else
        {
            for (int i{0}; i < RADIUS; i += valid_block_size)
            {
                // Some threads might have to do one more job than other
                // threads.
                if (lindex - RADIUS + i < RADIUS)
                {
                    temp[lindex - RADIUS + i] = d_in[gindex - RADIUS + i];
                    temp[lindex + valid_block_size + i] =
                        d_in[gindex + valid_block_size + i];
                }
            }
        }
    }
    // Synchronize (ensure all the data is available)
    __syncthreads();

    if (gindex >= valid_array_size)
    {
        return;
    }

    // Apply the stencil
    int result{0};
    for (int offset{-RADIUS}; offset <= RADIUS; offset++)
    {
        result += temp[lindex + offset];
    }

    // Store the result
    d_out[gindex] = result;
}

void stencil_1d_cpu(int const* h_in, int* h_out, int radius,
                    int valid_array_size)
{
    for (int i{0}; i < valid_array_size; ++i)
    {
        int result{0};
        for (int offset{-radius}; offset <= radius; offset++)
        {
            result += h_in[i + offset];
        }
        h_out[i] = result;
    }
}

int main(int argc, char** argv)
{
    constexpr int const valid_array_size{1024 * 100 + 1};
    constexpr int const block_size{1024};
    constexpr int const grid_size{(valid_array_size + block_size - 1) /
                                  block_size};
    constexpr int const radius{1025};

    int const array_size{valid_array_size + 2 * radius};
    std::vector<int> const h_in(array_size, 1);
    std::vector<int> h_out{h_in};
    std::vector<int> h_out_reference{h_in};

    stencil_1d_cpu(h_in.data() + radius, h_out_reference.data() + radius,
                   radius, valid_array_size);

    int* d_in;
    int* d_out;

    CHECK_CUDA_ERROR(cudaMalloc(&d_in, array_size * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_out, array_size * sizeof(int)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_in, h_in.data(), array_size * sizeof(int),
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_out, h_out.data(), array_size * sizeof(int),
                                cudaMemcpyHostToDevice));

    stencil_1d_kernel<block_size, radius><<<grid_size, block_size>>>(
        d_in + radius, d_out + radius, valid_array_size);
    CHECK_LAST_CUDA_ERROR();

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(h_out.data(), d_out, array_size * sizeof(int),
                                cudaMemcpyDeviceToHost));

    for (int i{0}; i < h_out_reference.size(); ++i)
    {
        assert(h_out[i] == h_out_reference[i]);
    }

    CHECK_CUDA_ERROR(cudaFree(d_in));
    CHECK_CUDA_ERROR(cudaFree(d_out));
}
```

```shell
$ nvcc stencil_static_shared_memory.cu -o stencil_static_shared_memory
$ ./stencil_static_shared_memory
```

如果我们将 `radius` 从 `1025` 增加到一些更大的值，如 `6000`，我们将得到以下编译错误。

```shell
$ nvcc stencil_static_shared_memory.cu -o stencil_static_shared_memory
ptxas error   : Entry function '_Z17stencil_1d_kernelILi1024ELi6000EEvPKiPii' uses too much shared data (0xcb80 bytes, 0xc000 max)
```

这是因为用户只能分配最多 48 KB 的 CUDA 静态共享内存。在我们的用例中，`BLOCK_SIZE + 2 * RADIUS = 1024 + 2 × 6000 = 13024`，一个 int 的大小是 4 字节，因此，所需的共享内存是 `13024 × 4/1024 = 50.875 KB`，这大于我们可以拥有的最大静态共享内存。

## 动态共享内存

要使用大于 48 KB 的共享内存，我们必须使用动态共享内存，这是架构特定的。具体来说，除了在 CUDA 启动的 `<<<...>>>` 第三个参数中指定我们想要请求的动态共享内存大小外，还必须调用 CUDA Runtime API `cudaFuncSetAttribute`，我们应该始终检查其返回值，因为它可能在某些架构上的运行时失败。

平台 GPU 是 NVIDIA RTX 2080TI。根据 CUDA C 编程指南(https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x)，计算能力 7.x 设备允许单个线程块在图灵架构上动态分配最多 64 KB 的共享内存。因此我们可以在 NVIDIA RTX 2080TI 上运行半径为 `6000` 的模板程序。

这个使用动态共享内存的实现与使用静态共享内存的实现几乎相同。

```c++
#include <cassert>
#include <iostream>
#include <vector>

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

template <int BLOCK_SIZE = 1024, int RADIUS = 5>
__global__ void stencil_1d_kernel(int const* d_in, int* d_out,
                                  int valid_array_size)
{
    extern __shared__ int temp[];

    // This has to be int because we will use negative indices.
    int const gindex{static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x)};
    int const lindex{static_cast<int>(threadIdx.x) + RADIUS};

    int const valid_block_size{
        min(BLOCK_SIZE,
            valid_array_size - static_cast<int>(blockIdx.x * blockDim.x))};

    // Read input elements into shared memory
    if (gindex < valid_array_size)
    {
        temp[lindex] = d_in[gindex];
        if (RADIUS <= valid_block_size)
        {
            if (threadIdx.x < RADIUS)
            {
                temp[lindex - RADIUS] = d_in[gindex - RADIUS];
                temp[lindex + valid_block_size] =
                    d_in[gindex + valid_block_size];
            }
        }
        else
        {
            for (int i{0}; i < RADIUS; i += valid_block_size)
            {
                // Some threads might have to do one more job than other
                // threads.
                if (lindex - RADIUS + i < RADIUS)
                {
                    temp[lindex - RADIUS + i] = d_in[gindex - RADIUS + i];
                    temp[lindex + valid_block_size + i] =
                        d_in[gindex + valid_block_size + i];
                }
            }
        }
    }
    // Synchronize (ensure all the data is available)
    __syncthreads();

    if (gindex >= valid_array_size)
    {
        return;
    }

    // Apply the stencil
    int result{0};
    for (int offset{-RADIUS}; offset <= RADIUS; offset++)
    {
        result += temp[lindex + offset];
    }

    // Store the result
    d_out[gindex] = result;
}

void stencil_1d_cpu(int const* h_in, int* h_out, int radius,
                    int valid_array_size)
{
    for (int i{0}; i < valid_array_size; ++i)
    {
        int result{0};
        for (int offset{-radius}; offset <= radius; offset++)
        {
            result += h_in[i + offset];
        }
        h_out[i] = result;
    }
}

int main(int argc, char** argv)
{
    constexpr int const valid_array_size{1024 * 100 + 1};
    constexpr int const block_size{1024};
    constexpr int const grid_size{(valid_array_size + block_size - 1) /
                                  block_size};
    constexpr int const radius{6000};

    int const array_size{valid_array_size + 2 * radius};
    std::vector<int> const h_in(array_size, 1);
    std::vector<int> h_out{h_in};
    std::vector<int> h_out_reference{h_in};

    stencil_1d_cpu(h_in.data() + radius, h_out_reference.data() + radius,
                   radius, valid_array_size);

    int* d_in;
    int* d_out;

    CHECK_CUDA_ERROR(cudaMalloc(&d_in, array_size * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_out, array_size * sizeof(int)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_in, h_in.data(), array_size * sizeof(int),
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_out, h_out.data(), array_size * sizeof(int),
                                cudaMemcpyHostToDevice));

    int const shared_memory_bytes{(block_size + radius * 2) * sizeof(int)};
    CHECK_CUDA_ERROR(cudaFuncSetAttribute(
        stencil_1d_kernel<block_size, radius>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_bytes));
    stencil_1d_kernel<block_size, radius>
        <<<grid_size, block_size, shared_memory_bytes>>>(
            d_in + radius, d_out + radius, valid_array_size);
    CHECK_LAST_CUDA_ERROR();

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(h_out.data(), d_out, array_size * sizeof(int),
                                cudaMemcpyDeviceToHost));

    for (int i{0}; i < h_out_reference.size(); ++i)
    {
        assert(h_out[i] == h_out_reference[i]);
    }

    CHECK_CUDA_ERROR(cudaFree(d_in));
    CHECK_CUDA_ERROR(cudaFree(d_out));
}
```

```shell
$ nvcc stencil_dynamic_shared_memory.cu -o stencil_dynamic_shared_memory --gpu-architecture=compute_75 --gpu-code=sm_75
$ ./stencil_dynamic_shared_memory
```

## 结论

大容量共享内存只能为动态共享内存分配的原因是，并非所有 GPU 架构都能支持大于 48 KB 的特定大小的共享内存。如果允许大于 48 KB 的静态共享内存，CUDA 程序将会编译通过但在某些特定 GPU 架构上运行失败，这是不期望的。因此，要使用大于 48 KB 的共享内存，必须在运行时通过动态共享内存来请求。如果 GPU 架构不支持特定大小的共享内存，将返回 CUDA 运行时错误。

## 参考文献

- Shared Memory - CUDA C Programming Guide(https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x)
- CUDA Shared Memory(https://leimao.github.io/downloads/blog/2022-07-04-CUDA-Shared-Memory-Capacity/02-CUDA-Shared-Memory.pdf)

