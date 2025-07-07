> 博客来源：https://leimao.github.io/blog/CUDA-Local-Memory/ ，来自Lei Mao，已获得作者转载授权。后续会转载一些Lei Mao的CUDA相关Blog，也是一个完整的专栏，Blog会从稍早一些的CUDA架构到当前最新的CUDA架构，也会包含实用工程技巧，底层指令分析，Cutlass分析等等多个课题，是一个时间线十分明确的专栏。

# CUDA 本地内存

## 介绍

在CUDA编程中，本地内存（Local Memory）是执行线程的私有存储空间，对该线程外部不可见。本地内存空间位于设备内存中，因此本地内存访问具有与全局内存访问相同的高延迟和低带宽特性，并且受到相同的内存合并（memory coalescing）要求的限制。

> **注释：** 本地内存虽然名为"本地"，但实际上是位于设备内存（显存）中的，这是一个容易混淆的概念。它的访问速度实际上和全局内存一样慢。

在没有使用`__device__`、`__shared__`和`__constant__`内存空间说明符声明的自动变量，可能被编译器放置在寄存器或本地内存中。如果满足以下条件之一，变量很可能被放置在本地内存中：

- 编译器无法确定使用常量索引访问的数组
- 占用过多寄存器空间的大型结构体或数组
- 当 kernel 使用的寄存器数量超过可用数量时的任何变量（这也被称为寄存器溢出）

> **注释：** 第二点和第三点很容易理解，但第一点比较复杂。它暗示即使是很小的数组，如果索引不是编译时常量，也可能被放置在本地内存中，而大多数情况下我们希望这些小数组被放置在寄存器中以获得更好的性能。

理解第二点和第三点非常直观。然而，第一点有些复杂，因为它暗示即使对于非常小的数组，如果编译器无法确定它们是用常量索引访问的，也可能被放置在本地内存而不是寄存器中，而大多数时候我们希望这些小数组被放置在寄存器中以获得更好的性能。

在这篇博客文章中，我想展示一个编译器如何决定将数组放置在本地内存而不是寄存器中的例子，并讨论用户可以遵循的一般规则来避免小数组被放置在本地内存中。

## CUDA 本地内存示例

在以下示例中，我创建了两个CUDA kernel ，用于计算给定固定`window`大小的输入数组的滑动平均值。两个 kernel 都声明了一个大小在编译时已知的本地数组window。两个 kernel 的实现几乎完全相同，除了第一个 kernel 使用直接索引访问window数组，而第二个 kernel 使用看起来不太直观的索引。

> **注释：** 这个例子很好地展示了编译器如何根据数组索引的复杂性来决定变量的存储位置。

```c++
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

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

// 第一个 kernel ：使用简单索引，数组会被放置在寄存器中
template <int WindowSize>
__global__ void running_mean_register_array(float const* input, float* output,
                                            int n)
{
    float window[WindowSize];  // 这个数组会被放置在寄存器中
    int const thread_idx{
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)};
    int const stride{static_cast<int>(blockDim.x * gridDim.x)};
    for (int i{thread_idx}; i < n; i += stride)
    {
        // 将数据读入窗口
        for (int j{0}; j < WindowSize; ++j)
        {
            int const idx{i - WindowSize / 2 + j};
            window[j] = (idx < 0 || idx >= n) ? 0 : input[idx];
        }
        // 从窗口计算平均值
        float sum{0};
        for (int j{0}; j < WindowSize; ++j)
        {
            // 这里使用简单的常量索引 j，编译器可以轻松处理
            sum += window[j];
        }
        float const mean{sum / WindowSize};
        // 将平均值写入输出
        output[i] = mean;
    }
}

// 第二个 kernel ：使用复杂索引，数组会被放置在本地内存中
template <int WindowSize>
__global__ void running_mean_local_memory_array(float const* input,
                                                float* output, int n)
{
    float window[WindowSize];  // 这个数组会被放置在本地内存中
    int const thread_idx{
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)};
    int const stride{static_cast<int>(blockDim.x * gridDim.x)};
    for (int i{thread_idx}; i < n; i += stride)
    {
        // 将数据读入窗口
        for (int j{0}; j < WindowSize; ++j)
        {
            int const idx{i - WindowSize / 2 + j};
            window[j] = (idx < 0 || idx >= n) ? 0 : input[idx];
        }
        // 从窗口计算平均值
        float sum{0};
        for (int j{0}; j < WindowSize; ++j)
        {
            // 这个访问窗口数组的索引无法在编译时被编译器解析，
            // 即使这样的索引不会影响 kernel 的正确性。
            // 后果是编译器会将窗口数组放置在本地内存而不是寄存器文件中。
            int const idx{(j + n) % WindowSize};  // 复杂的索引表达式
            sum += window[idx];
        }
        float const mean{sum / WindowSize};
        // 将平均值写入输出
        output[i] = mean;
    }
}

template <int WindowSize>
cudaError_t launch_running_mean_register_array(float const* input,
                                               float* output, int n,
                                               cudaStream_t stream)
{
    dim3 const block_size{256, 1, 1};
    dim3 const grid_size{(n + block_size.x - 1) / block_size.x, 1, 1};
    running_mean_register_array<WindowSize>
        <<<grid_size, block_size, 0, stream>>>(input, output, n);
    return cudaGetLastError();
}

template <int WindowSize>
cudaError_t launch_running_mean_local_memory_array(float const* input,
                                                   float* output, int n,
                                                   cudaStream_t stream)
{
    dim3 const block_size{256, 1, 1};
    dim3 const grid_size{(n + block_size.x - 1) / block_size.x, 1, 1};
    running_mean_local_memory_array<WindowSize>
        <<<grid_size, block_size, 0, stream>>>(input, output, n);
    return cudaGetLastError();
}

// 验证给定窗口大小和启动函数的 kernel 的正确性
template <int WindowSize>
void verify_running_mean(int n, cudaError_t (*launch_func)(float const*, float*,
                                                           int, cudaStream_t))
{
    std::vector<float> h_input_vec(n, 0.f);
    std::vector<float> h_output_vec(n, 1.f);
    std::vector<float> h_output_vec_ref(n, 2.f);
    // 用值填充输入向量
    for (int i{0}; i < n; ++i)
    {
        h_input_vec[i] = static_cast<float>(i);
    }
    // 计算参考输出向量
    for (int i{0}; i < n; ++i)
    {
        float sum{0};
        for (int j{0}; j < WindowSize; ++j)
        {
            int const idx{i - WindowSize / 2 + j};
            float const val{(idx < 0 || idx >= n) ? 0 : h_input_vec[idx]};
            sum += val;
        }
        h_output_vec_ref[i] = sum / WindowSize;
    }
    // 分配设备内存
    float* d_input;
    float* d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, n * sizeof(float)));
    // 将数据复制到设备
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input_vec.data(), n * sizeof(float),
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_output, h_output_vec.data(),
                                n * sizeof(float), cudaMemcpyHostToDevice));
    // 启动 kernel 
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    CHECK_CUDA_ERROR(launch_func(d_input, d_output, n, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    // 将结果复制回主机
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_vec.data(), d_output,
                                n * sizeof(float), cudaMemcpyDeviceToHost));
    // 检查结果
    for (int i{0}; i < n; ++i)
    {
        if (h_output_vec.at(i) != h_output_vec_ref.at(i))
        {
            std::cerr << "Mismatch at index " << i << ": " << h_output_vec.at(i)
                      << " != " << h_output_vec_ref.at(i) << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
    // 释放设备内存
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}

int main()
{
    // 尝试从小到大的不同窗口大小
    constexpr int WindowSize{32};
    int const n{8192};
    verify_running_mean<WindowSize>(
        n, launch_running_mean_register_array<WindowSize>);
    verify_running_mean<WindowSize>(
        n, launch_running_mean_local_memory_array<WindowSize>);
    return 0;
}
```

要构建和运行示例，请运行以下命令。运行示例时应该不会遇到错误消息。

```shell
$ nvcc cuda_local_memory.cu -o cuda_local_memory
$ ./cuda_local_memory
```

要检查本地数组`window`是否被放置在寄存器或本地内存中，我们可以将代码编译为PTX并检查PTX代码。

> **注释：** PTX（Parallel Thread Execution）是NVIDIA GPU的中间表示形式，类似于汇编语言。通过查看PTX代码，我们可以了解编译器如何处理我们的CUDA代码。

要将代码编译为PTX，请运行以下命令：

```shell
$ nvcc --ptx cuda_local_memory.cu -o cuda_local_memory.ptx
```

在两个 kernel 的PTX代码中，我们可以发现第一个 kernel 没有使用`.local`指令声明任何内容，而第二个 kernel 有一个使用`.local`指令声明的本地数组`__local_depot1`。这证实了第一个 kernel 将数组`window`放置在寄存器中，而第二个 kernel 将数组`window`放置在本地内存中。即使两个 kernel 中声明的本地数组大小相同，但由于编译器无法确定第二个 kernel 中使用的数组是用常量索引访问的，所以它被放置在本地内存中。

> **注释：** `.local`指令表示变量被存储在本地内存中，而没有这个指令意味着变量被存储在寄存器中。这是判断变量存储位置的关键指标。

```shell
...

	// .globl	_Z27running_mean_register_arrayILi32EEvPKfPfi

.visible .entry _Z27running_mean_register_arrayILi32EEvPKfPfi(
	.param .u64 _Z27running_mean_register_arrayILi32EEvPKfPfi_param_0,
	.param .u64 _Z27running_mean_register_arrayILi32EEvPKfPfi_param_1,
	.param .u32 _Z27running_mean_register_arrayILi32EEvPKfPfi_param_2
)
{
	.reg .pred 	%p<99>;
	.reg .f32 	%f<162>;
	.reg .b32 	%r<41>;
	.reg .b64 	%rd<15>;
...
}
	// .globl	_Z31running_mean_local_memory_arrayILi32EEvPKfPfi
.visible .entry _Z31running_mean_local_memory_arrayILi32EEvPKfPfi(
	.param .u64 _Z31running_mean_local_memory_arrayILi32EEvPKfPfi_param_0,
	.param .u64 _Z31running_mean_local_memory_arrayILi32EEvPKfPfi_param_1,
	.param .u32 _Z31running_mean_local_memory_arrayILi32EEvPKfPfi_param_2
)
{
	.local .align 16 .b8 	__local_depot1[128];  // 这里声明了本地内存数组
	.reg .b64 	%SP;
	.reg .b64 	%SPL;
	.reg .pred 	%p<99>;
	.reg .f32 	%f<194>;
	.reg .b32 	%r<232>;
	.reg .b64 	%rd<82>;
...
}
```

## 结论

为了避免小数组被放置在本地内存中，我们应该避免使用编译器无法确定是否为常量的非常复杂的索引。但问题是我们如何知道编译器是否能够确定索引是常量？

> **注释：** 这是CUDA优化中的一个重要概念。理解编译器的行为可以帮助我们编写更高效的代码。

事实证明，寄存器实际上不能被索引，放置在寄存器中的数组也是如此。如果小数组被放置在寄存器中，小数组的等效常量索引也可以在程序中编写。

例如，来自第一个 kernel `running_mean_register_array`的以下实现：

```c++
constexpr int WindowSize{4};
float window[WindowSize];
float sum{0};
for (int j{0}; j < WindowSize; ++j)
{
    sum += window[j];
}
```

具有等效形式，就好像数组`window`的声明是不必要的：

```c++
float window0, window1, window2, window3;
float sum{0};
sum += window0;
sum += window1;
sum += window2;
sum += window3;
```

> **注释：** 这个例子很好地说明了为什么编译器可以将简单索引的数组放在寄存器中——因为它可以将数组"展开"为单独的寄存器变量。

而来自第二个 kernel `running_mean_local_memory_array`的以下实现：

```c++
constexpr int WindowSize{4};
float window[WindowSize];
float sum{0};
for (int j{0}; j < WindowSize; ++j)
{
    int const idx{(j + n) % WindowSize};
    sum += window[idx];
}
```

没有等效形式，就好像数组`window`的声明是必要的，因为`n`的值只能在运行时知道。

从数学上讲，它也等效于以下形式，但对于编译器来说，弄清楚这一点是一项非常困难的任务：

```c++
float window0, window1, window2, window3;
float sum{0};
sum += window0;
sum += window1;
sum += window2;
sum += window3;
```

> **注释：** 虽然数学上等效，但编译器无法进行这种复杂的优化，因为它需要进行复杂的数学推理来证明索引表达式的等效性。

实际上，这对于CUDA TensorCore MMA PTX也是如此，因为TensorCore MMA需要从寄存器读取数据以获得最佳性能。例如，CUTLASS中的`SM80_16x8x8_F16F16F16F16_TN`MMA实现如下，MMA PTX只访问寄存器，即使缓冲区被声明为数组：

> **注释：** TensorCore是NVIDIA GPU上专门用于加速矩阵乘法的硬件单元，它要求数据必须在寄存器中才能获得最佳性能。

```c++
// MMA 16x8x8 TN
struct SM80_16x8x8_F16F16F16F16_TN
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
      "{%0, %1},"
      "{%2, %3},"
      "{%4},"
      "{%5, %6};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM80_16x8x8_F16F16F16F16_TN without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};
```


这段CUTLASS代码展示了Ampere架构（SM80）TensorCore的关键特性：

**结构体命名解析：**
- `SM80_16x8x8_F16F16F16F16_TN`：SM80架构，M=16/N=8/K=8维度，F16数据类型，A转置B正常布局

**寄存器分配原理：**
```c++
using ARegisters = uint32_t[2];  // A矩阵片段：2个32位寄存器
using BRegisters = uint32_t[1];  // B矩阵片段：1个32位寄存器  
using CRegisters = uint32_t[2];  // C矩阵片段：2个32位寄存器
using DRegisters = uint32_t[2];  // D矩阵片段：2个32位寄存器
```

**关键概念：** 这里的寄存器数量不是存储完整矩阵，而是存储每个线程的**矩阵片段**。

**数据分布机制：**
- 16x8矩阵分布到32个线程（一个warp）
- 每个线程用2个32位寄存器存储4个F16值（每个32位寄存器打包2个F16）
- 总计：32线程 × 2寄存器 × 2个F16 = 128个F16值 = 16×8矩阵

**核心PTX指令：**
```c++
asm volatile(
  "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
  "{%0, %1},"      // 输出：d0, d1
  "{%2, %3},"      // A矩阵：a0, a1  
  "{%4},"          // B矩阵：b0
  "{%5, %6};\n"    // C矩阵：c0, c1
  : "=r"(d0), "=r"(d1) : "r"(a0), "r"(a1), "r"(b0), "r"(c0), "r"(c1));
```

**设计优势：**
1. **硬件特化**：TensorCore只能从寄存器读取数据，确保最高性能
2. **并行效率**：warp级别的数据分布，充分利用32个线程
3. **寄存器优化**：每个线程只需少量寄存器，避免寄存器压力
4. **内存带宽**：避免单线程存储大量数据，提高内存效率


## 参考资料
- Device Memory Accesses - CUDA C Programming Guide(https://docs.nvidia.com/cuda/archive/12.6.3/cuda-c-programming-guide/index.html#device-memory-accesses)
