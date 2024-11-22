> 我的课程笔记，欢迎关注：https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode 。这节课介绍了如何使用 CUDA C++ 核心库(CCCL)将 llm.c 移植为 llm.cpp。CCCL 作为 Thrust、CUB、libcu++ 等库的集合，提供了从高层到底层的完整工具链。课程展示了多个关键改进：将构建系统从 Makefile 迁移到 CMake 以获得更好的跨平台支持；使用 `thrust::device_vector` 替代原始的内存管理实现自动化和类型安全；采用 `cuda::std::mdspan` 简化多维数组操作；使用 `cuda::atomic_ref` 提供更清晰的线程作用域控制。通过 Kernel Fusion 和 CUB 的 BlockReduce 等优化手段，同时利用 NVBench 进行性能测试，最终实现了代码更加简洁、安全和可维护的同时保持了原有的性能和准确率。这次移植很好地展示了如何使用现代 CUDA C++ 工具链来改进传统 CUDA C 代码，在保持性能的同时提高代码质量和开发效率，尽管某些高级抽象可能会增加学习成本，需要在易用性和复杂性之间找到平衡。CCCL开源地址：https://github.com/NVIDIA/cccl

# 第16课，使用 CUDA C++ 核心库移植 llm.c

## 课程笔记

> CUDA C++ 核心库简称 CCCL 是之前已有的一些库比如 Thrust, CUB, libcu++的集合，CCCL(https://github.com/NVIDIA/cccl)致力于让我们使用这些CUDA C++库更方便快捷，同时也在致力于把这些库统一成一个单一的东西。

![](https://files.mdnice.com/user/59/f51ef27a-be15-4a54-9279-ed95cd2c3435.png)

这节课由Nvidia的开发CUDA C++ 核心库（后面简称 CCCL）的两位工程师来讲。

![](https://files.mdnice.com/user/59/82b9f16b-d1f7-422e-9f32-8438b0676d02.png)

CCCL的使命是通过提供高效的库，让 CUDA C++ 开发更高效、轻松。Slides的表格列出了 CUDA C++ 核心库的名称、主要特性、API支持，以及可用性。上面的表格列出了 CUDA C++ 核心库的名称、主要特性、API支持，以及可用性。

| **库名称**            | **主要特点**                              | **支持的 API**               | **可用性**         |
|-----------------------|------------------------------------------|-----------------------------|-------------------|
| **Thrust**            | 高层次的 CPU/GPU 并行算法                  | 设备（Device），主机（Host） | GitHub，CUDA Toolkit |
| **CUB**               | 低层次的 GPU 并行算法                     | 设备（Device），主机（Host） | GitHub，CUDA Toolkit |
| **libcu++**           | 异构 C++ 标准库，提供硬件功能抽象            | 设备（Device），主机（Host） | GitHub，CUDA Toolkit |
| **Cooperative Groups** | 提供线程组之间的命名、同步和通信             | 仅设备（Device）             | CUDA Toolkit       |
| **nvbench**           | 用于 CUDA 性能测试的框架                   | 无                          | GitHub             |

![](https://files.mdnice.com/user/59/3f8fa60a-3ceb-470b-bf08-bd53a74c3ccd.png)

标准 C++ 被定义为：C++ 语言 + 标准库（Standard Library）。然后标准库（Standard Library）提供了通用抽象（General purpose abstractions），数据结构（Data structures），算法（Algorithms），这些特性简化并增强了 C++ 应用开发。没有标准库的支持，C++ 开发将变得繁琐且容易出错。

CUDA C++ 被定义为：C++ 语言 + 主机标准库（Host Standard Library） + CUDA 语言扩展（CUDA Language Extensions） + CUDA C++ 核心库（CUDA C++ Core Libraries）。CUDA C++ 核心库的功能包含：
- 异构 C++ 标准库（Heterogeneous C++ Standard Library）：支持异构计算环境。
- CUDA 基础抽象（Fundamental CUDA Abstractions）：提供低层 GPU 操作的封装。
- 高性能并行算法（High-performance parallel algorithms）：支持复杂并行计算。

这些功能简化并增强了 CUDA C++ 应用开发。没有 CCCL（CUDA C++ 核心库） 的支持，CUDA C++ 的开发也会变得繁琐且容易出错。

![](https://files.mdnice.com/user/59/e306edd4-7b65-4e9a-9f29-804abf4fd163.png)

这张Slides描述了 **CUDA C++ 开发工具的层级范围（The CUDA C++ Spectrum）**，以及如何在不同开发需求下选择合适的工具。横轴表示工具的层次与控制能力，从左到右，工具从 高层且易用（High-Level & Productive） 到 底层且更具控制力（Low-Level & More Control）。左侧起点（绿色箭头 “Start Here”）建议从高层的工具开始，如 `Thrust`，因为这些工具更加易用和生产力高。右侧终点（红色箭头 “Don’t Start Here”）不建议直接从底层工具（如 PTX Wrappers）开始开发，因为它们复杂且难以维护。

高层工具（High-Level & Productive）包含`libcu++`提供 C++ 标准库扩展，如 `cuda::std::variant` 和 `cuda::std::optional`，便于使用容器和抽象化的功能。以及`Thrust`提供 CPU/GPU 并行算法，用于快速开发高层算法和数据处理。

中间层工具（中等抽象层次）包含**迭代器**（Fancy Iterators）如 `cuda::std::span` 和 `cuda::std::mdspan`，用于处理复杂的数据结构；**设备范围算法（Device-wide Algorithms）** 用于对设备内数据进行全局操作；**块范围算法（Block-Scope Algorithms）** 如 `cuda::memcpy_async`，适合更精细的块级控制。**Warp 范围算法（Warp-Scope Algorithms）**：使用 `cuda::atomic` 实现线程束间的同步与控制。

**底层工具（Low-Level & More Control）** 包含**PTX Wrappers** 提供对 PTX 汇编代码的封装，适用于需要极端性能优化的场景；**CUB**提供低级 GPU 并行算法的实现，更灵活但使用复杂。

![](https://files.mdnice.com/user/59/d65a6f20-d1d1-4005-92a2-a4ef9b7c0216.png)

这张Slides展示了把llm.c移植为llm.cpp之后对性能和准确率没有任何影响。

接下来开始从上到下的讲解把llm.c移植到llm.cpp所做的改动：

![](https://files.mdnice.com/user/59/4df13111-b2c7-478f-9168-91e1ae940773.png)

这张Slides展示了把llm.c移植到llm.cpp中首先做的就是把Makefile移植为CMakeLists.txt。

左侧的Makefile需要手动定义编译器（CC=clang）、编译选项（CFLAGS）、链接选项（LDFLAGS）、库路径（LDLIBS）等。并且根据系统环境（如是否存在 OpenMP）选择不同的编译器选项，包含大量的 if 条件语句和 shell 调用。另外，Makefile 中需要处理各种编译器、平台、依赖项的细节。缺乏现代构建系统的高层次抽象，手动管理依赖性容易出错。

右侧的CMakeLists.txt使用更简单的代码设置 C++ 和 CUDA 的标准（`set(CMAKE_CXX_STANDARD 20`)，`set(CMAKE_CUDA_STANDARD 20)`）。使用 `set(CMAKE_CUDA_ARCHITECTURES "native")` 自动检测 CUDA 架构，减少了手动配置的复杂性。使用 `find_package` 和 `CPMAddPackage` 自动管理依赖项：如 `OpenMP` 和 `CUDAToolkit`。还演示了如何通过 `gh:NVIDIA/cccl#main` 和 `gh:NVIDIA/nvbench#main` 添加其CCCL的最新代码依赖。使用 `add_executable` 定义可执行文件目标，并通过 `target_link_libraries` 轻松链接需要的库，如 `OpenMP` 和 `CUDA` 的库（`cublas` 和 `cublasLt`）。编译选项被简化为高层配置（`target_compile_options`），比如 `--use_fast_math` 和 `--extended-lambda`。

切换到CMakeLists之后，有如下优点：
- **同样的代码生成能力（Same code gen）**：构建产物（如二进制文件）与手动管理 Makefile 相同。
- **跨平台支持（Cross-platform）**：CMake 支持多平台（如 Windows 和 Linux），减少平台相关的配置。
- **减少编译器依赖（Reduced compiler dependencies）**：自动管理编译器和依赖项，避免手动设置编译选项。
- **更少错误（Less error-prone）**：提示缺失的 CUDA 架构等问题，而不是默默失败。
- **自动化依赖管理（Setup-free dependency management）**：通过 `CPM.cmake` 或 `find_package` 自动下载和管理依赖，省去手动安装依赖的麻烦。

![](https://files.mdnice.com/user/59/ac4387e1-2202-4b18-8872-5bcbca847dcf.png)

这张Slides介绍了Thrust在内存管理方面对llm.c的代码移植。Thrust容器的一个直接好处是它们自动管理内存释放，这意味着我们不用担心会意外忘记释放内存。

![](https://files.mdnice.com/user/59/bfba69bd-429b-4b37-92ba-e270d4156a10.png)

除了内存管理安全，Thrust还提供了Type安全。对于Slides这个把复数赋值到整数的例子，上面部分的代码类型安全性差，在 cudaMemcpy 的源和目标类型不匹配的情况下，编译器无法检测到问题（如示例中 `int*` 和 `cuda::std::complex<float>*`），但是使用Thrust时则会在编译期直接报类型不匹配错误。

![](https://files.mdnice.com/user/59/8f2964fd-24f8-4ae2-8085-e2330768d71a.png)

这个例子说明了手动内存管理中的类型错误（如 int 和 float 混用）可能导致难以察觉的错误，`cudaMemcpy` 虽然不会报错，但将 int 的二进制表示错误地解读为 `float`，导致 `d_float[0]` 的值是一个无意义的浮点数（如 `5.88545e-44`），而 Thrust 容器通过强类型检查和高层次抽象有效避免了这些问题，确保了代码的安全性和正确性。

![](https://files.mdnice.com/user/59/36fe3ca0-a405-42b7-9ea0-b63c16d36667.png)

这张Slides展示了Thrust的Customizable（可定制）特性，通过自定义分配器（如 pinned memory）满足特定需求。
。如Slides中高亮部分所示， `pinned_vector` 使用自定义分配器 `thrust::stateless_resource_allocator`，为固定内存（pinned memory）提供支持。

![](https://files.mdnice.com/user/59/4a47f0ac-0298-4ec1-bdc0-b822ae7d3ba9.png)

这张Slides展示了两种CUDA编程实现方式的对比：左侧展示了使用传统CUDA C API的`cudaMemset`函数的实现，它直接操作字节级内存但容易产生难以调试的错误；右侧则展示了使用现代Thrust库的`fill_n`函数的替代方案，这种方式不仅类型安全，而且代码更简洁清晰，不容易出错。这个对比很好地说明了如何通过使用更高级的CUDA库来提升代码的安全性和可维护性。

![](https://files.mdnice.com/user/59/d91b7e54-914e-40d6-844b-4f6160ceb574.png)

这张Slides对比了两种实现GELU函数的方法：“当前”的CUDA kernel方法需要手动管理执行细节，而“替代”的方法使用了`thrust::transform`来更清晰地表达意图。替代方法降低了心智负担，并通过抽象执行细节提供了更好的优化可能性。

![](https://files.mdnice.com/user/59/0e2463a1-e48b-41a4-9da6-689cf8f6525d.png)

这里还提到我们可以通过定义`THRUST_DEVICE_SYSTEM`来指定`thrust::transform`的执行设备，这张Slides中就展示了把这个kernel的执行设备定义为CPU，并且不需要修改kernel实现代码。

![](https://files.mdnice.com/user/59/f4705275-789a-4e2b-850d-8a0f4b7ed387.png)

这张Slides用于说明高级抽象并不会牺牲底层控制能力（High-level abstractions do not sacrifice low-level control）。例如可以通过`cub::CacheModifiedInputIterator<cub::LOAD_CS, float>`来等价cuda kernel中直接调用底层的`__ldcs`指令。

![](https://files.mdnice.com/user/59/eb8841b2-df9f-4db8-84c1-f6720ad67297.png)

这张Slides展示了使用`CacheModifiedInputIterator`不仅支持内置的数据类型，还可以支持复杂数据类型（如`cuda::std::complex<float>`）。此外，`CacheModifiedInputIterator`不仅局限于流式加载（`LOAD_CS`），还可以支持其他加载方式（`LOAD_LDG`，这个等价于kernel中的`__restrict__` 修饰）。

![](https://files.mdnice.com/user/59/1c0149f8-715a-4eea-979f-7fc92926f043.png)

这张Slides展示了算法自定义的两种方法：当前方法通过CUDA kernel函数手动实现，需显式管理线程和内存加载；替代方法使用CUB库和`thrust::transform`实现更简洁的代码，同时支持多种CUDA执行策略（如`thrust::device`、`thrust::cuda::par_nosync`和`thrust::cuda::par_on(stream)`），可以同步或者异步的运行kernel。Slides强调CUDA的执行策略不限于`thrust::device`，可以灵活选择。

![](https://files.mdnice.com/user/59/deb3dc06-8000-4f41-98e4-07cc1f9df232.png)

这张Slides展示了使用`tuple`优化CUDA代码的两种方式：llm.c中的当前方法通过手动管理索引和拆解计算逻辑，代码冗长且重复；替代方法使用`cuda::std::tuple`简化了索引计算和变量管理，提高了代码的可读性和可维护性。Slides强调：libcu++使许多标准类型（如`cuda::std::variant`、`cuda::std::tuple`、`cuda::std::pair`等）在设备代码中可用，同时实现了 **DRY（Don't Repeat Yourself）** 原则，减少代码重复。

![](https://files.mdnice.com/user/59/67d8c8be-66e7-4136-9960-7e45d165b0c5.png)

这张Slides对比了两种实现方式：llm.c中的当前方法通过显式的索引计算和内核调用实现数据的unpermute操作，代码复杂且难以维护；替代方法使用`thrust::make_transform_iterator`和`thrust::scatter`，通过迭代器抽象简化了索引计算和数据操作，提升了代码的简洁性和可读性。Slides展示了如何利用高级迭代器实现更高效、更易维护的CUDA代码。

![](https://files.mdnice.com/user/59/3e732f33-7039-4758-954a-223d03b1983c.png)

这张Slides介绍了`thrust::make_counting_iterator`和`thrust::make_transform_iterator`的使用方式，通过迭代器生成和转换实现更高效的索引计算和数据操作。示例展示了`make_counting_iterator`用于生成连续数字序列，`make_transform_iterator`通过自定义转换函数对生成的序列进行映射。Slides重点突出迭代器在简化CUDA代码、减少手动索引管理方面的作用，提高了代码的可读性和灵活性。

> 到这里，视频里面有一些有趣的讨论，很多人都认为这个unpermute kernel迁移的例子过于复杂了，原始的代码需要学习的概念很少很容易理解，但是Thrust的代码更加复杂并且需要学习更多的东西。

![](https://files.mdnice.com/user/59/8b147082-41c2-4cd4-a091-c2eeef78f310.png)

这张Slides比较了使用传统多维数组索引和MDSpan优化CUDA代码的方法。传统方法中，通过手动计算索引访问多维数组，代码复杂且难以维护；替代方法使用`cuda::std::mdspan`管理多维数据，通过抽象简化了索引计算并保留了编译期信息。下面我分别对这两组代码添加注释：

llm.c中的代码：

```c++
// permute_kernel函数实现矩阵重排列操作
__global__ void permute_kernel(float* q, float* k, float* v,
                             const float* inp, int B, int N, int NH, int d) {
    // 计算当前线程的全局索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 原始代码中的矩阵重排列计算
    // dlb[nh][n][d_] = inp[b][n][nh][d_]
    
    // 计算输入张量的各个维度索引
    int b = idx / (NH * N * d);      // batch维度
    int rest = idx % (NH * N * d);   // 剩余部分
    int nh = rest / (N * d);         // head维度
    rest = rest % (N * d);           // 继续分解剩余部分
    int n = rest / d;                // 序列长度维度
    int d_ = rest % d;               // 特征维度
    
    // 计算输入张量的线性索引
    int inp_idx = 
        (b * N * NH * d) +           // batch偏移
        (n * NH * d) +               // 序列长度偏移
        (nh * d) +                   // head偏移
        d_;                          // 特征维度偏移
    
    // 执行张量重排列操作
    q[idx] = __ldcs(&inp[inp_idx]); // 使用__ldcs进行缓存优化的内存读取
    k[idx] = __ldcs(&inp[inp_idx + NH * d]);
    v[idx] = __ldcs(&inp[inp_idx + 2 * NH * d]);
}

// attention_forward函数实现注意力前向传播
void attention_forward(float* out, float* veccum, float* qkv, float* presft, float* att,
                      int B, int T, int C, int NH) {
    const int block_size = 256;              // CUDA线程块大小
    const int softmax_block_size = 256;      // Softmax操作的线程块大小

    int HS = C / NH;                        // 每个head的维度大小
    
    // 计算每个head的维度大小
    float *q, *k, *v;
    q = qkv;                                 // 查询矩阵Q的起始位置
    k = qkv + B * T * C;                     // 键矩阵K的起始位置
    v = qkv + 2 * B * T * C;                 // 值矩阵V的起始位置
    
    // 计算需要的CUDA线程块数量
    int total_threads = B * NH * T * HS;
    int num_blocks = CEIL_DIV(total_threads, block_size);
    
    // 启动permute_kernel进行张量重排列
    permute_kernel<<<num_blocks, block_size>>>(q, k, v, qkv, B, T, NH, HS);
}
```

llm.cpp中的代码：

```c++
void attention_forward(float* out, float* vaccum, float* qkvr, float* prestt, float* att,
                      float* inp, int B, int T, int C, int NH) {
    // 设置CUDA块大小常量
    const int block_size = 256;
    const int softmax_block_size = 256;
    
    // 计算每个注意力头的维度大小
    int HS = C / NH;  // head size
    
    // 设置Q、K、V矩阵的指针，它们在内存中是连续存储的
    float *q, *k, *v;
    q = qkvr + 0 * B * T * C;      // Q矩阵起始位置
    k = qkvr + 1 * B * T * C;      // K矩阵起始位置
    v = qkvr + 2 * B * T * C;      // V矩阵起始位置
    
    // 使用CUDA动态内存分配
    constexpr auto dyn = cuda::std::dynamic_extent;
    using ext_t = cuda::std::extent<int, dyn, dyn, 3, dyn, dyn>;
    using mds_t = cuda::std::mdspan<const float, ext_t>;
    
    // 创建多维数组视图，用于更方便地访问数据
    ext_t extents(B, T, NH, HS);
    mds_t inp_md(inp, extents);
    
    // 使用thrust库创建迭代器，用于并行处理
    auto begin = thrust::make_counting_iterator(0);
    auto end = begin + B * NH * T * T;
    
    // 原始重排列操作的注释：Q[b][nh][t][d_] = inp[b][t][nh][d_]
    
    // 使用thrust并行处理每个元素
    thrust::for_each(thrust::cuda::par,
                    begin, end,
                    [=] __device__ (int idx) {
                        // 计算当前处理位置的各个维度索引
                        auto [b, t, nh_, hs] = idx2(idx, NH, T, HS);
                        
                        // 执行Q、K、V矩阵的数据重排列
                        q[idx] = inp_md(b, t, 0, nh_, hs);  // Q矩阵赋值
                        k[idx] = inp_md(b, t, 1, nh_, hs);  // K矩阵赋值
                        v[idx] = inp_md(b, t, 2, nh_, hs);  // V矩阵赋值
                    });
}
```

为了解释llm.cpp中的代码，这里需要解释一下`mdspan`，下面给了一个例子：

![](https://files.mdnice.com/user/59/077a8951-cab2-46c2-bb81-688f9bd7d163.png)

通过`mdspan`，我们可以把一个一维数组方便的以多维的方式进行访问。


![](https://files.mdnice.com/user/59/f4a0e44b-80e7-4c39-b1bf-0dc9b0128546.png)

在llm.c的kernel中还采用了`__ldcs`指令进行缓存优化的内存读取，MDSpan也可以通过streaming_processor的抽象来做到。

![](https://files.mdnice.com/user/59/057b0a59-249b-4d0d-90bd-8928edb2f007.png)


这张Slides比较了当前方法和使用`MDSpan`优化CUDA代码的替代方法。当前方法通过手动索引计算和kernel调用实现，代码冗长且难维护；替代方法利用`cuda::std::mdspan`定义了领域特定类型（如`float_3d_mds`和`float_2d_mds`），通过抽象封装多维数据访问，显著提高代码的清晰性和可维护性。此外，`MDSpan`支持直接在编译期管理多维索引，减少了代码中的手动计算和重复逻辑。

> 这里提出了一个问题，循环语句从`trhust::for_each`改成了`cub::DeviceFor::Bulk`，它们区别是什么？回答：`thrust::for_each`不仅可以在CUDA kernel中使用，还可以在CPU上使用，而`cub::DeviceFor::Bulk`只能在CUDA kernel中使用，是一个包含关系。所以如果只是实现CUDA kenrel，可以使用`cub::DeviceFor::Bulk`，这样执行kernel是异步的。

![](https://files.mdnice.com/user/59/48eb5343-a767-404f-8158-adcb949e3bfc.png)

这张Slides开始展示Thrust的kernel fuse实现，这里的代码比较复杂，继续解释一下。

```c++
// CUDA核函数：计算交叉熵的前向传播
__global__ void crossentropy_forward_kernel1(float* losses,
                                           float* probs, int* targets,
                                           int B, int T, int V) {
    // 计算全局线程索引
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 确保线程索引在有效范围内
    if (i < B * T) {
        // 计算batch和时间步索引
        int b = i / T;        // 批次索引
        int t = i % T;        // 时间步索引
        
        // 获取目标类别的概率值并计算负对数似然
        float probs_t = probs[b * T * V + t * V + t];
        int ix = targets[b * T + t];
        losses[b * T + t] = -logf(probs[ix]);
    }
}

// 交叉熵前向传播的包装函数
void crossentropy_forward(float* losses,
                         float* probs, int* targets,
                         int B, int T, int V) {
    // 定义CUDA网格和块的大小
    const int block_size = 128;
    const int N = B * T;
    const int grid_size = (N + block_size - 1) / block_size;
    
    // 启动CUDA核函数
    crossentropy_forward_kernel1<<<grid_size, block_size>>>(
        losses, probs, targets, B, T, V);
    
    // 检查CUDA错误
    cudaCheck(cudaGetLastError());
}

// 主函数中的调用示例
crossentropy_forward(acts_losses, acts_probs, model->targets, B, T, V);
cudaCheck(cudaMemcpy(model->cpu_losses, acts_losses, B * T * sizeof(float),
                     cudaMemcpyDeviceToHost));

// 计算平均损失
float mean_loss = 0.0f;
for (int i=0; i<B*T; i++) { 
    mean_loss += model->cpu_losses[i]; 
}
mean_loss /= mean_loss;
```


主要参数说明：
- losses: 输出的损失值数组
- probs: 模型预测的概率分布
- targets: 目标类别的索引
- B: batch size（批次大小）
- T: sequence length（序列长度）
- V: vocabulary size（词汇表大小）

```c++
// 创建目标矩阵,大小为 B×T
// B 是 batch size, T 是序列长度
target_matrix targets_md(thrust::raw_pointer_cast(model.targets.data()), B, T);

// 创建转换迭代器,用于计算目标索引位置
auto map = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0),  // 从0开始的计数迭代器
    [=] __device__ (int i) -> int {     // 设备端 lambda 函数
        int b = i / T;                  // 计算批次索引
        int t = i % T;                  // 计算时间步索引
        // 计算目标位置: 基础位置 + batch偏移 + 时间步偏移
        // V 是词汇表大小
        return targets_md(b, t) + b * T * V + t * V;
    }
));

// 创建排列迭代器,根据计算的索引访问概率值
auto permutation = thrust::make_permutation_iterator(acts.probs, map);

// 创建转换迭代器,计算每个概率值的负对数损失
auto losses = thrust::make_transform_iterator(
    permutation, 
    [] __device__ (float prob) -> float { return -logf(prob); }
);

// 计算平均损失:
// 1. 使用 reduce 对所有损失求和
// 2. 除以样本总数(B*T)得到平均值
model.mean_loss = thrust::reduce(thrust::device, losses, losses + B * T, 0.0) / (B * T);
```

![](https://files.mdnice.com/user/59/a9b2c76a-510a-4eb3-833f-2aa38a2c3f2c.png)

这张Slides展示了通过**Kernel Fusion**优化数据处理流程的对比。当前方法将中间结果（`dlosses` 和 `hlosses`）从GPU传回CPU后再计算平均值（`mean`），导致更多数据通过PCIe传输；替代方法将计算融合在GPU上完成，直接输出平均值，减少了`B * T`倍的数据跨PCIe传输，同时保留了排列步骤用于说明优化流程。Kernel Fusion显著提升了性能和数据传输效率。注意Slides中的permutation不是必要的。

![](https://files.mdnice.com/user/59/ae49c062-8c48-46a2-a436-d89d37fe060e.png)

这张Slides比较了传统`atomicAdd`与使用`cuda::atomic_ref`的替代方法。当前方法通过`atomicAdd`实现，但无法直观理解线程范围和内存顺序。替代方法使用`cuda::atomic_ref`明确指定线程范围（如`cuda::thread_scope_device`）和内存顺序（如`cuda::memory_order_relaxed`），提高了代码的可读性和可控性。Slides强调，替代方法的通用API不局限于内置类型，提供更灵活的原子操作支持。

![](https://files.mdnice.com/user/59/fe235238-aef0-4da3-8b7e-b97251d4d749.png)

这张Slides主要介绍了CUDA中的Thread Scope（线程作用域）概念。图中展示了一个包含1个主机（host）和2个设备（device）的系统架构，每个设备上有2个block。Thread Scope定义为一组可以直接交互的线程集合，这些线程之间能够建立内存一致性模型中描述的关系。图示通过不同颜色的波浪线（蓝色表示host线程，绿色表示device线程）直观地展示了不同作用域内的线程分布情况。这个概念对于理解CUDA程序中线程间的通信和同步机制非常重要。

![](https://files.mdnice.com/user/59/a2c39ded-63f5-4481-81f7-00b7cb9eb0d4.png)

这张Slides介绍了CUDA中的线程作用域(Thread Scope)概念。图中展示了一个包含1个主机(host)和2个设备(device)的系统架构，每个设备上有2个线程块(block)。其中特别标注了`cuda::thread_scope_block`是一个给定线程块内的线程集合。这个概念说明了在CUDA编程中，线程的组织和管理是基于层次化的结构，从主机到设备，再到设备内的线程块，最后到具体的线程集合，形成了一个清晰的层级关系。

![](https://files.mdnice.com/user/59/ff705776-61be-48d4-9acb-18f3791bec43.png)

`cuda::thread_scope_device`是给定设备内的线程集合。

![](https://files.mdnice.com/user/59/e4472d13-15ff-4246-ba0b-1ad5fb357a20.png)

`cuda::thread_scope_system`是整个系统内的线程集合。

![](https://files.mdnice.com/user/59/ca5e9247-c913-4705-b76b-03bbec4ac05f.png)

这张Slides展示了我们可以使用`cub::BlockReduce`来增加处理每一列LayerNorm的并行度，通过把线程数从32改成64。

```c++
__global__
void layernorm_forward_kernel3(float* out, float* mean, float* rstd,
                             const float* inp, const float* weight,
                             const float* bias, int N, int C) {
    // 获取当前线程块配置
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    // 计算当前线程的全局索引
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    
    // 计算输入数据的起始位置
    const float* x = inp + idx * C;
    
    // 计算均值
    float sum = 0.0f;
    // 使用warp内的线程并行计算和
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        sum += x[i];
    }
    
    // 使用warp级别的归约操作求和
    sum = cg::reduce(warp, sum, cg::plus<float>());
    
    // 计算最终均值并存储
    float m = sum / C;
    if(warp.thread_rank() == 0 && mean != nullptr) {
        mean[idx] = m;
    }
}
```


这段代码实现了LayerNorm的前向传播计算中的一部分，主要完成了均值的计算。具体解释：
1. 函数参数包含了输入数据、权重、偏置等必要参数，以及维度信息N和C
2. 使用CUDA Cooperative Groups来管理线程组织结构
3. 代码使用了warp级别的并行计算来提高效率
4. 均值计算分两步：
    - 首先在warp内的线程间并行累加
    - 然后使用warp级别的归约操作得到最终结果
5. 最后由warp中的第一个线程负责将结果写入到输出数组中
这种实现方式充分利用了CUDA的硬件特性，能够高效地完成LayerNorm所需的均值计算。

```c++
// 设置每个线程块包含的线程数
constexpr int block_size = 64;
// 定义核函数的线程块配置
__global__ __launch_bounds__(block_size)
void layernorm_forward_kernel3(float* out, float* mean, float* rstd,
                             const float* inp, const float* weight,
                             const float* bias, int N, int C) {
    // 获取线程在块内的局部索引
    int tid = threadIdx.x;
    // 获取当前处理的数据块索引
    int idx = blockIdx.x;
    
    // 计算当前线程处理的输入数据起始位置
    const float* x = inp + idx * C;
    
    // 计算均值
    float sum = 0.0;
    // 使用跨步循环方式让每个线程处理多个元素
    for (int i = tid; i < C; i += block_size) {
        sum += x[i];
    }
    
    // 使用CUB库的BlockReduce进行块内归约求和
    sum = cub::BlockReduce<float, block_size>().Sum(sum);
    
    // 在共享内存中存储计算结果
    __shared__ float shared_mean;
    if(tid == 0 && mean != nullptr) {
        // 计算最终均值
        float m = sum / C;
        // 存储到共享内存
        shared_mean = m;
        // 将结果写回全局内存
        __stcs(mean + idx, m);
    }
    
    // 确保所有线程同步，等待均值计算完成
    __syncthreads();
    // 从共享内存读取均值供后续计算使用
    const float m = shared_mean;
}
```


这个实现与前一个版本相比有以下几个主要区别：
1. 使用固定的block_size和__launch_bounds__来优化编译器生成的代码
2. 采用了更简单的线程索引计算方式
3. 使用CUB库的BlockReduce替代了Cooperative Groups的实现
4. 通过共享内存来共享计算结果
5. 使用显式的线程同步来确保数据一致性
使用CUB库的好处是能够利用经过优化的归约操作实现，并且在列方向上增大了并行度，可能有更好的性能表现。

![](https://files.mdnice.com/user/59/192b1e90-4d7f-4367-b94c-baf71cb37e57.png)

我们做基准测试的时候，如果多次运行kernel，第2次运行可能会读取缓存导致统计不准确，另外性能数据很少是正态分布的，因此理想情况下我们需要一个统计引擎来帮助我们决定统计数据是否足够。这两个问题都由NVBench这个库来解决。NVBench专门设计用于可靠的测量CUDA kernel性能。

```c++
// 定义kernel函数,用于神经网络计算
void kernel3(nvbench::state &state) {
    // 定义基本参数
    int B = 32;        // batch size
    int T = 1024;      // 序列长度
    int C = 768;       // 隐藏层维度

    // 在主机端分配向量内存
    thrust::host_vector<float> h_inp(B * T * C);    // 输入数据
    thrust::host_vector<float> h_weight(C);         // 权重
    thrust::host_vector<float> h_bias(C);           // 偏置

    // 初始化随机数生成器和分布
    thrust::default_random_engine gen(42);
    thrust::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    // 生成随机数据填充输入、权重和偏置
    thrust::generate(h_inp.begin(), h_inp.end(), [&] { return dis(gen); });
    thrust::generate(h_weight.begin(), h_weight.end(), [&] { return dis(gen); });
    thrust::generate(h_bias.begin(), h_bias.end(), [&] { return dis(gen); });

    // 在设备端分配内存
    thrust::device_vector<float> d_out(B * T * C);    // 输出
    thrust::device_vector<float> d_mean(B * T);       // 均值
    thrust::device_vector<float> d_rstd(B * T);       // 标准差的倒数
    thrust::device_vector<float> d_inp(h_inp);        // 输入数据拷贝到设备
    thrust::device_vector<float> d_weight(h_weight);  // 权重拷贝到设备
    thrust::device_vector<float> d_bias(h_bias);      // 偏置拷贝到设备

    // 计算网格和块的大小
    const int N = B * T;
    const int block_size = state.get_int64("block_size");
    const int grid_size = (N * 32 + block_size - 1) / block_size;

    // 分配全局内存
    state.add_global_memory_reads<float>(d_inp.size() + d_weight.size() + d_bias.size());
    state.add_global_memory_writes<float>(d_out.size() + d_mean.size() + d_rstd.size());

    // 执行kernel
    state.exec([&](nvbench::launch launch) {
        cudaStream_t stream = launch.get_stream();
        layernorm_forward_kernel3<<<grid_size, block_size, 0, stream>>>(
            // 传递原始指针给kernel函数
            thrust::raw_pointer_cast(d_out.data()),
            thrust::raw_pointer_cast(d_mean.data()),
            thrust::raw_pointer_cast(d_rstd.data()),
            thrust::raw_pointer_cast(d_inp.data()),
            thrust::raw_pointer_cast(d_weight.data()),
            thrust::raw_pointer_cast(d_bias.data())
        );
    });

    // 设置基准测试的参数
    NVBENCH_BENCH(kernel3).add_int64_axis("block_size", {32, 64, 128, 256, 512, 1024});
}
```

最终针对不同的block_size的测试结果如下所示：

![](https://files.mdnice.com/user/59/f454003f-7cc4-4bef-ae3d-275e962dfd0f.png)

![](https://files.mdnice.com/user/59/0fb830e9-68d0-42f9-a111-810f5b6a5d4b.png)

总结一下这节课的要点：
- 在内存管理方面，建议避免直接使用`cudaMalloc/cudaFree`这样的原始内存分配，而是使用`thrust::device_vector`这样的容器；
- 在编写`kernel`时，推荐使用CUB库的block/warp算法来构建基础模块，使用`cuda::atomic_ref`而不是atomicAdd，并善用`cuda::std`中的`array、variant、tuple`和`optional`等类型；
- 在开发自定义kernel之前，应该考虑使用Thrust（用于高层抽象和CPU/GPU支持）或CUB（用于底层CUDA控制）库中的现成算法，并利用迭代器来增强算法功能；
- 在一般性建议方面，推荐使用CMake作为CUDA C++的构建系统，使用NVBench进行可靠的CUDA性能测试，使用`cuda::std::span`代替原始指针，以及使用`cuda::std::mdspan`来处理多维数据。


![](https://files.mdnice.com/user/59/18545603-bd24-461a-bf8d-58adf66a33c8.png)

## 总结

这节课介绍了如何使用 CUDA C++ 核心库(CCCL)将 llm.c 移植为 llm.cpp。CCCL 作为 Thrust、CUB、libcu++ 等库的集合，提供了从高层到底层的完整工具链。课程展示了多个关键改进：将构建系统从 Makefile 迁移到 CMake 以获得更好的跨平台支持；使用 `thrust::device_vector` 替代原始的内存管理实现自动化和类型安全；采用 `cuda::std::mdspan` 简化多维数组操作；使用 `cuda::atomic_ref` 提供更清晰的线程作用域控制。通过 Kernel Fusion 和 CUB 的 BlockReduce 等优化手段，同时利用 NVBench 进行性能测试，最终实现了代码更加简洁、安全和可维护的同时保持了原有的性能和准确率。这次移植很好地展示了如何使用现代 CUDA C++ 工具链来改进传统 CUDA C 代码，在保持性能的同时提高代码质量和开发效率，尽管某些高级抽象可能会增加学习成本，需要在易用性和复杂性之间找到平衡。CCCL开源地址：https://github.com/NVIDIA/cccl



