> 我的课程笔记，欢迎关注：https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode 

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

