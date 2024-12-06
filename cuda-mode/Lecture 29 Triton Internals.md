> 我的课程笔记，欢迎关注：https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode 。

> 这节课详细介绍了 Triton 编译器的内部工作原理。文章首先介绍了 CUDA 编译器（NVCC）的工作流程，然后深入探讨了 Triton 编译器的架构设计，包括其如何将 Python DSL 代码通过多个中间表示（IR）最终编译成 CUDA 可执行文件。课程重点讲解了 Triton 与 MLIR（Multi-Level Intermediate Representation）的关系，展示了各种优化 Pass 的实现和作用，如 TritonCombineOps、TritonReorderBroadcast 等。通过一个 vector_add 的具体示例，详细展示了代码从 Python 层面到 GPU IR 的转换过程，以及在这个过程中各种中间产物（如 .cubin、.ptx 等）的生成。最后还介绍了如何在 Triton 中实现新的编译器 Pass，为读者提供了深入理解和扩展 Triton 编译器的实践指导。

# 第29课，Triton Internals

## 课程笔记

![](https://files.mdnice.com/user/59/7674c735-67a8-4bc0-a647-20e56b4182c4.png)

![](https://files.mdnice.com/user/59/3fe50325-46cb-40e9-b060-e6ff0045277e.png)

这是一场由 Meta 软件工程师 Kapil Sharma 主讲的技术分享，主题是 Triton 编译器的内部工作原理。演讲者目前在 Meta 的 RecSys/Ranking 基础设施团队工作，他在Slide中分享了自己的社交媒体和代码仓库链接，包括 LinkedIn、Twitter 和 GitHub。

![](https://files.mdnice.com/user/59/ec86d7b3-f6bc-4a9c-9242-9daf651e97aa.png)

这张Slide介绍了关于Triton的演讲概要。Triton是一个复杂的编译器/代码生成机制，之前已经有过一些相关演讲（包括Triton 101、Kernel融合和Liger kernel等）。研究人员普遍喜欢使用Triton，它既可用于研究也可用于生产环境。Triton的工作流程是从PyTorch开始，通过torch.compile编译成Triton内核，最后部署到目标硬件上。相关内容可以在三个系列博客文章中找到更详细的信息。这是一个将PyTorch代码优化并编译到GPU上运行的重要工具。

三个系列博客的链接：
- https://www.kapilsharma.dev/posts/deep-dive-into-triton-internals/
- https://www.kapilsharma.dev/posts/deep-dive-into-triton-internals-2/
- https://www.kapilsharma.dev/posts/deep-dive-into-triton-internals-3/

![](https://files.mdnice.com/user/59/abb3613e-8347-402a-ad45-ec826607a071.png)

这是本次演讲的目录。主要内容包括CUDA编译、Triton编译器、示例代码、JIT编译、Triton和MLIR(机器学习中间表示)的关系、IR(中间表示)的详细介绍、MLIR Pass示例等主题。如果时间允许，还会介绍一个新的编译器pass。

![](https://files.mdnice.com/user/59/88c6cfd1-1734-4865-95d8-fac8c1b95bfb.png)

![](https://files.mdnice.com/user/59/cbc5b22b-5f47-4b4b-bc31-09ee48257827.png)

这张Slide展示了NVIDIA CUDA编译器(NVCC)的完整工作流程。NVCC的主要作用是将CUDA代码分离成主机代码（C/C++）和设备代码（CUDA内核）。主机代码通过标准的C/C++编译器（如g++或clang）进行编译，而设备代码则被编译成PTX或cubin格式。整个流程通过一系列的中间步骤，包括预处理、编译和链接，最终生成可执行文件。图中清晰地展示了从.cu源文件到最终可执行文件的完整转换过程。

![](https://files.mdnice.com/user/59/8751c803-652b-4e4e-abfe-e32cfed442eb.png)

这张Slide来自NVIDIA的演讲，详细说明了NVCC编译器的内部结构和组件。NVCC是CUDA的主编译器，而CICC是基于LLVM的高级优化器和PTX生成器。PTX（Parallel Thread Execution）是一种虚拟指令集。图示展示了从.cu文件开始，经过CUDA C++前端处理，然后通过CICC生成PTX汇编代码，最后通过PTXAS和主机编译器生成最终的CUDA可执行文件的完整流程。

![](https://files.mdnice.com/user/59/6fc4b080-e342-4314-a6a6-b8d24a55a96f.png)

这张Slide重点介绍了NVCC编译器处理设备代码的两个关键阶段。第一阶段使用C++预处理器和CICC程序生成中间形式，第二阶段CICC程序优化代码并生成PTX。生成的PTX代码随后传递给ptxas，用于生成SASS（实际的GPU机器代码）。整个流程图清晰地展示了从源代码到最终GPU可执行代码的转换过程，并提到了Godbolt示例作为参考，我们也可以参考Compiler Explorer，这两个网站都是很好的在线CUDA编译器可视化工具。

![](https://files.mdnice.com/user/59/1c3ea27c-ac6d-4bf0-814d-d54632b5fa05.png)

这张Slide是OpenAI Triton介绍博客里面的图，主要介绍了Triton编译器的工作流程。它展示了代码从Python DSL(领域特定语言)到最终CUDA执行文件的转换过程。具体来说，Triton编译器会将DSL代码经过多个阶段的编译，最终生成CUBIN/fatbinary格式的可执行文件。这个生成的CUBIN文件是一个Slide格式的文件，包含了可以在CUDA kernel中内联加载的可执行代码。Slide通过展示了三个不同层次的代码(Python层、Triton-IR层和PTX层)来说明这个编译过程。


![](https://files.mdnice.com/user/59/f0beacd8-4792-44e2-9e3c-aca5750d16f3.png)

这张Slide展示了Triton编译器的架构设计图。从顶层看,它始于Triton语言,然后通过Triton IR(中间表示)向下流转。架构分为几个主要分支:
- 左侧分支通过SPIRV处理,最终编译到x86和Intel GPU后端
- 中间分支经过LLVM和AMD GPU后端处理
- 右侧分支通过LLVM和PTX处理,最终到达SASS(NVIDIA GPU汇编)
- 最右侧还有一个单独的分支用于处理各类加速器(Accelerators)

这种架构设计允许Triton代码能够被编译到不同的硬件平台上运行,包括CPU(x86)、各种GPU(Intel/AMD/NVIDIA)以及其他加速器,体现了良好的跨平台兼容性。

![](https://files.mdnice.com/user/59/7aef6509-91cd-42df-8e98-ad79dcda6986.png)

这是Triton的官方教程的第一个例子vector_add，然后我们可以使用Triton提供的编译工具来看一下生成可执行文件的过程中dump了哪些东西。

![](https://files.mdnice.com/user/59/be500a2c-4d8b-454b-842a-e7cbb7bd5fc2.png)

这里展示了一下vector_add在生成可执行文件过程中产生了哪些中间参数，包括编译和构建过程中生成的文件,如`.cubin、.json、.llir、.ptx`等，还有包含cuda kernel的最终源代码文件 `add_kernel.9969bdda_0123.c` (.c文件)和对应的头文件 (.h文件)。

- https://github.com/gpu-mode/lectures/blob/main/lecture_029/add_kernel.9969bdda_0123.h
- https://github.com/gpu-mode/lectures/blob/main/lecture_029/add_kernel.9969bdda_0123.c

我们可以看一下`add_kernel.9969bdda_0123.c`的内容，由于篇幅原因 `unsigned char CUBIN_NAME[10960]` 的内容只展示了一小部分：

```c++
/* clang-format off */
// 包含必要的头文件
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <cuda.h>


// 定义CUDA错误检查宏
#define CUDA_CHECK(ans) {\
    gpuAssert((ans), __FILE__, __LINE__);\
  }\

// CUDA错误检查辅助函数
static inline void gpuAssert(CUresult code, const char *file, int line) {
  if (code != CUDA_SUCCESS) {
    const char *prefix = "Triton Error [CUDA]: ";
    const char *str;
    cuGetErrorString(code, &str);  // 获取错误字符串
    char err[1024] = {0};
    strcat(err, prefix);  // 拼接错误前缀
    strcat(err, str);     // 拼接错误信息
    printf("%s\\n", err); // 打印错误信息
    exit(code);          // 退出程序
  }
}

// 全局变量定义
#define CUBIN_NAME add_kernel_9969bdda_0123_cubin
CUmodule add_kernel_9969bdda_0123_mod = NULL;    // CUDA模块句柄
CUfunction add_kernel_9969bdda_0123_func = NULL;  // CUDA函数句柄
// CUBIN二进制数据,包含编译后的CUDA内核代码
unsigned char CUBIN_NAME[10960] = { 0x7f, 0x45, 0x4c, 0x46, 0x02, 0x01, 0x01, 0x33, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0xbe, 0x00, 0x7c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xc0, 0x14.... };


// 卸载CUDA内核模块
void unload_add_kernel_9969bdda_0123(void) {
    CUDA_CHECK(cuModuleUnload(add_kernel_9969bdda_0123_mod));
}

// 加载CUDA内核模块
void load_add_kernel_9969bdda_0123() {
    int dev = 0;
    void *bin = (void *)&CUBIN_NAME;
    int shared = 0;
    // 加载CUBIN数据到CUDA模块
    CUDA_CHECK(cuModuleLoadData(&add_kernel_9969bdda_0123_mod, bin));
    // 获取add_kernel函数句柄
    CUDA_CHECK(cuModuleGetFunction(&add_kernel_9969bdda_0123_func, add_kernel_9969bdda_0123_mod, "add_kernel"));
    
    // 配置共享内存
    int shared_optin;
    // 获取设备支持的最大共享内存大小
    CUDA_CHECK(cuDeviceGetAttribute(&shared_optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, dev));
    // 如果需要的共享内存大于默认值且设备支持,则设置更大的共享内存
    if (shared > 49152 && shared_optin > 49152) {
      CUDA_CHECK(cuFuncSetCacheConfig(add_kernel_9969bdda_0123_func, CU_FUNC_CACHE_PREFER_SHARED));
      CUDA_CHECK(cuFuncSetAttribute(add_kernel_9969bdda_0123_func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_optin))
    }
}

/*
内核配置参数:
['BLOCK_SIZE=64', 'num_warps=1', 'num_stages=3']
*/
// CUDA内核启动函数
CUresult add_kernel_9969bdda_0123(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr output_ptr, int32_t n_elements) {
    // 如果函数未加载则先加载
    if (add_kernel_9969bdda_0123_func == NULL)
       load_add_kernel_9969bdda_0123();
    // 设置网格维度
    unsigned int gX = 1024;
    unsigned int gY = 1024;
    unsigned int gZ = 1024;
    // 准备内核参数
    void *args[4] = { &x_ptr, &y_ptr, &output_ptr, &n_elements };
    // 启动CUDA内核
    if(gX * gY * gZ > 0)
      return cuLaunchKernel(add_kernel_9969bdda_0123_func, gX, gY, gZ, 1 * 32, 1, 1, 0, stream, args, NULL);
}
```

![](https://files.mdnice.com/user/59/42787035-4c32-4088-bde8-31c793519bba.png)

这里展示了`add_kernel`相关编译产物(Artifacts)。包括Triton IR、Triton GPU IR、LLVM IR、PTX和CUBIN。同时作者还发现了一些有用的工具命令，比如使用`readelf`查看elf格式文件、用`cuobjdump`导出sass和ptx代码，以及使用`nvidisasm`工具让`cubin`文件变得可读。最后提到更多关于Python绑定的详细信息可以在博客系列的第2部分找到。这些内容主要是为了帮助开发者理解和分析CUDA kernel的编译过程和中间产物。

![](https://files.mdnice.com/user/59/5d9c54c6-199a-4d68-8f9a-8c4ad1b42712.png)

这张Slides展示了Triton另外一种即时编译的方式。代码示例展示了vector_add的PyTorch CUDA kernel的配置和执行过程：首先设定size参数，创建输入张量x和y（都在CUDA设备上），定义计算网格（grid），创建输出张量，然后编译并执行kernel。最后通过`compiled_kernel.asm.keys()`方法可以获取所有的代码生成键值，这些键值包括了各种中间表示形式（如'ttir'、'ttgir'、'ptx'、'cubin'等）。

![](https://files.mdnice.com/user/59/2685dcc4-c19d-408f-8b06-abbb6c23596d.png)

这张Slide介绍了 JIT Compiled Kernel (即时编译内核)的工作原理，它展示了从 Python DSL 开始，经过 IR、PTX 到 CUBIN/fatbinary 和 launcher.so 的多层代码生成过程。系统会通过 TRITON_CACHE_DIR 和缓存管理器(https://github.com/triton-lang/triton/blob/4348109b0a8e1aac748aa9b1bbbcd858e9488940/python/triton/runtime/cache.py#L50-L71)将编译结果保存到磁盘，并以 fatbinary (cubin) 格式内联加载。目标硬件都有各自的驱动程序，这些驱动提供了被 Python 模块封装的原生代码，并负责将 cubin 加载到 CUDA 驱动中(https://github.com/triton-lang/triton/blob/main/third_party/nvidia/backend/driver.c#L389-L406)。此外，还导出了 cuda_utils.so(https://github.com/triton-lang/triton/blob/main/third_party/nvidia/backend/driver.py#L72-L86) 和 triton_launcher.so(https://github.com/triton-lang/triton/blob/main/third_party/nvidia/backend/driver.py#L413-L426) 两个共享库，并提供了 compile_module_from_src(https://github.com/triton-lang/triton/blob/main/third_party/nvidia/backend/driver.py#L48-L64) 和 triton.runtime.build(https://github.com/triton-lang/triton/blob/main/python/triton/runtime/build.py#L21-L80) 两个重要的代码关键点。

![](https://files.mdnice.com/user/59/ccac255a-aba9-4828-abde-9669a6f1c0e8.png)

这张Slide主要介绍了Triton和MLIR(Multi-Level Intermediate Representation)的关系和基本概念。MLIR是在2022年完全重写的现代优化编译器基础设施，是LLVM生态系统的一部分。它包含中间表示(IR)规范和转换工具包，通过方言(dialects)机制来扩展MLIR框架(https://mlir.llvm.org/docs/Passes/)。TensorFlow是第一个使用MLIR的主要机器学习框架。在Triton中，所有方言都使用table-gen（一种DSL/代码生成工具）来处理MLIR的样板代码，并且可以通过设置`MLIR_ENABLE_DUMP=1`来dump每次编译过程中的IR信息。
- tablegen的优质资源推荐：https://www.jeremykun.com/2023/08/10/mlir-using-tablegen-for-passes/
- MLIR应用到深度学习框架的资源：https://www.youtube.com/watch?v=R5LLIj8EMxw

![](https://files.mdnice.com/user/59/1961ff7e-2fa9-45fb-9039-40e7765ab5f1.png)

![](https://files.mdnice.com/user/59/f34650de-eace-40bb-b434-89d6f9bc718e.png)

这里展示了MLIR(Multi-Level Intermediate Representation)的基本使用示例。第一张展示了MLIR的"Hello World"示例代码，其中创建了一个ModuleOp并使用PassManager添加了两个优化pass(CSEPass和DeadCodeEliminationPass)来处理模块。第二张则通过一个简单的Python函数优化示例，展示了如何使用MLIR进行代码优化 - 将一个包含多余计算的函数(a = b + c; e = b + c; d = e; return d)优化简化为更简洁的形式(a = b + c; d = a; return d)，体现了MLIR在代码优化方面的应用。

![](https://files.mdnice.com/user/59/8b343b94-d4dd-4135-ba2d-65e000972bfc.png)

这张Slide展示了MLIR的常见Pass(https://mlir.llvm.org/doxygen/namespacemlir.html)，我们可以将其应用到Triton中。

![](https://files.mdnice.com/user/59/2706a0b2-49d6-4185-aa03-a518d2002dc7.png)

这张slide介绍了Triton编译器中的几个重要优化Pass（编译优化阶段）：

- `TritonCombineOps`：用于合并基础运算操作，比如点乘和地址计算
- `TritonReorderBroadcast`：重排广播和分片操作的顺序，使其更高效
- `TritonRewriteTensorPointer`：移除tensor指针相关的操作
- `TritonLoopUnroll`：根据指定的因子展开循环结构

这些Pass都是Triton编译器用来优化GPU代码性能的重要步骤，通过这些优化可以生成更高效的GPU代码。可以参考这里的代码：https://github.com/triton-lang/triton/blob/576426bccfb9a2c90f2abaa405995738d4a79403/include/triton/Dialect/Triton/Transforms/Passes.td#L27

![](https://files.mdnice.com/user/59/32cd5984-2e88-4bab-960a-fcecfc180b2b.png)

这里展示了 Triton GPU 的编译优化流程（Passes）。代码显示了如何使用 Triton 的编译器优化管道来处理 GPU 代码。具体包括了几个重要的优化Pass，如线程本地化优化（optimize_thread_locality）、布局转换（layout_conversions）、矩阵乘法加速（accelerate_matmul）以及张量操作优化（optimize_dot_operands）等。这些优化passes的目的是为了提高 GPU 上代码的执行效率，是 Triton 编译器优化框架的核心组成部分。我们可以通过 https://github.com/triton-lang/triton/blob/main/include/triton/Dialect/TritonGPU/Transforms/Passes.td 和 https://github.com/triton-lang/triton/blob/main/include/triton/Dialect/TritonNvidiaGPU/Transforms/Passes.td 这两个Tablegen文件查到这Triton GPU 优化Pass的定义。 

其实从这两个文件的命名我们可以发现 Triton 在GPU Passes应该还在做通用和专用的抽象，比如 `TritonGPU` 和 `TritonNvidiaGPU` 的命名。

![](https://files.mdnice.com/user/59/d1ab5c5e-00f2-48c4-be85-b499ab47d2d6.png)

这里列了一些Triton GPU优化Pass：
- 内存合并访问（Coalescing）
- F32 点积运算优化
- CTA（合作线程阵列）规划
- 线程局部性优化
- 矩阵乘法加速
- 点运算操作数优化
- 数据去重
- 指令重排序
- TMA（Tensor Memory Access）lowering等优化

接下来我们可以浏览一下vector_add的GPU IR：

![](https://files.mdnice.com/user/59/5ebcd20f-08ee-4f50-906a-72cf6c28ae98.png)

![](https://files.mdnice.com/user/59/787f605e-fc73-43a4-a627-c7ab23e44308.png)

![](https://files.mdnice.com/user/59/5e84d4c4-bb03-4f0a-ad41-aff4c63e4598.png)

![](https://files.mdnice.com/user/59/7d62efeb-e1c5-44a8-91ea-0d605af28a66.png)

![](https://files.mdnice.com/user/59/7a75caf4-65cf-419c-ac5b-600f610b0b9a.png)

![](https://files.mdnice.com/user/59/3c776737-a18d-48b9-8a0b-ea64160ab7f6.png)

这里展示了一下triton vector_add lowering到gpu IR的结果，我们可以看到其中一些很容易对应到Python code的内容，例如：

1. Module Attributes:
- `triton_gpu.num-ctas = 1`：指定了一个 CTA (Cooperative Thread Array)
- `triton_gpu.num-warps = 4`：每个 CTA 包含 4 个 warp
- `triton_gpu.threads-per-warp = 32`：每个 warp 包含 32 个线程
- 目标平台是 "cuda:89"（计算能力 8.9 的 CUDA 设备）

2. Program ID and Range Creation:
- `%c1024_i32` 创建常量 1024
- `%0` 获取程序 ID
- `%1` 将程序 ID 乘以 1024
- `%2` 创建一个从 0 到 1024 的范围，生成一个 1024xi32 的张量

3. Ops (Splat and add operations):
- `%3` 使用 splat 操作将标量广播到 1024xi32 张量
- `%4` 执行加法操作
- `%13` 显示了浮点数加法操作 (addf)

4. Load and store operations:
- `%7-%9` 展示了数据加载操作：
- `splat` 操作将指针广播
- `addptr` 计算偏移后的地址
- `load` 从计算出的地址加载数据
- 最后使用 `tt.store` 将结果存回内存
- `tt.return` 标志着 kernel 的结束

可以看到这个IR展示了向量加法操作的底层实现步骤。

![](https://files.mdnice.com/user/59/4746e403-53ec-4b2c-9a39-f212ea580290.png)

![](https://files.mdnice.com/user/59/3da42a13-f5b5-4dff-9c05-5c4167785a6c.png)

如果你想进一步了解优化Pass的实现，例如TritonGPUAccelerateMatmul这个Pass，你在Triton代码仓库搜索的时候应该能看到3个相关的地方(https://github.com/search?q=repo%3Atriton-lang%2Ftriton%20TritonGPUAccelerateMatmul&type=code)：

![](https://files.mdnice.com/user/59/9087e6f0-47d5-462e-98ba-683f94a8c8b2.png)

分别是这个Pass的Tablegen定义，具体的MLIR实现和Python Binding。

接下来作者简单展示了一下在Triton的codebase基础上实现了2个新的不需要传参数的简单Pass，打印OpGraph和记录Op数量的Pass，大家感兴趣也可以尝试下。

![](https://files.mdnice.com/user/59/c321dbfb-0a2b-4ae0-b583-15bdac38c5e2.png)

## 总结

这节课详细介绍了 Triton 编译器的内部工作原理。文章首先介绍了 CUDA 编译器（NVCC）的工作流程，然后深入探讨了 Triton 编译器的架构设计，包括其如何将 Python DSL 代码通过多个中间表示（IR）最终编译成 CUDA 可执行文件。课程重点讲解了 Triton 与 MLIR（Multi-Level Intermediate Representation）的关系，展示了各种优化 Pass 的实现和作用，如 TritonCombineOps、TritonReorderBroadcast 等。通过一个 vector_add 的具体示例，详细展示了代码从 Python 层面到 GPU IR 的转换过程，以及在这个过程中各种中间产物（如 .cubin、.ptx 等）的生成。最后还介绍了如何在 Triton 中实现新的编译器 Pass，为读者提供了深入理解和扩展 Triton 编译器的实践指导。

