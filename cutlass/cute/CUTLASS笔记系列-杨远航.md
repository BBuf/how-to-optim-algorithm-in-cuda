> 知乎 https://www.zhihu.com/column/c_1938664963049763058 专栏笔记拷贝。汇总后连续看。

# CUTLASS 笔记：导读

![](https://files.mdnice.com/user/59/8c3adf94-000a-4a09-a0dd-2573176d6f9d.png)

本 CUTLASS 笔记系列将从一个最小的 Minimal GEMM 开始，逐步扩展 CuTe、CUTLASS 各类组件和 Hopper、Blackwell 等新架构的特性，最终实现一个高性能的 GEMM 融合算子。

## 1. 前言

众所周知，CUTLASS 因其开源、灵活、高性能的特点，广泛应用于自定义 CUDA 算子开发和性能优化场景，例如 Pytorch、vLLM、FA2、FA3 等常用框架和算子库均使用了 CUTLASS 开发。但其学习曲线陡峭，且需要对 CUDA、C++ 语法及特性较为熟悉，普通算法开发者使用起来难度较大，这也是 triton 等 Python DSL 流行的原因之一。然而要追求性能上限，同时对 kernel 中计算、通信、存储的各种细节有灵活控制的需求，CUTLASS 仍然是不可替代的。并且学习 CUTLASS 本质上就是在学习 CUDA / PTX，即使考虑到开发成本后需要写 triton，在有 CUDA 知识储备和优化方法论的情况下，写 triton kernel 也能得心应手。如果只会 triton，遇到性能问题和不符预期的结果，想要深入排查问题是极其困难的。此外，CUTLASS 4.0 开始推出的 Python DSL，在原汁原味保留了 C++ 版本的 API 之外提供了 Python 接口，也是一种高性价比的开发方案。无论从何种角度说，学习 CUTLASS 对于算子开发人员来说都是必要的。目前有非常多优秀的 CUTLASS 文章（见下面的教程推荐部分），但我觉得现在还没有一个**非常完整且系统的，从零开始循序渐进直到能在最新硬件架构上写出 optimized kernel，既适合新手学习，又适合老手时常翻阅参考的 CUTLASS 内容**，所以才有了这个系列文章。

## 2. CUTLASS 简介

一句话介绍：**CUTLASS 以模板库和众多可复用组件的形式，提供了围绕 GEMM 的算法开发和优化的解决方案。**

![图1: CUTLASS GEMM Hierarchy](https://files.mdnice.com/user/59/8c3adf94-000a-4a09-a0dd-2573176d6f9d.png)


由于 NV 硬件下的 Tensor Core 性能强大，且当前主流的计算任务均离不开基础的 GEMM 算子，如何方便开发者实现高效的 GEMM 运算，以及 GEMM 和其他计算的 overlap 和融合，发挥出硬件的算力上限，从而在各种上层任务上提质增效，就是 CUTLASS 要解决的问题。

CUTLASS 的一大优点的是它完全白盒！在复杂的 C++ 模板下，CUTLASS 本质就是对 PTX 指令的封装，因此我们不需要学习使用各种复杂的 CUDA Runtime API（然后猜测它背后做了什么），只需了解 PTX 指令并使用对应的 CUTLASS API，就可以近乎完全地掌控硬件行为，这对于研究和工程都非常重要！
、
## 3. Why CUTLASS？

在 2025 年，要写一个高性能算子或者融合算子，已经有非常多解决方案了。这里列出了三种主要的实现路径：

- 基于自动编译的路径。典型例子是 torch.compile；
- 基于 Python DSL + PTX 编译器的路径。像 triton、CuTe DSL、TileLang、Mojo 等等都是走了这条路径；
- 基于 C++ 模板封装 PTX 的路径。例如 CUTLASS、Thunder Kittens 等。

（当然还有手搓 PTX 的路径 hhh，例如 https://github.com/xlite-dev/LeetCUDA，手动 @DefTruth）

对于各个实现路径的优缺点，已有许多文章进行了相关讨论。从我个人的角度看，相比于其他方案，CUTLASS 独特的优点就是**可控且灵活**：由于 CUTLASS 背后就是 PTX 指令，我可以在写代码的过程就可以感知到硬件会如何运行这段代码，并且当新的算法和指令集出来后，我可以方便地扩展 CUTLASS 的能力，在已有组件的基础上新增某种 scheduler，新写一段 GEMM 编排方式，或者融入新的指令。因此除了 nvcc、PTX、SASS，再没有其他能够约束算子开发的东西了！

如果使用 triton，要么得等开发团队更新 DSL 然后我去调用接口，要么就得自己改编译器，自己写 pass 了。我认为 triton 在提升开发效率的同时，牺牲了灵活性，且随着 NV 的架构 DSA 化，triton 的语法也越来越复杂，所以个人不是特别喜欢 triton，而是更加青睐 TileLang 和 CuTe DSL 的使用方式。

当然，如果希望深入算子开发和优化，CUTLASS 应该是必须要掌握的，如果这个笔记系列对您有帮助，那是我的莫大荣幸。

## 4. 前置知识

本 CUTLASS 笔记系列的宗旨是让零基础的读者也能学懂 CUTLASS，但这里还是列出前置的知识储备吧（如果大多数都不会，那就需要猛猛补课啦）：

- 了解 NV GPU 的编程模型和 SIMT 线程并行，例如线程层级（grid、block、warp、thread）和内存层级（GMEM、SMEM、Register）的基础概念，了解 CPU 和 GPU 执行计算的区别；
- 熟悉 Python、C++ 基础语法，在此之上最好能读懂 C++17 标准下的 C++ 模版语法；
- 了解 CUDA C++ 的扩展语法，例如 __device__ 和 __global__ 的区别、threadIdx 和 blockIdx 的使用方法等等；
- 会写一个简单的 CUDA kernel（例如 element-wise 向量加）。

## 5. 笔记目录

Part 1: CUTLASS CuTe 基础知识详解，使用 SM80 以及之前架构的特性实现性能最优的 GEMM kernel。
...
CUTLASS 笔记 (7)：SMEM Swizzling

CUTLASS 笔记 (8)：Dynamic MMA

CUTLASS 笔记 (9)：Pipelining

CUTLASS 笔记 (10)：CUTLASS GEMM API

Part 2: CUTLASS 进阶内容，详解 SM90 Hopper 架构新特性，在 H 卡上实现性能最优的 GEMM kernel。

CUTLASS 笔记 (11)：TMA load/store

CUTLASS 笔记 (12)：TMA multicast reduce

CUTLASS 笔记 (13)：Warpgroup MMA

CUTLASS 笔记 (14)：Warp Specialization

## 6. 教程推荐

这里推荐下优秀的 CUTLASS 文章作者：

- @reed 
    - 中文社区最好的 CUTLASS 教程，相信包括我的许多人也是从这些文章开始学习 CUTLASS 的。不夸张的说，如果没有 reed 佬的珠玉在前，估计也不会有这个笔记系列了。
- Colfax Research：https://research.colfax-intl.com/blog/
    - 作为 FA3 主要开发团队，Colfax 的博客是我看到的外网最好的 CUTLASS 教程，我也从这些博客中学习了大量 Hopper 特性和代码样例。
- @进击的Killua
    - 有丰富的代码示例和图片讲解，适合新手快速入门和使用 CUTLASS。
- @Anonymous
    - 许多文章对 CUTLASS API 和 PTX 指令的使用细节和行为研究地非常深入，对 GPU 体系结构有独到的理解。
- CUTLASS Discussions 讨论区：https://github.com/NVIDIA/cutlass/discussions
    - CUTLASS 开发者在 Discussions 的回复非常积极，也能从许多讨论中学习到源码和博客看不到的细节，非常推荐在 Discussions 讨论区提出你遇到的疑惑/问题。
当然由于个人精力有限，一定有其他优秀的文章我没发现，上面的推荐列表难免挂一漏万，欢迎大家推荐你觉得好的文章和作者！

# Extra：基础知识参考

## 1）SM 架构，CUDA core 与 Tensor Core

NV GPU 中的 SM 架构经过了若干代的演进。下图展示了从 Pascal 架构到最新 Blackwell 架构的 SM 结构：

![图2: 从 Volta 到 Blackwell 的 SM 架构演进](https://files.mdnice.com/user/59/102c3f46-f964-4be0-99ca-212b080e3f47.jpg)

我们重点关注其计算单元：
- Pascal 架构的计算单元是 Unified Int32 & FP32 Core，单 Core 可以执行整数和浮点计算；
- Volta、Ampere、Hopper 分离了 Int32 和 FP32 计算单元，新增 FP64 计算单元，同时从 Volta 架构开始新增了 Tensor Core 计算单元；
- Blackwell 架构又将 FP32 和 Int32 单元统一了，Tensor Core 升级至第五代。

我们常用 CUDA Core 的个数衡量一个 GPU 的运算能力，但 SM 架构中并没有写出哪些是 CUDA Core。借用 NV 论坛上的一句话：“CUDA core” is a marketing term, not a technical term. 一般人们将使用最频繁的 FP32 计算单元称为 CUDA Core，但为方便表述，**这个笔记系列中的 CUDA Core 指代除 Tensor Core 之外的其他计算单元（包括 Int、FP、SFU 单元等）**。


CUDA Core 一般用于执行标量和向量的数值运算，还会负责地址计算、线程常量相关计算（如 blockIdx、threadIdx）等辅助工作。对于常用的矩阵运算，CUDA Core 也可以通过循环的方式实现计算：

```c++
__global__ void mm(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;

        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}
```

当然对于 GPU 而言，写一个高效的 SGEMM kernel 需要考虑到 thread hierarchy 和 memory hierarchy，选择合适的 block tile 和 thread tile，并利用好 shared memory、向量化访存指令等特性。

CUDA Core 是较为通用的计算单元，可以用于执行各类计算任务，编写程序时需要以单个数值的粒度去考虑如何组织计算。随着 Tensor Core 的出现，我们关注的点就从单个数值变为 MxN 的单元矩阵了，编程时（不包括写 PTX）也可以站在单元矩阵的角度上编排计算了，而硬件指令的 DSA 化更凸显了张量运算的核心地位。最新的 Blackwell 架构上，我们甚至可以只在 1 个线程上完成所有 Tensor Core 的计算调度。

由于在矩阵计算任务上，Tensor Core 远比 CUDA Core 高效，因此当今所有 GEMM 算子工作均围绕着最大化利用 Tensor Core 来优化，其中就包括了 CUTLASS。

第一代 Tensor Core（Volta）在一个时钟周期内可计算 4*4 规模的 FP16 MMA 运算，算力为 2*4*4*4 = 128 FLOPs/cycle。后续每一代的 Tensor Core 相比前一代的算力都翻一倍。最新的 Blackwell 架构的Tensor Core 已经发展到第五代了，单个 TC 算力为 2048 FLOPs/cycle。

![图2: Volta 架构下的单指令 Tensor Core 计算](https://files.mdnice.com/user/59/9d2347e2-1b8c-4e42-b71b-c6c492ea16f9.png)

实际上，我们可以根据公开数据，手动计算出当前 NVIDIA 硬件的理论算力，也就是所有 Tensor Core 最大算力之和，计算公式为：

**理论算力 = Tensor Core 时钟频率 x 单个 Tensor Core 算力 x SM 个数 x 单个 SM 中 Tensor Core 个数**

以下是几种常用硬件的性能指标（其中 B200 硬件具体 spec 未公开，数据仅供参考）：

![](https://files.mdnice.com/user/59/2b5eaab1-b86e-449d-8988-6a3e757575c1.png)

此外在使用 CUTLASS 的过程中，我们常会遇到 PTX 相关指令。一般而言，涉及 CUDA Core 的计算会使用通用的计算指令集，如 add、sub、sin、cos、ex2 等，而涉及 Tensor Core 的张量计算则使用与 SM 架构强相关的特殊指令集，如 mma、wgmma、tcgen05 等。我们会在后续的笔记中分析代码使用了哪些特殊的 PTX 指令，以及它们的具体功能。

# CUTLASS 笔记 (1)：Minimal GEMM Kernel

本篇将详细介绍 CUTLASS CuTe 的基本组件和使用方法，并从零开始用 CuTe 写一个单 MMA 指令 16x8x8 的 GEMM kernel。此外还将介绍如何在 Python 中调用算子，如何完成精度验证和性能测试，以及如何使用 Nsight Compute、ncu 等工具对算子进行分析

本篇所使用的 CUTLASS 版本为 4.1.0，硬件架构为 SM90。

## 1. CuTe 基础组件

CUTLASS 3.0 起推出了 CuTe 库，它提供的 Layout 和 Tensor 抽象可以让我们聚焦于算法逻辑的开发，减轻了开发算子的心智负担，因此我们首先来介绍 CuTe 中的两大关键组件：Tensor 与 Layout。

### 1.1 Tensor 和 Layout

CuTe 中的 Tensor 和 Pytorch 的 Tensor 非常类似，都表示了一个张量的存储对象，并提供各类重载方法来方便执行计算。Tensor 中的张量在内存的存储结构就是一种 Layout，它包括了 Shape 和 Stride 两部分，其中 Shape 表示张量的形状，Stride 表示张量在每个维度的连续性。知道了 Tensor Layout，我们就可以知道张量中的每个元素在内存中的排布是什么样的了。

一般而言，CuTe 中的 Layout 是一种**映射关系**，在不同的语境下有不同的含义。**Tensor Layout 是我们接触的第 1 种 Layout，它表达了 tensor 坐标与内存地址 offset 之间的映射关系。**

![图1: CuTe 的第 1 种 Layout —— Tensor Layout](https://files.mdnice.com/user/59/35df7f28-250f-4fed-96fb-5ba5d787200d.png)


在 CuTe 中，我们将 Layout 表示为 **shape : stride** 的格式，其中 shape 和 stride 可以是一个 tuple，也可以是嵌套着 tuple 的 tuple。**Layout 的可嵌套性是它不同于 Pytorch Tensor 的重要特点**。有了嵌套表示，我们可以创建许多更为复杂的 Tensor pattern。

![图2: CuTe 中的嵌套 Layout](https://files.mdnice.com/user/59/1d6312e5-c2a4-48d0-928c-1ad9c6de76c7.png)


我们遇到的第 1 个 CuTe API 就是创建一个 Tensor 的方法：`make_tensor`，传入的三个参数分别为 data_ptr, shape, stride（也可以不提供 shape 和 stride，直接传入 layout）。**当不提供 stride 时，CuTe 会默认创建 left-major 的 stride，而 Pytorch 默认是 right-major 的，这是 CuTe Tensor 与 Pytorch Tensor 的第二个不同点。**

![图3: CuTe API —— make_tensor](https://files.mdnice.com/user/59/832151f4-0e32-458a-b52e-533df5e2fe77.png)


CuTe Tensor 的维度习惯地称为 mode，例如最左侧的维度就是 first mode / 0th mode，而嵌套 Layout 的各个维度称为 sub mode。我们可以通过 size<mode>(tensor) 的方式获取 tensor 各个维度的大小。

### 1.2 Tiling API

在较大规模的 GEMM 计算中，我们需要将其分块处理，在各层级存储大小的约束下高效地实现并行计算，一般我们将这种分块处理称为 tiling。在 CuTe 中也有对 Tensor 进行分块的 API：`local_tile`。

![图4: CuTe API —— local_tile](https://files.mdnice.com/user/59/fba3aed3-6842-4d99-9948-1712323fe21b.png)

给定每个 tile 的 shape 大小，我们可以将一个 Tensor 切分为若干小 Tensor（tile），并用一个坐标 coord 来获取其中一个 tile。

```c++
Tensor gA = local_tile(mA, make_shape(Int<kTileM>{}, Int<kTileK>{}), make_coord(0, 0))
```

我们也可以用一个高维的 tiler，并传入 Step 在指定维度上进行分块，这样多个分块处理可以复用同一个 tiler 和 coord。

```c++
  auto tiler = make_tile(Int<kTileM>{}, Int<kTileN>{}, Int<kTileK>{});
  auto coord = make_coord(0, 0, 0);

  Tensor gA = local_tile(mA, tiler, coord, Step<_1,  X, _1>{});
```

> Note: `make_tile` 和 `make_coord`，包括上面的 `make_shape` 和 `make_stride`，最终返回的都是一个 `cute::tuple` 类型的值，而 `Tile`、`Coord`、`Shape`、`Stride`、`Step` 类都是 `cute::tuple` 的别名，因此可以用相同的方法使用它们。

一般情况下，`local_tile` 用于从一个完整的 GEMM 中获取一个 block 需要计算的矩阵分片，而 block 内部的 tiling 依赖于 MMA 计算指令，需要交给 MMA API 处理。


### 1.3 MMA API

MMA 指矩阵乘加运算（Matrix Multiply-Accumulate），公式为 D = A * B + C，而通常的 GEMM 运算是 MMA 的子集，可以表示为 D = A * B。Tensor Core 提供了若干特定 shape 大小的 MMA 计算指令，包括本篇要实现的 16x8x8 MMA。所有 MMA 指令及其对应的 shape、sparsity、precision 可参考 PTX 文档：https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-shape 

CuTe 中的 MMA_Atom 对象对应了一个特定的 mma 指令，例如我们需要完成 16x8x8 的 MMA 运算，且所有数值精度均为 FP16，那么我们可以创建一个 MMA op：

```c++
using MMA_op = SM80_16x8x8_F16F16F16F16_TN;
```

使用 CUDA Core 进行矩阵运算时，本质上是每个线程独立完成矩阵元素的乘加，每个线程每次循环的计算指令只涉及一次 mul + add 指令。不同的是，Tensor Core 对应的 **mma 指令集要求多个线程协同完成计算**。在单指令 16x8x8 MMA 的场景下，我们使用的 mma 指令如下：


```c++
mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16
  {%Rd0, %Rd1},
  {%Ra0, %Ra1},
  {%Rb0},
  {%Rc0, %Rc1};
```

这个指令要求一个 warp 的 32 个线程协同完成 16x8x8 的 MMA 计算，每个线程需要拿到 4 个矩阵 A 的元素、2 个矩阵 B 的元素和 4 个矩阵 C 的元素进行计算，计算完成后保存 4 个矩阵 D 的元素。**只有当每个线程拿到正确的矩阵元素，并将计算结果存放于正确的线程中，才能正确完成一个 MMA 计算。**


在 PTX 文档中详细记录了每个 mma 指令相应的矩阵元素和每个线程中寄存器的映射关系，例如上面的 mma 指令的映射关系可见：https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-1688

我们可以用 CuTe 打印出这个 16x8x8 mma 指令的映射关系如下：

![图5: Minimal GEMM kernel 中的 Tiled MMA](https://files.mdnice.com/user/59/0f1499bf-3b23-4132-adde-9fbd33a523c2.png)

其中左下矩阵为 A，右上矩阵为 B，右下矩阵为 C/D。每个矩阵元素中的 TxVy 表示该元素是线程 x 的第 y 个数据。注意在上图中，矩阵 A 的 shape 为 MxK，矩阵 B 的 shape 为 KxN，所有矩阵均为 K-major。

如果要手搓 PTX，我们就需要依据上图，让每个线程从 A/B/C 矩阵中获取相应元素放置于寄存器中，再将寄存器喂给 mma 指令，最后将结果写回矩阵 D 的对应位置。如果换用不同的 mma 指令，这个映射关系也要相应修改。显然这是一件极为繁琐的事。

幸运的是，在 Layout Algebra 的加持下，CuTe 提供的 MMA API 帮我们建立了这些复杂的映射关系。我们只需选取正确的 MMA op，并传给 `make_tiled_mma`，CuTe 会自行找到 MMA op 对应的映射关系。

```c++
using TiledMMA = decltype(make_tiled_mma(MMA_op{}));
```

我们可以在 kernel 创建 TiledMMA 对象，并通过 `get_slice` 拿到对应线程的 tiler（即 CuTe 的 ThrMMA 实例）。调用这个 tiler 的 `partition_A` 方法，就拿到了该线程完成 MMA 计算所需的 A 矩阵元素的 Tensor 表示，这个 Tensor 表示了 global memory 上 A 矩阵对应到这个线程的分片。相应还有 `partition_B`、`partition_C` 方法，它们的作用类似。

```c++
TiledMMA tiled_mma;
  ThrMMA thr_mma = tiled_mma.get_slice(tid);
  Tensor tCgA = thr_mma.partition_A(gA);  // (MMA, MMA_M, MMA_K)
```

ThrMMA 还有一个 partition_fragment_A 方法，它返回的 Tensor 的 shape 和 partition_A 相同，但这个 Tensor 不表示 global memory 的数据，而是表示该线程内的一组连续的寄存器。

```c++
Tensor tCrA = thr_mma.partition_fragment_A(gA);  // (MMA, MMA_M, MMA_K)
```

### 1.4 Copy API 与 GEMM API

我们可以用 CuTe 提供的 Copy API 完成数据的拷贝。例如下面的代码完成了数据从 global memory 到寄存器的拷贝：

```c++
auto copy_atom = AutoVectorizingCopy{};
copy(copy_atom, tCgA, tCrA);
```

![图6：GMEM 到 Register 的拷贝](https://files.mdnice.com/user/59/7baad795-2988-40ad-8fbc-64caec770b2f.png)

其中 copy_atom 对应了数据拷贝所使用的指令。为了最大化访存效率，我们希望一条拷贝指令能拷贝尽可能多的连续内存（即**向量化访存**），常规单条拷贝指令最多可拷贝 128 bits 的数据，但很多情况下需要拷贝的数据并不连续，因此 AutoVectorizingCopy 可以让 CuTe 根据 MMA 自动选取最大的连续数据长度，以此确定具体的拷贝指令。

数据就绪后，我们可以调用 CuTe GEMM API 进行 mma 的计算：

```c++
gemm(tiled_mma, tCrD, tCrA, tCrB, tCrC);
```

随后，我们可以将结果写回 global memory：

```c++
copy(copy_atom, tCrD, tCgD);
```

![图7：Register 到 GMEM 的拷贝](https://files.mdnice.com/user/59/42139c80-059f-4667-855b-a5f480669e39.png)

接下来我们将使用上面的基础 API，实现一个单指令的 16x8x8 规模的 MMA 计算。

## 2. 编写 Minimal GEMM kernel


在编写 Minimal kernel 前，我们首先要确定算子的各类细节，包括问题规模、grid 的切分，每个 block 的线程个数、各个 tile 的维度等等，这有助于把握编写算子的核心步骤。

本场景下的算子详情如下表所示：

![](https://files.mdnice.com/user/59/bf5b70a0-7329-4173-a138-c90004f3e1f6.png)

由于我们仅用单条指令计算 MMA，因此只需要启动一个 block 的 32 个线程即可。在本场景下无需做分级 tiling，因此所有 tile shape 均等于单指令 MMA atom shape。

### 2.1 Kernel Spec 参数类

开发算子的过程中有许多类似于上面表格中的常量，通常我们把它们放在统一的一个参数类，便于归类和修改，且修改参数不会影响到 device kernel 的逻辑。

```c++
template <typename T_, int kTileM_ = 16, int kTileN_ = 8, int kTileK_ = 8>
struct KernelSpec {
  using T = T_;

  static constexpr int kTileM = kTileM_;
  static constexpr int kTileN = kTileN_;
  static constexpr int kTileK = kTileK_;

  using MMA_op = SM80_16x8x8_F16F16F16F16_TN;
  using TiledMMA = decltype(make_tiled_mma(MMA_op{}));

  static constexpr int kThreadNum = size(TiledMMA{});
  static constexpr int kShmSize = 0;
};
```

### 2.2 编写 kernel 代码

为简化场景，Minimal GEMM kernel 只完成两类计算：C = A * B 和 C = A * B + C。通过传入不同的模板参数，我们可以选用不同的 kernel 完成不同的计算模式。我们还可以将上面的参数类通过模板传入 kernel。

kernel 的函数签名如下：

```c++
template <typename Spec, bool IsGemm>
__global__ void
minimal_gemm(void *Cptr, const void *Aptr, const void *Bptr, int m, int n, int k);
```

首先，我们建立起 A、B、C 三个矩阵的 Tensor 表示：

```c++
Tensor mA = make_tensor(make_gmem_ptr((T *)Aptr),
                        make_shape(m, k),
                        make_stride(k, Int<1>{}));  // (M, K)
Tensor mB = make_tensor(make_gmem_ptr((T *)Bptr),
                        make_shape(n, k),
                        make_stride(k, Int<1>{}));  // (N, K)
Tensor mC = make_tensor(make_gmem_ptr((T *)Cptr),
                        make_shape(m, n),
                        make_stride(n, Int<1>{}));  // (M, N)
```


我们规定传入 kernel 的三个矩阵都是行连续的，因此 stride 的最后一个维度均为 1。`make_gmem_ptr` 用于表明指针对应的数据位于 GMEM 上。mA、mB、mC 即代表整个问题规模的 A、B、C 矩阵。

随后，我们需要从完整的 A、B、C 矩阵中，获取这个 block 计算所需的分块矩阵：


```c++
auto tiler = make_tile(Int<kTileM>{}, Int<kTileN>{}, Int<kTileK>{});
auto coord = make_coord(0, 0, 0);

Tensor gA = local_tile(mA, tiler, coord, Step<_1,  X, _1>{});  // (kTileM, kTileK)
Tensor gB = local_tile(mB, tiler, coord, Step< X, _1, _1>{});  // (kTileN, kTileK)
Tensor gC = local_tile(mC, tiler, coord, Step<_1, _1,  X>{});  // (kTileM, kTileN
```

在单指令 MMA 的场景下，tiler 的大小和问题规模一致，因此仅切分出 (0, 0, 0) 这一个 tile，我们直接获取即可。gA、gB、gC 代表该 block 的分块矩阵，前缀 g 代表 global memory。

在 Block 内部，我们使用 MMA API 对 global memory 上的分块继续做切分：

```c++
TiledMMA tiled_mma;
ThrMMA thr_mma = tiled_mma.get_slice(tid);

Tensor tCgA = thr_mma.partition_A(gA);  // (MMA, MMA_M, MMA_K)
Tensor tCgB = thr_mma.partition_B(gB);  // (MMA, MMA_N, MMA_K)
Tensor tCgC = thr_mma.partition_C(gC);  // (MMA, MMA_M, MMA_N)

Tensor tCrA = thr_mma.partition_fragment_A(gA);  // (MMA, MMA_M, MMA_K)
Tensor tCrB = thr_mma.partition_fragment_B(gB);  // (MMA, MMA_N, MMA_K)
Tensor tCrC = thr_mma.partition_fragment_C(gC);  // (MMA, MMA_M, MMA_N)
```

上面的代码有两处需要说明的点：

**1）Tensor 的命名习惯**。在 CUTLASS 中，除上面的 mA、gA 以及在共享内存的 sA 等这些命名习惯外，我们常用 txgy、txry 等名称指代一个经过 tiling 的 tensor。其中 t 代表 tiling，x 代表按照何种方式进行 tiling，由于整个 MMA 最终产出的是 C 矩阵，tC 代表着这个 tensor 是从计算 C 矩阵的 MMA tile 出来的。第三个字母是 g/r 表示 tensor 数据的存储位置是 global memory/register file，第四个字母指代矩阵名称。许多 CUTLASS 相关代码均遵循这一套命名规范。


**2）注释中 Tensor shape 的含义**。此处的 Tensor 的 shape 均为 (MMA, MMA_M/N, MMA_K/N) 。第一个维度 MMA 表示单个 MMA 指令（MMA Atom）所需的矩阵元素个数，本场景下，tCgA/tCrA 的 MMA 为 4，tCgB/tCrB 的 MMA 为 2。后两个维度代表了 MMA Atom 扩展后的维度，这里并没有对 MMA 做扩展，因此后两个维度均为 1。CUTLASS 代码中常常对 Tensor 的 shape 进行注解，以方便阅读代码。


同时，CuTe 提供了展示 Tensor 的 print 函数，可以展示特定 Tensor 的详细信息，而 print_tensor 函数可以打印出特定 Tensor 的所有数据，它们都是调试的好工具。（这里考考大家，tCgA 和 tCrA 的 stride 是什么呢？如果不清楚，可以尝试 print 一下哦）

```c++
if (thread0()) {
  print(tCgA); printf("\n");
  print_tensor(tCgA); printf("\n");
}
```

完成 tiling 后，我们可以执行实际的拷贝和计算工作了：

```c++
auto copy_atom = AutoVectorizingCopy{};

copy(copy_atom, tCgA, tCrA);
copy(copy_atom, tCgB, tCrB);

if constexpr (IsGemm) clear(tCrC);  // Set the accumulators to zero
else copy(copy_atom, tCgC, tCrC);

gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);

copy(copy_atom, tCrC, tCgC);
```

可以发现，如果我们计算的是 C = A * B，那么无需拷贝 C 矩阵，而是通过 clear 函数将 C 的 tiling Tensor，也就是 accumulator 设置为 0。

至此，一个最小的 Minimal GEMM 算子就完成了。完整的 kernel 代码可见 Github 仓库：cutlass-notes

## 3. 使用 Minimal GEMM kernel

### 3.1 编写 Pytorch binding

为了在 Pytorch 使用 Minimal GEMM kernel，我们还需要写一个连接 kernel 和 Python 的函数，它只需要接受 a，b，c 三个 torch::Tensor 即可，其中 c 为可选参数。函数签名如下：

```c++
template<typename ComputeType, typename AccType = ComputeType>
torch::Tensor
run_minimal_gemm(const torch::Tensor &a,
                 const torch::Tensor &b,
                 std::optional<torch::Tensor> &_c);
```

在函数内，我们通常使用 Pytorch 提供的 C++ 接口 libtorch 来进行 kernel 前的预处理和预检测。本场景最重要的步骤是区分 MM 和 MMA 场景，我们可以写如下判断来为 c 设置初始值：

```c++
torch::Tensor c;
bool is_gemm;

if (!_c.has_value()) {
  auto options = torch::TensorOptions().dtype(torch_acc_type).device(torch::kCUDA);
  c = torch::empty({M, N}, options);
  is_gemm = true;
} else {
  c = _c.value();
  is_gemm = false;
}
```

若未传入 c tensor，则创建一个空 tensor，且设置 `is_gemm` 为 true，反之则正常传入 c，并设置 `is_gemm` 为 false。

随后根据 is_gemm 的值，判断使用哪个 kernel 的实现，并启动 kernel。

```c++
BOOL_SWITCH(is_gemm, IsGemm, [&] {
  cudaEventRecord(start, stream);
  minimal_gemm<Spec, IsGemm><<<grid, block, shm_size, stream>>>(
    reinterpret_cast<AccType*>(c.data_ptr()),
    reinterpret_cast<ComputeType*>(a.data_ptr()),
    reinterpret_cast<ComputeType*>(b.data_ptr()),
    M, N, K
  );
  cudaEventRecord(stop, stream);
});
```

最后使用 pybind11 提供 Python 接口：

```c++
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("minimal_gemm", &(run_minimal_gemm<cute::half_t>), "Run a single 16x8x8 MMA operation.");
}
```

在 Python 侧，我们可以用 Pytorch 提供的接口即时编译算子，并加载编译后的动态库。算子的使用方式如下：

```python
a = torch.randn(M, K, device="cuda", dtype=torch.half)
b = torch.randn(N, K, device="cuda", dtype=torch.half)
c = torch.randn(M, N, device="cuda", dtype=torch.half)

# Case 1: MM
kernel_output = lib.minimal_gemm(a, b, None)

# Case 2: MMA
kernel_output = lib.minimal_gemm(a, b, c)
```

完整的 Python 用户代码可见 Github 仓库：cutlass-notes

### 3.2 精度验证与性能测试

算子开发结束后，均需要进行精度验证和性能验证。

精度验证可以将 Pytorch 的计算结果作为 base，比较我们的 kernel 和 torch 输出结果的最大差值（Max Diff），平均差值（Mean Diff）和相对误差（Relative Error），输出比较可参考如下代码：

```python
def relative_error(target: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8):
    diff = target - ref
    norm_diff = torch.norm(diff, p=2)
    norm_diff_ref = torch.norm(ref, p=2)

    return (norm_diff / (norm_diff_ref + eps)).item()

def compare_matrix(kernel_output: torch.Tensor, torch_output: torch.Tensor):
    kernel_output = kernel_output.float()
    torch_output = torch_output.float()

    max_diff = torch.max(torch.abs(torch_output - kernel_output))
    mean_diff = torch.mean(torch.abs(torch_output - kernel_output))
    re = relative_error(kernel_output, torch_output)
    is_correct = re < 0.001

    if not is_correct:
        print(
            f" Kernel Output: {tuple(kernel_output.shape)} ".center(PRINT_LENGTH, "-")
        )
        print(kernel_output[:8, :8])

        print(f" Torch Output: {tuple(torch_output.shape)} ".center(PRINT_LENGTH, "-"))
        print(torch_output[:8, :8])

    print(
        f" Result: {'Success' if is_correct else 'Failed'}, Max diff = {max_diff:.5f}, Mean diff = {mean_diff:.5f}, RE = {(re * 100):.2f}% ".center(
            PRINT_LENGTH, "-"
        )
    )
```

性能验证上，我们可以使用 CUDA Event 来计时 kernel，具体而言，我们需要在 launch kernel 前后插入 event，同步 CUDA 后可以在 CPU 侧打印运行时间（不包括 kernel launch 时间），代码样例如下：

```c++
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaDeviceSynchronize();

// Kernel launch
BOOL_SWITCH(is_gemm, IsGemm, [&] {
  cudaEventRecord(start, stream);
  minimal_gemm<Spec, IsGemm><<<grid, block, shm_size, stream>>>(
    reinterpret_cast<AccType*>(c.data_ptr()),
    reinterpret_cast<ComputeType*>(a.data_ptr()),
    reinterpret_cast<ComputeType*>(b.data_ptr()),
    M, N, K
  );
  cudaEventRecord(stop, stream);
});

cudaDeviceSynchronize();

auto error = cudaGetLastError();
if (error != cudaSuccess) {
  throw std::runtime_error(
    std::string("CUDA error: ") + cudaGetErrorString(error) +
    " (error code: " + std::to_string(error) + ")");
}

float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
printf("Kernel execution time: %.3f ms\n", milliseconds);

cudaEventDestroy(start);
cudaEventDestroy(stop);
```

运行代码，可得到如下结果：

```shell
------------------------------------------ M=16, N=8, K=8 ------------------------------------------
Block Size: (32, 1, 1) | Grid Size: (1, 1, 1) | Shared Memory Size: 0 Bytes
Kernel execution time: 0.008 ms
--------------- Result: Success, Max diff = 0.00000, Mean diff = 0.00000, RE = 0.00% ---------------
Block Size: (32, 1, 1) | Grid Size: (1, 1, 1) | Shared Memory Size: 0 Bytes
Kernel execution time: 0.008 ms
--------------- Result: Success, Max diff = 0.00000, Mean diff = 0.00000, RE = 0.00% ---------------
----------------------------------- Summary: 2 Succeed, 0 Failed -----------------------------------
```

### 3.3 Nsight Compute 和 ncu 的使用方法

我们可以用 NVIDIA 提供的 ncu 命令和 Nsight Compute 工具，对我们写的算子进行更深入的分析。

运行下面的命令后会生成 ncu_prof_1.ncu-rep 文件，该文件可以在 Nsight Compute 打开。

```shell
ncu -o ncu_prof_1 --import-source 1 --set full --kernel-name "minimal_gemm" -f python minimal_gemm.py
```

打开软件后，我们可以看到所有算子 profile 后的总览界面。重要的观测指标有运行时间（Duration）、计算和访存利用率（Compute/Memory Throughput），使用的寄存器数量（#Registers），以及 Grid/Block size。

![图8：Nsight Compute 概览界面](https://files.mdnice.com/user/59/a0b8b717-6a23-44f2-8374-260470d8bdfc.png)

关于运行时间，可以发现 ncu profile 的时间（~3us）要小于通过 CUDA Event 计算的时间（~8us）。通过 nsys profile 进一步确认，发现 ncu 计算的 kernel 运行时间更准确。个人认为原因可能是：CUDA Event 记录的是把当前 Event 插入到 stream 队列后，Event 开始执行的时间。这意味着，当 kernel launch 到 Event launch 之间的 CPU 时间大于 kernel 实际执行时间时，Event 的计时是不准的。

![图9-1：Event 的计时原理](https://files.mdnice.com/user/59/c3774d2f-161b-4004-91d6-42b560bb0f73.png)

![图9-2：nsys 中的 CUDA Event](https://files.mdnice.com/user/59/4c5862db-f06d-41f2-a997-c43c3dda10ba.png)

详细的利用率指标可以在 Details 选项卡的第一个列表下找到，它可以从宏观层面帮助我们观测和比较 kernel 性能。如果计算利用率高而访存利用率低，通常说明算子的计算是瓶颈（Compute bound），反之则说明算子的访存为瓶颈，需要进一步分析访存受限的原因。

![图10：计算与访存的利用率](https://files.mdnice.com/user/59/394b7fe1-893c-4542-b47a-4044c0c97416.png)

在实际应用中，许多算子都是访存瓶颈的。如果希望优化访存效率，在 Nsight Compute 中，我们可以从 Memory Chart 中直观分析哪里是访存链路的瓶颈，哪些地方的访存效率不及预期等等。

![图11：Kernel Memory Chart](https://files.mdnice.com/user/59/15a6dd9f-8993-473b-8196-0ac48935a25f.png)

我们也可以从表格形式观察单个 kernel 使用了多少访存指令，传输的数据量有多少，以及硬件执行了多少 memory transaction。读者可以尝试分析这里的数据是如何计算出来的。

![图12：Kernel Memory Table](https://files.mdnice.com/user/59/0f82fbf8-ac71-4913-85fa-09848c77074a.png)

Nsight Compute 还有许多常用的组件，我们会在后续的笔记中进一步介绍，并结合 profile 的数据进行算子分析。

### 3.4 PTX / SASS 分析

在编译算子时开启 --generate-line-info 选项后，我们就可以在 Nsight Compute 中看到 kernel 的 **PTX code** 和 **SASS code**。

SASS code 是交给 GPU 硬件实际执行的机器码，不同的 SM 架构下的 SASS code 可能有较大幅度的变化；而 PTX 是一种虚拟指令集，它在不同的 SM 架构下保持了前向兼容性，因此在老架构下编译出的 PTX 也可以在新架构下运行。

目前 triton 等许多编译器的最终产物其实就是 PTX code。当我们有了 PTX code 后，可以通过 nvcc/ptxas 将它离线编译成 SASS binary 拿来使用，也可以直接在 binary 中放入 PTX code，并让 NVRTC 在线将其编译成 SASS。有关 PTX 和 SASS 的更多信息，请参考官方文档。

对于 Minimal GEMM kernel，我们使用了 CuTe API 编写了数据拷贝和 MMA 的计算逻辑，那么在更加底层的指令层面是如何执行的呢？此处我们以 C = A * B 的算子 kernel 为例，看一下它的 PTX/SASS code。

在 Source 选项卡中，可以选择 View PTX and SASS，就可以看到左侧的 PTX code 和右侧的 SASS code。

![图13：Nsight Compute 的 Source 界面](https://files.mdnice.com/user/59/279ea09a-1ae5-4c2a-8ff5-b8f31aa3978f.png)

回忆下，一个 Minimal GEMM kernel 主要完成了 4 项工作：

- 对全局矩阵进行 tiling，获取需要计算的数据分块的地址；
- 从 GMEM 加载数据至 Register；
- 执行 MMA 指令；
- 将 Register 数据保存至 GMEM。

PTX 指令的前半部分均在完成地址的计算，核心的 load-mma-save 步骤对应到如下 6 条 PTX 指令：

```c++
ld.global.u32 	%r5, [%rd9];
ld.global.u32 	%r6, [%rd11];
ld.global.u32 	%r7, [%rd15];

mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%r1, %r2},{%r3, %r4},{%r5},{%r6, %r7};

st.global.u32 	[%rd11], %r1;
st.global.u32 	[%rd15], %r2;
```

Minimal GEMM kernel 的 SASS code 则更为简洁，除末尾的 BRA 指令外，整个程序仅有 34 条指令：

![图14：Minimal GEMM 的 SASS 代码（SM90 架构）](https://files.mdnice.com/user/59/04abc2a4-4245-473b-9630-bcc228b71044.png)

除去读取传参、读取常量和计算地址外，核心的指令也只有 6 条：

- 22、25、26 行的 **LDG.E** ，从 GMEM 读取 32 bits 的数据，存放在 1 个寄存器中；
- 31 行的 **HMMA.1688.F16**，对应到 PTX 的 mma 指令；
- 32、33 行的 **STG.E**，将寄存器的数据存放至 GMEM。

这些指令和 PTX 可以形成对应关系（实际 PTX 和 SASS 指令无法一一对应，表格仅供展示）。

![](https://files.mdnice.com/user/59/f9aa158f-47cd-4e65-9327-57bc80c3d831.png)

SASS 指令相关的资料较少，但 PTX 指令的行为在 NV 的文档中有详细记录，因此我们在后续的笔记中更关注 PTX 层面的编程，况且 CUTLASS 中也大量使用了 PTX 内联。有兴趣的读者可以从 Minimal GEMM kernel 的 CuTe API 入手，去找到最内层的访存和计算的 PTX 命令，看看是否和我们展示的结果一致。

此外，Nsight Compute 还可以展示相关指令的地址操作和寄存器数据的生命周期：

![图15：SASS 指令访存信息与寄存器生命周期](https://files.mdnice.com/user/59/132b9720-833d-45f6-a264-707f677f37d8.png)

## 4. 总结

这篇笔记介绍了 CuTe 的基础 API，并从 0 到 1 使用 CuTe 完成了 Minimal GEMM 算子的开发、精度验证和性能测试，并简要介绍了 Nsight Compute 的使用方法。

在完成最基础的 kernel 开发流程后，下一步我们将在更复杂的数值精度下完成 GEMM 的计算，并通过 PTX/SASS code 分析硬件是如何在 kernel 内部进行数值精度的转换的。

# CUTLASS 笔记 (2)：混合精度 GEMM kernel

本篇将介绍如何实现一个支持不同输入精度、输出精度和累加精度的 GEMM 算子，并解析在算子内部实现数值精度转换的一些技术细节。对于 CUTLASS 尚未集成但 PTX 指令支持的 MMA op，本篇还介绍了如何根据 PTX 文档实现一个自定义的 FP8 精度的 GEMM kernel。

## 1. MMA 场景下的算子精度

在上一篇笔记中，我们从 0 到 1 实现了一个单指令的 16x8x8 的 MMA 算子，然而这个算子的输入、输出、累加的数值精度均为 FP16，使用场景是受限的。因此接下来我们在 Minimal GEMM kernel 的基础上，支持多样化的数值精度，甚至可以混合两种 FP8 format 做 GEMM 运算！

在真实场景下，一个 MMA 算子涉及到的精度包括：**1）输入数据的精度；2）累加器精度；3）输出数据的精度。** 如果 GEMM 算子包括 **epilogue（后处理）** 部分，则还有与 epilogue 相关的计算和累加精度。

考虑 MMA 场景下的 `D = A * B + C` 的计算，其中输入数据精度包括 A、B、C 三种，我们分别记为 **ComputeTypeA**、**ComputeTypeB**、**ComputeTypeC**，累加器精度即为 A*B 的结果与 C 进行加法后产出数据的精度（注意它并不是 Tensor Core 实际的累加精度），我们记为 **AccType**，这也是 A*B+C 计算结果。如果我们需要的 D 的精度就是 AccType，则算子可以直接返回 A*B+C 的结果，反之则需要额外做一步精度转换操作，将 AccType 转换成我们需要的输出精度，这个输出精度记为 **OutType**。

![图1: MMA 场景下的数值精度](https://files.mdnice.com/user/59/e448feca-0fac-4a2f-a575-e6370e5de8bb.png)

PTX 的 MMA 指令一般会标明该指令所对应的数值精度，例如 

```c++
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
```

这个指令的 AccType 为 FP32，ComputeTypeA、ComputeTypeB 均为 BF16，ComputeTypeC 为 FP32。由于 PTX 和硬件所限，并不是任意的精度组合都有对应的 MMA 指令，且在不同的 MMA shape、sparsity 以及 PTX 版本下，支持的数值精度是有所不同的，具体可见 PTX 文档提供的表格: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-shape

我们绝大多数使用的 MMA 指令的 ComputeTypeC 与 AccType 都相同，因此本篇中所有代码示例均假定 AccType = ComputeTypeC。

## 2. 为什么我们需要一个混合精度算子？

**混合数值精度计算 GEMM 的需求非常广泛，但常常被人们忽视**，有时候我们甚至不会意识到一个简单的 GEMM 会牵扯到如此多的精度。以下列举两个例子：

**1）当前控制算子精度的手段有限。在数值敏感的训练场景，GEMM 的累加精度很可能会影响最终的训练效果。**

例如，使用 `torch.matmul` 计算两个 BF16 的矩阵乘法，它的累加精度是 FP32，但计算两个 FP16 的矩阵乘法，累加精度却是 FP16。并且 Pytorch 并没有提供其他 API 可以让我们修改累加精度。

又例如 `torch.addmm` 完成了一个 MMA 计算，但它要求输入的 A、B、C 三个 Tensor 的数值精度必须是相同的，因此我们无法直接用它完成 BF16 * BF16 + FP32 的计算。所以控制计算精度并不是直接调用 Pytorch API 就能轻松实现的。

**2）低精度下的 GEMM 计算必然会使用到混合数值精度。**

我们拿 DeepSeek V3 论文中展示的 FP8 Linear 计算为例：

![图2: DeepSeek V3 中的 FP8 精度计算](https://files.mdnice.com/user/59/4867f2b0-2af3-4612-8e3b-5e36cd49f60a.png)


以前向 Linear 计算为例，模型权重的精度是 FP8（带量化参数），输入数据的精度是 BF16。在执行算子之前，首先需要在算子外，将 BF16 的输入量化为 FP8，随后将 FP8 的输入和 FP8 的模型权重交给算子计算 GEMM，累加精度为 FP32，最后在算子内将计算结果转换为 BF16 并输出，因此这里需要完成一个 BF16 = FP8 * FP8 + FP32 的 GEMM。显然，我们仅靠 Pytorch 没有简便的方法完成这样的计算。

混合数值精度计算如此重要，但想用起来并没有想象中地那么容易，因此我们有必要探讨如何写一个正确的，且满足精度需求的 GEMM 算子。

## 3. 实现混合精度 GEMM 算子

首先，我们列出本篇要实现的 2 个算子详情：

![](https://files.mdnice.com/user/59/d483ae55-5126-4cd9-92c5-a0c2dfaa4bdb.png)

Kernel 1 要求输入的 A、B 矩阵精度为 BF16，C 矩阵精度、累加精度和 D 矩阵精度为 FP32，而 Kernel 2 要求 D 矩阵的精度为 BF16。那么我们应该如何实现这两种精度的计算呢？

一般而言，在算子内转换精度有两种方式：1）**使用特定 MMA 指令**；2）**在寄存器中转换精度**。接下来我们逐一介绍这两种方式。

### 3.1 使用特定 MMA 指令

对于 FP32 = BF16 * BF16 + FP32 这组精度，现有的 PTX MMA 指令可以直接计算，且 CUTLASS 也提供了封装该指令的 MMA op，因此我们只需要替换 MMA op 即可：

```c++
using MMA_op = SM80_16x8x8_F32BF16BF16F32_TN;
```

### 3.2 在寄存器中转换精度

对于 BF16 = BF16 * BF16 + FP32 这组精度，现有的 PTX MMA 并不支持直接计算，因此我们可以在计算出 FP32 的结果后，再对它进行一次精度转换的操作。在 CuTe 中，我们创建一个与输出结果有相同的 shape，但数值精度为 BF16 的寄存器 Tensor，并将 FP32 Tensor 拷贝到 BF16 Tensor，即可完成精度转换。

```c++
// OutType = float，tCrC 的精度为 FP32
auto tCrO = make_tensor_like<OutType>(tCrC);
copy(tCrC, tCrO);  // Convert precision
// 后续将 tCrO 拷贝至 GMEM
```

这里的 copy 操作其实等价于循环赋值：

```c++
for (int i = 0; i < size(tCrC); ++i) {
  tCrO(i) = tCrC(i);
}
```

### 3.3 Kernel 侧代码改动

在上篇 Minimal GEMM Kernel 中，我们简化了计算场景为 C = A * B + C，此处我们需要修改为常规的 MMA 场景 D = A*B+C，所以除 A、B、C 矩阵外，还需要创建 output 矩阵的 Tensor 表示。output 矩阵仅有指针类型和 C 矩阵不同。

```c++
Tensor mA = make_tensor(make_gmem_ptr((ComputeTypeA *)Aptr),
                        make_shape(m, k),
                        make_stride(k, Int<1>{}));  // (M, K)
Tensor mB = make_tensor(make_gmem_ptr((ComputeTypeB *)Bptr),
                        make_shape(n, k),
                        make_stride(k, Int<1>{}));  // (N, K)
Tensor mC = make_tensor(make_gmem_ptr((ComputeTypeC *)Cptr),
                        make_shape(m, n),
                        make_stride(n, Int<1>{}));  // (M, N)
Tensor mO = make_tensor(make_gmem_ptr((OutType *)Outptr),
                        make_shape(m, n),
                        make_stride(n, Int<1>{}));  // (M, N)
```

在计算完 GEMM 之后，根据是否需要做精度转换，我们选择不同的执行路径。若无需精度转换，我们直接把 FP32 的数据拷贝至 GMEM，否则需要转换精度后再拷贝。

```c++
if constexpr (!cvt_out_precision) {
  copy(copy_atom, tCrC, tCgC);
} else {
  auto tCrO = make_tensor_like<OutType>(tCrC);
  copy(tCrC, tCrO);  // Convert precision

  Tensor tCgO = thr_mma.partition_C(gO);  // (MMA, MMA_M, MMA_N)
  copy(copy_atom, tCrO, tCgO);
}
```

### 3.4 PTX / SASS 分析

通过 Nsight Compute ，我们可以方便找到精度转换对应的 PTX 和 SASS 指令。

首先，在替换 MMA op 后，PTX MMA 指令变为：

```c++
mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%f1,  %f2,  %f3,  %f4},{%r1,  %r2},{%r3},{%f8,  %f8,  %f8,  %f8};
```

而 SASS 指令变为：

```c++
HMMA.1688.F32.BF16 R4, R4, R2, RZ
```

相比于全 FP16 精度，此处只改变了指令对精度的描述，读者可以清晰地看到 PTX/SASS 是如何描述一个 MMA 指令的精度的。

当我们做了额外的精度转换操作后，PTX 多了 4 条 `cvt` 指令，对应到每个线程在寄存器存放的 4 个 D 矩阵的数据：

```shell
cvt.rn.bf16.f32 %rs2, %f2;
cvt.rn.bf16.f32 %rs1, %f1;
cvt.rn.bf16.f32 %rs4, %f4;
cvt.rn.bf16.f32 %rs3, %f3;
```

其中 `.rn` 表示 rounds to nearest even，如果希望使用其他舍入方式，我们可以换用不同的指令，具体可参考 PTX 文档：

![图3: PTX 浮点数 rounding 方式](https://files.mdnice.com/user/59/ba7defdf-5a3e-4905-8382-00336c706cdc.png)

不同于 PTX，SASS 侧仅多出 2 条指令，含义是将 4 个寄存器存放的 4 个 FP32 数据转换精度，打包后存放在 2 个寄存器中，每个寄存器存 2 个 BF16 数据。

```sass
F2FP.BF16.F32.PACK_AB R5, R5, R4   // (R4, R5) -> (R5)
F2FP.BF16.F32.PACK_AB R7, R7, R6   // (R6, R7) -> (R7)
```

随后拷贝 R5、R7 这两个寄存器的数据到 GMEM。

```sass
STG.E desc[UR4][R12.64], R5
STG.E desc[UR4][R14.64], R7
```

## 4. 自定义 FP8 GEMM 算子

上面我们复用了 CUTLASS 提供的 MMA op，但在某些场景下，PTX 指令支持一种精度组合，但 CUTLASS 没有提供对应的封装，应该怎么办呢？此时我们就可以扩展 CUTLASS，来写一个自定义的 MMA op。

自 Ada 架构起，NV 提供了 FP8 精度的 MMA 指令，我们可以来尝试写一个最小的 FP8 GEMM kernel，参考 PTX 文档，其最小的 shape 为 (16, 8, 32)。这里我们假定一个 fancy 的场景，即混合 FP8 的两种 format 精度（E4M3、E5M2）做 GEMM。实际上 PTX 指令是支持这种精度的，但 CUTLASS 没有对应的 MMA op，所以我们就来自己写吧！

![](https://files.mdnice.com/user/59/410f173e-bc5f-4f9a-a124-529d3cb9cfcc.png)

### 4.1 揭开 MMA Atom 的面纱

在上篇笔记中，我们已经介绍了使用单个 MMA 指令需要注意的事项，总的来说有两大方面：

**1）根据 SM 架构、MMA shape 大小、算子精度等情况，选取正确的 PTX MMA 指令。**

**在 CUTLASS 中，MMA op 对象用于描述特定的 PTX MMA 指令**。以 Minimal GEMM kernel 为例，我们使用的 MMA op 为 SM80_16x8x8_F16F16F16F16_TN，在 CUTLASS 中体现为对应 PTX MMA 指令的封装。


```c++
// SM80_16x8x8_F16F16F16F16_TN：Ampere 架构上的 mma.sync 指令封装
// 命名含义：SM80=Ampere, 16x8x8=M×N×K tile尺寸, F16F16F16F16=D/A/B/C均为fp16, TN=A行主序(T)/B列主序(N)
struct SM80_16x8x8_F16F16F16F16_TN
{
  // 每个线程持有的寄存器数量（每个 uint32_t 可存放 2 个 fp16 元素）
  // D(输出): 2个寄存器 → 4个fp16，对应 16x8=128 个元素 ÷ 32线程 = 每线程4个
  using DRegisters = uint32_t[2];
  // A: 2个寄存器 → 4个fp16，对应 16x8=128 个元素 ÷ 32线程 = 每线程4个
  using ARegisters = uint32_t[2];
  // B: 1个寄存器 → 2个fp16，对应 8x8=64 个元素 ÷ 32线程 = 每线程2个
  using BRegisters = uint32_t[1];
  // C(输入累加器): 2个寄存器 → 4个fp16，与 D 布局相同
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,  // 输出 D 的2个寄存器
      uint32_t const& a0, uint32_t const& a1,  // 输入 A 的2个寄存器
      uint32_t const& b0,                       // 输入 B 的1个寄存器
      uint32_t const& c0, uint32_t const& c1)  // 输入累加器 C 的2个寄存器
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      // PTX 指令：warp 内32线程同步执行 16×8×8 矩阵乘累加，D = A*B + C
      // row.col 表示 A 行主序、B 列主序
      "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
      "{%0, %1},"   // D: 输出寄存器 d0, d1
      "{%2, %3},"   // A: 输入寄存器 a0, a1
      "{%4},"       // B: 输入寄存器 b0
      "{%5, %6};\n" // C: 累加器输入 c0, c1
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

**2）按照 MMA 指令内生的矩阵元素和每个线程中寄存器的映射关系，让每个线程拿到正确的矩阵元素，并将计算结果存放于正确的线程中。**

在 CUTLASS 中，MMA Traits 对象用于描述特定 MMA 指令的这种内生的映射关系。还是以上面的 MMA op 为例，它对应的 MMA Traits 如下：

```c++
template <>
struct MMA_Traits<SM80_16x8x8_F16F16F16F16_TN>
{
  // 各矩阵的元素类型均为 fp16
  using ValTypeD = half_t;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  // 该 MMA 指令覆盖的 tile 尺寸：M=16, N=8, K=8
  using Shape_MNK = Shape<_16,_8,_8>;

  // 参与本次 MMA 的线程数：一个完整的 warp（32线程）
  using ThrID   = Layout<_32>;

  // ALayout 描述：A 矩阵 (M=16, K=8) 的元素在 32 个线程寄存器中的分布
  // Shape  : ((4,8), (2,2))  → 第一模式(4,8)枚举32个线程ID，第二模式(2,2)枚举每线程持有的4个元素
  // Stride : ((32,1),(16,8)) → 线程ID贡献 32*i+j，元素内偏移贡献 16*p+8*q
  // 线性索引 = 32*i + j + 16*p + 8*q，对应 A[M][K] 中的行列位置
  using ALayout = Layout<Shape <Shape < _4,_8>,Shape < _2,_2>>,
                         Stride<Stride<_32,_1>,Stride<_16,_8>>>;

  // BLayout 描述：B 矩阵 (N=8, K=8) 的元素在 32 个线程寄存器中的分布
  // Shape  : ((4,8), 2)   → (4,8)枚举32个线程ID，2枚举每线程持有的2个元素
  // Stride : ((16,1), 8)  → 线程ID贡献 16*i+j，元素内偏移贡献 8*k
  // 线性索引 = 16*i + j + 8*k，对应 B[N][K] 中的行列位置
  using BLayout = Layout<Shape <Shape < _4,_8>,_2>,
                         Stride<Stride<_16,_1>,_8>>;

  // CLayout 描述：C/D 矩阵 (M=16, N=8) 的元素在 32 个线程寄存器中的分布
  // 布局结构与 ALayout 完全相同，每线程持有4个元素
  // 线性索引 = 32*i + j + 16*p + 8*q，对应 C[M][N] 中的行列位置
  using CLayout = Layout<Shape <Shape < _4,_8>,Shape < _2,_2>>,
                         Stride<Stride<_32,_1>,Stride<_16,_8>>>;
};
```

### 4.2 TV Layout 与 MN Layout

此处需要解答的一个核心问题是，A/B/C Layout 是如何表达一个 MMA 指令中，矩阵元素和每个线程中寄存器的映射关系的呢？换句话说，每个线程是怎么知道应该拿到哪些矩阵元素，并将哪部分计算结果存放到自己的寄存器中的呢？

在上篇笔记中，我们强调 “CuTe 中的 Layout 是一种**映射关系**，在不同的语境下有不同的含义”。此处我们遇到了 CuTe 的第 2 种 Layout：**TV Layout**，它表达了（线程ID，矩阵元素index）这个二元组与矩阵坐标（M，N）的映射关系，可以表示为（T，V）->（M，N）。上面的 A/B/C Layout 都是一种 TV Layout。

一般而言，TV Layout 是一个双射，即（T，V）和（M，N）具有一一对应关系，因此它也有对应的逆映射，这也是我们接触的第 3 种 Layout：**MN Layout**，它表示了（M，N）->（T，V）的映射关系。我们在上篇笔记中展示的 MMA 映射关系（见下图），其实就是这个 MMA 指令对应的 MN Layout。

![图4: MMA 指令的 MN Layout 示意图](https://files.mdnice.com/user/59/b55cd492-6a34-4d50-95a1-288209675e4b.png)

我们以 ALayout = ((4, 8), (2, 2)) : ((32, 1), (16, 8)) 为例，说明线程是如何找到需要的矩阵元素的。

**TV Layout 的两个 mode 分别为线程 idx 和矩阵元素 idx**。参考 MN Layout 示意图，对于 A 矩阵而言，总共有 16 x 8 = 128 个元素，MMA 共有 32 个线程参与，因此每个线程需要拿到 4 个 A 矩阵的元素。故线程 idx 的取值范围是 0-31，矩阵元素 idx 的取值范围是 0-3，而 A 矩阵对应的 TV Layout 的 shape 就是 (32, 4)，与 ALayout 的 shape 是一样的。

![图5: TV Layout 映射关系的含义](https://files.mdnice.com/user/59/e90146f0-ce0a-4c32-8ea3-328a17d66b3c.png)

假设我们已经知道了 ALayout = ((4, 8), (2, 2)) : ((32, 1), (16, 8))，那么线程 11 的第 2 个元素应该从矩阵的哪个坐标拿数据呢？

- 由于 ALayout 具有嵌套 mode，我们需要将 (T, V) = (11, 2) 变换为嵌套坐标。具体而言，对于每一个 mode，我们将这个 mode 的 idx 经过 idx2crd 这个变换，转换为坐标形式。例如第一个 mode，我们将 11 对应到 shape = (4, 8) 的坐标上，也就是 (11 % 4, 11 / 4) = (3, 2)。因此参考 ALayout 的 shape，可以将 (11, 2) 变换为 ((3, 2), (0, 1))。
- 随后参考 ALayout 的 stride，将坐标经过 ALayout 映射为 (M, N) 空间的 idx = 3x32 + 2x1 + 0x16 + 1x8 = 106。
- 最后在 (M, N) 空间中经过 idx2crd 变换，参考 shape = (16, 8)，可以将 106 转换为坐标形式 (106 % 16, 106 / 16) = (10, 6)。因此线程 11 的第 2 个元素应该从矩阵的 (10, 6) 这个位置拿数据。


所以 **ALayout 将 (T, V) = (11, 2) 映射为 (M, N) = (10, 6)**。我们可以在 MN Layout 的图例中确认，矩阵中 (10, 6) 这个元素的确对应到了 T11 V2，证明我们的计算是正确的。对于 B/C Layout 的解读同理。

> 实际上，图 4 中的 A 矩阵部分应该叫做 MK Layout，B 应该叫做 KN Layout，只有 C 才是正儿八经的 MN Layout。不过为了表述方便，我们都统一把它们称为 MN Layout

于是只要知道了 TV Layout，每个线程就可以计算出自己应该读取和存放矩阵的哪些元素了。

### 4.3 FP8 MMA op 和 MMA Traits

在了解 TV Layout 和 MN Layout 后，我们可以来写对应 FP32 = E4M3 * E5M2 + FP32 的 MMA op 和 MMA Traits 了。

仿照 CUTLASS 的样例，我们命名该 MMA op 为 `SM90_16x8x32_F32E4M3E5M2F32_TN`，选取 PTX 指令为：

```c++
mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32
```

写 MMA op 时基本只需要注意一点，也就是每个线程每个矩阵需要传入多少个寄存器。例如此处每个线程需要 16 个 A 矩阵元素，8 个 B 矩阵元素，4 个 C/D 矩阵元素，并且考虑到 A、B 的数值精度为 FP8，C、D 的数值精度为 FP32，那么我们可以算出 A、B、C、D 分别需要 4、2、4、4 个寄存器。

```c++
struct SM90_16x8x32_F32E4M3E5M2F32_TN
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float    const& c0, float    const& c1, float    const& c2, float    const& c3)
  {
#if defined(CUTE_ARCH_MMA_SM89_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM90_16x8x32_F32E4M3E5M2F32_TN without CUTE_ARCH_MMA_SM89_ENABLED");
#endif
  }
};
```

写 MMA Traits 会麻烦一些。首先需要从 PTX 文档中找到该指令对应的所有矩阵的 MN Layout。这里以 A 矩阵为例，其 MN Layout 如下：


![图6: FP8 A 矩阵 MN Layout 示意图](https://files.mdnice.com/user/59/e483b9f8-c5e4-4138-bf44-df244e6fa90c.png)

然后我们需要从这个 MN Layout 反推出 TV Layout。不难从上图中推导出 A 矩阵的 TV Layout 为 ((4, 8), (4, 2, 2)) : ((64, 1), (16, 8, 256))。

> 什么？你问不难在哪？那么这里有一个小诀窍可以推导出每个 mode 的 Shape 和 Stride。
以 T 这个 mode 为例，我们知道一共有 32 个线程，那么先看从 T0V0 到 T1V0 的步长，从上图看知道 MN 坐标从 (0, 0) 到 (0, 5) ，步长是 64，从 T1V0 到 T2V0 步长也是 64，一直到 T3V0 -> T4V0，发现步长有突变，因此 T 这个 mode 还有一个 sub-mode 维度，步长可以从 T0V0 -> T4V0 算出来是 1，后续 T4V0 -> T8V0，T8V0 -> T12V0，一直到 T24V0 -> T28V0，步长均为 1。
因此我们知道 T 这个 mode 有两个 sub-mode，Shape 为 (4, 8)，Stride 为 (64, 1)，因此 T 这部分的 mode 就推导出来了。V 部分的 mode 同理。

同理可以写出 B、C 的 TV Layout，因此我们就可以写出这个 MMA op 对应的 MMA Traits。

```c++
template <>
struct MMA_Traits<SM90_16x8x32_F32E4M3E5M2F32_TN>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16,_8,_32>;
  using ThrID   = Layout<_32>;
  using ALayout = Layout<Shape <Shape < _4,_8>,Shape < _4,_2,  _2>>,
                         Stride<Stride<_64,_1>,Stride<_16,_8,_256>>>;
  using BLayout = Layout<Shape <Shape < _4,_8>,Shape <_4,  _2>>,
                         Stride<Stride<_32,_1>,Stride<_8,_128>>>;
  using CLayout = Layout<Shape <Shape < _4,_8>,Shape < _2,_2>>,
                         Stride<Stride<_32,_1>,Stride<_16,_8>>>;
};
```

最后，我们将 MMA op 设置为我们自己写的 op 类，就大功告成了！

```c++
using MMA_op = SM90_16x8x32_F32E4M3E5M2F32_TN;
```

### 4.4 精度验证

至此我们写了 4 类算子，每个算子均支持 MM 和 MMA 两种场景，因此总共需要测试 8 个样例，运行测试脚本后结果如下：

```shell
------------------------------------------ M=16, N=8, K=8 ------------------------------------------
Block Size: (32, 1, 1) | Grid Size: (1, 1, 1) | Shared Memory Size: 0 Bytes
Kernel execution time: 2.438 ms
--------------- Result: Success, Max diff = 0.00000, Mean diff = 0.00000, RE = 0.00% ---------------
Block Size: (32, 1, 1) | Grid Size: (1, 1, 1) | Shared Memory Size: 0 Bytes
Kernel execution time: 0.010 ms
--------------- Result: Success, Max diff = 0.00000, Mean diff = 0.00000, RE = 0.00% ---------------
------------------------------------------ M=16, N=8, K=8 ------------------------------------------
Block Size: (32, 1, 1) | Grid Size: (1, 1, 1) | Shared Memory Size: 0 Bytes
Kernel execution time: 0.010 ms
--------------- Result: Success, Max diff = 0.00000, Mean diff = 0.00000, RE = 0.00% ---------------
Block Size: (32, 1, 1) | Grid Size: (1, 1, 1) | Shared Memory Size: 0 Bytes
Kernel execution time: 0.011 ms
--------------- Result: Success, Max diff = 0.00000, Mean diff = 0.00000, RE = 0.00% ---------------
----------------------------------------- M=16, N=8, K=32 ------------------------------------------
Block Size: (32, 1, 1) | Grid Size: (1, 1, 1) | Shared Memory Size: 0 Bytes
Kernel execution time: 0.010 ms
--------------- Result: Success, Max diff = 0.00000, Mean diff = 0.00000, RE = 0.00% ---------------
Block Size: (32, 1, 1) | Grid Size: (1, 1, 1) | Shared Memory Size: 0 Bytes
Kernel execution time: 0.010 ms
--------------- Result: Success, Max diff = 0.00000, Mean diff = 0.00000, RE = 0.00% ---------------
----------------------------------------- M=16, N=8, K=32 ------------------------------------------
Block Size: (32, 1, 1) | Grid Size: (1, 1, 1) | Shared Memory Size: 0 Bytes
Kernel execution time: 0.010 ms
--------------- Result: Success, Max diff = 0.00000, Mean diff = 0.00000, RE = 0.00% ---------------
Block Size: (32, 1, 1) | Grid Size: (1, 1, 1) | Shared Memory Size: 0 Bytes
Kernel execution time: 0.010 ms
--------------- Result: Success, Max diff = 0.00000, Mean diff = 0.00000, RE = 0.00% ---------------
----------------------------------- Summary: 8 Succeed, 0 Failed -----------------------------------
```

### 4.5 PTX / SASS 分析

不出所料，PTX 指令就是我们在 MMA op 中编写的内联汇编：

```c++
mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32 {%f1,  %f2,  %f3,  %f4},{%r1,  %r2,  %r3,  %r4},{%r5,  %r6},{%f8, %f8, %f8, %f8};
```

SASS 则稍显复杂，它主要包括了一连串的 Pack 指令，最终是用 HMMA.16816.F32 这个指令来计算 MMA 的，目前暂不清楚其中的原理。

```c++
F2FP.F16.E4M3.UNPACK_B R4, R9
F2FP.F16.E5M2.UNPACK_B R22, R0
F2FP.F16.E5M2.UNPACK_B R23, R8
F2FP.F16.E4M3.UNPACK_B R6, R10
F2FP.F16.E4M3.UNPACK_B R5, R11
F2FP.F16.E4M3.UNPACK_B R7, R18
HMMA.16816.F32 R4, R4, R22, RZ
F2FP.F16.E5M2.UNPACK_B R21, R8.H1
F2FP.F16.E4M3.UNPACK_B R8, R9.H1
F2FP.F16.E4M3.UNPACK_B R9, R11.H1
F2FP.F16.E5M2.UNPACK_B R20, R0.H1
F2FP.F16.E4M3.UNPACK_B R10, R10.H1
F2FP.F16.E4M3.UNPACK_B R11, R18.H1
HMMA.16816.F32 R4, R8, R20, R4
```

## 5. 总结

这篇笔记主要介绍如何在 Minimal GEMM kernel 的基础上实现多种精度组合的 MMA 运算，并实现了一个自定义的 FP8 精度的 GEMM kernel。本篇中的 TV Layout 和 MN Layout 是 CUTLASS CuTe 的四大重要 Layout 之二，我们还会在之后的笔记中再遇到它们。深刻理解 Layout 本质上是一种映射关系，是精通 CuTe 编程的一个重要环节。

# CUTLASS 笔记 (3)：Tiled MMA

本篇将解析 GEMM 算子的重要概念模型——三级 Tiling，并详细介绍 CUTLASS CuTe 中如何实现 Tiled MMA 层级的运算，特别是 Tiled MMA API 各种参数的用法和含义。在本篇结束后，我们的 GEMM kernel 将从单指令运算扩展至单 Tile 运算。

在前两篇笔记中，我们实现了一个单指令的 16x8x8 的 MMA 算子，并支持了多种混合精度的计算。真实场景下的矩阵运算规模往往更大，因此当我们理解了单条指令的行为后，就需要考虑如何将单指令 MMA 运算扩展至更大的规模。

对于一个规模为 (M, N, K) 的矩阵乘法问题，为了利用 GPU 的并行计算能力，我们可以将它划分为**若干个可并行处理的分块矩阵**，将这些分块矩阵从内存传输到不同的计算单元（一般为 Tensor Core），每个计算单元获得数据后**执行一条或多条矩阵运算指令（如 mma）**，最后再将计算结果传输回内存即可。

因此，只要我们有处理矩阵运算的单条 MMA 指令，就可以在 SM 间并行计算矩阵分块，**在 SM 中循环执行若干条 MMA 指令**，从而实现任意规模的矩阵运算。

![图1: 单指令扩展至任意规模的矩阵运算](https://files.mdnice.com/user/59/9a8618b6-b6fc-4457-ae14-bb74a1aab80a.png)

现实层面上，我们不仅要考虑运算的可行性，还要考虑如何高效地完成运算。写一个正确的 GEMM 算子并不难，难的是如何利用硬件和指令集的特性，写出性能最佳的 GEMM 算子。

为了实现一个高效的 GEMM 算子，我们需要从宏观上讨论算子优化的各种手段，在这些优化方法论的指导下，我们就能知道为什么 GEMM 需要进行多级的分块 Tiling 了。

## 1. 算子优化方法论

宏观上看，一个算子主要有三个优化方向：一是计算，二是通信，三是存储。

**计算层面**，我们关注单位时间内能够进行多少计算，常用指标是每秒浮点数运算次数 FLOPS。在现代以 Tensor Core 为计算核心的硬件架构下，我们通常将算子的 **Tensor Core 利用率**作为是否充分利用了硬件算力的指标。

受到功耗墙的影响，我们无法发挥出 Tensor Core 的理论最大算力，但我们总可以在算子内堆满 mma 指令来达到实际的最大算力。在硬件的算子上限面前，想要优化计算就需要从**算法层面**入手，例如以更少的 FLOPs 实现相同的计算，或是合理安排计算依赖让 Tensor Core 满负荷运行，等等。

值得注意的是，**FLOPS 或者 Tensor Core 利用率低并不意味着计算层面需要优化**。很多时候，数据通信和存储容量的瓶颈会导致 Tensor Core 无法接受更多的计算任务，从而浪费了 Tensor Core 算力，因此具体问题需要具体分析。

**通信层面**，在只考虑单卡的情况下，我们主要关注将数据从一个存储介质搬运到另一个存储介质的耗时，也就是 latency。在现代硬件架构下，Tensor Core 算力非常大，计算数据的时间往往小于数据通信的时间，因此当前大部分常见的算子在 Tensor Core 面前都是 memory bound 的。所以**当 FLOPS 未达预期时，通常要优先分析通信层面是否存在瓶颈** 。由于两个不同存储介质的 latency 由硬件决定，想要优化通信，需要尽可能减少高 latency 的通信量，或者通过流水线的方式将通信时间掩盖在计算时间之中。

**存储层面，GPU 的存储层级结构对数据通信有着重大影响，进而影响计算效率**。当前可编程的 GPU 存储单元有 3 种：

1）片外的 Global Memory，简写为 GMEM，也就是我们常说的显存。GMEM 的数据可被多个算子所使用，它的容量最大，但延迟最高，意味着从 GMEM 读写数据是最慢的；

2）片内的 Shared Memory，简写为 SMEM，也称为共享内存。一个 thread block 中的所有线程共享同一个 SMEM 区域，容量通常为几十到几百 KB，延迟相比 GMEM 更低；

3）片内的 Register File，简写为 RF 或者 RMEM，也就是 GPU 的寄存器。寄存器的数据可被计算单元直接获取，延迟相比 SMEM 更低，但容量更加受限。

Flash Attention 等算法利用了 GMEM 和 SMEM 访存效率的差异，通过 SMEM 这个中间存储介质减少了大量的 GMEM 读写，从而优化了 Attention 算子。很多算法同学在 FA 的影响下也逐渐认识到硬件访存对于算子的重要性。

然而在许多场景下，SMEM 和寄存器的存储容量甚至比访存效率更加关键，我们将这些存储资源的利用率称为 Occupancy。对于 SM90（Hopper）和 SM100（Blackwell）架构，一个 thread block 最多仅可申请 227 KB 的 SMEM，且最多只能使用 64K 个寄存器，同时受编译器限制，每个线程可用的寄存器最多为 255 个。

在单指令 MMA 规模越来越大的今天，SMEM 和 RMEM 面临的存储压力也越来越大，而一旦 SMEM 溢出，算子便无法正常运行，RMEM 溢出则会造成 Local Memory 的读写，在最坏情况下其访存效率约等于 GMEM 的访存效率，考虑到大部分场景下寄存器会被访问上千万次，这会极大程度上影响算子的执行效率。相应的，Occupancy 不足虽然不会有严重后果，但说明算子仍有潜在的访存效率和数据复用率的优化空间。

因此，优化存储需要在尽可能增加 Occupancy 的情况下避免任何副作用，充分利用硬件给我们提供的资源，以存储换效率。

计算、通信、存储虽有不同的优化手段，但它们三位一体，相互影响，在对 GEMM 进行性能优化的过程中，我们会逐渐认识到这三者的重要性。

接下来，我们需要回答一个关键问题：**为什么 GEMM 算子需要做多级的分块 Tiling？**

## 2. GEMM 三级 Tiling

对于一个任意规模的 GEMM 任务 D = AB，我们当然可以只做一级 Tiling，仅在一个 block 中通过循环执行 16x8x8 的 MMA 指令，完成任意规模的 GEMM 运算。

![图2: 单 Block 完成一个 GEMM](https://files.mdnice.com/user/59/b66be4b8-ef80-42a4-acc0-f60e5da08d4e.png)


这种实现的问题显然不少，其中最明显的问题是没有利用 GPU 多 SM 并行的能力。

为了让计算任务可并行，我们可以将 D 矩阵以 16x8 的大小切分成若干个 tile，由于 tile 间可以并行计算，一个 tile 的计算任务就可以交给一个 block 完成。在 block 内部，我们可以沿着 k 维度循环执行，从 GMEM 拷贝 16x8 的 A 矩阵分片和 8x8 的 B 矩阵分片，存至寄存器中，随后执行 MMA 指令，完成 tile 的计算，并将结果写回 GMEM。

![图3: SM 间并行的实现](https://files.mdnice.com/user/59/18e8c736-dce1-4369-a820-e3823df0ef11.png)


### 2.1 扩展 Tile 的规模

SM 并行起来了，但每个 SM 的算力并没有充分利用。上面的一次循环只让一个 warp（32 threads）执行了一次 mma 指令，而每个线程执行一个 FP16 16x8x8 的 mma 指令只需要 5-7 个寄存器，一个 warp 最多仅需 224 个寄存器，远远低于单 block 64K 寄存器的个数。此外，单个 SM 有 4 个 Tensor Core，单 warp 循环执行 mma 指令只能使用一个 Tensor Core，没有发挥出 Tensor Core 的全部算力。

于是我们有两种思路来提升一个 block 的性能：

1. 增加并行度，也就是扩展线程数，增加 warp 数量；
2. 在每个 warp 内，循环执行多个 mma 指令。

增加线程和 mma 指令个数，不仅有助于发挥计算单元的最大算力，还可以一次性从 GMEM 拷贝更多数据，充分利用 GMEM 到 RMEM 的带宽。

于是，我们可以将单个 mma 指令扩展至更大的 tile，例如将线程扩展至 8 倍（M 维度 2 倍，N 维度 4 倍），每个 warp 的 mma 指令向 K 维度扩展至 2 倍，这样一个 tile 的大小变为 32x32x16，共有 256 个线程执行一个 tile 的计算。

![图4: Tile Tiling](https://files.mdnice.com/user/59/29528f44-78ea-46a0-be4b-60cf6f9a4d3f.png)

一个 tile 的大小可以无限扩展吗？理论上可以，但在考虑效率的情况下不能。首先，单个 block 的线程数量不能大于 2048 个；其次，无论是扩展线程数还是 mma 指令都需要更多的寄存器，那么就会遇到 RMEM 的存储限制。我们当然希望能充分利用寄存器空间，但一旦扩展规模过大，就会产生 register spilling，造成性能损失和大量的 GMEM 显存使用（还有非常长的编译时间）。因此合理选择线程数量和 tile 的规模是非常重要的。

### 2.2 扩展 Block 的规模

当我们解决了单个 SM 的算力利用问题后，又面临了新的瓶颈：GMEM 的访存量大，且有许多重复访存的情况，**数据复用** 率低。

在 GEMM 任务下，同一行的 tile 共用同一行 A 矩阵分片，同一列的 tile 共用同一列 B 矩阵分片。如果我们在一个 block 里面循环完成多个 tile 的计算，并将这些 tile 共用的 A、B 矩阵分片从 GMEM 拷贝到 SMEM，那么就可以减少重复拷贝 GMEM 的数据量，提升每个 tile 的访存效率。

因此，我们可以将单个 tile 向 (M, N, K) 三个维度分别扩展 (4, 4, 2) 倍，从而将一个 block 的计算规模扩展至 128x128x32。

![图5: Block Tiling](https://files.mdnice.com/user/59/0fda9597-0f13-4601-bb7f-ef2895f3c2f5.jpg)

一个 block 的大小可以无限扩展吗？理论上可以，但在**考虑效率的情况下不能** 。由于单个 block 的 SMEM 存储大小有限，在大 block 的情况下，我们没法一次性将 A/B 矩阵的分片完全放到 SMEM 中。即使我们能够分批拷贝数据，过大的 block 会减少总 block 数量，从而影响 SM 维度的并行计算效率。因此合理调整 block size 也是算子优化的重要环节。

### 2.3 Global MMA Tiling

从全局的角度考虑，我们可以按照最初的一级 Tiling 模式，将 D 矩阵划分为可并行计算的一个个 block 来完成整个 GEMM 的计算。

考虑到 block 的 SMEM 限制，我们通常要对参与计算的 A/B 矩阵分片做进一步的分块，每一轮计算仅拷贝一对 A/B 分块，计算得到的结果在前一轮的基础上进行累加，从而通过 K 维度的循环来完成一个 block 的完整计算。

![图6: GEMM 三级 Tiling](https://files.mdnice.com/user/59/ec878284-400a-4c1b-8d99-5a6eb66ca51f.png)

至此，GEMM 运算被细分成了 Global 到 Block、Block 到 Tile、Tile 到 MMA Atom 的三级 Tiling。值得注意的是，**每一级 Tiling 都和 GPU 硬件特性紧密相关** ：

- 为了让多个 SM 并行计算，需要将 Global MMA 切分为可并行计算的若干个 Block；
- 为了利用 SMEM 访存低延迟的特点，尽可能减少 GMEM 访存量，需要将 Block 切分成可复用数据的若干个 Tile；
- 为了充分利用多核 Tensor Core 的算力，每一个 Tile 需要包含足够多的 MMA Atom 计算指令。

> 当硬件特性进行更新后，Tiling 的方式很有可能会发生变化，例如 Blackwell 架构下的 2SM MMA 就利用了 Distributed SMEM，增加了在 Cluster 层面的 Tiling。

本 CUTLASS 笔记系列的宗旨是“自底向上”。在本篇中，我们着重解析如何在 CUTLASS 中扩展 MMA Atom，将单指令计算扩展至一个 Tile 的计算。

![图7: 我们先来关注 Tile 级别的计算](https://files.mdnice.com/user/59/36f0e0ed-2f10-45cf-b7b7-e73362656198.png)

## 3. Tiled MMA 实现

首先写出本篇实现的算子详情，此处单指令规模仍为 16x8x8，单个 Tile 的规模扩展至 32x32x16，同时线程数也从 32 扩展至 256。

![](https://files.mdnice.com/user/59/d85126c6-9823-4675-bac0-b99c98a9a012.png)

## 3.1 make_tiled_mma API

在笔记（1）中，我们初次使用了 `make_tiled_mma` 这个 API，当时只传入了一个 mma op 作为参数，即没有做扩展。

```c++
using TiledMMA = decltype(make_tiled_mma(MMA_op{}));
```

上面提到，单指令扩展至 Tile 有两个思路，一是扩展 warp 数量，二是扩展每个 warp 计算的 mma 指令数量。因此，我们需要增加两个新的参数，来表示线程的扩展和 mma 指令的扩展。

![图8: make_tiled_mma API 图解](https://files.mdnice.com/user/59/4c4132e3-fb2c-456d-ac61-643a8e6af77a.jpg)

代码如下所示。可以看到 `make_tiled_mma` 新增的两个参数分别为 `MMAThrLayout` 和 `MMATileLayout`，也就是线程在 (M, N, K) 维度的扩展方式和单 Tile 的总规模。其中 `kMmaThrExpandM/N/K` 控制了线程在 `(M, N, K)` 维度的扩展规模，`kMmaValExpandM/N/K` 控制了 mma 指令在 `(M, N, K)` 维度的扩展规模。

```c++
using MMA_op = SM80_16x8x8_F32BF16BF16F32_TN;
using MMA_traits = MMA_Traits<MMA_op>;
using MMA_shape = MMA_traits::Shape_MNK;

static constexpr int kMmaThrExpandM = 2;
static constexpr int kMmaThrExpandN = 4;
static constexpr int kMmaThrExpandK = 1;

static constexpr int kMmaValExpandM = 1;
static constexpr int kMmaValExpandN = 1;
static constexpr int kMmaValExpandK = 2;

static constexpr int kMmaTileM = kMmaThrExpandM * kMmaValExpandM * get<0>(MMA_shape{});
static constexpr int kMmaTileN = kMmaThrExpandN * kMmaValExpandN * get<1>(MMA_shape{});
static constexpr int kMmaTileK = kMmaThrExpandK * kMmaValExpandK * get<2>(MMA_shape{});

using MMAThrLayout = decltype(make_layout(make_shape(Int<kMmaThrExpandM>{},
                                                     Int<kMmaThrExpandN>{},
                                                     Int<kMmaThrExpandK>{})));
using MMATileLayout = Tile<Int<kMmaTileM>, Int<kMmaTileN>, Int<kMmaTileK>>;
using TiledMMA = decltype(make_tiled_mma(MMA_op{}, MMAThrLayout{}, MMATileLayout{}));
```

此时我们可以打印出经过扩展后的 TiledMMA，可以清晰地看到，相比于单指令 MMA Atom，我们实现的 Tiled MMA 在 (M,N,K) 三个维度分别扩展了 (2,4,2) 倍。其中 **M、N 维度的扩展是线程扩展**，因此线程编号从 T0-T31 扩展至 T0-T255（图中部分矩阵元素只展示了一个 T，但实际上可能会被多个线程读取），K 维度是 mma 指令扩展，对于线程来说其实就是寄存器扩展，因此 K 维度的 T 不变，而 V 的范围扩大了 1 倍。

![图9: Tiled MMA 图示](https://files.mdnice.com/user/59/37e55435-0a90-496a-85bc-f388fc24c9c6.jpg)

`make_tiled_mma` 新增的两个参数分别为 `MMAThrLayout`和`MMATileLayout`，接下来我们详细解析它们的一些细节。


`MMAThrLayout`对应的是一个 **Layout** ，表示 mma 在 (M, N, K) 三个维度扩展线程 / warp 的方式，我们知道 Layout 本质是一个映射，那么这里的 `MMAThrLayout` 表示的是 (M, N, K) 三维坐标到 warp_idx 的映射，(M, N, K) 坐标对应了上图中的一个 MMA Atom，而 warp_idx 表示这个 MMA Atom 交给对应 index 的 warp 计算。在我们的示例中，MMAThrLayout = (2,4,1):(1,2,8)，那么当 (M, N, K) = (1, 2, 0) 时，我们找到上图中编号为 (M, K) = (1, 0) 的蓝色块，编号为 (K, N) = (0, 2) 的红色块和编号为 (M, N) = (1, 2) 的绿色块，这个就是坐标对应的 MMA Atom，并且通过 Layout 映射，我们知道这个 Atom 在 warp = 5，也就是 T160-T191 来计算，符合上图的表示。

> warp_idx = m×1 + n×2 + k×8

> 这里有一个小问题请读者思考：为什么我们一般让 MMAThrLayout 的 K 维度为 1，即不在 K 维度扩展线程？

![](https://files.mdnice.com/user/59/52802c28-3386-4e0d-9b93-7ab127781167.png)

`MMATileLayout` 是一个长度为 3 的 tuple，分别代表 M、N、K 三个维度的排列，其中每个维度排列都用一个 Layout 表示。调整某个维度的 Layout，就可以改变各个 MMA Atom 在这个维度的排列次序（Permutation）。

改变 MMA Atom 排列的 Layout，实际上是 CUTLASS 四大重要 Layout 的最后一种Layout：**Permutation Layout** 。它表示了从旧位置坐标（old_index）到新位置坐标（new_index）的映射。根据 Permutation Layout，我们就可以知道在新的排列下，原先的 MMA Atom 应该在哪个位置了。

这里举一个官方的例子，当 Permutation Layout = (4,4,2):(1,8,4) 时，旧排列和新排列可展示为：

```shell
old m-coord:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
new m-coord:  0  1  2  3  8  9 10 11 16 17 18 19 24 25 26 27  4  5  6  7 12 13 14 15 20 21 22 23 28 29 30 31
```

读者可以自由修改三个 Layout 并比较修改前后的 TiledMMA 图，来理解 Permutation Layout 的作用。

在修改 TiledMMA 后，原先使用的 copy 和 gemm API 无需变化，因为这两个 API 会根据 TiledMMA 的扩展情况，自动帮我们处理 MMA Atom 的循环计算。

## 3.2 Tensor Metadata 详解

在笔记（1）我们提到 partition 后的 Tensor 还有一些维度 MMA_M/N/K，这些维度其实就是 mma 指令的扩展维度。

```c++
Tensor tCgA = thr_mma.partition_A(gA);  // (MMA, MMA_M, MMA_K)
Tensor tCgB = thr_mma.partition_B(gB);  // (MMA, MMA_N, MMA_K)
Tensor tCgC = thr_mma.partition_C(gC);  // (MMA, MMA_M, MMA_N)

Tensor tCrA = thr_mma.partition_fragment_A(gA);  // (MMA, MMA_M, MMA_K)
Tensor tCrB = thr_mma.partition_fragment_B(gB);  // (MMA, MMA_N, MMA_K)
Tensor tCrC = thr_mma.partition_fragment_C(gC);  // (MMA, MMA_M, MMA_N)
```

借此机会，我们介绍下如何来看打印出来的 Tensor 信息。

打印出这六个 Tensor 的信息如下：

```c++
gmem_ptr[16b](0x7f61c3e00000) o ((_2,_2),_1,_2):((_1,128),_0,_8)
gmem_ptr[16b](0x7f61c3e00400) o (_2,_1,_2):(_1,_0,_8)
gmem_ptr[32b](0x7f61c3e01800) o ((_2,_2),_1,_1):((_1,256),_0,_0)
ptr[16b](0x7f61d9fffca0) o ((_2,_2),_1,_2):((_1,_2),_0,_4)
ptr[16b](0x7f61d9fffcb0) o (_2,_1,_2):(_1,_0,_2)
ptr[32b](0x7f61d9fffcc0) o ((_2,_2),_1,_1):((_1,_2),_0,_0)
```

- gmem_ptr 代表数据的存储介质，类似还有 smem_ptr，rmem_ptr，tmem_ptr 等，没有前缀的 ptr 则为普通指针，不包含存储介质信息；
- 16b、32b 为 Tensor 单个数据长度；
- 0x7f61c3e00000 为 Tensor 数据的基地址；
- 分隔符 o 代表它是一个 Tensor，分隔符前后分别表示 Tensor data 和 Tensor Layout；
- 后面的 Layout 就是这个 Tensor 的 Tensor Layout。

可以看出，此处的 MMA_M/N/K 分别为 (1, 1, 2)，也就是我们填入的 kMmaValExpandM/N/K的值，印证了我们上面的说法。

---

> 以下补充说明由 Claude 4.6 生成，用于辅助理解。

## 补充：MMAThrLayout stride (1,2,8) 的来源

`MMAThrLayout = (2,4,1):(1,2,8)` 中的步长 `(1,2,8)` **不需要手动指定**，是 `make_layout(make_shape(2,4,1))` 按**列优先（column-major）紧凑步长**自动推导的结果：

| 维度 | Shape | Stride 推导 | Stride 值 |
|------|-------|------------|-----------|
| M    | 2     | 最内层，固定为 1 | **1** |
| N    | 4     | shape[M] = 2 | **2** |
| K    | 1     | shape[M] × shape[N] = 2×4 | **8** |

因此 warp\_idx 的计算公式为：

```
warp_idx = m×1 + n×2 + k×8
```

以 `(M, N, K) = (1, 2, 0)` 为例：`warp_idx = 1×1 + 2×2 + 0×8 = 5`，对应 T160\~T191，与图示吻合。

K 维度 shape=1、stride=8 只是紧凑布局的副产品——由于 K 方向只有一个 Atom 槽，k 项永远为 0，stride 取什么值对结果没有影响。

---

## 补充：为什么 MMAThrLayout 的 K 维度一般为 1

**K 是规约维度，沿 K 扩展线程会引入额外的跨线程归约开销，而沿 M/N 扩展线程则完全独立、无需通信。**

- **M/N 维度**：不同输出元素彼此独立，不同 warp 各自持有独立的 accumulator 寄存器，天然并行，无需同步。
- **K 维度**：同一输出元素 `C[m,n]` 需要对所有 k 求和。若把 K 拆给多个 warp，每个 warp 只算出部分和，最终还需要跨 warp reduce（shared memory 或 warp shuffle），增加延迟和代码复杂度。

K 方向的扩展由 `kMmaValExpandK` 承担：同一线程**串行多发几条 MMA 指令**，把更多 K 的贡献累积到同一组 accumulator 寄存器，无需任何线程间通信：

```cpp
// kMmaValExpandK = 2：每个线程对 K 方向连续发 2 条 MMA 指令
// 两条指令都写到同一块 accumulator 寄存器，自然完成 K 方向的累积
for (int k = 0; k < kMmaValExpandK; k++) {
    gemm(tiled_mma, accum, tA(_, _, k), tB(_, _, k), accum);
}
```

| 扩展方式 | 执行者 | 输出寄存器 | 需要归约？ |
|----------|--------|------------|-----------|
| M/N 线程扩展（`MMAThrLayout`） | 不同 warp 各算各的输出元素 | 独立，互不重叠 | 不需要 |
| K 线程扩展（假设） | 不同 warp 算同一输出元素的部分和 | 需要合并 | **需要跨 warp 归约** |
| K 指令扩展（`MMATileLayout`） | 同一线程串行多发 MMA | 同一组 accumulator 原地累积 | 不需要 |

---

## 补充：Tensor 输出格式与 MMA_M/N/K=(1,1,2) 的对应关系

### 输出格式解读

每条输出的格式统一为：

```
<存储介质>[<元素位宽>](<基地址>) o <Shape>:<Stride>
```

| 字段 | 含义 |
|------|------|
| `gmem_ptr` / `ptr` | 存储介质；`gmem_ptr` 是全局内存，无前缀的 `ptr` 为普通指针（此处为寄存器） |
| `16b` / `32b` | 单个元素位宽；BF16=16b，F32=32b |
| 地址 | Tensor 数据起始地址 |
| `o` | 分隔符，左边是数据，右边是 Layout |
| `Shape:Stride` | CuTe Layout，括号嵌套表示层级 |

### 逐条对应

配置参数回顾：`MMA_op = SM80_16x8x8`（单 atom: M=16, N=8, K=8），扩展系数：

```
kMmaThrExpandM/N/K = (2, 4, 1)   ← 线程扩展（warp 数量）
kMmaValExpandM/N/K = (1, 1, 2)   ← 指令扩展（每线程发几条 MMA）
```

partition 后的 shape 语义为 `(MMA, MMA_M, MMA_N/K)`，其中：
- **MMA**：单条 MMA 指令中该线程持有的元素数
- **MMA_M/N/K**：该线程需要在各方向执行多少条 MMA 指令，等于 `kMmaValExpand*`

| 变量 | 原始输出 | Shape 含义 | Stride 含义 |
|------|----------|-----------|------------|
| `tCgA` | `((_2,_2),_1,_2):((_1,128),_0,_8)` | MMA=(2,2)=4个A元素；MMA_M=**1**；MMA_K=**2** | MMA 内步长 (1,128)：前2个连续，后2个隔 128 行（全局内存行跨度）；MMA_K 步长 8：K 方向每跨一个 atom 跳 8 个元素 |
| `tCgB` | `(_2,_1,_2):(_1,_0,_8)` | MMA=2个B元素；MMA_N=**1**；MMA_K=**2** | B元素连续；MMA_K 步长 8 同上 |
| `tCgC` | `((_2,_2),_1,_1):((_1,256),_0,_0)` | MMA=(2,2)=4个C元素；MMA_M=**1**；MMA_N=**1** | MMA 内步长 (1,256)；MMA_M/N 均为 1 故步长为 0 |
| `tCrA` | `((_2,_2),_1,_2):((_1,_2),_0,_4)` | 与 tCgA shape 相同 | 寄存器紧凑布局：全局内存步长 128 缩为 2；MMA_K 步长从 8 缩为 4 |
| `tCrB` | `(_2,_1,_2):(_1,_0,_2)` | 与 tCgB shape 相同 | 寄存器紧凑，MMA_K 步长 2 |
| `tCrC` | `((_2,_2),_1,_1):((_1,_2),_0,_0)` | 与 tCgC shape 相同 | 寄存器紧凑，步长 2 |

### MMA_M/N/K = (1, 1, 2) 的来源

**MMA_M/N/K 描述的是"每个线程要循环执行多少条 MMA 指令"，它正好等于 `kMmaValExpand*`**，而不是 `kMmaThrExpand*`。

`kMmaThrExpand*` 靠增加 warp 数量覆盖更多输出元素，不同 warp 各算各的，单个 warp 执行的指令条数没有增加。`kMmaValExpand*` 则是让同一个线程多执行几条 MMA 指令来覆盖更多数据，体现在分区后 Tensor 的 MMA_* 维度上：

```
kMmaValExpandM = 1  →  tCgA/tCgC 中 MMA_M = _1
kMmaValExpandN = 1  →  tCgB/tCgC 中 MMA_N = _1
kMmaValExpandK = 2  →  tCgA/tCgB 中 MMA_K = _2   ← K 方向每个线程要执行 2 条 MMA 指令
```

三个 Tensor 交叉印证：`MMA_M=1, MMA_N=1, MMA_K=2`，恰好等于 `kMmaValExpandM/N/K = (1,1,2)`。

C 矩阵（`tCgC`）没有 MMA_K 维度，因为 K 是规约维度——两条 K 方向的 MMA 指令都写到同一组 accumulator 寄存器，不产生额外的输出维度。

全局内存（`gmem_ptr`）和寄存器（`ptr`）版本 shape 相同，但 stride 不同：全局内存 stride 反映矩阵行列跨度（如 128、256），寄存器 stride 是紧凑的小整数（如 2、4），说明寄存器中数据被重新紧凑排列。

### 3.3 SASS 分析

当我们在 K 维度扩展了 mma 指令后，单线程的 mma 指令就变成了 2 条，相应地，LDG 的指令也从 3 条增加到了 6 条。由于我们在 **K 维度扩展 mma，两次 A * B 计算的是同一个 D ，因此第二个 mma 的累加器需要使用第一个 mma 的计算结果** 。从 RMEM 拷贝回 GMEM 则仍然只需要 2 次 STG。

```c++
// ---- 阶段 1：提前发出全部 6 条 LDG，掩盖访存延迟 ----
// SM80_16x8x8 每个线程持有 A 的 2 个寄存器（4 个 BF16）、B 的 1 个寄存器（2 个 BF16）
// kMmaValExpandK=2 → 需要 2 套 A/B 数据，共 6 条 LDG（原来 kMmaValExpandK=1 时只有 3 条）

LDG.E R4,  desc[UR4][R14.64]        // A fragment[K=0]，第 1 个寄存器（BF16 × 2）
LDG.E R5,  desc[UR4][R16.64]        // B fragment[K=0]，第 1 个寄存器（BF16 × 2）
LDG.E R6,  desc[UR4][R18.64]        // A fragment[K=1]，第 1 个寄存器（BF16 × 2）

LDG.E R11, desc[UR4][R18.64+0x10]   // A fragment[K=1]，第 2 个寄存器（+16 字节 = K 方向下一段）
LDG.E R2,  desc[UR4][R14.64+0x10]   // A fragment[K=0]，第 2 个寄存器
LDG.E R3,  desc[UR4][R16.64+0x10]   // B fragment[K=1]，第 2 个寄存器

// ---- 阶段 2：2 条 HMMA 指令，对应 kMmaValExpandK=2 的两次 MMA atom ----
// HMMA.1688.F32.BF16：SM80_16x8x8，accumulator 为 F32，输入为 BF16
// 格式：HMMA dest(D), srcA(A), srcB(B), srcC(C)   →   D = A × B + C

HMMA.1688.F32.BF16 R4, R4, R6, RZ   // 第 1 条：D[R4] = A[K=0][R4] × B[K=0][R6] + 0（RZ=零寄存器，首次不累加）
HMMA.1688.F32.BF16 R4, R2, R11, R4  // 第 2 条：D[R4] = A[K=1][R2] × B[K=1][R11] + D[R4]（续接第 1 条结果，完成 K 规约）

// ---- 阶段 3：F32 accumulator 转 BF16，写回 GMEM ----
// F2FP.BF16.F32.PACK_AB：将两个 F32 转换并打包为一个 BF16×2 寄存器
F2FP.BF16.F32.PACK_AB R5, R5, R4    // 将 accumulator (R4) 的 F32 结果转为 BF16，与 R5 打包
F2FP.BF16.F32.PACK_AB R7, R7, R6    // 同上，处理另一部分

// K 方向扩展了 2 倍，但 D 仍是同一块 tile，STG 条数不变（仍为 2 条）
STG.E desc[UR4][R12.64], R5         // 将打包后的 BF16 结果写回 GMEM
STG.E desc[UR4][R2.64],  R7
```

值得注意的是，此时所有的访存指令前面都有一个感叹号，告诉我们这个访存指令多读取了 GMEM 的数据。原来，当 MMA 扩展至 Tile 级别时，访存的行为就发生了变化。我们会在下一篇的笔记中深入分析访存问题并给出解决方案。

> 以下 SASS 逐步拆解由 Claude 4.6 生成，用于辅助理解。

**第一步：搞清楚每个线程持有哪些数据**

`SM80_16x8x8` 对 32 个线程均摊数据：

| 矩阵 | Tile 大小 | 每线程元素数 | 每线程寄存器数 |
|------|-----------|-------------|--------------|
| A (BF16) | 16×8 | 4 个 BF16 | 2 个寄存器（每寄存器 packed 2 个 BF16） |
| B (BF16) | 8×8 | 2 个 BF16 | 1 个寄存器 |
| D (F32) | 16×8 | 4 个 F32 | 4 个寄存器 |

`kMmaValExpandK=2` → 需要 **2 套 A/B** 寄存器，共 6 个 → 对应 6 条 LDG。

**第二步：6 条 LDG 的寄存器分配**

```
K=0 的 MMA atom：A[K=0] → R4（reg1），R5（reg2）；B[K=0] → R6
K=1 的 MMA atom：A[K=1] → R2（reg1），R3（reg2）；B[K=1] → R11
```

地址规律：`R14.64` 和 `R16.64` 是 A 矩阵两组行的基址（对应线程持有的不同行），`R18.64` 是 B 矩阵基址；`+0x10`（= +16 字节 = +8 个 BF16）是在 K 方向跳过第一个 atom（K=0~7），取第二个 atom（K=8~15）的数据。6 条 LDG 集中在最前面发出，是为了**掩盖全局内存访问延迟**——GPU 在等待 load 完成期间可以调度其他 warp，数据到达后再执行 HMMA。

**第三步：2 条 HMMA 指令**

HMMA 的格式为 `HMMA dest, A, B, C`，语义是 `D = A × B + C`，其中 A 占 2 个连续寄存器，B 占 1 个寄存器，D/C 占 4 个连续寄存器。

```
HMMA.1688.F32.BF16 R4, R4, R6, RZ
  → D(R4~R7) = A[K=0](R4,R5) × B[K=0](R6) + 0
  → RZ 是零寄存器，第一条从零开始累积

HMMA.1688.F32.BF16 R4, R2, R11, R4
  → D(R4~R7) = A[K=1](R2,R3) × B[K=1](R11) + D（上一条输出）
  → 第二条的 C 用第一条的输出，完成 K 方向的规约
```

两条 HMMA 写的是**同一组** D 寄存器（R4~R7），第二条在第一条基础上继续累加，这就是"K 维度扩展不产生额外输出维度"的硬件实现。

**第四步：F2FP + STG 写回**

D 寄存器是 F32，但输出 C 矩阵是 BF16，需要先转换再存储。K 方向扩展了 2 倍，但输出 tile（M=16, N=8）大小不变，所以 STG 仍只有 **2 条**。

整体流水时序：

```
[LDG×6]──────────────→[HMMA K=0]→[HMMA K=1]→[F2FP×2]→[STG×2]
 ↑提前发出，掩盖内存延迟   ↑等数据就绪后串行累积      ↑转换写回
```

![图10: Tiled MMA 的部分 SASS code](![](https://files.mdnice.com/user/59/9670a9de-d5f7-4685-bcf0-6d7ec51ff210.png))

## 4. 总结

本篇重点介绍了 GEMM 的三级 Tiling ，并将单指令的 MMA Atom 扩展至由多个 mma 指令组成的 Tiled MMA，实现了 32x32x16 的 MMA 运算。

在后续几篇的笔记中，我们将以三级 Tiling 为主线，持续扩展 GEMM 的规模。并详细解析每一层级 Tiling 的实现细节和优化手段。

下一步，我们将深入 CUTLASS 的另一类 API：Tiled Copy，探究如何在 Tile 维度下完成高效的数据搬运工作，敬请期待！

# CUTLASS 笔记 (4)：Tiled Copy

本篇主要阐述了 CuTe TiledCopy 的核心原理，介绍 TiledCopy API 的各类参数和含义，并解析如何在 Tile 维度实现数据拷贝。此外，本篇还介绍了 GPU 全局内存访存特性的基础知识。

本篇所使用的 CUTLASS 版本为 4.1.0，硬件架构为 SM90。

**本笔记系列的相关代码已全部开源，代码仓库见：[cutlass-notes](https://github.com/ArthurinRUC/cutlass-notes)，欢迎大家多多 star～**

CUTLASS 笔记系列导读及文章列表详见：[CUTLASS 笔记：导读](https://zhuanlan.zhihu.com/p/1937220431728845963)

---

在上一篇笔记中，我们介绍了如何从单个 Tile 维度进行多 warp、多指令的矩阵运算，并详细解析了 CUTLASS CuTe 的其中一个重要 API，即 TiledMMA。

上一篇文章见：[CUTLASS 笔记 (3)：Tiled MMA](https://zhuanlan.zhihu.com/p/1950555644814946318)

> TiledMMA 本身包含了一系列复杂的 Tensor 分块逻辑（例如 partition_A/B/C、partition_fragment_A/B/C），为了理解这些分块是如何实现的，读者需要对 CuTe Layout 的运算有一定的认识。由于是否理解分块实现原理，一般并不影响 MMA 计算逻辑的编写，因此我们还未曾对其进行详细的介绍。我们会在后续笔记介绍 CuTe Layout 之后，进一步补充 TiledMMA 的细节。

在硬件进行实际的运算前，必须要将正确的数据放置于硬件上特定的存储单元。本篇笔记将解析 CUTLASS CuTe 的另一个重要 API，即 TiledCopy。TiledCopy 描述了单个 Tile 的数据在不同存储地址之间的拷贝逻辑，并提供类似于 TiledMMA 的矩阵分块逻辑，用于将拷贝任务分配给每个线程。

同时，我们也可以向 `copy` 这个 API 传入 TiledCopy 对象，使用 TiledCopy 描述的拷贝逻辑完成实际的拷贝任务。

```cpp
// Before
copy(copy_atom, tCgA, tCrA);

// After
copy(tiled_copy, tCgA, tCrA);
```

由于最佳的数据拷贝策略和硬件特性相关，我们首先需要了解 NV GPU 在访存侧的**硬件特性**，实现向量化访存和合并访存。为了确保向量化访存和合并访存能够正确执行，我们有必要深入了解 **TiledCopy 的核心原理**，清楚地知道每个线程的拷贝逻辑，从而写出高效的拷贝策略。

我们首先来介绍 GPU 上 GMEM 的访存特性，其他存储单元的特性将在后续笔记中逐步介绍。

## 1. NV GPU 的全局内存访存特性

为在 NV GPU 实现最佳的访存效率，通常有两种优化手段：**向量化访存**和**合并访存**。

### 1.1 向量化访存

向量化访存的目的是使用更大长度的访存指令，减少频繁的访存指令调度，提升指令级并行度，充分发挥现代内存系统的带宽潜力。

我们在第 1 篇 Minimal GEMM 的笔记中介绍了读取 GMEM 的指令。在 Minimal GEMM 场景中，读取 A 矩阵是用两个 32bit 长度的指令完成的，即 `ld.global.u32` 或 `LDG.E`。实际上单个 PTX/SASS 指令还可以支持 64bit、128bit 长度的访存指令，目前单指令最大访存量就是 128bit。

|      | PTX              | SASS      |
|------|------------------|-----------|
| 32bit  | `ld.global.u32`    | `LDG.E`     |
| 64bit  | `ld.global.v2.u32` | `LDG.E.64`  |
| 128bit | `ld.global.v4.u32` | `LDG.E.128` |

用两个 32bit 而不是一个 64bit 的访存指令，是因为在 Minimal GEMM 场景下，两个 32bit 数据块是不连续的。如果我们能有一种手段能将这两个数据块连续存放，那么编译器就会选用更长的 64bit 拷贝指令了。

![图1: 使用 2 个 LDG.E 完成 A 矩阵分片的拷贝](https://pic4.zhimg.com/v2-8851d0497f1bf147499c12575aa3d437_1440w.jpg)

因此在编写访存逻辑时，我们要尽可能地让单个线程的访存数据连续，以使用更大长度的访存指令。

### 1.2 合并访存

合并访存是硬件如何组织线程的内存访问模式，目的是将各个线程的访存指令合并成一系列的 transaction。当一个 warp 同时执行一个访存指令时，我们需要整个 warp 访问的数据也要尽可能**连续且对齐**，否则实际的 GMEM 访存量就可能增加。

众所周知，NV GPU 采用 SIMT 架构，一个 warp 上的 32 个线程会同时运行读取内存或写入内存的指令，硬件层面会将这个 warp 的一次访存请求合并为若干个 **transaction**。

**在 GMEM 层面，transaction 是硬件访存的最小单元**，一次 transaction 能够访问一段**连续且内存对齐的 32 bytes 数据**，通常称这段数据为 **sector**。从 GMEM 读取的 sector 数据同样会经过各级 Cache，无论这些数据是否实际被使用。当一次访存数据量小于 32 bytes，或者一次非连续、非对齐的访存涉及了多个 sector 时，实际的 GMEM 访存量就会大于访存指令所需的数据量。

如下图所示，当我们读取 0-384 这段 384 bytes 的数据时，硬件实际上会进行 `384 / 32 = 12` 次 transaction。当我们读取横跨多个 sector 的数据，或者读取非连续的多个 sector 时，被访问过的 sector 的**所有数据**实际上都会被读取并被写入各级缓存。

![图2: 各种访存情况示例](https://picx.zhimg.com/v2-5c84dfa2e0028604e74787716fe96695_1440w.jpg)

> 在较早架构的 GPU 硬件（SM60 以下），当访存经过 L1 Cache 时，一次 transaction 的访存数据量会变为 128 bytes。
>
> 在较新的架构（SM60 及以上），无论访存是否经过 L1 Cache，一个 transaction 的访存数据量固定为 32 bytes。

由此可见，如果访问的内存不连续、不对齐，实际的 GMEM 访存量就可能会增加，进而影响算子性能。

我们在上篇笔记的最后发现了 Tiled MMA 算子存在着访存问题，ncu 提示我们一些指令多读取了 GMEM 的数据。接下来我们运用上面的知识，分析产生这个问题的原因。

![图3: Tiled MMA 中的访存问题](https://pic1.zhimg.com/v2-bc919e348bec08c42a12697500f2ef92_1440w.jpg)

以 A 矩阵的访存为例，我们来关注第一个 warp 的访存情况。可以从下图中发现，第一个 warp T0-T31 所需的数据均在 A(0,0) 和 A(0,1) 区域，其中每一行的长度恰好等于一个 sector 的长度 32 bytes。因此在最优情况下，读取 A 矩阵的数据仅需读取 16 个 sector，即进行 16 次 transaction。

![图4: A 矩阵的 sector 分布](https://picx.zhimg.com/v2-39a71bfd675d20777a5c6890b79fc21b_1440w.jpg)

由于每个线程有 4 块不连续的数据，实际访存是通过 4 条 `LDG.E` 指令完成的。对于 A(0,0) 这块数据，我们需要用 2 个 `LDG.E` 指令读取 GMEM。

![图5: LDG.E 的访存不连续性](https://pica.zhimg.com/v2-3255d13671710343293365e1aeb5912e_1440w.jpg)

然而每一个 `LDG.E` 的访存是不连续的内存区域，这会导致一个 `LDG.E` 就读取了 8 个 sector 的数据，即使实际所需的数据只有读取数据的一半。因此 4 条 `LDG.E` 实际上读取了 32 个 sector 的数据，实际的 GMEM 访存量多了一倍（因为有 Cache，所以访存时间的增加应小于一倍）。这就是为什么 ncu 提示每一条 `LDG.E` 有 50% 的 GMEM 访存是多余的。

那么有什么解决方案吗？在不引入其他内存层级的条件下，我们要么按照向量化访存的思路，将每个线程不连续的数据块调整为连续的，要么按照合并访存的思路，将 K 维度的长度固定为 8，以此保证 `LDG.E` 的访存区域是内存连续的。

很遗憾的是，这两个方案都不是令人满意的方案。首先第一个思路实际上不可行，因为我们没有办法通过控制 MMA Permutation，也就是任意交换 MMA Atom 的行和列，来让同一个 MMA Atom 同一个线程对应的两个不连续的数据块变成连续的；而第二个思路会让我们无法在 K 维度扩展 MMA 规模，在处理大规模的矩阵运算时，性能是受限的。

> 我们可以让线程拷贝一段连续的数据，然后通过 warp 内线程数据交换来拿到每个线程各自需要的数据。但这种做法增加了一个数据交换的步骤，并且编程复杂度较大。

要彻底解决上面的访存问题，我们需要使用共享内存 SMEM。关于 SMEM 的特性和优化手段，我们将在后续的笔记中介绍。

---

在了解硬件访存特性后，我们从第一性原理出发，阐述 TiledCopy 这个重要 API 的核心原理。

## 2. 理解 TiledCopy 的核心原理

所有数据拷贝操作，归根到底可以总结为一行代码：

```cpp
dst = src;
```

即从数据源获取数据，并将其放置于目标位置。这里产生了两个关键问题：**从哪里获取数据？将数据放置在哪个目标位置？** 在计算机世界中，这两个问题的答案其实就是找到**源地址**和**目标地址**。找到了这两个地址，我们就可以获取源数据并将其写入目标位置。

```cpp
*dst_ptr = *src_ptr;
```

在 CuTe 中，数据以 Tensor 的结构存储。Tensor 由两部分组成，一部分是 **Tensor Data**，表示数据块的头指针，另一部分是 **Tensor Layout**，描述了如何通过 Tensor 的**逻辑坐标**找到相对于头指针的**内存地址偏移量（offset）**。当我们知道了 Tensor Data 和 Tensor Layout，也就知道了 Tensor 中任何数据的地址，也就可以访问这个 Tensor 的任何一个数据。

因此，**在 CuTe 中，我们天然可以完成两个 Tensor 的数据拷贝**，其本质上是通过 for 循环，遍历源 Tensor 的**每个坐标**，通过坐标计算出对应的地址，获取对应的源数据，随后将数据写入目标 Tensor 下**相同坐标**对应的地址。

```cpp
copy(dst, src);

// Equivalent to:
for (int i = 0; i < size(src); ++i) {
  dst(i) = src(i);
}
```

> 需要说明的是，**CuTe 中表示 Tensor 的坐标既可以表示为整数形式 idx，也可以表示为向量形式 `(m, n)`，两者等价且可以互相转化。**
>
> 我们在笔记（2）中有介绍过坐标转换的运算 `idx2crd`，这里再举一个例子，对于 Shape 为 `(3, 4)` 的 Tensor 来说，`idx = 8` 等价于 `idx = (2, 2)`。**后续的笔记中，我们对坐标进行形式的转换时不再特别指出。**

用一张示意图表示上面的拷贝过程：

![图6: CuTe 中的 Tensor 拷贝基本原理](https://picx.zhimg.com/v2-6813d93a9899c2e4cd6a08ad23318d61_1440w.jpg)

然而，这种拷贝模式并不能涵盖所有的 Tensor 拷贝场景。**所有源坐标和目标坐标不相等的情况，都无法使用 `copy(dst, src)` 这种简便的拷贝方法。**

例如倒序拷贝：

```cpp
for (int i = 0; i < size(src); ++i) {
  dst(size(src) - i - 1) = src(i);
}
```

移位拷贝：

```cpp
for (int i = 0; i < size(src); ++i) {
  dst((i + c) % size(src)) = src(i);
}
```

……以及各种各样奇形怪状的拷贝。

从本质上看，无论这些拷贝模式怎么变化，它们都建立了 `src` Tensor 坐标到 `dst` Tensor 坐标的映射，我们记这个映射为 `f`，含义为 `dst_idx = f(src_idx)`。那么任何一种拷贝模式都可以表示为：

```cpp
for (int i = 0; i < size(src); ++i) {
  dst(f(i)) = src(i);
}
```

当这个映射 `f` 为**恒等映射**时，我们可以使用 `copy(dst, src)` 这个 API 代替上面的 for 循环。

实际上，在 CuTe 中有一个隐含的假设：**映射 `f` 一定是恒等映射**。也就是说，**参与 copy 运算的 src 和 dst，在相同的坐标下一定对应着同一个数据**。因为我们总是能通过变换 dst Tensor 的映射，令 `dst' = dst \circ f`，使得 src 和 `dst'` 的坐标能够一一对应到相同的数据。

```cpp
dst' = composition(dst, f);

for (int i = 0; i < size(src); ++i) {
  dst'(i) = src(i);
}
```

因此在后续的笔记中，我们可忽略 `f` 的存在，始终用 `dst(i) = src(i)` 表示拷贝操作。

> 由于类似 `ldmatrix` 的指令会做线程间数据交换，因此无法简单将所有拷贝模型定义为 `dst(i) = src(i)`，更准确的表达方式是 **`copy_instruction(dst(i), src(i))`**，即将 src 的数据交给拷贝指令，并将指令返回的数据放在 dst。
>
> 但为了表述方便，我们后续将仍然使用 `dst(i) = src(i)` 来表示拷贝操作。读者需注意 `dst = src` 在不同拷贝指令下，底层的运行机制会有所差异。

由于在 SIMT 架构下，每个线程需要拿到各自的数据分块。因此，我们不是直接通过坐标去索引 Tensor，而是用 `(t, v)` 二元组来索引 Tensor。因此拷贝代码应表示为：

```cpp
for (int t = 0; t < size<0>(src); ++t) {
  for (int v = 0; v < size<1>(src); ++v) {
    dst(t, v) = src(t, v);
  }
}
```

这是从全局角度来看的拷贝。从线程的角度看，每个线程需要从 src 和 dst 拿到自己的数据分块，然后对数据分块做 for 循环拷贝：

```cpp
int t = threadIdx.x;

src_frg = src(t, _);
dst_frg = dst(t, _);

copy(dst_frg, src_frg);

// Equivalent to:
for (int v = 0; v < size(src_frg); ++v) {
  dst_frg(v) = src_frg(v);
}
```

**那么现在的关键问题是，我们应该如何构造这个 src 和 dst，使得我们可以使用上面的 `copy(dst_frg, src_frg)` 这个简单的 API 完成拷贝呢？**

既然拷贝操作的本质是 `dst(i) = src(i)`，**如果我们能够变换 src 和 dst，让 src 的 `(t, v)` 和 dst 的 `(t, v)` 都映射到同一个坐标 `i` 下，那不就可以用 `src(t, v) = dst(t, v)` 来拷贝了**？这里的映射其实可以通过两个 TV Layout 来表示。

- **Src TV Layout** 记为 `s`，含义为 `s(src_t, src_v) = src_idx`，描述每个线程应该从 src 读取哪个坐标下的数据；
- **Dst TV Layout** 记为 `d`，含义为 `d(dst_t, dst_v) = dst_idx`，描述每个线程应该将数据写入 dst 的哪个坐标。

有了这两个 TV Layout，正常的思路是，src 拿到 `(t, v)` 也就是 `(src_t, src_v)`，通过 Src TV Layout 映射为 `src_idx`，而 dst 的 `(t, v)` 通过 Dst TV Layout 映射为 `dst_idx`。**问题在于，我们怎么确保对于任何相同的 `(t, v)`，经过映射得到的 `src_idx` 和 `dst_idx` 是一样的呢？更进一步说，我们应该如何去建立起 `(src_t, src_v)` 和 `(dst_t, dst_v)` 之间的联系？**

这个联系的桥梁必然是**数据**。我们为什么要让 `(src_t, src_v)` 和 `(dst_t, dst_v)` 映射到同一个 idx，本质上是希望它们能映射到同一份数据。而**拷贝操作的数据移动必然是不改变数据的**，所有 src 里面的数据都能在 dst 里面找到对应，反之亦然。这是拷贝操作的本质特征，无论数据的存储介质、存储地址是什么，内存是否连续、是否对齐，都不会改变这个本质特征。因此，**我们在建立 `(src_t, src_v)` 和 `(dst_t, dst_v)` 的映射时，中间的桥梁一定是具体的数据，而不是坐标、地址偏移 offset、位置等其他的信息**。

所以我们可以给参与拷贝的数据编个序号（ID），比如说从 0 到 N - 1 编号，然后分别建立起 `(src_t, src_v)` 到 ID、`(dst_t, dst_v)` 到 ID 的映射，就可以建立 `(src_t, src_v) <-> ID <-> (dst_t, dst_v)` 的联系了。而这个联系，**必然是由拷贝指令的本质特性决定的**，只有拷贝指令能确定哪个线程的哪个数据能拷贝到哪个线程的哪个数据，程序无法改变这些映射，因此上面的两个映射一定是在 CopyAtom 中记录的。

![图7: SrcLayout 和 DstLayout](https://pic2.zhimg.com/v2-26454e1956844788900f7c479be1dd5b_1440w.jpg)

在 CopyAtom 记录的两个 Layout 映射分别称为 **SrcLayout** 和 **DstLayout**。

- **SrcLayout** 的映射记为 `S`，含义为 `S(src_t, src_v) = ID`；
- **DstLayout** 的映射记为 `D`，含义为 `D(dst_t, dst_v) = ID`。

那么 **`(src_t, src_v)` 到 `(dst_t, dst_v)` 的映射，就可以表示为一个复合函数 `D^{-1} \circ S`**，其中 `D^{-1}` 代表 `D` 的逆映射，即 `ID -> (dst_t, dst_v)`。同理 `S^{-1} \circ D` 的复合函数建立了 `(dst_t, dst_v)` 到 `(src_t, src_v)` 的映射。

> 这里补充一点：**常规**的拷贝操作，也就是类似于 `dst = src` 的代码，是不会在线程之间交换数据的，同一个线程获取的数据在搬运过程中始终不变，此时 `(src_t, src_v)` 与 `(dst_t, dst_v)` 始终是相等的，所以不管是 `D^{-1} \circ S` 还是 `S^{-1} \circ D` 都是恒等映射。然而 **NV GPU 有部分拷贝指令（例如 `ldmatrix`）会进行线程间的数据交换**，某个线程 `(src_t, src_v)` 拿到的数据，最终会被另一个线程 `(dst_t, dst_v)` 拿到，因此对于这些指令，上面的映射就不是恒等映射了。

于是，如果我们将 `(src_t, src_v)` 映射到 `(dst_t, dst_v)`，或者反过来，将 `(dst_t, dst_v)` 映射到 `(src_t, src_v)`，那么就可以**统一 src 和 dst 的 `(t, v)` 空间**，再用**同一个** TV Layout 就可以将映射到 idx 了。这样就实现了 `(src_t, src_v)` 和 `(dst_t, dst_v)` 映射到同一个 idx 的目标。

![图8: src (t, v) 和 dst (t, v) 映射至同一个 idx](https://pic1.zhimg.com/v2-bcb64d8939a177e18413c3dc73774100_1440w.jpg)

那么到底是选择 src `(t, v)` 转成 dst `(t, v)`，还是 dst `(t, v)` 转成 src `(t, v)` 呢？按理其实都可以，毕竟只要对齐 `(t, v)`，而这个 `(t, v)` 具体是在 src 空间还是 dst 空间不是很重要。**但选用不同的转化方式，我们要记录的 TV Layout 也不一样，实际上哪个 TV Layout 容易获取或者容易计算，我们就用哪个，并选择对应的 `(t, v)` 映射方式。**

**在 CuTe 中定义了一个映射 `R`，它等于 `S` 和 `D` 的其中一个映射，具体定义为 `R(ref_t, ref_v) = ID`。** 当 `R = S` 时，就是 `R(src_t, src_v) = ID`；当 `R = D` 时，就是 `R(dst_t, dst_v) = ID`。

当我们将 `R^{-1}` 和 `S` 变为一个复合映射时，这个复合映射其实就是 `(src_t, src_v) -> (ref_t, ref_v)`，同理将 `R^{-1}` 和 `D` 复合就变成 `(dst_t, dst_v) -> (ref_t, ref_v)` 的映射。**也就是说，当 `(src_t, src_v)` 经过一个 `R^{-1} \circ S` 的映射，`(dst_t, dst_v)` 经过一个 `R^{-1} \circ D` 的映射后，两个 `(t, v)` 都会统一到同一个 `(t, v)` 空间，变为 `(ref_t, ref_v)`。**

- 当 `R = S` 时，`(src_t, src_v)` 经过的是恒等映射，结果仍然是 `(src_t, src_v)`，而 `(dst_t, dst_v)` 转化成 `(src_t, src_v)`，此时 `(ref_t, ref_v)` 等于 `(src_t, src_v)`，我们需要记录 Src TV Layout 来将 `(src_t, src_v)` 映射为 idx 坐标；
- 同理，当 `R = D` 时，两个 `(t, v)` 最后都变成了 `(dst_t, dst_v)`，`(ref_t, ref_v)` 等于 `(dst_t, dst_v)`，我们需要记录 Dst TV Layout 来将 `(dst_t, dst_v)` 映射为 idx 坐标。

之后，我们将记录的这个 TV Layout 称为 **Ref TV Layout**，记为 `r`，它等于 `s` 和 `d` 的其中一个。

对于不同拷贝指令来说，这个 Ref TV Layout 的获取难易程度也不同，因此 `R` 到底是等于 `S` 还是等于 `D` 和拷贝指令也相关，所以 `R` 同 `S`、`D` 一样，也被记录在 CopyAtom 中。

**回到我们的关键问题，为完成拷贝操作，新的 src 和 dst 应该怎么构建呢？** 对于 src，我们从 `(src_t, src_v)` 出发，经过 `R^{-1} \circ S` 映射到 `(ref_t, ref_v)`，再通过 Ref TV Layout 映射到 idx，这个 idx 最后传给 Src Tensor Layout，也就是原本的 src，来获取数据在 src 中的实际地址。那么最终的 `src' = src \circ r \circ R^{-1} \circ S`。同理有 `dst' = dst \circ r \circ R^{-1} \circ D`。读者可以从源码看到，**这个复合 Layout 的构建过程，其实就是 TiledCopy 中 `partition_S` 和 `partition_D` 的原理**。

并且，我们可以借助 Ref TV Layout 来获取 Src TV Layout 和 Dst TV Layout：

- `s = r \circ R^{-1} \circ S`
- `d = r \circ R^{-1} \circ D`

当 `R = S` 时，有 `s = r`，即 Ref TV Layout 就是 Src TV Layout；`R = D` 时同理。

以上就是 TiledCopy 的核心原理，我们用一张图来总结。

![图9: TiledCopy 的核心原理图](https://pica.zhimg.com/v2-36a3f50276940c682da4109df468b4b0_1440w.jpg)

---

接下来，我们介绍 CuTe 中与拷贝相关的重要 API。

## 3. Tiled Copy 实现

### 3.1 Copy_Traits 与 CopyAtom

根据 TiledCopy 的核心原理，我们需要在指令层级记录三个映射：`S`、`D` 和 `R`，也就是 `SrcLayout`、`DstLayout`、`RefLayout`，其中前两个 Layout 代表了这个拷贝指令的本质特征，`RefLayout` 则规定了使用这个指令需要传入哪种 TV Layout。

以下是 `SM75_U32x4_LDSM_N` 这个指令的 Copy_Traits。

```cpp
template <>
struct Copy_Traits<SM75_U32x4_LDSM_N>
{
  // Logical thread id to thread idx (warp)
  using ThrID = Layout<_32>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape < _32,_128>,
                           Stride<_128,  _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_32,Shape <_32,   _4>>,
                           Stride<_32,Stride< _1,_1024>>>;

  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
};
```

CopyAtom 从 Traits 里面获取这些 Layout，并根据数据类型做一些处理工作，类似于 MMA Atom 和 MMA Traits 的协作模式。

```cpp
template <class... Args, class CopyInternalType>
struct Copy_Atom<Copy_Traits<Args...>, CopyInternalType>
  : Copy_Traits<Args...>
{
  using Traits = Copy_Traits<Args...>;

  // Bit and Thr layouts from the Copy_Traits
  using ThrID        = typename Traits::ThrID;
  using BitLayoutSrc = typename Traits::SrcLayout;
  using BitLayoutDst = typename Traits::DstLayout;
  using BitLayoutRef = typename Traits::RefLayout;

  using ValType = CopyInternalType;

  using ValLayoutSrc = decltype(recast_layout<uint1_t, ValType>(BitLayoutSrc{}));
  using ValLayoutDst = decltype(recast_layout<uint1_t, ValType>(BitLayoutDst{}));
  using ValLayoutRef = decltype(recast_layout<uint1_t, ValType>(BitLayoutRef{}));

  ...
}
```

### 3.2 TiledCopy 与 ThrCopy

TiledCopy 在 CopyAtom 基础上进行了**线程**和**数据**两个维度的扩展，这点和 TiledMMA 类似，我们不再赘述。

同时，TiledCopy 也记录了 **Ref TV Layout**，用于对 src 和 dst 进行变换，也就是下面的 `TiledLayout_TV`。线程和数据维度的扩展，就是通过扩展 `TiledLayout_TV` 的 T 维度和 V 维度来实现的。

`Tiler_MN` 是这个 TiledCopy 拷贝的总规模，是一个 Layout。这里的 `Tiler_MN` 同样可以代表线程和数据维度的扩展，因为无论是 `Tiler_MN` 还是 `TiledLayout_TV`，它们的 size 都是拷贝数据的个数。

```cpp
template <class Copy_Atom,
          class LayoutCopy_TV,  // (tid,vid) -> coord   [Need not be 2D...]
          class ShapeTiler_MN>  // coord space
struct TiledCopy : Copy_Atom
{
  // Layout information from the CopyAtom
  using AtomThrID     = typename Copy_Atom::ThrID;        // thrid -> thr_idx
  using AtomLayoutSrc = typename Copy_Atom::ValLayoutSrc; // (thr,val) -> offset
  using AtomLayoutDst = typename Copy_Atom::ValLayoutDst; // (thr,val) -> offset
  using AtomLayoutRef = typename Copy_Atom::ValLayoutRef; // (thr,val) -> offset

  using AtomNumThr = decltype(size<0>(AtomLayoutRef{}));
  using AtomNumVal = decltype(size<1>(AtomLayoutRef{}));

  // Layout information for the TiledCopy
  using Tiler_MN       = ShapeTiler_MN;
  using TiledLayout_TV = LayoutCopy_TV;
  using TiledNumThr    = decltype(size<0>(TiledLayout_TV{}));
  using TiledNumVal    = decltype(size<1>(TiledLayout_TV{}));

  ...
}
```

类似于 ThrMMA，**ThrCopy** 也提供了 Tensor 分块能力，让每个线程都能拿到自己的任务块。ThrCopy 提供 `partition_S`、`partition_D` 两个 API 完成分块操作。`partition_S` 的工作是将输入的全局 Tensor **src** 经过上面提到的一系列映射转化为 **src'**，同时选取当前线程对应的数据分块；`partition_D` 同理。

```text
TiledCopyA g2r_tiled_copy_a;
ThrCopy g2r_thr_copy_a = g2r_tiled_copy_a.get_slice(tid);

Tensor tAgA = g2r_thr_copy_a.partition_S(gA);    // (CPY, CPY_M, CPY_K)
```

> **原错误内容：** 类似于 `(MMA, MMA_M, MMA_K)`，这里的 CPY 是每个线程每个拷贝指令参与的数据量，CPY_M 和 CPY_K 是 TiledCopy 在 M 和 K 维度的数据扩展规模。

不同于 `(MMA, MMA_M, MMA_K)`，这里的 CPY 指的是**每个线程拷贝单个 Tile 的总数据量**，而并非每个线程每个 Copy Atom 的数据量；而 **CPY_M 和 CPY_K 指的是从 Block 层面看，M 和 K 方向扩展了多少个 Tile**。因为我们还没有将 Tile 扩展到 Block，因此这里的 CPY_M 和 CPY_K 都是 1。

对于经过 Ref TV Layout 映射并分块得到的 Tensor，ThrCopy 的另一个功能是改变这个 Tensor 的 Layout 排布，在保持 size 不变的情况下适配后续拷贝操作的 Shape。ThrCopy 提供了 `retile_S`、`retile_D` 两个 API。

```text
Tensor tCgA = thr_mma.partition_A(gA);  // (MMA, MMA_M, MMA_K)
Tensor tCrA = thr_mma.partition_fragment_A(gA);  // (MMA, MMA_M, MMA_K)

Tensor tAgA = g2r_thr_copy_a.retile_S(tCgA);     // (CPY, CPY_M, CPY_K)
Tensor tArA = g2r_thr_copy_a.retile_D(tCrA);     // (CPY, CPY_M, CPY_K)

copy(g2r_tiled_copy_a, tAgA, tArA);
```

> `partition_S`/`partition_D`，以及 `retile_S`/`retile_D` 的代码解析，我们在解析完 CuTe Layout 之后再来介绍。其实它们就是 TiledCopy 核心原理的工程实现。

### 3.3 make_tiled_copy API

![图10: make_tiled_copy API](https://pica.zhimg.com/v2-5e5344d1475f682ee8f508832c7089bc_1440w.jpg)

`make_tiled_copy` 和 `make_tiled_mma` 类似，也接收三个参数，分别是 **CopyAtom**、**ThrLayout** 和 **ValLayout**。其中 ThrLayout 表示线程扩展方式，ValLayout 表示数据扩展方式。在 API 内部会根据 ThrLayout 和 ValLayout 计算出 TiledCopy 所需的 Ref TV Layout 和拷贝规模 `Tiler_MN`。

由于 TiledMMA 本身就记录了 A、B、C 矩阵的 TV Layout，当我们需要的 Ref TV Layout 恰好就是 TiledMMA 里面的 TV Layout 时，我们还可以直接将 TiledMMA 实例传给 `make_tiled_copy_A/B/C`，来创建一个 TiledCopy，实质上是将 TiledMMA 对应矩阵的 TV Layout 作为 Ref TV Layout，将 TiledMMA 的数据规模作为 TiledCopy 的数据规模。

```text
using Copy_op = AutoVectorizingCopy;
using CopyA_atom = Copy_Atom<Copy_op, ComputeTypeA>;

using TiledCopyA = decltype(make_tiled_copy_A(CopyA_atom{}, TiledMMA{}));
```

> 什么情况下 Ref TV Layout 恰好是 TiledMMA 的 TV Layout 呢？其实就是在拷贝完成后马上交给 MMA 指令运算的情况下，或者完成 MMA 计算后马上拷贝出去的情况下，因为当这个拷贝的 Ref TV Layout 使用 MMA 的 TV Layout 后，我们就可以在不改变 Tensor 排布的情况下直接拿拷贝过来的 Tensor 用于 MMA 计算，或者算完之后直接拷贝出去。

### 3.4 算子代码示例

本篇的算子详情和上篇笔记一致。代码层面与 TiledMMA 的区别仅在创建和使用 TiledCopy 部分。

| 项目 | 数值 |
|------|------|
| 问题规模 | `(32, 32, 16)` |
| 算子精度 | `BF16 = BF16 * BF16 + FP32` |
| Grid shape | `(1, 1, 1)` |
| Block shape | `(256, 1, 1)` |
| Block tile shape | `(32, 32, 16)` |
| Tiled MMA shape | `(32, 32, 16)` |
| MMA atom shape | `(16, 8, 8)` |

由于我们只需要完成从 GMEM 到寄存器的拷贝，并且只完成 MMA 运算，因此可以直接用 `make_tiled_copy_A/B/C` 创建 TiledCopy 模版类。

此处，我们仍然采用常规的拷贝指令，并尽可能选用最大的向量化访存长度，因此选择 `AutoVectorizingCopy` 作为 Copy op 指令。

```cpp
using Copy_op = AutoVectorizingCopy;

using CopyA_atom = Copy_Atom<Copy_op, ComputeTypeA>;
using CopyB_atom = Copy_Atom<Copy_op, ComputeTypeB>;
using CopyC_atom = Copy_Atom<Copy_op, ComputeTypeC>;
using CopyO_atom = Copy_Atom<Copy_op, OutType>;

using TiledCopyA = decltype(make_tiled_copy_A(CopyA_atom{}, TiledMMA{}));
using TiledCopyB = decltype(make_tiled_copy_B(CopyB_atom{}, TiledMMA{}));
using TiledCopyC = decltype(make_tiled_copy_C(CopyC_atom{}, TiledMMA{}));
using TiledCopyO = decltype(make_tiled_copy_C(CopyO_atom{}, TiledMMA{}));
```

在 kernel 内部，我们以 A 矩阵为例，首先创建 TiledCopy 实例，根据当前线程创建 ThrCopy 实例，随后通过 `partition_S/D` 或者 `retile_S/D` 方法获取当前线程负责拷贝的 Tensor 分块，我们先前已经有了 TiledMMA 的 Tensor 分块，这里只用 `retile_S/D` 方法就可以了。

```cpp
TiledCopyA g2r_tiled_copy_a;
ThrCopy g2r_thr_copy_a = g2r_tiled_copy_a.get_slice(tid);
Tensor tAgA = g2r_thr_copy_a.retile_S(tCgA);     // (CPY, CPY_M, CPY_K)
// Equivalent to:
// Tensor tAgA = g2r_thr_copy_a.partition_S(gA);    // (CPY, CPY_M, CPY_K)
Tensor tArA = g2r_thr_copy_a.retile_D(tCrA);     // (CPY, CPY_M, CPY_K)
```

按照 TiledCopy 的核心原理，这对分块可以直接交给 copy API，本质上是 for 循环来完成拷贝操作。在这里，我们需要传入 ThrCopy 对象，使用其中的拷贝指令完成拷贝操作。

```cpp
copy(g2r_tiled_copy_a, tAgA, tArA);
```

注意拷贝完成后，我们仍然需要使用 TiledMMA 的 Tensor 分块来完成 MMA 运算（尽管在这个示例中，TiledCopy 分块和 TiledMMA 分块是相同的）。

```cpp
gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);
```

计算出结果后，还需要将 C 矩阵分块拷贝回 GMEM，代码与上面类似，仅仅是拷贝方向的改变。

```cpp
TiledCopyO r2g_tiled_copy_o;

ThrCopy r2g_thr_copy_o = r2g_tiled_copy_o.get_slice(tid);
Tensor tCrC_r2g = r2g_thr_copy_o.retile_S(tCrC);   // (CPY, CPY_M, CPY_N)
Tensor tCgC_r2g = r2g_thr_copy_o.retile_D(tCgC);   // (CPY, CPY_M, CPY_N)

copy(r2g_tiled_copy_o, tCrC_r2g, tCgC_r2g);
```

### 3.5 metadata 解析和 Latex 图解

在 cute 中，我们可以打印出 TiledCopy 的信息，也可以用 latex 可视化这个 TiledCopy。

```cpp
cute::print(typename Spec::TiledCopyA{});
cute::print_latex(typename Spec::TiledCopyA{});
```

TiledCopy metadata 展示了所有关键的 Layout 信息，我们在核心原理部分已经详细介绍过。下面是 TiledCopyA 的 metadata：

```text
TiledCopy
  Tiler_MN:       (_32,_16)
  TiledLayout_TV: ((_4,_8,_2,_4),((_2,_2,_2),(_1,_1))):((_64,_1,_16,_0),((_32,_8,_256),(_0,_0)))
Copy_Atom
  ThrID:        _1:_0
  ValLayoutSrc: (_1,_1):(_0,_0)
  ValLayoutDst: (_1,_1):(_0,_0)
  ValLayoutRef: (_1,_1):(_0,_0)
  ValueType:    16b
```

用 latex 可视化 TiledCopyA，得到以下的图片。其中左侧为 **Src MN Layout**，也就是 Src TV Layout 的逆映射，右侧是 Dst MN Layout，左右两侧相同的坐标对应着同一个数据。在这个例子中，两个 MN Layout 是完全相同的，并且与 TiledMMA 的 A TV Layout 完全相同，读者可以和笔记（3）中的 latex 图进行比较。

两个 MN Layout 完全相同，意味着拷贝没有线程间的数据交换，每一个线程只负责自己所需数据的拷贝工作。根据核心原理，**`s = d` 等价于 `S = D`**，因此我们所用的拷贝指令的 SrcLayout 和 DstLayout 也是完全相同的，这点也可以从上面的 metadata 中看出来。

![图11: TiledCopy latex 可视化](https://picx.zhimg.com/v2-0525da556047cdfc5092a4aea93c5d3b_1440w.jpg)

本篇示例代码的 PTX 和 SASS 没有发生变化，此处不再对其进行分析。

## 4. 总结

本篇笔记的重点是 **TiledCopy 的核心原理**。理解其中各个 Layout 映射的作用，以及映射之间的复合和变换，就可以把握住潜藏在 API 下的底层机制，向理解 Layout 运算、灵活运用 CuTe Layout 代数编写算子逻辑迈出了坚实的一步！

下一步，我们将把问题规模从 Tile 维度扩展至 Block 维度，在新的 Tiling 层级下探究如何实现高效的运算，敬请期待！

下一篇文章见：[CUTLASS 笔记 (5)：Block MMA](https://zhuanlan.zhihu.com/p/1970162570636816559)

**最后，感谢您阅读到这里！如果您觉得本文对您有帮助，就点个赞吧，谢谢～**
