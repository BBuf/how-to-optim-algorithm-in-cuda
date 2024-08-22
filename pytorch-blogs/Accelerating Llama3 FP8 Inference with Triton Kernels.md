> 翻译：https://pytorch.org/blog/accelerating-llama3/

## Accelerating Llama3 FP8 Inference with Triton Kernels

> by Adnan Hoque, Less Wright, Chih Chieh Yang 

### 1.0 总结

我们提出了一种优化的Triton FP8 GEMM（通用矩阵乘法）kernel TK-GEMM，它利用了SplitK并行化。对于小batch size推理，在NVIDIA H100 GPU上针对Llama3-70B，TK-GEMM相比基础Triton矩阵乘法实现可提供高达**1.94**倍的性能提升，比cuBLAS FP8快**1.87**倍，比**cuBLAS FP16**快1.71倍。

![](https://files.mdnice.com/user/59/1c02cbba-f642-4517-baae-f8aebece1834.png)

**图1. TK-GEMM相对于PyTorch（调用cuBLAS）在Llama3-70B注意力层矩阵形状下的加速比（N=K=8192）**

在这篇博客中，我们将介绍如何使用Triton为FP8推理设计一个高效的kernel，并针对Llama3-70B推理进行调优。我们将讨论FP8（8位浮点数），这是Hopper代GPU（SM90）支持的新数据类型，Triton支持的SM90的关键特性，以及我们如何修改并行化以便能够为内存受限（推理）问题规模最大化内存吞吐量。

我们还专门讨论了CUDA Graph，这是一项重要技术，它将有助于实现kernel级别的加速，并使希望在生产环境中使用Triton kernel的开发人员能够获得额外的性能提升。

代码仓库和源码可在以下地址获取：https://github.com/pytorch-labs/applied-ai

### 2.0 FP8数据类型

FP8数据类型是由Nvidia、Arm和Intel联合推出的，作为16位浮点类型的继承者。由于位数减半，它有潜力为Transformer网络提供比其前身显著的吞吐量改进。FP8数据类型包含两种格式：

**E4M3**（4位指数和3位尾数）。能够存储+/-448和NaN（非数值）。
**E5M2**（5位指数和2位尾数）。能够存储  +/-57,334、NaN（非数值）和inf（无穷大）。

![](https://files.mdnice.com/user/59/5cf0d6d8-e414-41b5-a14a-34633fa34518.png)

我们在推理和前向传播训练中使用E4M3，因为它具有更高的精度；在反向传播训练中使用E5M2，因为它具有更高的动态范围。Nvidia已经设计了他们的H100 FP8 Tensor Core，可以提供高达3958 TFLOPS的峰值性能，是FP16 Tensor Core的**2倍**FLOPS。

我们在设计Triton kernel时考虑了这些硬件创新，在博客的其余部分，我们将讨论利用这些特性的方法，并验证这些特性确实被Triton编译器所使用。

### 3.0 Triton对Hopper的支持和FP8 Tensor Core指令

Hopper GPU架构增加了以下新特性，我们预计这些特性将加速FP8 GEMM。

- TMA（张量内存加速器）硬件单元
- WGMMA（线程束组矩阵乘法-累加指令）
- Threadblock Clusters（线程块簇）

Triton目前利用了其中一个特性，即wgmma指令，而PyTorch（调用cuBLAS）则利用了全部3个特性，这使得这些加速效果更加令人印象深刻。为了充分利用Hopper FP8 Tensor Core，尽管仍然支持旧的mma.sync指令，但wgmma是必要的。

mma和wgmma指令之间的关键区别在于，不是由1个CUDA线程束负责一个输出分片，而是整个线程束组（4个CUDA线程束）异步地贡献于一个输出分片。

为了看看这个指令在实践中是什么样子，并验证我们的Triton kernel确实在利用这个特性，我们使用nsight compute分析了PTX和SASS汇编。

![](https://files.mdnice.com/user/59/b5321070-1d9c-4630-b7b8-b80a2ce4f490.png)

这两条指令都告诉我们，我们正在将两个FP8 E4M3格式的输入张量相乘，并在F32（32位浮点数）中进行累加。这确认了TK-GEMM kernel确实在利用FP8 Tensor Core，并且lowering正在正确进行。

### 4.0 SplitK 工作分解

![](https://files.mdnice.com/user/59/e3e62990-9df5-4d0e-92c3-6ce5d60f698d.png)

基础的Triton FP8 GEMM实现在小M范围内表现不佳(https://github.com/triton-lang/triton/issues/3104)，即对于矩阵乘法A (MxN) x B (NxK)，当M < N, K时。为了优化这种矩阵配置，我们应用了SplitK工作分解，而不是在基础Triton kernel中的数据并行分解。这大大改善了小M范围内的延迟。

作为背景介绍，SplitK沿k维度启动额外的线程块来计算部分输出和。然后使用原子归约对每个线程块的部分结果进行求和。这允许更细粒度的工作分解，从而带来性能改进。关于SplitK的更多详细信息可以在我们的arxiv论文(https://arxiv.org/abs/2402.00025)中找到。

在仔细调整了我们 kernel的其他相关超参数（如tile大小、线程束数量和流水线阶段数）以适应Llama3-70B问题规模后，我们能够相对于Triton基础实现(https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)产生高达1.94倍的加速。有关超参数调优的更全面介绍，请参阅我们的博客(https://pytorch.org/blog/accelerating-moe-model/#30-work-decomposition---splitk)。

![](https://files.mdnice.com/user/59/44d86182-df16-47a8-abd9-3058b08818c1.png)

请注意，从M=32开始，cuBLAS FP8 kernel的性能开始超过TK-GEMM。对于M >= 32的情况，我们怀疑我们找到的超参数不是最优的，因此需要另一组实验来确定中等大小M范围的最佳参数。

### 5.0 CUDA GRAPHS的端到端加速

为了能在端到端的设置中实现这些加速效果，我们必须同时考虑 kernel执行时间（GPU持续时间）和wall时间（CPU+GPU持续时间）。Triton kernel是手写的（与torch compile生成的相对），已知会遭受高 kernel启动延迟的影响。如果我们使用torch profiler来跟踪TK-GEMM kernel，我们可以看到CPU端的调用栈，从而精确定位导致速度减慢的原因。

![](https://files.mdnice.com/user/59/45269de7-83bc-46a7-b9c4-21be872acec0.png)

从上面可以看出，我们优化后的kernel的大部分wall时间都被即时（JIT）编译开销所主导。为了解决这个问题，我们可以使用CUDA graphs。

![](https://files.mdnice.com/user/59/0d2c1732-518e-469e-a8a9-89f9b0edc028.png)

关键思想是，我们不是多次启动 kernel，而是创建和实例化一个graph（一次性成本），然后提交该graph的实例来执行。为了说明这一点，我们模拟了一个Llama3-70B注意力层。如下图所示（使用nsight systems生成），由于CPU kernel启动开销，每个GEMM之间的时间间隔为165微秒，而实际矩阵乘法只花费12微秒。这意味着在注意力层中，92%的时间GPU是空闲的，没有做任何工作。

![](https://files.mdnice.com/user/59/a41164a7-fc44-4eb9-acf2-5d8af6c356a2.png)

为了展示CUDA Graph的影响，我们在这个简单的注意力层中为TK-GEMM kernel创建了一个图，并replayed了该graph。从下图可以看到， kernel执行之间的间隔减少到了6.65微秒。

![](https://files.mdnice.com/user/59/f0a0a8e6-a2f7-49a1-bd73-e278ec19a466.png)

在实践中，与在不使用CUDA Graphs的模型中简单使用TK-GEMM相比，这种优化将使Llama3-70B中的单个注意力层获得6.4倍的加速。

### 6.0 潜在的未来优化路径

![](https://files.mdnice.com/user/59/5e586db5-ac7b-4eeb-a2c4-110f230a9ff7.png)

Nvidia H100 配备了 TMA（张量内存加速器）硬件单元。专用的 TMA 单元释放了寄存器和线程来执行其他工作，因为地址生成完全由 TMA 处理。对于内存受限的问题规模，当 Triton 启用对这一特性的支持时，这可以提供更进一步的性能提升。

![](https://files.mdnice.com/user/59/fb8e8b48-d6f1-40ed-974f-fe762a80d37a.png)

为了确定我们对Tensor Core的利用程度，我们可以分析RoofLine图。注意到，正如我们对小M值所预期的那样，我们处于内存受限区域。要改善kernel延迟，我们可以增加算术强度（在固定问题规模的情况下，只能通过利用数据局部性和其他循环优化来实现）或增加内存吞吐量。这需要一个更优化的并行算法，专门针对FP8数据类型以及我们在FP8推理中预期看到的问题规模特征。

![](https://files.mdnice.com/user/59/bb0d9d30-e3e5-4ca6-a16f-bb601b90a554.png)

> DRAM吞吐量已圈出，在H100上为1.65TB/s，相比峰值3.35TB/s（M=16，N=8192，K=8192）

最后，我们可以看到在NVIDIA H100上我们只达到了峰值DRAM吞吐量的约50%。高性能GEMM kernel通常能达到峰值吞吐量的70-80%。这意味着还有很大的改进空间，而且需要运用上述提到的技术（如循环展开、优化并行化）来获得额外的性能提升。

### 7.0 将来工作

对于未来的研究，我们希望探索CUTLASS 3.x和CuTe，以便更直接地控制Hopper架构的特性，特别是在获得对TMA的直接控制和探索pingpong架构方面，这些在FP8 GEMM中已经显示出了有前景的结果。














