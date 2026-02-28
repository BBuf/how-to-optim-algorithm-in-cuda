> 博客来源：https://pytorch.org/blog/accelerating-gemms-triton/ 这里做了翻译。这篇博客主要讲了如何用 Triton 来优化 Float8 格式的矩阵乘法(GEMM)运算。文章提出了一个叫 GridQuant 的方法，通过把大矩阵分成 256x256 的小块，然后再把每个小块分成更小的 32x32 的格子来处理数据。这种方法比之前的方案快了将近两倍。另外，文章还介绍了三个新技术：Warp 专门化、TMA（张量内存加速器）和持久化kernel，这些技术让不同的计算任务可以更好地并行执行，充分利用 GPU 的硬件特性。通过这些优化，在某些特定场景下比之前最好的方案又快了约 1.2 倍，特别适合用在大语言模型的推理阶段。不过这里的Triton代码还没有开源。

# 使用 Triton 加速 2D 动态块量化 Float8 GEMM


Float8 (FP8)的2D块量化有望提高Float8量化的精度，同时加速推理和训练中的GEMM运算。在这篇博客中，我们展示了使用Triton进行块量化Float8 GEMM的两个主要阶段的进展。

对于从高精度(BFloat16)到Float8的A和B张量的输入量化，我们展示了GridQuant，它利用mini-grid stride loop风格的处理方式，相比当前的2D块量化kernel实现了近2倍的加速(99.31%)。

对于Float8 GEMM，我们展示了Triton的3个新发展 - Warp Specialization、TMA和persistent kernel，有效地创建了一个协作式kernel(作为Ping-Pong调度的替代方案 [PyTorch 博客 CUTLASS Ping-Pong GEMM Kernel 简介](https://mp.weixin.qq.com/s/QWS9YEjsbM7hzy5tJm--1g))。因此，我们比去年最好的SplitK kernel实现了约1.2倍的加速。

![图1：在不同大小下，2D量化相对于当前基准的加速比较。(越低越好)](https://files.mdnice.com/user/59/f886c16e-5aec-45b2-866a-0e59d378a977.jpg)

## 为什么选择FP8的2D块量化？

一般来说，当我们从张量级缩放，到行级缩放，再到2D块级缩放，最后到列级缩放时，fp8量化的精度会逐步提高。这是因为给定token的特征存储在每一列中，因此该张量中的每一列都有更相似的缩放。

为了最小化给定数值集合中的异常值数量，我们希望找到共性，使得数字以相似的方式进行缩放。对于transformer来说，这意味着基于列的量化可能是最优的...然而，由于数据在内存中是按行连续布局的，列式内存访问效率极低。因此，列式加载需要在内存中进行大跨度访问来提取孤立的值，这与高效内存访问的核心原则相违背。

然而，2D是次优选择，因为它包含了一些列式的特点，同时由于我们可以使用2D向量化这些加载，所以内存效率更高。因此，我们希望找到提高2D块量化速度的方法，这就是我们开发GridQuant kernel的原因。

对于量化过程，我们需要对高精度BF16输入张量(A = 输入激活，B = 权重)进行2D块量化，然后使用量化张量及其2D块缩放值进行Float8矩阵乘法，并返回BF16格式的输出C张量。

## GridQuant如何提高2D块量化效率？

GridQuant kernel相比最初基于标准tile的基准量化实现有几项改进。GridQuant kernel对整个输入张量进行两次完整的遍历，工作方式如下：

### 阶段1 - 确定来自高精度张量的每个256x256子块的最大绝对值。

1 - 我们将BF16张量分成256 x 256的子块。这个量化大小是可配置的，但256x256是默认值，因为它在量化精度和处理效率之间提供了良好的平衡。

2 - 每个256x256子块被细分为8x8模式排列的64个子块，每个子块处理32x32元素块。一个warp(32个线程)处理其分配的32x32块内的所有元素计算。

3 - 我们在共享内存中声明一个32x32的max_vals数组。这将存储2d向量块在整个256x256子块中移动时每个位置i,j的当前最大值。

这是一个重要的改进，因为这意味着我们可以对max vals评分系统进行向量化更新，而不是标量更新，从而实现更高效的更新。

![图2：输入张量的分块布局 - 在张量上创建256x256的网格，在每个256x256块内，进一步细分为32x32子块。为每个256x256块创建32x32 max_vals。](https://files.mdnice.com/user/59/558cd48a-6552-42a6-8330-5884655409a9.png)

4 - 每个warp处理一个32x32块，因为我们使用4个warp，我们确保Triton编译器可以将下一个32x32块的内存加载与当前块的absmax计算流水线化。这确保了warp调度器能够在加载数据的warp和处理数据的warp之间切换，使SM持续忙碌。

5 - 32x32 2D向量块处理以网格步进循环的方式在整个256x256子块中移动，每个warp根据其当前32x32子块更新共享内存32x32 max_vals。因此max_vals[i,j]在处理每个子块时保持最新的最大值。

完成256x256块网格步进循环后，maxvals矩阵然后自身被归约以找到整个256块的绝对单一最大值。

这给出了这个2D 256 x 256块的最终缩放因子值。

### 阶段2 - 使用阶段1中找到的单一最大值缩放因子，将256x256块值量化为Float8。

接下来，我们对整个256x256块进行第二次遍历，使用阶段1中找到的最大值来重新缩放所有数字，将它们转换为float 8格式。

因为我们知道需要进行2次完整的遍历，所以在阶段1部分的加载期间，我们指示triton编译器以更高优先级将这些值保持在缓存中(evict policy = last)。

这意味着在第二次遍历期间，我们可以从L2缓存获得高命中率，这比直接访问HBM提供更快的内存访问。

当所有256 x 256块处理完成后，2D块量化处理完成，我们可以返回新的Float8量化张量及其缩放因子矩阵，这将在GEMM处理的下一阶段使用。这个输入量化对第二个输入张量也重复进行，这意味着我们最终得到A_Float 8、A_scaling_matrix和B_Float8以及B_scaling matrix。

## GridQuant - GEMM Kernel

GridQuant-GEMM kernel接收上述量化的四个输出进行处理。我们的高性能GEMM kernel具有几个新的Triton开发特性，以在LLM推理解码阶段相关的矩阵形状配置中实现SOTA性能。

这些新特性常见于使用CUTLASS 3.x构建的Hopper优化kernel，如FlashAttention-3(https://arxiv.org/abs/2407.08608)和Machete(https://neuralmagic.com/blog/introducing-machete-a-mixed-input-gemm-kernel-optimized-for-nvidia-hopper-gpus/)。在这里，我们讨论这些方法并展示使用Triton实现它们可以获得的性能优势。

## 张量内存加速器(TMA)

NVIDIA Hopper GPU上的TMA单元是一个专用的硬件单元，用于处理AI工作负载中常见的多维张量的加载/存储操作。这有几个重要的好处。

从全局内存和共享内存传输数据可以在不涉及GPU SM上其他资源的情况下进行，释放寄存器和CUDA核心。此外，当在warp专用kernel中使用时，轻量级TMA操作可以分配给生产者warp，允许内存传输和计算高度重叠。

关于TMA在Triton中的使用详情，请参见我们的[前一篇博客](https://mp.weixin.qq.com/s/cZRoRq_gzAdA2iaMpZ08VA)。

## Warp专用化(协作式Persistent Kernel设计)

Warp专用化是一种利用GPU流水线并行性的技术。这个实验性特性通过`tl.async_task` API(https://github.com/facebookexperimental/triton/tree/ws)实现了专用线程的表达，允许用户指定Triton程序中的操作应该如何在warp之间"分割"。协作式Triton kernel执行不同类型的计算和加载，每种操作都在其专用硬件上进行。为每个专用任务提供专用硬件使得对于没有数据依赖的操作能够高效地实现并行性。

![图3. NVIDIA H100 SM中专用硬件单元的逻辑视图](https://files.mdnice.com/user/59/ac0d3206-f4d5-4aa0-a4ea-11a0353084f4.png)

我们的kernel中创建流水线的操作是：

A - 从GMEM加载每块缩放到SMEM (cp.async引擎)

B - 从GMEM加载激活(A)和权重(B)tile到SMEM (TMA)

C - A tile和B tile的矩阵乘法 = C tile (Tensor Core)

D - 用A的每块缩放和B的每块缩放来缩放C tile (CUDA core)

这些步骤可以分配给threadblock中专用warp组执行的"任务"。协作策略有三个warp组。一个负责给计算单元提供数据的生产者warp组和2个执行计算的消费者warp组。两个消费者warp组各自处理同一输出tile的一半。

![图4. Warp专用化Persistent协作式kernel (来源：NVIDIA(https://drive.google.com/file/d/18sthk6IUOKbdtFphpm_jZNXoJenbWR8m/view))](https://files.mdnice.com/user/59/3d07a5ab-7350-4dcc-8183-d2050b2ccdb9.png)

这与我们在之前博客中讨论的ping-pong调度不同，在ping-pong调度中，每个消费者warp组处理不同的输出tile。我们注意到Tensor Core操作与epilogue计算不重叠。在计算的epilogue阶段减少Tensor Core流水线的利用率将减少消费者warp组的寄存器压力，相比ping-pong总是保持Tensor Core忙碌，这允许更大的tile大小。

最后，当网格大小超过H100 GPU上可用计算单元数量(132)时，我们的kernel被设计为persistent。Persistent kernel在GPU上保持活跃较长时间，在其生命周期内计算多个输出tile。我们的kernel利用TMA异步共享到全局内存存储，同时继续处理下一个输出tile，而不是承担调度多个threadblock的成本。

## 微基准测试

![图5：在小批量范围和Llama3 8192 N,K大小下，Gridquant-GEMM与我们最佳性能SplitK kernel的延迟比较(微秒)。(越低越好)](https://files.mdnice.com/user/59/77117538-8772-4921-b7e1-fd2c1566cfbb.png)

Warp专用化Triton kernel在上述小M和方阵形状下实现了SOTA性能，相比SplitK Triton kernel(这是Triton GEMM在这个低算术强度范围内之前最佳性能的策略)实现了近1.2倍的加速。对于未来的工作，我们计划调优我们的kernel在中到大M范围和非方阵上的性能。

## 结论和未来工作

未来工作包括对端到端工作流进行gridquant基准测试。此外，我们计划对非方阵(矩形)矩阵以及中到大M大小进行更广泛的基准测试。最后，我们计划探索Triton中的ping-pong风格warp专用化与当前协作式实现的对比。

