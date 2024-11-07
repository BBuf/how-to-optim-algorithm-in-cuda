> 来源： https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487 。这篇文档介绍了PyTorch中新实现的异步张量并行(Async-TP)技术。该技术通过将通信和计算操作分解并重叠执行，显著提升了大规模语言模型训练的性能。在Llama3系列模型上的测试表明，该技术可以使前向传播速度提升20-29%，端到端训练速度提升约8%。文章详细讨论了实现过程中的技术挑战，包括通信开销和计算效率问题，并提出了相应的解决方案。目前该技术已集成到TorchTitan中，可通过torch.compile或eager模式使用，但仍存在一些限制，如需要NVSwitch支持且仅适用于节点内配置等。这项技术对提升大规模分布式训练效率具有重要意义。

# [分布式训练与TorchTitan] PyTorch中异步张量并行的介绍

## 摘要

- 我们在PyTorch中实现了实验性的异步张量并行支持。
- 我们将其集成到TorchTitan中,并观察到:
    - 在Llama3 7B模型中,前向传播速度提升最高达~29%,端到端速度提升最高达~8%。
    - 在Llama3 70B模型中,前向传播速度提升最高达~20%,端到端速度提升最高达~8%。
- 我们简要讨论了性能挑战以及我们设计的解决方案。

## TorchTitan中的分布式训练

GitHub仓库torchtitan是一个使用原生PyTorch进行大规模LLM训练的概念验证项目。它设计得易于理解、使用和扩展,可用于不同的训练目的,通过模块化组件支持多维度并行。在这一系列主题中,我们介绍了在TorchTitan中启用的最新PyTorch分布式训练特性。

- 主题1: [【翻译】在FSDP2中开启Float8 All-Gather](https://mp.weixin.qq.com/s/44zFNWr5aVtA3zPtegY9dg)
- → 主题2: 在PyTorch中引入异步张量并行

## 张量并行

张量并行(TP)是一种广泛使用的模型并行技术。与数据并行不同,数据并行仅限于在批次维度上分片计算,而TP进一步沿特征维度分布计算,允许多个GPU同时处理相同的样本。这一特性使得TP对大规模LLM训练至关重要,因为它打破了设备数量超出全局批次大小的限制。

![图1: 在2个设备上应用TP的两层FFN示意图](https://files.mdnice.com/user/59/bbb78b1d-f1ca-4566-b500-cf934d8b5c6e.png)

作为简要回顾,图示展示了在2个设备上应用TP的两层FFN。我们首先从行分片的输入[X0, X1],列分片的线性权重[A0, A1],和行分片的线性权重[B0, B1]开始。首先,对[X0, X1]执行all-gather操作以生成未分片的输入X。然后,在每个设备上独立计算X @ A0 @ B0和X @ A1 @ B1,同时保持激活分片。最后,使用reduce-scatter将未分片的输出部分和组合,形成最终的分片输出。

这种方法通过尽可能长时间地保持激活分片,有效地最小化了通信量。然而,通信仍然存在效率挑战,因为它暴露了。异步张量并行是一种优化设计,旨在解决这一问题。

## 异步张量并行

据我们所知,异步张量并行(async-TP)的概念最早是在论文《Breaking the Computation and Communication Abstraction Barrier in Distributed Machine Learning Workloads》(https://arxiv.org/abs/2105.05720)中提出的,尽管也有一些平行的研究工作,包括Wang等人2022年的工作(https://dl.acm.org/doi/abs/10.1145/3567955.3567959)和Chang等人2024年的工作(https://arxiv.org/abs/2406.06858)。其关键洞见在于,通过分解相互依赖的通信和计算算子,我们可以创造出原本无法实现的重叠机会。

![图2: 将异步TP应用于all-gather和矩阵乘法的示意图](https://files.mdnice.com/user/59/f41edfef-0bfc-45e0-b902-b2732b17dff1.png)

> 上半部分 (Original传统方式): 两个分区(Partition 0和1)按顺序执行, 先进行AllGather操作收集数据, 然后执行Einsum矩阵计算，这种方式计算和通信是串行的,存在等待时间。下半部分 (Overlapped异步方式): 通信被分解为异步Send和Recv操作，计算也被分解为更小的Einsum操作。两个分区可以同时进行: Partition 0发送A0到Partition 1,同时计算Einsum(A0, B0)，Partition 1发送A1到Partition 0,同时计算Einsum(A1, B1)，通过Dynamic Update更新计算结果。这种方法通信和计算可以重叠执行，减少了整体等待时间，如图右侧箭头所示,相比传统方式节省了时间。

Wang等人的图示展示了如何将此技术应用于all-gather后跟一个matmul。all-gather被分解为send和recv操作,而matmul被分为子matmuls。通过这种分解,可以同时计算一个子matmul,同时传输下一个子matmul所需的数据,从而有效地隐藏通信延迟。

## 性能挑战

尽管异步张量并行的概念在理论上很简单直观,但要在CUDA中实现高性能的实现却面临着一些挑战。在本节中,我们将探讨这些挑战,并讨论我们采用的解决方案。

**致谢**: 这些挑战中的许多最初是由Luca Wehrstedt(https://discuss.pytorch.org/u/lcw/summary)探索的。PyTorch中的异步TP实现从他在xformers中的异步TP工作中获得了重要启发。

### 通信开销

在分解通信时,使用NCCL的send/recv操作可能很有诱惑力,因为它们易于使用。然而,NCCL的send/recv操作具有一些特性,使其不太适合异步张量并行:

- **重叠计算和通信之间的竞争** - 虽然人们普遍认为计算和通信是可以独立使用的两种资源,但实际上它们的独立性是有细微差别的,确实会发生竞争。在节点内设置(TP最常见的情况)中,NCCL的send/recv kernel 会利用SM通过NVLink传输数据,这减少了可用于重叠矩阵乘法 kernel 的SM数量,从而降低了速度。有趣的是,观察到的速度下降可能超过通信 kernel 消耗的资源百分比。由于cuBLAS试图选择以完整waves执行的 kernel ,通信 kernel 占用的SM可能会打破平衡,导致矩阵乘法 kernel 需要执行额外的wave。

- **双向同步** - NCCL的send/recv kernel 执行双向同步,这意味着发送方和接收方都会被阻塞直到操作完成。这种方法对于算子内并行中的数据传输并不总是最优的。根据具体情况,可能更适合对多个数据传输执行单次同步,或者在向远程节点推送数据和从远程节点拉取数据之间进行选择。

幸运的是,我们可以通过利用CUDA的P2P机制来避免前面提到的缺点。该机制允许设备通过将其映射到虚拟内存地址空间来访问对等设备上分配的内存。这种机制使得内存操作(加载/存储/原子等)可以通过NVLink执行(目前,PyTorch中的async-TP实现需要所有设备对之间都有NVLink连接(例如通过NVSwitch)才能实现加速。这是我们计划在未来解决的限制)。此外,当通过cudaMemcpyAsync在对等设备之间传输连续数据时,该操作由拷贝引擎(拷贝引擎是GPU上的专用硬件单元,用于管理不同内存位置之间的数据移动,独立于GPU的计算核心(SM)运行)处理,不需要任何SM,从而避免了前面讨论的竞争问题(通过拷贝引擎的数据传输仍然共享相同的内存带宽。然而,这不太可能造成显著的竞争,因为(1)传输速率受限于NVLink带宽,低到足以避免内存带宽竞争,(2)重叠的矩阵乘法是计算密集型的)。

为了在未来利用这种机制实现async-TP和类似用例,我们开发了一个名为SymmetricMemory的实验性抽象(https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/SymmetricMemory.hpp)。从概念上讲,它表示一个在设备组之间对称分配的缓冲区,通过虚拟内存/多播地址为每个GPU提供对其对等设备上所有相应缓冲区的访问。使用async-TP不需要直接与SymmetricMemory交互,但用户可以利用它来创建类似于async-TP的自定义细粒度、节点内/算子内优化。

### 由于工作分解导致的Wave量化效率放大问题

分块矩阵乘法 kernel 按SM数量以 wave 的形式执行。如果最后一个 wave 只包含少量块,完成时间几乎与完整的 wave 一样长,这就导致了所谓的量化效率问题。将一个矩阵乘法分解会导致每个 kernel 的块数减少,分解后的矩阵乘法的组合量化效率损失可能会超过原始矩阵乘法。

![图3: 分解矩阵乘法会导致量化效率问题被放大](https://files.mdnice.com/user/59/008e7749-7fa6-431f-98b0-355f7b9da7fc.png)

为了说明这个问题,让我们来看一个all-gather → matmul的例子。在这个例子中,A被分片到4个设备上。在不使用async-TP的情况下,首先从4个设备上收集A,然后在所有设备上计算A @ B。使用async-TP时,A @ B被分解为A0 @ B、A1 @ B、A2 @ B和A3 @ B。一个原生的async-TP实现会在一个流中顺序执行这些子矩阵乘法,同时在另一个流中预取下一个子矩阵乘法所需的数据。这种方法有效地隐藏了通信延迟。然而,由于矩阵乘法被分解成更小的部分,部分wave的数量增加,导致整体矩阵乘法执行时间变长。

![图4: 交替流实现允许部分wave与下一个子矩阵乘法重叠](https://files.mdnice.com/user/59/d493d63d-436e-41a7-b972-7265212439f6.png)

为了解决放大的量化效率问题,我们采用了交替流的方法。我们不使用专门的计算和通信流,而是使用两个交替角色的对称流。这种方法不仅允许计算和通信重叠,还能让当前子矩阵乘法的部分wave与下一个子矩阵乘法重叠,从而缓解了分解导致的额外量化效率问题。

![图5: 部分wave与下一个子矩阵乘法重叠的性能分析跟踪](https://files.mdnice.com/user/59/ab411db0-48aa-45b7-a364-f9df65d98637.png)

![图6: 基线和async-TP的性能分析跟踪对比](https://files.mdnice.com/user/59/a62d48c9-0734-4f2d-b84e-8d26695d708b.png)


## 端到端性能评估

我们使用TorchTitan对Llama3 8B和70B进行了端到端性能评估。在Llama3 8B上,我们观察到前向传播速度提升了约29%,端到端速度提升了约8%;在Llama3 70B上,前向传播速度提升了约20%,端到端速度提升了约8%。

基准测试配置:

- 基准测试使用了64个H100 GPU(用于基准测试的H100 GPU是非标准的。它们使用HBM2e并且TDP受限。实际峰值TFLOP应该在SXM和NVL之间,我们不知道确切的值。因此报告的MFU低于实际MFU,因为我们直接使用了SXM的峰值TFLOP),每个主机配备8个GPU和NVSwitch。
- 基线和async-TP配置都启用了torch.compile。
- 模型使用bf16精度进行训练。
- 我们对Llama3 8B应用了选择性激活检查点,对Llama3 70B应用了完整激活检查点。

![图7: 使用async-TP的端到端加速效果](https://files.mdnice.com/user/59/42148fd5-05b9-45d5-b4a4-e4991a8b647b.png)

![图8: 使用async-TP的前向传播加速效果](https://files.mdnice.com/user/59/7a8f2577-d95c-44d5-9cd9-be43c48fe153.png)

![图9: 端到端基准测试数据](https://files.mdnice.com/user/59/50cdcdf6-7da9-4e5f-8dfd-15e145e44625.png)

我们还在Llama 3.1 405B上进行了async-TP的基准测试。你可以在这里找到详细信息(https://github.com/pytorch/torchtitan/blob/main/docs/performance.md)

## 在TorchTitan中使用Async-TP

Async-TP支持已经集成到TorchTitan中。要启用它,只需在使用张量并行训练时提供`--experimental.enable_async_tensor_parallel`选项即可。

## 在PyTorch中使用Async-TP

Async-TP支持在最新的PyTorch nightly builds中可用。你可以通过torch.compile或直接在eager模式下使用它。

### 使用torch.compile的Async-TP:

![图10: torch.compile自动检测TP模式并将其重写为async-TP算子](https://files.mdnice.com/user/59/671b5881-dab2-45ef-a4c5-ffeca5555f54.png)

torch.compile 目前是我们推荐的应用 async-TP 的方法:

- 它能自动检测模型中的 TP 模式并将其重写为 async-TP 算子,使模型能保持其原始结构。
- 优化的 async-TP 实现要求输入具有特定的布局;否则会发生额外的复制。torch.compile 会自动确保上游算子尽可能以所需布局生成输出。
- torch.compile 还能检测 all-gather 可以与多个矩阵乘法重叠的情况,从而更好地隐藏通信延迟。

虽然这些在 eager 模式下也可以手动实现,但可能会导致模型代码和优化逻辑之间的耦合更紧密。

![图11: torch.compile可以自动将async-TP应用于all-gather操作以及后续使用all-gather结果的多个矩阵乘法(例如QKV投影)](https://files.mdnice.com/user/59/d8ee33ca-f30a-4ada-8516-75e9311b7062.png)

对于编写TP逻辑,我们推荐使用PyTorch Tensor Parallel APIs。你可以在这里找到教程(https://pytorch.org/tutorials/intermediate/TP_tutorial.html)以及在TorchTitan中的示例(https://github.com/pytorch/torchtitan/blob/1923ce4/torchtitan/parallelisms/parallelize_llama.py#L158-L183)。此外,torch.compile可以将async-TP应用于使用功能性集合操作以及`torch.mm`、`torch.matmul`或`torch._scaled_mm`手动编写的TP逻辑。你可以在这里找到一个示例(https://github.com/pytorch/pytorch/blob/16b8146/test/distributed/tensor/parallel/test_micro_pipeline_tp.py#L206-L208)。

```python
from torch.distributed._symmetric_memory import enable_symm_mem_for_group

# Enable symmetric memory for the TP process group
enable_symm_mem_for_group(tp_group.group_name)

# Tell torch.compile to enable async-TP
torch._inductor.config._micro_pipeline_tp = True

# Apply torch.compile to the model
model = torch.compile(model)

# Or apply torch.compile to only the model region that contains TP logic
model.tp_submodule = torch.compile(model.tp_submodule)
```

### 在eager模式下使用Async-TP:

也可以通过直接调用async-TP算子在eager模式下应用async-TP:

```python
from torch.distributed._symmetric_memory import enable_symm_mem_for_group

# Enable symmetric memory for the TP process group
enable_symm_mem_for_group(tp_group.group_name)

# Invoke the async-TP operators directly
# all-gather -> matmul
ag_output, mm_outputs = torch.ops.symm_mem.fused_all_gather_matmul(
    x,
    [wq, wk, wv],
    gather_dim=1,
    group_name=tp_group.group_name,
)

# matmul -> reduce-scatter
output = torch.ops.symm_mem.fused_matmul_reduce_scatter(
    x,
    w,
    "avg",
    scatter_dim=0,
    group_name=tp_group.group_name,
)
```

## 局限性和未来工作

PyTorch中async-TP支持的当前局限性:

- **针对大型矩阵乘法问题优化**: 目前,PyTorch中的async-TP在大型矩阵乘法操作上表现最佳,特别是那些分解后不需要改变分块大小的操作。我们正在探索更细粒度的流水线解决方案,以提高在推理工作负载等较小问题规模上的性能。
- **需要NVSwitch**: 目前,PyTorch中的async-TP依赖NVSwitch来获得最佳性能。我们正在考虑根据社区反馈和需求扩展对NVLink环形拓扑的支持。
- **仅限于节点内配置**: PyTorch中的async-TP目前仅适用于节点内设置。我们将来可能会探索将此支持扩展到跨节点环境。

## 注释

- PyTorch Distributed选择使用"async-TP"这个术语来描述这种技术,但它可能不会被普遍地这样称呼。
- 目前,PyTorch中的async-TP实现需要所有设备对之间都有NVLink连接(例如通过NVSwitch)才能实现加速。这是我们计划在未来解决的一个限制。
- Copy引擎是GPU上的专用硬件单元,用于管理不同内存位置之间的数据移动,并独立于GPU的计算核心(SMs)运行。
- 通过copy引擎的数据传输仍然共享相同的内存带宽。但是,这不太可能造成显著的竞争,因为(1)传输速率受NVLink带宽限制,低到足以避免内存带宽竞争,(2)重叠的矩阵乘法是计算密集型的。
- 我们假设矩阵乘法问题规模足够大,因此分解后分块形状不会改变,分解开销的主要来源是量化效率。
- 用于基准测试的H100 GPU是非标准的。它们使用HBM2e并限制在较低的TDP。实际峰值TFLOPs应该在SXM和NVL之间,我们不知道确切的值。因此报告的MFU低于实际MFU,因为我们直接使用SXM的峰值TFLOPs。

