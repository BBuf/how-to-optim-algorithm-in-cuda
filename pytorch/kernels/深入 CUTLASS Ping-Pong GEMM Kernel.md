> 博客来源：https://pytorch.org/blog/cutlass-ping-pong-gemm-kernel/ 这里做了个翻译。这篇PyTorch的blog简要介绍了 CUTLASS 中的 Ping-Pong GEMM kernel 设计，它是专门为 Hopper GPU 架构优化的高性能矩阵乘法实现。通过采用生产者-消费者模式的异步流水线设计，结合 TMA 硬件加速和专门化的 warp 组，实现了对 Tensor Core 的高效利用。文章通过benchmark表明，这种设计相比 cuBLAS 和 Triton 等其他实现具有明显优势，充分展现了在新一代 GPU 架构上如何通过深度异步化来最大化计算吞吐量。同时也把这部分cutlass代码单独抽出来写了一个PyTorch可以用的扩展，见 https://github.com/pytorch-labs/applied-ai/tree/main/kernels/cuda/cutlass_gemm 。

# 深入解析 CUTLASS Ping-Pong GEMM kernel 

![图1. FP8 GEMM 吞吐量对比：CUTLASS vs Triton](https://files.mdnice.com/user/59/6d460462-07c4-4314-9e1e-e45afbc96516.png)

## 摘要

在这篇文章中，我们将概述 CUTLASS Ping-Pong GEMM kernel ，并提供相关的 FP8 推理kernel 基准测试。

Ping-Pong 是 Hopper GPU 架构上可用的最快矩阵乘法（GEMM）kernel 架构之一。Ping-Pong 属于 Warp Group Specialized Persistent Kernels 家族，该家族包括 Cooperative 和 Ping-Pong 两种变体。相对于之前的 GPU，Hopper 的强大Tensor Core计算能力需要深度异步软件流水线来实现峰值性能。
 
Ping-Pong 和 Cooperative kernel 很好地展示了这一范式，其关键设计模式是持久化kernel （用于摊销启动和prologue开销），以及"全面异步"的专门化 warp 组（包含两个消费者和一个生产者），以创建高度重叠的处理流水线，能够持续为Tensor Core提供数据。

当 H100 (Hopper) GPU 发布时，Nvidia 将其称为第一款真正的异步 GPU。这一说法突出了 H100 特定kernel 架构也需要异步化，以充分最大化计算/GEMM 吞吐量。

CUTLASS 3.x 中引入的 pingpong GEMM 通过将kernel 的所有方面都移至"完全异步"处理范式来体现这一点。在这篇博客中，我们将展示 ping-pong kernel 设计的核心特性，并展示其在推理工作负载上与 cublas 和 triton split-k kernel 相比的性能。

## Ping-Pong kernel 设计

Ping-Pong（或者从技术上说是'sm90_gemm_tma_warpspecialized_pingpong'）采用异步流水线运行，利用 warp 专门化。与传统的同质kernel 不同，"warp 组"承担专门的角色。需要注意的是，一个 warp 组由 4 个 warp（每个 warp 32 个线程）组成，总共 128 个线程。

在早期架构中，通常通过在每个 SM 上运行多个线程块来隐藏延迟。然而，在 Hopper 上，Tensor Core吞吐量如此之高，以至于需要转向更深的流水线。这些更深的流水线会阻碍在每个 SM 上运行多个线程块。因此，持久化线程块现在会在多个Tile 和多个 warp 组之间发出集体main loops 。线程块集群根据总 SM 数量进行分配。

对于 Ping-Pong 来说，每个 warp 组都承担数据生产者或数据消费者的专门角色。

生产者 warp 组专注于通过 TMA 产生数据移动来填充共享内存缓冲区。另外两个 warp 组是专门的消费者，它们处理使用Tensor Core的数学（MMA）部分，然后进行任何后续工作并将结果写回全局内存（epilogue）。

生产者 warp 组与 TMA（张量内存加速器）一起工作，并被刻意保持尽可能轻量级。事实上，在 Ping-Pong 中，它们故意减少寄存器资源以提高占用率。生产者将其最大寄存器数减少 40 个，而消费者将其最大寄存器数增加 232 个，这种效果我们可以在 CUTLASS 源代码和相应的 SASS 中看到：

![](https://files.mdnice.com/user/59/781332c5-73e1-46cf-88d2-90910f24abf1.png)

Ping-Pong 的独特之处在于，每个消费者在不同的 C 输出Tile 上工作。（作为参考，cooperative kernel 在很大程度上等同于 Ping-Pong，但两个消费者组在同一个 C 输出Tile 上工作）。此外，两个消费者 warp 组然后在main loops  MMA 和 epilogue 之间分配它们的工作。

这在下图中显示：

![图2：Ping-Pong kernel 流水线概览。时间从左向右移动。](https://files.mdnice.com/user/59/fe3d6ff8-7bdf-4ab3-8766-94127421de59.jpg)

通过拥有两个消费者，意味着一个可以使用Tensor Core进行 MMA，而另一个执行 epilogue，然后反之亦然。这最大化了每个 SM 上Tensor Core的"连续使用"，这是实现最大吞吐量的关键原因之一。Tensor Core可以持续获得数据以实现（接近）最大计算能力。（参见上图 Fig 2 的底部部分）。

与生产者线程仅专注于数据移动类似，MMA 线程仅发出 MMA 指令以实现峰值发出率。MMA 线程必须发出多个 MMA 指令，并使这些指令在 TMA 等待屏障上保持运行。

下面展示了kernel 代码的一个摘录，以巩固专门化方面：

```c++
// Two types of warp group 'roles' 
enum class WarpGroupRole {
      Producer = 0,
      Consumer0 = 1,
      Consumer1 = 2
    };

//warp group role assignment
auto warp_group_role = WarpGroupRole(canonical_warp_group_idx());
```

## 使用生产者和张量内存加速器的数据移动

生产者 warp 专注于数据移动 - 具体来说，它们被保持尽可能轻量级，实际上会将一些寄存器空间让给消费者 warp（只保留 40 个寄存器，而消费者将获得 232 个）。它们的主要任务是在共享内存缓冲区被信号标记为空时，立即发出 TMA（张量内存加速器）命令，将数据从全局内存移动到共享内存。

关于 TMA（张量内存加速器）的更多说明，TMA 是随 H100 引入的一个硬件组件，它异步处理从 HBM（全局内存）到共享内存的内存传输。通过拥有专门的硬件单元进行内存移动，工作线程可以从事其他工作，而不是计算和管理数据移动。TMA 不仅处理数据本身的移动，还计算所需的目标内存地址，可以对数据应用任何转换（归约等），并可以处理布局转换，以"交错"模式将数据传递到共享内存，使其可以在没有任何Bank 冲突的情况下使用。最后，如果需要，它还可以将相同的数据多播到属于同一线程集群的其他 SM。一旦数据传递完成，TMA 将向相关的消费者发出信号，表明数据已准备就绪。

## CUTLASS 异步流水线类

生产者和消费者之间的这种信号传递通过新的异步流水线类进行协调，CUTLASS 对其描述如下：

"实现持久化 GEMM 算法需要管理数十种不同类型的异步执行操作，这些操作使用组织为循环列表的多个屏障进行同步。

这种复杂性对于人类程序员来说太难手动管理。

因此，我们开发了 [CUTLASS Pipeline Async Class](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/pipeline/sm90_pipeline.hpp)..."

## Ping-Pong 异步流水线中的屏障和同步

生产者必须通过'producer_acquire'来"获取"给定的共享内存缓冲区。在开始时，流水线是空的，这意味着生产者线程可以立即获取屏障并开始移动数据。

```c++
PipelineState mainloop_pipe_producer_state = cutlass::make_producer_start_state<MainloopPipeline>();
```

一旦数据移动完成，生产者发出'producer_commit'方法来向消费者线程发出数据准备就绪的信号。
然而，对于 Ping-Pong 来说，这实际上是一个空操作指令，因为基于 TMA 的生产者屏障在 TMA 完成写入时会自动更新。

consumer_wait - 等待来自生产者线程的数据（阻塞）。

consumer_release - 向等待的生产者线程发出信号，表明它们已完成消费给定共享内存缓冲区中的数据。换句话说，允许生产者开始用新数据重新填充这个缓冲区。

从那里开始，同步将开始认真进行，生产者将通过阻塞的 producer acquire 等待，直到它们可以获取锁，此时它们的数据移动工作将重复。这将持续到工作完成为止。

提供一个伪代码概述：

```shell
//producer
While (work_tile_info.is_valid_tile) {

	collective_mainloop.dma() // fetch data with TMA
	scheduler.advance_to_next_work()
	Work_tile_info = scheduler.get_current_work()

}

// Consumer 1, Consumer 2
While (work_tile_info.is_valid_tile()) {

	collective_mainloop.mma()
	scheduler.advance_to_next_work()
	Work_tile_info = scheduler.get_current_work()

}
```

以及一个将所有内容与底层硬件结合在一起的鸟瞰图：

![图3：Ping-Pong 完整异步流水线概览](https://files.mdnice.com/user/59/3473c860-1382-4a69-aee6-ad3785590d76.png)

补充一下图的细节：

1. 主要组件:
- SM 内包含 Consumer MMA Warps 和 Producer DMA Warps
- Tensor Core: 执行矩阵乘法运算
- SMEM (共享内存): 带有异步屏障机制
- RMEM (寄存器内存)
- TMA (张量内存加速器)
- GMEM (全局内存)
2. 数据流动路径:
- Producer DMA Warps 通过 `cp_async_bulk` 指令与 TMA 交互
- TMA 负责在 GMEM 和 SMEM 之间传输数据
- Consumer MMA Warps 通过 `wgmma.mma_async` 指令从 SMEM 读取数据到 Tensor Core
- Tensor Core 计算结果写入 RMEM
- 数据可以多播到其他线程块
3. 同步机制:
- Producer 和 Consumer 之间通过 Acquire/Commit 和 Wait/Release 操作进行同步
- SMEM 中的异步屏障用于协调数据访问
- TMA 处理异步数据传输
4. 关键特点:
- 整个流程是高度异步的
- 使用专门化的 warp 组实现生产者-消费者模式
- 通过 TMA 实现高效的内存传输
- 支持跨线程块的数据多播


## Ping-Pong 计算循环的逐步分解

最后，对 Ping-Pong 处理循环的更详细的逻辑分解：

A - 生产者（DMA）warp 组获取共享内存缓冲区的锁。

B - 这允许它通过单个线程向 tma 芯片发起 tma `cp_async.bulk` 请求。

C - TMA 计算所需的实际共享内存寻址，并将数据移动到共享内存。作为这个过程的一部分，执行交错操作以便在共享内存中布局数据，以实现最快（无Bank 冲突）的访问。

C1 - 可能的情况下，数据也可以多播到其他 SM，和/或它可能需要等待来自其他 tma 多播的数据以完成加载。（线程块集群现在在多个 SM 之间共享共享内存！）

D - 此时，屏障被更新以向共享内存发出数据到达的信号。

E - 相关的消费者 warp 组现在开始工作，发出多个 `wgmma.mma_async` 命令，这些命令然后从共享内存读取数据到Tensor Core，作为其 `wgmma.mma_async` 矩阵乘法操作的一部分。

F - 当Tile 完成时，MMA 累加器值被写入寄存器内存。

G - 消费者 warp 组释放共享内存上的屏障。

H - 生产者 warp 组开始工作，发出下一个 tma 指令以重新填充现在空闲的共享内存缓冲区。

I - 消费者 warp 组同时对累加器应用任何 epilogue 操作，然后将数据从寄存器移动到不同的共享内存缓冲区。

J - 消费者 warp 发出 `cp_async` 命令，将数据从共享内存移动到全局内存。

这个循环重复进行，直到工作完成。希望这能让你对支持 Ping-Pong 令人印象深刻性能的核心概念有一个工作性的理解。

## 微基准测试

为了展示 Ping-Pong 的一些性能，下面是一些与我们设计快速推理kernel 相关的比较图表。

首先是目前三个最快kernel 的一般基准测试（越低越好）：

![图4：FP8 GEMM 基准测试时间，数值越低越好（越快）](https://files.mdnice.com/user/59/e0f3a181-9142-41ca-b617-5920c09a8223.png)

将其转换为 Ping-Pong 与 cuBLAS 和 Triton 的相对加速图：

![图5：Ping-Pong 相对于两个最接近kernel 的速度提升](https://files.mdnice.com/user/59/1a93a37d-b35f-4564-9f38-12ebcd4d807f.jpg)

Ping-Pong kernel 的完整源代码在这里（619 行深度模板化的 CUTLASS 代码，或者用著名的乌龟模因来说就是 - "全都是模板...一直都是！"）：

- https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp

此外，我们已经将 PingPong 实现为 CPP 扩展，使其易于与 PyTorch 集成（同时附带一个简单的测试脚本展示其用法）：

- https://github.com/pytorch-labs/applied-ai/tree/main/kernels/cuda/cutlass_gemm

最后，为了继续学习，Nvidia 有两个深入探讨 CUTLASS kernel 设计的 GTC 视频：

- Developing Optimal CUDA Kernels on Hopper Tensor Cores | GTC Digital Spring 2023 | NVIDIA On-Demand(https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51413/)
- CUTLASS: A Performant, Flexible, and Portable Way to Target Hopper Tensor Cores | GTC 24 2024 | NVIDIA On-Demand(https://www.nvidia.com/en-us/on-demand/session/gtc24-s61198/)

## 未来工作

数据移动通常是任何kernel 实现最高性能的最大障碍，因此对 Hopper 上的 TMA（张量内存加速器）有最优策略理解至关重要。我们之前发布了关于 Triton 中 TMA 使用的工作(https://mp.weixin.qq.com/s/cZRoRq_gzAdA2iaMpZ08VA)。一旦在 Triton 中启用了 warp 专门化等功能，我们计划再次深入研究 Triton kernel （如 FP8 GEMM 和 FlashAttention）如何利用 Ping-Pong 等kernel 设计在 Hopper GPU 上加速。

