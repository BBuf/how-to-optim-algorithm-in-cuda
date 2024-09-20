# 深入探讨 Hopper TMA 单元在 FP8 GEMM 运算中的应用

- PyTorch博客资料：https://pytorch.org/blog/hopper-tma-unit/
- PyTorch实现和使用Demo：https://github.com/pytorch-labs/applied-ai/blob/main/kernels/triton/inference/fp8/tma_gemm.py 在本文最后也添加了这个代码的解释

## 摘要

Hopper（H100）GPU架构被称为"第一款真正的异步GPU"，它包含了一个新的、完全异步的硬件复制引擎，用于在全局内存和共享内存之间进行大规模数据移动，这个引擎被称为张量内存加速器（TMA）。虽然CUTLASS通过其异步流水线范式内置了对TMA的支持(https://github.com/NVIDIA/cutlass/blob/56b46e2d13875b46b8f6a03f9f5ac91e2bfdc01a/include/cute/arch/copy_sm90_tma.hpp)，但Triton则通过一个实验性API (https://github.com/triton-lang/triton/blob/538556a66ee49630e1cb0b239f93e63b968b2478/python/triton/tools/experimental_descriptor.py#L25) 来提供TMA支持。

在这篇文章中，我们将深入探讨TMA的工作原理细节，以帮助开发者理解这个新的异步复制引擎。我们还将展示利用TMA对H100 kernel的重要性，通过在Triton中构建一个支持TMA的FP8 GEMM kernel，该内核在小到中等问题规模上相比cuBLAS FP16可获得1.4-2.2倍的性能提升。最后，我们将展示Triton和CUTLASS之间的关键实现差异，这些差异可能解释了在Triton中使用TMA时报告的性能回归。我们将我们的实现开源，以便于复现和审查，代码地址为：https://github.com/pytorch-labs/applied-ai/tree/main/kernels

![图1. 各种Triton和cuBLAS FP8和FP16 kernel的吞吐量（以TFLOPs为单位），条件为M=M，N=4096，K=4096。红线代表Triton TMA，展示了利用TMA的优势。](https://files.mdnice.com/user/59/d14103c6-7eec-4ea4-b0df-77d98a6c03ba.png)

## TMA 背景

TMA是H100硬件的一个新增功能，它允许应用程序异步且双向地在GPU全局内存和共享内存之间传输1D-5D张量。此外，TMA不仅可以将相同的数据传输到调用SM的共享内存，还可以传输到同一线程块Cluster中其他SM的共享内存。这种功能被称为"multicast"。

TMA非常轻量级，只需一个单独的线程就可以启动TMA传输。通过直接将数据从GMEM（全局内存）移动到SMEM（共享内存），这避免了早期GPU中使用寄存器在不同内存空间之间移动数据的要求。

![图2. A100风格的数据移动与使用TMA的H100对比。TMA硬件消除了大量线程和寄存器参与批量数据传输的需求。（图片来源：NVIDIA）](https://files.mdnice.com/user/59/cfe5fe55-9cf8-4118-bdba-3d441d6ac566.png)

单个线程可以发出大规模数据移动指令，使得给定线程块的大部分线程能在数据传输过程中继续执行其他指令。结合异步流水线技术，这使得内存传输可以轻易地被隐藏，确保大多数线程块簇能专注于计算任务。

这种轻量级的数据移动调用方式使得创建专门化的 warp-group kernel 成为可能，其中不同的 warp-group 可以承担不同的角色，即生产者和消费者。生产者选出一个领导线程来发起TMA请求，这些请求随后通过到达 barrier 与消费者（MMA）warp-group 异步协调。然后消费者使用 warp-group MMA 处理数据，并在完成从共享内存 buffer 读取数据后向生产者发出信号，周而复始。

此外，在线程块 clusters 内，生产者可以降低其最大寄存器需求，因为它们只负责发出TMA调用，实际上是将额外的寄存器转移给 MMA 消费者，这有助于缓解消费者的寄存器压力。

另外，TMA处理共享内存目标地址的计算，即请求的数据应该放置的位置。这就是为什么调用线程（生产者）可以如此轻量级的原因。

为确保最大的读取访问速度，TMA可以基于 swizzling 指令来布局到达的数据，确保消费者能以最快的速度读取到达的数据，因为交织模式有助于避免共享内存的 Bank 冲突。

最后，对于从共享内存到全局内存的外出TMA指令，TMA还可以包括归约操作（加/最小/最大）和位运算（与/或）操作。

## TMA 在 Triton 中的使用

**Pre-Hopper Load**:

```python
offs_m = pid_m*block_m + tl.arange(0, block_m)
offs_n = pid_n*block_n + tl.arange(0, block_n)
offs_k = tl.arange(0, block_k)

a_ptrs = a_ptr + (offs_am[:, None]*stride_am + offs_k[None, :]*stride_ak)
b_ptrs = b_ptr + (offs_k[:, None]*stride_bk + offs_bn[None, :]*stride_bn)

a = tl.load(a_ptrs)
b = tl.load(b_ptrs)
```

>  Triton中传统风格的全局内存到共享内存的批量加载

在上面展示的Triton例子中，我们看到了一个Hopper架构之前的加载方式。每个线程块通过计算全局偏移量(a_ptrs, b_ptrs)来加载张量a和b的数据，这些偏移量是基于它们相关的program_id (pid_m, pid_n, k)计算得出的，然后发出请求将内存块移入a和b的共享内存中。

现在让我们来看看如何在Triton中使用TMA执行加载操作。

TMA指令需要一个特殊的数据结构，称为张量映射（tensor map），这与上面直接传递全局内存指针的方式不同。为了构建张量映射，我们首先在CPU上创建一个TMA描述符。该描述符通过使用cuTensorMapEncode API (https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY) 来处理张量映射的创建。张量映射包含了诸如张量在全局和共享内存中的布局等元数据，并作为存储在全局内存中的多维张量结构的压缩表示。

![图4. 通过copy描述符生成TMA地址（图片来源：Nvidia）](https://files.mdnice.com/user/59/8f80d5ed-b3ed-4d11-b9c3-346f99fbb133.png)

TMA描述符包含张量的关键属性：
- 基指针
- 形状和块大小
- 数据类型

TMA描述符在kernel执行前在主机上创建，然后通过将描述符传递给torch张量来移动到设备上。因此，在Triton中，GEMM kernel接收一个指向张量映射的全局指针。

## Triton Host Code

```python
   desc_a = np.empty(TMA_SIZE, dtype=np.int8)
   desc_b = np.empty(TMA_SIZE, dtype=np.int8)
   desc_c = np.empty(TMA_SIZE, dtype=np.int8)

   triton.runtime.driver.active.utils.fill_2d_tma_descriptor(a.data_ptr(), m, k, block_m, block_k, a.element_size(), desc_a)

   triton.runtime.driver.active.utils.fill_2d_tma_descriptor(b.data_ptr(), n, k, block_n, block_k, b.element_size(), desc_b)

   triton.runtime.driver.active.utils.fill_2d_tma_descriptor(c.data_ptr(), m, n, block_m, block_n, c.element_size(), desc_c)
  
   desc_a = torch.tensor(desc_a, device='cuda')
   desc_b = torch.tensor(desc_b, device='cuda')
   desc_c = torch.tensor(desc_c, device='cuda')
```

这是在kernel调用函数中用于设置描述符的代码。

## Triton Device Code

**偏移量/指针算术:**

```python
   offs_am = pid_m * block_m
   offs_bn = pid_n * block_n
   offs_k = 0
```

**Load:**

```python
  a = tl._experimental_descriptor_load(a_desc_ptr, [offs_am, offs_k], [block_m, block_k], tl.float8e4nv)
  b = tl._experimental_descriptor_load(b_desc_ptr, [offs_bn, offs_k], [block_n, block_k], tl.float8e4nv)
```

**Store:**

```python
 tl._experimental_descriptor_store(c_desc_ptr, accumulator, [offs_am, offs_bn])
```

我们不再需要在 kernel 中为加载和存储函数计算指针数组。相反，我们只需传递一个描述符指针、偏移量、块大小和输入数据类型。这简化了地址计算并减少了寄存器压力，因为我们不再需要在软件中进行复杂的指针算术，也不需要专门分配CUDA Core来进行地址计算。

## TMA 性能分析

下面，我们讨论Hopper架构上不同加载机制的PTX指令。

**用于Load Tile 的 PTX（cp.async）- H100无TMA**

```python
# 这两行计算了共享内存中的目标地址。%r100可能是共享内存的基地址,%r8和%r9是偏移量。
add.s32 	%r27, %r100, %r8;
add.s32 	%r29, %r100, %r9;
# 这行根据条件%p18选择%r102或0,结果存入%r30。这可能用于控制是否执行复制操作。
selp.b32 	%r30, %r102, 0, %p18;

# 这两行是关键的异步复制指令。它们从全局内存(%rd20和%rd21)复制数据到共享内存(%r27和%r29)。0x10表示复制16字节。%p1是一个谓词,控制是否执行这些指令。
@%p1 cp.async.cg.shared.global [ %r27 + 0 ], [ %rd20 + 0 ], 0x10, %r30;
@%p1 cp.async.cg.shared.global [ %r29 + 0 ], [ %rd21 + 0 ], 0x10, %r30;

# 这行提交之前的异步复制操作组,确保它们开始执行。
cp.async.commit_group ;
```

> 总的来说,这段代码实现了从全局内存到共享内存的异步数据复制。它使用了H100之前的cp.async指令,而不是新的TMA机制。这种方法需要更多的寄存器来计算地址,并且每个线程都参与了数据移动,这与TMA的轻量级、单线程触发的方式形成对比。

在这里，我们观察到较旧的cp.async指令负责全局内存复制。从下面的跟踪中我们可以看到，两次加载都绕过了L1缓存。

- 新旧加载方式的区别：
   - 旧方式：在A和B的数据块（tiles）准备好被Tensor Core使用之前，需要执行ldmatrix指令将数据从共享内存移动到寄存器文件中。
   - 新方式（TMA）：在Hopper架构上，数据可以直接从共享内存中被重复使用，无需额外的ldmatrix指令。

![图5. H100内存图表，显示GMEM吞吐量 = 910.22 GB/s（不使用TMA的Triton GEMM），条件为M=128，N=4096，K=4096](https://files.mdnice.com/user/59/64aff0d2-5c63-48a0-9a18-e3d73d1cf959.png)

通过利用我们上面提到的Triton API变更来使用TMA，我们可以研究Triton为单个2D tile load生成的PTX代码。

**PTX for Loading Tile (cp.async.bulk.tensor) - H100 using TMA**

```python
# 这行代码是一个同步指令,确保所有线程都到达这个点后才继续执行。
bar.sync 	0; 
# 将寄存器%r4中的值右移5位,结果存入%r5。这可能是在计算某种偏移量或索引。
shr.u32 	%r5, %r4, 5;
# 这是一个洗牌指令,在warp内部交换数据。它将%r5的值广播给warp中的所有线程,结果存入%r66。
shfl.sync.idx.b32	%r66, %r5, 0, 31, -1;

# 这是一个选举指令,用于在warp中选择一个线程作为leader。结果存储在谓词%p7中。
elect.sync _|%p7, 0xffffffff;

# 将%r65和%r67中的值相加,结果存入%r24。这可能是在计算目标地址。
add.s32 	%r24, %r65, %r67;
# 将%r66中的值左移7位,结果存入%r25。这可能是在计算某种偏移量。
shl.b32 	%r25, %r66, 7;

# 这是TMA的核心指令。它异步地将2D张量数据从全局内存复制到共享内存。
# @%p8: 只有当谓词%p8为真时才执行此指令
# [%r24]: 目标地址(共享内存)
# [%rd26, {%r25,%r152}]: 指向张量映射的指针、张量映射坐标
# [%r19]: 指向mbarrier对象的指针
# 这条指令展示了TMA的优势:单个线程可以发起大规模数据传输,而不需要每个线程都参与计算地址和移动数据。
@%p8
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r24], [%rd26, {%r25,%r152}], [%r19];
```

`cp.async.bulk.tensor.2d.shared` TMA指令依次传递了共享内存中的目标地址、指向张量映射的指针、张量映射坐标以及指向mbarrier对象的指针。

![图6. H100内存图表 GMEM吞吐量 = 1.45 TB/s（使用TMA的Triton GEMM），条件为M=128，N=4096，K=4096](https://files.mdnice.com/user/59/c96ee170-7211-40c2-a66a-09046c87f6f3.png)

为了获得最佳性能，我们对TMA GEMM kernel进行了广泛的调优。除了块大小、线程束数量和流水线阶段数等其他参数外，我们观察到内存吞吐量的最大增长是在将TMA_SIZE（描述符大小）从128增加到512时发生的。从上面的NCU概况中，我们可以看到最终调优后的 kernel 将全局内存传输吞吐量从910 GB/s提高到了1.45 TB/s，相比非TMA Triton GEMM kernel，GMEM吞吐量增加了59%。

**CUTLASS和Triton FP8 GEMM及TMA实现的比较 - kernel 架构**

![图7. Triton 对比 CUTLASS PingPong FP8 GEMM TFLOPs，M=M，N=4096，K=4096](https://files.mdnice.com/user/59/2cd94df4-1a77-4940-9c2f-c5a190e7e034.png)

上图展示了CUTLASS Ping-Pong GEMM kernel(https://github.com/NVIDIA/cutlass/blob/637b15906358191cb4238af419d408a65819d7ec/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp)与Triton的性能对比。Ping-Pong kernel使用TMA的方式与Triton不同。它利用了所有的硬件和软件功能，而Triton目前并未如此。具体而言，CUTLASS支持以下TMA特性，这些特性有助于解释纯GEMM性能的差距：

- TMA Multicast : 实现从GMEM到多个SM的数据复制
- 线程束专门化 : 使线程块内的线程束组能够承担不同的角色
- 张量映射（TMA描述符）预取 : 实现从GMEM预取张量映射对象，从而允许TMA加载的流水线处理

为了更好地理解性能数据，下面我们将展示一个"加速"图表，以百分比形式突出显示延迟差异：

![图8：CUTLASS Ping-Pong 相比 Triton FP8 使用TMA的百分比加速](https://files.mdnice.com/user/59/ec3e30b6-46ab-4529-9972-02355c4e88b0.png)

这种加速纯粹是 kernel 吞吐量的比较，不包括端到端（E2E）启动开销，我们将在下面讨论这一点。

**TMA描述符移动 - Triton和CUTLASS在端到端性能影响上的一个关键区别**

如前所述，2D+维TMA描述符的创建发生在主机端，然后传输到设备端。然而，这个传输过程根据不同的实现方式会有很大差异。

这里我们展示Triton传输TMA描述符的方式与CUTLASS相比的差异。

回想一下，TMA传输需要一个特殊的数据结构，即通过cuTensorMap API在CPU上创建的张量映射。对于FP8 GEMM kernel 来说，这意味着需要创建三个描述符，分别对应A、B和C。我们可以看到，对于Triton和CUTLASS kernel，都调用了相同的CPU程序。

![图7. 对cuTensorMapEncodeTiled的调用（Triton和CUTLASS都使用这个路径）](https://files.mdnice.com/user/59/f1b6890d-8747-4028-b5cc-81e99865ccc9.png)

然而，对于Triton来说，每个描述符都是在其自己的独立 copy kernel 中传输的，这增加了大量的开销，并成为在端到端推理场景中使用这个kernel的障碍。


![图8. 在 kernel 执行之前，为A、B和C分别启动了三个 H2D copy kernel](https://files.mdnice.com/user/59/9625a3f7-0bf1-440c-9276-ea2a3de7b640.png)

在CUTLASS的实现中并没有观察到这些复制操作，这是由于TMA描述符传递给kernel的方式不同。从下面的PTX（并行线程执行）代码中我们可以看到，在Cutlass中，张量映射是通过值传递给kernel的。

```python
# .entry 声明了一个设备函数的入口点。这是一个CUTLASS GEMM kernel的入口函数。
.entry _ZN7cutlass13device_kernelIN49_GLOBAL__N__8bf0e19b_16_scaled_mm_c3x_cu_2bec3df915cutlass_3x_gemmIaNS_6half_tENS1_14ScaledEpilogueEN4cute5tupleIJNS5_1CILi64EEENS7_ILi128EEES9_EEENS6_IJNS7_ILi2EEENS7_ILi1EEESC_EEENS_4gemm32KernelTmaWarpSpecializedPingpongENS_8epilogue18TmaWarpSpecializedEE10GemmKernelEEEvNT_6ParamsE(

# .param .align 64 .b8 [...] _param_0[1024] 定义了一个1024字节的参数空间,用于传递kernel参数。
.param .align 64 .b8 _ZN7cutlass13device_kernelIN49_GLOBAL__N__8bf0e19b_16_scaled_mm_c3x_cu_2bec3df915cutlass_3x_gemmIaNS_6half_tENS1_14ScaledEpilogueEN4cute5tupleIJNS5_1CILi64EEENS7_ILi128EEES9_EEENS6_IJNS7_ILi2EEENS7_ILi1EEESC_EEENS_4gemm32KernelTmaWarpSpecializedPingpongENS_8epilogue18TmaWarpSpecializedEE10GemmKernelEEEvNT_6ParamsE_param_0[1024]

# mov.b64 %rd110, _ZN7cutlass13device_kernelIN... 将kernel参数的地址移动到寄存器 %rd110 中。
mov.b64 	%rd110, _ZN7cutlass13device_kernelIN49_GLOBAL__N__8bf0e19b_16_scaled_mm_c3x_cu_2bec3df915cutlass_3x_gemmIaNS_10bfloat16_tENS1_14ScaledEpilogueEN4cute5tupleIJNS5_1CILi64EEES8_NS7_ILi256EEEEEENS6_IJNS7_ILi1EEESB_SB_EEENS_4gemm24KernelTmaWarpSpecializedENS_8epilogue18TmaWarpSpecializedEE10GemmKernelEEEvNT_6ParamsE_param_0;

# add.s64 %rd70, %rd110, 704 计算参数中TMA描述符的偏移地址,存储在 %rd70 中。
add.s64 	%rd70, %rd110, 704;
# cvta.param.u64 %rd69, %rd70 将参数地址转换为通用地址空间。
cvta.param.u64 	%rd69, %rd70;

# 这是关键的TMA指令:
# 从全局内存加载2D张量数据到共享内存
# [%rd69, {%r284, %r283}] 指定了源地址(TMA描述符)和坐标
# [%r1880] 可能是指向目标共享内存地址
cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd69, {%r284, %r283}], [%r1880];
```
> 这段代码展示了CUTLASS如何直接在kernel参数中传递TMA描述符,而不是像Triton那样使用单独的内存拷贝。这种方法可以减少启动开销,提高端到端性能。

通过直接传递TMA描述符而不是传递全局内存指针，CUTLASS kernel避免了三个额外的主机到设备（H2D）复制 kernel，相反，这些复制被包含在单个设备 kernel 启动中，用于执行GEMM（通用矩阵乘法）。

由于描述符移动到设备的方式不同，包括准备被TMA消耗的张量在内的kernel延迟有显著差异。对于M=1-128，N=4096，K=4096的情况，CUTLASS Ping Pong kernel 的平均延迟为10微秒，而Triton TMA kernel 平均需要4毫秒才能完成。这慢了大约3330倍，似乎直接与Triton进行TMA描述符传输时的3个独立 kernel 启动有关。

CUDA Graph可能是减少这种开销的一种方法，但考虑到H2D复制造成的开销，当前Triton实现在端到端测量时并不具有竞争力。重新设计Triton编译器管理TMA描述符的方式可能会解决这个差距。因此，在我们上面的数据中，我们专注于比较实际计算 kernel 的吞吐量，而不是端到端性能。

![图10: Triton FP8 TMA GEMM TFLOPs 比较](https://files.mdnice.com/user/59/af57fb3a-3c87-46b3-b3ed-27b96f536322.png)

![](https://files.mdnice.com/user/59/1849d913-b0e7-4e66-b16d-ddd4e2177489.png)

上面的图表和表格总结了我们在单个NVIDIA H100上通过利用TMA硬件单元，相对于非TMA Triton kernel 和高性能CUDA（cuBLAS）kernel，在FP8 GEMM上所能达到的性能提升。需要注意的关键点是，这个kernel相对于竞争对手在批处理大小增加时表现出优越的扩展性。我们基准测试的问题规模代表了在小到中等批量大小的LLM（大型语言模型）推理中常见的矩阵形状。因此，对于那些有兴趣利用这个 kernel 进行FP8 LLM部署的用例来说，中等M范围（M=32到M=128）的TMA GEMM kernel性能将至关重要，因为FP8压缩数据类型可以允许更大的矩阵适应GPU内存。

总结我们的分析，Triton和CUTLASS中的TMA实现在完整功能集支持（如Multicast、预取等）和TMA描述符传递给GPUkernel的方式上存在差异。如果这个描述符能以更接近CUTLASS内核的方式传递（通过值传递），就可以避免多余的主机到设备（H2D）复制，从而大大改善端到端（E2E）性能。

## 未来工作

在未来的研究中，我们计划通过与社区合作，将CUTLASS架构的TMA加载方式整合到Triton中，并研究FP8 GEMM的Cooperative kernel（一种对PingPong kernel 的改进策略），以进一步改善这些结果。

此外，一旦线程块 clusters 和TMA原子操作等特性在Triton中启用，我们可能通过在TMA GEMM kernel 中利用 SplitK 策略获得进一步的加速。这是因为在Hopper架构上，原子操作可以在分布式共享内存（DSMEM）中执行，而不是在L2缓存中。我们还注意到 NVIDIA Hopper GPU与其他AI硬件加速器（如Google的TPU和IBM的AIU）作为数据流架构的相似性。在Hopper上，由于增加了本博客中广泛讨论的TMA和我们计划在未来文章中介绍的DSMEM，数据现在可以从全局内存（GMEM）"流动"到一个相互连接的流处理器（SM）网络。

## 补充：博客代码讲解

下面的注释初版由Cursor自带的claude-3.5-sonnet来生成，我做了一些正确性调整。

```python
import triton
import triton.language as tl
import numpy as np
import torch

# 定义一个使用TMA(Tensor Memory Accelerator)的GEMM(通用矩阵乘法)kernel
@triton.jit
def gemm_kernel_tma(a_desc_ptr, b_desc_ptr, c_desc_ptr,  #
                      prob_m, prob_n, prob_k, block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr):
    
    # 获取当前程序的ID
    pid = tl.program_id(axis=0)
    # 计算M和K维度上的块数
    num_pid_m = tl.cdiv(prob_m, block_m)
    num_pid_k = tl.cdiv(prob_k, block_k)
    # 计算当前块在M和N维度上的索引
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    # 计算A和B矩阵的偏移量
    offs_am = pid_m * block_m
    offs_bn = pid_n * block_n
    offs_k = 0

    # 初始化累加器为零矩阵
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    # 在K维度上进行循环
    for kk in range(0, num_pid_k):
        # 使用TMA从全局内存加载A和B矩阵的块
        a = tl._experimental_descriptor_load(a_desc_ptr, [offs_am, offs_k], [block_m, block_k], tl.float8e4nv)
        b = tl._experimental_descriptor_load(b_desc_ptr, [offs_bn, offs_k], [block_n, block_k], tl.float8e4nv)
        
        # 执行矩阵乘法并累加结果
        accumulator = tl.dot(a, b.T, acc=accumulator, out_dtype=tl.float32)
        offs_k += block_k

    # 将结果转换为float16类型
    accumulator = accumulator.to(tl.float16)
    # 将结果存储到全局内存
    tl._experimental_descriptor_store(c_desc_ptr, accumulator, [offs_am, offs_bn])


# 定义矩阵乘法函数
def matmul(a, b, config=None):

    # 获取输入矩阵的维度
    m, _ = a.shape
    k, n = b.shape

    # 如果提供了配置，则使用配置中的参数
    if config:
        block_m = config["block_m"]
        block_n = config["block_n"]
        block_k = config["block_k"]
        num_warps = config["num_warps"]
        num_stages = config["num_stages"]
    
    # 否则使用默认参数
    block_m = 64
    block_n = 64
    block_k = 256
    num_warps = 4
    num_stages = 4
    TMA_SIZE = 512

    # 创建TMA描述符
    desc_a = np.empty(TMA_SIZE, dtype=np.int8)
    desc_b = np.empty(TMA_SIZE, dtype=np.int8)
    desc_c = np.empty(TMA_SIZE, dtype=np.int8)

    # 创建输出矩阵
    c = torch.empty((m, n), dtype=torch.float16, device='cuda')
    
    # 填充TMA描述符
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(a.data_ptr(), m, k, block_m, block_k, a.element_size(),
                                                            desc_a)
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(b.data_ptr(), n, k, block_n, block_k, b.element_size(),
                                                            desc_b)
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(c.data_ptr(), m, n, block_m, block_n, c.element_size(),
                                                            desc_c)
    
    # 将描述符转移到GPU
    desc_a = torch.tensor(desc_a, device='cuda')
    desc_b = torch.tensor(desc_b, device='cuda')
    desc_c = torch.tensor(desc_c, device='cuda')

    # 计算总块数
    total_blocks_m = triton.cdiv(m, block_m)
    total_blocks_n = triton.cdiv(n, block_n)
    
    # 设置网格大小
    grid = (total_blocks_m * total_blocks_n, 1, 1)
    
    # 启动kernel
    k = gemm_kernel_tma[grid](
        desc_a, desc_b, desc_c,
        m, n, k,
        block_m,
        block_n,
        block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    # 返回结果矩阵
    return c


if __name__ == '__main__':

    # 设置矩阵维度
    M = 128
    N = 4096
    K = 4096

    # 创建随机输入矩阵并转换为float8类型
    a = torch.randn((M, K), device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
    b = b.T.contiguous()

    # 执行矩阵乘法
    triton = matmul(a, b)
```





