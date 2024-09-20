# 深入探讨 Hopper TMA 单元在 FP8 GEMM 运算中的应用

- PyTorch博客资料：https://pytorch.org/blog/hopper-tma-unit/
- PyTorch实现和使用Demo：https://github.com/pytorch-labs/applied-ai/blob/main/kernels/triton/inference/fp8/tma_gemm.py

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
bar.sync 	0;
shr.u32 	%r5, %r4, 5;
shfl.sync.idx.b32	%r66, %r5, 0, 31, -1;

elect.sync _|%p7, 0xffffffff;


add.s32 	%r24, %r65, %r67;
shl.b32 	%r25, %r66, 7;

@%p8
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r24], [%rd26, {%r25,%r152}], [%r19];
```



