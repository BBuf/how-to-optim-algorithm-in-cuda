> 博客来源：https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/

任何一个 CUDA® 教程系列如果没有 GEMM(通用矩阵乘法)的内容都是不完整的。GEMM 可以说是现代 GPU 上最重要的计算例程,它构成了神经网络、大语言模型和许多图形应用中的大部分计算。尽管 GEMM 使用如此广泛,但要高效实现它却非常困难。

这个由3部分组成的教程系列旨在让读者全面了解如何使用 CUTLASS 库在 NVIDIA Hopper GPU 上编写高效的 GEMM kernel。

- [第1部分,也就是本文] 讨论 warpgroup 矩阵乘累加(WGMMA)指令。这些是针对基于 Hopper 架构的 NVIDIA GPU 的 Tensor Core 的基本指令。
- [第2部分] 将讨论高效 GEMM kernel 的整体设计(https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md),包括 CUTLASS kernel 中使用的高级技术,如 warp 专用化和Ping-Pong调度。
- [第3部分] 将讨论持久 kernel和 Stream-K(https://arxiv.org/abs/2301.03598),这是一种在大量问题几何中实现最先进效率的 GEMM 负载均衡策略。

大局观。我们系列的3个部分遵循 GEMM kernel 的整个开发过程,但是是"由内而外"的。首先,我们有调用 Tensor Core 来最终执行计算的分块 GEMM 基元。其次,我们有"每个 CTA"的 GEMM kernel设计 - 由 prologue, mainloop, 和 epilogue 组成 - 其中主要挑战是不要让内存加载上成为快速的 Tensor Core 计算的瓶颈。最后,我们在最外层网格级别有 CTA 的调度,在这里负载均衡考虑成为首要问题。

我们希望在完成这个系列后,读者将成为 GEMM 算法的专家,并能够利用这个算法中的一些精妙想法来设计和实现他们自己工作中的其他 kernel。

## 异步 Warpgroup MMA (WGMMA)

Hopper 引入了异步 warpgroup 级别的矩阵乘累加操作(WGMMA)。一个 warpgroup 由4个连续的 warp 组成,即128个连续的线程,其中第一个 warp 的 warp-rank 是4的倍数。`wgmma.mma_async` 指令由 warpgroup 中的所有128个线程共同执行。这个操作通常遵循以下形式之一,其中矩阵 C 作为累加器:

- C = A * B + C 
- C = A * B, 其中累加器 C 的输入被禁用。

一个值得注意的要求是,操作数 B 必须始终存储在共享内存(SMEM)中。相比之下,操作数 A 可以位于共享内存或寄存器内存(RMEM)中,累加器 C 总是位于 RMEM 中。

这个教程系列组织如下。首先,我们讨论在 CUTLASS 中调用 `wgmma.mma_async` 指令的基本要素。这涉及构建相关的 `TiledMMA` 对象,以及创建和分区 SMEM 张量,以与 WGMMA 兼容。其次,我们讨论确保 WGMMA 正确性的必要同步机制。最后,我们更详细地讨论 WGMMA 中使用的布局,包括核心矩阵和用于来自 SMEM 的操作数的矩阵描述的概念。

在整个教程中,为了简洁起见,我们将 `wgmma.mma_async` 缩写为 `wgmma`。我们的主要代码参考将是 Pradeep Ramani 贡献的 CUTLASS `wgmma` 教程(https://github.com/NVIDIA/cutlass/blob/be60a0b27204078dc0f3f1d6ed4a95cdb2114111/examples/cute/tutorial/wgmma_sm90.cu),它是在 3.5.1 版本中添加的。

## CUTLASS kernel中的WGMMA

本教程的主要目标是解释用于调用Hopper Tensor Cores执行基于tile的GEMM的`wgmma`原语,以及如何将其作为`cute::gemm`调用的一部分来调用。为了说明这一点,让我们考虑一个标准的GEMM kernel,它接受维度为`MxNxK`的输入矩阵A和B,并计算`C=A*B`。为了并行化计算,kernel固定静态tile大小`bM`、`bN`和`bK`,并启动一个`⌈M/bM⌉x⌈N/bN⌉`大小的CTA网格,每个CTA计算输出矩阵的一个`bMxbN`大小的tile `rC`。这将保存在CTA的RMEM中,然后写回到全局C矩阵。

对于每个CTA,我们有kernel的 mainloop 。在`⌈K/bK⌉`次迭代中,我们遍历内部维度,并依次将A和B的bMxbK和bNxbK大小的tile从全局内存加载到共享内存中作为`sA`和`sB`;注意在CUTLASS中,我们将`sB`的形状固定为数学上的转置。(实际上,遵循常见做法,我们将A和B的tile加载到循环`SMEM`缓冲区中,其中阶段数由编译时整数(如2或3)给出。`sA`和`sB`的shape元组的最后一个模式由这个阶段计数给出。)`cute::gemm`调用然后计算`sA`和`sB`的(分阶段切片的)乘积,并将值依次累加到`rC`中。主循环完成后,`epilogue`将`rC`写出到全局内存。

现在,我们想要解释以下`cute::gemm`调用及其参数,它们出现在我们从wgmma教程中选择性提取的以下代码片段中(https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/wgmma_sm90.cu#L73)(隐藏了与我们无关的程序部分,如流水线TMA加载):

```c++
template <class TiledMMA, ... >
__global__ device_gemm(TiledMMA tiled_mma, ...) {
  // PROLOGUE
  // ...
  // Define A/B partitioning and C accumulators
  ThrMMA thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);  // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tCsB = thr_mma.partition_B(sB);  // (MMA,MMA_N,MMA_K,PIPE)
  Tensor tCgC = thr_mma.partition_C(gC);  // (MMA,MMA_M,MMA_N)
 
  // Allocate accumulators and clear them
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);  // (MMA,MMA_M,MMA_N)
  clear(tCrC);
 
  // Allocate "fragments"
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);  // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);  // (MMA,MMA_N,MMA_K,PIPE)
   
  // PIPELINED MAIN LOOP
  while (k_tile_count > -K_PIPE_MAX) {
    // ...
    // MMAs to cover 1 K_TILE
    cute::warpgroup_arrive();
    // (V,M,K) x (V,N,K) => (V,M,N)
    cute::gemm(tiled_mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);
    cute::warpgroup_commit_batch();
    // Wait for all MMAs in a K_TILE to complete
    cute::warpgroup_wait<0>();
    // ...
  }
 
  // EPILOGUE
  // ...
}
```

在CUTLASS的MMA范式中(https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/0t_mma_atom.md),`cute::gemm`方法旨在通过统一的接口暴露特定架构的MMA指令。(事实上,如果你查看SM80教程的GEMM kernel(https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/sgemm_sm80.cu#L275),你会发现那里的`cute::gemm`调用在语法上与上面给出的**完全相同**。)然而,`cute::gemm`调用中涉及的参数定义包含了许多WGMMA特有的方面:

- `TiledMMA`对象`tiled_mma`的定义封装了`cute::gemm`调度到特定wgmma PTX指令所需的信息。
- SMEM张量`sA`和`sB`的布局必须定义为与wgmma兼容。
- 片段`tCrA`、`tCrB`和`tCrC`是使用`TiledMMA`对象构建的数据的线程级分区,因此具有程序员应该了解的WGMMA特定布局。
- 片段`tCrA`(如果从SMEM获取操作数A)和`tCrB`不是从SMEM复制值的寄存器支持张量,而是在SMEM之上构建的矩阵描述符。

最后,当然还有围绕`cute::gemm`调用的warpgroup同步原语。我们将依次解释所有这些概念。

### TiledMMA object for WGMMA

假设数据类型为FP16,A和B为`MN-major`,因此根据BLAS符号,我们计算一个NT GEMM。我们在主机上使用`cute::make_tiled_mma`方法构造`TiledMMA`对象,如下所示:

```c++
TiledMMA tiled_mma = cute::make_tiled_mma(
  SM90_64x64x16_F16F16F16_SS<GMMA::Major::MN,GMMA::Major::MN>{});
```

尽管`cute::make_tiled_mma`也有一些可选参数,让我们关注手头的参数——`MMA Atom`。这是一个封装底层PTX调用的结构体,在这种情况下是:

```c++
wgmma.mma_async.sync.aligned.m64n64k16.f16.f16.f16
```

CUTLASS的表示方法使得我们可以直接看出包装的PTX指令和MMA Atom操作之间的关系。首先,SM90是Hopper架构的另一个名称。SM90 MMA Atom操作被标记为`SM90_MxNxK_XYZ_SS`或`SM90_MxNxK_XYZ_RS`,带有两个模板参数,可以是`GMMA::Major::MN`或`GMMA::Major::K`。它们的含义如下:

- X和Y是操作数的数据类型。
- Z是累加器的数据类型。
- MxNxK是wgmma指令计算的tile大小 - 即"wgmma Atom"。并非所有MxNxK值都是可能的。这里是允许的形状列表(https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shape): M总是64,N是8到256之间8的倍数,对于16位操作数数据类型,K是16(更一般地说,K固定为32字节)。
- 后缀RS或SS表示操作数A是从寄存器(R)还是共享内存(S)获取。操作数B总是从共享内存获取,因此是S。
- 两个模板参数表示操作数A和B是在MN模式还是K模式下内存连续。例如,在BLAS表示法中,操作数都是K-major将对应于TN gemm(参见此表(https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/0x_gemm_tutorial.md#aside-m-major-n-major-k-major))。注意,对于16位操作数数据类型,内存布局可以是MN-major或K-major。但是,对于非16位操作数数据类型,**布局必须始终是K-major**。

这就是你需要了解的MMA Atom操作的语法!现在,我们已经强调WGMMA是一个warpgroup范围的指令。在代码中,你可以使用其size来获取由TiledMMA对象定义的MMA操作中参与的线程数。例如,以下主机代码

```c++
dim3 dimBlock(cute::size(tiled_mma));
```

规定每个CTA在kernel中启动时有1个warpgroup,包含128个线程。

假设我们想要2个warpgroup来执行WGMMA,让不同的warpgroup独立计算输出tile的一半(并且每个warpgroup发出它们各自的wgmma指令)。为了实现这一点,我们可以将一个非平凡的布局(AtomLayoutMNK)作为第二个参数传递给make_tiled_mma方法。例如,以下代码

```c++
TiledMMA tiled_mma = make_tiled_mma(
 SM90_64x64x16_F16F16F16_SS{},
 Layout<Shape<_2,_1,_1>>{});
```

定义了一个WGMMA操作,其中warpgroup 1和2分别计算沿M维度划分的输出tile的上半部分和下半部分(现在假设bM是128的倍数)。此外,`size(tiled_mma)`的值将等于256。

一般来说,`make_tiled_mma`的两个可选布局参数 - `AtomLayoutMNK`和`PermutationMNK` - 对任何MMA Atom的工作方式都是相同的。要理解`PermutationMNK`的用法,我们推荐参考Cris Cecka的精彩解释(https://github.com/NVIDIA/cutlass/discussions/1345)。


### WGMMA的共享内存布局约束

接下来,我们解释在给定MMA atom选择的情况下,SMEM中操作数矩阵的tile大小和布局的约束。首先,对于任何MMA指令,MMA atom的MxNxK需要能够整除操作数和累加器tile的大小。在我们的例子中,这意味着bM应该是64的倍数,bN是64的倍数,bK是16的倍数。

其次,WGMMA对sA和sB的SMEM布局(包括形状和步长)施加了一个额外的特定约束,这个约束随所选择的交织模式而变化。特别是,(分阶段切片的)sA的布局通常不仅仅是`(bM,bK):(1,bM)`或`(bM,bK):(bK,1)`,sB也是如此。

要深入理解这些要求,需要核心矩阵的概念,我们将在下面介绍。然而,从实践的角度来看,我们总是可以使用CUTLASS提供的某些预定义布局原子,然后使用`cute::tile_to_shape`方法来构造保证与wgmma兼容的布局。在我们的例子中,我们在主机上准备tile大小和sA、sB如下(其中`T=cutlass::half_t`是CUTLASS对FP16的命名):


```c++
auto bM = Int<128>{};
auto bN = Int<128>{};
auto bK = Int< 64>{};  
auto bP = Int<  3>{};  // Pipeline
 
auto sA = cute::tile_to_shape(
    GMMA::Layout_MN_SW128_Atom<T>{},
    cute::make_shape(bM, bK, bP)
);
auto sB = cute::tile_to_shape(
    GMMA::Layout_MN_SW128_Atom<T>{},
    cute::make_shape(bN, bK, bP)
);
```

这里,MN表示布局原子适用于MN-major操作数,SW128是128字节的交织模式。打印sA或sB会显示

```c++
Sw<3,4,3> o smem_ptr[16b](unset) o ((_64,_2),(_8,_8),_3):((_1,_512),(_64,_1024),_8192)
```

这个布局是从哪里来的? `cute::tile_to_shape` 接收一个布局(即所谓的tile)并将其复制到更大的形状上(类似于numpy.tile)。暂时不考虑swizzle函数`Sw<3,4,3>`,我们可以看到布局原子由`(64,8):(1,64)`给出,并以列优先的方式平铺到形状`(128, 64, 3)`上。对于MxK形状,较小的外部步长512位于M模式,而较大的外部步长1024位于K模式。(最大的步长8192位于阶段计数P模式,这是有道理的,因为sA或sB的不同阶段切片不应该在内存中混合。)

注意,64乘以sizeof(`half_t`)等于128字节,这就是swizzle模式的名称。这是有意设计的:由于核心矩阵的工作方式,我们总是安排布局原子在连续方向上的长度等于swizzle字节数 - 要么是无swizzle时的16,要么是32、64或128之一。

相比之下,如果我们考虑:

```c++
auto sA = cute::tile_to_shape(
  GMMA::Layout_K_SW128_Atom<T>{},
  cute::make_shape(bM,bK,bP)
);
auto sB = cute::tile_to_shape(
  GMMA::Layout_K_SW128_Atom<T>{},
  cute::make_shape(bN,bK,bP)
);
```

那么打印sA会显示

```c++
Sw<3,4,3> o smem_ptr[16b](unset) o (_128,_64,_3):(_64,_1,_8192)
```

因为我们改为将 `(8,64):(64,1)` 平铺到 `(128,64,3)` 上。(注意布局 `((_8,_16),(_64,_1),_3):((_64,_512),(_1,_0),_8192)` 合并为 `(_128,_64,_3):(_64,_1,_8192)`)。

```markdown
上面的布局转换可能有点绕，这里尝试解释一下。

1. 首先，`(8,64):(64,1)` 是初始的布局原子(Layout Atom)：
- `8x64` 表示一个基本块的形状
- `(64,1)` 表示在内存中的步长(stride)，其中64是行步长，1是列步长
2. 当这个布局原子被平铺(tile)到 `(128,64,3)` 的形状上时，CUTLASS会进行布局转换。中间过程显示为： `((_8,_16),(_64,_1),_3):((_64,_512),(_1,_0),_8192)` 这可以这样理解：
- `(_8,_16)` 表示将原来的128行分割成了8x16的块
- `_64,_1` 表示64列保持不变
- `_3` 表示有3个pipeline阶段
- 对应的步长分别是：`(_64,_512)` 表示行方向的步长，`(_1,_0)` 表示列方向的步长，`_8192`表示pipeline方向的步长
3. 最终这个复杂的表达式被合并简化为： `(_128,_64,_3):(_64,_1,_8192)` 这是因为：
- `8x16`的块在行方向上合并成了128(8*16=128)
- 64列保持不变
- 3个pipeline阶段保持不变
- 步长也相应简化：64是行步长，1是列步长，8192是pipeline步长
```

一般来说,我们可以在8种布局原子中选择,它们对应于MN或K优先以及四种交织模式之一:

- 无交织: 不进行交织。隐含16字节边界。
- 32字节交织: 交织2个连续的16字节段。
- 64字节交织: 交织4个连续的16字节段。
- 128字节交织: 交织8个连续的16字节段。

这些布局原子在CUTLASS代码库中定义如下(https://github.com/NVIDIA/cutlass/blob/36cbfcf483cc9d2ee65a55c199176ce96da1e33e/include/cute/atom/mma_traits_sm90_gmma.hpp#L66):

```c++
GMMA::Layout_MN_INTER_Atom<T>
GMMA::Layout_MN_SW32_Atom<T>
GMMA::Layout_MN_SW64_Atom<T>
GMMA::Layout_MN_SW128_Atom<T>
 
GMMA::Layout_K_INTER_Atom<T>
GMMA::Layout_K_SW32_Atom<T>
GMMA::Layout_K_SW64_Atom<T>
GMMA::Layout_K_SW128_Atom<T>
```

这些布局原子必须传递到`tile_to_shape`中,并传递SMEM形状为`make_shape(bM,bK,bP)`或`make_shape(bN,bK,bP)`,形状的模式按该顺序给出,使得布局原子的tile大小能整除较大SMEM形状的tile大小。这最终是由于选择的交织模式对SMEM形状的约束,与MMA原子形状引起的约束是分开的。

### WGMMA Fragments and Descriptors

我们已经在主机上创建了 TiledMMA 对象，并相应地准备了 SMEM 布局。现在，在设备上，我们可以使用 TiledMMA 对象 `tiled_mma` 来构造要传递给 `cute::gemm` 调用的适当的分区 tensor。首先，我们通过在 `tiled_mma` 上调用 `get_thread_slice` 方法来创建一个 `ThrMMA` 对象，称为 `thr_mma`，其中线程索引从 0 到 127（包括 127）。
然后，引用上面的 kernel 代码片段，**对于任何线程索引**，打印 tensors tCsA 和 tCsB 将显示以下内容：

```c++
tCsA: Sw<3,4,3>_smem_ptr[16b](0x7f8800000400) o
    ((_64,(_8,_2)),_2,_4,_3):((_1,(_64,_1024)),_512,_2048,_8192)
tCsB: Sw<3,4,3>_smem_ptr[16b](0x7f880000c400) o
    ((_64,(_8,_2)),_2,_4,_3):((_1,(_64,_1024)),_512,_2048,_8192)
```

根据注释，tCsA 的形状可以看作是 (`MMA,MMA_M,MMA_K,PIPE`):

- `MMA` 是 MMA Atom 的 `NxK` 形状。
- `MMA_M` 和 `MMA_K` 是 sA 的 M 和 K 模式上的平铺范围（因此 `MMA_M=bM/64=2` 和 `MMA_K=bK/16=4`）。
- `PIPE` 是阶段的数量。

步长和 Swizzle 模式来自 sA。WGMMA 特有的注意点是，tCsA 不是 SMEM 的线程级别切片，而是整个 SMEM 张量的重新组织的布局。

接下来，对于任何线程索引，打印“片段”tCrA 和 tCrB 将显示：

```c++
tCrA: GMMA::DescriptorIterator o (_1,_2,_4,_3):(_0,_64,_256,_1024)
tCrB: GMMA::DescriptorIterator o (_1,_2,_4,_3):(_0,_64,_256,_1024)
```
在内部，CUTLASS 构造了一个“矩阵描述符“(https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor)，它是一个 64 位寄存器值，描述了 SMEM 并且适合 wgmma 指令使用。对于程序员，需要注意的是，SMEM 的值并不是被复制到 RMEM，而是访问 tCrA 和 tCrB 的值实际上访问的是这些 64 位描述符。此外，这些张量被称为“迭代器”意味着只有在给定 wgmma 指令中使用的单个 64 位描述符被保存在寄存器中（例如，而不是所有的 24 个）。

与操作数相比，累加器张量被定义得更为标准化。对于线程 0，打印 tCgC 和 tCrC 将显示：

```c++
tCgC: gmem_ptr[16b](0x7f877a780000) o ((_2,_2,_8),_2,_2):((512,_8,4096),_64,32768)
tCrC: ptr[16b](0x7feee1fffbe0) o ((_2,_2,_8),_2,_2):((_1,_2,_4),_32,_64)
```

tCgC 是我们在 epilogue 中想要将累加器的值复制到的输出 GMEM 张量的切片，而 tCrC 是在 mainloop 中计算这些值的寄存器支持张量。这些张量的 (MMA,MMA_M,MMA_N) 形状可以解释如下：在 MMA atom 的 MxN=64x64 输出 tile 中，128 个线程的每一个持有 `32=2*2*8` 个值，而 MMA_M=MMA_N=2 与 tCsA 和 tCsB 相同。

每个线程以一种需要将 32 因子化为 (2,2,8) 的方式持有 atom 的 32 个值，以便能够定义 tCgC 的布局对应的步长。该分区模式可以从 PTX 文档(https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#wgmma-64n16-d)中的这张图片中读取。


![](https://files.mdnice.com/user/59/2ff62d95-7486-4f48-936b-003b6ae1c525.png)

这说明了线程持有 32 个值的复制 Z 模式。例如，线程 0 持有位于 (0,0) 、 (0,1) 、 (8,0) 和 (8,1) 的值，并且每隔 8 列向右重复。

### The gemm call, revisited

Let’s return to line 25 of the kernel code snippet above:

```c++
// (V,M,K) x (V,N,K) => (V,M,N)
cute::gemm(tiled_mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);
```

The various overloads of the `cute::gemm` method serve to first loop over the outer modes MMA_M/N and MMA_K. Once those coordinates are chosen, we’re just computing with the MMA atom tile shape. Put another way, we first reduce to the overload of `cute::gemm` for the dispatch shape (V)x(V)=>(V)(https://github.com/NVIDIA/cutlass/blob/be60a0b27204078dc0f3f1d6ed4a95cdb2114111/include/cute/algorithm/gemm.hpp#L178).

The code then invokes the fma operation(https://github.com/NVIDIA/cutlass/blob/be60a0b27204078dc0f3f1d6ed4a95cdb2114111/include/cute/arch/mma_sm90_gmma.hpp#L401) of the MMA atom (precisely, within the mma_unpack(https://github.com/NVIDIA/cutlass/blob/be60a0b27204078dc0f3f1d6ed4a95cdb2114111/include/cute/atom/mma_traits.hpp#L112) method). This contains the inline PTX assembly:

```c++
CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a,
      uint64_t const& desc_b,
      uint32_t& d00, uint32_t& d01, uint32_t& d02, uint32_t& d03,
      uint32_t& d04, uint32_t& d05, uint32_t& d06, uint32_t& d07,
      uint32_t& d08, uint32_t& d09, uint32_t& d10, uint32_t& d11,
      uint32_t& d12, uint32_t& d13, uint32_t& d14, uint32_t& d15,
      GMMA::ScaleOut const scale_D = GMMA::ScaleOut::One)
  {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
    asm volatile(
    "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %18, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f16.f16.f16 "
      "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
      " %8,  %9,  %10, %11, %12, %13, %14, %15},"
      " %16,"
      " %17,"
      " p,   %19, %20, %21, %22;\n"
    "}\n"
      : "+r"(d00), "+r"(d01), "+r"(d02), "+r"(d03),
        "+r"(d04), "+r"(d05), "+r"(d06), "+r"(d07),
        "+r"(d08), "+r"(d09), "+r"(d10), "+r"(d11),
        "+r"(d12), "+r"(d13), "+r"(d14), "+r"(d15)
      : "l"(desc_a),
        "l"(desc_b),
        "r"(int32_t(scale_D)),
        "n"(int32_t(scaleA)),
        "n"(int32_t(scaleB)),
        "n"(int32_t(tnspA)),
        "n"(int32_t(tnspB)));
#else
    CUTE_INVALID_CONTROL_PATH(
        "Attempting to use SM90_64x64x16_F16F16F16_SS "
        "without CUTE_ARCH_MMA_SM90A_ENABLED");
#endif
  }
```

The corresponding PTX documentation for this syntax is here(https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma). Consistent with the descriptions of the tensors tCrA, tCrB, and tCrC above, observe that we have the uint64 variables desc_a and desc_b for the operands along with 16 uint32 variables for the accumulator. scale_D is either 0 or 1, and controls whether or not the accumulator is zero-initialized.

In addition, the variables `scaleA`, `scaleB`, `tnspA`, `tnspB` are determined at compile-time outside the fma method via template parameters. `scaleA` and `scaleB` are either 1 or -1 for negating the operand, while `tnspA` and `tnspB` indicate whether to transpose the operand, and are 0 or 1 for `GMMA::Major::K` or `GMMA::Major::MN`, respectively.

### Synchronization for WGMMA

It remains to explain the synchronization primitives surrounding the `cute::gemm` call:

```c++
cute::warpgroup_arrive();
cute::gemm(tiled_mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);
cute::warpgroup_commit_batch();
cute::warpgroup_wait<0>();
```

Why are these additional commands necessary at all? They have to do with wgmma‘s nature as an asynchronous instruction. In the context of the Hopper architecture, asynchronous indicates that wgmma can run concurrently with other operations, hence necessitating a synchronization mechanism for dependent steps. This mechanism is elaborated upon in the PTX memory consistency model(https://docs.nvidia.com/cuda/archive/12.3.2/parallel-thread-execution/index.html#program-order-async-operations). Improper synchronization in code can result in (a) subtle race conditions, leading to challenging bugs, (b) the compiler serializing the wgmma instructions, which can cause significant performance degradation, or (c) undefined behavior.

The highlighted cute methods wrap the following PTX instructions:

- `cute::warpgroup_arrive()` — `wgmma.fence.sync.aligned`;
- `cute::warpgroup_commit_batch()` — `wgmma.commit_group.sync.aligned`;
- `cute::warpgroup_wait<N>()` — `wgmma.wait_group.sync.aligned N`;

(Note that we’ve been using wgmma as shorthand for wgmma.mma_async throughout, but in this subsection only we disambiguate this.) Let’s connect the usage of these commands to the following description of WGMMA-based GEMM taken verbatim from the PTX documentation(https://docs.nvidia.com/cuda/archive/12.3.2/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-multiply-accumulate-instructions):

- Load matrices A, B, and D into registers or into shared memory.
- Perform the following fence operations:
    - `wgmma.fence` operations to indicate that the register/shared-memory across the warpgroup have been written into.
    - `fence.proxy.async` operation to make the generic proxy operations visible to the async proxy.
- Issue the asynchronous matrix multiply and accumulate operations using the `wgmma.mma_async` operation on the input matrices. The `wgmma.mma_async` operation is performed in the async proxy.
- Create a wgmma-group and commit all the prior outstanding `wgmma.mma_async` operations into the group, by using `wgmma.commit_group` operation.
- Wait for the completion of the required wgmma-group using `wgmma.wait_group`.
Once the wgmma-group completes, all the `wgmma.mma_async` operations have been performed and completed.


We explain these points in order. First, a `wgmma.fence` instruction ensures that `wgmma.mma_async` only accesses certain RMEM addresses after all prior accesses to such addresses have finished. Without the `wgmma.fence`, the behavior is undefined. An exception to this rule is that Hopper allows multiple `wgmma.mma_async` instructions to be in flight simultaneously. As long as these `wgmma.mma_async` instructions have the same accumulator shape, they can share the same accumulator tensor, i.e., write to the same register memory addresses. In that case, a fence is not required. For example, we don’t need to insert a `wgmma.fence` within the loop over `MMA_K` done as part of the `cute::gemm` call.


Just like TMA operations(https://research.colfax-intl.com/tutorial-hopper-tma/), `wgmma.mma_async` is performed in the async proxy. Hence, if operations performed in the generic proxy affect the SMEM read by `wgmma.mma_async`, we need to issue `fence.proxy.async`. For example, this would be the case if we copied A and B into SMEM via ordinary ld.global / st.shared operations. Since we use TMA load, we don’t need `fence.proxy.async` in our example, and indeed it doesn’t appear in the WGMMA tutorial code or in the mainloop of CUTLASS Hopper GEMM kernels. (To verify this, note that `fence.proxy.async` is wrapped by `cutlass::arch::fence_view_async_shared()`).

The `wgmma.commit_group` instruction creates a new wgmma-group per warpgroup and batches all prior `wgmma.mma_async` instructions initiated by the executing warpgroup but not committed to any wgmma-group into the new wgmma-group. In our example, `cute::warpgroup_commit_batch()` batches `MMA_M*MMA_N*MMA_K` many `wgmma.mma_async` instructions into one wgmma-group.

Finally, the `wgmma.wait_group` instruction with argument N will make the executing thread wait until only N or fewer of the most recent wgmma-groups are pending and all the prior wgmma-groups committed by the executing threads are complete. In our example, we let N=0, so the warpgroup simply waits for the completion of the entire wgmma-group before continuing to execute any subsequent instructions.

In situations where the warpgroup has the opportunity to perform independent computation, flexibility with the parameter N comes in handy. For example, this comes into play with the GEMM-softmax overlapping strategy employed in the design of FlashAttention-3.


### WGMMA core matrices

This last section discusses further the layout requirements for tiles of matrices A and B loaded into SMEM, supposing that wgmma sources both of its operands from SMEM. To simplify the discussion, first suppose that A is row-major and B is column-major (i.e., both are K-major). Recall also that the wgmma instruction’s tile shape MxNxK is constrained so that M is 64, K times the size of the datatype is 32 bytes, and N is a multiple of 8 running from 8 to 256. To avoid confusion with A/B or sA/sB, let’s notate the WGMMA atom tiles as wA and wB.

The matrices wA and wB are divided into a number of smaller matrices called core matrices. Each core matrix has a strided direction and a contiguous direction, such that its length is 8 in the strided direction and 16 bytes in the contiguous direction. Matrix wA is made up of 8x2 core matrices and Matrix wB is made up of 2x(N/8) core matrices. We illustrate a tiling of wA and wB by core matrices as follows (with images taken from the PTX documentation):

![](https://files.mdnice.com/user/59/dcebbc6a-2310-4ba8-ae20-66d748960e5b.png)

As mentioned above, wgmma in SS mode requires matrix descriptors(https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor) for both wA (desc-a) and wB (desc-b) as inputs. This descriptor encodes five parameters:

- Start address: starting base address of the operand in SMEM.
- LBO (leading dimension byte offset): the distance, in bytes, between two adjacent core matrices in the K dimension.
- SBO (stride dimension byte offset): the distance, in bytes, between two adjacent core matrices in the M or N dimension.
- Swizzling mode: none, 32, 64, or 128 bytes.
- Matrix base offset: This is used to resolve SMEM alignment problems in case SMEM addresses are not aligned to the byte boundary of the repeating pattern for the swizzle mode.

LBO and SBO are indicated in the figures above.

The `make_gmma_desc`(https://github.com/NVIDIA/cutlass/blob/06b21349bcf6ddf6a1686a47a137ad1446579db9/include/cute/atom/mma_traits_sm90_gmma.hpp#L194C1-L194C54) method in CUTLASS constructs the descriptor (as an instance of GmmaDescriptor(https://github.com/NVIDIA/cutlass/blob/be60a0b27204078dc0f3f1d6ed4a95cdb2114111/include/cute/arch/mma_sm90_desc.hpp#L86)) based on the SMEM tensor’s layout provided as input. Provided that the input tensor’s layout is created using one of the eight canonical GMMA layout atoms and `tile_to_shape`, as previously detailed in “SMEM layout constraints for WGMMA”, `make_gmma_desc` will accurately calculate the LBO and SBO, determine the swizzling mode, and construct the descriptor. For example, the GmmaDescriptor describes the following admissible WGMMA layouts in the K-major case (where `T*sizeof(dtype)=16`):


```c++
No swizzle       : Swizzle<0,4,3> o smem_ptr o ((8,m),(T,2)):((1T,SBO),(1,LBO))
32-byte swizzle  : Swizzle<1,4,3> o smem_ptr o ((8,m),(T,2)):((2T,SBO),(1, T ))
64-byte swizzle  : Swizzle<2,4,3> o smem_ptr o ((8,m),(T,2)):((4T,SBO),(1, T ))
128-byte swizzle : Swizzle<3,4,3> o smem_ptr o ((8,m),(T,2)):((8T,SBO),(1, T ))
```

For the compact(https://github.com/NVIDIA/cutlass/blob/be60a0b27204078dc0f3f1d6ed4a95cdb2114111/include/cute/layout.hpp#L415) layouts produced by the GMMA layout atom => tile_to_shape pattern (note the GMMA layout K atoms have a larger K-mode than the WGMMA atom shape in the case of 64 and 128-byte swizzle!), we have these corresponding values for LBO and SBO:

```c++
No swizzle       : LBO = 16x8 = 128 bytes. SBO = 32x8 = 256 bytes.
32-byte swizzle  : SBO = 32x8 = 256 bytes.
64-byte swizzle  : SBO = 64x8 = 512 bytes.
128-byte swizzle : SBO = 128x8 = 1024 bytes.
```

Most notably, for 64 and 128-byte swizzle, the strides are such that the given admissible WGMMA layouts are **not** compact. Rather, one has sets of 2 or 4 WGMMA atom operand tiles stacked side-by-side in the K-direction, resulting in strides of 4T and 8T for the core matrix M-mode. Put another way, when swizzling one interleaves in memory the 2, 4, or 8 core matrices logically adjacent in the K-mode, and these core matrices will belong to different WGMMA atoms for 64 and 128-byte swizzle.

For the sake of completeness, we also give the admissible WGMMA layouts in the MN-major case:

```c++
No swizzle       : Swizzle<0,4,3> o smem_ptr o ((T,1,m),(8,k)):((1,T,SBO),(1T,LBO))
32-byte swizzle  : Swizzle<1,4,3> o smem_ptr o ((T,2,m),(8,k)):((1,T,LBO),(2T,SBO))
64-byte swizzle  : Swizzle<2,4,3> o smem_ptr o ((T,4,m),(8,k)):((1,T,LBO),(4T,SBO))
128-byte swizzle : Swizzle<3,4,3> o smem_ptr o ((T,8,m),(8,k)):((1,T,LBO),(8T,SBO))
```

## Conclusion

In this [Part 1] of the GEMM series, we have covered the core concepts involved in using WGMMA (warpgroup matrix-multiply and accumulate) as a primitive in Hopper-based GEMM.

WGMMA requires a warpgroup — 128 threads — to collectively execute the matrix multiplication, and can only operate on certain fragments of matrices. We went into the special shapes and layouts involved in this, with an emphasis on how to construct operand layouts guaranteed to be accepted by WGMMA using the canonical GMMA Layout => `tile_to_shape` pattern.

For its usage to be well-defined, WGMMA also requires certain synchronization mechanisms. To this end, we explained the uses of `wgmma.fence`, `fence.proxy.async`, `wgmma.commit_group` and `wgmma.wait_group` in relation to `wgmma.mma_async`.

Lastly, we explained in some detail the inner workings of WGMMA core matrices and how CUTLASS constructs matrix descriptors for those operands sourced from SMEM.

Taken as a whole, this blog post should enable the programmer to write CUTLASS kernels on Hopper that use WGMMA. In [Part 2], we will extend this discussion to incorporate TMA, and how to use TMA and WGMMA in tandem in a Hopper GEMM kernel so as to overlap copy and compute.



