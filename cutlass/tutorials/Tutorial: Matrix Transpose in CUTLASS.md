> 博客来源：https://research.colfax-intl.com/tutorial-matrix-transpose-in-cutlass/ ，这里做了一个翻译学习一下，通过这篇博客可以对CuTe编程时涉及到的内存拷贝（矩阵转置）有简单的了解，是一篇很优质的技术博客。

# 教程：CUTLASS中的矩阵转置 (使用CuTe把矩阵转置优化到GPU内存带宽上下限)

本教程的目标是阐明在使用CUTLASS及其核心后端库CuTe在NVIDIA® GPU上编程时涉及的内存复制概念和技术。具体来说，我们将以矩阵转置任务作为说明这些概念的示例。我们选择这个任务是因为它除了将数据从一组地址复制到另一组地址外不涉及其他操作，这使我们能够单独研究内存复制那些方面的优化，如内存合并访问，这些方面可以与同时涉及计算的工作负载分开。

我们的方法从Mark Harris的高效矩阵转置教程(https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)中获得灵感，我们推荐该教程作为对矩阵转置问题的深入讨论，该讨论不直接涉及我们在这里使用的CuTe抽象。相反，对于已经熟悉Harris教程的读者来说，我们的教程也可以作为这些抽象的介绍。无论如何，我们将在解释如何使用CuTe实现相应的优化解决方案之前，先回顾一下该教程中的关键思想。

## 合并访问的回顾

在许多计算工作负载中，特别是那些在机器学习/人工智能应用中发现的我们经常使用被称为张量的多维数组。由于计算机内存本质上是一维的，这些张量必须被线性化，或组织成这种一维空间。因此，某些维度中相邻的元素在内存中可能不相邻。我们说，当一个维度中相邻的元素在内存中也相邻时，这个维度是连续的。连续维度中连续元素的块也被称为连续的。

对连续内存块的访问 — 即读取或写入 — 被称为合并访问，而对非连续内存块的访问被称为跨步访问。合并访问通常比跨步访问提供更快的性能，因为它们与GPU的内存架构更好地对齐，允许更高效的数据缓存和检索。因此，在为GPU编程时，优化合并内存访问是非常理想的。

然而，某些工作负载需要跨步访问，无法以其他方式实现。矩阵转置 — 或更一般地说，张量置换操作 — 是跨步访问不可避免的典型例子。在这种情况下，关键是要最小化这些低效访问模式对性能的影响。一种标准技术是仅在GPU内存层次结构的较低且更快的层级执行跨步访问，我们现在将回顾这一点。

就我们的讨论而言，GPU内存层次结构有三个可编程级别。从最高到最低级别，我们有全局内存、共享内存和寄存器内存。

全局内存（GMEM），即高带宽内存（HBM），是三者中最大的，也是读写最慢的。例如，NVIDIA H100 Tensor Core GPU有80 GB的GMEM。在这里进行跨步访问将对性能产生最坏的影响。

接下来是共享内存（SMEM），它比GMEM小得多但速度快得多。例如，NVIDIA H100 Tensor Core GPU每个流多处理器（SM）有高达228KB的SMEM。对于更熟悉内存架构的读者，我们注意到SMEM在物理上是从L1缓存中划分出来的。SMEM在同一协作线程数组（CTA）内的所有线程之间共享，每个CTA在SMEM的自己的段内操作。这里的跨步访问仍然不是最优的，但比在GMEM中进行跨步访问要好得多。

最后，我们有寄存器内存（RMEM），它专用于单个线程。

在本教程中，内存访问仅包括复制数字（例如，32位浮点数）的操作，要么从一个级别到另一个级别，要么在同一级别的不同位置之间。

Harris教程中讨论的naive转置方法开始时在GMEM中进行跨步访问：`GMEM -transpose-> GMEM`。然后，他通过首先将数据从GMEM复制到SMEM来改进这一点，这样我们就有：`GMEM -> SMEM -transpose-> SMEM -> GMEM`。这样，跨步加载发生在SMEM中，而GMEM访问都是合并的。

## CuTe的方式

我们现在讨论如何使用CuTe库实现这两种方法。我们首先从朴素方法开始，主要是为了展示什么不应该做。

在CuTe框架中，数据被抽象为`cute::Tensor`对象。CuTe张量由一个指针（C语言意义上的）指向张量的第一个元素，以及一个`cute::Layout`对象组成，后者通过定义`shape`和`stride`整数元组来描述张量中每个元素相对于第一个元素的偏移量。例如，对于一个M乘N的行主序矩阵，我们会将Layout定义为shape `(M, N)`和stride `(N, 1)`。

对于`cute::Layout`，我们注意到在定义新张量的Layout时，可用选项之一是指定它是行主序还是列主序（以步幅的形式表示为`GenRowMajor`或`GenColMajor`）。在列主序矩阵中，同一列内的相邻元素在内存中是连续的，而跨列的相邻元素在内存中是跨步的。默认情况下，CuTe使用列主序Layout。更一般地，我们可以为Layout的形状的每个维度指定步幅。

实现转置的一种简单方法是将输入定义为列主序，输出定义为行主序，然后让CuTe处理复制操作。

```c++
using namespace cute;
int M = 2048, N = 2048;
float *d_S, *d_D; // 声明指向GPU内存的指针，用于源矩阵(d_S)和目标矩阵(d_D)。
// Allocate and initialize d_S and d_D on device (omitted).
 
// Create the row major layouts.
auto tensor_shape = make_shape(M, N);
auto tensor_shape_trans = make_shape(N, M); // 创建表示矩阵形状的对象，一个是原始形状，一个是转置后的形状。
auto gmemLayoutS = make_layout(tensor_shape, GenRowMajor{});
auto gmemLayoutD = make_layout(tensor_shape_trans, GenRowMajor{}); // 为源矩阵和目标矩阵创建行主序 Layout。
 
// Create the row major tensors.
Tensor tensor_S = make_tensor(make_gmem_ptr(d_S), gmemLayoutS);
Tensor tensor_D = make_tensor(make_gmem_ptr(d_D), gmemLayoutD); // 使用之前创建的 Layout和GPU内存指针创建CuTe张量对象。
 
// Create a column major layout. Note that we use (M,N) for shape.
auto gmemLayoutDT = make_layout(tensor_shape, GenColMajor{}); // 创建一个列主序 Layout，注意这里使用的是原始形状(M,N)。
 
// Create a column major view of the dst tensor.
Tensor tensor_DT = make_tensor(make_gmem_ptr(d_D), gmemLayoutDT); // 创建目标数据的列主序视图。这个视图和tensor_D指向相同的内存，但 Layout不同
```

这里有一个重要注意事项，虽然我们有三个张量，但实际上只有两份数据副本。这是因为tensor_D和tensor_DT都使用d_D中的数据 — 它们是同一数据的两种不同视图。我们将在转置kernel中使用列主序视图，但在验证转置结果时使用行主序视图。

接下来，我们需要确定如何将输入张量分成更小的块，以便我们可以在CTA（协作线程数组）上分配。我们可以使用`cute::tiled_divide`方法来实现这一点:

```c++
using namespace cute;
using b = Int<32>;
auto block_shape = make_shape(b{}, b{});       // (b, b)
Tensor tiled_tensor_S  = tiled_divide(tensor_S, block_shape); // ([b,b], m/b, n/b)
Tensor tiled_tensor_DT = tiled_divide(tensor_DT, block_shape); // ([b,b], n/b, m/b)
```

在这里，我们将Tile大小指定为32 x 32。Tile大小是一个重要的调优参数，应该针对每个特定的工作负载进行调整。实际上，32 x 32并不是转置kernel的最优值，我们将在基准测试之前对其进行调优。

`tiled_divide`创建一个具有相同数据但不同 Layout的张量，即数据的不同视图。在我们的例子中，对于`tensor_S`，我们从一个大小为`(M, N)`的2D矩阵开始。`cute::tiled_divide`与Tile大小b一起生成一个大小为`([b,b], M/b, N/b)`的3D矩阵视图；即在`M/b x N/b`网格中的`b x b`小矩阵。

这种视图使得在kernel中访问正确的Tile变得更加容易。

```c++
Tensor tile_S = tiled_tensor_S(make_coord(_, _), blockIdx.x, blockIdx.y);
Tensor tile_DT = tiled_tensor_DT(make_coord(_, _), blockIdx.x, blockIdx.y);
```

在这里，将`make_coord(_, )`作为第一个参数会取整个第一维 (也就是`[b, b]`)，而将第二和第三维的整数值指定为块索引则会取张量的相应切片。（对于熟悉numpy的人来说：CuTe中的下划线(_)等同于numpy中的冒号(:)表示法。）换句话说，tile_S表示位于网格点`(blockIdx.x, blockIdx.y)`的整个`b x b`矩阵。注意，在切片`tiled_tensor_DT`时我们不交换blockIdx.x和blockIdx.y，因为我们已经采用了形状为(M, N)的列主序视图（相比之下，如果我们采用tensor_D的tile划分，我们就需要交换块索引，然后在`local_partition`中为源和目标使用不同的线程 Layout）。然后我们可以通过以下方式获得分配给特定线程的部分：

```c++
auto thr_layout =
      make_layout(make_shape(Int<8>{}, Int<32>{}), GenRowMajor{});
Tensor thr_tile_S = local_partition(tile_S, thr_layout, threadIdx.x);
Tensor thr_tile_DT = local_partition(tile_DT, thr_layout, threadIdx.x); 
```

在这里，我们以每个CTA 256个线程启动kernel，并选择了一个线程 Layout，使得从gmem的加载是合并的，而存储到gmem是非合并的（如我们上面强调的，无论选择哪种线程 Layout，都会有非合并访问）。最后，我们可以使用`cute::copy`来将数据从thr_tile_S复制到thr_tile_DT。

```c++
Tensor rmem = make_tensor_like(thr_tile_S);
copy(thr_tile_S, rmem);
copy(rmem, thr_tile_DT);
```

现在我们可以将这个方法与纯copy kernel进行基准测试比较。copy kernel的代码基于CUTLASS的tiled_copy示例(https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/tiled_copy.cu)，所以我们将解析它作为读者的练习。此外，我们通过实验发现，对于我们的工作负载，`32 x 1024`的tile大小提供了最佳性能。

![基准测试在NVIDIA H100 PCIe GPU上进行，矩阵大小M=N=32768。使用PyTorch基准测试工具Timer进行测量。](https://files.mdnice.com/user/59/8b71d72d-288b-4f18-8ccb-50cf7f458586.png)


正如我们在Harris的文章中看到的，这种朴素方法的速度并不理想。这是因为这种复制是从GMEM到GMEM的跨步复制。为了确认这一点，让我们使用NVIDIA Nsight™ Compute对这个转置进行性能分析。这个性能分析工具可以检测代码中导致性能降低的问题。对朴素转置进行性能分析，GUI的摘要页面显示：

![](https://files.mdnice.com/user/59/33ab6cfd-bb44-479d-9b6d-23ba8a6795c5.png)

Nsight Compute提供了广泛的工具来帮助优化，但全面探索Nsight超出了本文的范围。在本文中，我们将只关注摘要页面。在上面的摘要页面中，我们看到非合并访问的问题确实构成了主要报告的问题。

## CuTe naive实现代码详解

> 这一节是我补充的，可以让读者更加熟悉上面的代码细节。代码在：https://github.com/ColfaxResearch/cfx-article-src/blob/master/transpose-cute/include/transpose_naive.h

```c++
#include "shared_storage.h"
#include "util.h"

using namespace cute;

// 使用模板来支持不同类型的张量和线程 Layout。__launch_bounds__(256, 1)指定了每个线程块最多有256个线程。
template <class TensorS, class TensorD, class ThreadLayoutS, class ThreadLayoutD>
__global__ static void __launch_bounds__(256, 1)
transposeKernelNaive(TensorS const S, TensorD const DT,
                ThreadLayoutS const tS, ThreadLayoutD const tD) {
  using Element = typename TensorS::value_type;

  // 创建了输入和输出张量的局部视图 gS 和 gDT。
  Tensor gS = S(make_coord(_, _), blockIdx.x, blockIdx.y);   // (bM, bN)
  Tensor gDT = DT(make_coord(_, _), blockIdx.x, blockIdx.y); // (bN, bM)

  // Concept:                   Tensor  ThrLayout       ThrIndex
  Tensor tSgS = local_partition(gS, tS, threadIdx.x); // (ThrValM, ThrValN)
  Tensor tDgDT = local_partition(gDT, tD, threadIdx.x); // (ThrValM, ThrValN)

  // 创建一个寄存器内存 rmem。
  Tensor rmem = make_tensor_like(tSgS);

  // 使用 copy 函数将数据从输入复制到寄存器内存，然后从寄存器内存复制到输出，完成转置操作。
  copy(tSgS, rmem);
  copy(rmem, tDgDT);
}

template <typename Element> void transpose_naive(TransposeParams<Element> params) {
  
  //
  // Make Tensors
  //
  auto tensor_shape = make_shape(params.M, params.N);
  auto tensor_shape_trans = make_shape(params.N, params.M);
  auto gmemLayoutS = make_layout(tensor_shape, LayoutRight{});
  auto gmemLayoutD = make_layout(tensor_shape_trans, LayoutRight{});
  Tensor tensor_S = make_tensor(make_gmem_ptr(params.input), gmemLayoutS);
  Tensor tensor_D = make_tensor(make_gmem_ptr(params.output), gmemLayoutD);
  
  // Make a transposed view of the output
  auto gmemLayoutDT = make_layout(tensor_shape, GenColMajor{});
  Tensor tensor_DT = make_tensor(make_gmem_ptr(params.output), gmemLayoutDT);
  
  //
  // Tile tensors
  //
  
  using bM = Int<64>;
  using bN = Int<64>;
  
  auto block_shape = make_shape(bM{}, bN{});       // (bM, bN)
  auto block_shape_trans = make_shape(bN{}, bM{}); // (bN, bM)
  
  Tensor tiled_tensor_S = tiled_divide(tensor_S, block_shape); // ((bM, bN), m', n')
  Tensor tiled_tensor_DT = tiled_divide(tensor_DT, block_shape_trans); // ((bN, bM), n', m')
  
  auto threadLayoutS =
      make_layout(make_shape(Int<8>{}, Int<32>{}), LayoutRight{});
  auto threadLayoutD =
      make_layout(make_shape(Int<8>{}, Int<32>{}), LayoutRight{});
  
  dim3 gridDim(
      size<1>(tiled_tensor_S),
      size<2>(tiled_tensor_S)); // Grid shape corresponds to modes m' and n'
  dim3 blockDim(size(threadLayoutS)); // 256 threads
  transposeKernelNaive<<<gridDim, blockDim>>>(tiled_tensor_S, tiled_tensor_DT,
                                            threadLayoutS, threadLayoutD);
};
```

> 代码中blockDim设置为一维的256，但是我们需要设置每个block的thread Layout，这个thread Layout的维数和Tile的维数应该是相同的，这个例子中设置为`make_layout(make_shape(Int<8>{}, Int<32>{}), LayoutRight{});` 。cutlass官方的tiled_copy也是这么设置的：https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/tiled_copy.cu#L202 

接下来，我们研究改进的算法：首先将数据从GMEM复制到SMEM，然后进行转置，最后从SMEM复制回GMEM。

为了将跨步访问移到SMEM，我们需要一个使用SMEM的张量。我们将使用CuTe在CTA的SMEM中分配一个`array_aligned`对象。

```c++
using namespace cute;
using CuteArray = array_aligned<Element, cosize_v<SmemLayout>>;
 
extern __shared__ char shared_memory[];
CuteArray &smem = *reinterpret_cast<CuteArray*>(shared_memory);
```

这里， smemLayout 是用于单个Tile的SMEM的 Layout。我们现在可以创建一个数据指针为shared_memory的张量：

```c++
Tensor sS = make_tensor(make_smem_ptr(smem.data()), smemLayout);
```

这里有一个重要注意事项，我们必须确保SMEM张量小到足以适应单个SM。换句话说，smemLayout的大小乘以每个Element的字节数必须小于单个SM上的总SMEM容量。除此之外，我们还需要根据每个CTA使用的SMEM来考虑占用率问题。

现在我们可以重复我们对GMEM中数据所做的列主序视图技巧，只是这次我们将其应用于SMEM。我们创建SMEM的两个不同视图 — 一个行主序和一个列主序。

```c++
using namespace cute;
using b = Int<32>;
auto block_shape = make_shape(b{}, b{});       // (b, b)
 
// Create two Layouts, one col-major and one row-major
auto smemLayout = make_layout(block_shape, GenRowMajor{});
auto smemLayoutT = make_layout(block_shape, GenColMajor{});
 
// Create two views of smem
Tensor sS  = make_tensor(make_smem_ptr(smem.data()), smemLayout);
Tensor sD = make_tensor(make_smem_ptr(smem.data()), smemLayoutT);
```

最后，我们可以使用 `cute::copy` 来从 GMEM 复制到 SMEM，然后再从 SMEM 复制回 GMEM。请注意，这里的 S 和 D 是 tensor_S 和 tensor_D 的 `tiled_divide`，而 tS 和 tD 是选择的线程 Layout，以确保对 GMEM 的合并访问（事实上，它们都等于上面的 thr_layout）。

```c++
// Slice to get the CTA's view of GMEM.
Tensor gS = S(make_coord(_, _), blockIdx.x, blockIdx.y); // (bM, bN)
Tensor gD = D(make_coord(_, _), blockIdx.y, blockIdx.x); // (bN, bM)
 
// Create the thread partitions for each Tensor.
Tensor tSgS = local_partition(gS, tS, threadIdx.x);
Tensor tSsS = local_partition(sS, tS, threadIdx.x);
Tensor tDgD = local_partition(gD, tD, threadIdx.x);
Tensor tDsD = local_partition(sD, tD, threadIdx.x);
 
// Copy GMEM to SMEM.
cute::copy(tSgS, tSsS); 
 
// Synchronization step. On SM80 and above, cute::copy
// does LDGSTS which necessitates async fence and wait.
cp_async_fence();
cp_async_wait<0>();
__syncthreads();
 
// Copy transposed SMEM to GMEM.
cute::copy(tDsD, tDgD);
```

现在当我们进行基准测试时，我们得到了一个更好的结果。


![基准测试在 NVIDIA H100 PCIe GPU 上进行，M=N=32768。使用 PyTorch 基准测试工具 Timer 进行测量。](https://files.mdnice.com/user/59/5f7cb12a-7b34-4429-9608-2fe00223bc2c.png)


尽管如此，我们的结果仍然与复制操作的结果有一定差距。再次对代码进行分析，我们可以发现下一个需要解决的问题——Memory Bank Conflict。

## Memory Bank Conflict

带步长的SMEM版本比朴素版本性能好得多，但仍然无法匹配复制操作的性能。这种差异的很大一部分是由于Memory Bank Conflict造成的。在大多数NVIDIA GPU上，共享内存被组织成32个Memory Bank。一个线程束中只有一个线程能够在同一时间访问一个Memory Bank；这对读取和写入访问都适用。因此，如果多个线程试图访问同一个Memory Bank，这些访问就会被串行化。这被称为Bank Conflict。关于Bank Conflict的更深入讨论，我们推荐Lei Mao的优秀博客文章(https://leimao.github.io/blog/CUDA-Shared-Memory-Bank/)。

更具体地说，元素以32位为单位以轮询方式分配给Memory Bank。前32位分配给0号Bank，接下来32位分配给1号Bank，依此类推，直到第33组32位再次分配给0号Bank。所以在一个32x32（行主序）的float类型tile中，每一列都映射到同一个Memory Bank。这是最坏的情况；对于一个有32个线程的线程束，这会导致32路Bank Conflict。

Mark Harris的教程通过将行填充1个数字来解决这个问题。这会偏移元素，使得一列中的每个元素落在不同的Bank中。我们可以通过使用非默认步长在CuTe中复制这种解决方法。CuTe Layout包含有关步长的信息，它定义了每个维度中元素之间的偏移。我们可以通过将列的步长设置为33而不是32来添加填充。在代码中，这可以简单地通过以下方式完成：

```c++
auto block_shape = make_shape(Int<32>, Int<33>); // (b, b+1)
 
// Create two Layouts, one col-major and one row-major
auto smemLayout = make_layout(block_shape, GenRowMajor{});
auto smemLayoutT = make_layout(block_shape, GenColMajor{});
```

然而，这会浪费SMEM中额外32个数字的内存。在本文中，我们将实现一个替代解决方案——交织（swizzle）。

## 交织（Swizzle）和Layout组合

为了讨论交织，我们首先需要详细解释CuTe Layout。 Layout不仅仅是存储张量结构信息的容器，而是一个将一个坐标映射到另一个坐标的函数。例如，考虑一个列主序的张量A，有M行和N列。给定坐标(4,5)——第4列，第5行——这个 Layout会将元组(4,5)映射到整数5M+4。这是1D指针中坐标(4,5)处元素的索引。这种抽象消除了在处理高维张量时经常出现的令人困惑的坐标数学计算。

通常，坐标计算是使用张量的步长来完成的，步长定义了一个维度中相邻元素在1D内存空间中的偏移。例如，对于同一个张量A，步长是(1,M)。一列中的元素相邻，即偏移为1，而一行中的元素偏移为M。

CuTe提供了更复杂的坐标映射函数工具。其中之一就是交织。交织的具体细节超出了本教程的范围，我们建议好奇的读者参考NVIDIA的PTX文档(https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-swizzling-modes)。

通过定义适当的交织函数，CuTe程序员可以像在非交织情况下一样访问数据，而无需担心Bank Conflict。CuTe通过使用组合操作将交织作为张量 Layout的属性来抽象交织细节(https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/02_layout_algebra.md#composition)。

组合，顾名思义，创建了 Layout参数的函数组合。具体来说，当程序员在SMEM中访问交织张量的数据时——比如在CuTe中调用`tensor(i)`，其中逻辑索引i是他们认为的访问位置——他们实际上是在访问swizzle_function(tensor(i))处的数据。

回到转置，我们需要的交织函数是`Swizzle<5,0,5>`。这里的数字5指的是掩码中的位数(https://github.com/NVIDIA/cutlass/blob/main/include/cute/swizzle.hpp#L44)。根据CuTe文档，这个函数通过取下5位与上5位的异或来修改下5位（掩码）。然后将这个模式应用到32x32的地址集合中，一列中没有元素映射到同一内存Banck，从而避免了所有Bank Conflict。我们将这个交织模式添加到我们的 Layout中。

```c++
auto tileLayoutS = make_layout(block_shape, GenRowMajor{});
auto smemLayoutS_swizzle = composition(Swizzle<5, 0, 5>{}, tileLayoutS);
```

![Swizzle<5,0,5>应用于32x32 Tile的结果。索引取模32。](https://files.mdnice.com/user/59/85eb5049-00dd-43db-bb56-4e3ed0fad3ba.png)

我们还注意到，SMEM中的其它数据存储模式将需要不同的交织函数。我们鼓励读者尝试CuTe提供的通用交织函数，并选择最适合他们的函数。

## 通过Layout组合进行转置

上文中，我们讨论了如何通过定义tile的列主序 Layout在SMEM中转置一个tile。这里，我们展示了一种使用 Layout组合的替代方法。具体来说，我们创建一个由swizzled LayoutS和LayoutD组成的 Layout。

```c++
auto tileLayoutD = make_layout(block_shape_trans, GenRowMajor{});
auto smemLayoutD_swizzle = composition(smemLayoutS_swizzle, tileLayoutD);
```

这里的技巧在于，这两个 Layout都被定义为行主序，但CuTe默认使用列主序，包括 Layout代数。我们现在声称 composition(tileLayoutS,tileLayoutD) 等同于

```c++
auto tileLayoutDT = make_layout(block_shape_trans, GenColMajor{});
```

为了解释，让块的维度为`bM`和`bN`，所以`tileLayoutS`和`tileLayoutD`的Shape:Stride分别为`(bM,bN):(bN,1)`和`(bN,bM):(bM,1)`。然后我们有：

```c++
tileLayoutS(tileLayoutD(x,y)) = tileLayoutS(bM*x+y).
```

现在要计算`bM*x+y`在`tileLayoutS`下映射到什么整数，将其表示为域形状`(bM,bN)`中的坐标对很方便。但由于CuTe代数将1D索引映射到坐标时是列主序的（或从左到右），我们发现`bM*x+y`对应于坐标(y,x)。因此，我们得到：

```c++
tileLayoutS(bM*x+y) = tileLayoutS((y,x)) = bN*y+x.
```

这表明组合的 Layout函数等同于 `Layout(bN,bM):(1,bN)`的函数，验证了我们的声明。最后，我们注意到，在存在后置组合（post-composition）与swizzle函数时，前置组合（pre-composition）保留了原地swizzle，从而避免了一些代码重复。

我们的swizzled解决方案使我们接近复制kernel的性能，就像Mark Harris的文章中所做到的那样。

![基准测试在 NVIDIA H100 PCIe GPU 上进行，M=N=32768。使用 PyTorch 基准测试工具 Timer 进行测量。](https://files.mdnice.com/user/59/86603b62-15e3-43ee-bbe2-ecf6198db79a.png)

随着性能接近带宽限制，我们正在接近硬件限制。在prfoile swizzle版本时，ncu摘要页面显示：

![](https://files.mdnice.com/user/59/229dac2f-58e0-433c-9d23-c2973add72b5.png)

我们看到我们已经解决了Memory Bank Conflict问题。最后报告的长记分板停顿问题可以忽略，因为我们正在分析一个完全受内存限制的kernel。

## CuTe transpose_smem 代码实现补充

这部分的代码对应为：https://github.com/ColfaxResearch/cfx-article-src/blob/master/transpose-cute/include/transpose_smem.h

代码里把是否使用Swizzle作为一个模板参数 isSwizzled ，方便对比是否产生Bank Conflict时的性能。

```c++
template <class TensorS, class TensorD, class SmemLayoutS, class ThreadLayoutS,
          class SmemLayoutD, class ThreadLayoutD>
__global__ static void __launch_bounds__(256, 1)
    transposeKernelSmem(TensorS const S, TensorD const D,
                        SmemLayoutS const smemLayoutS, ThreadLayoutS const tS,
                        SmemLayoutD const smemLayoutD, ThreadLayoutD const tD) {
  using namespace cute;
  using Element = typename TensorS::value_type;

  // Use Shared Storage structure to allocate aligned SMEM addresses.
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorageTranspose<Element, SmemLayoutD>;
  SharedStorage &shared_storage =
      *reinterpret_cast<SharedStorage *>(shared_memory);

  // two different views of smem
  Tensor sS = make_tensor(make_smem_ptr(shared_storage.smem.data()),
                          smemLayoutS); // (bM, bN)
  Tensor sD = make_tensor(make_smem_ptr(shared_storage.smem.data()),
                          smemLayoutD); // (bN, bM)

  Tensor gS = S(make_coord(_, _), blockIdx.x, blockIdx.y); // (bM, bN)
  Tensor gD = D(make_coord(_, _), blockIdx.y, blockIdx.x); // (bN, bM)

  Tensor tSgS = local_partition(gS, tS, threadIdx.x); // (ThrValM, ThrValN)
  Tensor tSsS = local_partition(sS, tS, threadIdx.x); // (ThrValM, ThrValN)
  Tensor tDgD = local_partition(gD, tD, threadIdx.x);
  Tensor tDsD = local_partition(sD, tD, threadIdx.x);

  cute::copy(tSgS, tSsS); // LDGSTS

  cp_async_fence();
  cp_async_wait<0>();
  __syncthreads();

  cute::copy(tDsD, tDgD);
}

template <typename Element, bool isSwizzled = true> void transpose_smem(TransposeParams<Element> params) {

  using namespace cute;

  //
  // Make tensors
  //
  auto tensor_shape = make_shape(params.M, params.N);
  auto tensor_shape_trans = make_shape(params.N, params.M);
  auto gmemLayoutS = make_layout(tensor_shape, LayoutRight{});
  auto gmemLayoutD = make_layout(tensor_shape_trans, LayoutRight{});
  Tensor tensor_S = make_tensor(make_gmem_ptr(params.input), gmemLayoutS);
  Tensor tensor_D = make_tensor(make_gmem_ptr(params.output), gmemLayoutD);

  //
  // Tile tensors
  //

  using bM = Int<64>;
  using bN = Int<64>;

  auto block_shape = make_shape(bM{}, bN{});       // (bM, bN)
  auto block_shape_trans = make_shape(bN{}, bM{}); // (bN, bM)

  Tensor tiled_tensor_S =
      tiled_divide(tensor_S, block_shape); // ((bM, bN), m', n')
  Tensor tiled_tensor_D =
      tiled_divide(tensor_D, block_shape_trans); // ((bN, bM), n', m')

  auto tileShapeS = make_layout(block_shape, LayoutRight{});
  auto tileShapeD = make_layout(block_shape_trans, LayoutRight{});

  auto smemLayoutS = tileShapeS;
  auto smemLayoutD = composition(smemLayoutS, tileShapeD);
  auto smemLayoutS_swizzle = composition(Swizzle<5, 0, 5>{}, tileShapeS);
  auto smemLayoutD_swizzle = composition(smemLayoutS_swizzle, tileShapeD);

  auto threadLayoutS =
      make_layout(make_shape(Int<8>{}, Int<32>{}), LayoutRight{});
  auto threadLayoutD =
      make_layout(make_shape(Int<8>{}, Int<32>{}), LayoutRight{});

  size_t smem_size = int(
      sizeof(SharedStorageTranspose<Element, decltype(smemLayoutS_swizzle)>));

  //
  // Determine grid and block dimensions
  //

  dim3 gridDim(
      size<1>(tiled_tensor_S),
      size<2>(tiled_tensor_S)); // Grid shape corresponds to modes m' and n'
  dim3 blockDim(size(threadLayoutS)); // 256 threads

  if constexpr (isSwizzled) {
    transposeKernelSmem<<<gridDim, blockDim, smem_size>>>(
        tiled_tensor_S, tiled_tensor_D, smemLayoutS_swizzle, threadLayoutS,
        smemLayoutD_swizzle, threadLayoutD);
  } else {
    transposeKernelSmem<<<gridDim, blockDim, smem_size>>>(
        tiled_tensor_S, tiled_tensor_D, smemLayoutS, threadLayoutS,
        smemLayoutD, threadLayoutD);
  }
}
```


## TMA: 

注意，GMEM和SMEM之间的数据传输占用了我们转置kernel中绝大部分时间。张量内存加速器（TMA）是NVIDIA Hopper™架构中引入的一项特性，可以用来替代GMEM和SMEM之间的常规加载和存储指令，从而可能提高我们转置kernel的性能。我们研究了本教程中TMA的使用情况，并得到了一些混合结果，我们将在本节中描述。

回顾一下，TMA是一个专用的异步内存复制单元，用于在GMEM和SMEM之间复制多维数据。在TMA的异步复制模型中，不是让CTA中的线程/线程束协作复制源张量的一部分到目标张量，而是选择CTA中的单个线程来发出加载或存储TMA指令。虽然指令在异步代理中执行，但线程可以自由地执行其他独立工作。使用屏障对象和同步原语（fence、arrive和wait）来同步依赖于数据的计算与数据移动。当与软件流水线(https://github.com/NVIDIA/cutlass/blob/main/test/unit/pipeline/pipeline_tma_async_warp_specialized.cu)方案结合使用时，TMA允许内存复制指令与计算重叠，这有助于隐藏延迟。然而，由于转置kernel只进行内存复制，我们在本教程中没有机会展示TMA的这一优势。

为了明确TMA在单纯内存复制中的性能，我们首先研究了TMA加载和存储复制kernel与其他替代方案的性能，如CuTe的TiledCopy教程（https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/tiled_copy.cu） ，该教程执行128位向量化的加载和存储，仅通过RMEM传递。我们发现在这种情况下，TMA的性能与这种更简单的替代方案（经过两者的tile大小tuning后）相当，都达到了设备的内存带宽规格。这个结果符合我们的预期——事实上，我们没有理由期望TMA在纯内存复制的情况下表现更好。

相比之下，在转置kernel中同时使用TMA进行加载和存储，使用与上述相同的tile大小，其性能比我们性能最佳的版本要差。这是由于存在Bank Conflict！直接的问题是TMA只支持有限的swizzle函数集（旨在与WGMMA一起使用）；例如，请参见CuTe代码库中的这个部分（https://github.com/NVIDIA/cutlass/blob/033d9efd2db0bbbcf3b3b0650acde6c472f3948e/include/cute/atom/copy_traits_sm90_tma_swizzle.hpp#L48-L62）。 特别是，它不支持我们上面使用的Swizzle<5,0,5>函数，这使得完全消除Bank Conflict变得不那么直接。但请注意，我们没有理由认为这是一个本质问题，只是我们选择不在复制kernel的基准测试背景下进一步研究这条线。此外，当尝试一个版本，TMA仅用于128位向量化加载到寄存器然后写入SMEM时，我们发现其性能仅略低于标准版本，尽管profiler 仍报告了shared store Bank Conflict（但避免了TMA从SMEM到GMEM存储的Bank Conflict）。

由于这些混合结果，我们不详细描述如何使用TMA的机制，而将此推迟到未来的博客文章，我们将在更适合其优势的背景下研究TMA。


这一节的实验代码见：https://github.com/ColfaxResearch/cfx-article-src/blob/master/transpose-cute/include/transpose_tmastore_vectorized.h

## 结论

在本教程中，我们向读者介绍了一些基本的GPU内存概念，以及如何使用CuTe库通过实现高效的矩阵转置kernel来编程这些概念。

从合并读取和写入开始，我们涉及了CuTe布局和张量、Bank Conflict、swizzle函数和TMA的概念。除了TMA之外，我们已经看到了对这些概念的良好理解对实现高效的转置kernel是必要的。在后续的文章中，我们计划在一个对优化重要的场景中研究TMA。

为了总结本教程，我们展示了我们讨论的各种kernel的运行时间。我们包括了`JustCopy` kernel作为可达到的性能上限，以及一个朴素的PyTorch实现（通过在`torch.transpose上`调用`contiguous()`）和一个使用`torch.compile`的实现，以展示通过编写这些low-level kernel可获得的效率提升的规模。

所有这些kernel的源代码以及基准测试脚本可在Colfax Research GitHub仓库(https://github.com/ColfaxResearch/cfx-article-src/tree/master/transpose-cute)中获得。

![](https://files.mdnice.com/user/59/11b25161-1bf4-4cda-94d2-aa43ba2579a9.png)


