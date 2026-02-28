> 博客原地址：https://research.colfax-intl.com/tutorial-hopper-tma/
> 博客对应的完整代码：https://github.com/ColfaxResearch/cfx-article-src/tree/master/tma

# CUTLASS 教程：掌握 NVIDIA® 张量内存加速器 (TMA)

TMA（张量内存加速器）是NVIDIA Hopper™架构中引入的一项新功能，用于在GPU的全局内存（GMEM）和其线程块（即CTA）的共享内存（SMEM）之间进行异步内存复制。与之前的方法相比，TMA提供了许多优势，例如：(1) 通过异步促进专用线程束(https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md#warp-specialization)的kernel调度，从而提高GPU利用率；(2) 通过TMA复制描述符以单线程方式处理辅助复制数据（如地址和步长）的计算，这种方式既更节省寄存器，又能必要地处理谓词（如边界检查）。NVIDIA的技术博客(https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)和Hopper调优指南(https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#tensor-memory-accelerator)很好地阐述了这些优势，我们强烈建议读者阅读这些资料以理解TMA设计背后的原理。

与那些资源不同，本博文专注于如何编写使用TMA的kernel，以达到操作性理解。在整个过程中，我们依赖CuTe库，该库通过包装底层GPU指令的API来暴露TMA。这些指令包括PTX指令`cp.async.bulk.tensor`和`cp.reduce.async.bulk.tensor`，以及`cuTensorMap`操作数，我们也将在本文中讨论这些内容。

我们将本博文组织成三个主要部分：第一部分是关于TMA load，第二部分是关于TMA save，最后第三部分涵盖了更高级的操作，如TMA load reduce 和 TMA save Multicast。本质上，TMA load将数据从GPU的GMEM复制（"加载"）到其CTA的SMEM中，而TMA save则将数据从CTA的SMEM复制（"存储"）到GPU的GMEM中。由于TMA load、TMA save和更高级的变体共享许多概念，我们将在TMA load部分介绍大部分必要概念，然后在后续部分只关注剩余的差异。

此外，鉴于TMA是一种异步操作（在异步代理中执行），我们需要使用某些内存一致性强制工具，如异步内存屏障（即`mbarrier`）和异步内存栅栏（即`fence.proxy.async`），以确保 kernel 的正确行为。同步本身就是一个广泛的讨论主题，所以我们只会在实际使用所需的程度上涵盖这些概念。

最后，对于寻找不涉及CUTLASS或CuTe概念但涵盖许多相同要点的资源的读者，我们推荐CUDA®编程指南中关于TMA的部分(https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#tensor-memory-access)。

## TMA Load

TMA load将数据从GMEM复制到SMEM。在本节中，我们将演示如何编写使用TMA load来实现此目标的kernel。使用TMA load的kernel与使用其他内存复制方法的kernel有很大不同，因此我们首先将展示如何为一个简单的示例任务编写这样的kernel。然后，我们将解释涉及的概念。

### 示例任务

为了演示TMA load的使用，我们考虑一个简单的任务：对2D行主序矩阵进行分块。给定一个形状为`[m,n]`的矩阵A和两个正整数`CTA_M`和`CTA_N`。注意，`CTA_M`和`CTA_N`在编译时已知，而`m`和`n`是在运行时通过矩阵A给出的。为简单起见，我们还假设`m % CTA_M == n % CTA_N == 0`，尽管我们稍后会看到这个要求可以放宽。

我们启动一个大小为`{m/CTA_M, n/CTA_N, 1}`的CTA grid，其中第`(i,j)`个CTA的SMEM保存来自A的形状为`[CTA_M, CTA_N]`的第`(i,j)`个分块。我们可以用numpy伪代码来描述这个分配：

```python
A = np.random.uniform(M, N)
for i in range(M):
  for j in range(N):
    cta_i_j = A.reshape(M // CTA_M, CTA_M, N // CTA_N, N)[i, :, j, :]
```

**两阶段过程**。为了执行此任务，我们使用TMA load。在CuTe中，TMA load操作分两步实现。第一步是在主机代码中构建TMA复制描述符，而第二步是在kernel代码中使用此描述符执行实际的TMA load。注意，这个两步过程与我们通常使用CuTe的TiledCopy的方式不同在：TiledCopy所有复制步骤都写在kernel代码中——如教程(https://github.com/NVIDIA/cutlass/blob/637b15906358191cb4238af419d408a65819d7ec/examples/cute/tutorial/tiled_copy.cu#L120-L124)所示。


### Host Code

在主机端，我们创建三个对象：我们从中复制的GMEM张量、我们复制到的每个CTA上的SMEM张量的布局，以及一个以这两者为参数的`tma_load`对象。注意，由于我们在主机端创建SMEM布局，所有CTA将共享相同的SMEM布局以用于TMA load。一旦我们有了这些对象，它们就可以被传递到设备上的kernel中，在kernel中调用TMA load操作。

主机端的整个代码块如下：

```c++
template <typename T, int CTA_M, int CTA_N>
void host_fn(T* data, int M, int N) {
  using namespace cute;
 
  // create the GMEM tensor
  auto gmem_layout = make_layout(make_shape(M, N), LayoutRight{});
  auto gmem_tensor = make_tensor(make_gmem_ptr(T), gmem_layout);
 
  // create the SMEM layout
  auto smem_layout = make_layout(make_shape(CTA_M, CTA_N), LayoutRight{});
 
  // create the TMA object
  auto tma_load = make_tma_copy(SM90_TMA_LOAD{}, gmem_tensor, smem_layout);
 
  // invoke the kernel
  tma_load_kernel<CTA_M, CTA_N>
                 <<<1, dim3{M / CTA_M, N / CTA_N, 1}>>>
                 (tma_load, gmem_tensor, smem_layout);
}
```

创建gmem_layout、gmem_tensor和smem_tensor的代码行仅使用了基本的CuTE概念，所以我们建议读者参考这些CuTe教程(https://github.com/NVIDIA/cutlass/blob/637b15906358191cb4238af419d408a65819d7ec/media/docs/cute/01_layout.md, https://github.com/NVIDIA/cutlass/blob/637b15906358191cb4238af419d408a65819d7ec/media/docs/cute/02_layout_algebra.md, https://github.com/NVIDIA/cutlass/blob/637b15906358191cb4238af419d408a65819d7ec/media/docs/cute/03_tensor.md)来复习记忆。在这里，我们专注于解释`tma_load`对象。这个对象是cute::TiledCopy的一个实例，它包含了执行CTA范围内复制操作的信息和实现方法。在代码片段中，`tma_load`对象是通过`cute::make_tma_copy`函数的这个显式默认值创建的。这个函数的完整实现有一些细微差别，我们将在稍后讨论`MULTICAST`时深入探讨，但对于大多数用例（如我们的示例任务）来说，显式默认值就足够了。我们建议使用显式默认值以避免不必要的复杂性（和错误）。

让我们看看我们用于`make_tma_copy`的签名：

- 它的最后两个参数是`gmem_tensor`和`smem_layout`。在底层，`make_tma_copy`使用这些信息创建一个`TmaDescriptor`，这只是`CUtensorMap`的一个别名(https://github.com/NVIDIA/cutlass/blob/637b15906358191cb4238af419d408a65819d7ec/include/cute/arch/copy_sm90_desc.hpp#L178)。这个描述符对象在TMA kernel中使用。
- 它的第一个参数是SM90_TMA_LOAD(https://github.com/NVIDIA/cutlass/blob/637b15906358191cb4238af419d408a65819d7ec/include/cute/arch/copy_sm90_tma.hpp#L269)的一个实例。这个对象将复制操作分派到所需的`cp.async.bulk.tensor` PTX调用，我们将在下面的第三部分中深入探讨。

### Kernel code

相关的 kernel 代码片段如下所示。这些代码行包含了许多重要的TMA概念，我们将在下面进行解释。

![](https://files.mdnice.com/user/59/811e099f-b702-4403-bd09-301d866c5c4e.png)

首先，在第2行，kernel的tma_load参数必须用`__grid_constant__ const`注解。如果我们有两个要从GMEM复制到SMEM的张量，每个张量都必须有自己的`TiledCopy`实例，并且每个实例都必须是`__grid_constant__ const`。这是从主机传递`cuTensorMap`到设备的要求，例如在这里有文档(https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-data-copies-using-tensor-memory-access-tma)说明。

下一个重要点是，对于TMA Copy，只有一个线程负责发出TMA操作。在代码片段中，所有与TMA相关的变量和指令都包含在从第12行开始的if块中，该块仅由线程0执行。另一方面，第30行包含一条指令，让CTA中的所有线程等待TMA操作完成。

**坐标和算术元组**

现在，让我们看看TMA load逻辑。这从第13行开始，我们创建一个`gmem_tensor_coord`对象，它保存要复制的GMEM张量的坐标。如果我们尝试以下操作：

```c++
if (cute::thread(0)) { cute::print(gmem_tensor_coord); }
```

那么我们会看到如下输出（对于M=N=1024）：

```c++
ArithTuple(_0,_0) o (1024,1024):(_1@1,_1@0)
```

对于熟悉CuTe中tiled copy工作方式的读者来说，第15-18行是不言自明的，其中GMEM张量被tiled成更小的partitions，每个CTA根据块坐标切片到tiled张量中以获得其GMEM视图。但是请注意，partitions适用于上述表示gmem_tensor坐标的ArithTuple，而不是gmem_tensor本身。特别是，ArithTuple被分成形状为`[CTA_M,CTA_N]`的块，然后每个CTA取其块。

如果我们使用`print_tensor`打印`gmem_tensor_coord_cta`，如下所示：

```c++
if (cute::block(7)) { cute::print_tensor(gmem_tensor_coord_cta); }
```

我们会看到如下输出：

```c++
ArithTuple(0,112) o (_16,_16):(_1@1,_1@0):
  (0,112)  (1,112)  (2,112)  (3,112)  (4,112)  (5,112)  (6,112)  (7,112)  (8,112)  (9,112)  (10,112)  (11,112)  (12,112)  (13,112)  (14,112)  (15,112)
  (0,113)  (1,113)  (2,113)  (3,113)  (4,113)  (5,113)  (6,113)  (7,113)  (8,113)  (9,113)  (10,113)  (11,113)  (12,113)  (13,113)  (14,113)  (15,113)
  // more lines
  (0,127)  (1,127)  (2,127)  (3,127)  (4,127)  (5,127)  (6,127)  (7,127)  (8,127)  (9,127)  (10,127)  (11,127)  (12,127)  (13,127)  (14,127)  (15,127)
```

这些数字是`gmem_tensor`中的坐标，其值将被复制到CTA 7的`smem_tensor`中。我们鼓励读者尝试运行这段代码片段，将`cute::block(7)`替换为其他索引，以理解不同的CTA从`gmem_tensor`的哪些坐标复制数据。

接下来，在第25-27行发出的复制操作本身具有TiledCopy操作的常见签名，其中源张量被partitions后的坐标所替代。

**Memory barrier**

我们省略了第20、22和30行，这些行都涉及SMEM中的`uint64_t`变量`tma_load_mbar`。这是我们用来同步TMA load 与 kernel 消费 load 到SMEM中的结果数据的其余部分的**异步事务屏障**。NVIDIA关于Hopper架构的技术博客(https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)中给出了这种屏障的高级描述。就我们的kernel而言，重要的点如下：

- 我们在第20行的共享内存中初始化mbarrier对象。CuTe方法`initialize_barrier`包装了PTX指令`mbarrier.init.shared.b64`，该指令需要一个额外的到达计数参数。在我们的上下文中，由于单个线程将启动TMA load，我们应该将到达计数设置为1。此外，mbarrier的起始阶段将始终设置为0。
- 我们在第22行同时执行arrive-on操作并为mbarrier对象设置预期的事务计数，使用CuTe方法`set_barrier_transaction_bytes`，它包装了PTX指令`mbarrier.arrive_expect_tx.shared::cta.b64`。事务计数设置为等于TMA load传输的字节数，我们在第4行计算这个值。
- 在第25-27行，复制指令（它分派到所需的`cp.async.bulk.tensor`类型）总是将其完成机制设置为`barrier::complete_tx::bytes`，并使用提供的mbarrier对象。
- 在第30行，我们在mbarrier对象上执行等待操作。注意，所有线程都在mbarrier上等待，这与只有线程0到达mbarrier形成对比，并且在`wait_barrier`之前调用`__syncthreads()`是必要的，以解决线程分歧。
这里，`wait_barrier`包装了PTX指令`mbarrier.try_wait.parity.shared::cta.b64`。`try_wait`限定符（与`test_wait`相对）表示等待是一个阻塞指令。`parity`限定符（其使用需要提供一个相位）表示线程睡眠直到mbarrier的那个相位翻转。因为这是初始化后首次使用mbarrier来跟踪完成，我们提供0作为相位。如果我们要进行另一次TMA load，我们就需要翻转相位以重用mbarrier。
总的来说，CUTLASS Pipeline APIs(https://github.com/NVIDIA/cutlass/blob/main/media/docs/pipeline.md)提供了一种更高级的方式来处理一系列TMA load时mbarrier对象的生命周期，就像在软件流水线(https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md#pipelining)方案中可能做的那样。
- 在`wait_barrier`之后，内存一致性模型为我们提供以下保证：TMA load对SMEM的写入对所有调用`wait_barrier`的线程（在我们的示例kernel中，是CTA中的所有线程）都是可见的。

**使用TMA的剩余TILES和步长要求**

在我们上面的例子中，我们假设`m%CTA_M==0`和`n%CTA_N==0`。然而，为了进行TMA load，我们可以完全摒弃这个假设。我们不需要自己处理从GMEM到SMEM load 余数块时的越界逻辑，TMA复制单元会必然地限制(https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/0y_predication.md)内存复制不读取越界数据。这与上述TMA load中使用特殊的"隐式"CuTe张量和`ArithTuple`一致 — 如果我们使用普通的CuTe张量，它们可能会被切片产生新的CuTe张量，可能包含指向GMEM的越界指针，这不可避免地会导致bug。

然而，对于TMA，需要记住GMEM张量本身的步长有一个重要要求，即**16字节边界要求**。正如人们所预期的，TMA不支持复制GMEM中任意步长的区域。相反，我们需要假设被复制的块有(i)一个连续的方向（步长为1），以及(ii)其他步长是16字节的倍数。这在CUTLASS代码库中得到了断言(https://github.com/NVIDIA/cutlass/blob/7d49e6c7e2f8896c47f586706e67e1fb215529dc/include/cute/atom/copy_traits_sm90_tma.hpp#L846)。

例如，对于我们的行主序GMEM浮点张量，形状为(m, n)，步长为(n, 1)，这就要求n%4==0。如果不满足这个条件，那么可以在调用kernel之前将输入张量填充到正确的大小。

## TMA Store

掌握了TMA load的基础知识后，由于这两种操作之间的诸多相似性，学习TMA store变得容易得多。与TMA load类似，实现TMA store也是一个两步过程：在主机上定义TMA复制描述符，然后在kernel中发出TMA store操作。

### 示例任务和代码

为了说明起见，让我们考虑TMA load的反向示例，即从多个CTA的SMEM复制到分区GMEM张量中的相应块。这里的一个区别是，我们将在复制到GMEM之前用一个简单的数字模式填充CTA中的SMEM块（否则，我们将复制未定义的值）。一个功能性的代码片段如下：

![](https://files.mdnice.com/user/59/e4ce0dc2-0a9f-4227-881c-8fa4f2d742cd.png)

主机代码看起来几乎与TMA load相同，除了对tma_store_kernel的调用。注意，我们安排每个CTA有CTA_M个线程。我们的示例中，每个CTA在SMEM中持有一个`[CTA_M,CTA_N]`的块，这样在第29-32行，线程i用值i填充第i行。

在 kernel 代码中，第39-49行的if块与tma_load_kernel中的if块相似。特别是，只有线程0发出TMA store操作。所有的张量分块逻辑在概念上是相同的。然而，复制方向是相反的：对于TMA store，`tma_store_per_cta.partition_S`方法应用于`smem_tensor`，而`tma_store_per_cta.partition_D`方法应用于GMEM张量的坐标。注意，坐标也表示为`ArithTuple`，类似于TMA load。


**内存栅栏**

TMA load和存储代码之间最重要的区别是，我们不再看到任何与TMA store一起使用的mbarrier对象。这是因为TMA store使用另一种机制来强制内存一致性：内存栅栏(memory fence)。

内存栅栏的目的是在执行线程在栅栏之前和之后请求的内存访问之间建立保证的顺序。在我们的示例中，我们需要确保第29-32行对SMEM的所有写入对线程0执行的TMA store是可见的。为此，在第35行我们有CuTe方法`tma_store_fence()`，它包装了PTX指令`fence.proxy.async.shared::cta`。

这个指令包含两个重要的限定符，描述了栅栏的效果：范围和代理类型。范围表示参与栅栏强制执行的顺序的线程集。在我们的例子中，限定符(https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#scope) cta将范围定义为CTA中的所有线程（这是内存一致性模型目的的最小可能范围）。代理类型(https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#proxies)表示除了通用代理外，将参与栅栏强制执行的顺序的代理类型。在我们的例子中，我们选择代理类型为`async.shared`，因为TMA store在异步代理中执行（相对于每个CTA）。如果我们用不涉及异步代理的其他内存栅栏原语（如`__threadfence_block()`）替换异步栅栏，我们将破坏kernel正确行为所需的保证，在实践中导致竞争条件。

**TMA store到达和等待**

在第49和51行，我们有`tma_store_arrive()`，它提交TMA store操作（技术上，作为`cp.async.bulk-group`），和`tma_store_wait<Count>()`，它等待直到最多`Count`个已提交的TMA store操作处于待处理状态（例如，如果所有操作都应完成，则将`Count`设置为0）。当kernel中有其他工作等待TMA store完成时，这些操作很有用——例如，这在写出后重用释放的SMEM时是必需的。然而，因为我们的kernel在TMA store完成后简单地退出，所以我们在这里不需要TMA store到达和等待模式，因此我们注释掉了这些行。

## 深入了解TMA操作

TMA load和TMA store操作对比表：

![TMA操作总结。](https://files.mdnice.com/user/59/2e8aa550-700e-4a5f-a55c-115119485404.png)


到目前为止，我们已经学习了如何调用TMA load和TMA store操作。上表比较和对比了这些操作。要调用任一操作，我们需要通过主机代码中的`cute::make_tma_copy`方法创建一个类似于TiledCopy的对象，然后将此对象传递到kernel函数中，在那里我们使用`cute::copy`来实际调用操作。在本节中，我们深入探讨当我们在kernel函数中调用这些TiledCopy对象时实际发生的情况。从这次深入探讨中，我们讨论两个扩展：TMA store归约和TMA load multicast。

### TMA load和存储的PTX指令

PTX（并行线程执行）是NVIDIA GPU的低级中间语言。就我们的讨论而言，PTX的相关部分包括一组可以通过`asm volatile`关键字包装的块插入CUDA代码的指令。特别是，当我们调用`cute::copy(tma_load, ...)`或`cute::copy(tma_store, ...)`时，如前几节所述，会调用某些PTX指令来执行这些操作。通过研究PTX，我们可以更好地理解TMA load和TMA store。

让我们从TMA load开始。回想一下，当我们在主机代码中创建`tma_load`对象时，我们必须提供GMEM张量（包含要复制的源数据）和SMEM布局（描述数据在每个CTA中的存储方式）。使用这个张量和布局，CuTe确定在kernel中调用`cute::copy(tma_load, ...)`时要执行的底层PTX指令。PTX指令的选择取决于GMEM张量的秩（注意，这里的秩指的是张量的维度数，而不是线性代数中的矩阵秩/零性）。在我们的例子中，GMEM张量的秩为二，所以将执行以下PTX指令(https://github.com/NVIDIA/cutlass/blob/637b15906358191cb4238af419d408a65819d7ec/include/cute/arch/copy_sm90_tma.hpp#L100-L106)：

```c++
// 使用内联汇编来执行TMA load操作
asm volatile (
  // PTX指令 "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
  // 该指令用于从全局内存（GMEM）异步 load 数据到共享内存（SMEM）
  // 其中 "2d" 表示二维张量，"shared::cluster" 表示目标是共享内存集群，
  // "global" 表示源数据在全局内存中，"mbarrier::complete_tx" 表示使用内存屏障完成传输，
  // "bytes" 表示传输的数据单ython位是字节
  "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
  // 指令的操作数部分
  " [%0], [%1, {%3, %4}], [%2];"
  :
  // 输出操作数为空
  :
  // 输入操作数
  // "r"(smem_int_ptr) - 共享内存指针，指向SMEM中数据的目标位置
  // "l"(gmem_int_desc) - 全局内存描述符，描述GMEM中数据的源位置
  // "r"(smem_int_mbar) - 内存屏障，确保数据传输的顺序性
  // "r"(crd0) - 坐标0，表示二维张量的第一个维度
  // "r"(crd1) - 坐标1，表示二维张量的第二个维度
  : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
    "r"(crd0), "r"(crd1)
  // "memory" - 表示该指令会修改内存，防止编译器对内存操作进行优化
  : "memory");
```

看这条PTX指令，我们看到了许多熟悉的概念。例如，`gmem_int_desc`指的是TMA描述符中保存的坐标，而`mbarrier::complete_tx::bytes`和`smem_int_mbar`则指的是内存屏障。还要注意，`tensor.2d`表示我们正在复制一个二阶张量，即一个2D矩阵。

事实证明，不仅是TMA load，所有TMA操作都是某些`cp.async.bulk`指令的包装。NVIDIA PTX文档专门用一整节来讨论`cp.async.bulk`指令，特别是它们的语法和操作数。我们鼓励读者阅读该节以及其中的参考资料，以更全面地研究TMA操作，这些操作涵盖的范围远比本博文所打算讨论的要广。在这里，我们将讨论通过这些`cp.async.bulk`指令暴露的TMA的两个扩展。

#### TMA Store Reduce

回想一下，TMA store操作将多个CTA的SMEM中的数据复制到GMEM张量的对应块中。我们可以将TMA store解释为以下Python伪代码所示的赋值操作：

```python
for cta_idx in range(number_of_ctas):
    gmem_dst[cta_idx] = smem_src[cta_idx]
```

如果我们想要执行以下操作呢？

```python
for cta_idx in range(number_of_ctas):
    gmem_dst[cta_idx] += smem_src[cta_idx]
    # 或者这个：
    gmem_dst[cta_idx] = max(gmem_dst[cta_idx], smem_src[cta_idx])
    # 或者：
    gmem_dst[cta_idx] = min(gmem_dst[cta_idx], smem_src[cta_idx])
```

所有这些操作——即归约求和、归约求最大值和归约求最小值——在张量程序中都相当常见。特别是，归约求和在Split-K GEMM中是不可避免的子程序，而归约求最大值和归约求最小值经常用于注意力机制。尽管这些操作看起来很简单，但在CUDA kernel中实现它们并不那么直接。我们邀请读者在阅读下一段之前，简要思考一下在GMEM和SMEM之间必须进行多少轮数据移动才能实现这些目标。

一个CTA的SMEM中的值"累积"到GMEM张量中一个块的归约操作的原始实现包括一次GMEM读取、一个块处理和一次GMEM写入。首先，从GMEM load 原始值到CTA的SMEM或寄存器中，然后执行归约操作，最后将结果写回。这个过程很慢。

对TMA storeTiledCopy对象的构造函数进行轻微修改，允许我们将这个三步过程浓缩为仅一条PTX指令，即使用`cp.reduce.async.bulk`而不是`cp.async.bulk`(https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-reduce-async-bulk)。具体来说，我们可以在主机代码上进行以下一行更改：

```c++
// original: create a TMA store object
auto tma_store = make_tma_copy(SM90_TMA_STORE{}, gmem_tensor, smem_layout);
 
// to create a TMA reduce sum object
auto tma_reduce_sum = make_tma_copy(SM90_TMA_REDUCE_ADD{}, gmem_tensor, smem_layout);
```

然后使用`tma_reduce_sum`代替`tma_store`，它现在在底层调用`cp.reduce.async.bulk`而不是`cp.async.bulk`。

顺便说一下，PTX指令`cp.reduce.async.bulk`自CUDA 12.0发布以来就已经可用，但直到CUTLASS 3.5发布才通过CUTLASS和CuTe暴露出来。我们希望其他归约操作将在未来的版本中公开，但如果没有，适应CuTe代码以执行TMA归约来执行最大值和最小值归约以及其他按位归约相当简单（`cp.reduce.async.bulk`提供：and、or、xor、inc和dec）。


#### TMA Load Multicast

在前一节中，我们看到研究PTX指令让我们发现了TMA归约操作，这些操作可以用于某些应用场景替代TMA store。在本节中，我们将研究TMA load的 multicast 扩展。

为了帮助理解，我们首先看一下`cp.async.bulk.tensor`的完整语法(https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor)：

```c++
// global -> shared::cluster:
cp.async.bulk.tensor.dim.dst.src{.load_mode}.completion_mechanism
{.multicast}{.level::cache_hint}
  [dstMem],                // 目标内存地址
  [tensorMap, tensorCoords], // 张量映射和坐标
  [mbar]                   // 内存屏障
  {, im2colOffsets}        // 可选: im2col偏移量
  {, ctaMask}              // 可选: CTA掩码
  {, cache-policy}         // 可选: 缓存策略
 
.dst =                  { .shared::cluster } // 目标是集群共享内存
.src =                  { .global }          // 源是全局内存
.dim =                  { .1d, .2d, .3d, .4d, .5d } // 支持的张量维度
.completion_mechanism = { .mbarrier::complete_tx::bytes } // 使用内存屏障完成传输
.load_mode =            { .tile, .im2col }   // 加载模式: 平铺或im2col
.level::cache_hint =    { .L2::cache_hint }  // L2缓存提示
.multicast =            { .multicast::cluster  } // 集群内Multicast
```

再次强调，我们不需要完全理解PTX指令的语法，我们可以看到许多熟悉的概念，如`.dim`, `.global`用于`src`, 和`.mbarrier`用于`completion_mechanism`。本节重点关注`multicast`操作数。

Multicast指的是我们有一个GMEM张量中的块，我们想将其复制到多个CTA中的多个SMEM位置的情况。这通常发生在GEMM kernel（即矩阵乘法）中，其中一个输入矩阵列块需要用于多个行块，反之亦然。在这种情况下，虽然TMA Load仍然完全可用——我们只需为需要它的多个CTA提供相同的TMA描述符——但`.multicast`操作数允许我们保证L2缓存命中。

让我们考虑将上述TMA Load示例扩展为包含Multicast。首先，我们需要定义kernel的集群维度为非平凡的，因为要求一组CTA共同参与TMA Load Multicast操作的条件是它们属于同一个（线程块）集群。为了保持简单，我们将只更改网格维度如下：

```c++
// old grid dimensions and implicit trivial cluster dimensions
dim3 grid_dims = dim3{M / CTA_M, N / CTA_N, 1};
dim3 cluster_dums = dim3{1, 1, 1};
 
// new grid dimensions and cluster dimensions
dim3 grid_dims = dim3{M / CTA_M, N / CTA_N, 2};
dim3 cluster_dums = dim3{1, 1, 2};
```

注意，在使用集群时，集群维度必须均匀地划分网格维度，否则kernel将无法启动。在我们的新kernel中，我们将安排同一个GMEM块load到同一集群中每对CTA的SMEM中，这种情况发生在且仅在两个CTA具有相同的blockIdx.x和blockIdx.y时。

首先，在主机代码中，我们对TMA Load `TiledCopy`对象的定义做如下更改：

```c++
// original: create a TMA load object
auto tma_load = make_tma_copy(SM90_TMA_LOAD{}, gmem_tensor, smem_layout);
 
// new: create a TMA load multicast object for the given cluster size
auto tma_load = make_tma_copy(SM90_TMA_LOAD_MULTICAST{},
      gmem_tensor, smem_layout, cute::_2{});
```

我们为最后一个参数（集群大小）写入`_2{}`，以将其作为编译时常量传递，使用为此目的提供的CuTe整数类型(https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/01_layout.md#integers)。在实践中，更习惯的做法是我们会预先定义`ClusterShape`类型（在我们的情况下为`Shape<_1,_1,_2>`），然后为该参数写入`size<2>ClusterShape{}`。

然后我们按如下方式更改kernel代码：

![](https://files.mdnice.com/user/59/af40581a-af75-4954-b696-2a76d2b586ea.png)

我们已经突出显示了相关的更改。首先，我们现在需要跟踪CTA在其集群内的内部索引，我们通过CuTe方法`block_rank_in_cluster()`获取。这会返回特殊寄存器`%cluster_ctarank`的值，在我们的示例中将取值0和1。为简洁起见，让我们将其称为`ctaid`。
然后我们对代码进行以下三项修改：

- 额外的集群同步原语。
- 在 Multicast 操作中使用uint16位掩码。
- 使用`ctaid`来确定`TiledCopy`对象的切片部分，用于划分GMEM和SMEM张量。

对于(1)，我们使用CuTe方法`cluster_sync()`，它依次执行集群屏障到达和等待操作。我们在两个地方插入这个：在第26-27行，我们使用`cluster_sync()`和一个栅栏来确保集群范围内mbarrier初始化的可见性，在第41行，我们使用另一个`cluster_sync()`来确保集群中的两个CTA不会在另一个仍在等待Multicast load完成时过早退出。通常，会对load到SMEM中的数据进行计算，最后一个`cluster_sync()`会出现在kernel代码的最后。

对于(2)，我们向复制操作传递一个uint16位掩码，以指定哪些CTA将参与TMA Multicast load。掩码中设置为1的位表示哪些CTA处于活动状态，一个集群中最多有16个CTA（最大不可移植大小, https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#thread-block-clusters），位的位置对应于ctaid。因此，在我们的示例中，通过将`tma_mcast_mask`设置为`0b11`，我们指定集群中的两个CTA都将参与。

最后，对于(3)，`ctaid`用于指定从给定CTA启动的TMA Multicast load操作时切片到GMEM的偏移量。为了清楚地解释这一点，考虑以下示例：从GMEM加载一个16 x 16的整数块到集群中两个CTA的SMEM中，该块以升序行主序初始化为0-255。假设我们错误地为两个CTA的`tma_load.get_slice()`给出了0作为参数。那么在加载完成后，我们在两个CTA的SMEM中得到以下结果：

![](https://files.mdnice.com/user/59/cffb4c09-8c0a-4f57-a10a-e503e00cbb9d.png)

相比之下，如果我们为两个CTA都给出1作为参数，那么我们在两个CTA的SMEM中得到这个：

![](https://files.mdnice.com/user/59/5a2d130c-63da-44d2-97a7-d8ae5109dbda.png)

最后，从`ctaid` 1给出0，从`ctaid` 0给出1，或者从`ctaid` 0给出0，从`ctaid` 1给出1，都会正确地将整个块加载到两个CTA的SMEM中。这些输出说明了从集群中的一个CTA发出Multicast操作会将GMEM的一半加载到两个CTA的SMEM中，TiledCopy的切片决定各自的一半。这与PTX文档中`cp.async.bulk.tensor`的Multicast描述一致：

> 源数据被Multicast到每个目标CTA的共享内存中相同的CTA相对偏移量dstMem。

就`TiledCopy`对象而言，它通常具有将线程-值元组映射到切片逻辑坐标的`TiledLayout_TV`布局，CuTe将ctaid视为切片目的的线程索引。例如，打印我们16 x 16示例中的TiledCopy会产生以下结果：

```c++
TiledCopy
  Tiler_MN:       (_16,_16)
  TiledLayout_TV: (_2,((_16,_16))):(_8,((_16,_1)))
Copy_Atom
  ThrID:        _1:_0
  ValLayoutSrc: (_1,_256):(_0,_1)
  ValLayoutDst: (_1,_256):(_0,_1)
  ValLayoutRef: (_1,_256):(_0,_1)
  ValueType:    32b
```

这有两个"线程"对应于集群中的两个CTA，ctaid 1的偏移位置由`(16,16)`切片中的逻辑坐标`(8,0)`给出。

## 结论

在这篇博文中，我们通过几个简化的例子，展示了如何使用CUTLASS库提供的方法，在CUDA kernel中利用TMA Load、TMA Store、TMA Store Reduce和TMA Load Multicast来执行GMEM和SMEM之间的内存复制。

我们首先概述了TMA，并介绍了用户如何在GPU kernel中调用这些操作。然后，我们深入研究了低级别PTX指令，以获得对TMA更深入的理解。我们希望这篇博文对想要了解TMA、复习相关知识或调试现有使用TMA项目的读者有所帮助。

我们省略了一些重要主题，如TMA支持的swizzling模式以及TMA将GMEM复制到SMEM时以interleaved格式排列的能力，即在连续维度之外置换步长。这些在使用TMA配合Warpgroup矩阵-乘法-累加（WGMMA）指令时很重要，WGMMA指令也是Hopper架构的新特性，用于以与WGMMA兼容的内存格式加载张量数据。我们将在未来讨论基于Hopper的GEMM的帖子中解释这些要点。

最后，本博文中讨论的kernel的完整示例可以在我们的Colfax Research GitHub仓库中找到(https://github.com/ColfaxResearch/cfx-article-src/tree/master/tma)。



