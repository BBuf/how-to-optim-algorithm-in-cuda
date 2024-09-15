> blog地址：https://pytorch.org/blog/accelerating-triton/
> triton kernel 地址：https://github.com/foundation-model-stack/foundation-model-stack/blob/triton/triton/kernels/gptq/splitk_dequant_gemm.py#L51

# 加速 GPTQ 的 Triton Dequantization Kernel

## 太长不看版

利用第一性原理方法，我们展示了一个逐步过程，旨在将当前的Triton GPTQ kernel加速3倍（core GPTQ）和6倍（AutoGPTQ）。例如：在典型的Llama风格推理输入上，将处理时间从275微秒降低到47微秒。我们的目标是提供一个有用的模板，用于加速任何给定的Triton kernel。我们提供了Triton和GPTQ量化及反量化过程的背景信息，展示了合并内存访问对改善共享和全局内存吞吐量的影响，强调了为减少线程束停滞以提高总体吞吐量所做的更改，并概述了将Triton kernel集成到PyTorch代码中的方法。长期来看，我们希望我们的Triton kernel能够超越现有的CUDA原生GPTQ kernel。

![图1：在H100上对优化后的AutoGPTQ kernel与当前AutoGPTQ kernel进行性能基准测试](https://files.mdnice.com/user/59/e2dbf951-576e-4735-8303-88cf160f5bb6.png)

![图2：在A100上对新优化的AutoGPTQ内核与当前AutoGPTQ内核进行性能基准测试](https://files.mdnice.com/user/59/7ae34ac2-c23b-4624-ad9c-00dbb0bcd98a.png)

![图3：即使有这些改进，我们优化的Triton内核与A100上的CUDA原生AutoGPTQ内核之间仍然存在差距。更多进展即将到来...](https://files.mdnice.com/user/59/13b0734f-465f-4fbc-8ad6-2870913848b2.png)

## 1.0 对Triton的介绍

Triton框架提供了一种硬件无关的方式对GPU编程，目前支持NVIDIA和AMD，并正在进行对其他硬件供应商的支持。Triton现已成为PyTorch 2.0的主要组成部分，torch.compile将eager PyTorch代码分解并重新组装成大比例的Triton kernel，并配以PyTorch连接代码。

随着Triton被更广泛地采用，程序员需要了解如何系统地遍历Triton堆栈（从高层Python到底层SASS），以解决性能瓶颈，达到比torch.compile生成的Triton kernel更快。

在这篇文章中，我们将介绍Triton编程语言的一些核心概念，如何识别GPU kernel中常见的性能限制因素，并同时调优一个用于AutoGPTQ的量化kernel，该kernel可用于高吞吐量推理应用。

### GPTQ量化和反量化简介

GPTQ（https://arxiv.org/abs/2210.17323） 是一种量化算法，能够通过近似二阶信息（Hessian逆矩阵）有效地将超大型（175B+）LLM压缩为4位整数表示。AutoGPTQ (https://github.com/AutoGPTQ/AutoGPTQ) 是一个建立在GPTQ基础上的框架，允许快速反量化并推理或者服务使用GPTQ量化的LLM。

作为AutoGPTQ堆栈的一部分，他们提供了一个Triton GPTQ kernel来处理模型推理时的反量化。

INT量化的基本过程如下所示，涉及确定比例和零点，然后使用比例和零点计算量化的4位权重：

![](https://files.mdnice.com/user/59/4f70c4bd-83f8-4f05-9fce-fd585faecffa.png)

我们因此存储4位权重以及每组权重的比例和零点元信息。

要"反量化"这些权重，我们执行以下操作：

![](https://files.mdnice.com/user/59/884a7179-2770-4c74-86bc-6dda5250eba6.png)

然后进行**矩阵乘法**，将反量化的权重与该线性层的dense输入特征矩阵相乘。

## 2.0 识别瓶颈 - 优化矩阵乘法

事实证明，制作一个快速的矩阵乘法 kernel 并不简单。简单实现的矩阵乘法很少能在高度并行的机器（如GPU）上达到峰值吞吐量性能。因此，我们需要以分层方式处理GPU中的计算和内存子系统，以确保最大限度地利用每种资源。

我们通过运行未优化的Triton kernel，使用Nvidia Nsight Compute工具开始优化过程，并记录一些重要的指标和警告：

![图表显示GPU吞吐量，包括计算(SM)和内存使用百分比](https://files.mdnice.com/user/59/e9c0b976-22b5-4d78-ab13-d31ebdbf4f2c.png)

![nsight compute 的未合并的共享访问警告](https://files.mdnice.com/user/59/c3320ce7-3c8d-4df1-abba-76eee193981c.png)


我们首先注意到计算和内存吞吐量都很低，分别为7.40%和21.19%。考虑到典型的推理矩阵问题规模，我们处于内存受限的状态，我们将尝试通过应用针对A100 GPU内存子系统的代码更改来优化kernel。

这篇文章将涵盖三个主题：

1. L2 优化
2. 向量化加载
3. 线程束停滞

让我们逐一讨论每个主题，进行适当的更改，并观察其对我们Triton kernel的相应影响。这个Triton kernel是一个fuse反量化kernel，它将Packed的int32权重（我们将其称为B矩阵，一个INT32权重对应了8个INT4的权重）反量化为FP16数据类型，以FP16模式与激活张量（称为A矩阵）进行矩阵乘法，然后将结果存储回矩阵C。

> 我们在 https://github.com/foundation-model-stack/foundation-model-stack/blob/triton/triton/kernels/gptq/splitk_dequant_gemm.py#L51 这里看到的 // 8就是把 INT32 权重对应到INT4权重上。

上述过程被称为W4A16量化。请记住，我们描述的过程可以且应该用于开发任何GPU kernel，因为这些是任何未优化kernel中常见的瓶颈。

## 3.0 L2 优化

这种优化已经存在于AutoGPTQ kernel中，但我们想专门讨论一下，以帮助读者更好地理解Triton中线程块的映射和执行顺序是如何处理的。因此，我们将逐步介绍一个朴素映射，然后是一个更优化的映射，以观察其相应的影响。

让我们从朴素地构建我们kernel开始，首先是从全局内存进行"线性"加载，然后将其与更优化的"swizzled"加载进行比较。线性与交错决定了我们在GPU上工作网格的执行顺序。让我们看看Nvidia Nsight Compute工具在朴素情况下提供的关于我们kernel共享内存访问模式的提示：

![未合并的共享访问，估计本地加速：30.83%；该kernel有未合并的共享访问，导致总共524288个过多wavefronts（占总wavefronts 1572864的33%）。查看L1 Wavefronts Shared Excessive表以了解主要源码位置。《CUDA最佳实践指南》中有一个优化共享内存访问的示例。](https://files.mdnice.com/user/59/f5f33a50-d0b6-4b4c-865c-3b7d88cf8735.png)


为了解决这个问题，我们可以使用一种称为"tile-swizzling"的方法。这种方法的想法是以更友好于L2缓存的顺序启动我们的线程块。

让我们后退一步，熟悉一下Triton的一些语义，并做一个简单的CUDA类比来更好地理解这个概念。Triton kernel启动"程序"。这些所谓的程序映射到CUDA中的线程块概念，它是Triton kernel中并行性的基本单位。每个程序都有一个关联的"pid"，程序中的所有线程都保证执行相同的指令。

如果您对"pid"进行简单的线性映射到输出矩阵C的2D网格位置，Triton程序将以朴素的方式分布到您的SM上。

这个2D网格位置在Triton中由pid_m和pid_n确定。我们希望在分配我们的工作网格时利用GPU的L2缓存中的数据和缓存局部性。为此，我们可以在Triton中进行以下更改：

![](https://files.mdnice.com/user/59/c0386854-e858-4957-8228-0127d129a34e.png)

红色高亮的代码是朴素的"线性"tile排序，绿色高亮的代码是"交错"tile排序。这种启动方式改善了局部性。这里有一个可视化图帮助来更好地理解这一点。

![](https://files.mdnice.com/user/59/e2779623-74b2-4ca3-90c2-ba8d8f1c07ae.png)

在合并这个更改后，ncu profiler不再抱怨未合并的内存访问。让我们看看我们的内存吞吐量如何变化：

![图表显示了内存吞吐量的变化，交错方法达到58.22%，增加了112.07%](https://files.mdnice.com/user/59/ce021adc-0c6e-4feb-a3d6-5cc1bc43267d.png)

这个更改在一个简单的加载存储 kernel 上进行了测试。查看分析器中的GPU速度统计部分，我们还看到简单加载kernel的内存吞吐量增加了112.07%，这正是我们通过此优化想要达到的目标。

再次强调，这种优化已经存在于AutoGPTQ kernel中，但它是每个Triton kernel程序员在开始编写任何令人兴奋的反量化或矩阵乘法逻辑之前，都必须在kernel开头编写的样板逻辑。因此，重要的是要理解：

- 这种映射并不是唯一的
- Triton不会自动为程序员处理这种优化，必须仔细考虑以确保您的kernel以最佳方式处理共享内存访问

对于那些刚接触Triton的人来说，这些并不明显，因为大部分共享内存访问优化都是由Triton编译器处理的。然而，在编译器无法处理的情况下，重要的是要能够理解我们可以使用哪些工具和方法来影响内存行为。

## 4.0 向量化加载

现在，回到我们未优化 kernel 的原始问题。我们想要优化 kernel 的全局内存访问模式。从Nvidia Nsight计算工具的详细页面中，我们看到以下注释，其中分析器抱怨未合并的全局内存访问。让我们深入研究一下未优化内存读取的SASS（汇编）代码加载：

![图片显示了未优化的内存读取代码和相应的SASS指令](https://files.mdnice.com/user/59/030760d4-8d0c-4cc1-8e57-25a55fdcbc69.png)

这个加载操作导致了32个16位宽的全局加载操作，这并不理想。

我们希望以向量化的方式进行全局内存加载，以使其产生最少的加载指令。为了解决这个问题，我们可以给Triton编译器一些帮助。

![代码块显示了优化后的内存加载代码](https://files.mdnice.com/user/59/7b6e65fc-a53a-4191-b9f4-d3699010d73f.png)

上面绿色高亮的行作为编译器提示。它告诉编译器这些元素在内存中是连续的，并且这个加载操作可以被合并。

让我们看看添加这些行后在汇编中的效果。

![图片显示了优化后的汇编代码](https://files.mdnice.com/user/59/3a526b2c-fcdc-4c95-9887-48e0525d68f1.png)

加载现在通过4个每个128位宽的全局加载操作来执行，而不是32个16位的全局加载操作。这意味着减少了28个内存获取指令，更重要的是实现了合并的内存访问。这可以从单个线程不再访问连续内存地址的事实中看出，而没有编译器提示时则是这种行为。

在隔离的加载操作中，结果是73倍的加速，在将其整合到完整的反量化kernel中后，我们能够看到另外6%的加速。这是朝着正确方向迈出的又一步！

## 5.0 线程束停滞

![](https://files.mdnice.com/user/59/f6ec8f60-9d62-4935-94ce-e87a659ef473.png)

现在将所有更改重新应用到我们的完整反量化kernel中，我们看到以下性能限制因素：线程束停滞。

这些线程束停滞主要由"长记分板"停滞引起，占总数的92.63%。

从高层次来看，长记分板停滞（https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference） 发生在线程束为了进入issued(发射)状态必须的数据还没准备好的时候。换句话说，GPU是吞吐量机器，我们需要用计算指令来隐藏加载指令的延迟。通过加载更多数据并重新安排脚本中加载指令的位置，我们可以解决这个问题。

在理想情况下，每个线程束调度器能够每个时钟周期发出1条指令。注意 - A100 GPU上的每个SM有4个线程束调度器。

然而 - 我们的kernel有瓶颈，在AutoGPTQ Triton kernel认为最优的块大小下，在停滞状态下花费了4.4个周期。

我们如何改进这一点？

我们希望能够增加我们的内存吞吐量，这样当线程束发出指令时，我们不会等待加载被存储到SRAM中以便用于计算。我们尝试了多个参数（如流水线阶段数和线程束数），其中对k维度的块大小增加2倍产生了最大影响。

这些更改对计算和内存吞吐量都产生了直接影响。

![图表显示了性能改进的百分比](https://files.mdnice.com/user/59/53d60c7a-e5c1-45bf-8025-24dfb6806eef.png)


我们还看到在我们shift和scale量化权重的步骤中，长记分板等待时间显著下降，这是我们在源代码中识别的原始瓶颈。虽然在这一行仍然存在停滞，但只有68%是由长记分板停滞引起的，相比之前的92%。理想情况下，我们不希望观察到任何停滞，所以这里还有工作要做，但长记分板停滞数量的减少告诉我们，此时数据在指令执行时以更高的频率准备好被使用（在L1TEX内存中）。

![最后一张图片显示了更多代码和性能统计](https://files.mdnice.com/user/59/532863a5-59a9-4b0f-9e3d-f6c854b2c6cd.png)

相应的影响是我们kernel执行时间的1.4倍加速。

## 6.0 结果

通过有条不紊地解决所有这些问题，我们的最终 kernel 在Nvidia A100 GPU上的速度比AutoGPTQ提供的开箱即用的Triton kernel快6倍。

以相关的Llama推理样本数据点为例，我们开发的Triton kernel(https://github.com/foundation-model-stack/foundation-model-stack/blob/triton/triton/kernels/gptq/splitk_dequant_gemm.py)进行反量化和矩阵乘法只需47微秒，而AutoGPTQ kernel对相同大小的矩阵需要275微秒。

通过复制这种逐步方法，应该可以在其他kernel中获得类似的加速，并帮助理解常见的GPU瓶颈以及如何解决它们。

需要注意的是，虽然在改进AutoGPTQ Triton kernel的性能方面取得了进展，但我们仍然没有缩小与AutoGPTQ中当前的exllamaV2 CUDA内 kernel差距。

需要进行更多研究以了解如何进一步优化这个kernel以匹配等效的自定义CUDA kernel性能。

### 总结和未来工作

Triton通过允许在比CUDA编程更高的抽象层次上进行低级GPU优化来扩展PyTorch，其最终结果是添加优化的Triton kernel可以帮助PyTorch模型运行得更快。

我们在这篇文章中的目标是展示加速GPTQ 反量化kernel的一个例子，并提供一个模板工作流程，说明如何实现这些加速。

对于未来的工作，我们将研究矩阵乘法的SplitK工作分解作为潜在的加速方法。

### 将自定义Triton kernel集成到PyTorch中

鉴于上述显示的加速，一个常见问题是如何在给定的PyTorch代码库中实际使用自定义kernel。

Triton kernel将至少包含两个部分 - 实际的Triton kernel代码（将由Triton编译器编译）：

![](https://files.mdnice.com/user/59/6dcc0c08-a4fb-4c0c-81de-f26b5b7fc986.png)

除了实际的kernel代码之外，还有一个Python wrapper，它可能会也可能不会继承PyTorch autograd类 - 这取决于它是否要支持反向传播（即用于训练目的还是仅用于推理目的）。

你只需将Python类导入到你的PyTorch代码中，在你想使用它的地方就像使用任何其他Python/PyTorch函数一样。

![](https://files.mdnice.com/user/59/ad814fe0-5d97-44b1-9647-371a58dcff64.png)

在这种情况下，只需导入然后使用'fast_qlinear'就会调用底层的Triton kernel，将我们上面展示的加速应用到你的PyTorch模型中。

### 致谢

感谢IBM研究院的Jamie Yang和Hao Yu在收集这些结果时提供的技术指导。

