> 我的课程笔记，欢迎关注：https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode 
> 本篇文档的来源：https://github.com/stas00/ml-engineering 。这篇文档详细介绍了机器学习加速器的现状和技术细节，涵盖了从GPU、TPU到FPGA等多种加速器类型，强调了训练和推理的不同计算需求。文中分析了NVIDIA、AMD、Intel等厂商的高端加速器规格，特别是TFLOPS性能和内存带宽的重要性，并提供了不同加速器的比较表。文档还讨论了在云端和本地部署加速器的选择，以及如何通过优化硬件和软件配置来提高投资回报率。最后，文档提醒我们在购买或租用加速器时要注意电源和散热问题，以确保设备的稳定运行。

# 加速器

计算加速器是机器学习训练的主力。最初只有 GPU。但现在还有 TPU、IPU、FPGA、HPU、QPU、RDU 等，而且还在不断发明新的加速器。

机器学习有两种主要工作负载 - 训练和推理。还有微调工作负载，通常与训练相同，除非执行更轻量级的 LORA 风格(https://arxiv.org/abs/2106.09685)微调。后者比普通微调需要的资源和时间要少得多。

在语言模型的推理过程中，生成是按顺序进行的 - 一次生成一个 token。因此，它必须重复相同的 `forward` 调用数千次，每次执行一个小型 `matmul`（矩阵乘法或 GEMM）。这可以在加速器（如 GPU）上完成，或者在一些最新的可以高效处理推理的 CPU 上完成。

在训练过程中，整个序列长度在一个巨大的 `matmul` 操作中处理。因此，如果序列长度为 4k，同一模型的训练将需要一个计算单元，能够处理比推理多 4k 倍的操作，并且要快速完成。加速器在这项任务中表现出色。事实上，它们需要相乘的矩阵越大，计算效率就越高。

另一个计算上的区别是，虽然训练和推理在 `forward` 过程中都需要执行相同总量的 `matmul`，但在只用于训练的 `backward` 过程中，还需要额外执行 2 倍的 `matmul` 来计算输入和权重的梯度。如果使用激活重计算，还会额外执行一次 `forward`。因此，训练过程需要比推理多 3-4 倍的 `matmul`。

## 子章节

通用:
- 基准测试(https://github.com/stas00/ml-engineering/tree/master/compute/accelerator/benchmarks)

NVIDIA:
- NVIDIA GPU 故障排除(https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/nvidia/debug.md)

AMD:
- AMD GPU 故障排除(https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/amd/debug.md)
- AMD GPU 性能(https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/amd/performance.md)

## 高端加速器现状概览

虽然未来可能会发生变化,但与消费级 GPU 市场不同,截至本文撰写时,高端加速器的种类并不多。如果你在云端租用,大多数提供商提供的加速器选择都大同小异。

GPU:
- 截至今天,ML 云/HPC 已开始从 NVIDIA A100 过渡到 H100,由于 NVIDIA GPU 的常见短缺,这个过程将持续数月。H200 即将推出 - 承诺在 2024 年第四季度。B100、B200、GB200 在 2024 年第一季度宣布,但由于生产延迟,我们可能要到 2025 年中才能使用这些产品。
- AMD 的 MI250 开始零星出现,但目前还不清楚何时能够轻松获取。MI300X 现在开始在一些二线云提供商那里可用。

HPU:
- 英特尔的 Gaudi2 开始在英特尔云上缓慢出现 - 有一个庞大的阵容。它也可通过 Supermicro、WiWynn 等公司提供本地部署实施,不久后还会有更多选择。
- Gaudi3 预计将在 2024 年某个时候推出。

IPU:
- Graphcore 提供了他们的 IPU 产品。你可以通过 Paperspace(https://www.paperspace.com/graphcore) 的云笔记本尝试这些产品。

TPU:
- 谷歌的 TPU 当然是可用的,但它们并不是最受欢迎的加速器,因为你只能租用它们,而且软件在 GPU 和 TPU 之间的转换并不容易,所以许多(大多数?)开发者仍然停留在 GPU 领域,因为他们不想被锁定在谷歌垄断的硬件上。

关于 Pod 和机架:
- Cerebras 的 WaferScale Engine (WSE)
- SambaNova 的 DataScale
- 数十种不同的 pod 和机架配置,将上述 GPU 与超快速互连组合在一起。

这就是 2024 年第三季度的情况。

由于我们大多数人都是租用计算资源,从未亲眼见过它们的样子,这里展示了一个 8xH100 节点的实物图(这是 Dell PowerEdge XE9680 机架服务器的 GPU 托盘):

![nvidia-a100-spec](https://files.mdnice.com/user/59/9654f571-b9c1-4bf1-8755-2646b9977374.png)

## 术语表

- CPU: 中央处理器
- FPGA: 现场可编程门阵列
- GCD: 图形计算芯片
- GPU: 图形处理器
- HBM: 高带宽内存
- HPC: 高性能计算
- HPU: Habana Gaudi AI 处理器单元
- IPU: 智能处理单元
- MME: 矩阵乘法引擎
- QPU: 量子处理单元
- RDU: 可重构数据流单元
- TPU: 张量处理单元

## 最重要的需要理解的事情

我将在本书中多次重复以下观点 - 仅仅购买/租用最昂贵的加速器并不足以期望获得高投资回报率(ROI)。

ML训练的高ROI有两个指标:
1. 训练完成的速度,因为如果训练时间比计划长2-3倍,你的模型可能在发布前就已经过时了 - 在当前竞争激烈的ML市场中,时间就是一切。
2. 训练模型的总花费,因为如果训练时间比计划长2-3倍,你最终将花费2-3倍的金钱。

除非其他购买/租用的硬件经过仔细选择以匹配所需的工作负载,否则加速器很可能会大量闲置,从而浪费时间和金钱。最关键的组件是网络(https://github.com/stas00/ml-engineering/tree/master/network),其次是存储(https://github.com/stas00/ml-engineering/tree/master/storage),最不重要的是CPU和CPU内存(至少对于典型的训练工作负载而言,任何CPU限制都可以通过多个`DataLoader`工作进程来补偿)。

如果是租用计算资源,通常没有选择的自由 - 硬件要么是固定的,要么只有一些组件可以更换,但选择不多。因此,有时选择的云提供商可能无法提供足够匹配的硬件,在这种情况下,最好寻找其他提供商。

如果你购买自己的服务器,我建议在购买之前进行深入的尽职调查。

除了硬件,你当然还需要能够高效部署硬件的软件。

我们将在本书的各个章节中讨论硬件和软件方面的问题。你可以从这里(https://github.com/stas00/ml-engineering/tree/master/training/performance)和这里(https://github.com/stas00/ml-engineering/tree/master/training/model-parallelism)开始。



## 我们关心加速器的哪些特性

让我们在接下来的章节中以 NVIDIA A100 的规格作为参考点。

![nvidia-a100-spec](https://files.mdnice.com/user/59/36d7d242-20b6-4021-8ab0-0abe42045743.png)

source(https://www.nvidia.com/en-us/data-center/a100/)

### TFLOPS

如前所述,机器学习训练和推理的大部分工作都是矩阵乘法。如果你还记得代数中的矩阵乘法,它由许多乘法运算后跟着求和组成。可以计算每一个这样的运算,并定义芯片在一秒钟内可以执行多少这样的运算。

这是评判加速器的关键特征之一。TFLOPS这个术语定义了芯片每秒可以执行多少万亿次浮点运算。数值越高越好。对于不同的数据类型,有不同的定义。例如,以下是来自A100规格(https://www.nvidia.com/en-us/data-center/a100/)的理论峰值TFLOPS的几个条目:

| Data type \ TFLOPS     | w/o Sparsity | w/ Sparsity |
| :--------------------  | -----------: | ----------: |
| FP32                   |         19.5 |         n/a |
| Tensor Float 32 (TF32) |          156 |         312 |
| BFLOAT16 Tensor Core   |          312 |         624 |
| FP16 Tensor Core       |          312 |         624 |
| FP8 Tensor Core        |          624 |        1248 |
| INT8 Tensor Core       |          624 |        1248 |

注意事项：

* INT8是以TeraOperations（万亿次操作）来衡量的，因为它不是浮点运算。

* FLOPS这个术语可能指的是浮点运算的总数，例如在计算单个Transformer迭代需要多少FLOPS时；它也可能指每秒浮点运算次数 - 所以要注意上下文。当你阅读加速器规格时，它几乎总是指每秒定义。当讨论模型架构时，通常只是指浮点运算的总数。

因此，你可以看到int8的速度是bf16的2倍，而bf16又是tf32的2倍。

此外，TFLOPS取决于矩阵的大小，如下表所示：

![nvidia-a100-matmul-tflops](https://files.mdnice.com/user/59/f29fb4a3-f0ec-4967-9fd3-68a467399ce6.png)

source(https://developer.nvidia.com/blog/cuda-11-features-revealed/)

正如你所看到的,由于tile 和 wave quantization 的影响(https://github.com/stas00/ml-engineering/tree/master/training/performance#tile-and-wave-quantization),性能差异是非线性的。


#### TFLOPS 比较表

让我们来看看高端加速器支持的数据类型(https://github.com/stas00/ml-engineering/blob/master/training/dtype.md)和相应的理论峰值 TFLOPS 规格(不包括稀疏性)。按 bf16 列排序。

| Accelerator \ TFLOPS |  fp32 |   tf32 | fp16 | bf16 |  fp8 | int8 | fp6  | fp4    | Notes |
| :---------------     | ----: | -----: | ---: | ---: | ---: | ---: | --:  | -----: | ----: |
| NVIDIA GB200 SXM     |    ?? | 1250.0 | 2500 | 2500 | 5000 | 5000 | 5000 | 10000  |     2 |
| NVIDIA B200 SXM      |    ?? | 1125.0 | 2250 | 2250 | 4500 | 4500 | 4500 | 9000   |       |
| NVIDIA B100 SXM      |    ?? |  875.0 | 1750 | 1750 | 3500 | 3500 | 3500 | 7000   |       |
| AMD MI300X           | 163.4 |  653.7 | 1300 | 1300 | 2600 | 2600 | X    | X      |     3 |
| NVIDIA H200 SXM      |  67.0 |  494.5 |  989 |  989 | 1979 | 1979 | X    | X      |     4 |
| NVIDIA H100 SXM      |  67.0 |  494.5 |  989 |  989 | 1979 | 1979 | X    | X      |       |
| NVIDIA GH200 SXM     |  67.0 |  494.5 |  989 |  989 | 1979 | 1979 | X    | X      |     6 |
| Intel Gaudi3         |   229 |    459 |  459 | 1835 | 1835 |    V | X    | X      |     1 |
| NVIDIA H100 PCIe     |  51.0 |  378.0 |  756 |  756 | 1513 | 1513 | X    | X      |       |
| Intel Gaudi2         |     V |      V |    V |  432 |  865 |    V | X    | X      |     1 |
| Google TPU v5p       |     X |      X |    X |  459 |    X |  918 | X    | X      |       |
| AMD MI250X           |  47.9 |      X |  383 |  383 |    X |  383 | X    | X      |       |
| NVIDIA L40S          |  91.6 |  183.0 |  362 |  362 |  733 |  733 | X    | X      |       |
| AMD MI250            |  45.3 |      X |  362 |  362 |    X |  362 | X    | X      |       |
| NVIDIA A100 SXM      |  19.5 |  156.0 |  312 |  312 |    X |  624 | X    | X      |       |
| NVIDIA A100 PCIe     |  19.5 |  156.0 |  312 |  312 |    X |  624 | X    | X      |     5 |
| Google TPU v4        |     X |      X |    X |  275 |    X |  275 | X    | X      |       |
| Google TPU v5e       |     X |      X |    X |  197 |    X |  394 | X    | X      |       |
|                      |       |        |      |      |      |      |      |        |       |

特定行的注释:

1. Intel Gaudi2和3只发布了部分TFLOPS规格(https://www.intel.com/content/www/us/en/content-details/817486/intel-gaudi-3-ai-accelerator-white-paper.html)，但它确实支持FP32、TF32、BF16、FP16和FP8、INT8和INT16。这些数字是针对MME(矩阵)计算的。

2. 由于GB200是2个B200芯片，表格包含了每个芯片的TFLOPS以进行公平比较 - 对于真正的GB200你需要将其乘以2 - 它似乎还将B200芯片运行得稍快，因此规格比独立的B200更高。这也意味着，与典型的8-GPU节点不同，使用GB200你将获得4-GPU节点(但它相当于8个B200，计算速度还要快约10%)。

3. 我没有包括"NVIDIA H100双NVL"，因为它实际上是2个GPU - 所以这不公平 - 它的FLOPS与H100相同，但所有参数都是2倍，而且它的内存稍多一些(每个芯片94GB，相比H100的80GB)，内存也稍快。

4. H200与H100相同，但内存为141GB，而不是80GB的HBM内存，其内存更快，HBMe@4.8TBps vs HBM@3.35TBps - 所以基本上H200解决了H100的计算效率问题。

5. 奇怪的是，NVIDIA A100 PCIe和SXM版本的规格(https://www.nvidia.com/en-us/data-center/a100/)报告有相同的TFLOPS，这很奇怪，因为SXM版本使用30%更多的功率，并使用快5%的HBM。

6. GH200 - 与GB200相同的注释 - 这是2个芯片，所以表格包含了每个芯片的规格，不包括稀疏性。

一般注释:

* int8是以TeraOperations来衡量的，因为它不是浮点运算。

* 如果你发现数字是上述的两倍 - 通常意味着包含了稀疏性(目前几乎没有人能从中受益，因为我们的矩阵是密集的)。

* 查看规格时要非常小心你正在阅读的数字 - 许多供应商经常发布带有稀疏性的TFLOPS，因为它们大约是2倍大，但即使他们指出了这一点，通常也是用小字体。我不得不要求NVIDIA在他们的H100规格中添加一个注释，说明这些数字是带有稀疏性的，因为他们最初没有提到这个重要的技术事实。而且在撰写本文时，99%的情况下你不会使用稀疏性，因此你大多数时候关心的实际理论TFLOPs是不包括稀疏性的(即上表中的数据)。

* 还要注意，如果加速器A发布的TFLOPS高于加速器B，并不意味着A更快。这些是理论数字，不仅在实践中永远无法达到 - 实际的TFLOPS效率(HFU)可能因供应商而异，甚至对于同一供应商的不同加速器架构也可能有很大差异。



#### 最大可实现的FLOPS

理论峰值FLOPS是加速器规格中公布的数值。它的计算方式如下：

`理论FLOPS = 计算单元时钟速度 * 每个计算单元每个时钟周期的flops * 计算单元数量`

其中：
- `计算单元时钟速度` - 计算单元时钟每秒滴答的次数（以Hz为单位）
- `每个计算单元每个时钟周期的flops` - 计算单元每个时钟周期可以执行的操作数。
- `计算单元数量` - 设备中有多少个单元

公布的理论峰值FLOPS的问题在于，它们是**非常**理论化的，即使在所有完美条件下也无法在实践中实现。每个加速器都有自己的现实FLOPS，这些数值并未公布，而且有一些来自社区的轶事报告尽力找出实际的最佳值，但我还没有找到任何官方报告。

如果你找到可靠的报告（论文？）显示本章讨论的一个或多个高端加速器可以预期的实际TFLOPS，请kindly提交一个PR包含这些信息。关键是要有一个参考来源，读者可以验证所提议的信息。

为了给我所说的内容提供一个数字感，让我们以A100为例，其规格中的bf16峰值性能为312 TFLOPS。在FlashAttention发明之前，众所周知，对于fp16/bf16混合精度训练模式，150TFLOPS接近可以达到的最高水平。而使用FlashAttention，它大约在180+TFLOPS左右。当然，这是针对LLM训练测量的，其中涉及网络和IO，这些会造成额外的开销。所以在这里，最大可实现的峰值性能可能在200到300 TFLOPS之间。

你可以通过在单个加速器上执行完美对齐的最大尺寸矩阵`matmul`来测量实际的峰值TFLOPS。你可以使用最大可实现矩阵乘法FLOPS查找器(https://github.com/stas00/ml-engineering/tree/master/compute/accelerator/benchmarks#maximum-achievable-matmul-flops-finder)来重现结果。但是，当然，这只会告诉你你的给定加速器及其软件栈在`matmul`上的表现如何 - 根据工作负载，这可能是你需要知道的全部，也可能不是。


#### 最大可实现矩阵乘法 FLOPS 比较表

以下测量结果是针对 BF16 输入（无稀疏性）的 `matmul` TFLOPS（关于 MAMF 的含义请参见上文）。按加速器效率排序：

| Accelerator      |  MAMF | Theory | Efficiency |       Best Shape | Notes            |
| :--------------- | ----: | -----: | ---------: |  :-------------- | ---------------: |
| NVIDIA A100 SXM  | 267.9 |    312 |      85.9% |  6912x16384x2048 | CUDA-12.1        |
| NVIDIA GH200 SXM | 821.0 |    989 |      83.0% | 11264x19712x1536 | CUDA-12.5        |
| NVIDIA A100 PCIe | 256.4 |    312 |      82.2% |   2304x5120x1536 | CUDA-12.1        |
| NVIDIA H100 SXM  | 792.1 |    989 |      80.1% |  6144x17920x2816 | CUDA-12.1        |
| AMD MI250X       | 147.0 |  191.5 |      76.7% | 1024x14080x19968 | ROCm-6.2 / 1 GCD |
| AMD MI300X       | 781.9 |   1300 |      60.1% |  4096x10240x4864 | ROCm-6.2         |
|                  |       |        |            |                  |                  |

注意事项:这些数字是通过对各种形状执行`matmul`的非穷尽子空间进行暴力搜索获得的。请参见:最大可实现矩阵乘法 TFLOPS 查找器(https://github.com/stas00/ml-engineering/tree/master/compute/accelerator/benchmarks#maximum-achievable-matmul-flops-finder),使用测量时可用的软件组件,因此我强烈建议您在您的特定设置上重新运行`mamf-finder.py`,以获得真实的数据。此表中的数字是粗略估计,不应作为绝对值使用。随着软件的改进,这些数字将会提高,更接近理论规格。因此,理想情况下应该每6个月左右重新运行一次。

注意:
- 完整的理论数据集请参见理论加速器 TFLOPS(https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/README.md#tflops-comparison-table)
- 效率是 MAMF/理论值*100
- 最佳形状是脚本检测到的,但可能还有许多其他形状具有类似的性能 - 列出它是为了可重现性
- 如果您获得的性能远低于此表中的数字,请检查目标硬件是否有足够的散热,如果加速器过热,通常会降低其性能。当然,这里假设电源供应符合规格。后者在数据中心很少出现问题,但散热不良并非闻所未闻。
- 您使用的软件可能会产生巨大的差异 - 例如,使用 MI300X 时,我使用 ROCm-6.1 达到了 450TFLOPS,但正如您所看到的,ROCm-6.2 有了显著改进,增加了惊人的 300 TFLOPS
- 然后还有各种系统优化 - 例如,对于 MI300X,禁用内核设置中的 numa_balancing 是必须的。
- AMD MI250X 有 2 个 GCD - 所以理论 TFLOPS 需要减半,因为单个 matmul 只使用其中一个,而 383 TFLOPS 是针对 2 个 GCD 报告的。

此外，需要理解的是，知道在某个特定形状（如 `4352x13568x3840`）下的最大可实现 Matmul TFLOPS，并不意味着在实际应用中可以获得相同的性能，因为几乎不可能正好遇到这个形状。相反，要真正了解你的系统，你需要使用模型在训练过程中实际使用的形状运行 MAMF Finder（最大可实现矩阵乘法性能工具）。这才是该工具的核心意图。一旦得到了这些 TFLOPS 的测量结果，你就能大致判断，当你测量训练过程中的实际 TFLOPS 时，优化工作何时可以停止。

最后,我想再次重申,**这里的目的不是指出哪个加速器比另一个更高效,而是让人们了解情况,如何理解这些理论规格,并帮助您了解何时需要继续优化系统,何时停止。因此,从这些笔记和数字开始作为起点,然后测量您自己的用例,并使用后者的测量来获得最佳结果。**


### 加速器内存大小和速度

加速器使用高带宽内存(https://en.wikipedia.org/wiki/High_Bandwidth_Memory) (HBM)，这是SDRAM内存的3D版本。例如，A100-SXM配备了1.6TBps的HBM2，而H100-SXM配备了3.35TBps的HBM3。

以下是规格：

| 代 | 数据速率<br> (Gbps) | 每设备带宽<br> (GBps) | 堆叠<br> 高度 | 最大DRAM<br> 容量 (GB) | 最大设备<br> 容量 (GB) |
| :---  | --: | ---:  | -: | -: | -: |
| HBM   | 1.0 |   128 |  8 |  2 | 16 |
| HBM2  | 2.0 |   256 |  8 |  2 | 16 |
| HBM2e | 3.6 |   461 | 12 |  3 | 36 |
| HBM3  | 6.4 |   819 | 16 |  4 | 64 |
| HBM3e | 9.6 |  1229 | 16 |  4 | 64 |

由于HBM是多个DRAM芯片的堆叠，*堆叠高度*指定了每个设备有多少个芯片。

通常，加速器拥有的片上内存越多越好。在任何给定时刻，通常大部分模型权重都没有被使用，因为它们在等待轮到被处理，因此大内存允许更多的模型存储在加速器内存中，并可立即访问和更新。当内存不足时，有时模型必须分割到多个加速器上，或卸载到CPU和/或磁盘。

以下是最近高端加速器的内存规格（有些尚未正式发布），按内存大小，然后按带宽排序：

| 加速器               | 内存<br> (GB)     | 类型  | 峰值<br>带宽<br> (TBps)     |
| :------------------- | ----------------: | :---- | -------------------: |
| NVIDIA B200 SXM      |               192 | HBM3e |                 8.00 |
| NVIDIA B100 SXM      |               192 | HBM3e |                 8.00 |
| AMD MI300X           |               192 | HBM3  |                 5.30 |
| NVIDIA GH200 SXM (2) |               141 | HBM3e |                 4.80 |
| NVIDIA H200 SXM      |               141 | HBM3e |                 4.80 |
| Intel Gaudi3         |               128 | HBM2e |                 3.70 |
| AMD MI250            |               128 | HBM2e |                 3.28 |
| AMD MI250X           |               128 | HBM2e |                 3.28 |
| NVIDIA GH200 SXM (1) |                96 | HBM3  |                 4.00 |
| Intel Gaudi2         |                96 | HBM2e |                 2.46 |
| Google TPU v5p       |                95 | HBM2e |                 4.80 |
| NVIDIA H100 SXM      |                80 | HBM3  |                 3.35 |
| NVIDIA A100 SXM      |                80 | HBM2e |                 2.00 |
| NVIDIA H100 PCIe     |                80 | HBM3  |                 2.00 |
| NVIDIA L40S          |                48 | GDDR6 |                 0.86 |
| Google TPU v4        |                32 | HBM2  |                 1.20 |
| Google TPU v5e       |                16 | HBM2  |                 1.60 |

注意：

* 我没有包括 `NVIDIA H100 dual NVL`，因为它是2个H100 GPU，每个芯片额外有14GB内存，内存速度略快（3.9TBps vs 3.35TBps）- 但在上表中它会有不公平的优势，因为其他所有条目都是按单芯片计算的。（我猜AMD250也是2个GCD，但它们无论如何都不太具有竞争力，很快就会被这个表中的新产品取代）

内存速度（带宽）当然非常重要，因为如果速度不够快，计算单元最终会闲置，等待数据在内存之间移动。

### 散热

这对于购买自己的硬件时很重要，当你在云上租用时，供应商应该会负责适当的散热。

关于散热，唯一重要的实际理解是，如果加速器没有保持冷却，它们将降低计算时钟速度并减慢一切（有时甚至可能崩溃，尽管降频应该可以防止这种情况）。

## 用于LLM/VLM工作负载的高端加速器

### 云端和本地加速器集群
加速器

最常见的可以在计算云上租用或购买的加速器：

NVIDIA:
- B200 - 尚无官方规格 - 只能从DGX规格中推导：https://www.nvidia.com/en-us/data-center/hgx/（XXX：在发布官方规格时更新）
- B100 - 尚无官方规格 - 只能从DGX规格中推导：https://www.nvidia.com/en-us/data-center/hgx/（XXX：在发布官方规格时更新）
- H200(https://www.nvidia.com/en-us/data-center/h200/) - 主要与H100相同，但内存更大更快！预计将于2024年中旬左右推出。
- H100(https://www.nvidia.com/en-us/data-center/h100) - 比A100快2-3倍（半精度），fp8快6倍，自2023年第四季度以来已在所有一级计算云上可用。
- GH200(https://www.nvidia.com/en-us/data-center/grace-hopper-superchip/) - 一张卡上有2个芯片 - (1) H100，96GB HBM3或144GB HBM3e + (2) Grace CPU，624GB RAM - 据报道首批单位已开始供货。不要与H200混淆，那是一张不同的卡。
- L40S(https://www.nvidia.com/en-us/data-center/l40s/) - 一款强大的卡，据说比H100便宜2倍以上，而且比A100更强大。
- A100(https://www.nvidia.com/en-us/data-center/a100/#specifications) - 可用性很高，但已经开始过时。但考虑到比H100便宜得多，这仍然是一款很棒的GPU。

AMD:
- MI250(https://www.amd.com/en/products/accelerators/instinct/mi200/mi250.html) ~= A100 - 很少有云提供
- MI300X(https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html) ~= H100 - 刚刚开始出现 - 主要在二级云上（很多新创公司）。

Intel:
- Gaudi2(https://habana.ai/products/gaudi2/) 理论TFLOPS规格介于A100和H100之间(https://docs.habana.ai/en/latest/Gaudi_Overview/Gaudi_Architecture.html) - 目前在cloud.google.com(https://cloud.google.com)上可用性很低，有很长的等待名单，据说应该在2024年第一季度减少。AWS通过DL1实例(https://aws.amazon.com/ec2/instance-types/dl1/)提供较旧的Gaudi1。它也可通过Supermicro和WiWynn进行本地实施。
- Gaudi3(https://habana.ai/products/gaudi3/)，理论TFLOPS规格介于B100和B200之间(https://www.intel.com/content/www/us/en/content-details/817486/intel-gaudi-3-ai-accelerator-white-paper.html)

Graphcore:
- IPU(https://www.graphcore.ai/products) - 通过Paperspace(https://www.paperspace.com/graphcore)提供。最新产品MK2 (C600)每卡只有0.9GB SRAM，所以不清楚这张卡如何在ML方面做任何事情 - 即使是小模型的推理也无法容纳其模型权重 - 但Graphcore正在进行一些新的工作，据说我们很快就会发现。这里有一个关于IPU如何工作的很好解释(https://thytu.com/posts/ipus-101/)。

SambaNova:
- DataScale SN30(https://sambanova.ai/products/datascale/)

### 本地加速器集群

Cerebras:
- 集群(https://www.cerebras.net/product-cluster/)
- 系统(https://www.cerebras.net/product-system/)
基于晶圆级引擎（WSE）。

### 仅云端解决方案

这些只能通过云使用：

Google
- TPUs(https://cloud.google.com/tpu)，规格(https://cloud.google.com/tpu/docs/system-architecture-tpu-vm) - 锁定，无法切换到其他供应商如NVIDIA -> AMD

Cerebras:
- Cloud(https://www.cerebras.net/product-cloud/)

### 如何获得最优惠的价格

请记住，只要你愿意批量购买/租用或租用1-3年，广告价格几乎总是可以协商的。你会发现，你最终支付的实际价格可能比广告的"公开"价格低很多倍。一些云提供商在你在他们的网站上选择更长期承诺时已经包含了折扣，但直接与他们的销售团队协商总是最好的。除了或代替$$折扣，你可能会得到一些有用的免费功能/升级。

如果你的公司有风险投资者 - 提到这一点可能会有很大帮助，因为这样云提供商就知道你可能会在未来购买更多计算资源，更有可能给出更多折扣。

二级云可能会比一级云给出更好的价格。截至本文撰写时，一级云包括AWS、OCI、Azure和GCP。

对于基准价格，应该很容易找到一些提供最新跨云公开价格比较的好网站 - 只需搜索类似cloud gpu pricing comparison(https://www.google.com/search?q=cloud+gpu+pricing+comparison)的内容。一些好的起点：vast.ai(https://cloud.vast.ai/create/)，特别是对于集群gpulist.ai(https://gpulist.ai)。

在寻找解决方案时，请记住，仅仅租用最强大的加速器是不够的。你还需要快速的节点内(https://github.com/stas00/ml-engineering/tree/master/network#intra-node-networking)和节点间(https://github.com/stas00/ml-engineering/tree/master/network#inter-node-networking)连接以及足够快的存储(https://github.com/stas00/ml-engineering/tree/master/storage) - 没有这些，昂贵的加速器将闲置等待数据到达，你可能会浪费大量金钱并损失时间。

## 加速器详细信息

### NVIDIA

缩写：

- CUDA：统一计算设备架构（NVIDIA专有）

NVIDIA特有的GPU关键特征：
- CUDA Core - 类似于CPU核心，但与通常有10-100个强大核心的CPU不同，CUDA Core较弱，数量以千计，允许执行大规模通用计算（并行化）。与CPU核心一样，CUDA Core在每个时钟周期执行单个操作。
- Tensor Core - 专门设计用于执行快速乘法和加法运算（如矩阵乘法）的特殊计算单元。这些核心在每个时钟周期执行多个操作。它们可以对低精度或混合精度数据类型进行极快的计算，但会有一些精度损失（fp16、bf16、tf32、fp8等）。这些核心专为ML工作负载设计。
- 流式多处理器（SM）是CUDA Core、Tensor Core和其他组件的集群。

例如，A100-80GB有：

- 6912个CUDA Core
- 432个Tensor Core（第3代）
- 108个流式多处理器（SM）

H100有：

- 16896个FP32 CUDA Core
- 528个Tensor Core（第4代）
- 132个流式多处理器（SM）

### AMD

AMD特有的GPU关键特征：
- 流处理器 - 在功能上类似于CUDA Core - 即这些是并行计算单元。但它们并不相同，所以不能仅通过比较CUDA Core数量与流处理器数量来比较两个GPU。
- 计算单元 - 是流处理器和其他组件的集群

例如，AMD MI250有：
- 13,312个流处理器
- 208个计算单元

### Intel Gaudi2

架构(https://docs.habana.ai/en/latest/Gaudi_Overview/Gaudi_Architecture.html)

- 芯片上集成24个100千兆以太网（RoCEv2）- 其中21个用于节点内，3个用于节点间（因此节点内有`21*8=168`张卡（每GPU 262.5GBps），节点间有`3*8=24`张卡（节点间2.4Tbps））
- 板载96GB HBM2E内存，每芯片带宽2.45 TBps，每节点总计768GB

一个服务器/节点由8个GPU构建，然后可以用这些服务器的机架扩展。

没有发布官方的TFLOPS信息（据与Intel代表交谈，他们也无意发布任何信息）。他们发布了以下基准测试(https://developer.habana.ai/resources/habana-models-performance/)，但我不确定如何使用这些来与其他提供商进行比较。

比较：据说Gaudi2与NVIDIA H100竞争

## API

部署高端GPU需要哪些软件？

### NVIDIA

NVIDIA GPU运行在CUDA(https://developer.nvidia.com/cuda-toolkit)上

### AMD

AMD GPU运行在ROCm(https://www.amd.com/en/products/software/rocm.html)上 - 注意，在PyTorch中，你可以在基于ROCm的GPU上使用基于CUDA的软件！因此，切换到最近的AMD MI250、MI300X和其他新兴GPU应该很简单。

### Intel Gaudi

API通过Habana SynapseAI® SDK(https://habana.ai/training-software/)提供，支持PyTorch和TensorFlow。

有用的集成：
- HF Optimum Habana(https://github.com/huggingface/optimum-habana)，其中还包括DeepSpeed(https://github.com/microsoft/DeepSpeed)集成。

## 公平比较

很难比较不同产品的规格，因为几乎所有竞争对手都采用营销技巧，使人无法通过比较两组规格来了解实际差异。

- MLPerf通过MLCommons(https://mlcommons.org/en/)发布各种硬件基准测试，测量训练、推理、存储和其他任务的性能。例如，这里是截至本文撰写时最新的训练v3.0(https://mlcommons.org/en/training-normal-30/)和推理v3.1(https://mlcommons.org/en/inference-datacenter-31/)结果。

   只是我完全不知道该如何使用它——几乎无法理解或控制这个界面。在我看来，这个本意很好的工具因为过度设计而失去了价值，没有考虑用户如何真正受益。例如，我并不关心 CV 数据，我只想快速查看 LLM（大语言模型）相关的行，但我做不到。而且，这些比较也不是一对一的公平比较，所以我完全不知道该如何判断哪个硬件更好。

## 电源和散热

很可能你是租用加速器节点，由他人负责确保它们正常运行，但如果你拥有加速器，你确实需要知道如何提供足够的电源和适当的散热。

### 电源

一些高端消费级 GPU 显卡配有 2 个甚至 3 个 PCI-E 8 针电源插座。请确保每个插座都插入独立的 12V PCI-E 8 针电源线。不要使用同一根线分出的两个接口（也叫“猪尾线”）。也就是说，如果显卡有 2 个插座，你需要从电源（PSU）拉出两根独立的 PCI-E 8 针电源线，而不是一根线末端分出两个 PCI-E 8 针接口的那种！否则，显卡将无法发挥其全部性能。

每根 PCI-E 8 针电源线需要连接到电源的一条 12V 轨道上，每根线可提供最多 150W 的功率。

另外，有些显卡可能使用 PCI-E 12 针接口，这种接口可以提供高达 500-600W 的功率。

低端显卡可能使用 6 针接口，这类接口最多可提供 75W 的功率。

此外，你需要使用电压稳定的高端电源。某些低质量电源可能无法为显卡提供所需的稳定电压，从而导致其无法达到最佳性能。

当然，电源还需要有足够的剩余功率来支持显卡的运行。

### 散热

当GPU过热时，它会开始降频，无法发挥全部性能，如果温度过高，甚至可能关机。

很难说在GPU重负载时应该追求的确切最佳温度，但可能任何低于+80°C的温度都是好的，但越低越好 - 也许70-75°C是一个极好的范围。降频可能在84-90°C左右开始。但除了降低性能外，长期的非常高温可能会减少GPU的寿命。

