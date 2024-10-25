> 我的课程笔记，欢迎关注：https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode 
> 本篇文档的来源：https://github.com/stas00/ml-engineering 。

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


#### Maximum Achievable Matmul FLOPS comparison table

The following measurements are for `matmul` with BF16 inputs (no sparsity) TFLOPS (see above for what MAMF means). Sorted by accelerator efficiency:

| Accelerator      |  MAMF | Theory | Efficiency |       Best Shape | Notes            |
| :--------------- | ----: | -----: | ---------: |  :-------------- | ---------------: |
| NVIDIA A100 SXM  | 267.9 |    312 |      85.9% |  6912x16384x2048 | CUDA-12.1        |
| NVIDIA GH200 SXM | 821.0 |    989 |      83.0% | 11264x19712x1536 | CUDA-12.5        |
| NVIDIA A100 PCIe | 256.4 |    312 |      82.2% |   2304x5120x1536 | CUDA-12.1        |
| NVIDIA H100 SXM  | 792.1 |    989 |      80.1% |  6144x17920x2816 | CUDA-12.1        |
| AMD MI250X       | 147.0 |  191.5 |      76.7% | 1024x14080x19968 | ROCm-6.2 / 1 GCD |
| AMD MI300X       | 781.9 |   1300 |      60.1% |  4096x10240x4864 | ROCm-6.2         |
|                  |       |        |            |                  |                  |

Caveat emptor: these numbers were achieved by a brute-force search of a non-exhaustive sub-space of various shapes performing `matmul`. See:  Maximum Achievable Matmul TFLOPS Finder(https://github.com/stas00/ml-engineering/tree/master/compute/accelerator/benchmarks#maximum-achievable-matmul-flops-finder) using the software components available at the time of taking the measurement, so I highly recommend you re-run `mamf-finder.py` on your particular setup to get the true to your setup numbers. The numbers in this table are a rough estimation and shouldn't be used as absolute. As the software improves these numbers will improve coming closer to the theoretical spec. So ideally they ought to be re-rerun once in 6 months or so.

Notes:
- For the full set of theoretical ones see Theoretical accelerator TFLOPS(https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/README.md#tflops-comparison-table)
- Efficiency is MAMF/Theory*100
- Best shape is the one detected by the script, but there could be many others with similar performance - it's listed for reproducibility
- If you get a much lower performance than the numbers in this table, check that the target hardware has an adequate cooling, if the accelerator is overheated it'd usually throttle its performance down. And, of course, the assumption here is that the power supply matches the spec. The latter is rarely a problem in data centers, but bad cooling is not unheard of.
- Which software you use can make a huge difference - e.g. with MI300X I clocked 450TFLOPS using ROCm-6.1, but as you can see there was a dramatic improvement in ROCm-6.2 where it jumped a whooping additional 300 TFLOPS up
- Then there are various system optimizations - e.g. in the case of MI300X disabling numa_balancing in the kernel settings is a must.
- AMD MI250X has 2 GCDs - so the theoretical TFLOPS needs to be halved, as a single matmul uses only 1 of them and 383 TFLOPS is reported for 2 GCDs.

Also it's important to understand that knowing the Maximum Achievable Matmul TFLOPS at some particular shape like `4352x13568x3840` doesn't mean you can expect to get the same performance in your real application because chances are close to 0 that you will ever hit that exact shape. Instead, to know your system well, you'd run the MAMF Finder with the actual shapes your model is using during its training. This is really the key intention of this tool. Once you have that TFLOPS measurement you will have a good sense of where you can stop optimizing when you measure the actual TFLOPS reported by your training.

And to conclude this section I'd like to repeat again that **the intention here is not to point fingers at which accelerator is more efficient than another, but to give a sense of what's what and how to navigate those theoretical specs and to help you understand when you need to continue optimizing your system and when to stop. So start with these notes and numbers as a starting point, then measure your own use case and use that latter measurement to gain the best outcome.**


### Accelerator memory size and speed

The accelerators use High Bandwidth Memory(https://en.wikipedia.org/wiki/High_Bandwidth_Memory) (HBM) which is a 3D version of SDRAM memory. For example, A100-SXM comes with HBM2 at 1.6TBps, and H100-SXM comes with HBM3 at 3.35TBps.

Here are the specs:

| Gen | Data Rate<br> (Gbps) | Bandwidth per<br> Device (GBps) | Stack<br> Height |	Max. DRAM<br> Capacity (GB) | Max. Device<br> Capacity (GB) |
| :---  | --: | ---:  | -: | -: | -: |
| HBM   | 1.0 |   128 |  8 |  2 | 16 |
| HBM2  | 2.0 |   256 |  8 |  2 | 16 |
| HBM2e | 3.6 |   461 | 12 |  3 | 36 |
| HBM3  | 6.4 |   819 | 16 |  4 | 64 |
| HBM3e | 9.6 |  1229 | 16 |  4 | 64 |

Since HBM is a stack of multiple DRAM chips, the *Stack Height* specifies how many chips are per device.

Typically the more on-device memory the accelerator has the better. At any given time usually most of the model weights aren't being used as they wait for their turn to be processed and thus large memory allows more of the model to be on the accelerator memory and immediately available for access and update. When there is not enough memory, sometimes the model has to be split across multiple accelerators, or offloaded to CPU and/or disk.

Here are the memory specs for the recent high end accelerators (some aren't GA yet), sorted by memory size, then bandwidth:

| Accelerator          |  Memory<br> (GBs) | Type  | Peak<br>Bandwidth<br> (TBps) |
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

Notes:

* I didn't include `NVIDIA H100 dual NVL` as it's 2x H100 GPUs with 14GB memory extra per chip and slightly faster memory (3.9TBps vs 3.35TBps) - but it would have an unfair advantage in the above table as everything else is per-chip. (I guess AMD250 is also 2 GCDs, but they aren't very competitive anyway and will soon be displaced from this table by newer offerings)

Memory speed (bandwidth) is, of course, very important since if it's not fast enough than the compute ends up idling waiting for the data to be moved to and from the memory.

### Heat

This is of interest when you buy your own hardware, when you rent on the cloud the provider hopefully takes care of adequate cooling.

The only important practical understanding for heat is that if the accelerators aren't kept cool they will throttle their compute clock and slow everything down (and could even crash sometimes, albeit throttling is supposed to prevent that).




## High end accelerators for LLM/VLM workloads

### Cloud and on-premises  accelerator clusters
 accelerators

Most common accelerators that can be either rented on compute clouds or purchased:

NVIDIA:
- B200 - no official spec yet - only can be derived from the DGX spec: https://www.nvidia.com/en-us/data-center/hgx/ (XXX: update when official specs are released)
- B100 - no official spec yet - only can be derived from the DGX spec: https://www.nvidia.com/en-us/data-center/hgx/ (XXX: update when official specs are released)
- H200(https://www.nvidia.com/en-us/data-center/h200/) - mainly the same as H100, but with more and faster memory! Supposed to become available some time mid-2024.
- H100(https://www.nvidia.com/en-us/data-center/h100) - 2-3x faster than A100 (half precision), 6x faster for fp8, has been available on all Tier-1 compute clouds since Q4-2023.
- GH200(https://www.nvidia.com/en-us/data-center/grace-hopper-superchip/) - 2 chips on one card - (1) H100 w/ 96GB HBM3 or 144GB HBM3e + (2) Grace CPU w/ 624GB RAM - first units have been reported to become available. Do not confuse with H200, which is a different card.
- L40S(https://www.nvidia.com/en-us/data-center/l40s/) - a powerful card that is supposed to be more than 2x cheaper than H100, and it's more powerful than A100.
- A100(https://www.nvidia.com/en-us/data-center/a100/#specifications) - huge availability, but already getting outdated. But given the much lower cost than H100 this is still a great GPU.

AMD:
- MI250(https://www.amd.com/en/products/accelerators/instinct/mi200/mi250.html) ~= A100 - very few clouds have them
- MI300X(https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html) ~= H100 - just starting to emerge - and mainly on Tier-2 clouds (lots of new startups).

Intel:
- Gaudi2(https://habana.ai/products/gaudi2/) somewhere between A100 and H100 theoretical TFLOPS-wise [spec](https://docs.habana.ai/en/latest/Gaudi_Overview/Gaudi_Architecture.html) - [Currently there is a very low availability on cloud.google.com](https://cloud.google.com) with a long waiting list which supposedly should be reduced in Q1-2024. AWS has the older Gaudi1 via [DL1 instances](https://aws.amazon.com/ec2/instance-types/dl1/). It's also available on-premises implementations via Supermicro and WiWynn.
-  Gaudi3(https://habana.ai/products/gaudi3/), somewhere between B100 and B200 theoretical TFLOPS-wise [spec](https://www.intel.com/content/www/us/en/content-details/817486/intel-gaudi-3-ai-accelerator-white-paper.html)

Graphcore:
- IPU(https://www.graphcore.ai/products/ipu) - available via Paperspace(https://www.paperspace.com/graphcore). the latest product MK2 (C600) has only 0.9GB SRAM per card, so it's not clear how this card can do anything ML-wise - even inference of a small model won't fit its model weights - but there is something new at works at Graphcore, which I'm told we should discover soon. Here is is a good explanation of how IPU works(https://thytu.com/posts/ipus-101/).

SambaNova:
- DataScale SN30(https://sambanova.ai/products/datascale/)

### On-premises accelerator clusters

Cerebras:
- clusters(https://www.cerebras.net/product-cluster/)
- systems(https://www.cerebras.net/product-system/)
based on WaferScale Engine (WSE).

### Cloud-only solutions

These can be only used via clouds:

Google
- TPUs(https://cloud.google.com/tpu), specs(https://cloud.google.com/tpu/docs/system-architecture-tpu-vm) - lock-in, can't switch to another vendor like NVIDIA -> AMD

Cerebras:
- Cloud(https://www.cerebras.net/product-cloud/)



### How to get the best price

Remember that the advertised prices are almost always open to negotiations as long as you're willing to buy/rent in bulk or if renting for a 1-3 years. What you will discover is that the actual price that you end up paying could be many times less than the advertised "public" price. Some cloud providers already include the discount as you choose a longer commitment on their website, but it's always the best to negotiate directly with their sales team. In addition or instead of a $$-discount you could be offered some useful features/upgrades for free.

If your company has venture capital investors - it could help a lot to mention that, as then the cloud provider knows you are likely to buy more compute down the road and more likely to discount more.

Tier 2 clouds are likely to give better prices than Tier 1. Tier 1 as of this writing is AWS, OCI, Azure and GCP.

For the baseline prices it should be easy to find a few good sites that provide an up-to-date public price comparisons across clouds - just search for something like cloud gpu pricing comparison(https://www.google.com/search?q=cloud+gpu+pricing+comparison). Some good starting points: vast.ai(https://cloud.vast.ai/create/) and specifically for clusters gpulist.ai(https://gpulist.ai).

When shopping for a solution please remember that it's not enough to rent the most powerful accelerator. You also need fast intra-node(https://github.com/stas00/ml-engineering/tree/master/network#intra-node-networking) and inter-node(https://github.com/stas00/ml-engineering/tree/master/network#inter-node-networking) connectivity and sufficiently fast storage(https://github.com/stas00/ml-engineering/tree/master/storage) - without which the expensive accelerators will idle waiting for data to arrive and you could be wasting a lot money and losing time.

## Accelerators in detail

### NVIDIA

Abbreviations:

- CUDA: Compute Unified Device Architecture (proprietary to NVIDIA)

NVIDIA-specific key GPU characteristics:
- CUDA Cores - similar to CPU cores, but unlike CPUs that typically have 10-100 powerful cores, CUDA Cores are weaker and come in thousands and allow to perform massive general purpose computations (parallelization). Like CPU cores CUDA Cores perform a single operation in each clock cycle.
- Tensor Cores - special compute units that are designed specifically to perform fast multiplication and addition operations like matrix multiplication. These perform multiple operations in each clock cycle. They can execute extremely fast computations on low or mixed precision data types with some loss (fp16, bf16, tf32, fp8, etc.). These cores are specifically designed for ML workloads.
- Streaming Multiprocessors (SM) are clusters of CUDA Cores, Tensor Cores and other components.

For example, A100-80GB has:

- 6912 CUDA Cores
- 432 Tensor Cores (Gen 3)
- 108 Streaming Multiprocessors (SM)

H100 has:

- 16896 FP32 CUDA Cores
- 528 Tensor Cores (Gen 4)
- 132 Streaming Multiprocessors (SM)

### AMD

AMD-specific key GPU characteristics:
- Stream Processors - are similar in functionality to CUDA Cores - that is these are the parallel computation units. But they aren't the same, so one can't compare 2 gpus by just comparing the number of CUDA Cores vs the number of Stream Processors.
- Compute Units - are clusters of Stream Processors and other components

for example, AMD MI250 has:
- 13,312 Stream Processors
- 208 Compute Units

### Intel Gaudi2

Architecture(https://docs.habana.ai/en/latest/Gaudi_Overview/Gaudi_Architecture.html)

- 24x 100 Gigabit Ethernet (RoCEv2) integrated on chip - 21 of which are used for intra-node and 3 for inter-node (so `21*8=168` cards for intra-node (262.5GBps per GPU), and `3*8=24` cards for inter-node (2.4Tbps between nodes)
- 96GB HBM2E memory on board w/2.45 TBps bandwidth per chip, for a total of 768GB per node

A server/node is built from 8 GPUs, which can then be expanded with racks of those servers.

There are no official TFLOPS information published (and from talking to an Intel representative they have no intention to publish any.) They publish the following benchmarks(https://developer.habana.ai/resources/habana-models-performance/) but I'm not sure how these can be used to compare this compute to other providers.

Comparison: supposedly Gaudi2 competes with NVIDIA H100

## API

Which software is needed to deploy the high end GPUs?

### NVIDIA

NVIDIA GPUs run on CUDA(https://developer.nvidia.com/cuda-toolkit)

### AMD

AMD GPUs run on ROCm(https://www.amd.com/en/products/software/rocm.html) - note that PyTorch you can use CUDA-based software on ROCm-based GPUs! So it should be trivial to switch to the recent AMD MI250, MI300X, and other emerging ones.

### Intel Gaudi

The API is via Habana SynapseAI® SDK(https://habana.ai/training-software/) which supports PyTorch and TensorFlow.

Useful integrations:
- HF Optimum Habana(https://github.com/huggingface/optimum-habana) which also includes - DeepSpeed(https://github.com/microsoft/DeepSpeed) integration.

## Apples-to-apples Comparison

It's very difficult to compare specs of different offerings since marketing tricks get deployed pretty much by all competitors so that one can't compare 2 sets of specs and know the actual difference.

- MLPerf via MLCommons(https://mlcommons.org/en/) publishes various hardware benchmarks that measure training, inference, storage and other tasks' performance. For example, here is the most recent as of this writing training v3.0(https://mlcommons.org/en/training-normal-30/) and inference v3.1(https://mlcommons.org/en/inference-datacenter-31/) results.

   Except I have no idea how to make use of it - it's close to impossible to make sense of or control the view. This is a great intention lost in over-engineering and not thinking about how the user will benefit from it, IMHO. For example, I don't care about CV data, I only want to quickly see the LLM rows, but I can't do it. And then the comparisons are still not apples to apples so how can you possibly make sense of which hardware is better I don't know.

## Power and Cooling

It is most likely that you're renting your accelerator nodes and someone else is responsible for ensuring they function properly, but if you own the accelerators you do need to know how to supply a sufficient power and adequate cooling.

### Power

Some high end consumer GPU cards have 2 and sometimes 3 PCI-E 8-Pin power sockets. Make sure you have as many independent 12V PCI-E 8-Pin cables plugged into the card as there are sockets. Do not use the 2 splits at one end of the same cable (also known as pigtail cable). That is if you have 2 sockets on the GPU, you want 2 PCI-E 8-Pin cables going from your PSU to the card and not one that has 2 PCI-E 8-Pin connectors at the end! You won't get the full performance out of your card otherwise.

Each PCI-E 8-Pin power cable needs to be plugged into a 12V rail on the PSU side and can supply up to 150W of power.

Some other cards may use a PCI-E 12-Pin connectors, and these can deliver up to 500-600W of power.

Low end cards may use 6-Pin connectors, which supply up to 75W of power.

Additionally you want the high-end PSU that has stable voltage. Some lower quality ones may not give the card the stable voltage it needs to function at its peak.

And of course the PSU needs to have enough unused Watts to power the card.

### Cooling

When a GPU gets overheated it will start throttling down and will not deliver full performance and it can even shutdown if it gets too hot.

It's hard to tell the exact best temperature to strive for when a GPU is heavily loaded, but probably anything under +80C is good, but lower is better - perhaps 70-75C is an excellent range to be in. The throttling down is likely to start at around 84-90C. But other than throttling performance a prolonged very high temperature is likely to reduce the lifespan of a GPU.
