> 我的课程笔记，欢迎关注：https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode 

> 本篇文档的来源：https://github.com/stas00/ml-engineering 。这篇文档深入探讨了节点间和节点内网络硬件在大规模机器学习训练中的关键作用。首先强调了网络速度对于充分利用昂贵加速器的重要性，特别是在训练大型语言模型时。随后解释了一系列关键术语和概念，详细介绍了集群网络的三个主要组成部分：前端网络、后端网络和带外网络。文章还讨论了RDMA网络技术，比较了不同节点内网络技术的理论带宽，并通过案例分析解释了单GPU训练、单节点多GPU训练和多节点训练的性能差异。文档还深入探讨了通信和计算重叠的概念，以及如何计算TFLOPS，同时强调了高速节点间网络对大型模型训练的重要性。文章还解释了实际网络吞吐量与理论带宽的差异，提供了基准测试结果，并讨论了网络延迟的影响。此外，文章还涉及了使用专用网络硬件和NCCL的相关内容，强调了节点接近性的重要性，并提供了一些云服务提供商的相关解决方案。最后，文档讨论了共享节点间网络可能带来的性能波动问题。总的来说，文档为读者提供了丰富的技术细节和实践建议，全面阐述了网络硬件在大规模机器学习训练中的重要性。对AI系统的网络部分感兴趣的小伙伴推荐阅读。

# 节点间和节点内网络硬件

**子章节**:

- 网络调试
- 网络基准测试

> 因为这两个子章节涉及到代码，后续会单独翻译和代码解析。

## 介绍

购买/租用昂贵的加速器来快速训练和推理是不够的。你需要确保你的存储IO、CPU和网络足够快，以“喂饱加速器熔炉”。如果不能确保，那么昂贵的加速器将无法充分利用，导致资金损失、训练时间变慢和推理吞吐量降低。虽然可以是其他提到的组件之一，但网络通常是训练期间的瓶颈（假设你的DataLoader很快）。

如果你的模型适合单个加速器，你几乎不用担心。但如今大多数模型需要多个加速器来加载，而LLM/VLM模型需要多个计算节点进行训练，有些甚至用于推理。

大多数计算节点包含8个加速器，有些有4个，有些有16个，甚至更多加速器，最近还有一些每个节点有一个超级加速器。

当模型跨越多个加速器且不离开单个节点时，你只需关注快速的节点内网络。一旦模型需要多个节点，这通常是训练的情况，因为可以使用多个副本来并行化和加速训练，那么快速的节点间网络就成为关键。

本文涵盖了这两种类型的网络硬件，报告了它们的理论和实际带宽，并解释了它们如何相互作用。

## 术语和概念

在需要时可以安全地忽略这里列出的许多概念和缩写，然后再返回这里。

- ALU: 算术逻辑单元
- AR: 自适应路由（也可能意味着聚合路由器）
- DMA: 直接内存访问
- EFA: 弹性织物适配器
- HCA: 主机通道适配器
- IB: Infiniband
- MFU: 模型浮点运算利用率（例如，在A100上半精度时`mfu=0.5`来自于获得156TFLOPs，因为半精度的峰值规格是312TFLOPS，因此`156/312=0.5`）
- NIC: 网络接口卡
- OPA: 全路径架构
- OPX: 全路径快速
- OSFP: 八通道小型可插拔（收发器）
- RDMA: 远程直接内存访问
- RoCE: 以太网融合RDMA
- RoE: 以太网RDMA
- SHARP: 可扩展分层聚合减少协议
- VPI: 虚拟协议互连
- xGMI: 插槽到插槽全局内存接口

与速度相关的术语：
- 单向：从一个点到另一个点的单向传输 A -> B
- 双向，全双工：从一个点到另一个点的双向传输 A <-> B，通常是单向速度的2倍
- GBps, GB/s: 每秒千兆字节（1GBps = 8Gbps）在一个通道中传输
- GT/s: 每秒千兆传输 - 每秒发生的数据传输操作数
- Gbps, Gb/s: 每秒千兆位（1Gbps = 1/8GBps）在一个通道中传输
- 双向宽度：将网络分成两部分（不一定相等）所需切断的最小链路数。这些链路的带宽称为双向带宽 - 通常用作实际网络带宽的度量。有时它被称为最坏情况下的网络容量。这里有一个很好的答案(https://networkengineering.stackexchange.com/a/29662/93656)解释了这个和相关的概念，但你不太需要理解这个，只需知道它的含义，因为你的集群拓扑很可能已经由提供商完成。
- 自适应路由改进了静态路由，以便在网络上启用乱序数据包。数据包在每个交换机上进行负载均衡，以更好地分配网络工作负载。
- 远程直接内存访问

脚注：在以下部分中请注意1GBps = 8Gbps。


### 单向 vs 双向（全双工）

大多数基准测试/带宽测量工具会报告单向带宽。因此，当你查看单向与双向（全双工）速度时要小心。通常后者的速度大约是前者的2倍。

如果你在你的设置中测量带宽，发现它大约是官方标定速度的40%，请仔细检查官方标定速度是否标明是双向的，如果是，则将其减半，然后你测量的带宽现在应该大约是80%，这是预期的。

案例研究：有一段时间我无法理解为什么当我在一个A100节点上运行nccl-tests all_reduce基准测试时，官方标定的节点内速度为600GBps，而我只得到了235GBps（40%），直到Horace He好心地指出我应该查看单向速度，即300GBps，然后我得到了理论规格的80%，这就对了。


## 集群网络

集群的每个节点都有3个网络，每个网络的运行速度都非常不同。

1. 前端
2. 后端
3. 带外

### 前端网络

前端网络通常用于互联网连接（例如下载Python包和卸载到云存储）、分布式网络存储（例如检查点和数据集）和编排（例如SLURM和Kubernetes）。截至本文撰写时，典型节点可能有一个100-400Gbps的连接。

脚注：并非所有集群都可以使用外部互联网连接，例如，许多HPC环境仅通过特殊的仅CPU节点提供外部访问。

### 后端网络

后端网络用于执行GPU到GPU的连接，这使得训练和推理可以扩展到多个加速器（例如all-reduce、all-gather和其他集合通信）。这是AI集群中最重要的部分。通常这将是Infiniband或RoCEv2以太网。然后它分解为节点内网络和节点间网络 - 同一节点上的GPU通常可以比与其他节点上的GPU通信得更快。到本文撰写时，典型的最高速度大约为节点内5600Gbps和每个节点间3200Gbps。每个加速器至少有一个后端连接，有时每个加速器可能有多个连接，特别是如果使用低带宽NIC时。

脚注：并非所有提供商都能达到行业标准的网络速度 - 在某些情况下，节点间网络速度可能会慢10倍。因此，请始终检查你得到的是什么。

### 带外网络

带外（OOB）网络用于引导后端网络、监控节点的健康状况、远程重新映像节点等。它通常使用单个慢速1Gbps以太网连接。


## RDMA网络

远程直接内存访问类似于节点上的DMA（直接内存访问），但跨节点。它允许节点之间的数据交换，而无需使用本地处理器、操作系统内核和缓存的开销，这是TCP/IP使用的。主要有3种实现：

1. Infiniband
2. 收敛以太网上的RDMA（RoCE）（基于IB或UDP的RDMA）
3. iWARP（基于TCP的RDMA）

这里有一篇很好的概述文章(https://community.fs.com/article/roce-vs-infiniband-vs-tcp-ip.html)。



## 节点内网络

有多种平台/解决方案提供节点内网络：

1. 通用：PCIe
2. NVIDIA：NVLink 和 NVSwitch
3. AMD：Infinity Fabric
4. Intel：Gaudi2, Gaudi3

以下是当前解决方案的节点内单向理论点对点峰值带宽对比，按带宽排序：

| 互连技术        | 加速器       |  GBps |
| :-------------- | :---------- | ----: |
| NVIDIA NVLink 5 | B200, B*    | 900.0 |
| Intel           | Gaudi3      | 600.0 |
| NVIDIA NVLink 4 | H100, H*    | 450.0 |
| AMD XGMI        | MI300X      | 448.0 |
| AMD XGMI        | MI250X      | 350.0 |
| NVIDIA NVLink 3 | A100        | 300.0 |
| Intel           | Gaudi2      | 300.0 |
| PCIe 5          |             |  63.0 |
| PCIe 4          |             |  31.0 |

注意事项：

* NVSwitch的运行速度与同代的NVLink相同。参见NVSwitch和用于节点间的NVLink Switch。
* 请特别注意规格中提到的单向（unidirectional）和双向（bidirectional，双工）速度 - 如果你在网上看到的规格没有明确说明方向性，请寻找答案。我不得不查阅许多文档来弄清楚下表中的一些数据，因为有些供应商在发布的规格中省略了这个关键信息。我甚至不得不编辑一些维基页面以添加缺失的信息。记住，对于供应商来说，数字越大越好，所以他们几乎总是会使用双工数字，这通常是单向数字的2倍。

你将在以下各节中找到对每种技术的详细分析。

### PCIe

PCIe(https://en.wikipedia.org/wiki/PCI_Express) 是一种高速串行计算机扩展总线标准，即使在最便宜的台式电脑上也能找到。

| 互连技术 | 通道/方向 | 通道数 | 单向带宽 | 双向带宽 |
| :------ | --------: | -----: | -------: | -------: |
| PCIe 4  |  ~2.0 GBps |    16 |  31 GBps |  62 GBps |
| PCIe 5  |  ~4.0 GBps |    16 |  63 GBps | 126 GBps |
| PCIe 6  |  ~7.5 GBps |    16 | 121 GBps | 242 GBps |
| PCIe 7  | ~15.0 GBps |    16 | 242 GBps | 484 GBps |

如果比较最新一代的不同节点内网络技术（见下文各节），PCIe通常落后一个数量级。

### NVLink

- NVLink(https://en.wikipedia.org/wiki/NVLink) 是Nvidia开发的一种基于导线的多通道近程串行通信链路。这里有一个关于它的[What Is NVLink](https://blogs.nvidia.com/blog/what-is-nvidia-nvlink/)博客文章。

我发现维基页面很难理解，所以我会尝试帮助澄清这一点。

有效负载率：

| 互连技术     | 通道/方向       | 通道数 | 链接数 | 单向带宽     | 双向带宽   | GPU               |
| :----------- | -------------:   | ----: | ----: | -----------: | ---------: | :---------------- |
| NVLink 2     | 6.250 GBps       |     4 |     6 | 150 GBps     | 300 GBps   | V100              |
| NVLink 3     | 6.250 GBps       |     4 |    12 | 300 GBps     | 600 GBps   | A100              |
| NVLink 4     | 6.250 GBps       |     4 |    18 | 450 GBps     | 900 GBps   | H100, H200, GH200 |
|              |                  |       |       |              |            |                   |
|              | not sure yet     |       |       |              |            |                   |
|              | which is correct |       |       |              |            |                   |
| NVLink 5     | 6.250 GBps       |     8 |    18 | 900 GBps     | 1800 GBps  | B100, B\*, GB\*   |
| NVLink 5     | 12.50 GBps       |     4 |    18 | 900 GBps     | 1800 GBps  | B100, B\*, GB\*   |
|              |                  |       |       |              |            |                   |


NVLink 2、3 和 4 使用相同的硬件，每个链路有 4 条 6.250 GBps 的通道。每个链路的单向带宽为 25GB/s，因此每个双工链路的带宽为 50GB/s。唯一的区别在于链路的数量：

- NVLink 2 有 6 条链路 => `25* 6` => 150 GBps 单向和 300 GBps 双向
- NVLink 3 有 12 条链路 => `25*12` => 300 GBps 单向和 600 GBps 双向
- NVLink 4 有 18 条链路 => `25*18` => 450 GBps 单向和 900 GBps 双向

（等待答案）
- NVLink 5 有 18 条链路 => 900 GBps 单向和 1800 GBps 双向

最大的 PCIe 16x 插槽有 16 条通道。较小的插槽有更少的通道，1x == 1 条通道。

NVIDIA Hopper 节点通常配备 PCIe 5 和 NVLink 4。因此，NVLink 比 PCIe 快 7 倍。

NVIDIA Blackwell 节点将配备 PCIe 5 和 NVLink 5。因此，NVLink 将比 PCIe 快 14 倍。

让我们看几个节点的例子，并将理论与实际情况联系起来。

如果你使用多个 GPU，卡之间的互连方式会对总训练时间产生巨大影响。如果 GPU 在同一个物理节点上，你可以运行：

```
nvidia-smi topo -m
```

它会告诉你GPU是如何互连的。

在一台具有双GPU并通过NVLink连接的机器上，你很可能会看到类似这样的内容：

```
        GPU0    GPU1    CPU Affinity    NUMA Affinity
GPU0     X      NV2     0-23            N/A
GPU1    NV2      X      0-23            N/A
```

在没有 NVLink 的不同机器上，你可能会看到：
```
        GPU0    GPU1    CPU Affinity    NUMA Affinity
GPU0     X      PHB     0-11            N/A
GPU1    PHB      X      0-11            N/A
```

报告包括以下图例：

```
  X    = 自身
  SYS  = 连接穿越PCIe以及NUMA节点之间的SMP互连（例如，QPI/UPI）
  NODE = 连接穿越PCIe以及NUMA节点内的PCIe主桥之间的互连
  PHB  = 连接穿越PCIe以及PCIe主桥（通常是CPU）
  PXB  = 连接穿越多个PCIe桥（不穿越PCIe主桥）
  PIX  = 连接最多穿越一个PCIe桥
  NV#  = 连接穿越一组绑定的# NVLinks
```

第一个报告中的 `NV2` 告诉我们 GPU 之间通过 2 条 NVLink 互连，而第二个报告中的 `PHB` 则是典型的消费级 PCIe+桥接设置。

检查你的系统中使用的连接类型。有些连接类型会使卡之间的通信更快（例如 NVLink），而其他类型则会更慢（例如 PHB）。

根据所使用的扩展解决方案类型，连接速度可能会产生重大或轻微的影响。如果 GPU 需要很少同步，如在 DDP 中，较慢连接的影响将不太显著。如果 GPU 需要频繁互相发送消息，如在 ZeRO-DP 中，那么更快的连接对于实现更快的训练变得非常重要。

现在，让我们看看 A100 和 H100 节点的拓扑结构：


- A100 拓扑结构：

```
$ nvidia-smi topo -m
      GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7  CPU Affinity  NUMA Affinity
GPU0   X    NV12  NV12  NV12  NV12  NV12  NV12  NV12   0-23         0
GPU1  NV12   X    NV12  NV12  NV12  NV12  NV12  NV12   0-23         0
GPU2  NV12  NV12   X    NV12  NV12  NV12  NV12  NV12   0-23         0
GPU3  NV12  NV12  NV12   X    NV12  NV12  NV12  NV12   0-23         0
GPU4  NV12  NV12  NV12  NV12   X    NV12  NV12  NV12  24-47         1
GPU5  NV12  NV12  NV12  NV12  NV12   X    NV12  NV12  24-47         1
GPU6  NV12  NV12  NV12  NV12  NV12  NV12   X    NV12  24-47         1
GPU7  NV12  NV12  NV12  NV12  NV12  NV12  NV12   X    24-47         1
```
你可以看到有 12 条 NVLink 和 2 个 NUMA 组（每个 CPU 有 24 个核心）

- H100 拓扑结构：
```
$ nvidia-smi topo -m
      GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7  CPU Affinity  NUMA Affinity
GPU0   X    NV18  NV18  NV18  NV18  NV18  NV18  NV18   0-51         0
GPU1  NV18   X    NV18  NV18  NV18  NV18  NV18  NV18   0-51         0
GPU2  NV18  NV18   X    NV18  NV18  NV18  NV18  NV18   0-51         0
GPU3  NV18  NV18  NV18   X    NV18  NV18  NV18  NV18   0-51         0
GPU4  NV18  NV18  NV18  NV18   X    NV18  NV18  NV18  52-103        1
GPU5  NV18  NV18  NV18  NV18  NV18   X    NV18  NV18  52-103        1
GPU6  NV18  NV18  NV18  NV18  NV18  NV18   X    NV18  52-103        1
GPU7  NV18  NV18  NV18  NV18  NV18  NV18  NV18   X    52-103        1
```
你可以看到有 18 条 NVLink 和 2 个 NUMA 组（每个 CPU 有 52 个核心）

当然，其他 A100 和 H100 节点的报告可能会有所不同，例如不同的 CPU 核心数量。

### NVSwitch

NVSwitch(https://www.nvidia.com/en-us/data-center/nvlink/) 可以以 NVLink 的速度连接超过 8 个 GPU。它被宣传为在未来几代中可以连接多达 256 个 GPU。

连接更多 GPU 以 NVLink 速度的好处是它允许所有 GPU 之间的通信速度比任何节点内硬件所能提供的速度快得多。随着计算速度的不断增加，网络可能是导致 GPU 利用率不足的瓶颈，导致 GPU 利用率不足。

例如，在张量并行（Megatron）的宇宙中，人们不会使用超过 8 的 TP 度，因为 TP 仅在 NVLink 速度下有效。ZeRO-DP（Deepspeed/FSDP）如果整个集群使用 NVLink 速度并且没有慢速的节点间连接，则会快得多。

有两种 NVSwitch：
1. 用于节点内连接（L1）的 NVSwitch
2. 用于节点间连接（L2）的 NVLink Switch

NVSwitch gen 1 在 V100 上发布，gen 2 在 A100 上发布，gen 3 在 H100 上发布 - 速度对应于相同技术的 NVLink 版本。

NVIDIA DGX H100(https://developer.nvidia.com/blog/upgrading-multi-gpu-interconnectivity-with-the-third-generation-nvidia-nvswitch/) 有 3.6 TBps 的全双工 NVLink 网络带宽，由 72 条 NVLink（NVLink 4）提供。标准的 NVLink 4 有 18 条 NVLink（0.9 TBps 全双工）。因此，这个设置有 4 个交换机（`18*4=72`），因此 `0.9*4=3.6` TBps。注意，这台服务器有 8 个 GPU，所以这里我们得到了比标准 NVLink 4.0 更快的节点内通信，后者仅提供 0.9 TBps 的全双工连接。

NVIDIA DGX A100 有 6 个 12 条 NVLink 的交换机，总共 72 条。

DGX H100 SuperPOD(https://developer.nvidia.com/blog/upgrading-multi-gpu-interconnectivity-with-the-third-generation-nvidia-nvswitch/) 结合了 32 台 DGX H100 服务器，总共 256 个 GPU。看起来这里他们只使用了单个 DGX H100 的一半 NVLink，所以每个节点只有 1.8 TBps，总共 57.6 TBps。

此外，NVSwitch gen3 及更高版本附带 NVIDIA 可扩展分层聚合减少协议 (SHARP)，可以提高节点内和节点间的速度。例如，NCCL 正在研究 `NCCL_ALGO=NVLS`，这已经将节点内带宽提高到正常规格之上，并且根据本文撰写时，也在进行提升节点间带宽的工作。


### Infinity Fabric / xGMI

AMD MI* 加速器的节点内通信是通过 AMD Infinity Fabric 完成的，也被称为 xGMI（Socket to Socket Global Memory Interface）。

这是 AMD 对 NVLink 的回应。

以下是全互联带宽。

|               | peer-to-peer   |       | all-to-all   | all-to-all |
| :------------ | -------------: | ----: | -----------: | ---------: |
| MI375X        | 64 GBps        |     7 | 448 GBps     | 896 GBps   |
| MI350X        | 64 GBps        |     7 | 448 GBps     | 896 GBps   |
| MI300X        | 64 GBps        |     7 | 448 GBps     | 896 GBps   |
| MI250X        | 50 GBps        |     7 | 350 GBps     | 700 GBps   |

对等带宽只是单个链路/方向的带宽（第 2 列）。

其他节点内解决方案通常具有相同的对等和全互联带宽，因此 Infinity Fabric 似乎明显较慢。我猜这是因为这些主要是为推理创建的，因为这些慢速的速度会显著降低 LLM 训练的速度。

![AMD Infinity Platform Architecture](https://files.mdnice.com/user/59/ffc93caa-c011-41be-9e19-3fc04f43a6c9.png)

Platform 描述:
- MI250X(https://www.amd.com/en/products/accelerators/instinct/mi200/mi250x.html)
- MI300x(https://www.amd.com/en/products/accelerators/instinct/mi300/platform.html)

### Gaudi2

根据 Gaudi2 规格(https://habana.ai/wp-content/uploads/2023/10/HLS-Gaudi2_Datasheet_10_23.pdf)，这些节点提供相同的 100GbE RoCE v2 RDMA 硬件，用于节点内和节点间连接（每张卡 24x 100Gbps）。

- 节点内：8x 7x3 NICs - 卡到卡 300Gbps
- 节点间：8x 1x3 NICs - 总共 2.4Tbps（300GBps）

### Gaudi3

根据 Gaudi3 规格(https://www.intel.com/content/www/us/en/content-details/817486/intel-gaudi-3-ai-accelerator-white-paper.html)，这些节点的设置与 Gaudi2 相同，只是卡的速度提高了 2 倍，使用 200GbE RoCE v2 RDMA 进行节点内和节点间连接（每张卡 24x 200Gbps）。

- 节点内：8x 7x3 NICs - 卡到卡 600Gbps
- 节点间：8x 1x3 NICs - 总共 4.8Tbps（600GBps）

## 节点间网络

由于节点间硬件曾经比节点内硬件慢一个数量级，在这个领域中使用 Gbps 而不是 GBps。（1 GBps = 8 Gbps）（尽管最近节点间速度几乎和节点内一样快）

在节点间网络硬件方面，有 NVIDIA 的成熟 InfiniBand 和其他一些厂商的产品，各种基于 NVLink 的 NVIDIA 产品，还有许多新兴厂商，主要来自计算云提供商，他们无法在租用他人硬件的微薄利润中竞争，所以他们自己构建硬件（AWS EFA，Google GPUDirect-TCPX），还有 HPE 和 Cornelis Networks 最近更新的产品。

以下是当前技术的节点间单向理论峰值带宽对比，按常见节点设置的总带宽排序：

| Interconnect              | NICs x Gbps | Total GBps | Notes   |
| :-------------------      | ----------: | ---------: | :------ |
| Intel Gaudi3              |      24x200 |        600 |         |
| NVIDIA NVLink Switch gen3 |       8x450 |        450 | H100    |
| NVIDIA Quantum-2 IB       |       8x400 |        400 | H100    |
| AWS EFA v2                |      32x100 |        400 | H100    |
| NVLink Switch gen2        |       8x300 |        300 | A100    |
| Intel Gaudi2              |      24x100 |        300 |         |
| InfiniBand XDR1600        |       8x200 |        200 |         |
| NVIDIA NVLink Switch gen1 |       8x150 |        150 | V100    |
| Intel GPUDirect-TCPX      |       4x200 |        100 |         |
| HPE Slingshot             |       4x200 |        100 |         |
| Omni-Path CN100           |       8x100 |        100 |         |
| AWS EFA v1                |       4x100 |         50 |         |
| InfiniBand NDR400         |       4x100 |         50 |         |
|                           |             |            |         |
| in the future:            |             |            |         |
|                           |             |            |         |
| Omni-Path CN5000          |       8x400 |        400 | Q3-2024 |
| InfiniBand GDR3200        |       8x400 |        400 | 2025    |
| Omni-Path CN6000          |       8x800 |        800 | 2026    |

注意：

* 这些是常见/流行的节点设置 - 一些自定义节点可能有不同的配置，通常是更少的网卡，很少有更多网卡的情况。是的，AWS EFA v2 在每个节点上放置了32个网卡 - 那一定是很多线缆。
* 注意曾经存在的节点间和节点内带宽之间的数量级差异正在消失 - 我最近将这里的速度从Gbps重新调整为GBps。

你将在以下部分找到每种技术的详细分析。

### NVLink Switch

虽然NVSwitch（第1层交换机）用于节点内通信（L1），但NVLink Switch（第2层交换机）用于节点间通信。这些链路使用与NVLink相同的速度 - 所以当两者都被使用时，节点间和节点内的每链路带宽是相同的。

关于实际带宽，请参见NVLink部分。

NVLinks Switch gen3用NVLink Network替代了普通网络（https://developer.nvidia.com/blog/upgrading-multi-gpu-interconnectivity-with-the-third-generation-nvidia-nvswitch/）

### InfiniBand

InfiniBand(https://en.wikipedia.org/wiki/InfiniBand)（IB）已经存在了几十年，所以可以找到许多可用的配置。因此，如果有人说他们有InfiniBand，这是不够的信息。你需要知道的是信号速率和IB链路的数量。

InfiniBand是一个完整的网络协议，实现了RDMA（绕过TCP/IP）。

以下是你可能在当前硬件产品中看到的最新信号速率：

单向链路的信号速率（单位：Gbps）：
| Links | EDR | HDR |  NDR |  XDR |  GDR |  LDR |
| ----: | --: | --: |  --: |  --: |  --: |  --: |
|     1 |  25 |  50 |  100 |  200 |  400 |  800 |
|     4 | 100 | 200 |  400 |  800 | 1600 | 3200 |
|     8 | 200 | 400 |  800 | 1600 | 3200 | 4800 |
|    12 | 300 | 600 | 1200 | 2400 | 4800 | 9600 |

注意：
* GDR计划在2025年推出，LDR将在几年后推出

延迟（单位：微秒）：
| EDR | HDR | NDR | XDR | GDR | LDR |
| --: | --: | --: | --: | --: | --: |
| 0.5 | 0.6 | ??  | ??  | ??  | ??  |

`??` = NDR及之后的版本未公布延迟数据

InfiniBand提供RDMA（远程直接内存访问）(https://en.wikipedia.org/wiki/Remote_direct_memory_access)。

以下是一些使用最快IB的NVIDIA设备示例：

- NVIDIA DGX H100的一种配置配备了8个NVIDIA ConnectX-7（CX7）以太网/InfiniBand端口，每个端口200Gbps，总计1.6 Tbps，用于与其他DGX服务器连接。
- 对于DGX H100 SuperPOD，所有32台DGX服务器和相关InfiniBand交换机上的ConnectX-7提供25.6 TBps的全双工带宽，用于pod内部使用或扩展多个SuperPOD - 这相当于每个节点0.8 TBps（6.4Tbps！）。
- 基于NVIDIA GB200的解决方案将通过Quantum-2 InfiniBand 800G交换机（2x400G NDR接口）提供400Gbps或800Gbps NDR。

根据维基百科，虽然InfiniBand(https://en.wikipedia.org/wiki/InfiniBand)曾经有多个制造商，但目前只有英特尔（收购了QLogic）和NVIDIA（收购了Mellanox）。另请参阅InfiniBand贸易协会(https://www.infinibandta.org/)。

实用链接：
- InfiniBand实用工具(https://docs.nvidia.com/networking/display/ofedv512580/infiniband+fabric+utilities)（链接可能已过时，因为它是版本化的）- 这些在调试IB设置时很有用。

### NVIDIA Quantum-2 InfiniBand

NVIDIA Quantum-2 InfiniBand平台(https://www.nvidia.com/en-us/networking/quantum2/)支持每链路400Gbps带宽，提供RDMA，包括带有SHARP的网内计算，支持PCIe-5。

交换机可以以400Gbps的速度连接64个设备。

除了NVLink Switch，这是目前业界最快的H100节点所使用的唯一其他技术。



### EFA

弹性结构适配器（EFA）(https://aws.amazon.com/hpc/efa/) 是由AWS创建的一种最新的节点间网络技术。

- EFA v1 0.4 Tbps（all_reduce测试的有效带宽为340 Gbps）（AWS P4实例）
- EFA v2 3.2 Tbps（自2023年第三季度起，AWS P5实例 - 32个网卡！）


### Gaudi2（节点间）

根据Gaudi2规格(https://habana.ai/wp-content/uploads/2023/10/HLS-Gaudi2_Datasheet_10_23.pdf)，这些节点提供`3*8=24`个100GbE RoCE v2 RDMA网卡，总计2.4Tbps的节点间连接带宽，用于与其他Gaudi2节点连接。


### Gaudi3（节点间）

根据Gaudi3规格(https://www.intel.com/content/www/us/en/content-details/817486/intel-gaudi-3-ai-accelerator-white-paper.html)，这些节点提供`3*8=24`个200GbE RoCE v2 RDMA网卡，总计4.8Tbps的节点间连接带宽，用于与其他Gaudi3节点连接。


### HPE Slingshot互连

HPE Slingshot互连(https://www.hpe.com/ca/en/compute/hpc/slingshot-interconnect.html)似乎被高性能计算（HPC）所使用。截至本文撰写时，它每个链路提供200Gbps的带宽。一些HPC使用4个这样的链路来构建800Gbps的互连，当然，使用更多链路将提供更高的总体带宽。



### GPUDirect-TCPX

GPUDirect-TCPX是GCP的A3实例中引入的一种新的硬件/软件网络堆栈。文档很少，但这里有一些信息(https://cloud.google.com/compute/docs/gpus/gpudirect)。



### Omni-Path

Omni-Path架构(https://en.wikipedia.org/wiki/Omni-Path)（OPA）。最初由英特尔开发，该技术后来被出售给Cornelis Networks。它也被称为Omni-Path Express（OPX）。

案例研究：我在2022年在法国的JeanZay HPC上使用了这项技术。当时它只有135Gbps的速度，尽管供应商一年后尝试修复，但速度仍然相同。希望现在这个问题已经解决，速度已经大大提高。由于速度太慢，我们不得不使用Megatron-Deepspeed(https://github.com/bigscience-workshop/Megatron-DeepSpeed)来训练BLOOM-176B，而不是使用更容易的DeepSpeed ZeRO。

截至本文撰写时，我看到该产品提供100或200Gbps的带宽。因此，除非他们设法安装了许多网卡，否则你不太可能看到有人为ML工作负载提供这种解决方案。

[CN-100](Cornelis Omni-Path Accelerated Host Fabric Adapter CN-100HFA) 100Gbps网卡已经存在多年了。

CN5000(https://www.cornelisnetworks.com/solutions/cornelis-cn5000/) 400Gbps网卡将于2024年第三季度由Cornelis Networks推出。一个即将推出的MI300X设置使用8个这样的网卡，总计3200Gbps的单向节点间带宽。

Omni-Path提供RDMA(https://en.wikipedia.org/wiki/Remote_direct_memory_access)。


### Ultra Accelerator Link (UALink)

UALink倡议(https://www.google.ca/search?q=Ultra+Accelerator+Link)是一种尝试创建一个开放标准以与NVLink竞争。据说它将基于AMD的Infinity Fabric。截至本文撰写时，还没有实际的硬件可言。


## 其他重要的网络技术

### SHARP

NVIDIA可扩展分层聚合和归约协议（SHARP）(https://docs.nvidia.com/networking/display/sharpv300) - 允许在网络本身上执行数据归约和聚合（网内计算）。如果你进行大量的MPI、NCCL和其他支持SHARP的网络集体操作，这非常有用，因为这些操作的延迟应该会大大改善。

要理解这项技术的重要性 - 对于all-reduce操作，它将只需要N+1次发送，而不是2N次发送 - 所以对于大的N来说，它几乎将有效的all-reduce吞吐量翻倍。（N是通信的ranks/gpus的数量）。有关详细信息，请参见all-reduce操作兼容性(https://developer.nvidia.com/blog/upgrading-multi-gpu-interconnectivity-with-the-third-generation-nvidia-nvswitch/)（你需要向下滚动到该部分）。

最新的NCCL版本如果检测到可用，将自动使用这项技术。

作为NVSwitch或Infiniband交换机一部分的SHARP硬件包括算术逻辑单元（ALU），直接执行计算而不是使用GPU。据说它可以在FP64、FP32、FP16和BF16数据类型中执行数学运算。

案例研究：我在进行H100节点内NVLink 4.0 all-reduce(benchmarks/all_reduce_bench.py)基准测试时意外发现了SHARP，当4GB负载报告480GBps时，而理论规格只有450GBps！我们发现这是因为NCCL开启了新的`NVLS`算法，因为它检测到了Infiniband SHARP。我仍然不理解它如何达到比物理介质允许的更快的速度。我很确定那里的`busbw`计算算法需要从2N调整为N+1以获得真实速度。这里有一个详细的讨论(https://github.com/NVIDIA/nccl-tests/issues/153#issuecomment-1628415956)。结论：`busbw`可能会也可能不会给你真实的带宽数字，这取决于NCCL选择使用的`algo`，只有当使用`Ring`算法时，`busbw`才是正确的。

## 理解为什么节点间网络速度如此重要

这可能是你真正需要很好理解的最重要的多段部分之一。虽然它旨在展示节点间速度的重要性，但在构建案例的过程中，它还会教授许多重要的与训练相关的概念。

### 基础知识

首先，让我们对所有这些Gbps/GBps的实际含义有一些感觉。

如果你的模型有800亿参数，并且你需要在网络上传输每个参数或梯度一次，使用float32（fp32）格式，每个参数需要4字节，那么你需要发送`80*4` 320GB的数据，或2560Gb（`*8`）。如果你的网络带宽是200Gbps，传输将需要12.8秒（`2560/200`）。而如果你有1600Gbps的网络，那么只需要1.6秒。为什么这很重要？

### 单GPU训练

让我们从一个小得多的模型开始，比如说20亿参数，要训练它，你在混合半精度下至少需要每个参数18字节(https://github.com/stas00/ml-engineering/blob/master/training/performance/README.md#anatomy-of-models-memory-usage)。所以仅仅用于模型权重、优化器状态和梯度就需要`18*2` 36GB的内存。此外，你还需要额外的内存用于激活，这将取决于批量大小和序列长度。但是使用80GB的A100 GPU，我们肯定可以在单个GPU上训练这个模型。

我们暂时假设DataLoader的速度足够快，相比计算时间可以忽略不计。因此，我们得到了接近完美的MFU（模型FLOP利用率）：

```
[DL][  compute  ][DL][  compute  ][DL][  compute  ]
---------------------------------------------------> time
|<--iteration-->||<--iteration-->||<--iteration-->|
```

这意味着GPU只需要进行大量的矩阵乘法运算，而且它会以惊人的速度完成。在这种情况下，你获得了最高的投资回报率（ROI）。

### 单节点训练

之前的情况由于接近完美的MFU而非常理想，但你会意识到在单个GPU上的训练将会花费相当长的时间。由于我们正处于AI竞赛中，你可能希望尽快完成训练。所以你会问 - 我能否在8个GPU上训练模型呢？答案是 - 当然可以。但有一个注意事项 - 在每次迭代结束时，你需要在8个进程（每个GPU一个进程）之间同步梯度，这样每个参与训练的进程都能从其他7个进程在上一次迭代中学到的东西中受益。

注：当然，你可以使用少于8个GPU，只是现在大多数基于NVIDIA GPU的计算节点都有8个GPU，所以为什么不获得最佳投资回报呢？

注：在理想世界中，在1个GPU上训练8个时间单位，应该与在8个GPU上训练1个时间单位的成本相同。人们会期望花费相同的金钱并且完成速度快8倍。但由于数据同步的要求，实际情况并非如此。

如果实验模型仍然像上一节中那样包含20亿参数，且梯度以fp32格式存储，那么训练程序需要在每次迭代中发送8GB（20亿 * 4字节）的数据。此外，由于同步梯度需要执行all_reduce集合通信操作，它需要传输两次数据 - 首先是每个GPU发送梯度数据，计算梯度总和，然后将这个值发送回每个参与的GPU，这样每个训练进程都能从其对等进程在上一次迭代中所做的学习进展中受益。

以下是all-reduce集合通信操作的可视化：

![](https://files.mdnice.com/user/59/2c80cce0-2c65-4db0-b03d-43ae8b0134be.png)

(source(https://pytorch.org/tutorials/intermediate/dist_tuto.html#collective-communication))

所以我们需要发送8GB两次，这意味着我们需要发送16GB的数据。

注释：准确地说，all-reduce的2倍通信量实际上是`2*(n-1)/n`，其中n是参与的GPU数量。所以如果n=2，系数就是1，因为`2*(2-1)/2=1`；对于n=8，系数是1.75，因为`2*(8-1)/8=1.75`；当n=64时，它已经非常接近2了。

注释：还有一个重要的问题是网络延迟 - 由于数据从所有参与的GPU收集的方式，延迟被放大了几倍。但是，考虑到这里我们移动的是非常大的有效载荷，延迟贡献的开销非常小，为了简单起见可以忽略不计。

发送16GB数据需要多长时间？

- A100 @ 300GBps：`16/300` = 0.053秒
- H100 @ 450GBps：`16/450` = 0.035秒

这是非常快的！

以下是我们的时间线会是什么样子：

```
[DL][  compute ][comms][DL][  compute ][comms][DL][  compute ][comms]|
-----------------------------------------------------------------------> time
|<---- iteration ---->||<---- iteration ---->||<---- iteration ----->|
```

哦，这整个同步协议在PyTorch术语中被称为DDP（DistributedDataParallel，分布式数据并行）(https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)。

#### 通信和计算重叠

即使通信速度如此之快，网络仍然会造成瓶颈，导致GPU短暂空闲。为了解决这个问题，高级算法实现了通信和计算的重叠。到目前为止，我们将问题视为单次传输，但实际上每个模型由许多层组成，每一层都可以在下一层计算其梯度的同时传输它已经计算好的梯度。因此，如果你从模型层面来看，在`backward`路径中发生的情况是：


```
[   compute   ][   compute   ][   compute   ]
               [comms]        [comms]        [comms]
---------------------------------------------> time
<- layer -1 ->|<- layer -2 ->|<- layer -3 ->|
```

因此，一旦最后一层（-1）计算完其梯度，它就会进行all-reduce操作，同时倒数第二层执行其`backward`，以此类推，直到第一层完成梯度计算并最终发送出其梯度。

现在你已经理解了重叠是如何工作的，我们可以更新我们的大局图为：

现在我们的时序图变得非常类似于我们之前对单个GPU的图表：

```
[DL][  compute  ][DL][  compute  ][DL][  compute  ]
[  comms ]       [  comms]        [  comms]
---------------------------------------------------> time
|<--iteration-->||<--iteration-->||<--iteration-->|
```

我们希望通信速度比数据加载和计算更快，因为如果通信速度不够快，我们就会出现以下GPU空闲间隙：

```
[DL][  compute  ][idle][DL][  compute  ][idle][DL][  compute  ][idle]
[         comms       ][         comms       ][         comms       ]
----------------------------------------------------------------------> time
|<---  iteration  --->||<---  iteration  --->||<---  iteration  --->|
```

#### 计算TFLOPS

计算TFLOPS回答了执行计算需要多长时间的问题。

这里在术语上有一些混淆，因为TFLOPS中最后的`s`有时表示`秒`，有时仅表示`操作`。

例如，当你阅读A100规格(https://www.nvidia.com/en-us/data-center/a100/#specifications)时，那里的TFLOPS表示每秒万亿次浮点运算。

所以让我们准确定义这些缩写：

- TFLOPS - 每秒万亿次浮点运算（另一种写法是TFLOP/s）
- TFLOP - 万亿次浮点运算（或TFLOPs - 小写`s`但已经很混淆了）

更多说明请参见维基页面(https://en.wikipedia.org/wiki/FLOPS)。

对于GPT系列的解码器transformer模型，我们可以使用BLOOM-176文档(https://github.com/bigscience-workshop/bigscience/tree/master/math#calculate-tflops)中描述的数学公式：

以下是每秒处理的TFLOP数量：


这个公式假设使用激活重计算(https://github.com/stas00/ml-engineering/blob/master/training/performance/README.md#gradient-checkpointing)，这会在引入少量开销的同时节省GPU内存。如果不用它，则将`4`替换为`3`，因为模型只需要在`forward`中进行1x计算，在`backward`中进行2x计算（因为梯度计算了两次 - 一次用于输入，一次用于权重）。使用激活重计算时，`forward`会进行两次，因此你有一个额外的路径，导致乘数为`4`而不是`3`。

注释：激活重计算和梯度检查点都指的是相同的技术。

所以让我们去掉时间成分，这将给我们总的TFLOP：

```
tflop = model_size_in_B * 4 * 2 * seqlen * global_batch_size / (total_gpus * 1e3)
```

所以假设我们有：
- `seqlen=2048` (sequence length)
- `global_batch_size=16`

并且我们已经定义了：
- `total_gpus=8`
- `model_size_in_B=2`

这将给我们：

```
tflop = 2 * 4 * 2 * 2048 * 16 / (8 * 1e3) = 65.536 TFLOP
```

所以如果我们进行混合半精度训练，并且大多数操作都在半精度中完成，那么我们可以粗略地说我们在A100上执行了312 TFLOPS(https://www.nvidia.com/en-us/data-center/a100/#specifications)，并且通常一个优化良好的框架在一个优化良好的硬件上至少能达到50%的MFU - 也就是说它能够以大约一半的峰值性能进行计算。

注释：在H100上，它是一个~3x 989 TFLOPS(https://www.nvidia.com/en-us/data-center/h100)（向下滚动到末尾），并且它显示了一个误导性的2x稀疏性，所以你必须在心里除以2。

因此，继续这个思路，这意味着该设置将有大约156TFLOPS - 所以如果我们忽略DataLoader的开销（我们希望它接近即时），处理单次迭代（2次`forward`和2次`backward`计算）将需要0.42秒。

早些时候我们说过，典型的A100节点有300GBps的节点内NVLink连接，因此我们说发送16GB的梯度将花费`16/300` = 0.053秒。

我们测量的计算时间为0.42秒，所以在这里网络不是瓶颈，因为`0.42 > 0.053`，所以计算会比通信慢。

现在你可以进行几个思想实验 - 例如，如果你将批量大小或序列长度减半，你将使计算时间减半。

注释：这是一个非常粗略的建议，因为GPU在乘以巨大矩阵时工作得最快。但对于我们在这里进行的简化思想实验来说，这已经足够好了。在现实中，将维度减半不会使计算时间减半。

好的，但希望到这一点，很明显如果你保持在单个节点的边界内，你不需要担心你的GPU空闲。

但如果你想进一步加速训练，比如说使用4个8-gpu节点呢？（当然，如果你有一个更大的模型，你别无选择，只能使用多个节点）。突然间，通信可能成为一个更大的瓶颈。



### 多节点训练

在这里，我们继续使用2B参数模型的想法，现在我们将使用跨越4个节点的32个GPU来进一步加速训练。

虽然每组8个GPU仍然通过超快速的NVLink技术连接，但节点间的连接通常要慢一个数量级。

假设你有一个200Gbps的连接。让我们重复前一节的计算，看看reduce 16GB梯度需要多长时间。

16GB等于128Gb，所以在200Gbps的速度下，这将需要0.64秒。

如果我们坚持计算时间为0.42秒，在这里我们最终会发现通信时间比计算时间长，因为`0.64 > 0.42`。

让我们把两种用例放在一起比较：

| nodes | comms | compute | comms is a bottleneck |
|-------|-------|---------|-----------------------|
|     1 | 0.027 |    0.42 | no                    |
|     4 |  0.64 |    0.42 | yes                   |

在这个200Gbps的节点间设置中，通信速度比在节点内NVLink连接上执行的速度慢23倍。

所以在这个特定的设置中，如果你能够获得400Gbps的节点间连接，速度将翻倍，通信将在0.32秒内完成，因此将比计算时间0.42秒更快。

注释：你永远无法在应用程序级别获得官方规定的完全速度，所以如果官方规定为400Gbps，在最佳情况下，预期获得320Gbps（约80%）。所以也要考虑到这一点。此外，根据每次合通信的负载大小，负载越小，实际网络吞吐量越小。

记住，这都是在处理一个相当小的模型，考虑的是2B参数模型。

现在用20B和200B参数模型做同样的数学运算，你会看到你需要有一个快得多的节点间连接才能有效地扩展。

### 大型模型训练

当然，当我们训练大型模型时，我们不使用DDP，因为我们根本无法将整个模型放入单个GPU中，所以会使用各种其他技术。详细内容在专门的模型并行化章节(https://github.com/stas00/ml-engineering/tree/master/training/model-parallelism)中讨论，但现在需要立即理解的重要一点是，所有可扩展性技术都会产生更大的通信开销，因为它们需要传输的不仅仅是梯度。因此，网络上的流量很容易增长到我们目前探讨的DDP协议开销的3倍或更多。

要像我们在本章中那样进行近似计算是很困难的，因为实际计算时间取决于所选框架的效率、调优程度、DataLoader提供批次的速度以及许多其他因素，因此没有标准的MFU可以用于计算，你将在配置和运行大型模型训练的前几步时发现你的MFU。然后你可以阅读性能章节(https://github.com/stas00/ml-engineering/tree/master/training/performance)并进一步提高你的MFU。

正如我在这些部分所展示的，一旦你理解了特定的可扩展性技术及其网络成本，就应该能够进行粗略计算，这样你就可以提前知道需要向采购经理要求什么样的节点间网络速度。当然，你还需要理解特定的模型架构，并计算完成单次迭代需要多少TFLOP。

## 重要细节

### 实际网络吞吐量

广告宣传的网络吞吐量规格和实际吞吐量永远不会相同。在最好的情况下，你可以期望达到广告宣传规格的约80-90%。

然后，网络吞吐量将取决于每次通信期间发送的有效负载大小。有效负载越大，吞吐量就越高。

让我们使用nccl-tests(https://github.com/NVIDIA/nccl-tests)在单个A100节点上演示这一点
```
$ ./build/all_reduce_perf -b 32k -e 16G -f 2 -g 8 -n 50
[...]
           size    time   algbw   busbw
            (B)    (us)  (GB/s)  (GB/s)
         32_768   43.83    0.75    1.31
         65_536   46.80    1.40    2.45
        131_072   51.76    2.53    4.43
        262_144   61.38    4.27    7.47
        524_288   80.40    6.52   11.41
       1048_576   101.9   10.29   18.00
       2097_152   101.4   20.68   36.18
      4_194_304   101.5   41.33   72.33
      8_388_608   133.5   62.82  109.93
     16_777_216   276.6   60.66  106.16
     33_554_432   424.0   79.14  138.49
     67_108_864   684.6   98.02  171.54
    134_217_728  1327.6  101.10  176.92
    268_435_456  2420.6  110.90  194.07
    536_870_912  4218.4  127.27  222.72
  1_073_741_824  8203.9  130.88  229.04
  2_147_483_648   16240  132.23  231.41
  4_294_967_296   32136  133.65  233.88
  8_589_934_592   64074  134.06  234.61
 17_179_869_184  127997  134.22  234.89
```

注释：我对输出进行了处理，删除了不需要的列，并使大小更易读。

这个基准测试对32KB到16GB的各种有效载荷大小运行了`all_reduce`集合操作。我们关心的值是`busbw` - 这一列告诉我们实际的网络吞吐量，如这里所解释的(https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#bus-bandwidth)。

正如你所看到的，对于小于8MB的有效载荷，吞吐量非常低 - 并且在有效载荷大小约为536MB时开始饱和。这主要是因为延迟。减少单个4GB有效载荷比减少1000个4MB有效载荷要快得多。

这里有一个基准测试演示了这一点：all_reduce_latency_comp.py(https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/all_reduce_latency_comp.py)。让我们在同一个A100节点上运行它：

```
$ python -u -m torch.distributed.run --nproc_per_node=8 all_reduce_latency_comp.py

----------- 1x 4.0GB ----------------
 busbw: 1257.165 Gbps

----------- 1000x 0.004GB ----------------
 busbw: 374.391 Gbps
```

很容易看出，在这种情况下，发送相同的有效载荷但以1000个更小的块发送，速度要慢3倍。

因此，当您计算`all_reduce`给定有效载荷大小需要多长时间时，您需要使用相应的`busbw`条目（当然，您已经在您的特定硬件/环境中运行了此基准测试）。

弄清楚有效载荷可能很棘手，因为它取决于框架的实现。一些实现将每个权重的梯度单独减少，这显然会导致非常小的有效载荷，并且网络会非常慢。其他实现在减少之前将多个梯度组合在一起，增加有效载荷并最小化延迟影响。

但让我们回到基准测试结果表。此测试在运行NVLink的A100节点上进行，广告宣传为uni-directional 300GBs，因此我们获得了约78%的理论速度，有效载荷为17GB，超过该值基准测试崩溃。从表的最后几行可以看出，没有太多可以挤压的。

我们还可以运行p2pBandwidthLatencyTest(https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/p2pBandwidthLatencyTest)，该测试执行低级别的p2p基准测试：
单向300GBs，所以我们在17GB有效载荷时获得了理论速度的约78%，超过这个值基准测试就会崩溃。从表格的最后几行可以看出，已经没有太多可以再挤压的了。

我们还可以运行p2pBandwidthLatencyTest(https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/p2pBandwidthLatencyTest)，它执行低级别的点对点基准测试：

```
./p2pBandwidthLatencyTest
[...]
Unidirectional P2P=Enabled Bandwidth (P2P Writes) Matrix (GB/s)
   D\D     0      1      2      3      4      5      6      7
     0 1581.48 274.55 275.92 272.02 275.35 275.28 273.62 273.20
     1 274.70 1581.48 275.33 272.83 275.38 273.70 273.45 273.70
     2 274.81 276.90 1594.39 272.66 275.39 275.79 273.97 273.94
     3 273.25 274.87 272.12 1545.50 274.38 274.37 274.22 274.38
     4 274.24 275.15 273.44 271.57 1584.69 275.76 275.04 273.49
     5 274.37 275.77 273.53 270.84 274.59 1583.08 276.04 273.74
     6 275.61 274.86 275.47 273.19 272.58 275.69 1586.29 274.76
     7 275.26 275.46 275.49 273.61 275.50 273.28 272.24 1591.14
[...]
```

正如你在报告的单向部分所看到的,我们从宣传的300GBps中实际获得了274 GBps(约91%)。

请注意,当我在H100s(NVLink 4.0)上重新运行这个相同的测试时,我得到了更差的效率:

```
Unidirectional P2P=Enabled Bandwidth (P2P Writes) Matrix (GB/s)
   D\D     0      1      2      3      4      5      6      7
     0 2494.51 364.13 375.99 378.03 376.77 376.71 374.85 375.66
     1 375.18 2533.95 376.08 374.98 376.21 375.96 375.76 375.12
     2 363.43 393.28 2532.67 376.35 377.14 376.47 375.76 375.48
     3 369.90 375.92 393.63 2525.38 376.58 375.88 376.13 377.01
     4 376.20 376.28 375.20 393.52 2526.02 375.82 375.05 376.10
     5 376.26 376.60 375.54 375.52 376.81 2521.18 376.37 376.60
     6 374.31 376.19 376.80 376.32 376.83 376.44 2529.85 376.39
     7 376.17 376.49 376.53 374.95 376.30 376.82 375.71 2519.78
```

所以376GBps out of 450GBps是83%（不太好）。

底线 - 在这个特定的设置中：
1. 如果你有非常大的有效载荷，你将能够使用大约80%的广告宣传300GBps
2. 如果每次通信的有效载荷较小，则可能要低得多。

这张图也很有帮助，可以演示实际带宽如何随着消息大小的变化而变化：

![Low-level Uni-directional Bandwidth Measurements](https://files.mdnice.com/user/59/71953d7d-5fd1-4be2-ac6f-dec4dafe7bbd.png)


(source(https://ieeexplore.ieee.org/document/5238655))

另一个用于NVIDIA GPU带宽测量的工具是NVIDIA/nvbandwidth(https://github.com/NVIDIA/nvbandwidth)。

### 延迟

![Low-level Latency Measurements](https://files.mdnice.com/user/59/574e41d3-d3d9-4d60-bcdf-6247546c78f8.png)

(source(https://ieeexplore.ieee.org/document/5238655))

XXX: integrate/expand


### 专用网络硬件和NCCL

专用网络硬件供应商，如AWS（EFA）不披露其秘密，因此公共库，如nccl(https://github.com/NVIDIA/nccl)无法支持它们。这些供应商必须为其硬件的用户提供自己的网络集合通信库版本。

最初，专用硬件供应商使用了一个技巧，告诉用户使用`LD_LIBRARY_PATH`和/或`LD_PRELOAD`来动态重载`libnccl.so`以获取其自定义版本，并将其加载到PyTorch或另一个框架中。但最近NCCL开发了NCCL Net Plugin(https://github.com/NVIDIA/nccl/tree/master/ext-net)，现在应该使用它。这个功能是在NCCL v2.12中添加的。

现在，当NCCL初始化时，它将查找`libnccl-net.so`库，并动态加载它，然后在库中查找符号。这就是专用硬件供应商现在应该放置其自定义API的地方。当然，这个库仍然应该在`LD_LIBRARY_PATH`中，或者在`/etc/ld.so.conf`配置中。

有关动态库加载的更多信息，请参见此部分(https://github.com/stas00/the-art-of-debugging/tree/master/compiled-programs#shared-libraries-ldsoconf-nm-unresolved-symbols-ldd-ld_library_path-ld_preload)。

### 节点接近性

如果你从云中获得2个随机节点，它们可能不在同一个子网中，并且所有传输都会产生额外的延迟。

你希望确保用于单个训练的所有节点都位于同一个子网/脊椎中，这样它们之间都是一跳的距离。

当你计划最终拥有一个大型集群但从小型集群开始时，请确保你的提供商可以在保持所有节点彼此靠近的同时扩展集群。

以下是云特定的实现节点接近性的方法：

- Azure: availability set(https://learn.microsoft.com/en-us/azure/virtual-machines/availability-set-overview?source=recommendations)
- GCP: compact placement policies(https://cloud.google.com/compute/docs/instances/use-compact-placement-policies)

根据你拥有的包类型或租用的机器类型，你可能或可能无法使用这些方法。

### 共享节点间网络

如果你使用共享的HPC环境，或者即使你拥有自己的集群但与你的同事共享，请预期网络带宽不可靠且在一天中的不同时间波动。

这种情况不幸地使得很难微调训练设置的性能。因为每次运行测试时，TFLOPs都会有所不同，那么你如何进行优化呢？这是至少在基于SLURM的集群中出现的情况。显然，当使用Kubernetes时，可以使用集群命名空间来隔离网络。

案例研究：我们在JeanZay HPC进行初步实验时遇到了这个问题，当时我们正在训练BLOOM-176B。由于该HPC有许多用户，几乎不可能进行速度优化，因为即使再次运行完全相同的设置，也会给出不同的吞吐量结果。幸运的是，在我们启动BLOOM-176B训练之前，我们被给予了对当时新A100分区的独占访问权限，因此我们成为了唯一用户，并且能够大大优化吞吐量。
