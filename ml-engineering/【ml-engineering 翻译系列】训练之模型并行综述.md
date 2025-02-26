> 我的课程笔记，欢迎关注：https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode 
> 本篇文档的来源：https://github.com/stas00/ml-engineering 。这篇文档全面介绍了大规模深度学习模型训练中的并行化策略，包括传统的数据并行(DP)、ZeRO优化的数据并行、张量并行(TP)、流水线并行(PP)以及序列并行(SP)等方法。文档详细解释了每种并行方式的工作原理，以及它们如何解决大模型训练中的内存限制和计算效率问题。特别是对ZeRO并行策略进行了深入讲解，包括其实现原理、网络带宽要求和与其他并行方式的组合使用。针对不同的硬件配置（单GPU、单节点多GPU、多节点）和模型规模，文档提供了具体的并行化策略选择建议，帮助我们在实际应用中根据具体场景选择最适合的并行化方案，从而实现大规模模型的高效训练。

# 模型并行

## 并行概述

在现代机器学习中,各种并行方法被用于:

1. 克服 GPU 内存限制。例如:
   - 适配超大模型 - 例如,t5-11b 仅模型参数就需要 45GB
   - 适配超长序列 - 例如,
2. 显著加快训练速度 - 将需要一年的训练时间缩短到几小时

我们首先将深入讨论各种一维并行技术及其优缺点,然后看看如何将它们组合成二维和三维并行,以实现更快的训练速度并支持更大的模型。还将介绍其他各种强大的替代方法。

虽然主要概念很可能适用于任何其他框架,但本文重点关注基于 PyTorch 的实现。

有两种主要方法可以实现训练和推理比加速器内存更大的模型:
1. 3D 并行 - 网络效率很高,但可能会对建模代码造成很大干扰,需要更多工作才能正确运行
2. ZeRO 并行 - 网络效率不是很高,但几乎不需要对建模代码进行更改,很容易实现。

## 可扩展性概念

以下是本文稍后将深入描述的主要概念的简要说明。

1. 数据并行(DP) - 相同的设置被复制多次,每个副本处理一部分数据。处理过程并行执行,所有设置在每个训练步骤结束时同步。

2. 张量并行(TP) - 每个张量被分成多个块,而不是整个张量驻留在单个GPU上,每个分片驻留在其指定的GPU上。在处理过程中,每个分片在不同的GPU上单独并行处理,结果在步骤结束时同步。这可以称为水平并行,因为分割发生在水平层面。

3. 流水线并行(PP) - 模型在多个GPU之间垂直(层级)拆分,因此一个GPU上只放置一个或几个模型层。每个GPU并行处理流水线的不同阶段,处理小批量数据。

4. 零冗余优化器(ZeRO) - 也执行类似于TP的张量分片,但在前向或反向计算时会重建整个张量,因此不需要修改模型。它还支持各种卸载技术来补偿有限的GPU内存。Sharded DDP是各种其他ZeRO实现使用的基础ZeRO概念的另一个名称。

5. 序列并行 - 训练长输入序列需要大量GPU内存。这种技术将单个序列的处理分散到多个GPU上。

6. 专家并行 - 混合专家(MoE)可以进行分区,使每个专家都有一个专用GPU(或多个GPU)。

这篇论文的介绍部分可能是我找到的关于最常见并行技术的最好解释之一: Breadth-First Pipeline Parallelism(https://arxiv.org/abs/2211.05953)。

## 数据并行

### DDP

大多数拥有2个GPU的用户已经享受到`DataParallel`（DP）和`DistributedDataParallel`（DDP）带来的训练速度提升，这些功能很易于使用，这是Pytorch的内置特性。

详细信息请参见DistributedDataParallel（https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html）

### ZeRO数据并行

由ZeRO驱动的数据并行（ZeRO-DP）在以下博客文章中的图表中描述（https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/）

![](https://files.mdnice.com/user/59/8e968fe7-2499-4cf5-9502-ff787f6ec9d8.png)

这个概念可能一开始很难理解，但实际上非常简单。这就是普通的`DataParallel`(DP)，只不过每个GPU不是存储完整的模型参数、梯度和优化器状态的副本，而是只存储其中的一个切片。然后在运行时，当某个层需要完整的层参数时，所有GPU会同步以互相提供它们缺失的部分 - 就是这样。

考虑这个简单的3层模型,每层有3个参数:
```
La | Lb | Lc
---|----|---
a0 | b0 | c0
a1 | b1 | c1
a2 | b2 | c2
```
层 La 有权重 a0、a1 和 a2。

如果我们有3个GPU,分片DDP(= Zero-DP)会将模型拆分到3个GPU上,如下所示:

```
GPU0:
La | Lb | Lc
---|----|---
a0 | b0 | c0

GPU1:
La | Lb | Lc
---|----|---
a1 | b1 | c1

GPU2:
La | Lb | Lc
---|----|---
a2 | b2 | c2
```

从某种意义上说,如果你想象典型的DNN图,这与张量并行的水平切片是一样的。垂直切片是将整个层组放在不同的GPU上。但这只是起点。

现在每个GPU都会像在DP中一样获得常规的mini-batch:
```
x0 => GPU0
x1 => GPU1
x2 => GPU2
```

输入数据保持不变 - 它们认为会被正常模型处理。

首先,输入数据进入层 La。

让我们只关注 GPU0:x0 需要 a0、a1、a2 参数来完成前向传播,但 GPU0 只有 a0 - 它从 GPU1 获得 a1,从 GPU2 获得 a2,将模型的所有部分组合在一起。

同时,GPU1 获得 mini-batch x1,它只有 a1,但需要 a0 和 a2 参数,所以它从 GPU0 和 GPU2 获得这些参数。

GPU2 也是一样,它获得输入 x2。它从 GPU0 和 GPU1 获得 a0 和 a1,并用它的 a2 重建完整的张量。

所有3个 GPU 都重建了完整的张量并进行前向计算。

一旦计算完成,不再需要的数据就会被丢弃 - 它只在计算期间使用。重建是通过预取高效完成的。

整个过程在层 Lb、然后 Lc 的前向传播中重复,然后在反向传播中按 Lc -> Lb -> La 的顺序重复。

对我来说,这听起来像是一个高效的团队背包重量分配策略:

1. A 负责携带帐篷
2. B 负责携带炉子
3. C 负责携带斧头

每天晚上他们都会分享自己拥有的东西,并从其他人那里获得他们没有的东西,早上他们打包自己分配的装备类型并继续前进。这就是分片 DDP / Zero DP。

与每个人都必须携带自己的帐篷、炉子和斧头的简单策略相比,这种策略要高效得多。

在阅读这个主题的文献时,你可能会遇到以下同义词:Sharded、Partitioned。

如果你仔细观察 ZeRO 分区模型权重的方式 - 它看起来与稍后将讨论的张量并行非常相似。这是因为它对每个层的权重进行分区/分片,而不像接下来讨论的垂直模型并行。

ZeRO-DP 阶段 1+2+3 的实现:
- DeepSpeed(https://www.deepspeed.ai/tutorials/zero/)
- PyTorch(https://pytorch.org/docs/stable/fsdp.html) (最初在 FairScale(https://github.com/facebookresearch/fairscale/) 中实现,后来被 upstream 到 PyTorch Core)

Deepspeed ZeRO 集成:
- HF Trainer 集成(https://huggingface.co/docs/transformers/main_classes/deepspeed)
- Accelerate(https://huggingface.co/docs/accelerate/usage_guides/deepspeed)
- PyTorch Lightning(https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/deepspeed.html)
- Determined.AI(https://docs.determined.ai/latest/model-dev-guide/api-guides/apis-howto/deepspeed/_index.html)

FSDP 集成:
- HF Trainer 集成(https://huggingface.co/docs/transformers/main/en/fsdp)
- Accelerate(https://huggingface.co/docs/accelerate/main/en/usage_guides/fsdp)
- PyTorch Lightning(https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/fsdp.html)

重要论文:

Deepspeed 和 ZeRO 总体:
- ZeRO:面向训练万亿参数模型的内存优化(https://arxiv.org/abs/1910.02054)
- ZeRO-Offload:民主化十亿规模模型训练(https://arxiv.org/abs/2101.06840)
- ZeRO-Infinity:突破 GPU 内存墙以实现极端规模深度学习(https://arxiv.org/abs/2104.07857)
- ZeRO++:用于巨型模型训练的极其高效的集体通信(https://arxiv.org/abs/2306.10209)
- DeepSpeed Ulysses:实现训练极长序列 Transformer 模型的系统优化(https://arxiv.org/abs/2309.14509)
- AMSP:减少 ZeRO 的通信开销以实现高效的 LLM 训练(https://arxiv.org/abs/2311.00257)

PyTorch:
- PyTorch FSDP:扩展完全分片数据并行的经验(https://arxiv.org/abs/2304.11277)

主要 DeepSpeed ZeRO 资源:
- 项目 github(https://github.com/microsoft/deepspeed)
- 使用文档(https://www.deepspeed.ai/getting-started/)
- API 文档(https://deepspeed.readthedocs.io/en/latest/index.html)
- 博客文章(https://www.microsoft.com/en-us/research/search/?q=deepspeed)

#### 克服巨大全局批量大小的问题

如果你使用1024个加速器,每个加速器上的分片会很小,并且会有大量的空闲内存用于微批量大小(MBS),假设你可以设置MBS=32 - 最终得到GBS=32k - 这很可能不是你想要的。

所以你要么需要部署张量并行(这很难实现),要么通常更简单的方法是部署序列并行(https://arxiv.org/abs/2305.14343)。我还没有实际尝试过,但到目前为止我了解到:

- Deepspeed ZeRO 使用 Deepspeed-Ulysses(https://arxiv.org/abs/2309.14509)
- FSDP 使用 Paged Ring Attention(https://github.com/lucidrains/ring-attention-pytorch) 论文(https://arxiv.org/abs/2402.08268)

请注意,这可能不会像张量并行(https://arxiv.org/abs/2305.14343)那样高效 - 但我还不知道实际的额外开销。

#### 使用多个副本的 ZeRO

默认情况下,ZeRO 使用所有 GPU 来创建单个模型副本 - 即模型分布在所有 GPU 上。这会导致各种限制,例如:

1. 全局批量大小不灵活 - 它总是 `total_gpus*micro_batch_size` 的函数 - 在大型集群上可能会导致巨大的全局批量大小,这可能不利于高效收敛。当然可以使用很小的微批量大小来控制全局批量大小,但这会导致每个 GPU 上的矩阵更小,从而降低计算效率
2. 没有充分利用更快的节点内网络,因为较慢的节点间网络定义了通信的整体速度。

ZeRO++(https://arxiv.org/abs/2306.10209) 通过引入分层权重分区(hpZ)解决了第二个限制。在这种方法中,每个模型副本被限制在单个节点内,而不是将整个模型权重分散到所有 GPU 上。这会使内存使用量增加节点总数倍,但现在收集分片组件的 2x `all_gather` 调用是在更快的节点内连接上执行的。只有用于聚合和重新分配梯度的 `reduce_scatter` 是在较慢的节点间网络上执行的。

第一个限制并没有完全解决,因为总体全局批量大小保持不变,但由于每个副本更高效,并且由于额外的内存压力可能会限制每个 GPU 上可能的微批量大小,这总体上应该会提高系统的吞吐量。

PyTorch FSDP 在 shardingStrategy.HYBRID_SHARD(https://pytorch.org/docs/stable/fsdp.html) 中实现了这个功能

相关论文:

- ZeRO++: 巨型模型训练的极其高效的集体通信(https://arxiv.org/abs/2306.10209)
- PyTorch FSDP: 扩展完全分片数据并行的经验(https://arxiv.org/abs/2304.11277)


#### ZeRO 变体

提出对 ZeRO 协议进行修改的已发表论文:

- MiCS: 在公有云上训练巨型模型的近线性扩展(https://arxiv.org/abs/2205.00119) (2022)
- AMSP: 通过高级模型状态分区实现 LLM 训练的超级扩展(https://arxiv.org/abs/2311.00257) (2023)




## 流水线并行方法

### 朴素模型并行(垂直)

朴素模型并行(MP)是指将模型层组分布在多个 GPU 上。其机制相对简单 - 将目标层通过 `.to()` 切换到目标设备,现在当数据进出这些层时,将数据切换到与该层相同的设备,其余部分保持不变。

我们将其称为垂直 MP,因为如果你还记得大多数模型是如何绘制的,我们垂直切分层。例如,如果下图显示了一个8层模型:

```
===================  ===================
|  0 | 1 | 2 | 3  |  |  4 | 5 | 6 | 7  |
===================  ===================
        gpu0                 gpu1
```
我们将其垂直切分为2部分,将第0-3层放在GPU0上,将第4-7层放在GPU1上。

当数据从第0层传递到第1层,第1层到第2层,以及第2层到第3层时,这就像普通模型一样。但是当数据需要从第3层传递到第4层时,它需要从GPU0传输到GPU1,这会引入通信开销。如果参与的GPU位于同一计算节点(例如同一物理机)上,这种复制速度相当快,但如果GPU位于不同的计算节点(例如多台机器)上,通信开销可能会显著增加。

然后第4层到第5层到第6层到第7层的运行就像普通模型一样,当第7层完成时,我们通常需要将数据发送回第0层(标签所在的位置),或者将标签发送到最后一层。现在可以计算损失,优化器可以开始工作。

问题:
- 主要缺陷(也是为什么称之为"朴素"MP的原因)是在任何时刻只有一个GPU在工作,其他GPU都处于空闲状态。因此,如果使用4个GPU,这几乎等同于将单个GPU的内存量增加4倍,而忽略了其余的硬件。此外还有设备间数据复制的开销。所以4个6GB显卡使用朴素MP可以容纳与1个24GB显卡相同大小的模型,但后者会更快完成训练,因为它没有数据复制开销。但是,比如说,如果你有40GB的显卡,需要容纳一个45GB的模型,你可以用4个40GB的显卡(但由于梯度和优化器状态的存在,勉强可以)
- 共享嵌入（Embedding权重）可能需要在GPU之间来回复制。

### 流水线并行

流水线并行(PP)与朴素MP几乎相同,但它通过将输入批次分块成微批次并人为创建流水线来解决GPU空闲问题,这使得不同的GPU可以同时参与计算过程。

下面来自GPipe论文(https://ai.googleblog.com/2019/03/introducing-gpipe-open-source-library.html)的插图展示了朴素MP(上图)和PP(下图):

![](https://files.mdnice.com/user/59/3255a30e-2664-4fc9-838c-f3ebfebddf8f.png)


从中图可以很容易看出PP如何减少了GPU空闲的死区。这些空闲部分被称为"气泡"。

图中的两部分都展示了pp=4的并行性。也就是说有4个GPU参与流水线。因此有4个管道阶段的前向路径F0、F1、F2和F3,然后是反向顺序的反向路径B3、B2、B1和B0。

PP引入了一个新的超参数`chunks`来调优,它定义了通过同一管道阶段按顺序发送多少块数据。例如,在图中可以看到`chunks=4`。GPU0对块0、1、2和3执行相同的前向路径(F0,0、F0,1、F0,2、F0,3),然后等待其他GPU完成它们的工作,只有当它们的工作开始完成时,GPU0才会再次工作,对块3、2、1和0执行反向路径(B0,3、B0,2、B0,1、B0,0)。

注意,从概念上讲,这与梯度累积步骤(GAS)是相同的概念。PyTorch使用`chunks`,而DeepSpeed将相同的超参数称为GAS。

由于分块,PP引入了微批次(MBS)的概念。DP将全局数据批次大小分成小批次,因此如果DP度为4,全局批次大小1024会被分成4个每个256的小批次(1024/4)。如果`chunks`(或GAS)数量为32,我们最终得到微批次大小为8(256/32)。每个流水线阶段一次处理一个微批次。

要计算DP + PP设置的全局批次大小,我们执行:`mbs*chunks*dp_degree`(`8*32*4=1024`)。

让我们回到这个图。

当`chunks=1`时,你最终会得到朴素MP,这是非常低效的。而当`chunks`值很大时,你会得到非常小的微批次大小,这也可能不太高效。因此,需要通过实验来找到能够实现GPU最高效利用率的值。

虽然图中显示了一个无法并行化的"死亡"时间气泡(因为最后的`forward`阶段必须等待`backward`完成pipeline),但寻找最佳`chunks`值的目的是实现所有参与GPU的高并发利用率,这意味着要最小化气泡的大小。

调度的选择对高效性能至关重要,按发明顺序排列的最常见调度方式包括:

- 顺序 Gpipe: 使用流水线并行实现巨型神经网络的高效训练(https://arxiv.org/abs/1811.06965)
- 交错 1F1B Pipedream: 快速高效的流水线并行DNN训练(https://arxiv.org/abs/1806.03377)
- 循环、深度优先的高效大规模语言模型在GPU集群上的训练使用Megatron-LM(https://arxiv.org/abs/2104.04473)
- 广度优先的流水线并行(https://arxiv.org/abs/2211.05953)
- Llama 3 训练结合了深度优先和广度优先的方法以获得最佳性能，并且允许他们在训练过程中逐步修改全局批量大小，这在使用流水线并行时通常是非常困难的。请参阅《Llama 3 模型群体》(https://arxiv.org/abs/2407.21783) 第 3.3.2 节 关于模型扩展的并行性。

这里是一个交错流水线的例子:

![parallelism-sagemaker-interleaved-pipeline](https://files.mdnice.com/user/59/a302752e-b7b4-4c83-8592-634598fcdf2e.png)

在这里,气泡(空闲时间)通过优先处理反向传播进一步最小化。

DeepSpeed、Varuna和SageMaker等都使用了这种方式。

Varuna通过使用模拟来发现最有效的调度方式,从而进一步改进调度。

DeepSeek v3（https://arxiv.org/abs/2412.19437） 引入了一种更高效的PP，通过DualPipe减少了气泡大小，并实现了更好的计算与通信重叠。具体细节请参见论文第3.2.1节。

![来源：https://arxiv.org/abs/2412.19437](https://files.mdnice.com/user/59/36b857ea-fe9c-4f2f-9f81-33ff74de4f18.png)

PP解决方案有两类 - 传统的Pipeline API和更现代的解决方案,后者通过帮助部分或完全自动化流程,使最终用户使用起来更加容易:

1. 传统的Pipeline API解决方案:
- Megatron-LM
- DeepSpeed
- PyTorch

2. 现代解决方案:
- PiPPy
- Varuna
- Sagemaker
- DeepSeek

传统Pipeline API解决方案的问题:
- 必须对模型进行大量修改,因为Pipeline要求将模块的正常流程重写为相同模块的`nn.Sequential`序列,这可能需要更改模型的设计。
- 目前Pipeline API非常受限。如果在Pipeline的第一阶段有一堆Python变量需要传递,你必须找到解决方法。目前,pipeline接口只接受单个Tensor或Tensor元组作为唯一的输入和输出。这些张量的第一个维度必须是批次大小,因为pipeline会将mini batch分成micro-batches。可能的改进正在这里讨论(https://github.com/pytorch/pytorch/pull/50693)
- 在pipe阶段级别的条件控制流是不可能的 - 例如,像T5这样的编码器-解码器模型需要特殊的变通方法来处理条件编码器阶段。
- 必须安排每一层,使一个模型的输出成为另一个模型的输入。

我还没有尝试过Varuna和SageMaker,但根据他们的论文报告,他们已经克服了上述问题列表,并且对用户的模型只需要很小的改动。

实现:
- Pytorch(https://pytorch.org/docs/stable/pipeline.html) (在pytorch-1.8中初步支持,并在1.9和1.10中逐步改进)。一些示例(https://github.com/pytorch/pytorch/blob/master/benchmarks/distributed/pipeline/pipe.py)
- FairScale(https://fairscale.readthedocs.io/en/latest/tutorials/pipe.html)
- DeepSpeed(https://www.deepspeed.ai/tutorials/pipeline/)
- Megatron-LM(https://github.com/NVIDIA/Megatron-LM)有内部实现 - 没有API。
- Varuna(https://github.com/microsoft/varuna)
- SageMaker(https://arxiv.org/abs/2111.05972) - 这是一个只能在AWS上使用的专有解决方案。
- OSLO(https://github.com/eleutherAI/Oslo) - 这是基于Hugging Face Transformers实现的。
- PiPPy(https://github.com/pytorch/pippy) - 通过`torch.fx`自动PP
- nanotron(https://github.com/huggingface/nanotron)
- torchtitan(https://github.com/pytorch/torchtitan)

### 相关阅读

- 流水线并行：通过模型分区进行分布式训练(https://siboehm.com/articles/22/pipeline-parallel-training)


## 张量并行

在张量并行中,每个GPU只处理张量的一个切片,只在需要完整张量的操作时才聚合完整的张量。

在本节中,我们使用来自Megatron-LM(https://github.com/NVIDIA/Megatron-LM)论文的概念和图表:在GPU集群上高效训练大规模语言模型(https://arxiv.org/abs/2104.04473)。

任何transformer的主要构建块都是一个全连接层`nn.Linear`,后面跟着一个非线性激活函数`GeLU`。

按照Megatron论文的符号,我们可以将点积部分写为`Y = GeLU(XA)`,其中`X`和`Y`是输入和输出向量,`A`是权重矩阵。

如果我们以矩阵形式查看计算,很容易看出矩阵乘法如何在多个GPU之间拆分:

![Parallel GEMM](https://files.mdnice.com/user/59/f88b660b-c53f-4c23-bc83-8363e2e255d9.png)


如果我们将权重矩阵 `A` 按列分割到 `N` 个GPU上,并行执行矩阵乘法 `XA_1` 到 `XA_n`,那么我们将得到 `N` 个输出向量 `Y_1, Y_2, ..., Y_n`,它们可以独立地输入到 `GeLU` 中:

![independent GeLU](https://files.mdnice.com/user/59/ac0d80ee-29a4-4166-aa10-dcc9c1c8fad7.png)

使用这个原理,我们可以更新任意深度的MLP,在最后需要从分片重建输出向量之前,不需要在GPU之间进行任何同步。Megatron-LM论文作者为此提供了一个有帮助的示意图:

![parallel shard processing](https://files.mdnice.com/user/59/477e5763-1ae1-4e9a-96e2-0a13763373bc.png)

由于多头注意力层本身就具有多个独立的头,因此并行化多头注意力层甚至更简单!

![parallel self-attention](https://files.mdnice.com/user/59/fb28c287-9c49-495a-a04a-9a0e69d311d7.png)

重要提示:TP需要非常快速的网络,由于节点内网络通常比节点间网络快得多,因此不建议跨节点进行TP。实际上,如果一个节点有4个GPU,那么TP的最高程度就是4。如果你需要8度的TP,你需要使用至少有8个GPU的节点。

重要提示:TP程度不应跨节点。例如,如果节点有8个gpu,TP程度应该不超过8。

TP可以与其他并行化方法结合使用。

其他名称:
- DeepSpeed称之为张量切片(https://www.deepspeed.ai/tutorials/large-models-w-deepspeed/)

实现:
- Megatron-LM(https://github.com/NVIDIA/Megatron-LM)有内部实现,因为它是非常特定于模型的
- PyTorch(https://pytorch.org/docs/stable/distributed.tensor.parallel.html)
- SageMaker(https://arxiv.org/abs/2111.05972) - 这是一个只能在AWS上使用的专有解决方案
- OSLO(https://github.com/eleutherAI/Oslo)基于Transformers实现了张量并行
- nanotron(https://github.com/huggingface/nanotron)
- parallelformers(https://github.com/tunib-ai/parallelformers)(目前仅支持推理)
- torchtian(https://github.com/pytorch/torchtitan)

### 异步张量并行

TP的一个缺陷是很难将其通信与计算重叠。PyTorch提出使用异步TP(https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487)来克服这个问题,它将`all-gather + matmul`的依赖序列分解为一系列cudaMemcpyAsync调用和更小的部分matmul - 并且使用`torch.compile`可以自动为你完成这些!

- Megatron-LM也通过`--tp-comm-overlap`实现了这一功能。

### 相关阅读
- 张量并行和序列并行:详细分析(https://insujang.github.io/2024-01-11/tensor-parallelism-and-sequence-parallelism-detailed-analysis/#sequence-parallelism)

## TP+SP

TP可以与SP在同一进程组中结合使用，以最小化通信成本，具体解释见《在大型Transformer模型中减少激活重计算》(https://arxiv.org/abs/2205.05198)。例如，在LLMs中，TP用于嵌入、注意力和线性层，而当达到dropout和层归一化时则改用SP。

## DP+PP

以下来自 DeepSpeed pipeline 教程(https://www.deepspeed.ai/tutorials/pipeline/)的图表展示了如何将 DP 与 PP 结合使用。

![dp-pp-2d](https://files.mdnice.com/user/59/ce30cb13-6de8-4b7b-8a1f-bf2a609a0daa.png)

这里需要注意的是,DP rank 0看不到GPU2,DP rank 1看不到GPU3。对于DP来说,只有GPU 0和1,它像只有2个GPU一样向它们输入数据。GPU0使用PP"秘密地"将一些负载卸载到GPU2。GPU1也通过利用GPU3做同样的事情。

由于每个维度至少需要2个GPU,所以这里你至少需要4个GPU。

实现:
- DeepSpeed(https://github.com/microsoft/DeepSpeed)
- Megatron-LM(https://github.com/NVIDIA/Megatron-LM)
- Varuna(https://github.com/microsoft/varuna)
- SageMaker(https://arxiv.org/abs/2111.05972)
- OSLO(https://github.com/eleutherAI/Oslo)
- nanotron(https://github.com/huggingface/nanotron)
- torchtitan(https://github.com/pytorch/torchtitan)


## DP+PP+TP

为了获得更高效的训练,可以使用3D并行,即将PP与TP和DP结合使用。这可以从下图中看出。

![dp-pp-tp-3d](https://files.mdnice.com/user/59/e96ccef9-65a2-4439-980c-71d296a62a62.png)

这个图来自博客文章《3D并行:扩展到万亿参数模型》(https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/),这也是一篇值得一读的文章。

由于每个维度至少需要2个GPU,所以这里你至少需要8个GPU。

实现:
- DeepSpeed(https://github.com/microsoft/DeepSpeed) - DeepSpeed also includes an even more efficient DP, which they call ZeRO-DP.
- Megatron-LM(https://github.com/NVIDIA/Megatron-LM)
- Varuna(https://github.com/microsoft/varuna)
- SageMaker(https://arxiv.org/abs/2111.05972)
- OSLO(https://github.com/eleutherAI/Oslo)
- nanotron(https://github.com/huggingface/nanotron)
- torchtitan(https://github.com/pytorch/torchtitan)

## ZeRO DP+PP+TP

DeepSpeed的主要特性之一是ZeRO,它是DP的一个扩展。在ZeRO数据并行中已经讨论过了。通常它是一个独立的功能,不需要PP或TP。但它可以与PP和TP结合使用。

当ZeRO-DP与PP(和可选的TP)结合时,通常只启用ZeRO stage 1(优化器分片)。

虽然理论上可以将ZeRO stage 2(梯度分片)与流水线并行结合使用,但会对性能产生不良影响。每个微批次都需要一个额外的reduce-scatter集合来在分片之前聚合梯度,这会增加潜在的显著通信开销。由于流水线并行的本质,使用小的微批次,而重点是尝试平衡算术强度(微批次大小)与最小化流水线气泡(微批次数量)。因此这些通信成本会造成伤害。

此外,由于PP的原因,层数已经比正常情况下少了,所以内存节省不会很大。PP已经将梯度大小减少了"1/PP",因此在此基础上的梯度分片节省相比纯DP来说不那么显著。

由于同样的原因,ZeRO stage 3也不是一个好选择 - 需要更多的节点间通信。

由于我们有ZeRO,另一个好处是ZeRO-Offload。由于这是stage 1,优化器状态可以被卸载到CPU。

实现:
- Megatron-DeepSpeed(https://github.com/microsoft/Megatron-DeepSpeed)和来自BigScience的Megatron-Deepspeed(https://github.com/bigscience-workshop/Megatron-DeepSpeed),后者是前者的分支。
- OSLO(https://github.com/eleutherAI/Oslo)

重要论文:

- 使用DeepSpeed和Megatron训练Megatron-Turing NLG 530B,一个大规模生成语言模型(
https://arxiv.org/abs/2201.11990)



## 序列并行

机器学习任务,比如DNA测序,可能需要训练非常长的序列长度(例如256K),甚至普通的大语言模型也可能需要训练10k及更长的序列。

Self-Attention作为Transformer的关键组件,其内存需求与序列长度呈二次方关系,因此当序列长度达到一定长度时,即使batch size为1也可能无法装入单个GPU,需要沿序列维度进行额外的切分。一旦完成切分,序列可以是任意长度。

由于这种并行类型与本文档中描述的其他并行化类型是正交的,它可以与任何其他类型组合,从而形成4D、ZeRO-DP+SP等组合。

### Deepspeed-Ulysses SP

论文: DeepSpeed Ulysses: 支持训练超长序列Transformer模型的系统优化(https://arxiv.org/abs/2309.14509)

在这个实现中,有2个元素被分片:
1. 多头注意力权重在参与的GPU之间进行分割,使得每个GPU只有几个子头。这在模型创建/加载时完成。这有点类似于张量并行。
2. 在训练期间,每个输入序列被分成块,每个块被发送到其中一个GPU,这让我们想起了ZeRO-3分片,只不过这里分片的是输入而不是权重。

在计算过程中,每个序列块都被投影到QKV上,然后在每个设备上收集成完整序列的QKV,每个设备只计算它拥有的子头,然后再次收集到MLP块的完整注意力输出中。

![deepspeed-ulysses sp](https://files.mdnice.com/user/59/284c6ec7-8b0f-42cd-8fc9-04d3f7ba3ac7.png)

源码(https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-ulysses)

在图中:
1. 输入序列N被分割到P个可用设备上。
2. 输入序列的每个局部N/P分区被投影到查询(Q)、键(K)和值(V)嵌入。
3. 接下来,通过参与计算设备之间高度优化的all-to-all集合通信,将局部QKV嵌入收集到全局QKV中。
4. 然后对每个注意力头执行注意力计算:

![](https://files.mdnice.com/user/59/2211bb3c-6f7d-41dc-8d87-07fcfdfc1727.png)


5. 最后,另一个all-to-all集合将注意力计算的输出上下文张量转换为序列(N/P)并行,以供transformer层块中剩余模块的后续操作(MLP MatMul、层归一化等)使用。

示例:让我们考虑序列长度=8K,头数=128,单节点GPU数=8的情况

1. 每个GPU获得原始序列的1K长度块(`8K/8`)
2. 每个GPU分配16个子头(`128/8`) 
3. a. 在gpu0上,在`forward`之前,原始序列被重新收集为8K个token
   b. 在前16个子头上执行注意力计算
其余7个GPU执行相同的逻辑,每个GPU在其16个子头上计算8k注意力

你可以在这里阅读高效通信的具体细节(https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-ulysses#significant-communication-volume-reduction)。

DeepSpeed-Ulysses通过增加与消息大小或序列长度成比例的GPU数量来保持通信量的一致性。

### Colossal-AI的序列并行

论文: 从系统角度看序列并行:长序列训练(https://arxiv.org/abs/2105.13120)

Colossal-AI的序列并行实现使用环形自注意力机制,这是一种环形通信集合,其中查询投影是局部的,而键和值投影以环形方式传输以计算全局注意力,导致通信复杂度与消息大小M呈线性关系。

### Megatron-LM的序列并行

论文: 减少大型Transformer模型中的激活重计算(https://arxiv.org/abs/2205.05198)

Megatron-LM的序列并行与其张量并行紧密集成。Megatron-LM沿序列维度对序列进行切分,并应用allgather和reduce scatter集合操作来聚合QKV投影以进行注意力计算。无论计算设备数量如何,其通信量都与消息大小(M)呈线性增长。

### Ring Attention with Blockwise Transformers

论文: Ring Attention with Blockwise Transformers for Near-Infinite Context(https://arxiv.org/abs/2310.01889)

1. 张量始终沿序列维度进行分片:形状为(`seq_len // N, d_model`)
2. 在注意力层中,每个GPU首先使用其可用分片计算它们能够计算的注意力分数部分。
3. 同时,来自其他序列块的键和值在周围进行通信。
4. 一旦另一个块的键/值可用,每个GPU就使用来自序列这个新片段的键/值张量继续进行注意力计算
5. 继续直到注意力计算完成。

序列并行实现:
- Megatron-LM(https://github.com/NVIDIA/Megatron-LM)
- Deepspeed(https://github.com/microsoft/DeepSpeed)
- Colossal-AI(https://colossalai.org/)
- torchtitan(https://github.com/pytorch/torchtitan)

PyTorch也在开发这个功能,并将其称为上下文并行(CP)。

### DistFlashAttn

DISTFLASHATTN: 用于长上下文LLM训练的分布式内存高效注意力（https://arxiv.org/abs/2310.03294）据报道比Ring Attention快多倍，因为它在执行序列并行时在工作节点之间平衡了每个token的KVQ计算负载。

![](https://files.mdnice.com/user/59/219c1879-4474-4e30-b805-f7ce244a60b7.png)

### Related reading

- Tensor Parallelism and Sequence Parallelism: Detailed Analysis(https://insujang.github.io/2024-01-11/tensor-parallelism-and-sequence-parallelism-detailed-analysis/#sequence-parallelism)

## 专家并行

当使用混合专家模型(MoE)(特别是在推理过程中)时,可以为每个专家分配自己的加速器(如果一个不够的话可以分配多个)。这为并行化增加了另一个维度,并且可以显著加速可能会命中所有专家的大批量数据。

详细说明请参见:
- DeepSpeed-MoE:推进混合专家模型推理和训练以支持下一代AI规模(https://arxiv.org/abs/2201.05596)
- 混合专家模型解释(https://huggingface.co/blog/moe#parallelism)

## FlexFlow

FlexFlow(https://github.com/flexflow/FlexFlow)以略微不同的方式解决并行化问题。

论文:"超越深度神经网络的数据和模型并行" 作者:Zhihao Jia, Matei Zaharia, Alex Aiken(https://arxiv.org/abs/1807.05358)

它在样本-算子-属性-参数这4个维度上执行并行化。

1. 样本 = 数据并行(样本维度并行)
2. 算子 = 将单个操作并行化为多个子操作
3. 属性 = 数据并行(长度维度并行)
4. 参数 = 模型并行(不考虑维度 - 水平或垂直)

示例:
* 样本

假设有10个批次,每个序列长度为512。如果我们在样本维度上将它们并行到2个设备上,我们得到10 x 512变成5 x 2 x 512。

* 算子

如果我们执行层归一化,我们首先计算std然后计算mean,然后我们可以归一化数据。算子并行允许并行计算std和mean。所以如果我们在算子维度上将它们并行到2个设备(cuda:0, cuda:1),首先我们将输入数据复制到两个设备,cuda:0计算std,cuda:1同时计算mean。

* 属性

我们有10个批次,每个长度为512。如果我们在属性维度上将它们并行到2个设备,10 x 512将变成10 x 2 x 256。

* 参数

这与张量模型并行或简单的层级模型并行类似。

![](https://files.mdnice.com/user/59/3c560576-b425-46fd-a08c-9d92b0b5dfc0.png)

这个框架的重要性在于它可以处理如(1) GPU/TPU/CPU、(2) RAM/DRAM、(3) 快速内部连接/慢速外部连接等资源,并自动优化所有这些资源,以算法方式决定在哪里使用哪种并行化。

一个非常重要的方面是,FlexFlow 专门用于优化具有静态和固定工作负载的 DNN 并行化,因为具有动态行为的模型可能在不同迭代中倾向于不同的并行化策略。

所以这个承诺非常有吸引力 - 它在所选的集群上运行30分钟的模拟,并提出最佳策略来利用这个特定环境。如果你添加/删除/替换任何部分,它都会重新运行并重新优化计划。然后你就可以开始训练了。不同的设置将有其自己的定制优化。

### 并行网络集合

由于节点内和节点间的速度通常存在10倍差异，因此在进行节点内和节点间交互时选择不同的并行化技术至关重要。例如，TP必须始终保持在节点内部，因为其巨大的同步需求。此外，一些加速器，如最新的AMD MI3**系列，其GPU间连接速度非常慢，这同样影响了并行化的最佳性能。

这里有一个有用的提示：all-reduce集合可以分解为两个独立的阶段：reduce-scatter和all-gather。

![来源：https://engineering.fb.com/2021/07/15/open-source/fsdp/attachment/fsdp-graph-2a/](https://files.mdnice.com/user/59/e4205b8c-2fbe-4189-ad9f-4c05f29b2b1b.png)

以下是不同并行化策略所使用的集合操作的详细说明：

- DDP：1次all-reduce用于梯度 - 理想情况下与计算重叠 - 总通信量：2倍模型参数
- ZeRO-DP ZeRO-1/ZeRO-2：1次all-gather用于优化器状态加上1次reduce-scatter用于梯度 - 总通信量：2倍模型参数
- ZeRO-DP ZeRO-3：2次all-gather用于权重（在前向传播之前和反向传播之前）加上1次reduce-scatter用于梯度 - 总通信量：3倍模型参数（比DDP和ZeRO-1/ZeRO-2多1.5倍）
- TP：2次all-gather和2次reduce-scatter
- PP：2次发送 + 2次接收 - 在稳定阶段与计算重叠
- SP：取决于实现：对于隐藏层大小h、序列长度N和并行度P
    - Megatron-LM：2次all-gather和2次reduce-scatter，通信量为每个Transformer Layer `4*N*h`（参考论文第3.2节 https://arxiv.org/abs/2309.14509）
    - DeepSpeed Ulysses：2次all-to-all通信，通信量为每个Transformer Layer `4*N*h/P`（参考论文第3.2节 https://arxiv.org/abs/2309.14509）

你可能会发现不同的实现可能使用不同的通信模式。

## 使用ZeRO的节点间速度要求

ZeRO可扩展性协议,无论是Deepspeed ZeRO还是PyTorch FSDP,都需要比TP+PP+DP解决方案更多的节点间流量。有时它无法利用更快的节点内连接,因此如果你的节点间网络速度较慢,你的昂贵GPU可能会因通信而严重受限。

ZeRO协议部分地将通信与计算重叠,所以理想情况下你希望达到`通信时间 <= 计算时间`。重叠并不完美,所以总会有一些网络瓶颈,但我们要确保`通信时间`不会比`计算时间`大太多。

在ZeRO-3中,我们在`forward`中对权重进行`all_gather`,然后在`backward`中对权重进行`all_gather`,最后在backward中对梯度进行`reduce_scatter`。总共有3次全局集合调用,每次发送的模型大小乘以每个参数使用的字节数。例如,一个10B参数的bf16模型在ZeRO-3下需要发送`10*2*3` = 60GB的数据。

相比之下,DistributedDataParallel(DDP)使用单个`all_reduce`调用,但需要2倍的数据传输,因此10B参数的bf16模型在DDP下需要发送`10*2*2` = 40GB的数据。

ZeRO-1只分片优化器状态,像DDP一样,也需要传输40GB数据(一次`all_gather`和一次`reduce_scatter`)。

以下是如何计算通信和计算的时间(秒):

- `通信时间 = 传输次数 * 字节数 * 模型大小(B) / 节点间吞吐量(GBps)`
- `计算时间 = 计算次数 * 字节数 * 模型大小(B) * 序列长度 * 全局批量大小 / (总GPU数 * 1e3 * 无通信时的TFLOPS)`

计算时间公式是一个粗略估计,适用于任何基于Transformer块的模型。它忽略了任何小计算,只包括大型`matmul`。

让我们以IDEFICS-80B训练的数据点为例进行实验。

当我们使用340GBs EFA训练IDEFICS-80B时,使用Deepspeed ZeRO-3在A100上只能获得90TFLOPs,而Megatron的TP+PP+DP可以获得150+TFLOPs。而且模型的很大一部分是冻结的,因为我们正在基于一个语言模型和一个视觉模型构建新模型。所以我们的乘数小于3。另一方面,我们使用激活重计算来节省内存,所以这需要额外传输所有模型权重,而且由于nccl不支持适当的半精度reduction,我们对梯度reduction使用fp32,所以实际上我们的乘数不是3而是4.5。

IDEFICS-80B训练使用的值:
- `model_size_in_B` = `80`
- `n_bytes` = `2` (bf16是2字节)
- `n_transmissions` = `3` (ZeRO-3/FSDP的情况下是1次reduce_scatter + 2次all_gather(fwd + bwd)),ZeRO-1是2(1次reduce_scatter + 1次all_gather)
- 另外,对于IDEFICS-80B,我们决定在fp32中reduce梯度以最小化NCCL累积损失,所以实际上我们有`n_transmissions*n_bytes=3*2+2=4*2`用于额外的2字节,但由于模型一半被冻结,只有大约一半的梯度被发送,所以我们仍然有3的乘数。
- `n_passes` = `4` (使用激活重计算),或`3` (不使用)。模型在`forward`中只需要1次计算,在`backward`中需要2次(因为梯度计算了两次 - 一次是相对于输入,一次是相对于权重)。使用激活重计算时还要多做一次`forward`。
- `total_gpus` = `512`
- `global_batch_size` = `3584`
- `seqlen` = `1024`
- `inter-node-throughput_in_GBps` = 42.5 (340Gbps) (AWS EFA v1)
- `tflops_wo_comms`是没有通信开销的tflops。不是理论峰值,因为那是无法达到的,但在A100@BF16的情况下可能是75% - 所以是`312*0.75=234` TFLOPS

我们使用`all_reduce_bench.py`(https://github.com/BBuf/ml-engineering/blob/master/network/benchmarks/all_reduce_bench.py)得出340Gbps节点间网络吞吐量,它默认使用4GB的有效载荷。在IDEFICS-80B的情况下,我们有80层,所以每层大约有1B参数。这意味着每层对于bf16张量发送2GB数据,对于fp32张量发送4GB数据,这与网络基准相匹配。如果你的层大小小得多,我建议调整基准以适应该大小。例如,如果你的层大小只有100M参数,那么bf16张量的有效载荷将是0.2GB。由于这比较小一个数量级,网络可能会给你更低的带宽,你应该在计算中使用这个值。

注:如果你的模型部分被冻结,那么在同步梯度时会发送更少的数据。在IDEFICS中,我们有超过一半的模型被冻结,所以当梯度被reduce时,我们只有大约一半的流量。

这给我们:

- 通信 = `3 * 2 * 80 / 42.5` = 11秒
- 计算 = `4 * 2 * 80 * 1024 * 3584 / (512 * 1e3 * 250)` = 18秒

如果我们对照IDEFICS-80B的日志,每次迭代大约49秒。

好消息是数学计算是正确的,因为通信+计算与测量时间大致相符,除了

我们可以通过将计算公式输入我们记录的90 TFLOPS进行另一次完整性检查:

- 计算 = `4 * 2 * 80 * 1024 * 3584 / (512 * 1e3 * 90)` = 51秒

所以49和51秒非常接近。但这什么都说明不了,因为记录的TFLOPS是使用这个公式计算的,所以当然应该匹配。

在最好的情况下,我期望在公式中使用接近理论峰值的TFLOPS,并得到与系统上实际测量的计算时间大致相同的计算估计。记住,由于通信与计算交织在一起,当我们测量`forward`+`backward`的墙钟时间时,它包括了通信时间。

结论是什么?我认为需要更多的调查,因为显然这里有额外的隐藏瓶颈。我不再能访问这个设置进行调查,所以当我训练另一个较大的模型时,我会重新进行这个练习,并与你分享更新的数学计算。但这个练习应该让你感受到幕后发生的事情以及这些数字是如何协同工作的。

此外,这个讨论没有将梯度累积步骤(GAS)纳入数学计算。在IDEFICS-80B的情况下没有使用它。如果GAS>1,理论计算时间不变,但通信时间从`3*2*M/GBps`变为`GAS*3*2*M/GBps`。`forward`和`backward`的权重收集通过`all_gather`会发生与梯度累积步骤一样多的次数。理论上对于梯度只需要发生一次,但由于每个GPU上没有地方存储收集权重的中间梯度,它也需要reduce GAS次。这适用于ZeRO-2和ZeRO-3。对于ZeRO-1,GAS>1不需要额外的通信。

我们也没有讨论`DataLoader`作为这里的潜在瓶颈,但我们测试发现它不到1秒,即开销很小。

回到通信数学,我们也没有考虑各种硬件延迟,但在处理大型有效载荷时,它们不应该增加显著的额外开销。

现在你知道在你的系统网络上传输那么多GB需要多长时间。例如,如果网络比我们用于IDEFICS-80B训练的网络慢5倍,即8.5GBps(68Gbps),那么:

- 通信 = `3 * 2 * 80 / 8.5` = 56秒

这与更快的计算相比肯定会是一个巨大的瓶颈。

如果网络快5倍,即212GBs(1700Gbps),那么:

- 通信 = `3 * 2 * 80 / 212` = 2秒

这相对于计算时间来说将是微不足道的,特别是如果其中一些成功地与计算重叠。

此外,Deepspeed团队在384个V100 GPU(24个DGX-2节点)上对176B模型进行了经验基准测试,发现:

1. 使用100 Gbps IB,每个GPU只有<20 TFLOPs(差)
2. 使用200-400 Gbps IB,每个GPU达到合理的30-40 TFLOPs(可以)
3. 对于800 Gbps IB,每个GPU达到40+ TFLOPs(优秀)

提醒一下,NVIDIA V100在fp16的峰值TFLOPS是125 TFLOPS(https://www.nvidia.com/en-gb/data-center/tesla-v100/)。

但要小心 - 这个基准是针对V100的!它比A100慢2-3倍,比H100慢4-8倍(半精度)。所以对于H100节点,通信必须至少快4-8倍才能在半精度下匹配上述表格。我们需要更多使用更新硬件的基准测试。

注:2-3倍范围是因为官方规格声称V100->A100和A100->H100各增加3倍TFLOPS,但用户基准测试报告的差异最多为2.5倍改进。

他们还注意到,在大规模训练时,每个GPU的小微批量大小会使通信开销更加明显。而且我们可能无法增加微批量大小,因为全局批量大小通常是固定的,以实现良好的模型收敛率。这个问题通过最近引入的ZeRO++(https://github.com/BBuf/ml-engineering/blob/master/training/model-parallelism/README.md#zero-with-multiple-replicas)得到解决。

最后,在进行上述数学计算时,你需要知道在你的设置上获得的实际带宽 - 这会随有效载荷大小而变化 - 有效载荷越大,带宽越好。要获取这些信息,你需要查看Deepspeed配置文件中的`reduce_bucket_size`和`prefetch_bucket_size`设置,分别用于reduction和预取。默认是0.5B参数,在半精度下是1GB(0.5B x 2字节),如果使用fp32精度则是2GB(0.5B x 4字节)。所以为了测量实际吞吐量,你需要用那个有效载荷运行`all_reduce`基准测试,看看报告的带宽是多少。然后你可以将其输入到上述计算中。



## 何时使用哪种策略

以下是一个非常粗略的并行策略使用指南。每个列表中的第一个通常更快。

**⇨ 单GPU**

* 模型能装入单个GPU:

    1. 正常使用

* 模型无法装入单个GPU:

    1. ZeRO + CPU卸载,可选择性地使用NVMe
    2. 如果最大的层无法装入单个GPU,则使用上述方法加上内存中心分块(详见下文)

* 最大的层无法装入单个GPU:

1. ZeRO - 启用内存中心分块(https://deepspeed.readthedocs.io/en/latest/zero3.html#memory-centric-tiling)(MCT)。它允许通过自动分割并顺序执行来运行任意大的层。MCT减少了GPU上活跃的参数数量,但不影响激活内存。不过这种需求目前很罕见,用户需要手动重写`torch.nn.Linear`。

**⇨ 单节点/多GPU**

* 模型能装入单个GPU:

    1. DDP - 分布式数据并行
    2. ZeRO - 根据具体情况和使用的配置,可能更快也可能更慢

* 模型无法装入单个GPU:

    1. PP(流水线并行)
    2. ZeRO
    3. TP(张量并行)

    在具有NVLINK或NVSwitch的快速节点内连接的情况下,这三种方法的性能应该大致相当。如果没有这些,PP会比TP或ZeRO更快。TP的程度也可能产生差异。最好在你的特定设置上进行实验以找出最优方案。

    TP几乎总是在单个节点内使用。即TP大小 <= 每个节点的GPU数量。

* 最大的层无法装入单个GPU:

    1. 如果不使用ZeRO - 必须使用TP,因为单独的PP无法装入。
    2. 使用ZeRO时,参见上面"单GPU"部分的相同条目

**⇨ 多节点/多GPU**

* 如果模型能装入单个节点,首先尝试使用多副本的ZeRO(在本文档中搜索 使用多个副本的 ZeRO),因为这样你将在更快的节点内连接上进行ZeRO,在较慢的节点间连接上进行DDP

* 当你有快速的节点间连接时:

    1. ZeRO - 因为它几乎不需要对模型进行修改
    2. PP+TP+DP - 通信更少,但需要对模型进行大量更改

* 当你有较慢的节点间连接且GPU内存仍然不足时:

    1. DP+PP+TP+ZeRO-1


# 关于并行训练知乎的一些相关文献

- [一文搞懂MPI通信接口的特点及原理](https://zhuanlan.zhihu.com/p/653968730)
- [ring attention + flash attention：超长上下文之路](https://zhuanlan.zhihu.com/p/683714620)
- [大模型训练之序列并行双雄：DeepSpeed Ulysses & Ring-Attention](https://zhuanlan.zhihu.com/p/689067888)
- [序列并行做大模型训练，你需要知道的六件事](https://zhuanlan.zhihu.com/p/698031151)
- [我爱DeepSpeed-Ulysses：重新审视大模型序列并行技术](https://zhuanlan.zhihu.com/p/703669087)
- [大模型推理序列并行](https://zhuanlan.zhihu.com/p/703669087)
