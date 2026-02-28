> 博客链接：https://pytorch.org/blog/maximizing-training/ 。博客由 IBM 的 PyTorch 团队和 Meta 的 PyTorch 团队撰写。目前Torch也持续在训练Infra上面推理，除了DeepSpeed，Meagtron-LM之外，我们也可以选择PyTorch的FSDP来训练更大的例如72B内的模型。这篇博客介绍了基于FSDP如何对7B/13B/34B/70B的模型在A100/H100上高效训练，所有代码均开源在：https://github.com/foundation-model-stack/fms-fsdp 。除了这个博客中介绍的Pretrain和SFT训练之外，源码里也提供了HF的转换脚本让这个训练的模型可以使用Hugging Face生态中的Post Traning工具。

在这篇博客中，我们展示了 FSDP 的可扩展性，以一个预训练示例（一个训练了 2T 个 token 的 7B 模型）为例，并分享了我们用于实现每个 GPU 3,700 个 token/秒的快速训练速度的各种技术，即在 128 个 A100 GPU 上每天处理 40B 个 token。这相当于 57% 的模型 FLOPS 利用率（MFU）和硬件 FLOPS 利用率（HFU）。此外，我们观察到 FSDP 在扩展到 512 个 GPU 时几乎呈线性增长，这意味着使用这种方法在 512 个 GPU 上训练一个 7B 模型到 2T 个 token 只需不到两周的时间。

IBM 的研究人员将 Meta Llama 2 7B 架构训练了 2T 个 token，我们将其称为 LlamaT(est)。这个模型在各种学术基准测试中展示了与 Llama 2 相当的模型质量。所有的训练代码(https://github.com/foundation-model-stack/fms-fsdp)，以及我们实现这一吞吐量的方法，都可以在这篇博客中找到。我们还分享了适用于 Llama 2 模型的配置参数 - 针对 A100 和 H100 的 7B、13B、34B 和 70B 模型。

在这个过程中，我们还提出了一种适用于 FSDP 的新的选择性activation checkpointing机制，这使我们比开箱即用的 FSDP 提高了 10% 的性能。我们已经开源了训练代码库(https://github.com/foundation-model-stack/fms-fsdp)和相关的可扩展数据加载器，作为实现这一吞吐量的方法。

PyTorch 原生训练路径的一个关键优势是能够无缝地在多个硬件后端上进行训练。例如，最近由 AllenAI 通过 OLMo 发布的端到端训练栈也利用 PyTorch FSDP 在 AMD 和 NVIDIA GPU 上进行训练。我们利用 FSDP 的三个主要组件来实现我们的吞吐量：

- SDPA Flash attention(https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html)，它支持融合注意力内核和高效的注意力计算
- 计算和通信的重叠允许更好地利用 GPU(https://engineering.fb.com/2021/07/15/open-source/fsdp/)
- Selective activation checkpointing (https://arxiv.org/pdf/2205.05198)使我们能够在 GPU 内存和计算之间进行权衡

IBM 与 Meta 的 PyTorch 团队合作了近两年时间，在 PyTorch FSDP(https://arxiv.org/abs/2304.11277) 上：引入rate limiter(https://pytorch.org/blog/scaling-pytorch-fsdp-for-training-foundation-models-on-ibm-cloud/)以在以太网互连上实现更好的吞吐量，distributed checkpointing(https://pytorch.org/blog/performant-distributed-checkpointing/)以将checkpointing时间提高一个数量级，并为 FSDP 的混合分片模式实现早期版本的checkpointing。去年年底，我们使用 FSDP 端到端地训练了一个模型。

## 训练细节
7B模型在128个A100 GPU上进行训练，网络连接带宽为400Gbps，并使用GPU Direct RDMA。我们使用SDPA FlashAttention v2进行注意力计算，对于这个模型，我们关闭了限制批量大小的activation checkpointing，提供了最高的吞吐量 - 128个GPU的批量大小为每批100万个token，与activation checkpointing相比，吞吐量提高了约10%。使用这些参数，我们实现了计算和通信的几乎完全重叠。我们使用32位的AdamW优化器，beta1为0.9，beta2为0.95，权重衰减为0.1，学习率最终为3e-5，预热到最大学习率3e-4，并在2T个token上使用余弦调度降低到3e-5。训练使用混合精度bf16在内部数据集上进行。训练栈使用IBM的Foundation Model Stack(https://github.com/foundation-model-stack/foundation-model-stack/blob/main/fms/models/llama.py)作为模型架构，并使用PyTorch 2.2发布后的每日构建版本进行FSDP和SDPA。我们在2023年11月至2024年2月期间尝试了几个不同的每日构建版本，观察到吞吐量有所提高。

### Selective activation checkpointing

我们共同实现了一种简单的选择性activation checkpointing（AC）机制。在FSDP中，常见做法是checkpointing每个transformer块。一个简单的扩展是checkpointing每_n_个块，并减少重计算量，同时增加所需的内存。这对于13B模型大小非常有效，将吞吐量提高了10%。对于7B模型大小，我们根本不需要activation checkpointing。未来版本的FSDP将在操作符级别提供选择性activation checkpointing，以实现最佳的计算-内存权衡。上述代码在这里实现(https://github.com/foundation-model-stack/fms-fsdp/blob/main/fms_fsdp/policies/ac_handler.py)。


```python
from functools import partial

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)

# 创建一个非重入的checkpoint包装器
non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)

def apply_fsdp_checkpointing(model, block, p):
    """
    应用selective activation checkpointing。

    选择性由百分比p定义，这意味着我们在总块数的p比例上应用activation checkpointing。
    p是一个在[0, 1]范围内的浮点数。

    一些例子：
    p = 0: 所有块都不应用activation checkpointing。等同于`fsdp_activation_checkpointing=False`
    p = 1: 在每个块上应用activation checkpointing。即"完全activation checkpointing"。
    p = 1/2: [activation checkpointing, 无activation checkpointing, activation checkpointing, 无activation checkpointing, ...]
    p = 1/3: [无activation checkpointing, activation checkpointing, 无activation checkpointing, 无activation checkpointing, activation checkpointing, 无activation checkpointing, ...]
    p = 2/3: [activation checkpointing, 无activation checkpointing, activation checkpointing, activation checkpointing, 无activation checkpointing, activation checkpointing, ...]
    由于块是同质的，我们使activation checkpointing块在所有块中均匀分布。

    实现：
    对于给定的activation checkpointing比率p，我们本质上应该在每"1/p"个块上应用activation checkpointing。
    第一个activation checkpointing块可以早至第0个块，或晚至第"1/p"个块，我们选择中间的一个：第(0.5p)个块。
    因此，我们本质上是在以下块上应用activation checkpointing：
    第(0.5/p)个块、第(1.5/p)个块、第(2.5/p)个块等，当然，这些值会四舍五入到整数。
    由于activation checkpointing是递归应用的，我们可以在代码中简单地使用以下数学方法来在相应的块上应用activation checkpointing。
    """
    block_idx = 0
    cut_off = 1 / 2
    # 当p作为分数传递时（例如1/3），它在argv中会被解释为字符串，
    # 因此我们需要在这里对分数使用eval("1/3")。
    p = eval(p) if isinstance(p, str) else p

    def selective_checkpointing(submodule):
        nonlocal block_idx
        nonlocal cut_off

        if isinstance(submodule, block):
            block_idx += 1
            if block_idx * p >= cut_off:
                cut_off += 1
                return True
        return False

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=selective_checkpointing,
    )
```

### 吞吐量和 MFU、HFU 计算

虽然我们只将 7B 模型训练到 2T 个 token，但我们对其他模型大小进行了大量实验，以提供最佳配置选项。下表总结了两种基础设施的结果 — 一个是具有 128 个 GPU 和 400Gbps 节点间互连的 A100 集群，另一个是具有 96 个 GPU 和 800Gbps 节点间互连的 H100 集群。

| 模型大小 | 批量大小 | activation checkpointing | 吞吐量 tokens/秒/GPU (A100 80GB 和 400Gbps 互连) | MFU % (A100 80GB) | HFU % (A100 80GB) | 吞吐量 tokens/秒/GPU (H100 80GB 和 800Gbps 互连) | MFU % (H100 80GB) | HFU % (H100 80GB) |
|---------|---------|------------|-------------------------------------------------|------------------|------------------|-------------------------------------------------|------------------|------------------|
| 7B | 2 | 否 | 3700 | 0.57 | 0.57 | 7500 | 0.37 | 0.37 |
| 13B | 2 | 选择性 | 1800 | 0.51 | 0.59 | 3800 | 0.35 | 0.40 |
| 34B | 2 | 是 | 700 | 0.47 | 0.64 | 1550 | 0.32 | 0.44 |
| 70B | 2 | 是 | 370 | 0.50 | 0.67 | 800 | 0.34 | 0.45 |

表1：A100和H100 GPU上各种模型大小的模型和硬件FLOPS利用率

HFU数值是使用PyTorch FLOP计数器(https://github.com/pytorch/pytorch/blob/2240018c03744ee34ea14ad53481db934c37e384/torch/utils/flop_counter.py#L336)和A100、H100 GPU的理论bf16性能计算得出的，而MFU数值则是使用NanoGPT(https://github.com/karpathy/nanoGPT)和PaLM论文中概述的方法计算得出的。我们还注意到，对于较大的模型，我们故意将每个GPU的批量大小保持在2，以模仿训练4k序列长度模型时的选择，并在不超过流行的4M tokens批量大小的情况下，实现最多512个GPU的规模。超过这个规模，我们就需要使用张量并行或序列并行。

我们在上表中注意到，对于A100，activation checkpointing导致MFU降低，而HFU增加！随着更好的activation checkpointing方案的引入，我们预计MFU将会增加并赶上HFU。然而，我们观察到对于H100，MFU和HFU都相对较低。我们分析了H100上的PyTorch性能分析跟踪，发现由于网络"peeking"导致了10%的差距。此外，我们推测H100的HBM带宽是导致H100上HFU/MFU降低的原因，而不能获得3倍的性能提升（理论上H100比A100快3倍 - 312对989TFLOPS，但HBM带宽仅为A100的<2倍 - 2.0对3.35TBps）。我们计划尝试其他配置选项，如张量并行，以改善H100上70B模型的性能参数。

## 模型细节
训练的损失曲线如下图所示。

![](https://files.mdnice.com/user/59/a75ae192-f38d-4d05-ad95-d5841e346be3.png)

图1：LlamaT训练损失曲线

2T checkpointing 通过仓库中提供的脚本转换为Hugging Face格式，然后我们使用lm-evaluation-harness计算关键学术基准，并通过在Llama2-7B上运行来进行比较。这些结果在下表中捕获。

![](https://files.mdnice.com/user/59/63bbfb3b-9f28-40eb-91af-f546604f7fd3.png)

表1: LM评估工具得分

我们观察到该模型与Llama2相比表现具有竞争力(更粗体表示更好)。

## 训练记录
训练过程总体稳定，没有发生崩溃，但我们确实观察到了一些小问题：

**0-200B tokens**：我们观察到迭代时间（执行一个训练步骤所需的时间）有所减慢。我们停止了任务，以确保数据加载器没有造成任何减速，并且 checkpointing 操作是高效和准确的。我们没有发现任何问题。此时，PyTorch中已经有了HSDP checkpointing 代码，我们借此机会切换到了PyTorch的 checkpointing 代码。

**200B tokens-1.9T**：在12月下旬，我们没有对任务进行任何手动干预。当我们在1月初回来时，发现磁盘空间已经超出， checkpointing 无法写入，尽管训练任务仍在继续。最后已知的 checkpointing 是在1.5T。

**1.5T-1.7T**：我们使用lm-evaluation-harness评估了1.5T checkpointing ，发现模型在两个文档之间训练了一个额外的特殊token，这是因为Hugging Face分词器引入了一个分隔符token，而我们的数据加载器也附加了自己的文档分隔符。我们修改了数据加载器以消除额外的特殊token，并从1.7T token开始继续使用修改后的数据加载器进行训练。

**1.7T-2T**：由于特殊token的变化，损失最初出现了峰值，但在几十亿个token后迅速恢复。训练在没有任何其他手动干预的情况下完成！

## 关键要点和更多加速

我们展示了如何使用FSDP训练一个模型到2T个token，实现了每个GPU 3700 tokens/秒的出色性能，并生成了一个高质量的模型。作为这个实验的一部分，我们开源了所有用于训练的代码和实现这种吞吐量的调节参数。这些参数不仅可以用于大规模运行，也可以用于小规模的调优运行。你可以在这里找到代码(https://github.com/foundation-model-stack/fms-fsdp)。

FSDP API以PyTorch原生方式实现了ZeRO(https://pytorch.org/docs/stable/fsdp.html)算法，允许调优和训练大型模型。在过去，我们已经看到FSDP的证明点（Stanford Alpaca(https://github.com/tatsu-lab/stanford_alpaca)、Hugging Face(https://huggingface.co/blog/ram-efficient-pytorch-fsdp)、Llama 2 recipes(https://github.com/meta-llama/llama-recipes)）在使用简单的训练循环调优各种LLM（如Meta Llama 2 7B到70B Llama）时，实现了良好的吞吐量和训练时间。

最后，我们注意到有几个加速训练的杠杆：

- 可以加速特定操作的节点优化（例如，使用Flash Attention V2进行注意力计算）
- 图优化（例如，fuse kernels, torch.compile）
- 计算-通信重叠
- activation recomputation

在这篇博客中，我们利用了1、3和4的变体，并正在与Meta的PyTorch团队密切合作，以获得torch.compile（2）以及per-operator selective activation recomputation的更高级版本的4。我们计划分享一个简单的格式化代码和示例数据，以便导入到我们的数据加载器中，使其他人能够使用这个代码库进行模型训练。

## 致谢

有几个团队参与到达成这个证明点的过程中，我们想要感谢Meta和IBM的团队。具体来说，我们向PyTorch分布式团队、Facebook研究和应用AI团队表示感谢，他们构建了FSDP API(https://arxiv.org/abs/2304.11277)并根据我们的反馈进行了改进。我们还要感谢IBM研究院的数据团队，他们整理了本次实验中使用的数据语料库，以及IBM研究院的基础设施团队（特别是Claudia Misale、Shweta Salaria和Seetharami Seelam），他们优化了NCCL和网络配置。通过构建和利用所有这些组件，我们成功地展示了LlamaT的证明点。

Selective activation checkpointing的概念由IBM的Linsong Chu、Davis Wertheimer、Mudhakar Srivatsa和Raghu Ganti提出，并由Meta的Less Wright实现。

特别感谢Stas Bekman和Minjia Zhang，他们提供了大量反馈并帮助改进了这篇博客。他们的见解在突出训练优化的关键方面和探索进一步改进方面非常宝贵。

## 附录

### 通信计算重叠

在多节点设置中训练的另一个关键方面是重叠通信和计算的能力。在FSDP中，有多个重叠的机会 - 在前向传播的FSDP单元gather阶段以及反向传播计算期间。在前向传播期间重叠gather与前一个单元的计算，以及重叠反向计算与下一个单元的gather和梯度ReduceScatter，可以将GPU利用率提高近2倍。我们在400Gbps网络互连的A100 80GB GPU上展示了这一点。对于HSDP，在前向传播的预取阶段没有节点间流量，重叠仅发生在反向梯度计算阶段。当然，HSDP只有在模型可以在单个节点内分片时才可行，将模型大小限制在约30B参数左右。

下图显示了FSDP中的三个步骤，图像下半部分底部是节点之间的通信，顶部是计算流。对于没有激活重计算的7B模型，我们观察到重叠是完整的。在实践中，可能的重叠百分比是90%，因为前向传播期间的第一个块和反向传播期间的最后一个块无法重叠。

![](https://files.mdnice.com/user/59/39d5353a-928c-4f37-b063-389f9659be1d.png)

上述三步过程的单个步骤的放大视图如下所示。我们可以清楚地看到计算和通信的粒度，以及它们如何以交错的方式重叠。

![](https://files.mdnice.com/user/59/91edbee5-8af9-4726-b03d-5231f04aafb2.png)



> 实际上FSDP就是Zero3，这里提到的重叠需要结合Zero3的原理来理解，它告诉我们为什么可以在两个layer之间重叠通信和计算。推荐阅读 https://zhuanlan.zhihu.com/p/485208899 这篇文章里面的 FSDP 的工作原理图。
