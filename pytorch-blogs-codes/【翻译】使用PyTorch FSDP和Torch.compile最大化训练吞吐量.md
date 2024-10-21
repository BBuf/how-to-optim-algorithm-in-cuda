> 博客链接：https://pytorch.org/blog/maximizing-training-throughput/。博客由 IBM 的 PyTorch 团队和 Meta 的 PyTorch 团队撰写。在[【翻译】使用PyTorch FSDP最大化训练吞吐量](https://mp.weixin.qq.com/s/6wNX38rKcFjxLb4ooYQokw)的基础上使用torch.compile以及优化dataloader把7B模型的MFU从57%提升到68%，这里只是简要的介绍了一下相关概念，具体细节大家仍然可以到开源代码中查看。https://github.com/foundation-model-stack/fms-fsdp 。最后本文补充了一下这两篇博客中的MFU计算代码。此外，在这两篇博客中发现它引用硬件FLOPS的时候链接了https://github.com/stas00/ml-engineering，我浏览了仓库中的部分内容，发现它的相关内容对做AI系统的读者来说十分有用，后续也会选择一些进行翻译。


最近(https://pytorch.org/blog/maximizing-training/)，我们展示了如何使用FSDP和选择性激活检查点来实现在A100 GPU上训练7B模型时达到57%的MFU（模型浮点运算利用率）。我们还展示了如何训练出高质量的模型，我们将其作为Granite 7B基础模型(https://huggingface.co/ibm-granite/granite-7b-base)在Hugging Face Hub上开源，采用Apache v2.0许可证。

我们继续通过利用torch.compile来提高GPU的利用率。使用torch.compile和我们之前工作中的选择性激活检查点，我们在A100 GPU上为7B模型实现了68%的MFU！torch.compile将各种模型大小的训练MFU提高了10%到23%。

本博客分为三个部分：(1) 使用torch.compile进行训练时解决的挑战，(2) compile与no-compile的数值一致性，以及(3) MFU报告。

我们已经开源了所有代码并在fms-fsdp仓库(https://github.com/foundation-model-stack/fms-fsdp)中更新了它。我们还与Meta的PyTorch团队合作，将这些贡献到新发布的torch titan(https://github.com/pytorch/torchtitan)预训练仓库中。

## 使用torch.compile的挑战

torch.compile是一种图编译技术，可以提高GPU利用率。关于torch compile的工作原理详情，我们建议读者参考最近的PyTorch论文(https://pytorch.org/blog/pytorch-2-paper-tutorial/)和相关教程。使torch.compile表现良好的一个关键挑战是最小化（或消除）图断裂。我们最初从Meta提供的Llama实现开始，但编译它会导致太多图断裂，从而降低训练吞吐量。

模型架构的几个部分必须进行修复，其中最重要的是位置嵌入层（RoPE）。典型的RoPE实现使用复数，而在测试时torch.compile不支持复数。我们使用einops实现了RoPE，同时保持与原始模型架构实现的一致性。我们必须正确缓存频率，以避免在RoPE实现中遇到图断裂。

编译FSDP模型确实会导致图断裂，Meta的PyTorch团队正在努力消除这些断裂。然而，截至PyTorch 2.3，这些图断裂发生在FSDP单元边界，并不会显著影响吞吐量。

当使用自定义kernel时，我们需要通过暴露其API来包装每个kernel以供torch.compile使用。这涉及指示哪些参数被原地修改，如何修改，以及基于输入，它们的返回值将具有什么形状和步幅。在我们的情况下，SDPA Flash attention已经适当集成，我们能够让该kernel与torch.compile一起工作，没有图断裂。

我们还注意到，当将数据量从2T增加到6T tokens时，数据加载器成为了瓶颈。这个问题的一个关键原因是，之前我们在数据加载器中朴素地实现了文档洗牌，每个worker维护一个洗牌后的文档指针列表。

随着数据集变大，这些指针列表在每个工作进程中增长到数十万个条目。维护这种规模的指针列表变得如此昂贵，以至于CPU争用限制了我们的训练吞吐量。我们使用线性同余生成器(https://en.wikipedia.org/wiki/Linear_congruential_generator)重新实现了文档洗牌，无需任何指针列表。LCG是一种伪随机数生成器算法，它在一个群体上实现随机游走，提供无替换采样。

我们利用相同的想法来产生从有序到洗牌后的文档索引的隐式双射映射。这使我们能够将那些烦人的数十万指针列表缩小到LCG的单个整数状态。这消除了80%的瓶颈，并显著提升了我们的性能。我们将专门写一篇博客，详细介绍我们高性能的预训练数据加载器的所有细节。

## torch.compile和torch.no-compile的数值一致性

我们之前观察到在使用compile和no-compile选项进行训练时存在一致性问题，其中一个与使用SDPA有关。经过Meta和IBM的PyTorch团队几天的密集调试会议，我们成功实现了PyTorch compile和no-compile模式之间的一致性。为了记录和验证这种一致性，我们采用了一个1.4B大小的mini Llama模型架构，并在四种变体下训练到100B tokens - no-compile, compile with no activation checkpointing, compile with selective activation checkpointing, and compile with full activation checkpointing。

我们在下面绘制了这些选项的损失曲线和梯度范数：

![图1：各种编译选项的损失曲线和梯度范数](https://files.mdnice.com/user/59/f0b3ef50-ad06-4403-82bb-76c207a52c8d.png)

此外，我们运行lm-evaluation-harness并比较了不同基准测试上各种模型的得分，观察到compile和no-compile之间没有重大差异，如下所示。

![图2：compile和no-compile在各种基准测试上的lm-evaluation-harness比较](https://files.mdnice.com/user/59/5984bee2-5b95-410b-a5f8-a40e219b2752.png)

从所有这些结果中，我们观察到compile及其所有变体与no-compile选项相等，从而证明了compile和no-compile之间的一致性。

## MFU报告

最后，像我们之前的博客一样，我们计算了两个集群上四种不同模型大小的MFU。一个集群是128个A100 GPU，具有400 Gbps的节点间连接；另一个是464个H100 GPU，具有3.2 Tbps的节点间连接。除了compile，我们还使用了我们在之前博客中介绍的选择性激活检查点。我们在下表中捕捉了结果。

![表1：在128个A100 80GB GPU上（400Gbps节点间互连）对Llama2模型架构进行compile和no compile的MFU结果](https://files.mdnice.com/user/59/818717a8-140f-42bd-8d20-2d263645dd66.png)


![表2：在464个H100 80GB GPU上（3.2Tbps节点间互连）对Llama2模型架构进行compile和no compile的MFU结果](https://files.mdnice.com/user/59/2b9d18ea-c285-4a9e-b425-910fd6b0bdca.png)


我们还在内部进行了一次使用448个GPU的Llama2 7B架构的生产运行。使用compile和选择性激活检查点，全局批量大小为3.7M，我们在13天10小时内训练了4T tokens！

在训练期间，数据中心的冷却系统不得不启动额外的空调，我们的训练团队收到了这方面的警报，因为我们非常有效地使用了GPU ☺

从表1和表2中的一个关键观察是，MFU数值并不随模型大小线性增加。我们正在积极调查两种可能的解释：一是随着模型大小增加，FSDP的可扩展性，以及何时需要启用张量并行以更有效地使用GPU；二是批量大小，可以进一步增加以获得更好的MFU。我们计划探索FSDP v2和选择性算子检查点，以及张量并行特性，以研究FSDP随模型大小的扩展规律。

## 未来工作

我们计划开始测试将作为PyTorch 2.4一部分发布的FSDP v2。FSDP2提供了每个参数的分片和选择性算子检查点功能，这可能会提供更好的内存-计算权衡。

我们还与Meta的PyTorch团队合作，评估新的异步检查点功能，通过减少写入检查点的时间来进一步提高GPU利用率。

我们正在探索将当前用于推理的各种Triton kernel扩展到执行反向操作，以获得推理之外的加速。

最后，随着最近关于使用fp8的工作出现，我们计划探索如何使用这种承诺2倍加速的新数据类型来进一步加速模型训练。

## 致谢

有几个团队参与到达成这个证明点，我们要感谢Meta和IBM的团队。特别是，我们向Meta的PyTorch分布式和编译器团队以及IBM研究院表示感谢。

多人广泛参与了实现我们模型的torch.compile数值一致性的努力，我们希望感谢参与这项工作的关键人员；Meta的Animesh Jain和Less Wright，以及IBM研究院的Linsong Chu、Davis Wertheimer、Brian Vaughan、Antoni i Viros Martin、Mudhakar Srivatsa和Raghu Ganti。

特别感谢Stas Bekman，他提供了广泛的反馈并帮助改进了这篇博客。他们的见解在突出优化训练的关键方面和探索进一步改进方面非常宝贵。


## MFU补充

这篇博客和 [AI Infra论文阅读之通过打表得到训练大模型的最佳并行配置](https://mp.weixin.qq.com/s/D-14J482SFQf-zh-EFa-1w)  中的MFU使用的是PaLM中的计算方法，详细解释一下。


![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/806247b2b2460d6bb0306abbd05b5a53.png)

模型浮点运算量利用率（MFU）的计算遵循PaLM的方法。我们考虑理论上的矩阵乘法峰值吞吐量为P FLOPs每秒（例如，A100 GPU的峰值矩阵乘法TFLOPs为312）。然后，模型的FLOPs利用率是实际达到的每秒处理的token数与理论峰值吞吐量R = P/(6N + 12LHQT)的比率，其中L是层数，H是注意力头的数量，Q是注意力头的大小，T是序列长度。注意，L × H等于模型的隐藏层大小。N是参数量。计算的代码如下：

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/167b3ae27f170a0a9899e64a27e3234c.png)

据我了解，对于7B来说，68%的MFU已经和Meagtron-LM的MFU十分接近了，感兴趣的朋友可以阅读下[AI Infra论文阅读之通过打表得到训练大模型的最佳并行配置](https://mp.weixin.qq.com/s/D-14J482SFQf-zh-EFa-1w) 这篇博客。

