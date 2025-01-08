> 博客来源：https://pytorch.org/blog/training-using-float8-fsdp2/ 。by IBM and Meta 。这里主要是汇总一下FSDP2和FP8训练相关的内容，目前的实践主要集中在TorchTitan（DTensor，Async Tensor Parallelism，FP8 Allgather等等）和torchao上面，包括torch.compile编译器也在做对应的支持，PyTorch对于这个工作其实还没做到很稳定，和Meagtron-LM的FP8类似处于半成品阶段，例如API接口变动就很大，这里可以先简单了解一下他们的进展。

FSDP2和FP8训练 相关前置内容：

- [【翻译】使用PyTorch FSDP最大化训练吞吐量](https://mp.weixin.qq.com/s/6wNX38rKcFjxLb4ooYQokw)
- [【翻译】使用PyTorch FSDP和Torch.compile最大化训练吞吐量](https://mp.weixin.qq.com/s/YVVau7boVUEnVB6o_qKORA)
- [【翻译】在FSDP2中开启Float8 All-Gather](https://mp.weixin.qq.com/s/44zFNWr5aVtA3zPtegY9dg)
- [[分布式训练与TorchTitan] PyTorch中的Async Tensor Parallelism介绍](https://mp.weixin.qq.com/s/Jx4B-sF9dudg7OOT-FbsLg)

# 使用float8和FSDP2加速训练

> 作者：IBM: Tuan Hoang Trong, Alexei Karve, Yan Koyfman, Linsong Chu, Divya Kumari, Shweta Salaria, Robert Walkup, Praneet Adusumilli, Nirmit Desai, Raghu Ganti, Seetharami Seelam
> Meta: Less Wright, Wei Feng, Vasiliy Kuznetsov, Driss Guesseous

在本博客中，我们将展示如何在保持损失和评估基准一致性的同时，相比[FSDP1 bf16训练](https://mp.weixin.qq.com/s/YVVau7boVUEnVB6o_qKORA)实现高达50%的吞吐量提升。我们通过利用FSDP2、DTensor和torch.compile与torchao的float8线性层更新（计算）以及float8 all_gathers进行权重通信来实现这一提升。我们展示了这些改进在Meta LLaMa模型架构的不同规模上的效果，从1.8B小型模型一直到405B大型模型，使训练速度比以往更快。

我们使用Meta Llama3架构展示这些改进，并在两个规模上进行模型质量研究：8B模型规模的100B tokens训练和70B模型规模的50B tokens训练，这提供了float8和bf16训练损失曲线的精确比较。我们证明了与`bf16`相比，这些模型训练运行的损失曲线达到了相同的损失收敛。此外，我们使用FineWeb-edu数据集训练了一个3B模型到1T tokens，并运行标准评估基准以确保模型质量完整且与bf16运行相当。

在IBM研究院，我们计划采用这些功能进行数据消融实验，以提高在给定GPU预算内可以执行的实验数量。从长远来看，我们将通过更大规模的模型运行来展示`float8`训练的端到端可行性。

## 什么是Float8？

`float8`训练格式是由NVIDIA、ARM和Intel在2022年的一篇论文(https://arxiv.org/abs/2209.05433)中提出的，该论文证明了使用更低精度float8进行训练的可行性，且不会牺牲模型质量。随着NVIDIA Hopper系列等新型GPU的推出，由于原生float8张量核心支持，FP8训练变得可行，有望实现超过2倍的训练吞吐量提升。实现这一承诺面临一些挑战：
(i) 在`float8`中启用核心模型操作如`matmul`和`attention`，
(ii) 在分布式框架中启用`float8`训练，
(iii) 在`float8`中启用GPU之间的权重通信。
虽然NVIDIA库启用了`float8` `matmul`，但后两项是在FSDP2和torchao的最新更新中提供的。

在本博客中，我们使用torchtitan(https://github.com/pytorch/torchtitan)作为训练入口点，使用IBM的确定性数据加载器，来自torchao的`float8`线性层实现，以及最新PyTorch nightly版本中的`float8 all gather`与FSDP2结合。对于这次训练，我们使用的是`float8`每张量（tensorwise）缩放粒度而不是行级。我们利用`torch.compile`确保获得最大性能提升。我们使用SDPA在`bf16`中计算`attention`，目前正在努力将其也迁移到`float8`。

## 实验

我们进行了各种实验来展示float8训练的优势。首先是确保不会牺牲模型质量。为了验证这一点，我们训练了一个8B模型和70B模型几千步，并比较float8和bf16训练运行之间的损失曲线。我们的实验在三个不同的H100集群上进行，分别配置了128、256和512个H100 GPU，环境各不相同，以证明可重复性。第一个集群是Meta的Grand Teton(https://engineering.fb.com/2024/03/12/data-center-engineering/building-metas-genai-infrastructure/)上的定制集群，具有400Gbps定制互连；第二个是IBM研究集群，具有3.2Tbps Infiniband互连；第三个是IBM Cloud集群，具有3.2Tbps RoCE互连用于GPU到GPU通信。

首先，我们在下面的图中绘制了这两个模型的损失曲线比较，以展示几千步的损失一致性。

![](https://files.mdnice.com/user/59/203aebf3-a43b-434d-b6e3-c0f9aa60a18d.png)

![](https://files.mdnice.com/user/59/4afe8744-7668-4e20-b34f-b3f02c5ca696.png)

图1：(a) 8B模型2k步损失一致性，(b) 70B模型1k步损失一致性

我们观察到，在这些不同的模型和不同的环境中，我们在小规模tokens训练中获得了损失一致性。接下来，我们对从1.8B到405B的四种不同模型规模的吞吐量增益进行了表征。我们探索了float8和bf16训练运行的最佳批量大小和激活检查点方案，以确定**每GPU每秒的tokens数（wps）**指标并报告性能增益。对于405B模型，我们利用DTensor进行张量并行训练与FSDP2。我们所有的测量都使用8K的序列长度。

![](https://files.mdnice.com/user/59/49c71820-f879-4890-bd2f-8ce25b8c21ad.png)

表1：相对于bf16的性能增益（bf16和float8都使用torch.compile）

从表1中我们观察到，较大模型（70B和405B）的增益达到50%，较小模型的增益在20%到30%之间。在进一步的实验中，我们观察到float8 all_gather的添加使性能在float8计算本身的基础上提升了约5%，这与这篇博客(https://aws.amazon.com/cn/blogs/machine-learning/efficient-pre-training-of-llama-3-like-model-architectures-using-torchtitan-on-amazon-sagemaker/)中的观察结果一致。

其次，为了展示FP8模型的有效性，我们使用来自Hugging Face的FineWeb-edu数据集训练了一个遵循Llama3架构的3B模型，训练量达到1T tokens。我们使用lm-eval-harness框架进行评估，并在下表中展示了部分结果。我们观察到bf16的性能略优于float8分数（约一个百分点）。虽然某些分数在bf16下明显更好（例如，MMLU高出3分），但我们预计当选择正确的超参数和进行更大规模的训练运行时，这些差距会消失（例如，bf16运行的批量大小是一半，众所周知较小的批量大小运行可以提高评估分数）。

![](https://files.mdnice.com/user/59/60b80721-2fd9-4c69-8a24-6341e665a4be.png)

表2：float8训练模型在FP16下进行评估的基准分数（在FineWeb预训练的1T tokens处）。

最后，我们将实验扩展到IBM Cloud集群的512个H100 GPU上。我们能够在512 GPU规模上重现我们观察到的结果和加速。我们在下表中仅总结了大型模型（70B和405B）的这些结果。

![](https://files.mdnice.com/user/59/6204da43-dca0-465e-af4b-d73efa930024.png)

表3：512 GPU规模下相对于bf16的性能增益（bf16和float8都使用torch.compile）

## 未来工作

我们还在研究其他形式的并行性，如上下文并行性。我们计划评估所有这些特性，以展示可组合性和为大规模模型训练做出选择的能力。

## 致谢

我们感谢IBM Research的Davis Wertheimer为torchtitan运行启用数据加载器，使我们能够在多次运行中以相同顺序重放数据。我们还感谢IBM Cloud为我们提供H100集群的早期测试访问权限。

