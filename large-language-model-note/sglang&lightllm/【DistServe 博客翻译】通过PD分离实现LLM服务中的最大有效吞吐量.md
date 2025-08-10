> 博客转载于：https://hao-ai-lab.github.io/blogs/distserve/ ，仅用于学习和知识分享 。博客对应的论文见：https://hao-ai-lab.github.io/blogs/distserve/。

# 吞吐量不是全部：通过Prefill-Decode分离实现LLM服务中的最大有效吞吐量

> 2024年3月17日 · 13分钟 · Junda Chen, Yinmin Zhong, Shengyu Liu, Yibo Zhu, Xin Jin, Hao Zhang

![一个请求通过具有分离 Prefill 和 Decode 功能的LLM服务引擎](https://files.mdnice.com/user/59/ea895763-c3fc-42f6-bb4b-d4664d9a0926.png)

> 注意：这张图原博客里面是一张动图，可以点击原博客链接获得最佳体验。

**TL;DR**: 当今的LLM应用具有多样化的延迟要求。例如，聊天机器人可能需要快速的初始响应（例如，在0.2秒内），但 Decode 速度适中，只需要匹配人类阅读速度，而代码补全则需要快速的端到端生成时间以提供实时代码建议。

在这篇博客文章中，我们展示了优化**吞吐量**的现有服务系统在延迟标准下并不是最优的。我们主张使用**有效吞吐量**（goodput），即每秒完成的符合服务级别目标（SLO）的请求数量，作为LLM服务性能的改进衡量标准，以同时考虑成本和用户满意度。

为了优化有效吞吐量，我们引入了Prefill-Decode分离，即将 Prefill 与 Decode 分离到不同的GPU中。我们还构建了一个系统原型DistServe，与现有最先进的服务系统相比，在严格的延迟约束下实现了高达4.48倍的有效吞吐量或10.2倍更严格的SLO。我们正在将DistServe与vLLM集成，以将该技术带给社区。

# 背景：吞吐量 vs 有效吞吐量

大语言模型（LLMs）正在改变行业在其服务中采用AI技术的方式，但LLM服务的成本仍然很高。为了降低服务成本，当今许多公司专注于最大化整体LLM服务系统的**吞吐量**，即每秒服务的请求数量（或rps），作为最小化**每个请求的美元成本（$/req）**的代理。几乎所有流行的LLM服务引擎，如vLLM和TensorRT-LLM，都使用吞吐量作为相互比较性能的主要指标。

实际上，下游应用有不同的类型——它们可能对用户体验有不同的延迟要求，因此需要满足截然不同的服务级别目标（SLO）。LLM服务中最广泛使用的SLO包括：

- 首token延迟时间（**TTFT**）：测量LLM向用户输出第一个生成token所需的时间。
- 每个输出token的时间（**TPOT**）：测量两个后续生成token之间的平均延迟。

![图0. 应用具有多样化的SLO。](https://files.mdnice.com/user/59/2bb66bd4-632f-4d04-a50b-4391f6e39b69.png)


吞吐量测量所有用户和请求完成的请求或token数量，因此忽略了这些延迟要求。我们引入**有效吞吐量**（goodput），即每秒完成的符合SLO（TTFT和TPOT要求）的请求数量，并表明它是一个更好的指标，因为它捕获了SLO达成下的请求吞吐量——因此同时考虑了成本和服务质量。

为了简要说明有效吞吐量，假设一个应用要求TTFT < 200毫秒且TPOT < 50毫秒，至少90%的请求满足这些条件，我们得到以下定义：

有效吞吐量（P90 TTFT < 200ms且P90 TPOT < 50ms）= 当至少90%的请求同时满足TTFT < 200ms和TPOT < 50ms时的最大每秒请求率

**图1**显示了一个简单的情况，其中高吞吐量的应用可能具有低有效吞吐量。该应用的吞吐量为每秒10个请求。但在延迟约束下，只有3个请求保持在SLO约束内，产生每秒3个请求的有效吞吐量。正如您可以想象的，使用这种高吞吐量但低有效吞吐量的服务系统的用户仍然会遭受低服务质量的影响。

![图1. 高吞吐量 ≠ 高有效吞吐量。优化吞吐量的系统在特定SLO约束下可能具有低有效吞吐量。](https://files.mdnice.com/user/59/1768c8c3-8bcd-4a62-b8d2-698b1b501088.png)

让我们总结一下本小节中引入的术语：

- **有效吞吐量（Goodput）**：衡量LLM服务系统有效性的指标，同时考虑成本和用户满意度。它被定义为系统在满足指定服务级别目标（SLO）的同时能够维持的最大每秒请求率。
- **吞吐量（Throughput）**：LLM服务系统每秒处理的完成请求数量。
- **服务级别目标（SLO）**：LLM服务系统必须满足的一组目标，以提供令人满意的用户体验。常见的SLO包括首token时间（TTFT）、每个输出token时间（TPOT）、端到端延迟（E2E）和指数移动平均（EMA）延迟。
- ** Prefill （Prefill）**：LLM推理的第一阶段，消化所有输入token，填充 KV Cache ，并生成第一个输出token。
- ** Decode （Decode）**：后续阶段，逐个token生成直到终止。
- **首token时间（TTFT）**：LLM服务系统响应用户请求生成第一个token所需的时间。
- **每个输出token时间（TPOT）**：LLM服务系统响应用户请求生成后续token的平均时间。

# 为什么现有系统无法实现高有效吞吐量？

## LLM请求是如何被处理的？

在我们深入探讨之前，让我们重新回顾LLM服务中请求的生命周期。图2显示了这个过程。当请求进入LLM推理引擎时，系统将首先使用用户输入生成第一个token（** Prefill **），然后自回归地逐个token生成输出（** Decode **）。一个请求通常包含一个 Prefill 步骤和多个 Decode 步骤直到终止。

LLM服务系统通常使用称为迭代级调度或连续批处理的技术将所有 Prefill 和 Decode 一起批处理，以便GPU处理尽可能大的批次大小，运行一次迭代，并为所有这些请求生成一个token。这种技术有效地提高了整体吞吐量（每秒token数），并被vLLM和TensorRT-LLM等流行服务系统广泛采用。

![图2. 请求在传统LLM服务系统中是如何被处理的。](https://files.mdnice.com/user/59/73f0869a-59e6-4b86-95ac-c647f2fb80bd.png)

> 注意：这张图原博客里面是一张动图，可以点击原博客链接获得最佳体验。

然而，**这两个阶段在计算方面具有非常不同的特征**。 Prefill 是计算密集型的，意味着一小批 Prefill 甚至单个足够长的 Prefill 就很容易使GPU计算饱和。另一方面， Decode 需要更大的批次大小才能达到计算瓶颈，更容易受到GPU内存带宽限制的影响。

由于它们截然不同的计算模式和SLO，将这两个阶段共置对于实现高有效吞吐量并不是最优的，因为：

- 共置 Prefill 和 Decode 会导致它们之间的干扰。
- 共置 Prefill 和 Decode 耦合了它们的资源分配和并行策略。

我们接下来解释这些问题。

## 共置 Prefill 和 Decode 会导致干扰

**图3**显示了 Prefill 和 Decode 之间干扰的简化视图。在最左侧，我们将这2个请求在1个GPU中一起批处理。我们可以看到连续批处理显著延长了R1（ Decode ）的延迟，同时略微增加了R2（ Prefill ）的延迟。在右侧，我们有稳定的传入请求流。现在 Decode 阶段的请求每次有 Prefill 请求进入系统时都会"卡住"，导致 Decode 出现意外长的延迟。

![图3. 连续批处理导致干扰](https://files.mdnice.com/user/59/92e41eef-cca9-47a7-8a5b-468b9b11aeda.png)

由于这种干扰，如图4所示，当服务必须同时满足TTFT和TPOT SLO时，系统必须过度配置资源以满足延迟目标，特别是当任一SLO都很严格时。

![图4. 为了满足SLO，共置 Prefill 和 Decode 的系统需要过度配置资源以满足SLO目标](https://files.mdnice.com/user/59/41f7898b-3115-4182-a1c0-e75fc7f9f55e.png)

## 资源分配和并行策略是耦合的

此外，通过共置， Prefill 和 Decode 计算的并行策略（张量、流水线或数据并行）本质上是耦合的。如前所述，由于它们不同的计算模式和延迟目标， Prefill 和 Decode 阶段的最优并行策略通常不同。例如，当TTFT严格且TPOT宽松时， Prefill 阶段倾向于张量并行（TP）以满足严格的延迟目标，而 Decode 阶段倾向于数据或流水线并行以提高吞吐量。我们接下来描述解决这些问题的我们的新方法。

# Prefill-Decode分离

直觉很简单：将 Prefill 和 Decode 分离到不同的GPU中，并为每个阶段定制并行策略。这自然解决了上述两个问题：

- ** Prefill 和 Decode 之间没有干扰**使两个阶段都更快，更容易达到各自的SLO。
- **解耦的资源分配和并行策略**使得优化可以分别针对 Prefill 和 Decode 进行定制。

**图5**说明了请求在这种分离系统中是如何被处理的。当请求到达系统时，它首先进入 Prefill 工作器并完成其 Prefill 阶段。然后系统将其中间状态（主要是 KV Cache ）迁移到** Decode 工作器**，并执行多个 Decode 步骤以生成后续token。请求在完成生成后离开系统。

![图5. 当 Prefill / Decode 分离时请求是如何被处理的。](https://files.mdnice.com/user/59/039b7bf5-8ea6-4e8c-ac15-436c5f848422.png)

> 注意：这张图原博客里面是一张动图，可以点击原博客链接获得最佳体验。

让我们通过一个简单的实验来看看为什么分离是有益的。我们在单个A100-80GB GPU上服务13B LLM，使用长度为512的输入和长度为64的输出，遵循泊松到达的合成工作负载。我们逐渐增加请求率（x轴）并测量两个延迟（P90 TTFT和P90 TPOT，y轴）在**图6**中的变化。

假设我们将P90 TTFT的SLO设置为0.4秒，P90 TPOT设置为0.04秒（**图6**中的水平线）。我们观察到现有系统使用1个GPU可以支持大约3 rps保持在TTFT延迟约束内，而对于TPOT，它维持1.6 rps（**图6a**）。由于我们需要同时满足两个约束，现有共置系统的有效吞吐量变为：有效吞吐量（共置）= min(3, 1.6) = 1.6 rps（每GPU）。

分离后性能显著提升。如果只处理一个阶段， Prefill 工作器和 Decode 工作器都可以比之前实现更好的rps——如**图6**所示，一个 Prefill 工作器实现大约5.6 rps，一个 Decode 工作器实现大约10 rps。更重要的是，现在我们可以灵活地分配2个 Prefill 工作器与1个 Decode 工作器配对（记为2P1D），总共3个GPU。有效吞吐量变为：

有效吞吐量（2P1D）= min(5.6 x 2, 10) = 10 reqs/s / 3 GPUs ≈ 3.3 reqs/s（每GPU）。

这个实验表明，这种没有任何并行的简单分离产生了2倍的有效吞吐量。

![图6. 共置（a）比分离（b）灵活性更少，后者为 Prefill 分配2个GPU，为 Decode 分配1个GPU（2P1D）。](https://files.mdnice.com/user/59/6e043ebe-a232-4dfd-9135-c43b4c43d0bc.png)

实际上，除了为每个阶段分配不同的资源外，分离 Prefill 和 Decode 进一步使我们能够为每个阶段选择最佳的并行策略以优化有效吞吐量（称为"定制并行"），我们在我们的论文中详细研究了这一点。

##  KV Cache 传输

分离的代价是在 Prefill 和 Decode GPU之间传输中间状态（即 KV Cache ）。乍一看， KV Cache 是LLM推理中的一大内存支出，GPU之间 KV Cache 的传输听起来像是一个瓶颈。然而，我们展示了相反的情况：通过适当的放置， KV Cache 传输开销可以有效地最小化到低于 Decode 步骤的时间，这要归功于当今的高速网络，如NVLink和PCI-e 5.0。

为了看到这一点，假设我们有8通道PCIe 5.0 x 16（每链路64GB/s）作为GPU之间的节点内网络。给定一个2048个token的请求，我们对服务OPT-175B时传输 KV Cache 有以下估计：

延迟 = 2048 tokens * (4.5 MB/token) / (64GB/s * 8) = 17.6 ms

这个延迟小于OPT-175B的单个 Decode 步骤（在A100上约30-50ms）。对于更大的模型、更长的序列或更先进的网络（例如，带宽为600GB/s的A100-NVLink），如图7所示，相对于单个 Decode 步骤，与 KV Cache 传输相关的比较开销变得不那么显著。总之，仔细放置 Prefill 和 Decode 工作器以利用高带宽网络可以有效地隐藏 KV Cache 传输开销，这在我们论文中有详细讨论。

![图7.  KV Cache 传输开销可以有效地最小化到低于 Decode 步骤时间的程度。](https://files.mdnice.com/user/59/e33b153f-c638-451c-b79c-04b6bd7e8795.png)

## DistServe：评估分离的有效性

我们在一个称为DistServe的系统原型中实现了所提出的技术，并将其与现有系统在三个具有不同延迟约束的工作负载和数据集上进行比较：聊天机器人、代码补全和摘要，如下表所示。

| LLM应用 | 数据 | TTFT | TPOT |
|---------|------|------|------|
| 聊天机器人 | ShareGPT | 严格 | 中等 |
| 代码补全 | HumanEval | 严格 | 严格 |
| 摘要 | LongBench | 宽松 | 中等 |

> 表8. 我们评估中的工作负载和延迟要求

**图9**显示了DistServe与vLLM比较的结果：

- **聊天机器人**：DistServe与vLLM相比维持2.0x - 3.41x更高的有效吞吐量。
- **代码补全**：DistServe与vLLM相比维持3.2x更高的有效吞吐量和1.5x更严格的SLO。作为实时编码助手，代码补全任务比聊天机器人需要更低的TTFT，这导致两个系统最终都受到TTFT要求的约束。然而，通过消除 Decode 作业的干扰并为 Prefill 定制张量并行，DistServe减少了 Prefill 作业的平均延迟，从而满足更多请求的TTFT要求。
- **摘要**：DistServe与vLLM相比实现4.48x更高的有效吞吐量和10.2x更严格的SLO。正如预期的那样，由于vLLM将 Prefill 和 Decode 共置在一起，它在 Decode 中经历了更大的减速，无法满足TPOT要求。

有关更细粒度的实验结果，请参阅我们的技术报告。

![图8. 在各种数据集上DistServe与vLLM的评估。](https://files.mdnice.com/user/59/a48b4868-9907-47b5-b2bc-ee012026ae48.png)

## Prefill-Decode分离 vs Chunked Prefill

在本节中，我们将Prefill-Decode分离与最近的方法动态splitfuse（也称为Chunked Prefill和捎带）进行比较，并讨论它们的优缺点。

动态splitfuse的关键思想是将长 Prefill 分割成更小的块，从而形成一个通过将 Prefill 块与几个 Decode 任务组合来完全利用GPU的批次，这个过程被称为捎带。**块大小**是根据工作负载故意选择的，以便这种方法可以在所有步骤中保持GPU完全利用以提高整体系统效率。然而，它可能会增加TTFT和TPOT，在延迟约束下可能降低有效吞吐量。这是由于它无法完全分离 Prefill 和 Decode 操作，导致资源争用和TTFT与TPOT之间的妥协。

**对于TTFT**，Chunked Prefill会导致 Prefill 的开大（因此TTFT高）**无论**块大小如何。首先，选择显著低于GPU饱和点的块大小会延长 Prefill 任务的执行时间。例如，假设GPU在 Prefill 长度为512时饱和，将块大小设置为256会使所有超过512的 Prefill 的TTFT翻倍。其次，即使块大小被优化到几乎最大化GPU使用率，Chunked Prefill由于需要为每个后续块从GPU的HBM加载 KV Cache 到SRM，会显著增加 Prefill 任务的内存访问。这种情况在更长的 Prefill 中尤其严重，与未分块设置中的线性增加相比， KV Cache 加载操作呈二次增长，并且由于有限的 Decode token槽位而减少捎带机会。**至于TPOT**，正如我们之前揭示的，在批次中共置 Prefill 和 Decode 本质上会减慢所有这些 Decode 任务。

总之，Chunked Prefill在最大化整体吞吐量方面可能很有前景，但当应用不想在TTFT和TPOT之间进行权衡而是要同时遵守两者时，Prefill-Decode分离成为一个更好的选择。

# 今天的DistServe

我们正在与vLLM社区合作，将所提出的技术集成到vLLM生态系统中。

与我们的工作同时，Splitwise、TetriInfer和DéjàVu也采用了这种Prefill-Decode分离策略来将 Prefill 与 Decode 分离，以实现更好的LLM服务有效吞吐量。我们很高兴看到许多研究人员和公司采用Prefill-Decode分离来优化系统有效吞吐量，我们相信Prefill-Decode分离很快将成为LLM服务引擎的事实选择。

# 致谢

我们要感谢Vikranth Srivatsa、Lanxiang Hu、Will Lin为我们的博客提供了深刻的反馈。

# 引用

```shell
@article{zhong2024distserve,
 title={DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving},
 author={Zhong, Yinmin and Liu, Shengyu and Chen, Junda and Hu, Jianbo and Zhu, Yibo and Liu, Xuanzhe and Jin, Xin and Zhang, Hao},
 journal={arXiv preprint arXiv:2401.09670},
 year={2024}
}
```
