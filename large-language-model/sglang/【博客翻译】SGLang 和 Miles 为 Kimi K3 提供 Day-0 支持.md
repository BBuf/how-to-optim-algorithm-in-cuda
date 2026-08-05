> 原文地址（https://www.lmsys.org/blog/2026-07-27-kimi-k3-day0-support）

# SGLang 和 Miles 为 Kimi K3 提供 Day-0 支持

SGLang 团队，2026 年 7 月 27 日

我们很高兴地宣布，SGLang 和 Miles 已为 **Kimi K3**（https://platform.kimi.ai/docs/guide/kimi-k3-quickstart）提供 Day-0 支持。K3 是首个参数规模达到 3 万亿级别的开源模型，其混合架构几乎在服务栈所有依赖既有假设的地方都打破了常规。通过与 Moonshot AI 和 NVIDIA 团队合作，SGLang 和 Miles 在发布首日就完整覆盖了 K3：SGLang 负责推理，Miles 负责 RL 训练。本文将介绍为此所做的工作。

**亮点**

- **全新的混合架构。** 2.8T 参数，69 层 KDA 线性注意力与 24 层 MLA 交错排列，并包含 LatentMoE 和注意力残差（Attention Residuals）；服务栈的大多数假设都会在某处失效。
- **两种状态的内存管理**，其中包括一种会原地覆盖自身的循环状态；在此基础上重建了前缀缓存、重叠调度和分页，并设计了统一内存池，消除了最后一个容量估算问题。
- **一系列逐级递进的 kernel 优化**，在进行推测解码之前，就将 batch size 1 时的速度提升到了约 113 tok/s。
- **DSpark 推测解码，以及我们为 K3 训练的草稿模型**（https://huggingface.co/RadixArk/Kimi-K3-DSpark）：batch size 1 时的 decode 速度达到约 423 tok/s。ReplaySSM 负责处理 KDA 状态，它重放原始输入，而不是在每一步保存状态快照，从而将草稿窗口的内存占用降低了约 32 倍。
- **按阶段拆分并行策略**：prefill 采用分块流水线并行，decode 采用张量并行；两者在 PD 分离下组合后，单 GPU 吞吐达到 **2,808** tok/s。
- **Miles 基于原生 MXFP4 checkpoint 进行 LoRA RL**，trainer 和 rollout 共置于同一组 GPU：在 12 小时的训练中，AIME-2024 成绩从 43.3% 提升到 76.7%。

启动命令以及针对不同工作负载的配置指南，请参阅 Kimi K3 cookbook（https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3）。

## 接入 Kimi K3

Kimi K3（https://platform.kimi.ai/docs/guide/kimi-k3-quickstart）是首个参数规模达到 3 万亿级别的开源模型：拥有 2.8T 参数、1M token 上下文窗口和原生视觉理解能力。它的前代 K2.5 在架构上与 SGLang 已经能够很好支持的模型非常接近，因此服务栈的大部分能力都可以直接应用。K3 则截然不同——它在多个彼此独立的方面同时打破了常规：

| | Kimi K2.5 | Kimi K3 |
|---|---|---|
| 规模 | 约 1T 参数 | 2.8T 参数 |
| 注意力 | 全部采用 MLA，共 61 层 | 混合架构：69 层 KDA 线性注意力 + 24 层 MLA，共 93 层 |
| 注意力残差 | 无 | 每个注意力输出都存入 bank，并按 block 聚合 |
| MoE | 384 个专家，top-8 | LatentMoE：896 个专家，top-16，在 3584 维潜在空间中运行 |
| 专家激活函数 | SwiGLU | SiTU |
| 视觉塔 | MoonViT | MoonViT3d，一个专为 K3 设计的新栈 |

表中的每一行都会带来一个服务问题。混合注意力栈使 1M 上下文的成本变得可承受，但这也意味着服务器需要同时保存两种状态：每个请求对应一个固定大小的 KDA 状态，同时还有 MLA 的逐 token KV——下文的内存管理章节讨论的正是这个问题。注意力残差会让一组注意力输出贯穿整个模型栈，这打破了 SGLang 标准层间管线的假设，并迫使 DP attention 等部分采用 K3 专用路径。LatentMoE 在降维后的潜在空间中，从 896 个专家里路由 16 个专家，并使用 SiTU 激活函数而不是 SwiGLU，因此现有 MoE kernel 都无法直接适用。此外，视觉路径——视觉塔、投影器、处理器以及 Kimi 的 XTML 媒体格式——也都是从零接入的。下文的所有内容都建立在这些接入工作的基础上。

## 混合 KDA 内存管理

注意力 KV 只会追加。一个 token 的 KV 一旦计算完成，就不会再发生变化。正因为如此，调度器才能让所有拥有相同前缀的请求共享同一份物理副本，把它保存在基数树中，并且无需额外考虑，就能让它跨迭代进行流水线处理。KDA 层的状态则恰恰相反：它是一个固定大小的循环缓冲区，**每处理一个 token 都会被原地覆盖**。因此，从前缀缓存、重叠调度器、推测解码到分页，调度器原本可以直接为 KV 提供的所有能力，都必须针对这个在读取过程中会发生变化的值重新构建。K3 在 24 层 MLA 之间交错放置了 69 层 KDA，因此这并不是一个边缘情况，而是模型的主体。下文介绍的就是这套重建工作。它有一个令人满意的特性：其中每一个部分都是由“原地覆盖”这一事实必然推导出来的，没有任何部分是后来外挂上去的。

### 无竞争的状态移动

一个请求的实时状态位于单个工作 slot 中，每次 forward 都会读取并原地覆盖它。缓存这个状态意味着要从 GPU 正在主动修改的内存中复制数据，这会产生两种竞争。恢复操作可能与正在写入该 slot 的 forward 发生冲突；而用于刷新缓存的快照，也可能在前一个快照正被移交给基数树时将其覆盖。这两种竞争都无需设备级同步，也无需在热路径上加锁，就可以消除。

首先，每一次状态复制都作为 kernel 在串行 forward stream 上执行。将缓存 checkpoint 恢复到工作 slot 的写时复制操作，以及在 track 边界捕获状态的快照操作，都会被加入到产生和使用该状态的两个 forward 之间。因此，同一 stream 上的执行顺序天然提供了 happens-before 关系。每个 prefill chunk 会创建一次快照，decode 期间每经过一个 track interval 也会创建一次快照。

其次，唯一会离开请求的移动操作并不传输任何字节。快照会落入一对采用乒乓方式工作的 slot 中，我们称之为额外缓冲区（extra buffer）：一个 slot 保存最新快照，另一个 slot 接收下一份快照；第二个 slot 只在需要它的边界处分配，并在之后立即释放。在每个 prefill chunk 之后、从 prefill 交接到 decode 时，以及请求结束时，缓存系统会把最新快照的 slot 索引移交给基数树。随后由一个新 slot 补入缓冲区，因此快照的写入目标和移交操作的读取来源永远不会是同一个物理 slot。

<p align="center">
  <img src="https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-1/463af039-42fd-49d0-ba2b-d11eaf3fa57f.png" width="98%" alt="KDA 状态的三种移动：写时复制、快照和移交，以及每种移动相对于串行 forward stream 的位置。">
</p>

<p align="center">
  <em><b>三种状态移动。</b>写时复制、快照和移交，以及每一种移动相对于串行 forward stream 的位置。</em>
</p>

借助这一设计，系统可以在任意时刻安全且高效地捕获状态，因此前缀缓存、重叠调度器、分页 KV 和推测解码等功能都可以顺畅地与循环状态协同工作。

### 循环状态的前缀缓存

循环状态无法在任意 token 位置进行切分，因为你不能让它反向运行到更早的位置，所以系统只在 chunk 边界创建 checkpoint，而且即使在这些边界上也只稀疏地创建。每条路径的数量上限和 LRU 机制让每条路径上只有少量 checkpoint 保持有效；一个 checkpoint 也可以独立于它所标注的 KV 被驱逐，使对应节点成为墓碑节点。分支点会得到特殊处理，这个思路来自 Marconi（https://arxiv.org/abs/2411.19379）。分叉点是所有未来分支都必然共享的前缀，因此，当一个请求在某条边的中间位置发生分叉时，它会从上方最近的 checkpoint 开始重放，并在与 chunk 对齐的分叉点放置一个新 checkpoint。下一个分支可以直接从这里恢复，无需重放。

<p align="center">
  <img src="https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-2/70431d55-d1cc-4902-9b4e-8f8745f987d9.png" width="98%" alt="叠加在基数树上的稀疏 KDA 状态 checkpoint，以及新请求偏离已缓存前缀时的分支点。">
</p>

<p align="center">
  <em><b>基数树上的 checkpoint。</b>稀疏 checkpoint 覆盖层和分支点。</em>
</p>

### 每个请求使用恒定数量的状态 slot

由于可复用的前缀 checkpoint 位于共享且可驱逐的基数树中，一个正在运行的请求只需预留少量临时 slot，最少可以只有四个：一个工作 slot、一个用于快照的额外缓冲区 slot，以及两个用于保留状态的余量 slot；后两者中，一个用于必须常驻的已提交前缀状态，另一个用于分叉分支所需的副本。大量缓存状态的成本由整棵树共同分摊，而不是计入每个请求，因此状态池无需随着每个请求保持热状态的历史长度一起增长。

### 统一内存：用一个池容纳两种状态

上述方案分别在各自的池中管理两种状态，而这些池本身是整个设计中最后一个仍需估算的地方。两种分配单元的大小相差三个数量级：每个请求对应一个较大的 KDA 状态块（TP=8 时约为 54 MB，覆盖全部 69 层 KDA），每个 token 则对应一个较小的 MLA KV 块（约为 27 KB，覆盖全部 24 层 MLA）。因此，目前它们位于两个独立的池中，并在启动时确定池的大小。这种容量配置是在对流量下注；一旦估计有误，服务器就会在其中一个池仍有富余空间时，耗尽另一个池的内存。

统一内存将两个池替换为一个：KDA 状态从一端向内填充，MLA KV 块从另一端向内填充，两者之间尚未使用的字节构成一整块空闲区域。

<p align="center">
  <img src="https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/da33f826-4a2e-458c-9c71-029006659099.png" width="98%" alt="统一内存：两个静态池与一个同时保存两种状态的内存池，以及释放过程。">
</p>

<p align="center">
  <em><b>统一内存。</b>上：目前两种状态分别使用独立的池，并在启动时确定大小，因此一个池可能仍处于空闲状态，而另一个池已经满了。中：使用统一内存后，两种状态从同一个池的相对两端向内分配，中间只有一块连续的空闲区域。下：释放位于中间的状态块会留下空洞，此时把末端的一个块移动到该位置，使空闲区域保持为完整的一块。</em>
</p>

释放同样简单。当请求完成、被中止，或者在压力下被回退时，它的内存块会被释放。如果这在中间留下了空洞，就把末端的一个块移动到该位置，使空闲区域保持为完整的一块。

这种布局既支持灵活的页面大小，也不会产生内存碎片：54 MB 的 KDA 状态块和 27 KB 的 MLA KV 块从同一片内存中分配，不必强制它们使用统一的页面大小；上文的移动操作以几乎可以忽略的状态移动成本，让两端始终紧密排列。因此，空闲空间始终是一块连续区域，两种状态都可以使用。由此，容量会跟随工作负载变化，而不是取决于某个启动参数：大量短请求会用状态块填满内存池，少量长上下文会用 KV 填满内存池，两种情况都不需要重新配置。

统一内存目前通过 `--enable-unified-memory` 参数以可选方式提供。后续文章将详细介绍它的实现。

## 使用 DSpark 进行推测解码

K3 随附 DSpark block 推测解码，并由我们为 K3 训练的草稿模型（https://huggingface.co/RadixArk/Kimi-K3-DSpark）驱动。这次集成有两个部分值得单独讲述：只在值得验证的地方使用验证预算，以及让 K3 的循环 KDA 状态能够在推测过程中正常工作。

### 只验证值得验证的内容

DSpark 每一步都会提出一个由草稿 token 组成的 block，目标模型再通过一次 forward 验证整个 block。在 batch size 1 时，额外验证位置基本没有成本：这一步受延迟限制，多带几个 token 几乎不会增加开销。但随着 batch 逐渐填满，情况就不再如此。此时，验证 token 会与其他所有请求争用同一个 step 的时间，而且其中大多数 token 的押注最终都会失败：在聊天工作负载中，平均接受长度约为 2.7，因此一次典型 step 所验证的八个位置中，会有五个被拒绝。验证全部位置意味着为这些随后会被服务器丢弃的 token 支付完整成本。

解决这个问题所需的组件其实已经存在于系统中。草稿模型带有一个经过训练的置信度头，它会逐位置预测每个 token 通过验证的概率。另一方面，通过对服务器进行一次性 profile，可以记录在不同负载水平下，多验证一个 token 实际需要付出多少成本。逐 step planner 将二者结合起来：对于每个请求，只有当验证 token 的期望价值足以覆盖边际成本时，才保留这些 token；窗口的其余部分则会在目标模型 forward 启动之前被裁掉。保留下来的部分仍以与之前完全相同的方式验证，因此输出依然无损。这里的取舍是：用略短的接受序列换取成本更低的 step。

<p align="center">
  <img src="https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-2/647a05a4-97e8-4057-9867-34c8587d3960.png" width="98%" alt="decode 吞吐随 batch size 的变化：在聊天面板和 few-shot 数学面板上，对比验证全部位置和按置信度调度裁剪。">
</p>

<p align="center">
  <em><b>负载下裁剪更有优势。</b>在聊天面板（平均接受长度约 2.7，左）和 few-shot 数学面板（平均接受长度约 5.0，右）上，对比验证全部位置与裁剪时的 decode 吞吐。batch size 不超过 8 时两者基本持平，随后差距会随着 batch size 增大而拉开：batch size 256 时分别提升 68% 和 24%；接受长度分别从 2.7 缓慢降至 2.2、从 5.0 缓慢降至 4.3，而吞吐则不断上升。</em>
</p>

实际测得的成本曲线是其中最有意思的部分。验证 token 的边际成本并不平滑，而是呈阶梯状：在平坦的平台区间里，额外一个 token 可以进入当前这批 kernel wave，几乎没有成本；而在陡峭的上升处，它会启动新的一批 kernel wave。planner 会读取这个成本表面。如果一个请求接下来的 token 位于成本较低的平台上，planner 就保留它们；如果它们会触发一次阶梯上升，planner 就在这里截断。在数据中，这表现为接受长度曲线出现小幅波动，而吞吐曲线仍保持平滑且单调：planner 利用的是真实的硬件平台区间，而不是噪声。

batch size 小于 8 时，需要缓解的压力很小，因此裁剪的收益持平或略为负面；已知的后续工作是在 planner 中为小 batch 添加提前退出机制。

### ReplaySSM：通过重放原始输入恢复 KDA 状态

推测解码会一次验证 γ+1 个草稿 token，并且可能只接受其中一个前缀。对 MLA 而言，这不需要额外代价，因为 KV 只会追加，被拒绝的草稿只需释放相应 slot。KDA 层的状态则会在每个 token 处覆盖自身，因此 baseline 通过暴力方式获得可逆性：每完成一个 draft step，就为完整的 K×V 状态创建一次快照。当 K=V=128 时，每个请求、每一层、每个 head 的单份快照为 64 KB，再乘以 γ+1 个 step。在 K3 的 69 层 KDA 和完整 batch 上，这一大小会超过与其竞争内存的持久状态池；而且由于这部分内存要为每个正在运行的请求预留，它还会限制并发度。

**保存输入，而不是状态。** ReplaySSM（https://tridao.me/blog/2026/replayssm/）不再保存快照。verify kernel 会读取已经提交的 checkpoint，但绝不写入它；在执行过程中，它还会保存每个 step 的原始输入 `Sᵢ = (vᵢ, kᵢ, gkᵢ, βᵢ)`，大小约为 1 KB，而一份快照需要 64 KB。一旦 sampler 确定接受长度，一个覆盖所有层和 head 的 fold kernel 就会从 checkpoint 开始，只重放被接受的前缀，并原地推进该状态。被拒绝的草稿永远不会被重放，因此回滚没有任何成本。草稿窗口的内存占用从 512 KB 降至 16 KB，缩减约 32 倍。

**精确，而不是近似。** fold 是 verify 循环递推的逐字复刻，使用相同的 tile 和相同的归约顺序；它会使用 verify kernel 自己保存的 gate 值，而不是重新计算这些值。后一点不可省略。早期版本曾在 torch 侧使用一个略有差异的公式重新计算 gate，结果每个输出看起来都正确，底层状态却在悄然漂移。现在，重建后的状态与循环 baseline 原本会提交的状态在 bit 级别完全一致。

这项优化在 batch size 较小和较大时都能带来收益。batch size 较小时，以原始输入代替快照可以减少 verify kernel 的内存访问，而且重放本身只使用一个融合 kernel，从而降低 kernel launch 开销，因此每个 step 的耗时已经有所改善。batch size 较大时，效果则显著得多。快照暂存区完全是推测产生的额外开销，并且会随 batch size 和 γ 一起增长，因此恰恰在 γ 大到值得使用时，对状态池造成的挤压最严重。归还这部分内存后，并发上限可以提升数倍。一旦 baseline 开始让请求排队，ReplaySSM 仍能继续接纳请求，差距也正是在此时拉开。

<p align="center">
  <img src="https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-2/a55404b6-4115-457d-a847-5c5996d024b5.png" width="98%" alt="KDA ReplaySSM：verify kernel 读取已提交的 checkpoint，并把每个 draft step 的原始输入保存在一个较小的逐 slot 缓冲区中；接受结果确定后，一个 fold kernel 只重放被接受的前缀，并原地推进状态。">
</p>

<p align="center">
  <em><b>KDA ReplaySSM。</b>Verify（每层一次融合 launch）读取 checkpoint h₀ 但不写入它，并把每个 draft step 的原始输入保存到逐 slot 缓冲区中，每层、每个 head 各有一份。sampler 确定接受长度后，fold kernel（所有层和 head 共用一次 launch）只重放被接受的前缀，并原地覆盖该 slot；它会逐字执行相同的循环递推，并使用 verify kernel 自己保存的 gate 值，因此重建后的状态与循环 baseline 在 bit 级别完全一致。</em>
</p>

## Kernel

在一个 2.8T 混合模型上，对单个序列执行一次 decode step 并不受计算量限制——问题在于 kernel launch 的数量和延迟：每个 token 都要经过 93 个注意力层（69 层 KDA + 24 层 MLA）以及 92 层 latent-MoE；刚完成接入时，一个 step 会触发数百个小 kernel。整个优化过程由 profile 驱动：融合一个部分，按照固定流程做 A/B 测试，用 GSM8K 做门禁，然后重复这一过程。

![batch size 1 时按优化类别划分的瀑布图](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-1/4eb4b81d-a2f1-46d0-b4fe-6e3cde44fc94.png)

**消除 launch 和复制（P1–P4，+19.9 tok/s）。** 重点是让 step 更小，而不是让 kernel 更快：将 MoE 前端合并为一次 GEMM，并合并 KDA 中成对的窄投影（P1）；由 profile 引导的全面排查消除了逐层 upcast、复制和多余的 launch（P2、P3）；路由则变成对 `[M, 896]` logits 执行的一趟式、常驻寄存器的基数选择（P4）。

**NVIDIA 计算 kernel（P5–P8，+10.3 tok/s）。** 在默认方案不适合 batch size 1 形状的地方，四个阶段分别换用了与 NVIDIA 共同开发或由 NVIDIA 提供的 kernel：融合 KDA decode kernel、位于路由旁路之后的 trtllm-gen W4A8 SiTU MoE cubin、基于 TMA 的注意力残差聚合，以及在 M 较小时替代 cuBLAS 的 CuTe-DSL TGV bf16 GEMM。每一项都经过了与其他优化相同的 A/B 测试和精度门禁。

**通信融合（P9、P12、P13，+27.6 tok/s）。** 这是收益最大的阶段：在 CustomAllReduceV2 的对称内存平面上，为 MNNVL 互连实现了一套融合 all-reduce——小消息采用一次性 multicast store，大消息采用 NVLS 交换机内归约，同时把残差加法和 RMSNorm 一并放入 collective 中（P9）。随后，MoE finalize 被移入 collective 的 staging pass（P12）；up-projection 则从复制式 GEMM 改为列并行 GEMM，并通过 multicast all-gather 完成（P13）。

**重叠与序言融合（P10、P11、P14、P15，+10.4 tok/s）。** 其余优化用于缩短剩余的关键链：采用跨步输入的 MXFP8 量化（P10）；将残差写回融合到上游 kernel 的尾部，并让独立分支在 side stream 上运行（P11）；让 KDA 的 GEMV 链与 qkvg GEMM 重叠执行（P14）；将 MLA decode 的序言融合为一个 kernel，并用 PDL 启动其后的注意力 kernel（P15）。

四个柱形背后的按时间排序的优化阶梯如下：

![batch size 1 时的 decode 吞吐优化阶梯](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/acdc19a8-c5f9-4117-a918-bc85b2961262.png)

**可以推广的经验。** All-reduce 是一个同步点，因此在那里节省一微秒，就会一比一地转化为 step 时间的缩短；而位于另一个 stream 的重叠空隙中的 kernel，转化比例大约只有十分之一。在编写 kernel 之前，先检查它是否位于 trace 的关键路径上，是这次优化过程中杠杆率最高的单个习惯。

¹ 一次 rebase 将 P5 的 baseline 从 64.2 调整为 63.8；该阶段相对于 rebase 后的 baseline 进行测量。
² P13 是合并窗口结束后重新校准得到的规范 baseline，并不能归因于单个 PR。
³ P1–P4 之前的区段还包含后来被替代的中间优化（concat all-reduce → P9；tiny/1-CTA GEMV → P8；radix router v1 → P4；Marlin top-k-sum → P6/P12；早期 attn-res add → P7）；这些临时收益体现在曲线中，但没有对应的具名节点。

## K3 的并行化

对于 K3 的混合架构，传统的并行策略无法胜任。张量并行无法切分 MLA 的 KV cache（只有一个 KV head，没有可供切分的维度），因此每个 rank 都要保存完整副本；它还会把每个 GEMM 切成八份，并在每层支付一次 collective 的成本。纯 DP attention 则会在每个 rank 上复制注意力权重，其中 KDA 约为 61 GB、MLA 约为 11 GB，而 KV cache 和 KDA 状态也需要使用这些内存。在这些成本下，prefill 和 decode 会以不同的方式遇到瓶颈，因此 K3 按阶段拆分解决方案：prefill 采用分块流水线并行，decode 采用上下文并行。

### Prefill：分块流水线并行

在 TP prefill 中，每层末尾都有一次 AllReduce，这是一个无法与计算重叠的屏障。流水线并行则按层切分模型：K3 的 93 层被划分为 8 个 stage，prompt 也被切分为多个 chunk，并以流式方式通过这些 stage：

<p align="center">
  <img src="https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/ba087a67-471c-431d-bcf1-8323aab43b7e.png" width="98%" alt="分块流水线并行 prefill：长 prompt 被切成多个 chunk，并以流式方式通过流水线 stage；P2P 交接箭头在下一个 chunk 的计算期间执行。下方为 TP 对照，其中每层末尾都有 AllReduce 屏障。">
</p>

<p align="center">
  <em><b>分块流水线并行 prefill。</b>各 stage 同时处理不同的 chunk。stage 之间的交接会在该 stage 计算下一个 chunk 时执行，因此在 K3 上有 91% 的交接成本被隐藏。在 TP 中（底部条带），每层末尾都有一次 AllReduce，所有 rank 都必须等待。</em>
</p>

这会从三个方面带来收益。剩余的唯一通信操作——向下一个 stage 交接——隐藏在下一个 chunk 的计算之后。每个 rank 都会运行完整的层，因此 GEMM 的宽度扩大了八倍，效率也更高。并且每个 stage 只需为自己约 12 层保存 KV 和 activation，因此可以轻松地对非常长的 prompt 执行 prefill。不过，流水线必须足够深：较浅的 PP4×TP2 无法覆盖其交接成本，同时仍需支付 TP2 的 AllReduce 成本，benchmark 表现并不优于 TEP8。

在 2×4 GB300 上测量 8K prefill，唯一变量为拓扑：

<p align="center">
  <img src="https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-2/710c965f-e263-46fe-be41-4c63c437f2db.png" width="98%" alt="左：单 GPU prefill 吞吐随并发度的变化，对比 TEP8 和 PP8xTP1。右：每 1k token 的成本分解，包括 TP8、PP4xTP2、TEP8 和 PP8xTP1 的计算与暴露通信成本。">
</p>

<p align="center">
  <em><b>深度 PP 在两个维度上都胜出。</b>左：越过 c1–c4 的交叉点后，PP8×TP1 最终达到约为 TEP8 上限 1.7 倍的吞吐，同时 TTFT 更低。右：在每个 rank 的 FLOPs 相同的情况下，对比每 1k 个 prefill token 的成本；PP4×TP2 和 TEP8 分别在计算与通信之间做出取舍，最终打平，而 PP8 在两方面的成本都最低。</em>
</p>

PP8 只在单个请求时落后，而 prefill worker 如果空闲到这种程度，本身就是配置不当。在 K3 的分离式服务中，prefill 节点运行 PP8。每个节点的 prefill 容量是 TEP8 节点的 1.45 到 1.72 倍，因此一个 prefill 节点可以持续为多个 decode 节点供给数据。decode 节点运行 TP 或 DCP，下一节将对此进行介绍。

### Decode：上下文并行

decode 阶段的瓶颈是被复制的 KV cache：在 TP 下，多接纳一个请求或多增加一千个 token 的上下文，都会在每个 rank 上消耗同样多的字节。Decode Context Parallelism（DCP，decode 上下文并行）不按 head 切分 MLA 的 KV，而是按 token 位置进行切分。当 `p mod N = r` 时，rank r 负责位置 p，因此每个 rank 都以交错方式保存每个请求上下文的 1/N，而且这种切分对于注意力 kernel 之上的部分是不可见的：

<p align="center">
  <img src="https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-1/2a3ba15b-960f-4bba-9791-793a2427c9c3.png" width="98%" alt="TP 在每个 rank 上复制相同的 token 位置；DCP 以轮询方式对它们进行条带化切分，因此同样数量的 GPU 可以保存 N 倍的逻辑上下文。">
</p>

<p align="center">
  <em><b>按位置切分。</b>4 个 rank 上的同一组 16 个 token 位置：TP 保存 64 份物理副本，DCP 则保存 16 份。释放出来的字节会转化为逻辑 KV 容量；在 K3 上使用 DCP8 时，逻辑 KV 容量约为原来的 7.9 倍。</em>
</p>

按位置切分会破坏 softmax，因为每个 rank 只能看到 1/N 的 key，而局部 softmax 无法直接相加。解决方案与 FlashAttention 自身采用的方法相同：每个 rank 返回自己的局部注意力输出，以及每个 head 对应的 log-sum-exp；每层通过一次 all-to-all 交换它们，使每个 rank 最终获得覆盖完整上下文的 1/N head。随后，在本地按 log-sum-exp 进行精确合并，而结果已经是 TP 下 output projection 所期望的 head 布局。每层只需一次 collective，这就是全部通信成本：

<p align="center">
  <img src="https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/f239adce-3d71-4381-aeef-c6e247a8581b.png" width="98%" alt="两个 DCP rank 上逐层执行 MLA decode 的数据流：复制 query projection，对各自负责的位置执行局部注意力并产生局部输出和 LSE，进行一次打包后的 all-to-all，再在本地合并 LSE。">
</p>

<p align="center">
  <em><b>每一层的 decode step。</b>每个 rank 在本地投影包含全部 head 的 query，只关注自己负责的位置，并通过一次打包后的 all-to-all 发送局部输出及其 log-sum-exp。本地合并后会得到标准的 TP head 布局；注意力之后的任何部分都无需改变。</em>
</p>

其他部分全部保持不变。DCP group 构建在 TP group 内部，因此 TP8 搭配 DCP8 时依然只使用 8 个 GPU，MoE 也继续采用其原有的并行方式。KDA 是一个能够解释这条规则的例外：它的状态是每个请求对应一个固定大小的矩阵，而不是每个 token 对应一个矩阵，因此不存在可以切分的位置轴，KDA 层仍按 head 进行 TP 切分。整个功能只需一个参数：`--dcp-size N`。

这项能力在 agent 流量上的收益最为明显，因为长达数十万 token 的会话会不断堆积在 cache 中。我们在 2×4 GB300 上重放真实的编码 agent 会话，两组方案都使用 host memory KV 层，唯一的区别是是否使用 DCP：

<p align="center">
  <img src="https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-1/25da0e73-b7af-4ddf-8c65-58e8f656dfd1.png" width="98%" alt="聚合 decode 吞吐随并发 agent 会话数量的变化：对比 TP8 + hicache 和 DCP8 + hicache。当活跃集合超出设备 KV 容量时，TP8 在 16 个会话处崩塌；DCP8 则继续提升，在 48 个会话时达到 541 tok/s。">
</p>

<p align="center">
  <em><b>DCP 消除了活跃集合上限。</b>host memory 层可以挽救重新 prefill 的流量，但并发会话达到 16 个时，活跃工作集会超出 TP8 的设备 KV 容量，吞吐随之崩塌。DCP8 将逻辑 KV 容量从 1.5M token 提升到 12.2M token，并让相同工作负载在 48 个会话时达到 541 tok/s。</em>
</p>

DCP 可以与服务栈的其他部分组合。DSpark verify step 本身就是一个 decode step，因此它会沿用相同的“复制 Q、执行一次 all-to-all”路径；在 PD 分离下，prefill 侧无需感知 DCP，每个 decode rank 只需在传输边界拉取自己负责的位置，因此 PP 或 TP prefill 都可以为 DCP decode 供给数据。剩余的上限来自 KDA：它的逐请求状态无法按位置切分，因此 DCP 消除 MLA 的瓶颈后，正在运行的请求数量上限会成为约束容量的因素；前文内存章节介绍的统一内存设计就是为了解决这个问题。

### 组合所有策略

各个组件组合后的效果最终要通过测量来确定。把 PD 分离加入测试，并改变 prefill 拓扑、decode 拓扑以及 prefill:decode 比例，可以得到如下服务前沿：

<p align="center">
  <img src="https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-1/3d68adb4-9456-4a2a-9da9-8edd6659c327.png" width="98%" alt="PD 分离下的 K3 服务前沿：单 GPU 总吞吐与单用户 decode 速度之间的关系。PP8 和 TP8 prefill 分别为 TP8、DCP8、EP 以及多实例 TP8 decode 方案供给数据；前沿从吞吐端的 PP8 到 TP8，一直延伸至交互端由 TP8 为三个 TP8 实例供给数据的方案。">
</p>

<p align="center">
  <em><b>服务前沿。</b>在吞吐端，一个 PP8 prefill worker 为一个 TP8 decode 节点供给数据，在 fp4 路线上达到单 GPU 2,808 tok/s；DCP 组合方案紧随其后，两个 PP8 prefill worker 为两个 DCP8 decode 节点供给数据，达到 2,633 tok/s。向右移动体现了 prefill:decode 旋钮的作用：让一个 prefill worker 为两个、三个、再到四个独立 decode 实例供给数据，会以聚合吞吐为代价换取单用户速度，最终超过每用户 116 tok/s。</em>
</p>

## RL：在原生 MXFP4 基座上进行 LoRA 训练

K3 的 Day-0 RL 支持采用 Miles 进行共置 LoRA 训练：基于 Miles Megatron 后端的 BF16 trainer，与原生 packed MXFP4 SGLang rollout engine 共享同一组 64 个 GB300。该后端覆盖 KDA、NoPE-MLA、注意力残差 bank 和 latent MoE，并支持 TP/SP/PP/CP/EP。

### LoRA 服务与权重同步

engine 会直接按 checkpoint 发布时的形式提供服务，从不改写 checkpoint。每个 step 只传输 BF16 LoRA adapter；engine 会在量化基座 GEMM 之上，将增量作为单独的 BF16 `B(Ax)` 项应用，因此在 4 bit 基座权重上，策略更新会以完整精度传递到推理侧。dense projection 使用 SGLang 的 Triton LoRA 后端；896 个路由专家使用 Marlin 路径上的融合 MoE-LoRA kernel；共享专家的增量则被折叠到融合 MoE 前端 GEMM 中。adapter 位于 GPU 内存池中，每次同步都会在原位置换掉它们，因此 rollout 侧没有常驻的 BF16 副本，不需要重新同步完整权重，整个循环中也没有重新量化步骤。

### 并行

**流水线并行。** K3 的注意力残差快照 bank 必须跨越 stage 边界，但 Megatron 的 point-to-point 只能携带一个 hidden-state tensor，因此 stage 边界会把 `[prefix_sum, bank]` 打包进这个 tensor，并在进入下一 stage 时解包。Megatron 本身无需修改。

**上下文并行。** 由于每个注意力残差操作都是逐 token 执行的，因此该 bank 可以自然完成切分。MLA 同样不需要 K3 专用代码：Megatron MLA 中唯一需要感知 CP 的工作是切分 rotary table，而 K3 没有 rotary embedding，因此其 projection 可以直接使用标准 TE attention core 并继承 CP。KDA 才是需要额外处理的部分，它通过 fla 的 CP context 来处理循环状态和卷积 halo。KDA 需要 rank 本地的连续 chunk，而 Megatron 保存的是 ring attention 所需的之字形顺序，因此只在 KDA 周围执行重新布局。

**专家并行。** adapter 在全部 896 个路由专家之间共享潜在空间一侧的 factor，而另一侧的 factor 则逐专家保存，这与 engine 的融合 MoE-LoRA 约定一致。共享 factor 会跨 EP 复制，但被标记为 expert-parallel，因此 DDP 只会在 expert-DP 范围内对它执行归约，而 EP sum 是此次发布唯一自行添加的梯度归约。

### 内存

在一张 277 GiB 的卡上，原生 MXFP4 rollout 的峰值接近 225 GiB/GPU，BF16 trainer 在初始化时的峰值接近 155 GiB/GPU，因此二者永远不会同时驻留：一方运行时另一方进入休眠，而能否实现共置，取决于每次交接后是否没有任何残留。

**基座从不移动。** engine 会在整个运行期间让基座权重常驻 GPU；trainer 工作时，只释放 KV cache 和 CUDA graph。由于基座从未被释放，自然也从不需要恢复；LoRA 路径也不会传输任何基座字节：基座同步被完全跳过。

**进程组保持运行。** trainer 的 NCCL communicator 缓冲区不属于 torch allocator 管理的内存，因此 offload 无法释放它们。最直接的做法是在休眠时销毁进程组，并在唤醒时重建。但这样做相当于用少量常驻 GiB 换取每轮重建和 EP 预热的成本，只有当 engine 需要拿回这些字节时才值得——而这里并不需要，因为更新路径中没有任何基座权重传输。于是，进程组会一直保持运行。

**adapter 传输的内存由每个 chunk 限定。** adapter 约有 2,800 个 tensor。每个 chunk 都以一个展平的 CUDA IPC bucket 发送，总共有 278 个 bucket；接收方确认收到，并且所有生产方 rank 都越过 engine-group barrier 后，该 bucket 就会立即被回收。这样可以限制每个 chunk 的临时 IPC 内存，而不是让内存占用在整个传输过程中不断累积，峰值时可节省约 48 GiB/GPU。

**host 副本只存在于有组件需要读取它们的地方。** 每个 adapter 安装到 GPU 内存池后，engine 就会释放它的 CPU 副本，使 scheduler RSS 从 76–88 GiB 降至约 17 GiB；CPU 副本已经释放的 adapter 无法重新安装，因此内存池会用报错拒绝驱逐，而不是悄悄地用过期 slot 提供服务。在 trainer 侧，DDP 缓冲区会按生命周期拆分：adapter 参数缓冲区保留在由 CPU 备份的区域中，因为 trainer 休眠时权重更新仍需读取它们；梯度缓冲区则可以重建，因此会进入无备份区域，并在休眠时被丢弃。

### 验证

训练正确性是 RL 支持栈中最重要的部分。我们在制定训练 recipe 之前就构建了这些检查。

**训练/rollout KL** 会在每次 rollout 时记录。它是对采样 token 上 KL(rollout ‖ train) 的 Schulman k3 估计，根据 engine 返回的 log-probability 和 trainer 重新计算的 log-probability 得到。它是一项诊断指标，不属于训练目标的一部分；训练目标中不包含 KL 惩罚。它的下限约为 2e-3，这是由使用 MXFP4 基座提供服务决定的；如果该值随 step 数增长，则表明二者正在发生偏离。

**金丝雀锁步探针。** 从第一次 rollout 开始固定一条轨迹；在相同的策略版本下，每一步都由使用当前 adapter 的 engine 和 trainer 分别对它评分。两条曲线同步变化，才能证明训练后的 adapter 确实应用到了推理中：传输 checksum 只能证明字节已经到达，不能证明有任何组件读取了它们。

**tensor 级 dump 对比。** 让相同 token 分别通过两个 build 或两种并行布局，dump 每个 forward activation 和每个参数梯度，再以测得的噪声下限为基准，对比相对 L2 和 cosine。这里不要求 bit 级精确，因为即使是在两组不同 GPU 上执行相同的算术运算，也已经会在 ulp 级别产生差异。流水线边界和上下文并行布局正是通过这种方法验证的：当 CP=2 和 CP=1 构建出完全相同的调用时，两者在 bit 级别完全一致；更换 forward kernel 后，cosine 中位数为 0.996。

**权重同步断言。** 包括 trainer 与 engine 之间逐 tensor 的 SHA256 manifest；如果一次 adapter 更新没有改变任何导出的 tensor，就会报错的 validator；检查版本 1 时每个 `B` factor 是否为零；以及 adapter 梯度和 optimizer step 检查。

### 训练结果

本文报告的实验是在 16 个节点 × 每节点 4 个 GB300 上运行 DAPO math：TP8 / PP8 / EP8 的 BF16 trainer 与原生 MXFP4 rollout engine 共置；response 长度为 4096 token；每次 rollout 采样 64 个样本；每次 rollout 执行一个 optimizer step；使用 rank-32 / α-64 LoRA，学习率为 1e-5；采用不含 KL 项的 GRPO；运行到 12 小时 wall-clock time 上限。在 60 个 step 中，AIME-2024 greedy eval 从 43.3% 提升到 76.7%，即答对题数从 30 道中的 13 道增加到 23 道，并且在截止时仍在上升；与此同时，train/rollout KL 在整个运行期间都稳定保持在约 2e-3 的 MXFP4 下限。

<p align="center">
  <img src="https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-2/71b6e5d8-1faf-4530-8b09-4ab45148cbf4.png" width="100%" alt="并列的三条 RL 训练曲线：rollout 原始 reward 呈上升趋势，AIME-2024 eval 准确率不断提高，train/rollout KL 则稳定保持在 MXFP4 量化下限。">
</p>

<p align="center">
  <em><b>12 小时 DAPO 实验。</b>左：每次 rollout 采样的 64 个 response 的平均 reward，在 response channel 上评分；每次 rollout 都会抽取不同的 prompt batch，因此趋势才是信号，逐 step 的数值则是噪声。中：AIME-2024 greedy 通过率，每 10 次 rollout 评估一次，评估采用与训练相同的 4096 token 上限，因此即使解答正确，只要超出该上限，也会被计为未通过。右：采样 token 上 KL(rollout ‖ train) 的 Schulman k3 估计；该值只用于报告，从不加入 loss；目标是稳定保持在量化下限，单调增长则是发生偏离的标志。</em>
</p>

## 致谢

这项工作由 RadixArk 的 SGLang 与 Miles 团队、Moonshot AI 团队共同完成，NVIDIA、AMD、Approaching AI、Baseten 和 Modal 也参与其中。

**AMD**：Wun-guo Huang、Xinyi Song、Hai Xiao、Soga Lin、Duyi Wang、Thomas Wang

**Approaching AI**：Huanming Shen、Xiaohao Zhang、Nan Li、Mingxing Zhang

感谢 DigitalOcean 为我们的测试提供 AMD 实例。

感谢 Google Cloud、DigitalOcean、Nebius、fal、RunPod、DeepInfra 和 GMI Cloud 使用 SGLang 为 Kimi K3 提供服务。
