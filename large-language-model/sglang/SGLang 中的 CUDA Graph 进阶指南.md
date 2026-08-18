# SGLang 中的 CUDA Graph 进阶指南

作者：SGLang Team

## TL;DR

CUDA Graph 的目标是消除 kernel launch 开销，但要在真实的推理引擎中尽可能接近这一理想收益，就需要在不牺牲兼容性、启动时间和显存的前提下，将尽可能多的工作负载纳入 CUDA Graph。

在 SGLang 中，我们围绕统一的 Runner/Backend 接口重构了 CUDA Graph 支持，让不同的捕获策略可以复用于不同执行路径。**Breakable CUDA Graph（BCG）是由 SGLang 原创的推理服务技术：它最早由 SGLang 提出、命名、实现并开源**。首个实现于 2026 年 2 月 21 日通过 #19102（https://github.com/sgl-project/sglang/pull/19102）公开，Prefill 扩展随后于 2026 年 4 月 24 日通过 #22218（https://github.com/sgl-project/sglang/pull/22218）合入。SGLang 社区还率先在 FA4 和 FlashInfer attention backend 上实现了 Prefill 的 Full CUDA Graph。本文也会深入介绍 CUDA Graph 的显存管理，包括不同 shape 和 graph segment 之间的显存复用。这部分正在成为 SGLang 整体显存管理中越来越重要的一环。

目前，Breakable CUDA Graph 已经是 SGLang Prefill 路径的默认方案。它只用了约四分之一的代码量，就实现了与基于 `torch.compile` 的 piecewise backend 相同的分段执行效果（521 行对 1771 行）；由于不需要编译，构建 Prefill graph 的速度快了 3.8～5.2 倍，同时也能更自然地兼容复杂功能。Prefill 的 Full CUDA Graph 更进一步：即使面对动态的 Prefill workload，也可以通过 request padding 捕获完整 forward。只测 Prefill 阶段时，BCG 相比 Eager 执行快 1.70 倍，Full CUDA Graph 则达到 1.93 倍。

## 背景

一次推理并不是单个 kernel，而是由许多 GPU 操作组成的序列。在现代 LLM serving engine 中，CPU 反复发射这些操作会带来明显开销，尤其是在对延迟敏感的 workload 中。CUDA Graph 会先记录一次 GPU workload，再以更低的 launch 开销反复 replay，从而减少这部分成本。

但要在现代推理引擎中有效使用 CUDA Graph 并不容易。Graph 的设计需要适配不同的执行阶段，兼容复杂 kernel 和依赖运行时信息的行为，同时还要控制 graph capture 带来的时间和显存开销。随着推理软件栈越来越复杂，正确集成 CUDA Graph 也变得越来越重要。

本文将介绍 SGLang 如何构建 CUDA Graph 支持，以及我们完成的几项改动：

- SGLang 中的 CUDA Graph：Runner/Backend 解耦与灵活组合
- Breakable CUDA Graph：无需编译器的 Eager 断点
- Prefill 的 Full CUDA Graph
- CUDA Graph 的显存占用

## SGLang 中的 CUDA Graph：Runner/Backend 解耦与灵活组合

在这次重构之前，SGLang 的 CUDA Graph 支持是围绕各条执行路径分别发展起来的。Decode、Prefill 和 speculative decoding 都有各自的 CUDA Graph Runner，其中 capture shape、静态 buffer、replay 和 graph 配置等逻辑高度重叠。随着执行模式和捕获策略不断增加，这些重复逻辑不仅让基础设施难以复用，也让 CUDA Graph 相关的 server argument 变得越来越模糊。

重构 PR #23906（https://github.com/sgl-project/sglang/pull/23906）将这些职责拆成两层。**Runner** 管理 capture 和 replay 所需的执行路径相关状态，包括 capture shape、静态输入 buffer、attention metadata，以及将实时 batch padding 到 capture shape。**Backend** 决定如何捕获这次执行：捕获为一张完整的 graph、捕获为一系列可中断的 segment，或由编译器生成多个 graph piece。

由于 Runner 只依赖统一的 Backend 接口，每条执行路径都能独立选择自己的捕获策略。Prefill 和 Decode 分别拥有独立的 Runner，speculative decoding 则会增加更多 Runner：EAGLE Draft、Draft Extend 和 Frozen-KV MTP Draft 都会基于 Decode Runner 构建各自的 Runner；Target Verify 直接复用 Decode Runner，但每个 request 会捕获多个 token。

![](https://files.mdnice.com/user/59/a075d051-528f-4bdc-a307-c03ec7615011.png)

*Runner 负责为每条执行路径准备 capture 和 replay，Backend 则决定如何将 forward 变成可 replay 的 graph：捕获成一张完整 graph、在 capture 过程中分段，或者先 trace 再切分。*

### Full CUDA Graph

Full backend 会针对每个选定的 shape 捕获一张 `torch.cuda.CUDAGraph`。它没有 Eager region，在三种 backend 中，replay 时需要的 launch 次数最少。这种方案很适合 Decode：每个 request 只贡献一个 token，因此主要的 shape 变量是 batch size，可以通过捕获一组 batch size bucket 来覆盖。Prefill 的变化维度更多，实现难度也更高，后文会单独讨论。

### Breakable CUDA Graph

由 SGLang 首创的 Breakable CUDA Graph（BCG）会捕获能够安全进入 graph 的区域，同时允许部分操作在不同 graph segment 之间以 Eager 模式执行。不兼容的操作可以用 `@eager_on_graph` 标记；捕获过程会在进入被标记函数之前停止，函数执行结束后再恢复，从而得到一系列被 Eager region 分隔的 CUDA Graph segment。

与基于编译器的 piecewise capture 不同，这些断点是在 capture 过程中直接插入的，不需要先 trace 完整模型再识别断点。下一节将介绍它的具体机制，以及 SGLang 为什么改用这套设计。

### TC Piecewise CUDA Graph

第三种 Backend 通过编译器实现相似的分段效果。`torch.compile` 使用 `fullgraph=True` trace forward，在预先注册的切分点拆分得到的 FX graph，然后分别编译和捕获每个 piece。它是 SGLang 最早实现 partial CUDA Graph capture 的方案，目前仍用于尚未验证 Breakable capture 的平台。

## Breakable CUDA Graph：无需编译器的 Eager 断点

传统 CUDA Graph 要求被捕获的区域完全兼容 graph。但在真实场景中，现代推理 workload 往往包含无法直接捕获的操作，Prefill attention 就是一个常见例子：某些 attention backend 依赖运行时 metadata 和 host 侧的准备工作。因此，只要存在一个不兼容操作，就可能导致 forward 中大得多的一段区域都无法使用 CUDA Graph。

为提高捕获的灵活性，SGLang 首创并开源了 **Breakable CUDA Graph（BCG）**。它不是对已有 serving 方案的重新命名：SGLang 首先提出并实现了这套面向 LLM 推理引擎的 runtime capture-break-resume 机制，让开发者可以在 capture 过程中插入显式 Eager 断点，而不必依赖 `torch.compile` trace 完整 forward。BCG 允许选定的操作以 Eager 模式执行，同时捕获它们前后兼容 graph 的区域；从整体上看，forward 会变成由显式 Eager 断点连接起来的一系列 CUDA Graph segment。

### 原创性与公开时间线

- **2026 年 2 月 21 日：SGLang 公开首个 BCG 实现。** PR #19102（https://github.com/sgl-project/sglang/pull/19102）的初始提交已经包含 `BreakableCUDAGraph`、Eager break decorator，以及在运行时结束并恢复 CUDA Graph capture 的核心机制。该 PR 于 4 月 11 日合入 SGLang。
- **2026 年 4 月：SGLang 将 BCG 扩展到 Prefill。** PR #22218（https://github.com/sgl-project/sglang/pull/22218）明确基于 #19102，将 BCG 构建为不依赖 `torch.compile` 的 Prefill Breakable Piecewise Backend，并于 4 月 24 日合入。BCG 随后成为 SGLang 默认的 Prefill CUDA Graph 方案。

这里所说的“首创”，特指 SGLang 在开源 LLM 推理引擎中首先提出、命名并实现这套 runtime breakable capture/replay serving 机制，而不是泛指 CUDA Graph 分段或 Eager fallback 这些更宽泛的概念。公开 PR 和提交时间共同构成了清晰、可追溯的技术来源。

### 设计与机制

当 replay 沿着固定的 GPU 操作序列执行，并且不需要 host 参与时，CUDA Graph 的效果最好。但真实的推理 forward 中存在许多难以纳入这种模式的操作：attention backend 可能需要根据实时 sequence length 制订执行计划，collective 可能涉及运行时协调，serving feature 也可能动态更新状态。

如果每遇到一个这类操作就放弃 CUDA Graph，forward 中的大部分区域都会失去捕获机会。BCG 允许开发者直接使用 `@eager_on_graph` 标记不兼容区域。Capture 执行到被标记函数时，当前 graph segment 会结束；函数以 Eager 模式运行；随后再开启新的 segment 继续 capture。

Replay 时，已经记录的 graph segment 和 Eager function 会按原来的顺序执行。在 capture 阶段，被标记的函数会在两个 graph segment 之间运行一次，其返回 tensor 会被保留为持久化的边界 buffer，确保 device address 固定不变；后一个 segment 也是基于这个地址完成捕获的。每次 replay 时，Eager function 都会正常执行并返回一个新的 tensor，BCG 再将它复制到保留的 buffer 中，使后一个 segment 能从 capture 时绑定的地址读到更新后的值。BCG 不会检查或 trace Eager region 内部的操作，只要求这些操作能够正确执行。

从功能上看，BCG 和此前基于 `torch.compile` 的 Piecewise Backend 都会生成相同类型的可 replay 结构：由 Eager region 分隔的多个 CUDA Graph segment。关键区别在于构建方式。TC Piecewise 会先让编译器理解完整 forward，再切分得到的 graph；BCG 则在 capture 发生时直接放置切分点。

### 优势

**启动更快。** 对于基于编译器的 Piecewise Graph，准备阶段的主要开销来自编译，而不是 capture：`torch.compile` 占 Prefill graph 准备时间的 78%～86%，并且会随模型复杂度增加，在 235B MoE 上达到 90 秒，在 GLM-5.2 上达到 158 秒。BCG 完全移除了编译阶段，只需一次 capture 就能得到分段执行结构。

![](https://files.mdnice.com/user/59/9c072c97-5a78-4041-810e-4b23e25236ab.png)

*构建 Prefill CUDA Graph 的时间。每种配置捕获 42 个 shape，TP4，运行在 4×GB300 上。*

编译开销也会影响日常开发。当时在我们的 CI 环境中，编译过程经常需要在不同测试之间重复执行，明显拖慢 CUDA Graph 测试。更好的缓存可以缓解这个问题，但将编译器移出 capture 路径，也就消除了开发循环中的这一额外复杂性来源。

**兼容范围更广。** SGLang 大量使用自定义 CUDA、Triton 和 JIT 编译 kernel，它们并不是原生 PyTorch operator。为了让 `torch.compile` 识别这些 kernel，我们通常需要通过 `torch.library` 封装它们，并提供用于 trace 的 fake implementation。这让整个 kernel stack 中到处出现仅为适配编译器而存在的辅助代码。

更重要的是，编译器还会限制 **graph boundary 可以放在哪里**。跨越已注册 operator 边界的输入和输出必须能被编译器表示。当自然的切分边界涉及更特殊的运行时状态或返回类型时，我们有时不得不寻找其他切分点，或者扩大 Eager region，只为暴露一个编译器能够处理的接口。随着 serving stack 不断发展，编译器边界逐渐开始影响原本与编译无关的代码结构。

BCG 解除了 Eager 断点处的这项限制：graph 系统不需要理解被标记函数的实现，也不需要 trace 其内部，因此 graph boundary 可以遵循 serving logic，而不是受编译器的 trace 和类型要求支配。CUDA Graph 需要与 DP Attention、MoE All-to-All Backend、LoRA、PD Disaggregation、Hierarchical Cache、Deterministic Inference 等快速发展的功能共存，过去要让 CUDA Graph 正常工作，越来越像是在维护一个 `torch.compile` 集成项目。新增 kernel 往往意味着新增 custom operator registration 和 fake implementation，新功能也可能迫使我们移动 graph boundary，只为满足编译器要求。使用 BCG 后，不兼容区域可以继续采用普通的 Eager 执行，大幅减少了这类编译器专用的工程成本。

**天生便于调试。** 捕获后的 CUDA Graph 会作为一个不透明整体 replay，普通 Python 代码不会在内部执行，因此很难使用 print、assert 或逐步检查。BCG 会自然保留 Eager region，让普通 Python 代码在每次 replay 时仍能执行。

SGLang 还通过 `--debug-cuda-graph`（PR #19102：https://github.com/sgl-project/sglang/pull/19102）扩展了这一思路。这个选项实际上会把整个 forward 包在一个 Eager 断点中。模型仍然通过 CUDA Graph Runner、静态 buffer、replay 路径和 metadata 准备流程，但具体执行采用 Eager 模式。这样就形成了一个很有用的调试边界：如果问题依然存在，原因很可能在模型或 Runner 路径中；如果问题消失，则应优先怀疑 capture 本身。

### BCG 在 Diffusion 中的应用

SGLang 的 Diffusion stack 也已经采用 BCG，相关实现见 PR #27436（https://github.com/sgl-project/sglang/pull/27436）。Diffusion 会在 denoising 过程中反复执行同一套 DiT forward。当这些 forward 包含大量受 launch 开销限制的小 kernel 时，CUDA Graph 尤其有效。

- **捕获真实的 serving shape。** 分辨率、视频帧数、prompt conditioning 长度、CFG 模式和所选 Transformer 都可能影响 capture signature。我们会对实际 serving 的 shape 做 warmup；遇到未捕获过的 signature 时，则回退到 Eager 执行。
- **在动态操作前后设置断点。** Dynamic attention、依赖运行时信息的 metadata preparation 等操作继续使用 Eager 模式；BCG 捕获它们周围的稳定计算，不要求 `torch.compile` 理解完整的 DiT forward。
- **利用重复的 denoising 结构。** Diffusion 在各个 denoising step 中反复执行相同的 DiT 结构。BCG 只需捕获一次稳定区域，就能在整个 denoising loop 中持续 replay，动态区域则继续 Eager 执行。

当执行主要受 launch 开销限制时，这种方法尤其有效。例如，warmup 之后，单张 B200 上 512×512 的 Qwen-Image 端到端延迟从 6.48 秒降至 2.45 秒，Z-Image 则从 1.231 秒降至 0.662 秒。

![](https://files.mdnice.com/user/59/865dd97e-b63d-4151-8d37-9e89aaac31b0.png)

*Warmup 后的端到端延迟。每组柱状图使用相同的模型 workload 和 seed。*

更普遍的结论是：BCG 消除的是 launch 开销，它不会减少模型 FLOPs，也不会降低 compute-bound kernel 的计算成本。当暴露出来的 launch gap 在总执行时间中占比较高时，BCG 的优势最大。

## Prefill 的 Full CUDA Graph

Full CUDA Graph 很适合 Decode，因为每个 request 只贡献一个 token，主要变化的维度是 batch size。Prefill 更难处理，因为一个 batch 会同时在两个维度上变化：总 token 数，以及这些 token 所属的 request 数量；但被捕获的 graph 要求这两个维度都保持固定。再加上部分 attention backend 依赖运行时 metadata，Full CUDA Graph 一度很难应用于 Prefill，这也是我们在 Prefill 路径采用 Breakable CUDA Graph 的主要原因之一。

最近，我们找到了让 Prefill 执行足够静态的方法，从而实现 Full CUDA Graph，相关工作见 PR #27988（https://github.com/sgl-project/sglang/pull/27988）。其中包括重新设计 request slot 和 attention metadata 的表示方式，让支持的 attention backend 不必再放到 graph 外执行。

### 让 Prefill 静态化

SGLang 通过 token bucket 固定 token 维度。实时 batch 会 padding 到最近的已捕获 token 数量，这与 Decode 将 batch size padding 到某个已捕获 bucket 的做法类似。

Request 维度则单独处理。每张捕获的 graph 都会预留固定数量的 request slot。真实 request 占用前几个 slot；没有使用的 slot 会被重写为长度为 0 的 sentinel：sequence length 和 extend length 都设为 0，offset 则放到真实 token 之后。如果 batch 中的 request 数量超过 graph 预留的 slot 数，就回退到 Eager 执行。

![](https://files.mdnice.com/user/59/2c7a450f-8ebf-4707-808a-97040d9b1ed6.png)

*Replay 时，token 会 padding 到捕获的 bucket，未使用的 request slot 则填入长度为 0 的 sentinel。*

由于捕获的 graph 仍会读取完整的 request table，sentinel metadata 必须在每次 replay 时重写。Replay 之前，attention metadata 同样需要在 graph 外根据 padding 后的 batch 重建。因此，目前只有支持这种 metadata 准备方式的 attention backend 才能使用 Prefill Full CUDA Graph，包括 FlashAttention 和 FlashInfer。

### Padding 的代价是什么？

两种 padding 的成本差异很大。

Padding token 会产生真实计算。它们会成为捕获 batch 中的实际行，并作为同一批 GEMM 的一部分经过 dense projection。SGLang 会单独传递真实 token 数，使 MoE routing、attention 和 linear attention kernel 能够跳过 padding 区域的大部分计算，但 dense computation 仍然需要处理这些额外的行。

空 request slot 的成本要低得多。在 FlashAttention 的 variable-length scheduler 中，workload 根据每条 sequence 的真实长度决定，并不会为每个 request slot 分配固定的计算量。因此，长度为 0 的 request 基本不会产生 attention 计算，只会增加 metadata 和少量调度开销。

这种不对称性很重要：token padding 是昂贵的维度，request slot padding 则相对便宜。

Prefill Full CUDA Graph 目前仍是实验性功能，必须显式开启。Engine 会提示 `full` 仍处于实验阶段，并建议生产 workload 使用 `breakable` 或 `tc_piecewise`。当前主要支持 FlashAttention（FA4）和 FlashInfer Backend，因为只有它们能按照捕获路径的要求构建 Extend Mode Metadata。扩大 backend 支持范围，并继续调整 bucket 和 slot 的选择，仍是后续工作。

### Prefill Benchmark

在三种 Prefill capture 方式和 Eager baseline 都具备后，剩下的问题就是它们在 replay 时各自要付出多少成本。我们单独测量了 Prefill：固定输入长度、只生成一个输出 token、每次处理一个 request，并在所有对比项中关闭 Decode Graph。测试使用 gpt-oss-120b（TP4，4×GB300），四条路径都能正常运行。结果显示，Full CUDA Graph 相比 Eager 快 1.93 倍，BCG 快 1.70 倍，TC Piecewise 快 1.45 倍。因此，BCG 不只构建速度更快，replay 性能也比基于编译器的 Backend 高 17%。

差距来自每次 forward 的执行方式：BCG 直接 replay 已经记录好的 segment，而 TC Piecewise 每次都要重新调用 compiled callable，在执行自身捕获的 piece 之前，先支付 TorchDynamo 的 guard check 和 dispatch 开销。在 GLM-5.2 上，只有 BCG 能完成 capture：TC Piecewise 无法 trace forward，Full CUDA Graph 也不支持它的 sparse attention。BCG 在该模型上相比 Eager 快 1.60 倍。在 32 倍 prompt length 变化范围内，每条曲线都基本保持水平，这正是 launch 开销主导而非计算主导的典型特征。

![](https://files.mdnice.com/user/59/e28f3d84-5f85-4de4-a345-e590093808eb.png)

*gpt-oss-120b 上只测 Prefill 的延迟，四种 Backend 均可运行。*

## CUDA Graph 的显存占用

显存方面有两个不同的挑战：一是避免分段捕获让常驻显存成倍增加；二是让 capture 覆盖足够大的 shape，使常驻的 graph 显存真正替代最坏情况下的 Eager activation 峰值。

### 分段捕获内部的显存复用

分段 Backend 很容易让 graph 显存成倍增长：每个被捕获的 shape 都包含多个 graph segment，而每个 segment 的中间结果都必须在 replay 时保持有效。BCG 通过三种复用方式避免了这种增长。

- **不同 segment 共享一个显存池。** 同一个 capture shape 的所有 segment 都使用同一个 CUDA Graph pool，让中间存储可以复用，而不是由每个 segment 分别长期占用。
- **在 Eager 断点处使用弱引用。** 如果传入断点的 tensor storage 已由 graph pool 管理，就使用弱引用持有该 tensor，避免不必要的 Python reference 延长 tensor 生命周期。Tensor 弱引用技术来自 vLLM PR #9724（https://github.com/vllm-project/vllm/pull/9724）；该 PR 引入这一技术，是为了让捕获的 graph 能够共享输出 buffer，而不是每张 graph 各自固定一块 buffer。
- **不同 capture size 共享一个输出 buffer。** 所有 capture size 共用一块按最大尺寸分配的输出 buffer，每个 shape 只切出自己需要的行，避免为每个 shape 单独分配输出 buffer。

有一个值不能这样处理：跨越 Eager 断点传递数据的 tensor。后一个 graph segment 是基于它的地址完成 capture 的，因此这块 buffer 必须一直存活，并在每次 replay 时原地更新。

有了这些复用机制，即使 capture table 很大，显存开销仍然可控：在 GLM-5.2 上，为 78 层 MoE 捕获 42 个 shape，只增加了 2.4 GB graph 显存。

### Capture 覆盖整个 Chunked Prefill Size

CUDA Graph 会改变 Prefill 的显存使用形态。Graph 显存是常驻的：在 capture 时分配，并在 server 的整个生命周期内保留。Eager activation 则是瞬时的：每次 Prefill 都会分配工作显存，支持的最大 Prefill shape 决定其峰值。

捕获某个 Prefill shape，会将很大一部分瞬时工作集转移到 graph 的常驻显存池中。但这只对真正使用 graph replay 的 shape 有效。如果 capture ladder 在最大 Prefill size 之前就停止，那么最大的 Prefill 仍会回退到 Eager 执行，保留原来的 activation 峰值；与此同时，server 还要承担更小 shape 对应的所有常驻 graph 显存。

因此，capture 上限比捕获了多少个 shape 更重要。`chunked_prefill_size` 限制了单次 Prefill forward 的最大规模，只要一直捕获到这个 size，就能消除最坏情况下的 Eager activation 峰值。

![](https://files.mdnice.com/user/59/b5708d63-c40c-4edf-b2df-510afc9bb324.png)

*在恰好使用 Chunked Prefill Size 完成一次 Prefill 后，相比无 Graph 常驻基线多出的 Prefill 显存。*

当 capture 上限低于 chunk size 时，显存占用反而会略高于无 graph baseline：系统增加了常驻 graph，但 activation 峰值完全没有变化。一旦 capture 上限达到 chunk size，最大的 Prefill 终于可以 replay graph，峰值随即大幅下降。在 gpt-oss-120b 上几乎降为 0（从 0.56 GB 降至 0.001 GB）；在 GLM-5.2 上则从 1.55 GB 降至 0.35 GB，因为它的 sparse-attention indexer 仍然需要在断点处以 Eager 模式运行。

让 capture 覆盖整个 Chunked Prefill Size 会带来两项收益：

- **总显存更低。** Activation 峰值不再由每个 request 临时承担，总显存最终低于无 graph baseline：gpt-oss-120b 低 0.51 GB，GLM-5.2 低 1.10 GB。与几百 GB 的整体显存占用相比，这个数字不算大，但它意味着 CUDA Graph 带来的是节省，而不是额外成本。
- **显存使用可预测。** 依赖 workload 的 activation 瞬时峰值，变成了在 capture 阶段确定的固定分配。Engine 可以提前计入这部分显存，不必再为只在大规模 Prefill 时出现的瞬时峰值预留空间。

## 致谢

这项工作由 SGLang 团队和 Meta 团队合作完成。

SGLang：Yuwei An*、Cheng Wan、Xiaoyu Zhang、Mick Qian、Baizhou Zhang、Yusheng Su、Ke Bao

Meta：Shiyang Chen*、Lianmin Zheng

同时感谢 NVIDIA、AMD、Thinking Machines Lab 和 Meta PyTorch 团队在整个过程中给予的帮助。

（* 表示同等贡献）
