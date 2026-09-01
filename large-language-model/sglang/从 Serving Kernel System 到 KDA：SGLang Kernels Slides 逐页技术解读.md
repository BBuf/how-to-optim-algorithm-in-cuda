# 从 Serving Kernel System 到 KDA：SGLang Kernels Slides 逐页技术解读

这套 Slides 围绕 SGLang kernels 的工程设计与生产验证展开。它不是一份“把 SGLang 里的 CUDA kernel 罗列一遍”的目录，也不打算只展示几个漂亮的 microbenchmark 数字。我主要想回答三个问题：

1. 为什么一个推理服务系统，本质上也是一个 kernel system；
2. SGLang 如何把 kernel 做成可替换、可回退、可观测、可进入生产流量的系统组件；
3. 当 KDA（Kernel Design Agent）开始参与 kernel 设计后，如何保存 provenance、约束 dispatch，并用端到端证据决定一个实现能否上线。

下面按最终版 16 页逐页展开。文中的性能数字来自对应 PR 的实验记录；它们只适用于页面标注的硬件、模型、shape、并发和软件版本，不应被外推成无条件的通用结论。

# 0x0. Slide 1：Serving is a Kernel System

封面的核心句是 **SERVING IS A KERNEL SYSTEM**。

这里的 kernel system 不是说 serving 只剩下 CUDA kernel，而是说一次请求能否快速、稳定地完成，最终取决于一组 kernel 及其外围控制面的共同作用：计算 kernel、内存移动、KV Cache 布局、通信 collective、metadata 构造、dispatch、CUDA Graph 和 fallback。单个 kernel 再快，如果选错了调用时机、污染了 cache、破坏了 graph capture，或者只覆盖 benchmark 中的一个 shape，都不等于服务变快。

副标题 “From Hand-Tuned CUDA to Agent-Native Kernel Design” 给出了整场分享的时间线：前半段讲 SGLang 今天如何组织手写和 JIT kernel，后半段讲 KDA 生成的实现如何进入同一个生产体系。这里的 agent-native 不是让 agent 绕过工程约束直接写进热路径，而是让 agent 生成的候选实现从一开始就接受注册、验证、性能复现和 promotion contract 的约束。

# 0x1. Slide 2：可移植 kernel 只是起点

这一页讨论多芯片 kernel 如何进入 SGLang 的生产路径。

一套常见的组织方式是先提供跨平台参考实现，再为特定 GPU 增加架构专用 fast path。上层 operator contract 保持一致，底层实现根据芯片特征专门化。SGLang 的 backend registry、capability gate 和 fallback 解决的正是这类问题。

但线上服务比单题跑分多三层约束：

- **Graph**：实现能否在 CUDA Graph capture/replay 中保持正确，是否会在 replay 时分配内存或改变地址；
- **Fallback**：不支持的 dtype、layout、shape 和 architecture 是否能安静地回到参考实现；
- **E2E**：kernel 的局部收益能否穿过调度、同步、内存系统和其他算子，最终反映到 TTFT、TPOT 或 throughput。

这页最后的结论是：真正需要优化的不是一个脱离上下文的函数，而是 **operator contract × architecture × workload**。SGLang 通过真实模型路径、参考实现和端到端验证，把架构专用实现放进可控的服务环境。

# 0x2. Slide 3：一次请求会穿过六类 kernel

![Slide 3：一次请求跨越六类 kernel](https://files.mdnice.com/user/59/759e7784-6739-480a-af3e-dd80c1054054.png)

这页回答“为什么标题里要用 system”。一条典型生成请求会连续经过至少六组算子：

- Attention 与 KV Cache；
- GEMM 与量化；
- MoE routing、permute、grouped GEMM；
- 通信 collective；
- sampling 与 speculative decoding；
- state-space model 或 diffusion 等模型专用路径。

页面中的句子 “The end-to-end serving path is a composition of kernels, metadata, memory movement, and collectives” 特意使用 **end-to-end serving path**，而不是笼统地说 latency path。因为这里关心的不只是延迟，也包括吞吐、并发下的资源竞争和服务稳定性。

TTFT、TPOT 和 throughput 都是组合结果。例如一个 GEMM kernel 的 persistent scale tensor 可能让它自己的访存更快，却挤掉 attention/SSM state 在 L2 中的位置；microbenchmark 变快，模型吞吐反而下降。第 10 页的 Qwen3.8-27B 实验正是这个反例。

# 0x3. Slide 4：SGLang 当前暴露了多大的 kernel surface

![Slide 4：SGLang kernel source audit](https://files.mdnice.com/user/59/b64a18d1-b93b-450a-aafd-f79dae338c19.png)

这一页是对 2026 年 9 月 1 日 SGLang main（`8a191554`）的源码审计，而不是宣传口径。

页面给出三个数：

- `ops/` 下 21 个顶层 domain；
- 507 个 Python 文件；
- attention registry 中 24 个名称，其中一个是兼容旧接口的 alias。

最需要避免的误读是把 507 说成“507 个独立 kernel”。它只是 Python source footprint，里面包括 operator wrapper、backend adapter、metadata、测试辅助、兼容层和注册代码。因此页脚明确写着：这是一个 **kernel control surface**，不是 507 个独立 kernel 的 claim。

右侧按服务语义重新分组：Attention & KV、GEMM & Quant、MoE、Communication、Serving Primitives、Multimodal & Diffusion。这个分类比按 CUDA/C++/Triton/CuTe DSL 语言分类更有用，因为用户首先关心的是逻辑 operator；实现语言只是 backend 的一个属性。

# 0x4. Slide 5：一个逻辑 operator，多种实现

![Slide 5：统一 operator 与 backend registry](https://files.mdnice.com/user/59/c0063b9b-d8c4-43cc-a4fd-74aa6398cfdb.png)

这是整套 Slides 的架构中心。

上层模型只依赖稳定 namespace，例如：

```text
sglang.kernels.ops.<group>.<operator>
```

operator 进入 `BaseFusedOp + KernelRegistry` 后，再根据 priority 和 capability 选择 AOT、JIT、Triton、CuTe DSL、FlashInfer、DeepGEMM、AITER/NPU 或 KDA 等实现。注册层同时提供五个安全装置：

- Pure-Torch oracle：作为正确性参考和最终 fallback；
- Capability gates：约束 GPU 架构、dtype、layout、shape 等条件；
- Lazy import：未选中的 backend 不应带来启动依赖或导入失败；
- Trace + fallback：记录实际选择，并在不支持时回退；
- Forced backend：调试时可以强制整条 fused-op 路径使用参考 backend。

例如：

```bash
SGLANG_FORCE_FUSED_OP_BACKEND=torch
```

这能把“模型数值错了”拆成两类问题：如果强制 Torch 后恢复，问题大概率在某个 fused backend；如果仍然错误，就应继续查上层模型或数据路径。

页面把 **KDA** 加到 backend 枚举中，表达的是 provenance：这个实现来自 KDA 工作流。它不是一种编程语言或编译器。KDA 实现可以是 CuTe DSL（#36865）、CUDA JIT（#37385），也可以是 JIT + Triton（#37162）。至于它能在哪些 GPU、dtype 和 shape 上运行，仍由 capability 描述。

# 0x5. Slide 6：SGLang JIT kernel 的重点不只是“现场编译”

![Slide 6：SGLang JIT kernel 构建与缓存](https://files.mdnice.com/user/59/68c0d1f0-f0d7-4e50-afc8-8dd5e9d1bc71.png)

JIT 的价值是根据运行环境专门化实现，同时保持上层 API 不变。但生产 JIT 最难的部分往往不是编译器调用，而是 cache 一致性和多进程安全。

这一页把路径拆成：operator contract → build spec → 两级 cache → 安全发布 → TVM-FFI module。缓存 key 不只包含源文件和编译参数，还要覆盖 transitive dependency；否则头文件或被 include 的实现变了，旧二进制仍可能被错误复用。

并发启动时还需要 rank lock、staging 目录校验和 atomic rename。目标是任何进程都只能看到“完整且验证通过”的 artifact，不能读到另一个 rank 正在写的半成品。

页面给出的 H100 启动测量是：冷启动 5.26 秒、同一 clone 的 warm start 0.02 秒、第二个 clone 复用共享缓存 0.05 秒。这个数字说明 cache key 和发布协议不是边角工程，它们直接决定 JIT 能否进入生产环境。对应实现来自 PR #34274 和 `python/sglang/kernels/jit`。

# 0x6. Slide 7：Attention 是 backend system，不是一个 kernel

![Slide 7：Attention backend system](https://files.mdnice.com/user/59/20972512-967c-44d5-9082-6b78e8891904.png)

Attention 常被画成一个公式，但在 serving 里它至少包含四层工作：

1. 根据 batch、sequence 和 page 状态生成 metadata；
2. 解释 KV Cache 的页表、layout 和地址；
3. 处理 CUDA Graph capture/replay；
4. 在 prefill、decode、extend、dense、MLA、sparse 及不同平台 backend 之间 dispatch。

所以注册表里有 24 个名字并不意味着存在 24 个完全独立的数学算子，而是说明同一个 Attention contract 需要覆盖多种执行模式和 backend。稳定的逻辑接口让模型代码不用追随每次实现切换；新的 FlashInfer、FA3/FA4、FlashMLA、CuTe MLA 或平台实现可以在 capability 满足时被选中，不满足时回退。

这一页演讲时应强调：Attention 优化不仅是改 inner loop。metadata、KV layout 和 graph 行为常常决定一个 kernel 能不能在真实 decode 路径中发挥作用。

# 0x7. Slide 8：Custom AllReduce v2 为什么重新组织存储与算法

![Slide 8：Custom AllReduce v2](https://files.mdnice.com/user/59/f82511a6-7a99-41f0-b803-a4ff7f95c7de.png)

Custom AllReduce v2 的关键变化是把 **storage plane** 和 **compute plane** 解耦。

每个 rank 先准备 symmetric-memory slab；通信侧有 PushPlane 和 PullPlane，算法侧可以选择 one-shot push、one-shot pull 或 two-shot pull。调度不只看 message size，还需要看 GPU architecture、world size、eager/graph 模式和当前数据规模。超过合适的 crossover 后，NCCL 仍然是正确 fallback，而不是为了“自研”强行覆盖所有范围。

页面中的 8×B200 BF16 数据展示了这个选择空间：

| 数据量 | NCCL | AOT v1 | JIT Graph v2 |
| --- | ---: | ---: | ---: |
| 4 KB | 26.6 μs | 6.8 μs | 3.9 μs |
| 256 KB | 31.1 μs | 26.0 μs | 6.6 μs |
| 1 MB | 32.1 μs | 26.9 μs | 12.4 μs |
| 16 MB | 108.1 μs | 147.1 μs | 53.2 μs |

这些数据不应该被压缩成“v2 永远比 NCCL 快”。正确结论是：为 serving 中高频的消息规模和 graph 路径设计专门算法，并保留明确的 crossover/fallback，可以显著降低小中消息的同步成本。对应 PR 是 #31049。

# 0x8. Slide 9：单个 kernel 的收益如何累积成 diffusion 加速

![Slide 9：Qwen-Image 与 Z-Image-Turbo serving path](https://files.mdnice.com/user/59/0432ff61-ecf7-4483-a54b-09baf4f64735.png)

这一页用已经合并的 PR #36680 展示“组合收益”。页面列出三个局部优化：

- QKV pack：47.73 μs → 21.10 μs，2.26×；
- FA3 scheduler：159.40 μs → 147.06 μs，1.08×；
- 24 MiB collective：104.20 μs → 81.75 μs，1.27×。

这三项并不处在同一个抽象层：QKV pack 是数据重排，scheduler 影响 Attention 调度，collective 影响 TP 通信。它们共同作用后，Qwen-Image 在 TP2 + BCG、`quality=high` 下报告 +8.58% denoise speedup；Z-Image-Turbo TP2 报告 +5.05%。

Qwen-Image 的原始时间是 8.5406 秒降到 7.8657 秒。页面采用 `baseline / candidate - 1` 的 speedup 定义得到 8.58%；如果用 `(baseline - candidate) / baseline` 描述 latency reduction，则是 7.90%。写博客或讲数字时要说明口径，避免同一组时间产生两个百分比却看起来互相矛盾。

另外，`quality=lossless` 仍保留参考路径。这是很重要的产品设计：优化路径可以前进，但高质量或未覆盖场景不需要跟着冒险。

# 0x9. Slide 10：Qwen3.5 的收益，以及 Qwen3.8-27B 暴露的问题

![Slide 10：Qwen3.5 E2E 收益与 Qwen3.8-27B cache trap](https://files.mdnice.com/user/59/e12fe486-eb3d-4f95-b796-7b203402a3fe.png)

这一页经过重新设计，主角是 **Qwen3.5-4B 和 Qwen3.5-9B 的全链路收益**，右侧才是 Qwen3.8-27B 暴露的 dispatch 风险。

在页面标注的 RTX PRO 6000 / SM120、FlashInfer 0.6.18 环境下，PR #36865 当前 head `05a433d4d6` 的实验结果为：

| 模型 | 并发 | Throughput | TPOT | E2E latency |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-4B | 1 | +8.73% | -8.31% | -8.03% |
| Qwen3.5-4B | 2 | +6.84% | -6.91% | -6.51% |
| Qwen3.5-4B | 4 | +7.12% | -7.19% | -6.72% |
| Qwen3.5-4B | 8 | +6.52% | -6.63% | -6.10% |
| Qwen3.5-9B | 1 | +3.18% | -3.19% | -3.08% |
| Qwen3.5-9B | 2 | +2.78% | -2.86% | -2.74% |
| Qwen3.5-9B | 4 | +2.82% | -2.91% | -2.77% |
| Qwen3.5-9B | 8 | +2.70% | -2.78% | -2.62% |

这些结果说明 NVFP4 GEMM 的收益没有被模型运行时、Attention 和调度开销完全吞掉，而且在并发 1/2/4/8 下保持同方向变化。页面用区间总结：4B throughput +6.52–8.73%，9B +2.70–3.18%。

右侧的 Qwen3.8-27B 不是另一个加速 claim，而是一条负面证据。早期 broad dispatch 让 5.6–11 MiB 的 scale tensor 持续驻留，希望改善 GEMM；在完整模型中，它会挤压 Attention/SSM state 的 L2 空间，导致 E2E output throughput -0.76%。把 dispatch 收窄到真实获益的 `M=9, K=17408, N=5120` 后，其他 shape 回退，端到端结果转为 +0.98%。

这里真正的结论不是“exact shape 永远更好”，而是：**microbenchmark-positive 只能成为候选证据，不能直接成为部署策略**。扩宽 dispatch 必须逐个 shape 证明不会引入 cache、graph 和模型级回归；目前这条 PR 的生产资格仍是 model-specific、exact-shape、opt-in。

# 0xa. Slide 11：Humanize 管理优化循环，而不是生成 kernel

![Slide 11：Humanize 的 evidence-gated loop](https://files.mdnice.com/user/59/29f10ad1-de2c-49ef-b715-54bce15c721f.png)

Humanize 在这套架构里的角色是治理长时间运行的开发循环。页面中的流程是：

```text
task contract
  → implement
  → verify / profile
  → independent review
  → promotion decision
```

它提供三种普通 agent session 容易缺失的能力：

- durable state：跨多轮保留目标、实验和未解决问题；
- independent review：实现者不能用自己的解释替代审查；
- explicit termination：只有 acceptance criteria 达成，循环才应该停止。

因此要区分三者：Humanize 管理开发循环，KDA 搜索 kernel 设计空间，SGLang 决定实现能否进入生产 dispatch。把 Humanize 叫作 kernel generator，或者把一次 agent 生成代码等同于自动上线，都会误解这页。

# 0xb. Slide 12：KDA 的证据不是一个峰值数字，而是一组 portfolio

![Slide 12：KDA kernel evidence portfolio](https://files.mdnice.com/user/59/4f50b742-3119-40a3-9ff9-c115f1cb4c45.png)

这一页集中展示 KDA 已经积累的四类证据。

第一类是 Qwen3.8 QSA（#36845）：15/15 次真实 replay、150k stress launches、相对正确 Triton 实现 2.07× kernel geomean、真实 serving throughput +4.0–4.45%，GSM8K 保持 49/50。它说明 kernel speedup、长期状态稳定性、服务收益和模型准确性需要同时出现。

第二类是四个已经合并的 diffusion kernel（#27392、#29281、#29361、#29708），kernel 层收益分布在 1.279×–5.84×，并各自带有 denoise 或 E2E 测量。它们不会因为后来加了 `KernelBackend.KDA` 就改变实现语言；新 backend 只是把来源统一标出来。

第三类是 #36865 的 SM120 NVFP4 GEMM：kernel geomean 1.319×，并给出 16 个精确 production rows；真正重要的是第 10 页的 4B/9B E2E 收益，以及 Qwen3.8 broad dispatch -0.76%、exact dispatch +0.98% 这组正反证据。

第四类是 #37162 的 GB300 FLUX.2 FP8 路径：pixel-exact 的 E2E 1.0331×，denoise -3.246%，显存减少 438 MB，kernel 数量减少 19.9%，FP8 quant kernel 时间减少 83.4%，token-cat kernel 取得 1.77–2.60×。这类结果说明 KDA 不只可以追一个算子的峰值，也可以通过 fusion 减少 launch 和中间张量。

整页要守住一句话：KDA 的候选只有在 dispatch、correctness、accuracy 和 E2E 都成立时，才形成可信的工程结论。

# 0xc. Slide 13：为什么需要大写的 KDA backend

![Slide 13：KernelBackend.KDA](https://files.mdnice.com/user/59/de21d03a-350b-4567-9aab-f3af146bc2bd.png)

`KernelBackend.KDA` 解决的是 provenance 不可见的问题。

如果 KDA 生成的 CuTe DSL kernel 被登记为 `CUTE_DSL`，生成的 Triton kernel 被登记为 `TRITON`，运行时能知道“它用什么执行”，却不知道“它来自哪个设计和验证工作流”。大写 KDA backend 把这些实现放进统一 inventory，便于 forced selection、回退告警、测试覆盖和后续审计。

建议的 dispatch contract 包括：

- backend priority；
- capability 条件；
- 显式调用行为；
- forced backend 不满足条件时的 warning/fallback；
- registry inventory 和可查询的 provenance。

#36865 负责把新的 SM120 GEMM 接入 KDA backend；#37385 在此基础上注册已经合并的 diffusion kernel 和 FLUX.2 路径。需要准确描述 PR 状态：截至这套 Slides 的 source baseline，#37385 仍是 open，它注册的是此前已合并实现，不应说成“#37385 已经合并了这些 kernel”。

另外，KDA 只代表来源，不替代 capability。一个 KDA kernel 仍然可能只支持 SM120、某个 dtype、某种 layout 或一组精确 shape。

# 0xd. Slide 14：代码目录与 provenance 的边界

![Slide 14：KDA provenance boundary](https://files.mdnice.com/user/59/49046d2f-e211-4793-9cbf-29dcd720dc48.png)

这一页回应一个实际的代码组织问题：agent 生成的 kernel 是否都应该放进单独的 `kda_kernels` 目录？

我的划分是：

- 仍然以生成包、候选集或独立 artifact 形态存在的代码，放在 `python/sglang/kernels/kda_kernels/<generated-package>/`，保存机器生成过程和版本边界；
- 已经成熟、属于某个稳定业务 domain 的实现，进入 `ops/<group>/`，但注册时保留 `backend=KDA`。

这样既不会让模型热路径长期依赖一个杂乱的实验目录，也不会在代码成熟后抹掉它的生成历史。

provenance 至少要记录：task/revision、workflow 或 candidate hash、硬件和 shape、correctness/performance/E2E 证据、fallback 与 CUDA Graph 行为。promotion 则依次检查 registry、correctness、profile、CUDA Graph、模型 E2E、CI 和 review。

页面底部的总结是：**preserve machine history, serve qualified operators**。我们保存 agent 迭代的痕迹，但线上只服务已经通过资格审查的 operator。

# 0xe. Slide 15：从候选 kernel 到上游实现

这一页把前面的架构压缩成四个可执行步骤：

1. **TRACE**：先从真实服务中找到 shape 和频率，不凭直觉猜热点；
2. **CONTRACT**：定义 dtype、layout、数值误差、fallback 和 graph 约束；
3. **SPECIALIZE**：结合 compiler、memory hierarchy 和芯片原语做专门化；
4. **PROVE**：按 correctness → profile → graph → model-level impact 建证据链。

SGLang 为候选实现提供稳定 API、backend registry、Torch reference、JIT cache、trace 和 fallback。microbenchmark 领先只能说明这个实现值得继续验证。它还需要覆盖真实模型路径，并通过 CI、review 和端到端回归，才能进入生产 dispatch。

这套流程可以压缩成一句话：先在局部建立性能证据，再证明收益能安全地进入上游。

# 0xf. Slide 16：下一代 kernel system 是 human-agent system

最后一页把全场浓缩成四个节点：

```text
stable contract
  → specialized implementation
  → durable evidence
  → safe promotion
```

人的作用不是逐行包办所有 CUDA，agent 的作用也不是绕开 reviewer 自动写入 production。人定义目标、边界和风险判断；agent 扩大设计搜索和实验吞吐；SGLang 的 registry、fallback、测试与 E2E benchmark 把候选变成可管理的系统组件。

这也是标题从 “hand-tuned CUDA” 走向 “agent-native kernel design” 的真正含义：变化的不只是代码由谁写，而是 kernel 从产生、验证、审查到上线的整个生命周期。

# 0x10. 这套 Slides 的叙事结构

回头看 16 页，它们可以分成四段：

- Slide 1–4：建立 serving kernel system 的问题空间，并用源码审计限定讨论范围；
- Slide 5–8：解释 stable operator、backend registry、JIT、Attention 和 collective 的系统抽象；
- Slide 9–10：用 diffusion、Qwen3.5 和 Qwen3.8 的正反数据说明 E2E qualification；
- Slide 11–16：把 Humanize、KDA、provenance、promotion 和上游落地流程连接起来。

其中最重要的两条边界也贯穿全篇：

1. backend 描述实现来源和选择入口，capability 描述它在哪些条件下安全；
2. microbenchmark 决定一个实现是否值得继续验证，端到端 serving 才决定它是否值得进入生产 dispatch。

# 0x11. 参考链接

- SGLang 源码：https://github.com/sgl-project/sglang
- SGLang kernels RFC：https://github.com/sgl-project/sglang/issues/29630
- JIT kernel infrastructure：https://github.com/sgl-project/sglang/pull/34274
- Custom AllReduce v2：https://github.com/sgl-project/sglang/pull/31049
- Diffusion serving-path optimization：https://github.com/sgl-project/sglang/pull/36680
- KDA QSA kernel：https://github.com/sgl-project/sglang/pull/36845
- KDA NVFP4 GEMM 与 Qwen3.5/Qwen3.8 E2E 数据：https://github.com/sgl-project/sglang/pull/36865
- KDA FLUX.2 FP8 fusion：https://github.com/sgl-project/sglang/pull/37162
- KDA backend 与 diffusion kernel 注册：https://github.com/sgl-project/sglang/pull/37385
- Kernel Design Agents：https://github.com/mit-han-lab/kernel-design-agents
- Humanize：https://github.com/PolyArch/humanize
