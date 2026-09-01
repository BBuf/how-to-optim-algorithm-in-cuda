# 智源 FlagOS × SGLang 多芯片算子优化挑战赛：从可移植 Kernel 到生产级 Serving

作者：BBuf

智源 FlagOS × SGLang 多芯片算子优化挑战赛选取 SGLang 真实工作负载中的算子作为赛题。参赛者需要先实现可移植版本，也可以针对不同芯片加入专用 fast path。这个设置很接近推理框架的日常工作：上层 operator contract 不能跟着硬件变化，底层实现却必须理解架构、编译器和内存层次。

不过，赢下单个算子的 benchmark 只是开始。一个实现要进入 SGLang，还得面对 backend dispatch、CUDA Graph、fallback、数值正确性以及模型端到端性能。过去几个月，我在 SGLang 和 KDA（Kernel Design Agents）相关工作里反复遇到同一个问题：microbenchmark 里的胜利，到了完整 Serving 系统中不一定还成立。

这篇文章尝试把两件事接起来。一边是多芯片算子比赛，另一边是 SGLang 如何管理 kernel 的完整生命周期。文中的源码统计基于 2026 年 9 月 1 日的 SGLang main `8a191554`；性能数据只对应各 PR 标注的硬件、模型、shape、并发和软件版本。

## 从可移植实现到生产路径

![挑战赛算子如何进入 SGLang 生产路径](https://files.mdnice.com/user/59/144d753a-5414-4430-9a87-8e8ba40faca7.png)

比赛里的 portable kernel 给出了共同起点。架构专用实现负责在目标芯片上提速，SGLang 再判断它是否能进入真实模型路径。这三层不能混为一谈。

一个 production-ready operator 至少要回答下面几个问题：

- dtype、layout 和数值语义是否与参考实现一致；
- 不支持的 shape 或 GPU 架构是否会回退，而不是静默执行错误路径；
- CUDA Graph capture/replay 时是否分配新内存、改变地址或留下旧状态；
- 局部性能收益能否反映到 TTFT、TPOT、throughput 或 denoise latency。

因此，更准确的优化对象是：

```text
operator contract × architecture × workload
```

单独谈 kernel 代码，很容易漏掉后两项。单独看模型平均吞吐，又可能看不出是哪一个 shape、哪一次 dispatch 在起作用。比赛提供了一个具体算子和跨芯片目标，SGLang 则提供真实调用路径、参考实现与回退机制，把局部优化放回可控的系统环境。

## Serving 本身就是一个 kernel system

![一次请求会经过计算、内存、通信和调度组成的 kernel system](https://files.mdnice.com/user/59/18371f4f-ae78-4055-ba2d-61872186f087.png)

一次生成请求不会只经过 GEMM。Attention 要构造 metadata 并解释 KV Cache 页表；MoE 包含 routing、permute 和 grouped GEMM；多卡执行需要 collective；采样和 speculative decoding 也有各自的 kernel。模型换成 Mamba、Qwen-Image 或 FLUX 后，还会进入 SSM、multimodal 和 diffusion 专用路径。

所以我更愿意说 end-to-end serving path，而不是 latency path。Serving 关心的不只有延迟。吞吐、并发下的 cache 竞争、graph replay 和错误回退都由同一条执行链决定。

对 SGLang main 的一次源码审计能直观看到这个范围：`python/sglang/kernels/ops/` 下有 21 个顶层 domain、507 个 Python 文件，Attention registry 中有 24 个名称，其中一个是兼容旧接口的 alias。507 不能解读成“507 个独立 kernel”，因为里面还包括 wrapper、backend adapter、metadata、注册代码和兼容层。它描述的是 kernel control surface 的大小。

按服务语义看，这个 surface 大致分成 Attention & KV、GEMM & Quant、MoE、Communication、Serving Primitives、Multimodal & Diffusion。这样的分类比按 CUDA、Triton 或 CuTe DSL 分组更有用。用户调用的是逻辑 operator，编程语言只是某个 backend 的实现属性。

## 稳定的 operator，允许多种 backend

![SGLang 的统一 operator 与 backend registry](https://files.mdnice.com/user/59/7aaf302c-e464-486b-bc02-ebaf7165973b.png)

SGLang kernels 目前的核心抽象，是让模型代码依赖稳定 namespace：

```text
sglang.kernels.ops.<group>.<operator>
```

operator 进入 `BaseFusedOp + KernelRegistry` 后，再根据 priority 和 capability 选择 AOT、JIT、Triton、CuTe DSL、FlashInfer、DeepGEMM、AITER/NPU 或 KDA 实现。注册层同时保存 Pure-Torch oracle、capability gate、lazy import、trace、fallback 和 forced backend。

例如，排查模型数值错误时可以强制 fused-op 路径使用 Torch：

```bash
SGLANG_FORCE_FUSED_OP_BACKEND=torch
```

如果错误消失，问题大概率位于某个 fused backend；如果仍然存在，就应该继续检查模型或数据路径。这比逐个注释 kernel 快得多，也说明 registry 不只是“选最快实现”的性能组件，它还是调试和回归控制面。

KDA 放进 backend 枚举时，表达的是实现来源，而不是新的编程语言。KDA 生成的实现可以使用 CuTe DSL（#36865）、CUDA JIT（#37385），也可以是 JIT + Triton（#37162）。至于某个实现支持哪些 GPU、dtype、layout 和 shape，仍然由 capability 单独描述。

## JIT 的难点是缓存一致性

SGLang 的 `jit_kernel` 适合尺寸较小、需要跟随运行环境专门化的 CUDA kernel。上层 Python API 不变，build spec 负责收集源码、编译参数和依赖，然后生成 TVM-FFI module。真正棘手的地方不是调用一次 NVCC，而是让多进程服务始终拿到正确的二进制。

缓存 key 必须覆盖 transitive dependency。假设依赖链是：

```text
kernel.cu → common.cuh → sm120_utils.cuh
```

`sm120_utils.cuh` 虽然没有被入口源码直接 include，也必须进入 cache key。否则它修改后，JIT 仍可能复用旧二进制。这类问题很难在单元测试中稳定复现，却会在共享缓存和多 clone 部署里留下隐患。

并发启动还需要 rank lock、staging 目录校验和 atomic rename。任何 rank 都只能看到完整且通过校验的 artifact，不能读到另一个进程正在写的半成品。PR #34274 给出的 H100 启动测量是：冷启动 5.26 秒，同一 clone warm start 0.02 秒，第二个 clone 复用共享缓存 0.05 秒。这个差距决定了 JIT 能不能成为正常部署路径，而不只是开发阶段的便利功能。

## Attention 与通信为什么都需要系统抽象

Attention 常被画成一个公式，Serving 中却至少有四层工作：构造 batch/sequence/page metadata，解释 KV Cache layout，处理 CUDA Graph，再在 prefill、decode、extend、dense、MLA、sparse 和不同平台 backend 之间 dispatch。优化 inner loop 只是其中一部分。metadata 或 KV layout 不匹配，再快的实现也进不了真实 decode 路径。

通信也有同样的问题。Custom AllReduce v2 把 storage plane 与 compute plane 分开：每个 rank 先准备 symmetric-memory slab，通信侧提供 PushPlane/PullPlane，算法侧选择 one-shot push、one-shot pull 或 two-shot pull。最终调度要结合 architecture、world size、消息大小以及 eager/graph 模式；超过适用范围后继续回退 NCCL。

![Custom AllReduce v2 的存储、算法与 dispatch](https://files.mdnice.com/user/59/ebb6aba8-2ef5-4f6b-950c-154b0b6aa69d.png)

PR #31049 在 8×B200 BF16 环境记录了下面的数据：

| 数据量 | NCCL | AOT v1 | JIT Graph v2 |
| --- | ---: | ---: | ---: |
| 4 KB | 26.6 μs | 6.8 μs | 3.9 μs |
| 256 KB | 31.1 μs | 26.0 μs | 6.6 μs |
| 1 MB | 32.1 μs | 26.9 μs | 12.4 μs |
| 16 MB | 108.1 μs | 147.1 μs | 53.2 μs |

正确的结论不是“自定义实现永远比 NCCL 快”。这些数据说明，Serving 中高频的小中消息和 graph 路径值得专门优化，同时必须保留清楚的 crossover 与 fallback。

## 局部优化怎样穿过完整模型

PR #36680 是一个很好的组合案例。QKV pack 从 47.73 μs 降到 21.10 μs，FA3 scheduler 从 159.40 μs 降到 147.06 μs，24 MiB collective 从 104.20 μs 降到 81.75 μs。三项优化分别作用于数据重排、Attention 调度和 TP 通信，并不属于同一种 kernel。

它们组合后，Qwen-Image 在 TP2 + BCG、`quality=high` 下报告 8.58% denoise speedup，Z-Image-Turbo TP2 报告 5.05%。Qwen-Image 的时间从 8.5406 秒降到 7.8657 秒。这里使用 `baseline / candidate - 1` 得到 8.58%；如果按 `(baseline - candidate) / baseline` 计算 latency reduction，则是 7.90%。报告性能时需要写清口径，否则同一组数据会出现两个看似冲突的百分比。

`quality=lossless` 仍保留参考路径。这也是生产优化与单题跑分的区别：fast path 可以继续扩展，但没有覆盖的质量档位不需要跟着承担风险。

## KDA 生成候选，SGLang 决定能否晋级

当 Agent 开始参与 kernel 设计后，搜索速度提高了，验证责任却没有消失。我把 Humanize、KDA 和 SGLang 分成三个角色：Humanize 管理长周期开发循环，KDA 搜索 kernel 设计空间，SGLang 负责 production promotion。

Humanize 保存 task contract、实验记录、profile 证据和未解决风险，并引入独立 review。循环只有在 acceptance criteria 达成时才结束；否则需要明确记录下一步改动，而不是让 Agent 自己宣布完成。

KDA 的价值也不应该用一个峰值 speedup 概括。目前几组公开工作覆盖了不同层次的证据：

![KDA kernel 在真实 Serving gate 下的证据组合](https://files.mdnice.com/user/59/5a45ce10-cd78-4d9d-b10f-28b8fa72778a.png)

- Qwen3.8 QSA（#36845）完成 15/15 次真实 replay 和 150k stress launches，kernel geomean 相对正确 Triton 实现为 2.07×，真实 Serving throughput 提升 4.0%–4.45%，GSM8K 保持 49/50；
- 四个已经合并的 diffusion kernel（#27392、#29281、#29361、#29708）记录了 1.279×–5.84× 的 kernel 收益，并各自给出 denoise 或 E2E 测量；
- SM120 NVFP4 GEMM（#36865）在 16 个精确 production rows 上得到 1.319× kernel geomean，后面还会看到 4B、9B 与 27B 三种不同的模型结果；
- GB300 FLUX.2 FP8 路径（#37162）保持 pixel-exact，E2E 为 1.0331×，denoise latency 降低 3.246%，显存减少 438 MB，kernel 数量减少 19.9%。

这些实验的共同点是同时报告执行路径、正确性和端到端结果。只留下“最快的一行”，反而会丢掉最有价值的信息。

## 写 kernel 时常见的 reward hacking

这里的 reward hacking 通常不是 Agent 故意作弊。更实际的情况是 benchmark 合同不完整，Agent 很认真地优化了错误目标。我在近期的 KDA-Pilot 和 SGLang 工作中遇到过五类问题。

![Kernel reward hacking 与 production promotion gate](https://files.mdnice.com/user/59/49076515-7751-4d50-ae39-4ead606d2580.png)

第一类是优化了 host path，而不是 device kernel。KDA-Pilot #22 早期用 overlay 替换了 SGLang 中带 `@register_custom_op` 的公开入口，约 1.22× 的表面收益混入了绕过 production wrapper 的 host-side 差异。#24 和 #25 把 baseline 与 candidate 放回相同 public op、相同 wrapper 和 in-tree arbiter 后，可归因到 device kernel 的结果约为 1.12×。

第二类是数值捷径。candidate 单边打开 `--use_fast_math`、放松 tolerance、不检查 NaN/Inf，或没有确认输出 tensor 被完整覆写，都可能让错误实现通过。后续 benchmark 固定了两边的编译参数和 ABI，加入 poison output；LTX2 任务又使用 production bitwise contract（#152、#157、#158）。

第三类是挑 shape。看到结果后删掉变慢的 workload，或只报告 geomean，会藏住 production bucket 的退化。#40 的 baseline-vs-baseline A/A geomean 是 0.9992；#41 和 #43 冻结 production rows，并逐行记录 dispatch 与 fallback；#89 则接受 pure-speed no-go，只保留 correctness/safety 修复。正常的优化流程必须允许某些候选被拒绝。

第四类是 stale state 或 wrong path。CUDA Graph replay、复用 workspace 和 device counter 可能留下旧状态，silent fallback 也可能让 benchmark 实际测到 reference backend。KDA-Pilot #194 对 stateful kernel 使用 chained final-state checks；SGLang #36845 又执行 150,000 次连续 launch，确认 counter 每次都回到零。

最后一类是 isolated kernel 与完整模型不一致。microbenchmark 变快后，cache pressure、metadata 或下游 kernel 可能变慢。这个问题在 #36865 中出现过一次，而且非常有代表性。

我现在更愿意把 promotion reward 写成下面这组约束：

```text
semantic fidelity × executed path × frozen workload × serving E2E
```

任何一项为零，候选都不能晋级。挑战赛提交同样应该优化题目定义的 operator contract，而不是 benchmark 偶然留下的漏洞。

## Qwen3.5 的收益和 Qwen3.8-27B 的反例

![Qwen3.5 Serving 加速与 Qwen3.8-27B cache 反例](https://files.mdnice.com/user/59/b4c1ba84-d572-4a88-8c35-c22aaa56c700.png)

PR #36865 当前 head `05a433d4d6` 在 RTX PRO 6000 / SM120、FlashInfer 0.6.18 环境下记录了如下结果：

| 模型 | 并发 | Throughput | TPOT | E2E latency |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-4B | 1 | +8.73% | -8.31% | -8.03% |
| Qwen3.5-4B | 2 | +6.84% | -6.85% | -6.41% |
| Qwen3.5-4B | 4 | +7.12% | -7.02% | -6.65% |
| Qwen3.5-4B | 8 | +6.52% | -6.63% | -6.10% |
| Qwen3.5-9B | 1 | +3.18% | -3.19% | -3.08% |
| Qwen3.5-9B | 2 | +2.78% | -2.82% | -2.71% |
| Qwen3.5-9B | 4 | +2.82% | -2.81% | -2.74% |
| Qwen3.5-9B | 8 | +2.70% | -2.78% | -2.62% |

Qwen3.5-4B 的 throughput 在四个并发点提升 6.52%–8.73%，9B 提升 2.70%–3.18%；TPOT 和 E2E latency 也在每个点改善。production dispatch 只覆盖已验证的四组 `(K,N)` 与 `M={1,2,4,8}`，没有重新打开 broad gate。

Qwen3.8-27B 给出了反方向的证据。早期 broad dispatch 让每层 5.6–11 MiB 的 scale tensor 持续驻留。isolated GEMM 因此变快，但 scale tensor 挤掉了 Attention/SSM state 的 L2 空间，完整模型的 E2E output throughput 下降 0.76%。改成 one-pass streaming，并只放行 `(M,K,N)=(9,17408,5120)` 后，其他 shape 回退，端到端结果转为 +0.98%。

这组正反结果解释了为什么 dispatch gate 不能只看 microbenchmark。4B/9B 的收益可以进入精确 production rows；27B 暴露的 cache 问题则帮助我们收紧 promotion 条件。失败数据在这里不是附注，它决定了 backend 应该怎样上线。

## 为什么还需要大写的 KDA backend

![KDA 作为一等 provenance backend](https://files.mdnice.com/user/59/fc09b792-f002-4b9f-8e9b-a86b5e0ceedd.png)

如果 KDA 生成的 CuTe DSL kernel 只登记为 `CUTE_DSL`，Triton kernel 只登记为 `TRITON`，运行时能知道它们怎样执行，却不知道实现来自哪个设计和验证流程。`KernelBackend.KDA` 补上了这层 provenance，使 forced selection、回退告警、测试覆盖和 inventory 审计都有统一入口。

大写 KDA 不会取代底层编译器，也不会绕过 capability。一个 KDA kernel 仍然可能只支持 SM120、某个 dtype、特定 layout 或少量精确 shape。backend 表示来源，target 与 capability 决定它如何运行、能在哪里运行。

PR #36865 负责把新的 SM120 GEMM 接入 KDA backend；PR #37385 在此基础上注册此前已经合并的 diffusion kernel 和 FLUX.2 路径。PR 状态需要说清楚：在本文的 source baseline 上，#37385 仍然是 open。它负责注册已有实现，不能写成“#37385 合并了这些 kernel”。

代码目录也要保留这条边界。仍处于生成包、候选集或独立 artifact 阶段的代码，可以放进 `python/sglang/kernels/kda_kernels/<generated-package>/`，保存 task revision、candidate hash 和搜索历史。成熟后属于稳定业务 domain 的实现进入 `ops/<group>/`，但注册时继续保留 `backend=KDA`。

provenance 至少要记录 task/revision、workflow 或 candidate hash、目标硬件与 shape、correctness/performance/E2E 证据，以及 fallback 和 CUDA Graph 行为。机器生成历史值得保留，生产路径只接收通过资格审查的 operator。

## 从比赛提交走到 SGLang upstream

![从 FlagOS × SGLang 竞赛提交到 SGLang upstream](https://files.mdnice.com/user/59/e6fd79ef-ed01-412a-819e-69ee14e48991.png)

把前面的讨论压缩到实际开发流程，可以分成四步：

1. 从真实服务 trace 中确认 shape 分布和调用频率；
2. 固定 dtype、layout、数值语义、fallback 与 graph contract；
3. 结合编译器、内存层次和芯片原语做 specialization；
4. 依次完成 correctness、profile、CUDA Graph、模型 E2E、CI 和 review。

SGLang 为参赛实现提供稳定 operator API、backend registry、Torch reference、JIT cache、trace 和 fallback。microbenchmark 领先说明候选值得继续测试，最终能否进入 production dispatch，要看它是否沿着同一条调用路径通过模型级验证。

我理解的 agent-native kernel design 也落在这套流程里。人负责定义任务合同、风险边界和 promotion 标准；Agent 扩大设计搜索与实验吞吐；SGLang 把候选收进可回退、可测试、可审计的系统。代码由谁生成并不是最难的问题。难的是怎样让一份局部很快的实现，带着足够的证据走进上游。

对这次智源 FlagOS × SGLang 多芯片算子优化挑战赛来说，比较完整的路径应该是：

```text
competition submission
  → qualified SGLang backend
  → production serving
```

比赛负责把真实算子和多芯片目标开放出来，SGLang 社区负责把好的实现变成长期可维护的 backend。跑分结束后，工作才完成一半。

## 参考链接

- 智源 FlagOS × SGLang 多芯片算子优化挑战赛：https://flagos.io/race-detail-season2?id=782kzq4m&lang=en
- SGLang 源码：https://github.com/sgl-project/sglang
- SGLang kernels RFC：https://github.com/sgl-project/sglang/issues/29630
- JIT kernel infrastructure：https://github.com/sgl-project/sglang/pull/34274
- Custom AllReduce v2：https://github.com/sgl-project/sglang/pull/31049
- Diffusion serving-path optimization：https://github.com/sgl-project/sglang/pull/36680
- KDA QSA kernel：https://github.com/sgl-project/sglang/pull/36845
- KDA NVFP4 GEMM 与 Qwen3.5/Qwen3.8 E2E 数据：https://github.com/sgl-project/sglang/pull/36865
- KDA FLUX.2 FP8 fusion：https://github.com/sgl-project/sglang/pull/37162
- KDA backend 与 diffusion kernel 注册：https://github.com/sgl-project/sglang/pull/37385
- KDA-Pilot overlay 假收益与 production integration parity：https://github.com/BBuf/KDA-Pilot/pull/22 · https://github.com/BBuf/KDA-Pilot/pull/24 · https://github.com/BBuf/KDA-Pilot/pull/25
- KDA-Pilot benchmark contract、A/A、frozen rows 与 fallback/no-go：https://github.com/BBuf/KDA-Pilot/pull/40 · https://github.com/BBuf/KDA-Pilot/pull/41 · https://github.com/BBuf/KDA-Pilot/pull/43 · https://github.com/BBuf/KDA-Pilot/pull/79 · https://github.com/BBuf/KDA-Pilot/pull/89
- KDA-Pilot bitwise contract 与 production oracle：https://github.com/BBuf/KDA-Pilot/pull/152 · https://github.com/BBuf/KDA-Pilot/pull/157 · https://github.com/BBuf/KDA-Pilot/pull/158
- KDA-Pilot stateful kernel qualification：https://github.com/BBuf/KDA-Pilot/pull/194
- Kernel Design Agents：https://github.com/mit-han-lab/kernel-design-agents
- Humanize：https://github.com/PolyArch/humanize
