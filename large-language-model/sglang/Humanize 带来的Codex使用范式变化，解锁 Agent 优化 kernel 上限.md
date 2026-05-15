> Humanize 带来的Codex使用范式变化，解锁 Agent 优化 kernel 上限

# 0x0. 前言

之前做了几个模型 profile 调优相关的 SKILL，还有一个 kernel 优化相关的 SKILL，但效果只能说够用，还没有达到长期自动优化的程度。原因也简单：SKILL 本质上还是一次性注入 context，Agent 可以照着做一段时间，但很难自己长期维护目标、证据、失败记录和下一步方向。

GPT-5.5 这类模型已经足够强了，单次补代码不难。麻烦的是长任务：它需要在十几个小时里围绕同一个目标持续迭代，每轮都留下测试、benchmark、profile 和评审证据。最近重点看了 [Humanize](https://github.com/PolyArch/humanize)、Codex `/goal`、AVO 论文，并新做了 [KernelPilot](https://github.com/BBuf/kernel-pilot) 这个仓库。

模型单次写代码能力已经不是主要瓶颈，更麻烦的是能不能把它放进一个长期循环里。这个循环要能读代码、跑 benchmark、看 NCU、记录版本演进关系；当连续几轮收益很低时，还要按当前瓶颈补读新的 PR、源码、测试、benchmark 和 profile 记录，并把已读来源记下来，避免下一轮重复读；最后还要通过独立审查拦住过早完成。少任何一个环节，都会退化成普通 Vibe Coding：人需要手动多轮追问，不断提醒它下一步做什么，也很难稳定跑出后文这种性能优化效果。

# 0x1. Humanize 改变了什么

[Humanize](https://github.com/PolyArch/humanize) 这里用的是 RLCR，也就是 Ralph-Loop with Codex Review。它做的事情比较直接：把实现、总结、审查、继续迭代这些动作固定下来，让 Agent 不再靠当前对话窗口硬撑。

- 有 plan，有 acceptance criteria，不是想到哪改到哪。
- 每轮实现完必须写 summary。
- Codex 独立审查 summary 和代码，不通过就继续下一轮。
- 最后还有代码审查阶段，不能自己说完就完。
- 状态都落到 `.humanize/`，不是靠当前对话窗口硬记。

Humanize 最有用的点是把“完成”变成了一个外部判定，不再由 Agent 自己判定。以前用 Codex 做推理框架优化，经常要人工盯着它：跑到一半是不是忘了 profile？是不是只修了测试？是不是 benchmark 没对齐 baseline？Humanize 至少把这些问题转成了可审查的状态文件和下一轮 prompt。

所以它带来的变化不只是“多一个命令”。更准确地说，使用 Codex 的姿势变了：从一问一答，变成把 Codex 放进一个带审查的工程循环。

# 0x2. AVO 和 Codex /goal

AVO 论文 [Agentic Variation Operators for Autonomous Evolutionary Search](https://arxiv.org/abs/2603.24517) 讨论的是把 LLM 从候选生成器升级成 variation operator：Agent 可以自己看历史版本、知识库和执行反馈，然后决定读什么、改什么、测什么、怎么修。

这篇论文里 AVO 在 B200 上连续跑 7 天，探索了 500 多个优化方向，提交了 40 个 kernel 版本，最后在 MHA 上比 cuDNN 最多快 3.5%，比 FlashAttention-4 最多快 10.5%。后面把 MHA 迁移到 GQA 又只花了 30 分钟左右。相比这些数字本身，更有用的是它把一件事摆明了：kernel 优化可以按长期 autonomous search 来做，这和单次生成一个 kernel 是两回事。

Codex `/goal` 也在往这个方向走。OpenAI 文档里把 `/goal` 定义成 experimental long-running task target，需要 `features.goals`，目的是让 Codex 在长任务里有一个持久目标。不过单独一个 `/goal` 还不够。Kernel 优化只会继续跑还不够，必须带着证据继续跑。

简单对比一下：

| 项目 | 做什么 | 这里怎么用 |
| --- | --- | --- |
| Humanize | 外部审查门禁 + 状态机 | 防止 Agent 过早停止、目标漂移、误判完成 |
| Codex `/goal` | Codex 原生长目标 | 让长任务进入官方运行时能力范围 |
| AVO | 论文里的长期优化实验 | 看版本关系、知识库和执行反馈能把优化推进到什么程度 |
| KernelPilot | 当前的落地版本 | 把 Humanize 改造成 GPU kernel 优化循环 |

# 0x3. 普通 CUDA SKILL 的限制

普通 SKILL 最大的问题是太像“说明书”。它能告诉 Codex 怎么写 kernel、怎么 benchmark、怎么用 ncu，但不负责把一次次实验串起来。

实际 kernel 优化里最值钱的东西往往落在这些状态上：

- 当前 best 是哪个版本，为什么选它。
- 哪些方向已经试过，为什么失败。
- benchmark 的 shape、dtype、baseline 是否一致。
- NCU 到底显示是 tensor pipe 不够、L1/L2 replay、long scoreboard、shared bank conflict，还是 launch overhead。
- 连续两轮小于 1% 收益时，应该去读哪些新代码，而不是继续在原地抠一个参数。

以前把一大堆 SGLang、vLLM、TensorRT-LLM、CUTLASS、FlashInfer、DeepGEMM、GPU Mode、CUDA blog 的知识塞进 SKILL，最后还是会碰到 context 和检索问题。一整本文档塞进去没用，更需要一个会路由的知识库：当前是 GEMM 就读 GEMM，当前是 attention 就读 attention，连续两轮低收益就按主题和框架去补读新的 PR diff、kernel source、tests、benchmark 和 profile 记录，并且记住哪些来源已经读过。

# 0x4. KernelPilot 做了什么

[KernelPilot](https://github.com/BBuf/kernel-pilot) 基本就是把上面这套想法塞进 Humanize 之后的结果。它基于 vendored upstream Humanize，再加 kernel 专用工作流。

相比原始 Humanize，主要改了这些：

- 新增 `humanize-kernel-agent-loop`：用户只给一句 kernel 优化目标，它自己生成 plan、refine plan、建 standalone repo、启动 RLCR。
- 新增 `kernel-knowledge`：本地 GPU kernel 知识库，先按 topic/framework 路由，再读 source guide、PR notes、benchmark、test。
- 新增 `profile-evidence`：把 NCU 输出整理成 Profile Evidence Digest，最后必须落到一个具体下一步修改。
- 强制 standalone repo：候选 kernel 不直接污染 SGLang/vLLM 这种大仓库，单独保留 binding、tests、benchmarks、ledgers、profile artifacts。
- 强制记录：`attempt-ledger.md` 记录所有版本，`optimization-ledger.md` 只记录有效提速版本，`source-idea-ledger.md` 记录来源和不要重复阅读的标识，版本关系记录里保存父版本、动机、结果。
- 连续两轮收益小于 1% 时触发资料扩展：至少补读 50 个新的代码来源，再继续改。

知识库目前主要覆盖这些东西：

- SGLang、vLLM、TensorRT-LLM、FlashAttention、FlashInfer、CUTLASS/CuTe、DeepGEMM、TileLang、CCCL/CUB 等 CUDA 优化 PR。
- PyTorch、Triton、ThunderKittens、DeepSeek TileKernels、QuACK、blog companion code 这类 source-only 代码索引。
- AKO4ALL 的 CUDA/CUTLASS/Triton/NCU 参考资料。
- 按 topic 拆的 matmul-gemm、attention、moe、normalization、activation-fusion、kv-cache、communication 等入口。

NCU 里主要看这些指标：

| 类别 | 主要看什么 |
| --- | --- |
| 总体吞吐 | `Compute (SM) Throughput`、`Memory Throughput`、tensor pipe 利用率 |
| 内存层级 | DRAM / L2 / L1TEX throughput，L1/L2 hit rate |
| 调度 | `No Eligible`、`Eligible Warps Per Scheduler`、issue slot busy |
| stall | long scoreboard、short scoreboard、no instruction、imc miss、dispatch stall |
| 访存形态 | global load sector 利用率、uncoalesced global/shared access |
| shared memory | bank conflict、shared load/store replay |
| 资源 | registers/thread、shared memory/block、occupancy、waves/SM、spills |

NCU 的作用是先判断瓶颈类型，再缩小优化方向。compute bound 主要看 tensor pipe / SM throughput、tile 形状、epilogue 开销和指令选择；memory bound 主要看 DRAM/L2/L1TEX 吞吐、sector 利用率、访存合并、cache hit rate；latency bound 主要看 long/short scoreboard、eligible warps、occupancy、寄存器和 shared memory 冲突。不同瓶颈对应的改法不一样，比如 compute bound 倾向改 tile 和 tensor core 路径，memory bound 优先处理 coalesce、向量化和数据复用，latency bound 则更关注 occupancy、独立指令、shared bank conflict 和减少依赖链。

# 0x5. Prompt Cards 和用法

安装很简单：

```bash
git clone https://github.com/BBuf/kernel-pilot.git
cd kernel-pilot
./scripts/install-codex-skills.sh
```

然后重启 Codex，在 `/skills` 里确认能看到：

- `humanize-kernel-agent-loop`
- `kernel-knowledge`
- `profile-evidence`

这次 `int8_scaled_mm` 演示用的是从零手写 prompt，不碰 baseline kernel 代码：

```text
[$humanize-kernel-agent-loop] I want to optimize SGLang's H100 int8_scaled_mm kernel on H100. Implement the candidate kernel from scratch and use the existing SGLang/CUTLASS kernel only as the correctness/performance comparison baseline. Work in a clean standalone repo and keep source provenance plus version history.
```

这套流程自主跑了约 12 小时，从零写了一个 `int8_scaled_mm`，最后在 focused case 上比 SGLang baseline 快 `15.12%`。下面是 Humanize stop hook 和 KernelPilot ledger 的效果图：

![](https://raw.githubusercontent.com/BBuf/kernel-pilot/main/docs/assets/humanize-stop-hook-summary.png)

![](https://raw.githubusercontent.com/BBuf/kernel-pilot/main/docs/assets/kernelpilot-optimization-ledger.png)

不过对于 SGLang 已经有的 kernel，不建议默认从零开始。更快的方式是用第二种 prompt：把已有 baseline kernel 作为起点，在它的基础上继续优化，这样可以跳过很多从 correctness-first 到接近 baseline 的无效路程。

```text
[$humanize-kernel-agent-loop] I want to optimize SGLang's H100 int8_scaled_mm kernel on H100. Use the existing SGLang/CUTLASS kernel as the baseline and starting point. Work in a clean standalone repo, keep source provenance plus version history, and use the most appropriate kernel language for the candidate.
```

这次从零实验里，SGLang 只作为 correctness/performance baseline 和 prior art，候选实现限制为手写 CUDA C++，不用 Triton、CUTLASS、CuTe DSL、ThunderKittens、torch.compile、cuBLAS/cuBLASLt。

# 0x6. int8_scaled_mm 当前效果

这次实验目录在 `/Users/bbuf/工作目录/Common/int8_scaled_mm`，Humanize 缓存在 `/Users/bbuf/.cache/humanize/-Users-bbuf-Common-int8_scaled_mm`。

当前 selected 版本是 `v23-wmma-inblock-split-k8-two-n-half2`。在 H100 上，`M=64,N=2048,K=2048,fp16,bias=true` 这个 focused case：

- v0 scalar：`0.603872 ms`
- SGLang same-run baseline：`0.017888 ms`
- 当前 v23：`0.015184 ms`

也就是从最初手写 scalar 到 v23，大约 `39.8x`；相比同次 SGLang baseline，v23 快 `15.12%`。后面 v24-v29 又试了几轮，但是都没有过 1% selection gate，所以当前 best 仍然是 v23。Round 41 做完了新一轮资料扩展，下一步方向是 `v30-wmma-inblock-split-k8-two-n-combined-rowmajor-acc`，还没选中。

下面是主要版本记录：

| 版本 | p50(ms) | 结果 | 这一轮改了什么 |
| --- | ---: | --- | --- |
| v0 | 0.603872 | baseline | 一个 thread 算一个输出元素，先保证 correctness |
| v1 | 0.163648 | selected | warp-per-output，lane 沿 K 维合作累加 |
| v2 | 0.122496 | selected | shared memory staging A row，减少重复读 A |
| v3 | 0.126400 | rejected | shared-A 扩到 16 warps，变慢 |
| v4 | 0.179584 | rejected | K loop 双 accumulator，寄存器/依赖更差 |
| v5 | 0.074624 | selected | `__dp4a` packed-K 路线 |
| v6 | 0.058592 | selected | 一个 warp 同时算两个相邻 N |
| v7 | 0.048576 | selected | contiguous 32-bit stride fast path，压地址开销 |
| v8 | 0.052576 | rejected | dp4a K loop unroll2，变慢 |
| v9 | 0.048064 | selected | 一个 warp 算四个相邻 N，小幅收益 |
| v10 | 0.045792 | selected | `__launch_bounds__` 压寄存器 |
| v11 | 0.045088 | selected | B pointer 改成 base offset，减少 live state |
| v12 | 0.045088 | rejected | exact N/K specialization，没有收益 |
| v13 | 0.053024 | rejected | 放松 launch bounds 到 6 blocks/SM，回归 |
| v14 | 0.023392 | selected | 切到手写 WMMA int8 16x16 tensor core |
| v15 | 0.021216 | selected | split-K8，partial int32 再 reduce |
| v16 | 0.027776 | rejected | split-K8 + shared-A grouped-N，回归 |
| v17 | 0.018176 | selected | in-block split-K8，把 reduce/epilogue 合进一个 CTA |
| v18 | 0.018272 | rejected | shared accumulator padding 到 stride 20，没过 |
| v19 | 0.020160 | rejected | launch-bounds8，回归 |
| v20 | 0.018432 | rejected | split-K4，回归 |
| v21 | 0.021632 | rejected | exact 2048x2048 in-block split-K8，回归 |
| v22 | 0.017184 | selected | epilogue 用 half2 成对写回 |
| v23 | 0.015184 | selected | 一个 CTA 算两个 16x16 N tile，复用 A fragment |
| v24 | 0.015200 | rejected | v23 上再做 accumulator padding，基本持平 |
| v25 | 0.015168 | rejected | launch-bounds5，收益不到 1% |
| v26 | 0.015280 | rejected | epilogue reduction 用 `int2` pair load，回归 |
| v27 | 0.015168 | rejected | `cudaFuncCachePreferL1`，收益不到 1% |
| v28 | 0.023168 | rejected | inline `mma.m16n8k32` + register epilogue，明显回归 |
| v29 | 0.015088 | rejected | vectorized scale/bias load，只有 0.63%，没过 1% gate |

这里比较有参考价值的是，很多看起来“应该更快”的方向，实际 benchmark 后并没有收益。比如 padding shared accumulator、launch bounds、prefer L1、inline MMA，都需要靠测试和 profile 结果确认。Humanize + KernelPilot 的好处就在这里：每个方向都留下证据，失败也有用，下一轮不会重复尝试同一个方向。

# 0x7. 解锁更多模型优化的可能

按现在这套流程看，SGLang 这类推理框架的优化会变成三件事：

- 人定义目标、边界和验收标准。
- Agent 负责长期执行、读代码、改代码、跑实验。
- Harness 负责记忆、审查、profile 证据和停止条件。

如果只是让 Codex 写一个 kernel，效果其实很一般，甚至要求使用者本身就是 GPU kernel 专家，能给出方向、看懂 profile、判断结果是否可信。如果把 SGLang、vLLM、TensorRT-LLM、FlashInfer、CUTLASS、DeepGEMM 这些项目里的优化经验做成可检索知识库，再配上 Humanize/`goal`/NCU/benchmark 这种长期循环，模型优化和 kernel 优化就更像 AVO 那种 autonomous search。

我正在用 Humanize 把模型调优和 kernel 优化结合起来，后续会推到 [AI-Infra-Auto-Driven-SKILLS](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS) 这个仓库里。目标是让 SGLang 的模型优化和 kernel 优化全部自动驾驶起来，给出任务之后尽量不需要中途介入。

# 0x8. 参考

- Humanize: https://github.com/PolyArch/humanize
- KernelPilot: https://github.com/BBuf/kernel-pilot
- AVO paper: https://arxiv.org/abs/2603.24517
- Codex `/goal` docs: https://developers.openai.com/codex/cli/slash-commands#set-an-experimental-goal-with-goal
- AI-Infra-Auto-Driven-SKILLS: https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS
- SGLang: https://github.com/sgl-project/sglang
- vLLM: https://github.com/vllm-project/vllm
- TensorRT-LLM: https://github.com/NVIDIA/TensorRT-LLM
- FlashInfer: https://github.com/flashinfer-ai/flashinfer
- FlashAttention: https://github.com/Dao-AILab/flash-attention
- CUTLASS/CuTe: https://github.com/NVIDIA/cutlass
- DeepGEMM: https://github.com/deepseek-ai/DeepGEMM
- TileLang: https://github.com/tile-ai/tilelang
- CCCL/CUB: https://github.com/NVIDIA/cccl
- AKO4ALL: https://github.com/TongmingLAIC/AKO4ALL
