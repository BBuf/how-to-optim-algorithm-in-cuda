# 使用 Agent 助力 SGLang 开发的初步探索

日期：2026-07-02

![Agent-assisted SGLang development](https://files.mdnice.com/user/59/4699bcb7-4d2c-4120-8b6e-ff21fa46718e.png)

SGLang 开发越来越不像一次孤立的代码修改。一个仓库里同时有 LLM serving、distributed runtime、GPU kernels、diffusion pipelines、model-specific execution paths 和 production incident handling。过去很多流程依赖开发者个人记忆：某个模型怎么启动，profile trace 怎么读，CUDA crash 先加哪类日志，一个性能 PR 应该补哪些 benchmark。Agent 工具成熟以后，这些经验可以被整理成可执行的 `SKILL.md`、脚本、benchmark contract 和 review loop。

围绕 SGLang 的 Agent 开发，已经出现了一组面向 LLM 和 diffusion 的 skills：

- [SGLang `.claude/skills`](https://github.com/sgl-project/sglang/tree/main/.claude/skills) 维护在 SGLang 仓库里，覆盖 CUDA crash debug、kernel 接入、测试、CI、profiling、生产排障和源码树约定等 repo-level 开发流程。
- [SGLang diffusion `.claude/skills`](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/.claude/skills) 聚焦 diffusion 相关流程，包括添加 diffusion 模型、benchmark/profile denoise path、调性能参数和验证量化 pipeline。
- [BBuf/AI-Infra-Auto-Driven-SKILLS](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS) 覆盖跨框架 serving benchmark、capacity planning、profile/pipeline analysis、model compute simulation、SGLang human-style review、production incident triage、面向 SGLang 和其它开源推理框架的 SOTA loop，以及 model PR history。
- [kernel-design-agents](https://github.com/mit-han-lab/kernel-design-agents) 是 KDA 项目，也是 MLSys 2026 FlashInfer Kernel Contest 的获胜方案。
- [BBuf/KDA-Pilot](https://github.com/BBuf/KDA-Pilot) 把 KDA 风格的 Agent kernel workflow 用到 SGLang。它公开的 B200 diffusion summary 目前跟踪 10 个 SGLang kernel task。大部分行来自 KDA-Pilot 的公开 benchmark ledger；`residual_gate_add` 使用的是原 task baseline 变化后，合入 SGLang 的集成 PR 里报告的 B200 speedup。目前 KDA-Pilot 派生的工作已经有 3 个 SGLang integration PR 落地。

把这些工作放在一起看，可以看到一个共同方向：Agent 的价值主要来自流程化的工程知识，包括可执行步骤、可复现实验和可审查证据。

## 1. TL;DR

- Agent 在 SGLang 里最有用的地方，是沿着定义清楚的 workflow 持续推进。benchmark、profile、kernel API logging、diffusion pipeline 添加、production incident replay 和 SOTA loop 都可以写成 skills。
- SGLang skill 是一份可执行的开发规程。以 `debug-cuda-crash`、`sglang-diffusion-benchmark-profile`、`llm-torch-profiler-analysis` 为例，真正重要的是 preflight、硬失败门禁、artifact contract、复现命令和结果格式。
- 性能优化离不开 profile 证据。SGLang profiler skills 固定输出 kernel table、overlap-opportunity table 和 fuse-pattern table；KDA-Pilot 进一步要求 baseline/candidate 同 ABI、真实 workload、correctness gate、NCU 证据和 per-shape 结果。
- 长期优化开始进入 Loop Engineering 阶段。SGLang SOTA Performance Loop 把“追 SOTA”拆成公平 benchmark、gap decision、profile、patch 和 revalidation。Humanize/RLCR 增加外部审查；Codex Goal 则可以用更低的协作开销跑同一类循环。
- Review 会更重要。Agent 能跑更多实验，也会生成更多看起来合理、但仍需要仔细审查的改动。开发者的工作会更多转向定义问题、选择证据、设计 workflow，以及判断结果能否进入生产路径。

## 2. 为什么 SGLang 适合 Agent-assisted development

SGLang 是面向 LLM 和多模态模型的高性能 serving framework。模型类型和硬件路径变多后，开发里会反复遇到几类问题：

- LLM 路径复杂。一个性能问题可能跨 Python runtime、scheduler、CUDA graph、Triton/CUDA kernel、FlashInfer/FlashAttention、distributed collective 和 model-specific wrapper。
- Diffusion 路径也复杂。一次 denoise 变慢，可能和 pipeline/stage 划分、DiT block、attention backend、`torch.compile` graph break、CFG/SP 并行、VAE 或自定义 fused kernel 有关。
- 验证成本高。很多修改必须在 H100/H200/B200 或 RTX 5090 上跑真实模型和真实 workload，本地单元测试不够。
- Profile 很难手工复用。一次 trace 里可能有几百个 kernel launch。人工读 Perfetto 容易漏掉 kernel 到 Python source 的映射，也容易把 prefill 和 decode 混在一起。开发者会在读 profiler 的过程中积累很多 know-how，比如哪些 kernel 名字对应哪段模型逻辑，哪些 launch pattern 暗示 graph break，哪些 NCCL/attention/MLP 排布是正常的。如果这些经验只留在个人脑子里，下一个任务就复用不了。
- 性能结论高度依赖上下文。GPU 型号、shape、batch size、parallelism、precision、backend 和 compile 状态都会改变结果。孤立的 microbenchmark 往往不能证明真实模型端收益，所以还需要一个端到端的长测试过程，在固定 workload 下反复验证吞吐、延迟、显存、精度和稳定性。这件事本身就很费力。

这些问题很适合交给 Agent 处理。启动 server、固定 workload、采集 trace、初筛 profile row、补测试、记录实验结果，都有清楚的输入输出，也适合脚本化和重复执行。开发者需要定义边界：同一套 benchmark 设置、同一组 profile 解释规则、同一批 accuracy gate，以及什么情况下 Agent 必须停止继续改代码。

因此，这里讨论的 Agent 是被工程流程约束的执行器。SGLang 开发里反复出现的流程可以沉淀成 skills，让 Agent 承担重复执行、证据收集和状态记录；开发者负责定义目标、判断证据，并审查改动是否应该进入真实 serving path。

## 3. 从 Prompt Engineering 到 SKILL：协议和例子

在 SGLang 框架里，一个有用的 skill 至少要回答这些问题：

| 问题 | Skill 中需要沉淀的内容 |
| --- | --- |
| 什么时候使用 | 触发场景、支持模型、支持硬件、必须停止的情况 |
| 怎么开始 | preflight、环境变量、repo 状态、依赖检查、模型配置 |
| 怎么验证 | benchmark 命令、profile 命令、测试入口、accuracy gate |
| 怎么判断 | 输出表格、失败模式、优先级、risk 分类、fallback 条件 |
| 怎么交付 | artifact 目录、结果 schema、PR 描述、复现命令、审查要求 |

SGLang 相关 skills 覆盖不同层级。有些贴近源码修改，比如 debug、test、diffusion model 添加和 benchmark/profile workflow；另一些面向跨框架 benchmark、capacity planning、compute simulation、production incident triage、PR 优化知识库、SGLang human-style review，以及 Humanize/RLCR 这类更上层的 workflow。

### 3.1 当前 Skill 栈

目前常用的 SGLang agent-related skills 大致可以分成下面几类。

| 层级 | 代表 skill / 项目 | 解决的问题 |
| --- | --- | --- |
| CUDA crash | [`debug-cuda-crash`](https://github.com/sgl-project/sglang/tree/main/.claude/skills/debug-cuda-crash) | 在 custom op/kernel API 边界记录输入、异常和 dump，把瞬时 crash 转成可以离线分析的样本 |
| LLM benchmark | [`llm-serving-auto-benchmark`](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/skills/llm-serving-auto-benchmark) | 对 SGLang 和其它 OpenAI-compatible inference stack 做公平、有限预算、可恢复的 serving benchmark search |
| Capacity planning | [`llm-serving-capacity-planner`](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/skills/llm-serving-capacity-planner) | 解析 SGLang 和其它推理框架启动日志，解释 weight memory、KV cache budget、CUDA graph overhead、request capacity 和 OOM 压力 |
| Trace triage | [`llm-torch-profiler-analysis`](https://github.com/sgl-project/sglang/tree/main/.claude/skills/llm-torch-profiler-analysis) | 固定输出 kernel、overlap-opportunity、fuse-pattern tables，并把 kernel 映射回 Python source；同一套统一 workflow 也在 AI-Infra 里支持跨框架分析 |
| Pipeline/layer analysis | [`llm-pipeline-analysis`](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/skills/llm-pipeline-analysis) | 把 torch profiler trace 切成 forward pass、layer 和 kernel flow，定位稳态 pass、瓶颈层类型和 Perfetto 时间范围 |
| Model compute simulation | [`model-compute-simulation`](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/skills/model-compute-simulation) | 构建 LLM operator-level compute template，估算 tensor shape、FLOPs、MFU、kernel-to-op 映射和 parallelism what-if |
| Diffusion benchmark/profile | [`sglang-diffusion-benchmark-profile`](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile) | 采集 denoise latency、perf dump 和 torch profiler trace，并先确认执行路径确实是 native SGLang diffusion backend |
| Add diffusion model | [`sglang-diffusion-add-model`](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-add-model) | 从 Diffusers/reference pipeline 出发，把新 diffusion 模型接入 SGLang 的 pipeline/stage/model/config 结构 |
| Diffusion performance tuning | [`sglang-diffusion-performance`](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-performance) | 选择 `torch.compile`、warmup、SP/CFG parallelism、offload、attention backend、quantization 等性能参数 |
| Production triage | [`sglang-prod-incident-triage`](https://github.com/sgl-project/sglang/tree/main/.claude/skills/sglang-prod-incident-triage) | 从 live server 收集 bundle、保存 failing request、replay，再路由到 crash/hang/profile 等专项工具 |
| SGLang review / PR history | [`sglang-humanize-review`](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/skills/sglang-humanize-review) 和 [`model-pr-history-knowledge`](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/model-pr-optimization-history) | 用真实 maintainer discussion pattern 审查 SGLang patch，并把 PR-driven model evolution histories 放在离源码修改更近的位置 |
| SGLang SOTA Performance Loop (Loop Engineering) | [`sglang-sota-humanize-loop`](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/skills/sglang-sota-humanize-loop) | 先公平比较 SGLang 和指定开源推理框架，再把 gap decision、profile、patch 和 revalidation 放进 Humanize/RLCR loop |

这些条目不是在堆工具名，而是在把容易遗漏的步骤写成可执行协议，让 workflow 能运行、能恢复，也能被人审查。

### 3.2 近期优化和 Workflow 例子

下面几个例子来自近期合入的 SGLang PR。表中关注完整工程路径：benchmark、profile、定位、代码修改、测试和复测。

| 案例 | 结果 | 关键点 |
| --- | --- | --- |
| Router long-context tokenization deduplication，[SGLang PR #28744](https://github.com/sgl-project/sglang/pull/28744) | 在 DeepSeek-V4-Flash 部署上，60k/125k-token prompts 的 idle TTFT 分别下降约 `29%` / `41%`；60k-token load 下 TTFT 下降 `34%–49%` | Agent 同时处理 cache-aware routing、chat-encoder parity、engine-side `input_ids` fallback 和 proxy body construction，避免 router 和 engine 重复 tokenization |
| Qwen3-Next FlashInfer allreduce fusion，[SGLang PR #22664](https://github.com/sgl-project/sglang/pull/22664) | H100 TP=4 上 request throughput 从 `5.49 req/s` 提升到 `9.41 req/s`，约 `+71.4%`；mean TTFT 从 `456.24 ms` 降到 `167.54 ms` | 这是 profile-driven 的 LLM collective 优化：未融合的跨卡 reduce 主导 prefill，融合 allreduce path 后又用 MMLU/GSM8K accuracy checks 验证 |
| Cohere2Moe NVFP4 fused-MoE path，[SGLang PR #27401](https://github.com/sgl-project/sglang/pull/27401) | `CohereLabs/command-a-plus-05-2026-w4a4` 在 1x B300 上，相比之前 SGLang default，chat throughput 提升 `+26%`，summarization 提升 `+21%`；在该设置下比另一个开源推理框架高 `+4.1%` / `+6.8%` | 这个改动补齐了 routing semantics，让已有 `flashinfer_trtllm` NVFP4 fused-MoE kernel 可以正确进入真实模型路径，并补了 GSM8K/MMLU 检查 |
| Kimi Delta Attention CuteDSL prefill kernel on SM100，[SGLang PR #27488](https://github.com/sgl-project/sglang/pull/27488) | 对 `moonshotai/Kimi-Linear-48B-A3B-Instruct`，B200 上 Delta Attention prefill 比 Triton 快 `1.08x–1.52x`；GSM8K 从 `0.915` 到 `0.920`，并新增 realistic gate magnitudes regression test | 这个 kernel task 必须覆盖模型里的 gate 分布、数值溢出、host overhead、真实模型 accuracy 和单测，才能进入可合入状态 |
| Spectral Progressive Diffusion，[SGLang PR #27524](https://github.com/sgl-project/sglang/pull/27524) | 在报告的 RTX A6000 设置下，FLUX.1、FLUX.2、Z-Image、Wan、Qwen-Image 的 denoising speedup 分别达到 `1.63x`、`1.77x`、`2.07x`、`2.32x`、`1.6x` | 这是 diffusion 侧的系统优化：早期 denoising 在更低 latent resolution 上运行，到高频细节开始重要时，再用 GPU DCT upsampling 恢复到完整分辨率。其中 `2.32x` 对应 Wan 的 denoising 阶段加速，不是整条端到端 pipeline 加速 |
| LTX-2 VAE decode channels-last-3d，[SGLang PR #27431](https://github.com/sgl-project/sglang/pull/27431) | LTX-2 decode stage 从 `5.41 s` 到 `3.84 s`，约 `1.41x`；peak reserved memory 从 `71.81 GiB` 降到 `62.12 GiB`，节省约 `9.7 GiB` | Profile 指向 Conv3d 和 layout conversion，所以 fix 保留 causal padding 里的 memory format，并把 loader policy 接到 single-GPU LTX-2 |

这些例子里，Agent 的主要贡献是执行 workflow：跑 benchmark、读 profile、定位 Python source、改代码、补测试、复测和准备 PR 描述。没有 skills 的时候，很多步骤依赖人工提醒；写进 skills 后，流程更容易重复。

## 4. Profiling、Review 和 Loop Engineering

SGLang 性能优化里一个常见误区，是只看总耗时，或者打开 Perfetto 看几分钟就凭直觉说“这里应该 fuse 一下”。这对 Agent 更危险，因为它很容易把一个视觉上很热的 kernel 误判成真正 bottleneck。

实际分析时，通常先配合使用两个 profiler skills。`llm-torch-profiler-analysis` 负责第一层 trace triage，把全局 profile 固定成三张表：

- `Kernel Table`：按 stage 统计 GPU time share、launch count、kernel category，并尽量映射回 Python source 和 CPU op。
- `Overlap Opportunity Table`：根据 exclusive/hidden time share、dependency risk 和 kernel category 判断哪里还有 overlap/headroom。
- `Fuse Pattern Table`：用 source-backed pattern catalog 对照 SGLang、其它开源推理框架和 kernel library 中已有的 fusion/overlap path。

这三张表先回答最基本的问题：哪个 stage 的哪个 kernel 占了多少 GPU time，映射到哪行 Python，是否有现成的 fuse/overlap path 可以借鉴。如果 SGLang 落后于另一个推理框架，第一步应该让 profiler table 解释 gap，而不是立刻改代码。

再往下是 `llm-pipeline-analysis`。全局热点知道以后，还要知道热点落在哪个 forward pass、哪类 layer、哪条 kernel flow 上。这个 skill 会读取 Chrome trace JSON 和模型 `config.json`，用 layer-boundary anchor kernel 把 trace 切成 forward passes 和 layers，然后输出几类适合深挖的表：

- `Forward pass summary`：区分 cold-start 和 steady-state，避免把 warmup 阶段当成优化目标。
- `Per-layer timeline`：按 layer 给出 wall time、sum duration，以及 MLA、MoE、GEMM、NCCL、MHC、Hadamard 等类别占比。
- `Layer cluster statistics`：对有交替层结构的模型尤其有用，比如带 `compress_ratios` 的 NSA/hybrid-attention 模型，可以看出 C4_LIGHT、C128_HEAVY、HASH 等层类型谁在拖慢。
- `Compute flow table`：选中代表性 layer 后展开具体 kernel flow，给出 hotness、相对时间戳和输入维度，方便直接跳回 Perfetto。

这样 profile 分析会变成两步：先用 `llm-torch-profiler-analysis` 找到全 trace 的主要矛盾，再用 `llm-pipeline-analysis` 把问题落到稳态 forward pass、代表性 layer 和具体 kernel flow。前者避免凭感觉选方向，后者避免只盯着一个全局 hot kernel，却忽略模型结构里的层类型差异。

### 4.1 Humanize/RLCR：给 Loop 加外部审查

Humanize 处理的是长期任务里的状态和审查问题。一个高风险的 SGLang 性能任务通常不会“一轮实现”就结束，它可能经历多轮 benchmark、profile、patch、revert、换方向和再验证。Humanize 把流程拆成两个阶段：

1. 先 gen-plan。`humanize-gen-plan` 把草稿需求整理成结构化 `plan.md`，里面包含 goal description、acceptance criteria、positive/negative tests、path boundaries、milestones 和 implementation notes。
2. 再跑 RLCR loop。`humanize-rlcr` 从 `plan.md` 启动循环。每轮 Claude Code 读取 `.humanize/rlcr/<timestamp>/round-<N>-prompt.md`，实现、提交、写 summary；Codex Review 再检查状态文件、summary、git clean、review result、open question、max iteration 等 gate。不能靠一句“任务完成”跳过。

这个机制给 SGLang SOTA Performance Loop 提供执行和 review 基础。Claude Code 跑 benchmark、读 profile、改 SGLang 代码并复测；Codex Review 在每轮结束时检查证据、状态和风险。它适合会进入 PR、会影响 serving 正确性，或者需要多天多轮实验的任务。

实际使用时，命令顺序最好写得很明确，避免 Agent 直接跳进实现：

```text
1. Write a task draft under artifact_root/draft.md.
2. Run humanize-gen-plan to generate artifact_root/plan.md.
3. Start humanize-rlcr from artifact_root/plan.md.
4. Keep all decisions, summaries, and review state in the local Humanize workspace.
```

### 4.2 SGLang SOTA Performance Loop（Loop Engineering）

单个 skill 可以把一次任务做稳。但十几轮实验之后，另一个问题会出现：哪个候选版本最好，哪些方向已经失败，上一轮 NCU 说明了什么，benchmark 是否还对齐 baseline，什么时候该停下。这些状态不能只靠聊天上下文记着。

SGLang SOTA Performance Loop 是一个基于 Humanize/RLCR 的 Loop Engineering workflow。这里的 SOTA 指固定实验条件下的可复现最好结果：同一个模型、硬件、GPU 数量、precision、workload、SLA、framework commit 和 serving 参数。问题是：在这些条件不变的前提下，SGLang 能否达到当前可复现的最好结果。

![SGLang SOTA Performance Loop](https://files.mdnice.com/user/59/66b47a55-b8fe-4b8d-b6fa-9e603785ee68.png)

图 1：SGLang SOTA Performance Loop。固定公平 benchmark 先给出可复现 baseline，后续 gap decision、profiling、pipeline analysis、patching 和 revalidation 由 Humanize/RLCR loop 推进。

一个完整的 SGLang SOTA Performance Loop 包含以下阶段：

1. 定义目标边界。例如 `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`、single-node 2x B200、FP8、SGLang TP=2，并在同样 2 卡预算下和指定的开源推理框架比较。
2. 先做公平 search。在 patch SGLang 之前，先用同样 workload 和资源预算搜索 SGLang 以及每个指定开源推理框架的可复现最好命令。
3. 判断 gap。如果 SGLang 已经持平或领先，就记录完成证据；如果稳定落后超过阈值，进入 profiling。
4. 用 profile 解释 gap。不要急着改代码，先产出 kernel tables、pipeline tables、overlap/fuse tables，必要时补 NCU。
5. 只 patch 有证据支持的路径。例如 hybrid attention、Mamba/GDN、radix cache、target verify、CUDA graph、MoE/EP、quant kernel 或 model wrapper。
6. 回到同一 workload 复测。每一轮都记录 benchmark、profile、accuracy、失败尝试、环境信息和清理动作。

对 `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` on 2x B200 这类目标来说，loop 很重要，因为 benchmark 结果、profile trace、失败 patch 和中间结论都要和同一个模型、硬件、workload、framework commit 绑定在一起。如果这类任务拆成很多独立 prompt，很容易丢掉哪个命令产出了哪个结果，或者后续 profile 是否还对应原始 baseline。带证据和 review 的 loop 更容易保持条件一致。

### 4.3 Codex Goal：低成本 Loop 实现

上面的 SGLang SOTA Performance Loop 使用双角色设置：Claude Code 负责 benchmark、profile、patch 和 revalidation，Codex Review 在每轮结束时审查。这种设置适合严肃 PR，但每一轮都会消耗执行模型和 review 模型，成本和等待时间都更高。

Codex Goal 提供了另一种实现。把“fair benchmark -> gap decision -> profile -> patch -> revalidate -> artifact ledger”写进一个持久 Goal 后，单个 Codex Goal 可以承担执行、自检和复测，不需要双角色执行/审查设置。SGLang SOTA Performance Loop 的核心约束不变：固定 workload、evidence-driven patches、在同一实验条件下复测，以及每轮更新 artifact manifest。

两种方式的差异如下：

| 维度 | Humanize/RLCR SOTA Loop | Codex Goal |
| --- | --- | --- |
| 执行方式 | Claude Code 做实现和实验；Codex Review 每轮审查 | 一个 Codex Goal 持续执行、自检和复测 |
| 状态位置 | `.humanize/rlcr/...` 里的 plan、prompt、summary 和 review result | 当前 Goal 线程 + `artifact_root` 里的 manifest/evidence |
| 审查方式 | Stop hook、Codex Review、git/state/schema checks | Goal-level self-checks、artifact contracts 和人工抽查 |
| 成本 | 两个模型角色参与，每轮成本更高 | 一个 Goal 同时承载执行和检查，成本更低 |
| 主要风险 | loop 配置更复杂，等待时间更长 | 如果 hard-stop conditions 写得不清楚，容易 goal drift 或过早完成 |

下面是 [AI-Infra-Auto-Driven-SKILLS/prompts](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/prompts) 中的一个 2x B200 模型优化 prompt 例子。

Humanize/RLCR 版：

```text
Use the sglang-sota-humanize-loop workflow.

Task:
Optimize SGLang serving performance for Qwen/Qwen3-Next-80B-A3B-Instruct-FP8
on a single node with 2 NVIDIA B200 GPUs, FP8 precision, and initial SGLang
TP=2. SGLang should match or exceed the best reproducible result from the
requested open-source inference frameworks under the same 2-GPU budget, workload, SLA,
model, precision, and environment constraints.

Required workflow:
1. Create a draft task document under artifact_root.
2. Run humanize-gen-plan to turn the draft into a structured plan.md.
3. Start humanize-rlcr from that plan.md in the Claude Code session.
4. Keep benchmark, profile, patch, and revalidation decisions inside the same
   Humanize workspace.

Evidence and safety requirements:
- Before patching, run a fair bounded search for SGLang and the requested open-source inference framework set.
- Check relevant open PRs in sgl-project/sglang and BBuf/sglang before choosing
  the SGLang baseline.
- If SGLang is behind by more than 1%, profile before patching.
- Prioritize evidence around hybrid attention, Mamba/GDN, radix cache, target
  verify, and CUDA graph.
- Record benchmark commands, profile artifacts, failed attempts, and cleanup
  evidence for every round.
- Patch only evidence-supported SGLang code paths.
- If a PR is needed, push/open it only against BBuf/sglang and include benchmark,
  GSM8K, and full MMLU accuracy tables.

artifact_root:
/workspace/sglang-agent-artifacts/b200_qwen3_next_80b_a3b_instruct_fp8_sota_humanize
```

Codex Goal 版：

```text
/goal Keep optimizing SGLang serving for
`Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` on a single node with 2 NVIDIA B200
GPUs until SGLang matches or exceeds the best reproducible result from the
requested open-source inference frameworks under the same 2-GPU budget, FP8 precision,
workload, SLA, model, and environment constraints. The current Codex Goal is the loop: fixed fair
benchmarking, gap decision, profiling, pipeline analysis, evidence-backed
patching, revalidation, final report, and optional PR preparation all happen
inside this Goal. Completion requires benchmark evidence, profile evidence when
SGLang was behind, correctness/accuracy evidence, a final artifact manifest,
and no regression in environment safety constraints.

model_id: Qwen/Qwen3-Next-80B-A3B-Instruct-FP8
root_dir: /workspace
target_hardware: single-node 2x NVIDIA B200
minimum_gpu_count: 2
precision_quantization: FP8
initial_deployment: SGLang TP=2
artifact_root:
/workspace/sglang-agent-artifacts/b200_qwen3_next_80b_a3b_instruct_fp8_sota_goal

Requirements:
- Use the current Codex Goal as the only persistent loop.
- Before patching, run a fair bounded search for SGLang and the requested
  open-source inference frameworks under the same 2-GPU budget.
- If SGLang is behind by more than 1%, profile in the same Goal, then use
  llm-torch-profiler-analysis, llm-pipeline-analysis, and ncu-report-skill when
  needed before patching.
- Focus on hybrid attention, Mamba/GDN, radix cache, target verify, and CUDA graph.
- Update the artifact manifest, benchmark evidence, profile evidence, failed
  attempts, and next-step decision after every round.
- Stop and report a blocker if resources are unavailable, evidence is
  untrustworthy, the budget is exhausted, or no defensible next patch exists.
```

Goal 版保留同样的 benchmark、profile、accuracy 和 artifact 要求。区别在于执行和 review 被折叠进一个持久目标。只要 hard-stop conditions 写清楚，它可以用更少的编排成本承载同一个 SGLang SOTA Performance Loop。

## 5. 基于 KDA 的 SGLang 系统 CUDA kernel 优化

LLM 和 diffusion 的模型级优化之外，kernel 优化有更硬的规模问题。不存在一个脱离硬件和 workload 的“最好 kernel”。同一个 operator 在 H100、H200、B200 或 B300 上可能偏好不同实现；不同模型结构会暴露不同 tensor shape 和 layout 约束；serving workload 又会改变 batch size、sequence length、precision format、wrapper overhead、同步行为和 fallback path。实际搜索空间是 hardware、model、workload 三者的笛卡尔积。

这会带来组合优化负担。对每个 candidate kernel，开发者都要抽取代表性的 production rows，构建 same-ABI harness，跑 A/B measurement，检查 shape buckets 上的 correctness，读 NCU metrics，判断某个 bucket 是否值得 specialization，然后再回到真实 SGLang path 复测。手工为每个 hardware/model/workload 组合做这件事很贵。它也是 Agent 擅长的重复、重证据流程，前提是人类先定义 invariants，并审查最后进入框架的路径。

不过，直接让 Agent 写 CUDA 很容易出现 benchmark reward hacking：改 benchmark、走更轻的 wrapper、打开 baseline 没有的 fast math、只优化一个 shape、破坏数值语义，或者在真实 SGLang 路径里没有任何收益。

KDA-Pilot 把 kernel 优化拆成隔离任务，避免 Agent 在 SGLang 大仓里随意改：

- Workloads 来自真实 SGLang diffusion 模型。流程先跑 20 个 diffusion 模型并汇总实际 kernel metadata。
- Baseline 从 upstream SGLang main 复制，并记录 source lineage。
- Baseline 和 candidate 必须走同一个 local ABI、同一个 build/export path。
- Benchmark 使用固定 production rows、A/B interleaving、CUDA event 或 wall timing。
- Correctness 覆盖 production rows、canonical regression grid、NaN/Inf 检查、poison output 检查和 fallback contract。
- 每轮迭代刷新 task prompt、benchmark evidence、KernelWiki 和 ncu-report-skill。
- 允许 shape-specialized dispatch，但每个 bucket 都要记录条件、路径、latency 和 fallback。

一个具体快照更容易看清规模。公开的 KDA-Pilot B200 diffusion summary 目前列出了 10 个跟踪中的 SGLang kernel task。大多数行在 KDA-Pilot ledger 中有稳定的 B200 数字，extracted production rows 上的 wall-geomean speedup 从 `1.1341x` 到 `2.7499x` 不等。`residual_gate_add` 这一行显示为 `1.11x`，对应合入 upstream 的 LTX-2.3 B200 结果。

截至 2026 年 6 月 27 日，KDA-Pilot 派生的优化已经有 3 个 upstream SGLang PR 落地。第一个是 [SGLang PR #27392](https://github.com/sgl-project/sglang/pull/27392)，为 Qwen-Image-2512 合入 B200 native diffusion norm-scale-shift CUDA fast path。同一周又合入了 [SGLang PR #29281](https://github.com/sgl-project/sglang/pull/29281) 和 [SGLang PR #29361](https://github.com/sgl-project/sglang/pull/29361)，分别覆盖 Cosmos3 VAE causal Conv3D cat/pad copy path 和 LTX-2.3 residual-gate update path。

| Upstream PR | 目标路径 | Kernel 侧证据 | 模型路径证据 |
| --- | --- | --- | --- |
| [#27392](https://github.com/sgl-project/sglang/pull/27392) | Qwen-Image norm-scale-shift | profiler attribution 中 target kernel group 提升 `1.279x` | 单张 B200 上双方各 5 次 interleaved 运行，full request `1.125x`，denoise wall `1.130x` |
| [#29281](https://github.com/sgl-project/sglang/pull/29281) | Cosmos3 causal Conv3D cat/pad | B200 traced VAE decode calls 上 weighted kernel group 从 `10.621 ms` 降到 `5.240 ms`，约 `2.03x` | Cosmos3-Nano T2V 开启 `torch.compile` 后，median E2E 从 `181.521 ms` 降到 `177.687 ms`，约 `1.021x` |
| [#29361](https://github.com/sgl-project/sglang/pull/29361) | LTX-2.3 residual-gate update | B200 LTX-2.3 large rows 相比已有 Triton path 提升 `1.108x` 到 `1.130x`，相邻 diffusion rows 最高到 `2.587x` | LTX-2.3 HQ T2V E2E 从 `46644.08 ms` 降到 `45198.37 ms`，约 `1.032x` |

关键结论不是每个 standalone kernel win 都会变成很大的端到端收益，而是同一套 KDA-Pilot evidence package 可以把 kernel task 从隔离 benchmark 推进到可审查的 SGLang serving path。这个 package 包括固定 production rows、correctness gates、same-ABI comparisons、profiler attribution 和 real-model checks。

![KDA-Pilot B200 diffusion kernel results](https://files.mdnice.com/user/59/ba64dd77-4226-41d5-a7ce-9ceeea9bafe4.png)

图 2：KDA-Pilot 优化的 10 个 SGLang diffusion kernel task 的 B200 证据。大部分行报告 KDA-Pilot wall-geomean speedup；这里的 wall time 包含 Python dispatch、wrapper overhead、kernel launch 和 `cuda.synchronize()` 能观察到的同步开销，比单纯 kernel device time 更接近真实调用路径。

| Kernel task | B200 evidence | 主要优化方向 |
| --- | ---: | --- |
| `qknorm_rope` | `1.1341x` | shared RoPE staging、Q/K 复用、large-row fast path |
| `norm_infer` | `1.3523x` | warp-row RMS、tiled persistent RMS、8B/16B vector path |
| `rotary_embedding` | `1.4912x` | 128-bit vector I/O、cos/sin hoisting、LTX2 block matching |
| `cutedsl_norm_tanh_mul_add` | `1.4953x` | row-invariant math hoisting、launch-bounds tuning、exact tanh |
| `cutedsl_norm_scale_shift` | `1.3201x` | operand-class dispatch、16B/32B vector、two-pass variance |
| `fuse_scale_shift` | `2.7499x` | rowgrid/flatvec/exact-C paths、cache hints、one-pass reduction |
| `group_norm_silu` | `2.3118x` | split-group stats、channels-last direct path、fallback for giant rows |
| `attention_concat_copy` | `1.30x` | single-launch region copy、pitched 16B block gather、严格 layout/device rejection |
| `causal_conv3d_cat_pad` | `2.06x` | flat chunking、16B vectorized stores、stride-aware fallback、bitwise-exact gate |
| `residual_gate_add` | `1.11x` | one-pass CUDA fusion、pinned-GPU correctness、SGLang PR #29361 的 B200 Triton-row re-benchmark |

图和表都要放在实验设置里看：它们报告的是 extracted production rows 上的 kernel-task speedup，不是完整模型端到端提升。即便如此，它们仍然有价值。只要 baseline、workload、correctness、profile 和 review 固定住，Agent 就能在真实框架 kernel 上做出可审查的增量。

KDA-Pilot 的实验里，有两条规则最值得保留：

- 不要给 Agent 留 benchmark reward hacking 的空间。baseline/candidate ABI 不一致、fast math 设置不同、wrapper 路径不同，结果都会失真。还有一种常见问题是看完结果以后再换 benchmark shape set，比如把 candidate 变慢的 shape 从表里删掉。这类结果不能用。
- 接近 Roofline 的 bucket 应该允许 no-go 或 fallback。好的 kernel 优化任务不应该逼 Agent 每个 shape 都赢。对 giant contiguous bucket 或已经接近带宽上限的路径来说，记录 fallback 可能比继续堆复杂度更正确。

## 6. 几条实践规则

1. 先定义任务边界，再启动 Agent。

“优化 SGLang”太大。“在 2x B200、固定 `1000->1000` 和 `8000->1000` workloads 下，让 `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` 追平另一个开源推理框架”才是一个可执行目标。

2. 先固定 benchmark，再看 profile。

如果 workload 可以在看到结果后改变，Agent 可能会无意中优化到一个更简单的问题。SOTA loop 和 KDA-Pilot 都把固定 workload 放在 patch 之前。

3. 看 NCU 结果要先判断 kernel 的计算性质。

Memory-bound kernel 重点看 DRAM/L2 throughput、load/store efficiency 和 memory pipe utilization；compute-bound GEMM/attention kernel 重点看 Tensor Core utilization、SM busy、eligible warps 和主要 stall 原因；小而碎的 latency-bound kernel 则要看 launch count、单 kernel duration、同步点，以及是否有融合空间。只贴一张 trace 截图不够，下一步代码改动应该由具体指标支撑。

4. 信任 profile 之前，先检查 backend 和 fallback gate。

如果 LLM run 静默切换 attention backend、禁用 CUDA graph，或者走了和 benchmark 不同的 wrapper path，那这个 trace 已经不再描述目标 serving path。Diffusion 也是一样：如果日志里 fallback 到 diffusers backend，就不能把这个 trace 当作 native SGLang diffusion 的证据。这类 hard-stop conditions 应该写进 skill。

5. Kernel 优化要同 ABI、同 wrapper、同 compile flags。

尤其不要让 candidate 偷偷走更轻路径，也不要单边开启 `--use_fast_math`。

6. Review 能力比以前更重要。

Agent 能制造更多 PR，也能制造更多看似合理的错误。SGLang 这类高性能系统的 review 需要检查 shape、dtype、distributed execution、CUDA graph behavior、fallback behavior、accuracy、serving API、metrics 和 benchmark setup。

Agent 时代的 SGLang 开发，不会把开发者从系统里拿掉。更现实的变化是：把开发者经验写成 workflow，把重复执行交给 Agent，把判断、设计和审查留给人。省下来的时间可以投入更难的性能问题、模型路径和生产稳定性，也可以继续反哺 Agent workflow。对开源推理框架来说，这类基础设施值得长期投入。

## 7. 致谢

感谢帮助构建 SGLang agent skills 的 SGLang Team 成员和贡献者：Xiaoyu Zhang (BBuf)、Lianmin Zheng、Liangsheng Yin、Ke Bao、fzyzcjy、Kangyan Zhou、DarkSharpness、Mick、Alison Shao、Baizhou Zhang、Bingxu Chen、Cheng Wan、Ratish P、shuwenn、ykcai-daniel、Yuhao Yang 和 Artem Savkin。

感谢 KDA team：Dongyun Zou、Ligeng Zhu、Sihao Liu、Junxian Guo、Yixin Dong、Zijian Zhang、Hao Kang 和 Song Bian。

感谢 Humanize team 和贡献者：Sihao Liu、Ligeng Zhu、Zijian Zhang、Zenus Zhang、shinan6、DYZhang、Chao Liu、Zhou Yaoyang、gyy0592、AcrossForest、Emin、Qiming Chu、jiaxiaoyu、tastynoob 和 zhenwei。

## 8. 参考资料

- [SGLang GitHub Repository](https://github.com/sgl-project/sglang)
- [SGLang `.claude/skills`](https://github.com/sgl-project/sglang/tree/main/.claude/skills)
- [SGLang diffusion `.claude/skills`](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/.claude/skills)
- [AI-Infra-Auto-Driven-SKILLS](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS)
- [AI-Infra-Auto-Driven-SKILLS prompts](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/prompts)
- [Kernel Design Agents (KDA)](https://github.com/mit-han-lab/kernel-design-agents)
- [KernelWiki skill](https://github.com/mit-han-lab/KernelWiki)
- [ncu-report-skill](https://github.com/DongyunZou/ncu-report-skill)
- [KDA-Pilot](https://github.com/BBuf/KDA-Pilot)
- [SGLang Diffusion Advanced Optimizations, LMSYS Blog](https://lmsys.org/blog/2026-02-16-sglang-diffusion-advanced-optimizations/)
- [OpenAI Codex Prompting: Goal mode](https://developers.openai.com/codex/prompting#goal-mode)
- [Humanize](https://github.com/PolyArch/humanize)
