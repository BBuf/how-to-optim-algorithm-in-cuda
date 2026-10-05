# AI-Infra-Auto-Driven-SKILLS v0.1.5：给 Codex / Claude Code 的推理框架工作流

相关项目链接，欢迎 star 和使用：

- SGLang：https://github.com/sgl-project/sglang
- AI-Infra-Auto-Driven-SKILLS：https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS
- Kernel Design Agents（KDA）：https://github.com/NVlabs/kda
- Humanize：https://github.com/humanfia/humanize

之前写过一篇 v0.1.0 的介绍(https://zhuanlan.zhihu.com/p/2042740770457772060)，整理了自己在推理框架开发中使用的 benchmark、profile、排障和模型优化 SKILLS。这次发布 v0.1.5，更新一下这段时间的变化，以及我现在使用这些 skill 的方式。

这个版本新增了模型 Day-0 支持和 Torch Profiler 层标记两个 skill，也更新了已有的 profile 分析、容量估算和模型 PR 优化历史。还有一个比较大的变化：把之前的 SOTA loop skill 都删掉了。

最近使用 GPT-6 Astra、Opus 5.5 的感受是，模型已经强到了可以直接根据目标（一个prompt）组织模型优化工作并直接实现目标，不需要任何额外的harness工具了。下面介绍一下v0.1.5的更新。

v0.1.5 Release：AI-Infra-Auto-Driven-SKILLS v0.1.5(https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/releases/tag/v0.1.5)。

# 0x1. Core Skills

当前仍然是 10 个任务 skill。相比 v0.1.0，删除了两个 SOTA loop，新增了两个更具体的工具：

| Skill | 解决的问题 |
| --- | --- |
| `llm-serving-auto-benchmark` | 在相同模型、硬件、workload 和 SLA 下，比较 SGLang、vLLM、TensorRT-LLM、TokenSpeed 的部署配置。 |
| `llm-serving-capacity-planner` | 从启动日志分析显存、KV cache、请求容量和 OOM 原因。 |
| `llm-torch-profiler-analysis` | 分析 Torch Profiler trace，输出 kernel、overlap、fusion 三张表，区分 prefill/decode。 |
| `llm-pipeline-analysis` | 继续下钻到 forward、layer 和 kernel，定位代表层及 Perfetto 时间范围。 |
| `torch-profiler-layer-track`（新增） | 给已有 trace 添加 L0/L1/... 层导航标记，并整理 GPU 显示轨道。 |
| `model-compute-simulation` | 根据模型结构估算算子形状、FLOPs 和 MFU，再与实际 kernel 对照。 |
| `sglang-model-day0-support`（新增） | 整理新模型接入 SGLang 的支持范围、实现缺口、PR 依赖和验证要求。 |
| `sglang-humanize-review` | 参考公开的2015年前的 SGLang 人类 review 语料，检查正确性、测试、性能和维护性问题。 |
| `sglang-prod-incident-triage` | 从 replay 和运行证据入手，排查 timeout、wrong output、crash、hang 等 serving 问题。 |
| `model-architecture-diagram` | 查找模型的公开原始架构图，辅助理解 LLM、VLM、MoE、OCR 和 diffusion 结构。 |

另外还有 `model-pr-history-knowledge`，对应仓库里的 `model-pr-optimization-history/`，用于查询各个模型家族已有的实现和优化 PR。

两个新增 skill 简单说一下。

`torch-profiler-layer-track` 主要解决看 trace 时的导航问题。一个模型有很多层，打开 Perfetto 后很容易面对一大片重复的 kernel，不知道自己看的是哪一层。这个 skill 会先核对模型层数和每层重复出现的 anchor kernel，再添加 L0、L1 等辅助标记。碰到 CUDA Graph 带来的大量 stream 显示行，也可以生成更紧凑的 GPU 视图。原始 trace、kernel 时间和 stream 信息都会保留，方便回头核对。这里的层标记是导航区间，具体层耗时还要结合实际边界分析。

`sglang-model-day0-support` 面向新模型接入。模型发布时，加载权重、跑通短文本生成只是其中一部分，量化、KV/state、并行拓扑、推测解码、多模态和协议输出都可能有需要单独处理的地方。这个 skill 把这些事情整理成支持范围、架构缺口、PR 依赖图和验证矩阵，并提供 DeepSeek-V4/V4.1、Kimi K3 的公开案例作为参考。做 Day-0 时，可以先让 Agent 用它把任务梳理清楚就是要做什么支持。

# 0x2. 为什么删除两个 SOTA Loop skills

v0.1.0 里面的 `sglang-sota-humanize-loop` 和 `vllm-sota-humanize-loop`，负责把 benchmark、profile、读源码、修改和复测串起来。它们会在固定的 workload 和 SLA 下比较几个框架，发现差距后继续找瓶颈，再尝试优化。当时觉得需要把这套过程写得比较细也确实有作用。Agent 要知道什么时候去采 profile，什么时候应该读历史 PR，什么时候需要回到端到端 benchmark，以及什么情况下才能认为一轮优化完成。两个 loop 就把这些决策提前写进了 skill。但用到 GPT-6 Astra、Opus 5.5 之后，这种流程除了浪费token是没什么用的了。现在的模型已经很擅长基于profile驱动的去定位bottleneck，写kernel优化kernel，再次测试直到达成目标。现在只要知道不同框架跑相同场景数据的性能差距，基本上就是让Agent蹬就可以把性能差距问题彻底解决掉。

**从我自己的使用感受看，在当前仓库几个 profile 相关 skill 的辅助下，用一个 `/goal` 加上具体的性能目标，已经几乎可以达成我想要的各种模型优化目标。** 这也是这次删除 SOTA loop 最直接的原因。

不过这里仍然需要 skill 提供的是分析工作的具体上下文。例如 prefill 和 decode 怎么分开，某个 kernel 对应哪个 Python 调用点，fusion 在什么 dtype、shape 和并行配置下成立，以及上游是否已经做过类似的优化。`llm-torch-profiler-analysis`、`llm-pipeline-analysis`、`model-compute-simulation` 和模型 PR 历史，把这些信息提供给 Agent 后，它就可以围绕当前bottleneck自己组织工作。同时也降低冷启动的时间。除此之外，在优化流程中Agent写kernel也时刻在调用Kernel Design Agent(https://github.com/NVlabs/kda)里面的kernel wiki，ncu-report-skill，这两个也是很有用的，推荐给大家。

# 0x3. 现在怎么给一个优化目标

在 Codex 中，我现在会直接用 `/goal` 设置持续目标，把模型、硬件、性能指标和验收方式写进去。Goal 模式会围绕这个目标继续工作，期间也可以在同一个对话里补充约束或调整方向。具体用法见 OpenAI 的长任务文档(https://learn.chatgpt.com/docs/long-running-work)。

例如下面这个 prompt。里面的模型、测试配置和性能数字按自己的任务替换，10% 是示例目标：

```text
/goal 在指定的 8×B200 环境中基于profile驱动的方式优化 SGLang 的 <模型 ID>。

固定模型和 tokenizer revision、FP8 精度、TP=8，以及同一份请求数据和并发配置。
先记录 baseline，并固定 TTFT/TPOT 的 SLA 和精度测试通过标准。
目标是在满足这些条件的前提下，把 output tokens/s 提升至少 10%，并给出可以review的patch。

你可以使用 https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS 和 https://github.com/NVlabs/kda/tree/main/skills 相关的skills。
```

上面的 `/goal` 是 Codex 的用法。在 Claude Code 中，也是一样的使用。

# 0x4. 本地 v1.5 的已有 Skills 做了哪些细化

这次更新对照了 SGLang、vLLM、TensorRT-LLM、TokenSpeed 和 FlashInfer 的源码，检查采集接口、调用路径和优化条件。具体版本记录在 [upstream-source-contracts.md](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/blob/v0.1.5/docs/upstream-source-contracts.md)。这里挑几个与日常使用关系比较大的变化。

首先是 profile。不同框架的采集参数、阶段切分和 trace 命名并不相同，这次把这些接口重新核对了一遍，也修正了从共享目录取到旧 trace 的问题。fusion 和 overlap 目录继续补充了 graph、launch overhead、通信重叠以及量化融合的案例。分析时会结合实际 backend、shape、dtype 和调用条件，判断当前模型是否走到了对应实现。

DeepSeek-V4.1 相关的优化经验也重新整理了。可复用的判断原则统一放进 `heuristics.md` 和已有 catalog，具体模型的历史 PR 分析放回模型档案，不再为这些优化单独维护几份 DSV4.1 reference 文件。分析其他模型时，也可以沿用这些原则。同时也整理了v0.1到v0.1.5期间各种框架的支持新模型的优化方式，这里面不仅有人类的优化方式更多的也有Agent的优化方式，skill也实现了rsi的更新（不是

容量分析和计算量估算则补了更细的模型结构与运行时情况，比如 TP 超过 KV-head 数量、graph capture 之后的容量调整、draft/Mamba 的独立分配，以及 MLA、MoE、压缩 attention、KDA 等结构。MFU 的时间口径也作了修正：有重叠的 kernel 不能直接把 duration 加起来，当作整段 forward 的耗时。

模型 PR 历史目前有 118 份中英双语档案，其中 SGLang 45 份、vLLM 44 份、TensorRT-LLM 15 份、TokenSpeed 14 份。准备修改一个模型时，可以先查已有实现、历史优化和 open PR。自动生成的条目提供检索线索，具体结论还需要回到完整 diff 和当前源码确认。

`sglang-humanize-review` 的公开 review 人类语料也从早期的 2024–2025 范围更新到了 2026-07-27（为啥是这个时间就不说了），包括 inline review、PR 评论和多轮讨论。Serving 排障工具则更新了日志和状态字段的解析。这些改动都比较具体，目的是让 Agent 拿到的证据更准确，少在过时接口和错误分析口径上浪费时间。

# 0x5. 安装和使用

这套 SKILLS 仍然是普通的 `SKILL.md` 目录，可以按任务安装。下面以 v0.1.5 为例，把当前 skill 链接到 Codex 的目录；使用 Claude Code 时替换 `SKILL_DIR` 即可。

```bash
git clone --branch v0.1.5 https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS.git AI-Infra-Auto-Driven-SKILLS-v0.1.5
cd AI-Infra-Auto-Driven-SKILLS-v0.1.5

# Codex
SKILL_DIR="${CODEX_HOME:-$HOME/.codex}/skills"
# Claude Code 改为：
# SKILL_DIR="$HOME/.claude/skills"

mkdir -p "$SKILL_DIR"
for d in skills/*/ skills/model-optimization/*/; do
  [ -f "$d/SKILL.md" ] || continue
  ln -sfn "$PWD/${d%/}" "$SKILL_DIR/$(basename "$d")"
done
ln -sfn "$PWD/model-pr-optimization-history" "$SKILL_DIR/model-pr-history-knowledge"
```

如果之前装过 `sglang-sota-humanize-loop`，升级时把它们的旧副本或链接移除。上面的安装循环只安装当前目录里的 skill，不会自动清理旧安装项。

安装之后，已有 trace 可以直接让 Agent 用 profiler 和 pipeline skill 分析；需要优化模型，就给出环境和性能目标；完整的版本变化见 v0.1.5 release notes(https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/blob/v0.1.5/release-notes/v0.1.5.md)。

这次更新之后，后续会继续把精力放在 profile 工具、分析方法和上游实现的整理上。模型已经能自己完成的流程，就让它自己完成；实际任务里反复用到的知识和容易出错的细节，再沉淀成 skill更新。
