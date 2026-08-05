# 【翻译】用 Codex 构建迭代式修复循环

> 来源：OpenAI Cookbook，原文链接：https://github.com/openai/openai-cookbook/blob/main/examples/codex/Build_iterative_repair_loops_with_Codex.ipynb
> 说明：本文为中文翻译，保留原文中的代码、命令、配置片段和 mdnice 图片链接。

这篇 cookbook 介绍的是闭环 agent workflow：agent 先产出结果，再对结果进行验证，并利用反馈改进下一轮输出。

我们将探索一个文档可靠性 workflow，它能够检测、修复并验证已经过时或损坏的 API 与 SDK 示例。本文中的示例使用了从这个 Cookbook 仓库改写而来的、故意保留过时问题的 notebook。

我们会用 Codex 来构建这个 agent loop。Codex 会审查当前状态，应用有针对性的修改，运行验证，并在反馈显示仍有问题时继续重复这一过程。

notebook 任务只是一个示例。只要 agent 的输出可以通过可信反馈进行衡量，这种模式就同样适用。

这个 workflow 分为三个阶段：

- **Review：** 检查当前 artifact，并以结构化形式返回发现的问题，但不编辑文件。
- **Repair：** 根据 findings 和最新的验证反馈，对复制出的 artifact 进行有针对性的编辑。
- **Validate：** 运行相关检查，并报告仍然需要修复的内容。

验证负责闭环。修复后的 notebook 必须满足真正重要的检查项，任何遗留问题都会成为下一轮 repair 的输入。

<p align="center">
  <img src="https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/b8c15595-a0be-4824-b32d-0edf0d166107.png" alt="迭代式 repair loop workflow" />
</p>

## 设置

这个 notebook 以无头模式使用 [Codex CLI](https://developers.openai.com/codex/cli)，因此 repair 步骤可以直接从 Python 单元运行，而不需要 chat UI。第一个代码单元会安装 CLI；如果你已经安装好了，可以跳过该单元。

在运行实际的 repair loop 之前，请先在环境中设置 `OPENAI_API_KEY`。

这个 notebook 默认使用一个较快的 repair model，这样整个示例可以在合理时间内跑完。如果你想试验其他模型，请在开始前设置 `REPAIR_MODEL`。安装单元为了可复现性固定了一个已知的 Codex CLI 版本；如果你希望获得更新的 CLI 行为，请有意识地更新这个版本。

## 加载示例 artifacts

下面的单元会加载三个配套 notebook，并总结驱动 repair loop 的元数据。

这些示例刻意保持较小规模。它们运行很快，但依然能覆盖这套架构：review 发现实质性问题，repair 做出有针对性的修改，validation 产出供下一轮使用的反馈。

如果你只单独下载了这个 notebook，那么在运行下面这些单元前，也需要一并下载配套的 `data/docs/` 文件夹，并将其放在 notebook 旁边。代码默认这些示例 notebook 可在本地获取。

在这个示例里，validation 会端到端执行每个修复后的 notebook。在其他领域，validation 可能是单元测试、策略检查、schema validator、simulation，或者人工审批步骤。关键点在于，失败不会成为死路，而是会转化为结构化反馈。

## 定义业务规则与问题分类

在让 Codex review 或 repair 一个 artifact 之前，先给它一个小而共享的契约。这样可以让 loop 聚焦于真正重要的问题，而不是要求模型从零开始推断所有产品规则和风格规则。

下面这些规则定义了这些示例 notebook 中“什么才算好”：采用当前 API 模式、设置说明清晰、本地示例可运行、并且保留原始教学目标。在其他 workflow 中，这个契约则会描述该领域中的事实来源。

## 定义结构化输出

每个阶段都返回结构化数据，这样下一阶段就有明确的输入可用。

Review 返回 findings。Repair 返回修改摘要以及更新后 artifact 的路径。Validation 返回供下一轮使用的剩余 delta。有了这种结构化交接，整个 loop 更容易调试、重跑，也更容易迁移到其他 artifact 类型。

## Review 阶段

Review 阶段会读取 artifact，并返回结构化 findings。它不会运行 validation，也不会编辑文件。这样的分离让第一步保持专注：先识别可能存在的问题，再进行任何修改。

我们通过 JSON schema 将 review prompt 发送给 `codex exec`。schema 能保证结果是机器可读的，因此后续单元可以把 findings 直接传给 repair prompt，而不是从上一轮回答的自然语言中再做提取。

## Repair 阶段

Repair 阶段会拿到当前 artifact、review findings、业务规则，以及上一轮留下的任何验证反馈。随着 loop 学到更多信息，prompt 也会变得更具体。

Codex 会编辑 iteration 目录中的副本，并返回一段简短摘要说明改了什么。loop 不会假定这次编辑已经成功；是否成功由下一步的 validation 来决定。

## Validation 阶段

Validation 的工作方式类似一个小型 eval。我们先定义期望行为，运行相关检查，再让 judge 按照该 rubric 为结果打分。

对于文档示例来说，执行是第一位的。许多 notebook 问题只有在运行时才会暴露出来：缺失的 import、过时的文件路径、依赖旧 API 响应的单元，或者对于作者来说足够清楚、但对新读者并不清楚的 setup 指南。

如果 validation 失败，这个失败结果就会成为下一轮 repair 的证据。这能让下一次 repair 立足于观察到的实际行为，而不只是基于 diff 中“看起来正确”的内容。

## 保存每轮输出

每一轮都会写出一个 `record.json` 文件；在本示例中，还会在 `CODEX_REPAIR_RUNS_DIR/iteration_N/<sample_name>/` 下写出修复后的 notebook。如果你没有设置 `CODEX_REPAIR_RUNS_DIR`，这个 notebook 会把结果写到系统临时目录中，这样普通的仓库 checkout 就能保持干净。

这些文件就是审计轨迹。你可以看到 review 发现了什么、Codex 改了什么、执行是否通过，以及哪些反馈被带到了下一轮。

`record.json` 文件是一轮 loop 尝试的回执。它把各阶段之间的交接集中保存在一个地方：

```json
{
  "review": [{"issue_type": "deprecated_api", "severity": "high"}],
  "repair": {
    "changes_made": ["Updated the notebook to use the current API pattern."],
    "updated_artifact_path": "/tmp/codex_iterative_repair_loop_outputs/iteration_1/sample/updated.ipynb"
  },
  "validation": {
    "passed": false,
    "remaining_delta": ["One setup instruction is still unclear."]
  }
}
```

正是这种紧凑的记录，才让维护者不需要从 notebook diff 和终端日志中重建整个过程，也能审查这个 loop。

## 运行第 1 轮

每个 notebook case 都彼此独立，因此我们会并发处理这些 case。这样既能让演示保持快速，又能为每个示例保留完全相同的 review、repair 和 validation 流程。

第 1 轮会复用之前 review 单元中得到的初始 findings。跑完这一轮后，请查看返回的布尔值：通过的 case 可以停止，失败的 case 则会把 validation 反馈带入下一轮。

## 运行第 2 轮

到了第 2 轮，这个 loop 的价值就开始体现出来了。Codex 不再只是依据最初的 review 工作；它还会看到 validation 阶段实际发生了什么。

这会改变任务本身。我们不再要求它做一次宽泛的重写，而是根据上一轮运行得到的证据，要求它执行下一次有用的 repair：什么执行成功了、什么通过了、以及什么仍然需要关注。

对于这里附带的分阶段 fixture，这一轮的设计目标是清理掉中等深度的 Evals case，而更深层的 Knowledge Retrieval case 则会继续推进，只留下一个更小、也更具体的 delta。

## 运行第 3 轮

第 3 轮聚焦于最复杂的文档 case。

Knowledge Retrieval 这个 fixture 既要完成 API 形态的现代化，还要保证使用本地数据时依然可运行，同时保留 retrieval 教学流程。这些要求之间可能会彼此拉扯：某种 repair 也许能让 notebook 变得更现代，却意外降低了可运行性；而另一种为了保持本地可运行的 repair，又可能删掉了太多原始教学内容。

第三轮会把最新的 notebook 连同最后一次 validation delta 一起交给 Codex。这正是这个演示体现“为什么迭代很重要”的地方：agent 响应的是最后遗留的具体问题，而不是试图在一开始就预判所有事情。

## 总结改进情况

现在我们可以直接查看整个运行过程，而不必手动打开每一个中间 artifact。下面的摘要展示了最重要的信号：哪些 artifact 通过了、还剩下多少 validation findings，以及是否仍有 delta 被继续带到下一轮。

对于附带的这些 fixture，预期形态很简单：一个 notebook 在第 1 轮通过，另一个在第 2 轮通过，而最复杂的那个会在第 3 轮通过。在真实的维护 workflow 中，这张表能告诉你 loop 是在收敛，还是需要更清晰的约束或人工 review。

这个摘要对人工 review 同样有用。维护者可以先看通过/失败的模式，再打开那些仍然存在 delta 的记录，并只检查已经准备好进入 review 的修复后 artifact。

## 这个摘要告诉了我们什么

重要的信号并不是 Codex 做了编辑。真正重要的是，随着 loop 的运行，剩余的 validation delta 会逐步缩小。

| 轮次 | 需要关注的信号 | 为什么重要 |
| --- | --- | --- |
| 第 1 轮 | 最简单的 fixture 通过；更深的 fixture 仍保留一个较小的 delta。 | 这说明 loop 能先完成初始 repair，同时把仍然需要证据支持的 case 继续往后传。 |
| 第 2 轮 | 中等深度的 fixture 在看到 validation 反馈后通过。 | 运行时反馈和 judge 反馈已经能够转化成有用的 repair 指令。 |
| 第 3 轮 | 最复杂的 fixture 通过，或只留下一个聚焦明确的最终 delta。 | 这说明 loop 正在收敛，或者它已经为人工 reviewer 产出了清晰的交接结果。 |

`record.json` 文件正是在这里变得可审计。一个有用的记录必须回答四个问题：review 发现了什么、Codex 改了什么、notebook 是否执行成功、还剩下什么？这正是“看上去很惊艳的编辑”与“维护者可以信任的 repair workflow”之间的区别。

## 推广为持续循环

上面固定三轮的运行方式适合用来讲解这种模式。而在生产环境中，loop 应该能够自行决定何时停止。

一个良好的 loop 通常会因为四种原因之一而停止：validation 通过、达到最大尝试次数、剩余 delta 不再变化，或者下一个决策需要人工 review。这些停止条件和 repair prompt 本身一样重要。

另一个生产级细节是审计轨迹。要为每一轮保留 review findings、修复后的 artifact、validation 结果、validation judgment，以及剩余 delta。这份记录能让维护者理解：为什么 loop 会继续、为什么它会停止、以及哪个 artifact 已经可以进入 review。

## 还能应用到哪里

这个 notebook walkthrough 只是讲解这套架构的一种方式。只要 agent 会修改某个文件或流程，而这个修改在被接受前需要的不只是主观 review，这种模式就都很有帮助。

下面是一些高价值示例：

- **Protocol optimization：** 先起草更新内容供专家 review，再根据剂量规则、时间约束或必须满足的安全检查进行验证。
- **Regulatory remediation：** 起草受监管内容的更新，然后检查所需措辞、引用、审批，以及特定司法辖区的术语是否仍然完整保留。
- **Support knowledge refresh：** 更新一篇文章，再根据当前产品行为或已知解决方案对其进行测试，并把不匹配之处带入下一轮。
- **Code modernization：** 替换已废弃的 API，运行测试或静态检查，再利用剩余失败来指导下一轮 repair。

这些场景的共同点在于，修改本身很重要，而且每一轮都需要证据。无论目标是 notebook、policy、protocol、support article、pipeline 还是 codebase，这个 loop 都为 agent 提供了一种基于证据持续改进的方式，而这些证据也能被维护者审查。

## 结论

迭代式 repair loop 能让 agentic maintenance 更容易被 review 和运维，因为它把判断与证明分离开了。

Review 负责找出候选问题。Repair 负责做出有针对性的编辑。Validation 负责执行 artifact 并产出下一轮 delta。当这些阶段之间通过结构化输出进行交换时，整个 workflow 会更容易检查、复现和迁移。

核心思想其实很简单：不要只依赖单次尝试，而是给 workflow 一个机会，让它先从 artifact 中学习，再做一次边界清晰的 repair，并对真实的 validation 反馈作出响应。这个小小的变化，会让 agentic maintenance 变得实用得多。
