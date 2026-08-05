# 【翻译】使用 Codex 进行代码库现代化改造
> 来源：OpenAI Cookbook，原文链接：https://github.com/openai/openai-cookbook/blob/main/examples/codex/code_modernization.md
> 说明：本文为中文翻译，保留原文中的代码、命令、配置片段和 mdnice 图片链接。

## 引言

Codex 经过训练，能够阅读和推理大型、复杂的代码库，与工程师协同规划工作，并产出高质量的修改。代码现代化改造很快成为它最常见、最有价值的用途之一。在这种协作方式下，工程师专注于架构和业务规则，而 Codex 负责繁重工作：转换遗留模式、提出安全的重构建议，并在系统演进过程中保持文档和测试同步更新。

本 cookbook 展示了如何使用 **OpenAI 的 Codex CLI** 以一种满足以下要求的方式，对遗留仓库进行现代化改造：

* 便于新工程师理解
* 便于架构师和风险团队审计
* 可作为模式在其他系统中重复应用

我们将以一个基于 COBOL 的[投资组合系统](https://github.com/sentientsergio/COBOL-Legacy-Benchmark-Suite/)为贯穿全文的示例，并选择一条单独的 pilot 流程来聚焦。你也可以替换成任何遗留技术栈（例如 Java 单体、PL/SQL），只要其中包含遗留程序、编排逻辑（jobs、schedulers、scripts）或共享数据源即可。

---

## 高层概览

我们将整个过程拆分为 5 个不同阶段，这些阶段围绕一份执行计划（简称 ExecPlan）展开。ExecPlan 是一份设计文档，agent 可以依照它交付系统变更。

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-2/8c9f6ffb-193e-4d38-97e9-dec412ed26cf.png)

针对所选的 pilot 流程，我们将创建 4 类文档：

* **pilot_execplan.md** - 编排该 pilot 的 ExecPlan，用来回答：范围包含什么、为什么重要、我们将采取哪些步骤，以及如何判断工作完成。
* **pilot_overview.md** - 涉及哪些遗留程序（本例中是 COBOL）、哪些编排 jobs（这里是 JCL）和哪些数据源，它们之间的数据如何流动，以及整个业务流程实际在做什么。
* **pilot_design.md** - 系统的目标形态：哪个服务/模块将拥有这条流程、新的数据模型，以及对外公开的 API 或 batch 入口点。
* **pilot_validation.md** - 定义我们如何证明 parity：关键场景、共享输入数据集、如何并行运行 legacy 与 modern 版本，以及在实践中“输出一致”具体意味着什么。

这 4 个文件有助于明确：哪些代码会被修改、新系统应该长什么样，以及如何精确检查行为没有发生回归。

---

## Phase 0 - 设置 AGENTS 和 PLANS

**目标**：为这个仓库中的规划方式提供一份轻量级约定，让 Codex 可以遵循，同时不让流程给人带来负担。

我们借鉴了 [Using PLANS.md for multi-hour problem solving](https://cookbook.openai.com/articles/codex_exec_plans) 这篇 cookbook 的思路，创建一个 AGENTS.md 和一个 PLANS.md 文件，并将它们放在 `.agent` 文件夹中。

* AGENTS.md：如果你还没有为仓库创建 AGENTS.md，我建议使用 `/init` 命令。生成后，在你的 AGENTS.md 中增加一个小节，指示 agent 参考 PLANS.md。
* PLANS.md：以 cookbook 中提供的示例作为起点

这些文件说明了什么是 ExecPlan、什么时候需要创建或更新、它存放在哪里，以及每个计划必须包含哪些章节。

### Codex CLI 能提供什么帮助
如果你希望 Codex 针对你的特定仓库进一步完善 AGENTS 或 PLANS，可以运行：

```md
Please read the directory structure and refine .agent/AGENTS.md and .agent/PLANS.md so they are a clear, opinionated standard for how we plan COBOL modernization work here. Keep the ExecPlan skeleton but add one or two concrete examples.
```

---

## Phase 1 - 选择一个 pilot 并创建第一份 ExecPlan

**目标**：围绕一条既真实又有边界的 pilot 流程达成一致，并将 Phase 1 的计划记录到单个 ExecPlan 文件中。

**关键产物**：pilot_execplan.md

### 1.1 选择 pilot 流程
如果你还没有想好要用哪条流程作为 pilot，可以让 Codex 来提议。下面是在仓库根目录下的示例 prompt：

```md
Look through this repository and propose one or two candidate pilot flows for modernization that are realistic but bounded.
For each candidate, list:
- COBOL programs and copybooks involved
- JCL members involved
- The business scenario in plain language
- End with a clear recommendation for which flow we should use as the first pilot
```

在这个例子中，我们会选择一条报表流程作为 pilot。

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-1/4c9d4690-34b1-451f-b7cc-9ab5336dba71.jpg)

### 1.2 让 Codex 创建 pilot ExecPlan

```md
Create pilot_execplan.md following .agent/PLANS.md. Scope it to the daily reporting flow. The plan should cover four outcomes for this one flow:
- Inventory and diagrams
- Modernization Technical Report content
- Target design and spec
- Test plan for parity
Use the ExecPlan skeleton and fill it in with concrete references to the actual COBOL and JCL files.
```

这份计划现在就是你所有 pilot 工作的“主基地”。

---

## Phase 2 - 清点与发现

**目标**：梳理这条 pilot 流程当前实际在做什么：程序、jobs、数据流，以及业务规则。这样工程师无需逐行阅读所有 legacy 代码，也能推理出变更影响。

**关键产物**：pilot_reporting_overview.md

**工程师可以重点关注的地方**：

* 确认哪些 jobs 确实在生产环境中运行
* 补齐 Codex 无法仅从代码推断出的信息（SLA、运行上下文、owner）
* 对图示和描述做合理性校验

### 2.1 让 Codex 起草 overview
```md
Create or update pilot_reporting_overview.md with two top-level sections: “Inventory for the pilot” and “Modernization Technical Report for the pilot”.
Use pilot_execplan.md to identify the pilot flow.

In the inventory section, include:
1. The COBOL programs and copybooks involved, grouped as batch, online, and utilities if applicable
2. The JCL jobs and steps that call these programs
3. The data sets or tables they read and write
4. A simple text diagram that shows the sequence of jobs and data flows

In the modernization technical report section, describe:
1. The business scenario for this flow in plain language
2. Detailed behavior of each COBOL program in the flow
3. The data model for the key files and tables, including field names and meanings
4. Known technical risks such as date handling, rounding, special error codes, or tricky conditions
```

这份文档将帮助工程师在不通读全部代码的前提下，理解这条 pilot 的结构和行为。

下面是 pilot_reporting_overview.md 中流程图的示例：

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-2/6f1270e0-2eb0-4bf2-be1f-23090b4c630d.png)

### 2.2 更新 ExecPlan

当 overview 已经存在后，要求 Codex 保持计划与之对齐：

```md
Update pilot_execplan.md to reflect the new pilot_reporting_overview.md file.
- In Progress, mark the inventory and MTR sections as drafted.
- Add any notable findings to Surprises and discoveries and Decision log.
- Keep the ExecPlan readable for someone new to the repo.
```

在 Phase 2 结束时，你将拥有一份统一的 pilot overview 文档，同时承担系统清点报告与现代化技术报告（Modernization Technical Report）的角色。

---

## Phase 3 - 设计、规格与验证计划

**目标**

* 决定这条 pilot 流程的现代版本应该是什么样子
* 描述目标服务和数据模型
* 定义如何通过测试和并行运行来证明 parity

在这个阶段结束时，我们将明确要构建什么，以及如何证明它有效。

**关键产物**

* pilot_reporting_design.md
* pilot_reporting_validation.md
* modern/openapi/pilot.yaml
* modern/tests/pilot_parity_test.py

### 3.1 目标设计文档

```md
Based on pilot_reporting_overview.md, draft pilot_reporting_design.md with these sections:

# Target service design
- Which service or module will own this pilot flow in the modern architecture.
- Whether it will be implemented as a batch job, REST API, event listener, or a combination.
- How it fits into the broader domain model.

# Target data model
- Proposed database tables and columns that replace the current files or DB2 tables.
- Keys, relationships, and any derived fields.
- Notes about how legacy encodings such as packed decimals or EBCDIC fields will be represented.

# API design overview
- The main operations users or systems will call.
- A short description of each endpoint or event.
- A pointer to modern/openapi/pilot.yaml where the full schema will live.
```

### 3.2 API 规格

我们将这条 pilot 流程的外部行为记录在一个 OpenAPI 文件中，这样 modern 系统就有了一份清晰、与语言无关的契约。这份 spec 会成为实现、测试生成和未来集成的锚点，也能让 Codex 基于它具体地脚手架代码和测试。

```md
Using pilot_reporting_design.md, draft an OpenAPI file at modern/openapi/pilot.yaml that describes the external API for this pilot. Include:
- Paths and operations for the main endpoints or admin hooks
- Request and response schemas for each operation
- Field types and constraints, aligning with the target data model
```

示例输出：

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/c06bac87-7409-497a-affb-6483a0728668.png)

### 3.3 验证与测试计划

```md
Create or update pilot_reporting_validation.md with three sections:

# Test plan
- Key scenarios, including at least one happy path and a couple of edge cases.
- Inputs and outputs to capture for each scenario.

# Parity and comparison strategy
- How you will run the legacy COBOL flow and the modern implementation on the same input data.
- What outputs will be compared (files, tables, logs).
- How differences will be detected and triaged.

# Test scaffolding
- Notes about the test file modern/tests/pilot_parity_test.py, including how to run it.
- What needs to be filled in once the modern implementation exists.
```

然后让 Codex 脚手架测试：

```md
Using pilot_reporting_validation.md, create an initial test file at modern/tests/pilot_parity_test.py.

Include placeholder assertions and comments that reference the scenarios in the test plan, but do not assume the modern implementation is present yet.
```

### 3.4 更新 ExecPlan

```md
Update pilot_execplan.md so that Plan of work, Concrete steps, and Validation and acceptance explicitly reference:
1. pilot_reporting_overview.md
2. pilot_reporting_design.md
3. pilot_reporting_validation.md
4. modern/openapi/pilot.yaml
5. modern/tests/pilot_parity_test.py
```

在 Phase 3 结束时，你将拥有一份清晰的设计、一份机器可读的 spec，以及一个测试计划/脚手架，用来描述如何证明 parity。

---

## Phase 4 - 实现并比较

**目标：** 实现 modern pilot，让它与 COBOL 版本并行运行，并证明在计划中的场景下输出一致。

**关键产物**

* `modern/<stack>/pilot` 下的代码（例如 `modern/java/pilot`）
* 在 `modern/tests/pilot_parity_test.py` 中补全的测试
* `pilot_reporting_validation.md` 中更新后的章节，用于描述实际的并行运行步骤

### 4.1 生成 modern 代码的第一版草稿

```md
Using pilot_reporting_design.md and the COBOL programs listed in pilot_reporting_overview.md, generate initial implementation code under modern/<stack>/pilot that:
- Defines domain models and database entities for the key records and tables.
- Implements the core business logic in service classes, preserving behavior from COBOL paragraphs.
- Adds comments that reference the original COBOL paragraphs and copybooks.
- Treat this as a first draft for engineers to review.
```

你可以多次运行这一过程，每次聚焦不同的模块。

### 4.2 接上 parity 测试

```md
Extend modern/tests/pilot_parity_test.py so that it:
- Invokes the legacy pilot flow using whatever wrapper or command we have for COBOL (for example a script that runs the JCL in a test harness).
- Invokes the new implementation through its API or batch entry point.
- Compares the outputs according to the “Parity and comparison strategy” in pilot_reporting_validation.md.
```

### 4.3 记录并行运行步骤

与其单独写一个 `parallel_run_pilot.md`，不如复用验证文档：

```md
Update the Parity and comparison strategy section in pilot_reporting_validation.md so that it includes a clear, ordered list of commands to:
- Prepare or load the input data set
- Run the COBOL pilot flow on that data
- Run the modern pilot flow on the same data
- Compare outputs and interpret the results
- Include precise paths for outputs and a short description of what success looks like
```

### 4.4 （如有需要）使用 Codex 进行迭代修复

当测试失败或行为出现差异时，用短循环推进：

```md
Here is a failing test from modern/tests/pilot_parity_test.py and the relevant COBOL and modern code. Explain why the outputs differ and propose the smallest change to the modern implementation that will align it with the COBOL behavior. Show the updated code and any test adjustments.
```

每完成一块有意义的工作后，都让 Codex 更新 ExecPlan：

```md
Update pilot_execplan.md so that Progress, Decision log, and Outcomes reflect the latest code, tests, and validation results for the pilot.
```

你会发现，ExecPlan 中的 “progress” 和 “outcomes” 章节会被更新成类似下面的内容：

```md
Progress
- [x] Inventory and diagrams drafted (`pilot_reporting_overview.md` plus supporting notes in `system-architecture.md`).
- [x] Modernization technical report drafted (`pilot_reporting_overview.md` MTR section).
- [x] Target design spec drafted (`pilot_reporting_design.md` and `modern/openapi/pilot.yaml`).
- [x] Parity test plan and scaffolding documented (`pilot_reporting_validation.md` and `modern/tests/pilot_parity_test.py`).

Outcomes
- `pilot_reporting_overview.md`, `pilot_reporting_design.md`, and `pilot_reporting_validation.md` now provide an end-to-end narrative (inventory, design, validation).
- `modern/openapi/pilot.yaml` describes the API surface, and `modern/python/pilot/{models,repositories,services}.py` hold the draft implementation.
- `modern/tests/pilot_parity_test.py` exercises the parity flow using placeholders and helpers aligned with the validation strategy.
- Remaining work is limited to updating the operations test appendix and wiring the services to the real runtime.
```

---

## Phase 5 - 将 pilot 转化为可扩展的方法

**目标：** 为其他流程提供可复用模板，以及一份在本仓库中使用 Codex 的简短指南。

**关键产物**

* template_modernization_execplan.md
* how_to_use_codex_for_cobol_modernization.md

### 6.1 模板 ExecPlan

```md
Look at the pilot files we created:
1. pilot_reporting_overview.md
2. pilot_reporting_design.md
3. pilot_reporting_validation.md
4. pilot_execplan.md

Create template_modernization_execplan.md that a team can copy when modernizing another flow. It should:
1. Follow .agent/PLANS.md
2. Include placeholders for “Overview”, “Inventory”, “Modernization Technical Report”, “Target design”, and “Validation plan”
3. Assume a similar pattern: overview doc, design doc, validation doc, OpenAPI spec, and tests.
```

### 6.2 使用指南

```md
Using the same pilot files, write how_to_use_codex_for_cobol_modernization.md that:
1. Explains the phases at a high level (Pick a pilot, Inventory and discover, Design and spec, Implement and validate, Factory pattern).
2. For each phase, lists where coding agents helps and points to the relevant files and example prompts.
```

---

## 收尾

如果你按照这篇 cookbook 中的步骤，为任意一个 pilot 推进工作，最终得到的目录结构大致会是这样：一份 ExecPlan、三份 pilot 文档、一份 OpenAPI spec、一个 pilot 模块，以及一个 parity 测试。为了获得更清晰的结构，你还可以进一步把这些 markdown 文件组织到额外的 pilot 和 template 子目录中。

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/dca87b14-46c8-432f-89db-312c7c51265e.png)

你会注意到，`modern/python/pilot` 里还没有一个可运行的入口点，因为这些模块（`models.py`、`repositories.py`、`services.py`）目前只是第一版草稿式的构建块，用于启动工作。如果你想在本地做实验，有两个选择：

* 使用交互式 shell 或一个小脚本
* 创建你自己的 runner（例如 `modern/python/pilot/main.py`），把 repositories 和 services 组装起来

虽然这篇 cookbook 以 COBOL 的 pilot 流程作为贯穿示例，但相同的模式也会出现在非常不同类型的重构工作中。比如，有一个客户通过向 Codex 提供数百张 Jira 工单，让它标记高风险工作、识别横切依赖并起草代码修改，再由一个独立的 validator 审查和合并，最终完成了一个大型 monorepo 的迁移。

现代化 COBOL 仓库只是一个流行场景，但同样的方法适用于任何遗留技术栈或大规模迁移：把“现代化我们的代码库”拆解成一系列小而可测试的步骤（一份 ExecPlan、少量文档，以及一个以 parity 为优先的实现）。Codex 负责理解旧模式、生成候选迁移方案，并不断收紧 parity，而你和你的团队则可以持续专注于架构与权衡，从而让现代化改造在每一个你决定推进的系统上都变得更快、更安全、且可复用。
