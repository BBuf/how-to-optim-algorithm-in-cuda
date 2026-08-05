# 【翻译】使用 Codex CLI 和 Agents SDK 构建一致的工作流
> 来源：OpenAI Cookbook，原文链接：https://github.com/openai/openai-cookbook/blob/main/examples/codex/codex_mcp_agents_sdk/building_consistent_workflows_codex_cli_agents_sdk.ipynb
> 说明：本文为中文翻译，保留原文中的代码、命令、配置片段和 mdnice 图片链接。

### 确保可重复、可追踪、可扩展的 agentic 开发

## 引言
开发者在所做的一切事情上都追求一致性。借助 Codex CLI 和 Agents SDK，这种一致性现在可以以前所未有的方式扩展。无论你是在重构大型代码库、推出新功能，还是引入新的测试框架，Codex 都能无缝融入 CLI、IDE 和云端 workflow 中，用于自动化并强制执行可重复的开发模式。

在这一部分中，我们将使用 Agents SDK 构建单 agent 和多 agent 系统，并将 Codex CLI 作为一个 MCP Server 暴露出来。这将带来：
- **一致性与可重复性**：为每个 agent 提供有边界的上下文。
- **可扩展的编排能力**：用于协调单 agent 和多 agent 系统。
- **可观测性与可审计性**：通过查看完整的 agentic stack trace 来实现。

## 本文将涵盖的内容
- 将 Codex CLI 初始化为 MCP Server：如何将 Codex 作为一个长期运行的 MCP 进程启动。
- 构建单 Agent 系统：使用 Codex MCP 处理具备明确范围的任务。
- 编排多 Agent Workflow：协调多个具备专长的 agent。
- 跟踪 Agent 行为：利用 agent traces 获取可见性并进行评估。

## 前置条件与准备
在开始这一部分之前，请确保你具备以下条件：
- 基本编码基础：你应当熟悉 Python 和 JavaScript。
- 开发环境：你需要一个 IDE，例如 VS Code 或 Cursor。
- OpenAI API key：在 OpenAI Dashboard 中创建或找到你的 API key。


## 环境配置
1. 在你的目录中创建一个 `.env` 文件夹，并加入你的 `OPENAI_API_KEY` Key
2. 安装依赖

## 将 Codex CLI 初始化为 MCP Server
这里我们会在 Agents SDK 内部将 Codex CLI 作为一个 MCP Server 运行。我们提供了 `codex mcp` 的初始化参数。这个命令会将 Codex CLI 启动为一个 MCP server，并暴露出 MCP server 上可用的两个 Codex 工具：`codex()` 和 `codex-reply()`。这两个工具就是 Agents SDK 在需要调用 Codex 时所使用的底层工具。
- `codex()` 用于创建一个 conversation。
- `codex-reply()` 用于继续一个 conversation。

```python
import asyncio
from agents import Agent, Runner
from agents.mcp import MCPServerStdio

async def main() -> None:
    async with MCPServerStdio(
        name="Codex CLI",
        params={
            "command": "npx",
            "args": ["-y", "codex", "mcp-server"],
        },
        client_session_timeout_seconds=360000,
    ) as codex_mcp_server:
        print("Codex MCP server started.")
        # We will add more code here in the next section
        return
```

另外请注意，我们延长了 MCP Server 的超时时间，以便给 Codex CLI 足够的时间来执行并完成给定任务。

---

## 构建单 Agent 系统
先从一个简单示例开始，使用我们的 Codex MCP Server。我们定义两个 agent：
1. **Designer Agent**：为一个游戏进行头脑风暴，并创建一份简短说明。
2. **Developer Agent**：根据 Designer 的规格实现一个简单游戏。

```python
developer_agent = Agent(
    name="Game Developer",
    instructions=(
        "You are an expert in building simple games using basic html + css + javascript with no dependencies. "
        "Save your work in a file called index.html in the current directory."
        "Always call codex with \"approval-policy\": \"never\" and \"sandbox\": \"workspace-write\""
    ),
    mcp_servers=[codex_mcp_server],
)

designer_agent = Agent(
    name="Game Designer",
    instructions=(
        "You are an indie game connoisseur. Come up with an idea for a single page html + css + javascript game that a developer could build in about 50 lines of code. "
        "Format your request as a 3 sentence design brief for a game developer and call the Game Developer coder with your idea."
    ),
    model="gpt-5",
    handoffs=[developer_agent],
)

result = await Runner.run(designer_agent, "Implement a fun new game!")
```

注意，这里我们赋予了 Developer agent 向项目目录写文件的能力，并且无需向用户请求权限。

现在运行这段代码，你会看到生成了一个 `index.html` 文件。接着打开该文件，开始玩这个游戏吧！

下面是我的 agentic system 创建出的游戏截图。你的结果会有所不同！

| 示例玩法 | 游戏结束分数 |
| :---: | :---: |
| ![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/d49ea054-16aa-444e-9e60-40aa36161be3.png) | ![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-1/1c2227b4-3599-4404-b0a9-95e76c01d614.png) |

下面是可直接执行的完整代码。请注意，它可能需要几分钟才能运行完成。如果你看到生成了一个 `index.html` 文件，就说明它已经成功运行。你也可能会看到一些关于格式的 MCP events warnings，这些事件可以忽略。

---

## 编排多 Agent Workflow
对于更大的 workflow，我们会引入一个 agent 团队：
- **Project Manager**：拆解任务列表、创建需求，并协调工作。
- **Designer**：产出 UI/UX 规格。
- **Frontend Developer**：实现 UI/UX。
- **Backend Developer**：实现 API 和逻辑。
- **Tester**：根据验收标准验证输出。

在这个例子中，我们有意让 Project Manager agent 在每个下游专用 agent 之间强制执行 gating 逻辑。这可以确保在发生 handoff 之前，对应的 artifact 已经存在。这与现实世界中的企业 workflow 非常相似，例如 JIRA 任务编排、长链路发布流程，以及 QA 签核。

<div align="center">
  <img src="https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-1/412d3b8e-599c-4c47-a203-94ff7eb7dac6.png" alt="多 agent 编排与 gated handoff" />
  <br />
  <em>使用 Codex MCP 和带门控的 handoff 进行多 agent 编排，并生成各类 artifact。</em>
</div>


在这种结构中，我们的每个 agent 都承担一个专门的职责。Project Manager 总体负责跨所有其他 agent 进行协调，并确保整体任务完成。

## 定义 Codex CLI MCP Server
我们像单 agent 示例中那样设置 MCP Server，初始化 Codex CLI。

```python
async def main() -> None:
    async with MCPServerStdio(
        name="Codex CLI",
        params={
            "command": "npx",
            "args": ["-y", "codex", "mcp-server"],
        },
        client_session_timeout_seconds=360000,
    ) as codex_mcp_server:
        print("Codex MCP server started.")
        # We will add more code here in the next section
        return
```



## 定义每个专用 agent
下面我们定义每个专用 agent，并为其提供对 Codex MCP server 的访问权限。注意，我们还向每个 agent 传入了 `RECOMMMENDED_PROMPT_PREFIX`，它有助于系统在 agent 之间进行更优化的 handoff。

```python
# Downstream agents are defined first for clarity, then PM references them in handoffs.
designer_agent = Agent(
    name="Designer",
    instructions=(
        f"""{RECOMMENDED_PROMPT_PREFIX}"""
        "You are the Designer.\n"
        "Your only source of truth is AGENT_TASKS.md and REQUIREMENTS.md from the Project Manager.\n"
        "Do not assume anything that is not written there.\n\n"
        "You may use the internet for additional guidance or research."
        "Deliverables (write to /design):\n"
        "- design_spec.md – a single page describing the UI/UX layout, main screens, and key visual notes as requested in AGENT_TASKS.md.\n"
        "- wireframe.md – a simple text or ASCII wireframe if specified.\n\n"
        "Keep the output short and implementation-friendly.\n"
        "When complete, handoff to the Project Manager with transfer_to_project_manager."
        "When creating files, call Codex MCP with {\"approval-policy\":\"never\",\"sandbox\":\"workspace-write\"}."
    ),
    model="gpt-5",
    tools=[WebSearchTool()],
    mcp_servers=[codex_mcp_server],
    handoffs=[],
)

frontend_developer_agent = Agent(
    name="Frontend Developer",
    instructions=(
        f"""{RECOMMENDED_PROMPT_PREFIX}"""
        "You are the Frontend Developer.\n"
        "Read AGENT_TASKS.md and design_spec.md. Implement exactly what is described there.\n\n"
        "Deliverables (write to /frontend):\n"
        "- index.html – main page structure\n"
        "- styles.css or inline styles if specified\n"
        "- main.js or game.js if specified\n\n"
        "Follow the Designer’s DOM structure and any integration points given by the Project Manager.\n"
        "Do not add features or branding beyond the provided documents.\n\n"
        "When complete, handoff to the Project Manager with transfer_to_project_manager_agent."
        "When creating files, call Codex MCP with {\"approval-policy\":\"never\",\"sandbox\":\"workspace-write\"}."
    ),
    model="gpt-5",
    mcp_servers=[codex_mcp_server],
    handoffs=[],
)

backend_developer_agent = Agent(
    name="Backend Developer",
    instructions=(
        f"""{RECOMMENDED_PROMPT_PREFIX}"""
        "You are the Backend Developer.\n"
        "Read AGENT_TASKS.md and REQUIREMENTS.md. Implement the backend endpoints described there.\n\n"
        "Deliverables (write to /backend):\n"
        "- package.json – include a start script if requested\n"
        "- server.js – implement the API endpoints and logic exactly as specified\n\n"
        "Keep the code as simple and readable as possible. No external database.\n\n"
        "When complete, handoff to the Project Manager with transfer_to_project_manager_agent."
        "When creating files, call Codex MCP with {\"approval-policy\":\"never\",\"sandbox\":\"workspace-write\"}."
    ),
    model="gpt-5",
    mcp_servers=[codex_mcp_server],
    handoffs=[],
)

tester_agent = Agent(
    name="Tester",
    instructions=(
        f"""{RECOMMENDED_PROMPT_PREFIX}"""
        "You are the Tester.\n"
        "Read AGENT_TASKS.md and TEST.md. Verify that the outputs of the other roles meet the acceptance criteria.\n\n"
        "Deliverables (write to /tests):\n"
        "- TEST_PLAN.md – bullet list of manual checks or automated steps as requested\n"
        "- test.sh or a simple automated script if specified\n\n"
        "Keep it minimal and easy to run.\n\n"
        "When complete, handoff to the Project Manager with transfer_to_project_manager."
        "When creating files, call Codex MCP with {\"approval-policy\":\"never\",\"sandbox\":\"workspace-write\"}."
    ),
    model="gpt-5",
    mcp_servers=[codex_mcp_server],
    handoffs=[],
)
```



在每个角色完成其分配的任务之后，它会调用 `transfer_to_project_manager_agent`，让 Project Manager 确认所需文件已经存在（或者请求修复），然后再解锁下一个团队。

## 定义 Project Manager Agent
Project Manager 是唯一接收初始 prompt 的 agent，它会在项目目录中创建规划文档，并在每一次 transfer 前执行 gatekeeping 逻辑。

```python
project_manager_agent = Agent(
name="Project Manager",
instructions=(
    f"""{RECOMMENDED_PROMPT_PREFIX}"""
    """
    You are the Project Manager.

    Objective:
    Convert the input task list into three project-root files the team will execute against.

    Deliverables (write in project root):
    - REQUIREMENTS.md: concise summary of product goals, target users, key features, and constraints.
    - TEST.md: tasks with [Owner] tags (Designer, Frontend, Backend, Tester) and clear acceptance criteria.
    - AGENT_TASKS.md: one section per role containing:
        - Project name
        - Required deliverables (exact file names and purpose)
        - Key technical notes and constraints

    Process:
    - Resolve ambiguities with minimal, reasonable assumptions. Be specific so each role can act without guessing.
    - Create files using Codex MCP with {"approval-policy":"never","sandbox":"workspace-write"}.
    - Do not create folders. Only create REQUIREMENTS.md, TEST.md, AGENT_TASKS.md.

    Handoffs (gated by required files):
    1) After the three files above are created, hand off to the Designer with transfer_to_designer_agent and include REQUIREMENTS.md, and AGENT_TASKS.md.
    2) Wait for the Designer to produce /design/design_spec.md. Verify that file exists before proceeding.
    3) When design_spec.md exists, hand off in parallel to both:
        - Frontend Developer with transfer_to_frontend_developer_agent (provide design_spec.md, REQUIREMENTS.md, AGENT_TASKS.md).
        - Backend Developer with transfer_to_backend_developer_agent (provide REQUIREMENTS.md, AGENT_TASKS.md).
    4) Wait for Frontend to produce /frontend/index.html and Backend to produce /backend/server.js. Verify both files exist.
    5) When both exist, hand off to the Tester with transfer_to_tester_agent and provide all prior artifacts and outputs.
    6) Do not advance to the next handoff until the required files for that step are present. If something is missing, request the owning agent to supply it and re-check.

    PM Responsibilities:
    - Coordinate all roles, track file completion, and enforce the above gating checks.
    - Do NOT respond with status updates. Just handoff to the next agent until the project is complete.
    """
),
model="gpt-5",
model_settings=ModelSettings(
    reasoning=Reasoning(effort="medium")
),
handoffs=[designer_agent, frontend_developer_agent, backend_developer_agent, tester_agent],
mcp_servers=[codex_mcp_server],
)
```

构建好 Project Manager 之后，脚本会把每个专职角色的 handoff 再指回 Project Manager。这样可以确保在进入下一步之前，交付物都会先返回并接受验证。

```python
designer_agent.handoffs = [project_manager_agent]
frontend_developer_agent.handoffs = [project_manager_agent]
backend_developer_agent.handoffs = [project_manager_agent]
tester_agent.handoffs = [project_manager_agent]
```
## 加入你的任务列表
这就是 Project Manager 会进一步细化为整个系统具体需求和任务的任务内容。

```python
task_list = """
Goal: Build a tiny browser game to showcase a multi-agent workflow.

High-level requirements:
- Single-screen game called "Bug Busters".
- Player clicks a moving bug to earn points.
- Game ends after 20 seconds and shows final score.
- Optional: submit score to a simple backend and display a top-10 leaderboard.

Roles:
- Designer: create a one-page UI/UX spec and basic wireframe.
- Frontend Developer: implement the page and game logic.
- Backend Developer: implement a minimal API (GET /health, GET/POST /scores).
- Tester: write a quick test plan and a simple script to verify core routes.

Constraints:
- No external database—memory storage is fine.
- Keep everything readable for beginners; no frameworks required.
- All outputs should be small files saved in clearly named folders.
"""
```

接下来，运行你的系统，稍等片刻，你就会看到这些 agent 开始工作，并在几分钟内创建出一个游戏！下面已经包含了可直接执行的完整代码。执行完成后，你会看到生成如下文件目录。请注意，这个多 agent 编排通常需要大约 11 分钟才能全部完成。

```markdown
root_directory/
├── AGENT_TASKS.md
├── REQUIREMENTS.md
├── backend
│   ├── package.json
│   └── server.js
├── design
│   ├── design_spec.md
│   └── wireframe.md
├── frontend
│   ├── game.js
│   ├── index.html
│   └── styles.css
└── TEST.md
```

使用 `node server.js` 启动你的后端服务，然后打开你的 `index.html` 文件来玩这个游戏。

---

## 使用 Traces 跟踪 agent 行为
随着 agentic system 的复杂度不断增长，观察这些 agent 如何交互就变得很重要。我们可以通过 Traces dashboard 来做到这一点，它会记录：
- agent 之间的 prompts、tool calls 和 handoffs。
- MCP Server 调用、Codex CLI 调用、执行时间和文件写入。
- 错误与警告。

下面来看一下上面这个 agent 团队的 agent trace。

<div align="center">
  <img src="https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/baae2c8a-5387-4578-a878-83ec1211ac12.png" alt="Agent trace 总览" />
</div>

在这个 Trace 中，我们可以确认每一次 agent handoff 都是由我们的 Project Manager Agent 进行统一调度的，它会在 handoff 给下一个 agent 之前确认特定 artifact 是否存在。另外，我们还能看到 Codex MCP Server 的具体调用，并通过调用 Responses API 生成每一项输出。时间线条会突出显示执行时长，因此很容易识别长时间运行的步骤，并理解控制权如何在不同 agent 之间传递。

你甚至可以点进每一条 trace，查看 prompt、tool calls 和其他元数据的具体细节。随着时间推移，你可以利用这些信息进一步调优、优化并跟踪你的 agentic system 性能。

<div align="center">
  <img src="https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/b5ced19f-588b-4d15-b381-413fab2a1d22.png" alt="Trace 详情" />
</div>

---

## 回顾本文完成的内容
在本指南中，我们完整演示了如何使用 Codex CLI 和 Agents SDK 构建一致且可扩展的 workflow。具体包括：

- **Codex MCP Server 设置**：如何将 Codex CLI 初始化为一个 MCP server，并把它作为工具提供给 agent 进行交互。
- **单 Agent 示例**：一个由 Designer Agent 和 Developer Agent 组成的简单 workflow，其中 Codex 以确定性的方式执行具备明确范围的任务，最终生成一个可玩的游戏。
- **多 Agent 编排**：扩展为一个包含 Project Manager、Designer、Frontend Developer、Backend Developer 和 Tester 的更大 workflow，以模拟复杂任务编排和签核流程。
- **Traces 与可观测性**：使用内置的 Traces 捕获 prompts、tool calls、handoffs、执行时间和 artifacts，从而为调试、评估和后续优化提供完整的 agent 行为可见性。

---

## 后续方向：如何应用这些经验
现在你已经看到了 Codex MCP 和 Agents SDK 的实际运行方式，下面介绍如何将这些概念应用到真实项目中并获得价值：

### 1. 扩展到真实世界的发布流程
- 将同样的多 agent 编排方式应用到大型代码重构中（例如 500+ 文件、框架迁移）。
- 利用 Codex MCP 的确定性执行能力，支撑长时间运行且可审计的发布流程，并具备可追踪的进度。

### 2. 在不失控的前提下加快交付
- 组织由专用 agent 组成的团队并行推进开发，同时保留用于验证 artifact 的 gating 逻辑。
- 缩短新功能开发、测试或代码库现代化改造的周转时间。

### 3. 扩展并连接到你的开发 workflow
- 通过 webhooks 将基于 MCP 的 agent 与 Jira、GitHub 或 CI/CD pipelines 连接起来，形成自动化、可重复的开发周期。
- 在多 agent 服务编排中利用 Codex MCP：不仅用于 codegen，也可用于文档、QA 和部署。
