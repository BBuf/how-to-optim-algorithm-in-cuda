# 【翻译】使用 codex-cli 自动化 Jira 与 GitHub 工作流
> 来源：OpenAI Cookbook，原文链接：https://github.com/openai/openai-cookbook/blob/main/examples/codex/jira-github.ipynb
> 说明：本文为中文翻译，保留原文中的代码、命令、配置片段和 mdnice 图片链接。

## 本 cookbook 的目标

这篇 cookbook 提供了一套实用的分步方法，用于自动化 Jira 与 GitHub 之间的工作流。通过给 Jira issue 打上标签，你就可以触发一条端到端流程：创建 **GitHub pull request**、同步更新两个系统，并简化代码评审过程，整个过程几乎不需要人工介入。整个自动化由运行在 GitHub Action 中的 [`codex-cli`](https://github.com/openai/openai-codex) agent 驱动。

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-2/7dc1359c-5edb-4700-8eb2-410103f89cd1.png)

流程如下：

1. 给 Jira issue 打标签
2. Jira Automation 调用 GitHub Action
3. 该 action 启动 `codex-cli` 来实现变更
4. 打开一个 PR
5. Jira 被执行 transition 并添加注释，形成一个整洁的零点击闭环。这包括修改 ticket 状态、添加 PR 链接，以及在 ticket 中写入更新评论。

## 前置条件

* Jira：项目管理员权限，以及创建 automation rule 的能力
* GitHub：写权限、添加仓库 secrets 的权限，以及受保护的 `main` branch
* 作为仓库 secrets 放置的 API keys 和 secrets：
  * `OPENAI_API_KEY`：供 `codex-cli` 使用的 OpenAI key
  * `JIRA_BASE_URL`, `JIRA_EMAIL`, `JIRA_API_TOKEN`：供 action 发起 REST 调用
* 本地已安装 `codex-cli`（`pnpm add -g @openai/codex`），便于临时测试
* 一个包含 `.github/workflows/` 文件夹的仓库

## 创建 Jira Automation Rule

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-1/0ed7ec91-a1eb-4cd7-ad2b-be9e44265480.png)

这条 rule 的第一步是监听某个 issue 标签的变更。这样可以确保只有在标签被添加或修改时才触发自动化，而不需要处理该 issue 的每一次更新。

接下来，我们检查更新后的标签中是否包含某个特定关键字。在本例中我们使用的是 `aswe`。它相当于一个过滤器，只有被显式标记为需要自动化处理的 issue 才会继续执行，从而避免无关更新带来的噪音。

如果条件满足，我们就向 GitHub 的 `workflow_dispatch` endpoint 发送一个 `POST` 请求。这会以相关 issue 上下文启动一个 GitHub Actions workflow。我们会传入 issue key、summary，以及清理过的 description 版本，并对引号和换行进行转义，以确保 payload 能被 YAML/JSON 正确解析。JIRA 中还提供了[额外字段](https://support.atlassian.com/cloud-automation/docs/jira-smart-values-issues/)可作为变量使用，从而在 codex agent 执行期间提供更多上下文。

这种设置让团队可以严格控制哪些 Jira issue 会触发自动化，并确保 GitHub 接收到结构清晰、格式干净的元数据用于后续处理。我们也可以配置多个标签，每个标签触发不同的 GitHub Action。比如，一个标签可以启动快速修复 bug 的 workflow，而另一个标签则可以开始代码重构或 API stub 生成工作。

## 添加 GitHub Action

GitHub Actions 允许你通过 YAML 文件在 GitHub 仓库内定义自动化 workflow。这些 workflow 会声明一系列需要执行的 jobs 和 steps。当它们被手动触发或通过 POST 请求触发时，GitHub 会自动配置所需环境，并运行定义好的 workflow 步骤。

为了处理来自 JIRA 的 `POST` 请求，我们将在仓库的 `.github/workflows/` 目录下创建一个类似下面的 Github action YAML：

```yaml
name: Codex Automated PR
on:
  workflow_dispatch:
    inputs:
      issue_key:
        description: 'JIRA issue key (e.g., PROJ-123)'
        required: true
      issue_summary:
        description: 'Brief summary of the issue'
        required: true
      issue_description:
        description: 'Detailed issue description'
        required: true

permissions:
  contents: write           # allow the action to push code & open the PR
  pull-requests: write      # allow the action to create and update PRs

jobs:
  codex_auto_pr:
    runs-on: ubuntu-latest

    steps:
    # 0 – Checkout repository
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0       # full history → lets Codex run tests / git blame if needed

    # 1 – Set up Node.js and Codex
    - uses: actions/setup-node@v4
      with:
        node-version: 22
    - run: pnpm add -g @openai/codex

    # 2 – Export / clean inputs (available via $GITHUB_ENV)
    - id: vars
      run: |
        echo "ISSUE_KEY=${{ github.event.inputs.issue_key }}"        >> $GITHUB_ENV
        echo "TITLE=${{ github.event.inputs.issue_summary }}"        >> $GITHUB_ENV
        echo "RAW_DESC=${{ github.event.inputs.issue_description }}" >> $GITHUB_ENV
        DESC_CLEANED=$(echo "${{ github.event.inputs.issue_description }}" | tr '\n' ' ' | sed 's/"/'\''/g')
        echo "DESC=$DESC_CLEANED"                                    >> $GITHUB_ENV
        echo "BRANCH=codex/${{ github.event.inputs.issue_key }}"     >> $GITHUB_ENV

    # 3 – Transition Jira issue to "In Progress"
    - name: Jira – Transition to In Progress
      env:
        ISSUE_KEY:      ${{ env.ISSUE_KEY }}
        JIRA_BASE_URL:  ${{ secrets.JIRA_BASE_URL }}
        JIRA_EMAIL:     ${{ secrets.JIRA_EMAIL }}
        JIRA_API_TOKEN: ${{ secrets.JIRA_API_TOKEN }}
      run: |
        curl -sS -X POST \
          --url   "$JIRA_BASE_URL/rest/api/3/issue/$ISSUE_KEY/transitions" \
          --user  "$JIRA_EMAIL:$JIRA_API_TOKEN" \
          --header 'Content-Type: application/json' \
          --data  '{"transition":{"id":"21"}}'
          # 21 is the transition ID for changing the ticket status to In Progress. Learn more here: https://developer.atlassian.com/cloud/jira/platform/rest/v3/api-group-issues/#api-rest-api-3-issue-issueidorkey-transitions-get

    # 4 – Set Git author for CI commits
    - run: |
        git config user.email "github-actions[bot]@users.noreply.github.com"
        git config user.name  "github-actions[bot]"

    # 5 – Let Codex implement & commit (no push yet)
    - name: Codex implement & commit
      env:
        OPENAI_API_KEY:  ${{ secrets.OPENAI_API_KEY }}
        CODEX_QUIET_MODE: "1"          # suppress chatty logs
      run: |
        set -e
        codex --approval-mode full-auto --no-terminal --quiet \
              "Implement JIRA ticket $ISSUE_KEY: $TITLE. $DESC"

        git add -A
        git commit -m "feat($ISSUE_KEY): $TITLE"

    # 6 – Open (and push) the PR in one go
    - id: cpr
      uses: peter-evans/create-pull-request@v6
      with:
        token:  ${{ secrets.GITHUB_TOKEN }}
        base:   main
        branch: ${{ env.BRANCH }}
        title:  "${{ env.TITLE }} (${{ env.ISSUE_KEY }})"
        body: |
          Auto-generated by Codex for JIRA **${{ env.ISSUE_KEY }}**.
          ---
          ${{ env.DESC }}

    # 7 – Transition Jira to "In Review" & drop the PR link
    - name: Jira – Transition to In Review & Comment PR link
      env:
        ISSUE_KEY:      ${{ env.ISSUE_KEY }}
        JIRA_BASE_URL:  ${{ secrets.JIRA_BASE_URL }}
        JIRA_EMAIL:     ${{ secrets.JIRA_EMAIL }}
        JIRA_API_TOKEN: ${{ secrets.JIRA_API_TOKEN }}
        PR_URL:         ${{ steps.cpr.outputs.pull-request-url }}
      run: |
        # Status transition
        curl -sS -X POST \
          --url   "$JIRA_BASE_URL/rest/api/3/issue/$ISSUE_KEY/transitions" \
          --user  "$JIRA_EMAIL:$JIRA_API_TOKEN" \
          --header 'Content-Type: application/json' \
          --data  '{"transition":{"id":"31"}}'
          # 31 is the Transition ID for changing the ticket status to In Review. Learn more here: https://developer.atlassian.com/cloud/jira/platform/rest/v3/api-group-issues/#api-rest-api-3-issue-issueidorkey-transitions-get

        # Comment with PR link
        curl -sS -X POST \
          --url   "$JIRA_BASE_URL/rest/api/3/issue/$ISSUE_KEY/comment" \
          --user  "$JIRA_EMAIL:$JIRA_API_TOKEN" \
          --header 'Content-Type: application/json' \
          --data  "{\"body\":{\"type\":\"doc\",\"version\":1,\"content\":[{\"type\":\"paragraph\",\"content\":[{\"type\":\"text\",\"text\":\"PR created: $PR_URL\"}]}]}}"
```

## Workflow 中的关键步骤

1. **Codex Implementation & Commit**（步骤 5）
   - 使用 OpenAI API 实现 JIRA ticket 的需求
   - 以 full-auto 模式运行 codex CLI，无需终端交互
   - 用标准化的 commit message 提交全部变更

2. **Create Pull Request**（步骤 6）
   - 使用 `peter-evans/create-pull-request` action
   - 针对 main branch 创建 PR
   - 根据 JIRA ticket 信息设置 PR 标题和描述
   - 返回 PR URL 供后续使用

3. **JIRA Updates**（步骤 7）
   - 通过 JIRA API 将 ticket transition 到 `"In Review"` 状态
   - 在 JIRA ticket 中发布包含 PR URL 的评论
   - 使用 curl 命令与 JIRA REST API 交互

## 给 Issue 打标签

把特殊标签 `aswe` 添加到任意 bug/feature ticket 上：

1. **创建时**：在点击 *Create* 之前，在 “Labels” 字段中添加它
2. **已有 issue**：将鼠标悬停在标签区域 → 点击铅笔图标 → 输入 `aswe`

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/d515795b-a96b-48ac-be5d-a8830495485c.png)

## 端到端流程

1. Jira 标签被添加 → Automation 被触发
2. `workflow_dispatch` 被触发；action 在 GitHub 上启动
3. `codex-cli` 修改代码库并提交
4. 在生成的 branch 上打开 PR
5. Jira 被移动到 **In Review**，并发布一条带有 PR URL 的评论
6. Reviewer 会按照你现有的 branch protection 设置收到通知

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-1/0bce691d-970f-45c5-b0d6-565cd1737294.png)
![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/b327f08c-5e29-423f-9e7a-105891ff01e4.png)

## 审查并合并 PR

你可以打开发布在 JIRA ticket 中的 PR 链接，检查一切是否正常，然后将其合并。如果你启用了 branch protection 和 Smart Commits 集成，那么当 pull request 被合并时，对应的 Jira ticket 会被自动关闭。

## 结论

这套自动化通过在 Jira 与 GitHub 之间建立无缝集成，简化了你的开发工作流：

* **自动状态跟踪** - ticket 会沿着你的工作流自动推进，无需手动更新
* **更好的开发者体验** - 把注意力放在代码质量评审上，而不是编写样板代码
* **减少交接摩擦** - 只要 ticket 被打上标签，PR 就已经准备好进入 review

`codex-cli` 是一个强大的 AI 编程助手，可以自动化重复性的编程任务。你可以在[这里](https://github.com/openai/codex/)了解更多。
