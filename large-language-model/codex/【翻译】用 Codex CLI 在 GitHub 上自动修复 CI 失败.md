# 【翻译】用 Codex CLI 在 GitHub 上自动修复 CI 失败

> 来源：OpenAI Cookbook，原文链接：https://github.com/openai/openai-cookbook/blob/main/examples/codex/Autofix-github-actions.ipynb
> 说明：本文为中文翻译，保留原文中的代码、命令、配置片段和 mdnice 图片链接。

## 本文 cookbook 的目的

本文 cookbook 展示了如何将 OpenAI Codex CLI 嵌入到你的 CI/CD 流水线中，这样当构建或测试失败时，codex 就会自动生成并提交修复方案。下面给出的是一个 Node 项目的示例，其中 CI 运行在 GitHub Actions 中。

## 端到端流程

下面是我们将要实现的流水线流程：


![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-2/97749434-5fe8-477f-b4e5-c8c09597be16.png)

## 前置条件

- 你需要一个配置了 Actions workflow 的 GitHub Repo

- 你需要在 GitHub 设置中 https://github.com/{org-name}/{repo-name}/settings/secrets/actions 下创建 `OPENAI_API_KEY` 环境变量。你也可以在组织级别设置它（用于在多个 repo 之间共享 secret）

- Codex 依赖 Python 才能使用 `codex login`

- 你需要勾选相关设置，以允许 Actions 在你的 repo 中创建 PR，同时也要在你的组织中开启此设置：


![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-1/49c74bbf-092a-4df1-85c1-bd72bee32f37.png)

## 第 1 步：把 GitHub Action 加入你的 CI 流水线

下面的 YAML 展示了一个 GitHub Action：当 CI 失败时它会自动触发，安装 Codex，执行 `codex exec`，然后在失败分支上创建一个包含修复内容的 PR。请将 `"CI"` 替换为你想监控的 workflow 名称。

```yaml
name: Codex Auto-Fix on Failure

on:
  workflow_run:
    # Trigger this job after any run of the primary CI workflow completes
    workflows: ["CI"]
    types: [completed]

permissions:
  contents: write
  pull-requests: write

jobs:
  auto-fix:
    # Only run when the referenced workflow concluded with a failure
    if: ${{ github.event.workflow_run.conclusion == 'failure' }}
    runs-on: ubuntu-latest
    env:
      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      FAILED_WORKFLOW_NAME: ${{ github.event.workflow_run.name }}
      FAILED_RUN_URL: ${{ github.event.workflow_run.html_url }}
      FAILED_HEAD_BRANCH: ${{ github.event.workflow_run.head_branch }}
      FAILED_HEAD_SHA: ${{ github.event.workflow_run.head_sha }}
    steps:
      - name: Check OpenAI API Key Set
        run: |
          if [ -z "$OPENAI_API_KEY" ]; then
            echo "OPENAI_API_KEY secret is not set. Skipping auto-fix." >&2
            exit 1
          fi
      - name: Checkout Failing Ref
        uses: actions/checkout@v4
        with:
          ref: ${{ env.FAILED_HEAD_SHA }}
          fetch-depth: 0

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'

      - name: Install dependencies
        run: |
          if [ -f package-lock.json ]; then npm ci; else npm i; fi
      - name: Run Codex
        uses: openai/codex-action@main
        id: codex
        with:
          openai_api_key: ${{ secrets.OPENAI_API_KEY }}
          prompt: "You are working in a Node.js monorepo with Jest tests and GitHub Actions. Read the repository, run the test suite, identify the minimal change needed to make all tests pass, implement only that change, and stop. Do not refactor unrelated code or files. Keep changes small and surgical."
          codex_args: '["--config","sandbox_mode=\"workspace-write\""]'

      - name: Verify tests
        run: npm test --silent

      - name: Create pull request with fixes
        if: success()
        uses: peter-evans/create-pull-request@v6
        with:
          commit-message: "fix(ci): auto-fix failing tests via Codex"
          branch: codex/auto-fix-${{ github.event.workflow_run.run_id }}
          base: ${{ env.FAILED_HEAD_BRANCH }}
          title: "Auto-fix failing CI via Codex"
          body: |
            Codex automatically generated this PR in response to a CI failure on workflow `${{ env.FAILED_WORKFLOW_NAME }}`.
            Failed run: ${{ env.FAILED_RUN_URL }}
            Head branch: `${{ env.FAILED_HEAD_BRANCH }}`
            This PR contains minimal changes intended solely to make the CI pass.
```

## 第 2 步：Actions Workflow 被触发

你可以进入 Repo 下的 Actions 标签页，查看 Actions workflow 中失败的 job。

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-1/0096795a-2ca1-4c2e-a043-077a8de17502.png)

失败的 workflow 完成后，Codex workflow 应该会被触发。


![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-2/88dc604e-ead5-4d76-bf44-8d0d4ac26059.png)

## 第 3 步：确认 Codex 已创建可供审查的 PR
在 Codex workflow 执行完成后，它应该会从功能分支 `codex/auto-fix` 发起一个 pull request。检查一遍确认没有问题后，再将其合并。

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-1/19b298c7-a4de-4210-bb13-00fad96d05a5.png)

## 总结

这套自动化流程将 OpenAI Codex CLI 与 GitHub Actions 无缝集成起来，能够在 CI 运行失败时自动提出修复方案。

借助 Codex，你可以减少人工介入，加快代码审查速度，并让主分支保持健康。这个 workflow 能确保测试失败被快速且高效地处理，从而让开发者把精力放在更高价值的任务上。你可以在[这里](https://github.com/openai/codex/)进一步了解 codex-cli 及其能力。
