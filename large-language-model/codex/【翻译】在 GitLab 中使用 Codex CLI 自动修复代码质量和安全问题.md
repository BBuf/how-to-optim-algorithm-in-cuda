# 【翻译】在 GitLab 中使用 Codex CLI 自动修复代码质量和安全问题
> 来源：OpenAI Cookbook，原文链接：https://github.com/openai/openai-cookbook/blob/main/examples/codex/secure_quality_gitlab.md
> 说明：本文为中文翻译，保留原文中的代码、命令、配置片段和 mdnice 图片链接。

## 引言

在部署生产代码时，大多数团队都会依赖 CI/CD pipeline 在合并前验证变更。审阅者通常会查看单元测试结果、漏洞扫描以及代码质量报告。传统上，这些结果由基于规则的引擎生成，它们能够捕获已知问题，但往往会遗漏依赖上下文或更高阶的缺陷，同时还会留下大量噪声结果，让开发者难以确定优先级或采取行动。

有了 LLM，你可以为这个流程增加一层新的智能能力：推理代码质量问题，并解释安全发现。通过在 GitLab pipeline 中接入 **OpenAI 的 Codex CLI**，团队可以获得超越静态规则的洞察：

* **Code Quality** → 生成符合 GitLab 规范的 CodeClimate JSON 报告，在 merge request 中直接展示带上下文的问题。

* **Security** → 对现有 SAST 结果做后处理，合并重复项，按可利用性排序问题，并提供清晰、可执行的修复步骤。

本指南展示了如何在 GitLab pipeline 中为这两类场景集成 Codex CLI，在提供结构化、机器可读报告的同时，也给出可执行、便于人工理解的指导。

## 什么是 Codex CLI？

Codex CLI 是一个开源命令行工具，用于将 OpenAI 的推理模型引入你的开发工作流。有关安装、用法和完整文档，请参阅官方仓库：[github.com/openai/codex](https://github.com/openai/codex?utm_source=chatgpt.com)。

在这篇 cookbook 中，我们会在一次性的 GitLab runner 中使用 **Full Auto mode** 来生成符合标准的 JSON 报告。

### 前置条件

要跟着本文操作，你需要准备：

* 一个 GitLab 账号和项目
* 一个具有 **internet access** 的 GitLab runner（我们已在 Linux runner 上测试，配置为 2 vCPU、8GB 内存、30GB 存储）
* Runner 必须能够连接到 `api.openai.com`
* 一个 **OpenAI API key**（`OPENAI_API_KEY`）
* 在 **Settings → CI/CD → Variables** 下配置好的 GitLab CI/CD 变量

## 示例 #1 - 使用 Codex CLI 生成 Code Quality Report

### 背景

这个仓库是一个故意包含漏洞的 Node.js Express 演示应用，基于 [GitLab 的 node express template](https://gitlab.com/gitlab-org/project-templates/express/-/tree/main) 构建，用于展示 GitLab CI/CD 中的静态应用安全测试（SAST）和代码质量扫描。

代码中包含一些常见陷阱，例如命令注入、路径遍历、不安全的 `eval`、正则 DoS、弱加密（MD5）以及硬编码密钥。它被用于验证基于 Codex 的分析器能否生成 GitLab 原生报告（Code Quality 和 SAST），并在 merge request 中直接渲染。

CI 运行在 GitLab SaaS runner 上，使用 `node:24` 镜像，并额外安装少量工具（`jq`、`curl`、`ca-certificates`、`ajv-cli`）。这些 job 通过 `set -euo pipefail`、schema 校验以及严格的 JSON 标记进行加固，从而即使 Codex 输出有所波动，解析过程依然可靠。

这种 pipeline 模式，即 prompt、JSON 标记提取、schema 校验，也可以适配到其他技术栈，不过 prompt 的措辞和 schema 规则可能需要调整。由于 Codex 运行在沙箱中，一些系统命令（例如 `awk` 或 `nl`）可能会受到限制。

你的团队希望确保 **code quality 检查会在每次合并前自动运行**。为了让发现的问题直接显示在 GitLab 的 merge request widget 中，报告必须遵循 **CodeClimate JSON format**。[参考：GitLab 文档](https://docs.gitlab.com/ci/testing/code_quality/#import-code-quality-results-from-a-cicd-job)

### Code Quality CI/CD Job 示例

下面是一个可直接接入的 GitLab CI job，使用 **Codex CLI** 生成符合要求的 JSON 文件：
```yaml
stages: [codex]

default:
  image: node:24
  variables:
    CODEX_QA_PATH: "gl-code-quality-report.json"
    CODEX_RAW_LOG: "artifacts/codex-raw.log"
    # Strict prompt: must output a single JSON array (or []), no prose/markdown/placeholders.
    CODEX_PROMPT: |
      Review this repository and output a GitLab Code Quality report in CodeClimate JSON format.
      RULES (must follow exactly):
      - OUTPUT MUST BE A SINGLE JSON ARRAY. Example: [] or [ {...}, {...} ].
      - If you find no issues, OUTPUT EXACTLY: []
      - DO NOT print any prose, backticks, code fences, markdown, or placeholders.
      - DO NOT write any files. PRINT ONLY between these two lines:
        === BEGIN_CODE_QUALITY_JSON ===
        <JSON ARRAY HERE>
        === END_CODE_QUALITY_JSON ===
      Each issue object MUST include these fields:
        "description": String,
        "check_name": String (short rule name),
        "fingerprint": String (stable across runs for same issue),
        "severity": "info"|"minor"|"major"|"critical"|"blocker",
        "location": { "path": "<repo-relative-file>", "lines": { "begin": <line> } }
      Requirements:
      - Use repo-relative paths from the current checkout (no "./", no absolute paths).
      - Use only files that actually exist in this repo.
      - No trailing commas. No comments. No BOM.
      - Prefer concrete, de-duplicated findings. If uncertain, omit the finding.

codex_review:
  stage: codex
  # Skip on forked MRs (no secrets available). Run only if OPENAI_API_KEY exists.
  rules:
    - if: '$CI_PIPELINE_SOURCE == "merge_request_event" && $CI_MERGE_REQUEST_SOURCE_PROJECT_ID != $CI_PROJECT_ID'
      when: never
    - if: '$OPENAI_API_KEY'
      when: on_success
    - when: never

  script:
    - set -euo pipefail
    - echo "PWD=$(pwd)  CI_PROJECT_DIR=${CI_PROJECT_DIR}"
    # Ensure artifacts always exist so upload never warns, even on early failure
    - mkdir -p artifacts
    - ': > ${CODEX_RAW_LOG}'
    - ': > ${CODEX_QA_PATH}'
    # Minimal deps + Codex CLI
    - apt-get update && apt-get install -y --no-install-recommends curl ca-certificates git lsb-release
    - npm -g i @openai/codex@latest
    - codex --version && git --version
    # Build a real-file allowlist to guide Codex to valid paths/lines
    - FILE_LIST="$(git ls-files | sed 's/^/- /')"
    - |
      export CODEX_PROMPT="${CODEX_PROMPT}
      Only report issues in the following existing files (exactly as listed):
      ${FILE_LIST}"
    # Run Codex; allow non-zero exit but capture output for extraction
    - |
      set +o pipefail
      script -q -c 'codex exec --full-auto "$CODEX_PROMPT"' | tee "${CODEX_RAW_LOG}" >/dev/null
      CODEX_RC=${PIPESTATUS[0]}
      set -o pipefail
      echo "Codex exit code: ${CODEX_RC}"
    # Strip ANSI + \r, extract JSON between markers to a temp file; validate or fallback to []
    - |
      TMP_OUT="$(mktemp)"
      sed -E 's/\x1B\[[0-9;]*[A-Za-z]//g' "${CODEX_RAW_LOG}" \
        | tr -d '\r' \
        | awk '
            /^\s*=== BEGIN_CODE_QUALITY_JSON ===\s*$/ {grab=1; next}
            /^\s*=== END_CODE_QUALITY_JSON ===\s*$/   {grab=0}
            grab
          ' > "${TMP_OUT}"
      # If extracted content is empty/invalid or still contains placeholders, replace with []
      if ! node -e 'const f=process.argv[1]; const s=require("fs").readFileSync(f,"utf8").trim(); if(!s || /(<JSON ARRAY HERE>|BEGIN_CODE_QUALITY_JSON|END_CODE_QUALITY_JSON)/.test(s)) process.exit(2); JSON.parse(s);' "${TMP_OUT}"; then
        echo "WARNING: Extracted content empty/invalid; writing empty [] report."
        echo "[]" > "${TMP_OUT}"
      fi
      mv -f "${TMP_OUT}" "${CODEX_QA_PATH}"
      # Soft warning if Codex returned non-zero but we still produced a report
      if [ "${CODEX_RC}" -ne 0 ]; then
        echo "WARNING: Codex exited with code ${CODEX_RC}. Proceeding with extracted report." >&2
      fi

  artifacts:
    when: always
    reports:
      codequality: gl-code-quality-report.json
    paths:
      - artifacts/codex-raw.log
    expire_in: 14 days
```

1. 安装 Codex CLI（`npm -g i @openai/codex@latest`）
2. 使用 `git ls-files` 构建文件 allowlist
3. 以严格的“仅输出 JSON” prompt 在 **full-auto mode** 下运行 Codex
4. 提取标记之间的有效 JSON，验证其合法性，并在无效时回退为 `[]`
5. 将 artifact 发布到 GitLab，使结果以内联方式显示在 merge request 中

生成的 artifact 可以在 pipeline 页面下载

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/a9dcd2b6-be07-431e-8c4d-44ce4f428eba.png)

或者在以 feature 分支向 master 分支发起 merge 时，

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-2/7be882e6-d3e1-419c-8984-d7b626bbe871.png)

通过将 Codex CLI 嵌入 GitLab CI/CD pipeline，你可以 **把 code quality 检查提升到超越静态规则的层次**。这样一来，你不只是捕获语法错误或风格违规，还能启用基于推理的分析，在上下文中指出潜在问题。

这种方式有几个好处：

* **一致性**：每个 merge request 都由同一套推理流程进行审查
* **上下文感知**：LLM 能标记出基于规则的扫描器容易漏掉的细微问题
* **赋能开发者**：反馈即时、可见且可执行

随着团队采纳这种工作流，由 LLM 驱动的质量检查可以补充传统的 lint 和漏洞扫描，帮助确保发布到生产环境的代码既稳健又易于维护。

## 示例 #2 – 使用 Codex CLI 做 Security Remediation

### 背景

在这个示例中，我们使用 [OWASP Juice Shop](https://github.com/juice-shop/juice-shop?utm_source=chatgpt.com) 进行测试，它是一个故意包含漏洞的 Node.js Express 应用。它包含诸如注入、不安全的 `eval`、弱加密和硬编码密钥等常见缺陷，非常适合用来验证基于 Codex 的分析能力。

你的团队希望确保每当代码发生变更时，pipeline 都会在合并前自动检查安全漏洞。这通常已经由静态分析器和特定语言的扫描器完成，它们会生成符合 GitLab SAST JSON schema 的报告。然而，原始输出往往僵硬、噪声较多，而且通常无法为审阅者提供清晰的下一步行动。

通过在 pipeline 中加入 Codex CLI，你可以把 [GitLab SAST scanners](https://docs.gitlab.com/user/application_security/sast/)（或其他扫描器输出）生成的扫描结果转换为 **可执行的 remediation guidance**，甚至生成 **可直接应用的 git patch**：

### 第 1 步：生成建议

* Codex 读取 `gl-sast-report.json`。
* 合并重复发现。
* 按可利用性排序（例如用户输入 → 危险 sink）。
* 生成简洁的 `security_priority.md`，其中包含前 5 条行动建议和详细 remediation 说明。

#### Security Recommendations CI/CD Job 示例

**要求**：此 job 预期上游 SAST job 已经生成 `gl-sast-report.json`。Codex 会读取它并生成 `security_priority.md` 供审阅者查看。

```yaml
stages:
 - codex
 - remediation

default:
 image: node:24

variables:
 CODEX_SAST_PATH: "gl-sast-report.json"
 CODEX_SECURITY_MD: "security_priority.md"
 CODEX_RAW_LOG: "artifacts/codex-sast-raw.log"

 # --- Recommendations prompt (reads SAST → writes Markdown) ---
 CODEX_PROMPT: |
   You are a security triage assistant analyzing GitLab SAST output.
   The SAST JSON is located at: ${CODEX_SAST_PATH}

   GOAL:
   - Read and parse ${CODEX_SAST_PATH}.
   - Consolidate duplicate or overlapping findings (e.g., same CWE + same sink/function, same file/line ranges, or same data flow root cause).
   - Rank findings by realistic exploitability and business risk, not just library presence.
     * Prioritize issues that:
       - Are reachable from exposed entry points (HTTP handlers, controllers, public APIs, CLI args).
       - Involve user-controlled inputs reaching dangerous sinks (e.g., SQL exec, OS exec, eval, path/file ops, deserialization, SSRF).
       - Occur in authentication/authorization boundaries or around secrets/keys/tokens.
       - Have clear call stacks/evidence strings pointing to concrete methods that run.
       - Affect internet-facing or privileged components.
     * De-prioritize purely theoretical findings with no reachable path or dead code.

   CONSOLIDATION RULES:
   - Aggregate by (CWE, primary sink/function, file[:line], framework route/handler) when applicable.
   - Merge repeated instances across files if they share the same source-sink pattern and remediation is the same.
   - Keep a single representative entry with a count of affected locations; list notable examples.

   OUTPUT FORMAT (MARKDOWN ONLY, BETWEEN MARKERS BELOW):
   - Start with a title and short summary of total findings and how many were consolidated.
   - A table of TOP PRIORITIES sorted by exploitability (highest first) with columns:
     Rank | CWE | Title | Affected Locations | Likely Exploit Path | Risk | Rationale (1–2 lines)
   - "Top 5 Immediate Actions" list with concrete next steps.
   - "Deduplicated Findings (Full Details)" with risk, 0–100 exploitability score, evidence (file:line + methods), remediation, owners, references.
   - If ${CODEX_SAST_PATH} is missing or invalid JSON, output a brief note stating no parsable SAST findings.

   RULES (must follow exactly):
   - PRINT ONLY between these two lines:
     === BEGIN_SECURITY_MD ===
     <MARKDOWN CONTENT HERE>
     === END_SECURITY_MD ===
   - No prose, backticks, code fences, or anything outside the markers.
   - Be concise but specific. Cite only evidence present in the SAST report.

# ---------------------------
# Stage: codex → Job 1 (Recommendations)
# ---------------------------
codex_recommendations:
 stage: codex
 rules:
   - if: '$CI_PIPELINE_SOURCE == "merge_request_event" && $CI_MERGE_REQUEST_SOURCE_PROJECT_ID != $CI_PROJECT_ID'
     when: never
   - if: '$OPENAI_API_KEY'
     when: on_success
   - when: never
 script:
   - set -euo pipefail
   - mkdir -p artifacts
   - ": > ${CODEX_RAW_LOG}"
   - ": > ${CODEX_SECURITY_MD}"

   - apt-get update && apt-get install -y --no-install-recommends curl ca-certificates git lsb-release
   - npm -g i @openai/codex@latest
   - codex --version && git --version

   - |
     if [ ! -s "${CODEX_SAST_PATH}" ]; then
       echo "WARNING: ${CODEX_SAST_PATH} not found or empty. Codex will emit a 'no parsable findings' note."
     fi

   - FILE_LIST="$(git ls-files | sed 's/^/- /')"
   - |
     export CODEX_PROMPT="${CODEX_PROMPT}

     Existing repository files (for reference only; use paths exactly as listed in SAST evidence):
     ${FILE_LIST}"

   # Run Codex and capture raw output (preserve Codex's exit code via PIPESTATUS)
   - |
     set +o pipefail
     codex exec --full-auto "$CODEX_PROMPT" | tee "${CODEX_RAW_LOG}" >/dev/null
     CODEX_RC=${PIPESTATUS[0]}
     set -o pipefail
     echo "Codex exit code: ${CODEX_RC}"

   # Extract markdown between markers; fallback to a minimal note
   - |
     TMP_OUT="$(mktemp)"
     sed -E 's/\x1B\[[0-9;]*[A-Za-z]//g' "${CODEX_RAW_LOG}" | tr -d '\r' | awk '
       /^\s*=== BEGIN_SECURITY_MD ===\s*$/ {grab=1; next}
       /^\s*=== END_SECURITY_MD ===\s*$/   {grab=0}
       grab
     ' > "${TMP_OUT}"
     if ! [ -s "${TMP_OUT}" ]; then
       cat > "${TMP_OUT}" <<'EOF' # Security Findings Priority
       No parsable SAST findings detected in `gl-sast-report.json`._
     EOF
       echo "WARNING: No content extracted; wrote minimal placeholder."
     fi
     mv -f "${TMP_OUT}" "${CODEX_SECURITY_MD}"
     if [ "${CODEX_RC}" -ne 0 ]; then
       echo "WARNING: Codex exited with code ${CODEX_RC}. Proceeding with extracted report." >&2
     fi
 artifacts:
   when: always
   paths:
     - artifacts/codex-sast-raw.log
     - security_priority.md
   expire_in: 14 days
```
下面是我们收到的一份输出示例：

### 输出示例：合并后的 SAST Findings

已解析 `gl-sast-report.json` 并合并重叠问题。
**原始发现总数：** 5 → **合并后代表性条目：** 4
（跨多个 endpoint 的重复 SQL injection 模式已被合并）。

#### 摘要表

| 排名 | CWE      | 标题                                 | 受影响位置数 | 可能的利用路径                         | 风险     | 理由（1–2 行）                                                                                      |
|------|----------|--------------------------------------|--------------|----------------------------------------|----------|-----------------------------------------------------------------------------------------------------|
| 1    | CWE-798  | 硬编码 JWT 私钥                      | 1            | Auth token 签发 / 校验                 | Critical | 仓库泄露后可伪造有效 admin JWT；利用门槛极低，且面向 internet。                                     |
| 2    | CWE-89   | login 与 search 中的 SQL injection   | 2            | Login endpoint；product search         | Critical | 原始 SQL 字符串拼接；可通过公开 HTTP handler 直接绕过登录并窃取数据。                               |
| 3    | CWE-94   | 通过 eval 的 server-side code injection | 1          | User profile update handler            | High     | 对用户输入执行 `eval()` 可导致 RCE；虽为条件触发，但一旦可达影响仍然极高。                          |
| 4    | — (SSRF) | 通过任意图片 URL 获取触发 SSRF       | 1            | Image URL fetch/write flow             | High     | 对未校验 URL 发起外部请求，可访问内部服务/元数据（如 AWS metadata）。                               |

#### 前 5 条立即行动
1. 替换 `lib/insecurity.ts:23` 中硬编码的 JWT 签名密钥；改为从 secret storage 加载，轮换密钥，并使现有 token 失效。
2. 更新 `routes/login.ts:34`，改用参数化查询；移除原始拼接；校验并转义输入。
3. 修复 `routes/search.ts:23`，使用 ORM bind 参数或经过转义的 `LIKE` helper，而不是字符串拼接。
4. 重构 `routes/userProfile.ts:55–66`；用安全模板或白名单求值器替换 `eval()`。
5. 加固图片导入逻辑：对 scheme/host 做 allowlist，阻止 link-local/metadata IP，并添加超时与大小限制。

##### 去重后的 Findings（完整细节）

##### 1. CWE-798 — 硬编码 JWT 私钥
- 风险：Critical — 可利用性 98/100
- 证据：
  - 文件：`lib/insecurity.ts:23`
  - 信息：源码中嵌入 RSA 私钥，可用于伪造 admin token
- 建议的 Remediation：从源码中移除密钥，通过 env/secret manager 加载，轮换密钥，并强制缩短 TTL
- 负责人/团队：Backend/Core（lib）
- 参考：CWE-798；OWASP ASVS 2.1.1, 2.3.1
---
##### 2. CWE-89 — SQL injection（login 与 search）
- 风险：Critical — 可利用性 95/100
- 证据：
  - `routes/login.ts:34` — 典型的 `' OR 1=1--` 登录绕过
  - `routes/search.ts:23` — 通过 `%25' UNION SELECT ...` 做 UNION 提取
- 建议的 Remediation：使用参数化查询/ORM，校验输入，并增加 WAF/错误抑制
- 负责人/团队：Backend/API（routes）
- 参考：CWE-89；OWASP Top 10 A03:2021；ASVS 5.3
---
##### 3. CWE-94 — Server-side code injection（`eval`）
- 风险：High — 可利用性 72/100
- 证据：
  - `routes/userProfile.ts:55–66` — `eval()` 被用于动态用户名模式
- 建议的 Remediation：移除 `eval()`，或在严格白名单下进行沙箱化；校验/编码输入
- 负责人/团队：Backend/API（routes）
- 参考：CWE-94；OWASP Top 10 A03:2021
---
##### 4. SSRF — 任意图片 URL 获取
- 风险：High — 可利用性 80/100
- 证据：
  - 图片导入会获取任意 `imageUrl` → 可能访问内部服务（`169.254.169.254`）
- 建议的 Remediation：强制 HTTPS + DNS/IP allowlist，阻止 RFC1918/link-local，解析后再校验，且不允许重定向
- 负责人/团队：Backend/API（routes）
- 参考：OWASP SSRF Prevention；OWASP Top 10 A10:2021
---
### 第 2 步：根据建议修复安全问题
- Codex 同时消费 SAST JSON 和仓库文件树。
- 对每个 High/Critical 问题：
  - 构造结构化 prompt → 输出统一格式的 `git diff`。
  - 在保存为 `.patch` 前先用 `git apply --check` 验证 diff。

#### Remediation CI/CD Job 示例

**要求**：此 job 依赖前一个 stage 输出的 `security_priority.md` 文件，并以它作为输入来生成用于创建 MR 的 patch 文件：
```yaml
 stages:
  - remediation

default:
  image: node:24

variables:
  # Inputs/outputs
  SAST_REPORT_PATH: "gl-sast-report.json"
  PATCH_DIR: "codex_patches"
  CODEX_DIFF_RAW: "artifacts/codex-diff-raw.log"

  # --- Resolution prompt (produces unified git diffs only) ---
  CODEX_DIFF_PROMPT: |
    You are a secure code remediation assistant.
    You will receive:
    - The repository working tree (read-only)
    - One vulnerability (JSON from a GitLab SAST report)
    - Allowed files list

    GOAL:
    - Create the minimal, safe fix for the vulnerability.
    - Output a unified git diff that applies cleanly with `git apply -p0` (or -p1 for a/ b/ paths).
    - Prefer surgical changes: input validation, safe APIs, parameterized queries, permission checks.
    - Do NOT refactor broadly or change unrelated code.

    RULES (must follow exactly):
    - PRINT ONLY the diff between the markers below.
    - Use repo-relative paths; `diff --git a/path b/path` headers are accepted.
    - No binary file changes. No prose/explanations outside the markers.

    MARKERS:
    === BEGIN_UNIFIED_DIFF ===
    <unified diff here>
    === END_UNIFIED_DIFF ===

    If no safe fix is possible without deeper changes, output an empty diff between the markers.

# ---------------------------
# Stage: remediation → Generate unified diffs/patches
# ---------------------------
codex_resolution:
  stage: remediation
  rules:
    - if: '$OPENAI_API_KEY'
      when: on_success
    - when: never
  script:
    - set -euo pipefail
    - mkdir -p "$PATCH_DIR" artifacts

    # Deps
    - apt-get update && apt-get install -y --no-install-recommends bash git jq curl ca-certificates
    - npm -g i @openai/codex@latest
    - git --version && codex --version || true

    # Require SAST report; no-op if missing
    - |
      if [ ! -s "${SAST_REPORT_PATH}" ]; then
        echo "No SAST report found; remediation will no-op."
        printf "CODEX_CREATED_PATCHES=false\n" > codex.env
        exit 0
      fi

    # Pull High/Critical items
    - jq -c '.vulnerabilities[]? | select((.severity|ascii_downcase)=="high" or (.severity|ascii_downcase)=="critical")' "$SAST_REPORT_PATH" \
        | nl -ba > /tmp/hicrit.txt || true
    - |
      if [ ! -s /tmp/hicrit.txt ]; then
        echo "No High/Critical vulnerabilities found. Nothing to fix."
        printf "CODEX_CREATED_PATCHES=false\n" > codex.env
        exit 0
      fi

    # Ground Codex to actual repo files
    - FILE_LIST="$(git ls-files | sed 's/^/- /')"

    # Identity for any local patch ops
    - git config user.name "CI Codex Bot"
    - git config user.email "codex-bot@example.com"

    - created=0

    # Loop: build prompt (robust temp-file), run Codex, extract diff, validate
    - |
      while IFS=$'\t' read -r idx vuln_json; do
        echo "Processing vulnerability #$idx"
        echo "$vuln_json" > "/tmp/vuln-$idx.json"

        PROMPT_FILE="$(mktemp)"
        {
          printf "%s\n\n" "$CODEX_DIFF_PROMPT"
          printf "VULNERABILITY_JSON:\n<<JSON\n"
          cat "/tmp/vuln-$idx.json"
          printf "\nJSON\n\n"
          printf "EXISTING_REPOSITORY_FILES (exact list):\n"
          printf "%s\n" "$FILE_LIST"
        } > "$PROMPT_FILE"

        PER_FINDING_PROMPT="$(tr -d '\r' < "$PROMPT_FILE")"
        rm -f "$PROMPT_FILE"

        : > "$CODEX_DIFF_RAW"
        set +o pipefail
        codex exec --full-auto "$PER_FINDING_PROMPT" | tee -a "$CODEX_DIFF_RAW" >/dev/null
        RC=${PIPESTATUS[0]}
        set -o pipefail
        echo "Codex (diff) exit code: ${RC}"

        OUT_PATCH="$PATCH_DIR/fix-$idx.patch"
        sed -E 's/\x1B\[[0-9;]*[A-Za-z]//g' "$CODEX_DIFF_RAW" \
          | tr -d '\r' \
          | awk '
              /^\s*=== BEGIN_UNIFIED_DIFF ===\s*$/ {grab=1; next}
              /^\s*=== END_UNIFIED_DIFF ===\s*$/   {grab=0}
              grab
            ' > "$OUT_PATCH"

        if ! [ -s "$OUT_PATCH" ] || ! grep -qE '^\s*diff --git ' "$OUT_PATCH"; then
          echo "  No usable diff produced for #$idx; skipping."
          rm -f "$OUT_PATCH"
          continue
        fi

        # Validate: accept -p0 (repo-relative) or -p1 (a/ b/ prefixes)
        if git apply --check -p0 "$OUT_PATCH" || git apply --check -p1 "$OUT_PATCH"; then
          echo "  Patch validated → $OUT_PATCH"
          created=$((created+1))
        else
          echo "  Patch failed to apply cleanly; removing."
          rm -f "$OUT_PATCH"
        fi
      done < /tmp/hicrit.txt

      if [ "$created" -gt 0 ]; then
        printf "CODEX_CREATED_PATCHES=true\nPATCH_DIR=%s\n" "$PATCH_DIR" > codex.env
      else
        printf "CODEX_CREATED_PATCHES=false\n" > codex.env
      fi
  artifacts:
    when: always
    paths:
      - codex_patches/
      - artifacts/codex-diff-raw.log
    reports:
      dotenv: codex.env
    expire_in: 14 days
```

运行带有 Codex CLI 的 CI/CD job 后，我们会收到一个 Git patch，用来修复安全扫描器最初发现的问题：

```patch
<unified diff here>
diff --git a/routes/profileImageUrlUpload.ts b/routes/profileImageUrlUpload.ts
index 9b4a62d..c7f1a7e 100644
--- a/routes/profileImageUrlUpload.ts
+++ b/routes/profileImageUrlUpload.ts
@@ -5,17 +5,12 @@
 * SPDX-License-Identifier: MIT
 */
-import fs from 'node:fs'
-import { Readable } from 'node:stream'
-import { finished } from 'node:stream/promises'
import { type Request, type Response, type NextFunction } from 'express'
import * as security from '../lib/insecurity'
import { UserModel } from '../models/user'
-import * as utils from '../lib/utils'
-import logger from '../lib/logger'
export function profileImageUrlUpload () {
  return async (req: Request, res: Response, next: NextFunction) => {
    if (req.body.imageUrl !== undefined) {
      const url = req.body.imageUrl
      if (url.match(/(.)*solve\\/challenges\\/server-side(.)*/) !== null) req.app.locals.abused_ssrf_bug = true
      const loggedInUser = security.authenticatedUsers.get(req.cookies.token)
      if (loggedInUser) {
-        try {
-          const response = await fetch(url)
-          if (!response.ok || !response.body) {
-            throw new Error('url returned a non-OK status code or an empty body')
-          }
-          const ext = ['jpg', 'jpeg', 'png', 'svg', 'gif'].includes(url.split('.').slice(-1)[0].toLowerCase()) ? url.split('.').slice(-1)[0].toLowerCase() : 'jpg'
-          const fileStream = fs.createWriteStream(`frontend/dist/frontend/assets/public/images/uploads/${loggedInUser.data.id}.${ext}`, { flags: 'w' })
-          await finished(Readable.fromWeb(response.body as any).pipe(fileStream))
-          await UserModel.findByPk(loggedInUser.data.id).then(async (user: UserModel | null) => { return await user?.update({ profileImage: `/assets/public/images/uploads/${loggedInUser.data.id}.${ext}` }) }).catch((error: Error) => { next(error) })
-        } catch (error) {
-          try {
-            const user = await UserModel.findByPk(loggedInUser.data.id)
-            await user?.update({ profileImage: url })
-            logger.warn(`Error retrieving user profile image: ${utils.getErrorMessage(error)}; using image link directly`)
-          } catch (error) {
-            next(error)
-            return
-          }
-        }
+        try {
+          const user = await UserModel.findByPk(loggedInUser.data.id)
+          await user?.update({ profileImage: url })
+        } catch (error) {
+          next(error)
+          return
+        }
      } else {
        next(new Error('Blocked illegal activity by ' + req.socket.remoteAddress))
        return
```

## 关键收益
在 GitLab CI/CD 中使用 Codex CLI，可以增强现有的审查流程，让团队更快地交付代码。

* **互补性**：Codex 不会替代扫描器，而是解释它们的发现并加速修复。
* **可执行性**：审阅者看到的不只是漏洞，还有经过优先级排序的修复步骤。
* **自动化**：Patch 直接在 CI 中创建，可用于 `git apply` 或 remediation branch。

---

## 总结

在这篇 cookbook 中，我们探索了如何将 **Codex CLI** 嵌入 GitLab CI/CD pipeline，使软件交付过程更安全、更易维护：

* **Code Quality Reports**：生成符合 GitLab 规范的 CodeClimate JSON，让基于推理的发现与 lint、单元测试和风格检查一起展示。

* **Vulnerability Interpretation**：接收 SAST 和其他安全扫描器的原始输出（`gl-sast-report.json`），并将其转换为经过优先级排序、便于人工阅读的计划（`security_priority.md`），其中包含去重、可利用性排序以及可执行的下一步建议。

* **Automated Remediation**：通过让 Codex 生成统一格式的 git diff，并将其保存为 `.patch` 文件来扩展工作流。这些 patch 会经过验证（`git apply --check`），并可自动应用到新分支上。

综合来看，这些模式说明 **LLM 驱动的分析是在补充而非替代传统的基于规则的工具**。扫描器仍然是检测的事实来源，而 Codex 则增加了上下文感知、优先级判断、开发者指导，甚至还能通过 MR 提出具体修复方案。GitLab 的 schema 和 API 为这些输出提供了结构，使其可预测、可执行，并且能够完整集成到开发者工作流中。

其中最关键的一点是，要获得高质量结果，必须具备 **清晰的 prompt、schema 校验以及 guardrail**。JSON 标记、severity 白名单、schema 强制校验以及 diff 验证，都能确保输出真正可用。

展望未来，这种模式可以扩展为通过单一的 Codex 驱动 remediation 流程来统一处理各种主要扫描类型：

* **Dependency Scans** → 合并多个 lockfile 中的 CVE，推荐升级方案，并自动生成提升有漏洞版本的 diff。
* **Container/Image Scans** → 标记过期的基础镜像，并提出 Dockerfile 更新建议。
* **DAST Results** → 高亮可被利用的 endpoint，并通过修补 routing/middleware 来补上校验或访问控制。

通过把这些能力合并进单一的 Codex 驱动后处理 + remediation pipeline，团队就能在所有安全领域持续获得 **可执行的指导和经过验证的 patch**。

**更广泛的启示是：** 借助 prompt engineering、schema 校验以及与 GitLab 原生 MR 工作流的集成，LLM 会从“顾问”进化为 **一等公民的 CI/CD agent**，帮助团队交付的不只是可运行的代码，也包括更安全、更易维护、并且在可能时可自动完成修复的代码。
