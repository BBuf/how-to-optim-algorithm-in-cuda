# Triton Plugin Extensions 博客译述执行计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在仓库中新增一篇可直接用于微信公众号排版的 Triton Plugin Extensions 中文技术译述稿，并将正文图片替换为 mdnice 图床链接。

**Architecture:** 先从 PyTorch Blog 获取正文结构、代码、表格、链接和图片清单，再通过 Firefox 的现有登录态将原图上传 mdnice。文章按仓库近期“博客翻译·译述”稿的格式重写，最后用原文数据核对、Markdown 扫描和中文人工化检查收口。

**Tech Stack:** Markdown、PyTorch Blog HTML、Firefox、mdnice 图床、Git、命令行文本检查。

---

### Task 1: 固化原文结构和资源清单

**Files:**
- Read: `pytorch/inference/【博客翻译·译述】Serving DeepSeek-V4 on GB300 with SGLang：Day-0以来同等交互性吞吐提升5倍.md`
- Read: `pytorch/inference/【翻译】CUDA-Free Inference for LLMs.md`
- Source: `https://pytorch.org/blog/triton-plugin-extensions-enabling-tlx-and-custom-compiler-passes-out-of-the-box/`

- [ ] **Step 1: 提取正文元数据**

核对英文标题、作者 Corbin Robeck、Puyan Lotfi、Ian Barber、Shane Nay、Alexey Loginov、Oleksandr Stashuk、Wenyuan Chi，以及发布日期 2026 年 7 月 15 日。

- [ ] **Step 2: 建立章节核对清单**

清单必须覆盖 TLDR、扩展动机、插件系统、可覆盖编译流水线、自定义 op/dialect/lowering、逐 kernel 控制、TLX、H100、MI350、GPU MODE Trimul、相同 codegen、安装方法和后续方向。

- [ ] **Step 3: 提取图片和外部链接**

确认正文主图为：

```text
https://pytorch.org/wp-content/uploads/2026/06/Triton-Plugin-Extensions-Enabling-TLX-and-Custom-Compiler-Passes-Out-of-the-Box.png
```

记录 TLX、Colab、Gist、插件文档、triton-ext 和 PyPI 链接，后续统一写为“名称（裸链接）”。

### Task 2: 转存正文图片到 mdnice

**Files:**
- Temporary download: `/tmp/triton-plugin-extensions.png`
- Create remotely: one mdnice image-hosting object

- [ ] **Step 1: 下载并检查原图**

Run:

```bash
curl -L 'https://pytorch.org/wp-content/uploads/2026/06/Triton-Plugin-Extensions-Enabling-TLX-and-Custom-Compiler-Passes-Out-of-the-Box.png' -o /tmp/triton-plugin-extensions.png
file /tmp/triton-plugin-extensions.png
```

Expected: PNG image，分辨率 1920×1080。

- [ ] **Step 2: 使用 Firefox 登录态上传 mdnice**

在 mdnice 图片上传入口选择 `/tmp/triton-plugin-extensions.png`，等待返回 `https://files.mdnice.com/user/59/...png`。

- [ ] **Step 3: 验证图床链接**

Run:

```bash
curl -I "$(pbpaste)"
```

Expected: HTTP 200，内容类型为 `image/png`。

### Task 3: 编写中文译述稿

**Files:**
- Create: `pytorch/compile/【博客翻译·译述】Triton 插件扩展：开箱即用的 TLX 与自定义编译器 Pass.md`

- [ ] **Step 1: 写来源说明和 TLDR**

开头包含 mdnice 主图、原文标题、原文裸链接、作者、日期和“翻译·译述”说明。TLDR 直接解释 Triton 3.7 插件机制和 TLX 开箱即用带来的变化。

- [ ] **Step 2: 写插件系统主体**

解释 `TRITON_PLUGIN_PATHS`、`.so` 动态加载、TTIR/TTGIR/LLVM IR/PTX/AMDGCN 各阶段 hook，以及插入、禁用、替换 pass 和覆盖 stage 的能力。

- [ ] **Step 3: 写 TLX API 与硬件映射**

保留 TLX API 表格、H100 TMA/WGMMA persistent GEMM 代码和 MI350 基于寄存器的数据搬运代码。代码 API、变量名和控制流不得改写。

- [ ] **Step 4: 写性能与实际案例**

准确保留 H100 六组 GEMM 数据、MI350 四组 GEMM 数据和 GPU MODE Trimul 的 `19.2 ms -> 12.0 ms`、`1.61x` 结果，并说明动态插件路径与编译进 fork 的路径生成相同代码。

- [ ] **Step 5: 写安装方法和后续方向**

保留 Triton 构建、`triton-utlx` 安装、插件路径设置和 import 代码；列出上游文档与示例的裸链接，并按原文概括未来的自定义 backend、distributed primitive、profiling、pass 和 specialized op。

### Task 4: 公众号格式与文本质量检查

**Files:**
- Verify: `pytorch/compile/【博客翻译·译述】Triton 插件扩展：开箱即用的 TLX 与自定义编译器 Pass.md`

- [ ] **Step 1: 核对原文事实**

逐项检查所有硬件型号、矩阵尺寸、TFLOPS、百分比、API 名称、环境变量和链接。

- [ ] **Step 2: 检查图片和链接格式**

Run:

```bash
rg -n 'pytorch.org/wp-content/uploads|\]\(https?://' 'pytorch/compile/【博客翻译·译述】Triton 插件扩展：开箱即用的 TLX 与自定义编译器 Pass.md'
```

Expected: 不存在原站图片；除 Markdown 图片语法外，不存在 Markdown 超链接。

- [ ] **Step 3: 去除 AI 写作模式**

删除宣传性形容词、模糊归因、重复总结、机械三段式、过量粗体和破折号；保留作者原本的判断与技术语气。

- [ ] **Step 4: 检查 Markdown**

Run:

```bash
git diff --check -- 'pytorch/compile/【博客翻译·译述】Triton 插件扩展：开箱即用的 TLX 与自定义编译器 Pass.md'
```

Expected: 无空白错误，代码围栏成对，标题层级连续。

### Task 5: 提交译述稿

**Files:**
- Add: `docs/superpowers/plans/2026-07-24-triton-plugin-extensions-translation.md`
- Add: `pytorch/compile/【博客翻译·译述】Triton 插件扩展：开箱即用的 TLX 与自定义编译器 Pass.md`

- [ ] **Step 1: 确认提交范围**

Run:

```bash
git status --short
git diff --stat
```

Expected: 不包含用户现有的 SGLang 文章图片、绘图脚本或其他未跟踪文件。

- [ ] **Step 2: 仅暂存本次文件并提交**

```bash
git add 'docs/superpowers/plans/2026-07-24-triton-plugin-extensions-translation.md' 'pytorch/compile/【博客翻译·译述】Triton 插件扩展：开箱即用的 TLX 与自定义编译器 Pass.md'
git commit -m 'docs: translate Triton plugin extensions blog'
```

- [ ] **Step 3: 检查最终状态**

Run:

```bash
git show --stat --oneline HEAD
git status --short
```

Expected: 新提交只含计划和译述稿；用户原有未跟踪文件仍原样保留。
