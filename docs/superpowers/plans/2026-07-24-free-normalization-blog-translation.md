# Free Normalization 博客译述执行计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增一篇可直接发布到微信公众号的 Free Normalization 中文技术译述稿，将 15 张原图全部转存 mdnice，并提交、推送到 GitHub。

**Architecture:** 从 PyTorch Blog 和开源代码提取事实、公式、图像顺序与引用，使用 Firefox 的 mdnice cookie 调用上传 API。译述稿放在 `pytorch/kernels/`，按算法问题、三类融合方法和 FlashNormAttention 展开，最后通过图片、链接、数字、Markdown 和文本风格检查。

**Tech Stack:** Markdown、PyTorch Blog、curl、Firefox cookies.sqlite、mdnice upload API、Git。

---

### Task 1: 核对正文与图片顺序

**Files:**
- Source: `https://pytorch.org/blog/towards-free-normalization-fusing-normalization-into-gemm-and-attention-kernels/`
- Read: `pytorch/kernels/`
- Read: `pytorch/compile/【博客翻译·译述】Triton 插件扩展：开箱即用的 TLX 与自定义编译器 Pass.md`

- [ ] **Step 1: 核对元数据与技术主线**

记录原文作者，并确认文章依次讲 normalization tiling 冲突、Lazy Pre-Norm、Multi-CTA Norm、fusion regrouping、FlashNormAttention forward/backward 和致谢。

- [ ] **Step 2: 核对关键数字**

确认以下信息进入文章：

```text
Kunlun normalization latency: about 20%
Typical LLM normalization latency: about 10%
GEMM fusion hides up to 90% of norm latency
FlashNormAttention kernel speedup: up to 35%
Benchmark: bfloat16, NVIDIA B200, 750 W power cap
Naive small-N fusion saving: 17%-32% of LayerNorm latency
Single-CTA tile_n limit: 512
Multi-CTA portable N limit on Blackwell: 4096
```

- [ ] **Step 3: 固定 15 张图片的原始 URL 与章节映射**

```text
cover  -> All-PyTorch-Blog-Social-Images-12.png
fig01  -> 1.png
fig02  -> 2-1-scaled.png
fig03  -> 3.png
fig04  -> 4.png
fig05  -> 5.png
fig06  -> 6.png
fig07  -> 7.png
fig08  -> 8.png
fig09  -> 9.png
fig10  -> 10.png
fig11  -> 11-scaled.png
fig12  -> 12-scaled.png
fig13  -> 13.png
fig14  -> 14.png
```

### Task 2: 下载并上传全部图片

**Files:**
- Temporary directory: `/tmp/free-normalization-blog-images/`
- Remote output: 15 mdnice image objects

- [ ] **Step 1: 下载原始分辨率图片**

对 15 个 `https://pytorch.org/wp-content/uploads/2026/07/` URL 使用 `curl -L --fail` 下载，并运行：

```bash
file /tmp/free-normalization-blog-images/*.png
```

Expected: 15 个 PNG 文件，图片数量和原文一致。

- [ ] **Step 2: 从当前 Firefox profile 安全读取 mdnice token**

使用活动 profile：

```text
~/Library/Application Support/Firefox/Profiles/ugv99mtx.default-release-1/cookies.sqlite
```

先复制数据库到 `mktemp` 路径，再只在 shell 变量中读取 `.mdnice.com` 的 `token`；日志不得打印 token。

- [ ] **Step 3: 调用真实 mdnice 上传接口**

对每张图片执行：

```bash
curl --fail-with-body --silent --show-error \
  -X POST 'https://api.mdnice.com/file/user/upload' \
  -H "Authorization: Bearer $token" \
  -F "file=@$image"
```

Expected: 每个响应都是 `success: true`，并返回唯一的 `https://files.mdnice.com/user/59/...png`。

- [ ] **Step 4: 验证 15 个图床链接**

对返回的每个 URL 运行 `curl -I`。Expected: HTTP 200 且 `Content-Type: image/png`。

### Task 3: 编写中文译述稿

**Files:**
- Create: `pytorch/kernels/【博客翻译·译述】让归一化接近“免费”：把 Norm 融合进 GEMM 与 Attention Kernel.md`

- [ ] **Step 1: 写开头、来源与 TL;DR**

文章顶部放 mdnice 封面，随后写原文标题、原文裸链接、作者和译述说明。TL;DR 说明融合最多隐藏 90% norm latency，FlashNormAttention 最高加速 35%，并标明 benchmark 环境。

- [ ] **Step 2: 写 tiling 冲突与 naive fusion**

解释 norm 需要整行 reduction、GEMM 同时按 M/N 分块、强制 `tile_n=N` 会破坏 GEMM tiling，并插入 fig01、fig02、fig03。

- [ ] **Step 3: 写 Lazy Pre-Norm**

保留公式：

```text
rmsnorm(A) = A * rstd(A)[:, None]
rmsnorm(A) @ B = (A @ B) * rstd(A)[:, None]
```

解释在 K-loop 中累计平方和、把 row-wise scale 延迟到 epilogue，并写明 affine、LayerNorm 和 backward 限制。插入 fig04、fig05。

- [ ] **Step 4: 写 Multi-CTA Norm 与 fusion regrouping**

解释 CTA cluster、DSMEM reduction、调度限制和 N 上限。说明 forward epilogue 如何在 backward 变为 prologue，以及通过更换融合邻居让 forward/backward 都保持 epilogue。插入 fig06 到 fig10。

- [ ] **Step 5: 写 FlashNormAttention**

解释 short-KV GDPA 调整、两次 norm 和两次 residual 融合；介绍 buffer reuse、TMEM accumulate、register subtiling、warp specialization、register preload、backward recompute 与 TMA_REDUCE_ADD。插入 fig11 到 fig14。

- [ ] **Step 6: 写代码、致谢与参考资料**

使用文本加裸链接列出：

```text
https://github.com/facebookresearch/ads_model_kernel_library/tree/main/multi_cta_norm_fusion
https://github.com/facebookresearch/ads_model_kernel_library/tree/main/gdpa_megakernel
https://arxiv.org/abs/2602.10016
https://github.com/Dao-AILab/quack/blob/main/media/2025-07-10-membound-sol.md
https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
https://arxiv.org/pdf/2407.08608
```

保留原文对 Tri Dao、Markus Hoehnerbach、Jay Shah、Ted Zadouri、Vijay Thakkar、Wentao Guo，以及 PyTorch/Triton 团队的致谢。

### Task 4: 质量检查

**Files:**
- Verify: `pytorch/kernels/【博客翻译·译述】让归一化接近“免费”：把 Norm 融合进 GEMM 与 Attention Kernel.md`

- [ ] **Step 1: 检查图片**

确认正好 15 个 Markdown 图片链接，全部来自 `files.mdnice.com`，不存在 `pytorch.org/wp-content/uploads`。

- [ ] **Step 2: 检查普通链接**

使用 PCRE 扫描普通 Markdown 超链接。Expected: 除图片语法外没有 `[text](https://...)`。

- [ ] **Step 3: 检查事实标记**

使用 `rg` 确认 20%、10%、90%、35%、17%-32%、512、4096、B200、750 W、Lazy Pre-Norm、Multi-CTA Norm 和 FlashNormAttention 均出现。

- [ ] **Step 4: 去除 AI 写作模式**

扫描并重写宣传性词语、模糊归因、否定式排比、机械三段式、重复总结、过量粗体和破折号。保留具体观点和工程限制。

- [ ] **Step 5: 检查 Markdown**

运行代码围栏配对、尾随空格和 `git diff --check`。Expected: 全部通过。

### Task 5: 提交并推送

**Files:**
- Add: `docs/superpowers/plans/2026-07-24-free-normalization-blog-translation.md`
- Add: `pytorch/kernels/【博客翻译·译述】让归一化接近“免费”：把 Norm 融合进 GEMM 与 Attention Kernel.md`

- [ ] **Step 1: 检查范围**

运行 `git status --short` 和 `git diff --stat`。确认本次文件之外的 SGLang 图片、脚本和旧计划没有被暂存。

- [ ] **Step 2: 只暂存本次计划和文章**

```bash
git add \
  'docs/superpowers/plans/2026-07-24-free-normalization-blog-translation.md' \
  'pytorch/kernels/【博客翻译·译述】让归一化接近“免费”：把 Norm 融合进 GEMM 与 Attention Kernel.md'
```

- [ ] **Step 3: 提交**

```bash
git commit -m 'docs: translate free normalization blog'
```

- [ ] **Step 4: 同步并推送**

```bash
git fetch origin master
git rev-list --left-right --count origin/master...HEAD
git push origin master
```

Expected: 远端没有未合并提交；push 后 `origin/master` 与 HEAD SHA 一致。

