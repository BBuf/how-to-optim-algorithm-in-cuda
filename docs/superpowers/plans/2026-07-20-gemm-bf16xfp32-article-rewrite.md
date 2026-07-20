# `gemm_bf16xfp32` 通俗化重写 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按同目录原创文章的「效果先行」风格重写 `gemm_bf16xfp32` 笔记，让懂基本 GPU 概念但不懂数值分析的读者能够读懂。

**Architecture:** 正文按前言、PR 效果、通俗原理、可选公式推导组织。benchmark 数据和图片保持不变；主叙事只保留三个关键关系，完整的 `u -> u^2` 推导集中在标注 Codex GPT-5.6 辅助的 `0x3`。

**Tech Stack:** Markdown、LaTeX、Python/PyTorch 示例、Git

---

### Task 1: 重排前言与 benchmark

**Files:**
- Modify: `large-language-model/sglang/HPC-Ops gemm_bf16xfp32 Kernel笔记：FP32权重拆两个BF16跑Tensor Core.md`
- Preserve: `large-language-model/sglang/assets/gemm_bf16xfp32_pr30247_benchmark.png`

- [x] **Step 1: 把前言压缩成问题、办法和文章范围**

前言保留下面四点：BF16 激活乘 FP32 router 权重、Tensor Core 没有直接对应的输入组合、HPC-Ops 用两个 BF16 表示一个 FP32、CUDA 实现另文解析。上游源码继续使用裸 URL：

```text
https://github.com/Tencent/hpc-ops/blob/main/src/gemm/sm90/gemm_bf16xfp32.cu
```

- [x] **Step 2: 把 PR benchmark 移到 0x1**

microbenchmark 继续使用以下六行，不能改数值：

```text
Chat: m=64   15.5us vs 35.6us    2.31x
Chat: m=512  38.4us vs 120.4us   3.14x
Chat: m=8192 430.0us vs 1661.1us 3.86x
Lite: m=64   14.8us vs 23.1us    1.56x
Lite: m=512  20.7us vs 44.1us    2.13x
Lite: m=8192 113.1us vs 487.8us  4.31x
```

保留现有 PNG、GSM8K `0.798 -> 0.802`、invalid `0.000 -> 0.001` 和端到端吞吐表。先解释 kernel 加速，再解释为什么放回模型后只剩 prefill `+2.8%` 到 `+5.4%`。

### Task 2: 用直觉、代码和少量公式重写原理

**Files:**
- Modify: `large-language-model/sglang/HPC-Ops gemm_bf16xfp32 Kernel笔记：FP32权重拆两个BF16跑Tensor Core.md`

- [x] **Step 1: 用位宽表和实验解释直接 BF16 的问题**

保留 FP32/BF16 位宽表，不定义 unit roundoff。用「FP32 有 24 位有效精度，BF16 有 8 位；直接 cast 会丢掉后 16 位」解释问题。保留 `0.5775` 最大绝对误差的 PyTorch 实验。

- [x] **Step 2: 用十进制例子解释 high/low 拆分**

使用下面的例子：

```text
原数      1.234567
high      1.23
剩余      0.004567
low       4.57        # 把剩余放大 1000 倍再保留 3 位
重建      1.23 + 4.57 / 1000 = 1.23457
```

紧接三行 PyTorch 拆分代码：

```python
scale = 1 / 256
w_high = w.to(torch.bfloat16)
w_low = ((w - w_high.float()) / scale).to(torch.bfloat16)
```

- [x] **Step 3: 主文只保留三个关键关系**

```text
w ≈ w_high + w_low / 256
y = x @ w_high.T + (x @ w_low.T) / 256
权重误差量级：2^-8 -> 2^-16
```

用自然语言解释：high 保存主体，low 保存 high 舍入后漏掉的部分；两次 GEMM 都能走 BF16 Tensor Core；`0.5775 -> 0.0012` 是同一输入上的实测例子。

- [x] **Step 4: 写清代价和源码入口**

说明两次 BF16 GEMM 是一份 BF16 GEMM 的两倍计算量；两份 BF16 权重共 4 字节，等于一份 FP32；权重只需预处理一次。CUDA 不解析，只给两个裸 URL：

```text
https://github.com/Tencent/hpc-ops/blob/main/src/gemm/sm90/gemm_bf16xfp32.cu
https://github.com/BBuf/how-to-optim-algorithm-in-cuda/blob/master/cutlass/cute/HPC-Ops%20gemm_bf16xfp32%20kernel%E9%80%90%E8%A1%8C%E8%A7%A3%E6%9E%90%EF%BC%9A%E4%BB%8ECuTe%E5%9F%BA%E7%A1%80%E5%88%B0Hopper%20warp%20specialization.md
```

### Task 3: 增加可选的公式推导节

**Files:**
- Modify: `large-language-model/sglang/HPC-Ops gemm_bf16xfp32 Kernel笔记：FP32权重拆两个BF16跑Tensor Core.md`

- [x] **Step 1: 添加辅助声明**

`# 0x3. 公式推导（可选阅读）` 的第一段必须逐字包含：

```text
本节由 Codex GPT-5.6 辅助整理和检查。前面的原理理解不依赖这一节，想继续看误差界再往下读。
```

- [x] **Step 2: 只推导权重重建误差**

保留以下推导：

```text
u = 2^-8
w_high = w(1 + delta_high), |delta_high| <= u
r = w - w_high, |r| <= u|w|
w_low = (r / scale)(1 + delta_low), |delta_low| <= u
w_hat = w_high + scale * w_low = w + r * delta_low
|w_hat - w| <= u^2|w| = 2^-16|w|
```

结尾说明 `2^-16` 只描述权重表示误差，最终 GEMM 还受 FP32 累加和正负抵消影响。不得出现 Sterbenz lemma、`gamma_k` 或矩阵逐元素误差界。

### Task 4: 校验文风、结构和数据

**Files:**
- Check: `large-language-model/sglang/HPC-Ops gemm_bf16xfp32 Kernel笔记：FP32权重拆两个BF16跑Tensor Core.md`
- Check: `large-language-model/sglang/assets/gemm_bf16xfp32_pr30247_benchmark.png`

- [x] **Step 1: 检查篇幅与标题**

```bash
wc -l 'large-language-model/sglang/HPC-Ops gemm_bf16xfp32 Kernel笔记：FP32权重拆两个BF16跑Tensor Core.md'
rg -n '^# 0x' 'large-language-model/sglang/HPC-Ops gemm_bf16xfp32 Kernel笔记：FP32权重拆两个BF16跑Tensor Core.md'
```

Expected: 约 180–220 行；主标题只有 `0x0`、`0x1`、`0x2`、`0x3`。

- [x] **Step 2: 检查删除项与链接格式**

```bash
rg -n 'Sterbenz|gamma_k|jit_kernel|TVM-FFI|H200 上的重新调参|\[[^]]+\]\(https?://' 'large-language-model/sglang/HPC-Ops gemm_bf16xfp32 Kernel笔记：FP32权重拆两个BF16跑Tensor Core.md'
```

Expected: no output。

- [x] **Step 3: 检查必需数字和辅助声明**

```bash
rg -n '0\.5775|0\.0012|15\.5|430\.0|4\.31|0\.798|0\.802|\+5\.4%|Codex GPT-5\.6|2\^\{-16\}' 'large-language-model/sglang/HPC-Ops gemm_bf16xfp32 Kernel笔记：FP32权重拆两个BF16跑Tensor Core.md'
```

Expected: 每项至少命中一次。

- [x] **Step 4: 按 humanizer-zh 复核文风**

删除宣传语、机械排比、连续抽象定义和聊天式结尾。正文应保留作者常用的第一人称观察、具体数字和「简单来说」「这里的意思是」等自然解释，但不机械重复这些短语。五项评分至少 45/50。

### Task 5: 提交并推送

**Files:**
- Commit: rewritten article
- Commit: updated implementation plan

- [ ] **Step 1: 检查并暂存目标文件**

```bash
git diff --check
git add docs/superpowers/plans/2026-07-20-gemm-bf16xfp32-article-rewrite.md \
  'large-language-model/sglang/HPC-Ops gemm_bf16xfp32 Kernel笔记：FP32权重拆两个BF16跑Tensor Core.md'
git diff --cached --check
git diff --cached --stat
```

- [ ] **Step 2: 提交并推送 `master`**

```bash
git commit -m "docs: make bf16xfp32 note easier to read"
git push origin master
```

Expected: `origin/master` 与本地 HEAD 一致。
