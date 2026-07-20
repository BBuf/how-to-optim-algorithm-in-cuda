# `gemm_bf16xfp32` 原理文章改写 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 把现有 `gemm_bf16xfp32` 笔记改成以双 BF16 分解及误差推导为主的公众号文章，补充 PR #30247 的当前 benchmark 图表，并删除 JIT 集成和 H200 调参叙事。

**Architecture:** 文章只保留 `0x0` 到 `0x3`：背景、问题、数学原理和 PR benchmark。benchmark PNG 由 PR 正文中的固定数据生成；正文中的表格、图和结论使用同一组数据，再通过结构化检查防止错抄。

**Tech Stack:** Markdown、LaTeX 公式、Python 3、Matplotlib、Git

---

### Task 1: 生成 PR #30247 benchmark 图

**Files:**
- Create: `large-language-model/sglang/assets/gemm_bf16xfp32_pr30247_benchmark.png`
- Temporary generator: `/private/tmp/generate_gemm_bf16xfp32_chart.py`

- [x] **Step 1: 写入固定 benchmark 数据和双面板绘图脚本**

用 `apply_patch` 创建下面的完整脚本。数据来自 PR #30247 当前正文：

```python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/Users/bbuf/工作目录/Common/how-to-optim-algorithm-in-cuda")
OUTPUT = ROOT / "large-language-model/sglang/assets/gemm_bf16xfp32_pr30247_benchmark.png"

m_labels = ["64", "512", "8192"]
chat_speedup = [2.31, 3.14, 3.86]
lite_speedup = [1.56, 2.13, 4.31]
batch_labels = ["1", "16", "64"]
prefill_gain = [2.8, 3.5, 5.4]

plt.rcParams.update(
    {
        "font.sans-serif": ["Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 12,
    }
)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("#f8fafc")
fig.subplots_adjust(left=0.07, right=0.97, top=0.80, bottom=0.18, wspace=0.25)

x = np.arange(len(m_labels))
width = 0.34
bars_chat = ax0.bar(x - width / 2, chat_speedup, width, label="Chat: k=6144, n=768", color="#2563eb")
bars_lite = ax0.bar(x + width / 2, lite_speedup, width, label="Lite: k=3072, n=384", color="#f59e0b")
ax0.axhline(1.0, color="#64748b", linewidth=1, linestyle="--")
ax0.set_title("H200 上的单 kernel 加速比")
ax0.set_xlabel("token 数 m")
ax0.set_ylabel("相对 FP32 MM 的加速比")
ax0.set_xticks(x, m_labels)
ax0.set_ylim(0, 5.0)
ax0.legend(frameon=False, loc="upper left")
ax0.bar_label(bars_chat, labels=[f"{v:.2f}×" for v in chat_speedup], padding=3)
ax0.bar_label(bars_lite, labels=[f"{v:.2f}×" for v in lite_speedup], padding=3)

x1 = np.arange(len(batch_labels))
bars_e2e = ax1.bar(x1, prefill_gain, width=0.52, color="#16a34a")
ax1.set_title("LongCat-Flash-Lite-FP8 端到端 prefill")
ax1.set_xlabel("batch size")
ax1.set_ylabel("输入吞吐提升")
ax1.set_xticks(x1, batch_labels)
ax1.set_ylim(0, 6.4)
ax1.bar_label(bars_e2e, labels=[f"+{v:.1f}%" for v in prefill_gain], padding=3)

for ax in (ax0, ax1):
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    ax.set_axisbelow(True)

fig.suptitle("PR #30247 benchmark：kernel 快多少，模型能拿到多少", fontsize=18, fontweight="bold", y=0.94)
fig.text(0.5, 0.045, "数据来源：https://github.com/sgl-project/sglang/pull/30247", ha="center", color="#475569")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, dpi=180, facecolor=fig.get_facecolor())
plt.close(fig)
```

左图绘制两组 kernel speedup 柱，右图绘制 TP1 prefill 输入吞吐增幅。输出尺寸为 `14 x 6 inch`、`180 dpi`，柱顶直接显示 `x` 或 `%` 数值，标题中写明 `H200`。

- [x] **Step 2: 运行脚本并验证 PNG**

Run:

```bash
python3 /private/tmp/generate_gemm_bf16xfp32_chart.py
file large-language-model/sglang/assets/gemm_bf16xfp32_pr30247_benchmark.png
```

Expected: `PNG image data`，分辨率为 `2520 x 1080`。

- [x] **Step 3: 目视检查图像**

检查项目：中文字体正常、数据标签不重叠、左图六个加速比与右图三个吞吐增幅全部可读、图例没有遮挡柱子。

### Task 2: 重命名并重写文章

**Files:**
- Delete: `large-language-model/sglang/HPC-Ops gemm_bf16xfp32 Kernel笔记：FP32权重拆两个BF16跑Tensor Core，附SGLang集成与H200调优记录.md`
- Create: `large-language-model/sglang/HPC-Ops gemm_bf16xfp32 Kernel笔记：FP32权重拆两个BF16跑Tensor Core.md`

- [x] **Step 1: 写 0x0 和 0x1**

0x0 只介绍问题和上游源码：

```text
https://github.com/Tencent/hpc-ops/blob/main/src/gemm/sm90/gemm_bf16xfp32.cu
```

0x1 用 FP32 的 24 位有效精度和 BF16 的 8 位有效精度说明直接降精度的误差来源，并保留直接 BF16 权重最大绝对误差约 `0.5775` 的可复现实验。正文不出现 `jit_kernel`、TVM-FFI 或内嵌源码。

- [x] **Step 2: 写 0x2 的标量误差推导**

正文依次给出以下公式及文字解释：

```text
u = 2^-8
w_h = round_bf16(w)
r = w - w_h,              |r| <= u |w|
s = 2^-8
w_l = round_bf16(r / s)
w_hat = w_h + s w_l
|w_hat - w| <= u^2 |w| = 2^-16 |w|
```

说明 `r / s = 256r` 在正常数范围内只移动二进制指数，不消耗有效位。明确 `2^-16` 是重建权重的表示误差估计，而不是整个 GEMM 的逐位正确保证。

- [x] **Step 3: 写点积误差、实现代价与延伸阅读**

点积使用下式：

```text
|x^T(w_hat-w)| <= sum_i |x_i| |w_hat_i-w_i|
                 <= u^2 sum_i |x_i w_i|
```

说明输出接近零且正负项抵消时，相对误差可以很大，因此只承诺绝对前向误差界。随后写清两次 BF16 Tensor Core GEMM、FP32 累加、权重总存储量等于原 FP32 权重、计算量翻倍。加入双 BF16 误差约 `0.0012` 的实验结果。

CUDA 细节只给以下裸 URL：

```text
https://github.com/Tencent/hpc-ops/blob/main/src/gemm/sm90/gemm_bf16xfp32.cu
https://github.com/BBuf/how-to-optim-algorithm-in-cuda/blob/master/cutlass/cute/HPC-Ops%20gemm_bf16xfp32%20kernel%E9%80%90%E8%A1%8C%E8%A7%A3%E6%9E%90%EF%BC%9A%E4%BB%8ECuTe%E5%9F%BA%E7%A1%80%E5%88%B0Hopper%20warp%20specialization.md
```

- [x] **Step 4: 写 0x3 的 PR benchmark**

microbenchmark 表格使用这六行：

```text
Chat: m=64   HPC=15.5us  FP32 MM=35.6us    2.31x
Chat: m=512  HPC=38.4us  FP32 MM=120.4us   3.14x
Chat: m=8192 HPC=430.0us FP32 MM=1661.1us  3.86x
Lite: m=64   HPC=14.8us  FP32 MM=23.1us    1.56x
Lite: m=512  HPC=20.7us  FP32 MM=44.1us    2.13x
Lite: m=8192 HPC=113.1us FP32 MM=487.8us   4.31x
```

插入 `assets/gemm_bf16xfp32_pr30247_benchmark.png`。模型级表格写入 GSM8K `0.798 -> 0.802`、invalid `0.000 -> 0.001`，以及 prefill input throughput 的 `+2.8%/+3.5%/+5.4%`。decode 的 `+1.0%/-0.3%/-0.4%` 描述为接近持平。PR 使用裸 URL：

```text
https://github.com/sgl-project/sglang/pull/30247
```

### Task 3: 校验数据、结构和中文文风

**Files:**
- Check: `large-language-model/sglang/HPC-Ops gemm_bf16xfp32 Kernel笔记：FP32权重拆两个BF16跑Tensor Core.md`
- Check: `large-language-model/sglang/assets/gemm_bf16xfp32_pr30247_benchmark.png`

- [x] **Step 1: 运行结构和禁用内容检查**

Run:

```bash
rg -n '^# 0x[456]|jit_kernel|TVM-FFI|H200 上的重新调参|\[[^]]+\]\(https?://' 'large-language-model/sglang/HPC-Ops gemm_bf16xfp32 Kernel笔记：FP32权重拆两个BF16跑Tensor Core.md'
```

Expected: no output。

- [x] **Step 2: 检查必需内容**

Run:

```bash
rg -n '2\^\{-16\}|0\.5775|0\.0012|15\.5|430\.0|4\.31|0\.798|0\.802|\+5\.4%' 'large-language-model/sglang/HPC-Ops gemm_bf16xfp32 Kernel笔记：FP32权重拆两个BF16跑Tensor Core.md'
```

Expected: 每一项都至少命中一次。

- [x] **Step 3: 按 humanizer-zh 清理 AI 写作模式**

逐段检查并改写：删除“值得注意的是”“此外”“至关重要”“不仅……而且……”和泛化总结；减少粗体与机械列表；打散相同长度句子；把模糊判断替换成公式、测量值或具体限制。按直接性、节奏、信任度、真实性、精炼度五项评分，总分需达到 45/50。

- [x] **Step 4: 检查 Markdown 和 Git diff**

Run:

```bash
git diff --check
git status --short
git diff --stat
```

Expected: `git diff --check` 无输出；计划、文章重命名和 benchmark PNG 是唯一需要提交的内容。可视化伴侣产生的 `.superpowers/` 保持未暂存。

### Task 4: 提交并推送 GitHub

**Files:**
- Commit: `docs/superpowers/plans/2026-07-20-gemm-bf16xfp32-article-rewrite.md`
- Commit: rewritten article and benchmark PNG

- [ ] **Step 1: 暂存目标文件并复核 staged diff**

```bash
git add docs/superpowers/plans/2026-07-20-gemm-bf16xfp32-article-rewrite.md \
  'large-language-model/sglang/HPC-Ops gemm_bf16xfp32 Kernel笔记：FP32权重拆两个BF16跑Tensor Core，附SGLang集成与H200调优记录.md' \
  'large-language-model/sglang/HPC-Ops gemm_bf16xfp32 Kernel笔记：FP32权重拆两个BF16跑Tensor Core.md' \
  large-language-model/sglang/assets/gemm_bf16xfp32_pr30247_benchmark.png
git diff --cached --check
git diff --cached --stat
```

Expected: staged diff 包含旧文删除、新文创建、PNG 和实施计划；无空白错误。

- [ ] **Step 2: 提交文章改写**

```bash
git commit -m "docs: rewrite bf16xfp32 kernel note"
```

Expected: commit succeeds。

- [ ] **Step 3: 推送当前 `master` 到 `origin`**

```bash
git push origin master
```

Expected: remote `master` 更新到本地最新 commit。
