# `gemm_bf16xfp32` 通俗化重写设计

## 目标读者

读者知道 GEMM、BF16、FP32 和 Tensor Core 的基本含义，但没有学过浮点误差分析。正文需要先让读者建立直觉，再把严谨推导放进独立的可选阅读章节。

## 参考的作者风格

重写时参考同目录下这些原创文章：

- `分享一个DeepSeek V3和R1中 Shared Experts和普通Experts融合的小技巧.md`
- `记录下SGLang 开发，编译和Profile的几个小技巧.md`
- `SGLang 优化Triton FusedMoE 的一个新技巧​.md`

这些文章的共同写法是：先给背景或效果，用具体 shape、代码和测量值解释问题；原理部分多用「这里的意思是」「简单来说」等自然过渡，不连续堆叠抽象定义。文章保留作者的第一人称技术笔记语气，不改成教科书或论文。

## 文件

- 修改：`large-language-model/sglang/HPC-Ops gemm_bf16xfp32 Kernel笔记：FP32权重拆两个BF16跑Tensor Core.md`
- 保留：`large-language-model/sglang/assets/gemm_bf16xfp32_pr30247_benchmark.png`

## 文章结构

### 0x0. 前言

用两到三段交代：

- kernel 计算 BF16 激活乘 FP32 权重；
- Tensor Core 没有直接对应的输入组合；
- HPC-Ops 把 FP32 权重拆成两个 BF16；
- 本文只讲思路与效果，CUDA 代码另文解析。

源码使用裸 URL。

### 0x1. SGLang PR #30247 的效果

把 benchmark 提到原理之前，沿用作者其他优化文章「先看效果」的顺序。

保留内容：

- H200 上两种 LongCat router shape 的六组 microbenchmark；
- 现有双面板 PNG；
- GSM8K accuracy/invalid 数据；
- prefill `+2.8%/+3.5%/+5.4%` 和 decode 接近持平的数据；
- 单 kernel 加速不会等比例变成模型端到端加速的解释。

数字继续以 PR 当前正文为准，不改图表数据。

### 0x2. 原理：FP32 为什么可以拆成两个 BF16

主叙事不使用数值分析术语，按下面顺序展开：

1. 用位宽表说明 FP32 有 24 位有效精度，BF16 有 8 位。
2. 用现有 PyTorch 实验说明直接转 BF16 的最大绝对误差是 `0.5775`。
3. 用十进制 `1.234567` 的 high/low 拆分例子建立直觉。
4. 给出三行权重拆分代码：

```python
scale = 1 / 256
w_high = w.to(torch.bfloat16)
w_low = ((w - w_high.float()) / scale).to(torch.bfloat16)
```

5. 主文只保留三个关键关系：

```text
w ≈ w_high + w_low / 256
y = x @ w_high.T + (x @ w_low.T) / 256
权重误差量级：2^-8 -> 2^-16
```

6. 用 `0.5775 -> 0.0012` 的实验结果说明精度改善。
7. 解释代价：做两次 BF16 GEMM；两份 BF16 权重的总字节数等于一份 FP32；适合权重静态、BF16 Tensor Core 明显更快的场景。
8. CUDA 细节不展开，只保留上游 `.cu` 文件和逐行解析文章的裸 URL。

### 0x3. 公式推导（可选阅读）

章节开头明确写：

> 本节由 Codex GPT-5.6 辅助整理和检查。前面的原理理解不依赖这一节，想继续看误差界再往下读。

推导只保留一条主线：

```text
u = 2^-8
w_high = w(1 + delta_high), |delta_high| <= u
r = w - w_high, |r| <= u|w|
w_low = (r / scale)(1 + delta_low), |delta_low| <= u
w_hat = w_high + scale * w_low = w + r * delta_low
|w_hat - w| <= u^2|w| = 2^-16|w|
```

随后用一小段话说明：`2^-16` 描述权重重建误差，不表示整个 GEMM 逐位等同于 FP32；FP32 累加和正负抵消仍会影响最终输出。

删除以下内容：

- Sterbenz lemma；
- `gamma_k` 累加误差公式；
- 点积和矩阵逐元素误差界的完整推导；
- 连续多屏公式。

## 链接与文风

- 外部资料全部写裸 URL，不使用 Markdown 超链接。
- 图片继续使用仓库内相对路径。
- 保留 `0x0` 到 `0x3`，不新增 `0x4`、`0x5`、`0x6`。
- 不出现 SGLang JIT、TVM-FFI、kernel 移植或 H200 调参叙事。
- 每段只解释一个问题；公式前后都用口语化短句翻译含义。
- 不写宣传式结论、机械排比和泛化总结。

## 验收标准

- 正文约 180–220 行，明显短于当前 306 行。
- benchmark 表格和 PNG 数字保持不变。
- 主叙事只保留三个容易理解的关键关系。
- 完整误差推导只出现在 `0x3`，并明确标注由 Codex GPT-5.6 辅助。
- 普通 CUDA 读者跳过 `0x3` 仍能解释 high/low 拆分和两次 GEMM 的用途。
- 所有 URL 均为裸链接，且没有旧 JIT/调参内容。
