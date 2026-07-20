# `gemm_bf16xfp32` 原理文章改写设计

## 目标

把现有文章从“SGLang JIT 集成与 H200 调参记录”改成一篇以数学原理为主的独立文章。正文应让普通 CUDA 读者看懂 FP32 权重为什么可以拆成两个 BF16、误差为什么从一阶 BF16 舍入降到二阶舍入，以及这种分解如何换取 Tensor Core 吞吐。

## 文件与标题

- 原文件：`large-language-model/sglang/HPC-Ops gemm_bf16xfp32 Kernel笔记：FP32权重拆两个BF16跑Tensor Core，附SGLang集成与H200调优记录.md`
- 新文件：`large-language-model/sglang/HPC-Ops gemm_bf16xfp32 Kernel笔记：FP32权重拆两个BF16跑Tensor Core.md`
- 新增图片：`large-language-model/sglang/assets/gemm_bf16xfp32_pr30247_benchmark.png`

## 文章结构

### 0x0. 前言

说明 `bf16` 激活乘 `fp32` 权重的应用背景，并给出 HPC-Ops 原始实现的裸 URL。正文不再叙述把 kernel 拷入 `sglang/jit_kernel`、TVM-FFI JIT 编译或 H200 配置扫描。

### 0x1. 问题从哪里来

解释 FP32 与 BF16 的符号位、指数位和尾数位。说明 Hopper Tensor Core 没有直接执行 BF16 输入乘 FP32 输入的路径；直接把权重降成 BF16 虽然能用 Tensor Core，却会引入约 `2^-8` 量级的相对舍入误差。结合 router top-k 选择解释为什么 logits 的小偏差可能改变专家选择。

### 0x2. 两段 BF16 分解的数学原理

依次完成以下推导：

1. 定义 BF16 单位舍入误差 `u = 2^-8`。
2. 定义高位 `w_h = round_bf16(w)`，残差 `r = w - w_h`，得到 `|r| <= u|w|`。
3. 取 `s = 2^-8`，定义 `w_l = round_bf16(r/s)`。因为除以 `s` 等价于乘 `256`，正常数范围内只改变指数，不损失有效位。
4. 重建 `w_hat = w_h + s w_l`。若第二次舍入写成 `w_l = (r/s)(1 + delta_l)`，则 `w_hat - w = r delta_l`，所以 `|w_hat-w| <= u^2|w| = 2^-16|w|`。
5. 对点积给出绝对前向误差界 `|x^T(w_hat-w)| <= u^2 sum_i |x_i w_i|`，并说明发生抵消时不存在统一的小相对误差保证。
6. 说明实际 kernel 还包含两个 BF16 GEMM 的 FP32 累加误差和最后一次 FP32 融合舍入，因此 `2^-16` 是权重表示误差的主项估计，不是整个 GEMM 的逐位正确承诺。

保留可复现的 PyTorch 数值例子，对比直接 BF16 权重和双 BF16 分解的最大绝对误差。随后说明两次 Tensor Core GEMM、权重存储总量与原 FP32 相同、计算量翻倍等代价。

CUDA 代码不在本文展开。原始 `.cu` 文件和逐行解析文章均以裸 URL 给出。

### 0x3. PR #30247 benchmark

数据以 2026-07-20 的 PR 正文为准：

- H200 microbenchmark：LongCat-Flash Chat 与 Lite 两个 router shape，在 `m=64/512/8192` 下对比 `hpc.gemm_bf16xfp32` 和 FP32 MM。
- 双面板图：左侧画两种 shape 的 kernel speedup，右侧画 LongCat-Flash-Lite-FP8 H200 TP1 在 `bs=1/16/64` 下的 prefill 输入吞吐提升。
- 模型级验证：GSM8K 1319 题的 accuracy 与 invalid rate；decode 吞吐只报告接近持平，不把噪声写成性能回退。

文章须明确区分 microbenchmark 与模型端到端收益，不把 1.56–4.31 倍的单 kernel 加速写成整网加速。

## 图表设计

输出一张适合公众号横向排版的 PNG。左图使用两组柱状条展示 Chat/Lite speedup；右图使用柱状条展示 prefill 输入吞吐增幅。图内使用中文说明，保留必要的英文技术词，数据标签直接标在柱顶，避免读者来回对照图例。

## 链接与文风

- 正文不使用 `[文字](URL)` 形式，外部资料全部写成裸 URL。
- 图片使用 Markdown 图片语法引用仓库内 PNG；它不是跳转链接。
- 删除聊天式开场、空泛总结、宣传语、机械三段式与过量粗体。
- 公式前后用短句解释每个符号的物理意义。数学严谨性优先，但不堆论文术语。
- 对实测结果只作数据能够支持的判断，不使用“完全等价”“精度没有任何区别”等绝对表述。

## 验收标准

- 文件已按新标题重命名，旧文件不存在。
- 正文只保留 `0x0` 到 `0x3`，不存在旧 0x4、0x5、0x6 内容。
- 不出现 `sglang jit_kernel`、`TVM-FFI`、内嵌 kernel 或 H200 配置扫描叙事。
- HPC-Ops 源码、逐行解析文章和 PR #30247 均使用裸 URL。
- 数学推导包含标量误差界、点积误差界及适用条件。
- benchmark 数字与 PR 当前正文逐项一致。
- PNG 图中的数值与文章表格一致。
- 中文通过 humanizer-zh 检查，读起来像工程师写的技术笔记，而不是生成式摘要。
