> 原文：SOTA Normalization Performance with torch.compile
> 原文地址：<https://pytorch.org/blog/sota-normalization-performance-with-torch-compile/>
> 作者：Shunting Zhang, Paul Zhang, Elias Ellison, Markus Hoehnerbach, Jason Ansel, Natalia Gimelshein
> 发布时间：2026 年 4 月 8 日
>
> 说明：PyTorch 官网页脚标注 All rights reserved。本文是面向公众号阅读的中文翻译/译述稿，保留原文技术主线、关键数字和图，但不是逐字全译。

# 【博客翻译·译述】PyTorch 用 torch.compile 把 LayerNorm 和 RMSNorm 做到 SOTA

## 1. 简介

LayerNorm 和 RMSNorm 是深度学习里非常基础的 normalization 方法。它们通过归一化输入值，让模型训练过程更平滑。PyTorch 这篇文章关注的问题很直接：在 NVIDIA H100 和 B200 上，`torch.compile` 生成的 LayerNorm/RMSNorm kernel 能不能追上手写 SOTA kernel？

文章的结论是可以。通过改进 TorchInductor 的 forward reduction heuristic、backward fused reduction、split-size autotuning 和 software pipelining，`torch.compile` 在逐 kernel 维度上可以接近甚至超过 Quack / Liger 等手写 kernel baseline。更重要的是，`torch.compile` 还能自动融合周围的 pointwise/reduction op，从端到端角度看有更大的优化空间。

## 2. Forward：LayerNorm 和 RMSNorm

### LayerNorm

LayerNorm 最早来自论文 <https://arxiv.org/abs/1607.06450>。它会对输入做均值和方差归一化，并使用可学习参数 `gamma`（weight）和 `beta`（bias）做缩放和平移。

![LayerNorm 公式和计算流程](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-1/53383e2b-a6af-43c3-808d-966d4f77575e.png)

图 1：LayerNorm 的基本计算形式。

### RMSNorm

RMSNorm，也就是 root mean square norm，是 LayerNorm 的后续变体，论文见 <https://arxiv.org/abs/1910.07467>。它不再围绕均值中心化，而是使用输入平方和对应的 RMS 进行归一化。RMSNorm 仍然使用 `gamma`（weight）作为可学习缩放参数，但不再有 bias。

![RMSNorm 公式和计算流程](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-2/8b9f6d4a-0680-4ed8-b9bc-5ff118f4d0b6.png)

图 2：RMSNorm 的基本计算形式。

LayerNorm 和 RMSNorm 的 forward pass 结构很像：通常是在连续维度上做一次 reduction，再接若干 pointwise op。RMSNorm 通常更轻一点，因为 FLOPs 更少，而且没有 bias。文章后面的 benchmark 也会在 LayerNorm 和 RMSNorm 间交替展示，因为两者 kernel 形态接近。

## 3. SOTA baseline：Quack

Quack 是 Tri Dao 的高性能 CuteDSL kernel 库：<https://github.com/Dao-AILab/quack>。它包含一组高度优化的 reduction kernel。Quack README 中展示了 H100 上的结果：对这些 reduction kernel，Quack 明显快于当时的 `torch.compile`。原始对比里，`torch.compile` 通常只有 Quack 大约 50% 的性能。

![Quack README 中的 baseline 对比](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/f32721b1-3a70-4944-a232-b7214204f980.png)

图 3：Quack README 中展示的 baseline。PyTorch 文章以 Quack 作为 SOTA 手写 kernel baseline 来衡量 `torch.compile`。

## 4. torch.compile 生成的 forward kernel

下面这张图展示了 `torch.compile` 为 LayerNorm forward 生成 kernel 时的大体逻辑。RMSNorm forward 也采用类似思路。这里假设输入的 reduction dimension，也就是 `rnumel`，是连续维度。在 TorchInductor 中，这类 reduction 被称为 Inner reduction。

![torch.compile 生成 LayerNorm forward kernel 的基本逻辑](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/d4fc350c-5ef3-4d49-8fe2-88ac87f60b22.png)

图 4：`torch.compile` 生成的 LayerNorm forward kernel 逻辑示意。

图看起来可能有点绕，但核心步骤其实很简单：

- 为输入 `X` 的每一行维护大小为 `R_BLOCK` 的 partial sums。
- 用 partial sums 计算 mean 和 variance。
- 根据 LayerNorm 公式对 `X` 做 elementwise 计算。
- 写出 elementwise output。
- 如果开启 `elementwise_affine=True` 且 backward 需要梯度，则保存 mean 和 variance 给 backward 用。

如果 `R` 小于某个 heuristic 阈值，比如 1024，Inductor 会生成 persistent reduction。此时不再需要沿着 `r` 维循环，而是直接进入 mean 计算。

在最初对比中，`torch.compile` 生成的 RMSNorm forward 确实明显慢于 Quack，在 H100 和 B200 上都能复现这个问题。后续改进主要来自 autotuning 和由 autotuning 反推出来的 Inductor 默认 heuristic 调整。

文章提到几项关键改动：

- benchmark 时插入 `torch._dynamo.reset()`。这样可以避免 `torch.compile` 自动假设 dynamic shapes。之前每个 shape 调一次 compile，会让 compiler 推断成动态形状，从而影响结果。
- 修正默认 autotune config 的选择。默认 heuristic 在 H100 和 B200 上会选到次优配置；`mode='max-autotune'` 可以缓解，但更好的默认 heuristic 更重要。
- 放大 inner reduction 的 `RBLOCK`。
- 对较小 reduction（`numel <= 2048`）放大 persistent reduction 中的 `XBLOCK`。
- 根据 reduction dimension 减小 `num_warps`。之前 `num_warps` 往往偏大，影响 peak vectorization。

这里的关键是 vectorization。LayerNorm/RMSNorm 这类 kernel 通常是 memory-bound。要打满内存带宽，需要足够多 bytes in flight。Blackwell 的内存带宽更高，对这点更敏感。

## 5. Forward benchmark 结果

文章对比了 `torch.compile 2.11` 和 Quack（2026 年 3 月 24 日 trunk）在 Quack benchmark shape 以及一些真实场景常见 shape 上的结果，特别是 large M、small N 的情况。结论是：经过上述改进后，`torch.compile` forward 通常已经能和 Quack 持平。

![torch.compile 与 Quack 的 forward benchmark 对比 1](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-2/a62837b2-d670-4371-af20-e9ff704d512a.png)

图 5：`torch.compile 2.11` 与 Quack 的 forward benchmark 对比。

![torch.compile 与 Quack 的 forward benchmark 对比 2](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/cced9172-f618-4aff-8f5e-f5b87cc0b973.png)

图 6：更多 forward benchmark shape 上的结果。

仍然存在两类 regression：

- `N=384` 上有小幅退化。原因是 Triton 很难干净地表达非 2 次幂 block size。
- H100 上 very large N 会有较大退化。原因是 Triton 目前无法表达 distributed shared memory。

## 6. Backward：为什么更难

LayerNorm/RMSNorm backward 比 forward 复杂很多。至少要计算两个梯度：输入梯度 `dX` 和 weight 梯度 `dW`；LayerNorm 还可能要计算 bias 梯度 `dB`。

从性能角度看，这些 gradient 计算需要沿着 incoming gradient `dY` 的两个维度做 reduction。`dY` 是 forward 输出的梯度。

朴素做法是拆成多个 kernel：一个 kernel 算 `dX`，另一个 kernel 算 `dW` 和 `dB`。当 reduction dimension 很大时，有时这也是难以避免的选择。但这种拆分会让同一份输入 `dY` 被两个 kernel 分别读取。归一化 kernel 本来就 memory-bound，重复读 `dY` 会显著增加 latency。

![拆分 backward reduction 时的额外读带宽问题](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-1/3a07b1e0-4002-4266-842f-02d70b43b6fe.png)

图 7：分开计算 `dX` 和 `dW/dB` 会重复读取 `dY`，对 memory-bound kernel 很伤。

## 7. Fused Reductions

对于合理 shape，如果 `numel` 不太大，一行可以比较舒服地放进一个 thread block，通常 `<= 16384`，就可以写一个 fused kernel。它既做 `dW/dB` 的 reduction，又在每一行里同步做 `dX` 所需的列方向 reduction，而且不会把 shared memory 或 register 使用量撑爆。

这种 fusion 思路并不新。Liger、Meta 的 fused semi-persistent normalization backward，以及 Quack 的 CuteDSL fused kernel，都有类似工作。

在 Inductor 中，reduction 被分成不同类型：

- `INNER` reduction：沿 stride=1 的连续维度做 reduction。
- `OUTER` reduction：沿剩余维度做 reduction。

按照这个定义，fused backward kernel 等价于在同一个输入 tensor 上同时做 `INNER` 和 `OUTER` reduction。其中 `INNER` reduction 对应 `dX`，沿连续维度；`OUTER` reduction 对应 `dW/dB`。

真实模型里经常出现 `xnumel`，也就是 batch 维，比 `rnumel` 大很多的情况。这时更好的做法通常是先沿 X 维处理 partial sums，最后再对 partial sums 做一次 `torch.sum`，以提高并行度。Triton layernorm tutorial 也展示了 split reduction，但它使用 lock 和 atomic，并且让单个 thread block 负责单独的 row。对大 batch dimension 来说，这个做法性能不好，也可能带来数值不一致。

Inductor 目前也有 split reduction 能力，会为 partial sums 分配 workspace tensor，但不使用 atomic。它会保证一个 CTA 处理多行，并写入 workspace tensor 中唯一的位置。

![Split reduction 与 workspace tensor](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/bc265ce3-0bc4-499f-975e-cd9535d8794f.png)

图 8：split reduction 的基本思路：先把 partial sums 写入 workspace，再做最终 reduction。

## 8. Inductor 自动生成 fused norm backward：MixOrderReduction

把 fused reduction 和 split reduction 结合起来，TorchInductor 就可以自动生成高性能 normalization backward kernel。更重要的是，由 compiler 生成这类 kernel 后，周围的 op 也有机会被自动 fusion，autotuning 空间也更大。

这件事的核心难点是：同一个输入 tensor 上存在 reduction order 不同的两个 reduction，要把它们融合起来。文章把这个优化称为 `MixOrderReduction`。

对一个 `[M, N]` shape 的输入，生成的 kernel 大体会做：

- 按 `SPLIT_SIZE` 把 M 维切成多个 chunk。
- 每个 chunk 在 workspace tensor 中有一行，用来保存 `OUTER` reduction 的 partial result，例如每列的 partial sum，也就是 `dW/dB` 的部分结果。
- 对 chunk 内的每一行做常规 `INNER` reduction，例如 `dX` 所需的一整行求和。
- 把加载的 row 和 workspace 中对应 row 结合起来，更新 `OUTER` reduction 的 partial result。

之后还需要一个额外 reduction，把 workspace tensor 中的 partial reduced results 合成最终 `OUTER` reduction 结果。这个额外 kernel 的输入 tensor 小很多，所以单独执行不会带来很大开销。

在 Inductor codegen 里，识别到 mix order reduction pattern 后会做几步变换：

- 对 `OUTER` reduction kernel，把 `reduction` 和 `store_reduction` node 替换成新的 `partial_accumulate` node。这个 node 跟踪被 reduction 的值、reduction 类型等信息。这一步会把 `OUTER` reduction kernel 转成一个 pointwise kernel，记作 `PW1`。
- 对变换后的 pointwise kernel `PW1` 做 loop reorder，得到 `PW2`。
- 此时 `PW2` 和 `INNER` reduction 具有相同 loop order，于是可以 fuse。

## 9. Split-size autotuning

`SPLIT_SIZE` 对 MixOrderReduction kernel 的性能非常关键。文章给了一个很直观的例子：在 shape `(1152000, 384)`、dtype 为 `bfloat16` 的 H100 上，Liger RMSNorm backward 默认 split size 只能达到 `0.417 TB/s`；把 `SPLIT_SIZE` 缩小 32 倍后，可以达到 `1.912 TB/s`。

![不同 split size 对 MixOrderReduction 性能的影响](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/e1225485-61d6-4ccd-9461-977eabd98b5f.png)

图 9：不同 shape 和 split size 下的性能曲线，测试 dtype 为 `torch.bfloat16`。

图里的结论很清楚：

- 不合适的 split size 会导致超过 2 倍性能退化。
- 曲线大致呈抛物线形状。沿着 2 倍或 1/2 倍方向扩展搜索，直到找到最大值，是一个很有效的 autotuning 策略。

Inductor 原有 split-reduction 也会为 outer reduction 选择 split size，但那套 heuristic 与 MixOrderReduction 的问题不完全匹配，可能选到很差的值。因此，MixOrderReduction 不再直接复用原 split-reduction 的 split size，而是使用自己的 heuristic 或 autotuning 机制。

## 10. Software Pipelining

在追 backward kernel 的峰值带宽时，团队还发现 software pipelining，也就是 prefetch load，非常有帮助。

过去通常只有 GEMM、Attention 这类 compute-intensive workload 才会做 pipelining。pointwise/reduction 这类 memory-bound kernel 一般不需要，Inductor 里也没有针对它们做 `num_stages` autotuning，Liger 示例里也没有类似机制。

但 Quack kernel 中存在一定程度的 prefetching。因此团队把 `num_stages` 加入 Inductor kernel 的 autotuning 参数。对某些 shape，尤其是 large M、small N，MixOrderReduction 上可以看到显著收益，最高可达 20%。

![Software pipelining / prefetching 对性能的影响](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/c54dbd78-d620-4b7e-88f6-df12c9aab509.png)

图 10：为 MixOrderReduction 加入 `num_stages` autotuning 后，部分 shape 有明显收益。

## 11. Backward benchmark 结果

文章最后比较了 MixOrderReduction 与 PyTorch eager、旧版 compile，以及 Quack、Liger 等 OSS baseline。benchmark 运行在一台 750W B200 机器上，CUDA 12.9，时间是 2025 年末。

![MixOrderReduction 与 eager、旧 compile、Quack、Liger 的 RMSNorm backward benchmark](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-1/282cb6f1-8fde-4d63-8249-11f9f9016a2d.png)

图 11：RMSNorm backward benchmark。

文章观察到：

- 带 MixOrderReduction 的 `torch.compile` 比 eager 快 `17.07x`。
- 不带 MixOrderReduction 的 `torch.compile` 只比 eager 快 `9.93x`。
- 带 MixOrderReduction 的 `torch.compile` 比 Liger 快 `1.45x`，比 Quack 快 `1.34x`。

LayerNorm 也给出了类似 benchmark。由于 LayerNorm 和 RMSNorm 的 kernel 结构相近，趋势基本一致。

![LayerNorm backward benchmark](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-2/8800f3da-e240-4766-9f00-6fb751e7f475.png)

图 12：LayerNorm backward benchmark。带 MixOrderReduction 的 `torch.compile` 相比旧 compile baseline 接近 2 倍加速，也更接近峰值内存带宽。

## 12. 结论

这篇文章的核心结论是：`torch.compile` 已经能在 H100 和 B200 上为标准 LayerNorm/RMSNorm shape 生成接近 SOTA 的 forward 和 backward normalization kernel，并且可以和 Quack 这类手写 kernel baseline 正面对比。

forward 侧，收益主要来自更好的 reduction block、persistent reduction heuristic、`num_warps` 选择和 autotuning。backward 侧，关键是把同一输入上的不同 reduction order 融合起来，也就是 MixOrderReduction，再配合 split-size autotuning 和 software pipelining。

从 LLM serving 和训练系统角度看，这个结果也挺有意思。LayerNorm/RMSNorm 属于典型 memory-bound 小算子，过去很容易依赖手写 kernel 追性能。但 compiler 方案如果能自动生成接近手写性能的 kernel，同时继续融合周围 pointwise/reduction op，就可能在端到端图优化里获得更好的收益。对 SGLang 这类推理系统来说，后续在 profile 里看到 norm 相关瓶颈时，也值得关注 `torch.compile` / Inductor 在这些标准 pattern 上的进展。

## 参考链接

- PyTorch 原文：<https://pytorch.org/blog/sota-normalization-performance-with-torch-compile/>
- LayerNorm 论文：<https://arxiv.org/abs/1607.06450>
- RMSNorm 论文：<https://arxiv.org/abs/1910.07467>
- Quack：<https://github.com/Dao-AILab/quack>
- Liger Kernel：<https://github.com/linkedin/Liger-Kernel>
