![Towards Free Normalization](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/b8805768-a3f9-4e21-b343-dad44868c3db.png)

> 原文：Towards Free Normalization: Fusing Normalization into GEMM and Attention Kernels
> 原文地址：PyTorch Blog（https://pytorch.org/blog/towards-free-normalization-fusing-normalization-into-gemm-and-attention-kernels/）
> 作者：Jacky (Junqing) Zhou、Hongtao Yu、Jackie (Jiaqi) Xu、Menglu Yu、Ethan Che、Han Xu、Darren Liu、Peng Chen、Daohang Shi、Max Leung
>
> 说明：本文是面向公众号阅读的中文翻译·译述稿，保留原文的技术主线、实现约束和性能数据，但不是逐句直译。外部链接统一写成“文本（链接）”形式，正文图片已转存到 mdnice 图床。

# 【博客翻译·译述】让归一化接近“免费”：把 Norm 融合进 GEMM 与 Attention Kernel

## TL;DR

LayerNorm 和 RMSNorm 的 FLOPs 不多，但它们需要单独读写整行 Tensor，训练时还会增加 kernel launch 和中间结果保存。在 Meta 的 Kunlun 广告推荐模型中，归一化约占训练总延迟的 20%；在典型 LLM 中，这一比例也可能接近 10%。

本文讨论三种融合方法：

- **Lazy Pre-Norm**：利用 RMSNorm 的线性结构，把输入侧归一化推迟到 GEMM epilogue，最多隐藏约 98% 的 RMSNorm 延迟。
- **Multi-CTA Norm**：让一个 CTA cluster 协作完成长行归一化，再和 GEMM epilogue 融合。前向最多隐藏约 92% 的归一化延迟，反向最多隐藏约 71%。
- **FlashNormAttention**：把 Attention 前后的 LayerNorm、RMSNorm 和 residual 一并纳入 attention kernel。原文在 B200 上测得最高约 35% 的前向加速和 18% 的反向加速。

这三种方案的共同思路，是把归一化的访存和归约藏到相邻计算密集型 kernel 的流水线里。代价也很明确：tile 排布、CTA 调度、共享内存、寄存器和前反向的数据流都要重新设计。

原文所有性能测试均使用 BF16，在 Meta 数据中心的 NVIDIA B200 上完成，GPU 功耗上限为 750 W。除非特别说明，测试关闭了归一化中的逐元素 affine 参数。

## 1. 归一化为什么会成为瓶颈

LayerNorm 和 RMSNorm 都要沿最后一个维度做归约。以 RMSNorm 为例，kernel 先计算一整行平方和，再得到倒数均方根，最后把整行乘上这个缩放系数：

```python
rstd = rsqrt(mean(x * x) + eps)
y = x * rstd
```

单看计算量，这部分并不大。问题在于它通常是 memory-bound：输入至少要读一次，输出还要再写一次；训练时还可能保存统计量，反向再发起相应 kernel。模型里的矩阵乘、Attention 已经很快后，这些短小 kernel、额外显存流量和 launch 开销就会变得显眼。

最自然的想法，是把 Norm 融到前一个 GEMM 的 epilogue，或者融到后一个 GEMM 的 prologue。但这里有一个 tile 维度冲突。

![Normalization and GEMM tiling mismatch](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/fba55352-61d6-428c-83a8-fde6087fb3ce.png)

归一化需要看到一整行，也就是完整的最内层维度；GEMM 则通常同时在 M、N 两个维度切 tile，每个 CTA 只负责输出矩阵的一小块。一个 CTA 算出的局部 tile 不足以得到整行统计量。

## 2. 直接把 Norm 塞进 GEMM，为什么效果不好

一种直接做法是强制 GEMM 的 `tile_n` 覆盖整个 N。这样一个 CTA 拿到了完整输出行，可以在 epilogue 里完成归一化。

![Naive full-N GEMM fusion](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/f2bfd311-5c44-4308-9969-0dfa18703321.png)

这个设计只适合较小的 N。原文给出了一笔 Blackwell 上的共享内存预算：每个 SM 约有 228 KB shared memory，BF16 GEMM 采用双缓冲，假设 `tile_m=32`、`tile_k=32`，可支持的 `tile_n` 介于 512 和 1024 之间。考虑实际 kernel 通常选择 2 的幂，最大可用 N 大约是 512。

强行把更大的 N 放进一个 CTA，会降低 occupancy，挤压流水线 buffer，也破坏 GEMM 原本合适的 tile 形状。下面的实测反映了这个问题。

![Naive fusion benchmark](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-2/96cfec8c-4c89-40a5-8e7c-f1e9bfb41e7c.png)

图中的指标是“融合后隐藏了多少独立 LayerNorm 延迟”。小 shape 可以隐藏 17% 到 32%；当 K、N 增大到 256 一带时，收益消失，部分配置甚至出现明显回退。`K=N=256` 时，结果为 -64.38%。

所以后续方案都没有继续要求单个 CTA 同时负责完整行和完整 GEMM，而是重新安排计算顺序或让多个 CTA 协作。

## 3. Lazy Pre-Norm：把 RMSNorm 推迟到 GEMM 之后

先看一个常见结构：输入 `A` 经过 RMSNorm 后，与权重 `B` 做矩阵乘。

```python
C = rmsnorm(A) @ B
```

忽略 affine 参数时：

```python
rstd(A) = rsqrt((A ** 2).sum(dim=-1) / A.shape[-1] + eps)
rmsnorm(A) = A * rstd(A)[:, None]
```

`rstd` 对每一行是一个标量，因此可以从矩阵乘左侧移到输出侧：

```text
(A * rstd[:, None]) @ B = (A @ B) * rstd[:, None]
```

这就是 Lazy Pre-Norm 的核心。GEMM 仍然直接读取未经归一化的 `A`。Tensor Core 沿 K 维累加时，另一个 warp 同步累加 `A` 的平方和；等 GEMM 完成后，再在 epilogue 中用每行 `rstd` 缩放输出。

![Lazy Pre-Norm warp specialization](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/f00435d5-adbc-4ae4-9d8e-28b3fee22b56.png)

这样既不要求 GEMM 的 N tile 覆盖整行，也省去了归一化结果的物化。Norm 的输入读取、平方和归约以及输出写回，大部分都被 GEMM 主循环和 epilogue 吸收了。

![Lazy Pre-Norm benchmark](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-2/851d04a3-36cb-43f9-b0a3-3230015f5f58.png)

在图中的测试范围内，Lazy Pre-Norm 隐藏了 41% 到 98% 的 RMSNorm 延迟。其中 `K=2048、N=512` 时达到 98.01%；更大的 N 会增加 GEMM epilogue 和数据移动压力，收益随之下降。

这条路很有效，但适用范围有限：

- 逐元素 affine scale 沿列变化，不能像行标量 `rstd` 那样直接移到 GEMM 输出侧。
- LayerNorm 还要减去均值，无法直接套用同一个恒等式。
- 前向不再物化归一化后的 `A`，反向所需数据要重新组织或重算。

因此，Lazy Pre-Norm 更适合无 affine 参数的 RMSNorm 前向路径。

## 4. Multi-CTA Norm：用 CTA cluster 处理长行

更通用的办法是把一行拆给多个 CTA。原文受 Quack 的设计启发，使用 CTA cluster 和 distributed shared memory（DSMEM）交换局部归约结果。

![Multi-CTA normalization split](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-1/005f6c67-7c2c-46b9-9f3f-6a9eed6f08ab.png)

每个 CTA 负责同一行的一个 N 分片，先计算局部平方和或局部均值；cluster 内再合并这些很小的统计量，得到整行结果。大块 Tensor 数据留在各自 CTA，本身不需要跨 CTA 复制。

![Multi-CTA reduction algorithm](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-1/336176f0-63f0-4e2b-a7e9-bd144935b798.png)

这套归约被放进 GEMM epilogue。相邻 CTA 必须拥有相同的 `m_tile`，并覆盖不同的 `n_tile`，这样它们才是在协作处理同一批输出行。

CTA cluster 也给 GEMM 调度带来一些限制：

- paired-CTA 路径更难安排。
- 不能继续使用部分 tile super-grouping 策略。
- 单个 CTA 能高效处理的 N 仍约为 512。Blackwell 上若使用可移植的 8-CTA cluster，整体 N 上限约为 4096。

原文的实现基于 TLX warp-specialized GEMM。测试 shape 来自广告推荐模型，M 固定为 256K，K、N 主要位于数百到数千的范围。作者只展示到 K、N 为 2048，因为 N 达到 4096 后，独立 Norm 在总耗时中的占比已经低于 5%。

![Multi-CTA Norm forward benchmark](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-1/2b137784-1ab3-4b8d-80b2-c0b731216ecc.png)

前向结果中，RMSNorm 融合隐藏了约 48% 到 72% 的归一化延迟；LayerNorm 为约 50% 到 92%。最高值出现在 LayerNorm 的某些 shape，达到 91.61%。

### 4.1 反向不能简单照搬前向

前向中，前一个计算密集型算子输出后接 Norm，适合做 epilogue fusion。到了反向，数据流方向改变，同一段 Norm backward 可能落在 GEMM prologue 一侧。这样做通常更难：统计量必须在 GEMM 开始前准备好，难以隐藏延迟。

原文采用了 fusion regrouping。它不要求前向和反向都绑定同一对算子，而是分别为两个方向选择位置合适的相邻 GEMM，让 Norm 在两边都尽量落到 epilogue。

![Forward and backward fusion regrouping](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-2/a624230b-6e3a-400f-8849-5d9ac8508c94.png)

这种安排需要从完整计算图看依赖，不能只在单个 PyTorch op 边界上做局部替换。

![Multi-CTA Norm backward benchmark](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-2/6b277d59-851d-4afd-b4fc-5f65c365e7a2.png)

反向测试中，RMSNorm 融合隐藏了约 31% 到 69% 的原始 Norm backward 延迟，LayerNorm 为约 37% 到 71%。最高值为 70.71%。相比前向，反向的依赖和归约更多，多数配置仍能省掉大半 Norm 时间。

## 5. FlashNormAttention：把 Norm 和 Attention 放进一个 kernel

前两种方法围绕 GEMM 展开。原文还把同样的思路用于 Kunlun 模型里的 GDPA（Generalized Dot Product Attention）。这一结构在 Attention 前后穿插了 LayerNorm、RMSNorm 和 residual：

![GDPA block before fusion](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-1/00f9e1a2-a7f3-4558-aace-e15ed27b04a1.png)

如果每一步都单独执行，Attention 本体之外会出现多次中间 Tensor 读写。FlashNormAttention 将这些步骤并入一个 attention kernel，让中间值尽量留在寄存器、shared memory 或 Tensor Memory（TMEM）中。

原文的实现面向一个短 KV 场景：Q 的序列长度约为数千，K/V 只有约 128。常规 FlashAttention 往往按 Q 方向切 grid、在 kernel 内循环 K/V；这里反过来按 KV 方向布置 grid，并在 CTA 内循环 Q。因为 K/V 很短且不再切片，这种交换不会破坏 softmax 的完整性。

### 5.1 多个 CTA 如何协作

Multi-CTA cluster 中的 CTA 共享同一个 batch index，分别处理不同 attention head。跨 head 汇总局部统计量后，就可以完成 Q 侧 LayerNorm。

这里同样只传递小型归约结果。大块 Q、K、V 和 attention 中间值由各 CTA 自己维护，避免让 DSMEM 成为新的带宽瓶颈。

### 5.2 Buffer 复用与 warp 分工

融合更多算子后，资源压力会迅速上升。原文用了几项针对 Blackwell 的手段：

- 复用生命周期不重叠的 shared memory buffer。
- 用 TMEM 和 Tensor Core 累加 LayerNorm 所需的 Q residual。
- 对寄存器数据继续切 sub-tile，缩短单段代码里的 live range。
- 在计算前预取 Q，减少流水线空泡。

执行流水线原来有四个分区：load、MMA、activation 和 epilogue。融合 LayerNorm 后增加第五个分区。activation 使用 8 个 warp（0～7），LayerNorm 使用 4 个 warp（8～11），各分区通过生产者—消费者依赖并行推进。

![FlashNormAttention warp-specialized pipeline](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-2/85429f38-ebe6-4dbd-8c22-e95f0d922ee2.png)

### 5.3 前向性能

前向 benchmark 固定 K/V 长度为 128，Q 使用平均稀疏度 0.5，batch size 为 768，head dimension 为 128；测试改变 Q 的最大长度和 head 数量。

![FlashNormAttention forward benchmark](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-1/03b19fa6-9a51-4d84-899c-3c023dd1f1c8.png)

不同配置下，端到端 kernel 延迟降低约 16% 到 35%。最大值 35.39% 出现在 Q 最大长度为 3000、head 数为 2 的配置。head 数增多后，Attention 主体计算占比提高，Norm 融合带来的相对收益会下降。

### 5.4 反向：重算换显存流量

反向需要 LayerNorm 的输入与统计量。前向只保存一维的均值和方差，反向则重新构造必要数据：

```text
rmsnorm_out = kernel_out - q
rmsnorm_in  = rmsnorm_out / rstd
```

随后重算 `ln(q)`，避免前向把多个大中间 Tensor 写回 HBM。

反向 kernel 把 warp 分成 MMA、activation、load 和 reduction 四组，LayerNorm backward 放在 reduction 分区。由于同时存在 attention backward、归一化归约和 residual，shared memory、TMEM 都采用了更激进的复用，tile 也比前向更小。

一个很有意思的取舍是 outer residual。实现使用 `TMA_REDUCE_ADD`，把 shared memory 中的结果直接累加到 HBM。它会增加一点显存流量，但在计算和 pipeline stall 已成为瓶颈时，可以减少寄存器占用与额外调度，整体反而更快。

![FlashNormAttention backward benchmark](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/e1feca02-bcbd-4165-a498-3f52fcbf7ba1.png)

反向端到端延迟降低约 4% 到 18%，最高为 18.08%。收益小于前向，原因是反向数据依赖更复杂，部分中间量也需要重算和归约。

## 6. 三种融合方法该怎么选

| 方法 | 适用场景 | 主要收益 | 主要限制 |
| --- | --- | --- | --- |
| Lazy Pre-Norm | 无 affine 的 RMSNorm + 后续 GEMM | 最多隐藏约 98% RMSNorm 延迟 | 不直接适用于 LayerNorm，反向处理更复杂 |
| Multi-CTA Norm | 大 N 的 GEMM epilogue，RMSNorm/LayerNorm | 前向最多约 92%，反向最多约 71% | 受 cluster 大小和 GEMM tile 调度约束 |
| FlashNormAttention | Attention 前后存在 Norm 与 residual | 前向最多约 35%，反向最多约 18% 的 kernel 加速 | 强依赖具体 Attention 数据流与硬件资源规划 |

这些数字来自不同 benchmark，不能横向当成同一个基线比较。前两项统计的是“独立 Norm 延迟被隐藏的比例”，FlashNormAttention 统计的是整个融合 kernel 的延迟改善。

选择融合边界时，可以先看三件事：

1. Norm 的统计量是行标量还是逐元素量，能否沿线性算子移动。
2. 相邻 kernel 有没有足够长的计算流水线，可以覆盖归约和数据搬运。
3. 融合后会不会破坏主 kernel 的 tile、occupancy 或前反向调度。

如果 Norm 只占总耗时几个百分点，为它牺牲 GEMM 或 Attention 的主体效率通常不划算。原文在 N 达到 4096 后停止扩展 Multi-CTA benchmark，也正是因为这时 Norm 占比已低于 5%。

## 7. 代码与参考资料

原文公开了两部分实现：

- Multi-CTA Norm Fusion（https://github.com/facebookresearch/ads_model_kernel_library/tree/main/multi_cta_norm_fusion）
- GDPA Megakernel / FlashNormAttention（https://github.com/facebookresearch/ads_model_kernel_library/tree/main/gdpa_megakernel）

相关背景资料：

- Generalized Dot Product Attention（https://pytorch.org/blog/generalized-dot-product-attention-tackling-real-world-challenges-in-gpu-training-kernels/）
- Kunlun 论文（https://arxiv.org/abs/2602.10016）
- Meta Generative Ads Model GEM（https://engineering.fb.com/2025/11/10/ml-applications/metas-generative-ads-model-gem-the-central-brain-accelerating-ads-recommendation-ai-innovation/）
- Quack：Memory-Bound Kernel 优化（https://github.com/Dao-AILab/quack/blob/main/media/2025-07-10-membound-sol.md）
- Triton Matrix Multiplication 教程（https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html）
- FlashAttention-3（https://arxiv.org/pdf/2407.08608）
- Megakernels（https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles）

## 小结

归一化的计算量不大，但独立执行时会产生额外的整行读写和 kernel launch。想让它接近“免费”，需要先找到合适的融合边界，机械地把几行公式塞进另一个 kernel 往往不够。

Lazy Pre-Norm 通过代数变换移动行缩放；Multi-CTA Norm 用 CTA cluster 解决长行归约；FlashNormAttention 则重新规划整个 Attention 的 warp、buffer 和前反向数据流。三种方案都说明了一点：融合后的主 kernel 仍要保持高效，否则省下的 Norm 时间很容易被更差的 GEMM 或 Attention 吞掉。

## 致谢

原文作者感谢 Tri Dao、Markus Hoehnerbach、Jay Shah、Ted Zadouri、Vijay Thakkar 和 Wentao Guo 在 FlashAttention 与 Quack 上的工作，也感谢 PyTorch 和 Triton 团队提供 Helion 与 TLX。这些基础设施和前期研究构成了本文实现的基础。
