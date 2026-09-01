![MiniMax-H3 在 8×H200 上：无损加速 1.95 倍，在 0.76–0.91 SSIM 下最高加速 6.24 倍](https://files.mdnice.com/user/59/048939bc-942f-48e1-a54d-27d538842411.png)

> SGLang 仓库地址： https://github.com/sgl-project/sglang
>
> 原文地址（https://www.lmsys.org/blog/2026-08-27-minimax-h3-h200）
>
> GitHub Markdown 原文（https://github.com/lm-sys/lm-sys.github.io/blob/4b234feda2b341fccf9db17d8daa102116b454fc/blog/2026-08-27-minimax-h3-h200.md）

# MiniMax-H3 在 8×H200 上：无损加速 1.95 倍，在 0.76–0.91 SSIM 下最高加速 6.24 倍

SGLang Diffusion 团队、Cache-DiT 团队、NVIDIA、蚂蚁集团，2026 年 8 月 27 日

## TL;DR

我们使用 SGLang Diffusion（https://github.com/sgl-project/sglang），在 8 张 NVIDIA H200 上测试了 MiniMax-H3（https://github.com/MiniMax-AI）的视频生成性能。六组工作负载使用相同的 prompt、随机种子、分辨率、帧率和去噪步数。

- **SGLang 的稠密无损路径比 Diffusers 快 1.85–1.95 倍**，没有使用任何近似方法：去噪计算完全相同，只是运行时更快。
- **叠加 step reuse 和稀疏注意力后，最高可加速 6.24 倍，平均 SSIM 为 0.76–0.91。** 本次测试中最快的配置是 SubBlock 0.80 + Cache-DiT stride：在 5 秒/10 秒 T2VA 上分别加速 **5.06 倍/5.72 倍**，在 FL2VA 上分别加速 **5.86 倍/6.24 倍**。不同任务上的质量损失并不一致：FL2VA 的 SSIM 仍有 0.85–0.91，而 T2VA 降至 0.76–0.78。
- **如果更重视质量，建议只启用 Cache-DiT**，最高可加速 2.99 倍，平均 SSIM 为 0.90–0.92。如果希望兼顾速度和质量，**SubBlock 0.75 + Cache-DiT stride** 可在 SSIM 0.79–0.90 下获得 4.90–5.93 倍加速。
- 性能提升来自三个可以叠加的层次：**融合 Kernel**（单独测试各个非 GEMM 算子时加速 2.00–12.16 倍，这些微基准结果不能直接累加为端到端收益）、**step reuse**（Cache-DiT 跳过冗余的去噪步骤），以及 **SubBlock 稀疏注意力**（NVIDIA 的块稀疏 forward，降低实际执行步骤的计算成本）。

**测试范围。** 这次对比只覆盖了 SGLang Diffusion 的三个加速开关。它还支持量化、渐进式分辨率等有损路径，但这些方法没有纳入本轮测试。因此，下面的数据只是现有加速空间的一部分，并不是上限。所有结果均来自实测而非估算，文末的视频也方便读者直接判断质量损失。

|                |                                                                              |
| -------------- | ---------------------------------------------------------------------------- |
| **硬件**       | 8× NVIDIA H200（141 GB）                                                     |
| **工作负载**   | MiniMax-H3 · 1344×768 · 24 FPS · 50 个去噪步骤 · 输出 5 秒或 10 秒视频      |
| **并行配置**   | 所有模式均使用 8 张 GPU；Diffusers 使用 CP8，SGLang 使用 SP/Ulysses degree 8 |
| **版本**       | SGLang `v0.5.18`（`d90318b3e2`）                                           |
| **测试日期**   | 2026-08-18                                                                   |

---

## 背景

SGLang Diffusion 已经为 MiniMax-H3 提供了很快的无损路径，但社区一直希望进一步加速高质量视频的有损生成。SGLang Diffusion 长期积累了多种有损加速开关，过去几周也一直在推进这项工作。本文首次给出这些开关在实际测试中的表现。

视频扩散的开销主要来自两部分：去噪循环会反复执行同一个 Transformer 数十次，而每一步的大部分时间又花在超长 token 序列的注意力计算上。生成一段 5 秒、1344×768、24 FPS、使用 50 个去噪步骤的视频，早已超出单张 GPU 的实用范围。因此，问题已经不是要不要做并行，而是还能省掉多少计算。

下面三种加速从不同方向减少开销，并且可以组合使用：

- **融合 Kernel** 在不改变数学计算的前提下，降低每个步骤的固定成本。
- **Cache-DiT** 在去噪步骤之间复用结果，让一部分步骤不再执行。
- **SubBlock 稀疏注意力** 跳过贡献低于阈值的注意力块，降低仍需执行的步骤的成本。

第一种方法是无损的，后两种方法则用相似度换速度。因此，本文中的每一组数据都同时报告了相对于 SGLang 无损基线的 SSIM。

## 概览

加速比取决于比较基线。与配置匹配的 Diffusers 相比，SGLang 的稠密无损路径在两种任务和两种视频时长下都已经快了约 2 倍。Cache-DiT 复用不同去噪步骤之间的计算，SubBlock 稀疏注意力则降低仍需执行的步骤的成本。把两者组合起来，就是本次测试矩阵里最快的路径。

如果更看重质量，可以只使用 **Cache-DiT conservative 或 Cache-DiT stride**，不启用 SubBlock。如果希望平衡速度与质量，可以使用 **SubBlock 0.75 + Cache-DiT stride**：生成 5 秒视频时加速 4.90–5.64 倍，生成 10 秒视频时加速 5.44–5.93 倍。

下面两张图汇总了完整基准测试表格。

<p align="center">
  <img src="https://files.mdnice.com/user/59/e2d21a17-09b6-4d91-9227-7e33676f90f9.png" width="98%" alt="MiniMax-H3 在 H200 上的延迟对比。">
</p>

<p align="center">
  <img src="https://files.mdnice.com/user/59/e1660863-81ee-4f02-ab2b-4f3f212f2f7f.png" width="98%" alt="MiniMax-H3 在 H200 上的加速比对比。">
</p>

---

## 详细结果

本文中的每一种配置都可以通过 SGLang 的 MiniMax-H3 Cookbook 页面（https://docs.sglang.ai/cookbook/diffusion/MiniMax/MiniMax-H3）复现，页面中给出了各模式的完整启动参数。

我们报告的是生成侧推理时间，不包括服务器启动、warmup、HTTP 轮询和 MP4 下载时间。对于每一种任务和视频时长，延迟与 SSIM 都是在三个不同 prompt 上评估的。加速比以对应的 Diffusers 配置为基线；SSIM 则对所有 YUV420 帧计算，并与对应的 SGLang 无损视频比较。

### T2VA

| 模式                                   | 5 秒中位延迟 / 加速比 | 10 秒中位延迟 / 加速比 | 5 秒平均 SSIM | 10 秒平均 SSIM |
| -------------------------------------- | --------------------: | ----------------------: | -------------: | --------------: |
| Diffusers                              | 74.34 s / 1.00×       | 207.71 s / 1.00×        | —              | —               |
| SGLang lossless                        | 39.67 s / 1.87×       | 112.44 s / 1.85×        | 1.0000         | 1.0000          |
| Cache-DiT conservative                 | 28.02 s / 2.65×       | 78.28 s / 2.65×         | 0.8986         | 0.9179          |
| SubBlock 0.75                          | 30.90 s / 2.41×       | 77.12 s / 2.69×         | 0.8006         | 0.8301          |
| SubBlock 0.75 + Cache-DiT conservative | 21.41 s / 3.47×       | 57.48 s / 3.61×         | 0.7936         | 0.8288          |
| Cache-DiT stride                       | 18.13 s / 4.10×       | 52.07 s / 3.99×         | 0.8037         | 0.8078          |
| SubBlock 0.75 + Cache-DiT stride       | 15.16 s / 4.90×       | 38.21 s / 5.44×         | 0.7713         | 0.7834          |
| SubBlock 0.80                          | 29.49 s / 2.52×       | 72.85 s / 2.85×         | 0.7858         | 0.8193          |
| **SubBlock 0.80 + Cache-DiT stride**   | **14.68 s / 5.06×**   | **36.29 s / 5.72×**     | **0.7584**     | **0.7765**      |

### FL2VA

| 模式                                   | 5 秒中位延迟 / 加速比 | 10 秒中位延迟 / 加速比 | 5 秒平均 SSIM | 10 秒平均 SSIM |
| -------------------------------------- | --------------------: | ----------------------: | -------------: | --------------: |
| Diffusers                              | 80.44 s / 1.00×       | 217.31 s / 1.00×        | —              | —               |
| SGLang lossless                        | 41.31 s / 1.95×       | 114.02 s / 1.91×        | 1.0000         | 1.0000          |
| Cache-DiT conservative                 | 26.90 s / 2.99×       | 78.24 s / 2.78×         | 0.9389         | 0.9771          |
| SubBlock 0.75                          | 31.27 s / 2.57×       | 76.95 s / 2.82×         | 0.8946         | 0.9385          |
| SubBlock 0.75 + Cache-DiT conservative | 20.64 s / 3.90×       | 56.39 s / 3.85×         | 0.8924         | 0.9414          |
| SubBlock 0.75 + SageAttention          | 30.64 s / 2.63×       | 74.42 s / 2.92×         | 0.8827         | 0.9219          |
| Cache-DiT stride                       | 18.02 s / 4.46×       | 51.31 s / 4.24×         | 0.8903         | 0.9248          |
| SubBlock 0.75 + Cache-DiT stride       | 14.27 s / 5.64×       | 36.62 s / 5.93×         | 0.8629         | 0.9202          |
| SubBlock 0.80                          | 29.74 s / 2.71×       | 72.44 s / 3.00×         | 0.8837         | 0.9350          |
| **SubBlock 0.80 + Cache-DiT stride**   | **13.73 s / 5.86×**   | **34.80 s / 6.24×**     | **0.8498**     | **0.9144**      |

### 关键结论

- **SGLang 的稠密路径是最容易获得的第一笔收益。** 在工作负载保持不变的情况下，两种任务和两种视频时长上的加速比均为 1.85–1.95 倍。
- **SubBlock 0.75 + Cache-DiT stride 是较均衡的配置。** 它在保持较好输出质量的同时，生成 5 秒视频时加速 4.90–5.64 倍，生成 10 秒视频时加速 5.44–5.93 倍。
- **Stride caching 带来的吞吐提升最大。** 单独使用时可加速 3.99–4.46 倍，而 conservative 配置为 2.65–2.99 倍。
- **FL2VA 从 Cache + sparse 组合中获得的收益略高。** 最快的 FL2VA 配置达到 5.86 倍/6.24 倍，T2VA 则为 5.06 倍/5.72 倍。
- **速度与质量之间的取舍很清楚。** Conservative Cache-DiT 的 SSIM 为 0.8986–0.9771；更激进的 0.80 + stride 配置牺牲了一部分余量，换来了最低延迟。

<p align="center">
  <img src="https://files.mdnice.com/user/59/aea7bc82-a6f0-4b9e-99ab-18401eeb8f62.png" width="98%" alt="MiniMax-H3 在 H200 上的速度与质量取舍，并标出代表性配置。">
</p>

---

## 加速来自哪里

配置层面的收益来自三种机制。

**融合 Kernel** 降低每个实际执行步骤的成本。H3 路径融合了 indexed AdaLN 更新、gated residual、SwiGLU 激活，以及带 3D RoPE 的 QK RMSNorm，从而减少中间 Tensor、内存流量和 Kernel launch。下一节给出这些 Kernel 的独立测试结果。它们负责优化单步实现，而 Cache-DiT 和 SubBlock 决定这套实现实际会执行多少。

**Cache-DiT** 为 MiniMax-H3 共享的 DiT block stack 绑定一个 DBCache context。warmup 结束后，它会计算配置好的边界 block，并将归一化残差变化与上一次缓存状态比较。如果变化低于阈值，并且连续缓存次数没有超过上限，中间 block 就会复用缓存结果；否则重新计算整个 stack 并刷新缓存。所有缓存模式都使用 `Fn=1`、`Bn=0` 和四个 warmup 步骤：

- conservative：共享 packed stack 的 RDT 为 `0.04`，最多连续缓存 `1` 步；
- stride：共享 packed stack 的 RDT 为 `0.08`，最多连续缓存 `3` 步。

MiniMax-H3 只有一个 `MiniMaxH3DiTModel`，它的 block stack 同时承载打包后的视频和音频 token。因此，Cache-DiT 会对整个 packed stack 做一次共享决策，而不是分别维护视频缓存和音频缓存。worker 只记录一份合并后的 Cache-DiT 步骤列表，轨迹图的图例也遵循这一执行模型。

**SubBlock 稀疏注意力** 减少实际计算步骤中读取的 KV block 数。它使用 `n_k=n_q=4`：前十个去噪步骤仍运行稠密注意力，之后才启用 SubBlock；最短序列长度为 `4096`。测试矩阵使用 `0.75` 和 `0.80` 两种 sparsity，后者更快，但在几组 T2VA 测试上的 SSIM 更低。

汇总结果反映的是各配置的端到端表现，并没有单独拆分 Kernel 时间，也没有给出逐步骤成本。下面的轨迹是请求级执行轨迹，不是算子级计时。

### 一次实测的 49 步执行轨迹

工作负载配置了 50 个推理步骤。由于 sigma schedule 包含区间的两个端点，去噪循环实际执行 49 次模型计算，即 `len(sigmas) - 1`。这里的“49 步轨迹”指的正是这 49 次模型计算。

为了直观展示执行模式，我们用一条 5 秒 T2VA 请求测试了六种配置：lossless、Cache-DiT conservative、SubBlock 0.75、SubBlock 0.75 + conservative Cache-DiT、Cache-DiT stride，以及 SubBlock 0.80 + stride。worker 记录了每个请求实际的 `cached_steps` 列表。视频和音频 token 共用同一个 packed H3 block stack，因此一次 cache hit 会复用合并后的输出，这条路径不存在“视频命中缓存、音频重新计算”这样的状态。SubBlock 行中的蓝色方格表示前十个去噪步骤之后使用稀疏注意力的计算步骤。

<p align="center">
  <img src="https://files.mdnice.com/user/59/2f8404d2-1264-4e79-ac67-4e43b8849cde.png" width="98%" alt="MiniMax-H3 在 H200 上真实测得的 49 步执行轨迹。">
</p>

这次轨迹测试的耗时分别为：lossless 37.78 秒、Cache-DiT conservative 26.82 秒、SubBlock 0.75 29.97 秒、SubBlock 0.75 + conservative Cache-DiT 22.18 秒、Cache-DiT stride 17.23 秒，以及 SubBlock 0.80 + stride 14.34 秒。这些数字只用于标识本次轨迹，不替代前面基于三个 prompt 统计的中位数。

---

## Kernel 层

缓存决定运行多少个去噪步骤，Kernel 则决定每个实际计算步骤有多快。MiniMax-H3 将视频和音频 token 打包进同一个序列，因此整条非 GEMM 路径遵循同一个原则：减少内存流量、中间 Tensor 和 Kernel launch。AdaLN modulation 与 gated residual 根据 token 索引查找参数，并在一次遍历中更新 activation；SwiGLU 直接在融合后的 `gate_up` buffer 上操作；QK RMSNorm 和 3D RoPE 也被融合进同一个 Kernel，不再作为多个 eager 算子分别执行。

下表使用一条 5 秒 T2VA 请求在 1344×768×124 帧下的真实 per-rank shape：经 SP/Ulysses-8 padding 后为 4,722 行，hidden size 为 5,376，56 个 attention head，head dimension 为 128，RoPE dimension 为 96，输入类型为 BF16。每个数字都是 10 轮、每轮 20 次调用的 CUDA event 单次耗时中位数，基线是对应的 eager 组合。

<p align="center">
  <img src="https://files.mdnice.com/user/59/dc140c75-98fc-47f1-bff6-ea9a375e6ef1.png" width="98%" alt="MiniMax-H3 在 H200 上的融合 Kernel 加速比。">
</p>

| 算子                                   | Eager 组合 | SGLang Kernel | 加速比 |
| -------------------------------------- | ---------: | ------------: | -----: |
| AdaLN modulation（indexed scale-shift） | 136.7 μs   | 38.2 μs       | 3.58×  |
| AdaLN gated residual（indexed）         | 93.2 μs    | 46.6 μs       | 2.00×  |
| SwiGLU activation（in place）           | 364.5 μs   | 105.2 μs      | 3.46×  |
| QK RMSNorm                             | 334.0 μs   | 76.9 μs       | 4.35×  |
| QK RMSNorm + 3D RoPE（单个 Kernel）     | 1335.6 μs  | 109.8 μs      | 12.16× |

这些数据来自各个位置的独立微基准，不能直接累加成端到端延迟收益。融合 QK-Norm + RoPE 的结果使用 main 分支上的精确舍入路径，即 `round_norm_before_rope=True`。

---

## SubBlock 稀疏注意力的工作原理

SubBlock 是一个无需训练的块稀疏注意力 router。它先把序列划分为 64-token 的 query block 和 key block，再把每个 block 的两侧都拆成四个 16-token sub-block（`n_q=n_k=4`）。轻量级 pooling 和 log-sum-exp score 会估算每个 query block、每个 head 对应的各 key block 的未归一化 softmax mass。router 保留得分最高的 key block，并把索引交给块稀疏注意力 Kernel，整个注意力矩阵始终不会被实例化。

`sparsity` 表示允许丢弃的 key block 比例，而不是保留比例。因此，`sparsity=0.75` 会为每个 query block 保留约 25% 的 key block。更激进的 `0.80` 配置速度更快，但允许更大的近似误差，这也与最激进配置中更低的 SSIM 相符。

下图给出了 score 分布，竖线表示两种展示预算下逐行 routing cutoff 的中位数。这里加入 `sparsity=0.50` 只是为了辅助观察，正式基准配置使用的是 `0.75` 和 `0.80`。router 会为每个 query block 和 head 独立排序 key block，因此在按 8 个 block 对预算取整后，`sparsity=0.50` 和 `0.75` 分别保留该行大约一半和四分之一的可用 key block。在这些工作负载上，`0.75` 的预算保留了 row-local 中位数以上的大部分 score mass，同时把选择集中到高分尾部。

<p align="center">
  <img src="https://files.mdnice.com/user/59/feb210a5-52bf-43af-8685-3548372cd196.png" width="98%" alt="SubBlock 的 score 分布与 cutoff 区间。">
</p>

稀疏路径只会用于 Kernel 支持的长序列、非因果 DiT 注意力调用：输入为 BF16、head dimension 为 128、序列长度至少为 4096。前十个去噪步骤使用稠密注意力；短序列、token refiner 和不受支持的调用都回退到稠密路径。在 H200/SM90 上，选出的 64×64 routing plan 由 SGLang 的 CuTe 块稀疏 FlashAttention Kernel 执行。

---

## 效果演示

每个选定的 prompt 都展示了四种模式：

- **Prompt 1** · T2VA · 5 秒 · 三只猫带着铜管乐器，在熟睡的主人旁边演奏；
- **Prompt 2** · T2VA · 10 秒 · 夜晚雨中的赛博朋克城市；
- **Prompt 3** · FL2VA · 5 秒 · 黏土狐狸的续写视频。

四种模式分别是 SGLang lossless、Cache-DiT conservative、SubBlock 0.75 + Cache-DiT stride，以及 SubBlock 0.80 + Cache-DiT stride。文件名编码了 prompt、任务、模式和时长；SVG 图表也位于同一个目录。

**Prompt 1 · T2VA · 5 秒**

<div style="display:flex; flex-wrap:wrap; gap:1%;">
<div style="width:49%;">
<video controls preload="metadata" playsinline src="https://www.lmsys.org/images/blog/minimax-h3-h200/Prompt1__T2VA__sglang_lossless__5s.mp4" style="width:100%;"></video>
<div style="font-size:0.85em; text-align:center;">SGLang lossless</div>
</div>
<div style="width:49%;">
<video controls preload="metadata" playsinline src="https://www.lmsys.org/images/blog/minimax-h3-h200/Prompt1__T2VA__cachedit_conservative__5s.mp4" style="width:100%;"></video>
<div style="font-size:0.85em; text-align:center;">Cache-DiT conservative</div>
</div>
<div style="width:49%;">
<video controls preload="metadata" playsinline src="https://www.lmsys.org/images/blog/minimax-h3-h200/Prompt1__T2VA__subblock_cachedit_stride__5s.mp4" style="width:100%;"></video>
<div style="font-size:0.85em; text-align:center;">SubBlock 0.75 + Cache-DiT stride</div>
</div>
<div style="width:49%;">
<video controls preload="metadata" playsinline src="https://www.lmsys.org/images/blog/minimax-h3-h200/Prompt1__T2VA__subblock_080_cachedit_stride__5s.mp4" style="width:100%;"></video>
<div style="font-size:0.85em; text-align:center;">SubBlock 0.80 + Cache-DiT stride</div>
</div>
</div>

**Prompt 2 · T2VA · 10 秒**

<div style="display:flex; flex-wrap:wrap; gap:1%;">
<div style="width:49%;">
<video controls preload="metadata" playsinline src="https://www.lmsys.org/images/blog/minimax-h3-h200/Prompt2__T2VA__sglang_lossless__10s.mp4" style="width:100%;"></video>
<div style="font-size:0.85em; text-align:center;">SGLang lossless</div>
</div>
<div style="width:49%;">
<video controls preload="metadata" playsinline src="https://www.lmsys.org/images/blog/minimax-h3-h200/Prompt2__T2VA__cachedit_conservative__10s.mp4" style="width:100%;"></video>
<div style="font-size:0.85em; text-align:center;">Cache-DiT conservative</div>
</div>
<div style="width:49%;">
<video controls preload="metadata" playsinline src="https://www.lmsys.org/images/blog/minimax-h3-h200/Prompt2__T2VA__subblock_cachedit_stride__10s.mp4" style="width:100%;"></video>
<div style="font-size:0.85em; text-align:center;">SubBlock 0.75 + Cache-DiT stride</div>
</div>
<div style="width:49%;">
<video controls preload="metadata" playsinline src="https://www.lmsys.org/images/blog/minimax-h3-h200/Prompt2__T2VA__subblock_080_cachedit_stride__10s.mp4" style="width:100%;"></video>
<div style="font-size:0.85em; text-align:center;">SubBlock 0.80 + Cache-DiT stride</div>
</div>
</div>

**Prompt 3 · FL2VA · 5 秒**

<div style="display:flex; flex-wrap:wrap; gap:1%;">
<div style="width:49%;">
<video controls preload="metadata" playsinline src="https://www.lmsys.org/images/blog/minimax-h3-h200/Prompt3__FL2VA__sglang_lossless__5s.mp4" style="width:100%;"></video>
<div style="font-size:0.85em; text-align:center;">SGLang lossless</div>
</div>
<div style="width:49%;">
<video controls preload="metadata" playsinline src="https://www.lmsys.org/images/blog/minimax-h3-h200/Prompt3__FL2VA__cachedit_conservative__5s.mp4" style="width:100%;"></video>
<div style="font-size:0.85em; text-align:center;">Cache-DiT conservative</div>
</div>
<div style="width:49%;">
<video controls preload="metadata" playsinline src="https://www.lmsys.org/images/blog/minimax-h3-h200/Prompt3__FL2VA__subblock_cachedit_stride__5s.mp4" style="width:100%;"></video>
<div style="font-size:0.85em; text-align:center;">SubBlock 0.75 + Cache-DiT stride</div>
</div>
<div style="width:49%;">
<video controls preload="metadata" playsinline src="https://www.lmsys.org/images/blog/minimax-h3-h200/Prompt3__FL2VA__subblock_080_cachedit_stride__5s.mp4" style="width:100%;"></video>
<div style="font-size:0.85em; text-align:center;">SubBlock 0.80 + Cache-DiT stride</div>
</div>
</div>

<details>
<summary>Prompt 1 · 完整 Prompt（英文原文）</summary>

```text
integrated_multimodal_description: [Shot 1] Live-action, whimsical cinematic, a medium-wide shot frames a dim bedroom at night where the owner sleeps under the covers. A bedroom door opens and three cats enter in single file, each carrying a tiny brass instrument. The camera tracks sideways with small amplitude at slow speed as the cats march beside the bed and play a short, lively diegetic brass tune in synchrony; the sleeping owner shifts slightly but does not wake. The cats finish with one crisp flourish, pivot together, and abruptly file back out through the doorway, with the last cat's tail disappearing from frame. No character speaks and no human voice is heard.

overall_soundscape: Quiet nighttime room tone, the owner's steady breathing, soft pawsteps on the floor, a faint door creak, and light bedding rustle as the procession passes.

non_diegetic_music: N/A
```

</details>

<details>
<summary>Prompt 2 · 完整 Prompt（英文原文）</summary>

```text
integrated_multimodal_description: [Shot 1] Live-action, cinematic, a wide establishing shot frames a futuristic cyberpunk city at night as rain falls across dense towers, elevated transit lines, and a crowded street lined with vivid neon light. The camera pushes forward with small amplitude at slow speed above the wet pavement while pedestrians in reflective coats pass beneath transparent umbrellas, a compact hovering vehicle glides through the intersection, and saturated magenta, cyan, and amber reflections ripple across puddles. Steam drifts from a street vent and briefly catches the neon glow as the vehicle recedes between the towers. No dialogue or voiceover is heard.

overall_soundscape: Steady rainfall, distant traffic, the low hum of elevated transit, electrical buzzing from signs, soft footsteps through shallow water, and a brief rush of air as the hovering vehicle passes.

non_diegetic_music: A slow electronic pulse with deep analog bass, sparse metallic percussion, and sustained synthesizer tones that gradually increase in volume before fading.
```

</details>

<details>
<summary>Prompt 3 · 完整 Prompt（英文原文）</summary>

```text
For the target video, at 0.00 seconds into the target video, <Picture 1> is fully referenced.

integrated_multimodal_description:
[Shot 1] A handcrafted stop-motion clay animation begins from <Picture 1>. A small orange clay fox with large expressive eyes trots along a mossy path through a warm, richly detailed miniature forest. The camera tracks the fox smoothly at eye level while layered clay trees and shrubs create gentle parallax. The fox looks curiously toward the camera, slows near the middle of the path, flicks its tail, then continues toward the small wooden cabin in the distance. Preserve the exact clay textures, warm amber lighting, forest layout, fox proportions, and family-friendly whimsical tone established by <Picture 1>. Motion remains coherent and physically plausible for stop-motion animation.

overall_soundscape:
Soft clay footsteps, rustling leaves, distant birds, and a light forest breeze accompany the fox's movement.

non_diegetic_music:
A gentle playful score with pizzicato strings, wooden percussion, and soft flute.
```

</details>

## 致谢

这次基准测试由多个团队共同完成，感谢所有参与者。

- **SGLang Diffusion 团队（https://github.com/sgl-project/sglang）**：撰写本文初稿，持续推进这些结果所依赖的 SGLang Kernel，并提供本次测试使用的 Diffusion runtime、融合 Kernel 和并行能力。
- **蚂蚁集团 Ji Huang（@IPostYellow，https://github.com/IPostYellow）**：完成 H200 基准测试，将 SubBlock 稀疏注意力引入 SGLang Diffusion，并参与本文修订。
- **Cache-DiT 团队（https://github.com/vipshop/cache-dit）**：感谢 @DefTruth 和 vipshop.com 团队开发 Cache-DiT，并协助把它的缓存配置接入 SGLang Diffusion。
- **MiniMax（https://github.com/MiniMax-AI）**：开源 MiniMax-H3，本文的所有测试都基于该模型。
- **NVIDIA**：提供底层 SubBlock 稀疏注意力支持，包括这些测试依赖的块稀疏注意力 forward。

---

测试于 2026 年 8 月 18 日在 8 张 NVIDIA H200 上完成。复现方法和逐 prompt 原始数据见基准测试仓库（https://github.com/BBuf/how-to-optim-algorithm-in-cuda/pull/26）。
