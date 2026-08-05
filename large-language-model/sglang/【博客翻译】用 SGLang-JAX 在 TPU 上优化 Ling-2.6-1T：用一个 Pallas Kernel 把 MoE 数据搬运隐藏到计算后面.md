> 本文翻译自 LMSYS Blog：<https://www.lmsys.org/blog/2026-06-17-ling-2-6-tpu/>。原文 Markdown 来自 `lm-sys/lm-sys.github.io`，仓库采用 MIT License。
>
> 原标题：Optimizing Ling-2.6-1T on TPU with SGLang-JAX: Hiding MoE Data Movement Behind Compute with One Pallas Kernel
>
> 原作者：Prayer, JamesBrianD, Haolin Fu, Haoguang Cai, Qinghan Chen

# 用 SGLang-JAX 在 TPU 上优化 Ling-2.6-1T：用一个 Pallas Kernel 把 MoE 数据搬运隐藏到计算后面

SGLang-JAX 现在已经支持在 TPU v7x 上高效服务 inclusionAI 的 Ling-2.6-1T。在可运行的 baseline 建好后，profile 指向了 Mixture-of-Experts（MoE）路径：这是主要瓶颈。每一层都要把 token scatter 到 32 个 JAX device 上（每个 v7x chip 暴露两个 device），执行 expert FFN，然后再把输出 gather 回来。本文先聚焦 Fused MoE V2，这是一个新的 Pallas kernel，把 scatter、expert FFN 和 gather 融合在一起，同时重叠 TPU 上的计算和数据搬运。

使用 Fused MoE V2 后，MoE prefill 延迟从 **5.16 ms 降到 2.42 ms**。在同一个 SGLang decode benchmark 上，**16 个 TPU v7x chip 的输出吞吐达到 16 张 H200 GPU 的 1.29 倍到 1.77 倍**。完整数据如下。

![Ling-2.6-1T decode throughput, TPU v7x vs GPU H200](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-1/00fca46a-ff67-43b5-ad60-679fe13e069e.png)

图 1：Ling-2.6-1T 在 TPU v7x-16 和 H200×16 上的 decode 吞吐对比。benchmark 使用 SGLang 默认的 `random` 数据集（从 ShareGPT 采样），输入长度 16,384 token，输出长度 1,024 token。

## TL;DR

- **Fused MoE V2：** 相比 Fused MoE V1，MoE prefill 延迟降低 **53%**（**5.16 -> 2.42 ms**）；decode kernel 延迟降低约 **15%**（**0.249 -> 0.211 ms**）。
- **端到端收益：** 只替换 MoE kernel，prefill 吞吐提升 **24.8%**，decode 吞吐提升 **18.5% 到 35.3%**。
- **TPU vs H200 decode：** `mc=128` 时，TPU v7x-16 的 decode 输出吞吐是 H200×16 的 **1.29 倍**；`mc=512` 时达到 **1.77 倍**。
- **MoE 之外：** 完整的 Ling-2.6-1T bring-up 还包括 hybrid KV/recurrent memory pool、GLA linear attention，以及 single-controller data parallelism。

**Ling-2.6-1T 简介：** 这是一个 **1T 稀疏 MoE** 模型，每个 token 激活 **63B 参数**，有 **256 个 routed expert，top-8 routing，外加一个 shared expert**，MoE 权重使用 **per-channel fp8**，主干是混合 **MLA + Lightning Linear**。前半部分的 kernel 优化主要由 MoE 结构驱动；后面的 memory pool 和 GLA bring-up 则来自这个 hybrid backbone 的需求。

## 背景：优化 Fused MoE Kernel

除非额外说明，本节所有 MoE 数据都来自 `jax.profiler` device trace。测试环境是一组 16-chip TPU v7x slice：`ep=32`，2×2×4 ICI torus，每个 chip 两个 device。workload 是 Ling-2.6-1T，16,384-token prefill，512-token decode batch，MoE 权重为 per-channel fp8。本节所有 lower bound 都按 device 计算，大约是 chip 级计算能力和带宽的一半；chip 规格见附录。

Fused MoE V2 的核心，是改变 routed token、expert weight 和 accumulator 在 VMEM、HBM、ICI 之间流动的方式。

### 1. MoE kernel 成本模型

Ling-2.6-1T 每层有 256 个 routed expert 和 1 个 shared expert，routing 是 top-8。`ep=32` 时，每个 device 拥有 8 个本地 routed expert。一个 token 选出的 8 个 expert 通常会分布在不同 device 上，所以每层 routed path 的形态都是：

```text
scatter tokens -> local expert FFN -> gather results
```

在这个结构里，MoE 的运行成本不只是 GEMM FLOPs。kernel 必须处理三条昂贵的数据路径：token 在 chip 之间 routing，从 HBM 读取 expert weight 到 VMEM，以及围绕矩阵乘法单元（MXU）的 fp8 layout / scale 处理。

shared expert 是本地 dense 路径。它会增加本地 FFN 计算量，但不参与 routed all-to-all，对 token routing payload 的影响很小。

#### Compute lower bound

prefill 16,384、top-8 routing、`ep=32` 时，每个 device 处理：

```text
16384 * 8 / 32 = 4096 routed rows / device
```

平均来看，8 个本地 routed expert 中每个 expert 大约看到 512 行。shared expert 不通过 top-k fan out；它只在本地 4096 行上运行一次。routed + shared FFN 的计算量为：

```text
FFN1: 8 experts * 2 matrices * (2 * 512 * 8192 * 2048) = 274.9 GFLOP
FFN2: 8 experts * 1 matrix  * (2 * 512 * 2048 * 8192) = 137.4 GFLOP
Routed total: 412.3 GFLOP / device
Shared expert: 3 matrices * (2 * 4096 * 8192 * 2048) = 412.3 GFLOP
Total: 824.6 GFLOP / device
```

TPU v7x 公开规格给出的 fp8 计算能力约为每 chip 4.614 PFLOP/s。在这次部署里，一个 chip 暴露为两个 device，所以粗略的 per-device fp8 峰值是 2.307 PFLOP/s。理想计算 lower bound 为：

```text
824.6 GFLOP / 2307 TFLOP/s = 0.36 ms
```

这是一个理想 lower bound，不包含数据搬运、fp8 pack/unpack，也不包含向量处理单元（VPU）上的 scale handling。生产 trace 里测到的 **2.42 ms** 仍然是这个 lower bound 的约 **7 倍**，所以纯 GEMM FLOPs 解释不了这个延迟。

#### ICI token routing lower bound

每个 device 的 scatter payload 是：

```text
4096 rows * 8192 hidden = 33,554,432 elements
bf16: 67.1 MB
fp8 : 33.5 MB
```

TPU v7x 每个 chip 有 1.2 TB/s 双向 ICI 带宽，折算下来每条 link 每个方向约 100 GB/s。2×2×4 torus 让每个 chip 有 4 条有效 link，所以有效单向 chip 带宽大约是 4 × 100 GB/s = 400 GB/s。两个 device 共享一个 chip，因此粗略的 per-device 单向 injection bandwidth 约为 200 GB/s。

如果只看 injection bandwidth，不考虑 hop 和 contention，lower bound 是：

| payload | one scatter | scatter + gather |
|---|---:|---:|
| fp8 33.5 MB | 0.17 ms | 0.34 ms |
| bf16 67.1 MB | 0.34 ms | 0.67 ms |

但 all-to-all 不是单链路带宽测试。在 2×2×4 torus 上，一个随机目标平均约两跳：x 方向约 0.5 hop，y 方向约 0.5 hop，z 方向约 1.0 hop。把 hop 因子算进去后，拓扑调整后的 lower bound 更接近：

| payload | one scatter, avg-hop adjusted | scatter + gather |
|---|---:|---:|
| fp8 | 0.34 ms | 0.67 ms |
| bf16 | 0.67 ms | 1.34 ms |

这仍然没有包含 link contention、小 DMA 粒度、runtime overhead 和 fp8 layout handling。即便如此，token routing 已经和 routed + shared 的理想计算 lower bound 处在同一量级，并且明显高于 routed-only compute lower bound。

#### HBM weight movement lower bound

再看 routed expert weight 的 HBM 读取成本。如果 weight prefetch 不能被 pipeline 隐藏，这部分成本会立刻暴露出来。

一个本地 expert 的 fp8 权重为：

```text
W1 + W3 + W2 = 3 * 8192 * 2048 bytes = 50.3 MB
8 local experts = 402 MB
```

shared expert 会再增加一组本地 FFN 权重，大小大致等于一个本地 expert，但它不引入 all-to-all traffic。下面的估算只关注 routed expert path。

TPU v7x HBM 带宽约为每 chip 7.38 TB/s，也就是每 device 约 3.69 TB/s。读取全部 8 个本地 expert 一次的 lower bound 为：

```text
402 MB / 3.69 TB/s = 0.11 ms
```

实际 kernel 会在每个 token-staging tile 上重新读取一次权重。tile size 由 `bts` 决定，即 block token staging size：一次 expert FFN tile 里搬进 VMEM 的 routed token 行数。Ling prefill 使用 `bts=160`。由于每个 expert 大约看到 512 行，prefill 需要 `ceil(512 / 160) = 4` 个 token staging tile。V2 会跨这些 tile 对 weight prefetch 做流水，因此 HBM read lower bound 大约为：

```text
4 * 402 MB / 3.69 TB/s = 0.44 ms
```

weight 读取不一定要出现在关键路径上。V2 通过 double buffering 把它藏在 MXU 窗口后面。这些数字解释了为什么必须这么调度：如果 HBM read 在 GEMM 前串行执行，它本身就已经超过纯计算 lower bound。

#### 小结

TPU 上的 MoE 主要是数据搬运和 overlap 问题：

- routed + shared FFN compute lower bound：约 **0.36 ms**；
- fp8 scatter + gather 拓扑 lower bound：约 **0.67 ms**；
- expert weight HBM read lower bound：每个 tile 约 **0.11 ms**，`bts=160` 时约 **0.44 ms**；
- fp8 packing、scale broadcast、layout reorder 仍然会消耗 VPU 和 VMEM 带宽。

优化目标不是减少 FFN FLOPs，而是把 token routing、weight prefetch 和 fp8 reorder 隐藏到 routed compute window 后面。

### 2. 为什么需要 Pallas fused kernel

接下来会用到一些 TPU 术语。简化来看：TensorCore 里有 MXU、VPU 和 VMEM；HBM 在 chip 外；chip 之间通过 ICI 通信。

![Simplified TPU execution model](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/b4b05dbb-9aa7-41bd-a1d2-b2882fe3692d.png)

图 2：本节使用的简化 TPU 执行模型，改编自 [JAX Scaling Book TPU overview](https://jax-ml.github.io/scaling-book/tpus/)。

在 MoE kernel 中，这些单元对应的工作如下：

| Hardware unit | TPU role | Work in MoE |
|---|---|---|
| MXU | 矩阵乘法单元 | routed expert 的 W1/W3 gate-up GEMM 和 W2 down GEMM |
| VPU | 向量数学、规约、layout 工作 | SiLU、gating multiply、scale multiply、fp8 pack/unpack、lane reorder |
| VMEM | 靠近 MXU/VPU 的片上 scratchpad | routed token tile、expert intermediate、output accumulator、prefetched weight tile |
| HBM | 每个 chip 连接的大容量片外内存 | expert weight、token staging buffer、大型 intermediate buffer |
| HBM-DMA | HBM 和 VMEM 之间的数据搬运 | 把当前 / 下一个 expert 的权重 prefetch 到 VMEM；必要时移动 staging buffer |
| ICI / ICI-DMA | TPU slice 内的直接 chip 间网络 | 在源 chip 和目标 chip 之间移动 routed token payload；scatter 到 expert owner，再把输出 gather 回 token 顺序 |

纯 JAX native MoE 可以正确表达 routing、expert FFN 和 output aggregation。它暴露不了的是单个 MoE layer 内部的细粒度调度。一旦 scatter、expert FFN、HBM weight movement、fp8 layout work 和 gather 跨过多个 JAX op 或 collective 边界，XLA 就很难把 ICI-DMA、HBM-DMA、MXU 和 VPU 工作稳定地放进一个手动调度的 pipeline。

这条路径也不能简单当成一个独立 sparse lookup 然后卸载到 SparseCore：scatter 得到的本地 expert token layout、per-expert offset、expert output 和最终 token order 彼此依赖。真正有用的优化空间就在 MoE kernel 内部。

![Naive fused MoE pipeline](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-1/3c0de24f-148b-44aa-9a97-e0337ae48188.png)

图 3：naive fused pipeline，其中通信和计算阶段串行。语义是正确的，但各个引擎没有被细粒度 overlap 调度起来。

理想的 steady state 是：MXU 在计算 expert *i* 时，HBM-DMA prefetch expert *i+1* 的权重，ICI-out 发送下一批 routed token，ICI-in 接收上一批输出，VPU 处理上一次 matmul 产生的 scale 和 layout 工作。

要表达这种 schedule，scatter、expert FFN 和 gather 必须放进一个 Pallas kernel。fusion 的主要目的不是减少 op 数，而是创造一个调度空间，让依赖的阶段可以手动排布到 MXU、VPU、HBM-DMA 和 ICI-DMA 上。

### 3. V1：已经融合，但 hidden tiling 很碎

我们的起点是 Fused MoE V1，它最早由 Jevin Jiang、Kyuyeun Kim 等人在 tpu-inference 项目中提出和优化 [4]，之后带着一些修改被适配进 SGLang-JAX，成为 `FusedEPMoE` [5]。V1 已经把 scatter、expert FFN 和 gather 放进一个 Pallas call，并且在每个 device 上执行 8 个本地 expert。它满足了 kernel 内通信 / 计算调度的前提，但还没达到上面那个理想 steady state。

问题出在 expert 内部。一个 MoE expert 需要的不只是 input token tile 和一个 GEMM output。为了 overlap 通信和计算，kernel 还需要 weight staging buffer、intermediate activation、output accumulator 和 DMA double buffer。Ling 的 hidden size 是 8192，如果让完整 hidden dimension 常驻 VMEM，很快就会耗尽 VMEM，尤其是 f32 accumulator 和 W1/W3/W2 staging 都要占空间。

所以 V1 选择了保守路径：切分 hidden dimension，用较小 working set 流过 VMEM。

Ling 16,384 prefill 的 V1 配置为：

```text
bf=1024 / bd1=512 / bd2=512 / bts=128 / btc=128
```

这个 block config 回答的是 placement 问题：哪些 token 行、intermediate channel 和 hidden channel 留在 VMEM，哪些从 HBM stream 进来。

这些参数可以理解成 GEMM 各个轴上的 tile size：

| Param | Controls | Performance meaning |
|---|---|---|
| `bts` | staged 到 VMEM 的 routed token 行数，也就是一个 expert tile | 控制 M；太小会摊不平 DMA / VPU / MXU 固定成本 |
| `btc` | `bts` 里送入一次 compute loop 的 token 行数 | 内层 M compute tile；不能超过 `bts`，通常整除 `bts` |
| `bf` | W1/W3/W2 的 intermediate channel | 控制 FFN intermediate tile；通常越大 MXU window 越长，但消耗更多 VMEM |
| `bd1` | FFN1 hidden reduction-K slice | V1 切 hidden K；`bd1` 越小，FFN1 dot 越多且越小 |
| `bd2` | FFN2 hidden output-N slice | V1 切 output hidden；`bd2` 越小，partial output 越频繁地往返 HBM |

所以 `bf/bd1/bd2` 主要控制 feature / hidden dimension，`bts/btc` 控制每个 expert 的 token 行数。它们共同决定一个 tile 能否塞进 64 MB VMEM，以及围绕 MXU 可以 overlap 多少 HBM-DMA / VPU 工作。

V1 要付出三类结构性成本：

| Cost | V1 behavior | Why it hurts |
|---|---|---|
| FFN1 dot 太小 | `bd1=512`；经过 fp8 packing 后，effective K 大约是 256，所以 V1 要扫描 16 个 hidden-dimension slice | `vmatmul` 固定开销摊不平 |
| token staging 太频繁 | `num_bf * num_bd1 * num_token_tiles = 2 * 16 * 4 = 128` 次 HBM->VMEM staging | 很多小 DMA 和 layout step |
| FFN2 partial spill 到 HBM | partial output 写到 `a2a_s_acc_x2_hbm`，之后为了 `bf` accumulation 再读回来 | HBM read-modify-write 把关键路径切碎 |

V1 有一些微小 overlap，但 hidden-dimension slice 让 overlap window 很短。prefetch 一次只能覆盖一个小 slice，FFN2 partial output 仍然要往返 HBM。V1 prefill 延迟为 **5.16 ms**。

### 4. V2：VMEM residency 和 weight double buffering

V2 不是简单把 V1 tile 变大。它改变了 tensor lifetime。V1 的 loop 围绕 hidden-dimension slice 循环；V2 则让 routed token、gate/up intermediate 和 output accumulator 在 FFN loop 中常驻 VMEM，同时 W1/W3/W2 通过 double buffer 从 HBM stream 进来。

这会把更多 VMEM 用在长生命周期 tensor 上，但也移除了大部分 hidden-slice staging，几乎消除了 FFN2 的 HBM read-modify-write 路径。

Ling 16,384 prefill 的 V2 生产配置为：

```text
bf=512 / bts=160 / btc=80
```

V2 没有 `bd1` 和 `bd2`，因为它不再切 hidden dimension。结构变化如下：

| Per expert | V1 | V2 | Effect |
|---|---|---|---|
| FFN1 dot | 每个 hardware dot 的 effective K 大约 256 | fp8 chunk K 大约 2048；4 个 chunk 覆盖完整 hidden | K 约大 8 倍 |
| W2 output | `bd2=512`，每次产生很窄的 hidden slice | output chunk 大约 4096 个 hidden channel | N 约大 16 倍 |
| token staging | 128 次小 staging | 约 4 次 full-hidden staging | staging 次数约少 32 倍 |
| FFN2 accumulator | partial output 经 HBM spill / reload | `b_y_acc_vmem` 在 VMEM 内跨 `bf` 累加 | HBM read-modify-write 基本消失 |

这也解释了为什么只在 V1 里增大 `bd1` / `bd2` 不够。V1 中更大的 hidden tile 也会放大 weight buffer、token staging buffer 和 partial-output staging，很快碰到 64 MB VMEM 上限。更关键的是，V1 仍然围绕 hidden slice 循环；它没有让 token 和 output accumulator 常驻。

有了这个 VMEM-resident working set，V2 获得了更大的 MXU tile、更少的 HBM spill、更长的 routed compute window。在 activation quantization 之前，V2 已经把 prefill 延迟从 **5.16 ms** 降到 **3.02 ms**。开启 activation quantization 和 in-kernel shared expert overlap 后，生产 trace 达到 **2.42 ms**，比 V1 低约 **53%**。

decode 逻辑类似，但可优化空间更小。512-token decode batch 下，kernel 延迟从 **0.249 ms** 降到 **0.211 ms**，约 **15%**。每个 expert 的 effective M 维很小，MXU tile 没那么容易摊平固定开销；这条路径也更接近 expert-weight HBM read lower bound，decode trace 已经达到约 80% 的 HBM bandwidth utilization。所以 V2 对 decode 仍然有帮助，但不像 prefill 那样能完整吃到 VMEM residency 和 routed-window 的收益。

![V1 and V2 fused MoE pipeline](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/bf819538-0ab2-47ab-9156-870644a329ad.png)

图 4：V1 和 V2 fused MoE 的概念时间线。V1 因为 hidden-dimension slice 频繁循环，只能创造很小的 overlap window；V2 让 token 和 accumulator 常驻 VMEM，对 expert weight 做 double buffer，并把大部分 scatter/gather traffic 隐藏到 routed compute window 后面。

### 5. V2 的针对性优化

#### Per-channel `direct_scaled_dot`

fp8 weight quantization 的 scale 粒度决定 MXU 看到的是一个大 GEMM，还是一串小 GEMM。

使用 per-block quantization 时，scale 依赖 K block：

```text
out[m,n] = sum_k A[m,k] * W[k,n] * scale[block(k),n]
```

scale 不能从 reduction 里提出来，因此 K 必须切成 block。每个 block 执行一次小 fp8 dot，乘上该 block 的 scale，再累加。一个大 GEMM 会变成很多小 GEMM，中间插入 VPU 工作。

使用 per-channel quantization 时，scale 只依赖输出 channel：

```text
out[m,n] = (sum_k A[m,k] * W[k,n]) * scale[n]
```

scale 可以在 reduction 后应用。V2 的 `direct_scaled_dot` 直接把 fp8 token 和 fp8 weight 送入 MXU，拿到 f32 partial 后再应用 per-token / per-channel scale。Ling 的 MoE weight 使用 per-channel scale，因此可以走这条路径。

这保留了完整 K dot，避免把一个大 GEMM 按 scale block 切碎。剩下的成本是 fp8 sub-word packing、scale broadcast 和 lane reorder。per-block quantization 则会在这之外再增加 K segmentation 和 inter-block scale handling。

#### Activation quantization

V2 在 scatter 之前把 activation 从 bf16 quantize 到 fp8，直接把 routed token payload 减半。在 Ling 16,384 prefill 上，kernel 内 scatter 阶段从 **1.39 ms** 降到 **0.65 ms**。

这和前面的 ICI lower-bound 计算一致：payload 从 bf16 的 67 MB 降到 fp8 的 33.5 MB 后，通信 lower bound 近似减半。

Ling-2.6-1T 支持 activation quantization，所以 V2 使用 dynamic per-token fp8。在我们的评估中没有观察到精度回退（见附录里的 AIME 2026 检查）。

#### In-kernel shared expert

Ling 每层还有一个 shared expert。如果它作为单独 dense MLP 运行，就会给关键路径增加一段。V2 把 shared expert 移到同一个 kernel 中，复用 routed expert 的 token / weight VMEM buffer，并把它安排在 scatter window 内。

shared expert 自身计算约 **0.159 ms**，但只给关键路径增加 **0.068 ms**，约 **2.7%**。原因很简单：shared expert 不需要跨 chip token dispatch，所需 token 都在本地，因此可以和 routed FFN 前面的 scatter 阶段 overlap。

### 6. 收益来自哪里

下面的 breakdown 展示了开启 activation quantization 和 in-kernel shared expert 后，prefill 16,384 的关键路径。带阴影的区域是真实工作，只是被其他阶段遮住了。

![Ling prefill critical-path breakdown](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-2/70328502-08a9-42c5-aeb7-bf6c0155c1aa.png)

图 5：Fused MoE V2 实测 overlap 结构。大部分 scatter/gather traffic 被藏在 routed expert window 下面，只有 scatter lead 和 gather tail 还暴露出来。

metadata block 是 routing bookkeeping：token 到 expert/device 的映射、per-expert offset/count，以及 scatter/gather index。它只移动少量 metadata，耗时几十微秒，不是 prefill 的核心成本。

对同一个 V2 kernel 做 ablation，可以看到关键路径上剩下什么：

| Ablation / component | Result | Interpretation |
|---|---:|---|
| full V2 production | 2.42 ms | 本节使用的标准 MoE prefill 延迟 |
| disable all expert matmuls | 相比 full 只少 2.2% | 纯 MXU compute 没有暴露出来 |
| visible scatter | 0.42 ms | communication lead 仍在关键路径上 |
| visible gather | 0.18 ms | gather tail 仍在关键路径上 |
| scatter + gather without compute to hide under | 约 2.4 ms | overlap 之前，真实通信量接近完整 kernel 延迟 |

这和成本模型吻合。即便包含 shared expert，理想 compute lower bound 也只有约 **0.36 ms**，移除 matmul 几乎不改变总延迟。scatter/gather 工作接近 2.4 ms，但其中约 1.8 ms 被隐藏在 routed compute window 下面。

因此，V2 的收益来自三件事：

- token 和 accumulator 常驻 VMEM，减少 token staging 和 HBM read-modify-write；
- expert weight double-buffered，让 HBM read 藏在 MXU work 后面；
- scatter/gather 使用 banked buffer 和 outbound/inbound ICI channel，与 routed compute overlap。

### 7. V2 之后还剩什么

overlap 后，图 5 中最长的剩余片段是 routed compute window，约占 **2.42 ms** 总时间的 68%。这不代表问题又回到了纯 FLOPs：Mosaic LLO dump 显示，剩余瓶颈主要是 fp8 packing / lane reorder / scale broadcast，以及 VMEM 对 tile size 的限制。

#### 通信受拓扑限制

在我们的测量里，flat all-to-all 优于 hierarchical all-to-all。flat 配置下，send/recv partition 直接由最终 expert owner 构建，一个 32-way all-to-all 把 routed token payload 从 source device 发到最终 target device。

我们也测了 hierarchical 配置：沿着 2×2×4 ICI torus 把 32-device exchange 拆开，先在本地维度内 reshuffle，再沿下一个维度 relay，直到每个 token 到达目标 expert 的 device。每轮通信范围更小，但同一份 routed token payload 要跨过多个 relay stage，会增加 staging buffer、同步边界，并且总搬运字节数几乎翻倍。两种模式都是在 fused kernel 外作为 standalone all-to-all benchmark 测得，因此不能直接和 kernel 内 trace 比较。

| Mode (payload = 16384 × 8192 × dtype_size) | bf16 | fp8 |
|---|---:|---:|
| flat all-to-all | 2.09 ms | 1.34 ms |
| hierarchical all-to-all | 3.12 ms | 1.88 ms |

所以通信侧真正有用的手段不是更复杂的 routing algorithm，而是更少的字节和更好的 overlap。Activation quantization 做的正是这件事。

#### Routed compute 受 VPU / VMEM 限制

Routed FFN1（W1+W3）实测约 **0.72 ms**，而理想 dense fp8 GEMM lower bound 约 **0.12 ms**。这个差距不是 activation quantization 造成的：开启 act quant 时 FFN1 约 0.74 ms，关闭时约 0.71 ms。

tile sweep 也显示当前配置接近局部最优：

| `bts` / `btc` | kernel latency | VMEM |
|---|---:|---:|
| **160 / 80** | **2.42 ms** | 47 MB |
| 160 / 160 | 2.44 ms | 47 MB |
| 128 / 128 | 3.12 ms | 44 MB |
| 256 / 128 | 3.19 ms | 54 MB |
| 256 / 256 | 3.23 ms | 54 MB |
| 384 / 128 | OOM | 62 MB |

Mosaic LLO dump 解释了原因。整个 kernel 只有 4096 条真实的 `vmatmul` 指令，而 fp8 layout 和 vector-side preparation 占据了 instruction stream 的大头：

| LLO instruction | Count | Role |
|---|---:|---|
| `vselect` | 50880 | sublane select / blend |
| `vbitcast` | 46566 | fp8 sub-word reinterpretation |
| `vcombine` | 36380 | sublane merge |
| `vpack_format` | 34368 | MXU input packing |
| `slane` | 29960 | sublane movement |
| `vunpack` | 25600 | fp8 unpack |
| `matmul_data_format` | 25600 | MXU 前的 format conversion |
| `vrot` | 21524 | lane rotation |
| `vmatres` / `vmatprep` | 17408 / 10240 | MXU drain / feed |
| `vslreplicate` | 6032 | scale broadcast |
| **`vmatmul`** | **4096** | 实际矩阵乘法 |

V2 避免了 per-block quantization 的 K-slicing，但 fp8 sub-word packing、scale broadcast 和 MXU feed/drain 仍然消耗大量 VPU / layout 工作。由于 VMEM 限制在 64 MB，`bts` 不能一直增大；tile 小的时候，这些固定成本很难彻底摊掉。

#### 小结

V2 隐藏掉大部分显式通信和 HBM weight movement 后，剩余瓶颈仍然是数据搬运，只是换了一种形式：fp8 layout work、VMEM 容量压力，以及如何持续给 MXU 喂数据。

- ICI all-to-all 受 torus topology 和 contention 限制。
- HBM weight read 必须用 double buffering 隐藏。
- fp8 packing 和 scale handling 会让 MXU 等数据变成正确形态。
- VMEM 容量限制了 tile size，也限制了能同时存在的 overlap buffer 数量。

下一步需要改变约束本身：

- **Kernel 侧：** 减少 fp8 pack/unpack 和 scale handling，但这越来越依赖让模型量化方式对齐 TPU-native execution format，比如 TPU-friendly scale granularity、fp8 layout，或者未来 MXU-native 低精度格式，例如 FP4 或 MXFP8。
- **Workload 侧：** 跨 batch 做 overlap，让 routed window 可以和其他 layer work 并行。
- **Hardware 侧：** 提供更适合 all-to-all 的互联拓扑，或者提供更大的 VMEM / 更高的 ICI 带宽。

关于未来 TPU 硬件，可以参考 Google Cloud 的 [TPU 8t and TPU 8i technical deep dive](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive)。

## Ling-2.6-1T Bring-up

MoE fusion 只是让 Ling-2.6-1T 在 TPU 上跑好的其中一部分。其余 bring-up 工作主要是让 runtime 匹配模型的 hybrid backbone：给 full-attention layer 和 linear-attention layer 分配不同状态，让 GLA prefill 和 decode 走 TPU-friendly kernel，并映射 DP/TP，使 grouped RMSNorm 保持 chip-local。

### Hybrid Memory Pools

Ling-2.6-1T 并没有向 runtime 暴露一个统一的 attention state。它的 10 个 MLA full-attention layer 会写 token-indexed KV cache，而 70 个 Lightning / GLA layer 携带 request-indexed recurrent state。因此 allocator 必须同时管理两种容量：MLA 的 resident history tokens，以及 linear-attention layer 的 active request slots。

这里的单位很容易读错。TP=4、bf16 KV、fp32 recurrent state 时，10 个 full-attention layer 的 MLA KV cache 大约每 device 每 token 12.5 KiB。70 个 linear layer 的 Lightning recurrent state 大约每 device 每 request 70 MiB。只有把这两个数字放回 request 中才有意义：一个 16K-token prompt 每个 request 大约需要 200 MiB MLA KV，一个 256K-token prompt 大约需要 3.1 GiB，而 recurrent state 一直约为 70 MiB。recurrent state 是固定并发成本；KV cache 是 token 容量成本，会随上下文长度线性增长。

SGLang-JAX 把这两类 state 分离，同时保留一个 request lifecycle：`HybridLinearKVPool` 只为 10 个 full-attention layer 保存 KV（70 个 linear layer 不消耗 KV slot），`RecurrentStatePool` 为每个 active request 保存一个 fp32 recurrent slot，`HybridReqToTokenPool` 把两者连接起来：request 在 admission 时同时申请两者，在 finish 时同时释放。chunked prefill 和 decode 会从同一个 recurrent slot 继续，而不是按 chunk 或 token 分配新 state。HBM budget 也按同样方式切分：一部分可配置比例留给 recurrent slot，用来限制 concurrency；剩余部分给 KV cache，用来限制 resident token。

JAX 还带来一个额外约束：runtime 不能像 CUDA 路径那样原地更新这些 buffer。SGLang-JAX 把 KV pool 和 recurrent pool 包装进一个 `MemoryPools` pytree，并作为 donated JIT argument 传给模型。每次 forward pass 返回更新后的 pool buffer，runtime 再通过 `replace_all()` 写回。这让 buffer donation、TP/DP sharding 和未来 pool 扩展都停留在 container 层，而不是把特殊逻辑散落在 forward loop 里。

### GLA（Gated Linear Attention）

每个 GLA layer [7] 把历史保存在固定大小的 recurrent state 里，而不是为每个过去 token 存一个 KV entry。它的更新可以写成：

$$
S_t = \gamma_t\, S_{t-1} + k_t^\top v_t, \qquad o_t = q_t\, S_t
$$

这会把 attention history 从逐 token 增长的数据结构，变成每个 active request 一个 state tensor。长上下文下，这就是主要收益：history 的计算仍然线性，state 大小固定，而不是 materialize 并读取不断增长的 KV history。

**Prefill：让 recurrence 足够并行，适合 TPU。** 直接看上面的 recurrence，它是串行的：token *t* 依赖 token *t-1* 衰减并更新后的 state。用这种方式跑 prefill 会把一个 16K 或 256K prompt 变成很长的逐 token scan，这正是 TPU 不擅长的形态。

SGLang-JAX 使用数学等价的 chunk-wise 形式。sequence 被切成固定 64-token chunk。chunk 之间，上一个 chunk 的 final state 作为下一个 chunk 的 initial state，因此长程依赖仍然沿时间向前推进。但在 chunk 内部，recurrence 被重排为 token block 上的 dense matrix operation。只有 chunk boundary 保留串行；每个 chunk 内的工作作为 block-parallel TPU math 执行。

**Decode：recurrence 的自然形态。** decode 更简单：prefill 已经把 prompt 折叠进 recurrent state，所以每个新 token 读取当前 request 的 state，执行一次 recurrent update，输出 attention result，然后写回新 state。问题从长序列并行转向了高效的小 state update。

**Serving 集成：让 GLA 留在同一条 runtime path 里。** GLA 作为 layer-level backend choice 集成，而不是单独的 scheduler mode。full-attention layer 读写 KV cache；GLA layer 读写 recurrent state；两者都在同一批 prefill 和 decode batch 中前进。scheduler 仍然只看到一套 lifecycle：admit、prefill、decode、release。

这个集成在功能上已经完整，但 prefill kernel 还没有被优化到 Fused MoE V2 的程度。GLA 数学本身不需要变；需要变的是执行 schedule。

### Single-Controller Data Parallelism Support

Ling-2.6-1T 的 grouped post-attention RMSNorm 对 tensor parallelism 有硬约束。每个 norm group 包含 8 个 head。如果一个 group 跨 chip，variance 计算就会在每一层变成跨 chip reduce，并且直接落在 decode 关键路径上。纯 TP 因此没有好设置：tp <= 8 可以让 norm group chip-local，但会让万亿参数模型并行不足；tp > 8 会切开 norm group，并付出 all-reduce 成本。

Single-controller DP 通过把 data parallelism 当成另一个 mesh axis 来解决这个矛盾。mesh 被拆成 DP group；每个 group 使用足够小的 TP，以保持 grouped RMSNorm chip-local，请求则分配到不同 DP rank。权重仍然在每个 DP group 内做 TP shard。per-layer norm reduce 消失，释放出来的 ICI/HBM budget 可以用于更高并发。

关键设计是：DP 是 SPMD runtime 的一部分，而不是一组独立 server replica。SGLang-JAX 运行一个逻辑 scheduler，`dp_rank` 会绑定到 request、KV allocation 和 prefix-cache key 上。这让系统可以从一个 load snapshot 做全局 admission control，跨 host 构造确定性的 batch，并维护一个全局 prefix-cache 结构，其中 entry 以 `(dp_rank, prefix)` 作为 key。

这也能和 hybrid runtime 其他部分干净组合。把 mesh 扩到更大配置，比如在每个 data-parallel group 内加入 tensor parallelism，本质上是 mesh-shape change，而不是 scheduler fork，因此 memory pool、batching path 和 attention backend 都保持同一套心智模型。

## 实验和 benchmark

所有 TPU 结果都使用 SGLang-JAX，在同一个 TPU v7x slice 上服务 Ling-2.6-1T；V1/V2 ablation 的设置完全相同，只有 MoE kernel config 不同。

### Benchmark 配置

- **硬件：** TPU v7x，16 chip（2×2×4 ICI torus）-> 32 device
- **并行：** tp = ep = 32, dp = 8
- **模型：** Ling-2.6-1T，bf16 activation，per-channel fp8 MoE weight
- **数据集：** SGLang 默认 `random` benchmark 数据集（从 ShareGPT 采样）
- **Runtime：** SGLang-JAX（JAX 0.8.1），dvfs p_state=7
- **输入长度：** 16384
- **Prefill：** output 1, concurrency 128
- **Decode：** output 1024, concurrency 128 / 512

![Ling-2.6-1T prefill throughput, Fused v1 vs v2](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-2/a5d32f67-78ff-45dd-8970-a0b95a55b0c6.png)

Prefill input throughput，输入长度 16384 token，mc=128。设置完全相同，只有 MoE kernel config 不同：Fused v1 -> v2 base -> v2 +act-quant -> v2 +act +SE-overlap（+24.8%）。

![Ling-2.6-1T peak decode output throughput, Fused v1 vs v2](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-1/088caf7a-961b-4994-9898-f25592a10c7d.png)

Peak output（decode）throughput，输入 16384 token，输出 1024 token，配置为 np=512/mc=128 和 np=2048/mc=512。百分比表示相对 Fused v1 的提升。

![Ling-2.6-1T TPU vs GPU, same model and workload](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-2/6ca8c531-6fac-4bb6-9133-e898429c4ff9.png)

图 6：完整 TPU-vs-GPU 对比：TPU v7x-16（fused_v2）对比 GPU H200×16（2 nodes, tp8·pp2）。两边使用同一个模型和同一个 SGLang bench workload，各自 16 个 accelerator。关于 prefill 差距见下面的说明。

> **关于端到端 prefill 和 MoE-kernel speedup 的说明：** Fused MoE V2 kernel 把 MoE layer prefill 延迟降低约 53%（device trace），但端到端 prefill 吞吐只提升约 25%（v1 -> v2）。MoE layer 已经不再是 prefill 的主导成本：GLA（gated linear attention）prefill kernel 现在是主要瓶颈，而且还没有被优化到同等程度，因此稀释了端到端 prefill speedup。这也是为什么图 6 中 TPU v7x-16 在 prefill 列落后于 H200×16，但在两个 decode 点都领先。让 GLA prefill kernel 跟上来是正在进行的工作，我们预计这会释放更大的端到端 prefill 收益。

## 限制和未来工作

本次发布的 Ling-2.6-1T 支持有意保持了范围约束；还有几项我们正在做的后续工作：

- **GLA / Linear-Attention prefill kernel。** 如 benchmark 部分所说，GLA（Lightning Linear）prefill kernel 现在是主要 prefill 成本。通过更好的 chunking/tiling、融合 gating 和 recurrent-state update、以及复用 MoE kernel 中同样的 MXU/VPU/DMA-overlap 处理方式来把它优化到同等水平，是剩余最直接的端到端 prefill 优化方向。
- **Dynamic Expert-Parallel Load Balancing（EPLB）。** 当前 `FusedEPMoE` 路径使用静态 expert-to-device placement，但真实 workload 下 256 个 routed expert 的命中率并不均匀。一个 dynamic EPLB pass 如果能根据观测流量周期性重新平衡 expert-to-rank 映射，就能缩小 peak 和 average per-device utilization 之间的差距，尤其是在更高 batch size 下。
- **Hybrid memory pool 上的 Radix cache。** SGLang 的 RadixAttention [9] prefix cache 假设只有一个 per-token KV pool；而 Ling-2.6-1T 同时混合了 per-token KV 和 per-request recurrent state，因此朴素 prefix-share 会在 linear layer 上悄悄混合不同 request 的 state。我们正在设计一个扩展：按 token prefix 共享 MLA KV，同时对每个共享 prefix snapshot 并重新 key recurrent state，让共享 system prompt 和长 agent trace 可以在不损失正确性的前提下复用。
- **MTP / EAGLE speculative decoding。** Ling-2.6-1T checkpoint 带了一个 EAGLE-style MTP head（3 个 speculative step、4 个 draft token、top-k 1）。当前路径只运行 base-model decode；把 MTP head 接入 SGLang-JAX speculative-decoding runtime，是 decode 吞吐的下一个里程碑。hybrid memory-pool layer 已经考虑了 draft-step state，剩下的工作在 verifier 和 draft-acceptance kernel 上。

## 附录

### 成本模型中使用的 TPU v7x 规格

TPU v7x 公开规格列出了每 chip 约 4.614 PFLOP/s fp8 计算能力、7.38 TB/s HBM 带宽，以及 1.2 TB/s 双向 ICI 带宽。在这次部署中，每个 chip 暴露为两个 device，因此成本模型部分的 per-device lower bound 使用大约一半的 chip 级计算能力和带宽。关于 TPU memory hierarchy 和执行单元（MXU、VPU、VMEM、HBM、ICI）的背景，可以参考 Google Cloud 的 [TPU system architecture](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm)。

### 性能复现

两边使用同一个模型和同一个 SGLang benchmark workload：
prefill（out 1, mc 128）· decode（out 1024, mc 128）· decode（out 1024, mc 512）。

**TPU：SGLang-JAX（Fused MoE V1 / V2）。** TPU v7x，16 chip（2×2×4 ICI torus -> 32 device），tp = ep = 32，dp = 8，per-channel fp8 MoE weight。

TPU run 使用 sgl-jax branch `fused-moe-v2-with-sp-rs` @ `49c2ed1` 和 image `jax-ai-image/tpu:jax0.8.1`。

V1/V2 ablation 只改变 MoE flags：Fused v1 = `--moe-backend fused`；v2 base = `fused_v2 --no-moe-fused-act-quant --no-moe-fused-shared-experts`。

v2 +act-quant case 增加 `--moe-fused-act-quant`；v2 +act +SE-overlap 两个都打开。两个 external-shared-expert config 使用 `--mem-fraction-static 0.85`，因为它们在 0.88 下会 OOM。

**GPU：SGLang（H200×16，reference）。** 2 nodes × 8× H200，tp = 8，pp = 2；使用和 TPU run 相同的模型与 benchmark workload。

完整性能 benchmark 命令见 [SGLang-JAX cookbook][ling-26-cookbook]。

### Server launch 和精度复现

AIME 2026 检查使用 `MathArena/aime_2026`，30 道题，pass@1 为 **26 / 30 = 86.7%**。这次运行没有 request error，所有 response 都正常结束（`finish_reason=stop`，没有在 32768 token 截断）。这说明 fp8 fused-MoE serving path 没有明显精度回退。

完整 launch-server 命令、request 和 tool-calling 示例，以及 AIME 2026 精度复现，见同一个 [SGLang-JAX cookbook][ling-26-cookbook]。

[ling-26-cookbook]: https://github.com/sgl-project/sglang-jax/blob/main/docs/cookbook/autoregressive/InclusionAI/Ling-2.6.md

## 参考资料

[1] [Ling-2.6-1T model card](https://huggingface.co/inclusionAI/Ling-2.6-1T)

[2] [Hybrid models meet SGLang (blog)](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/)

[3] [Ragged Paged Attention](https://arxiv.org/abs/2604.15464)

[4] [Fused MoE V1 kernel, tpu-inference](https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/kernels/fused_moe/v1/kernel.py)

[5] [Fused MoE V1 kernel adapted in SGLang-JAX](https://github.com/sgl-project/sglang-jax/blob/main/python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py)

[6] [DeepSeek-V2 (MLA)](https://arxiv.org/abs/2405.04434)

[7] [Gated Linear Attention (GLA)](https://arxiv.org/abs/2312.06635)

[8] [MiniMax-01 (Lightning Attention)](https://arxiv.org/abs/2501.08313)

[9] [SGLang (RadixAttention)](https://arxiv.org/abs/2312.07104)

## 致谢

**AntGroup-ASystem Core Team：** Zhenxuan Pan, Guowei Wang, Yuhong Guo, Shuo Wan

**SGLang-JAX team：** jimoosciuc, Prayer, aolemila, neo, leos, pathfinder-pf, Haolin Fu, Qinghan Chen, JamesBrianD, Haoguang Cai, Yuhao Hu, cjx0709, Zhengke Zhou, Yuxin Wei, Lianfang Wang, 0xaskr
