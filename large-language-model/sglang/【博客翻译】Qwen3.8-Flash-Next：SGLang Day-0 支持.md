![Qwen3.8-Flash-Next：SGLang Day-0 支持](https://files.mdnice.com/user/59/82b497c3-5c7c-4057-af06-e78876b1286a.png)

> SGLang 仓库地址： https://github.com/sgl-project/sglang
>
> 原文地址（https://www.lmsys.org/blog/2026-08-26-qwen-flash-next）
>
> GitHub Markdown 原文（https://github.com/lm-sys/lm-sys.github.io/blob/4b0995cc114d3877a38d95bb3c0622a0e9e40dc1/blog/2026-08-26-qwen-flash-next.md）

# Qwen3.8-Flash-Next：SGLang Day-0 支持

SGLang 团队，2026 年 8 月 26 日

## 引言

今天，Qwen 团队开源了 <strong>Qwen3.8-Flash-Next</strong>。这是一个多模态 MoE 模型，也是 <strong>Qwen4</strong> 架构的早期预览版。它之于 Qwen4，就像 Qwen3-Next 之于 Qwen3.5。Qwen3.5 到 Qwen3.8 一直沿用 <strong>Gated DeltaNet + Gated Attention</strong> 混合设计。通过与 Qwen、NVIDIA 和 AMD 团队合作，SGLang 在模型发布首日就提供了完整支持。

<strong>Qwen3.8-Flash-Next</strong> 在多个方面升级了模型架构：

- <strong>GDN + QSA 混合注意力</strong>：Gated DeltaNet（GDN）高效压缩历史信息，Qwen Sparse Attention（QSA）则使用轻量级索引器，以微块粒度选取重要上下文，从而控制长序列的注意力成本。
- <strong>门控残差（Gated Residual，GR）</strong>：把残差流扩展为 4 个分支，并通过动态门控控制信息的读取和写回，加强跨层信息流动。
- <strong>N-gram Embedding</strong>：根据局部上下文执行查表，为常见短语和局部模式补充额外表示，只增加很少的计算量就能扩展模型容量。

<strong>亮点</strong>：

- <strong>混合架构</strong>：主模型包含 125B 参数，另有 51B 参数的 N-gram Embedding，每个 token 激活 6B 参数。模型共 48 层，其中 36 层是 GDN 线性注意力，12 层是 QSA 稀疏注意力。MoE 层包含 512 个专家，采用 top-10 路由。
- <strong>我们量化的 NVFP4 checkpoint</strong>：RadixArk/Qwen3.8-Flash-Next-NVFP4（https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4），同样在首日发布。
- <strong>N-gram Embedding</strong>：把 N-gram embedding 卸载到 host memory，可大幅降低 GPU 显存占用；异步预取与模型计算重叠，几乎不增加额外开销。
- <strong>门控残差</strong>：SGLang 与 NVIDIA 联合开发，并通过 FlashInfer 发布。Mix/Combine HyperConnection 算子使用低延迟的单 GEMM 路径，kernel 级加速达到 2.05 倍。
- <strong>GDN + QSA</strong>：为 GDN + QSA 混合架构实现了 KV Cache 显存管理，并兼容 Radix Cache。
- <strong>推测解码</strong>：为 MTP 草稿模型加入索引复用，减少长上下文下草稿模型的索引器耗时。在 B200 上使用 TP4 时，NVFP4 checkpoint 配合 MTP，在 batch size 1 下达到 <strong>540 tok/s</strong>，平均接受长度为 3.3（包含 bonus token）。

启动命令以及针对不同 workload 的配置建议，请参阅 SGLang Cookbook（https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next）。

## 模型架构

<p align="center">
  <img src="https://files.mdnice.com/user/59/c7dbbc95-2de9-41af-ab6a-001ce038a205.png" alt="Qwen3.8-Flash-Next 模型架构" width="60%">
</p>

- <strong>GDN + QSA 混合架构</strong>：Qwen3.8-Flash-Next 延续了 Qwen3.5 引入的 GDN + Attention 混合设计。每 4 层中，3 个 GDN 层把历史压缩到固定大小的状态，剩下 1 层对完整上下文执行精确检索。对于全局 Attention 层，Qwen3.8-Flash-Next 又引入了 <strong>Qwen Sparse Attention（QSA）</strong>。随着上下文变长，计算量和 KV Cache 的访存成本都会明显增长；稀疏注意力只关注重要上下文，因此可以减少长序列计算。QSA 在此基础上把序列聚合为微块，在块级别估计重要性，再选出最相关的区域，同时降低索引开销和注意力成本。
- <strong>门控残差（GR）</strong>：GR 结合了两个思路。它沿用 Hyper-Connection，把残差流扩展为多个分支；同时把 GatedNorm 式的<strong>逐元素动态门控</strong>引入残差读取。原来单一的残差流被扩展为 <strong>4 个并行分支</strong>，模型可以根据当前内容动态决定从每个分支读取多少信息，以及向每个分支写回多少信息。
- <strong>N-gram Embedding</strong>：模型根据“当前 token 及其前几个 token”组成的局部上下文查表，为常见短语和局部模式补充表示，而每个 token 几乎不用承担额外计算。N-gram Embedding 可以完全放在 host memory 中以节省显存：系统提前计算查表位置并异步预取，因此它不需要常驻 GPU。最终，模型只在网络前部附近使用<strong>一个 N-gram Embedding 层</strong>，以较低成本加入大规模的“局部模式记忆”。
- <strong>IndexShare MTP</strong>：`draft-extend` 会针对目标模型刚刚接受的 token 计算 QSA top-k 选择结果。这份结果在整个 MTP 迭代中保持不变，因此每个 draft decode step 都可以跳过索引器，直接读取冻结的选择结果，再加上捕获后新起草的位置。对于长上下文，这能显著加快 MTP 的草稿生成步骤。

## Qwen Sparse Attention：粗粒度检索，精确执行注意力

Qwen3.8-Flash-Next 使用压缩比为 4 的 QSA，也就是 <strong>c4</strong>。每个 QSA 层有两条路径：轻量级索引器负责决定“看哪里”，稀疏 GQA 则从原始 Attention K/V Cache 中读取被选中的条目。

<p align="center">
  <img src="https://files.mdnice.com/user/59/4814b44f-7c35-466c-87e3-1e43a8012d04.png" width="98%" alt="SGLang 中 QSA 索引和注意力的数据流">
</p>

索引器会投影出 4 个 128 维 query head，以及 1 个共享 key head。每 4 个原始索引 key 先用 FP32 求平均，再执行归一化，并使用第一个 token 的 MRoPE 位置进行旋转，得到一个压缩 key。Query 按下式对所有可见的压缩块打分：

$$
s_{t,b} = \frac{1}{\sqrt{128}}
\sum_{h=1}^{4}
\mathrm{ReLU}
\left(\left\langle q^I_{t,h}, \bar{k}^I_b \right\rangle\right).
$$

QSA 保留得分最高的 512 个块，把它们重新展开为 2048 个逻辑 token 位置，再附加当前未完整块中的 0 到 3 个 token。因此，最终的稀疏注意力最多只需要处理 2051 个位置。这里有一点很重要：压缩 key 只用于索引，最后的 softmax 和 value 聚合仍然使用<strong>原始、未压缩的 K/V</strong>。

这相当于用少量 Cache 容量换取更低的长上下文计算量和显存流量。索引器只需扫描约 $L/4$ 个小 key，稀疏注意力随后读取约 2K 个完整 K/V 条目，而不再读取全部 $L$ 个条目。模型级 KV 节省来自混合层布局：48 层中只有 12 层需要保存随上下文增长的 Attention K/V，其余 36 个 GDN 层使用固定大小的状态；它并不是靠丢弃 QSA 层内部的 K/V 实现的。

SGLang 只在全注意力层上挂载索引器，并复用这些层的 MRoPE 实现。原始 K/V 仍保存在常规的分页内存池中。QSA 每 4 个 token 额外保存一个 BF16 压缩索引 key；未完整块中的原始 key，则放在每个请求各自拥有的 4-slot 环形缓冲区里。这样不必为完整上下文保留全部原始索引 key，可将 QSA 索引 Cache 的额外开销降低 80%。页对齐的 `full_slot / 4` 寻址方式，让压缩 Cache 可以直接跟随 Radix Cache 的所有权，不需要单独维护生命周期。

在 prefill 阶段，一个定制 GPU kernel 负责计算索引分数，快速 top-k 选出目标块，随后 Triton 展开索引并运行稀疏 GQA。Decode 使用同一个打分器的分页版本，压紧被选中的原始 K/V，并在 Blackwell 上分派到 TRTLLM-Gen，在其他平台上分派到 packed FlashAttention。索引器可以在第二条 CUDA stream 上与主 Q/K/V projection 重叠，metadata 路径也兼容 CUDA Graph。

## IndexShare MTP：在多个草稿步骤之间复用 QSA 选择结果

QSA 层先运行<strong>索引器</strong>，选出需要关注的 token，再对这些 token 执行稀疏注意力。第二阶段的 token 预算固定；第一阶段则需要用 query 对所有 `⌈L/4⌉` 个压缩块打分。因此，当上下文超过几千个 token 后，决定这一层成本的不再是稀疏注意力，而是为它提供索引的索引器。推测解码会放大这项成本：当配置为 `--speculative-num-steps N` 时，一次 MTP 迭代最多把草稿向前推进 `N` 个位置，却要调用 `N` 次索引器，其中包括 `N - 1` 次 draft decode forward 和 1 次 `draft-extend`。

为此，draft decode step 完全不再运行索引器。每次 MTP 迭代都先对目标模型刚刚接受的 token 执行一次 `draft-extend`，而这次 forward 原本就需要运行索引器。系统在这里捕获每个请求最后一行被接受 token 的选择结果，并在整个草稿循环中复用。查表时还会补入 `N + 1` 列，填上捕获之后新起草的位置，因此草稿模型仍然能看到自己正在生成的 token。选择结果是一组<strong>逻辑</strong> token 索引，而请求只会增长，所以这些索引永远不会越界。与此同时，query 在长度为 `L` 的上下文中最多只移动 `N` 个位置，复用的排序结果与重新计算的结果基本相同，接受长度也不会变化。这样一来，每次 MTP 迭代中，草稿模型的索引器调用次数便从 `N` 次降为 1 次。仅用于给索引器提供输入的小型 metadata kernel 也可以从 draft decode step 中删除，其中包括压缩 decode view、pending-ring 和 group-ring 布局。

## HyperConnection Kernel 优化

HyperConnection（HC）维护 4 条并行残差流，而 Attention 和 MoE 只处理一个 hidden state。因此，每个 block 都要先用 <strong>Mix</strong> 从 4 条残差流中读取信息，再用 <strong>Combine</strong> 把输出写回。这里的 `M` 表示一次调用处理的 token 数：decode 和推测验证时 `M` 较小，prefill 时则可能达到数千。SGLang 会根据 `M` 选择不同的 kernel。

### Mix

Mix 通过低秩投影生成逐元素 gate，并把 4 条残差流归约为一个 hidden state。当 `M ≤ 16` 时，我们使用 FlashInfer PR #4266（https://github.com/flashinfer-ai/flashinfer/pull/4266）中的低延迟 split-K CuTe GEMM。Split-K 切分 K 维，让多个 CTA 并行处理同一输出区域，以弥补 M 维并行度不足的问题。SiLU、Sigmoid、门控和最终归约被融合进两个 GEMM epilogue，避免把中间结果写入 global memory。Up-projection 权重会在线下重排，使每个输出对应的 4 个 gate 值可以直接在 tile 内完成局部归约。`M` 较大时则使用 cuBLAS，它在这些 shape 上效率更高。

在 NVIDIA B300 上，当 `M = 4` 时，这条融合路径把 Mix 延迟从 12.36 微秒降至 6.03 微秒，获得 <strong>2.05 倍的 kernel 级加速</strong>。在端到端推测解码测试中，与之前的 Triton 路径相比，吞吐提升了 <strong>7.6%</strong>。

### Combine

Combine 会计算 4 个注入系数，并把残差更新应用到 4 条流上。当 `M` 较大时，一个融合 kernel 就能单趟处理每个 token row。`M` 较小时，这种映射能提供的 CTA 太少，因此 `M ≤ 32` 的路径会沿 hidden dimension 切分每一行。这套双 kernel 实现既提供了足够的并行度，也保留了参考实现的 FP32 累加顺序，输出可做到 bitwise identical。

当 `M = 4` 时，切分路径把 Combine 延迟从 4.17 微秒降至 2.13 微秒，获得 <strong>1.96 倍的 kernel 级加速</strong>。另一组端到端测试以原始的“每行一个 CTA”kernel 为基线，吞吐提升了 <strong>5.49%</strong>。对于较大的 `M`，融合 kernel 最多比基于 cuBLAS 的 baseline 快 <strong>2.54 倍</strong>，有效带宽达到 6144 GB/s。

根据 shape 分派执行路径后，HC 可以同时兼顾低延迟 decode 和大规模 prefill。

## 逐层嵌入（Per-Layer Embeddings，PLE）

### 架构

模型把 PLE 放在第二个 decoder block 中，配置的 layer ID 为 2，对应从 0 开始计数的索引 1。PLE 是一种<strong>通过哈希寻址、可学习的 N-gram embedding memory</strong>。它包含 51.2B 个 embedding 参数，在 BF16 下约占 95.4 GiB；这些数据是固定的模型权重，不是 KV Cache，也不是会变化的 Attention memory。

对于 token `x_t`，8 个 2-gram 哈希 head 使用 `(x_{t-1}, x_t)`，另外 8 个 3-gram 哈希 head 使用 `(x_{t-2}, x_{t-1}, x_t)`，一共生成 16 个 embedding row ID。每一行提供 160 个值，拼接后得到 shape 为 `[2560]` 的 `E_t`。

<p align="center">
  <img src="https://files.mdnice.com/user/59/59854777-8ca2-416d-a049-4a74d5335466.png" width="98%" alt="PLE 数据流，以及 SGLang 基于 pinned host memory 的稀疏卸载路径">
</p>

<p align="center">
  <em><b>第二个 decoder block 中的 PLE。</b>稀疏 N-gram 检索先通过 gate 注入 4 个 HC 分支，再执行 HC Mix。SGLang 把 vocabulary-parallel table 的 shard 放到 pinned host memory 中，每个 token 只收集被选中的 16 行。</em>
</p>

$$
E_t \in \mathbb{R}^{2560}
\longrightarrow
K_t \in \mathbb{R}^{4 \times 2560},
\qquad
V_t \in \mathbb{R}^{2560}
$$

$$
R_t \in \mathbb{R}^{4 \times 2560}
\longrightarrow
Q_t \in \mathbb{R}^{4 \times 2560}
$$

$$
g_t = \mathrm{Gate}(\mathrm{Norm}(Q_t), \mathrm{Norm}(K_t))
\in \mathbb{R}^{4 \times 1},
\qquad
U_t = g_t \odot V_t
$$

$$
\Delta_t = U_t + \mathrm{SiLU}(\mathrm{DWConv}(\mathrm{RMSNorm}(U_t)))
$$

$$
\widetilde{R}_t = R_t + \Delta_t,
\qquad
\widetilde{R}_t \xrightarrow{\mathrm{HC\ Mix}} h_t \in \mathbb{R}^{2560}
$$

第四行把门控后的 value 与其短卷积输出相加，形成 PLE delta；第五行再把这个 delta 注入 HC state。PLE 会为每个请求保留两类局部状态：用于哈希的最近两个 token ID，以及 shape 为 `[10240, 9]` 的短卷积历史。目标模型在 prefill、decode 和 target verification 阶段都会保留 PLE；只有单层 MTP 草稿模型会关闭它。

### 基于 Pinned Host Memory 的稀疏卸载

每个 token 只会访问 16 行，因此 SGLang 把各个 rank 的 vocabulary-parallel table shard 放在 pinned host memory 中，再用 Triton UVA kernel 把选中的行收集到一个很小的 BF16 GPU buffer。专用 CUDA stream 让这次 gather 与第一个 decoder block 的计算重叠。原有 TP reduction 和 DP gather/scatter 路径都保持不变：卸载改变的只是存储位置，不会改变 table 的所有权和 PLE 的计算方式。当模型的实际 dtype 为 BF16 时，这条 CUDA 路径会默认开启；它也独立于 KV Cache 卸载或通用的 layer offload。

在 H200 上使用 TP4 和 MTP-213（2 个 draft step、top-k 为 1、每次目标模型验证 3 个 draft token）时，卸载把每张 GPU 上的目标模型权重从 83.91 GiB 降至 60.45 GiB，减少了 <strong>23.46 GiB</strong>；在相同 memory fraction 下，可分配的 KV 容量从 184 万 token 增加到 328 万 token，提升 <strong>78.54%</strong>。并发请求数为 1、2 和 4 时，对齐后的吞吐几乎没有变化，几何平均值只下降 <strong>0.07%</strong>。测试还使用了 4 个固定 prompt，每个 prompt 生成 128 个 token，其输出 ID 完全一致；第一组测试记录的 chosen-token logprob trace 也完全一致。

## 致谢

这项工作由 RadixArk 的 SGLang 团队、Qwen、NVIDIA 和 AMD 共同完成。

<strong>SGLang 社区</strong>：Qiaolin Yu、Yuhao Yang、Cheng Wan、Xinyuan Tong、Zijie Xia、Ke Bao、Mingyi Lu、Haoguang Cai、Banghua Zhu、Ying Sheng

<strong>Qwen</strong>：Yi Zhang、Yizhong Cao、Guangda Liu

<strong>AMD</strong>：Andy Luo、Haichen Zhang

<strong>NVIDIA</strong>：NVIDIA 与 SGLang 联合优化了 Qwen3.8-Flash-Next 在 Blackwell 和 Hopper 上的性能。
