> 代码：https://github.com/pytorch/FBGEMM/blob/main/fbgemm_gpu/experimental/gen_ai/src/attention/gqa_attn_splitk.cu

> PyTorch 博客：https://pytorch.ac.cn/blog/int4-decoding/


# 用于 LLM 推理的 INT4 Decoding GQA CUDA 优化

## 介绍

生成式 AI 以其像人类一样生成内容的能力席卷全球。许多这些生成式 AI 工具都由大型语言模型 (LLM) 提供支持，例如 Meta 的 Llama 模型和 OpenAI 的 ChatGPT。LLM 的主要挑战之一是支持大型“上下文长度”（也称为“序列长度”）。上下文长度是指模型用来理解输入上下文和生成回复的token数量。更长的上下文长度通常会转化为回复中更高的精度和质量。但是，较长的上下文长度在计算和内存方面都需求很高。这主要是因为以下原因：

- Attention的计算复杂度随着上下文长度的增加而线性增长（增长率取决于注意力算法）。因此，在使用较长的上下文长度时，Attention层可能会成为瓶颈，尤其是在prefill阶段，此时 Attention 是compute bound的。
- KV Cache的大小与上下文长度线性增长，因此对内存需求造成更大的压力，从而减慢了本来就memory bound的 attention decoding。此外，由于内存容量有限，当 KV Cache变大时，批次大小会减小，这通常会导致吞吐量下降。

与上述其他问题相比，计算复杂度增长难以解决。解决 KV Cache 大小增长问题的一种方法是使用低精度 KV Cache。从我们的实验中可以看出，在 Meta Llama 2 推理的解码阶段，分组 INT4 量化在准确性方面与 BF16 KV Cache 相比具有可比性。但是，尽管在注意力解码层中读取的数据量减少了 4 倍，但我们没有观察到任何延迟改进。这意味着 INT4 注意力在利用宝贵的 HBM 带宽方面，效率比 BF16 注意力低 4 倍。

在本说明中，我们将讨论我们应用于 INT4 GQA（分组查询注意力——我们在 LLM 推理阶段使用的注意层）的 CUDA 优化，这些优化将性能提高了 **在 NVIDIA A100 GPU 上最高 1.8 倍** 和 **在 NVIDIA H100 GPU 上最高 1.9 倍**。

- 优化的 CUDA INT4 GQA 优于 INT4 Flash Decoding GQA（我们之前提到的实验中使用的性能最佳的 INT4 GQA, https://pytorch.org/blog/flash-decoding/ ）。 **在 A100 上快 1.4-1.7 倍 和 在 H100 上快 1.09-1.3 倍。**
- 优化的 CUDA INT4 GQA 的性能优于 BF16 快速解码 GQA 。**在 A100 上快 1.5-1.7 倍，在 H100 上快 1.4-1.7 倍。**

## 背景

### 用于 LLM 推理的 GQA

分组查询注意力 (GQA) 是多头注意力 (MHA) 的一种变体，其中每个 KV Cache头在查询头组之间共享。我们的 LLM 推理在预填充和解码阶段都采用 GQA 作为注意层，以减少对 KV Cache的容量需求。我们在推理中使用多个 GPU，其中 KV Cache和查询头分布在各个 GPU 上。每个 GPU 运行一个注意层，该层包含一个 KV 头和一组 Q 头。因此，从单个 GPU 的角度来看，GQA 组件也可以被描述为 MQA（多查询注意力）.

图 1 说明了Decoding GQA 的简化工作流程。GQA 接收三个主要输入：输入查询（用 $Q$ 表示）、K 缓存（用 $K$ 表示）和 V 缓存（用 $V$ 表示）。我们当前的 GQA 推理对 $Q、K$ 和 $V$ 使用 BF16。

- Q 是一个形状为 $(B, 1, H_Q, D)$ 的 4D BF16 张量
- K 是一个形状为 $(B, T_{max}, H_{KV}, D)$ 的 4D BF16 张量
- V 是一个形状为 $(B, T_{max}, H_{KV}, D)$ 的 4D BF16 张量

其中，

- $B$ 是批次大小（输入提示的数量）
- $H_Q$ 是查询头的数量
- $H_{KV}$ 是KV 头的数量（$H_Q$ 必须可被 $H_{KV}$ 整除）
- $T_{max}$ 是最大上下文长度
- $D$ 是头维度（固定为 128）

GQA 只是 $bmm(softmax(bmm(Q, K^T) / sqrt(D)), V)$ 。这将生成一个单一的输出张量（用 $O$ 表示），它是一个形状与 $Q$ 相同的 4D BF16 张量。请注意，矩阵乘法是使用 BF16 执行的，但是累加和 softmax 是在 FP32 中执行的。我们将其称为“BF16 GQA”，因为 KV 缓存是 BF16。

![图 1 用于 LLM 推理的 BF16 GQA 的简化工作流程](https://files.mdnice.com/user/59/280fe9b0-daf6-4657-bd0e-ad8ef760590c.png)

### INT4 GQA

为了进一步减小 KV Cache的大小，我们探索了使用 INT4 而不是 BF16 来存储 KV Cache的可能性。我们通过计算 INT4 GQA 的计算强度 (CI) 并将其与 BF16 GQA 的计算强度进行比较来估计潜在的性能改进，因为 CI 代表每字节的 FLOPS。我们计算了 $QK^T$ 和 $PV$（如等式 1 所示）的 CI，因为它们将 KV Cache作为操作数。请注意，我们忽略了 Q 加载，因为它与 KV Cache相比微不足道。我们还忽略了任何不在全局内存上的中间数据加载/存储。因此，CI 仅考虑计算 FLOPS 和 KV Cache加载。

![等式1](https://files.mdnice.com/user/59/704eefec-d9be-4531-af44-065af5cb5665.png)

假设 $H_Q = 8$ 且 $H_{KV} = 1$，BF16 KV Cache 的 CI 为 8，而 INT4 KV Cache 的 CI 为 32。CI 表明，BF16 和 INT4 GQA 都是内存受限的（A100 和 H100 的 BF16 Tensor Core的峰值 CI 为 312 TF / 2 TB/s = 141(https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/a100-80gb-datasheet-update-nvidia-us-1521051-r2-web.pdf) 和 990 TF / 3.35 TB/s = 269(https://www.nvidia.com/en-us/data-center/h100/)；请注意，这些 TF 数字不包括稀疏性）。此外，使用 INT4 KV Cache，我们应该期望与 BF16 GQA 相比性能提高 4 倍。

为了在 GQA 中启用 INT4 KV Cache 支持，我们可以在将 KV Cache 传递给 BF16 GQA 运算符之前，将其从 INT4 反量化为 BF16。但是，由于 KV Cache 通常很大，因此从/向全局内存复制它可能代价高昂。此外，Decoding GQA 是一个内存受限的操作（内存单元的使用比计算单元更频繁）。图 2 显示了 xFormers 中的 FMHA CUTLASS BF16 GQA kernel(https://github.com/facebookresearch/xformers/blob/9f6abadabdec17cd4b5c301632a44bf8216a7f35/xformers/csrc/attention/cuda/fmha/autogen/impl/cutlassF_bf16_aligned.cu#L33) 的 NCU 配置文件，它是 GQA 的最先进实现之一。从图中可以明显看出，内存是一个瓶颈。

![图 2 xFormers FMHA CUTLASS BF16 kernel 的 NCU profile结果 ](https://files.mdnice.com/user/59/aaa767f0-f8b1-44c4-8799-6cd33d342aff.png)

一种更有效的替代方案是将 INT4 反量化与 GQA 操作融合（如图 3 所示）。换句话说，让 GQA 直接读取 INT4 KV Cache，并在 kernel 中执行 INT4 到 BF16 的转换。这种变化可能会减少 KV Cache 所需的全局内存读取量，从而导致延迟降低。我们将其称为“INT4 GQA”。

![图 3 融合 INT4 GQA 的工作流程](https://files.mdnice.com/user/59/c34b72c6-22bb-4278-a458-afadaa31f923.png)

我们在下表中列出了 GQA 的最先进实现，以及它们在表 1 中的功能。

**表 1** 最先进的 GQA 实现

![](https://files.mdnice.com/user/59/85d114a5-8441-4898-a096-4c9f78e29fb1.png)

除 CU 外，所有实现都支持 split-K 和非 split-K。CU 只有 split-K 实现。只有 FA 在后端有一个启发式方法来确定是否运行 split-K 或非 split-K kernel。对于其他实现，用户必须显式选择要运行的版本。在本说明中，我们重点关注较长的上下文长度（在我们的实验中，我们使用 8192 的上下文长度），因此尽可能选择 split-K 版本。

作为基线，我们在 NVIDIA A100 和 H100 GPU 上测量了最先进的 GQA 实现的性能。表 2 中报告了延迟（以微秒为单位）和达到的带宽（GB/s）。请注意，我们运行了一系列 split-K（从 2 到 128 个splits），并报告了每个实现的最佳性能。对于所有实验，我们使用 8192 的上下文长度。对于 INT4 GQA，我们使用按行量化（即，量化组数量 = 1）。

![](https://files.mdnice.com/user/59/3c7e1cec-1d15-4b5e-9e59-f52d96709f7f.png)
![](https://files.mdnice.com/user/59/d0f434f0-ece0-4731-bad3-220203dd3b98.png)


首先，让我们讨论 BF16 GQA 性能：在所有实现中，CU 的性能排名最后。FD 和 FA 的性能相当。当批次大小小于或等于 64 时，FA 使用 split-K kernel，性能略好于 FD。但是，当批次大小大于 64 时，FD 的性能更好。

INT4 GQA 的趋势相同。但是，我们没有测量 FA 的性能，因为它不支持 INT4 KV Cache。在所有情况下，FD 的性能都优于 CU。

当比较 FD 在 BF16 和 INT4 GQA 之间的延迟时，我们发现它们几乎相同。这表明INT4 GQA 效率极低，这一点可以从与 BF16 GQA 相比，INT4 GQA 的可达带宽明显更低得到进一步证实。观察 CU 的性能时，也是如此。

### 使用 Tensor Core 的 CUDA INT4 GQA 实现

在本节中，我们简要介绍我们的基线实现，即使用 Tensor Core 的 CUDA INT4 GQA (CU)。每个线程块只处理一个 KV 头和来自一个输入提示的一组查询头。因此，每个线程块执行 $mm(softmax(mm(Q, K^T) / sqrt(D))$, V)；请注意，$mm$ 正在执行，而不是 $bmm$。此外，由于这是一个 split-K 实现，因此 KV Cache 中的 token 在不同的线程块之间拆分。请注意，每个线程块包含 4 个 warp（每个 warp 包含 32 个线程，适用于 NVIDIA A100 和 H100 GPU）。每个线程块中的工作在 warp 之间拆分。在每个 warp 中，我们使用 WMMA API 在 Tensor Core 上计算矩阵乘法。图 4 演示了 CU 中的工作分区。

![图四：工作分区](https://files.mdnice.com/user/59/30e632aa-b9b6-4488-9126-025ff0a49c04.jpg)











