## 这里总结 FasterTransformer Decoder(GPT) 的cuda相关优化技巧

> 解读：https://github.com/NVIDIA/FasterTransformer/blob/main/docs/gpt_guide.md

### FasterTransformer GPT

这篇文档讲了一下 FasterTransformer 为 GPT 模型提供了什么，解释了工作流程和优化。并且还提供了在 FastTransformer 上运行 GPT 模型的指南。最后还提供了 Benchmark 测试来说明 FastTransformer 在 GPT 模型上的性能。我这里只针对 FasterTransformer GPT 的工作流程和做的优化进行讲解。

下面的 FasterTransformer GPT 介绍 和 FasterTransformer GPT 结构是简单翻了下文档。

### FasterTransformer GPT 介绍（翻译）
GPT 是 Decooding 模型的一种变体，没有 Encoder 模块，没有交叉多头注意力模块，使用 GeLU 作为激活函数。2020 年，OpenAI 在[他们的论文](https://arxiv.org/abs/2005.14165)中表明，使用非常庞大的模型和大量的训练数据可以显著提高 GPT 模型的容量。 但是，不可能将这样的模型放入单个 GPU 中。 例如，最大的模型 GPT-3 有 1750 亿个参数，half 数据类型下大约需要 350 GB显存。 因此，多GPU，甚至多节点，是很有必要的。 为了解决模型大小导致的延迟和内存瓶颈，FasterTransformer 提供了高性能、低内存占用的 kernel，并使用了模型并行技术。

#### 支持的特性

* Checkpoint converter
  * Huggingface
  * Megatron
  * Nemo Megatron
  * TensorFlow
* Data type
  * FP32
  * FP16
  * BF16
  * INT8 weight only PTQ.
    * 限制:
      * 权重被切分后，隐藏层的维度必须是 64 的倍数。
      * cuda kernel通常只为小的 batch（如32和64）和权重矩阵很大时提供性能优势。
      * 权重的 PTQ 量化只支持 FP16/BF16。
      * 仅支持 Volta 和更新的 GPU 架构。
    * Note:
      * 根据当前 GPU 的情况，权重被提前离线预处理，以降低 TensorCore 做权重对齐的开销。目前，我们直接使用 FP32/BF16/FP16 权重并在推理前对其进行量化。如果我们想存储量化的权重，必须要在推理的 GPU 上来进行预处理。
      * 使用 torch API 时，int8 模式只能通过 Parallel GPT Op 使用。 Parallel GPT Op 也可以在单个 GPU 上使用。
  * INT8 with SmoothQuant
  * FP8 (**Experimental**)
* Feature
  * Multi-GPU multi-node inference
  * Dynamic random seed
  * Stop tokens
  * Beam search and sampling are both supported
  * Loading FP32 or FP16 weights
* Frameworks
  * TensorFlow
  * PyTorch
  * C++
  * Triton backend

### FasterTransformer GPT 结构（翻译）

#### 工作流

![Fig 1. Workflow of GPT model](https://user-images.githubusercontent.com/35585791/215260593-c5db412c-f67e-4167-83dc-51f44b284758.png)

Fig 1展示了 FasterTransformer GPT 的工作流程。 与 BERT 和编码器-解码器结构不同，GPT 接收一些输入 id 作为上下文，并生成相应的输出 id 作为响应。在这个工作流中，主要的瓶颈是 GptDecoderLayer （Transformer块），因为当我们增加层数的时候耗时也是线性增加的。在GPT-3中，GptDecoderLayer占用了大约95%的时间。

Faster Transformer把整个工作流分成了两个部分。第一个部分是：“根据上下文context(也就是输入ids)计算k/v cache”。第二个部分是：“自回归的生成输出ids”。这两部分的操作类似，但是selfAttention部分的输入tensors的形状是不一样的。所以Faster Transformer提供了2种计算方式，如Fig2所示。在`DecoderSelfAttention`里面，query的序列长度总是1，所以我们使用自定义的fused masked multi-head attention kernel 进行处理。另一方面，在`ContextSelfAttention`中，query的序列长度最大时输入的长度，所以我们使用cuBLAS来利用TensorCore。



