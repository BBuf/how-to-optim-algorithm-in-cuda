> 代码：https://github.com/pytorch/FBGEMM/blob/main/fbgemm_gpu/experimental/gen_ai/src/attention/gqa_attn_splitk.cu

> PyTorch 博客：https://pytorch.ac.cn/blog/int4-decoding/


# 用于 LLM 推理的 INT4 Decoding GQA CUDA 优化

## 介绍

生成式 AI 以其像人类一样生成内容的能力席卷全球。许多这些生成式 AI 工具都由大型语言模型 (LLM) 提供支持，例如 Meta 的 Llama 模型和 OpenAI 的 ChatGPT。LLM 的主要挑战之一是支持大型“上下文长度”（也称为“序列长度”）。上下文长度是指模型用来理解输入上下文和生成回复的token数量。更长的上下文长度通常会转化为回复中更高的精度和质量。但是，较长的上下文长度在计算和内存方面都需求很高。这主要是因为以下原因：

- Attention的计算复杂度随着上下文长度的增加而线性增长（增长率取决于注意力算法）。因此，在使用较长的上下文长度时，Attention层可能会成为瓶颈，尤其是在prefill阶段，此时 Attention 是compute bound的。
- KV Cache的大小与上下文长度线性增长，因此对内存需求造成更大的压力，从而减慢了本来就memory bound的 attention decoding。此外，由于内存容量有限，当 KV Cache变大时，批次大小会减小，这通常会导致吞吐量下降。




