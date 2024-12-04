> 我的课程笔记，欢迎关注：https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode 。

> 这节课介绍了两个LinkedIn开发的 Liger Kernel的核心优化：RMSNorm和Fused Linear Cross Entropy。本笔记详细记录了课程中的优化的数学原理、实现方法和测试验证过程，包括一些脚本的解读。RMSNorm部分展示了反向传播的推导过程和内存优化技巧；Fused Linear Cross Entropy部分展示了如何通过检查点(checkpointing)、分块(chunking)和前向梯度(gradient-in-forward)等技术来减少内存使用。此外，课程还分享了在Triton框架开发上面的优化kernel中的一些实践经验，比如连续性(Contiguity)问题和索引越界问题的处理。通过这些优化，Liger Kernel能够将多GPU训练吞吐量提升20%并减少60%的内存使用，是Triton在工业界很好的一次实践。

# 第28课，Liger Kernel

Liger Kernel(https://github.com/linkedin/Liger-Kernel) 是一个专门为 LLM 训练设计的 Triton kernels 集合，由LinkedIn的工程师开发和维护。它能有效地将多 GPU 训练吞吐量提高 20%，并将内存使用量减少 60%。目前已经实现了与 HuggingFace 兼容的 `RMSNorm`、`RoPE`、`SwiGLU`、`CrossEntropy`、`FusedLinearCrossEntropy` 等功能，未来还会有更多。Liger Kernel可以直接与 Flash Attention、PyTorch FSDP 和 Microsoft DeepSpeed 配合使用。我们欢迎社区贡献，共同收集最佳的 LLM 训练kernel。

## 课程笔记之 RMSNorm

![](https://files.mdnice.com/user/59/31039d3c-a155-429c-9318-41a0c07c0bee.png)

![](https://files.mdnice.com/user/59/8e9c1f09-e240-42d8-ae49-89ffae4ce468.png)

这张Slides介绍了本节课的大纲。具体包括了LLM（大语言模型）训练中的性能瓶颈问题，为什么选择使用Triton框架，并介绍如何实现RMS Norm和Fused Linear Cross Entropy，这可以减少很多内存，也会提到如何测试减少的内存。同时提供了三个重要的优化技巧：收敛性测试、连续性优化和地址范围处理。最后给Liger kernel打个广告。

![](https://files.mdnice.com/user/59/51cf829d-5e5f-454b-a6b2-dbcefda1ce99.png)

这里讲到大模型的瓶颈不仅包含显存导致的OOM，还包含效率问题，有一个误区就是GPU利用率越高速度越快，实际上这是错误的理解，GPU利用率高只是说明GPU很忙，推荐阅读 https://arthurchiao.art/blog/understanding-gpu-performance/ 这篇文章来理解这个问题。此外，Profiler是理解一切性能问题的基础，推荐cuda-mode的lecture1和lecture16来学习如何使用Profiler。

![](https://files.mdnice.com/user/59/20133ed9-f65d-4db0-a236-99d411d3c463.png)


然后作者对一个LLama模型做了一个在线的Profile，我们可以看到在内存变化阶段Cross entropy有一个峰值突变，它消耗了很多内存。如下图所示：

![](https://files.mdnice.com/user/59/f6219530-19a7-484c-a867-f50b9642225e.png)

由于使用了Checkpointing技术，在前向和反向阶段的每个Transformer Block上也能观察到内存的升降，因为计算下一个Transformer Block的时候会释放当前Transformer Block占用的内存。这里的重点是Cross Entropy的内存消耗，来源是具体化logits的过程中产生的峰值内存，因为vocab size很大。

接下来作者介绍了一下kernel trace部分，从这部分我们可以看到LLama模型有很多elmentwise ops和很多cuda kernel launch的overhead。作者也提到这个kernel trace由于是用FSDP训练LLamaT，所以我们可以在每个Transformer Block看到2次all gather和一次reduce scatter。FSDP具体原理图可以参考：https://zhuanlan.zhihu.com/p/485208899 ，然后可以参考[【翻译】使用PyTorch FSDP最大化训练吞吐量](https://mp.weixin.qq.com/s/6wNX38rKcFjxLb4ooYQokw) 。

简单总结一下，FSDP训练LLamaT可以观察到两个明显的瓶颈，第一个是Cross Entropy的大量内存消耗，第二个是elmentwise ops和很多cuda kernel launch的overhead。

![](https://files.mdnice.com/user/59/2a5dfb8f-ebfb-4909-a039-73eda0ec2cf0.png)

这张Slides介绍了为什么要选择使用Triton（一个GPU编程框架）的几个主要原因：它比CUDA更容易编程，开发内核速度更快；它采用类似Numpy的向量化思维方式而不是传统的线程思维；对AI研究人员更友好，便于他们理解和扩展；作为Python原生框架不需要处理多个文件类型；并且依赖关系简单，在大多数情况下都能正常工作。总的来说，这些优势使Triton成为一个更现代、更易用的GPU编程解决方案。

![](https://files.mdnice.com/user/59/562d0b95-9a88-4e54-a6b6-2f44dabefbce.png)

这张Slides说明使用Triton写RMS Forward 会很简单，但是写Backward会比较难，下面会展示一些作者总结的技巧。

![](https://files.mdnice.com/user/59/70c5aa15-9275-485d-b7bb-8ece47151b21.png)

这张slides介绍了Backward Pass(Backprop)的基础知识，以"Backprop 101"为标题，主要强调了在学习Backward Pass时应该**按元素思考**，因为标量微积分比向量微积分更容易推导，同时建议复习微积分基础知识和矩阵-矩阵乘法公式。slides最后给出了矩阵乘法 Y = XW 的Backward Pass推导结果，包括对X的梯度 `∂L/∂X = (∂L/∂Y)W^T` 和对W的梯度 `∂L/∂W = X^T(∂L/∂Y)`，这些基础知识对于理解和实现更复杂操作（如RMS Norm）的Backward Pass非常重要。

![](https://files.mdnice.com/user/59/cff21c52-a749-4b7e-a17f-8974e6ee4500.png)

这张Slides展示了RMSNorm(Root Mean Square Normalization)操作的Backward Pass(backprop)推导过程。主要包含两个关键公式：

1. Forward Pass公式：yi = (xi * wi) / sqrt((1/n) * ∑xk²)，表示对输入xi进行归一化
2. Backward Pass公式：dxi = ∂o/∂xi = ∑k (∂o/∂yk * ∂yk/∂xi)，使用链式法则计算梯度

这里特别强调了链式法则的应用原因：因为输入xi会影响所有的输出yi，所以在计算梯度时需要考虑xi对所有yi的影响并求和。这是RMSNormBackward Pass计算中的核心思想。

![](https://files.mdnice.com/user/59/e7b96e3e-b6c4-4a20-b272-c3bbc83d9f36.png)

这张Slides展示了RMSNormBackward Pass的详细数学推导，特别强调了需要分开处理k=i和k≠i两种情况。其中引入了RMS（Root Mean Square）变量来简化表达式，最终得到了当k=i时的偏导数公式。通过数学变换，将一个复杂的表达式简化为更简洁的形式：`(wi - 1/(RMS^2) * 1/n * xi^2 * wi)/RMS`。这个推导过程对于实现RMSNorm的Backward Pass计算非常重要，它为后续的代码实现提供了理论基础。

![](https://files.mdnice.com/user/59/6ceaf0d6-f1e7-4452-ac38-56083943f891.png)

这张Slides进一步展示了k≠i时的RMSBackward Pass的数学推导。

![](https://files.mdnice.com/user/59/322f0204-9c24-4a5e-916b-13e4bba6b339.png)

这张Slides把k=i和k≠i的两种情况合并起来获得了RMSBackward Pass的完整数学推导。并且我们可以从单个元素的推导推广到向量，这样就可以在Triton中方便的实现了。

![](https://files.mdnice.com/user/59/8c1cb619-c22b-4774-98ef-7cdbf8784066.png)

这张Slides展示了Liger-Kernel实现RMSNorm的时候使用的2个技巧，Inplac Tensor reuse和Cache rms，可以从 https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/rms_norm.py 源码中看到。

![](https://files.mdnice.com/user/59/8415e344-3a5c-40b6-a3fd-f84d78207601.png)

作者提供了一个jupyter的demo，用来展示对Liger-Kernel的测试过程，包含正确性，性能，内存等方面的测试。 https://colab.research.google.com/drive/1CQYhul7MVG5F0gmqTBbx1O1HgolPgF0M?usp=sharing ，我们接下来解读一下这个测试。


## Live Demo: RMSNorm: 确认正确性和性能的测试解读

现在你已经学会了如何推导RMSNorm的Backward Pass以及内存节省技巧，实现本身相对来说比较直接。因此，我们将重点关注测试，并使用来自Liger Kernel的现有实现。

### 为什么需要测试？

假设我们已经有了一个可运行的RMSNorm版本，在将其部署到生产环境之前，我们需要验证以下几点：

1. 正确性：确保kernel的精度与原始实现一致。任何偏差都可能影响模型收敛或导致严重错误。
2. 性能：确认kernel在时间和内存使用上都比原始版本更高效。如果没有这些改进，用Triton重新实现就失去了意义。

### 正确性测试

准备一个纯PyTorch实现，比如使用HuggingFace提供的版本。

我们需要用不同的输入形状和数据类型来测试实现。除了像2的幂这样的规则形状外，测试不规则形状也很重要，以确保能正确处理边界情况。

设置容差可能比较棘手。通常对于`fp32`，使用`atol = 1e-7`和`rtol = 1e-5`。对于`bf16`，使用`atol = 1e-3`和`rtol = 1e-2`。但在实践中，即使kernel是精确的，有时也可能需要进一步放宽容差。

稍后，我们将讨论另一种测试方法来验证kernel不会对端到端的收敛产生负面影响。

```python
import torch
import torch.nn as nn


# Copy from HuggingFace

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm是等价于T5LayerNorm的实现
        参数:
            hidden_size: 隐藏层维度大小
            eps: 用于数值稳定性的小常数
        """
        super().__init__()
        # 初始化可学习的缩放参数
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # 保存epsilon值用于避免除零
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # 保存输入数据类型
        input_dtype = hidden_states.dtype
        # 转换为float32以提高精度
        hidden_states = hidden_states.to(torch.float32)
        # 计算方差
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # 标准化操作
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # 应用可学习参数并恢复原始数据类型
        return self.weight * hidden_states.to(input_dtype)
```

```python
import torch
from liger_kernel.transformers.rms_norm import LigerRMSNorm


input_data = [
    (4, 16, 32, torch.float32, 1e-6, 1e-4),
    (8, 32, 64, torch.float32, 1e-6, 1e-4),
    (16, 64, 128, torch.float32, 1e-6, 1e-4),
    (3, 9, 13, torch.float32, 1e-6, 1e-4),
    # T4 GPU doesn't support bf16 :(
    # (16, 64, 128, torch.bfloat32, 1e-3, 1e-2),
]

for bs, sl, hd, dtype, atol, rtol in input_data:
    # h
    _tensor = torch.randn(bs, sl, hd, device="cuda", dtype=dtype)

    h1 = _tensor.clone().requires_grad_(True)
    h2 = _tensor.clone().requires_grad_(True)

    # do
    do = torch.randn(bs, sl, hd, device="cuda", dtype=dtype)

    # llama
    llama_rms = LlamaRMSNorm(hidden_size=hd).to("cuda").to(dtype)
    llama_o = llama_rms(h1)
    llama_o.backward(do.clone(), retain_graph=True)

    # triton
    triton_rms = LigerRMSNorm(hidden_size=hd).to("cuda").to(dtype)
    triton_o = triton_rms(h2)
    triton_o.backward(do.clone(), retain_graph=True)

    assert torch.allclose(llama_o, triton_o, atol=atol, rtol=rtol) is True

    # print(h1.grad, h2.grad)
    assert torch.allclose(h1.grad, h2.grad, atol=atol, rtol=rtol) is True
```

### 性能测试

我们需要测试两个维度:速度和内存。但是应该使用什么输入形状来测试呢?你可以使用训练时的实际输入形状来测试。例如,在微调LLaMA 3-8B模型时,我们通常使用batch size为4,hidden size为2048。我们将序列长度作为变量。

这样,测试结果就能反映出我们在生产环境训练中可以预期的实际收益。这里使用了Triton提供的自动测试工具。

```shell
import os

import torch
import torch.nn as nn
import triton


@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            x_names=["seq_len"],
            x_vals=[2**i for i in range(8, 11)], # 256, 512, 1024
            xlabel="seq len",
            line_arg="provider",
            line_vals=["liger", "huggingface"],
            line_names=["Liger", "Hugging Face"],
            styles=[("blue", "solid"), ("orange", "solid")],
            ylabel="time (ms)",
            plot_name="rmsnorm-full-speed-benchmark",
            args={"batch_size": 4, "hidden_size": 2048, "dtype": torch.float32, "mode": "full"},
        ),
    ]
)
def bench_speed_rms_norm(batch_size, seq_len, hidden_size, dtype, provider, mode, eps=1e-5, device="cuda"):
    x_shape = (batch_size, seq_len, hidden_size)

    triton_rms = LigerRMSNorm(hidden_size=hidden_size).to("cuda")
    llama_rms = LlamaRMSNorm(hidden_size=hidden_size).to("cuda")

    x = torch.randn(x_shape, dtype=dtype, device="cuda")
    dy = torch.randn_like(x)
    x.requires_grad_(True)

    x = x.view(batch_size * seq_len, hidden_size)
    dy = dy.view(batch_size * seq_len, hidden_size)

    quantiles = [0.5, 0.2, 0.8]


    def full():
        if provider == "liger":
            y = triton_rms(x)
        elif provider == "huggingface":
            y = llama_rms(x)

        y.backward(dy, retain_graph=True)

    ms, min_ms, max_ms = triton.testing.do_bench(
        full, quantiles=quantiles, grad_to_none=[x], rep=500
    )

    return ms, max_ms, min_ms


bench_speed_rms_norm.run(show_plots=True, print_data=True)
```

![](https://files.mdnice.com/user/59/8809d391-7fa2-45c4-9b1c-bdc4b1963198.png)

```python
def test_memory(func, _iter):
    total_mem = []

    for _ in range(_iter):
        torch.cuda.memory.reset_peak_memory_stats()
        func()
        mem = torch.cuda.max_memory_allocated() / (2**20)
        total_mem.append(mem)

    return sum(total_mem) / len(total_mem)

@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            x_names=["seq_len"],
            x_vals=[2**i for i in range(8, 11)], # 256, 512, 1024
            xlabel="seq len",
            line_arg="provider",
            line_vals=["liger", "huggingface"],
            line_names=["Liger", "Hugging Face"],
            styles=[("blue", "solid"), ("orange", "solid")],
            ylabel="Memory (MB)",
            plot_name="rmsnorm-full-memory-benchmark",
            args={"batch_size": 4, "hidden_size": 2048, "dtype": torch.float32, "mode": "full"},
        ),
    ]
)
def bench_memory_rms_norm(batch_size, seq_len, hidden_size, dtype, provider, mode, eps=1e-5, device="cuda"):
    x_shape = (batch_size, seq_len, hidden_size)

    triton_rms = LigerRMSNorm(hidden_size=hidden_size).to("cuda")
    llama_rms = LlamaRMSNorm(hidden_size=hidden_size).to("cuda")

    x = torch.randn(x_shape, dtype=dtype, device="cuda")
    dy = torch.randn_like(x)
    x.requires_grad_(True)

    x = x.view(batch_size * seq_len, hidden_size)
    dy = dy.view(batch_size * seq_len, hidden_size)

    quantiles = [0.5, 0.2, 0.8]


    def full():
        if provider == "liger":
            y = triton_rms(x)
        elif provider == "huggingface":
            y = llama_rms(x)

        y.backward(dy, retain_graph=True)

    mem = test_memory(full, 10)

    return mem

bench_memory_rms_norm.run(show_plots=True, print_data=True)
```

![](https://files.mdnice.com/user/59/4947e25e-b3c6-4f52-b11c-266fdee2a9b5.png)

### RMSNorm测试总结

我们可以清楚地看到,Triton实现在速度和内存使用上都优于原始实现,并且我们也验证了其正确性!由于Google Colab的GPU限制,我们只进行了部分测试。在Liger-Kernel的实际测试中,我们还使用更大的输入尺寸验证了bf16的性能。完整版本请参考 https://github.com/linkedin/Liger-Kernel 。

## 课程笔记之Fused Linear Cross Entropy

![](https://files.mdnice.com/user/59/81f3ed55-6c6b-4c26-b991-028ca3c1bf46.png)

这张Slides展示了Transformer模型里面 Linear Cross Entropy 的Forward Pass和Backward Pass过程。图中左侧显示了Forward Pass(Forward)流程，从输入(input)通过 lm_head 层产生激活值(Activations)，然后输出(output)与目标值(target)计算交叉熵(Cross Entropy)。右侧显示了Backward Pass(Backward)流程，展示了梯度(Gradients)的传递方向。图片底部指出了一个问题：大词汇表尺寸(Large Vocab Size)是这个模型面临的主要挑战。

![](https://files.mdnice.com/user/59/4f9e009e-be08-41aa-b596-89a91df924a2.png)

![](https://files.mdnice.com/user/59/e62811c6-96f1-458e-aa72-7e6a84f9fa90.png)

![](https://files.mdnice.com/user/59/d6f97bda-2fa4-4ed0-b94a-20cd00c80dcd.png)

![](https://files.mdnice.com/user/59/b4963af6-8fd5-46e7-8410-01949644fbf5.png)

![](https://files.mdnice.com/user/59/c19faf75-1789-4e79-849f-f1419ba0e969.png)

这5张slides展示了完整的线性层和交叉熵梯度计算的推导过程。首先介绍了线性层的Forward Pass(`y = Wx`)和Backward Pass`(∂o/∂x = W^T∂y)`，接着对交叉熵损失函数`l = -∑yⱼlog(exp(xⱼ)/∑exp(xᵢ))`求偏导，将其分解为两项分别推导：一项是针对包含xₖ的项，另一项是针对其他所有项。经过复杂的代数运算和化简，最终得到了简洁的梯度表达式`∂l/∂xₖ = -yₖ + softmax(xₖ)`，并讨论了`yₖ=1`和`yₖ=0`两种特殊情况下的结果，将这个复杂的梯度计算问题优化为目标值与softmax的差值形式。有了最后的这个等式，我们就可以比较方便的在Triton中计算向量的交叉熵梯度了。

![](https://files.mdnice.com/user/59/b720a960-3f9b-4329-b140-75269a9d195c.png)

这里需要注意下第二点，由于Cross Entropy是最后一层，它的输出一定是一个标量，所以我们可以在forward的时候就计算梯度。

![](https://files.mdnice.com/user/59/993dd72b-cf1f-4e38-be2a-756305960ade.png)

这张Slides展示了Fused Linear Cross Entropy中的梯度检查点(Gradient Checkpointing)技术。在模型训练过程中，左侧展示了正向传播路径：从input经过lm_head层得到output，然后与target计算交叉熵；右侧展示了Backward Pass路径。关键点在于，在Backward Pass时会重新计算前向过程(Forward Recomputation)，而不是保存激活值，这样可以节省存储空间。图中用"×"表示激活值(Activations)，用"△"表示梯度(Gradients)。底部说明文字强调了这个策略的核心：在Backward Pass时重新计算前向过程，避免了需要持久化存储激活值。

![](https://files.mdnice.com/user/59/2bc5f6a1-0270-4ed6-98a8-7999f9e63b57.png)

这张Slides展示了Fused Linear Cross Entropy中的gradient-in-forward优化技术。与之前的梯度检查点方法不同，这里在Forward Pass过程中就同时计算梯度，避免了需要重新计算前向过程。图中显示了从input通过lm_head层到output，再与target计算交叉熵的流程，其中lm_head层同时包含了激活值(用"×"表示)和梯度(用"△"表示)。底部说明文字强调了这种方法的优势：通过在Forward Pass时就计算梯度，可以消除重新计算前向过程的需求，从而提高计算效率。（可以在前向过程中计算梯度的原因是因为Cross Entropy的输出是一个Scalar，所以上游的梯度稳定为1。这样就避免了在Backward Pass计算梯度和重计算了）

![](https://files.mdnice.com/user/59/44840c81-68b0-4d9c-bf68-5271fda0accc.png)

这张Slides展示了Fused Linear Cross Entropy中的Chunking（分块）技术。图中显示input被虚线分成多个块（chunks），表明输入数据被分成若干小块进行处理。这种方法的核心思想是：每次只处理输入数据的一个块，因此在任意时刻只需要在内存中保存当前块的激活值（用"×"表示）和梯度（用"△"表示）。底部说明文字解释了这种策略的优势：通过逐块处理输入数据，可以显著减少内存使用，因为同一时刻只需要保存一小块数据的激活值和梯度信息，而不是全部数据。需要注意的是，对输入进行分chunk处理由于Cross Entropy的梯度计算存在Softmax操作，具体见上面几张Slides，当我们对输入进行分块之后我们需要像Online Softmax算法那样逐chunk更新缩放系数，最后才能对 hidden_states 得到正确的梯度。


## Live Demo: FusedLinearCrossEntropy: 确认内存减少

我们已经讨论了检查点(checkpointing)、分块(chunking)和前向梯度(gradient-in-forward)这些概念,它们可以避免具体化logits以及重新计算。由于实现比较复杂,所以这部分就留给大家自学。在这个notebook中,我们将验证FusedLinearCrossEntropy是否真的可以在保持相当速度的同时减少峰值内存使用。

### FusedLinearCrossEntropy基准测试

比较FusedLinearCrossEntropy和Hugging Face实现在速度和内存方面的表现

```python
import os

import torch
import triton

from liger_kernel.transformers.fused_linear_cross_entropy import (
    LigerFusedLinearCrossEntropyLoss,
)


class TorchLMHeadCE(torch.nn.Module):
    """Ground truth implementation of the linear fused with torch based cross entropy loss.

    :param H: hidden size
    :param V: vocab size
    :param ignore_index: index to ignore
    :param reduction: reduction method
    """

    def __init__(self, H: int, V: int, dtype: torch.dtype, ignore_index: int = -100):
        super().__init__()
        # 定义线性层,不带偏置
        self.lin = torch.nn.Linear(
            in_features=H, out_features=V, bias=False, dtype=dtype
        )
        # 定义交叉熵损失函数
        self.ce_loss = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction="mean"
        )

    def forward(self, x, y):
        # 前向传播:先通过线性层,再计算交叉熵损失
        logits = self.lin(x)
        return self.ce_loss(logits, y)


class LigerLMHeadCE(torch.nn.Module):
    def __init__(self, H: int, V: int, dtype: torch.dtype, ignore_index: int = -100):
        super().__init__()
        # 定义线性层,不带偏置
        self.lin = torch.nn.Linear(
            in_features=H, out_features=V, bias=False, dtype=dtype
        )
        # 定义Liger的融合交叉熵损失函数
        self.ce_loss = LigerFusedLinearCrossEntropyLoss(
            ignore_index=ignore_index, reduction="mean"
        )

    def forward(self, x, y):
        # 前向传播:直接使用权重矩阵、输入和目标计算融合的交叉熵损失
        return self.ce_loss(self.lin.weight, x, y)


def test_memory(func, _iter):
    # 用于测试内存使用的辅助函数
    total_mem = []

    for _ in range(_iter):
        # 重置CUDA峰值内存统计
        torch.cuda.memory.reset_peak_memory_stats()
        func()
        # 获取最大分配内存(MB)
        mem = torch.cuda.max_memory_allocated() / (2**20)
        total_mem.append(mem)

    # 返回平均内存使用量
    return sum(total_mem) / len(total_mem)

@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            x_names=["BT"],
            x_vals=[2**i for i in range(10, 13)], # 1024, 2048, 4096
            xlabel="B x T",
            line_arg="provider",
            line_vals=["liger", "huggingface"],
            line_names=["Liger", "Hugging Face"],
            styles=[
                ("blue", "solid"),
                ("orange", "solid"),
            ],
            ylabel="GPU memory usage (MB)",
            plot_name="fused-linear-cross-entropy-memory-benchmark",
            args={"H": 4096, "V": 128256, "dtype": torch.float32},
        )
    ]
)
def bench_memory_cross_entropy(BT, H, V, provider, dtype, device="cuda"):
    # 打印基准测试参数
    print(
        f"Running benchmark with BT={BT}, H={H}, V={V}, dtype={dtype} provider={provider}"
    )
    # 初始化PyTorch和Liger模型
    torch_lm_head_ce = TorchLMHeadCE(H=H, V=V, dtype=dtype).to(device)
    liger_lm_head_ce = LigerLMHeadCE(H=H, V=V, dtype=dtype).to(device)

    # 生成随机输入和目标
    _input = torch.randn(BT, H, requires_grad=True, dtype=dtype, device=device)
    target = torch.randint(V, (BT, 1), dtype=torch.long, device=device).squeeze(1)

    def fwd():
        # 根据provider选择不同的前向实现
        if provider == "liger":
            return liger_lm_head_ce(_input, target)
        elif provider == "huggingface":
            return torch_lm_head_ce(_input, target)

    def full():
        # 完整的前向+反向传播
        y = fwd()
        y.backward()

    # 测试内存使用并返回结果
    mem = test_memory(full, _iter=10)
    return mem


bench_memory_cross_entropy.run(show_plots=True, print_data=True)
```

![](https://files.mdnice.com/user/59/0dd4506d-feba-436f-8272-40858ed59a98.png)

```python
@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            x_names=["BT"],
            x_vals=[2**i for i in range(10, 13)], # 1024, 2048, 4096
            xlabel="B x T",
            line_arg="provider",
            line_vals=["liger", "huggingface"],
            line_names=["Liger", "Hugging Face"],
            styles=[
                ("blue", "solid"),
                ("orange", "solid"),
            ],
            ylabel="Time (ms)",
            plot_name="fused-linear-cross-entropy-speed-benchmark",
            args={"H": 4096, "V": 128256, "dtype": torch.float32},
        )
    ]
)
def bench_speed_cross_entropy(BT, H, V, provider, dtype, device="cuda"):
    print(
        f"Running benchmark with BT={BT}, H={H}, V={V}, dtype={dtype} provider={provider}"
    )
    torch_lm_head_ce = TorchLMHeadCE(H=H, V=V, dtype=dtype).to(device)
    liger_lm_head_ce = LigerLMHeadCE(H=H, V=V, dtype=dtype).to(device)

    _input = torch.randn(BT, H, requires_grad=True, dtype=dtype, device=device)
    target = torch.randint(V, (BT, 1), dtype=torch.long, device=device).squeeze(1)

    def fwd():
        if provider == "liger":
            return liger_lm_head_ce(_input, target)
        elif provider == "huggingface":
            return torch_lm_head_ce(_input, target)

    def full():
        y = fwd()
        y.backward()

    quantiles = [0.5, 0.2, 0.8]

    ms, min_ms, max_ms = triton.testing.do_bench(full, quantiles=quantiles, rep=100)
    return ms, min_ms, max_ms


bench_speed_cross_entropy.run(show_plots=True, print_data=True)
```

![](https://files.mdnice.com/user/59/4235bf5b-f01c-41c9-bbf0-bf23ba524c5f.png)

### FusedLinearCrossEntropy测试总结

1. 我们可以观察到我们的实现在内存使用上有显著优势，因为我们在任何时候都不会具体化完整的logits。
2. 速度略微变慢，但由于lm_head + cross_entropy只执行一次，而transformer block要执行N次，这个开销是可以接受的。这使我们能够增加batch size、序列长度，或者关闭梯度检查点。


## Triton 操作的是physical view Contiguous()非常重要

![](https://files.mdnice.com/user/59/a2f77c69-382b-4c1a-9c4e-95d99ce29af7.png)

这张Slides主要介绍了收敛性测试中逐层比较的重要性，指出仅进行单元正确性和性能测试对于生产环境是不够的，因为实际生产中可能会遇到连续性、张量形状和数据类型的差异。因此建议通过模拟真实的生产训练环境来验证模型输出(logits)、权重(weights)和损失值(loss)，Slides底部还提供了一个Google Colab链接用于比较Triton内核补丁版本与原始模型的逐层对比，这些内容强调了在模型部署到生产环境之前进行全面测试的重要性(https://colab.research.google.com/drive/1e52FH0BcE739GZaVp-3_Dv7mc4jF1aif?usp=sharing)。这个脚本比较简单，我们也可以让ChatGPT写，这里就不单独看了。

![](https://files.mdnice.com/user/59/bad2844d-067d-479d-9457-30039475fc7f.png)

这张Slides主要讲解了在CUDA编程中"连续性(Contiguity)"这个容易被忽视但非常重要的问题。它指出连续性问题可能会导致难以调试的静默bug，需要花费大量时间来解决。幻灯片通过张量(Tensor)的例子，展示了逻辑视图(logical view)和物理视图(physical view)之间的区别，并用一个具体的带有stride的张量表示方式来说明这个概念。图中展示了一个2x2的张量在逻辑上和物理存储上的不同表现形式，以及相应的大小(sizes)和步长(strides)参数。对于Triton来说，它是在物理视图(physical view)上进行计算的，所以需要特别注意contiguous()的问题。https://colab.research.google.com/drive/1llnAdo0hc9FpxYRRnjih0l066NCp7Ylu?usp=sharing#scrollTo=1jTVlU1NC-TN 这个juypter展示了Liger Kernel的RoPE实现，因为忽略了对输入进行contiguous()操作导致单独的单元测试总是可以通过的，但是进行模型训练时始终出现loss发散的问题，这个问题也是通过上面的收敛性测试中逐层比较发现的。


## Triton 中的index越界bug

![](https://files.mdnice.com/user/59/268d0ccc-eeb8-4f30-9404-22b54f1c27ca.png)

这张Slides讲的是Triton的program_id是int32来表示的，然后在开发Cross Entropy时没有考虑到这一点，导致在较大的Vocab Size时index会越界。https://colab.research.google.com/drive/1WgaU_cmaxVzx8PcdKB5P9yHB6_WyGd4T?usp=sharing#scrollTo=X_Dn9wzVNpMC 这个juypter展示了这个问题，修复的方案是把program_id转换为int64。不过，因为32位寻址可能会导致性能很慢，所以需要非常谨慎的处理这个问题。例如在PyTorch中，针对这两种不同的数据类型会通过C++模板来处理，它们的实现会共享一个kernel，但是可以避免这个index溢出的问题。


## Liger Kernel 相关开源信息和反响

![](https://files.mdnice.com/user/59/34761a6b-648e-4dfb-99f5-c19b6516e675.png)

![](https://files.mdnice.com/user/59/961296da-fffd-4983-8a9a-d1705eda6913.png)

![](https://files.mdnice.com/user/59/546c6986-4b9d-4560-8b61-b8115c1f4555.png)

这几张Slides总结一下，LinkedIn开发的Liger Kernel是一个专为LLM训练优化的GPU高效运行时kernel，它能够将多GPU训练吞吐量提升20%并减少60%的内存使用，支持多种Hugging Face兼容功能，可与Flash Attention、PyTorch等主流框架协同工作。该项目在开源社区获得了积极反响，开发者反馈显示其性能表现优异。项目的成功离不开众多贡献者的支持，包括LOGO设计、训练灵感来源、测试数据集提供等方面的帮助，以及来自CUDA/Triton社区的大力支持，充分体现了开源协作的力量。

## 课程笔记总结

这节课介绍了两个核心优化：RMSNorm和Fused Linear Cross Entropy。本笔记详细记录了课程中的优化的数学原理、实现方法和测试验证过程，包括一些脚本的解读。RMSNorm部分展示了反向传播的推导过程和内存优化技巧；Fused Linear Cross Entropy部分展示了如何通过检查点(checkpointing)、分块(chunking)和前向梯度(gradient-in-forward)等技术来减少内存使用。此外，课程还分享了在Triton框架开发上面的优化kernel中的一些实践经验，比如连续性(Contiguity)问题和索引越界问题的处理。通过这些优化，Liger Kernel能够将多GPU训练吞吐量提升20%并减少60%的内存使用，是Triton在工业界很好的一次实践。








