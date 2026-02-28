> 来源：https://pytorch.org/blog/accelerating-generative-ai/ 。这篇文档介绍了PyTorch团队如何通过torch.compile、GPU量化、SDPA、半结构化稀疏性、嵌套张量和Triton自定义算子等原生PyTorch特性，将Meta的Segment Anything模型性能提升了8倍，在没有精度损失的情况下实现了显著的推理加速和内存优化。

本文是专注于如何使用纯原生 PyTorch 加速生成式 AI 模型的多系列博客的第一部分。我们很兴奋地分享众多新发布的 PyTorch 性能特性，以及这些特性如何结合使用的实际示例，看看我们能将 PyTorch 原生性能推到多远。

正如在 PyTorch 开发者大会 2023 上宣布的那样，PyTorch 团队重写了 Meta 的 Segment Anything ("SAM") 模型(https://github.com/facebookresearch/segment-anything)，结果比原始实现快了 8 倍，且没有精度损失，全部使用原生 PyTorch 优化。我们利用了众多新的 PyTorch 特性：

- Torch.compile：PyTorch 模型的编译器
- GPU 量化(https://github.com/pytorch/ao/tree/main#torchao)：使用降低精度操作加速模型
- 缩放点积注意力 (SDPA)：内存高效的注意力实现
- 半结构化 (2:4) 稀疏性(https://pytorch.org/tutorials/prototype/semi_structured_sparse.html)：GPU 优化的稀疏内存格式
- 嵌套张量(https://pytorch.org/tutorials/prototype/nestedtensor.html)：将不同尺寸的非均匀数据批量组合成单个张量，如不同尺寸的图像。
- **使用 Triton 的自定义算子**：使用 Triton Python DSL 编写 GPU 操作，并通过自定义算子注册轻松集成到 PyTorch 的各种组件中。

我们鼓励读者从我们在 Github 上的 SAM 实现(https://github.com/pytorch-labs/segment-anything-fast)中复制粘贴代码，并在 Github 上向我们提问。

![快速一瞥使用我们新发布的 PyTorch 原生特性提高吞吐量和减少内存开销。基准测试在 p4d.24xlarge 实例 (8x A100s) 上运行。](https://files.mdnice.com/user/59/41d3ccfe-07eb-4b49-8d6d-18cdc8dd4699.png)

# SegmentAnything 模型

SAM 是一个用于生成可提示图像掩码的零样本视觉模型。

![](https://files.mdnice.com/user/59/b9b480e2-961c-4a40-9452-44b67d5b9a6d.jpg)

SAM 架构[在其论文中描述(https://arxiv.org/abs/2304.02643)]包括基于 Transformer 架构的多个提示和图像编码器。在这其中，我们测量了最小和最大视觉变换器骨干网络的性能：ViT-B 和 ViT-H。为了简单起见，我们只展示 ViT-B 模型的追踪。

# 优化

下面我们讲述优化 SAM 的故事：性能分析、识别瓶颈，以及构建解决这些问题的新 PyTorch 特性。在整个过程中，我们展示了我们的新 PyTorch 特性：torch.compile、SDPA、Triton kernel 、嵌套张量和半结构化稀疏性。以下部分是逐步构建的，最终形成我们的 SAM-fast，现在可在 Github 上获得(https://github.com/pytorch-labs/segment-anything-fast)。我们使用真实的kernel 和内存追踪来激发每个特性，使用完全的 PyTorch 原生工具，并使用 Perfetto UI(https://perfetto.dev/#viewer)可视化这些追踪。

## 基线

我们的 SAM 基线是 Facebook Research 的未修改模型，使用 float32 数据类型和批量大小为 1。经过一些初始预热后，我们可以使用 PyTorch Profiler 查看kernel 追踪：

![](https://files.mdnice.com/user/59/f83a0658-0355-42fc-bd06-e0ba48658919.png)

我们注意到两个成熟的优化领域。

第一个是对 aten::index 的长时间调用，这是张量索引操作（例如，[]）的底层调用。虽然在 aten::index 上花费的实际 GPU 时间相对较低。aten::index 正在启动两个kernel ，并且在两者之间发生了阻塞的 cudaStreamSynchronize。这意味着 CPU 正在等待 GPU 完成处理，直到它启动第二个kernel 。为了优化 SAM，我们应该致力于移除导致空闲时间的阻塞 GPU 同步。

第二个是在矩阵乘法上花费了大量的 GPU 时间（上面流 7 上的深绿色）。这在 Transformers 中很常见。如果我们能减少在矩阵乘法上花费的 GPU 时间，就能显著加速 SAM。

我们可以从开箱即用的 SAM 测量吞吐量（img/s）和内存开销（GiB）来建立基线：

![](https://files.mdnice.com/user/59/965b7f81-e87a-4e0a-951b-334d36a75091.png)

## Bfloat16 半精度（+GPU 同步和批处理）

为了解决在矩阵乘法上花费较少时间的第一个问题，我们可以转向 bfloat16。Bfloat16 是一种常用的半精度类型。通过降低每个参数和激活的精度，我们可以在计算中节省大量时间和内存。在降低参数精度时，验证端到端模型精度至关重要。

![](https://files.mdnice.com/user/59/496022b4-f412-46fb-bee5-51af1eab027c.png)

这里显示了用半精度 bfloat16 替换填充数据类型的示例。代码在这里(https://github.com/pytorch-labs/segment-anything-fast/blame/main/segment_anything_fast/modeling/prompt_encoder.py#L86)。

除了简单地设置 `model.to(torch.bfloat16)`，我们还必须更改一些假设默认数据类型的小地方。

现在，为了移除 GPU 同步，我们需要审计导致它们的操作。我们可以通过在 GPU 追踪中搜索对 `cudaStreamSynchronize` 的调用来找到这些代码片段。事实上，我们找到了两个能够重写为无同步的位置。

![](https://files.mdnice.com/user/59/7a9acc28-8c65-46ec-8b59-0a953361791c.jpg)

具体来说，我们看到在 SAM 的图像编码器中，有一些变量充当坐标缩放器，q_coords 和 k_coords。这些都在 CPU 上分配和处理。然而，一旦这些变量用于在 rel_pos_resized 中索引，索引操作就会自动将这些变量移动到 GPU。这种复制导致了我们上面观察到的 GPU 同步。我们注意到在 SAM 的提示编码器中对索引的第二次调用：我们可以使用 torch.where 重写，如上所示。

### kernel 追踪

应用这些更改后，我们开始看到各个kernel 调用之间有显著的时间间隔。这通常在小批量大小（这里是 1）下观察到，这是由于启动kernel 的 GPU 开销。为了更仔细地查看实际的优化领域，我们可以开始使用批量大小 8 来分析 SAM 推理：

![](https://files.mdnice.com/user/59/47e36556-1212-4cf4-9d54-77f8ae565ce0.png)

查看每个kernel 花费的时间，我们观察到 SAM 的大部分 GPU 时间花费在逐元素kernel 和 softmax 操作上。因此，我们现在看到矩阵乘法已经成为一个相对较小的开销。

![](https://files.mdnice.com/user/59/003c375a-fce8-4f50-bb48-fed65d4d5c82.png)

综合 GPU 同步和 bfloat16 优化，我们现在已经将 SAM 性能提升了高达 3 倍

![](https://files.mdnice.com/user/59/1e1353b9-4487-42c0-96d2-68dc0cca27e0.png)

## Torch.compile（+图中断和 CUDA 图）

当观察到大量小操作时，比如上面分析的逐元素kernel ，转向编译器来融合操作可以带来强大的好处。PyTorch 最近发布的 torch.compile 在以下方面做得很好：

- 将操作序列（如 nn.LayerNorm 或 nn.GELU）融合到一个被调用的 GPU kernel 中
- epilogues：融合紧跟矩阵乘法kernel 的操作，以减少 GPU kernel 调用的数量。

通过这些优化，我们减少了 GPU 全局内存往返次数，从而加速了推理。我们现在可以在 SAM 的图像编码器上尝试 torch.compile(https://github.com/pytorch-labs/segment-anything-fast/blob/3bd74614fe7285de4de3d763d8ec2e951c4c589c/experiments/eval_combo.py#L196-L201)。为了最大化性能，我们使用一些高级编译技术，例如：

- 使用 torch.compile 的 max-autotune 模式启用 CUDA 图和具有自定义epilogues的形状特定kernel 
- 通过设置 TORCH_LOGS="graph_breaks,recompiles"，我们可以手动验证我们没有遇到图中断(https://pytorch.org/docs/main/torch.compiler_faq.html#graph-breaks)或重新编译。
- 用零填充输入到编码器的图像批次确保编译接受静态形状，从而能够始终使用具有自定义epilogues的形状特定优化kernel ，而不需要重新编译。

```python
predictor.model.image_encoder = \
    torch.compile(predictor.model.image_encoder, mode=use_compile)
```

### kernel 追踪

![](https://files.mdnice.com/user/59/b2423571-bd3b-4b50-b0f9-e03d77a64ef7.jpg)

torch.compile 工作得很好。我们启动一个单一的 CUDA 图，它在计时区域内占据了 GPU 时间的很大一部分。让我们再次运行性能分析并查看特定kernel 花费的 GPU 时间百分比：

![](https://files.mdnice.com/user/59/05f590ab-6c31-4e87-89a9-45fbd2799b9f.jpg)

我们现在看到 softmax 占据了大部分时间，其次是各种 GEMM 变体。总结我们观察到批量大小 8 和上述更改的以下测量结果。

![](https://files.mdnice.com/user/59/4bbc6e30-613a-4c76-a34d-09cdabb214c7.png)

## SDPA：scaled_dot_product_attention

接下来，我们可以解决变换器性能开销最常见的领域之一：注意力机制。朴素的注意力实现在时间和内存方面随序列长度呈二次缩放。PyTorch 的 scaled_dot_product_attention 操作建立在 Flash Attention、FlashAttentionV2 和 xFormer 的内存高效注意力的原理之上，可以显著加速 GPU 注意力。结合 torch.compile，这个操作允许我们表达并融合 MultiheadAttention 变体中的常见模式。经过一小组更改(https://github.com/facebookresearch/segment-anything/compare/50cb459d080bcd783a4b481d3bde4150d35ac497...7dc75fdf283693f73606f2fe7fdcb693afcb16b9)，我们可以调整模型以使用 scaled_dot_product_attention。

![](https://files.mdnice.com/user/59/8bec9731-21e6-474a-b8e9-4c1405a880b9.png)

PyTorch 原生注意力实现，查看代码(https://github.com/pytorch-labs/segment-anything-fast/blob/main/segment_anything_fast/modeling/image_encoder.py#L236)。

### kernel 追踪

我们现在可以看到，特别是内存高效注意力kernel 在 GPU 上占用了大量计算时间：

![](https://files.mdnice.com/user/59/b0737707-88d1-4a2a-b986-c46c260982e2.png)

使用 PyTorch 的原生 scaled_dot_product_attention，我们可以显著增加批量大小。我们现在观察到批量大小 32 和上述更改的以下测量结果。

![](https://files.mdnice.com/user/59/53121902-0fd3-4c89-b93b-be4ee0b3d686.png)

## Triton：用于融合相对位置编码的自定义 SDPA

暂时从推理吞吐量转移，我们开始分析整体 SAM 内存。在图像编码器中，我们看到了显著的内存分配峰值：

![](https://files.mdnice.com/user/59/64e67ba9-a8b0-4d59-bfc8-015819704d49.png)

放大看，我们看到这个分配发生在 add_decomposed_rel_pos 中，在以下行(https://github.com/pytorch-labs/segment-anything-fast/blob/main/segment_anything_fast/modeling/image_encoder.py#L373)：

![](https://files.mdnice.com/user/59/c535cc16-f847-4cf0-9cfc-a3b73f8ff032.png)

这里的 attn 变量是两个较小张量的加法：形状为 (B, q_h, q_w, k_h, 1) 的 rel_h 和形状为 (B, q_h, q_w, 1, k_w) 的 rel_w。

内存高效注意力kernel （通过 SDPA 使用）在注意力偏置大小超过 3.0GiB 时需要很长时间并不令人惊讶。如果我们不分配这个大的 attn 张量，而是将两个较小的 rel_h 和 rel_w 张量传递给 SDPA，并仅在需要时构造 attn，我们预期会有显著的性能提升。

不幸的是，这不是一个简单的修改；SDPA kernel 是高度优化的，用 CUDA 编写。我们可以转向 Triton，使用他们易于理解和使用的 FlashAttention 实现教程(https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)。经过大量挖掘并与 xFormer 的 Daniel Haziza 密切合作，我们找到了一种输入形状的情况，其中实现kernel 的融合版本相对简单。详细信息已添加到存储库(https://github.com/pytorch-labs/segment-anything-fast/blob/main/segment_anything_fast/flash_4.py)。令人惊讶的是，对于推理情况，这可以在不到 350 行代码中完成。

这是使用 Triton 代码直接构建的新kernel 扩展 PyTorch 的一个很好的例子。

### kernel 追踪

![](https://files.mdnice.com/user/59/5f93a44f-b57c-4870-9b6e-172df86f5506.jpg)

使用我们自定义的位置 Triton kernel ，我们观察到批量大小 32 的以下测量结果。

![](https://files.mdnice.com/user/59/d89a889f-0805-400f-9b22-da3a0ccc8344.png)

## NT：NestedTensor 和批处理 predict_torch

我们在图像编码器上花费了很多时间。这是有道理的，因为它占用了最多的计算时间。然而，此时它已经相当优化了，占用最多时间的算子需要大量额外投资才能改进。

我们发现了掩码预测管道的一个有趣观察(https://github.com/pytorch-labs/segment-anything-fast/blob/7cd6ba3cea451602acb7d36d176da06c70ac68f1/experiments/eval_combo.py#L137-L157)：对于我们拥有的每个图像，都有一个关联的 size、coords 和 fg_labels 张量。这些张量每个都有不同的批量大小。每个图像本身也有不同的大小。这种数据表示看起来像锯齿状数据(https://en.wikipedia.org/wiki/Jagged_array)。使用 PyTorch 最近发布的 NestedTensor(pytorch.org/tutorials/prototype/nestedtensor.html)，我们可以修改我们的数据管道，将批量坐标和 fg_labels 张量组合成单个 NestedTensor。这对跟随图像编码器的提示编码器和掩码解码器可能有显著的性能好处。调用：

```python
torch.nested.nested_tensor(data, dtype=dtype, layout=torch.jagged)
```

### kernel 追踪

![](https://files.mdnice.com/user/59/7d10b00f-944e-41cd-850a-f5fcc5629e81.png)

![](https://files.mdnice.com/user/59/f99e260b-3f57-48a1-b14d-fde55767e011.jpg)

我们现在可以看到，我们可以比 GPU 处理更快地从 CPU 启动kernel ，并且它在我们的计时区域末尾花费了很长时间等待 GPU 完成（cudaDeviceSynchronize）。我们也不再看到 GPU 上kernel 之间的空闲时间（空白区域）。

使用嵌套张量，我们观察到批量大小 32 和上述更改的以下测量结果。

![](https://files.mdnice.com/user/59/5c002eaf-9333-4efd-a012-efbceb2e612f.png)

## int8：量化和近似矩阵乘法

我们注意到在上面的追踪中，现在在 GEMM kernel 上花费了大量时间。我们已经优化得足够多，现在看到矩阵乘法在推理中比缩放点积注意力占用更多时间。

基于从 fp32 到 bfloat16 的早期学习，让我们更进一步，通过 int8 量化模拟更低的精度。查看量化方法，我们专注于动态量化(https://docs.pytorch.org/tutorials/recipes/quantization.html)，其中我们的模型观察层的可能输入和权重的范围，并细分可表达的 int8 范围以均匀地"分散"观察到的值。最终，每个浮点输入将映射到范围 [-128, 127] 中的单个整数。更多信息请参见 PyTorch 的量化教程(https://docs.pytorch.org/tutorials/recipes/quantization.html)。

降低精度可以立即节省峰值内存，但要实现推理加速，我们必须充分利用通过 SAM 操作的 int8。这需要构建高效的 int8@int8 矩阵乘法kernel ，以及从高精度到低精度（量化）的转换逻辑，以及从低精度回到高精度的反向转换（反量化）。利用 torch.compile 的强大功能，我们可以编译并将这些量化和反量化例程融合到高效的单个kernel 和我们矩阵乘法的epilogues中。生成的实现相当简短，不到 250 行代码(https://github.com/pytorch-labs/segment-anything-fast/blob/21b0208ae46eefc5659f7f200a2bf447add8765b/segment_anything_fast/dynamic_quant.py)。有关 API 和用法的更多信息，请参见 pytorch-labs/ao。

虽然在推理时量化模型时通常会看到一些精度回归，但 SAM 对低精度推理特别稳健，精度损失最小。添加量化后，我们现在观察到**批量大小 32**和上述更改的以下测量结果。

![](https://files.mdnice.com/user/59/7bb0c2b4-9b32-4f4f-a50d-f297fe4d75b4.png)

## sparse：半结构化 (2:4) 稀疏性

矩阵乘法仍然是我们的瓶颈。我们可以使用模型加速手册中的另一种经典方法来近似矩阵乘法：稀疏化。通过稀疏化我们的矩阵（即，将值归零），我们理论上可以使用更少的位来存储权重和激活张量。我们决定将张量中哪些权重设置为零的过程称为剪枝。剪枝背后的想法是，权重张量中的小权重对层的净输出贡献很少，通常是权重与激活的乘积。剪掉小权重可能会减少模型大小而不会显著损失精度。

剪枝方法多种多样，从完全非结构化（其中权重被贪婪地剪枝）到高度结构化（其中张量的大子组件一次被剪枝）。方法的选择并非微不足道。虽然非结构化剪枝理论上对精度的影响可能最小，但 GPU 在乘以大的密集矩阵方面也非常高效，在稀疏情况下可能会遭受显著的性能下降。PyTorch 支持的一种最近的剪枝方法寻求取得平衡，称为半结构化（或 2:4）稀疏性。这种稀疏存储将原始张量减少了显著的 50%，同时产生可以利用高性能 2:4 GPU kernel 的密集张量输出。请参见以下图片进行说明。

![](https://files.mdnice.com/user/59/553ec62c-1a6a-4cea-8a5c-a33b10a975de.png)

来自 developer.nvidia.com/blog/exploiting-ampere-structured-sparsity-with-cusparselt

为了使用这种稀疏存储格式和相关的快速kernel ，我们需要剪枝我们的权重，使它们符合格式的约束。我们选择在 1x4 区域中剪枝两个最小的权重，测量性能与精度的权衡。将权重从其默认的 PyTorch（"strided"）布局更改为这种新的半结构化稀疏布局很容易。要实现 `apply_sparse(model)`，我们只需要 32 行 Python 代码：

```python
import torch
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor

# Sparsity helper functions
def apply_fake_sparsity(model):
    """
    This function simulates 2:4 sparsity on all linear layers in a model.
    It uses the torch.ao.pruning flow.
    """
    # torch.ao.pruning flow
    from torch.ao.pruning import WeightNormSparsifier
    sparse_config = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            sparse_config.append({"tensor_fqn": f"{name}.weight"})

    sparsifier = WeightNormSparsifier(sparsity_level=1.0,
                                      sparse_block_shape=(1,4),
                                      zeros_per_block=2)
    sparsifier.prepare(model, sparse_config)
    sparsifier.step()

    sparsifier.step()
    sparsifier.squash_mask()


def apply_sparse(model):
    apply_fake_sparsity(model)
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            mod.weight = torch.nn.Parameter(to_sparse_semi_structured(mod.weight))
```

使用 2:4 稀疏性，我们在 vit_b 和批量大小 32 的 SAM 上观察到峰值性能：

![](https://files.mdnice.com/user/59/0299ec8e-11e8-4686-8e98-65d8ce19968c.png)

# 结论

总结一下，我们很兴奋地宣布(https://www.youtube.com/watch?v=IWpM_9AsC-U)了我们迄今为止最快的 Segment Anything(https://github.com/facebookresearch/segment-anything) 实现。我们使用众多新发布的特性在纯 PyTorch 中重写了 Meta 的原始 SAM，且没有精度损失：

- Torch.compile PyTorch 的原生 JIT 编译器，提供快速、自动化的 PyTorch 操作融合 [教程]
- GPU 量化 使用降低精度操作加速模型 [https://github.com/pytorch/ao/tree/main#torchao]
- 缩放点积注意力 (SDPA) 一种新的内存高效注意力实现 [https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html]
- 半结构化 (2:4) 稀疏性 使用更少的位来存储权重和激活来加速模型 [https://docs.pytorch.org/tutorials/prototype/semi_structured_sparse.html]
- 嵌套张量 用于非均匀批量和图像大小的高度优化的锯齿状数组处理 [https://docs.pytorch.org/tutorials/prototype/nestedtensor.html]
- Triton kernel 。通过 Triton 轻松构建和优化的自定义 GPU 操作

有关如何重现本博客文章中呈现的数据的更多详细信息，请查看 segment-anything-fast 的实验文件夹(https://github.com/pytorch-labs/segment-anything-fast/tree/main/experiments)。如果您遇到任何技术问题，请不要犹豫联系我们或提出问题。

在我们的下一篇文章中，我们很兴奋地分享我们使用 PyTorch 原生编写的 LLM 的类似性能提升！

## 致谢

我们要感谢 Meta 的 xFormers(https://github.com/facebookresearch/xformers) 团队，包括 Daniel Haziza 和 Francisco Massa，他们编写了 SDPA kernel 并帮助我们设计了自定义的一次性 Triton kernel 。



