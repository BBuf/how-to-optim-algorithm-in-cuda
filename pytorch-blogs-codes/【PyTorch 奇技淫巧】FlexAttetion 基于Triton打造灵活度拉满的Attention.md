> blog链接：https://pytorch.org/blog/flexattention/
> 代码示例：https://github.com/pytorch-labs/attention-gym/blob/main/examples/flex_attn.ipynb

# FlexAttention: PyTorch 的灵活性与 FlashAttention 的性能

> by Team PyTorch: Horace He, Driss Guessous, Yanbo Liang, Joy Dong

![](https://files.mdnice.com/user/59/76f9c77b-49f7-421b-88f5-fcb9f78d8624.png)

理论上，Attention is All You Need。然而在实践中，我们还需要像FlashAttention这样的优化注意力实现。

尽管这些融合的注意力实现显著提高了性能并支持了长上下文，但这种效率是以灵活性为代价的。你不能再通过编写几个PyTorch操作符来尝试新的注意力变体——你通常需要编写一个新的自定义kernel！这对机器学习研究人员来说就像是一种“软件彩票”——如果你的注意力变体不适合现有的优化kernel之一，你注定会面临缓慢的运行时间和CUDA内存不足的问题。

一些注意力变体的例子包括因果关系、相对位置嵌入（https://paperswithcode.com/method/relative-position-encodings）、Alibi（https://paperswithcode.com/method/alibi）、滑动窗口注意力（https://mistral.ai/news/announcing-mistral-7b/）、PrefixLM（https://x.com/andersonbcdefg/status/1800907703688339569?mx=2）、Document Masking/Sample Packing/Jagged Tensors（https://github.com/pytorch/torchtune/pull/875）、Tanh Soft-Capping（https://x.com/LysandreJik/status/1807779471891538199）、分页注意力（https://arxiv.org/abs/2309.06180）等。更糟糕的是，人们通常想要这些的组合！滑动窗口注意力 + Document Masking + 因果关系 + 上下文并行？或者分页注意力 + 滑动窗口 + Tanh Soft-Capping？

下图左侧代表了当今世界的现状——一些掩码 + 偏置 + 设置的组合已经有了现有的kernel实现。但各种选项导致了指数级的设置数量，因此我们最终得到了相当零散的支持。更糟糕的是，研究人员提出的新注意力变体将完全没有支持。

![](https://files.mdnice.com/user/59/0c64c6d6-2813-4ff8-9840-f50ed34c893e.png)

为了彻底解决这个超立方体问题，我们引入了**FlexAttention**，一个新的PyTorch API。

- 我们提供了一个灵活的API，允许在几行惯用的PyTorch代码中实现许多注意力变体（包括博客文章中提到的所有变体）。
- 我们通过`torch.compile`将其lower为融合的FlashAttention kernel，生成的FlashAttention kernel不会具体化任何额外内存，并且性能与手写的kernel相当。
- 我们还利用PyTorch的自动求导机制自动生成反向传播。
- 最后，我们还可以利用注意力掩码中的稀疏性，从而在标准注意力实现上取得显著改进。

通过FlexAttention，我们希望尝试新的注意力变体将仅受限于你的想象力。

你可以在Attention Gym中找到许多FlexAttention示例：https://github.com/pytorch-labs/attention-gym。如果你有任何酷炫的应用，欢迎提交示例！

PS：我们也发现这个API非常令人兴奋，因为它以一种有趣的方式利用了大量现有的PyTorch基础设施——更多内容将在最后介绍。

## FlexAttention

这里是经典的注意力方程：

![](https://files.mdnice.com/user/59/d4cbc585-804e-40f1-891a-99081c5a45d3.png)

以代码形式表示：

```python
Q, K, V: Tensor[batch_size, num_heads, sequence_length, head_dim]
score: Tensor[batch_size, num_heads, sequence_length, sequence_length] = (Q @ K) / sqrt(head_dim)
probabilities = softmax(score, dim=-1)
output: Tensor[batch_size, num_heads, sequence_length, head_dim] = probabilities @ V
```

FlexAttention 允许用户定义函数 `score_mod`：

![](https://files.mdnice.com/user/59/8fd6793e-f305-4ec8-8e8c-f8b9ed9b0574.png)

以代码形式表示：

```python
Q, K, V: Tensor[batch_size, num_heads, sequence_length, head_dim]
score: Tensor[batch_size, num_heads, sequence_length, sequence_length] = (Q @ K) / sqrt(head_dim)
modified_scores: Tensor[batch_size, num_heads, sequence_length, sequence_length] = score_mod(score)
probabilities = softmax(modified_scores, dim=-1)
output: Tensor[batch_size, num_heads, sequence_length, head_dim] = probabilities @ V
```

这个函数允许你在softmax之前修改注意力分数。令人惊讶的是，这对于绝大多数注意力变体来说已经足够了（如下面的例子所示）！

具体来说，`score_mod`的预期签名有些独特。

```python
def score_mod(score: f32[], b: i32[], h: i32[], q_idx: i32[], kv_idx: i32[])
    return score # noop - standard attention
```

换句话说，`score` 是一个标量 PyTorch 张量，表示query token和key token的点积。其余的参数告诉你当前正在计算哪个点积——`b`（当前批次中的元素），`h`（当前头），`q_idx`（query中的位置），`kv_idx`（key/value张量中的位置）。

要应用这个函数，我们可以这样实现

```python
for b in range(batch_size):
    for h in range(num_heads):
        for q_idx in range(sequence_length):
            for kv_idx in range(sequence_length):
                modified_scores[b, h, q_idx, kv_idx] = score_mod(scores[b, h, q_idx, kv_idx], b, h, q_idx, kv_idx)
```

当然，FlexAttention 的底层实现并不是这样的。通过利用 `torch.compile`，我们可以自动将你的函数降级为一个单一的融合 FlexAttention 内核——保证做到，否则退款！

这个 API 最终表现出令人惊讶的表达能力。让我们来看一些例子。

## Score Mod 示例
### 全注意力
让我们首先实现“全注意力”，或标准的双向注意力。在这种情况下，`score_mod` 是一个空操作（no-op）——它接受输入的分数，然后原样返回它们。

```python
def noop(score, b, h, q_idx, kv_idx):
    return score
```
端到端使用的例子如下（包括前向和后向）：
```python
from torch.nn.attention.flex_attention import flex_attention

flex_attention(query, key, value, score_mod=noop).sum().backward()
```

### Relative Position Encodings

一种常见的注意力变体是“相对位置编码”。与在query和key中编码绝对距离不同，相对位置编码根据query和key之间的“距离”调整分数。

```python
def relative_positional(score, b, h, q_idx, kv_idx):
    return score + (q_idx - kv_idx)
```

请注意，与经典的实现不同，这不需要具体化一个SxS的张量。相反，FlexAttention在kernel中“即时”计算偏置值，从而显著提高了内存和性能。

![](https://files.mdnice.com/user/59/8d8646bb-a53d-45e6-9726-aedea28d2e3d.png)

### ALiBi Bias

![](https://files.mdnice.com/user/59/8b73747b-182a-4e44-9872-f9d238866df8.png)

ALiBi 是在《Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation(https://arxiv.org/abs/2108.12409)》中引入的，并声称在推理时具有长度外推的有益特性。值得注意的是，MosaicML 指出“缺乏kernel支持”（https://x.com/jefrankle/status/1804567458092605736）是他们最终从 ALiBi 切换到旋转嵌入的主要原因。

Alibi 与相对位置编码类似，但有一个例外——它有一个通常预先计算的per-head因子。

```python
alibi_bias = generate_alibi_bias() # [num_heads]

def alibi(score, b, h, q_idx, kv_idx):
    bias = alibi_bias[h] * (q_idx - kv_idx)
    return score + bias
```

这展示了 `torch.compile` 提供的一个有趣的灵活性——即使 `alibi_bias` 没有被显式地作为输入传递进来，我们也可以从中加载数据！生成的 Triton kernel将计算从 `alibi_bias` 张量中正确加载的数据并将其融合。请注意，即使重新生成 `alibi_bias`，我们也不需要重新编译。

### Soft-capping

Soft-capping 是一种在 Gemma2(https://huggingface.co/blog/gemma2#soft-capping-and-attention-implementations) 和 Grok-1 中使用的技术，用于防止 logits 过度增长。在 FlexAttention 中，它看起来像这样：

```python
softcap = 20
def soft_cap(score, b, h, q_idx, kv_idx):
    score = score / softcap
    score = torch.tanh(score)
    score = score * softcap
    return score
```

请注意，我们在这里还自动从正向pass生成反向pass。此外，尽管此实现语义上是正确的，但由于性能原因，我们可能希望在这种情况下使用 tanh 近似。有关更多详细信息，请参见 attention-gym(https://github.com/pytorch-labs/attention-gym/blob/main/attn_gym/mods/softcapping.py)。


### Causal Mask

尽管双向注意力是最简单的，但《Attention is All You Need》论文和大多数LLM在仅解码器设置中使用注意力，其中每个token只能关注其之前的token。人们通常认为这是一个下三角掩码，通过`score_mod` API，它可以表示为：

```python
def causal_mask(score, b, h, q_idx, kv_idx):
    return torch.where(q_idx >= kv_idx, score, -float("inf"))
```

基本上，如果查询token在键token之后，我们保留分数。否则，我们通过将其设置为-inf来将其掩码掉，从而确保它不会参与softmax计算。

然而，与其他修改相比，掩码是特殊的——如果某些内容被掩码掉，我们可以完全跳过其计算！在这种情况下，因果掩码大约有50%的稀疏性，因此如果不利用这种稀疏性，将会导致2倍的减速。尽管这个`score_mod`足以正确实现因果掩码，但要获得稀疏性的性能优势，还需要另一个概念——`mask_mod`。

## Mask Mods

为了利用掩码带来的稀疏性，我们需要做更多的工作。具体来说，通过将 `mask_mod` 传递给 `create_block_mask`，我们可以创建一个 `BlockMask`。然后，FlexAttention 可以使用 `BlockMask` 来利用这种稀疏性！

`mask_mod` 的签名与 `score_mod` 非常相似——只是没有分数。特别是

```python
# returns True if this position should participate in the computation
mask_mod(b, h, q_idx, kv_idx) => bool
```

请注意，`score_mod` 比 `mask_mod` 更具表达力。然而，对于掩码操作，建议使用 `mask_mod` 和 `create_block_mask`，因为它们的性能更好。请参阅常见问题解答，了解为什么 `score_mod` 和 `mask_mod` 是分开的。

现在，让我们看看如何使用 `mask_mod` 实现因果掩码。

### Causal Mask

```python
from torch.nn.attention.flex_attention import create_block_mask

def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

# Because the sparsity pattern is independent of batch and heads, we'll set them to None (which broadcasts them) 
block_mask = create_block_mask(causal, B=None, H=None, Q_LEN=1024, KV_LEN=1024)
# In this case, we don't need a score_mod, so we won't pass any in.
# However, score_mod can still be combined with block_mask if you need the additional flexibility.
flex_attention(query, key, value, block_mask=block_mask)
```

请注意，`create_block_mask` 是一个**相对昂贵的操作**！尽管 FlexAttention 在更改时不需要重新编译，但如果你不注意缓存它，它可能会导致显著的减速（查看常见问题解答以获取最佳实践建议）。

![](https://files.mdnice.com/user/59/6028ea3f-3133-4a61-aa81-a710fb5e4458.png)

尽管TFlops大致相同，但mask_mod版本的执行时间快了2倍！这表明我们可以利用BlockMask提供的稀疏性，而不会损失硬件效率。

### Sliding Window + Causal

![](https://files.mdnice.com/user/59/4730b440-32f0-4a9e-9924-e29c458e0ab3.png)

由Mistral(https://arxiv.org/abs/2310.06825)推广的滑动窗口注意力（也称为局部注意力）利用了最近token最有用的直觉。特别是，它允许query token仅关注最近的1024个token。这通常与因果注意力一起使用。

```python
SLIDING_WINDOW = 1024

def sliding_window_causal(b, h, q_idx, kv_idx):
    causal_mask = q_idx >= kv_idx
    window_mask = q_idx - kv_idx <= SLIDING_WINDOW 
    return causal_mask & window_mask

# If you want to be cute...
from torch.nn.attention import or_masks

def sliding_window(b, h, q_idx, kv_idx)
    return q_idx - kv_idx <= SLIDING_WINDOW

sliding_window_causal = or_masks(causal_mask, sliding_window)
```

我们将其与带有滑动窗口掩码的 `F.scaled_dot_product_attention` 以及带有因果掩码的 FA2（作为性能参考点）进行基准测试。我们不仅显著快于 `F.scaled_dot_product_attention`，而且也显著快于带有因果掩码的 FA2，因为这种掩码具有显著更高的稀疏性。

![](https://files.mdnice.com/user/59/dd399aa9-4d5c-4820-b103-22c95518061b.png)

### PrefixLM

![](https://files.mdnice.com/user/59/107d0fa2-0684-4105-ba87-8f2fec5ae564.png)

T5架构，在《Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer(https://arxiv.org/abs/1910.10683)》中提出，描述了一种注意力变体，它在“前缀”上执行完全双向注意力，而在其余部分执行因果注意力。我们再次组合两个掩码函数来实现这一点，一个用于因果掩码，另一个基于前缀长度。

```python
prefix_length: [B]
def prefix_mask(b, h, q_idx, kv_idx):
    return kv_idx <= prefix_length[b]

prefix_lm_causal = or_masks(prefix_mask, causal_mask)
# 在这种情况下，我们的掩码在每个序列中都不同，因此我们将 B 设置为我们的批次大小
block_mask = create_block_mask(prefix_lm_causal, B=B, H=None, S, S)
```

就像 `score_mod` 一样，`mask_mod` 允许我们引用那些不是函数显式输入的额外张量！然而，对于 prefixLM，稀疏模式会随每个输入而变化。这意味着对于每个新的输入批次，我们需要重新计算 `BlockMask`。一个常见的模式是在模型的开始时调用 `create_block_mask`，并在模型的所有注意力调用中重用该 `block_mask`。请参阅“Recomputing Block Masks vs. Recompilation”。

然而，作为交换，我们不仅能够为 prefixLM 提供一个高效的注意力kernel，还能够利用输入中存在的任何稀疏性！FlexAttention 将根据 BlockMask 数据动态调整其性能，而无需重新编译kernel。

### Document Masking/Jagged Sequences

另一种常见的注意力变体是文档掩码/锯齿序列。想象一下，你有一组长度不同的序列。你希望将它们一起训练，但不幸的是，大多数操作符只接受矩形张量。

通过 `BlockMask`，我们也可以在 FlexAttention 中高效地支持这一点！

- 首先，我们将所有序列展平为一个包含 sum(sequence lengths) 个 token 的单一序列。
- 然后，我们计算每个 token 所属的 document_id。
- 最后，在我们的 `mask_mod` 中，我们只需判断 query 和 kv token 是否属于同一个文档！

```python
# The document that each token belongs to.
# e.g. [0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2] corresponds to sequence lengths 3, 2, and 6.
document_id: [SEQ_LEN]

def document_masking(b, h, q_idx, kv_idx):
    return document_id[q_idx] == document_id[kv_idx]
```

就是这样！在这种情况下，我们最终得到了一个块对角掩码。

![](https://files.mdnice.com/user/59/c455a710-2969-462f-ac18-4170c66dd17f.png)

文档掩码的一个有趣方面是，很容易看出它如何与任意组合的其他掩码结合。例如，我们在上一节中已经定义了 `prefixlm_mask`。我们现在是否还需要定义一个 `prefixlm_document_mask` 函数呢？

在这些情况下，我们发现一个非常有用的模式是我们称之为“更高层次的修改”。在这种情况下，我们可以采用现有的 `mask_mod` 并自动将其转换为适用于锯齿序列的掩码！

```python
def generate_doc_mask_mod(mask_mod, document_id):
    # 获取唯一的文档ID及其计数
    _, counts = torch.unique_consecutive(document_id, return_counts=True)
    # 创建累积计数（偏移量）
    offsets = torch.cat([torch.tensor([0], device=document_id.device), counts.cumsum(0)[:-1]])
    def doc_mask_wrapper(b, h, q_idx, kv_idx):
        # 判断query和kv token是否属于同一个文档
        same_doc = document_id[q_idx] == document_id[kv_idx]
        # 计算逻辑索引
        q_logical = q_idx - offsets[document_id[q_idx]]
        kv_logical = kv_idx - offsets[document_id[kv_idx]]
        # 应用内部的掩码函数
        inner_mask = mask_mod(b, h, q_logical, kv_logical)
        # 返回组合掩码
        return same_doc & inner_mask
    return doc_mask_wrapper
```

例如，给定上面的 `prefix_lm_causal` 掩码，我们可以将其转换为适用于packed documents的掩码，如下所示：

```python
prefix_length = torch.tensor(2, dtype=torch.int32, device="cuda")
def prefix_mask(b, h, q_idx, kv_idx):
    return kv_idx < prefix_length
prefix_lm_causal = or_masks(prefix_mask, causal_mask)
doc_prefix_lm_causal_mask = generate_doc_mask_mod(prefix_lm_causal, document_id)
```

![](https://files.mdnice.com/user/59/14dddd97-d012-409e-bf87-c67134d4d1d3.png)

> packed documents：前缀上(2)执行双向注意力，其它部分执行因果注意力。

现在，这个掩码是“块-前缀LM-对角线”形状的。 :)

这就是我们所有的例子！还有更多的注意力变体是我们无法一一列举的，所以请查看Attention Gym以获取更多示例。我们希望社区也能贡献一些他们最喜欢的FlexAttention应用。

## FAQ

**Q: FlexAttention 何时需要重新编译？**

由于 FlexAttention 利用 `torch.compile` 进行图捕获，它实际上可以在广泛的情况下避免重新编译。值得注意的是，即使捕获的张量值发生变化，它也不需要重新编译！

```python
flex_attention = torch.compile(flex_attention)
def create_bias_mod(bias)
    def bias_mod(score, b, h, q_idx, kv_idx):
        return score + bias
    return bias_mod
bias_mod1 = create_bias_mod(torch.tensor(0))
flex_attention(..., score_mod=bias_mod1) # Compiles the kernel here 

bias_mod2 = create_bias_mod(torch.tensor(2))
flex_attention(..., score_mod=bias_mod2) # Doesn't need to recompile! 
```

即使块稀疏度发生变化，也不需要重新编译。然而，如果块稀疏度发生变化，我们需要重新计算 BlockMask。

**Q: 何时应该重新计算BlockMask？**

每当块稀疏度发生变化时，我们需要重新计算BlockMask。尽管计算BlockMask比重新编译便宜得多（以微秒为单位，而不是秒），但您仍应注意不要过度重新计算BlockMask。

以下是一些常见模式以及一些关于如何处理它们的建议。

**掩码永不改变（例如因果掩码）**
在这种情况下，您可以简单地预计算块掩码并全局缓存它，以便在所有注意力调用中重复使用。

```python
block_mask = create_block_mask(causal_mask, 1, 1, S,S)
causal_attention = functools.partial(flex_attention, block_mask=block_mask)
```






