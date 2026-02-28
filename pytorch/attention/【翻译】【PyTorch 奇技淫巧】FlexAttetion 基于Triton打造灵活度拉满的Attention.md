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
**掩码每批次变化（例如文档掩码）**
在这种情况下，我们建议在模型的开始处计算BlockMask，并将其传递到模型中——在所有层中重复使用BlockMask。

```python
def forward(self, x, doc_mask):
    # Compute block mask at beginning of forwards
    block_mask = create_block_mask(doc_mask, None, None, S, S)    
    x = self.layer1(x, block_mask)
    x = self.layer2(x, block_mask)
    ...
    # amortize block mask construction cost across all layers
    x = self.layer3(x, block_mask) 
    return x
```

**每层掩码变化（例如数据依赖的稀疏性）**
这是最困难的设置，因为我们无法在多个 FlexAttention 调用中分摊块掩码的计算。尽管 FlexAttention 仍然可以在此情况下受益，但 BlockMask 的实际好处取决于您的注意力掩码的稀疏程度以及我们构建 BlockMask 的速度。这引导我们……

> 注：这里没有继续给出解决方法

**Q: 如何更快地计算BlockMask？**

`create_block_mask` 不幸地相当昂贵，无论是从内存还是计算的角度来看，因为确定一个块是否完全稀疏需要评估 `mask_mod` 在块中的每一个点。有几种方法可以解决这个问题：

- 如果你的掩码在批次大小或头数上是相同的，确保你在这些维度上进行广播（即在 `create_block_mask` 中将它们设置为 `None`）。
- 编译 `create_block_mask`。不幸的是，目前 `torch.compile` 不能直接在 `create_block_mask` 上工作，因为一些不幸的限制。然而，你可以设置 `_compile=True`，这将显著减少峰值内存和运行时间（在我们的测试中通常是一个数量级的减少）。
- 编写一个自定义的 BlockMask 构造器。BlockMask 的元数据非常简单（参见文档 https://pytorch.org/docs/main/nn.attention.flex_attention.html#blockmask ）。它基本上是两个张量。a. `num_blocks`：为每个query块计算的 KV 块的数量。b. `indices`：为每个query块计算的 KV 块的位置。

例如，这里是一个用于 `causal_mask` 的自定义 BlockMask 构造器。

```python
def create_causal_mask(S):
    BLOCK_SIZE = 128
    # 第一个查询块计算一个块，第二个查询块计算两个块，依此类推。
    num_blocks = torch.arange(S // BLOCK_SIZE, device="cuda") + 1
    # 由于我们总是从左到右计算，
    # 我们可以为每个查询块使用索引 [0, 1, 2, ...]。
    indices = torch.arange(S // BLOCK_SIZE, device="cuda").expand(
        S // BLOCK_SIZE, S // BLOCK_SIZE
    )
    num_blocks = num_blocks[None, None, :]
    indices = indices[None, None, :]
    return BlockMask(num_blocks, indices, BLOCK_SIZE=BLOCK_SIZE, mask_mod=causal_mask)
```

**Q: 为什么 `score_mod` 和 `mask_mod` 不同？难道 `mask_mod` 不就是 `score_mod` 的一个特例吗？**
非常敏锐的问题！事实上，任何 `mask_mod` 都可以很容易地转换为 `score_mod`（我们不建议在实践中使用这个函数！）

```python
def mask_mod_as_score_mod(b, h, q_idx, kv_idx):
    return torch.where(mask_mod(b, h, q_idx, kv_idx), score, -float("inf"))
```

所以，如果 `score_mod` 可以实现 `mask_mod` 的所有功能，那么为什么还需要 `mask_mod` 呢？

一个直接的挑战是：`score_mod` 需要实际的分数值作为输入，但在我们预计算 BlockMask 时，我们没有实际的分数值。我们可以通过传入全零来伪造这些值，如果 `score_mod` 返回 `-inf`，那么我们就认为它被掩码了（事实上，我们最初就是这样做的！）。

然而，有两个问题。第一个问题是这是hacky的——如果用户的 `score_mod` 在输入为0时返回 `-inf` 怎么办？或者如果用户的 `score_mod` 用一个大的负值而不是 `-inf` 来掩码怎么办？看起来我们正在试图把一个圆钉子塞进一个方孔里。然而，还有一个更重要的原因来分离 `mask_mod` 和 `score_mod`——它从根本上来说更高效！

事实证明，对每个计算的元素应用掩码实际上相当昂贵——我们的基准测试显示性能下降了大约15-20%！因此，尽管我们可以通过跳过一半的计算来获得显著的加速，但我们从需要掩码每个元素中失去了部分加速！

幸运的是，如果我们可视化因果掩码，我们会注意到绝大多数块根本不需要“因果掩码”——它们是完全计算的！只有对角线上的块，部分计算和部分掩码，才需要应用掩码。

![](https://files.mdnice.com/user/59/cc316171-a969-44ca-9f4b-8647191c173a.png)

BlockMask 之前告诉我们哪些块需要计算，哪些块可以跳过。现在，我们进一步增强这个数据结构，以告诉我们哪些块是“完全计算”的（即可以跳过掩码），哪些块是“部分计算”的（即需要应用掩码）。然而，需要注意的是，尽管在“完全计算”的块上可以跳过掩码，但其他 `score_mods` 如相对位置嵌入仍然需要应用。

仅给定一个 `score_mod`，我们无法明确地知道它的哪些部分是“掩码”。因此，用户必须将这些部分自己分离到 `mask_mod` 中。

**Q: BlockMask 需要多少额外的内存？**
BlockMask 的元数据大小为 `[BATCH_SIZE, NUM_HEADS, QUERY_LEN//BLOCK_SIZE, KV_LEN//BLOCK_SIZE]`。如果掩码在批次或头维度上是相同的，它可以在这个维度上进行广播以节省内存。

在默认的 `BLOCK_SIZE` 为 128 的情况下，我们预计大多数用例的内存使用量将非常小。例如，对于序列长度为 100 万的情况，BlockMask 只会使用 60MB 的额外内存。如果这是一个问题，你可以增加块大小：`create_block_mask(..., BLOCK_SIZE=1024)`。例如，将 `BLOCK_SIZE` 增加到 1024 将使这个元数据减少到不到 1MB。

**Q: 数值比较如何？**
尽管结果不是逐位相同的，但我们有信心FlexAttention在数值精度上与FlashAttention相当。我们生成了以下差异分布，比较了FlashAttention与FlexAttention在大量输入上的因果和非因果注意力变体。误差几乎相同。

![](https://files.mdnice.com/user/59/ab342da2-b26b-410e-a7b7-547f4fdc169b.png)

## Performance
一般来说，FlexAttention 的性能几乎与手写的 Triton kernel相当，这并不令人意外，因为我们大量利用了手写的 Triton kernel。然而，由于其通用性，我们确实会受到一些性能损失。例如，我们必须承担一些额外的延迟来确定下一个要计算的块。在某些情况下，我们提供了一些kernel选项，这些选项可以在改变其行为的同时影响kernel的性能。它们可以在这里找到：性能旋钮（https://github.com/pytorch/pytorch/blob/ee09d066d35d7e17cf7e9479c0b8bfc70cffc264/torch/_inductor/kernel/flex_attention.py#L146-L155）

作为一个案例研究，让我们探讨这些旋钮如何影响因果注意力的性能。我们将在 A100 上比较 Triton kernel与 FlashAttentionv2 的性能。脚本可以在这里找到（https://github.com/pytorch/pytorch/blob/main/benchmarks/transformer/score_mod.py）。

FlexAttention 在前向pass中达到了 FlashAttention2 性能的 90%，在后向pass中达到了 85%。FlexAttention 目前使用了一种确定性算法，该算法比 FAv2 重新计算更多的中间结果，但我们有计划改进 FlexAttention 的后向算法，并希望缩小这一差距！

![](https://files.mdnice.com/user/59/4ac2f33b-5c56-41d2-a52a-bca48be3b7fa.png)

![](https://files.mdnice.com/user/59/d806f798-e280-4916-9de9-9413fa612101.png)

## 结论

我们希望您在使用FlexAttention时能像我们开发它时一样有趣！在开发过程中，我们最终发现了比预期更多的API应用。我们已经看到它将torchtune的sample packing吞吐量提高了71%(https://github.com/pytorch/torchtune/pull/1193)，取代了研究人员花费一周时间编写自己的定制Triton kernel的需求，并以与定制手写注意力变体相竞争的性能交付。

实现FlexAttention非常有趣的一个最后一点是，我们能够以一种有趣的方式利用大量现有的PyTorch基础设施。例如，TorchDynamo（torch.compile的前端）的一个独特之处在于，它不需要在编译函数中使用的张量显式地作为输入传递。这使我们能够编译像document masking这样的修改，这些修改需要访问全局变量，而这些全局变量需要更改！

```python
bias = torch.randn(1024, 1024)
def score_mod(score, b, h, q_idx, kv_idx):
    return score + bias[q_idx][kv_idx] # The bias tensor can change!
```

此外，`torch.compile` 作为一个通用的图捕获机制，也使其能够支持更多“高级”的转换，例如将任何 `mask_mod` 转换为适用于锯齿张量的更高阶转换。

我们还利用了 TorchInductor（torch.compile 的后端）基础设施来支持 Triton 模板。这不仅使支持 FlexAttention 的代码生成变得容易，还自动为我们提供了对动态形状以及epilogue融合（即在注意力末尾融合操作符）的支持！在未来，我们计划扩展这种支持，以允许量化版本的注意力或类似 RadixAttention(https://lmsys.org/blog/2024-01-17-sglang/)的东西。

此外，我们还利用了高阶操作、PyTorch 的 autograd 来自动生成反向传播，以及 vmap 来自动应用 `score_mod` 来创建 BlockMask。

当然，如果没有 Triton 和 TorchInductor 生成 Triton 代码的能力，这个项目是不可能实现的。

我们期待着利用我们在这里使用的方法，在未来应用于更多的应用！

## 限制与未来工作

- FlexAttention 目前仅在 PyTorch 的 nightly 版本中可用，我们计划在 2.5.0 版本中将其作为原型功能发布。
- 我们没有在这里介绍如何使用 FlexAttention 进行推理（或如何实现 PagedAttention）——我们将在后续文章中介绍这些内容。
- 我们正在努力改进 FlexAttention 的性能，以匹配 H100 GPU 上的 FlashAttention3。
- FlexAttention 要求所有序列长度必须是 128 的倍数——这个问题将很快得到解决。
- 我们计划很快添加 GQA 支持——目前，您可以简单地复制 kv heads。

## 致谢

我们想强调一些先前的工作（和人），这些工作启发了 FlexAttention。
- Tri Dao 在 FlashAttention 上的工作
- Francisco Massa 和 Xformers 团队在 Triton 中的 BlockSparseAttention
- Jax 团队在 SplashAttention 上的工作
- Philippe Tillet 和 Keren Zhou 帮助我们使用 Triton
- Ali Hassani 在邻域注意力上的讨论
- 所有抱怨注意力kernel不支持他们最喜欢的注意力变体的人 :)








