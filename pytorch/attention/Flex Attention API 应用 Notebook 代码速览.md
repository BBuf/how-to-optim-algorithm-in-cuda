> 对FlexAttention的常见API的使用方法做一个解读，博客来源：https://github.com/pytorch-labs/attention-gym/blob/main/examples/flex_attn.ipynb ，在此基础上我对部分代码添加了一些解释，修复了几个代码中的bug并使用PyTorch的nightly版本运行了示例，得到了每个custom attention的输出，展示在了下面的每个示例代码后面。最后还补充了一下torch compile inductor后端中实现FlexAttention的入口的代码浏览。

# FlexAttention API 使用 NoteBook

本笔记本演示了新的 FlexAttention API 的使用方法，该 API 允许用户指定对缩放点积注意力（SDPA）中计算的注意力分数进行修改。

## 目录

1. [介绍](#介绍)
2. [设置](#设置)
3. [基本用法](#基本用法)
4. [分数修改 vs 分数掩码](#分数修改vs分数掩码)
5. [分数修改示例](#分数修改示例)
   - [全注意力（无操作）](#全注意力)
   - [标准因果掩码](#标准因果掩码)
   - [滑动窗口注意力](#滑动窗口注意力)
   - [前缀 LM（双向 + 因果）](#prefix-lm-bidirectional-causal)
   - [文档掩码](#文档掩码)
   - [NATTEN 掩码](#natten-masking)
   - [Alibi 偏置](#alibi-bias)
   - [Tanh 软上限](#tanh-soft-capping)
   - [嵌套不规则张量](#nested-jagged-tensor)
   - [Flamingo 交叉注意力](#flamingo-cross-attention)

## 介绍

FlexAttention API 允许用户在Fused Scaled Dot Product Attention Kernel中指定对注意力分数的自定义修改。这使得各种注意力模式和偏置能够高效地实现，并具有潜在的运行时和内存节省。API 还将根据用户定义的修改生成融合的反向kernel。

## 设置
首先，让我们导入必要的库并设置我们的环境。

```python
import random
from functools import lru_cache, partial

import torch
import torch.nn.functional as F

from tabulate import tabulate
from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    create_block_mask,
    create_mask,
    flex_attention,
)
from triton.testing import do_bench

torch.set_default_device("cuda")
torch.manual_seed(0)

torch._dynamo.config.cache_size_limit = 1000

# Compile the flex_attention function
flex_attention = torch.compile(flex_attention, dynamic=False)

# For better performance, you can use:
# flex_attention = torch.compile(_flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")

data_type = torch.float16

# The kernels will utilize block sparisty to increase performance
print(f"Using the default sparsity block size: {_DEFAULT_SPARSE_BLOCK_SIZE}")
```

我们将定义一些有用的测试工具，这些工具将打印score_mod函数和mask_fn的块稀疏表示。

此外，它将比较以下几种实现的性能：

- FlexAttention
- 一种FlashAttentionV2的SOTA实现，带有因果掩码。
- `nn.F.scaled_dot_product_attention` + 完全具体化的attn_mask。这将dispatch到一个融合实现`EFFICIENT_ATTENTION`，允许任意掩码。

```python
@lru_cache
def create_block_mask_cached(score_mod, B, H, M, N, device="cuda"):
    """
    创建并缓存块掩码。
    
    参数:
    - score_mod: 分数修改函数
    - B: 批次大小
    - H: 头数
    - M: 查询序列长度
    - N: 键值序列长度
    - device: 设备类型
    
    返回:
    - block_mask: 创建的块掩码
    """
    block_mask = create_block_mask(score_mod, B, H, M, N, device=device)
    return block_mask


def calculate_tflops(flops: float, time_ms: float, multiplier: int) -> float:
    """
    计算TFLOPS。
    
    参数:
    - flops: 浮点运算次数
    - time_ms: 时间（毫秒）
    - multiplier: 乘数
    
    返回:
    - TFLOPS值
    """
    return multiplier * flops * (1e3 / time_ms) / 1e12


def test_mask(
    score_mod=None,
    mask_mod=None,
    B=16,
    H=16,
    S=8192,
    D=64,
    skip_correctness=False,
    print_mask=True,
):
    """
    测试掩码功能。
    
    参数:
    - score_mod: 分数修改函数
    - mask_mod: 掩码修改函数
    - B: 批次大小
    - H: 头数
    - S: 序列长度
    - D: 嵌入维度
    - skip_correctness: 是否跳过正确性检查
    - print_mask: 是否打印掩码
    """
    assert (
        score_mod is not None or mask_mod is not None
    ), "Must provide a score_mod or mask_mod"
    
    # 创建输入张量
    query = torch.randn(
        B, H, S, D, device="cuda", dtype=torch.float16, requires_grad=True
    )
    key = torch.randn(
        B, H, S, D, device="cuda", dtype=torch.float16, requires_grad=True
    )
    value = torch.randn(
        B, H, S, D, device="cuda", dtype=torch.float16, requires_grad=True
    )
    gradOut = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)

    # 创建块掩码
    if mask_mod is not None:
        block_mask = create_block_mask_cached(mask_mod, 1, 1, S, S, device=query.device)
    else:
        block_mask = None
    
    # 确定掩码函数
    sdpa_mask_fn = mask_mod if mask_mod is not None else score_mod
    mask = create_mask(sdpa_mask_fn, 1, 1, S, S, device=query.device)

    # 定义不同的注意力计算函数
    causal_fa2 = lambda: F.scaled_dot_product_attention(
        query, key, value, is_causal=True
    )
    xformers_mask = lambda: F.scaled_dot_product_attention(
        query, key, value, attn_mask=mask
    )
    flex_attention_call = lambda: flex_attention(
        query, key, value, score_mod=score_mod, block_mask=block_mask
    )

    results = []
    
    # 计算密度
    if block_mask is not None:
        density = (100 - block_mask.sparsity()) / 100
    else:
        density = 1.0
    
    # 计算浮点运算次数
    causal_fav2_flops = 0.5 * B * H * D * S * S
    flops = density * B * H * D * S * S

    # 前向传播时间
    causal_fa2_time = do_bench(causal_fa2)
    xformers_mask_time = do_bench(xformers_mask)
    flex_ms = do_bench(flex_attention_call)

    # 后向传播时间
    causal_fa2_out = causal_fa2()
    xformers_out = xformers_mask()
    flex_out = flex_attention_call()

    causal_fa2_bw_time = do_bench(
        lambda: causal_fa2_out.backward(gradOut, retain_graph=True)
    )
    xformers_mask_bw_time = do_bench(
        lambda: xformers_out.backward(gradOut, retain_graph=True)
    )
    flex_bw_ms = do_bench(lambda: flex_out.backward(gradOut, retain_graph=True))

    # 正确性检查
    if not skip_correctness:
        xformers_outs = []
        flex_outs = []

        query.grad = None
        key.grad = None
        value.grad = None

        out1 = xformers_mask()
        xformers_outs.append(out1)
        out1.backward(gradOut)
        xformers_outs += [query.grad, key.grad, value.grad]

        query.grad = None
        key.grad = None
        value.grad = None

        out2 = flex_attention_call()
        flex_outs.append(out2)
        out2.backward(gradOut)
        flex_outs += [query.grad, key.grad, value.grad]
        for flex, xformer in zip(flex_outs, xformers_outs):
            torch.testing.assert_close(flex, xformer, atol=1e-1, rtol=1e-2)

        print("Correctness check passed ✅")
    
    # 结果格式化
    results = [
        [
            "causal FA2",
            f"{causal_fa2_time:.4f}",
            f"{calculate_tflops(causal_fav2_flops, causal_fa2_time, 4):.2f}",
            f"{causal_fa2_bw_time:.4f}",
            f"{calculate_tflops(causal_fav2_flops, causal_fa2_bw_time, 10):.2f}",
        ],
        [
            "F.sdpa + mask",
            f"{xformers_mask_time:.4f}",
            f"{calculate_tflops(flops, xformers_mask_time, 4):.2f}",
            f"{xformers_mask_bw_time:.4f}",
            f"{calculate_tflops(flops, xformers_mask_bw_time, 10):.2f}",
        ],
        [
            "flexattention",
            f"{flex_ms:.4f}",
            f"{calculate_tflops(flops, flex_ms, 4):.2f}",
            f"{flex_bw_ms:.4f}",
            f"{calculate_tflops(flops, flex_bw_ms, 10):.2f}",
        ],
    ]
    print(
        f"\nResults for {score_mod.__name__ if score_mod is not None else mask_mod.__name__}:"
    )
    print(
        tabulate(
            results,
            headers=[
                "Operation",
                "FW Time (ms)",
                "FW FLOPS (TF/s)",
                "BW Time (ms)",
                "BW FLOPS (TF/s)",
            ],
            tablefmt="grid",
        )
    )
    if print_mask:
        print(f"\nBlock Mask:\n{block_mask}")

    # 清理内存
    del query, key, value, gradOut, causal_fa2_out, xformers_out, flex_out
    torch.cuda.empty_cache()
```

> 这里的multiplier为什么是4和10没搞清楚。

## 基本用法

以下是如何使用FlexAttention API的基本示例：

```python

def checkerboard(score, batch, head, token_q, token_kv):
    score = torch.where(torch.abs(token_kv - token_q) % 1 == 0, score * 0.5, score)
    score = torch.where(torch.abs(token_kv - token_q) % 2 == 0, score * 2.0, score)
    return score


# Create input tensors
query = torch.randn(8, 8, 2048, 64, device="cuda", dtype=torch.float32)
key = torch.randn(8, 8, 2048, 64, device="cuda", dtype=torch.float32)
value = torch.randn(8, 8, 2048, 64, device="cuda", dtype=torch.float32)

# Call flex_attention with the checkerboard score modification
output = flex_attention(query, key, value, score_mod=checkerboard)

# Compile and run
compiled_flex_attention = torch.compile(flex_attention)
out_compiled = compiled_flex_attention(query, key, value, score_mod=checkerboard)

# Check if the results are close
torch.testing.assert_close(output, out_compiled, atol=2e-2, rtol=2e-2)
```

## 分数修改vs分数掩码

我们将暂时离开主题，描述两个关键概念，这些概念对于理解如何获得FlexAttention的最大性能优势非常重要。flex_attention的完整API如下：

```python
flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    block_mask: Optional[torch.nn.attention.flex_attention.BlockMask] = None,
    scale: Optional[float] = None,
)
```

你可能会好奇为什么我们需要同时使用 `score_mod` 和 `block_mask`。

- 当你想在注意力权重矩阵中修改分数值时，应该使用 `score_mod` 函数。
- 当你想在注意力权重矩阵中掩码分数值时，应该使用 `mask_mod` 函数，这些分数值独立于分数值本身，仅依赖于位置信息。

注意：任何 `block_mask` 也可以用 `score_mod` 表示，但kernel的性能将不是最优的。

### 让我们通过因果注意力来突出差异。

使用score_mod的实现：

```python
def causal_bias(score, b, h, q_idx, kv_idx):
    return torch.where(q_idx >= kv_idx, score, -float("inf"))
```
每当你编写一个 `score_mod` 函数，该函数对某些元素传递原始分数，而对其他元素设置为 -inf 时，你应该可能使用 `mask_mod`。

使用 `mask_mod` 的实现：

```python
def casual_mask(b,h,q_idx, kv_idx):
    return q_idx >= kv_idx
```

正如你所见，它们看起来非常相似，都返回标量张量。关键的区别在于：

- `mask_mods` 返回布尔张量，其中 `True` 表示应该计算该分数，而 `False` 表示我们想要掩码该分数。
- `mask_mods` 不接受 `score` 参数，因为它们在计算过程中不允许依赖实际值。

### 当我同时使用 score_mod 和 mask_mod 时会发生什么？

score_mod 函数将应用于每个未被掩码的元素。

### 我有一个 mask mod 函数，如何创建一个 BlockMask？

问得好，读者！除了 flex_attention，我们还提供了一个主要的 API。

```python
create_block_mask(
    mask_mod (Callable): mask_mod function.
    B (int): Batch size.
    H (int): Number of heads.
    Q_LEN (int): Sequence length of query.
    KV_LEN (int): Sequence length of key/value.
    device (str): Device to run the mask creation on.
    KV_BLOCK_SIZE (int): Block size of block mask for each query.
    Q_BLOCK_SIZE (int): Block size of block mask for each key/value.
    _compile (bool): Whether to compile the mask creation.
)
```

因此，对于上述示例，调用flex_attention的最优性能方式是：

```python
causal_block_mask = create_block_mask(causal_mask, B, H, M, N)
flex_attention(query, key, value, block_mask = causal_block_mask)
```

B,H,Q_LEN,KV_LEN 分别是 batch_size、num_heads、query_sequence_length 和 key_sequence_length。

### 为什么两者都有？

纯粹是为了性能。因果掩码实际上非常稀疏。只有注意力分数的下三角部分是重要的。如果不生成BlockMask，我们将需要做两倍的工作！下面我们将比较这两种实现的性能差异。

## 分数修改示例
让我们探索可以使用FlexAttention API的各种分数修改示例。

图例：我们将打印这些score_mod + mask_fns的稀疏性表示。

任何块的缺失意味着它被完全掩码，实际上不需要计算最终的注意力输出
- ██ 这个块计算所有查询和键token之间的完全注意力
- ░░ 这个块部分掩码，一些查询token关注一些键token，但一些被掩码为-inf

### 全注意力

应用一个“无操作”的分数修改。保持注意力分数不变。

```python
def noop(score, b, h, q_idx, kv_idx):
    return score

test_mask(noop, print_mask=True)
```

执行后的输出为：

```python
Results for noop:
+---------------+----------------+-------------------+----------------+-------------------+
| Operation     |   FW Time (ms) |   FW FLOPS (TF/s) |   BW Time (ms) |   BW FLOPS (TF/s) |
+===============+================+===================+================+===================+
| causal FA2    |        14.6478 |            150.13 |        41.1986 |            133.44 |
+---------------+----------------+-------------------+----------------+-------------------+
| F.sdpa + mask |        58.8032 |             74.79 |       125.07   |             87.91 |
+---------------+----------------+-------------------+----------------+-------------------+
| flexattention |        27.3449 |            160.84 |        94.4015 |            116.47 |
+---------------+----------------+-------------------+----------------+-------------------+

Block Mask:
None
```

### 标准因果掩码

标准因果掩码是自回归语言模型中的关键技术，确保每个token只能关注序列中自身及其之前的token。块稀疏表示展示了这种掩码的下三角性质。

有关这些实现的更多详细信息，请参阅上面的《分数修改vs分数掩码》

```python
def causal_bias(score, b, h, q_idx, kv_idx):
    return torch.where(q_idx >= kv_idx, score, -float("inf"))

test_mask(score_mod=causal_bias)

def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

test_mask(mask_mod=causal_mask)
```

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-2/726eab1c-14c4-42ac-b627-373c2f9e2326.png)

### 滑动窗口注意力

Mistral 论文中有一个非常好的图示描述了这种偏置。本质上，你定义一个固定大小的“滑动窗口”，在自回归解码中，你只允许 `torch.abs(q_tokens - kv_tokens) < SLIDING_WINDOW` 的 token 相互关注。通常，这也会与因果注意力结合使用。我们将通过一个很好的模式来实现这一点，即掩码组合。通常，掩码可以概念上分为几个部分，然后组合在一起。

我们将编写两个掩码函数，一个用于执行 `因果掩码`，另一个用于执行 `窗口注意力`，并将它们组合在一起以生成最终的掩码函数。正如我们之前所知，掩码函数返回布尔值，其中 `True` 表示该元素应参与注意力计算。

```python
SLIDING_WINDOW = 1024


def sliding_window_causal_mask(b, h, q_idx, kv_idx):
    causal_mask = q_idx >= kv_idx
    windowed_mask = (
        q_idx - kv_idx <= SLIDING_WINDOW
    )  # We dont need to check the right side of the sliding window since we are applying the causal mask

    return causal_mask & windowed_mask

test_mask(mask_mod=sliding_window_causal_mask)
```

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/ba95a3d9-7949-4f48-b8bb-8a743173077e.png)

### 前缀 LM（双向 + 因果）

T5 架构的论文（https://paperswithcode.com/method/t5）描述了一种执行前缀注意力的注意力变体。其中，一定数量的 `前缀` token允许完全参与，然后所有后续token执行因果注意力。我们再次组合两个掩码函数来实现这一点，一个用于因果掩码，另一个基于前缀长度。

```python
PREFIX_LENGTH = 2048

def prefix_lm_causal_mask(b, h, q_idx, kv_idx):
    prefix_mask = kv_idx <= PREFIX_LENGTH
    causal_mask = q_idx >= kv_idx
    return prefix_mask | causal_mask

test_mask(mask_mod=prefix_lm_causal_mask)
```

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-1/1211c505-855e-4dde-97e1-b80e80b760dc.png)

### 文档掩码

想象一下，我们有多个不同长度的文档。我们希望掩码掉文档之间的注意力，但允许同一文档内的token之间的注意力。我们可以通过使用一个document_id张量来实现这一点，该张量给出了每个token所属的文档。然后，我们可以掩码掉所有document_id[q_idx]与document_id[kv_idx]不同的注意力分数。

注意：只有当`score_mod`改变时，我们才需要编译一个新的kernel（它会使用torch.compile基础设施自动检测到这一点）。这个示例代码是通过缓存BlockMask实现的，但一般来说，改变BlockMask不需要重新编译。也就是说，对于文档掩码，我们只需要在文档长度改变时计算一个新的BlockMask，而不是一个新的kernel。

```python
document_id = torch.zeros(32768, dtype=torch.int, device="cuda")
document_id[:4096] = 0
document_id[4096:8192] = 1
for i in range(8192, 32768, 8192):
    document_id[i : i + 8192] = i // 8192 + 1

def document_causal_mask(b, h, q_idx, kv_idx):
    causal_mask = q_idx >= kv_idx
    document_mask = document_id[q_idx] == document_id[kv_idx]
    return causal_mask & document_mask

test_mask(mask_mod=document_causal_mask, S=32768)
```

我在4090上跑会oom，这里把长度改小一点：

```python
document_id = torch.zeros(8192, dtype=torch.int, device="cuda")
document_id[:4096] = 0
document_id[4096:8192] = 1
# for i in range(8192, 32768, 8192):
#     document_id[i : i + 8192] = i // 8192 + 1

def document_causal_mask(b, h, q_idx, kv_idx):
    causal_mask = q_idx >= kv_idx
    document_mask = document_id[q_idx] == document_id[kv_idx]
    return causal_mask & document_mask

test_mask(mask_mod=document_causal_mask, S=8192)
```

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-2/702dc016-23ab-45b1-b587-04f120d8a6a6.png)

### 独立自注意力掩码

在这种情况下，想象我们有一个大小为 (H x W) 的二维图像，被展平成一个token序列。我们只想关注8个`像素`内的token，但从二维角度来看。

我们可以通过首先将一维位置转换为二维坐标来实现这个mask_mod。然后，我们可以简单地检查两个坐标的距离是否在窗口内。

更多细节请查看论文，Stand-Alone Self-Attention in Vision Models(https://arxiv.org/abs/1906.05909)

```python
H = 128
W = 128
WINDOW = 8

def get_x_y(idx):
    return idx // W, idx % W

def sasa_mask(b, h, q_idx, kv_idx):
    q_x, q_y = get_x_y(q_idx)
    kv_x, kv_y = get_x_y(kv_idx)
    horizontal_mask = (q_x - kv_x).abs() <= WINDOW
    vertical_mask = (q_y - kv_y).abs() <= WINDOW
    return horizontal_mask & vertical_mask

test_mask(mask_mod=sasa_mask)
```

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-1/44b5dec7-a666-42df-926e-0ec9e7f219ef.png)


### NATTEN 掩码

考虑一个大小为 (H x W) 的二维图像，被展平成一个token序列。查询关注键在一个固定kernel区域 (K_H x K_W) 内，尽可能以查询为中心，同时保持在画布内并始终包括查询。

这与SASA类似，但有额外的处理来保持kernel在画布内，确保所有查询关注固定数量的键。键将其位置与kernel中心进行比较，而不是查询。kernel中心试图跟随查询位置，但被限制在画布边缘保持固定距离（其半长度）。

更多信息请参见NATTEN仓库(https://github.com/SHI-Labs/NATTEN)。
> 注意：更完整的NATTEN实现将包括对kernel膨胀的支持。NATTEN未融合的kernel还具有诸如能够交叉关注寄存器token等功能。这种能力可以在Flex Attention中表达，但这里没有尝试。

```python
H = 128
W = 128
K_H = 7
K_W = 7

def get_x_y(idx):
    return idx // W, idx % W

def natten_mask(
    b,
    h,
    q_idx,
    kv_idx,
):
    q_x, q_y = get_x_y(q_idx)
    kv_x, kv_y = get_x_y(kv_idx)
    # kernel nominally attempts to center itself on the query, but kernel center
    # is clamped to a fixed distance (kernel half-length) from the canvas edge
    kernel_x = q_x.clamp(K_W // 2, (W - 1) - K_W // 2)
    kernel_y = q_y.clamp(K_H // 2, (H - 1) - K_H // 2)
    hori_mask = (kernel_x - kv_x).abs() <= K_W // 2
    vert_mask = (kernel_y - kv_y).abs() <= K_H // 2
    return hori_mask & vert_mask

test_mask(mask_mod=natten_mask)
```

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-1/37c61482-1108-472e-8dc4-01b6d32d3886.png)

### Alibi 偏置

Alibi 注意力偏置在 Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation(https://arxiv.org/abs/2108.12409) 中变得流行，并声称在推理时具有长度外推的有益特性。"ALiBi 不会将位置嵌入添加到词嵌入中；相反，它通过与它们距离成比例的惩罚来偏置查询-键注意力分数。"

我们将以两种方式实现这一点，以突出一个新的功能，即在分数修改函数中利用其他张量的能力。尽管函数签名不接受其他张量，但用户可以通过 `closure` 来实现这一点。在这里，我们利用了我们非常熟悉的因果掩码函数以及各个头的偏置。

```python
# Alibi Bias
def generate_alibi_bias():
    alibi_bias = []
    for h in range(H):
        alibi_bias.append(-((h + 1) * 8.0 / H))
    alibi_bias = torch.tensor(alibi_bias, device="cuda")
    alibi_bias = torch.exp2(alibi_bias)
    return alibi_bias


alibi_bias = generate_alibi_bias()


# In this case we are going to use a mask_mod and a score_mod
def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def alibi_and_causal_closure(score, b, h, q_idx, kv_idx):
    bias = alibi_bias[h] * (q_idx - kv_idx)
    return score + bias


def alibi_and_causal_functional(score, b, h, q_idx, kv_idx):
    scale = torch.exp2(-((h + 1) * 8.0 / H))
    bias = (q_idx - kv_idx) * scale
    return score + bias


# Correctness check here is simple and only works with mask_fns and not actual score_mods

test_mask(
    alibi_and_causal_closure,
    mask_mod=causal_mask,
    skip_correctness=True,
    print_mask=False,
)
test_mask(
    alibi_and_causal_functional,
    mask_mod=causal_mask,
    skip_correctness=True,
    print_mask=False,
)
```

> 这里的H没有定义，我们写一个H=64来看下结果。另外需要把print_mask改成True才能看到mask长什么样。

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/e8f5f35d-fcff-4ddb-a4ad-2958d0aa3046.png)

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/fc2cb332-abab-4370-a824-9a7ba1c85a96.png)



### Tanh 软上限
我们也可以使用这个API实现tanh软上限。通过tanh进行logit软上限在Gemma 2中变得流行。

在这种情况下，有一些细微差别。特别是，PyTorch（和CUDA/Triton）中的标准`tanh`操作符会降低到一个数值上准确但（相对）较慢的SASS实现。参见https://godbolt.org/z/W8afevWv1了解SASS的样子。

因此，在这种情况下，我们希望将`tanh`降低到近似tanh实现。我们可以通过在PyTorch中注册一个自定义操作符，然后进行Inductor降低来实现这一点。

```python
def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

# Tanh Soft-Capping
@torch.library.custom_op("approx::tanh", mutates_args=())
def tanh_approx(inp: torch.Tensor) -> torch.Tensor:
    return torch.tanh(inp)


@tanh_approx.register_fake
def _(inp: torch.Tensor) -> torch.Tensor:
    return torch.tanh(inp)


from torch._inductor.lowering import make_pointwise, register_lowering

# Some internal torch.compile details
from torch._inductor.virtualized import ops

def tanh_approx_lowering(inp):
    fn = partial(ops.inline_asm_elementwise, asm="tanh.approx.f32 0,1;")
    return make_pointwise(fn)(inp)

register_lowering(torch.ops.approx.tanh)(tanh_approx_lowering)

class TanhApprox(torch.autograd.Function):
    @staticmethod
    def forward(x):
        return torch.ops.approx.tanh(x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        (x,) = inputs
        result = output
        ctx.save_for_backward(result)

    @staticmethod
    def backward(ctx, grad_output):
        (result,) = ctx.saved_tensors
        return grad_output * (1 - result * result)

tanh_approx = TanhApprox.apply

def tanh_soft_cap(score, b, h, q_idx, kv_idx):
    score = score / 2
    score = tanh_approx(score)
    return score * 2

# The baseline (xformers) does not have a way to generate tanh-softcapping so we skip correctness checks
test_mask(tanh_soft_cap, mask_mod=causal_mask, skip_correctness=True)
```

> 代码里面的asm代码有错误，这个例子无法运行。报错信息如下：

```shell
ptxas /tmp/tmpmehxr5i1.ptx, line 3972; error   : Arguments mismatch for instruction 'tanh'
ptxas /tmp/tmpmehxr5i1.ptx, line 3977; error   : Arguments mismatch for instruction 'tanh'
ptxas /tmp/tmpmehxr5i1.ptx, line 3982; error   : Arguments mismatch for instruction 'tanh'
ptxas /tmp/tmpmehxr5i1.ptx, line 3987; error   : Arguments mismatch for instruction 'tanh'
ptxas /tmp/tmpmehxr5i1.ptx, line 3992; error   : Arguments mismatch for instruction 'tanh'
ptxas /tmp/tmpmehxr5i1.ptx, line 3997; error   : Arguments mismatch for instruction 'tanh'
ptxas /tmp/tmpmehxr5i1.ptx, line 4002; error   : Arguments mismatch for instruction 'tanh'
ptxas /tmp/tmpmehxr5i1.ptx, line 4007; error   : Arguments mismatch for instruction 'tanh'
ptxas /tmp/tmpmehxr5i1.ptx, line 4012; error   : Arguments mismatch for instruction 'tanh'
ptxas fatal   : Ptx assembly aborted due to errors

```

### 嵌套不规则张量

嵌套张量是一种张量子类，用于高效地表示和计算不规则数据。可以使用FlexAttention处理这种数据，以高效地对不同长度的序列批次执行因果注意力。

在底层，NJT(嵌套不规则张量)将其不规则数据存储为连续数据 `[[sequence_0], [sequence_1], ..., [Sequence_B]], sum(*),..`

```python
# 设置随机种子以确保结果可重复
random.seed(0)
torch.manual_seed(0)

# 定义批次大小、头数和维度
batch_size = 16
n_heads = 16
D = 64

# 准备QKV值，使其可以计算梯度
def prepare_qkv_values(tensor):
    return tensor._values.detach().requires_grad_()

# 构建序列索引表
def build_seq_idx(tensor: torch.Tensor):
    offsets = tensor.offsets()
    total_length = tensor.offsets()[-1].item()
    # 创建从0到total_length的范围张量
    range_tensor = torch.arange(total_length, device="cuda", dtype=torch.int32)

    # 使用searchsorted查找每个位置的索引
    seq_idx = torch.searchsorted(offsets, range_tensor, right=True) - 1

    return seq_idx

# 创建NJT包装器，将密集掩码函数转换为NJT掩码函数
def create_njt_wrapper(orig_mask_mod, offsets, seq_idx):
    """通用包装器，将密集掩码函数转换为NJT掩码函数"""

    def njt_score_mod(b, h, q_idx, kv_idx):
        q_nested = q_idx - offsets[seq_idx[q_idx]]
        kv_nested = kv_idx - offsets[seq_idx[kv_idx]]
        is_same_sequence = seq_idx[q_idx] == seq_idx[kv_idx]
        return orig_mask_mod(b, h, q_nested, kv_nested) & is_same_sequence

    return njt_score_mod

# 密集得分掩码函数
def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx
    # return torch.where(q_idx >= kv_idx, score, -float("inf"))

# 当前限制：总序列长度必须能被128整除
sentence_lengths = [random.randint(1, 1024) for _ in range(batch_size - 1)]
total = sum(sentence_lengths)
sentence_lengths.append(128 - total % 128)
total = sum(sentence_lengths)

# 创建不规则张量
ragged_tensors = [torch.randn(l, n_heads, D, device="cuda") for l in sentence_lengths]
query = torch.nested.nested_tensor(
    ragged_tensors, layout=torch.jagged, requires_grad=True
)
key = torch.nested.nested_tensor(
    ragged_tensors, layout=torch.jagged, requires_grad=True
)
value = torch.nested.nested_tensor(
    ragged_tensors, layout=torch.jagged, requires_grad=True
)

# 构建seq_idx查找表
offsets = query.offsets()
seq_idx = build_seq_idx(query)

# 创建NJT因果得分掩码函数
causal_score_mod_njt = create_njt_wrapper(causal_mask, offsets, seq_idx)

# 准备QKV值
query_values = prepare_qkv_values(query)
key_values = prepare_qkv_values(key)
value_values = prepare_qkv_values(value)

# 创建块掩码
block_mask = create_block_mask_cached(
    causal_score_mod_njt, 1, 1, total, total, device=query_values.device
)
# 使用FlexAttention计算输出
out_flex = flex_attention(
    query_values.view(1, -1, n_heads, D).transpose(1, 2),
    key_values.view(1, -1, n_heads, D).transpose(1, 2),
    value_values.view(1, -1, n_heads, D).transpose(1, 2),
    block_mask=block_mask,
)
# 使用Scaled Dot-Product Attention计算输出
out_sdpa = F.scaled_dot_product_attention(
    query.transpose(1, 2),
    key.transpose(1, 2),
    value.transpose(1, 2),
    is_causal=True,
)

# 存储输出结果
sdpa_outs = []
flex_outs = []

# 创建梯度输出
gradOut = torch.randn_like(out_sdpa)

# 计算并存储SDPA的输出和梯度
sdpa_outs.append(out_sdpa)
out_sdpa.backward(gradOut)
sdpa_outs += [query.grad, key.grad, value.grad]

# 计算并存储FlexAttention的输出和梯度
flex_outs.append(out_flex)
out_flex.backward(gradOut._values.unsqueeze(0))
flex_outs += [query_values.grad, key_values.grad, value_values.grad]

# 比较两种方法的输出和梯度
for flex, sdpa in zip(flex_outs, sdpa_outs):
    flex = flex.squeeze(0)
    torch.testing.assert_close(flex, sdpa._values, atol=1e-2, rtol=1e-2)

# 打印正确性检查结果
print("Correctness check passed ✅")
print(block_mask)
```

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-2/8ca85ee2-b2fc-4ce4-80b3-b7f84f5b1917.png)

### Flamingo Cross Attention

🦩 Flamingo 论文（https://arxiv.org/pdf/2204.14198）介绍了一种“视觉语言模型（VLM）家族，它们以交错的视觉数据和文本作为输入，并生成自由形式的文本作为输出。”

它利用 `VisionCrossAttentionMask` 来确保文本只关注相关的图像。TorchTune 对这种掩码类型有很好的描述：VisionCrossAttentionMask（https://github.com/pytorch/torchtune/blob/bbc48e089b072c7cbaea175bc70501b2193ba482/torchtune/modules/transforms/_transforms.py#L22-L43）

这种注意力机制确保文本序列完全关注前面的图像，而不关注其他未来的或不相关的图像。

```python
Example:
    >>> text = "<img1><img2>These are two dogs. <img3>This is a cat."
    >>> image_token_id = 1
    >>> tokens = [1, 1, 9673, 527, 1403, 12875, 13, 1, 1115, 374, 264, 8415]
    >>> transform = VisionCrossAttentionMask(tile_size=400, patch_size=40, image_token_id=1)
    >>> intervals = transform._get_image_attention_intervals(tokens)
    >>> print(intervals)
    [[0, 7], [1, 7], [7, 12]]
```

在上面的例子中，我们将生成一个12 x sum(image_tokens_1 + image_tokens_2 + image_tokens_3)的掩码

假设image_tokens的大小为3

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/b82c1996-a282-42a2-9b3e-11d90969caf1.png)

```python
# Given information
num_tokens = 12
num_images = 3
image_token_length = 3
num_image_tokens = num_images * image_token_length
intervals = [[0, 7], [1, 7], [7, 12]]
# This is only needed if your images have different number of tokens per image
# If they are all the same number of tokens you can use image_idx = kv_idx // image_token_length
image_boundaries = [image_token_length * i for i in range(num_images)]
image_boundaries = (
    [0] * image_token_length + [1] * image_token_length + [2] * image_token_length
)

image_boundaries = torch.tensor(image_boundaries, dtype=torch.long, device="cuda")
intervals = torch.tensor(intervals, dtype=torch.long, device="cuda")


def vision_x_attention_mask(b, h, q_idx, kv_idx):
    image_idx = image_boundaries[kv_idx]
    interval = intervals[image_idx]
    return (q_idx >= interval[0]) & (q_idx < interval[1])


mask = create_mask(vision_x_attention_mask, 1, 1, num_tokens, num_image_tokens, "cuda")

print(mask)
```


# FlexAttention是如何实现的

FlexAttention是通过PyTorch编译器来实现的，通过inductor后端生成FlexAttention的各种变体对应的Triton代码。具体实现见：https://github.com/pytorch/pytorch/blob/ee09d066d35d7e17cf7e9479c0b8bfc70cffc264/torch/_inductor/kernel/flex_attention.py#L317 ，下面对 flex_attention 的核心入口简单浏览一下：

```python
# TODO: We probably also need a layout constraint?
@register_lowering(torch.ops.higher_order.flex_attention, type_promotion_kind=None)
def flex_attention(
    query,
    key,
    value,
    subgraph,
    block_mask,
    scale,
    score_mod_other_buffers,
    mask_mod_other_buffers,
):
    # 这行代码实际上就获取了我们在API应用中定义的score_mod和mask_mod之后真正要计算的Q,K,V
    (
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        q_num_blocks,
        q_indices,
        full_q_num_blocks,
        full_q_indices,
        SPARSE_KV_BLOCK_SIZE,
        SPARSE_Q_BLOCK_SIZE,
        mask_graph,
    ) = block_mask
    // 创建占位符输入列表，包含score、b、h、m、n五个占位符，类型分别为query的类型和int32
    placeholder_inps = [
        create_placeholder(name, dtype, query.get_device())
        for name, dtype in [
            ("score", query.get_dtype()),
            ("b", torch.int32),
            ("h", torch.int32),
            ("m", torch.int32),
            ("n", torch.int32),
        ]
    ]
    // 构建子图缓冲区，包含占位符输入和其他分数修改缓冲区
    subgraph_buffer = build_subgraph_buffer(
        placeholder_inps + list(score_mod_other_buffers), subgraph
    )
    // 创建掩码图的占位符输入列表，包含b、h、m、n四个占位符，类型均为int32
    mask_graph_placeholder_inps = [
        create_placeholder(name, dtype, query.get_device())
        for name, dtype in [
            ("b", torch.int32),
            ("h", torch.int32),
            ("m", torch.int32),
            ("n", torch.int32),
        ]
    ]
    // 构建掩码图缓冲区，包含掩码图的占位符输入和其他掩码修改缓冲区
    mask_graph_buffer = build_subgraph_buffer(
        mask_graph_placeholder_inps + list(mask_mod_other_buffers), mask_graph
    )
    // 如果使用Flex解码，则返回创建的Flex解码内核
    if _use_flex_decoding(query):
        return create_flex_decoding_kernel(
            query,
            key,
            value,
            block_mask,
            scale,
            subgraph_buffer,
            mask_graph_buffer,
            score_mod_other_buffers,
            mask_mod_other_buffers,
        )
    // 对所有缓冲区进行realize操作，确保它们被实例化
    for buf in [
        query,
        key,
        value,
        kv_num_blocks,
        kv_indices,
        q_num_blocks,
        q_indices,
        full_kv_num_blocks,
        full_kv_indices,
        full_q_num_blocks,
        full_q_indices,
    ]:
        if buf is not None:
            buf.realize()

    // 创建布局对象，包含设备、数据类型、大小和步幅信息
    layout = FixedLayout(
        query.get_device(),
        query.get_dtype(),
        query.get_size(),
        query.get_stride(),
    )
    // 计算logsumexp的形状，即query的形状去掉最后一个维度
    logsumexp_shape = query.get_size()[:-1]  # [B, H, M]
    // 创建logsumexp张量，类型为float32，设备与query相同
    logsumexp = empty_strided(
        logsumexp_shape,
        None,
        dtype=torch.float32,  # The logsumexp is always stored in fp32 regardless of the input dtype
        device=query.get_device(),
    )
    // 判断是否存在完整块，如果full_kv_num_blocks为None，则不存在完整块
    has_full_blocks = full_kv_num_blocks is not None
    if full_kv_num_blocks is None:
        full_kv_num_blocks, full_kv_indices = (
            empty(0, device=query.get_device()) for _ in range(2)
        )
    // 初始化选择列表和配置列表
    choices: List[Any] = []
    configs: List[Tuple[int, int, int, int]] = []
    // 添加默认配置
    configs.append(_get_default_config_fwd(query))
    // 如果启用了最大自动调优，则添加其他配置
    if config.max_autotune:
        configs += [
            (128, 64, 4, 3),
            (128, 128, 4, 3),
            (128, 128, 8, 2),
            (64, 128, 4, 3),
            (64, 64, 4, 3),
        ]

    // 遍历所有配置，如果块大小不匹配或配置为2阶段，则跳过
    for BLOCK_M, BLOCK_N, num_warps, num_stages in configs:
        if SPARSE_KV_BLOCK_SIZE % BLOCK_N != 0 or SPARSE_Q_BLOCK_SIZE % BLOCK_M != 0:
            continue
        if num_stages == 2:
            continue

        // 将当前配置添加到选择列表中
        flex_attention_template.maybe_append_choice(
            choices=choices,
            input_nodes=[
                query,
                key,
                value,
                logsumexp,
                kv_num_blocks,
                kv_indices,
                full_kv_num_blocks,
                full_kv_indices,
            ],
            layout=layout,
            subgraphs=[
                subgraph_buffer,
                mask_graph_buffer,
            ],
            mutated_inputs=[
                logsumexp,
            ],
            num_stages=num_stages,
            num_warps=num_warps,
            call_sizes=query.get_size(),
            OUTPUT_LOGSUMEXP=True,
            SM_SCALE=scale,
            BLOCK_DMODEL=query.get_size()[-1],
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
            SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
            ROWS_GUARANTEED_SAFE=False,
            PRESCALE_QK=False,
            HAS_FULL_BLOCKS=has_full_blocks,
        )
    // 创建用于自动调优的输入列表
    inputs_for_autotuning = (
        [
            query,
            key,
            value,
            logsumexp,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
        ]
        + list(score_mod_other_buffers)
        + list(mask_mod_other_buffers)
    )
    // 创建输入生成函数映射
    input_gen_fns = {
        4: create_num_blocks_fake_generator(full_kv_indices),
        5: create_indices_fake,
    }
    // 返回自动调优选择算法的结果和logsumexp
    return (
        autotune_select_algorithm(
            "flex_attention",
            choices,
            inputs_for_autotuning,
            layout,
            input_gen_fns=input_gen_fns,
        ),
        logsumexp,
    )

```


这个`flex_attention`函数的`block_mask`参数是通过上面API应用中提到的`create_block_mask`函数来创建的。然后这个函数接受查询（query）、键（key）、值（value）、子图（subgraph）、块掩码（block_mask）、缩放因子（scale）以及分数和掩码修改缓冲区作为输入。函数内部通过创建占位符输入、构建子图和掩码图缓冲区，并根据配置选择合适的kernel来实现 FlexAttention 计算。最终返回自动调优选择算法的结果和 logsumexp 张量。感兴趣的朋友也可以看下这里的triton kernel的具体实现。











