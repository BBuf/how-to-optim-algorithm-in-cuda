import torch
import math

from flash_attn import flash_attn_varlen_func

# 朴素实现的缩放点积注意力函数
# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    # 调整输入张量的维度顺序
    query = query.transpose(0, 1)  # [nheads, seqlen, headdim]
    key = key.transpose(0, 1)      # [nheads, seqlen, headdim]
    value = value.transpose(0, 1)  # [nheads, seqlen, headdim]
    
    L, S = query.size(1), key.size(1)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias = attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    
    # 调整注意力计算以适应多头
    attn_weight = torch.matmul(query, key.transpose(-2, -1)) * scale_factor  # [nheads, L, S]
    attn_weight = attn_weight + attn_bias.unsqueeze(0)  # 广播 attn_bias 到所有头
    attn_weight = torch.softmax(attn_weight, dim=-1)
    
    if dropout_p > 0.0:
        attn_weight = torch.nn.functional.dropout(attn_weight, p=dropout_p, training=True)
    
    output = torch.matmul(attn_weight, value)  # [nheads, L, headdim]
    return output.transpose(0, 1)  # 返回 [L, nheads, headdim]


# 设置随机种子以确保结果可复现
torch.manual_seed(0)

# 参数设置
batch_size = 2
seq_lengths = [128, 256]  # 两个序列的长度
nheads = 16
headdim = 32
dropout_p = 0.0
causal = True  # 是否使用因果性掩码
scale = None   # 缩放因子，默认为 1 / sqrt(headdim)

# 为每个序列生成随机的 q, k, v 张量
qs = []
ks = []
vs = []
for seqlen in seq_lengths:
    q = torch.randn(seqlen, nheads, headdim, requires_grad=True, dtype=torch.bfloat16, device="cuda")  # (L, nheads, headdim)
    k = torch.randn(seqlen, nheads, headdim, requires_grad=True, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(seqlen, nheads, headdim, requires_grad=True, dtype=torch.bfloat16, device="cuda")
    qs.append(q)
    ks.append(k)
    vs.append(v)

# 将所有序列的 q, k, v 拼接起来
q_total = torch.cat(qs, dim=0)  # (total_q, nheads, headdim)
k_total = torch.cat(ks, dim=0)
v_total = torch.cat(vs, dim=0)

# 计算累积序列长度，用于索引
cu_seqlens_q = torch.zeros(batch_size + 1, dtype=torch.int32, device="cuda")
cu_seqlens_q[1:] = torch.cumsum(torch.tensor(seq_lengths, dtype=torch.int32), dim=0)
cu_seqlens_k = cu_seqlens_q.clone()

print('cu_seqlens_q: ', cu_seqlens_q)

# 最大序列长度
max_seqlen_q = max(seq_lengths)
max_seqlen_k = max(seq_lengths)

# 这里我们使用默认的 softmax_scale，即 1 / sqrt(headdim)
softmax_scale = 0.2

# 调用 flash_attn_varlen_func 函数
out_flash = flash_attn_varlen_func(
    q_total,
    k_total,
    v_total,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=dropout_p,
    softmax_scale=softmax_scale,
    causal=causal,
)

# 使用朴素实现对每个序列进行计算，并将输出拼接起来
outputs_naive = []
for i in range(batch_size):
    q = qs[i]  # (L_i, nheads, headdim)
    k = ks[i]
    v = vs[i]
    out = scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=dropout_p,
        is_causal=causal,
        scale=softmax_scale
    )  # 输出形状为 (L_i, nheads, headdim)
    outputs_naive.append(out)

# 将朴素实现的输出拼接起来
out_naive = torch.cat(outputs_naive, dim=0)  # (total_q, nheads, headdim)



print('out_naive st: ', out_naive.flatten()[:10])
print('out_flash st: ', out_flash.flatten()[:10])
print('='*20)
print('out_naive en: ', out_naive.flatten()[-10:])
print('out_flash en: ', out_flash.flatten()[-10:])

# 比较两个实现的输出是否一致
assert torch.allclose(out_flash, out_naive, atol=1e-2), "Outputs do not match!"

print("测试通过")
