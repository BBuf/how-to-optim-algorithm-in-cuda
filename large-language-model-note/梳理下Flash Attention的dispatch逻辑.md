# 0x1. 前言

这篇文章来源是当运行下面的对HuggingFace Qwen2.5-7B-Instruct模型使用Flash Attention的代码时，使用Nsight System工具抓取的kernel trace会发现在prefill和decode阶段，Flash Attention调用了不同的kernel并且decoding的Flash Attention kernel使用了split_kv的实现。然后如果把下面代码中max_new_tokens改成64，我发现在Nsight System工具抓取的kernel trace中，decode阶段的Flash Attention kernel又变成了和prefill阶段一样的kernel，并没有使用split_kv的实现。这篇文章就尝试跟踪下Flash Attention的dispatch逻辑，弄清楚什么情况下decode阶段的Flash Attention kernel会使用split_kv的实现。

```python
# /opt/nvidia/nsight-systems/2024.5.1/bin/nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node -o hf_qwen2.5_7b_flash_attn python3 debug.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import nvtx
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "/mnt/bbuf/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

prompt = "帮我计划一次去北京的旅行，我想明年春天出发，大概五天的行程。"

model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

for i in range(1):
    with nvtx.annotate(f"step={i}", color="blue"):
        
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

![](https://files.mdnice.com/user/59/4be78d18-a75b-4e8e-ab42-7a09682526b3.png)

这张图是max_new_tokens=512时，prefill和decode阶段的Flash Attention kernel的trace。红色框表示prefill阶段调用的Flash Attention kernel，绿色框表示decode阶段调用的Flash Attention kernel。可以看到prefill阶段调用了`flash_fwd_kernel`，decode阶段调用了`flash_fwd_splitkv_kernel`和`flash_fwd_splitkv_combine_kernel`两种kernel。

![](https://files.mdnice.com/user/59/b8a4c1e1-28fe-43f0-8c2e-b9130a473a47.png)

这张图是max_new_tokens=64时，prefill和decode阶段的Flash Attention kernel的trace。可以看到两个阶段都调用了同一个`flash_fwd_kernel`。

为什么产生了这种差别，什么情况下decode阶段的Flash Attention kernel会使用split_kv的实现？我们需要深入看一下Flash Attention的相关Dispatch逻辑。

# 0x2. Qwen2是如何访问Flash Attention API的

下面是 HuggingFace  Qwen2 模型 Qwen2FlashAttention2 模块的实现，我们可以从这个代码中看到 flash attention 的 API 是如何被调用的。这里调用的 `_flash_attention_forward` 实际上又是调用了 flash-attention 库(https://github.com/Dao-AILab/flash-attention)中的 `flash_attn_varlen_func` api，这个api是flash attention库中用来处理Attention前向计算的核心函数，并且可以从名字看出来这个api还支持可变长的多个序列的Attention计算。

```python
class Qwen2FlashAttention2(Qwen2Attention):
    # ...
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入隐藏状态
        attention_mask: Optional[torch.Tensor] = None,  # 注意力mask
        position_ids: Optional[torch.LongTensor] = None,  # 位置编码id
        past_key_value: Optional[Cache] = None,  # KV缓存
        output_attentions: bool = False,  # 是否输出注意力权重
        use_cache: bool = False,  # 是否使用KV缓存
        cache_position: Optional[torch.LongTensor] = None,  # 缓存位置
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # 位置编码,在v4.46中将成为必需
    ):
        # 获取输入维度
        bsz, q_len, _ = hidden_states.size()

        # QKV投影
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # 重塑维度以适应多头注意力
        query_states = query_states.view(-1, self.num_heads*self.head_dim)
        key_states = key_states.view(-1, self.num_key_value_heads*self.head_dim)
        value_states = value_states.view(-1, self.num_key_value_heads*self.head_dim)
        
        # 应用旋转位置编码(RoPE)
        query_states, key_states = self.rotary_emb(position_ids, query_states, key_states)

        # 重塑维度为[batch_size, num_heads, seq_len, head_dim]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # 处理KV缓存
        if past_key_value is not None:
            cache_kwargs = {"cache_position": cache_position}  # RoPE模型特有的参数
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # 如果KV头数小于注意力头数,需要重复KV
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # 处理数据类型转换
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"输入隐藏状态似乎被静默转换为float32,这可能与embedding或layer norm层被上采样到float32有关。"
                f"我们会将输入转回{target_dtype}。"
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # 重塑维度以适应Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # 处理滑动窗口注意力
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
        else:
            sliding_window = None

        # 调用Flash Attention前向传播
        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=sliding_window,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        # 重塑输出并应用输出投影
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
```

这里的代码省略掉了类的相关初始化，在forward函数中涉及到rope，kv cache更新，reshape输入以适应Flash Attention的输入格式，以及调用Flash Attention，以及应用输出投影等等Attention计算的细节。

# 0x3. Flash Attention单独的调用例子

这里来关注一下使用 `flash_attn_varlen_func` 这个 api 的单独例子。由于它可以支持多个不同的序列，所以这里我们用2个序列来调用一下，我写了一个测试，脚本如下：

```python
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

# 任意传入一个softmax_scale
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
```

这个测试是可以通过的，相信通过上面2个对上层接口调用的例子可以让我们对Flash Attention的接口调用有比较清晰的认识。下面我们可以关注一下Flash Attention这个借口的实现，我们不需要深入到cuda实现中，只需要把握一下整体的调用逻辑，搞清楚文章开头抛出的问题即可。

# 0x4. Flash Attention的dispatch逻辑


