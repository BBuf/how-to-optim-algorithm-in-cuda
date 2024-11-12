# 0x1. 前言

这篇文章来源是当运行下面的对HuggingFace Qwen2.5-7B-Instruct模型使用Flash Attention的代码时，使用Nsight System工具抓取的kernel trace会发现在prefill和decode阶段，Flash Attention调用了不同的kernel并且decoding的Flash Attention kernel使用了split_kv的实现。然后如果把下面代码中max_new_tokens改成64，我发现在Nsight System工具抓取的kernel trace中，decode阶段的Flash Attention kernel又变成了和prefill阶段一样的kernel，并没有使用split_kv的实现。这篇文章就尝试跟踪下Flash Attention的dispatch逻辑，弄清楚什么情况下decode阶段的Flash Attention kernel会使用split_kv的实现(split_kv的实现也被叫作Flash Decoding，专用大模型的Decoding阶段)。

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

# 0x4. flash_attn_interface.py中的上层接口

flash-attention 库中使用 cuda 实现了Flash Attention的计算，然后通过 Torch Binding 将`varlen_fwd`这个接口暴露给Python，而`flash_attn_varlen_func`则是对`varlen_fwd`的进一步封装，我们可以在 https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py 中查看到`flash_attn_varlen_func`这个接口的实现。去掉了反向相关的逻辑，如下所示：

```python
def _flash_attn_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor, 
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,  # Q序列的累积长度
    cu_seqlens_k: torch.Tensor,  # K序列的累积长度
    max_seqlen_q: int,          # Q序列的最大长度
    max_seqlen_k: int,          # K序列的最大长度
    dropout_p: float,           # dropout概率
    softmax_scale: float,       # softmax缩放因子
    causal: bool,               # 是否使用因果掩码
    window_size_left: int = -1,  # 滑动窗口左侧大小
    window_size_right: int = -1, # 滑动窗口右侧大小
    softcap: float = 0.0,       # softmax的上限值
    alibi_slopes: Optional[torch.Tensor] = None,  # ALiBi位置编码的斜率
    return_softmax: bool = False,  # 是否返回softmax结果
    block_table: Optional[torch.Tensor] = None,  # 分块表
    leftpad_k: Optional[torch.Tensor] = None,    # K序列左侧填充
    seqused_k: Optional[torch.Tensor] = None,    # K序列使用的长度
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # 确保输入张量是连续的内存布局
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    
    # 调用CUDA实现的前向传播函数
    out, softmax_lse, S_dmask, rng_state = flash_attn_cuda.varlen_fwd(
        q, k, v,
        None,  # 原始掩码矩阵(未使用)
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
        leftpad_k,
        block_table,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        False,  # 未使用的参数
        causal,
        window_size_left,
        window_size_right,
        softcap,
        return_softmax,
        None,  # 随机数生成器状态(未使用)
    )
    return out, softmax_lse, S_dmask, rng_state

# FlashAttnVarlenQKVPackedFunc类实现了PyTorch的自动微分接口
class FlashAttnVarlenQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,  # 上下文对象,用于保存反向传播需要的信息
        qkv,  # 打包的QKV张量
        cu_seqlens,  # 累积序列长度
        max_seqlen,  # 最大序列长度
        dropout_p,   # dropout概率
        softmax_scale,  # softmax缩放因子
        causal,      # 是否使用因果掩码
        window_size,  # 滑动窗口大小
        softcap,     # softmax上限值
        alibi_slopes,  # ALiBi位置编码斜率
        deterministic,  # 是否确定性计算
        return_softmax,  # 是否返回softmax结果
    ):
        # 如果未指定缩放因子,使用默认的1/sqrt(head_dim)
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
            
        # 分离Q、K、V并detach,避免建立反向图
        q, k, v = qkv[:, 0].detach(), qkv[:, 1].detach(), qkv[:, 2].detach()
        
        # 获取原始head size
        head_size_og = q.size(2)
        
        # 如果head size不是8的倍数,进行padding
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
            
        # 调用前向计算函数    
        out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_varlen_forward(
            q, k, v,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
            block_table=None,
        )
        # 移除padding,恢复原始head size
        out = out_padded[..., :head_size_og]
        
        # 根据需要返回softmax结果
        return out if not return_softmax else (out, softmax_lse, S_dmask)

def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
):
    return FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        block_table,
    )
```

上面这段代码清晰展示了 `flash_attn_varlen_func` 这个接口的调用逻辑，接下来我们就可以去看一下`flash_attn_cuda.varlen_fwd`这个接口的具体dispatch逻辑了。

# 0x5. flash_attn_cuda.varlen_fwd的初步dispatch逻辑

首先来到这里：https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/flash_api.cpp#L1518 ，

```c++
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashAttention";
    m.def("fwd", &mha_fwd, "Forward pass");
    m.def("varlen_fwd", &mha_varlen_fwd, "Forward pass (variable length)");
    m.def("bwd", &mha_bwd, "Backward pass");
    m.def("varlen_bwd", &mha_varlen_bwd, "Backward pass (variable length)");
    m.def("fwd_kvcache", &mha_fwd_kvcache, "Forward pass, with KV-cache");
}
```

可以发现`flash_attn_cuda.varlen_fwd`接口对应了`mha_varlen_fwd`这个c++函数。从这里我们应该就可以看到flash attention forward的dispatch逻辑了。

```c++
std::vector<at::Tensor>
mha_varlen_fwd(at::Tensor &q,  // total_q x num_heads x head_size, total_q为每个batch中序列长度的总和
               const at::Tensor &k,  // total_k x num_heads_k x head_size, total_k为每个batch中序列长度的总和，如果有block_table则为num_blocks x page_block_size x num_heads_k x head_size
               const at::Tensor &v,  // total_k x num_heads_k x head_size, total_k为每个batch中序列长度的总和，如果有block_table则为num_blocks x page_block_size x num_heads_k x head_size
               c10::optional<at::Tensor> &out_, // total_q x num_heads x head_size, total_q为每个batch中序列长度的总和
               const at::Tensor &cu_seqlens_q,  // b+1
               const at::Tensor &cu_seqlens_k,  // b+1
               c10::optional<at::Tensor> &seqused_k, // b。如果提供了该参数，则每个batch元素只使用这么多个key
               c10::optional<const at::Tensor> &leftpad_k_, // batch_size
               c10::optional<at::Tensor> &block_table_, // batch_size x max_num_blocks_per_seq
               c10::optional<at::Tensor> &alibi_slopes_, // num_heads或b x num_heads
               int max_seqlen_q,
               const int max_seqlen_k,
               const float p_dropout,
               const float softmax_scale,
               const bool zero_tensors,
               bool is_causal,
               int window_size_left,
               int window_size_right,
               const float softcap,
               const bool return_softmax,
               c10::optional<at::Generator> gen_) {

    // 获取当前CUDA设备的属性
    auto dprops = at::cuda::getCurrentDeviceProperties();
    
    // 检查GPU架构版本
    // 判断是否为Ampere(SM8x)架构
    bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
    // 判断是否为Hopper(SM90)架构 
    bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
    
    // 检查GPU架构要求 - 目前只支持Ampere或更新的架构
    TORCH_CHECK(is_sm90 || is_sm8x, "FlashAttention only supports Ampere GPUs or newer.");
    
    // 检查输入数据类型
    auto q_dtype = q.dtype();
    // 只支持fp16和bf16数据类型
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
                
    // bf16只在Ampere及以上架构支持
    if (q_dtype == torch::kBFloat16) {
        TORCH_CHECK(is_sm90 || is_sm8x, "bfloat16 is only supported on Ampere GPUs or newer");
    }
    
    // 检查QKV的数据类型一致性
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    
    // 检查序列长度累加和的数据类型为int32
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");

    // 检查所有输入tensor是否在同一设备上
    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(cu_seqlens_q);
    CHECK_DEVICE(cu_seqlens_k);

    // 检查分块表相关参数
    at::Tensor block_table;
    const bool paged_KV = block_table_.has_value(); // 是否使用分页KV缓存
    if (paged_KV) {
        block_table = block_table_.value();
        CHECK_DEVICE(block_table); // 检查设备
        TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table必须是int32类型");
        TORCH_CHECK(block_table.stride(-1) == 1, "block_table最后一维必须连续");
    }

    // 检查QKV张量的内存布局
    TORCH_CHECK(q.stride(-1) == 1, "输入张量最后一维必须连续");
    TORCH_CHECK(k.stride(-1) == 1, "输入张量最后一维必须连续"); 
    TORCH_CHECK(v.stride(-1) == 1, "输入张量最后一维必须连续");
    CHECK_CONTIGUOUS(cu_seqlens_q); // 检查序列长度累加和是否连续
    CHECK_CONTIGUOUS(cu_seqlens_k);

    const auto sizes = q.sizes(); // 获取Q的形状

    // 获取基本参数
    const int batch_size = cu_seqlens_q.numel() - 1; // 批次大小
    int num_heads = sizes[1];  // Q的注意力头数
    const int head_size = sizes[2]; // 每个头的维度
    const int num_heads_k = paged_KV ? k.size(2) : k.size(1); // K的注意力头数

    // softcap和dropout不能同时使用
    if (softcap > 0.f) { TORCH_CHECK(p_dropout == 0.f, "Softcapping暂不支持dropout"); }

    // 分页KV缓存相关参数
    const int max_num_blocks_per_seq = !paged_KV ? 0 : block_table.size(1); // 每个序列最大块数
    const int num_blocks = !paged_KV ? 0 : k.size(0); // 总块数
    const int page_block_size = !paged_KV ? 1 : k.size(1); // 每块大小
    TORCH_CHECK(!paged_KV || page_block_size % 256 == 0, "分页KV缓存块大小必须是256的倍数");

    // 因果掩码和窗口大小相关处理
    if (max_seqlen_q == 1 && !alibi_slopes_.has_value()) { is_causal = false; }
    if (is_causal) { window_size_right = 0; }

    void *cu_seqlens_q_d = cu_seqlens_q.data_ptr();

    // 判断是否需要对Q进行重排
    // 满足以下条件时需要重排:
    // 1. Q序列长度为1(即解码阶段)
    // 2. Q的注意力头数大于K的注意力头数(即MQA/GQA场景) 
    // 3. 不使用滑动窗口(window_size_left和window_size_right都为-1)
    // 4. 不使用dropout
    // 5. head_size是8的倍数
    // 6. 不使用ALiBi位置编码
    const int seqlenq_ngroups_swapped = max_seqlen_q == 1 && num_heads > num_heads_k 
        && window_size_left < 0 && window_size_right < 0 
        && p_dropout == 0.f && head_size % 8 == 0 
        && !alibi_slopes_.has_value();
    
    // 计算每个K/V头对应多少个Q头
    const int ngroups = num_heads / num_heads_k;

    // 如果需要重排
    if (seqlenq_ngroups_swapped) {
        // 将Q的形状从(batch_size, 1, num_heads_k * ngroups, head_size)
        // 重排为(batch_size * ngroups, num_heads_k, head_size)
        // 这样可以让同一个K/V头对应的Q头在内存上连续,提高访问效率
        q = q.reshape({batch_size, num_heads_k, ngroups, head_size})
             .transpose(1, 2)
             .reshape({batch_size * ngroups, num_heads_k, head_size});
        
        // 更新相关参数
        max_seqlen_q = ngroups;  // Q序列长度变为ngroups 
        num_heads = num_heads_k;  // Q的头数变为K的头数
        cu_seqlens_q_d = nullptr;  // 不再需要Q的序列长度累加和
    }

    const int total_q = q.sizes()[0]; // Q的总token数

    // 检查输入参数的合法性
    // 1. batch_size必须为正数
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    // 2. head_size必须小于等于256,这是Flash Attention的限制
    TORCH_CHECK(head_size <= 256, "FlashAttention forward only supports head dimension at most 256");
    // 3. head_size必须是8的倍数,这是为了内存对齐和CUDA优化
    TORCH_CHECK(head_size % 8 == 0, "query, key, value, and out_ must have a head_size that is a multiple of 8");
    // 4. Q的head数必须是K/V的head数的整数倍,这是为了支持MQA/GQA
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    // 如果滑动窗口大小超过了K序列的最大长度,则设置为-1表示不使用滑动窗口
    if (window_size_left >= max_seqlen_k) { window_size_left = -1; }
    if (window_size_right >= max_seqlen_k) { window_size_right = -1; }

    // 检查Q张量的形状是否正确: [total_q, num_heads, head_size]
    CHECK_SHAPE(q, total_q, num_heads, head_size);
    
    // 根据是否使用分页KV缓存来检查K/V张量的形状
    if (!paged_KV) {
        // 不使用分页KV缓存时,K/V的形状应为[total_k, num_heads_k, head_size]
        const int total_k = k.size(0);
        CHECK_SHAPE(k, total_k, num_heads_k, head_size);
        CHECK_SHAPE(v, total_k, num_heads_k, head_size);
    } else {
        // 使用分页KV缓存时,K/V的形状应为[num_blocks, page_block_size, num_heads_k, head_size]
        // block_table的形状应为[batch_size, max_num_blocks_per_seq]
        CHECK_SHAPE(k, num_blocks, page_block_size, num_heads_k, head_size);
        CHECK_SHAPE(v, num_blocks, page_block_size, num_heads_k, head_size);
        CHECK_SHAPE(block_table, batch_size, max_num_blocks_per_seq);
    }

    // 检查序列长度累加和张量的形状,应为[batch_size + 1]
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
    
    // 如果提供了K序列使用长度的信息,检查其属性
    if (seqused_k.has_value()){
        auto seqused_k_ = seqused_k.value();
        // 数据类型必须是int32
        TORCH_CHECK(seqused_k_.dtype() == torch::kInt32, "seqused_k must have dtype int32");
        // 必须在CUDA设备上
        TORCH_CHECK(seqused_k_.is_cuda(), "seqused_k must be on CUDA device");
        // 必须是连续的内存布局
        TORCH_CHECK(seqused_k_.is_contiguous(), "seqused_k must be contiguous");
        // 形状必须是[batch_size]
        CHECK_SHAPE(seqused_k_, batch_size);
    }

    // 创建输出张量
    at::Tensor out;
    // 如果提供了输出张量
    if (out_.has_value()) {
        out = out_.value();
        // 检查输出张量的属性:
        // 1. 数据类型必须与输入相同
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        // 2. 必须在同一设备上
        CHECK_DEVICE(out);
        // 3. 最后一维必须是连续的
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        // 4. 形状必须正确
        CHECK_SHAPE(out, sizes[0], sizes[1], head_size);
        // 如果序列长度和组数需要交换
        if (seqlenq_ngroups_swapped) {
            // 重塑张量维度并转置,用于处理分组注意力
            out = out.reshape({batch_size, num_heads_k, ngroups, head_size})
                     .transpose(1, 2)
                     .reshape({batch_size * ngroups, num_heads_k, head_size});
        }
    } else {
        // 如果没有提供输出张量,创建一个与输入形状相同的空张量
        out = torch::empty_like(q);
    }

    // 定义一个lambda函数,用于将数字向上取整到m的倍数
    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    // 计算head_size的对齐值:
    // - 如果head_size <= 192,向上取整到32的倍数
    // - 否则设为256
    const int head_size_rounded = head_size <= 192 ? round_multiple(head_size, 32) : 256;
    // 将Q序列长度向上取整到128的倍数
    const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    // 将K序列长度向上取整到128的倍数
    const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

    // 设置CUDA设备,确保在正确的GPU上执行
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    // 获取输入张量q的选项(设备、数据类型等)
    auto opts = q.options();
    // 创建softmax_lse张量,用于存储每个注意力头的log-sum-exp值
    auto softmax_lse = torch::empty({num_heads, total_q}, opts.dtype(at::kFloat));
    at::Tensor p;
    // 只有在有dropout时才返回softmax结果,以减少编译时间
    if (return_softmax) {
        // 确保dropout概率大于0
        TORCH_CHECK(p_dropout > 0.0f, "return_softmax is only supported when p_dropout > 0.0");
        // 创建p张量存储softmax结果
        p = torch::empty({ batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded }, opts);
    }
    else {
        // 如果不需要返回softmax,创建一个空张量
        p = torch::empty({ 0 }, opts);
    }

    // 如果需要将张量初始化为0
    if (zero_tensors) {
        out.zero_();  // 输出张量置0
        softmax_lse.fill_(-std::numeric_limits<float>::infinity());  // softmax_lse填充负无穷
        if (return_softmax) {p.zero_();}  // softmax结果张量置0
    }

    // 创建前向传播参数结构体
    Flash_fwd_params params;
    // 设置前向传播的各项参数
    set_params_fprop(params,
                     batch_size,
                     max_seqlen_q, max_seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q, k, v, out,
                     cu_seqlens_q_d,
                     cu_seqlens_k.data_ptr(),
                     seqused_k.has_value() ? seqused_k.value().data_ptr() : nullptr,
                     return_softmax ? p.data_ptr() : nullptr,
                     softmax_lse.data_ptr(),
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap,
                     seqlenq_ngroups_swapped,
                     /*unpadded_lse*/true);
    params.total_q = total_q;

    // 如果使用分页KV缓存
    if (paged_KV) {
        params.block_table = block_table.data_ptr<int>();  // 设置分块表指针
        params.block_table_batch_stride = block_table.stride(0);  // 设置分块表的batch步长
        params.k_batch_stride = k.stride(0);  // 设置K的batch步长
        params.v_batch_stride = v.stride(0);  // 设置V的batch步长
    }
    params.page_block_size = page_block_size;  // 设置页块大小

    // 保持对这些张量的引用以延长其生命周期
    at::Tensor softmax_lse_accum, out_accum;
    if (seqlenq_ngroups_swapped) {
        // 仅在解码时应用split-k
        std::tie(softmax_lse_accum, out_accum) =
            set_params_splitkv(params, batch_size, num_heads, head_size,
                               max_seqlen_k, max_seqlen_q, head_size_rounded,
                               p_dropout, /*num_splits*/ 0, dprops, opts);
    }

    // 如果提供了K序列的左侧填充信息
    if (leftpad_k_.has_value()) {
        auto leftpad_k = leftpad_k_.value();
        // 检查:不能同时使用分页KV和左侧填充
        TORCH_CHECK(!paged_KV, "We don't support Paged KV and leftpad_k running at the same time yet");
        // 检查数据类型必须是int32
        TORCH_CHECK(leftpad_k.dtype() == torch::kInt32, "leftpad_k must have dtype int32");
        CHECK_DEVICE(leftpad_k);  // 检查设备
        CHECK_CONTIGUOUS(leftpad_k);  // 检查连续性
        CHECK_SHAPE(leftpad_k, batch_size);  // 检查形状
        params.leftpad_k = static_cast<int *>(leftpad_k.data_ptr());  // 设置左侧填充指针
    }

    // 为每个线程生成随机数的次数,用于偏移THC随机状态中的philox计数器
    // 我们使用自定义的RNG,将偏移量增加batch_size * num_heads * 32
    int64_t counter_offset = params.b * params.h * 32;
    // 创建一个CUDA上的float32类型的张量选项
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    // 创建一个大小为2的int64类型的空张量,用于存储RNG状态
    auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));
    // 前向传播kernel将用种子和偏移量填充内存
    params.rng_state = reinterpret_cast<uint64_t*>(rng_state.data_ptr());

    // 如果设置了dropout
    if (p_dropout > 0.0)  {
        // 获取默认的CUDA生成器或使用提供的生成器
        auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
            gen_, at::cuda::detail::getDefaultCUDAGenerator());
        // 使用互斥锁保护随机数生成器的访问
        std::lock_guard<std::mutex> lock(gen->mutex_);
        // 设置philox随机数生成器的状态
        params.philox_args = gen->philox_cuda_state(counter_offset);
    }

    // 设置ALiBi(Attention with Linear Biases)的参数
    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    // 如果K序列长度大于0,执行前向传播
    if (max_seqlen_k > 0) {
        // 获取当前CUDA流
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        // 运行前向传播kernel
        run_mha_fwd(params, stream, paged_KV);
    } else {
        // 如果K序列长度为0,说明是空张量,需要将输出置零
        out.zero_();
        // 将softmax的对数和填充为负无穷
        softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    // 如果进行了序列长度和组数的交换
    if (seqlenq_ngroups_swapped) {
        // 定义reshape前后的维度大小
        int64_t size_before[] = {batch_size, max_seqlen_q, num_heads_k, head_size};
        int64_t size_after[] = {batch_size, num_heads_k * max_seqlen_q, head_size};
        // 重新排列输出张量的维度
        out = out.reshape(size_before).transpose(1, 2).reshape(size_after);
        q = q.reshape(size_before).transpose(1, 2).reshape(size_after);
        // 重新排列softmax对数和的维度
        softmax_lse = softmax_lse.reshape({num_heads * max_seqlen_q, batch_size});
    }

    // 返回输出张量、softmax对数和、注意力分布(如果需要)和RNG状态
    return {out, softmax_lse, p, rng_state};
}
```

由于Flash Attention的准备工作比较多，上面的代码很长，我们主要关注

```c++
if (seqlenq_ngroups_swapped) {
        // 仅在解码时应用split-k
        std::tie(softmax_lse_accum, out_accum) =
            set_params_splitkv(params, batch_size, num_heads, head_size,
                               max_seqlen_k, max_seqlen_q, head_size_rounded,
                               p_dropout, /*num_splits*/ 0, dprops, opts);
    }
```

和

```c++
if (max_seqlen_k > 0) {
        // 获取当前CUDA流
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        // 运行前向传播kernel
        run_mha_fwd(params, stream, paged_KV);
    }
```

这几行代码即可，`set_params_splitkv`决定了是否使用split-k以及要在kv的序列纬度上切分多少次，`run_mha_fwd`会根据`set_params_splitkv`的配置以及在上面的函数中其它部分设置的params的参数来dispatch不同的kernel。现在来看一下`set_params_splitkv`的实现：

```c++
std::tuple<at::Tensor, at::Tensor> set_params_splitkv(Flash_fwd_params &params, const int batch_size,
    const int num_heads, const int head_size, const int max_seqlen_k, const int max_seqlen_q,
    const int head_size_rounded, const float p_dropout,
    const int num_splits, cudaDeviceProp *dprops, struct c10::TensorOptions opts) {

    // 这里的block_n需要和run_mha_fwd_splitkv_dispatch中的配置匹配
    // 根据head_size的大小选择不同的block_n:
    // - head_size <= 64: block_n = 256
    // - 64 < head_size <= 128: block_n = 128  
    // - head_size > 128: block_n = 64
    const int block_n = head_size <= 64 ? 256 : (head_size <= 128 ? 128 : 64);
    
    // 计算在K序列维度上需要多少个block
    const int num_n_blocks = (max_seqlen_k + block_n - 1) / block_n;
    
    // 对于splitKV kernel,kBlockM固定为64
    // 一般在推理时Q序列长度不会超过64
    const int num_m_blocks = (max_seqlen_q + 64 - 1) / 64;
    
    // 设置切分数量
    params.num_splits = num_splits;
    
    // 声明用于存储中间结果的tensor
    at::Tensor softmax_lse_accum;
    at::Tensor out_accum;

    // splitKV目前不支持dropout
    if (p_dropout == 0.0f) {  
        if (num_splits < 1) {
            // 如果num_splits < 1,则使用启发式方法计算切分数量
            // 这里乘以2是因为每个block使用128个线程
            params.num_splits = num_splits_heuristic(batch_size * num_heads * num_m_blocks, 
                                                   dprops->multiProcessorCount * 2, 
                                                   num_n_blocks, 128);
        }
        
        // 如果需要切分(num_splits > 1)
        if (params.num_splits > 1) {
            // 分配存储中间结果的tensor
            softmax_lse_accum = torch::empty({params.num_splits, batch_size, num_heads, max_seqlen_q}, 
                                           opts.dtype(at::kFloat));
            out_accum = torch::empty({params.num_splits, batch_size, num_heads, max_seqlen_q, head_size_rounded}, 
                                   opts.dtype(at::kFloat));
            
            // 设置指向中间结果的指针
            params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
            params.oaccum_ptr = out_accum.data_ptr();
        }
        
        // 切分数量不能超过128
        TORCH_CHECK(params.num_splits <= 128, "num_splits > 128 not supported");
    }

    return std::make_tuple(softmax_lse_accum, out_accum);
}
```

由于调用`set_params_splitkv`时设置了`num_splits=0`所以上面的代码会进入到启发式计算切分数量的逻辑中，启发式计算切分数量的逻辑在`num_splits_heuristic`中，我们来看一下这个函数的实现：

```python
// 这个函数用于找到最大化 GPU 占用率的切分数量。
// 例如,如果 batch * n_heads = 48,且有 108 个 SM,那么:
// - 使用 2 个切分(效率 = 0.89)比使用 3 个切分(效率 = 0.67)更好
// 但是我们也不希望切分太多,因为这会导致更多的 HBM 读写。
// 所以我们先找到最佳效率,然后找到能达到最佳效率 85% 的最小切分数量。
inline int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits) {
    // 如果当前 batch_nheads_mblocks 已经能填充 80% 的 SM,就不需要切分了
    if (batch_nheads_mblocks >= 0.8f * num_SMs) { return 1; }
    
    // 取 max_splits、SM数量和 n_blocks 三者的最小值作为最大切分数量
    max_splits = std::min({max_splits, num_SMs, num_n_blocks});
    float max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    
    // 向上取整除法
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
    
    // 有些切分数量是无效的。例如,如果我们有 64 个 blocks:
    // - 选择 11 个切分,我们会有 6 * 10 + 4 个 blocks
    // - 选择 12 个切分,我们会有 6 * 11 + (-2) 个 blocks(实际上还是 11 个切分)
    // 所以我们需要检查每个切分的 block 数量是否与前一个切分数量相同
    auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
        return num_splits == 1 || ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
    };
    
    // 第一轮循环:计算每个切分数量的效率,并找到最大效率
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) {
            efficiency.push_back(0.f);
        } else {
            // n_waves 表示每个 SM 平均需要处理多少波 blocks
            float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
            // 效率 = 理论处理时间 / 实际处理时间
            float eff = n_waves / ceil(n_waves);
            if (eff > max_efficiency) { max_efficiency = eff; }
            efficiency.push_back(eff);
        }
    }
    
    // 第二轮循环:找到能达到最佳效率 85% 的最小切分数量
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) { continue; }
        if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
            return num_splits;
        }
    }
    return 1;
}
```

从上面的代码我们就可以看出来影响splitkv的参数不仅有`max_seqlen_k`，`head_num`，还有SM个数等等。对于文章开头的例子，`head_num`和SM个数是固定的，但由于`max_new_tokens`从512变成64引起了`max_seqlen_k`的改变从而导致了`num_splits`的改变，最终表现为我们在`max_new_tokens`为512的nsys中观察到了decoding时使用了splitkv的flash attention实现，而在`max_new_tokens`为64的nsys中则没有使用splitkv的flash attention实现。

`run_mha_fwd`的dispatch逻辑为：

```c++
void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel=false) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                if (params.num_splits <= 1 && !force_split_kernel) {  // If we don't set it num_splits == 0
                    run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                } else {
                    run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                }
            });
        });
    });
}
```

可以看到这里对num_splits进行判断，如果num_splits <= 1且没有设置force_split_kernel则dispatch不使用splitkv的kernel，否则dispatch使用splitkv的kernel。

`flash_attn_cuda.varlen_fwd`的初步dispatch逻辑就梳理完了，不过我们从文章开头的nsys可以看到调用splitkv实现的时候每个decoding step的每个Attenion计算都有2个kernel：


![](https://files.mdnice.com/user/59/43630614-77d2-4178-a2bb-2c5c0702f1e0.png)


![](https://files.mdnice.com/user/59/5a7019f2-e0df-40c9-8545-30585d63bad6.png)

![](https://files.mdnice.com/user/59/ec5489ae-1531-4cc8-a2a8-f7c28f34fffe.png)

在KV的seq纬度切分之后还需要把单独计算的结果组合成最终的计算结果，这就是`flash_fwd_splitkv_combine_kernel`的作用。实际上这个也被叫作Flash Decoding，你可以参考[https://mp.weixin.qq.com/s/hvqPhNo3l0tL_-lf978euw](https://mp.weixin.qq.com/s/hvqPhNo3l0tL_-lf978euw) 这里的介绍。


# 0x5. run_mha_fwd_splitkv_dispatch的上层实现逻辑

```python
template<typename Kernel_traits, bool Is_causal>
void run_flash_splitkv_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    // 确保kernel特征不支持Q在寄存器中和Q/K共享共享内存
    static_assert(!Kernel_traits::Is_Q_in_regs, "SplitKV implementation does not support Is_Q_in_regs");
    static_assert(!Kernel_traits::Share_Q_K_smem, "SplitKV implementation does not support Share_Q_K_smem");
    
    // 获取共享内存大小
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    
    // 计算M维度的block数量
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    
    // 设置grid维度:
    // x: M维度的block数
    // y: 如果有splits则为splits数量,否则为batch size
    // z: 如果有splits则为batch*heads,否则为heads数量
    dim3 grid(num_m_block, params.num_splits > 1 ? params.num_splits : params.b, params.num_splits > 1 ? params.b * params.h : params.h);

    // 判断序列长度是否能被block大小整除
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    
    // 判断head维度是否匹配
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;

    // 使用一系列宏来根据不同条件选择不同的kernel实现
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal, Is_local, [&] {
                BOOL_SWITCH(params.num_splits > 1, Split, [&] {
                    BOOL_SWITCH(params.knew_ptr != nullptr, Append_KV, [&] {
                        ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                            SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                                // 选择合适的kernel实现
                                auto kernel = &flash_fwd_splitkv_kernel<Kernel_traits, Is_causal, Is_local && !Is_causal, Has_alibi, IsEvenMNConst && !Append_KV && IsEvenKConst && !Is_local && Kernel_traits::kHeadDim <= 128, IsEvenKConst, Is_softcap, Split, Append_KV>;
                                
                                // 如果共享内存超过48KB,需要设置属性
                                if (smem_size >= 48 * 1024) {
                                    C10_CUDA_CHECK(cudaFuncSetAttribute(
                                        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                                }
                                
                                // 启动kernel
                                kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                                C10_CUDA_KERNEL_LAUNCH_CHECK();
                            });
                        });
                    });
                });
            });
        });
    });

    // 如果有splits,需要启动combine kernel来合并结果
    if (params.num_splits > 1) {
        // 根据head维度选择合适的block大小
        constexpr static int kBlockM = Kernel_traits::kHeadDim % 128 == 0 ? 4 : (Kernel_traits::kHeadDim % 64 == 0 ? 8 : 16);
        dim3 grid_combine((params.b * params.h * params.seqlen_q + kBlockM - 1) / kBlockM);
        
        // 根据splits数量选择合适的combine kernel
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            if (params.num_splits <= 2) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 1, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 4) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 2, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 8) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 3, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 16) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 4, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 32) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 5, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 64) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 6, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 128) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 7, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    }
}

// 根据head维度选择合适的block大小并调用run_flash_splitkv_fwd
template<typename T, int Headdim, bool Is_causal>
void run_mha_fwd_splitkv_dispatch(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int kBlockM = 64;  // 固定M维度的block大小为64
    // 根据head维度选择N维度的block大小:
    // head维度<=64: 256
    // head维度<=128: 128 
    // 其他: 64
    constexpr static int kBlockN = Headdim <= 64 ? 256 : (Headdim <= 128 ? 128 : 64);
    run_flash_splitkv_fwd<Flash_fwd_kernel_traits<Headdim, kBlockM, kBlockN, 4, false, false, T>, Is_causal>(params, stream);
}
```

我们可以看到无论是在序列维度切分计算的`flash_fwd_split_kv_kernel`还是最后合并结果的`flash_fwd_splitkv_combine_kernel`，他们都有非常多的模板来决定当前的输入下应该使用哪种kernel来获得最佳性能。如果你对这里的cuda实现感兴趣可以自行阅读源码学习或者修改。


# 0x6. 总结

本文主要探讨了Flash Attention在不同场景下的kernel dispatch逻辑，特别关注了decode阶段使用split_kv实现的触发条件。通过分析源码发现，Flash Attention的dispatch逻辑主要由max_seqlen_k（K序列的最大长度）、head_num（注意力头数）、SM数量（GPU的流处理器数量）等因素决定。这些因素会通过启发式函数num_splits_heuristic来计算num_splits（KV序列维度的切分数量），该函数的目标是找到能最大化GPU利用率的切分数量。当计算得到num_splits > 1时，会使用split_kv实现，这种实现会启动两个kernel：`flash_fwd_splitkv_kernel`用于在KV序列维度上进行切分计算，`flash_fwd_splitkv_combine_kernel`用于合并各个切分的计算结果。这就解释了文章开头的例子中，当max_new_tokens=512时由于序列长度较长导致num_splits > 1而使用split_kv实现，而max_new_tokens=64时由于序列长度较短导致num_splits <= 1而使用普通实现的现象。这种灵活的dispatch机制设计使得Flash Attention能够在不同场景下都获得较好的性能表现：在长序列场景下通过split_kv更好地利用GPU资源，在短序列场景下避免不必要的开销。

