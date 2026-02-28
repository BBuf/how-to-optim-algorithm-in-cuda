# 问题1: GTX 4090 的 Attention 诡异

最近想探索一个 HunyuanVideo 的 Flash Attention 算子的 MFU，所以把 HunyuanVideo 的 调用 FlashAttention 的 q, k, v 的shape记录下来了。然后写了一个测试 FlashAttention MFU的脚本，具体如下：

```python
import torch
import math
import time
from flash_attn import flash_attn_varlen_func

# 设置随机种子以确保结果可复现
torch.manual_seed(0)

# 设置新的参数
context_length = 111856
hidden_size = 3072
nheads = 24
headdim = 128  # hidden_size // nheads
batch_size = 1
dropout_p = 0.0
causal = False
softmax_scale = 1.0 / math.sqrt(headdim)

# 生成输入数据
q = torch.randn(context_length, nheads, headdim, dtype=torch.bfloat16, device="cuda", requires_grad=True)
k = torch.randn(context_length, nheads, headdim, dtype=torch.bfloat16, device="cuda", requires_grad=True)
v = torch.randn(context_length, nheads, headdim, dtype=torch.bfloat16, device="cuda", requires_grad=True)

# 设置累积序列长度
cu_seqlens_q = torch.tensor([0, context_length], dtype=torch.int32, device="cuda")
cu_seqlens_k = cu_seqlens_q.clone()

# 计算理论FLOPS
def calculate_attention_flops():
    return context_length * context_length * headdim * nheads * 4

# 获取GPU理论峰值FLOPS (根据您的GPU型号可能需要调整)
def get_gpu_peak_flops():
    # 以下是一些常见GPU的理论峰值FLOPS (BF16)
    # A100: 312 TFLOPS
    # A6000: 142 TFLOPS
    # GTX 4090: 165 TFLOPS
    # 请根据您的实际GPU调整
    return 165 * 1024 * 1024 * 1024 # GTX 4090

# 预热
torch.cuda.nvtx.range_push("warmup")
for _ in range(10):
    _ = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q,
        cu_seqlens_k,
        context_length,
        context_length,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
    )
torch.cuda.nvtx.range_pop()
# 计时和性能测试
torch.cuda.synchronize()
start_time = time.time()

num_iters = 200
torch.cuda.nvtx.range_push("test")
for i in range(num_iters):
    torch.cuda.nvtx.range_push(f"iter_{i}")
    out = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q,
        cu_seqlens_k,
        context_length,
        context_length,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
    )
    torch.cuda.nvtx.range_pop()
torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
end_time = time.time()

# 计算性能指标
elapsed_time = end_time - start_time
avg_time = elapsed_time / num_iters
flops = calculate_attention_flops()
flops_per_sec = flops / (avg_time*1000)
theoretical_peak_flops = get_gpu_peak_flops()
mfu = flops_per_sec / theoretical_peak_flops

print(f"\n性能测试结果:")
print(f"平均运行时间: {avg_time*1000:.2f} ms")
print(f"MFU: {mfu*100:.2f}%")
```

具体来说，我在A800和GTX 4090上运行了测试。`def get_gpu_peak_flops():`这个函数里面需要根据实际的GPU型号调整，GTX 4090 的bf16峰值 FLOPS 是 165 TFLOPS,  A800 的bf16峰值 FLOPS 是 312 TFLOPS。然后分别跑出来的测试结果为：

- GTX 4090

性能测试结果:
平均运行时间: 967.51 ms
MFU: 89.69%

- A800:

性能测试结果:
平均运行时间: 698.68 ms
MFU: 65.68%

诡异的地方就是 GTX 4090 上的  MFU 几乎到了 90%，感觉十分诡异。

所以我只调用一次这个api，使用下面的脚本来用nsight compute profile。

```python
import torch
import math
import time
from flash_attn import flash_attn_varlen_func

# 设置随机种子以确保结果可复现
torch.manual_seed(0)

# 设置新的参数
context_length = 111856
hidden_size = 3072
nheads = 24
headdim = 128  # hidden_size // nheads
batch_size = 1
dropout_p = 0.0
causal = False
softmax_scale = 1.0 / math.sqrt(headdim)

# 生成输入数据
q = torch.randn(context_length, nheads, headdim, dtype=torch.bfloat16, device="cuda", requires_grad=True)
k = torch.randn(context_length, nheads, headdim, dtype=torch.bfloat16, device="cuda", requires_grad=True)
v = torch.randn(context_length, nheads, headdim, dtype=torch.bfloat16, device="cuda", requires_grad=True)

# 设置累积序列长度
cu_seqlens_q = torch.tensor([0, context_length], dtype=torch.int32, device="cuda")
cu_seqlens_k = cu_seqlens_q.clone()

torch.cuda.synchronize()
out = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q,
    cu_seqlens_k,
    context_length,
    context_length,
    dropout_p=dropout_p,
    softmax_scale=softmax_scale,
    causal=causal,
)
torch.cuda.synchronize()
print(out.shape)
```

我在 4090 和 A800 上都用 nsight compute 抓取了profile结果，Speed Of Light显示A800达到了70%+的SM利用率，而4090的SM利用率只有40%+。这和计算出的MFU结果不一致。查看了ncu的RoofLine发现，这个kernel在两个平台上都落在了memory bound区间上。然后还查看了Memory Workloads Analysis，发现这个kernel在A800上从HBM读了16GB数据，在GTX 4090上从HBM读了6.2GB数据，理论上读取QKV只需要不到2GB，不知道为什么会gap这么大，感觉也是个很奇怪的问题。

# 问题2: FlashInfer性能问题(A800)

下面是flashinfer在上面的输入设置下的测速脚本，调用了`flashinfer.single_prefill_with_kv_cache`：

```python
import torch
import math
import time
import flashinfer

# 设置随机种子以确保结果可复现
torch.manual_seed(0)

# 设置参数
context_length = 111856
hidden_size = 3072
nheads = 24
headdim = 128  # hidden_size // nheads
qo_len = context_length
kv_len = context_length
num_qo_heads = nheads
num_kv_heads = nheads  # 非MQA/GQA模式，kv_heads等于qo_heads
head_dim = headdim
batch_size = 1
causal = False
softmax_scale = 1.0 / math.sqrt(headdim)

# 生成输入数据
q = torch.randn(qo_len, num_qo_heads, head_dim, dtype=torch.bfloat16, device="cuda", requires_grad=True)
k = torch.randn(kv_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda", requires_grad=True)
v = torch.randn(kv_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda", requires_grad=True)

# 创建attention mask - 非causal情况下使用全1矩阵
mask = torch.full((qo_len, kv_len), True, device="cuda")

# 计算理论FLOPS
def calculate_attention_flops():
    return qo_len * kv_len * head_dim * num_qo_heads * 4

# 获取GPU理论峰值FLOPS (以A100为例)
def get_gpu_peak_flops():
    return 312 * 1024 * 1024 * 1024  # A100 BF16 TFLOPS

# 预热
print("开始预热...")
for _ in range(10):
    _ = flashinfer.single_prefill_with_kv_cache(
        q, k, v, 
        causal=False,
        allow_fp16_qk_reduction=True
    )

# 计时和性能测试
torch.cuda.synchronize()
start_time = time.time()

num_iters = 20
print(f"\n开始测试 {num_iters} 次迭代...")

for i in range(num_iters):
    print(f"第 {i+1} 次迭代")
    out = flashinfer.single_prefill_with_kv_cache(
        q, k, v,
        custom_mask=mask,
        causal=False,
        allow_fp16_qk_reduction=True
    )
    torch.cuda.synchronize()

torch.cuda.synchronize()
end_time = time.time()

# 计算性能指标
elapsed_time = end_time - start_time
avg_time = elapsed_time / num_iters
flops = calculate_attention_flops()
flops_per_sec = flops / (avg_time*1000)
theoretical_peak_flops = get_gpu_peak_flops()
mfu = flops_per_sec / theoretical_peak_flops

print(f"\nFlashInfer 性能测试结果:")
print(f"输入形状:")
print(f"- Query: {q.shape}")
print(f"- Key/Value: {k.shape}")
print(f"平均运行时间: {avg_time*1000:.2f} ms")
print(f"TFLOPS: {flops_per_sec/1e12:.2f}")
print(f"MFU: {mfu*100:.2f}%")

# 验证输出形状
print(f"\n输出形状: {out.shape}")
print(f"预期形状: torch.Size([{qo_len}, {num_qo_heads}, {head_dim}])") 
```

我在A800测试，flashinfer的这个算子MFU只有22%，然后我也ncu profile了一下，脚本如下：

```python
import torch
import math
import time
import flashinfer

# 设置随机种子以确保结果可复现
torch.manual_seed(0)

# 设置参数
context_length = 111856
hidden_size = 3072
nheads = 24
headdim = 128  # hidden_size // nheads
qo_len = context_length
kv_len = context_length
num_qo_heads = nheads
num_kv_heads = nheads  # 非MQA/GQA模式，kv_heads等于qo_heads
head_dim = headdim
batch_size = 1
causal = False
softmax_scale = 1.0 / math.sqrt(headdim)

# 生成输入数据
q = torch.randn(qo_len, num_qo_heads, head_dim, dtype=torch.bfloat16, device="cuda", requires_grad=True)
k = torch.randn(kv_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda", requires_grad=True)
v = torch.randn(kv_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda", requires_grad=True)

# 创建attention mask - 非causal情况下使用全1矩阵
mask = torch.full((qo_len, kv_len), True, device="cuda")

# 计时和性能测试
torch.cuda.synchronize()

out = flashinfer.single_prefill_with_kv_cache(
    q, k, v,
    custom_mask=mask,
    causal=False,
    allow_fp16_qk_reduction=False
)
torch.cuda.synchronize()


print(f"\n输出形状: {out.shape}")
```

flashinfer的这个算子在这种情况下，算力没有打满，是compute bound的，然后存在大量的内存非合并访问，然后从Memory Workloads Analysis看到从HBM读数据达到了惊人的647GB。

