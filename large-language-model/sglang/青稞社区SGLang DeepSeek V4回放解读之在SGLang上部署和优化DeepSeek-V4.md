> 这篇是对青稞社区 SGLang DeepSeek V4 回放中「Deploying and Optimizing DeepSeek-V4 on SGLang」这个分享的技术解读。和单纯复述 slides 不同，这篇主要基于 SGLang 最新 main 源码来解释：SGLang 是怎么把 DeepSeek-V4 的 SWA / CSA / HCA、ShadowRadix、多级 KV pool、Flash Compressor、Lightning TopK、MTP、HiSparse、MegaMoE、CP / PD 部署串成一个可运行系统的。本文基于 `/Users/bbuf/工作目录/Common/sglang`，2026-05-19 同步到 `origin/main` 后的 commit `4c9f31b85`。

# 0x0. 前言

![](https://files.mdnice.com/user/59/c4c5e935-e8a1-4591-a5b4-fdd9eb2e3889.png)

DeepSeek-V4 给推理系统带来的麻烦，主要不在「模型更大」或者「MoE 更重」，而在 attention runtime 的状态空间。每层都有 SWA，同时又会在 CSA 和 HCA 两种压缩注意力之间切换。SGLang 不能继续把 KV cache 当成一套 per-layer raw KV 来处理，它需要同时维护 full-token 坐标、SWA 物理池、C4 压缩池、C128 压缩池、C4 indexer 池、compress state 池，以及这些池子在 prefix cache、CUDA Graph、PD disaggregation、HiSparse offload 里的映射关系。

![](https://files.mdnice.com/user/59/bc7ac8b1-ff74-4e9e-8450-02cb14cef01d.png)

Slides 的 Highlights 把推理、RL 训练和硬件支持都放在一起讲。本文主要展开推理侧，因为这部分已经能在 SGLang 源码里看到比较完整的实现。先看这几组文件：

```text
配置和启动默认值：
python/sglang/srt/configs/deepseek_v4.py
python/sglang/srt/arg_groups/deepseek_v4_hook.py
python/sglang/srt/environ.py

模型结构和 forward 主路径：
python/sglang/srt/models/deepseek_v4.py
python/sglang/srt/models/deepseek_v4_nextn.py

Attention backend / metadata / indexer / compressor：
python/sglang/srt/layers/attention/deepseek_v4_backend.py
python/sglang/srt/layers/attention/dsv4/indexer.py
python/sglang/srt/layers/attention/dsv4/compressor_v2.py
python/sglang/srt/layers/attention/dsv4/metadata_kernel.py

KV cache 和压缩状态：
python/sglang/srt/model_executor/pool_configurator.py
python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py
python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py
python/sglang/srt/mem_cache/deepseek_v4_compress_state.py

JIT/CUDA kernel：
python/sglang/jit_kernel/deepseek_v4.py
python/sglang/jit_kernel/dsv4/compress.py
python/sglang/jit_kernel/csrc/deepseek_v4/
python/sglang/jit_kernel/include/sgl_kernel/deepseek_v4/

部署 recipe：
docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx
docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx
```

如果只想抓住这套实现的主线，可以先看下面这条链路：

```text
ServerArgs
  -> apply_deepseek_v4_defaults
  -> DSV4PoolConfigurator
  -> DeepSeekV4TokenToKVPool
  -> DeepseekV4AttnBackend.init_forward_metadata
  -> MQALayer.forward
  -> C4Indexer / CompressorV2 / FlashMLA
  -> DeepseekV4DecoderLayer.mlp
```

后面每一节都会沿着这条链路展开。

# 0x1. 启动入口：SGLang 先把 DeepSeek-V4 收进一组强约束里

DeepSeek-V4 不是靠用户手动选一堆 backend 跑起来的。SGLang 在启动阶段就给它套了一组模型专用默认值，入口在 `python/sglang/srt/arg_groups/deepseek_v4_hook.py`：

```python
server_args.attention_backend = "dsv4"
server_args.page_size = 256

if server_args.max_running_requests is None:
    server_args.max_running_requests = 256

if server_args.kv_cache_dtype == "auto":
    server_args.kv_cache_dtype = "fp8_e4m3"
assert server_args.kv_cache_dtype in ["fp8_e4m3"]
```

这几行是整个 DSv4 runtime 的地基：

- `attention_backend="dsv4"`：让模型进入专用 backend，而不是复用通用 MLA / FlashAttention 路径。
- `page_size=256`：后面 C4/C128/SWA 的 page 推导都依赖这个常量。
- `kv_cache_dtype=fp8_e4m3`：KV cache 的 nope 部分按 FP8 存，RoPE 部分按 BF16 存。
- `max_running_requests=256`：给 cookbook recipe 一个默认并发上限。

投机解码也在这里被收紧：

```python
assert server_args.speculative_algorithm == "EAGLE"
assert server_args.speculative_eagle_topk == 1
```

这里的意思不是说 EAGLE 不属于 speculative decoding。相反，EAGLE 就是当前 DSv4 允许的投机解码路径。限制在于：如果开启 speculative，`speculative_algorithm` 必须是 `EAGLE`，并且 `speculative_eagle_topk` 必须是 1；其它 speculative algorithm，以及 EAGLE topk 大于 1 的多分支候选路径，当前都会被拒掉。

Context Parallelism 的约束也在同一个文件里：

```python
server_args.enable_dp_attention = True
server_args.moe_dense_tp_size = 1
server_args.attn_cp_size = server_args.tp_size // server_args.dp_size
assert server_args.dp_size == 1
assert server_args.tp_size <= 8
```

DeepSeek-V4 的 CP 当前只支持 `round-robin-split`，并且限定在单机 TP 范围内。这个限制会在后面的 metadata reindex 和 MLP/DeepEP 路径里继续出现。

`python/sglang/srt/environ.py` 里还有一组 DSv4 相关默认开关。几个对源码阅读特别重要：

```text
SGLANG_OPT_USE_COMPRESSOR_V2=True
SGLANG_OPT_USE_TOPK_V2=True
SGLANG_OPT_FUSE_WQA_WKV=True
SGLANG_OPT_USE_FUSED_STORE_CACHE=True
SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=True
SGLANG_PREP_IN_CUDA_GRAPH=True
SGLANG_OPT_CACHE_SWA_TRANSLATION=True
SGLANG_DSV4_FP4_EXPERTS=True
```

按最新 main 的默认配置，DeepSeek-V4 会走 v2 compressor、topk v2、Q/KV A 投影融合、fused store cache、多流 overlap，以及 CUDA Graph 内 metadata prepare。后面解释源码时，也以这条默认路径为主。

# 0x2. 模型配置：compress_ratios 决定每层是 SWA、CSA 还是 HCA

![](https://files.mdnice.com/user/59/483420dd-42df-4bfa-b6a8-22072f75c23b.png)

Slides 这页给出了 DeepSeek-V4 的 attention 结构：

- SWA：每层都有，窗口大小 128。
- CSA：4:1 compressed sparse attention，top-k 默认 512。
- HCA：128:1 heavily compressed attention，dense 访问压缩后的 KV。

这些参数在 `python/sglang/srt/configs/deepseek_v4.py` 里直接落成 config 字段：

```python
index_head_dim = 128
index_n_heads = 64
index_topk = 512
window_size = 128

q_lora_rank = 1024
qk_nope_head_dim = 448
qk_rope_head_dim = 64
v_head_dim = 512

compress_rope_theta = 40000
compress_ratios: List[int]

hc_mult = 4
hc_sinkhorn_iters = 20
```

`compress_ratios` 先决定每一层的 attention 类型。它没有写死在 layer class 里，而是由 `compress_ratios[layer_id]` 决定。`MQALayer.__init__` 中会把它收敛到三个取值：

```python
compress_ratio = (
    compress_ratio_override
    if compress_ratio_override is not None
    else config.compress_ratios[layer_id]
)
assert compress_ratio in [0, 4, 128]
self.compress_ratio = compress_ratio
```

三个取值分别对应：

- `0`：不创建 compressor / indexer，只走 SWA。
- `4`：创建 attention compressor，同时创建 C4 indexer。
- `128`：创建 attention compressor，但不创建 C4 indexer。

源码里的分支是这样的：

```python
if self.compress_ratio:
    self.compressor = Compressor(..., compress_ratio=self.compress_ratio)

if self.compress_ratio == 4:
    self.indexer = C4Indexer(...)
```

这就把 slides 里的 CSA / HCA 变成了可执行结构：

- CSA 层需要 `Compressor(ratio=4)` 维护 C4 KV，还需要 `C4Indexer` 计算 top-k sparse page。
- HCA 层需要 `Compressor(ratio=128)` 维护 C128 KV，不需要 sparse top-k indexer。
- SWA-only 层只需要写入 SWA cache，然后通过 FlashMLA 访问最近窗口。

还有一个细节：压缩层使用的 RoPE base 不同。`MQALayer` 对压缩层使用 `config.compress_rope_theta`，对非压缩层使用普通 `rope_theta`：

```python
rope_base = config.compress_rope_theta if self.compress_ratio else rope_theta
```

压缩注意力没有复用 raw attention 的 RoPE 参数，它有自己的 RoPE 频率配置。

# 0x3. 多级 KV pool：ShadowRadix 背后的物理存储

![](https://files.mdnice.com/user/59/456565b1-4fa4-4389-b3b9-eddfb5c1a7cb.png)

Slides 里说 ShadowRadix 的要点是：Radix tree 仍然索引虚拟 full-token slot，各层共享同一套 token 坐标；写入 KV 时，再把 full-token slot 投影到 SWA / C4 / C128 的物理池。

SGLang 里这套物理池由 `DeepSeekV4TokenToKVPool` 管理。它不是单个 per-layer KV pool，而是四类池子的组合：

```python
self.swa_kv_pool = DeepSeekV4SingleKVPool(...)
self.c4_kv_pool = DeepSeekV4SingleKVPool(...) 或 HiSparseC4DevicePool(...)
self.c128_kv_pool = DeepSeekV4SingleKVPool(...)
self.c4_indexer_kv_pool = DeepSeekV4IndexerPool(...)
```

创建入口在 `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`：

```python
self.token_to_kv_pool = DeepSeekV4TokenToKVPool(
    max_num_reqs=self.max_running_requests,
    swa_size=self.swa_max_total_num_tokens,
    c4_size=self.c4_max_total_num_tokens,
    c128_size=self.c128_max_total_num_tokens,
    c4_state_pool_size=self.c4_state_pool_size,
    c128_state_pool_size=self.c128_state_pool_size,
    page_size=self.page_size,
    swa_page_size=swa_page_size,
    compression_ratios=compression_ratios,
    enable_hisparse=self.enable_hisparse,
)
```

这里的 `compression_ratios` 还有一个特殊处理：如果当前 worker 是 MTP draft worker，那么所有层都被改成 `COMPRESS_RATIO_NEXTN_LAYER=0`。draft worker 因此不拥有 C4/C128/state pool，只复用 SWA 路径。

DSv4 的内存不会直接按 full tokens 分配，而是由 `DSV4PoolConfigurator` 拆成几类 token 容量：

```python
full_token = full_token // page_size * page_size
swa_tokens = int(full_token * self.swa_ratio) // page_size * page_size

c4_max_total_num_tokens = full_token // (4 * c4_shrink_factor)
c128_max_total_num_tokens = full_token // 128
c4_state_pool_size = swa_tokens // swa_page_size * c4_ring_size
c128_state_pool_size = swa_tokens // swa_page_size * c128_ring_size
```

`swa_ratio` 默认来自前面的 `swa_full_tokens_ratio=0.1`。这表示系统不会按 full token 容量给 SWA 池完整分配，而是按一定比例保留 SWA 工作集。C4/C128 则按压缩比例缩小。

具体到 KV buffer 布局，`DeepSeekV4SingleKVPool` 每 token 的存储是 584 bytes：

```python
qk_nope_head_dim FP8: 448 bytes
qk_rope_head_dim BF16: 64 * 2 bytes
nope FP8 scales + scale_pad: 8 bytes
```

源码里有一个 assert 直接把这个布局写死：

```python
assert bytes_per_token == 448 + 64 * 2 + 8
```

然后再按 page 做 padding：

```python
bytes_per_page_non_padded = self.page_size * bytes_per_token
self.bytes_per_page_padded = ceil_div(bytes_per_page_non_padded, 576) * 576
```

这个 page padding 是为了让底层 kernel 的访问布局更规整。

Layer 到压缩池的映射在 `_init_compressed_layer_mapping` 里完成：

```python
if ratio == 0:
    layer_mapping[idx] = DeepSeekV4LayerItem(compress_ratio=0, ...)
elif ratio == 4:
    layer_mapping[idx] = DeepSeekV4LayerItem(
        compress_ratio=4,
        compress_layer_id=c4_cnt,
        compress_kv_pool=self.c4_kv_pool,
    )
elif ratio == 128:
    layer_mapping[idx] = DeepSeekV4LayerItem(
        compress_ratio=128,
        compress_layer_id=c128_cnt,
        compress_kv_pool=self.c128_kv_pool,
    )
```

这里的 `compress_layer_id` 是 bucket 内局部编号。比如全模型第 10 层可能是第 4 个 C4 层，那么它在 `c4_kv_pool` 里使用的 layer id 就是 3，而不是 10。这样 C4/C128 池可以只为对应层分配 buffer。

ShadowRadix 的主要投影函数是：

```python
def translate_loc_from_full_to_swa(self, kv_indices):
    return self.full_to_swa_index_mapping[kv_indices].to(torch.int32)
```

所有写 SWA cache 的路径都会先从 full-token raw loc 转成 SWA loc：

```python
swa_loc = self.translate_loc_from_full_to_swa(raw_loc)
self.swa_kv_pool.set_key_buffer(...)
```

默认还会缓存这次 translation：

```python
SGLANG_OPT_CACHE_SWA_TRANSLATION=True
```

这是因为同一个 forward batch 里很多 layer 都要把同一批 `out_cache_loc` 翻译到 SWA pool，缓存一次可以省掉重复映射开销。

# 0x4. Attention metadata：把 request 状态变成 FlashMLA / compressor / indexer 能吃的结构

DeepSeek-V4 的 attention backend 在 `python/sglang/srt/layers/attention/deepseek_v4_backend.py`。初始化时有几个硬约束：

```python
head_dim == 512
self.swa_page_size = 128
self.page_size = model_runner.page_size
assert self.page_size == 256
self.c4_topk = model_config.index_topk
assert speculative_eagle_topk in [0, 1]
```

这里有两个容易混在一起的 page 概念：backend 里的 `self.swa_page_size=128` 服务于 SWA attention metadata，和模型的 `window_size=128` 对齐；KV pool 在 paged SWA mode 下使用的物理 `swa_page_size` 会跟系统 `page_size=256` 对齐。最终 FlashMLA 读 SWA 时还会用 `swa_topk_lengths = clamp(seq_len, max=128)` 限制逻辑窗口，所以物理 page 可以是 256，但实际 SWA attention 仍然只看最近 128 个 token。

`DSV4AttnMetadata` 是这个后端的 metadata 容器。它同时保存 SWA、C4、C128 三类 attention 需要的信息：

```python
page_table
raw_out_loc
seq_lens_casual
positions_casual

swa_page_indices
swa_topk_lengths

c4_out_loc
c4_topk_lengths_raw
c4_topk_lengths_clamp1
c4_sparse_topk_lengths
c4_sparse_page_indices

c128_out_loc
c128_page_indices
c128_topk_lengths_clamp1

c1_flashmla_metadata
c4_flashmla_metadata
c128_flashmla_metadata
```

这里有两个容易混淆的概念：

- `page_table` 是 full-token 坐标下的 page table。
- `swa_page_indices`、`c4_sparse_page_indices`、`c128_page_indices` 是传给 FlashMLA 的实际访问 index。

`make_core_attn_metadata` 负责把 request 级别的 `req_to_token`、`req_pool_indices`、`seq_lens` 转成上面的结构。SWA 的 page index 由 `get_swa_page_indices` 生成：

```python
offsets = pos_causal.unsqueeze(1) - torch.arange(SWA_WINDOW)
raw_indices = req_to_token[req_pool_indices_repeated[:, None], offsets]
swa_indices = token_to_kv_pool.translate_loc_from_full_to_swa(raw_indices)
```

也就是对每个 query token，向前取最多 128 个 raw token，再投影到 SWA pool。

C4/C128 metadata 则由 Triton kernel 生成：

```python
(
    c4_out_loc,
    c4_positions,
    c4_seq_lens_raw,
    c4_seq_lens_clamp1,
    c128_out_loc,
    c128_positions,
    c128_seq_lens_clamp1,
    c128_page_indices,
) = init_compression_metadata(...)
```

生成后还会做 64 对齐：

```python
self.c128_page_indices = _pad_last_dim(self.c128_page_indices)
self.swa_page_indices = _pad_last_dim(self.swa_page_indices)
self.c4_sparse_page_indices = _pad_last_dim(self.c4_sparse_page_indices)
```

这个对齐来自 `PAGE_INDEX_ALIGNED_SIZE = 64`，因为后面 FlashMLA / kernel 访问希望 index 的最后一维是 64 的倍数。

Metadata 的初始化按 forward mode 分成几条路径：

```python
decode         -> init_forward_metadata_decode
prefill        -> init_forward_metadata_prefill
target_verify  -> init_forward_metadata_target_verify
draft_extend   -> init_forward_metadata_draft_extend
```

这个分流用于支持 MTP 和 CUDA Graph。普通 decode、prefill、verify、draft extend 对 out loc、seq lens、num tokens 的形状要求都不同，不能共用一套 metadata builder。

默认 `SGLANG_PREP_IN_CUDA_GRAPH=True`。这时 decode / verify 可以先返回一个 raw metadata：

```python
DSV4RawDecodeMetadata(req_pool_indices, seq_lens, out_cache_loc)
DSV4RawVerifyMetadata(...)
```

真正需要访问 `c4_compress_metadata`、`c128_compress_metadata`、`indexer_metadata` 时，再通过 `_maybe_upgrade_forward_metadata` 升级成完整 `DSV4Metadata`：

```python
if isinstance(self.forward_metadata, DSV4RawVerifyMetadata):
    self.forward_metadata = self.make_forward_metadata_from_raw_verify(...)
elif isinstance(self.forward_metadata, DSV4RawDecodeMetadata):
    self.forward_metadata = self.make_forward_metadata_from_raw_decode(...)
```

为什么要这么绕？因为 DeepSeek-V4 的 metadata 准备本身不轻，如果这部分每步 decode 都在 CUDA Graph 外做，会吃掉不少投机解码和 overlap 的收益。Raw metadata + graph 内升级，就是为了让 prepare 也进入 capture/replay 机制。

# 0x5. MQA forward：Q、KV cache、compressor、indexer、FlashMLA 的执行顺序

![](https://files.mdnice.com/user/59/bae2b5ca-806d-48fe-a429-e355fd2ebc01.png)

`MQALayer.forward` 是 DeepSeek-V4 attention 的主执行路径。可以把它拆成四步：

```text
1. 计算 Q：wq_a / wq_b + q_norm + fused_q_norm_rope
2. 计算并写入 SWA KV cache：wkv + fused_k_norm_rope_flashmla
3. 如果是 C4/C128 层，运行 compressor；如果是 C4 层，再运行 indexer
4. 调用 DeepseekV4AttnBackend.forward，最终进入 flash_mla_with_kvcache
```

Q path 先看 `_compute_q_b`：

```python
q, _ = self.wq_b(q_lora)
q = q.view(-1, self.n_local_heads, self.head_dim)
fused_q_norm_rope(q, q_out, self.eps, self.freqs_cis, positions)
```

KV path 先看 `_compute_kv_to_cache`：

```python
kv, _ = self.wkv(x)
token_to_kv_pool.set_swa_key_buffer_radix_fused_norm_rope(
    layer_id=self.layer_id,
    raw_loc=forward_batch.out_cache_loc,
    kv=kv,
    kv_weight=self.kv_norm.weight.data,
    eps=self.eps,
    freqs_cis=self.freqs_cis,
    positions=positions,
)
```

默认路径不会先产生一个完整 BF16 K，再单独 norm、RoPE、量化、写 cache，而是直接调用 `fused_k_norm_rope_flashmla`，把 norm + RoPE + 写入 FlashMLA paged cache 融成一段 JIT kernel。只有 NSA prefill CP 场景需要 BF16 KV 做跨 rank all-gather，才走 `_compute_kv_bf16`。

多流 overlap 也在 `MQALayer` 里。默认 `SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=True`，模型初始化会创建 5 条辅助 stream：

```python
self.alt_streams = [torch.cuda.Stream() for _ in range(5)]
```

`_forward_prepare_multi_stream` 会把 indexer、KV cache write、compressor 分到不同 stream：

```python
stream_kv = self.alt_streams[0]
stream_compressor = self.alt_streams[1]
stream_indexer = self.alt_streams[2]

q_lora = self._compute_q_a(...)
q_lora_ready = current_stream.record_event()

with torch.cuda.stream(stream_indexer):
    self.indexer(..., q_lora_ready=q_lora_ready)

with torch.cuda.stream(stream_kv):
    self._compute_kv_to_cache(...)

with torch.cuda.stream(stream_compressor):
    attn_backend.forward_core_compressor(...)

q = self._compute_q_b(...)
```

但它不是所有情况都启用。源码里有一组条件：

```python
enable_multi_stream = (
    SGLANG_OPT_USE_MULTI_STREAM_OVERLAP
    and self.alt_streams is not None
    and get_is_capture_mode()
    and x.shape[0] <= self._multi_stream_bs_limit
    and not nsa_prefill_cp
)
```

多流 overlap 主要服务 CUDA Graph capture 场景下的小/中 batch；Blackwell 上 batch limit 是 128，其他 CUDA 平台是 64。CP 场景因为要做跨 rank all-gather，不走这个路径。

最后进入 backend forward：

```python
o = attn_backend.forward(
    q=q,
    k=attn_k,
    v=attn_k,
    compress_ratio=self.compress_ratio,
    save_kv_cache=False,
)
```

这里 `save_kv_cache=False` 很重要，因为 cache write 已经在 `_forward_prepare*` 里完成了。backend forward 只负责从 SWA/C4/C128 cache 里读。

`DeepseekV4AttnBackend.forward` 最终会调用 FlashMLA：

```python
flash_mla.flash_mla_with_kvcache(
    q=q,
    k_cache=swa_k_cache,
    indices=swa_page_indices,
    topk_length=swa_topk_lengths,
    attn_sink=attn_sink,
    extra_k_cache=extra_k_cache,
    extra_indices_in_kvcache=extra_indices,
    extra_topk_length=extra_topk_lengths,
)
```

其中：

- `k_cache=swa_k_cache` 永远存在，对应 SWA。
- `extra_k_cache=None` 时就是 SWA-only。
- `compress_ratio=4` 时，`extra_k_cache=c4_kv_pool`，`extra_indices=c4_sparse_page_indices`，`extra_topk_length=c4_sparse_topk_lengths`。
- `compress_ratio=128` 时，`extra_k_cache=c128_kv_pool`，`extra_indices=c128_page_indices`，`extra_topk_length=c128_topk_lengths_clamp1`。

这就是 SGLang 把 SWA + CSA/HCA 统一塞进 FlashMLA 的方式：SWA 是主 cache，C4/C128 作为 extra cache 参与同一次 attention。

# 0x6. C4 Indexer 和 Lightning TopK：CSA sparse page 是怎么选出来的

![](https://files.mdnice.com/user/59/7731e674-d316-4690-8cfe-00f6448785eb.png)

CSA 的问题是：C4 压缩后仍然有大量历史 block，不能全部 dense attend，所以要先选 top-k sparse page。DeepSeek-V4 默认 `index_topk=512`，部分大模型配置也支持 1024。

C4 indexer 在 `python/sglang/srt/layers/attention/dsv4/indexer.py`。它本身也是一个小 attention-like 结构：

```python
self.n_heads = config.index_n_heads
self.head_dim = config.index_head_dim
self.index_topk = config.index_topk
self.wq_b = ReplicatedLinear(q_lora_rank, n_heads * head_dim)
self.weights_proj = ReplicatedLinear(hidden_size, n_heads)
self.compressor = Compressor(..., compress_ratio=4, head_dim=index_head_dim, rotate=True)
```

它做三件事：

第一，生成 indexer query，并在一个 fused kernel 里做 RoPE、Hadamard、FP8 quant：

```python
q, _ = self.wq_b(q_lora)
q = q.view(-1, self.n_local_heads, self.head_dim)
q_fp8, weights = fused_q_indexer_rope_hadamard_quant(
    q, weight, self.weight_scale, self.freqs_cis, positions
)
```

第二，维护 C4 indexer 自己的 compressed key cache。这个 cache 独立于 attention C4 KV pool：

```python
c4_indexer_kv_cache = token_to_kv_pool.get_index_k_with_scale_buffer(layer_id)
```

第三，用 DeepGEMM 或 TileLang 算 `q_fp8` 对 indexer KV cache 的 logits：

```python
logits = fp8_paged_mqa_logits(
    q_fp8,
    c4_indexer_kv_cache,
    weights,
    c4_seq_lens,
    page_table,
    deep_gemm_metadata,
    max_c4_seq_len,
)
```

拿到 logits 后才进入 top-k transform。默认路径是 topk v2：

```python
topk_transform_512_v2(
    logits,
    indexer_metadata.c4_seq_lens,
    core_metadata.page_table,
    core_metadata.c4_sparse_page_indices,
    indexer_metadata.c4_page_size,
    indexer_metadata.topk_metadata,
)
```

对应的 JIT/CUDA 代码在：

```text
python/sglang/jit_kernel/deepseek_v4.py
python/sglang/jit_kernel/csrc/deepseek_v4/topk_v2.cuh
python/sglang/jit_kernel/include/sgl_kernel/deepseek_v4/topk/
```

`topk_v2.cuh` 没有直接调用 sort，而是按输入规模选择不同策略：

- short path：小输入用更轻的 transform。
- fused one-stage：中等 batch 尽量一阶段完成。
- two-stage / cluster path：大输入用 cluster topk。

`cluster.cuh` 的思路是先做 histogram 和 threshold 预测，再 scatter 出高于阈值的元素和必要的 tie 元素。它服务的目标很明确：CSA 只需要 top-512 page，不需要完整排序。

Slides 里 Lightning TopK 从约 100us 降到约 15us，背景就在这里。对于每层都要做 sparse selection 的 CSA 来说，top-k 不是外围小优化，它本来就是 attention prep 的主要开销之一。

# 0x7. Flash Compressor：把 C4/C128 压缩状态维护变成 all-in-one 写 cache

![](https://files.mdnice.com/user/59/69da0a74-e6de-4bf4-8865-c41bf8323a23.png)

压缩注意力要快，不能只优化读 cache 的 FlashMLA，还要优化写压缩 cache 的路径。C4/C128 层每次 forward 都要把新 token 对应的 KV/score 合并进压缩状态，并在压缩边界写入 C4/C128 KV cache。

默认路径使用 `compressor_v2.py`：

```python
if envs.SGLANG_OPT_USE_COMPRESSOR_V2.get():
    from sglang.srt.layers.attention.dsv4.compressor_v2 import ...
```

`Compressor` 本身在 `compressor.py` 中定义，负责生成 `kv_score`：

```python
kv_score = linear_bf16_fp32(x, self.wkv_gate.weight)
```

然后 v2 backend 接手 all-in-one 路径：

```python
kv_compressed = compress_forward(
    kv_score_buffer=state_pool.kv_score_buffer.kv_score,
    kv_score_input=kv_score_input,
    ape=compressor.ape,
    plan=plan,
    compress_ratio=compress_ratio,
    head_dim=head_dim,
    is_online=is_online,
)

compress_norm_rope_store(
    kv_compressed,
    plan,
    norm_weight=norm.weight,
    norm_eps=norm.variance_epsilon,
    freq_cis=freqs_cis_cache,
    out_loc=c4_or_c128_out_loc,
    kvcache=kv_cache,
    page_size=page_size,
)
```

这个顺序正好对应 slides 里的 Flash Compressor：

```text
输入 hidden states
  -> wkv_gate 得到 KV/score
  -> compress_forward 更新状态并产出 compressed KV
  -> norm + RoPE + store 直接写入 C4/C128 KV cache
```

它和旧路径的区别在于：旧路径会先产生 compressed KV，再经过 norm/RoPE/pack/store 等多段动作；v2 把压缩、归一化、RoPE、写 cache 尽量压到一条 kernel 链路里，减少 HBM round-trip。

压缩计划由 `create_paged_compressor_data` 生成。它会把 full-token loc、SWA loc、ring buffer loc 之间的关系交给 C++ planner：

```python
CompressorPrefillPlan.generate(
    compress_ratio=compress_ratio,
    req_pool_indices=req_pool_indices,
    seq_lens=seq_lens,
    extend_lens=extend_lens,
    req_to_token=req_to_token,
    full_to_swa=full_to_swa,
    swa_page_size=swa_page_size,
    ring_size=ring_size,
    use_cuda_graph=use_prefill_cuda_graph,
)
```

Decode 则走：

```python
CompressorDecodePlan.generate(
    compress_ratio=compress_ratio,
    req_pool_indices=req_pool_indices,
    req_to_token=req_to_token,
    full_to_swa=full_to_swa,
    seq_lens=seq_lens,
    swa_page_size=swa_page_size,
    ring_size=ring_size,
)
```

State pool 的布局在 `deepseek_v4_compress_state.py`。非 online 模式下，每个 slot 存的是 KV 和 score：

```python
last_dim = 2 * (1 + overlap) * head_dim
```

C4 有 overlap，所以 `overlap=True`；C128 默认没有 overlap。C128 还支持一个 online compress 实验路径：

```python
SGLANG_OPT_USE_ONLINE_COMPRESS=False
```

如果打开 online C128，state pool 变成 `3 * head_dim`，保存 max / sum / kv，并强制 `ring_size=1`。不过源码里明确限制它还不支持 MTP：

```python
assert mr.spec_algorithm.is_none()
```

在线 C128 压缩目前约束更强，默认 production path 仍然是非 online。

# 0x8. jit_kernel 视角：DeepSeek-V4 kernel 是一套压缩注意力 runtime

上面几节是从模型 forward 往下看。这里补上 `python/sglang/jit_kernel` 目录里的实现细节。DeepSeek-V4 直接相关的文件大致分成几组：

```text
python/sglang/jit_kernel/deepseek_v4.py
python/sglang/jit_kernel/dsv4/compress.py

python/sglang/jit_kernel/csrc/deepseek_v4/c_plan.cuh
python/sglang/jit_kernel/csrc/deepseek_v4/common.cuh
python/sglang/jit_kernel/csrc/deepseek_v4/c4.cuh
python/sglang/jit_kernel/csrc/deepseek_v4/c4_v2.cuh
python/sglang/jit_kernel/csrc/deepseek_v4/c128.cuh
python/sglang/jit_kernel/csrc/deepseek_v4/c128_online.cuh
python/sglang/jit_kernel/csrc/deepseek_v4/c128_v2.cuh
python/sglang/jit_kernel/csrc/deepseek_v4/c128_online_v2.cuh
python/sglang/jit_kernel/csrc/deepseek_v4/rmsnorm.cuh
python/sglang/jit_kernel/csrc/deepseek_v4/rope.cuh
python/sglang/jit_kernel/csrc/deepseek_v4/fused_norm_rope.cuh
python/sglang/jit_kernel/csrc/deepseek_v4/fused_norm_rope_v2.cuh
python/sglang/jit_kernel/csrc/deepseek_v4/main_norm_rope.cuh
python/sglang/jit_kernel/csrc/deepseek_v4/store.cuh
python/sglang/jit_kernel/csrc/deepseek_v4/topk_v2.cuh
python/sglang/jit_kernel/csrc/deepseek_v4/topk.cuh
python/sglang/jit_kernel/csrc/deepseek_v4/topk_1024.cuh
python/sglang/jit_kernel/csrc/deepseek_v4/hash_topk.cuh
python/sglang/jit_kernel/csrc/deepseek_v4/hisparse_transfer.cuh
python/sglang/jit_kernel/csrc/deepseek_v4/mega_moe_pre_dispatch.cuh
python/sglang/jit_kernel/csrc/deepseek_v4/silu_and_mul_masked_post_quant.cuh
python/sglang/jit_kernel/csrc/deepseek_v4/silu_and_mul_masked_post_quant_tmp.cuh
python/sglang/jit_kernel/csrc/deepseek_v4/paged_mqa_metadata.cuh

python/sglang/jit_kernel/include/sgl_kernel/deepseek_v4/compress.cuh
python/sglang/jit_kernel/include/sgl_kernel/deepseek_v4/compress_v2.cuh
python/sglang/jit_kernel/include/sgl_kernel/deepseek_v4/fp8_utils.cuh
python/sglang/jit_kernel/include/sgl_kernel/deepseek_v4/kvcacheio.cuh
python/sglang/jit_kernel/include/sgl_kernel/deepseek_v4/topk/*.cuh
```

DeepSeek-V4 的 JIT kernel 不只是给某个 PyTorch op 加速。它们要维护压缩 KV 状态，处理 page 级索引和 cache layout 转换，还要覆盖 indexer topk、HiSparse 搬运、MegaMoE 预处理等 runtime 工作。模型结构的变化直接改到了 serving runtime 的 kernel 边界。

上面列表里既有当前主路径，也有旧路径和兼容路径。当前压缩路径主要看 `dsv4/compress.py + c_plan.cuh + *_v2.cuh`；`deepseek_v4.py` 里还保留了 `common.cuh / c4.cuh / c128.cuh / c128_online.cuh / fused_norm_rope.cuh` 这些早期 wrapper。`compress.cuh / compress_v2.cuh` 放共享 plan 结构和校验，`fp8_utils.cuh` 放 FP8 pack / UE8M0 scale 这类公共工具。`rope.cuh`、`rmsnorm.cuh` 是拆开的 fallback / helper kernel；`topk.cuh`、`topk_1024.cuh` 是固定输出宽度的早期 topk；`topk_v2.cuh + include/.../topk/*.cuh` 是现在的 Lightning TopK 组合实现。`silu_and_mul_masked_post_quant_tmp.cuh` 在目录里存在，但当前 Python wrapper 实际加载的是 `silu_and_mul_masked_post_quant.cuh`。

先看 Python wrapper。`deepseek_v4.py` 里所有 JIT 模块都会用同一个命名前缀：

```python
def make_name(name: str) -> str:
    return f"dpsk_v4_{name}"
```

这个前缀主要用于定位 JIT 编译缓存。wrapper 暴露出的 kernel 分工如下：

```python
_jit_main_q_norm_rope_module
_jit_main_k_norm_rope_flashmla_module
_jit_main_q_indexer_rope_hadamard_quant_module

_jit_rmsnorm_head_module
_jit_fused_rope_module
_jit_norm_rope_module
_jit_fused_store_module

_jit_compress_module
_jit_compress_128_online_module
_jit_compress_plan_module
_jit_compress_norm_rope_module

_jit_topk_module
_jit_topk1024_module
_jit_topk_v2_module
_jit_mask_topk_module
_jit_hash_topk_module
_jit_metadata_module

_jit_mega_moe_pre_dispatch_module
_jit_silu_mul_quant_varlen_module
_jit_silu_mul_quant_contig_module
_jit_silu_and_mul_clamp_module
_jit_hisparse_transfer_module
```

这些函数可以按执行阶段理解，而不是按文件名理解。

第一类是主 MLA 路径的 norm / RoPE / cache 写入。`main_norm_rope.cuh` 里有三个 kernel：

```cpp
FusedQNormRopeKernel
FusedKNormRopeFlashMLAKernel
FusedQIndexerRopeHadamardQuantKernel
```

`FusedQNormRopeKernel` 做主 attention Q 侧的 rmsnorm-self + RoPE，一般是 warp-per-(token, head)。`FusedKNormRopeFlashMLAKernel` 做 K 侧的 rmsnorm + RoPE，并直接写进 FlashMLA paged cache。这个 kernel 对 layout 的假设写死为：

```cpp
static_assert(kHeadDim == 512 && kRopeDim == 64, "FlashMLA layout requires (512, 64)");
```

DeepSeek-V4 的主 KV cache 写入不会“算完 K 再交给通用 cache writer”，而是在同一个 JIT kernel 里完成 norm、RoPE、FP8 layout 打包和 paged cache 地址计算。Indexer Q 的 kernel 还会做 RoPE + Hadamard + FP8 act quant，输出给 C4 indexer 后面的 MQA logits / topk 使用。

第二类是 Flash Compressor。`dsv4/compress.py` 是主要入口，它把 plan、compress、norm_rope_store 三段串起来：

```python
plan = CompressorPrefillPlan.generate(...)
compressed = compress_forward(...)
compress_norm_rope_store(...)
```

`compress_v2.cuh` 里定义了三个 plan 结构：

```cpp
struct alignas(16) DecodePlan {
  uint32_t seq_len;
  int32_t write_loc;
  int32_t read_page_0;
  int32_t read_page_1;
};

struct alignas(16) CompressPlan {
  uint32_t seq_len;
  uint16_t ragged_id;
  uint16_t buffer_len;
  int32_t read_page_0;
  int32_t read_page_1;
};

struct alignas(8) WritePlan {
  uint32_t ragged_id;
  int32_t write_loc;
};
```

`DecodePlan` 服务 decode，一行对应一个 batch item；`CompressPlan` 服务 prefill 的“要压缩哪些 token”；`WritePlan` 服务 prefill 的“哪些 token 只写 state 不产生 compressed output”。这些 plan 会直接控制 kernel 的读写位置。`c_plan.cuh` 还保留了 GPU 输入路径：当 `seq_lens` 已经在 GPU 上，planner 可以直接在 device 上生成 plan，避免 CUDA Graph capture 场景里出现 host sync。

C4 和 C128 的数据布局差异也体现在 kernel 里。`c4_v2.cuh` 开头直接写明：

```cpp
// kv_buffer: [num_indices, 8, head_dim * 4]
// - last dimension layout: | kv overlap | kv | score overlap | score |
// kv_input: [batch_size, head_dim * 4]
// kv_output: [batch_size, head_dim]
// score_bias (ape): [8, head_dim]
```

C4 有 overlap，所以一个 state slot 里既有当前 window 的 KV / score，也有 overlap window 的 KV / score。Decode 时写入新 token，如果当前位置到达 C4 边界，就在 8 个候选位置上加 APE bias，做 safe online softmax 和 weighted sum，输出 compressed KV。

`c128_v2.cuh` 对应一整块 128-token 压缩：

```cpp
// kv_buffer: [num_indices, 128, head_dim * 2]
// - last dimension layout: | kv | score |
// score_bias (ape): [128, head_dim]
```

C128 不需要 C4 那样的 overlap，但 block 内要处理 128 个位置的 score 和 KV。源码里用 16 个 warp 做跨 warp reduction，最后写出一个 compressed KV。`c128_online_v2.cuh` 是另一个实验路径，state pool 不是保存 `kv | score`，而是保存：

```cpp
// Buffer layout: [max | sum | kv]
```

这对应 online softmax 的增量状态。它节省了重新扫描 128 个 score 的压力，但当前约束更强，head_dim 固定 512，而且前面提到默认 env 并不开。

第三类是 compressed KV 写 cache。`fused_norm_rope_v2.cuh` 做的是 compressor 输出之后的“最后一公里”：把压缩后的 KV 做 norm、RoPE、量化，然后写到两种不同 cache：

```cpp
// Indexer variant: kHeadDim = 128
// Cache layout: 132 bytes/token (128 fp8 nope + 4 fp32 scale)

// FlashMLA variant: kHeadDim = 512
// Cache layout: 584 bytes/token = 448 fp8 nope + 64 bf16 rope + 8 scale
```

同一个 `compress_norm_rope_store` 能服务 indexer cache 和 FlashMLA cache，是因为 kernel 会根据 `kHeadDim == 128` 还是 `kHeadDim == 512` 选择不同路径。Indexer 路径是一 warp 一个 token，还包含 Hadamard 和 UE8M0 scale；FlashMLA 路径是一整个 block 处理一个 token，写 584 bytes/token 的 FlashMLA layout。

如果已经提前算好了 norm / RoPE，`store.cuh` 则提供纯粹的 `fused_store_cache`，分别写 `flashmla` 和 `indexer` 两种 cache。DeepSeek-V4 的 cache writer 按消费方拆成了多个特化 kernel。

第四类是 indexer topk。`topk.cuh` 和 `topk_1024.cuh` 是早期固定 512 / 1024 输出的 transform。当前主版本是 `topk_v2.cuh`，Python 入口是：

```python
metadata = plan_topk_v2(seq_lens)
topk_transform_512_v2(scores, seq_lens, page_tables, out_page_indices, page_size, metadata)
```

`topk_v2.cuh` 的 `CombinedTopKKernel` 会先根据 batch 内 seq_len 分布生成 metadata：

```cpp
struct alignas(16) GlobalMetadata {
  uint32_t cluster_threshold;
  uint32_t num_cluster_items;
  uint32_t reserved[2];
};
```

然后按长度走三套实现：

```cpp
using Large = impl::ClusterTopK<K>;
using Medium = impl::StreamingTopK<K>;
using Small = impl::RegisterTopK<K>;
```

短序列走 register / shared memory 的 topk；中等长度走 streaming topk；长序列走 cluster topk。`cluster.cuh` 里用了 `__cluster_dims__(1, 8, 1)`，把一个长序列的候选 page 分给 cluster 内多个 CTA，先做 histogram 和 threshold，再做 scatter。这个 kernel 解决的不是 MoE expert topk，而是“给压缩注意力从历史 page 里挑 page”。它和 DeepSeek-V3/R1 常见的 grouped topk / expert routing topk 面向的对象不同。

第五类是 paged MQA metadata。`paged_mqa_metadata.cuh` 里有一个固定参数：

```cpp
constexpr uint32_t kSplitKV = 256;
```

它会根据每个请求的 `seq_lens` 生成 `schedule_metadata`，把 MQA logits 的 split-KV work 分摊到 SM 上。这里的 metadata 是给 indexer attention logits 服务的，配合 C4 indexer 输出的候选页做后续选择。

第六类是 HiSparse 搬运。`hisparse_transfer.cuh` 调 `include/sgl_kernel/deepseek_v4/kvcacheio.cuh` 里的 layout helper。DeepSeek-V4 的 HiSparse cache 不是简单线性数组，GPU 侧是按 page 对齐的 FlashMLA layout：

```cpp
inline constexpr int64_t kGPUPageSize = 64;
inline constexpr int64_t kValueBytes = 576;
inline constexpr int64_t kScaleBytes = 8;
inline constexpr int64_t kCPUItemBytes = kValueBytes + kScaleBytes;
inline constexpr int64_t kGPUPageBytes =
    host::div_ceil(kCPUItemBytes * kGPUPageSize, 576) * 576;
```

CPU 侧则是无 padding 的线性 584 bytes/token。`transfer_item` 根据方向选择 GPU pointer 或 CPU pointer，因此同一个搬运逻辑能覆盖 DeviceToDevice、DeviceToHost、HostToDevice。HiSparse 在上层看起来是“把冷 cache offload 到 host”，但 kernel 层要解决的是 paged GPU layout 和 linear CPU layout 之间的转换。

第七类是 MoE 相关的 DeepSeek-V4 JIT kernel。这里有三个点：

```python
hash_topk(...)
mega_moe_pre_dispatch(...)
silu_and_mul_masked_post_quant(...)
```

`hash_topk.cuh` 里的 `moe_hash_topk_fused` 用 `input_ids -> tid2eid` 映射直接写 `topk_ids/topk_weights`，这是 DeepSeek-V4 hash-routed expert 的专用路径。`mega_moe_pre_dispatch.cuh` 的 `MegaMoEPreDispatchKernel` 会在 dispatch 前把 BF16 hidden 量化成 FP8 E4M3，并写 UE8M0 scale，同时把 topk id / weight 复制到 DeepGEMM MegaMoE 的对称 buffer 里，尾部 padding 的 expert id 会填成 -1。`silu_and_mul_masked_post_quant.cuh` 则把专家 FFN 的 SiLU+mul、可选 SwiGLU clamp、FP8 post-quant 融在一起，且源码里明确 DeepSeek-V4 的 limit 需要在 BF16 上 clamp。

把这些 kernel 放在一起看，DeepSeek-V4 和 DeepSeek-V3/R1 的 kernel 范式差异就很明显了。

DeepSeek-V3/R1 在 SGLang 里更多是“通用 serving kernel + 少量模型特化”：MLA attention 主要依赖 FlashMLA / FlashInfer，MoE 主要围绕 grouped topk、DeepGEMM、DeepEP、FP8/FP4 GEMM 和 expert dispatch，其他是 fused RMSNorm、RoPE、activation、量化这类局部融合。它们的 kernel 边界大多仍然是一个数学算子或一个通信 / GEMM 阶段。

DeepSeek-V4 则把 kernel 边界推进到了 runtime 状态机：

```text
request metadata
  -> full_to_swa / req_to_token
  -> plan_d / plan_c / plan_w
  -> C4/C128 state transition
  -> compressed KV
  -> norm + RoPE + FP8 cache layout
  -> indexer logits metadata
  -> page topk
  -> FlashMLA paged cache / HiSparse swap
```

对比 DeepSeek-V3/R1，V4 的差异主要在这里：

- V3/R1 的 attention kernel 直接消费 KV cache；V4 的 kernel 先维护 C4/C128 压缩 state，再把结果写成 FlashMLA / indexer 可消费的 cache。
- V3/R1 的 topk 多数服务 MoE expert routing；V4 的 Lightning TopK 服务 attention page selection，输入是 indexer logits 和 page table。
- V3/R1 的 metadata 多数是调度辅助；V4 的 `DecodePlan / CompressPlan / WritePlan` 是压缩状态机本身的一部分。
- V3/R1 的 cache layout 相对统一；V4 同时存在 C4 state、C128 state、online C128 state、FlashMLA 584 bytes/token、Indexer 132 bytes/token、HiSparse CPU linear layout。
- V3/R1 的 kernel 通常可以复用于同类 MLA / MoE 模型；V4 的 kernel 和 `compress_ratios`、SWA page、ShadowRadix、Lightning Indexer、mHC / MTP 等模型设计强绑定。

读 DeepSeek-V4 源码时不能只搜 `forward`。性能和正确性相关的一大块逻辑在 `jit_kernel/deepseek_v4.py` 和 `jit_kernel/csrc/deepseek_v4` 下面。这些 kernel 不是模型外面的加速补丁，而是推理路径本身的一部分。

# 0x9. MTP / NextN：draft worker 为什么只走 SWA

![](https://files.mdnice.com/user/59/b10990a4-f93b-4eeb-96b7-e3121603ab43.png)

Slides 里强调 MTP layer 使用 SWA-only attention。源码里直接这样写：

```python
# python/sglang/srt/models/deepseek_v4_nextn.py
COMPRESS_RATIO_NEXTN_LAYER = 0
```

NextN 模型创建 decoder layer 时强制覆盖压缩比例：

```python
self.decoder = DeepseekV4DecoderLayer(
    ...,
    is_nextn=True,
    compress_ratio_override=COMPRESS_RATIO_NEXTN_LAYER,
)
```

这意味着 draft layer 不创建 C4/C128 compressor，也不创建 C4 indexer。它只复用主模型的 SWA attention 形态。这样做有两个好处：

- Draft token 的目标是快速预测候选，不能承担完整 CSA/HCA metadata 和 compressor 成本。
- Draft worker 不需要拥有 C4/C128/state pool，内存池初始化时也会把相关容量置零。

`DeepseekV4ModelNextN.forward` 还会把 target 模型传来的 hidden states 和当前 token embedding 合起来：

```python
hc_flat = forward_batch.spec_info.hidden_states.view(n_tokens * hc_mult, d)
h_proj_hidden_states = self.h_proj(self.hnorm(hc_flat)).view(n_tokens, hc_mult, d)
e_proj_hidden_states = self.e_proj(self.enorm(hidden_states))
hidden_states = e_proj_hidden_states[:, None, :] + h_proj_hidden_states
```

所以 MTP 不是一个完全独立的小模型，它会消费 target worker 捕获的 auxiliary hidden states，再通过自己的 decoder 输出 draft logits。

Attention backend 还有一个 `DeepseekV4MultiStepBackend`，用于 speculative 多步。它会为每个 speculative step 准备独立 backend：

```python
for i in range(self.speculative_num_steps):
    self.attn_backends.append(
        DeepseekV4AttnBackend(..., speculative_step_id=i)
    )
```

Cookbook 里对应的 recipe 是：

```text
low-latency:    speculative-num-steps=3, draft-tokens=4
balanced:       speculative-num-steps=1, draft-tokens=2
max-throughput: MTP disabled
```

这和 serving 目标有关：低延迟场景更愿意用 MTP 减少主模型 decode 次数；满载吞吐场景下 verify step 可能比节省的主模型 token 更贵，所以 max-throughput recipe 反而关掉 MTP。

# 0xA. HiSparse：C4 pool 的 CPU offload 和 indexer swap-in

![](https://files.mdnice.com/user/59/2384a3cb-2ec0-4073-975c-b6e492db2aef.png)

HiSparse 这页讲的是 KV Cache Offloading。DeepSeek-V4 的实现重点是：先 offload C4 pool，SWA 和 C128 保持在 GPU 上。

`DeepSeekV4TokenToKVPool` 初始化时，如果启用 HiSparse，会把 C4 pool 类替换成 `HiSparseC4DevicePool`：

```python
c4_kv_pool_type = DeepSeekV4SingleKVPool
if enable_hisparse:
    c4_kv_pool_type = HiSparseC4DevicePool
self.c4_kv_pool = c4_kv_pool_type(...)
```

`HiSparseC4DevicePool` 只处理 C4，因为它写死了 `compress_ratio=4`：

```python
self.compress_ratio = 4
```

它提供了 full-token loc 到 compressed loc，再到 HiSparse device loc 的映射：

```python
def translate_loc_from_full_to_compressed(full_indices):
    mask = (full_indices + 1) % 4 == 0
    compressed_indices = full_indices[mask] // 4
    return compressed_indices

def translate_loc_to_hisparse_device(compressed_indices):
    return full_to_hisparse_device_index_mapping[compressed_indices]
```

为什么只先动 C4？因为 C4 是 CSA 的历史访问池，容量大、访问稀疏，最适合做 host/device 分层。SWA 是最近 128 token，访问高频且窗口小；C128 已经 128:1 压缩，容量压力相对低。

HiSparse 的 coordinator 在 `model_runner.py` 里初始化：

```python
if self.enable_hisparse:
    self.hisparse_coordinator = HiSparseCoordinator(
        req_to_token_pool=self.req_to_token_pool,
        token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
        top_k=index_topk,
        device_buffer_size=hisparse_cfg.device_buffer_size,
        host_to_device_ratio=hisparse_cfg.host_to_device_ratio,
    )
```

真正和 indexer 接上的地方在 `C4IndexerBackendMixin.forward_c4_indexer`。如果是 decode 并且存在 HiSparse coordinator，top-k transform 会额外拿到 raw indices，然后 coordinator swap-in 这些 page：

```python
core_metadata.c4_sparse_page_indices = (
    hisparse_coordinator.swap_in_selected_pages(
        req_pool_indices=forward_batch.req_pool_indices,
        compressed_seq_lens=indexer_metadata.c4_seq_lens,
        top_k_result=raw_indices,
        layer_id=compress_layer_id,
    )
)
```

非 decode 场景则只做 loc 翻译：

```python
core_metadata.c4_sparse_page_indices = (
    token_to_kv_pool.c4_kv_pool.translate_loc_to_hisparse_device(
        core_metadata.c4_sparse_page_indices
    )
)
```

HiSparse 不是在 attention 后端里简单「从 CPU 读 KV」。它位于 C4 indexer 和 FlashMLA 之间：indexer 先选出要访问的 C4 page，HiSparse 再保证这些 page 在 device buffer 中可访问，并把 indices 改成 device-side loc。

# 0xB. mHC：DeepSeek-V4 模型层里另一个不能忽略的结构

Slides 主要讲 attention 和 serving，但源码里 DeepSeek-V4 还有一个很重要的结构：mHC。它在 `DeepseekV4DecoderLayer` 的 attention 前后、FFN 前后都会出现。

模型 hidden states 不是简单的 `[tokens, hidden]`，它会扩成：

```python
hidden_states = hidden_states.unsqueeze(1).repeat(1, hc_mult, 1)
```

默认 `hc_mult=4`，所以很多层内计算的形状是 `[tokens, 4, hidden]`。每个 decoder layer 的 forward 大致是：

```text
residual = hidden_states
hidden_states, post, comb = hc_pre(..., input_layernorm)
hidden_states = self_attn(hidden_states)
hidden_states = hc_post(hidden_states, residual, post, comb)

residual = hidden_states
hidden_states, post, comb = hc_pre(..., post_attention_layernorm)
hidden_states = mlp(hidden_states)
hidden_states = hc_post(hidden_states, residual, post, comb)
```

`hc_pre` 有多条优化路径：

- TileLang：`SGLANG_OPT_USE_TILELANG_MHC_PRE=True`
- AITER/HIP：`SGLANG_OPT_USE_AITER_MHC_PRE=True`
- DeepGEMM TF32 prenorm：`SGLANG_OPT_DEEPGEMM_HC_PRENORM=True`
- fallback torch impl

最后一层还会经过 `hc_head`：

```python
pre_hc_head = hidden_states.flatten(1)
hidden_states = self.hc_head(hidden_states, hc_head_fn, hc_head_scale, hc_head_base)
hidden_states = self.norm(hidden_states)
```

这里 `pre_hc_head` 会传给 logits processor：

```python
hidden_states_before_norm=pre_hc_head
```

这也是 MTP 需要 auxiliary hidden states 的原因之一。NextN 模型会读取 target 传来的 `spec_info.hidden_states`，再做 `h_proj`，和当前 token embedding 的 `e_proj` 合并。

所以读 DeepSeek-V4 源码时，不要只盯着 attention。mHC 改变了 layer 内 hidden state 的形状、PP IPC 的传输形态，以及 MTP hidden state 的捕获方式。

# 0xC. MoE、FP4 和 MegaMoE

![](https://files.mdnice.com/user/59/ed675f32-1519-4d2a-b6da-6553b4b6b35a.png)

DeepSeek-V4 仍然是超大 MoE 模型，MoE 路径在 `DeepseekV4DecoderLayer` 里复用 `DeepseekV2MoE`，但会显式传入：

```python
is_deepseek_v4=True
```

还有一个很重要的限制：DeepSeek-V4 默认禁用 shared experts fusion。

```python
def determine_num_fused_shared_experts(self):
    self.num_fused_shared_experts = 0
    if disable_shared_experts_fusion:
        return

    disable_shared_experts_fusion = True
    log_info("DeepSeek V4 requires different clamping for shared and routed experts.")
```

原因写在日志里：DeepSeek-V4 的 shared experts 和 routed experts 需要不同 clamping，不能直接用旧的 shared experts fusion 假设。

FP4 专家权重的检测在 config 里：

```python
if dtype in ("U8", "I8", "F4"):
    return True
if dtype == "F8_E4M3":
    return False
```

部署上，Blackwell 默认走原始 FP4 experts + FP8 attention/dense 的混合 checkpoint；Hopper 可以走 FP8 converted checkpoint，也可以用 Marlin / FlashInfer MXFP4 跑原始 FP4 experts。

SGLang 里现在有两条 FlashInfer MXFP4 MoE 适配：

```text
python/sglang/srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py
python/sglang/srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py
```

`mxfp4_flashinfer_trtllm_moe.py` 里会把 topk ids 和 topk weights 打包，再调用 FlashInfer 的 TensorRT-LLM FP4 routed MoE：

```python
packed_topk = PackTopkIds.execute(topk_ids, topk_weights)
output = trtllm_fp4_block_scale_routed_moe(
    topk_ids=packed_topk,
    ...
)
```

`mxfp4_flashinfer_cutlass_moe.py` 则是 FlashInfer SM90 CUTLASS 路线。它也会处理 DeepSeek-V4 的 `swiglu_limit`，并把 FP4 block scale 布局整理成后端需要的格式。

MegaMoE 在源码里是另一条 MoE backend。入口在：

```text
python/sglang/srt/layers/moe/mega_moe.py
python/sglang/jit_kernel/csrc/deepseek_v4/mega_moe_pre_dispatch.cuh
```

MegaMoE 的前处理会把 hidden states、topk ids、topk weights 整理到 DeepGEMM 期望的 symmetric buffer：

```python
mega_moe_pre_dispatch(
    hidden_states,
    topk_ids_in,
    topk_weights_in,
    buf.x,
    buf.x_scales,
    buf.topk_idx,
    buf.topk_weights,
)
```

然后调用：

```python
deep_gemm.fp8_fp4_mega_moe(...)
```

如果打开：

```text
SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS=1
SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND=1
```

激活也会走 FP4 packed 路径，进一步减少 symmetric buffer footprint。

Cookbook 里对 MegaMoE 的限制也写得很清楚：

- 主要面向 Blackwell。
- 不支持 Hopper。
- 不支持 low-latency / CP recipe。
- 默认 W4A8，也可以用 W4A4。
- 需要根据 workload 调 `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK`。

MegaMoE 不是一个可随意开启的通用开关，它面向特定硬件和高吞吐 recipe。

# 0xD. CP 和 PD：DeepSeek-V4 的并行不是简单拼 flag

![](https://files.mdnice.com/user/59/b3e7fb87-c6d3-49d0-b0f7-f40067d89f91.png)

Slides 这页把 DP / TP / CP / EP / PD 放在一起，是因为 DeepSeek-V4 的并行维度确实强耦合。

CP 路径里，metadata 要先做 round-robin reindex：

```python
core_meta.apply_cp_reindex()
core_meta.init_flashmla_related()
metadata.indexer_metadata = init_forward_metadata_indexer(core_meta)
```

`apply_cp_reindex` 会把这些字段按 CP rank 切开：

```python
seq_lens_casual
positions_casual
swa_page_indices
swa_topk_lengths
page_table
c4_topk_lengths_raw
c4_topk_lengths_clamp1
c128_page_indices
c128_topk_lengths_clamp1
```

但这几个字段保持 global，不切：

```python
raw_out_loc
c4_out_loc
c128_out_loc
```

原因也写在源码注释里：compressor write path 仍然需要 global out loc。因此 CP 只切 attention 读取相关 metadata，写 cache 的位置信息必须保持全局语义。

模型层里 CP 会影响两个地方：

第一，attention 的 KV path 需要 BF16 KV 做 all-gather，所以不能走默认 fused cache write：

```python
kv = self._compute_kv_bf16(...)
kv = cp_all_gather_rerange_output(...)
attn_backend.store_cache(...)
```

第二，MLP / MoE 前要按 CP rank 切 input ids，并且要求 DeepEP：

```python
assert get_moe_a2a_backend().is_deepep()
input_ids = input_ids[cp_rank::cp_size].contiguous()
```

这解释了为什么 CP recipe 不是一个单独的 attention flag。它会同时影响 metadata、KV cache write、MLP input ids、DeepEP 后端和 TP/DP 组合。

PD disaggregation 也有 DeepSeek-V4 专门处理。Prefill 侧在设置 KVArgs 时，如果发现 token_to_kv_pool 是 `DeepSeekV4TokenToKVPool`，会额外带上 `mla_compression_ratios`：

```python
if isinstance(self.token_to_kv_pool, DeepSeekV4TokenToKVPool):
    kv_args.mla_compression_ratios = list(
        self.token_to_kv_pool.compression_ratios
    )
```

连接层收到这个字段后，会知道 DSv4 的 KV pointer list 不是 per-layer 布局，而是按 buffer type 分段：

```text
kv_data layout:
[c4 layers]
[c4 indexer layers]
[c128 layers]

state_data layout:
[swa layers]
[compress_state for c4/c128]
[indexer_compress_state for c4]
```

`_mla_slice_ptrs_for_pp` 会根据 `compression_ratios` 和 PP stage 的 start/end layer，把 decode 侧 full-model pointer list 切成和 prefill 侧一致的子范围。

这个限制也解释了 `deepseek_v4_hook.py` 里为什么要求 `PD disaggregation requires pp_size=1`。最新源码已经在 disaggregation common path 里支持 compressed-MLA pointer slicing，但 DSv4 的 buffer-type-organized KV 指针仍然比普通 per-layer KV 复杂很多，部署 recipe 不能随意组合。

![](https://files.mdnice.com/user/59/cda88364-d9fc-439c-9864-318e3166029f.png)

Cookbook 把 deployment recipe 分成几类：

- Low-latency：TP + MTP 3/4，优先压单请求延迟。
- Balanced：DP attention + DeepEP + MTP 1/2。
- Max-throughput：DP attention + DeepEP，通常关闭 MTP。
- CP：TP + DeepEP + context-parallel flags。
- PD-Disagg：Prefill / Decode 分离，通过 router 对外服务。

这些 recipe 给的是几条已验证的部署路线。实际选择时要看硬件和 workload：MTP、DeepEP、MegaMoE、CP、PD、Hopper/Blackwell、FP4/FP8 都会改变吞吐、延迟和显存占用。

# 0xE. 部署矩阵：硬件、模型变体和量化路线

DeepSeek-V4 cookbook 中当前主要有两个 instruct 变体：

```text
DeepSeek-V4-Flash：约 284B，总体更适合单节点部署
DeepSeek-V4-Pro：约 1.6T，需要更强的 TP/多节点/大显存组合
```

硬件矩阵在 `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx` 里维护。几个点：

```text
B200  -> FP4 weights，Flash TP=4，Pro TP=8
GB200 -> FP4 weights，Flash TP=4，Pro TP=8，两节点
GB300 -> FP4 weights，Flash TP=4，Pro TP=4
H200  -> FP8 converted checkpoint，Flash TP=4，Pro TP=16，两节点
H200 FP4 -> 原始 FP4 checkpoint + Marlin/FlashInfer MXFP4，TP-only
H100 FP4 -> 原始 FP4 checkpoint + Marlin，Flash TP=8，Pro TP=16
```

Cookbook 里有一个很实用的说明：DeepSeek 官方 Instruct repo 是 FP4 MoE experts + FP8 attention/dense 的混合 checkpoint；Base 变体是纯 FP8 mixed，但不是 chat/tool calling 用途。Hopper 如果不跑 FP4 mixed experts，则需要 SGLang 发布的 FP8 converted checkpoint：

```text
sgl-project/DeepSeek-V4-Flash-FP8
sgl-project/DeepSeek-V4-Pro-FP8
```

Recipe 生成器里还对 MegaMoE 做了显式 gating：

- 只支持 Blackwell。
- 不支持 H100 / H200 / H200-FP4。
- 不支持 low-latency / CP。
- 打开后会把 `--moe-a2a-backend deepep` 改成 `--moe-a2a-backend megamoe`。

这部分对于读源码也很有帮助：当你在源码里看到多个 MoE backend、多个 FP4 backend、多个 env 开关时，不要默认它们可以自由组合。真正支持的组合以 cookbook / deployment generator 里的 verified recipe 为准。

# 0xF. RL slides：训练侧信息怎么理解

![](https://files.mdnice.com/user/59/89b93932-c4d5-4464-a9ce-c9a9a8cd5e2a.png)

![](https://files.mdnice.com/user/59/a19e83fe-b44f-4fc6-b0cd-627c70cbac8b.png)

Slides 后半段讲了 Day-0 RL Support 和 DAPO 结果，关键词包括：

- DP / TP / SP / EP / PP / CP 完整并行。
- TileLang attention。
- Enhanced stability。
- FP8 training。

这部分我不展开源码，因为当前本地 SGLang 推理 repo 里最直接、最完整的是 serving runtime。训练侧 slides 主要说明 SGLang 相关工作围绕 DeepSeek-V4 覆盖的不只是 online serving，也包括模型发布后的训练/后训练接入。读者如果关注这部分，可以把它看成路线图和能力展示，不必和前面 `DeepseekV4AttnBackend` 一一对应。

# 0x10. 测试和验证入口

如果读者想顺着源码继续验证，可以从这些测试入口开始：

```text
test/registered/dsv4/test_deepseek_v4_flash_fp4_b200.py
test/registered/dsv4/test_deepseek_v4_flash_fp4_h200.py
test/registered/dsv4/test_deepseek_v4_flash_fp8_h200.py
test/registered/dsv4/test_deepseek_v4_flash_fp4_megamoe_b200.py
test/registered/distributed/test_disaggregation_dsv4.py

test/manual/dsv4/test_dsv4_flash_sanity_tp8.py
test/manual/dsv4/test_dsv4_flash_sanity_dp4.py
test/manual/dsv4/test_dsv4_flash_mtp_tp8.py
test/manual/dsv4/test_dsv4_flash_mtp_dp4.py
test/manual/dsv4/test_dsv4_pd_disagg_nixl.py

python/sglang/jit_kernel/tests/deepseek_v4/test_c4_v2.py
python/sglang/jit_kernel/tests/deepseek_v4/test_c128_v2.py
python/sglang/jit_kernel/tests/test_hisparse.py
```

这些测试大致覆盖：

- FP4 / FP8 模型在 B200 / H200 上的 serving。
- MegaMoE recipe。
- TP/DP 下的 sanity。
- MTP TP8 / DP4。
- PD disaggregation。
- C4/C128 compressor v2 kernel。
- HiSparse JIT kernel。

源码阅读时，建议按这样的顺序走：

```text
1. 看 cookbook 选一个 recipe。
2. 看 deepseek_v4_hook.py 确认启动参数会被改成什么。
3. 看 pool_configurator.py 算各类 pool 大小。
4. 看 model_runner_kv_cache_mixin.py 创建 DeepSeekV4TokenToKVPool。
5. 看 deepseek_v4_backend.py 的 init_forward_metadata。
6. 看 deepseek_v4.py 的 MQALayer.forward。
7. 根据 compress_ratio 分别看 compressor_v2.py / indexer.py。
8. 最后再看 JIT/CUDA kernel。
```

这个顺序比直接从 CUDA kernel 看起更容易，因为 DeepSeek-V4 的 kernel 大多依赖前面已经构造好的 page table、state loc、out loc、topk metadata。

# 0x11. Roadmap 和小结

![](https://files.mdnice.com/user/59/57573dff-9ee6-48c1-8124-4c37eb28bf2b.png)

Slides 的 roadmap 提到几个方向：

- Pipeline Parallelism + PD Disaggregation。
- FP4 Indexer。
- HiSparse 继续 offload SWA KV pool。
- DeepEP v2。
- 更多硬件支持：SM120、SM80。

这些方向都不是锦上添花。结合最新源码看，它们分别对应当前实现里的几个压力点：

- PP + PD：当前 DSv4 KV pointer list 是 buffer-type-organized，不是普通 per-layer 布局，PP 切分要依赖 `compression_ratios` 做复杂 slicing。
- FP4 Indexer：现在 C4 indexer 已经有 FP8 query / key cache 和 topk v2，继续低 bit 化会影响 sparse selection 的误差和性能。
- SWA offload：当前 HiSparse 先动 C4，因为 SWA 是最近窗口高频访问；要 offload SWA，延迟风险更大。
- DeepEP v2：CP、DP attention、MoE A2A、MegaMoE 都和 MoE 通信后端强相关。
- SM120 / SM80：DeepSeek-V4 目前很多默认优化明显偏 Blackwell/Hopper，老卡和新卡都需要单独 kernel/recipe 适配。

![](https://files.mdnice.com/user/59/24579f3e-62e0-46a6-8860-0fb0cbd4331c.png)

把实现路径整理成下面这张图：

```text
DeepSeek-V4 config
  -> compress_ratios: 0 / 4 / 128
  -> MQALayer: SWA-only / CSA / HCA

ShadowRadix / full-token coord
  -> DeepSeekV4TokenToKVPool
  -> SWA pool + C4 pool + C128 pool + C4 indexer pool

Forward metadata
  -> DSV4AttnMetadata
  -> SWA page indices
  -> C4 topk lengths + sparse page indices
  -> C128 page indices
  -> FlashMLA metadata

Layer forward
  -> fused Q norm + RoPE
  -> fused K norm + RoPE + SWA cache write
  -> C4/C128 compressor_v2
  -> C4 indexer + Lightning TopK
  -> flash_mla_with_kvcache(SWA + extra C4/C128)

System features
  -> MTP NextN uses SWA-only
  -> HiSparse swaps selected C4 pages
  -> mHC wraps attention and FFN
  -> MoE uses DeepEP / FlashInfer MXFP4 / MegaMoE
  -> CP / PD rely on DSv4-specific metadata and pointer layout
```

![](https://files.mdnice.com/user/59/15fb7233-2d08-4ad3-91cb-d233fc6c8f5b.png)

![](https://files.mdnice.com/user/59/04d020a2-d41c-4b3a-b192-1f3b71625065.png)

![](https://files.mdnice.com/user/59/e2587c9c-1b6d-492e-b6b2-c15d50126680.png)

![](https://files.mdnice.com/user/59/06216acc-50c4-46c8-b1e9-0c69d6dd4b08.png)

SGLang 对 DeepSeek-V4 的实现可以按这条线理解：先用 ShadowRadix 和多级 KV pool 管住新的 attention 状态，再用 metadata planner、Flash Compressor、Lightning TopK、多流 overlap 和 FlashMLA 执行这些状态，最后通过 cookbook recipe 给不同硬件和 workload 提供可复用的组合。DeepSeek-V4 难就难在这里：单个 kernel 看起来都能解释，真正上线时这些层要一起对齐。
