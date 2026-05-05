> 这篇是对 2026 年 1 月 17 日蚂蚁开源 x SGLang Meetup 中「蚂蚁 Theta 面向 DeepSeek 系列模型的深度优化和实践」这个分享的回放解读。原始 slides 讲的是一整套 H20-96G 上部署 DeepSeek-R1/V3/V3.1/V3.2 的工程优化，这里我尽量把每一页背后的 SGLang PR、DeepEP/DeepGEMM/FlashMLA 代码路径和实现细节串起来。

# 0x0. 前言

![](https://files.mdnice.com/user/59/8c5ccd59-675b-48d0-b655-e84ba48fa0f5.png)

这次分享的主角是蚂蚁 Theta 团队在 H20-96G 上做 DeepSeek 系列模型推理优化的实践。它不是单点 kernel 优化，而是从部署形态、Prefill、Decode、MoE 通信、Expert Load Balance、投机解码、观测诊断，到 DeepSeek-V3.2 DSA 支持的一整套方案。

我读完 slides 和公开 PR 后的感觉是，这套优化真正有意思的地方不在某一个 trick，而在它把 H20 的硬件特性吃得很细：H20 算力弱于 H800，但显存容量、带宽、NVLink 都不差，所以 Prefill 和 Decode 不能按一个固定套路去做，必须拆开部署，再按瓶颈分别优化。

![](https://files.mdnice.com/user/59/4faacfe1-ed81-4160-9606-9a26affa1ad2.png)

分享嘉宾张天雨（墨纭）是蚂蚁 Theta 推理云 DeepSeek 系列模型的迭代负责人。从公开 PR 看，这次 slides 里的不少优化都能在 SGLang 和蚂蚁 fork 的 PR 里找到对应实现。一个很重要的入口是 AntGroup 这个部署汇总 PR：[Deploying DeepSeek-R1 on H20-96G with SGLang: Best Practices](https://github.com/antgroup/sglang/pull/4)。它不是为了 merge，而是把复现实验的镜像、启动参数、profile 链接和相关 PR 都放在一起，后面很多线索都是从这里展开的。

这套优化也对应 LMSYS 上的那篇博客：[Together with SGLang: Best Practices for Serving DeepSeek-R1 on H20-96G](https://www.lmsys.org/blog/2025-09-26-sglang-ant-group/)。博客里把 H20 的挑战、Prefill/Decode 分离、FP8 FlashMLA、SwapAB、SBO、Expert Affinity EPLB、DeepXTrace 这些点按生产部署视角串了一遍；这篇文章则主要沿着 slides 和 PR 代码，把实现细节展开。

![](https://files.mdnice.com/user/59/9892fed6-51c3-460f-9a6a-604cab09432c.png)

Slides 的目录分成四块：Challenges、Methodology、Evaluation & Conclusion、DeepSeek V3.2。下面我也按这个顺序来讲，但重点会放在 Methodology 里面的代码实现。

# 0x1. H20-96G 的约束

![](https://files.mdnice.com/user/59/41b557a8-d38a-4916-924d-cba07392b333.png)

这页先把 H20 和 H800 的硬件差异摆出来。H20-96G 相比 H800-80G 的核心特点是：

- FP8 / BF16 峰值算力只有 H800 的大约 15%；
- 显存容量 96GB，比 H800 的 80GB 更大；
- 显存带宽 4000GB/s，比 H800 的 3352GB/s 更高；
- NVLink 带宽 900GB/s，比 H800 的 400GB/s 高很多；
- RDMA 网卡带宽只有 H800 的一半。

所以它不是「全面更弱」的卡，而是一张很偏科的卡。算力弱，但是显存和单机互联条件不错。这个偏科直接决定了后面的部署策略：

- Prefill 阶段更吃 attention、长上下文和 TTFT，需要控制单请求延迟；
- Decode 阶段更吃小 batch 下的 MoE、跨卡通信和 TPOT，需要控制每 token 的稳定延迟；
- 跨节点 RDMA 是短板，能留在节点内 NVLink 的东西尽量留在节点内；
- 显存够大，可以把 Decode 做成更大的 DP/EP 形态，换更小的故障域和更好的吞吐。

# 0x2. Prefill/Decode 分离

![](https://files.mdnice.com/user/59/561462d0-d4da-441e-a1e1-0fed15175e16.png)

![](https://files.mdnice.com/user/59/c8a934ae-e642-4f74-bde4-8f752003f477.png)

这里的部署策略是典型的 PD disaggregation：

- Prefill 使用 TP8。目标是满足 TTFT 约束，并且 Prefill 节点可以按流量弹性扩缩。
- Decode 使用 DP16 + EP16。H20-96G 显存更大，NVLink 带宽也高，比较适合把 Decode 的注意力做 DP，把 MoE 做 EP，减少不必要的 TP 通信。

Ant 的复现 PR 里给了非常具体的启动参数。Prefill 侧大致是：

```bash
PYTHONUNBUFFERED=1 \
SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0 \
python3 -m sglang.launch_server \
  --model-path /path/to/DeepSeek-R1 \
  --disaggregation-mode prefill \
  --tp-size 8 \
  --attention-backend fa3 \
  --chunked-prefill-size 16384 \
  --quantization fp8 \
  --kv-cache-dtype fp8_e4m3
```

Decode 侧的关键参数是：

```bash
PYTHONUNBUFFERED=1 \
SGL_ENABLE_JIT_DEEPGEMM=1 \
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=96 \
ENABLE_SWAPAB=1 \
python3 -m sglang.launch_server \
  --model-path /path/to/DeepSeek-R1 \
  --disaggregation-mode decode \
  --attention-backend flashmla \
  --nnodes 2 \
  --tp-size 16 \
  --dp-size 16 \
  --enable-dp-attention \
  --moe-dense-tp-size 1 \
  --enable-deepep-moe \
  --enable-dp-lm-head \
  --cuda-graph-max-bs 48 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 2 \
  --init-expert-location /root/expert_workload.json \
  --moe-a2a-backend deepep \
  --deepep-mode low_latency_overlap \
  --enable-single-batch-overlap
```

这组参数基本把后面的优化都串起来了：Prefill 侧有 `fa3`、chunked prefix、TP scattered input；Decode 侧有 `flashmla`、DP attention、DeepEP low latency、SwapAB、SBO、Expert location 初始化和 NEXTN/Eagle。

# 0x3. Prefill 优化

![](https://files.mdnice.com/user/59/4aace48a-3ee3-4ecd-bacb-9e5fa94cbb8a.png)

Prefill 这页列了三个主要瓶颈：

- TP 通信后紧跟 RMSNorm 和 `qkv_a`，原始路径里很多算子在处理完整 hidden；
- chunked prefix 下，MLA/MHA 的选择并不总是 MLA 更优；
- MoE 计算量虽然更小，但下投影的延迟不合理。

对应的公开 PR 主要是这三个：

- [#10568 Opt tp: tp attn support tp reduce scattered input](https://github.com/sgl-project/sglang/pull/10568)
- [#10953 Opt MHA chunked prefix: merge prefix and extend kv cache to run mha once](https://github.com/sgl-project/sglang/pull/10953)
- [#10567 Opt fused triton moe: add tma for down proj kernel](https://github.com/sgl-project/sglang/pull/10567)

## 0x3.1 TP Reduce Scatter + RMSNorm + qkv_a

PR [#10568](https://github.com/sgl-project/sglang/pull/10568) 的核心很直接：原来是

```text
embed/mlp all reduce + RMSNorm + fused_qkv_a_proj_with_mqa
```

优化后变成：

```text
embed/mlp reduce scatter + RMSNorm + fused_qkv_a_proj_with_mqa + all gather
```

为什么这样能省？因为在 TP8 下，`RMSNorm` 和 `fused_qkv_a_proj_with_mqa` 原来要处理完整 hidden，现在先 reduce-scatter，每张卡只处理 1/8 的 token 分片。后面再 all-gather 的时候，最后一维已经从 hidden size 7168 变成了 `(q_lora_rank + kv_lora_rank + qk_rope_head_dim)`，也就是 `1536 + 512 + 64`，通信量明显小了。

PR 里的 16K chunked prefill profile 数据很有代表性：

- `fused_qkv_a_proj_with_mqa` 从 205.1ms 降到 26.14ms；
- 通信总延迟从 267.1ms 降到 249.63ms；
- `RMSNorm` 从 82.303ms 降到 43.398ms；
- 输入长度 1000/2000/4000/4096 下，请求吞吐分别从 12.82/6.52/2.49/2.41 req/s 提到 14.22/7.33/2.72/2.63 req/s。

当前 SGLang 代码里，这个优化由 `--enable-attn-tp-input-scattered` 控制。`AttnTpContext` 会检查一组约束，只有 DeepSeek MLA 这类 `q_lora_rank` 不为空、TP 大于 1、没有 DP attention、没有 MoE A2A、没有 EAGLE3 等条件满足时才启用：

```python
class AttnTpContext:
    def init_context(self, q_lora_rank, is_nsa):
        self.allow_input_scattered = (
            get_global_server_args().enable_attn_tp_input_scattered
            and (_is_cuda or _is_npu)
            and q_lora_rank is not None
            and not is_nsa
            and get_tensor_model_parallel_world_size() > 1
            and not is_dp_attention_enabled()
            and get_moe_a2a_backend().is_none()
            and not enable_moe_dense_fully_dp()
            and get_global_server_args().disable_piecewise_cuda_graph
            and get_global_server_args().speculative_algorithm != "EAGLE3"
        )

    def use_input_scattered(self, forward_batch: ForwardBatch):
        return (
            self.allow_input_scattered
            and forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_target_verify()
            and not forward_batch.forward_mode.is_draft_extend()
            and forward_batch.input_ids is not None
            and not forward_batch.can_run_tbo
        )
```

后面 attention 真正需要完整 hidden 时，再通过 `fetch_hidden_states()` 做一次 TP all-gather：

```python
def fetch_hidden_states(self):
    if self.hidden_states_ is not None:
        return self.hidden_states_
    self.hidden_states_ = self.hidden_states_local
    if get_attn_tp_context().input_scattered:
        self.hidden_states_ = self.tp_all_gather_hidden_states(
            self.hidden_states_, self.forward_batch
        )
    return self.hidden_states_
```

这个优化属于「不改变数学，只改变中间张量放在哪个时刻变完整」的类型，收益也正来自这个时序调整。

## 0x3.2 Chunked Prefix 下的 MHA One-Shot

PR [#10953](https://github.com/sgl-project/sglang/pull/10953) 解决的是另一个 Prefill 问题：DeepSeek 的 MLA 在长上下文上很香，但 chunked prefix cache 下面，如果 prefix 和 extend 分开跑 MHA，再用 `merge_state` 合并，中间会有多次拷贝、类型转换和额外 attention 调用。这个 PR 的策略是：当 `seq_lens <= 128K` 时，把 prefix KV 和 extend KV 合起来，只跑一次 `attn_mha`。

PR 里建议直接通过环境变量切到 MHA 路径：

```bash
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
```

当前代码里，attention backend handler 会在 `fa3`、`flashinfer`、`flashmla` 这些 backend 上判断是否能走 `MHA_ONE_SHOT`：

```python
MHA_ONE_SHOT_SUPPORTED_BACKENDS = ["fa3", "flashinfer", "flashmla"]

def _support_mha_one_shot(attn, forward_batch, backend_name):
    attn_supported = backend_name in MHA_ONE_SHOT_SUPPORTED_BACKENDS
    sum_seq_lens = (
        sum(forward_batch.seq_lens_cpu) if forward_batch.seq_lens_cpu is not None else 0
    )
    return attn_supported and sum_seq_lens <= forward_batch.get_max_chunk_capacity()

def _handle_attention_backend(attn, forward_batch, backend_name):
    ...
    if forward_batch.forward_mode.is_extend_without_speculative():
        if _support_mha_one_shot(attn, forward_batch, backend_name):
            return AttnForwardMethod.MHA_ONE_SHOT
        return AttnForwardMethod.MHA_CHUNKED_KV
    else:
        return _dispatch_mla_subtype(attn, forward_batch)
```

`forward_mha.py` 里把三种路径说得很清楚：

```python
# 1. forward_normal: AttnForwardMethod.MHA
#    use multi-head attention with empty kv cache
#
# 2. forward_normal_one_shot: AttnForwardMethod.MHA_ONE_SHOT
#    use multi-head attention with short kv prefix length
#    the kv latent vectors are fetched from memory pool,
#    with combined kv_indices of prefix part and extended part
#
# 3. forward_normal_chunked_kv: AttnForwardMethod.MHA_CHUNKED_KV
#    multiple phases of multi-head attention with chunked kv cache
#    acc_o_i, acc_lse_i = merge_state(...)
```

这个 PR 的 benchmark 也很说明问题。在一个 prefix/extend 交错的例子里：

- BF16 下 MLA 是 117us，优化前 MHA chunked KV 是 193us，MHA merged KV 是 101us；
- FP8 下 MLA 是 373us，其中包含 244us 的 FP8 KV cache cast；MHA chunked KV 是 227us，MHA merged KV 是 125us。

所以这页 slides 里说「MLA vs MHA tuning optional by seq len」并不是泛泛而谈。短一些、能塞进 one-shot 的时候，MHA 反而更便宜。

## 0x3.3 FusedMoE Down Projection TMA

这部分我之前单独写过一篇：[SGLang 优化Triton FusedMoE 的一个新技巧](./SGLang%20优化Triton%20FusedMoE%20的一个新技巧%E2%80%8B.md)。这里直接沿用那篇的主体思路。

在 H20(96GB) TP8 prefill 性能分析时，作者发现一个很奇怪的现象：每层第二个 MoE，也就是 down projection 的 Fused Triton MoE，延迟竟然和第一个 up projection 差不多。但 down projection 的权重数据量和计算量只有 up projection 的一半，这显然不合理。

PR [#10567](https://github.com/sgl-project/sglang/pull/10567) 对这个问题做了几个优化：

- 优化 FP8 block quant 下 `b_scale` 的读取和计算；
- 基于 TMA 重构 down projection 的输入 A 和权重 B 访问；
- 用真实推理过程采集到的 `topk_ids` 做 tuning；
- up projection 和 down projection 分别加载不同的 tuned config。

PR 给出的关键数字是：down projection 的计算利用率从 **45.20% 提升到 81.12%**，8K tokens 场景下 100 次采样的平均延迟从 **2.430ms 降到 1.435ms**。

真实 topk tuning 的流程也很实用。先在推理过程中存下每层的 `topk_ids`：

```python
# DeepseekV2MoE::forward_normal
if hidden_states.shape[0] == 16384 and get_tensor_model_parallel_rank() == 0:
    topk_ids_dir = xxxx
    if not hasattr(self, "save_idx"):
        self.save_idx = 0
    if self.save_idx <= 1:
        torch.save(
            topk_output.topk_ids,
            f"{topk_ids_dir}/topk_idx_layer{self.layer_id}_idx{self.save_idx}.pt",
        )
    self.save_idx += 1
```

然后用 `tuning_fused_moe_triton_sep.py` 对真实分布做 tuning：

```bash
python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py \
    --model $model_path \
    --tp-size 8 \
    --dtype fp8_w8a8 \
    --topk-ids-dir ${topk_ids_dir} \
    --tune
```

这会生成两套 config，一套给 up projection，一套给 down projection，后者文件名带 `_down`。

当前 SGLang 主线代码已经挪到了 `moe_runner/triton_utils` 下面。核心逻辑是 `_prepare_fused_moe_run` 同时拿普通 config 和 down config，然后从 down config 里取 `USE_TMA`：

```python
config, (down_config, _) = try_get_optimal_moe_config(
    w1.shape,
    (w2.shape[0], w2.shape[1], w2.shape[2] - padded_size),
    topk_ids.shape[1],
    config_dtype,
    num_tokens,
    block_shape=block_shape,
    per_channel_quant=per_channel_quant,
    return_down_config=True,
)
down_moe_use_tma = (
    _down_moe_use_tma()
    and down_config is not None
    and down_config.pop("USE_TMA", False)
)
```

如果启用 TMA，up projection 的输出会按 down projection 想要的顺序写好：

```python
invoke_fused_moe_kernel(
    hidden_states,
    w1,
    ...,
    config,
    ...,
    c_sorted=down_moe_use_tma,
)
```

随后 down projection 的 kernel 调用会把输入 A 和权重 B 都交给 TMA descriptor 路径：

```python
invoke_fused_moe_kernel(
    intermediate_cache2,
    w2,
    ...,
    down_config or config,
    ...,
    a_use_tma=down_moe_use_tma,
    b_use_tma=down_moe_use_tma,
)
```

这就是这页 slides 里「MoE down_proj with TMA, tuned configs」背后的关键。不是单纯加一个 TMA flag，而是先用真实 expert 分布做 tuning，再让 up kernel 的输出布局服务于 down kernel 的 TMA 访问。

![](https://files.mdnice.com/user/59/b96daf49-3cfd-485e-bdcc-e76d49887db9.png)

Prefill 评测这页的提升也和前面的三类优化能对上：输入越长，attention 和 MoE 在 TTFT 里的占比越大，所以收益越明显。Slides 给出的整体提升是：

- 1K 输入提升 34%；
- 2K 输入提升 45%；
- 4K 输入提升 68%。

# 0x4. Decode 优化一：SwapAB GEMM

![](https://files.mdnice.com/user/59/e5249b49-ddcd-4aa1-9e57-32eed33095d7.png)

Decode 的 MoE 和 Prefill 不一样。Prefill 往往 token 多，GEMM 的 M 比较大；Decode 是小 batch，每次进 MoE 的 token 数很少。Hopper WGMMA 的 `block_m` 常见粒度是 64，当实际 M 小于 64 时，会做很多无效计算。

这页 slides 说的 SwapAB，本质就是把原来 GEMM 里的小 M 维映射到 WGMMA 更舒服的维度上。DeepGEMM 侧对应 PR 是 [deepseek-ai/DeepGEMM#192](https://github.com/deepseek-ai/DeepGEMM/pull/192)，标题就是 `support swapAB for m_grouped_fp8_gemm_nt_masked`。PR 描述里写得很直白：

- 对 `BLOCK_M = 32` 或 `M % 64 < 32` 的情况收益明显；
- 做法是 `Swap A B: WGMMA::wgmma(desc_b, desc_a, accum, k)`；
- H20 上 `BLOCK_N=256` 是一个重要配置；
- 通过 `export ENABLE_SWAPAB=1` 开启。

SGLang 主线里也有对应实现链路：

- [#15712 Add SwapAB Optimization for triton fused_moe_kernel on SM90](https://github.com/sgl-project/sglang/pull/15712)
- [#16723 Rework Add SwapAB Optimization for triton fused_moe_kernel on SM90](https://github.com/sgl-project/sglang/pull/16723)
- [#17133 Optimize fused moe configs for H20 & H20-3E based on swapab](https://github.com/sgl-project/sglang/pull/17133)
- [#17965 Triton TP MoE Dpsk V3/Qwen3 Coder with SwapAB](https://github.com/sgl-project/sglang/pull/17965)

当前 Triton FusedMoE kernel 里，是否启用 SwapAB 的判断非常克制：

```python
# swap_ab benefits SM90 GPUs (H20, H100, H200, etc.) for certain block shapes.
@functools.lru_cache(maxsize=8)
def should_enable_swap_ab(
    BLOCK_SIZE_M: int,
    BLOCK_SIZE_N: int,
) -> bool:
    if not _is_cuda or is_batch_invariant_mode_enabled():
        return False

    return is_sm90_supported() and BLOCK_SIZE_M < 64 and BLOCK_SIZE_N >= 64
```

也就是说，只有 SM90 且 `BLOCK_SIZE_M < 64`、`BLOCK_SIZE_N >= 64` 时才走这条路。进入 kernel 后，accumulator 的 shape 会反过来：

```python
if swap_ab:
    accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
else:
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
```

FP8 dot 前，A/B 和 scale 也会交换：

```python
if swap_ab:
    a, b = tl.trans(b, (1, 0)), tl.trans(a, (1, 0))
    a_scale, b_scale = b_scale, a_scale
...
accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
```

最后再把 accumulator 转置回来写出：

```python
if swap_ab:
    accumulator = tl.trans(accumulator, (1, 0))
```

这个实现很短，但它背后的点很关键：Decode 小 batch 下，不要让实际很小的 token 维硬贴着 WGMMA 的 M 维跑。把 A/B 互换后，小 M 的浪费会少很多。

PR [#17965](https://github.com/sgl-project/sglang/pull/17965) 里给了 H200 上的端到端数据。DeepSeek-V3.1 TP8 下，2048 tokens decode 从 18.790s 降到 17.074s，速度从 109.00 token/s 提到 119.95 token/s，差不多 8% 的提升。Qwen3 Coder 的 spec decode 场景里也从 172.27 token/s 提到 186.87 token/s。

## 0x4.1 Decode Attention：FP8 FlashMLA

Decode 启动参数里用的是 `--attention-backend flashmla`，Ant 的汇总 PR 也把 [deepseek-ai/FlashMLA#82](https://github.com/deepseek-ai/FlashMLA/pull/82) 单独列为 FP8 MLA 相关优化。这个 PR 的主题是新的 FP8 MLA pipeline。

它和之前的 BF16 / FP8 MLA 相比，主要做了几件事：

- 使用 WGMMA FP8，Q 和 KV 都走 FP8 dtype；
- 减少 shared memory 占用，重新组织 `sP0/sP1/sVt0/sVt1` 做 ping-pong；
- 对 TMA copy 和 WGMMA 之间的 pipeline 做更细粒度调度；
- 对 transposed V 重建 pipeline，用 4 个 named barriers 在 4 个 buffer 间切换；
- 用 128-bit STSM/LDSM 在 `rP` 和 `sP` 之间搬数据；
- 细粒度 QK tiles 让 ROPE 可以用 BF16 算，从而修掉之前 FP8 路径的精度问题。

代码层面主要新增了 `csrc/sm90/kernels/splitkv_mla_fp8.cu`，同时改了 `traits.h`、`utils.h`、`fp8_transpose_v.h` 和 `flash_mla_interface.py`。这几个文件对应的就是 kernel traits、FP8 transposed V、SM90 工具函数和 Python 入口。

PR 给的 H20 数据里，cache length=8196、head_num=64 时，batch size 32/48/64/128 下，新 FP8 MLA 相比 BF16 MLA 的提升分别是 69%/62%/62%/74%，相比之前 FP8 PR 也还有约 5% 的提升。Decode 侧小 batch 的 MoE 之外，attention backend 这条线也很重要，否则 MoE 优化完以后瓶颈会很快转移到 MLA。

# 0x5. Decode 优化二：为什么不用 TBO

这页在解释为什么 H20 Decode 上 Two-Batch Overlap(TBO) 不理想。原因有两个：

1. Hopper WGMMA 的 `block_m` 通常是 64，小 batch Decode 下 MLP GEMM 会有冗余计算；
2. TBO 要在 batch 足够大时才容易转正收益，但 H20 算力弱，batch 大了以后 TPOT/ITL 的 SLA 又容易炸。

所以 slides 里说 TBO 不适合这个在线服务场景。公开实现对应到 [#9660 Single Batch Overlap for MoE Models](https://github.com/sgl-project/sglang/pull/9660)。这个 PR 的 Motivation 和 slides 几乎是一致的：小 batch 时 TBO 的正收益不够稳定，需要一个对单 batch 也有效的 overlap。

# 0x6. Decode 优化三：SBO

![](https://files.mdnice.com/user/59/193194d9-dc59-4e08-af1d-f8e1f4f31c50.png)

SBO 做了两个 overlap：

1. Shared Expert 计算和 Dispatch Recv 通信 overlap；
2. Down GEMM 计算和 Combine Send 通信 overlap。

这条链路涉及三个仓库：

- SGLang 集成 PR：[sgl-project/sglang#9660](https://github.com/sgl-project/sglang/pull/9660)
- DeepEP 通信侧 PR：[deepseek-ai/DeepEP#390](https://github.com/deepseek-ai/DeepEP/pull/390)，SGLang PR 正文里也提到了后续 DeepEP [#483](https://github.com/deepseek-ai/DeepEP/pull/483)
- DeepGEMM 计算侧 PR：[deepseek-ai/DeepGEMM#183](https://github.com/deepseek-ai/DeepGEMM/pull/183)，SGLang PR 正文里也提到了 `sgl-project/DeepGEMM#14`

Down GEMM 和 Combine Send 的 overlap 是一个 producer-consumer 模型：每个 local expert 按 `block_m` token 粒度分配 signal。Down GEMM 算完某个 `block_m` 后用 atomic 更新 signal；Combine Send 轮询 signal，达到 threshold 后就把对应 token 发出去。

当前 SGLang 的 `single_batch_overlap.py` 负责计算 overlap 参数。这里有几个关键字段：

```python
@dataclass
class CombineOverlapArgs:
    # this "overlap" flag means overlapping with down gemm
    overlap: bool
    stream: torch.cuda.Stream
    wait_event: torch.cuda.Event
    num_sms: Optional[int] = None
    signal: Optional[torch.Tensor] = None
    block_m: Optional[int] = 64
    threshold: Optional[int] = 0

@dataclass
class DownGemmOverlapArgs:
    num_sms: int
    signal: torch.Tensor
    start_event: torch.cuda.Event
```

`compute_overlap_args` 会把 SM 分成通信和计算两部分，Hopper 上默认通信用 3 个 SM，剩下给 DeepGEMM：

```python
if envs.SGLANG_DEEPEP_LL_COMBINE_SEND_NUM_SMS.is_set():
    communicate_num_sms = envs.SGLANG_DEEPEP_LL_COMBINE_SEND_NUM_SMS.get()
else:
    communicate_num_sms = 32 if is_blackwell() else 3
compute_num_sms = total_num_sms - communicate_num_sms
```

如果启用 Down GEMM + Combine Send overlap，就创建 signal，并把 signal 同时传给 combine 和 down gemm：

```python
combine_signal_size = num_local_experts * (
    (num_tokens_static + MIN_BLOCK_M - 1) // MIN_BLOCK_M
)
combine_signal = torch.zeros(
    combine_signal_size, dtype=torch.int32, device=hidden_states.device
)

down_gemm_overlap_args = DownGemmOverlapArgs(
    signal=combine_signal,
    start_event=combine_wait_event,
    num_sms=compute_num_sms,
)
combine_overlap_args.overlap = True
combine_overlap_args.signal = combine_signal
combine_overlap_args.threshold = compute_num_sms
```

在 `DeepseekV2MoE` 里，SBO 是通过 hook 接入 DeepEP dispatcher 的。Dispatch 后计算 overlap args，并分别塞给 dispatcher 和 experts runner：

```python
def _post_dispatch_hook(dispatcher: BaseDispatcher, dispatch_output: DispatchOutput):
    combine_overlap_args, down_gemm_overlap_args, meta_overlap_args = (
        compute_overlap_args(dispatch_output, self.alt_stream)
    )
    dispatcher.set_overlap_args(
        combine_overlap_args=combine_overlap_args,
        meta_overlap_args=meta_overlap_args,
    )
    self.experts.set_overlap_args(
        down_gemm_overlap_args=down_gemm_overlap_args,
        meta_overlap_args=meta_overlap_args,
    )
```

DeepEP combine 侧会把这些参数传给 `low_latency_combine`：

```python
overlap_args_dict = dict(
    overlap=overlap_args.overlap,
    packed_recv_count=self.packed_recv_count,
    comp_signal=overlap_args.signal,
    block_m=meta_overlap_args["block_m"],
    threshold=meta_overlap_args["threshold"],
    num_sms=overlap_args.num_sms,
)

combined_hidden_states, event, hook = buffer.low_latency_combine(
    x=hidden_states,
    topk_idx=topk_ids,
    topk_weights=topk_weights,
    handle=self.handle,
    async_finish=not self.return_recv_hook,
    return_recv_hook=self.return_recv_hook,
    **overlap_args_dict,
)
```

DeepGEMM runner 侧则会从 signal GEMM 的返回值里拿到动态 `block_m` 和 `threshold`，再写回 `meta_overlap_args`，供 combine 使用：

```python
deep_gemm_return_value = deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
    (down_input, down_input_scale),
    (w2_weight, w2_scale),
    down_output,
    masked_m,
    expected_m,
    **gemm_overlap_args_dict,
)
meta_overlap_args = running_state.get("meta_overlap_args", None)
if meta_overlap_args is not None:
    block_m, threshold = deep_gemm_return_value
    meta_overlap_args["block_m"] = block_m
    meta_overlap_args["threshold"] = threshold
```

DeepEP PR [#390](https://github.com/deepseek-ai/DeepEP/pull/390) 里对应的是 `low_latency_dispatch` 记录 token 的 `src_rank`，以及 `internode_ll::combine` 在 overlap 模式下减少 SM、轮询 signal、发送 token、写 finish flag。DeepGEMM PR [#183](https://github.com/deepseek-ai/DeepGEMM/pull/183) 里对应的是 `m_grouped_fp8_gemm_nt_signal`、`SM90FP8SignalGemm1D2DRuntime`，以及 kernel 在算完对应 `block_m` 后 `atomicAdd` 写 signal。

PR [#9660](https://github.com/sgl-project/sglang/pull/9660) 的端到端评测是 5 节点、每节点 8 张 H20，Prefill TP8，Decode DP_Attn16 + EP16，输入 4096、输出 1536。bs 32 下输出吞吐从约 6667 tok/s 提到 7111/7169 tok/s，请求吞吐从 4.34 req/s 提到 4.63/4.67 req/s，平均 ITL 从 73.1ms 降到 67ms 左右。这个收益对在线 Decode 很实在。

# 0x7. Expert Affinity EPLB

![](https://files.mdnice.com/user/59/526bf9fd-6452-4c1b-9651-4ac9b134b074.png)

DeepSeek 这类 MoE 模型的 Expert Load Balance 不只是「每张卡算力均衡」的问题。标准 EPLB 会尽量把 expert 的计算负载打平，但如果经常一起被激活的 experts 被放到了不同节点，就会制造更多跨节点通信。H20 的 RDMA 带宽又是短板，这个问题就会被放大。

AntGroup 的 [#2 feat: Add Expert Affinity Aware EPLB algorithm](https://github.com/antgroup/sglang/pull/2) 正是做这件事。它在已有 expert load tracking 的基础上，进一步记录每次 iteration 激活的 top-k expert groups，计算 expert co-activation 的 affinity matrix，然后在 EPLB 负载均衡后再根据 affinity 调整放置，尽量把经常一起出现的 experts 放到同一个节点。

PR 新增的 `comm_matrix_process.py` 很短：

```python
def compute_expert_co_occurrence_matrix(history_data, num_experts):
    history_data = history_data.cpu().numpy()
    num_samples, num_layers, top_k = history_data.shape
    expert_co_occurrence = np.zeros(
        (num_layers, num_experts, num_experts), dtype=np.int64
    )

    for sample_idx in range(num_samples):
        for layer_idx in range(num_layers):
            experts = history_data[sample_idx, layer_idx]
            if (-1 in experts) or (len(set(experts)) < top_k):
                continue
            for i in range(top_k):
                for j in range(i + 1, top_k):
                    expert_i = experts[i]
                    expert_j = experts[j]
                    if expert_i < num_experts and expert_j < num_experts:
                        expert_co_occurrence[layer_idx, expert_i, expert_j] += 1
                        expert_co_occurrence[layer_idx, expert_j, expert_i] += 1
    return torch.tensor(expert_co_occurrence, dtype=torch.int64)

def generate_comm_matrix(history_data, num_experts):
    if history_data.numel() == 0:
        return None
    co_occurrence = compute_expert_co_occurrence_matrix(history_data, num_experts)
    comm_matrix = co_occurrence.float()
    comm_matrix = comm_matrix / comm_matrix.max()
    return comm_matrix
```

也就是把同一 token 的 top-k experts 两两计数，再归一化成通信矩阵。

数据来源在 DeepEP dispatcher。PR 在 `dispatch_a` 里记录 `topk_idx`：

```python
if topk_idx.numel() > 0:
    get_global_expert_distribution_recorder().record_topk_ids(topk_idx)
else:
    logger.warning("topk_idx is empty in DeepEP low latency dispatch.")
```

放置优化的核心是 `optimize_group_placement`。它先把 physical experts 分 group，构造 group-to-node 映射，再用每个 group 的 leader expert 去查 communication cost。后面会尝试交换不同节点上的 group，只要 swap 能降低跨节点通信 cost，就执行：

```python
if best_gain > 0 and best_swap:
    node1, g1_idx, node2, g2_idx = best_swap
    g1 = node_groups[node1][g1_idx]
    g2 = node_groups[node2][g2_idx]

    node_groups[node1][g1_idx] = g2
    node_groups[node2][g2_idx] = g1

    for offset in range(group_size):
        idx1 = g1 * group_size + offset
        idx2 = g2 * group_size + offset
        optimized_pphy2log[layer, idx1], optimized_pphy2log[layer, idx2] = \
            optimized_pphy2log[layer, idx2].item(), optimized_pphy2log[layer, idx1].item()

    improved = True
```

这个 PR 给出的 benchmark 里，batch 1536/2048 时，Expert-Affinity Aware EPLB 相比 vanilla EPLB 的 P90-TPOT 从 84ms 左右降到 81ms 左右，P95-ITL 从 100ms 左右降到 97ms 左右。PR 里写的是相对标准 EPLB 额外约 5% 提升。

还有一个相关 PR 是 [sgl-project/sglang#8529](https://github.com/sgl-project/sglang/pull/8529)，做的是 EPLB async rebalance。它通过后台线程广播 `logical_count`、计算 `ExpertLocationMetadata`，再用 TP barrier 和 gloo CPU signal 保证所有 rank 原子切换 expert location。这和 affinity placement 是互补关系：一个优化放置策略，一个优化 rebalance 的执行方式。

# 0x8. Hierarchical Dispatch

![](https://files.mdnice.com/user/59/79e5ef2a-cc88-42f3-a7ca-4a2a7db7e232.png)

这页 slides 讲的是 Hierarchical Low-latency Dispatch：原始低延迟 dispatch 是所有 rank 直接走 inter-node RDMA，高流量 RDMA 会把延迟打高；hierarchical dispatch 的想法是先跨节点 RDMA 做一级转发，再在节点内通过 NVLink 做二级转发。结合前面 H20 的硬件特性，这个方向非常自然，RDMA 是短板，NVLink 是长板。

不过这页我没有找到对应的公开 PR。这里我扫了 `sgl-project/sglang`、`antgroup/sglang`、`deepseek-ai/DeepEP`、DeepGEMM 相关仓库，按 `Hierarchical Dispatch`、`RDMA-NVLink`、`hierarchical dispatch`、`low_latency_dispatch` 等关键词找了一遍，没有找到和 slides 这页完全对应的公开实现。

目前公开代码里能对上的基础路径是 SGLang 的 DeepEP low-latency dispatcher，也就是 Decode 侧 `--moe-a2a-backend deepep --deepep-mode low_latency_overlap` 这条路。核心调用还在 `token_dispatcher/deepep.py`：

```python
DeepEPBuffer.set_dispatch_mode_as_low_latency()
return DeepEPBuffer.get_deepep_buffer(
    self.group,
    self.hidden_size,
    self.params_bytes,
    self.deepep_mode,
    self.num_max_dispatch_tokens_per_rank,
    self.num_experts,
)
```

如果后续这页对应的代码公开，我会优先看两个位置：

- DeepEP 的 low-latency dispatch/combine kernel，尤其是 inter-node 和 intra-node 的分层路由；
- SGLang `token_dispatcher/deepep.py` 里 dispatch handle 的组织方式，以及是否出现 node-local relay 或 NVLink second-stage dispatch。

这里不硬编 PR。公开信息能确认的是方向和瓶颈，不能确认具体实现已经开放。

# 0x9. Simple Eagle

![](https://files.mdnice.com/user/59/8937b768-28fe-4d03-83ed-7fd4e0d50049.png)

Slides 里的 Simple Eagle 说了两个问题：

- 原始 NEXTN/Eagle-2 有一些 prepare invalid ops；
- draft-extend 不在 CUDA Graph 里，导致 decode 的图捕获和 replay 不够干净。

公开 PR 里，比较接近这页的是这一串：

- [#11398 Beta spec-overlap for EAGLE](https://github.com/sgl-project/sglang/pull/11398)
- [#11434 move eagle draft post process to cuda graph](https://github.com/sgl-project/sglang/pull/11434)
- [#11643 Abstraction for spec worker and code cleanup](https://github.com/sgl-project/sglang/pull/11643)
- [#11653 Fix 1-step draft model forward](https://github.com/sgl-project/sglang/pull/11653)
- [#12443 spec-overlap supporting DP-ATTN/PD-disaggregation/NPU graph](https://github.com/sgl-project/sglang/pull/12443)

PR [#11398](https://github.com/sgl-project/sglang/pull/11398) 引入了 beta EAGLE v2，用 `SGLANG_ENABLE_SPEC_V2=1` 开启。它的初始测试里，Llama3.1 8B + EAGLE 的速度从 246.80 token/s 提到 273.43 token/s。

当前 `eagle_worker_v2.py` 的 decode 路径已经比较像 slides 里说的「verify + draft extend」流水：

```python
with self.draft_worker.draft_tp_context(
    self.draft_worker.draft_runner.tp_group
), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
    verify_input: EagleVerifyInput = self.draft_worker.draft(model_worker_batch)

assert verify_input.is_verify_input()
model_worker_batch.spec_info = verify_input
batch_output = self.verify(model_worker_batch)

with self.draft_worker.draft_tp_context(
    self.draft_worker.draft_runner.tp_group
), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
    self.draft_worker._draft_extend_for_decode(model_worker_batch, batch_output)
return batch_output
```

PR [#11434](https://github.com/sgl-project/sglang/pull/11434) 则把 draft post-processing 移进 CUDA Graph，并把 tree-building 工具整理到 `eagle_utils.py`。现在的 `EAGLEDraftExtendCudaGraphRunner` 会为 draft extend 准备静态输入和 batch sizes：

```python
class EAGLEDraftExtendCudaGraphRunner:
    def __init__(self, eagle_worker: EAGLEWorker, ...):
        ...
        self.forward_mode = ForwardMode.DRAFT_EXTEND_V2
        self.graphs = {}
        self.output_buffers = {}
        self.capture_bs, self.compile_bs = get_batch_sizes_to_capture(model_runner)

        self.num_tokens_per_bs = self.speculative_num_steps + 1
        self.max_bs = max(self.capture_bs)
        self.max_num_token = self.max_bs * self.num_tokens_per_bs

        self.draft_extend_attn_backend.init_cuda_graph_state(
            self.max_bs, self.max_num_token
        )
```

这类优化的收益不一定体现在单个算子耗时上，而是减少 decode loop 里每步的 Python/prepare/graph 外碎片。对小 batch decode 来说，这些碎片也会变成可见 TPOT。

# 0xA. DeepXTrace

![](https://files.mdnice.com/user/59/0cd87d1c-4d2c-49ca-85c9-2fdd536570f7.png)

这页讲的是观测。MoE 分布式推理最烦的一类问题是「慢 rank」：你看到的是某个 dispatch/combine 变慢，但真正原因可能是发送端算慢、接收端热点、网络链路问题，或者三者混在一起。

Ant 开源了 [DeepXTrace](https://github.com/antgroup/DeepXTrace)，README 里对它的定位是：通过给 DeepEP/MC2 这类通信库打 probe，低开销定位 MoE 分布式环境中的 slow ranks。它分两部分：

- MoE COMM Metrics Probe：在通信算子里采集诊断指标；
- DeepXTrace Metrics Analysis：把各 rank 的指标聚合成 latency matrix，再做异常分析和可视化。

它支持三类 slowdown：

- Comp-Slow：发送端因为 Attention/MoE 等计算慢，导致 send 晚发；
- Mixed-Slow：接收端或热点 expert 导致 recv 行为异常，或者网络 incast；
- Comm-Slow：通信链路本身慢。

DeepXTrace 的核心视角是构造一个 `N x N` 的矩阵 `M`，其中 `Mij` 表示 `rank_i` 等 `rank_j` 的延迟。比如 Dispatch 矩阵里某一列明显偏高，通常说明这一列对应的 rank 是瓶颈来源。这个视角比只看单 rank 日志好很多，因为 MoE 通信的问题往往是「我慢，是因为我在等别人」。

README 里给的 DeepEP LL mode 接入方式大致是：

```python
_diagnose = ds.Diagnose(group=group, enable_async=True)
_diagnose.start_async_diagnose()

dispatch_wait_recv_cost_stats = _diagnose.get_stats_ll_stats_tensor()[0]
_buffer.low_latency_dispatch(
    hidden_states,
    topk_idx,
    num_max_dispatch_tokens_per_rank,
    num_experts,
    dispatch_wait_recv_cost_stats=dispatch_wait_recv_cost_stats,
    use_fp8=True,
)

combine_wait_recv_cost_stats = _diagnose.get_stats_ll_stats_tensor()[1]
_buffer.low_latency_combine(
    hidden_states,
    topk_idx,
    topk_weights,
    handle,
    combine_wait_recv_cost_stats=combine_wait_recv_cost_stats,
)
```

这也解释了为什么 slides 把它放在 Methodology 最后：前面的 SwapAB/SBO/EPLB/Dispatch 都是「怎么优化」，DeepXTrace 是「线上怎么知道哪里又坏了」。

# 0xB. Decode 评测和结论

![](https://files.mdnice.com/user/59/1a4e5fd9-ec23-4e3b-8a24-bdfd03d6b19e.jpg)

![](https://files.mdnice.com/user/59/cf635c63-5ec6-40b4-9470-a2c35f739c49.png)

Decode 评测的配置是 input=4096、output=1536，Decode 侧 DP16 + EP16，开启 DP attention、MTP=(1,1,2) 之类的 decode 优化。Slides 给出的提升随着 batch 从小到大逐步变化：

- batch 32：提升 42%；
- batch 48：提升 32%；
- batch 64：提升 21%；
- batch 96：提升 12%。

这和前面的优化方向也吻合。小 batch 时 SwapAB/SBO 对无效计算和通信等待的改善最明显；batch 变大后，GEMM 自身利用率上来，收益比例自然会下降。

![](https://files.mdnice.com/user/59/c17a7087-da50-4cb1-926d-e4f945debb66.png)

结论页里说 Ant 在 H20 上对 DeepSeek-R1/V3/V3.1 累积了比较完整的一套优化，Prefill 和 Decode 都做到了比较强的水平。公开入口主要有两个：

- AntGroup 的复现和汇总 PR：[antgroup/sglang#4](https://github.com/antgroup/sglang/pull/4)
- 相关 SGLang/DeepEP/DeepGEMM/FlashMLA PR，也就是上面一路引用的那些。

# 0xC. DeepSeek-V3.2：DSA 带来的新问题

![](https://files.mdnice.com/user/59/bbe8ad1b-c4df-4614-9bcb-879ab8fdd1d5.jpg)

![](https://files.mdnice.com/user/59/cb45c553-be9d-4f2f-a36d-d5f249fcfc20.png)

DeepSeek-V3.2 的新变量是 DSA，也就是 Dynamic Sparse Attention。Slides 里把它拆成两个部分：

- Top-K Selector + Lightning Indexer，大约 0.85B 参数；
- Attention 由传统 MLA 的 `O(L^2)`，变成和选出的 sparse KV 数量相关的 `O(Lk)`。

这个结构对长上下文是很有吸引力的，但工程实现上有几个麻烦：

- FlashMLA-DSA 对 `h_q` 有 multiple-of-64 的限制；
- Lightning Indexer 当时不支持 TP split；
- 如果用 DP attention，48.5K ISL 的 TTFT 会太高；
- 如果纯 TP8，H20 上 `h_q` padding 到 64 会带来 3 倍计算浪费，Indexer 不切权重还会有 7 倍计算浪费。

也就是说，DSA 理论上减少 attention 复杂度，但新的 indexer 和 kernel 约束又把一部分收益吃掉了。

公开 PR 里，DeepSeek-V3.2 相关的几条比较关键：

- [#12065 support context parallel with deepseekv3.2-DSA](https://github.com/sgl-project/sglang/pull/12065)
- [#11892 DeepSeek-V3.2: Add Adaptive MHA Attention Pathway for Short-Sequence Prefill](https://github.com/sgl-project/sglang/pull/11892)
- [#12094 Fuse wk and weight_proj in Indexer for DeepSeekV3.2-FP4](https://github.com/sgl-project/sglang/pull/12094)
- [#17205 DeepSeekV3.2: optimize indexer weight_proj-mma performance](https://github.com/sgl-project/sglang/pull/17205)
- [#16637 Overlap indexer weights_proj during dual_stream decode](https://github.com/sgl-project/sglang/pull/16637)

PR [#12065](https://github.com/sgl-project/sglang/pull/12065) 的思路是给 DSA prefill 引入 context parallel。以 TP=EP=4、DP=2 为例，每个 DP 接一个独立 request，embedding 后把 `(batch * seq_len, H)` 按 context parallel 切给 attention TP ranks。MoE 部分也吃切分后的 hidden，最后再 all-gather 回完整 hidden。

当前代码里开关是：

```python
def is_nsa_enable_prefill_cp():
    return get_global_server_args().enable_nsa_prefill_context_parallel
```

并支持两种切分模式：

```python
def is_nsa_prefill_cp_in_seq_split():
    return (
        is_nsa_enable_prefill_cp()
        and get_global_server_args().nsa_prefill_cp_mode == "in-seq-split"
    )

def is_nsa_prefill_cp_round_robin_split():
    return (
        is_nsa_enable_prefill_cp()
        and get_global_server_args().nsa_prefill_cp_mode == "round-robin-split"
    )
```

round-robin split 的注释写得很清楚，它不是简单把连续 token 分给不同 rank，而是按 `token_idx % cp_size` 做交错切分：

```python
# token0, token1, token2, token3, token4, token5, ...
#
# dp_atten_tp0: token0, token4, token8,  token12, ...
# dp_atten_tp1: token1, token5, token9,  token13, ...
# dp_atten_tp2: token2, token6, token10, token14, ...
# dp_atten_tp3: token3, token7, token11, token15, ...
```

这样做是为了缓解 causal attention 下不同 rank 计算量不均的问题。普通连续切分里，前面的 rank 看历史 KV 少，后面的 rank 看历史 KV 多；交错后每个 rank 的 token 在序列上分布更均匀。

模型 forward 入口会准备 CP metadata：

```python
if self.nsa_enable_prefill_cp:
    if can_nsa_cp_split(len(input_ids), self.cp_size, self.use_nsa, forward_batch):
        forward_batch.attn_cp_metadata = prepare_context_parallel_metadata(
            len(input_ids),
            self.cp_rank,
            self.cp_size,
            forward_batch.seq_lens_cpu.tolist(),
        )
```

每层里，如果启用 NSA prefill CP，就换成 `NSACPLayerCommunicator`：

```python
if self.nsa_enable_prefill_cp:
    self.layer_communicator = NSACPLayerCommunicator(
        layer_scatter_modes=self.layer_scatter_modes,
        input_layernorm=self.input_layernorm,
        post_attention_layernorm=self.post_attention_layernorm,
        allow_reduce_scatter=True,
        is_last_layer=(
            is_nextn or (self.layer_id == self.config.num_hidden_layers - 1)
        ),
        qkv_latent_func=self.self_attn.prepare_qkv_latent,
    )
```

PR [#11892](https://github.com/sgl-project/sglang/pull/11892) 则对应 slides 里 Future Work 的「seq_len < 2K 用 masked MHA」。它的逻辑是：DeepSeek-V3.2 prefill 里如果所有长度都走 MLA，不一定最优。短序列下 MLA 的 compression/decompression 和 absorbed attention 开销不划算，MHA 更快。当前 NSA backend 会在 prefill 时根据长度、设备、dtype、chunk capacity 等条件决定是否走 MHA：

```python
def set_nsa_prefill_impl(self, forward_batch: Optional[ForwardBatch] = None):
    if forward_batch and forward_batch.forward_mode.is_extend_without_speculative():
        max_kv_len = forward_batch.seq_lens_cpu.max().item()
        sum_seq_lens = sum(forward_batch.seq_lens_cpu)
        device_sm = get_device_sm()

        self.use_mha = (
            (device_sm == 90 or (device_sm >= 100 and device_sm < 110))
            and max_kv_len
            <= envs.SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD.get()
            and forward_batch.token_to_kv_pool.dtype
            in [torch.bfloat16, torch.float8_e4m3fn]
            and sum_seq_lens <= forward_batch.get_max_chunk_capacity()
            and (not is_nsa_enable_prefill_cp())
            and (forward_batch.hisparse_coordinator is None)
        )
    else:
        self.use_mha = False
```

PR [#12094](https://github.com/sgl-project/sglang/pull/12094) 和 [#17205](https://github.com/sgl-project/sglang/pull/17205) 都是 Indexer 侧的优化。前者把 `wk` 和 `weight_proj` 在 FP4 模型里融合成一次 GEMM，后者把 `weights_proj` 的计算从 FP32 调整为 BF16 权重计算再把输出转回 FP32，解决 indexer 里相对耗时的 `weight_proj-mma`。

![](https://files.mdnice.com/user/59/09a65a4e-025b-4da7-81e0-e013efa19b2f.png)

Slides 给出的 DeepSeek-V3.2 Prefill 最终方案是：

- H20-141G 单机 8 卡：Attention-CP8，MoE-TP8，ISL=48.5K，TTFT=5s；
- H20-96G 两机 16 卡：PP2 * (Attention-CP8, MoE-TP8)，TTFT=3.1s。

这里最关键的是把 attention 的 context parallel 和 MoE 的 TP 拆开看。Attention 需要按序列切，MoE 需要按 hidden/expert 组织；两者强行用同一个 parallel 维度，往往会让一边很别扭。

![](https://files.mdnice.com/user/59/f8f9b20c-ac81-4e01-989c-9d06b307c571.png)

Future Work 里提了几件事：

- Operator：`seq_len < 2K` 时使用 masked MHA；
- Indexer：继续优化性能，并支持 TP；
- H20-96G PD：Prefill 用 `PP2 * (Attn-CP8, MoE-TP8)`，Decode 侧按 TPOT SLA 分两档；
- KV cache for DSA：Hierarchical Sparsification。

其中 masked MHA 和一部分 Indexer 优化已经能在上面的公开 PR 里看到；Decode 侧 TP8 或 DP16/EP16 的 DeepSeek-V3.2 完整形态，看起来还在继续推进。

# 0xD. 小结

这次分享如果只看 slides，会觉得是很多优化点堆在一起。但把 PR 和代码串起来后，主线其实很清楚：

1. 先承认 H20 的偏科：算力弱、显存/NVLink 强、RDMA 弱。
2. 用 PD disaggregation 把 Prefill 和 Decode 的瓶颈拆开。
3. Prefill 侧做 TP scattered input、MHA one-shot、MoE down projection TMA。
4. Decode 侧围绕小 batch MoE 做 SwapAB 和 SBO。
5. 通信侧用 Expert Affinity EPLB 减少跨节点通信，用 DeepXTrace 定位慢 rank。
6. DeepSeek-V3.2 侧重新处理 DSA 带来的 CP、Indexer、MHA/MLA 选择问题。

我觉得最值得借鉴的是两个习惯。第一，优化不是「看到一个 kernel 慢就调一个 kernel」，而是先把硬件约束和部署形态定下来，再决定 kernel、通信、调度各自要解决什么。第二，很多优化都用了真实线上分布，比如 MoE tuning 用真实 `topk_ids`，EPLB 用 co-activation matrix，这比随机 benchmark 更接近实际服务。

参考链接：

- AntGroup 复现汇总 PR：https://github.com/antgroup/sglang/pull/4
- LMSYS Blog：https://www.lmsys.org/blog/2025-09-26-sglang-ant-group/
- Prefill TP scattered input：https://github.com/sgl-project/sglang/pull/10568
- Prefill MHA one-shot：https://github.com/sgl-project/sglang/pull/10953
- FusedMoE down projection TMA：https://github.com/sgl-project/sglang/pull/10567
- SwapAB DeepGEMM：https://github.com/deepseek-ai/DeepGEMM/pull/192
- SGLang SwapAB rework：https://github.com/sgl-project/sglang/pull/16723
- FP8 FlashMLA：https://github.com/deepseek-ai/FlashMLA/pull/82
- SBO in SGLang：https://github.com/sgl-project/sglang/pull/9660
- SBO in DeepEP：https://github.com/deepseek-ai/DeepEP/pull/390
- SBO in DeepGEMM：https://github.com/deepseek-ai/DeepGEMM/pull/183
- Expert Affinity EPLB：https://github.com/antgroup/sglang/pull/2
- EAGLE spec-overlap：https://github.com/sgl-project/sglang/pull/11398
- EAGLE draft post process CUDA Graph：https://github.com/sgl-project/sglang/pull/11434
- DeepXTrace：https://github.com/antgroup/DeepXTrace
- DeepSeek-V3.2 DSA CP：https://github.com/sgl-project/sglang/pull/12065
- DeepSeek-V3.2 adaptive MHA：https://github.com/sgl-project/sglang/pull/11892
