> MTP 在概念上只是“多预测几个 token”，但工程上会牵出 proposer/verifier、采样状态、KV cache 和 NPU 执行路径，这篇就顺着这条线拆。

# 0x0. 前言

MTP 在论文里常被描述成多预测几个 token，但工程实现里，验证、采样、KV 分配、hidden state 选择和硬件同步都会变成实际瓶颈。这篇按 slides 和 Omni-Infer 代码，把这条链路拆开看。

# 0x1. 资料和代码落点

代码落点：

- Omni-Infer：`omni/adaptors/sglang/patches/mtp.patch`，SGLang 侧新增 `MTPWorker`、`SpeculativeAlgorithm.MTP`、sampling info repeat/restore、verify 逻辑。
- Omni-Infer：`omni/layers/sampler.py`，NPU sampler、rejection sampler、penalty cache 相关逻辑。
- Omni-Infer：`omni/adaptors/vllm/spec_decode/post_drafter.py`，vLLM 侧 proposer/verifier 与 adaptive speculation。
- Omni-Infer：`omni/layers/attention/deepseek_mla.py` 和 `omni/accelerators/cache/omni_cache.py`，MTP 下 decode token 数和 MLA actual seq length 处理。
- LMSYS MTP blog：`https://lmsys.org/blog/2025-07-17-mtp/`，给出了 SGLang 里 MTP worker、draft/verify 循环和 acceptance 统计的完整背景。

先用 LMSYS 这张通用流程图校准 MTP 的主循环，再回到 Omni-Infer 看 NPU 适配：

<img src="https://files.mdnice.com/user/59/babf8956-7a0f-4d17-8155-3cb8df8eff6b.png" referrerpolicy="no-referrer" />

这张图把 MTP 拆成三个阶段：主模型先跑出当前 token 和 hidden states，MTP module 根据 hidden states 连续 draft 多个 token，最后主模型一次性 verify 这些候选 token。只要接受 token 数大于 1，decode 的串行步数就下降。Omni-Infer 这篇 slides 的重点是在 Ascend NPU 上把这套流程跑稳：sampling info 要能 repeat/restore，KV cache 要为多个候选 token 留位置，verify 后又要把被拒绝的 token 清掉。后面看 `mtp.patch` 时，这张图可以作为总索引。

# 0x2. Slides 逐页解读

### Slide 1：Omni-Infer 中的 MTP：昇腾亲和的高吞吐投机推理

<img src="https://files.mdnice.com/user/59/490a2254-b26e-4d18-8221-6928fbdbbf08.png" referrerpolicy="no-referrer" />

重点是 Ascend NPU 上的投机推理：Decode 阶段单 token 迭代很容易受带宽和同步开销限制，MTP 一次预测多个后续 token，目标是把串行 decode 步合并成更适合 NPU 执行的大步。

### Slide 2：Decode memory-bound 与投机推理

<img src="https://files.mdnice.com/user/59/3677c274-d289-4776-b82a-e8be40db7587.png" referrerpolicy="no-referrer" />

decode 的算术强度低，尤其小 batch 时更明显。投机推理用 draft/MTP 层多算一些候选 token，再由 target model 验证，只要接受率足够高，总迭代次数就会下降。

### Slide 3：社区 MTP/EAGLE 实现的通用流程

<img src="https://files.mdnice.com/user/59/b964ac17-6192-4b8d-a8ad-4dd3ea8f0a85.png" referrerpolicy="no-referrer" />

社区实现通常分三步：target model 验证 draft tokens，选出被接受的 token，把最后接受位置对应的 hidden state 送给 MTP/draft 继续预测。看起来简单，落到 NPU 上会遇到同步和采样语义问题。

### Slide 4：验证后的 token 和 hidden state 选择

<img src="https://files.mdnice.com/user/59/f99b7bd4-d34f-49ce-ab86-a9047bcb5586.png" referrerpolicy="no-referrer" />

这页强调 hidden state 的选择。MTP 下一步输入不能随便取最后一个位置，而要取“验证后最后接受 token”的状态；否则 draft 会沿着错误前缀继续猜。

### Slide 5：CPU-NPU 同步气泡

<img src="https://files.mdnice.com/user/59/f39884bc-aa4b-4bcb-900b-d0ac53487f70.png" referrerpolicy="no-referrer" />

CPU-NPU 同步气泡是 Ascend 场景的关键问题。如果每次验证都回 CPU 算 accept length、再回 NPU 准备下一步，投机推理省下的 decode 步可能被同步吃掉。

### Slide 6：Omni-Infer 的 MTP 支持范围

<img src="https://files.mdnice.com/user/59/54c33693-a707-46c9-b300-c0b10229a8b3.png" referrerpolicy="no-referrer" />

Omni-Infer 支持 DeepSeek V3 MTP、Qwen2 EAGLE/EAGLE3、Pangu Ultra MoE MTP 等。它不是单个模型 patch，而是把 speculative method 做成推理引擎能力。

### Slide 7：采样和验证策略

<img src="https://files.mdnice.com/user/59/622bc963-b16a-4b82-a59b-5d67f37f91a6.png" referrerpolicy="no-referrer" />

采样策略包括简单 verify、rejection sampling、forced top-k。不同策略影响输出分布是否与 target model 一致。做 benchmark 时只看速度不看分布，很容易得到假收益。

### Slide 8：社区 verifier 的问题

<img src="https://files.mdnice.com/user/59/0363beaf-1588-4800-a249-9802e33cde27.png" referrerpolicy="no-referrer" />

社区 verifier 的坑在 sampling params。presence/frequency/repetition penalty、temperature、top-p/top-k 都会影响每个 speculative token 的验证。如果 sampler 假设一个 request 只采一个 token，就会错。

### Slide 9：validator 两次 sampler 的设计

<img src="https://files.mdnice.com/user/59/577b633a-96e7-4ab5-a718-bcd3cb88d623.png" referrerpolicy="no-referrer" />

validator 两次 sampler 的设计，是为了让 target token 和 draft token 在同一套 sampling 约束下比较。这里还要求每个 request speculative token 数一致，后面 slides 才讲任意 token 数。

### Slide 10：任意 speculative token 数的采样参数处理

<img src="https://files.mdnice.com/user/59/091f3dcf-8e5d-417e-bea5-ed12ecd490e5.png" referrerpolicy="no-referrer" />

任意 speculative token 数需要按 `spec_metadata` 切 sampling params。每个 token 可能有不同 allowed ids、temperature、min_p/top_k/top_p，还要维护 penalty cache，不能把一组参数粗暴 repeat。

### Slide 11：Adaptive speculation

<img src="https://files.mdnice.com/user/59/efaf206e-8fd6-45d1-b9d9-cdfe7d033426.png" referrerpolicy="no-referrer" />

Adaptive speculation 根据接受情况动态调整猜测长度。接受率高就多猜，接受率低就少猜，避免在困难 prompt 上浪费 MTP 计算。

### Slide 12：MLA 中 MTP 的 KV 复用优化

<img src="https://files.mdnice.com/user/59/4b2000fc-edde-4ec8-84d9-828df2b6bb8a.png" referrerpolicy="no-referrer" />

MLA 优化页很硬核：MTP 会产生 `m+1` 个 Q，但它们共享 K。优化目标是减少重复 K 加载，让多个 Q 尽量留在 L1/cache 里，降低 HBM 访问。

### Slide 13：Omni-Infer 仓库和接入方式

<img src="https://files.mdnice.com/user/59/2d337110-3710-4ea7-a3bb-d0c18d3b97e8.png" referrerpolicy="no-referrer" />

仓库页给出 Omni-Infer。公开实现里既有 vLLM/Ascend 适配，也有 SGLang patch。SGLang patch 里新增 `SpeculativeAlgorithm.MTP` 和 `MTPWorker`，能直接对上本场 slides。

### Slide 14：总结

<img src="https://files.mdnice.com/user/59/1bc90c26-3291-4b45-8ccc-b9755bb3ef85.png" referrerpolicy="no-referrer" />

总结起来，Omni-Infer 的 MTP 不是“把 EAGLE 搬到 NPU”。它处理的是 NPU 上 sampler、verifier、graph、MLA 和同步开销这些真实工程细节。

# 0x3. 关键代码拆解

SGLang patch 里新增的 `MTPWorker` 是主入口。decode 分支会把当前 token 和 draft tokens 拼起来，让 target model 一次验证：

```python
seq_lens = batch.seq_lens.repeat_interleave(self.speculative_num_steps + 1, dim=0)
seq_lens[1::2] += 1
locs = seq_lens.clone() - 1
input_ids = torch.cat(
    (batch.input_ids.reshape(-1, 1), batch.spec_info.draft_token), dim=-1
).flatten()

model_worker_batch.input_ids = input_ids
model_worker_batch.seq_lens = seq_lens
model_worker_batch.spec_info.positions = positions
self.prepare_sampling_info(batch.sampling_info)
```

验证后把状态转回 draft 输入，再跑 MTP：

```python
verified_id, accept_length, last_accepted_index, bouns_ids, evict_mask = (
    self.verify(model_worker_batch.input_ids, next_token_ids, num_reqs, batch)
)

draft_input = EagleDraftInput(
    hidden_states=logits_output.hidden_states,
    verified_id=verified_id,
    accept_length=accept_length,
    seq_lens_for_draft_extend=batch.seq_lens,
    req_pool_indices_for_draft_extend=batch.req_pool_indices,
    capture_hidden_mode=CaptureHiddenMode.LAST,
)

forward_batch.spec_info = draft_input
forward_batch.input_ids = next_token_ids.to(torch.int64)
logits_output, _ = self.draft_model_runner.forward(
    forward_batch, skip_attn_backend_init=True
)
```

sampling params 不能只给原 batch 用一次。patch 里先 repeat，跑完再 restore：

```python
def prepare_sampling_info(self, sampling_info):
    self.raw_temperatures = sampling_info.temperatures
    self.raw_top_ks = sampling_info.top_ks
    self.raw_top_ps = sampling_info.top_ps
    self.raw_min_ps = sampling_info.min_ps

    sampling_info.temperatures = self.raw_temperatures.repeat_interleave(
        self.speculative_num_steps + 1, dim=0
    )
    sampling_info.top_ks = self.raw_top_ks.repeat_interleave(
        self.speculative_num_steps + 1, dim=0
    )
    sampling_info.top_ps = self.raw_top_ps.repeat_interleave(
        self.speculative_num_steps + 1, dim=0
    )
```

verify 本身是纯张量逻辑，避免频繁回 CPU。它比较 draft token 和 target forward token，找第一个 reject 位置：

```python
accepted = input_ids.view(num_reqs, -1)[:, 1:] == \
    forward_token_ids.view(num_reqs, -1)[:, :-1]
accepted_mask = accepted.to(dtype=torch.int32)
accepted_mask = torch.cat((accepted_mask, padding_zero), dim=1)
accepted_num = accepted_mask.argmin(dim=1).to(dtype=torch.int32)

last_accepted_index = torch.arange(num_reqs, device=input_ids.device, dtype=torch.int32) \
    * num_sampling_tokens_per_req + accepted_num
output_token_ids = forward_token_ids[last_accepted_index]
```

MLA 侧会把 speculative token 数写进 decode 序列长度。`deepseek_mla.py` 中可以看到：

```python
self.num_speculative_tokens = 0 if not cur_vllm_config.speculative_config \
    or not model_extra_config.operator_opt_config.mtp_remove_redundant_kv \
    else cur_vllm_config.speculative_config.num_speculative_tokens

self.actual_seq_lengths[batch_size] = (1 + self.num_speculative_tokens) * \
    torch.arange(1, batch_size // (1 + self.num_speculative_tokens) + 1,
                 dtype=torch.int64, device=current_platform.device_type)
```

这就是 slide 里 MLA 优化的代码侧入口：decode 不是一个 token 一个 token 看，而是把 speculative window 当成一个更大的 query group 去组织。

# 0x4. 小结

Omni-Infer 这篇可以当作“投机推理在 NPU 上怎么落地”的案例。真正难的不是 MTP 概念，而是 sampler 分布一致性、acceptance 张量化、hidden state 选择、KV/MLA 复用和同步气泡。
