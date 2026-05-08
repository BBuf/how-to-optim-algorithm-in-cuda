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

标题页给出主题：Omni-Infer 中的 MTP，重点是昇腾亲和的高吞吐投机推理。这里的“亲和”不是简单支持 NPU，而是要把投机推理里的采样、校验、hidden state 选择和 MLA kernel 都改到适合昇腾执行的形态。

Decode 阶段单 token 迭代很容易受带宽和同步开销限制，MTP 一次预测多个后续 token，目标是把串行 decode 步合并成更适合 NPU 执行的大步。后面 slides 的主线就是：先说明社区实现的问题，再看 Omni-Infer 怎么减少 CPU-NPU 同步和 sampler 限制。

### Slide 2：Decode memory-bound 与投机推理

<img src="https://files.mdnice.com/user/59/3677c274-d289-4776-b82a-e8be40db7587.png" referrerpolicy="no-referrer" />

这页从 Decode 阶段的计算特征讲起：每次迭代，每条 request 输入一个 token 产生一个 token，计算密度较低。对昇腾这类高计算密度/带宽比的硬件来说，单 token decode 很难把硬件吃满。

投机推理的做法是一次推理迭代计算一条请求的多个 token。draft/MTP 层先提出候选 token，主模型再验证。只要接受率足够高，就能用一次主模型验证替代多次单 token decode，从而减少迭代次数和调度开销。

### Slide 3：社区 MTP/EAGLE 实现的通用流程

<img src="https://files.mdnice.com/user/59/b964ac17-6192-4b8d-a8ad-4dd3ea8f0a85.png" referrerpolicy="no-referrer" />

这页画的是社区里常见的 MTP/EAGLE 执行方式。prefill 之后，主模型产出 hidden states，MTP 或 draft 模块基于 hidden states 生成一串候选 token；decode 时，主模型对这串候选进行验证，接受前缀 token，并把后续状态交给下一轮 draft。

这个流程在概念上很直接，但落到 NPU 上会遇到两个问题。第一，验证后要选“正确 tokens 及其对应 hidden states”作为 MTP 输入，选择逻辑如果回到 CPU，会打断 NPU 算子下发。第二，采样参数和 penalty 状态不是每个 token 都一样，验证阶段不能只做简单 token equality。

### Slide 4：验证后的 token 和 hidden state 选择

<img src="https://files.mdnice.com/user/59/f99b7bd4-d34f-49ce-ab86-a9047bcb5586.png" referrerpolicy="no-referrer" />

这页上半部分把 CPU stream 和 NPU stream 分开。CPU 侧负责输入准备和输出处理，NPU 侧跑主体模型和投机模型；prefill 阶段主体模型产 hidden states，投机模型用这些 hidden states 生成候选 token。进入 decode 后，主体模型先验证投机 tokens，再把验证后的 token 和对应 hidden states 交给 MTP 层继续预测。

下半部分的 token 图更具体：prefill 中 t0/g0/t1/g1... 交替出现，main model 负责算 probs、verify 和 sampling；验证后不能直接取“最后一个 draft 位置”的 hidden state，而是要取最后接受 token 对应的 hidden state。右侧文字指出这种做法会引入 CPU 和 NPU 同步，影响算子下发，在主体模型和 MTP 执行之间形成气泡。后面优化采样和 verifier，基本都在消这类同步成本。

### Slide 5：CPU-NPU 同步气泡

<img src="https://files.mdnice.com/user/59/f39884bc-aa4b-4bcb-900b-d0ac53487f70.png" referrerpolicy="no-referrer" />

这页进入 Omni-Infer 的 MTP 实现。图里把主体模型、MTP 模型、验证与采样串起来，重点是让验证后的 token 选择、hidden state 选择和下一轮 draft 尽量留在设备侧完成，减少 CPU 介入。

CPU-NPU 同步气泡是 Ascend 场景的关键问题。如果每次验证都回 CPU 算 accept length、再回 NPU 准备下一步，投机推理省下的 decode 步可能被同步吃掉。后面代码里 `verify` 尽量用张量逻辑找 accept length 和 output token，就是为了把这段路径留在设备上。

### Slide 6：Omni-Infer 的 MTP 支持范围

<img src="https://files.mdnice.com/user/59/54c33693-a707-46c9-b300-c0b10229a8b3.png" referrerpolicy="no-referrer" />

这页明确列了当前支持的模型：DeepSeek V3 的 MTP，Qwen2 的 EAGLE/EAGLE3，Pangu Ultra MoE 的 MTP。也就是说，Omni-Infer 不是只给某个模型写一条特殊路径，而是把 speculative method 做成推理引擎能力。

不同模型的 draft 模块来源不同：DeepSeek V3 MTP 属于模型自带多 token prediction，Qwen2 EAGLE/EAGLE3 走 draft model/hidden state 预测，Pangu Ultra MoE 又要处理 MoE 场景。统一到引擎里后，公共部分就是 sampling info、verify、KV/cache、hidden state 选择和 kernel 优化。

### Slide 7：采样和验证策略

<img src="https://files.mdnice.com/user/59/622bc963-b16a-4b82-a59b-5d67f37f91a6.png" referrerpolicy="no-referrer" />

这页把采样和校验的概率关系写出来。投机模型生成 token 的概率记为 `q_i`，主模型生成 token 的概率记为 `p_i`。简单校验如果按主模型采样结果比较，相同就接收，否则拒绝，接受率和 `sum p_i q_i` 有关；如果 draft 只做 greedy sample，接受率退化成主模型给该 token 的概率 `p_i`。

拒绝采样更严格：以 `min(p_i, q_i) / q_i` 的概率接收 token；拒绝后，用 `max(p_i - q_i, 0)` 归一化后的分布重新采样。这样做的目标是保持输出分布接近主模型，而不是只追求速度。slide 还提到强制对投机层应用 top-k 后再做拒绝采样，接受率会变成 `sum min(p_i, q_tilde_i)`。

### Slide 8：社区 verifier 的问题

<img src="https://files.mdnice.com/user/59/0363beaf-1588-4800-a249-9802e33cde27.png" referrerpolicy="no-referrer" />

这页指出社区 verifier 的两个问题。第一，对于应用 penalty 等其它采样参数的场景，简单 verifier 会影响大模型精度；第二，Sampler 假设每条输入只有一个 token。

问题集中在 sampling params。presence/frequency/repetition penalty、temperature、top-p/top-k、allowed token ids 都会影响每个 speculative token 的验证。如果 sampler 假设一个 request 只采一个 token，那么 speculative window 里的多个 token 会共享错误的采样状态，输出分布和主模型就对不上。

### Slide 9：validator 两次 sampler 的设计

<img src="https://files.mdnice.com/user/59/577b633a-96e7-4ab5-a718-bcd3cb88d623.png" referrerpolicy="no-referrer" />

这页画了 verifier 的现状。上方 t0/g0/t1/g1... 是交错的 target token 和 draft token，main model 一次性输出这些位置的 logits。下面把 logits 分给两个 Sampler：一边采 target 对应的候选 f0/f1/f2/f3，另一边采 draft 后面的备用 token b0/b1/b2/b3，最后统一进入 rejection sampler 做 verify & sampling。

右侧两条限制就是问题根源：validator 里要调用两次 sampler，而且每条 request 需要处理的 token 数量必须相同。这个假设对真实 serving 不友好，因为不同请求的 speculative length、sampling params、penalty 状态都可能不同。代码里后续需要把 sampling info 按 speculative token 展开，再在 verify 后恢复到 request 粒度。

### Slide 10：任意 speculative token 数的采样参数处理

<img src="https://files.mdnice.com/user/59/091f3dcf-8e5d-417e-bea5-ed12ecd490e5.png" referrerpolicy="no-referrer" />

这页标题是“任意数量投机 token 支持”。右侧两条写得很具体：第一，根据 `spec_metadata` 复制 sampling parameters，支持任意长度，包括 allowed token ids、temperature、min_p、top_k、top_p；第二，对 logits 循环切片应用 penalty，因为 penalty 依赖 penalty cache。

这里不能只把参数简单 repeat。不同 request 的 speculative token 数可能不同，同一条 request 内每个候选 token 的 penalty 状态也会随已接受 token 变化。实现上需要把 request 级 sampling info 展开到 token 级，验证后再恢复到 request 级，否则 sampler 和 rejection sampler 会处理错位。

### Slide 11：Adaptive speculation

<img src="https://files.mdnice.com/user/59/efaf206e-8fd6-45d1-b9d9-cdfe7d033426.png" referrerpolicy="no-referrer" />

这页标题写的是 “Coming soon: 自适应投机”。左图里，蓝色 `t0/t1/t2/t3` 是 main model 验证出来的位置，绿色 `g0/g3` 表示被接受的 draft token，红色 `g1/g2` 表示被拒绝的位置；下面 MTP 分支继续产生 `f0/f1/f2/f3` 和 `h0/h1/h2/h3`。竖虚线可以理解成一次次 speculative window 的边界：有的窗口接受得多，有的窗口很快失败。

右图展示自适应后的形态：每个窗口不再固定投机同样数量的 token，而是根据上一轮接受情况决定下一轮要让 MTP 继续猜几个。中间那行 `f0/h0/f1/h1...` 是不同深度的 draft 候选，下面的“投机 token 过滤”会把已经没有必要验证的 token 去掉，只保留需要进入下一轮 main model 的位置。这样做的目标不是让每轮都猜满，而是在接受率高时扩大窗口，在接受率低时缩短窗口，把 MTP 的算力花在更可能被接受的位置上。

### Slide 12：MLA 中 MTP 的 KV 复用优化

<img src="https://files.mdnice.com/user/59/4b2000fc-edde-4ec8-84d9-828df2b6bb8a.png" referrerpolicy="no-referrer" />

这页讲 MLA 算子层优化。投机 m 个 token 时，MLA 计算里会出现 `m+1` 个 Q 矩阵和同一个 K 矩阵相乘。朴素实现会让每个 Q 都重新加载 K，K 矩阵在 HBM 和 L1 之间来回搬，decode 的带宽瓶颈会被放大。

右图给了 tiling 思路：Q1、Q2 常驻 L1 缓存，K_j 以流式方式加载，和多个 Q 连续相乘。左侧小字说得更细：通过调整 L1 缓存分配，多个 Q 同时进 L1，K 每次加载后直接服务多个 Q；预测 m 个 token 时，K 矩阵加载次数从按 Q 重复加载，变成大约按 tile 分批加载。另一个优化是让 Q 矩阵常驻 L1，避免 softmax 结果和 V 相乘时覆盖掉 Q 的缓存。代码侧对应到 MLA decode length 里把 speculative window 纳入一次更大的 query group。

### Slide 13：Omni-Infer 仓库和接入方式

<img src="https://files.mdnice.com/user/59/2d337110-3710-4ea7-a3bb-d0c18d3b97e8.png" referrerpolicy="no-referrer" />

仓库页给出 Omni-Infer 的 Gitee 地址。公开实现里既有 vLLM/Ascend 适配，也有 SGLang patch。SGLang patch 里新增 `SpeculativeAlgorithm.MTP` 和 `MTPWorker`，能直接对上本场 slides 的 verifier、sampling info 和 draft/verify 主循环。

读代码时可以按三条线索走：一是 `MTPWorker` 如何构造 target model 验证输入；二是 sampler 参数如何 repeat/restore；三是 MLA 和 KV 侧如何感知 speculative token 数。这样能把 slides 里的“减少同步、支持任意 token、MLA 复用”落到具体实现。

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
