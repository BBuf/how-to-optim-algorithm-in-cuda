> 投机解码训练这块很容易被 serving 端的性能数据盖过去。这里把 SpecForge、EAGLE3 训练和 SGLang serving 侧怎么接起来讲清楚。

# 0x0. 前言

投机解码的 serving 侧大家讲得多，训练侧反而容易被低估。EAGLE3 这类方法的 acceptance length 很依赖 draft model 质量，而 draft model 的训练又牵涉 target hidden states、特殊 mask、递归 unroll 和模型适配。SpecForge 这场分享补上了这块拼图。

# 0x1. 资料和代码落点

对应资料和代码：

- SpecForge 仓库：`specforge/core/eagle3.py`、`specforge/core/eagle3_adapters.py`、`specforge/modeling/draft/flex_attention.py`。
- SpecForge 文档：`docs/basic_usage/training.md`，解释 online/offline hidden states。
- LMSYS 博客：`https://lmsys.org/blog/2025-07-25-spec-forge/`，介绍 SpecForge 的定位和 EAGLE3 训练；`https://lmsys.org/blog/2025-08-27-gpt-oss/` 对应 GPT-OSS EAGLE 实践。
- SGLang serving 侧的 speculative decoding 和 multi-LoRA 能力是下游消费端，下面还是把重心放在训练框架本身。

LMSYS 的 SpecForge blog 把训练问题讲得更清楚：EAGLE3 的 draft model 不只吃 token embedding，还要吃目标模型若干中间层的 hidden states。这样做的好处是 draft model 能更接近 target model 的局部推理状态，坏处是训练流程不再像普通 LM 那么干净，必须先拿到 target hidden states，再把多层 hidden states 投影、拼接、递归 unroll。

![](https://files.mdnice.com/user/59/41c0c2fb-7f9b-4b79-a1b3-3e96363e6395.PNG)

这张图就是 EAGLE3 的训练数据流：target model 负责提供 logits 和 hidden states，draft model 用这些中间表示预测后续 token，训练时还要模拟多步生成。LMSYS blog 里强调 SpecForge 支持 online/offline 两条路径，原因也在这里。如果在线跑 target model，训练时 GPU 压力大，但磁盘不会被 hidden states 数据集打爆；如果先离线生成 hidden states，训练阶段便宜很多，但中间数据会非常大。

![](https://files.mdnice.com/user/59/48b7c84d-440c-462a-b381-4acd0863a18e.jpg)

这张 online/offline 图对应到代码就是两个入口：online 路径在训练 step 内部调用 target model 拿 hidden states，offline 路径把 hidden states 当成 dataset 字段读进来。读 SpecForge 源码时，只要抓住这件事就不容易迷路：`Eagle3Model` 处理的是共同的 TTT/unroll/loss 逻辑，online/offline 差别主要在 hidden states 从哪里来。GPT-OSS 那篇 LMSYS blog 的价值也在这里，它说明 SpecForge 不是只适配一种 Llama-like 模型，而是把 target backend 做成可扩展层，新模型结构变了以后只需要把 hidden states、tokenizer、draft config 对齐。

# 0x2. Slides 逐页解读

### Slide 1：SpecForge：Speculative Decoding Models 的训练框架

![](https://files.mdnice.com/user/59/ba69be4e-bd0e-4084-b9e4-6e50b89d6769.png)

SpecForge 解决的是投机解码的“训练侧”：SGLang 已经能 serve EAGLE/EAGLE3 draft model，但 draft model 从哪里来、怎么跟目标模型 hidden states 对齐、怎么处理特殊 attention mask，是另一个完整工程。

### Slide 2：目录：从投机解码到自定义训练

![](https://files.mdnice.com/user/59/55dc0967-bfd2-483e-b9ec-d9f60ad7f763.png)

目录顺序很清楚：先解释 speculative decoding 为什么能降低 decode latency，再讲 EAGLE3 和 SpecForge，接着落到 GPT-OSS、Flex Attention、VLM、LoRA 和自定义训练。它不是单模型脚本，而是一个训练框架。

### Slide 3：为什么 decode 阶段值得用投机解码

![](https://files.mdnice.com/user/59/1e043b4d-8c64-4611-8930-890a66145a1e.png)

小 batch decode 很容易 memory-bound。目标模型每步只出一个 token，GPU 算力吃不满。投机解码用一个便宜 draft model 一次猜多个 token，再让 target model 并行验证，用更多计算换更少串行步数。

### Slide 4：EAGLE3 在 SGLang 里的收益

![](https://files.mdnice.com/user/59/355584e2-8a73-4a6f-ba0d-af9f6390963d.png)

EAGLE3 的意义在于 draft model 不只看 token，还看 target model 的中间 hidden states。SGLang 侧能拿到较高 acceptance length，slides 上给出 Llama3.1-8B 约 2.4x 的例子。

### Slide 5：SpecForge 的定位：训练 draft model

![](https://files.mdnice.com/user/59/c4aaf488-684b-4041-91bc-71da68ff0e88.png)

SpecForge 的定位是“把 EAGLE3 训练流程标准化”。它把 target hidden states 生成、draft forward、TTT unroll、loss/accuracy 计算封在框架里，用户不用从 SafeAILab/EAGLE 的训练脚本开始手搓。

### Slide 6：开箱支持主流模型和 SGLang

![](https://files.mdnice.com/user/59/da6fe822-7840-4de7-99de-ed4b6c49b40a.png)

开箱支持并不只是模型名列表。GPT-OSS、Qwen、Llama、Qwen2.5-VL 这类模型的 tokenizer、hidden state 层选择、attention backend、FSDP/TP 都有差异。SpecForge 把这些差异放进 target/draft backend。

### Slide 7：为什么把训练框架放到 SGLang 生态里

![](https://files.mdnice.com/user/59/3582d06f-f627-4756-a79e-910141a466bc.png)

放到 SGLang 生态的好处是训练和 serving 能用同一套模型假设：训练出来的 draft model 可以直接给 SGLang speculative decoding 使用，评测 acceptance length 也更接近线上行为。

### Slide 8：EAGLE3 的 Training-Time Test

![](https://files.mdnice.com/user/59/e5d6a54e-310e-41a3-bb78-3b0e2a86458e.png)

EAGLE3 Training-Time Test 是这场分享的技术核心。训练时不是一次 teacher forcing 结束，而是递归 unroll 多步：上一步 draft 输出会影响下一步输入，attention mask 和 position 都要跟着移动。

### Slide 9：online/offline hidden states 两种训练路径

![](https://files.mdnice.com/user/59/fcdcf4b3-1bd3-4dfe-8b1d-ba724b1060a9.png)

online/offline 的取舍很现实。online 训练时 target model 现场产 hidden states，省磁盘但吃 GPU；offline 先把 hidden states 写盘，训练时只读数据，能低 GPU 门槛跑，但磁盘占用会非常夸张。

### Slide 10：Online & Offline Training 对比

![](https://files.mdnice.com/user/59/706d8a11-191b-4445-93e5-52a67d08a802.png)

这一页把 online/offline 的代价摊开了。online 训练时直接调用 target model，磁盘占用低，但大模型必须常驻训练流程，GPU 压力更大；offline 只在数据准备阶段用 target model，把 hidden states 预先落盘，训练 draft 时甚至可以降到单卡，但 UltraChat + ShareGPT 这类数据会把磁盘推到 TB 级。

### Slide 11：GPT-OSS EAGLE 的例子

![](https://files.mdnice.com/user/59/6b3f5677-0d2f-4b8c-9514-0f3c35fde91e.png)

GPT-OSS 的例子说明 SpecForge 不是只服务 Llama。开源模型结构变化快，如果训练框架把 target model 细节写死，很快就不能用。SpecForge 把 GPT-OSS 的 target backend 单独适配，draft 侧保持 EAGLE3 逻辑。图里的 acceptance length 对比也在提醒我们：draft model 训练质量会直接体现在 serving 侧吞吐上。

### Slide 12：Flex Attention 降显存和提速

![](https://files.mdnice.com/user/59/5a6bd810-b1cd-402a-8fa4-603949d09036.png)

Flex Attention 这页对应的是 EAGLE3 特殊 mask 的成本。递归 TTT 会构造很大的 attention pattern，传统 SDPA 容易显存爆。SpecForge 用 PyTorch Flex Attention 和 DynamicCache，把 mask 压成 block 语义后再跑，目标不是换一个 attention 名字，而是把 TTT 的稀疏依赖表达清楚。

### Slide 13：VLM 也能训练 EAGLE3 draft

![](https://files.mdnice.com/user/59/cbb669a3-d453-4a4b-9f55-6f7b894090d4.png)

VLM 页最关键的是 hidden states 不只来自文本。Qwen2.5-VL 这类模型要处理 image grid、mrope position id、视觉 token 和文本 token 对齐，SpecForge 的 VLM wrapper 会把目标模型输出的多层 hidden states 接到 EAGLE3 draft。

### Slide 14：LoRA 与 speculative decoding 共存

![](https://files.mdnice.com/user/59/f42be788-ad7c-41ce-95bf-d47fa235fe62.png)

LoRA 页讲的是部署现实：线上 base model 可能同时挂多个 LoRA adapter。如果 speculative decoding 只更新 base draft，不处理 adapter 侧差异，acceptance rate 会掉。SpecForge/SGLang 需要让 draft/base 的 LoRA 状态对齐。

### Slide 15：自定义训练参数和 chat template

![](https://files.mdnice.com/user/59/df13584f-dc0c-4d99-b5e3-d1a95ed47ccd.png)

自定义训练首先是把参数和数据格式接上。slides 里给的是 online 训练入口和 `--help` 参数说明，旁边的 chat template 代码说明 SpecForge 会通过 template registry 统一 prompt 格式。这里不要小看模板：target hidden states、labels、loss mask 都依赖 token 边界，chat template 一错，训练看起来能跑，acceptance length 会很难看。

### Slide 16：自定义 target model 和 draft model

![](https://files.mdnice.com/user/59/16c88cf3-70ef-48ea-be8a-5b95cdeb3774.png)

这一页才是模型接入入口。target 侧要提供 wrapper，把目标模型的 logits、hidden states、tokenizer 行为整理成 SpecForge 期望的格式；draft 侧要继承基础 draft model，把 backbone、embedding、lm head、projection 这些部件拼好。EAGLE3 训练主体可以复用，但 hidden states 层选择、维度、position ids 这些模型相关细节必须在 wrapper 里说清楚。

### Slide 17：结束页

![](https://files.mdnice.com/user/59/dba1426a-9d8c-4b84-9560-f6d2b2b9abca.png)

结束页没有新的技术点。回到主线看，SpecForge 的价值是把 EAGLE3 训练里最难维护的 hidden states、TTT unroll、特殊 mask 和模型适配收进框架里，让训练出来的 draft model 能更自然地进入 SGLang serving。

# 0x3. 关键代码拆解

SpecForge 的 EAGLE3 训练主循环在 `specforge/core/eagle3.py`。类注释已经把步骤写得很直白：

```python
class OnlineEagle3Model(Eagle3Model):
    """
    Online training means we have the target hidden_states available during training.
    1. extract hidden states from the target model.
    2. concatenate hidden states from 3 aux layers.
    3. project 3*hidden_size to hidden_size.
    4. concat projected hidden states and embedding output.
    5. run TTT to train the draft model.
    """
```

进入 forward 后，先把 target 分布 pad 成 TTT 需要的形状，再把三层 hidden states 投影回目标 hidden size：

```python
target_p_padded, position_mask = _compute_target_p_padded(
    target=target,
    t2d=self.draft_model.t2d,
    loss_mask=loss_mask,
    length=self.length,
)

batch_size, seq_length, _ = hidden_states.shape
hidden_states = self.draft_model.project_hidden_states(hidden_states)
```

TTT loop 是最值得看的部分。每一步都从 adapter 取当前 step 的 view，然后重新 embed input ids、跑 draft backbone、算 loss/accuracy，并把下一步需要的 input/mask 补出来：

```python
adapter = self._make_adapter()
for idx in range(self.length):
    state = adapter.step_view(
        idx=idx,
        ttt_length=self.length,
        global_input_ids=global_input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        position_ids=position_ids,
        hidden_states=hidden_states,
        target_p_padded=target_p_padded,
        position_mask=position_mask,
        seq_length=seq_length,
    )

    inputs_embeds = self.draft_model.embed_input_ids(state.input_ids)
    hidden_states_out = self.draft_model.backbone(
        input_embeds=inputs_embeds,
        hidden_states=state.hidden_states,
        attention_mask=state.attention_mask,
        position_ids=state.position_ids,
        past_key_values=past_key_values,
        cache_hidden=cache_hidden,
    )
```

Flex Attention 路径对应 slide 里的内存优化。代码里当 backend 是 `flex_attention` 时改用 `DynamicCache`，mask 收缩交给 attention module：

```python
if self.attention_backend in ["sdpa", "fa", "usp"]:
    cache_hidden = [[], []]
    past_key_values = None
elif self.attention_backend == "flex_attention":
    cache_hidden = None
    past_key_values = DynamicCache()
else:
    raise ValueError(f"Unknown attention backend: {self.attention_backend}")
```

online/offline 的差别可以看文档里的定义。online 现场跑 target model：

```text
Online training is suitable for users with limited disk space but sufficient GPUs.
```

offline 先生成 hidden states 再训练：

```text
Offline training is suitable for users with sufficient disk space but limited GPUs.
```

我更建议把它理解成工程旋钮：当你有 H100 但没有几十 TB 高速盘，online 更舒服；当你只有少量 GPU 但能接受数据准备时间，offline 可以把训练门槛降下来。

# 0x4. 小结

SpecForge 的价值不在“又写了一个 EAGLE3 脚本”，而在把 target hidden states、TTT mask、VLM/LoRA/新模型 backend 这些麻烦事收进一个框架。SGLang 负责服务端验证和吞吐，SpecForge 负责让 draft model 训练变得可重复。
