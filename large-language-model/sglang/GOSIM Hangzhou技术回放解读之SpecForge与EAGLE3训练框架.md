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

<img src="https://files.mdnice.com/user/59/41c0c2fb-7f9b-4b79-a1b3-3e96363e6395.PNG" referrerpolicy="no-referrer" />

这张图就是 EAGLE3 的训练数据流：target model 负责提供 logits 和 hidden states，draft model 用这些中间表示预测后续 token，训练时还要模拟多步生成。LMSYS blog 里强调 SpecForge 支持 online/offline 两条路径，原因也在这里。如果在线跑 target model，训练时 GPU 压力大，但磁盘不会被 hidden states 数据集打爆；如果先离线生成 hidden states，训练阶段便宜很多，但中间数据会非常大。

<img src="https://files.mdnice.com/user/59/48b7c84d-440c-462a-b381-4acd0863a18e.jpg" referrerpolicy="no-referrer" />

这张 online/offline 图对应到代码就是两个入口：online 路径在训练 step 内部调用 target model 拿 hidden states，offline 路径把 hidden states 当成 dataset 字段读进来。读 SpecForge 源码时，只要抓住这件事就不容易迷路：`Eagle3Model` 处理的是共同的 TTT/unroll/loss 逻辑，online/offline 差别主要在 hidden states 从哪里来。GPT-OSS 那篇 LMSYS blog 的价值也在这里，它说明 SpecForge 不是只适配一种 Llama-like 模型，而是把 target backend 做成可扩展层，新模型结构变了以后只需要把 hidden states、tokenizer、draft config 对齐。

# 0x2. Slides 逐页解读

### Slide 1：SpecForge：Speculative Decoding Models 的训练框架

<img src="https://files.mdnice.com/user/59/ba69be4e-bd0e-4084-b9e4-6e50b89d6769.png" referrerpolicy="no-referrer" />

SpecForge 解决的是投机解码的“训练侧”：SGLang 已经能 serve EAGLE/EAGLE3 draft model，但 draft model 从哪里来、怎么跟目标模型 hidden states 对齐、怎么处理特殊 attention mask，是另一个完整工程。

### Slide 2：目录：从投机解码到自定义训练

<img src="https://files.mdnice.com/user/59/55dc0967-bfd2-483e-b9ec-d9f60ad7f763.png" referrerpolicy="no-referrer" />

目录顺序很清楚：先解释 speculative decoding 为什么能降低 decode latency，再讲 EAGLE3 和 SpecForge，接着落到 GPT-OSS、Flex Attention、VLM、LoRA 和自定义训练。它不是单模型脚本，而是一个训练框架。

### Slide 3：为什么 decode 阶段值得用投机解码

<img src="https://files.mdnice.com/user/59/1e043b4d-8c64-4611-8930-890a66145a1e.png" referrerpolicy="no-referrer" />

小 batch decode 很容易 memory-bound。目标模型每步只出一个 token，GPU 算力吃不满。投机解码用一个便宜 draft model 一次猜多个 token，再让 target model 并行验证，用更多计算换更少串行步数。

### Slide 4：EAGLE3 在 SGLang 里的收益

<img src="https://files.mdnice.com/user/59/355584e2-8a73-4a6f-ba0d-af9f6390963d.png" referrerpolicy="no-referrer" />

EAGLE3 的意义在于 draft model 不只看 token，还看 target model 的中间 hidden states。SGLang 侧能拿到较高 acceptance length，slides 上给出 Llama3.1-8B 约 2.4x 的例子。

### Slide 5：SpecForge 的定位：训练 draft model

<img src="https://files.mdnice.com/user/59/c4aaf488-684b-4041-91bc-71da68ff0e88.png" referrerpolicy="no-referrer" />

SpecForge 的定位是“把 EAGLE3 训练流程标准化”。它把 target hidden states 生成、draft forward、TTT unroll、loss/accuracy 计算封在框架里，用户不用从 SafeAILab/EAGLE 的训练脚本开始手搓。

### Slide 6：开箱支持主流模型和 SGLang

<img src="https://files.mdnice.com/user/59/da6fe822-7840-4de7-99de-ed4b6c49b40a.png" referrerpolicy="no-referrer" />

开箱支持并不只是模型名列表。GPT-OSS、Qwen、Llama、Qwen2.5-VL 这类模型的 tokenizer、hidden state 层选择、attention backend、FSDP/TP 都有差异。SpecForge 把这些差异放进 target/draft backend。

### Slide 7：为什么把训练框架放到 SGLang 生态里

<img src="https://files.mdnice.com/user/59/3582d06f-f627-4756-a79e-910141a466bc.png" referrerpolicy="no-referrer" />

放到 SGLang 生态的好处是训练和 serving 能用同一套模型假设：训练出来的 draft model 可以直接给 SGLang speculative decoding 使用，评测 acceptance length 也更接近线上行为。

### Slide 8：EAGLE3 的 Training-Time Test

<img src="https://files.mdnice.com/user/59/e5d6a54e-310e-41a3-bb78-3b0e2a86458e.png" referrerpolicy="no-referrer" />

这页左边的图要从 target model 和 draft model 两块看。target model 从 train data 经过 embedding 和多层 decoder，取出 low/mid/high 三层 hidden states；高层 hidden 直接喂给 draft 侧的 high hidden，三层 hidden 还会被送到 draft 侧做融合。draft model 这边先把 low/mid/high hidden 过一层 FC，得到 `g hidden`，再和输入 token 的 embedding 融成 `fuse hidden`，送进 draft decoder，最后用 `plogp_loss` 训练 draft 对下一批 token 的预测。

右侧文字讲的 Training-Time Test 不是一个评测脚本，而是训练结构的一部分。EAGLE3 训练时要模拟多步生成：draft 第一步生成的 token 会进入下一步输入，mask、position、hidden state 都跟着移动。这个递归 loop 让 draft model 在训练时就接触到“前一步预测偏离 target”之后的状态。SpecForge 这里做的事，是把官方 EAGLE3 里这套特殊 attention mask 和递归数据循环收进框架，用户不用每接一个模型都重新写一遍 TTT。

### Slide 9：online/offline hidden states 两种训练路径

<img src="https://files.mdnice.com/user/59/fcdcf4b3-1bd3-4dfe-8b1d-ba724b1060a9.png" referrerpolicy="no-referrer" />

Online 路径在上半部分。数据从 `Train Data -> embedding -> Target Model` 进入 target，target 一边输出 low/mid/high hidden，三层 hidden concat 后过 FC 得到 `Fuse Hidden`，另一边把 final hidden 送到 `Target LM Head` 得到 logits。虚线框右侧就是 Training-Time Test：训练输入 ids 自己也会 embedding，然后和 fuse hidden 一起喂 draft model；target logits 和 draft 输出一起计算 `plogp_loss`，其中 slide 特别标了 “Left Shift Logits and input ids”，也就是 target logits 和训练 token 要错一位对齐。

Offline 路径在下半部分。左边 `SGLang Phase` 只负责提前跑 target model，把 high/mid/low/final hidden 写到 disk；右边 `SpecForge Phase` 训练时从 disk 读 hidden，再走 concat hidden、fuse hidden、draft model 和 `plogp_loss`。两条路径的训练主体一致，差别是 hidden states 在训练时生成，还是提前生成后当成数据字段读。

### Slide 10：Online & Offline Training 对比

<img src="https://files.mdnice.com/user/59/706d8a11-191b-4445-93e5-52a67d08a802.png" referrerpolicy="no-referrer" />

表格有四行。Target Model Usage 这一行说 online 训练会在训练期间调用 target model，offline 只在数据准备阶段用 target model。Disk Space Requirement 这一行对应最直观的代价：online 几乎不存 hidden states，磁盘压力低；offline 要落盘 low/mid/high/final hidden，slides 给的 UltraChat + ShareGPT 例子大约需要 12TB。GPU Requirement 则反过来：online 训练时 target model 和 draft 训练同场，target 大时 GPU 压力更高；offline 训练阶段只加载 draft，最低可以 1 张 GPU 跑起来。

最后一行 One-liner Rationale 可以理解成选择原则。online 是 “generates hidden states on the fly”，适合不想维护巨大中间数据、且训练资源够放 target 的场景；offline 是 “precomputes hidden states once and reuses them efficiently”，适合反复训练 draft、扫超参、或者 target model 很重但磁盘能承受的场景。

### Slide 11：GPT-OSS EAGLE 的例子

<img src="https://files.mdnice.com/user/59/6b3f5677-0d2f-4b8c-9514-0f3c35fde91e.png" referrerpolicy="no-referrer" />

GPT-OSS 的例子说明 SpecForge 不是只服务 Llama。开源模型结构变化快，如果训练框架把 target model 细节写死，很快就不能用。SpecForge 把 GPT-OSS 的 target backend 单独适配，draft 侧保持 EAGLE3 逻辑。图里的 acceptance length 对比也在提醒我们：draft model 训练质量会直接体现在 serving 侧吞吐上。

### Slide 12：Flex Attention 降显存和提速

<img src="https://files.mdnice.com/user/59/5a6bd810-b1cd-402a-8fa4-603949d09036.png" referrerpolicy="no-referrer" />

Flex Attention 这页有两张曲线。左边是速度对比，横轴是 sequence length，纵轴是时间；蓝线是普通 Eagle(SDPA)，红线是 Flex Attention。序列越长，蓝线抬得越快，红线增长更慢。右边是显存对比，蓝线在长序列处冲到九十多 GB，红线仍在十 GB 左右。底部小字写的是 10-20x less memory 和 H200 上约 2x speedup。

这个结果来自 EAGLE3 的 mask 形态。TTT 递归展开后，attention 依赖不是普通 causal mask，如果直接交给 SDPA 做 dense mask，会把很多本来不需要看的位置也展开成显存和计算。SpecForge 用 PyTorch Flex Attention 把依赖关系写成 block mask，再配合 DynamicCache 维护递归 unroll 的 KV 状态。它优化的不是一个单独 kernel 名字，而是把 TTT 这类稀疏 attention pattern 用更贴近语义的形式表达出来。

### Slide 13：VLM 也能训练 EAGLE3 draft

<img src="https://files.mdnice.com/user/59/cbb669a3-d453-4a4b-9f55-6f7b894090d4.png" referrerpolicy="no-referrer" />

VLM 页最关键的是 hidden states 不只来自文本。Qwen2.5-VL 这类模型要处理 image grid、mrope position id、视觉 token 和文本 token 对齐，SpecForge 的 VLM wrapper 会把目标模型输出的多层 hidden states 接到 EAGLE3 draft。

### Slide 14：LoRA 与 speculative decoding 共存

<img src="https://files.mdnice.com/user/59/f42be788-ad7c-41ce-95bf-d47fa235fe62.png" referrerpolicy="no-referrer" />

LoRA 页讲的是部署现实：线上 base model 可能同时挂多个 LoRA adapter。如果 speculative decoding 只更新 base draft，不处理 adapter 侧差异，acceptance rate 会掉。SpecForge/SGLang 需要让 draft/base 的 LoRA 状态对齐。

### Slide 15：自定义训练参数和 chat template

<img src="https://files.mdnice.com/user/59/df13584f-dc0c-4d99-b5e3-d1a95ed47ccd.png" referrerpolicy="no-referrer" />

自定义训练首先是把参数和数据格式接上。左侧代码给的是 online 训练入口：`torchrun --standalone --nproc_per_node 8 ./scripts/train_eagle3_online.py`，核心参数包括 `--target-model-path meta-llama/Llama-3.1-8B-Instruct`、`--draft-model-config ./configs/llama3-8B-eagle3.json`、`--train-data-path ./cache/dataset/sharegpt.jsonl`、`--output-dir ./outputs/llama3-8b-eagle3`、`--num-epochs 10`、`--batch-size 1`、`--learning-rate 1e-4`、`--max-length 2048`、`--chat-template llama3` 和 `--cache-dir ./cache`。这些参数把 target model、draft config、训练数据、输出目录和上下文长度都显式传进去。

右侧是 chat template 注册。SpecForge 会在 `specforge.data.template.py` 的 `TEMPLATE_REGISTRY` 里注册 `ChatTemplate`，字段包括 `assistant_header`、`user_header`、`system_prompt`、`end_of_turn_token`。模板会直接决定 token 边界，后面的 target hidden states、labels、loss mask 都跟着它走；chat template 一错，训练看起来能跑，但 draft 学到的对齐关系会偏，acceptance length 通常会很难看。

### Slide 16：自定义 target model 和 draft model

<img src="https://files.mdnice.com/user/59/16c88cf3-70ef-48ea-be8a-5b95cdeb3774.png" referrerpolicy="no-referrer" />

这一页才是模型接入入口。左侧 target model 部分说，如果只是 HuggingFace 能直接加载的小模型，改 `--target-model-path` 就够；如果模型太大、需要 tensor parallel，就要在 `specforge.modeling.target` 目录实现自己的并行版本。截图里的代码提示自定义 target model 要继承 distributed target model 类，并实现类似 `load_weights` 这样的入口，再注册到 Auto target model。

右侧 draft model 部分要求新建类继承 `Eagle3DraftModel`，位置在 `specforge.modeling.draft.base.py`。draft 侧要把 backbone、embedding、lm head、projection 这些部件拼好，并注册 draft config/model mapping。EAGLE3 训练主体可以复用，但 hidden states 层选择、维度、position ids、mrope 或多模态位置编码这些模型相关细节必须在 wrapper 里说清楚。

### Slide 17：结束页

<img src="https://files.mdnice.com/user/59/dba1426a-9d8c-4b84-9560-f6d2b2b9abca.png" referrerpolicy="no-referrer" />

结束页没有新的技术点。回到主线看，SpecForge 的价值是把 EAGLE3 训练里最难维护的 hidden states、TTT unroll、特殊 mask 和模型适配收进框架里，让训练出来的 draft model 能更自然地进入 SGLang serving。

# 0x3. 关键代码拆解

下面按源码走一遍，主要看 `specforge/core/eagle3.py`、`specforge/modeling/target/eagle3_target_model.py`、`specforge/data/preprocessing.py`、`specforge/core/eagle3_adapters.py` 和 `specforge/modeling/draft/flex_attention.py`。这几个文件正好对应 slides 里的 hidden states 采集、online/offline 数据流、TTT unroll、Flex Attention 和自定义 backend。

先看 target hidden states 从哪里来。HF backend 没有直接打开 `output_hidden_states=True` 去拿所有层，因为那会把显存压力抬得很高。SpecForge 在 `HFEagle3TargetModel.generate_eagle3_data` 里只给 EAGLE3 需要的三层注册 forward hook：

```python
def set_aux_hidden_states_layers(self, aux_hidden_states_layers=None):
    if aux_hidden_states_layers is None:
        num_layers = self.model.config.num_hidden_layers
        aux_hidden_states_layers = [1, num_layers // 2 - 1, num_layers - 4]
    self.aux_hidden_states_layers = aux_hidden_states_layers
    assert len(self.aux_hidden_states_layers) == 3
```

```python
captured_states = {}
handles = []

def get_hook(layer_idx):
    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured_states[layer_idx] = hidden
    return hook

layers = self._get_transformer_layers()
target_indices = self.aux_hidden_states_layers

for idx in target_indices:
    handles.append(layers[idx].register_forward_hook(get_hook(idx)))

try:
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=False,
        output_attentions=False,
        output_router_logits=False,
        use_cache=False,
    )
finally:
    for handle in handles:
        handle.remove()

hidden_states = torch.cat(
    (
        captured_states[target_indices[0]],
        captured_states[target_indices[1]],
        captured_states[target_indices[2]],
    ),
    dim=-1,
)
```

这段代码对应 Slide 8/9 里的“三层 hidden states + target logits”。EAGLE3 需要的不是完整 activation dump，而是低层、中层、高层各一份。hook 只截这三层，`output_hidden_states=False` 也能正常跑，这一点很重要：online 训练本来就要让 target model 常驻，如果再把每层 hidden states 都保留下来，显存会很快顶住。

拿到三层 hidden states 后，训练主循环在 `OnlineEagle3Model.forward`。类注释写得很直接：

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

进入 `forward` 后先处理 target logits。`_compute_target_p_padded` 会把 target token 分布整理成 TTT unroll 需要的滑动窗口；随后把三层 hidden states 从 `3 * hidden_size` 投影回 `hidden_size`：

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

draft model 侧的投影不是抽象概念，`LlamaForCausalLMEagle3.project_hidden_states` 就是一层线性层：

```python
def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
    # eagle 3 requires hidden states from 3 layers
    assert hidden_states.size(-1) == self.config.hidden_size * 3
    return self.fc(hidden_states)
```

`Eagle3DraftModel` 把 draft 侧必须实现的接口收得很窄：embedding、hidden state projection、logits 三件事。新模型接入时，最先要对齐的就是这几个函数，而不是从训练 loop 开始改。

```python
class Eagle3DraftModel(PreTrainedModel, ABC):
    @abstractmethod
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...
```

TTT loop 是代码里需要仔细看的部分。每一步都从 adapter 取当前 step 的 view，然后重新 embed input ids、跑 draft backbone、算 loss/accuracy，并把下一步需要的 input/mask 往右 pad 一格：

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

    hidden_states = hidden_states_out
    logits = self.draft_model.compute_logits(hidden_states)
    acc, loss = self._acc_and_loss(
        logits=logits,
        target_p=state.target_p,
        position_mask=state.position_mask,
        loss_mask=state.loss_mask,
        adapter=adapter,
    )

    if not is_last:
        global_input_ids = padding(global_input_ids, left=False)
        position_mask = padding(position_mask, left=False)
        loss_mask = padding(loss_mask, left=False)
```

这就是 Slide 8 里 Training-Time Test 的代码形态。普通 LM 训练只需要一次 teacher forcing，EAGLE3 这里要模拟“draft 连续猜多步”的状态，所以每一步的 `input_ids`、`loss_mask`、`position_mask` 都会变。`hidden_states = hidden_states_out` 也说明下一步不是重新喂 target hidden states，而是接着 draft 自己的 hidden state 走，这才会逼 draft model 学会多步生成后的误差传播。

adapter 负责把不同 attention backend 的 view 切出来。SDPA/FA 路径是完整序列 view：

```python
class SdpaLikeAdapter(BackendAdapter):
    def step_view(...):
        target_p = target_p_padded[:, idx : idx + seq_length, :].contiguous()
        return StepState(
            input_ids=global_input_ids,
            hidden_states=hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            target_p=target_p,
            position_mask=position_mask,
            loss_mask=loss_mask,
        )
```

USP 路径会按 sequence parallel rank 切本地 chunk，并保留 `ttt_length` 的 overlap。这个细节对长序列训练很关键：每张卡只看自己的 local sequence，但 TTT shift 不能把边界处的上下文直接切断。

```python
class UspAdapter(BackendAdapter):
    def step_view(...):
        usp_chunk_size = seq_length - ttt_length
        target_p = target_p_padded[:, idx : idx + usp_chunk_size, :]
        return StepState(
            input_ids=global_input_ids[:, :usp_chunk_size],
            hidden_states=hidden_states[:, :usp_chunk_size, :],
            position_ids=position_ids[:, : usp_chunk_size * self.sp_ulysses_degree],
            attention_mask=attention_mask[:, :usp_chunk_size],
            target_p=target_p,
            position_mask=position_mask[:, :usp_chunk_size, :],
            loss_mask=loss_mask[:, :usp_chunk_size, :],
        )

    def reduce_loss(self, loss: torch.Tensor) -> torch.Tensor:
        loss = dist_nn.all_reduce(loss, op=dist.ReduceOp.SUM, group=self.sp_group)
        return loss / self.sp_world_size
```

online/offline 的差别不是训练算法变了，而是 hidden states 的来源变了。offline 数据集直接从磁盘读 `aux_hidden_state` 和 `hidden_state`，再喂给同一套 EAGLE3 训练逻辑：

```python
class OfflineEagle3Dataset(torch.utils.data.Dataset):
    @staticmethod
    def process_data(data, max_len, transform=None):
        hidden_state = data["aux_hidden_state"].squeeze(0)[:max_len][None, :]
        target = data["hidden_state"].squeeze(0)[:max_len][None, :]

        input_ids = data["input_ids"][:max_len][None, :]
        loss_mask = data["loss_mask"][:max_len][None, :]
        loss_mask[0, -1] = 0

        new_data["attention_mask"] = torch.ones_like(loss_mask, dtype=torch.long)
        new_data["loss_mask"] = loss_mask
        new_data["target"] = target
        new_data["hidden_state"] = hidden_state
        new_data["input_ids"] = input_ids
        return new_data
```

如果打开 USP 预处理，offline 数据也会在 dataset 阶段按 SP rank 切块，并额外加上 TTT overlap：

```python
chunk_size = (global_len + sp_size - 1) // sp_size
start = sp_rank * chunk_size
local_len = chunk_size + ttt_length

new_data["hidden_state"], _ = _slice_and_pad(data["aux_hidden_state"])
new_data["target"], _ = _slice_and_pad(data["hidden_state"])
new_data["input_ids"], valid_len = _slice_and_pad(input_ids)
```

所以 Slide 9/10 的 online/offline 对比可以翻译成一句代码侧判断：online 把 `generate_eagle3_data` 放进训练 step，offline 把 `aux_hidden_state` 变成 dataset 字段。前者吃 GPU，后者吃磁盘。

Flex Attention 路径对应 Slide 12 的显存优化。`OnlineEagle3Model` 里当 backend 是 `flex_attention` 时改用 `DynamicCache`，mask 收缩交给 attention module：

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

真正调用 Flex Attention 的地方在 `specforge/modeling/draft/flex_attention.py`。这里用了 singleton，把 `torch.compile(flex_attention)` 的编译结果缓存起来，避免每次 forward 都重新触发编译：

```python
class WrappedFlexAttention:
    _instance = None
    _is_flex_compiled = False
    _compiled_flex_attention = None

    @torch.compiler.disable(recursive=False)
    def __init__(self):
        if not self._is_flex_compiled:
            self._compiled_flex_attention = torch.compile(flex_attention)
            self._is_flex_compiled = True

def compile_friendly_flex_attention(query, key, value, **kwargs):
    flex_attention_compiled = (
        WrappedFlexAttention()() if not is_torchdynamo_compiling() else flex_attention
    )
    return flex_attention_compiled(query, key, value, **kwargs)
```

这段代码能解释 slides 上“Flex Attention 降显存和提速”的来源。EAGLE3 TTT 的 mask 不是普通 causal mask，直接展开成 dense attention mask 会很难受；Flex Attention 更适合表达这种 block mask，同时 `DynamicCache` 让递归 unroll 的 KV 状态不用每步都用同一种 list cache 语义硬拼。

VLM 的支持也不是在外面套一层 processor 就完了。`QwenVLOnlineEagle3Model.forward` 在训练 step 里先调用 target model 准备多模态数据，再走和文本 EAGLE3 类似的 projection/TTT：

```python
hidden_states, target, loss_mask, input_ids = self._prepare_data(
    input_ids, attention_mask, loss_mask, pixel_values, image_grid_thw
)

target_p_padded, position_mask = _compute_target_p_padded(
    target=target,
    t2d=self.draft_model.t2d,
    loss_mask=loss_mask,
    length=self.length,
)

hidden_states = self.draft_model.project_hidden_states(hidden_states)
```

Qwen2.5-VL 还要处理 MRoPE。代码里会调用 target model 的 `get_rope_index`，用 `image_grid_thw` 和 attention mask 算 position id：

```python
position_ids, rope_deltas = self.target_model.model.get_rope_index(
    input_ids,
    image_grid_thw,
    None,
    second_per_grid_ts=None,
    attention_mask=attention_mask_tensor,
)
```

这对应 Slide 13：VLM 训练 draft 的难点不在“多了图片”，而是视觉 token、文本 token、MRoPE position、loss mask 必须同时对齐。只要这些错一个，训练 loss 可能还会下降，但 serving 的 acceptance length 会很差。

GPT-OSS 和自定义模型接入则落在 target/draft backend 和 chat template。`template.py` 里有专门的 `gpt-oss` 模板，使用 `openai-harmony` parser：

```python
TEMPLATE_REGISTRY.register(
    name="gpt-oss",
    template=ChatTemplate(
        assistant_header=None,
        user_header=None,
        system_prompt=None,
        end_of_turn_token=None,
        parser_type="openai-harmony",
    ),
)
```

这个点和 Slide 11/15 是连着的。GPT-OSS 的 chat format 不是普通 `<|im_start|>` 风格，训练数据的 assistant 边界、loss mask、target hidden states 都依赖 parser。SpecForge 把 template 放进 registry，至少保证“同一份数据该从哪里算 loss”这件事有一个统一入口。

最后看一下训练脚本参数。GPT-OSS online 训练例子会指定 target backend 为 SGLang：

```bash
torchrun --nproc_per_node $NUM_GPUS scripts/train_eagle3.py \
  --target-model-path openai/gpt-oss-20b \
  --draft-model-config configs/gpt-oss-20B-eagle3.json \
  --train-data-path cache/dataset/perfect-blend-gptoss-20B.jsonl \
  --chat-template gpt-oss \
  --tp-size $TP_SIZE \
  --target-model-backend sglang
```

VLM 例子则显式打开 `--is-vlm`，并把像素范围也放进训练参数：

```bash
torchrun --nproc_per_node $NUM_GPUS scripts/train_eagle3.py \
  --target-model-path Qwen/Qwen2.5-VL-7B-Instruct \
  --chat-template qwen2-vl \
  --is-vlm \
  --min-pixels 50176 \
  --max-pixels 802816
```

这些参数看起来琐碎，但它们正是 SpecForge 这种框架的价值所在：训练 EAGLE3 draft 不只是写一个 loss，真正麻烦的是把 target backend、template、hidden states 层选择、attention backend、VLM position 这些小齿轮都咬上。

# 0x4. 小结

SpecForge 这套代码的工程边界比较清楚：target model 只负责产 logits 和三层 hidden states，draft model 只暴露 embedding/projection/logits/backbone，训练 loop 负责 TTT unroll，adapter 负责不同 attention/sequence parallel 的 view。边界清楚以后，GPT-OSS、Qwen2.5-VL、LoRA 和后续新模型才有接入空间。SGLang 负责把 draft model 用到 serving 侧，SpecForge 负责让这个 draft model 训练得出来、复现得了。
