> 我的课程笔记，欢迎关注：https://github.com/BBuf/how-to-optim-algorithm-in-cuda 。

> 这里简要解析了一下SGLang v0.4版本中针对DeepSeek模型引入的MLA Data Parallelism Attention优化。这个优化可以通过Data Parallelism的方式共享KV Head来避免在每个TP Worker中都重复计算KV Head，这对于DeepSeek 系列模型来说非常有用，因为它的MLA KV Head无法使用TP的方式正常切分多个GPU中，所以只能在不同RANK上复制，但是因为启用了TP就会导致KV Cache的占用比MLA Data Parallelism Attention高TP倍，因为要计算TP次。大家如果对多节点的MLA Data Parallelism Attention实现感兴趣可以看 https://github.com/sgl-project/sglang/pull/2925 。

# 0x0. 前言

SGLang 在 v0.4 版本中针对 DeepSeek V2/V3/R1 引入了一个 Data Parallelism Attention 优化，这里尝试解读一下。原始的介绍见：https://lmsys.org/blog/2024-12-04-sglang-v0-4/#data-parallelism-attention-for-deepseek-models ，翻译一下这里的描述：

我们最常用的并行策略是张量并行。但是，对于某些模型，这可能不是最有效的策略。例如，DeepSeek 模型使用 MLA 机制，只有一个 KV 头。如果我们在 8 个 GPU 上使用张量并行，它将导致 KV 缓存的冗余和不必要的内存使用。

为了克服这个问题，我们为 DeepSeek 模型实现了数据并行 (DP) 的多头潜在注意 (MLA) 机制，以提高推理的吞吐量。通过对注意力组件采用 DP，我们可以大大减少 KV 缓存，从而允许使用更大的批量大小。在我们的 DP 注意力实现中，每个 DP worker都独立处理不同类型的批处理 (prefill、decode、idle)，然后将注意力处理后的数据在所有worker之间 all-gather，以便在 Mixture-of-Experts (MoE) 层中使用。最后，在 MoE 层中处理完毕后，数据将被重新分配回每个worker。下图展示了这个想法。

![](https://files.mdnice.com/user/59/d207afbd-a1bf-4c98-ae88-6efd2750736d.png)


如果你看这个描述还没有理解到或者不太清楚怎么实现，你可以继续阅读本文的剩下部分。MLA Data Parallelism Attention 在单节点上的的核心实现由 https://github.com/sgl-project/sglang/pull/1970 这个PR完成，我下面就以高到低的视角来理解下这个feature对应的工程实现。

# 0x1. 模型实现上的改动

我这里把SGLang DeepSeek 的模型实现精简了一下，只留下和使用MLA DP Attention相关的逻辑，这样可以快速看出MLA DP Attention相比于普通的张量并行模式的核心改动。

```python
class DeepseekV2AttentionMLA(nn.Module):
    """DeepSeek V2模型的多头注意力层，支持MLA(Memory-Latency-Aware)优化和数据并行。
    
    该模块实现了两种并行策略:
    1. Data Parallel (DP): 使用ReplicatedLinear层，每个设备都有完整的参数副本
    2. Tensor Parallel (TP): 使用ColumnParallelLinear和RowParallelLinear层，在设备间分片参数
    """
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,          # 隐藏层维度
        num_heads: int,            # 注意力头数量
        qk_nope_head_dim: int,     # 不使用旋转位置编码的Q/K头维度
        qk_rope_head_dim: int,     # 使用旋转位置编码的Q/K头维度
        v_head_dim: int,           # V头维度
        q_lora_rank: int,          # Q矩阵的LoRA秩
        kv_lora_rank: int,         # KV矩阵的LoRA秩
        rope_theta: float = 10000, # RoPE位置编码的theta参数
        rope_scaling: Optional[Dict[str, Any]] = None,  # RoPE缩放配置
        max_position_embeddings: int = 8192,  # 最大位置编码长度
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        layer_id=None,             # 层ID
        use_dp=False,              # 是否使用数据并行
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        
        # 获取张量并行的世界大小
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        # 如果使用DP，则每个设备使用所有头；否则在设备间分片
        self.num_local_heads = num_heads if use_dp else num_heads // tp_size

        if use_dp:
            # 数据并行模式：使用ReplicatedLinear，每个设备都有完整的参数副本
            if self.q_lora_rank is not None:
                # 使用LoRA时的Q投影
                self.q_a_proj = ReplicatedLinear(
                    self.hidden_size,
                    self.q_lora_rank,
                    bias=False,
                    quant_config=quant_config,
                )
                self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
                self.q_b_proj = ReplicatedLinear(
                    q_lora_rank,
                    self.num_heads * self.qk_head_dim,
                    bias=False,
                    quant_config=quant_config,
                )
            else:
                # 不使用LoRA时的Q投影
                self.q_proj = ReplicatedLinear(
                    self.hidden_size,
                    self.num_heads * self.qk_head_dim,
                    bias=False,
                    quant_config=quant_config,
                )
            # KV和输出投影
            self.kv_b_proj = ReplicatedLinear(
                self.kv_lora_rank,
                self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
                bias=False,
                quant_config=quant_config,
            )
            self.o_proj = ReplicatedLinear(
                self.num_heads * self.v_head_dim,
                self.hidden_size,
                bias=False,
                quant_config=quant_config,
            )
        else:
            # 张量并行模式：使用ColumnParallelLinear和RowParallelLinear在设备间分片参数
            if self.q_lora_rank is not None:
                self.q_a_proj = ReplicatedLinear(
                    self.hidden_size,
                    self.q_lora_rank,
                    bias=False,
                    quant_config=quant_config,
                )
                self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
                self.q_b_proj = ColumnParallelLinear(
                    q_lora_rank,
                    self.num_heads * self.qk_head_dim,
                    bias=False,
                    quant_config=quant_config,
                )
            else:
                self.q_proj = ColumnParallelLinear(
                    self.hidden_size,
                    self.num_heads * self.qk_head_dim,
                    bias=False,
                    quant_config=quant_config,
                )
            self.kv_b_proj = ColumnParallelLinear(
                self.kv_lora_rank,
                self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
                bias=False,
                quant_config=quant_config,
            )
            self.o_proj = RowParallelLinear(
                self.num_heads * self.v_head_dim,
                self.hidden_size,
                bias=False,
                quant_config=quant_config,
            )

def all_gather(
    input_tensor: torch.Tensor, forward_batch: ForwardBatch, rank, world_size, group
):
    """在数据并行模式下收集并同步各个设备上的张量。
    
    Args:
        input_tensor: 输入张量
        forward_batch: 前向计算批次信息
        rank: 当前设备的rank
        world_size: 并行设备总数
        group: 通信组
        
    Returns:
        tuple: (gathered_tensors, start_index, end_index)
            - gathered_tensors: 收集到的所有设备的张量
            - start_index: 当前设备数据的起始索引
            - end_index: 当前设备数据的结束索引
    """
    if world_size == 1:
        return input_tensor

    # 获取每个设备的token数量
    all_lens = forward_batch.global_num_tokens
    max_len = max(forward_batch.global_num_tokens)

    # 对输入张量进行填充，使其长度达到max_len
    padded_tensor = torch.nn.functional.pad(
        input_tensor, (0, 0, 0, max_len - input_tensor.shape[0])
    )

    # 使用all_gather收集所有设备的张量
    torch.distributed.all_gather_into_tensor(
        forward_batch.gathered_buffer, padded_tensor, group=group
    )

    # 将收集到的张量按实际长度拼接
    gathered_tensors = torch.concat(
        [
            forward_batch.gathered_buffer[i * max_len : i * max_len + all_lens[i]]
            for i in range(world_size)
        ]
    )

    # 计算当前设备数据的起始和结束索引
    start_index = 0 if rank == 0 else sum(all_lens[:rank])
    end_index = start_index + all_lens[rank]

    return gathered_tensors, start_index, end_index


class DeepseekV2DecoderLayer(nn.Module):
    """DeepSeek V2模型的解码器层，支持数据并行注意力机制。"""
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # 根据配置决定是否启用数据并行注意力
        self.enable_dp_attention = (
            not global_server_args_dict["disable_mla"]
            and global_server_args_dict["enable_dp_attention"]
        )
        if self.enable_dp_attention:
            # 初始化数据并行相关的参数
            self.tp_rank = get_tensor_model_parallel_rank()
            self.tp_size = get_tensor_model_parallel_world_size()
            self.tp_group = get_tp_group().device_group

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # 数据并行模式下的前向计算
        if self.enable_dp_attention:
            # 收集所有设备的隐藏状态
            hidden_states, start_idx, end_idx = all_gather(
                hidden_states, forward_batch, self.tp_rank, self.tp_size, self.tp_group
            )
            # 执行Fused MoE MLP计算
            hidden_states = self.mlp(hidden_states)
            # 提取当前设备对应的部分
            hidden_states = hidden_states[start_idx:end_idx]

        return hidden_states, residual


class DeepseekV2ForCausalLM(nn.Module):
    """DeepSeek V2因果语言模型，支持数据并行和张量并行两种模式。"""
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = DeepseekV2Model(config, quant_config)
        
        if global_server_args_dict["enable_dp_attention"]:
            # 数据并行模式：使用ReplicatedLinear作为语言模型头
            self.lm_head = ReplicatedLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
            )
            # 跳过all_gather操作的LogitsProcessor
            self.logits_processor = LogitsProcessor(config, skip_all_gather=True)
        else:
            # 张量并行模式：使用ParallelLMHead
            self.lm_head = ParallelLMHead(
                config.vocab_size, config.hidden_size, quant_config=quant_config
            )
            self.logits_processor = LogitsProcessor(config)
```

从这个模型实现代码可以看到SGLang中针对DeepSeek模型的Data Parallelism Attention优化主要解决了模型在使用MLA Attention时KV缓存冗余的问题。该优化通过将传统的张量并行(TP)改为数据并行(DP)的方式来实现：在`DeepseekV2AttentionMLA`类中支持使用`ReplicatedLinear`层进行完整参数复制的DP模式和使用`ColumnParallelLinear/RowParallelLinea`r层进行参数分片的TP模式；通过`all_gather`函数实现DP worker间的数据同步，使得每个worker可以独立处理不同类型的批处理，然后在MoE层处理完后重新分配数据。这种并行策略的改变不仅减少了KV缓存的内存占用，还支持了更大的批处理大小，从而提高了模型的推理吞吐量。

在上面的all_gather实现中，我们发现`forward_batch`（`ForwardBatch`类型）维护了`global_num_tokens`和`gathered_buffer`两个成员变量来辅助我们在Fused MoE Layer之前做allgather以及计算完Fused MoE之后再Split。

接下来就关注一下和Data Parallelism Attention优化相关的更底层的改动，包括managers 和 model_executor 两大方面。实际上涉及到的改动包括SGLang的TPModelWorker(https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/tp_worker.py) 和 ModelRunner(https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner.py) 两个部分，当然还有负责`TpModelWorker`调度相关的Scheduler部分也做了对应修改，但改的东西其实不多，下面分点看一下。

# 0x2. model_executor 的改动

## `python/sglang/srt/model_executor/forward_batch_info.py` 的改动

![](https://files.mdnice.com/user/59/ab9d52c2-1636-4e3e-a2ed-edb5556204eb.png)
![](https://files.mdnice.com/user/59/28d32563-7c17-4728-9af7-1af50d884686.png)

首先，这里在`ForwardMode`类新增了一个新的模式`IDLE`，用于数据并行注意力机制。注释说明当某些worker没有序列做forward时，worker将处于IDLE状态（可以看文章开头那个图）。

接着，在`ForwardBatch`中增加了数据并行注意力相关的成员变量：
- `global_num_tokens`: 类型为`Optional[List[int]]`，初始值为None
- `gathered_buffer`: 类型为`Optional[torch.Tensor]`，初始值为None

最后，是对于`compute_erope_positions`方法的改动：当`global_num_tokens`不为None时，计算最大长度`max_len = max(ret.global_num_tokens)`；创建一个新的`gathered_buffer`张量，使用`torch.zeros`初始化设置张量的属性，包括`size`、`dtype`和`device`等。增加了对`forward_mode.is_idle()`的判断，如果是IDLE模式则直接返回ret。

## `python/sglang/srt/model_executor/model_runner.py` 的改动

![](https://files.mdnice.com/user/59/56021a9b-9b23-4578-a0a7-3116f0a20690.png)

这里只是增加了对`idel`模式的判断。

# 0x3. managers 的改动

这里主要改动的地方就是scheduler相关和data_parallel_controller，分别浏览一下。

## `python/sglang/srt/managers/data_parallel_controller.py` 的改动

![](https://files.mdnice.com/user/59/f370af6c-65cf-45fc-a1ad-5f1f9b590c9c.png)
![](https://files.mdnice.com/user/59/b9eec329-7d0d-4888-a1b7-847ac7a6874b.png)


从修改的流程来看，首先最外面的循环为每个数据并行(DP)等级创建一个专门的进程，这些进程同时处理数据并行和张量并行的计算。然后，每个进程被分配一个唯一的GPU（通过`base_gpu_id`递增实现）确保不同的数据并行rank使用不同的GPU资源。在通信上，使用`mp.Pipe`建立进程间的通信管道，并使用ZMQ套接字进行消息传递，最后所有reader都被收集到scheduler_pipe_readers列表中，用于后续的通信。

## `python/sglang/srt/managers/scheduler.py` 的改动

![](https://files.mdnice.com/user/59/3c45c5cb-7364-48fb-b22d-53af24c1f250.png)
![](https://files.mdnice.com/user/59/9f20fa67-a80b-4383-84e9-35f033f9ffad.png)
![](https://files.mdnice.com/user/59/e26e9442-72f4-49b1-b67c-222da8905e22.png)

这里需要关注的是新增的`prepare_dp_attn_batch`函数，它用来对每个DP worker的`local_num_tokens`进行allgather通信获得`global_num_tokens`，最后这个信息将用于我们在第一节提到在Fused MoE层之后把数据重新split开。

```python
def prepare_dp_attn_batch(self, local_batch: ScheduleBatch):
    # Check if other DP workers have running batches
    if local_batch is None:
        num_tokens = 0
    elif local_batch.forward_mode.is_decode():
        num_tokens = local_batch.batch_size()
    else:
        num_tokens = local_batch.extend_num_tokens

    local_num_tokens = torch.tensor(
        num_tokens, dtype=torch.int64, device=self.device
    )
    global_num_tokens = torch.empty(
        self.tp_size, dtype=torch.int64, device=self.device
    )
    torch.distributed.all_gather_into_tensor(
        global_num_tokens,
        local_num_tokens,
        group=self.tp_worker.get_tp_device_group(),
    )

    if local_batch is None and global_num_tokens.max().item() > 0:
        local_batch = self.get_idle_batch()

    if local_batch is not None:
        local_batch.global_num_tokens = global_num_tokens.tolist()

    return local_batch
```

# 0x4. 扩展

上面介绍的是单节点的原理和实现，如果要将这个Feature扩展到多个节点实现会比较复杂，x-AI的contributor在 https://github.com/sgl-project/sglang/pull/2925 实现了DP Attention的多节点扩展，目前在DeepSeek V3/R1等模型的多节点部署中都可以顺利开启这个优化。感兴趣的读者可以自行阅读和研究多节点实现这部分。

# 0x5. 总结

这里简要解析了一下SGLang v0.4版本中针对DeepSeek模型引入的MLA Data Parallelism Attention优化。这个优化可以通过Data Parallelism的方式共享KV Head来避免在每个TP Worker中都重复计算KV Head，这对于DeepSeek 系列模型来说非常有用，因为它的MLA KV Head无法使用TP的方式正常切分多个GPU中，所以只能在不同RANK上复制，但是因为启用了TP就会导致KV Cache的占用比MLA Data Parallelism Attention高TP倍，因为要计算TP次。大家如果对多节点的MLA Data Parallelism Attention实现感兴趣可以看 https://github.com/sgl-project/sglang/pull/2925 。







