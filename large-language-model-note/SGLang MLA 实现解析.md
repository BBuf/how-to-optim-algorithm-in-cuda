# 0x0. 前言

上回讲到 SGLang 中的 DP MLA 特性 [SGLang DP MLA 特性解读](https://mp.weixin.qq.com/s/X2uA507VbQVCv3JIQ8EtPA) ，这里简单回顾一下核心idea。之所以在 MLA 中使用DP的方式是因为 MLA 在存储 KV Cache的时候对于一个token 存储的shape是`(1, 1, kv_lora_rank+qk_rope_head_dim)`，而不是普通MHA下的`(1, kv_head_num, head_dim)`。这就导致如果按照以前的TP并行方式需要在每张卡上都维护重复的KV Cache才行，为了避免这个问题就引入DP让每张卡去维护它拥有的batch的全量KV Cache，我们就不需要在每个rank上都复制所有batch的KV Cache了。当然，这里还有个问题就是如果DP MLA出现了负载不均衡问题，必然会导致某些GPU处于等待状态，这个问题怎么解决呢？我目前也不清楚。

现在来到这次的话题，因为SGLang MLA除了DP之外还有挺多相关的Feature，所以打算在这里再梳理一下SGLang MLA的实现以及支持的Feature。9个月之前我在 [大模型KV Cache节省神器MLA学习笔记（包含推理时的矩阵吸收分析）](https://mp.weixin.qq.com/s/cBMrRUdM1IM0T1ji_ODxng) 这篇文章记录了一下学习 MLA 的学习笔记，那个时候是DeepSeek V2发布的时期。然后我在学习笔记中记录了一下 MLA 的原理以及矩阵吸收分析等，读者可以将这个笔记作为前置知识，我在本博客中将主要关注 SGLang 的 MLA 实现，欢迎捉虫。

这里的代码解读仍然采用从上到下的方式。

# 0x1. `DeepseekV2DecoderLayer`类速览

```python
class DeepseekV2DecoderLayer(nn.Module):
    """
    DeepseekV2模型的解码器层实现。
    
    该类实现了Deepseek V2模型的单个Transformer解码器层，包含自注意力机制和前馈神经网络。
    根据配置，可以使用不同类型的注意力机制(MLA或标准)和不同类型的前馈网络(MoE或标准MLP)。
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        is_nextn: bool = False,
    ) -> None:
        """
        初始化DeepseekV2解码器层。
        
        参数:
            config: 预训练模型的配置对象，包含模型结构参数
            layer_id: 当前层的ID，用于确定是否使用MoE以及在注意力计算中的位置信息
            quant_config: 可选的量化配置，用于模型量化
            is_nextn: 是否为nextn模型，影响是否使用MoE
        """
        super().__init__()
        # 保存隐藏层大小
        self.hidden_size = config.hidden_size
        # 获取RoPE（旋转位置编码）相关参数，如果配置中没有则使用默认值
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        
        # 确定是否启用数据并行注意力机制
        # 当MLA(多查询注意力)未禁用且启用了数据并行注意力时为True
        self.enable_dp_attention = (
            not global_server_args_dict["disable_mla"]
            and global_server_args_dict["enable_dp_attention"]
        )
        
        # 如果启用数据并行注意力，获取张量并行的相关信息
        if self.enable_dp_attention:
            self.tp_rank = get_tensor_model_parallel_rank()  # 当前张量并行的rank
            self.tp_size = get_tensor_model_parallel_world_size()  # 张量并行的总大小
            self.tp_group = get_tp_group()  # 张量并行的通信组
            
        # 根据是否禁用MLA选择不同的注意力机制实现
        if not global_server_args_dict["disable_mla"]:
            # 使用DeepseekV2AttentionMLA
            self.self_attn = DeepseekV2AttentionMLA(
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,  # 不使用位置编码的Q和K的头维度
                qk_rope_head_dim=config.qk_rope_head_dim,  # 使用位置编码的Q和K的头维度
                v_head_dim=config.v_head_dim,  # V的头维度
                q_lora_rank=(
                    config.q_lora_rank if hasattr(config, "q_lora_rank") else None
                ),  # 对应 query 压缩后的隐向量的维度 d'_c
                kv_lora_rank=config.kv_lora_rank,  # 对应 key-value 压缩后的隐向量维度 d_c
                rope_theta=rope_theta,  # RoPE的θ参数
                rope_scaling=rope_scaling,  # RoPE的缩放参数
                max_position_embeddings=max_position_embeddings,  # 最大位置编码长度
                quant_config=quant_config,  # 量化配置
                layer_id=layer_id,  # 层ID
                use_dp=self.enable_dp_attention,  # 是否使用数据并行注意力
            )
        else:
            # 使用标准的DeepseekV2Attention
            self.self_attn = DeepseekV2Attention(
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=(
                    config.q_lora_rank if hasattr(config, "q_lora_rank") else None
                ),
                kv_lora_rank=config.kv_lora_rank,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                max_position_embeddings=max_position_embeddings,
                quant_config=quant_config,
                layer_id=layer_id,
            )
            
        # 确定是否使用MoE（混合专家模型）作为前馈网络
        # 在以下情况使用MoE：
        # 1. 是nextn模型
        # 2. 配置中指定了路由专家数量，且当前层ID大于等于first_k_dense_replace，且层ID是moe_layer_freq的倍数
        if is_nextn or (
            config.n_routed_experts is not None
            and layer_id >= config.first_k_dense_replace
            and layer_id % config.moe_layer_freq == 0
        ):
            # 使用MoE作为前馈网络
            self.mlp = DeepseekV2MoE(config=config, quant_config=quant_config)
        else:
            # 使用标准MLP作为前馈网络
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
            )
            
        # 初始化层归一化，用于自注意力前的输入归一化
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 初始化层归一化，用于自注意力后的输出归一化
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        解码器层的前向传播函数。
        
        参数:
            positions: 位置编码张量，用于RoPE计算
            hidden_states: 输入隐藏状态
            forward_batch: 前向计算批次信息，包含模式、批大小等
            residual: 可选的残差连接张量，如果为None则使用hidden_states作为残差
            
        返回:
            hidden_states: 更新后的隐藏状态
            residual: 更新后的残差连接
        """
        # 自注意力部分
        # 只有在非空闲模式下才执行计算
        if not forward_batch.forward_mode.is_idle():
            # 如果没有提供残差，则使用当前隐藏状态作为残差，并对隐藏状态进行归一化
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                # 如果提供了残差，则同时对隐藏状态和残差进行归一化
                hidden_states, residual = self.input_layernorm(hidden_states, residual)

            # 执行自注意力计算
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
            
            # 对自注意力的输出和残差进行归一化
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )

        # 前馈神经网络部分
        if self.enable_dp_attention:
            # 如果启用了数据并行注意力，需要在计算MLP前收集所有进程的hidden_states
            hidden_states, start_idx, end_idx = all_gather(
                hidden_states, forward_batch, self.tp_rank, self.tp_size, self.tp_group
            )
            # 执行MLP计算
            hidden_states = self.mlp(hidden_states)
            # 只保留当前进程负责的部分
            hidden_states = hidden_states[start_idx:end_idx]
        else:
            # 标准MLP计算
            hidden_states = self.mlp(hidden_states)

        # 返回更新后的隐藏状态和残差
        return hidden_states, residual
```


这是一个上层接口，我们可以发现打开`disable_mla` MLA部分就会使用原始的DeepseekV2Attention实现，而默认情况下会使用DeepseekV2AttentionMLA的实现。

# 0x2. `DeepseekV2Attention` 类速览

```python
class DeepseekV2Attention(nn.Module):
    """
    DeepseekV2模型的注意力机制实现。
    
    该类实现了Deepseek V2模型的自注意力机制，支持张量并行和旋转位置编码(RoPE)。
    注意力机制包含了查询(Q)、键(K)和值(V)的投影，以及多头注意力的计算。
    """

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        layer_id=None,
    ) -> None:
        """
        初始化DeepseekV2注意力层。
        
        参数:
            config: 预训练模型的配置对象
            hidden_size: 隐藏层维度
            num_heads: 注意力头的数量
            qk_nope_head_dim: 不使用位置编码的Q和K的头维度
            qk_rope_head_dim: 使用位置编码的Q和K的头维度
            v_head_dim: V的头维度
            q_lora_rank: 对应query压缩后的隐向量的维度d'_c
            kv_lora_rank: 对应key-value压缩后的隐向量维度d_c
            rope_theta: RoPE的θ参数，默认为10000
            rope_scaling: RoPE的缩放参数，默认为None
            max_position_embeddings: 最大位置编码长度，默认为8192
            quant_config: 量化配置，默认为None
            layer_id: 层ID，用于注意力计算
        """
        super().__init__()
        # 保存层ID
        self.layer_id = layer_id
        # 保存隐藏层大小
        self.hidden_size = hidden_size
        # 不使用位置编码的Q和K的头维度
        self.qk_nope_head_dim = qk_nope_head_dim
        # 对应$d_h^R$， 表示应用了rope的 queries 和 key 的一个 head 的维度。
        self.qk_rope_head_dim = qk_rope_head_dim
        # 每一个注意力头的维度应该是两部分之和
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        # value 的一个注意力头的隐藏层为度
        self.v_head_dim = v_head_dim
        # 对应query压缩后的隐向量的维度d'_c
        self.q_lora_rank = q_lora_rank
        # 对应key-value压缩后的隐向量维度d_c
        self.kv_lora_rank = kv_lora_rank
        # 注意力头的总数量
        self.num_heads = num_heads
        # 获取张量并行的大小
        tp_size = get_tensor_model_parallel_world_size()
        # 确保注意力头的数量能被张量并行的大小整除
        assert num_heads % tp_size == 0
        # 计算每个并行进程的本地注意力头数量
        self.num_local_heads = num_heads // tp_size
        # 计算注意力缩放因子
        self.scaling = self.qk_head_dim**-0.5
        # 保存RoPE的θ参数
        self.rope_theta = rope_theta
        # 保存最大位置编码长度
        self.max_position_embeddings = max_position_embeddings

        # 根据是否提供q_lora_rank选择不同的Q投影实现
        if self.q_lora_rank is not None:
            # 使用两阶段投影：先将hidden_size投影到q_lora_rank，再投影到最终维度
            # 第一阶段投影：hidden_size -> q_lora_rank，对应paper公式中的W^DQ
            self.q_a_proj = ReplicatedLinear(
                self.hidden_size,
                self.q_lora_rank,
                bias=False,
                quant_config=quant_config,
            )
            # 对第一阶段投影的输出进行归一化
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            # q_b_proj 大小为 [q_lora_rank, num_heads * q_head_dim] = 
            # [q_lora_rank, num_attention_heads * (qk_nope_head_dim + qk_rope_head_dim)]
            # 对应上述公式中的W^UQ和W^QR合并后的大矩阵，仅仅只是内存放在一起
            self.q_b_proj = ColumnParallelLinear(
                q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
            )
        else:
            # 直接投影：hidden_size -> num_heads * qk_head_dim
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
            )

        # KV的第一阶段投影：hidden_size -> kv_lora_rank + qk_rope_head_dim
        # 与Q向量类似，KV向量的生成也是先投影到一个低维的 compressed_kv 向量（对应c_t^{KV}）
        # 再升维展开。具体的代码涉及 kv_a_proj_with_mqa 和 kv_b_proj 两个参数矩阵。
        # 其中 kv_a_proj_with_mqa 大小为 [hidden_size， kv_lora_rank + qk_rope_head_dim]
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
            # FIXME: quick fix for skip quantization
            prefix=f"self_attn.kv_a_proj_with_mqa",
        )
        # 对KV的第一阶段投影输出进行归一化
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        # KV的第二阶段投影：kv_lora_rank -> num_heads * (qk_nope_head_dim + v_head_dim)
        # kv_b_proj 大小为 [kv_lora_rank， num_heads * (q_head_dim - qk_rope_head_dim + v_head_dim)] 
        # 对应 paper 公式中的W^{UK}和W^{UV}。
        # 由于 W^{UK} 只涉及 non rope 的部分所以维度中把 qk_rope_head_dim 去掉了，就是上面的-号。
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
        )
        # 输出投影：将注意力的输出投影回原始隐藏层维度
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
        )
        # 设置RoPE的类型为"deepseek_yarn"
        rope_scaling["rope_type"] = "deepseek_yarn"
        # 初始化RoPE包装器
        self.rotary_emb = get_rope_wrapper(
            qk_rope_head_dim,
            rotary_dim=qk_rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=False,
            device=global_server_args_dict["device"],
        )

        # 如果提供了RoPE缩放参数，调整注意力缩放因子
        if rope_scaling:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
            scaling_factor = rope_scaling["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        # 初始化RadixAttention，用于高效的注意力计算
        # TODO, support head_size 192
        self.attn = RadixAttention(
            self.num_local_heads,
            256,  # 固定的内部维度，用于计算效率
            self.scaling,
            num_kv_heads=self.num_local_heads,
            layer_id=layer_id,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """
        注意力层的前向传播函数。
        
        参数:
            positions: 位置编码张量，用于RoPE计算
            hidden_states: 输入隐藏状态
            forward_batch: 前向计算批次信息
            
        返回:
            output: 注意力层的输出
        """
        # 计算查询向量Q
        if self.q_lora_rank is not None:
            # 使用两阶段投影计算Q
            # 第一阶段：hidden_states -> q_lora_rank
            q = self.q_a_proj(hidden_states)[0]
            # 对第一阶段输出进行归一化
            q = self.q_a_layernorm(q)
            # 第二阶段：q_lora_rank -> num_heads * qk_head_dim，并重塑为多头形式
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            # 直接投影计算Q，并重塑为多头形式
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
        
        # 将Q分为不使用位置编码的部分和使用位置编码的部分
        _, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        
        # 计算KV的第一阶段投影
        latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        # 分离KV的第一阶段输出和用于RoPE的部分
        kv_a, _ = latent_cache.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        # 为后续处理增加维度
        latent_cache = latent_cache.unsqueeze(1)
        # 对KV的第一阶段输出进行归一化
        kv_a = self.kv_a_layernorm(kv_a.contiguous())
        # 计算KV的第二阶段投影
        kv = self.kv_b_proj(kv_a)[0]
        # 重塑为多头形式
        kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        # 分离K的不使用位置编码部分和V
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        # 获取K的使用位置编码部分
        k_pe = latent_cache[:, :, self.kv_lora_rank :]
        
        # 应用RoPE到Q和K的位置编码部分
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        # 将处理后的位置编码部分放回Q
        q[..., self.qk_nope_head_dim :] = q_pe
        
        # 构建完整的K，包括不使用位置编码的部分和使用位置编码的部分
        k = torch.empty_like(q)
        k[..., : self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim :] = k_pe
        
        # 将Q、K、V填充到固定维度256（RadixAttention的内部维度），并重塑为适合注意力计算的形式
        q = torch.nn.functional.pad(q, [0, 256 - self.qk_head_dim], value=0).view(
            -1, self.num_local_heads * 256
        )
        k = torch.nn.functional.pad(k, [0, 256 - self.qk_head_dim], value=0).view(
            -1, self.num_local_heads * 256
        )
        v = torch.nn.functional.pad(v, [0, 256 - self.v_head_dim], value=0).view(
            -1, self.num_local_heads * 256
        )
        
        # 执行注意力计算
        attn_output = self.attn(q, k, v, forward_batch)
        
        # 重塑注意力输出，并只保留有效的V维度部分
        attn_output = attn_output.view(-1, self.num_local_heads, 256)[
            ..., : self.v_head_dim
        ].reshape(-1, self.num_local_heads * self.v_head_dim)
        
        # 通过输出投影将注意力输出投影回原始隐藏层维度
        output, _ = self.o_proj(attn_output)
        
        return output
```


对于`DeepseekV2Attention`类来说，和 DeepSeek V2/V3 的 HuggingFace 提供的 MLA 实现一样，这里的使用的KV Cache实际上是解压缩之后的MHA KV Cache的格式，不是缓存的Latent，并没有实现MLA的缓存节省效果。

# 0x3. `DeepseekV2AttentionMLA` 详解

由于这里的代码比较长，这里就只从流程出发，尽量少展示代码。先把DeepSeek MLA的公式截图到这里：


![](https://files.mdnice.com/user/59/6afdc732-4224-4a4d-aaf5-a87732020e68.png)


## 0x3.1 权重介绍

首先汇总一下init中的各个权重介绍，其实和`DeepseekV2Attention`上面的权重基本一致，只不过它对`self.kv_b_proj `做了一个拆分。

具体来说，`DeepseekV2AttentionMLA`初始化部分包含：

```python
# 使用两阶段投影：先将hidden_size投影到q_lora_rank，再投影到最终维度
# 第一阶段投影：hidden_size -> q_lora_rank，对应paper公式中的W^DQ
self.q_a_proj = ReplicatedLinear(
    self.hidden_size,
    self.q_lora_rank,
    bias=False,
    quant_config=quant_config,
)
# q_b_proj 大小为 [q_lora_rank, num_heads * q_head_dim] = 
# [q_lora_rank, num_attention_heads * (qk_nope_head_dim + qk_rope_head_dim)]
# 对应上述公式中的W^UQ和W^QR合并后的大矩阵，仅仅只是内存放在一起
self.q_b_proj = ColumnParallelLinear(
    q_lora_rank,
    self.num_heads * self.qk_head_dim,
    bias=False,
    quant_config=quant_config,
)
# KV的第一阶段投影：hidden_size -> kv_lora_rank + qk_rope_head_dim
# 与Q向量类似，KV向量的生成也是先投影到一个低维的 compressed_kv 向量（对应c_t^{KV}=w^{DKV}h_t）
# 再升维展开。具体的代码涉及 kv_a_proj_with_mqa 和 kv_b_proj 两个参数矩阵。
# 其中 kv_a_proj_with_mqa 大小为 [hidden_size， kv_lora_rank + qk_rope_head_dim]
self.kv_a_proj_with_mqa = ReplicatedLinear(
    self.hidden_size,
    self.kv_lora_rank + self.qk_rope_head_dim,
    bias=False,
    quant_config=quant_config,
    # FIXME: quick fix for skip quantization
    prefix=f"self_attn.kv_a_proj_with_mqa",
)
# KV的第二阶段投影：kv_lora_rank -> num_heads * (qk_nope_head_dim + v_head_dim)
# kv_b_proj 大小为 [kv_lora_rank， num_heads * (q_head_dim - qk_rope_head_dim + v_head_dim)] 
# 对应 paper 公式中的W^{UK}和W^{UV}。
# 由于 W^{UK} 只涉及 non rope 的部分所以维度中把 qk_rope_head_dim 去掉了，就是上面的-号。
self.kv_b_proj = ColumnParallelLinear(
    self.kv_lora_rank,
    self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
    bias=False,
    quant_config=quant_config,
)
# 输出投影：将注意力的输出投影回原始隐藏层维度
self.o_proj = RowParallelLinear(
    self.num_heads * self.v_head_dim,
    self.hidden_size,
    bias=False,
    quant_config=quant_config,
)
```


接着，初始化过程中还有两个`self.w_kc，self.w_vc`，它们分别对应了将`self.kv_b_proj`拆分后的$W^{UK}$和$W^{UV}$。拆分的代码如下：

```python
w = self_attn.kv_b_proj.weight
w_kc, w_vc = w.unflatten(
                    0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
                ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
self_attn.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
self_attn.w_vc = w_vc.contiguous().transpose(1, 2)
```

我们来分析一下这里的shape变化，先确定一下DeepSeek R1下相关的超参数：`self.qk_nope_head_dim = 128`，`self.v_head_dim = 128`，`self.kv_lora_rank = 512`，`self.num_heads = 128`，w 的形状为 `[32768, 512]`，即 `[128*(128+128), 512]`。

`w.unflatten(0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim))` 这一步将 w 的第一个维度 32768 重新组织为两个维度 `[-1, 256]`，其中 256 = 128 + 128。这里的 `-1` 会自动计算为 `32768 / 256 = 128`，所以 unflatten 后的形状为 `[128, 256, 512]`。`.split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)` 这一步沿着第二个维度（索引为1）将张量分割成两部分：`w_kc` 的形状为 `[128, 128, 512]`，`w_vc` 的形状为 `[128, 128, 512]`；

`self_attn.w_kc` 的最终形状为 `[128, 128, 512]`，即 `[num_heads, qk_nope_head_dim, kv_lora_rank]`；
`self_attn.w_vc` 的最终形状为 `[128, 512, 128]`，即 `[num_heads, kv_lora_rank, v_head_dim]`


## 0x3.2 `forward` 控制逻辑

`DeepseekV2AttentionMLA`类的前向实现分为普通实现（没有矩阵吸收的版本），矩阵吸收的版本还有针对ROCM的吸收并且fuse mla和rope的版本，什么时候选用哪个版本的前向实现是在`forward`中进行控制的，这里来梳理一下它的控制逻辑。代码比较短，解析如下：

```python
def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """
        DeepseekV2 多层注意力(MLA)的前向传播函数。
        
        根据不同的执行模式(prefill/extend/decode)选择不同的计算路径：
        1. forward_normal: 不使用权重吸收的标准注意力计算
        2. forward_absorb: 使用权重吸收优化的注意力计算
        3. forward_absorb_fused_mla_rope: 针对ROCm平台的融合MLA+RoPE优化计算
        
        参数：
            positions：位置编码张量，用于RoPE计算
            hidden_states：输入隐藏状态
            forward_batch：前向计算批次信息，包含计算模式和缓存信息
            
        返回：
            torch.Tensor：注意力层的输出
        """

        def no_absorb() -> bool:
            """
            判断是否应该使用标准注意力计算而不是权重吸收优化。
            
            根据不同的执行环境和模式决定：
            - 对于启用了flashinfer MLA的情况：仅在禁用radix缓存且处于extend模式时不使用权重吸收
            - 对于使用Triton的情况：在prefill阶段使用标准计算，在extend/decode阶段使用权重吸收
              但有特殊情况例外（如目标验证、草稿扩展或有前缀长度）
              
            返回：
                bool：True表示使用标准计算，False表示使用权重吸收优化
            """
            if global_server_args_dict["enable_flashinfer_mla"]:
                # Flashinfer MLA模式：仅在禁用radix缓存且处于extend模式时不使用权重吸收
                return (
                    global_server_args_dict["disable_radix_cache"]
                    and forward_batch.forward_mode.is_extend()
                )
            else:
                # Triton模式：在prefill阶段使用标准计算，在extend/decode阶段使用权重吸收
                # 但以下特殊情况例外：
                # 1. 目标验证模式(target_verify)
                # 2. 草稿扩展模式(draft_extend)
                # 3. 有前缀长度的情况
                return (
                    forward_batch.forward_mode.is_extend()
                    and not forward_batch.forward_mode.is_target_verify()
                    and not forward_batch.forward_mode.is_draft_extend()
                    and forward_batch.extend_prefix_lens.sum() == 0
                )

        # 根据no_absorb()的结果选择不同的计算路径
        if no_absorb():
            # 使用标准注意力计算（不使用权重吸收优化）
            return self.forward_normal(positions, hidden_states, forward_batch)
        else:
            # 使用权重吸收优化的计算路径
            if is_hip_:
                # 针对AMD GPU（ROCm）平台的特殊优化
                if (
                    os.getenv("SGLANG_ROCM_FUSED_DECODE_MLA") == "1"
                    and forward_batch.forward_mode.is_decode()
                ):
                    # 使用融合的MLA+RoPE优化计算（仅在ROCm平台的decode模式下）
                    return self.forward_absorb_fused_mla_rope(
                        positions, hidden_states, forward_batch
                    )
                else:
                    # 使用标准的权重吸收优化
                    return self.forward_absorb(positions, hidden_states, forward_batch)
            else:
                # 非ROCm平台（如CUDA）使用标准的权重吸收优化
                return self.forward_absorb(positions, hidden_states, forward_batch)
```


## 0x3.3 `forward_normal` 的实现

`forward_normal`的实现和上面的`DeepseekV2Attention`类的实现是一样的，不过在这个实现里面现在Cache的是Latent，而不是解压缩之后MHA KV Cache的格式，所以是可以达到节省显存的目的的。

另外需要注意的是`forward_normal`的实现中在运行MHA之前没有再对 q,k,v的 `head_dim` 进行 padding 到 256 的操作了，这大概是历史遗留问题，在实现这个函数的时候FlashInfer支持了这个headim。对比这里的`self.attn_mha`定义：

```python
self.attn_mha = RadixAttention(
            self.num_local_heads,
            self.qk_nope_head_dim + self.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=self.num_local_heads,
            layer_id=layer_id,
            v_head_dim=self.v_head_dim,
        )
```

和之前的

```python
# 初始化RadixAttention，用于高效的注意力计算
# TODO, support head_size 192
self.attn = RadixAttention(
    self.num_local_heads,
    256, 
    self.scaling,
    num_kv_heads=self.num_local_heads,
    layer_id=layer_id,
)
```

可以发现是TODO被解决了。

## 0x3.4 `forward_absorb` 的实现

这部分代码不长，我们可以直接代入DeepSeek R1的超参数来读一下，假设TP=8，`self.num_local_heads=128/8=16`，`self.kv_lora_rank=512`，`self.qk_rope_head_dim=64`：

### 000

```python
def forward_absorb(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        q_len = hidden_states.shape[0] # 序列长度，token数
        # attention 的输入Q，shape： ([q_len, 16, 576]), 
        # 其中576 包含 kv_lora_rank(512) + qk_rope_head_dim(64)。
        # 这里建立了一个未初始化的Tensor，后续往里面填。
        q_input = hidden_states.new_empty(
            q_len, self.num_local_heads, self.kv_lora_rank + self.qk_rope_head_dim
        )
```

下面的`q_lora_rank`对应query压缩后的隐向量的维度`d'_c`，在DeepSeek R1中`q_lora_rank=1536`。


又注意到，hidden_states的shape是`[bs, q_len, hidden_size]`，且`self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim = 128 + 64 = 192`：

```python
# self.q_a_proj = ReplicatedLinear(
#    self.hidden_size,
#    self.q_lora_rank,
#    bias=False,
#    quant_config=quant_config,
# )
# self.q_b_proj = ColumnParallelLinear(
#    q_lora_rank,
#    self.num_heads * self.qk_head_dim,
#    bias=False,
#    quant_config=quant_config,
# )

if self.q_lora_rank is not None:
    q = self.q_a_proj(hidden_states)[0]
    q = self.q_a_layernorm(q)
    q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
else:
    q = self.q_proj(hidden_states)[0].view(
        -1, self.num_local_heads, self.qk_head_dim
    )
```

- 对于 `q = self.q_a_proj(hidden_states)[0]` ，输入形状：`[bs, q_len, hidden_size]`；`self.q_a_proj` 是一个ReplicatedLinear层，将hidden_size维度映射到q_lora_rank维度；输出形状：`[bs, q_len, q_lora_rank] = [bs, q_len, 1536]`
- 对于 `q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)`，输入形状：`[bs, q_len, 1536]`；`self.q_b_proj`是一个ColumnParallelLinear层，将q_lora_rank维度映射到 `num_heads * qk_head_dim` 维度
中间输出形状：`[bs, q_len, num_heads * qk_head_dim] = [bs, q_len, 128 * 192]`；但由于TP=8，每个GPU只负责128/8=16个头，所以实际输出形状是：`[bs, q_len, 16 * 192]`；然后通过view操作重塑为：`[-1, self.num_local_heads, self.qk_head_dim] = [bs * q_len, 16, 192]`。

后续分析将假设bs=1。为了方便下面的代码分析，这里再复制一下paper的MLA公式。

![](https://files.mdnice.com/user/59/6afdc732-4224-4a4d-aaf5-a87732020e68.png)

### 001

```python
q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
```

q被分成 q_nope 和q_pe, 其shape分别是`[q_len, 16, 128]`， `[q_len, 16, 64]`。

q_nope就是论文中公式38所得到的$q_t^C$，而q_pe后续用来会做ROPE，是论文中公式39中RoPE括号中的部分。

### 002

```python
if self.w_kc.dtype == torch.float8_e4m3fnuz:
    # TODO(kernel): add bmm_fp8 for torch.float8_e4m3fnuz
    q_nope_out = torch.bmm(
        q_nope.to(torch.bfloat16).transpose(0, 1),
        self.w_kc.to(torch.bfloat16) * self.w_scale,
    )
elif self.w_kc.dtype == torch.float8_e4m3fn:
    q_nope_val, q_nope_scale = input_to_float8(
        q_nope.transpose(0, 1), torch.float8_e4m3fn
    )
    q_nope_out = bmm_fp8(
        q_nope_val, self.w_kc, q_nope_scale, self.w_scale, torch.bfloat16
    )
else:
    q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)
q_input[..., : self.kv_lora_rank] = q_nope_out.transpose(0, 1)
```

忽略掉fp8的分支，从之前的分析知道`self.w_kc，self.w_vc`它们分别对应了将`self.kv_b_proj`拆分后的$W^{UK}$和$W^{UV}$。`q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)`这行代码就是paper的公式42。在这行代码中，q_nope 和`self.w_kc`相乘，得到`q_nope_out`，shape 从`[q_len, 16, 128]` 变成`[q_len, 16, 512]`。

然后`q_input[..., : self.kv_lora_rank] = q_nope_out.transpose(0, 1)`将`q_nope_out`填充到`q_input`的前512个channel中。


### 003 W^{UK}的吸收

```python
latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
v_input = latent_cache[..., : self.kv_lora_rank]
v_input = self.kv_a_layernorm(v_input.contiguous()).unsqueeze(1)
k_input = latent_cache.unsqueeze(1)
k_input[..., : self.kv_lora_rank] = v_input
k_pe = k_input[..., self.kv_lora_rank :]

q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
q_input[..., self.kv_lora_rank :] = q_pe
k_input[..., self.kv_lora_rank :] = k_pe

attn_output = self.attn_mqa(q_input, k_input, v_input, forward_batch)
attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)
```

`self.kv_a_proj_with_mqa`包含公式中的$W^{DKV}$和$W^{KR}$两个权重，`latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]`这行代码对hidden_states进行投影得到Latent，它的shape为`[q_len, 576]`，其中前512个dim对应了$c_t^{KV}$也就是对应了公式41，后64个dim对应了公式43的RoPE的括号里面的部分也就是$W^{KR}h_t$（还没有应用RoPE）。


接着，`v_input = latent_cache[..., : self.kv_lora_rank]`取出了$c_t^{KV}$，`k_input[..., : self.kv_lora_rank] = v_input`这行代码表示k和v共享相同的latent。接着就是`k_pe = k_input[..., self.kv_lora_rank :]`拿到k_pe准备做RoPE，最后就是做RoPE和Attention了。

关注一下这里的shape变化，其中`q_input`的shape为`[q_len, 16, 576]`；`k_input`的shape为`[q_len, 1, 576]`，`v_input`的shape为`[q_len, 1, 512]`，`attn_output`的shape就是`[q_len, 16, 512]`。

又注意到

```python
self.attn_mqa = RadixAttention(
      self.num_local_heads,
      self.kv_lora_rank + self.qk_rope_head_dim,
      self.scaling,
      num_kv_heads=1,
      layer_id=layer_id,
      v_head_dim=self.kv_lora_rank,
  )
```

所以这里的attn计算可以看成一个Multi Query Attention，其中Query的 head 是 16个， QK head_dim是576，V head_dim是512。QK的head_dim中包含不做RoPE的512和做RoPE的64两个维度。

其实这个MQA就是DeekSeek在开源周开源的FlashMLA，如下图所示：

![](https://files.mdnice.com/user/59/085133cc-3802-43c6-a3b3-b58e63c59af4.png)


我们还需要注意的是，这个地方的$W^{UK}$矩阵吸收并没有在init的时候提前完成，而是直接在forward的时候通过矩阵运算结合律来算。可以用paper公式31和32来说明：

![](https://files.mdnice.com/user/59/c2b91998-55d4-475b-9479-0706ec545fb4.png)

如果我们保持`forward_normal`那种计算方式，也就是说先对Latent解压缩再计算，则Attn的计算是一个实打实的Multi Head Attention，会增大计算量。

### 004 W^{UV}的吸收

```python
if self.w_vc.dtype == torch.float8_e4m3fnuz:
  # TODO(kernel): add bmm_fp8 for torch.float8_e4m3fnuz
  attn_bmm_output = torch.bmm(
      attn_output.to(torch.bfloat16).transpose(0, 1),
      self.w_vc.to(torch.bfloat16) * self.w_scale,
  )
elif self.w_vc.dtype == torch.float8_e4m3fn:
  attn_output_val, attn_output_scale = input_to_float8(
      attn_output.transpose(0, 1), torch.float8_e4m3fn
  )
  attn_bmm_output = bmm_fp8(
      attn_output_val,
      self.w_vc,
      attn_output_scale,
      self.w_scale,
      torch.bfloat16,
  )
else:
  attn_bmm_output = torch.bmm(attn_output.transpose(0, 1), self.w_vc)
attn_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
output, _ = self.o_proj(attn_output)

return output
```


忽略掉fp8的分支，从之前的分析知道`self.w_kc，self.w_vc`它们分别对应了将`self.kv_b_proj`拆分后的$W^{UK}$和$W^{UV}$。同样，$W^{UV}$也可以吸收到$W_O$里面，这通过`attn_bmm_output = torch.bmm(attn_output.transpose(0, 1), self.w_vc)`和`output, _ = self.o_proj(attn_output)`这两行代码来完成，也没有在init中去做。根据之前在 [大模型KV Cache节省神器MLA学习笔记（包含推理时的矩阵吸收分析）](https://mp.weixin.qq.com/s/cBMrRUdM1IM0T1ji_ODxng) 这里提到的结论，不在init的时候做矩阵吸收的预处理反而速度是更快的，SGLang MLA也沿用了这一结论。

# 0x4. 结论

本文详细分析了一下 SGLang MLA 的代码实现，并且指出了矩阵吸收以及FlashMLA应该应用的位置，主要是自己理清相关逻辑记录的笔记。

# 0x5. 参考资料

- https://zhuanlan.zhihu.com/p/714686419
- https://zhuanlan.zhihu.com/p/19849661536
- https://arxiv.org/abs/2405.04434









