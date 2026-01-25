本文记录 LightX2V 中 Ulysses Attention 的实现方式与张量形状推导，覆盖以下文件：

- `LightX2V/lightx2v/common/ops/attn/ulysses_attn.py`
- `LightX2V/lightx2v/common/ops/attn/utils/all2all.py`（`all2all_head2seq` / `all2all_seq2head`）
- `LightX2V/lightx2v/common/ops/attn/flash_attn.py`（FlashAttention wrapper 的输出形状约定）

性能相关内容只讨论实现机制层面的影响因素，我没具体测过所以不做什么确定性的描述。写这个的原因是对Ulysses Attention的实现有点忘了，周末自己再推一下。



# 0x0. 背景

在多模态/视频场景中，图像（或视频）token 序列通常远长于文本 token 序列。单卡上 FlashAttention 主要优化显存占用与访存行为；在多卡并行时，额外需要处理：

- 长序列下的并行切分方式（按序列切分、按 head 切分，或二者组合）
- 由切分方式带来的通信（collective）与数据重排开销

Ulysses 在 LightX2V 的实现，可概括为一种 layout 变换闭环：

- 输入侧：图像 token 采用 **sequence shard + full heads** 的布局
- Attention 侧：通过一次 all-to-all 变换为 **full sequence + head shard**，以便每个 rank 计算其负责的 head 子集
- 输出侧：再通过一次 all-to-all 将图像输出变回 **sequence shard + full heads**

文本部分在该实现中采用按 head 的聚合（`all_gather`）方式拼回 full heads。


# 0x1. 符号与形状约定

记：

- \(P\)：`world_size = dist.get_world_size(seq_p_group)`（并行组大小）
- \(H\)：attention head 数（代码变量 `heads`）
- \(D\)：每个 head 的维度（代码变量 `hidden_dims`）
- \(\text{shardH} = H/P\)：每个 rank 负责的 head 数（要求 \(H\) 能被 \(P\) 整除）

序列长度区分图像与文本（对应 `ulysses_attn.py` 中的分段逻辑）：

- \(\text{imgShardS}\)：当前 rank 上图像 token 的长度（`img_qkv_len`，分片后）
- \(\text{imgS} = \text{imgShardS}\cdot P\)：全局图像 token 长度（`global_img_seqlen`）
- \(\text{txtS}\)：文本 token 长度（`txt_qkv_len`）
- \(\text{S} = \text{imgS} + \text{txtS}\)：拼接后的总长度（此处先忽略 `txt_mask_len` 分支）

FlashAttention wrapper 的输出形状约定（见 `LightX2V/lightx2v/common/ops/attn/flash_attn.py`）：

- 输入 `q/k/v`：`[T, shardH, D]`（或从 4D reshape 到 3D）
- 输出：`[T, shardH * D]`（flatten输出）

因此在 `ulysses_attn.py` 中，凡是将 attention 输出恢复为 3D 形状的步骤，都依赖 `D` 与 `shardH` 的一致性。


# 0x2. `UlyssesAttnWeight`（默认实现）解析

目标文件：`LightX2V/lightx2v/common/ops/attn/ulysses_attn.py`，类：`UlyssesAttnWeight`。

以下以 `enable_head_parallel=False` 的主路径为主进行说明。

## 0x2.1 输入形状与图像/文本拆分

函数输入注释写为：

- `q/k/v`: `[shard_seqlen, heads, hidden_dims]`

实现中还支持 4D 输入，进入后会 reshape 到 3D：

- 4D：`[B, S, H, D]`
- 3D：`[T, H, D]`，其中 \(T=B\cdot S\)

当 `img_first=True` 时，按图像与文本长度拆分：

- `img_q/img_k/img_v`: `[imgShardS, H, D]`
- `txt_q/txt_k/txt_v`: `[txtS, H, D]`

## 0x2.2 图像 QKV 的 layout 变换（seq shard -> head shard）

将图像 QKV 堆叠：

- `img_qkv = stack([img_q, img_k, img_v], dim=0)`：`[3, imgShardS, H, D]`

将 head 维 \(H\) 拆为 `[P, shardH]`：

- `reshape(3, imgShardS, P, shardH, D)`：`[3, imgShardS, P, shardH, D]`

为 all-to-all 准备分箱维度（将 `P` 放到 dim0）：

- `permute(2, 1, 0, 3, 4)`：`[P, imgShardS, 3, shardH, D]`

执行 all-to-all：

- `dist.all_to_all_single(output_qkv, input_t, group=seq_p_group)`
- `output_qkv` 形状仍为：`[P, imgShardS, 3, shardH, D]`

随后将其整理回标准 Q/K/V：

- `output_qkv.reshape(imgS, 3, shardH, D).transpose(0, 1)`：`[3, imgS, shardH, D]`
- 得到 `shard_img_q/k/v`：`[imgS, shardH, D]`

该步骤的结果是：每个 rank 拥有全量图像序列（长度 `imgS`）与其负责的 head 子集（`shardH`）。

## 0x2.3 文本分支：按 head slice

文本 QKV 不做 seq 维 all-to-all，仅按 head 维选择本 rank 的 head 子集：

- `shard_txt_q/k/v`：`[txtS, shardH, D]`

## 0x2.4 拼接并计算 attention（注意输出flatten）

拼接图像与文本：

- `q/k/v = cat([shard_img_*, shard_txt_*], dim=0)`：`[imgS + txtS, shardH, D]`

调用 `attention_module.apply(...)`（例如 `FlashAttn2Weight/FlashAttn3Weight`）得到：

- `attn`：`[imgS + txtS, shardH * D]`

实现中会重建 `cu_seqlens_qkv` 与 `max_seqlen_qkv` 以适配 varlen 接口；该部分不改变张量形状推导结论。

## 0x2.5 输出拆分与回传 layout（head shard -> seq shard）

按序列拆分输出（`img_first=True`）：

- `img_attn`：`[imgS, shardH*D]`
- `txt_attn`：`[txtS, shardH*D]`

### 0x2.5.1 文本：`all_gather` 拼回 full heads

对 `txt_attn` 做 `all_gather` 并在 dim=1 拼接：

- 每 rank：`[txtS, shardH*D]`
- 拼接后：`[txtS, P*shardH*D] = [txtS, H*D]`

### 0x2.5.2 图像：`_reshape_img_attn()` + `all2all_head2seq()`

先将flatten输出恢复为 3D：

- `img_attn.reshape(imgS, shardH, D)`：`[imgS, shardH, D]`

调用 `all2all_head2seq`：

- 输入：`[seq_len, heads/P, D]`（此处为 `[imgS, shardH, D]`）
- 输出：`[seq_len/P, heads, D]`（此处为 `[imgShardS, H, D]`）

最后flatten回：

- `[imgShardS, H*D]`

至此，图像输出回到 **sequence shard + full heads** 的布局。


# 0x3. `all2all_head2seq` 维度推导（逐步对应实现）

目标文件：`LightX2V/lightx2v/common/ops/attn/utils/all2all.py`

函数注释描述：

- 输入：`[seq_len, heads/P, D]`
- 输出：`[seq_len/P, heads, D]`

令输入张量 \(X\)：

- \(X \in \mathbb{R}^{S \times (H/P) \times D}\)
- \(\text{shardS} = S/P\)

## 0x3.1 reshape：按 seq 切分为 P 份

代码等价于：

- `reshape(P, shardS, shardH, D)`
- 得到 \(X_1 \in \mathbb{R}^{P \times \text{shardS} \times \text{shardH} \times D}\)

## 0x3.2 transpose：重排维度以便合并 head

代码：

- `transpose(1, 2)`
- 得到 \(X_2 \in \mathbb{R}^{P \times \text{shardH} \times \text{shardS} \times D}\)

## 0x3.3 all-to-all：交换 dim0 上的 P 份数据

代码：

- `dist.all_to_all_single(output, input_t, group=group)`

形状保持为：

- `[P, shardH, shardS, D]`

## 0x3.4 合并 head 并转回目标布局

合并前两维：

- `[P, shardH, shardS, D] -> [H, shardS, D]`

再转置得到：

- `[shardS, H, D]`

因此 `all2all_head2seq` 将 **full sequence + head shard** 映射为 **sequence shard + full heads**。

对应地，`all2all_seq2head` 完成相反方向的映射，可视为其逆过程。



# 0x4. `enable_head_parallel` 分支的结构性差异

当 `enable_head_parallel=True` 时，默认实现将 `shardH` 个 head 拆分为循环处理：

- 对每个 local head：单独执行一次 all-to-all，并对单 head 的 QKV 计算 attention
- 最终将各 head 的输出在 head 维拼接

该分支的主要差异是 collective 调用次数与 attention kernel 调用次数显著增加；其效果取决于通信与计算是否存在可观的重叠空间，以及具体运行环境的调度与延迟特性。



# 0x5. `use_fp8_comm` 通信量化分支

实现提供 `use_fp8_comm=True` 选项，在图像 QKV 的 all-to-all 与图像输出回传中引入：

- FP8 量化（`quant_fp8_vllm`）与 scale
- FP8 与 scale 的通信
- 反量化（`dequant_fp8_vllm`）

可能的影响因素：

- 通信字节数：FP8 数据通常小于 bf16/fp16，但额外包含 scale 的传输
- 计算开销：引入量化与反量化操作
- 可用性：依赖 `vllm._custom_ops` 的实现与运行环境
- 数值误差：由 FP8 量化引入，是否可接受取决于模型与任务



# 0x6. `Ulysses4090AttnWeight`：点对点轮转通信实现

`Ulysses4090AttnWeight` 使用“循环赛配对”的点对点 `isend/irecv` 轮转实现来模拟 all-to-all：

- `generate_round_robin_pairs()`：生成 \(P-1\) 轮配对（要求 `world_size` 为偶数）
- `load_balanced_all_to_all()`：每轮每个 rank 与一个 partner 交换一次，轮转结束后得到全对全交换结果

在张量布局效果上，与默认实现保持一致：

- 图像 QKV：按 head 切分为 \(P\) 份，轮转交换后在 seq 维拼接得到 `[imgS, shardH, D]`
- 图像输出：按 seq 切分为 \(P\) 份，轮转交换后在 head 维拼接得到 `[imgShardS, H, D]`

在 `use_fp8_comm=True` 时，该实现将 FP8 数据与 scale 打包为字节缓冲区进行单次收发，减少消息数量。



# 0x7. 总结

LightX2V 的 Ulysses Attention 实现通过两次图像 all-to-all（输入侧与输出侧）完成 layout 变换闭环，使 attention 计算在 **full sequence + head shard** 的布局上进行；文本部分通过 `all_gather` 拼回 full heads。实现同时提供了：

- `enable_head_parallel`：更细粒度的 head 级处理
- `use_fp8_comm`：FP8 通信量化
- `Ulysses4090AttnWeight`：点对点轮转模拟 all-to-all

这些选项的效果受运行环境、通信后端与数据规模影响，需要结合实际配置进行评估。我自己看着好像是给视频模型用的，这里只是记录一下Ulysses Attention的形状推导和流程，不做更多猜测。




