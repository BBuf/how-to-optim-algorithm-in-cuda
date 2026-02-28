本文记录 LightX2V 中 Ulysses Attention 的实现方式与张量形状推导，覆盖以下文件：

- `LightX2V/lightx2v/common/ops/attn/ulysses_attn.py`
- `LightX2V/lightx2v/common/ops/attn/utils/all2all.py`（`all2all_head2seq` / `all2all_seq2head`）

性能相关内容只讨论实现机制层面的影响因素，我没具体测过所以不做什么确定性的描述。写这个的原因是对Ulysses Attention的实现有点忘了，周末自己再推一下。代码仓库为：https://github.com/ModelTC/LightX2V

> 笔者注：下面的代码片段笔者没有加注释，里面的注释是从LightX2V拷贝代码片段时就已经自带的。基于这个commit号设置了下行号：bef76dc298983053df2506d8eaaa97d1895ec077


# 0x0. 背景

在多模态/视频场景中，图像（或视频）token 序列通常远长于文本 token 序列。单卡上 FlashAttention 主要优化显存占用与访存行为；在多卡并行时，额外需要处理：

- 长序列下的并行切分方式（按序列切分、按 head 切分，或二者组合）
- 由切分方式带来的通信（collective）与数据重排开销

Ulysses 在 LightX2V 的实现，可概括为一种 layout 变换：

- 输入侧：图像 token 采用 **sequence shard + full heads** 的布局
- Attention 侧：通过一次 all-to-all 变换为 **full sequence + head shard**，以便每个 rank 计算其负责的 head 子集
- 输出侧：再通过一次 all-to-all 将图像输出变回 **sequence shard + full heads**

文本部分在该实现中采用按 head 的聚合（`all_gather`）方式拼回 full heads。


# 0x1. 符号与形状约定

记（使用代码符号表示）：

- `P`: `world_size = dist.get_world_size(seq_p_group)`（并行组大小）
- `H`: attention head 数（代码变量 `heads`）
- `D`: 每个 head 的维度（代码变量 `hidden_dims`）
- `shardH = H // P`: 每个 rank 负责的 head 数（要求 `H % P == 0`）

序列长度区分图像与文本（对应 `ulysses_attn.py` 中的分段逻辑）：

- `imgShardS`: 当前 rank 上图像 token 的长度（`img_qkv_len`，分片后）
- `imgS = imgShardS * P`: 全局图像 token 长度（`global_img_seqlen`）
- `txtS`: 文本 token 长度（`txt_qkv_len`）
- `S = imgS + txtS`: 拼接后的总长度（此处先忽略 `txt_mask_len` 分支）

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
- 3D：`[T, H, D]`，其中 `T = B * S`

对应代码（4D -> 3D reshape）：

```47:50:LightX2V/lightx2v/common/ops/attn/ulysses_attn.py
        if len(q.shape) == 4:
            q = q.reshape(-1, q.shape[-2], q.shape[-1])
            k = k.reshape(-1, k.shape[-2], k.shape[-1])
            v = v.reshape(-1, v.shape[-2], v.shape[-1])
```

当 `img_first=True` 时，按图像与文本长度拆分：

- `img_q/img_k/img_v`: `[imgShardS, H, D]`
- `txt_q/txt_k/txt_v`: `[txtS, H, D]`

对应代码（img/text 分段与 `.contiguous()`）：

```56:96:LightX2V/lightx2v/common/ops/attn/ulysses_attn.py
        # 获取序列长度和文本相关的长度
        if img_first:
            img_qkv_len = slice_qkv_len
            if len(cu_seqlens_qkv) == 3:
                txt_qkv_len = cu_seqlens_qkv[1] - slice_qkv_len  # 文本查询、键和值的长度
                txt_mask_len = cu_seqlens_qkv[2] - slice_qkv_len  # 文本掩码长度
            elif len(cu_seqlens_qkv) == 2:
                txt_qkv_len = cu_seqlens_qkv[1] - slice_qkv_len  # 文本查询、键和值的长度
                txt_mask_len = None
        else:
            # assert len(cu_seqlens_qkv) == 2
            txt_qkv_len = slice_qkv_len
            img_qkv_len = cu_seqlens_qkv[1] - slice_qkv_len
            txt_mask_len = None
 
        # 分割图像和文本的查询、键和值
        if img_first:
            img_q, img_k, img_v = q[:img_qkv_len, :, :].contiguous(), k[:img_qkv_len, :, :].contiguous(), v[:img_qkv_len, :, :].contiguous()
            txt_q, txt_k, txt_v = q[img_qkv_len:, :, :].contiguous(), k[img_qkv_len:, :, :].contiguous(), v[img_qkv_len:, :, :].contiguous()
        else:
            txt_q, txt_k, txt_v = q[:txt_qkv_len, :, :].contiguous(), k[:txt_qkv_len, :, :].contiguous(), v[:txt_qkv_len, :, :].contiguous()
            img_q, img_k, img_v = q[txt_qkv_len:, :, :].contiguous(), k[txt_qkv_len:, :, :].contiguous(), v[txt_qkv_len:, :, :].contiguous()
```

## 0x2.2 图像 QKV 的 layout 变换（seq shard -> head shard）

将图像 QKV 堆叠：

- `img_qkv = stack([img_q, img_k, img_v], dim=0)`：`[3, imgShardS, H, D]`

将 head 维 `H` 拆为 `[P, shardH]`：

- `reshape(3, imgShardS, P, shardH, D)`：`[3, imgShardS, P, shardH, D]`

为 all-to-all 准备分箱维度（将 `P` 放到 dim0）：

- `permute(2, 1, 0, 3, 4)`：`[P, imgShardS, 3, shardH, D]`

执行 all-to-all：

- `dist.all_to_all_single(output_qkv, input_t, group=seq_p_group)`
- `output_qkv` 形状仍为：`[P, imgShardS, 3, shardH, D]`

随后将其整理回标准 Q/K/V：

- `output_qkv.reshape(imgS, 3, shardH, D).transpose(0, 1)`：`[3, imgS, shardH, D]`
- 得到 `shard_img_q/k/v`：`[imgS, shardH, D]`

对应代码（stack + reshape；以及两条分支的 permute/all_to_all/reshape）：

```97:177:LightX2V/lightx2v/common/ops/attn/ulysses_attn.py
        img_qkv = torch.stack([img_q, img_k, img_v], dim=0).reshape(3, img_qkv_len, world_size, shard_heads, hidden_dims)
        original_dtype = img_qkv.dtype
 
        if enable_head_parallel:
            img_qkv = img_qkv.permute(3, 2, 1, 0, 4).contiguous()  # (shard_heads, world_size, img_qkv_len, 3, hidden_dims)
            output_qkv = torch.empty_like(img_qkv)
            # ... per-head all_to_all_single + reshape ...
            qkv = output_qkv[h].reshape(global_img_seqlen, 3, single_head, hidden_dims).transpose(0, 1)
            shard_img_q = qkv[0]  # (global_img_seqlen, single_head, hidden_dims)
            shard_img_k = qkv[1]
            shard_img_v = qkv[2]
        else:
            img_qkv = img_qkv.permute(2, 1, 0, 3, 4).contiguous()  # (world_size, img_qkv_len, 3, shard_heads, hidden_dims)
            # ... all_to_all_single ...
            qkv = output_qkv.reshape(global_img_seqlen, 3, shard_heads, hidden_dims).transpose(0, 1)
            shard_img_q = qkv[0]  # (global_img_seqlen, shard_head, hidden_dims)
            shard_img_k = qkv[1]
            shard_img_v = qkv[2]
```

对应代码（enable_head_parallel=True：逐 head all_to_all + reshape 到 `[global_img_seqlen, 3, 1, hidden_dims]`）：

```100:156:LightX2V/lightx2v/common/ops/attn/ulysses_attn.py
        if enable_head_parallel:
            img_qkv = img_qkv.permute(3, 2, 1, 0, 4).contiguous()  # (shard_heads, world_size, img_qkv_len, 3, hidden_dims)
            output_qkv = torch.empty_like(img_qkv)
 
            # 通信图像的查询、键和值
            if use_fp8_comm:
                img_qkv_fp8, img_qkv_scale = quant_fp8_vllm(img_qkv.reshape(-1, hidden_dims))
                img_qkv_fp8 = img_qkv_fp8.reshape(shard_heads, world_size, img_qkv_len, 3, hidden_dims)
                img_qkv_scale = img_qkv_scale.reshape(shard_heads, world_size, img_qkv_len, 3, 1)
                output_qkv_fp8 = torch.empty_like(img_qkv_fp8)
                output_qkv_scale = torch.empty_like(img_qkv_scale)
                comm_fp8_works = []
                comm_scale_works = []
                for h in range(shard_heads):
                    work_fp8 = dist.all_to_all_single(output_qkv_fp8[h], img_qkv_fp8[h], group=seq_p_group, async_op=True)
                    work_scale = dist.all_to_all_single(output_qkv_scale[h], img_qkv_scale[h], group=seq_p_group, async_op=True)
                    comm_fp8_works.append(work_fp8)
                    comm_scale_works.append(work_scale)
            else:
                comm_works = []
                for h in range(shard_heads):
                    work = dist.all_to_all_single(output_qkv[h], img_qkv[h], group=seq_p_group, async_op=True)
                    comm_works.append(work)
 
            # 逐个head完成Attention计算
            single_head = 1
            head_attns = []
            for h in range(shard_heads):
                if use_fp8_comm:
                    comm_fp8_works[h].wait()
                    comm_scale_works[h].wait()
                    output_qkv[h] = dequant_fp8_vllm(output_qkv_fp8[h], output_qkv_scale[h], original_dtype)
                else:
                    comm_works[h].wait()
 
                qkv = output_qkv[h].reshape(global_img_seqlen, 3, single_head, hidden_dims).transpose(0, 1)
                shard_img_q = qkv[0]  # (global_img_seqlen, single_head, hidden_dims)
                shard_img_k = qkv[1]
                shard_img_v = qkv[2]
 
                # 处理文本的查询、键和值，选择当前进程的当前头
                shard_txt_q = txt_q[:, (cur_rank * shard_heads + h) : (cur_rank * shard_heads + h + 1), :]
                shard_txt_k = txt_k[:, (cur_rank * shard_heads + h) : (cur_rank * shard_heads + h + 1), :]
                shard_txt_v = txt_v[:, (cur_rank * shard_heads + h) : (cur_rank * shard_heads + h + 1), :]
 
                # 合并图像和文本的查询、键和值
                q = torch.cat((shard_img_q, shard_txt_q), dim=0)
                k = torch.cat((shard_img_k, shard_txt_k), dim=0)
                v = torch.cat((shard_img_v, shard_txt_v), dim=0)
 
                # 调用注意力函数计算注意力结果
                head_attn = attention_module.apply(q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_qkv, cu_seqlens_kv=cu_seqlens_qkv, max_seqlen_q=max_seqlen_qkv, max_seqlen_kv=max_seqlen_qkv, **kwargs)
                head_attns.append(head_attn)
 
            # 合并当前进程的所有head的attn
            attn = torch.cat(head_attns, dim=1)
```

对应代码（enable_head_parallel=False：一次 all_to_all + reshape 到 `[global_img_seqlen, 3, shard_heads, hidden_dims]`）：

```157:192:LightX2V/lightx2v/common/ops/attn/ulysses_attn.py
        else:
            img_qkv = img_qkv.permute(2, 1, 0, 3, 4).contiguous()  # (world_size, img_qkv_len, 3, shard_heads, hidden_dims)
 
            # 通信图像的查询、键和值
            if use_fp8_comm:
                img_qkv_fp8, img_qkv_scale = quant_fp8_vllm(img_qkv.reshape(-1, hidden_dims))
                img_qkv_fp8 = img_qkv_fp8.reshape(world_size, img_qkv_len, shard_heads, 3, hidden_dims)
                img_qkv_scale = img_qkv_scale.reshape(world_size, img_qkv_len, shard_heads, 3, 1)
                output_qkv_fp8 = torch.empty_like(img_qkv_fp8)
                output_qkv_scale = torch.empty_like(img_qkv_scale)
                dist.all_to_all_single(output_qkv_fp8, img_qkv_fp8, group=seq_p_group)
                dist.all_to_all_single(output_qkv_scale, img_qkv_scale, group=seq_p_group)
                output_qkv = dequant_fp8_vllm(output_qkv_fp8, output_qkv_scale, original_dtype)
            else:
                output_qkv = torch.empty_like(img_qkv)
                dist.all_to_all_single(output_qkv, img_qkv, group=seq_p_group)
 
            # 完成Attention计算
            qkv = output_qkv.reshape(global_img_seqlen, 3, shard_heads, hidden_dims).transpose(0, 1)
            shard_img_q = qkv[0]  # (global_img_seqlen, shard_head, hidden_dims)
            shard_img_k = qkv[1]
            shard_img_v = qkv[2]
 
            # 处理文本的查询、键和值，选择当前进程的当前头
            shard_txt_q = txt_q[:, cur_rank * shard_heads : (cur_rank + 1) * shard_heads, :]
            shard_txt_k = txt_k[:, cur_rank * shard_heads : (cur_rank + 1) * shard_heads, :]
            shard_txt_v = txt_v[:, cur_rank * shard_heads : (cur_rank + 1) * shard_heads, :]
 
            # 合并图像和文本的查询、键和值
            q = torch.cat((shard_img_q, shard_txt_q), dim=0)
            k = torch.cat((shard_img_k, shard_txt_k), dim=0)
            v = torch.cat((shard_img_v, shard_txt_v), dim=0)
 
            # 调用注意力函数计算注意力结果
            attn = attention_module.apply(q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_qkv, cu_seqlens_kv=cu_seqlens_qkv, max_seqlen_q=max_seqlen_qkv, max_seqlen_kv=max_seqlen_qkv, **kwargs)
```

该步骤的结果是：每个 rank 拥有全量图像序列（长度 `imgS`）与其负责的 head 子集（`shardH`）。

## 0x2.3 文本分支：按 head slice

文本 QKV 不做 seq 维 all-to-all，仅按 head 维选择本 rank 的 head 子集：

- `shard_txt_q/k/v`：`[txtS, shardH, D]`

对应代码（两条分支对文本做 head slice）：

```140:183:LightX2V/lightx2v/common/ops/attn/ulysses_attn.py
                # 处理文本的查询、键和值，选择当前进程的当前头
                shard_txt_q = txt_q[:, (cur_rank * shard_heads + h) : (cur_rank * shard_heads + h + 1), :]
                shard_txt_k = txt_k[:, (cur_rank * shard_heads + h) : (cur_rank * shard_heads + h + 1), :]
                shard_txt_v = txt_v[:, (cur_rank * shard_heads + h) : (cur_rank * shard_heads + h + 1), :]
...
            # 处理文本的查询、键和值，选择当前进程的当前头
            shard_txt_q = txt_q[:, cur_rank * shard_heads : (cur_rank + 1) * shard_heads, :]
            shard_txt_k = txt_k[:, cur_rank * shard_heads : (cur_rank + 1) * shard_heads, :]
            shard_txt_v = txt_v[:, cur_rank * shard_heads : (cur_rank + 1) * shard_heads, :]
```

## 0x2.4 拼接并计算 attention（注意输出flatten）

拼接图像与文本：

- `q/k/v = cat([shard_img_*, shard_txt_*], dim=0)`：`[imgS + txtS, shardH, D]`

调用 `attention_module.apply(...)`（例如 `FlashAttn2Weight/FlashAttn3Weight`）得到：

- `attn`：`[imgS + txtS, shardH * D]`

对应代码（拼接 q/k/v 并调用 attention_module.apply）：

```145:191:LightX2V/lightx2v/common/ops/attn/ulysses_attn.py
                # 合并图像和文本的查询、键和值
                q = torch.cat((shard_img_q, shard_txt_q), dim=0)
                k = torch.cat((shard_img_k, shard_txt_k), dim=0)
                v = torch.cat((shard_img_v, shard_txt_v), dim=0)
 
                # 调用注意力函数计算注意力结果
                head_attn = attention_module.apply(q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_qkv, cu_seqlens_kv=cu_seqlens_qkv, max_seqlen_q=max_seqlen_qkv, max_seqlen_kv=max_seqlen_qkv, **kwargs)
...
            # 合并图像和文本的查询、键和值
            q = torch.cat((shard_img_q, shard_txt_q), dim=0)
            k = torch.cat((shard_img_k, shard_txt_k), dim=0)
            v = torch.cat((shard_img_v, shard_txt_v), dim=0)
 
            # 调用注意力函数计算注意力结果
            attn = attention_module.apply(q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_qkv, cu_seqlens_kv=cu_seqlens_qkv, max_seqlen_q=max_seqlen_qkv, max_seqlen_kv=max_seqlen_qkv, **kwargs)
```

实现中会重建 `cu_seqlens_qkv` 与 `max_seqlen_qkv` 以适配 varlen 接口；该部分不改变张量形状推导结论。

## 0x2.5 输出拆分与回传 layout（head shard -> seq shard）

按序列拆分输出（`img_first=True`）：

- `img_attn`：`[imgS, shardH*D]`
- `txt_attn`：`[txtS, shardH*D]`

对应代码（从 attn 中切出 img/text 输出）：

```193:197:LightX2V/lightx2v/common/ops/attn/ulysses_attn.py
        # 分割图像和文本的注意力结果
        if img_first:
            img_attn, txt_attn = attn[:global_img_seqlen, :], attn[global_img_seqlen:]
        else:
            txt_attn, img_attn = attn[:txt_qkv_len, :], attn[txt_qkv_len:]
```

### 0x2.5.1 文本：`all_gather` 拼回 full heads

对 `txt_attn` 做 `all_gather` 并在 dim=1 拼接：

- 每 rank：`[txtS, shardH*D]`
- 拼接后：`[txtS, P*shardH*D] = [txtS, H*D]`

对应代码（all_gather + cat）：

```202:206:LightX2V/lightx2v/common/ops/attn/ulysses_attn.py
        # 收集所有进程的文本注意力结果
        gathered_txt_attn = [torch.empty_like(txt_attn) for _ in range(world_size)]
        dist.all_gather(gathered_txt_attn, txt_attn, group=seq_p_group)
        txt_attn = torch.cat(gathered_txt_attn, dim=1)  # 合并所有进程的文本注意力结果
```

### 0x2.5.2 图像：`_reshape_img_attn()` + `all2all_head2seq()`

先将flatten输出恢复为 3D：

- `img_attn.reshape(imgS, shardH, D)`：`[imgS, shardH, D]`

调用 `all2all_head2seq`：

- 输入：`[seq_len, heads/P, D]`（此处为 `[imgS, shardH, D]`）
- 输出：`[seq_len/P, heads, D]`（此处为 `[imgShardS, H, D]`）

最后flatten回：

- `[imgShardS, H*D]`

至此，图像输出回到 **sequence shard + full heads** 的布局。

对应代码（img_attn reshape -> all2all_head2seq -> reshape flatten）：

```215:231:LightX2V/lightx2v/common/ops/attn/ulysses_attn.py
    @torch.compiler.disable
    def _reshape_img_attn(self, img_attn, world_size, shard_seqlen, shard_heads, hidden_dims, seq_p_group, use_fp8_comm):
        img_attn = img_attn.reshape(world_size * shard_seqlen, shard_heads, hidden_dims)  # 重塑图像注意力结果
 
        # 将头的格式转换回序列格式
        if use_fp8_comm:
            original_dtype = img_attn.dtype
            original_shape = img_attn.shape
            img_attn_fp8, attn_scale = quant_fp8_vllm(img_attn.reshape(-1, original_shape[-1]))
            img_attn_fp8 = all2all_head2seq(img_attn_fp8.reshape(original_shape), group=seq_p_group)
            attn_scale = all2all_head2seq(attn_scale.reshape(original_shape[0], original_shape[1], 1), group=seq_p_group)
            img_attn = dequant_fp8_vllm(img_attn_fp8, attn_scale, original_dtype)
        else:
            img_attn = all2all_head2seq(img_attn, group=seq_p_group)
 
        img_attn = img_attn.reshape(shard_seqlen, -1)  # 重塑为 [shard_seqlen, -1] 形状
        return img_attn
```

对应代码（最终拼接输出 attn，按 img_first 决定拼接顺序）：

```207:213:LightX2V/lightx2v/common/ops/attn/ulysses_attn.py
        # 合并图像和文本的注意力结果
        if img_first:
            attn = torch.cat([img_attn, txt_attn], dim=0)
        else:
            attn = torch.cat([txt_attn, img_attn], dim=0)
 
        return attn  # 返回最终的注意力结果
```


# 0x3. `all2all_head2seq` 维度推导

`all2all_head2seq` 的目标是把输入 `X: [S, H//P, D]`（要求 `S % P == 0`、`H % P == 0`）转换为输出 `[S//P, H, D]`。也就是说先把 `S` 切为 `P` 份并 reshape 为 `[P, shardS, shardH, D]`（`shardS = S//P`、`shardH = H//P`），再 transpose 得到 `[P, shardH, shardS, D]`，执行 `dist.all_to_all_single` 后形状保持不变，最后合并前两维得到 `[H, shardS, D]`，再 transpose 得到 `[shardS, H, D]`。`all2all_seq2head` 完成相反方向的映射。


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

对应代码（默认实现：FP8 量化/scale reshape 与 dequant，分别位于两条分支）：

```105:117:LightX2V/lightx2v/common/ops/attn/ulysses_attn.py
            if use_fp8_comm:
                img_qkv_fp8, img_qkv_scale = quant_fp8_vllm(img_qkv.reshape(-1, hidden_dims))
                img_qkv_fp8 = img_qkv_fp8.reshape(shard_heads, world_size, img_qkv_len, 3, hidden_dims)
                img_qkv_scale = img_qkv_scale.reshape(shard_heads, world_size, img_qkv_len, 3, 1)
                output_qkv_fp8 = torch.empty_like(img_qkv_fp8)
                output_qkv_scale = torch.empty_like(img_qkv_scale)
                comm_fp8_works = []
                comm_scale_works = []
                for h in range(shard_heads):
                    work_fp8 = dist.all_to_all_single(output_qkv_fp8[h], img_qkv_fp8[h], group=seq_p_group, async_op=True)
                    work_scale = dist.all_to_all_single(output_qkv_scale[h], img_qkv_scale[h], group=seq_p_group, async_op=True)
                    comm_fp8_works.append(work_fp8)
                    comm_scale_works.append(work_scale)
```

```161:169:LightX2V/lightx2v/common/ops/attn/ulysses_attn.py
            if use_fp8_comm:
                img_qkv_fp8, img_qkv_scale = quant_fp8_vllm(img_qkv.reshape(-1, hidden_dims))
                img_qkv_fp8 = img_qkv_fp8.reshape(world_size, img_qkv_len, shard_heads, 3, hidden_dims)
                img_qkv_scale = img_qkv_scale.reshape(world_size, img_qkv_len, shard_heads, 3, 1)
                output_qkv_fp8 = torch.empty_like(img_qkv_fp8)
                output_qkv_scale = torch.empty_like(img_qkv_scale)
                dist.all_to_all_single(output_qkv_fp8, img_qkv_fp8, group=seq_p_group)
                dist.all_to_all_single(output_qkv_scale, img_qkv_scale, group=seq_p_group)
                output_qkv = dequant_fp8_vllm(output_qkv_fp8, output_qkv_scale, original_dtype)
```



# 0x6. 总结

LightX2V 的 Ulysses Attention 实现通过两次图像 all-to-all（输入侧与输出侧）完成 layout 变换，使 attention 计算在 **full sequence + head shard** 的布局上进行；文本部分通过 `all_gather` 拼回 full heads。实现同时提供了：

- `enable_head_parallel`：更细粒度的 head 级处理
- `use_fp8_comm`：FP8 通信量化

这些选项的效果受运行环境、通信后端与数据规模影响，需要结合实际配置进行评估。我自己看着好像是给视频模型用的，这里只是记录一下Ulysses Attention的形状推导和流程，不做更多猜测。




