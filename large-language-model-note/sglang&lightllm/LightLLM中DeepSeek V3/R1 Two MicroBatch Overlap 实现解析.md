# LightLLM中DeepSeek V3/R1 Two MicroBatch Overlap 实现解析

## 目录

- [0x00 引言](#0x00-引言)
- [0x01 Two MicroBatch Overlap技术概述](#0x01-two-microbatch-overlap技术概述)
- [0x02 数据结构设计](#0x02-数据结构设计)
  - [0x02.1 MicroBatch数据结构](#0x021-microbatch数据结构)
  - [0x02.2 推理状态管理](#0x022-推理状态管理)
- [0x03 数据预处理实现](#0x03-数据预处理实现)
  - [0x03.1 Decode阶段的MicroBatch拆分](#0x031-decode阶段的microbatch拆分)
  - [0x03.2 Prefill阶段的MicroBatch拆分](#0x032-prefill阶段的microbatch拆分)
- [0x04 Two MicroBatch Overlap 核心执行流程](#0x04-two-microbatch-overlap-核心执行流程)
  - [0x04.1 MicroBatch Overlap Decode实现](#0x041-microbatch-overlap-decode实现)
  - [0x04.2 MicroBatch Overlap Prefill实现](#0x042-microbatch-overlap-prefill实现)
- [0x05 DeepSeek V3的MoE优化](#0x05-deepseek-v3的moe优化)
  - [0x05.1 Decoding阶段的Two MicroBatch Overlap实现](#0x051-decoding阶段的two-microbatch-overlap实现)
  - [0x05.2 Context/Prefill阶段的MoE重叠](#0x052-contextprefill阶段的moe重叠)
- [0x06 分布式通信优化](#0x06-分布式通信优化)
  - [0x06.1 进程组管理](#0x061-进程组管理)
  - [0x06.2 通信重叠策略](#0x062-通信重叠策略)
- [0x07 CUDA Graph优化](#0x07-cuda-graph优化)
  - [0x07.1 overlap模式的CUDA Graph](#0x071-overlap模式的cuda-graph)
  - [0x07.2 Graph replay 优化](#0x072-graph-replay-优化)
- [0x08 内存管理优化](#0x08-内存管理优化)
  - [0x08.1 动态内存分配](#0x081-动态内存分配)
  - [0x08.2 KV Cache管理](#0x082-kv-cache管理)
- [0x09 性能优化策略](#0x09-性能优化策略)
  - [0x09.1 Hook机制](#0x091-hook机制)
  - [0x09.2 事件同步](#0x092-事件同步)
- [0x10 LightLLM Two Micro Batch Overlap 配置和启用](#0x10-lightllm-two-micro-batch-overlap-配置和启用)
  - [0x10.1 环境变量配置](#0x101-环境变量配置)
  - [0x10.2 运行时检查](#0x102-运行时检查)
- [0x11 和DeepSeek Blog/Profiler的区别](#0x11-和deepseek-blogprofiler的区别)

---

## 0x00 引言

在DeepSeek-V3/R1推理系统概览中提到，多机多卡的专家并行会引入比较大的通信开销，所以DeepSeek使用了双 batch 重叠来掩盖通信开销，提高整体吞吐。下面的截图来自DeepSeek官方Blog：https://zhuanlan.zhihu.com/p/27181462601

![](https://files.mdnice.com/user/59/ce568f92-2837-4ebf-aae7-251c7dc7a456.png)

然后LightLLM比较早的在DeepSeek V3中实现了一下这个feature（SGLang目前也已经实现了这个Feature），这里就以LightLLM的Two MicroBatch Overlap为例子来walk through一下原理和实现细节。

## 0x01 Two MicroBatch Overlap技术概述

一图胜千言，这里再描述一下上面Blog中展示Two MicroBatch Overlap的图。

在DeepSeek V3的多机多卡专家并行推理中，由于专家并行会引入较大的通信开销，因此采用了双batch重叠技术来掩盖通信开销，提高整体吞吐量。该技术的核心思想是将推理过程拆分为两个micro batch，通过精心设计的执行顺序实现计算和通信的重叠。

对于Prefill阶段：两个batch的计算和通信交错进行，当一个batch在进行计算时可以掩盖另一个batch的通信开销。具体表现为108个流多处理器(SMs)负责计算任务，包括ATTN(注意力机制和MoE路由门控)、MLP等模块，而24个SMs专门处理通信任务，包括COMBINE(专家结果合并)和DISPATCH(专家分发)操作。两个micro batch(用橙色和绿色区分)在时间轴上交替执行，形成流水线式的overlap模式。

对于Decode阶段：由于不同阶段的执行时间存在差异，系统将attention部分拆分为两个stage，总共形成5个stage的流水线来实现计算和通信的重叠。132个SMs负责计算，包括SHARED(共享专家)、ATTN-0(MLA下投影和combine all-to-all之后的其他操作)、MLP、ATTN-1(核心注意力、注意力输出投影和MoE路由门控)等模块，而通信部分包括DISPATCH和COMBINE操作。这种设计使得在一个micro batch进行计算的同时，另一个micro batch可以进行通信操作，从而有效隐藏通信延迟，显著提升推理性能。

## 0x02 数据结构设计

### 0x02.1 MicroBatch数据结构

LightLLM定义了专门的数据结构来表示micro batch：

```python
@dataclass
class DecodeMicroBatch:
    batch_size: int
    total_token_num: int
    max_len_in_batch: int
    input_ids: torch.Tensor
    mem_indexes: torch.Tensor
    b_req_idx: torch.Tensor
    b_seq_len: torch.Tensor

@dataclass
class PrefillMicroBatch:
    batch_size: int
    total_token_num: int
    max_len_in_batch: int
    input_ids: torch.Tensor
    mem_indexes: torch.Tensor
    b_req_idx: torch.Tensor
    b_seq_len: torch.Tensor
    b_ready_cache_len: torch.Tensor
    multimodal_params: list
```

这些数据结构的设计考虑了以下几个方面：

**内存管理**：
- `mem_indexes`：管理KV cache的内存索引
- `b_req_idx`：请求索引，用于跟踪每个请求的状态

**批处理信息**：
- `batch_size`：micro batch的大小
- `total_token_num`：总token数量
- `max_len_in_batch`：批次中的最大序列长度

**模态支持**：
- `multimodal_params`：支持多模态输入参数

### 0x02.2 推理状态管理

每个micro batch都有独立的推理状态：

```python
def create_inferstate(cur_batch: DecodeMicroBatch, batch_index):
    infer_state = self.infer_state_class()
    infer_state.is_prefill = False
    infer_state.batch_size = cur_batch.batch_size
    infer_state.total_token_num = cur_batch.total_token_num
    infer_state.max_len_in_batch = cur_batch.max_len_in_batch
    infer_state.microbatch_index = batch_index  # 关键：micro batch索引
    
    # 分布式通信组设置
    infer_state.dist_group = dist_group_manager.get_group(batch_index)
    
    # 内存管理
    infer_state.mem_manager = self.mem_manager
    infer_state.req_manager = self.req_manager
    infer_state.mem_index = cur_batch.mem_indexes
    
    return infer_state
```

## 0x03 数据预处理实现

### 0x03.1 Decode阶段的MicroBatch拆分

```python
def padded_overlap_prepare_decode_inputs(req_objs: List[InferReq], max_decode_num: int, is_multimodal=False):
    assert max_decode_num != 0
    micro_batch_size = triton.cdiv(max_decode_num, 2)  # 平均分配
    micro_batch1_req_num = triton.cdiv(len(req_objs), 2)
    
    # 创建第一个micro batch
    micro_batch, run_reqs, padded_req_num = _padded_prepare_decode_micro_batch(
        req_objs[0:micro_batch1_req_num], micro_batch_size, is_multimodal=is_multimodal
    )
    
    # 创建第二个micro batch
    micro_batch1, run_reqs1, padded_req_num1 = _padded_prepare_decode_micro_batch(
        req_objs[micro_batch1_req_num:], micro_batch_size, is_multimodal=is_multimodal
    )

    return micro_batch, run_reqs, padded_req_num, micro_batch1, run_reqs1, padded_req_num1
```

**关键实现细节**：

1. **均匀分配**：使用`triton.cdiv`确保两个micro batch大小尽可能均匀
2. **Padding处理**：对于不足的请求，使用fake请求进行padding
3. **内存分配**：每个micro batch都有独立的内存索引

### 0x03.2 Prefill阶段的MicroBatch拆分

```python
def padded_overlap_prepare_prefill_inputs(req_objs: List[InferReq], max_prefill_num: int, is_multimodal=False):
    """
    为Prefill阶段准备重叠执行的两个MicroBatch
    
    Args:
        req_objs: 待处理的推理请求列表
        max_prefill_num: 最大prefill数量限制
        is_multimodal: 是否为多模态输入
    
    Returns:
        tuple: (micro_batch, run_reqs, padded_req_num, micro_batch1, run_reqs1, padded_req_num1)
    """
    assert max_prefill_num != 0
    
    # 将请求列表平均分成两部分，为两个micro batch做准备
    # 使用triton.cdiv确保向上取整，避免遗漏请求
    micro_batch1_req_num = triton.cdiv(len(req_objs), 2)
    
    # 创建第一个micro batch：处理前半部分请求
    micro_batch, run_reqs, padded_req_num = _padded_prepare_prefill_micro_batch(
        req_objs[0:micro_batch1_req_num], is_multimodal=is_multimodal
    )
    
    # 创建第二个micro batch：处理后半部分请求
    micro_batch1, run_reqs1, padded_req_num1 = _padded_prepare_prefill_micro_batch(
        req_objs[micro_batch1_req_num:], is_multimodal=is_multimodal
    )

    return micro_batch, run_reqs, padded_req_num, micro_batch1, run_reqs1, padded_req_num1


def _padded_prepare_prefill_micro_batch(req_objs: List[InferReq], is_multimodal=False):
    """
    为单个MicroBatch准备Prefill阶段的数据
    
    Args:
        req_objs: 分配给当前micro batch的请求列表
        is_multimodal: 是否支持多模态输入
    
    Returns:
        tuple: (micro_batch, run_reqs, padded_req_num)
    """
    # === 初始化数据收集变量 ===
    run_reqs = []                    # 实际运行的请求列表
    nopad_total_token_num = 0        # 未padding前的总token数量
    nopad_max_len_in_batch = 0       # 批次中的最大输入长度
    input_ids = []                   # 所有请求的input token ids
    nopad_b_req_idx = []             # 请求索引列表
    nopad_b_seq_len = []             # 每个请求的序列长度
    
    # prefill阶段只需要padding一个请求形成micro_batch
    # 并不需要两个micro batch的batch_size相同（与decode阶段不同）
    padded_req_num = 1 if len(req_objs) == 0 else 0
    
    b_ready_cache_len = []           # 每个请求已缓存的KV长度
    batch_multimodal_params = []     # 多模态参数列表

    # === 处理每个真实请求 ===
    for req in req_objs:
        run_reqs.append(req)
        batch_multimodal_params.append(req.multimodal_params)
        nopad_b_req_idx.append(req.req_idx)

        # 获取请求的完整input token序列
        input_token_ids = req.get_chuncked_input_token_ids()
        seq_len = len(input_token_ids)
        
        # 计算需要处理的新token长度（总长度 - 已缓存长度）
        input_token_len = seq_len - req.cur_kv_len
        
        # 提取需要处理的新token部分
        input_id = input_token_ids[req.cur_kv_len :]

        # 收集批次统计信息
        nopad_b_seq_len.append(seq_len)
        input_ids.extend(input_id)                    # 将新token添加到批次中
        nopad_total_token_num += seq_len              # 累加总token数
        nopad_max_len_in_batch = max(nopad_max_len_in_batch, input_token_len)  # 更新最大长度
        b_ready_cache_len.append(req.cur_kv_len)      # 记录已缓存长度

    # === 添加padding请求（如果需要）===
    # 当没有真实请求时，添加一个fake请求进行padding
    for _ in range(padded_req_num):
        input_ids.append(1)  # 添加一个dummy token
        nopad_b_req_idx.append(g_infer_context.req_manager.HOLD_REQUEST_ID)  # 使用占位请求ID
        nopad_b_seq_len.append(1)        # 序列长度为1
        b_ready_cache_len.append(0)      # 无已缓存内容
        nopad_total_token_num += 1       # 更新总token数
        nopad_max_len_in_batch = max(nopad_max_len_in_batch, 1)  # 更新最大长度

    # === 转换为CUDA张量 ===
    # 将Python列表转换为GPU上的张量，提高计算效率
    input_ids = torch.tensor(input_ids, dtype=torch.int64, device="cuda")
    nopad_b_req_idx = torch.tensor(nopad_b_req_idx, dtype=torch.int32, device="cuda")
    nopad_b_seq_len = torch.tensor(nopad_b_seq_len, dtype=torch.int32, device="cuda")
    b_ready_cache_len = torch.tensor(b_ready_cache_len, dtype=torch.int32, device="cuda")

    # === 动态内存管理 ===
    # 获取全局推理状态锁，确保内存分配的线程安全
    g_infer_state_lock.acquire()
    
    # 如果启用了radix cache，先释放足够的缓存空间
    if g_infer_context.radix_cache is not None:
        g_infer_context.radix_cache.free_radix_cache_to_get_enough_token(
            input_ids.shape[0] - padded_req_num  # 只为真实token分配空间
        )
    
    # 为真实token分配KV cache内存索引
    mem_indexes = g_infer_context.req_manager.mem_manager.alloc(
        input_ids.shape[0] - padded_req_num
    ).cuda()
    
    # 释放锁
    g_infer_state_lock.release()
    
    # === 处理padding token的内存索引 ===
    if padded_req_num > 0:
        # 为padding token创建占位内存索引
        padding_indexs = torch.full(
            (padded_req_num,),
            fill_value=g_infer_context.req_manager.mem_manager.HOLD_TOKEN_MEMINDEX,  # 占位索引值
            dtype=torch.int32,
            device="cuda",
        )
        # 将真实token和padding token的内存索引合并
        mem_indexes = torch.cat((mem_indexes, padding_indexs), dim=0)

    # === 创建PrefillMicroBatch对象 ===
    micro_batch = PrefillMicroBatch(
        batch_size=nopad_b_seq_len.shape[0],          # 批次大小（请求数量）
        total_token_num=nopad_total_token_num,        # 总token数量
        max_len_in_batch=nopad_max_len_in_batch,      # 批次中最大输入长度
        input_ids=input_ids,                          # 输入token序列
        mem_indexes=mem_indexes,                      # KV cache内存索引
        b_req_idx=nopad_b_req_idx,                    # 请求索引
        b_seq_len=nopad_b_seq_len,                    # 序列长度
        b_ready_cache_len=b_ready_cache_len,          # 已缓存长度
        multimodal_params=batch_multimodal_params,    # 多模态参数
    )

    return micro_batch, run_reqs, padded_req_num
```

## 0x04 Two MicroBatch Overlap 核心执行流程

### 0x04.1 MicroBatch Overlap Decode实现

```python
@torch.no_grad()
def microbatch_overlap_decode(self, batch: DecodeMicroBatch, batch1: DecodeMicroBatch):
    assert batch.batch_size == batch1.batch_size
    assert batch.mem_indexes.is_cuda
    assert batch1.mem_indexes.is_cuda
    
    input_ids, input_ids1 = batch.input_ids, batch1.input_ids

    # 创建两个独立的推理状态
    infer_state = create_inferstate(batch, 0)
    infer_state1 = create_inferstate(batch1, 1)

    # 初始化额外状态
    infer_state.init_some_extra_state(self, input_ids)
    infer_state1.init_some_extra_state(self, input_ids1)

    batch_size = batch.batch_size
    max_len_in_batch = max(batch.max_len_in_batch, batch1.max_len_in_batch)

    # CUDA Graph优化
    if self.graph is not None and self.graph.can_run(batch_size, max_len_in_batch):
        if self.graph.need_capture(batch_size):
            infer_state.is_cuda_graph = True
            infer_state1.is_cuda_graph = True

            predict_logics, predict_logics1 = self.graph.capture_decode(
                self._overlap_tpsp_token_forward,
                input_ids, infer_state,
                input_ids1=input_ids1, infer_state1=infer_state1,
            )
        else:
            predict_logics, predict_logics1 = self.graph.replay(
                input_ids, infer_state, 
                input_ids1=input_ids1, infer_state1=infer_state1
            )
    else:
        # 执行重叠的token forward
        predict_logics, predict_logics1 = self._overlap_tpsp_token_forward(
            input_ids, infer_state, 
            input_ids1=input_ids1, infer_state1=infer_state1
        )
    
    return predict_logics, predict_logics1
```

### 0x04.2 MicroBatch Overlap Prefill实现

```python
@torch.no_grad()
def microbatch_overlap_prefill(self, batch: PrefillMicroBatch, batch1: PrefillMicroBatch):
    assert batch.mem_indexes.is_cuda
    assert batch1.mem_indexes.is_cuda
    
    input_ids, input_ids1 = batch.input_ids, batch1.input_ids

    # 创建推理状态
    infer_state = create_inferstate(batch, 0)
    infer_state1 = create_inferstate(batch1, 1)

    infer_state.init_some_extra_state(self, input_ids)
    infer_state1.init_some_extra_state(self, input_ids1)

    # 执行重叠的context forward
    predict_logics, predict_logics1 = self._overlap_tpsp_context_forward(
        input_ids, infer_state, 
        input_ids1=input_ids1, infer_state1=infer_state1
    )
    
    # 清理DeepEP冲区
    dist_group_manager.clear_deepep_buffer()
    
    return predict_logics, predict_logics1
```

## 0x05 DeepSeek V3的MoE优化

### 0x05.1 Decoding阶段的Two MicroBatch Overlap实现

虽然这里的函数命名是overlap_tpsp_token_forward，但是其实这里实现的是EP模式下Decoding/Context阶段的Two MicroBatch Overlap，读者可以注意一下。看下面的代码可以结合一下上面的DeepSeek Blog中的流程图，不过无论是Decoding模式还是后面的Prefill(Context模式)，LightLLM对SHARED部分的调用顺序相比于DeepSeek Blog中的图都是有区别的，比如这里的Decoding阶段是把MoE计算这个块往后平移了一位，然后ATTN和Shared expert的计算并不是Blog中那种不同Micro Batch的一起计算，而是同一个Micro Batch的依次算。这样做应该也可以达到overlap的作用，但是是否有DeepSeek profiler中的作用还需要做一下profile，我感觉可以把这里的顺序和DeepSeek blog进行对齐，否则在某些特定case下可能会出现一些GPU上只有通信没有计算导致的空转情况。

```python
def overlap_tpsp_token_forward(
        self,
        input_embdings: torch.Tensor,
        input_embdings1: torch.Tensor,
        infer_state: Deepseek2InferStateInfo,
        infer_state1: Deepseek2InferStateInfo,
        layer_weight: Deepseek2TransformerLayerWeight,
    ):
        if not self.is_moe:
            return super().overlap_tpsp_token_forward(
                input_embdings, input_embdings1, infer_state, infer_state1, layer_weight
            )
        # 0 attention
        _0_input1 = self._att_norm(input_embdings, infer_state, layer_weight)
        _0_cache_kv = self._pre_cache_kv(infer_state, layer_weight)
        _0_q, _0_cache_kv = self._tpsp_get_qkv(_0_input1, _0_cache_kv, infer_state, layer_weight)
        _0_input1 = None
        self._post_cache_kv(_0_cache_kv, infer_state, layer_weight)
        _0_o = self._token_attention_kernel(_0_q, infer_state, layer_weight)
        _0_q = None
        _0_o = self._tpsp_get_o(_0_o, infer_state, layer_weight)
        input_embdings.add_(_0_o.view(-1, self.embed_dim_))
        _0_o = None
        _0_input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        _0_router_logits = layer_weight.moe_gate.mm(_0_input1)
        # 1 hook
        if getattr(infer_state1, "hook", None) is not None:
            infer_state1.hook()
            infer_state1.hook = None

        # 0 shared expert
        if self.n_shared_experts is not None:
            _0_shared_output = LlamaTransformerLayerInfer._ffn(self, _0_input1, infer_state, layer_weight)

        # 0 dispatch
        (
            _0_recv_x,
            _0_masked_m,
            _0_topk_idx,
            _0_topk_weight,
            _0_handle,
            _0_hook,
        ) = layer_weight.experts.low_latency_dispatch(_0_input1, _0_router_logits)
        infer_state.hook = _0_hook

        # 1 attention
        _1_input1 = self._att_norm(input_embdings1, infer_state1, layer_weight)
        _1_cache_kv = self._pre_cache_kv(infer_state1, layer_weight)
        _1_q, _1_cache_kv = self._tpsp_get_qkv(_1_input1, _1_cache_kv, infer_state1, layer_weight)
        _1_input1 = None
        self._post_cache_kv(_1_cache_kv, infer_state1, layer_weight)
        _1_o = self._token_attention_kernel(_1_q, infer_state1, layer_weight)
        _1_q = None
        _1_o = self._tpsp_get_o(_1_o, infer_state1, layer_weight)
        input_embdings1.add_(_1_o.view(-1, self.embed_dim_))
        _1_o = None
        _1_input1 = self._ffn_norm(input_embdings1, infer_state1, layer_weight)
        # to do gate and disptatch

        _1_router_logits = layer_weight.moe_gate.mm(_1_input1)
        # 0 hook
        if getattr(infer_state, "hook", None) is not None:
            infer_state.hook()
            infer_state.hook = None

        # 1 shared expert
        if self.n_shared_experts is not None:
            _1_shared_output = LlamaTransformerLayerInfer._ffn(self, _1_input1, infer_state1, layer_weight)

        # 1 dispatch
        (
            _1_recv_x,
            _1_masked_m,
            _1_topk_idx,
            _1_topk_weight,
            _1_handle,
            _1_hook,
        ) = layer_weight.experts.low_latency_dispatch(_1_input1, _1_router_logits)
        infer_state1.hook = _1_hook

        # moe calu
        expected_m = triton.cdiv(
            input_embdings.shape[0] * get_global_world_size() * self.num_experts_per_tok, self.n_routed_experts
        )
        _0_moe_out = layer_weight.experts.masked_group_gemm(_0_recv_x, _0_masked_m, input_embdings.dtype, expected_m)

        # 1 hook
        if getattr(infer_state1, "hook", None) is not None:
            infer_state1.hook()
            infer_state1.hook = None

        # 0 combine
        _0_ffn_out, _0_hook = layer_weight.experts.low_latency_combine(
            _0_moe_out, _0_topk_idx, _0_topk_weight, _0_handle
        )

        infer_state.hook = _0_hook

        # to do moe caclue
        _1_moe_out = layer_weight.experts.masked_group_gemm(_1_recv_x, _1_masked_m, input_embdings1.dtype, expected_m)

        # 0 hook
        if getattr(infer_state, "hook", None) is not None:
            infer_state.hook()
            _0_ffn_out *= self.routed_scaling_factor
            if self.n_shared_experts is not None:
                _0_ffn_out.add_(_0_shared_output)
            input_embdings.add_(_0_ffn_out.view(-1, self.embed_dim_))
            infer_state.hook = None

        # 1 combine
        _1_ffn_out, _1_hook = layer_weight.experts.low_latency_combine(
            _1_moe_out, _1_topk_idx, _1_topk_weight, _1_handle
        )

        def _1_hook_post():
            _1_hook()
            nonlocal _1_ffn_out
            _1_ffn_out *= self.routed_scaling_factor
            if self.n_shared_experts is not None:
                _1_ffn_out.add_(_1_shared_output)
            input_embdings1.add_(_1_ffn_out.view(-1, self.embed_dim_))
            return

        infer_state1.hook = _1_hook_post

        return input_embdings, input_embdings1
```



### 0x05.2 Context/Prefill阶段的MoE重叠

在prefill（Context）阶段，也是参考上面Blog中的流程图就可以了，但是需要注意的一点是LightLLM把2个橙色的SHARED和ATTN交换了位置 ，并且把绿色的SHARED也移动到了绿色的MoE之前做，代码如下：

```python
def overlap_tpsp_context_forward(
        self,
        input_embdings: torch.Tensor,
        input_embdings1: torch.Tensor,
        infer_state: Deepseek2InferStateInfo,
        infer_state1: Deepseek2InferStateInfo,
        layer_weight: Deepseek2TransformerLayerWeight,
    ):
        if not self.is_moe:
            return super().overlap_tpsp_context_forward(
                input_embdings, input_embdings1, infer_state, infer_state1, layer_weight
            )
        # 0 attention
        _0_input1 = self._att_norm(input_embdings, infer_state, layer_weight)
        _0_cache_kv = self._pre_cache_kv(infer_state, layer_weight)
        _0_q, _0_cache_kv = self._tpsp_get_qkv(_0_input1, _0_cache_kv, infer_state, layer_weight)
        _0_input1 = None
        self._post_cache_kv(_0_cache_kv, infer_state, layer_weight)
        _0_o = self._context_attention_kernel(_0_q, _0_cache_kv, infer_state, layer_weight)
        _0_q = None
        _0_o = self._tpsp_get_o(_0_o, infer_state, layer_weight)
        input_embdings.add_(_0_o.view(-1, self.embed_dim_))
        _0_o = None
        _0_input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        _0_router_logits = layer_weight.moe_gate.mm(_0_input1)

        # wait last 1 combine
        if getattr(infer_state1, "hook", None) is not None:
            infer_state1.hook()
            infer_state1.hook = None

        _0_topk_weight, _0_topk_idx, _0_qinput_tensor = layer_weight.experts.select_experts_and_quant_input(
            _0_input1, _0_router_logits
        )
        from deep_ep import Buffer

        _0_overlap_event = Buffer.capture()

        # 1 attention
        _1_input1 = self._att_norm(input_embdings1, infer_state1, layer_weight)
        _1_cache_kv = self._pre_cache_kv(infer_state1, layer_weight)
        _1_q, _1_cache_kv = self._tpsp_get_qkv(_1_input1, _1_cache_kv, infer_state1, layer_weight)
        _1_input1 = None
        self._post_cache_kv(_1_cache_kv, infer_state1, layer_weight)
        _1_o = self._context_attention_kernel(_1_q, _1_cache_kv, infer_state1, layer_weight)
        _1_q = None
        _1_o = self._tpsp_get_o(_1_o, infer_state1, layer_weight)
        input_embdings1.add_(_1_o.view(-1, self.embed_dim_))
        _1_o = None
        _1_input1 = self._ffn_norm(input_embdings1, infer_state1, layer_weight)
        # to do gate and disptatch

        _1_router_logits = layer_weight.moe_gate.mm(_1_input1)

        # 0 dispatch execute
        (
            _0_recv_x,
            _0_recv_topk_idx,
            _0_recv_topk_weight,
            _0_num_recv_tokens_per_expert_list,
            _0_handle,
            _0_hook,
        ) = layer_weight.experts.dispatch(_0_qinput_tensor, _0_topk_idx, _0_topk_weight, overlap_event=_0_overlap_event)
        infer_state.hook = _0_hook

        # wait 0 dispatch
        if getattr(infer_state, "hook", None) is not None:
            infer_state.hook()
            infer_state.hook = None

        _1_topk_weight, _1_topk_idx, _1_qinput_tensor = layer_weight.experts.select_experts_and_quant_input(
            _1_input1, _1_router_logits
        )

        _1_overlap_event = Buffer.capture()

        # 0 shared expert
        if self.n_shared_experts is not None:
            _0_shared_output = LlamaTransformerLayerInfer._ffn(self, _0_input1, infer_state, layer_weight)

        # 1 shared expert
        if self.n_shared_experts is not None:
            _1_shared_output = LlamaTransformerLayerInfer._ffn(self, _1_input1, infer_state1, layer_weight)

        # 0 moe calu
        _0_moe_out = layer_weight.experts.prefilled_group_gemm(
            _0_num_recv_tokens_per_expert_list, _0_recv_x, _0_recv_topk_idx, _0_recv_topk_weight
        )

        # 1 dispatch execute
        (
            _1_recv_x,
            _1_recv_topk_idx,
            _1_recv_topk_weight,
            _1_num_recv_tokens_per_expert_list,
            _1_handle,
            _1_hook,
        ) = layer_weight.experts.dispatch(_1_qinput_tensor, _1_topk_idx, _1_topk_weight, overlap_event=_1_overlap_event)
        infer_state1.hook = _1_hook

        # wait 1 dispatch
        if getattr(infer_state1, "hook", None) is not None:
            infer_state1.hook()
            infer_state1.hook = None

        _0_combine_event = Buffer.capture()
        # 0 combine execute
        _0_ffn_out, _0_hook = layer_weight.experts.combine(_0_moe_out, _0_handle, _0_combine_event)
        infer_state.hook = _0_hook

        # 1 moe calc
        _1_moe_out = layer_weight.experts.prefilled_group_gemm(
            _1_num_recv_tokens_per_expert_list, _1_recv_x, _1_recv_topk_idx, _1_recv_topk_weight
        )

        # wait 0 combine
        if getattr(infer_state, "hook", None) is not None:
            infer_state.hook()
            infer_state.hook = None

        _1_combine_event = Buffer.capture()

        _0_ffn_out *= self.routed_scaling_factor
        if self.n_shared_experts is not None:
            _0_ffn_out.add_(_0_shared_output)
        input_embdings.add_(_0_ffn_out.view(-1, self.embed_dim_))

        # 1 combine execute
        _1_ffn_out, _1_hook = layer_weight.experts.combine(_1_moe_out, _1_handle, _1_combine_event)

        def _1_hook_post():
            _1_hook()
            nonlocal _1_ffn_out
            _1_ffn_out *= self.routed_scaling_factor
            if self.n_shared_experts is not None:
                _1_ffn_out.add_(_1_shared_output)
            input_embdings1.add_(_1_ffn_out.view(-1, self.embed_dim_))
            return

        infer_state1.hook = _1_hook_post

        return input_embdings, input_embdings1
```

## 0x06 分布式通信优化

### 0x06.1 进程组管理

Two MicroBatch Overlap需要特殊的进程组管理策略：

```python
class DistributeGroupManager:
    def __init__(self):
        self.groups = []

    def create_groups(self, group_size: int):
        for i in range(group_size):
            group = CustomProcessGroup()
            group.init_custom_gather()
            group.init_custom_reduce()
            self.groups.append(group)
        return

    def get_group(self, group_index: int) -> CustomProcessGroup:
        return self.groups[group_index]
```

**关键特性**：
- **独立通信组**：每个micro batch使用独立的通信组
- **自定义操作**：支持自定义的gather和reduce操作 (这个我不确定用到没)
- **组索引管理**：通过索引快速获取对应的通信组

### 0x06.2 通信重叠策略

```python
# 在模型初始化时创建多个通信组
group_size = 1
if enable_decode_overlap or args.enable_prefill_microbatch_overlap:
    assert batch_size % 2 == 0, "batch size must be even number"
    group_size = 2

dist_group_manager.create_groups(group_size=group_size)
```

这里和DeepSeek Blog中看起来不一样，这里是每个Micro Batch的dispatch和Combine都在一个单独的通信组里。

## 0x07 CUDA Graph优化

### 0x07.1 overlap模式的CUDA Graph

```python
def warmup_overlap(self, model):
    logger.info("Begin capture overlap cudagraph, use the --disable_cudagraph to disable it.")
    for batch_size in range(self.max_batch_size, 0, -1):
        decode_batches = []
        for micro_batch_index in [0, 1]:
            # 为每个micro batch创建dummy数据
            # ... 数据准备
            
            micro_batch = DecodeMicroBatch(
                batch_size=batch_size,
                total_token_num=total_token_num,
                max_len_in_batch=prefill_input_len + 1,
                input_ids=torch.from_numpy(predict_ids).cuda().reshape(-1),
                mem_indexes=mem_indexes,
                b_req_idx=b_req_idx,
                b_seq_len=b_seq_len,
            )
            decode_batches.append(micro_batch)

        # 捕获重叠执行的CUDA Graph
        _, _ = model.microbatch_overlap_decode(decode_batches[0], decode_batches[1])
```

### 0x07.2 Graph replay 优化

```python
def _replay_overlap(self, input_ids, infer_state, input_ids1, infer_state1):
    batch_size = input_ids.shape[0]
    (
        graph_obj,
        graph_input_ids,
        graph_infer_state,
        graph_input_ids1,
        graph_infer_state1,
        graph_predict_logics,
        graph_predict_logics1,
    ) = self.graph[batch_size]
    
    # 复制输入数据到graph缓冲区
    graph_input_ids.copy_(input_ids)
    graph_infer_state.copy_for_cuda_graph(infer_state)
    graph_input_ids1.copy_(input_ids1)
    graph_infer_state1.copy_for_cuda_graph(infer_state1)
    
    #  replay graph
    graph_obj.replay()
    
    return graph_predict_logics, graph_predict_logics1
```

## 0x08 内存管理优化

### 0x08.1 动态内存分配

```python
def alloc_tensor(
    self,
    shape: Union[torch.Size, Iterable[int]],
    dtype: torch.dtype,
    device: str = "cuda",
    is_graph_out: bool = False,
    microbatch_index: int = 0,
) -> torch.Tensor:
    """
    microbatch_index参数是为了支持microbatch overlap模式所添加的参数，
    其值只能为0或者1，用以标记申请的tensor是用于第几个microbatch的。
    """
    return g_cache_manager.alloc_tensor(
        shape, dtype, device=device, 
        is_graph_out=is_graph_out, 
        microbatch_index=microbatch_index
    )
```

### 0x08.2 KV Cache管理

```python
def copy_kv_index_to_req(req_to_token_indexs, b_req_idx, b_seq_len, mem_index):
    """将KV cache索引复制到请求管理器中"""
    # 为每个micro batch独立管理KV cache索引
    # 确保两个micro batch之间的内存隔离
```

## 0x09 性能优化策略

### 0x09.1 Hook机制

LightLLM使用hook机制来同步两个micro batch之间的执行：

```python
# 等待上一个micro batch的hook完成
if getattr(infer_state1, "hook", None) is not None:
    infer_state1.hook()
    infer_state1.hook = None

# 设置当前micro batch的hook
def _1_hook_post():
    _1_hook()
    nonlocal _1_ffn_out
    _1_ffn_out *= self.routed_scaling_factor
    if self.n_shared_experts is not None:
        _1_ffn_out.add_(_1_shared_output)
    input_embdings1.add_(_1_ffn_out.view(-1, self.embed_dim_))
    return

infer_state1.hook = _1_hook_post
```

### 0x09.2 事件同步

使用CUDA事件来精确控制计算和通信的重叠：

```python
from deep_ep import Buffer

_0_overlap_event = Buffer.capture()  # 捕获重叠事件
_0_combine_event = Buffer.capture()  # 捕获合并事件
```


> 这个_0_overlap_event命名有点奇怪，其实改成_0_dispatch_event更合理，因为这里捕获的是dispatch事件。dispatch和combine event都属于overlap event。


## 0x10 LightLLM Two Micro Batch Overlap 配置和启用

### 0x10.1 环境变量配置

```python
# 启用decode阶段的microbatch overlap
--enable_decode_microbatch_overlap

# 启用prefill阶段的microbatch overlap  
--enable_prefill_microbatch_overlap
```

### 0x10.2 运行时检查

```python
def __init__(self) -> None:
    super().__init__()
    self.enable_decode_microbatch_overlap = get_env_start_args().enable_decode_microbatch_overlap
    self.enable_prefill_microbatch_overlap = get_env_start_args().enable_prefill_microbatch_overlap

# 运行时选择执行路径
if not self.enable_decode_microbatch_overlap:
    self.normal_decode(decode_reqs, max_decode_num, uninit_reqs, ok_finished_reqs)
else:
    self.overlap_decode(decode_reqs, max_decode_num, uninit_reqs, ok_finished_reqs)
```

## 0x11 和DeepSeek Blog/Profiler的区别

上面提到了dispatch/combine以及compute的op dispatch顺序和DeepSeek Blog/Profiler中的图有区别。此外，对于2个Micro Batch LightLLM使用了不同的通信Stream，并且这个做法在Prefill时应该无法做到108 SMs来做deepgemm，然后24 SMs来做deepep通信？

这里提到的代码片段可以在 https://github.com/ModelTC/lightllm 找到。

