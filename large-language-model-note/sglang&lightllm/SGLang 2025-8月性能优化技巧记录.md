本笔记记录2025年8月SGLang框架的性能优化实践，主要包含以下技术要点：

1. 时间线说明：第20点及后续内容为8月下旬新增优化点，实际应排列在文档最前位置。第1-19点为8月前3周已实现方案。

2. 内容范围：基于个人技术视角，精选部分熟悉/感兴趣的优化点进行核心原理说明。对各优化点仅做技术概要分析，具体实现细节请参考对应PR。

3. 特别声明：
   - 本笔记为知识传播用途，不代表官方观点
   - 所有优化效果数据均来自原始PR验证结果
   - 欢迎对技术细节进行指正与讨论

## 1. GPT-OSS 和 DeepSeek-V3/R1 应用allreduce_add_rmsnorm优化

使用了FlashInfer提供的trt-llm-allreduce-add-rmsnorm kernel api，应用到SGLang之后在GPT-OSS和DeepSeek-V3/R1上取得了性能提升，这里还调了一些影响性能的参数。

相关PR: https://github.com/sgl-project/sglang/pull/9278 & https://github.com/sgl-project/sglang/pull/8731 & https://github.com/sgl-project/sglang/pull/7775 & https://github.com/sgl-project/sglang/pull/7621

Benchmark结果可以参考：https://github.com/sgl-project/sglang/pull/8731#issuecomment-3173435636 & https://github.com/sgl-project/sglang/pull/7775#issue-3201868800 。例如对于bs=1来说，gpt-oss-120b b200 tp4部署时启用此优化可以端到端提升8%的输出吞吐，b200 tp8部署DeepSeek-V3/R1时可以提升14%的输出吞吐。

后续在 https://github.com/sgl-project/sglang/pull/9339 中也可以在Hopper中开启这个优化。

## 2. Quant 和 RoPE kernel 支持 PDL

PR见：https://github.com/sgl-project/sglang/pull/9106

对于bs=1来说，gpt-oss-120b tp4部署时端到端吞吐可以提升3%。

## 3. 优化H20上Cutlass Fused MoE的性能

PR见：https://github.com/sgl-project/sglang/pull/9272

在H20上，为fp8_blockwise_scaled_group_mm kernel配置了专门的调优参数：

- `MmaTileShape: 64x128x128`
- `ClusterShape: 1x2x1`
- 使用`KernelPtrArrayTmaWarpSpecializedPingpongFP8BlockScaledAccum`调度策略

在Hopper上，`fp8_blockwise_scaled_group_mm`整体的逻辑为：

```c++
if (is_h20 && tuning_H20_kernel) {
    // 使用H20专用优化配置
    using execute_gemm_config = sm90_fp8_pp_config_64_128_128_1_2_1;
} else {
    if (multiProcessorCount == 78 && a.size(1) > 128) {
        // K > 128时，使用Pingpong调度策略 (MmaConfig0)
        // MmaTileShape: 64x128x128, ClusterShape: 2x1x1
    } else {
        // K <= 128时，使用Cooperative调度策略 (MmaConfig1) 
        // MmaTileShape: 128x128x128, ClusterShape: 1x2x1
    }
}
```

Benchmark结果：

![](https://files.mdnice.com/user/59/394fa025-d875-4bef-a763-0c40515f303b.png)


## 4. FlashInfer Cutlass MoE DP通信优化：FP4量化+Allgatherv

针对FlashInfer Cutlass MoE在数据并行（DP）场景下的通信瓶颈进行了专门优化，通过在allgather之前进行FP4量化来减少通信量，并引入了Allgatherv和Reducescatterv集体通信操作。

相关PR：https://github.com/sgl-project/sglang/pull/7667

主要改进包括：

1. 添加Allgatherv集体通信操作：基于PyNCCL实现的TensorRT-LLM allgather版本，支持不同rank的变长输入和tensor列表作为输入
2. 添加Reducescatterv集体通信操作：基于PyNCCL实现的TensorRT-LLM reducescatter版本，支持不同rank的变长输入
3. MoE通信流程优化：对于带有DP的FlashInfer MoE，使用allgatherv分发tokens，在allgather之前进行FP4量化减少通信量，最后使用reducescatterv合并结果替代all_reduce

启用条件：当`--enable-flashinfer-cutlass-moe`、`--enable-dp-attention`和`dp_size == ep_size`同时满足时自动启用，可通过`--disable-flashinfer-cutlass-moe-fp4-allgather`禁用。

性能提升：在DeepSeek-R1-0528-FP4模型上测试，端到端吞吐量提升9.38%（从27,763 tok/s提升到30,367 tok/s）。

## 5. per_token_group_quant_8bit kernel启用Fast Math优化

针对CUDA版本间性能差异问题，通过为`per_token_group_quant_8bit` kernel启用Fast Math编译选项显著提升了量化算子的性能。

相关PR：https://github.com/sgl-project/sglang/pull/9177

问题发现：在测试中发现SGLang在CUDA 12.4上比CUDA 12.8更快，通过分析SASS代码发现CUDA 12.4默认启用了`-ftz`或`--use_fast_math`选项，而CUDA 12.8没有。

技术实现：通过CMake的`set_source_files_properties`为特定源文件`csrc/gemm/per_token_group_quant_8bit`添加`--use_fast_math`编译选项。

```cmake
set_source_files_properties("csrc/gemm/per_token_group_quant_8bit" 
                            PROPERTIES COMPILE_OPTIONS "--use_fast_math")
```

性能提升对比：
- CUDA 12.4（默认Fast Math）：Duration 53.60μs，SM吞吐率70.90%
- CUDA 12.8（无Fast Math）：Duration 81.02μs，SM吞吐率68.67%
- 优化效果：启用Fast Math后性能提升约34%，将执行时间从81.02μs降低到53.60μs

关键SASS代码差异：
```asm
// CUDA 12.4 (Fast Math)
FMNMX.FTZ R14, R15, |R14|, !PT

// CUDA 12.8 (Standard Math)  
FMNMX R8, R8, |R17|, !PT
```

## 6. FP4 MoE量化Kernel优化：Grid-Stride布局+动态启动配置调优

移植vLLM的FP4 MoE kernel优化方案到SGLang，专门针对NVIDIA Blackwell GPU上的专家混合模型FP4量化性能进行了全面优化。

相关PR：https://github.com/sgl-project/sglang/pull/8777

优化技巧：

1. Grid-Stride循环布局：替换传统的每块行处理方式，实现更好的线程级并行性
2. 动态启动配置调优：当网格大小小于SM数量且块大小较大时，自动将网格大小翻倍、块大小减半，提高GPU占用率
3. 分层内存访问优化：
   - 小规模场景：专家偏移直接读入寄存器，实现低开销查找
   - 大规模场景：专家偏移先加载到共享内存，通过二分搜索访问提高效率

动态启动配置调优技术实现：
```c++
// 动态启动配置调优逻辑
int const numBlocksPerSM = 2048 / block.x;
if (grid.x < numSMs && block.x > threshold) {
    grid.x *= 2;    // 网格大小翻倍
    block.x /= 2;   // 块大小减半
}
```

![](https://files.mdnice.com/user/59/10ead02e-8d4d-416f-b999-d8fc496cefec.png)

## 7. DeeoSeek-V3, Qwen和Llama4模型支持DP Attention下的Reduce-Scatter通信优化

扩展reduce-scatter通信优化到更多模型架构，为DeepSeek-V3、Qwen2 MoE、Qwen3 MoE和Llama4在启用DP attention最大填充时，在MoE/MLP层后使用reduce-scatter替换all-reduce操作。

相关PR：https://github.com/sgl-project/sglang/pull/8539 & https://github.com/sgl-project/sglang/pull/9101

优化原理：在使用数据并行（DP）attention的max padding场景下，通过LayerCommunicator实现reduce-scatter替换all-reduce，减少通信开销和内存占用。

涉及模型：
- Qwen2 MoE：修改`qwen2_moe.py`支持reduce-scatter通信模式
- Qwen3 MoE：修改`qwen3_moe.py`支持reduce-scatter通信模式  
- Llama4：修改`llama.py`和`llama4.py`，确保MLP层实现的一致性

关键实现：复用`LayerCommunicator`框架，在forward pass中根据DP状态自动选择通信策略：
```python
if self.tp_size > 1:
    if skip_all_reduce:
        # 使用reduce-scatter模式
        output = tensor_model_parallel_reduce_scatter(output)
    else:
        # 传统all-reduce模式
        output = tensor_model_parallel_all_reduce(output)
```

性能提升：
- Qwen3-235B测试：总token吞吐量达到12,692 tok/s，端到端延迟显著降低

相关资料：[梳理SGLang中DP Attention及其Padding问题](https://mp.weixin.qq.com/s/W0e6W4-v8PmzP10qXY71rQ)

这里有一个问题，为什么reduce-scatter可以代替all-reduce？

传统通信流程：
```
MLP/MoE层 → all-reduce → layer后处理 → scatter到各DP rank
```

优化后通信流程：
```
MLP/MoE层 → 跳过all-reduce → layer后处理 → reduce-scatter (合并reduce+scatter)
```

源码实现关键逻辑：

1. 判断条件 (`communicator.py:264-270`)：
```python
def should_use_reduce_scatter(self, forward_batch: ForwardBatch):
    return (
        self.allow_reduce_scatter
        and forward_batch.dp_padding_mode.is_max_len()  # 使用max padding
        and self._communicate_summable_tensor_pair_fn is _scatter_hidden_states
    )
```

2. MoE层跳过all-reduce (`qwen2_moe.py:190-191`)：
```python
if self.tp_size > 1 and not use_reduce_scatter:
    final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
```

3. 延迟到layer结束执行reduce-scatter (`communicator.py:603-605`)：
```python
if allow_reduce_scatter and forward_batch.dp_padding_mode.is_max_len():
    # 在这里执行reduce-scatter，替代之前的all-reduce + scatter
    dp_reduce_scatter_tensor(hidden_states, global_hidden_states)
```

数学等价性：
- 传统方式：`scatter(all_reduce(X)) = scatter(sum(X_i)) = sum(X_i) / DP_size`
- 优化方式：`reduce_scatter(X) = sum(X_i) / DP_size` 

性能优势：
- 通信量减少：避免中间的全量数据传输，直接将reduce和scatter操作合并
- 内存优化：无需存储全量的all-reduce结果
- Pipeline效率：减少同步点，提高并行度

前提条件：
- Max Padding模式：确保各rank间数据对齐，支持直接reduce-scatter
- DP场景：数据并行环境下各rank计算相同内容的不同部分
- Layer结构：MLP/MoE输出需要最终分散到各DP rank

## 8. GPT-OSS模型支持FlashAttention-3后端：Attention Sinks优化

为GPT-OSS模型添加FlashAttention-3（FA3）后端支持，引入attention sinks功能，进一步提升长序列推理的性能和内存效率。

相关PR：https://github.com/sgl-project/sglang/pull/9028

核心特性：
- Attention Sinks支持：通过attention层传递`sinks`参数，启用FA3的attention sinks功能
- 数据类型优化：将`sinks`参数的数据类型更新为`bfloat16`，提升计算效率和一致性
- 后端扩展：在服务器参数中添加"fa3"作为有效的attention后端选项

技术实现：
```python
# 支持FA3 attention sinks
def forward(self, sinks=None, **kwargs):
    if self.attention_backend == "fa3":
        # 使用FA3后端处理attention sinks
        attn_output = flashattn3_forward(sinks=sinks, ...)
```

Benchmark结果：

GPT-OSS-20B (TP1, 4k输入/1k输出)：
- Triton后端：
  - Concurrency=1: 输出吞吐303.066 tok/s，TTFT 95.071ms
  - Concurrency=32: 输出吞吐3,067.798 tok/s，TTFT 1,861.553ms
- FA3后端：
  - Concurrency=1: 输出吞吐309.425 tok/s，TTFT 75.511ms (提升2.1%)
  - Concurrency=32: 输出吞吐3,057.230 tok/s，TTFT 1,271.047ms (TTFT降低31.7%)

从Benchmark结果来看，似乎并发=32的时候吞吐还略有降低。

## 9. Blackwell GPU上的FP8 CUTLASS Kernel调优：动态配置分发

针对NVIDIA Blackwell（SM100）GPU架构，移植vLLM的FP8 GEMM性能调优技术，通过动态kernel分发机制显著提升矩阵乘法运算性能。

相关PR：https://github.com/sgl-project/sglang/pull/8818

优化技巧：
- 动态Kernel分发：根据输入矩阵的M维度自动选择最优的CUTLASS kernel配置
- 分段配置优化：针对不同矩阵大小范围提供专门的启动配置
- 替换单一配置：从原来的通用配置改为精细化的分段配置策略

具体实现细节：

配置分段策略：
- [1, 16]：小规模矩阵的轻量级配置
- (16, 64]：中小规模矩阵的平衡配置  
- (64, 256]：中等规模矩阵的性能配置
- (256, xxx]：大规模矩阵的高吞吐配置

动态分发逻辑：
```cpp
// 根据M维度动态选择kernel配置
template<typename T>
auto select_fp8_gemm_config(int M) {
    if (M <= 16) return small_config;
    else if (M <= 64) return medium_small_config;  
    else if (M <= 256) return medium_config;
    else return large_config;
}
```

benchmark有点长了，可以直接看PR。

## 10. FlashInfer的TensorRT-LLM FP8 Blockscale GEMM后端优化

升级SGLang的FlashInfer CUTLASS后端至TensorRT-LLM FP8 GEMM实现，专注 Low Latency 优化。在DeepSeek-R1-0528模型TP8+DP8部署中，请求吞吐提升6%（0.83→0.88 req/s），首token延迟降低9%（10.1→9.2秒），整体吞吐提升6.7%（7.6k→8.1k tok/s）。

相关PR：https://github.com/sgl-project/sglang/pull/8588

## 11. 自定义set kv buffer Kernel fuse

针对H100 GPU上`set_kv_cache`操作的性能开销问题，开发自定义CUDA kernel融合key和value缓存的存储操作，显著降低内存操作开销。

相关PR：https://github.com/sgl-project/sglang/pull/8884

问题识别：通过nsys性能分析工具发现，在H100上`set_kv_cache`操作存在明显的性能开销，成为推理过程中的瓶颈之一。

核心优化：
- 操作融合：将原来分离的key cache和value cache存储操作融合为单个CUDA kernel
- 内存访问优化：减少内存操作次数，提高内存带宽利用率

技术实现：

自定义CUDA Kernel：
```cpp
// 融合KV缓存存储的CUDA kernel
__global__ void set_kv_buffer_kernel(
    scalar_t* k_cache,
    scalar_t* v_cache, 
    const int64_t* loc,
    const scalar_t* k,
    const scalar_t* v
) {
    // 同时处理key和value的存储操作
    // 减少内存访问开销
}
```

Python接口封装：
```python
def set_kv_buffer_kernel(k_cache, v_cache, loc, k, v, fallback=False):
    try:
        if fallback:
            raise RuntimeError("Fallback to torch implementation")
        torch.ops.sgl_kernel.store_kv_cache(k_cache, v_cache, loc, k, v)
    except RuntimeError:  # 回退到PyTorch实现
        k_cache[loc] = k
        v_cache[loc] = v
```

![](https://files.mdnice.com/user/59/b7f0a008-19c8-4173-a4ba-38f758b62033.png)

## 12. RoPE + Set KV Buffer kernel fuse

将KV缓存写入操作直接融合到RoPE（旋转位置编码）kernel中，通过消除独立的内存操作和kernel启动开销，提升推理性能。

相关PR：https://github.com/sgl-project/sglang/pull/9077 & https://github.com/sgl-project/sglang/pull/9014

伪代码：

```cpp
// 在RoPE kernel中直接写入KV缓存
__global__ void BatchQKApplyRotaryPosIdsCosSinCacheEnhanced(
    // RoPE相关参数
    scalar_t* __restrict__ query,
    scalar_t* __restrict__ key, 
    const scalar_t* cos_ptr,
    const scalar_t* sin_ptr,
    // KV缓存融合参数
    std::optional<scalar_t*> kv_buffer = std::nullopt,
    std::optional<int64_t*> cache_loc = std::nullopt
) {
    // 执行RoPE计算
    apply_rotary_embedding(query, key, cos, sin);
    
    // 融合的KV缓存写入
    if (kv_buffer.has_value()) {
        write_to_kv_cache(key, value, kv_buffer, cache_loc);
    }
}
```

从gpt-oss-120b tp4的部署结果来看，端到端的吞吐可以提升3%。

## 13. GPT-OSS模型MoE层填充与量化Kernel融合优化

针对GPT-OSS模型的MoE层特性，将隐藏状态填充操作融合到量化kernel中，消除冗余内存操作。相关PR：https://github.com/sgl-project/sglang/pull/9005

核心改进：
- 移除GPT-OSS模型中`FusedMoELayer`的显式填充/反填充步骤
- 在量化函数`mxfp8_quantize`中直接处理维度对齐
- 通过`output_hidden_size`参数动态控制填充逻辑

实现对比：
```python
# 优化前（分离式）
def forward(self, hidden_states):
    hidden_states = pad(hidden_states, target_size)  # 显式填充
    x_quant = mxfp8_quantize(hidden_states)
    output = moe_computation(x_quant)
    return unpad(output, original_size)  # 显式反填充

# 优化后（融合式） 
def forward(self, hidden_states):
    x_quant = mxfp8_quantize(hidden_states, output_hidden_size=target_size)
    return trtllm_fp4_block_scale_moe(x_quant)
```

量化函数改进：
```python
def mxfp8_quantize(x, output_hidden_size=None):
    if output_hidden_size:  # 自动处理GPT-OSS的维度对齐需求
        x = F.pad(x, (0, output_hidden_size - x.shape[-1]))
    return quantize_fp8(x)  # 保持原有量化逻辑
```

## 14. DP场景MoE非填充Token计数优化

修复数据并行场景下MoE计算中token计数不准确问题

PR：https://github.com/sgl-project/sglang/pull/9107

问题分析：原实现错误使用全局DP rank的token数，导致：
1. 高估实际有效token数
2. MoE路由无法正确屏蔽填充token
3. 浪费计算资源处理无效token

解决方案：

```python
# 精确local token计数
def get_num_token_non_padded_local(total_tokens, tp_size, tp_rank):
    base = total_tokens // tp_size
    extra = 1 if tp_rank < (total_tokens % tp_size) else 0
    return base + extra

# MoE层调用
def forward(self, hidden_states, batch_info):
    local_tokens = get_num_token_non_padded_local(
        batch_info.total_tokens,
        batch_info.tp_size,
        batch_info.tp_rank
    )
    return moe_layer(hidden_states, num_token=local_tokens)
```

主要改动：
- 在batch信息中添加TP并行维度参数
- 所有MoE层统一适配新计数逻辑
- 保持原有接口兼容性

Benchmark结果，DeepSeek-V3-0324 DP部署：

```shell
python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3-0324 --trust-remote-code --tp 8 --enable-dp-attention --dp 2 --moe-dense-tp-size 1 --moe-a2a-backend deepep --cuda-graph-max-bs 32 --max-running-requests 32 --speculative-algo EAGLE --speculative-draft lmsys/DeepSeek-V3-0324-NextN --speculative-num-steps 2 --speculative-eagle-topk 4 --speculative-num-draft-tokens 4 --disable-radix-cache --stream-output
python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompt 32 --random-input 512 --random-output 32 --random-range-ratio 1 --host 127.0.0.1 --port 30000 --max-concurrency 1
```

吞吐量从53.05 → 57.07 tok/s，提升7.6%

## 15. TRTLLM-MLA FP8 

支持FlashInfer的TRTLLM-MLA FP8 KV Cache Backend。详情请看PR：https://github.com/sgl-project/sglang/pull/8638

Benchmark结果：

![](https://files.mdnice.com/user/59/ae4530bb-cad3-4f05-a666-9ee70b4bc1ee.png)

## 16. MoE routed scaling factor kernel fuse

将 routed scaling factor 计算融合到`moe_fused_gate`和`select_experts` kernel，减少独立操作开销。

PR：https://github.com/sgl-project/sglang/pull/8770


实现示例：
```python
def moe_fused_gate(..., apply_routed_scaling_factor_on_output=False):
    return fused_gate_with_scaling(...) if apply else traditional(...)
```

```cpp
__global__ void moe_fused_gate_kernel(...) {
    if (apply_scaling_factor) gate_output *= topk_weights[idx];
}
```

Benchmark结果：

![](https://files.mdnice.com/user/59/37321d1f-1bb8-4803-bb53-8498141d4ac6.png)

## 17. GPT-OSS模型Attention Sinks支持与TRT-LLM MHA后端优化

针对GPT-OSS 开源模型，增强TensorRT-LLM多头注意力（MHA）后端支持，引入Attention Sinks机制，显著提升长序列推理性能和内存效率。

相关PR：https://github.com/sgl-project/sglang/pull/8834 & https://github.com/sgl-project/sglang/pull/8782

主要修改：

1. TRT-LLM MHA后端集成：直接调用TensorRT-LLM生成的多头注意力模块，利用其高度优化的kernel实现
2. Attention Sinks支持：为TRT-LLM后端添加attention sinks功能，通过`sk`参数传递sink配置信息
3. 长序列优化：专门针对长序列推理场景的内存和计算优化

Attention Sinks集成：
```python
# TRT-LLM MHA后端中的attention sinks支持
def forward_extend(layer, forward_batch, save_kv_cache=True, **kwargs):
    # 获取attention sink配置
    attention_sink = kwargs.get("sk", None)
    
    # 在TRT-LLM MHA中应用attention sinks
    if attention_sink is not None:
        # sink: additional value per head in the denominator of the softmax
        trtllm_mha_forward_with_sinks(attention_sink=attention_sink, ...)
```

后端选择逻辑：
```python
# 支持trtllm_mha作为attention backend选项
def get_attention_backend():
    valid_backends = ["triton", "fa3", "trtllm_mha"]
    if backend == "trtllm_mha":
        return TRTLLMMHABackend()
```

Benchmark结果：

GPT-OSS-20B模型测试：
```bash
# TRT-LLM MHA后端测试
python3 -m sglang.launch_server --model-path lmsys/gpt-oss-20b-bf16 \
    --trust-remote-code --attention-backend trtllm_mha \
    --enable-triton-kernel-moe --mem-fraction-static 0.7 \
    --tp-size 8 --disable-cuda-graph --disable-hybrid-swa-memory

# Benchmark对比
python3 benchmark/gsm8k/bench_sglang.py --num-shots 8 --num-questions 1000 --parallel 1000
```

- TRT-LLM MHA后端：17,151.680 token/s 
- Triton后端：14,607.150 token/s


## 18. TBO优化：Two Chunk Overlap技术

改进两批次重叠(Two Batch Overlap, TBO)策略，引入Two Chunk Overlap技术，通过将长序列智能分割为两个chunk并行执行，显著提升大规模分布式推理的吞吐性能。

相关PR：https://github.com/sgl-project/sglang/pull/8144

主要修改：

1. Two Chunk Overlap策略：将传统的Two Batch Overlap改进为Two Chunk Overlap，消除idle batch，提高GPU利用率
2. 智能序列分割：根据token分布阈值动态决定是否启用chunk分割，优化不同长度输入的处理效率
3. 分布式优化配置：针对多节点DP+TP部署场景进行专门优化，支持大规模模型推理
4. 参数可调节控制：通过`tbo-token-distribution-threshold`参数精确控制优化策略的触发条件

核心技术原理：

传统Two Batch Overlap vs Two Chunk Overlap：
```python
# 传统Two Batch Overlap (存在idle batch问题)
# batch_size = 1, extend_seq_len = [3072], extend_prefix_len = [0]
micro_batch0: extend_seq_len = [3072], extend_prefix_len = [0]  # 活跃批次
micro_batch1: extend_seq_len = [0], extend_prefix_len = [0]     # 空闲批次

# Two Chunk Overlap (消除idle batch)  
# batch_size = 1, extend_seq_len = [3072], extend_prefix_len = [0]
micro_batch0: extend_seq_len = [1536], extend_prefix_len = [0]    # 第一个chunk
micro_batch1: extend_seq_len = [1536], extend_prefix_len = [1536] # 第二个chunk
```

动态启用逻辑：
```python
# 根据token分布阈值决定是否启用Two Chunk Overlap
def should_enable_two_chunk_overlap(batch_info, threshold=0.48):
    token_distribution_ratio = calculate_token_distribution(batch_info)
    return token_distribution_ratio > threshold and enable_tbo
```

分布式部署配置优化：
```bash
# DeepSeek-V3大规模部署配置
SGLANG_TBO_DEBUG=1 SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=1 \
python3 -m sglang.launch_server \
    --model-path /dev/shm/DeepSeek-V3-0324 --tp 16 --dp 16 \
    --chunked-prefill-size 65536 --max-prefill-tokens 170000 \
    --enable-dp-attention --enable-deepep-moe \
    --enable-two-batch-overlap --tbo-token-distribution-threshold 0.48 \
    --disable-overlap-schedule --disable-radix-cache --disable-cuda-graph
```

Benchmark结果：

测试环境：2x8x H800, DeepSeek-V3-0324

场景1：特殊情况（每个DP单一长请求，长度3072）：
```bash
# 测试配置  
python3 -m sglang.bench_serving --backend sglang \
    --dataset-name random --num-prompt 1024 \
    --random-input 3072 --random-output 1 --random-range-ratio 1 \
    --max-concurrency 1024
```

性能提升：
- Baseline(禁用Two Chunk Overlap)：64,820.55 / 64,901.24 / 63,900.35 / 63,931.12 tok/s
- Two Chunk Overlap优化后：72,391.92 / 72,845.28 / 71,511.55 / 73,152.61 tok/s  
- 吞吐提升：平均12.56%

场景2：通用情况（变长输入30-3072 tokens）：
```bash
# 测试配置
python3 -m sglang.bench_serving --backend sglang \
    --dataset-name random --num-prompt 2048 \
    --random-input 3072 --random-output 1 --random-range-ratio 0.01 \
    --max-concurrency 1024
```

性能提升：
- Baseline(禁用Two Chunk Overlap)：71,695.06 / 68,585.39 / 75,784.86 / 77,115.82 tok/s
- Two Chunk Overlap优化后：77,532.36 / 77,059.32 / 75,654.72 / 78,041.45 tok/s
- 吞吐提升：平均5.15%

核心优势：
- 消除Idle Batch：相比传统TBO，消除了空闲微批次，提高GPU计算效率
- 更好的Overlap效果：两个微批次的延迟更加均衡，有利于计算通信重叠
- 智能适配：根据输入长度分布自动调节优化策略
- 大规模扩展性：在多节点DP场景下效果显著

该优化适用于处理长序列、大batch size的分布式推理场景，能够显著提升GPU集群的整体吞吐能力。从测试结果来看，genaral场景下似乎也有一些提升。

这里的智能分割方案比较复杂，感兴趣的小伙伴请自行到SGLang实现中了解。

## 19. DP Attention优化：LayerNorm前置到AllGather之前

将LayerNorm操作前置到数据并行(DP)的allgather通信之前执行，在1/DP数量的token上进行归一化计算，减少计算开销。仅在DP==TP时启用确保数值稳定性。

相关PR：https://github.com/sgl-project/sglang/pull/8631

Benchmark结果：

在DeepSeek-R1-0528-FP4模型上端到端吞吐提升3.79%（27,310→28,345 tok/s）。


## 20. H100/H200/H800 GPU上FP8 MoE Kernel调度策略优化

针对从H20迁移到H100/H200/H800时`fp8_blockwise_scaled_grouped_mm`性能回归问题，根据GPU架构动态选择最优调度策略：H20使用Pingpong Schedule，H100/H200/H800使用Cooperative Schedule。

相关PR：https://github.com/sgl-project/sglang/pull/8722

问题分析：H20到H100/H200/H800迁移时，Tensor Core MMA时间减少到1/4，但Pingpong调度中CUDA Core FMA无法重叠，导致SM Tensor Pipe吞吐下降。

**解决方法**：通过SM数量识别GPU架构，H20(78 SMs)继续使用Pingpong调度，其他Hopper架构使用Cooperative调度解决Tensor Core竞争问题。 

Benchmark结果：

![](https://files.mdnice.com/user/59/e866c5b9-7d17-4fd4-a0df-5af47ae1d596.png)

相关资料：关于Pingpong和Cooperative的一些感性理解(https://zhuanlan.zhihu.com/p/1922067252909434076) 以及对当前这个PR的详细解释：Pingpong Schedule并不是万能钥匙(https://zhuanlan.zhihu.com/p/1935338652726204054)

## 21. FlashInfer MoE Blockscale FP8后端支持TP MoE

扩展FlashInfer MoE blockscale FP8后端支持到张量并行(TP) MoE配置，新增`FlashInferFusedMoE`类封装优化逻辑，解耦对EP MoE的依赖。

相关PR：https://github.com/sgl-project/sglang/pull/8450

移除`enable_ep_moe`强制要求，TP MoE可独立使用`trtllm_fp8_block_scale_moe` kernel优化，支持per-token group量化和权重重排。 

Benchmark结果：https://github.com/sgl-project/sglang/pull/8450#issuecomment-3129426265

## 22. TRTLLM生成的MLA解码Kernel集成

集成TensorRT-LLM生成的多头潜在注意力(MLA)解码kernel，为DeepSeek系列模型提供专门优化的attention实现，支持SM100架构。

相关PR：https://github.com/sgl-project/sglang/pull/8632

针对MLA架构的解码阶段优化，通过特殊优化的kernel提升DeepSeek模型的推理性能，添加SM100兼容性检查确保硬件支持。 

Benchmark结果：

![](https://files.mdnice.com/user/59/9ce6ea6a-3fe0-4c02-bcfd-fad7d8cca3bb.png)

![](https://files.mdnice.com/user/59/555d4909-4af7-4e2e-ae15-b0a9f6f202f7.jpg)

![](https://files.mdnice.com/user/59/c8a07fbc-1f35-4ed1-9733-ea73be2264d8.png)

## 23. CUDA图捕获时禁用Python垃圾回收器优化

在CUDA图捕获期间禁用Python垃圾回收器(GC)，通过`gc.freeze()`避免GC扫描长期存在的对象，显著提升启动速度。

相关PR：https://github.com/sgl-project/sglang/pull/8577

性能提升：CUDA图捕获速度提升2.3x-3.7x。Llama4模型从25秒降至10秒，Qwen3-0.6B模型从6秒降至1秒。 

## 24. MRoPE多模态旋转位置编码torch.compile优化

为`MRotaryEmbedding.forward()`添加`torch.compile(dynamic=True)`，减少kernel启动开销，显著提升小型VLM模型的推理性能。

相关PR：https://github.com/sgl-project/sglang/pull/9487

性能提升：在Qwen2.5-VL-3B-Instruct上请求吞吐提升28%（2.53→3.25 req/s），MRoPE延迟减少8倍，ITL从5.86ms降至4.48ms。 

## 25. 垃圾回收器冻结功能减少延迟抖动

添加GC冻结功能，通过`freeze_gc` API将服务器预热后的长期对象排除在垃圾回收范围外，避免gen2 GC导致的100ms-300ms停顿，维持低延迟。

相关PR：https://github.com/sgl-project/sglang/pull/9241

实现特性：新增`/freeze_gc` HTTP端点、分布式GC管理、可配置GC警告阈值(`gc_warning_threshold_secs`)，有效解决P99延迟抖动问题。 这里的核心修改如下：

```python
def gc_object_counts():
    """获取各代垃圾回收器中的对象数量统计
    
    Python的垃圾回收器使用分代机制:
    - gen0: 新创建的对象，回收频率最高
    - gen1: 从gen0中存活下来的对象  
    - gen2: 从gen1中存活下来的长期对象，回收成本最高
    """
    import gc

    g0 = len(gc.get_objects(0))  # 统计第0代对象数量
    g1 = len(gc.get_objects(1))  # 统计第1代对象数量
    g2 = len(gc.get_objects(2))  # 统计第2代对象数量（长期对象）
    return g0, g1, g2


def configure_gc_warning(warn_threshold_secs):
    """配置垃圾回收警告机制
    
    当GC耗时超过指定阈值时，记录警告日志并提供优化建议。
    这有助于识别可能导致延迟抖动的长时间GC操作。
    
    Args:
        warn_threshold_secs: GC耗时警告阈值（秒）
    """
    import gc

    gc_start_time = {}  # 记录各代GC开始时间

    def gc_callback(phase, info):
        """GC事件回调函数，监控GC执行时间"""
        gen = info.get("generation", "?")  # 获取当前回收的代数
        
        if phase == "start":
            # GC开始时记录时间戳
            gc_start_time[gen] = time.time()
        elif phase == "stop":
            # GC结束时计算耗时并检查是否需要警告
            duration = time.time() - gc_start_time.get(gen, time.time())
            if duration > warn_threshold_secs:
                g0, g1, g2 = gc_object_counts()
                logger.warn(
                    f"LONG GARBAGE COLLECTION DETECTED | Generation {gen} | Duration: {duration:.4f}s | # Objects: gen0={g0}, gen1={g1}, gen2={g2} | "
                    f"This may cause latency jitter. Consider calling the freeze_gc API after sending a few warmup requests."
                )

    # 注册GC事件回调函数
    gc.callbacks.append(gc_callback)


def freeze_gc(context: str):
    """冻结垃圾回收器，将当前对象移出GC管理范围
    
    调用gc.freeze()后，当前存在的所有对象将被移到"永久代"，
    不再参与垃圾回收过程。这可以显著减少GC开销，特别是
    gen2回收的成本，从而减少延迟抖动。
    
    适合在服务器预热完成后调用，将长期存在的对象（如模型参数、
    缓存等）排除在GC范围外。
    
    Args:
        context: 调用上下文描述，用于日志记录
    """
    import gc

    # 冻结前记录各代对象数量
    g0_before, g1_before, g2_before = gc_object_counts()
    
    # 执行GC冻结操作 - 核心函数调用
    gc.freeze()
    
    # 冻结后记录各代对象数量变化
    g0_after, g1_after, g2_after = gc_object_counts()
    
    # 记录冻结操作的效果，便于监控和调试
    logger.info(
        f"Freezing GC in {context} process. "
        f"gen0: {g0_before}->{g0_after}, "
        f"gen1: {g1_before}->{g1_after}, "
        f"gen2: {g2_before}->{g2_after}"
    )
```

## 26. FlashInfer GPU-CPU同步优化

修复FlashInfer在page size=1时的不必要GPU-CPU同步问题，当page-size大小为1时直接构造`torch.ones`tensor替代GPU到CPU的数据传输。

相关PR：https://github.com/sgl-project/sglang/pull/9409

改动：

![](https://files.mdnice.com/user/59/9e06e122-5374-4cf8-b1cb-71f9daba6521.png)

Benchmark结果：

![](https://files.mdnice.com/user/59/367728b5-38c5-4bd4-972f-f59a5e91c86a.png)

在B200上以并发=1运行Qwen2.5-7B模型，总吞吐从425.01提升到437.64 tok/s，提升3.0%。 

## 27. FlashInfer GQA Tensor Core解码阈值优化

将FlashInfer中启用Tensor Core解码的GQA组大小阈值从`>4`降低到`>=4`，使Llama3-8B等具有4个GQA组的模型能够利用Tensor Core加速。

相关PR：https://github.com/sgl-project/sglang/pull/8624

性能提升：显著减少ITL（Inter-Token Latency），特别是对于GQA组大小为4的模型如Llama3-8B，因为FlashInfer将head group与token维度融合使得组大小为4就足以受益于Tensor Core。 

![](https://files.mdnice.com/user/59/3f9df819-4a5a-49ce-aaa2-98b8ca63a015.png)


## 28. CUTLASS 4.2升级与K-Major Scale Factor支持

升级CUTLASS库至4.2版本，为SM90 FP8 Blockwise Group GEMM启用K-Major Scale Factor，统一与Blackwell的代码路径，消除`per_group_transpose`格式转换开销。

相关PR：https://github.com/sgl-project/sglang/pull/9559

优化改进：支持K-Major格式scale factors、通过矩阵交换优化小M场景(M<=2048)、使用ATen接口优化H20设备检测性能，避免`cudaGetDeviceProperties`调用开销。 

优化提升见这个comment：https://github.com/sgl-project/sglang/pull/9559#issue-3349421711

## 29. FlashInfer/FlashMLA后端支持 Chunked Prefill 缓存优化

为FlashInfer和FlashMLA后端添加MHA Chunked Prefill 缓存支持，移除page size=1限制，支持更大的Page Size 以提升内存效率。

相关PR：https://github.com/sgl-project/sglang/pull/8616

功能扩展：支持page size>1的MHA Chunked Prefill 缓存、增强FlashInfer/FlashMLA后端兼容性，准确性测试显示不同page size配置下精度保持一致(GSM8K: 0.954-0.955)。 

Benchmark结果中在降低TTFT上有明显作用：

https://github.com/sgl-project/sglang/pull/8616#issue-3280333135



