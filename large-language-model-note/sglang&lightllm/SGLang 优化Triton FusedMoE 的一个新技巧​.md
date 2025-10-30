> Antgroup最近在SGLang中提交的一个针对FusedMoE的优化，也是一个有趣的发现，这里就学习记录并广播下。

# 0x0. 前言

最近蚂蚁在SGLang中提交了一个优化PR（[#10567](https://github.com/sgl-project/sglang/pull/10567)），针对Fused Triton MoE的Down Projection kernel加入了TMA（Tensor Memory Accelerator）优化。这个优化在DeepSeek-R1模型的H200 TP8 prefill场景下，将第二个MOE（down projection）的计算利用率从**45.20%提升到了81.12%**，单个MOE的延迟从2.430ms降低到了1.435ms，**性能提升约41%**。更关键的是，这个优化在8K tokens场景下让整体TTFT降低了约8-9%。

这里详细分析这个优化的背景、原理和实现细节。

## 0x1. 问题背景

### 0x1.1 不合理的性能表现

在H200(96GB) TP8 prefill性能分析的时候，作者发现了一个很奇怪的现象：每一层的第二个MOE（down projection）的延迟竟然和第一个MOE（up projection）相当，但是第二个MOE的权重数据量和计算量都只有第一个MOE的一半。这个性能表现明显不合理。

我们知道DeepSeek-V2/V3/R1这类模型采用的是MoE架构，每一层有两个MOE操作：
- 第一个MOE（up projection）：将hidden_states从较小的维度映射到较大的中间维度
- 第二个MOE（down projection）：将中间结果从较大的维度映射回原来的维度

理论上，down projection的计算量更小，应该更快才对。但实际profile下来发现延迟差不多，说明第二个MOE的计算利用率存在严重问题。

### 0x1.2 性能瓶颈分析

通过Nsight Compute分析发现，第二个MOE的计算利用率只有45.20%，远低于第一个MOE的利用率。问题主要出在：

![](https://files.mdnice.com/user/59/10eb178d-ff89-4dd2-9013-094878293674.png)

1. **内存访问模式不优**：第二个MOE的输入数据访问模式比较分散，没有充分利用GPU的内存带宽
2. **配置参数不匹配**：两个MOE使用了相同的kernel配置，但它们的计算特征完全不同
3. **Expert负载不均**：实际推理中expert的负载分布是不均匀的，但tuning时用的是随机生成的expert分布



## 0x2. 优化思路

针对上面发现的问题，作者提出了一个比较全面的优化方案，主要包含四个方面：

### 0x2.1 优化 b_scale 的获取和计算

在使用FP8量化的时候，需要读取权重的scale因子（b_scale）。原来的实现中，每次读取`BLOCK_SIZE_N`个b_scale元素。但实际上，当`BLOCK_SIZE_N`小于等于`group_n`（block量化参数）时，每次只需要读取一个b_scale元素就够了。

优化后的逻辑是：可以先把a_scale（输入量化scale）和b_scale相乘，然后再乘以dot product的结果，这样可以减少重复的scale计算开销。

### 0x2.2 基于TMA重构第二个MOE

这是整个优化的核心。TMA（Tensor Memory Accelerator）是NVIDIA在Hopper架构引入的硬件加速单元，专门用于加速GPU和HBM之间的数据传输。

原来的实现中，输入A的访问模式是按`(num_tokens * top_k)`这个维度来组织的，访问比较分散。优化后改成按`(num_blocks * block_size_m)`来组织，让数据访问变得连续，然后用TMA来封装输入A和权重B的访问过程。

具体来说：
- 将输入A重新组织，让同一个block内的token连续存放
- 使用TMA descriptor来描述输入A和权重B的内存布局
- 利用TMA硬件单元完成高效的数据搬运

这样做的好处是显著提升内存带宽利用率。从profile结果看，使用TMA后内存访问效率大幅提升。

### 0x2.3 使用真实Expert分布进行Tuning

原来的tuning脚本用的是随机生成的expert分布，但实际推理中expert的负载是不均匀的。比如某些expert可能会被更频繁地选中，而另一些expert几乎不会被用到。

优化方案是在真实推理过程中收集topk_ids（哪些token选择了哪些expert），然后用这些真实数据来做tuning。具体流程是：

1. 修改`srt/models/deepseek_v2.py`，添加保存topk_ids的逻辑：
```python
# import get_tensor_model_parallel_rank
# DeepseekV2MoE::forward_normal
if hidden_states.shape[0] == 16384 and get_tensor_model_parallel_rank() == 0:
    topk_ids_dir = xxxx
    if not hasattr(self, "save_idx"):
        self.save_idx = 0
    if self.save_idx <= 1:
        torch.save(topk_output.topk_ids, 
                  f"{topk_ids_dir}/topk_idx_layer{self.layer_id}_idx{self.save_idx}.pt")
    self.save_idx += 1
```

2. 设置chunked prefix size为16384，发送一个较长的请求

3. 停止服务器，使用收集到的topk_ids进行tuning：
```bash
model_path=/home/deepseek-ai__DeepSeek-R1
topk_ids_dir=xxxxx

python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py \
    --model $model_path \
    --tp-size 8 \
    --dtype fp8_w8a8 \
    --topk-ids-dir ${topk_ids_dir} \
    --tune
```

这样tuning出来的配置会生成两个文件：一个给up projection用，一个给down projection用（文件名后缀带`_down`）。

### 0x2.4 为两个MOE独立加载配置

既然两个MOE的计算特征不同，那就应该用不同的kernel配置。优化后的代码会分别为up projection和down projection加载最优配置。

关键代码在`fused_moe_triton_config.py`的`try_get_optimal_moe_config`函数中：

```python
def try_get_optimal_moe_config(
    w1_shape: Tuple[int, ...],
    w2_shape: Tuple[int, ...],
    ...
    return_down_config: bool = False,
):
    ...
    if return_down_config:
        down_configs = get_moe_configs(E, N, dtype, block_n, block_k, down_moe=True)
        if down_configs:
            down_config = down_configs[
                min(down_configs.keys(), key=lambda x: abs(x - M))
            ]
    ...
```

在实际执行时，会根据`down_moe_use_tma`这个标志来决定是否启用TMA优化：

```python
down_moe_use_tma = (
    _down_moe_use_tma()
    and down_config is not None
    and down_config.pop("USE_TMA", False)
)
```

## 0x3. 代码实现细节

### 0x3.1 TMA支持检测

首先需要检测当前环境是否支持TMA：

```python
try:
    from triton.tools.tensor_descriptor import TensorDescriptor
    _support_tensor_descriptor = True
except:
    _support_tensor_descriptor = False

def support_tensor_descriptor():
    return _support_tensor_descriptor

@functools.lru_cache()
def _down_moe_use_tma():
    return support_tensor_descriptor()
```

### 0x3.2 输入数据重组

使用TMA的关键是要让数据访问连续。在`fused_experts_impl`函数中，会根据是否使用TMA来决定是否需要额外的padding：

```python
max_padded_tokens = (
    min(M * topk, E + 1) * (max_block_m - 1) if down_moe_use_tma else 0
)
total_tokens = M * topk + max_padded_tokens
```

这个padding是为了让每个block的token数量对齐，方便TMA访问。计算完成后，会创建合适大小的intermediate cache：

```python
cache = torch.empty(
    total_tokens * max(N, w2.shape[1]),
    device=hidden_states.device,
    dtype=hidden_states.dtype,
)
```

### 0x3.3 Kernel调用

在调用第二个MOE kernel时，会传入TMA相关的参数：

```python
invoke_fused_moe_kernel(
    intermediate_cache2,
    w2,
    b2,
    intermediate_cache3,
    ...
    a_use_tma=down_moe_use_tma,  # 输入A使用TMA
    b_use_tma=down_moe_use_tma,  # 权重B使用TMA
    filter_expert=filter_expert,
)
```

同时在第一个MOE的kernel调用中，会设置`c_sorted=down_moe_use_tma`，这样第一个MOE的输出会按照第二个MOE需要的顺序来组织，方便后续使用TMA访问。

## 0x4. 性能测试结果

### 0x4.1 Kernel级别的性能提升

在16372 chunked size场景下，发送一个长度为16372×3 + 502的请求：

**优化前**：
- fused_moe_kernel总耗时：1.525s
- 平均每次调用：4.383ms
- 调用次数：348次（两个MOE共用同一个kernel）

**优化后**：
- fused_moe_kernel（up projection）总耗时：706.7ms，平均4.061ms，调用174次
- fused_moe_down_tma_kernel（down projection）总耗时：430.0ms，平均2.471ms，调用174次

可以看到，down projection的kernel平均延迟从4.383ms降到了2.471ms，性能提升约**43.6%**。而且两个kernel加起来的总耗时（1136.7ms）也比优化前（1525ms）少了约25%。

更重要的是，通过Nsight Compute看到down projection的计算利用率从**45.20%提升到了81.12%**，说明优化确实让硬件资源得到了更充分的利用。

### 0x4.2 端到端TTFT测试

在不同输入长度下的TTFT测试显示，优化后在各个长度上都有8-9%的TTFT降低：

| Input Lens | Before TTFT | After TTFT | TTFT降低 |
|------------|-------------|------------|---------|
| 512        | 115.78ms    | 112.86ms   | 2.52%   |
| 1024       | 138.63ms    | 127.31ms   | 8.17%   |
| 2048       | 219.31ms    | 200.50ms   | 8.58%   |
| 4096       | 393.37ms    | 358.95ms   | 8.75%   |
| 8192       | 801.73ms    | 738.52ms   | 7.88%   |

可以看到，在较长的输入长度下（2K-8K tokens），TTFT的降低比例稳定在8%左右，这对于用户体验的提升是非常明显的。

### 0x4.3 精度验证

优化不能影响模型精度。PR中提供了GSM8K和MMLU的测试结果，优化前后精度基本一致：

**GSM8K**：
- 优化前：Accuracy 0.951
- 优化后：Accuracy 0.953

**MMLU**：
- 优化前后平均准确率都是0.871左右

说明TMA优化没有引入精度损失。

## 0x5. 社区讨论和进一步思考

### 0x5.1 为什么只优化down_proj？

PR的comment里有人问到，为什么TMA优化只应用到down_proj，up_proj是否也能获得类似的加速？

作者xu-yfei的回答是：up_proj的权重虽然可以使用TMA，但本地测试显示提升不明显（1-2%左右）。因为up_proj的输入A是`hidden_states`，它的值是按token离散获取的，无法像down_proj那样重组成连续的块来使用TMA。

而down_proj的输入A是上一个MOE的输出，这个输出可以在计算时就按照连续块的方式来组织，所以能充分发挥TMA的优势。

### 0x5.2 TMA会增加kernel耗时吗？

有人在测试中发现，使用TMA后单个kernel的耗时反而增加了。作者解释说这是正常现象，因为：

1. 使用TMA后的最优配置和不使用TMA的最优配置是不同的
2. 虽然单个kernel耗时可能增加，但end-to-end的吞吐量会提升

作者还提供了一个配置对比表，展示了在H200上使用不同配置时的性能：

| configs | tokens | gateup proj (us) | gateup proj with TMA (us) | down proj (us) | down proj with TMA (us) |
|---------|--------|-----------------|--------------------------|----------------|------------------------|
| "64 128 128 1 4 3" | 8192 | 2311 | 2290 | 2164 | 1456 |
| "64 128 128 32 4 3" | 8192 | 2311 | 2292 | 2204 | 1391 |
| "64 128 128 16 4 3" | 8192 | 2309 | 2289 | 2172 | 1377 |

可以看到down projection使用TMA后性能提升非常明显（从2164us降到1456us，提升约33%）。

### 0x5.3 为什么不合并kernel？

另一个有意思的讨论是，是否可以把两个MOE的kernel合并起来？作者尝试过合并，但发现性能会显著下降。原因是合并后引入了一些分支判断，导致性能退化。经过一些优化后性能有所恢复，但仍然不如分开的版本。

这个也提醒我们，kernel fusion并不总是能带来性能提升，有时候分开执行反而更好，尤其是两个操作的计算特征差异较大的时候。

### 0x5.4 其他模型是否受益？

有人问到Qwen3-MoE是否也能从这个优化中受益。作者表示Qwen3-MoE也使用相同的MoE operator，理论上也能获得类似的加速效果。但建议在大token长度场景（如8K）和大TP规模（如TP8）下测试，因为这种场景下MOE的计算占比更高，优化效果会更明显。

## 0x6. 总结

这个PR展示了一个完整的性能优化过程：

1. **发现问题**：通过profiling发现down projection的计算利用率低，延迟不合理
2. **分析问题**：找出内存访问模式、配置参数、expert分布等多个影响因素
3. **提出方案**：针对性地使用TMA优化、调整配置、使用真实数据tuning
4. **验证效果**：从kernel级别到端到端，全面验证性能和精度

最终实现了计算利用率从45.20%到81.12%的提升，单kernel延迟降低41%，端到端TTFT降低8-9%。


## 0x7. 参考资料

- PR链接：https://github.com/sgl-project/sglang/pull/10567
- Triton TMA文档：https://triton-lang.org/main/programming-guide/chapter-2/index.html
- SGLang文档：https://docs.sglang.ai
