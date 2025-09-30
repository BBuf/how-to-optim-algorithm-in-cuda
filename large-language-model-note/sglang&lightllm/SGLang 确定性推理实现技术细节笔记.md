# 0x0. 前言

最近基于SGLang的确定性推理feature (Deterministic Inference)支持(https://lmsys.org/blog/2025-09-22-sglang-deterministic/) 补了一下这个知识。这个是Thinking Machines最近提出的攻克大语言模型推理中的不确定性在SGLang上的工程实现。要查看Thinking Machines的原文请查看：https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ 或者 它的中文版本：https://zhuanlan.zhihu.com/p/1949285893413278978

确定性推理的核心挑战在于深度学习推理过程中存在多个随机性来源：注意力机制中的随机数生成、采样过程的随机性、以及不同batch size下的计算顺序差异等。SGLang通过引入batch invariant ops、固定随机数种子、以及特殊的attention backend配置等技术手段，成功实现了确定性推理。

这篇文章记录一下SGLang实现确定性推理流程的相关技术细节。

# 0x1. SGLang确定性推理启用

SGLang通过`--enable-deterministic-inference`参数启用确定性推理模式：

```bash
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --attention-backend flashinfer \
    --enable-deterministic-inference
```

在`server_args.py`中定义了这个参数：

```python
# 来源：sglang/python/sglang/srt/server_args.py
parser.add_argument(
    "--enable-deterministic-inference",
    action="store_true",
    help="Enable deterministic inference mode with batch invariant ops.",
)
```

整个确定性推理主要包含几个部分：Batch Invariant Ops（保证不同batch size下结果一致）、确定性采样（用固定种子生成可重现的随机序列）、FlashInfer和Triton Attention Backend的特殊配置，还有一堆环境变量来控制各个模块的行为。

# 0x2. 批次不变性原理与Batch Invariant Ops

## 0x2.1 批次不变性介绍

批次不变性(Batch Invariance)指模型在处理不同批次大小的输入时，对于相同的输入数据能够产生完全一致的输出结果。考虑一个简单的场景：我们有一个输入样本x，分别用batch_size=1和batch_size=4（其中4个样本都是x）来推理，理论上第一个输出应该完全相同。但实际情况往往不是这样：

```python
import torch

# 单样本推理
x = torch.randn(1, 128, device='cuda')
output_single = model(x)

# 批次推理（4个相同样本）
x_batch = x.repeat(4, 1)
output_batch = model(x_batch)

# 检查第一个输出是否相同
print(torch.allclose(output_single, output_batch[0:1]))  # 经常是False!
```

这种不一致性的根本原因在于GPU的并行计算特性。不同的batch size可能会触发不同的并行化策略、内存访问模式和数值计算顺序，从而导致浮点运算的累积误差不同。

## 0x2.2 实现批次不变性算子

根据 https://link.zhihu.com/?target=https%3A//thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ ，实现批次不变性算子需要考虑几点：

- 矩阵乘法的并行化差异：GPU上的矩阵乘法会根据输入大小选择不同的tile分解和线程块调度策略。batch_size=1时可能使用一种tile大小，batch_size=64时可能使用另一种，导致浮点运算顺序不同。
- 归约操作的顺序敏感性：像sum、mean这样的归约操作在GPU上是并行执行的，不同的并行度会导致不同的累加顺序，而浮点数的加法不满足结合律，因此结果会有差异。
- 内存访问模式的影响：不同batch size下的内存布局和访问模式不同，可能触发不同的缓存行为和内存合并模式，间接影响计算结果。
- 算子实现的形状依赖：某些算子的实现会根据输入形状选择不同的代码路径，比如小batch用一种实现，大batch用另一种实现。

## 0x2.3 SGLang 实现 Batch Invariant Ops

SGLang首先引入了来自`thinking-machines-lab/batch_invariant_ops`项目的Batch Invariant算子来解决这些问题。然后在Attention Backend和Sampler方面都做了确定性kernel的支持以达到整体的推理流程确定。

### Matmul Persistent Kernel的设计思路

传统的GPU kernel通常是"一个线程块处理一个输出tile"的模式，这种模式下不同的输入大小会启动不同数量的线程块，导致调度顺序的不确定性。

Persistent kernel采用了"固定数量的线程块，每个处理多个tile"的模式：

```python
# 传统kernel启动方式
def traditional_launch(M, N, BLOCK_M, BLOCK_N):
    num_blocks_m = (M + BLOCK_M - 1) // BLOCK_M
    num_blocks_n = (N + BLOCK_N - 1) // BLOCK_N
    return (num_blocks_m, num_blocks_n)  # 块数量随输入大小变化

# Persistent kernel启动方式  
def persistent_launch(M, N, BLOCK_M, BLOCK_N, NUM_SMS):
    total_tiles = ((M + BLOCK_M - 1) // BLOCK_M) * ((N + BLOCK_N - 1) // BLOCK_N)
    return (min(NUM_SMS, total_tiles),)  # 固定使用NUM_SMS个块
```

核心思想就是让每个SM处理的工作量固定，用软件调度代替硬件调度，这样计算顺序就确定了。

### 确定性的Tile调度算法

在persistent kernel里面，还需要一个确定性算法来决定每个线程块处理哪些tile。SGLang用的是基于tile ID的确定性映射：

```python
@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    # 计算当前tile属于哪个group
    group_id = tile_id // num_pid_in_group
    
    # 计算该group在M维度上的起始位置
    first_pid_m = group_id * GROUP_SIZE_M
    
    # 计算该group的实际大小（处理边界情况）
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    
    # 在group内的确定性映射
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    
    return pid_m, pid_n
```

这个算法保证相同的tile_id总是映射到相同的(pid_m, pid_n)，不管总tile数量怎么变。

### 其它确定性kernel的算子实现

除了Matmul算子调度要确定，算子本身也得是Batch Invariant的。拿RMSNorm举个例子：

```python
@triton.jit
def rms_norm_kernel(input_ptr, weight_ptr, output_ptr, eps, n_cols, BLOCK_SIZE: tl.constexpr):
    # 每行独立处理，避免batch间相互影响
    row_idx = tl.program_id(0)
    
    # 使用固定精度的计算顺序
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # 加载数据并转换为float32确保精度
    x = tl.load(input_ptr + row_idx * n_cols + col_offsets, mask=mask, other=0.0)
    x = x.to(tl.float32)
    
    # 固定的数值稳定计算
    x_squared = x * x
    mean_x_squared = tl.sum(x_squared, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(mean_x_squared + eps)
    
    # 归一化和缩放
    normalized = x * rstd
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
    output = normalized * weight
    
    # 存储结果
    tl.store(output_ptr + row_idx * n_cols + col_offsets, output, mask=mask)
```

关键是用固定的计算顺序和精度，避免不同并行度导致的数值差异。

### Batch Invariant Ops的引入和启用

SGLang直接用了`thinking-machines-lab/batch_invariant_ops`项目的算子。在`model_runner.py`里，开启确定性推理时会导入并启用batch invariant模式：

```python
# 来源：sglang/python/sglang/srt/model_executor/model_runner.py
if server_args.enable_deterministic_inference:
    from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode
    enable_batch_invariant_mode()
```

### Batch Invariant Ops 算子实现解析（补充）

#### MatMul Persistent Kernel 解析

首先，PID(Program ID)计算函数确保tile的调度顺序是确定的：

```python
# 来源：sglang/python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py
@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    """
    计算确定性的Program ID，确保tile调度顺序固定
    
    Args:
        tile_id: 当前tile的全局ID
        num_pid_in_group: 每个group中的PID数量
        num_pid_m: M维度上的PID数量
        GROUP_SIZE_M: M维度上的group大小
        NUM_SMS: SM数量
    """
    # 计算当前tile属于哪个group
    group_id = tile_id // num_pid_in_group
    
    # 计算该group在M维度上的起始位置
    first_pid_m = group_id * GROUP_SIZE_M
    
    # 计算该group的实际大小（处理边界情况）
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    
    # 计算在M维度上的PID
    pid_m = first_pid_m + (tile_id % group_size_m)
    
    # 计算在N维度上的PID
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    
    return pid_m, pid_n
```

这个函数的关键在于确定性分组，通过`group_id = tile_id // num_pid_in_group`确保tile分组是确定的。M和N维度的PID计算都基于tile_id，保证相同的tile_id总是映射到相同的(pid_m, pid_n)。边界处理使用`min(num_pid_m - first_pid_m, GROUP_SIZE_M)`确保边界group的处理一致。

```python
# 来源：sglang/python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py
@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_persistent(
    a_ptr, b_ptr, c_ptr, bias_ptr,  # 输入输出指针
    M, N, K,  # 矩阵维度
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,  # 步长
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, NUM_SMS: tl.constexpr,
    A_LARGE: tl.constexpr, B_LARGE: tl.constexpr, C_LARGE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    # 获取当前SM的起始PID
    start_pid = tl.program_id(axis=0)
    
    # 计算tile数量和分布
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)  # M维度上的tile数量
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)  # N维度上的tile数量
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)    # K维度上的tile数量
    num_tiles = num_pid_m * num_pid_n     # 总tile数量
    
    # 用于输出阶段的tile ID计算
    tile_id_c = start_pid - NUM_SMS
    
    # 预计算一些常用的偏移量
    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    
    # Persistent循环：每个SM处理多个tile
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        # 计算当前tile的PID
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        
        # 计算当前tile在M和N维度上的起始位置
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        
        # 计算A和B矩阵的偏移量
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        
        # 处理大矩阵的索引类型转换
        if A_LARGE:
            offs_am = offs_am.to(tl.int64)
        if B_LARGE:
            offs_bn = offs_bn.to(tl.int64)
        
        # 边界检查和掩码处理
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        
        # 内存访问优化：确保连续访问和对齐
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        
        # 初始化累加器
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        
        # K维度循环：执行矩阵乘法
        for ki in range(k_tiles):
            # 计算K维度的偏移量
            if A_LARGE or B_LARGE:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
            else:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            
            # 计算A和B的内存地址
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
            
            # 加载数据并处理边界
            a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0)
            
            # 执行矩阵乘法累加
            accumulator = tl.dot(a, b, accumulator)
        
        # 输出阶段：计算输出位置
        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        
        # 计算输出矩阵的偏移量
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        
        # 处理大矩阵的输出索引
        if C_LARGE:
            offs_cm = offs_cm.to(tl.int64)
            offs_cn = offs_cn.to(tl.int64)
        
        # 计算输出地址和掩码
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        
        # 处理bias（如果存在）
        if HAS_BIAS:
            bias_ptrs = bias_ptr + offs_cn
            bias = tl.load(bias_ptrs, mask=offs_cn < N, other=0.0).to(tl.float32)
            accumulator += bias
        
        # 类型转换和存储
        if c_ptr.dtype.element_ty == tl.float8e4nv:
            c = accumulator.to(tl.float8e4nv)
        else:
            c = accumulator.to(tl.float16)
        
        # 存储结果
        tl.store(c_ptrs, c, mask=c_mask)
```

#### Persistent MatMul Kernel 配置和启动

```python
# 来源：sglang/python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py
def matmul_persistent(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None):
    # 输入验证
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    assert bias is None or bias.dim() == 1, "Currently assuming bias is 1D"
    
    # 获取硬件信息
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    
    # 分配输出张量
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    
    # 定义grid函数：确保启动的block数不超过SM数量
    def grid(META):
        return (
            min(
                NUM_SMS,
                triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            ),
        )
    
    # 针对不同数据类型的优化配置
    configs = {
        torch.bfloat16: {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8, "num_stages": 3, "num_warps": 8,
        },
        torch.float16: {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8, "num_stages": 3, "num_warps": 8,
        },
        torch.float32: {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8, "num_stages": 3, "num_warps": 8,
        },
    }
    
    # 启动kernel
    matmul_kernel_persistent[grid](
        a, b, c, bias, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        NUM_SMS=NUM_SMS,
        A_LARGE=a.numel() > 2**31, B_LARGE=b.numel() > 2**31, C_LARGE=c.numel() > 2**31,
        HAS_BIAS=bias is not None,
        **configs[dtype],
    )
    return c
```

#### 确定性Log Softmax算子

Log Softmax算子实现了数值稳定且确定性的softmax计算：

```python
# 来源：sglang/python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py
@triton.jit
def _log_softmax_kernel(
    input_ptr, output_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr,
):
    """
    计算2D张量最后一个维度的log_softmax
    每个block处理输入张量的一行
    """
    # 获取当前block处理的行索引
    row_idx = tl.program_id(0).to(tl.int64)
    
    # 计算输入和输出行的基地址
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    
    # 第一步：找到行中的最大值（数值稳定性）
    max_val = -float("inf")
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        
        # 加载值
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=-float("inf"))
        
        # 更新最大值
        max_val = tl.max(tl.maximum(vals, max_val))
    
    # 第二步：计算exp(x - max_val)的和
    sum_exp = 0.0
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        
        # 加载值
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
        
        # 计算exp(x - max_val)并累加
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(tl.where(mask, exp_vals, 0.0))
    
    # 计算log(sum_exp)
    log_sum_exp = tl.log(sum_exp)
    
    # 第三步：计算最终的log_softmax值: x - max_val - log_sum_exp
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        
        # 加载值
        vals = tl.load(row_start_ptr + col_idx, mask=mask)
        
        # 计算log_softmax
        output = vals - max_val - log_sum_exp
        
        # 存储结果
        tl.store(output_row_start_ptr + col_idx, output, mask=mask)
```

#### 确定性Mean算子

Mean算子实现了沿指定维度的均值计算：

```python
# 来源：sglang/python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py
@triton.jit
def mean_kernel(
    input_ptr, output_ptr,
    input_stride0, input_stride1, input_stride2,
    output_stride0, output_stride1,
    M, N, K,  # M: 缩减维度前的大小, N: 缩减维度大小, K: 缩减维度后的大小
    BLOCK_SIZE: tl.constexpr,
):
    """
    沿单个维度计算均值的kernel
    输入被视为(M, N, K)，其中N是被缩减的维度
    """
    # Program ID给出我们正在计算的输出元素
    pid = tl.program_id(0)
    
    # 计算输出索引
    m_idx = pid // K
    k_idx = pid % K
    
    # 边界检查
    if m_idx >= M or k_idx >= K:
        return
    
    # 沿缩减维度累加求和
    acc = 0.0
    for n_start in range(0, N, BLOCK_SIZE):
        n_offsets = n_start + tl.arange(0, BLOCK_SIZE)
        mask = n_offsets < N
        
        # 计算输入索引
        input_idx = (
            m_idx * input_stride0 + n_offsets * input_stride1 + k_idx * input_stride2
        )
        
        # 加载并累加
        vals = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
        acc += tl.sum(vals)
    
    # 计算均值并存储
    mean_val = acc / N
    output_idx = m_idx * output_stride0 + k_idx * output_stride1
    tl.store(output_ptr + output_idx, mean_val)
```

#### 算子注册和替换机制

在SGLang在开启Batch Invariant模式中会把PyTorch的原生算子替换为batch invariant版本，如下所示：

```python
# 来源：sglang/python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py
def enable_batch_invariant_mode():
    """启用batch invariant模式，替换PyTorch原生算子"""
    global _batch_invariant_MODE, _batch_invariant_LIB
    if _batch_invariant_MODE:
        return
    
    _batch_invariant_MODE = True
    
    # 创建PyTorch库实现
    _batch_invariant_LIB = torch.library.Library("aten", "IMPL")
    
    # 注册算子替换
    _batch_invariant_LIB.impl("aten::mm", mm_batch_invariant, "CUDA")
    _batch_invariant_LIB.impl("aten::addmm", addmm_batch_invariant, "CUDA")
    _batch_invariant_LIB.impl("aten::_log_softmax", _log_softmax_batch_invariant, "CUDA")
    _batch_invariant_LIB.impl("aten::mean.dim", mean_batch_invariant, "CUDA")

# 包装函数
def mm_batch_invariant(a, b):
    """矩阵乘法的batch invariant实现"""
    return matmul_persistent(a, b)

def addmm_batch_invariant(bias, a, b):
    """带bias的矩阵乘法的batch invariant实现"""
    return matmul_persistent(a, b, bias=bias)

def _log_softmax_batch_invariant(input, dim, _half_to_float):
    """Log softmax的batch invariant实现"""
    assert not _half_to_float, "not implemented"
    return log_softmax(input, dim=dim)

def mean_batch_invariant(input, dim, keepdim=False, dtype: torch.dtype | None = None):
    """Mean的batch invariant实现"""
    assert dtype is None or dtype == torch.float32, f"unsupported dtype: {dtype}"
    if len(dim) == 1:
        return mean_dim(input, dim[0], keepdim=keepdim)
    else:
        # 多维度缩减的fallback实现
        assert input.dtype in {torch.float16, torch.bfloat16, torch.float32}
        n_elems = 1
        for d in dim:
            n_elems *= input.shape[d]
        return torch.sum(input, dim=dim, keepdim=keepdim, dtype=torch.float32) / n_elems
```

# 0x3. 确定性Sampling机制

## 0x3.1 确定性Sampling的原理

### 普通采样为啥不确定

在文本生成中，采样过程是推理结果不确定的主要原因。传统采样有这些问题：

```python
# 传统采样的不确定性示例
import torch

def traditional_sampling_demo():
    """演示传统采样的不确定性"""
    logits = torch.tensor([2.0, 1.0, 0.5, 3.0])  # 固定的logits
    
    # 多次采样会得到不同结果
    results = []
    for i in range(5):
        probs = torch.softmax(logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)
        results.append(sampled.item())
    
    print(f"Traditional sampling results: {results}")
    # 输出可能是: [3, 0, 3, 2, 3] - 每次都不同
    
traditional_sampling_demo()
```

不确定性主要来源于：系统随机数生成器依赖系统时间或硬件噪声，GPU并行执行时不同线程执行顺序影响随机数序列，还有内存访问模式对随机数生成器状态的影响。

### 确定性采样的原理

SGLang确定性采样的核心思路：给每个token位置分配唯一的确定性种子，用Gumbel-Max采样方法（数学上等价但确定性），通过哈希函数把种子映射成高质量伪随机数。具体来说：

#### Gumbel-Max采样原理

Gumbel-Max采样是确定性采样的数学基础：

```python
# Gumbel-Max采样原理演示
import torch
import math

def gumbel_max_principle():
    """演示Gumbel-Max采样的数学原理"""
    
    # 原始概率分布
    logits = torch.tensor([2.0, 1.0, 0.5, 3.0])
    probs = torch.softmax(logits, dim=-1)
    print(f"Original probabilities: {probs}")
    
    # Gumbel-Max采样过程
    # 1. 生成Gumbel噪声
    uniform = torch.tensor([0.3, 0.7, 0.1, 0.9])  # 确定性的"随机"数
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-9) + 1e-9)
    print(f"Gumbel noise: {gumbel_noise}")
    
    # 2. 添加到log概率
    perturbed_logits = logits + gumbel_noise
    print(f"Perturbed logits: {perturbed_logits}")
    
    # 3. 选择最大值对应的索引
    sampled_idx = torch.argmax(perturbed_logits)
    print(f"Sampled index: {sampled_idx.item()}")
    
    # 验证：多次使用相同的uniform值会得到相同结果
    for i in range(3):
        same_gumbel = -torch.log(-torch.log(uniform + 1e-9) + 1e-9)
        same_result = torch.argmax(logits + same_gumbel)
        print(f"Repeat {i}: {same_result.item()}")

gumbel_max_principle()
```

#### 位置相关种子生成

SGLang使用位置信息生成唯一的种子：

```python
# 位置相关种子生成示例
def position_dependent_seeding():
    """演示位置相关的种子生成"""
    
    base_seed = 42
    sequence_positions = [0, 1, 2, 3, 4]  # token位置
    
    # SGLang的种子生成算法
    for pos in sequence_positions:
        # 使用大质数进行哈希，确保不同位置有不同种子
        step_seed = base_seed * 19349663 ^ pos * 73856093
        
        # 进一步哈希生成最终种子
        final_seed = step_seed * 8589934591 % (2**32)
        
        print(f"Position {pos}: base_seed={base_seed} -> step_seed={step_seed} -> final_seed={final_seed}")
    
    # 验证：相同位置总是产生相同种子
    print("\n验证一致性:")
    for _ in range(3):
        pos = 2
        step_seed = base_seed * 19349663 ^ pos * 73856093
        final_seed = step_seed * 8589934591 % (2**32)
        print(f"Position {pos} (repeat): {final_seed}")

position_dependent_seeding()
```

## 0x3.2 采样种子的传递和实现

根据PR #10687(https://github.com/sgl-project/sglang/pull/10687)，SGLang在请求结构中添加了`sampling_seed`字段，允许用户为每个请求指定确定的随机种子：

```python
# 来源：sglang/python/sglang/srt/sampling/sampling_params.py
class SamplingParams:
    def __init__(
        self,
        # ... 其他参数
        sampling_seed: int = 42,  # 新增的采样种子参数
    ) -> None:
        # ... 其他参数赋值
        self.sampling_seed = sampling_seed
```


在`SamplingBatchInfo`中，采样种子只有在启用确定性推理时才会被处理：

```python
# 来源：sglang/python/sglang/srt/sampling/sampling_batch_info.py
def __init__(self, reqs, vocab_size, device):
    # ... 其他初始化代码
    
    # 检查是否启用确定性推理
    enable_deterministic = global_server_args_dict["enable_deterministic_inference"]
    
    # 只有在确定性模式下才创建采样种子张量
    sampling_seed = (
        torch.tensor(
            [r.sampling_params.sampling_seed for r in reqs],
            dtype=torch.int32,
            device=device,
        )
        if enable_deterministic
        else None
    )
    
    # ... 其他字段初始化
    self.sampling_seed = sampling_seed
```

这种设计确保了采样种子只在需要时才被处理，避免了不必要的内存开销。

## 0x3.3 确定性多项式采样实现

根据PR #10678(https://github.com/sgl-project/sglang/pull/10678)，SGLang实现了支持温度大于0的确定性采样。核心的确定性采样函数`multinomial_with_seed`实现了基于种子的确定性采样：

```python
# 来源：sglang/python/sglang/srt/layers/sampler.py
def multinomial_with_seed(
    inputs: torch.Tensor, seed: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """
    使用唯一随机种子对输入张量进行确定性采样
    
    Args:
        inputs: 形状为(n, m)的浮点张量，表示n个分类分布
        seed: 形状为(n,)的整数张量，包含每行对应的随机种子
        positions: token在序列中的位置，用于生成唯一种子
    
    Returns:
        形状为(n,)的张量，每个元素是从对应分布中采样的索引
    """
    n, m = inputs.shape
    col_indices = torch.arange(m, device=inputs.device).unsqueeze(0)
    
    # 生成位置相关的种子，确保每个位置都有唯一的随机性
    step_seed = seed * 19349663 ^ positions * 73856093
    seed_expanded = step_seed.unsqueeze(-1)
    
    # 使用哈希函数生成伪随机数
    hashed = seed_expanded * 8589934591 ^ col_indices * 479001599
    uniform_samples = (hashed % (2**24)).float() / (2**24)
    
    # 使用Gumbel-Max技巧进行采样
    epsilon = 1e-9
    gumbel_noise = -torch.log(-torch.log(uniform_samples + epsilon) + epsilon)
    log_probs = torch.log(inputs + epsilon)
    perturbed_log_probs = log_probs + gumbel_noise
    
    return torch.argmax(perturbed_log_probs, dim=1, keepdim=True)
```

这个实现的关键技术点：

1. **位置相关种子生成**: `step_seed = seed * 19349663 ^ positions * 73856093`确保每个token位置都有唯一的随机性
2. **哈希函数**: 使用大质数进行哈希，生成高质量的伪随机数
3. **Gumbel-Max采样**: 通过Gumbel噪声实现确定性的多项式采样

## 0x3.4 Sampler更上层接口的变动

在`Sampler`类中，当检测到采样种子时会使用确定性采样。根据实际代码实现，确定性采样的调用位置在`sample_token`函数中：

```python
# 来源：sglang/python/sglang/srt/layers/sampler.py
def sample_token(
    probs: torch.Tensor,
    sampling_seed: Optional[torch.Tensor],
    positions: torch.Tensor,
):
    """
    Token采样函数
    
    当sampling_seed不为None时，启用确定性推理，使用每个请求的采样种子进行采样。
    这是PR #10678实现的核心功能，支持温度大于0时的确定性采样。
    """
    if sampling_seed is not None:
        # 使用确定性采样
        sampled_index = multinomial_with_seed(probs, sampling_seed, positions)
    else:
        # 使用传统的随机采样
        sampled_index = sampling_from_probs_torch(probs)
    
    return sampled_index
```

### Sampler的forward方法集成

在`Sampler`的`forward`方法中，确定性采样被集成到整个采样流程中：

```python
# 来源：sglang/python/sglang/srt/layers/sampler.py
def forward(self, logits_output, sampling_info, return_logprob, top_logprobs_nums, 
            token_ids_logprobs, positions):
    # ... 预处理logits
    
    if not sampling_info.is_all_greedy:
        # ... 处理温度、top_p等参数
        
        # 关键：调用确定性采样
        batch_next_token_ids = sample_token(
            probs_sort,
            sampling_info.sampling_seed,  # 传递采样种子
            positions,
        )
    
    # ... 后续处理
```

这种设计确保了确定性采样能够无缝集成到SGLang的采样流程中，同时保持向后兼容性。

# 0x4. Attention Backend的确定性配置

建议先看看 https://zhuanlan.zhihu.com/p/1949285893413278978 中关于批次不变Attention的原理介绍。

## 0x4.1 FlashInfer Attention Backend的确定性支持

FlashInfer Attention Backend的确定性支持基于PR #10645(https://github.com/sgl-project/sglang/pull/10645)和FlashInfer项目的PR #1675(https://github.com/flashinfer-ai/flashinfer/pull/1675)。其核心实现通过固定split tile size来确保batch invariant的attention计算：

```python
# 来源：sglang/python/sglang/srt/layers/attention/flashinfer_backend.py
def __init__(self, model_runner):
    # ... 其他初始化代码
    
    # 确定性推理配置
    # 当启用确定性推理时，tensor cores应该用于decode阶段
    # 同时从环境变量设置prefill和decode的split tile sizes，并禁用cuda graph的kv split
    # 更多信息：https://github.com/flashinfer-ai/flashinfer/pull/1675
    self.enable_deterministic = (
        model_runner.server_args.enable_deterministic_inference
    )
    self.prefill_split_tile_size = None
    self.decode_split_tile_size = None
    self.disable_cuda_graph_kv_split = False
    
    if self.enable_deterministic:
        # 强制在decode阶段使用tensor cores
        self.decode_use_tensor_cores = True
        
        # 设置固定的split tile size以确保确定性
        self.prefill_split_tile_size = get_int_env_var(
            "SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE", 4096
        )
        self.decode_split_tile_size = get_int_env_var(
            "SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE", 2048
        )
        
        # 禁用CUDA Graph的KV split以确保确定性
        self.disable_cuda_graph_kv_split = True
        
        # 增大workspace size到2GB以支持更大的split
        global_config.flashinfer_workspace_size = 2048 * 1024 * 1024
```

FlashInfer用的是固定split size的FA2 kernel。传统FlashAttention会根据输入大小动态选择split策略，不同batch size下计算顺序就不一样了。FlashInfer的解决方案：

1. **固定Split Tile Size**: 通过环境变量预设prefill和decode阶段的split tile大小
2. **Batch Invariant Kernel**: 使用专门设计的batch invariant FA2 kernel
3. **禁用动态优化**: 禁用可能导致不确定性的动态KV split优化

SGLang中FlashInfer Attention Backend确定性模式的关键配置包括：

- **SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE**: Prefill阶段的split tile大小，默认4096
- **SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE**: Decode阶段的split tile大小，默认2048  
- **decode_use_tensor_cores**: 强制在decode阶段使用tensor cores
- **disable_cuda_graph_kv_split**: 禁用CUDA Graph模式下的KV split优化
- **workspace_size**: 增大到2GB以支持更大的split操作

## 0x4.2 Triton Attention Backend的确定性支持以及确定性kernel推理如何测试？

Triton Attention Backend也有相应的确定性配置：

```python
# 来源：sglang/python/sglang/srt/layers/attention/triton_backend.py
def __init__(self, model_runner):
    # ... 其他初始化代码
    
    self.enable_deterministic = (
        model_runner.server_args.enable_deterministic_inference
    )
    
    if self.enable_deterministic:
        # 从环境变量获取split tile size
        self.split_tile_size = get_int_env_var(
            "SGLANG_TRITON_ATTENTION_SPLIT_TILE_SIZE", None
        )
    else:
        # 使用服务器参数中的配置
        self.split_tile_size = (
            model_runner.server_args.triton_attention_split_tile_size
        )
    
    # 计算split数量
    if self.split_tile_size is not None:
        self.num_splits = (
            self.max_context_len + self.split_tile_size - 1
        ) // self.split_tile_size
```

## 0x4.3 FlashAttention3 Attention Backend的确定性支持

FlashAttention3 Attention Backend (FA3) 通过控制split数量来实现确定性推理。根据PR #10651(https://github.com/sgl-project/sglang/pull/10651)的实现，FA3的确定性支持主要通过以下方式：

```python
# 来源：sglang/python/sglang/srt/layers/attention/flashattention_backend.py
def __init__(self, model_runner):
    # ... 其他初始化代码
    
    # 确定性推理配置：控制split数量
    # 当启用确定性推理时，num_splits设为1，否则设为0（自动选择）
    # 参考：https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
    self.num_splits = (
        1 if model_runner.server_args.enable_deterministic_inference else 0
    )
```

FA3确定性推理的核心原理是固定split策略。在非确定性模式下，FlashAttention3可能会根据输入大小自动选择不同的split数量来优化性能，但这会导致不同batch size下的计算顺序不同。通过将`num_splits`固定为1，确保了计算的一致性。

根据PR #10651的测试结果，FA3在确定性模式下通过了所有测试：

```bash
# Prefix模式测试 - 测试不同前缀长度的确定性
python3 -m sglang.test.test_deterministic --test-mode prefix --n-trials 50
# 结果：
# Prompt 0 with prefix length 1: total samples: 312, Unique samples: 1
# Prompt 1 with prefix length 511: total samples: 334, Unique samples: 1
# Prompt 2 with prefix length 2048: total samples: 326, Unique samples: 1
# Prompt 3 with prefix length 4097: total samples: 303, Unique samples: 1

# Mixed模式测试 - 测试混合长度prompt的确定性
python3 -m sglang.test.test_deterministic --test-mode mixed --n-trials 50
# 结果：
# Prompt 1: total samples: 530, Unique samples: 1
# Prompt 2: total samples: 530, Unique samples: 1
# Long prompt: total samples: 215, Unique samples: 1
```

所有测试的`Unique samples`都是1，说明确实做到了确定性。

此外，FA3的确定性实现还支持Radix Cache，测试结果显示：

```bash
# 启用Radix Cache的prefix模式测试
# 结果：
# Prompt 0 with prefix length 1: total samples: 315, Unique samples: 1
# Prompt 1 with prefix length 511: total samples: 299, Unique samples: 1
# Prompt 2 with prefix length 1728: total samples: 302, Unique samples: 1
# Prompt 3 with prefix length 2345: total samples: 359, Unique samples: 1
```

说明FA3的确定性实现和SGLang的缓存机制兼容。

# 0x5. 确定性推理中的AllReduce改动

在多GPU分布式推理中，SGLang对AllReduce操作也做了特殊处理来确保通信的确定性。传统AllReduce可能因为通信顺序随机、浮点累加顺序不同、算法动态选择等原因导致结果不确定。

SGLang的解决方案很直接：强制使用NCCL的tree算法，同时禁用自定义AllReduce实现。

```python
# 来源：sglang/python/sglang/srt/server_args.py
def _handle_deterministic_inference(self):
    if self.enable_deterministic_inference:
        if self.tp_size > 1:
            # 强制使用tree算法
            os.environ["NCCL_ALGO"] = "allreduce:tree"
            # 禁用自定义AllReduce实现
            self.disable_custom_all_reduce = True
```

PR见：https://github.com/sgl-project/sglang/pull/10930

# 0x6. SGLang 确定性推理环境变量以及测试脚本

SGLang用了一堆环境变量来控制确定性推理：


```bash
# FlashInfer prefill阶段的split tile size，默认4096
export SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE=4096

# FlashInfer decode阶段的split tile size，默认2048
export SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE=2048
```


```bash
# Triton attention的split tile size
export SGLANG_TRITON_ATTENTION_SPLIT_TILE_SIZE=2048
```


在`server_args.py`中，确定性推理会自动设置相关环境变量：

```python
# 来源：sglang/python/sglang/srt/server_args.py
def prepare_server_env(self):
    # 设置确定性推理环境变量
    os.environ["SGLANG_DETERMINISTIC_INFERENCE"] = (
        "1" if self.enable_deterministic_inference else "0"
    )
    
    # 其他环境变量设置
    if self.enable_deterministic_inference:
        # 可能需要设置其他相关的环境变量
        pass
```


SGLang提供了专门的测试脚本`test_deterministic.py`来验证确定性推理的效果：

```python
# 来源：sglang/python/sglang/test/test_deterministic.py
def test_deterministic(args):
    # 预热阶段
    for i in range(3):
        send_single(args, 16, args.profile)
    
    if args.test_mode == "single":
        # 单一模式：使用相同prompt测试不同batch size
        texts = []
        for i in range(1, args.n_trials + 1):
            batch_size = i
            text = send_single(args, batch_size, args.profile)
            texts.append(text.replace("\n", " "))
        
        print(f"Total samples: {len(texts)}, Unique samples: {len(set(texts))}")
    
    elif args.test_mode == "mixed":
        # 混合模式：在同一batch中混合不同长度的prompt
        # ... 混合测试逻辑
        
    elif args.test_mode == "prefix":
        # 前缀模式：测试不同长度的公共前缀
        # ... 前缀测试逻辑
```

- Single模式: 使用相同prompt测试不同batch size下的一致性
- Mixed模式: 在同一batch中混合不同长度的prompt
- Prefix模式: 测试具有不同长度公共前缀的prompt


理想的确定性推理测试结果应该是：

```bash
# Single模式测试结果
Total samples: 50, Unique samples: 1

# Mixed模式测试结果  
Prompt 1: total samples: 459, Unique samples: 1
Prompt 2: total samples: 600, Unique samples: 1
Long prompt: total samples: 216, Unique samples: 1

# Prefix模式测试结果
Prompt 0 with prefix length 1: total samples: 314, Unique samples: 1
Prompt 1 with prefix length 511: total samples: 297, Unique samples: 1
Prompt 2 with prefix length 2048: total samples: 340, Unique samples: 1
Prompt 3 with prefix length 4097: total samples: 324, Unique samples: 1
```

所有测试的`Unique samples`都应该是1。

# 0x6. 使用确定性kernel推理的建议和最佳实践

## 0x6.1 启用确定性kernel推理

```bash
# 基本启用方式
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --attention-backend flashinfer \
    --enable-deterministic-inference

# 自定义split tile size
export SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE=4096
export SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE=2048
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --attention-backend flashinfer \
    --enable-deterministic-inference
```

## 0x6.2 客户端请求示例

```python
import requests

# 发送确定性推理请求
response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "Tell me about machine learning",
        "sampling_params": {
            "temperature": 0.7,
            "max_new_tokens": 100,
            "sampling_seed": 42  # 指定确定的种子
        }
    }
)
```

# 0x8. 总结

这篇笔记分析了SGLang确定性推理的技术实现。SGLang主要通过五个方面解决了深度学习推理的随机性问题：

1. **Batch Invariant Ops** - 保证不同batch size下计算结果一致
2. **确定性采样** - 基于Gumbel-Max和位置相关种子生成
3. **Attention Backend确定性配置** - 支持FlashInfer、Triton、FA3等多种backend
4. **AllReduce确定性改动** - 在分布式场景下确保通信的确定性
5. **环境变量控制** - 通过各种环境变量精细调控

整体来说，这套方案还是挺完整的，基本覆盖了推理流程中可能产生随机性的各个环节。https://lmsys.org/blog/2025-09-22-sglang-deterministic/ 官方的Blog中将这套确定性Kernel实现和Slime结合获得了一个100%可复现的稳定RL训练框架流程。








