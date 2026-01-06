> 博客转载翻译自：https://substack.com/inbox/post/183346100?triedRedirect=true ，仅做知识学习用处，侵删

# 142 TFLOPS 的差距：为什么在 **Blackwell** 上 **FP4 MoE** 算子工程至关重要

如何通过 **kernel 融合**、面向 **Blackwell** 的优化与**专家感知（expert-aware）**计算，在小 batch 推理上相对 `vLLM` 获得 **1.84×** 加速

作者：advpropx（X）

## 引言

当 NVIDIA 宣布 **Blackwell** 原生支持 **FP4** 时，承诺非常明确：**2×** 的显存带宽节省，以及面向大模型推理的显著吞吐提升。但硬件能力只是故事的一半；另一半来自：**kernel 工程**。

我在一张 **Blackwell B200**（`sm_100a`）GPU 上，对三种主流 **MoE**（**Mixture of Experts**）后端做了基准测试：`vLLM`、`SGLang`、`FlashInfer CuteDSL`。测试模型为 `GPT-OSS-20B`，每层 **32** 个专家、`top-4` 路由，使用 **FP4** 量化。在相同硬件、相同模型结构下，仅 kernel 实现不同。

结果如下：

- **SGLang**：峰值吞吐 **1168 TFLOPS**
- **vLLM**：峰值吞吐 **1026 TFLOPS**
- **FlashInfer CuteDSL**：峰值吞吐 **1156 TFLOPS**

这意味着 `SGLang` 与 `vLLM` 之间存在 **142 TFLOPS** 的差距。在 batch size = 1（交互式推理最常见的场景）时，`SGLang` **快 1.84×**。

这不是分布式训练，也不是多机推理的对比，而是**单卡推理**下的 **grouped GEMM** kernel 之争。差异主要来自三项关键优化：

- **Kernel 融合**（全局内存来回 3 次 → 1 次）
- 面向 **Blackwell** 的 **CUTLASS** schedule（原生 **FP4 warp specialization**）
- **自适应 grid sizing**（在小 batch 下最大化 **SM occupancy**）

下面先看数据，再拆解这些 kernel 到底差在哪里。

## 基准测试：Blackwell B200 上的 `GPT-OSS-20B`

### 模型配置

```shell
Architecture: GPT-OSS-20B
Experts: 32 total, top-4 routing
Hidden size: 2880
Intermediate size: 7680 (per expert)
Quantization: NVFP4 (4-bit floating point, E2M1 format)
Hardware: NVIDIA Blackwell B200 (sm_100a)
```

### 峰值吞吐（Batch Size = 4096）

图 1：不同 batch size 下的有效 TFLOPS。`SGLang` 始终领先，并且在小 batch 下优势更明显。

![](https://files.mdnice.com/user/59/0e3575b0-d737-47cf-b6bb-db445ffea1ca.png)

![top k = 2](https://files.mdnice.com/user/59/77e5bac8-caf8-4475-b5e3-0b0cb460b2f3.png)

### 时延拆解（Batch Size = 128）

图 2：batch size = 128（decode 的“甜点区间”）时，每层的时延分解。

![](https://files.mdnice.com/user/59/486980b2-19ce-43c4-80f7-abf4886f38f5.png)

在 batch = 128 的情况下生成 1000 个 token：

- `vLLM`：20.3 秒

- `SGLang`：16.2 秒

`SGLang` 节省 4.1 秒（约 **20%** 更快）

### 小 batch 的优势

真正有意思的是：batch 越小，性能差距越大。

这点非常关键。交互式推理（聊天机器人、代码补全、Agent 等）往往运行在 batch size **1-16**。在这个区间里，`SGLang` 的 **1.34×–1.84×** 优势会直接转化为用户体验。


## Kernel 发现 #1：**融合**消除内存瓶颈

### `vLLM` 的串行实现（7 次 kernel launch）

`vLLM` 的 **MoE** 前向会启动 7 个独立的 CUDA kernel：

来源：`vllm/model_executor/layers/fused_moe/cutlass_moe.py:671-712`

```python
# 1. Reorder tokens by expert assignment
rep_a_fp4 = ops.shuffle_rows(a_fp4, a_map, (m * topk, k))
rep_a_blockscale = ops.shuffle_rows(a_blockscale, a_map, (m * topk, k // 16))

# 2. Quantize activations to FP4
a_fp4, a_blockscale = ops.scaled_fp4_experts_quant(a, a1_gscale, ...)

# 3. First GEMM: gate_up projection
c1 = ops.cutlass_fp4_moe_mm(rep_a_fp4, w1_fp4, rep_a_blockscale, w1_blockscale, ...)

# 4. SiLU activation
intermediate = torch.empty((m * topk, n), device=device, dtype=out_dtype)
torch.ops._C.silu_and_mul(c1, intermediate)

# 5. Quantize intermediate activations
int_fp4, int_blockscale = ops.scaled_fp4_experts_quant(intermediate, a2_gscale, ...)

# 6. Second GEMM: down projection
c2 = ops.cutlass_fp4_moe_mm(int_fp4, w2_fp4, int_blockscale, w2_blockscale, ...)

# 7. Reorder output back to original token order
output = ops.shuffle_rows(c2, c_map, (m, k))
```

代价是什么？

- 7 次 kernel launch（每次约有 `~5-10μs` 的额外开销）
- 7 次全局内存往返
- kernel 之间有 6 处同步点
- 需要为 `rep_a_fp4`、`c1`、`intermediate`、`int_fp4`、`c2` 等分配中间 buffer

在 batch size = 4 时，仅 kernel launch 的开销就会占总时延的 **10%–20%**。

### `SGLang` 的融合归约 kernel

`SGLang` 将“第一次 shuffle + 最后一次 shuffle + 最终归约（reduction）”融合进一个 kernel：

来源：`sglang/sgl-kernel/csrc/moe/prepare_moe_input.cu:258-321`

```c++
template <typename T, typename scalar_t>
__global__ void apply_shuffle_mul_sum_kernel(
    const T* __restrict__ input,           // [m*topk, k]
    const int* __restrict__ permutation,   // [m*topk] mapping
    const scalar_t* __restrict__ weights,  // [m, topk] routing weights
    T* __restrict__ output,                // [m, k]
    int m, int k, int topk
) {
    // 128-bit vectorized loads
    constexpr uint32_t vec_size = 16 / sizeof(scalar_t);
    using vec_t = flashinfer::vec_t<T, vec_size>;

    const int token_idx = blockIdx.x;
    const int feature_idx = threadIdx.x * vec_size;

    vec_t sum;
    sum.fill(scalar_t(0));

    // Iterate over top-k experts for this token
    for (int k_idx = 0; k_idx < topk; k_idx++) {
        const int src_idx = permutation[token_idx * topk + k_idx];
        const scalar_t weight = weights[token_idx * topk + k_idx];

        // Vectorized load from expert output
        vec_t expert_output;
        expert_output.load(input + src_idx * k + feature_idx);

        // Multiply by routing weight and accumulate
        #pragma unroll
        for (int i = 0; i < vec_size; i++) {
            sum[i] += expert_output[i] * weight;
        }
    }

    // Vectorized store to output
    sum.store(output + token_idx * k + feature_idx);
}
```

一个 kernel 里完成三件事：

- Token 重排（`permutation` 查询）
- 路由权重乘法（routing weight multiply）
- `TopK` 归约（对选中的 experts 求和）

收益：

- 激活相关的内存带宽需求降低约 **3×**（只做一次全局内存 pass）
- 更好的 cache 局部性（在 `L1` 里复用 `permutation` 与 `weights`）
- 少 2 个中间 buffer 的分配
- kernel 数量从 7 降到 5（`7 → 5`）

`128-bit` 向量化意味着每个线程每次 load 处理 8 个 `bfloat16` 元素，即使在小 batch 下也能更充分地榨干内存带宽。

### `FlashInfer CuteDSL`：另一种取舍

`FlashInfer` 采用了不同的布局：**expert-first**，而不是 **token-first**。

来源：我的基准脚本

```python
def prepare_flashinfer_input_vectorized(hidden_states, topk_ids, topk_weights, num_experts, topk, device, dtype):
    """Prepare input in expert-first format for FlashInfer CuteDSL.

    Reshapes from [batch, hidden] to [num_experts, max_tokens_per_expert, hidden]
    """
    batch_size, hidden_dim = hidden_states.shape

    # Count tokens per expert using vectorized bincount
    flat_ids = topk_ids.flatten()
    expert_counts = torch.bincount(flat_ids.to(torch.int64), minlength=num_experts).to(torch.int32)
    max_tokens_per_expert = expert_counts.max().item()

    # Sort tokens by expert ID
    sorted_indices = torch.argsort(flat_ids)
    sorted_hidden = weighted_hidden[sorted_indices]
    sorted_expert_ids = flat_ids[sorted_indices]

    # Create expert-first tensor [num_experts, max_tokens, hidden_dim]
    expert_hidden = torch.zeros((num_experts, max_tokens_per_expert, hidden_dim), device=device, dtype=dtype)

    # Fill using advanced indexing
    expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)
    expert_offsets[1:] = expert_counts.cumsum(0)

    token_positions = torch.arange(len(sorted_expert_ids), device=device)
    position_in_expert = token_positions - expert_offsets[sorted_expert_ids]

    expert_hidden[sorted_expert_ids, position_in_expert] = sorted_hidden
    return expert_hidden, expert_counts
```

取舍在于：`FlashInfer` 的预处理更重（`sorting`、`scatter` 等），但 **expert-first** 的布局能带来更好的“按专家维度的 batch 化”。在小 batch（`BS=1-16`）时，这部分开销会明显拖累性能；而在大 batch 下，这种布局的优势更容易摊薄开销，因此整体表现可以接近 `SGLang`。

## Kernel 发现 #2：面向 **Blackwell** 的 **CUTLASS** 特化 schedule

### `SGLang` 的原生 **FP4** schedule

`SGLang` 使用了一个针对 `sm_100a` 上 **grouped FP4 GEMM** 专门设计的、面向 **Blackwell** 优化的 **CUTLASS** schedule：

来源：`sglang/sgl-kernel/csrc/moe/nvfp4_blockwise_moe.cu:196-201`

```c++
// SM100/Blackwell B200 configuration
using ThreadBlockShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;
using AlignmentA = 32;
using AlignmentB = 32;
using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100;
```

`KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100` 带来的收益包括：

- 面向 **FP4** 的 **warp specialization**：为 load **FP4**、反量化到 `FP16/BF16`、以及用 `FP32` 累加分别分配专用 warp 角色，避免通用的 “load-convert-compute” 路径。
- 集成 **TMA**（**Tensor Memory Accelerator**）：异步批量张量 load，绕过 `L1 cache`，直接把数据喂到 shared memory；这要求严格的 **128-byte 对齐**。
- **1 SM grouping**：每个 SM 处理多个专家，而不是“一个专家占一个 SM”。对专家规模不均的 **MoE** 工作负载更友好。
- 原生 **NvFP4** 支持：使用 **Blackwell** 的硬件 **FP4** 指令，而不是软件模拟。

### **TMA** 对齐强制

来源：`sglang/sgl-kernel/csrc/moe/nvfp4_blockwise_moe.cu:89-103`

```c++
// Strict TMA alignment enforcement
assert((reinterpret_cast<uintptr_t>(a_scales_offsets[expert_id]) % 128) == 0
       && "TMA requires 128-byte alignment");

*layout_sfa_ptr = ScaleConfig::tile_atom_to_shape_SFA(
    cute::make_shape(static_cast<int>(m), static_cast<int>(n),
                     static_cast<int>(k), 1)
);
```

`SGLang` 会把 `blockscale offsets` pad 到 **128-token** 边界，来保证 **TMA 对齐**：

来源：`sglang/sgl-kernel/csrc/moe/prepare_moe_input.cu:55-73`

```c++
// Round to 128-token boundaries for TMA
blockscale_offsets[expert_id + 1] =
    (expert_offsets[expert_id + 1] + 127) / 128 * 128;
```

这会浪费少量显存（每个 expert 最多多出 127 个 `float`），但可以确保不会因为未对齐而产生 **TMA stall**。

### `vLLM` 的通用 **CUTLASS** 配置

`vLLM` 使用的是更通用的 **CUTLASS 3.x** schedule：可以跨 `Ampere/Hopper/Blackwell` 工作，但缺少面向 **Blackwell** 的特化优化。

来源：`vllm/csrc/quantization/fp4/nvfp4_blockwise_moe_kernel.cu:93-101`

```c++
// Generic TMA alignment checks (no padding)
assert(reinterpret_cast<uintptr_t>(a_scales) % 128 == 0);
assert(reinterpret_cast<uintptr_t>(b_scales) % 128 == 0);
```

`vLLM` 会检查对齐，但不会通过 padding 去“强制对齐”。因此如果 token 数量天然无法满足对齐条件，**TMA** 可能会回退到更慢的路径。

#### 影响：

影响包括：

- 更高的有效内存带宽（**TMA** vs `L1 cache` 路径）
- 更高的 warp 利用率（特化角色 vs 通用路径）
- 更少的寄存器 spill（针对 `sm_100a` 调优）

## Kernel 发现 #3：面向小 batch 的**自适应 grid sizing**

#### 小 batch 下的 **occupancy** 问题

GPU kernel 想要接近峰值性能，需要足够的并行度去“喂饱”GPU。以拥有 **142** 个 SM 的 `B200` 为例，你至少需要 **142** 个 thread block 才能让所有 SM 都忙起来。

但 **MoE** 在 batch size = 1 时会遇到一个典型问题：

- `64 experts × 4 topk = 256 tokens` 需要处理
- 如果每个 thread block 处理 `128 tokens`，那么只会启动 `2 blocks`
- 结果是约 **98.6%** 的 SM 处于空闲状态

标准 **CUTLASS** 的 launch heuristic 并不擅长适配这种极端小规模的场景。

#### `SGLang` 的动态 block sizing

当并行度不足时，`SGLang` 使用一种自适应策略：用更小的 block 来换取更大的 grid，从而提高总体并行度：

来源：`sglang/sgl-kernel/csrc/gemm/nvfp4_expert_quant.cu:456-477`

```c++
// Adaptive kernel launch configuration
int const workSizePerRow = k / ELTS_PER_THREAD;  // 8 FP4 elements per thread
int const totalWorkSize = m_topk * workSizePerRow;
dim3 block(std::min(workSizePerRow, 512));
int const numBlocksPerSM = 2048 / block.x;

dim3 grid(std::min(
    static_cast<int>((totalWorkSize + block.x - 1) / block.x),
    multiProcessorCount * numBlocksPerSM
));

// Dynamic adjustment: halve block size, double grid size
while (grid.x <= multiProcessorCount && block.x > 64) {
    grid.x *= 2;
    block.x = (block.x + 1) / 2;
}
```

以 batch size = 1 为例：

初始配置：

```c++
m_topk = 256 tokens (1 batch × 64 experts × 4 topk)
k = 2880 (hidden size)
workSizePerRow = 2880 / 8 = 360
totalWorkSize = 256 × 360 = 92,160

block.x = min(360, 512) = 360
grid.x = min(92160 / 360, 142 × 5) = min(256, 710) = 256
```

自适应调整后：

```c++
Iteration 1: grid.x=256 > 142, no change
Final: grid=256, block=360
```

如果工作量更小，上面的 `while` 循环就会触发：

```c++
Iteration 1: grid=128, block=180 → grid=256, block=90
Iteration 2: grid=256 > 142, stop
```

结果：通过在 `block size` 与 `grid size` 之间找到合适平衡，尽可能最大化 **SM occupancy**。

### `vLLM` 的固定 heuristic

`vLLM` 更依赖 **CUTLASS** 默认的 launch heuristic（更偏向大矩阵规模的优化）。在小 batch 下，这会导致：

- block 更大（`256–512` threads）
- block 数更少（利用率不足）
- **occupancy** 更低

实测影响：在 `BS=1-4` 时，`SGLang` 的自适应 sizing 最高可贡献到 **1.84×** 的加速幅度。


## DeepSeek 关联：规模化的**专家并行**（EP）

### DeepSeek-V3 架构

`DeepSeek-V3` 把 **MoE** 推到了更极端的规模：

- 每层 **256** 个 experts
- `Top-8` 路由（激活专家数量约 **2×**）
- `hidden dim = 7168`，`intermediate = 18432`（每个专家的计算量非常大）

这类架构的设计目标是 **专家并行**（**EP**，Expert Parallelism）：把专家切分到多个 GPU/节点上，并通过 **all-to-all** 通信路由 token。我这里没有测 `DeepEP/EP` 的通信开销（那又是另一个话题）。

我基准测试了一个能塞进单张 `B200` 的缩小版本：

```shell
Experts: 256
TopK: 8
Hidden: 2560 (scaled from 7168)
Intermediate: 8960 (scaled from 18432)
```

![](https://files.mdnice.com/user/59/7a09cc85-d033-402b-8107-b3895c112848.png)

![](https://files.mdnice.com/user/59/532c9d7c-19f2-4f52-8eab-21a69a56a557.png)

### 为什么专家更多时差距会缩小

当 experts 数达到 **256** 时，系统天然并行度更高。即便 launch heuristic 不够理想，`vLLM` 在大 batch 下也更容易把 GPU 吃满；此时 **kernel 融合** 与面向 **Blackwell** 的优化优势，在更偏 **compute-bound** 的区间里会变得不那么显著。

但在小 batch 下，`SGLang` 的**自适应 grid sizing** 依旧更占优。因为 experts 更多时路由更不可预测，专家负载不均更明显，而 `SGLang` 的**专家感知量化**（expert-aware quantization）对这种情况处理得更好。

### DeepEP：多节点专家并行

如果要达到真正的 `DeepSeek-V3` 规模（`256` experts 且保持完整维度），通常需要跨节点做 **EP**。`SGLang` 实现了 `DeepEP`（DeepSeek Expert Parallelism），其核心流程包括：

- **All-to-All dispatch**：把 token 路由到拥有对应 expert 的 rank
- **Local GEMM**：每个 rank 计算自己负责的 experts
- **All-to-All combine**：把结果再按原 token 顺序聚合回去

来源：`sglang/srt/layers/moe/token_dispatcher/deepep.py:398-457`

```python
def _dispatch_core(
    self,
    x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    previous_event,
):
    buffer = self._get_buffer()

    # Compute dispatch layout (which tokens go to which rank)
    (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        previous_event,
    ) = buffer.get_dispatch_layout(
        topk_ids,
        self.num_experts,
        previous_event=previous_event,
        async_finish=self.async_finish,
        allocate_on_comm_stream=previous_event is not None,
    )

    # All-to-all dispatch
    (
        recv_x,
        recv_topk_ids,
        recv_topk_weights,
        num_recv_tokens_per_expert,
        self.handle,
        event,
    ) = buffer.dispatch(
        x,
        topk_idx=topk_ids,
        topk_weights=topk_weights,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        previous_event=previous_event,
        async_finish=self.async_finish,
        allocate_on_comm_stream=(previous_event is not None) and self.async_finish,
        expert_alignment=128 if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM else 1,
        config=DeepEPConfig.get_instance().normal_dispatch_config,
    )

    return (
        recv_x,
        recv_topk_ids,
        recv_topk_weights,
        num_recv_tokens_per_expert,
        event,
    )
```

关键细节：

- 使用 `NCCL/RDMA` 实现低时延的 all-to-all
- 支持 **FP8** 通信以降低带宽压力（dispatch 前对激活做量化）
- 有两种模式：`Normal`（prefill）与 `Low-Latency`（decode）

dispatch 后的本地 `GEMM` 阶段同样适用前文提到的 kernel 优化（融合、**Blackwell** schedule、自适应 sizing）。因此 **1.25×–1.84×** 的单卡加速，会与多节点并行进一步叠加。

## `FlashInfer CuteDSL`：第三位选手

`CuteDSL` 是一个相对较新的方向，聚焦于为 **MoE** 工作负载做基于模板的 kernel 生成。

### 性能对比（`GPT-OSS-20B`）

在峰值吞吐（`BS=4096`）时，`FlashInfer` 比 `SGLang` 慢约 **1.0%**（`1156` vs `1168 TFLOPS`）。但在小 batch 下，它会慢 **1.6×–2.4×**。

### 为什么 `FlashInfer` 在小 batch 下吃亏

`FlashInfer` 的 **expert-first** 布局需要较重的预处理：

- 用 `bincount` 统计每个 expert 的 token 数
- 用 `argsort` 按 expert 分组 token
- 用 `scatter` 将数据填入 expert-first 张量（可能引入 padding）

当 `batch size = 1` 且 experts 数为 `64` 时，这些预处理开销会成为主要瓶颈；当 `batch size = 4096` 时，开销被摊薄，`FlashInfer` 的“按专家 batch 化”的优势更容易体现。

### `FlashInfer` 的强项：masked GEMM

`FlashInfer` 支持 **masked GEMM**：把每个 expert 的输出 pad 到固定大小。这带来：

- 更好的 memory coalescing（不再是非规则 stride）
- 更简单的 kernel 逻辑（不需要变长 batch 处理）

当 experts 数很大（`256+`）时，这种方案可能超过 **token-first** 布局；但在 `GPT-OSS-20B`（`64` experts）上，预处理成本往往超过收益。


## 显存带宽分析

下面把 **kernel 融合**带来的显存带宽节省做一个量化。

### `vLLM` 的内存流量

以 `batch size = 128`、`hidden size = 2880`、`64 experts`、`topk = 4` 为例：

```shell
Tokens processed: 128 × 4 = 512 tokens
Hidden size: 2880
Data type: bfloat16 (2 bytes)
```

内存操作：

- `shuffle_rows`（输入）：读取 `512 × 2880 × 2 = 2.95 MB`，写入 `2.95 MB`

- `scaled_fp4_quant`：读取 `2.95 MB`，写入 `1.47 MB（FP4） + 0.09 MB（scales）`

- `cutlass_fp4_moe_mm`（GEMM1）：读取 `1.47 + 0.09 MB（activations） + 56 MB（weights）`，写入 `7.87 MB（intermediate）`

- `silu_and_mul`：读取 `7.87 MB`，写入 `3.94 MB`

- `scaled_fp4_quant`：读取 `3.94 MB`，写入 `1.97 MB + 0.12 MB（scales）`

- `cutlass_fp4_moe_mm`（GEMM2）：读取 `1.97 + 0.12 MB + 28 MB（weights）`，写入 `2.95 MB`

- `shuffle_rows`（输出）：读取 `2.95 MB`，写入 `2.95 MB`

总内存流量：`2×(2.95) + 2×(2.95) + 7.87 + 3.94 + 2.95 = 26.5 MB`

（不含权重读取：权重读取占比很大，但两种实现都类似，因此这里先不计）

### `SGLang`（融合后）的内存流量

`SGLang` 把步骤 `1` 与 `7` 融合进 `apply_shuffle_mul_sum`：

- `scaled_fp4_quant`：读取 `2.95 MB`，写入 `1.47 MB + 0.09 MB`

- `cutlass_fp4_moe_mm`（GEMM1）：读取 `1.47 + 0.09 MB + 56 MB`，写入 `7.87 MB`

- `silu_and_mul`：读取 `7.87 MB`，写入 `3.94 MB`

- `scaled_fp4_quant`：读取 `3.94 MB`，写入 `1.97 MB + 0.12 MB`

- `cutlass_fp4_moe_mm`（GEMM2）：读取 `1.97 + 0.12 MB + 28 MB`，写入 `2.95 MB`

- `apply_shuffle_mul_sum`：读取 `2.95 MB（c2） + 0.01 MB（c_map, topk_weights）`，写入 `2.95 MB`

总内存流量：`2.95 + 7.87 + 3.94 + 2.95 + 2.95 + 0.01 = 20.7 MB`

节省比例：`(26.5 - 20.7) / 26.5 = 21.9%`，即激活相关内存流量减少 **21.9%**。

以 `B200` 的显存带宽（`8 TB/s`）估算：

- `vLLM`：`26.5 MB / 8000 GB/s = 0.0033 ms`
- `SGLang`：`20.7 MB / 8000 GB/s = 0.0026 ms`
- 节省：每层约 `0.0007 ms`

对于 `24` 层、`1000` 次前向（生成 `1000` 个 token）：

- 总节省：`0.0007 × 24 × 1000 = 16.8 ms`

这还是一个偏保守的估计；考虑 cache 效应与 kernel launch 开销减少，真实收益往往更高。

## 为什么 kernel 工程的收益会“叠加放大”

这些优化看上去可能都是“增量提升”：**142 TFLOPS** 的差距、每 1000 token 省 4.1 秒、激活内存流量减少 **21.9%**。但它们会在多个维度上叠加：

### 1) 层数

现代大模型通常有 `24–80` 层，每一层都会跑一遍 **MoE** 前向，因此单层的节省会被放大 `24–80×`。

### 2) token 数

单个请求往往会生成数百到数千个 token；聊天、代码生成与 agent 工作流经常超过 `10K` token。以 `24` 层为例：

- `vLLM`：`0.847 ms/layer × 24 layers × 10,000 tokens = 203 seconds`
- `SGLang`：`0.676 ms/layer × 24 layers × 10,000 tokens = 162 seconds`（按 `128` 请求一个 batch 口径）
- 节省：每个请求约 `320 ms`

对中等规模推理负载（例如 `10K req/day`）而言，这类 kernel 优化往往能在一年内节省可观的成本。

## 给推理框架开发者的建议

如果你在开发推理框架并希望在 **Blackwell** GPU 上获得高性能，下面这些点最关键：

### 1) 更激进地做融合

`vLLM` 的 7-kernel 拆分对调试与模块化来说是合理的，但面向生产的 kernel 应该尽量融合：

- `shuffle + reduce`（如 `SGLang` 的 `apply_shuffle_mul_sum`）
- `quantization + GEMM`（避免单独的 quant kernel）
- `activation + quantization`（例如把 `SiLU` 与后续量化融合）

目标：把 token-to-token 的时延压到 `3–4` 次 kernel launch（`prepare`、`GEMM1`、`GEMM2`、`reduce`）。

### 2) 使用面向硬件的特化 schedule

不要只依赖通用的 **CUTLASS** 配置。NVIDIA 会为不同架构提供各自的优化 schedule：


- Ampere：`KernelTmaWarpSpecialized`
- Hopper：`KernelTmaWarpSpecializedPingpong`
- Blackwell：`KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100`

务必在目标硬件上测试：由于 Tensor Core 配置差异，面向 `Hopper` 的 kernel 在 `Blackwell` 上可能反而不够好。

### 3) 自适应 launch heuristic

固定的 `grid/block` 策略在极端 batch 下往往失效。建议实现：

- 小 batch（`1–16`）：尽量增大 `grid`、减小 `block`
- 大 batch（`512+`）：使用标准 **CUTLASS** heuristic
- 动态调参：首次运行做 profiling，并缓存配置

`SGLang` 的 `while` 循环 heuristic 简单但有效；更复杂的方法（例如 TVM 风格的 auto-tuning）还可以进一步优化。

### 4) 为 **TMA** 强制对齐

如果你在用 **TMA**（在 `Blackwell` 上也确实应该用），就应该把张量 pad 到 **128-byte** 对齐。相对性能收益而言，这点额外内存开销可以忽略。

### 5) 重点 benchmark 小 batch

多数公开 benchmark 关注 `batch size = 128–512`，但交互式推理实际运行在 `BS=1–16`。如果你的目标是聊天、代码补全或 agent，应该优先把小 batch 优化到位。

## 结论

`SGLang` 与 `vLLM` 在 **FP4 MoE** 推理上的 **142 TFLOPS** 差距，不是“CUDA 魔法”或“秘方”，而是系统化的 **kernel 工程**：

- **kernel 融合**：消除 **21.9%** 的激活内存流量
- 面向 **Blackwell** 的 **CUTLASS** schedule：释放原生 **FP4** 与 **TMA** 加速
- **自适应 grid sizing**：在小 batch 下最大化 **SM occupancy**

这些优化会在层数、token 数与请求数上叠加，最终带来：

- batch size = 1（交互式推理）：**1.84×** 加速
- batch size = 128（decode 甜点区间）：**1.25×** 加速

`CuteDSL` 也说明了：**expert-first** 布局在大 batch 下可以接近甚至竞争，但在小 batch 下会被预处理开销拖累。

核心结论是：硬件对 **FP4** 的支持是必要条件，但远远不够；你不能指望“换了卡推理就会自然变快”。想要真正释放性能，需要 kernel 充分利用 **Blackwell** 的独特特性：**TMA**、**warp specialization**、原生 **FP4** 指令等。

随着模型走向 `256+` experts、并且多节点 **EP** 变成常态，这些优化会更重要。今天愿意投入 kernel 工程的框架，往往会定义明天的性能上限。

## 附录：完整基准数据

### 硬件

```shell
GPU: NVIDIA Blackwell B200 on Nebius
Compute Capability: sm_100a
```

### 软件

```shell
vLLM: v0.11.0
SGLang: v0.5.5rc2
FlashInfer CuteDSL: from sglang's
CUDA: 13.0
```

### 基准方法

- 预热（Warmup）：20 次迭代

- 计量（Measurement）：200 次迭代

- 指标：平均时延（`μs/ms`）、标准差、`TFLOPS`

- `TFLOPS` 计算方式：

```shell
flops = batch_size * topk * (
    2 * hidden_dim * inter_dim * 2 +  # up-projection (gate + up)
    2 * inter_dim * hidden_dim         # down-projection
)
tflops = flops * 1e-12 / (mean_ms * 1e-3)
```

- 同步：每次迭代后 `torch.cuda.synchronize()`

- 显存：不同配置之间清理 GPU cache


## 参考链接

### `SGLang`

- GitHub：`sgl-project/sglang`（https://github.com/sgl-project/sglang）

- 融合归约 kernel：`prepare_moe_input.cu`

- Blackwell FP4 GEMM：`nvfp4_blockwise_moe.cu`

- 专家量化：`nvfp4_expert_quant.cu`

- DeepEP 集成：`deepep.py`

- `CuteDSL`（`cutedsl moe` 是 `sglang` 项目的一部分）

### `vLLM`

- GitHub：`vllm-project/vllm`（https://github.com/vllm-project/vllm）

- FP4 MoE layer：`cutlass_moe.py`

- CUTLASS kernel：`nvfp4_blockwise_moe_kernel.cu`

### `FlashInfer`

- GitHub：`flashinfer-ai/flashinfer`（https://github.com/flashinfer-ai/flashinfer）


### `NVIDIA`

- CUTLASS：`NVIDIA/cutlass`（https://github.com/NVIDIA/cutlass）

- Blackwell 架构：NVIDIA Blackwell White Paper（https://resources.nvidia.com/en-us-blackwell-architecture）

- TMA 文档：CUDA Programming Guide - Tensor Memory Accelerator（https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#tensor-memory-accelerator）
