> 本文由 datacrunch 的博客作者 @Paul Chang 授权转载和翻译并发表到本公众号。

# DeepSeek V3 SGLang 优化

继续我们的DeepSeek V3与SGLang集成的技术系列，我们旨在全面概述可用于提高性能和效率的各种优化策略。最终目标是在为DeepSeek V3模型系列（包括R1）提供基于原生优化的竞争性推理性能的同时，培养LLM前沿改进的专业知识。作为推理服务引擎，SGLang与ML基础设施堆栈的多个组件交互，为不同级别的优化提供了机会。大多数优化都以`launch_server` CLI的标志形式出现。这些标志为了解SGLang生态系统中随时间实现的各种性能增强提供了一个便捷的入口点。

## 优化总结表

| 优化                                  | 描述/优势                                                      | 相关标志/注释                                                     |
|--------------------------------------|--------------------------------------------------------------|------------------------------------------------------------------|
| **CUDA Graph执行**                        | 通过重放记录的CUDA操作减少 kernel启动开销                          | `--cuda_graph_max_bs`, `--disable_cuda_graph`                     |
| **Torch编译**                         | 应用 kernel融合、算子消除和图优化                                  | `--enable-torch-compile`, `--torch-compile-max-bs`                |
| **BF16 / FP8 BMM kernel**                | 高精度的内存高效批量矩阵乘法                                    | 无标志（内部 kernel优化）                                             |
| **NextN推测解码 (EAGLE-2)**            | 基于树的验证的并行推测token生成                                  | `--speculative-algo`, `--speculative-draft`, `--speculative-*`    |
| **MLA的数据并行注意力**                | 为多头潜在注意力启用数据并行                                    | `--enable-dp-attention`                                           |
| **重叠调度器**                         | 将CPU调度与GPU执行重叠以减少空闲时间                            | `--disable-overlap-schedule`                                      |
| **FlashInfer MLA优化**                | 融合MLA操作以加快预填充和解码                                  | `--enable-flashinfer-mla`                                         |
| **FP8精度改进**                        | 分块/分片缩放，FP32累加以减少溢出                               | 无标志（在 kernel内处理）                                             |
| **FP8 GEMM kernel调优**                   | 为每个GPU选择最佳块形状以获得最优FP8性能                        | 脚本: `quantizationtuning_block_wise_fp8.py`                      |
| **FP8 GEMM (CUTLASS kernel)**             | 高效的融合量化和矩阵乘法                                        | 无标志（ kernel级实现）                                               |
| **融合MoE kernel+调优**                   | 使用自定义SGLang kernel调优的更快专家混合                          | 脚本: `tuning_fused_moe_triton.py`                                |


##  kernel执行优化


#### 相关标志:

```bash
--disable_cuda_graph: # 禁用CUDA Graph。
--cuda_graph_bs: # 由`CudaGraphRunner`捕获的批量大小。
--cuda_graph_max_bs: # 使用CUDA Graph时调整最大批量大小。
--enable-torch-compile: # 启用捕获的CUDA Graph的torch.compile编译。
```

#### **背景:**

CUDA Graph(https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)和`torch.compile`(https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)标志都致力于提高 kernel操作的效率。CUDA Graph通过记录和重放CUDA操作序列作为单个单元，显著减少了 kernel启动开销，消除了推理期间每个 kernel的启动成本。同时，`torch.compile`采用 kernel融合、算子消除和专门的 kernel选择来优化计算图。然而，SGLang的`torch.compile`可以使用PyTorch生成的图或CUDA Graph来连接这两种优化。

#### **提交:**

支持triton后端的CUDA Graph(https://github.com/sgl-project/sglang/pull/1401)，支持DP注意力的CUDA Graph #2061(https://github.com/sgl-project/sglang/pull/2061)


#### **基准测试:**

```bash
$ python3 -m sglang.bench_one_batch --batch-size 1  --input 256
--output 32 --model deepseek-ai/DeepSeek-V3  --trust-remote-code  --tp 8
--torch-compile-max-bs 1 --disable-cuda-graph
--profile

$ python3 -m sglang.bench_one_batch --batch-size 1  --input 256
--output 32 --model deepseek-ai/DeepSeek-V3  --trust-remote-code  --tp 8
--torch-compile-max-bs 1 --cuda-graph-max-bs 1
--profile

$ python3 -m sglang.bench_one_batch --batch-size 1  --input 256
--output 32 --model deepseek-ai/DeepSeek-V3  --trust-remote-code  --tp 8
--enable-torch-compile --torch-compile-max-bs 1 --cuda-graph-max-bs 1
 
```

#### **结果:**

![compile-bench](https://files.mdnice.com/user/59/ce1fb767-b855-4675-8921-5fc4c1fbd273.png)

正如预期的那样，当堆叠优化（torch.compiler / CUDA Graph + torch.compiler / torch.compiler(CUDA Graph) + torch.compiler）时，我们减少了总延迟（`7.322 / 1.256 / 1.011 s`）并提高了总吞吐量（`39.34 / 229.27 / 284.86 token/s`）。

**注意:** 我们看到预填充阶段延迟的下降，这是由于torch.compiler编译和CUDA Graph未捕获预填充阶段操作导致的初始计算增加（`0.21180 / 0.25809 / 0.26079 s`）和吞吐量（`1208.67 / 991.92 / 981.64 token/s`）。

### bf16批量矩阵乘法 (bmm)

#### **背景:**

批量矩阵乘法是LLM中执行的主要工作负载。由于DeepSeek-V3使用不同的量化fp8数据类型（float8_e5m2和float8_e4m3fn）进行训练（从而减少内存分配），我们测试了具有不同fp8和基础bf16数据类型组合的随机bmm集合的精度和延迟。此优化不依赖于标志。

#### **提交:** 

(修复MLA的fp8并支持DeepSeek V2的bmm fp8(https://github.com/sgl-project/sglang/pull/1285)，在AMD GPU上启用DeepseekV3(https://github.com/sgl-project/sglang/pull/2601)，将bmm_fp8 kernel集成到sgl-kernel中(https://github.com/sgl-project/sglang/pull/3056))

#### **基准测试:**

```bash
$ pytest -s test_bmm_fp8.py
```

* 使用修改版的`test_bmm_fp8.py`获得的结果  

#### **结果:**

![bmm-bench](https://files.mdnice.com/user/59/0882db5b-273c-409e-9982-6e8b6837a095.png)

结果之间的相似度接近相同（余弦相似度=1相同），这表示没有精度损失，而fp8的延迟比bf16差，这是由于类型转换计算导致的。 

### 支持nextn推测解码

#### **相关标志:**

```bash
--speculative-num-steps: # 从草稿模型中采样的步骤数。
--speculative-eagle-topk: # 在EAGLE-2的每个步骤中从草稿模型中采样的token数。
--speculative-num-draft-tokens: # 在推测解码中从草稿模型中采样的token数。
--speculative-draft: # 要使用的草稿模型。它需要与验证器模型相同的分词器（默认：SGLang/DeepSeek-V3-NextN）。
```

#### **背景:**

推测解码通过引入草稿模型（一个更小、更快的模型）来加速推理，该模型一次生成多个token。然后验证步骤检查这些草稿token是否与更大、更准确的LLM的预测匹配。

其主要缺陷是，由于Naive的推测解码生成单个线性草稿token序列，如果序列中的单个token被拒绝，所有后续token都会被丢弃，降低了接受率。

SGLang的NextN实现基于EAGLE-2和SpecInfer:

![speculative_decoding.png](https://files.mdnice.com/user/59/7be3fb51-cc3e-415d-adbd-e66ec3e31a58.png)

使用基于树的推测解码（SpecInfer和EAGLE-2），预测被组织为树，其中每个节点代表一个可能的下一个token。通过这种方法，我们生成多个可以并行验证器LLM验证的推测分支，提高了接受率。

EAGLE-2的关键改进是基于上下文的动态草稿树和基于草稿模型置信度分数的节点剪枝。

#### **提交:** 

([Track] DeepSeek V3/R1 nextn进度 #3472,(https://github.com/sgl-project/sglang/issues/3472)，支持DeepSeek-V3/R1的NextN (MTP)推测解码 #3582(https://github.com/sgl-project/sglang/pull/3582)，支持Triton后端的Eagle2 #3466(https://github.com/sgl-project/sglang/pull/3466)，Eagle推测解码第4部分：添加EAGLE2工作器 #2150(https://github.com/sgl-project/sglang/pull/2150))

#### **基准测试:**

无标志。

```bash
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code
python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-input 256 --random-output 32 --random-range-ratio 1 --num-prompts 1 --host 127.0.0.1 --port 30000
```

`--speculative-algo NEXTN --speculative-draft SGLang/DeepSeek-V3-NextN --speculative-num-steps 2 --speculative-eagle-topk 4 --speculative-num-draft-tokens 4`

```bash
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --speculative-algo NEXTN --speculative-draft SGLang/DeepSeek-V3-NextN --speculative-num-steps 2 --speculative-eagle-topk 4 --speculative-num-draft-tokens 4 --tp 8 --trust-remote-code
python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-input 256 --random-output 32 --random-range-ratio 1 --num-prompts 1 --host 127.0.0.1 --port 30000 
```

#### **结果:**

![spec-bench](https://files.mdnice.com/user/59/23464004-5ddb-4297-aa16-65ba1960c295.png)

我们实现了总体吞吐量（请求、输入和输出）的改进，并显著（x6）减少了端到端延迟。

> GiantPandaLLM注：这里的测试结果存疑，应该没这么大加速，不过先了解这是一个有用的优化就可以。

## MLA

### TP+DP注意力

#### **相关标志:**

```bash
--enable-dp-attention: # 启用兼容的MLA数据并行。
```

#### **背景:**

张量并行（TP）通过将KV Cache按TP设备（通常是8）分割来与MHA一起工作，因此每个设备处理KV Cache的1/TP。 [1]

如果我们将其应用于多头潜在注意力（MLA）和TP，每个GPU沿`head_num`维度分割`kv cache`。然而，MLA的`kvcache`的`head_num`为`1`，使其无法分割。因此，每个GPU必须维护一个完整的`kvcache` → `kvcache`在每个设备上被复制。

当对MLA使用DP（数据并行）时，它按请求分割，不同请求的潜在状态缓存存储在不同的GPU中。例如，由于我们无法分割唯一的KV Cache，我们将数据分成批次并将它们并行化到执行不同任务（预填充、解码）的不同工作器中。

在MLA之后，执行all-gather操作，允许每个GPU获取所有序列的`hidden_state`。然后，在**MOE（专家混合）**之后，每个GPU使用**slice**操作提取其对应的序列。

![dp_attn.png](https://files.mdnice.com/user/59/7f50d6d4-e639-4fc9-b854-d780214156ac.png)

#### **提交:** 

(支持DP注意力的CUDA Graph(https://github.com/sgl-project/sglang/pull/2061)，支持多节点DP注意力(https://github.com/sgl-project/sglang/pull/2925)，多节点张量并行(https://github.com/sgl-project/sglang/pull/550)，支持DP MLA(https://github.com/sgl-project/sglang/pull/1970))

#### **基准测试:**

无标志

```bash
# 使用分析器环境启动服务器
export SGLANG_TORCH_PROFILER_DIR=/sgl-workspace/profiler_env_folders/ # 可选用于分析
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code

# 预填充
python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-input 512 --random-output 1 --random-range-ratio 1 --num-prompts 10000 --host 127.0.0.1 --port 30000 
# 解码
python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-input 1 --random-output 512 --random-range-ratio 1 --num-prompts 10000 --host 127.0.0.1 --port 30000
```

`—enable-dp-attention`

```bash
# 使用分析器环境启动服务器
export SGLANG_TORCH_PROFILER_DIR=/sgl-workspace/profiler_env_folders/
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code --enable-dp-attention

# 预填充
python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-input 512 --random-output 1 --random-range-ratio 1 --num-prompts 10000 --host 127.0.0.1 --port 30000
# 解码
python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-input 1 --random-output 512 --random-range-ratio 1 --num-prompts 10000 --host 127.0.0.1 --port 30000
```

#### **结果:**

![dp-bench](https://files.mdnice.com/user/59/1e8640d8-689d-4a53-a20a-105c40bbc800.png)

由于它是一个调度器范式，在使用大批量大小时表现更好。否则，添加的开销大于实际的数据并行化。

对于较大的批量大小（在这种情况下为10,000），我们看到从端到端延迟到总体吞吐量和并发性的预填充和解码阶段都有整体改进。

### 支持与DP注意力重叠调度器


#### **相关标志:**

```bash
--disable-overlap-schedule: # 禁用开销调度器
```

#### **背景:**

我们可以将CPU调度与GPU计算重叠。调度器提前运行一个批次并准备下一个批次所需的所有元数据。通过这样做，我们可以让GPU在整个持续时间内保持忙碌，并隐藏昂贵的开销，如radix cache操作。 

![overlap_scheduler.png](https://files.mdnice.com/user/59/ab3bea07-ac46-4f00-bab3-0525ffc13b71.png)

#### **提交:** 

(更快的重叠调度器(https://github.com/sgl-project/sglang/pull/1738)，默认启用重叠(https://github.com/sgl-project/sglang/pull/2067)，为triton注意力后端默认启用重叠调度器(https://github.com/sgl-project/sglang/pull/2105))

#### **基准测试:**

`--disable-overlap-schedule`

```bash
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code --disable-overlap-schedule
python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-input 256 --random-output 32 --random-range-ratio 1 --num-prompts 10000 --host 127.0.0.1 --port 30000
```

无标志 → 启用重叠调度器:

```bash
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code

python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompts 2500 --random-input-len 1024 --random-output-len 1024 --random-range-ratio 1
```

#### **结果:**

![fig_overlap](https://files.mdnice.com/user/59/d214897b-ebea-4d0b-a0ce-507aacfa0976.png)

我们看到延迟的普遍减少：端到端（标准：`1080152.26s`|重叠：`1066166.84s`），每个输出token的时间（标准：`348.10s`|重叠：`196.79s`）和token间延迟（标准：`350.62s`|重叠：`197.96s`），尽管第一个token的时间呈现了调度开销的下降结果（标准：`724050.93s`|重叠：`864850.926s`）。

对于更大的输入和输出请求大小，重叠调度器的效果将更加明显。

### FlashInfer预填充和MLA解码

#### **相关标志:**

```bash
--enable-flashinfer-mla: # 启用FlashInfer MLA优化
```

#### **背景:**

使用FlashInfer后端代替triton。

#### **提交:**
(为flashinfer mla添加快速解码plan,(https://github.com/sgl-project/sglang/pull/3987) 无权重吸收的MLA预填充(https://github.com/sgl-project/sglang/pull/2349))

#### **基准测试:**

无标志:

```bash
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code
python3 benchmark/gsm8k/bench_sglang.py --num-shots 8 --num-questions 1319 --parallel 1319
```

```bash
Accuracy: 0.951
Latency: 77.397 s
Output throughput: 1809.790 token/s
```

使用`--enable-flashinfer-mla`

```bash
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code --enable-flashinfer-mla
python3 benchmark/gsm8k/bench_sglang.py --num-shots 8 --num-questions 1319 --parallel 1319
```

```bash
Accuracy: 0.948
Latency: 71.480 s
Output throughput: 1920.021 token/s
```


#### **结果:**

![flashinfer_mla png](https://files.mdnice.com/user/59/a1b99699-7e50-4c15-a696-e743d550cdee.png)


由于FlashInfer融合操作，我们在相似精度下获得了更低的延迟和更高的输出吞吐量。

## FP8

### 提高FP8的精度

#### **背景:**

当值超过给定数值格式（如FP8）的可表示范围时，会发生数值溢出，导致不正确或无限值。在Tensor Core上FP8量化的上下文中，溢出发生是因为FP8具有非常有限的动态范围。为了防止数值溢出，在量化之前使用矩阵的最大元素将值缩小，尽管这使其对异常值敏感。为了避免这种情况，DeepSeek团队提出了分块和分片缩放，其中权重矩阵的每个128×128子矩阵和激活向量的每个1×128子向量分别进行缩放和量化。

NVIDIA H800 Tensor Core上的FP8 GEMM累加限制在约`14位`精度，这显著低于FP32累加精度。这就是为什么DeepSeek使用CUDA Core的单独FP32累加器寄存器，从而减轻精度损失。反量化缩放因子也应用于这个FP32累加器。

![fp8_deepseek.png](https://files.mdnice.com/user/59/61ed24d7-948a-432f-b3c7-04cffb853afa.png)

#### **提交:** 
(支持分块fp8矩阵乘法 kernel #3267(https://github.com/sgl-project/sglang/pull/3267)，添加分块fp8的单元测试#3156(https://github.com/sgl-project/sglang/pull/3156)，集成分块fp8 kernel#3529(https://github.com/sgl-project/sglang/pull/3529)，[Track] DeepSeek V3/R1精度(https://github.com/sgl-project/sglang/issues/3486))

#### **基准测试:**

```bash
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-R1 --tp 8 --trust-remote-code
python3 benchmark/gsm8k/bench_sglang.py --num-shots 8 --num-questions 1319 --parallel 1319
```
#### **结果:**

![](https://files.mdnice.com/user/59/edec0dc6-caf5-42db-9bd8-955f35ee540d.png)


对于相同的精度（gsm8k上为`0.955` vs `0.957`），我们观察到更高的输出吞吐量和更低的延迟。

### FP8 GEMM调优

#### **背景:**

SGLang采用FP8分块量化调优来优化不同GPU的性能。该实现特别针对AMD和CUDA架构对FP8 GEMM（通用矩阵乘法） kernel进行基准测试，测试不同的块形状以基于延迟确定最有效的配置。

这种方法确保分块量化与GEMM操作的最佳块大小对齐，在最大化计算效率的同时最小化精度损失。计算在FP8中执行，但累加在BF16中进行以在最终输出存储之前保持数值稳定性。

关键函数:

```bash
# fn: benchmark_config(A_fp8, B_fp8, As, Bs, block_size, config, out_dtype=torch.float16, num_iters=10)
A: torch.Tensor,     # 输入矩阵 (FP8) - 通常是激活
B: torch.Tensor,     # 输入矩阵 (FP8) - 通常是权重
As: torch.Tensor,    # `A`的每个token组的缩放因子
Bs: torch.Tensor,    # `B`的每个块的缩放因子
block_size: List[int],  # 量化的块大小 (例如, [128, 128])
config: Dict[str, Any],  #  kernel配置参数
output_dtype: torch.dtype = torch.float16,  # 输出精度
```

```bash
# fn: tune(M, N, K, block_size, out_dtype, search_space):
M,N,K: int  # 矩阵乘法的形状 (M × K @ K × N → M × N)
block_size: int # 定义分块量化大小的元组 ([block_n, block_k])
out_dtype: str # 输出精度 (例如, float16, bfloat16)
search_space: List[dict{str,int}] # 要测试的配置列表 (例如, 块大小, warp数量)。

# search_space示例:
{
"BLOCK_SIZE_M": block_m,
"BLOCK_SIZE_N": block_n,
"BLOCK_SIZE_K": block_k,
"GROUP_SIZE_M": group_size,
"num_warps": num_warps,
"num_stages": num_stages,
}
```

#### **提交:** 
(添加分块fp8调优#3242]https://github.com/sgl-project/sglang/pull/3242))

#### **基准测试:**

```bash
$python3 benchmark/kernels/quantizationtuning_block_wise_fp8.py
```

#### **结果:**

 kernel的最佳配置示例：`N=512,K=7168,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]`

```bash
[...]
{
    "2048": {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 4
    },
    "3072": {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 3
    },
    "4096": {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 64,
        "num_warps": 4,
        "num_stages": 3
    }
}
```

对于所有要调优的批量大小和给定的FP8数据类型，给定脚本测试并比较不同的模型权重维度（N和K）以基于最低延迟优化FP8 GEMM分块量化。这为每个批量大小获得块平铺维度(`BLOCK_SIZE_M/N/K`)、组大小(`GROUP_SIZE_M`)（用于一起分组的块数量，改善L2缓存使用）、每个线程块（即每个块）的warp数量(`num_warps`)以及用于将块加载到共享内存作为预取的阶段数量(`num_stages`)的最优配置。反过来，这实现了不同配置的计算参数的自动调优。

### FP8 GEMM CUTLASS实现

#### 背景:

量化操作可以融合到FP8矩阵乘法操作中以提高效率。在`sgl-kernel/src/sgl-kernel/csrc/int8_gemm_kernel.cu`中，有一个CUDA加速的8位整数(int8)缩放矩阵乘法实现，与W8A8量化融合。

#### **提交:** 
(支持CUTLASS的w8a8 fp8 kernel #3047(https://github.com/sgl-project/sglang/pull/3047)，支持cutlass Int8 gemm #2752(https://github.com/sgl-project/sglang/pull/2752)，支持sm90 Int8 gemm#3035(https://github.com/sgl-project/sglang/pull/3035)，来自NVIDIA/cutlass的FP8分块缩放 #1932(https://github.com/NVIDIA/cutlass/pull/1932))

#### **基准测试:**

```bash
root@cluster-h200-02-f2:/sgl-workspace/sglang/sgl-kernel/benchmark# python3 bench_int8_gemm.py 
```
#### **结果:**

![](https://files.mdnice.com/user/59/ebf26385-59b1-4cb8-9b3f-c813ceb9818c.png)


基准测试测量每个批量大小的GB/s（吞吐量的另一种度量）。比较vLLM kernel（int8 gemm）与SGLang kernel，我们为不同配置（N和K）的不同批量大小获得更高的吞吐量。

**注意**: 我们使用DeepSeek-Coder-V2-Lite-Instruct测试了这个基准，因为DeepSeek-V3的代码尚未在SGLang中实现。

## MoE

### H200的FusedMoE调优

以下是使用token和专家矩阵的专家混合(MOE)融合计算的实现，使用乘法`A @ B`（token×专家矩阵乘法）进行top-k路由，并支持：

- `fp16`、`bfloat16`、`fp8`、`int8`格式
- 通过`A_scale`、`B_scale`进行权重/激活缩放
- 分块量化
- 通过`expert_ids`进行专家路由

#### **背景:**

SGLang的自定义融合MoE kernel，使用vLLM作为参考和基准，由以下组成： 

`tuning_fused_moe_triton.py`：用于调优`fused_moe_triton` kernel的工具。改编自vllm的benchmark_moe.py(https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py)，增加了对各种模型架构的支持。

`benchmark_vllm_vs_sglang_fused_moe_triton.py`：用于比较vLLM和SGLang实现之间融合MoE kernel性能的工具。支持各种模型架构和数据类型。

`benchmark_torch_compile_fused_moe.py`：用于对融合MoE kernel与`torch.compile`和原始融合MoE kernel进行基准测试的工具。

#### **提交:** 
(为fused_moe添加单元测试(https://github.com/sgl-project/sglang/pull/2416)，MoE专家并行实现(https://github.com/sgl-project/sglang/pull/2203)，`benchmark/kernels/fused_moe_triton/README.md`(https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton))

#### **基准测试:**

```bash
$ python3 benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py --model deepseek-ai/DeepSeek-V3 --tp-size 8  --dtype fp8_w8a8 --tune
                                                                          
Writing best config to E=256,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json...                                                                       
Tuning took 5267.05 seconds
```

FusedMoE基准测试sgl-kernel vs vllm:
```bash
python3 benchmark/kernels/fused_moe_triton/benchmark_vllm_vs_sglang_fused_moe_triton.py
[...]
benchmark sglang_fused_moe_triton with batch_size=505
benchmark vllm_fused_moe_triton with batch_size=506
benchmark sglang_fused_moe_triton with batch_size=506
benchmark vllm_fused_moe_triton with batch_size=507
benchmark sglang_fused_moe_triton with batch_size=507
benchmark vllm_fused_moe_triton with batch_size=508
benchmark sglang_fused_moe_triton with batch_size=508
benchmark vllm_fused_moe_triton with batch_size=509
benchmark sglang_fused_moe_triton with batch_size=509
benchmark vllm_fused_moe_triton with batch_size=510
benchmark sglang_fused_moe_triton with batch_size=510
benchmark vllm_fused_moe_triton with batch_size=511
benchmark sglang_fused_moe_triton with batch_size=511
benchmark vllm_fused_moe_triton with batch_size=512
benchmark sglang_fused_moe_triton with batch_size=512

fused-moe-performance:
[...]
     batch_size  vllm_fused_moe_triton  sglang_fused_moe_triton
505       506.0               1.014688                 0.507488
506       507.0               1.011744                 0.509344
507       508.0               1.007200                 0.504288
508       509.0               1.007232                 0.505696
509       510.0               1.007792                 0.507712
510       511.0               1.011072                 0.507248
511       512.0               1.012992                 0.507840
````

#### 结果:

我们对DeepSeek-V3的融合MoE kernel进行了FP8量化调优，获得了每个批量大小的最优配置，类似于调优FP8 GEMM时：

> 对于块平铺维度(`BLOCK_SIZE_M/N/K`)，组大小(`GROUP_SIZE_M`)用于一起分组的块数量，改善L2缓存使用，每个线程块（即每个块）的warp数量(`num_warps`)，以及用于将块加载到共享内存作为预取的阶段数量(`num_stages`)。

![](https://files.mdnice.com/user/59/902b6fec-c08c-4a70-85ae-58eff0a0f788.png)


然后我们比较SGLang的融合MoE kernel实现与vLLM的基准实现，获得了一个更精细的版本，在增加批量大小时几乎保持恒定延迟。

![fused_moe_latency_comparison.png](https://files.mdnice.com/user/59/026774ab-0451-45b5-98d8-b2a12b144ecf.png)

作为结束语，本技术博客使用的版本是sglang: v0.4.3.post2, sgl-kernel: 0.0.3.post6, torch: 2.5.1和CUDA: 12.5。 

我们强烈支持sglang的协作，它作为DeepSeek模型系列的事实上的开源推理引擎。 

未来的工作计划通过分析关键组件（如FlashMLA kernel、FlashAttention和sglang Triton kernel）的性能和增量改进来进一步探索这些优化。我们还建议探索sglang团队实现的新优化，如预填充和解码阶段的分割以及DeepGemm与FusedMoE的集成。

感谢sglang团队的帮助、对本博客的审查以及他们在项目上的联合协作。

## 参考文献

- sglang kernel测试: https://github.com/sgl-project/sglang/tree/main/sgl-kernel/tests

- sglang kernel基准测试: https://github.com/sgl-project/sglang/tree/main/sgl-kernel/benchmark

- [功能] DeepSeek V3优化 #2591: https://github.com/sgl-project/sglang/issues/2591

- 博客 deepseek v3 10倍效率背后的关键技术: https://dataturbo.medium.com/key-techniques-behind-deepseek-models-10x-efficiency-1-moe-9bd2534987c8

- AI编译器Sglang优化工作: https://carpedm30.notion.site/02-19-2024-2nd-meeting

- lmsys sglang 0.4数据并行: https://lmsys.org/blog/2024-12-04-sglang-v0-4/#data-parallelism-attention-for-deepseek-models

- lmsys sglang 0.4零开销批处理调度器: https://lmsys.org/blog/2024-12-04-sglang-v0-4/#zero-overhead-batch-scheduler

- spaces.ac.cn: MQA, GQA, MLA博客: https://spaces.ac.cn/archives/10091

- 基于树的推测解码论文: https://arxiv.org/pdf/2305.09781

- EAGLE2推测解码论文: https://arxiv.org/pdf/2406.16858

- DeepSeek v3论文: https://arxiv.org/pdf/2412.19437

- 知乎博客: EAGLE: 推测采样需要重新思考特征不确定性: https://zhuanlan.zhihu.com/p/687404563

- 知乎博客: MLA tp和dp: https://zhuanlan.zhihu.com/p/25573883266

- 知乎博客: MLA tp和dp第2部分: https://zhuanlan.zhihu.com/p/15280741714

- Colfax deepseekv3 fp8混合精度训练: https://research.colfax-intl.com/deepseek-r1-and-fp8-mixed-precision-training/

  
