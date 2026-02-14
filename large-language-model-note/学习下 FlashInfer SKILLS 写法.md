
## 0x0. 背景

最近在看 FlashInfer 仓库的时候发现它的 `.claude/skills/` 目录下维护了三个 SKILL 文件， SKILL 允许开发者在项目仓库中放置结构化的指引文档（通常是 `SKILL.md` 文件），供 AI 代码助手在执行任务时读取和遵循。它的核心思路是：将项目特有的开发流程、调试方法、最佳实践等知识编码成文档，这样当开发者在 Claude Code/Cursor 中让 AI 帮忙完成相关任务时，AI 会先读取对应的 SKILL 文件，然后按照其中的步骤来执行，而不是靠通用知识去猜测。

举个例子，如果你在 Cursor 中让 AI "帮我给 FlashInfer 添加一个新的 CUDA kernel"，AI 会自动读取 `add-cuda-kernel/SKILL.md`，然后严格按照 FlashInfer 项目自身定义的文件结构、命名规范、测试要求来生成代码，而不是用一个通用的模板去做。这对于像 FlashInfer 这样有复杂构建系统（TVM-FFI、JIT 编译）和特定代码组织规范的项目来说，价值是很大的，可以避免很多重复性的多轮对话，节省时间和大量token。

FlashInfer 目前维护了三个 SKILL：

- `debug-cuda-crash`：CUDA crash 调试教程
- `benchmark-kernel`：Kernel 性能基准测试指南
- `add-cuda-kernel`：添加新 CUDA kernel 的完整流程

下面逐个介绍下（算是简短翻译吧）。

## 0x1. debug-cuda-crash：CUDA Crash 调试

这个 SKILL 的核心是围绕 FlashInfer 的 `@flashinfer_api` 日志装饰器来做 CUDA crash 调试。

### 问题背景

CUDA 错误（非法内存访问、越界、NaN/Inf 等）经常会直接让程序崩溃，崩溃后没有留下任何调试信息。FlashInfer 的 `@flashinfer_api` 装饰器的做法是在 API **执行之前**就把输入信息记录下来，这样即使程序崩溃了，也能看到最后一次调用的输入是什么。

### 使用方式

通过环境变量控制日志级别和输出目标：

| 变量 | 值 | 说明 |
|------|-----|------|
| `FLASHINFER_LOGLEVEL` | `0` | 不记录（默认） |
| | `1` | 只记录函数名 |
| | `3` | 记录输入/输出的元信息（shape, dtype, device 等） |
| | `5` | 额外记录 tensor 统计信息（min/max/mean/nan_count/inf_count） |
| `FLASHINFER_LOGDEST` | `stdout` | 输出到控制台（默认） |
| | `stderr` | 输出到 stderr |
| | `<path>` | 输出到文件 |
| | `log_%i.txt` | 多进程模式，`%i` 会被替换为进程 ID |

一个典型的调试流程是：

```bash
export FLASHINFER_LOGLEVEL=3
export FLASHINFER_LOGDEST=debug.log
python my_script.py
```

然后查看 `debug.log`，最后一条 API 调用记录就是崩溃前的输入。比如一个 shape mismatch 的例子，日志会显示：

```
[2025-12-18 10:32:15] FlashInfer API Call: batch_decode_with_padded_kv_cache
Positional input arguments:
  arg[0]:
    Tensor(
      shape=(32, 8, 128)      # Query tensor
      ...
    )
Keyword input arguments:
  kv_cache=
    Tensor(
      shape=(1024, 2, 8, 64)  # ❌ Wrong! Should be (..., 128) not (..., 64)
      ...
    )
```

这样就能定位到 `head_dim` 不匹配（64 vs 128）的问题。

### 常见错误的排查方法

文档梳理了四类常见 CUDA 错误的排查方法：

1. **Illegal Memory Access**：用 Level 3 检查 tensor shape、是否在 CUDA 上、stride 是否合理、是否 contiguous。
2. **NaN/Inf**：用 Level 5 查看 `nan_count`、`inf_count`、`min`/`max` 是否异常。常见原因包括除零、溢出、未初始化内存。
3. **Out of Memory**：用 Level 3 检查 tensor shape 是否意外过大。
4. **Wrong Dtype**：用 Level 3 直接看 dtype 字段。

### 多进程调试

多 GPU 场景下可以用 `%i` 模式为每个进程生成独立日志：

```bash
export FLASHINFER_LOGDEST=debug_rank_%i.txt
torchrun --nproc_per_node=4 my_script.py
```

### 进阶调试

文档还介绍了结合 `compute-sanitizer` 和 `cuda-gdb` 的用法，以及在 CUDA kernel 中用 `printf()` 调试的方法。这里有一个值得注意的点：对于 warp-specialized kernel，不能简单用 `threadIdx.x == 0` 来做打印条件，因为这样只有 warp 0 会打印，需要根据 kernel 设计选择每个 group 的代表线程。

另外文档提到，Level 5 的统计信息在 CUDA graph capture 期间会自动跳过（避免同步），这是正常行为。日志功能在关闭时（`LOGLEVEL=0`）是零开销的，装饰器直接返回原始函数。

## 0x2. benchmark-kernel：Kernel 基准测试

这个 SKILL 介绍了如何对 FlashInfer 的 kernel 做准确的性能测试。

### 计时方法

FlashInfer 支持两种计时方式：

1. **CUPTI（推荐）**：硬件级别的 profiling，测量纯 GPU 计算时间，没有 host-device 同步开销。需要 `cupti-python >= 13.0.0`（CUDA 13+）。
2. **CUDA Events（回退）**：标准的 CUDA event 计时，如果 CUPTI 不可用会自动使用。精度稍差，对于非常快的 kernel（5-50 us）会有同步开销，但对较长的 kernel 影响可忽略。

框架会自动检测 CUPTI 是否可用，不需要手动切换。安装方式是 `pip install -U cupti-python`。

### 方法一：使用 flashinfer_benchmark.py

这是推荐的基准测试方式。支持的测试 routine 包括：

- **Attention**：`BatchDecodeWithPagedKVCacheWrapper`、`BatchPrefillWithPagedKVCacheWrapper`、`BatchPrefillWithRaggedKVCacheWrapper`、`BatchMLAPagedAttentionWrapper`
- **GEMM**：`bmm_fp8`、`gemm_fp8_nt_groupwise`、`group_gemm_fp8_nt_groupwise`、`mm_fp4`
- **MOE**：`trtllm_fp4_block_scale_moe`、`trtllm_fp8_block_scale_moe`、`trtllm_fp8_per_tensor_scale_moe`、`cutlass_fused_moe`

一个 decode attention 的基准测试示例：

```bash
python benchmarks/flashinfer_benchmark.py \
    --routine BatchDecodeWithPagedKVCacheWrapper \
    --backends fa2 fa2_tc cudnn \
    --page_size 16 \
    --batch_size 32 \
    --s_qo 1 \
    --s_kv 2048 \
    --num_qo_heads 32 \
    --num_kv_heads 8 \
    --head_dim_qk 128 \
    --head_dim_vo 128 \
    --q_dtype bfloat16 \
    --kv_dtype bfloat16 \
    --num_iters 30 \
    --dry_run_iters 5 \
    --refcheck \
    -vv
```

输出包含四个关键指标：

- **median time**：kernel 执行时间的中位数（越低越好）
- **std**：标准差（越低说明越稳定）
- **achieved tflops**：有效 TFLOPS 吞吐
- **achieved tb_per_sec**：内存带宽利用率

还支持批量测试，把多组参数写到一个 testlist 文件里一次性跑：

```bash
python benchmarks/flashinfer_benchmark.py \
    --testlist my_benchmarks.txt \
    --output_path results.csv \
    --generate_repro_command \
    --refcheck
```

常用的 flag 包括 `--num_iters`（测量迭代次数，默认 30）、`--dry_run_iters`（warmup 次数，默认 5）、`--refcheck`（验证输出正确性）、`--use_cuda_events`（强制用 CUDA events）、`--no_cuda_graph`（禁用 CUDA graph）、`--generate_repro_command`（打印可复现的命令）等。

### 方法二：在 Python 中使用 bench_gpu_time()

对于自定义的 benchmark 脚本，可以直接用 FlashInfer 提供的 `bench_gpu_time` 函数：

```python
from flashinfer.testing import bench_gpu_time

median_time, std_time = bench_gpu_time(
    my_kernel_wrapper,
    args=(q, k, v),
    enable_cupti=True,          # 优先 CUPTI，自动回退到 CUDA events
    num_iters=30,
    dry_run_iters=5,
)
```

还支持 `cold_l2_cache=True` 来做冷 L2 cache 的基准测试。

### 问题排查

文档列举了几个常见问题：结果不稳定时可以增加 warmup 和测量迭代次数，或者用 `cold_l2_cache`；reference check 失败时可以加 `--allow_output_mismatch` 继续跑；某些 backend 不支持当前 GPU 架构时会有明确的警告信息。

## 0x3. add-cuda-kernel：添加新 CUDA Kernel

这个 SKILL 是最长也是最详细的一个，以一个简单的 element-wise scale 操作（`scale(x, factor) = x * factor`）为例子，走了一遍在 FlashInfer 中添加新 kernel 的完整流程，一共分 10 步。

### Step 1：在 include/ 定义 CUDA Kernel

创建 `include/flashinfer/scale.cuh`，要求是**框架无关的**（不依赖 Torch 头文件），使用原始指针和模板来支持多种 dtype：

```cpp
namespace flashinfer {

template <typename T>
__global__ void ScaleKernel(const T* input, T* output, T factor, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = input[idx] * factor;
  }
}

template <typename T>
cudaError_t ScaleLauncher(const T* input, T* output, T factor, int n,
                          cudaStream_t stream = nullptr) {
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  ScaleKernel<T><<<blocks, threads, 0, stream>>>(input, output, factor, n);
  return cudaGetLastError();
}

}  // namespace flashinfer
```

### Step 2：在 csrc/ 创建 Launcher

创建 `csrc/scale.cu`，这一层负责把 TVM-FFI 的 `TensorView` 转换为原始指针，做输入校验，以及 dtype dispatch。这里只有 `csrc/` 目录下才允许引入 TVM FFI 的头文件。

文档详细介绍了 TVM-FFI 的错误处理机制：
- `TVM_FFI_THROW(ValueError) << "message"`：常规运行时错误
- `TVM_FFI_LOG_AND_THROW(InternalError) << "message"`：用于构造函数或初始化阶段，异常可能无法正常传播的情况

### Step 3：创建 TVM-FFI Binding

创建 `csrc/scale_jit_binding.cu`，用 `TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, scale_launcher)` 把 launcher 函数导出为 TVM-FFI 接口。

### Step 4：创建 JIT Generator

创建 `flashinfer/jit/scale.py`，负责 JIT 编译流程。对于简单的 kernel 不需要 Jinja 模板，直接把源文件复制到生成目录即可。URI 用于唯一标识模块配置。文档强调**不能往 package 目录写文件**。

这一步还介绍了 `CompilationContext` 机制，用于管理 CUDA 架构目标。可以通过 `supported_major_versions` 参数指定 kernel 支持的 SM 版本：

| 参数 | 支持的架构 | 使用场景 |
|------|-----------|---------|
| `None` | 所有可用 GPU | 通用 kernel |
| `[9, 10, 11, 12]` | SM90, SM100, SM110, SM120 | Hopper 及更新 |
| `[10, 11, 12]` | SM100, SM110, SM120 | Blackwell 及更新 |
| `[12]` | SM120 | 特定架构 |

也可以通过环境变量 `FLASHINFER_CUDA_ARCH_LIST` 手动覆盖。

### Step 5：创建 Python API

创建 `flashinfer/scale.py`，这是用户直接调用的接口。文档中展示了几个关键的设计模式：

- **`@functools.cache`**：缓存编译好的模块，避免重复编译
- **`@flashinfer_api`**：启用日志功能（和 debug-cuda-crash SKILL 中的机制一致）
- **Destination passing style**：输出 tensor 作为可选参数传入（`out: Optional[torch.Tensor] = None`），允许用户传入预分配的 buffer 以避免分配开销
- **`@backend_requirement` 和 `@supported_compute_capability` 装饰器**：做输入校验和 backend 选择

`@backend_requirement` 装饰器有三种使用模式：

1. **单 backend**：`backend_checks={}` 表示没有 backend 选择，只做通用校验
2. **多 backend**：在 `backend_checks` 字典中为每个 backend 注册独立的校验函数
3. **自动 backend 选择**：提供 `heuristic_func` 来根据输入参数自动选择最优 backend

装饰器还会给函数添加 `is_backend_supported()`、`is_compute_capability_supported()`、`has_backend()` 等辅助方法，以及 `skip_check=True` 参数用于性能关键路径跳过校验。

### Step 6-10：测试、AOT 注册、导出、运行、Benchmark

- **Step 6**：用 pytest 写单元测试，用 `pytest.mark.parametrize` 测多种 dtype 和 size 组合，对比 reference 实现。如果 kernel 有架构要求，用 `pytest.skip` 跳过不支持的 GPU。
- **Step 7**：在 `flashinfer/aot.py` 中注册，预编译常用配置，这样安装了 `flashinfer-jit-cache` 的用户可以跳过 JIT 编译。
- **Step 8**：在 `flashinfer/__init__.py` 中导出 API。
- **Step 9**：直接跑测试，kernel 会在首次使用时自动编译。
- **Step 10**：添加 benchmark 脚本，使用 `bench_gpu_time` 函数做性能测试。文档强调所有新 kernel 都应该有 benchmark。

### 最终文件清单

整个流程涉及的文件：

```
include/flashinfer/scale.cuh              # 新增：CUDA kernel 定义
csrc/scale.cu                              # 新增：Launcher
csrc/scale_jit_binding.cu                  # 新增：TVM-FFI binding
flashinfer/jit/scale.py                    # 新增：JIT generator
flashinfer/scale.py                        # 新增：Python API
flashinfer/__init__.py                     # 修改：导出 API
flashinfer/aot.py                          # 修改：注册 AOT
tests/test_scale.py                        # 新增：单元测试
benchmarks/bench_scale.py                  # 新增：Benchmark 脚本
```

## 0x4. 总结

FlashInfer 的这三个 SKILL 文件把项目中三个最核心的开发场景（调试、性能测试、添加新 kernel）的完整流程和最佳实践都文档化了。从实用角度看，`add-cuda-kernel` 这个 SKILL 对想要给 FlashInfer 贡献新 kernel 的开发者帮助最大，因为它把从 CUDA kernel 定义到 Python API 暴露的整条链路都走了一遍，涉及的 TVM-FFI、JIT 编译、装饰器模式等 FlashInfer 特有的机制如果没有这个文档的话，理解成本会高很多。`debug-cuda-crash` 和 `benchmark-kernel` 则更偏向日常使用，前者在排查 CUDA 错误时可能有用，后者在做性能对比时能提供准确的 kernel 级别计时。

个人感觉对于添加kernel的繁琐流程来说，做成SKILLS是很不错的，利好所有人。

