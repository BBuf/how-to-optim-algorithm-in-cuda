> 本文原文存放在：https://github.com/BBuf/how-to-optim-algorithm-in-cuda/blob/master/large-language-model-note/sglang%26lightllm/SGLang_JIT_%E6%8A%80%E6%9C%AF%E6%96%87%E6%A1%A3.md 

## 标题

SGLang JIT Kernel 介绍

## 0x0. 前言

之前在 SGLang 中如果要开发CUDA Kernel，需要经历在sgl-kernel中编写cuda kernel，导出pybind接口，注册算子，修改cmakelists等流程，开发完毕合入main之后还需要经历一次sgl-kernel的release才可以用起来，开发流程比较繁琐。且随着模板化的kernel越来越多，编译时间越来越长，换一台机器或者换一个Docker环境从头编译可能要编译1个小时，很影响软件开发的速度。为了加速迭代，我们在最近几个月基于 TVM-FFI 开始探索JIT kernel的开发方式，大大加速了Kernel的迭代速度。现在开发者不需要再付出大量的编译时间，Kernel的应用和SGLang源代码兼容，不用再经历繁琐的 sgl-kernel 发版流程就能快速把当前开发的Kernel用在LLM/Diffusion模型中获取性能收益。

本文面向希望理解/开发 SGLang JIT kernel 的开发者，梳理下JIT Kernel的机制，抽象，以及新增 Kernel 的流程：

- 机制：JIT kernel 如何从 Python 调用走到运行时编译、再到 CUDA kernel launch
- 抽象：`jit_kernel` 这套代码提供了哪些通用设施（校验、错误定位、kernel launch、向量化）
- 流程：以 `add_constant` 和 `fused_add_rmsnorm` 为例，如何组织 C++/CUDA 代码、Python 封装、以及 test/benchmark

Q&A：

- 社区发起了 sgl-kernel 往 JIT Kernel 的迁移计划，需要说明的是这个计划在 sgl-kernel 中只会移除几个模板非常大的 gemm kernel 达到减少 sgl-kernel wheel包体积的效果，而并非移除 sgl-kernel 已有的所有 AOT kernel，对 sgl-kernel 的 AOT kernel 有依赖的用户或者项目在这个过程中基本不会受到影响。如果你对迁移计划感兴趣或者想做贡献欢迎查看这个ISSUE：https://github.com/sgl-project/sglang/issues/17865 
- JIT Kernel的基础设施特别是 TVM-FFI 对接部分主要由 mini-sglang 的作者 https://github.com/DarkSharpness lead 完成，持续维护中
- CUTE Dsl环境在SGLang环境中已登陆，也欢迎大家用 CuteDSL 实现高性能kernel然后存放到 JIT Kernel 下，比如：https://github.com/sgl-project/sglang/pull/14717

## 0x1. 环境设置

推荐使用 `clangd` 作为语言服务器进行 JIT kernel 开发。Ubuntu/Debian 可从 [apt.llvm.org](https://apt.llvm.org/) 下载 clangd。VS Code 用户建议安装 `clangd` 扩展。

所有 JIT 相关文件位于 `python/sglang/jit_kernel`。与提前编译（AOT）的 `sgl-kernel` 不同，JIT kernel 在运行时编译，无法生成静态的 `compile_commands.json`。

为了让 `clangd` 支持代码补全，在当前目录执行：

```bash
python -m sglang.jit_kernel
```

生成 `.clangd` 配置文件后，重启 clangd 语言服务器即可识别所有 JIT kernel 文件。

## 0x2. 目录结构

```
python/sglang/jit_kernel/
├── __main__.py                         # 生成 .clangd（clangd 补全/跳转）
├── utils.py                            # load_jit/cache_once/make_cpp_args
├── csrc/
│   ├── add_constant.cuh                # 简单示例
│   └── elementwise/
│       └── fused_add_rmsnorm.cuh       # 进阶示例
├── include/sgl_kernel/                 # C++/CUDA 通用抽象
│   ├── utils.h                         # DebugInfo/RuntimeCheck/PanicError/irange
│   ├── utils.cuh                       # LaunchKernel/PDL/RuntimeDeviceCheck
│   ├── tensor.h                        # TensorMatcher（shape/stride/dtype/device 校验）
│   ├── type.cuh                        # dtype_trait/cast/packed_t
│   ├── vec.cuh                         # AlignedVector（最高 32B=256-bit 向量化）
│   └── runtime.cuh                     # runtime: get_cc_major/get_sm_count 等
├── add_constant.py                     # Python 接口（add_constant）
├── norm.py                             # Python 接口（fused_add_rmsnorm）
├── tests/
│   ├── test_add_constant.py
│   └── test_fused_add_rmsnorm.py
└── benchmark/
    └── bench_fused_add_rmsnorm.py
```

C++ 实现位于 `python/sglang/jit_kernel/csrc`，可复用函数位于 `python/sglang/jit_kernel/include`。

Python 接口定义在 `python/sglang/jit_kernel`。使用 [tvm-ffi](https://github.com/apache/tvm-ffi) 进行高效的外语言绑定。通常 `tvm::ffi::TensorView` 足够从 Python 传递 PyTorch Tensor。

## 0x3. JIT 链路：从 Python 调用到 CUDA kernel launch

对应实现位于 `python/sglang/jit_kernel/utils.py`。

### cache_once：JIT module 的缓存方式

使用自定义的 `cache_once` 而非 `functools.lru_cache`，因为后者与 `torch.compile` 不兼容。功能：

- 用（参数 → 编译产物）做缓存
- 同一套参数只触发一次编译，后续复用已加载的 module

### make_cpp_args：把 Python 参数变成 C++ 模板参数

`make_cpp_args` 会把 `int/float/bool/torch.dtype` 转成 C++ 模板参数字符串，比如 dtype 会映射到 `bf16_t/fp16_t/fp32_t` 这类类型别名。

### load_jit：运行时编译 + 导出符号 + 加载

`load_jit` 核心流程三步：

1. 把 `cuda_files=["add_constant.cuh"]` 变成 `#include "绝对路径"` 拼进编译单元
2. 用 `TVM_FFI_DLL_EXPORT_TYPED_FUNC(name, (symbol))` 导出 C++ 符号
3. 调 `tvm_ffi.cpp.load_inline` 编译并加载，返回一个 `Module`

`args` 除了用于模板实例化，还会进入 module 的唯一名字（`sgl_kernel_jit_${args_joined}`）。不同变体的 `args` 必须区分，否则缓存会冲突。

通过 `cuda_wrappers=[("func", "cpp_func")]` 导出 C++ 函数，在 Python 中作为 `module.func` 调用。

## 0x4. 通用抽象：写 kernel 时尽量复用的基础设施

`python/sglang/jit_kernel/include/sgl_kernel/` 提供的基础设施，收敛重复、易错、影响排障效率的样板代码。

### irange（utils.h）：整数范围迭代

类似 PyTorch，提供 `irange` 函数表示整数范围。

```cpp
#include <sgl_kernel/utils.h>

void test() {
  for (auto i : host::irange(100)) {        // [0, 100)
    // do something
  }
  for (auto i : host::irange(0, 100)) {     // [0, 100)
    // do something
  }
}
```

### RuntimeCheck / RuntimeDeviceCheck（utils.h & utils.cuh）

JIT kernel 调试成本通常来自参数传递错误（shape/dtype/stride/device 不匹配）。

- `RuntimeCheck`：条件不满足时抛异常，附带文件名和行号，支持可选参数用于错误报告
- `RuntimeDeviceCheck`：验证最近一次 kernel launch 的状态

```cpp
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

void test(int hidden_size, int elements_in_vec) {
  using namespace host;
  
  RuntimeCheck(hidden_size % elements_in_vec == 0,
               "hidden_size=", hidden_size,
               " is not aligned to elements_in_vec=", elements_in_vec);
  
  RuntimeDeviceCheck();
  RuntimeDeviceCheck(cudaGetLastError());  // 显式传入 cudaError_t
}
```

建议在 host 入口（`::run`）里做此类检查，比 kernel launch 后 silent wrong 更易定位。

### TensorMatcher（tensor.h）

`TensorMatcher` 用于入口校验，减少遗漏并提供完整报错信息。

特性：

- **默认 contiguous**：不写 `with_strides(...)` 则要求 `view.is_contiguous()`
- **错误信息完整**：失败时打印实际 tensor 的 `shape/strides/dtype/device` 和 root cause
- **Symbolic 变量**：必须在所有验证中解析为相同值，用 `.unwrap()` 获取匹配值
- **灵活匹配**：传 `-1` 给 size 或 stride 匹配任意值

配置：

- `with_strides`：省略则期望 tensor contiguous
- `with_dtype`：模板参数限制允许的数据类型
- `with_device`：模板参数限制允许的设备类型
- `with_xxx` 传入的值强制相等检查

```cpp
#include <sgl_kernel/tensor.h>

using namespace host;

void check(const tvm::ffi::TensorView input,
           const tvm::ffi::TensorView residual,
           const tvm::ffi::TensorView weight) {
  auto N = SymbolicSize{"num_tokens"};
  auto D = SymbolicSize{"hidden_size"};
  auto dtype = SymbolicDType{};
  auto device = SymbolicDevice{};

  TensorMatcher({N, D})                       // input: [N, D]
      .with_strides({D, 1})                   // 要求最后一维 contiguous
      .with_dtype<bf16_t, fp16_t>(dtype)      // 限制允许 dtype
      .with_device<kDLCUDA, kDLCPU>(device)   // 限制允许 device
      .verify(input);

  TensorMatcher({N, D})                       // residual: [N, D]
      .with_strides({D, 1})
      .with_dtype<bf16_t, fp16_t>(dtype)      // dtype 必须与 input 一致
      .with_device<kDLCUDA, kDLCPU>(device)   // device 必须与 input 一致
      .verify(residual);

  TensorMatcher({D})                          // weight: [D]
      .with_dtype<bf16_t, fp16_t>(dtype)
      .with_device<kDLCUDA, kDLCPU>(device)
      .verify(weight);
  
  size_t num_tokens = N.unwrap();
  size_t hidden_size = D.unwrap();
}
```

注意：`TensorMatcher` 是临时表达式，不要存储在变量中。在 `TensorMatcher` 链末尾添加 `//` 强制正确缩进。

### LaunchKernel（utils.cuh）

`LaunchKernel` 统一处理以下细节：

- 从 `DLDevice` 解析出当前 stream（与 PyTorch stream 语义对齐）
- 用 `cudaLaunchKernelEx` 发射
- 自动做 `RuntimeDeviceCheck(...)`（等价于检查 `cudaGetLastError()`）
- 可选 `.enable_pdl(true/false)` 控制 programmatic stream serialization 属性

`LaunchKernel::resolve_device` 从 PyTorch 获取当前 `cudaStream`。

```cpp
#include <sgl_kernel/utils.cuh>
#include <dlpack/dlpack.h>

using namespace host;

__global__ void kernel(float* x) { /* ... */ }

void test() {
  const auto num_blocks = 1;
  const auto num_threads = 32;
  const auto dynamic_smem = 0;

  DLDevice dev;  // 假设已正确初始化
  
  // 方式1：直接从 DLDevice launch
  LaunchKernel(num_blocks, num_threads, dev)(kernel, x);
  
  // 方式2：显式获取 stream 后 launch
  cudaStream_t stream = LaunchKernel::resolve_device(dev);
  LaunchKernel(num_blocks, num_threads, stream, dynamic_smem)(kernel, x);
}
```

### AlignedVector（vec.cuh）

`AlignedVector<T, N>` 是对齐的 POD 封装，用于向量化访存。`fused_add_rmsnorm` 在 B200 上走 32B 路径时依赖它。

核心约束：`sizeof(T) * N <= 32`，即最大 32 字节（256-bit）。

```cpp
#include <sgl_kernel/vec.cuh>

using device::AlignedVector;

__global__ void vec_ldg_stg(const half* src, half* dst) {
  // 32B: half(2B) * 16 = 32B
  using vec_t = AlignedVector<half, 16>;
  vec_t v;
  v.load(src, /*offset=*/blockIdx.x);
  v.store(dst, /*offset=*/blockIdx.x);
}
```

向量化前提：**地址对齐** 与 **长度可整除**。`fused_add_rmsnorm` 的 host 入口做了相应检查：

- `elements_in_vec = max_vec_size_byte / sizeof(DType)`
- `RuntimeCheck(hidden_size % elements_in_vec == 0, ...)`

## 0x5. 完整示例：add_constant（端到端流程）

`add_constant` kernel 对输入 tensor 的每个元素加上一个常量。

Python 接口概念：

```python
def add_constant(src: torch.Tensor, c: int):
    return src + c
```

### STEP 1：编写 C++ kernel

创建文件 `python/sglang/jit_kernel/csrc/add_constant.cuh`，将常量作为模板参数传递。

```cpp
#include <sgl_kernel/tensor.h>   // TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.cuh>  // LaunchKernel
#include <sgl_kernel/utils.h>    // div_ceil, RuntimeCheck

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstddef>
#include <cstdint>

namespace {

template <int32_t kConstant>
__global__ void add_constant_kernel(int32_t* dst, const int32_t* src, size_t length) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < length) {
    dst[idx] = src[idx] + kConstant;
  }
}

constexpr size_t kBlockSize = 256;

template <int32_t kConstant>
void add_constant(tvm::ffi::TensorView dst, tvm::ffi::TensorView src) {
  using namespace host;

  // 1. 验证输入 tensors
  SymbolicSize N = {"num_elements"};
  SymbolicDevice device_;
  TensorMatcher({N})                  // 1D tensor，必须 contiguous
      .with_dtype<int32_t>()          // 必须是 int32
      .with_device<kDLCUDA>(device_)  // 必须在 CUDA 设备上
      .verify(dst)                    // 检查 tensor dst
      .verify(src);                   // 检查 tensor src

  // 2. 提取参数，准备 kernel launch
  const size_t num_elements = N.unwrap();
  const size_t grid_size = div_ceil(num_elements, kBlockSize);
  const DLDevice device = device_.unwrap();
  
  RuntimeCheck(num_elements > 0, 
               "We only support non-empty tensors, got num_elements = ", num_elements);

  // 3. Launch kernel，自动检查错误码
  LaunchKernel(grid_size, kBlockSize, device)(
      add_constant_kernel<kConstant>,
      static_cast<int32_t*>(dst.data_ptr()),
      static_cast<int32_t*>(src.data_ptr()),
      num_elements);
}

}  // namespace
```

### STEP 2：创建 Python 接口

创建文件 `python/sglang/jit_kernel/add_constant.py`。

```python
from __future__ import annotations
from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_add_constant_module(constant: int) -> Module:
    args = make_cpp_args(constant)
    return load_jit(
        "add_constant",
        *args,
        cuda_files=["add_constant.cuh"],
        cuda_wrappers=[("add_constant", f"add_constant<{args}>")],
    )


def add_constant(src: torch.Tensor, constant: int) -> torch.Tensor:
    dst = torch.empty_like(src)
    module = _jit_add_constant_module(constant)
    module.add_constant(dst, src)
    return dst
```

### STEP 3：使用 kernel

```python
from sglang.jit_kernel.add_constant import add_constant

x = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device='cuda')
y = add_constant(x, 10)
# y = tensor([11, 12, 13, 14], device='cuda:0')
```

完整示例参考 `python/sglang/jit_kernel/tests/test_add_constant.py`。

## 0x6. 进阶示例：fused_add_rmsnorm

### Motivation

和 FlashInfer 的版本相比，主要两点：

1. **避免把 `inp+res` 存/取到 shared memory**
2. **在 B200 上用 256-bit LDG（32B 向量化）**

### Modifications

引入时改动的文件：

- `python/sglang/jit_kernel/norm.py`
- `python/sglang/jit_kernel/csrc/elementwise/fused_add_rmsnorm.cuh`
- `python/sglang/jit_kernel/tests/test_fused_add_rmsnorm.py`
- `python/sglang/jit_kernel/include/sgl_kernel/vec.cuh`

### Python 封装

编译并缓存 module，提供 Python 函数调用导出的符号（`python/sglang/jit_kernel/norm.py`）。

```python
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_fused_add_rmsnorm_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "fused_add_rmsnorm",
        *args,
        cuda_files=["elementwise/fused_add_rmsnorm.cuh"],
        cuda_wrappers=[("fused_add_rmsnorm", f"FusedAddRMSNormKernel<{args}>::run")],
    )


def fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> None:
    module = _jit_fused_add_rmsnorm_module(input.dtype)
    module.fused_add_rmsnorm(input, residual, weight, eps)
```

说明：

- `cuda_files` 指向 `csrc/` 下的实现文件（相对 `csrc/` 路径）
- `cuda_wrappers` 把 `FusedAddRMSNormKernel<...>::run` 导出成 Python 侧的 `module.fused_add_rmsnorm`
- `@cache_once` 确保同一 dtype 的 module 只编译/加载一次

### CUDA/C++ 实现

实现文件：`python/sglang/jit_kernel/csrc/elementwise/fused_add_rmsnorm.cuh`。

结构分为两层：

1. **host 入口**：`FusedAddRMSNormKernel<DType>::run`  
   做 `TensorMatcher` 校验、根据架构选 16B/32B 向量化、算 threads、最后用 `LaunchKernel(...).enable_pdl(false)(...)` 发射。
2. **device kernel**：`fused_add_rmsnorm_reg_kernel<DType, 16/32>`  
   核心操作：
   - `residual <- input + residual`（in-place 写回）
   - 用同一份 `inp+res` 计算平方和得到 `rsqrt`，再做 RMSNorm 写回 `input`（in-place）

避免 shared memory 往返：`inp+res` 直接写回 residual，同时在寄存器路径上继续完成 RMSNorm 输出，中间量未写入 shared。

B200 使用 256-bit 向量化：host 入口通过 `get_cc_major` 判断架构，`cc_major >= 10` 时选择 32B 向量化路径，底层向量化依赖 `device::AlignedVector`。

## 0x7. 总结

本文梳理了 SGLang JIT kernel 的机制、抽象和开发流程。希望对想在SGLang中开发JIT kernel的开发者有所帮助。


