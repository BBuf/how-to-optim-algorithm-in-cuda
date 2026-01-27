# CuTe DSL 4.3.5 文档


# 概览

CUTLASS 4.x 旨在弥合 CUDA Kernel 开发中“生产力”和“性能”之间的鸿沟。它通过为强大的 CUTLASS C++ 模板库提供基于 Python 的 DSL，使得在 NVIDIA GPU 上进行高性能线性代数开发时能够更快迭代、更容易做原型验证，并显著降低学习门槛。

总体而言，我们将 CUTLASS DSLs 设想为一套领域特定语言（DSL）家族。在 4.0 版本中，我们首先发布其中的 CuTe DSL。它是一种低层次（low-level）的编程模型，与 CuTe 的 C++ 抽象保持完全一致——对外暴露了诸如布局（layout）、张量（tensor）、硬件原子操作（hardware atoms）等核心概念，并允许对硬件线程与数据层次结构进行完全控制。

# 为什么需要 CUTLASS DSLs？

尽管 CUTLASS 通过 C++ 模板抽象提供了卓越的性能，但其复杂性也会给许多开发者带来挑战。CUTLASS 4.x 通过以下方式来解决这一问题：

- **简化元编程**：用 Python 进行元编程相比 C++ 更直观

- **加速迭代**：使用熟悉的 Python 语法快速原型开发，并具备极快的编译速度

- **降低门槛**：降低 GPU 编程概念的学习曲线，并保持 CuTe C++ 与 DSL 在概念上的一致性

- **保持性能**：生成的代码会复用经过优化的 CUTLASS 基元（primitives）

学生可以在不被 C++ 模板复杂性困扰的情况下学习 GPU 编程概念。研究人员和性能工程师则可以在进入生产级实现之前，快速探索算法、做原型验证，并对 kernel 进行调参与优化。

# 关键概念与方法

CUTLASS DSLs 会将 Python 代码翻译为自定义的中间表示（IR），然后借助 MLIR 与 ptxas 在运行时进行即时编译（JIT），生成优化后的 CUDA kernels。

## CuTe DSL 的核心抽象

- **Layouts（布局）** – 描述数据在内存中以及在线程之间是如何组织的。

- **Tensors（张量）** – 将数据指针或迭代器与布局元数据结合起来。

- **Atoms（原子）** – 表示基础硬件操作，例如矩阵乘加（MMA）或内存拷贝。

- **Tiled Operations（分块操作）** – 定义 atoms 如何在 thread block 与 warp 范围内应用（例如：`TiledMma`、`TiledCopy`）。

关于 CuTe 抽象的更多内容，请参考 CuTe C++ 库文档(https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/00_quickstart.md)。

**Python 风格的 Kernel 表达**

开发者可以使用熟悉的 Python 语法与控制流来表达 kernel 的逻辑、数据搬运以及计算过程。

这些 DSL 让你能用简洁的 Python 代码更容易地表达循环分块（tiling）、线程策略以及数据变换。

**JIT 编译**

Python 编写的 kernels 会在运行时通过 MLIR 基础设施与 NVIDIA 的 `ptxas` 工具链编译为 CUDA 设备端代码，从而支持快速迭代与交互式调试。

# 与 CUTLASS C++ 的关系

CUTLASS DSLs 并不是 CUTLASS C++ 库或其 2.x / 3.x API 的替代品。相反，它的目标是成为一个高生产力的 kernel 编写框架，并与 CUTLASS 3.x C++ API（例如 CuTe、流水线（pipelines）、调度器（schedulers）等）共享同一套概念体系。

- **性能**：生成的 kernels 目标是在性能上对齐 CUTLASS C++ kernels；但由于 CUTLASS C++ 多年来积累的一些优化可能尚未体现在 DSL 示例中，因此仍可能存在一定的性能差距。

- **库形态**：CUTLASS DSLs 目前并未提供像 CUTLASS C++ 那样完整的 GEMM/Conv 自动调优 profiler 或库接口。它更聚焦于生成并自动调优单个 kernel 实例（例如通过探索 tile size），以及与支持自动调优的深度学习框架进行原生集成。

# 快速开始

- 快速开始指南(https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html)- 初始配置与安装。

- CuTe DSL(https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html) – 概述使用 CuTe DSL 的典型开发流程与工作流。

- CuTe DSL API(https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api.html) – 参考完整的 API 文档。

- 限制(https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/limitations.html) – 了解当前 CuTe DSL 的约束以及与 C++ 的差异。

- 常见问题(https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/faqs.html) – 常见问题与已知问题。

# 当前状态与路线图
CuTe DSL 目前处于公开 Beta 阶段，并在积极迭代中。随着系统的持续改进，接口与功能可能会发生变化。

# 即将到来的里程碑

- 计划于 **2025 年夏季**公开发布
- 扩展对更多数据类型与更多 kernel 类型的支持
- 可用性改进：更好的错误信息、调试工具以及更精简的 API
- 更广泛地集成 CUTLASS 的基元与特性

有关已知问题与解决方法，请参考 Limitations 与 FAQs。

# 社区与反馈

我们欢迎开发者社区的贡献与反馈！

你可以：

- 通过 GitHub Issues 页面提交 bug 报告或功能需求(https://github.com/NVIDIA/cutlass/issues)
- 加入 Discord 上的 CUTLASS 社区提问并分享想法(https://discord.com/channels/1019361803752456192/1150868614921064590)
- 为 DSLs 贡献示例、教程或增强改进
- 报告不清晰或缺失的文档
- 提议支持更多数据类型或 kernel 变体
- 通过给 GitHub issue 点赞，帮助我们对路线图功能进行优先级排序

感谢你帮助塑造 CUTLASS DSLs 的未来！

# 功能支持

CUTLASS DSL 4.0 版本仅支持 **Python 3.12**。它与 CUDA Toolkit 12.9(https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) 具有相同的驱动要求。具体来说，驱动版本必须为 575.51.03 或更高。

目前仅支持 Linux x86_64。未来版本会增加对更多平台的支持。

# 支持的 MMA 操作

**NVIDIA Ampere 架构：**

- FP16 / BF16 Tensor Core 指令

**NVIDIA Hopper 架构：**

- FP16 / BF16
- FP8

**NVIDIA Blackwell 架构：**

- FP16 / BF16
- TF32
- I8
- F8

# 主要限制

关于当前的约束与不支持的特性，请参考 Limitations(https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/limitations.html) 小节。

# 快速开始指南

CUTLASS DSL 4.0 版本目前仅支持 Linux 与 **Python 3.12**。要安装 CUTLASS DSLs（目前仅包含 CuTe DSL），请使用以下命令。

## 安装

为确保与 GitHub 上的示例与代码保持兼容，请使用仓库中对应 commit 的 requirements.txt 文件。

```shell
git clone https://github.com/NVIDIA/cutlass.git
pip install -r cutlass/python/CuTeDSL/requirements.txt
```

如果你只是想试用“已知的最近稳定版本”的 CUTLASS DSL（可能与最新示例/代码不兼容），请运行：

```shell
pip install nvidia-cutlass-dsl
```

`nvidia-cutlass-dsl` 的 wheel 包含生成 GPU kernels 所需的一切组件。它要求的 NVIDIA 驱动版本与 CUDA Toolkit 12.9 一致。

# 推荐依赖

为了运行示例并开始开发，我们建议安装：
```shell
pip install torch jupyter
```

# Jupyter Notebook 推荐的 Python 环境变量

我们建议在运行 jupyter notebook 时设置以下环境变量。

```shell
export PYTHONUNBUFFERED=1
```

# CuTe DSL > 介绍

## 概览

CuTe DSL 是一种基于 Python 的领域特定语言（DSL），用于对数值计算与面向 GPU 的代码进行动态编译。其主要目标包括：

- **与 CuTe C++ 保持一致**，允许用户在完全控制硬件的前提下表达 GPU kernels。
- **JIT 编译**，同时支持 host 与 GPU 端执行。
- **DLPack 集成**，实现与框架（如 PyTorch、JAX）的无缝互操作。
- **JIT 缓存**，使得对同一函数的重复调用可以复用缓存的 IR 模块。
- **原生类型与类型推断**，减少样板代码并提升性能。
- **可选的更低层控制**，提供对 GPU 后端或专用 IR dialect 的直接访问。

## 装饰器

CuTe DSL 提供了两个主要的 Python 装饰器，用于通过动态编译生成优化代码：

1. `@jit` — Host 侧 JIT 编译函数
2. `@kernel` — GPU kernel 函数

这两个装饰器都可以选择启用 `preprocessor`，它会自动将 Python 控制流（循环、条件分支）展开为底层 IR 可消费的操作。

### `@jit`

声明 JIT 编译函数，可从 Python 调用，也可从其他 CuTe DSL 函数中调用。

#### 装饰器参数

- `preprocessor`:
  - `True`（默认）— 自动将 Python 控制流（例如循环、if 语句）翻译为 IR 操作。
  - `False` — 不进行自动展开；Python 控制流需要手动处理或避免使用。

#### 调用端参数

- `no_cache`:
  - `True` — 禁用 JIT 缓存，每次调用都强制重新编译。
  - `False`（默认）— 启用缓存，使后续调用更快。


### `@kernel`

定义 GPU kernel 函数，通过动态编译生成专用的 GPU 符号。

#### 装饰器参数：

- `preprocessor`:
    - `True`（默认）— 自动将 Python 循环/if 展开为 GPU 兼容的 IR 操作。
    - `False` — 期望手动实现或更简化的 kernel 实现。


#### Kernel 启动参数

- `grid`：以整数列表指定 grid 大小。
- `block`：以整数列表指定 block 大小。
- `cluster`：以整数列表指定 cluster 大小。
- `smem`：以字节数（整数）指定 shared memory 大小。

## 调用约定

| 调用方 | 被调用方 | 是否允许 | 编译/运行时 |
| --- | --- | --- | --- |
| Python function | `@jit` | Yes | DSL runtime |
| Python function | `@kernel` | No | N/A (error raised) |
| `@jit` | `@jit` | Yes | Compile-time call, inlined |
| `@jit` | Python function | Yes | Compile-time call, inlined |
| `@jit` | `@kernel` | Yes | Dynamic call via GPU driver or runtime |
 | `@kernel` | `@jit` | Yes | Compile-time call, inlined |
 | `@kernel` | Python function | Yes | Compile-time call, inlined |
 | `@kernel` | `@kernel` | No | N/A (error raised) |
 
 

# 端到端代码生成

## 1. 将 Python 转换为中间表示（IR）的技术

### 1.1 AST 重写

函数的抽象语法树（AST）会在**执行之前**被分析。Python 控制流（`for`/`while`、`if`/`else`）以及内置函数会被转换为结构化的中间表示（IR）构造。而每个区域内部的计算在此阶段保持不变。

**优点**

- 能看到整个程序，因此可以保留每一个分支与循环。
- 保持循环结构完整，有利于做分块（tiling）、向量化（vectorisation）或 GPU 线程映射等优化。

**缺点**

- 需要一个重写器能够理解的、定义良好的 Python 子集。

### 1.2 Tracing（跟踪）

被装饰的函数会使用代理参数执行一次；重载的运算符会记录实际运行的每一个张量操作，并生成一份扁平的 trace，随后被 lower 为中间表示（IR）。

**优点**

- 编译延迟几乎为零，非常适合直线型（straight-line）的算术计算。
- 无需解析 Python 源码，因此支持许多动态 Python 特性（而 Python 的动态特性非常多）。

**缺点**

- 未被执行到的分支会“消失”，因此生成的 kernel 对其他输入可能是不正确的。
- 循环会被扁平化为 tracing 时观察到的迭代次数。
- 依赖数据的控制流会被固定为 trace 时的单一路径。

## 2. CuTe DSL 的代码生成模式
CuTe 的 Python 前端将上述技术组合为两种互斥模式，可通过 `@jit` 装饰器的 `preprocessor` 标志进行选择：

1. Tracing 模式 `@jit(preprocess=False)` — 仅 tracing。这是编译路径最快的一种方式，仅推荐用于能够保证为“直线型算术”的 kernel。它会受到上一节列出的 tracing 局限性影响。
2. 预处理器模式（**默认**）`@jit(preprocess=True)` — **AST 重写 + tracing**。AST 阶段会捕获每一个循环与分支，避免纯 tracing 带来的正确性与优化问题；随后 tracing 再补齐算术计算。这条混合的“预处理器（preprocessor）”流水线是 CuTe DSL 独有的，专门用于克服上述缺点。
 
 
 ![](https://files.mdnice.com/user/59/680a3096-98f3-4989-b144-29443a86de27.jpg)
 
 图 1：左：tracing 模式只记录实际执行到的路径。右：预处理器模式会在 tracing 算术之前，为每一个分支与循环生成结构化的中间表示（IR）。
 
 ## 为什么仅 Tracing 不足以处理控制流
 
- **分支丢失** — `if`/`else` 中未被执行到的一侧不会被 lower。
 - **循环被“展开/扁平化”** — 循环会被压平成 tracing 时观察到的迭代次数，从而破坏并行映射与分块（tiling）所需的结构。
 - **数据依赖路径被冻结** — 依赖张量数值的控制流会在 tracing 时被冻结为单一路径。
 
 预处理器模式通过“先 lower 控制流，再把算术交给 tracer”来修复上述问题。
 
 

# 控制流

## 概览

CuTe DSL 会遍历 Python 的 AST，并将其中的每一种控制流结构转换为结构化的中间表示（IR）。因此你可以像写普通 Python 一样写循环与分支，而编译器会逐条语句决定：

- 如果它是原生 Python 控制流，则**在编译期求值（evaluate at compile time）**；或
- 当控制流被标记为动态时，**生成中间表示（IR）（emit intermediate representation）**。

将中间表示（IR）的值传给原生 Python 控制流会导致错误。

关于整体流水线的高层讨论，请参阅 code-generation overview(https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_code_generation.html)。

## for 循环

CuTe DSL 能识别 `for` 循环中三类 range：

- `range` — Python 内置函数，总是会被 lower 为中间表示（IR）
- `cutlass.range` — 与 Python 内置 `range` 类似，但支持更高级的展开（unrolling）与流水线（pipelining）控制
- `cutlass.range_constexpr` — 在编译期展开（unrolled at compile time）

### `range(...)` / `cutlass.range(...)`

当你希望无论输入是否为 Python 值，都在生成的中间表示（IR）中保留循环结构时使用。

### `cutlass.range_constexpr(...)`

在 Python 解释器中运行，并在代码生成前完全展开。所有循环索引都必须是 `Constexpr`（编译期的 Python 值）。

示例：

```python
@cute.jit
def control_flow_examples(bound: cutlass.Int32):
    n = 10

    # ✅ 这是 Python 循环，在编译期求值。
    for i in cutlass.range_constexpr(n):
        cute.printf("%d\\n", i)

    # ✅ 这是动态循环，即使 bound 是 Python 值也是如此。
    for i in range(n):
        cute.printf("%d\\n", i)

    # ❌ 循环上界是动态值，不能用于 Python 循环。
    # 应该改用 `range`。
    for i in cutlass.range_constexpr(bound):
        cute.printf("%d\\n", i)

    # ✅ 这是动态循环，会生成 IR 循环。
    for i in range(bound):
        cute.printf("%d\\n", i)

    # ✅ 这是动态循环，会生成带展开（unrolling）的 IR 循环
    for i in cutlass.range(bound, unroll=2):
        cute.printf("%d\\n", i)
```

## 软件流水线（Software Pipelining）

软件流水线是一种用于优化循环的技术。通常需要编写一个预取（prefetch）循环和一个主循环。

```python
@cute.jit
def example():
    ...
    # 构建一个环形缓冲区
    buffer = ...

    # 预取循环
    for i in range(prefetch_stages):
        cute.copy(atom, gmem[i], buffer[i], ...)

    # 主循环
    for i in range(bound):
        if i + prefetch_stages < bound:
            cute.copy(atom, gmem[i + prefetch_stages], buffer[(i + prefetch_stages) % total_stages], ...)

        use(buffer[i % total_stages])

    ...
```

这段代码写起来和调参都会比较繁琐。CuTe DSL 提供了循环属性来让编译器帮你完成这一点。

```python
@cute.jit
def example():
    ...
    # 构建一个环形缓冲区
    buffer = ...

    for i in cutlass.range(bound, prefetch_stages=prefetch_stages):
        # 编译器会自动处理流水线：
        # - 为初始阶段生成预取循环
        # - 在主循环中，一边使用当前数据，一边预取未来数据
        cute.copy(atom, gmem[i], buffer[i % total_stages], ...)
        use(buffer[i % total_stages])  # 使用的是前几次迭代预取的数据

    ...
```

编译器会自动生成一个迭代次数为 `prefetch_stages` 的预取循环，以及与之对应的主循环。

该特性仍处于实验阶段，仅支持 sm90 及以上架构。

## If-Else 语句

支持标准 Python 的 `if`/`elif`/`else`。

- **未加注解的谓词（predicate）** → 会被 lower 为中间表示（IR）。
- **用 `cutlass.const_expr` 注解的谓词** → 在编译期求值。

示例：

```python
@cute.jit
def main(const_var: cutlass.Constexpr, dynamic_var: cutlass.Int32):
    # ✅ 这是 Python 分支，在编译期求值。
    if cutlass.const_expr(const_var):
        cute.printf("Const branch\n")
    else:
        cute.printf("Const else\n")

    # ✅ 这是动态分支，会生成 IR 分支。
    if dynamic_var == 10:
        cute.printf("Dynamic True\n")
    else:
        cute.printf("Dynamic False\n")

    # ❌ 不允许把动态值用于 `cutlass.const_expr`。
    if cutlass.const_expr(dynamic_var == 10):
        cute.printf("Bound is 10\n")
```

## While 循环

支持标准 Python 的 `while`。

- **未加注解的条件（condition）** → 会被 lower 为中间表示（IR）。
- **用 `cutlass.const_expr` 注解的条件** → 在编译期求值。

示例：

```python
@cute.jit
def main(dynamic_var: cutlass.Int32):
    n = 0

    # ✅ 这是 Python while 循环，在编译期求值。
    while cutlass.const_expr(n < 10):
        cute.printf("Const branch\n")
        n += 1

    # ✅ 这是动态 while 循环，会生成 IR while 循环。
    while dynamic_var == 10:
        cute.printf("Dynamic True\n")
        n += 1

    # ❌ 不允许把动态值用于 `cutlass.const_expr`。
    while cutlass.const_expr(n < dynamic_var):
        n += 1
```

## 编译期元编程（Compile-Time Metaprogramming）

将编译期构造与普通 CuTe DSL 代码混合使用，可以在不引入运行时开销的前提下生成特化（specialised）的 kernel。例如，可以通过编译期 flag 来启用/禁用可选的 ReLU epilogue：

```python
@cute.kernel
def gemm(..., do_relu: cutlass.Constexpr):
    # GEMM 主体计算
    ...

    if cutlass.const_expr(do_relu):  # 编译期 guard
        # 只有当 do_relu 为 True 时才会生成 ReLU 代码
        ...
```

```python
gemm(..., False)  # 生成的 |IR| 中会省略 ReLU
gemm(..., True)   # 生成的 |IR| 中会包含 ReLU
```

## 动态控制流的限制

- 目前尚不支持在控制流体内提前退出：`break`、`continue`、`pass`，或从控制流体内抛出异常。
- 控制流体内的操作只有在该区域启用 tracing 时才会被 trace。
- 控制流体内产生的值不能在控制流之外使用。
- 不允许在控制流体内改变变量类型。

**示例**：

```python
@cute.jit
def control_flow_negative_examples(predicate: cutlass.Boolean):
    n = 10

    # ❌ 该循环是动态的，不允许提前退出。
    for i in range(n):
        if i == 5:
            break  # 提前退出

    if predicate:
        val = 10

    # ❌ 不允许从控制流体内 return。
    return

    # ❌ 不允许从控制流体内抛异常。
    raise ValueError("This is not allowed")

    # ❌ 不允许在控制流体内使用 pass。
    pass

    # ❌ val 在动态 if 之外不可用
    cute.printf("%d\n", val)

    if predicate:
        # ❌ 不允许在控制流体内改变变量类型。
        n = 10.0
```

# JIT 函数参数生成

## 概述

当你使用 `@jit` 或 `@kernel` 装饰器来定义一个 JIT 编译函数时，函数的参数会被 trace，用来确定该 JIT 函数的签名（signature）。CuTe DSL 提供了一种更 Pythonic 的方式，让你像写普通 Python 一样书写 JIT 函数的参数声明，其余工作由 CuTe DSL 自动完成。

具体而言，CuTe DSL 在生成 JIT 函数参数时遵循以下规则：

- 默认情况下，JIT 函数参数会被认为是 **动态参数（dynamic arguments）**。
- 如果某个参数显式用 `cutlass.Constexpr` 进行类型标注，则会被视为 **编译期常量（compile-time constant）**。
- 如果提供了类型标注，CuTe DSL 会在编译期对参数类型进行校验，以保证 **类型安全（type safety）**。
- CuTe DSL 提供了可在运行时检查的协议（`JitArgument` 与 `DynamicExpression`），用于为自定义类型生成 JIT 函数参数。

下文将分别对以上各点进行更详细的说明。

## 静态参数 vs. 动态参数

CuTe DSL 支持 JIT 函数的静态参数与动态参数。

1. **静态参数** 保存编译期就已知的值。它不会被包含在生成的 JIT 函数签名中。
2. **动态参数** 保存只在运行期才知道的值。

默认情况下，CuTe DSL 会假设参数是动态参数，并尝试根据调用点（call-site）的实参类型推断形参类型。你也可以显式使用类型标注 `cutlass.Constexpr` 来指定某个参数为静态参数。

示例：

```python
import cutlass
import cutlass.cute as cute


@cute.jit
def foo(x: cutlass.Int32, y: cutlass.Constexpr):
    print("x = ", x)      # 输出 x = ?
    print("y = ", y)      # 输出 y = 2
    cute.printf("x: {}", x)  # 输出 x: 2
    cute.printf("y: {}", y)  # 输出 y: 2


foo(2, 2)
```

在上面的示例中，`x` 是一个类型为 `cutlass.Int32` 的动态参数，而 `y` 是一个静态参数。

借助 `cutlass.Constexpr` 标注，在 JIT kernel 中使用静态参数的一个更复杂的用法如下所示：

```python
import cutlass
import cutlass.cute as cute


@cute.kernel
def kernel(
    self,
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA_mk1: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB_nk1: cute.Tensor,
    tma_atom_c: Optional[cute.CopyAtom],
    mC_mn1: cute.Tensor,
    cluster_layout_vmnk: cute.Layout,
    a_smem_layout_staged: cute.ComposedLayout,
    b_smem_layout_staged: cute.ComposedLayout,
    c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
    epi_tile: cute.Tile,
    epilogue_op: cutlass.Constexpr,
):
    ...
    # 对累加器执行 epilogue 操作，并转换为 C 的数据类型
    acc_vec = tTR_rAcc.load()
    acc_vec = epilogue_op(acc_vec.to(self.c_dtype))
    tTR_rC.store(acc_vec)
```

在这个示例中，`epilogue_op` 是 JIT kernel 的一个静态参数，用于进行 epilogue 融合（fusion）。在调用 kernel 时，可以把一个逐元素（elementwise）的 lambda 函数作为 `epilogue_op` 传入。例如，如果你想在 epilogue 融合中应用 ReLU，只需要把 `epilogue_op` 设置为 `lambda x: cute.where(x > 0, x, cute.full_like(x, 0))`。

完整示例可参考 Blackwell dense GEMM 示例(github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell/dense_gemm_persistent.py
)。

## 类型安全

CuTe DSL 利用 JIT 函数签名中的类型注解，在编译期对 JIT 函数参数类型进行校验，以保证 **类型安全（type safety）**。

```python
import cutlass
import cutlass.cute as cute
import numpy as np


@cute.jit
def foo(x: cute.Tensor, y: cutlass.Float16):
    ...


a = np.random.randn(10, 10).astype(np.float16)
b = 32

foo(a, b)
foo(b, a)  # 由于类型不匹配，这将在编译期失败
```

类型安全检查可以在编译期尽早捕获类型不匹配问题，并给出清晰的错误信息，从而避免更难排查、且通常调试成本更高的运行时错误。在上面的示例中，第二次调用 `foo` 会因为类型不匹配而在编译期失败，并给出清晰的错误信息：

```python
cutlass.base_dsl.common.DSLRuntimeError: DSLRuntimeError: expects argument #1 (a) to be <class 'cutlass.cute.typing.Tensor'>, but got <class 'int'>
```

## 支持自定义类型的 JIT 函数参数

CuTe DSL 通过提供两个可在运行时检查的协议，来支持 JIT 函数参数使用自定义类型：

- `JitArgument`：用于从 Python 调用的 host 侧 JIT 函数。
  - `__c_pointers__`：为当前对象生成 ctypes 指针列表。
  - `__get_mlir_types__`：为当前对象生成 MLIR 类型列表。
  - `__new_from_mlir_values__`：从 MLIR values 创建一个新对象。
- `DynamicExpression`：用于在 host 侧 JIT 函数中调用的 device 侧 JIT 函数。
  - `__extract_mlir_values__`：为当前对象生成一个 dynamic expression。
  - `__new_from_mlir_values__`：从 MLIR values 创建一个新对象。

关于这些协议 API 的更多细节，请参考 `typing.py`(https://github.com/NVIDIA/cutlass/tree/main/python/CuTeDSL/base_dsl/typing.py)。

针对不同类型的自定义类型场景，CuTe DSL 提供了更便捷的方式来接入自定义类型作为 JIT 函数参数。

### 1. 在自定义类型中直接实现协议

一种方式是在自定义类型中直接实现协议方法，以启用基于协议的 JIT 函数参数生成。

```python
import cutlass
import cutlass.cute as cute


# 实现 DynamicExpression 协议的自定义类型
class MyDynamicExpression:
    def __init__(self, tensor, offset):
        self._tensor = tensor  # 动态参数
        self._offset = offset  # 动态参数

    def __extract_mlir_values__(self):
        return (
            self._tensor.__extract_mlir_values__(),
            self._offset.__extract_mlir_values__(),
        )

    def __new_from_mlir_values__(self, values):
        return MyDynamicExpression(values[0], values[1])


@cute.kernel
def my_kernel(x: MyDynamicExpression):
    ...
```

在上面的示例中，`MyDynamicExpression` 实现了 `DynamicExpression` 协议，因此 CuTe DSL 会基于这些协议方法为 JIT kernel `my_kernel` 生成对应的 JIT 函数参数。

### 2. 基于适配器（Adaptor）的协议实现（用于自定义类型）

当你无法直接修改自定义类型来实现协议时，CuTe DSL 提供了基于适配器（adaptor）的方式，将自定义类型适配为可用于 JIT 函数参数生成的形式。

JIT 函数参数适配器是一个可调用对象，它为注册的自定义类型实现所需的协议方法。这样一来，CuTe DSL 会自动查询 JIT 参数适配器注册表，为指定的自定义类型生成 JIT 函数参数。

```python
@cutlass.register_jit_arg_adapter(MyFrameworkObject)
class MyFrameworkObjectAdapter:
    """
    将第三方框架对象转换为遵循 JitArgument 协议的 JIT 函数参数
    """

    def __init__(self, arg):
        self._arg = arg

    def __c_pointers__(self):
        # 通过其 C-ABI 接口，将框架对象转换为与 C-ABI 兼容的对象
        return [self._arg.get_cabi_pointer()]

    def __get_mlir_types__(self):
        # 返回该框架对象所表示的 MLIR 类型列表
        return [self._arg.get_data().mlir_type]

    def __new_from_mlir_values__(self, values):
        # 将 MLIR values 转换回框架对象
        return MyFrameworkObject(values[0])
```

在这个示例中，`MyFrameworkObjectAdapter` 实现了一个适配器类，用于桥接 CuTe DSL 与第三方框架类型 `MyFrameworkObject`。注册方式也很简单：只需要用 `cutlass.register_jit_arg_adapter` 装饰该适配器，并指定要适配的自定义类型。完成注册后，CuTe DSL 会自动使用该适配器，为 `MyFrameworkObject` 类型的参数生成对应的 JIT 函数参数。

# 静态布局 vs. 动态布局

## 静态布局

在与主流深度学习框架集成时，一个常见问题是：如何处理转换得到的 `cute.Tensor` 的 layout。例如，当把 `torch.Tensor` 转换为 `cute.Tensor` 时，会把 `torch.Tensor` 的 shape 体现在 `cute.Tensor` 的 layout 中。

```python
import torch
import cutlass
from cutlass.cute.runtime import from_dlpack

@cute.jit
def foo(tensor):
    print(f"tensor.layout: {tensor.layout}")  # 在编译期打印 tensor 的 layout
    cute.printf("tensor: {}", tensor)         # 在运行期打印 tensor 的值
```

在这个示例中，我们定义了一个 JIT 函数 `foo`，它以 `cute.Tensor` 作为输入，并打印其 layout。注意，这里使用 Python 的 `print` 在编译期打印 layout。对于编译期已知值的静态布局，这样做是可行的。

现在我们尝试用不同 shape 的输入 `torch.Tensor` 来运行该 JIT 函数 `foo`。

```python
a = torch.tensor([1, 2, 3], dtype=torch.uint16)
a_pack = from_dlpack(a)
compiled_func = cute.compile(foo, a_pack)
compiled_func(a_pack)
```

这里我们先用 `from_dlpack` 把一个包含 3 个元素的一维 `torch.Tensor` 转换为 `cute.Tensor`。然后用转换后的 `cute.Tensor` 编译 JIT 函数 `foo`，并调用编译后的函数。

```python
tensor.layout: (3):(1)
tensor: raw_ptr(0x00000000079e5100: i16, generic, align<2>) o (3):(1) = (1, 2, 3)
```

你会看到 layout 打印为 `(3):(1)`，因为转换得到的 `cute.Tensor` 的静态 layout shape 为 `(3)`，与 `a` 的 shape 一致。

如果我们用不同 shape 的输入 `torch.Tensor` 来调用这个已编译的函数，由于类型（更准确地说是 layout）不匹配，会在运行时得到非预期结果：`compiled_func` 期望接收 layout 为 `(3):(1)` 的 `cute.Tensor`，但 `b` 的 shape 是 `(5)`。

```python
b = torch.tensor([11, 12, 13, 14, 15], dtype=torch.uint16)
b_pack = from_dlpack(b)
compiled_func(b_pack)  # ❌ 类型不匹配会导致运行时非预期结果
```

由于类型不匹配，输出会出现非预期结果，如下所示：

```python
tensor: raw_ptr(0x00000000344804c0: i16, generic, align<2>) o (3):(1) =
(11, 12, 13)
```

要解决这个问题，我们需要针对 `b` 的新 shape 触发一次新的代码生成与编译。

```python
compiled_func_2 = cute.compile(foo, b_pack)  # 会触发一次新的编译
compiled_func_2(b_pack)                      # ✅ 现在可以正常工作
```

如上所示，使用新编译得到的 `compiled_func_2`，我们就可以把 `b_pack` 传入编译后的 JIT 函数 `compiled_func_2`。

```python
tensor.layout: (5):(1)
tensor: raw_ptr(0x0000000034bb2840:: i16, generic, align<2>) o (5):(1) =
(11, 12, 13, 14, 15)
```

现在它会重新编译并正确打印 `b` 的值。

很明显，对于不同的静态 layout，我们需要生成并编译不同版本的代码：在这个例子中，一个对应 layout `(3):(1)`，另一个对应 layout `(5):(1)`。

## 动态布局

为了避免针对不同 shape 的输入 `torch.Tensor` 重复生成与编译多次，CuTe DSL 提供了使用动态布局（dynamic layout）来生成和编译 JIT 函数的方法。

要获得 `cute.Tensor` 的动态布局，可以直接将 `torch.Tensor` 对象传入 JIT 函数。这会指示 CuTe DSL 在转换后的 `cute.Tensor` 上，按 layout 的主维度（leading dimension）自动调用 `cute.mark_layout_dynamic`。

```python
import torch
import cutlass
from cutlass.cute.runtime import from_dlpack

@cute.jit
def foo(tensor):
    print(tensor.layout)  # 对动态布局会打印 (?,?):(?,1)

a = torch.tensor([[1, 2], [3, 4]], dtype=torch.uint16)
compiled_func = cute.compile(foo, a)
compiled_func(a)

 b = torch.tensor([[11, 12], [13, 14], [15, 16]], dtype=torch.uint16)
 compiled_func(b)  # 对不同 shape 复用同一个已编译函数
 ```
 
 在上面的示例中，JIT 函数 `foo` 只编译一次，就可以对不同 shape 的输入 `torch.Tensor` 复用。这是因为转换得到的 `cute.Tensor` 采用了动态布局 `(?,?):(?,1)`，它与两次调用的输入 `torch.Tensor` 的 shape 都兼容。
 
 另外，对于紧凑布局（compact layout），可以调用 `cute.mark_compact_shape_dynamic` 做更细粒度的控制：用于指定动态布局的 mode，以及动态维度的可整除性约束。
 
 关于 `from_dlpack`、`mark_layout_dynamic` 与 `mark_compact_shape_dynamic` 的更多细节，请参考 Integration with Frameworks(https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/framework_integration.html)。
 
 ## 静态布局 vs. 动态布局
 
 根据前面的内容可知：静态布局会导致为不同 shape 生成不同的 JIT 代码；而动态布局则可以通过一次编译覆盖多种 shape。
 
 但需要说明的是：当你的使用场景针对固定 shape 的输入数据时，使用静态布局来创建 JIT 函数仍然很有价值。因为编译期可获得更多信息，编译器就能启用一些对动态布局代码不一定可用的优化。
 
 另一方面，当输入数据的 shape 经常变化时，动态布局更灵活。它能让生成的代码更具可扩展性，以适配不同 shape 的输入数据。
 
 ## 编程时使用静态布局和动态布局
 
 CuTe DSL 提供了直观的方式来编程时使用静态布局和动态布局。
 
```python
import torch
import cutlass
from cutlass.cute.runtime import from_dlpack

@cute.jit
def foo(tensor, x: cutlass.Constexpr[int]):
    print(cute.size(tensor))  # 第一次调用会打印 3
                              # 第二次调用会打印 ?
    if cute.size(tensor) > x:
        cute.printf("tensor[2]: {}", tensor[2])
    else:
        cute.printf("tensor size <= {}", x)

a = torch.tensor([1, 2, 3], dtype=torch.uint16)
foo(from_dlpack(a), 3)   # 第一次调用：静态布局

b = torch.tensor([1, 2, 3, 4, 5], dtype=torch.uint16)
foo(b, 3)                # 第二次调用：动态布局
```

在这个示例中，JIT 函数 `foo` 在第一次调用时会使用静态布局 `(3):(1)` 进行编译，这意味着张量的 size 在编译期已知。CuTe DSL 会充分利用这一点，并在编译期自动处理 if 条件，因此生成的代码会更高效，甚至不会包含 if 条件。

在第二次调用时，JIT 函数 `foo` 会使用动态布局 `(?):(1)` 进行编译，因此张量的 size 只能在运行期求值。CuTe DSL 会自动生成代码，以在运行期处理动态布局与 if 条件。

同样的逻辑也适用于循环：

```python
@cute.jit
def foo(tensor, x: cutlass.Constexpr[int]):
    for i in range(cute.size(tensor)):
        cute.printf("tensor[{}]: {}", i, tensor[i])

a = torch.tensor([1, 2, 3], dtype=torch.uint16)
foo(from_dlpack(a), 3)   # 第一次调用：静态布局

b = torch.tensor([1, 2, 3, 4, 5], dtype=torch.uint16)
foo(b, 3)                # 第二次调用：动态布局
```

在第一次调用的静态布局下，CuTe DSL 能在编译期把循环完全展开。而在第二次调用中，生成的代码会基于动态布局在运行期执行循环。

通过同一个 JIT 函数实现，CuTe DSL 能够处理控制流结构，并针对不同场景自动生成优化后的代码。这一切之所以可行，是因为 CuTe DSL 能遍历 Python AST，并将其中的每一种控制流结构按需转换。

更多细节请参考 Control Flow(https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_control_flow.html)。

# JIT 缓存

## 零编译（Zero Compile）与 JIT Executor

零编译（Zero Compile）是一项特性，它允许通过 `cute.compile` 按需显式编译 kernel。当调用 `cute.compile` 时，它会编译 kernel 并返回一个 JIT Executor 实例。这个 JIT Executor 实例可以被缓存，并在后续执行中直接复用，而无需再次编译 kernel。

JIT Executor 是一个用于独立执行已编译代码的组件。它既可以通过 `cute.compile` 创建，也可以通过隐式编译创建。JIT Executor 实例的行为类似一个可调用对象，用于执行已编译代码。每个 JIT Executor 实例维护一个已编译的 host 函数。

它包含执行所需的全部组件：

- Host 函数指针以及其 MLIR 执行引擎
- CUDA modules（可选）
- 参数规格（argument specifications），用于定义如何将 Python 参数转换为 C ABI 兼容类型。注意：带有 `cutlass.Constexpr` 提示的参数会被排除在参数规格之外，因为它们在编译期求值，而不是运行期。

例如，在下面的代码中，`print_result` 是一个 `cutlass.Constexpr` 值，它**不会**在运行期求值：

```python
import cutlass.cute as cute

@cute.jit
def add(a, b, print_result: cutlass.Constexpr):
   if print_result:
      cute.printf("Result: %d\n", a + b)
   return a + b

jit_executor = cute.compile(add, 1, 2, True)

jit_executor(1, 2) # 输出：``Result: 3``
```

JIT Executor 会确保在编译完成后，所有组件都被正确初始化与加载。例如，会加载所有 CUDA modules（通过 `cuModuleLoad`），并提取 kernel 的函数指针（通过 `cuModuleGetFunction`）。

调用一个 JIT Executor 实例时，它会：

- 解析 Python 运行期参数，并根据参数规格把它们转换为 C ABI 兼容类型
- 使用转换后的参数调用 host 函数

## 使用 `cute.compile` 的自定义缓存

`cute.compile` 会绕过 CuTe DSL 内置的缓存机制，并且总是执行编译，返回一个固定的 JIT Executor 实例。这使得你可以实现自定义的缓存策略。

示例：

```python
@cute.jit
def add(b):
   return a + b

# 定义一个自定义缓存
custom_cache = {}

a = 1
compiled_add_1 = cute.compile(add, 2)
custom_cache[1] = compiled_add_1
compiled_add_1(2) # 结果 = 3

a = 2
compiled_add_2 = cute.compile(add, 2)
custom_cache[2] = compiled_add_2
compiled_add_2(2) # 结果 = 4

# 使用自定义缓存
custom_cache[1](2) # 结果 = 3
custom_cache[2](2) # 结果 = 4
```

## CuTe DSL 中的缓存
 
 默认情况下，CuTe DSL 会隐式启用缓存，以避免在 kernel 在未发生变化的情况下被反复调用时触发重复编译。
 
 该缓存实现为一个 map，用于在 CuTe DSL 内部存储已编译的 JIT Executor 实例。
 
 缓存的 key 由以下内容的哈希共同组成：
 
 - CuTe DSL 生成的 MLIR 程序对应的 MLIR bytecode
 - 所有 CuTe DSL Python 源文件
 - 所有 CuTe DSL 共享库
 - 所有 CuTe DSL 环境变量
 
 缓存的 value 是一个已编译的 JIT Executor 实例。
 
 命中缓存（cache hit）时，会跳过编译，并复用缓存中的 JIT Executor 实例。
 
 未命中缓存（cache miss）时，会编译 kernel，并将新的 JIT Executor 实例存入缓存。
 
 下面是一个示例，用于演示 `add` kernel 的自动缓存：

```python
# 全局变量
a = 1

@cute.jit
def add(b):
   return a + b

# 缓存起初为空

# 第一次调用：未命中缓存触发编译
result = add(2) # 结果 = 3
# 缓存现在包含一个实例

# 第二次调用：命中缓存，复用缓存的 JIT Executor
result = add(2) # 结果 = 3

a = 2
# 第三次调用：由于 IR 代码发生变化而未命中缓存，触发重新编译
result = add(2) # 结果 = 4
# 缓存现在包含两个实例
```

缓存可以被序列化到文件中，以便后续运行复用。序列化之后，已编译的 MLIR bytecode 会被存储到文件里。缓存目录为 `/tmp/{current_user}/cutlass_python_cache`。在编译期间，缓存会按需将对应的 kernel 从文件加载到内存（如果文件存在）；编译结束后，也会把新编译出的可执行文件保存回文件。

注意：出于效率考虑，默认缓存目录位于临时文件夹中。但这个位置并不持久，可能会被系统清理（例如重启或磁盘空间清理时）。如果你希望跨会话保留缓存，请设置 `CUTE_DSL_CACHE_DIR` 环境变量，使其指向一个持久化目录。

以下环境变量用于控制文件缓存：

```shell
# 禁用文件缓存，同时保留内存缓存可用；默认值为 False。
export CUTE_DSL_DISABLE_FILE_CACHING=True

# 缓存目录；默认值为 /tmp/{current_user}/cutlass_python_cache。
export CUTE_DSL_CACHE_DIR=/home/user/local_cutlass_python_cache/dense_gemm_cache/
```

## 限制
 
 缓存的目的，是为了降低每次执行前的 host 侧 launch 开销。正如上面的示例所示，由于全局变量等动态因素的影响，很难保持原始 Python 代码与 MLIR 程序之间的一致性。因此，必须（**MUST**）始终生成 MLIR 程序，以验证 kernel 内容与之前构建的内容一致。
 
 为了获得最佳的 host launch 延迟，我们建议使用上面基于 `cute.compile` 的自定义缓存方法。
 
 # JIT 编译选项

## JIT 编译选项概览

当使用 CuTe DSL 编译 JIT 函数时，你可能希望控制编译过程的各个方面，例如优化等级或调试开关。CuTe DSL 在调用 `cute.compile` 时提供了灵活的接口来指定这些编译选项。

编译选项允许你自定义 JIT 编译函数的构建与执行方式。这在以下场景中很有用：

- 启用或禁用特定的编译器优化
- 生成用于排查问题的调试信息

这些选项可以作为关键字参数传给 `cute.compile`，也可以全局设置以作用于所有 JIT 编译。可用选项及其效果会在接下来的小节中介绍，并配有使用示例帮助你快速上手。

CuTe DSL 提供了多种方式来指定编译选项：既可以通过给 `cute.compile` 传入额外参数来指定，也可以采用更 Pythonic 的方式——为 `cute.compile` 使用独立的 Python 类型来表达这些选项。

## `cute.compile` 的字符串形式编译选项
 
 你可以在调用 `cute.compile` 时，以字符串形式提供额外的编译选项。CuTe DSL 使用 `argparse` 来解析这些选项；如果指定了任何无效选项，则会抛出错误。
 
 ### 选项
 
 | 选项 | 描述 | 默认值 | 类型 |
 | --- | --- | --- | --- |
 | `opt-level` | 编译的优化等级。等级越高，启用的优化越多。有效取值范围为 `[0, 3]`。 | `3`（最高优化等级） | `int` |
 | `enable-assertions` | 启用 host 与 device 代码断言。 | `False` | `bool` |
 | `keep-cubin` | 保留生成的 CUBIN 文件。 | `False` | `bool` |
 | `keep-ptx` | 保留生成的 PTX 文件。 | `False` | `bool` |
 | `ptxas-options` | 传递给 PTX Compiler 库的选项。 | `""` | `str` |
 | `generate-line-info` | 生成用于调试的行号信息。 | `False` | `bool` |
 | `gpu-arch` | 要编译到的 GPU 架构。 | `""` | `str` |
 | `enable-tvm-ffi` | 启用 Apache TVM FFI。 | `False` | `bool` |

你可以使用下面的代码来指定编译选项：

```python
jit_executor_with_opt_level_2 = cute.compile(add, 1, 2, options="--opt-level 2")
jit_executor_with_opt_level_1 = cute.compile(add, 1, 2, options="--opt-level 1")
jit_executor_with_enable_device_assertions = cute.compile(add, 1, 2, options="--enable-assertions")
jit_executor_with_keep_cubin = cute.compile(add, 1, 2, options="--keep-cubin")
jit_executor_with_keep_ptx = cute.compile(add, 1, 2, options="--keep-ptx")
jit_executor_with_ptxas_options = cute.compile(add, 1, 2, options="--ptxas-options '--opt-level=2'")
```

## `cute.compile` 使用独立 Python 类型表示编译选项
 
 另外，你也可以用一种更 Pythonic 的方式：使用独立的 Python 类型来指定编译选项。编译选项可以通过 tuple 以编程方式组合，并单独传递给 `cute.compile`。

```python
from cutlass.cute import OptLevel, EnableAssertions, GenerateLineInfo, KeepCUBIN, KeepPTX

my_debugging_options = (OptLevel(1), EnableAssertions, GenerateLineInfo, KeepCUBIN, KeepPTX)
compiled_kernel_1 = cute.compile[my_debugging_options](my_kernel_1, ...)
compiled_kernel_2 = cute.compile[my_debugging_options](my_kernel_2, ...)
```

 这种方式会让无效选项立刻报错，因此当你同时指定多个选项时，更容易发现拼写错误。需要注意的是（Notebly），为了使用方便，布尔选项会被自动转换为该选项类型的 `True` 实例。

```python
jit_executor_with_opt_level_2 = cute.compile[OptLevel(2)](add, 1, 2)
jit_executor_with_opt_level_1 = cute.compile[OptLevel(1)](add, 1, 2)
jit_executor_with_enable_device_assertions = cute.compile[EnableAssertions](add, 1, 2)
jit_executor_with_keep_cubin = cute.compile[KeepCUBIN](add, 1, 2)
jit_executor_with_keep_ptx = cute.compile[KeepPTX](add, 1, 2)
jit_executor_with_ptxas_options = cute.compile[PtxasOptions("--opt-level=2")](add, 1, 2)
```

# 与框架集成
 
为了方便将 CUTLASS Python 与常见框架集成，我们利用 DLPack 协议(https://github.com/dmlc/dlpack)，并将这些框架产生的张量转换为 CuTe 张量。本页面记录了相关约定、用户可用的 API，并提供了常见用法的示例代码片段。我们还提供了一个小节，介绍如何绕过 DLPack 协议并直接调用 JIT 函数。

## 隐式转换
 
来自支持 DLPack 协议的框架的张量，可以作为普通参数直接传给 JIT 函数。CuTe DSL 的运行时会隐式地把原始张量转换为 CuTe 张量，其布局除与 leading dimension 对应的 stride 元素外，其余部分都是完全动态的。下面的示例演示了这一用法。

```python
import torch
import cutlass.cute as cute

@cute.jit
def foo(src):
    """
    下面几行会打印

    ptr<f32, generic> o (?,?,?):(?,?,1)
    <class 'cutlass.cute.core._Tensor'>
    """
    print(src)
    print(type(src))


a = torch.randn(30, 20, 32, device="cpu")
foo(a)
```

## 使用 `from_dlpack` 的显式转换

CuTe DSL 的运行时提供了一个接口，用于将 DLPack 兼容的张量转换为 CuTe 张量：

```python
b = cute.runtime.from_dlpack(a)
```

其中 `a` 是一个支持 DLPack 协议的张量，并实现了 `__dlpack__` 与 `__dlpack_device__` 方法。转换得到的 CuTe 张量 `b` 具有完全静态的布局。该转换过程不会拷贝任何张量数据，从而实现与主流框架的无缝集成。用户可以用 NumPy、PyTorch 等创建张量，并将其直接喂给用 CuTe DSL 编写的 JIT 函数。

转换得到的 CuTe 张量与原始张量共享同一底层内存缓冲区。这种零拷贝方式通过消除不必要的数据复制来最大化性能。但需要注意：CuTe 张量的有效性与原始张量的生命周期绑定。如果源张量被销毁或超出作用域，对应的 CuTe 张量会因为引用了原始内存位置而变得无效。

`from_dlpack` 的完整函数签名如下：

```python
def from_dlpack(tensor, assumed_align=None, use_32bit_stride=False):
    ...
```

`assumed_align` 整型参数用于指定张量的对齐（单位：字节）。张量的基地址必须能被 `assumed_align` 整除。如果未显式提供，则对齐会被设置为张量元素类型的自然对齐（natural alignment）。注意：对齐信息是生成 IR 中指针类型的一部分。因此，不同对齐会对应不同的 IR；而要命中 CuTe DSL 的 kernel 缓存机制，需要 IR 完全一致。

`use_32bit_stride` 参数用于决定是否对张量的动态 stride 值使用 32-bit stride。默认设置为 `False`（64bit），以确保地址计算不会有溢出风险。对于较小的问题规模（满足 `cosize(layout_of_tensor) <= Int32_MAX`），用户可以将其设为 `True`（32bit），通过减少寄存器使用与地址计算指令数量来提升性能。当 `use_32bit_stride` 设为 `True` 时，会执行一次运行期检查以确保布局不会溢出。请注意：该参数只有在张量的 layout 被标记为 dynamic 时才会生效。

### 代码示例

下面的代码演示了如何使用默认参数，通过 `from_dlpack` 函数将 PyTorch 张量转换为 CuTe 张量。

```python
import torch
import cutlass
from cutlass.cute.runtime import from_dlpack

x = torch.randn(30, 20, device="cpu")
y = from_dlpack(x)
```

转换完成后，我们可以通过多种属性访问该张量的信息。下面列出转换后张量的属性：
- `tensor.shape`：张量的 shape
- `tensor.stride`：张量的 stride
- `tensor.memspace`：张量所在的内存空间（memory space）
- `tensor.element_type`：张量元素的数据类型

```python
import torch
import cutlass
from cutlass.cute.runtime import from_dlpack

x = torch.randn(30, 20, device="cpu")
y = from_dlpack(x)

print(y.shape)        # (30, 20)
print(y.stride)       # (20, 1)
print(y.memspace)     # generic（如果 torch tensor 位于 device memory 上，memspace 会是 gmem）
print(y.element_type) # Float32
print(y)              # Tensor<0x000000000875f580@generic o (30, 20):(20, 1)>
```

生成的 CuTe 张量的字符串格式为：

```python
Tensor<0x{tensor.data_ptr:016x}@{tensor.memspace} o {tensor.shape}:{tensor.stride}>
```

从上面的例子可以看到，`from_dlpack` 首先会得到一个具有静态布局（static layout）的张量。若想在调用 `from_dlpack` 后得到动态布局（dynamic）或静态/动态混合布局，可以使用 `mark_layout_dynamic` 与 `mark_compact_shape_dynamic`，它们会在后续小节中介绍。

## 何时使用显式转换？

DLPack 协议被广泛用于不同框架之间的互操作（interoperability）。但它也会带来一定开销。根据我们的 benchmark，每次调用 `from_dlpack` 通常需要 2 到 3 us。

显式转换允许你缓存转换后的 CuTe 张量，从而避免重复调用 `from_dlpack` 带来的开销。

```python
x = torch.randn(30, 20, device="cpu")
if key not in cached_tensors:
    # 仅在未命中缓存时执行转换
    cached_tensors[key] = cute.runtime.from_dlpack(x)
foo(cached_tensors[key])
```

显式转换的另一个用例，是从生成程序的视角出发，更细粒度地控制张量的哪些 mode 会被视为动态（dynamic）。

## 使用 `mark_layout_dynamic` 将张量布局标记为动态

调用该函数后，所有 shape mode 都会变成动态。stride mode 也会变成动态，但有以下两个例外：

- leading dimension 的 stride 仍固定为 1；
- stride 元素等于 0（表示 broadcasting）的部分会被保留。

`mark_layout_dynamic` 的完整函数签名如下：

```python
def mark_layout_dynamic(self, leading_dim: int|None = None):
```

`leading_dim` 参数用于指定张量的 leading dimension。leading dimension 的 stride 会被设置为 1，除非这与 DLPack 张量的布局不一致。例如：

- 对于布局为 `(2,2,3,4):(2,1,4,12)` 的张量，如果将 `leading_dim` 指定为 1，则 layout 会被标记为 `(?,?,?,?):(?,1,?,?)`。
- 如果将 `leading_dim` 指定为 0，则会抛出推断失败（deduction failure）错误，因为维度 0 的 stride 是 2（而不是 1）。

`leading_dim` 的默认值是 `None`。此时系统会根据张量的布局自动推断 leading dimension，推断逻辑如下：

- 如果某个维度的 stride 为 1，则该维度会被标记为 leading dimension。
- 如果有多个维度满足条件 1，则会抛出错误提示推断失败。注意：将 **PyTorch** 张量转换为 DLPack 格式后，shape 为 1 的维度的 stride 会被规范化（canonicalized）为 1。这种规范化会增加推断失败的概率。该行为是 PyTorch 特有的，例如在 NumPy 中不会发生。
- 如果没有任何维度满足条件 1，则所有 strides 都会被标记为动态。

例如：

- 对于布局为 `(2,2,3,4):(2,1,4,12)` 的张量，leading dimension 是 1，因此 layout 会被标记为 `(?,?,?,?):(?,1,?,?)`。
- 对于布局为 `(1,5,1):(1,1,1)` 的张量，如果不指定 leading_dim，会抛出推断失败错误。
- 对于布局为 `(2,2):(8,2)` 的张量，由于没有任何维度的 stride 为 1，所有维度都会被标记为动态：`(?,?):(?,?)`。

leading dimension 支持负索引，这意味着维度从最后一个维度开始计数。例如：

- 对于布局为 `(2,2,3,4):(2,1,4,12)` 的张量，如果将 leading_dim 指定为 -1，则 layout 会被标记为 `(?,?,?,?):(?,?,?,1)`。

### 代码示例

下面的示例演示了如何使用 `mark_layout_dynamic` 来指定动态张量布局。

- `t0` 展示在不指定 `leading_dim` 时使用 `mark_layout_dynamic`，以及 leading dimension 的自动推断。
- `t1` 与 `t2` 展示在指定 `leading_dim` 时使用 `mark_layout_dynamic`。
- `t3` 展示在没有 leading dimension 的情况下使用 `mark_layout_dynamic`。
- `t4` 展示在存在广播（broadcast）维度时使用 `mark_layout_dynamic`。
- `t5` 展示当 stride 等于 1 的维度多于一个时的推断失败。
- `t6` 与 `t7` 展示 `leading_dim` 设置错误时的预期报错。

```python
import torch
from cutlass.cute.runtime import from_dlpack

# (8,4,16,2):(2,16,64,1)
a = torch.empty(16, 4, 8, 2).permute(2, 1, 0, 3)
# (1,4,1,32,1):(4,1,4,4,4) => torch tensor when dimension has shape 1, its stride is degenerated to 1,
# resulting in (1,4,1,32,1):(1,1,1,4,1)
b = torch.empty(32, 1, 1, 1, 4).permute(3, 4, 1, 0, 2)

# 自动推断 leading dimension 为 3
t0 = from_dlpack(a).mark_layout_dynamic()
print(t0)
# (?,?,?,?):(?,?,?,1)

t1 = from_dlpack(b).mark_layout_dynamic(leading_dim=0)
print(t2)
# (?,?,?,?,?):(1,?,?,?,?)

t2 = from_dlpack(b).mark_layout_dynamic(leading_dim=2)
print(t3)
# (?,?,?,?,?):(?,?,1,?,?)

t3 = from_dlpack(c).mark_layout_dynamic()
print(t3)
# (?,?):(?,?)

t4 = from_dlpack(d).mark_layout_dynamic()
print(t4)
# (?,?,?,?):(?,0,0,1)

t5 = from_dlpack(b).mark_layout_dynamic()
# 无法从 layout 推断 leading dimension，请显式指定 leading_dim。

t6 = from_dlpack(a).mark_layout_dynamic(leading_dim=1)
# 期望 strides[leading_dim] == 1，但得到 16

t7 = from_dlpack(b).mark_layout_dynamic(leading_dim=3)
# 期望 strides[leading_dim] == 1，但得到 4

c = torch.empty(1000000000, 1000000000)
t8 = from_dlpack(c, use_32bit_stride=True).mark_layout_dynamic()
# DLTensorWrapper 中的 layout 存在 int32 溢出风险。请将 use_32bit_stride 设为 False。
```

## 使用 `mark_compact_shape_dynamic` 将张量布局标记为动态

`mark_compact_shape_dynamic` 函数为紧凑布局（compact layout）提供了对动态 shape 的细粒度控制。`mark_compact_shape_dynamic` 的完整函数签名如下：

```python
def mark_compact_shape_dynamic(
    self,
    mode: int,
    stride_order: tuple[int, ...] | None = None,
    divisibility: int = 1,
):
    ...
```

`mode` 参数用于决定哪一个 shape 维度会变成动态。调用该函数后，由 `mode` 指定的那个 shape 维度会立即被标记为动态，同时 stride 会相应更新。对于 shape 大小为 1 的 mode，其 stride 会被规范化（canonicalized）为 0。

`stride_order` 参数用于指定张量的 stride 排序方式。它与 `torch.Tensor.dim_order()` 一致，默认值为 `None`。该参数表示：如果把当前布局转换为 row-major order，各个 mode（维度）的顺序是什么；从左到右阅读时，它表示从最外层维度到最内层维度的顺序。当无法从张量布局中自动推断 stride 顺序时（例如存在多个维度的 stride 为 1），必须显式设置该参数。

例如：

- 布局 `(4,2):(1,4)` 的 stride_order 为 `(1,0)`，表示最内层维度是 0（4:1），最外层维度是 1（2:4）。
- 布局 `(5,3,2,4):(3,1,15,30)` 的 stride_order 为 `(3,2,0,1)`，表示最内层维度是 1（3:1），最外层维度是 3（4:30）。

如果未指定 `stride_order`，系统会根据张量布局按如下逻辑自动推断：

- 将 strides 按降序排序。
- 如果有多个维度的 stride 为 1，则会抛出推断失败错误。

例如：

- 对于布局为 `(2,2,3,4):(2,1,4,12)` 的张量，推断得到的 stride_order 是 `[3,2,0,1]`。
- 对于布局为 `(1,5,1):(1,1,1)` 的张量，`stride_order` 推断会失败，因为所有维度的 stride 都同为 1，无法确定正确的顺序。

如果指定了 `stride_order`，系统会验证该顺序是否与张量布局一致。

`divisibility` 参数用于指定动态 shape 的可整除性（divisibility）。它可用于表达对输入对齐（alignment）的假设。默认值为 1。

注意：该 API 仅适用于紧凑张量（compact tensors）。对于非紧凑张量，可以在 host JIT 函数中使用 `cute.assume` 将可整除性信息附加到某个特定的 shape mode 上，如下例所示：

```python
@cute.jit
def foo(a: cute.Tensor):
    new_shape = a.shape
    # 使用 cute.assume 将 mode=0 的 shape 设置为可被 16 整除
    new_shape[0] = cute.assume(new_shape[0], 16)
    new_layout = cute.make_layout(new_shape, stride=a.stride)
    new_a = cute.make_tensor(a.iterator, new_layout)
```

### 代码示例

下面的示例演示了如何使用 `mark_compact_shape_dynamic` 来指定动态张量布局。

- `t0` 与 `t1` 展示在不指定 stride_order 的情况下，使用不同的 mode 与 divisibility 调用 `mark_compact_shape_dynamic`。
- `t2` 展示在不指定 stride_order 的情况下，连续调用 `mark_compact_shape_dynamic`，并使用不同的 mode 与 divisibility。
- `t3` 与 `t4` 展示在指定不同 stride_order 的情况下使用 `mark_compact_shape_dynamic`。
- `t5`、`t6`、`t7`、`t8`、`t9`、`t10`、`t11` 与 `t12` 展示参数设置错误时的预期报错。

```python
import torch
from cutlass.cute.runtime import from_dlpack

# (8,4,16,2):(2,16,64,1)
a = torch.empty(16, 4, 8, 2).permute(2, 1, 0, 3)
# (1,4,1,32,1):(4,1,4,4,4) => 当 torch tensor 的某个维度的 shape 为 1 时，该维度的 stride 会退化为 1，
# 从而变为 (1,4,1,32,1):(1,1,1,4,1)
# b.dim_order() 为 (3,2,4,0,1)
b = torch.empty(32, 1, 1, 1, 4).permute(3, 4, 1, 0, 2)

# 自动推断 stride order 为 [2,1,0,3]
t0 = from_dlpack(a).mark_compact_shape_dynamic(
    mode=0, divisibility=2
)
# (?{div=2},4,16,2):(2,?{div=4},?{div=16},1)
print(t0)

t1 = from_dlpack(a).mark_compact_shape_dynamic(
    mode=1, divisibility=2
)
# (8,?{div=2},16,2):(2,16,?{div=32},1)
print(t1)

t2 = from_dlpack(a).mark_compact_shape_dynamic(
    mode=1, divisibility=2
).mark_compact_shape_dynamic(
    mode=3, divisibility=2
)
# (8,?{div=2},16,?{div=2}):(?{div=2},?{div=16},?{div=32},1)
print(t2)

t3 = from_dlpack(b).mark_compact_shape_dynamic(
    mode=2, divisibility=1, stride_order=(3, 0, 2, 4, 1)
)
# (1,4,?,32,1):(0,1,4,?{div=4},0)
print(t3)

t4 = from_dlpack(b).mark_compact_shape_dynamic(
    mode=2, divisibility=1, stride_order=(2, 3, 4, 0, 1)
)
# (1,4,?,32,1):(0,1,128,4,0)
print(t4)

t5 = t2.mark_compact_shape_dynamic(
    mode=3, divisibility=5, stride_order=(0, 1, 2, 3)
)
# stride_order 与上一次的 stride_order 不一致

t6 = from_dlpack(a).mark_compact_shape_dynamic(
    mode=3, divisibility=5, stride_order=(0, 1, 2, 3)
)
# stride_order 与推断得到的 stride_order 不一致

t7 = from_dlpack(b).mark_compact_shape_dynamic(
    mode=0, divisibility=4
)
# 无法推断 layout，请显式指定 stride_order

t8 = from_dlpack(b).mark_compact_shape_dynamic(
    mode=30, divisibility=5, stride_order=(3, 0, 2, 4, 1)
)
# 期望 mode 的取值范围为 [0, 5)，但得到 30

t9 = from_dlpack(b).mark_compact_shape_dynamic(
    mode=3, divisibility=5, stride_order=(2, 1, 2, 3, 4)
)
# 期望 stride_order 包含张量的所有维度，但它不包含 0。

t10 = from_dlpack(b).mark_compact_shape_dynamic(
    mode=3, divisibility=5, stride_order=(0, 1, 2, 3, 4, 5)
)
# 期望 stride_order 有 5 个元素，但得到 6。

t11 = from_dlpack(b).mark_compact_shape_dynamic(
    mode=0, divisibility=4, stride_order=b.dim_order()
)
# mode(0) 的 shape(1) 不能被 divisibility(4) 整除

t12 = from_dlpack(b).mark_compact_shape_dynamic(
    mode=0, divisibility=1, stride_order=(2, 1, 3, 0, 4)
)
# stride_order 与 layout 不一致

c = torch.empty(1000000000, 1000000000)
t13 = from_dlpack(c, use_32bit_stride=True).mark_compact_shape_dynamic(
    mode=0, divisibility=1
)
# DLTensorWrapper 中的 layout 存在 int32 溢出风险。请将 use_32bit_stride 设为 False。
```


## 利用 TVM FFI 加速 PyTorch 互操作

最新版本的 CuTe DSL 支持 TVM FFI，用于提升与 PyTorch 以及其他机器学习框架的互操作能力。使用 TVM FFI 具备以下特性：

- 更快的 JIT 函数调用。
- 可直接接受 `torch.Tensor` 对象作为函数参数。
- 更强的错误处理与 kernel 校验。
- 可与多种编程语言无缝集成。

更多细节请参考 Compile with TVM FFI(https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/compile_with_tvm_ffi.html)。

## 绕过 DLPack 协议

在某些场景下，用户可能希望绕过 DLPack 协议并直接调用 JIT 函数。可以通过给现有 JIT 函数套一层轻量级的 JIT wrapper 来实现：使用 `cute.ptr` 与 `cute.make_tensor` 直接传递指针并构造张量。

绕过 DLPack 的典型用例包括：
1. 用户希望直接调用 JIT 函数，以避免 DLPack 协议引入的额外开销。
2. DLPack 会将 shape 为 1 的维度的 stride 规范化为 1，这可能导致对齐（alignment）信息传播不正确，从而影响内存访问或性能。
3. DLPack 可能缺少对某些窄数据类型（narrow data types）的支持。

下面的示例展示了在调用 JIT 函数时如何绕过 DLPack 协议。假设我们已有一个预定义的 `TensorOpGemm` kernel，其 JIT 接口期望 3 个 `cute.Tensor` 类型的参数。为了在不使用 DLPack 的情况下直接调用，我们首先定义一个 JIT wrapper 函数，它接受 `cute.Pointer` 类型作为参数。在该 wrapper 中，我们用 `cute.make_tensor` 从传入指针构造张量，然后像往常一样调用 `TensorOpGemm` kernel。

```python
@cute.jit
def tensor_op_gemm_wrapper(
    a_ptr: cute.Pointer,
    b_ptr: cute.Pointer,
    c_ptr: cute.Pointer,
    m: cutlass.Int32,
    n: cutlass.Int32,
    k: cutlass.Int32,
    l: cutlass.Int32,
):

    # 假设 shape 满足对齐条件，以调用 tensorop_gemm 示例
    m = cute.assume(m, divby=8)
    n = cute.assume(n, divby=8)

    # Torch 为行主序（row major）
    a_layout = cute.make_ordered_layout((m, k, l), order=(0, 1, 2))
    b_layout = cute.make_ordered_layout((n, k, l), order=(0, 1, 2))
    c_layout = cute.make_ordered_layout((m, n, l), order=(1, 0, 2))
    mA = cute.make_tensor(a_ptr, layout=a_layout)
    mB = cute.make_tensor(b_ptr, layout=b_layout)
    mC = cute.make_tensor(c_ptr, layout=c_layout)

    # TensorOpGemm 是我们示例中预定义的 kernel
    tensor_op_gemm = TensorOpGemm(
        a_ptr.value_type, c_ptr.value_type, cutlass.Float32, (2, 2, 1)
    )

    tensor_op_gemm(mA, mB, mC)
```

为了把 PyTorch 张量传给这个新的 JIT wrapper，我们需要从 PyTorch 张量中获取原始指针（raw pointer），并使用 `cute.make_ptr` 创建一个 `cute.Pointer` 实例。这样就能完全绕过 DLPack 协议，避免其开销以及在处理 shape 为 1 的维度时可能出现的问题。

```python
a = torch.randn(
    m, k, l, dtype=torch.float16, device="cuda"
).permute(2, 1, 0)
b = torch.randn(
    n, k, l, dtype=torch.float16, device="cuda"
).permute(2, 1, 0)
c = torch.randn(
    n, m, l, dtype=torch.float16, device="cuda"
).permute(1, 2, 0)

# from cutlass.cute.runtime import make_ptr
a_ptr = make_ptr(
    cutlass.Float16, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
)
b_ptr = make_ptr(
    cutlass.Float16, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
)
c_ptr = make_ptr(
    cutlass.Float16, c.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
)
tensor_op_gemm_wrapper(a_ptr, b_ptr, c_ptr, m, n, k, l)
```

## 调试

本页面概述了 CuTe DSL 程序的调试技巧与工具。

## 了解限制

在深入了解完整的调试能力之前，理解 CuTe DSL 的限制非常重要。了解这些限制可以帮助你从一开始就避免潜在陷阱。

更多细节请参考 **Limitations**(https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/limitations.html)。

## 源代码关联（Source Code Correlation）

CuTe DSL 支持 Python 代码与 PTX/SASS 的关联：通过在编译 kernel 时生成行号信息（line info），你可以使用带有调试符号的方式对生成的 kernel 进行 profiling/调试。

你可以通过环境变量 `CUTE_DSL_LINEINFO=1` 全局启用该功能。或者，你也可以使用编译选项为每个 kernel 单独启用。更多细节请参考 **JIT Compilation Options**(https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_jit_compilation_options.html)。

## DSL 调试

CuTe DSL 提供了内置的日志机制，帮助你理解代码的执行流程以及部分内部状态。

### 启用日志

CuTe DSL 提供了环境变量来控制日志级别：

```shell
# 启用控制台日志（默认：False）
export CUTE_DSL_LOG_TO_CONSOLE=1

# 将日志写入文件而不是控制台（默认：False）
export CUTE_DSL_LOG_TO_FILE=my_log.txt

# 控制日志详细程度（0, 10, 20, 30, 40, 50；默认：10）
export CUTE_DSL_LOG_LEVEL=20
```

### 日志类别与等级

与标准 Python logging 类似，不同日志级别对应不同的细节程度：

| 等级 | 描述 |
| --- | --- |
| 0 | 禁用 |
| 10 | 调试（Debug） |
| 20 | 信息（Info） |
| 30 | 警告（Warning） |
| 40 | 错误（Error） |
| 50 | 致命（Critical） |

## 导出生成的 IR

对于熟悉 MLIR 与编译器的用户，CuTe DSL 支持导出中间表示（Intermediate Representation，IR）。这有助于你验证 IR 是否按预期生成。

```shell
# 导出生成的 CuTe IR（默认：False）
export CUTE_DSL_PRINT_IR=1

# 将生成的 CuTe IR 保存在文件中（默认：False）
export CUTE_DSL_KEEP_IR=1
```

## 导出生成的 PTX 与 CUBIN

对于熟悉 PTX 与 SASS 的用户，CuTe DSL 支持导出生成的 PTX 与 CUBIN。

```shell
# 将生成的 PTX 导出为 .ptx 文件（默认：False）
export CUTE_DSL_KEEP_PTX=1

# 将生成的 cubin 导出为 .cubin 文件（默认：False）
export CUTE_DSL_KEEP_CUBIN=1
```

若想进一步从 cubin 获取 SASS，用户可以使用 `nvdisasm`（通常随 CUDA toolkit 安装）对 cubin 进行反汇编。

```shell
nvdisasm your_dsl_code.cubin > your_dsl_code.sass
```

## 以编程方式访问导出内容

对于已编译的 kernel，也可以通过以下属性以编程方式访问生成的 PTX/CUBIN/IR：

- `__ptx__`：已编译 kernel 的 PTX 代码。
- `__cubin__`：已编译 kernel 的 CUBIN 数据。
- `__mlir__`：已编译 kernel 的 IR 代码。

```python
compiled_foo = cute.compile(foo, ...)
print(f"PTX: {compiled_foo.__ptx__}")
with open("foo.cubin", "wb") as f:
    f.write(compiled_foo.__cubin__)
```

## 更改导出目录

默认情况下，所有导出的文件都会保存在当前工作目录中。若要为导出文件指定不同的目录，请相应设置环境变量 `CUTE_DSL_DUMP_DIR`。

## Kernel 功能调试

### 使用 Python 的 `print` 与 CuTe 的 `cute.printf`

CuTe DSL 程序既可以使用 Python 原生的 `print()`，也可以使用我们提供的 `cute.printf()`，在 kernel 生成与执行期间输出调试信息。它们在一些关键方面有所不同：

- Python 的 `print()` 仅在编译期执行（对生成的 kernel 没有影响），通常用于打印静态值（例如完全静态的 layout）。
- `cute.printf()` 在 GPU 上的运行期执行，并会改变生成的 PTX。它可用于在运行期打印张量的值以进行诊断，但会带来类似 CUDA C 中 `printf()` 的性能开销。

有关如何使用这些函数进行调试的详细示例，请参考 **Educational Notebooks**(https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/notebooks.html) 中引用的相关 notebook。

## 处理无响应/卡死的 Kernel

当 kernel 变得无响应且 `SIGINT`（`CTRL+C`）无法终止时，你可以按以下步骤强制结束进程：

1. 使用 `CTRL+Z` 挂起无响应的 kernel
2. 执行以下命令终止被挂起的进程：

```shell
# 终止最近一次被挂起的进程
kill -9 $(jobs -P | tail -1)
```

CuTe DSL 也可以使用标准的 NVIDIA CUDA 工具进行调试。

## 使用 Compute-Sanitizer

用于检测内存错误与竞态条件：

```shell
compute-sanitizer --some_options python your_dsl_code.py
```

更多细节请参考 compute-sanitizer 文档(https://developer.nvidia.com/compute-sanitizer)。

## 总结

本页面介绍了调试 CuTe DSL 程序的若干关键方法。有效的调试通常需要组合使用这些方法。如果你遇到 DSL 相关问题，可以启用日志并将日志作为 GitHub issue 分享给 CUTLASS 团队以报告 bug。

## 自动调优（Auto-Tuning）指南

我们的代码库中提供了大量 GEMM kernel 代码示例。当将这些 kernel 集成到框架中时，为了获得最佳性能，自动调优（auto-tuning）就变得至关重要。这通常需要根据真实应用的输入来选择合适的 kernel 参数。接下来我们会简要介绍一些自动调优的技巧。

自动调优流程通常包含以下步骤：

1. 定义搜索空间（search space）
2. 对每个配置进行 benchmark，并选择性能最佳的 kernel
3. 启用缓存以降低调优成本

搜索空间定义了可用于运行 kernel 的有效参数组合。不同输入（shape、数据类型等）通常需要不同的 kernel 参数才能达到最优性能。搜索空间与 kernel 本身相关。这里以 Blackwell GEMM persistent kernel 为例，其搜索空间如下：

- `mma_tiler_mn`：定义每条矩阵乘加（Matrix Multiply-Accumulate，MMA）指令在一次操作中处理的矩阵 tile 尺寸。
- `cluster_shape_mn`：指定一个 cluster 内沿各维度的 CTA 数量。不同张量数据类型下可用的 mma tiler size 与 cluster shape 请参考 Parallel Thread Execution ISA 文档(https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensorcore-5th-generation-family-instructions)。
- `use_2cta_instrs`：是否使用 Blackwell 的 2 CTA 指令用于 MMA/Copy。
- `use_tma_store`：是否使用 Tensor Memory Access（TMA）指令将结果写回全局内存。

在定义搜索空间之后，我们可以遍历所有参数组合以找到最优的 kernel。下面的 `autotune_gemm` 函数演示了一种简单的穷举搜索（exhaustive search）方法：它遍历不同配置，对每个 kernel 进行编译与 benchmark，并返回性能最佳的那个。由于 kernel 编译会带来额外开销，为了最小化 host launch 延迟，缓存并复用已编译的 kernel 非常重要。CuTe DSL 通过分离编译与执行的工作流来支持这一点。更多细节可参考 JIT Caching。正如 `autotune_gemm` 函数所示（在 `begin of cache the compiled GEMM kernel` 与 `end of cache the compiled GEMM kernel` 注释之间），我们可以使用 `cute.compile()` 编译一次 kernel，将编译结果缓存起来，并在多次 kernel 执行中复用缓存的 JIT executor。我们可以维护一个全局的“配置 -> kernel”字典（例如 `config_kernel_dict`）来缓存已编译的 GEMM kernel，其中每个 key（`kernel_cache_key`）基于 kernel 特征唯一标识一个 kernel。通常我们可以用 {dtype + kernel configs} 作为 GEMM 编译的缓存 key。例如：

```python
kernel_cache_key = f"{ab_dtype}x{c_dtype}x{acc_dtype}x{use_2cta_instrs}x{mma_tiler}x{cluster_shape_mn}x{use_tma_store}"
```

如果输入张量的 layout 是静态的，我们也应该把 shape 加入缓存 key。用户可以自定义 `benchmark` 函数来测量 kernel 执行时间。为了获得稳定可靠的性能测量结果：

1. 先跑一些 warmup 迭代（例如 5-10 次）以稳定 GPU 温度
2. 运行多次计时迭代（例如 100-1000 次）以获得统计意义
3. 使用 CUDA events 与同步来进行精确计时
4. 使用 nvidia-smi 锁定 GPU 频率（SM 与显存频率）
5. 对结果进行处理：去除离群值，并使用 min/avg 等统计值作为测量指标

这样可以通过规范的 benchmark 确保 kernel 选择的可靠性。

```python
# 为给定输入张量获取最佳 GEMM kernel
def autotune_gemm(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    stream: cuda.CUstream,
    use_2cta_instrs_list: List[bool] = [True],
    use_tma_store_list: List[bool] = [True],
    mma_tiler_m_list: List[int] = [256],
    mma_tiler_n_list: List[int] = [256],
    cluster_shape_m_list: List[int] = [2],
    cluster_shape_n_list: List[int] = [1],
):
    best_kernel = None
    min_time = float("inf")
    # 遍历搜索空间
    for use_2cta_instrs in use_2cta_instrs_list:
        for use_tma_store in use_tma_store_list:
            for mma_tiler_mn in product(mma_tiler_m_list, mma_tiler_n_list):
                for cluster_shape_mn in product(cluster_shape_m_list, cluster_shape_n_list):
                    acc_dtype = cutlass.Float32
                    hardware_info = cutlass.utils.HardwareInfo()
                    max_active_clusters = hardware_info.get_max_active_clusters(
                        cluster_shape_mn[0] * cluster_shape_mn[1]
                    )
                    # 实例化一个 GEMM kernel
                    gemm = PersistentDenseGemmKernel(
                        acc_dtype,
                        use_2cta_instrs,
                        mma_tiler_mn,
                        cluster_shape_mn,
                        use_tma_store,
                    )
                    # 开始缓存已编译的 GEMM kernel
                    if kernel_cache_key not in config_kernel_dict:
                        # 编译 gemm kernel
                        compiled_gemm = cute.compile(
                            gemm,
                            a,
                            b,
                            c,
                            max_active_clusters,
                            stream,
                        )
                        config_kernel_dict[kernel_cache_key] = compiled_gemm
                    else:
                        compiled_gemm = config_kernel_dict[kernel_cache_key]
                    # 结束缓存已编译的 GEMM kernel
                    try:
                        # 定义 benchmark 函数，用于测量已编译 GEMM kernel 的执行时间
                        cur_time = benchmark(
                            partial(compiled_gemm, a, b, c, stream),
                        )
                    except Exception as e:
                        print(f"Execution error: {e}")
                        cur_time = float("inf")
                    if cur_time < min_time:
                        min_time = cur_time
                        best_kernel = compiled_gemm
    if best_kernel is None:
        raise ValueError("No best kernel found")
    return best_kernel
```

这种暴力穷举的方法可以确保找到最优参数，但代价是需要尝试所有可能性。对于更高级的用例，用户可以探索更复杂的优化技术，例如搜索空间剪枝（search space pruning）与遗传算法（genetic algorithms），以降低调优开销，并更高效地发现更优配置。

为了进一步优化调优性能，我们可以利用缓存机制来避免重复计算。例如，可以把调优结果缓存在一个“输入 -> kernel”的字典中（如 `input_kernel_dict`）。当处理具有相同 `config_key` 的输入时，可以直接复用缓存的 kernel，而无需重新调优。`config_key` 与输入张量的特征有关，例如 shape、数据类型等。`config_key` 的设计非常灵活，用户可以根据自己的应用进行定制。例如，如果用户应用中的数据类型是固定的，我们可以使用输入张量的 shape 作为 key，即 `(m, n, k)`。为了进一步降低调优开销，也可以考虑使用简化的 key，例如 `config_key = (power_of_2(m), power_of_2(n), power_of_2(k))`，其中 `m`、`n`、`k` 会向上舍入到最接近的 2 的幂。该简化可以显著减少唯一 key 的数量，同时在大多数情况下仍保持良好性能。不过，务必验证这种近似不会对你的特定用例造成负面影响。

```python
config_key = (m, n, k)
if config_key in input_kernel_dict:
    compiled_gemm = input_kernel_dict[config_key]
else:
    compiled_gemm = autotune_gemm(...)
    input_kernel_dict[config_key] = compiled_gemm

# 启动 gemm kernel
compiled_gemm(a_tensor, b_tensor, c_tensor, stream)
```

通过以上方法，你可以定制自己的 auto-tuner，为特定的矩阵维度与数据类型找到最优的 GEMM kernel 配置，从而显著提升模型的计算性能。

# 教育笔记本（Educational Notebooks）

CUTLASS GitHub 仓库中提供了一些用于学习的 notebook。下面给出若干链接：

- “Hello world”（入门）(github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks/hello_world.ipynb)
- Printing（打印）(https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/notebooks/print.ipynb)
- Data Types Basics（数据类型基础）(https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/notebooks/print.ipynb)
- Tensors（张量）(github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks/tensor.ipynb)
 - The TensorSSA Abstraction（TensorSSA 抽象）(github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks/tensorssa.ipynb)
 - Layout Algebra（布局代数）(https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks/cute_layout_algebra.ipynb)
 - Element-wise Add Tutorial（逐元素加法教程）(https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/notebooks/elementwise_add.ipynb)
 - Using CUDA Graphs（使用 CUDA Graphs）(github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks/cuda_graphs.ipynb)
 
 
 

# 使用 TVM FFI 编译

Apache TVM FFI 是一个面向机器学习系统的开放 ABI 与 FFI。更多信息可参考官方文档(https://tvm.apache.org/ffi/)。

要安装 TVM FFI，你可以运行以下命令：

```shell
pip install apache-tvm-ffi
# 可选：用于提升 torch tensor 调用性能的包
pip install torch-c-dlpack-ext
```

在 CuTe DSL 中，可以为 JIT 编译函数启用 TVM FFI 作为一个选项。使用 TVM FFI 可以加速 JIT 函数调用，并提供与机器学习框架更好的互操作能力（例如直接把 `torch.Tensor` 作为参数传入）。

## 在 CuTe DSL 中启用 Apache TVM FFI

首先，请按照其安装指南(https://tvm.apache.org/ffi/#installation) 安装 `tvm-ffi` 包。

在 CuTe DSL 中启用 TVM FFI 有两种方式：

1. 在 `cute.compile` 中使用 `options` 参数来指定 TVM FFI 选项。例如：

```python
# 假设你已经定义了一个由 @cute.jit 装饰的函数 `add`
def example_compile():
   a_torch = torch.randn(10, 20, 30).to(torch.float16)
   b_torch = torch.randn(10, 20, 30).to(torch.float16)
   a_cute = cute.runtime.from_dlpack(a_torch, enable_tvm_ffi=True).mark_layout_dynamic()
   b_cute = cute.runtime.from_dlpack(b_torch, enable_tvm_ffi=True).mark_layout_dynamic()

   compiled_add = cute.compile(add, a_torch, b_torch, options="--enable-tvm-ffi")
```

注意：`cute.compile` 返回的对象是一个特定于 TVM FFI 的 Python 函数。

2. 另外，你也可以通过设置环境变量 `CUTE_DSL_ENABLE_TVM_FFI=1` 来全局启用 TVM FFI。请注意，这个设置会对当前环境内的所有 JIT 编译生效。

## 最小化 Host 开销

在一些对延迟敏感的应用中，CPU host 侧的 eager kernel 调用开销有时会成为瓶颈。TVM FFI 可以显著降低这部分开销。为了最大化性能收益，我们建议按如下方式组织你的工作流（后续小节会给出更详细的说明）：

- **启用 TVM FFI 编译 kernel**。
- **使用假张量（fake tensors）声明 shape 约束**，并在整个执行过程中复用已编译函数。
- **直接把 PyTorch 张量传给已编译函数**，避免显式 DLPack 转换。
- **使用环境 stream flag** 来隐式传递当前 PyTorch stream。
- **依赖编译后的参数校验（argument-validation）**，而不是 Python 侧的属性校验，因为 TVM FFI 函数会执行快速的编译态检查。

按以上步骤操作，可以显著降低 eager kernel 执行的 host 侧开销。下面的小节会对每一步给出更详细的示例与解释。阅读完实现细节后，你也可以回头参考这份总结。

## 用于编译的假张量（Fake tensor）

TVM FFI 函数可以接受 DLPack 兼容的张量作为参数，例如来自 torch 或 jax 的张量。但在编译期间，需要在 CuTe DSL 中指定这些张量的动态属性。为了清晰地区分“编译阶段”和“运行阶段”，CuTe DSL 提供了一种可用于编译的“假张量（fake tensor）”。例如：

```python
import cutlass.cute as cute
import torch

@cute.kernel
def device_add_one(a: cute.Tensor, b: cute.Tensor):
   threads_per_block = 128
   cta_x_, _, _ = cute.arch.block_idx()
   tid_x, _, _ = cute.arch.thread_idx()
   tid = cta_x_ * threads_per_block + tid_x
   if tid < a.shape[0]:
      b[tid] = a[tid] + 1.0

@cute.jit
def add_one(a: cute.Tensor, b: cute.Tensor):
   n = a.shape[0]
   threads_per_block = 128
   blocks = (n + threads_per_block - 1) // threads_per_block
   device_add_one(a, b).launch(
      grid=(blocks, 1, 1),
      block=(threads_per_block, 1, 1),
   )

def example_add_one():
   n = cute.sym_int()
   a_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   b_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   # 使用 "--enable-tvm-ffi" 选项和示例输入张量来编译 kernel
   compiled_add_one = cute.compile(add_one, a_cute, b_cute, options="--enable-tvm-ffi")
   # 此时 compiled_add_one 是一个 TVM-FFI 函数，可以用 torch.Tensor 作为输入来调用
   a_torch = torch.arange(10, dtype=torch.float32, device="cuda")
   b_torch = torch.empty(10, dtype=torch.float32, device="cuda")
   compiled_add_one(a_torch, b_torch)
   print("result of b_torch after compiled_add_one(a_torch, b_torch)")
   print(b_torch)
```

 假张量（fake tensor）是一种占位符，它模拟真实张量的接口，但不包含真实数据，也不允许进行索引访问。它用于只需要 shape/type/layout 信息的编译或测试场景。任何尝试访问或修改数据的操作都会抛出错误。

## 关于 Stride Order 的说明

注意：CuTe 的约定是从左到右书写各维度的 stride order，order 数字越小表示优先级越高。在 `make_fake_compact_tensor` API 的语境下，对于 shape `(2, 3, 4)` 且 stride order 为 `(0, 1, 2)` 的情况，其 stride 为 `(1, 2, 6)`，这通常被称为列主序（column-major order）。如果你想创建一个紧凑的行主序（row-major order）假张量，应当在调用 `make_fake_compact_tensor` 时显式传入 `stride_order=tuple(reversed(range(len(shape))))`。另外，你也可以通过 `make_fake_tensor` API 的 `stride` 参数始终精确地控制 stride。

## 用于 TVM FFI 的 `cute.Tensor` 适配器

要让 `cute.Tensor` 适配 TVM FFI 函数，你可以使用带有 `enable_tvm_ffi=True` 选项的 `cute.runtime.from_dlpack`，或者设置环境变量 `CUTE_DSL_ENABLE_TVM_FFI=1`。例如：

```python
def example_from_dlpack():
    a_cute = cute.runtime.from_dlpack(a_torch, enable_tvm_ffi=True).mark_layout_dynamic()
    b_cute = cute.runtime.from_dlpack(b_torch, enable_tvm_ffi=True).mark_layout_dynamic()

    compiled_add_one(a_cute, b_cute)
```

注意：由于 `cute.runtime.from_dlpack` 会执行一次显式 DLPack 转换，因此它比直接传递 `torch.Tensor` 更低效。你也可以在 `cute.compile` 中使用 `cute.Tensor` 作为参数提示（argument hint）。

```python
compiled_add_one = cute.compile(add_one, a_cute, b_cute, options="--enable-tvm-ffi")
```

## 与 torch Tensor 协作

正如上面示例所展示的那样，经 TVM FFI 编译的函数可以直接接受 `torch.Tensor` 对象（以及其他 DLPack 兼容张量）作为输入。得益于优化后的调用路径，这类函数额外开销很小，从而能实现更快的 eager 调用。

## 与 Stream 协作

在很多情况下，CuTe kernel 需要在指定的 CUDA stream 上运行。CuTe DSL 通过 TVM FFI 提供了两种与 stream 协作的方式。第一种是把 stream 作为参数显式传入。下面的示例演示了这种方式；该函数接受 `torch.cuda.Stream`、`CUstream` 或任何实现了 CUDA stream 协议的 stream 类。

```python
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream

@cute.kernel
def device_add_one(a: cute.Tensor, b: cute.Tensor):
   threads_per_block = 128
   cta_x_, _, _ = cute.arch.block_idx()
   tid_x, _, _ = cute.arch.thread_idx()
   tid = cta_x_ * threads_per_block + tid_x
   if tid < a.shape[0]:
      b[tid] = a[tid] + 1.0

@cute.jit
def add_one_with_stream(a: cute.Tensor, b: cute.Tensor, stream: CUstream):
   n = a.shape[0]
   threads_per_block = 128
   blocks = (n + threads_per_block - 1) // threads_per_block
   device_add_one(a, b).launch(
      grid=(blocks, 1, 1),
      block=(threads_per_block, 1, 1),
      stream=stream,
   )

def example_add_one_with_stream():
   n = cute.sym_int()
   a_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   b_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   # Fake stream 是 stream 参数的占位符
   stream = cute.runtime.make_fake_stream()
   compiled_add_one = cute.compile(
      add_one_with_stream, a_cute, b_cute, stream, options="--enable-tvm-ffi"
   )
   a_torch = torch.arange(10, dtype=torch.float32, device="cuda")
   b_torch = torch.empty(10, dtype=torch.float32, device="cuda")
   torch_stream = torch.cuda.current_stream()
   compiled_add_one(a_torch, b_torch, torch_stream)
   torch_stream.synchronize()
   print("result of b_torch after compiled_add_one(a_torch, b_torch, torch_stream)")
   print(b_torch)
```

## 使用环境 Stream

第二种方式是依赖环境 stream flag。给 `make_fake_stream` 传入 `use_tvm_ffi_env_stream=True`，即可把 stream 参数标记为“环境 stream”，这意味着它不再需要显式传入。TVM FFI 会自动使用其环境 stream（即当前 PyTorch stream）作为 stream 参数。下面示例演示了这一流程：

```python
def example_add_one_with_env_stream():
   n = cute.sym_int()
   a_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   b_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   # Fake stream 是 stream 参数的占位符
   # 我们将使用 TVM FFI 的环境 stream
   stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
   compiled_add_one = cute.compile(
      add_one_with_stream, a_cute, b_cute, stream, options="--enable-tvm-ffi"
   )
   a_torch = torch.arange(10, dtype=torch.float32, device="cuda")
   b_torch = torch.empty(10, dtype=torch.float32, device="cuda")
   torch_stream = torch.cuda.current_stream()
   with torch.cuda.stream(torch_stream):
      # 无需显式传入 stream；环境 stream 会在函数调用前
      # 与 torch.cuda.current_stream() 同步。
      compiled_add_one(a_torch, b_torch)
   torch_stream.synchronize()
   print("result of b_torch after compiled_add_one(a_torch, b_torch)")
   print(b_torch)
```

使用环境 stream flag 不仅能加速调用，还能简化与 PyTorch 等框架的集成，因为不需要显式 stream 参数。我们建议使用环境 stream flag，以同时简化框架集成并最小化 host 侧调用开销。

## 使用元组（Tuples）

TVM FFI 函数也可以接受元组作为参数。元组可以递归地由 TVM FFI 支持的类型组合而成。下面的示例展示了如何使用元组作为参数：

```python
import torch
from cutlass import cute

@cute.kernel
def device_add_one(a: cute.Tensor, b: cute.Tensor, c: cute.Float32):
   threads_per_block = 128
   cta_x_, _, _ = cute.arch.block_idx()
   tid_x, _, _ = cute.arch.thread_idx()
   tid = cta_x_ * threads_per_block + tid_x
   if tid < a.shape[0]:
      b[tid] = a[tid] + c

@cute.jit
def add_one_with_tuple(a: Tuple[cute.Tensor, cute.Tensor, cute.Float32]):
   n = a[0].shape[0]
   threads_per_block = 128
   blocks = (n + threads_per_block - 1) // threads_per_block
   device_add_one(a[0], a[1], a[2]).launch(grid=(blocks, 1, 1), block=(threads_per_block, 1, 1))

def example_add_one_with_tuple():
   n = cute.sym_int()
   a_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   b_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   compiled_add_one = cute.compile(
      add_one_with_tuple, (a_cute, b_cute, cute.Float32(4)),
      options="--enable-tvm-ffi"
   )
   a_torch = torch.arange(10, dtype=torch.float32, device="cuda")
   b_torch = torch.empty(10, dtype=torch.float32, device="cuda")
   compiled_add_one((a_torch, b_torch, 5))
   print("result of b_torch after compiled_add_one((a_torch, b_torch, 5))")
   print(b_torch)

example_add_one_with_tuple()
```

## 使用可变长元组（Variadic Tuples）

有时，把一个元组标注为“没有显式元素类型”会很有用。这有助于构建一个通用模板，用于接受可变数量元素的函数。编译后的函数签名将由传给 `cute.compile` 的元组实参决定。下面的示例展示了如何使用可变长元组来构建这样的通用模板。

```python
import cutlass
import torch
from cutlass import cute

@cute.kernel
def device_add_one(a: cute.Tensor, b: cute.Tensor, extra_value: tuple):
   threads_per_block = 128
   cta_x_, _, _ = cute.arch.block_idx()
   tid_x, _, _ = cute.arch.thread_idx()
   tid = cta_x_ * threads_per_block + tid_x
   if tid < a.shape[0]:
      if cutlass.const_expr(len(extra_value) != 0):
            b[tid] = a[tid] + 1 + extra_value[0]
      else:
            b[tid] = a[tid] + 1

@cute.jit
def add_one_with_extra_value(a: cute.Tensor, b: cute.Tensor, extra_value: tuple):
   n = a.shape[0]
   threads_per_block = 128
   blocks = (n + threads_per_block - 1) // threads_per_block
   device_add_one(a, b, extra_value).launch(grid=(blocks, 1, 1), block=(threads_per_block, 1, 1))

def example_add_one_with_variadic_tuple():
   n = cute.sym_int()
   a_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   b_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   compiled_add_one_no_extra = cute.compile(
      add_one_with_extra_value, a_cute, b_cute, (),
      options="--enable-tvm-ffi"
   )
   compiled_add_one_with_extra = cute.compile(
      add_one_with_extra_value, a_cute, b_cute, (cute.Float32(4),),
      options="--enable-tvm-ffi"
   )
   a_torch = torch.arange(10, dtype=torch.float32, device="cuda")
   b_torch = torch.empty(10, dtype=torch.float32, device="cuda")
   compiled_add_one_no_extra(a_torch, b_torch, ())
   print("result of b_torch after compiled_add_one_no_extra(a_torch, b_torch, ())")
   print(b_torch)
   compiled_add_one_with_extra(a_torch, b_torch, (4,))
   print("result of b_torch after compiled_add_one_with_extra(a_torch, b_torch, (4,))")
   print(b_torch)

example_add_one_with_variadic_tuple()
```

## 使用命名元组（Named Tuples）

命名元组同样受支持，并且有助于在逻辑上把相关参数分组。下面示例展示了如何使用命名元组作为参数。在底层实现中，命名元组会在 ABI 层以“未命名元组”的形式传递。当发生错误时，错误信息中的函数签名也会显示未命名元组参数。请确保编译期 CuTe 命名元组类型定义的字段与运行期 PyTorch 命名元组一致。目前用户需要在条件语句之外显式解包命名元组，然后在条件语句内部使用解包后的变量。

```python
from typing import NamedTuple
from cutlass import cute
import torch

class CuteNamedTuple(NamedTuple):
   a: cute.Tensor
   b: cute.Tensor
   c: cute.Float32 = cute.Float32(1)

   def __new_from_mlir_values__(self, values):
      return CuteNamedTuple(*values)

class TorchNamedTuple(NamedTuple):
   a: torch.Tensor
   b: torch.Tensor
   c: float = 1

@cute.kernel
def device_add_one_named_tuple(value: CuteNamedTuple):
   tid = cute.arch.block_idx()[0] * 128 + cute.arch.thread_idx()[0]
   # 需要在条件语句之外解包 namedtuple
   a = value.a
   b = value.b
   c = value.c
   if tid < a.shape[0]:
      b[tid] = a[tid] + c

@cute.jit
def add_one_with_named_tuple(value: CuteNamedTuple):
   n = value.a.shape[0]
   threads_per_block = 128
   blocks = (n + threads_per_block - 1) // threads_per_block
   device_add_one_named_tuple(value).launch(grid=(blocks, 1, 1), block=(threads_per_block, 1, 1))

def example_add_one_with_named_tuple():
   n = cute.sym_int()
   a_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   b_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))

   compiled_add_one = cute.compile(
      add_one_with_named_tuple, CuteNamedTuple(a=a_cute, b=b_cute),
      options="--enable-tvm-ffi"
   )
   a_torch = torch.arange(10, dtype=torch.float32, device="cuda")
   b_torch = torch.empty(10, dtype=torch.float32, device="cuda")
   compiled_add_one(TorchNamedTuple(a=a_torch, b=b_torch))
   print("result of b_torch")
   print(b_torch)

example_add_one_with_named_tuple()
```


## 支持的类型

TVM FFI 函数支持以下 CuTe DSL 特有类型作为参数：

- `cute.Tensor`
- `cutlass.Boolean`, `cutlass.Int8`, `cutlass.Int16`, `cutlass.Int32`, `cutlass.Int64`, `cutlass.Uint8`, `cutlass.Uint16`, `cutlass.Uint32`, `cutlass.Uint64`, `cutlass.Float32`, `cutlass.Float64`
- `cute.Shape`, `cute.Stride`, `cute.Coord`, `cute.Tile`, `cute.IntTuple`

| 编译期类型 | 调用期类型 |
| --- | --- |
| `cute.Pointer` | `ctypes.c_void_p` or a class that implements `__tvm_ffi_opaque_ptr__` protocol. |
| `cute.runtime.FakeTensor` | `torch.Tensor` and other DLPack-compatible tensors. |
| Scalar types (e.g. `cutlass.Boolean`, `cutlass.Int32`) | Python scalars (e.g. True, 123). |
| CuTe algebra types (e.g. `cute.Shape`, `cute.Stride`) | `tvm_ffi.Shape` or python tuple of ints. |
| CUDA stream `cuda.CUstream` | A stream class that implements the CUDA stream protocol (e.g. `torch.cuda.Stream`, `cuda.CUstream`). |
| Tuple of types (e.g. `Tuple[cute.Tensor, cute.Tensor, cutlass.Int32]`) | Python tuple of corresponding call-time types. |

## 错误处理

TVM FFI 函数会启用参数校验（validation），以确保传入实参符合用户声明的期望类型与取值约束。这些检查会被编译进函数中，执行速度很快，并且在函数调用时没有可观察到的额外开销。每一种错误都会被转换为可捕获与处理的 Python 异常。下面的示例展示了一些可被检查到的错误场景：

```python
def example_constraint_checks():
   n = cute.sym_int(divisibility=16)
   # 假设对齐到 16 bytes（4 个 int32），并且 a 与 b 应共享同一个 shape 变量 n
   a_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,), assumed_align=16)
   b_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,), assumed_align=16)
   compiled_add_one = cute.compile(add_one, a_cute, b_cute, options="--enable-tvm-ffi")
   a = torch.zeros(128, dtype=torch.float32, device="cuda")
   b = torch.zeros(128, dtype=torch.float32, device="cuda")

   try:
      # 触发类型不匹配错误：因为我们期望 a 和 b 都是 float32
      compiled_add_one(a, 1)
   except TypeError as e:
      # Mismatched type on argument #1 when calling:
      # `add_one(a: Tensor([n0], float32), b: Tensor([n0], float32))`,
      # expected Tensor
      print(f"TypeError: {e}")

   try:
      # 触发 shape 不匹配错误：因为我们期望 a 和 b 都具有 shape [n]
      compiled_add_one(a, b[:126])
   except ValueError as e:
      # Mismatched b.shape[0] on argument #1 when calling:
      # `add_one(a: Tensor([n0], float32), b: Tensor([n0], float32))`,
      # expected to match a.shape[0]
      print(f"ValueError: {e}")

   try:
      # 触发可整除性不匹配错误：因为 126 不能被 16 整除
      compiled_add_one(a[:126], b[:126])
   except ValueError as e:
      # Invalid a.shape[0] on argument #0 when calling:
      # `add_one(a: Tensor([n0], float32), b: Tensor([n0], float32)`,
      # expected to be divisible by 16
      print(f"ValueError: {e}")

   try:
      a = torch.zeros(129, dtype=torch.float32, device="cuda")
      b = torch.zeros(129, dtype=torch.float32, device="cuda")
      # 触发数据对齐不匹配错误：因为 x 和 y 没有按 16 bytes 对齐
      compiled_add_one(a[1:], b[1:])
   except ValueError as e:
      # raises: Misaligned Tensor data on argument #0 when calling:
      # `add_one(a: Tensor([n0], float32), b: Tensor([n0], float32)`,
      # expected data alignment=16 bytes
      print(f"ValueError: {e}")
```

TVM FFI 函数也会把遇到的任何 CUDA 错误自动转换为 Python 异常。

```python
@cute.jit
def add_one_invalid_launch(a: cute.Tensor, b: cute.Tensor):
   # 故意超过最大 block 维度（1024 threads），使得
   # CUDA runtime 报告无效配置错误。
   device_add_one(a, b).launch(grid=(1, 1, 1), block=(4096, 1, 1))

def example_error_cuda_error():
   a_torch = torch.zeros((10,), dtype=torch.float32, device="cuda")
   b_torch = torch.zeros((10,), dtype=torch.float32, device="cuda")

   a_cute = cute.runtime.from_dlpack(a_torch, enable_tvm_ffi=True)
   b_cute = cute.runtime.from_dlpack(b_torch, enable_tvm_ffi=True)
   compiled_add_one_invalid_launch = cute.compile(
      add_one_invalid_launch, a_cute, b_cute, options="--enable-tvm-ffi"
   )

   try:
      compiled_add_one_invalid_launch(a_torch, b_torch)
   except RuntimeError as e:
      # raises RuntimeError: CUDA Error: cudaErrorInvalidValue
      print(f"RuntimeError: {e}")
```

## 与设备（Devices）协作

TVM FFI 编译函数可以自然地跨 GPU 设备工作。第一个输入的 GPU 张量的 device index 决定了 kernel 的 device context。TVM FFI 函数会在启动 kernel 前调用 `cudaSetDevice`，根据该张量的 device index 设置正确的设备。对于传入原始指针而非张量的高级场景，你应当通过 CUDA Python API 显式调用 `cudaSetDevice`。

## 导出已编译模块

TVM FFI 函数支持将已编译模块导出为 object 文件，以便后续使用。例如：

```python
import subprocess
import cutlass.cute as cute

def example_add_one_export():
   n = cute.sym_int()
   a_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   b_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   # 使用 "--enable-tvm-ffi" 选项和示例输入张量来编译 kernel
   compiled_add_one = cute.compile(add_one, a_cute, b_cute, options="--enable-tvm-ffi")
   # 将已编译模块导出为 object 文件
   compiled_add_one.export_to_c("./add_one.o", function_name="add_one")
   # obtain necessary runtime libs for loading the shared library
   runtime_libs = cute.runtime.find_runtime_libraries(enable_tvm_ffi=True)
   # compile the object file to a shared library
   cmd = ["gcc", "-shared", "-o", "./add_one.so", "./add_one.o", *runtime_libs]
   print(cmd)
   subprocess.run(cmd, check=True)
   print(f"Successfully created shared library: ./add_one.so")
```

然后你可以把导出的模块重新加载回来，并用不同方式使用它：

```python
import torch
from cutlass import cute

def example_load_module_add_one():
   mod = cute.runtime.load_module("./add_one.so")
   a_torch = torch.arange(10, dtype=torch.float32, device="cuda")
   b_torch = torch.empty(10, dtype=torch.float32, device="cuda")
   mod.add_one(a_torch, b_torch)
   print("result of b_torch after mod.add_one(a_torch, b_torch)")
   print(b_torch)
```

导出的 object 文件会暴露一个与 TVM FFI 兼容的函数符号 `__tvm_ffi_add_one`，可在多种框架与编程语言中使用。你既可以构建一个 shared library 并重新加载，也可以把该 object 文件直接链接到你的应用中，并通过 TVM FFI 的 `InvokeExternC` 机制来调用该函数。更多信息请参考官方文档中的快速开始指南(https://tvm.apache.org/ffi/get_started/quickstart)。

当你构建自己的库时，请确保链接了必要的运行时库。你可以使用 `cute.runtime.find_runtime_libraries(enable_tvm_ffi=True)` 获取这些库的路径。`cute.runtime.load_module` 会在加载导出模块之前自动加载这些库。在更高级的场景中，你也可以手动加载这些库。

## 关键字参数与默认值

`cute.compile` 返回的函数支持关键字参数（keyword arguments）与默认值（defaults）。下面示例展示了如何使用关键字参数与默认值：

```python
import torch
from cutlass import cute

@cute.kernel
def device_add_scalar(a: cute.Tensor, b: cute.Tensor, offset: cutlass.Float32):
   threads_per_block = 128
   cta_x_, _, _ = cute.arch.block_idx()
   tid_x, _, _ = cute.arch.thread_idx()
   tid = cta_x_ * threads_per_block + tid_x
   if tid < a.shape[0]:
      b[tid] = a[tid] + offset

@cute.jit
def add_constant(a: cute.Tensor, b: cute.Tensor, offset: cutlass.Float32=cutlass.Float32(1)):
   n = a.shape[0]
   threads_per_block = 128
   blocks = (n + threads_per_block - 1) // threads_per_block
   device_add_scalar(a, b, offset).launch(grid=(blocks, 1, 1), block=(threads_per_block, 1, 1))

def example_kwargs_and_defaults():
   n = cute.sym_int()
   a_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   b_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   compiled_add_constant = cute.compile(add_constant, a_cute, b_cute, options="--enable-tvm-ffi")
   a_torch = torch.arange(10, dtype=torch.float32, device="cuda")
   b_torch = torch.empty(10, dtype=torch.float32, device="cuda")
   compiled_add_constant(a_torch, b_torch)
   print("result of b_torch after compiled_add_constant(a_torch, b_torch)")
   print(b_torch)
   compiled_add_constant(a_torch, b_torch, offset=4)
   print("result of b_torch after compiled_add_constant(a_torch, b_torch, offset=4)")
   print(b_torch)
```

出于效率与可移植性考虑，TVM FFI ABI 支持“仅位置参数”（positional-only arguments）的函数。如果你把已编译模块导出为 object 文件并重新加载，那么该函数将只接受按函数签名顺序传入的位置参数。你可以重新封装（rewrap）该函数，或使用 TVM FFI wrapper generator 生成一个 kwargs wrapper。下面的代码块展示了如何实现：

```python
def example_kwargs_and_defaults():
   n = cute.sym_int()
   a_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   b_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   compiled_add_constant = cute.compile(add_constant, a_cute, b_cute, options="--enable-tvm-ffi")
   # 将已编译模块导出为 object 文件
   compiled_add_constant.export_to_c("./add_constant.o", function_name="add_constant")
   # 获取加载 shared library 所需的运行时库
   runtime_libs = cute.runtime.find_runtime_libraries(enable_tvm_ffi=True)
   # 将 object 文件编译为 shared library
   cmd = ["gcc", "-shared", "-o", "./add_constant.so", "./add_constant.o", *runtime_libs]
   subprocess.run(cmd, check=True)

   a_torch = torch.arange(10, dtype=torch.float32, device="cuda")
   b_torch = torch.empty(10, dtype=torch.float32, device="cuda")

   mod = cute.runtime.load_module("./add_constant.so")
   try:
      mod.add_constant(a_torch, b_torch)
   except Exception as e:
      # 会抛出缺少参数错误，因为 kwargs 与默认值信息已经丢失
      print(e)
   # 重新封装函数以恢复参数与 kwargs 支持。
   # 或者，使用 TVM FFI wrapper generator 生成 kwargs wrapper 函数。
   from tvm_ffi.utils import kwargs_wrapper
   # arg_defaults 会与参数列表末尾对齐
   wrapped_func = kwargs_wrapper.make_kwargs_wrapper(
      mod.add_constant, arg_names=["a", "b", "offset"], arg_defaults=(1,)
   )
   wrapped_func(a_torch, b_torch)
   print("result of b_torch after wrapped_func(a_torch, b_torch)")
   print(b_torch)
   # 你也可以使用原始函数的 signature 来生成 kwargs wrapper 函数。
   # 注意要排除运行期不会包含的参数，
   # 例如 'self'、constexpr、以及 env stream 参数。
   wrapped_func = kwargs_wrapper.make_kwargs_wrapper_from_signature(
      mod.add_constant, signature=inspect.signature(add_constant),
      exclude_arg_names=["self"]
   )
   wrapped_func(a_torch, b_torch, offset=4)
   print("result of b_torch after wrapped_func(a_torch, b_torch, offset=4)")
   print(b_torch)
```
 
 
 

# 限制

## 概览

CuTe DSL 是嵌入在 Python 中的一种领域特定语言（DSL）。它使用 Python 语法的一个子集来提供更简洁的编程体验。需要注意的是：CuTe DSL 在其 JIT 编译过程中**不会**实现完整的 Python 语言语义。

本节记录了 CuTe DSL 当前的限制。虽然其中一些限制可能会在未来版本中被解决，但在使用该 DSL 构建应用时，开发者应当了解这些限制。

## 主要不支持的特性

- Programmatic Dependent Launch（PDL）
- 卷积（convolutions）
- 对 ahead-of-time compilation 的完整支持
- preferred clusters
- 基于 CLC 的 tile schedulers
- EVT 支持
- Windows 支持

## 编程模型

### CuTe Layout Algebra 仅支持 32-bit

目前，CuTe layouts 仅支持 32-bit 的 shapes/strides。未来版本计划支持 64-bit 或任意位宽。

### Python 原生数据类型

CuTe DSL 支持在“元编程（meta-programming）”场景中使用 Python 数据结构，但这些结构不能被当作运行期可修改的动态值。例如，列表与字典可以在编译期用于配置 kernel 参数，或作为动态值的容器，但在 kernel 执行期间，它们的结构与组织方式不能被改变。

- 静态值（Static Values）：
  - 在 JIT 编译阶段求值
  - 编译完成后不可变
  - 大多数 Python 原生类型（lists、tuples、dictionaries）会被当作静态值处理
  - 主要用于“元编程”和配置
  - 示例：列表可以包含动态值，但其结构在 kernel 执行期间不能被修改

- 动态值（Dynamic Values）：
  - 在运行期执行阶段求值
  - 在 JIT 编译函数执行期间可修改
  - 只有一小部分 Python 类型支持作为动态值
  - 基础类型在作为函数实参传入时会被自动转换：
    - `int` -> `Int32`（未来版本可能更新为 `Int64`）
    - `bool` -> `Bool`
    - `float` -> `Float32`（未来版本可能更新为 `Float64`）

JIT 编译器会把 Python 原生类型的处理方式类比于 C++ 模板参数。编译后的代码无法操纵列表、元组、字典等复合类型的动态值。

例如，下面的代码在 JIT 函数内部无法像传统 Python 程序那样工作。

```python
@cute.jit
def foo(a: Float32, b: Float32, i: Int32, res: cute.Tensor):
    xs = [a, b]
    # 在 CuTe DSL 中，不支持用动态索引访问 list：
    res[0] = xs[i]

    if i == 0:
        # 无论 `i` 的运行期值是什么，这里都会在 list 里追加 Float32(3.0)
        xs.append(Float32(3.0))

    for i in range(10):
        # 由于该循环在编译期不会展开（unroll），
        # 这里在编译期只会向 list 追加一个元素
        xs.append(Float32(1.0))
```

### Python 函数

目前该 DSL 尚不支持 Python 函数的返回值（return values），不过这一能力计划在未来版本中提供。

示例：

```python
@cute.jit
def foo():
    return 1  # 当前在 CuTe DSL 中不支持
```

### 具有依赖类型（Dependent Type）的表达式或语句

CuTe DSL 实现了静态类型（static typing），并且不支持依赖类型（dependent types）。每个表达式的类型都必须在编译期可确定，这与采用动态类型（dynamic typing）的标准 Python 形成对比。

以下示例展示了在标准 Python 中可用，但在该 DSL 中不支持的用法：

```python
# 在标准 Python 中有效，但在 CuTe DSL 中不支持
max(int(1), float(2.0))  # => 2.0 : float
max(int(3), float(2.0))  # => 3 : int
```

在 CuTe DSL 中，类型会被提升（promoted）。例如：

```python
@cute.jit
def foo(a: Int32, b: Float32, res: cute.Tensor):
    res[0] = max(a, b)  # 类型会被自动提升为 Float32
```

以下使用内联 if-else 表达式并具有依赖类型的代码在 CuTe DSL 中不受支持：

```python
@cute.jit
def foo(cond: Boolean, a: Int32, b: Float32, res: cute.Tensor):
    res[0] = a if cond else b
```

### 控制流

在抽象语法树（AST）处理过程中，该 DSL 会把 Python 控制流语句（`if`、`for`、`while`）转换为 MLIR 中的结构化控制流（structured control flow），并遵循与依赖类型相同的约束。例如，不允许在循环体内部改变变量的类型。

- 变量必须在控制流语句之前定义
- 在整个控制流语句中必须保持类型一致性
- 不支持在 if-else 语句中提前退出或 return

以下示例展示了在标准 Python 中可用，但在该 DSL 中不支持的用法：

```python
@cute.jit
def foo():
    a = Int32(1)
    for i in range(10):
        a = Float32(2)  # 在 DSL 中不允许在循环体内改变类型
```

### 内置运算符

该 DSL 会把 `and`、`or`、`max`、`min` 等内置运算符转换为 MLIR 操作，并同样遵循依赖类型的约束。例如，`a and b` 要求 `a` 与 `b` 的类型相同。

### 特殊变量

该 DSL 将 `_` 视为一个特殊变量，表示其值应被忽略。在该 DSL 中不允许读取 `_`。

以下示例展示了在标准 Python 中可用，但在该 DSL 中不支持的用法：

```python
@cute.jit
def foo():
    _ = 1
    print(_)  # 在 DSL 中不允许
```

### 面向对象编程（OOP）

该 DSL 构建在 Python 之上，并支持使用 Python 的面向对象编程（OOP）特性在编译期进行元编程。

但与其他复合数据类型类似，当对象包含动态值时，该 DSL 对 OOP 的支持是有限的。强烈建议你避免在代码中通过类的状态在成员方法之间传递动态值。

下例展示了在未实现 `DynamicExpression` 协议时，该 DSL 中不支持的 Python 用法：

```python
class Foo:
    def __init__(self, a: Int32):
        self.a = a

    def set_a(self, i: Int32):
        self.a = i

    def get_a(self):
        return self.a


@cute.jit
def foo(a: Int32, res: cute.Tensor):
    foo = Foo(a)
    for i in range(10):
        foo.set_a(i)

    # 这会编译失败，因为 `Foo.a` 被赋值为一个在 for 循环体内部定义的局部值，
    # 而该值在循环体之外不可见
    res[0] = foo.get_a()
```

上面的示例编译失败，是因为 `Foo.a` 被赋值为一个在 for 循环体内部定义的局部值，而该值在循环体之外不可见。

CuTe DSL 通过 protocol 实现了一套内部机制，为 OOP 模式提供有限支持。随着 DSL 持续演进以支持更多特性，该机制可能会发生变化；为了更好的可移植性，不建议用户在代码中直接使用。

### 原生 Python 环境中的 CuTe Layout Algebra

CuTe Layout Algebra 的全部操作与 API 都需要 JIT 编译。这些功能只能在 JIT 编译函数内部使用，无法在标准 Python 执行环境中访问。

此外，能够作为 JIT 编译函数参数传入的数据类型也受到限制，这进一步约束了它们在原生 Python 语境下的使用。只有以下 CuTe algebra 类型可作为 JIT 函数参数：`Tensor`、`Pointer`、`Shape`、`Stride`、`Coord` 与 `IntTuple`。对于 `Stride`，我们不支持在原生 Python 环境下使用 `ScaledBasis`。遗憾的是，在首个版本中，我们也不支持在原生 Python 环境下传入 `Layout`。

## 建议

为了获得可靠且可预测的结果：

- 避免在代码中使用依赖类型（dependent types）
- 对动态值实现显式的类型转换
- 清晰地区分静态值（编译期）与动态值（运行期）
- 尽可能使用类型标注，帮助 JIT 编译器识别类型以避免歧义

```python
# 展示显式类型的示例
alpha = 1.0  # 使用 `1.0` 而不是 `1` 来显式表示 float
beta = 2.0   # 或使用 `float(1)`
result = max(alpha, beta)  # 将正确执行 float 比较
```

### 调试能力

与 C++ API 相比，Python DSL 的调试工具与设施目前更受限。例如，我们不支持对 JIT 编译代码进行单步调试（single-stepping）。同时，JIT 编译代码缺少异常处理（exception handling）也会在一些情况下增加调试难度。

### 与框架集成

与某些深度学习框架的集成仍处于早期开发阶段，可能存在限制。例如，将框架张量转换为 `cute.Tensor` 已知会产生开销：由于我们通过通用 DLPack 协议进行转换（以兼容所有框架），每个张量的转换开销大约为 2us~3us。

### DSL API 与对象的哈希（Hash）

DSL API 与对象会受到 MLIR context、region 等上下文信息的影响，而这些信息跨 context 时并无意义。任何依赖 `__hash__` 的有状态设计都可能出现非预期行为。例如，`functools.lru_cache` 与 `@cute.jit` 结合时，可能会缓存来自某个 context 的 MLIR 对象，并在另一个 context 中复用。

## 未来改进

CuTe DSL 开发团队正在积极解决这些限制。后续版本将致力于：

- 支持 JIT 编译函数的返回值
- 改进内置运算符支持，以处理更多不含依赖类型的场景
- 增强调试能力与工具
- 改进错误信息，提供更精确的诊断信息
- 扩展对更多数值数据类型的支持
- 通过对不同框架的原生支持来提升将框架张量转换为 `cute.Tensor` 的性能
- 提供更易用的 benchmark 方法论

## 可能会长期保留的设计限制

CuTe DSL 的核心目标是提供一种领域特定语言，用于以最优 GPU 性能表达复杂的 CUDA kernel，而不是在 GPU 硬件上执行任意 Python 代码。

以下限制很可能会作为设计取舍而长期保留：

- 复合数据结构作为动态值：列表、元组、字典等将继续作为静态容器使用。虽然它们可以存储动态值，但在 JIT 编译函数执行期间，它们的结构（添加/删除元素）不能被修改。
- 依赖类型：支持依赖类型会引入大量复杂性，并对生成代码的性能特征产生负面影响。
- CuTe Layout Algebra：我们不计划在原生 Python 环境下扩展 CuTe Layout Algebra 的支持。我们计划扩展对数据类型的支持，并允许 JIT 函数与原生 Python 代码进行交互。

# 常见问题：常规

## DSLs 会取代 C++ templates 吗？

TL;DR：不会——但从某种意义上也算是。CUTLASS 4.0（CuTe DSL）以及未来所有对 Python 原生编程模型的扩展，都不会以牺牲 CUTLASS C++ 为代价。我们将继续为所支持的架构维护并更新 CUTLASS 2.x 与 3.x 的 C++ API。不过，CUTLASS 4.x 的 CuTe DSL 在编程模型与性能上与 Blackwell 上的 CuTe C++ 完全同构（isomorphic）。我们希望社区能够拥抱这一点：让自定义 kernel 的开发更容易，同时仍保持同等的高性能。这也是我们从 NVIDIA Ampere Architecture 起，为所有架构提供 CuTe DSL 支持的原因。

## CuTe DSL、CUTLASS Python 与 CUTLASS DSLs 有什么区别？

CUTLASS Python 是通过 Python 前端实例化 C++ kernels 的 Python 接口。随着 CUTLASS 4.0 的发布，它已经被弃用（deprecated）。CUTLASS DSLs 是一组用于在 Python 中进行原生 device 编程的 Python DSL 家族。目前，这一体系仅包含我们初版发布的 CuTe DSL，但未来版本会逐步加入更高层抽象，在“控制力”和“易用性”之间做渐进式的权衡。

## 我应该学习 CUTLASS C++ 还是 Python DSLs？

我们认为 Python DSLs 能显著改善学习曲线，并建议所有新手从它们开始，因为它们免除了为 GPU kernel 编程学习 C++ 元编程的固有复杂性。并且由于 CuTe C++ 与 CuTe DSL 在编程模型与模式上完全同构，你在 DSL 中获得的知识最终也都可以迁移到 C++。

## 代码将发布在哪里？PIP wheel 还是 GitHub repo？我需要自己构建吗？

与 CUTLASS C++ 和以往的 Python DSLs 相比，这是一个重大变化。未来，GitHub 上的代码主要用于让用户提交 issues 与 pull requests。虽然你也可以配合 pip wheel 使用 GitHub 代码，但除非你在开发/修改 DSL 本身，否则我们不建议大多数用户这样做。对于其他用户，我们建议直接 `pip install nvidia-cutlass-dsl`，并将 pip wheel 作为 dialect compiler 与 DSL 实现的唯一权威来源（single source of truth）。CUTLASS 的 GitHub 仓库会提供一个 `requirements.txt` 来固定 wheel 版本，使其与开源仓库状态一致（见 Quick Start Guide）。这意味着上手 CUTLASS 比以往更简单：不再需要学习 CMake 命令行，也不再需要手动触发构建。只需安装 pip wheel 并运行示例即可。

# 迁移

## 我应该把 C++ templates 的代码迁移到 Python 吗？

几乎可以肯定不需要，除非你对 kernel 的 JIT 时间有极端苛刻的要求，而 C++ 编译时间已经成为你的阻碍。2.x 与 3.x API 将继续受支持，并且 Nvidia 在 Hopper 与 Blackwell 架构上的 3.x 实现也会持续在特性与性能上改进。

## 使用 Python 后，可移植性承诺会不同吗？

在初始发布阶段（DSL 仍处于 beta），我们不对可移植性做任何承诺，因为我们可能会对 DSL 本身进行修改。虽然我们不预期 CuTe operations 会发生变化，但 DSL 的工具函数、装饰器、以及 pipelines、schedulers 等辅助类可能会随着社区反馈不断迭代。我们鼓励用户在 beta 期间在 GitHub 上提交 issues 与讨论并提供反馈。

从长期来看，我们会继续认真对待开源社区。就像 CUTLASS 以往一样，除非必要，我们不会破坏用户代码；但如果我们认为对社区与项目的整体收益更大，我们也保留做有限 breaking changes 的权利。此类变更会提前公告，并/或在每个版本的 CHANGELOG 中明确标注。

# 技术

## 将支持哪些 NVIDIA 架构？

CuTe DSL 将支持从 NVIDIA Ampere Architecture（SM80）开始的所有 NVIDIA GPU 架构。

## 是否兼容深度学习框架（如 PyTorch、JAX）？

兼容。我们会提供工具将 DLPack 支持的张量格式转换为 `cute.Tensor`。这能让用户在使用偏好的框架编写模型代码时无需离开 Python。目前我们对 JAX 的互操作能力不如 PyTorch 成熟，不过我们正在积极改进，也欢迎在该方向做贡献。

## 会编译到 PTX 还是 SASS？

CuTe DSL 会将程序编译到 PTX。随后，我们目前使用随 CUDA toolkit 提供的 PTX 编译器将 PTX 再编译到 SASS。未来我们计划移除这一限制：当用户未安装 CUDA toolkit 时，也能使用 CUDA driver 中自带的 PTX JIT。

## 我需要使用 NVCC 或 NVRTC 吗？

不需要，`nvidia-cutlass-dsl` wheel 包含生成 GPU kernels 所需的一切。它与 CUDA 12.9 toolkit 具有相同的 driver 需求（可参考对应说明）。

## 如何调试代码？

由于 CuTe DSL 不是原生 Python，而是一种嵌入式 DSL，因此无法使用 pdb 等工具。不过如果你有 GPU kernel 编程经验，调试方法会非常类似。通常，编译期与运行期打印类型和值是最直接的方式。请参考 printing 文档，了解如何在编译期和运行期打印类型和值。你也可以使用 `cuda-gdb` 在程序中设置断点并单步执行，或使用 `compute-sanitizer` 来检测和定位程序中的 bug。随着 DSL 的成熟，我们对 Python 用户程序的源位置（source location）跟踪也会改进，从而在设置断点以及使用 nsight 等工具时提供更好的源码级映射。

## 如何在 CuTe DSL 中实现 warp specialization？

实现方式与 C++ 本质上相同，只是语法改为 Python 原生风格。详细指南请参考我们的 Control Flow 与 “Blackwell kernel example”。

## 我可以在函数里调用其他函数，或使用 OOP 吗？

可以。我们经常在函数之间相互调用，并搭建类继承体系来组织 pipelines 与 schedulers 的代码结构、实现模块化。更多细节请参考 Introduction 文档或我们的示例。

# 许可证

## CuTe DSL 与相关 GitHub samples 的许可证是什么？

在 GitHub 以及通过 nvidia-cutlass-dsl Python pip wheel 提供的 CuTe DSL 组件，均在 “NVIDIA Software End User License Agreement (EULA)” 下发布。由于 pip 包包含一个与 CUDA Toolkit 共享若干组件的编译器，因此其使用条款与限制与 CUDA SDK 类似。具体使用条款请参考 EULA。

在 GitHub 上发布的 CuTe DSL samples 与 Jupyter notebooks 采用 BSD 3-Clause License，你可以在其条款下使用与再分发。这一区分确保开发者在使用或修改代码示例时具备灵活性，而无需与受 EULA 管束的编译器与运行时组件绑定。

如果你有任何问题或需要进一步澄清，欢迎联系我们。
