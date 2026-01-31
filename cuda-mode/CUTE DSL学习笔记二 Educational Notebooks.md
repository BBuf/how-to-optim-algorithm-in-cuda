# CUTE DSL 学习笔记二：Educational Notebooks

本文档是 CUTLASS 4.3.5 CuTe DSL Educational Notebooks 的中文翻译和学习笔记。

## 目录

1. [Hello World（入门）](#1-hello-world入门)
2. [Printing（打印）](#2-printing打印)
3. [Data Types Basics（数据类型基础）](#3-data-types-basics数据类型基础)
4. [Tensors（张量）](#4-tensors张量)
5. [The TensorSSA Abstraction（TensorSSA 抽象）](#5-the-tensorssa-abstractiontensorssa-抽象)
6. [Layout Algebra（布局代数）](#6-layout-algebra布局代数)
7. [Element-wise Add Tutorial（逐元素加法教程）](#7-element-wise-add-tutorial逐元素加法教程)
8. [Using CUDA Graphs（使用 CUDA Graphs）](#8-using-cuda-graphs使用-cuda-graphs)

---

## 1. Hello World（入门）

### 简介

欢迎！在本教程中，我们将使用 CuTe DSL 编写一个在 GPU 上运行的简单"Hello World"程序。这将帮助您了解使用我们的框架进行 GPU 编程的基础知识。

### 学习内容

- 如何编写在 CPU（主机）和 GPU（设备）上运行的代码
- 如何启动 GPU 内核（在 GPU 上运行的函数）
- 基本的 CUDA 概念，如线程和线程块

### 步骤 1：导入所需的库

首先，让我们导入需要的库：

```python
import cutlass
import cutlass.cute as cute
```

### 步骤 2：编写 GPU 内核

GPU 内核是在 GPU 上运行的函数。这是一个打印"Hello World"的简单内核。

**关键概念：**
- `@cute.kernel`：此装饰器告诉 CUTLASS 此函数应在 GPU 上运行
- `cute.arch.thread_idx()`：获取当前 GPU 线程的 ID（类似工作者的 ID 号）
- 我们只希望一个线程（线程 0）打印消息以避免多次打印

```python
@cute.kernel
def kernel():
    # 获取线程索引的 x 分量（y 和 z 分量未使用）
    tidx, _, _ = cute.arch.thread_idx()
    # 只有第一个线程（线程 0）打印消息
    if tidx == 0:
        cute.printf("Hello world")
```

### 步骤 3：编写主机函数

现在我们需要一个设置 GPU 并启动内核的函数。

**关键概念：**
- `@cute.jit`：此装饰器用于在 CPU 上运行但可以启动 GPU 代码的函数
- 我们需要在使用 GPU 之前初始化 CUDA
- `.launch()` 告诉 CUDA 要使用多少块、线程、共享内存等

```python
@cute.jit
def hello_world():
    # 从主机代码打印 hello world
    cute.printf("hello world")

    # 启动内核
    kernel().launch(
        grid=(1, 1, 1),  # 单个线程块
        block=(32, 1, 1),  # 每个线程块一个 warp（32 个线程）
    )
```

### 步骤 4：运行程序

有 2 种方式可以运行程序：

1. 编译并立即运行
2. 分离编译，允许我们编译代码一次并多次运行

**注意：** 方法 2 的 `Compiling...` 在第一个内核的"Hello world"之前打印。这显示了 CPU 和 GPU 打印之间的异步行为。

```python
# 初始化用于启动内核的 CUDA 上下文，带错误检查
# 我们使上下文初始化显式化，以允许用户控制上下文创建
# 并避免多个上下文的潜在问题
cutlass.cuda.initialize_cuda_context()

# 方法 1：即时（JIT）编译 - 立即编译并运行代码
print("Running hello_world()...")
hello_world()

# 方法 2：先编译（如果要多次运行相同代码则很有用）
print("Compiling...")
hello_world_compiled = cute.compile(hello_world)

# 在编译时转储 PTX/CUBIN 文件
from cutlass.cute import KeepPTX, KeepCUBIN

print("Compiling with PTX/CUBIN dumped...")
# 或者，使用基于字符串的选项进行编译，如
# cute.compile(hello_world, options="--keep-ptx --keep-cubin") 也可以工作。
hello_world_compiled_ptx_on = cute.compile[KeepPTX, KeepCUBIN](hello_world)

# 运行预编译版本
print("Running compiled version...")
hello_world_compiled()
```

**输出：**
```
Running hello_world()...
Compiling...
hello world
Hello world
Compiling with PTX/CUBIN dumped...
Running compiled version...
hello world
Hello world
```

---

## 2. Printing（打印）

本 notebook 演示了在 CuTe 中打印值的不同方式，并解释了静态（编译时）和动态（运行时）值之间的重要区别。

### 关键概念

- **静态值**：在编译时已知
- **动态值**：仅在运行时已知
- 不同场景的不同打印方法
- CuTe 中的布局表示
- 张量可视化和格式化

```python
import cutlass
import cutlass.cute as cute
import numpy as np
```

### Print Example 函数

`print_example` 函数演示了几个重要概念：

#### 1. Python 的 `print` vs CuTe 的 `cute.printf`
- `print`：只能在编译时显示静态值
- `cute.printf`：可以在运行时显示静态和动态值

#### 2. 值类型
- `a`：动态 `Int32` 值（运行时）
- `b`：静态 `Constexpr[int]` 值（编译时）

#### 3. 布局打印
显示布局在静态与动态上下文中的不同表示：
- 静态上下文：未知值显示为 `?`
- 动态上下文：显示实际值

```python
@cute.jit
def print_example(a: cutlass.Int32, b: cutlass.Constexpr[int]):
    """
    演示 CuTe 中的不同打印方法以及它们如何处理静态与动态值。

    此示例显示：
    1. Python 的 print 函数如何在编译时处理静态值但无法显示动态值
    2. cute.printf 如何在运行时显示静态和动态值
    3. 静态与动态上下文中的类型差异
    4. 布局在两种打印方法中的表示方式

    参数：
        a: 将在运行时确定的动态 Int32 值
        b: 静态（编译时常量）整数值
    """
    # 使用 Python `print` 打印静态信息
    print(">>>", b)  # => 2
    # `a` 是动态值
    print(">>>", a)  # => ?

    # 使用 `cute.printf` 打印动态信息
    cute.printf(">?? {}", a)  # => 8
    cute.printf(">?? {}", b)  # => 2

    print(">>>", type(a))  # => <class 'cutlass.Int32'>
    print(">>>", type(b))  # => <class 'int'>

    layout = cute.make_layout((a, b))
    print(">>>", layout)  # => (?,2):(1,?)
    cute.printf(">?? {}", layout)  # => (8,2):(1,8)
```

### 编译并运行

**直接编译和运行**
- `print_example(cutlass.Int32(8), 2)`
- 一步编译并运行将执行静态和动态打印
  * `>>>` 代表静态打印
  * `>??` 代表动态打印

```python
print_example(cutlass.Int32(8), 2)
```

**输出：**
```
>>> 2
>>> ?
>>> Int32
>>> <class 'int'>
>>> (?,2):(1,?)
>?? 8
>?? 2
>?? (8,2):(1,8)
```

### 编译函数

当使用 `cute.compile(print_example, cutlass.Int32(8), 2)` 编译函数时，Python 解释器跟踪代码并仅评估静态表达式并打印静态信息。

```python
print_example_compiled = cute.compile(print_example, cutlass.Int32(8), 2)
```

**输出：**
```
>>> 2
>>> ?
>>> Int32
>>> <class 'int'>
>>> (?,2):(1,?)
```

### 调用编译的函数

仅打印运行时信息

```python
print_example_compiled(cutlass.Int32(8))
```

**输出：**
```
>?? 8
>?? 2
>?? (8,2):(1,8)
```

### 格式字符串示例

`format_string_example` 函数显示了一个重要的限制：
- CuTe 中的 F-strings 在编译时评估
- 这意味着动态值不会在 f-strings 中显示其运行时值
- 当需要查看运行时值时使用 `cute.printf`

```python
@cute.jit
def format_string_example(a: cutlass.Int32, b: cutlass.Constexpr[int]):
    """
    格式字符串在编译时评估。
    """
    print(f"a: {a}, b: {b}")

    layout = cute.make_layout((a, b))
    print(f"layout: {layout}")


print("Direct run output:")
format_string_example(cutlass.Int32(8), 2)
```

**输出：**
```
Direct run output:
a: ?, b: 2
layout: (?,2):(1,?)
```

### 打印张量示例

CuTe 通过 `print_tensor` 操作提供了用于打印张量的专门功能。`cute.print_tensor` 接受以下参数：
- `Tensor`（必需）：要打印的 CuTe 张量对象。张量必须支持加载和存储操作
- `verbose`（可选，默认=False）：控制输出详细程度的布尔标志。设置为 True 时，将打印张量中每个元素的索引详细信息。

下面的示例代码显示了 verbose ON 和 OFF 之间的区别，以及如何打印给定张量的子范围。

```python
from cutlass.cute.runtime import from_dlpack


@cute.jit
def print_tensor_basic(x: cute.Tensor):
    # 打印张量
    print("Basic output:")
    cute.print_tensor(x)


@cute.jit
def print_tensor_verbose(x: cute.Tensor):
    # 以 verbose 模式打印张量
    print("Verbose output:")
    cute.print_tensor(x, verbose=True)


@cute.jit
def print_tensor_slice(x: cute.Tensor, coord: tuple):
    # 从 3D 张量切片 2D 张量
    sliced_data = cute.slice_(x, coord)
    y = cute.make_rmem_tensor(sliced_data.layout, sliced_data.element_type)
    # 通过将切片数据加载到片段中转换为 TensorSSA 格式
    y.store(sliced_data.load())
    print("Slice output:")
    cute.print_tensor(y)
```

默认的 `cute.print_tensor` 将输出带有数据类型、存储空间、CuTe 布局信息的 CuTe 张量，并以 torch 风格格式打印数据。

```python
def tensor_print_example1():
    shape = (4, 3, 2)

    # 创建 [0,...,23] 并重塑为 (4, 3, 2)
    data = np.arange(24, dtype=np.float32).reshape(*shape)

    print_tensor_basic(from_dlpack(data))


tensor_print_example1()
```

**输出：**
```
Basic output:
tensor(raw_ptr(0x000000000a5f1d50: f32, generic, align<4>) o (4,3,2):(6,2,1), data=
       [[[ 0.000000,  2.000000,  4.000000, ],
         [ 6.000000,  8.000000,  10.000000, ],
         [ 12.000000,  14.000000,  16.000000, ],
         [ 18.000000,  20.000000,  22.000000, ]],

        [[ 1.000000,  3.000000,  5.000000, ],
         [ 7.000000,  9.000000,  11.000000, ],
         [ 13.000000,  15.000000,  17.000000, ],
         [ 19.000000,  21.000000,  23.000000, ]]])
```

verbose 打印将显示张量中每个元素的坐标详细信息。下面的示例显示了我们如何在 2D 4x3 张量空间中索引元素。

```python
def tensor_print_example2():
    shape = (4, 3)

    # 创建 [0,...,11] 并重塑为 (4, 3)
    data = np.arange(12, dtype=np.float32).reshape(*shape)

    print_tensor_verbose(from_dlpack(data))


tensor_print_example2()
```

**输出：**（示例显示坐标详细信息）

要打印给定张量中的元素子集，我们可以使用 `cute.slice_` 选择给定张量的一个范围，将它们加载到寄存器中，然后使用 `cute.print_tensor` 打印值。

要打印设备内存中的张量，您可以在 CuTe JIT 内核中使用 `cute.print_tensor`。

**注意：** 目前，`cute.print_tensor` 仅支持整数数据类型和 `Float16`/`Float32`/`Float64` 浮点数据类型的张量。我们将来会支持更多数据类型。

---

## 3. Data Types Basics（数据类型基础）

```python
import cutlass
import cutlass.cute as cute
```

### 理解 CuTe DSL 中的数据结构

在大多数情况下，CuTe DSL 中的数据结构与 Python 数据结构的工作方式相同，但显著区别是 Python 数据结构在大多数情况下被视为静态数据，这些数据由嵌入在 Python 解释器内的 DSL 编译器解释。

为了区分编译时和运行时值，CuTe DSL 引入了表示 JIT 编译代码中动态值的原始类型。

CuTe DSL 提供了一套全面的原始数字类型，用于表示运行时的动态值。这些类型在 CuTe DSL 类型系统中正式定义：

### 整数类型
- `Int8` - 8 位有符号整数
- `Int16` - 16 位有符号整数  
- `Int32` - 32 位有符号整数
- `Int64` - 64 位有符号整数
- `Int128` - 128 位有符号整数
- `Uint8` - 8 位无符号整数
- `Uint16` - 16 位无符号整数
- `Uint32` - 32 位无符号整数
- `Uint64` - 64 位无符号整数
- `Uint128` - 128 位无符号整数

### 浮点类型
- `Float16` - 16 位浮点数
- `Float32` - 32 位浮点数 
- `Float64` - 64 位浮点数
- `BFloat16` - Brain Floating Point 格式（16 位）
- `TFloat32` - Tensor Float32 格式（用于张量运算的降精度格式）
- `Float8E4M3` - 8 位浮点数，4 位指数和 3 位尾数
- `Float8E5M2` - 8 位浮点数，5 位指数和 2 位尾数

这些专门的类型旨在表示 CuTe DSL 代码中将在运行时评估的动态值，与 Python 的内置数字类型（在编译期间评估）形成对比。

### 示例用法：

```python
x = cutlass.Int32(5)        # 创建 32 位整数
y = cutlass.Float32(3.14)   # 创建 32 位浮点数

@cute.jit
def foo(a: cutlass.Int32):  # 通过 ABI 将 `a` 注释为传递给 jit 函数的 32 位整数
    ...
```

```python
@cute.jit
def bar():
    a = cutlass.Float32(3.14)
    print("a(static) =", a)  # 打印 `a(static) = ?`
    cute.printf("a(dynamic) = {}", a)  # 打印 `a(dynamic) = 3.140000`

    b = cutlass.Int32(5)
    print("b(static) =", b)  # 打印 `b(static) = 5`
    cute.printf("b(dynamic) = {}", b)  # 打印 `b(dynamic) = 5`


bar()
```

**输出：**
```
a(static) = ?
b(static) = ?
a(dynamic) = 3.140000
b(dynamic) = 5
```

### 类型转换 API

CUTLASS 数字类型通过所有数字类型上可用的 `to()` 方法提供类型转换。这允许您在运行时在不同的数字数据类型之间转换。

**语法：**

```python
new_value = value.to(target_type)
```

`to()` 方法支持以下转换：
- 整数类型（Int8、Int16、Int32、Int64、UInt8、UInt16、UInt32、UInt64）
- 浮点类型（Float16、Float32、Float64、BFloat16）
- 混合整数/浮点转换

请注意，从浮点类型转换为整数类型时，小数部分会被截断。在不同范围的类型之间转换时，如果值超过目标类型的可表示范围，则可能会被限制或损失精度。

```python
@cute.jit
def type_conversion():
    # 从 Int32 转换为 Float32
    x = cutlass.Int32(42)
    y = x.to(cutlass.Float32)
    cute.printf("Int32({}) => Float32({})", x, y)

    # 从 Float32 转换为 Int32
    a = cutlass.Float32(3.14)
    b = a.to(cutlass.Int32)
    cute.printf("Float32({}) => Int32({})", a, b)

    # 从 Int32 转换为 Int8
    c = cutlass.Int32(127)
    d = c.to(cutlass.Int8)
    cute.printf("Int32({}) => Int8({})", c, d)

    # 从 Int32 转换为 Int8，值超过 Int8 范围
    e = cutlass.Int32(300)
    f = e.to(cutlass.Int8)
    cute.printf("Int32({}) => Int8({}) (truncated due to range limitation)", e, f)


type_conversion()
```

**输出：**
```
Int32(42) => Float32(42.000000)
Float32(3.140000) => Int32(3)
Int32(127) => Int8(127)
Int32(300) => Int8(44) (truncated due to range limitation)
```

### 运算符重载

CUTLASS 数字类型支持 Python 的内置运算符，允许您编写自然的数学表达式。运算符适用于 CUTLASS 数字类型和 Python 原生数字类型。

支持的运算符包括：
- 算术：`+`、`-`、`*`、`/`、`//`、`%`、`**`
- 比较：`<`、`<=`、`==`、`!=`、`>=`、`>`
- 位运算：`&`、`|`、`^`、`<<`、`>>`
- 一元：`-`（取负）、`~`（按位 NOT）

```python
@cute.jit
def operator_demo():
    # 算术运算符
    a = cutlass.Int32(10)
    b = cutlass.Int32(3)
    cute.printf("a: Int32({}), b: Int32({})", a, b)

    x = cutlass.Float32(5.5)
    cute.printf("x: Float32({})", x)

    cute.printf("")

    sum_result = a + b
    cute.printf("a + b = {}", sum_result)

    y = x * 2  # 与 Python 原生类型相乘
    cute.printf("x * 2 = {}", y)

    # 混合类型算术（Int32 + Float32），整数被转换为 float32
    mixed_result = a + x
    cute.printf("a + x = {} (Int32 + Float32 promotes to Float32)", mixed_result)

    # 使用 Int32 的除法（注意：整数除法）
    div_result = a / b
    cute.printf("a / b = {}", div_result)

    # 浮点除法
    float_div = x / cutlass.Float32(2.0)
    cute.printf("x / 2.0 = {}", float_div)

    # 比较运算符
    is_greater = a > b
    cute.printf("a > b = {}", is_greater)

    # 位运算符
    bit_and = a & b
    cute.printf("a & b = {}", bit_and)

    neg_a = -a
    cute.printf("-a = {}", neg_a)

    not_a = ~a
    cute.printf("~a = {}", not_a)


operator_demo()
```

**输出：**
```
a: Int32(10), b: Int32(3)
x: Float32(5.500000)

a + b = 13
x * 2 = 11.000000
a + x = 15.500000 (Int32 + Float32 promotes to Float32)
a / b = 3.333333
x / 2.0 = 2.750000
a > b = 1
a & b = 2
-a = -10
~a = -11
```

---

## 4. Tensors（张量）

```python
import cutlass
import cutlass.cute as cute
```

### 张量

CuTe 中的张量是通过两个关键组件的组合创建的：

1. **引擎（Engine）** (E) - 一个随机访问、类指针对象，支持：
   - 偏移操作：`e + d → e`（按布局余域的元素偏移引擎）
   - 解引用操作：`*e → v`（解引用引擎以产生值）

2. **布局（Layout）** (L) - 定义从坐标到偏移的映射

张量正式定义为引擎 E 与布局 L 的组合，表示为 `T = E ∘ L`。在坐标 c 处评估张量时，它：

1. 使用布局将坐标 c 映射到余域
2. 相应地偏移引擎
3. 解引用结果以获得张量的值

这可以用数学方式表示为：

```
T(c) = (E ∘ L)(c) = *(E + L(c))
```

### 示例用法

这是一个使用指针和布局 `(8,5):(5,1)` 创建张量并填充 1 的简单示例：

```python
@cute.jit
def create_tensor_from_ptr(ptr: cute.Pointer):
    layout = cute.make_layout((8, 5), stride=(5, 1))
    tensor = cute.make_tensor(ptr, layout)
    tensor.fill(1)
    cute.print_tensor(tensor)
```

这创建了一个张量，其中：
- 引擎是一个指针
- 布局的形状为 `(8, 5)`，步长为 `(5, 1)`
- 生成的张量可以使用布局定义的坐标进行评估

我们可以通过使用 torch 分配缓冲区并使用指向 torch 张量的指针运行测试来测试：

```python
import torch

from cutlass.torch import dtype as torch_dtype
import cutlass.cute.runtime as cute_rt

a = torch.randn(8, 5, dtype=torch_dtype(cutlass.Float32))
ptr_a = cute_rt.make_ptr(cutlass.Float32, a.data_ptr())

create_tensor_from_ptr(ptr_a)
```

### DLPACK 支持

CuTe DSL 设计为原生支持 dlpack 协议。这提供了与支持 DLPack 的框架的轻松集成，例如 torch、numpy、jax、tensorflow 等。

有关更多信息，请参阅 DLPACK 项目：https://github.com/dmlc/dlpack

调用 `from_dlpack` 可以转换任何支持 `__dlpack__` 和 `__dlpack_device__` 的张量或 ndarray 对象。

```python
from cutlass.cute.runtime import from_dlpack


@cute.jit
def print_tensor_dlpack(src: cute.Tensor):
    print(src)
    cute.print_tensor(src)
```

```python
a = torch.randn(8, 5, dtype=torch_dtype(cutlass.Float32))

print_tensor_dlpack(from_dlpack(a))
```

```python
import numpy as np

a = np.random.randn(8, 8).astype(np.float32)

print_tensor_dlpack(from_dlpack(a))
```

### 张量评估方法

张量支持两种主要的评估方法：

#### 1. 完全评估
当应用完整坐标 c 的张量评估时，它计算偏移，将其应用于引擎，并解引用它以返回存储的值。这是您想要访问张量的特定元素的直接情况。

#### 2. 部分评估（切片）
当使用不完整的坐标 c = c' ⊕ c*（其中 c* 表示未指定的部分）进行评估时，结果是一个新张量，它是原始张量的切片，其引擎偏移以考虑提供的坐标。此操作可以表示为：

```
T(c) = (E ∘ L)(c) = (E + L(c')) ∘ L(c*) = T'(c*)
```

切片有效地减少了张量的维度，创建了可以进一步评估或操作的子张量。

```python
@cute.jit
def tensor_access_item(a: cute.Tensor):
    # 使用线性索引访问数据
    cute.printf(
        "a[2] = {} (equivalent to a[{}])",
        a[2],
        cute.make_identity_tensor(a.layout.shape)[2],
    )
    cute.printf(
        "a[9] = {} (equivalent to a[{}])",
        a[9],
        cute.make_identity_tensor(a.layout.shape)[9],
    )

    # 使用 n-d 坐标访问数据，以下两个是等效的
    cute.printf("a[2,0] = {}", a[2, 0])
    cute.printf("a[2,4] = {}", a[2, 4])
    cute.printf("a[(2,4)] = {}", a[2, 4])

    # 将值赋给 tensor@(2,4)
    a[2, 3] = 100.0
    a[2, 4] = 101.0
    cute.printf("a[2,3] = {}", a[2, 3])
    cute.printf("a[(2,4)] = {}", a[(2, 4)])


# 使用 torch 创建具有顺序数据的张量
data = torch.arange(0, 8 * 5, dtype=torch.float32).reshape(8, 5)
tensor_access_item(from_dlpack(data))

print(data)
```

### 张量作为内存视图

在 CUDA 编程中，不同的内存空间在访问速度、范围和生命周期方面具有不同的特性：

- **generic**：默认内存空间，可以引用任何其他内存空间。
- **全局内存（gmem）**：所有块的所有线程都可以访问，但延迟较高。
- **共享内存（smem）**：块内所有线程都可以访问，延迟比全局内存低得多。
- **寄存器内存（rmem）**：线程私有内存，延迟最低，但容量有限。
- **张量内存（tmem）**：NVIDIA Blackwell 架构中引入的用于张量运算的专门内存。

在 CuTe 中创建张量时，您可以指定内存空间以根据访问模式优化性能。

有关 CUDA 内存空间的更多信息，请参阅 [CUDA 编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy)。

### 坐标张量

#### 定义和属性

坐标张量 $T: Z^n → Z^m$ 是在坐标空间之间建立映射的数学结构。与将坐标映射到标量值的标准张量不同，坐标张量将坐标映射到其他坐标，形成张量运算和变换的基本构建块。

#### 示例

考虑一个 `(4,4)` 坐标张量：

**行主序布局（C 风格）：**
```
[[（0,0） (0,1) (0,2) (0,3)]
 [(1,0) (1,1) (1,2) (1,3)]
 [(2,0) (2,1) (2,2) (2,3)]
 [(3,0) (3,1) (3,2) (3,3)]]
```

**列主序布局（Fortran 风格）：**
```
[[(0,0) (1,0) (2,0) (3,0)]
 [(0,1) (1,1) (2,1) (3,1)]
 [(0,2) (1,2) (2,2) (3,2)]
 [(0,3) (1,3) (2,3) (3,3)]]
```

#### 单位张量

单位张量 $I$ 是坐标张量的一种特殊情况，实现了恒等映射函数：

 **定义：**
 对于给定的形状 $S = (s_1, s_2, \ldots, s_n)$，单位张量 $I$ 满足：

 $$
 I(c) = c, \quad \forall\, c \in \prod_{i=1}^{n} \left[0, s_i\right)
 $$

 **属性：**
 1. **双射映射**：单位张量在坐标之间建立一对一的对应关系。
 2. **布局不变性**：逻辑结构保持恒定，而不管底层内存布局如何。
3. **坐标保留**：对于任何坐标 c，I(c) = c。

CuTe 通过词典排序在 1-D 索引和 N-D 坐标之间建立同构。对于形状为 S = (s₁, s₂, ..., sₙ) 的单位张量中的坐标 c = (c₁, c₂, ..., cₙ)：

 **线性索引公式：**

 $$
 \text{idx} = c_1 + \sum_{i=2}^{n} \left(c_i \prod_{j=1}^{i-1} s_j\right)
 $$

 **示例：**
```python
 # 从给定形状创建单位张量
coord_tensor = make_identity_tensor(layout.shape())

# 使用线性索引访问坐标
coord = coord_tensor[linear_idx]  # 返回 N-D 坐标
```

这种双向映射能够有效地从线性索引转换为 N 维坐标，促进张量运算和内存访问模式。

```python
@cute.jit
def print_tensor_coord(a: cute.Tensor):
    coord_tensor = cute.make_identity_tensor(a.layout.shape)
    print(coord_tensor)
    cute.print_tensor(coord_tensor)


a = torch.randn(8, 4, dtype=torch_dtype(cutlass.Float32))
print_tensor_coord(from_dlpack(a))
```

---

## 5. The TensorSSA Abstraction（TensorSSA 抽象）

```python
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

import numpy as np
```

### CuTe DSL 中的 TensorSSA 简介

本教程介绍什么是 `TensorSSA` 以及为什么我们需要它。我们还提供了一些示例来展示如何使用 `TensorSSA`。

### 什么是 TensorSSA

`TensorSSA` 是一个 Python 类，它以静态单赋值（SSA）形式表示 CuTe DSL 中的张量值。您可以将其视为驻留在（模拟）寄存器中的张量。

### 为什么需要 TensorSSA

`TensorSSA` 将底层 MLIR 张量值封装到一个更易于在 Python 中操作的对象中。通过重载大量 Python 运算符（如 `+`、`-`、`*`、`/`、`[]` 等），它允许用户以更 Pythonic 的方式表达张量计算（主要是逐元素运算和归约）。然后将这些逐元素运算转换为优化的向量化指令。

它是 CuTe DSL 的一部分，作为用户描述的计算逻辑和较低级别 MLIR IR 之间的桥梁，特别是用于表示和操作寄存器级数据。

### 何时使用 TensorSSA

`TensorSSA` 主要用于以下场景：

#### 从内存加载和存储到内存

```python
@cute.jit
def load_and_store(res: cute.Tensor, a: cute.Tensor, b: cute.Tensor):
    """
    从内存加载数据并将结果存储到内存。

    :param res: 用于存储结果的目标张量。
    :param a: 要加载的源张量。
    :param b: 要加载的源张量。
    """
    a_vec = a.load()
    print(f"a_vec: {a_vec}")  # 打印 `a_vec: vector<12xf32> o (3, 4)`
    b_vec = b.load()
    print(f"b_vec: {b_vec}")  # 打印 `b_vec: vector<12xf32> o (3, 4)`
    res.store(a_vec + b_vec)
    cute.print_tensor(res)


a = np.ones(12).reshape((3, 4)).astype(np.float32)
b = np.ones(12).reshape((3, 4)).astype(np.float32)
c = np.zeros(12).reshape((3, 4)).astype(np.float32)
load_and_store(from_dlpack(c), from_dlpack(a), from_dlpack(b))
```

#### 寄存器级张量运算

在编写内核逻辑时，对加载到寄存器中的数据执行各种计算、转换、切片等。

```python
@cute.jit
def apply_slice(src: cute.Tensor, dst: cute.Tensor, indices: cutlass.Constexpr):
    """
    对 src 张量应用切片运算并将结果存储到 dst 张量。

    :param src: 要切片的源张量。
    :param dst: 用于存储结果的目标张量。
    :param indices: 用于切片源张量的索引。
    """
    src_vec = src.load()
    dst_vec = src_vec[indices]
    print(f"{src_vec} -> {dst_vec}")
    if cutlass.const_expr(isinstance(dst_vec, cute.TensorSSA)):
        dst.store(dst_vec)
        cute.print_tensor(dst)
    else:
        dst[0] = dst_vec
        cute.print_tensor(dst)


def slice_1():
    src_shape = (4, 2, 3)
    dst_shape = (4, 3)
    indices = (None, 1, None)

    """
    a:
    [[[ 0.  1.  2.]
      [ 3.  4.  5.]]

     [[ 6.  7.  8.]
      [ 9. 10. 11.]]

     [[12. 13. 14.]
      [15. 16. 17.]]

     [[18. 19. 20.]
      [21. 22. 23.]]]
    """
    a = np.arange(np.prod(src_shape)).reshape(*src_shape).astype(np.float32)
    dst = np.random.randn(*dst_shape).astype(np.float32)
    apply_slice(from_dlpack(a), from_dlpack(dst), indices)


slice_1()
```

### 算术运算

如前所述，有许多张量运算的操作数是 `TensorSSA`。它们都是逐元素运算。我们在下面给出一些示例。

#### 二元运算

对于二元运算，LHS 操作数是 `TensorSSA`，RHS 操作数可以是 `TensorSSA` 或 `Numeric`。当 RHS 是 `Numeric` 时，它将广播为 `TensorSSA`。

```python
@cute.jit
def binary_op_1(res: cute.Tensor, a: cute.Tensor, b: cute.Tensor):
    a_vec = a.load()
    b_vec = b.load()

    add_res = a_vec + b_vec
    cute.print_tensor(add_res)  # 打印 [3.000000, 3.000000, 3.000000]

    sub_res = a_vec - b_vec
    cute.print_tensor(sub_res)  # 打印 [-1.000000, -1.000000, -1.000000]

    mul_res = a_vec * b_vec
    cute.print_tensor(mul_res)  # 打印 [2.000000, 2.000000, 2.000000]

    div_res = a_vec / b_vec
    cute.print_tensor(div_res)  # 打印 [0.500000, 0.500000, 0.500000]

    floor_div_res = a_vec // b_vec
    cute.print_tensor(res)  # 打印 [0.000000, 0.000000, 0.000000]

    mod_res = a_vec % b_vec
    cute.print_tensor(mod_res)  # 打印 [1.000000, 1.000000, 1.000000]


a = np.empty((3,), dtype=np.float32)
a.fill(1.0)
b = np.empty((3,), dtype=np.float32)
b.fill(2.0)
res = np.empty((3,), dtype=np.float32)
binary_op_1(from_dlpack(res), from_dlpack(a), from_dlpack(b))
```

#### 一元运算

```python
@cute.jit
def unary_op_1(res: cute.Tensor, a: cute.Tensor):
    a_vec = a.load()

    sqrt_res = cute.math.sqrt(a_vec)
    cute.print_tensor(sqrt_res)  # 打印 [2.000000, 2.000000, 2.000000]

    sin_res = cute.math.sin(a_vec)
    res.store(sin_res)
    cute.print_tensor(sin_res)  # 打印 [-0.756802, -0.756802, -0.756802]

    exp2_res = cute.math.exp2(a_vec)
    cute.print_tensor(exp2_res)  # 打印 [16.000000, 16.000000, 16.000000]


a = np.array([4.0, 4.0, 4.0], dtype=np.float32)
res = np.empty((3,), dtype=np.float32)
unary_op_1(from_dlpack(res), from_dlpack(a))
```

#### 归约运算

`TensorSSA` 的 `reduce` 方法应用指定的归约操作（`ReductionOp.ADD`、`ReductionOp.MUL`、`ReductionOp.MAX`、`ReductionOp.MIN`）从初始值开始，并沿 `reduction_profile` 指定的维度执行此归约。结果通常是具有减少维度的新 `TensorSSA`，或者如果跨所有轴归约则为标量值。

```python
@cute.jit
def reduction_op(a: cute.Tensor):
    """
    对 src 张量应用归约运算。

    :param src: 要归约的源张量。
    """
    a_vec = a.load()
    red_res = a_vec.reduce(cute.ReductionOp.ADD, 0.0, reduction_profile=0)
    cute.printf(red_res)  # 打印 21.000000

    red_res = a_vec.reduce(cute.ReductionOp.ADD, 0.0, reduction_profile=(None, 1))
    cute.print_tensor(red_res)  # 打印 [6.000000, 15.000000]

    red_res = a_vec.reduce(cute.ReductionOp.ADD, 1.0, reduction_profile=(1, None))
    cute.print_tensor(red_res)  # 打印 [6.000000, 8.000000, 10.000000]


a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
reduction_op(from_dlpack(a))
```

### 广播

`TensorSSA` 支持遵循 NumPy 广播规则的广播运算。广播允许您在满足某些条件时对不同形状的数组执行运算。关键规则是：

1. 源形状用 1 填充以匹配目标形状的秩
2. 源形状的每个模式的大小必须为 1 或等于目标形状
3. 广播后，所有模式都应匹配目标形状

让我们看一些实际的广播示例：

```python
import cutlass
import cutlass.cute as cute


@cute.jit
def broadcast_examples():
    a = cute.make_rmem_tensor((1, 3), dtype=cutlass.Float32)
    a[0] = 0.0
    a[1] = 1.0
    a[2] = 2.0
    a_val = a.load()
    cute.print_tensor(a_val.broadcast_to((4, 3)))
    # tensor(raw_ptr(0x00007ffe26625740: f32, rmem, align<32>) o (4,3):(1,4), data=
    #    [[ 0.000000,  1.000000,  2.000000, ],
    #     [ 0.000000,  1.000000,  2.000000, ],
    #     [ 0.000000,  1.000000,  2.000000, ],
    #     [ 0.000000,  1.000000,  2.000000, ]])

    c = cute.make_rmem_tensor((4, 1), dtype=cutlass.Float32)
    c[0] = 0.0
    c[1] = 1.0
    c[2] = 2.0
    c[3] = 3.0
    cute.print_tensor(a.load() + c.load())
    # tensor(raw_ptr(0x00007ffe26625780: f32, rmem, align<32>) o (4,3):(1,4), data=
    #        [[ 0.000000,  1.000000,  2.000000, ],
    #         [ 1.000000,  2.000000,  3.000000, ],
    #         [ 2.000000,  3.000000,  4.000000, ],
    #         [ 3.000000,  4.000000,  5.000000, ]])


broadcast_examples()
```

上面的示例演示了两个关键的广播场景，展示了 `TensorSSA` 如何自动处理算术运算中行向量和列向量的广播，遵循每个维度必须为 1 或匹配目标大小的广播规则。广播在运算期间隐式处理，使得使用不同形状的张量变得容易。

---

## 6. Layout Algebra（布局代数）

### 使用 Python DSL 的 CuTe 布局代数

参考 CuTe C++ 的 [01_layout.md](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/01_layout.md) 和 [02_layout_algebra.md](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/02_layout_algebra.md) 文档，我们总结：

在 CuTe 中，`Layout`：
- 由一对 `Shape` 和 `Stride` 定义
- 将坐标空间映射到索引空间
- 支持静态（编译时）和动态（运行时）值

CuTe 还提供了一组强大的操作——*布局代数*——用于组合和操作布局，包括：
- 布局组合：布局的函数组合
- 布局"除法"：将布局拆分为两个组件布局
- 布局"乘积"：根据另一个布局重现布局

在本 notebook 中，我们将演示：
1. 如何使用 Python DSL 的 CuTe 关键布局代数运算
2. 静态和动态布局在 Python DSL 中打印或操作时的行为方式

我们使用来自 [02_layout_algebra.md](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/02_layout_algebra.md) 的示例，我们建议读者阅读以获取更多详细信息。

```python
import cutlass
import cutlass.cute as cute
```

### 布局代数运算

这些运算形成了 CuTe 布局操作能力的基础，使得：
- 高效的数据Tile 和分区
- 使用规范类型表示线程和数据布局以分离两者
- 原生描述和操作对张量核心程序至关重要的线程和数据的分层张量
- 混合静态/动态布局转换
- 布局代数与张量运算的无缝集成
- 将复杂的 MMA 和复制表达为规范循环

### 1. Coalesce（合并）

`coalesce` 操作通过在可能的情况下展平和组合模式来简化布局，而不改变其大小或作为整数函数的行为。

它确保后置条件：
- 保留大小：cute.size(layout) == cute.size(result)
- 展平：depth(result) <= 1
- 保留函数性：对于所有 i，0 <= i < cute.size(layout)，layout(i) == result(i)

#### 示例

- 基本 Coalesce 示例：

```python
@cute.jit
def coalesce_example():
    """
    演示 coalesce 操作展平和组合模式
    """
    layout = cute.make_layout(
        (2, (1, 6)), stride=(1, (cutlass.Int32(6), 2))
    )  # 动态步长
    result = cute.coalesce(layout)

    print(">>> Original:", layout)
    cute.printf(">?? Original: {}", layout)
    print(">>> Coalesced:", result)
    cute.printf(">?? Coalesced: {}", result)


coalesce_example()
```

**输出：**
```
>>> Original: (2,(1,6)):(1,(?,2))
>>> Coalesced: 12:1
>?? Original: (2,(1,6)):(1,(6,2))
>?? Coalesced: 12:1
```

- 按模式 Coalesce 示例：

```python
@cute.jit
def bymode_coalesce_example():
    """
    演示按模式合并
    """
    layout = cute.make_layout((2, (1, 6)), stride=(1, (6, 2)))

    # 使用模式配置文件 (1,1) 进行 Coalesce = 合并两个模式
    result = cute.coalesce(layout, target_profile=(1, 1))

    # 打印结果
    print(">>> Original: ", layout)
    print(">>> Coalesced Result: ", result)


bymode_coalesce_example()
```

**输出：**
```
>>> Original:  (2,(1,6)):(1,(6,2))
>>> Coalesced Result:  (2,6):(1,2)
```

### 2. Composition（组合）

布局 `A` 与布局 `B` 的 `Composition` 创建一个新布局 `R = A ◦ B`，其中：
- `B` 的形状与 `R` 的形状兼容，以便 `B` 的所有坐标也可以用作 `R` 的坐标
- `R(c) = A(B(c))` 对于 `B` 域中的所有坐标 `c`

布局组合对于重塑和重新排序布局非常有用。

#### 示例

- 基本组合示例：

```python
@cute.jit
def composition_example():
    """
    演示基本布局组合 R = A ◦ B
    """
    A = cute.make_layout((6, 2), stride=(cutlass.Int32(8), 2))  # 动态步长
    B = cute.make_layout((4, 3), stride=(3, 1))
    R = cute.composition(A, B)

    # 打印静态和动态信息
    print(">>> Layout A:", A)
    cute.printf(">?? Layout A: {}", A)
    print(">>> Layout B:", B)
    cute.printf(">?? Layout B: {}", B)
    print(">>> Composition R = A ◦ B:", R)
    cute.printf(">?? Composition R: {}", R)


composition_example()
```

**输出：**
```
>>> Layout A: (6,2):(?,2)
>>> Layout B: (4,3):(3,1)
>>> Composition R = A ◦ B: ((2,2),3):((?{div=3},2),?)
>?? Layout A: (6,2):(8,2)
>?? Layout B: (4,3):(3,1)
>?? Composition R: ((2,2),3):((24,2),8)
```

- 按模式组合示例：

按模式组合允许我们将组合运算应用于布局的各个模式。这在您想要独立操作特定模式布局（例如行和列）时特别有用。

```python
@cute.jit
def bymode_composition_example():
    """
    演示使用 tiler 的按模式组合
    """
    # 定义原始布局 A
    A = cute.make_layout(
        (cutlass.Int32(12), (cutlass.Int32(4), cutlass.Int32(8))),
        stride=(cutlass.Int32(59), (cutlass.Int32(13), cutlass.Int32(1))),
    )

    # 定义用于按模式组合的 tiler
    tiler = (3, 8)  # 将 3:1 应用于模式 0，将 8:1 应用于模式 1

    # 应用按模式组合
    result = cute.composition(A, tiler)

    # 打印静态和动态信息
    print(">>> Layout A:", A)
    cute.printf(">?? Layout A: {}", A)
    print(">>> Tiler:", tiler)
    cute.printf(">?? Tiler: {}", tiler)
    print(">>> By-mode Composition Result:", result)
    cute.printf(">?? By-mode Composition Result: {}", result)


bymode_composition_example()
```

### 3. Division（除法/分割成Tile）

CuTe 中的 Division 操作用于将布局拆分为Tile，这对于跨线程或内存层次结构分区数据特别有用。

#### 示例：

- Logical divide：

当应用于两个 Layouts 时，`logical_divide` 将布局拆分为两个模式——第一个模式包含 tiler 指向的元素，第二个模式包含剩余元素。

```python
@cute.jit
def logical_divide_1d_example():
    """
    演示 1D logical divide
    """
    # 定义原始布局
    layout = cute.make_layout((4, 2, 3), stride=(2, 1, 8))  # (4,2,3):(2,1,8)

    # 定义 tiler
    tiler = cute.make_layout(4, stride=2)  # 应用于布局 4:2

    # 应用 logical divide
    result = cute.logical_divide(layout, tiler=tiler)

    # 打印结果
    print(">>> Layout:", layout)
    print(">>> Tiler :", tiler)
    print(">>> Logical Divide Result:", result)
    cute.printf(">?? Logical Divide Result: {}", result)


logical_divide_1d_example()
```

**输出：**
```
>>> Layout: (4,2,3):(2,1,8)
>>> Tiler : 4:2
>>> Logical Divide Result: ((2,2),(2,3)):((4,1),(2,8))
>?? Logical Divide Result: ((2,2),(2,3)):((4,1),(2,8))
```

- Zipped Divide、Tiled Divide、Flat Divide：

这些是 `logical_divide` 的变体，可能将模式重新排列成更方便的形式。

### 4. Product（乘积/重现Tile）

CuTe 中的 Product 操作用于根据另一个布局重现一个布局。它创建一个新布局，其中：
- 第一个模式是原始布局 A
- 第二个模式是重新调整步长的布局 B，指向 A 的"唯一复制"的原点

这对于跨数据Tile重复线程布局以创建"重复"模式特别有用。

#### 示例

- Logical Product：

```python
@cute.jit
def logical_product_1d_example():
    """
    演示 1D logical product
    """
    # 定义原始布局
    layout = cute.make_layout((2, 2), stride=(4, 1))  # (2,2):(4,1)

    # 定义 tiler
    tiler = cute.make_layout(6, stride=1)  # 应用于布局 6:1

    # 应用 logical product
    result = cute.logical_product(layout, tiler=tiler)

    # 打印结果
    print(">>> Layout:", layout)
    print(">>> Tiler :", tiler)
    print(">>> Logical Product Result:", result)
    cute.printf(">?? Logical Product Result: {}", result)


logical_product_1d_example()
```

**输出：**
```
>>> Layout: (2,2):(4,1)
>>> Tiler : 6:1
>>> Logical Product Result: ((2,2),(2,3)):((4,1),(2,8))
>?? Logical Product Result: ((2,2),(2,3)):((4,1),(2,8))
```

- Blocked 和 Raked Product：
  - Blocked Product：以块状方式组合 A 和 B 的模式，通过在乘积后重新关联来保留模式的语义含义。
  - Raked Product：以交错或"耙状"方式组合 A 和 B 的模式，创建Tile的循环分布。

---

## 7. Element-wise Add Tutorial（逐元素加法教程）

```python
import torch
from functools import partial
from typing import List

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
```

### 内核教程：使用 CuTe DSL 构建高效的逐元素加法内核

本教程演示如何使用 CuTe DSL 实现和优化 GPU 逐元素加法内核。

### 学习目标

在本教程中，您将逐步学习在 CuTe DSL 中构建高效的逐元素内核：
- 如何使用基本 CUDA 技术在 CuTe DSL 中实现基本 GPU 内核
- 如何对内核性能进行基准测试
- 如何Tile 和分区张量并映射到基本 CuTe 布局
- 什么是线程和值布局以及从线程和值索引到逻辑坐标的映射
- 如何使用 TV 布局实现高级内核并调整性能以达到峰值性能

### 理解逐元素加法

逐元素加法是线性代数和深度学习中的基本运算。给定两个相同形状的张量，该操作执行逐元素加法以产生相同形状的结果张量。

对于形状为 $(M, N)$ 的两个 2D 张量 $A$ 和 $B$，逐元素加法运算 $C = A + B$ 定义为：

$C_{i,j} = A_{i,j} + B_{i,j}$

其中：
- $i \in [0, M-1]$ 表示行索引
- $j \in [0, N-1]$ 表示列索引
- $A_{i,j}$, $B_{i,j}$ 和 $C_{i,j}$ 是张量 $A$, $B$ 和 $C$ 中位置 $(i,j)$ 的元素

此操作具有几个重要特征：
1. **可并行化**：每个元素可以独立计算
2. **内存受限**：性能受内存带宽而非计算限制
3. **对合并敏感**：效率取决于内存访问模式
4. **友好于向量化**：多个元素可以一起处理

### 朴素的逐元素加法内核

让我们从一个朴素的实现开始，在探索优化之前建立基线。

```python
# 基本内核实现
# ---------------------
# 这是我们第一个逐元素加法内核的实现。
# 它遵循线程和张量元素之间的简单 1:1 映射。


@cute.kernel
def naive_elementwise_add_kernel(
    gA: cute.Tensor,  # 输入张量 A
    gB: cute.Tensor,  # 输入张量 B
    gC: cute.Tensor,  # 输出张量 C = A + B
):
    # 步骤 1：获取线程索引
    # ------------------------
    # CUDA 线程组织在线程块的 3D 网格中
    # 这里我们只使用 x 维度以简化
    tidx, _, _ = cute.arch.thread_idx()  # 块内的线程索引（0 到 bdim-1）
    bidx, _, _ = cute.arch.block_idx()  # 网格中的块索引（0 到 grid_dim-1）
    bdim, _, _ = cute.arch.block_dim()  # 每个块的线程数

    # 计算全局线程索引
    # 这给每个线程在所有块中一个唯一的 ID
    thread_idx = bidx * bdim + tidx  # 全局线程 ID

    # 步骤 2：将线程索引映射到张量坐标
    # -------------------------------------------
    # 每个线程将处理输入张量的一个元素
    m, n = gA.shape  # 获取张量维度（M 行 × N 列）

    # 将线性线程索引转换为 2D 坐标：
    # - ni：列索引（0 到 n-1）
    # - mi：行索引（0 到 m-1）
    ni = thread_idx % n  # 列索引（更快变化的维度）
    mi = thread_idx // n  # 行索引（较慢变化的维度）

    # 步骤 3：加载和处理数据
    # ---------------------------
    # 从输入张量加载值
    # 张量布局自动处理从逻辑索引 (mi, ni) 到物理内存地址的转换
    a_val = gA[mi, ni]  # 从张量 A 加载元素
    b_val = gB[mi, ni]  # 从张量 B 加载元素

    # 步骤 4：存储结果
    # ------------------
    # 将总和写回输出张量
    gC[mi, ni] = a_val + b_val
```

### 内核结构

朴素内核实现遵循一个直接但有效的 GPU 并行处理结构。以下是其工作原理的详细分解：

1. **线程组织和索引**
   - 每个 CUDA 线程使用以下组合唯一标识：
     * `thread_idx`（tidx）：块内的线程索引（0 到 bdim-1）
     * `block_idx`（bidx）：网格中的块索引
     * `block_dim`（bdim）：每个块的线程数
   - 全局线程 ID 计算为：`thread_idx = bidx * bdim + tidx`

2. **坐标映射**
   - 内核将每个线程的全局 ID 映射到 2D 张量坐标：
     * `ni = thread_idx % n`（列索引 - 更快变化）
     * `mi = thread_idx // n`（行索引 - 较慢变化）
   - 此映射确保通过让相邻线程访问相邻内存位置来实现合并内存访问

3. **内存访问模式**
   - 每个线程：
     * 从张量 A 加载一个元素：`a_val = gA[mi, ni]`
     * 从张量 B 加载一个元素：`b_val = gB[mi, ni]`
     * 执行加法：`a_val + b_val`
     * 将结果存储到张量 C：`gC[mi, ni] = result`
   - 内存考虑
     * 使用 1:1 线程到元素映射
     * 当 warp 中的线程访问连续元素时，内存访问是合并的
     * 不显式使用共享内存或寄存器分块
     * 由于单元素处理，隐藏内存延迟的能力有限

这个朴素的实现为理解后续更优化的版本提供了基线，这些版本引入了：
- 向量化内存访问
- 线程和值（TV）布局
- 高级Tile 策略
- 自定义二元运算

有关合并内存访问的更多详细信息，请阅读：https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#coalesced-access-to-global-memory

### 内核启动配置和测试

本节演示如何：
1. 使用 `cute.jit` 函数配置和启动内核
2. 使用 `torch` 设置测试数据
3. 验证正确性

**启动配置**
- 每个块使用 256 个线程（对于良好占用率的常见选择）
- 根据总元素计算的网格大小：`(m * n) // threads_per_block`
- 单维度块和网格配置以简化

#### 用于启动内核的主机 JIT 函数

```python
@cute.jit  # 即时编译装饰器
def naive_elementwise_add(
    mA: cute.Tensor,  # 输入张量 A
    mB: cute.Tensor,  # 输入张量 B
    mC: cute.Tensor,  # 输出张量 C
):
    # 配置内核启动参数
    # --------------------------------
    # 选择每个块的线程数
    # 256 是一个常见的选择，因为它：
    # - 在大多数 GPU 上允许良好的占用率
    # - 是 32（warp 大小）的倍数
    # - 提供足够的线程来隐藏延迟
    num_threads_per_block = 256

    # 获取输入维度
    m, n = mA.shape  # 矩阵维度（M 行 × N 列）

    # 创建内核实例
    kernel = naive_elementwise_add_kernel(mA, mB, mC)

    # 使用计算的网格维度启动内核
    # -------------------------------------------
    # 网格大小计算：
    # - 总元素：m * n
    # - 需要的块：ceil(total_elements / threads_per_block)
    # - 这里使用整数除法假设 m * n 是 threads_per_block 的倍数
    kernel.launch(
        grid=((m * n) // num_threads_per_block, 1, 1),  # x,y,z 中的块数
        block=(num_threads_per_block, 1, 1),  # x,y,z 中每个块的线程数
    )
```

#### 使用 torch 设置测试数据

```python
# 测试设置
# ----------
# 定义测试维度
M, N = 16384, 8192  # 使用大矩阵来测量性能

# 在 GPU 上创建测试数据
# ----------------------
# 使用 float16（半精度）的原因：
# - 减少内存带宽需求
# - 在现代 GPU 上性能更好
a = torch.randn(M, N, device="cuda", dtype=torch.float16)  # 随机输入 A
b = torch.randn(M, N, device="cuda", dtype=torch.float16)  # 随机输入 B
c = torch.zeros(M, N, device="cuda", dtype=torch.float16)  # 输出缓冲区

# 计算总元素数用于带宽计算
num_elements = sum([a.numel(), b.numel(), c.numel()])

# 将 PyTorch 张量转换为 CuTe 张量
# -------------------------------------
# from_dlpack 创建 PyTorch 张量的 CuTe 张量视图
# assumed_align=16 确保适当的内存对齐以进行向量化访问
a_ = from_dlpack(a, assumed_align=16)  # CuTe 张量 A
b_ = from_dlpack(b, assumed_align=16)  # CuTe 张量 B
c_ = from_dlpack(c, assumed_align=16)  # CuTe 张量 C
```

#### 编译并运行

```python
# 为特定输入类型编译内核
naive_elementwise_add_ = cute.compile(naive_elementwise_add, a_, b_, c_)

# 运行内核
naive_elementwise_add_(a_, b_, c_)

# 验证结果
# -------------
# 将我们的内核输出与 PyTorch 的原生实现进行比较
torch.testing.assert_close(c, a + b)  # 如果结果不匹配则引发错误
```

### 性能分析和基准测试

要理解和改进我们内核的性能，我们需要测量其执行时间和内存吞吐量。让我们分析几个关键指标：

* **执行时间**
   - 以微秒为单位测量原始内核性能
   - 越低越好
   - 受 GPU 时钟速度、内存带宽和内核效率的影响
* **内存吞吐量**
   - 测量我们可以复制数据的速度（GB/s）
   - 越高越好
   - 理论峰值因 GPU 型号而异
   - 对于逐元素加法：
     * 读取：2 个元素（A 和 B）
     * 写入：1 个元素（C）
     * 总字节数 = (2 次读取 + 1 次写入) × 元素数 × sizeof(dtype)

下面是我们的基准测试实用程序，用于测量这些指标：

```python
def benchmark(callable, a_, b_, c_):
    avg_time_us = cute.testing.benchmark(
        callable,
        kernel_arguments=cute.testing.JitArguments(a_, b_, c_),
        warmup_iterations=5,
        iterations=100,
    )

    # 计算指标
    # ----------------
    dtype = a_.element_type

    # 计算传输的总字节数：
    # - 2 次读取（A 和 B）+ 1 次写入（C）
    # - 每个元素是 dtype.width 位
    bytes_per_element = dtype.width // 8
    total_bytes = num_elements * bytes_per_element

    # 计算实现的带宽
    achieved_bandwidth = total_bytes / (avg_time_us * 1000)  # GB/s

    # 打印结果
    # ------------
    print(f"Performance Metrics:")
    print(f"-------------------")
    print(f"Kernel execution time: {avg_time_us:.4f} us")
    print(f"Memory throughput: {achieved_bandwidth:.2f} GB/s")


benchmark(naive_elementwise_add_, a_, b_, c_)
```

### 理论分析

本节通过几个理论框架分析我们逐元素加法内核的性能特征和优化机会。

#### Little's Law

Little's Law 提供了关于延迟、带宽和In-flight操作之间关系的关键见解：

$L = \lambda \times W$

其中：
- $L$：需要的In-flight内存操作数
- $\lambda$：目标内存带宽（字节/周期）
- $W$：内存系统延迟（周期）

根据 *Little's Law*，朴素实现有
   - 每个线程 1 个元素（4 字节加载 + 2 字节存储）
   - 256 线程/块 × N 块
   - In-flight操作有限

在某些 GPU 上，这是不足够的并行性来饱和内存带宽。

#### 优化策略

基于此分析，一种常用技术是**向量化**。不是每个线程每次加载 1 个元素，向量化允许每次加载多个元素
   - 减少指令数量
   - 改善内存合并
   - 增加In-flight操作

### 向量化加载和存储

为了根据 Little's Law 提高性能，我们需要增加In-flight请求的数量。我们可以通过向量化内存访问增加每个线程在每次加载和存储操作中处理的字节数来实现这一点。

由于 Ampere GPU 支持每次加载/存储最多 128 位，并且每个元素是 32 位，我们可以在连续行上每次向量化操作加载 4 个元素。
CuTe Tile 操作使这种向量化变得简单。

使用 ``tiled_tensor = cute.zipped_divide(tensor, tiler)``，我们可以将输入 ``tensor`` 分区为 ``tiler`` 块组。对于向量化，我们将 ``tiler`` 指定为每个线程访问的数据块（同一行中的 4 个连续元素，或 ``(1,4)``）。
然后，不同的线程可以通过索引到 ``tiled_tensor`` 的第二个模式来访问不同的块。

```python
mA : cute.Tensor                           # (2048,2048):(2048,1)
gA = cute.zipped_divide(a, tiler=(1, 4))   # tiled/vectorized => ((1,4),(2048,512)):((0,1),(2048,4))
```

（示例展示了如何使用向量化来提高性能）

### TV 布局

朴素和向量化内核都遵循一个共同的模式，通过两个步骤将线程索引映射到物理地址：

步骤 1：将线程索引映射到 `(M, N)` 中的逻辑坐标

* `mi = thread_idx // n`
* `ni = thread_idx % n`

在朴素版本中，每个线程处理 1 个元素，在这种情况下，`mi` 和 `ni` 是数据张量 `mA`、`mB` 和 `mC` 的逻辑坐标。

在向量化版本中，每个线程处理输入和输出张量的多个值。
逻辑坐标应使用线程和值索引计算。

* `thread_idx // n`
* `(thread_idx % n) * 4 + value_idx`

步骤 2：使用张量布局将 `(M, N)` 中的逻辑坐标映射到物理地址

CuTe 引入 TV 布局来表示从线程索引和值索引（即每个线程加载的 4 个元素）到张量逻辑坐标空间的映射。
通过配置不同的 TV 布局，我们可以用最少的代码更改尝试不同的内存访问模式。

**定义：** *TV Layout* 是 rank-2 布局，它将 `(thread_index, value_index)` 映射到张量的逻辑坐标。

我们总是有规范形式的 *TV Layout*，如 `(thread_domain, value_domain):(..., ...)`。

使用 *TV Layout*，每个线程可以找到分区到当前线程的数据的逻辑坐标或索引。

### 使用 TV 布局的逐元素加法

在本示例中，我们使用两级Tile 重写逐元素内核：
* 线程块级别
* 使用 TV 布局和Tile 的线程级别

对于线程块级别的Tile ，每个输入和输出张量首先在主机端被划分为一组 ``(TileM, TileN)`` 子张量。请注意，在这种情况下，我们仍然使用 `zipped_divide`，但用于线程块级别的Tile 。

在 GPU 内核内部，我们使用第二个模式的线程块索引对Tile 张量进行切片，如 ``gA[((None, None), bidx)]``，这将返回单个 ``(TileM, TileN)`` 子张量的线程块局部视图。此子张量将 ``(TileM, TileN)`` 内的逻辑坐标映射到元素的物理地址。

在线程级别Tile 时，我们将上述子张量（逻辑坐标到物理地址）与 TV 布局（线程和值索引到逻辑坐标）进行组合。这为我们提供了一个Tile 的子张量，该子张量直接从线程和值索引映射到物理地址。

然后我们使用线程索引对其进行切片，如 ``tidfrgA[(tidx, None)]``，以获得每个线程访问的数据的线程局部视图。请注意，线程索引现在在第一个模式中，因为 TV 布局通常具有形式 ``(thread_domain, value_domain):(...,...)``。

#### 内核代码

```python
@cute.kernel
def elementwise_add_kernel(
    gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor, tv_layout: cute.Layout
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # --------------------------------
    # 线程块级别视图切片
    # --------------------------------
    blk_coord = ((None, None), bidx)

    # 逻辑坐标 -> 地址
    blkA = gA[blk_coord]  # (TileM, TileN) -> 物理地址
    blkB = gB[blk_coord]  # (TileM, TileN) -> 物理地址
    blkC = gC[blk_coord]  # (TileM, TileN) -> 物理地址

    # --------------------------------
    # 组合线程索引和值索引到物理映射
    # --------------------------------
    # blockA:    (TileM, TileN) -> 物理地址
    # tv_layout: (tid, vid)     -> (TileM, TileN)
    # tidfrgA = blkA o tv_layout
    # tidfrgA:   (tid, vid) -> 物理地址
    tidfrgA = cute.composition(blkA, tv_layout)
    tidfrgB = cute.composition(blkB, tv_layout)
    tidfrgC = cute.composition(blkC, tv_layout)

    print("与 TV 布局组合后:")
    print(f"  tidfrgA: {tidfrgA.type}")

    # --------------------------------
    # 线程级别视图切片
    # --------------------------------
    # `None` 表示切片整个每线程数据
    thr_coord = (tidx, None)
    # thr_coord = (tidx, cute.repeat_like(None, gA.shape[1]))

    # 为线程切片: vid -> 地址
    thrA = tidfrgA[thr_coord]  # (V) -> 物理地址
    thrB = tidfrgB[thr_coord]  # (V) -> 物理地址
    thrC = tidfrgC[thr_coord]  # (V) -> 物理地址

    thrC[None] = thrA.load() + thrB.load()
```

#### 主机代码

下面的主机代码展示了 TV 布局的构造。通过组合线程布局 ``(4,64):(64,1)``（64 个线程读取行维度上的连续元素，然后 64 线程组（2 个 warp）读取不同的行）与值布局 ``(16,8):(8,1)``（每个线程在行维度上读取 8 个连续的 16b 元素，跨越 4 个连续的行）。

为了通用化，我们从字节布局开始以字节描述元素的布局。这是为了确保使用 128 位向量化加载存储。然后我们利用 ``recast_layout`` 转换为元素布局。

```python
    # 源类型位数: 8
    # 目标类型位数: 元素类型的位数
    val_layout = cute.recast_layout(dtype.width, 8, bit_val_layout)
```

```python
@cute.jit
def elementwise_add(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
):
    # mA 布局: (M, N):(N, 1)
    # TV 布局将线程和值索引映射到 (64, 512) 逻辑 Tile
    #  - 连续的线程索引映射到 mode-1，因为输入布局在 mode-1 上是连续的
    #     以实现合并加载存储
    #  - 每个线程每行加载连续的 16 字节，加载 16 行
    coalesced_ldst_bytes = 16

    # 编译时验证：期望所有输入张量具有相同的元素类型
    assert all(t.element_type == mA.element_type for t in [mA, mB, mC])
    dtype = mA.element_type

    thr_layout = cute.make_ordered_layout((4, 64), order=(1, 0))
    val_layout = cute.make_ordered_layout((16, coalesced_ldst_bytes), order=(1, 0))
    val_layout = cute.recast_layout(dtype.width, 8, val_layout)
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    print(f"[DSL INFO] Tiler: {tiler_mn}")
    print(f"[DSL INFO] TV Layout: {tv_layout}")

    gA = cute.zipped_divide(mA, tiler_mn)  # ((TileM, TileN), (RestM, RestN))
    gB = cute.zipped_divide(mB, tiler_mn)  # ((TileM, TileN), (RestM, RestN))
    gC = cute.zipped_divide(mC, tiler_mn)  # ((TileM, TileN), (RestM, RestN))

    print("Tile 输入张量:")
    print("[DSL INFO] Tile 张量:")
    print(f"[DSL INFO]   gA = {gA.type}")
    print(f"[DSL INFO]   gB = {gB.type}")
    print(f"[DSL INFO]   gC = {gC.type}")

    # 异步启动内核
    # 也可以指定异步令牌作为依赖项
    elementwise_add_kernel(gA, gB, gC, tv_layout).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


a = torch.randn(M, N, device="cuda", dtype=torch.float16)
b = torch.randn(M, N, device="cuda", dtype=torch.float16)
c = torch.zeros(M, N, device="cuda", dtype=torch.float16)

a_ = from_dlpack(a, assumed_align=16)
b_ = from_dlpack(b, assumed_align=16)
c_ = from_dlpack(c, assumed_align=16)

elementwise_add_ = cute.compile(elementwise_add, a_, b_, c_)
elementwise_add_(a_, b_, c_)

# 验证正确性
torch.testing.assert_close(c, a + b)
```

#### 布局解释

让我们更仔细地看看使用 zipped divided 输入张量 `gA` 作为示例。
我们还选择了更小的 M/N，`(256,512)`，以使其更容易解释和可视化。

```
Tile 到线程块:

    ((16,256),(16,2))  : ((512,1),(8192,256))
     ~~~~~~~~  ~~~~~~      ~~~~~
        |        |           |
        |        |           |
        |        `-----------------------> 线程块数量
        |                    |
        |                    |
        `-------------------'
                  |
                  V
             线程块
               Tile

切片到线程块局部子张量（一个 (16, 256) Tile）:  gA[((None, None), bidx)]

    (16,256)   :  (512,1)
     ~~~~~~        ~~~~~~
        |             |        Tile /与 TV 布局组合
        |             |
        |             |    o   ((32,4),(8,4)):((128,4),(16,1))
        V             V
~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~
((32,4),(8,4))  :  ((8,2048),(1,512))
    |      |
    |      `--------> 每线程片段
    |
线程块
  形状

切片到线程局部子张量（一个 (4,8) Tile）:  tidfrgA[(tidx, None)]
```

#### TV 布局可视化

要可视化 TV 布局，我们可以首先安装 *`cute-viz`*

```
pip install -U git+https://github.com/NTT123/cute-viz.git
```

```python
try:
    from cute_viz import display_tv_layout

    @cute.jit
    def visualize():
        # 创建并将布局渲染到文件
        # layout = cute.make_layout( ((16,16),(256,2)), stride=((512,8192),(1,256)))
        # display_layout(layout)

        tv_layout = cute.make_layout(((32, 4), (8, 4)), stride=((128, 4), (16, 1)))
        display_tv_layout(tv_layout, (16, 256))

        thr_block_layout = cute.make_layout((16, 256), stride=(512, 1))
        print(cute.composition(thr_block_layout, tv_layout))

    visualize()
except ImportError:
    pass
```

#### 为什么当张量是行主序时，TV 布局的线程域模式看起来被交换了？

我们可能会注意到上面示例中的 *TV 布局* 是 `((32,4),(8,4)):((128,4),(16,1))`。
然而，在可视化中，线程索引被排列为形状 `(4,32)` 而不是 *TV 布局* 的 `(32,4)`。

这是内部团队和社区的开发人员经常问的问题。

重要的是要记住，*TV 布局* 将 `(thread_index, value_index)` 映射到逻辑域 `(TileM, TileN)` 的 `(row_index, column_index)`。但是，可视化显示了逻辑域 `(TileM, TileN)` 到 `(thread_domain, value_domain)` 的**逆**映射，因为这对人类开发者来说更直观。

这就是为什么 *TV 布局* 的域形状不一定与逻辑视图匹配。

```python
benchmark(elementwise_add_, a_, b_, c_)
```

#### 重映射/转置线程块索引

由于本示例中的张量是行主序，我们可能希望线程块尽可能多地加载连续内存。

我们可以应用简单的线程块重映射来转置行优先顺序中线程块索引的映射。
`cute.composition(gA, (None, remap_block))` 仅应用Tile 布局第二个模式的转置，但保持第一个模式不变。

```python
    remap_block = cute.make_ordered_layout(
        cute.select(gA.shape[1], mode=[1, 0]), order=(1, 0)
    )
    gA = cute.composition(gA, (None, remap_block))
    gB = cute.composition(gB, (None, remap_block))
    gC = cute.composition(gC, (None, remap_block))
```

```python
@cute.jit
def elementwise_add(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
):
    # mA 布局: (M, N):(N, 1)
    # TV 布局将线程和值索引映射到 (64, 512) 逻辑 Tile
    #  - 连续的线程索引映射到 mode-1，因为输入布局在 mode-1 上是连续的
    #     以实现合并加载存储
    #  - 每个线程每行加载连续的 16 字节，加载 16 行
    coalesced_ldst_bytes = 16

    # 编译时验证：期望所有输入张量具有相同的元素类型
    assert all(t.element_type == mA.element_type for t in [mA, mB, mC])
    dtype = mA.element_type

    thr_layout = cute.make_ordered_layout((4, 64), order=(1, 0))
    val_layout = cute.make_ordered_layout((16, coalesced_ldst_bytes), order=(1, 0))
    val_layout = cute.recast_layout(dtype.width, 8, val_layout)
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    print(f"[DSL INFO] Tiler: {tiler_mn}")
    print(f"[DSL INFO] TV Layout: {tv_layout}")

    gA = cute.zipped_divide(mA, tiler_mn)  # ((TileM, TileN), (RestM, RestN))
    gB = cute.zipped_divide(mB, tiler_mn)  # ((TileM, TileN), (RestM, RestN))
    gC = cute.zipped_divide(mC, tiler_mn)  # ((TileM, TileN), (RestM, RestN))

    # (RestM, RestN) -> (RestN, RestM)
    remap_block = cute.make_ordered_layout(
        cute.select(gA.shape[1], mode=[1, 0]), order=(1, 0)
    )
    gA = cute.composition(gA, (None, remap_block))
    gB = cute.composition(gB, (None, remap_block))
    gC = cute.composition(gC, (None, remap_block))

    print("Tile 输入张量:")
    print("[DSL INFO] Tile 张量:")
    print(f"[DSL INFO]   gA = {gA.type}")
    print(f"[DSL INFO]   gB = {gB.type}")
    print(f"[DSL INFO]   gC = {gC.type}")

    # 异步启动内核
    # 也可以指定异步令牌作为依赖项
    elementwise_add_kernel(gA, gB, gC, tv_layout).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


a = torch.randn(M, N, device="cuda", dtype=torch.float16)
b = torch.randn(M, N, device="cuda", dtype=torch.float16)
c = torch.zeros(M, N, device="cuda", dtype=torch.float16)

a_ = from_dlpack(a, assumed_align=16)
b_ = from_dlpack(b, assumed_align=16)
c_ = from_dlpack(c, assumed_align=16)

elementwise_add_ = cute.compile(elementwise_add, a_, b_, c_)
elementwise_add_(a_, b_, c_)

# 验证正确性
torch.testing.assert_close(c, a + b)
```

```python
benchmark(elementwise_add_, a_, b_, c_)
```

### 使用 Lambda 函数

CuTe DSL 建立在 Python 之上。它可以利用 Python 实现元编程以生成灵活的内核。
例如，我们可以编写接受自定义二元运算的内核模板，以为任意二元运算生成内核。

```python
@cute.jit
def elementwise_apply(
    op: cutlass.Constexpr,
    inputs,
    result: cute.Tensor
):
    ...
```

```python
@cute.kernel
def elementwise_apply_kernel(
    op: cutlass.Constexpr,
    mInputs: List[cute.Tensor],
    mC: cute.Tensor,
    cC: cute.Tensor,  # 坐标张量
    shape: cute.Shape,
    tv_layout: cute.Layout,  # (tid, vid) -> 逻辑坐标
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    ###############################################################################
    # 切片到线程块的局部 Tile
    ###############################################################################
    blk_crd = ((None, None), bidx)

    # 利用 DSL 的元编程能力为每个输入切片张量
    # 下面所有输入张量上的 for 循环将在编译时自动完全展开
    # 逻辑坐标 -> 内存地址
    gInputs = [t[blk_crd] for t in mInputs]  # (TileM, TileN)
    gC = mC[blk_crd]  # (TileM, TileN)
    gCrd = cC[blk_crd]  # (TileM, TileN)

    print("[DSL INFO] 每个线程块的切片张量:")
    for i in cutlass.range_constexpr(len(gInputs)):
        print(f"[DSL INFO]   ctaInputs{i} = {gInputs[i].type}")
    print(f"[DSL INFO]   gC = {gC.type}")
    print(f"[DSL INFO]   gCrd = {gCrd.type}")

    ###############################################################################
    # 与线程块 TV 布局组合以将线程和值索引映射到内存地址
    ###############################################################################
    # (tid, vid) -> 内存地址
    tidfrgInputs = [cute.composition(t, tv_layout) for t in gInputs]
    tidfrgC = cute.composition(gC, tv_layout)
    tidfrgCrd = cute.composition(gCrd, tv_layout)

    # 重复 None 类似 vid 以移除布局的层次结构
    thr_crd = (tidx, cute.repeat_like(None, tidfrgInputs[0][1]))

    ###############################################################################
    # 切片到线程的局部 Tile
    ###############################################################################
    # vid -> 地址
    thrInputs = [t[thr_crd] for t in tidfrgInputs]  # (V)
    thrC = tidfrgC[thr_crd]  # (V)
    thrCrd = tidfrgCrd[thr_crd]

    print("[DSL INFO] 每个线程的切片张量:")
    for i in cutlass.range_constexpr(len(thrInputs)):
        print(f"[DSL INFO]   thrInputs{i} = {thrInputs[i].type}")
    print(f"[DSL INFO]   thrC = {thrC.type}")
    print(f"[DSL INFO]   thrCrd = {thrCrd.type}")

    ###############################################################################
    # 计算边界检查的谓词
    ###############################################################################
    frgPred = cute.make_fragment(thrCrd.shape, cutlass.Boolean)
    print(f"[DSL INFO]   frgPred = {frgPred.type}")

    for i in cutlass.range_constexpr(cute.size(frgPred)):
        frgPred[i] = cute.elem_less(thrCrd[i], shape)

    # if tidx == 0 and bidx == 0:
    #     cute.print_tensor(frgPred)

    ##########################################################
    # 加载数据并计算结果
    ##########################################################

    # 在使用前加载数据。编译器将优化复制和加载操作
    # 将一些内存 ld/st 转换为寄存器使用。
    result = op(*[thrInput.load() for thrInput in thrInputs])
    thrC.store(result)


@cute.jit
def elementwise_apply(op: cutlass.Constexpr, inputs, result: cute.Tensor):
    # 使用 128bit(16B) 加载作为 val_layout 的规范化形式，然后重铸为目标元素类型
    coalesced_ldst_bytes = 16

    # 编译时验证：期望所有输入张量具有相同的元素类型
    assert all(t.element_type == inputs[0].element_type for t in inputs)
    dtype = inputs[0].element_type

    thr_layout = cute.make_ordered_layout((4, 64), order=(1, 0))
    val_layout = cute.make_ordered_layout((16, coalesced_ldst_bytes), order=(1, 0))
    val_layout = cute.recast_layout(dtype.width, 8, val_layout)
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    mInputs = [cute.zipped_divide(input, tiler_mn) for input in inputs]
    mC = cute.zipped_divide(result, tiler_mn)  # ((TileM, TileN), (RestM, RestN))

    # (RestM, RestN) -> (RestN, RestM)
    remap_block = cute.make_ordered_layout(
        cute.select(mInputs[0].shape[1], mode=[1, 0]), order=(1, 0)
    )
    for i, t in enumerate(mInputs):
        mInputs[i] = cute.composition(t, (None, remap_block))

    mC = cute.composition(mC, (None, remap_block))

    idC = cute.make_identity_tensor(result.shape)
    cC = cute.zipped_divide(idC, tiler=tiler_mn)

    # 异步启动内核
    # 将输入张量分组到一个列表中作为单个参数
    elementwise_apply_kernel(op, mInputs, mC, cC, result.shape, tv_layout).launch(
        grid=[cute.size(mC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


a = torch.randn(M, N, device="cuda", dtype=torch.float16)
b = torch.randn(M, N, device="cuda", dtype=torch.float16)
c = torch.zeros(M, N, device="cuda", dtype=torch.float16)

a_ = from_dlpack(a, assumed_align=16)
b_ = from_dlpack(b, assumed_align=16)
c_ = from_dlpack(c, assumed_align=16)
```

```python
from operator import mul

elementwise_apply(mul, [a_, b_], c_)

# 验证正确性
torch.testing.assert_close(c, mul(a, b))
```

#### 使用自定义函数

自定义运算符可以更复杂。例如，这是一个执行乘法后跟 ReLU 的函数：

```python
def mul_relu(a, b):
    tmp = a * b
    return cute.where(tmp > 0, tmp, cute.full_like(tmp, 0))


# 由于我们在自定义运算中使用 cute.where，我们需要创建另一个 relu 函数
def mul_relu_ref(a, b):
    tmp = a * b
    return torch.relu(tmp)


elementwise_apply(mul_relu, [a_, b_], c_)

# 验证正确性
torch.testing.assert_close(c, mul_relu_ref(a, b))
```

---

## 8. Using CUDA Graphs（使用 CUDA Graphs）

在本示例中，我们演示如何通过 PyTorch 使用 CUDA graphs 与 CuTe DSL。
与 PyTorch 的 CUDA graph 实现交互的过程需要将 PyTorch 的 CUDA 流暴露给 CUTLASS。

要在 Blackwell 上使用 CUDA graphs 需要支持 Blackwell 的 PyTorch 版本。
这可以通过以下方式获得：
- [PyTorch NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
- [PyTorch 2.7 with CUDA 12.8 或更高版本](https://pytorch.org/)（例如，`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`）
- 直接使用您的 CUDA 版本构建 PyTorch。

```python
# 导入 torch 用于 CUDA graphs
import torch
import cutlass.cute as cute

# 从 cuda 驱动程序绑定导入 CUstream 类型
from cuda.bindings.driver import CUstream

# 从 torch 导入 current_stream 函数
from torch.cuda import current_stream
```

### 内核创建

我们创建一个打印"Hello world"的内核以及一个用于启动内核的主机函数。
然后我们通过传入默认流来编译内核以在我们的 graph 中使用。

在 graph 捕获之前进行内核编译是必需的，因为 CUDA graphs 在 graph 执行期间无法进行 JIT 编译内核。

```python
@cute.kernel
def hello_world_kernel():
    """
    一个打印 hello world 的内核
    """
    cute.printf("Hello world")


@cute.jit
def hello_world(stream: CUstream):
    """
    在流中启动我们的 (1,1,1), (1,1,1) 网格的主机函数
    """
    hello_world_kernel().launch(grid=[1, 1, 1], block=[1, 1, 1], stream=stream)


# 从 PyTorch 获取流，这也会初始化我们的上下文
# 所以我们可以省略 cutlass.cuda.initialize_cuda_context()
stream = current_stream()
hello_world_compiled = cute.compile(hello_world, CUstream(stream.cuda_stream))
```

### 创建和重放 CUDA Graph

我们通过 torch 创建一个流以及一个 graph。
当我们创建 graph 时，我们可以将想要捕获的流传递给 torch。我们同样使用流作为 CUstream 运行编译的内核。

最后，我们可以重放我们的 graph 并同步。

```python
# 创建 CUDA Graph
g = torch.cuda.CUDAGraph()
# 捕获我们的 graph
with torch.cuda.graph(g):
    # 将我们的 torch Stream 转换为 cuStream 流。
    # 这是通过使用 .cuda_stream 获取底层 CUstream 来完成的
    graph_stream = CUstream(current_stream().cuda_stream)
    # 运行 2 次我们编译的内核
    for _ in range(2):
        # 在流中运行我们的内核
        hello_world_compiled(graph_stream)

# 重放我们的 graph
g.replay()
# 同步所有流（等效于 C++ 中的 cudaDeviceSynchronize()）
torch.cuda.synchronize()
```

**输出：**
```
Hello world
Hello world
```

我们的运行在 NSight Systems 中查看时会导致以下执行：

![两个 hello world 内核在 CUDA graph 中连续运行的图像](images/cuda_graphs_image.png)

我们可以观察到两个内核的启动，然后是 `cudaDeviceSynchronize()`。

现在我们可以确认这最小化了一些启动开销：

```python
# 从 PyTorch 获取我们的 CUDA 流
stream = CUstream(current_stream().cuda_stream)

# 创建一个包含 100 次迭代的更大 CUDA Graph
g = torch.cuda.CUDAGraph()
# 捕获我们的 graph
with torch.cuda.graph(g):
    # 将我们的 torch Stream 转换为 cuStream 流。
    # 这是通过使用 .cuda_stream 获取底层 CUstream 来完成的
    graph_stream = CUstream(current_stream().cuda_stream)
    # 运行 2 次我们编译的内核
    for _ in range(100):
        # 在流中运行我们的内核
        hello_world_compiled(graph_stream)

# 创建 CUDA 事件以测量性能
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# 运行我们的内核以预热 GPU
for _ in range(100):
    hello_world_compiled(stream)

# 记录我们的开始时间
start.record()
# 运行 100 个内核
for _ in range(100):
    hello_world_compiled(stream)
# 记录我们的结束时间
end.record()
# 同步（cudaDeviceSynchronize()）
torch.cuda.synchronize()

# 计算在流中启动内核时花费的时间
# 结果以 ms 为单位
stream_time = start.elapsed_time(end)

# 再次预热我们的 GPU
g.replay()
# 记录我们的开始时间
start.record()
# 运行我们的 graph
g.replay()
# 记录我们的结束时间
end.record()
# 同步（cudaDeviceSynchronize()）
torch.cuda.synchronize()

# 计算在 graph 中启动内核时花费的时间
# 单位是 ms
graph_time = start.elapsed_time(end)
```

```python
# 打印使用 CUDA graphs 时的加速
percent_speedup = (stream_time - graph_time) / graph_time
print(f"{percent_speedup * 100.0:.2f}% speedup when using CUDA graphs for this kernel!")
```

**输出：**
```
8.94% speedup when using CUDA graphs for this kernel!
```

---

## 总结

本文档翻译和整理了 CUTLASS 4.3.5 CuTe DSL 的 Educational Notebooks，涵盖了从基础的 Hello World 到高级的逐元素加法优化和 CUDA Graphs 使用的完整内容。通过这些示例，您可以逐步掌握 CuTe DSL 的核心概念和高级特性。
