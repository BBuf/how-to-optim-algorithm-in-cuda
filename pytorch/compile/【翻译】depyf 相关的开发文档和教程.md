> 地址：https://depyf.readthedocs.io/en/latest/

# depyf 总览

在学习 ``depyf`` 的使用之前，我们建议阅读 《【翻译】torch.compile 的详细示例解析教程》，以便您了解 ``depyf`` 如何帮助您。

``depyf`` 旨在解决 ``torch.compile`` 的两个痛点：

- ``torch.compile`` 转换 Python 字节码，但很少有开发者能够阅读 Python 字节码（除非你的大脑里有一个栈机...）来理解发生了什么。``depyf`` 帮助将转换后的字节码反编译回 Python 源代码，以便开发者能够理解 ``torch.compile`` 如何转换他们的代码。这大大帮助用户适应他们的代码到 ``torch.compile``，以便他们可以编写对 ``torch.compile`` 友好的代码。
- ``torch.compile`` 动态生成许多函数，这些函数只能作为一个黑盒运行。用户无法逐行调试代码。``depyf`` 帮助将源代码转储到文件中，并将这些函数与源代码文件链接，以便用户可以使用调试器逐行调试这些函数。这大大帮助用户理解 ``torch.compile`` 并在训练过程中调试诸如 ``NaN`` 等问题。

从《torch.compile 的详细示例解析教程》中获取工作流程：

![](https://files.mdnice.com/user/59/f8aa1ff8-ef72-4ac3-ad41-aed2dc4b4eac.png)

> 图片解释：

- **源代码 (Source Code)**： 图中展示了一个简单的 Python 函数 `function(inputs)`，该函数对输入的两个张量 `x` 和 `y` 进行一些数学运算（例如 `cos` 和 `mean`）后返回 `x * y`。
- **输入 (Input)**： 显示了该函数的输入 inputs 包含 `x` 和 `y` 两个 `torch.Tensor` 类型的张量。
- **守护函数 (Guard)**： 守护函数基于输入和 Python 字节码生成，确保编译器在运行时验证张量 `x` 和 `y` 的形状和类型，以决定是否需要重新编译或可以直接执行。
- **Python 字节码分析 (Python Bytecode Analysis)**： 图中显示了 Python 源代码经过编译器分析后生成的字节码。这里将 Python 操作（与张量计算无关的部分）和纯张量计算的 PyTorch 操作（如 cos、mean 等）分开列出。
- **转换后的字节码 (Transformed Bytecode)**： 在 PyTorch 编译器将源码转换成字节码后，会根据不同操作生成相应的字节码指令，同时根据守护条件来决定执行流程。比如图中展示了 `_compiled_fn` 的实现和条件跳转操作。
- **恢复函数 (Resume Functions)**： 当某些操作需要的条件不满足（比如某个张量值需要重新计算）时，编译器会触发恢复函数（resume function），该恢复函数可以继续进行需要的计算，并且递归地触发字节码分析。
- **执行流程 (Execution Workflow)**： 最后，图中展示了如何从原始的字节码指令通过守护条件、张量计算图、恢复函数等一系列操作形成最终的执行字节码流程。

> 注意：字节码后标注了对应的原始的python代码

``depyf`` 帮助：

- 提供上述工作流程的源代码描述，以便用户可以轻松理解。（实际工作流程发生在 C 语言和 CPython 解释器内部，我们提供工作流程的 Python 源代码描述，以便用户可以轻松理解。）
- 为转换后的字节码和Resume Functions生成源代码。
- 将图计算函数与磁盘上的代码链接，以便调试器可以逐行执行代码。

``depyf`` 的主要用法涉及两个上下文管理器，建议使用调试器启动脚本：

```python

import torch

@torch.compile
def function(inputs):
    x = inputs["x"]
    y = inputs["y"]
    x = x.cos().cos()
    if x.mean() > 0.5:
        x = x / 1.1
    return x * y

shape_10_inputs = {"x": torch.randn(10, requires_grad=True), "y": torch.randn(10, requires_grad=True)}
shape_8_inputs = {"x": torch.randn(8, requires_grad=True), "y": torch.randn(8, requires_grad=True)}

import depyf
with depyf.prepare_debug("./debug_dir"):
    # warmup
    for i in range(100):
        output = function(shape_10_inputs)
        output = function(shape_8_inputs)
# the program will pause here for you to set breakpoints
# then you can hit breakpoints when running the function
with depyf.debug():
    output = function(shape_10_inputs)
```

第一个上下文管理器 ``depyf.prepare_debug()`` 接受一个目录路径来转储所有源代码。在这个上下文管理器内部，PyTorch 的所有内部细节将被 ``depyf`` 钩住，它会为你转储必要的源代码。

第二个上下文管理器 ``depyf.debug()`` 没有参数，它只是禁用新的编译条目。进入这个上下文管理器时，程序将暂停，你可以在你指定的目录下浏览所有源代码（在这个例子中是 ``"./debug_dir"``）。入口文件是 ``full_code_for_xxx.py``。你可以在这些文件中设置断点。最重要的是，你设置的那些断点，在这个上下文管理器下可以被命中。你可以逐行浏览代码，调试可能的 ``NaN`` 值或理解代码发生了什么。

下图展示了 ``depyf`` 的两种典型用法，并列出了所有生成的文件。

![](https://files.mdnice.com/user/59/d1caaaa5-282e-4f83-9edf-5f80472cec72.png)


## API参考

### 理解和调试 ``torch.compile``

> 使用下面的函数之前建议阅读 《【翻译】torch.compile 的详细示例解析教程》 对`torch.compile`有一个基本的理解。

#### `depyf.prepare_debug`

用于转储 `torch.compile` 调试信息的一个上下文管理器。
它应该包裹实际触发编译的代码，而不是应用 ``torch.compile`` 的代码。

示例：

```python
import torch

@torch.compile
def toy_example(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b

def main():
    for _ in range(100):
        toy_example(torch.randn(10), torch.randn(10))

if __name__ == "__main__":
    # main()
    # surround the code you want to run inside `with depyf.prepare_debug`
    import depyf
    with depyf.prepare_debug("./dump_src_dir"):
        main()
```

运行代码后，你将在目录 ``dump_src_dir`` 中找到转储的信息。详细信息组织如下：

- ``full_code_for_xxx.py`` 用于每个使用 torch.compile 的函数
- ``__transformed_code_for_xxx.py`` 用于与每个图相关的 Python 代码
- ``__transformed_code_for_xxx.py.xxx_bytecode`` 用于 Python 字节码，转储的代码对象，可以通过 ``dill.load(open("/path/to/file", "wb"))`` 加载。注意，加载函数可能导入一些模块，如 transformers。请确保你已安装这些模块。
- ``__compiled_fn_xxx.py`` 用于每个计算图及其优化：
    - ``Captured Graph``：一个简单的正向计算图
    - ``Joint Graph``：来自 AOTAutograd 的联合正向-反向图
    - ``Forward Graph``：来自 AOTAutograd 的正向图
    - ``Backward Graph``：来自 AOTAutograd 的反向图
    - ``kernel xxx``：来自 Inductor 的编译 CPU/GPU kernel包装器

参数：

- ``dump_src_dir``：转储源代码的目录
- ``clean_wild_fx_code``：是否清理未识别的编译函数部分中的 wild fx 代码。它们通常由 PyTorch 内部使用。
- ``log_bytecode``：是否记录字节码（原始字节码、来自 Dynamo 的转换字节码和反编译并重新编译的字节码）

#### `depyf.debug`

用于调试编译代码的上下文管理器。本质上，它设置了一个断点来暂停程序，并允许您在 `depyf.prepare_debug()` 的 `dump_src_dir` 参数中以 `full_code_for_` 为前缀的文件中检查完整的源代码，并根据函数名称在其单独的 `__transformed_code_` 文件中设置断点。然后继续您的调试。

### 反编译通用 Python 字节码/函数

#### `depyf.decompile`

将任何可调用对象或代码对象反编译为 Python 源代码。
这对于一些动态生成的代码特别有用，例如 ``torch.compile`` 或 ``dataclasses``。

示例用法：

```python
from dataclasses import dataclass
@dataclass
class Data:
    x: int
    y: float

import depyf
print(depyf.decompile(Data.__init__))
print(depyf.decompile(Data.__eq__))
```

输出:

```python
def __init__(self, x, y):
    self.x = x
    self.y = y
    return None

def __eq__(self, other):
    if other.__class__ is self.__class__:
        return (self.x, self.y) == (other.x, other.y)
    return NotImplemented
```

输出的源代码在语义上等同于函数，但在语法上并不相同。它详细地添加了许多在Python代码中隐藏的细节。例如，上面``__init__``的输出代码显式地返回了``None``，这在通常情况下是被忽略的。

另一个细节是，当类型不同时，``__eq__``的输出代码返回了``NotImplemented``而不是引发``NotImplemented``异常。乍一看，这似乎是一个错误。然而，这实际上是正确的行为。当类型不同时，``__eq__``方法应该返回``NotImplemented``，以便另一个对象可以尝试与当前对象进行比较。更多详情请参见`Python文档 <https://docs.python.org/3/library/numbers.html#implementing-the-arithmetic-operations>`_。

### 增强 PyTorch 日志记录

#### `depyf.install`

安装 PyTorch 的字节码钩子，集成到 PyTorch 的日志系统中。

示例：

```python
import torch
import depyf
depyf.install()
# anything with torch.compile
@torch.compile
def f(a, b):
    return a + b
f(torch.tensor(1), torch.tensor(2))
```

通过 `export TORCH_LOGS="+bytecode"` 开启字节码日志，并执行脚本。我们将在日志中看到反编译的源代码：

```shell
ORIGINAL BYTECODE f test.py line 5 
7           0 LOAD_FAST                0 (a)
            2 LOAD_FAST                1 (b)
            4 BINARY_ADD
            6 RETURN_VALUE


MODIFIED BYTECODE f test.py line 5 
5           0 LOAD_GLOBAL              0 (__compiled_fn_1)
            2 LOAD_FAST                0 (a)
            4 LOAD_FAST                1 (b)
            6 CALL_FUNCTION            2
            8 UNPACK_SEQUENCE          1
            10 RETURN_VALUE


possible source code:
def f(a, b):
    __temp_2, = __compiled_fn_1(a, b)
    return __temp_2

If you find the decompiled code is wrong,please submit an issue at https://github.com/thuml/depyf/issues.
```
卸载这个钩子使用 `depyf.uninstall()`.

#### `depyf.uninstall`

卸载 PyTorch 的字节码钩子。应在 `depyf.install()` 之后调用。

## 优化教程

在本教程中，我们将演示如何使用 ``torch.compile`` 优化代码，并借助 ``depyf`` 库的帮助。

### 示例代码

让我们从一个简单的示例开始，这是我们想要优化的代码：

```python
import torch

class F(torch.nn.Module):
    def __init__(self, i):
        super().__init__()
        self.i = i

    def forward(self, x):
        return x + self.i

class Mod(torch.nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.fs = torch.nn.ModuleList([F(i) for i in range(n)])

    @torch.compile
    def forward(self, x):
        for f in self.fs:
            x = f(x)
        return x

total_time = 0
import time

mod = Mod(100)
mod(torch.tensor([1]))  # Compile the function

x = torch.tensor([2])  # Create input tensor
start = time.time()
for i in range(10000):
    y = mod(x)
    # do something with y
end = time.time()
total_time += end - start
print(total_time)
```

这个示例有意简化了计算过程，以便更容易理解。在实际场景中，函数会执行更复杂的操作。在MacOS机器上，运行编译后的函数10,000次大约需要0.7秒。我们的目标是优化代码以减少执行时间。

### 使用 Depyf 理解代码

为了优化代码，我们首先需要理解执行过程中发生了什么。``depyf`` 库可以反编译字节码并提供洞察。我们可以在之前的代码中添加两行：

```python
# Insert these lines before ``mod(torch.tensor([1]))``
import depyf
with depyf.prepare_debug("dump_src_dir/"):
    mod(torch.tensor([1]))  # Compile the function
# Remaining code as above
```

运行代码后，会出现一个名为 ``dump_src_dir/`` 的新目录。该目录包含反编译后的源代码。例如，在文件 ``full_code_for_forward_0.py`` 中，您将找到：

```python
def __guard_0_for_forward(L, G, **___kwargs_ignored):
    __guard_hit = True
    __guard_hit = __guard_hit and utils_device.CURRENT_DEVICE == None
    __guard_hit = __guard_hit and ___check_global_state()
    __guard_hit = __guard_hit and check_tensor(L['x'], Tensor, DispatchKeySet(CPU, ...), ...)
    ...
    __guard_hit = __guard_hit and len(L['self'].fs) == 100
    __guard_hit = __guard_hit and L['self'].fs[0].i == 0
    __guard_hit = __guard_hit and L['self'].fs[1].i == 1
    __guard_hit = __guard_hit and L['self'].fs[2].i == 2
    ...
    return __guard_hit
```

这是 ``torch.compile`` 生成的代码，用于检查函数的输入是否仍然对编译后的函数有效。然而，这些检查中有许多过于保守。例如，``L['self'].fs[0].i == 0`` 检查 ``self.fs[0].i`` 是否仍然是 ``0``，即使我们的代码没有修改这个值。

> ``torch.compile`` 是一个即时编译器。这意味着每次调用函数时都会执行上述所有检查，从而引入显著的开销。

### 优化代码

由于 ``torch.compile`` 每次调用函数时都会执行这些检查，因此引入了开销。为了优化代码，我们可以绕过这些检查。一种方法是修改 ``__guard_0_for_forward`` 函数以立即返回 ``True``，但 ``torch.compile`` 没有提供直接的机制来实现这一点。

相反，我们可以使用 ``depyf`` 直接调用编译后的函数而无需检查。以下代码演示了这种方法：

```python

import torch
import depyf
from depyf.optimization import TorchCompileWrapperWithCustomDispatcher

class F(torch.nn.Module):
    def __init__(self, i):
        super().__init__()
        self.i = i

    def forward(self, x):
        return x + self.i

class Mod(TorchCompileWrapperWithCustomDispatcher):
    def __init__(self, n: int):
        self.fs = torch.nn.ModuleList([F(i) for i in range(n)])
        compiled_callable = torch.compile(self.forward)
        super().__init__(compiled_callable)

    def forward(self, x):
        for f in self.fs:
            x = f(x)
        return x

    def __call__(self, x):
        if len(self.compiled_codes) == 1:
            with self.dispatch_to_code(0):
                return self.forward(x)
        else:
            return self.compiled_callable(x)

total_time = 0
import time

mod = Mod(100)
mod(torch.tensor([1]))  # Compile

x = torch.tensor([2])  # Input tensor
start = time.time()
for i in range(10000):
    y = mod(x)
end = time.time()
total_time += end - start
print(total_time)
```

在这段代码中，使用了 ``TorchCompileWrapperWithCustomDispatcher`` 类来绕过检查。通过这样做，执行时间从原来的 0.7 秒下降到大约 0.05 秒。这表明检查是大部分开销的来源。

### 工作原理

``TorchCompileWrapperWithCustomDispatcher`` 劫持了 ``torch.compile`` 生成的字节码，并直接调用编译后的函数，跳过了守卫检查。``__call__`` 方法检查是否已经存在编译版本，如果存在，则直接分派到编译后的代码。

### 实际应用

这是一个极端的例子，计算非常简单，但Dynamo引入的开销却不成比例地大。在实际应用中，开销通常不会这么严重。然而，在高性能环境中，例如在TPU上运行代码时，这种开销仍然可能非常显著。TPU代码通常对性能非常敏感，移除不必要的检查可以带来显著的加速。

例如，在 `vLLM的TPU集成 <https://github.com/vllm-project/vllm/pull/7898>`_ 中，这种优化技术被用来移除Dynamo的开销，从而将TPU的吞吐量提高了4%。

### 结论

使用 ``torch.compile`` 优化代码涉及以下步骤：

1. 使用 ``depyf`` 反编译字节码并理解性能瓶颈。
2. 识别并解决不必要的检查或其他开销来源。
3. 使用 ``depyf`` 直接调用编译后的函数，在适当的地方绕过不必要的步骤。

通过遵循这些步骤，我们可以显著提高性能，特别是在执行时间至关重要的环境中。

## 开发者文档

对于开发者，如果你想了解和贡献代码库，本节适合你。

### 库的总体架构

![](https://files.mdnice.com/user/59/f9a4bbf8-39d8-445a-86e0-1f578bbf0319.png)

上图展示了库的整体架构。

1. 通常，当我们执行Python代码时，代码会被编译成Python字节码，然后由Python解释器执行。
2. 当使用``torch.compile``时，PyTorch会将函数编译成一个新的字节码对象来执行。它通过向Python解释器注册一个帧评估函数来实现这一点。每当函数被执行时，帧评估函数都会被调用。PyTorch通过``torch._C._dynamo.eval_frame.set_eval_frame``函数包装了帧评估回调注册。因为PyTorch直接生成字节码，所以它没有源代码信息。字节码直接由Python解释器执行。
3. 当``depyf``与PyTorch一起使用时，它将通过``torch._dynamo.convert_frame.register_bytecode_hook``向PyTorch注册一个字节码钩子（我们与PyTorch团队合作设计了这个字节码钩子机制）。每当PyTorch编译一个函数时，钩子都会被调用。钩子会将字节码反编译成源代码并将其转储到磁盘。然后，源代码被编译成一个新的字节码对象，该对象在功能上等同于PyTorch生成的字节码，但包含源代码信息。PyTorch将使用这个新的字节码对象来执行函数。与``depyf``相关的部分标记为绿色。

由此可以看出，该库是一个与PyTorch紧密集成的Python字节码反编译器。它自然地分为两个部分：

- 反编译器部分实现在 ``depyf/decompiler.py`` 文件中。它也可以作为一个独立的库来反编译Python字节码。
- PyTorch集成部分实现在 ``depyf/explain/enable_debugging.py`` 文件中。它还包含大量代码来处理 PyTorch 编译器的其他部分，例如图编译和转换、代码守护和缓存等。

相对而言，PyTorch集成部分更容易理解和贡献。我们的主要目标是使 ``depyf`` 兼容从PyTorch 2.2开始的所有先前版本的PyTorch。为了实现这一目标，测试针对PyTorch的nightly构建版本运行。每当我们发现兼容性问题时，我们都会以向后兼容的方式修复它。如果无法进行这样的修复，我们将与PyTorch团队讨论以找到解决方案。

反编译器部分更具挑战性。它很复杂，需要处理各种随机的Python实现细节。幸运的是，我们只需要处理官方发布的Python版本，这使得任务更加可管理。反编译器只需要在我们发现错误或发布新版本的Python时进行更新。

如果你想深入研究反编译器部分，请继续阅读。

### 反编译器概述

要熟悉阅读字节码，建议首先阅读以下材料：

- `torchdynamo deepdive <https://www.youtube.com/watch?v=egZB5Uxki0I>`_：这个视频解释了torchdynamo的动机和设计。特别是，它提到了Python字节码如何像堆栈机一样工作，这有助于理解字节码的执行方式。
- `Python字节码文档 <https://docs.python.org/3/library/dis.html>`_：这份文档解释了Python字节码指令。请注意，Python字节码不保证任何向后兼容性，因此字节码指令可能会随着每个Python版本而变化。在实现反编译器时，我们应该考虑所有支持的Python版本。
- `用Python编写的Python解释器 <https://aosabook.org/en/500L/a-python-interpreter-written-in-python.html>`_：本书章节解释了如何用Python编写Python解释器。它是理解Python字节码如何执行的一个很好的起点。

反编译过程是通过执行Python字节码并记录堆栈和变量来实现的，变量的值由其源代码表示。

例如，考虑以下简单的函数：

```python
def f(a, b):
    return a + b
```

它具有以下字节码：

```shell
0 LOAD_FAST                0 (a)
2 LOAD_FAST                1 (b)
4 BINARY_ADD
6 RETURN_VALUE
```

当我们执行第一个字节码 ``LOAD_FAST`` 时，我们不是将变量加载到堆栈中，而是将变量名 ``"a"`` 推入堆栈，这是一个字符串表示的变量。

当我们执行第二个字节码 ``LOAD_FAST`` 时，同样地，我们将变量名 ``"b"`` 推入堆栈。

当我们执行第三个字节码 ``BINARY_ADD`` 时，它打算将两个变量相加，我们从堆栈中弹出这两个变量，并执行字符串连接 ``"a + b"``。连接后的字符串被推回堆栈。

最后，当我们执行第四个字节码 ``RETURN_VALUE`` 时，我们从堆栈中弹出字符串，在其前面加上 ``return`` 关键字，然后我们得到反编译后的源代码 ``"return a + b"``。

为了准确地反编译字节码，我们需要忠实地遵循 Python 字节码指令的语义。值得注意的是，`Python 字节码文档 <https://docs.python.org/3/library/dis.html>`_ 也可能过时和不准确。黄金标准是参考 CPython 源代码和 Python 解释器的行为。`torchdynamo 源代码 <https://github.com/pytorch/pytorch/blob/main/torch/_dynamo/symbolic_convert.py>`_ 也是一个很好的参考，可以理解 PyTorch 如何生成 Python 字节码。

如果您有任何进一步的问题，请随时在 `GitHub Issues <https://github.com/thuml/depyf/issues>`_ 部分提问。

## 常见QA

### ``depyf`` 是一个通用的 Python 字节码反编译器吗？

这个包旨在理解生成的 PyTorch 字节码，并不旨在完全覆盖 Python 的所有语法。例如，像 ``async/await`` 这样的异步操作是不支持的。

所有在基准测试 timm 和 huggingface transformers 时生成的 PyTorch 字节码都收集在 `这里 <https://github.com/thuml/learn_torch.compile>`_。我们可以做出以下观察：

- 没有 while 循环（没有跳回指令）。
- try-except-finally 只有 try-finally。
- 没有像 ``if a and b or c or (d and e)`` 这样复杂的条件。

然后，我们可以使反编译器过度拟合以适用于 PyTorch 生成的字节码。怎么做？纯粹的劳动工作。为所有支持的 Python 版本逐个实现所有字节码。是的，就是这样。

#### 我如何知道支持的字节码集合？

你可以通过简单地运行 ``python python_coverage.py`` 来查看覆盖率报告。该脚本位于 `仓库 <https://github.com/thuml/depyf/blob/master/python_coverage.py>`_ 中。

#### ``depyf`` 和 ``TORCH_COMPILE_DEBUG`` 有什么区别？

这是一个非常好的问题。

让我们以 readme 中的这个简单示例代码为例：

```python
# test.py
import torch
from torch import _dynamo as torchdynamo
from typing import List

@torch.compile
def toy_example(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b

def main():
    for _ in range(100):
        toy_example(torch.randn(10), torch.randn(10))

if __name__ == "__main__":
    main()
```

当你运行 ``TORCH_COMPILE_DEBUG=1 python test.py`` 时，你会得到一个名为 ``torch_compile_debug/run_2024_02_05_23_02_45_552124-pid_9520`` 的目录。目录内部包含：

```shell

    .
    ├── torchdynamo
    │   └── debug.log
    └── torchinductor
        ├── aot_model___0_debug.log
        ├── aot_model___10_debug.log
        ├── aot_model___11_debug.log
        ├── model__4_inference_10.1
        │   ├── fx_graph_readable.py
        │   ├── fx_graph_runnable.py
        │   ├── fx_graph_transformed.py
        │   ├── ir_post_fusion.txt
        │   ├── ir_pre_fusion.txt
        │   └── output_code.py
        ├── model__5_inference_11.2
        │   ├── fx_graph_readable.py
        │   ├── fx_graph_runnable.py
        │   ├── fx_graph_transformed.py
        │   ├── ir_post_fusion.txt
        │   ├── ir_pre_fusion.txt
        │   └── output_code.py
        └── model___9.0
            ├── fx_graph_readable.py
            ├── fx_graph_runnable.py
            ├── fx_graph_transformed.py
            ├── ir_post_fusion.txt
            ├── ir_pre_fusion.txt
            └── output_code.py
```

当你使用 ``depyf`` 和以下代码时：

```python
# test.py
import torch
from torch import _dynamo as torchdynamo
from typing import List

@torch.compile
def toy_example(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b

def main():
    for _ in range(100):
        toy_example(torch.randn(10), torch.randn(10))

if __name__ == "__main__":
    import depyf
    with depyf.prepare_debug("depyf_debug_dir"):
        main()
```

运行 ``python test.py`` 后，你会得到一个名为 ``depyf_debug_dir`` 的目录，其中包含以下文件：

```shell
    .
    ├── __compiled_fn_0 AFTER POST GRAD 0.py
    ├── __compiled_fn_0 Captured Graph 0.py
    ├── __compiled_fn_0 Forward graph 0.py
    ├── __compiled_fn_0 kernel 0.py
    ├── __compiled_fn_3 AFTER POST GRAD 0.py
    ├── __compiled_fn_3 Captured Graph 0.py
    ├── __compiled_fn_3 Forward graph 0.py
    ├── __compiled_fn_3 kernel 0.py
    ├── __compiled_fn_4 AFTER POST GRAD 0.py
    ├── __compiled_fn_4 Captured Graph 0.py
    ├── __compiled_fn_4 Forward graph 0.py
    ├── __compiled_fn_4 kernel 0.py
    ├── __transformed_code_0_for_torch_dynamo_resume_in_toy_example_at_8.py
    ├── __transformed_code_0_for_toy_example.py
    ├── __transformed_code_1_for_torch_dynamo_resume_in_toy_example_at_8.py
    └── full_code_for_toy_example_0.py
```

那么有什么区别呢？

1. 当你使用 ``TORCH_COMPILE_DEBUG`` 时，``torchdynamo/debug.log`` 文件很长且难以理解。``depyf`` 帮助将字节码反编译成可读的源代码，存储在 ``__transformed_code_xx.py`` 文件中。
2. 当你使用 ``TORCH_COMPILE_DEBUG`` 时，``torchinductor/model__5_inference_11.2`` 等名称非常难以理解。用户必须手动找出 ``torchdynamo/debug.log`` 中的哪个函数对应哪个目录。而在 ``depyf`` 中，``__compiled_fn_0`` 和其他函数的名称与它们在 ``torchdynamo/debug.log`` 中出现的名称完全相同。

总之，``depyf`` 是 ``torch.compile`` 调试信息的改进版本，比 ``TORCH_COMPILE_DEBUG`` 更加完善。


