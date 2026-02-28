> 教程链接：https://depyf.readthedocs.io/en/latest/walk_through.html


# ``torch.compile`` 的详细示例解析

本教程旨在涵盖 PyTorch 编译器的以下几个方面：

- 基本概念（Just-In-Time 编译器、Ahead-of-time 编译器）
- Dynamo（图捕获，将用户代码分离为纯 Python 代码和纯 PyTorch 相关代码）
- AOTAutograd（从前向计算图生成反向计算图）
- Inductor/其他后端（给定计算图，如何在不同设备上更快地运行）

这些组件将使用不同的后端选项进行调用：

- ``torch.compile(backend="eager")`` 仅使用 Dynamo
- ``torch.compile(backend="aot_eager")`` 使用 Dynamo 和 AOTAutograd
- ``torch.compile(backend="inductor")``（默认参数）使用 Dynamo、AOTAutograd 和 PyTorch 的内置图优化后端 ``Inductor``。

## PyTorch 编译器是一个 Just-In-Time 编译器

我们首先需要了解的概念是 PyTorch 编译器是一个Just-In-Time 编译器。那么什么是"Just-In-Time 编译器"呢？让我们看一个例子：

```python
import torch

class A(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.exp(2 * x)

class B(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.exp(-x)

def f(x, mod):
    y = mod(x)
    z = torch.log(y)
    return z

# 用户可能会使用
# mod = A()
# x = torch.randn(5, 5, 5)
# output = f(x, mod)

```

我们编写了这个有趣的函数 ``f``，它包含一个模块调用（会调用 ``mod.forward``）和一个 ``torch.log`` 调用。由于众所周知的代数简化恒等式 $\log(\exp(a\times x))=a\times x$，我们迫不及待地想要将代码优化如下：

```python
def f(x, mod):
    if isinstance(mod, A):
        return 2 * x
    elif isinstance(mod, B):
        return -x
```

我们可以称之为我们的第一个编译器，尽管它是由我们的大脑"编译"的，而不是由自动化程序完成的。

如果我们想要更严谨一些，我们的编译器示例应该更新如下：

```python
def f(x, mod):
    if isinstance(x, torch.Tensor) and isinstance(mod, A):
        return 2 * x
    elif isinstance(x, torch.Tensor) and isinstance(mod, B):
        return -x
    else:
        y = mod(x)
        z = torch.log(y)
        return z
```

我们必须检查每个参数，以确保我们的优化条件是合理的，并且在无法优化代码时回退到原始代码。

这引出了（Just-In-Time）编译器中的两个基本概念：守卫（guards） 和 转换后的代码。守卫是函数可以被优化的条件，而转换后的代码是函数的优化版本。在上面的简单编译器示例中，``isinstance(mod, A)`` 是一个守卫，而 ``return 2 * x`` 是在守卫条件下等价于原始代码但明显更快的相应转换后的代码。

上面的例子是一个Ahead-of-time编译器：我们检查所有可用的源代码，并在运行任何函数之前（即Ahead-of-time），根据所有可能的守卫和转换后的代码编写优化后的函数。

另一类编译器是Just-In-Time 编译器：在函数执行之前，它会分析执行是否可以被优化，以及在什么条件下可以优化函数执行。希望这个条件对新的输入足够通用，以便编译带来的好处超过即时编译的成本。如果所有条件都失败，它将尝试在新条件下优化代码。

Just-In-Time 编译器的基本工作流程应该如下所示：

```python
def f(x, mod):
    for guard, transformed_code in f.compiled_entries:
        if guard(x, mod):
            return transformed_code(x, mod)
    try:
        guard, transformed_code = compile_and_optimize(x, mod)
        f.compiled_entries.append([guard, transformed_code])
        return transformed_code(x, mod)
    except FailToCompileError:
        y = mod(x)
        z = torch.log(y)
        return z
```

Just-In-Time 编译器只针对它所见到的内容进行优化。每次它看到一个不满足任何守卫条件的新输入时，它都会为新输入编译一个新的守卫和转换后的代码。

让我们逐步解释编译器的状态（以守卫和转换后的代码的形式）：

```python
import torch

class A(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.exp(2 * x)

class B(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.exp(-x)

@just_in_time_compile # 一个假想的编译器函数
def f(x, mod):
    y = mod(x)
    z = torch.log(y)
    return z

a = A()
b = B()
x = torch.randn((5, 5, 5))

# 在执行 f(x, a) 之前，f.compiled_entries == [] 为空。
f(x, a)
# 执行 f(x, a) 之后，f.compiled_entries == [Guard("isinstance(x, torch.Tensor) and isinstance(mod, A)"), TransformedCode("return 2 * x")]

# 第二次调用 f(x, a) 命中一个条件，所以我们可以直接执行转换后的代码
f(x, a)

# f(x, b) 将触发编译并添加一个新的编译条目
# 在执行 f(x, b) 之前，f.compiled_entries == [Guard("isinstance(x, torch.Tensor) and isinstance(mod, A)"), TransformedCode("return 2 * x")]
f(x, b)
# 执行 f(x, b) 之后，f.compiled_entries == [Guard("isinstance(x, torch.Tensor) and isinstance(mod, A)"), TransformedCode("return 2 * x"), Guard("isinstance(x, torch.Tensor) and isinstance(mod, B)"), TransformedCode("return -x")]

# 第二次调用 f(x, b) 命中一个条件，所以我们可以直接执行转换后的代码
f(x, b)
```

在这个例子中，我们对类型进行守卫，如 ``isinstance(mod, A)``，而 ``TransformedCode`` 也是 Python 代码；对于 ``torch.compile``，它对更多条件进行守卫，如设备（CPU/GPU）、数据类型（int32, float32）、形状（``[10]``、``[8]``），而其 ``TransformedCode`` 是 Python 字节码。我们可以从函数中提取这些编译条目，详情请参阅 `PyTorch 文档 <https://pytorch.org/docs/main/torch.compiler_deepdive.html#how-to-inspect-artifacts-generated-by-torchdynamo>`_。尽管守卫和转换后的代码有所不同，但 ``torch.compile`` 的基本工作流程与这个例子相同，即它作为一个Just-In-Time 编译器工作。

## 超越代数简化的优化

上面的例子是关于代数简化的。然而，这种优化在实践中相当罕见。让我们看一个更实际的例子，看看 PyTorch 编译器如何处理以下代码：

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
# 预热
for i in range(100):
    output = function(shape_10_inputs)
    output = function(shape_8_inputs)

# 执行编译后的函数
output = function(shape_10_inputs)
```

这段代码试图实现一个激活函数$\text{cos}(\text{cos}(x))$，并根据其激活值缩放输出，然后将输出与另一个张量 ``y`` 相乘。

## Dynamo 如何转换和修改函数？

在我们理解了 ``torch.compile`` 作为Just-In-Time 编译器的全局图景后，我们可以深入了解它是如何工作的。与 ``gcc`` 或 ``llvm`` 等通用编译器不同，``torch.compile`` 是一个领域特定的编译器：它只关注与 PyTorch 相关的计算图。因此，我们需要一个工具将用户代码分为两部分：纯 Python 代码和计算图代码。

``Dynamo``，位于模块 ``torch._dynamo`` 中，就是用于完成这项工作的工具。通常我们不直接与这个模块交互。它在 ``torch.compile`` 函数内部被调用。

从概念上讲，``Dynamo`` 做以下几件事：

- 找到第一个不能在计算图中表示但需要计算值的操作（例如，``print`` 一个张量的值，使用张量的值来决定 Pytho    n 中的 ``if`` 语句控制流）。
- 将之前的操作分为两部分：一个纯粹关于张量计算的计算图，和一些关于操作 Python 对象的 Python 代码。
- 将剩余的操作作为一个或两个新函数（称为 ``resume functions``）留下，并再次触发上述分析。

为了实现对函数如此细粒度的操作，``Dynamo`` 在 Python 字节码级别上操作，这是比 Python 源代码更低的级别。

以下过程描述了 Dynamo 对我们的函数所做的操作：

![](https://files.mdnice.com/user/59/48fcf697-335c-4880-8d45-87c0ba298fb7.png)

> 图片解释：

- **源代码 (Source Code)** ： 图中展示了一个简单的 Python 函数 `function(inputs)`，该函数对输入的两个张量 `x` 和 `y` 进行一些数学运算（例如 `cos` 和 `mean`）后返回 `x * y`。
- **输入 (Input)** ： 显示了该函数的输入 inputs 包含 `x` 和 `y` 两个 `torch.Tensor` 类型的张量。
- **守护函数 (Guard)** ： 守护函数基于输入和 Python 字节码生成，确保编译器在运行时验证张量 `x` 和 `y` 的形状和类型，以决定是否需要重新编译或可以直接执行。
- **Python 字节码分析 (Python Bytecode Analysis)** ： 图中显示了 Python 源代码经过编译器分析后生成的字节码。这里将 Python 操作（与张量计算无关的部分）和纯张量计算的 PyTorch 操作（如 cos、mean 等）分开列出。
- **转换后的字节码 (Transformed Bytecode)** ： 在 PyTorch 编译器将源码转换成字节码后，会根据不同操作生成相应的字节码指令，同时根据守护条件来决定执行流程。比如图中展示了 `_compiled_fn` 的实现和条件跳转操作。
- **恢复函数 (Resume Functions)** ： 当某些操作需要的条件不满足（比如某个张量值需要重新计算）时，编译器会触发恢复函数（resume function），该恢复函数可以继续进行需要的计算，并且递归地触发字节码分析。
- **执行流程 (Execution Workflow)** ： 最后，图中展示了如何从原始的字节码指令通过守护条件、张量计算图、恢复函数等一系列操作形成最终的执行字节码流程。

> 注意：字节码后标注了对应的原始的python代码

``Dynamo`` 的一个重要特性是，它可以分析 ``function`` 函数内部调用的所有函数。如果一个函数可以完全用计算图表示，那么该函数调用将被内联，函数调用将被消除。

``Dynamo`` 的任务是以安全和可靠的方式从 Python 代码中提取计算图。一旦我们有了计算图，我们就可以进入计算图优化的世界了。

> 上述工作流程包含许多难以理解的字节码。对于那些无法阅读 Python 字节码的人来说，``depyf`` 可以帮上忙！查看 https://depyf.readthedocs.io/en/latest/ 了解更多详情。

## Dynamo 的动态形状支持

深度学习编译器通常偏好静态形状输入。这就是为什么上面的守卫条件包括形状守卫。我们的第一个函数调用使用形状为 ``[10]`` 的输入，但第二个函数调用使用形状为 ``[8]`` 的输入。它将失败形状守卫，因此触发新的代码变换。

默认情况下，Dynamo 支持动态形状。当形状守卫失败时，它将分析和比较形状，并尝试泛化形状。在这种情况下，在看到形状为 ``[8]`` 的输入后，它将尝试泛化为任意一维形状 ``[s0]``，称为动态形状或符号形状。

## AOTAutograd：从前向图生成反向计算图

上述代码只处理前向计算图。一个重要的缺失部分是如何获得反向计算图来计算梯度。

在普通的 PyTorch 代码中，反向计算是通过对某个标量损失值调用 ``backward`` 函数触发的。每个 PyTorch 函数在前向计算期间存储反向所需的内容。

为了解释在 eager 模式下反向过程中发生的事情，我们有以下实现，模仿 ``torch.cos`` 函数的内置行为（需要一些关于如何在 PyTorch 中编写具有自动求导支持的自定义函数的 `背景知识 <https://pytorch.org/docs/main/notes/extending.html#extending-torch-autograd>`_）：

```python

import torch
class Cosine(torch.autograd.Function):
    @staticmethod
    def forward(x0):
        x1 = torch.cos(x0)
        return x1, x0

    @staticmethod
    def setup_context(ctx, inputs, output):
        x1, x0 = output
        print(f"保存大小为 {x0.shape} 的张量")
        ctx.save_for_backward(x0)

    @staticmethod
    def backward(ctx, grad_output):
        x0, = ctx.saved_tensors
        result = (-torch.sin(x0)) * grad_output
        return result

# 将 Cosine 包装在一个函数中，以便更清楚地看出输出是什么
def cosine(x):
    # `apply` 将调用 `forward` 和 `setup_context`
    y, x= Cosine.apply(x)
    return y

def naive_two_cosine(x0):
    x1 = cosine(x0)
    x2 = cosine(x1)
    return x2
```

使用需要梯度的输入运行上述函数，我们可以看到保存了两个张量：

```python
input = torch.randn((5, 5, 5), requires_grad=True)
output = naive_two_cosine(input)
```

输出：

```shell
保存大小为 torch.Size([5, 5, 5]) 的张量
保存大小为 torch.Size([5, 5, 5]) 的张量
```

如果我们提前有计算图，我们可以将计算转换如下：

```python
class AOTTransformedTwoCosine(torch.autograd.Function):
    @staticmethod
    def forward(x0):
        x1 = torch.cos(x0)
        x2 = torch.cos(x1)
        return x2, x0

    @staticmethod
    def setup_context(ctx, inputs, output):
        x2, x0 = output
        print(f"保存大小为 {x0.shape} 的张量")
        ctx.save_for_backward(x0)

    @staticmethod
    def backward(ctx, grad_x2):
        x0, = ctx.saved_tensors
        # 在反向中重新计算
        x1 = torch.cos(x0)
        grad_x1 = (-torch.sin(x1)) * grad_x2
        grad_x0 = (-torch.sin(x0)) * grad_x1
        return grad_x0

def AOT_transformed_two_cosine(x):
    # `apply` 将调用 `forward` 和 `setup_context`
    x2, x0 = AOTTransformedTwoCosine.apply(x)
    return x2
```

使用需要梯度的输入运行上述函数，我们可以看到只保存了一个张量：

```python
input = torch.randn((5, 5, 5), requires_grad=True)
output = AOT_transformed_two_cosine(input)
```

输出：

```shell
保存大小为 torch.Size([5, 5, 5]) 的张量
```

我们可以检查两种实现与原生 PyTorch 实现的正确性：

```python

input = torch.randn((5, 5, 5), requires_grad=True)
grad_output = torch.randn((5, 5, 5))

output1 = torch.cos(torch.cos(input))
(output1 * grad_output).sum().backward()
grad_input1 = input.grad; input.grad = None

output2 = naive_two_cosine(input)
(output2 * grad_output).sum().backward()
grad_input2 = input.grad; input.grad = None

output3 = AOT_transformed_two_cosine(input)
(output3 * grad_output).sum().backward()
grad_input3 = input.grad; input.grad = None

assert torch.allclose(output1, output2)
assert torch.allclose(output1, output3)
assert torch.allclose(grad_input1, grad_input2)
assert torch.allclose(grad_input1, grad_input3)
```

以下计算图显示了朴素实现的细节：

![](https://files.mdnice.com/user/59/b17a88b4-da1a-4622-b549-d386c9382712.png)

以下计算图显示了转换后实现的细节：

![](https://files.mdnice.com/user/59/15e9cab7-3836-4bea-9ce3-53727461ae81.png)

我们只需保存一个值，并重新计算第一个 ``cos`` 函数以获得反向所需的另一个值。请注意，额外的计算并不意味着更多的计算时间：现代设备如 GPU 通常受内存限制，即内存访问时间主导了计算时间，稍微多一些计算并不重要。

AOTAutograd 自动执行上述转换。本质上，它动态生成一个类似以下的函数：

```python

class AOTTransformedFunction(torch.autograd.Function):
    @staticmethod
    def forward(inputs):
        outputs, saved_tensors = forward_graph(inputs)
        return outputs, saved_tensors

    @staticmethod
    def setup_context(ctx, inputs, output):
        outputs, saved_tensors = output
        ctx.save_for_backward(saved_tensors)

    @staticmethod
    def backward(ctx, grad_outputs):
        saved_tensors = ctx.saved_tensors
        grad_inputs = backward_graph(grad_outputs, saved_tensors)
        return grad_inputs

def AOT_transformed_function(inputs):
    outputs, saved_tensors = AOTTransformedFunction.apply(inputs)
    return outputs
```

这样，保存的张量被明确化，而 ``AOT_transformed_function`` 接受与原始函数完全相同的输入，同时产生与原始函数完全相同的输出，并具有与原始函数完全相同的反向行为。

通过改变 ``saved_tensors`` 的数量，我们可以为反向保存更少的张量，从而减轻前向的内存占用。AOTAutograd 将自动选择最佳的内存节省方式。具体来说，它使用 `最大流最小割 <https://en.wikipedia.org/wiki/Minimum_cut>`_ 算法将联合图切割成前向图和反向图。更多讨论可以在 `这个thread <https://dev-discuss.pytorch.org/t/min-cut-optimal-recomputation-i-e-activation-checkpointing-with-aotautograd/467>`_ 中找到。

这基本上就是 AOT Autograd 的工作原理！

注意：如果你好奇如何获得函数的联合图，这里是代码：

```python

def run_autograd_ahead_of_time(function, inputs, grad_outputs):
    def forward_and_backward(inputs, grad_outputs):
        outputs = function(*inputs)
        grad_inputs = torch.autograd.grad(outputs, inputs, grad_outputs)
        return grad_inputs
    from torch.fx.experimental.proxy_tensor import make_fx
    wrapped_function = make_fx(forward_and_backward, tracing_mode="fake")
    joint_graph = wrapped_function(inputs, grad_outputs)
    print(joint_graph._graph.python_code(root_module="self", verbose=True).src)

def f(x0):
    x1 = torch.cos(x0)
    x2 = torch.cos(x1)
    return x2

input = torch.randn((5, 5, 5), requires_grad=True)
grad_output = torch.randn((5, 5, 5)) # 与输出形状相同
run_autograd_ahead_of_time(f, [input], [grad_output])
```

这个函数将从真实输入创建一些假张量，并仅使用元数据（形状、设备、数据类型）进行计算。因此，AOTAutograd 组件是提前运行的。这就是它得名的原因：AOTAutograd 是提前运行自动求导引擎。

## 后端：编译和优化计算图

最后，在 ``Dynamo`` 将 PyTorch 代码从 Python 代码中分离出来，并且在 ``AOTAutograd`` 从前向计算图生成反向计算图之后，我们进入了纯计算图的世界。

这就是 ``torch.compile`` 中的 ``backend`` 参数发挥作用的地方。它将上述计算图作为输入，并生成可以在不同设备上执行上述计算图的优化代码。

``torch.compile`` 的默认后端是 ``"inductor"``。它将尝试它所知道的每一种优化技术来优化计算图。每种优化技术被称为一个 ``pass``。一些优化 pass 可以在 `PyTorch 仓库 <https://github.com/pytorch/pytorch/tree/main/torch/_inductor/fx_passes>`_ 中找到。

> 一个非常重要的优化是"kernel fuse"。对于不熟悉硬件进展的人来说，他们可能会惊讶于现代硬件在计算方面如此之快，以至于它们通常受内存限制：从内存读取和写入内存占用了最多的时间。在上面的例子中，优化前的反向图读取 :$\nabla x_2, x_1, x_0$ 并写入 $\nabla x_0$；优化后，反向图读取 :$\nabla x_2, x_0$ 并写入 $\nabla x_0$。优化后的反向图可能只需要优化前图 75% 的时间，尽管它重新计算了 $x_1$。

此外，不进行优化也是一种可能的优化。这在 PyTorch 中被称为 ``eager`` 后端。

严格来说，``torch.compile`` 中的 ``backend`` 选项影响反向计算图是否存在以及如何优化计算图。在实践中，自定义后端通常与 ``AOTAutograd`` 一起工作以获得反向计算图，它们只需要处理计算图优化，无论是前向图还是反向图。

## 总结

下表显示了 ``torch.compile`` 中几个 ``backend`` 选项的区别。如果我们想要适应我们的代码到 ``torch.compile``，建议首先尝试 ``backend="eager"`` 来看看我们的代码如何转换为计算图，然后尝试 ``backend="aot_eager"`` 来看看我们是否满意反向图，最后尝试 ``backend="inductor"`` 来看看我们是否能获得任何性能提升。

![](https://files.mdnice.com/user/59/933488bc-3a0b-4793-a805-c02bcdf6a28a.png)

























































