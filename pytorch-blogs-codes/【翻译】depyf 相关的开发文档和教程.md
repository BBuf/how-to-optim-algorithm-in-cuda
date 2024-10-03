> 地址：https://depyf.readthedocs.io/en/latest/

# depyf 总览

在学习 ``depyf`` 的使用之前，我们建议阅读 《【翻译】torch.compile 的详细示例解析教程》，以便您了解 ``depyf`` 如何帮助您。

``depyf`` 旨在解决 ``torch.compile`` 的两个痛点：

- ``torch.compile`` 转换 Python 字节码，但很少有开发者能够阅读 Python 字节码（除非你的大脑里有一个栈机...）来理解发生了什么。``depyf`` 帮助将转换后的字节码反编译回 Python 源代码，以便开发者能够理解 ``torch.compile`` 如何转换他们的代码。这大大帮助用户适应他们的代码到 ``torch.compile``，以便他们可以编写对 ``torch.compile`` 友好的代码。
- ``torch.compile`` 动态生成许多函数，这些函数只能作为一个黑盒运行。用户无法逐行调试代码。``depyf`` 帮助将源代码dumps到文件中，并将这些函数与源代码文件链接，以便用户可以使用调试器逐行调试这些函数。这大大帮助用户理解 ``torch.compile`` 并在训练过程中调试诸如 ``NaN`` 等问题。

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

第一个上下文管理器 ``depyf.prepare_debug()`` 接受一个目录路径来dumps所有源代码。在这个上下文管理器内部，PyTorch 的所有内部细节将被 ``depyf`` 钩住，它会为你dumps必要的源代码。

第二个上下文管理器 ``depyf.debug()`` 没有参数，它只是禁用新的编译条目。进入这个上下文管理器时，程序将暂停，你可以在你指定的目录下浏览所有源代码（在这个例子中是 ``"./debug_dir"``）。入口文件是 ``full_code_for_xxx.py``。你可以在这些文件中设置断点。最重要的是，你设置的那些断点，在这个上下文管理器下可以被命中。你可以逐行浏览代码，调试可能的 ``NaN`` 值或理解代码发生了什么。

下图展示了 ``depyf`` 的两种典型用法，并列出了所有生成的文件。

![](https://files.mdnice.com/user/59/d1caaaa5-282e-4f83-9edf-5f80472cec72.png)


## API参考

### 理解和调试 ``torch.compile``

> 使用下面的函数之前建议阅读 《【翻译】torch.compile 的详细示例解析教程》 对`torch.compile`有一个基本的理解。

#### `depyf.prepare_debug`

用于dumps `torch.compile` 调试信息的一个上下文管理器。
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

After running the code, you will find the dumped information in the directory ``dump_src_dir``. The details are organized into the following:

- ``full_code_for_xxx.py`` for each function using torch.compile
- ``__transformed_code_for_xxx.py`` for Python code associated with each graph.
- ``__transformed_code_for_xxx.py.xxx_bytecode`` for Python bytecode, dumped code object, can be loaded via ``dill.load(open("/path/to/file", "wb"))``. Note that the load function might import some modules like transformers. Make sure you have these modules installed.
- ``__compiled_fn_xxx.py`` for each computation graph and its optimization:
    - ``Captured Graph``: a plain forward computation graph
    - ``Joint Graph``: joint forward-backward graph from AOTAutograd
    - ``Forward Graph``: forward graph from AOTAutograd
    - ``Backward Graph``: backward graph from AOTAutograd
    - ``kernel xxx``: compiled CPU/GPU kernel wrapper from Inductor.

Arguments:

- ``dump_src_dir``: the directory to dump the source code.
- ``clean_wild_fx_code``: whether to clean the wild fx code that are not recognized for parts of compiled functions. They are usually used by PyTorch internally.
- ``log_bytecode``: whether to log bytecode (original bytecode, transformed bytecode from Dynamo, and decompiled_and_compiled_back_code).

#### `depyf.debug`

A context manager to debug the compiled code. Essentially, it sets a breakpoint to pause the program and allows you to check the full source code in files with prefix `full_code_for_` in the `dump_src_dir` argument of `depyf.prepare_debug()`, and set breakpoints in their separate `__transformed_code_` files according to the function name. Then continue your debugging.

### Decompile general Python Bytecode/Function

#### `depyf.decompile`

Decompile any callable or code object into Python source code.
It is especially useful for some dynamically generated code, like ``torch.compile``,
or ``dataclasses``.

Example usage:

.. code-block:: python

    from dataclasses import dataclass
    @dataclass
    class Data:
        x: int
        y: float

    import depyf
    print(depyf.decompile(Data.__init__))
    print(depyf.decompile(Data.__eq__))

Output:

.. code-block:: python

    def __init__(self, x, y):
        self.x = x
        self.y = y
        return None

    def __eq__(self, other):
        if other.__class__ is self.__class__:
            return (self.x, self.y) == (other.x, other.y)
        return NotImplemented

The output source code is semantically equivalent to the function, but not syntactically the same. It verbosely adds many details that are hidden in the Python code. For example, the above output code of ``__init__`` explicitly returns ``None``, which is typically ignored.

Another detail is that the output code of ``__eq__`` returns ``NotImplemented`` instead of raising ``NotImplemented`` exception when the types are different. At the first glance, it seems to be a bug. However, it is actually the correct behavior. The ``__eq__`` method should return ``NotImplemented`` when the types are different, so that the other object can try to compare with the current object. See `the Python documentation <https://docs.python.org/3/library/numbers.html#implementing-the-arithmetic-operations>`_ for more details.

### Enhance PyTorch Logging

#### `depyf.install`

Decompile any callable or code object into Python source code.
It is especially useful for some dynamically generated code, like ``torch.compile``,
or ``dataclasses``.

Example usage:

.. code-block:: python

    from dataclasses import dataclass
    @dataclass
    class Data:
        x: int
        y: float

    import depyf
    print(depyf.decompile(Data.__init__))
    print(depyf.decompile(Data.__eq__))

Output:

.. code-block:: python

    def __init__(self, x, y):
        self.x = x
        self.y = y
        return None

    def __eq__(self, other):
        if other.__class__ is self.__class__:
            return (self.x, self.y) == (other.x, other.y)
        return NotImplemented

The output source code is semantically equivalent to the function, but not syntactically the same. It verbosely adds many details that are hidden in the Python code. For example, the above output code of ``__init__`` explicitly returns ``None``, which is typically ignored.

Another detail is that the output code of ``__eq__`` returns ``NotImplemented`` instead of raising ``NotImplemented`` exception when the types are different. At the first glance, it seems to be a bug. However, it is actually the correct behavior. The ``__eq__`` method should return ``NotImplemented`` when the types are different, so that the other object can try to compare with the current object. See `the Python documentation <https://docs.python.org/3/library/numbers.html#implementing-the-arithmetic-operations>`_ for more details.

#### `depyf.uninstall`

Uninstall the bytecode hook for PyTorch. Should be called after depyf.install().

## 优化教程

In this tutorial, we will demonstrate how to optimize code using ``torch.compile``, with the help of the ``depyf`` library.

### Example Code

Let's start with a simple example that we want to optimize:

.. code-block:: python

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

This example is intentionally simplified to make the computation easy to follow. In a real-world scenario, the function would perform more complex operations. On a MacOS machine, running the compiled function 10,000 times takes around 0.7 seconds. Our goal is to optimize the code to reduce this execution time.

### Understanding the Code with Depyf

To optimize the code, we first need to understand what's happening during execution. The ``depyf`` library can decompile the bytecode and provide insights. We can add two lines to the previous code:

.. code-block:: python

    # Insert these lines before ``mod(torch.tensor([1]))``
    import depyf
    with depyf.prepare_debug("dump_src_dir/"):
        mod(torch.tensor([1]))  # Compile the function
    # Remaining code as above

After running the code, a new directory named ``dump_src_dir/`` will appear. This directory contains the decompiled source code. For example, in the file ``full_code_for_forward_0.py``, you will find:

.. code-block:: python

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

This is the code that ``torch.compile`` generates to check whether the function's input remains valid for the compiled function. However, many of these checks are overly conservative. For example, ``L['self'].fs[0].i == 0`` checks that ``self.fs[0].i`` is still ``0``, even though our code doesn't modify this value.

Remember from the :doc:`walk_through`, that ``torch.compile`` is a just-in-time compiler. It means all the above checks are executed every time we call the function, introducing significant overhead.

### Optimizing the Code

Since ``torch.compile`` performs these checks every time the function is called, they introduce overhead. To optimize the code, we can bypass these checks. One approach is to modify the ``__guard_0_for_forward`` function to return ``True`` immediately, but ``torch.compile`` doesn't provide a direct mechanism for this.

Instead, we can use ``depyf`` to directly call the compiled function without the checks. The following code demonstrates this approach:

.. code-block:: python

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

In this code, the ``TorchCompileWrapperWithCustomDispatcher`` class is used to bypass the checks. By doing this, the execution time drops to about 0.05 seconds, compared to the original 0.7 seconds. This shows that the checks were responsible for most of the overhead.

### How It Works

``TorchCompileWrapperWithCustomDispatcher`` hijacks the bytecode generated by ``torch.compile`` and directly calls the compiled function without the guards. The ``__call__`` method checks whether a compiled version already exists, and if so, it dispatches directly to the compiled code.

### Real-World Applications

This is an extreme example with a trivial computation, where the overhead introduced by Dynamo is disproportionately large. In practice, the overhead is typically not as severe. However, it can still be significant in high-performance environments, such as when running code on TPU. TPU code is often performance-sensitive, and removing unnecessary checks can lead to substantial speedups.

For example, in `vLLM's TPU integration <https://github.com/vllm-project/vllm/pull/7898>`_, this optimization technique is used to remove Dynamo's overhead, improving TPU throughput by 4%.

### Conclusion

Optimizing code with ``torch.compile`` involves:

1. Using ``depyf`` to decompile the bytecode and understand the performance bottlenecks.
2. Identifying and addressing unnecessary checks or other sources of overhead.
3. Using ``depyf`` to directly call the compiled function, bypassing unnecessary steps where appropriate.

By following these steps, we can significantly improve performance, especially in environments where execution time is critical.

## 开发者文档

For developers, if you want to understand and contribute to the codebase, this section is for you.

### Overall architecture of the library

![](https://files.mdnice.com/user/59/f9a4bbf8-39d8-445a-86e0-1f578bbf0319.png)

The figure above shows the overall architecture of the library.

1. Normally, when we execute Python code, the code is compiled into Python bytecode, which is then executed by the Python interpreter.
2. When ``torch.compile`` is used, PyTorch will compile the function into a new bytecode object to execute. It achieves this via registering a frame-evaluation function to the Python interpreter. The frame-evaluation function will be called whenever the function is executed. PyTorch wraps the frame-evaluation callback registration via the ``torch._C._dynamo.eval_frame.set_eval_frame`` function. Because PyTorch directly generates the bytecode, it does not have the source code information. The bytecode is directly executed by the Python interpreter.
3. When ``depyf`` is used together with PyTorch, it will register a bytecode hook to PyTorch via ``torch._dynamo.convert_frame.register_bytecode_hook`` (we work together with the PyTorch team to design this bytecode hook mechanism). The hook will be called whenever PyTorch compiles a function. The hook will decompile the bytecode into source code and dump the source code to disk. The source code is then compiled into a new bytecode object, which is functionally equivalent to the bytecode generated by PyTorch, but with source code information. PyTorch will use the new bytecode object to execute the function. The part related with ``depyf`` is marked as green.

With this, it is clear that the library is a Python bytecode decompiler with tight integration with PyTorch. It naturally falls into 2 parts:

* The decompiler is implemented in the ``depyf/decompiler.py`` file. It can also be used as a standalone library to decompile Python bytecode.
* The PyTorch integration is implemented in the ``depyf/explain/enable_debugging.py`` file. It also contains lots of code to deal with the rest of the PyTorch compiler, such as graph compilation and transformation, code guards and caches, etc.

Relatively speaking, the PyTorch integration part is easier to understand and contribute. Our main goal for the integration is to make ``depyf`` compatible with all previous versions of PyTorch starting from PyTorch 2.2 . To achieve this goal, the test is run against the nightly build of PyTorch. Whenever we find a compatibility issue, we will fix it in a backward-compatible way. If such a fix is not possible, we will discuss with the PyTorch team to find a solution.

The decompiler part is more challenging. It is complicated and needs to deal with all sorts of random Python implementation details. Fortunately, we only need to deal with official release versions of Python, which makes the task more manageable. The decompiler only needs to be updated once we find a bug or a new Python version is released.

If you want to dive deeper into the decompiler part, please go on reading.

Overview of the decompiler
--------------------------

To become comfortable with reading bytecode, it is recommended to read the following materials first:

- `torchdynamo deepdive <https://www.youtube.com/watch?v=egZB5Uxki0I>`_ : This video explains the motivation and design of torchdynamo. In particular, it mentions how Python bytecode acts like a stack machine, which helps to understand how the bytecode is executed.
- `Python bytecode documentation <https://docs.python.org/3/library/dis.html>`_ : This documentation explains the Python bytecode instructions. Note that Python bytecode does not guarentee any backward compatibility, so the bytecode instructions may change for every Python versions. We should consider all the supported Python versions when implementing the decompiler.
- `A Python Interpreter Written in Python <https://aosabook.org/en/500L/a-python-interpreter-written-in-python.html>`_ : This book chapter explains how to write a Python interpreter in Python. It is a good starting point to understand how Python bytecode is executed.

The decompilation process is achieved by executing the Python bytecode and recording the stack and the variables, with the value of the variables represented by their source code.

For example, consider the following simple function:

.. code-block:: python

    def f(a, b):
        return a + b

It has the following bytecode:

.. code-block:: text

    0 LOAD_FAST                0 (a)
    2 LOAD_FAST                1 (b)
    4 BINARY_ADD
    6 RETURN_VALUE

When we execute the first bytecode ``LOAD_FAST``, instead of loading a variable into the stack, we push the variable name ``"a"`` in the stack, which is a string representation of the variable.

When we execute the second bytecode ``LOAD_FAST``, likewise, we push the variable name ``"b"`` in the stack.

When we execute the third bytecode ``BINARY_ADD``, which intends to add the two variables, we pop the two variables from the stack, and perform the string concatenation ``"a + b"``. The concatenated string is pushed back to the stack.

Finally, when we execute the fourth bytecode ``RETURN_VALUE``, we pop the string from the stack, prefix it with the ``return`` keyword, and then we get the decompiled source code ``"return a + b"``.

To accurately decompile the bytecode, we need to faithfully respect the semantics of the Python bytecode instructions. It is noteworthy that the `Python bytecode documentation <https://docs.python.org/3/library/dis.html>`_ can be outdated and inaccurate, too. The golden standard is to refer to the CPython source code and the Python interpreter's behavior. The `torchdynamo source code <https://github.com/pytorch/pytorch/blob/main/torch/_dynamo/symbolic_convert.py>`_ is also a good reference to understand how the Python bytecode is generated by PyTorch.

Should you have any further questions, feel free to ask in the `GitHub Issues <https://github.com/thuml/depyf/issues>`_ section.

## 常见QA

### Is ``depyf`` a general purpose Python bytecode decompiler?

This package is intended to understand the generated PyTorch bytecode, and does not aim to fully cover all the syntax of Python. For example, async operations like ``async/await`` are not supported.

All the bytecode generated by PyTorch when benchmarking timm and huggingface transformers are collected `here <https://github.com/thuml/learn_torch.compile>`_. We can make several observations:

- No while loops (no jump back instructions).
- try-except-finally only has try-finally.
- No complicated conditions like ``if a and b or c or (d and e)``.

Then, we can overfit the decompiler to work for the bytecode generated by PyTorch. How? Pure labor work. Implement all bytecode for all the supported Python versions, one by one. Yes, that's it.

#### How do I know the set of supported bytecode?


You can see the coverage report by simply running ``python python_coverage.py``. The script is located in the `repository <https://github.com/thuml/depyf/blob/master/python_coverage.py>`_.

#### What is the difference between ``depyf`` and ``TORCH_COMPILE_DEBUG``?

That's a very good question.

Let's take this simple example code from readme:

.. code-block:: python

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

When you run ``TORCH_COMPILE_DEBUG=1 python test.py``, you will get a directory named ``torch_compile_debug/run_2024_02_05_23_02_45_552124-pid_9520``. Inside the directory:

.. code-block:: text

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

When you use ``depyf`` with the following code:

.. code-block:: python

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

After running ``python test.py``, you get a directory ``depyf_debug_dir``, under which are these files:

.. code-block:: text

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

So what's the difference?

1. When you use ``TORCH_COMPILE_DEBUG``, the ``torchdynamo/debug.log`` is long and difficult to understand. ``depyf`` helps to decompile the bytecode into readable source code in ``__transformed_code_xx.py`` file.
2. When you use ``TORCH_COMPILE_DEBUG``, the ``torchinductor/model__5_inference_11.2`` etc names are very difficult to understand. Users have to manually figure out which function inside ``torchdynamo/debug.log`` corresponds to which directory. Meanwhile, in ``depyf``, ``__compiled_fn_0`` and other functions have exactly the same names as they appear in ``torchdynamo/debug.log``.

In summary, ``depyf`` is a much more improved version of debugging information for ``torch.compile`` than ``TORCH_COMPILE_DEBUG``.
