> 博客链接：https://pytorch.org/blog/introducing-depyf/

> 最近了解`torch.compile`的时候，发现清华推出了一个可以帮助我们理解`torch.compile`到底对我们的代码做了什么优化的库depyf，这篇教程是这个库的一个简要介绍，前面对这个教程做了一个翻译。后面一部分，我利用cursor来完整展示了如何完整的阅读depfy生成的`torch.compile`编译产物的例子，我们可以看到`torch.compile`优化的每个子图以及产生的fuse kernel，希望对感兴趣的读者有帮助。



# 介绍 depyf：轻松掌握 torch.compile

![](https://files.mdnice.com/user/59/bb428f7f-2674-4be7-9007-d19ffd790dd8.png)

我们很高兴介绍 depyf，这是 PyTorch 生态系统中的一个新项目，旨在帮助用户理解、学习和适应 `torch.compile` ！

## 动机

`torch.compile` 是 PyTorch 2.x 的一个基石，为加速机器学习工作流程提供了一个直接的途径，只需一行代码就可以同时用于训练和推理。仅仅包含 `@torch.compile` 就可以显著提升你的代码性能。然而，找到 `torch.compile` 的最佳插入点并不容易，更不用说为了最大效率而调整各种参数的复杂性。

`torch.compile` 技术栈的复杂性，包括 Dynamo、AOTAutograd、Inductor 等，呈现出一个陡峭的学习曲线。这些对深度学习性能优化至关重要的组件，如果没有坚实的基础知识，可能会令人望而生畏。

> 注：关于 `torch.compile` 工作原理的入门示例，请参阅这个逐步说明(https://depyf.readthedocs.io/en/latest/walk_through.html)。

## 一个常用工具：TORCH_COMPILE_DEBUG

为了揭开 `torch.compile` 的神秘面纱，常用的方法是利用 `TORCH_COMPILE_DEBUG` 环境变量。虽然它提供了更多信息，但解读输出仍然是一项艰巨的任务。

例如，当我们有以下代码：

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

当我们用 `TORCH_COMPILE_DEBUG=1 python test.py` 运行它时，我们会得到一个名为 `torch_compile_debug/run_2024_02_05_23_02_45_552124-pid_9520` 的目录，其中包含这些文件：

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

生成的文件和日志常常引发的问题比它们解答的还多，让开发者对数据的含义和关系感到困惑。`TORCH_COMPILE_DEBUG` 的常见疑问包括：

- `model__4_inference_10.1` 是什么意思？
- 我只有一个函数，但目录中有三个 `model__xxx.py`，它们之间有什么对应关系？
- `debug.log` 中那些 `LOAD_GLOBAL` 是什么东西？

## 更好的工具：`DEPYF` 来救援

让我们看看 `depyf` 如何帮助开发者解决上述挑战。要使用 `depyf`，只需执行 `pip install depyf` 或按照项目页面 https://github.com/thuml/depyf 安装最新版本，然后用 `with depyf.prepare_debug` 包围主代码。

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

执行 `python test.py` 后，`depyf` 将生成一个名为 `depyf_debug_dir`（`prepare_debug` 函数的参数）的目录。在该目录下，会有这些文件：

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

这里有两个明显的好处：

- 冗长且难以理解的 `torchdynamo/debug.log` 不见了。它的内容被整理并以人类可读的源代码形式显示在 `full_code_for_xxx.py` 和 `_transformed_code{n}_for_xxx.py` 中。值得注意的是，`depyf` 最艰巨和困难的任务是将 `torchdynamo/debug.log` 中的字节码反编译成 Python 源代码，从而使开发者免于被 Python 内部结构所困扰。
- 函数名称与计算图之间的对应关系得到了保留。例如，在 `__transformed_code_0_for_toy_example.py` 中，我们可以看到一个名为 `__compiled_fn_0` 的函数，我们立即就知道它对应的计算图在 `__compiled_fn_0_xxx`.py 中，因为它们共享相同的 `__compiled_fn_0` 前缀名称。

从 `full_code_for_xxx.py` 开始，并跟随涉及的函数，用户将清楚地了解 `torch.compile` 对他们的代码做了什么。

## 再补充一点：逐步调试功能

使用调试器逐行步进代码是理解代码工作原理的好方法。然而，在 `TORCH_COMPILE_DEBUG` 模式下，这些文件仅供用户参考，无法与用户关心的数据一起执行。

> 注：这里的"调试"指的是检查和改进程序的过程，而不是纠正有问题的代码。

**`depyf` 的一个突出特点是它能够为 `torch.compile` 提供逐步调试功能：它生成的所有文件都与 Python 解释器内部的运行时代码对象链接，我们可以在这些文件中设置断点**。使用方法很简单，只需添加一个上下文管理器 `with depyf.debug()`，它就能发挥作用。

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
   with depyf.debug():
       main()
```

需要注意的一点是：调试 `torch.compile` 的工作流程与标准调试工作流程有所不同。使用 `torch.compile` 时，许多代码是**动态**生成的。因此，我们需要：

- 启动程序
- 当程序退出 `with depyf.prepare_debug("depyf_debug_dir")` 时，代码将在 `depyf_debug_dir` 中可用。
- 当程序进入 `with depyf.debug()` 时，它会在内部自动设置一个断点，使程序暂停。
- 导航到 `depyf_debug_dir` 设置断点。
- 继续运行代码，调试器将会命中这些断点！


![](https://files.mdnice.com/user/59/41640327-d62f-45ce-a820-4f39ddc0bf30.png)

这是它看起来的样子的截图。所有代码和张量变量都是实时的，我们可以检查任何变量，并像日常调试工作流程一样逐步执行代码！唯一的区别是我们正在调试 `torch.compile` 生成的代码，而不是人工编写的代码。

## 结论

`torch.compile` 是一个无价的工具，可以轻松加速 PyTorch 代码。对于那些希望深入研究 `torch.compile` 的人来说，无论是为了充分发挥其潜力还是集成自定义操作，学习曲线可能会非常陡峭。`depyf` 旨在降低这个障碍，提供一个用户友好的体验来理解、学习和适应 `torch.compile`。

请亲自探索 `depyf` 并体验其好处！该项目是开源的，可以在 https://github.com/thuml/depyf 上获取。安装非常简单，只需通过 pip install depyf 即可。我们希望 `depyf` 能够改善每个人使用 `torch.compile` 的开发工作流程。

## 个人补充：用cursor辅助阅读depyf code

使用上面的例子：

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

可以顺利得到下面的：

![](https://files.mdnice.com/user/59/a318bd60-f874-41ae-9e01-a081271eeaab.png)

从`full_code_for_toy_example_0.py`中我们就可以来把握`torch.compile`到底对我们的代码做了什么优化。这个过程中我们还可以利用LLM加速我们理解代码的过程。例如，在Cursor中，我选中这个文件后输入：可以根据这个文件的内容帮我画一个流程图出来吗？它就会根据这个代码生成一个流程图出来，输出如下：

```shell
开始
  |
  v
[输入 a 和 b]
  |
  v
[检查 guard 条件]
  |
  +-> 如果 guard 条件满足:
  |     |
  |     v
  |   [执行 __transformed_code_0_for_toy_example]
  |     |
  |     v
  |   [计算 x = a / (torch.abs(a) + 1)]
  |     |
  |     v
  |   [检查 b.sum() < 0]
  |     |
  |     +-> 如果为真:
  |     |     |
  |     |     v
  |     |   [执行 __resume_at_30_2]
  |     |     |
  |     |     v
  |     |   [b = b * -1]
  |     |
  |     +-> 如果为假:
  |           |
  |           v
  |         [执行 __resume_at_38_3]
  |
  +-> 如果 guard 条件不满足:
        |
        v
      [执行原始的 toy_example 函数]
        |
        v
      [计算 x = a / (torch.abs(a) + 1)]
        |
        v
      [检查 b.sum() < 0]
        |
        +-> 如果为真:
        |     |
        |     v
        |   [b = b * -1]
        |
        +-> 如果为假:
              |
              v
            [不修改 b]
  |
  v
[返回 x * b]
  |
  v
结束
```

这个流程图描述了 toy_example 函数的执行过程,包括了优化后的路径和原始路径。主要的区别在于是否满足 guard 条件,这决定了是执行优化后的代码还是原始代码。两种路径最终都会计算并返回 x * b。

然后可以问它`__guard_0_for_torch_dynamo_resume_in_toy_example_at_8`这个函数的作用：

![](https://files.mdnice.com/user/59/d792cb11-9fc0-4b37-a8dd-3a820ba8f6ad.png)

这些检查都是根据输入Tensor的meta信息，python object信息，以及当前运行环境等来判断的。通过上面的流程图我们就可以一步步的去看`torch.compile`做了什么，例如`__transformed_code_0_for_toy_example`函数中`__resume_at_30_2`：

```python
def __transformed_code_1_for_torch_dynamo_resume_in_toy_example_at_8(b, x):
    a = None # this line helps Python to generate bytecode with at least the same number of local variables as the original function
    __temp_9, = __compiled_fn_7(b, x)
    return __temp_9

# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_30_2(b, x):
    b = b * -1
    return x * b

def transformed___resume_at_30_2(b, x):
    __local_dict = {"b": b, "x": x}
    __global_dict = globals()
    if __guard_1_for_torch_dynamo_resume_in_toy_example_at_8(__local_dict, __global_dict):
        return __transformed_code_1_for_torch_dynamo_resume_in_toy_example_at_8(b, x)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_30_2(b, x)

def __transformed_code_0_for_toy_example(a, b):
    __temp_2, __temp_3 = __compiled_fn_1(a, b)
    x = __temp_2
    if __temp_3:
        return __resume_at_30_2(b, x)
    return __resume_at_38_3(b, x)
```

这个时候我们就轻松知道我们应该去查看`__compiled_fn_7`这个函数对应的编译产物了，如下图红色所示：

![](https://files.mdnice.com/user/59/9af81855-4d8c-4802-b334-bb7f97c79ce2.png)

打开`_compiled_fn_7_kernel0.py`文件，我们可以看到原始的：

```python
def __resume_at_30_2(b, x):
    b = b * -1
    return x * b
```

被fuse成了一个kernel，实现为：

```python
cpp_fused_mul_0 = async_compile.cpp_pybinding(['const float*', 'const float*', 'float*'], '''
#include "/tmp/torchinductor_root/sk/cskh5dx62fglpphcrl6723dnmowdabouerrzy3dmqcngbxwfa7bv.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        #pragma omp simd simdlen(8) 
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(10L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_ptr0[static_cast<long>(x0)];
            auto tmp1 = in_ptr1[static_cast<long>(x0)];
            auto tmp2 = static_cast<float>(-1.0);
            auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
            auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
            out_ptr0[static_cast<long>(x0)] = tmp4;
        }
    }
}
''')
```


对于cuda程序来说，整体流程也是类似的。

上面展示了一个完整的阅读depfy生成的`torch.compile`编译产物的例子，希望对大家有帮助。




