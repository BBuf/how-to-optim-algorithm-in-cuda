# CuTe DSL 文档

# Overview

CUTLASS 4.x bridges the gap between productivity and performance for CUDA kernel development. By providing Python-based DSLs to the powerful CUTLASS C++ template library, it enables faster iteration, easier prototyping, and a gentler learning curve for high-performance linear algebra on NVIDIA GPUs.

Overall we envision CUTLASS DSLs as a family of domain-specific languages (DSLs). With the release of 4.0, we are releasing the first of these in CuTe DSL. This is a low level programming model that is fully consistent with CuTe C++ abstractions — exposing core concepts such as layouts, tensors, hardware atoms, and full control over the hardware thread and data hierarchy.

# Why CUTLASS DSLs?

While CUTLASS offers exceptional performance through its C++ template abstractions, the complexity can present challenges for many developers. CUTLASS 4.x addresses this by:

- **Simplifying metaprogramming**: Metaprogramming in Python is a lot more intuitive than with C++

- **Accelerating Iteration**: Rapid prototyping with familiar Python syntax and blazing fast compile times

- **Lowering Barriers**: Reduced learning curve for GPU programming concepts and consistency between CuTe C++ and DSL

- **Maintaining Performance**: Generated code leverages optimized CUTLASS primitives

Students can learn GPU programming concepts without the complexity of C++ templates. Researchers and performance engineers can rapidly explore algorithms, prototype, and tune kernels before moving to production implementations.

# Key Concepts and Approach

CUTLASS DSLs translate Python code into a custom intermediate representation (IR), which is then Just-In-Time (JIT) compiled into optimized CUDA kernels using MLIR and ptxas.

## Core CuTe DSL Abstractions

- **Layouts** – Describe how data is organized in memory and across threads.

- **Tensors** – Combine data pointers or iterators with layout metadata.

- **Atoms** – Represent fundamental hardware operations like matrix multiply-accumulate (MMA) or memory copy.

- **Tiled Operations** – Define how atoms are applied across thread blocks and warps (e.g., `TiledMma`, `TiledCopy`).

For more on CuTe abstractions, refer to the CuTe C++ library documentation(https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/00_quickstart.md).

**Pythonic Kernel Expression**

Developers express kernel logic, data movement, and computation using familiar Python syntax and control flow.

The DSLs simplify expressing loop tiling, threading strategies, and data transformations using concise Python code.

**JIT Compilation**

Python kernels are compiled at runtime into CUDA device code using MLIR infrastructure and NVIDIA’s `ptxas` toolchain, enabling rapid iteration and interactive debugging.

# Relationship to CUTLASS C++

CUTLASS DSLs are not a replacement for the CUTLASS C++ library or its 2.x and 3.x APIs. Instead, it aims to be a high-productivity kernel authoring framework that shares all concepts with CUTLASS 3.x C++ API such as CuTe, pipelines, schedulers etc.

- **Performance**: Generated kernels aim to match CUTLASS C++ kernels in performance; however, some performance gaps may exist due to missing optimizations that have been added over the years to CUTLASS C++ and may be missing in the DSLs examples.

- **Library**: The CUTLASS DSLs do not currently ship with a full GEMM/Conv autotuning profiler or library interface akin to CUTLASS C++. Instead, it focuses on generating and autotuning individual kernel instances (for example: via tile size exploration) and via native integration DL frameworks that support auto-tuning.

# Getting Started

- Quick Start Guide(https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html)- Initial setup and installation.

- CuTe DSL(https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html) – Overview of the typical development and workflow using CuTe DSL.

- CuTe DSL API(https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api.html) – Refer to the full API documentation.

- Limitations(https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/limitations.html) – Understand current CuTe DSL constraints and differences from C++.

- FAQs(https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/faqs.html) – Common questions and known issues.

# Current Status & Roadmap
CuTe DSL is in public beta and actively evolving. Interfaces and features are subject to change as we improve the system.

# Upcoming Milestones

- Public release targeted for **Summer 2025**
- Expanded support for additional data types and kernel types
- Usability improvements: better error messages, debugging tools, and streamlined APIs
- Broader integration of CUTLASS primitives and features

For known issues and workarounds, please consult the Limitations and FAQs.

# Community & Feedback

We welcome contributions and feedback from the developer community!

You can:

- Submit bug reports or feature requests via our GitHub Issues page(https://github.com/NVIDIA/cutlass/issues)
- Join the CUTLASS community on Discord(https://discord.com/channels/1019361803752456192/1150868614921064590) to ask questions and share ideas
- Contribute examples, tutorials, or enhancements to the DSLs
- Report unclear or missing documentation
- Propose support for additional data types or kernel variants
- Help prioritize roadmap features by upvoting GitHub issues

Thank you for helping shape the future of CUTLASS DSLs!

# Functionality

The CUTLASS DSL 4.0 release supports **Python 3.12** only. It shares the same driver requirements as the CUDA Toolkit 12.9(https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html). Specifically, the driver version must be 575.51.03 or later.

Currently, only Linux x86_64 is supported. Additional platform support will be added in future releases.

# Supported MMA Operations

**NVIDIA Ampere Architecture:**

- FP16 / BF16 tensor core instructions

**NVIDIA Hopper Architecture:**

- FP16 / BF16
- FP8

**NVIDIA Blackwell Architecture:**

- FP16 / BF16
- TF32
- I8
- F8

# Notable Limitations

For current constraints and unsupported features, refer to the Limitations(https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/limitations.html) section.

# Quick Start Guide

The CUTLASS DSL 4.0 release currently supports Linux and **Python 3.12** only. To install CUTLASS DSLs (limited to CuTe DSL for now), use the following command

## Installation

To ensure compatibility with the examples and code on GitHub, use the requirements.txt file from the corresponding commit in the repository.

```shell
git clone https://github.com/NVIDIA/cutlass.git
pip install -r cutlass/python/CuTeDSL/requirements.txt
```

If you just want to try out the last known stable release of the CUTLASS DSL (may not compatible with the latest examples and code), run:

```shell
pip install nvidia-cutlass-dsl
```

The `nvidia-cutlass-dsl` wheel includes everything needed to generate GPU kernels. It requires the same NVIDIA driver version as the CUDA Toolkit 12.9.

# Recommended Dependencies

To run examples and begin development, we recommend installing:
```shell
pip install torch jupyter
```

# Recommended Python environment variables for jupyter notebooks

We recommend setting the following environment variable when running jupyter notebooks.

```shell
export PYTHONUNBUFFERED=1
```

# CuTe DSL > Introduction

## Overview

CuTe DSL is a Python-based domain-specific language (DSL) designed for dynamic compilation of numeric and GPU-oriented code. Its primary goals are:

- **Consistent with CuTe C++**, allowing users to express GPU kernels with full control of the hardware.
- **JIT compilation** for both host and GPU execution.
- **DLPack integration**, enabling seamless interop with frameworks (e.g., PyTorch, JAX).
- **JIT caching**, so that repeated calls to the same function benefit from cached IR modules.
- **Native types and type inference** to reduce boilerplate and improve performance.
- **Optional lower-level control**, offering direct access to GPU backends or specialized IR dialects.

## Decorators

CuTe DSL provides two main Python decorators for generating optimized code via dynamic compilation:

1. `@jit` — Host-side JIT-compiled functions
2. `@kernel` — GPU kernel functions

Both decorators can optionally use a `preprocessor` that automatically expands Python control flow (loops, conditionals) into operations consumable by the underlying IR.

### `@jit`

Declares JIT-compiled functions that can be invoked from Python or from other CuTe DSL functions.

#### Decorator Parameters

- `preprocessor`:
  - `True` (default) — Automatically translate Python flow control (e.g., loops, if-statements) into IR operations.
  - `False` — No automatic expansion; Python flow control must be handled manually or avoided.

#### Call-site Parameters

- `no_cache`:
  - `True` — Disables JIT caching, forcing a fresh compilation each call.
  - `False` (default) — Enables caching for faster subsequent calls.


### `@kernel`

Defines GPU kernel functions, compiled as specialized GPU symbols through dynamic compilation.

#### Decorator Parameters:

- `preprocessor`:
    - `True` (default) — Automatically expands Python loops/ifs into GPU-compatible IR operations.
    - `False` — Expects manual or simplified kernel implementations.


#### Kernel Launch Parameters

- `grid`: Specifies the grid size as a list of integers.
- `block`: Specifies the block size as a list of integers.
- `cluster`: Specifies the cluster size as a list of integers.
- `smem`: Specifies the size of shared memory in bytes (integer).

## Calling Conventions

| Caller | Callee | Allowed | Compilation/Runtime |
| --- | --- | --- | --- |
| Python function | `@jit` | Yes | DSL runtime |
| Python function | `@kernel` | No | N/A (error raised) |
| `@jit` | `@jit` | Yes | Compile-time call, inlined |
| `@jit` | Python function | Yes | Compile-time call, inlined |
| `@jit` | `@kernel` | Yes | Dynamic call via GPU driver or runtime |
 | `@kernel` | `@jit` | Yes | Compile-time call, inlined |
 | `@kernel` | Python function | Yes | Compile-time call, inlined |
 | `@kernel` | `@kernel` | No | N/A (error raised) |
 
 

# End-to-End Code Generation

## 1. Techniques for Turning Python into intermediate representation (IR)

### 1.1 AST rewrite

The function’s abstract-syntax tree is analysed **before execution**. Python control-flow (`for`/`while`, `if`/`else`) and built-ins are converted to structured intermediate representation (IR) constructs. Computation inside each region is left untouched at this stage.

**Advantages**

- Sees the entire program, so every branch and loop is preserved.
- Keeps loop structure intact for optimization such as tiling, vectorisation or GPU thread mapping.

**Disadvantages**

- Requires a well-defined Python subset that the rewriter understands.

### 1.2 Tracing

The decorated function is executed once with proxy arguments; overloaded operators record every tensor operation that actually runs and produce a flat trace that is lowered to intermediate representation (IR).

**Advantages**

- Near-zero compile latency, ideal for straight-line arithmetic.
- No need to parse Python source, so it supports many dynamic Python features, and Python has many features.

**Disadvantages**

- Untaken branches vanish, so the generated kernel may be wrong for other inputs.
- Loops are flattened to the iteration count observed during tracing.
- Data-dependent control-flow freezes to a single execution path.

## 2. CuTe DSL Code-Generation Modes

CuTe’s Python front-end combines the techniques above into two mutually exclusive modes, selectable with the `preprocessor` flag of the `@jit` decorator:

1. Tracing mode `@jit(preprocess=False)` — tracing only. This results in the fastest compilation path and is recommended only for kernels that are guaranteed to be straight-line arithmetic. It suffers from all tracing limitations listed in the previous section.
2. Preprocessor mode (**default**) `@jit(preprocess=True)` — **AST rewrite + tracing**. The AST pass captures every loop and branch, eliminating the correctness and optimisation problems of pure tracing; tracing then fills in the arithmetic. This hybrid “preprocessor” pipeline is unique to CuTe DSL and was designed specifically to overcome the disadvantages identified above.


![](https://files.mdnice.com/user/59/680a3096-98f3-4989-b144-29443a86de27.jpg)

Figure 1: Left: tracing mode records only the path that executed. Right: preprocessor mode emits structured intermediate representation (IR) for every branch and loop before tracing the arithmetic.

## Why Tracing-Only Is Insufficient for Control-Flow

- **Branch loss** — The untaken side of an `if`/`else` is never lowered.
 - **Loop unrolling** — Loops are flattened to the iteration count observed, destroying structure needed for parallel mapping and tiling.
 - **Data-dependent paths** — Control-flow that depends on tensor values freezes to a single execution path at trace time.
 
 The preprocessor mode fixes all of these by lowering control-flow first and delegating only the arithmetic to the tracer.
 
 

# Control Flow

## Overview

CuTe DSL walks Python’s AST and converts each control-flow construct it finds into structured intermediate representation (IR). You can therefore write ordinary Python loops and branches while the compiler decides—statement by statement—whether to:

- **evaluate at compile time** if it’s a native Python control flow, or
- **emit intermediate representation (IR)** when the control flow is marked as dynamic.

Passing intermediate representation (IR) values to a native Python control flow will result in an error.

For a high-level discussion of the overall pipeline, see the code-generation overview(https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_code_generation.html).

## For Loops

CuTe DSL recognises three kinds of ranges for `for` loops:

- `range` — the Python built-in, always lowered to intermediate representation (IR)
- `cutlass.range` — Same as Python built-in `range`, but supports advanced unrolling and pipelining control
- `cutlass.range_constexpr` — unrolled at compile time

### `range(...)` / `cutlass.range(...)`

Use when you always want a loop in the generated intermediate representation (IR), even if the inputs are Python values.

### `cutlass.range_constexpr(...)`

Runs in the Python interpreter and is fully unrolled before code generation. All loop indices must be `Constexpr` (compile-time Python value).

Example:

```python
@cute.jit
def control_flow_examples(bound: cutlass.Int32):
    n = 10

    # ✅ This loop is Python loop, evaluated at compile time.
    for i in cutlass.range_constexpr(n):
        cute.printf("%d\\n", i)

    # ✅ This loop is dynamic, even when bound is Python value.
    for i in range(n):
        cute.printf("%d\\n", i)

    # ❌ This loop bound is a dynamic value, not allowed in Python loop.
    # Should use `range` instead.
    for i in cutlass.range_constexpr(bound):
        cute.printf("%d\\n", i)

    # ✅ This loop is dynamic, emitted IR loop.
    for i in range(bound):
        cute.printf("%d\\n", i)

    # ✅ This loop is dynamic, emitted IR loop with unrolling
    for i in cutlass.range(bound, unroll=2):
        cute.printf("%d\\n", i)
```

## Software Pipelining

Software pipelining is a technique used to optimize loops. Typically, this involves writing a prefetch loop and a main loop.

```python
@cute.jit
def example():
    ...
    # build a circular buffer
    buffer = ...

    # prefetch loop
    for i in range(prefetch_stages):
        cute.copy(atom, gmem[i], buffer[i], ...)

    # main loop
    for i in range(bound):
        if i + prefetch_stages < bound:
            cute.copy(atom, gmem[i + prefetch_stages], buffer[(i + prefetch_stages) % total_stages], ...)

        use(buffer[i % total_stages])

    ...
```

This can be tedious to write and tune. CuTe DSL provides a loop attribute to ask the compiler to do this.

```python
@cute.jit
def example():
    ...
    # build a circular buffer
    buffer = ...

    for i in cutlass.range(bound, prefetch_stages=prefetch_stages):
        # Compiler automatically handles the pipelining:
        # - Generates prefetch loop for initial stages
        # - In main loop, prefetches future data while using current data
        cute.copy(atom, gmem[i], buffer[i % total_stages], ...)
        use(buffer[i % total_stages])  # Uses data from previous iterations

    ...
```

Compiler will automatically generate the prefetch loop with `prefetch_stages` iterations and a corresponding main loop.

This feature is experimental and only supported on sm90 and above.

## If-Else Statements

Standard Python `if`/`elif`/`else` is supported.

- **Predicate without annotation** → lowered to intermediate representation (IR).
- **Predicate annotated with `cutlass.const_expr`** → evaluated at compile time.

Example:

```python
@cute.jit
def main(const_var: cutlass.Constexpr, dynamic_var: cutlass.Int32):
    # ✅ This branch is Python branch, evaluated at compile time.
    if cutlass.const_expr(const_var):
        cute.printf("Const branch\n")
    else:
        cute.printf("Const else\n")

    # ✅ This branch is dynamic branch, emitted IR branch.
    if dynamic_var == 10:
        cute.printf("Dynamic True\n")
    else:
        cute.printf("Dynamic False\n")

    # ❌ Using a dynamic value with `cutlass.const_expr` is not allowed.
    if cutlass.const_expr(dynamic_var == 10):
        cute.printf("Bound is 10\n")
```

## While Loops

Standard Python `while` is supported.

- **Condition without annotation** → lowered to intermediate representation (IR).
- **Condition annotated with `cutlass.const_expr`** → evaluated at compile time.

Example:

```python
@cute.jit
def main(dynamic_var: cutlass.Int32):
    n = 0

    # ✅ This is Python while loop, evaluated at compile time.
    while cutlass.const_expr(n < 10):
        cute.printf("Const branch\n")
        n += 1

    # ✅ This is dynamic while loop, emitted IR while loop.
    while dynamic_var == 10:
        cute.printf("Dynamic True\n")
        n += 1

    # ❌ Using a dynamic value with `cutlass.const_expr` is not allowed.
    while cutlass.const_expr(n < dynamic_var):
        n += 1
```

## Compile-Time Metaprogramming

Mix compile-time constructs with normal CuTe DSL code to generate specialised kernels without runtime overhead. A compile-time flag can, for example, toggle an optional ReLU epilogue:

```python
@cute.kernel
def gemm(..., do_relu: cutlass.Constexpr):
    # main GEMM work
    ...

    if cutlass.const_expr(do_relu):  # compile-time guard
        # ReLU code is emitted only when do_relu is True
        ...
```

```python
gemm(..., False)  # ReLU is omitted from the generated |IR|
gemm(..., True)   # ReLU is included
```

## Limitations of Dynamic Control Flow

- Early-exit `break`, `continue`, `pass` or raising exception from control flow body are not yet supported.
- Operations in the control flow body are traced only when tracing is active in that region.
- Values originating in control flow body are not available outside the control flow.
- Changing type of a variable in control flow body is not allowed.

**Example**:

```python
@cute.jit
def control_flow_negative_examples(predicate: cutlass.Boolean):
    n = 10

    # ❌ This loop is dynamic, early-exit isn't allowed.
    for i in range(n):
        if i == 5:
            break  # Early-exit

    if predicate:
        val = 10

    # ❌ return from control flow body is not allowed.
    return

    # ❌ Raising exception from control flow body is not allowed.
    raise ValueError("This is not allowed")

    # ❌ Using pass in control flow body is not allowed.
    pass

    # ❌ val is not available outside the dynamic if
    cute.printf("%d\n", val)

    if predicate:
        # ❌ Changing type of a variable in control flow body is not allowed.
        n = 10.0
```

# JIT Function Argument Generation

## In a nutshell

When using the `@jit` or `@kernel` decorators to define a JIT-compiled function, the arguments to the function are traced to determine the JIT function’s signature. CuTe DSL provides a Pythonic way to write the arguments for JIT function as one normally would in Python, and the CuTe DSL will take care of the rest for you.

Specifically, CuTe DSL honors the following when generating the JIT function’s arguments:

- JIT function arguments are assumed to be **dynamic arguments** by default.
- If an argument is explicitly type annotated with `cutlass.Constexpr`, it is treated as a **compile-time constant**.
- If type annotation is provided, CuTe DSL validates the argument type at compile time for **type safety**.
- CuTe DSL provides runtime checkable protocols (`JitArgument` and `DynamicExpression`) for generating JIT function arguments for customized types.

More details below for each of the above.

## Static argument vs. Dynamic argument

CuTe DSL supports both static and dynamic arguments for JIT functions.

1. **Static arguments** hold values that are known at compile time. It is not included in the generated JIT function signature.
2. **Dynamic arguments** hold values that are only known at runtime.

By default, CuTe DSL assumes dynamic arguments and tries to infer the argument types from the call-site argument types. An explicit type annotation `cutlass.Constexpr` can be used to specify a static argument.

Example:

```python
import cutlass
import cutlass.cute as cute


@cute.jit
def foo(x: cutlass.Int32, y: cutlass.Constexpr):
    print("x = ", x)      # Prints x = ?
    print("y = ", y)      # Prints y = 2
    cute.printf("x: {}", x)  # Prints x: 2
    cute.printf("y: {}", y)  # Prints y: 2


foo(2, 2)
```

In the example above, `x` is a dynamic argument with type `cutlass.Int32` and `y` is a static argument.

With the `cutlass.Constexpr` annotation, a more sophisticated use case of static argument in the JIT kernels can be something like:

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
    # Perform epilogue op on accumulator and convert to C type
    acc_vec = tTR_rAcc.load()
    acc_vec = epilogue_op(acc_vec.to(self.c_dtype))
    tTR_rC.store(acc_vec)
```

In this example, `epilogue_op` is a static argument in the JIT kernel where the argument is used for the epilogue fusion. Upon calling the kernel, an elementwise lambda function can be passed in as the `epilogue_op` argument. For example, a ReLU can be applied for epilogue fusion by simply setting the `epilogue_op` to `lambda x: cute.where(x > 0, x, cute.full_like(x, 0))`

Refer to the Blackwell dense GEMM example(github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell/dense_gemm_persistent.py
) for a complete example.

## Type safety

CuTe DSL makes good use of type annotation in JIT function signature and validates the JIT function argument types at compile time for **type safety**.

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
foo(b, a)  # This will fail at compile time due to type mismatch
```

The type safety check helps catch the type mismatch issue early at the compile time with clear error message to avoid tricky runtime errors which is usually more expensive to debug. In the example above, the second call to `foo` will fail at compile time due to the type mismatch with a clear error message:

```python
cutlass.base_dsl.common.DSLRuntimeError: DSLRuntimeError: expects argument #1 (a) to be <class 'cutlass.cute.typing.Tensor'>, but got <class 'int'>
```

## JIT function arguments with customized types

CuTe DSL supports customized types for JIT function arguments by providing two runtime checkable protocols:

- `JitArgument` which is used for host JIT functions to be called from Python.
  - `__c_pointers__`: Generate a list of ctypes pointers for the current object.
  - `__get_mlir_types__`: Generate a list of MLIR types for the current object.
  - `__new_from_mlir_values__`: Create a new object from MLIR values.
- `DynamicExpression` which is used for device JIT functions to be called from the host JIT functions.
  - `__extract_mlir_values__`: Generate a dynamic expression for the current object.
  - `__new_from_mlir_values__`: Create a new object from MLIR values.

Refer to `typing.py`(https://github.com/NVIDIA/cutlass/tree/main/python/CuTeDSL/base_dsl/typing.py) for more details on these protocol APIs.

Depending on different cases of the customized types, CuTe DSL provides easy ways to adopt customized types for JIT function arguments.

### 1. Direct protocol implementation in customized types

One way is to implement the protocol methods directly in the customized types to enable the protocol based JIT function argument generation.

```python
import cutlass
import cutlass.cute as cute


# Customized type that implements the DynamicExpression protocol
class MyDynamicExpression:
    def __init__(self, tensor, offset):
        self._tensor = tensor  # Dynamic argument
        self._offset = offset  # Dynamic argument

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

In the example above, `MyDynamicExpression` implements the `DynamicExpression` protocol and CuTe DSL will generate the JIT function arguments for the JIT kernel `my_kernel` based on the protocol methods.

### 2. Adaptor based protocol implementation for customized types

For the case where directly changing the customized types to implement the protocol is not feasible, CuTe DSL provides adaptor based approach to adapt the customized types for JIT function argument generation.

The JIT function argument adaptor is a callable object that implements the desired protocol methods for the registered customized types. This way, CuTe DSL automatically queries the JIT argument adaptor registry to generate the JIT function arguments for the given customized types.

```python
@cutlass.register_jit_arg_adapter(MyFrameworkObject)
class MyFrameworkObjectAdapter:
    """
    Convert a 3rd party framework object to a JIT function argument with JitArgument protocol
    """

    def __init__(self, arg):
        self._arg = arg

    def __c_pointers__(self):
        # Convert the framework object to a C-ABI compatible object
        # thru its C-ABI interface
        return [self._arg.get_cabi_pointer()]

    def __get_mlir_types__(self):
        # Return the list of MLIR types the framework object represents
        return [self._arg.get_data().mlir_type]

    def __new_from_mlir_values__(self, values):
        # Convert the MLIR values back to the framework object
        return MyFrameworkObject(values[0])
```

In this example, the `MyFrameworkObjectAdapter` implements an adaptor class which bridges the CuTe DSL and the 3rd party framework type `MyFrameworkObject`. The registration is done by just decorating the adaptor with `cutlass.register_jit_arg_adapter` for the customized type. With the registered adaptor, CuTe DSL will automatically use the adaptor to generate the JIT function arguments for `MyFrameworkObject` typed arguments.

# Static vs Dynamic layouts

## Static Layout

When integrating with popular deep learning frameworks, one question is how to deal with the layout of the converted `cute.Tensor`. For example, when converting a `torch.Tensor` to a `cute.Tensor`, the shape of the `torch.Tensor` is honored for the layout of `cute.Tensor`.

```python
import torch
import cutlass
from cutlass.cute.runtime import from_dlpack

@cute.jit
def foo(tensor):
    print(f"tensor.layout: {tensor.layout}")  # Prints tensor layout at compile time
    cute.printf("tensor: {}", tensor)         # Prints tensor values at runtime
```

In this example, we define a JIT function `foo` that takes a `cute.Tensor` as input and prints its layout. Note that Python print is used to print the layout at compile time. This works fine for static layout whose value is known at compile time.

Now let’s try to run the JIT function `foo` with different shapes of the input `torch.Tensor`.

```python
a = torch.tensor([1, 2, 3], dtype=torch.uint16)
a_pack = from_dlpack(a)
compiled_func = cute.compile(foo, a_pack)
compiled_func(a_pack)
```

Here we first convert a 1D `torch.Tensor` with 3 elements to a `cute.Tensor` using `from_dlpack`. Then we compile the JIT function `foo` with the converted `cute.Tensor` and call the compiled function.

```python
tensor.layout: (3):(1)
tensor: raw_ptr(0x00000000079e5100: i16, generic, align<2>) o (3):(1) = (1, 2, 3)
```

It prints `(3):(1)` for the layout because the converted `cute.Tensor` has a static layout with shape `(3)` which is the shape of the `a`.

Now if we call the compiled function with a different shape of the input `torch.Tensor`, it would result in an unexpected result at runtime due to the mismatch of the type since `compiled_func` expects a `cute.Tensor` with layout `(3):(1)` while `b` has shape `(5)`.

```python
b = torch.tensor([11, 12, 13, 14, 15], dtype=torch.uint16)
b_pack = from_dlpack(b)
compiled_func(b_pack)  # ❌ This results in an unexpected result at runtime due to type mismatch
```

Following is the output which is unexpected due to the type mismatch.

```python
tensor: raw_ptr(0x00000000344804c0: i16, generic, align<2>) o (3):(1) =
(11, 12, 13)
```

To fix that, we would have to trigger another code generation and compilation for the new shape for `b`.

```python
compiled_func_2 = cute.compile(foo, b_pack)  # This would trigger another compilation
compiled_func_2(b_pack)                      # ✅ Now this works fine
```

As shown in the example above, with the newly compiled `compiled_func_2`, we can pass in b_pack to the compiled JIT function `compiled_func_2`.

```python
tensor.layout: (5):(1)
tensor: raw_ptr(0x0000000034bb2840:: i16, generic, align<2>) o (5):(1) =
(11, 12, 13, 14, 15)
```

Now it recompiles and prints the values of `b` correctly.

It’s obvoius that we need distinct codes generated and compiled for different static layout. In this case, one for layout `(3):(1)` and the other for layout `(5):(1)`.

## Dynamic Layout

In order to avoid generating and compiling multiple times for different shapes of the input `torch.Tensor`, CuTe DSL provides a way to generate and compile JIT function with dynamic layout.

To get dynamic layout of the `cute.Tensor`, a `torch.Tensor` object can be passed into the JIT function directly which instructs CuTe DSL to call `cute.mark_layout_dynamic` automatically on the converted `cute.Tensor` per the leading dimension of the layout.

```python
import torch
import cutlass
from cutlass.cute.runtime import from_dlpack

@cute.jit
def foo(tensor):
    print(tensor.layout)  # Prints (?,?):(?,1) for dynamic layout

a = torch.tensor([[1, 2], [3, 4]], dtype=torch.uint16)
compiled_func = cute.compile(foo, a)
compiled_func(a)

b = torch.tensor([[11, 12], [13, 14], [15, 16]], dtype=torch.uint16)
compiled_func(b)  # Reuse the same compiled function for different shape
```

In the example above, a single compilation of the JIT function `foo` is reused for different shapes of the input `torch.Tensor`. This is possible because the converted `cute.Tensor` has a dynamic layout `(?,?):(?,1)` which is compatible with the shape of the input `torch.Tensor` of both calls.

Alternatively, for compact layout, `cute.mark_compact_shape_dynamic` can be called for a finer-grained control to specify the mode of the layout for dynamic and the divisibility constraint for the dynamic dimension.

Refer to Integration with Frameworks(https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/framework_integration.html) for more details on `from_dlpack`, `mark_layout_dynamic`, and `mark_compact_shape_dynamic`.

## Static Layout vs. Dynamic Layout

Per the previous sections, we have seen that static layout leads to distinct JIT code generations while dynamic layout leads to a single compilation for different shapes.

That said, creating JIT function with static layout is useful when the use cases target input data with fixed shapes. Since more information is available at compile time, the compiler can kick in optimizations that otherwise would not be possible for code generated for dynamic layout.

On the other hand, dynamic layout is more flexible for cases where input data has varying shapes. This provides more scalability for the generated code to deal with varying input data of different shapes.

## Programming with Static and Dynamic Layout

CuTe DSL provides intuitive way to program with static and dynamic layout in the code.

```python
import torch
import cutlass
from cutlass.cute.runtime import from_dlpack

@cute.jit
def foo(tensor, x: cutlass.Constexpr[int]):
    print(cute.size(tensor))  # Prints 3 for the 1st call
                              # Prints ? for the 2nd call
    if cute.size(tensor) > x:
        cute.printf("tensor[2]: {}", tensor[2])
    else:
        cute.printf("tensor size <= {}", x)

a = torch.tensor([1, 2, 3], dtype=torch.uint16)
foo(from_dlpack(a), 3)   # First call with static layout

b = torch.tensor([1, 2, 3, 4, 5], dtype=torch.uint16)
foo(b, 3)                # Second call with dynamic layout
```

In this example, the JIT function `foo` is compiled with a static layout `(3):(1)` for the first call, which means the size of the tensor is known at compile time. CuTe DSL makes good use of this and automatically handles the if condition at the compile time. Hence the generated codes are efficient without the if condition at all.

For the second call, the JIT function `foo` is compiled with a dynamic layout `(?):(1)` hence the tensor size is only evaluated at runtime. CuTe DSL automatically generates the code to handle the dynamic layout and the if condition at runtime.

The same applies to loop as well:

```python
@cute.jit
def foo(tensor, x: cutlass.Constexpr[int]):
    for i in range(cute.size(tensor)):
        cute.printf("tensor[{}]: {}", i, tensor[i])

a = torch.tensor([1, 2, 3], dtype=torch.uint16)
foo(from_dlpack(a), 3)   # First call with static layout

b = torch.tensor([1, 2, 3, 4, 5], dtype=torch.uint16)
foo(b, 3)                # Second call with dynamic layout
```

With the static layout in the first call, CuTe DSL is able to fully unroll the loop at compile time. While in the second call, the generated codes will have the loop executed at runtime based on the dynamic layout.

With the single JIT function implementation, CuTe DSL is able to handle control-flow constructs and automatically generate the optimized codes for different cases. This is all possible because CuTe DSL is able to walk the Python AST and convert each control-flow construct it finds accordingly.

Please refer to Control Flow(https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_control_flow.html) for more details.

# JIT Caching

## Zero Compile and JIT Executor

Zero Compile is a feature that enables explicit kernel compilation on demand through `cute.compile`. When cute.compile is called, it compiles the kernel and returns a JIT Executor instance. This JIT Executor instance can be cached and reused directly for subsequent executions without compiling the kernel again.

The JIT Executor is a component that independently executes compiled code. It can be created either through cute.compile or implicit compilation. The JIT Executor instance behaves like a callable object to execute the compiled code. Each JIT Executor instance maintains a single compiled host function.

It encompasses all necessary execution components:

- Host function pointer and its MLIR execution engine
- CUDA modules (optional)
- Argument specifications defining how Python arguments are converted to C ABI-compatible types. Note that arguments with the `cutlass.Constexpr` hint are excluded from argument specifications since they are evaluated at compile time rather than runtime.

For example, in the following code, `print_result` is a `cutlass.Constexpr` value that is **NOT** evaluated at runtime:

```python
import cutlass.cute as cute

@cute.jit
def add(a, b, print_result: cutlass.Constexpr):
   if print_result:
      cute.printf("Result: %d\n", a + b)
   return a + b

jit_executor = cute.compile(add, 1, 2, True)

jit_executor(1, 2) # output: ``Result: 3``
```

The JIT Executor ensures all components are properly initialized and loaded after compilation. For example, all CUDA modules are loaded (via `cuModuleLoad`) and kernel function pointers are extracted (via `cuModuleGetFunction`).

When calling a JIT Executor instance, it:

- Parses Python runtime arguments and converts them to C ABI-compatible types according to argument specifications
- Invokes the host function with the converted arguments

## Custom Caching with `cute.compile`

`cute.compile` bypasses caching in CuTe DSL and always performs compilation, returning a fixed JIT Executor instance. This allows implementing custom caching strategies.

Example:

```python
@cute.jit
def add(b):
   return a + b

# Define a custom cache
custom_cache = {}

a = 1
compiled_add_1 = cute.compile(add, 2)
custom_cache[1] = compiled_add_1
compiled_add_1(2) # result = 3

a = 2
compiled_add_2 = cute.compile(add, 2)
custom_cache[2] = compiled_add_2
compiled_add_2(2) # result = 4

# Use the custom cache
custom_cache[1](2) # result = 3
custom_cache[2](2) # result = 4
```

## Cache in CuTe DSL

By default, cache in CuTe DSL is implicitly enabled to avoid recompilation when kernels are called repeatedly without changes.

The cache is implemented as a map storing compiled JIT Executor instances within CuTe DSL.

The cache key combines hashes of:

- MLIR bytecode of the MLIR program generated by CuTe DSL
- All CuTe DSL Python source files
- All CuTe DSL shared libraries
- All CuTe DSL environment variables

The cache value is a compiled JIT Executor instance.

On a cache hit, compilation is skipped and the cached JIT Executor instance is reused.

On a cache miss, the kernel is compiled and the new JIT Executor instance is stored in the cache.

Here is an example demonstrating automatic caching of the `add` kernel:

```python
# Global variable
a = 1

@cute.jit
def add(b):
   return a + b

# Cache is empty at beginning

# First call: cache miss triggers compilation
result = add(2) # result = 3
# Cache now has one instance

# Second call: cache hit reuses cached JIT Executor
result = add(2) # result = 3

a = 2
# Third call: cache miss due to changed IR code triggers recompilation
result = add(2) # result = 4
# Cache now has two instances
```

The cache can be serialized to files for subsequent runs. After serialization, compiled MLIR bytecode is stored in file. The cache directory is `/tmp/{current_user}/cutlass_python_cache`. During compilation, the cache loads the corresponding kernel from file (if it exists) into memory as needed, and after compilation, it saves any newly compiled executables back to file.

Note that for efficiency, the default cache directory is located in a temporary folder. However, this location is not persistent, it may be cleared by the system (for example, during a reboot or disk space cleanup). If you wish to preserve the cache across sessions, set the `CUTE_DSL_CACHE_DIR` environment variable to point to a persistent directory.

The following environment variables control file caching:

```shell
# Disable file caching while keeping in-memory cache available, defaults to False.
export CUTE_DSL_DISABLE_FILE_CACHING=True

# Cache directory, defaults to /tmp/{current_user}/cutlass_python_cache.
export CUTE_DSL_CACHE_DIR=/home/user/local_cutlass_python_cache/dense_gemm_cache/
```

## Limitations

The intention of caching is to reduce the host launch overhead before each execution. As above example shows, the consistency between the original Python code and the MLIR program is hard to maintain because of the impact of dynamic factors such as global variables. Therefore, the MLIR program **MUST** always be generated to verify that the kernel content matches what was previously built.

For optimal host launch latency, we recommend using above custom caching method with `cute.compile`.

# JIT Compilation Options

## JIT Compilation Options Overview

When compiling a JIT function using CuTe DSL, you may want to control various aspects of the compilation process, such as optimization level, or debugging flags. CuTe DSL provides a flexible interface for specifying these compilation options when invoking `cute.compile`.

Compilation options allow you to customize how your JIT-compiled functions are built and executed. This can be useful for:

- Enabling or disabling specific compiler optimizations
- Generating debug information for troubleshooting

These options can be passed as keyword arguments to `cute.compile` or set globally for all JIT compilations. The available options and their effects are described in the following sections, along with usage examples to help you get started.

The CuTe DSL provides multiple ways to specify compilation options - either by specifying additional arguments to `cute.compile` or by using a more Pythonic approach with separate Python types for `cute.compile`.

## `cute.compile` Compilation Options as strings

You can provide additional compilation options as a string when calling `cute.compile`. The CuTe DSL uses `argparse` to parse these options and will raise an error if any invalid options are specified.

### Options

| Option | Description | Default | Type |
| --- | --- | --- | --- |
| `opt-level` | Optimization level of compilation. The higher the level, the more optimizations are applied. The valid value range is `[0, 3]`. | `3` (highest level of optimization) | `int` |
| `enable-assertions` | Enable host and device code assertions. | `False` | `bool` |
| `keep-cubin` | Keep the generated CUBIN file. | `False` | `bool` |
| `keep-ptx` | Keep the generated PTX file. | `False` | `bool` |
| `ptxas-options` | The options to pass to the PTX Compiler library. | `""` | `str` |
| `generate-line-info` | Generate line information for debugging. | `False` | `bool` |
| `gpu-arch` | The GPU architecture to compile for. | `""` | `str` |
| `enable-tvm-ffi` | Enable Apache TVM FFI. | `False` | `bool` |

You can use the following code to specify compilation options:

```python
jit_executor_with_opt_level_2 = cute.compile(add, 1, 2, options="--opt-level 2")
jit_executor_with_opt_level_1 = cute.compile(add, 1, 2, options="--opt-level 1")
jit_executor_with_enable_device_assertions = cute.compile(add, 1, 2, options="--enable-assertions")
jit_executor_with_keep_cubin = cute.compile(add, 1, 2, options="--keep-cubin")
jit_executor_with_keep_ptx = cute.compile(add, 1, 2, options="--keep-ptx")
jit_executor_with_ptxas_options = cute.compile(add, 1, 2, options="--ptxas-options '--opt-level=2'")
```

## `cute.compile` Compilation Options as separate Python types

Alternatively, you can also use a more Pythonic way to specify compilation options with separate Python types. Compilation options can be programmatically composed using tuple and passed to `cute.compile` separately.

```python
from cutlass.cute import OptLevel, EnableAssertions, GenerateLineInfo, KeepCUBIN, KeepPTX

my_debugging_options = (OptLevel(1), EnableAssertions, GenerateLineInfo, KeepCUBIN, KeepPTX)
compiled_kernel_1 = cute.compile[my_debugging_options](my_kernel_1, ...)
compiled_kernel_2 = cute.compile[my_debugging_options](my_kernel_2, ...)
```

This approach causes invalid options to raise errors immediately, making it much easier to detect typos when specifying multiple options. Notebly, boolean options are automatically converted to `True` instances of the option type for convenience.

```python
jit_executor_with_opt_level_2 = cute.compile[OptLevel(2)](add, 1, 2)
jit_executor_with_opt_level_1 = cute.compile[OptLevel(1)](add, 1, 2)
jit_executor_with_enable_device_assertions = cute.compile[EnableAssertions](add, 1, 2)
jit_executor_with_keep_cubin = cute.compile[KeepCUBIN](add, 1, 2)
jit_executor_with_keep_ptx = cute.compile[KeepPTX](add, 1, 2)
jit_executor_with_ptxas_options = cute.compile[PtxasOptions("--opt-level=2")](add, 1, 2)
```

 # Integration with Frameworks
 
In order to facilitate the integration of CUTLASS Python with popular frameworks, we leverage the DLPack protocol(https://github.com/dmlc/dlpack) and transform tensors originating from these frameworks to CuTe tensors. The present page documents the conventions, the API available to the user, and provide example code snippets for common usage patterns. We also provide a section on how to bypass the DLPack protocol and directly call the JIT function.

## Implicit Conversion

Tensors originating from frameworks supporting the DLPack protocol can be directly provided to a JIT function as a regular parameter. CuTe DSL’s runtime implicitly converts the original tensor to a CuTe tensor with a fully dynamic layout except for the stride element corresponding to the leading dimension. The example below demonstrates this use case.

```python
import torch
import cutlass.cute as cute


@cute.jit
def foo(src):
    """
    The following lines print

    ptr<f32, generic> o (?,?,?):(?,?,1)
    <class 'cutlass.cute.core._Tensor'>
    """
    print(src)
    print(type(src))


a = torch.randn(30, 20, 32, device="cpu")
foo(a)
```

## Explicit conversion using `from_dlpack`

CuTe DSL’s runtime provides an interface for converting DLPack-compatible tensors to CuTe tensors,

```python
b = cute.runtime.from_dlpack(a)
```

where `a` is a tensor supporting the DLPack protocol with the `__dlpack__` and `__dlpack_device__` methods. The resulting CuTe tensor `b` has a fully static layout. This conversion is performed without copying any tensor data, enabling seamless integration with major frameworks. Users can create tensors using NumPy, PyTorch, etc. and directly feed them into JIT functions writtnen using CuTe DSL.

The resulting CuTe tensor shares the same underlying memory buffer as the original tensor. This zero-copy approach maximizes performance by eliminating unnecessary data duplication. However, it is important to note that the CuTe tensor’s validity is tied to the lifetime of the original tensor. If the source tensor is destroyed or goes out of scope, the corresponding CuTe tensor becomes invalid since it references the original memory location.

The full signature of from_dlpack is as follows:

```python
def from_dlpack(tensor, assumed_align=None, use_32bit_stride=False):
    ...
```

The `assumed_align` integer parameter specifies the alignment of the tensor in unit of bytes. The tensor’s base address must be divisible by `assumed_align`. When not provided explicitly, the alignment is set to the natural alignment of the tensor’s element type. Note that the alignment information is part of the pointer type in the generated IR. Therefore, programs with different alignments have a different IR and identical IRs are required for hitting the kernel caching mechanism of CuTe DSL.

The `use_32bit_stride` parameter determines whether to use 32-bit stride for the tensor’s dynamic stride values. By default, it is set to `False` (64bit) to ensure that address calculations do not risk overflow. For smaller problem sizes (where `cosize(layout_of_tensor) <= Int32_MAX`), users may set it to `True` (32bit) to improve performance by reducing register usage and the number of address calculation instructions. When `use_32bit_stride` is set to `True`, a runtime check is performed to ensure that the layout does not overflow. Please note that this parameter only has an effect when the tensor’s layout is marked as dynamic.

### Code Example

The following code demonstrates how to convert a PyTorch tensor to a CuTe tensor using the `from_dlpack` function with default parameters.

```python
import torch
import cutlass
from cutlass.cute.runtime import from_dlpack

x = torch.randn(30, 20, device="cpu")
y = from_dlpack(x)
```

Once converted, we can access the tensor’s information through various attributes. The following list shows the attributes of the converted tensor:
- `tensor.shape`: the tensor’s shape
- `tensor.stride`: the tensor’s stride
- `tensor.memspace`: the tensor’s memory space
- `tensor.element_type`: the tensor’s element data type

```python
import torch
import cutlass
from cutlass.cute.runtime import from_dlpack

x = torch.randn(30, 20, device="cpu")
y = from_dlpack(x)

print(y.shape)        # (30, 20)
print(y.stride)       # (20, 1)
print(y.memspace)     # generic (if torch tensor in on device memory, memspace will be gmem)
print(y.element_type) # Float32
print(y)              # Tensor<0x000000000875f580@generic o (30, 20):(20, 1)>
```

The string format of the resulting CuTe tensor is

```python
Tensor<0x{tensor.data_ptr:016x}@{tensor.memspace} o {tensor.shape}:{tensor.stride}>
```

As can be seen in the example above, `from_dlpack` first results in a tensor with a static layout. To obtain dynamic or mixed static/dynamic layouts after calling `from_dlpack`, the `mark_layout_dynamic` and `mark_compact_shape_dynamic` functions are used and described in the following sections.

## When to Use Explicit Conversion?

The DLPack protocol is a widely used protocol for interoperability between different frameworks. However, there is some associated overhead. Based on our benchmark, it usually takes between 2 to 3 us per call to `from_dlpack`.

Explicit conversion allows for caching the converted CuTe tensors in order to avoid the overhead of repeated calls to `from_dlpack`.

```python
x = torch.randn(30, 20, device="cpu")
if key not in cached_tensors:
    # Do the conversion only for cache misses
    cached_tensors[key] = cute.runtime.from_dlpack(x)
foo(cached_tensors[key])
```

Another use case for explicit conversion is to gain fine-grain control over which modes of a tensor are considered dynamic from the perspective of the generated program.

## Mark the Tensor’s Layout as Dynamic with `mark_layout_dynamic`

After calling this function, all shape modes become dynamic. The stride modes also become dynamic with the following two exceptions:

- the leading dimension’s stride remains fixed at 1;
- stride elements equal to 0 (which indicates broadcasting) are retained.

The full signature of `mark_layout_dynamic` is as follows:

```python
def mark_layout_dynamic(self, leading_dim: int|None = None):
```

The `leading_dim` parameter specifies the leading dimension of the tensor. The leading dimension’s stride is set to 1 unless inconsistent with the layout of the DLPack tensor. For example,

- For a tensor with layout `(2,2,3,4):(2,1,4,12)`, if `leading_dim` is specified to be 1, the layout will be marked as `(?,?,?,?):(?,1,?,?)`.
- If `leading_dim` is specified to be 0, a deduction failure error is raised because the stride of dimension 0 is 2 (not 1).

The default value for `leading_dim` is `None`. In such case, the system automatically deduces it from the tensor’s layout using the following logic:

- If a dimension’s stride is 1, that dimension is marked as the leading dimension.
- If multiple dimensions satisfy condition 1, an error is thrown indicating deduction failure. Note that after converting a **PyTorch** tensor to the DLPack format, the stride for dimensions with size 1 are canonicalized to 1. This canonicalization can increase the likelihood of deduction failures. This behavior is specific to PyTorch and does not occur with NumPy for example.
- If no dimension satisfies condition 1, all strides are marked as dynamic.

For example:

- For a tensor with layout `(2,2,3,4):(2,1,4,12)`, the leading dimension is 1. The layout will be marked as `(?,?,?,?):(?,1,?,?)`.
- For a tensor with layout `(1,5,1):(1,1,1)`, if leading_dim is not specified, a deduction failure error is raised.
- For a tensor with layout `(2,2):(8,2)`, since no dimension has stride 1, all dimensions are marked as dynamic: `(?,?):(?,?)`.

The leading dimension accepts negative index which means the dimension is counted from the last dimension. For example,

- For a tensor with layout `(2,2,3,4):(2,1,4,12)`, if leading_dim is specified to be -1, the layout will be marked as `(?,?,?,?):(?,?,?,1)`.


### Code Example

The following example demonstrates how to use `mark_layout_dynamic` to specify dynamic tensor layouts.

- `t0` shows the usage of `mark_layout_dynamic` with unspecified `leading_dim` and the automatic deduction of leading dimension.
- `t1` & `t2` shows the usage of `mark_layout_dynamic` with specified `leading_dim`.
- `t3` shows the usage of `mark_layout_dynamic` with no leading dimension.
- `t4` shows the usage of `mark_layout_dynamic` with broadcasted dimensions.
- `t5` demonstrates the deduction failure when the there’re more than one dimensions with stride equals to 1.
- `t6` & `t7` demonstrates incorrect settings for leading_dim and expected errors.


```python
import torch
from cutlass.cute.runtime import from_dlpack

# (8,4,16,2):(2,16,64,1)
a = torch.empty(16, 4, 8, 2).permute(2, 1, 0, 3)
# (1,4,1,32,1):(4,1,4,4,4) => torch tensor when dimension has shape 1, its stride is degenerated to 1,
# resulting in (1,4,1,32,1):(1,1,1,4,1)
b = torch.empty(32, 1, 1, 1, 4).permute(3, 4, 1, 0, 2)
# (2,2):(8,2)
c = torch.empty(3, 4)[::2, ::2]
# (3,1,1,5):(5,0,0,1)
d = torch.empty(3, 1, 1, 5).expand(3, 4, 2, 5)

# auto deduce the leading dimension to be 3
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
# Can't decude the leading dimension from layout, please specify the leading_dim explicitly.

t6 = from_dlpack(a).mark_layout_dynamic(leading_dim=1)
# Expected strides[leading_dim] == 1, but got 16

t7 = from_dlpack(b).mark_layout_dynamic(leading_dim=3)
# Expected strides[leading_dim] == 1, but got 4

c = torch.empty(1000000000, 1000000000)
t8 = from_dlpack(c, use_32bit_stride=True).mark_layout_dynamic()
# Layout in DLTensorWrapper has int32 overflow risk. Please set use_32bit_stride to False.
```

## Mark the Tensor’s Layout as Dynamic with `mark_compact_shape_dynamic`

The `mark_compact_shape_dynamic` function provides fine-grain control over dynamic shapes for compact layouts. The full signature of `mark_compact_shape_dynamic` is as follows:

```python
def mark_compact_shape_dynamic(
    self,
    mode: int,
    stride_order: tuple[int, ...] | None = None,
    divisibility: int = 1,
):
    ...
```

The `mode` parameter determines which shape dimension becomes dynamic. After calling this function, the specific shape dimension given by `mode` is marked as dynamic immediately. The stride will be updated accordingly. For modes that have a shape of size 1, their stride are canonicalized to 0.

The `stride_order` parameter specifies the ordering of strides in the tensor. It is consistent with `torch.Tensor.dim_order()` and defaults to `None`. The parameter indicates the order of modes (dimensions) if the current layout were to be converted to row-major order. It starts from the outermost to the innermost dimension when reading it from left to right. This parameter must be explicitly set when the stride order cannot be automatically deduced from the tensor’s layout, such as when multiple dimensions have a stride of 1.

For example:

- Layout `(4,2):(1,4)` has a stride_order of `(1,0)` indicates the innermost dimension is 0 (4:1), the outermost dimension is 1 (2:4).
- Layout `(5,3,2,4):(3,1,15,30)` has a stride_order of `(3,2,0,1)` indicates the innermost dimension is 1 (3:1), the outermost dimension is 3 (4:30).

If `stride_order` is not specified, the system automatically deduces it from the tensor’s layout using the following logic:

- Sort the strides in descending order.
- If multiple dimensions have a stride of 1, a deduction failure error is raised.

For example:

- For a tensor with layout `(2,2,3,4):(2,1,4,12)`, the deduced stride_order is `[3,2,0,1]`.
- For a tensor with layout `(1,5,1):(1,1,1)`, `stride_order`’s deduction fails because all dimensions have an identical stride of 1, making it impossible to determine the correct ordering.

If `stride_order` is specified, the system validates that the order is consistent with the tensor’s layout.

The `divisibility` parameter specifies the divisibility of the dynamic shape. It could be used to represent the assumption alignment of the input. Defaults to 1.

Note that this API is only available for compact tensors. For non-compact tensors, we can use `cute.assume` to attach divisibility information to a specific shape mode in a host JIT function, as demonstrated in the following example:

```python
@cute.jit
def foo(a: cute.Tensor):
    new_shape = a.shape
    # use cute.assume to set shape of mode=0 with divisibility=16
    new_shape[0] = cute.assume(new_shape[0], 16)
    new_layout = cute.make_layout(new_shape, stride=a.stride)
    new_a = cute.make_tensor(a.iterator, new_layout)
```

### Code Example

The following example demonstrates how to use `mark_compact_shape_dynamic` to specify dynamic tensor layouts.

- `t0` & `t1` show the usage of `mark_compact_shape_dynamic` with unspecified stride_order and different mode and divisibility.
- `t2` shows the usage of consecutive `mark_compact_shape_dynamic` with unspecified stride_order and different mode and divisibility.
- `t3` & `t4` show the usage of `mark_compact_shape_dynamic` with different specified stride_order.
- `t5`, `t6`, `t7`, `t8`, `t9`, `t10`, `t11`, and `t12` demonstrate incorrect settings for parameters and expected errors.

```python
import torch
from cutlass.cute.runtime import from_dlpack

# (8,4,16,2):(2,16,64,1)
a = torch.empty(16, 4, 8, 2).permute(2, 1, 0, 3)
# (1,4,1,32,1):(4,1,4,4,4) => torch tensor when dimension has shape 1, its stride is degenerated to 1,
# resulting in (1,4,1,32,1):(1,1,1,4,1)
# b.dim_order() is (3,2,4,0,1)
b = torch.empty(32, 1, 1, 1, 4).permute(3, 4, 1, 0, 2)

# auto deduce the stride order to be [2,1,0,3]
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
# The stride_order is not consistent with the last stride_order

t6 = from_dlpack(a).mark_compact_shape_dynamic(
    mode=3, divisibility=5, stride_order=(0, 1, 2, 3)
)
# The stride_order is not consistent with the deduced stride_order

t7 = from_dlpack(b).mark_compact_shape_dynamic(
    mode=0, divisibility=4
)
# The layout could not be deduced, please specify the stride_order explicitly

t8 = from_dlpack(b).mark_compact_shape_dynamic(
    mode=30, divisibility=5, stride_order=(3, 0, 2, 4, 1)
)
# Expected mode value to be in range [0, 5), but got 30

t9 = from_dlpack(b).mark_compact_shape_dynamic(
    mode=3, divisibility=5, stride_order=(2, 1, 2, 3, 4)
)
# Expected stride_order to contain all the dimensions of the tensor, but it doesn't contain 0.

t10 = from_dlpack(b).mark_compact_shape_dynamic(
    mode=3, divisibility=5, stride_order=(0, 1, 2, 3, 4, 5)
)
# Expected stride_order to have 5 elements, but got 6.

t11 = from_dlpack(b).mark_compact_shape_dynamic(
    mode=0, divisibility=4, stride_order=b.dim_order()
)
# The shape(1) of mode(0) is not divisible by the divisibility(4)

t12 = from_dlpack(b).mark_compact_shape_dynamic(
    mode=0, divisibility=1, stride_order=(2, 1, 3, 0, 4)
)
# The stride_order is not consistent with the layout

c = torch.empty(1000000000, 1000000000)
t13 = from_dlpack(c, use_32bit_stride=True).mark_compact_shape_dynamic(
    mode=0, divisibility=1
)
# Layout in DLTensorWrapper has int32 overflow risk. Please set use_32bit_stride to False.
```


## Leveraging TVM FFI for Faster PyTorch Interop

The latest version of CuTe DSL supports TVM FFI to improve interoperability with PyTorch and other machine learning frameworks. Using TVM FFI provides the following features:

- Faster JIT function invocation.
- Direct acceptance of torch.Tensor objects as function arguments.
- Enhanced error handling and kernel validation.
- Seamless integration with multiple programming languages.

For more details, see Compile with TVM FFI(https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/compile_with_tvm_ffi.html).

## Bypass the DLPack Protocol

In certain scenarios, users may wish to bypass the DLPack protocol and invoke the JIT function directly. This can be accomplished by creating a lightweight JIT wrapper around the existing JIT function, utilizing `cute.ptr` and `cute.make_tensor` to pass pointers and construct tensors directly.

Typical use cases for bypassing DLPack include: 1. Users want to call the JIT function directly to avoid the overhead introduced by the DLPack protocol. 2. DLPack canonicalizes the stride of shape-1 dimensions to 1, which may result in incorrect alignment propagation and affect memory access or performance. 3. DLPack may lack support for some narrow data types.

The following example illustrates how to bypass the DLPack protocol when invoking a JIT function. Assume we have a pre-defined `TensorOpGemm` kernel whose JIT interface expects three arguments of type `cute.Tensor`. To enable direct invocation without DLPack, we first define a JIT wrapper function that accepts `cute.Pointer` types as parameters. Within this wrapper, we use `cute.make_tensor` to construct tensors from the provided pointers, and then call the `TensorOpGemm` kernel as usual.

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

    # Assume alignment of shape to call tensorop_gemm example
    m = cute.assume(m, divby=8)
    n = cute.assume(n, divby=8)

    # Torch is row major
    a_layout = cute.make_ordered_layout((m, k, l), order=(0, 1, 2))
    b_layout = cute.make_ordered_layout((n, k, l), order=(0, 1, 2))
    c_layout = cute.make_ordered_layout((m, n, l), order=(1, 0, 2))
    mA = cute.make_tensor(a_ptr, layout=a_layout)
    mB = cute.make_tensor(b_ptr, layout=b_layout)
    mC = cute.make_tensor(c_ptr, layout=c_layout)

    # TensorOpGemm is a pre-defined kernel from our example
    tensor_op_gemm = TensorOpGemm(
        a_ptr.value_type, c_ptr.value_type, cutlass.Float32, (2, 2, 1)
    )

    tensor_op_gemm(mA, mB, mC)
```

To pass a PyTorch tensor to this new JIT wrapper, we retrieve the raw pointer from the PyTorch tensor and create a `cute.Pointer` instance using `cute.make_ptr`. This approach allows us to bypass the DLPack protocol entirely, avoiding its overhead and potential issues with shape-1 dimension handling.

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

## Debugging

This page provides an overview of debugging techniques and tools for CuTe DSL programs.

## Getting Familiar with the Limitations

Before diving into comprehensive debugging capabilities, it’s important to understand the limitations of CuTe DSL. Understanding these limitations will help you avoid potential pitfalls from the start.

Please refer to **Limitations**(https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/limitations.html) for more details.

## Source Code Correlation

CuTe DSL provides Python code to PTX/SASS correlation to enable the profiling/debugging of generated kernels with debug symbols by generating line info when compiling the kernel.

You can enable that globally via the environment variable `CUTE_DSL_LINEINFO=1`. Alternatively, you can use compilation options to enable that per kernel. Please refer to **JIT Compilation Options**(https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_jit_compilation_options.html) for more details.

## DSL Debugging

CuTe DSL provides built-in logging mechanisms to help you understand the code execution flow and some of the internal state.

### Enabling Logging

CuTe DSL provides environment variables to control logging level:

```shell
# Enable console logging (default: False)
export CUTE_DSL_LOG_TO_CONSOLE=1

# Log to file instead of console (default: False)
export CUTE_DSL_LOG_TO_FILE=my_log.txt

# Control log verbosity (0, 10, 20, 30, 40, 50, default: 10)
export CUTE_DSL_LOG_LEVEL=20
```

### Log Categories and Levels

Similar to standard Python logging, different log levels provide varying degrees of detail:

| Level | Description |
| --- | --- |
| 0 | Disabled |
| 10 | Debug |
| 20 | Info |
| 30 | Warning |
| 40 | Error |
| 50 | Critical |

## Dump the generated IR

For users familiar with MLIR and compilers, CuTe DSL supports dumping the Intermediate Representation (IR). This helps you verify whether the IR is generated as expected.

```shell
# Dump Generated CuTe IR (default: False)
export CUTE_DSL_PRINT_IR=1

# Keep Generated CuTe IR in a file (default: False)
export CUTE_DSL_KEEP_IR=1
```

## Dump the generated PTX & CUBIN

For users familiar with PTX and SASS, CuTe DSL supports dumping the generated PTX and CUBIN.

```shell
# Dump generated PTX in a .ptx file (default: False)
export CUTE_DSL_KEEP_PTX=1

# Dump generated cubin in a .cubin file (default: False)
export CUTE_DSL_KEEP_CUBIN=1
```

To further get SASS from cubin, users can use `nvdisasm` (usually installed with CUDA toolkit) to disassemble the cubin.

```shell
nvdisasm your_dsl_code.cubin > your_dsl_code.sass
```

## Access the dumped contents programmatically

For compiled kernels, the generated PTX/CUBIN/IR can be accessed programmatically as well through following attributes:

- `__ptx__`: The generated PTX code of the compiled kernel.
- `__cubin__`: The generated CUBIN data of the compiled kernel.
- `__mlir__`: The generated IR code of the compiled kernel.

```python
compiled_foo = cute.compile(foo, ...)
print(f"PTX: {compiled_foo.__ptx__}")
with open("foo.cubin", "wb") as f:
    f.write(compiled_foo.__cubin__)
```

## Change the dump directory

By default, all dumped files are saved in the current working directory. To specify a different directory for the dumped files, please set the environment variable `CUTE_DSL_DUMP_DIR` accordingly.

## Kernel Functional Debugging

### Using Python’s `print` and CuTe’s `cute.printf`

CuTe DSL programs can use both Python’s native `print()` as well as our own `cute.printf()` to print debug information during kernel generation and execution. They differ in a few key ways:

- Python’s `print()` executes during compile-time only (no effect on the generated kernel) and is typically used for printing static values (e.g. a fully static layouts).
- `cute.printf()` executes at runtime on the GPU itself and changes the PTX being generated. This can be used for printing values of tensors at runtime for diagnostics, but comes at a performance overhead similar to that of `printf()` in CUDA C.

For detailed examples of using these functions for debugging, please refer to the associated notebook referenced in **Educational Notebooks**(https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/notebooks.html).

## Handling Unresponsive/Hung Kernels

When a kernel becomes unresponsive and `SIGINT` (`CTRL+C`) fails to terminate it, you can follow these steps to forcefully terminate the process:

1. Use `CTRL+Z` to suspend the unresponsive kernel
2. Execute the following command to terminate the suspended process:

```shell
# Terminate the most recently suspended process
kill -9 $(jobs -P | tail -1)
```

CuTe DSL can also be debugged using standard NVIDIA CUDA tools.

## Using Compute-Sanitizer

For detecting memory errors and race conditions:

```shell
compute-sanitizer --some_options python your_dsl_code.py
```

Please refer to the compute-sanitizer documentation(https://developer.nvidia.com/compute-sanitizer) for more details.

## Conclusion

This page covered several key methods for debugging CuTe DSL programs. Effective debugging typically requires a combination of these approaches. If you encounter issues with DSL, you can enable logging and share the logs with the CUTLASS team as a GitHub issue to report a bug.

## Guidance for Auto-Tuning

Numerous GEMM kernel code examples are offered within our codebase. When integrating these kernels into frameworks, auto-tuning becomes essential for achieving optimal performance. This involves selecting the appropriate kernel parameters based on the inputs of real applications. Next, we’ll briefly introduce some tips on how to perform auto-tuning.

The auto-tuning process typically involves the following steps:

1. Define search space
2. Benchmark each configuration and select the kernel with the best performance
3. Enable caching to reduce the tuning cost

The search space defines the valid combinations of kernel parameters that can be used to run the kernels. Different inputs (shapes, data types, etc.) typically require different kernel parameters to achieve optimal performance. The search space is related to the kernel. We take the Blackwell GEMM persistent kernel as an example. The search space is as follows:

- `mma_tiler_mn`: Defines the dimensions of the matrix tile that each Matrix Multiply-Accumulate (MMA) instruction processes in a single operation.
- `cluster_shape_mn`: Specifies the number of CTAs along each dimension within a cluster. Refer Parallel Thread Execution ISA documentation(https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensorcore-5th-generation-family-instructions) for the possible mma tiler size and cluster shape for different tensor data types.
- `use_2cta_instrs`: Whether to utilize Blackwell’s 2 CTA instructions for MMA/Copy.
- `use_tma_store`: Whether to use Tensor Memory Access (TMA) instructions to store the result back to global memory.

     

After defining the search space, we could traverse all parameter combinations to find the optimal kernel. The `autotune_gemm` function below demonstrates a simple exhaustive search approach - it iterates through configurations, compiles and benchmarks each kernel, and returns the best performing one. Since kernel compilation incurs overhead, it’s important to cache and reuse compiled kernels to minimize host launch latency. CuTe DSL facilitates this through its separate compilation and execution workflow. More details can be found in JIT Caching. As demonstrated in the `autotune_gemm` function (between the `begin of cache the compiled GEMM kernel` and `end of cache the compiled GEMM kernel` comments), we can use `cute.compile()` to compile a kernel once, cache the compiled result, and reuse the cached JIT executor for multiple kernel executions. We could maintain a global configuration-to-kernel dictionary (`config_kernel_dict`) to cache the compiled GEMM kernels, where each key (`kernel_cache_key`) uniquely identifies a kernel based on its characteristics. Usually we could use the {dtype + kernel configs} as the cached key for GEMM compilation. For example,

```python
kernel_cache_key = f"{ab_dtype}x{c_dtype}x{acc_dtype}x{use_2cta_instrs}x{mma_tiler}x{cluster_shape_mn}x{use_tma_store}"
```

If the input tensor’s layout is static, we should add the shape in the cached key too. Users can customize the `benchmark` function to measure kernel execution time. For stable and reliable performance measurements:

1. Run a few warmup iterations (e.g., 5-10) to stabilize GPU temperature
2. Execute multiple timed iterations (e.g., 100-1000) for statistical significance
3. Use CUDA events and synchronization for precise timing
4. Lock GPU frequencies (SM and memory frequencies) with nvidia-smi
5. Process results by removing outliers and using min/avg statistics as measurements.

This ensures reliable kernel selection through proper benchmarking.

```python
# get the best GEMM kernel for given input tensors
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
    # traverse the search space
    for use_2cta_instrs in use_2cta_instrs_list:
        for use_tma_store in use_tma_store_list:
            for mma_tiler_mn in product(mma_tiler_m_list, mma_tiler_n_list):
                for cluster_shape_mn in product(cluster_shape_m_list, cluster_shape_n_list):
                    acc_dtype = cutlass.Float32
                    hardware_info = cutlass.utils.HardwareInfo()
                    max_active_clusters = hardware_info.get_max_active_clusters(
                        cluster_shape_mn[0] * cluster_shape_mn[1]
                    )
                    # instance a GEMM kernel
                    gemm = PersistentDenseGemmKernel(
                        acc_dtype,
                        use_2cta_instrs,
                        mma_tiler_mn,
                        cluster_shape_mn,
                        use_tma_store,
                    )
                    # begin of cache the compiled GEMM kernel
                    if kernel_cache_key not in config_kernel_dict:
                        # compile gemm kernel
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
                    # end of cache the compiled GEMM kernel
                    try:
                        # define a benchmark function to measure the execution time of the compiled GEMM kernel
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

This brute-force approach ensures we could find the optimal parameters, though at the cost of trying every possibilities. For more advanced use cases, users can explore sophisticated optimization techniques like search space pruning and genetic algorithms to reduce tuning overhead and discover better configurations more efficiently.

To further optimize tuning performance, we can utilize caching mechanisms to avoid redundant computations. We could cache the tuning results in a input-to-kernel dictionary (e.g., `input_kernel_dict`). When processing inputs with matching `config_key` values, the cached kernel can be reused directly without re-tuning. The `config_key` is related with the input tensor’s characteristics, such as the shape, data type, etc. The setup of `config_key` is very flexible, users can customize it based on their own application. For instance, if the data type is fixed in users’ application, we could use the input tensor’s shape as the key, i.e., `(m, n, k)`. To further reduce tuning overhead, we could consider using a simplified key like `config_key = (power_of_2(m), power_of_2(n), power_of_2(k))`, where `m`, `n`, and `k` are rounded up to the nearest power of 2. This simplification can significantly reduce the number of unique keys while still maintaining good performance in most cases. However, it’s important to validate that this approximation doesn’t negatively impact performance for your specific use case.

```python
config_key = (m, n, k)
if config_key in input_kernel_dict:
    compiled_gemm = input_kernel_dict[config_key]
else:
    compiled_gemm = autotune_gemm(...)
    input_kernel_dict[config_key] = compiled_gemm

# launch gemm kernel
compiled_gemm(a_tensor, b_tensor, c_tensor, stream)
```

By following the methods above, you can customize your own auto-tuner to find the optimal GEMM kernel configuration for specific matrix dimensions and data types, significantly improving computational performance for models.

# Educational Notebooks

A number of notebooks for educational purposes are provided in the CUTLASS GitHub repository. A list with handful links is given below:

- “Hello world”(github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks/hello_world.ipynb)
- Printing(https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/notebooks/print.ipynb)
- Data Types Basics(https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/notebooks/print.ipynb)
- Tensors(github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks/tensor.ipynb)
 - The TensorSSA Abstraction(github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks/tensorssa.ipynb)
 - Layout Algebra(https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks/cute_layout_algebra.ipynb)
 - Element-wise Add Tutorial(https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/notebooks/elementwise_add.ipynb)
 - Using CUDA Graphs(github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks/cuda_graphs.ipynb)
 
 
 

# Compile with TVM FFI

Apache TVM FFI is an open ABI and FFI for machine learning systems. More information can be found in the official documentation(https://tvm.apache.org/ffi/).

To install TVM FFI, you can run the following command:

```shell
pip install apache-tvm-ffi
# optional package for improved torch tensor calling performance
pip install torch-c-dlpack-ext
```

In CuTe DSL, TVM FFI can be enabled as an option for JIT-compiled functions. Using TVM FFI can lead to faster JIT function invocation and provides better interoperability with machine learning frameworks (e.g., directly take `torch.Tensor` as arguments).

## Enable Apache TVM FFI in CuTe DSL

First, install the `tvm-ffi` package by following its installation guide(https://tvm.apache.org/ffi/#installation).

There are two ways to enable TVM FFI in CuTe DSL:

1. Use the `options` argument in `cute.compile` to specify the TVM FFI option. For example:

```python
# Assuming you have defined a function `add` decorated with @cute.jit
def example_compile():
   a_torch = torch.randn(10, 20, 30).to(torch.float16)
   b_torch = torch.randn(10, 20, 30).to(torch.float16)
   a_cute = cute.runtime.from_dlpack(a_torch, enable_tvm_ffi=True).mark_layout_dynamic()
   b_cute = cute.runtime.from_dlpack(b_torch, enable_tvm_ffi=True).mark_layout_dynamic()

   compiled_add = cute.compile(add, a_torch, b_torch, options="--enable-tvm-ffi")
```

Note that the object returned by `cute.compile` is a Python function specific to TVM FFI.

2. Alternatively, you can enable TVM FFI globally by setting the environment variable `CUTE_DSL_ENABLE_TVM_FFI=1`. Please note that this setting will apply to all JIT compilations within the environment.

## Minimizing Host Overhead

Eager kernel invocation overhead on the CPU host can sometimes become a bottleneck for latency-sensitive applications. TVM FFI can help greatly reduce this overhead. To maximize performance benefits, we recommend setting up your workflow as follows (detailed instructions are provided in subsequent sections):

- **Compile the kernel with TVM FFI enabled**.
- **Declare shape constraints using fake tensors** and reuse the compiled function throughout your execution.
- **Pass PyTorch tensors directly** to the compiled function to avoid explicit DLPack conversion.
- **Use the environment stream flag** to implicitly pass the current PyTorch stream.
- **Rely on compiled argument-validation** instead of Python-side attribute validation, as TVM FFI functions perform fast compiled checks.

Following these steps can significantly reduce the host-side overhead of eager kernel execution. The sections below provide detailed examples and explanations for each step. You may find it helpful to refer back to this summary after you review the implementation details.

## Fake tensor for compilation

The TVM FFI function accepts DLPack-compatible tensors as arguments, such as those from torch or jax. However, during compilation, it is necessary to specify the tensors’ dynamic properties in CuTe DSL. To clearly distinguish between the compilation phase and runtime, CuTe DSL provides a “fake tensor” that can be used for compilation. For example:

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
   # compile the kernel with "--enable-tvm-ffi" option and example input tensors
   compiled_add_one = cute.compile(add_one, a_cute, b_cute, options="--enable-tvm-ffi")
   # now compiled_add_one is a TVM-FFI function that can be called with torch.Tensor as input
   a_torch = torch.arange(10, dtype=torch.float32, device="cuda")
   b_torch = torch.empty(10, dtype=torch.float32, device="cuda")
   compiled_add_one(a_torch, b_torch)
   print("result of b_torch after compiled_add_one(a_torch, b_torch)")
   print(b_torch)
```

The fake tensor is a placeholder that mimics the interface of a real tensor but does not hold real data or allow indexing. It is used in compilation or testing scenarios where only shape/type/layout information is needed. All attempts to access or mutate data will raise errors.

## Note on Stride Order

Note that CuTe’s convention is to write the stride order for dimensions from left to right, where a lower order number means higher priority. In the context of the `make_fake_compact_tensor` API, for shape `(2, 3, 4)` and stride order `(0, 1, 2)`, the stride is `(1, 2, 6)`. This is commonly known as column-major order. If you want to create a fake tensor with compact row-major order, you should explicitly pass in `stride_order=tuple(reversed(range(len(shape))))` to `make_fake_compact_tensor`. Alternatively, you can always precisely control the stride via the `stride` argument in the `make_fake_tensor` API.

## `cute.Tensor` adapter for TVM FFI

To adapt the `cute.Tensor` to the TVM FFI function, you can use the `cute.runtime.from_dlpack` function with the `enable_tvm_ffi=True` option or the environment variable `CUTE_DSL_ENABLE_TVM_FFI=1`. For example:

```python
def example_from_dlpack():
    a_cute = cute.runtime.from_dlpack(a_torch, enable_tvm_ffi=True).mark_layout_dynamic()
    b_cute = cute.runtime.from_dlpack(b_torch, enable_tvm_ffi=True).mark_layout_dynamic()

    compiled_add_one(a_cute, b_cute)
```

Note that because the `cute.runtime.from_dlpack` function performs an explicit DLPack conversion, it is less efficient than passing the `torch.Tensor` directly. You can also use `cute.Tensor` as an argument hint for `cute.compile`.

```python
compiled_add_one = cute.compile(add_one, a_cute, b_cute, options="--enable-tvm-ffi")
```

## Working with torch Tensors

As you may have noticed in the examples above, TVM FFI-compiled functions can directly accept `torch.Tensor` objects (and other DLPack-compatible tensors) as inputs. The resulting functions add minimal overhead, enabling faster eager invocations thanks to the optimized calling path.

## Working with Streams

In many cases, a CuTe kernel needs to run on a specific CUDA stream. CuTe DSL provides two ways to work with streams through TVM FFI. The first is to pass the stream explicitly as an argument. The following example demonstrates this approach; the function accepts `torch.cuda.Stream`, `CUstream` or any stream class that implements the CUDA stream protocol.

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
   # Fake stream is a placeholder for stream argument
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

## Using Environment Stream

The second option is to rely on the environment stream flag. Pass `use_tvm_ffi_env_stream=True` to `make_fake_stream` to mark the stream argument as an environment stream, which means it no longer needs to be provided explicitly. TVM FFI will automatically use its environment stream (i.e., the current PyTorch stream) as the stream argument. The example below demonstrates this flow:

```python
def example_add_one_with_env_stream():
   n = cute.sym_int()
   a_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   b_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   # Fake stream is a placeholder for stream argument
   # we will use TVM FFI environment stream
   stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
   compiled_add_one = cute.compile(
      add_one_with_stream, a_cute, b_cute, stream, options="--enable-tvm-ffi"
   )
   a_torch = torch.arange(10, dtype=torch.float32, device="cuda")
   b_torch = torch.empty(10, dtype=torch.float32, device="cuda")
   torch_stream = torch.cuda.current_stream()
   with torch.cuda.stream(torch_stream):
      # no need to pass in the stream explicitly, env stream will be synced
      # to torch.cuda.current_stream() before the function call.
      compiled_add_one(a_torch, b_torch)
   torch_stream.synchronize()
   print("result of b_torch after compiled_add_one(a_torch, b_torch)")
   print(b_torch)
```

Using the environment stream flag both speeds up calls and simplifies integration with frameworks such as PyTorch, since no explicit stream parameter is required. We recommend using the environment stream flag to both simplify framework integration and minimize host-side calling overhead.

## Working with Tuples

TVM FFI functions can also accept tuples as arguments. Tuples can be recursively composed of the types that are supported by TVM FFI. The example below shows how to use tuples as arguments:

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

## Working with Variadic Tuples

Sometimes it is helpful to annotate a tuple with no explicit element types. This can be useful to build up a generic template for a function that accepts a variable number of elements. The compiled function’s signature will be determined by the tuple argument passed to the `cute.compile` function. The following example shows how to use a variadic tuple to build such a generic template.

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

## Working with Named Tuples

Named tuples are also supported and help logically group related arguments together. The example below shows how to use named tuples as arguments. Under the hood, named tuples are passed as unnamed tuples at the ABI level. When errors occur, the function signature in error messages will display unnamed tuple arguments. Ensure that the compile-time CuTe named tuple type definition has the same fields as the runtime PyTorch named tuple. Currently, users need to explicitly unpack the named tuple outside of conditionals and then use the unpacked variables inside the conditionals.

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
   # need to unpack namedtuple outside conditionals
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


## Supported types

The TVM FFI function supports the following CuTe DSL-specific types as arguments:

- `cute.Tensor`
- `cutlass.Boolean`, `cutlass.Int8`, `cutlass.Int16`, `cutlass.Int32`, `cutlass.Int64`, `cutlass.Uint8`, `cutlass.Uint16`, `cutlass.Uint32`, `cutlass.Uint64`, `cutlass.Float32`, `cutlass.Float64`
- `cute.Shape`, `cute.Stride`, `cute.Coord`, `cute.Tile`, `cute.IntTuple`

| Compile-time type | Call-time type |
| --- | --- |
| `cute.Pointer` | `ctypes.c_void_p` or a class that implements `__tvm_ffi_opaque_ptr__` protocol. |
| `cute.runtime.FakeTensor` | `torch.Tensor` and other DLPack-compatible tensors. |
| Scalar types (e.g. `cutlass.Boolean`, `cutlass.Int32`) | Python scalars (e.g. True, 123). |
| CuTe algebra types (e.g. `cute.Shape`, `cute.Stride`) | `tvm_ffi.Shape` or python tuple of ints. |
| CUDA stream `cuda.CUstream` | A stream class that implements the CUDA stream protocol (e.g. `torch.cuda.Stream`, `cuda.CUstream`). |
| Tuple of types (e.g. `Tuple[cute.Tensor, cute.Tensor, cutlass.Int32]`) | Python tuple of corresponding call-time types. |

## Error handling

TVM FFI functions will enable validation of arguments to make sure they match the expected type and value constraints declared by the user. These checks are compiled into the function, run very fast, and have no observable overhead during function invocation. Each of those errors will translate into a proper Python exception that can be caught and handled. The example below shows some example error cases that can be checked:

```python
def example_constraint_checks():
   n = cute.sym_int(divisibility=16)
   # assume align to 16 bytes (4 int32), both should share same shape variable n
   a_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,), assumed_align=16)
   b_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,), assumed_align=16)
   compiled_add_one = cute.compile(add_one, a_cute, b_cute, options="--enable-tvm-ffi")
   a = torch.zeros(128, dtype=torch.float32, device="cuda")
   b = torch.zeros(128, dtype=torch.float32, device="cuda")

   try:
      # raises type mismatch error because we expect a and b to be float32
      compiled_add_one(a, 1)
   except TypeError as e:
      # Mismatched type on argument #1 when calling:
      # `add_one(a: Tensor([n0], float32), b: Tensor([n0], float32))`,
      # expected Tensor
      print(f"TypeError: {e}")

   try:
      # raises shape mismatch error because we expect both a and b have shap [n]
      compiled_add_one(a, b[:126])
   except ValueError as e:
      # Mismatched b.shape[0] on argument #1 when calling:
      # `add_one(a: Tensor([n0], float32), b: Tensor([n0], float32))`,
      # expected to match a.shape[0]
      print(f"ValueError: {e}")

   try:
      # triggers divisibility mismatch error because 126 is not divisible by 16
      compiled_add_one(a[:126], b[:126])
   except ValueError as e:
      # Invalid a.shape[0] on argument #0 when calling:
      # `add_one(a: Tensor([n0], float32), b: Tensor([n0], float32)`,
      # expected to be divisible by 16
      print(f"ValueError: {e}")

   try:
      a = torch.zeros(129, dtype=torch.float32, device="cuda")
      b = torch.zeros(129, dtype=torch.float32, device="cuda")
      # triggers data alignment mismatch error because x and y are not aligned to 16 bytes
      compiled_add_one(a[1:], b[1:])
   except ValueError as e:
      # raises: Misaligned Tensor data on argument #0 when calling:
      # `add_one(a: Tensor([n0], float32), b: Tensor([n0], float32)`,
      # expected data alignment=16 bytes
      print(f"ValueError: {e}")
```

Any CUDA errors encountered will also be automatically converted into Python exceptions by the TVM FFI function.

```python
@cute.jit
def add_one_invalid_launch(a: cute.Tensor, b: cute.Tensor):
   # Intentionally exceed the maximum block dimension (1024 threads) so the
   # CUDA runtime reports an invalid configuration error.
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

## Working with Devices

TVM FFI-compiled functions naturally work across GPU devices. The device index of the first input GPU tensor determines the kernel’s device context. The TVM FFI function calls `cudaSetDevice` to set the correct device before launching the kernel based on that tensor’s device index. For advanced scenarios that pass raw pointers instead of tensors, you should call `cudaSetDevice` explicitly through the CUDA Python API.

## Exporting Compiled Module

The TVM FFI function supports exporting the compiled module to an object file for further use. For example:

```python
import subprocess
import cutlass.cute as cute

def example_add_one_export():
   n = cute.sym_int()
   a_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   b_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   # compile the kernel with "--enable-tvm-ffi" option and example input tensors
   compiled_add_one = cute.compile(add_one, a_cute, b_cute, options="--enable-tvm-ffi")
   # export the compiled module to object file
   compiled_add_one.export_to_c("./add_one.o", function_name="add_one")
   # obtain necessary runtime libs for loading the shared library
   runtime_libs = cute.runtime.find_runtime_libraries(enable_tvm_ffi=True)
   # compile the object file to a shared library
   cmd = ["gcc", "-shared", "-o", "./add_one.so", "./add_one.o", *runtime_libs]
   print(cmd)
   subprocess.run(cmd, check=True)
   print(f"Successfully created shared library: ./add_one.so")
```

Then you can load back the exported module and use it in different ways:

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

The exported object file exposes the function symbol `__tvm_ffi_add_one` that is compatible with TVM FFI and can be used in various frameworks and programming languages. You can either build a shared library and load it back, or link the object file directly into your application and invoke the function via the `InvokeExternC` mechanism in TVM FFI. For more information, see the quick start guide(https://tvm.apache.org/ffi/get_started/quickstart) in the official documentation.

When you build your own libraries, make sure you link against the necessary runtime libraries. You can use `cute.runtime.find_runtime_libraries(enable_tvm_ffi=True)` to get the path to these libraries. `cute.runtime.load_module` will load these libraries automatically before loading an exported module. You can also manually load these libraries in advanced use cases.

## Keyword Arguments and Defaults

The function returned by `cute.compile` supports keyword arguments and defaults. The example below shows how to use keyword arguments and defaults:

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

For efficiency and portability reasons, TVM FFI ABI supports functions with positional-only arguments. If you export the compiled module to an object file and then load it back, the function will only accept positional arguments in the order of the arguments in the function signature. You can rewrap the function or use the TVM FFI wrapper generator to generate a kwargs wrapper. The code block below shows how to do this:

```python
def example_kwargs_and_defaults():
   n = cute.sym_int()
   a_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   b_cute = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
   compiled_add_constant = cute.compile(add_constant, a_cute, b_cute, options="--enable-tvm-ffi")
   # export the compiled module to object file
   compiled_add_constant.export_to_c("./add_constant.o", function_name="add_constant")
   # obtain necessary runtime libs for loading the shared library
   runtime_libs = cute.runtime.find_runtime_libraries(enable_tvm_ffi=True)
   # compile the object file to a shared library
   cmd = ["gcc", "-shared", "-o", "./add_constant.so", "./add_constant.o", *runtime_libs]
   subprocess.run(cmd, check=True)

   a_torch = torch.arange(10, dtype=torch.float32, device="cuda")
   b_torch = torch.empty(10, dtype=torch.float32, device="cuda")

   mod = cute.runtime.load_module("./add_constant.so")
   try:
      mod.add_constant(a_torch, b_torch)
   except Exception as e:
      # Raises a missing arguments error because kwargs and default information are lost
      print(e)
   # We rewrap the function to regain argument and kwargs support.
   # Alternatively, use the TVM FFI wrapper generator to generate a kwargs wrapper function.
   from tvm_ffi.utils import kwargs_wrapper
   # arg_defaults are aligned to the end of the argument list
   wrapped_func = kwargs_wrapper.make_kwargs_wrapper(
      mod.add_constant, arg_names=["a", "b", "offset"], arg_defaults=(1,)
   )
   wrapped_func(a_torch, b_torch)
   print("result of b_torch after wrapped_func(a_torch, b_torch)")
   print(b_torch)
   # You can also use the signature of the original function
   # to generate a kwargs wrapper function. Make sure to exclude
   # arguments that are not included in the runtime,
   # such as 'self', constexpr, and env stream arguments.
   wrapped_func = kwargs_wrapper.make_kwargs_wrapper_from_signature(
      mod.add_constant, signature=inspect.signature(add_constant),
      exclude_arg_names=["self"]
   )
   wrapped_func(a_torch, b_torch, offset=4)
   print("result of b_torch after wrapped_func(a_torch, b_torch, offset=4)")
   print(b_torch)
```
 
 
 

# Limitations

## Overview

CuTe DSL is an embedded domain-specific language within Python. It utilizes a subset of Python’s syntax to provide a streamlined programming experience. It is important to understand that CuTe DSL does NOT implement the complete Python language semantics in its JIT compilation process.

This section documents the current limitations of the CuTe DSL. While some of these limitations may be addressed in future releases, developers should be aware of them when building applications with the DSL.

## Notable unsupported features

- Programmatic Dependent Launch (PDL)
- convolutions
- full support for ahead of time compilation
- preferred clusters
- CLC-based tile schedulers
- EVT support
- Windows support

## Programming Model

### CuTe Layout Algebra Only support 32bit

Today, we only support 32bit shapes/strides in CuTe layouts. 64bit or arbitrary width support is planned for future releases.

### Python Native Data Types

CuTe DSL supports Python data structures when used for “meta-programming”, but these structures cannot be treated as dynamic values modifiable at runtime. For instance, lists and dictionaries can be used to configure kernel parameters during compilation or serve as containers for dynamic values, but their structure and organization cannot be altered during kernel execution.

- Static Values:
  - Evaluated during JIT compilation phase
  - Immutable after compilation completes
  - Most Python native types (lists, tuples, dictionaries) are processed as static values
  - Primarily utilized for “meta-programming” and configuration purposes
  - Example: Lists can contain dynamic values but their structure cannot be modified during kernel execution

- Dynamic Values:
  - Evaluated during runtime execution
  - Modifiable during execution of JIT-compiled functions
  - Only a specific subset of Python types are supported as dynamic values
  - Primitive types are automatically converted when passed as function arguments:
    - `int` -> `Int32` (may be updated to `Int64` in future releases)
    - `bool` -> `Bool`
    - `float` -> `Float32` (may be updated to `Float64` in future releases)

The JIT compiler processes Python native types analogously to C++ template parameters. The compiled code cannot manipulate dynamic values of composite types such as lists, tuples, or dictionaries.

For example, following code doesn’t work as traditional Python program inside JIT function.

```python
@cute.jit
def foo(a: Float32, b: Float32, i: Int32, res: cute.Tensor):
    xs = [a, b]
    # indexing list with dynamic index is not supported in CuTe DSL:
    res[0] = xs[i]

    if i == 0:
        # This will always append Float32(3.0) to the list regardless
        # of the runtime value of `i`
        xs.append(Float32(3.0))

    for i in range(10):
        # This only append one element to the list at compile-time
        # as loop doesn’t unroll at compile-time
        xs.append(Float32(1.0))
```

### Python Function

The DSL currently does not implement support for return values from Python functions, although this capability is planned for future releases.

Example:

```python
@cute.jit
def foo():
    return 1  # Currently unsupported in CuTe DSL
```

### Expression or Statement with Dependent Type

CuTe DSL implements static typing and does not support dependent types. The type of each expression must be determinable during compile time, in contrast to standard Python which implements dynamic typing.

Example illustrating functionality in Python that is not supported in the DSL:

```python
# Valid in standard Python, but unsupported in CuTe DSL
max(int(1), float(2.0))  # => 2.0 : float
max(int(3), float(2.0))  # => 3 : int
```

In CuTe DSL, types are promoted. For example:

```python
@cute.jit
def foo(a: Int32, b: Float32, res: cute.Tensor):
    res[0] = max(a, b)  # Type is automatically promoted to Float32
```

Following code using inlined if-else expression with dependent types is not supported in CuTe DSL:

```python
@cute.jit
def foo(cond: Boolean, a: Int32, b: Float32, res: cute.Tensor):
    res[0] = a if cond else b
```

### Control Flow

The DSL transforms Python control flow statements (`if`, `for`, `while`) during Abstract Syntax Tree (AST) processing into structured control flow in MLIR which has the same constraints as dependent types. For instance, changing type of a variable in loop body is not allowed.

- Variables must be defined prior to the control flow statement
- Type consistency must be maintained throughout the control flow statement
- Don’t support early exit or return from if-else statements

Example illustrating functionality in Python that is not supported in the DSL:

```python
@cute.jit
def foo():
    a = Int32(1)
    for i in range(10):
        a = Float32(2)  # Changing type inside loop-body is not allowed in the DSL
```

### Built-in Operators

The DSL transforms built-in operators like `and`, `or`, `max`, `min`, etc. into MLIR operations. They also follow the same constraints of dependent types. For instance, `a and b` requires `a` and `b` to be of the same type.

### Special Variables

The DSL treats `_` as a special variable that it’s value is meant to be ignored. It is not allowed to read `_` in the DSL.

Example illustrating functionality in Python that is not supported in the DSL:

```python
@cute.jit
def foo():
    _ = 1
    print(_)  # This is not allowed in the DSL
```

### Object Oriented Programming

The DSL is implemented on top of Python and supports Python’s object-oriented programming (OOP) features for meta-programming at compile-time.

However, similar to other composed data types, the DSL provides limited support for OOP when objects contain dynamic values. It is strongly recommended to avoid passing dynamic values between member methods through class state in your code.

The following example illustrates functionality in Python that is not supported in the DSL without implementing the `DynamicExpression` protocol:

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

    # This fails to compile because `Foo.a` is assigned a local value defined within the for-
    # loop body, which is not visible outside of the loop body
    res[0] = foo.get_a()
```

The example above fails to compile because `Foo.a` is assigned a local value defined within the for-loop body, which is not visible outside of the loop body.

The CuTe DSL implements an internal mechanism that provides limited support for OOP patterns via protocol. As the DSL continues to evolve to support additional features, this mechanism is subject to change and is not recommended for direct use in users’ code for better portability.

### CuTe Layout algebra in native Python

Entirety of CuTe Layout algebra operations and APIs require JIT compilation. These functionalities are exclusively available within JIT-compiled functions and cannot be accessed in standard Python execution environments.

Additionally, there exists a restricted set of data types that can be passed as arguments to JIT-compiled functions, which further constrains their usage in native Python contexts. Only following CuTe algebra types are supported as JIT function arguments: `Tensor`, `Pointer`, `Shape`, `Stride`, `Coord` and `IntTuple`. For `Stride`, we don’t support `ScaledBasis` from native Python Context. Unfortunately, in the first release, we don’t support passing `Layout` under native Python Context.

## Suggestions

For reliable and predictable results:

- Avoid dependent types in your code
- Implement explicit type conversion for dynamic values
- Clearly distinguish between static (compile-time) and dynamic (runtime) values
- Use type annotations as much as possible to help JIT compiler to identify type to avoid ambiguity

```python
# Example demonstrating explicit typing
alpha = 1.0  # Explicitly defined as float using `1.0` instead of `1`
beta = 2.0   # or `float(1)`
result = max(alpha, beta)  # Will correctly perform float comparison
```

### Debugging Capabilities

Debugging tools and facilities for the Python DSL are currently more limited in comparison to the C++ API. For instance, we don’t support single-stepping through the JIT-compiled code. And lack of exception handling in JIT-compiled code makes it hard to debug in some cases.

### Integration with Frameworks

Integration with certain deep learning frameworks is in early development stages and may have limitations. For instance, converting frameworking tensor to `cute.Tensor` is known to have overhead with 2us~3us per tensor as we convert from general DLPack protocol which offers compatibility with all frameworks.

### Hashing DSL APIs and Objects

DSL APIs and Objects are sensitive to MLIR context, region or other contextual information which has no meaning cross different context. Any stateful design rely on `__hash__` likely misbehave with unexpected results. An example is `functools.lru_cache`, which combined with `@cute.jit`, it may cache MLIR object from one context and use in another one.

## Future Improvements

The CuTe DSL development team is actively addressing these limitations. Upcoming releases will aim to:

- Implement support for return values from JIT compiled functions
- Improve support for built-in operators to handle more cases without dependent types
- Enhance debugging capabilities and tools
- Improve error messages with precise diagnostic information
- Extend support for additional numeric data types
- Improve performance of converting framework tensor to `cute.Tensor` with native support for different frameworks
- Offer more user friendly benchmarking methodology

## Design Limitations Likely to Remain

The primary objective of CuTe DSL is to provide a domain-specific language for expressing complex CUDA kernels with optimal GPU performance, not to execute arbitrary Python code on GPU hardware.

The following limitations will likely remain by design:

- Complex Data Structures as Dynamic Values: Lists, tuples, and dictionaries will continue to function as static containers. While they can store dynamic values, their structure (adding/removing elements) cannot be modified during execution of JIT-compiled functions.
- Dependent Types: Supporting dependent types would introduce substantial complexity and adversely affect the performance characteristics of generated code.
- CuTe Layout Algebra: We don’t have plan to extend the support of CuTe Layout Algebra under native Python Context. We are planning to extend support for data types and allow JIT function to interoperate with native Python code.

# FAQ: General

## Are the DSLs replacing C++ templates?

TL;DR: No - but also yes. The CUTLASS 4.0 release (CuTe DSL), along with all future extensions to our Python-native programming models, does not come at the expense of CUTLASS C++. CUTLASS 2.x and 3.x C++ APIs are both going to continue receiving fixes and updates for the architectures we support them for. However, CUTLASS 4.x CuTe DSL is fully isomorphic in its programming model and performance with CuTe C++ for Blackwell, and it is our hope that the community embraces this for much easier while still equally performant custom kernel development. This is why we are releasing CuTe DSL with support for all architectures starting with the NVIDIA Ampere Architecture.

## What is the difference between CuTe DSL, CUTLASS Python, and CUTLASS DSLs?

CUTLASS Python was the Python interface for instantiating C++ kernels via a Python frontend. This is now deprecated with the release of CUTLASS 4.0. CUTLASS DSLs are a family of Python DSLs for native device programming in Python. Currently, this is limited to our initial release of CuTe DSL, but future versions will include higher-level abstractions that gradually trade off control for convenience.

## What should I learn, CUTLASS C++ or the Python DSLs?

We believe the Python DSLs will significantly improve the learning curve and recommend starting with them for all newcomers, as they eliminate the inherent complexity of learning C++ metaprogramming for GPU kernel programming. Since CuTe C++ and CuTe DSL share fully isomorphic programming models and patterns, any knowledge gained can eventually be applied to C++.

## Where will the code live? PIP wheel or GitHub repo? Do I have to build it myself?

This is a major change compared to CUTLASS C++ and Python DSLs. Going forward, the GitHub code only exists as a way for users to file issues and pull requests against. While it can be used with the pip wheel, we do not recommend most users do so unless they are hacking on the DSL itself. For all other users, we recommend they simply `pip install nvidia-cutlass-dsl` and use the pip wheel as the single source of truth for the dialect compiler and DSL implementation. CUTLASS GitHub repository will contain a `requirements.txt` file pinning the version of the wheel consistent with the state of the OSS repository (please see Quick Start Guide). This means getting started with CUTLASS is easier than ever: no more CMake command lines to learn and no more builds to kick off. Simply install the pip wheel and start running the examples.

# Migration

## Should I port my code from C++ templates to Python?

Almost certainly not, unless you need extremely fast JIT times for your kernel and C++ compile times are a blocker for you. The 2.x and 3.x APIs will continue to be supported, and Nvidia’s Hopper and Blackwell architectures 3.x will continue to improve in terms of features and performance.

## Are portability promises different with Python?

For the initial release while the DSL is still in beta, we do not promise any portability as we may make changes to the DSL itself. While we do not expect any changes to the CuTe operations, the DSL utilities, decorators, helper classes like pipelines and schedulers may change as we refine them with community feedback. We encourage users to file issues and discussions on GitHub during this beta period with their feedback!

In the long term, we plan to continue to treat the OSS community with care. Just like the prior history of CUTLASS, we plan not to break users unless necessary, but we reserve the right to make limited breaking changes in case we believe it is a net benefit to the community and project. These will be announced ahead of time and/or clearly highlighted in the CHANGELOG of each release.

# Technical

## What NVIDIA architectures will it support?

CuTe DSL will support all NVIDIA GPU architectures starting with NVIDIA Ampere Architecture (SM80).

## Will it be compatible with DL frameworks (e.g., PyTorch, JAX)?

Yes, we will provide utilities to convert from DLPack-supported tensor formats to `cute.Tensor`. This should allow a user to never have to leave Python when writing model code in their framework of choice. Our JAX interoperability story is not as strong as PyTorch’s today, however, we are actively working on improving it and welcome contributions in this space.

## Does it compile to PTX or SASS?

CuTe DSL compiles the program down to PTX. After that, we currently use the PTX compiler that ships with the CUDA toolkit to compile the PTX down to SASS. We plan to remove this limitation in the future and allow the use of the PTX JIT that is included in the CUDA driver in case a user does not have a CUDA toolkit installed.

## Do I need to use NVCC or NVRTC?

No, the `nvidia-cutlass-dsl` wheel packages is everything needed to generate GPU kernels. It shares the driver requirements of the 12.9 toolkit which can be found here.

## How would one debug the code?

Since CuTe DSL is not native python and an embedded DSL instead, tools like pdb cannot be used. However, if you have experience with GPU kernel programming, the debugging techniques will be nearly identical. Typically, compile time and runtime printing of types and values are the most expedient. Please see documentation on printing to learn how to print types and values at both compile time and runtime. You can also use `cuda-gdb` to set breakpoints in the program and step through the execution or use tools such as `compute-sanitizer` to detect and triage bugs in your program. As the DSL matures, our source location tracking from Python user programs will also improve to provide more helpful source-level mapping when setting breakpoints and using other tools such as nsight.

## How would one implement warp specialization in CuTe DSL?

Exactly the same way you would in C++ but in a Python-native syntax instead. Consult our Control Flow and "Blackwell kernel example" for a detailed how-to guide.

## Can I call functions from other functions or use OOP?

Yes. We frequently call functions from one another and set up class hierarchies to organize and modularize our code for pipelines and schedulers. Consult the Introduction documentation or our examples for more details.

# License

## What is the license for CuTe DSL and the associated GitHub samples?

CuTe DSL components available on Github and via the nvidia-cutlass-dsl Python pip wheel are released under the “NVIDIA Software End User License Agreement (EULA)”. Because the pip package includes a compiler that shares several components with the CUDA Toolkit, it is subject to usage terms and restrictions similar to those of the CUDA SDK. Please refer to the EULA for specific terms of use.

CuTe DSL samples and Jupyter notebooks, released on GitHub are provided under the BSD 3-Clause License and may be used and redistributed under those terms. This distinction ensures that developers have flexibility when using or modifying the code samples, independent of the compiler and runtime components governed by the EULA.

If you have any questions or need clarification, feel free to contact us.
