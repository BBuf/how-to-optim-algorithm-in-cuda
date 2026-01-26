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

- Quick Start Guide – Initial setup and installation.

CuTe DSL – Overview of the typical development and workflow using CuTe DSL.

CuTe DSL API – Refer to the full API documentation.

Limitations – Understand current CuTe DSL constraints and differences from C++.

FAQs – Common questions and known issues.
