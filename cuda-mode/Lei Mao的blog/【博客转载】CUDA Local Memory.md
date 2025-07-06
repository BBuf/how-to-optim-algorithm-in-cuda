> 博客来源：https://leimao.github.io/blog/CUDA-Local-Memory/ ，来自Lei Mao，已获得作者转载授权。后续会转载一些Lei Mao的CUDA相关Blog，也是一个完整的专栏，Blog会从稍早一些的CUDA架构到当前最新的CUDA架构，也会包含实用工程技巧，底层指令分析，Cutlass分析等等多个课题，是一个时间线十分明确的专栏。

# CUDA Local Memory

## Introduction

In CUDA programming, local memory is private storage for an executing thread, and is not visible outside of that thread. The local memory space resides in device memory, so local memory accesses have the same high latency and low bandwidth as global memory accesses and are subject to the same requirements for memory coalescing.

An automatic variable declared without the `__device__`, `__shared__` and `__constant__` memory space specifiers can either be placed in registers or in local memory by the compiler. It will be likely placed in local memory if it is one of the following:

- Arrays for which it cannot determine that they are indexed with constant quantities,
- Large structures or arrays that would consume too much register space,
- Any variable if the kernel uses more registers than available (this is also known as register spilling).

It is very straightforward to understand the second and the third points. However, the first point is being a little bit tricky since it implies that even for very small arrays it can be placed in local memory rather than in registers and most of the time we would like those small arrays to be placed in registers for better performance.

In this blog post, I would like to show an example of how the compiler decides to place an array in local memory rather than in registers and discuss the general rules that a user can follow to avoid small arrays being placed in local memory.

## CUDA Local Memory

In the following example, I created two CUDA kernels that compute the running mean of an input array given a fixed `window` size. Both of the kernels declared a local array window whose size is known at the compile time. The implementations of the two kernels are almost exactly the same except the first kernel uses a straightforward indexing to access the window array, while the second kernel uses an index that seems to be less trivial.


