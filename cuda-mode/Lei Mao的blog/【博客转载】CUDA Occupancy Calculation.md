> 原地址：https://leimao.github.io/blog/CUDA-Occupancy-Calculation/ ，来自Lei Mao，已获得作者转载授权。

# CUDA 占用率计算

## 简介

占用率是每个SM上活动线程束数量与可能活动线程束最大数量的比值。

更高的占用率并不总是等同于更高的性能——有一个阈值，超过该阈值，额外的占用率不会提高性能。然而，低占用率总是会干扰隐藏内存延迟的能力，导致性能下降。

在这篇博客文章中，我想讨论CUDA占用率计算。

## CUDA 占用率计算

### Excel 占用率计算器

虽然基于Excel的占用率计算器已被弃用，但我们仍然可以使用它来计算计算能力8.6及以下的占用率。

![](https://files.mdnice.com/user/59/a0a538ea-2287-4a64-8ac3-4990e9ff249e.jpg)

### GPU 计算能力的物理限制

从Excel表格中的"GPU计算能力的物理限制"部分，我们可以看到，例如，在计算能力7.0的设备上，每个SM有65,536个32位寄存器，最多可以同时驻留2048个线程（64个线程束 × 每个线程束32个线程）。在计算能力7.0的设备上，寄存器分配会向上舍入到每个块最接近的256个寄存器。线程束分配粒度是4。

这些是计算占用率的关键因素。

### 占用率计算示例

让我们手动计算一些示例的占用率。

例如，在计算能力7.0的设备上，考虑一个使用128线程块、每个线程使用37个寄存器的 kernel。

我们从"GPU计算能力的物理限制"中知道，对于计算能力7.0，可能活动线程束的最大数量是64。

一个线程束所需的寄存器数量是

$⌈\frac{37 × 32}{256}⌉ × 256 = 1280$

其中32是每个线程束的线程数，256是寄存器分配单元大小。

考虑到线程束分配粒度，每个SM的最大活动线程束数量是

$⌊\frac{65536/1280}{4}⌋ × 4 = 48$

因为一个128线程块包含128/32 = 4个线程束，我们最多可以运行48/4 = 12个线程块。因此，占用率是12 × 4/64 = 75%。

考虑另一个示例，在计算能力7.0的设备上，一个使用320线程块、每个线程使用37个寄存器的 kernel。

因为一个320线程块包含320/32 = 10个线程束，我们最多可以运行48/10 = 4个线程块。因此，占用率是10 × 4/64 = 63%。

这些手动计算的占用率可以使用基于Excel的占用率计算器进行验证。

### 寄存器数量

最后，我们可以使用`nvcc`的`--ptxas-options=-v`选项获取每个 kernel中每个线程使用的寄存器数量。例如：

```shell
$ wget https://raw.githubusercontent.com/NVIDIA-developer-blog/code-samples/master/series/cuda-cpp/overlap-data-transfers/async.cu
$ nvcc async.cu -o async --ptxas-options=-v
ptxas info    : 24 bytes gmem
ptxas info    : Compiling entry function '_Z6kernelPfi' for 'sm_52'
ptxas info    : Function properties for _Z6kernelPfi
    32 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 22 registers, 332 bytes cmem[0], 48 bytes cmem[2]
```

## 结论

手动计算占用率有时很繁琐且容易出错。然而，我们可以使用现有的工具，如基于Excel的占用率计算器，来进行计算。

## 参考文献

- CUDA Occupancy Calculator(https://docs.nvidia.com/cuda/archive/11.6.2/cuda-occupancy-calculator/index.html)
- CUDA Best Practices Guide - Calculating Occupancy(https://docs.nvidia.com/cuda/archive/11.6.2/cuda-c-best-practices-guide/index.html#calculating-occupancy)


