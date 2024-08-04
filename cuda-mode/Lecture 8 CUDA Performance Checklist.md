> 我的课程笔记，欢迎关注：https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode 

## CUDA-MODE课程笔记 第8课: CUDA性能检查清单

![](https://files.mdnice.com/user/59/8f9a0665-871e-4c5a-ac67-86bfa79f00c7.png)


这节课实际上算是[CUDA-MODE 课程笔记 第一课: 如何在 PyTorch 中 profile CUDA kernels](https://mp.weixin.qq.com/s/owF7AFR61SLrOosUPdZPQQ) 这节课更细节的讲解。另外关于nsight compute相关指标细节解释可以参考 [CUDA-MODE 第一课课后实战（上）](https://mp.weixin.qq.com/s/9XeJPWUsKTaMU2OdPkL-OQ)，
[CUDA-MODE 第一课课后实战（下）](https://mp.weixin.qq.com/s/FCqnQESCQTtlqCG_BSLulA) 这两篇笔记。

将GPU用于计算，我们最关心的肯定是性能。比较幸运的是，当我们掌握一些性能优化技巧之后它往往会经常被用到。这节课将会系统的介绍这些性能优化技巧。

![](https://files.mdnice.com/user/59/56f3a5e1-c0e2-4182-aa17-4d03c9f3ef6d.png)

本节课的课件和代码都在 https://github.com/cuda-mode/lectures 开源，我们可以用nvcc编译lecture8下面的cu文件，然后使用ncu进行profile。此外，这里的方法遵循了  https://arxiv.org/pdf/1804.06826.pdf 这篇论文的风格。up主也非常推荐大家阅读下这篇论文，用claude 3.5问了一下paper的主要内容，如下截图：

![](https://files.mdnice.com/user/59/e00c198b-2e70-466f-be35-b4af3d324ddd.png)

可以看到这篇论文主要是对Volta架构的GPU的架构细节进行分析，对性能优化是很重要的。

![](https://files.mdnice.com/user/59/828daf13-e5e3-4fbc-8370-49e59497ae74.png)

这张Slides从物理学的角度分析了下SRAM和DRAM的区别：
- DRAM由1个晶体管和1个电容器构成；SRAM: 由6个晶体管构成
- SRAM 比 DRAM 更快，但也更贵；SRAM 占用更多空间且发热更多；
实际上SRAM就对应了GPU的Shared Memory，而DRAM对应的则是Shared Memory。

这里的youtube链接作者Bill是NVIDIA的首席科学家，他解释了很多为什么GPU设计成现在这个样子，并且由浅入深，基础细节讲的非常清楚。

![](https://files.mdnice.com/user/59/f676c3c5-544b-41b3-9203-c464b0b497e0.png)

从软件的角度，我们有以下可以优化GPU性能的技巧，除了2个斜体部分都在PMPP书中有提到。

![](https://files.mdnice.com/user/59/51bdc9fd-4cc6-41a3-b558-40790a80fc5c.png)

这里的"性能检查清单"（Performance checklist），列出了一系列优化GPU程序性能的策略和技巧：
- 合并全局内存访问（Coalesced Global Memory Access）
- 最大化占用率（Maximize occupancy）
- 理解是内存受限还是计算受限（Understand if memory or compute bound）
- 最小化线程分化（Minimize control divergence）
- Tiling以更好的重用数据（Tiling of reused data）
- 私有化（Privatization）
- Thread Coarsening
- 使用更好的数学方法重写算法（Rewrite your algorithm using better math）

这里的Privatization指的应该就是Shared Memory优化内存读取，而Coarsening大概指的就是一个线程应该完成多少任务，一般情况下我们让一个线程完成的任务尽量少，但是在Compute Bound情况下，让一个线程执行更多的工作可以让程序运行得更快。最后一点的经典例子就是Flash Attention。

![](https://files.mdnice.com/user/59/ea47a608-f087-4625-bc0f-25afb0ffe301.png)

这张Slides讲述了GPU内存访问延迟的相关内容，下面的Figure3和表格都来自 https://arxiv.org/pdf/2208.11174 ，这个表格（Table IV），列出了不同类型内存的访问延迟（以时钟周期为单位）：

- 全局内存（Global memory）: 290 cycles
- L2 缓存: 200 cycles
- L1 缓存: 33 cycles
- 共享内存（Shared Memory）: 读取23 cycles，写入19 cycles

> 我后面也找到了这个paper里面做micro benchmark的代码：https://www.stuffedcow.net/research/cudabmk?q=research/cudabmk ，后面如果有空继续阅读下这篇 paper 以及测试代码。

![](https://files.mdnice.com/user/59/c46250e5-8aa3-4af3-8a8a-59e9eb4502eb.png)

这张Slides讲述了延迟（latency）在计算机系统中的重要性和一些相关概念。
- 标题 "It's the latency stupid" 强调了延迟的重要性。
- 吞吐量（Throughput）和延迟（Latency）的对比：
    - 吞吐量容易提高，但延迟却很难降低。
    - 举例说明：即使你可以并行使用80条电话线，每条线传输一个比特，但100毫秒的延迟仍然存在。
- 量化（Quantization）技术：
    - 用于减少数据包大小的一种方法。
    - 例如，Bolo（可能是某个系统或协议）尽可能使用字节（byte）而不是16位或32位字来减少数据包大小。
- 底部提供了一个网址链接，包含更多关于这个话题的详细讨论。

![](https://files.mdnice.com/user/59/8bf4fb12-76f6-4fc1-a663-0df2ebbae546.png)

这张Slides开始介绍内存合并（Memory Coalescing）的概念。我们无法减少延迟，但可以通过读取连续的内存元素来隐藏延迟。Slides建议在进行案例研究时要关注以下三个方面：
- DRAM Throughput（DRAM吞吐量）
- Duration（持续时间）
- L1 cache throughput（L1缓存吞吐量）

这里说的内存合并的案例就是 https://github.com/cuda-mode/lectures/blob/main/lecture_008/coalesce.cu 这里所展示的。代码如下：

```c++
#include <iostream>
#include <cuda_runtime.h>

__global__ void copyDataNonCoalesced(float *in, float *out, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        out[index] = in[(index * 2) % n];
    }
}

__global__ void copyDataCoalesced(float *in, float *out, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        out[index] = in[index];
    }
}

void initializeArray(float *arr, int n) {
    for(int i = 0; i < n; ++i) {
        arr[i] = static_cast<float>(i);
    }
}

int main() {
    const int n = 1 << 24; // Increase n to have a larger workload
    float *in, *out;

    cudaMallocManaged(&in, n * sizeof(float));
    cudaMallocManaged(&out, n * sizeof(float));

    initializeArray(in, n);

    int blockSize = 128; // Define block size
    // int blockSize = 1024; // change this when talking about occupancy
    int numBlocks = (n + blockSize - 1) / blockSize; // Ensure there are enough blocks to cover all elements

    // Launch non-coalesced kernel
    copyDataNonCoalesced<<<numBlocks, blockSize>>>(in, out, n);
    cudaDeviceSynchronize();

    initializeArray(out, n); // Reset output array

    // Launch coalesced kernel
    copyDataCoalesced<<<numBlocks, blockSize>>>(in, out, n);
    cudaDeviceSynchronize();

    cudaFree(in);
    cudaFree(out);

    return 0;
}
```

这里段程序比较简单，用于演示内存合并（Memory Coalescing）的概念和其对性能的影响。它主要做了以下事情：
- 定义了两个CUDA kernel：
    - copyDataNonCoalesced kernel：非合并内存访问模式，以非连续的方式读取输入数组（使用 (index * 2) % n 作为索引），这种访问模式会导致非合并的内存访问，可能降低性能。
    - copyDataCoalesced kernel：合并内存访问模式，以连续的方式读取输入数组（直接使用 index 作为索引），这种访问模式允许合并内存访问，可以提高性能。
- 主函数：
    - 分配统一内存（Unified Memory）用于输入和输出数组，初始化输入数组。
    - 设置CUDA网格和块的大小，分别运行非合并和合并的kernel，在每次kernel执行后使用 cudaDeviceSynchronize() 确保GPU操作完成。


接着使用`nvcc -o benchmark coalesce.cu`来编译程序，然后执行`ncu benchmark`来Profile程序。

![](https://files.mdnice.com/user/59/dc2d062f-e5b0-467c-ac9f-507278a74526.png)

对于copyDataNonCoalesced kernel来说，DRAM内存吞吐量大约是89%，L1 Cache的吞吐量是30%，kernel的执行时间是764us。

![](https://files.mdnice.com/user/59/1a9d18ca-9a2c-4ff8-a405-147cc9892a04.png)

对于copyDataCoalesced kernel来说，L1 Cache的吞吐量大约是37%，DRAM内存吞吐量是82%，执行时间是558us。

我们可以看到合并内存访问的kernel是有明显的性能提升的。可以预见，随着输入数据量的增大合并内存访问的优势会更明显。ncu的结果里面还提示计算的理论occupancy（100.0%）和实测的实际occupancy占用（77%）之间的差异可能是由于 kernel 执行期间的warp调度开销或工作负载不平衡导致的。在同一kernel 的不同块之间以及块内的不同 warps 之间都可能发生负载不平衡。把上面程序中的`int blockSize = 128`改成`int blockSize = 1024`再次用ncu profile，可以发现occupancy提升到了85.94%。

![](https://files.mdnice.com/user/59/e4100f32-4ecf-4fdf-a2f5-123d25cb088c.png)

这张Slides讨论了GPU中的占用率（Occupancy）问题，主要内容如下：
- 两种quantization问题：
    - a) Tile quantization：矩阵维度不能被线程块Tile大小整除。
    - b) Wave quantization：Tile总数不能被GPU上的SM（流多处理器）数量整除。
- 性能图表比较和分析：
    - 左图(a)：cuBLAS v10 上 NN GEMM 的性能
    - 右图(b)：cuBLAS v11 上 NN GEMM 的性能
    - 两图都是在 M = 1024, N = 1024 的矩阵维度下进行的测试
    - 左图(a)显示性能呈现明显的阶梯状，有大幅波动。
    - 右图(b)显示性能波动较小，整体更加平滑。
我们可以看到cuBLAS v11 可能采用了更好的调度策略或优化技术，减少了由于Tile和Wave Quantization 导致的性能波动。

![](https://files.mdnice.com/user/59/7556b215-7af4-4174-a2b4-ac96a9cf1617.png)

