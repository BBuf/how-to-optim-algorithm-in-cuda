## CUDA-MODE 第一课课后实战

## Nsight Compute简介

Nsight Compute是一个CUDA kernel分析器，它通过硬件计数器和软件收集指标。它使用内置的专业知识来检测kernel常见的性能问题并指出发生这些问题的位置并给出一些解决方法的建议。这一内置规则集和指南就是我们所说的Guided Analysis。下面就结合Lecture1的例子来深入了解下Nsight Compute。

### Nsight Compute Profile流程

这里就使用 Lecture 1讲义中的 Triton 实现的矩阵开方代码使用Nsight Compute进行Profile，看一下当前Nsight Compute可以帮助我们获得哪些关键信息。Nsight Compute安装包在 https://developer.nvidia.com/tools-overview/nsight-compute/get-started 可以获得。Nsight Compute提供了Windows/Linux/MacOS等多种操作系统的支持，我们可以根据自己的操作系统选择合适的版本进行安装。我这里选择的方式就是分别在Linux服务器和本地Mac上进行安装，在服务器上使用Nsight Compute Profile之后把生产的`xxx.ncu-rep`文件在本地Mac上用Nsight Compute打开。


![](https://files.mdnice.com/user/59/5292a5a4-3cf6-49e8-88bd-b0af96ed851b.png)


Profile的代码如下所示，命名为 `triton_square.py`：

```python
# Adapted straight from https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
import triton
import triton.language as tl
import torch

# if @triton.jit(interpret=True) does not work, please use the following two lines to enable interpret mode
# import os
# os.environ["TRITON_INTERPRET"] = "1"

@triton.jit
def square_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # The rows of the softmax are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))

    square_output = row * row
    
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, square_output, mask=col_offsets < n_cols)


def square(x):
    n_rows, n_cols = x.shape
    # The block size is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    # Allocate output
    y = torch.empty_like(x)
    # Enqueue kernel. The 1D launch grid is simple: we have one kernel instance per row o
    # f the input matrix
    square_kernel[(n_rows, )](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


torch.manual_seed(0)
x = torch.randn(1823, 781, device='cuda')
y_triton = square(x)
y_torch = torch.square(x)
assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)
```

Profile的命令为：

```bash
../NVIDIA-Nsight-Compute-2024.2/ncu --set full -o matrix_square python3 triton_square.py
```

当出现如下界面并且生成了`maxtrix_square.ncu-rep`时就说明Profile程序成功运行，我们就可以分析这个文件了。


![](https://files.mdnice.com/user/59/77c1458e-7a16-462b-bc4b-f5300940026d.png)

### Nsight Compute Profile结果分析

我们不仅仅可以利用Nsight Compute对程序进行Profile，还可以通过NSight Compute来学习到CUDA的编程模型以及内存模型等等。

#### Summary部分

Nsight Compute打开`maxtrix_square.ncu-rep`后的第一页长这样：

![](https://files.mdnice.com/user/59/87e43127-f8f3-44fd-ae35-30c01c9916f1.png)


1. 第一个红框部分是顶部的工具栏，Result表示当前选定的Kernel名字是`605 - square_kernel_0d1d234`。Size表示的是这个Kernel的启动参数，即Grid Size和Block Size。Time, Cycles, GPU分 别表示执行时间是 28.99 us，周期数是28,327，使用的GPU是NVIDIA GeForce RTX 3080 Ti Laptop GPU。SM Frequency, Process, Attributes分别表示 SM 频率是973.60 MHz，进程ID是[400]，运行的程序是python3.10。

2. 第二个绿色框部分有多种选择，比如Summary，Details，Source等，一般只关心前3种选择。

3. 第三个白色框部分表示这个Summary选择下可以看到的Profile程序里面有哪些Kernel，可以通过光标来选择查看具体的Kernel，比如从Function Name这一栏可以看到当前选择的是`605 - square_kernel_0d1d234`这个Kernel。列一下这个表格的所有列：
- ID: 每个函数的唯一标识符。
- Estimated Speedup: 估计的加速比，表示如果优化这个函数可能带来的速度提升。
- Function Name: 函数的名称。
- Demangled Name: 去掉修饰符的函数名称。
- Duration: 函数执行时间（以ns为单位）。
- Runtime Improvement: 估计的运行时间提示（以ns为单位），表示如果优化这个函数可能带来的运行时间提升。
- Compute Throughput: 计算吞吐量。
- Memory Throughput: 内存吞吐量。
- Registers: 每个线程使用的寄存器数量。
- GridSize：kernel启动的网格大小
- BlockSize：每个Block的线程数
- Cycles：指令周期。

当我们鼠标悬停在Compute Throughput这一列时，会显示如下界面：


![](https://files.mdnice.com/user/59/8b20a3fe-6eb7-4fa0-92d4-51f9d91c73f1.png)

SM 吞吐量假设在 SMSPs 间负载平衡理想的情况下
（此吞吐量指标表示在所有子单元实例的经过周期内达到的峰值持续率的百分比）
 
sm：流式多处理器（Streaming Multiprocessor）以32个线程为一组执行kernel，称为warp。
warp 进一步分组为合作线程数组（CTA），在CUDA中称为块。
CTA 的所有warp在同一个SM上执行。
CTA 在其线程间共享各种资源，例如共享内存。

SMSPs：每个SM被划分为四个处理块，称为SM子分区。
SM子分区是SM上的主要处理单元。
一个子分区管理一个固定大小的warp池。

当我们鼠标悬停在Memory Throughput这一列时，会显示如下界面：


![](https://files.mdnice.com/user/59/9ac644bb-d99f-4962-bbcd-c3a695326e5e.png)

计算内存管道吞吐量
（此吞吐量指标表示在所有子单元实例的经过周期内达到的峰值持续率的百分比）

gpu：整个图形处理单元。

同样，当我们鼠标悬停在 #Registers 这一列时，会显示如下界面：

![](https://files.mdnice.com/user/59/5ffd3b2d-e871-439f-a9b9-1f1be0051d0b.png)

每个线程分配的寄存器数量。

寄存器：每个子分区有一组32位寄存器，由硬件以固定大小的块分配。

线程：在GPU的一个SM单元上运行的单个线程。

最后，对Cycles上的界面的翻译为：


![](https://files.mdnice.com/user/59/2a43d0e8-e530-409c-aa5e-fd0749d88ebd.png)


在 GPC 上经过的周期数
（此计数器指标表示所有子单元实例中的最大值）

gpc：通用处理集群（General Processing Cluster）包含以 TPC（纹理处理集群）形式存在的 SM、纹理和 L1 缓存。
它在芯片上被多次复制。


> 从这里展示的细节，我们可以更加详细的了解到到CUDA的编程模型。

4. 第4个蓝色框部分都是根据目前kernel的指标给出的粗浅调优建议，比如这里第一条就是因为活跃wave太低给出的调整grid_size和block_size的建议。第二条是计算的理论occupancy（100.0%）和实测的实际occupancy占用（76.0%）之间的差异可能是由于 kernel 执行期间的warp调度开销或工作负载不平衡导致的。在同一kernel 的不同块之间以及块内的不同 warps 之间都可能发生负载不平衡。 第三条则是需要验证内存访问模式是否最优，是否需要使用Shared Memory。

#### Details部分

##### SOL部分

首先是 GPU Speed Of Light Throughput部分，它通常位于Details部分的顶部。它清晰的描述了GPU资源的利用情况。在下面的截图中，我们同样可以通过鼠标悬停的方式去看每个指标的细节，本文就不再赘述了。

![](https://files.mdnice.com/user/59/4e4386fd-c7f5-4c67-ac04-d27e8c6dc7b5.png)

从这个结果可以看出：

- 内存吞吐量(83.56%)远高于计算吞吐量(15.55%)，表明这可能是一个内存密集型任务。
- L1/TEX和L2缓存吞吐量相对较低，可能存在优化空间。
- DRAM吞吐量与总体内存吞吐量相同，说明主要的内存操作直接与DRAM交互。


##### Memory Workload Analysis 部分

![](https://files.mdnice.com/user/59/f124f60e-26b2-4fdb-8b43-798cd76cfe4d.png)


从上到下对每个部分解析一下：

1. 顶部性能指标

Detailed analysis of the memory resources of the GPU. Memory can become a limiting factor for the overall kernel performance when fully utilizing the involved hardware units (Mem Busy), exhausting the available communication bandwidth between those units (Max Bandwidth), or by reaching the maximum throughput of issuing memory instructions (Mem Pipes Busy). Detailed chart of the memory units. Detailed tables with data for each memory unit.

> 翻译：GPU内存资源的详细分析。当完全利用相关硬件单元（Mem Busy），耗尽这些单元之间可用的通信带宽（最大带宽），或达到发出内存指令的最大吞吐量（内存管道忙碌度）时，内存可能成为整体kernel性能的限制因素。内存单元的详细图表。每个内存单元的详细数据表如下。

- Memory Throughput: 387.77 Gbyte/s
- L1/TEX Hit Rate: 31.49%
- L2 Hit Rate: 53.87%
- L2 Compression Success Rate: 0%
- Mem Busy: 46.04%
- Max Bandwidth: 83.56%
- Mem Pipe Busy: 13.55%
- L2 Compression Ratio: 0


