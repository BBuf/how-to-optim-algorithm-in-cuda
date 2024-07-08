> 我的课程笔记，欢迎关注：https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode 

## 第四课: 计算和内存基础（基于PMPP 书的第4-5章）

![](https://files.mdnice.com/user/59/226f775a-9061-4639-81f9-56ad86a8cb67.png)

### 第4章：计算架构和调度，如何保持整个GPU繁忙

接下来2张Slides展示了一下书中对CPU，GPU结构的对比，由于这两页Slides很过时，这里就不截图了。

![](https://files.mdnice.com/user/59/31ebb230-1e25-42e9-b3f7-566eea7814fa.png)

RTX 3090有82个流式多处理器（SM, Streaming Multiprocessor），每个SM包含多个RT Core（光线追踪核心）和Tensor Core（张量核心）。所有SM共用L2缓存。


消费级/非数据中心GPU中几乎没有FP64（双精度浮点）单元。每个SM有2个FP64单元，相比128个FP32（单精度浮点）单元。

GA102 GPU实际上有168个FP64单元（每个SM两个），但图中未显示。FP64的TFLOP（每秒浮点运算次数）速率是FP32的1/64。包含少量FP64硬件单元是为了确保任何包含FP64代码的程序都能正确运行，包括FP64 Tensor Core代码。

> GA：代表 "Graphics Ampere"，指的是 NVIDIA 的 Ampere 架构。102：是这个特定 GPU 型号的数字标识符。通常，较高的数字表示更高端或更大规模的 GPU 设计。GA102 被用于多款显卡，包括 GeForce RTX 3090, RTX 3080 和一些 Quadro 系列专业卡。

从图中可以数一下，RTX 3090的完整SM个数应该是12x7=84个，但是其中2个没有启用，所以可以工作的SM个数是82。

![](https://files.mdnice.com/user/59/587238e3-22c8-4867-817c-229b02627003.png)

这张Slides描述了NVIDIA GA10x GPU架构中的流式多处理器(Streaming Multiprocessor, SM)的结构和特性：
- SM结构：
    - 4个处理单元，每个包含FP32（单精度浮点）和INT32（整数）运算单元
    - 每个处理单元有一个第三代Tensor Core
    - 寄存器文件（16,384 x 32位）
    - L0 I-Cache和Warp调度器
    - 128KB的L1数据缓存/共享内存
    - 第二代RT Core（光线追踪核心）
- 线程块分配：
    - 一个线程块被分配给一个SM
    - 每个SM最多可分配1536个线程
    - 无法控制网格中的哪个块分配到哪里（Hopper+架构可以有线程块组）
- Warp执行：
    - 4个warp或"部分warp"可以在一个周期内计算
    - 这些warp共享一条指令（Volta+架构每个线程都有程序计数器）
- 计算单元：
    - 32个FP32单元（这32个FP32单元对应一个warp的32个线程，在任何给定的时钟周期，32个FP32单元可以同时处理一个warp中的32个线程）
    - 其中16个同时支持INT32运算
- 寄存器：
    - 16k个32位寄存器在同一块上调度的任务之间共享
- 缓存和共享内存：
    - L1缓存和共享内存共享128KB硬件
    - 共享内存可以配置为0/8/16/32/64/100KB
    - L1缓存使用剩余空间（至少28KB）

![](https://files.mdnice.com/user/59/17911374-e434-40fa-b7d9-4f1d370b5969.png)

这张图解释了CUDA编程中的线程(Threads)、线程束(Warps)和线程块(Blocks)的概念和关系：

- CUDA内核启动：指定块布局（每个块中的线程数）；指定网格布局（要启动的块数）
- 一个线程块内的线程：同一块内的线程在同一个流式多处理器(SM)上并行执行（可以访问SM的共享内存）
- 除了及其新的GPU，块之间完全独立；CUDA可以自由地将块分配给SM；块的执行顺序是随机的
- 一个线程块在SM上运行时被划分为32线程的线程束；每个线程束在SM的固定处理单元上运行；同时分配给处理单元的所有线程束轮流执行，但寄存器状态保持不变（这里应该指的是线程束切换的时候可以保留寄存器状态，例如当一个线程束暂停执行让位于另一个线程束时，它的寄存器状态会被保存。当这个线程束再次获得执行时间时，它可以从之前的状态继续执行，而不需要重新初始化。）；
- 在AMD硬件和术语中，线程束称为Wavefronts，默认大小为64？
- 右侧图表展示了线程块如何分配到不同的SM上。


![](https://files.mdnice.com/user/59/b5194fcb-7a6a-4198-9b8e-8a96a2bea574.png)


这张slides解释了CUDA中线程的线性化和分组为线程束（warps）的过程。使用T(x,y,z)表示线程索引，其中x、y、z表示三个维度的索引。将多维的线程索引转换为一维的线性索引的公式为：threadId = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)。线性化后的线程被分组为32个线程一组的线程束，图底部显示了线程如何被分组成连续的线程束。

![](https://files.mdnice.com/user/59/31d7ca1e-e618-4458-810e-c7fa0445f6e6.png)

![](https://files.mdnice.com/user/59/5f8d46d3-4ed8-4ab4-8296-55d0e8400253.png)

这个kernel的目的是为3D空间中的每个点计算其在warp内的32个"邻居"的索引。它利用了CUDA的warp级别shuffle操作来高效地在线程间交换数据。输出是一个5D张量，维度为(8, 8, 8, 32, 3)，其中：

- 前三个维度(8, 8, 8)对应3D空间中的点
- 32表示每个点计算32个邻居
- 3表示每个邻居的x、y、z坐标

kernel输出的结果和上上张Slides描述是一致的。

这里需要注意下 `__shfl_sync()` 是一个在CUDA编程中用于线程间通信的内置函数，它允许同一个Warp（一组32个同时执行的线程）内的线程共享数据。这个函数实现了所谓的“shuffle”操作，即它可以从Warp内的任意线程获取数据，并将其广播给同一Warp中的其他线程。

__shfl_sync() 函数的基本语法如下：

```c++
__shfl_sync(mask, var, srcLane, width);
```

- `mask`: 是一个掩码值，用于确定哪些线程参与此操作。通常使用0xffffffff表示所有线程都参与。
- `var`: 是要共享的变量。
- `srcLane`: 指定从哪个线程（lane ID）获取数据。
- `width`: 表示shuffle操作的范围，通常与Warp大小相关联，对于标准的CUDA Warp大小（32个线程），宽度可以是1, 2, 4, 8, 16, 或32。
例如，如果在一个Warp中，你想让所有线程都得到第9个线程的`value`变量的值，你可以这样调用`__shfl_sync()`：

```c++
int sharedValue = __shfl_sync(0xffffffff, value, 9, 32);
```

这里，`sharedValue`将会被设置为第9个线程的`value`变量的值，对于所有其它线程而言。


![](https://files.mdnice.com/user/59/fd1e20b0-6640-4be3-839c-fe3be10a8ec7.png)

这张Slides解释了CUDA中的线程束分歧（Warp Divergence）现象，特别是在Pascal及之前的GPU架构中。这段代码展示了一个条件语句，根据线程ID执行不同的操作。执行执行流程为：
- 所有线程首先到达分歧点。
- 执行 A 和 B 的线程（threadIdx.x < 4）继续执行，其他线程等待。
- 然后执行 X 和 Y 的线程继续，之前的线程等待。
- 最后所有线程重新汇合执行 Z。

关键点为：
- 旧方法：线程共享程序计数器，但有一个"活动掩码"。
- 需要小心在 if 语句内部不进行线程间通信或同步（除非使用掩码）。
- 自动重新汇合：执行完分歧部分后，所有线程自动在 Z 处重新汇合。

性能影响：

- 线程束分歧会导致性能下降，因为部分线程在等待时无法执行有效工作。
- 理想情况下，同一线程束内的所有线程应执行相同的指令路径。
- 对于加载/存储指令，典型的模式（cond ? x[i] : 0f）不会导致分歧。这是因为硬件可以有效地处理这种简单的条件执行。

> 关键点为<=Pascal架构时，warp内的所有线程共享程序计数器。

![](https://files.mdnice.com/user/59/a3daada4-9206-4bfa-9899-a765fe45c31d.png)

这张Slides描述了NVIDIA Volta及之后架构中处理线程束分歧（Warp Divergence）的新方法。

执行流程：
- 所有线程首先到达分歧点。
- 执行 A 的线程（threadIdx.x < 4）继续执行。
- 同时，执行 X 的线程（threadIdx.x >= 4）也开始执行。
- B 和 Y 分别在各自的线程组中执行。
- 最后，所有线程执行 Z，但不需要显式的重新汇合。

主要改进：
- 独立程序计数器：每个线程都有自己的PC，允许更灵活的执行。
- 并行执行分歧路径：不同的执行路径可以同时进行，提高效率。
- 没有自动重新汇合：线程可以独立执行，直到自然地到达相同的指令。
- 更好的延迟隐藏：如果两个分支都涉及从DRAM加载数据，可以并行进行，提高效率。
- 更高的硬件利用率：减少了线程等待的情况。

> 注意线程分化时不再有自动重新汇合：开发者需要更加注意同步点。

![](https://files.mdnice.com/user/59/f538ee2e-210b-46d1-8efb-cda5e476ca59.png)

这张Slides展示的代码与之前相同，但在最后添加了__syncwarp()函数。关键变化和概念：

- 没有自动重新汇合：Volta架构不再自动在分支结束处重新同步线程。
- 显式同步：使用__syncwarp()函数来手动重新同步线程束。
- 线程间通信：像shuffle这样的操作也会同步参与的线程。
- 块级同步：__syncthreads()函数同步整个线程块，而不仅仅是线程束。

![](https://files.mdnice.com/user/59/21f1681e-26c9-431c-aab2-e3593110adba.png)

这张Slides展示了CUDA编程中由于循环上限不同导致的线程束分歧（Warp Divergence）情况。

![](https://files.mdnice.com/user/59/f1719439-3fff-471d-b5af-c9922dbd57b5.png)

这张Slides讨论了在CUDA编程中如何获得良好的GPU占用率（occupancy）和平衡资源使用。要点为：
- 有82个SM（流多处理器）是很好的，因为这意味着可以运行多个块。作为对比，Jetson Xavier有8个Volta SM。
- 每个SM可以调度多达1536个线程。建议块大小是512的2的幂次方（如256或512），这有利于性能优化。一些其他GPU支持2048个线程。
- 尽量避免线程束（warp）内的分歧，以便每个周期都能执行整个warp（32个线程）。
- 在Gx102（GeForce / Workstation GPUs）上，如果可能的话，避免使用FP64/INT64数据类型。
- 共享内存和寄存器资源限制了SM上可调度的线程数量。使用`__launch_bounds__ / C10_LAUNCH_BOUNDS`来建议编译器为寄存器分配线程数。注意：寄存器溢出会降低性能。
- 以前有一个Excel表格用于计算占用率，现在这个功能集成在Nsight Compute中。
- 使用`torch.cuda.get_device_properties(<gpu_num>)`来获取设备属性（如`max_threads_per_multi_processor`）。


![](https://files.mdnice.com/user/59/dbeb96ff-b639-41ca-933f-b7c98a12608b.png)


![](https://files.mdnice.com/user/59/feba35b3-c412-4b51-bd82-348621035c5f.png)


### 第5章：内存架构和数据局部性（也是获得fast kernel的基础）
