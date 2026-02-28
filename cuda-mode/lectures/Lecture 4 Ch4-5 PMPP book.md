> 我的课程笔记，欢迎关注：https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode 

## 第四课: 计算和内存基础（基于PMPP 书的第4-5章）

![](https://files.mdnice.com/user/59/226f775a-9061-4639-81f9-56ad86a8cb67.png)

### 第4章：计算架构和调度，如何保持整个GPU繁忙

接下来2张Slides展示了一下书中对CPU，GPU结构的对比，由于这两页Slides很过时，这里就不截图了。

![](https://files.mdnice.com/user/59/31ebb230-1e25-42e9-b3f7-566eea7814fa.png)

RTX 3090有82个流式多处理器（SM, Streaming Multiprocessor），每个SM包含多个RT Core（光线追踪核心）和Tensor Core（张量核心）。所有SM共用L2缓存。


消费级/非数据中心GPU中几乎没有FP64（双精度浮点）单元。每个SM有2个FP64单元，相比128个FP32（单精度浮点）单元。

GA102 GPU实际上有168个FP64单元（每个SM两个），但Slides中未显示。FP64的TFLOP（每秒浮点运算次数）速率是FP32的1/64。包含少量FP64硬件单元是为了确保任何包含FP64代码的程序都能正确运行，包括FP64 Tensor Core代码。

> GA：代表 "Graphics Ampere"，指的是 NVIDIA 的 Ampere 架构。102：是这个特定 GPU 型号的数字标识符。通常，较高的数字表示更高端或更大规模的 GPU 设计。GA102 被用于多款显卡，包括 GeForce RTX 3090, RTX 3080 和一些 Quadro 系列专业卡。

从Slides中可以数一下，RTX 3090的完整SM个数应该是12x7=84个，但是其中2个没有启用，所以可以工作的SM个数是82。

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

![](https://files.mdnice.com/user/59/cc6a5fc0-32a8-46e5-8e0d-281710f48f7a.png)

这张Slides讨论了PyTorch程序如何分配其运行时间，以及一些优化建议。

PyTorch程序的时间分配（高层次概述）：

- Python处理
- 数据"管理开销"（如分配Tensor结构等）
- 数据获取（I/O）- 建议在深入GPU优化前检查这部分
- GPU计算，包括：
    - 固定成本（如内核启动等）
    - 内存访问（读取输入/写入结果）- 当前章节（第5章）的重点
    - "实际"计算（FLOPs）- 占用率是关键，在第4章已经讨论

Thomas的经验法则：

- 如果GPU利用率（在nvidia-smi中）未接近100%，应优先改进数据获取等方面
- 当处理的Tensor只有几百个元素时，"Python很慢"，数据管理开销占比为个位数百分比
- 算法选择也很重要（后续章节会讨论并行算法）

![](https://files.mdnice.com/user/59/15c38f04-e8dc-42d9-8dd2-1bcf3aa84a1f.png)


这张Slides讨论了内存访问作为性能瓶颈的问题：
- Eager PyTorch对每个操作都执行"加载输入、计算、存储输出"的过程。
- 如果能够合并内核，执行"加载输入、多次计算、存储输出"，效率会更高。
- PyTorch的优化焦点：
    - 长期以来，PyTorch一直关注这个问题。
    - PyTorch JIT的原始目的是将elementwise操作融合到一个内核中，例如提高LSTM接近CuDNN的性能。
    - 第二代PyTorch JIT fusers增加了收缩操作等（NVFuser在https://github.com/NVIDIA/Fuser 上持续改进）。
    - 当前的inductor/Triton基础优化也部分针对这点，但支持更复杂的操作。
- 内存访问优化也是flash attention的核心组成部分。图片右侧展示了内存层次结构，包括带宽和内存大小。图来自FLash Attention的Paper。

![](https://files.mdnice.com/user/59/36579dc8-a90f-4d62-ae11-9587f8ccc357.png)

接着举了这个GeLU fuse前后执行时间对比的例子，说明我们把所有的elementwise操作fuse之后的有效性。

![](https://files.mdnice.com/user/59/2208e6cf-5317-456a-b9ab-8f38d32fb266.png)

这里还展示了一下如何使用CUDA手动编写这个fuse cuda kernel。

![](https://files.mdnice.com/user/59/7744f25a-052d-4450-a71a-464683a2ee81.png)

这张Slides讨论了内存访问和计算在图像处理中的性能影响：

- RGB转灰度图示例：
    - 每个像素需要加载3字节
    - 计算I（在32位整数中进行1次乘法和1次加法）
    - 计算5次操作（3次乘法，2次加法，理想情况下在32位中进行）+ 数据转换
    - 存储1字节
- 性能预期（基于2048 x 2048图像和RTX3090显卡）：
    - NVIDIA列出的内存带宽约为900GB/s，传输4*4M字节需要约18μs（"光速"）
    - 计算能力：35.6FP32 TFLOP/s或16.8 Int32 TFLOP/s，约需2μs（慷慨估计）
    - 内核启动时间：约3μs（使用空内核测量）
    - 注意：使用32位或16位时要考虑元素大小

- 实际测量结果：
    - 内核执行时间（使用"f"作为常量）：27μs，约为理论可能性的74%

注意：作者创建了一个"out"函数来分离内存分配，只要使用缓存分配器，这个过程相对较快。对齐有助于提高性能（建议尝试带stride的copy内核）

这里说的27us就是 https://github.com/cuda-mode/lectures/blob/main/lecture_004/cuda-mode-session-4.ipynb 这里的第一个cuda kernel的输出，如下图红色框所示。

![](https://files.mdnice.com/user/59/9cd24fb1-f4f0-41a0-bb42-50ba6d609834.png)


![](https://files.mdnice.com/user/59/62bf05d4-28de-45ca-9ab8-39b25091b2ab.png)

这张Slides介绍了带有延迟隐藏的屋顶线模型（Roofline Model with latency hiding）。
这是一个性能分析模型，用于评估计算密集型应用在特定硬件上的性能上限。横轴表示计算密度（Computational intensity），单位是FLOP/B（每字节内存传输的浮点运算数）。纵轴表示计算吞吐量（Computational throughput），单位是GFLOP/s（每秒十亿次浮点运算）。

一些概念：
- 计算密度：FLOP/Byte of memory transfer。
- 延迟隐藏：在SM（Streaming Multiprocessor）上使用多个warps，允许一些warps在计算时其他warps等待
- 峰值吞吐量（Peak throughput）：硬件能达到的最大计算速度
- 内存带宽（Peak bandwidth）：内存传输的最大速度

A1、A2、A3：代表不同算法或优化的性能点。越接近屋顶线的点，表示性能越接近硬件极限。对于内存受限区域：优化内存访问模式，减少数据传输。对于计算受限区域：提高计算效率，如使用更高效的算法。此外，通过并行执行多个warps，可以有效隐藏内存访问延迟，使得实际性能曲线更接近理论上限。

![](https://files.mdnice.com/user/59/4c46f94b-3cdd-46d0-88d0-14496b12853e.png)

这张Slides描述了CUDA设备内存模型的概览：

- 设备代码（Device code）可以访问的内存类型：
    - 每线程寄存器（R/W per-thread registers）
    - 每线程本地内存（R/W per-thread local memory）
    - 每块共享内存（R/W per-block shared memory）
    - 每网格全局内存（R/W per-grid global memory）
    - 只读每网格常量内存（Read only per-grid constant memory）
- 主机代码（Host code）可以：
    - 向/从每网格全局内存和常量内存传输数据
- 设备（Device）网格结构：
    - 由多个块（Block）组成
    - 每个块内有共享内存（Shared Memory）
    - 每个块内有多个线程（Thread）
    - 每个线程有自己的寄存器（Registers）
- 内存层次：
    - 全局内存（Global Memory）：所有块和线程都可访问
    - 常量内存（Constant Memory）：所有块和线程都可读取
    - 共享内存（Shared Memory）：块内的线程可共享
    - 寄存器（Registers）：每个线程私有

纹理内存（Texture memory）：Slides中未显示，因为这个教材未涵盖其用途。

![](https://files.mdnice.com/user/59/72559adb-e67e-4eea-b3c3-466762bf18d2.png)

- 数组以外的自动变量：Register（寄存器），Thread（线程作用域），Grid（网格生命周期）
- 自动数组变量：Local（本地内存），Thread（线程作用域），Grid（网格生命周期）
- SharedVar：Shared（共享内存），Block（块作用域），Grid（网格生命周期）
- GlobalVar：Global（全局内存），Grid（网格作用域），Application（应用程序生命周期）
- ConstVar：Constant（常量内存），Grid（网格作用域），Application（应用程序生命周期）


![](https://files.mdnice.com/user/59/7d78b5f9-628c-4ead-8b62-7b0a6c85febd.png)

这张SLides讨论了为什么在某些计算操作中使用分块（Tiling）技术，并展示了内存层次结构。
- Tiling（分块）的原因：
    - 在矩阵乘法（Matmul）中，每个输出使用2n个输入（一共n^2个输出）。
    - 每个输入被使用n次，如果每次都从主内存中naive地读取n次，会非常低效。
    - 解决方案：尝试重用参数（try to reuse param）。
- 应用场景：
    - 类似的情况也出现在卷积（Convolution）和FlashAttention等操作中。
- 内存层次结构（Memory Hierarchy）和特点：
    - GPU SRAM（静态随机存取内存）：带宽19 TB/s，容量20 MB
    - GPU HBM（高带宽内存）：带宽1.5 TB/s，容量40 GB
    - Main Memory（主内存，CPU DRAM）：带宽12.8 GB/s，容量>1 TB
    - 从上到下，内存容量逐渐增大，但访问速度（带宽）逐渐降低。
    - Slides中提到这个内存层次结构来自Dao等人的Flash Attention论文。

总的来说，这里解释了为什么在某些计算密集型操作中使用分块技术很重要。通过重用数据和利用更快的内存层（如GPU SRAM），可以显著提高计算效率。
同时，Slides中展示的内存层次结构清楚地说明了不同级别内存之间在速度和容量上的权衡，这进一步强调了优化内存访问模式的重要性。

![](https://files.mdnice.com/user/59/4737b84e-265a-45d8-89b6-01be99556b7f.png)


这张Slides解释了矩阵乘法中的分块(Tiling)技术，要点是：
- 将输出和输入矩阵分割成"tiles"，例如16x16的小块。
- 每个输出tile依赖于2n/TILE_SIZE个大小为TILE_SIZE*TILE_SIZE的输入tile。
- 总共有(n/TILE_SIZE)²个tile。
- 每个输入只需从主内存读取n/TILE_SIZE次。
- 需要将输入tile存储在共享内存(shmem)中。这样block中的各个线程可以在TILE_SIZE次计算中使用这些数据。
- 最简单的设置是使用TILE_SIZE²个线程。

这张图中在A矩阵的行上画了2个连续的双向箭头可能会给人误解为n=BLOCK_SIZE*2，我感觉这里是画错了，以下面的代码实现为准。

下面这张图展示了普通的矩阵乘CUDA实现：

![](https://files.mdnice.com/user/59/67a478fb-eccd-44ac-8c69-74ecb1a70cd0.png)

耗时情况是：934 µs ± 1.42 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

下面的代码则是对上面Slides中矩阵分块的实现：

```python
cuda_src = cuda_begin + r"""
constexpr int TILE_SIZE = 16;

__global__ void tiled_matmul_kernel(float* out, float* M, float* N, int h, int w, int k) {
  __shared__ float M_tile[TILE_SIZE][TILE_SIZE];
  __shared__ float N_tile[TILE_SIZE][TILE_SIZE];
  
  // idxes into tile
  int ir = threadIdx.y;
  int ic = threadIdx.x;
  
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;

  // note: cannot just exit if we want to do padding!
  
  float res = 0.0f;
  for (int K_tileidx = 0; K_tileidx < (k + TILE_SIZE -1) / TILE_SIZE; K_tileidx++) {
    // note how threadIdx.x is the fastes moving bit --> coalesced memory access
    M_tile[ir][ic] = (((r < h) && (K_tileidx * TILE_SIZE + ic < k)) ? M[r * k + K_tileidx * TILE_SIZE + ic] : 0.f);
    N_tile[ir][ic] = ((((K_tileidx * TILE_SIZE + ir) < k) && (c < w)) ? N[(K_tileidx * TILE_SIZE + ir) * w + c] : 0.f);
    //M_tile[ir][ic] = M[r * k + K_tileidx * TILE_SIZE + ic];
    //N_tile[ir][ic] = N[(K_tileidx * TILE_SIZE + ir) * w + c];
    __syncthreads();
    for (int idx = 0; idx < TILE_SIZE; idx++) {
       res += M_tile[ir][idx] * N_tile[idx][ic];
    }
    __syncthreads(); // important! (why?)
  }
  if ((r < h) && (c < w)) {
    out[r * w + c] = res;
  }
}

torch::Tensor tiled_matmul(const torch::Tensor& m, const torch::Tensor& n) {
    CHECK_INPUT(m); CHECK_INPUT(n);
    int h = m.size(0);
    int w = n.size(1);
    int k = m.size(1);
    TORCH_CHECK(k==n.size(0), "Size mismatch");
    //TORCH_CHECK((k % TILE_SIZE == 0) && (h % TILE_SIZE == 0) && (w % TILE_SIZE == 0), "Padding not done");
    auto output = torch::empty({h, w}, m.options());

    dim3 tpb(TILE_SIZE, TILE_SIZE);
    dim3 blocks(cdiv(w, tpb.x), cdiv(h, tpb.y));
    tiled_matmul_kernel<<<blocks, tpb>>>(
        output.data_ptr<float>(), m.data_ptr<float>(), n.data_ptr<float>(), h, w, k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

"""
cpp_src = """
torch::Tensor tiled_matmul(const torch::Tensor& m, const torch::Tensor& n);
"""

tiled_matmul_module = torch.utils.cpp_extension.load_inline(
    "test_ext_tiled_matmul", cpp_src, cuda_src, 
    functions=['tiled_matmul'], extra_cuda_cflags=['--ptxas-options=-v'], verbose=True)
```

耗时情况为：707 µs ± 6.36 µs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

这个Cuda Kernel实现比较简单，这里不再赘述。

![](https://files.mdnice.com/user/59/d22185d6-5579-459d-b60c-dfa832128e0b.png)


这是第4章和第5章的总结，列出了关于GPU编程的关键要点。
- GPU通过线程(threads)、束(warps)和块(blocks)来组织计算。
- 尽可能充分利用硬件（提高占用率），平衡各种瓶颈。
- 避免线程分化，以提高性能。
- 使用roofline模型和"理论最大速度"来分析性能。
- 尽量减少对全局内存的读写操作。
- 下一章将讨论连续和对齐的全局内存位置的读写（合并内存访问）。

