> CUTLASS GEMM模板中有大量可以调节和设置的模板参数，这些参数的设置会高度影响Kernel性能。这个分享将为大家介绍从2.x到3.x，CUTLASS kernel实现的变化，这些参数的原理和选择的最佳实践。Slides来自BiliBili NVIDIA英伟达频道 上传的《TensorRT-LLM中的 Quantization GEMM（Ampere Mixed GEMM）的 CUTLASS 2.x 实现讲解》视频讲解。这里参考视频并更详细记录了每一页Slides的要点，通过这个视频初步宏观了解了CUTLAS。我将其作为CUDA-MODE的CUTLASS课程的前置学习内容。

![](https://files.mdnice.com/user/59/0f5fa6fa-d376-4e53-b314-581a99e43758.png)

![](https://files.mdnice.com/user/59/c1ad8dcf-f981-4f4c-97af-69e9dd098b08.png)

这张Slides展示了CUTLASS会话的整体结构,主要包含三个部分:

Part I: CUTLASS介绍
    - 主讲人: Petrick Liu
    - 主题包括:
        - CUTLASS 2x和基本GEMM概念
        - 使用CUTLASS 2x创建SOL GEMM的指南
        - CUTLASS 3x的重要GEMM概念
        - 使用CUTLASS 3x创建SOL GEMM的指南
Part II: CUTLASS 2x中的MixedGEMM
    - 主讲人: Yilin Zhang
    - 主题包括:
        - TRT-LLM中的量化
        - 使用CUTLASS 2.x的MixedGEMM
        - 权重布局细节
Part III: CUTLASS 3x中的MixedGEMM
    - 主讲人: Qi Zhang & Petrick Liu
    - 主题包括:
        - CuTe介绍
        - 使用CuTe的GEMM数据流
        - MixedGEMM代码演练

Slides中还显示了各部分之间的联系:
    - 如何根据需求修改CUTLASS 2x和3x GEMM
    - 3x和2x之间的差异

这里还提到CUTLASS主要是为了做自定义算子的，但它的学习曲线非常陡峭，从了解和学习CUTLASS到使用CUTLASS做出work的东西之间有很大的gap。第二部分会介绍对于Hopper架构之前的GPU，CUTLASS是怎么做MixedGEMM的。第三部分介绍Hopper以及Hopper之后的GPU，CUTLASS 3x的MixedGEMM。

![](https://files.mdnice.com/user/59/98e04e45-ebdc-4522-bafb-ff159ca506fb.png)

![](https://files.mdnice.com/user/59/f06cd6f3-f26f-4808-88f4-a2e46d9089f6.png)

这张Slides概述了CUTLASS（CUDA Templates for Linear Algebra Subroutines）从2.x版本到3.x版本的主要特性和进展：

CUTLASS 2.x 特性:

- 支持多种Pre-Hopper架构的GPU，包括Ada(sm_89)、Ampere(sm_8x)、Turing(sm_75)和Volta(sm_70)。
- 提供各种核心功能：GEMM、卷积、稀疏GEMM。
- 支持Group GEMM、B2B GEMMs、FMHA。
- GEMM layernorm融合、GEMM softmax融合。
- 支持Syrk、Trmm、Complex、Planner Complex等操作。
- 在FP32精度上支持3xTF32仿真。

CUTLASS 3.x 特性:

- 保留了CUTLASS 2.x的所有特性。增加了对Hopper架构(sm_90a)的支持。
- 采用了CuTe抽象 (CUDA Templates)。
- 支持sm_90/sm_90a的新特性，包括TMA 、wgmma、集群配置等。
- 引入了持久化风格(Persistent style)和生产者-消费者模型。

> CUTLASS 3.x 的代码风格有很大变化。

![](https://files.mdnice.com/user/59/7e5b5500-207d-4995-88b2-2e30c7789c85.png)

这张Slides讲解了关于CUTLASS库版本选择的指导原则。

- 常见问题：应该使用CUTLASS 2.x还是CUTLASS 3.x？
- 如果你**在Hopper架构的GPU上**工作，并且想要**充分利用**芯片的性能，选择CUTLASS 3.x
- 否则选择CUTLASS 2.x：
    - 大多数Pre-Hopper（即Hopper之前的架构）的特性在Hopper芯片上仍然受支持，这意味着CUTLASS 2.x可以在Hopper芯片上运行。
    - 如果你想使用CUTLASS 2.x的所有扩展和kernel变体。

![](https://files.mdnice.com/user/59/9fe8de40-e579-42a2-84cb-441fab14a66a.png)

这张是CUTLASS GEMM的核心概念图。我们以C矩阵为视角，我们把矩阵C切成小块让每个BLOCK去认领一块做计算。接着要指定WARP去做计算，WARP会认领这个小块中的某一块，比如图中Thread Block Tile的绿色块。每个WARP有32个线程，每个线程又应该做哪一部分计算呢？Warp Tile这个图进一步放大细节，其中4个绿色块代表其中一个线程需要负责计算矩阵C的的部分。最后到线程级别，每个线程都有自己的寄存器去负责做自己的工作。再往右就是Epilogue，这个是很多人使用CUTLASS的第一步比如在GEMM后面做一个Activation后处理。最后把数据写回Global Memory完成整个运算过程。分块的关键参数以及Epilogue的操作类型由图上的using语句所指定。

这个图是CUTLASS的概念，但这里还画出了数据流动，数据需要从Global Memory逐级传递的。除了Tiling之外另外一个重要的概念是我们需要把数据尽可能的复用在高级缓存上，享受到更高的带宽，避免频繁的读取global memory的数据。因此，我们要把数据放在Shared Memory, 寄存器上，然后Tensor Core在寄存器上算完后写Shared Memory，最后从Shared Memory写回Global Memory。

接着，上图的左下角和右下角分别表示读写Global Memory的粒度，以FP16为例设置为8（128 bits / size_bits_of(datatp)）。除了Tiling之外还要考虑Overlap，现在我们有Tling来决定线程块/线程需要做哪些事情，有内存Streaming的过程让数据尽可能的复用在各级存储上，NumStage这个模板参数用来决定开多少个额外的Buffer来做计算和传输的Overlap（参考Double Buffering）,如最下方的中间的图所示。

通过上面的参数，我们就可以完整的配置出来一个CUTLASS的Kernel了。

![](https://files.mdnice.com/user/59/b4ac1d5e-a532-4ad4-b39b-3ac4077d9de3.png)


这张Slides展示了如何在CUTLASS 2.x中构建一个GEMM（通用矩阵乘法）操作，标题强调了CUTLASS 2.x的编码风格："All about is Template Configuration"（一切都是关于模板配置）。

代码示例展示了CUTLASS 2.x中GEMM操作的各种配置项：
- 数据类型定义：
    - 输入、输出矩阵的元素类型
    - 累加器类型
    - 后处理操作类型
- 矩阵布局：
    - 定义输入和输出矩阵的行主序或列主序
- 硬件相关配置：
    - 选择使用张量核心还是常规SIMT核心
    - 指定CUDA SM架构版本（如SM80）
- 计算相关配置：
    - 线程块的tile大小
    - warp的tile大小
    - MMA（矩阵乘累加）操作的大小
- GPU上的线程块调度方式
- CD(输出张量?)内存访问的对齐要求
- 流水线阶段数(上面提到的计算和内存访问流水)
- 张量A和B的对齐方式

最后通过一个Gemm类型的定义，将所有这些配置项组合在一起成为一个可以被实例化的Type。

![](https://files.mdnice.com/user/59/8a68759b-a6b7-4b96-90a5-955f8e72514a.png)

这张Slides展示了如何通过选择合适的GEMM变体和配置参数来满足不同的需求。同时，它也指出了CUTLASS提供了丰富的GEMM实现，可以处理各种特殊情况和优化场景。通过这种方式，开发者可以根据具体需求选择最合适的GEMM实现，并进行细粒度的性能优化。

![](https://files.mdnice.com/user/59/e184b42f-ba45-4746-b8c0-53c38fb93fae.png)

这张Slides讲解了在CUTLASS 2.x中构建GEMM（通用矩阵乘法）时的关键优化选项。主要内容包括：
- 核心思想：调优（Tunning）是关键。
- Option 1: ThreadBlockShape（线程块形状）
    - 需要Tunning。
    - 对于大型GEMM，128x256或256x128通常是最佳的tile大小。
    - 对于小型或中型GEMM，需要仔细调整这个形状。
    - K维度的tile大小选项：32B, 64B, 128B。
- Option 2: WarpShape（warp形状）
    - 也需要Tunning。
    - 对于计算密集型kernel，4-warp配置通常是最常用/首选的。
    - 8-warp配置可能会在prologue或epilogue阶段引入一些延迟。
    - 对于非常小的GEMM问题，可以尝试2-warp或1-warp配置。
    - 2x2 warp配置比4x1或1x4配置有更低的共享内存读取压力。
    - 过大的WarpShape会导致寄存器压力过大，可以通过-Xptxas -v命令查看。
- Option 3: Instruction Shape（指令形状）
    - 总是选择可能的最大形状。
    - 例如：在Ampere架构上有两种FP16 MMA指令变体：`mma.fp16.16.8.8`和`mma.fp16.16.16.16`。
    - 选择最大的一个，这样可以让编译器更容易重叠MMA指令和非MMA指令。

![](https://files.mdnice.com/user/59/f2abfd5f-a62d-40e0-ad7e-6b4afa97f201.png)

这张Slides继续讲解了在CUTLASS 2.x中构建GEMM时的更多优化选项。主要内容包括：
- Option 4: Stage number（阶段数）
    -  需要Tunning，但有一些原则。
    - 原则：最大化SM上高速资源的利用率。高速资源包括寄存器文件（RFs）和共享内存（Smem）。
    - 共享内存使用量计算公式：(Mtile x Ktile + Ntile x Ktile) x sizeof(datatp) x Stage
    - 寄存器使用量可以通过ncu报告或-XptxAs -v命令查看。
    - 一个SM上可以同时运行的线程块数量 = 65536(总RF数) / 每个线程块使用的RF数量
    - 一个SM上可以同时运行的线程块数量 = 163KB(A100 GPU的SMEM容量) / 每个线程块使用的SMEM量
    - 这两个计算得出的数值应该相同。如果不相同，意味着其中一种资源（RF或SMEM）没有被充分利用，造成了浪费。这个原则的核心思想是要同时最大化RF和SMEM的利用率。如果从RF角度计算出的并发块数与从SMEM角度计算出的不同，那么就意味着其中一种资源成为了瓶颈，而另一种资源未被充分利用。
- Option 5: BlockSwizzling（块交织）
    - 当问题规模很大时需要Tunning。
    - 可以提高L2缓存命中率，减少DRAM事务，特别是在M维度和N维度很大时。
    - 这是在大型GEMM形状下实现SOL（Speed Of Light）的重要方法。
- Option 6: Alignment（对齐）
    - 取决于每个张量的ldm（leading dimension）。
    - 总是尝试使用最大的对齐粒度进行访问。硬件支持的最大粒度是16B/线程。
    - 例如：对于FP16，最大对齐是8个元素；对于INT8，最大对齐是16个元素。



![](https://files.mdnice.com/user/59/bbc79863-54ec-46a1-b890-2e1b0bcf3698.png)

这张Slides展示了CUTLASS2.x到CUTLASS3.x的架构变化。在CUTLASS 2.0（类Ampere架构风格）中提到，除了Tiling和复用内存层级，我们还希望使用软流水的方法把global memory读写latency和计算隐藏起来。也就是做当前计算的时候去读后续计算轮次需要的数据，这个过程可以流水起来。在进入稳定流水之前，也就是图中的Main loop body会有一个建立流水的过程。做了计算预取，当前计算的数据肯定在上几轮计算就取出来了，数据预取叫Prologue，GEMM的整个计算过程分为Prologue, Main Loop和Epilogue部分，Tensor Core的计算发生在Main Loop部分。


![](https://files.mdnice.com/user/59/8e61fdfe-64ae-409d-a106-0a6977d0523c.png)


这张Slides对比了CUTLASS 2.x和CUTLASS 3.x在配置GEMM操作时的主要区别。

- CUTLASS 2.x:
    - 需要明确指定更多的配置参数，包括：
        - 输入、输出和计算的数据类型
        - 矩阵布局
        - 使用张量核心还是SIMT核心
        - CUDA SM架构版本
        - 线程块的tile大小
        - Warp的tile大小
        - MMA（矩阵乘累加）指令的大小
        - 线程块调度方式
    - 强调了需要指定WarpShape和InstShape，因为这与MMA指令相关。

- CUTLASS 3.x:
    - 配置更加简化，主要需要指定：
        - 矩阵A、B和C/D的元素类型、布局和对齐方式
        - 核心kernel配置，包括累加器类型、架构标签、操作类等
        - TileShape（BlockShape）和ClusterShape
    - 不需要明确指定WarpShape和InstShape，因为WGMMA（Warp级分组矩阵乘累加）指令是由warp group构成的。

![](https://files.mdnice.com/user/59/18b31bcb-194a-4858-bf4a-ab7b062d3a7d.png)

在CUTLASS 3.x中，重点关注配置主循环（Mainloop）和后处理（Epilogue）部分。具体来说，使用CollectiveEpilogue类型定义后处理操作。包含架构、tile形状、累加器类型等参数。使用CollectiveMainloop类型定义主循环。包含架构标签、操作类、矩阵配置、tile形状等参数。定义GemmKernel，组合Mainloop和Epilogue。最后定义Gemm，使用GemmUniversalAdapter。

CUTLASS 3.x相比2.x版本提供了更高层次的抽象，简化了用户需要手动指定的参数。


![](https://files.mdnice.com/user/59/125d4077-348c-4cb6-8982-41ddafa9bdeb.png)

这张Slides展示了从CUTLASS 2.x到CUTLASS 3.x的演进原因，主要通过比较不同GPU架构（Ampere和Hopper）上GEMM操作的执行效率来说明。

- Ampere架构（1 CTA/SM，6 CTAs）：
    - 显示了prolog（前导）、mainloop（主循环）和epilog（后续处理）的执行时间线。
    - 暴露了prolog/epilog的开销，影响整体效率。
- Ampere架构（2 CTA/SM，6 CTAs）：
    - 仍然存在exposed prolog/epilog开销。但相比于上面的情况已经加速了。
    - 当只有1 CTA/SM运行时，效率降低，只利用了1/2的SMEM（共享内存），减少了延迟隐藏能力。
- Hopper架构（1 CTA/SM，持久化GEMM与warp专业化）：
    - 引入了新的执行模型。
    - 使用持久化GEMM技术，允许一个CTA持续占用SM。
    - 实现了warp专业化，不同的warp组执行不同的任务。
    - 可以在第一个tile还在loop/epilog阶段时，就开始为第二个tile获取数据（prolog）
    - 充分利用SMEM进行延迟隐藏。
    - 在一个warp组中的epilog可以与其他warp组的数学计算重叠。

![](https://files.mdnice.com/user/59/7661e302-22dc-44d7-bc66-7aa1e4226913.png)

这张Slides提供了CUTLASS库中使用的一些关键术语和缩写的解释：

- WGMMA: Hopper Warp Group MMA (矩阵乘累加操作)
解释：这是Hopper架构上的Warp组矩阵乘累加操作
- WS: Warp Specialized
解释：表示warp专业化，即不同的warp执行不同的专门任务
- SS: Src operator of GMMA are both from SMEM
解释：GMMA操作的两个源操作数都来自共享内存(SMEM)
- RS: Src operator A of GMMA is from RF, Src operator B is from SMEM
解释：GMMA操作的源操作数A来自寄存器文件(RF)，源操作数B来自共享内存(SMEM)
- FAST_ACCUM: No additional operation to promote the accum precision
解释：不进行额外的操作来提高累加器的精度

> 下面是Construct CUTLASS 3.x GEMM & Guidelines

![](https://files.mdnice.com/user/59/7f951a64-0f17-44f8-9b2b-3b60eebba81f.png)

这张Slides主要讲解了在CUTLASS 3.x中构建Hopper架构的GEMM操作时，关于CollectiveMainloop中的Stage配置选项。CUTLASS 3.x中Stage可以是固定的常数，如_2, _3, _4等。也可以通过Epilogue Smem使用情况自动计算。警告：小的Stage数量可能会导致全局内存（gmem）延迟暴露。

Slides下方的代码片展示了compute_stage_count_or_override函数的实现，该函数用于计算最大可用的Stage数。
- 考虑了barrier所需的额外字节（32字节）。
- 计算每个stage所需的字节数，包括A和B矩阵的数据。
- 最后根据总容量和每个stage的大小计算可用的stage数量。

![](https://files.mdnice.com/user/59/d03ccb41-ff1a-4ad5-872f-495699116532.png)

这张Slides展示了CollectiveMainloop的配置代码，特别强调了KernelScheduleAuto参数。指出Mainloop kernel Scheduler选项定义在cutlass/gemm/dispatch_policy.hpp文件中。Kernel Scheduler类型包含LDGSTS和UTMALDG两种类型的异步指令，分别针对的是Ampere和Hopper架构。在UTMALDG类型指令可用的情况下，选用它会更快。

![](https://files.mdnice.com/user/59/f33c9705-9c0f-41d8-89be-0944cd50c942.png)

这张Slides介绍了在CUTLASS 3.x中构建Hopper架构GEMM操作时，CollectiveMainloop中Kernel Scheduler的配置选项和自动选择机制。要点如下：
- KernelSchedulerAuto:
    - 这是一个基于配置的自动选择器。
    - 推荐作为首次尝试的好选择。
- TMA (Tensor Memory Accelerator) 限制:
    - TMA 只支持 16 字节对齐的buffer。
    - 对于 8 字节或 4 字节对齐的情况，需要使用 LDGSTS (CpAsync)。
- 具体实现细节:
    - 使用 constexpr bool 进行编译时条件检查。
    - 根据 CUDA 工具包版本选择不同的调度策略。
    - 对于 CUDA Toolkit 版本 >= 12.1，持久化调度表现最佳。
    - KernelTmaWarpSpecializedCooperative 要求 TileShape_M 至少为 128。
    ...

![](https://files.mdnice.com/user/59/d1b2889d-b450-41ea-9072-8f9a33466a91.png)

这张Slides主要讲解了在CUTLASS 3.x中构建Hopper架构GEMM操作时，CollectiveEpilogue的配置选项。强调了EpilogueScheduleAuto参数，这是Epilogue的Kernel调度器。指出Epilogue kernel Scheduler选项定义在cutlass/epilogue/dispatch_policy.hpp文件中。另外，还强调Epilogue Scheduler需要与Mainloop Scheduler配对使用。在CUTLASS 3.x中，使用EpilogueScheduleAuto一定能得到一个合法的Kernel。Epilogue Scheduler选项列出了几种可用的Epilogue Scheduler类型：

- NoSmemWarpSpecialized
- PtrArrayNoSmemWarpSpecialized
- TmaWarpSpecialized
- TmaWarpSpecializedCooperative

![](https://files.mdnice.com/user/59/91c1117a-0d23-46f7-97bd-8cd60d9966b3.png)

这张Slidesz指出，EpilogueScheduleAuto的选择会推断出NoSmemWarpSpecialized的Epilogue Scheduler类型，效率会偏低。

![](https://files.mdnice.com/user/59/376b777d-e1c6-4c24-8f87-091cfeb29f8b.png)

这张Slides讲解了CUTLASS 3.x中Hopper架构GEMM操作的CollectiveEpilogue配置，特别是关于EpilogueTile的设置。特别指出我们可以始终使用EpilogueTileAuto，这代表整个CTile。它会根据Mainloop Scheduler类型计算合理的epilogue tile。

右侧展示了sm90_compute_tile_shape_or_override这个函数的实现，它根据不同条件自动计算epilogue tile的大小：
- a. 对于cooperative调度：
    - 如果TileShape_M >= 128，返回Shape<128, N_tile>
    - 否则返回Shape<64, N_tile>
- b. 对于warp specialized调度：
    - 如果ElementD是8字节，返回Shape<64, N_tile>
    - 否则返回Shape<64, N_tile>（但N_tile的计算方式不同）

![](https://files.mdnice.com/user/59/05610102-b09c-4b77-856c-83e1423e68ce.png)

这张Slides介绍了CUTLASS库中Hopper架构GEMM操作的Kernel Scheduler，特别是KernelMultistage调度器。

- Kernel Scheduler类型列表：分为两类：LDGSTS (Load Global Store Shared) 和 UTMALDG (Unified Tensor Memory Accelerator Load Global)。
- 相关代码文件：cutlass/gemm/kernel/sm70_gemm.hpp & cutlass/gemm/collective/sm80_mma_multistage.hpp
- KernelMultistage执行模型图示：
    - 展示了4个warp (Warp0到Warp3) 的执行模式。
    - 每个warp的执行过程分为三个阶段：LDGSTS（绿色）、TC（蓝色，可能代表Tensor Core计算）和Epilogue（灰色）。
    - prologue和epilogue的延迟是暴露的（exposed）。
- 执行特征：
    - 标注为"Ampere Style But Hopper GMMA"，表明这种执行方式类似于Ampere架构，但使用了Hopper的GMMA（General Matrix Multiply-Accumulate）指令。
    - 非持久化（Non persistent）执行模式。
- 性能特点：
    - prologue和epilogue的延迟暴露，这可能会影响整体性能。
    - 每个warp独立执行完整的GEMM操作流程，包括数据加载、计算和结果存储。

> 每个WARP做一样的事情，并且图中蓝色块之前的绿色小块表示的数据预取的num_stages数。

![](https://files.mdnice.com/user/59/6648ccb9-a251-4ab5-b5aa-7ab5d6724c57.png)

这张Slides介绍了CUTLASS库中Hopper架构GEMM操作的Kernel TMA (Tensor Memory Accelerator) 调度器。
- 相关代码文件：cutlass/gemm/kernel/sm90_gemm_tma.hpp & cutlass/gemm/collective/sm90_mma_tma_gmma_ss.hpp
- KernelTMA执行模型图示：
    - 展示了4个warp (Warp0到Warp3) 的执行模式。
    - 每个warp的执行过程分为三个阶段：TMA（绿色，仅Warp0执行）、TC（蓝色，代表Tensor Core计算）和Epilogue（灰色）。
    - Warp1到Warp3的TMA部分是灰色的，表示它们不执行TMA操作。
- 执行特征：
    - 标注为"Ampere Style But Hopper GMMA + TMA"，表明这种执行方式类似于Ampere架构，但使用了Hopper的GMMA指令和TMA技术。
    - 非持久化（Non persistent）执行模式。
- 性能特点：
    - prologue和epilogue的延迟仍然是暴露的。
    - 只有Warp0执行TMA操作，这可能意味着更高效的内存访问。
- 与KernelMultistage的区别：
    - 主要区别在于使用TMA替代了LDGSTS操作。
    - TMA操作集中在Warp0中执行，而不是每个warp都执行。

![](https://files.mdnice.com/user/59/d43824eb-192d-47c6-84a4-6bb2551a48a4.png)


这张Slides介绍了CUTLASS库中Hopper架构GEMM操作的Warp Specialized调度器。
- 相关代码文件：cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized.hpp & cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp
- Warp Specialized执行模型图示：
    - 展示了8个warp (Warp0到Warp7) 的执行模式。
    - Warp0和Warp1专门执行TMA（绿色）操作。
    - Warp2和Warp3不执行任何操作（灰色）。
    - Warp4到Warp7专门执行TC（蓝色，Tensor Core计算）和Epilogue（灰色）操作。
- 执行特征：
    - 标注为"Hopper Warp Specialized Style"，表明这是Hopper架构特有的warp专业化执行方式。
    - 非持久化（Non persistent）执行模式。
- 性能特点：
    - Prologue和Epilogue的延迟仍然是暴露的。
    - 寄存器文件（RF）利用率较低。
- 与之前调度器的区别：
    - 明显的warp分工：部分warp专门负责内存操作，部分负责计算。
    - 更高效地利用了TMA技术。

> Warp Specialized调度器的Tensor Core计算和Epilogue还是没有Overlap在一起，无法充分发挥Tensor Core的能力。

![](https://files.mdnice.com/user/59/002eda2a-46da-4264-bb9e-0bb9f223f9f2.png)

这张Slides补充了寄存器分析，TMA warps (Warp0和Warp1)：每个线程使用32个寄存器，总共128 x 32 = 4K个寄存器。TC warps (Warp4到Warp7)：每个线程最多可以使用255个寄存器，总共128 x 255 = 32K个寄存器。

Slides还指出"This is not optimal for the SOL impl."（这对于SOL实现来说不是最优的）。Epilogue延迟仍然暴露，即使使用持久化编程且寄存器文件（RF）利用率较低。


![](https://files.mdnice.com/user/59/7306b2ee-7d8a-4381-af29-f71e9c90fe00.png)

这张Slides介绍了CUTLASS库中Hopper架构GEMM操作的Warp Specialized + Cooperative (Persistent) 调度器。
- 相关代码文件：cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp & cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp
- 核心优化策略为"Use CTA reconfiguration to dealloc and alloc RF to fully utilize the RFs"
使用CTA（Cooperative Thread Array）重配置来释放和分配寄存器文件（RF），以充分利用寄存器资源。
- 与普通WarpSpecialized实现的区别：
    - 更多的Warps，更好的TC利用率。
    - TMA warps释放RF，数学计算Warps分配更多RF。
    - 持久化风格（Persistent style）。
- 性能上Epilogue延迟仍然暴露，但有更好的RF利用率。

![](https://files.mdnice.com/user/59/f0a9abff-4ad1-45f0-91ef-8686fb2474bf.png)

这张Slides介绍了CUTLASS库中Hopper架构GEMM操作的Warp Specialized + Pingpong (Persistent) 调度器。
- 相关代码文件：cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp & cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp
- 与Warp Specialized Cooperative实现的主要区别：
    - Mainloop和epilogue过程重叠，以实现最佳的TC（Tensor Core）利用率。
    - TC Warp组之间的同步。
- 执行模型图示:
    - 展示了12个warp（Warp0到Warp11）的执行模式。
    - Warp0和Warp1执行TMA操作（绿色），但以pingpong方式交替进行。
    - Warp4到Warp11执行TC（浅蓝色和深蓝色）和Epilogue（灰色）操作，同样以交替方式进行。

![](https://files.mdnice.com/user/59/ee00936a-5a53-43ad-aa69-f10a39863f03.png)

这张Slides介绍了一下Hopper架构上的Warp Specialized GEMM实现，采用了生产者-消费者模型。内容如下：
- 源代码位置：cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized_mixed_input.hpp
- 总体架构分为Producer Warps (TMA Warps) 和 Consumer Warps (TC Warps)，通过共享内存进行数据交换。
- Producer Warps (TMA Warps):
    - 使用CollectiveMma::load(...) & Persistent方法
    - 等待smem_empty barrier
    - 发出TMA指令加载A和B矩阵，更新smem_full barrier
    - 更新传输字节数并到达smem_full barrier
    - 循环K次迭代
- Consumer Warps (TC Warps):
    - 使用CollectiveMma::mma(...) & Persistent方法
    - 等待smem_full barrier
    - 发出WGMMA_SS指令并等待前一个TC工作完成
    - 到达smem_empty barrier
    - 循环K次迭代
    - 使用SWIZZLE将寄存器文件(RF)写入共享内存(SMEM)
    - 发出TMA指令将结果写回全局内存
- 共享内存结构：
    - 包含Mbarrier和Data Buffer两部分
    - 每个stage有两个buffer：Mat A MtilexKtile 和 Mat B NtilexKtile
    - 使用smem_empty和smem_full标志来同步Producer和Consumer
- 执行流程：
    - Producer和Consumer交替工作，通过共享内存和 barrier机制同步
    - 多个stage (0 到 N-1) 用于流水线操作
    - 循环执行直到完成所有tile的计算

![](https://files.mdnice.com/user/59/0d8b9a3f-84a0-4fdc-8d1d-041f1e58a923.png)

这张Slides展示了Hopper架构上不同GEMM kernel调度器的性能基准测试结果。

- 测试矩阵大小和kernel类型：
    - 测试了6种不同大小的矩阵乘法运算
    - 比较了5种不同的kernel调度器：KernelTMA, WS_TMA (无/有共享内存), Pingpong_TMA, Coop_TMA
- 性能数据和分析：
    - 表格中显示了各种组合的执行时间（单位：微秒）
    - 黄色高亮标记了每行中性能最佳的结果
    - Warp Specialized kernels 通常表现更好
    - 使用共享内存(SMEM)的Epilogue性能更佳
    - 在大多数情况下，Pingpong策略是首选
    - 对于较大的矩阵（如8192x8192x8192），Pingpong_TMA策略显著优于其他方法
    - 对于某些特定大小（如1024x1024x1024），WS_TMA with smem表现最佳
    - KernelTMA + No Smem在所有情况下性能都较差
- 配置信息：
    - FP16输入，FP32累加，D = alpha x A x B 运算，CTA tile大小 = 128x128x64，Cluster shape = 2x1x1，使用H800 NVL*硬件，预热10次，迭代20次，NVCC版本12.3。

![](https://files.mdnice.com/user/59/4edbf258-2598-4088-837c-f15385f01845.png)

这张Slides比较了Hopper架构上CPAsync和TMA两种不同内存访问方式在GEMM操作中的性能表现。主要结论为：TMA方法在几乎所有情况下都优于CPAsync方法，在TMA方法中，Pingpong_TMA通常表现最好，尤其是对于大型矩阵。"CPAsync is the reluctant choice. Always use TMA if the alignment requirement is satisfied."（CPAsync是不得已的选择。如果满足对齐要求，总是使用TMA。）

![](https://files.mdnice.com/user/59/8d280271-4589-4a31-964f-07ff06489d1c.png)

这张Slides总结了在Hopper架构上构建GEMM（通用矩阵乘法）时的几个关键决策点和建议。
- Option 1: CpAsync vs TMA
    - 选择取决于内存对齐情况，TMA只能处理16字节对齐的情况。如果对齐较差，只能使用CpAsync。如果满足16字节对齐要求，应使用TMA以获得更好的性能。
- Option 2: Non-Warp-Specialized vs Warp-Specialized
    - 建议总是使用Warp-Specialized，Hopper硬件提供快速同步机制，同步开销不大。使用Non-Warp-Specialized时需要调整stage以获得更好的性能，对于小型GEMM问题，可以考虑使用Ampere风格的kernel。
- Option 3: Warp Specialized vs Pingpong vs Cooperative
    - 选择取决于问题的形状，如果C Tile数量小于SM数量（1个wave），epilogue延迟暴露不可避免，三种方法都可以。如果C Tile数量超过1个wave，推荐使用Pingpong方法。

![](https://files.mdnice.com/user/59/e9e7ea0e-efc5-4fd6-9699-c97bb08bf167.png)

![](https://files.mdnice.com/user/59/7208b3ee-ebc0-4142-983a-b53310abd9b7.png)

最后这张Slides继续讨论了构建Hopper GEMM (通用矩阵乘法) 的2个关键点:

- 选项3的更新：比较了Warp Specialized、Pingpong和Cooperative三种方法。选择取决于问题形状和tile大小：
    - 对于128x256的tile大小：
        - 使用FP32累加时，Cooperative是唯一选择，因为Pingpong会遇到寄存器溢出问题。
        - 使用FP16累加时，Pingpong始终是最佳选择。
- 选项4：
    - 为了获得更好的性能，需要调整多个参数，例如：
        - Tile大小
        - CGA (Cooperative Grid Array) 大小
        - CTA swizzle

代码示例展示了CUTLASS 3.x和2.x版本在处理swizzle size参数上的区别：
- CUTLASS 3.x：swizzle size可以是运行时参数
- CUTLASS 2.x：swizzle size是模板参数

总结一下：

CUTLASS库在2.x到3.x版本的迭代中有了显著的变化,这主要是为了适应NVIDIA GPU架构从Ampere到Hopper的演进。3.x版本着重强调了持久化编程风格和warp专业化的特性,旨在充分利用计算资源并优化性能。

在配置GEMM运算时,2.x版本需要手动指定大量的low level参数,如数据类型、tile大小等。而3.x版本则提供了更高层次的抽象,简化了配置过程。

如果在Hopper架构上使用CUTLASS,建议采用3.x版本,并参考以下最佳实践:

- 在访问内存时,优先使用TMA(Tensor Memory Accelerator)而非CpAsync。
- 优先选择Warp Specialized类型的kernel。
- 根据问题的规模,在Warp Specialized、Pingpong、Cooperative三种kernel中选择最合适的一种。
- 通过调整BlockShape、ClusterShape、CTA swizzle等参数,进一步优化性能。

End!
