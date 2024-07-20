> CUTLASS GEMM模板中有大量可以调节和设置的模板参数，这些参数的设置会高度影响Kernel性能。本次分享将为大家介绍从2.x到3.x，CUTLASS kernel实现的变化，这些参数的原理和选择的最佳实践。

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

