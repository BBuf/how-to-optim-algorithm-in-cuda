> 我的课程笔记，欢迎关注：https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode 

## 第四课: 计算和内存基础（基于PMPP 书的第4-5章）

![](https://files.mdnice.com/user/59/226f775a-9061-4639-81f9-56ad86a8cb67.png)

第4章：计算架构和调度，如何保持整个GPU繁忙

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

- CUDA内核启动：
    