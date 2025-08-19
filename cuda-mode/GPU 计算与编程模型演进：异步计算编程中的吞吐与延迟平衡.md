> Slides来自BiliBili NVIDIA英伟达频道上传的NVIDIA AI技术开放日2025初夏《GPU 计算与编程模型演进：异步计算编程中的吞吐与延迟平衡》视频讲解。这里参考视频并更详细记录了每一页Slides的要点，作为学习使用。

![](https://files.mdnice.com/user/59/92ed783e-08ec-4688-9e5e-af81ee9a9fc4.png)

这里展示了三个重要的技术分享主题：第一个由Petrick Liu和Jiang Shao主讲，聚焦于异步编程中计算吞吐量与延迟的平衡技术；第二个由Allard Hendriksen主讲，介绍最大化内存带宽利用和隐藏延迟的CUDA技术；第三个由Albert Di和Vincent Zhang主讲，探讨如何从CUTLASS C++到CUTLASS Python实现最大开发吞吐量。第二个演讲的笔记上篇文章已经记录了，当前这个课程是第一个演讲的内容。


## GPU 计算与编程模型演进：异步计算编程中的吞吐与延迟平衡

![](https://files.mdnice.com/user/59/96c6604c-4cf1-42ff-97ca-787ffe5d5c37.jpg)

![](https://files.mdnice.com/user/59/1f038c0a-d66c-460c-8616-e3e9abd3c6d5.png)

这张Slides是当前这个Talk的目录，首先回顾Hopper架构之前用于最大化吞吐量的各种技术手段；然后深入介绍Mbarrier作为异步编程的关键使能技术；接着探讨如何通过无处不在的流水线技术来实现吞吐量最大化；随后概述从Hopper到Blackwell架构演进过程中的重要变化；最后通过W4A8 Hopper内核的具体案例研究来展示这些理论技术在实际应用中的实现效果。

![](https://files.mdnice.com/user/59/bdc56ce6-90f2-4549-8679-36d690dedfbe.png)

这张Slides展示了Hopper架构之前GPU异步编程中的传统优化技术和执行模式，注意这张Slides展示的是没有优化的情况。图的上半部分阐释了CutLass GEMM计算的完整内存层次结构，从全局内存经由共享内存、寄存器文件最终到达CUDA/Tensor Core的数据流动路径，并展示了不同粒度的分块策略（Blocked GEMM、Thread Block Tile、Warp Tile、Thread Tile）如何逐层细化数据处理单元。右侧的代码片段展现了传统同步编程模式的典型结构，包括数据Tile加载（load_A_tile、load_B_tile）、线程同步（syncthreads()）、片段加载（load_A_frag、load_B_frag）和矩阵乘累加运算（mma）的顺序执行流程。图的下半部分通过时间线图形象地描绘了MMA（矩阵乘累加）操作和LDGSTS（全局内存到共享内存的数据传输）操作在多个Ktile周期中的调度安排，以及Tensor Core的激活状态。可以看到由于加载数据的延迟，Tensor Core出现了周期性的工作状态，这个时候无法达到最大化吞吐。

![](https://files.mdnice.com/user/59/e24058f9-e0d7-4828-a5dc-9710a39fd0e8.png)

这张Slides展示了Ampere架构下GPU异步编程中的流水线优化策略，重点阐释了如何通过精心设计的Prologue阶段来建立稳定的计算流水线。图中详细描绘了GEMM计算的完整执行流程，包括从Ktile0到Ktile5的连续数据预取过程，通过提前启动多个LDGSTS（全局内存到共享内存的加载）操作来有效隐藏全局内存访问延迟。时间线图清晰地展现了LDG（全局内存加载）、Wait（等待）、LDS（共享内存加载）和MMA（矩阵乘累加）操作之间的精确调度关系，以及Tensor Core的激活周期。这种设计的核心思想是通过软件流水线技术，让数据预取操作与计算操作重叠执行，从而在Tensor Core进行当前Ktile计算的同时，后续Ktile的数据已经在并行加载中，实现了计算与内存访问的Overlap。然而，我们可以看到Shared Memory的Load延迟依然是暴露在整个pipeline中的，TensorCore的执行依然有bubble。

![](https://files.mdnice.com/user/59/8f38271b-8ec7-49db-8ab3-1fe80a5ccbf5.png)

更进一步，我们可以把Load Shared Memory也流水化，从而进一步隐藏延迟。左侧展示了Prologue阶段建立稳定流水线的过程，而右侧重点说明了实现最大吞吐量必须采用高度流水线化设计，但这种设计的代价是增加了prologue延迟。图的下面那部分详细描绘了Main Loop中的稳定流水线状态，其中全局内存加载（Gmem Loading）、共享内存加载（Smem Loading）和Tensor Core计算（TC Computing）三个关键操作实现了完全重叠执行。时间线图精确展现了RF双缓冲机制的工作原理，以及LDS（共享内存加载）和MMA（矩阵乘累加）操作的精细调度安排。

上面的几张图的核心思想就是在每个有Latency的地方想办法去隐藏延迟，以达到最大吞吐。

![](https://files.mdnice.com/user/59/d32b01a4-1a7e-41cf-b80a-715b5998fedf.png)

来到Hopper架构，Tensor Core计算更快，期望所有的延迟Tensor Core都在工作，就需要进一步的在任务级别让一个CTA去做更多的事情，把MainLoop和Prologue/Epilogue阶段重叠起来。

这张Slides其实展示了从CUTLASS 2.x到CUTLASS 3.x的演进过程，主要通过比较不同GPU架构（Ampere和Hopper）上GEMM操作的执行效率来说明。

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


![](https://files.mdnice.com/user/59/6a821870-48d5-4cfc-99af-878825984319.png)

![](https://files.mdnice.com/user/59/e58c9094-b2c4-4c8c-889f-5a6d381d4b5b.png)


这张Slides介绍了一下Hopper架构上的Warp Specialized GEMM实现，采用了生产者-消费者模型。内容如下：
- 总体架构分为`Producer Warps` (TMA Warps) 和 `Consumer Warps` (TC Warps)，通过共享内存进行数据交换。
- `Producer Warps` (TMA Warps):
    - 使用 `CollectiveMma::load(...) & Persistent` 方法
    - 等待`smem_empty barrier`
    - 发出TMA指令加载A和B矩阵，更新`smem_full barrier`
    - 更新传输字节数并到达`smem_full barrier`
    - 循环K次迭代
- `Consumer Warps` (TC Warps):
    - 使用 `CollectiveMma::mma(...) & Persistent` 方法
    - 等待`smem_full barrier`
    - 发出`WGMMA_SS`指令并等待前一个TC工作完成
    - 到达`smem_empty barrier`
    - 循环K次迭代
    - 使用`SWIZZLE`将寄存器文件(RF)写入共享内存(SMEM)
    - 发出TMA指令将结果写回全局内存
- 共享内存结构：
    - 包含`Mbarrier`和`Data Buffer`两部分
    - 每个stage有两个buffer：`Mat A MtilexKtile` 和 `Mat B NtilexKtile`
    - 使用`smem_empty`和`smem_full`标志来同步Producer和Consumer
- 执行流程：
    - `Producer`和`Consumer`交替工作，通过共享内存和 barrier机制同步
    - 多个stage (0 到 N-1) 用于流水线操作
    - 循环执行直到完成所有tile的计算


![](https://files.mdnice.com/user/59/fd0bdcae-1d93-4704-9ab1-cc7053cabb0b.png)

这张Slides介绍Hopper架构中Mbarrier和TMA编程模式的核心机制和工作原理。图中展示了从两个不同角度观察这一编程模式的视图：左侧是TMA Warp的视角，右侧是TC WarpGroups的视角。中心部分重点描绘了Mbarrier（内存屏障）的数据结构，包含五个关键字段：Phase（阶段，初始值为0）、Expected Arrival Count（期望到达计数，初始值为1）、Actual Arrival Count（实际到达计数，初始值为0）、Expected Transfer Bytes（期望传输字节数，初始值为0）和Actual Transfer Bytes（实际传输字节数，初始值为0）。上方的代码片段展示了Mbarrier的初始化过程，通过`init_mbarrier(&bar, 1)`函数设置屏障参数，并通过`mbarrier_fence()`建立内存屏障。

![](https://files.mdnice.com/user/59/4975dc5f-7b47-4d1b-9913-81a338cc8926.png)

这张Slides继续展示了Hopper架构中Mbarrier和TMA编程模式的具体实现机制和执行流程。图中从两个关键视角详细阐释了异步数据传输的协作模式：左侧展现了TMA Warp的操作逻辑，通过`tma_thread`条件判断来执行`issue_TMA_bulk_load`异步批量加载操作（传输16KB数据），随后调用`mbarrier_arrve_expect`函数通知屏障预期的数据传输量；右侧展示了TC WarpGroups的等待逻辑，通过`while`循环和`try_wait`函数持续检查屏障状态，当`phase`未翻转时会在此处阻塞等待。中心的Mbarrier数据结构清晰展现了同步机制的核心状态：Phase保持为0，Expected和Actual到达计数均为1，Expected传输字节数设置为16KB而Actual传输字节数仍为0，表明数据传输正在进行中。图下方的注释进一步说明了各字段的含义：TMA事务字节数会更新到该屏障，期望传输字节数等于所有执行线程的字节数总和，实际到达计数等于已执行线程的总数。这种设计实现了TMA硬件单元与Tensor Core计算单元之间的精确异步协调，确保数据在完全传输就绪后才进行计算，是Hopper架构异步编程的核心创新。

![](https://files.mdnice.com/user/59/45a1421d-c9c8-40b5-b92d-698cae2acc41.png)

![](https://files.mdnice.com/user/59/1fb52679-9225-41bc-9302-3f9f1144933f.png)

这两张Slides展示了TMA Warp的数据传输还在继续，分别传输到了1KB和4KB，这个时候MBarrier的状态没有发生改变。

![](https://files.mdnice.com/user/59/b891d721-c404-4278-903d-952fed3cb790.png)

与前一张图相比，这里显示了重要的状态变化：TMA Warp已经成功执行了16KB的批量数据传输，Mbarrier中的Actual到达计数已从0更新为1，Actual传输字节数也从0更新为16KB，表明数据传输操作已经完成。然而，关键的Phase字段仍然保持为0未发生翻转，这意味着虽然mbarrier已经记录了完整的16KB事务，但同步条件尚未完全满足。因此，右侧TC WarpGroups中的线程仍然在while循环中持续等待，try_wait函数继续返回阻塞状态，因为phase还没有翻转到期望的状态。


![](https://files.mdnice.com/user/59/8d420e33-dcb6-448e-bb73-9f78e4eb46c6.png)

这张Slides展示了Mbarrier和TMA编程模式的最终同步完成状态，标志着一个完整同步周期的结束和计算阶段的开始。图中最关键的变化是Mbarrier的Phase字段从0成功翻转到1，这表明所有同步条件都已满足，同步屏障被正式触发。与此同时，其他字段都被重置到初始状态：Actual到达计数从1重置为0，Expected和Actual传输字节数都从16KB重置为0，为下一轮同步周期做好准备。右侧TC WarpGroups的执行流程发生了根本性转变：while循环检测到phase已经翻转后成功退出阻塞状态，注释明确显示"Pass here! Phase has been flipped!"，表明等待的线程现在可以继续执行。接下来开始实际的计算工作，通过WGMMA（WarpGroup Matrix Multiply-Accumulate）操作来消费共享内存中的数据（SmemA和SmemB），并将结果累积到Accums中。

![](https://files.mdnice.com/user/59/c485da12-7267-4936-8df8-d5313affeaa8.png)

这张Slides展示了bar_full（数据满屏障）和bar_empty（数据空屏障），形成了一个精密的同步控制系统。左侧TMA Warp首先需要等待bar_empty屏障（期望到达计数为128，因为这是一个warpgroup级别的消费者），确认共享内存可用后才能执行数据传输操作；右侧TC WarpGroups则先等待bar_full屏障确认数据就绪，然后执行WGMMA计算消费共享内存数据，完成计算后通过WAIT_WGMMAs()确保Tensor Core完全消费完数据，最后调用`mbarrier_arrive(&bar_empty)`释放共享内存资源供下一轮使用。两个Mbarrier的状态显示了这种双向同步机制：Smem_Full屏障的Phase已翻转为1表示数据就绪，而Smem_Empty屏障的Expected和Actual到达计数都为128表示所有相关线程都参与了同步。这种设计实现了共享内存的安全复用，确保在数据完全被消费之前不会被新数据覆盖，体现了Hopper架构异步编程中资源管理和同步控制的高度精细化，为高效的流水线计算提供了硬件级别的可靠保障。

![](https://files.mdnice.com/user/59/d0855ec2-6301-4dc9-914b-84430f68f752.png)

这张Slides展示了Hopper架构中双屏障Mbarrier和TMA编程模式的一个关键转换时刻，即共享内存资源成功释放并准备进入下一轮数据传输的状态。图中最重要的变化是左侧TMA Warp的`empty_phase`重置为0，表明系统正在为新的同步周期做准备，而`bar_empty`屏障已经成功翻转（如底部注释"Phase flip!"所示），这意味着TC WarpGroups已经完成了对共享内存的消费并释放了资源。左侧代码注释明确显示"pass here, will loop back // issue next TMA inst to refill data"，表明TMA线程现在可以跳出等待循环，准备发起下一个TMA指令来重新填充数据。右侧TC WarpGroups的phase也重置为0，准备等待新一轮的数据就绪信号。两个屏障的状态显示了这种周期性转换：Smem_Full屏障保持Phase=1状态等待新数据到达，而Smem_Empty屏障的Phase翻转为1表示共享内存已清空可用。

![](https://files.mdnice.com/user/59/0c1c6a68-91cd-442f-9c9a-218f73b09dee.png)

接着就是到第2次迭代中的Mbarrier状态了，和上一个迭代的区别就是这里等待FLIP就是从1到0了。

![](https://files.mdnice.com/user/59/ca4b6c3c-0548-452a-bf15-a47c671c74b6.png)

这张Slides上面已经展示过了，不过额外加了几条总结。这里展示了一下Hopper架构上的Warp Specialized GEMM实现中的生产者-消费者模型的几个好处。生产者和消费者代码完全解耦、生产者可更新任何空闲stage、完成状态在block级别可见、支持异步状态检查以减少阻塞，体现了Hopper架构通过硬件异步支持和精细化任务分工来最大化GPU计算效率和吞吐量的设计理念。

![](https://files.mdnice.com/user/59/9ea4292f-59c7-4aab-85c9-2893a3300352.jpg)


这张Slides展示了Hopper架构中Warp特化GEMM使用CUTLASS流水线原语实现Ring Buffer的核心机制。左侧展现了CUTLASS中`PipelineState`结构体的完整代码实现，包括索引（index）、阶段（phase）和计数（count）三个关键字段，以及`operator++`和`advance`等操作函数，特别突出了phase翻转逻辑——当迭代次数跨越stage边界时会触发phase翻转（`phase_ ~= 1`），这是实现异步同步的核心机制。右侧的共享内存布局图清晰展示了多stageRing Buffer的组织方式，每个Stage都配备独立的Mbarrier（包括Smem_empty和Smem_full屏障）和对应的数据缓冲区（Mat A MtilexKtile和Mat B NtilexKtile），支持矩阵数据的分块存储。中间的绿色箭头形象地表达了circular pipeline的核心思想，即如何在有限的stage之间循环复用缓冲区资源，通过精确的phase管理和屏障同步，实现数据加载、计算执行和资源释放的高效流水线化，这种设计确保了GPU内存带宽的最大化利用和计算资源的持续活跃状态。

![](https://files.mdnice.com/user/59/2ccce951-cb0e-4e72-8315-9571c30d1b5b.jpg)

这张Slides展示了CUTLASS中TMA-TC生产者-消费者模型的示例代码实现，左侧展现了`PipelineTmaAsync`类的核心定义，包括`FullBarrier`和`EmptyBarrier`的类型别名、`SharedStorage`共享存储结构（包含多stage的屏障数组）、`ThreadCategory`枚举（定义NonParticipant、Producer、Consumer、ProducerConsumer四种线程角色）以及`Params`参数结构体。中间部分详细展示了Producer（TMA Warps）的完整API集合，包括`producer_try_acquire`（尝试获取资源）、`producer_acquire`（正式获取资源）、`producer_commit`（提交数据传输）和`producer_tail`（防止producer block提前退出）等关键函数，每个函数都配合PipelineState进行精确的stage和phase管理。右侧对应展现了Consumer（TC WarpGroups）的API实现，包括`consumer_try_wait`、`consumer_test_wait`、`consumer_wait`和`consumer_release`等函数，实现了对数据就绪状态的检测和资源的释放控制。这种设计通过CUTLASS内置原语完美封装了Hopper硬件的异步特性，为开发者提供了高级抽象接口，使得复杂的TMA-TC协作模式能够以清晰、类型安全的方式实现。

![](https://files.mdnice.com/user/59/f5ef40a9-9117-4a6d-9ff9-267200ebe493.png)




