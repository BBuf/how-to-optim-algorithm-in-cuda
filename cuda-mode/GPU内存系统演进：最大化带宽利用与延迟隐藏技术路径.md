> Slides来自BiliBili NVIDIA英伟达频道上传的NVIDIA AI技术开放日2025初夏《GPU 内存系统演进：最大化带宽利用与延迟隐藏的技术路径》视频讲解。这里参考视频并更详细记录了每一页Slides的要点，作为学习使用。


**这场演讲深入分析了现代GPU内存系统演进中面临的核心挑战与解决方案。随着GPU架构从V100发展到H100、H200乃至B200，虽然内存带宽大幅提升，但每个SM可用带宽的增长使得传统简单kernel越来越难以充分利用硬件性能。演讲基于Little's Law阐述了"in-fight字节数"的关键作用，并系统介绍了三大技术路径：通过指令级并行（ILP）和数据级并行（DLP）增加in-fight数据，利用异步加载技术从cp.async到TMA实现更高效的内存访问，以及针对小规模问题特有的kernel启动延迟瓶颈，提出了CUDA Graph、程序化依赖启动（PDL）等创新技术组合。这些优化策略不仅解释了为什么新架构需要更复杂的编程技巧，更为开发者在不同规模问题上选择最佳内存优化方案提供了清晰的决策框架和实践指导。**


![](https://files.mdnice.com/user/59/a4210d81-678e-4203-a48c-1e04c565f683.jpg)

这张图片展示了NVIDIA AI技术开放日2025初夏活动的演讲封面，主题为"GPU内存系统演进：最大化带宽利用与延迟隐藏的技术路径"。

![](https://files.mdnice.com/user/59/67ff1ba8-7e20-402b-a7ef-cb9ca5ba2103.png)


这张图片展示了GPU内存系统的技术架构和演进路径，是这个演讲的目录。包含硬件发展趋势，理解内存带宽，最大化内存带宽以及为小尺寸解决隐藏延迟。

![](https://files.mdnice.com/user/59/3ec473db-da40-425b-a088-8f81aa6d8dd3.png)


这张图片展示了GPU硬件发展的趋势分析，包含了不同代GPU架构的内存带宽对比、SMs数量对比以及每个SM的内存带宽对比。

![](https://files.mdnice.com/user/59/13ac86b8-0848-49b9-b308-a972ce05b983.png)

这张图展示了随着每个SM可用带宽持续增大，简单kernel已无法再吃满带宽。绿色柱子显示代际带宽（V100→A100→H100→H200→B200）在上升；橙色曲线“BWUtil”显示利用率在下降——简单kernel在新架构上越来越难达到峰值带宽。


![](https://files.mdnice.com/user/59/a8498248-d77a-48e5-8bbd-245ea39c413b.png)

理解内存带宽，看一下Little's Law。

![](https://files.mdnice.com/user/59/408be90c-2c8a-4f00-b239-86a9a9ca1451.png)

这张Slides讲的是用“小法则”(Little’s Law)解释“in-fight量”(in-fight)决定吞吐。公式为：系统平均in-fight数 L = 到达率 λ × 逗留时间 W；等价地，稳定时吞吐 X = L / W。

示例电梯:
- 每 2 秒来一级台阶 → 峰值到达率 λ_peak = 0.5 人/秒
- 共 20 级台阶 → 逗留时间 W = 20×2 = 40 秒
- 若只有 1 个人在扶梯上：X = 1 / 40 = 0.025 人/秒（远低于峰值）
- 要想打满峰值 0.5 人/秒，需 L = W×λ = 40×0.5 = 20 人同时in-fight（每级台阶都有人）

![](https://files.mdnice.com/user/59/a66f72e0-2603-450b-9533-fd2ab2516e67.png)

bytes-in-flight = bandwidth × mean latency。硬件给定带宽与延迟；软件需把“in-fight字节数”堆到这个水平，才能吃满带宽。想达到≈90% 峰值带宽，H100 每个SM大约需要≈32KB 的 bytes-in-flight；H200/GB200 需要≈64KB/SM。新架构想打满显存带宽，必须在每个SM上制造更多“in-fight数据”。

![](https://files.mdnice.com/user/59/1ad4fc18-ba42-4864-8426-09b6a39a41e2.png)

这张图特别添加了一下H20的情况，可以看到H100约在 32 KB/SM 时接近饱和（≈90%+）。而 H20 需要更多in-fight字节，约 64 KB/SM 才接近饱和。B200 与 H20 类似，约 64 KB/SM。总的来说H20 相比 H100 需要更高的 bytes‑in‑flight 才能打满显存带宽，且与 B200 基本相当。

![](https://files.mdnice.com/user/59/04c168a6-4f35-450a-9cbd-085c5c4d5088.png)

我们是否可以用Little’s Law 判断“简单 kernel 能否吃满显存带宽？”

首先估算一下in-fight字节数/SM。estimated bytes‑in‑flight/SM = (#loads/thread) × (#bytes/load) × (#threads/block) × (#blocks/SM) 。向量加法里：2 × 4 × 256 × 8 = 16 KiB/SM（假设100% Occupancy）。

新架构要接近峰值需≈32–64 KiB/SM；16 KiB 明显不够 → 简单 kernel 难以打满带宽。右图显示真实带宽在上升但是简单kernel的利用率却在下降。

![](https://files.mdnice.com/user/59/22fe1b99-8dbb-4bba-ab85-6619a0fcdcb2.png)

如果我们把每个线程的负载从2改到3，我们可以看到kernel的带宽利用率就增加了，但是这里改变了问题，我们应该如何通用的去解决这个问题呢？

![](https://files.mdnice.com/user/59/4095e363-f578-43dc-ab46-6641a0d0ed23.png)

如何增加in-fight数据和带宽？

![](https://files.mdnice.com/user/59/7e68d828-9b3a-470f-8be6-6ae3201731b3.png)

这里展示了三种把in-fight bytes做大的通用手段，以吃满带宽。

三种工具:
- 单线程更多独立内存操作：每线程处理多元素、循环展开、增加并行加载（提升 ILP）。
- 向量化内存操作：用 float4/uint4 等、合并访问，单次事务搬更多字节并减少指令。
- 异步数据拷贝：用 cp.async/TMA 预取到 shared，双/三缓冲，让加载与计算重叠。

![](https://files.mdnice.com/user/59/2940fc61-74c4-4e6c-ba54-df8b0883df0f.png)

我们的目标是提高 ILP（指令级并行）来把“in-fight字节数”做大，从而更接近显存带宽上限。这个kernel每次迭代每线程发起两次全局加载：load a、load b，然后 mul、store c。估算每线程in-fight字节数：#loads/thread × bytes/load = 2 × 4B = 8B。ILP 低，内存延迟不易被隐藏。

![](https://files.mdnice.com/user/59/a1e8663e-a149-4d50-970b-f27db844174e.png)

通过 `#pragma unroll 2` 或手工展开，让一个线程在使用前先发起两对独立加载：`a[i1], b[i1], a[i2], b[i2]`，再进行两次计算与写回。
估算每线程in-fight字节数翻倍：#loads/thread = 4，4 × 4B = 16B。in-fight请求更多，带宽利用更高。

![](https://files.mdnice.com/user/59/bad03506-80a9-4591-83ea-6d66a386039a.png)

另外一种技巧是用“向量化全局访存”提升 DLP/bytes‑in‑flight，从而更接近峰值带宽。这个大家都比较熟悉，可以显式使用向量类型：float2、float4（对应 64/128‑bit 事务）。或对标量指针做向量指针的强制转换；前提是地址按 8/16B 对齐。右边的例子是：例子: 每线程一次读入 `a[i1],a[i2]` 与 `b[i1],b[i2]`，计算两次，再写回两次 → 一次迭代有 16B数据在in-fight，指令/事务更少。需要确保对齐与边界处理（`N%2`、N%4 的尾部）、可能的寄存器压力；`__restrict__` 有助编译器优化。


![](https://files.mdnice.com/user/59/d1b33d68-b00a-4b96-a494-513e8278899c.png)

这里展示了一下unroll和vectorize benchmark的对比，基准为元素乘（elementwise vector multiply），数据量 4GiB；比较不同手段对带宽的提升幅度（相对标量、无展开的基线）。绿色为循环展开 `unroll2/4`，粉/紫为向量化 `vec2/vec4`（64/128bit 事务）。

代际越新，收益越大：V100 仅≈1–2%，A100≈5–6%，H100≈12–13%，H200≈21–25%，B200≈18–23%。在 H200/B200 上，vec4 ≈ 最优（约25%/23%），unroll4 次之；老架构差距小。

因此，新架构要吃满带宽更依赖增大 `bytes‑in‑flight`；优先尝试向量化到 128bit（保证对齐），再配合适度 unroll，注意寄存器与占用权衡。

![](https://files.mdnice.com/user/59/1eccc616-02f2-4d15-b353-7b975382d3ab.png)

增加ILP（指令级并行/展开）和DLP（数据级并行/向量化）会增加寄存器压力。右侧图表显示了不同GPU世代（H100、H200、GB200-NVL）在寄存器使用率与峰值带宽之间的关系，特别指出在B200上达到SOL带宽需要使用40%的寄存器。

![](https://files.mdnice.com/user/59/0d9f6ec6-1817-4bb9-8e47-3c4359863b6e.png)

介绍了GPU内存复制机制的演进，从传统的普通加载（全局内存到寄存器）发展到Ampere架构引入的异步加载（全局内存到共享内存，支持多线程并行），再到Hopper架构的异步批量加载（支持16B到100+KB的大块数据传输，每个块只需一个线程处理），展现了NVIDIA在提升内存访问效率和减少线程开销方面的技术进步。

![](https://files.mdnice.com/user/59/f9a82b94-0ae3-4ebf-9267-a59f871c90dd.png)


继续讲了一下异步加载（Async Loads）技术的工作原理和优势：通过跳过寄存器直接将数据从全局内存传输到共享内存，既节省了宝贵的寄存器资源用于计算，又减少了L1缓存的流量负担，相比传统的同步复制（需要经过寄存器中转），异步复制提供了两种模式——L1绕过模式和L1访问模式，实现了更高效的内存数据传输。

![](https://files.mdnice.com/user/59/c66e1e0c-728a-40ed-bdb1-d7ddd0ba6a5a.png)

这张Slides展示了如何将传统的同步内存访问代码（左侧简单的kernel）改写为使用异步加载的版本（右侧）：通过引入CUDA pipeline机制，使用`cuda::memcpy_async`将全局内存数据异步复制到共享内存缓冲区，并通过`producer_commit`和`consumer_wait_prior`进行同步控制，从而在保持代码逻辑相似性的同时实现更高效的内存访问模式。

![](https://files.mdnice.com/user/59/d860aedb-5098-4baf-94b9-47ecd891e2c4.png)

这张Slides比较了异步批量加载（TMA）与普通异步加载的区别：普通异步加载需要多个线程参与复制并在线程作用域pipeline中完成同步，而TMA只需要一个线程启动复制操作，通过共享内存屏障完成同步控制，实现了更高效的大块数据传输，减少了线程协调的开销。

![](https://files.mdnice.com/user/59/87bbc65e-f8b1-4399-829b-fc7325521859.png)

这里展示了一下Async Bulk Loads的代码示例实现：通过`cuda::memcpy_async`，只需单个线程就能启动大块数据复制，使用共享内存屏障（`cuda::barrier`）进行完成同步控制，当源/目标地址16字节对齐且大小为16的倍数时会自动使用TMA硬件加速，否则回退到同步复制模式。

![](https://files.mdnice.com/user/59/fa842750-061c-4882-aea6-91debf201b83.png)


这张Slides详细对比了四种GPU内存加载方式及其特点：
- 同步加载（Synchronous Loads）：无对齐约束，无额外好处，是最基础的加载方式。
- 异步加载（Async Loads）：需要4、8或16字节对齐，额外好处是批处理复制可以增加in-fight字节数，提高内存带宽利用率。
- 异步批量加载（Async Bulk Loads）：要求16字节对齐，额外好处是批处理复制减少指令数量，降低指令开销提升效率。
- 异步批量张量加载（Async Bulk Tensor Loads）：对齐要求最严格（SMEM需128字节、GMEM需16字节、步长必须是16字节倍数），但提供最丰富的额外好处：SMEM交换能力和越界处理，支持更复杂的内存访问模式。
- 表格底部特别强调：为了获得最佳性能，应始终优先选择全局内存和共享内存都128字节对齐的方案。

![](https://files.mdnice.com/user/59/4436e812-84a9-41a6-abc4-798b7fd9d772.png)

这张Slides展示了一个GPU内存优化的决策流程图，指导开发者如何根据具体情况选择最佳的内存访问策略：从"START HERE"开始，首先判断当前是否有足够的在途字节数（bytes-in-flight），如果足够就无需优化；如果不够，接下来判断数据加载目标是寄存器（REG）还是共享内存（SMEM）——选择寄存器就进行展开/向量化优化，选择共享内存则需要进一步考虑对齐方式和数据块大小：4/8字节对齐时使用异步加载，16字节对齐时根据Tile大小选择策略（小于1KB用普通异步加载，1-2KB之间可选择批量或非批量异步加载，大于2KB则使用异步批量加载），这个决策树帮助开发者系统性地选择最适合当前场景的内存优化技术。

![](https://files.mdnice.com/user/59/686f239b-d79e-4006-854b-b6e40978bf0b.png)

随着H100、B200等新一代GPU拥有更高的带宽密度（每个流式多处理器SM的带宽更大），为了充分利用这些带宽需要更多的in-fight字节数来隐藏内存延迟，而传统方法通过展开/向量化来增加in-fight字节数会消耗大量寄存器资源，这在寄存器有限的情况下成为瓶颈，因此解决方案是采用异步加载技术，将原本存储在寄存器中的in-fight数据转移到共享内存中，既能保持足够的in-fight字节数来饱和内存带宽，又能节省宝贵的寄存器资源用于计算，这正是现代GPU架构下内存优化的核心策略。

![](https://files.mdnice.com/user/59/023a4b9c-3043-4144-95a7-2d32620755c2.png)

![](https://files.mdnice.com/user/59/683ae458-b542-4044-a27b-cbd20a66576a.png)

这张Slide探讨了小规模问题在GPU内存优化中面临的挑战：左侧展示了一个简单的向量加法kernel，通过计算可知每个SM有足够的in-fight字节数（64KB = 2048线程 × 每线程2次加载 × 每次加载16B），理论上应该能达到速度限制带宽（SoL BW），但右侧的性能曲线图清楚地显示了问题所在——即使单个SM层面的优化已经到位，在小规模问题（数据传输量小于100MB）时，整体系统仍无法达到峰值带宽，只有当问题规模足够大（接近1GB以上）时才能真正饱和DRAM带宽达到7TB/s的峰值，这揭示了GPU架构中小规模问题固有的效率瓶颈，说明内存优化技术虽然重要，但问题规模本身也是影响性能的关键因素。


![](https://files.mdnice.com/user/59/c934928d-4588-42c7-802e-7de623bf05ee.png)

这张Slides对比了B200与H20在处理不同规模问题时的性能差异，揭示了一个重要的扩展性问题：图表将问题规模分为三个区域——小问题（红色区域，10KB-1MB范围）完全没有性能提升，两代GPU都表现相当且远未达到峰值带宽；中等问题（黄色区域，1MB-100MB范围）显示弱扩展特性，B200相比H20有1x到2x的性能提升；只有大问题（绿色区域，100MB以上）才展现强扩展特性，B200达到预期的约2倍性能提升并能充分利用其7TB/s的峰值带宽，这个分析表明虽然新一代GPU硬件能力大幅提升，但小规模问题由于无法充分激活GPU的并行计算和内存带宽潜力，导致硬件升级带来的性能收益微乎其微，从而提出了"我们能做什么？"这一关键问题，暗示需要新的优化策略来解决小规模问题的性能瓶颈。

![](https://files.mdnice.com/user/59/7a93c4af-fa90-49ad-aade-bda882bd000c.png)

![](https://files.mdnice.com/user/59/755dfc8e-1742-4800-b4e4-429b669fe95d.png)

这张Slides分析了小规模问题性能优化的目标和策略：我们的目标是"让曲线左移"，即在相同问题规模下提高实际达到的带宽，通过减少总运行时间来实现这一目标，而减少总运行时间需要针对性地降低延迟；在延迟分析中，内存延迟是硬件固定的无法改变，块启动延迟虽然存在但不是关键因素（因为单波次需要148个SM × 每SM 64KB ≈ 10MB数据，如图中蓝点所示），真正的关键是kernel启动延迟会影响所有问题规模，这是制约小规模问题性能的主要瓶颈，因此优化重点应该放在减少kernel启动开销上，从而让性能曲线在较小的数据传输量下就能达到更高的带宽水平，实现图中绿色箭头所示的性能提升效果。

![](https://files.mdnice.com/user/59/7423ee1f-f882-49f1-8d2a-786d9e73ae1a.png)

这张Slides介绍了用于研究小规模问题性能瓶颈的实验设置：使用一个简单的向量加法kernel（左侧代码），该kernel接收三个float4指针参数并执行基本的元素相加操作，实验方法是将这个kernel运行1000次以获得稳定的性能测量，同时通过轮换使用a、b、c数组来避免L2缓存命中带来的性能干扰，确保每次都是真实的内存访问，然后在不同的数据规模下测量实际达到的带宽性能，通过这种控制变量的方法来量化kernel启动延迟对不同问题规模的具体影响，右侧的性能曲线图显示了B200 GPU在这种设置下的带宽表现，绿色箭头指示的优化目标就是通过减少启动开销让整个性能曲线向左移动，使得即使在较小的数据传输量下也能达到更高的有效带宽。

![](https://files.mdnice.com/user/59/c7f6e246-eb67-4fb6-a3a3-fc6ea6d51ebb.png)

这张Slides展示了使用CUDA Graph技术来减少kernel启动延迟的解决方案：左侧代码演示了CUDA Graph的完整使用流程，包括四个关键阶段——捕获阶段（Capture）通过cudaStreamBeginCapture和cudaStreamEndCapture将1000次kernel调用序列记录成图结构，创建阶段（Create）用cudaGraphInstantiate将捕获的图转换为可执行的图实例，启动阶段（Launch）通过cudaGraphLaunch一次性提交整个工作负载而不是逐个启动kernel，最后清理阶段（Cleanup）释放相关资源；右侧性能曲线对比图清楚地显示了CUDA Graph技术的效果，绿色区域标注的50%性能提升证明了通过批量提交kernel调用、减少CPU-GPU交互开销，确实能够显著改善小规模问题的带宽利用率，让性能曲线向左移动，在较小的数据传输量下就能达到更高的有效带宽，这是解决小规模问题性能瓶颈的有效技术手段之一。

![](https://files.mdnice.com/user/59/759c1895-ba10-423a-8698-dba1ac7050e6.png)

这张Slides介绍了减少kernel启动延迟的另一种技术：程序化依赖启动（Programmatic Dependent Launch, PDL），这是一种更细粒度的优化方法：左侧代码展示了具体实现，在kernel函数中需要调用`cudaGridDependencySynchronize()`来确保前一个kernel的全局内存写入操作对当前kernel可见，而在启动配置中通过设置`cudaLaunchAttributeProgrammaticStreamSerialization`属性和相关参数来启用这一特性；右侧说明了该技术的核心优势——允许kernel提前启动，即在前一个kernel的全局内存存储操作完全可见之前就开始执行，这样可以实现更多的预取操作（如kernel参数、常量存储等），从而进一步隐藏启动延迟，但为了保证程序正确性，kernel必须在需要访问前一个kernel写入的数据之前显式调用`cudaGridDependencySynchronize()`进行同步，这种技术相比CUDA Graph提供了更精细的控制，可以在保证数据依赖关系正确的前提下最大化地重叠kernel启动和执行过程。

![](https://files.mdnice.com/user/59/3d0af22c-76cb-4a85-be03-bc4826af0bf9.png)

这张Slides展示了将CUDA Graph和程序化依赖启动（PDL）技术结合使用所带来的累积性能提升效果：左侧的kernel代码简化了之前复杂的启动配置，仅需在kernel内部调用cudaGridDependencySynchronize()来处理数据依赖关系，而右侧的性能对比图清楚地展示了优化的渐进效果——浅灰色曲线代表仅使用CUDA Graph技术的性能表现，深灰色曲线则是CUDA Graph+PDL的组合效果，绿色区域标注显示性能提升从单独使用CUDA Graph的50%进一步提升到了50%→70%，这表明两种技术的结合产生了叠加优化效果：CUDA Graph解决了批量kernel提交的开销问题，而PDL进一步通过提前启动和更细粒度的依赖管理减少了kernel间的同步等待时间，最终实现了更显著的小规模问题性能改善，证明了通过系统性地减少启动延迟可以有效地让GPU性能曲线向左移动，在更小的数据传输量下就能达到更高的有效带宽。

![](https://files.mdnice.com/user/59/960dc7c7-cae6-4080-a7cc-0d12232e071d.png)

这张幻灯片介绍了kernel启动延迟优化的最新技术：在PDL基础上添加`cudaTriggerProgrammaticLaunchCompletion函`数调用，实现更细粒度的启动控制：左侧代码显示在kernel中先调用`cudaGridDependencySynchronize()`确保数据依赖，然后立即调用`cudaTriggerProgrammaticLaunchCompletion()`发送"提前完成"信号，最后才执行实际的计算工作；右侧说明了这一技术的核心机制——该函数允许块在实际退出之前就向GPU调度器发送完成信号，只需要块中的一个线程执行该调用即可，从而改变了传统的kernel间同步模式：正常情况下下一个kernel必须等待前一个kernel的所有块真正退出后才能启动，而现在下一个kernel只需等待所有块都执行了`cudaTriggerProgrammaticLaunchCompletion()`就可以提前启动，这样就实现了计算和下一个kernel启动的重叠，进一步减少了kernel间的空闲等待时间，是对PDL技术的精细化扩展。

![](https://files.mdnice.com/user/59/724978b0-6bac-480c-b145-f3264977fb7c.png)

这张Slides展示了完整的kernel启动延迟优化技术栈的最终性能效果：左侧代码整合了所有优化技术，包括`cudaGridDependencySynchronize()`用于数据依赖同步和`cudaTriggerProgrammaticLaunchCompletion()`用于提前完成信号，右侧性能对比图清楚地展现了渐进式优化的累积效果——浅灰色曲线代表仅使用CUDA Graph的基线性能，深灰色曲线显示CUDA Graph+PDL的组合效果，而绿色曲线则展示了CUDA Graph+PDL+提前退出（EarlyExit）的最终优化结果，绿色区域标注的性能提升从之前的70%进一步提升到75%，虽然`cudaTriggerProgrammaticLaunchCompletion()`带来的额外5%提升相对较小，但这展示了通过系统性地结合多种启动延迟优化技术，可以实现显著的累积性能改善，最终让GPU在小规模问题上也能达到更高的有效带宽利用率，证明了精细化的kernel启动控制对于现代GPU性能优化的重要价值。

![](https://files.mdnice.com/user/59/e6fecb77-d026-409b-b357-b125d225de30.png)


这张Slides通过双重视角全面展示了组合优化技术的整体效果：左侧的加速比图表显示了在不同数据传输规模下的性能提升效果，揭示了一个重要的规律——在最小规模（10KB左右）时，三种组合优化技术可以实现高达3倍的加速比，但随着数据量增加，加速效果逐渐递减，当数据量达到1GB以上时加速比趋近于1倍（即基本无提升）；右侧的DRAM带宽图表从另一个角度确认了同样的优化效果，在小规模数据传输时实现了125%的性能提升，在中等规模时仍有75%的提升；这种对比清楚地说明了kernel启动延迟优化技术的价值主要体现在小规模问题上——正是在传统GPU优化技术效果有限的场景下，通过CUDA Graph、PDL和提前退出的组合应用，成功解决了小规模问题的性能瓶颈，而对于大规模问题由于其本身就能充分利用GPU资源，这些启动优化技术的边际效益相对较小，体现了针对性优化的重要意义。

![](https://files.mdnice.com/user/59/6f068174-c202-42ce-b74a-16a2810b79a7.png)


这张Slides是对小规模问题优化策略的精炼总结：它首先明确了GPU性能优化的分层策略——硬件升级对大规模问题有效，因为大规模问题能够充分激活GPU的并行计算资源和内存带宽；但对于小规模问题，关键在于通过软件技术减少kernel启动延迟这一核心瓶颈，具体包括三个层次递进的优化技术：CUDA Graph（通过批量提交减少CPU-GPU交互开销）、PDL（实现kernel间的细粒度依赖控制和提前启动）、以及提前退出（通过提前发送完成信号实现计算与下一个kernel启动的时间重叠），通过这些纯软件层面的优化技术组合，可以在小规模问题上实现高达3倍的性能提升，这个总结强调了一个重要观点：在现代GPU架构下，当问题规模无法天然地利用硬件优势时，精细化的软件优化技术就成为弥补性能差距的关键手段，特别体现了针对性优化的价值和重要性。


