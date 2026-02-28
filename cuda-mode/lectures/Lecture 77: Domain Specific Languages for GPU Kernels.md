> 我的课程笔记，欢迎关注：https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode 。

> 这个课程中有关Cute-DSL的代码可以在 https://github.com/Dao-AILab/quack 这里找到。这节课由Tri Dao介绍了GPU kernel开发的DSL生态，从PyTorch、Triton到Cute-DSL，展示了不同抽象层次的性能与开发效率权衡，并通过Softmax，GEMM，Attention展示了各DSL的性能，控制的硬件层次，开发难度。来源：https://www.youtube.com/watch?v=5qSN-R_E3w0


![](https://files.mdnice.com/user/59/79a2d3af-c9ea-4c0f-845e-5e10da43d5af.png)

这里Tri Dao将介绍用于GPU kernels的特定领域的语言。

![](https://files.mdnice.com/user/59/b7201d9e-1373-4576-ac80-834d2b318677.png)

![](https://files.mdnice.com/user/59/148022e2-4f62-4728-828a-3e4648e929f6.png)

Tri Dao提出了一个计算效率评估公式：Intelligence/Dollar = (Intelligence/FLOPS) × (FLOPS/Dollar)。这个公式把AI性价比拆解成两部分：算法效率（Intelligence/FLOPS）和硬件效率（FLOPS/Dollar）。

![](https://files.mdnice.com/user/59/a464e467-0ab0-483e-8813-8fa333ba01b7.jpg)

为什么需要DSL？这张图用韦恩图说明了原因：算法研究追求更好的模型，硬件优化追求更好的扩展性，DSL正好处在两者交集，既能保证研究生产力又能充分利用硬件。另外一个bonus是DSL + 好的抽象让LLM更容易生成高效的GPU kernels。

![](https://files.mdnice.com/user/59/06b2e9f8-7811-4ec2-82de-ab424c681d3e.png)

我会讨论一系列DSL以及基于它们开发的softmax和gemm以及attn的perf结果。

![](https://files.mdnice.com/user/59/f34720bf-cee8-464e-8f08-78f7bf792cf4.jpg)

首先我认为PyTorch是第一个DSL，通过PyTorch编写程序让代码变成在GPU上运行的一系列kernels。然后PyTorch 2.x利用Dynamo可以捕获你的程序让它通过Triton去编译执行。

![](https://files.mdnice.com/user/59/b8313bed-512d-4a1f-a9a6-b6b8638dafbc.png)


第二个DSL是Triton。对比表格显示了Triton相比CUDA的优势：内存合并、共享内存管理、SM内调度这三个方面，CUDA需要手动优化，Triton可以自动处理。只有跨SM调度两者都需要手动管理。Triton的tile-based编程模型让开发者专注算法逻辑，不用操心底层优化细节。

![](https://files.mdnice.com/user/59/d8d43246-ad2b-4056-808f-f9c30b9d63aa.jpg)

第三个DSL是Cute-DSL，把Cutlass C++嵌入到Python中。用电梯比喻来说：Triton像直达电梯（高层抽象），Cutlass像自动扶梯（精细控制），PTX像螺旋楼梯（底层细节）。Cute-DSL结合了Cutlass的高性能和Python的易用性。

![](https://files.mdnice.com/user/59/58d20531-ee29-4baa-bf31-5d5317dddfa4.jpg)

Cute-DSL相比Triton的重要优势：能暴露GPU完整的四层线程/内存层次结构。从上到下分别是：线程寄存器和本地内存、块共享内存、集群分布式共享内存、全局内存。Triton只能暴露线程块和网格两层，限制了硬件控制能力。Cute-DSL暴露完整层次结构，提供更细粒度的内存管理和线程协调。

![](https://files.mdnice.com/user/59/d294727e-a857-4444-a683-7b57eaae7cbc.jpg)

其他值得关注的DSL工具：ThunderKittens（斯坦福开发，简单快速的AI kernel框架）、TileLang（基于tile的GPU编程抽象）、Mojo（结合Python易用性和系统级性能）、Mosaic GPU（另一种GPU编程抽象）。

![](https://files.mdnice.com/user/59/776b9a31-d0d0-4b00-ae9d-f0dff7855638.jpg)

两个Triton扩展项目：Gluon（基于Triton编译器技术的更低级语言，暴露对Layout、调度和内存的精细控制）和TLX（low-level、warp-aware的Triton扩展，提供硬件特定内置函数如wgmma、async_copy、barrier等）。这两个项目都为专家用户提供更接近硬件的控制能力。

![](https://files.mdnice.com/user/59/3dca4df8-5160-4b55-9ff1-7a1746e44354.png)

这里选了一个Softmax的例子来看不同DSL的实现，首先是Torch Compile。

![](https://files.mdnice.com/user/59/e395cc3b-26bd-4b0a-9d64-dfde8495cbbc.png)

Liger Kernel的SoftMax Triton实现，代码简洁。使用`@triton.jit`装饰器，核心逻辑：获取行ID和偏移量→加载数据并处理边界→计算最大值（数值稳定）→计算exp和归一化→输出结果。

![](https://files.mdnice.com/user/59/b84808ee-28d3-43f1-b351-006e9283fd8f.png)

Triton Softmax的multi-block实现，用分块策略处理更大序列。两个主要循环：第一个计算全局最大值和累积指数和（使用在线算法更新统计信息），第二个基于全局统计信息计算最终softmax输出。这种实现能处理超出单块能力的大规模数据。

![](https://files.mdnice.com/user/59/a0276594-9aca-40c4-89e3-615d6d7e950f.jpg)

Cute-DSL在Softmax中使用async copy优化。相比Triton更精细的硬件控制：分配共享内存→创建异步拷贝原子操作（支持CopyG2SOp等硬件指令）→异步数据传输→管理提交和同步。右侧图显示合并内存访问优化，实现计算与内存传输重叠，提升kernel性能。

![](https://files.mdnice.com/user/59/33f419e7-87ee-4978-8383-aa2502c9638a.png)

Cute-DSL中的thread reduction实现。使用CuTe的`TensorSSA.reduce(op, init_val, reduction_profile)`接口，示例：`max_X = X.reduce(cute.ReductionOp.MAX, init_val=float('-inf'), reduction_profile=0)`。右侧图显示Thread 0处理一系列数值，通过归约操作合并到单个寄存器。

![](https://files.mdnice.com/user/59/f2d51623-f30c-4b43-aa8a-72a059e92d7a.jpg)

Cute-DSL中warp reduction实现。自定义`warp_reduce`函数使用`@cute.jit`装饰器，通过循环和`cute.arch.shuffle_sync_bfly`指令执行butterfly模式的warp内归约。右侧图显示butterfly归约过程：32个线程通过多轮shuffle操作逐步归约到单个结果。充分利用GPU的warp同步原语，实现高效多层次归约。

![](https://files.mdnice.com/user/59/640f7f6b-e75c-4186-a75a-5b9305215395.jpg)

Cute-DSL中的thread block reduction实现。`block_reduce`函数实现多阶段归约：(1)每个warp的lane 0将warp归约结果写入共享内存缓冲区→(2)同步等待所有写操作完成→(3)部分线程从缓冲区读取数据进行进一步归约→(4)调用`warp_reduce`完成最终块级归约。

![](https://files.mdnice.com/user/59/ebe10beb-206d-4d02-80de-77a6591e1215.jpg)

Cute-DSL的cluster reduction（最高层次归约）。左图：每个warp将归约结果写入本块和集群内其他块的归约缓冲区（得益于H100分布式共享内存）。右图：每个warp从本块缓冲区读取数据进行最终归约。这种设计突破了传统GPU编程中线程块间无法直接通信的限制。

![](https://files.mdnice.com/user/59/e3ad0771-9380-428b-9261-47ff9028676c.png)

Cute-DSL Softmax的完整归约层次结构。代码流程：加载数据到寄存器→线程归约→warp归约→条件判断（如果每行有多个warp）→根据集群配置选择块归约或集群归约。这种自适应设计能根据不同GPU配置选择最优归约策略。

![](https://files.mdnice.com/user/59/2554df27-ec72-4736-840f-7d9700b0b0ec.jpg)

Softmax性能对比（H100, bf16, M=32k）：Torch compile（蓝）、Liger Triton（橙）、Quack Cute-DSL（绿）。小序列长度（1k-4k）时性能相近，序列增长后差异显现。图中标注了两个关键区域：Warp reduction w/o block reduction和Cluster reduction。Cute-DSL在大部分情况下表现最佳，特别是长序列场景下保持稳定高性能。


![](https://files.mdnice.com/user/59/6e718115-6c21-4f24-84b5-3342f36a2db6.png)

![](https://files.mdnice.com/user/59/c9e58056-6ae3-487f-9914-08fe57c8a56c.jpg)

Hopper架构GEMM A@B性能对比（bf16, M=N=8k）：cuBLAS 13.0（蓝）vs Cute-DSL（橙）。Cute-DSL在所有测试点都优于cuBLAS，特别K=2k-3k区域（Pingpong to overlap epilogue）Cute-DSL达800 TFLOPS，cuBLAS仅760 TFLOPS。Cute-DSL在整个K范围保持稳定高性能，体现了对Hopper硬件特性的精细控制能力。

![](https://files.mdnice.com/user/59/ddedfcb7-e405-4be4-9d88-278384101105.jpg)

Ping-Pong Schedule的介绍可以看PyTorch的blog: https://pytorch.org/blog/cutlass-ping-pong-gemm-kernel/

![](https://files.mdnice.com/user/59/11bc3899-06a3-4f5e-844a-395860b96ac5.jpg)

在Blackwell上cuBlas的性能暂时比基于Cute-DSL的要好，不过这个3%的gap应该会被解决。

![](https://files.mdnice.com/user/59/b51c2c69-6c26-4677-acb8-2393eb0ff9a8.png)

![](https://files.mdnice.com/user/59/bfbaf547-a3e7-4d9d-9f9f-8bec341afabc.png)

Hopper架构GEMM + SwiGLU融合操作性能对比（bf16, M=8k, N=5.3d, K=d）：cuBLAS + Triton（蓝）vs Cute-DSL（橙）。Cute-DSL在所有测试点都显著优于组合方案，Cute-DSL稳定在约790 TFLOPS，而cuBLAS + Triton从530 TFLOPS提升到740 TFLOPS但始终较低。通过epilogue融合技术，Cute-DSL在常用GEMM操作中实现7-15%性能提升。

![](https://files.mdnice.com/user/59/19160574-8180-4e22-a8f8-7bd0872f01b7.jpg)

这张Slides展示了基于Cute-DSL的FA4相比于CUDNN实现的Flash Attention的优势。


![](https://files.mdnice.com/user/59/563b35a3-0cd4-4348-b30f-3103e65bfc34.png)


这里总结一下几种DSL的位置，Torch位于生产力最高但性能相对较低的位置，Triton在中间提供了良好的平衡，而Cute-DSL、CUDA和PTX则位于高性能但需要更多开发努力的区域。下方的对比表格提供了具体的量化数据：在内存受限场景下，Torch compile和Triton都能达到约90%的性能，而Cute-DSL能达到100%；在计算受限场景下，Torch compile约为70-80%，Triton为80-90%，Cute-DSL仍为100%；在上手时间方面，Torch compile只需几小时到几天，Triton需要几天到几周，而Cute-DSL则需要几周到几个月。

![](https://files.mdnice.com/user/59/bb3cee01-1591-44d0-8881-3cab3e4deac3.png)

最后给出了几个使用建议。


补充一个SoftMax相关的benchmark，在h200测试的：

```shell
softmax-bandwidth:
    token_num  hidden_size  HuggingFace  Torch Compile  FlashInfer        Quack
0       512.0       4096.0   498.372634      11.278891  404.543218   324.435651
1       512.0       8192.0   627.889810      22.135866  524.025960  1074.360676
2       512.0      16384.0  1044.398446      44.155216  572.679419  1409.376308
3       512.0      32768.0  1151.648537     889.566057  517.432039  1539.759166
4      1024.0       4096.0   667.882818     633.198054  518.071135  1081.006193
5      1024.0       8192.0   733.783037     812.534647  551.012070  1506.574707
6      1024.0      16384.0  1115.506385     918.997405  547.845344  1632.024895
7      1024.0      32768.0  1258.039589     957.603691  539.044331  1790.906990
8      2048.0       4096.0   790.781316     771.863094  535.534216  1510.916454
9      2048.0       8192.0   726.412168     894.880339  550.433609  1706.388872
10     2048.0      16384.0  1254.277480    1000.549669  584.490511  1815.716100
11     2048.0      32768.0  1287.979119    1059.033959  566.166629  1952.201120
12     4096.0       4096.0   854.237044     839.196500  537.180338  1706.388872
13     4096.0       8192.0   788.847816     961.996300  587.684453  1833.976427
14     4096.0      16384.0  1357.599659    1069.498055  608.487442  1965.926353
15     4096.0      32768.0  1425.421950    1111.957558  580.094950  2024.522377
16     8192.0       4096.0   914.788178     897.753389  571.275400  1837.189726
17     8192.0       8192.0   821.687552    1006.552476  603.366738  1971.470724
18     8192.0      16384.0  1414.605063    1108.577756  621.839001  2029.174741
19     8192.0      32768.0  1462.512876     609.338297  246.685668  2061.589571
20    16384.0       4096.0   958.041157     931.756935  591.205008  1974.719297
21    16384.0       8192.0   837.792610    1051.467569  609.725833  2027.213223
22    16384.0      16384.0  1443.542907     545.893445  253.550994  2056.472343
23    16384.0      32768.0   543.884852     489.570549  247.817994  2086.134646
24    32768.0       4096.0   968.549584     964.540379  599.464617  2030.402447
25    32768.0       8192.0   286.402022    1067.694412  252.425616  2055.716456
26    32768.0      16384.0   535.658202     485.785690  255.072718  2079.154372
27    32768.0      32768.0   778.791610     492.339755  266.321390   945.948067
```


可以看到Quack基本所有的情况下相比于Torch Naive/Torch Compile/FlashInfer带宽都是最高的。




















