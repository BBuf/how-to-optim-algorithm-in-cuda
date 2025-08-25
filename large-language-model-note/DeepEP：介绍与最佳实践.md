
![](https://files.mdnice.com/user/59/ff7b2032-cb6b-49fb-9554-eebe12b43e20.png)

这个演讲的目录，主要包含6个内容。

- DeepEP overview - DeepEP概述，介绍这个技术框架的基本情况
- DeepEP dispatch/combine introduce - 介绍DeepEP的调度和合并机制
- DeepEP performance limit - 讨论DeepEP的性能限制和瓶颈
- Critical Updates in DeepEP - DeepEP中的关键更新内容
- New Design Principle for PCIE GPU - 针对PCIe GPU的新设计原则
- DeepEP Deployment Guide - DeepEP部署指南

![](https://files.mdnice.com/user/59/7e93185f-f296-4eeb-afff-70b34f118ab8.png)

这张Slides介绍了一下DeepEP的架构概述和核心特性。DeepEP是一个专门为MoE（混合专家模型）和EP（专家并行）设计的通信库。主要特性包括：
- 优化的alltoall通信：专门针对推理中的稀疏通信进行了优化，能够更好地支持MoE模型中的dispatch和combine过程。
- 多种通信支持：
  - 节点内通信：支持NVLINK和RDMA
  - 节点间通信：支持RDMA
- 双kernel选择：
  - Normal Kernel：用于训练和推理的预填充阶段
  - Low-Latency Kernel：专门用于推理的解码阶段，优化延迟性能

下方的架构图显示了DeepEP的分层设计，从上层的Normal Kernel和Low Latency Kernel，到中间的PTX执行层和基于NVshmem通信库中的ISRC、IBGDA协议，最后到RDMA网络层。这里提到NVSHMEM 是 NVIDIA 开发的并行编程接口，旨在为 GPU 集群提供高效且可扩展的通信能力。它基于 OpenSHMEM 规范，创建了一个跨多个 GPU 内存的全局地址空间，允许通过 GPU 发起的异步数据传输，减少 CPU 和 GPU 之间的同步开销。NVSHMEM 支持多种通信方式，包括节点内的 NVLink 和 RDMA，以及节点间的 RDMA，适用于高性能计算（HPC）系统中的大规模 GPU 集群。

![](https://files.mdnice.com/user/59/9d9ce91f-f69a-466d-8902-735161d7cff8.png)


这张Slides展示了DeepEP中dispatch和combine机制在Normal Kernel和Low Latency Kernel两种模式下的不同工作原理。图中展示了两个节点（NodeA和NodeB），每个节点配置8个GPU（GPU0-GPU7），通过RDMA进行跨节点通信，通过NVLink（NVL）进行节点内通信。

**Normal Kernel模式（左侧）：** 在这种模式下，dispatch和combine过程相对复杂，涉及更多的跨节点和跨GPU通信。从箭头流向可以看出，数据需要在多个GPU之间进行复杂的路由和转发，包括从NodeA的某些GPU向NodeB的对应GPU进行数据传输，通信路径较为密集，在节点间通信时不仅有NVLink的路径还有RDMA的路径。

**Low Latency Kernel模式（右侧）：** 这种模式专门为推理场景的低延迟需求而优化。相比Normal Kernel，通信模式更加简化和直接，只使用RDMA，没用NVLink。

![](https://files.mdnice.com/user/59/81f61589-d53f-4220-b3eb-bea3a68de751.png)

这张Slides左侧API函数说明了四个关键步骤:

- `internode::get_dispatch_layout`：这是布局计算阶段，负责确定数据分发策略，包括计算topk索引数组（每个token对应的前k个expert）、每个rank处理的token数量、每个RDMA rank的token分配，以及每个expert将处理的token数量。
- `internode::notify_dispatch`：通知阶段，为每个rank/node/channel计算校验和，确保数据传输的完整性和一致性。
- `internode::dispatch`：实际分发阶段，通过RDMA或NVLink将token发送到对应的expert所在位置。
- `internode::combine`：聚合阶段，通过RDMA或NVLink接收处理完成的token并进行结果合并。

右侧的dispatch flow图展示了具体的数据流动过程：从NodeA的输入开始，经过GPU 0的RDMA接收缓冲区，然后分别流向GPU 1和GPU 2的nvlink缓冲区，最终输出结果。整个流程体现了DeepEP如何高效地在多GPU环境中处理MoE模型的expert并行计算，通过精心设计的布局计算、通知机制、数据分发和结果聚合，实现了优化的通信模式和数据流管理。

![](https://files.mdnice.com/user/59/dfb22e1c-0932-4929-b1e8-5925d64e08b9.png)

这张Slides详细展示了DeepEP Normal Kernel的性能限制和带宽瓶颈分析。

**DeepEP 官网的性能测试结果表格显示：**

- **节点内通信（Intranode）**：在8个专家并行的情况下，dispatch和combine操作都能达到约153-158 GB/s的NVLink带宽，性能表现最佳。

- **节点间通信（Internode）**：随着专家并行数量从16增加到64，RDMA带宽从会下降。

**理论带宽计算公式为：** `BW = (1+M/M*(N-1)) * NIC_BW`，其中M表示每个节点的GPU数量，N表示节点数量，NIC_BW表示每个网络接口的带宽。

**具体的理论带宽计算示例：**
- 2节点配置：(1+8/8)*50=100 GB理论带宽
- 4节点32专家：(1+8/24)*50=66.7 GB理论带宽  
- 8节点64专家：(1+8/56)*50=57.1 GB理论带宽

图片还解释了带宽计算的实际场景，例如在2节点配置中，NodeA的GPU0需要8/16的数据通过NVLink传输，8/16的数据通过RDMA传输，总体带宽受限于网络通信能力。这表明随着节点数量增加，跨节点的RDMA通信成为主要瓶颈，限制了整体的扩展性能。

这里EP 16似乎出现了一个bad case，演讲者说这个和演讲中开始提到的同步开销有关。

![](https://files.mdnice.com/user/59/56c46693-50e1-44fa-bda0-0b323ff9047a.png)

这张Slides展示了DeepEP Low Latency Kernel的性能限制和延迟特性，通过一个完整的性能测试表格揭示了随着专家并行数量从8增加到256，dispatch和combine操作在延迟和RDMA带宽方面的变化趋势。数据显示，在较小规模（8个专家）时，dispatch延迟仅为77微秒，RDMA带宽可达98 GB/s，combine操作延迟为114微秒，带宽为127 GB/s，但随着专家并行数量增加，延迟逐步上升而带宽逐渐下降，当达到256个专家时，dispatch延迟增加到194微秒，带宽降至39 GB/s，combine延迟更是达到360微秒，带宽为40 GB/s。图中还提供了具体的带宽计算示例，以单节点为例，dispatch数据大小为`128*7408*8=7.5MB`，通过77微秒的延迟时间计算得出98GB带宽，combine数据大小为`128*7168*2*8=14.5MB`，通过114微秒延迟计算得出127GB带宽，同时给出了理论带宽公式`BW=(1+M/M*(N-1))*NIC_BW`，并展示了2节点、4节点、8节点配置下的理论带宽分别为100GB、66.7GB、57.1GB，这些数据清楚地表明了Low Latency Kernel在大规模扩展时面临的性能瓶颈和延迟挑战。

需要注意，这里的128是token数，7408是hidden_size，8是EP8，图中Combine公式中的318us打错了，应该是114us。

![](https://files.mdnice.com/user/59/c0f1bfcd-bc3e-40bc-bff9-c60006ffebd4.png)


这张Slides展示了DeepEP Low Latency Kernel的一项重要性能更新，即通过在节点内使用NVLink来优化通信性能，图中通过左右两个架构对比清晰地说明了优化前后的差异：左侧的原始架构中，NodeA的多个GPU主要通过RDMA与NodeB的GPU0进行通信，然后由GPU0再分发到NodeB内的其他GPU，而右侧的更新架构中增加了更多的NVLink连接（用绿色箭头表示），使得节点内通信能够更好地利用高带宽的NVLink而不是依赖RDMA，从而显著改善了整体性能。底部的H20测试数据表格验证了这种优化的效果，显示了在不同专家并行规模（8、16、32、64个EP）下dispatch和combine操作的延迟和带宽改进情况，其中每个数据都用颜色标注了改进前后的对比，例如在8个专家并行的情况下，dispatch延迟从172微秒降低到37微秒，带宽从43.5 GB/s提升到203.2 GB/s，combine操作的延迟从319微秒降低到59微秒，带宽从45.5 GB/s大幅提升到244.2 GB/s，这些数据清楚地表明通过更好地利用NVLink进行节点内通信，DeepEP能够在保持低延迟的同时实现更高的通信带宽，显著提升了MoE模型推理的整体性能。

![](https://files.mdnice.com/user/59/dbad1e16-9624-4809-b693-4ea7fa33592a.png)


这张Slides展示了DeepEP Normal Kernel的重要更新，该更新基于腾讯TRMT-DEEP团队的最佳实践（相关PR链接为https://github.com/deepseek-ai/DeepEP/pull/130） ，主要包含三个关键技术改进：首先是采用Multi-QP替代single-QP以实现网络接口卡的全带宽利用，其次是使用IBGDA替代IBRC来获得更低的延迟，第三是添加amo QP锁机制来确保同步操作的正确性。图中通过左右两个架构对比清晰地展示了优化前后的差异，左侧显示原始DeepEP基于CPU的NIC通信方式（IBRC），数据需要经过CPU进行处理和转发，而右侧展示了TRMT-DeepEP采用的GPU直连NIC通信方式（IBGDA），GPU可以直接与网络接口进行通信，绕过CPU瓶颈，从而实现更高的带宽和更低的延迟。底部的性能改进表格量化地证明了这些优化的效果，在节点内通信（Intranode）方面，8个专家并行时NVLink带宽保持在153-158 GB/s的高水平，而在节点间通信（Internode）方面，通过这些优化技术，32个专家并行的RDMA带宽从原来的44 GB/s提升到58 GB/s，64个专家并行时从46 GB/s提升到51 GB/s，这些数据表明通过采用更先进的GPU直连通信架构和多队列优化策略，DeepEP在大规模MoE模型的分布式推理场景中能够实现显著的性能提升。

这里只是简单了解了一下概念，我不懂这个IBRC和IBGDA的区别，以及QP锁机制， 感兴趣的小伙伴需要自行研究。

![](https://files.mdnice.com/user/59/7d04d985-ba06-42a5-aae3-3e9fa1bab6fb.png)

这张Slides阐述了DeepEP针对PCIe GPU环境的设计原则和限制条件，重点说明了在没有NVLink高速互联的PCIe GPU环境中DeepEP的运行约束和缓冲区分配策略。图中明确指出DeepEP公共Normal模式无法在缺乏NVLink的环境中运行，同时Normal模式由于缓冲区分配机制的限制无法重用Low Latency的设计，这表明两种模式在内存管理上存在根本性差异。针对缓冲区分配，图中详细列出了不同模式下的内存计算公式：在Normal模式中，RDMA缓冲区大小约等于通道数乘以RDMA ranks数量乘以最大RDMA分块接收token数量乘以隐藏层字节数再乘以2，而NVLink缓冲区则不需要最后的乘2操作；在Low Latency模式中，RDMA缓冲区的计算更加复杂，约等于专家数量乘以每个rank的最大dispatch token数量乘以每个dispatch消息的字节数再乘以4（两个2相乘），这些精确的内存分配公式反映了DeepEP在不同硬件环境和工作模式下对内存资源的精细化管理，确保在PCIe GPU这种相对受限的硬件环境中仍能实现高效的MoE模型推理性能。

对于Decoding阶段，token数为128的时候这个buffer占用就大概个GB了，因此我们没办法直接沿用Low Latency Kernel的设计，因为Prefill时token数可能会很大，例如8K，16K这样buffer的内存就爆了。

![](https://files.mdnice.com/user/59/3b413af2-2eda-4a2c-815e-b54854197678.png)


这张Slides展示了DeepEP针对PCIe GPU环境的优化设计原则，特别是在没有NVLink高速互联的情况下如何实现高效通信。该设计巧妙地结合了Low Latency模式的拓扑结构和Normal模式的缓冲区分配策略，确保除了自身和相邻ranks之外的所有通信流量都通过网络接口卡（NIC）进行传输，从而最大化利用可用的网络带宽。左侧的架构图显示了NodeA和NodeB之间通过RDMA进行数据块传输的连接方式，其中GPU0作为主要的通信枢纽负责处理跨节点的数据交换，而右侧的Rank 0缓冲区布局详细说明了内存组织结构，包括发送缓冲区（Send Buffer）和接收缓冲区（Recv Buffer），以及多个通道（Channel 0到Channel N）的划分，每个通道下又细分为多个RDMA ranks（从Rank 0到Rank M），底部的公式num_max_chunked_recv_tokens * token_size表明了缓冲区大小的计算方法。最重要的是，这种设计无需修改任何代码就能直接在现有框架（如sglang）中运行，并保持所有功能特性（如TBO），这使得DeepEP能够在PCIe GPU环境中实现即插即用的高性能MoE模型推理，为那些无法配置NVLink但仍需要高效专家并行计算的用户提供了实用的解决方案。


![](https://files.mdnice.com/user/59/1663a3ed-e21b-4f28-b43c-7108e7a163c8.png)

这张Slides提供了DeepEP的完整部署指南，详细展示了从环境准备到最终安装的四个关键步骤的具体操作流程。首先是Docker环境构建阶段，需要拉取NVIDIA PyTorch 24.04-py3镜像，然后使用特定的docker run命令启动容器（包括GPU访问权限、特权模式、主机网络、数据卷挂载等配置），并在容器内创建libmlx5.so.1到libmlx5.so的软链接以确保RDMA库的正确链接；接下来是源码获取阶段，需要进入工作目录并从GitHub克隆DeepEP仓库（https://github.com/deepseek-ai/DeepEP.git） ，同时使用wget下载NVSHMEM 3.2.5-1版本的源码包，解压后重命名目录并应用DeepEP提供的补丁文件；第三步是NVSHMEM的编译构建，需要设置一系列复杂的编译参数（包括启用IBGDA支持、禁用SHMEM、UCX、NCCL、PMIX等多个组件支持，并指定使用Ninja构建系统和特定的安装前缀），然后执行cmake构建和安装；最后是DeepEP本身的构建安装，需要进入DeepEP目录，设置TORCH_CUDA_ARCH_LIST为9.0+PTX以支持相应的CUDA架构，指定NVSHMEM_DIR为之前安装的路径，最终通过python setup.py install完成整个DeepEP系统的部署，整个过程涉及Docker容器化、依赖库编译、环境变量配置等多个技术环节，确保DeepEP能够在目标GPU集群环境中正常运行。

![](https://files.mdnice.com/user/59/079913b8-91e8-40d0-96b7-3d7f0bb43aa0.jpg)

这张Slides详细展示了DeepEP的用户使用指南，涵盖了单节点和多节点两种部署场景的具体操作方法和性能测试结果。在单节点模式下，用户可以分别运行Normal模式（python tests/test_intranode.py）和Low Latency模式（python tests/test_low_latency.py）进行测试，而在多节点模式下，需要在每个节点上配置相同的环境变量包括MASTER_ADDR（主节点IP地址10.6.131.5）、WORLD_SIZE（节点总数2）、MASTER_PORT（通信端口40303），并为每个节点设置不同的RANK值（node 0设为0，node 1设为1），然后运行相应的测试脚本（test_internode.py或test_low_latency.py）。右侧的测试输出结果显示了在H20 GPU环境中运行Low Latency模式的详细性能数据，包括2115MB的缓冲区分配以及8个rank的dispatch和combine操作的带宽和延迟指标，其中每个rank的dispatch+combine带宽都稳定在179-180 GB/s左右，平均延迟约为122微秒，同时还显示了更详细的dispatch和combine分离统计，dispatch带宽约193-203 GB/s，combine带宽约184-196 GB/s，这些数据充分证明了DeepEP在多GPU环境中的高效通信性能，为用户提供了清晰的性能基准和部署参考。

