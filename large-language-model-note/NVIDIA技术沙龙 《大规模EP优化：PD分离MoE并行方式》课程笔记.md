> Slides来自BiliBili NVIDIA英伟达频道上传的NVIDIA专家面对面技术沙龙《大规模EP优化：PD分离MoE并行方式》视频讲解。这里参考视频并更详细记录了Slides的要点，作为学习使用。这个演讲对应的实际上就是SGLang的Large-Scale EP+PD分离的方案，部分Slides也是从SGLang博客 https://lmsys.org/blog/2025-05-05-large-scale-ep/ 摘出来的。


![](https://files.mdnice.com/user/59/2695746e-fded-4b72-b000-6f46cdaa4a05.jpg)

![](https://files.mdnice.com/user/59/adae078e-bfa5-4c5d-9709-49881faeb102.png)

这张Slides主要讲述了五个关键技术创新：
- PD分离（Prefill-Decode Disaggregation）通过将Compute-Bound的Prefil阶段和Memory-Bound的Decoding阶段分离到专用服务器，避免了中断并优化GPU利用率，通过非阻塞RDMA传输将延迟降低50%；
- 大规模专家并行（Large-Scale EP）使用DeepEP将MoE专家权重分布到多个GPU上，并集成EPLB专家并行负载均衡，在规模化场景下将吞吐量提升1.5-2.5倍；
- 算子优化包括集成DeepGEMM加速MoE矩阵乘法，以及DP Attention与DP Dense FFNs(针对DeepSeek-V3/R1的前面3层的Dense ): 消除KV Cache重复，减少50%通信开销；
- TBO（双批次Overlap）将批次分割为Micro-Batch来重叠计算和通信，提升Prefill吞吐量27-35%并支持更大批次；
- MTP（多Token预测）使用轻量级Draft模型并行预测和验证未来Token，与大规模EP和PD分离平滑结合，提供高达60%的输出吞吐量提升。

这些技术共同构成了一个高效的大规模推理系统优化方案。

![](https://files.mdnice.com/user/59/41ba0002-0f62-4ab3-8098-bf0650cc16eb.jpg)

本次演讲的目录：背景与动机（Background and Motivation）介绍技术发展的背景和推动力；原理与关键技术（Principles and Key Technologies）深入讲解核心技术原理；部署配置（Deployment Configuration）说明实际部署方案；性能突破（Performance Breakthroughs）展示优化效果和性能提升；参考资料（References）包括NVIDIA SA团队的工作成果、博客文章和技术洞察。

![](https://files.mdnice.com/user/59/e0307133-3c3f-49d3-93f1-a26c7dc41ccc.jpg)

![](https://files.mdnice.com/user/59/12ab94e5-4fee-48af-8ed7-5fb69d34d133.png)


这里回顾了一下DeepSeek-V3背景。Slides展示了DeepSeek-V3的完整结构：从标准的Transformer Block（包含Feed-Forward Network、RMSNorm和Attention组件）连接到DeepSeekMoE模块，该模块通过Router从多个专家中选择Top-K个进行路由，区分了Routed Expert（绿色实心框）和Shared Expert（绿色虚线框），并详细展示了Multi-Head Latent Attention (MLA)的结构，包括输入隐藏层通过多头 Attention 机制处理q、k、v并应用RoPE生成潜在表示的过程。右侧则提供了详细的模型配置对比表格，显示Kimi K2虽然总参数量达到1.04T（比DeepSeek-V3的671B增加54%）且专家总数达到384个（比DeepSeek-V3的256个增加50%），但在激活参数量（32.6B vs 37B）、 Attention 头数（64 vs 128）和密集层数（1 vs 3）上都有所减少，但更加稀疏了。

![](https://files.mdnice.com/user/59/43042e36-999e-4043-8e25-b270f314525b.png)

这里继续对比了下Qwen3-480B和GPT-OSS-120B的模型的架构设计与配置特点。左边展示了Qwen3-480B模型，具体配置了Qwen3-480B-A35B-Instruct模型：这是一个经过预训练和后训练的因果语言模型，总参数量达到480B，其中激活参数为35B，模型深度为62层， Attention 机制采用GQA（Grouped Query Attention）配置，包含96个Q头和8个KV头，总共部署了160个专家，每个token激活8个专家，原生上下文长度高达262,144个token。

右边详细介绍了GPT-OSS-120B模型，同样采用细粒度专家设计，MoE配置为128个专家，路由器为每个token选择Top-4专家（覆盖90%以上的总参数）， Attention 机制支持GQA（带学习型sink的SWA或全密集 Attention ）。右侧还提供了详细的参数构成对比表格，显示120b版本的MLP参数为114.71B、 Attention 参数为0.96B、嵌入与非嵌入参数为1.16B、激活参数为5.13B、总参数为116.83B、检查点大小为60.8GiB，而20b版本则对应MLP参数19.12B、 Attention 参数0.64B、嵌入与非嵌入参数1.16B、激活参数3.61B、总参数20.91B、检查点大小为12.8GiB，这里清晰展示了两种GPT-OSS模型的架构区别。


![](https://files.mdnice.com/user/59/36a81240-2eb4-4d3b-88c0-9809ac184d85.png)

这张Slides了两种大型语言模型推理服务架构的性能特点和优化策略。左侧的"Co-located（共置）"架构展示了传统推理批处理模式的问题：在IFB（In-Flight Batching）模式下，多个请求（R1-R6）的 Prefill 阶段（实心彩色块）和 Decode 阶段（条纹彩色块）都在同一组GPU上执行，由于较长的前缀块导致"Generation stall"（生成停滞）， Decode 阶段被显著延迟；而通过引入"分块捎带"（Chunked Piggybacking）技术，将 Prefill 分解成更小的块，使 Decode 阶段能够更早开始并与后续请求的 Prefill 重叠，有效缓解了生成停滞问题。右侧的"Disaggregated（解耦）"架构则通过将 Prefill 和 Decode 任务分离到不同GPU上实现根本性优化：Context GPU专门处理 Prefill 任务，能够高效并行处理多个请求的 Prefill 阶段，显著"减少生成第一个token的延迟"；Generation GPU专门处理 Decode 任务，专注于生成后续token，通过将计算密集型的 Prefill 与内存密集型的 Decode 分离，"减少了 Decode 和 Prefill 阶段之间的干扰"，使Generation GPU能够更流畅连续地进行token生成，从而提升整体吞吐量和效率。


![](https://files.mdnice.com/user/59/523feae8-7aaf-462f-acef-323506bc256e.jpg)


![](https://files.mdnice.com/user/59/1f356bb3-0414-47ca-95ec-b02f6f6ac6aa.png)

这张Slides展示了SGLang PD分离的架构流程图。图中展示了五个核心组件（客户端、 Decode 节点、引导服务器、 Prefill 节点、 Transfer Engine ）之间的交互流程：从请求初始化阶段（客户端向 Decode 节点发送请求， Decode 节点通过引导服务器查询 Prefill 节点地址并发送KV传输请求）到 KV Cache 传输阶段（ Prefill 节点准备KV数据并分组连续块，通过 Transfer Engine 执行GPU-GPU直接的RDMA传输，传输内容包含会话ID、目标KV索引和目标地址，标记为"Parallel Transfer"循环）再到验证与执行阶段（ Decode 节点验证所有块是否接收完成并向客户端返回生成的token）以及健康监控阶段（ Decode 节点定期向引导服务器发送心跳请求，标记为"Periodic Heartbeat"循环）。

右边的内容直接记录很抽象难以理解，上面的架构图看得也不是很清楚，要了解SGLang PD分离架构建议阅读下面的材料。



可以参考 https://zhuanlan.zhihu.com/p/1912106909617624371 & https://zhuanlan.zhihu.com/p/1921162497592886258 这两篇讲解 SGLang PD分离架构的知乎文章。

还有这个演讲对应的原Blog：https://mp.weixin.qq.com/s/DJpuqJnTCelMvNerDD2_Og

以及最权威的SGLang PD分离官方设计文档：https://docs.google.com/document/d/1rQXJwKd5b9b1aOzLh98mnyMhBMhlxXA5ATZTHoQrwvc/edit?tab=t.0#heading=h.i3s2t1j0e1ik

![](https://files.mdnice.com/user/59/ae5c305e-e571-447f-a968-72fb1147df25.png)

这张Slides展示了"Prefill-Decode Disaggregation"架构中KV（Key-Value）传输的完整工作流程和技术实现。左侧的序列图展示了四个核心组件（KVManager、KVSender、KVReceiver和TransferBackend）之间的交互过程：在初始化阶段，KVManager首先向KVSender发送"Init TransferEngine"指令，KVSender随后向TransferBackend注册GPU地址以进行RDMA（远程直接内存访问），同时KVManager向KVReceiver发送"Init request_pool mapping"指令并指示其"Start ZMQ server (bind port)"；在KV发送操作阶段，KVSender将[bootstrap_room: metadata]添加到内部的request_pool中，并通过ZMQ消息向KVReceiver发送包含bootstrap_room、kv_data_ptrs和aux_data_ptrs的元数据；在 Prefill 线程阶段，KVSender通过TransferBackend执行transferSync RDMA write操作将KV数据写入，TransferBackend在数据传输完成后向KVSender返回"Transfer complete"信号，随后KVSender向KVReceiver发送"Send completion signal (bootstrap_room + Done)"通知KV数据传输完成；在 Decode 线程阶段，KVReceiver接收到完成信号后从其request_pool中移除对应的bootstrap_room并继续执行 Decode 操作。

右边的内容是：KVSender和KVReceiver负责管理KV数据的发送和接收过程；后台传输在实际的数据传输线程中进行，同时暴露Python接口用于与该线程通信，确保主程序的非阻塞性；系统支持多种KV传输后端（如Mooncake、NIXL等），提供灵活性和可扩展性；所有操作都是非阻塞的，可以轮询传输过程状态，实现高效的异步处理。

![](https://files.mdnice.com/user/59/eaec4716-a029-4a9f-9bfe-da317fd26c24.png)

这张Slides展示了SGLang 大EP部署DeepSeek-V3的并行方案，Slides分为四个主要部分： Attention 机制的并行优化支持混合DP与TP策略，关键优势在于消除跨设备KV Cache重复，通过避免冗余KV Cache存储减少通信开销，图(a)展示了"DP Dense FFN Network 与DP Attention "的架构，其中每个批次（Batch1-4）都独立地通过DP Dense FFN和DP Attention模块；Dense FFN的并行优化支持纯DP或纯TP，指出纯TP会导致碎片化问题（如TP32将18,432维拆分为576单元块与GPU友好的128字节边界不对齐），引入DP旨在减少碎片化提高内存和计算效率，关键优势包括将纯TP中的两次all-reduce操作替换为一次reduce-scatter加一次all-gather（减少50%开销），以及纯 DP Dense MLP 与纯DP Attention 结合可以完全消除设备间通信； 

Sparse FFN Network  的并行优化采用专家并行与DeepEP结合，DeepEP提供多种模式（用于非PD分离服务的自动模式、用于PD Prefill 服务器的Normal模式、用于PD Decode 服务器的低延迟模式），关键优势是更高效的 token 路由，其他优化方向包括通信开销（可通过TBO进一步改进）和工作负载不平衡（可通过EPLB进一步改进），图(b)展示了"EP Sparse FFN Network  与 DP Attention "架构，其中 Attention 层采用数据并行， Sparse FFN Network  通过DeepEP Dispatch进行专家调度，将每个批次的DP Attention输出路由到EP Sparse FFN，再通过DeepEP Combine聚合结果；

语言模型LM_Head的并行优化使用DP而非TP（传统词汇并行），关键优势是更低的内存开销和简化的通信。

![](https://files.mdnice.com/user/59/03e59183-f81e-41a4-850e-4b923f431e96.png)

这张Slides展示了一下DeepEP的workflow，这个workflow是DeepSeek官方提出的结合了DeepEP两种形式的kernel和TBO得到的可以overlap计算和通信的workflow。

![](https://files.mdnice.com/user/59/f0c12710-5358-4105-b524-37460baa52b0.png)

DeepGEMM有3种形式的kernel，一种是Dense用于q,k,v projection这种矩阵乘法，一种是Contiguous模式用在DeepEP Normal Dispatch后面，一种是Masked模式用在DeepEP Low Latency Dispatch后面。

DeepGEMM优化方面主要是用了Warp专门化, Warp调度与执行流程图展示了三种类型的warp，其中TMA warps负责数据加载，通过"TMA Issue"（黄色框）发出指令并执行"Data load"（蓝色条）将数据从全局内存（GMEM）加载到共享内存（SMEM），Math warps 0和Math warps 1负责实际计算，它们交替执行"WGMMA"（Warp Group Matrix Multiply Accumulate，绿色框）进行Tneosr Core计算以及"Promotion"（黄色框）将WGMMA结果在CUDA Core中累积以提高精度；

CUDA Block的Warp Group分配显示每个CUDA block使用3个warp-groups（1个TMA Warp Group专门用于从GMEM加载矩阵A和B的块到SMEM，2个Math Warp Groups用于执行计算，为节省计算资源会跳过填充的数学warp），"Promotion"黄色框明确表示WGMMA结果在CUDA Core中的累积过程；

DeepGEMM的优化策略会根据GEMM操作的整体形状M和N智能选择WGMMA的N维度（输出C tile的block_n）以实现最佳GPU利用率；矩阵块划分示例包括图A展示block_k如何被划分为两个block_m/2=64的子块（每个子块细分为sa1到sa4四个部分）、图B展示block_n如何被垂直划分为sb1到sb4四个子块、图C展示输出矩阵C的block_n维度如何被划分为两个block_m/2=64的区域并分别由WG1和WG2两个Warp Group负责处理

![](https://files.mdnice.com/user/59/a89269ef-9575-4bb0-826f-f2075ebc551c.png)

这张Slides详细介绍了"TBO (Two-batch Overlapping)"，该技术旨在与DeepEP和DeepGEMM协同工作以解决多节点环境中因通信带宽限制导致的延迟问题。Slides通过左右两部分呈现：

**左侧的"Core Design of TBO"图示**展示了两种TBO的调度方式，其中图(a)"Two-batch overlap with an improper launch order"描绘了CPU、计算流（Computation Stream）和通信流（Communication Stream）在不当调度下的情况（CPU发出两个Dispatch指令，计算流中的ATTN和MLP操作与通信流中的Dispatch操作交错进行，但由于调度不当在第二个ATTN操作完成后通信流中的第二个Dispatch操作需要等待，导致明显的"Wasted"时间段表明GPU存在空闲），图(b)"Two-batch overlap with a proper launch order"展示了通过优化调度如何消除图(a)中的"Wasted"时间（CPU同样发出两个Dispatch指令，但计算流中的第二个ATTN操作和通信流中的第二个Dispatch操作被提前启动，与前一个批次的通信和计算操作实现有效重叠，这种"proper launch order"通过将单个批次拆分为两个 Micro Batch 使计算和通信能够并行进行从而显著提高效率

**右侧的文字部分**详细阐述了TBO的优势和具体技术（主要目标是解决多节点环境中有限通信带宽引起的延迟问题，核心策略是将单个批次拆分为两个 Micro Batch 以实现计算和通信重叠，关键效益包括降低峰值内存使用和通过最小化空闲时间提高GPU利用率），使用的GEMM kernel 包括针对普通密集GEMM的"deep_gemm.gemm_fp8_fp8_bf16_nt"、针对MoE Prefill 的"m_grouped_gemm_fp8_fp8_bf16_nt_contiguous"和针对MoE Decode 的"m_grouped_gemm_fp8_fp8_bf16_nt_masked"，使用的注意力 kernel 采用FlashAttention3。

![](https://files.mdnice.com/user/59/f570d6ee-40c6-4fa1-b627-ee991a721daa.png)

这张Slides展示了"TBO (Two-batch Overlapping)"技术在SGLang框架下的代码实现示例，特别针对DeepSeek模型中的 Prefill （Prefill）和 Decode （Decode）操作策略。

左右两部分分别呈现了两种操作的Python代码片段：**左侧是Prefill操作策略**（`_compute_moe_deepseek_blog_prefill`），它定义了一个`OperationsStrategy`，其中`tbo_delta_stages`设置为0，并列出了一系列操作包括注意力（`attn`）、多层感知机（`mlp`）的准备、核心计算、门控、专家选择、分派、组合以及后处理等，其中穿插了两次`operations.YieldOperation()`调用，表明在这些点可以进行批次间的操作重叠；

**右侧是Decode操作策略**（`_compute_moe_deepseek_blog_decode`），同样定义了一个`OperationsStrategy`，但其`tbo_delta_stages`设置为2，并且在操作序列中包含了多达五次`operations.YieldOperation()`调用，这表明 Decode 阶段的重叠策略更为细致和频繁，以适应其内存密集型和延迟敏感的特性，例如在注意力准备、专家选择、共享专家处理以及两次组合操作之后都设置了`YieldOperation`，旨在最大化计算与通信的并行度从而提升整体吞吐量和效率。通过对比可以看出，`tbo_delta_stages`参数和`YieldOperation`的放置是实现TBO在不同阶段优化效果的关键。

针对不同的模型，例如Qwen，它就没有shared experts，要使用TBO就要针对性的去调这些策略。

![](https://files.mdnice.com/user/59/4cf1c928-a13d-4c04-ab2d-52c16a531f96.png)

这张Slides讲了 **EPLB（Expert Parallelism Load Balancer，专家并行负载均衡器）** 的作用及其对性能的影响。

图中分为两个主要部分：**专家分布统计** 展示了四个条形图，分别对应模型中不同层（Layer 39、40、41、42）的专家选择情况，每个层需要处理的总 token 数均为3888056，其中Layer 39和Layer 40的图表显示了明显的负载不均衡（少数几个专家通过高亮的条形如红色框所示被分配了远超其他专家的 token 处理任务，而大多数专家则负载很轻甚至没有负载，表明在没有有效负载均衡的情况下计算资源利用率低下存在瓶颈），Layer 41和Layer 42的图表则展示了更均匀的专家负载分布（条形高度更加一致没有出现明显峰值，意味着 token 被更平均地分配给了所有专家从而提高了并行处理效率和整体资源利用率）；

**EPLB案例研究：吞吐量与均衡性** 是一个折线图，展示了在不同 Decode 步骤下输出吞吐量（Output Throughput per Device，单位：tokens/second，蓝色线）和均衡性（Balancedness，红色线）的变化趋势，图表显示吞吐量和均衡性之间存在高度正相关（在 Decode 步骤20到40左右两者都迅速上升达到峰值，吞吐量接近2850 tokens/second，均衡性接近0.88，随后两者都逐渐下降但趋势保持一致，在 Decode 步骤50附近吞吐量有明显下降而均衡性也随之出现轻微下降，进一步印证了负载均衡对系统吞吐量的直接影响）。

![](https://files.mdnice.com/user/59/0522c56a-38af-4051-a480-9a4a2a34612f.png)


这里是展示了下在模拟的数据分布上开启EPLB的收益，这个只是一个参考数据，我们应该以实际的数据分布上的自测结果为准。

![](https://files.mdnice.com/user/59/9483ea2a-c608-4dc0-bd0b-a27120ba3a45.png)


这张Slides介绍了**MTP（Multiple Token Prediction，多 token 预测）**技术及其在SGLang框架下的集成与优化。

流程图展示了MTP的工作机制，分为** Prefill 阶段（Prefill Stage）**和** Decode 阶段（Decode Stage）**。实际上这里的工作机制在  https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/speculative-decoding/speculative-decoding.md 这里详细解释了。

刚才提到的其它的优化和MTP都是兼容的。

![](https://files.mdnice.com/user/59/72afa521-807c-49e5-ab4d-d0d9d00d1a85.png)

这张Slides展示了 MTP 在提升大型语言模型推理性能方面的显著效果，通过两个柱状图和一个表格呈现了不同配置下的吞吐量数据。**MTP在实际应用中的吞吐量测试（16块H200 GPU）** 显示在DeepSeekV3模型上使用全局批次大小为32时，未启用MTP时每GPU吞吐量为51.0 tokens/s，启用3-token MTP后每GPU吞吐量提升至81.5 tokens/s实现了60%的性能提升，启用4-token MTP后每GPU吞吐量进一步提升至82.0 tokens/s实现了60.8%的性能提升；

**MTP在大规模部署中的吞吐量测试（96块H200 GPU）** 显示在DeepSeekV3模型上未启用MTP时总吞吐量为1391 tokens/s，启用MTP后总吞吐量提升至1588 tokens/s实现了14.2%的性能提升；

![](https://files.mdnice.com/user/59/c945e484-ce27-4dd6-8aa3-948998ae4ba3.jpg)

![](https://files.mdnice.com/user/59/201760f9-0a85-4364-9aff-e87d1f1108ba.png)

这里展示了一下DeepSeek官方，SGLang，以及NVIDIA在H20上对SGLang 大EP+PD分离方案的复现的具体数据。

![](https://files.mdnice.com/user/59/4643f145-703a-4244-adfb-558a285c8e41.jpg)

![](https://files.mdnice.com/user/59/6e2f6dad-8063-49c6-a07a-8604ba495488.png)

这张Slides展示了DeepSeek模型在 Prefill （Prefill）和 Decode （Decode）阶段的整体吞吐量性能，并将其与传统的张量并行（TP16）以及其他专家并行（EP）配置进行了对比，突出显示了相对于传统TP（黄色）的显著吞吐量提升。

 
![](https://files.mdnice.com/user/59/927c62c1-d893-4116-bd06-8bc0af1e2698.png)

这张Slides展示了一下 **Kimi K2模型在Prefill（ Prefill ）和Decode（ Decode ）阶段的整体吞吐量性能** ，并将其与DeepSeek模型进行了对比，同时阐述了实现这些性能的关键技术细节和配置。

Slides顶部标题明确指出"Overall Throughput (Kimi K2)"，并给出了目标性能指标（ Prefill 吞吐量达到56k+输入tokens/秒/节点， Decode 吞吐量达到24k+输出tokens/秒/节点）。下方列出了Kimi K2的三个关键特性和优化点：Kimi K2激活了每token 384个专家的子集并在 Decode 节点上使用了96个冗余专家（表明其采用了MoE架构并针对 Decode 阶段进行了专家冗余优化），模型支持2k的输入序列长度（ISL）和100的输出序列长度（OSL）且 Decode 批次大小为480，为了优化PD分离架构的比例优先考虑 Decode 节点以最大化KV Cache Poll大小（这对于将批次大小扩展到480至关重要）。

底部提供了详细的性能对比表格，比较了DeepSeek和Kimi K2两种模型（DeepSeek模型配置了256个专家使用96块Hopper GPU实现了52.3k tokens/秒/节点的 Prefill 吞吐量和22.3k tokens/秒/节点的 Decode 吞吐量，Kimi K2模型配置了384个专家使用128块Hopper* GPU实现了56k tokens/秒/节点的 Prefill 吞吐量和24k tokens/秒/节点的 Decode 吞吐量），表格下方脚注解释了Kimi K2的128块Hopper* GPU配置采用了1P3D架构（具体为4个 Prefill 节点和12个 Decode 节点）。Slides上面的1P1D好像是typo。

![](https://files.mdnice.com/user/59/8aeebae3-1483-48a3-8d35-349c6edd73cd.jpg)

![](https://files.mdnice.com/user/59/1f5b6d83-69c4-456e-9064-71e32c479ef3.png)

Slides最后展示了一下他们在SGLang上做的工作。





