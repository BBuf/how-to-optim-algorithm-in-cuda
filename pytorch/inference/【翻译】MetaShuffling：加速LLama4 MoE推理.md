> blog链接：https://pytorch.org/blog/metashuffling-accelerating-llama-4-moe-inference/

# MetaShuffling: 加速LLama4 MoE推理

> By Shikai Li, Gefei Zuo, Jianyu Huang, Jason Park, Zoey Sun, Xiaozhu Meng, Xiaodong Wang, Hongtao Yu, Changkyu Kim, CQ Tang, Stephen ChenMay 12, 2025	

Mixture-of-Experts (MoE) 是一种流行的LLM模型架构。虽然它通过每个token激活更少的参数来减少训练和推理的计算量，但在实现最佳计算效率、高内存和通信压力以及处理模型动态和稀疏性方面带来了额外挑战。在这里，我们介绍了一种新的MoE推理解决方案，MetaShuffling，它使我们能够高效地部署Llama 4模型进行生产推理。

![](https://files.mdnice.com/user/59/40a1e6e5-9ef0-4404-ae0c-c65c1019d38c.png)

Llama 4 Scout and Maverick模型正式发布。Scout / Maverick具有共享专家和16 / 128路由专家，使用dropless token选择路由和Top-1选择每个MoE层。此外，共享和路由专家都使用SwiGLU激活，具有3个线性层。有关模型的更多信息，请参阅The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation(https://ai.meta.com/blog/llama-4-multimodal-intelligence/)。

## 关键概念

MoE层中引入了处理动态性和稀疏性问题的多种常见解决方案。在这里，我们展示了使用Top-1选择的不同token选择路由解决方案。


![](https://files.mdnice.com/user/59/2b514ed5-1e0d-4ec6-ac0a-022237b4940a.png)

上图显示了Padding设计。每个框代表一个token，黄色/绿色代表路由到不同专家的有效token，灰色代表Padding的token。第二步中的每一行框代表不同的路由专家。Ti代表来自数据并行组当前rank的第i个token。

- **Padding**：在这种方法中，我们将激活Padding到每个专家的最大序列长度，并运行单个批量矩阵乘法（BMM）。这会导致：
    - 保存Padding数据增加内存占用。
    - 处理Padding数据增加延迟。注意，可以通过锯齿状kernel来避免处理Padding，但当专家数量很大时，锯齿状kernel也可能产生很高的开销。

![](https://files.mdnice.com/user/59/264d18c7-0012-4dc4-8436-9a08758b14d7.png)

- **Slicing**: 在这种方法中，我们将激活切片到每个专家的精确序列长度，并运行多个矩阵乘法（MM）。它避免了Padding的问题，但会导致：
    - 由于在小形状上重复kernel启动，导致kernel效率降低。
    - 由于频繁的主机和设备同步，以及动态形状的额外kernel启动开销，导致设备利用率降低，因为它与图捕获机制（如CUDAGraph和torch.compile）不兼容。

![](https://files.mdnice.com/user/59/5cd3e786-7763-492c-8d94-02cfa9a34ce9.png)

- **Concatenation**: 在这种方法中，我们在切片后进一步连接激活，并运行单个分组矩阵乘法（GMM）。它避免了切片中的kernel效率问题，但仍然会导致：
    - 由于仍然需要主机和设备同步，导致设备利用率降低，因为它与图捕获机制（如CUDAGraph和torch.compile）不兼容。

为了进一步改进解决方案，我们提出了一种基于shuffle的机制：

![](https://files.mdnice.com/user/59/3991b788-2ddd-47a5-abf2-6d7db3f7caf7.png)

- **Shuffling**: 在这种方法中，我们直接对token进行排序，使得路由token按路由专家的ID排序。这样做，没有引入Padding或分割，并且分配给相同专家的token被存储在一起，可以在GroupedGEMM内部一起处理。它提供了一个密集模型接口，避免了上述所有问题。
    - 没有Padding，因为激活保持为密集张量。
    - 没有主机和设备同步，因为激活保持为静态形状的张量。

我们基于这个设计构建了一个端到端的MoE推理解决方案，MetaShuffling。

## 运行时设计

### 单GPU推理无需并行化

![](https://files.mdnice.com/user/59/64d8f5fd-cc81-490e-ad57-104b18e7f77d.png)


上图是单GPU推理无需模型并行化的整体运行时设计。注意,为了优化性能,SwiGLU激活的第一层和第三层线性层被合并为GroupedGEMM13 / GEMM13。

- 实心深蓝色/橙色框代表在路由/共享专家流上的Tensor Core密集型kernel。
- 实心浅蓝色/橙色框代表在路由/共享专家流上的CUDA Core或内存流量密集型kernel。
- 红色箭头代表激活张量的数据流。
- 绿色箭头代表元数据张量的数据流。

所有元数据张量都放置在设备上。没有阻塞的设备到主机同步。所有kernel都是连续启动的,没有气泡。该图仅显示数据流,而不是实际性能分析Trace的演示。
### Kernel接口和数据流

- **RoutingScores**: 一个用于处理路由分数计算的函数或融合kernel。
    - 输入: input_tokens: [T, D] (T: token数量; D: 特征维度); router_weights: [D, E] (E: 专家数量); router_biases: [E];
    - 输出: routing_scores: [T, E]; scaling_factors: [T, E];

- **IndexShuffling**: 一个用于处理索引shuffle和排序的融合kernel。我们将在Kernel设计部分介绍一个优化的实现。
    - 输入: routing_scores: [T, E]; K (top-k路由的阈值);
    - 输出: routed_token_indices: [K * T]; routed_expert_indices: [K * T]; routed_token_counts_per_expert: [E];

- **GatherMul**: 一个用于根据排序的索引shuffle token并缩放它们的融合kernel。
    - 输入: input_tokens: [T, D]; routed_token_indices: [K * T]; routed_expert_indices: [K * T]; scaling_factors: [T, E];
    - 输出: scaled_routed_tokens: [K * T, D]

- **GroupedGEMM**: 一个优化的GroupedGEMM kernel，可以处理M维度上批次的设备内形状信息，没有限制。我们将在Kernel设计部分介绍一个优化的实现。
    - 输入: tokens: [K * T, D]; weights: [E, D, HD] (HD: 隐藏维度); routed_token_counts_per_expert: [E];
    - 输出: tokens: [K * T, HD]

- **GEMM**: 一个优化的GEMM kernel。与密集模型接口类似。

- **NonLinearity**: 一个处理非线性的融合kernel。与密集模型接口类似。

- **ScatterAdd**: 一个优化的kernel，基于排序的索引反转token shuffling，并直接执行scatter add到共享专家输出，无需具体化未shuffle的张量。
    - 输入: shared_output_tokens: [T, D]; routed_output_tokens: [K * T, D]; routed_token_indices: [K * T];
    - 输出: combined_output_tokens: [T, D]

注意：如果应用量化，则激活量化kernel会被融合到前面的非GEMM kernel中，这意味着对于GroupedGEMM13融合到GatherMul中，对于GroupedGEMM2融合到NonLinearity中，等等。

注意：如果使用较大的K * T，GatherMul和ScatterAdd操作可以进一步融合到后续/前置的GroupedGEMM操作中，这应该作为前序/后序中全局内存到共享内存/寄存器或共享内存到全局内存的步骤来完成。然而，这在kernel设计层面上增加了与tensor core执行重叠的额外挑战。此外，融合ScatterAdd需要共享专家在路由专家之前完成，如果这些kernel可以用来隐藏AlltoAll延迟，这可能不是一个好的设计选择。

## 单主机推理的张量并行化

![](https://files.mdnice.com/user/59/78a2e46f-9a76-4e47-a08d-8578626079a6.png)

上图是单主机推理使用张量并行(TP)的整体运行时设计。与单GPU推理相比,额外增加的步骤是:

- 实心浅薄荷色框代表网络通信密集型的通信kernel。

所有元数据张量仍然放置在设备上,没有设备到主机的同步。所有kernel都是连续启动的,没有气泡。该图仅显示数据流,而不是实际性能分析Trace的演示。

### 工作负载分片和额外的Kernels

与单GPU推理用例相比,没有引入额外的自定义kernel。对于GEMM、GroupedGEMM和非线性kernel,激活和权重都沿不同维度共享到1/TP,计算/内存开销也共享到1/TP。

如果仅应用张量并行,最后一步应该是AllReduce。或者,如果张量并行与序列并行一起应用,则使用ReduceScatter。

## 多卡推理的专家并行化

为了启用专家并行化(EP),我们将数据并行维度从路由专家中交换出来,作为路由专家内部的专家并行维度。注意,为了获得更好的GEMM效率,专家并行可以进一步与张量并行交换,但这会增加路由不平衡的风险,我们不会在本博客中介绍这种设计。

如果在token-choice路由中启用了专家并行,由于路由到不同专家组的token数量是动态的,我们必须在使用密集张量或使用静态形状之间做出选择。

- 当优先使用eager模式时,我们使用密集张量和动态形状,以避免运行未Padding的AlltoAll造成的网络流量和内存空间浪费。
- 当优先使用图模式时,我们使用稀疏张量和静态形状,以避免通过运行CUDAGraph导致的CPU启动开销和设备到主机同步产生的GPU气泡。

注意,使用Padding激活的网络流量浪费也可以通过使用自定义AlltoAll实现来避免,但我们不会在本博客中介绍任何关于自定义通信或通信和计算融合kernel的主题。

![](https://files.mdnice.com/user/59/8887046d-adf9-47e8-ad3e-45c961fddcc0.png)
上图是使用张量并行和专家并行的多主机推理的整体运行时设计。与使用张量并行的单主机推理相比:

- 实心红色箭头表示节点内通信。
- 实心紫色箭头表示节点间通信。

### Kernel接口和数据流

对于增加的基于专家并行的通信,我们使用3次All2All通信来交换形状和token:

- 第1次A2A:交换设备上关于路由到每个专家的token数量的元数据张量,即`routed_token_counts_per_expert: [E]`,这是由IndexShuffling kernel生成的输出。
- 第2次A2A:将token从基于数据并行转换为基于专家并行,根据路由分发到不同的EP ranks。
- 第3次A2A:将token从基于专家并行转换为基于数据并行,根据路由从不同的EP ranks组合。

此外,我们增加了2个额外的shuffling kernel和1个特殊的scatter kernel:

- **CombineShuffling(密集或Padding)**: 将接收到的token从按rank排序重新排列为按expert排序。后面的T*表示从所有对等节点接收的总token数,可以根据routed_token_counts_per_rank_per_expert张量的形状信息进一步解释为不规则维度。
    - 输入:received_tokens: [T*, D](首先按dp ranks排序,然后按专家索引排序); routed_token_counts_per_rank_per_expert: [EP, E // EP];
    - 输出:reshuffled_tokens: [T*, D](首先按专家索引排序,然后按dp ranks排序); routed_token_counts_per_expert: [E // EP];
- **SplitShuffling(密集或Padding)**:CombineShuffling的反向过程。将待发送token从专家优先顺序重新排序为rank优先顺序。
    - 输入:reshuffuled_tokens: [T*, D](首先按专家索引排序,然后按dp ranks排序); routed_token_counts_per_rank_per_expert: [EP, E // EP];
    - 输出:to_send_tokens: [T*, D](首先按dp ranks排序,然后按专家索引排序);
- **ScatterAdd(Padding)**:从Padding张量中scatter adds有效token。
    - 输入:共享输出token: [T, D]; 接收到的Padding路由输出token: [EP, K*T, D]; 路由token索引: [K * T]; 每个专家的路由token数量: [E];
    - 输出:组合输出token: [T, D]

我们将在"图模式下使用静态形状的Padding通信"部分详细介绍上述kernel。

### Eager模式下使用动态形状的非Padding通信

![](https://files.mdnice.com/user/59/66783d0e-0b63-4ac4-a404-7d6b881c050e.png)

![](https://files.mdnice.com/user/59/b0b90a94-438d-47a4-958a-56269db39d1d.png)


运行时行为的高层示意图。不同组件的实际运行时间可能会根据软件和硬件的不同而变化。

#### 最小化动态形状的使用

由于路由是每个MoE层动态的,所需的最小设备/主机同步次数为每层一次。为了实现这一点,我们延迟了`send_sizes`的D2H复制,并将其与`recv_sizes`连接起来,通过单个D2H复制一起传输。这减少了设备/主机同步次数为每层一次。

#### 最小化动态形状的负面影响

为了进一步隐藏设备/主机同步开销,我们进一步将共享专家分为两部分。

- 我们首先分发第一部分,在路由之后,但在分发A2A之前。然后,当设备/主机同步发生时,设备仍然保持忙碌运行共享专家。
- 我们第二部分在MoE之后,但在组合A2A之前分发。这将进一步帮助重叠第二个A2A。

### 图模式下使用静态形状的Padding通信

![](https://files.mdnice.com/user/59/8602148f-773b-4132-ab90-92ec9e374b56.png)

#### 最小化Padding的使用

在无丢弃token选择设计中,路由到任何单个专家的最大可能token数量是T。然而,如果我们通过专家并行分片将多个专家组合在一起并放置在单个GPU上,对于TopK路由:

- 路由到1个专家的最大token数量是T。
- 路由到2个专家的最大token数量是2 * T。
- ...
- 路由到K个专家的最大token数量是K * T。
- 路由到K+1个专家的最大token数量仍然是K * T。
- ...

因此,路由到N个专家组的最大token数量将被限制在min(N, K) * T个token。

对于Top1路由,路由到任意大小的专家组的token数量将始终被限制在T个token,由于有EP个专家组,分配和保存动态token所需的最小内存是EP * T个token。

为了实现最小所需的Padding,我们直接使用AllGather来从不同的EP ranks收集所有活跃token,然后通过自定义kernel在本地拆分和重新排列路由token。激活大小被压缩到1 / (E // EP),这对应于内存和网络流量的减少。

![](https://files.mdnice.com/user/59/6bc9e7aa-5ae6-4b09-a728-0f38a04e560b.png)

上图展示了Padding设计。每个方框代表一个token,蓝色/绿色表示具有专家分配的有效token,灰色表示Paddingtoken。RiTj表示专家并行组中第i个rank的第j个token。

#### 最小化Padding的负面影响

尽管Padding被减少到最小允许,我们还通过承担设备形状信息`routed_token_counts_per_expert`或`routed_token_counts_per_rank_per_expert`确保Padding只导致内存空间(分配)和网络流量(通信),而不是导致冗余计算(GroupedGEMM / NonLinear),冗余内存带宽(CombineShuffling / SplitShuffling / ScatterAdd)。


![](https://files.mdnice.com/user/59/b3f84716-9c3c-4009-bace-7e44e2cd66ca.png)

**激活的概念解释**

- 最重要的是,当所有EP ranks上的活跃token总数较小时,这样做很重要,以避免在GroupedGEMM中激活冗余专家并导致额外内存流量。
- 当所有EP ranks上的活跃token总数较大时,这样做也很重要,以避免将GroupedGEMM从memory bound转换为compute bound。 

**CombineShuffling**: 当前EP rank分配的token被重新排列为从专家优先顺序到rank优先顺序,在AllGather之后。未分配的token不会被复制,并且张量末尾剩余的分配内存空间保持不变。

![](https://files.mdnice.com/user/59/1e1f62a3-2566-4b30-9ead-df4080e989ee.png)

**SplitShuffling**: 当前EP rank分配的token被重新排列为从rank优先顺序到专家优先顺序,在AlltoAll之前。未分配的token不会被复制,并且重新排列的张量具有交错存储的Padding。

![](https://files.mdnice.com/user/59/a5c6b1d0-d670-4596-a37e-5904977f96e2.png)


**ScatterAdd (Padded)**: 每个EP rank最终接收来自所有其他rank计算的激活,它将理解哪些是有效token,哪些是Padding token,然后只读取有效token进行scatter_add。

![](https://files.mdnice.com/user/59/6f9cc50e-361c-47b6-815f-dca633b99b3a.png)


#### 通信去重

不同张量并行rank在第一个GroupedGEMM之前和第二个GroupedGEMM之后具有相同的激活,因此相同的token在节点之间重复交换。 

![](https://files.mdnice.com/user/59/7a53bac0-e78c-44ba-9dd0-4bb05c40469a.png)

我们启用了通信去重,以均匀分布节点间通信工作负载到不同的rank,同时引入额外的节点内通信。DP2/TP8/EP2的示例:

- 在eager模式下第一个AlltoAll,将$T*D$节点间AlltoAll拆分为$T*D/8$节点间AlltoAll和$T*D$节点内AllGather。

![](https://files.mdnice.com/user/59/5630ae8c-c5e4-4ecc-a299-a7603450da0f.png)

- 在eager/图模式下,第二个AlltoAll,将$T*D$节点间AlltoAll拆分为$T*D/8$节点内ReduceScatter和$T*D/8$节点间AlltoAll。

![](https://files.mdnice.com/user/59/74eeb169-d38d-4b7c-a6be-aa1be05a0818.png)

- 在图模式下,第一个AllGather,将$2*T*D$节点间AlltoAll拆分为$2*T*D/8$节点间AllGather和$2*T*D$节点内AllGather。

![](https://files.mdnice.com/user/59/f2ff9397-4e17-4718-b3e7-d77f0d4c2a32.png)


## Kernel Design


我们实现了超过10个自定义kernel来支持**MetaShuffling** MoE推理设计,在Nvidia H100 GPU和AMD MI300X GPU上运行。我们开源了所有计算kernel作为PyTorch operators在FBGEMM Generative AI Kernel Library(https://github.com/pytorch/FBGEMM/tree/main/fbgemm_gpu/experimental/gen_ai)。我们希望它可以帮助用户**高效地**在他们的首选框架和首选加速器上服务Llama 4模型,例如vLLM / SGLang。在本博客中,我们将重点介绍2个最有趣的kernel设计,作为提高推理性能的关键,GroupedGEMM和IndexShuffling。

### GroupedGEMM

我们实现了基于Triton的GroupedGEMM kernel,用于BF16 / FP16 / FP8 Rowwise。

#### 接口

```python
def grouped_gemm_fp8_rowwise(
	x: torch.Tensor, 		# shape: [M, K]
	w: torch.Tensor, 		# shape: [G*N, K]
	m_sizes: torch.Tensor, 	# shape: [G]
	x_scales: torch.Tensor,	# shape: [M]
	w_scales: torch.Tensor, 	# shape: [G*N]
) -> torch.Tensor:               # shape: [M, N]
	...
```

该接口与单个GEMM非常相似,它接收一个左矩阵、一个右矩阵作为输入,并产生一个输出。从运行时的角度来看,没有动态性或稀疏性。

然而,该kernel使用`m_sizes`的数据动态地分割左矩阵的M维度,并使用`m_sizes`的形状静态地分割右矩阵的N维度。这种设计有几个优点:

- 不同批次的M之间不需要额外的填充或对齐要求。因此只要总和不超过`M`,`m_sizes`可以存储任何非负值。
- `m_sizes`可以为零值以跳过未激活专家的权重加载。
- `m_sizes`的总和可以小于`M`,以便在不产生额外开销的情况下跳过末尾填充token的计算。
- `m_sizes`或左矩阵激活的分割对设备是已知的,但对主机是未知的。因此它支持动态路由信息而不会导致设备到主机的同步。

#### 工作负载分区

我们采用持久kernel设计,每个SM启动1个CTA,并让所有CTA以交错方式运行所有分割的tile。概念上,工作负载分区如下所示。

![](https://files.mdnice.com/user/59/800f86d1-a10a-40cd-8c42-be8f4c4822a7.png)

```python
def partition_workload(G: int, Ms: List[int], N: int):
	partitions = []
	for g in range(G):
		for n in range(0, N, BLOCK_N):
			for m in range(0, Ms[g], BLOCK_M):
				partitions.append((g, m, n))
	paritions_per_cta = [[] for _ in NUM_SMS]
	for i, part in enumerate(partitions):
		paritions_per_cta[i % NUM_SMS].append(part)

```

工作负载在设备侧动态计算,开销很小。然而,通过这样做,我们可以实现:

- 不同SM之间的工作负载平衡。
- 每个SM只启动1个CTA的小启动开销。
- 高L2缓存命中率。工作负载分区的顺序确保权重/激活最可能从HBM加载一次并缓存在L2中。因为相同权重/激活tile的使用几乎总是从不同SM并发/连续发生。

#### 持久kernel与warp特化

![](https://files.mdnice.com/user/59/29fe257c-b713-4e3b-99d6-16c8533334b6.png)


我们采用了主机侧tensor map-based的激活和权重加载,以及可选的设备侧tensor map-based的输出存储,以减少Hopper GPU上的内存传输开销。通过激活的连续存储格式,我们可以使用单个主机侧TMA (Tensor Memory Accelerator)描述符来加载激活并掩码属于其他专家的token。然而,我们需要创建多个设备侧TMA描述符来存储输出,而不支持动态掩码。

我们采用了基于warp特化的kernel设计,使kernel以真正的持久方式运行,每个SM在3个warp组之间切换(1个生产者和2个消费者)。这种设计保持了TMA引擎、Tensor core和CUDA core执行的交错,利用了异步TMA指令和WGMMA (Asynchronous Warpgroup Level Matrix Multiply-Accumulate)指令,以及共享内存上的内存屏障。我们收到了Meta的Triton编译器团队的大量帮助来实现它。只有通过warp特化,才能隐藏prologue和epilogue,因为传统的软件流水线方法无法处理带有指针追踪的复杂控制流。

### IndexShuffling

我们实现了基于CUDA / HIP的index shuffling kernel。

#### 接口

```python
def index_shuffling(
	scores: torch.Tensor,			        # shape: [T, E]
):
	token_counts: torch.Tensor = ...		# shape: [E]
	expert_indices: torch.Tensor = ...	        # shape: [T]
	token_indices: torch.Tensor = ...		# shape: [T]
	return token_counts, expert_indices, token_indices
```

该kernel接收所有专家上所有token的路由分数,确定每个token被路由到哪个专家,重新排列token索引,使得所有被路由到同一个专家的token连续放置,并返回:

- `token_counts`: 作为每个专家被路由到的token数量。它将被馈送到上面讨论的GroupedGEMM kernel。
- `expert_indices`: 作为每个shuffled token所属的专家索引。它将被馈送到上面讨论的GatherMul kernel。
- `token_indices`: 作为每个shuffled token所属的原始token索引。它将被馈送到上面讨论的GatherMul和ScatterAdd kernel。


#### Cooperative Kernel

我们采用了协作式kernel设计,并将kernel分为两个主要阶段:top-k归约阶段和桶排序阶段,中间有一个全局同步。

![](https://files.mdnice.com/user/59/275d702f-ffc8-4ee1-9892-32fd080b8511.png)


1. 加载分数:
    - 将路由分数的一个tile从全局内存(HBM)加载到共享内存(SMEM)
    - 同时将相关的专家索引也存储在SMEM中

2. 归约:
    - 在E维度上对SMEM执行TopK归约
    - 对于Llama 4用例,它执行ArgMax排序作为Top1归约,包括对SMEM上的分数和相关专家索引进行2D并行树归约
    - 在不同的树归约阶段:
        - 所有线程将在SMEM上并发处理多个token的归约
        - 每个线程将在SMEM上顺序处理多个token的归约

3. 计数和存储缓冲区:
    - 遍历tile上的所有token
    - 从SMEM获取选定的专家索引,将其存储到HBM上的缓冲区(`buf_expert_index`)
    - 对HBM上的输出计数器(`token_counts`)执行`atomicAdd`操作
    - 有趣的是,`atomicAdd`操作将返回内存位置上的先前值,这表示token在组内的位置
    - 我们将此值存储在缓冲区(`buf_local_token_index`)中,并使用它来确定所有token之间的全局顺序
    - 重复步骤1-3,直到处理完分配给CTA的所有token

4. 全局同步:
    - 对HBM上的全局计数器执行`atomicAdd`操作
    - 之后,所有CTA将等待,直到全局计数器达到总token数
    - 使用`st.release` + `ld.aquire`屏障来保护前面的存储操作和后面的加载操作,以确保正确性

5. 扫描:
    - 执行简单的加载和`token_counts`的前缀和
    - 将其转换为SMEM上的`token_counts_cumsums`

6. 加载缓冲区和存储输出:
    - 遍历分配给此CTA的所有token
    - 对于每个token:
        - 从`buf_expert_index`加载token被分配到的专家索引
        - 然后计算shuffling后的新token索引,作为以下两项的和:
            - 属于前面专家的token数量(使用SMEM张量`token_counts_cumsums`)
            - 属于同一专家的之前token数量(使用HBM张量`buf_local_token_index`)
    - 最后,在shuffling后的新token索引位置直接存储到`expert_indices`和`token_indices`输出


## 性能

### 示例内核性能

我们的测试环境使用了H100 80GB SMX5 HBM3 700W SKU、Python 3.12和CUDA 12.8。单个H100的理论峰值HBM内存带宽为3.35 TB/s。

### 分组GEMM

#### Prefill性能

下表展示了该kernel在Llama 4 Scout和Maverick单主机服务上的Prefill性能。实验设置假定总token数为16,384,并使用张量并行分片。

![](https://files.mdnice.com/user/59/7a28bd5a-04a0-48fc-9de9-56af939345c7.png)

注意: G表示组数。M表示每组的token数。N表示每组的输出特征维度。K表示每组的输入特征维度。FP8表示FP8行缩放(激活的每token缩放和权重的每通道缩放)快速累加。量化kernel未包含在基准测试中。缩放未包含在内存带宽计算中。使用rotating buffers和CUDAGraphs进行基准测试。

#### 解码性能

下表展示了该kernel在Llama 4 Scout和Maverick单主机服务上的解码性能。实验设置假定总token数为128,并使用张量并行分片。


![](https://files.mdnice.com/user/59/7d2be696-37e0-43f4-b5f8-6a59cb3737c1.png)


### IndexShuffling

下表展示了该kernel在Llama 4 Scout和Maverick单主机服务上的性能,与原生PyTorch实现进行比较。

![](https://files.mdnice.com/user/59/e3b87b4a-b3d7-4d5b-bdb2-f7cf4dbf978e.png)

使用rotating buffers和CUDAGraphs进行基准测试。

## 示例Trace分析

### Llama 4 Scout BF16 解码

这是使用我们的MetaShuffling MoE推理解决方案对64个token进行Llama 4 Scout BF16解码的示例Trace。

![](https://files.mdnice.com/user/59/13d15897-3a32-455b-a1dc-48a3cdcbe0c0.png)


- MoE的总内存流量(忽略激活值):
    - 路由器: 5120x16x2 = 163,840 字节
    - 共享专家: (2048×5120 + 5120×1024)x2=31,457,280 字节
    - 路由专家: 16x(2048×5120 + 5120×1024)x2=503,316,480 字节
    - 总计: 163,840 + 31,457,280 + 503,316,480=534,937,600 字节
    - MoE的总执行时间为197.456微秒,实现的内存带宽为534,937,600 / (197.456 * 10^-6)=2,709,148,367,231 字节/秒 ~= 2.71 TB/秒,这达到了H100 80GB SMX5 HBM3理论峰值HBM内存带宽3.35 TB/秒的80.90%。

以下是Trace分析中不同组件的细分。

![](https://files.mdnice.com/user/59/cc96d958-066c-4c95-97ef-3d7d1d6c92db.png)


首先,Router和Shared Experts的细分。这两个组件在2个不同的流上并发运行,以实现更好的资源利用。

对于Router流(标记为红色框):
    1. Router GEMM: 基于CuBLAS的GEMM,采用split-k设计。启动2个kernel,第二个kernel用于规约计算。
    2. Sigmoid(Router Activation): PyTorch原生sigmoid。
    3. IndexShuffling: 基于FBGEMM的索引重排,采用协作式kernel设计。可以看作是topk、bincount和sort这3个操作的融合。启动2个kernel,第一个kernel用于设置。
    4. GatherMul: 基于FBGEMM的gather缩放。可以看作是gather(tokens)、gather(scores)和mul这3个操作的融合。

对于共享专家流(标记为橙色框):

    5. 共享专家GEMM13: 基于CuBLAS的GEMM,采用split-k设计。启动2个kernel,第二个kernel用于规约计算。
    6. SwiGLU: 融合的SwiGLU。可以看作是sigmoid和mul这2个操作的融合。
    7. 共享专家GEMM2: 基于CuBLAS的GEMM。

![](https://files.mdnice.com/user/59/b36156fa-e133-466d-8484-e7db13c12e2b.png)



其次是路由专家的细分。该组件专门在1个流上运行,以让GroupedGEMM kernel完全占用所有SM。

对于路由专家流(标记为红色框):

    8. 路由专家GroupedGEMM13: 基于FBGEMM的GroupedGEMM,采用持久化kernel设计。
    9. SwiGLU: 融合的SwiGLU。如6中所述。
    10. 路由专家GroupedGEMM2: 基于FBGEMM的GroupedGEMM,采用持久化kernel设计,在epilogue中融合了scatter add。

解码步骤使用CUDAGraph在具有静态形状的密集张量上运行。

### Llama 4 Maverick FP8 Prefill

这是使用我们的**MetaShuffling** MoE推理解决方案的Llama 4 Maverick FP8预填充的5000个token的示例Trace。注意路由专家的FP8行缩放,以及Router和共享专家的BF16数据类型。

与解码Trace相比:

- 它使用单个流来避免路由器和共享专家之间的kernel交互。由于kernel处理的问题规模足够大,可以饱和计算资源,额外的重叠只会导致资源竞争,尤其是在L2缓存上。
- 它在具有静态形状的密集张量上以eager模式运行。由于kernel执行时间足够长且没有设备/主机同步,kernel可以连续启动而不会产生气泡。

以下我们重点介绍这两个Trace之间的kernel差异(不包括执行时间):

- Router GEMM和SharedExpertGEMM13: 基于CuBLAS的GEMM,不使用split-k设计。因此它只启动1个kernel而不是2个。

![](https://files.mdnice.com/user/59/896eebda-6cfa-4eff-acfd-5405167779a9.png)


- 4 GatherMul (FP8按行量化): 基于FBGEMM的gather缩放和量化。可以看作是8个操作的融合:gather(tokens)、gather(scores)、mul、max、divide、mul、clamp和类型转换。
- 9 SwiGLU (FP8按行量化): 融合的SwiGLU和量化。可以看作是7个操作的融合:sigmoid和mul、max、divide、mul、clamp和类型转换。

![](https://files.mdnice.com/user/59/6d1f216e-9c4b-4afb-8ac8-f24eb42a82ea.png)

### Takeaway

我们逐步采取以下步骤来优化MoE解决方案的推理性能:

- 通过避免主机和设备同步来提高设备级利用率。
- 通过删除填充或避免处理填充来减少浪费的资源。
- 通过激进的kernel融合来减少kernel启动和I/O开销。
- 通过各种kernel优化来提高计算和内存效率,推动性能接近硬件极限。
- 通过并发执行计算、内存流量或网络流量密集的kernel来提高硬件组件级利用率,但同时避免不希望的资源竞争。


## 单主机服务

我们使用1000个随机提示基准测试了Llama 4 Maverick和Llama 4 Scout的单主机服务性能,使用我们的内部**MetaShuffling** MoE推理堆栈。我们使用FP8运行Maverick,使用BF16运行Scout,在一个8xH100主机上,最大批量大小为64。我们的设置使用了H100 80GB SMX5 HBM3 700W SKUs、Python 3.12和CUDA 12.8。我们开源了所有计算kernel(https://github.com/pytorch/FBGEMM/tree/main/fbgemm_gpu/experimental/gen_ai)和**MetaShuffling** MoE推理堆栈的示例实现(https://github.com/pytorch/FBGEMM/blob/def50a6219d645c809d744f04d4ec2cbe9784620/fbgemm_gpu/experimental/gen_ai/gen_ai/moe/layers.py#L205)。

![](https://files.mdnice.com/user/59/d09eb1c8-31cb-44af-80f9-a8857ac6f1a9.png)

为了保持最佳精度,我们在路由专家上使用FP8精度基准测试Llama 4 Maverick,在注意力线性层、注意力、共享专家、路由器和KV缓存上使用BF16精度。

![](https://files.mdnice.com/user/59/9f43753e-973f-490e-907e-80d17b6a017f.png)


我们使用BF16精度基准测试了Llama 4 Scout的所有线性层(注意力线性层、共享专家、路由器和路由专家)、注意力和KV缓存。

最后,我们希望社区能够持续打破记录,提高服务Llama 4模型的效率,并期待更好的数字被报告。


## 致谢

我们感谢Jing Zhang、Ying Zhang和Manman Ren提供的技术审查和指导。

我们还要感谢Bradley Davis、Yan Cui、Rengan Xu、Josh Fromm、Jiawen Liu、Sarunya Pumma、Jie Wang、Xinfeng Xie、Benson Ma、Michael Shu、Bingzhe Liu、Jingyi Yang、Min Si、Pavan Balaji、Dhruva Kaushal对本项目的贡献。




紧接着昨天那篇PyTorch Blog的内容[MetaShuffling：Meta的Fused MoE kernel工程方案，更激进的Kernel优化和尽量避免Padding](https://mp.weixin.qq.com/s/MdztXkwIzw0ERTOVoCUz3g)，我把fbgemm开源的moe grouped gemm kernel(https://github.com/pytorch/FBGEMM/tree/main/fbgemm_gpu/experimental/gen_ai)拷贝了一下，在H100(Hopper)和SGLang的Grouped GEMM Triton Kernel对比了一下正确性和性能，在正确性没问题的情况下，性能可以提升挺多的。细节放在这里了：https://github.com/sgl-project/sglang/pull/6924 。结论就是fbgemm可以在MoE模型上相比于SGLang的grouped gemm实现较大的性能提升，这个kernel可以在fp16/bf16和fp8 per-tensor quant的情况下直接应用到sglang的ep-moe的grouped gemm kernel中提升性能。不过TP模式下的Triton Fused MoE不是最直接的Grouped GEMM，需要有小伙伴按照这个Triton优化的技巧去改kernel才可以。

## FBGEMM GroupedGEMM 基准测试结果

当使用 triton==3.2.0 运行基准测试时,会出现以下警告:我们无法使用 warp专用化, 但持久化kernel和 TMA load/store 仍然可用。

```shell
/home/ubuntu/bbuf/sglang/benchmark/kernels/fbgemm/fbgemm_grouped_gemm.py:1104: UserWarning: Warp specialization is disabled as the Triton build in current environment doesn't have such support. Please build from https://github.com/facebookexperimental/triton/tree/ws-3.2.x to enable it for best performance on Nvidia's SM90 GPUs.
```


### Qwen2-57B-A14B-Instruct BF16 W8A8 TP4

```shell
python3 benchmark/kernels/fbgemm/benchmark_fbgemm_grouped_gemm.py --model Qwen/Qwen2-57B-A14B-Instruct --tp-size 4

grouped-gemm-performance:
    batch_size  FBGEMM Grouped GEMM BF16  SGLang Grouped GEMM BF16
0          1.0                  0.032352                  0.022272
1          2.0                  0.032096                  0.022080
2          4.0                  0.032640                  0.021984
3          8.0                  0.031840                  0.021472
4         16.0                  0.030832                  0.021536
5         32.0                  0.032192                  0.021632
6         64.0                  0.393504                  0.595008
7        128.0                  0.393872                  0.598048
8        256.0                  0.394848                  0.589760
9        512.0                  0.397488                  0.605888
10      1024.0                  0.401248                  0.581952
11      2048.0                  0.407232                  0.559232
12      4096.0                  0.416368                  0.717936
```


![](https://files.mdnice.com/user/59/b47939d5-516e-47e7-884f-7d16a661d0f7.png)



### Qwen2-57B-A14B-Instruct FP8 W8A8 TP4

```
python3 benchmark/kernels/fbgemm/benchmark_fbgemm_grouped_gemm.py --model Qwen/Qwen2-57B-A14B-Instruct --tp-size 4 --use-fp8-w8a8 

    batch_size  FBGEMM Grouped GEMM FP8  SGLang Grouped GEMM FP8
0          1.0                 0.042560                 0.022336
1          2.0                 0.041312                 0.022128
2          4.0                 0.040384                 0.022240
3          8.0                 0.041184                 0.022016
4         16.0                 0.040128                 0.022816
5         32.0                 0.014272                 0.021440
6         64.0                 0.212832                 0.595040
7        128.0                 0.211328                 0.598688
8        256.0                 0.211776                 0.590992
9        512.0                 0.213504                 0.606304
10      1024.0                 0.216864                 0.582624
11      2048.0                 0.220512                 0.558128
12      4096.0                 0.227296                 0.718848
```


![](https://files.mdnice.com/user/59/2e795cf6-ed2f-4ae5-958d-beb403739891.png)



### meta-llama/Llama-4-Scout-17B-16E-Instruct FP16 TP8

```shell
python3 benchmark/kernels/fbgemm/benchmark_fbgemm_grouped_gemm.py --model meta-llama/Llama-4-Scout-17B-16E-Instruct --tp-size 8 

grouped-gemm-performance:
    batch_size  FBGEMM Grouped GEMM BF16  SGLang Grouped GEMM BF16
0          1.0                  0.034592                  0.022816
1          2.0                  0.033440                  0.022016
2          4.0                  0.033984                  0.022400
3          8.0                  0.324592                  0.532960
4         16.0                  0.321024                  0.516960
5         32.0                  0.322736                  0.695840
6         64.0                  0.321184                  0.607008
7        128.0                  0.321264                  0.475136
8        256.0                  0.321984                  0.419232
9        512.0                  0.325728                  0.363392
10      1024.0                  0.339616                  0.693824
11      2048.0                  0.396928                  1.383792
12      4096.0                  0.732640                  2.761792
```


![](https://files.mdnice.com/user/59/623d8f6e-2769-4b44-984a-0b95eb863d64.png)


### meta-llama/Llama-4-Scout-17B-16E-Instruct FP8 TP8

```shell
python3 benchmark/kernels/fbgemm/benchmark_fbgemm_grouped_gemm.py --model meta-llama/Llama-4-Scout-17B-16E-Instruct --tp-size 8 --use-fp8-w8a8

grouped-gemm-performance:
    batch_size  FBGEMM Grouped GEMM FP8  SGLang Grouped GEMM FP8
0          1.0                 0.042336                 0.020592
1          2.0                 0.006464                 0.013536
2          4.0                 0.006464                 0.014112
3          8.0                 0.171712                 0.531744
4         16.0                 0.170944                 0.518208
5         32.0                 0.170432                 0.693952
6         64.0                 0.172704                 0.608352
7        128.0                 0.173248                 0.475200
8        256.0                 0.175040                 0.420544
9        512.0                 0.178400                 0.367200
10      1024.0                 0.196736                 0.697968
11      2048.0                 0.230688                 1.385600
12      4096.0                 0.383872                 2.766432
```


![](https://files.mdnice.com/user/59/7304821d-fe5d-411e-a479-c1ea28b42fa7.png)


结论是FBGEMM相比SGLang的分组GEMM实现在MoE模型上可以实现显著的性能提升。这个kernel可以直接应用到SGLang的EP-MoE分组GEMM kernel中,以提升fp16/bf16和per-tensor量化fp8条件下的性能。

## 局限性

当前的局限性在于如果不编译Meta特定的Triton版本,warp专用kernel似乎无法使用。此外,这个kernel目前仅支持fp16/bf16和per-tensor量化的fp8w8a8 - 要与DeepSeek兼容还需要进一步修改。



