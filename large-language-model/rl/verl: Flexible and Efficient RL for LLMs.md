> Slides链接见：https://tongyx361.github.io/blogs/posts/verl-intro/#/title-slide ，对应的演讲见：https://www.youtube.com/watch?v=fct7Jd8-bW8

![](https://files.mdnice.com/user/59/7dc971e5-d3b8-4bef-a11b-90d5b2daf982.png)

![](https://files.mdnice.com/user/59/d630339e-f356-4941-a06d-9f737fed5043.png)

![](https://files.mdnice.com/user/59/418adf81-6b9d-48a9-9227-583c99a3a9f3.png)

这张Slides题为"1.1 Learning to Reason with Large-Scale RL"，通过Table 1详细展示了大规模强化学习（Large-Scale RL）对大型语言模型（LLMs）推理性能的显著提升效果。表格比较了三款大型语言模型在是否采用大规模强化学习以及在多个推理基准测试上的表现：GPT-4o（OpenAI 2024）未采用大规模强化学习（❌），其在AIME 2024、MATH 500、GPQA Diamond和Code Forces上的得分分别为44.6、60.3、50.6和>11.0%；相比之下，o1（OpenAI 2024）和R1（DeepSeek-AI 2025）均采用了大规模强化学习（✅），性能显著提升：o1模型在各项基准测试中得分分别为74.4（AIME）、94.8（MATH）、77.3（GPQA）和>89.0%（Code Forces），而R1模型（预计2025年发布）则进一步刷新了记录，在AIME和MATH上分别达到79.8和97.3，Code Forces达到>96.3%，GPQA Diamond为71.5。表格清晰地展示了大规模强化学习对LLM推理性能的巨大促进作用，采用RL的模型在所有评估指标上均远超未采用RL的GPT-4o，特别是在数学推理（MATH）、编程竞赛（Code Forces）和科学问答（GPQA）等复杂推理任务上表现尤为突出，证明了大规模强化学习是提升大语言模型推理能力的关键技术路径。

![](https://files.mdnice.com/user/59/ca71a513-32b4-4e1f-9a98-f58b1470619c.png)

这张Slides题为"1.2 Learning as Agent with Large-Scale RL"，详细阐述了OpenAI在2025年展望的Deep Research方向，即通过大规模强化学习（Large-Scale RL）实现 Agent 学习，旨在开发能够独立地从网络中发现、推理并整合洞察力的 Agent 。该研究目标聚焦于构建具有自主学习和推理能力的 Agent 系统，这些 Agent 能够主动探索网络资源，从中提取有价值的信息，并通过复杂的推理过程将这些信息整合成有意义的洞察。为实现这一目标，该 Agent 被训练用于处理需要浏览器和Python工具使用的真实世界任务，这意味着 Agent 不仅需要具备文本理解和生成能力，还需要掌握实际工具的使用技能，包括网页浏览、数据抓取、代码执行等复杂操作，体现了从纯文本模型向多模态工具使用 Agent 的重要转变。该研究采用了与OpenAI o1（OpenAI的首个推理模型）相同的强化学习方法，这表明其延续了o1在推理能力方面的技术优势，通过大规模强化学习训练来提升 Agent 在复杂任务中的表现。Slides最后还提及可以查看OpenAI Deep Research的演示视频以获取更多细节，强调了该研究在构建通用 Agent 方面的进展和潜力，代表了人工智能从被动响应向主动学习和探索的重要发展方向，为未来构建真正具有自主学习和推理能力的 Agent 系统奠定了重要基础。

![](https://files.mdnice.com/user/59/e6b5c2fe-ed05-40b5-a7c8-68b1fbd223fa.png)


![](https://files.mdnice.com/user/59/cefa93fe-2f5a-4a47-a52c-8befdfb6c2c7.png)

这张Slides题为"2.1 RL is Complex Dataflow"，核心内容通过图表和文字阐述了强化学习（RL）算法可以被建模为复杂的**数据流图**。图1详细展示了三种典型的RL算法（PPO, Safe-RLHF, ReMax）的数据流图表示，并提供了图例来区分不同组件：红色圆圈代表"actor"（策略网络），黄色圆圈代表"critic"（价值网络），蓝色圆圈代表"reference policy"（参考策略），绿色圆圈代表"reward model"（奖励模型），紫色圆圈代表"cost model"（成本模型）。PPO (a) 算法的数据流图分为三个阶段：1. Actor Gen（策略生成阶段，由Actor模型负责）；2. Forward Pass（数据前向传播阶段，包括"Ref Fwd"参考策略前向、"RM Fwd"奖励模型前向和"Critic Fwd"评论家前向）；3. Training（训练阶段，包括"Actor Training"策略训练和"Critic Training"评论家训练），整个流程从Actor生成数据，经过参考策略、奖励模型和评论家的前向计算，最终用于Actor和Critic的训练。Safe-RLHF (b) 算法的数据流图更为复杂，也分为三个阶段：1. Actor Gen（策略生成）；2. Forward Pass（包括"Ref Fwd"、"RM Fwd"、"Cost Fwd"成本模型前向和"Critic Fwd"）；3. Training（包括"Actor Fwd"策略前向、"Actor Training"和"Critic Training"），值得注意的是"Actor Fwd"与"Actor Training"之间存在一个Lptx（预训练损失）的连接，表明在训练Actor时可能结合了预训练目标。ReMax (c) 算法的数据流图相对简洁，分为三个阶段：1. Actor Gen；2. Forward Pass（包括"RM Fwd"和"Ref Fwd"）；3. Training（仅包含"Actor Training"）。图下方的文字说明指出，这些数据流图的建模参考了Schulman et al. 2017 (PPO), Dai et al. 2023, Li et al. 2024 (Safe-RLHF), 以及Sheng et al. 2025 (ReMax)。整体而言，Slides强调了强化学习算法的复杂性，并提出将其抽象为数据流图是一种有效的分析和理解方式，这与Schaarschmidt et al. 2019, Liang et al. 2021, Sheng et al. 2025等研究的观点一致，为后续构建灵活的RL框架提供了理论基础。

![](https://files.mdnice.com/user/59/1c24ea57-74fe-4525-8bf5-056850b5f989.png)

这张Slides题为"2.2 LLM Workloads Are Distributed"，核心内容通过一个三维方块图详细展示了大型语言模型（LLM）工作负载在多GPU环境中的分布式并行策略。图表以一个4x2xN（N为深度方向，图中未完全展开但由"Model Parallel"轴指示）的GPU网格为例，清晰地描绘了三种主要的并行方式：**ZERO Data Parallel**（零数据并行）、**Pipeline Parallel**（流水线并行）和**Model Parallel**（模型并行）。具体来说：**ZERO Data Parallel**轴垂直排列，表示不同的GPU处理不同的数据分片，例如图中的GPU 0和GPU 4位于同一流水线阶段，但处理不同的数据批次；**Pipeline Parallel**轴水平排列，表示不同的GPU负责模型计算的不同阶段或层，例如GPU 0、GPU 8、GPU 16和GPU 24构成了一个流水线，每个GPU处理模型的一个连续部分；**Model Parallel**轴深入图表，表示单个模型被分割成多个部分，由不同的GPU共同处理，这通常意味着模型的不同层或层内的不同部分被分配到不同的GPU上。图中的方块代表各个GPU，并标有具体的GPU编号（如GPU 0, GPU 4, GPU 8, GPU 12, GPU 16, GPU 20, GPU 24, GPU 28），通过不同的颜色进一步区分了处于不同流水线阶段的GPU组。Slides的文字部分总结了LLM工作负载的两个关键特征：涉及**许多GPU**，强调了LLM训练和推理对计算资源的需求巨大；采用**复杂的并行策略**，指出为了高效利用这些GPU资源，需要结合多种并行技术（如数据并行、流水线并行和模型并行）来优化性能。这张图旨在说明，为了应对LLM的巨大规模和计算需求，分布式计算和多维度并行是不可或缺的策略，通过精细的并行设计，可以有效提升LLM的训练和推理效率，为后续讨论RL在分布式环境中的实现提供了重要的背景基础。

![](https://files.mdnice.com/user/59/08674be8-2828-4ee0-9b78-f744e3aa06d4.png)

这张Slides题为"2.3 RL with LLMs is Large-Scale Distributed Dataflow"，核心观点阐明了将强化学习（RL）与大型语言模型（LLMs）结合时，整个系统构成了一个大规模的分布式数据流。左侧的"RL Dataflow Graph (Inter-Operator)"图展示了一个典型的RL算法（如PPO）的数据流，它由多个操作符（operators）组成，并分为三个主要阶段：1. **Actor Gen**（策略生成），由一个红色的椭圆形表示；2. **Forward Pass**（前向传播），包括"Ref Fwd"（参考策略前向，蓝色椭圆形）、"RM Fwd"（奖励模型前向，绿色椭圆形）和"Critic Fwd"（评论家前向，黄色椭圆形）；3. **Training**（训练阶段），包括"Actor Training"（策略训练，红色椭圆形）和"Critic Training"（评论家训练，黄色椭圆形）。这些操作符之间通过箭头连接，形成一个清晰的计算依赖图。右侧的两个"LLM Large-Scale Distributed Workload (Intra-Operator)"图则进一步揭示了每个RL操作符本身就是一个大规模的分布式计算工作负载。每个分布式工作负载都以一个3D GPU网格（例如4x2xN的配置）表示，并结合了三种并行策略：**ZERO Data Parallel**（垂直方向，不同GPU处理不同数据分片）、**Pipeline Parallel**（水平方向，不同GPU处理模型不同阶段）和**Model Parallel**（深度方向，单个模型分割到多个GPU）。图中明确标注了GPU编号（如GPU 0, GPU 4, GPU 8, GPU 12, GPU 16, GPU 20, GPU 24, GPU 28），并用虚线连接RL数据流图中的操作符到这些分布式工作负载图，直观地说明了在RL与LLMs结合的场景下，不仅RL算法本身是数据流，而且数据流中的每个基本操作（如Actor Gen、Ref Fwd等）都是一个复杂的、多维度并行的LLM分布式计算任务。这强调了RL与LLMs的结合需要高度复杂的分布式系统来支持其大规模计算需求，每个逻辑操作都映射到物理上分布式的GPU集群上执行，体现了现代AI系统中算法复杂性与系统复杂性相互交织的特点。

![](https://files.mdnice.com/user/59/5e160b57-9e4f-4860-893e-04d3d95aa784.png)

这张Slides题为"2.4 Constraints: Data Dependencies & Resource Limitations"，核心内容详细阐述了在大型语言模型（LLMs）中实现强化学习（RL）算法时，由于数据依赖性和资源限制而需要进行的复杂权衡。图表左侧展示了一个简化的"Dataflow Graph D"，描绘了RL算法的逻辑计算流程："Gen"（生成器，红色）产生数据，这些数据被"Ref"（参考策略，蓝色）、"RM"（奖励模型，绿色）和"Value"（价值网络，黄色）模型使用，它们的输出进而驱动"Actor Training"（策略训练，红色）和"Critic Training"（评论家训练，黄色）。图表中间部分通过"Placement"和"Execution Pattern"展示了如何将这些逻辑组件映射到物理资源上："Placement"定义了模型到机器和GPU的映射关系，例如"Actor"映射到Machine A的GPU 0-1，"Critic"映射到Machine B的GPU 2-3，而"Ref"和"RM"则共同映射到Machine C的GPU 4-5；"Execution Pattern"则具体展示了这些模型在不同机器上的执行方式，例如Machine A上的GPU 0和GPU 1并行执行"Gen"和"Actor Training"，Machine B上的GPU 2和GPU 3并行执行"Value"和"Critic Training"，而Machine C上的GPU 4和GPU 5则处理"Ref"和"RM"。Slides强调了几个关键约束和优化原则："Computation with dependencies executes sequentially"（有依赖关系的计算必须顺序执行），"Models in the same stage and on different GPUs can be parallelized"（同一阶段但在不同GPU上的模型可以并行），以及"Colocated models execute sequentially"（共存的模型顺序执行）。这些原则共同揭示了在分布式环境中优化RL算法的挑战，即如何在满足数据依赖性的同时，通过合理的模型放置和执行调度来最大化并行度并有效利用有限的GPU资源。总结指出，实现RL算法与LLMs的结合通常需要复杂的权衡，以平衡数据依赖性、资源分配和并行效率，这对于构建灵活高效的RL系统至关重要（Sheng et al. 2025）。

![](https://files.mdnice.com/user/59/e10e974b-1276-4773-bbf7-8a11db825d0f.png)


![](https://files.mdnice.com/user/59/0257c4ae-a690-4510-8edb-c0f6731d3827.png)

这张Slides题为"3.1 Flexibility: 'Single-Controller'"，核心内容通过一个详细的数据流图（Figure 5）和一段简洁的PPO核心代码（Listing 1）共同阐述了带有KL正则化的PPO（Proximal Policy Optimization）算法的执行流程，并强调了其"单控制器"的灵活性。数据流图清晰地展示了PPO算法的三个主要阶段：1. **Generation stage（生成阶段）**：从"Prompts"（输入提示）开始，通过"Gen"（红色圆圈，代表Actor策略网络）生成"Prompts & Response"（提示与响应）；2. **Experience Preparation stage（经验准备阶段）**：生成的"Prompts & Response"被并行地送入多个模型进行评估，包括"Ref log prob"（蓝色圆圈，代表Reference Policy参考策略）计算参考策略的对数概率、"log prob"（红色圆圈，代表Actor策略网络）计算当前策略的对数概率、"Values"（黄色圆圈，代表Critic价值网络）评估当前状态的价值、"Reward"（绿色圆圈，代表Reward Model奖励模型）计算奖励，所有这些计算结果汇聚成"Experiences"（经验数据），然后存储到"Buffer"（缓冲区）中；3. **Training stage（训练阶段）**：从"Buffer"中取出经验数据，用于更新"Actor Update"（红色圆圈，代表Actor策略网络）和"Critic Update"（黄色圆圈，代表Critic价值网络）。图例明确了红色圆圈代表actor，黄色代表critic，蓝色代表reference policy，绿色代表reward model。Listing 1 "PPO core code in a few lines in verl"提供了一个简洁的Python-like实现，直接对应了数据流图的三个阶段：Stage 1 (Generation) 通过`actor.generate_sequences(prompts)`完成；Stage 2 (Experience Preparation) 依次调用`reward.compute_reward(batch)`、`reference.compute_log_prob(batch)`、`critic.compute_values(batch)`和`compute_advantage(batch, "gae")`来准备经验数据；Stage 3 (Training) 则通过`critic.update_critic(batch)`和`actor.update_actor(batch)`来更新模型。Slides底部还强调了该框架的三个关键特性：编程接口基于"single-controller"范式，RL算法的核心逻辑仅需几行代码即可实现，并且支持PPO、GRPO、RLOO、ReMax、PRIME、DAPO等多种强化学习算法，充分展现了其在RL算法实现上的灵活性、简洁性和广泛适用性。

![](https://files.mdnice.com/user/59/b287ad19-b7d2-43fe-97ce-ce599f75284d.png)

这张Slides题为"3.2 Efficiency: 'Multi-Controller'"，核心内容阐述了`verl`框架如何通过"multi-controller"范式及其一系列特性，实现对算子内部（intra-operator）操作的高效处理。这些特性被细分为四个主要类别：首先是**并行算法（Parallelism Algorithms）**，包括数据并行（Data Parallelism）、张量并行（Tensor Parallelism）、流水线并行（Pipeline Parallelism）以及上下文/序列并行（Context / Sequence Parallelism），这些算法旨在优化大规模模型在多设备上的计算分布，通过不同的并行策略来最大化计算效率和资源利用率；其次是**高效内核（Efficient Kernels）**，它利用了Flash Attention（一种高效的注意力机制，显著减少内存使用和计算复杂度）、Torch Compile（PyTorch的即时编译优化，通过JIT编译提升执行效率）和Liger Kernel（一个自定义或优化的内核，针对特定计算模式进行优化）来加速底层计算；再者是**训练后端（Training Backends）**，`verl`集成了FSDP（Fully Sharded Data Parallel，全分片数据并行）、FSDP2（FSDP的增强版本，提供更好的内存管理和通信优化）和Megatron等主流分布式训练框架，以支持大规模模型的训练，这些后端能够处理TB级别的模型参数和PB级别的训练数据；最后是**生成后端（Generation Backends）**，它支持vLLM（一个用于LLM的高吞吐量推理引擎，专门优化推理性能）和SGLang（一个用于结构化生成任务的语言/运行时，支持复杂的生成模式），并预示未来将支持更多后端。这些全面的效率特性共同构成了`verl`在处理大型语言模型和强化学习工作负载时实现高性能和可扩展性的基础，通过multi-controller范式实现了算子级别的精细控制和优化。


![](https://files.mdnice.com/user/59/99d976b0-e80f-4165-aa39-186dc90f7088.png)

这张Slides题为"3.3 Efficiency: 'Hybrid Engine'"，核心内容阐述了`verl`框架如何通过"混合引擎"（hybrid engine）范式实现算子间（inter-operator）的高效处理，主要利用了**offloading & reloading**（卸载与重载）和**resharding**（重新分片）两大特性。offloading & reloading 能够充分利用GPU内存，而resharding 则支持切换到最优的并行策略。图6以"混合引擎在不同工作负载间切换，改变DP以适应TP"为例，详细展示了这一过程：左侧的PPO数据流图（a）描绘了一个典型的强化学习工作负载，分为三个阶段：1. **Actor Gen**（策略生成）；2. **Forward Pass**（前向传播），包括Ref Fwd、RM Fwd和Critic Fwd；3. **Training**（训练），包括Actor Training和Critic Training。这些阶段通过红色虚线连接到"Source Model"（DeviceMesh 1），该模型配置为`DP=2, TP=2, PP=1`，意味着数据并行度为2，张量并行度为2，流水线并行度为1，由DP0和DP1两个设备处理器组成，每个处理器内部包含TP0和TP1两个张量并行单元。通过混合引擎的resharding功能，系统可以动态切换到"Destination Model"（DeviceMesh 2），其配置为`DP=1, TP=4, PP=1`，即数据并行度变为1，张量并行度变为4，流水线并行度仍为1，此时只有一个设备处理器DP0，但其内部包含TP0到TP3四个张量并行单元。图中蓝色线条展示了从Source Model的四个TP单元到Destination Model的四个TP单元的复杂连接，直观地体现了在不同工作负载或阶段之间，系统如何通过动态调整数据并行（DP）和张量并行（TP）等并行策略，以实现资源的最优配置和计算效率的最大化，从而应对强化学习中复杂且多变的数据流和计算需求。

![](https://files.mdnice.com/user/59/7da5d1d3-85d5-4e29-8190-6fe008241ac2.png)

![](https://files.mdnice.com/user/59/3ac4d18b-815d-4711-a2a9-e1ab25e06b8e.png)

![](https://files.mdnice.com/user/59/2a0d0c0b-562c-4a0e-b497-14266bc4d35c.png)

这里介绍了一下开源的进展。

![](https://files.mdnice.com/user/59/445fe7b9-9549-46ee-8341-763f15c8cc02.png)

![](https://files.mdnice.com/user/59/966e9d3d-fba8-4aeb-90e8-f2acc50a0f0a.png)

这两张Slides分别介绍了"4 Paradigm behind verl: HybridFlow (Sheng et al. 2025)"和"4.1 Background: Single-Controller vs. Multi-Controller"，详细阐述了`verl`框架背后的核心范式HybridFlow以及两种不同的分布式计算架构。第一张Slides标题"4 Paradigm behind verl: HybridFlow (Sheng et al. 2025)"简洁地指出了`verl`框架的理论基础，即HybridFlow范式，这是由Sheng等人在2025年提出的创新性编程范式。第二张Slides题为"4.1 Background: Single-Controller vs. Multi-Controller"，通过Figure 7详细对比了两种分布式计算架构：**Single-Controller (MPMD)**和**Multi-Controller (SPMD)**。Single-Controller (MPMD)架构展示了一个集中式控制模式，其中单一的"Ctrlr"（控制器）负责协调多个"Host"（主机）和"Dev"（设备）对，控制器通过一系列操作（紫色和蓝色圆圈）向三个独立的Host-Dev工作单元发送指令和数据，每个Host-Dev对都执行自己的任务流，Host先进行一些操作（黄色圆圈），然后将任务分派给Dev，Dev执行计算（不同颜色的矩形块），并可能进行内部同步，Host和Dev之间存在数据依赖和通信，不同Host-Dev对之间也可能存在依赖关系，这种模式下一个中心控制器管理所有工作节点，每个节点可以运行不同的程序。

Multi-Controller (SPMD)架构则展示了一个分布式控制模式，包含两个独立的计算单元，每个单元都由一个"Host"、一个"Ctrlr"（控制器）和一个"Dev"（设备）组成，每个单元内部Host、Ctrlr和Dev各自执行一系列操作（橙色圆圈、黄色和绿色矩形块），并通过虚线箭头进行通信和数据传输，操作流程从"step k"进行到"step k+1"，每个单元的计算流程相似，都包含一系列橙色圆圈操作、黄色和绿色矩形块的计算、同步点，最后以"read"操作结束，这种模式下每个工作节点都有自己的控制器，它们运行相同的程序但处理不同的数据。这两种架构在并行计算和资源管理上各有特点，适用于不同的应用场景，为HybridFlow范式的设计提供了重要的理论基础。

![](https://files.mdnice.com/user/59/30e24c79-bad3-40a8-ab72-6be87067335b.png)

这张Slides题为"4.2 Trade-off: Single-Controller or Multi-Controller?"，核心内容通过Table 2详细讨论了单控制器和多控制器两种范式之间的权衡取舍。表格清晰地对比了两种范式的优缺点：**Single-Controller**范式的优点是"Flexible"（灵活），能够适应不同的计算需求和动态调整，但其缺点是"Communication Overhead"（通信开销），由于需要中心控制器协调所有工作节点，会产生大量的通信成本和延迟；**Multi-Controller**范式的优点是"Efficient"（高效），每个工作节点都有独立的控制器，减少了通信开销并提高了并行效率，但其缺点是"Complex Programming"（编程复杂），需要处理多个控制器之间的同步和协调问题，增加了编程的复杂性和调试难度。Slides底部通过两个表情符号和问题进一步强调了这种权衡的挑战性。


![](https://files.mdnice.com/user/59/7745a0d5-2af4-48aa-afa9-06c2b037d63d.png)

这张Slides题为"4.3 New Paradigm: Hybrid-Controller!"，核心内容引入了一种名为Hybrid-Controller（混合控制器）的新范式，其定义为"Hybrid-Controller = Single-Controller + N x Multi-Controller"，旨在结合单控制器和多控制器的优势来高效处理数据流。图表（Figure 8）详细展示了这种混合控制器的工作机制：左侧是"Single-Controller (Inter-Operator)"部分，描绘了一个典型的强化学习（RL）算法的数据流，类似于PPO，从"Prompts"（输入提示）开始，通过"Gen"（生成器，红色圆圈）产生"Prompts + Responses"（提示与响应），这些响应随后被用于计算多个关键指标：通过"Ref log prob"（参考策略对数概率，蓝色圆圈）、"log prob"（当前策略对数概率，红色圆圈）、"Values"（价值，黄色圆圈）和"Reward"（奖励，绿色圆圈）来生成"Experiences"（经验数据），最终驱动"Actor Update"（策略网络更新，红色圆圈）和"Critic Update"（价值网络更新，黄色圆圈），这个单控制器负责协调RL算法中不同操作符之间的高层数据流和依赖关系，体现了其"Inter-Operator"（算子间）的管理特性。右侧是两个"Multi-Controller (Intra-Operator)"部分，每个都代表一个大规模的分布式计算工作负载，用于执行单个RL操作符，展示了一个3D GPU网格，包含GPU 0、GPU 4、GPU 8、GPU 12、GPU 16、GPU 20、GPU 24、GPU 28等多个GPU，结合了三种并行策略：ZERO Data Parallel（垂直方向，不同GPU处理不同数据分片）、Pipeline Parallel（水平方向，不同GPU负责模型计算的不同阶段）和Model Parallel（深度方向，单个模型分割到多个GPU），不同颜色的GPU块表示它们在流水线中的不同阶段，这些多控制器负责优化单个操作符内部的执行效率，通过精细的分布式并行策略来最大化GPU资源的利用率。图中的虚线连接了左侧单控制器的"Actor Update"和"Critic Update"节点到右侧的两个多控制器图，表明混合控制器通过一个高层级的"Single-Controller"来管理整个强化学习算法的逻辑数据流（Inter-Operator），同时利用多个"Multi-Controller"来高效地执行数据流中的每个大规模、分布式操作符（Intra-Operator），从而实现了灵活性与效率的结合，以应对大型语言模型强化学习中复杂的计算需求。

![](https://files.mdnice.com/user/59/4eba9107-785c-4c8e-9c5d-cc14ce4dcc32.png)

这张Slides题为"4.4 Implementation in verl"，核心内容详细阐述了在`verl`框架中，单控制器（single-controller）如何通过远程过程调用（RPC）与多控制器（multi-controller）工作组进行交互，以实现分布式强化学习算法的灵活且高效执行。左侧的"Listing 2: PPO core code in single-controller"代码片段展示了PPO算法在单控制器环境下的核心逻辑，它是一个简洁的Python风格循环，从`for prompts in dataloader:`开始，清晰地分为三个阶段：1. **Stage 1: Generation**（生成阶段），通过`batch = actor.generate_sequences(prompts)`生成序列；2. **Stage 2: Experience Preparation**（经验准备阶段），计算奖励`batch = reward.compute_reward(batch)`、参考策略对数概率`batch = reference.compute_log_prob(batch)`、评论家价值`batch = critic.compute_values(batch)`和优势函数`batch = compute_advantage(batch, "gae")`；3. **Stage 3: Training**（训练阶段），更新评论家`critic.update_critic(batch)`和策略网络`actor.update_actor(batch)`。这段代码以顺序调用的方式清晰地表达了RL算法的逻辑数据流。右侧的"Listing 3: Example distributed code in multi-controller"代码片段则展示了多控制器环境下的分布式实现细节。它定义了两个工作类：`class CriticWorker(3DParallelWorker):`和`class ActorWorker(3DParallelWorker):`，两者都继承自`3DParallelWorker`，并使用`@register(dispatch_mode=3D_PROTO)`装饰器进行注册。`CriticWorker`类包含`def compute_values(self, batch: DataProto):`方法，负责执行评论家模型的前向传播`values = self.critic.forward(batch)`并更新批次数据`batch.update(values=values)`；`ActorWorker`类包含`def update_actor(self, batch: DataProto):`方法，负责执行策略网络的前向传播`loss = self.actor(batch)`并进行反向传播`loss.backward()`。这些被`@register`装饰的方法实际上是单控制器中对应函数（如`critic.compute_values`和`actor.update_actor`）的远程实现。Slides底部文字进一步解释道，`register`装饰器工具负责管理分布式数据传输，从而简化了多控制器编程的复杂性。这种设计使得开发者可以在单控制器中编写简洁的RL算法逻辑，而底层的分布式执行和数据传输则由`verl`框架通过RPC和`@register`机制透明地处理，实现了灵活性与效率的结合。


![](https://files.mdnice.com/user/59/cf7543a8-f242-4cd5-85c5-fa29fa056599.png)

![](https://files.mdnice.com/user/59/4f64b15d-8250-4891-8dab-a6dce54d4927.png)

这张Slides题为"5.1 Async Engine for Multi-Turn Rollout"，主要通过Figure 9详细对比了同步（Synchronous）和异步（Asynchronous）两种引擎在多轮（Multi-Turn）Rollout中的工作方式。**Synchronous Rollout** 展示了传统的批处理模式：系统并行初始化"Runtime 0"和"Runtime 1"，每个Runtime执行"LLM Gen"（大语言模型生成）→"Env Exec"（环境执行）→"LLM Gen"的序列，关键特点是当"Trajectory 0 finishes, start a new trajectory"（轨迹0完成，开始新轨迹）时，需要等待当前轨迹达到同步点才能初始化"Runtime 2"，整个流程是线性的，即使有并行初始化，后续步骤也倾向于等待前一个批次或轨迹的关键阶段完成，最终进行"Reward Calculation"（奖励计算）。

**Asynchronous Rollout** 则展示了更高效的并行模式：同样并行初始化"Runtime 0"和"Runtime 1"，但与同步模式不同，异步模式下"LLM Gen"和"Env Exec"的步骤在不同轨迹之间高度交错和并行，例如"Runtime 0"的"LLM Gen Env Exec LLM Gen"与"Runtime 1"的"LLM Gen Env Exec LLM Gen"在时间线上有显著重叠，关键优势在于一旦"Trajectory 0 finishes"（轨迹0完成），立即开始初始化"Runtime 2"，无需等待"Runtime 1"的完成，类似地，一旦"Trajectory 1 finishes"（轨迹1完成），立即开始初始化"Runtime 3"，这种设计能够更灵活地利用资源，一旦一个轨迹完成就可以立即启动新的轨迹，从而提高整体吞吐量。Slides底部通过文字进一步明确了两种引擎的核心区别：**Synchronous Engine（同步引擎）**在批处理中会"returns all the outputs in the batch at the same time"（同时返回批处理中的所有输出），而**Asynchronous Engine（异步引擎）**则会"returns each output as soon as it is ready"（一旦每个输出准备好就立即返回）。这种异步引擎的设计使得`verl`框架能够在多轮强化学习任务中实现更高的并行度和响应速度，通过动态启动新任务和立即处理完成的任务，显著提升了整体系统的效率和资源利用率。

![](https://files.mdnice.com/user/59/77d779d9-c5ed-425e-8041-072672f1eed0.png)

![](https://files.mdnice.com/user/59/5235c423-04d4-4453-8be5-d295471c42c2.png)

- https://github.com/volcengine/verl/issues/1882

这两张Slides分别介绍了"5.2 Basic Capability Support"和"5.3 Diverse Environments & Tools (Ongoing)"，详细阐述了`verl`框架在基础能力支持和多样化环境工具集成方面的进展。第一张Slides题为"5.2 Basic Capability Support"，列出了框架目前支持的基础能力：1. **Multi-Modal**（多模态支持），包括"Qwen2.5-VL, Kimi-VL, etc."等视觉语言模型，表明`verl`框架已经能够处理文本和图像等多种模态的输入；2. **Multi-Turn & Tool Using**（多轮对话和工具使用），标注了"see progress at #1882"，其中"#1882"以蓝色高亮显示，表明这是一个可点击的链接或参考ID，指向GitHub上的具体进展页面，显示了框架在多轮对话和工具调用能力方面的开发进度；3. **...**（省略号）表示列表内容可能还有更多项目或正在进行中。第二张Slides题为"5.3 Diverse Environments & Tools (Ongoing)"，强调了这项工作正在进行中，并邀请观众讨论或贡献以下几个方面：首先是"Our ongoing RFC #1172"（我们正在进行的RFC #1172），RFC（Request for Comments）是技术规范文档，表明团队正在制定相关的技术标准或协议；其次是"Integrating protocols like MCP"（集成像MCP这样的协议），MCP可能指Model Context Protocol或其他相关协议，显示了框架在协议集成方面的努力；最后是"Integrating existing environments & tools"（集成现有的环境和工具），并列举了两个具体例子：一个是"KORGym @ ByteDance Seed (Shi et al. 2025)"，这是字节跳动Seed团队开发的强化学习环境；另一个是"Atropos @ Nous Research (Dakota Mahan 2025)"，这是Nous Research团队开发的环境。这两张Slides共同展示了`verl`框架在基础能力建设方面的全面性和在多样化环境工具集成方面的开放性，体现了框架致力于构建一个支持多模态、多轮对话、工具使用以及各种强化学习环境的综合性平台。

- https://github.com/volcengine/verl/issues/1172
- https://github.com/multimodal-art-projection/KORGym
- https://github.com/NousResearch/atropos


![](https://files.mdnice.com/user/59/77d7813c-c6d2-4a92-b741-3043489ed821.png)

![](https://files.mdnice.com/user/59/00431367-6f08-471c-b916-d0920a97c650.png)

这张Slides介绍了"6.1 Efficient RL with Huge MoE like DeepSeek-V3-671B (V0.4+)"和，详细阐述了`verl`框架在支持大规模混合专家模型（MoE）高效强化学习训练方面的能力。明确指出`verl`支持对像DeepSeek-V3-671B这样的大规模MoE模型进行高效的强化学习训练，并基于以下特性：1. **Training**（训练）："MoE models classes supporting diverse parallelism strategies like Expert Parallelism based on Megatron GPTModel"（MoE模型类支持基于Megatron GPTModel的多样化并行策略，如专家并行），这表明框架能够处理MoE模型特有的专家并行计算模式；2. **Inference**（推理）："Multi-node inference"（多节点推理），支持跨多个节点的分布式推理；3. **Hybrid**（混合）："Parameter sharding manager for Megatron-Core V0.12 + latest inference engines"（为Megatron-Core V0.12和最新推理引擎提供参数分片管理器），实现了训练和推理引擎之间的无缝集成。

![](https://files.mdnice.com/user/59/f6a00140-d5c4-4087-aef7-fa25d721d217.png)

![](https://files.mdnice.com/user/59/f6176327-fd12-4d0a-94fe-69baf8239606.png)


![](https://files.mdnice.com/user/59/73b092da-bb6e-4255-94c0-7ccfbc153c67.png)

参考和2025年Q3 Roadmap。

![](https://files.mdnice.com/user/59/55fa9381-2059-4bb3-a88c-ce346b5840bd.png)

![](https://files.mdnice.com/user/59/cbbb73cf-5a49-4198-b291-6f95b0d9e68b.png)

![](https://files.mdnice.com/user/59/392ca0db-004f-421c-8958-0b5ee9c325f9.png)

这张Slides题为"7.1 Sequence Packing"，主要介绍了如何通过序列打包（Sequence Packing）来优化注意力机制，以避免不同数据序列之间的交叉污染。Slides的核心内容通过Figure 10的两个注意力掩码（attention mask）图示进行对比说明：左侧的图示展示了一个包含两个数据序列（"The Statue of Liberty <eod>"和"Hi Alice <eod>"）的原始打包序列，其注意力掩码呈现标准的因果（causal）模式，即每个词元可以关注到其自身以及之前的所有词元，在这种情况下，第二个序列的词元（如"Hi"）可以错误地关注到第一个序列的词元（如"The Statue of Liberty"），这被称为交叉污染；一个向右的箭头指向右侧的图示，该图示展示了经过"调整"（tweaked）的注意力掩码，在这个调整后的掩码中，每个数据序列都拥有独立的因果注意力范围，即"The Statue of Liberty <eod>"只关注其自身序列内的词元，而"Hi Alice <eod>"也只关注其自身序列内的词元，有效地阻止了不同序列间的注意力流，从而避免了交叉污染。Slides还列出了实现这一优化的两个关键步骤：1. 移除填充（padding）词元并将多个数据序列打包成一行；2. 调整注意力掩码和位置ID（position IDs）以避免交叉污染。最后，Slides指出要启用此功能，需要使用`use_remove_padding`参数。这种序列打包技术通过智能的注意力掩码调整，在保持计算效率的同时确保了不同数据序列之间的独立性，避免了模型在处理多个序列时可能出现的注意力泄露问题，这对于强化学习中的批量数据处理和模型训练具有重要意义。


![](https://files.mdnice.com/user/59/2dd7c04a-1228-4b9e-988a-894e812a651d.png)

![](https://files.mdnice.com/user/59/fdcf15ab-326e-4628-b7fa-8ab3886ad524.png)

![](https://files.mdnice.com/user/59/c0c917d5-2a35-48ca-91b5-c0c3925aa95e.png)

这三张Slides分别介绍了"7.2 DP Balancing"、"7.2.1 Load Imbalance in DP"和"7.2.2 Balancing across DP Ranks"，详细阐述了数据并行（Data Parallelism, DP）中的负载均衡问题及其解决方案。第一张Slides标题为"7.2 DP Balancing"，简洁地指出了数据并行平衡的主题。第二张Slides题为"7.2.1 Load Imbalance in DP"，通过三个要点详细分析了数据并行中的负载不均衡问题：1. **同步需求**：并行计算通常需要在不同rank之间进行**同步**（synchronization），这是分布式计算的基本要求；2. **DP的广泛应用**：**数据并行（DP）**如ZeRO是最常用的并行策略，在大型语言模型训练中被广泛采用；3. **负载不均衡问题**：然而，DP性能可能受到**负载不均衡**（load imbalance）的损害，这在长上下文训练中尤其严重，因为不同样本的序列长度差异很大，导致不同GPU处理的有效token数量差异显著。第三张Slides题为"7.2.2 Balancing across DP Ranks"，通过Figure 11详细对比了有无DP平衡的情况：**无DP平衡**（w/o DP Balancing）展示了典型的负载不均衡场景；**有DP平衡**（w/ DP Balancing）则展示了通过样本重排序实现的负载均衡，通过交叉箭头表示样本在rank之间的重新分配，使得每个rank都处理差不多数量的有效token，实现了负载的均匀分布。Slides底部说明了实现机制：通过"reordering the samples in each batch"（重排序每个批次中的样本）来"balance the valid tokens dispatched to each rank"（平衡分派到每个rank的有效token），并指出要启用此功能，需要使用`balance_batch`参数。这种DP平衡技术通过智能的样本重排序，有效解决了数据并行中的负载不均衡问题，特别是在长上下文训练场景下，能够显著提升分布式训练的效率，确保所有GPU资源得到充分利用，避免因负载不均衡导致的性能瓶颈和资源浪费。

![](https://files.mdnice.com/user/59/a79a5871-fff2-4e41-86df-24cea6b6eee0.png)

这张Slides题为"7.2.3 Balancing across Micro Batches"，深入阐述了在梯度累积（gradient accumulation）过程中，仅平衡批次内的有效token是不够的，因为数据并行（DP）是以微批次（micro batch）为单位进行同步的，这导致了更细粒度的负载不均衡问题。Slides首先明确了问题所在：在梯度累积中，"not enough to only balance valid tokens in a batch"（仅平衡批次内的有效token是不够的），根本原因是"since DP syncs in the unit of micro batch"（因为DP是以微批次为单位进行同步的），这意味着即使整个大批次在宏观上是平衡的，但如果同步发生在更细粒度的微批次级别，这些微批次内部的负载不均衡仍然会导致效率低下。为了解决这个问题，`verl`框架"further supports to"（进一步支持）通过"balance the valid tokens across micro batches"（平衡微批次间的有效token）来实现更精细的负载均衡。具体实现机制是"by evenly dividing the data sequences in the batch before packing into micro batches"（通过在打包成微批次之前均匀分割批次中的数据序列），这表明框架在预处理阶段会智能地将输入数据序列分配到各个微批次中，确保每个微批次的计算负载更加均匀。要启用此功能，需要使用`use_dynamic_bsz`参数。这种跨微批次的平衡技术解决了分布式训练中梯度累积阶段的细粒度负载不均衡问题，通过智能的数据序列分割和微批次级别的负载均衡，确保每个微批次的计算负载相对均匀，从而减少同步等待时间，提高整体训练效率，特别是在处理变长序列和长上下文训练场景中具有重要价值。


![](https://files.mdnice.com/user/59/5d9a61a5-ce8d-4b62-9ed0-d08388f86909.png)

其他的一些feature。

![](https://files.mdnice.com/user/59/3c68730e-557c-4bc2-befa-6488fcdf8e7a.png)

![](https://files.mdnice.com/user/59/93014d61-11e5-4ab1-ac59-a9514c184e6a.png)

![](https://files.mdnice.com/user/59/977a5090-041a-448f-853f-85094465abb7.png)

![](https://files.mdnice.com/user/59/7f7c0c8f-0190-4cc9-9872-4b50302d471a.png)

第一张Slides标题为"8 Programming Guide"，简洁地指出了编程指南的主题。

第二张Slides题为"8.1 Customizing the Dataset"，详细说明了`verl`框架中标准强化学习（RL）数据集的结构和必需字段：**`prompt`**字段被描述为"a list of messages"（消息列表），具有字典结构`{"role": "...", "content": "..."}`，代表对话或交互历史；**`data_source`**字段"used to choose the reward function"（用于选择奖励函数），在确定奖励计算方式中发挥作用；**`reward_model`**字段是一个包含两个子字段的字典：`"ground_truth"`（真实或目标奖励值）和`"style"`（可以是"like 'model' or 'rule'"（如"模型"或"规则"），表示定义奖励风格的不同方法）；**`extra_info`**是可选字段，"a dict containing extra information"（包含额外信息的字典），为额外数据提供灵活性。对于视觉语言模型（VLM）RL，`verl`期望字段"images"和/或"videos"，表明视觉模态被纳入到这种特定类型RL的数据集中。Slides还提供了实际实现的参考："For examples, please check the `examples/data_preprocess`"（有关示例，请查看`examples/data_preprocess`）。

第三，四张Slides说明了用户可以通过配置文件自定义字段名称，并指向`ppo_trainer.yaml`等文件中的`data`部分获取更多信息，对于高级自定义，`verl`框架提供`data.custom_cls`配置。


![](https://files.mdnice.com/user/59/f53ae6e5-0266-4cfa-a564-51c6e152a15c.png)


这两张Slides讲了在`verl`框架中如何自定义奖励函数的配置和实现方法。第一张Slides题为"8.2 Customizing the Reward"，首先说明了`verl`允许通过`custom_reward_function`配置来定义自定义奖励函数，然后通过Listing 7展示了自定义奖励函数的YAML配置结构：`custom_reward_function`包含`path`（指向包含函数定义的`.py`文件的路径，初始值为null）和`name`（函数名称，设置为"compute_score"，表示在`def`后定义的函数名）两个字段，同时配置了`reward_model`的`reward_manager`为"naive"。

通过Listing 8提供了CLI配置示例，展示了如何通过命令行参数来设置自定义Reward函数：`--custom_reward_function.path=./examples/reward_fn/custom_reward_fn.py`（指定自定义奖励函数文件路径为`./examples/reward_fn/custom_reward_fn.py`）、`--custom_reward_function.name=compute_score`（设置自定义奖励函数名称为"compute_score"）和`--reward_model.reward_manager=naive`（配置奖励管理器为"naive"）。这些Slides共同为开发者提供了在`verl`框架中集成自定义奖励函数的完整指导，从YAML配置文件到命令行参数设置，涵盖了通过指定Python文件路径和函数名称来定义自定义奖励函数的所有必要步骤，同时展示了默认的"naive"奖励管理器设置，为强化学习中的奖励函数定制提供了灵活且易于使用的配置机制。这个代码段和上面的config的配置是等价的。


![](https://files.mdnice.com/user/59/e69f950f-4375-4dad-bc34-c4017d681b37.png)

这张Slides展示了reward 函数是如何在NaiveRewardManager里面被调用的。

![](https://files.mdnice.com/user/59/00ac563a-b907-4cdf-851f-ae9a61efb0c4.png)

![](https://files.mdnice.com/user/59/56a4c109-6e3f-45c4-a625-db7ce691b15d.png)

这张Slides题为"8.3 Customizing the Loss Function"，详细阐述了在`verl`框架中自定义损失函数的三种主要方法。首先，Slides明确指出修改损失函数最便捷的方式是寻找`.backward()`调用，这是PyTorch等深度学习框架中触发反向传播的关键方法，通过定位这个调用点可以找到损失计算的核心位置。接下来，Slides提供了三种具体的自定义损失函数的策略：第一种是修改类似`compute_policy_loss`的函数，这类函数通常负责计算策略网络的损失，是强化学习算法中的核心组件，通过修改这些函数可以改变策略优化的目标和方式；第二种是添加损失项，如`entropy_loss`（熵损失），熵损失常用于鼓励策略的探索性，防止策略过早收敛到局部最优，通过在总损失中加入熵项可以平衡探索与利用的关系；第三种方法没有在slides中完整显示，但从编号"3."可以推断还有其他自定义损失的方式。Slides还通过一个具体的代码示例（Listing 10）展示了`DataParallelPPOActor.update_policy`方法中简化的损失函数定义：代码展示了一个典型的PPO损失函数计算过程，包括调用`compute_policy_loss`函数计算策略损失`pg_loss`（传入`old_log_prob`、`log_prob`、`advantages`等参数），计算熵损失`entropy_loss = agg_loss(loss_mat=entropy)`，然后将策略损失和熵损失结合形成最终的策略损失`policy_loss = pg_loss - entropy_loss * entropy_coeff`，接着计算KL散度损失`kld = kl_penalty`和`kl_loss = agg_loss(loss_mat=kld)`，最终将所有损失项组合成总损失`policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef`，并调用`loss.backward()`进行反向传播。这个示例清晰地展示了如何在PPO算法中组合多种损失项（策略损失、熵损失、KL散度损失），并通过相应的系数（如`entropy_coeff`、`kl_loss_coef`）来平衡不同损失项的重要性，为开发者提供了在`verl`框架中自定义和扩展损失函数的具体参考和实现模板。

![](https://files.mdnice.com/user/59/d2c25606-c09e-45b6-98e6-082e54d728fc.png)

![](https://files.mdnice.com/user/59/b1fb8717-b182-44fc-b2dd-b205aa390b12.png)

这两张Slides分别题为"8.4 Customizing the Training Logic"，详细阐述了在`verl`框架中如何自定义训练逻辑的核心方法和具体实现。第一张Slides明确指出，如前所述，主要的训练逻辑集中在训练器类（如`RayPPOTrainer`）的`fit`函数中，这是整个训练流程的核心控制点。作为具体示例，Slides提到`DAPORayTrainer`类重写了`fit`函数以实现"动态采样"（dynamic sampling）特性，这是一种高级的训练优化技术，能够根据训练过程中的反馈动态调整数据采样策略，从而提高训练效率和模型性能。第二张Slides通过Listing 11展示了`DAPORayTrainer`中简化的`fit`函数实现，其中动态采样部分被特别高亮显示：代码从第4行开始，初始化`batch = None`，然后进入训练数据加载器的循环`for batch_dict in self.train_dataloader:`，在第6行创建新批次`new_batch = DataProto.from_single_dict(batch_dict)`并递增生成批次计数器`num_gen_batches += 1`，第8行执行Actor rollout工作组的序列生成`gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)`，第9行将生成的输出与新批次合并`new_batch = new_batch.union(gen_batch_output)`。关键的动态采样逻辑从第10行开始：首先检查`if not self.config.algorithm.filter_groups.enable:`，如果未启用过滤组功能，则直接使用新批次`batch = new_batch`（第11行）；否则进入`else`分支（第12行），在第13行添加注释"Getting `kept_traj_idxs` ..."表示获取保留轨迹索引的过程，第14行根据保留的轨迹索引过滤新批次`new_batch = new_batch[kept_traj_idxs]`，第15行实现条件批次累积逻辑`batch = new_batch if batch is None else DataProto.concat([batch, new_batch])`，第16行获取提示批次大小`prompt_bsz = self.config.data.train_batch_size`，第17-20行实现动态采样的核心控制逻辑：当批次中的提示数量小于设定的批次大小时`if num_prompt_in_batch < prompt_bsz:`，获取最大生成批次数`max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches`，如果达到最大生成批次数或其他终止条件`if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:`，则继续循环`continue`，否则跳出循环处理当前累积的批次。这个实现展示了DAPO算法如何通过动态采样机制，根据数据质量和训练需求智能地调整批次大小和数据选择策略，从而在保证训练效果的同时提高计算资源的利用效率，为开发者提供了在`verl`框架中实现复杂训练逻辑和算法创新的具体参考模板。




