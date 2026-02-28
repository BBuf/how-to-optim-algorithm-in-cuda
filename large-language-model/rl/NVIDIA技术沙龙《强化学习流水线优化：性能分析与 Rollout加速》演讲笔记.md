> Slides来自BiliBili NVIDIA英伟达频道上传的NVIDIA专家面对面技术沙龙《强化学习流水线优化：性能分析与 Rollout加速》视频讲解。这里参考视频并更详细记录了Slides的要点，作为学习使用。这个演讲系统性地回顾了强化学习（RL）流水线优化的关键策略和实践。它首先强调了使用Nsight Systems对RL框架进行性能Profile的重要性，具体包括如何在Ray Actor中添加Nsight参数，以及如何解释从RL框架中获取的Nsight Porfile文件。其次，Slides提供了优化训练/生成参数的详细技巧，这又细分为两个主要方面：Actor训练优化，涵盖了推荐启用的动态批处理（dynamic batching）和序列打包（sequence packing），以及对卸载（offload）、重计算（recompute）和融合内核（fused kernel）的策略性使用；Rollout优化，则侧重于提高GPU内存利用率（gpu_memory_utilization）、利用CUDA Graph以及解决Rollout阶段的长尾问题。最后，Slides以Qwen 2.5 7B模型为例总结了RL参数调优的实践经验，并提及了Qwen3 235B MoE模型的SOTA设置，为实际应用中的性能调优提供了全面的指导。

![](https://files.mdnice.com/user/59/747ab0e8-f2eb-4917-ae3a-fd96a58a10be.png)

![](https://files.mdnice.com/user/59/3e6df066-fdf0-4258-bbc8-95850f2883fe.png)


这个演讲主要涵盖五个核心主题：首先是对强化学习系统的整体概述，接着深入探讨如何使用NVIDIA Nsight Systems工具对强化学习流水线进行性能分析，然后介绍提升强化学习性能的具体方法和技巧，随后通过调优Qwen强化学习模型的实例来展示实际应用，最后对整个内容进行总结。

![](https://files.mdnice.com/user/59/80224cfc-12f2-49de-874c-7c24ecb04c53.png)

这张Slides探讨了Reasoning语言模型（如Kimi K1.5、Qwen3、Seed-thinking和Llama-Nemotron）的发展现状，重点分析了它们与Chat模型的根本区别以及强化学习在模型后训练阶段的关键作用。Slides通过对比表格详细阐述了两种模型在四个核心维度上的差异：扩展范式（Chat模型使用"下一token预测"，Reasoning模型采用"思维链上的强化学习"）、Reasoning类型（Chat模型依赖"系统1快速直觉推理"，Reasoning模型运用"系统2慢速努力推理"）、指令方式（Chat模型关注"如何做"的过程指导，Reasoning模型聚焦"做什么"的结果导向）以及交互模式（Chat模型强调"聊天/交互性"，Reasoning模型偏向"研究或规划/后台运行"）。Slides下方展示了强化学习系统的复杂性，指出其需要多个模型协同参与（包括负责序列生成和Policy更新的Actor模型、评估响应质量的Reward模型、预测响应质量的Critic模型以及维持模型稳定性的Reference模型），这种多组件架构为效率优化带来了挑战，且强化学习有很多种类型的算法如PPO、GRPO、DAPO等算法。

![](https://files.mdnice.com/user/59/e2cd2fd9-ff79-44a3-ba46-01e4b0b49e9b.png)

这张Slides展示了一个基于GRPO的Single Controller强化学习训练流水线的完整数据流架构和性能分析。整个系统分为三个核心阶段：首先是Rollout阶段，从prompt开始通过Actor模型生成多个response序列，这是数据收集和经验生成的关键环节；其次是评估与损失计算阶段，涉及多个模型的协同工作，包括Reference模型计算Ref logprob用于KL散度约束、Actor模型计算当前策略Logprob、Old Actor模型计算旧策略Old logprob用于重要性采样、Reward模型评估响应质量生成奖励值R_N，基于这些输出计算KL loss、Token loss和Advantage值，最终组合成Policy loss；最后是训练阶段，Policy loss反向传播更新Actor模型参数，然后更新后的Actor模型参与新一轮的Rollout。

Nsight System性能时间线分析显示，整个训练迭代总耗时约501.110秒，其中Rollout生成阶段占205.707秒，训练步骤包含old_log_prob计算（85.205秒）、reference模型计算（80.609秒）、reward和advantage计算以及update_actor参数更新（126.090秒），为理解RL系统性能瓶颈和优化方向提供了量化的技术依据。

![](https://files.mdnice.com/user/59/f5af908b-1869-402a-8670-2d47715944cd.png)

![](https://files.mdnice.com/user/59/14353aac-a15a-4717-9ba8-a1ddab6ac46b.png)


RL训练很耗时，并且整个pipline里面有多个模块，我们需要关注每个模块的计算和通信的延迟，我们可以使用Nsight Systems来做系统的profiling。

![](https://files.mdnice.com/user/59/09c81854-21cc-4bd1-b9c6-312fb95e56d7.png)

Slides先介绍了nsys的通用命令结构`nsys [command_switch][optional command_switch_options][application][optional application_options]`，并通过具体示例展示了如何使用`nsys profile`命令收集CUDA API和NVTX事件的性能数据。例如我们可以在Megatron-LM使用这个命令直接做profile。

不过在RL训练中使用nsys有个问题：由于RL框架采用Ray进行任务编排，传统的nsys命令无法直接对分布式任务（如策略模型更新、Rollout数据收集和评论模型更新等）进行 profile ，因为这些任务由Ray在远程节点上启动执行。为解决这一问题，Slides提到了两种集成方案：一是通过`ray.remote`装饰器在`runtime_env`中指定`"nsight": "default"`，二是在初始化`RayActor`实例时通过`RayActor.options(runtime_env={"nsight":"default"}}).remote()`设置，这些方法使得nsys能够在Ray的分布式 worker 中运行，有效捕获RL训练中分布式任务的性能数据。例如在VerL中可以使用第2种方式，最底部的链接给出了这两种方式的说明文档。

![](https://files.mdnice.com/user/59/75173d3b-884a-4593-a7d7-af8c15b029b4.png)

这里展示了如何在基于Ray的强化学习框架（以`verl`为例）中集成NVIDIA Nsight Systems (nsys)进行Profile，在Single Controller架构下需要同时追踪控制器进程和worker以获取完整的性能信息。Slides提供了两种具体的集成方法：一是针对Single Controller进程的 profile ，通过在`verl/verl/trainer/main_ppo.py`文件中使用`TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()`将nsight配置注入到`TaskRunner`的运行时环境；二是针对worker的 profile ，在`verl/verl/single_controller/ray/base.py`文件中通过`ray_cls_with_init.update_options({"runtime_env": {"env_vars": env_vars, "nsight": self.worker_nsight_options,},"name": name,})`更新Ray类的运行时环境，将 worker 特有的nsight选项和环境变量传递进去。Slides还详细列出了`nsight_options`的配置参数，包括追踪CUDA API、NVTX事件、cuBLAS和UCX库活动，启用CUDA内存使用和CUDA图追踪，以及设置由CUDA Profiler API控制的捕获范围，为在分verL中实现精细化性能分析提供了参考。

![](https://files.mdnice.com/user/59/fc02723e-3513-4499-b237-a9df575a1ba8.png)

在VerL中使用Nsight Systems进行Porf主要问题是，由于RL训练通常涉及多个步骤和数百甚至数十个GPU，生成的nsys性能文件会变得异常庞大，难以分析。为解决这一问题，Slides提出了通过精细化控制 profile 范围来减少存储压力的策略，具体包括：提供 profile 捕获范围（Profile Capture-Range）能力，通过修改`verl/verl/utils/profiler/nvtx_profile.py`文件中的`start`和`stop`函数，实现对`torch.cuda.profiler.start()`和`torch.cuda.profiler.stop()`的条件性调用；特定步骤（Specific Step） profile ，通过捕获范围能力和`discrete`参数为单个 worker 的特定训练步骤生成独立的nsys文件；特定rank（Specific Rank） profile ，选择性地对特定GPU或 worker 进行性能 profile ；以及离散化（Discrete） profile ，通过设置`discrete=True`使每个nsys文件仅捕获一个独立的子阶段（如Rollout、reference模型 log_prob 计算、Actor模型训练和奖励计算等），而不是整个复杂的RL步骤，从而实现更细粒度的性能分析和更小的文件体积，为在分布式RL训练中利用nsys工具进行高效、有选择性性能 profile 提供了完整的技术方案。

![](https://files.mdnice.com/user/59/280f7a02-80ca-40cc-b581-fde409d70e2c.png)

根据这个 Slides 的内容，我可以总结如下：

这个Slides讲了一下如何在`verl`环境中配置NVIDIA Nsight Systems (nsys)进行性能Profile，核心在于通过精细化参数控制 profile 的范围和粒度。它首先指出需要配置 profile 的" rank "（ranks）和"步骤"（steps），并决定是否采用"离散化"（discrete） profile 模式。具体的Nsight profile 配置包括：`PROFILE_STEPS`用于指定需要进行 profile 的训练步骤（例如`"[1,2,5]"`），`PROFILE_RANKS_ALL`作为一个布尔值，控制是否 profile 所有 worker 的 rank ，而`PROFILE_RANKS`则允许指定具体的 rank （例如`[0,4,8,12]`）进行 profile 。`DISCRETE`参数（可设置为`True`或`False`）决定了 profile 文件的生成方式：当`DISCRETE=False`时，所有 worker 的 profile 数据将合并到一个文件中；而当`DISCRETE=True`时，每个 worker 会生成独立的 profile 文件，这有助于在分布式环境中进行更细致的分析。这些配置参数随后被应用于强化学习流水线中的不同组件，例如`actor_rollout_ref.profiler`、`critic.profiler`和`trainer`，通过设置其`ranks`、`all_ranks`、`discrete`和`profile_steps`属性来精确控制每个阶段的 profile 行为。此外， Slides 还提供了启用nsys profile的指令链接，并明确指出nsys生成的 profile 文件默认保存在每个节点的`/tmp/ray/session_latest/logs/nsight/`路径下，为开发者提供了完整的配置和文件查找指南。

![](https://files.mdnice.com/user/59/9baaea7f-ef69-44c0-8c2f-dd9ba3d3916f.png)

这张Slides展示了一下按照之前的设置在VerL中使用Nsight Systems进行Profile得到的可视化结果，我们可以清晰的看到整个流程里最耗时的部分。另外如果设置了discrete=True，那么我们需要用Nsight System File下面的New multi-report view把各个阶段的profile文件拼起来成为一个完整的Step。


![](https://files.mdnice.com/user/59/141b1c62-a935-423f-b0da-42acf20fbac5.png)

根据这个幻灯片的内容，我可以总结如下：

这里展示了一下VerL中生成的nsys profile文件的命名约定，首先通过配置`PROFILE_STEPS`（例如`"[2, 6, 10]"`）来指定要profile的step，`PROFILE_RANKS_ALL=False`和`PROFILE_RANKS=[0,4]`来选择性地 profile 特定worker的rank，以及`DISCRETE=True`来启用离散化 profile 模式。当`DISCRETE=True`时，nsys会为每个工 worker 的每个特定训练步骤生成独立的性能文件，其命名格式为`worker_process_{pid}.nsys-rep`。Slides进一步解释说，文件名中的`pid`（进程ID）对应于特定的rank，而`{1-4}`数字后缀表示在特定`step 2`内，针对不同角色（如`rollout`、`log_prob`、`reference`、`actor training`）生成的独立profile文件，从而实现了对强化学习流水线中各个细分阶段和角色的精细化性能追踪，为分布式RL训练的性能分析和优化提供了系统化的文件管理方案。

![](https://files.mdnice.com/user/59/6c570489-394c-4315-bfeb-5536f951bf9a.png)

![](https://files.mdnice.com/user/59/051d9a79-099e-4300-a861-0f5353ec1a88.png)

这里指出RL训练过程复杂，包含Rollout、策略更新和奖励计算等多个模块，且各阶段特性迥异。普遍存在的挑战包括：需要调优大量参数以达到最佳性能，以及开发者必须同时熟悉训练和推理框架。在Policy模型训练阶段，具体问题表现为内存使用效率低下、并行设置不合理、计算气泡（即GPU空闲时间）以及因数据填充导致的无意义计算。

而在Rollout阶段，主要瓶颈是长尾任务导致大多数GPU节点出现大量空闲时间，以及整体吞吐量未能得到充分优化。这些问题共同构成了RL系统性能优化的关键挑战。

![](https://files.mdnice.com/user/59/ac8153f7-115f-40ea-b7eb-57f88334bed1.png)

这张Slides讲了一下在VerL Megatron-LM Backend下优化Actor模型训练的优化策略。核心优化点包括：首先是 **序列打包（Sequence packing）**，通过将多个短序列拼接成一个"打包"序列，有效避免了因填充（padding）导致的计算资源浪费；对于FSDP后端，这需要设置`actor_rollout_ref.model.use_remove_padding=True`，而在Megatron中，序列打包是默认行为。

其次是**启用动态批处理大小（Dynamic batch size）**，这有助于平衡不同微批次间的计算负载和时间消耗，并通过动态调整批次大小来避免长序列导致的显存溢出（OOM），强烈推荐启用，其配置涉及将`use_dynamic_bsz`设置为`True`，并应用于`actor_rollout_ref.actor`、`actor_rollout_ref.ref.log_prob`和`actor_rollout_ref.rollout.log_prob`等组件。

最后，需要合理设置 **`max_token_len_per_gpu`** 参数，它定义了每个GPU在正向和反向传播中处理的最大token数量；建议尽可能增大`ppo_max_token_len_per_gpu`，同时指出前向传播专用的参数可以设置得更大，例如设置为`(max_prompt_length + max_response_length) * 10`，因为推理的显存占用少，具体的配置示例展示了如何为Actor、Reference模型的 log-prob 计算以及Rollout阶段的 log-prob 计算设置这些 token 长度限制。

![](https://files.mdnice.com/user/59/86bc45ae-a619-437f-807f-90df36b1833d.png)

这张Slides在Megatron-LM框架下优化Actor模型训练的 **添加fuse kernel** 策略，通过设置`actor_rollout_ref.model.use_fused_kernels=True`，可以有效减少约3-10GB的峰值内存使用。这个fuse kernel是和corss-entropy相关的优化。

其次，Slides提供了 **推荐的并行设置** ，指出 Nemo Benchmark 可以作为在Actor模型训练期间调整Megatron-LM参数的参考。接着一个详细的表格展示了针对不同模型（如Qwen3和Llama3）及其规模（如30B、235B、70B、8B）在`hopper`系统上的具体并行配置，包括使用的GPU数量（`num_gpus`）、序列长度（`seq_len`）、以及各种并行策略的尺寸：张量并行（`TP_size`）、流水线并行（`PP_size`）、上下文并行（`CP_size`）、专家并行（`EP_size`）和虚拟流水线并行（`VP_size`），同时还列出了微批次大小（`mbs`）和全局批次大小（`gbs`），例如，Qwen3 235B模型在256个GPU上，采用`TP_size=2`, `PP_size=8`, `EP_size=32`, `VP_size=4`的配置，实现了2048的全局批次大小。

然后我们可以基于这个基础的并行设置用NVIDIA在Megatron-LM中提供的内存计算器去进一步判断和调整并行配置。

![](https://files.mdnice.com/user/59/14e8f2cd-edf2-435a-ab35-b845972783c4.png)

还有两种优化策略：

首先是**Offload（卸载）**，我们可以通过将模型参数、优化器状态和梯度从GPU内存卸载到CPU内存来节省显存，推荐在训练大型模型并且资源受限时启用，这个优化会增加额外的训练时间；如果GPU内存充足，则应将`offload`设置为`False`。

其次是 **Recompute（重计算）**，这项技术通过在反向传播时重新计算前向传播的某些中间激活值来减少显存占用，但同样会引入额外的计算时间，因此需要谨慎启用。

![](https://files.mdnice.com/user/59/d37a7131-45e6-4fe2-a590-4fcd7aa6dfd9.png)

这张Slides给出了一些提升Rollout性能的优化策略。

- 首先是**提高`gpu_memory_utilization`**，通过增加GPU内存利用率，对于`vllm`等框架，这可以预分配更多的GPU缓存，从而获得更大的KV缓存空间；如果启用了`offload=True`，则可以进一步设置更高的`gpu_memory_utilization`。

- 其次是**调整`max_num_batched_tokens`或`max_num_seqs`**，当GPU内存利用率较低时，增大这两个参数可以有效扩大解码阶段的有效批处理大小，进而支持每个批次处理更多的并发请求。

- 第三是**采用更小的`tensor_parallel_size`**，这通常有助于减少通信开销或优化特定场景下的计算效率。

- 最后是**启用分块预填充（Chunked prefill）**，通过这种方式可以显著提高Rollout生成的整体吞吐量。

![](https://files.mdnice.com/user/59/c9fe8cf3-a672-4425-b786-4431e043823a.png)

另外一点就是我们应该打开cuda graph的支持，VerL中对于vLLM默认是没有开启cuda graph的，对于SGLang是默认开启的。

Slides上展示的是qwen2-7b然后response length 512的时候做的测试，然后上面的图是没有开启cuda graph的时候，kernel的空隙很大。在generate_sequence阶段端到端加速17%。另外右下角的图展示qwen3-30b，2048的prompt length，8192的response length开启cuda graph会加速1倍。

开启cuda-graph的时候可能会碰到OOM的问题，可以去调整cuda graph capture size降低内存占用。

![](https://files.mdnice.com/user/59/4d7222c6-0392-49a9-8ee0-70463e778a78.png)

该实验展示了强化学习Rollout阶段的长尾效应问题，通过对比同一训练步骤中不同GPU节点（rank0与rank4）的Rollout执行耗时分布差异，揭示计算资源利用率不均衡现象。

我们可以看到图中绿色的generate_sequence的部分，rank4的部分比rank0的部分时间要少很多，后面的红框部分都是rank4在等rank0的时间。

![](https://files.mdnice.com/user/59/745644bb-eac8-4ffa-9c8c-92fff35e66da.png)

在强化学习（RL）的Rollout阶段中，同步（sync）Rollout面临着"长尾问题"及其导致的性能瓶颈，然后这里提出了一种名为"无缝异步DAPO（Async DAPO recipe）"的解决方案。同步Rollout的主要问题包括：需要耗费大量时间等待所有计算单元（ranks）完成Rollout，导致GPU利用率低下，并最终拖慢整个批次训练过程。为解决这些问题，Slides介绍的无缝异步DAPO（其代码链接为`verl/pull/2799`）基于VeRL异步Rollout机制，并具备多项关键特性：首先是**无缝Rollout**，通过非阻塞的并发请求处理实现最大化的吞吐量；其次是**早期停止机制**，当达到目标数量的提示（prompts）完成并经过验证（考虑奖励方差）后，系统能够智能地提前终止Rollout；再者是**动态负载均衡**，采用全局负载均衡器进行实时服务器分配，优化资源利用；最后是**无缝奖励计算与提示过滤**，每个提示的响应一旦完成，即可立即进行奖励计算和过滤，无需等待其他提示的完成，从而显著提升了Rollout阶段的效率和整体训练速度，为分布式强化学习训练中的资源利用率优化提供了创新的异步处理架构。

![](https://files.mdnice.com/user/59/00eb4e4a-5788-450f-ae67-ded9b31325cc.png)

这个优化的性能提升会在20%-40%之间。


![](https://files.mdnice.com/user/59/2c5ec1d0-1f77-44c3-a46f-fa7cc09fa667.png)

![](https://files.mdnice.com/user/59/465a68c3-b796-4608-bb49-8a49f6f395f3.png)

这里展示了一下强化学习（RL）参数调优的实践过程，以Qwen2.5 7B GRPO模型为例。首先，将Qwen2.5 7B模型作为示例，并以Nemo SFT（监督微调）配置作为参考来设定并行参数。其次，Slides提供了一个基于Nemotron 8B的初步并行设置草案，包括`num_gpus=8`、`mbs=2`（微批次大小）、`tp=2`（张量并行）、`pp=1`（流水线并行）和`seq_len=4096`（序列长度），并强调这些并行设置需要根据实际序列长度进行进一步估算。

接着，Slides明确了关键的长度和批次大小参数：`max_prompt_length`设置为2048，`max_response_length`设置为8192，`ppo_micro_batch_size_per_gpu`设置为2，Policy训练设置`tp=2`、`pp=1`，而Rollout本身的张量并行（`rollout tp`）设置为1。

最后，Slides指出Rollout的张量并行（`tp`）应尽可能小，并且如果使用Megatron-LM后端，Ref模型的对数概率（reference logprob）也将在Megatron中计算，因此建议Ref模型与Actor模型使用相同的并行设置，以确保一致性，为实际RL模型训练的参数配置提供了系统化的指导方案。

![](https://files.mdnice.com/user/59/cb4d5520-7fb5-4ec9-ac5e-7f9536887a67.png)


这张Slides建议使用Megatron内存估算器来验证给定序列长度下的配置。Slieds展示了上一页的配置对应的显存占用：模型为Qwen2.5 7B，使用8个GPU，微批次大小为2，序列长度为10240，并启用了分布式优化器且未进行重计算。并行设置包括张量并行（TP）为2，流水线并行（PP）、专家并行（EP）和上下文并行（CP）均为1。在此配置下，内存使用明细显示，每个GPU的总内存占用为83.27 GB，其中参数占用3.81 GB，激活占用27.57 GB和51.36 GB，权重优化器状态占用31.92 GB。此外，幻灯片提出了多项优化建议：启用融合核（fused kernel）以减少峰值内存，建议将`dynamic bs`设置为`False`进行观察，并指出在内存允许的情况下，将`offload`和`recompute`设置为`False`可以节省时间。最后，Slides提供了一系列性能指标，包括MFU为30.3，总步骤时间为416.93秒，序列生成时间为154.22秒，reshard时间为1.77秒，rollout时间为190.44秒，Reward为3.19秒，旧对数概率计算时间为49.14秒，优势计算时间为0.39秒，Actor更新时间为172.19秒，以及总吞吐量为2437.09 tokens/s，为RL模型训练的性能评估和优化提供了量化的技术依据。

![](https://files.mdnice.com/user/59/81d3a699-f99c-490a-b139-b09232fbdc0e.png)

如果打开dynamic batch并设置合适的max_token_per_gpu之后的性能，可以看到MFU提升非常大达到了45.96%。

![](https://files.mdnice.com/user/59/c60105d5-c209-40a4-b367-bfa76a1af20b.png)

这里展示了一下基于VerL的Qwen 3 235B MoE模型的最佳参数组以及达到的最高MFU，大家可以参考。

![](https://files.mdnice.com/user/59/ca0293c7-17f6-43ac-8555-1c7b0acd7a66.png)

![](https://files.mdnice.com/user/59/754b6ae8-19f6-4804-b51d-b2a24fecfa2c.png)

这里打错了个单词。这个演讲系统性地回顾了强化学习（RL）流水线优化的关键策略和实践。它首先强调了使用Nsight Systems对RL框架进行性能Profile的重要性，具体包括如何在Ray Actor中添加Nsight参数，以及如何解释从RL框架中获取的Nsight Porfile文件。其次，Slides提供了优化训练/生成参数的详细技巧，这又细分为两个主要方面：Actor训练优化，涵盖了推荐启用的动态批处理（dynamic batching）和序列打包（sequence packing），以及对卸载（offload）、重计算（recompute）和融合内核（fused kernel）的策略性使用；Rollout优化，则侧重于提高GPU内存利用率（gpu_memory_utilization）、利用CUDA Graph以及解决Rollout阶段的长尾问题。最后，Slides以Qwen 2.5 7B模型为例总结了RL参数调优的实践经验，并提及了Qwen3 235B MoE模型的SOTA设置，为实际应用中的性能调优提供了全面的指导。






