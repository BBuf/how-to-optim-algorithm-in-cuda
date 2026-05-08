> verl 放在 SGLang 目录里看，核心原因是 rollout。训练框架要高吞吐采样，推理框架也要接受权重同步、显存 offload 和 multi-turn agent 采样这些训练侧需求。

# 0x0. 前言

这篇和前两篇 SGLang 文章的连接点在 rollout：训练框架需要高吞吐推理后端，推理后端又必须接受训练框架的权重同步、显存 offload 和多轮 agent 采样。

# 0x1. 资料和代码落点

相关资料和代码：

- verl 仓库：`verl/workers/rollout/sglang_rollout/sglang_rollout.py`，SGLang HTTP server adapter、显存 release/resume、bucket 权重更新。
- verl 仓库：`verl/experimental/agent_loop/tool_agent_loop.py`，Agentic RL 多轮工具调用状态机。
- verl 单 Controller 抽象：`verl/single_controller/base/decorator.py`，定义多种 dispatch/collect 行为。
- slime 资料：`https://lmsys.org/blog/2025-07-09-slime/` 和 `THUDM/slime`，可作为 SGLang-native RL 系统的对照。

LMSYS 的 slime blog 适合放在这里对照看。verl 和 slime 不是同一个项目，但它们都在处理同一个系统问题：RL 训练框架要不断生成 rollout，rollout 又需要一个高吞吐推理引擎；训练权重更新以后，推理引擎必须尽快同步新参数。slime 的系统图很直观：

<img src="https://files.mdnice.com/user/59/e62c3bfc-5f68-4989-a912-c8495837396d.png" referrerpolicy="no-referrer" />

slime 的设计更偏 SGLang-native：rollout server group、router、weight sync、partial rollout 都围绕 SGLang 展开。verl 则更像通用 RL 编排框架，把 FSDP/Megatron、vLLM/SGLang、Ray worker 放到统一 controller 下。读这篇 slides 时可以把 slime 当作对照组：如果只服务 SGLang，很多路径可以写得更紧；如果要兼容多种训练和推理后端，就需要 verl 这种更抽象的 dispatch/collect、worker group 和 backend adapter。

LMSYS 另一篇 deterministic inference blog 对 Agentic RL 也有参考价值。它讨论的问题是：同一个 prompt、同一组采样参数，在分布式推理和多轮 rollout 下能不能复现。RL 训练里 reward 波动本来就大，如果 rollout engine 还因为 batch 形状、kernel 路径或调度顺序产生额外随机性，排查会很痛苦。slime/SGLang 的确定性路线可以理解为给训练系统加一条“可复现实验”的保险丝：

<img src="https://files.mdnice.com/user/59/27b393ab-80dc-41da-931c-b164acc24a58.png" referrerpolicy="no-referrer" />

# 0x2. Slides 逐页解读

### Slide 1：verl：面向 Agentic Tasks 的大规模 LLM RL 框架

<img src="https://files.mdnice.com/user/59/0fe522e1-9759-43b4-8938-44d127f9ee82.png" referrerpolicy="no-referrer" />

标题里有两个关键词：Large-Scale 和 Agentic Tasks。前者说明 verl 不是单卡 RL demo，而是要把 actor、critic、reference、reward、rollout engine 都放到分布式资源上；后者说明 rollout 不再只是一次 `generate`，而是会穿插工具调用、环境执行和多轮对话。

放在 SGLang 目录里看，最重要的连接点是 rollout backend。SGLang 负责高吞吐采样，verl 负责把采样结果变成训练数据，并在每轮训练后同步新权重、释放/恢复推理侧显存。后面所有 Hybrid Controller、3D-HybridEngine、AgentLoop 都围绕这个闭环展开。

### Slide 2：项目背景

<img src="https://files.mdnice.com/user/59/0a778b33-4eea-4d4f-9ff0-1519dae69c81.png" referrerpolicy="no-referrer" />

这一页介绍 ByteDance Seed Team。对技术正文来说，它主要交代背景：verl 来自一个需要长期做 RLHF、reasoning、agent、tool-use 的团队，所以设计目标不是把某个算法跑通一次，而是让不同 RL 算法、训练后端和推理后端可以持续接入。

这也解释了为什么 verl 的抽象层比较厚。PPO、GRPO、RLOO 这类算法可以共享 controller 和 worker group；FSDP、Megatron、vLLM、SGLang 这些后端则通过 adapter 接入。系统复杂度被压在 dispatch/collect、worker group、rollout adapter 这些层里。

### Slide 3：Seed 团队项目背景

<img src="https://files.mdnice.com/user/59/4834c013-8b60-492a-b638-2ee25807c74f.png" referrerpolicy="no-referrer" />

这一页继续给 Seed 团队入口。虽然不是技术图，但它把分享的上下文补齐了：verl 服务的是一组持续演进的模型和任务，不是一次性的论文复现实验。大模型 post-training 里，算法 recipe、数据、评测、rollout 吞吐、权重同步都会频繁变化。

因此后面代码里会看到很多“看起来不像算法”的逻辑，例如显存 release/resume、server mode、sticky session、DataProto padding。这些不是外围杂项，而是让 RL 训练能长期跑在真实集群里的基础设施。

### Slide 4：SFT 和 RL 的边界

<img src="https://files.mdnice.com/user/59/0687d159-14be-4f2f-a24b-f678991b21e1.png" referrerpolicy="no-referrer" />

这页把 SFT 和 RL 的差别压得很清楚。SFT 是从标注样本学习，通常“一个模型 + 一份静态数据集”就能描述主流程；RL 是基于 reward 优化，训练数据由当前 policy 生成，policy 每更新一次，下一轮 rollout 分布也会变。

所以 RL 框架必须把数据生产和模型更新放在同一个闭环里：actor rollout 出 response，reference/actor 重新算 logprob，reward 或 verifier 给分，最后再根据 advantage 更新 actor。SGLang 在这个闭环里承担的就是高吞吐数据生产端。

### Slide 5：为什么 LLM RL 需要系统框架

<img src="https://files.mdnice.com/user/59/156b6ef3-b8b2-4ecf-adbb-6460d57064f6.png" referrerpolicy="no-referrer" />

这页的时间线从 2023 的 human alignment，走到 2024 的 reasoning，再到 2025 的 agents。它想表达的是 RL 任务形态在变：RLHF 主要优化偏好，reasoning 开始引入可验证答案，agentic LLM 则要把工具、桌面操作、代码助手、游戏环境纳入训练。

系统压力也随之变化。普通 reward model 评分还相对规整，agentic rollout 会出现长尾环境耗时、多轮工具返回、样本提前终止等情况。LLM RL 需要同时处理 actor、reference、reward/verifier、rollout engine、advantage、logprob，以及训练端和推理端之间的权重更新。

### Slide 6：RL 数据流比监督学习复杂得多

<img src="https://files.mdnice.com/user/59/bcf1ce0e-56eb-4224-a12b-963d2088056d.png" referrerpolicy="no-referrer" />

这页的数据流图是 verl 入门的关键。slide 上写 RL 可以被建模成 complex dataflow graph，里面有 multiple models、multiple stages、multiple workloads。multiple models 包括 actor、critic、reference、reward model；multiple stages 包括 generation、experience preparation、training；multiple workloads 则分别对应 generation、inference、training。

一次 rollout 产生的不只是 response 文本，还会产生 token ids、attention mask、response mask、old logprobs、rewards、values 等训练张量。后面训练时又要按 micro-batch、sequence length、DP rank 做切分。只要 mask 或 logprob 对不齐，loss 不一定会报错，但训练信号会偏。

### Slide 7：大规模分布式数据流

<img src="https://files.mdnice.com/user/59/9b91b06c-3db6-4960-9b64-ba761d2263f8.png" referrerpolicy="no-referrer" />

这页强调 LLM RL 的每个 dataflow operator 本身都是大规模分布式 workload。训练侧要用 Megatron-LM/FSDP 这类 ND parallelism，应对 Qwen 235B、DeepSeek 671B 这样的模型规模；序列长度也从 8k 往 1M 走，单个 batch 的 shape 已经足够复杂。

分布式后，数据流不再是本地函数调用，而是跨 Ray actor、placement group、device mesh 和通信组的调度。verl 的 Controller 要保证“算法上下一步该调谁”和“系统上数据该切给哪个 worker group”同时成立。

### Slide 8：RL 数据流里的依赖和资源限制

<img src="https://files.mdnice.com/user/59/dbb12ae4-cabd-46af-9d99-58a1e6832fc9.png" referrerpolicy="no-referrer" />

这页题目里有两个词：Data Dependencies 和 Resource Limitations。数据依赖是指 generation 之后才能算 reward/logprob/value，advantage 出来以后才能训练；资源限制是指 actor、critic、reference、rollout engine 经常要抢同一批 GPU 或同一组显存预算。

这正是 colocate、release/resume、weight sync 出现的原因。rollout 阶段训练相关模块可以 offload，训练阶段 rollout engine 可以释放 KV/weights/graph；两边切换时再恢复和同步权重。verl 的 Hybrid Controller 把这些阶段依赖写成可读的 Python 控制流，底层 worker group 再各自处理并行执行。

### Slide 9：社区和采用情况

<img src="https://files.mdnice.com/user/59/d5642148-3456-4983-aeb0-a84cc5e0b2a2.png" referrerpolicy="no-referrer" />

社区页列了 10k+ stars、1k+ forks、1.1k+ PRs、250+ contributors，以及 TinyZero、SimpleRL-Zoo、rllm、SkyThought、OpenManus-RL 等项目。这说明 verl 已经不是论文附属代码，而是被很多 RL 项目当作底层系统来用。

社区采用度会反过来逼迫接口更通用：有人要 FSDP，有人要 Megatron，有人 rollout 用 vLLM，有人 rollout 用 SGLang；有人只做 GRPO，有人要 tool-use 和 agentic loop。verl 的 adapter 层就是为了让这些变化不直接污染算法主流程。

### Slide 10：verl 的功能面

<img src="https://files.mdnice.com/user/59/a1c4d479-2790-4460-adeb-a7600be8329a.png" referrerpolicy="no-referrer" />

这页列了 verl 的 highlight features：Hybrid Controller 让 PPO/GRPO 这类 RL dataflow 可以用少量代码表达；3D-HybridEngine 负责训练和 generation 阶段的 actor resharding；modular APIs 复用 FSDP、Megatron-LM、vLLM、SGLang；device mapping 支持不同 GPU placement；大 MoE 也在支持范围内。

对 SGLang 用户来说，重点是“Seamless integration of existing LLM infra”。SGLang 不只是被 `requests.post("/generate")` 调一下，而是要参与 server group 管理、显存 release/resume、bucket 权重更新、PD 角色划分和 AgentLoop server mode。

### Slide 11：Hybrid Controller：把控制流留在 Python

<img src="https://files.mdnice.com/user/59/dc3f2492-e100-4190-8cba-bb067e79600a.png" referrerpolicy="no-referrer" />

这页借 Pathways 的图解释两种分布式编程范式。左边 Single-Controller(MPMD) 是一个中心 controller 管所有 worker，不同 worker 可以执行不同程序；图里的长条 step k 代表全局调度，下面每条 host/dev timeline 上有 send/recv、计算和等待。右边 Multi-Controller(SPMD) 是每个 worker 自己带 controller，大家跑同一份程序但吃不同数据；图里 step k、step k+1 在不同 device 上同步推进，读写发生在各自 controller 里。

verl 选择 Hybrid Controller，是把这两者拼起来：算法层仍然像 Single-Controller 一样在 Python 里写“先 generation，再 reward/logprob/value，再 actor update”；具体算子层可以像 Multi-Controller 一样在 worker group 内部并行。这样 RL 代码能保持顺序可读，底层又能挂 FSDP、Megatron、vLLM/SGLang 这类多进程后端。

### Slide 12：单 Controller 驱动多 Worker

<img src="https://files.mdnice.com/user/59/cd0badbd-f787-4c1e-b1c3-cee6bdf43364.png" referrerpolicy="no-referrer" />

这页把 Hybrid Controller 画成 `Single Controller + N x Multi-Controller`。左侧 single controller 收 prompts，触发 `Gen`，拿到 prompts+responses 后再依次调 ref logprob、actor logprob、values、reward，最后汇成 experiences，送给 actor update 和 critic update。右侧两个三维 GPU mesh 表示 worker group 内部的并行结构：zero data parallel、pipeline parallel、model parallel 都可以存在，controller 不需要知道每个 kernel 怎么排布。

这也是 verl 和只写一个分布式训练脚本的差别。controller 处理的是 inter-operator 数据流，比如 generation 的输出要喂给 reward 和 logprob；multi-controller 处理的是 intra-operator 并行，比如 actor update 内部怎么做 DP/TP/PP。源码里的 worker group、dispatch/collect 和 backend adapter 正是在这个层次上分工。

### Slide 13：Dispatch/Collect 的数据分发语义

<img src="https://files.mdnice.com/user/59/199975db-56fa-4425-b2ad-aebd6c5d8706.png" referrerpolicy="no-referrer" />

这页左边把 RL 流程拆成三个 stage：Generation stage 里 prompts 进入 actor 生成 responses；Experience Preparation stage 会对 prompts & responses 分别计算 reference log prob、actor log prob、values 和 reward；Training stage 再用 buffer 里的 experiences 更新 actor/critic。右边的代码正好对应这个顺序：`actor.generate_sequences(prompts)`，然后 `reward.compute_reward`、`reference.compute_log_prob`、`critic.compute_values`、`compute_advantage(batch, "gae")`，最后 `critic.update_critic` 和 `actor.update_actor`。

Dispatch/Collect 负责让这段看起来像单机 Python 的流程能跑在多 worker 上。prompt 数据通常按 DP 切给 rollout worker；logprob/value/reward 可能走不同 worker group；训练更新又要按 actor/critic 自己的 mesh 聚合回来。verl 的源码里 `Dispatch.DP_COMPUTE_PROTO` 这类策略做的就是这件事：分发前切 batch，必要时 padding，collect 后再 concat 回 DataProto。

### Slide 14：FSDP、Megatron、vLLM、SGLang 后端

<img src="https://files.mdnice.com/user/59/8a5c0533-16eb-4cce-98d0-e9e074f7ff95.png" referrerpolicy="no-referrer" />

这页把 multi-controller 能接的后端摊开：并行算法包括 DP、TP、PP、context/sequence parallel；训练后端包括 FSDP、FSDP2、Megatron、torchtitan；推理后端包括 vLLM 和 SGLang；kernel 侧还能用 FlashAttention、torch compile、Liger Kernel。

这说明 verl 的“multi-controller”不是抽象口号。actor update 可以在 Megatron mesh 里跑，rollout 可以在 SGLang server group 里跑，reward/logprob 又可能是另一套 worker group。controller 只需要描述数据依赖，具体 operator 内部怎么并行交给各自后端。

### Slide 15：3D-HybridEngine 和 colocate

<img src="https://files.mdnice.com/user/59/9c98a861-0533-4b50-80cd-f76112c33110.png" referrerpolicy="no-referrer" />

这页区分了 colocate strategy 和 split strategy。colocate 在训练和 generation 阶段使用同一组 GPU 分组，split 则让两阶段用不同分组。下面的例子里，训练是 `TP=4, DP=2, PP=1`，generation 是 `TP=2, DP=4, PP=1`；从 Train 到 Gen 的箭头表示同一批 GPU 在阶段切换时要重组并同步权重，图中 `All-Gather within Micro-DP group` 对应权重从训练切片形态变成推理需要的完整/重分片形态。

slide 下面两条小字需要单独看：3D-HybridEngine 下 colocate 能减少 training/generation 切换时的通信开销；offloading & reloading 让 GPU memory 可以被充分利用。落到 SGLang，就是 rollout 结束后 release KV/weights/graph，训练结束后 resume 并同步新权重。没有 release/resume，训练 peak 会和推理常驻显存叠在一起；没有权重同步，恢复出来的 rollout server 又还是旧策略。

### Slide 16：verl 的编程方式

<img src="https://files.mdnice.com/user/59/a4dbb7c0-7504-4122-8410-25852913e2ac.png" referrerpolicy="no-referrer" />

这一页展示的是 verl 的编程接口。slide 小字写得很直接：single-controller 里的每次调用，比如 `critic.compute_values`、`actor.update_actor`，本质上都是发给 multi-controller worker group 的 RPC；`register` decorator 管理 dataflow 节点之间的分布式数据传递。

这就是后面代码里 `Dispatch` 和 `Collect` 的来源。算法作者看到的是普通 Python：生成、算 reward/logprob/value、算 advantage、更新 actor/critic；系统层实际做的是 batch 切分、padding、跨 worker group 调用、结果 concat。这个接口设计决定了 verl 能同时服务 PPO、GRPO、RLOO、ReMax、PRIME、DAPO 等算法。

### Slide 17：Agentic RL 章节过渡

<img src="https://files.mdnice.com/user/59/13866e10-62e7-471d-8407-eb8fe33951fb.png" referrerpolicy="no-referrer" />

这一页是章节过渡，开始进入 Agentic RL。前面讲的是普通 RL 数据流和 Hybrid Controller，后面要处理的是更难调度的 rollout：模型会调用工具、等待环境返回，把 observation 写回上下文，再继续生成。

这一步对 SGLang 也有影响。普通 rollout 可以把一批 prompt 发给后端等 response；Agentic rollout 需要保留每条 trajectory 的会话状态、KV cache、工具调用状态和 request id。推理后端如果不支持异步 server mode 和 sticky session，多轮交互会不断重建上下文。

### Slide 18：什么是 Agent

<img src="https://files.mdnice.com/user/59/1a51f878-78f8-4662-9cbd-8f051f9a167c.png" referrerpolicy="no-referrer" />

这页给出 Agent 定义：software systems that use AI to reasoning, planning, memory and autonomy。slide 下面列了三项能力：tool calling 让 LLM 按需选择工具；memory 让 agent 使用历史步骤信息；planning 让模型形成并执行多步计划。

对 RL 框架来说，Agent RL 训练的是复杂动态环境里的 decision making。rollout 不再是固定长度的一次 `generate`，而是 `message -> action/tool call -> observation -> next message` 的循环。训练数据也从单条 response 变成 multi-turn trajectory，里面还要区分模型生成 token 和环境返回 token。

### Slide 19：ReTool：工具调用型训练

<img src="https://files.mdnice.com/user/59/38bbf1b5-7ae4-4f02-8095-99d1366ddce4.png" referrerpolicy="no-referrer" />

这页给的例子是 ReTool：training LLM to write python code to solve math problem。模型不是直接吐最终答案，而是先生成 Python 代码，让环境执行，再根据代码输出继续推理或回答。对数学题来说，代码执行结果天然可以作为 verifier 的一部分。

ReTool 把普通 RLHF 里的 response 扩展成 action/observation 轨迹：模型生成代码是 action，sandbox 返回结果是 observation，最终答案再被 reward/verifier 检查。代码侧对应 `ToolAgentLoop`：生成阶段解析 tool calls，工具阶段并发执行，工具响应再通过 chat template 追加回 `prompt_ids`。

### Slide 20：Agentic RL 的同步 rollout 问题

<img src="https://files.mdnice.com/user/59/87ea2976-4cc8-4337-867e-fdbf01e1853e.png" referrerpolicy="no-referrer" />

这页用三条 timeline 对比 rollout 编排。最上面 synchronous rollout 里，`Initialize Runtime`、`LLM Gen`、`Env Exec`、下一轮 `LLM Gen` 基本串行，最后才到 `Reward Calculation`。中间 asynchronous rollout 允许不同 trajectory 交错执行，某条 trajectory 结束后可以启动新 trajectory，但 reward 仍然靠后。最下面的 async rollout + 3-stage producer-consumer pipeline 把 runtime 初始化、LLM 生成、环境执行、reward calculation 拆成流水线，让多条 trajectory 可以同时处在不同阶段。

右侧列的三个 drawback 对应上图：batch generate 和 environment execution 串行；rollout 和 reward calculation 串行；rollout 和 training 串行。Agentic RL 里每个样本的工具耗时和轮数差异很大，如果所有样本都按同步批次等待，慢样本会拖住一整批训练数据，推理和训练两边都会出现空窗。

### Slide 21：AgentLoop 状态机

<img src="https://files.mdnice.com/user/59/3674cef1-8c90-44b1-a72c-1b85ca0bfc7f.png" referrerpolicy="no-referrer" />

这页给出 AgentLoop 的接口定义：给一个 user prompt，执行用户定义的 loop，输出 multi-turn chat history 作为 trajectory。右侧列了几类环境：online web search、MCP tools、code sandbox、virtual machine、Android emulator。也就是说 rollout 不再只是 `prompt -> response`，而是模型不断生成 action，环境返回 observation，再把观察写回上下文。

下面代码里的 `AgentLoopBase(ABC)` 暴露 `async def run(self, messages, sampling_params) -> AgentLoopOutput`。这个异步接口是关键，因为工具调用和环境执行天然有等待时间。实现上仍然可以有状态机，例如 `PENDING -> GENERATING -> PROCESSING_TOOLS -> TERMINATED`，但对上层 trainer 来说，它只需要拿到最终 trajectory 和中间 token/mask/reward 信息。

### Slide 22：AgentLoop 的 server mode

<img src="https://files.mdnice.com/user/59/3c1b21f8-eaf5-499b-811f-8d916c40e7e2.png" referrerpolicy="no-referrer" />

这页左图把 server mode 的数据流画得很细。`PPOTrainer` 调 `generate_sequences`，进入 `AgentLoop Manager`，Manager 再把 prompts 分发到多个 `AgentLoopWorker`。每个 worker 内部有 `AgentLoop` 和 `AsyncLLMServer Manager`，真正的模型生成通过下面的 `AsyncSglangServer/AsyncvLLMServer` 发到一组 model runner。底部标了两个 vLLM group，每组 tensor_parallel_size=4，外层还有 FSDP group world_size=8。

右侧三条 highlight 对应这个结构：server mode 可以接 vLLM/SGLang AsyncLLM engine；parallel running 用 asyncio 同时跑多个 prompts；load balance and sticky session 用来提升 KV cache 利用率。sticky session 的意思是同一条多轮 agent trajectory 尽量留在同一个后端会话上，这样前面生成的 KV cache 不需要每轮重新构建。

### Slide 23：ReTool with AgentLoop

<img src="https://files.mdnice.com/user/59/c6bd3548-fbed-4eb7-b74b-f4818ddc2c10.png" referrerpolicy="no-referrer" />

这页把 ReTool 复现配置和训练曲线放在一起。Overview 里写的 base model 是 Qwen/Qwen2.5-32B-Instruct，SFT dataset 是 JoeYing/ReTool-SFT，RL dataset 是 ByteTsinghua-SIA/DAPO-Math-17k，val dataset 是 yentinglin/aime_2025，recipe 指向 `verl/recipe/retool`。底部阶段也很明确：stage 1 是 SFT，stage 2 是 GRPO。

三张曲线分别说明训练状态：左边 `train/loss` 在 SFT 阶段下降；中间 `val-score/aime_2025/acc/mean@30` 在 GRPO 阶段上升；右边 `val-aux/num_turns/mean` 也上升，说明模型在验证集上更频繁地进行多轮工具交互。放回 AgentLoop 语义里，就是模型生成工具调用，环境执行工具，把 observation 写回上下文，然后继续生成；训练前再把这些轨迹整理成 token、mask、logprob 和 reward 张量。

### Slide 24：ReTool 复现经验

<img src="https://files.mdnice.com/user/59/dbea2a89-9dbb-4629-8684-31b999c22e58.png" referrerpolicy="no-referrer" />

这页是 ReTool reproduction 的经验总结，slide 上写了两条 lesson，标题都指向 `token-in-token-out vs chat completion`。这里的冲突很实际：训练框架内部最稳定的是 token ids、attention mask、response mask 这些张量；但 agent/tool 生态常用 chat completion 语义，消息里有 role、tool call、tool response、multi-modal payload。

如果中间转换不严谨，问题会直接反映到训练信号上。比如工具返回 token 被误标成 response token，loss 会训练模型去“模仿环境输出”；chat template 少了 tool role，下一轮生成会把 observation 当成用户问题；token 边界错了，logprob、KL、reward mask 都会偏。ReTool 复现真正麻烦的地方就在这里：不是只把 GRPO 跑起来，而是让 token-level 训练张量和 chat-level agent 轨迹一一对齐。

### Slide 25：Roadmap：更大的 MoE 和更强推理后端

<img src="https://files.mdnice.com/user/59/1c2f87a3-fc72-4ca8-8ca5-62ab1125f571.png" referrerpolicy="no-referrer" />

这一页是 Q3/Q4 Roadmap 过渡。它没有技术细节，但位置很重要：前面讲完普通 RL 和 Agentic RL 后，roadmap 会回到大 MoE、partial rollout、async pipeline、server 化 rollout 这些工程问题。

更大的 MoE 模型会把前面的矛盾全部放大：参数更多，expert routing 更复杂，训练端 checkpoint/reshard 更重，rollout 端显存更紧。SGLang 后端的 DP attention、EP、MTP、memory saver、PD 和权重更新能力，都会影响 RL 系统的上限。

### Slide 26：大 MoE RL 训练更新

<img src="https://files.mdnice.com/user/59/d299f10a-532d-48bd-98c0-49aea8c20f59.png" referrerpolicy="no-referrer" />

这页讲 Trainer Updates，标题是 Scalable RL for large MoE models。slide 写到 verl 已经有 preview 支持 DeepSeek-V3-671B 这类巨大 MoE：训练侧基于 Megatron-Core GPTModel，例子里 DeepSeek 671B 用 96 张 H20，Qwen3 235B 用 32 张 H20；推理侧支持 multi-node inference；Hybrid 部分要做 Megatron-Core V0.12 和最新推理引擎之间的 parameter sharding manager。

这页最后一句 “Further Performance Optimization is required” 很诚实。大 MoE RL 的瓶颈不会只在算子里，训练权重如何同步到 rollout server、Megatron 分片如何变成推理端需要的分片、multi-node SGLang/vLLM 如何保持吞吐，都是系统问题。slime 这类 SGLang-native 系统可以作为对照：它把 SGLang server group、router、weight sync 和 partial rollout 都放到 RL 系统核心路径上。

### Slide 27：Partial rollout 和 async rollout

<img src="https://files.mdnice.com/user/59/dd64e373-0b24-43ae-8a0e-33d1acc9780b.png" referrerpolicy="no-referrer" />

Roadmap 这页列了四个方向：modular design、partial rollout & fully-async training pipeline、native vLLM/SGLang HTTP server、rollout performance optimizations(fp8)。第一项是把 FSDP2、Megatron 等 model engine 抽象得更可组合；第三项提到 slime，说明 SGLang/vLLM server 化 rollout 会继续靠近训练框架。

Partial rollout 的核心是不要让 rollout 只能以“完整 trajectory”为单位进入训练系统。Agentic 任务里，有的样本工具执行很慢，有的样本很快结束；如果必须等整批完成，长尾会拖住 trainer。partial rollout 要求 data buffer 能保存半截 trajectory、request 状态、KV/cache 对应关系和已生成 token，下轮继续补齐或先用已完成部分推进训练。

### Slide 28：更真实的 Agentic 任务

<img src="https://files.mdnice.com/user/59/09723001-3598-4ef3-8998-d8ada87aa22d.png" referrerpolicy="no-referrer" />

这一页列了更真实的 agentic task：Deep Research、Code/SWE-bench、Multi-modal GUI/browser 等。它们的共同点是环境变重了：Deep Research 要搜索和读网页，SWE-bench 要改代码和跑测试，GUI/browser 要处理截图、坐标、点击、输入和页面状态。

这些任务会把 rollout 后端推向“长期会话服务”。同一条 trajectory 可能跨很多轮生成，期间夹着工具执行和环境等待；如果每轮都把完整上下文重新 prefill，成本会很高。SGLang 这类后端要配合 sticky session、KV cache 复用、异步请求和多模态输入，才能支撑这类 Agentic RL。

### Slide 29：社区协作方向

<img src="https://files.mdnice.com/user/59/77311927-0b64-4cac-a3e2-b1eab98a4580.png" referrerpolicy="no-referrer" />

这一页是社区邀请页，给出 verl repository、联系人和社群入口。技术上可以把它理解成一个开放接口的提醒：Agentic RL 的任务和环境还在快速变化，框架需要持续吸收新算法、新工具协议和新推理后端。

对贡献者来说，入口并不只有算法。SGLang rollout adapter、AgentLoop、新 verifier、server mode、partial rollout buffer、MoE 权重同步都可以独立演进。verl 的价值正在于这些改动能进入同一条训练闭环，而不是散落成一堆实验脚本。

### Slide 30：总结

<img src="https://files.mdnice.com/user/59/fc135309-9983-4b77-b911-016dfc4e2240.png" referrerpolicy="no-referrer" />

总结页回到主题：verl 的价值不是某个 RL 算法，而是把复杂 RL 数据流、分布式 worker、rollout server 和 agent 环境放到一个能改、能调、能扩展的框架里。

这篇从 SGLang 视角读下来，可以把结论压成三点：第一，LLM RL 的 rollout 是系统核心路径，不是辅助脚本；第二，训练和推理之间必须有权重同步、显存切换和数据格式对齐；第三，Agentic RL 让 server mode、异步调度、sticky session、trajectory mask 这些推理系统细节直接影响训练效果。

# 0x3. 关键代码拆解

先看 verl 的 SGLang adapter。它会根据当前 rank 算出自己对应哪个 SGLang server，PD 分离时还要区分 prefill/decode：

```python
if disagg is not None and getattr(disagg, "enabled", False):
    footprint = prefill_tp + disagg.decode_replicas * decode_tp
    local = self.rollout_rank % footprint
    if local < prefill_tp:
        self._pd_role = "prefill"
        self._pd_server_index = 0
        self._pd_tp_local_rank = local
    else:
        off = local - prefill_tp
        self._pd_role = "decode"
        self._pd_server_index = off // decode_tp
        self._pd_tp_local_rank = off % decode_tp
```

Hybrid Controller 的 dispatch/collect 也能在源码里直接看到。`Dispatch.DP_COMPUTE_PROTO` 这类模式不是简单把 Python 参数广播出去，而是对 `DataProto` 按 worker 数切分；切不均时会自动 padding，收回来再 concat：

```python
def dispatch_dp_compute_data_proto(worker_group, *args, **kwargs):
    assert isinstance(worker_group, WorkerGroup)
    # enable auto padding for dp compute DataProto
    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto_with_auto_padding(
        worker_group.world_size,
        *args,
        **kwargs,
    )
    return splitted_args, splitted_kwargs

def collect_dp_compute_data_proto(worker_group, output):
    assert BatchData(output).is_concatable()
    output = collect_dp_compute(worker_group, output)
    return _concat_data_proto_or_future(output)
```

这段对应 Slide 13/16。verl 想让上层写普通 Python 控制流，但底层 rollout、reward、logprob、ref logprob 都是分布式 worker 在算，所以它需要一套显式的“参数怎么分发、结果怎么收回”的语义。`DataProto` 自动 padding 的意义也很实际：RL batch 经常不是 world size 的整数倍，框架不能因为最后几个样本让用户手工补齐。

release/resume 直接打到 SGLang HTTP 接口。这里的 `sleep_level` 对 LoRA 场景影响很大，LoRA adapter 模式下可以只释放 KV cache，保留 base weights：

```python
async def resume(self, tags: list[str]):
    await self._init_server_adapter()
    if self._engine is None:
        return
    if self._is_server_tp_leader() and self.config.free_cache_engine:
        await self._engine.resume_memory_occupation(tags=tags)

async def release(self):
    await self._init_server_adapter()
    if self._engine is None:
        return
    if self._is_server_tp_leader() and self.config.free_cache_engine:
        if self.sleep_level == 1:
            tags = ["kv_cache"]
        else:
            tags = ["kv_cache", "weights"]
        await self._engine.release_memory_occupation(tags=tags)
```

权重更新路径里有一行注释很值钱：所有 rank 都必须迭代 weights generator。原因是 `DTensor.full_tensor()` 里面可能触发 FSDP all_gather，某些 rank 跳过就会把其它 rank 卡住。

```python
async for params_batch in get_named_tensor_buckets(weights, update_weights_bucket_bytes):
    await sgl_update_weights(
        engine=self._engine,
        params_batch=params_batch,
        device_mesh_key="infer_tp",
        device_mesh=self.device_mesh,
    )
```

Agentic RL 可以看 `ToolAgentLoop`。它就是一个状态机：

```python
state = AgentState.PENDING
while state != AgentState.TERMINATED:
    if state == AgentState.PENDING:
        state = await self._handle_pending_state(agent_data, sampling_params)
    elif state == AgentState.GENERATING:
        state = await self._handle_generating_state(agent_data, sampling_params)
    elif state == AgentState.PROCESSING_TOOLS:
        state = await self._handle_processing_tools_state(agent_data)
```

生成阶段调用 server，解析 tool calls：

```python
output: TokenOutput = await self.server_manager.generate(
    request_id=agent_data.request_id,
    prompt_ids=agent_data.prompt_ids,
    sampling_params=sampling_params,
    image_data=agent_data.image_data,
    video_data=agent_data.video_data,
)
agent_data.response_ids = output.token_ids
agent_data.prompt_ids += agent_data.response_ids
agent_data.response_mask += [1] * len(agent_data.response_ids)
_, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(
    agent_data.response_ids, tools
)
```

工具阶段并发执行，然后把工具响应追加回上下文；注意工具响应的 `response_mask` 是 0：

```python
tasks = []
for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
    tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data))
responses = await asyncio.gather(*tasks)

agent_data.messages.extend(add_messages)
response_ids = await self.apply_chat_template(
    add_messages,
    images=images,
    videos=videos,
    remove_system_prompt=True,
)
agent_data.prompt_ids += response_ids
agent_data.response_mask += [0] * len(response_ids)
agent_data.user_turns += 1
return AgentState.GENERATING
```

这段代码把 slide 里的 Agentic RL 抽象落到了训练数据：模型生成的 token 训练，环境返回的 token 只作为上下文。

# 0x4. 小结

verl 这套系统的重点是“可编排”。普通 RL 已经需要多模型、多后端、多通信组；Agentic RL 又加上多轮工具和环境状态。SGLang 在这里不是可有可无的 serving 组件，而是 rollout 数据生产线的一部分。
