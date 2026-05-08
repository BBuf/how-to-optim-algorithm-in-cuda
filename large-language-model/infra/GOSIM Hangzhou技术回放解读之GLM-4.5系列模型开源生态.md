> GLM-4.5 可以从能力榜单讲，但放在 infra 目录里，我会把重点落到模型生态背后的训练、rollout 和开源适配链路上。

# 0x0. 前言

这篇会把 GLM-4.5 的 slides 当成生态解读，而不是复述榜单。模型结果很重要，但对 infra 目录来说，更需要看清楚的是：GLM 系列的 post-training 怎么被系统化，rollout 后端怎么接 SGLang，模型开源之后又如何进入 Transformers、vLLM、SGLang、PEFT 这些生态。

# 0x1. 资料和代码落点

相关资料和代码：

- GLM-4.5 官方入口：`https://github.com/zai-org/GLM-4.5`，GLM-V 入口：`https://github.com/zai-org/GLM-V`。
- slime：`README.md` 明确列出 GLM-4.5/4.6/4.7/5 系列背后的 RL framework。
- slime：`slime/backends/sglang_utils/sglang_engine.py`，SGLang engine 启动、release/resume、update weights。
- slime：`slime/ray/rollout.py`，SGLang server group、router、colocate offload/onload、rollout 数据回流。
- slime 文档：`docs/zh/get_started/quick_start.md` 和 `docs/zh/examples/glm4.7-355B-A32B.md`，能看到 SGLang 参数、partial rollout、MTP 配置等。
- LMSYS slime blog：`https://lmsys.org/blog/2025-07-09-slime/`，解释 SGLang-native RL 系统为什么要把训练、rollout、权重同步放进同一个框架。

GLM-4.5 的模型生态如果只看权重发布会少一块：这些模型后续 RL/post-training 怎么跑，rollout 后端怎么接，权重怎么同步。LMSYS 的 slime blog 可以作为公开参考：

<img src="https://files.mdnice.com/user/59/e62c3bfc-5f68-4989-a912-c8495837396d.png" referrerpolicy="no-referrer" />

这张图里的 SGLang Server Group、Router、Weight Sync 和 Partial Rollout，基本就是后面读 slime 代码的地图。GLM 系列模型体量大、MoE 多、上下文长，训练端每轮更新后都要让 rollout 侧尽快吃到新权重。slime 选择围绕 SGLang 做原生集成，原因也在这里：推理引擎的 release/resume、server group 管理、采样参数和 rollout 数据回流，都需要进入训练框架的调度面。

# 0x2. Slides 逐页解读

### Slide 1：GLM-4.5 系列模型开源生态

<img src="https://files.mdnice.com/user/59/8e9e3b5e-658f-41ee-9d46-b081e2a3bad4.png" referrerpolicy="no-referrer" />

标题页给出主题：GLM-4.5 系列模型开源生态。这里的“生态”不是只把模型权重传到 HuggingFace，而是语言模型、视觉模型、训练框架、推理框架、文档、社区活动一起构成的使用路径。

和 infra 目录最相关的是 slime：GLM-4.5 背后的 RL/post-training 框架公开后，训练和 SGLang rollout 的工程细节也能被讨论。后面看 GLM-4.5 强化学习那几页时，slime 的 training、rollout、data buffer 会成为主线。

### Slide 2：目录

<img src="https://files.mdnice.com/user/59/9e72dc41-aa5f-40e5-bcc9-cf7524246546.png" referrerpolicy="no-referrer" />

目录分三段：GLM-4.5 语言模型、GLM-4.5V 视觉模型、GLM-4.5 系列模型开源生态。这个顺序也很自然：先说明语言模型能力和训练，再说明视觉模型的结构和数据工程，最后讲开源适配与社区传播。

从工程角度看，第三部分不是收尾材料，而是模型能否被真正使用的关键。大模型权重发布后，用户还需要 tokenizer、processor、推理样例、框架兼容、量化/服务方案和 issue 修复，这些都会影响模型的实际可用性。

### Slide 3：语言模型能力背景

<img src="https://files.mdnice.com/user/59/093767d4-574e-4842-9c15-682319a4cd53.png" referrerpolicy="no-referrer" />

这一页是语言模型章节的分隔页。后面几页会围绕 GLM-4.5 的 all-round 定位展开：reasoning、coding、agent 和通用能力都要覆盖。

这个定位会直接影响训练系统。只做聊天模型时，SFT 和偏好对齐占主导；要做 coding 和 agent，就需要执行反馈、工具调用、多轮环境和可验证 reward。也就是从“训练模型回答问题”走向“训练模型完成任务”。

### Slide 4：GLM-4.5：All-round model

<img src="https://files.mdnice.com/user/59/a5d8c235-f42e-4fc5-b92e-27c33a843844.png" referrerpolicy="no-referrer" />

这一页把 GLM-4.5 定位成“全优生”。slide 小字写到：复杂推理、代码生成、智能体交互是内置能力；代码和智能体能力位列全球开源模型第一，推理能力第二。这里的表述强调的是“均衡”：不是只在某一个 benchmark 上冲高，而是让模型能覆盖真实任务里的多种能力。

均衡模型对 post-training 更敏感。reasoning 需要可验证题目和过程奖励，coding 需要执行反馈和测试用例，agent 需要工具调用轨迹和环境反馈；这些数据最后都要进入 RL 或 rejection sampling 流程。slime/SGLang 的意义就在这里：它让大规模 rollout 和训练更新能形成闭环。

### Slide 5：Agent 能力对比

<img src="https://files.mdnice.com/user/59/7461aada-03eb-4dd7-b686-828aa89365cb.png" referrerpolicy="no-referrer" />

Agent 能力对比页提到 τ-bench 和 BFCL-v3，并说 GLM-4.5 在这些基准上与 Claude 4 Sonnet 接近。τ-bench 关注真实工具环境下的任务完成，BFCL-v3 关注 function calling 的工具选择、参数生成和多轮调用。

这类 benchmark 的难点不在单轮语言质量，而在状态跟踪和工具协议。模型要知道什么时候调用工具、调用哪个工具、参数是否完整，工具返回后还要继续规划下一步。放回系统侧，就对应 agentic rollout、tool parser、chat template、reward/verifier 和 server-side session 管理。

### Slide 6：代码能力

<img src="https://files.mdnice.com/user/59/1cd41d0a-7630-442f-b328-0086c36ca486.png" referrerpolicy="no-referrer" />

代码能力页给的是 52 个编程任务实测，任务覆盖前端开发、工具开发、数据分析、测试和算法实现等。slide 里还给了对比胜率：相对 Kimi K2、Qwen3-Coder、Claude-4-Sonnet 分别展示不同胜率。

代码能力提升通常不是单靠 pretrain。真正能拉开差距的是高质量指令数据、执行反馈、单测/verifier、错误样本回流，以及能稳定产生代码轨迹的 rollout 系统。这里和 GLM-4.5 强化学习部分是连着的：coding 的 reward 往往来自测试结果，不能只靠人工偏好评分解决。

### Slide 7：通用能力

<img src="https://files.mdnice.com/user/59/89098ef1-d8db-4bc1-9c85-0a330b86d3ab.png" referrerpolicy="no-referrer" />

通用能力页说明模型没有只往 agent/coding 偏。全能型模型如果只为某个窄任务优化，很容易在普通问答、写作、知识、数学之外出现能力塌陷。

这类模型训练里，数据 mixture、loss balancing、post-training 阶段切换会直接影响最终能力。比如 agent/coding 的数据比例过高，模型可能变得过度工具化；通用对话数据不足，又会影响日常可用性。后面 GLM-V 的多模态训练也会遇到类似的“多任务进度不一致”问题。

### Slide 8：搜索工具调用示例

<img src="https://files.mdnice.com/user/59/7fd3013d-dbfd-42e4-8a57-297b3955955c.png" referrerpolicy="no-referrer" />

这一页展示的是搜索类工具调用。画面里模型会先理解用户问题，再把问题拆成可搜索的查询，拿到外部信息后组织成回答。它不是核心系统图，但能看出 GLM-4.5 的产品化目标：模型不只回答静态问题，还要能把外部工具纳入推理过程。

对训练系统来说，这类样本会把 rollout 从单轮文本生成推到 tool-use 轨道。训练数据里需要保留 tool call、tool result、最终回答之间的边界；推理时需要把同一条多轮会话尽量留在同一个 backend 上，以减少 KV 重建成本。

### Slide 9：模型即产品的能力展示

<img src="https://files.mdnice.com/user/59/bc6626cd-eaeb-40e2-98f6-4d1c2bd1da3a.png" referrerpolicy="no-referrer" />

这页是“模型即产品”的展示：复杂需求会被拆成搜索、规划、写作、格式控制等多个步骤。它放在模型能力部分里，说明 GLM-4.5 不是只追 benchmark 分数，也在强调 agent/product workflow 的可用性。

对训练系统来说，这类页面提示了一个评价方向：模型输出最终结果只是最后一步，中间是否会规划、是否能把搜索结果整合进答案、是否能按目标格式输出，都要进入数据构造和评测。Agent 能力如果只看最终文本，很容易漏掉中间决策质量。

### Slide 10：Tic-tac-toe 示例

<img src="https://files.mdnice.com/user/59/93220e50-7220-4c6f-b3d3-2cbf1406de1b.png" referrerpolicy="no-referrer" />

Tic-tac-toe 示例的 prompt 是“一句话，做一个真的玩的井字棋游戏”。它看起来是产品 demo，但技术上覆盖了代码生成、交互状态管理和前端逻辑。模型不能只写一段静态 HTML，还要让棋盘状态、胜负判断、重开逻辑都能工作。

这类例子反映了 coding-agent 任务的变化：用户不是要一段孤立代码，而是要一个能运行、能交互、能满足约束的小应用。训练和评测时也不能只看自然语言相似度，最好要进入执行环境或浏览器环境验证。

### Slide 11：PPT/小红书生成示例

<img src="https://files.mdnice.com/user/59/265cbf10-d658-4afd-a594-4a6ed0e2073d.png" referrerpolicy="no-referrer" />

PPT/小红书示例偏产品化，但它测的是另一类能力：长结构化输出、视觉排版、风格控制和工具调用的组合。用户只给一句需求，模型要生成有格式、有层次、有审美约束的内容。

这和前面的搜索页连起来看，就是从“会调用工具”走向“能完成一个有格式要求的任务”。训练数据里如果只记录最终文本，不记录中间规划和工具状态，模型很难稳定学会这类 workflow。

这类任务还会拉高上下文和输出长度：模型要先理解主题，再组织标题、段落、图文位置和风格。服务端如果要承接这种 workflow，除了模型能力，还要考虑长输出 decode、工具调用超时和中间状态保存。

### Slide 12：模型结构对比

<img src="https://files.mdnice.com/user/59/a90ddb09-200c-4bd7-8570-826cdde2f961.png" referrerpolicy="no-referrer" />

模型结构页讲的是 GLM-4.5 系列的底座差异。表格通常会把总参数、激活参数、上下文长度、模型尺寸和不同版本放在一起比较。对 infra 读者来说，重点不只是参数规模，而是 MoE、active 参数、长上下文和推理成本会共同决定后面的训练、rollout 和 serving 方案。

MoE 的好处是总参数可以上去，单 token active 参数受控；代价是训练和推理都要处理 expert routing、EP/DP 负载不均、checkpoint 转换和 serving cache。具体数值以官方仓库为准，正文不凭截图二次转述。

### Slide 13：GLM-4.5 训练过程

<img src="https://files.mdnice.com/user/59/1d0ff2df-d84f-4d0c-a1ea-6da05aa1d8aa.png" referrerpolicy="no-referrer" />

训练过程页把 pretraining、mid-training、SFT、RL 串起来。pretraining 打底，mid-training 调整能力分布，SFT 对齐指令格式，RL 再根据 reward/verifier 优化 reasoning、coding、agent 等任务。

slime 正是后训练阶段的系统支撑之一：training 模块从 data buffer 读数据并更新模型，rollout 模块用 SGLang + router 生成新样本，reward/verifier 的结果再写回 data buffer。它把“训练”和“生成训练数据”从两个脚本变成一个联动系统。

### Slide 14：GLM-4.5 强化学习

<img src="https://files.mdnice.com/user/59/953e3746-b2b5-406d-a8c9-506fe31946f3.png" referrerpolicy="no-referrer" />

强化学习第一页说明 GLM-4.5 的 post-training 不只做静态 SFT。RL 需要 rollout 引擎不断生成样本，再把 reward/verifier 结果回流训练。模型越大，rollout 吞吐、显存占用和权重同步越会变成系统问题。

从 slide 的图看，slime 由 training、rollout 和 data buffer 三块组成。training 负责主要训练过程，并在训练后把参数同步给 rollout；rollout 由 SGLang + router 生成新数据，包括奖励/验证器输出；data buffer 负责管理 prompt 初始化、自定义数据和 rollout 生成方式。这就是 GLM-4.5 RL 里最核心的 infra 路径。

### Slide 15：强化学习策略细节

<img src="https://files.mdnice.com/user/59/0c8c7f51-9222-4ad2-9ae7-e5f77870213f.png" referrerpolicy="no-referrer" />

这一页继续展开 RL recipe。Step-wise Rule-based RL 用过程奖励显式约束分步推理，提高复杂任务里的逻辑一致性；End-to-end Multi-turn RL 直接优化完整交互过程，让模型学会主动提问、澄清和规划；Pathology RL 则针对混语、重复、格式问题这类低频错误构造专门数据并施加惩罚。

这些策略都依赖稳定 rollout。过程奖励需要拿到中间步骤，multi-turn RL 需要保存多轮轨迹，Pathology RL 需要把低频失败样本找出来再回流。slime 的 data buffer 和 SGLang rollout 不是附属工具，而是这些 RL 策略能跑起来的基础。

### Slide 16：slime 开发者文档

<img src="https://files.mdnice.com/user/59/698be5f3-f606-4c60-a51b-5f73d64a3972.png" referrerpolicy="no-referrer" />

slime 文档入口是这里的代码落点。slide 截图里是开发者文档页面，说明 GLM-4.5 的 RL 系统不是内部口头方案，而是已经公开到可以按文档启动和调试。

slime README 明确写到它是 GLM-4.5 背后的 RL framework，并且用 Megatron + SGLang 连接训练和 rollout。后面代码拆解会沿着 SGLang engine、router 和 weight update 看：server 怎么启动，rollout worker 怎么注册到 router，训练后的权重怎么更新到推理侧。

### Slide 17：GLM-4.5V 视觉理解模型

<img src="https://files.mdnice.com/user/59/bedf3563-08bf-4f9e-b798-e009a6abf4f1.png" referrerpolicy="no-referrer" />

GLM-4.5V 把话题转向视觉语言模型。这里不只是“语言模型加图片输入”，而是要处理 grounding、OCR、视频、GUI、视觉问答和多模态安全这类任务。

VLM 的 RL 比纯文本更复杂，因为输入输出可能涉及区域位置、截图状态、GUI 操作、视觉 verifier。训练样本除了 token ids，还可能带图像 patch、box、time index、视频帧和任务动作。

这一页作为视觉模型章节的开头，也把后面几页串起来：先讲 grounding 和安全检测，再讲模型结构、预训练数据、训练策略和 VLM RL。理解 GLM-4.5V 要同时看结构和数据工程，不能只看一张能力榜单。

### Slide 18：Grounding 和语义理解

<img src="https://files.mdnice.com/user/59/e34aeda2-253a-4587-b00c-c15a0dbb3178.png" referrerpolicy="no-referrer" />

Grounding 和语义能力页强调视觉定位。图里展示的不是普通 caption，而是模型要在图片中找到目标、理解区域关系，再用语言表达出来。训练数据里需要 box、region、OCR、语义关系等监督，推理时也要保留足够视觉分辨率。

VLM 的难点不仅是“把图片塞进上下文”，还要让空间信息能被语言模型稳定消费。vision encoder 产出的 patch token 经过 projector 进入 language decoder，如果 token 过少会丢细节，token 过多又会推高上下文成本。

### Slide 19：安全检测与 Grounding 能力

<img src="https://files.mdnice.com/user/59/8154eb49-f8fe-4760-ae5b-562a9f026df8.jpg" referrerpolicy="no-referrer" />

这一页把 grounding 能力和安全检测放在一起，示例是基于 GLM-4.1V-Thinking 的安全检测系统，包括火灾、烟雾和安全帽佩戴。它不是简单分类任务，很多场景需要模型指出风险发生在哪个区域。

多模态安全比文本更难，因为图像里可能隐藏文本、符号、姿态或场景线索；而 grounding 能力又要求模型真的看懂位置，不能只靠语言先验猜。工业落地时，这类能力还要和告警阈值、人工复核、误报漏报成本一起设计。

### Slide 20：GLM-4.5V 模型介绍

<img src="https://files.mdnice.com/user/59/fc82782d-3810-4182-9309-5b8f16c45902.png" referrerpolicy="no-referrer" />

GLM-4.5V 模型介绍页把视觉模型和语言模型联系起来。它通常会交代底座、输入分辨率、支持图像/视频类型，以及模型在 grounding、OCR、GUI、视频理解上的定位。

开源生态里，VLM 不能只给权重。用户还需要 processor、chat template、图片/视频预处理、推理示例和评测脚本；否则模型虽然开源，真实业务或评测环境里还是接不起来。

这里也为 Slide 21 的结构图做铺垫：视觉输入不是一张图直接塞给 LLM，而是经过 ViT encoder、projector、token 拼接之后进入 language decoder。视频还要考虑时间压缩和 time index token。

### Slide 21：V 模型结构

<img src="https://files.mdnice.com/user/59/db736f29-19d1-448e-95c4-89dbd3b6dfcd.png" referrerpolicy="no-referrer" />

这页把 GLM-V 的输入序列画得比较清楚。底部是原生分辨率输入：Image 1、Image 2 和一段约 20s 的 Video 1，ViT Encoder 负责抽视觉特征，并且视频路径带 2x temporal compression。中间的 MLP Projector 把视觉特征投到语言模型 hidden size。上方的 Language Decoder 收到的是一串混合 token：普通文本 token、图像 token、视频 token、time index token，以及右上角虚线框里的 predicted token。

图里标了 token 数：第一张图约 1574 tokens，第二张图约 5187 tokens，视频约 13650 tokens。这个数量级说明 VLM 的上下文压力主要来自视觉 token，不是用户那句 “Could you tell me...?”。所以 GLM-V 这类模型必须在 native resolution、时间压缩、projector 和长上下文推理之间取平衡；否则图像/视频理解能力上去了，serving 成本会很快顶住。

### Slide 22：预训练

<img src="https://files.mdnice.com/user/59/a1edb74f-8661-4277-8be2-9ef515459558.png" referrerpolicy="no-referrer" />

这页的曲线横轴是 sample 数量 k，纵轴是 Pass@k。蓝线是 GLM-4.1V-9B-Base，绿线是 InternVL3-9B-Pretrain。两条线都随着 k 从 1 增到 64 持续上升，说明视觉理解任务里多采样仍然能换来更高通过率；蓝线整体高于绿线，并且在 k=4 到 k=16 区间提升很明显。

把这页放在“预训练”章节里看，重点不是只比一个最终分数，而是说明视觉底座训练会影响后面多样化采样的上限。VLM 预训练里的图文配对质量、OCR 数据、视频帧采样都会影响 grounding、GUI 和视觉问答能力。这里的数据工程比单个模型结构改动更能决定上限。

### Slide 23：数据工程第一部分

<img src="https://files.mdnice.com/user/59/12988909-d625-42cc-a1eb-c3f1bfadd22f.png" referrerpolicy="no-referrer" />

这页把图文预训练数据拆成两块。左边是 Image Caption Data，规模写着 10B+ high-quality image-text pairs，来源包括 LAION、DataComp、DFN、Wukong 和 web sources。下面的 multi-stage refinement 有四步：先按分辨率、caption 长度、去重做 heuristic filtering；再用 CLIP-score，阈值标成大于 0.3；然后做 concept-balanced resampling，参考 MetaCLIP 的思路；最后做 factual-centered recaptioning，用迭代模型训练给 caption 去噪和补充事实。

右边是 Interleaved Image-Text Data，强调 beyond alt-text，也就是不只要图片标题，还要复杂图文关系。Web Data Pipeline 的来源包括 MINT、MMC4、OmniCorpus，过滤里有 CLIP-score relevance、广告/二维码噪声移除、高知识密度图片分类器、低文本密度样本过滤。Academic Book Pipeline 则来自 100M+ digitized STEM books，并做领域过滤、PDF 图文抽取和深度解析。这一页对应 VLM 训练里“图像 caption”和“图文交错长上下文”两条数据线。

### Slide 24：数据工程第二部分

<img src="https://files.mdnice.com/user/59/ce03fc77-26e7-4df0-95da-8e99d3d66430.png" referrerpolicy="no-referrer" />

这页继续把数据源拆成 OCR、grounding、video 和 instruction tuning。OCR Data 标了 220M total images，包含 synthetic documents、natural scene text 和 academic documents：synthetic documents 是多背景文本渲染，natural scene text 来自 Paddle-OCR 抽取的文字框，academic documents 来自 LaTeXML 处理过的 arXiv 论文。Grounding Data 标了 40M natural image annotations 和 140M+ GUI QA pairs，下面两类是 natural image grounding 和 GUI grounding，后者把 DOM elements 与 Playwright interactions 结合起来。

左下的 Video Data 来自 academic、web 和 proprietary sources，强调 fine-grained human annotation、cinematic elements 和 rigorous filtering protocol。右下 Instruction Tuning Data 标了 50M samples，包含 task coverage & taxonomy、complex scenario augmentation 和 data contamination check。把这页和上一页连起来看，GLM-V 的数据工程不是单一 caption 数据，而是把 OCR、区域定位、GUI 操作、视频理解和指令数据混在一起做能力覆盖。

### Slide 25：训练策略

<img src="https://files.mdnice.com/user/59/986279af-c2ba-4d73-87b5-c5aa67dd343f.png" referrerpolicy="no-referrer" />

训练策略页把模型结构、数据和并行训练放到一起。VLM 训练比纯文本多一层数据调度：图像 caption、OCR、grounding、视频、GUI、instruction tuning 的采样比例都要设计；视觉 token 长度又会直接影响吞吐。

大 MoE/VLM 训练还要处理 TP/PP/EP/DP、checkpoint 转换和 rollout 同步。这里也是 slime/SGLang 这类训练-推理联动系统会进入视野的原因：一旦 VLM 也进入 RL，rollout 端不仅要生成文本，还要处理图像输入、GUI 环境和 verifier。

### Slide 26：RL for VLM

<img src="https://files.mdnice.com/user/59/99994339-2c92-4d14-b2d3-9336a2f32ae5.png" referrerpolicy="no-referrer" />

RL for VLM 页说明多模态也要进入 RL。GUI、grounding、OCR 这类任务往往有可验证反馈，适合做 RL 或 rejection sampling：框选是否命中、按钮是否点对、OCR 是否正确，都能构造成 reward 或 verifier。

难点在于 rollout 样本不再只是 token 序列，还包含图像输入、区域位置和环境反馈。训练系统要保存的不只是 `input_ids` 和 `response_mask`，还要保存视觉输入的索引、processor 参数、截图状态、动作执行结果。否则 reward 很难准确回到对应 token。

### Slide 27：GUI Agents 和 CogAgent

<img src="https://files.mdnice.com/user/59/c763ccff-0c88-4d93-b5b2-dad829991396.png" referrerpolicy="no-referrer" />

GUI Agents 和 CogAgent 页把 VLM 推向实际环境操作。slide 提到 integrated with CogAgent、task-oriented data collection & improving loop、cross-platform GUI instruction capabilities。也就是说，模型不只是看截图回答问题，而是要在真实 UI 里完成任务。

训练框架要支持多轮、截图输入、动作输出和环境状态，这和 agentic RL 的系统需求一致：生成动作，执行动作，观察界面变化，再生成下一步。GUI 数据还会带来平台差异，比如 Web、移动端、桌面软件的控件体系不同，不能把它当成普通 VQA 数据处理。

### Slide 28：开源生态

<img src="https://files.mdnice.com/user/59/8f769407-82df-43c7-a1c9-7992d9409e9b.png" referrerpolicy="no-referrer" />

开源生态页是第三部分的开头。前面讲模型能力和训练，这里转到模型如何被社区使用：权重、代码、demo、文档、框架适配、issue/PR、技术活动都会影响 adoption。

对 GLM-4.5 这种体量的模型来说，生态适配往往决定用户能不能真正跑起来。模型越大，用户越依赖现有 serving 框架、量化方案、推理文档和模型转换脚本；缺一个环节，使用门槛都会明显上升。

这也是这篇文章把 slime 放在代码拆解部分的原因。模型生态不是发布截图结束，而是训练系统、rollout 系统、推理后端和开源框架一起把模型变成可使用的工程资产。

### Slide 29：生态地图

<img src="https://files.mdnice.com/user/59/fefa4c73-d177-46e0-b3dd-d522e628f8b9.png" referrerpolicy="no-referrer" />

生态地图页展示周边项目和文档入口。它的作用是把模型从“一个 repo”扩展成“一个可导航的生态”：模型仓库、技术报告、训练框架、推理框架、社区教程都在地图里占位。

对 infra 读者来说，重点是 slime、SGLang、Megatron、serving 框架之间的连接方式。训练、rollout、权重格式、推理配置都需要有明确落点；否则生态地图会变成链接集合，而不是可复现路径。

### Slide 30：HuggingFace trending

<img src="https://files.mdnice.com/user/59/4be48ff8-28dd-4294-a614-95cc88ce1375.png" referrerpolicy="no-referrer" />

HF trending 页说明发布后的社区反馈。HuggingFace Trending 不能替代技术评测，但能说明模型资产被真实用户拿去试，下载、复现、报 issue 的人多，框架兼容问题会更快暴露。

开源模型的热度会反过来推动推理框架、量化方案和训练系统补齐兼容性。比如 tokenizer 特殊 token、MoE 权重命名、VLM processor 字段、SGLang/vLLM 参数，都可能在社区使用中被不断修正。

### Slide 31：框架采用情况

<img src="https://files.mdnice.com/user/59/7b249fee-0bf7-4fa1-b12f-c8e1544b118b.png" referrerpolicy="no-referrer" />

框架采用情况页展示主动适配主流开源框架。这里至少包括 Transformers、PEFT、Accelerate、Diffusers 这类基础库，也包括 vLLM/SGLang 这类 serving 框架。对开发者来说，能不能用熟悉的框架加载和部署，往往比论文分数更影响上手。

大模型开源如果只能在单一脚本跑，传播会慢很多；真正可用的开源，需要让不同 serving 和训练工具都有清楚的接入路径。GLM-4.5 这种 MoE/长上下文模型尤其需要框架侧适配，否则高性能推理和多卡部署会卡在模型加载阶段。

### Slide 32：开源流程

<img src="https://files.mdnice.com/user/59/0eee7cb4-1b5e-4c4d-9e4e-64e2e7931d94.png" referrerpolicy="no-referrer" />

开源流程页画了多方协作：原始模型权重转换成 HuggingFace 权重，算法重构和代码适配进入代码仓库，合作伙伴做适配支持，社区通过 PR/Issue 反馈，最后落到推理与应用、模型微调、品牌和生态推广。

真正省别人时间的是明确权重格式、推理配置、训练 recipe 和已知限制。模型发布之后，框架适配、文档修复和社区反馈会继续改变可用性。这个流程也解释了为什么 GLM-4.5 的开源生态要和 slime/SGLang 放在一起看：训练和推理链路都需要被持续维护。

### Slide 33：社区反馈

<img src="https://files.mdnice.com/user/59/7a2bba40-60d7-45b2-8bed-3a386da00e97.png" referrerpolicy="no-referrer" />

社区反馈页强调技术解读、开源生态活动和文档。slide 里提到每月至少超过一场活动/直播，目标是降低开发者入门门槛。对大模型来说，文档和活动不是宣传附属品，而是生态维护的一部分。

很多用户遇到的问题不是模型能力不足，而是“怎么部署”“怎么接工具”“怎么开长上下文”“为什么显存爆了”。频繁的技术活动和 issue 反馈能让这些问题更快沉淀成文档、脚本和框架 patch。

### Slide 34：论文和技术社区

<img src="https://files.mdnice.com/user/59/eee3f7cd-591c-46d5-8940-a4a2cd3cca2a.png" referrerpolicy="no-referrer" />

论文和技术社区页给出入口：GLM-4.5 paper、GLM-4.5 GitHub、GLM-4.5V/GLM-4.1V paper、GLM-V GitHub。读者如果要核对模型结构、benchmark、processor 和推理示例，这些是第一手资料。

代码讲解选择 slime，因为它和 GLM-4.5 RL 训练关系最直接；模型结构和能力细节则回到官方 GLM-4.5/GLM-V 仓库核对。这样可以把“能力解读”和“系统实现”分开，避免把截图上的发布信息当成长期稳定事实。

### Slide 35：结束页

<img src="https://files.mdnice.com/user/59/81d7b369-ad20-49ef-9e6d-2123531bfe45.png" referrerpolicy="no-referrer" />

结束页。GLM-4.5 的技术细节会继续变化，博客里更关注当前公开源码能验证的训练系统路径。

# 0x3. 关键代码拆解

slime README 直接给出了架构：training 用 Megatron，rollout 用 SGLang + router，中间是 data buffer。代码里 `SGLangEngine` 是对 SGLang server 的 Ray actor 包装。

启动时，slime 会构造 SGLang `ServerArgs`，拉起 HTTP server，并把 worker 注册到 router：

```python
self.process = launch_server_process(ServerArgs(**server_args_dict))

payload = {
    "url": f"http://{self.server_host}:{self.server_port}",
    "worker_type": self.worker_type,
}
if self.worker_type == "prefill":
    payload["bootstrap_port"] = server_args_dict["disaggregation_bootstrap_port"]
requests.post(f"http://{self.router_ip}:{self.router_port}/workers", json=payload)
```

训推一体时，rollout engine 要能释放显存。slime 直接调用 SGLang 的接口：

```python
def release_memory_occupation(self):
    self.flush_cache()
    return self._make_request("release_memory_occupation")

def resume_memory_occupation(self, tags: list[str] = None):
    return self._make_request(
        "resume_memory_occupation",
        {"tags": tags},
    )
```

权重同步有 tensor 和 distributed 两条路径：

```python
def update_weights_from_tensor(self, serialized_named_tensors, load_format=None,
                               flush_cache=False, weight_version=None):
    payload = {
        "serialized_named_tensors": serialized_named_tensors,
        "load_format": load_format,
        "flush_cache": flush_cache,
    }
    if weight_version is not None:
        payload["weight_version"] = weight_version
    return self._make_request("update_weights_from_tensor", payload)
```

```python
def update_weights_from_distributed(self, names, dtypes, shapes, group_name,
                                    flush_cache=False, weight_version=None):
    payload = {
        "names": names,
        "dtypes": [str(dtype).replace("torch.", "") for dtype in dtypes],
        "shapes": shapes,
        "group_name": group_name,
        "flush_cache": flush_cache,
    }
    return self._make_request("update_weights_from_distributed", payload)
```

`rollout.py` 里的 `ServerGroup` 负责 SGLang engine 的生命周期。注意它给 server actor 注入了 memory saver 相关环境变量：

```python
env_vars = {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST} | {
    "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
    "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "true",
    "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "false",
}
```

offload/onload 对 colocate 训练很直接：

```python
def offload(self):
    if not self.needs_offload:
        return []
    return [engine.release_memory_occupation.remote()
            for engine in self.engines if engine is not None]

def onload(self, tags: list[str] | None = None):
    if not self.needs_offload:
        return []
    return [engine.resume_memory_occupation.remote(tags=tags)
            for engine in self.engines if engine is not None]
```

这和 GLM-4.5 slides 里的 RL 生态能对上：模型能力背后，需要一个能把 Megatron 训练、SGLang rollout、router 和 data buffer 组织起来的系统。

# 0x4. 小结

GLM-4.5 这篇的源码重点不是模型层内部，而是开源生态里的训练系统。slime 把 Megatron 和 SGLang 接起来，提供 rollout、显存 offload、权重更新和 partial rollout 等能力，支撑大模型 RL 从 demo 走到大规模训练。
