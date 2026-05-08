> OpenSeek 这组 slides 里，和 infra 最贴近的是 Dynamic Mask Attention 和 flash-dmattn。项目背景会讲，但更多笔墨放在 sparse attention 怎么落到 kernel。

# 0x0. 前言

OpenSeek 不是单个模型发布，而是一个围绕数据、算法和系统的开源协作计划。slides 里有不少项目层面的信息，本文会重点展开 Dynamic Mask Attention 和 flash-dmattn。

# 0x1. 资料和代码落点

相关资料和源码：

- OpenSeek：`README.md`、`configs/OpenSeek-Small-v1-Baseline/train/train_deepseek_v3_1_4b.yaml`，对应 baseline、数据、checkpoint、训练配置。
- flash-dmattn：`flash_sparse_attn/utils/mask.py`，对应 top-k/relu dynamic mask。
- flash-dmattn：`flash_sparse_attn/ops/triton/flash_sparse_fwd.py` 和 `flash_gated_fwd.py`，对应 sparse softmax skip 和 gated attention kernel。
- slides 中给出的论文入口是 Trainable Dynamic Mask Sparse Attention，代码仓库是 `https://github.com/SmallDoges/flash-dmattn`。

# 0x2. Slides 逐页解读

### Slide 1：OpenSeek：开源驱动的下一代 AI 模型

<img src="https://files.mdnice.com/user/59/2266a825-3c84-42b5-a12d-729d1ff9cf75.png" referrerpolicy="no-referrer" />

标题页给出 OpenSeek 的定位：open-source driven next AI models。它不是单个模型发布，而是把数据、算法、系统、评测和社区贡献放在一个开放协作框架里。

这组 slides 里有大量项目机制介绍，但也有一个很具体的系统点：DMA/flash-dmattn。文章会先按 slides 讲 OpenSeek 如何组织开源协作，再把 sparse attention 的代码落到 mask、autograd wrapper 和 Triton kernel。

### Slide 2：目录：项目、数据算法、效率和未来

<img src="https://files.mdnice.com/user/59/03e0f5ef-6ef9-4b60-bd5b-2d4943829196.png" referrerpolicy="no-referrer" />

目录把 OpenSeek 拆成四段：项目介绍，数据与算法中的退火和 RL，注意力机制演进，未来展望。这个顺序说明它不是只讲模型分数，而是先讲开源协作机制，再讲训练 recipe，最后进入系统效率。

和单模型发布不同，OpenSeek 更像一个开源协作计划，试图把数据、训练配置、评测和系统优化都打开。后面 DMA/flash-dmattn 的内容就是系统赛道能贡献的典型例子。

### Slide 3：OpenSeek 项目介绍

<img src="https://files.mdnice.com/user/59/6e3fec09-fb69-4273-a42a-63faf3ec8c48.png" referrerpolicy="no-referrer" />

这一页是项目介绍章节的分隔页。后面会用“忒修斯之船”和“开源社区即造船厂”两个比喻说明 OpenSeek 的工程范式：模型不是一次性训练完成的静态物体，而是数据、算法、系统持续替换后的结果。

OpenSeek 想解决的是开放协作里的 pipeline 问题。社区可能有数据、有算法、有系统优化，但如果没有统一 baseline、评估和合并机制，这些改动很难积累到同一个模型上。

### Slide 4：忒修斯之船：工程范式的替换

<img src="https://files.mdnice.com/user/59/9489f7fd-3273-4b71-857c-7714d6e25aa7.png" referrerpolicy="no-referrer" />

这一页用忒修斯之船解释 AI 模型的持续演进。左边是哲学思辨：船的木板陆续替换后，它是否还是原船；右边是工程范式：把大模型的数据、算法、系统持续更替，类比为替换木船的木板。

重点在“连续性”和“可回溯”。模型保持同一条演进主线，但每次替换一块数据、算法或系统木板都要能评估、复现、回滚。OpenSeek 希望把分散改进变成统一前进动力，而不是一堆互不兼容的 fork。

### Slide 5：开源社区作为船坞

<img src="https://files.mdnice.com/user/59/ae6cd900-357d-4518-bb43-6a2739f18728.png" referrerpolicy="no-referrer" />

这页把开源社区比作造船厂。图里三类贡献对应三块“木板”：拉取请求替换算法木板，数据贡献替换数据木板，系统优化替换系统木板。它强调的不是只有代码 PR 才算贡献，数据和算力/系统优化同样进入模型演进。

下方小字给了流程：“提案-验证-合并”。这是一套很工程化的说法：先提出增量改动，跑统一验证，再决定是否合并到主线。没有公开数据、训练脚本和评测，算法改进很难被别人接上，也很难判断它是不是在其它任务上退化。

### Slide 6：累计创新和竞赛机制

<img src="https://files.mdnice.com/user/59/c64ae432-3689-4785-ad02-64817eb6db5b.png" referrerpolicy="no-referrer" />

累计创新页讲的是全年滚动主题竞赛，每场比赛聚焦替换一块关键木板。slide 上列了 iter1 数据去噪、iter2 长文本、iter3 工具调用、iter4 安全对齐，正好覆盖数据、上下文、agent 能力和安全。

评审维度包括性能、资源、代码和可解释性。参赛者提交的是增量补丁而不是完整模型，这能降低参与门槛，也能减少“每个人训一套模型但无法比较”的问题。系统优化类贡献，比如 flash-dmattn，就适合用这种方式接入。

### Slide 7：OpenSeek Working Groups

<img src="https://files.mdnice.com/user/59/db58f485-27b0-4c2e-a480-5308d61950e8.png" referrerpolicy="no-referrer" />

Working Groups 页把协作拆成 System、Data、Algo。这个分工能让贡献路径更清楚：数据组负责清洗、合成和评测集；算法组负责训练 recipe、RL、结构改动；系统组负责训练/推理效率、kernel、并行和部署。

大模型开源如果没有分工，很容易变成一个 repo 里堆 issue。OpenSeek 的工作组设计是在承认一件事：模型能力不是单个方向能推动的，数据质量、算法策略和系统效率要同时往前走。

### Slide 8：非线性跃迁

<img src="https://files.mdnice.com/user/59/904713b1-b6eb-4f7b-9eee-587def83a228.png" referrerpolicy="no-referrer" />

非线性跃迁页接着“木板”比喻：当所有关键木板完成一轮替换，模型能力会从实验木船进化到商业航母。下面列了四个变化：推理成本下降、长文本窗口增大、工具调用成功率提升、安全对齐改善。

这就是开源协作的复利。一次数据清洗、一次 attention kernel、一次训练稳定性改进，单独看可能只是小幅提升；如果都能在同一条主线上被验证和合并，就可能改变模型能力曲线。

### Slide 9：开源竞赛和协作

<img src="https://files.mdnice.com/user/59/1b7fd886-d90a-45db-91e7-7108b5d474bd.png" referrerpolicy="no-referrer" />

这页介绍“超越杯”挑战赛。slide 写到算法和系统双赛道，初赛有 500 多支队伍报名、100 多支队伍提交结果，其中算法赛道占 60%；两个赛道各取前 10 名晋级复赛，优秀方案会全开源。

竞赛机制的价值在于给贡献者明确目标和统一评估。比如数据 mix、annealing schedule、RL recipe、sparse attention、kernel 优化，都可以用同一个 baseline 比较。对 OpenSeek 这种项目，竞赛不是单独活动，而是把外部贡献纳入主线的一种机制。

### Slide 10：OpenSeek 时间线

<img src="https://files.mdnice.com/user/59/96afe9b9-40cc-4216-af9d-f9180f828a49.png" referrerpolicy="no-referrer" />

时间线展示从一个组织发起到社区驱动开源的过程。2025.2 是 initiation，准备数据和合成；2025.5 Stage 1 发布 CCI4.0 数据集、OpenSeek-Small 和 pipeline；2025.9 Stage 2 启动竞赛并与贡献者一起训练 OpenSeek-Mid；2025.11 Stage 3 计划发布 OpenSeek-Mid(10B)、代码、数据和 checkpoint。

这张图说明 OpenSeek 的节奏不是“先闭门训好再开源”，而是边发布 baseline、边组织竞赛、边合并贡献。README 里能看到 CCI4.0-M2、OpenSeek-Small v1 和 100B baseline，这些就是外部贡献者复现实验的锚点。

### Slide 11：数据与算法：Annealing + RL

<img src="https://files.mdnice.com/user/59/4fde32cc-8091-49dc-bfe7-35c31d941c16.png" referrerpolicy="no-referrer" />

这一页是第二部分标题：数据与算法，退火与 RL 的双轮驱动。退火对应 mid-training 里数据分布和学习率/训练阶段的调整，RL 对应 post-training 中根据 reward 优化 reasoning 等能力。

OpenSeek-Small 的路线不是只堆 pretrain tokens，而是在 mid-training 和 post-training 里调整数据分布和训练目标。这个章节后面会用 two-stage mid-training、SFT、GRPO 来说明这条训练路径。

### Slide 12：OpenMDW 和 OpenSeek-Small

<img src="https://files.mdnice.com/user/59/3e707147-c692-4db7-b933-74c2965e6160.png" referrerpolicy="no-referrer" />

这一页说明 OpenSeek 系列模型采用 OpenMDW 协议，并给出 HuggingFace collection 和 OpenSeek-Small-v1-SFT 链接。OpenMDW 的重点是给 AI 模型开放协作提供更明确的授权基础，让数据、模型和衍生工作能在规则下共享。

对工程复现来说，链接本身不是全部。真正有用的是 checkpoint、wandb、config、eval、数据说明都放出来，方便别人定位某个改动的收益。否则开源只停留在模型下载，难以支持增量创新。

### Slide 13：效率方向

<img src="https://files.mdnice.com/user/59/f65a8b48-4763-4088-b1c3-4b88887e0ffa.png" referrerpolicy="no-referrer" />

这一页是 Model Efficiency 的章节过渡，后面开始进入训练方法和 attention 机制。效率在 OpenSeek 里不是单独的“部署优化”，而是和模型结构、训练 recipe、长上下文能力绑在一起。

长上下文 attention 的 O(N^2) 成本会限制训练和推理，DMA/NSA 这类 sparse attention 希望把无效 token 的计算跳掉。后面 flash-dmattn 的代码就是系统组贡献能落到模型效率上的例子。

### Slide 14：训练方法总览

<img src="https://files.mdnice.com/user/59/7fa673c9-031f-47f6-a71e-aa21db73ced2.png" referrerpolicy="no-referrer" />

训练方法总览页标题是 OpenSeek 的训练之道，强调数学推理能力提升。图里把训练分成 Mid-training 和 Post-training：前者用高质量专业数据，后者用指令微调和强化学习。

OpenSeek-Small 的 config 里可以看到 MoE、router、group top-k 等训练配置。这里的训练方法不是孤立算法，而是数据、模型结构和系统配置共同构成的 recipe。

### Slide 15：Mid-training 两阶段

<img src="https://files.mdnice.com/user/59/749964be-2da7-4959-8b7d-6e379cc018fa.png" referrerpolicy="no-referrer" />

Mid-training 页写了 two-stage training。Stage 1 是 Stable，用最多 200B math corpus 训练，让模型获得更深的数学知识；Stage 2 是 Decay，用 20B tokens 做连续训练，巩固和深化能力。

这个设计可以理解为把能力强化和分布回调分开。先用高质量数学数据集中强化，再用 decay 阶段缓解只在数学域上越训越偏的问题。slide 还引用了 OctoThinker，说明 mid-training 可以影响后续 RL scaling。

### Slide 16：Post-training：SFT + GRPO

<img src="https://files.mdnice.com/user/59/13107511-8346-4435-8190-f6071bcf137e.png" referrerpolicy="no-referrer" />

Post-training 页分两步。Step1 是 SFT，目标是提升 instruction-following，数据用 Infinity-Instruct-core，1.4M 高质量指令能达到完整 7M 数据集 95.7% 的性能；Step2 是 RL，算法用 GRPO，数据来自 GSM8K、MATH 等数学推理训练集。

这里的信号很清楚：数据筛选比盲目加量更重要。SFT 先把格式和指令跟随能力稳住，GRPO 再通过可验证数学任务优化推理路径。这个流程也方便外部贡献者分别替换数据、奖励、算法或训练配置。

### Slide 17：OpenSeek-Small 结果

<img src="https://files.mdnice.com/user/59/6007a647-f017-47f1-8e55-1ad274993268.png" referrerpolicy="no-referrer" />

结果页给出两点：最终 Decay model 在 MATH500 等数学 benchmark 上表现有竞争力，超过一些更大的对比模型；同时验证了 two-stage training 和 incremental innovation approach，也就是基础模型可以通过系统化增强获得更强能力。

作为 baseline，这页最重要的是给后续贡献一个参照物，而不是宣称终局模型。开源协作需要一个可比较起点：后续数据、算法、系统 patch 都要能和这个 baseline 进行同条件评估。

### Slide 18：训练曲线

<img src="https://files.mdnice.com/user/59/43e15568-2f6b-4165-811c-d9c6a013a6a4.png" referrerpolicy="no-referrer" />

训练曲线页分成 learning curves 和 benchmark performance。learning curves 用来看训练是否稳定、是否出现 loss spike 或 plateau；benchmark performance 用来看某个阶段的 checkpoint 在任务上是否真的提升。

大模型开源如果只给最终分数，不给训练曲线，外部贡献者很难知道自己的改动是在早期收敛、后期泛化，还是某个 benchmark 上偶然波动。OpenSeek 把曲线公开出来，更适合作为可迭代 baseline。

### Slide 19：Attention 演进

<img src="https://files.mdnice.com/user/59/bda5de41-896f-48f6-a1c7-442e903f765d.png" referrerpolicy="no-referrer" />

这一页是注意力机制演进章节的分隔页。后面会从传统 attention 的 O(N^2) 复杂度讲到 DMA，再讲 Trainable Dynamic Mask Sparse Attention 和 flash-dmattn 代码。

长上下文里，真正难的是 mask 既要省算力，又不能丢掉关键 token。固定 local window 容易漏掉远距离依赖，纯手写 sparse pattern 又不一定适配所有任务，DMA 的目标是让 mask 跟随输入动态变化。

### Slide 20：Dynamic Mask Attention

<img src="https://files.mdnice.com/user/59/fc766803-efc7-4903-a326-fcd425ceef33.png" referrerpolicy="no-referrer" />

这页先把问题压成一个复杂度对比：传统 attention 是 `O(N^2)`，序列长度翻倍，QK 和 softmax 的矩阵就按平方膨胀。DMA 的思路是给每个 token 动态选择少量关键历史 token，把计算复杂度降到 `O(N*w)`，其中 `w` 是保留下来的 token 或窗口数量。

这里的“动态”很重要。固定 local window 只能看最近一段上下文，长文档里的关键信息可能在很远的位置；DMA 希望 mask 由输入内容决定，既能跳过大部分无效位置，又能保留跨段依赖。后面 flash-dmattn 的 kernel 代码，做的就是让这种动态稀疏 mask 不只停留在公式上。

### Slide 21：Trainable Dynamic Mask Sparse Attention

<img src="https://files.mdnice.com/user/59/e99130af-10ce-486e-980c-76c30edb320f.png" referrerpolicy="no-referrer" />

这页是论文标题页：Trainable Dynamic Mask Sparse Attention。重点在 `Trainable` 和 `Dynamic Mask` 两个词。mask 不再是手写规则，也不是固定滑窗，而是通过可训练参数给不同历史位置打分，然后只保留 top-w 的位置进入 attention。

这样做比纯 sparse pattern 更贴近语言任务：模型可以学习“当前 query 应该回看哪几类 token”，比如定义、约束、代码块开头或前文答案，而不是机械地只看最近 w 个 token。代价是 mask 生成本身也要高效，否则省下的 attention 计算会被 mask 计算吃掉。

### Slide 22：flash-dmattn 代码和论文

<img src="https://files.mdnice.com/user/59/f1dbfa63-415f-4141-bdda-8331dd32600c.png" referrerpolicy="no-referrer" />

flash-dmattn 仓库给出了 Triton/CuTe 实现，是这场分享里最明确的代码落点。slide 同时给了 GitHub 地址和 Alphaxiv 页面，说明这部分已经从论文/想法落到公开实现。

它支持 dense、sparse、gated、local、GQA/MQA，以及 sparse softmax threshold。后面代码拆解会重点看 `topk_mask`、`FlashGatedAttnFunc` 和 Triton forward：这些位置分别对应 mask 生成、可训练 gate 和 tile 级跳过计算。

### Slide 23：DMA vs NSA

<img src="https://files.mdnice.com/user/59/53789e96-7dbc-45cb-ac76-c8dbdd5a923f.png" referrerpolicy="no-referrer" />

DMA vs NSA 这页把两类稀疏注意力放在一起。NSA 更偏预设或结构化稀疏，DMA 则强调 dynamic mask：根据当前输入和可训练参数决定哪些 token 进入 attention。

slides 里更偏 DMA 的 trainable dynamic mask，代码里 `topk_mask` 和 gated attention 是关键入口。读这页时要把“省 FLOPs”和“保留关键 token”一起看：mask 越激进越省算，但 long-context retrieval 和通用 benchmark 可能掉得越快。

### Slide 24：top-w / delta mask 方法

<img src="https://files.mdnice.com/user/59/aee2a8a0-56ec-487a-82c9-c5d0fab62d01.png" referrerpolicy="no-referrer" />

这页给了 DMA mask 的公式。原始 attention 是 `softmax(QK^T / sqrt(d_head))V`。DMA 先定义一个额外的 bias 项 `delta = exp(softplus(VΔ)A)`，其中 `Δ` 是 head 维度里的可训练矩阵，`A` 是按 head 的可训练系数。然后从 `delta` 里选 top-w 个值，其余位置置成 `-inf`，再把 `delta` 展开到和 `QK^T` 同样大小，得到 refined attention：`softmax((QK^T + delta) / sqrt(d_head))V`。

右侧的 “Reduce RAM / skip computation” 对应两个收益。置成 `-inf` 的位置不需要参与 softmax 的有效计算，稀疏结构明确后，kernel 可以跳过这些块，减少中间矩阵和 HBM 读写。这里不是先算完整 attention 再裁剪结果，而是把 mask 提前放进 attention 计算路径里。

### Slide 25：实验设置

<img src="https://files.mdnice.com/user/59/e8b96842-ae20-4e58-aced-1871f16d0246.png" referrerpolicy="no-referrer" />

实验设置页列了 All Experimental Environments、Pre-training Corpus、Training Framework、Eval Framework for Perplexity Tasks、Eval Framework for Downstream Tasks。它在提醒读者：稀疏 attention 的结果必须放在统一训练和评测环境里看。

sparse attention 的收益必须和精度一起评估。只报速度会掩盖 needle 类任务的召回问题；只报 perplexity 又看不到真实 latency 是否下降。后面 Scaling、MQAR、Needle、General benchmark 四页就是按这个思路组织的。

### Slide 26：Scaling：更少 FLOPs

<img src="https://files.mdnice.com/user/59/4b67a1fc-607c-4444-93cb-a16ed902845a.png" referrerpolicy="no-referrer" />

这页标题是 Scaling Law 实验，图上写着 “DMA require fewer FLOPs than the standard MHA and NSA”。横轴是 FLOPs，纵轴是 perplexity，曲线对比 MHA、SWA、MLA、NSA 和 DMA。紫色 DMA 曲线整体低于红色 NSA 和蓝色 MHA，在相近 perplexity 下需要的 FLOPs 更少；绿色 MLA 曲线在这组实验里不占优。

底部小字写的是 “Maintaining the Pareto advantage under different parameters”。这句话对应稀疏 attention 的核心验证方式：不能只证明少算，还要证明少算以后 perplexity 仍然在更好的 Pareto 前沿上。DMA 的动态 mask 如果能在不同参数规模下维持这个关系，才说明它学到的是可迁移的稀疏模式。

### Slide 27：MQAR 速度

<img src="https://files.mdnice.com/user/59/3278b713-601c-43dd-ba5b-888364c8a1e8.png" referrerpolicy="no-referrer" />

MQAR 是 multi-query associative recall，适合测长上下文里的键值检索速度。图的横轴是 sequence length，从 1024 到 8192；纵轴是 speed(ms)。蓝色 MHA 在长序列处急剧变慢，8192 时接近 1700ms。SWA、NSA、DMA 都比 MHA 低很多，柱子上方的百分比可以理解成相对提速：在 4096 处 DMA 标到约 78.4%，在 8192 处约 87.0%。

页眉中间那句 “dynamic skipping is theoretical efficiency into a real-world reduction in latency” 是这页的重点。上一页证明 FLOPs 少，这页看实际 latency 是否下降。底部两条小字也在强调同一点：动态跳过把理论稀疏变成真实延迟降低，而且序列越长收益越明显。

### Slide 28：Needle-in-a-haystack

<img src="https://files.mdnice.com/user/59/485196bf-0873-4f91-b94c-ec6a300c8bac.png" referrerpolicy="no-referrer" />

Needle-in-a-haystack 是对稀疏 attention 的召回压力测试：把关键信息埋在长上下文不同深度，看模型能否找回。图里三块热力图分别是 MHA、Native Sparse Attention 和 Dynamic Mask Attention，横轴是 token limit，从 1K 到 16K；纵轴是 depth percent，从 0% 到 100%；颜色越接近绿色表示 score 越高。

白色虚线标在 8K 位置，可以理解成预训练最大上下文附近的边界。MHA 和 NSA 在 10K、12K、14K、16K 这些列会出现更多黄/橙色块，说明越过训练长度后召回不稳定；DMA 右侧仍然大面积保持绿色。底部小字说 “beyond the maximum context of pre-training” 仍能保持 retrieval accuracy，也就是 DMA 的动态 mask 没有把 needle 需要的远距离关联裁掉。

### Slide 29：通用 benchmark

<img src="https://files.mdnice.com/user/59/c3f66cfc-12cb-4afc-8031-49f165792633.png" referrerpolicy="no-referrer" />

这页表格把 DMA 放到通用 benchmark 里对比，列包括 Pile/Lambada perplexity、Lambada/MMLU/TriviaQA/ARC/PIQA/HellaSwag/OBQA/WinoGrande accuracy，以及 LongBench average。箭头说明 PPL 越低越好，ACC/AVG 越高越好。表格分 Zero-Shot 和 Five-Shot 两块，行里除了 MHA，还有 H2O、InfLLM、Quest、DAM、Exact-Top、NSA 和 DMA。

Zero-Shot 里 DMA 的 Pile PPL 是 45.12，LongBench Avg 是 16.2，MMLU、ARC、PIQA、WinoGrande 等列也有加粗项；Five-Shot 里 DMA 在 Lambada PPL、MMLU、PIQA、OBQA、WinoGrande 等列表现靠前。这里要看的不是某个单点最高，而是 DMA 在长上下文、perplexity 和常规能力之间没有明显偏科。长上下文优化如果只在合成任务上有效，实际价值会很有限。

### Slide 30：未来计划

<img src="https://files.mdnice.com/user/59/35184237-ea88-4ea6-8c68-ae5dd1cd8949.png" referrerpolicy="no-referrer" />

未来计划页是最后一章的分隔页。前面已经讲了 OpenSeek-Small 和 DMA，后面进入 OpenSeek-mid 10B 计划和三大支柱。

更大模型、更长训练、更复杂 attention 结构都会把系统优化放到台前。10B 规模虽然比百亿以上模型小，但已经足够验证数据效率、训练效率和结构效率是否能共同提升。

### Slide 31：OpenSeek-mid 10B 计划

<img src="https://files.mdnice.com/user/59/7e98f904-8122-4f31-a0c2-58b7d8e3a9f5.png" referrerpolicy="no-referrer" />

OpenSeek-mid 页把下一阶段拆成三列：Data Efficiency 是 3-4TB token，数据来源包括 CCI4.0/Decay/Midtraining；Training Efficiency 是用约 3B model initialize 10B；Structure Efficiency 是 DMA/NSA。

这个计划说明 sparse attention 不是独立实验，而是会进入模型训练 recipe。数据、初始化策略和结构效率一起改，才能判断 DMA/NSA 对真实模型训练是不是有稳定收益。

### Slide 32：三根支柱：数据、算法、系统

<img src="https://files.mdnice.com/user/59/c5c1ea41-bb47-4e23-a6f5-241d176ec5f1.png" referrerpolicy="no-referrer" />

三根支柱页把开源模型演进拆成 Data、Algo、System。Data 包括退火、合成；Algo 包括 RL 和模型结构；System 包括新结构支持和效率优化。

OpenSeek 的特点是不把系统当后处理。attention kernel、训练框架、推理效率和数据 recipe 同时开放，才有机会让系统贡献真正反馈到模型能力上。flash-dmattn 放在这篇文章里讲，就是因为它正好对应 System 这根支柱。

### Slide 33：社区邀请

<img src="https://files.mdnice.com/user/59/35b249c7-ea90-4182-9bae-befe81ebb450.png" referrerpolicy="no-referrer" />

社区邀请页很直接：开源模型要靠持续贡献。对工程贡献者来说，flash-dmattn、FlagScale、训练 config 都是可参与入口。

slide 上把贡献者分成三类：拥有稀缺数据的领域专家、擅长压榨算力的系统工程师、关注对齐哲学的研究者。对应到 OpenSeek 的三根支柱，就是 Data、System、Algo 都可以上船。系统侧最直接的切入点，就是 attention kernel、训练效率和推理效率。

### Slide 34：展台和联系方式

<img src="https://files.mdnice.com/user/59/55ff92bf-497c-44cb-9d1b-1a7af16a5a35.png" referrerpolicy="no-referrer" />

最后是展台和联系方式。这页没有新的技术增量，后面还是把重点放在 OpenSeek 的公开资产和 DMA 代码上。

# 0x3. 关键代码拆解

OpenSeek README 里 baseline 的公开方式比较完整：100B 数据、训练代码、wandb、checkpoint、eval 都列出来了。训练配置里还能看到 MoE router 的参数，例如 group top-k、router scaling 等。这类 config 对开源复现实验很重要。

slides 里 Dynamic Mask Attention 对应的公开代码在 `flash-dmattn`。先看模型层，`FlashSparseAttention` 比普通 attention 多了两个投影：`a_proj` 和 `d_proj`。Q/K/V 还是正常算，`alpha_states`、`delta_states` 则交给 gated sparse kernel 判断哪些 tile 该算：

```python
class FlashSparseAttention(nn.Module):
    def __init__(self, config, layer_idx=None):
        self.q_proj = nn.Linear(
            config.hidden_size, self.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.a_proj = nn.Linear(
            config.hidden_size, self.num_attention_heads, bias=False
        )
        self.d_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads, bias=False
        )
```

```python
def forward(self, hidden_states: torch.Tensor, **kwargs):
    query_states = self.q_proj(hidden_states).view(
        bsz, seq_len, self.num_attention_heads, self.head_dim
    )
    key_states = self.k_proj(hidden_states).view(
        bsz, seq_len, self.num_key_value_heads, self.head_dim
    )
    value_states = self.v_proj(hidden_states).view(
        bsz, seq_len, self.num_key_value_heads, self.head_dim
    )

    alpha_states = self.a_proj(hidden_states)
    delta_states = self.d_proj(hidden_states)

    attn_output = flash_gated_attn_func(
        query_states,
        key_states,
        value_states,
        alpha_states,
        delta_states,
        is_causal=self.is_causal,
        softmax_scale=self.scaling,
        softmax_threshold=self.softmax_threshold,
        gate_threshold=self.gate_threshold,
    )
```

这段代码可以解释 Slide 20/21：DMA 不是静态 sparse pattern。模型会根据当前 hidden states 产出 gate 相关的 `alpha/delta`，kernel 再用它们决定 sparse attention 的实际计算范围。

mask 路径先看 `topk_mask`。它根据 attention bias 选出每个 query 该看的 key：

```python
def topk_mask(attention_bias, attention_mask, window_size, min_dtype, block_size=None, **kwargs):
    attention_bias = attention_bias.detach()
    attention_bias = (
        attention_bias.masked_fill(~attention_mask, min_dtype)
        if attention_mask is not None
        else attention_bias
    )
    topk_values, topk_indices = torch.topk(
        attention_bias, window_size, dim=-1, largest=True, sorted=False
    )
    attention_mask = torch.zeros_like(
        attention_bias, dtype=torch.bool, device=attention_bias.device
    ).scatter_(-1, topk_indices, topk_values != min_dtype)

    if block_size is not None and block_size > 1:
        key_len = attention_mask.shape[-1]
        attention_mask = block_smooth(attention_mask, key_len, block_size)
    return attention_mask
```

`block_smooth` 会把 token mask 平滑成 block 级选择。这样做是为了贴近 GPU kernel 的 tile 访问，不然稀疏到单 token 粒度，kernel 很难高效。`create_mask` 则负责把普通二维 attention mask reshape 成 kernel 能消费的四维 mask，再根据 `type` 选择 top-k 或 relu mask：

```python
def create_mask(
    attention_bias: torch.Tensor,
    query_len: int,
    type: str = "topk",
    attention_mask: Optional[torch.Tensor] = None,
    window_size: Optional[int] = None,
    min_dtype: Optional[float] = None,
    block_size: Optional[int] = None,
) -> torch.Tensor:
    if min_dtype is None:
        min_dtype = torch.finfo(attention_bias.dtype).min

    if attention_mask is not None and attention_mask.dim() == 2:
        attention_mask = attention_mask[:, None, None, :]

    if type == "topk":
        return topk_mask(
            attention_bias,
            attention_mask,
            window_size,
            min_dtype,
            block_size=block_size,
        )
```

再往下是 PyTorch autograd wrapper。`FlashSparseAttnFunc.forward` 调用 Triton forward，保存 `query/key/value/out/lse` 给 backward；`FlashGatedAttnFunc` 额外保存 `alpha/delta`，因为 gate 本身也是可训练的：

```python
class FlashSparseAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, is_causal=False,
                softmax_scale=None, softmax_threshold=None, window_size=(None, None),
                is_split_kv=False, pack_gqa=False, return_lse=False):
        out, lse, softmax_scale, softmax_threshold = _flash_sparse_attn_base_forward(
            query=query,
            key=key,
            value=value,
            is_causal=False if query.shape[1] == 1 else is_causal,
            softmax_scale=softmax_scale,
            softmax_threshold=softmax_threshold,
            window_size=window_size,
            is_split_kv=is_split_kv,
            pack_gqa=pack_gqa,
        )

        ctx.save_for_backward(query, key, value, out, lse)
        ctx.softmax_scale = softmax_scale
        ctx.softmax_threshold = softmax_threshold
        ctx.window_size = window_size
        return out
```

```python
class FlashGatedAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, alpha, delta, is_causal=False,
                softmax_scale=None, softmax_threshold=None, gate_threshold=None,
                is_logsigmoid_gate=True, is_adapt_gate=True,
                window_size=(None, None), is_split_kv=False, pack_gqa=False,
                return_lse=False):
        out, lse, softmax_scale, softmax_threshold, gate_threshold = (
            _flash_gated_attn_base_forward(
                query=query,
                key=key,
                value=value,
                alpha=alpha,
                delta=delta,
                is_causal=False if query.shape[1] == 1 else is_causal,
                softmax_scale=softmax_scale,
                softmax_threshold=softmax_threshold,
                gate_threshold=gate_threshold,
                is_logsigmoid_gate=is_logsigmoid_gate,
                is_adapt_gate=is_adapt_gate,
                window_size=window_size,
                is_split_kv=is_split_kv,
                pack_gqa=pack_gqa,
            )
        )

        ctx.save_for_backward(query, key, value, alpha, delta, out, lse)
        return out
```

这一层包装的意义是把 DMA 做成正常 PyTorch 模块可以训练，而不是只提供一个 inference kernel。Slide 21 里说 trainable dynamic mask，落到代码就是 `alpha/delta` 走进 autograd，backward 里会继续回传到 `a_proj/d_proj`。

Triton sparse forward 里，先算 QK，再走 online sparse softmax。如果某个 block 贡献太小，`skip_softmax` 会让它不加载 V：

```python
acc_s = tl.dot(q_tile, k_tile)

p, block_max, row_max, row_sum, row_scale, skip_softmax = (
    activations.online_sparse_softmax(
        acc_s=acc_s,
        block_max=block_max,
        row_max=row_max,
        row_sum=row_sum,
        scale_log2=softmax_scale_log2,
        softmax_threshold_log2=softmax_threshold_log2,
        CHECK_INF=CHECK_INF,
    )
)

if not skip_softmax:
    v_tile = tl.load(v_ptrs, boundary_check=(0, 1), cache_modifier=".cg")
    acc_o = activations.rescale_o(acc_o, row_scale, LAZY_RESCALE=False)
    acc_o += tl.dot(p.to(v_tile.dtype), v_tile)
```

Gated attention 进一步在 tile 级判断是否需要算当前块。代码里先用 `online_gate` 估计 gate，再决定下一个 tile 是否跳过：

```python
gate_max, skip_gate_next = activations.online_gate(
    acc_s=acc_s,
    gate_max=gate_max,
    gate_threshold_log2=gate_threshold_log2,
)

if not skip_gate_next:
    acc_s += tl.dot(q_tile, k_tile)
```

这套实现和 slides 里的 O(N*w) 目标一致：mask/gate 让 attention 不必遍历所有 key，kernel 侧再把“跳过”落实为少加载 K/V、少做 softmax 和 V dot。论文或者 slides 里的“稀疏”如果停在算法图上，通常很容易漏掉这一层：真正省时间的地方不是生成了一个 mask，而是 Triton kernel 能在 tile 级少访存、少算 softmax、少算 `P @ V`。

# 0x4. 小结

OpenSeek 这篇可以看成一套开源训练工程记录：数据、recipe、评测和系统优化一起公开。DMA/flash-dmattn 是里面最具体的系统点，它把“长上下文省算力”从算法图落到四个位置：模型层的 `a_proj/d_proj`，mask 生成的 top-k/block smooth，autograd wrapper 的可训练 gate，Triton kernel 的 tile skip。
