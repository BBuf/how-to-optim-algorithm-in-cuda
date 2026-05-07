> 这篇是 GOSIM Hangzhou 里 `OpenSeek: 开源驱动的下一代AI模型` 这场分享的回放解读。原始 PDF 已经按页转成 mdnice 图片，正文里每一页 slides 都保留了对应图片；技术页我会尽量落到公开代码，讲清楚这页到底对应什么实现。

# 0x0. 前言

OpenSeek 不是单个模型发布，而是一个围绕数据、算法和系统的开源协作计划。slides 里有不少项目层面的信息，但技术上最值得展开的是 Dynamic Mask Attention 和 flash-dmattn。

# 0x1. 资料和代码落点

相关资料和源码：

- OpenSeek：`README.md`、`configs/OpenSeek-Small-v1-Baseline/train/train_deepseek_v3_1_4b.yaml`，对应 baseline、数据、checkpoint、训练配置。
- flash-dmattn：`flash_sparse_attn/utils/mask.py`，对应 top-k/relu dynamic mask。
- flash-dmattn：`flash_sparse_attn/ops/triton/flash_sparse_fwd.py` 和 `flash_gated_fwd.py`，对应 sparse softmax skip 和 gated attention kernel。
- slides 中给出的论文入口是 Trainable Dynamic Mask Sparse Attention，代码仓库是 `https://github.com/SmallDoges/flash-dmattn`。

# 0x2. Slides 逐页解读

### Slide 1：OpenSeek：开源驱动的下一代 AI 模型

![](https://files.mdnice.com/user/59/2266a825-3c84-42b5-a12d-729d1ff9cf75.png)

OpenSeek 这场分享偏项目和算法路线，但里面有一个很具体的系统点：DMA/flash-dmattn。文章会先按 slides 讲项目，再把 sparse attention 代码落下来。

### Slide 2：目录：项目、数据算法、效率和未来

![](https://files.mdnice.com/user/59/03e0f5ef-6ef9-4b60-bd5b-2d4943829196.png)

目录把 OpenSeek 拆成项目机制、数据算法、效率优化和未来。和单模型发布不同，它更像一个开源协作计划，试图把数据、训练配置、评测和系统优化都打开。

### Slide 3：OpenSeek 项目介绍

![](https://files.mdnice.com/user/59/6e3fec09-fb69-4273-a42a-63faf3ec8c48.png)

项目介绍页强调 BAAI 发起和开源协同。我的理解是，OpenSeek 想补齐 DeepSeek 之后社区里“有想法但缺完整 pipeline”的问题。

### Slide 4：忒修斯之船：工程范式的替换

![](https://files.mdnice.com/user/59/9489f7fd-3273-4b71-857c-7714d6e25aa7.png)

忒修斯之船的比喻指向工程范式：模型不是一个固定物体，而是数据、算法、系统、评测持续替换后的结果。开源项目要让这些替换可复现。

### Slide 5：开源社区作为船坞

![](https://files.mdnice.com/user/59/ae6cd900-357d-4518-bb43-6a2739f18728.png)

开源社区作为船坞，意味着每个组件都有人能换、能测、能比较。没有公开数据和训练脚本，算法改进很难被别人接上。

### Slide 6：累计创新和竞赛机制

![](https://files.mdnice.com/user/59/c64ae432-3689-4785-ad02-64817eb6db5b.png)

累计创新和竞赛机制这页说明 OpenSeek 不只收论文点子，也希望用 leaderboard/competition 让改动有统一评估。

### Slide 7：OpenSeek Working Groups

![](https://files.mdnice.com/user/59/db58f485-27b0-4c2e-a480-5308d61950e8.png)

Working Groups 把工作拆成数据、算法、系统等组。大模型开源如果没有分工，很容易变成一个 repo 里堆 issue。

### Slide 8：非线性跃迁

![](https://files.mdnice.com/user/59/904713b1-b6eb-4f7b-9eee-587def83a228.png)

非线性跃迁讲的是开源协作的复利：一次数据清洗、一次 attention kernel、一次训练稳定性改进，单独看都小，叠起来可能改变模型能力曲线。

### Slide 9：开源竞赛和协作

![](https://files.mdnice.com/user/59/1b7fd886-d90a-45db-91e7-7108b5d474bd.png)

竞赛机制的价值在于给贡献者明确目标。比如数据 mix、annealing schedule、RL recipe、sparse attention，都可以用统一 baseline 比较。

### Slide 10：OpenSeek 时间线

![](https://files.mdnice.com/user/59/96afe9b9-40cc-4216-af9d-f9180f828a49.png)

时间线展示了 OpenSeek 从 meetup、baseline 到数据和模型发布的节奏。README 也能看到 CCI4.0-M2、OpenSeek-Small v1 和 100B baseline。

### Slide 11：数据与算法：Annealing + RL

![](https://files.mdnice.com/user/59/4fde32cc-8091-49dc-bfe7-35c31d941c16.png)

数据算法页提到 annealing + RL。OpenSeek-Small 的路线不是只堆 pretrain tokens，而是在 mid-training 和 post-training 里调整数据分布和目标。

### Slide 12：OpenMDW 和 OpenSeek-Small

![](https://files.mdnice.com/user/59/3e707147-c692-4db7-b933-74c2965e6160.png)

OpenMDW 和 HuggingFace 链接对应公开数据/模型资产。真正有用的是 checkpoint、wandb、config、eval 都放出来，方便别人定位某个改动的收益。

### Slide 13：效率方向

![](https://files.mdnice.com/user/59/f65a8b48-4763-4088-b1c3-4b88887e0ffa.png)

效率方向进入本文代码重点。长上下文 attention 的 O(N^2) 成本会限制训练和推理，DMA/NSA 这类 sparse attention 希望把无效 token 的计算跳掉。

### Slide 14：训练方法总览

![](https://files.mdnice.com/user/59/7fa673c9-031f-47f6-a71e-aa21db73ced2.png)

训练方法总览把 pretrain、mid-training、post-training 串起来。OpenSeek-Small 的 config 里可以看到 MoE、router、group top-k 等训练配置。

### Slide 15：Mid-training 两阶段

![](https://files.mdnice.com/user/59/749964be-2da7-4959-8b7d-6e379cc018fa.png)

Mid-training 两阶段：先稳定 200B math tokens，再做 20B decay。这个设计更像把能力强化和分布回调分开，避免只在数学域上越训越偏。

### Slide 16：Post-training：SFT + GRPO

![](https://files.mdnice.com/user/59/13107511-8346-4435-8190-f6071bcf137e.png)

Post-training 用 SFT 数据压缩和 GRPO。slides 提到 Infinity-Instruct-core 1.4M 达到 7M 数据 95.7% 效果，这说明数据筛选比盲目加量更重要。

### Slide 17：OpenSeek-Small 结果

![](https://files.mdnice.com/user/59/6007a647-f017-47f1-8e55-1ad274993268.png)

结果页展示 OpenSeek-Small 的 benchmark。作为 baseline，最重要的是给后续贡献一个参照物，而不是宣称一个终局模型。

### Slide 18：训练曲线

![](https://files.mdnice.com/user/59/43e15568-2f6b-4165-811c-d9c6a013a6a4.png)

训练曲线能帮助判断 recipe 是否稳定。大模型开源如果只给最终分数，不给训练曲线，外部贡献者很难知道自己的改动是在早期收敛还是后期泛化上起作用。

### Slide 19：Attention 演进

![](https://files.mdnice.com/user/59/bda5de41-896f-48f6-a1c7-442e903f765d.png)

Attention 演进页从 dense 到 local、sparse、dynamic mask。长上下文里，真正难的是 mask 既要省算力，又不能丢掉关键 token。

### Slide 20：Dynamic Mask Attention

![](https://files.mdnice.com/user/59/fc766803-efc7-4903-a326-fcd425ceef33.png)

Dynamic Mask Attention 的目标是把复杂度从 O(N^2) 往 O(N*w) 靠，其中 w 是动态选择的窗口或 token 数。

### Slide 21：Trainable Dynamic Mask Sparse Attention

![](https://files.mdnice.com/user/59/e99130af-10ce-486e-980c-76c30edb320f.png)

Trainable Dynamic Mask Sparse Attention 让 mask 可训练，而不是固定 local window。模型可以学会哪些历史 token 更值得看。

### Slide 22：flash-dmattn 代码和论文

![](https://files.mdnice.com/user/59/f1dbfa63-415f-4141-bdda-8331dd32600c.png)

flash-dmattn 仓库给出了 Triton/CuTe 实现，是这场分享最硬的代码落点。它支持 dense/sparse/gated/local/GQA/MQA，以及 sparse softmax threshold。

### Slide 23：DMA vs NSA

![](https://files.mdnice.com/user/59/53789e96-7dbc-45cb-ac76-c8dbdd5a923f.png)

DMA vs NSA 的差异可以粗略理解为 mask 生成和稀疏结构不同。slides 里更偏 DMA 的 trainable dynamic mask，代码里 `topk_mask` 和 gated attention 是关键入口。

### Slide 24：top-w / delta mask 方法

![](https://files.mdnice.com/user/59/aee2a8a0-56ec-487a-82c9-c5d0fab62d01.png)

top-w / delta mask 方法会根据 attention bias 选择 top-k 或阈值位置，再做 block smoothing。这样 mask 不只是 token 级随机稀疏，也能更贴近 kernel block。

### Slide 25：实验设置

![](https://files.mdnice.com/user/59/e8b96842-ae20-4e58-aced-1871f16d0246.png)

实验设置页交代模型、上下文和任务。sparse attention 的收益必须和精度一起看，只报速度会掩盖 needle 类任务的召回问题。

### Slide 26：Scaling：更少 FLOPs

![](https://files.mdnice.com/user/59/4b67a1fc-607c-4444-93cb-a16ed902845a.png)

Scaling 结果说明在更少 FLOPs 下保持能力，是 sparse attention 真正想证明的东西。否则只是把 attention 算错得更快。

### Slide 27：MQAR 速度

![](https://files.mdnice.com/user/59/3278b713-601c-43dd-ba5b-888364c8a1e8.png)

MQAR 是多 query associative recall，适合测长上下文键值检索。DMA 如果能在这里提速且不掉太多精度，说明 mask 没有把关键关联裁掉。

### Slide 28：Needle-in-a-haystack

![](https://files.mdnice.com/user/59/485196bf-0873-4f91-b94c-ec6a300c8bac.png)

Needle-in-a-haystack 更直观：长上下文里把关键句埋进去，看模型能不能找回。动态 mask 要在这里经受压力测试。

### Slide 29：通用 benchmark

![](https://files.mdnice.com/user/59/c3f66cfc-12cb-4afc-8031-49f165792633.png)

通用 benchmark 用来确认 sparse attention 没把模型泛化能力打坏。长上下文优化如果只在合成任务上有效，实际价值会打折。

### Slide 30：未来计划

![](https://files.mdnice.com/user/59/35184237-ea88-4ea6-8c68-ae5dd1cd8949.png)

未来计划进入 OpenSeek-mid。更大模型、更长训练、更复杂 attention 结构都会把系统优化放到台前。

### Slide 31：OpenSeek-mid 10B 计划

![](https://files.mdnice.com/user/59/7e98f904-8122-4f31-a0c2-58b7d8e3a9f5.png)

OpenSeek-mid 约 10B、3-4TB token、3B init 10B、DMA/NSA 是下一阶段重点。这个计划说明 sparse attention 不是独立实验，而是会进入模型训练 recipe。

### Slide 32：三根支柱：数据、算法、系统

![](https://files.mdnice.com/user/59/c5c1ea41-bb47-4e23-a6f5-241d176ec5f1.png)

三根支柱是数据、算法、系统。OpenSeek 的特点就是不把系统当后处理，attention kernel、训练框架和数据 recipe 同时开放。

### Slide 33：社区邀请

![](https://files.mdnice.com/user/59/35b249c7-ea90-4182-9bae-befe81ebb450.png)

社区邀请页很直接：开源模型要靠持续贡献。对工程贡献者来说，flash-dmattn、FlagScale、训练 config 都是可参与入口。

### Slide 34：展台和联系方式

![](https://files.mdnice.com/user/59/55ff92bf-497c-44cb-9d1b-1a7af16a5a35.png)

最后是展台和联系方式。文章里就不展开活动信息了，重点还是把 OpenSeek 的公开资产和 DMA 代码讲清楚。

# 0x3. 关键代码拆解

OpenSeek README 里 baseline 的公开方式比较完整：100B 数据、训练代码、wandb、checkpoint、eval 都列出来了。训练配置里还能看到 MoE router 的参数，例如 group top-k、router scaling 等。这类 config 对开源复现实验很重要。

DMA 的代码核心先看 mask。`topk_mask` 根据 attention bias 选出每个 query 该看的 key：

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
        attention_mask = block_smooth(attention_mask, key_len, block_size)
    return attention_mask
```

`block_smooth` 会把 token mask 平滑成 block 级选择。这样做是为了贴近 GPU kernel 的 tile 访问，不然稀疏到单 token 粒度，kernel 很难高效。

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

这套实现和 slides 里的 O(N*w) 目标一致：mask/gate 让 attention 不必遍历所有 key，kernel 侧再把“跳过”落实为少加载 K/V、少做 softmax 和 V dot。

# 0x4. 小结

OpenSeek 这篇我更愿意看成开源训练工程的宣言：数据、recipe、评测和系统优化一起公开。DMA/flash-dmattn 是其中最具体的技术点，它把“长上下文省算力”从论文假设落到了 mask 生成和 Triton kernel。
