> 推理网关不是简单转发层。LLM 请求的长短、cache 命中和后端 batch 状态都会改变成本，这篇主要看这些信息怎么进入路由决策。

# 0x0. 前言

这场分享很适合放在 infra 目录，因为它讲的是推理集群最外层的调度问题。模型服务做得再快，如果网关把长 prompt 路由到冷节点、把 decode 压到少数实例，TTFT 和吞吐都会被浪费掉。

# 0x1. 资料和代码落点

代码对应关系需要说清楚：Ant 内部 AI Gateway 的完整实现没有在公开仓库里找到，所以正文不会假装有同款 PR。公开代码里可以对上的有两类：

- Mooncake：`mooncake_connector_v1.py` 里的 `MooncakeConnectorScheduler` / `MooncakeConnectorWorker`，以及 `mooncake_store_service.py` 里的 `MooncakeStoreService`，对应 slides 里的 KVCache Store、PD 远程 prefill/decode、KV transfer。
- Envoy AI Gateway：`api/v1alpha1/ai_gateway_route.go`，对应云原生 AI Gateway 的路由 CRD、model header、token cost 元数据。
- Mooncake README 里提到 SGLang HiCache 和 PD 场景下 TTFT 降低的实践数据，可作为 KVCache 共享方向的公开参考。

# 0x2. Slides 逐页解读

### Slide 1：蚂蚁 AI 网关：大规模在线推理服务集群优化

<img src="https://files.mdnice.com/user/59/e6216fee-f2ff-4293-bd43-d27d0499e02b.png" referrerpolicy="no-referrer" />

标题页给出分享范围：蚂蚁 AI 网关在大规模在线推理服务集群里的性能优化。演讲人同时是 Mooncake 核心成员和 Envoy Golang Maintainer，这也解释了后面为什么会同时出现 KVCache Store、PD 路由和云原生网关。

这篇讨论的是推理网关，不是普通 HTTP 网关。LLM 请求的成本取决于 prompt 长度、decode 长度、cache 命中、PD 分离状态和后端 batch，因此路由层必须懂一些模型推理语义。

### Slide 2：目录：从负载特征到网关调度

<img src="https://files.mdnice.com/user/59/4ba5cbfb-765e-4c00-87ba-0cb7b1733acc.png" referrerpolicy="no-referrer" />

目录分三段：大规模推理集群的挑战、蚂蚁 AI 网关实践、未来演进。这个顺序很重要：先说明为什么 round-robin/least-connection 不够，再讲 v1/v2 怎么补负载和 cache 信号，最后才进入预测时延、Mooncake Store、PD Router 和云原生架构。

它的主线是让网关从“转发请求”升级成“推理集群调度器”。网关不再只看 HTTP 连接，而要把 input tokens、output tokens、prefix cache、prefill/decode 状态和多租 SLO 一起纳入决策。

### Slide 3：AI Gateway 在推理集群中的位置

<img src="https://files.mdnice.com/user/59/7c132901-9696-4cfa-bd14-00cdc146911e.png" referrerpolicy="no-referrer" />

这一页把 AI Gateway 定义成推理服务智能枢纽，右侧列了四类目标：智能路由、过载保护、多租 QoS、自动故障转移；上方两个收益是降低时延和提升吞吐。也就是说，它既要做入口治理，也要理解后端执行状态。

AI Gateway 位于用户/API 和推理后端之间。它要做鉴权、限流、路由、熔断，也要知道后端 SGLang/vLLM 实例的负载状态。对大模型来说，路由错一次可能带来几秒 TTFT 差异，尤其是长 prompt 被打到没有 prefix cache 的冷节点时。

### Slide 4：LLM 请求负载是非线性的

<img src="https://files.mdnice.com/user/59/257b8694-b4db-44de-89ce-1c3663fcd3df.png" referrerpolicy="no-referrer" />

这页标题是“负载与请求量的非线性关系”。左边列推理计算特征：计算量大、单点并发小；请求 input/output 变化很大、负载波动大；prefix 语义 cache 可复用。右边列经典算法不再适用：round-robin 只做请求数均衡，least-connection 只做并发数均衡。

LLM 负载不是请求数线性可加。一个 200 token prompt 和一个 20k token prompt 对 prefill 的压力完全不同；decode 阶段又按输出 token 持续占用 batch slot。再加上 prefix cache，两个“看起来一样空闲”的实例，对同一个请求的真实成本可能完全不同。

### Slide 5：Prefill 和 Decode 两个阶段

<img src="https://files.mdnice.com/user/59/fd9c63b7-0e23-4fe7-8673-028028f43fea.png" referrerpolicy="no-referrer" />

这页把推理过程拆成 Prefill 和 Decode。Prefill 的小字写的是 compute bound，几乎没有并发能力；Decode 则强调组 batch 和小并发。图的作用是提醒路由层：同一个请求在不同阶段的资源画像完全不同。

Prefill 主要吃长序列 attention 计算和 KV 写入，prompt 越长 TTFT 越明显；Decode 每步只生成少量 token，但会在 batch slot 里停留很多轮。PD 分离就是把这两种负载拆开，让不同节点承担不同阶段，并通过 KV transfer 把 prefill 结果交给 decode。

### Slide 6：Attention 与 FFN 的资源差异

<img src="https://files.mdnice.com/user/59/0fc0af91-eaf6-4037-8104-860dfb07a433.png" referrerpolicy="no-referrer" />

这页继续拆模型内部资源：Attention 的计算和访存都与 context length 正相关；FFN 的计算主要和 batch size 正相关，访存相对固定。也就是说，同一个 batch size 下，长上下文会让 attention 侧更重；同一个上下文长度下，大 batch 会让 FFN 侧更重。

这也是网关很难只看 QPS 的原因。后端实例可能请求数不多，但正在处理超长上下文 prefill；也可能连接数很多，但大部分是短 decode。负载指标如果不区分 context length、batch size 和阶段，就会把不同瓶颈混在一起。

### Slide 7：从 request 调度到 token 调度

<img src="https://files.mdnice.com/user/59/cd05a051-4f57-4eda-9ffe-0872ba806dfd.png" referrerpolicy="no-referrer" />

这页把智能路由本质写成 request 粒度和 token 粒度的协同优化。request 粒度主要影响时延和吞吐：请求选到哪个实例、排队多久、prefill 是否命中缓存；token 粒度主要影响 batch 组织和硬件利用率：decode 阶段每步要把哪些 token 合批。

请求级调度只决定“这个 prompt 去哪台机器”。token 级调度才决定 decode 阶段持续占用多少资源。服务端 batch scheduler 和网关路由是一体两面：网关把请求送到合适实例，实例内部 scheduler 再决定 token 如何进入 batch。

### Slide 8：传统负载均衡为什么不够

<img src="https://files.mdnice.com/user/59/f821c877-ec3e-443f-a1a9-03e5bf08ebbf.png" referrerpolicy="no-referrer" />

这页列了三个关键点：大模型计算过程、权衡、真实压测成本高。它是在解释为什么网关优化不能直接套经典负载均衡经验。LLM 的计算过程有阶段差异，路由时要在 cache 命中、当前负载、预期 decode 长度之间权衡；而真实线上压测成本高，很多策略不能靠反复试错调整。

传统 LB 的 round-robin 或 least-connection 对 LLM 不够用。两台实例连接数一样，但一台可能在处理长 prompt prefill，另一台只是短 decode；两台 queue 数一样，但其中一台可能有大段 prefix cache 可以复用。

### Slide 9：蚂蚁 AI 网关实践章节过渡

<img src="https://files.mdnice.com/user/59/d63be6a3-4ab2-46ba-998c-e00a5f74a16c.png" referrerpolicy="no-referrer" />

这一页是实践章节的过渡。前面已经说明 LLM 请求成本不是线性的，后面开始讲蚂蚁 AI 网关如何从 v1 的简单 queue-num，迭代到 v2 的 self-loop metrics 和 cache-aware，再到 v3 的预测时延。

这里也能看出优化路径：先把“负载是不是忙”看清楚，再把“这个请求在哪个节点能少算”看清楚，最后把多种信号统一成预测延迟。

### Slide 10：Ant AI Gateway 架构

<img src="https://files.mdnice.com/user/59/a5ff0c73-4b37-4876-afb7-d041a76a8e46.png" referrerpolicy="no-referrer" />

这页画的是 Ant AI Gateway 的基本位置：机房级入口、推理实例级路由、多租共享。机房级入口意味着它不是某个模型实例前的小代理，而是一个集群入口；推理实例级路由意味着它需要知道具体后端实例的状态；多租共享则要求它能处理不同租户的 SLO 和隔离。

Ant AI Gateway 的架构页把控制面和数据面分开：数据面接请求并路由，控制面收集后端 metrics、维护实例状态和路由策略。后面的 v1/v2/v3，本质上都是在丰富这套状态和策略。

### Slide 11：v1：轮询 metrics 和 queue-num score

<img src="https://files.mdnice.com/user/59/40cfe92e-cf12-4aee-811f-874b491e3e4d.png" referrerpolicy="no-referrer" />

v1 用定期轮询 metrics，score 直接取 `queue-num`，selection 是 `topK + random`。也就是先选队列较短的一组实例，再随机挑一个，避免所有请求都挤到同一个最优实例。

这个版本简单，适合先把系统跑起来，但它隐含了两个假设：queue 长度能代表未来延迟，cache 命中不影响路由。这两个假设在 LLM 场景都不稳定。长短请求混在一起时，queue 数相同不代表负载相同；prefix cache 存在时，冷节点不一定比热节点快。

### Slide 12：v1 的问题：指标滞后和缓存不敏感

<img src="https://files.mdnice.com/user/59/6cd6b488-ffcd-48c0-a772-b1c106d44e5c.png" referrerpolicy="no-referrer" />

v1 问题页分成指标采集和算法两侧。指标采集侧有三点：时效性差，多引擎适配成本高，定期采集会给引擎带来开销。算法侧也有三点：负载指标单一，长短请求干扰很大，没有 cache-aware。

这几个问题是连在一起的。metrics 采样间隔内负载可能已经变了，queue-num 又不知道某个请求是 100 token 还是 100k token；多引擎场景下，不同引擎暴露 metrics 的方式不一样，网关要统一采集也会变重。

### Slide 13：v2：self-loop metrics

<img src="https://files.mdnice.com/user/59/c12c90ef-4e62-461e-9618-d488411dcc17.png" referrerpolicy="no-referrer" />

v2 首先改指标采集，引入自闭环统计。slide 上列了两个核心指标：未完成请求数和 prefill 长度。未完成请求数比轮询 queue 更贴近数据面当前状态；prefill 长度则把长 prompt 的负载显式纳入路由。

这样做的收益也写在 slide 上：时效性和 prefill 负载。相比纯控制面轮询，自闭环统计更接近请求路径，也更适合短时间窗口的负载估计。它仍然不是完整预测模型，但已经能避免很多“队列短但正在跑超长 prefill”的误判。

### Slide 14：cache-aware prefix tree

<img src="https://files.mdnice.com/user/59/658a8974-abb0-48a9-a39a-4c994a6abd57.png" referrerpolicy="no-referrer" />

cache-aware prefix tree 是这场分享里最贴近 LLM 推理语义的一页。图里左侧是 Metadata-center，里面维护近似 Radix-Tree；AI 网关在 `1. LB 选择前` 会向 Metadata-center `查询 cache`，拿到候选实例的前缀命中情况。请求被引擎处理到 `3. 响应首 token` 后，网关/引擎再把新产生的 cache 信息 `save cache` 回 Metadata-center，为下一次相似 prompt 服务。

右侧三条小字说明实现细节：先按文本切 chunk，再对 chunk 计算哈希；前缀树不是无限增长，需要做近似淘汰；多模态输入不能直接把文本 hash 逻辑套上去，图片、音频这类连续特征或者特殊 token 需要额外处理。也就是说，网关维护的不是完整 prompt 原文，而是能估算“这个请求在某个实例上能复用多少 KV”的元数据。

### Slide 15：score = cache_ratio - request_load - prefill_load

<img src="https://files.mdnice.com/user/59/1fece6d3-1e23-426f-8c48-a9789cd9e4f9.png" referrerpolicy="no-referrer" />

这页给出 v2 的多因子打分：`score = W1*cache_ratio - W2*request_load - W3*prefill_load`。`cache_ratio` 是 prefix cache 命中率，越高越应该选；`request_load` 是请求队列数，越高说明排队压力越大；`prefill_load` 是当前处于 prefill 阶段的 prompt 长度，越高说明实例正在吃长上下文 prefill。

这个公式更像工程排序项：在网关侧把三类信号合到同一个分数里。只看 cache 会把请求都打到热节点，只看 queue 又会错过大段 KV 复用；加上 prefill_load 后，长 prompt 正在跑的实例会被降权，避免 TTFT 被前面的 prefill 拖住。

### Slide 16：v2 优化效果

<img src="https://files.mdnice.com/user/59/15f0b99c-4599-41cc-99d6-3e8cfb0f9a47.png" referrerpolicy="no-referrer" />

v2 结果页给了三条收益：KVCache 命中率提升一倍，TTFT 平均值降低 50%，TTFT 长尾数量级降低。这里 TTFT 收益来自两个方向：命中 prefix cache 后少做 prefill；长 prompt 不再频繁被路由到冷节点。

长尾数量级降低尤其重要。平均值降低说明整体更快，但线上用户更容易感知的是 P99/P999 的等待。cache-aware routing 把可复用前缀尽量送到已有 KV 的实例，能显著减少“某些请求突然慢很多”的情况。

### Slide 17：在线场景的稳定性约束

<img src="https://files.mdnice.com/user/59/e4db108b-8579-4b78-865a-e5679047214f.png" referrerpolicy="no-referrer" />

这一页是未来演进的过渡。前面 v2 已经利用了自闭环 metrics 和近似前缀树，但它仍然是启发式打分。未来演进要解决的是更精确的延迟预测、更准确的 cache-aware、更复杂的 PD/EP/DP 分层路由。

在线系统不能只看平均延迟，还要看抖动、故障恢复和多租优先级。网关策略如果过度追逐 cache，可能把少数热节点压爆；如果过度追逐负载，又会牺牲 cache 复用，导致长 prompt 反复 prefill。

### Slide 18：v2 仍然解决不了的地方

<img src="https://files.mdnice.com/user/59/5f758ffe-14a0-4629-9046-fb49fd415959.png" referrerpolicy="no-referrer" />

v2 问题页也分指标和算法两侧。指标侧缺少 decode context length，而且 cache-aware 还是近似，准确度约 80%；算法侧的问题是权重参数寻优难、可解释性差、无法实现优先级调度。底部公式把 decode_load 也加进去：`W1*cache_ratio - W2*request_load - W3*prefill_load - W4*decode_load`。

这说明 v2 的剩余问题在于 score 仍然是启发式。真实延迟受模型、batch、KV、decode 长度、PD 链路影响，手调 W1/W2/W3/W4 很难覆盖所有场景。多租 SLO 下还会出现“不是选最快节点，而是选满足 SLO 且资源更合适的节点”的需求。

### Slide 19：Latency prediction

<img src="https://files.mdnice.com/user/59/c9ce8fd6-6265-412f-bdc4-324be6e6b1b7.png" referrerpolicy="no-referrer" />

v3 这页把启发式打分改成预测建模。上半部分先做指标采集：prefill 阶段采 `input-length & cache-ratio`，decode 阶段采 `batch-size & context-length`。中间预测器分三项建模：TTFT、TPOT 和 Output Length。下方算法再基于预测时延做选择，同时支持多租 SLO 筛选。

右侧几条话说明它和 v2 的区别：多因子最后归一化为时延，路由目标从“分数最高”变成“预测时延最合适”；多租 SLO 场景下，不一定选全局最快节点，而是先过滤出满足 SLO 的候选，再在候选里做负载和 cache 权衡。关键点是预测准确度，预测器偏差太大时，路由会把错误放大到整个集群。

### Slide 20：Mooncake Store 和 KVCache 共享

<img src="https://files.mdnice.com/user/59/d87aa43b-f644-475c-aac2-9fdd3e1cb7b5.png" referrerpolicy="no-referrer" />

Mooncake Store 这页是在讲更精确的 cache-aware。流程从 `1. 推理请求` 进入 AI 网关开始；网关根据请求内容 `2. 生成 KVCache key`，再向 Mooncake `3. 查询 KVCache`；如果 store 里能找到对应 KV，引擎就可以在 `4. 推理请求` 时把 KV 拉进来复用。右侧两条收益分别是提升 KVCache 本地命中率，以及降低 KVCache 传输带宽和耗时。

公开 Mooncake 代码里有 KV transfer connector 和 store service，能把远端 prefill 生成的 KV 拉到 decode 节点，避免重复 prefill。和 v2 的前缀树相比，Mooncake Store 更像把 cache 元数据和 cache 数据都纳入系统：网关不只是猜哪个节点可能命中，也可以借助 store 让 KV 在节点之间流动。

### Slide 21：TTFT 与 TPOT 的建模

<img src="https://files.mdnice.com/user/59/2bbb5658-9f9f-49d2-8dc6-f68a224c3089.png" referrerpolicy="no-referrer" />

这页标题是“时延预测模型”，左侧直接给出两个建模假设：TTFT 用二次关系，TPOT 用分阶段线性。左下图的横轴是 input token length，纵轴是 Time to First Token；蓝点是单次测量，红点是均值和标准差，绿色拟合曲线明显向上弯，说明 prompt 变长后 prefill 代价不是线性增加。它会受到 attention 计算、cache 命中率、batching 和排队共同影响，所以用二次项近似更稳。

右上图是 TPOT vs Batchsize，蓝点随 batch size 增大整体上升，红线是拟合曲线。右下图是 TPOT vs Total Tokens，点非常密，红线斜率较小，说明 decode 每 token 延迟更像“批大小、上下文长度、系统负载”叠加后的分段线性关系。路由侧把 TTFT 和 TPOT 拆开预测，就能区分“prefill 很重但 decode 可接受”和“decode 队列已经很满”这两种情况。

### Slide 22：PD Router 和 DP imbalance

<img src="https://files.mdnice.com/user/59/a0783f4a-28ab-4a3f-873b-5824606c53e6.png" referrerpolicy="no-referrer" />

这页讲分离式推理后路由层的形态。左图是三层结构：最上面 `Global Router` 先做全局入口决策，中间 `PD Router` 决定 prefill/decode 的分配，下面 `DP LB` 再在 data parallel 粒度做负载均衡。右侧第一条写“PD 分离催生 PD Router”，意思是 prefill 和 decode 拆开后，请求不再只需要挑一个实例，而是要决定 prefill 放哪、decode 放哪、KV 怎么传。

第二条是“大 EP 伴随大 DP 产生 DP imbalance 问题”。MoE 里 EP 变大后，不同 DP rank 上的 expert 命中和请求长度都可能不均衡，单纯按实例级 queue 长度做 LB 会掩盖 rank 内部的不平衡。下面两条“简化了负载均衡策略”和“DP 粒度的最优决策”就是这个矛盾：分层决策更容易工程落地，但最优调度可能需要统一看 PD、DP、KV 传输和 cache 状态。

### Slide 23：云原生 Go 扩展

<img src="https://files.mdnice.com/user/59/a869b709-9cce-4100-9236-da3462f12811.png" referrerpolicy="no-referrer" />

这页把 AI 网关落到云原生架构。左上控制面接 `k8s Gateway API` 和 `Service/Inference Pool`，说明模型入口、服务池和路由规则希望用 Kubernetes 资源表达。左下数据面是同进程实例，底座是 `envoy + Golang`；中间插件机制列了九类能力：指标采集、均衡算法、认证鉴权、限流、预测模型、Trace、过载保护、多租 SLO、可观测。

右侧单独画了负载指标链路：`Metadata-center` 和 Mooncake 提供 cache/负载元数据，网关插件再把这些指标接入路由决策。旁边三组 bullet 分别对应工程取舍：Golang 扩展的好处是灵活、维护成本低；弱引擎依赖意味着可以水平扩展网关层，不把所有逻辑塞进推理引擎；云原生底座则让模型池、路由和限流能用标准抽象管理。

### Slide 24：总结

<img src="https://files.mdnice.com/user/59/b406a783-38e3-40c7-a04d-4a916f80bcbf.png" referrerpolicy="no-referrer" />

最后这页可以总结成一句：大模型网关要懂请求形态、cache 位置和后端执行状态。否则它只是 HTTP 转发器，帮不了推理集群省算力。

# 0x3. 关键代码拆解

Mooncake 的 vLLM connector 很适合解释 slides 里的 PD/KVCache 共享。`MooncakeConnectorScheduler` 侧先判断这个请求是否要从远端加载 prefill KV：

```python
def get_num_new_matched_tokens(self, request, num_computed_tokens):
    params = request.kv_transfer_params
    if params is not None and params.get("do_remote_prefill"):
        count = len(request.prompt_token_ids) - num_computed_tokens
        if count > 0:
            return count, True
    return 0, False
```

分配 KV blocks 后，connector 会把 request 记录到待接收队列，worker 侧随后异步拉取：

```python
if params.get("do_remote_prefill"):
    local_block_ids = (blocks.get_unhashed_block_ids()
                       if num_external_tokens > 0 else [])
    self._reqs_need_recv[request.request_id] = (request, local_block_ids)
    params["do_remote_prefill"] = False
```

请求在 producer 侧结束时，如果要交给 decode 节点继续跑，会返回一段新的 transfer params：

```python
return delay_free_blocks, dict(
    do_remote_prefill=True,
    do_remote_decode=False,
    remote_host=self.side_channel_host,
    remote_port=self.side_channel_port,
)
```

Mooncake Store Service 则把 store 操作暴露成 REST API。`/api/reconfigure` 可以在 prefill/decode 模式间切换，decode 模式会 mount 共享内存段：

```python
app.add_routes([
    web.post('/api/reconfigure', _timed_handler("RECONFIGURE", self.handle_reconfigure)),
    web.post('/api/mount_shm', _timed_handler("MOUNT_SHM", self.handle_mount_shm)),
    web.post('/api/unmount_shm', _timed_handler("UNMOUNT_SHM", self.handle_unmount_shm)),
    web.put('/api/put', _timed_handler("PUT", self.handle_put)),
    web.get('/api/get/{key}', _timed_handler("GET", self.handle_get)),
])
```

```python
if mode == "decode":
    result = self.store.mount_segment(path, size, offset, protocol, location)
    self.mounted_segment_ids = list(result["segment_ids"])
    self.current_mode = "decode"
elif mode == "prefill":
    if self.mounted_segment_ids:
        ret = self.store.unmount_segment(self.mounted_segment_ids)
    self.current_mode = "prefill"
```

云原生部分可以看 Envoy AI Gateway 的 CRD。它把模型路由和 token 成本写进 Kubernetes 对象：

```go
type AIGatewayRouteSpec struct {
    ParentRefs []gwapiv1.ParentReference `json:"parentRefs,omitempty"`
    Rules []AIGatewayRouteRule `json:"rules"`
    LLMRequestCosts []LLMRequestCost `json:"llmRequestCosts,omitempty"`
}
```

注释里提到 `x-ai-eg-model` header 会从请求体里抽取 model name 后用于路由匹配。这和 slides 里的“AI 网关不是普通七层网关”是同一个方向：数据面需要理解 OpenAI-compatible 请求，而不是只看 URL。

# 0x4. 小结

这篇的重点是路由层开始理解 LLM 推理。cache-aware routing、latency prediction、PD/KV transfer 都是在回答一个问题：这个请求发到哪里，才能少算、少等、少抖。Mooncake 给了公开的 KVCache 共享实现，Envoy AI Gateway 给了云原生路由 CRD 的参考。
