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

这篇讨论的是推理网关，不是普通 HTTP 网关：LLM 请求的成本取决于 prompt 长度、decode 长度、cache 命中、PD 分离状态和后端 batch，因此路由层必须懂一些模型推理语义。

### Slide 2：目录：从负载特征到网关调度

<img src="https://files.mdnice.com/user/59/4ba5cbfb-765e-4c00-87ba-0cb7b1733acc.png" referrerpolicy="no-referrer" />

目录从负载特征开始，再讲 Ant 网关两代调度策略，最后进入 latency prediction、Mooncake Store 和云原生扩展。它的主线是：让网关从“转发请求”升级成“推理集群调度器”。

### Slide 3：AI Gateway 在推理集群中的位置

<img src="https://files.mdnice.com/user/59/7c132901-9696-4cfa-bd14-00cdc146911e.png" referrerpolicy="no-referrer" />

AI Gateway 位于用户/API 和推理后端之间。它要做鉴权、限流、路由、熔断，也要知道后端 SGLang/vLLM 实例的负载状态。对大模型来说，路由错一次可能带来几秒 TTFT 差异。

### Slide 4：LLM 请求负载是非线性的

<img src="https://files.mdnice.com/user/59/257b8694-b4db-44de-89ce-1c3663fcd3df.png" referrerpolicy="no-referrer" />

LLM 负载不是请求数线性可加。一个 200 token prompt 和一个 20k token prompt 对 prefill 的压力完全不同；decode 阶段又按输出 token 持续占用 batch slot。

### Slide 5：Prefill 和 Decode 两个阶段

<img src="https://files.mdnice.com/user/59/fd9c63b7-0e23-4fe7-8673-028028f43fea.png" referrerpolicy="no-referrer" />

Prefill 主要吃 attention 的长序列计算和 KV 写入，decode 每步短但持续时间长。PD 分离就是把这两种负载拆开，让不同节点承担不同阶段。

### Slide 6：Attention 与 FFN 的资源差异

<img src="https://files.mdnice.com/user/59/0fc0af91-eaf6-4037-8104-860dfb07a433.png" referrerpolicy="no-referrer" />

Attention 和 FFN 的资源画像也不同。Attention 更受 KV cache、序列长度和带宽影响；FFN/MoE 更受矩阵乘、expert 路由影响。网关如果只看 QPS，很难判断哪个节点真的忙。

### Slide 7：从 request 调度到 token 调度

<img src="https://files.mdnice.com/user/59/cd05a051-4f57-4eda-9ffe-0872ba806dfd.png" referrerpolicy="no-referrer" />

请求级调度只决定“这个 prompt 去哪台机器”。token 级调度才决定 decode 阶段持续占用多少资源。服务端 batch scheduler 和网关路由其实是一体两面。

### Slide 8：传统负载均衡为什么不够

<img src="https://files.mdnice.com/user/59/f821c877-ec3e-443f-a1a9-03e5bf08ebbf.png" referrerpolicy="no-referrer" />

传统 LB 的 round-robin 或 least-connection 对 LLM 不够用。两台实例连接数一样，但一台可能在处理长 prompt prefill，另一台只是短 decode。

### Slide 9：Prefix cache 改变路由目标

<img src="https://files.mdnice.com/user/59/d63be6a3-4ab2-46ba-998c-e00a5f74a16c.png" referrerpolicy="no-referrer" />

Prefix cache 让路由目标变成“命中最多缓存的节点”。如果一个 prompt 前缀在某节点已有 KV，哪怕它当前稍忙，也可能比空闲节点重新 prefill 更快。

### Slide 10：Ant AI Gateway 架构

<img src="https://files.mdnice.com/user/59/a5ff0c73-4b37-4876-afb7-d041a76a8e46.png" referrerpolicy="no-referrer" />

Ant AI Gateway 的架构页把控制面和数据面分开：数据面接请求并路由，控制面收集后端 metrics、维护实例状态和路由策略。

### Slide 11：v1：轮询 metrics 和 queue-num score

<img src="https://files.mdnice.com/user/59/40cfe92e-cf12-4aee-811f-874b491e3e4d.png" referrerpolicy="no-referrer" />

v1 用后端轮询 metrics，score 近似取 queue-num。这个版本简单，但它隐含了两个假设：queue 长度能代表未来延迟，cache 命中不影响路由。这两个假设在 LLM 场景都不太稳。

### Slide 12：v1 的问题：指标滞后和缓存不敏感

<img src="https://files.mdnice.com/user/59/6cd6b488-ffcd-48c0-a772-b1c106d44e5c.png" referrerpolicy="no-referrer" />

v1 的问题来自滞后和粗粒度。metrics 采样间隔内负载可能已经变了；而且 queue-num 不知道某个请求是 100 token 还是 100k token。

### Slide 13：v2：self-loop metrics

<img src="https://files.mdnice.com/user/59/c12c90ef-4e62-461e-9618-d488411dcc17.png" referrerpolicy="no-referrer" />

v2 引入 self-loop metrics，让网关数据面更快看到请求执行反馈。相比纯控制面轮询，它更接近请求路径，也更适合短时间窗口的负载估计。

### Slide 14：cache-aware prefix tree

<img src="https://files.mdnice.com/user/59/658a8974-abb0-48a9-a39a-4c994a6abd57.png" referrerpolicy="no-referrer" />

cache-aware prefix tree 是这场分享最有 LLM 味的一页。图里左侧是 Metadata-center，里面维护近似 Radix-Tree；AI 网关在 `1. LB 选择前` 会向 Metadata-center `查询 cache`，拿到候选实例的前缀命中情况。请求被引擎处理到 `3. 响应首 token` 后，网关/引擎再把新产生的 cache 信息 `save cache` 回 Metadata-center，为下一次相似 prompt 服务。

右侧三条小字说明实现细节：先按文本切 chunk，再对 chunk 计算哈希；前缀树不是无限增长，需要做近似淘汰；多模态输入不能直接把文本 hash 逻辑套上去，图片、音频这类连续特征或者特殊 token 需要额外处理。也就是说，网关维护的不是完整 prompt 原文，而是能估算“这个请求在某个实例上能复用多少 KV”的元数据。

### Slide 15：score = cache_ratio - request_load - prefill_load

<img src="https://files.mdnice.com/user/59/1fece6d3-1e23-426f-8c48-a9789cd9e4f9.png" referrerpolicy="no-referrer" />

这页给出 v2 的多因子打分：`score = W1*cache_ratio - W2*request_load - W3*prefill_load`。`cache_ratio` 是 prefix cache 命中率，越高越应该选；`request_load` 是请求队列数，越高说明排队压力越大；`prefill_load` 是当前处于 prefill 阶段的 prompt 长度，越高说明实例正在吃长上下文 prefill。

这个公式更像工程排序项：在网关侧把三类信号合到同一个分数里。只看 cache 会把请求都打到热节点，只看 queue 又会错过大段 KV 复用；加上 prefill_load 后，长 prompt 正在跑的实例会被降权，避免 TTFT 被前面的 prefill 拖住。

### Slide 16：v2 优化效果

<img src="https://files.mdnice.com/user/59/15f0b99c-4599-41cc-99d6-3e8cfb0f9a47.png" referrerpolicy="no-referrer" />

v2 结果页说明 cache-aware routing 对长 prompt 和重复 prompt 有明显收益。我的理解是：它减少了“本可命中缓存却被路由到冷节点”的浪费。

### Slide 17：在线场景的稳定性约束

<img src="https://files.mdnice.com/user/59/e4db108b-8579-4b78-865a-e5679047214f.png" referrerpolicy="no-referrer" />

在线系统不能只看平均延迟，还要看抖动和故障恢复。网关策略如果过度追逐 cache，可能把少数节点压爆；如果过度追逐负载，又会牺牲 cache。

### Slide 18：v2 仍然解决不了的地方

<img src="https://files.mdnice.com/user/59/5f758ffe-14a0-4629-9046-fb49fd415959.png" referrerpolicy="no-referrer" />

v2 的剩余问题在于 score 仍然是启发式。真实延迟受模型、batch、KV、decode 长度、PD 链路影响，单个线性公式很难覆盖所有场景。

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
