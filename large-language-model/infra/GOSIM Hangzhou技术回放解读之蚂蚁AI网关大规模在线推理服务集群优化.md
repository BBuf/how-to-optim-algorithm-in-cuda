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

![](https://files.mdnice.com/user/59/e6216fee-f2ff-4293-bd43-d27d0499e02b.png)

这篇讨论的是推理网关，不是普通 HTTP 网关：LLM 请求的成本取决于 prompt 长度、decode 长度、cache 命中、PD 分离状态和后端 batch，因此路由层必须懂一些模型推理语义。

### Slide 2：目录：从负载特征到网关调度

![](https://files.mdnice.com/user/59/4ba5cbfb-765e-4c00-87ba-0cb7b1733acc.png)

目录从负载特征开始，再讲 Ant 网关两代调度策略，最后进入 latency prediction、Mooncake Store 和云原生扩展。它的主线是：让网关从“转发请求”升级成“推理集群调度器”。

### Slide 3：AI Gateway 在推理集群中的位置

![](https://files.mdnice.com/user/59/7c132901-9696-4cfa-bd14-00cdc146911e.png)

AI Gateway 位于用户/API 和推理后端之间。它要做鉴权、限流、路由、熔断，也要知道后端 SGLang/vLLM 实例的负载状态。对大模型来说，路由错一次可能带来几秒 TTFT 差异。

### Slide 4：LLM 请求负载是非线性的

![](https://files.mdnice.com/user/59/257b8694-b4db-44de-89ce-1c3663fcd3df.png)

LLM 负载不是请求数线性可加。一个 200 token prompt 和一个 20k token prompt 对 prefill 的压力完全不同；decode 阶段又按输出 token 持续占用 batch slot。

### Slide 5：Prefill 和 Decode 两个阶段

![](https://files.mdnice.com/user/59/fd9c63b7-0e23-4fe7-8673-028028f43fea.png)

Prefill 主要吃 attention 的长序列计算和 KV 写入，decode 每步短但持续时间长。PD 分离就是把这两种负载拆开，让不同节点承担不同阶段。

### Slide 6：Attention 与 FFN 的资源差异

![](https://files.mdnice.com/user/59/0fc0af91-eaf6-4037-8104-860dfb07a433.png)

Attention 和 FFN 的资源画像也不同。Attention 更受 KV cache、序列长度和带宽影响；FFN/MoE 更受矩阵乘、expert 路由影响。网关如果只看 QPS，很难判断哪个节点真的忙。

### Slide 7：从 request 调度到 token 调度

![](https://files.mdnice.com/user/59/cd05a051-4f57-4eda-9ffe-0872ba806dfd.png)

请求级调度只决定“这个 prompt 去哪台机器”。token 级调度才决定 decode 阶段持续占用多少资源。服务端 batch scheduler 和网关路由其实是一体两面。

### Slide 8：传统负载均衡为什么不够

![](https://files.mdnice.com/user/59/f821c877-ec3e-443f-a1a9-03e5bf08ebbf.png)

传统 LB 的 round-robin 或 least-connection 对 LLM 不够用。两台实例连接数一样，但一台可能在处理长 prompt prefill，另一台只是短 decode。

### Slide 9：Prefix cache 改变路由目标

![](https://files.mdnice.com/user/59/d63be6a3-4ab2-46ba-998c-e00a5f74a16c.png)

Prefix cache 让路由目标变成“命中最多缓存的节点”。如果一个 prompt 前缀在某节点已有 KV，哪怕它当前稍忙，也可能比空闲节点重新 prefill 更快。

### Slide 10：Ant AI Gateway 架构

![](https://files.mdnice.com/user/59/a5ff0c73-4b37-4876-afb7-d041a76a8e46.png)

Ant AI Gateway 的架构页把控制面和数据面分开：数据面接请求并路由，控制面收集后端 metrics、维护实例状态和路由策略。

### Slide 11：v1：轮询 metrics 和 queue-num score

![](https://files.mdnice.com/user/59/40cfe92e-cf12-4aee-811f-874b491e3e4d.png)

v1 用后端轮询 metrics，score 近似取 queue-num。这个版本简单，但它隐含了两个假设：queue 长度能代表未来延迟，cache 命中不影响路由。这两个假设在 LLM 场景都不太稳。

### Slide 12：v1 的问题：指标滞后和缓存不敏感

![](https://files.mdnice.com/user/59/6cd6b488-ffcd-48c0-a772-b1c106d44e5c.png)

v1 的问题来自滞后和粗粒度。metrics 采样间隔内负载可能已经变了；而且 queue-num 不知道某个请求是 100 token 还是 100k token。

### Slide 13：v2：self-loop metrics

![](https://files.mdnice.com/user/59/c12c90ef-4e62-461e-9618-d488411dcc17.png)

v2 引入 self-loop metrics，让网关数据面更快看到请求执行反馈。相比纯控制面轮询，它更接近请求路径，也更适合短时间窗口的负载估计。

### Slide 14：cache-aware prefix tree

![](https://files.mdnice.com/user/59/658a8974-abb0-48a9-a39a-4c994a6abd57.png)

cache-aware prefix tree 是这场分享最有 LLM 味的一页。网关维护 prompt 前缀树，用来估计某个请求在不同实例上的 prefix cache 命中比例。

### Slide 15：score = cache_ratio - request_load - prefill_load

![](https://files.mdnice.com/user/59/1fece6d3-1e23-426f-8c48-a9789cd9e4f9.png)

score 公式把 cache_ratio、request_load、prefill_load 放在一起。它不是追求数学完美，而是把三个最影响 TTFT 的量拉到同一决策里：能命中缓存、队列别太长、prefill 别太拥挤。

### Slide 16：v2 优化效果

![](https://files.mdnice.com/user/59/15f0b99c-4599-41cc-99d6-3e8cfb0f9a47.png)

v2 结果页说明 cache-aware routing 对长 prompt 和重复 prompt 有明显收益。我的理解是：它减少了“本可命中缓存却被路由到冷节点”的浪费。

### Slide 17：在线场景的稳定性约束

![](https://files.mdnice.com/user/59/e4db108b-8579-4b78-865a-e5679047214f.png)

在线系统不能只看平均延迟，还要看抖动和故障恢复。网关策略如果过度追逐 cache，可能把少数节点压爆；如果过度追逐负载，又会牺牲 cache。

### Slide 18：v2 仍然解决不了的地方

![](https://files.mdnice.com/user/59/5f758ffe-14a0-4629-9046-fb49fd415959.png)

v2 的剩余问题在于 score 仍然是启发式。真实延迟受模型、batch、KV、decode 长度、PD 链路影响，单个线性公式很难覆盖所有场景。

### Slide 19：Latency prediction

![](https://files.mdnice.com/user/59/c9ce8fd6-6265-412f-bdc4-324be6e6b1b7.png)

Latency prediction 的方向是把历史请求和后端状态喂给预测器，直接估计 TTFT/TPOT。这样网关可以路由到“预计最快完成”的节点，而不是只看当前指标。

### Slide 20：Mooncake Store 和 KVCache 共享

![](https://files.mdnice.com/user/59/d87aa43b-f644-475c-aac2-9fdd3e1cb7b5.png)

Mooncake Store 对应 KVCache 跨节点共享。公开 Mooncake 代码里有 KV transfer connector 和 store service，能把远端 prefill 生成的 KV 拉到 decode 节点，避免重复 prefill。

### Slide 21：TTFT 与 TPOT 的建模

![](https://files.mdnice.com/user/59/2bbb5658-9f9f-49d2-8dc6-f68a224c3089.png)

TTFT 通常会随 prefill 长度呈更高阶增长，TPOT 更像 decode 阶段分段线性。把二者分开建模，比用一个总 latency score 更符合推理过程。

### Slide 22：PD Router 和 DP imbalance

![](https://files.mdnice.com/user/59/a0783f4a-28ab-4a3f-873b-5824606c53e6.png)

PD Router 和 DP imbalance 讲的是分离式部署后的新问题：prefill/decode 实例比例、DP rank 负载、KV 传输链路都会影响整体吞吐。

### Slide 23：云原生 Go 扩展

![](https://files.mdnice.com/user/59/a869b709-9cce-4100-9236-da3462f12811.png)

云原生 Go 扩展页可以对照 Envoy AI Gateway：Kubernetes CRD 描述模型路由、后端权重、token cost 和 rate limit，再由控制器生成底层 HTTPRoute。

### Slide 24：总结

![](https://files.mdnice.com/user/59/b406a783-38e3-40c7-a04d-4a916f80bcbf.png)

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
