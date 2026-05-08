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

先看系统关系：verl 负责 RL 训练编排，SGLang 可以作为 rollout 后端。Agentic 任务把采样变成多轮交互后，推理后端的生命周期管理会变得更重要。

### Slide 2：项目背景

<img src="https://files.mdnice.com/user/59/0a778b33-4eea-4d4f-9ff0-1519dae69c81.png" referrerpolicy="no-referrer" />

这一页的项目定位很清楚：verl 不是单一算法实现，而是把 PPO/GRPO/RLOO、FSDP/Megatron、vLLM/SGLang、Ray 编排拼在一起的训练系统。

### Slide 3：Seed 团队项目背景

<img src="https://files.mdnice.com/user/59/4834c013-8b60-492a-b638-2ee25807c74f.png" referrerpolicy="no-referrer" />

这一页仍然是背景页，重点放在 Seed 系列项目和 RL 训练需求。放到这篇文章里看，它的作用是说明 verl 面对的不是小 demo，而是大模型 post-training、tool-use 和 agentic task 这类真实系统压力。

### Slide 4：SFT 和 RL 的边界

<img src="https://files.mdnice.com/user/59/0687d159-14be-4f2f-a24b-f678991b21e1.png" referrerpolicy="no-referrer" />

SFT 的数据是静态的，RL 的数据来自当前 policy。模型一训练，policy 就变，下一轮 rollout 分布也变。所以 RL 框架必须把“生成数据”和“消费数据”放在同一个控制闭环里。

### Slide 5：为什么 LLM RL 需要系统框架

<img src="https://files.mdnice.com/user/59/156b6ef3-b8b2-4ecf-adbb-6460d57064f6.png" referrerpolicy="no-referrer" />

LLM RL 不是多跑几条 generate 就完事。它要同时处理 actor、reference、reward/verifier、rollout engine、advantage、logprob，以及模型权重在训练端和推理端之间的更新。

### Slide 6：RL 数据流比监督学习复杂得多

<img src="https://files.mdnice.com/user/59/bcf1ce0e-56eb-4224-a12b-963d2088056d.png" referrerpolicy="no-referrer" />

这页的数据流图通常是 verl 入门的关键。一次 rollout 会产生 responses、logprobs、rewards、masks；训练时又要按 micro-batch、sequence length、DP rank 做切分。任何一步形状不对，后面 loss 就会悄悄错。

### Slide 7：大规模分布式数据流

<img src="https://files.mdnice.com/user/59/9b91b06c-3db6-4960-9b64-ba761d2263f8.png" referrerpolicy="no-referrer" />

分布式后，数据流从函数调用变成跨 actor 的调度。Ray actor、placement group、device mesh 和通信组都要对齐。verl 的 Controller 负责把这些事情放在一个可读的 Python 流程里。

### Slide 8：RL 数据流里的依赖和资源限制

<img src="https://files.mdnice.com/user/59/dbb12ae4-cabd-46af-9d99-58a1e6832fc9.png" referrerpolicy="no-referrer" />

这一页还在解释为什么 RL 数据流难写：不同阶段有数据依赖，actor/reference/reward/rollout 又抢同一批 GPU 资源。verl 后面的 Hybrid Controller 设计，就是为了把这些依赖和资源切换放进一个可维护的控制流里。

### Slide 9：社区和采用情况

<img src="https://files.mdnice.com/user/59/d5642148-3456-4983-aeb0-a84cc5e0b2a2.png" referrerpolicy="no-referrer" />

社区页说明 verl 已经不是论文附属代码。它的使用者来自开源模型、研究项目和工业训练，带来的后果是后端适配必须现实：FSDP、Megatron、vLLM、SGLang 都要能接。

### Slide 10：verl 的功能面

<img src="https://files.mdnice.com/user/59/a1c4d479-2790-4460-adeb-a7600be8329a.png" referrerpolicy="no-referrer" />

功能面覆盖 PPO/GRPO、FSDP/Megatron actor、vLLM/SGLang rollout、multi-turn、tool calling。对 SGLang 用户来说，重点是 rollout 后端已经有显存释放、权重同步和 server adapter。

### Slide 11：Hybrid Controller：把控制流留在 Python

<img src="https://files.mdnice.com/user/59/dc3f2492-e100-4190-8cba-bb067e79600a.png" referrerpolicy="no-referrer" />

Hybrid Controller 把“谁先跑、谁等谁、数据怎么分发”放在一个 Controller 进程里。Worker 只实现具体计算。这个设计牺牲了一点极致自动调度，换来算法流程可读。

### Slide 12：单 Controller 驱动多 Worker

<img src="https://files.mdnice.com/user/59/cd0badbd-f787-4c1e-b1c3-cee6bdf43364.png" referrerpolicy="no-referrer" />

单 Controller 多 Worker 的好处是调试。你能沿着 Python 调用栈看到 actor rollout、reference logprob、reward 这些步骤，而不是在多个服务之间猜消息顺序。

### Slide 13：Dispatch/Collect 的数据分发语义

<img src="https://files.mdnice.com/user/59/199975db-56fa-4425-b2ad-aebd6c5d8706.png" referrerpolicy="no-referrer" />

Dispatch/Collect 是 verl 的核心抽象。不同数据要按 rank0、all-to-all、DP compute 等模式分发。比如 prompt 数据按 DP 分片，模型参数更新又按 TP/PP mesh 走不同路径。

### Slide 14：FSDP、Megatron、vLLM、SGLang 后端

<img src="https://files.mdnice.com/user/59/8a5c0533-16eb-4cce-98d0-e9e074f7ff95.png" referrerpolicy="no-referrer" />

后端层面，verl 不把自己绑死在一个训练框架上。FSDP 适合 HuggingFace/torch 生态，Megatron 适合大 MoE；rollout 端可以接 vLLM，也可以接 SGLang。

### Slide 15：3D-HybridEngine 和 colocate

<img src="https://files.mdnice.com/user/59/9c98a861-0533-4b50-80cd-f76112c33110.png" referrerpolicy="no-referrer" />

3D-HybridEngine 指的是训练和推理在 3D 并行、显存、执行阶段上的复用。Colocate 时，推理引擎必须支持 release/resume，否则训练 peak 一来就 OOM。

### Slide 16：verl 的编程方式

<img src="https://files.mdnice.com/user/59/a4dbb7c0-7504-4122-8410-25852913e2ac.png" referrerpolicy="no-referrer" />

这一页展示的是 verl 的编程接口：用户在单 controller 里写 Python 流程，背后调用会分发到多个 worker group。它把分布式细节收进 decorator、dispatch 和 collect 语义里，让算法代码还能保持“先 rollout，再算 logprob/reward，再更新”的形状。

### Slide 17：Agentic RL 章节过渡

<img src="https://files.mdnice.com/user/59/13866e10-62e7-471d-8407-eb8fe33951fb.png" referrerpolicy="no-referrer" />

这一页是章节过渡，开始进入 Agentic RL。前面讲的是普通 RL 数据流和 Hybrid Controller，后面要处理的是更麻烦的 rollout：模型会调用工具、等待环境返回，再继续生成。

### Slide 18：什么是 Agent

<img src="https://files.mdnice.com/user/59/1a51f878-78f8-4662-9cbd-8f051f9a167c.png" referrerpolicy="no-referrer" />

这页先定义 Agent：模型不只是生成一段答案，而是在环境里观察状态、规划动作、调用工具、读取反馈，再决定下一步。对 RL 框架来说，这意味着 rollout 不再是固定长度的一次 `generate`。

### Slide 19：ReTool：工具调用型训练

<img src="https://files.mdnice.com/user/59/38bbf1b5-7ae4-4f02-8095-99d1366ddce4.png" referrerpolicy="no-referrer" />

ReTool 这类工作把工具调用纳入训练。工具返回的文本或图片会进入下一轮上下文，reward 也可能来自工具结果是否满足 verifier。它把普通 RLHF 里的 response，扩展成了 action/observation 交替出现的轨迹。

### Slide 20：Agentic RL 的同步 rollout 问题

<img src="https://files.mdnice.com/user/59/87ea2976-4cc8-4337-867e-fdbf01e1853e.png" referrerpolicy="no-referrer" />

这一页讲的是 naive synchronous rollout 的缺点。多轮 agent 样本长度和工具耗时差异很大，慢样本会拖住一整批训练数据；如果 rollout 引擎一直等环境，GPU 利用率也会掉。

### Slide 21：AgentLoop 状态机

<img src="https://files.mdnice.com/user/59/3674cef1-8c90-44b1-a72c-1b85ca0bfc7f.png" referrerpolicy="no-referrer" />

AgentLoop 状态机是 verl 代码里最直观的实现：`PENDING -> GENERATING -> PROCESSING_TOOLS -> TERMINATED`。这比把所有逻辑写成一个大 while 更容易扩展新的环境，也方便把 tool latency 和模型生成解耦。

### Slide 22：AgentLoop 的 server mode

<img src="https://files.mdnice.com/user/59/3c1b21f8-eaf5-499b-811f-8d916c40e7e2.png" referrerpolicy="no-referrer" />

server mode 的意思是把 AgentLoop 变成可服务化的 rollout 组件。训练 controller 不必把工具环境和生成逻辑全塞在一个进程里，而是通过服务接口请求 agent rollout，再把生成轨迹收回来训练。

### Slide 23：ReTool with AgentLoop

<img src="https://files.mdnice.com/user/59/c6bd3548-fbed-4eb7-b74b-f4818ddc2c10.png" referrerpolicy="no-referrer" />

这页把 ReTool 放回 AgentLoop 框架里：模型生成工具调用，环境执行工具，把观察结果写回上下文，然后继续生成。训练前要把这些轨迹整理成统一的 token、mask、logprob 和 reward 张量。

### Slide 24：ReTool 复现经验

<img src="https://files.mdnice.com/user/59/dbea2a89-9dbb-4629-8684-31b999c22e58.png" referrerpolicy="no-referrer" />

这页是 ReTool reproduction 的经验总结。我的理解是，Agentic RL 的难点不只在算法，而在数据格式、工具协议、reward/verifier 和异步执行都要对齐。只要一个环节不稳定，最后看到的就是 reward 曲线抖、复现困难。

### Slide 25：Roadmap：更大的 MoE 和更强推理后端

<img src="https://files.mdnice.com/user/59/1c2f87a3-fc72-4ca8-8ca5-62ab1125f571.png" referrerpolicy="no-referrer" />

更大的 MoE 模型会把问题放大：参数更多、expert routing 更复杂、rollout 引擎更吃显存。SGLang 后端在 DP attention、EP、MTP、memory saver 上的能力会越来越关键。

### Slide 26：大 MoE RL 训练更新

<img src="https://files.mdnice.com/user/59/d299f10a-532d-48bd-98c0-49aea8c20f59.png" referrerpolicy="no-referrer" />

这页讲 Trainer Updates，重点是大 MoE 模型的 scalable RL。slime 这类 SGLang-native 系统可以作为对照：当模型规模继续变大，训练端和 rollout 端的权重同步、显存管理、路由和部分 rollout 都会变成一等公民。

### Slide 27：Partial rollout 和 async rollout

<img src="https://files.mdnice.com/user/59/dd64e373-0b24-43ae-8a0e-33d1acc9780b.png" referrerpolicy="no-referrer" />

Partial rollout 的想法很实用：动态采样里被提前 abort 的样本不要丢，下轮接着生成。这样能缓解长尾，但要求 data buffer 能保存半截请求状态。

### Slide 28：更真实的 Agentic 任务

<img src="https://files.mdnice.com/user/59/09723001-3598-4ef3-8998-d8ada87aa22d.png" referrerpolicy="no-referrer" />

这一页回到更真实的 agentic task。真实环境里会有更长的工具链、更复杂的观察结果和更强的异步需求。对 SGLang rollout 后端来说，这意味着 server 生命周期、请求状态和采样接口都要能承受多轮交互。

### Slide 29：社区协作方向

<img src="https://files.mdnice.com/user/59/77311927-0b64-4cac-a3e2-b1eab98a4580.png" referrerpolicy="no-referrer" />

社区方向主要是让算法和系统各自迭代。Agentic 任务还在快速变化，框架如果抽象太死，会很快跟不上工具协议和环境形式。

### Slide 30：总结

<img src="https://files.mdnice.com/user/59/fc135309-9983-4b77-b911-016dfc4e2240.png" referrerpolicy="no-referrer" />

总结页回到主题：verl 的价值不是某个 RL 算法，而是把复杂 RL 数据流、分布式 Worker 和推理后端管理放到一个能改、能调、能扩展的框架里。

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

release/resume 直接打到 SGLang HTTP 接口。这里的 `sleep_level` 对 LoRA 很关键，LoRA adapter 模式下可以只释放 KV cache，保留 base weights：

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
