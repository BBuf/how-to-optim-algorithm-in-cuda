> 这篇是 GOSIM Hangzhou 里 `GLM-4.5 系列模型开源生态` 这场分享的回放解读。原始 PDF 已经按页转成 mdnice 图片，正文里每一页 slides 都保留了对应图片；技术页我会尽量落到公开代码，讲清楚这页到底对应什么实现。

# 0x0. 前言

这篇会把 GLM-4.5 的 slides 当成生态解读，而不是复述榜单。模型结果当然重要，但对 infra 目录来说，更值得看的是真正支撑 GLM 系列 RL/post-training 的 slime 和 SGLang 链路。

# 0x1. 资料和代码落点

相关资料和代码：

- GLM-4.5 官方入口：`https://github.com/zai-org/GLM-4.5`，GLM-V 入口：`https://github.com/zai-org/GLM-V`。
- slime：`README.md` 明确列出 GLM-4.5/4.6/4.7/5 系列背后的 RL framework。
- slime：`slime/backends/sglang_utils/sglang_engine.py`，SGLang engine 启动、release/resume、update weights。
- slime：`slime/ray/rollout.py`，SGLang server group、router、colocate offload/onload、rollout 数据回流。
- slime 文档：`docs/zh/get_started/quick_start.md` 和 `docs/zh/examples/glm4.7-355B-A32B.md`，能看到 SGLang 参数、partial rollout、MTP 配置等。

# 0x2. Slides 逐页解读

### Slide 1：GLM-4.5 系列模型开源生态

![](https://files.mdnice.com/user/59/8e9e3b5e-658f-41ee-9d46-b081e2a3bad4.png)

这场分享讲 GLM-4.5 系列和开源生态。和 infra 目录的关系在于 slime：GLM-4.5 背后的 RL/post-training 框架公开后，训练和 SGLang rollout 的工程细节也能被讨论。

### Slide 2：目录

![](https://files.mdnice.com/user/59/9e72dc41-aa5f-40e5-bcc9-cf7524246546.png)

目录把语言模型、视觉模型和生态放在一起。GLM-4.5 不是只发布权重，还把训练框架、推理框架适配和社区使用一起展示。

### Slide 3：语言模型能力背景

![](https://files.mdnice.com/user/59/093767d4-574e-4842-9c15-682319a4cd53.png)

语言模型能力背景页给出问题设定：模型能力从聊天走向 agent、coding、reasoning、tool use。训练系统也随之从 SFT 扩展到 RL 和环境交互。

### Slide 4：GLM-4.5：All-round model

![](https://files.mdnice.com/user/59/a5d8c235-f42e-4fc5-b92e-27c33a843844.png)

GLM-4.5 被定位成 all-round model，意思是 reasoning、coding、agent、通用对话都要覆盖。这样的模型很依赖 post-training 数据和 RL recipe。

### Slide 5：Agent 能力对比

![](https://files.mdnice.com/user/59/7461aada-03eb-4dd7-b686-828aa89365cb.png)

Agent 能力对比页展示 tau-bench、BFCL-v3 等任务。Agent benchmark 的难点是多轮工具调用和环境反馈，和前面 verl 那篇的 AgentLoop 是同一类问题。

### Slide 6：代码能力

![](https://files.mdnice.com/user/59/1cd41d0a-7630-442f-b328-0086c36ca486.png)

代码能力页说明 GLM-4.5 在 SWE、代码生成等任务上的结果。代码能力提升通常不是单靠 pretrain，还需要指令、合成数据、执行反馈和 RL。

### Slide 7：通用能力

![](https://files.mdnice.com/user/59/89098ef1-d8db-4bc1-9c85-0a330b86d3ab.png)

通用能力页说明模型没有只往 agent/coding 偏。全能型模型训练里，数据 mixture 和 loss balancing 是关键。

### Slide 8：Tic-tac-toe 示例

![](https://files.mdnice.com/user/59/7fd3013d-dbfd-42e4-8a57-297b3955955c.png)

Tic-tac-toe 示例展示模型交互和规划能力。它不是核心技术页，但可以看作 agent 行为的直观样例。

### Slide 9：PPT/小红书生成示例

![](https://files.mdnice.com/user/59/bc6626cd-eaeb-40e2-98f6-4d1c2bd1da3a.png)

PPT/小红书示例偏产品化。技术上可以理解为模型在长结构化输出、风格控制和工具调用上的组合能力。

### Slide 10：模型结构

![](https://files.mdnice.com/user/59/93220e50-7220-4c6f-b3d3-2cbf1406de1b.png)

模型结构页通常会讲 MoE、active 参数、上下文长度等。具体参数以官方 GLM-4.5 仓库为准，本文不凭 slides 猜实现细节。

### Slide 11：训练流程

![](https://files.mdnice.com/user/59/265cbf10-d658-4afd-a594-4a6ed0e2073d.png)

训练流程页把 pretraining、mid-training、SFT、RL 串起来。slime 正是这个后训练阶段的系统支撑之一。

### Slide 12：RL 第一部分

![](https://files.mdnice.com/user/59/a90ddb09-200c-4bd7-8570-826cdde2f961.png)

RL 第一部分说明 GLM-4.5 的 post-training 不只做静态 SFT。RL 需要 rollout 引擎不断生成样本，再把 rewards/verifier 结果回流训练。

### Slide 13：RL 第二部分

![](https://files.mdnice.com/user/59/1d0ff2df-d84f-4d0c-a1ea-6da05aa1d8aa.png)

RL 第二部分继续展开 reward、verifier 或 agentic 数据。大模型 RL 的吞吐瓶颈往往在 rollout，因此 SGLang/vLLM 这类高吞吐服务成为训练链路的一部分。

### Slide 14：slime 文档入口

![](https://files.mdnice.com/user/59/953e3746-b2b5-406d-a8c9-506fe31946f3.png)

slime 文档入口是本文代码落点。slime README 明确写到它是 GLM-4.5 背后的 RL framework，并且用 Megatron + SGLang 连接训练和 rollout。

### Slide 15：GLM-4.5V

![](https://files.mdnice.com/user/59/0c8c7f51-9222-4ad2-9ae7-e5f77870213f.png)

GLM-4.5V 把话题转向视觉语言模型。VLM 的 RL 比纯文本更复杂，因为输入输出可能涉及图像 grounding、GUI 操作和视觉 verifier。

### Slide 16：Grounding 和语义能力

![](https://files.mdnice.com/user/59/698be5f3-f606-4c60-a51b-5f73d64a3972.png)

Grounding 和语义能力页强调视觉定位。训练数据里需要 box、region、OCR、语义关系等监督，推理时也要保留足够视觉分辨率。

### Slide 17：安全检测

![](https://files.mdnice.com/user/59/bedf3563-08bf-4f9e-b798-e009a6abf4f1.png)

安全检测页说明 VLM 不只看能力，也看安全边界。多模态安全比文本更难，因为图像里可能隐藏文本、符号或场景线索。

### Slide 18：GLM-4.5V 模型介绍

![](https://files.mdnice.com/user/59/e34aeda2-253a-4587-b00c-c15a0dbb3178.png)

GLM-4.5V 模型介绍页把视觉模型和语言模型联系起来。开源生态里通常需要同时给模型、processor、推理示例和评测脚本。

### Slide 19：V 模型结构

![](https://files.mdnice.com/user/59/8154eb49-f8fe-4760-ae5b-562a9f026df8.jpg)

V 模型结构页可能包含 vision encoder、projector、LLM backbone。和 Ming 那篇类似，关键是视觉特征如何进入语言模型上下文。

### Slide 20：预训练

![](https://files.mdnice.com/user/59/fc82782d-3810-4182-9309-5b8f16c45902.png)

预训练页讲数据规模和阶段。VLM 预训练里的图文配对质量、OCR 数据、视频帧采样都会影响后续 grounding 和 GUI 能力。

### Slide 21：数据工程第一部分

![](https://files.mdnice.com/user/59/db736f29-19d1-448e-95c4-89dbd3b6dfcd.png)

数据工程第一页展示数据 pipeline。对 GLM-4.5 这种模型，数据工程比单个 loss trick 更能决定上限。

### Slide 22：数据工程第二部分

![](https://files.mdnice.com/user/59/a1edb74f-8661-4277-8be2-9ef515459558.png)

数据工程第二页继续讲清洗、合成、过滤或标注。开源生态能不能复现，很大程度取决于这些过程是否有可执行描述。

### Slide 23：训练策略

![](https://files.mdnice.com/user/59/12988909-d625-42cc-a1eb-c3f1bfadd22f.png)

训练策略页把模型结构、数据和并行训练放到一起。大 MoE/VLM 训练要处理 TP/PP/EP/DP、checkpoint 转换和 rollout 同步。

### Slide 24：RL for VLM

![](https://files.mdnice.com/user/59/ce03fc77-26e7-4df0-95da-8e99d3d66430.png)

RL for VLM 页说明多模态也要进入 RL。GUI、grounding、OCR 这类任务往往有可验证反馈，适合做 RL 或 rejection sampling。

### Slide 25：GUI Agents 和 CogAgent

![](https://files.mdnice.com/user/59/986279af-c2ba-4d73-87b5-c5aa67dd343f.png)

GUI Agents 和 CogAgent 页把 VLM 推向实际环境操作。训练框架要支持多轮、截图输入、动作输出和环境状态，这和 agentic RL 的系统需求一致。

### Slide 26：开源生态

![](https://files.mdnice.com/user/59/99994339-2c92-4d14-b2d3-9336a2f32ae5.png)

开源生态页说明模型不是孤立权重。仓库、文档、推理后端、训练框架、社区 issue 都是生态的一部分。

### Slide 27：生态地图

![](https://files.mdnice.com/user/59/c763ccff-0c88-4d93-b5b2-dad829991396.png)

生态地图页展示周边项目。对 infra 读者来说，重点是 slime、SGLang、Megatron、serving 框架之间的连接方式。

### Slide 28：HuggingFace trending

![](https://files.mdnice.com/user/59/8f769407-82df-43c7-a1c9-7992d9409e9b.png)

HF trending 页说明发布后的社区反馈。它不能替代技术评测，但能说明模型资产被真实用户拿去试。

### Slide 29：框架采用情况

![](https://files.mdnice.com/user/59/fefa4c73-d177-46e0-b3dd-d522e628f8b9.png)

框架采用情况页展示 vLLM/SGLang/Transformers 等生态适配。大模型开源如果只能在单一脚本跑，传播会慢很多。

### Slide 30：开源流程

![](https://files.mdnice.com/user/59/4be48ff8-28dd-4294-a614-95cc88ce1375.png)

开源流程页讲 release 和协作。真正省别人时间的是明确权重格式、推理配置、训练 recipe 和已知限制。

### Slide 31：社区反馈

![](https://files.mdnice.com/user/59/7b249fee-0bf7-4fa1-b12f-c8e1544b118b.png)

社区反馈页说明 issue/PR/讨论会继续推动模型和框架修复。开源生态不是一次发布结束，而是后续兼容和性能优化。

### Slide 32：论文和 GitHub

![](https://files.mdnice.com/user/59/0eee7cb4-1b5e-4c4d-9e4e-64e2e7931d94.png)

论文和 GitHub 页给出入口。本文代码讲解选择 slime，因为它和 GLM-4.5 RL 训练关系最直接。

### Slide 33：总结

![](https://files.mdnice.com/user/59/7a2bba40-60d7-45b2-8bed-3a386da00e97.png)

总结页可以回到一条线：GLM-4.5 系列把模型能力、RL 训练系统和开源生态绑在一起，slime/SGLang 是其中的基础设施部分。

### Slide 34：补充页：生态继续扩展

![](https://files.mdnice.com/user/59/eee3f7cd-591c-46d5-8940-a4a2cd3cca2a.png)

补充页在 PDF 里更像附加信息。这里不额外延展，只提醒读者以官方仓库和文档为准。

### Slide 35：结束页

![](https://files.mdnice.com/user/59/81d7b369-ad20-49ef-9e6d-2123531bfe45.png)

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
