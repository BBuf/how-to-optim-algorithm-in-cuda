> Continuous learning 不是“再训一次模型”这么简单。真正麻烦的是数据、重训、评测、服务验证和 adapter 发布要能串成一条可重复的链路。

# 0x0. 前言

这篇和模型推理关系没那么直接，但它补上了另一个生产问题：模型能力不是一次训练定终身，尤其 function calling 这类协议变化快的任务，需要持续收集数据、重训 adapter、评测并上线。

# 0x1. 资料和代码落点

代码落点：

- retrain-pipelines：`pkg_src/retrain_pipelines/dag_engine/core/core.py`，DAG Task、TaskGroup、trace capture。
- retrain-pipelines：`sample_pipelines/dag_engine/example_wf_7.py`，并行 task、taskgroup、merge function 示例。
- retrain-pipelines：`sample_pipelines/Unsloth_Qwen_FuncCall/legacy/retraining_pipeline.py`，Unsloth/PEFT function-calling adapter 训练流水线。
- retrain-pipelines：`pkg_src/retrain_pipelines/model/mf_unsloth_func_call_litserve/litserve/litserve_server.py`，`UnslothLitAPI` 实现 multi-adapter single endpoint serving。
- retrain-pipelines：`pkg_src/retrain_pipelines/model/mf_unsloth_func_call_litserve/eval.py`，function calling 评测解析和指标。

# 0x2. Slides 逐页解读

### Slide 1：Industrializing Continuous Learning

<img src="https://files.mdnice.com/user/59/208be9f5-c987-4823-8e2e-70f71fe9fd57.png" referrerpolicy="no-referrer" />

重点不是 LLM serving，而是 continuous learning 的工业化：把 retraining pipeline、评测、服务验证和 function-calling adapter 放到一个闭环里。

### Slide 2：目录：retraining framework

<img src="https://files.mdnice.com/user/59/9d70fd30-a455-4f72-a2b0-0eb82d213272.png" referrerpolicy="no-referrer" />

这一页是目录页，先把 retraining framework 放在第一部分。它提示后面不是讲某个单点训练脚本，而是讲一条连续学习流水线：数据、训练、评测、服务验证和产物追踪都要被纳入流程。

### Slide 3：目录：retrain-pipelines 与 function calling

<img src="https://files.mdnice.com/user/59/53b97736-dae4-452f-9296-a8fa27bd53c1.png" referrerpolicy="no-referrer" />

这一页仍然是目录页，把 Tool-Calling Task 和 Training/Evaluating 串了起来。这个案例很好，因为它同时涉及数据构造、LoRA/adapter 训练、function calling 评测和 serving 验证，正好能体现 continuous learning 为什么需要工程化闭环。

### Slide 4：pip-installable sandbox/production 环境

<img src="https://files.mdnice.com/user/59/39ae044b-41c8-466b-9851-f192ae7a5136.png" referrerpolicy="no-referrer" />

pip-installable 环境页强调低门槛。continuous learning 如果只能在少数专家机器上跑，就很难进入日常迭代。sandbox 和 production 环境的分离也能减少误发布。

### Slide 5：Notebook、CLI、Python 启动

<img src="https://files.mdnice.com/user/59/83c0e099-bb6a-407d-a5b1-b6d0e0e52f5f.png" referrerpolicy="no-referrer" />

启动方式包括 notebook、CLI 和 Python。对团队来说这很实用：研究者可以在 notebook 试，工程化后用 CLI 或脚本进 CI/CD。

### Slide 6：内部 DAG engine

<img src="https://files.mdnice.com/user/59/2573fc13-1b30-404f-8a4f-b2c41cf75e6d.png" referrerpolicy="no-referrer" />

内部 DAG engine 是 retrain-pipelines 的核心。左侧代码里能看到 `task`、`taskgroup`、`parallel_task`、`dag` 这些装饰器；底部那行 `start >> parallel >> snake_heads_A >> join_snake_heads >> merge >> end` 就是在用 Python 表达一条 DAG。右侧小字强调两点：pipeline 声明要简单；同时要能组合 taskgroups 和 sub-DAGs。

这里 taskgroup 和 sub-DAG 的区别也写在 slide 上：taskgroup 是一组异步并行任务，拿到同一份输入；sub-DAG 是并行分支，每个分支拿到上游任务输入的一部分。连续学习里常见的“同一份数据跑多个训练配置”和“把数据切片后并行处理”正好分别对应这两种模式。

### Slide 7：TaskGroup、sub-DAG、parallel branches

<img src="https://files.mdnice.com/user/59/ccb1a605-513c-4305-aa0a-787d5d919500.png" referrerpolicy="no-referrer" />

这一页把 `@parallel_task` 和 `@taskgroup` 放大了。`parallel(payload: TaskPayload)` 表示一个并行任务入口；`snake_heads_A()` 这个 taskgroup 返回 `snake_head_A1, snake_head_A2`，注释里写得很清楚：这是一组可以独立并行运行的任务，它们拿同一组输入，下游任务会等它们全部完成后再开始。

对应到模型迭代，taskgroup 可以用来同时训练两个 LoRA 配置、同时跑两套数据清洗或评测；sub-DAG 则适合把数据分片后各自处理。DAG engine 要解决的不是“能不能起多个进程”，而是上游输入、下游等待、失败恢复和产物归档这些关系。

### Slide 8：Aggregator 和 merge function

<img src="https://files.mdnice.com/user/59/1995d9f1-add8-45f6-9b19-d19fd95b6178.png" referrerpolicy="no-referrer" />

Aggregator 和 merge function 用来收敛并行分支结果。图里 `matrix_sum_cols` 是一个聚合函数，输入是二维矩阵，返回每列求和后的列表；下面 `@task(merge_func=matrix_sum_cols)` 说明 `merge` 节点收到的是多个并行上游任务的结果，先由 merge function 聚合，再继续做自定义处理。

这对 retraining pipeline 很实用。多个训练分支会产出不同 metrics、checkpoint 和日志，merge 节点可以做模型选择、生成汇总报告、决定是否进入 serving validation。没有这个节点，parallel 只是把任务跑散了，后面的模型发布仍然要靠人工整理。

### Slide 9：WebConsole

<img src="https://files.mdnice.com/user/59/03cea076-fd3a-4b0d-bf58-0c1213e7d6e9.png" referrerpolicy="no-referrer" />

WebConsole 提供运行列表、DAG 可视化、Gantt timeline 等。continuous learning 很需要可观测性，不然失败后只能翻散落日志。

### Slide 10：团队协作

<img src="https://files.mdnice.com/user/59/b6100d54-d253-4186-b727-9166152376f4.png" referrerpolicy="no-referrer" />

团队协作页强调共享运行记录、产物和文档。模型迭代不能停在单机脚本，尤其涉及生产发布时，大家要能看见同一份 pipeline 状态。

### Slide 11：Pipeline card

<img src="https://files.mdnice.com/user/59/d13fbb3e-e23c-4427-8e1f-87788190ca9e.png" referrerpolicy="no-referrer" />

Pipeline card 是这套系统很有意思的部分。它把 EDA、训练、评测、关键 artifact、DAG 信息整理成可浏览文档，减少“这次模型怎么来的”这种口头传递。

### Slide 12：HuggingFace Hub integration

<img src="https://files.mdnice.com/user/59/07ff0995-d7d4-4953-beff-3a4e7f55c07a.png" referrerpolicy="no-referrer" />

HuggingFace Hub integration 让模型和 artifact 能进入公共或私有 Hub。对 function-calling adapter 来说，base model、adapter、tokenizer/template 都要可追踪。

### Slide 13：Inspector：快速查看运行产物

<img src="https://files.mdnice.com/user/59/c059400e-1987-4829-9db8-e4e565106415.png" referrerpolicy="no-referrer" />

Inspector 是面向开发者的便利工具。比如直接打开本地 pipeline card，不用去 WebConsole 点很多层。

### Slide 14：Inspector：源码和 artifact 回溯

<img src="https://files.mdnice.com/user/59/954d8b99-67c4-4dff-93aa-3dd317d159f3.png" referrerpolicy="no-referrer" />

源码和 artifact 回溯很关键。continuous learning 里，如果某次模型变好或变坏，你必须知道当时用的是哪版训练代码、哪份数据、哪些参数。

### Slide 15：Inspector：模型和数据资产

<img src="https://files.mdnice.com/user/59/b4ec6445-0fb4-4dfb-b092-3fa193f9d488.png" referrerpolicy="no-referrer" />

模型和数据资产 inspector 解决的是“跑完之后去哪找东西”。这类工具不改变训练算法本身，但能减少团队沟通成本，也能让连续学习流水线更容易复盘。

### Slide 16：目录：进入 Tool-Calling

<img src="https://files.mdnice.com/user/59/77c21c39-fe50-4fe0-8b5d-12a8ca250fe8.png" referrerpolicy="no-referrer" />

这一页是目录过渡，从 retraining framework 转到 Tool-Calling。前半部分讲的是 pipeline 怎么组织，接下来进入具体任务：让一个小 adapter 稳定学会工具调用协议。

### Slide 17：Function calling 当前状态

<img src="https://files.mdnice.com/user/59/d1d89d19-5260-4a1a-990e-b1d687f463fe.png" referrerpolicy="no-referrer" />

Function calling 当前状态页进入案例。工具调用模型要学会什么时候调用、调用哪个工具、参数 JSON 怎么写，以及什么时候不调用。

### Slide 18：Tool calling 与 constrained generation

<img src="https://files.mdnice.com/user/59/b2498fea-e5ed-4a27-bdc6-331e7f69027c.png" referrerpolicy="no-referrer" />

Tool calling 与 constrained generation 说明输出格式很重要。function call 不是普通自然语言，JSON schema、参数类型、工具名都需要约束或后处理。

### Slide 19：Code interpreter 和工具响应

<img src="https://files.mdnice.com/user/59/095379e3-237a-432d-a9a9-6f51a9a87750.png" referrerpolicy="no-referrer" />

Code interpreter 和工具响应页把任务扩展到更复杂环境。模型不仅生成调用，还要读工具返回，继续回答或再次调用。

### Slide 20：Completion API 与 Responses API

<img src="https://files.mdnice.com/user/59/ba9f696b-ba2e-4a75-bf1f-bb0d06eb7ce6.png" referrerpolicy="no-referrer" />

Completion API 与 Responses API 页说明 API 形态变化会影响训练数据格式。adapter 如果绑定旧模板，服务端升级时就会出错。

### Slide 21：API 形态变化

<img src="https://files.mdnice.com/user/59/ebc3c6b8-8666-456d-8fbe-620f797e598d.png" referrerpolicy="no-referrer" />

API 形态变化页继续强调兼容性。训练 pipeline 要把 prompt template、tool schema、response parser 作为 artifact 管起来。

### Slide 22：Berkeley Function-Calling leaderboard

<img src="https://files.mdnice.com/user/59/3e9272e4-5d06-4e8b-811a-157c3d5f07e2.png" referrerpolicy="no-referrer" />

Berkeley Function-Calling leaderboard 给出评测参照。function calling 的评估不只是字符串匹配，还要解析 JSON、比较参数和工具选择。

### Slide 23：目录：进入 Training & evaluating

<img src="https://files.mdnice.com/user/59/46f26a5f-c6d1-4550-b7fe-d7c99f2bb6e9.png" referrerpolicy="no-referrer" />

这一页也是目录过渡，开始进入 Training & evaluating。前面讲清楚了 function calling 的任务形态，后面才回答怎么构造数据、训练 LoRA adapter，以及怎么评估它是不是真的会调用工具。

### Slide 24：Function-calling LoRA adapter

<img src="https://files.mdnice.com/user/59/204baf2d-3779-4c12-83e1-efed53e53f1e.png" referrerpolicy="no-referrer" />

Function-calling LoRA adapter 页对应代码里的 Unsloth + PEFT 训练。它把工具 schema 放进 prompt，让 adapter 学会按工具协议输出。

### Slide 25：数据集和无工具调用样本

<img src="https://files.mdnice.com/user/59/d2a1d2d5-8dab-44db-b0a6-328e39e86308.png" referrerpolicy="no-referrer" />

数据集页提到 legitimate absence of tool calls。这点很容易忽略：不是每个 query 都应该调用工具，训练集中必须保留“不调用”的正例。

### Slide 26：数据增强和 enrichment

<img src="https://files.mdnice.com/user/59/2a10f98f-540d-4f47-afbc-d221c2733ea1.png" referrerpolicy="no-referrer" />

数据增强和 enrichment 用来扩展工具调用覆盖面。比如生成更多参数组合、改写 query、补充无调用样本，都会影响 adapter 的 precision/recall。

### Slide 27：PEFT/Unsloth CPT + SFT

<img src="https://files.mdnice.com/user/59/d0fa60a5-9ada-43f2-bbbc-6c8ba94bdb08.png" referrerpolicy="no-referrer" />

PEFT/Unsloth CPT + SFT 页对应训练脚本。CPT 可以先适应工具格式和域数据，SFT 再对齐具体 function call 输出。

### Slide 28：评测结果：75.5%

<img src="https://files.mdnice.com/user/59/3239281d-0419-4a41-8022-73702425a300.png" referrerpolicy="no-referrer" />

评测 75.5% 这页给出结果。更重要的是后面 false negative 分析，因为 function calling 评测很容易把语义等价的 JSON 判错。

### Slide 29：False negatives 第一类

<img src="https://files.mdnice.com/user/59/8149a8b7-01fe-403b-9ade-bec34416ed16.png" referrerpolicy="no-referrer" />

False negatives 第一类通常来自参数等价但格式不同，或者工具调用顺序不影响结果。评测脚本需要在严格和宽松之间取平衡。

### Slide 30：False negatives 第二类

<img src="https://files.mdnice.com/user/59/0b837a76-e5e8-4cba-beba-f5b66137fea4.png" referrerpolicy="no-referrer" />

False negatives 第二类可能来自数据标注或 parser 边界。工业 pipeline 里，失败样本回流比单次分数更有价值。

### Slide 31：目录：进入 Serving

<img src="https://files.mdnice.com/user/59/cdbe24d3-bb73-4956-9e00-1381beb9ae3c.png" referrerpolicy="no-referrer" />

这一页是 Serving 章节的目录过渡。训练完 adapter 只是闭环的一半，真正进入生产还要能和 base model 一起在线服务，并允许请求按任务指定 adapter。

### Slide 32：Multi-adapter single endpoint

<img src="https://files.mdnice.com/user/59/9dbc203d-2f74-4471-9600-2fab5773c656.png" referrerpolicy="no-referrer" />

Multi-adapter single endpoint 这页先说 serving 形态：`transformers` 加载的 base LLM 可以挂 PEFT 兼容 adapter，adapter 能按需启停、切换。这样一个 base model 常驻显存，多个 LoRA adapter 作为“专家”按请求选择，不需要每个 adapter 起一套独立服务。

中间那句 task-specific system prompt 也很重要。SFT 训练时，query/response pair 前面会加任务专属 system prompt，adapter 其实是在这个模板下学会任务边界。推理时如果切 adapter 却没有同时切 `prompt_template`，模型收到的格式就和训练时不一致，function calling 这类任务会尤其明显：工具名、参数 JSON、是否调用工具都可能偏掉。

### Slide 33：每个 adapter 的 prompt template

<img src="https://files.mdnice.com/user/59/e7d336ac-3ae6-42ff-bc9b-925c275c57df.png" referrerpolicy="no-referrer" />

这页截的是 `retraining_pipeline.py` 里的 `supervised_finetuning`。代码里构造了 `self.sft_prompt_template = dedent("""...""")`，模板第一句明确要求模型 “return a list of tool calls based on your knowledge of known tools”。下面 rules 也很关键：只能使用已知工具，不能新造工具；如果 query 不匹配任何已知工具，返回空列表 `[]`；信息缺失时不要勉强调用；输出必须是合法 JSON array。

底部高亮的 `tokenizer.chat_template = self.sft_prompt_template` 说明模板不是文档说明，而是直接写进 tokenizer 的 chat template。这个设计和 serving 侧强相关：function-calling adapter 是在这套模板下训练出来的，如果推理时换回普通聊天模板，模型就很容易输出自然语言解释，而不是工具调用数组。

### Slide 34：LitServe 请求：不指定 adapter

<img src="https://files.mdnice.com/user/59/efd8cfad-0d60-43b1-96d9-91ab69913f0e.png" referrerpolicy="no-referrer" />

这页展示自定义 LitServe server 的调用方式。上方小字说明用的是 retrain-pipelines 对 Lightning AI LitServe 的自定义实现，服务启动时通过 YAML 配置拿到 base model 和待加载 adapter 列表。截图里的 cURL 请求打到 `http://localhost:8765/predict`，body 里 `adapter_name` 是空字符串，queries 包含 `"Hello there."` 和 `"Is 48 a perfect square?"`。

左侧竖排标注写着 “no adapter raw base-model”，右侧是 base model inference server response。这个例子故意不启用 adapter，用来对比下一页：同一个 endpoint、同样的 queries，只改 `adapter_name`，输出行为就会从普通问答切到 function-calling。

### Slide 35：base model 的原始响应

<img src="https://files.mdnice.com/user/59/47f13cd9-9754-41f8-bb6f-7137632fa728.jpg" referrerpolicy="no-referrer" />

这页把不指定 adapter 的 response body 放大了。返回是一个数组，每个元素包含 `query`、`input_tokens_count`、`completion` 和 `new_tokens_count`。`Hello there.` 对应的 completion 是一大段自然语言和 JavaScript/HTML 示例，`new_tokens_count` 到了 401；`Is 48 a perfect square?` 也返回了推理过程式文本。

这正是 function-calling adapter 要解决的问题：base model 可以回答问题，但没有稳定输出工具调用 schema，也不会把 “48 是否为平方数” 转成预期的 `is_perfect_square` 调用。服务层需要 adapter_name 和 prompt_template 同步切换，才能把同一个 base model 变成特定任务的专家。

### Slide 36：`func_caller_lora` adapter 响应

<img src="https://files.mdnice.com/user/59/f3b69c11-b2aa-4c09-90df-717a0d19e417.png" referrerpolicy="no-referrer" />

这一页启用了 `adapter_name: "func_caller_lora"`。请求还是同两个 query，但 response 已经变成工具调用风格：`Hello there.` 没有匹配已知工具，所以 completion 是 `[]`；`Is 48 a perfect square?` 被转成 `[{"name": "is_perfect_square", "arguments": {"num": 48}}]`。`new_tokens_count` 也从上一页的数百 token 降到二十来个 token。

从工程角度看，named adapters 开关对应 PEFT 的 `set_adapter/enable_adapters/disable_adapters`。服务端收到 batch 后，需要按请求里的 adapter 名称切换 LoRA，并且换到对应 prompt template。这里如果只切 LoRA 不切模板，输出 schema 仍然不稳；只切模板不切 LoRA，又缺少针对工具调用的参数增量。

### Slide 37：同一 endpoint 切换 adapter

<img src="https://files.mdnice.com/user/59/d846d51a-53c2-4265-b75d-359b919791e3.png" referrerpolicy="no-referrer" />

这页把 no-adapter 和 `func_caller_lora` 两种请求放在同一个图里，右侧斜着写了 “Switching on/off any of the named adapters for batch queries”。它想表达的不是多起几个服务，而是同一个 `/predict` endpoint 接收 batch queries，按 `adapter_name` 切换是否启用某个 named adapter。

这类方案的好处是 base model 常驻一次，多个小 LoRA 作为专家挂在同一套服务里。不同任务走不同 adapter，比维护很多 full model 轻；但它也要求 batch 调度、adapter 切换和 prompt template 管理足够明确，否则同一个 batch 内不同请求的输出协议会混在一起。

### Slide 38：Army of specialized experts

<img src="https://files.mdnice.com/user/59/063f00d4-eecc-4c5b-9bde-0e232f8c74d6.png" referrerpolicy="no-referrer" />

这一页把上面的例子抽象成 “specialized experts”。横幅里写它是迈向大规模、可适配企业 agentic systems 的一步。下面五条分别是：小模型运行，效率高、显存低；自托管，自己控制完整栈；部署简单；很多 domain-expert adapters 可以互换，且没有长上下文 prompt 带来的额外开销；一个 base model 加一组 adapter 组成完整系统。

放回 continuous learning 主线看，adapter 不是一次性训练产物，而是后续可持续替换的专家模块。数据回流发现某类工具调用失败，就重训对应 adapter；评测通过后推到服务端，用 named adapter 开关接入，不需要重启整套 base model 服务。

### Slide 39：目录：回到整体闭环

<img src="https://files.mdnice.com/user/59/5ea59869-6812-4db1-b2bd-041d459d7a51.png" referrerpolicy="no-referrer" />

最后回到目录页，意思是把 retraining framework、tool-calling、training/eval 和 serving 收成一条线。continuous learning 的难点不是训练一次，而是让数据、训练、评测、发布和回滚长期可维护。

### Slide 40：结束页

<img src="https://files.mdnice.com/user/59/51344965-7af8-4acd-ae57-fed96e1ba667.png" referrerpolicy="no-referrer" />

结束页就不展开了，留下 pipeline 和 function-calling adapter 的代码脉络，后面复用更方便。

# 0x3. 关键代码拆解

DAG engine 的基本节点是 `TaskType`。它在声明时包装函数，保存 parent/child，并且在运行时捕获 stdout/stderr/logging 写入 DB：

```python
class TaskType(BaseModel):
    func: Callable
    is_parallel: bool = False
    merge_func: Optional[Callable] = Field(default=None)
    tasktype_uuid: UUID = Field(default_factory=uuid4)

    _parents: List["TaskType"] = PrivateAttr(default_factory=list)
    _children: List["TaskType"] = PrivateAttr(default_factory=list)
    _task_group: Optional["TaskGroup"] = PrivateAttr(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        self.func = self._wrap_func(self.func)
        if self.merge_func is not None:
            self.merge_func = self._wrap_merge_func(self.merge_func)
```

example_wf_7 展示了 parallel task、taskgroup 和 merge：

```python
@parallel_task(ui_css=UiCss(background="#00ff37"))
def parallel(payload: TaskPayload):
    return [payload * 10 + i for i in range(2)]

@taskgroup(ui_css=UiCss(background="#000000", color="#e00000", border="#00fff7"))
def snake_heads_A():
    return snake_head_A1, snake_head_A2

@task(merge_func=matrix_sum_cols, ui_css=UiCss(background="#ff0000"))
def merge(payload: TaskPayload) -> List[int]:
    result = list(map(lambda x: x * 2, payload))
    return result
```

这类 DAG 对 retraining 很自然：并行试多个训练配置，merge 阶段做模型选择或汇总。

Serving 侧的 multi-adapter LitServe 更贴近 function-calling 案例。`UnslothLitAPI.setup` 先加载 base model，再加载多个 adapter 和各自 tokenizer：

```python
model, self.tokenizer = FastLanguageModel.from_pretrained(
    model_name=(Config.BASE_MODEL_PATH or Config.BASE_MODEL_REPO_ID),
    revision=(Config.BASE_MODEL_REVISION if Config.BASE_MODEL_PATH is None else None),
    max_seq_length=Config.MAX_SEQ_LENGTH,
    load_in_4bit=False,
)
self.model = FastLanguageModel.for_inference(model)

self.adapter_tokenizers = {}
for adapter_name, adapter in Config.adapters.items():
    self.model.load_adapter(
        peft_model_id=adapter_repo_id,
        revision=adapter_revision,
        adapter_name=adapter_name,
    )
    self.adapter_tokenizers[adapter_name] = AutoTokenizer.from_pretrained(adapter_repo_id)
```

predict 时按请求选择 adapter。如果请求没指定或 adapter 不存在，就 disable adapters 用 base model：

```python
if request.adapter_name in get_model_status(self.model).available_adapters:
    if set([request.adapter_name]) != set(self.model.active_adapters()):
        self.model.set_adapter(adapter_name=request.adapter_name)
    self.model.enable_adapters()
    for module in self.model.modules():
        if isinstance(module, ModulesToSaveWrapper):
            module.enable_adapters(enabled=True)
    tokenizer = self.adapter_tokenizers[request.adapter_name]
else:
    self.model.disable_adapters()
    tokenizer = self.tokenizer
```

最后用 adapter 自己的 chat template 格式化输入：

```python
formatted_inputs = [(tokenizer.chat_template or "{}").format(query, "")
                    for query in request.queries_list]

tokenized_inputs = tokenizer(
    formatted_inputs,
    padding=True,
    truncation=True,
    return_tensors="pt",
).to("cuda")

outputs = self.model.generate(
    input_ids=tokenized_inputs["input_ids"],
    attention_mask=tokenized_inputs["attention_mask"],
    max_new_tokens=Config.MAX_NEW_TOKENS,
    use_cache=True,
)
```

这段代码解释了 slides 里的“single endpoint, named adapters”：base model 常驻，adapter 和 tokenizer/template 成为可切换的专家。

# 0x4. 小结

Continuous learning 工业化的难点在闭环：训练代码、数据、评测、文档、服务验证都要可追踪。retrain-pipelines 用 DAG 和 pipeline card 管流程，用 LoRA adapter 和 LitServe 演示了 function-calling 专家如何训练和上线。
