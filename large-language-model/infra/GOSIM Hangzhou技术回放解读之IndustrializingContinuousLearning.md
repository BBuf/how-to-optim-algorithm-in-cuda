> 这篇是 GOSIM Hangzhou 里 `Industrializing Continuous Learning` 这场分享的回放解读。原始 PDF 已经按页转成 mdnice 图片，正文里每一页 slides 都保留了对应图片；技术页我会尽量落到公开代码，讲清楚这页到底对应什么实现。

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

![](https://files.mdnice.com/user/59/208be9f5-c987-4823-8e2e-70f71fe9fd57.png)

这场分享不像前面几场专注 LLM serving，而是讲 continuous learning 的工业化。它把 retraining pipeline、评测、服务验证和 function-calling adapter 放到一个闭环里。

### Slide 2：Speaker 和主题背景

![](https://files.mdnice.com/user/59/9d70fd30-a455-4f72-a2b0-0eb82d213272.png)

背景页说明项目来自 retrain-pipelines。它的目标不是替代训练框架，而是把“数据进来、模型重训、评测、发布验证、文档产出”串成可重复流程。

### Slide 3：目录：retrain-pipelines 与 function calling

![](https://files.mdnice.com/user/59/53b97736-dae4-452f-9296-a8fa27bd53c1.png)

目录分两大块：先讲 retrain-pipelines 的 MLOps 能力，再用 function calling adapter 作为具体案例。这个案例很好，因为它同时涉及数据、LoRA、评测和 serving。

### Slide 4：pip-installable sandbox/production 环境

![](https://files.mdnice.com/user/59/39ae044b-41c8-466b-9851-f192ae7a5136.png)

pip-installable 环境页强调低门槛。continuous learning 如果只能在少数专家机器上跑，就很难进入日常迭代。sandbox 和 production 环境的分离也能减少误发布。

### Slide 5：Notebook、CLI、Python 启动

![](https://files.mdnice.com/user/59/83c0e099-bb6a-407d-a5b1-b6d0e0e52f5f.png)

启动方式包括 notebook、CLI 和 Python。对团队来说这很实用：研究者可以在 notebook 试，工程化后用 CLI 或脚本进 CI/CD。

### Slide 6：内部 DAG engine

![](https://files.mdnice.com/user/59/2573fc13-1b30-404f-8a4f-b2c41cf75e6d.png)

内部 DAG engine 是 retrain-pipelines 的核心。pipeline 不是线性脚本，而是有并行任务、子图、合并节点和运行上下文的 DAG。

### Slide 7：TaskGroup、sub-DAG、parallel branches

![](https://files.mdnice.com/user/59/ccb1a605-513c-4305-aa0a-787d5d919500.png)

TaskGroup、sub-DAG、parallel branches 对应真实训练流程：数据准备、多个候选训练、评测、模型 blessing 可以并行或分组执行。

### Slide 8：Aggregator 和 merge function

![](https://files.mdnice.com/user/59/1995d9f1-add8-45f6-9b19-d19fd95b6178.png)

Aggregator 和 merge function 用来收敛并行分支结果。比如多个并行实验产出 metrics，merge 节点可以选择最佳模型或生成汇总报告。

### Slide 9：WebConsole

![](https://files.mdnice.com/user/59/03cea076-fd3a-4b0d-bf58-0c1213e7d6e9.png)

WebConsole 提供运行列表、DAG 可视化、Gantt timeline 等。continuous learning 很需要可观测性，不然失败后只能翻散落日志。

### Slide 10：团队协作

![](https://files.mdnice.com/user/59/b6100d54-d253-4186-b727-9166152376f4.png)

团队协作页强调共享运行记录、产物和文档。模型迭代不是个人脚本，尤其涉及生产发布时，大家要能看见同一份 pipeline 状态。

### Slide 11：Pipeline card

![](https://files.mdnice.com/user/59/d13fbb3e-e23c-4427-8e1f-87788190ca9e.png)

Pipeline card 是这套系统很有意思的部分。它把 EDA、训练、评测、关键 artifact、DAG 信息整理成可浏览文档，减少“这次模型怎么来的”这种口头传递。

### Slide 12：HuggingFace Hub integration

![](https://files.mdnice.com/user/59/07ff0995-d7d4-4953-beff-3a4e7f55c07a.png)

HuggingFace Hub integration 让模型和 artifact 能进入公共或私有 Hub。对 function-calling adapter 来说，base model、adapter、tokenizer/template 都要可追踪。

### Slide 13：Inspector：快速查看运行产物

![](https://files.mdnice.com/user/59/c059400e-1987-4829-9db8-e4e565106415.png)

Inspector 是面向开发者的便利工具。比如直接打开本地 pipeline card，不用去 WebConsole 点很多层。

### Slide 14：Inspector：源码和 artifact 回溯

![](https://files.mdnice.com/user/59/954d8b99-67c4-4dff-93aa-3dd317d159f3.png)

源码和 artifact 回溯很关键。continuous learning 里，如果某次模型变好或变坏，你必须知道当时用的是哪版训练代码、哪份数据、哪些参数。

### Slide 15：Inspector：模型和数据资产

![](https://files.mdnice.com/user/59/b4ec6445-0fb4-4dfb-b092-3fa193f9d488.png)

模型和数据资产 inspector 解决的是“跑完之后去哪找东西”。这类工具看起来不酷，但能大量减少团队沟通成本。

### Slide 16：小结：MLOps 基础设施

![](https://files.mdnice.com/user/59/77c21c39-fe50-4fe0-8b5d-12a8ca250fe8.png)

MLOps 小结页可以理解为：retrain-pipelines 把模型训练周边那些容易被脚本化但很难维护的东西，整理成标准 pipeline。

### Slide 17：Function calling 当前状态

![](https://files.mdnice.com/user/59/d1d89d19-5260-4a1a-990e-b1d687f463fe.png)

Function calling 当前状态页进入案例。工具调用模型要学会什么时候调用、调用哪个工具、参数 JSON 怎么写，以及什么时候不调用。

### Slide 18：Tool calling 与 constrained generation

![](https://files.mdnice.com/user/59/b2498fea-e5ed-4a27-bdc6-331e7f69027c.png)

Tool calling 与 constrained generation 说明输出格式很重要。function call 不是普通自然语言，JSON schema、参数类型、工具名都需要约束或后处理。

### Slide 19：Code interpreter 和工具响应

![](https://files.mdnice.com/user/59/095379e3-237a-432d-a9a9-6f51a9a87750.png)

Code interpreter 和工具响应页把任务扩展到更复杂环境。模型不仅生成调用，还要读工具返回，继续回答或再次调用。

### Slide 20：Completion API 与 Responses API

![](https://files.mdnice.com/user/59/ba9f696b-ba2e-4a75-bf1f-bb0d06eb7ce6.png)

Completion API 与 Responses API 页说明 API 形态变化会影响训练数据格式。adapter 如果绑定旧模板，服务端升级时就会出错。

### Slide 21：API 形态变化

![](https://files.mdnice.com/user/59/ebc3c6b8-8666-456d-8fbe-620f797e598d.png)

API 形态变化页继续强调兼容性。训练 pipeline 要把 prompt template、tool schema、response parser 作为 artifact 管起来。

### Slide 22：Berkeley Function-Calling leaderboard

![](https://files.mdnice.com/user/59/3e9272e4-5d06-4e8b-811a-157c3d5f07e2.png)

Berkeley Function-Calling leaderboard 给出评测参照。function calling 的评估不只是字符串匹配，还要解析 JSON、比较参数和工具选择。

### Slide 23：为什么需要专门 adapter

![](https://files.mdnice.com/user/59/46f26a5f-c6d1-4550-b7fe-d7c99f2bb6e9.png)

为什么需要专门 adapter？因为 base model 可能聊天能力很好，但工具调用格式不稳。LoRA adapter 可以专门强化这类结构化输出。

### Slide 24：Function-calling LoRA adapter

![](https://files.mdnice.com/user/59/204baf2d-3779-4c12-83e1-efed53e53f1e.png)

Function-calling LoRA adapter 页对应代码里的 Unsloth + PEFT 训练。它把工具 schema 放进 prompt，让 adapter 学会按工具协议输出。

### Slide 25：数据集和无工具调用样本

![](https://files.mdnice.com/user/59/d2a1d2d5-8dab-44db-b0a6-328e39e86308.png)

数据集页提到 legitimate absence of tool calls。这点很容易忽略：不是每个 query 都应该调用工具，训练集中必须保留“不调用”的正例。

### Slide 26：数据增强和 enrichment

![](https://files.mdnice.com/user/59/2a10f98f-540d-4f47-afbc-d221c2733ea1.png)

数据增强和 enrichment 用来扩展工具调用覆盖面。比如生成更多参数组合、改写 query、补充无调用样本，都会影响 adapter 的 precision/recall。

### Slide 27：PEFT/Unsloth CPT + SFT

![](https://files.mdnice.com/user/59/d0fa60a5-9ada-43f2-bbbc-6c8ba94bdb08.png)

PEFT/Unsloth CPT + SFT 页对应训练脚本。CPT 可以先适应工具格式和域数据，SFT 再对齐具体 function call 输出。

### Slide 28：评测结果：75.5%

![](https://files.mdnice.com/user/59/3239281d-0419-4a41-8022-73702425a300.png)

评测 75.5% 这页给出结果。更重要的是后面 false negative 分析，因为 function calling 评测很容易把语义等价的 JSON 判错。

### Slide 29：False negatives 第一类

![](https://files.mdnice.com/user/59/8149a8b7-01fe-403b-9ade-bec34416ed16.png)

False negatives 第一类通常来自参数等价但格式不同，或者工具调用顺序不影响结果。评测脚本需要在严格和宽松之间取平衡。

### Slide 30：False negatives 第二类

![](https://files.mdnice.com/user/59/0b837a76-e5e8-4cba-beba-f5b66137fea4.png)

False negatives 第二类可能来自数据标注或 parser 边界。工业 pipeline 里，失败样本回流比单次分数更有价值。

### Slide 31：Serving 需求

![](https://files.mdnice.com/user/59/cdbe24d3-bb73-4956-9e00-1381beb9ae3c.png)

Serving 需求页进入部署。训练完 adapter 后，要能和 base model 一起在线服务，并允许请求指定 adapter。

### Slide 32：Multi-adapter single endpoint

![](https://files.mdnice.com/user/59/9dbc203d-2f74-4471-9600-2fab5773c656.png)

Multi-adapter single endpoint 的好处是减少服务数量。一个 base model 常驻，多个 LoRA adapter 按请求切换，适合“很多小专家”的场景。

### Slide 33：每个 adapter 的 prompt template

![](https://files.mdnice.com/user/59/e7d336ac-3ae6-42ff-bc9b-925c275c57df.png)

每个 adapter 的 prompt template 必须跟 adapter 一起管理。function-calling adapter 如果用错 chat template，效果会比不用 adapter 还差。

### Slide 34：自定义 LitServe server

![](https://files.mdnice.com/user/59/efd8cfad-0d60-43b1-96d9-91ab69913f0e.png)

自定义 LitServe server 是公开代码里最直接的落点：启动时加载 base model 和多个 adapter，predict 时按 `adapter_name` set adapter。

### Slide 35：YAML 配置

![](https://files.mdnice.com/user/59/47f13cd9-9754-41f8-bb6f-7137632fa728.jpg)

YAML 配置让 adapter 列表、base model、端口、max tokens 这些部署参数不写死在代码里。这样 pipeline 可以产出 serving 配置并做 infra validation。

### Slide 36：Named adapters 开关

![](https://files.mdnice.com/user/59/f3b69c11-b2aa-4c09-90df-717a0d19e417.png)

Named adapters 开关页对应 PEFT 的 `set_adapter/enable_adapters/disable_adapters`。代码里还修了 `ModulesToSaveWrapper` 的启用问题。

### Slide 37：专家 adapter 集群

![](https://files.mdnice.com/user/59/d846d51a-53c2-4265-b75d-359b919791e3.png)

专家 adapter 集群可以理解为一堆小 LoRA 专家挂在同一个 base model 上。不同任务走不同 adapter，比维护很多 full model 轻得多。

### Slide 38：Army of specialized experts

![](https://files.mdnice.com/user/59/063f00d4-eecc-4c5b-9bde-0e232f8c74d6.png)

Army of specialized experts 是这个案例的产品化目标：持续学习不断产出小专家，服务层按请求选择专家。

### Slide 39：回到目录和开放问题

![](https://files.mdnice.com/user/59/5ea59869-6812-4db1-b2bd-041d459d7a51.png)

回到目录页通常是开放问题收束。continuous learning 的难点不是训练一次，而是让数据、训练、评测、发布和回滚长期可维护。

### Slide 40：结束页

![](https://files.mdnice.com/user/59/51344965-7af8-4acd-ae57-fed96e1ba667.png)

结束页。本文重点保留 pipeline 和 function-calling adapter 的代码脉络，方便后续复用。

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
