# 智源 FlagOS × SGLang 多芯片算子优化挑战赛：从可移植 Kernel 到生产级 Serving

作者：BBuf

这篇文章按 17 张 Slides 的顺序展开，但不是把画面上的文字再念一遍。我会补上每一页背后的 SGLang 源码、设计取舍、性能数据口径，以及它和下一页之间的关系。

整场分享想回答一个问题：智源 FlagOS × SGLang 多芯片算子优化挑战赛中的一个参赛实现，怎样从 portable kernel 出发，经过芯片专用优化、SGLang backend 接入和端到端验证，最后成为可以进入真实 Serving 路径的实现？

Slides 的源码审计快照固定在 SGLang `73a24142`，这样目录数量和注册数量可以复现。写作时我又拉取了最新 `origin/main@fe3d4b9bbb` 做复核。性能数字则保留各 PR 当时的硬件、软件、shape 和测试方法；“代码已经合入”不等于“所有历史数据都在最新 main 上重新测过”。

## Slide 1：SGLang Multi-Chip Operator Optimization

![Slide 1：SGLang Multi-Chip Operator Optimization](https://files.mdnice.com/user/59/ce1615eb-2e6a-4cec-b983-bdd4a2550d4d.png)

封面把活动背景和整场分享的判断放在一起：Serving performance depends on many kernels。

这里的 “many kernels” 不只是说模型里 GEMM 数量很多。一条请求还会经过 Attention、KV Cache、量化、通信、采样、数据重排和调度。某个 kernel 的耗时下降，只说明局部候选值得继续验证；只有 TTFT、TPOT、吞吐或 diffusion denoise latency 也改善，才说明它真正帮助了 Serving。

这正是多芯片比赛与单卡 microbenchmark 的连接点。比赛要求参赛者面对真实 SGLang task，在共同的算子语义下为不同硬件寻找更好的实现；SGLang 则负责把候选放回完整调用链，检查它能否安全地工作。

封面右侧的 compute、memory、collectives、dispatch 是后续内容的四条线：计算本身要快，数据要以合适的 layout 移动，多卡通信要匹配消息大小，最后还要确保 dispatch 只把已验证的输入送入 fast path。

## Slide 2：比赛同时考查三个目标

![Slide 2：The challenge has three optimization goals](https://files.mdnice.com/user/59/776ba49d-6ea1-4a1c-a268-d77f8ee7f592.png)

我把一个完整提交拆成三层。

第一层是 portability。不同芯片上的实现必须遵守同一个 operator contract：输入、输出、dtype、layout、数值语义和异常行为不能因为 backend 改变。Portable kernel 不一定是最终最快的版本，但它给出了共同语义和可比较的参考路径。

第二层是 specialization。真正的性能通常来自对目标硬件的理解，例如 tile 形状、shared memory/TMEM 使用方式、向量化宽度、warp 分工、异步拷贝、量化格式和拓扑。不同芯片可以有不同 fast path，只要上层 operator 不变。

第三层是 serving safety。一个实现还必须回答：不支持的 shape 是否回退？CUDA Graph capture 和 replay 是否正确？多 rank 是否共享了错误状态？模型精度和端到端指标是否保持？

这三层可以写成：

```text
operator contract × target architecture × real workload
```

挑战赛中的好成绩是第一批证据，而不是最后一道部署决定。后面的 Slides 会依次解释 SGLang 怎样承接这三层。

## Slide 3：一次请求会使用六类 kernel

![Slide 3：One request uses six kernel families](https://files.mdnice.com/user/59/2bc4a649-c520-4b54-ac42-35e8ba315c09.png)

一条 LLM 请求从 prefill 进入 decode，至少会碰到六类工作：

- Attention + KV：QK、softmax、PV、RoPE、KV pack 和分页缓存；
- GEMM + Quant：BF16、FP8、NVFP4，以及 per-token/per-group scale；
- MoE：routing、top-k、permute、grouped GEMM 和 expert fusion；
- Communication：all-reduce、all-gather、reduce-scatter；
- Sampling + Speculative：grammar mask、top-k/top-p、draft tree verification；
- State + Diffusion：Mamba/linear attention state、DiT fusion 和 layout 变换。

所以副标题使用的是 “kernel work, metadata, data movement, and collectives”。metadata 不是装饰信息。以 paged Attention 为例，kernel 必须知道每个请求的序列长度、页索引、KV 地址和运行模式；metadata 构造得慢，或者 CUDA Graph replay 读取了旧地址，inner loop 再快也没有意义。

同样，吞吐也不是单个 kernel latency 的同义词。高并发下，调度、L2 占用、workspace、通信和后续算子都会改变最终结果。这一页底部那句话最重要：最快的 isolated kernel，仍然可能输掉端到端 Serving。第 12、13 页会给出真实反例。

## Slide 4：SGLang 把 Serving kernels 放进统一 API

![Slide 4：SGLang groups serving kernels under one API](https://files.mdnice.com/user/59/2f42e39c-c0e6-4733-af82-b1c7cb798543.png)

这一页是源码范围审计，不是 kernel 数量排行榜。

在固定快照 `73a24142` 上，`python/sglang/kernels/ops/` 有 21 个顶层 domain、502 个 Python 文件；Attention registry 注册了 24 个名字，其中 `nsa` 是指向 `dsa` 的兼容 alias。502 的准确含义是 “Python files under ops/”，里面包括 API、wrapper、metadata、site hook、backend adapter 和兼容代码，绝不能读成 502 个独立 GPU kernel。

到本文复核的最新 `origin/main@fe3d4b9bbb`，顶层 domain 仍是 21 个，Python 文件已变为 510。这个变化正好说明为什么 Slides 必须固定 source baseline：main 在继续前进，而演讲中的统计需要可复现。

源码目录按逻辑功能分组，而不是按 CUDA/Triton/CuTe DSL 分组：

```text
python/sglang/kernels/ops/
├── attention/       # Attention 与部分 KV 相关 operator
├── gemm/            # GEMM 及专用矩阵乘路径
├── communication/   # all-reduce 等通信 operator
├── moe/             # routing、permute、grouped GEMM
├── sampling/        # 采样
├── diffusion/       # diffusion fusion、layout、RoPE、site
└── ...
```

这样组织的原因很直接：模型代码关心“我要调用哪个逻辑算子”，不应该关心“它今天恰好由哪种语言实现”。真正的 backend 选择留给下一层 registry。

源码入口：[kernels/ops](https://github.com/sgl-project/sglang/tree/73a24142895cbd169bc9f699fd72dfa6e4f61c15/python/sglang/kernels/ops)、[attention_registry.py](https://github.com/sgl-project/sglang/blob/73a24142895cbd169bc9f699fd72dfa6e4f61c15/python/sglang/srt/layers/attention/attention_registry.py)。

## Slide 5：一个逻辑 operator，多种实现

![Slide 5：One logical operator, many implementations](https://files.mdnice.com/user/59/943abce7-fc52-4ece-9a67-ebf51994d966.png)

这是整套架构的核心页。模型和 runtime 依赖稳定入口：

```text
sglang.kernels.ops.<group>.<operator>
```

底层实现则由 `KernelSpec` 或 `BaseFusedOp` 描述。以 Qwen3.x NVFP4 GEMM 为例，main 中的注册大致如下：

```python
register_kernel(
    KernelSpec(
        op="gemm.qwen3x_nvfp4",
        backend=KernelBackend.KDA,
        target=f"{_KDA_PACKAGE}.qwen3x_nvfp4_gemm:try_qwen3x_nvfp4_gemm",
        capabilities=_SM120,
    )
)
```

`op` 是稳定 lookup key，`backend` 记录实现来源，`target` 是延迟导入的 Python callable，`capabilities` 决定当前平台是否有资格运行。`KernelSpec.load()` 在实际调用前不会导入重型依赖，因此 `import sglang.kernels` 不会顺带加载 Triton、CUTLASS 或触发 JIT。

`BaseFusedOp` 又把同一 operator 的多个 `forward_<backend>` 放在一个 `nn.Module` 后面。当前 dispatch 顺序是：显式 backend、全局强制 backend、out-of-tree 平台覆盖、满足 capability 的优化 backend、平台专用 forward、最后回到 Pure-Torch reference。源码中的默认优化优先级以 KDA 开始，但“优先尝试”不等于“无条件执行”；每个 operator 仍可通过 `backend_eligible()` 增加 shape/dtype gate。

Slides 底部把 runtime backend 与 agent candidate source 放在一行，是为了展示生态关系，不是说每个标签都属于 `KernelBackend` 枚举。特别要区分：

- CUDA JIT、Triton、CuTe DSL 回答代码如何编译或表达；
- KDA 回答实现来自 Kernel Design Agents 工作流；
- CAKE 在这里表示另一条 agent 候选生成路径，当前 main 并没有 `KernelBackend.CAKE`；
- `torch_compile` 仍存在于源码枚举，只是为了版面空间没有放进这一行，自动 dispatch 也不会意外触发它。

排查数值问题时可以设置：

```bash
SGLANG_FORCE_FUSED_OP_BACKEND=torch
```

这会尽量让整模型回到 reference path。若某个 operator 没有被强制 backend 覆盖，全局 debug 模式会逐 operator 告警并继续正常 fallback，而不是让整模型直接不可用。

源码入口：[spec.py](https://github.com/sgl-project/sglang/blob/fe3d4b9bbbff744d056dda122ea9157c2932a2bd/python/sglang/kernels/spec.py)、[fused_op.py](https://github.com/sgl-project/sglang/blob/fe3d4b9bbbff744d056dda122ea9157c2932a2bd/python/sglang/kernels/fused_op.py)、[ops/gemm/__init__.py](https://github.com/sgl-project/sglang/blob/fe3d4b9bbbff744d056dda122ea9157c2932a2bd/python/sglang/kernels/ops/gemm/__init__.py)。

## Slide 6：JIT 在运行时专门化，但不改变 API

![Slide 6：JIT kernels specialize at runtime](https://files.mdnice.com/user/59/87b308ce-34c0-4ec8-9264-59d22b812e0f.png)

SGLang 的 `jit_kernel` 适合轻量、需要按当前环境专门化的 CUDA/HIP 模板。上层 operator contract 不变，`BuildSpec` 收集 source、flags、target、wrapper exports，再由 `load_jit()` 查找或构建 TVM-FFI module。

这套机制最难的部分不是调用一次 NVCC，而是缓存正确性。当前缓存有两级 key：

```text
build_key：编译前已知的参数、flags、target、直接源码内容
deps_key：编译器枚举出的完整传递依赖内容
```

假设依赖链是：

```text
kernel.cu → common.cuh → sm120_utils.cuh
```

`sm120_utils.cuh` 没有被入口文件直接 include，但它仍是 transitive dependency，也就是间接构建输入。只要链上任何文件的内容变化，`deps_key` 就必须变化，否则服务可能复用旧 `.so`。源码会对依赖内容做 SHA256，并把每个 immutable leaf 放在自己的 `deps-<key>/` 目录，而不是维护一个多进程共同修改的 manifest。

多 rank 启动还有另一个竞态。`loader.py` 先查缓存；miss 后获取文件锁；拿到锁之后再查一次。第二次检查把 N 个同时启动的 TP rank 变成“一次编译 + N-1 次 cache hit”。构建先进入私有 `.staging-<uuid>`，模块能够成功加载后才 atomic rename 发布，其他进程不会看到半成品。

源码中的关键流程可以缩成：

```python
prebuilt = cache.find_prebuilt(...)
if prebuilt is not None:
    return _load(prebuilt)

with _build_lock(scope):
    prebuilt = cache.find_prebuilt(...)  # lock 后再次检查
    if prebuilt is not None:
        return _load(prebuilt)
    # build in private staging, verify load, then publish atomically
```

Slides 上的 5.26 s、0.02 s、0.05 s 是 PR #34274 的 H100 启动/加载时间：分别对应 fresh build、同 clone 缓存加载和第二个 clone 复用共享缓存。它们不是 kernel 执行 latency。这个数量级差异解释了为什么 cache correctness 和共享能力决定 JIT 能否成为正常部署路径。

源码入口：[cache.py](https://github.com/sgl-project/sglang/blob/fe3d4b9bbbff744d056dda122ea9157c2932a2bd/python/sglang/kernels/jit/utils/compile/cache.py)、[loader.py](https://github.com/sgl-project/sglang/blob/fe3d4b9bbbff744d056dda122ea9157c2932a2bd/python/sglang/kernels/jit/utils/compile/loader.py)。

## Slide 7：Attention 不是一个 kernel 名字

![Slide 7：Attention uses several kernels and backend-specific logic](https://files.mdnice.com/user/59/a0a863c0-11f0-4226-86ac-58c15cb91f6f.png)

Attention 经常被简化成 `softmax(QKᵀ)V`，但 Serving 还要处理 metadata、KV layout、CUDA Graph、prefill/decode/speculative 模式和硬件差异。

`attention_registry.py` 的注册方式很朴素：decorator 把名字映射到 backend 构造函数，真正的模块仍然延迟导入。

```python
ATTENTION_BACKENDS = {}

def register_attention_backend(name):
    def decorator(fn):
        ATTENTION_BACKENDS[name] = fn
        return fn
    return decorator
```

简单的字典背后有大量 eligibility 规则。例如 `trtllm_mla` 只接受 MLA 模型，在特定 DCP + speculative decoding 组合下还会拒绝运行，因为它没有传递循环 DCP metadata，也没有返回跨 rank merge 所需的局部 LSE。`fa3` 会检查设备 capability；`triton` 对 encoder-decoder cross attention 有限制。也就是说，registry 列出的 24 个名称不是 24 个实现同时执行，而是 24 条带条件的候选路径。

Slides 把候选分为 Dense、MLA、Sparse + Platform 三组，方便理解覆盖面：FlashInfer/FA3/FA4、FlashMLA/CuTe DSL MLA、DSA/DSV4、AITER/Wave、Ascend、Intel AMX/XPU 等。`nsa` 只是 `dsa` 的 deprecated alias，因此页面特意写了 “1 is a deprecated alias”。

这里与第 5 页的共同点是：上层 API 保持不变，选择发生在 registry 和 backend 的运行条件里。不同的是，Attention 的条件不仅是 GPU 型号，还与模型结构、执行阶段、并行模式和 metadata contract 有关。

源码入口：[attention_registry.py](https://github.com/sgl-project/sglang/blob/fe3d4b9bbbff744d056dda122ea9157c2932a2bd/python/sglang/srt/layers/attention/attention_registry.py)。

## Slide 8：custom_allreduce_v2 把存储与算法分开

![Slide 8：custom_allreduce_v2 separates storage from algorithms](https://files.mdnice.com/user/59/6284b0db-e5e8-4eef-ac69-c2ebbbbcac6f.png)

多芯片 Serving 的通信不能只写成一个 `all_reduce(x)`。消息大小、world size、拓扑、eager/graph 模式和 multicast 能力都会改变最佳算法。

先看 v1。Legacy `CustomAllreduce` 也是由 Python 创建内存，并不是所有资源都在 C++ 中分配。它会准备 `meta_ptrs`、eager 模式使用的 `buffer_ptrs` 和保存 graph buffer 地址的 `rank_data`，再通过 `ops.init_custom_ar(...)` 得到一个 `_ptr` handle：

```python
self.meta_ptrs = self.create_shared_buffer(ops.meta_size() + max_size)
self.buffer_ptrs = self.create_shared_buffer(max_size)
self.rank_data = torch.empty(max_size, dtype=torch.uint8, device=self.device)
self._ptr = ops.init_custom_ar(self.meta_ptrs, self.rank_data, rank, full_nvlink)
ops.register_buffer(self._ptr, self.buffer_ptrs)
```

这套 v1 设计能处理 eager 与 CUDA Graph，但 IPC buffer、同步 metadata、graph address registration 和执行路径都收在同一个 legacy handle 后面。`custom_all_reduce()` 的调用侧主要判断 input 是 registered 还是 unregistered；`should_custom_ar()` 主要检查连续性、world size、NVLink 和最大消息大小。具体算法不以独立 storage plane 的形式暴露。

CAR v2 的关键重构是 decoupled storage plane。Python 先分配并持有 symmetric-memory slab，再切成 PushPlane 和 PullPlane：

- PushPlane：`2 × world_size` 个 slot 对应两个 phase，另有 rank-local phase counter；
- PullPlane：pull workspace、per-block semaphore，以及可选 multicast 地址；
- Communicator：持有两个 plane，CUDA kernel 只接收稳定 handle；
- 算法：`1shot_push`、`1shot_pull`、`2shot_pull` 独立选择。

源码 `_init_workspace()` 只做一次 symmetric-memory rendezvous，然后用 offset 切出所有子区域。这样 storage 生命周期、graph pointer、multicast 映射和具体算法可以分别演进。CUDA Graph 输入在 capture 后交换地址，replay 时 kernel 从 device-side pointer table 读取当前 row，而不是把一次 capture 的 host pointer 假设成永远有效。

因此，v1 与 v2 的差别不是“Python 管内存”对“C++ 管内存”，而是 monolithic handle 对 decoupled planes。v2 把 storage contract 写清后，算法选择可以独立发生：同一个 Communicator 能按当前调用走 push、pull 或 two-shot，不需要为每种算法重新设计一套 workspace 生命周期。

算法选择也写得很直白：

```python
if nbytes <= one_shot_push_threshold:
    return ONE_SHOT_PUSH
if nbytes <= one_shot_pull_threshold:
    return ONE_SHOT_PULL
if multicast_range.contains(nbytes):
    return TWO_SHOT_PULL, use_multicast=True
if nbytes <= two_shot_pull_threshold:
    return TWO_SHOT_PULL
return None  # 回到 NCCL 等其他路径
```

Slides 表格来自 8×B200、BF16，单位是微秒：

| 消息大小 | NCCL | AOT v1 | JIT graph v2 |
| --- | ---: | ---: | ---: |
| 4 KB | 26.6 | 6.8 | 3.9 |
| 256 KB | 31.1 | 26.0 | 6.6 |
| 1 MB | 32.1 | 26.9 | 12.4 |
| 16 MB | 108.1 | 147.1 | 53.2 |

这些数据证明 CAR v2 在给定硬件和 graph workload 上很有价值，但不能推出“自定义 AllReduce 对所有消息都比 NCCL 快”。16 MB 行里 AOT v1 已经慢于 NCCL，正好说明 crossover 和 fallback 是算法的一部分，不是失败后的补丁。

当前 `dispatch_custom_allreduce()` 在 CUDA 上默认选择 JIT v2；设置 `SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=0` 可以回到 legacy v1。ROCm 和 MUSA 仍保留各自的实现分支，所以这张 Slides 只比较 8×B200 BF16 的 CUDA 路径，不能扩展成跨平台结论。

源码入口：[legacy custom_all_reduce.py](https://github.com/sgl-project/sglang/blob/fe3d4b9bbbff744d056dda122ea9157c2932a2bd/python/sglang/srt/distributed/device_communicators/custom_all_reduce.py)、[custom_all_reduce_v2.py](https://github.com/sgl-project/sglang/blob/fe3d4b9bbbff744d056dda122ea9157c2932a2bd/python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py)、[all_reduce.py](https://github.com/sgl-project/sglang/blob/fe3d4b9bbbff744d056dda122ea9157c2932a2bd/python/sglang/kernels/ops/communication/all_reduce.py)。

## Slide 9：端到端收益来自完整 Serving path

![Slide 9：End-to-end speedup comes from the full serving path](https://files.mdnice.com/user/59/6f6619c0-5394-4f7e-b3c6-d052a26ea79d.png)

Qwen-Image 是一个很适合解释系统组合收益的例子。PR #36680 同时处理了三条路径：

- QKV pack：47.73 μs → 21.10 μs，约 2.26×；
- FA3 scheduler：159.40 μs → 147.06 μs，约 1.08×；
- 24 MiB collective：104.20 μs → 81.75 μs，约 1.27×。

第一项新增 `fused_pack_segmented_qkv`。它把 prefix/main 看成一条虚拟拼接序列，直接根据 `indices` 从两段来源 gather Q/K/V，不再先物化三份 dense concatenation。核心索引逻辑是：

```python
row_in_batch = src_row - batch * (PREFIX_ROWS + MAIN_ROWS)
from_prefix = row_in_batch < PREFIX_ROWS
# 根据 from_prefix 从 prefix 或 main 读取，再直接写 packed output
```

第二项改变 Attention scheduling。第三项把 diffusion 的 custom-AR workspace 上限扩到 32 MiB，因为 1024×1024 Qwen-Image 的 TP row-parallel 输出是 24 MiB；默认 16 MiB workspace 会让它回到 NCCL。这个例子能看出，kernel、layout 和 runtime 配置必须一起修改。

三个局部倍数不能相乘。它们在完整 denoise path 中占比不同，还可能相互影响。最终模型结果是：Qwen-Image-2512 在 TP2 + BCG、`quality=high` 下从 8.5406 s 降到 7.8657 s；原 PR 用 `baseline / candidate - 1` 报告 +8.58%。若使用 `(baseline - candidate) / baseline` 计算 latency reduction，会得到 7.90%，两者只是分母不同。Z-Image-Turbo TP2 报告 +5.05%。

`quality=lossless` 仍保留三次独立 added-QKV GEMM。源码中的 site hook 会按请求质量挂载 fusion，因为把三次 BF16 reduction 合并成一次会改变加法结合顺序，不能声称 bit-exact。这种按质量档位保留 reference path 的做法，比“所有请求默认走快路径”更符合生产语义。

源码入口：[fused segmented QKV](https://github.com/sgl-project/sglang/blob/71cee04ebe8061af901e4880169f6a5e86f7c8c1/python/sglang/kernels/ops/diffusion/layout/varlen_pack_pad_triton.py)、[quality-gated site](https://github.com/sgl-project/sglang/blob/71cee04ebe8061af901e4880169f6a5e86f7c8c1/python/sglang/kernels/ops/diffusion/sites/qwen_image_added_qkv_site.py)、[diffusion group coordinator](https://github.com/sgl-project/sglang/blob/71cee04ebe8061af901e4880169f6a5e86f7c8c1/python/sglang/multimodal_gen/runtime/distributed/group_coordinator.py)。

## Slide 10：为什么转到 Agent-native kernel development

![Slide 10：SGLang also needs a repeatable kernel development loop](https://files.mdnice.com/user/59/d3105dfb-dd88-44fd-b104-ca7d85573dbb.png)

前九页讲的是 runtime：稳定 operator、backend、capability、fallback、Attention 和 collective。到了这一页，问题才从“运行时怎样选择 kernel”转向“社区怎样持续开发大量 kernel”。这个转场很重要，否则 KDA 会像一个突然出现的新主题。

当 operator API 稳定以后，开发过程仍然不断重复：从 trace 选择真实 shape，profile，修改实现，构建，correctness replay，CUDA Graph，模型 E2E，再进入 review。Agent-native coding 的目标不是降低验收标准，而是提高这条循环的吞吐。

Slides 右侧四个角色有清楚的边界：

- Skills 保存可重复执行的工程步骤；
- Humanize 维护长任务状态，并在循环中加入独立 review；
- KDA 搜索和生成更快的 kernel 候选；
- SGLang 定义 operator contract、运行路径和 release checks。

源码里的 `enable_fused_op_trace()` 很能说明两部分怎样接起来。它记录每次 fused-op 调用的 op、backend、tensor shape 和 dtype，为“哪些 production shape 值得优化”提供真实输入。Agent 可以运行更多实验，但每个候选仍然必须经过相同 dispatch、replay、accuracy 和 E2E 检查。

这一页不是在说 reviewer 可以被 Agent 替代。恰恰相反，重复劳动被自动化以后，人更应该把精力放在 contract、风险边界、实验是否公平，以及结果是否足以进入默认路径。

## Slide 11：KDA 生成候选，SGLang 在 Serving 中验证

![Slide 11：KDA generates kernels; SGLang validates them in serving](https://files.mdnice.com/user/59/47a740b7-313e-4b42-99d4-7c938ff20e44.png)

这里的 KDA 指 Kernel Design Agents，不是 Kimi Delta Attention。

左侧是 LMSYS《Agent-Assisted SGLang Development》公开的 B200 acceleration figure，覆盖 10 个 diffusion kernel task，图中结果约为 1.11×–2.7499×。它证明 Agent 可以连续处理一组真实工程任务，但这是公开时间点的工作流快照，不是今天的实时排行榜，也不表示图中每个候选都已经默认上线。

右侧三张卡故意把“kernel 结果”和“Serving 结果”写在一起，但口径必须分开读。

Diffusion portfolio 覆盖四类 kernel family：norm + scale/shift、causal Conv3D cat-pad、residual-gate add、LTX2 QKNorm + split-RoPE。其 kernel 收益为 1.279×–5.84×，同时继续检查 denoise 或 full-model run。相关实现可以在 `python/sglang/kernels/kda_kernels/` 找到。

SM120 NVFP4 GEMM 在 16 个精确 production row 上得到 1.319× kernel geomean；完整 Qwen3.5 Serving 中，4B throughput 提升 6.52%–8.73%，9B 提升 2.70%–3.18%。第 13 页会展开每个并发点。

Qwen3.8 packed QSA 是 PR #36845 的独立 validation package，不属于第 14 页统计的 main KernelSpec registry。10.66× 是 batch size 128 上的 kernel microbenchmark，不是模型 E2E。15 个真实 capture 的 kernel geomean 是 2.0702×；扩展 batch 17–128 的 42 个 case geomean 是 4.48×，最慢 case 仍有 1.58×。进入完整 TP1 NVFP4 + NEXTN Serving 后，吞吐提升是 4.00%–4.45%；GSM8K 为 49/50 对 49/50。

QSA 还做了两类很关键的状态验证。第一类是 15/15 production tensor replay，并检查每一行确实只有一次目标 CUDA activity；第二类是 150,000 次连续 launch，确认计数器每次回到零。单元测试还会在 graph capture 后修改 `cu_seqlens` 与 Q/K/V，再 replay 并与 reference 比较，避免 kernel 偷用 capture 时的旧 metadata。

源码入口：[QSA package README at merge commit](https://github.com/sgl-project/sglang/blob/78c5024e9d9f589dcb4deb7f4ba4fb23f7e85385/python/sglang/kernels/kda_kernels/qwen38_qsa_sm121/README.md)、[QSA CUDA Graph tests](https://github.com/sgl-project/sglang/blob/78c5024e9d9f589dcb4deb7f4ba4fb23f7e85385/test/registered/kernels/test_kda_qsa_sm121.py)、[KDA kernels README on main](https://github.com/sgl-project/sglang/blob/fe3d4b9bbbff744d056dda122ea9157c2932a2bd/python/sglang/kernels/kda_kernels/README.md)。

## Slide 12：近期 kernel 开发中遇到的三种 reward hacking

![Slide 12：Three reward-hacking problems we hit in kernel work](https://files.mdnice.com/user/59/fc03718b-dbdd-4f35-a6f7-820564abe6bc.png)

这里的 reward hacking 不是说 Agent 恶意作弊，而是 benchmark 奖励了错误的东西。结果可能更好看，却未必来自一个正确、可交付并且能改善完整 Serving 的 kernel。

近期 KDA-Pilot 和 SGLang kernel 开发中反复出现的问题，可以更直接地归成三类。

第一类是 **the path changed**。早期实验曾让 candidate 绕过带 `@register_custom_op` 的 production wrapper，表面收益约为 1.22×；baseline 和 candidate 回到同一个 public API、wrapper 与 dispatcher 后，可归因给 device kernel 的收益约为 1.12×。另一个风险是 silent fallback：如果候选路径不满足条件后悄悄回到 reference，benchmark 可能根本没有测到新 kernel。现在的规则是两边必须走同一 host path，并通过 trace、counter 或 dispatch 记录确认新实现确实运行。

第二类是 **the test changed**。candidate 单边打开 fast math、使用更宽松的 tolerance、没有检查 NaN/Inf，或没有确认 output 被完整覆写，都会让错误结果显得更快。看完数据后删掉较慢 shape，或只报告一个 geomean，也是在改变原始 workload。修正方法是在调优开始前冻结双方 compiler flags、reference、tolerance 和 shape set；output 可以先填 poison value，并按任务语义检查 exact、bitwise 或预先约定的误差界。每个 production row 都应保留，允许 specialization、fallback 和 no-go，而不是强迫每个 shape 都出现正收益。

第三类是 **only the kernel won**。isolated kernel 变快，仍可能因为 cache pressure、CUDA Graph replay 中的旧状态、workspace 复用或下游 kernel miss 让完整系统变慢。QSA 的 15/15 production replay 和 150,000 次连续 launch，是为了证明 state 能正确复位、graph replay 不会偷用旧 metadata；但这些仍只是进入模型验证的前置条件。最后还必须测 accuracy、TPOT、throughput 或 diffusion denoise latency。下一页 Qwen3.8-27B 的 L2 反例正是这种 system-level reward mismatch。

因此，判断链可以压缩为：

```text
same workload
→ correct output
→ confirmed dispatch
→ end-to-end improvement
```

前三项证明 benchmark 有效，最后一项决定实现能否进入生产路径。microbenchmark 胜利只是一个候选，不是部署结论。挑战赛里也应该优化真实 operator contract，而不是利用 benchmark 恰好留下的缺口。

相关工程记录可参考 [KDA-Pilot](https://github.com/BBuf/KDA-Pilot)、[QSA production validation](https://github.com/sgl-project/sglang/pull/36845) 和 [Qwen3.x NVFP4 GEMM integration](https://github.com/sgl-project/sglang/pull/36865)。

## Slide 13：Qwen3.5-4B 与 9B 的端到端 Serving 确实变快

![Slide 13：Qwen3.5-4B and 9B improve end-to-end serving](https://files.mdnice.com/user/59/8dbcde01-1564-4bb7-9e59-74a90d1bc18b.png)

这一页先给正向结论，再用 Qwen3.8-27B 的失败解释为什么 dispatch 必须精确。

数据来自 #36865 合入前在 PR head `73d4809cf4` 上进行的 RTX PRO 6000 / SM120、FlashInfer 0.6.18 验证，每个配置 32 个请求、3 轮。它们不是在本文最新 main 上重新跑出的数字。

| 模型 | 并发 | Throughput | TPOT | E2E latency |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-4B | 1 | +8.73% | -8.31% | -8.03% |
| Qwen3.5-4B | 2 | +6.84% | -6.85% | -6.41% |
| Qwen3.5-4B | 4 | +7.12% | -7.02% | -6.65% |
| Qwen3.5-4B | 8 | +6.52% | -6.63% | -6.10% |
| Qwen3.5-9B | 1 | +3.18% | -3.19% | -3.08% |
| Qwen3.5-9B | 2 | +2.78% | -2.82% | -2.71% |
| Qwen3.5-9B | 4 | +2.82% | -2.81% | -2.74% |
| Qwen3.5-9B | 8 | +2.70% | -2.78% | -2.62% |

4B 和 9B 在每个已测并发点都获得正 throughput，并且 TPOT、E2E latency 同步下降。这就是标题的直接含义，不需要用 “gains survive the full stack” 这类容易产生歧义的说法。

为什么收紧 dispatch 后，这两组收益仍然存在？因为当前 gate 不是关闭 KDA，而是只开放已验证的 shape。main 源码把支持集合写得非常具体：

```python
_SUPPORTED_SHAPES = {
    (m, k, n)
    for m in (1, 2, 4, 8)
    for (k, n) in (
        (2560, 18432), (9216, 2560),
        (4096, 24576), (12288, 4096),
    )
} | {(9, 17408, 5120)}
```

此外 `_supports()` 还检查 CUDA device、二维 packed input、uint8/FP8 scale dtype、BF16 output、stride、scale shape、同 device，以及设备 capability 是否为 SM120。不满足任一条件时 `try_qwen3x_nvfp4_gemm()` 返回 `None`，调用方继续 fallback。

右侧 Qwen3.8-27B 不是第三个加速 claim，而是负面证据。早期 broad dispatch 让每层 5.6–11 MiB 的 scale tensor 长时间占用 L2。isolated GEMM 变快了，但 Attention/SSM state 被挤出 L2，完整模型 output throughput 下降 0.76%。后续改成 weights/scales one-pass streaming，并只对 `(M,K,N)=(9,17408,5120)` 开启；其他 shape 回退，端到端结果转为 +0.98%。

这里的重点不是“精确 shape 总比宽 dispatch 好”，而是 promotion 只能覆盖已经通过模型级验证的模型与 shape。27B 的失败数据帮助我们定义了更安全的上线边界。

源码入口：[qwen3x_nvfp4_gemm.py](https://github.com/sgl-project/sglang/blob/fe3d4b9bbbff744d056dda122ea9157c2932a2bd/python/sglang/kernels/kda_kernels/qwen3x_nvfp4_gemm.py)、[merge commit](https://github.com/sgl-project/sglang/commit/c593527f33ee4c0f6068d7e1d7dd9052a4f26f0d)。

## Slide 14：main 中有 9 个 KDA KernelSpec registration

![Slide 14：main has nine KDA KernelSpec registrations](https://files.mdnice.com/user/59/c7fd1c12-27d2-49a5-a746-39a7c9a4f6eb.png)

这一页从“验证方法”回到“main 中到底落了什么代码”。

`KernelBackend.KDA = "KDA"` 已经是正式枚举值。源码对 backend 的定义是 implementation provenance，而不是 device：KDA 表明实现来自 Kernel Design Agents 工作流，但这个实现仍可能用 CUDA JIT、Triton 或 CuTe DSL。硬件支持由每个 `KernelSpec.capabilities` 单独声明。

在 `origin/main@fe3d4b9bbb` 中，共有 9 个 KDA `KernelSpec` registration：

- 6 个 target 指向 CUDA JIT 入口；
- 2 个 target 指向 Triton 实现；
- 1 个 target 指向 CuTe DSL 的 Qwen3.x NVFP4 GEMM。

其中 8 个 registration 位于 `ops/diffusion/__init__.py`，1 个位于 `ops/gemm/__init__.py`。PR #36845 的 QSA validation package 不在这 9 个 `KernelSpec` 中。diffusion 的例子包括 B200 norm + scale/shift、residual-gate add、causal Conv3D cat-pad、LTX2 QKNorm + split-RoPE，以及 FLUX.2 的 Triton/JIT fusion。

生成实现放在 `sglang/kernels/kda_kernels/`，稳定 facade 和注册留在 `sglang/kernels/ops/`。README 记录 kernel family、实现文件、来源 PR 和精确 revision。模型代码因此只依赖公开 operator；generated implementation 可以继续迭代而不污染调用点。

这层隔离也没有给生成代码“免检通行证”。`KernelBackend.KDA` 只记录来源，不证明性能、正确性或硬件覆盖。是否默认选中，仍取决于 priority、capability、per-call shape gate 和完整验证证据。

源码入口：[KernelBackend](https://github.com/sgl-project/sglang/blob/fe3d4b9bbbff744d056dda122ea9157c2932a2bd/python/sglang/kernels/spec.py)、[KDA implementations README](https://github.com/sgl-project/sglang/blob/fe3d4b9bbbff744d056dda122ea9157c2932a2bd/python/sglang/kernels/kda_kernels/README.md)、[diffusion registrations](https://github.com/sgl-project/sglang/blob/fe3d4b9bbbff744d056dda122ea9157c2932a2bd/python/sglang/kernels/ops/diffusion/__init__.py)。

## Slide 15：从比赛提交走到 SGLang upstream

![Slide 15：From competition submission to SGLang upstream](https://files.mdnice.com/user/59/be936517-1c3e-41f0-9d94-8e4dafb0f9a3.png)

这一页把前面的架构和案例压缩成参赛者可以执行的四步。

第一步 Measure。不要只选择“看上去像模型 shape”的随机矩阵。可以用 fused-op trace 或真实模型 profile 收集 op、backend、dtype、shape 和调用频率，再确定 benchmark rows。高频 decode shape、偶发 prefill shape 和 graph replay shape 应分开看。

第二步 Contract。先写清输入输出、dtype/layout、数值标准、in-place 语义、unsupported case 与 fallback。Qwen-Image 的 `quality=lossless` 和 Qwen3.x NVFP4 的 `_supports()` 都是 contract 的具体表现。

第三步 Optimize。根据 workload 选择合适工具和硬件特性。一个轻量模板可能适合 SGLang JIT；SM120 NVFP4 GEMM 可以使用 CuTe DSL；简单 data movement 可以用 Triton；通信则需要考虑 symmetric memory、multicast 和 graph pointer。技术选择服从问题，不应该先指定语言再寻找任务。

第四步 Validate。至少依次通过 reference correctness、逐 shape profile、实际 dispatch、CUDA Graph capture/replay、长时间状态测试、模型 accuracy 与 E2E。只有 benchmark 数字而没有 executed-path 证据，很难判断测到的是 candidate 还是 fallback。

SGLang 能提供的是稳定 operator API、多 backend、reference path、JIT specialization、workload trace 和 safe fallback。参赛实现提供的是新的硬件优化。两边合起来，才形成：

```text
competition submission
  → qualified SGLang backend
  → production serving
```

排行榜胜利说明候选值得继续；可复现的完整证据才支持 upstream merge。

## Slide 16：工程与比赛入口

![Slide 16：Engineering and competition links](https://files.mdnice.com/user/59/b2f675bd-70b1-493c-b963-82d407b0e408.png)

这一页把前文的两条线集中起来。左侧是 Agent-native SGLang 工程资源，右侧是智源 FlagOS × SGLang 多芯片算子优化挑战赛的公开入口。Slides 中的卡片和二维码适合现场扫码，文章里直接列出可点击链接。

工程侧：

- [LMSYS：Agent-Assisted SGLang Development](https://www.lmsys.org/blog/2026-07-02-agent-assisted-sglang-development)
- [SGLang executable skills](https://github.com/sgl-project/sglang/tree/main/.claude/skills)
- [BBuf AI-Infra Skills](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS)
- [Kernel Design Agents](https://github.com/mit-han-lab/kernel-design-agents)
- [Humanize / RLCR](https://github.com/PolyArch/humanize)

挑战赛侧：

- [FlagOS × SGLang 挑战赛详情](https://flagos.io/race-detail-season2?id=782kzq4m&lang=en)
- [报名入口](https://flagos.io/Register?raceId=8q4m2x7p&lang=en)
- [竞赛活动中心](https://flagos.io/events?tab=competition)
- [任务列表与 KernelGen](https://kernelgen.flagos.io/web)
- [提交仓库 FlagGems-sglang](https://github.com/flagos-ai/FlagGems-sglang)
- [Contributor License Agreement](https://cla-assistant.io/flagos-ai/FlagGems-sglang)

这一页不包含内部沟通或活动排期，只保留公开、可复现的工程与比赛信息。

## Slide 17：人定义契约，Agent 加速工作

![Slide 17：Humans define the contract; agents speed up the work](https://files.mdnice.com/user/59/0ca7bb00-a500-4a11-b882-bf2969922264.png)

最后一页把整场分享收束为两条链。

第一条是技术链：

```text
stable API → optimized kernel → test evidence → production decision
```

第二条是社区链：

```text
competition submission → SGLang integration → production serving
```

人需要定义 operator contract、测试条件、风险边界和合入标准。Agent 可以更快地搜索设计、编写候选、运行实验和整理失败；SGLang 则用 backend、capability、fallback、trace、CUDA Graph 与模型 E2E 证据决定它能否进入真实路径。

这里最值得保留的不是某一个倍数，而是一套能持续工作的关系：比赛开放真实问题，Agent 扩大搜索和实验规模，开源框架把结果放进长期维护的 API 与验证体系。一个参赛 kernel 跑得更快，是这条路的起点；它能被安全选择、明确回退、重复验证并最终帮助 Serving，才是终点。

## 相关源码与验证记录

- SGLang kernel namespace RFC：https://github.com/sgl-project/sglang/issues/29630
- Unified kernel namespace 系列：https://github.com/sgl-project/sglang/pull/30044 · https://github.com/sgl-project/sglang/pull/31666 · https://github.com/sgl-project/sglang/pull/32072 · https://github.com/sgl-project/sglang/pull/32648 · https://github.com/sgl-project/sglang/pull/33205
- JIT cache：https://github.com/sgl-project/sglang/pull/34274
- Custom AllReduce v2：https://github.com/sgl-project/sglang/pull/31049
- Qwen-Image 多芯片 Serving path：https://github.com/sgl-project/sglang/pull/36680
- QSA SM121：https://github.com/sgl-project/sglang/pull/36845
- Qwen3.x NVFP4 GEMM 与 E2E 数据：https://github.com/sgl-project/sglang/pull/36865
- FLUX.2 KDA fusions：https://github.com/sgl-project/sglang/pull/37162
- KDA backend registration：https://github.com/sgl-project/sglang/pull/37385
- KDA-Pilot integration parity：https://github.com/BBuf/KDA-Pilot/pull/22 · https://github.com/BBuf/KDA-Pilot/pull/24 · https://github.com/BBuf/KDA-Pilot/pull/25
- KDA-Pilot frozen rows、A/A 与 no-go：https://github.com/BBuf/KDA-Pilot/pull/40 · https://github.com/BBuf/KDA-Pilot/pull/41 · https://github.com/BBuf/KDA-Pilot/pull/43 · https://github.com/BBuf/KDA-Pilot/pull/79 · https://github.com/BBuf/KDA-Pilot/pull/89
- KDA-Pilot bitwise contract：https://github.com/BBuf/KDA-Pilot/pull/152 · https://github.com/BBuf/KDA-Pilot/pull/157 · https://github.com/BBuf/KDA-Pilot/pull/158
- KDA-Pilot stateful qualification：https://github.com/BBuf/KDA-Pilot/pull/194
