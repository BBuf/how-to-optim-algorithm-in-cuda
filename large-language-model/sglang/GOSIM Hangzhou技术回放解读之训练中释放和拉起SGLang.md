> Colocate RL 最麻烦的地方不是把 SGLang 拉起来，而是训练和推理共用一批 GPU 时，显存、CUDA Graph 和权重更新都要能按节奏切换。

# 0x0. 前言

这场分享的关键词是 colocate。训练框架想省卡，推理框架想保住高吞吐，二者争的是同一块 GPU 显存。周围很多讨论会停在“训练和推理错峰运行”，但这份 slides 讲到了实现层：SGLang 如何把显存按 tag 暂停，如何保住 CUDA Graph 的虚拟地址，训练后的权重又如何快速灌回推理进程。

# 0x1. 资料和代码落点

这篇主要对应这些公开实现：

- `torch_memory_saver`：`csrc/entrypoint.cpp` 和 `csrc/core.cpp`，实现 hook、VA 保留、pause/resume。
- SGLang：`python/sglang/srt/managers/scheduler_update_weights_mixin.py`，提供 `release_memory_occupation` 和 `resume_memory_occupation` 接口。
- SGLang：`python/sglang/srt/model_executor/model_runner.py`，实现 `update_weights_from_distributed`、`update_weights_from_tensor` 以及 `flattened_bucket`。
- SGLang：`python/sglang/srt/model_executor/cuda_graph_runner.py` 和 `torch_memory_saver_adapter.py`，把 CUDA Graph 捕获也纳入 memory saver tag。
- LMSYS slime blog：`https://lmsys.org/blog/2025-07-09-slime/`，可以作为 colocate rollout、weight sync 和 SGLang-native RL 系统的补充背景。

我在本地 `sglang` 和 `torch_memory_saver` 源码里能对上 slide 里的实现。这里没有强行写 PR 号，因为这组改动在当前代码树里已经变成了功能面；博客里按文件和关键函数讲更稳。

slime 的系统图能帮这篇文章校准视角：release/resume 不是孤立功能，而是 RL 系统里“训练阶段让出显存、rollout 阶段恢复推理”的一个环节。LMSYS blog 里把 SGLang rollout、weight sync、partial rollout、训练 actor 放在同一张图里，和这套 slides 的 colocate 主题刚好对上：

<img src="https://files.mdnice.com/user/59/e62c3bfc-5f68-4989-a912-c8495837396d.png" referrerpolicy="no-referrer" />

图里最关键的是 weight sync 和 rollout server group。训练 step 结束后，actor 侧的新权重不能慢慢通过 CPU 一层层搬过去，否则 rollout 时间会被同步开销吃掉；推理 server 恢复后也不能破坏 CUDA Graph 捕获过的虚拟地址。所以下面看 `torch_memory_saver` 和 SGLang 更新权重代码时，要把它们放回这个闭环里理解：release/resume 解决显存复用，bucket/IPC/broadcast 解决权重刷新，两者缺一块都跑不顺。

# 0x2. Slides 逐页解读

### Slide 1：Inference empowering training in Reinforcement Learning

<img src="https://files.mdnice.com/user/59/56e0d33e-6b3b-46fc-a381-7b9dc764f17d.png" referrerpolicy="no-referrer" />

RL 训练里的 SGLang 不是一个旁路 rollout 服务，而是训练系统里的可调度组件。需要它生成 rollout 时，它要有显存、有 CUDA Graph、有 KV cache；训练 step 要吃满显存时，它又要把自己占着的物理显存交出来。
### Slide 2：内容主线：释放/拉起 SGLang 和参数更新

<img src="https://files.mdnice.com/user/59/6aea82b2-2d02-4839-ac61-ec33ffe7c35f.png" referrerpolicy="no-referrer" />

目录把问题拆成两块：一块是训练中释放和拉起 SGLang，另一块是高效参数更新。前者解决 GPU 共存，后者解决每轮训练后推理权重怎么快速刷新。两件事连在一起，才是 colocate RL 能跑稳的基础。
### Slide 3：RL 训练里推理引擎不是旁路组件

<img src="https://files.mdnice.com/user/59/d2db621f-55db-43ea-a8c4-4164033f9d58.png" referrerpolicy="no-referrer" />

RLHF、GRPO 这类流程里，推理不是离线准备数据。策略模型采样、打分、过滤和训练不断交替，推理端会频繁看到新权重。SGLang 在这里承担 rollout engine，吞吐、首 token 延迟和显存占用都会直接影响训练效率。
### Slide 4：Rollout 和训练的资源错峰

<img src="https://files.mdnice.com/user/59/106158d6-5a88-4c4e-b75d-d45361b12f25.png" referrerpolicy="no-referrer" />

这页想表达的是资源时间片：rollout 阶段训练侧算子不工作，训练阶段 rollout runtime 又在等新权重。如果训练和推理各占一组卡，两边都会在对方阶段留下空转窗口；colocate 的目标就是把这两个时间片叠到同一批 GPU 上，用 release/resume 在阶段切换时让显存和 CUDA Graph 跟着切换。

### Slide 5：训练端显存高峰和推理端显存常驻

<img src="https://files.mdnice.com/user/59/c6fb83cc-9489-40b3-99ed-fc148170faa0.png" referrerpolicy="no-referrer" />

显存账是 colocate 的难点。训练侧需要 optimizer state、gradient、activation 或重计算缓冲；推理侧常驻权重、KV cache、CUDA Graph 池。单独看都合理，放在同一张卡上就会 OOM。
### Slide 6：SGLang 放进训练闭环后的问题

<img src="https://files.mdnice.com/user/59/0f3b9ff1-3cad-47e7-92b0-dd4ab965b928.png" referrerpolicy="no-referrer" />

把 SGLang 塞进训练 loop 后，不能只靠 `torch.cuda.empty_cache()`。SGLang 的权重、KV cache、CUDA Graph 捕获出来的地址都有生命周期，外部训练框架必须能告诉它什么时候睡眠、什么时候恢复。
### Slide 7：Qwen2-7B full-shard 的显存账

<img src="https://files.mdnice.com/user/59/738510ec-0eb2-4f0a-9644-41a5ac72a3ad.png" referrerpolicy="no-referrer" />

Qwen2-7B 的例子用来量化冲突：full-shard 训练的峰值已经接近 48GB，推理端还要留权重、KV 和 graph。这个数不是极限 case，而是中等模型就会遇到的日常情况。
### Slide 8：只有 8 张卡时不能浪费空闲窗口

<img src="https://files.mdnice.com/user/59/6c69e615-7570-4aa2-8edc-7f15ece27bdf.png" referrerpolicy="no-referrer" />

8 卡环境里，如果把 4 卡给训练、4 卡给推理，总有一边在等。Colocate 的收益来自把 idle 时间变成可用显存，而不是魔法般减少模型本身的需求。
### Slide 9：Colocate 的基本形态

<img src="https://files.mdnice.com/user/59/ccf89de3-30f1-4acb-b284-8c9a2e5a4e61.png" referrerpolicy="no-referrer" />

Colocate 后的编排是：rollout 时 SGLang online，训练前 SGLang release memory，训练完更新推理权重，随后 resume 继续 rollout。这里最危险的是 resume 后 CUDA Graph 地址不一致，后面源码会看到它为什么要保留 virtual address。
### Slide 10：LD_PRELOAD 接管 cudaMalloc/cudaFree

<img src="https://files.mdnice.com/user/59/4ef734e3-6566-469b-a1f7-b6feb6f277fb.png" referrerpolicy="no-referrer" />

这页对应 `torch_memory_saver`。它通过 LD_PRELOAD hook `cudaMalloc/cudaFree`，在“interesting region”里不用普通 cudaMalloc，而是走 CUDA Driver API 的 `cuMemAddressReserve/cuMemMap`。地址预留和物理分配分开，是后续 pause/resume 的根。
### Slide 11：CUDA Graph 要求虚拟地址稳定

<img src="https://files.mdnice.com/user/59/8e0148c7-4118-41e6-9441-53b7719b0935.png" referrerpolicy="no-referrer" />

CUDA Graph 捕获时会记住指针地址。释放后如果重新 malloc 到另一个地址，graph replay 可能直接读写旧地址。因此 memory saver 不能释放 virtual address，只能 unmap physical memory，resume 时再 map 回同一个 VA。
### Slide 12：按 weights、KV cache、CUDA graph 打标签

<img src="https://files.mdnice.com/user/59/14877c90-2134-47f6-8cc3-1c3659103a13.png" referrerpolicy="no-referrer" />

SGLang 里把显存分成 `weights`、`kv_cache`、`cuda_graph` 三类 tag。tag 不是装饰，它决定 release/resume 的顺序：KV 可以 flush 后暂停，weights 需要保存静态 buffer 状态，CUDA graph 要单独处理。
### Slide 13：release：只释放物理显存

<img src="https://files.mdnice.com/user/59/cf45d0d2-a2f4-494d-9654-dc800b667ef3.png" referrerpolicy="no-referrer" />

release 做的是“把物理显存还回去”。源码里 `pause()` 会先按 tag 找到 allocation；如果启用 CPU backup，会把内容拷到 pinned host memory；然后 `cuMemUnmap` 加 `cuMemRelease`。注意没有 `cuMemAddressFree`。
### Slide 14：resume：按原虚拟地址重新映射

<img src="https://files.mdnice.com/user/59/87870c0c-738f-4085-af13-cfd0e31bf931.png" referrerpolicy="no-referrer" />

resume 的动作正好相反：重新 `cu_mem_create` 物理内存，再 `cuMemMap((CUdeviceptr) ptr, ...)` 映射到原始指针。由于 `ptr` 没变，CUDA Graph 和模型里保存的 tensor storage 地址仍然有效。
### Slide 15：SGLang Scheduler 里的 pause/resume 边界

<img src="https://files.mdnice.com/user/59/d81dcad7-19db-4358-af8e-f61df2c05fc0.png" referrerpolicy="no-referrer" />

SGLang Scheduler 在 release 前要求 server idle，这个限制很朴素但必要：如果还有请求在跑，暂停 KV 或 weights 就会把正在执行的 batch 切断。它还会在暂停 weights 前导出 model buffers，恢复后再导入，避免非 parameter 状态丢失。
### Slide 16：在线参数更新的通信链路

<img src="https://files.mdnice.com/user/59/8318027f-d413-485f-a33d-8fbbb26329b0.png" referrerpolicy="no-referrer" />

参数更新链路分两种：一种是 distributed broadcast，一种是 tensor/IPC 方式。RL 框架通常持有训练权重，SGLang 只需要把对应张量接过来并 `model.load_weights`。瓶颈不在一次大拷贝，而在成千上万个小 tensor 的调度开销。
### Slide 17：CUDA IPC 更新权重

<img src="https://files.mdnice.com/user/59/280932a7-7549-4831-b1f1-f8a6705189fe.png" referrerpolicy="no-referrer" />

CUDA IPC 的思路是把训练侧 GPU tensor 序列化成可跨进程重建的 handle。SGLang 进程收到 handle 后直接在 GPU 上读，不走 CPU 中转。这也是为什么 colocate mapping 不能随便乱，IPC handle 和设备拓扑必须对得上。
### Slide 18：flattened bucket 降低 MoE 参数更新开销

<img src="https://files.mdnice.com/user/59/dc40c46e-19a8-425a-96d6-c08f813688a9.png" referrerpolicy="no-referrer" />

MoE 模型最容易暴露小 tensor 问题。Qwen3-30B-A3B 这类模型 expert 参数很多，逐 tensor API 调用会变成约 2000 次更新；flattened bucket 把多个 tensor flatten 成一个大 buffer，调用次数降到百级，端到端时间也明显下降。
### Slide 19：实现上的同步点和失败边界

<img src="https://files.mdnice.com/user/59/39dd0b52-d77b-45ee-94ff-d7d3dba3707a.png" referrerpolicy="no-referrer" />

这页其实是在提醒失败边界：更新权重如果中途异常，模型可能处于“部分新、部分旧”的状态，所以 SGLang 的错误信息会明确建议丢弃整份 weights。分布式更新还要有 barrier，否则某个 TP rank 提前进入下一轮也会出事。
### Slide 20：先恢复 weights 再更新，再恢复 KV/CUDA Graph

<img src="https://files.mdnice.com/user/59/576b69fc-ae89-40de-80e0-c1d6656763bf.png" referrerpolicy="no-referrer" />

顺序很重要：先恢复 weights，让新参数有落点；更新权重；最后恢复 KV cache 和 CUDA Graph。反过来做会让 graph 或 cache 看到半更新状态，debug 起来非常难。
### Slide 21：总结：推理引擎成为训练系统的一部分

<img src="https://files.mdnice.com/user/59/f74d2f51-530e-4540-b920-081519d4f82e.png" referrerpolicy="no-referrer" />

这场分享最值得记住的是抽象层级：SGLang 不再只是一个服务进程，而是训练系统能控制的 GPU resident runtime。显存、graph、权重同步都暴露成协议后，RL 系统才能做真正的训推一体。

# 0x3. 关键代码拆解

第一段先看 hook。`torch_memory_saver/csrc/entrypoint.cpp` 里，只有线程处于 interesting region 时才接管分配：

```cpp
#ifdef TMS_HOOK_MODE_PRELOAD
cudaError_t cudaMalloc(void **ptr, size_t size) {
    if (thread_local_config.is_interesting_region()) {
        return TorchMemorySaver::instance().malloc(
            ptr, CUDAUtils::cu_ctx_get_device(), size,
            thread_local_config.current_tag_,
            thread_local_config.enable_cpu_backup());
    } else {
        return APIForwarder::call_real_cuda_malloc(ptr, size);
    }
}

cudaError_t cudaFree(void *ptr) {
    return TorchMemorySaver::instance().free(ptr);
}
#endif
```

`current_tag_` 是后面按 weights/KV/CUDA Graph 分组释放的依据。这个设计很克制：不在 region 里的普通 CUDA 分配还是交给真实 `cudaMalloc`，避免把全进程 allocator 行为都改掉。

真正的关键在 `core.cpp`。分配时先 reserve 地址，再 map 物理内存：

```cpp
CURESULT_CHECK(cuMemAddressReserve((CUdeviceptr *) ptr, size, 0, 0, 0));
CURESULT_CHECK(cuMemMap((CUdeviceptr) * ptr, size, 0, allocHandle, 0));
CUDAUtils::cu_mem_set_access(*ptr, size, device);

allocation_metadata_.emplace(
    *ptr,
    AllocationMetadata{size, device, tag,
        AllocationState::ACTIVE, enable_cpu_backup, nullptr, allocHandle}
);
```

pause 只 unmap/release，不 free address：

```cpp
if (metadata.enable_cpu_backup) {
    cudaMallocHost(&metadata.cpu_backup, metadata.size);
    cudaMemcpy(metadata.cpu_backup, ptr, metadata.size, cudaMemcpyDeviceToHost);
}

CURESULT_CHECK(cuMemUnmap((CUdeviceptr) ptr, metadata.size));
CURESULT_CHECK(cuMemRelease(metadata.allocHandle));
metadata.state = AllocationState::PAUSED;
```

resume 则在同一个 `ptr` 上重新 map：

```cpp
CUmemGenericAllocationHandle newAllocHandle;
CUDA_ERROR_CHECK(CUDAUtils::cu_mem_create(&newAllocHandle, metadata.size, metadata.device));

CURESULT_CHECK(cuMemMap((CUdeviceptr) ptr, metadata.size, 0, newAllocHandle, 0));
CUDAUtils::cu_mem_set_access(ptr, metadata.size, metadata.device);

if (metadata.enable_cpu_backup) {
    cudaMemcpy(ptr, metadata.cpu_backup, metadata.size, cudaMemcpyHostToDevice);
    cudaFreeHost(metadata.cpu_backup);
    metadata.cpu_backup = nullptr;
}
metadata.state = AllocationState::ACTIVE;
metadata.allocHandle = newAllocHandle;
```

这就解释了 slide 里为什么一直强调 virtual address：真正被释放的是 physical allocation handle，不是 pointer。

SGLang 侧把这个能力包装成服务接口。`release_memory_occupation` 里能看到三个 tag 的处理顺序：

```python
if GPU_MEMORY_TYPE_KV_CACHE in tags:
    self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_KV_CACHE)
    self.flush_cache()

if GPU_MEMORY_TYPE_WEIGHTS in tags:
    self.stashed_model_static_state = _export_static_state(
        self.tp_worker.model_runner.model
    )
    torch.distributed.barrier(self.tp_cpu_group)
    self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_WEIGHTS)

if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
    self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_CUDA_GRAPH)
```

恢复时 CUDA Graph、weights、KV cache 依次回来：

```python
if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
    self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_CUDA_GRAPH)

if GPU_MEMORY_TYPE_WEIGHTS in tags:
    self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_WEIGHTS)
    torch.distributed.barrier(self.tp_cpu_group)
    _import_static_state(
        self.tp_worker.model_runner.model,
        self.stashed_model_static_state,
    )

if GPU_MEMORY_TYPE_KV_CACHE in tags:
    self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_KV_CACHE)
```

最后看参数更新。普通路径每个 tensor 一次 broadcast，bucket 路径先构造连续 buffer：

```python
if load_format == "flattened_bucket":
    return self._update_bucketed_weights_from_distributed(
        names, dtypes, shapes, group_name
    )

weights = []
handles = []
for name, dtype, shape in zip(names, dtypes, shapes):
    weight = torch.empty(shape, dtype=target_dtype, device=self.device)
    handles.append(torch.distributed.broadcast(
        weight, src=0, group=self._model_update_group[group_name], async_op=True
    ))
    weights.append((name, weight))
for handle in handles:
    handle.wait()
self.model.load_weights(weights)
```

```python
named_tensors = []
for name, dtype, shape in zip(names, dtypes, shapes):
    named_tensors.append((name, torch.empty(shape, dtype=target_dtype, device=self.device)))
bucket = FlattenedTensorBucket(named_tensors=named_tensors)
flattened_tensor = bucket.get_flattened_tensor()
torch.distributed.broadcast(flattened_tensor, src=0, group=self._model_update_group[group_name])
reconstructed_tensors = bucket.reconstruct_tensors()
self.model.load_weights(reconstructed_tensors)
```

MoE 场景里，bucket 优化不是为了减少总字节数，而是为了减少 Python/HTTP/collective 调度次数。这正是 slide 里从约 2000 次 API call 降到约 120 次的原因。

# 0x4. 小结

这一篇更像“训练系统如何控制推理 runtime”的案例：显存释放不是清 cache，而是 VA/physical memory 分离；权重更新不是简单 reload，而是跨进程、跨 TP rank 的在线同步。理解这两层，才能看懂为什么 SGLang 能在 RL colocate 里跑得起来。
