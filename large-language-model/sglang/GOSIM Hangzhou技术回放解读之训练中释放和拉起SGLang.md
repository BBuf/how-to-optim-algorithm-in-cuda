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

8 卡环境里，如果把 4 卡给训练、4 卡给推理，总有一边在等。Colocate 的收益来自把 idle 时间变成可用显存，并不是降低模型本身的显存需求，而是把空闲阶段的显存重新交给正在工作的那一侧。
### Slide 9：Colocate 的基本形态

<img src="https://files.mdnice.com/user/59/ccf89de3-30f1-4acb-b284-8c9a2e5a4e61.png" referrerpolicy="no-referrer" />

Colocate 后的编排是：rollout 时 SGLang online，训练前 SGLang release memory，训练完更新推理权重，随后 resume 继续 rollout。这里最危险的是 resume 后 CUDA Graph 地址不一致，后面源码会看到它为什么要保留 virtual address。
### Slide 10：torch-memory-saver 的四个 Driver API

<img src="https://files.mdnice.com/user/59/4ef734e3-6566-469b-a1f7-b6feb6f277fb.png" referrerpolicy="no-referrer" />

这页把 `torch-memory-saver` 的接口边界讲得很直接。它先用 `LD_PRELOAD` 劫持 `cudaMalloc/cudaFree`，但只在被 `memory_saver.region()` 包起来的区域里接管分配；region 外的普通 PyTorch/SGLang 分配仍然走原来的 allocator。这样做的好处是改动面比较小，只有权重、KV cache、CUDA Graph 这些需要 pause/resume 的 tensor 会进入 TMS 管理。

下面的小字列了四个 CUDA Driver API。`cuMemCreate` 创建 physical memory handle；`cuMemAddressReserve` 预留一段虚拟地址；`cuMemMap` 把 physical handle 映射到这段虚拟地址；`cuMemSetAccess` 决定哪些 device 可以访问这段映射。这里的核心不是“换一种 malloc”，而是把“指针地址”和“真实显存”拆开管理。推理阶段结束后可以释放 physical memory，训练阶段结束后再把 weights 和 KV cache 映射回同一段 virtual memory address。CUDA Graph 复用依赖这个地址不变。
### Slide 11：TMS 的使用方式

<img src="https://files.mdnice.com/user/59/8e0148c7-4118-41e6-9441-53b7719b0935.png" referrerpolicy="no-referrer" />

这页的代码片段说明 TMS 的用法：先 `import torch_memory_saver`，拿到 `torch_memory_saver.torch_memory_saver`，然后在 `with memory_saver.region():` 里面创建需要托管的 tensor。slide 里举的是一个 `torch.full((1_000_000_000,), 100, dtype=torch.uint8, device='cuda')`，大约 1GB。这个 tensor 创建出来以后，`memory_saver.pause()` 会释放它背后的 CUDA physical memory，`memory_saver.resume()` 会在同一个 virtual address 上重新分配并恢复映射。

这和 CUDA Graph 的关系很紧。CUDA Graph 捕获时会把 kernel 参数里的 pointer 也录进去。普通 `cudaFree` 后再 `cudaMalloc`，新 tensor 不保证拿到旧地址，graph replay 就可能访问已经失效的地址。TMS 的使用方式等于给这些 tensor 建了一个“地址壳”：壳不变，里面的 physical backing 可以按阶段拆掉再装回去。
### Slide 12：TMS 如何工作

<img src="https://files.mdnice.com/user/59/14877c90-2134-47f6-8cc3-1c3659103a13.png" referrerpolicy="no-referrer" />

这页就是用户截图里那张图，里面三列要一起看。中间的 `Virtual Memory` 是 PyTorch/SGLang 看到的地址空间，图上从 `0x000` 到 `0xfff` 只是示意，绿色 `occupied` 表示已经预留并映射的一段虚拟地址。右边的 `Physical Memory` 是真正占 GPU 显存的部分。左边的 `Metadata Map` 保存 `ptr -> metadata`，`ptr` 指向那段 virtual address，`metadata` 里有 size、device、tag、状态、CPU backup 指针和 physical allocation handle。

图上的 1 到 4 对应分配流程。第一步创建 `CUmemGenericAllocationHandle`，用 `cuMemCreate` 拿到一块 physical memory 的 handle；这个 handle 里会带上 memory location、是否可共享等属性。第二步用 `cuMemAddressReserve` 在虚拟地址空间里占一个连续范围。第三步用 `cuMemMap` 把 physical handle 绑定到这个虚拟地址范围。第四步把 virtual pointer 和 physical handle 记进 `Metadata Map`。后面 pause/resume 能找到某一块 weights 或 KV cache，就是靠这份 metadata。

SGLang 在 TMS 外面又加了一层 tag，常见的是 `weights`、`kv_cache`、`cuda_graph`。tag 的意义是让 scheduler 能分组控制这些 allocation：KV cache 可以先 flush 再 pause；weights 要先保存 model buffer 的静态状态；CUDA Graph 相关内存要保持 virtual address 可复用。
### Slide 13：release：只释放物理显存

<img src="https://files.mdnice.com/user/59/cf45d0d2-a2f4-494d-9654-dc800b667ef3.png" referrerpolicy="no-referrer" />

这页画的是 pause。三列还是 `Metadata Map / Virtual Memory / Physical Memory`，但右侧 physical memory 已经变成 `available`。流程只有两步：先用 `cuMemUnmap` 断开 virtual address 到 physical allocation 的映射，再从 metadata 里取出 `allocHandle`，调用 `cuMemRelease(metadata.allocHandle)` 释放真实显存。

注意中间那列 virtual memory 仍然标成 `occupied`。这说明 TMS 没有调用 `cuMemAddressFree`，地址范围还被保留着。也就是说，tensor 的指针值还在，CUDA Graph 捕获过的 pointer 也还在，只是这段地址暂时没有 physical backing。这个状态下不能让 kernel 去访问它，所以 SGLang 的 `release_memory_occupation` 前面会 assert server 必须 fully idle。若 allocation 开了 CPU backup，`pause()` 还会先把内容拷到 pinned host memory，resume 时再拷回 GPU。
### Slide 14：resume：按原虚拟地址重新映射

<img src="https://files.mdnice.com/user/59/87870c0c-738f-4085-af13-cfd0e31bf931.png" referrerpolicy="no-referrer" />

resume 图里右侧 physical memory 又出现了绿色 `occupied`，中间 virtual memory 的绿色块位置没有变。它的顺序是：创建新的 physical allocation handle，重新分配 physical memory，把新 physical memory 用 `cuMemMap` 映射到 metadata 里保存的旧 virtual address，最后更新 metadata 中的 handle。

这一步最关键的是 `ptr` 不变。PyTorch parameter 的 storage 地址、KV cache tensor 的地址、CUDA Graph 捕获时记录的地址都继续指向同一个 virtual address。变的只是背后的 physical handle。CPU backup 打开时，resume 还会把 host backup 拷回这段地址，再把 allocation state 标回 active。这样训练阶段释放出来的显存可以重新还给 SGLang，而 graph 不需要重新捕获。
### Slide 15：SGLang Scheduler 里的 pause/resume 边界

<img src="https://files.mdnice.com/user/59/d81dcad7-19db-4358-af8e-f61df2c05fc0.png" referrerpolicy="no-referrer" />

这页的系统图从 data buffer 开始：上方的 buffer 把 init prompt 发给 custom rollout generation，拿回自定义生成数据；左下角 megatron 负责训练，右侧 sglang router 后面挂多个 sglang server。红框圈出来的 `update weights from distributed/tensor` 就是本篇后半部分要讲的更新路径。箭头含义也很明确：训练数据从 buffer 下发给 megatron；rollout 数据从 SGLang 侧回流；train end 之后，megatron 侧的新权重要同步到 SGLang server。

SGLang Scheduler 在 release 前要求 server idle，这个限制来自上面几页的内存语义。如果还有请求在跑，暂停 KV 或 weights 会把正在执行的 batch 切断。源码里 release 会先记录 offload tag，KV cache 路径先 `pause` 再 `flush_cache`；weights 路径会导出 model buffers，做一次 TP CPU group barrier，再 pause weights；CUDA graph 也按 tag 单独 pause。恢复时再按 `cuda_graph -> weights -> kv_cache` 的顺序 resume。
### Slide 16：在线参数更新的通信链路

<img src="https://files.mdnice.com/user/59/8318027f-d413-485f-a33d-8fbbb26329b0.png" referrerpolicy="no-referrer" />

这页的图把 train end 后的显存状态画成两边。左边是训练阶段：GPU memory 里有 SGLang 的外壳，但主要空间被 `Training Part` 占住，`Model Weights` 是训练侧正在更新的参数。右边是恢复后的 SGLang GPU memory：上方是 KV cache，中间有 others，下面是 model weights，最底下红色 buffer 是参数同步时临时接收新权重的区域。底部的 CPU memory 里也有一份 model weights，蓝色箭头和红色箭头表示权重从训练侧/CPU 侧同步到 SGLang 的不同路径。

参数更新链路分两类：distributed broadcast 和 tensor/IPC。RL 框架通常持有训练权重，SGLang 端收到对应张量后走 `model.load_weights` 原地更新。图里的 buffer 对应参数同步时的临时落点，真正的瓶颈常常不是总字节数，而是大量小 tensor 带来的调度、通信和 load 调用开销。所以后面 slide 才会引出 CUDA IPC 和 flattened bucket。
### Slide 17：CUDA IPC 更新权重

<img src="https://files.mdnice.com/user/59/280932a7-7549-4831-b1f1-f8a6705189fe.png" referrerpolicy="no-referrer" />

这页标题写成 “A tensor's journey”，左侧五步就是一个参数张量从训练进程走到 SGLang 进程的路径：先在训练侧跨 rank gather tensor，再把 gather 后的 GPU tensor 序列化成 CUDA IPC handle，把 handle 发给 SGLang，SGLang 反序列化回 tensor，最后更新本地 weights。注意这里传的是 handle，不是把整块参数经 HTTP 复制成 Python bytes。

右侧图把 colocate 模式下的拓扑画出来了。顶部小字说明 Megatron 和 SGLang 进程共享同一批 GPU memory。左边 Megatron 里有 Worker 0 到 Worker 3，对应 PP0/PP1/PP2/PP3；中间 `all gather` 后，每个 rank 得到一个 `Gathered Tensor IPC Handler`，再汇成 `List of Cuda IPC Handles`；右边是两个 SGLang server，每个 server 下面挂 TP Worker 0/1，对应 TP0/TP1。这个图的重点是并行形态变了：训练侧可能按 pipeline/data/tensor parallel 切参数，推理侧按 TP worker 接收参数，所以更新不是简单地“把一个 state_dict 发过去”，而是要按 rank 和 TP worker 的对应关系重建 tensor。
### Slide 18：flattened bucket 降低 MoE 参数更新开销

<img src="https://files.mdnice.com/user/59/dc40c46e-19a8-425a-96d6-c08f813688a9.png" referrerpolicy="no-referrer" />

这页标题是 “Bucket Update by minimize http overhead”。上面的代码是低效版本：`for name, tensor in named_tensors.items()`，每个 tensor 都发一次 `requests.post(".../update_weights_from_tensor", json={"tensor_name": name, "tensor_data": serialize(tensor)})`。如果 MoE 模型有大量 expert 参数，这个循环会把一次权重更新拆成很多 HTTP 请求和很多次 handler 打开/关闭。

下面的代码是 bucket 版本。`get_param_info_buckets(args, model)` 先收集 `param_infos`，再按 `args.update_weight_buffer_size` 控制 bucket 大小；如果参数名里包含 `.experts.`，就用 expert tensor parallel world size 估算参数大小，否则用普通 tensor model parallel world size。循环里一旦 `buffer_size + param_size` 超过阈值，就新开一个 bucket。最后不再逐 tensor 调 `_update_weights_from_tensor`，而是对每个 `param_infos` bucket 调 `_update_bucket_weights_from_tensor`。右侧的 Qwen3-30B-A3B 例子给了直观数字：约 2000 次单 tensor API 调用耗时 50s，合 bucket 后约 120 次调用耗时 30s。
### Slide 19：flatten tensor 后的耗时拆分

<img src="https://files.mdnice.com/user/59/39dd0b52-d77b-45ee-94ff-d7d3dba3707a.png" referrerpolicy="no-referrer" />

这页继续拆 flattened tensor 的收益。左上角 `Before` 表格把 41ms 更新拆成三段：IPC Handler Open 22ms，占 54%；Load Weights 8ms，占 19%；IPC Handler Close 11ms，占 27%。右边 flamegraph 也能看到 `update_weights_from_tensor` 下面有很长的 `ipc handler open` 和 `ipc handler close` 区间，说明时间不是主要花在拷权重本身，而是花在反复创建、映射、关闭 IPC handler。

左下角 `After` 表格是 flatten 后的结果：IPC Handler Open 降到 3ms，Close 降到 200us，总耗时从 41ms 变成 20ms，标了 “51% improvement vs 41ms without flattening”。中间多出的 `Rebuild` 5ms 是把 flatten buffer 重新切回各个参数 tensor 的成本；Load Weights 变成 12ms，占比更高，但这是因为 handler 开关被压下去了，真正的数据加载开始成为主要部分。这一页和前一页连起来看，就是先用 bucket 减少 HTTP/API 次数，再用 flatten 减少 IPC handler 次数。
### Slide 20：先恢复 weights 再更新，再恢复 KV/CUDA Graph

<img src="https://files.mdnice.com/user/59/576b69fc-ae89-40de-80e0-c1d6656763bf.png" referrerpolicy="no-referrer" />

Multistage update 这页把前面的内存和参数更新串成一个时序。左图是训练阶段结束前：同一块 GPU memory 里外层仍然是 SGLang 的管理边界，但主要空间被 `Training Part` 占住，训练侧 `Model Weights` 还在蓝色区域里。`train end` 之后进入中间状态：SGLang 先只 resume `Model Weights`，底部留出红色 `Bucket` 作为参数更新缓冲，红色箭头表示 bucket 数据写入模型权重，蓝色箭头表示训练侧/CPU 侧权重进入 bucket。

右图是 `update end` 后的完整推理状态：GPU memory 里恢复出 `Model Weights`、`Others` 和 `KV Cache`。右侧列表写的三个步骤正好对应这三张图：1. resume weights；2. update weights；3. resume KV/CUDA Graph。顺序反过来会有两个问题：如果先 resume KV/CUDA Graph，请求可能看到半更新权重；如果不先 resume weights，参数更新没有稳定的 GPU 落点。代码里的 tag 分组和 barrier 也是围绕这个顺序设计的。
### Slide 21：总结：推理引擎成为训练系统的一部分

<img src="https://files.mdnice.com/user/59/f74d2f51-530e-4540-b920-081519d4f82e.png" referrerpolicy="no-referrer" />

这场分享最后落到一个抽象层级：SGLang 不再只是一个服务进程，而是训练系统能控制的 GPU resident runtime。显存、graph、权重同步都暴露成协议后，RL 系统才能做真正的训推一体。

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
