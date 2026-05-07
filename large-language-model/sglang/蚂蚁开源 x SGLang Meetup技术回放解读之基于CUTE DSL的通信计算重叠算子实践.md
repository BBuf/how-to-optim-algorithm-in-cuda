> 这篇是对 2026 年 1 月 17 日蚂蚁开源 x SGLang Meetup 中「SGLang 社区基于 CuTe DSL 的通信-计算重叠内核的实践经验」这个分享的回放解读。原始 slides 讲的是一个很硬核的 kernel 优化：把 Tensor Parallel 里的 GEMM 和 AllReduce 融到一个 CuTe DSL kernel 里，用 Blackwell 上的 multimem/NVLS 指令把部分通信藏到 GEMM epilogue 后面。这里我会按 slides 顺序拆，但重点放在公开代码：SGLang PR [#15103 Support TP overlap on Blackwell](https://github.com/sgl-project/sglang/pull/15103)、FlashInfer PR [#1695 cute_dsl gemm + all reduce two_shot](https://github.com/flashinfer-ai/flashinfer/pull/1695)，以及 CUTLASS 官方 CuTe DSL distributed examples。

# 0x0. 前言

![](https://files.mdnice.com/user/59/8fe0d2c2-6ba6-4d94-81ed-02f17ef4200f.jpg)

这个分享的题目是「基于 CUTE DSL 的通信计算重叠算子实践 GEMM + AR 经验分享」。嘉宾是 NVIDIA 的周沛源，slides 里写的时间是 2026/01/17。

我读这套 slides 的感受是，它不是在讲「怎么调用一个更快的 AllReduce」，而是在讲一个更底层的问题：如果 Tensor Parallel 里的 `RowParallelLinear` 每层都要做一次 GEMM 后的 AllReduce，那么这个 AllReduce 能不能不要作为一个独立阶段裸露在 timeline 上？更具体一点，能不能让 GEMM 写回结果之后，马上由同一个 persistent kernel 里的通信 warp 去发起 multimem reduce/broadcast，让通信和后续 tile 的计算重叠起来？

SGLang 这条线目前最直接的公开实现是 PR [#15103](https://github.com/sgl-project/sglang/pull/15103)。PR 描述里写得很直白：

```text
We do GEMM+AllReduce overlap during TensorParallel via cutlass based fused kernel on B200.
```

它把 fused GEMM+AllReduce 接到了模型的两个位置：

- Attention 后面的 `o_proj`，也就是 TP 下需要 all-reduce 的输出投影；
- MLP 后面的 `down_proj`，同样是 row parallel 后需要聚合的那一层。

分享嘉宾的背景也解释了为什么这套内容很偏底层。slides 后面出现的术语基本都在 CUTLASS/CuTe DSL、NVSHMEM、NVLS、多播地址、TMA store、persistent scheduler 这些东西附近打转。它跟上一篇 DeepSeek 部署优化不太一样，那篇更多是 SGLang server 层和模型执行路径的组合拳；这篇几乎是顺着一条 kernel 的 pipeline 往里钻。

![](https://files.mdnice.com/user/59/3859b993-496f-488f-9cff-e9aaca103df4.jpg)

目录只有三块：

1. 不同通信计算重叠方案的比较；
2. Blackwell 上用到的 multicast / multimem 指令；
3. GEMM + AllReduce 融合算子的实现。

下面我也按这个顺序写。为了方便对代码，我会把「AllReduce」简写成 AR。

# 0x1. 为什么不是简单开两个 kernel

![](https://files.mdnice.com/user/59/0339b3f0-ab61-47dc-a373-5ba40cec5850.png)

这页先把通信计算重叠方案分成两类：Inter-SM 和 Intra-SM。

Inter-SM 的思路是让不同 SM 干不同活。比如一部分 SM 跑 GEMM，一部分 SM 跑通信。slides 里列了三个常见做法：

- 两个 kernel 加 green context；
- 两个 kernel 加 priority stream；
- 一个 kernel 里按 CTA 划分，有些 CTA 做计算，有些 CTA 做通信。

这类方案的麻烦在于 GEMM 很吃 Tensor Core。如果你把一部分 SM 固定拿去做通信，Tensor Core 利用率会掉；如果用两个 kernel，调度顺序又很难刚好卡在你想要的 overlap 位置。尤其是计算 kernel 先把所有 SM 占满时，通信 kernel 只能排队等，timeline 上看起来像并发，实际可能没重叠多少。

Intra-SM 的思路是同一个 SM 里再切角色。比如一个 kernel 内按 warp 划分，MMA warp 做 GEMM，另一些 warp 做通信。它的代价也明显：通信 warp 会吃寄存器、shared memory、issue slot 和调度资源，写不好就会反过来拖慢 GEMM。

所以这页的重点不是说哪一种永远最好，而是引出一个判断：GEMM+AR 这种场景，如果要做得细，最终很可能要落到 warp specialization。也就是一个 persistent kernel 内部，把 TMA、MMA、Epilogue、AllReduce 都当成不同 warp group 的工作。

SGLang 社区里能看到两条公开探索路线。

第一条是 PR [#9058 Support TP overlap](https://github.com/sgl-project/sglang/pull/9058)，基于 ByteDance Seed 的 Triton-Distributed，在 H20 上做 GEMM+AR overlap。它接入方式大概是：

```python
from triton_dist.layers.nvidia import GemmARLayer

gemm_ar_attn_op = GemmARLayer(
    group=ps._TP,
    max_M=server_args.max_running_requests * model_config.context_len,
    N=model_config.hidden_size,
    K=model_config.hidden_size // server_args.tp_size,
    dtype=torch.bfloat16,
    NUM_COMM_SMS=2,
    persistent=True,
    copy_to_local=False,
)
```

这条路线的 PR 数据显示，在 8xH20 跑 Qwen2.5-72B 时，`bench_serving` 的 input throughput 从 `6466.94 token/s` 提到 `6959.78 token/s`，median TTFT 从 `6302.95 ms` 降到 `5838.61 ms`。不过这个 PR 后面也卡在 Triton-Distributed 安装、显存分配和主线集成问题上。

第二条就是这次 slides 主要对应的路线：SGLang PR [#15103](https://github.com/sgl-project/sglang/pull/15103)，基于 CUTLASS/CuTe DSL，在 B200/Blackwell 上做 fused GEMM+AR。这条路线更贴近 slides 后面讲的 multimem 指令和 CuTe DSL kernel。

# 0x2. 粒度：tile 还是 chunk

![](https://files.mdnice.com/user/59/38491e21-33c2-4d08-a5a8-0ca26025c074.png)

通信计算重叠还有一个很关键的问题：重叠粒度到底多大？

slides 把它分成 tile grain 和 chunk grain。

Tile grain 是更细的做法。通常一个 GEMM tile 算完，就把这个 tile 的结果送去通信。它天然适合 GEMM、GroupGEMM 这种二维 tile 结构，因为 CTA tile 本来就是 GEMM kernel 的基本调度单位。坏处是通信逻辑必须塞进 kernel 里，kernel 复杂度会高不少。

Chunk grain 是更粗的做法。先算一段比较大的 chunk，再用另一个 kernel 或 Copy Engine 去做通信。All-Gather+GEMM、Ring Attention 这类工作负载比较常见。好处是工程上相对干净，坏处是粒度粗，通信暴露出来的尾巴更长。

GEMM+AR 更适合 tile grain。原因也简单：RowParallelLinear 的输出矩阵本来就是按 tile 产生的，一个 tile 算完之后，这块数据已经在 register/shared/global 的路径上了。如果等整个 GEMM 都结束再开 AR kernel，就会错过最自然的 overlap 窗口。

# 0x3. 为什么 Blackwell 上更适合做 GEMM+AR fused kernel

![](https://files.mdnice.com/user/59/61649a51-30f2-49c4-9b17-ce28bbf77f2d.png)

这页是前面两页的结论：GEMM+AR 融合算子很适合用 tile-grain 的 fused kernel，尤其是 SM100/Blackwell。

这里有几个硬件和 kernel 结构上的理由。

第一，GEMM 是 Tensor Core 重负载。如果为了通信单独留 SM，计算吞吐会掉，尤其在大模型 MLP/Attention projection 这种矩阵乘里很明显。

第二，GEMM 的 CTA tile 本来就是一个很好的通信单位。每个 tile 的 epilogue 会把 accumulator 写出去，AllReduce 的输入正好就是这些输出。

第三，如果拆成两个 kernel，GEMM kernel 很容易先把所有 SM 都占住，通信 kernel 排在后面等调度。用 priority stream 可以改善一些情况，但不稳定，也不太像一个 kernel 级的确定性方案。

第四，Blackwell 引入 TMEM 后，MMA accumulator 不再以过去那种方式压在普通 register 上，寄存器压力会缓一些。这样同一个 CTA 里多放一个 AR warp group 的空间更大。SGLang PR [#15103](https://github.com/sgl-project/sglang/pull/15103) 的 `PersistentDenseGemmKernel` 就是按这个思路写的，docstring 里写了这几个关键词：

```python
class PersistentDenseGemmKernel:
    """
    This class implements batched matrix multiplication (C = A x B) with support
    for various data types and optimized for NVIDIA Blackwell architecture.

    It supports persistent tile scheduling, warp specialization, and optional
    all-reduce operations for distributed computation.
    """
```

更关键的是它在 warp 角色上多切了一组 all-reduce warp：

```python
if self.all_reduce != "none":
    self.epilog_warp_id = (0, 1, 2, 3)
    self.mma_warp_id = 4
    self.tma_warp_id = 5
    self.all_reduce_warp_id = (6, 7, 8, 9)
    self.threads_per_cta = 32 * (
        len(self.epilog_warp_id)
        + 1
        + 1
        + len(self.all_reduce_warp_id)
    )
```

这段代码基本就是 slide 里的 Intra-SM 方案落地：TMA warp 搬数据，MMA warp 驱动 Tensor Core，epilogue warp 做输出，AR warp group 专门等 tile ready 后做 multimem reduce 和 broadcast。

# 0x4. Multimem 地址和三条指令

![](https://files.mdnice.com/user/59/17fb9334-b37a-4a25-9ffa-1fc575e52004.png)

从这页开始进入 multimem。这里我先把概念说白一点。

普通 UC address 指向一个 GPU 上的一块物理内存。Multicast address，也就是 slides 里写的 MC address，是一个特殊虚拟地址，它能映射到多个 rank 上的对应物理内存。你不能拿普通 load/store 随便访问它，只能用 `multimem.*` 这类指令。

这套机制背后靠 NVSwitch/NVLS 做事。对 AR 来说，最重要的是三条：

- `multimem.ld_reduce`：从多个 rank 的同一 MC 地址读数据，在 NVSwitch 里做 reduce，把结果返回给发起指令的 rank；
- `multimem.st`：把数据发到 NVSwitch，再 multicast 到多个 rank；
- `multimem.red`：对多个 rank 上的目标地址做原子 reduce，常用来做跨 rank flag/barrier。

CUTLASS 官方 distributed README 里也把这几个东西放在同一个上下文里讲：CuTe DSL 的 distributed examples 使用 NVSHMEM4Py 创建 symmetric tensor 和 multicast tensor，然后用 multimem 指令走 NVLS/NVLink SHARP，把 reduce 和 broadcast 尽量交给交换芯片做。

SGLang PR [#15103](https://github.com/sgl-project/sglang/pull/15103) 里没有手写 inline asm，而是调用 CUTLASS/CuTe DSL 的封装：

```python
if self.c_dtype == cutlass.Float16:
    x, y, z, w = utils.distributed.multimem_ld_reduce_8xf16(mc_ptr)
elif self.c_dtype == cutlass.Float32:
    x, y, z, w = utils.distributed.multimem_ld_reduce_4xf32(mc_ptr)
elif self.c_dtype == cutlass.BFloat16:
    x, y, z, w = utils.distributed.multimem_ld_reduce_8xbf16(mc_ptr)
elif self.c_dtype == cutlass.Float8E4M3FN:
    x, y, z, w = utils.distributed.multimem_ld_reduce_16xe4m3(mc_ptr)
elif self.c_dtype == cutlass.Float8E5M2:
    x, y, z, w = utils.distributed.multimem_ld_reduce_16xe5m2(mc_ptr)

utils.distributed.multimem_st_4xb32(mc_ptr, x, y, z, w)
```

这段代码要配合后面的 task partition 看：每个线程只处理自己负责的 128bit 数据。`ld_reduce` 把这一小段从各 rank 聚合回来，`st` 再广播回所有 rank 的对应位置。

![](https://files.mdnice.com/user/59/6fcf8256-35c9-49d3-9f74-a5dda8f58b25.png)

`multimem.ld_reduce` 是 reduce 读。slide 里的流程是：

1. 某个 GPU 发起 `multimem.ld_reduce`；
2. NVSwitch 根据 MC address 找到各个 rank 上对应的物理地址；
3. 从各个 rank 读数据，在交换芯片里做 reduce；
4. 把 reduce 后的数据返回给发起方的 register。

官方 CuTe DSL 的封装里，128bit 是基本粒度之一。以 FP32 为例，四个 FP32 正好 128bit：

```python
@cute.jit
def multimem_ld_reduce_4xf32(mc_ptr: cute.Pointer) -> tuple:
    return multimem_ld_reduce_128bit_base(
        "multimem.ld_reduce.weak.global.add.v4.f32 "
        "{$0, $1, $2, $3}, [$4];",
        mc_ptr,
    )
```

FP16/BF16/FP8 只是一个 128bit 里塞的元素数不同：

```python
multimem_ld_reduce_8xf16(...)
multimem_ld_reduce_8xbf16(...)
multimem_ld_reduce_16xe4m3(...)
multimem_ld_reduce_16xe5m2(...)
```

slide 上写的是 `sys.relaxed` 形式，官方 helper 里有些版本用了 `.weak`。这里不用纠结拼写差异，核心语义是一样的：对 MC address 做一次 reduce load，结果落到寄存器。

![](https://files.mdnice.com/user/59/f62c4748-d581-4a2d-a4c8-38d69d67e670.png)

`multimem.st` 是广播写。它接在 `ld_reduce` 后面用最自然：既然某个 rank 已经拿到了这 128bit 的 reduce 结果，就顺手把它广播回所有 rank 的输出地址。

CuTe DSL helper 里对应的是：

```python
@cute.jit
def multimem_st_4xb32(mc_ptr: cute.Pointer, val0, val1, val2, val3):
    cute.arch.asm(
        "multimem.st.weak.global.v4.f32 [$0], {$1, $2, $3, $4};",
        inputs=[mc_ptr.toint(), val0, val1, val2, val3],
        is_volatile=True,
    )
```

名字里写 `4xb32` 是因为 inline asm 按 4 个 32bit register 传参。对于 FP16/BF16/FP8，前面的 `ld_reduce` 已经把若干个低精度元素打包到了这四个 32bit register 里，store 阶段不用再关心原始 dtype。

![](https://files.mdnice.com/user/59/ccfce5cb-7b7c-4d0b-9447-57a6acdbc8be.png)

`multimem.red` 更像一个跨 rank 原子操作。slide 里说它可以拿来做 multi-rank sync，这正是 GEMM+AR kernel 里的用法。

比如一个 tile 的 TMA store 完成之后，epilogue warp 需要告诉 AR warp：这个 tile 在各 rank 上都已经写好了，可以开始 reduce。这个 flag 就可以通过 MC address 做一次 `multimem.red.add`。

CUTLASS 的封装大概是这样：

```python
@cute.jit
def multimem_red_add1(lock_ptr: cute.Pointer, order: cutlass.Constexpr, scope: cutlass.Constexpr):
    if order == "release" and scope == "sys":
        cute.arch.asm(
            "multimem.red.release.sys.global.add.s32 [$0], 1;",
            inputs=[lock_ptr.toint()],
            is_volatile=True,
        )
    elif order == "relaxed" and scope == "gpu":
        cute.arch.asm(
            "multimem.red.relaxed.gpu.global.add.s32 [$0], 1;",
            inputs=[lock_ptr.toint()],
            is_volatile=True,
        )
```

后面可以看到，tile ready 的局部 flag 和 kernel 结束前的全局 sync，使用的 scope/order 不一样。

# 0x5. 怎么拿到 MC address

![](https://files.mdnice.com/user/59/c054f647-14d5-4f25-ab17-1852041e3069.png)

这页讲初始化。要用 multimem 指令，首先得有 MC address。slides 里列了两种方式：

- nvshmem4py；
- torch symmetric memory。

两者本质上都要解决一件事：多张 GPU 上分配一组 symmetric memory，然后得到一组能被 multimem 指令访问的 multicast pointer。UC address 和 MC address 指向同一批物理内存，只是访问路径不同。

SGLang PR [#15103](https://github.com/sgl-project/sglang/pull/15103) 新增了 `NvshmemCommunicator`，用 nvshmem4py 做这件事：

```python
class NvshmemCommunicator:
    def __init__(self, group, device, rank, world_size):
        from nvshmem.core import get_unique_id, init

        unique_id = get_unique_id()
        torch.distributed.broadcast_object_list([unique_id], src=0, group=group)
        init(unique_id, rank, world_size)

    def create_symmetric_tensor(self, size, dtype):
        return nvshmem.core.tensor(size, dtype=dtype, device=self.device)

    def create_multicast_pointer(self, tensor):
        local_ptr = tensor.data_ptr()
        return nvshmem.bindings.mc_ptr(nvshmem.core.Teams.TEAM_NODE, local_ptr)
```

在 `GemmARLayer` 里，barrier flag 会优先走 NVSHMEM，失败后再 fallback 到 PyTorch symmetric memory：

```python
def _create_barrier_flags(self):
    num_flags = self.num_tiles + self.num_sms

    nvshmem_comm = get_nvshmem_comm()
    if nvshmem_comm is not None:
        barrier_flag = nvshmem_comm.create_symmetric_tensor(
            num_flags, dtype=torch.int32
        )
        barrier_flag_mc = nvshmem_comm.create_multicast_pointer(barrier_flag)
        return barrier_flag, barrier_flag_mc

    barrier_flag = symm_mem.empty(
        num_flags, dtype=torch.int32, device=torch.cuda.current_device()
    )
    handle = symm_mem.rendezvous(barrier_flag, self.group_name)
    barrier_flag_mc = handle.multicast_ptr
    return barrier_flag, barrier_flag_mc
```

SGLang 主线里也有一条和 symmetric memory 相关的普通 AllReduce 路线：PR [#10571 Support Torch Symm Mem AllReduce](https://github.com/sgl-project/sglang/pull/10571)。它不做 GEMM fusion，但对理解 MC address 很有帮助。当前代码里会先分配 symmetric buffer，再拿 rendezvous handle：

```python
self.buffer = torch_symm_mem.empty(
    buffer_size,
    dtype=torch.uint8,
    device=device,
)
self.symm_mem_hdl = torch_symm_mem.rendezvous(
    self.buffer,
    self.group.group_name,
)

if self.symm_mem_hdl.multicast_ptr == 0:
    self.disabled = True
```

然后根据硬件和 world size 选择 `multimem_all_reduce_` 或 `two_shot_all_reduce_`：

```python
if self.world_size in self._WORLD_SIZES_MULTIMEM[cc]:
    torch.ops.symm_mem.multimem_all_reduce_(
        self.buffer[: inp.numel()].view(inp.shape),
        "sum",
        self.group.group_name,
    )
else:
    torch.ops.symm_mem.two_shot_all_reduce_(
        self.buffer[: inp.numel()].view(inp.shape),
        "sum",
        self.group.group_name,
    )
```

所以 slide 11 虽然看起来只是初始化图，但它其实是整个 fused kernel 能成立的前提：输出矩阵和 barrier flag 都得放在支持 MC address 的内存上。

# 0x6. Kernel workflow：多塞一个 AR warp group

![](https://files.mdnice.com/user/59/c9e0ca37-1f4a-4ac5-9a3e-7b8e7cdb9b74.png)

这页给出 kernel 的主流程。原来的 persistent GEMM kernel 里已经有 TMA load、MMA、epilogue。现在多加一个 AR warp group：

1. MMA warp 算 tile；
2. epilogue warp 把 tile 结果写到 MC address 对应的 global memory；
3. epilogue warp 用 `multimem.red` 对 flag 加一；
4. AR warp 等这个 flag 到 `num_ranks`；
5. AR warp 对自己负责的 tile 分片做 `multimem.ld_reduce`；
6. AR warp 再用 `multimem.st` 把 reduce 结果广播回所有 rank。

SGLang PR [#15103](https://github.com/sgl-project/sglang/pull/15103) 里，tile ready 的等待逻辑是这样的：

```python
if warp_idx == self.all_reduce_warp_id[0]:
    with cute.arch.elect_one():
        flag = barrier_flag.iterator + tile_id
        utils.distributed.spin_lock_atom_cas_relaxed_wait(
            lock_ptr=flag,
            expected_val=num_ranks,
            reset_val=0,
            scope="gpu",
        )

self.all_reduce_sync_barrier.arrive_and_wait()
```

这里 `expected_val=num_ranks` 很关键。每个 rank 的 epilogue 都会对同一个 tile flag 做一次 `+1`，AR warp 看到 flag 到了 `num_ranks`，才说明这个 tile 的输入在所有 rank 上都写好了。

FlashInfer PR [#1695](https://github.com/flashinfer-ai/flashinfer/pull/1695) 的 CuTe DSL kernel 也用了同样的 two-shot 思路。它在 epilogue 后等待 TMA store 完成，然后到 MC flag 上做 arrive：

```python
if self.all_reduce == "two_shot":
    if warp_idx == self.epilog_warp_id[0]:
        cute.arch.cp_async_bulk_wait_group(0, read=False)
        with cute.arch.elect_one():
            flag = barrier_flag_mc.iterator + tile_id
            cute.arch.fence_acq_rel_gpu()
            spin_lock_multimem_arrive(flag)
            cute.arch.fence_proxy("alias")
```

这段比 SGLang PR 更接近一个最小教学版本。`cp_async_bulk_wait_group(0, read=False)` 保证 TMA store 真完成了，`spin_lock_multimem_arrive` 本质是对 multicast flag 做一次 `multimem.red.add`。

对应 helper 很短：

```python
@cute.jit
def spin_lock_multimem_arrive(lock_ptr: cute.Pointer):
    distributed.multimem_red_relaxed_gpu_add1(lock_ptr)
```

这个流程看起来绕，但背后的条件很清楚：AllReduce 不能读到还没写完的 tile。flag 是 tile 粒度的生产者消费者关系，multimem red 则让这个 flag 同时对多卡可见。

# 0x7. 任务划分：每个 rank 只负责 tile 的一块

![](https://files.mdnice.com/user/59/32c12604-52e8-4adf-8f4f-9c63ee76ace5.png)

这页讲 task partition。它是整个 two-shot 算法最容易被忽略，但也最值得看的一页。

假设输出 tile 是 `M x N`，有 `num_ranks` 张卡。如果每个 rank 都对整个 tile 做 `ld_reduce + st`，那所有 rank 会做重复工作。two-shot 的做法是把 tile 按 rank 切开，每个 rank 只负责其中一段。做完 reduce 后，再通过 `multimem.st` 把自己负责的结果广播给所有 rank。这样最终每张卡都拿到完整 all-reduce 输出，但 reduce 工作被摊开了。

SGLang PR [#15103](https://github.com/sgl-project/sglang/pull/15103) 里对应代码是：

```python
cta_mma_tile_m = self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape)
m_local_rank = int(cta_mma_tile_m / self.num_ranks)

tCgC_mc_slice_partitioned = cute.zipped_divide(
    tCgC_mc_slice,
    (m_local_rank, self.mma_tiler[1]),
)
tCgC_mc_local_rank = cute.slice_(
    tCgC_mc_slice_partitioned,
    ((None, None), (rank_id, 0)),
)

frgC_mc = thr_copy_fake.partition_S(tCgC_mc_local_rank)
```

`m_local_rank` 就是每个 rank 在 M 维负责的高度。`zipped_divide` 把 tile 切成 `(m_local_rank, N)` 的小块，`cute.slice_(..., (rank_id, 0))` 取出当前 rank 负责的那一块。

接下来，每个线程按 128bit 粒度遍历自己的元素：

```python
atom, loop_m, loop_n = frgC_mc.shape

for i in cutlass.range_constexpr(loop_m):
    for j in cutlass.range_constexpr(loop_n):
        if cute.elem_less(frpC[0, i, j], c_mc.shape):
            mc_ptr = frgC_mc[None, i, j].iterator

            x, y, z, w = utils.distributed.multimem_ld_reduce_8xbf16(mc_ptr)
            utils.distributed.multimem_st_4xb32(mc_ptr, x, y, z, w)
```

FlashInfer 的实现里能看到 128bit 粒度是怎么反推线程布局的：

```python
atom_val = 128 // c_mc.element_type.width
atom_thr_n = self.mma_tiler[1] // atom_val
atom_thr_m = len(self.all_reduce_warp_id) * (WARP_SIZE // atom_thr_n)
```

比如 BF16 的 element width 是 16bit，`atom_val=8`，一个线程一次处理 8 个 BF16。FP32 是 4 个，FP8 是 16 个。slides 上写「each instruction handle 128 bits」就是这个意思。

这里还有一个细节：slides 写同一个 SM 在不同 GPU 上处理同一个 tile。这样 flag 和 tile id 的对应关系会简单很多，AR warp 等的就是「这个 tile 在所有 rank 上的结果」。这也是 persistent scheduler 要和 distributed allreduce 逻辑配合的地方。

# 0x8. Workflow 细节：pre-sync、tile AR、final sync

![](https://files.mdnice.com/user/59/252e5c14-5782-4bee-bd30-fe4bc726b9bd.png)

这页把流程画得更细。按我的理解，它有三层同步。

第一层是 tile allreduce pre-sync。每个 tile 的 epilogue 写完后，通过 MC flag 告诉其他 rank：这个 tile ready 了。AR warp 等到 flag 到 `num_ranks`，才开始读这个 tile。

第二层是 tile 内部的 `ld_reduce + st`。每个线程只处理自己的 128bit 数据，没有跨线程依赖，所以不需要在每个元素之间加额外同步。这个设计很干净，线程之间少了很多麻烦。

第三层是 kernel 尾部的 final sync。因为 `multimem.st` 把结果广播到了所有 rank，kernel 结束前需要确保所有 SM 的 AR 工作都完成，否则后续 kernel 可能读到还没完全 broadcast 完的数据。

SGLang PR [#15103](https://github.com/sgl-project/sglang/pull/15103) 里 final sync 是按 SM 做的：

```python
last_tile_id_linear = self.num_tiles

utils.distributed.multimem_red_add1(
    lock_ptr=barrier_flag_mc.iterator + last_tile_id_linear + sm_id_linear,
    scope="sys",
    order="release",
)

utils.distributed.spin_lock_atom_cas_relaxed_wait(
    lock_ptr=barrier_flag.iterator + last_tile_id_linear + sm_id_linear,
    expected_val=num_ranks,
    reset_val=0,
    scope="sys",
)
```

这里和 tile ready flag 不一样，scope 变成了 `sys`，order 也用了 `release`。这个 sync 不是只在一个 GPU 内协调 warp，而是要保证所有 GPU 上的 multimem store 都对后续系统可见。

FlashInfer PR [#2171](https://github.com/flashinfer-ai/flashinfer/pull/2171) 后面专门修过这块。它抽了一个 `sm_wise_inter_gpu_multimem_barrier`：

```python
@cute.jit
def sm_wise_inter_gpu_multimem_barrier(barrier, barrier_mc, num_ranks):
    pid = cute.arch.block_idx()[0]

    distributed.multimem_red_release_sys_add1(barrier_mc + pid)
    cute.arch.fence_proxy("alias")
    spin_lock_atom_cas_acquire_wait(
        barrier + pid,
        expected_val=num_ranks,
        reset_val=0,
        scope="sys",
    )
```

这个修复 PR 的背景是 CUTLASS DSL 升级后 two-shot allreduce 出现 regression。也就是说，这类 kernel 对 memory ordering 和 barrier 写法很敏感，不能只看数学上等价。

# 0x9. Memory order：能 relax 的地方就 relax

![](https://files.mdnice.com/user/59/7102210e-06e1-4c3c-b4c8-6cc1381f5701.png)

![](https://files.mdnice.com/user/59/b530b233-6173-4f45-962c-fb552d6b7283.png)

slides 里有两页 memory order，内容基本一样。核心观点是：memory order 要尽量弱，但不能弱错地方。

对数据路径，也就是 `multimem.ld_reduce` 和 `multimem.st`，slides 给的是 relaxed：

```ptx
multimem.ld_reduce.sys.relaxed.global.add.v4.f32 {$0, $1, $2, $3}, [$4];
multimem.st.sys.relaxed.global.v4.f32 [$1], {$2, $3, $4, $5};
```

原因有两个。

第一，线程之间没有复杂依赖。前面 task partition 已经把每个线程负责的 128bit 切好了，同一段数据不会被多个线程重复处理。

第二，同一个线程里 `ld_reduce` 的输出 register 马上作为 `multimem.st` 的输入。这个顺序由单线程程序顺序保证，不需要给每个数据元素加更重的 acquire/release 语义。

但 flag 不一样。flag 表达的是「数据已经写好」或者「所有 rank 的 AR 已经完成」。这里必须让后续等待者看到正确的内存状态，所以 slide 里用了 release：

```ptx
multimem.red.release.sys.global.add.u32 [$0], 1;
```

SGLang PR [#15103](https://github.com/sgl-project/sglang/pull/15103) 的 final barrier 也正是这个思路：

```python
utils.distributed.multimem_red_add1(
    lock_ptr=barrier_flag_mc.iterator + last_tile_id_linear + sm_id_linear,
    scope="sys",
    order="release",
)
```

而 tile 内等待 flag 的地方，用的是 relaxed atomic CAS wait：

```python
utils.distributed.spin_lock_atom_cas_relaxed_wait(
    lock_ptr=flag,
    expected_val=num_ranks,
    reset_val=0,
    scope="gpu",
)
```

这里我觉得 slide 的讲法很实用：不是一上来就把所有操作都写成最强语义，而是先看数据依赖到底在哪里。数据元素本身可以 relaxed，跨 rank 的完成信号才需要 release/sys 这一类更强语义。否则性能会被 ordering 白白吃掉。

# 0xA. SGLang 里怎么接到模型执行路径

前面讲的是 kernel。SGLang PR [#15103](https://github.com/sgl-project/sglang/pull/15103) 还做了比较完整的上层接入，入口在 `model_runner.py`。

启动开关是环境变量：

```bash
SGL_USE_TP_OVERLAP=1
```

`ModelRunner` 初始化时会创建一个单独的 TP overlap group，然后初始化 symmetric tensor / multicast pointer。接着创建两个 `GemmARLayer`：

```python
self.gemm_ar_attn_op = GemmARLayer(
    group=_TP_OVERLAP_GROUP,
    N=self.model_config.hidden_size,
    K=self.model_config.hidden_size // self.tp_size,
    dtype=torch.bfloat16,
    mma_tiler_mn=(256, 256),
    cluster_shape_mn=(2, 1),
    use_2cta_instrs=True,
    use_tma_store=True,
    all_reduce="LDMCxSTMC",
)

self.gemm_ar_mlp_op = GemmARLayer(
    group=_TP_OVERLAP_GROUP,
    N=self.model_config.hidden_size,
    K=self.model_config.intermediate_size // self.tp_size,
    dtype=torch.bfloat16,
    mma_tiler_mn=(256, 256),
    cluster_shape_mn=(2, 2),
    use_2cta_instrs=True,
    use_tma_store=True,
    all_reduce="LDMCxSTMC",
)
```

这两个 op 会挂到每一层的 `self_attn.o_proj` 和 `mlp.down_proj` 上：

```python
layer.self_attn.o_proj.gemm_ar_attn_op = self.gemm_ar_attn_op
layer.mlp.down_proj.gemm_ar_mlp_op = self.gemm_ar_mlp_op
```

真正替换原始 all-reduce 的位置在 `linear.py`。原本 `RowParallelLinear` 的路径是先做本地 GEMM，再调用 `tensor_model_parallel_all_reduce(output_parallel)`。PR 里加了一个快速分支：

```python
def forward(self, input_, skip_all_reduce=False):
    if (
        input_.shape[0] >= 128
        and get_int_env_var("SGL_USE_TP_OVERLAP", 0) == 1
    ):
        if self.gemm_ar_attn_op:
            output = self.gemm_ar_attn_op.forward(input_, self.weight, self.bias)
        else:
            output = self.gemm_ar_mlp_op.forward(input_, self.weight, self.bias)

        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    output_parallel = self.quant_method.apply(self, input_, bias=bias)

    if self.reduce_results and self.tp_size > 1 and not skip_all_reduce:
        output = tensor_model_parallel_all_reduce(output_parallel)
```

这个 `input_.shape[0] >= 128` 很现实。小 batch/token 数下，fused kernel 的额外复杂度和启动成本不一定划算。PR 先把它放在更容易获益的大 M 场景里。

`GemmARLayer` 本身还做了一些约束检查：

```python
assert self.world_size in [2, 4, 8]
assert bias is None
```

所以这不是一个已经覆盖所有模型、所有 dtype、所有并行策略的通用替换。它更像一条 B200 上 TP overlap 的工程起点。

PR 讨论里也有一些边界很值得记。有人问能不能支持 Qwen3 / DeepSeek。周沛源的回复大意是：Qwen3 MoE 里 dense attention 的 O projection 可以用，但 MoE MLP down projection 还需要 group GEMM + AR；DeepSeek-V3 FP8 blockwise 还涉及 activation/weight scale，当前 kernel 还没支持这条 scale 路径。所以它不是「打开开关所有模型都变快」，模型结构和量化格式都要对上。

# 0xB. CUTLASS 官方示例和 SGLang PR 的关系

这套实现不是凭空写出来的。SGLang PR 讨论里直接贴了 CUTLASS 官方 examples 的位置：

```text
https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/distributed
```

现在 CUTLASS 仓库里能看到几个非常贴近 slides 的文件：

- [`all_reduce_two_shot_multimem.py`](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/cute/blackwell/kernel/distributed/all_reduce_two_shot_multimem.py)
- [`distributed_gemm_all_reduce_blackwell.py`](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/cute/blackwell/kernel/distributed/distributed_gemm_all_reduce_blackwell.py)
- [`distributed.py`](https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/utils/distributed.py)

其中 `distributed_gemm_all_reduce_blackwell.py` 的结构和 SGLang PR 非常接近。它的 docstring 直接写了几个关键词：

```text
Persistent tile scheduling, warp specialization, TMA, tcgen05.mma,
and LDMCxSTMC all-reduce.
```

`LDMCxSTMC` 这个名字也很贴切：load multicast reduce，再 store multicast。SGLang 里 `all_reduce="LDMCxSTMC"` 就来自这个思路。

FlashInfer 的 PR [#1695](https://github.com/flashinfer-ai/flashinfer/pull/1695) 则更像这条技术路线的早期独立实现。PR 标题是 `[cute_dsl] add gemm + all reduce (two_shot)`，描述只有一句：

```text
this pr adds gemm overlapped with two-shot allreduce (with multimem instructions)
```

它后面又经历了几个和 CUTLASS DSL 版本相关的维护 PR：

- [#1812](https://github.com/flashinfer-ai/flashinfer/pull/1812)：升级 cutlass，非 SM100 跳过 two-shot allreduce 测试；
- [#2171](https://github.com/flashinfer-ai/flashinfer/pull/2171)：修 CUTLASS DSL 4.3.1 升级后的 regression，并补了 `sm_wise_inter_gpu_multimem_barrier`；
- [#2833](https://github.com/flashinfer-ai/flashinfer/pull/2833)：把 `nvidia-cutlass-dsl` 要求提到 `>=4.4.2`。

这几个 PR 放在一起看，会更容易理解为什么 slides 里专门讲 memory order 和 final sync。GEMM+AR 不是把两个算子写进一个文件就完事，它对 CuTe DSL 版本、NVSHMEM/symmetric memory、barrier 语义和 SM100 指令支持都很敏感。

# 0xC. 性能结果

![](https://files.mdnice.com/user/59/20baf866-2b3e-4afc-9eb9-c8e2c821a546.png)

slides 给的结论是：大约 60% 的通信时间可以被隐藏，SGLang `bench_serving` 端到端吞吐提升 `+6.42%`。

PR [#15103](https://github.com/sgl-project/sglang/pull/15103) 里的数据和这页 slides 对得上。测试模型是 Qwen2.5-72B-Instruct，TP4，B200。原始吞吐和开启 overlap 后的吞吐是：

```text
original total token throughput: 22633.41 tok/s
overlap  total token throughput: 24086.24 tok/s
```

提升大概就是 `6.42%`。

slides 里的表格还拆了 prefill 和 decode：

```text
Prefill, Qwen2.5-72B-Instruct, TP4, ISL/OSL=4096/1
origin:          latency 5195.9 ms, throughput 21947 tok/s
origin+overlap:  latency 4902.7 ms, throughput 23356 tok/s

Decode, Qwen2.5-72B-Instruct, TP4, ISL/OSL=4096/128
origin:          latency 52.8 ms, throughput 685 tok/s
origin+overlap:  latency 50.5 ms, throughput 729 tok/s
```

这组数字不夸张，但很真实。TP overlap 不是把 GEMM 算得更快，而是把一部分通信尾巴藏起来。对端到端吞吐来说，6% 左右已经是很值得拿的收益，尤其是这种收益发生在大模型每层都会走的 row parallel projection 上。

PR 里还贴了一张 timeline 对比，GEMM+AR 从 `552us` 变成 `423us`。这也符合 slide 的说法：不是通信完全消失，而是大约一部分被 GEMM 后续 tile 的执行盖住了。

# 0xD. 小结

这套 GEMM+AR overlap 的主线可以压成几句话：

1. RowParallelLinear 的 GEMM 后面必有 AllReduce，所以 tile 算完之后立刻处理 AR，比等整个 GEMM 结束再起通信 kernel 更有机会重叠；
2. Blackwell 上 TMEM、TMA、NVLS/multimem 给了这个 fused kernel 一个比较合适的硬件窗口；
3. kernel 内部用 warp specialization 切出 AR warp group，epilogue warp 写 MC address 并发 flag，AR warp 等 tile ready 后做 `ld_reduce + st`；
4. two-shot partition 让每个 rank 只 reduce tile 的一部分，再 broadcast 给所有 rank，避免重复工作；
5. memory order 要小心，数据路径尽量 relaxed，完成信号和 final sync 才用 release/sys；
6. SGLang 当前公开 PR 已经把它接到了 attention `o_proj` 和 MLP `down_proj`，但模型结构、world size、量化格式和 CuTe DSL/NVSHMEM 环境还有限制。

我觉得这篇 slides 最有价值的地方，是它没有把「通信计算重叠」停在 timeline 解释上，而是直接落到了指令和线程分工：哪一个 warp 发 flag，哪一个 warp spin wait，哪一条 multimem 指令 reduce，哪一条 store broadcast，最后再用什么 scope 的 barrier 收口。这样的分享不太好讲，但对写 kernel 的人来说非常有用。

后续如果这条路线继续往前走，比较自然的方向应该是 group GEMM + AR、FP8 blockwise scale 支持，以及更稳定的 SGLang 主线集成。尤其是 MoE 模型里 MLP down projection 的 group GEMM，如果也能做成类似粒度的 overlap，收益空间会更有意思。
