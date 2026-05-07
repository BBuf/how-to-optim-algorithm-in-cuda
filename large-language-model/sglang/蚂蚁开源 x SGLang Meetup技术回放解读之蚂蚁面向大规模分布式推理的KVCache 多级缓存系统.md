> 这篇是对 2026 年 1 月 17 日蚂蚁开源 x SGLang Meetup 中「蚂蚁面向大规模分布式推理的 KVCache 多级缓存系统」这个分享的回放解读。slides 里真正的主线有三条：SGLang HiCache、面向 DeepSeek Sparse Attention 的 HiSparse、以及蚂蚁 Theta KVPool 在 Mooncake/SGLang 之上的生产化架构。这里我会按 slides 顺序讲，但重点放在代码和公开资料上：SGLang 的 HiCache/HiSparse/Mooncake Store 源码、DeepSeek-V3.2 相关 PR、Mooncake 的 Dummy/Real Client 文档，以及 LMSYS 的几篇 blog。

# 0x0. 前言

![](https://files.mdnice.com/user/59/ba77bb1f-3c13-495f-a8b4-faf8ad6480ec.png)

这场分享的题目是「蚂蚁面向大规模分布式推理的 KVCache 多级缓存系统」，嘉宾是蚂蚁集团的黄庭威和赵永科。黄庭威是 SGLang collaborator，GitHub ID 是 `huangtingwei9988`；赵永科是 Mooncake collaborator，GitHub ID 是 `zhaoyongke`。本次整理到的 mdnice 图片里少了讲者介绍页，所以这里不贴图，只在文字里补一下。

我读这套 slides 时最强的感受是：它不是单独讲「KVCache offload 到 CPU」这么简单。真正的问题是，当线上推理变成多租户、多轮对话、Agentic Coding、PD 分离、异构 TP、Sparse Attention 混在一起以后，KVCache 已经不是一个局部优化点，而是一个跨 scheduler、memory pool、storage backend、transfer engine 的系统问题。

SGLang 这条线最早可以追到 PR [#2693 Hierarchical Caching for SGLang](https://github.com/sgl-project/sglang/pull/2693)。后面又经历了 TP 修复、HiCache refactor、Mooncake/3FS/NIXL/AIBrix 后端接入、动态 backend、HiSparse 等一串工作。slides 里很多看似「平台架构」的图，落到代码里其实都能在这些文件里看到影子：

- `python/sglang/srt/mem_cache/hiradix_cache.py`
- `python/sglang/srt/managers/cache_controller.py`
- `python/sglang/srt/mem_cache/memory_pool_host.py`
- `sgl-kernel/csrc/kvcacheio/transfer.cu`
- `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py`
- `python/sglang/srt/managers/hisparse_coordinator.py`
- `python/sglang/jit_kernel/csrc/hisparse.cuh`

LMSYS blog 也有几篇刚好能对上这套 slides：

- [SGLang HiCache: Fast Hierarchical KV Caching with Your Favorite Storage Backends](https://www.lmsys.org/blog/2025-09-10-sglang-hicache/)：对应 slides 前半部分的 HiCache 数据面、控制面、Mooncake/3FS 后端。
- [SGLang Day 0 Support for DeepSeek-V3.2 with Sparse Attention](https://www.lmsys.org/blog/2025-09-29-deepseek-V32/)：对应 DeepSeek Sparse Attention 的背景，也就是 DSA、Lightning Indexer、Top-k Selector。
- [HiSparse: Turbocharging Sparse Attention with Hierarchical Memory](https://www.lmsys.org/blog/2026-04-10-sglang-hisparse/)：对应 slides 中间的分层稀疏化和 hot buffer/LRU/swap-in kernel。
- [Together with SGLang: Best Practices for Serving DeepSeek-R1 on H20-96G](https://www.lmsys.org/blog/2025-09-26-sglang-ant-group/)：不是这场 KVPool 分享的同一篇文章，但里面的蚂蚁 H20 大规模推理、Prefill/Decode 部署口径，可以帮助理解后面 Theta KVPool 的生产背景。

![](https://files.mdnice.com/user/59/c4739884-ecaf-40ac-b02d-e8de86617f3e.png)

目录分四部分：

1. SGLang Hierarchical Cache 分层缓存架构；
2. Hierarchical Sparse Attention 分层稀疏化；
3. Theta KVPool 架构设计与性能实测；
4. 未来规划。

我下面也按这个顺序写。先说 HiCache，因为它是后面所有东西的地基。

# 0x1. 从 RadixCache 到 HiCache

![](https://files.mdnice.com/user/59/32ee5985-705f-4970-8ec9-18d1255f240f.png)

这一页只是章节页，但它点出了第一个关键词：Hierarchical Cache。SGLang 原本已经有 RadixCache，也就是 RadixAttention 里的 GPU prefix cache。HiCache 做的是把这棵 radix tree 往 CPU 和远端存储扩展。

![](https://files.mdnice.com/user/59/017fc121-14ee-4d88-9e0f-b015602b8bcb.png)

RadixCache 的基本思想可以回到 LMSYS 2024 年那篇 [Fast and Expressive LLM Inference with RadixAttention and SGLang](https://www.lmsys.org/blog/2024-01-17-sglang/)。多轮对话、few-shot、self-consistency、agentic coding 这些场景都有共享 prefix。prefix 对应的 KVCache 如果还在 GPU 上，就可以跳过一大段 prefill 计算。

SGLang 的 `RadixCache` 把 token 序列挂在 radix tree 上，value 是 GPU KV page 的 index。命中时返回连续 prefix 的 device indices；插入时把新 KV page 接到树上；GPU 空间不够时按 LRU 从叶子驱逐。

问题也很直接：GPU HBM 就这么大。slides 里举了 DeepSeek V3 的例子，8 张 H20 只能放大约 130K token 的 KVCache。在线上多租户场景里，prompt 分布很散，某个会话刚刚写进 GPU 的 cache，很快就可能被别的请求挤掉。RadixCache 的结构没问题，容量不够。

![](https://files.mdnice.com/user/59/c0c241a5-6a70-4c3b-a6aa-1200d3254048.png)

这页用了「以存代算，存比算快」这个说法。我觉得它挺准确，但要补两个限定条件：

第一，必须是长 prefix 或长上下文。短 prompt 为了几十个 token 去 L3 拉 KV，可能不划算。

第二，缓存路径要能被隐藏或者批量化。如果每次 decode 都同步等 CPU/远端存储，那只是把算力瓶颈换成 I/O 瓶颈。

HiCache 的设计正好围绕这两点：L1 是 GPU，L2 是 CPU pinned memory，L3 是 file/Mooncake/3FS/NIXL/AIBrix 这类后端。Radix tree 不再只记录 GPU 上有没有 KV，而是扩展成 HiRadixTree，记录一个 prefix 的 KV 在 GPU、CPU、还是远端存储里。

LMSYS HiCache blog 里有一句话很关键：HiCache 把 RadixAttention 扩展成一个 HiRadixTree，用它当 page table，引用 GPU/CPU/外部存储里的 KV cache。代码里对应的是 `HiRadixCache(RadixCache)`：

```python
class HiRadixCache(RadixCache):
    def __init__(self, params: CacheInitParams, server_args: ServerArgs):
        self.page_size = params.page_size
        self.kv_cache = params.token_to_kv_pool_allocator.get_kvcache()

        if isinstance(self.kv_cache, MHATokenToKVPool):
            self.token_to_kv_pool_host = MHATokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
            )
        elif isinstance(self.kv_cache, MLATokenToKVPool):
            self.token_to_kv_pool_host = MLATokenToKVPoolHost(...)

        self.cache_controller = HiCacheController(
            params.token_to_kv_pool_allocator,
            self.token_to_kv_pool_host,
            self.page_size,
            self.tp_group,
            load_cache_event=self.load_cache_event,
            write_policy=server_args.hicache_write_policy,
            io_backend=server_args.hicache_io_backend,
            storage_backend=server_args.hicache_storage_backend,
            prefetch_threshold=prefetch_threshold,
            storage_backend_extra_config=extra_config,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
        )
```

这里有两层意思：

- `HiRadixCache` 仍然继承 `RadixCache`，prefix 匹配、树节点拆分、LRU 这些基本逻辑没有推倒重来；
- 具体的数据搬运交给 `HiCacheController`，树只关心某个 node 的 `value`、`host_value`、`backuped`、`evicted` 等元数据。

所以 HiCache 不是「另起炉灶做一个 cache」，而是给 RadixCache 增加了多级地址。

![](https://files.mdnice.com/user/59/73446102-19e4-475d-ba55-2d3fc0a5d25f.png)

这页是 HiCache 的总图。L1 GPU、L2 CPU、L3 Storage。Storage 可以是 Mooncake Store，也可以是 3FS。控制面有 CacheController，数据面有高效 I/O kernel 和 zero-copy。

`match_prefix` 可以看到这套元数据是怎么工作的：

```python
def match_prefix(self, params: MatchPrefixParams):
    value, last_node = self._match_prefix_helper(self.root_node, key)
    value = torch.cat(value) if value else empty_value

    host_hit_length = 0
    last_host_node = last_node
    while last_node.evicted:
        host_hit_length += len(last_node.host_value)
        last_node = last_node.parent
    while not last_host_node.backuped:
        last_host_node = last_host_node.parent

    return MatchResult(
        device_indices=value,
        last_device_node=last_node,
        last_host_node=last_host_node,
        host_hit_length=host_hit_length,
    )
```

如果 prefix 的前半段还在 GPU，`device_indices` 就能直接给 prefill 跳过用；如果后半段被 GPU 驱逐了，但 CPU 还有备份，就通过 `last_host_node` 和 `host_hit_length` 告诉 scheduler：这段有机会 load back。

真正从 CPU 拉回 GPU 的逻辑在 `load_back`：

```python
def load_back(self, node: TreeNode, mem_quota: Optional[int] = None):
    nodes_to_load = []
    while node.evicted:
        assert node.backuped
        nodes_to_load.insert(0, node)
        node = node.parent

    host_indices = torch.cat([n.host_value for n in nodes_to_load])
    if len(host_indices) < self.load_back_threshold:
        return None

    device_indices = self.cache_controller.load(
        host_indices=host_indices,
        node_id=last_hit_node.id,
        **self._get_extra_pools(),
    )
    if device_indices is None:
        self.evict(EvictParams(num_tokens=len(host_indices)))
        device_indices = self.cache_controller.load(...)

    for node in nodes_to_load:
        node.value = device_indices[offset : offset + len(node.host_value)].clone()
```

这里有一个小细节：`load_back_threshold` 默认是 10。太短的 host hit 不值得搬回 GPU，因为搬运本身也有开销。这个判断跟 slides 里「长序列 Load Cache 比计算快」是同一个思路：cache 不是永远要用，只有收益超过阈值才用。

L3 的预取走 `prefetch_from_storage`：

```python
def prefetch_from_storage(self, req_id, last_host_node, new_input_tokens, ...):
    prefetch_key = RadixKey(new_input_tokens, extra_key=last_host_node.key.extra_key)
    prefetch_key = prefetch_key.page_aligned(self.page_size)

    if (
        not self.enable_storage
        or len(prefetch_key) < self.prefetch_threshold
        or self.cache_controller.prefetch_rate_limited()
    ):
        return

    host_indices = self.cache_controller.mem_pool_host.alloc(len(prefetch_key))
    operation = self.cache_controller.prefetch(
        req_id,
        host_indices,
        prefetch_key,
        last_hash,
        prefix_keys,
        **self._get_extra_pools(),
    )
    self.ongoing_prefetch[req_id] = (
        last_host_node,
        prefetch_key,
        host_indices,
        operation,
    )
```

这就是 slides 里「L3 命中后先预取到 L2，再由 L2 load back 到 L1」的代码路径。注意它不是发现 L3 hit 后马上堵住主线程等远端 I/O 完成，而是把 prefetch 交给 controller 的后台线程。

# 0x2. Host Memory Pool 的布局为什么要改

![](https://files.mdnice.com/user/59/7d116e55-fc36-4741-b5b9-9f82cc212d56.png)

这页非常重要。它讲的是 HiCache 数据面的核心：GPU 侧计算天然是 layer-first，但 CPU/L3 侧 I/O 更喜欢 page-first。

GPU 上的 KVCache 一般按 layer 组织，因为 attention 计算是一层一层跑的。对于某一层，kernel 只需要这一层的 K/V。可是 L3 存储的粒度是 page，远端 get/set 最好一次拿一整页。如果 CPU 也用 layer-first，那么一个 page 的所有层数据在内存里并不连续，传给 Mooncake/3FS 时就很别扭。

所以 SGLang 的 host memory pool 支持几种 layout：

```python
def init_kv_buffer(self):
    if self.layout == "layer_first":
        dims = (2, self.layer_num, self.size, self.head_num, self.head_dim)
    elif self.layout == "page_first":
        dims = (2, self.size, self.layer_num, self.head_num, self.head_dim)
    elif self.layout == "page_first_direct":
        dims = (
            2,
            self.page_num,
            self.layer_num,
            self.page_size,
            self.head_num,
            self.head_dim,
        )
    elif self.layout == "page_head":
        dims = (
            2,
            self.page_num,
            self.head_num,
            self.page_size,
            self.layer_num,
            self.head_dim,
        )
```

slides 里那几行 shape 就来自这里。

`page_first` 把同一个 token/page 的所有 layer 连起来，方便 L3 I/O；`page_first_direct` 再进一步按 `page_num, layer_num, page_size` 组织，让 CPU->GPU 的 direct transfer 可以按「某 page 的某 layer」聚合；`page_head` 是后面异构 TP 要用的布局，它把 head 维度提到 page 里面，方便按 head 切分。

参数入口在 `server_args.py`：

```python
parser.add_argument(
    "--hicache-io-backend",
    choices=["direct", "kernel", "kernel_ascend"],
    help="The IO backend for KV cache transfer between CPU and GPU",
)
parser.add_argument(
    "--hicache-mem-layout",
    choices=[
        "layer_first",
        "page_first",
        "page_first_direct",
        "page_first_kv_split",
        "page_head",
    ],
    help="The layout of host memory pool for hierarchical cache.",
)
```

还有一段自动兼容逻辑：

```python
def _resolve_layout_io_compatibility(self):
    if self.hicache_mem_layout == "page_first_direct" and self.hicache_io_backend == "kernel":
        self.hicache_io_backend = "direct"

    if self.hicache_mem_layout == "page_first" and self.hicache_io_backend == "direct":
        self.hicache_mem_layout = "page_first_direct"

def _resolve_storage_layout_compatibility(self):
    if self.hicache_storage_backend != "mooncake" or self.hicache_mem_layout != "layer_first":
        return

    if self.hicache_io_backend == "direct":
        new_layout = "page_first_direct"
    elif self.hicache_io_backend == "kernel":
        new_layout = "page_first"
    self.hicache_mem_layout = new_layout
```

这段代码说明了线上默认倾向：如果后端是 Mooncake，就不要再用 `layer_first` 做 L3 存储。因为 Mooncake 要的是连续 page buffer 和 zero-copy，`layer_first` 不适合。

CPU/GPU 之间的 copy 有两套后端：

- `direct`：更接近普通 indexing/copy，适合 `page_first_direct`；
- `kernel`：走 SGLang 自己的 GPU-assisted I/O kernel，适合 `page_first`、`page_head` 这些需要 layout transform 的路径。

`load_to_device_per_layer` 把 layout 和 backend 映射到具体 kernel：

```python
def load_to_device_per_layer(self, device_pool, host_indices, device_indices, layer_id, io_backend):
    if io_backend == "kernel":
        if self.layout == "layer_first":
            transfer_kv_per_layer(...)
        elif self.layout == "page_first":
            transfer_kv_per_layer_pf_lf(...)
        elif self.layout == "page_head":
            transfer_kv_per_layer_ph_lf(...)

    elif io_backend == "direct":
        if self.layout == "layer_first":
            transfer_kv_direct(...)
        elif self.layout == "page_first_direct":
            transfer_kv_per_layer_direct_pf_lf(...)
```

反方向的 `backup_from_device_all_layer` 也类似：

```python
def backup_from_device_all_layer(self, device_pool, host_indices, device_indices, io_backend):
    if io_backend == "kernel":
        if self.layout == "page_first":
            transfer_kv_all_layer_lf_pf(...)
        elif self.layout == "page_head":
            transfer_kv_all_layer_lf_ph(...)
    elif io_backend == "direct":
        if self.layout == "page_first_direct":
            transfer_kv_all_layer_direct_lf_pf(...)
```

`sgl-kernel/csrc/kvcacheio/transfer.cu` 是 slides 里「IO kernel 3x throughput」的落点。它不是简单 `cudaMemcpyAsync`，而是用 warp 粒度搬连续 item：

```cpp
transfer_item_warp(int32_t lane_id, const void* src_addr, void* dst_addr, int64_t item_size_bytes) {
  const uint64_t* __restrict__ src = static_cast<const uint64_t*>(src_addr);
  uint64_t* __restrict__ dst = static_cast<uint64_t*>(dst_addr);
  const int total_chunks = item_size_bytes / sizeof(uint64_t);

  for (int j = lane_id; j < total_chunks; j += WARP_SIZE) {
    uint64_t tmp;
    asm volatile("ld.global.nc.b64 %0,[%1];" : "=l"(tmp) : "l"(src + j) : "memory");
    asm volatile("st.global.cg.b64 [%0],%1;" ::"l"(dst + j), "l"(tmp) : "memory");
  }
}
```

`ld.global.nc` 是 non-coherent load，`st.global.cg` 走 cache-global store。这个选择就是典型的流式搬运：KV page 搬过去马上给 attention 用，不需要像普通 tensor 算子那样反复读写同一块 cache line。

`page_head` 的 offset 也在这个文件里：

```cpp
// page head layout: [page_num, head_num, page_size, layer_num, head_dim]
return base + page_id / page_size * page_size * page_dim
     + page_dim / head_num * head_id * page_size
     + page_id % page_size * page_dim / head_num
     + layer_id * item_size_bytes / head_num;
```

如果是 `page_head`，一个 token 的所有 head 在内存中不是一个简单连续块，所以 kernel 要多一层 head loop：

```cpp
for (int64_t layer_id = start_layer_id; layer_id < start_layer_id + num_layers_to_process; ++layer_id) {
  for (int64_t head_id = 0; head_id < head_num; ++head_id) {
    const char* src_k_ptr = SrcOffsetFn(..., layer_id, ..., head_id, head_num, page_size);
    char* dst_k_ptr = DstOffsetFn(..., layer_id, ..., head_id, head_num, page_size);
    transfer_item_warp(lane_id, src_k_ptr, dst_k_ptr, head_size_bytes);
    ...
  }
}
```

这就是 slides 里「每个 layout 都有一个高效 IO kernel」的含义。layout 是为了 L3 和异构 TP，kernel 是为了把 layout transform 的成本压下来。

# 0x3. HiCache Scheduling Pipeline

![](https://files.mdnice.com/user/59/b4641516-73a1-4967-aeec-0215bc4228b5.png)

这一页是章节页。slides 标题叫 Scheduling Pipeline，但真正要看的是 `HiCacheController`。它负责三类事：

- GPU -> CPU：write through / write back；
- CPU -> GPU：load back，并且按 layer 做 overlap；
- CPU <-> L3：prefetch 和 backup 的后台线程。

`HiCacheController` 初始化时会注册 layer transfer counter：

```python
class HiCacheController:
    def __init__(..., write_policy="write_through_selective", io_backend="", storage_backend=None, ...):
        self.mem_pool_device_allocator = token_to_kv_pool_allocator
        self.mem_pool_device = token_to_kv_pool_allocator.get_kvcache()
        self.mem_pool_host = mem_pool_host
        self.write_policy = write_policy
        self.io_backend = io_backend

        self.layer_num = self.mem_pool_device.layer_num
        self.layer_done_counter = LayerDoneCounter(self.layer_num)
        self.mem_pool_device.register_layer_transfer_counter(self.layer_done_counter)

        self.write_buffer = TransferBuffer(self.stop_event)
        self.load_buffer = TransferBuffer(self.stop_event, buffer_count=10, max_buffer_size=100)
        self.write_stream = device_module.Stream()
        self.load_stream = device_module.Stream()
```

LMSYS HiCache blog 里提到一个点：CPU->GPU 命中时，HiCache 会做 layer-wise overlap。代码在 `start_loading`：

```python
def start_loading(self) -> int:
    producer_id = self.layer_done_counter.update_producer()
    op = CacheOperation.merge_ops(self.load_queue)
    host_indices, device_indices = self.move_indices(op.host_indices, op.device_indices)
    producer_event = self.layer_done_counter.events[producer_id]

    with device_module.stream(self.load_stream):
        for i in range(self.layer_num):
            self.mem_pool_host.load_to_device_per_layer(
                self.mem_pool_device,
                host_indices,
                device_indices,
                i,
                self.io_backend,
            )
            producer_event.complete(i)
```

这里每完成一层，就通过 `producer_event.complete(i)` 通知计算侧：第 i 层的数据已经可以用了。它不是等所有层都搬完再开始 prefill，而是 layer N 计算时，load stream 可以准备 layer N+1。这也是 HiCache 能把 CPU->GPU 传输藏进 prefill 的关键。

GPU->CPU 写回走 `start_writing`：

```python
def start_writing(self) -> None:
    op = CacheOperation.merge_ops(self.write_queue)
    host_indices, device_indices = self.move_indices(op.host_indices, op.device_indices)

    with device_module.stream(self.write_stream):
        self.mem_pool_host.backup_from_device_all_layer(
            self.mem_pool_device,
            host_indices,
            device_indices,
            self.io_backend,
        )
        finish_event.record()

    self.ack_write_queue.append(HiCacheAck(start_event, finish_event, op.node_ids))
```

这里 `CacheOperation.merge_ops` 是另一个容易忽略的小优化。多个 node 的写回/加载会被合并成一个批量操作，减少 kernel launch 和小块 DMA。

L3 prefetch 是后台线程。`prefetch_thread_func` 先查 L3 命中长度，再根据阈值决定要不要真的拉：

```python
def prefetch_thread_func(self):
    while not self.storage_stop_event.is_set() or not self.prefetch_queue.empty():
        operation = self.prefetch_queue.get(block=True, timeout=1)
        hash_value, storage_hit_count = self._storage_hit_query(operation)

        storage_hit_count_tensor = torch.tensor(storage_hit_count, dtype=torch.int)
        self._all_reduce_prefetch_groups(
            storage_hit_count_tensor,
            torch.distributed.ReduceOp.MIN,
        )
        storage_hit_count = storage_hit_count_tensor.item()

        if storage_hit_count < self.prefetch_threshold:
            self.prefetch_revoke_queue.put(operation.request_id)
            self.append_host_mem_release(operation.host_indices)
        else:
            operation.hash_value = hash_value[: storage_hit_count // self.page_size]
            operation.host_indices = operation.host_indices[:storage_hit_count]
            self.prefetch_buffer.put(operation)
```

这个 `all_reduce(min)` 很关键。TP 多卡时，每个 rank 都要对同一个 prefix 形成一致判断。如果 rank0 认为 L3 命中了 1024 token，rank1 只命中了 960 token，最终只能按 960 来，否则 page table 会不一致。

写到 L3 的入口是 `write_storage`：

```python
def write_storage(self, host_indices, token_ids, hash_value=None, prefix_keys=None) -> int:
    operation = StorageOperation(
        host_indices,
        token_ids,
        hash_value=hash_value,
        prefix_keys=prefix_keys,
    )
    self.backup_queue.put(operation)
    return operation.id
```

`HiRadixCache.write_backup_storage` 会在 CPU 写回完成后把 node 的 host page 写到 L3：

```python
def write_backup_storage(self, node: TreeNode):
    operation_id = self.cache_controller.write_storage(
        node.host_value,
        node.key,
        node.hash_value,
        prefix_keys,
        **self._get_extra_pools(),
    )
    self.ongoing_backup[operation_id] = node
    node.protect_host()
```

这一套控制面对应 slides 里的「CacheController 灵活支持 cache-aware scheduling 和 latency hiding」。它不只是 I/O wrapper，还要负责异步事件、rank 间一致性、host 内存回收、write policy、prefetch stop policy。

![](https://files.mdnice.com/user/59/7732c903-0e83-4bbf-9bc3-b4b37528f290.png)

这页讲 PD 分离和 TP 异构兼容，信息量很大。

先说 PD 分离。SGLang 的 P/D 模式里，Prefill 节点生成 KV，Decode 节点接着用。Mooncake TransferEngine 可以用 GDR/RDMA 做高速 KV transfer。问题是 Decode 节点不走 radix tree，也就是说 Decode 端生成的新 KV 没有自然进入 Prefill 端的 prefix cache。slides 里的方案是给 Decode 加 offload manager，把 Decode 阶段的 KV 也回写到远端 Global Remote Storage。下一轮请求如果落到 Prefill 节点，就能从 L3 复用。

SGLang 文档里的启动方式也能看到这个方向：

```bash
python3 -m sglang.launch_server \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-decode-enable-offload-kvcache \
  --hicache-storage-backend hf3fs
```

也就是说，Prefill 侧开 HiCache 用来跨实例复用，Decode 侧打开异步 offload，把 decode 产生的 KV 持久化回 L3。这正是 slides 中「P 节点用 Global Remote Storage 复用 D 节点 KVCache」的意思。

再说 TP 异构。不同服务集群可能用不同 TP 数。例如一个 Prefill 集群 TP8，一个 Decode 或另一个复用集群 TP4。MHA/GQA 下每个 rank 只保存一部分 head。如果直接用 `{model, token, rank}` 做 L3 key，TP4 和 TP8 的 rank 切分不一样，KV 就对不上。

`page_head` 的作用就是把 head 维度显式放进 layout，然后按 head shard 写多个对象。SGLang 里有一个专门的函数：

```python
def get_split_heads_page_buffer_meta(self, indices: torch.Tensor, split_factor: int):
    """
    get meta data for zero copy of heterogeneous ranks' KVCache
    """
    assert self.layout == "page_head"
    assert len(indices) % self.page_size == 0
    assert self.head_num % split_factor == 0
    ptr_list = []
    indices = indices.tolist()

    for index in range(0, len(indices), self.page_size):
        for head_id in range(0, self.head_num, self.head_num // split_factor):
            k_ptr = kv_buffer_data_ptr + ...
            v_ptr = k_ptr + v_offset
            ptr_list.append(k_ptr)
            ptr_list.append(v_ptr)

    element_size = (
        self.layer_num
        * self.dtype.itemsize
        * self.page_size
        * self.head_num
        * self.head_dim
        // split_factor
    )
    return ptr_list, [element_size] * len(ptr_list)
```

Mooncake Store 里会调用它：

```python
def _get_mha_split_heads_buffer_meta(self, keys, indices):
    ptr_list, element_size_list = (
        self.mem_pool_host.get_split_heads_page_buffer_meta(
            indices,
            self.split_factor,
        )
    )
    key_list = []
    for key_ in keys:
        for suffix in self.mha_suffix:
            key_list.append(f"{key_}_{suffix}_k")
            key_list.append(f"{key_}_{suffix}_v")
    return key_list, ptr_list, element_size_list
```

`split_factor` 来自 `tp_lcm_size`。如果线上共享 L3 的 TP 集合是 `{4, 8}`，LCM 是 8，那么 L3 里按更细的 8 份 head shard 存。TP4 的 rank 读两个 shard，TP8 的 rank 读一个 shard。这样 KVCache key 就不再绑死某个具体 TP 拓扑。

这个 feature 在官方 HiCache 文档里也有描述：MHA 模型用 Mooncake + `page_head` layout 时，HiCache 会基于 `tp_lcm_size` split head shards，让不同 TP 部署共享 KVCache。slides 这页画的 GCD/不同 TP clusters，本质就是这件事。

# 0x4. DeepSeek Sparse Attention 为什么还会卡显存

![](https://files.mdnice.com/user/59/ccede706-f72d-40dc-9dc7-f46bd55622d4.png)

这里进入第二部分：Hierarchical Sparse Attention。共同作者里有阿里云的 `hzh0425` 和 Stanford 的 `xiezhq-hermann`。这两位也正好是 HiCache/HiSparse 相关 PR 里经常能看到的名字。

![](https://files.mdnice.com/user/59/4b7e8dd3-9a9e-477b-a100-48ec47014d1d.png)

这页讲 DeepSeek Sparse Attention。DeepSeek-V3.2 相比 V3.1 加了 DSA。LMSYS 的 [DeepSeek-V3.2 Day 0 blog](https://www.lmsys.org/blog/2025-09-29-deepseek-V32/) 里也讲了同一件事：DSA 用 Lightning Indexer 快速筛选相关 token，再通过 Top-k Selector 只对选中的 KV 做 attention。

SGLang 侧对应的是 Native Sparse Attention，也就是 `nsa_backend.py` 和 `nsa_indexer.py` 这条路径。DSA 的 indexer 先产出 top-k token indices：

```python
topk_result = metadata.topk_transform(logits, self.index_topk)
```

attention backend 再把 top-k indices 转成 page table：

```python
if topk_transform_method == TopkTransformMethod.PAGED:
    page_table_1 = transform_index_page_table_prefill(
        page_table=metadata.page_table_1,
        topk_indices=topk_indices,
        extend_lens_cpu=metadata.nsa_extend_seq_lens_list,
        page_size=1,
    )
```

LMSYS blog 里说 DSA 把核心 attention 复杂度从 `O(L^2)` 降到 `O(Lk)`。这当然是大事，但 slides 下一页马上指出了系统层面的尴尬。

![](https://files.mdnice.com/user/59/afc39cae-87f4-467a-a898-f0368574ab47.png)

DSA 降的是计算量，不自动降低 KVCache 常驻显存。

原因很朴素：Top-k 是 decode 每一步、每一层动态选出来的。在算出 Top-k 之前，系统不知道下一步会访问哪些历史 token。为了保证 attention 能马上读到 KV，传统实现只能把全量历史 KV 留在 GPU。于是 attention 算子只读 2K 个 token，但 128K 的 KVCache 仍然占着显存。

slides 里给的数字是：128K 输入下，每个 step 有 98.5% KVCache 不会被访问。这里的浪费不是 compute，而是 capacity。GPU 显存被大量「这一刻不用，但可能下一刻要用」的 KV 占住，batch size 上不去。Sparse attention 的吞吐曲线早早 plateau，就是这个原因。

LMSYS HiSparse blog 也用了同样的判断：sparse attention 可能从 compute-bound 变成 capacity-bound。没有分层内存时，top-k 的稀疏性无法转换成并发度。

![](https://files.mdnice.com/user/59/f1298075-9682-4acd-90ee-737a44665f8e.png)

这页给出 HiSparse 的核心方案：

- CPU pinned memory 存完整 KV；
- GPU 只给每个 request 留一个小 hot buffer，比如 2K 或 4K/6K token slots；
- 每一步 decode 根据 Top-k 把需要的 KV swap-in 到 GPU；
- 新生成 token 的 KV 再异步 backup 到 CPU。

SGLang 文档里已经有 [HiSparse Guide](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/hisparse_guide.md)，里面写得很直白：HiSparse 目前面向 DSA 模型，例如 DeepSeek-V3.2、GLM-5，decode 阶段只保留一个 small hot KV buffer，完整 KV 放在 CPU pinned memory。

相关 PR 主要是这些：

- [#20343 HiSparse for Sparse Attention](https://github.com/sgl-project/sglang/pull/20343)：把 HiSparse 接到 model runner、KV pool、NSA backend。
- [#21932 Optimize the scheduling of decode backup](https://github.com/sgl-project/sglang/pull/21932)：优化 decode backup 调度。
- [#22238 Add readme docs for HiSparse Feature](https://github.com/sgl-project/sglang/pull/22238)：补文档。
- [#22425 Add HiSparse-DSA Model's nightly CI](https://github.com/sgl-project/sglang/pull/22425)：加 CI。
- 更早还有 [#11191](https://github.com/sgl-project/sglang/pull/11191) 和 [#14619](https://github.com/sgl-project/sglang/pull/14619) 这类 sparse + HiCache 的探索 PR。

入口在 `model_runner_kv_cache_mixin.py`。如果是 NSA/DSA 模型，并且开了 `--enable-hisparse`，SGLang 会换成 HiSparse 专用的 KV pool：

```python
if is_nsa_model:
    nsa_pool_kwargs = dict(
        size=self.max_total_num_tokens,
        page_size=self.page_size,
        dtype=self.kv_cache_dtype,
        device=self.device,
        kv_cache_dim=self.calculate_mla_kv_cache_dim(),
        index_head_dim=get_nsa_index_head_dim(self.model_config.hf_config),
    )
    if self.enable_hisparse:
        hisparse_cfg = parse_hisparse_config(self.server_args)
        nsa_pool_kwargs["host_to_device_ratio"] = hisparse_cfg.host_to_device_ratio
        self.token_to_kv_pool = HiSparseNSATokenToKVPool(**nsa_pool_kwargs)
    else:
        self.token_to_kv_pool = NSATokenToKVPool(**nsa_pool_kwargs)
```

allocator 也要换：

```python
if self.enable_hisparse:
    self.token_to_kv_pool_allocator = HiSparseTokenToKVPoolAllocator(
        self.max_total_num_tokens,
        page_size=self.page_size,
        dtype=self.kv_cache_dtype,
        device=self.device,
        kvcache=self.token_to_kv_pool,
        ...
    )
```

然后在 `model_runner.py` 初始化 `HiSparseCoordinator`：

```python
if self.enable_hisparse:
    hisparse_cfg = parse_hisparse_config(self.server_args)
    self.hisparse_coordinator = HiSparseCoordinator(
        req_to_token_pool=self.req_to_token_pool,
        token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
        top_k=hisparse_cfg.top_k,
        device_buffer_size=hisparse_cfg.device_buffer_size,
        device=self.device,
        tp_group=self.attention_tp_group.cpu_group if self.server_args.enable_dp_attention else self.tp_group.cpu_group,
        host_to_device_ratio=hisparse_cfg.host_to_device_ratio,
    )
```

一个常见配置是：

```bash
python3 -m sglang.launch_server \
  --kv-cache-dtype bfloat16 \
  --nsa-decode-backend flashmla_sparse \
  --enable-hisparse \
  --hisparse-config '{"top_k": 2048, "device_buffer_size": 6144, "host_to_device_ratio": 10}'
```

这里的 `top_k=2048` 对应 DSA 每层每步要看的 token 数，`device_buffer_size=6144` 就是 GPU hot buffer 容量。buffer 比 top-k 大，是为了让相邻 step 的 top-k 交集留在 GPU 上，不要每步都从 CPU 重拉。

![](https://files.mdnice.com/user/59/1f765622-e6b1-4aab-b3fa-cebf24b6dc2e.png)

这一页是「分层稀疏化框架」章节页。可以把它理解为：HiCache 是 prefix 粒度的层次缓存，HiSparse 是 top-k sparse attention 粒度的层次缓存。前者服务 prefill/prefix reuse，后者服务 decode/high concurrency。

# 0x5. HiSparse 的 hot buffer、diff kernel 和 LRU

![](https://files.mdnice.com/user/59/b6053f70-7a29-4fe1-9d95-c5eb29ebda4d.png)

这页讲增量 cache transfer。相邻 token 的 Top-k overlap 可以到 80%-90%，所以没必要每一步都把 2K 个 Top-k KV 从 CPU 拷到 GPU。正确做法是算 diff：

- 当前 Top-k 已在 device buffer：直接返回 device loc；
- 当前 Top-k 不在 device buffer：从 host pool 找 host loc，选择一个可驱逐 slot，拷到 device buffer；
- 更新 device buffer 的 token->slot 映射和 LRU。

SGLang 的 `HiSparseCoordinator` 就是这个协调层。初始化时它会创建一个 host KV pool 和若干映射表：

```python
class HiSparseCoordinator:
    def __init__(..., top_k, device_buffer_size, host_to_device_ratio=2):
        self.top_k = top_k
        self.device_buffer_size = device_buffer_size
        self.mem_pool_device = self.token_to_kv_pool_allocator.get_kvcache()
        self.mem_pool_host = MLATokenToKVPoolHost(
            device_pool=self.mem_pool_device,
            host_to_device_ratio=host_to_device_ratio,
            page_size=1,
            layout="layer_first",
            override_kv_cache_dim=self.mem_pool_device.kv_cache_dim,
        )

        self.req_to_device_buffer = torch.zeros((max_num_reqs, self.padded_buffer_size), ...)
        self.req_to_host_pool = torch.full((max_num_reqs, max_context_len), -1, ...)

        self.req_device_buffer_tokens = torch.full(
            (layer_num, max_num_reqs, self.padded_buffer_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.req_device_buffer_token_locs = torch.full(...)
        self.lru_slots = torch.arange(self.device_buffer_size, dtype=torch.int16, device=device)...
        self.top_k_device_locs_buffer = torch.full((max_num_reqs, self.top_k), -1, ...)
```

几个表的含义：

- `req_to_host_pool[req, token_pos]`：完整 KV 在 CPU host pool 的位置；
- `req_to_device_buffer[req, slot]`：hot buffer slot 对应的 device KV loc；
- `req_device_buffer_tokens[layer, req, slot]`：某层某 request 某个 slot 当前放的是哪个 token；
- `req_device_buffer_token_locs[layer, req, slot]`：slot 对应的 device loc；
- `lru_slots[layer, req, :]`：这个 request 这一层的 LRU 顺序。

prefill 结束后，HiSparse 要把已有 KV 备份到 CPU：

```python
def admit_request_into_staging(self, req: Req) -> None:
    logical_indices = self.req_to_token_pool.req_to_token[
        req.req_pool_idx, : len(req.fill_ids)
    ]
    device_indices = self.mem_pool_device._translate_loc_to_hisparse_device(logical_indices)

    host_indices = self.mem_pool_host.alloc(prefill_len)
    self.req_to_host_pool[req.req_pool_idx, :prefill_len] = host_indices

    with device_module.stream(self.write_staging_stream):
        self.mem_pool_host.backup_from_device_all_layer(
            self.mem_pool_device,
            host_indices,
            device_indices,
            io_backend="kernel",
        )
```

PD 分离下还有一个更有意思的 direct-to-host 路径：

```python
def admit_request_direct(self, req: Req) -> None:
    """Direct-to-host path: KV data already resides in host pool via RDMA."""
    self.alloc_device_buffer(req)

    if req.kv_allocated_len <= self.device_buffer_size:
        self._preload_to_device_buffer(req)
    else:
        self.req_device_buffer_tokens[
            :, req.req_pool_idx, : self.device_buffer_size
        ] = -1

    self._skip_first_backup[req.req_pool_idx] = True
```

这段和 HiSparse Guide 的「Prefill GPU 通过 RDMA 直接写 Decode Host Pool」对应。Decode 节点不需要先把完整 KV 接到 GPU，再从 GPU staging 到 CPU。它只分配一个小 device buffer，之后每步按 Top-k on-demand 拉。

每步 decode 还要把上一步新生成 token 的 KV 备份回 CPU：

```python
def map_last_loc_to_buffer(self, seq_lens, out_cache_loc, req_pool_indices, seq_lens_cpu):
    self._eager_backup_previous_token(
        seq_lens,
        req_pool_indices,
        seq_lens_cpu,
        req_pool_indices.cpu(),
    )
    reserved_buffer_loc = self._grow_device_buffers(...)
    self.req_device_buffer_token_locs[
        :, req_pool_indices, self.device_buffer_size
    ] = reserved_buffer_loc.to(torch.int32)
```

PR [#21932](https://github.com/sgl-project/sglang/pull/21932) 的核心就在这里附近：decode backup 不能挡住主 decode 路径，所以它用了独立 stream 和 event，把「上一 token 写回 CPU」这件事尽量排到旁路。

真正的 swap-in 在 `swap_in_selected_pages`：

```python
def swap_in_selected_pages(self, req_pool_indices, seq_lens, top_k_result, layer_id):
    top_k_indices = self.top_k_device_locs_buffer[:num_reqs]
    top_k_indices.fill_(-1)

    load_cache_to_device_buffer_mla(
        top_k_tokens=top_k_result,
        device_buffer_tokens=self.req_device_buffer_tokens[layer_id],
        host_cache_locs=self.req_to_host_pool,
        device_buffer_locs=self.req_device_buffer_token_locs[layer_id],
        host_cache=self.mem_pool_host.kv_buffer[layer_id],
        device_buffer=self.mem_pool_device.kv_buffer[layer_id],
        top_k_device_locs=top_k_indices,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        lru_slots=self.lru_slots[layer_id],
        item_size_bytes=self.mem_pool_host.token_stride_size,
        num_top_k=self.top_k,
        hot_buffer_size=self.device_buffer_size,
        page_size=1,
    )
    return top_k_indices
```

NSA backend 里会把 page table 翻译到 HiSparse device loc：

```python
if forward_batch.hisparse_coordinator is not None:
    page_table_1 = (
        forward_batch.token_to_kv_pool.translate_loc_to_hisparse_device(
            page_table_1
        )
    )
```

也就是说，attention kernel 后面看到的还是「Top-k 对应的一组 device loc」，只是这些 loc 不再是完整 KVCache pool 的 loc，而是 hot buffer 里的 loc。

`hisparse.cuh` 是这页最值得看的代码。kernel 每个 block 处理一个 request，短序列走 fast path：

```cpp
if (seq_len <= HOT_BUFFER_SIZE) {
  const int count = (seq_len < NUM_TOP_K) ? static_cast<int>(seq_len) : NUM_TOP_K;
  for (int i = tid; i < count; i += BLOCK_SIZE) {
    int32_t token_pos = req_top_k_tokens[i];
    if (token_pos >= 0) {
      req_top_k_device_locs[i] = req_device_buffer_locs[token_pos];
    }
  }
  return;
}
```

长序列才进入 diff 逻辑。第一步，把当前 top-k token 插入 shared memory hash table：

```cpp
for (int i = tid; i < NUM_TOP_K; i += BLOCK_SIZE) {
  int32_t token_idx = req_top_k_tokens[i];
  if (token_idx == newest_token) {
    s_top_k_tokens[i] = TOKEN_HIT;
    req_top_k_device_locs[i] = req_device_buffer_locs[newest_slot];
    s_newest_hit = 1;
  } else {
    int slot = hash_slot(token_idx, HASH_SIZE);
    while (true) {
      int32_t old = atomicCAS(&s_hash_keys[slot], HASH_EMPTY, token_idx);
      if (old == HASH_EMPTY || old == token_idx) {
        s_hash_vals[slot] = static_cast<int16_t>(i);
        break;
      }
      slot = (slot + 1) % HASH_SIZE;
    }
    s_top_k_tokens[i] = token_idx;
  }
}
```

第二步，扫描当前 hot buffer 的 LRU slots，看哪些 token 已经在 GPU：

```cpp
int16_t buf_slot = has_valid_slot ? req_lru_slots[slot_idx] : -1;
int32_t my_buffer_token = (buf_slot >= 0) ? req_device_buffer_tokens[buf_slot] : -1;

if (my_buffer_token >= 0) {
  int h = hash_slot(my_buffer_token, HASH_SIZE);
  while (true) {
    int32_t k = s_hash_keys[h];
    if (k == my_buffer_token) {
      my_found_top_k_idx = static_cast<int32_t>(s_hash_vals[h]);
      break;
    }
    if (k == HASH_EMPTY) break;
    h = (h + 1) % HASH_SIZE;
  }
}

if (is_hit) {
  s_top_k_tokens[my_found_top_k_idx] = TOKEN_HIT;
  req_top_k_device_locs[my_found_top_k_idx] = req_device_buffer_locs[buf_slot];
}
```

第三步，miss 的 token 选择 evict slot，然后从 host copy 到 device：

```cpp
if (is_miss) {
  int miss_offset = s_chunk_offset[chunk_idx] + local_miss_offset;
  int16_t evict_slot = s_lru_slots_out[HOT_BUFFER_SIZE - 1 - miss_offset];
  s_top_k_tokens[miss_offset] = my_token;
  req_top_k_device_locs[my_token_idx] = req_device_buffer_locs[evict_slot];
  req_device_buffer_tokens[evict_slot] = my_token;
}

for (int miss_idx = warp_id; miss_idx < total_misses; miss_idx += NUM_WARPS) {
  const int32_t miss_token = s_top_k_tokens[miss_idx];
  const int16_t evict_slot = s_lru_slots_out[HOT_BUFFER_SIZE - 1 - miss_idx];

  const int64_t src_loc = req_host_cache_locs[miss_token];
  const int64_t dst_loc = static_cast<int64_t>(req_device_buffer_locs[evict_slot]);

  const auto src_k = static_cast<const char*>(host_cache_k) + src_loc * item_size_bytes;
  auto dst_k = static_cast<char*>(device_buffer_k) + dst_loc * item_size_bytes;
  transfer_item_warp(lane_id, src_k, dst_k, item_size_bytes);
}
```

这个 kernel 同时做三件事：查 hit、排 LRU、拷 miss。它就是 slides 里「Diff Kernel」的实现。

![](https://files.mdnice.com/user/59/cf2ab333-8822-4748-b0b1-efd5d29ac745.png)

这页解释 hot buffer size 为什么会影响 hit rate。长序列越长，Top-k 的候选范围越大，相邻 step 的 Top-k overlap 会下降，miss 数上升。miss 多了以后，CPU->GPU copy 变成 decode 临界路径。

因此 `device_buffer_size` 不是随便设的。比如 `top_k=2048`，如果 hot buffer 也只有 2048，那么每一步几乎没有给历史命中留下余量；设到 4096 或 6144，才能让过去几十步的 top-k 结果在 GPU 上滚动保留。HiSparse blog 里也有类似结论：更大的 hot buffer 加 LRU 会显著降低 miss count。

这里要注意一个权衡：hot buffer 越大，每个 request 占的 GPU KV 空间越大，batch size 上限越低。HiSparse 不是把 GPU KV 变成 0，而是把「和上下文长度线性增长」变成「每个 request 固定一个 buffer」。这已经足够大幅提高长上下文 decode 并发。

![](https://files.mdnice.com/user/59/4e7c603c-755a-4573-b307-643356bba2ee.png)

这页讲 HiSparse 和 Radix Tree 的兼容性。问题很细：Radix Tree 管的是连续 prefix，Sparse Attention 每层访问的是离散 Top-k。甚至不同 layer 的 Top-k 还不一样，同一个 hot buffer slot 在 layer0 和 layer20 可能对应不同 token。那还能不能让 Radix Tree 直接管理 GPU hot buffer？

答案是不应该。

slides 里的方案是：CPU KV 是完整的，Radix Tree 只匹配 Host Indices；GPU Hot Buffer 由 Sparse Coordinator 自己管理和释放。这个设计在代码里也很明显：

- `HiRadixCache` 负责 prefix/page 级别的 host storage；
- `HiSparseCoordinator` 自己维护 `req_to_host_pool`、`req_device_buffer_tokens`、`lru_slots`；
- attention backend 拿到 Top-k 后，通过 `hisparse_coordinator.swap_in_selected_pages` 得到 hot buffer device loc。

这避免了把 radix tree 搞成「每层一棵离散 Top-k cache tree」。那样不仅复杂，而且和 prefix cache 的语义也不匹配。

![](https://files.mdnice.com/user/59/a21cbf53-67a4-40bc-be6d-75b77a7606fd.png)

这页是长序列压测结果：BatchSize 提升 5x，Decode throughput 提升 200%+。我不强行解读每个柱子的细节，因为 slides 截图里没有完整压测脚本和轴说明。但从系统角度看，这个结果是合理的：DSA 已经把每步 attention compute 降到 Top-k，HiSparse 再把 per-request GPU KV footprint 降到固定 hot buffer，于是 decode batch 可以明显变大。

如果要复现这条线，优先看三个地方：

- `docs/advanced_features/hisparse_guide.md`
- `test/registered/8-gpu-models/test_dsa_models_hisparse.py`
- PR [#22425](https://github.com/sgl-project/sglang/pull/22425)

CI 里跑的是 GLM5/DSA 模型的 HiSparse smoke。真实性能压测需要 PD 环境、H20/H200 这类机器、足够长的输入输出，以及 `flashmla_sparse` backend。

# 0x6. Mooncake 和 Theta KVPool 的背景

![](https://files.mdnice.com/user/59/1566bfae-a001-45ad-97de-ab80057aad42.png)

第三部分进入 Theta KVPool。这里 slides 从 SGLang 开源实现切到蚂蚁内部平台，但底层还是 Mooncake/SGLang HiCache 这套思想。

![](https://files.mdnice.com/user/59/861b2eb9-278d-4968-aae5-4f7c1910836f.jpg)

Mooncake 是一个 KVCache-centric 的分布式推理架构。论文是 [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://arxiv.org/abs/2407.00079)，代码在 [kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake)。SGLang 里默认的 P/D disaggregation transfer backend 也长期是 Mooncake。

在 HiCache 里，Mooncake 主要扮演 L3 distributed KV store。SGLang 的 Mooncake Store wrapper 在：

```text
python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py
```

配置可以来自环境变量：

```python
@dataclass
class MooncakeStoreConfig:
    local_hostname: str
    metadata_server: str
    global_segment_size: int
    protocol: str
    device_name: str
    master_server_address: str
    standalone_storage: bool
    client_server_address: str

    @staticmethod
    def load_from_env() -> "MooncakeStoreConfig":
        """
        export MOONCAKE_MASTER=10.13.3.232:50051
        export MOONCAKE_PROTOCOL="rdma"
        export MOONCAKE_DEVICE=""
        export MOONCAKE_TE_META_DATA_SERVER="P2PHANDSHAKE"
        """
```

也可以通过 SGLang 参数传：

```bash
python -m sglang.launch_server \
  --enable-hierarchical-cache \
  --hicache-storage-backend mooncake \
  --hicache-storage-backend-extra-config \
    '{"standalone_storage": true, "client_server_address": "127.0.0.1:50052"}'
```

这个 `standalone_storage` 就会和后面 slides 的 Dummy/Real Client 架构对上。

![](https://files.mdnice.com/user/59/9a0bc420-cd28-401b-af1c-497af1e97565.png)

Theta 是蚂蚁的大模型服务平台。slides 这页更多是平台介绍：模型服务接入、轻量微调部署、AI 应用、稳定性、成本和安全。

这一页没有对应的 SGLang 开源代码。它的意义在于告诉我们：KVPool 不是一个单机 demo，而是服务平台里的共享基础设施。只有放到平台层，L3 KVCache 才真的能跨实例、跨 P/D 角色、跨租户 workload 复用。

![](https://files.mdnice.com/user/59/ea22d5ca-c6d4-4bfb-ac53-d3ea3323b0da.png)

这页列了不同模型每 token KVCache 大小。大概能看出两件事：

第一，GQA/MHA 模型的 KV 通常比 MLA 模型大很多。比如 GLM 4.7 GQA 是 368KB/token，Qwen2.5-72B GQA 是 320KB/token；DeepSeek V3/R1 MLA 是 68.6KB/token，Kimi K2 MLA 也是 68.6KB/token。

第二，即使是 MLA，长上下文和大 batch 下总量也很可怕。68.6KB/token 乘 128K token 就是单请求接近 8.6GB 量级的 KV。线上不是一个请求，是上百上千个请求。靠 GPU HBM 硬扛一定会出问题。

SGLang 里的 MLA host pool 也专门处理了 MLA 的 KV 形态。Mooncake Store 里会判断 `is_mla_backend`，MLA 只需要 K 侧对象，不像 MHA 那样还要 K/V 两份 head shard：

```python
if self.is_mla_backend:
    key_multiplier = 1
else:
    key_multiplier = 2
```

这也是 HiCache 对 MLA 做写回优化的原因：MLA 下多 TP rank 可能持有相同 KV，没必要每个 rank 都往 L3 写一份。

![](https://files.mdnice.com/user/59/c3f3a23e-dacc-46e3-a181-4ad6db0c57a7.png)

这页讲 KVCache 扩容方式。模型架构层面可以用 MLA 降 KV 大小，系统层面可以用 BF16/FP8 KV、CPU/SSD/远端存储、多级缓存、PD 分离和全局 KVPool。

从 SGLang 代码看，HiCache 这套接口已经把 storage backend 抽象出来：

```python
class HiCacheStorage:
    def register_mem_pool_host(self, mem_pool_host): ...
    def batch_exists_v2(self, keys, pool_transfers=None, extra_info=None): ...
    def batch_get_v2(self, transfers, extra_info=None): ...
    def batch_set_v2(self, transfers, extra_info=None): ...
```

v2 接口的意义是支持 hybrid model。除了 KV 以外，DSA 的 indexer cache、Mamba 的 state/cache 也可以作为 `PoolTransfer` 参与 L3 读写：

```python
@dataclass
class PoolTransfer:
    name: str
    keys: List[str]
    host_indices: torch.Tensor
    hit_policy: PoolHitPolicy = PoolHitPolicy.ALL_PAGES
```

PR [#21259](https://github.com/sgl-project/sglang/pull/21259) 就是 Mooncake backend 支持 DSA & Mamba model；PR [#23241](https://github.com/sgl-project/sglang/pull/23241) 是 3FS backend 支持 DSA & Mamba model。换句话说，slides 里说「KVPool」时，未来并不一定只池化标准 attention KV，也可能池化 DSA indexer/mamba state 这类更杂的推理状态。

# 0x7. Dummy/Real Client 和 Zero-Copy

![](https://files.mdnice.com/user/59/0701a700-31f2-4a0c-a5ae-1edba31859e5.png)

这页是 Theta KVPool 的关键架构：Dummy Client 和 Real Client 分离。

普通社区部署里，SGLang 进程本身既跑推理，又持有 Mooncake client 的 RDMA/memory/resource 管理。这种方式简单，但在生产平台里有几个麻烦：

- 推理进程重启会影响本地注册内存和连接状态；
- RDMA 资源、内存池、SSD 资源和 engine 生命周期绑得太死；
- 多模型、多实例、平台调度时，状态难共享。

Mooncake 官方文档里也有 [Deployment with Dummy Client](https://kvcache-ai.github.io/Mooncake/getting_started/examples/sglang-integration/hicache-integration-v1.html) 这一节。它的表述是：SGLang Server 作为 Dummy Client，通过 RPC/IPC 连接本地 Store Service；Store Service 是 Real Client，负责真实内存池和 RDMA 连接。

slides 里的 Theta KVPool 更像是把这个模式平台化：

- `DummyClient` 绑定 engine，只转发请求给 `KVMaster`，自己不分配重资源，尽量无状态；
- `Real Client` 负责多级存储分配，接收 `KVMaster` 路由来的 Put/Get，持有状态；
- `KVMaster` 管 KV metadata 和路由。

SGLang 现有 `MooncakeStoreConfig` 已经有这几个字段：

```python
class MooncakeStoreConfig:
    standalone_storage: bool
    client_server_address: str
```

当 `standalone_storage=True` 时，SGLang 侧要求 host tensor allocator 来自 Mooncake：

```python
if config.standalone_storage:
    if not isinstance(mem_pool.allocator, MooncakeHostTensorAllocator):
        raise RuntimeError(
            "MooncakeStore with standalone_storage=True requires MooncakeHostTensorAllocator."
        )
```

这可以理解为社区版 Dummy Client 的雏形。Theta 这页多出来的是平台侧的 `KVMaster` 和更完整的 Real Client 资源管理。

![](https://files.mdnice.com/user/59/c8df87e8-2056-4ff7-a77c-bb887a61b7f9.png)

这页讲 Zero-Copy。普通 API 的问题是中间 buffer 太多：

1. 远端数据先到 Real Client 的 Local Buffer；
2. 再拷到 Dummy Client/Engine 的 buffer；
3. 最后写进 engine KV tensor。

batch 大了以后，瓶颈不是网络，而是本机内存 copy 和 CPU 调度。

Mooncake Store 的 zero-copy 接口正好解决这个问题。SGLang 注册 host pool buffer：

```python
def register_mem_pool_host(self, mem_pool_host: HostKVCache):
    super().register_mem_pool_host(mem_pool_host)
    assert self.mem_pool_host.layout in [
        "page_first",
        "page_first_direct",
        "page_head",
        "page_first_kv_split",
    ]
    buffer = self.mem_pool_host.kv_buffer
    super().register_buffer(buffer)

    bytes_per_page = mem_pool_host.get_ksize_per_token() * mem_pool_host.page_size
    self.gb_per_page = bytes_per_page / (1 << 30)
```

读取/写入不再传 tensor value，而是传目标地址和大小：

```python
def batch_set(self, keys, values=None, target_locations=None, target_sizes=None):
    assert len(keys) == len(target_locations) == len(target_sizes)
    put_result = self._put_batch_zero_copy_impl(
        set_keys,
        set_target_locations,
        set_target_sizes,
    )

def batch_get(self, keys, target_locations=None, target_sizes=None):
    get_result = self._get_batch_zero_copy_impl(
        keys,
        target_locations,
        target_sizes,
    )
```

底层就是 Mooncake 的 `batch_put_from` 和 `batch_get_into`：

```python
def _put_batch_zero_copy_impl(self, key_strs, buffer_ptrs, buffer_sizes):
    return self.store.batch_put_from(key_strs, buffer_ptrs, buffer_sizes)

def _get_batch_zero_copy_impl(self, key_strs, buffer_ptrs, buffer_sizes):
    return self.store.batch_get_into(key_strs, buffer_ptrs, buffer_sizes)
```

这和 slides 里的「注册 Engine KV Tensors 到 TransferEngine，Real Client 直接传到 Engine KV Tensors」是同一个方向。公开 SGLang 代码现在主要注册的是 host pool，Theta 图里更进一步强调 engine KV tensor 也可以被注册到传输引擎，从而减少 Dummy/Real 之间的中间拷贝。

Hybrid v2 路径也是 zero-copy：

```python
def _batch_io_v2(self, transfers: List[PoolTransfer], is_set: bool):
    for transfer in transfers:
        host_pool = self.registered_pools.get(transfer.name)
        ptr_list, element_size_list = host_pool.get_page_buffer_meta(host_indices)
        key_strs, key_multiplier = self._get_hybrid_page_component_keys(keys, transfer)

        if is_set:
            put_results = self._put_batch_zero_copy_impl(
                [key_strs[i] for i in missing_idx],
                [ptr_list[i] for i in missing_idx],
                [element_size_list[i] for i in missing_idx],
            )
        else:
            io_results = self._get_batch_zero_copy_impl(
                key_strs,
                ptr_list,
                element_size_list,
            )
```

所以这页不是抽象的「少拷贝」口号，代码里已经把接口形态改成了 ptr + size。

![](https://files.mdnice.com/user/59/030c3dcf-40c1-4270-bd37-fc990c9ff043.png)

这页是单节点部署：SGLang engine 主容器负责队列、GPU 推理 kernel、KV 生成和备份/恢复调度；`KVMaster` sidecar 管元数据；`KVPool Real Client` sidecar 管存储后端和预分配资源。

这个架构和 Mooncake Dummy Client 文档基本一致，只是 Theta 多了 KVMaster 这个平台层组件。对于生产环境，这个拆分有几个实用价值：

- engine 滚动升级不一定清空本机 KVPool；
- Mooncake/RDMA 资源可以由 sidecar 独立管理；
- KVMaster 可以把 metadata 做成服务，不必散在每个 engine 内；
- 单节点先打通以后，扩到 P/D 分离和跨实例共享更自然。

SGLang 侧对应的配置就是 `standalone_storage` 和 `client_server_address`。Mooncake 文档里启动方式大概是：

```bash
mooncake_master --eviction_high_watermark_ratio=0.95
mooncake_client --global_segment_size=4GB

python -m sglang.launch_server \
  --enable-hierarchical-cache \
  --hicache-storage-backend mooncake \
  --hicache-storage-backend-extra-config \
    '{"standalone_storage": true, "client_server_address": "127.0.0.1:50052"}'
```

如果把这里的 `mooncake_client` 换成 Theta 的 `KVPool Real Client sidecar`，基本就是 slides 里的单节点形态。

![](https://files.mdnice.com/user/59/9e850e87-bf2d-4162-93b0-40441ae5e760.png)

这页是 P-D 分离部署。差异是 `KVMaster` 在一个 P-D instance 内唯一，并且把 metadata 同步到 Tbase，用来做跨 instance KV 数据共享。

这里可以把它和前面的 HiCache/P-D 串起来：

- Prefill 节点生成大段 prefix KV；
- Decode 节点持续追加新 token KV；
- Decode offload manager 把这些 KV 写回 KVPool；
- KVMaster/Tbase 记录哪些 prefix/page 在哪些 Real Client 上；
- 新请求进来时，Prefill 节点可以从全局 KVPool 找到已有 KV，减少 recompute。

SGLang 开源里的 HiCache L3 metadata 目前主要由 storage backend 查询和 radix hash value 驱动；Theta 这页里的 KVMaster/Tbase 则是平台侧更强的 metadata 服务。二者不矛盾：SGLang engine 负责本地 tree 和执行路径，平台 KVPool 负责跨实例 metadata 和资源路由。

![](https://files.mdnice.com/user/59/c1a95b5d-25a9-4167-9ad6-5915957c0c4e.png)

这页是性能数据。slides 给了两个业务 case：

- Qwen3 Coder 单 8 H20-3e，TP8，scale-up，KV 扩容 18.2x；
- DeepSeek V3 PD，4 机 32 H20，2 个 Prefill(TP8) + 1 个 Decode(EP16)，scale-out，KV 扩容 25x。

右边柱状图里有 -23.17%、-39.26%、-19.16%、+9.06%、+20.09%、+8.4% 这些数字。截图没有完整 legend，我不在这里猜每个柱子的定义。能确定的是，这页想说明两件事：

1. KVPool 扩容后，缓存命中和长上下文吞吐有明显收益；
2. 这个收益不是单节点小实验，而是在 Qwen3 Coder 和 DeepSeek V3 的 H20 生产形态上测出来的。

这和 LMSYS HiCache blog 中 Ant Group 的反馈也能对上：DeepSeek-R1-671B 在 PD 分离部署、一般 QA 请求采样下，cache hit 相比 full recompute 能显著降低 TTFT。具体数字在 blog 里写的是 cache hit 平均 TTFT 降低 84%。slides 这里是另一套 Theta KVPool 口径，不能直接混算，但方向一致。

![](https://files.mdnice.com/user/59/90dd3ef4-215b-4557-beff-8b7c4e8622cc.png)

未来规划分两条。

HiCache：

- 支持 EPD；
- 支持 Hybrid LLMs；
- 优化 hierarchical sparse 性能。

这里的 EPD 可以联想到 SGLang 后续的 encoder-prefill-decode 分离方向；Hybrid LLMs 则对应 DSA/Mamba/linear attention 这类不只有标准 KV 的模型。PR [#21259](https://github.com/sgl-project/sglang/pull/21259) 和 [#23241](https://github.com/sgl-project/sglang/pull/23241) 已经能看到这条路：Mooncake/3FS backend 不再只处理 KV，还要处理 `PoolName.INDEXER`、`PoolName.MAMBA` 这些额外状态。

Mooncake：

- central Meta -> distributed Meta；
- 国产加速卡支持；
- KVCache quantization。

central Meta 到 distributed Meta 很好理解：全局 KVPool 如果所有 metadata 都卡在一个中心服务上，规模上去后会变成新瓶颈。国产加速卡支持和 KVCache quantization 则是平台落地必然会碰到的成本问题。

![](https://files.mdnice.com/user/59/0b51c575-e351-4fa4-9e73-7147d84e47b6.png)

最后一页给了社区贡献入口：

- SGLang HiCache Slack：`https://sgl-fru7574.slack.com/archives/C095B2L7UEB`
- Mooncake roadmap：`https://github.com/kvcache-ai/Mooncake/issues/1035`

如果只从代码切入，我建议按这个顺序读：

1. 先读 `hiradix_cache.py`，搞清楚 HiCache 如何在 RadixCache 上扩展 L1/L2/L3 元数据；
2. 再读 `cache_controller.py`，看 load/write/prefetch 的异步调度和 layer-wise overlap；
3. 然后读 `memory_pool_host.py` 和 `transfer.cu`，理解 `layer_first/page_first/page_first_direct/page_head` 为什么存在；
4. 接着读 `mooncake_store.py`，看 zero-copy 的 `target_locations/target_sizes` 怎么接入 Mooncake；
5. 最后读 `hisparse_coordinator.py` 和 `hisparse.cuh`，理解 DSA 下 hot buffer + diff kernel + LRU 的实现。

# 0x8. 小结

这套 slides 其实把 SGLang 近一年 KVCache 方向的几个大主题串到了一起。

HiCache 解决的是「GPU prefix cache 容量不够」：RadixTree 继续做 prefix metadata，CPU 和 L3 扩容，CacheController 负责异步搬运、预取、回填和写回。

Host layout 和 IO kernel 解决的是「多级存储不能只靠 cudaMemcpy」：GPU 计算喜欢 layer-first，L3 存储喜欢 page-first，异构 TP 还需要 page-head。SGLang 为这些 layout 写了专门的 transfer kernel，尽量把 layout transform 的成本压下去。

HiSparse 解决的是「Sparse Attention 省了计算但没省显存」：完整 KV 放 CPU，GPU 只保留固定 hot buffer，decode 每步根据 Top-k 做 hit/miss diff 和 LRU swap-in。这部分的核心代码已经在 `hisparse.cuh`，不是纸面设计。

Theta KVPool 解决的是「单个 engine 内的 cache 还不够生产化」：把 Mooncake/HiCache 的 L3 能力平台化，拆出 Dummy/Real Client、KVMaster、sidecar、zero-copy、跨 P/D instance metadata。公开 SGLang 里能看到 Mooncake Store 和 standalone storage 的基础形态，Theta slides 展示的是蚂蚁生产平台把这件事继续往前推了一层。

这类系统最容易被误解成「把 KVCache 放 CPU/SSD 就完了」。实际不是。真正难的是：什么时候值得拉、拉多少、怎么和计算重叠、不同 TP 怎么复用、decode 新 KV 怎么回写、远端 metadata 怎么查、I/O 小块怎么合并、hot buffer miss 怎么处理。slides 里每一页都在回答其中一个小问题，合起来才是一个能在线上跑的大规模 KVCache 多级缓存系统。
