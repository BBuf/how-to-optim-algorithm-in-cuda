# SGLang Custom AllReduce v2 线程模型和工作原理细节

这是 custom allreduce 细节系列的第二篇。[上一篇](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/blob/master/large-language-model/sglang/SGLang%20Custom%20AllReduce%20v1%20%E7%BA%BF%E7%A8%8B%E6%A8%A1%E5%9E%8B%E5%92%8C%E5%B7%A5%E4%BD%9C%E5%8E%9F%E7%90%86%E7%BB%86%E8%8A%82.md)把 v1 的 grid 配置、Signal 握手、双相位、内存序逐行拆完；更早的[《SGLang Custom AllReduce v1 与 v2 实现原理详解》](https://zhuanlan.zhihu.com/p/2065205306540531895)在 0x6/0x7 介绍过 v2 的结构划分和三个算法，但同样停在概念层。本文把 v2 拆到与 v1 篇相同的深度：kernel 起多少 block 多少线程、push 路径为什么一个跨 rank barrier 都没有、Semaphore 和 v1 的 Signal 在机制上差在哪、2shot 的中间 barrier 去哪了。读本文前建议先读完 v1 篇，两代的差异全部建立在 v1 的机制之上。

本文引用的代码来自 `python/sglang/kernels/jit/csrc/distributed/custom_all_reduce.cuh`（三算法 kernel 与 host 分发）、`python/sglang/kernels/jit/include/sgl_kernel/distributed/communicator.cuh`（`Counter` / `Semaphore` 原语与 `CommunicatorObj`）、`python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py`（Python 侧存储与算法选择）、`configs/custom_all_reduce_v2.py`（调优表）。分发链位置、graph 指针表的注册流程、调优表的完整数字见前篇的 0x8/0x9，这里只在涉及 kernel 行为时引用。

**TL;DR**

- **v2 把 v1 的 grid 策略反了过来：v1 固定线程数、按消息定 block 数；v2 固定 block 数、按消息定线程数。** push 的 grid 恒等于 SM 数（B200 是 148），pull 恒 96（sm90 是 64），multicast 恒 32；block size 在 128 到 1024 之间随消息挑。grid 必须固定，因为 push 的相位计数器和 pull 的 Semaphore 都是按 block 分配的状态。
- **1shot_push 没有任何跨 rank barrier。** 数据就绪由载荷本身的位型判定：+0.0 是"空槽"哨兵，push 前把真实的 +0.0 改写成数值等价的 -0.0，接收侧逐元素轮询到全部非零即到齐。关键路径 = 一次 NVLink 写 + 对端一次轮询命中。
- **push 的双相位靠数据依赖自我限速。** 快 rank 的下一次 push 排在自己这次 poll 之后，而 poll 必须等齐所有人的数据——任何 rank 领先不会超过一个相位，两半 buffer 交替正好够，这是 v1 双相位推演的无 barrier 版本。
- **pull 系的 Semaphore 把 v1 的"数值配对握手"换成"计数"。** 每个 lane 向所有 rank 的 flag 发一笔 `red.add.sys`（单向原子加，不回读），只有 lane == 本 rank 编号的那一个线程自旋等自家 flag 累计满 world_size。增量可交换、永不覆盖，所以不再需要 v1 的双相位槽位。
- **2shot 把 all-gather 融合进了写回。** 每个 rank 归约完自己的分片后，直接写回全部 world_size 家 workspace 的同一位置——这次写就是 all-gather，v1 的 stage2 和中间 barrier 整段消失。graph 模式下 workspace 就是各 rank 的输入张量，out = in，真正的原地 all-reduce。
- **内存序判据与 v1 完全一致：数据写和同步写是否在同一个 kernel 里。** pull 的入口恒 relaxed（数据写在 kernel 之前）；1shot 出口 relaxed（kernel 内只写本地输出）；2shot 出口 release/acquire（kernel 内写了 peer 的 workspace）；push 干脆不用 fence（就绪信息就在数据里）。

## 存算分离之后，kernel 拿到什么

v1 的 C++ 类持有 IPC 映射、注册表游标和算法选择；v2 的 kernel 文件头写明了新的分工："the kernels carry no storage or IPC logic: all pointers arrive via `CommunicatorObj` (owned by Python) and the per-call `AllReduceParams`"。`CommunicatorObj` 只做两件事：用 `TensorMatcher` 校验 Python 递进来的每块 tensor 的形状、dtype、设备，然后把裸指针存下来。分配、IPC 交换、生命周期全部归 Python。

kernel 每次调用拿到的是一个按值传入的参数包：

```c++
// 来源：sglang/python/sglang/kernels/jit/csrc/distributed/custom_all_reduce.cuh
template <uint32_t kWorldSize>
struct AllReduceParams {
  const void* __restrict__ input;
  void* __restrict__ output;
  uint32_t num_elements;
  uint32_t rank;
  void* const* __restrict__ graph_params;   // graph 模式：指针表某一行的地址
  uint8_t* pull_workspaces[kWorldSize];     // 以下三组必须是 symmetric memory
  uint8_t* push_workspaces[kWorldSize];
  Semaphore* pull_semaphores[kWorldSize];
  Counter* push_counter;                    // rank 本地
  uint8_t* pull_mc_workspace;               // multicast 地址（可为空）
  int64_t push_buffer_stride;               // push 单 buffer 字节数
};

template <typename Impl, uint32_t kWorldSize, int kShot>
__global__ void __launch_bounds__(1024, 1)
all_reduce_kernel(const __grid_constant__ AllReduceParams<kWorldSize> params);
```

对照 v1 的 kernel 签名可以看出连续性和变化。连续的是延迟绑定：graph 模式下 peer 的输入地址依然要等 capture 结束才知道，所以依然要多绕一层指针——只是 v1 的 `RankData*` 指向 C++ 类内部游标分配的条目，v2 的 `graph_params` 指向 Python 侧那张集中的 `[131072, world_size]` u64 表的某一行。变化的是其余一切都摊平成了值：workspace、semaphore 指针在初始化后不变，直接以 `__grid_constant__` 参数进 kernel 常量区，C++ 侧不再有任何查表。

模板参数也换了口味：v1 预编译 ngpus ∈ {2, 4, 6, 8} 四个实例进 sgl-kernel；v2 走 tvm-ffi JIT，按 (dtype, world_size, 是否 PDL) 惰性实例化，`kMaxWorldSize = 16`，world_size 2 到 16 连奇数都支持——GB200 NVL72 的 ws=16 就靠这个（多机 MNNVL 时 graph 零拷贝关闭，workspace 走 fabric 句柄，见 `can_use_custom_all_reduce_v2`）。

## grid 策略：和 v1 反着来

v1 的做法是 threads 固定 512、block 数随消息从 2 涨到 36 封顶。v2 完全反过来——**block 数固定、block size 随消息变**：

```c++
// 来源：sglang/python/sglang/kernels/jit/csrc/distributed/custom_all_reduce.cuh
// Pick the smallest block size whose grid still fits in one wave; the kernels
// are grid-stride so any choice is correct, this only tunes occupancy.
uint32_t choose_block_size(uint32_t num_threads) {
  for (const uint32_t block_size : {128u, 256u, 512u}) {
    if (host::div_ceil(num_threads, block_size) <= kNumSM) return block_size;
  }
  return 1024u;
}
```

block 数从哪来？三个算法各有一个常数，全部出自调优表而不是消息尺寸：

| 算法 | block 数（sm100 / sm90，ws=8） | 出处 |
|---|---|---|
| 1shot_push | 148（= SM 数）/ 132 | `num_push_blocks`，注释标明 "not configurable (bound to the counter array)" |
| 1shot_pull / 2shot_pull | 96 / 64 | `num_pull_blocks`（ws=2 时 sm100 用满 SM 数） |
| 2shot·multicast | 32 / 16，固定 512 线程 | `num_mc_blocks`，源码注释 "too much traffic will degrade performance in multicast" |

<p align="center">
  <img src="https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/article-assets-sglang-custom-allreduce-v2/custom_ar_v2_fig1_grid.png" width="98%" alt="v1 与 v2 的 grid 策略对比">
</p>

<p align="center">
  <em><b>grid 策略对比。</b>v1 固定 512 线程、block 数随消息 2..36；v2 三个算法各有固定 block 数（push=SM 数、pull=96、mc=32），block size 由 choose_block_size 在 128..1024 里挑。</em>
</p>

grid 为什么必须固定？答案藏在同步结构里，这是 v2 最值得记住的设计约束之一。push 的相位计数器 `push_counter` 每 block 一个，相位 = 计数 % 2 决定这次调用写双缓冲的哪一半——但数据槽位是按（相位, 来源 rank）编址的，**所有 block 写的是同一半 buffer**。只有每次调用让每个 block 的计数器都恰好加一，全部 block 才对"当前相位"有一致答案；一旦某次调用少起了几个 block，没跑到的 block 计数落后，下次调用里它们会写错半区。pull 的 Semaphore 数组按 `num_pull_blocks` 条分配，block b 永远用第 b 条，每次调用每条 flag 恰好累计 2×ws 个增量——同样要求 grid 恒定。v1 的 grid 随消息变没有这个问题，因为它的 Signal 槽位记录的是绝对轮次号，每个 block 的握手自成一体。

代一个数字感受下差异：16 KB 消息（1024 个 16B 元素）在 v1 里起 2 个 block × 512 线程 = 1024 线程，线程数与数据严丝合缝；在 v2 里 push 起 148 × 128 = 18944 线程、pull 起 96 × 128 = 12288 线程，只有前 1024 个线程碰数据，**其余线程和 block 纯粹为了维持同步状态的 lockstep 而存在**。小消息下多余 block 的成本是每 block 一次计数器操作或一组 red.add，微不足道；换来的是大消息时有整机的 SM 可用（v1 封顶 36 个）。

## workspace：一次 symm-mem 分配切出全部共享状态

v1 的三块存储（Signal+中转、预注册输入缓冲、rank_data 表）由 C++ 手工 `cudaMalloc` + cudaIpc 逐块交换；v2 只做一次 `_SymmetricMemory.empty_strided_p2p` + rendezvous，然后在 Python 里切片：

```python
# 来源：sglang/python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py
# Layout per rank: [2 * world_size push buffers | pull buffer | pull semaphores]
push_num_bufs = 2 * self.world_size  # 2 phases x world_size peers
push_ws_bytes = push_num_bufs * self.max_push_size
pull_ws_bytes = self.max_pull_size
pull_sem_bytes = _SEMAPHORE_BYTES * cfg.num_pull_blocks   # 128 B × 96
```

<p align="center">
  <img src="https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/article-assets-sglang-custom-allreduce-v2/custom_ar_v2_fig2_workspace.png" width="98%" alt="v2 的 workspace 布局">
</p>

<p align="center">
  <em><b>一块 symmetric memory 切出全部共享状态（B200，ws=8，默认 16 MB cap）。</b>push 区 2×8 个 buffer 按 [相位][来源 rank] 编址；pull 区单块；semaphore 每 block 一条 128 B。另有两件 rank 本地状态：push_counter 和 graph_params 指针表。</em>
</p>

把 B200、ws=8 的默认配置代进去算一笔账。push 单 buffer 尺寸取调优表的 push 阈值上限（graph 0.5 MB 与 eager 0.75 MB 取大）= 768 KB，乘 16 个 buffer 是 12 MB；pull 区按表想要 128 MB，被默认 16 MB cap（`SGLANG_CUSTOM_ALL_REDUCE_V2_MAX_SIZE_KB`）压到 16 MB；semaphore 96 × 128 B ≈ 12 KB。每 rank 常驻约 28 MB 对称内存，外加本地的 graph_params 表 131072 × 8 × 8 B = 8 MB。启动日志里 "All Reduce config: symmetric_memory = ..." 打印的就是这笔账。rendezvous 顺带返回 NVLS multicast 地址——pull 区在 multicast 空间的别名，供 2shot 的 multicast 档用；硬件不支持时该档关闭。

`Semaphore` 128 B 对齐单住一条 cache line，避免相邻 block 的 flag 伪共享。这块存储平面也不是 all-reduce 专用：K3 的 fused TP QK-norm 就创建 pull 极小化的 push-only 实例复用同一套协议，Python 侧为此特意保证两个方向各至少分配 1 KB。

## 1shot_push：零 barrier，就绪信息在载荷位型里

这是 v2 相对 v1 真正的新算法，也是小消息（decode bs=1 的 16 KB 正落在此档）的默认路径。先看数据面：

```c++
// 来源：sglang/python/sglang/kernels/jit/csrc/distributed/custom_all_reduce.cuh
static SGL_DEVICE void push_impl(uint32_t num_vecs, void* (&data)[kWorldSize], const void* src) {
  for (auto vid = global_tid; vid < num_vecs; vid += num_threads) {
    vec_t vec;
    ld_global_16B(vec, src, vid);          // 读本地输入 16 B
    for (uint32_t j = 0; j < kVecSize; ++j) {
      clear_pos_zero(vec[j].x);            // 载荷里真实的 +0.0 改写成 -0.0
      clear_pos_zero(vec[j].y);
    }
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      st_relaxed_16B(vec, data[i], vid);   // 同一份 vec 打给全部 ws 家（含自己）
    }
  }
}
```

`data[i]` 指向 rank i 的 push 区中属于本 rank 的槽位：`push_workspaces[i] + rank × stride + phase × stride × ws`——每家 workspace 里给每个来源 rank 预留了独立 buffer，按（相位, 来源）编址，写入之间没有任何竞争。

接收侧不等旗标，等数据本身：

```c++
// 来源：同上（poll_impl 主干）
do {
  bool has_zero = false;
  for (uint32_t i = 0; i < kWorldSize; ++i) ld_relaxed_16B(vec[i], data[i], vid);
  for (...) has_zero |= is_pos_zero(vec[i][j].x) | is_pos_zero(vec[i][j].y);
  if (!has_zero) break;                    // 全部元素非 +0.0 = ws 家都到齐
} while (true);
const auto out_vec = reduce(vec);          // fp32 累加，固定 0..ws-1 序
st_global_16B(out_vec, out, vid);
for (uint32_t i = 0; i < kWorldSize; ++i)
  st_global_16B(pos_zero_vec, data[i], vid);   // 回填 +0.0，恢复空槽标记
```

<p align="center">
  <img src="https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/article-assets-sglang-custom-allreduce-v2/custom_ar_v2_fig3_push.png" width="98%" alt="1shot_push 的数据面与哨兵">
</p>

<p align="center">
  <em><b>1shot_push（world_size=4，站在 rank1 视角）。</b>上：本 rank 的 vec 经 clear_pos_zero 后用 st.relaxed.sys 写进 4 家 push 区的槽位 [相位][1]。下：poll 反复读自己的 4 个槽位，任一元素还是 +0.0 就继续读；到齐后 fp32 归约、写输出、槽位回填 +0.0。</em>
</p>

把这套机制的几个关键点拆开。

**哨兵在元素粒度上工作，所以不怕撕裂。** workspace 初始化时清零，全部字节是 +0.0 的位型，这就是"空槽"标记。push 前把载荷里真实出现的 +0.0 改写成 -0.0——IEEE 754 下两者数值相等，加法结果不变——于是**载荷的每一个 2 B（bf16/fp16）或 4 B（fp32）元素都保证非"空"**。poll 端逐元素检查：16 B 的远程写就算在传输中被拆开，已到的元素显示非零、未到的显示 +0.0，判定依然正确，不需要假设 16 B 原子性。一个可以接受的边角：如果某个元素在所有 rank 上都是 ±0.0，累加结果是 -0.0 而不是 +0.0——数值等价，位型上与 NCCL 的 +0.0 不同，这是该算法为零 barrier 付出的唯一精度层面代价。

**同步语义全部由 relaxed 访存承担。** push 用 `st.relaxed.sys.global.v4.b32`、poll 用 `ld.relaxed.sys.global.v4.b32`——sys scope 保证跨设备的写能被对端观察到，relaxed 表示不对周围访存排序。回想 v1 篇的判据："数据写和旗标写是否在同一个 kernel 里"决定要不要 fence；push 路径里**数据就是旗标**，两者合一，判据退化，连 volatile/release 的选择题都不存在了。消费后的回填用普通 `st.global` 写自己显存，下一个读它的人是两次调用之后的远程 relaxed load，中间隔着足够的因果链。

**没有 barrier，靠什么防止覆盖？** 唯一的"同步"是本地的：

```c++
static SGL_DEVICE bool sync_enter_push(const AllReduceParams<kWorldSize>& params) {
  device::PDLWaitPrimary<kUsePDL>();
  return (params.push_counter[blockIdx.x].get() % 2) != 0;   // 读相位
}
static SGL_DEVICE void sync_exit_push(const AllReduceParams<kWorldSize>& params) {
  device::PDLTriggerSecondary<kUsePDL>();
  __syncthreads();
  if (threadIdx.x == 0) params.push_counter[blockIdx.x].inc(1);  // 相位翻转
}
```

进门读相位、出门翻相位，全程不看任何 peer 的状态。防覆盖的推演和 v1 的双相位一节同构，只是载体从旗标换成了数据：担心的场景是快 rank 的第 k+2 次 push（与第 k 次同相位）覆盖我还没消费完的第 k 次数据。但它的第 k+2 次 push 排在它自己第 k+1 次 poll 之后（stream 序），第 k+1 次 poll 必须等到**我**的第 k+1 次 push，而我的第 k+1 次 push 又排在我第 k 次 poll（含回填）之后。链条闭合：任何 rank 到达 k+2 时，所有 rank 的第 k 次调用连同回填都已了结。v1 用"通过 barrier k+1 证明全员消费完 k"，push 用"poll 等齐数据"达成同一个不变式——**数据依赖天然限速，领先不超过一个相位，两半 buffer 正好够**。

延迟账：关键路径 = 本 rank 一轮远程写 + 对端轮询命中，没有任何 barrier 的往返。代价是双相位 buffer 的显存（2 × ws 份）和消费后的回填写。graph 模式对 push 没有意义也不支持——它只读本地输入，不需要 peer 指针表，图内图外行为一致，host 侧直接 `RuntimeCheck(!use_graph)`。

## Semaphore：从数值握手到计数

pull 系的两个算法仍然需要真 barrier，但原语换了。先看结构：

```c++
// 来源：sglang/python/sglang/kernels/jit/include/sgl_kernel/distributed/communicator.cuh
struct alignas(128) Semaphore {
  SGL_DEVICE void put_relaxed()  { asm volatile("red.relaxed.sys.global.add.u32 [%0], 1;" ...); }
  SGL_DEVICE void put_release()  { asm volatile("red.release.sys.global.add.u32 [%0], 1;" ...); }
  SGL_DEVICE uint32_t get_relaxed() const { ... "ld.relaxed.sys.global.u32" ... }
  SGL_DEVICE uint32_t get_acquire() const { ... "ld.acquire.sys.global.u32" ... }
 private:
  uint32_t m_flag;      // 被所有 rank red.add 的计数 flag
  Counter m_counter;    // 只有 owner rank 动的基线预留计数
};
```

barrier 本体（进门版；出门版结构相同，方向相反）：

```c++
// 来源：sglang/python/sglang/kernels/jit/csrc/distributed/custom_all_reduce.cuh
template <bool kFence>
static SGL_DEVICE uint32_t sync_enter_pull(const AllReduceParams<kWorldSize>& params) {
  uint32_t current_counter_val = 0;
  if (const auto tx = threadIdx.x; tx < kWorldSize) {          // 前 ws 个 lane 参与
    device::PDLWaitPrimary<kUsePDL>();
    const auto semaphore = &params.pull_semaphores[tx][blockIdx.x];
    const auto counter = semaphore->counter_ptr();
    const auto current = tx == params.rank ? counter->inc(2 * kWorldSize) : 0;  // 只动自家
    current_counter_val = current;
    if constexpr (kFence) semaphore->put_release(); else semaphore->put_relaxed();
    if (tx == params.rank) {                                   // 只有一个 lane 自旋
      while ((kFence ? semaphore->get_acquire() : semaphore->get_relaxed()) - current < kWorldSize);
    }
  }
  __syncthreads();
  return current_counter_val + kWorldSize;                     // 出门自旋的基线
}
```

<p align="center">
  <img src="https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/article-assets-sglang-custom-allreduce-v2/custom_ar_v2_fig4_semaphore.png" width="98%" alt="v2 的 Semaphore 计数协议">
</p>

<p align="center">
  <em><b>Semaphore 协议（world_size=4，rank1 的 block b）。</b>lane 0..3 各向 rank tx 的 flag 发一笔 red.add.sys；只有 lane 1（== 本 rank 编号）自旋等自家 flag 自基线起累计满 4；基线由进门时对自家 counter 一次 atomicAdd(2×ws) 预留。</em>
</p>

和 v1 的 `multi_gpu_barrier` 放在一起看，同与不同都很清楚。

**相同的骨架**：per-block 独立同步（block b 只和各 rank 的 block b 互等，v1 篇讲的"grid 一致"前提和同号 block 闭合的推论原样适用）；每 block 前 ws 个 lane 参与；lane tx 负责与 rank tx 的那条边；barrier 前后的 `__syncthreads()` 承担同样的两个角色（归拢本 block 的写、把可见性转交全 block）。

**写的动作变了**：v1 的 lane 用 `st` 写一个**轮次号数值**到对方槽位；v2 的 lane 用 `red.add.sys` 给对方的 flag **加一**。`red` 是 PTX 的单向归约原子——加法在目标端的内存系统完成，不像 `atomicAdd` 那样要把旧值取回来，一笔纯出站流量，和 v1 的 st 一样便宜，但它是原子的：ws 个 rank 的增量打到同一个 flag 上不会互相丢失。这就是槽位数从 v1 的"每 block × 每 rank 一个"塌缩成"每 block 一个 flag"的原因——v1 必须给每个写者独立槽位以避免非原子写互踩，v2 让硬件把它们加到一起。

**等的动作也变了**：v1 每 block 有 ws 个 lane 各自自旋比对自己槽位的值；v2 每 block 只有 lane == rank 的**一个**线程自旋，条件是 `flag - current < ws`——自基线起累计满 ws 个增量即全员到达。基线 `current` 来自进门时对自家 `m_counter` 的一次 `atomicAdd(2 × kWorldSize)`：一口气预留了本次调用进门 + 出门总共 2ws 个增量的配额，出门自旋直接用 `current + ws` 做基线。flag 单调累加永不清零，回绕由无符号减法自然处理。

**为什么不再需要双相位**：v1 篇用整整一节推演了单槽位死锁——本质是"写数值"会**覆盖**，晚读的 rank 会错过自己在等的值。v2 的增量可交换：快 rank 出门 barrier 的 +1 和慢 rank 还在等的进门 +1 落在同一个 flag 上互不干扰，慢 rank 的判断条件 `flag - current >= ws` 只会被"提前满足"吗？不会——出门的增量确实会提前累进 flag，但慢 rank 进门自旋的阈值恰好是 ws，而在它自己没 put 出门旗之前，flag 至多收到 ws（全员进门）+ (ws-1)（其他人出门）个增量，减去它的进门基线后 ≥ ws 恒成立时全员必已进门。相邻 barrier 的增量混在同一个计数里不损坏语义，靠的是每次调用固定消耗 2ws 个增量、基线由 counter 预留精确对齐。**一个 flag 顶了 v1 的 2 × ws 个槽位。**

fence 的选择延续 v1 判据：进门恒 relaxed（数据写在 kernel 之前，kernel 边界已是系统级可见点）；1shot 出门 relaxed（kernel 内只写了本地输出）；2shot 出门 release/acquire（kernel 内写了 peer 的 workspace，见下节）。

流量上有一个和 v1 方向相反的变化值得指出：v1 bs=1 时一次 barrier 只有 8 lane × 2 block = 16 笔 4 B store；v2 的 grid 固定 96，一次 barrier 是 8 × 96 = 768 笔 red.add——同步消息多了一个量级，但每笔是无回读的单向原子，且 96 个 block 并行发射，不在关键路径上串行累积。

## 1shot_pull：v1 的 1stage 换上新地基

1shot_pull 的骨架与 v1 的 1stage 完全同构：进门 barrier → 读全部 ws 份数据 fp32 归约（固定 0..ws-1 序，位一致）→ 写本地输出 → 出门 barrier。出门 barrier 护的依然是数据源的复用：得等所有 peer 读完，本 rank 的数据源才能被下一轮覆盖。

差异在数据源是三选一的（`PullMode`）：

- **Graph**：`data[i] = params.graph_params[i]`——解引用指针表的一行，直接读各 rank 输入张量的原始地址，零拷贝。这一行的地址在 capture 时录进图里，内容在 capture 结束后由批量 IPC / VMM 交换回填（机制与 v1 的 RankData 占位相同，前篇 0x8 已详述）；
- **Eager**：host 侧先把输入拷进本 rank 的 pull workspace（`data[i] = pull_workspaces[i]`），kernel 读 workspace。进场拷贝在满足对齐且消息不大时（Blackwell ≤ 1 GB、Hopper ≤ 8 MB）用带 PDL 的 `memcpy_kernel` 替代 `cudaMemcpyAsync`：128 线程一个 block，每线程搬一笔最大向量宽度（Blackwell 32 B、Hopper 16 B）；
- **Multicast**：见后文，读一个地址等于让交换机归约 ws 份。

## 2shot_pull：all-gather 融合进写回

v1 的 2stage 是"归约自己的分片 → 带 fence 的中间 barrier → 各自取回全部分片"。v2 的 2shot 把第三步取消了：

```c++
// 来源：sglang/python/sglang/kernels/jit/csrc/distributed/custom_all_reduce.cuh（reduce_impl，kIs2shot=true）
vec_t vec[kWorldSize];
for (uint32_t i = 0; i < kWorldSize; ++i) vec[i].load(data[i], vid);   // 读 ws 份自己分片
const auto out_vec = reduce(vec);
for (uint32_t i = 0; i < kWorldSize; ++i) out_vec.store(data[i], vid); // 写回全部 ws 家同位置
```

<p align="center">
  <img src="https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/article-assets-sglang-custom-allreduce-v2/custom_ar_v2_fig5_2shot.png" width="98%" alt="2shot_pull 的融合写回">
</p>

<p align="center">
  <em><b>2shot_pull（world_size=4，站在 rank1 视角）。</b>rank1 读 4 份 shard1 归约后，把结果写回全部 4 家 workspace 的 shard1 位置——写回本身就是 all-gather。出门 barrier 用 release/acquire，因为 kernel 内发生了远程写。</em>
</p>

每个 rank 归约自己的分片后，把结果**原地写回所有 ws 家 workspace 的同一偏移**。4 个 rank 各写自己那片，出门 barrier 过后每家 workspace 都拼成了完整结果——all-gather 就是这次写回，不存在 stage2，也就不存在 v1 那个带 fence 的中间 barrier。同步点从三个（v1：入口、中间、无出口）变成两个（入口 relaxed、出口 release/acquire），出口的 fence 正是 v1 中间 barrier 的职责搬了家：kernel 内写了 peer 的内存，peer 之后要读，release/acquire 护送这批写。

这个结构在 graph 模式下产生 v2 最漂亮的性质：`data[i]` 来自指针表，指向的就是各 rank 的**输入张量本身**——写回全部 ws 家等于把结果直接写进每个 rank 的输入里。host 侧 `const bool inplace = use_graph && algo == "2shot_pull"; Tensor out = inplace ? in_ : empty_like(in_);`——**out 就是 in，真·原地 all-reduce**，连输出分配都没有。eager 模式则前后各付一次拷贝（进 workspace、出 workspace），共三次 launch。

两个与 v1 不同的小决策也值得记录。其一，**分片余数分给前 rem 个 rank**（`local_num_vecs = avg + (rank < rem)`），v1 是最后一个 rank 吃下全部余数；注释顺带提到 hidden 通常是 1024 的倍数，分片边界大多落在 128 B 对齐上。其二，**指针不再轮转**：v1 的 2stage 用 `(rank+i) % ngpus` 错峰读源，v2 直接按 0..ws-1 原序读。每个分片仍然只有一个 owner，位一致依旧由"单一 owner"保证；错峰的诉求在这一代被放弃了——2shot 接管的消息区间较大、时间上远离纯延迟极限，读源顺序的影响退居次要。

"两个 stage 必须同 tid 映射"的约束呢？随着 stage2 消失，它自动退场了：写回和归约是同一个循环里的同一批线程，不存在跨阶段的映射对齐问题。v1 篇里那段 per-block 同步链的分析，在 v2 里只剩出口 barrier 需要它——写我 workspace 的 peer 线程和我这边确认它写完的 barrier，仍然按同号 block 闭合。

## multicast 档：把归约交给交换机

Hopper/Blackwell 的 NVSwitch 支持 multicast 地址，2shot 的分片归约可以退化成两条指令：

```c++
// 来源：sglang/python/sglang/kernels/jit/csrc/distributed/custom_all_reduce.cuh
asm volatile("multimem.ld_reduce.weak.add.acc::f32.v4.bf16x2 {%0,%1,%2,%3}, [%4];" ...);
// 交换机读全部 ws 份副本、求和后返回；.acc::f32 提升累加精度
asm volatile("multimem.st.weak.v4.f32 [%4], {%0,%1,%2,%3};" ...);
// 对 multicast 地址写一次 = 写到所有 ws 份副本
```

每个 rank 只处理自己的分片：一笔 `ld_reduce` 拿到归约结果，一笔 `multimem.st` 广播写回——fan-in 和 fan-out 都在交换机里完成，SM 只负责发射。位一致依然成立：每个分片一个 owner 做 `ld_reduce`，广播回来的是同一份位型。源码里有一条前人踩过的坑的注释值得抄下来：bf16/fp16 的 `ld_reduce` 结果寄存器仍是 b32（`"=r"`），`.acc::f32` 只提升累加精度、不改变结果寄存器类型，写 `"=f"` 会被 ptxas 以 "Arguments mismatch" 拒绝。

grid 是三个算法里最小的：sm100 ws=8 只有 32 个 block × 固定 512 线程。交换机的归约吞吐吃不下满 grid 的并发请求，"too much traffic will degrade performance in multicast"——这也是它单独一个 `num_mc_blocks` 而不复用 `num_pull_blocks` 的原因。

## PDL：出口自旋藏进后继 kernel 的启动

v1 的 kernel 没有 PDL，自旋期间整条 stream 干等。v2 三个算法统一的放置是：**入口 wait、出口在自旋/收尾之前 trigger**。

- push：`sync_enter_push` 第一句 `PDLWaitPrimary`（等前驱放行后才读输入和相位）；`sync_exit_push` 第一句 `PDLTriggerSecondary`——先放行后继，再做本地的计数器翻转；
- pull：`PDLWaitPrimary` 在入口 barrier 的 lane 分支内；`PDLTriggerSecondary` 在出口 barrier 的最前面——**先放行后继，再进入出口自旋**。此时本 rank 的输出已全部写完（写输出在 trigger 之前的归约循环里），后继 kernel 的 launch 和 prolog 得以与"等慢 rank 出门"的自旋段重叠。出口自旋本身仍然扣住 kernel 不退出，非 PDL 的后继（包括大消息回退的 `cudaMemcpyAsync`）依旧被完整串行化；
- eager 的进出拷贝：`memcpy_kernel` 也带 wait/trigger，让 copy–kernel–copy 三段之间同样能重叠。

这正好回收了 K3 那篇文章的结论：all-reduce 是同步点，出口自旋是 straggler 成本最直接的落点，把它和后继 kernel 的启动重叠是 v2 相对 v1 在延迟上最"免费"的一笔收益。

## 两代机制对照

| 机制点 | v1 | v2 |
|---|---|---|
| grid 策略 | threads 固定 512，block 数 2..36 随消息 | block 数固定（push=SM 数 / pull=96 / mc=32），block size 128..1024 随消息 |
| 小消息同步原语 | Signal 双相位数值握手，ws 个 lane 各自自旋 | push：零 barrier，pos_zero 哨兵 + 本地相位计数 |
| pull 系 barrier | 每 block × 每 rank 一个槽位，写轮次号、比对相等 | 每 block 一个 128 B Semaphore，red.add 计数、单 lane 自旋 |
| 防覆盖 | peer_counter[2] 双相位槽位 | push：数据依赖限速；pull：增量可交换，无需相位 |
| fence 判据 | 同一 kernel 内写→读用 release/acquire | 同一判据；push 数据即旗标，判据退化 |
| 2shot 结构 | reduce-scatter + 中间 fence barrier + all-gather | all-gather 融合进写回，中间 barrier 消失 |
| 原地性 | 恒出参（result 独立分配） | graph 2shot 真原地（out = in） |
| 分片余数 | 最后一个 rank 吃 | 前 rem 个 rank 各多一份 |
| 读源顺序 | 2stage 轮转错峰 | 原序 0..ws-1 |
| 硬件归约 | 无 | multimem.ld_reduce / multimem.st（NVLS） |
| PDL | 无 | 三算法 + eager memcpy 全接，出口 trigger 先于自旋 |
| 实例化 | 预编译 ngpus ∈ {2,4,6,8} | JIT per (dtype, ws, PDL)，ws 2..16 含奇数 |

## 小结

v2 在 kernel 层面的三步棋，每一步都是对 v1 某个具体成本的定点拆除。v1 小消息的主要成本是两轮 barrier 的旗标往返，push 算法把就绪信息编码进载荷位型（+0.0 哨兵 + -0.0 改写），跨 rank 同步的流量和往返彻底归零，防覆盖交给"poll 必须等齐数据"这条天然的依赖链。v1 的 barrier 本身有 ws 个 lane 自旋、2×ws 个槽位的开销，Semaphore 用 red.add 的计数语义把它压缩到每 block 一个 flag、一个自旋 lane——增量可交换，双相位的存在理由随之消失。v1 大消息的 2stage 要三个同步点两段数据搬运，v2 把 all-gather 做成归约结果的多目标写回，同步点减到两个，graph 模式下更是连输出分配和进出拷贝一起省掉，成为真正的原地操作。

代价也在明面上：grid 因为和同步状态绑死而必须固定，小消息下大多数 block 空转陪跑；push 的双相位 buffer 让常驻显存多了 2×ws 份；调优表要按 (架构, world_size) 逐格实测维护。这组交换在 decode 小消息、高频次的负载特征下是划算的——上一代用 36 个 block 和两轮握手做到的事，这一代在最热的档位上用一次写加一次轮询完成。

## 参考资料

- [《SGLang Custom AllReduce v1 线程模型和工作原理细节》](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/blob/master/large-language-model/sglang/SGLang%20Custom%20AllReduce%20v1%20%E7%BA%BF%E7%A8%8B%E6%A8%A1%E5%9E%8B%E5%92%8C%E5%B7%A5%E4%BD%9C%E5%8E%9F%E7%90%86%E7%BB%86%E8%8A%82.md)，本文的前篇，v1 的 grid、Signal、双相位与内存序逐行拆解
- [《SGLang Custom AllReduce v1 与 v2 实现原理详解》](https://zhuanlan.zhihu.com/p/2065205306540531895)，系列首篇，覆盖分发链、graph 指针表注册与调优表全貌
- [sglang `python/sglang/kernels/jit/csrc/distributed/custom_all_reduce.cuh`](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/jit/csrc/distributed/custom_all_reduce.cuh)，v2 三算法 kernel 源码
- [PDL 在 SGLang Kimi K3 中的应用](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/blob/master/large-language-model/sglang/PDL%20%E5%9C%A8%20SGLang%20Kimi%20K3%20%E4%B8%AD%E7%9A%84%E5%BA%94%E7%94%A8.md)，wait/trigger 放置与正确性问题的上下文
