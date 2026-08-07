# SGLang Custom AllReduce v1 线程模型和工作原理细节

上一篇[《SGLang Custom AllReduce v1 与 v2 实现原理详解》](https://zhuanlan.zhihu.com/p/2065205306540531895)把两代实现的结构过了一遍，其中 0x4（Signal 与 barrier）和 0x5（v1 的实现）只写到概念这一层：一组计数器负责跨 rank 同步，算法分为 1stage 和 2stage。至于 kernel 起多少线程、占多少个 SM、每个线程搬哪几个字节、flag 写进哪个槽位，以及 fence 和出口 barrier 为什么这样设计，上一篇没有展开。本文只分析 v1，并把这些问题落到源码上。v2 留到下一篇。

本文引用的代码来自 `sgl-kernel/csrc/allreduce/custom_all_reduce.cuh`（kernel 与 C++ `CustomAllreduce` 类）、`sgl-kernel/csrc/allreduce/custom_all_reduce.cu`（torch binding）、`python/sglang/srt/distributed/device_communicators/custom_all_reduce.py`（Python 侧）。实现从 vLLM v0.8.2 移植，文中忽略 `USE_MUSA` 与 HIP 分支。v1 的启用条件（`SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=0`）、在 all_reduce 分发链上的位置、IPC 交换和 CUDA graph 注册的流程见上一篇的 0x2/0x3/0x5，这里不重复。

**TL;DR**

- **v1 的一次 all-reduce 就是一个 kernel，grid 很小。** threads 固定 512，blocks = min(36, packed 元素数 / 512 向上取整)。bs=1、hidden 8192 的 16 KB 消息只起 2 个 block，也就是说整台 GPU 只有 2 个 SM 在做这次 all-reduce；8 MB 上限消息也只用 36 个 SM（H100 共 132 个）。
- **数据以 16 字节为单位处理。** 每个线程一次 `ld.128` 搬 16 B（bf16 下是 8 个元素），upcast 到 fp32 累加后一次 `st.128` 写出，grid-stride 循环覆盖整段消息。
- **跨 rank 同步是纯自旋 barrier，每 block 只有前 world_size 个线程参与。** lane p 把轮次号写进 rank p 的槽位、自旋等 rank p 的 lane r 写自己的槽位。写 flag 是一笔 4 B 的 NVLink store，自旋轮询只打本地显存。
- **peer_counter 有两组，按轮次号奇偶交替。** 快 rank 可以领先慢 rank 一个 barrier，单槽位会被下一轮的 flag 覆盖导致死锁；barrier 的语义又保证领先不会超过一层，所以两组正好够。
- **内存序的取舍只看一件事：数据写和 flag 写是否在同一个 kernel 里。** 1stage 的数据由 kernel 之前的 copy 或上游 kernel 写入，kernel 完成即系统级可见，flag 用 volatile 就够；2stage 的中转缓冲是本 kernel 刚写的，中间 barrier 必须用 `st.release.sys` / `ld.acquire.sys` 排序。
- **1stage 有出口 barrier、2stage 没有，保护对象不同。** 1stage 的输入缓冲会被 kernel 之外的下一轮 copy 覆盖，必须等所有 peer 读完才能退出；2stage 对中转缓冲的写发生在入口 barrier 之后，天然由下一轮的入口 barrier 挡住。

## 一次调用给 kernel 什么：参数与 grid

先看 host 侧的入口。`CustomAllreduce::allreduce<T>` 把元素数换算成 16 B packed 元素数，再由此定 grid：

```c++
// 来源：sglang/sgl-kernel/csrc/allreduce/custom_all_reduce.cuh（allreduce<T> 节选）
auto d = packed_t<T>::P::size;      // bf16/half: 8，float: 4
if (size % d != 0) throw ...;       // 消息字节数必须是 16 的倍数
size /= d;                          // 换算成 packed 元素数
auto bytes = size * sizeof(typename packed_t<T>::P);
int blocks = std::min(block_limit, (size + threads - 1) / threads);
// threads = kDefaultThreads = 512，block_limit = kDefaultBlockLimit = 36
// Python 侧从不传自定义值，所以 512/36 就是线上配置

#define KL(ngpus, name) name<T, ngpus><<<blocks, threads, 0, stream>>>( \
    ptrs, sg_, self_sg_, output, rank_, size);
```

几个常用消息尺寸代进去：

| 消息（bf16） | packed 元素 | blocks | 参与的 SM | 每线程处理的元素 |
|---|---|---|---|---|
| 16 KB（bs=1，hidden 8192） | 1024 | 2 | 2 | 1 |
| 256 KB（ws=8 的 1stage/2stage 切换点） | 16384 | 32 | 32 | 1 |
| 512 KB | 32768 | 36 | 36 | 1~2 |
| 8 MB（v1 入场上限） | 524288 | 36 | 36 | ~28 |

kernel 签名如下，后面的线程和同步分析都依赖这些参数：

```c++
// 来源：sglang/sgl-kernel/csrc/allreduce/custom_all_reduce.cuh
template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1) cross_device_reduce_1stage(
    RankData* _dp,        // 8 个输入指针（按指针传，为 graph 延迟绑定留的间接层）
    RankSignals sg,       // 8 个 Signal*（按值传，64 B 进 kernel 参数区）
    Signal* self_sg,      // 本 rank 自己的 Signal
    T* __restrict__ result,
    int rank, int size);

struct __align__(16) RankData  { const void* __restrict__ ptrs[8]; };
struct __align__(16) RankSignals { Signal* signals[8]; };
```

`RankSignals` 和 `RankData*` 一个按值一个按指针，不是随手写的。`RankSignals` 里的 8 个 Signal 指针在初始化时就完成了 IPC 交换，之后永不变化，直接放进 kernel 参数区最省事。`RankData` 却必须多绕一层指针：CUDA graph capture 时 peer 的输入地址还没交换到（capture 期间不能做跨 rank 的 host 同步），v1 的做法是先把 `_dp` 指向 `rank_data` 表的一个空位录进图里，capture 结束后再把真实的 8 个 peer 指针回填到那个位置。kernel 里第一件事就是解这层间接：

```c++
auto dp = *_dp;   // 把 64 B 的指针表从显存加载进寄存器，graph replay 时读到的是回填后的真实地址
```

eager 模式不走占位路径：Python 侧先把输入 `cudaMemcpyAsync` 进初始化时注册好的 `buffer_ptrs`，kernel 拿到的 `_dp` 是那次注册的固定条目，代价是多一次进场拷贝。

后面的分析依赖一个契约：all-reduce 是集合通信，所有 rank 以相同的 size 调用，因此**所有 rank 的 grid 配置完全一致**。rank 0 起 2 个 block 时，其它 rank 也一定是 2 个 block。

## 线程模型：16 字节一个单位

kernel 内部不按标量元素分工，而是按 16 B 的 packed 单位：

```c++
// 来源：sglang/sgl-kernel/csrc/allreduce/custom_all_reduce.cuh
template <typename T, int sz>
struct __align__(alignof(T) * sz) array_t { T data[sz]; ... };

template <typename T>
struct packed_t {
  using P = array_t<T, 16 / sizeof(T)>;   // 载荷类型：bf16 下是 8 个元素，16 B
  using A = array_t<float, 16 / sizeof(T)>; // 累加类型：8 个 fp32
};
```

对齐到 16 B 的数组类型让编译器把一次读写生成 `ld.global.128` / `st.global.128`，这是 GPU 单线程一条指令能搬的最大宽度，也是入场校验要求消息字节数是 16 的倍数、v2 进一步要求指针 16 B 对齐的原因。数据遍历是标准的 grid-stride 循环，线程 t 处理第 t、t+stride、t+2·stride… 个 packed 元素，stride = blocks × 512。

<p align="center">
  <img src="https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/article-assets-sglang-custom-allreduce-v1/custom_ar_v1_fig1_grid.png" width="98%" alt="v1 的 grid 配置与线程到数据的映射">
</p>

<p align="center">
  <em><b>v1 的 grid 配置与线程→数据映射。</b>上：16 KB 消息只起 2 个 block，线程 t 对应元素 t，一人一个。下：8 MB 消息 36 个 block 封顶，同一线程按 stride=18432 跳着处理约 28 个元素。</em>
</p>

上半部分对应 decode bs=1：这次 all-reduce 只有 1024 个线程，占用 2 个 SM。此时访存和计算都不构成瓶颈，时间主要花在同步上，后面几节会具体分析。下半部分是 8 MB 消息的情况：36 个 block 是硬上限（`kMaxBlocks`），更大的消息只会让每个线程多循环几轮，不会增加参与的 SM。36 的来历放在倒数第二节讲。

## Signal：每个字节归谁写、归谁读

跨 rank 同步的全部状态都在 `Signal` 结构里，每个 rank 在初始化时分配一份并把地址 IPC 交换给所有 peer：

```c++
// 来源：sglang/sgl-kernel/csrc/allreduce/custom_all_reduce.cuh
using FlagType = uint32_t;
struct Signal {
  alignas(128) FlagType self_counter[kMaxBlocks][8];      // 36×8×4B = 1152 B
  alignas(128) FlagType peer_counter[2][kMaxBlocks][8];   // 2×36×8×4B = 2304 B
};
// sizeof(Signal) = 3456 B，torch binding 里的 meta_size() 返回的就是它
```

两个数组的第二维都是 8。`kMaxBlocks` 个 block × 最多 8 个 rank，每个 (block, lane) 组合一个 4 B 槽位。这也解释了 36 为什么是硬上限而不只是调优值：同步槽位就分配了这么多。

`meta_ptrs` 那块 IPC 内存的布局是 Signal 打头、中转缓冲随后，`get_tmp_buf()` 直接取 Signal 尾后地址：

```c++
template <typename P>
DINLINE P* get_tmp_buf(Signal* sg) { return (P*)(((Signal*)sg) + 1); }
```

<p align="center">
  <img src="https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/article-assets-sglang-custom-allreduce-v1/custom_ar_v1_fig2_signal.png" width="98%" alt="Signal 内存布局与 barrier 握手配对">
</p>

<p align="center">
  <em><b>Signal 的内存布局与握手配对（world_size=4）。</b>上：meta 区 = 3456 B 的 Signal + max_size 的中转缓冲，全部 128 B 对齐。下：rank1 的 lane 2 与 rank2 的 lane 1 成对握手——各自把轮次号写进对方的槽位，再自旋读自己的槽位。</em>
</p>

两个计数器数组的读写所有权不同：

- `self_counter[b][t]`：**只有本 rank 读写**。它是 (block b, lane t) 这一组握手的私有轮次计数。语义上整个 rank 一个计数器就够（所有 barrier 都让它加一），做成每 block × 每 lane 一份是为了让自增留在寄存器和本地 L2 里，不用经过 shared memory 广播——注释原话是 "we use multiple per block to eliminate the need to share the counter via smem"。
- `peer_counter[phase][b][p]`：**由 rank p 跨 NVLink 写入，本 rank 自旋读取**。写它的具体是 rank p 上 block b 的 lane r（r 是本 rank 的编号）。

握手配对是对称的：**rank r 的 lane p，和 rank p 的 lane r，互为对方的写者**。每个 block 内 world_size 个 lane 各管一条边，ws=8 时 8 个线程把本 rank 与全部 8 个 rank（含自己）的握手做完。

flag 是**推**到 peer 的，等待则是在本地自旋。把一次 barrier 的写流量算个账：每个 block 里 ws 个 lane 各发一笔 4 B 的轮次号 store，而 barrier 是每个 block 独立做一遍的，所以一个 rank 一次 barrier 总共发 ws × blocks 笔。代入 16 KB 消息的例子：2 个 block、ws=8，就是 8 × 2 = 16 笔、合计 64 B。其中每个 block 有一笔是 lane r 写自己的 Signal、走本地，真正跨 NVLink 的只有 14 笔。1stage 一次调用有入口、出口两个 barrier，flag 流量也就百来字节，和 16 KB 的数据搬运相比是零头。自旋轮询读的全是自己显存里的槽位，不占 NVLink。慢 rank 忙等再久也不会刷出跨设备流量，barrier 的成本几乎全部是等待时间本身。

## multi_gpu_barrier 逐行

barrier 本体不长，所有点都在这十几行里：

```c++
// 来源：sglang/sgl-kernel/csrc/allreduce/custom_all_reduce.cuh
template <int ngpus, bool is_start, bool need_fence = false>
DINLINE void multi_gpu_barrier(const RankSignals& sg, Signal* self_sg, int rank) {
  if constexpr (!is_start) __syncthreads();          // 出口/中间 barrier：先等全 block 干完活
  static_assert(!(is_start && need_fence));          // 入口 barrier 不可能需要 fence
  if (threadIdx.x < ngpus) {                         // 每 block 只有前 ws 个 lane 参与
    auto val = self_sg->self_counter[blockIdx.x][threadIdx.x] += 1;   // 轮次号自增
    auto peer_counter_ptr =                          // 我要写的：rank tx 的槽位 [相位][b][我]
        &sg.signals[threadIdx.x]->peer_counter[val % 2][blockIdx.x][rank];
    auto self_counter_ptr =                          // 我要等的：自己的槽位 [相位][b][tx]
        &self_sg->peer_counter[val % 2][blockIdx.x][threadIdx.x];
    if constexpr (need_fence) {
      st_flag_release(peer_counter_ptr, val);        // st.release.sys.global.u32
      while (ld_flag_acquire(self_counter_ptr) != val);  // ld.acquire.sys.global.u32
    } else {
      st_flag_volatile(peer_counter_ptr, val);       // st.volatile.global.u32
      while (ld_flag_volatile(self_counter_ptr) != val); // ld.volatile.global.u32
    }
  }
  if constexpr (is_start || need_fence) __syncthreads();  // 把"全员到齐"传达给整个 block
}
```

按执行顺序把每一行的机制拆开看。

**① `if constexpr (!is_start) __syncthreads();`——前置同步。** 出口和中间 barrier 在握手之前先做一次 block 内同步。`__syncthreads()` 不只是"等人"，它同时是一个 block 范围的内存栅栏：过了它，本 block 所有线程在它之前的读写都已完成、且互相可见。对出口 barrier，这保证 512 个线程都读完了 peer 数据，握手 lane 才有资格替整个 block 对外宣布"block b 干完了"；对中间 barrier，这保证 stage1 写进中转缓冲的数据都已落定，接下来的 release store 才有东西可护送。

**② `auto val = self_sg->self_counter[blockIdx.x][threadIdx.x] += 1;`——轮次号自增。** 这是普通的读改写，不是原子操作，也不需要是：这个计数器只有本 rank 上 (block b, lane t) 这一个线程会碰，无人竞争。每次 barrier 所有握手 lane 各自加一，所以同一 block 里 ws 个计数器永远同步递增，val 的含义就是"这是本 block 经历的第几次 barrier"。它决定两件事：写给 peer 的 flag 值，以及 `val % 2` 选哪组相位槽位。

**③ 两个指针：写谁、等谁。** `peer_counter_ptr` 指向 rank tx 的 Signal——`sg.signals[threadIdx.x]` 是初始化时 IPC 映射进本进程地址空间的 peer 指针，对它 store，硬件把这笔写经 NVLink 路由到对端显存，在 kernel 代码层面和写本地内存没有任何区别。`self_counter_ptr` 则指向自己 Signal 里的槽位。这个变量名有点误导：它指的不是 `self_counter` 数组，而是**自己 Signal 里的 `peer_counter` 槽位**——"self"说的是这份 Signal 归自己所有，往里写的人是 rank tx 的 lane r。

**④ volatile 这一对：`st_flag_volatile(peer_counter_ptr, val)` 与 `while (ld_flag_volatile(self_counter_ptr) != val);`。** volatile 解决的是"这笔访存必须真的发生"。没有它，编译器完全有权把自旋循环里的 load 提出循环——读一次进寄存器，然后对着寄存器空转，永远看不到 peer 后来写入的值；store 侧同理可能被合并或推迟。`st.volatile` / `ld.volatile` 强制每次访问都走到内存系统（绕过 L1，落在 L2 这个一致性点上），但**不**对周围其它内存操作施加任何顺序约束——这既是它比 release/acquire 便宜的原因，也是它只能用在"flag 本身就是全部载荷"场景的原因。自旋条件写 `!= val` 而不是 `>= val` 也有讲究：轮次号是会回绕的 uint32，大小比较在回绕点出错，相等比较配合双相位则永远正确（同一槽位里的残留旧值只能是 val−2，不可能撞上 val）。

**⑤ release/acquire 这一对：`st_flag_release(peer_counter_ptr, val)` 与 `while (ld_flag_acquire(self_counter_ptr) != val);`。** `st.release.sys.global.u32` 在写 flag 之外附带一个单向承诺：本线程在这条指令之前的所有内存写，对任何 acquire 到这个 flag 的观察者，都先于 flag 可见。`ld.acquire.sys` 是对称的另一半：它之后的读写不会被调度提前到它前面。两者配对就在两张 GPU 之间建立了一条 happens-before 边——peer 读到 val，就一定能读到 val 之前写好的中转数据。`.sys` scope 是关键：`.gpu` scope 的排序只覆盖本卡内部，观察者在另一张 GPU 上就必须用系统范围。①的前置 `__syncthreads()` 在这里再次出场：中转数据是全 block 512 个线程写的，而 release 只护送"本线程之前"的写，得靠 block 栅栏先把全 block 的写归拢到握手 lane 的"之前"，release 才护得全。sm70 之前没有这组带内存序的指令，代码里的老路径用 `membar.sys` + `st.volatile` 手工拼出同样的语义。

**⑥ `if constexpr (is_start || need_fence) __syncthreads();`——后置同步。** 它有两个身份。执行上：lane ws..511 不参与握手，没有这道栅栏它们会径直冲进归约循环、在 peer 数据就绪前开始读——后置同步把整个 block 扣在原地，直到握手 lane 自旋归来。可见性上（need_fence 时）：acquire 只发生在握手 lane 身上，`__syncthreads()` 作为 block 级栅栏把 acquire 拿到的可见性转交给全 block，之后任何线程去读中转缓冲都是安全的。出口 barrier（`<false, false>`）两个条件都不满足，这道栅栏省略——握手完 kernel 就退出，没有下文需要保护。

两个模板布尔组合出 v1 实际用到的三种形态，对照上面的逐行归纳成一张表：

| 形态 | 使用位置 | 前置 `__syncthreads` | 后置 `__syncthreads` | flag 语义 |
|---|---|---|---|---|
| `<true, false>` 入口 | 1stage/2stage 开头 | 无 | 有 | volatile |
| `<false, false>` 出口 | 1stage 结尾 | 有 | 无 | volatile |
| `<false, true>` 中间 | 2stage 两阶段之间 | 有 | 有 | release/acquire |

对应关系是：入口 barrier 刚进 kernel，没有需要先完成的工作，省前置；出口 barrier 之后 kernel 直接退出，省后置；中间 barrier 前有 stage1 的写、后有 stage2 的读，两头都不能省。

barrier 是 **per-block** 的。rank r 的 block b 只和所有 rank 的 block b 互等。block 0 已经过了 barrier 时，block 5 可能还没到，不同 block 之间没有任何同步关系。

这套逐 block 的握手要成立，前提是"所有 rank 的 grid 一致"，原因有两层。第一层是**能不能等到人**：block b 的握手 lane 自旋等的是"每个 rank 的 block b"写来的轮次号，假如某个 rank 按自己的消息尺寸只起了 2 个 block，而别的 rank 起了 4 个，那么后者的 block 2、block 3 等的 flag 永远不会有人写，自旋直接挂死。第二层是**等对了人没有**：barrier 过了只能证明"各 rank 的 block b 都到了这一步"，而每个 block 摸哪些数据是由 grid-stride 映射决定的——grid 一致时 stride 一致，元素 idx 在每个 rank 上都落在同一个 block 编号手里，"和 peer 的同号 block 互等"恰好把自己要读写的那批元素的写者全部等到；grid 不一致则映射错位，同号 block 等到的可能根本不是自己数据的写者。好在这个前提不需要额外机制来保证：all-reduce 是集合调用，所有 rank 传入相同的 size 和 dtype，代进 blocks 公式得到的 grid 自然一模一样。它的另一个推论（同步链只在同号 block 之间闭合）会在 2stage 一节里变成关键约束。

## peer_counter 为什么是两组

`Signal` 里 `peer_counter` 的第一维是 2，源码注释解释了原因：peer 的 block 可能已经到达第二个同步点、开始写 counter+1，而本 rank 的 block 还在第一个同步点等 counter。展开成具体的执行序列更容易看清：

<p align="center">
  <img src="https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/article-assets-sglang-custom-allreduce-v1/custom_ar_v1_fig3_phase.png" width="98%" alt="peer_counter 双相位的必要性">
</p>

<p align="center">
  <em><b>双相位的必要性。</b>上：假设只有一组槽位，快 rank 在 barrier#2 把慢 rank 还没读到的轮次号 1 覆盖成 2，慢 rank 的自旋永不返回。下：奇数轮次走 peer_counter[1]、偶数走 peer_counter[0]，相邻两个 barrier 落在不同槽位上。</em>
</p>

死锁链条是这样的：barrier 只等 flag、不等消费确认，rank A 写完 flag 进入自旋，只要它等的人都到了它就走人——它不知道自己写给 rank B 的 flag 有没有被 B 读到。A 通过 barrier#1 后做完归约到达 barrier#2，再次向 B 的同一槽位写入 val=2；如果 B 这时还没在 barrier#1 里读到 val=1（它在等第三个 rank，或单纯被调度慢了），槽位里的 1 就没了，B 的 `while (slot != 1)` 永远等不到。

那为什么两组就封顶了？barrier 有一个不变式：**我通过 barrier k，就证明所有 rank 都到达了 barrier k**。到达意味着它们已经写完 barrier k 的 flag，也必然消费完了 barrier k-1 的 flag。所以想到达 barrier k+2，先得通过 barrier k+1，而通过 k+1 时全员对 barrier k 的读写都已了结。任何时刻"在飞"的 barrier 至多是相邻的 k 和 k+1 两个，`val % 2` 交替正好错开。

轮次号是 `uint32_t`，注释专门说明了溢出无害：无符号回绕是良定义行为，而同一个槽位里可能残留的旧值恒为 val−2（同相位的上上轮），拿 `!= val` 做自旋条件不会误命中。按 decode 每步几百次 all-reduce、每次两个 barrier 算，跑到回绕大约要几天，回绕后一切照旧。

## 内存序：判据只有一条

barrier 里两套 flag 读写的 PTX 只差一个内存序修饰：

```c++
// 来源：sglang/sgl-kernel/csrc/allreduce/custom_all_reduce.cuh（sm70+ 路径）
static DINLINE void st_flag_release(FlagType* flag_addr, FlagType flag) {
  asm volatile("st.release.sys.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
}
static DINLINE FlagType ld_flag_acquire(FlagType* flag_addr) {
  FlagType flag;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
  return flag;
}
static DINLINE void st_flag_volatile(FlagType* flag_addr, FlagType flag) {
  asm volatile("st.volatile.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
}
static DINLINE FlagType ld_flag_volatile(FlagType* flag_addr) {
  FlagType flag;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
  return flag;
}
```

上一节已经讲过这四个函数：volatile 只保证访存发生，release/acquire 才建立跨设备的先后关系。是否需要付 release/acquire 的开销，只取决于一件事：**数据写和 flag 写是否在同一个 kernel 里**。

- **不在同一个 kernel（volatile 够用）。** 1stage 读的输入数据由 kernel 之前的 `cudaMemcpyAsync` 或上游 producer kernel 写入。CUDA 保证一个 kernel 或拷贝完成后，它的写入对系统可见（host 同步后能读到正确数据靠的就是这条）；peer 的 AR kernel 在自己的 stream 上排在数据写入之后，所以它开始执行、写出入口 flag 时，它的输入早已在它的 L2/HBM 里就位，NVLink 远端读直接命中。flag 此时只承担"我到了"的计数功能，先后关系不需要它维护。
- **在同一个 kernel（必须 release/acquire）。** 2stage 的 stage1 刚把归约结果写进本 rank 的中转缓冲，紧接着就在中间 barrier 里写 flag，peer 过了 barrier 立刻来读。数据写和 flag 写出自同一个 kernel 的执行流，没有 kernel 边界替它们排序，`st.volatile` 完全可能越过还在写回路上的数据先到达远端。这里 `st.release.sys` 保证中转数据先行可见，`ld.acquire.sys` 保证 barrier 之后的读不被提前。

## 1stage：kernel 全文与时间线

1stage 的 kernel 体非常短，全文如下：

```c++
// 来源：sglang/sgl-kernel/csrc/allreduce/custom_all_reduce.cuh
template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1) cross_device_reduce_1stage(
    RankData* _dp, RankSignals sg, Signal* self_sg, T* __restrict__ result, int rank, int size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  // note: we don't reorder the address so the accumulation order is the same
  // for all ranks, ensuring bitwise identical results
  auto dp = *_dp;
  multi_gpu_barrier<ngpus, true>(sg, self_sg, rank);                    // 入口
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    ((P*)result)[idx] = packed_reduce<P, ngpus, A>((const P**)&dp.ptrs[0], idx);
  }
  multi_gpu_barrier<ngpus, false>(sg, self_sg, rank);                   // 出口
}

template <typename P, int ngpus, typename A>
DINLINE P packed_reduce(const P* ptrs[], int idx) {
  A tmp = upcast(ptrs[0][idx]);                 // 先取 rank0 的 16 B，升到 fp32
#pragma unroll
  for (int i = 1; i < ngpus; i++) {
    packed_assign_add(tmp, upcast(ptrs[i][idx])); // 固定按 rank 1..ws-1 顺序累加
  }
  return downcast<P>(tmp);                      // 最后一次性降回 bf16 写出
}
```

每个线程对它负责的每个 packed 元素做的事：向 8 个 rank 的同一偏移各发一笔 `ld.128`（其中 7 笔走 NVLink 到 peer 显存），upcast 成 8 组 fp32 累加，downcast 后一笔本地 `st.128`。`#pragma unroll` 把 8 笔远程读全部展开，它们之间没有数据依赖，可以同时在飞，用 MLP（memory-level parallelism）摊薄单笔远程读的高延迟。fp32 累加避免了 bf16 逐次舍入的误差累积。累加顺序又固定从 rank 0 到 rank 7，所有 rank 的浮点舍入路径完全相同，因此输出位一致。kernel 顶上的注释说的就是这个契约。

流量上，1stage 每 rank 读 ws×N 字节（(ws−1)/ws 是远程）、写 N 字节本地，读放大 ws 倍。16 KB 消息 ws=8 也就 128 KB 读流量，NVLink 毫无压力；这个算法输在大消息，赢在只有两轮同步。

<p align="center">
  <img src="https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/article-assets-sglang-custom-allreduce-v1/custom_ar_v1_fig4_1stage_timeline.png" width="98%" alt="1stage 的一次执行时间线">
</p>

<p align="center">
  <em><b>1stage 的一次执行（world_size=4，rank2 上游慢）。</b>入口 barrier 把所有 rank 卡到最慢者到达；中段是归约本体；出口 barrier 保证全员读完才允许任何 rank 的 kernel 退出。</em>
</p>

入口 barrier 的自旋段长短不一：rank2 上游慢，其它三个 rank 就都在自旋里等它。all-reduce 的 kernel 时长由最慢 rank 决定，profile 里这个 kernel 显得"慢"时，多半是 straggler 在别处。

出口 barrier 保护的对象在 kernel **外面**：本 rank 的 kernel 一退出，stream 上排队的下一轮 `cudaMemcpyAsync` 就会开始覆盖输入缓冲，而此刻某个慢 peer 可能还在读这块缓冲的尾巴。出口 barrier 把"kernel 退出"推迟到所有 peer 读完之后。它不需要 fence，因为没有数据需要排序，只需要确认所有 peer 都已到达。

出口 barrier 的保护以 block 为粒度闭合：所有 rank 用同一套 grid-stride 映射，元素 idx 在每个 rank 上都归 block f(idx) 处理，我的 block b 挡住的就是所有 peer 的 block b 对同一批元素的读取。全部 36 个（或 2 个）block 各自过完出口 barrier，kernel 才算结束，从而覆盖全部元素。

## 2stage：分片、轮转、以及为什么没有出口 barrier

2stage 是 reduce-scatter + all-gather，kernel 全文：

```c++
// 来源：sglang/sgl-kernel/csrc/allreduce/custom_all_reduce.cuh
template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1) cross_device_reduce_2stage(
    RankData* _dp, RankSignals sg, Signal* self_sg, T* __restrict__ result, int rank, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  int part = size / ngpus;
  int start = rank * part;
  int end = rank == ngpus - 1 ? size : start + part;   // 最后一个 rank 吃余数
  int largest_part = part + size % ngpus;
  const P* ptrs[ngpus];
  P* tmps[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; i++) {
    int target = (rank + i) % ngpus;                   // 从自己开始轮转
    ptrs[i] = (const P*)_dp->ptrs[target];
    tmps[i] = get_tmp_buf<P>(sg.signals[target]);
  }
  auto tmp_out = tmps[0];                              // tmps[0] 恒为自己的中转缓冲
  multi_gpu_barrier<ngpus, true>(sg, self_sg, rank);   // 入口

  // stage 1: reduce scatter
  for (int idx = start + tid; idx < end; idx += stride) {
    tmp_out[idx - start] = packed_reduce<P, ngpus, A>(ptrs, idx);
  }
  multi_gpu_barrier<ngpus, false, true>(sg, self_sg, rank);   // 中间，need_fence

  // stage 2: allgather。注释：两个 stage 必须用相同的 tid 映射，
  // 跨设备可见性只在相同 tid 的线程之间由 barrier 保证
  for (int idx = tid; idx < largest_part; idx += stride) {
#pragma unroll
    for (int i = 0; i < ngpus; i++) {
      int gather_from_rank = ((rank + i) % ngpus);
      if (gather_from_rank == ngpus - 1 || idx < part) {
        int dst_idx = gather_from_rank * part + idx;
        ((P*)result)[dst_idx] = tmps[i][idx];
      }
    }
  }
  // 没有出口 barrier
}
```

<p align="center">
  <img src="https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/article-assets-sglang-custom-allreduce-v1/custom_ar_v1_fig5_2stage.png" width="98%" alt="2stage 的数据流">
</p>

<p align="center">
  <em><b>2stage 的数据流（world_size=4，站在 rank1 视角）。</b>stage1：rank1 从 4 家读 shard1 归约进自己的中转缓冲，读序从自己开始轮转；带 fence 的中间 barrier 之后，stage2 从 4 家中转缓冲取回分片拼出完整输出。</em>
</p>

**指针轮转用于错开流量，同时保持位一致。** `ptrs[i] = (rank+i) % ngpus` 让 rank 1 按 1→2→3→0 的顺序读、rank 2 按 2→3→0→1 读，每个 rank 从自己开始转一圈。如果所有 rank 都按 0→1→2→3 的固定顺序读，stage1 一开始 8 个 rank 会同时对 rank0 的显存发起读，NVLink 上出现 incast 热点；轮转把同一时刻的读均匀摊到 8 个源上。这里要澄清上一篇 0xa.1 的一个笼统说法（"两个 kernel 都不按距离重排指针数组"）：**1stage 确实全员同序，2stage 是轮转的**。轮转不会破坏位一致，因为 2stage 里每个分片只由一个 rank 归约。shard s 的累加顺序由 rank s 决定（s→s+1→…→s−1），算完后所有 rank 取回的都是同一份位型，不存在"两个 rank 各算一遍要求结果相同"的问题。位一致的保证方式从"全员同序"换成了"单一 owner"。

**两个 stage 使用相同的 tid 映射，是因为 barrier 按 block 闭合。** stage1 里线程 tid 写 `tmp_out[j]`，j ≡ tid (mod stride)；stage2 里读 `tmps[i][j]` 的也是满足 j ≡ tid (mod stride) 的同一个 tid。看起来像是多余的讲究——反正中间隔着 barrier——但回想 barrier 是 per-block 的：rank B 的 block 3 过了中间 barrier，只证明**所有 rank 的 block 3** 完成了 stage1，rank A 的 block 5 可能还在写它负责的那段中转数据。所有 rank 共用同一套 tid→偏移映射时，中转缓冲偏移 j 的写者（peer 的某个 block）和读者（本 rank 的同号 block）恰好是 barrier 同步过的一对；映射一旦错位，读者可能落到一个自己的 barrier 管不着的 block 头上，release/acquire 链条就断了。源码把它写成显式契约："If thread i computes the sum of start + i in the first stage, then thread i also gathers start + i from all ranks"。

**grid 按全量 size 定，多数 block 只跑 barrier。** blocks 的公式对两个 kernel 是同一个，但 2stage 每个 rank 只处理 1/ws 的分片。代一组数：512 KB、ws=8 时 size=32768、blocks=36、part=4096，stage1 的循环条件 `start + tid < end` 只让 tid < 4096 的线程（block 0..7）碰数据，stage2 同样只有前 4096 个线程干活——36 个 block 里 28 个从头到尾只参与三次握手。这是从 vLLM 原样移植的行为，小消息下多余 block 的代价只是多几笔 4 B flag。

**没有出口 barrier，但每块内存都有人守。** 对照 1stage 想一下哪里不同。2stage 的 kernel 涉及三类缓冲：peer 的输入区、各家的中转缓冲、本地输出。输入区在 stage1 之后就没人再读，而"全员过了中间 barrier"本身就证明全员 stage1 结束——中间 barrier 顺手把 1stage 出口 barrier 的活干了。中转缓冲呢？stage2 还在读它，但下一轮对它的写发生在**下一轮 kernel 的入口 barrier 之后**（stage1 在入口 barrier 后面）：慢 rank 还在本轮 stage2 时不会到达下一轮入口 barrier，快 rank 就算把下一个 kernel 都启动了，也会被那个入口 barrier 拦在写 tmp 之前。1stage 必须要出口 barrier，原因是它的输入区由 kernel **外面**的 copy 写入。那次 copy 不受任何 barrier 管辖，只受"前一个 kernel 退出"约束，所以退出这件事本身必须被同步。

## 多少个 SM 在干活

`__launch_bounds__(512, 1)` 告诉 ptxas 这个 kernel 每 block 最多 512 线程、期望每 SM 至少驻留 1 个 block，寄存器分配按这个目标做。grid 从不超过 36 个 block，而 A100 有 108 个 SM、H100 有 132 个，硬件调度器会把这些 block 摊到不同的 SM 上。因此"参与的 SM 数"就等于 block 数，v1 的 all-reduce 最多动用 36 个 SM，bs=1 时只动用 2 个。

36 的来历写在 `allreduce<T>` 的注释里，作者是 vLLM custom allreduce 的原作者 hanzhi713：

> Block and grid default configs are results after careful grid search. Using 36 blocks give the best or close to the best runtime on the devices I tried: A100, A10, A30, T4, V100. You'll notice that NCCL kernels also only take a small amount of SMs. Not quite sure the underlying reason, but my guess is that too many SMs will cause contention on NVLink bus.

翻译过来：36 是在 A100/A10/A30/T4/V100 上网格搜索出的最优或接近最优值；NCCL 的 kernel 同样只占少量 SM；猜测的机理是更多 SM 参与会加剧 NVLink 总线争用。这也符合它的负载特征：小消息下瓶颈是延迟而非带宽，几十个 SM 的 LSU 足以把在飞的远程读填满，再添 SM 只是往同一条 NVLink 上塞更多并发请求，还会增加 barrier 的握手规模（槽位数 = blocks × ws）。这个值是 Ampere 一代之前的实测，代码里也留了一句 TODO，说 A100 和 H100 的阈值应该分开调。v2 则把 block 数做成按 (架构, world_size) 查表。

对 decode 关键路径来说，bs=1 时一次 all-reduce 只占 2 个 SM，剩下 130 个 SM 在这段时间里是空的。但它仍会在同一条 stream 上串行阻塞后续 kernel，所以 K3 那篇文章才说"在 all-reduce 上省一微秒，一比一转化为 step 时间"。v1 的 kernel 没有接 PDL，同步自旋期间也没法帮后继 kernel 做任何事，这是 v2 的一个改进方向。

## 算法切换与入场上限

两个 kernel 之间的选择写死在 host 侧的 `REDUCE_CASE` 宏里：ws=2 恒用 1stage；full NVLink 时 ws≤4 且消息 <512 KB、或 ws≤8 且 <256 KB 用 1stage，其余用 2stage。`SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage|2stage` 可以强制指定，配合 `nsys` 对拍两个算法很方便。模板按 ngpus ∈ {2, 4, 6, 8} 实例化，奇数 world_size 在 `init_custom_ar` 里直接拒绝。

顺带一个防御性细节：`REDUCE_CASE` 里 ws>2 且非 full NVLink 的组合没有任何分支，既不 launch kernel 也不报错，输出缓冲区将保持未初始化。这个组合依赖 Python 侧 `should_custom_ar` 提前挡掉（它要求 ws==2 或 full_nvlink），C++ 侧没有兜底。这是一层由 Python 侧保证的入场契约。

消息上限 8 MB（`_MAX_CAR_SIZE`）由 Python 侧把关，超过的消息在分发链上交回 pynccl/NCCL；中转缓冲和预注册输入缓冲的尺寸也都按这个值分配，这是上一篇 0x5.1 讲过的存储布局，不再展开。

## 小结

把 v1 压缩成一句话：一个最多 36 个 block 的 kernel，用「每 block 前 ws 个线程的对称自旋握手」实现跨 rank barrier，barrier 之间夹一段 16 B 粒度、fp32 累加的 grid-stride 归约。内存序取决于数据写和 flag 写的位置：不在同一个 kernel 里就用 volatile，在同一个 kernel 里就用 release/acquire。1stage 的两个 barrier 都是前者，2stage 的中间 barrier 是后者。缓冲区的生命周期则决定是否需要出口 barrier：kernel 外的 copy 不受 barrier 管，必须同步 kernel 退出这个事件；2stage 的中转缓冲在入口 barrier 之后写入，可以由下一轮入口 barrier 保护。

v1 有三处明显限制：自旋 barrier 期间 SM 干等，也没有 PDL 放行后继 kernel；1stage/2stage 的切换阈值和 36 块的 grid 是 Ampere 之前的实测值；graph 注册仍是逐 buffer 的 IPC 握手。v2 分别改用 push 算法、接入 PDL，并使用集中式指针表。下一篇再分析 v2 的三个算法和 Semaphore。

## 参考资料

- [《SGLang Custom AllReduce v1 与 v2 实现原理详解》](https://zhuanlan.zhihu.com/p/2065205306540531895)，本文的前篇，覆盖分发链、IPC 交换、CUDA graph 注册与 v2 结构
- [sglang `sgl-kernel/csrc/allreduce/custom_all_reduce.cuh`](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/allreduce/custom_all_reduce.cuh)，v1 kernel 源码
- [PDL 在 SGLang Kimi K3 中的应用](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/blob/master/large-language-model/sglang/PDL%20%E5%9C%A8%20SGLang%20Kimi%20K3%20%E4%B8%AD%E7%9A%84%E5%BA%94%E7%94%A8.md)，通信 kernel 与 PDL 配合的上下文
