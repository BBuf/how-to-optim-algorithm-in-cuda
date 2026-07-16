# 0x0. 前言

这篇文档逐段解析腾讯 HPC-Ops 里的 `gemm_bf16xfp32` kernel，源码在：https://github.com/Tencent/hpc-ops/blob/main/src/gemm/sm90/gemm_bf16xfp32.cu ，一共 557 行。这个 kernel 做的事情是 `y[m,n] = x[m,k](bf16) @ w[n,k]^T(fp32)`，把 fp32 权重拆成高低两个 bf16 走 Tensor Core，最初是在 H20 上给 Hunyuan3 的 MoE router 做加速用的，我们最近把它拷进了 SGLang（PR https://github.com/sgl-project/sglang/pull/30247 ），优化背景和 H200 调参过程见另一篇《HPC-Ops gemm_bf16xfp32 Kernel笔记》，这篇只讲 kernel 源码本身。

这个 kernel 是一个很好的 CuTe/Hopper 学习样本，因为它只有 557 行，但把 Hopper 上手写高性能 GEMM 的全套东西都用上了：TMA、GMMA、mbarrier 流水线、warp specialization、寄存器重分配、persistent kernel、block swizzle、单 kernel split-k。比 CUTLASS 3.x 的 CollectiveMma 那套模板好读得多，适合作为看懂 CUTLASS 之前的过渡材料。

阅读前建议先过一遍本仓库里的这几篇：

- `cutlass/cute/cute笔记-reed.md`：CuTe 的 Layout/Tensor/MMA 抽象，Ampere 上的实现
- `cutlass/wgmma/Tutorial: 在Hopper GPU上使用WGMMA的快速矩阵乘.md`：GMMA 指令
- `cutlass/tma/CUTLASS Tutorial: Mastering the NVIDIA® Tensor Memory Accelerator (TMA).md`：TMA

没看过也没关系，0x1 节会把这个 kernel 用到的概念都过一遍，只是会比较浓缩。

# 0x1. 预备知识

## 1.1 Hopper 的三个硬件机制：TMA、GMMA、mbarrier

这个 kernel 的整个结构都是围绕 Hopper（SM90）的三个硬件机制组织的，先把它们说清楚。

**TMA（Tensor Memory Accelerator）** 是一个独立于 SM 计算单元的 DMA 引擎，负责 global memory 和 shared memory 之间的搬运。CUDA 线程只需要一个线程发出一条指令（`cp.async.bulk.tensor`），告诉 TMA "把第 (i, j) 个 tile 搬到 smem 这个地址"，之后搬运完全异步进行，不占用任何线程的发射带宽和寄存器。搬运的形状、stride、swizzle 模式都提前编码在一个 128 字节的 TMA descriptor 里，descriptor 在 host 侧创建，通过 kernel 参数传进来。对比 Ampere 时代的 `cp.async`：那时每个线程搬 16 字节，一个 tile 要全 warp 算地址、发几十条指令；TMA 时代一条指令搬一个 tile，地址计算全部下沉到硬件。

**GMMA（wgmma.mma_async）** 是 Hopper 的矩阵乘指令。和 Ampere 的 `mma.sync`（一个 warp 32 线程协作算 16x8x16）不同，GMMA 以 warpgroup 为单位（4 个连续的 warp，128 线程）算一个大得多的矩阵块，比如 `m64n64k16`、`m64n256k16`。更重要的两个变化：一是操作数可以直接来自 shared memory（SS 模式，A、B 都在 smem，通过 matrix descriptor 描述位置和 swizzle，不需要先 load 到寄存器）；二是指令异步执行，发射之后线程可以继续干别的，通过 `wgmma.commit_group` / `wgmma.wait_group` 来同步。这个 kernel 用的就是 SS 模式。

**mbarrier** 是 smem 里的一个 64 位同步对象，是 Hopper 上生产者-消费者流水线的粘合剂。它支持：

- `mbarrier.init`：初始化，设定需要多少次 arrive 才算完成一个 phase；
- `mbarrier.arrive`：到达一次；
- `mbarrier.arrive.expect_tx`：告诉 barrier 接下来预期有 N 字节的 TMA 数据到达。TMA 硬件每写完一部分数据就自动向 barrier 报数，字节数攒够 N，这个 phase 就完成。这是 TMA 和 mbarrier 的联动机制，让"数据到齐"这件事完全不需要线程参与；
- `mbarrier.try_wait.parity`：带相位（phase）的等待。barrier 每完成一轮 arrive 就翻转一次相位（0→1→0→...），等待方拿着自己记录的相位去等，等到 barrier 翻过这个相位就返回。

相位这个设计值得多说一句，因为它是多级流水线复用同一个 barrier 的关键。流水线里每个 smem buffer 会被反复填充和消费，barrier 也跟着反复完成。如果只有"完成/未完成"两个状态，第二轮使用时就分不清这次完成是本轮的还是上一轮的。相位就是给每一轮完成编号（只保留奇偶），生产者和消费者各自在本地记一个 phase 变量，buffer 索引每绕回一圈就把自己的 phase 翻转一次，这样双方永远在等"属于自己这一轮"的完成事件。后面 3.7 节会看到具体代码。

## 1.2 warpgroup 与 warp specialization

Hopper kernel 的标准组织方式是 warp specialization：把一个 CTA 里的 warpgroup 分成两种角色，producer 只负责发 TMA 搬数据，consumer（math）只负责发 GMMA 算。两边通过 mbarrier 传递 buffer 的所有权，形成流水线。

这样做的好处是解耦：producer 不需要寄存器（TMA 不经过寄存器），math 不需要操心搬运。Hopper 还提供了配套的寄存器重分配指令 `setmaxnreg`，可以让 producer warpgroup 把自己的寄存器让出来给 math warpgroup 用。一个 SM 的寄存器文件是 64K 个 32 位寄存器，每线程默认上限 255 个。这个 kernel 里 producer 降到 24 个/线程，math 升到 168 个/线程，math warpgroup 就能把大块 accumulator 全放寄存器里，同时整个 CTA 的寄存器总量不超预算。

## 1.3 CuTe 最小必要知识

CuTe 是 CUTLASS 3.x 的底层张量抽象，这个 kernel 用到的概念就四个。

**Layout = Shape + Stride**。Layout 是一个从逻辑坐标到内存偏移的函数。`make_layout(make_shape(4, 8), make_stride(8, 1))` 表示一个 4x8 的行优先矩阵。Layout 可以嵌套、可以复合，`cosize(layout)` 返回这个 layout 覆盖的内存元素个数（这个 kernel 用它来算 smem buffer 的大小）。

**Tensor = 指针 + Layout**。`make_tensor(make_smem_ptr(p), layout)` 把一块 smem 包装成带形状的张量，之后所有切片、分块操作都在 Tensor 上做，不用手算偏移。

**TiledMMA / partition**。`make_tiled_mma(MMA_ATOM{})` 把一条 GMMA 指令包装成一个"计算 tile"，`tiled_mma.get_slice(tid)` 拿到当前线程的视角，然后 `thr_mma.partition_A(sW)` 把 smem 上的权重 tile 按 GMMA 指令对操作数的要求切分好——返回的张量第 0 维是单条指令消耗的数据（MMA），后面的维度是要循环的指令次数（MMA_M, MMA_K）。关键点：对 SS 模式的 GMMA 来说，`make_fragment_A` 得到的"fragment"里放的不是数据而是 matrix descriptor（指向 smem 的描述符），真正的数据到 GMMA 执行时才从 smem 读。只有 accumulator（`partition_fragment_C`）是真的寄存器。

**TiledCopy / TMA partition**。TMA 的 copy 对象由 host 侧 `make_tma_copy` 创建。kernel 里 `tma.get_tma_tensor(shape)` 返回一个"坐标张量"——它不含数据指针，只是把全局 tensor 的逻辑坐标空间摆出来，这样 `partition_S` 切出来的每一块就携带了"第 (i,j) 个 tile 在全局的什么位置"的信息，`cute::copy(tma.with(barrier), src_block, dst_block)` 时 TMA 硬件就知道该搬哪里。这是 CuTe 里比较绕的一个设计，记住"TMA tensor 是坐标不是数据"就行。

## 1.4 kernel 整体结构

先给一张整体的图，后面逐段解析时可以对着看：

```
grid = min(SM 数, tile 总数)，每个 CTA 常驻（persistent），循环领取任务

CTA 内部（以 kWarpGroupN=1 为例，256 线程）:

  warpgroup 1 (producer, 24 regs/thread)          warpgroup 0 (math, 168 regs/thread)
  ┌─────────────────────────────┐                ┌─────────────────────────────────┐
  │ 只有 1 个线程干活:           │                │ 128 线程一起:                    │
  │ loop over K tiles:          │   mbarrier     │ loop over K tiles:              │
  │   等 writable_x    ─────────┼───────────────>│   等 readable_x / readable_w    │
  │   TMA load x                │                │   GMMA(w_low  x x) -> acc_low   │
  │   等 writable_w[low]        │<───────────────┼── arrive writable_*             │
  │   TMA load w_low            │                │   GMMA(w_high x x) -> acc_high  │
  │   等 writable_w[high]       │                │ epilogue:                       │
  │   TMA load w_high           │                │   y = acc_high + scale*acc_low  │
  │   (TMA 完成自动 arrive       │                │   寄存器 -> smem (STSM)         │
  │    readable_*)              │                │   smem -> gmem (TMA store)      │
  └─────────────────────────────┘                └─────────────────────────────────┘

  smem: x buffer  [kStage 级]
        w buffer  [kStage 级][kWarpGroupN][2]   <- 2 = {w_low, w_high}
        y buffer  (epilogue 中转)
```

数据流是三级：gmem --TMA--> smem --GMMA(SS)--> 寄存器 accumulator --STSM--> smem --TMA store--> gmem。全程没有普通的 ld.global/st.global。

# 0x2. 算法回顾

kernel 名字里的 bf16xfp32 指 bf16 激活乘 fp32 权重。硬件没有这种 Tensor Core 指令，做法是离线把 fp32 权重拆成两个 bf16：

```python
scale = 1 / 256
w_high = w.to(bf16)                            # 保住前 8 位尾数
w_low  = ((w - w_high.float()) / scale).to(bf16)  # 残差放大 256 倍再存，又保住 8 位
```

kernel 里对每个 K tile 做两批 GMMA，`acc_low += w_low @ x`、`acc_high += w_high @ x`，最后 epilogue 合并 `y = acc_high + scale * acc_low`。scale 是 2 的幂所以乘除无损，两段拼起来相当于 16 位左右的尾数精度，对 router logits 这种用途和 fp32 没有实际区别。代价是 GMMA 数量翻倍。数学原理的详细推导和精度实验在博客那篇文章里，这里不重复。

这个拆分决定了 kernel 和普通 GEMM 的两个不同点：smem 里权重 buffer 是双份的（低位一份高位一份），主循环里每个 K tile 有两段 GMMA。除此之外它就是一个标准的 Hopper warp specialization GEMM。

# 0x3. 逐段解析

## 3.1 文件头（第 1~17 行）

```c++
#include <cuda.h>
#include <stdio.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "cutlass/arch/reg_reconfig.h"
#include "src/gemm/gemm.h"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace gemm {

namespace kernels {
```

`cute/tensor.hpp` 是 CuTe 的总入口，Layout/Tensor/TMA/GMMA 的封装都从这里来。`cutlass/arch/reg_reconfig.h` 提供 `warpgroup_reg_alloc/dealloc`（就是 `setmaxnreg` 的包装）。`src/utils/utils.cuh` 是 hpc-ops 自己的工具头，这个 kernel 实际只用了里面的一小部分：向量化 load/store（`vec_t`）、`load_global_volatile`、`syncwarpgroup`、`bar_sync`、`fence_async_global`。我们拷进 SGLang 的版本把这个子集直接内联进了同一个文件。

值得一提的是 `initialize_barrier`、`wait_barrier`、`arrive_barrier`、`set_barrier_transaction_bytes` 这些函数不在 utils 里，它们是 CuTe 自带的（`cute` 命名空间，kernel 内部 `using namespace cute` 之后直接用），分别对应 mbarrier 的 init/try_wait.parity/arrive/arrive.expect_tx 这几条 PTX。第一次读代码很容易找不到它们定义在哪。

## 3.2 get_next_tile：persistent kernel 的任务分配（第 19~41 行）

```c++
template <int kBlockSwizzle, int kSplitK>
__device__ __forceinline__ auto get_next_tile(int iblock, int num_tile_m, int num_tile_n,
                                              cutlass::FastDivmod swizzle_divider,
                                              cutlass::FastDivmod flat_divider) {
  int itile_m, itile_n;
  int num_tile_bxn = kBlockSwizzle * num_tile_n * kSplitK;
  int total_sizzle_blocks = num_tile_m / kBlockSwizzle * num_tile_bxn;

  if (iblock >= total_sizzle_blocks) {
    flat_divider(itile_m, itile_n, iblock);
  } else {
    int i_bxn, i_bxn_res;
    swizzle_divider(i_bxn, i_bxn_res, iblock);

    itile_m = i_bxn * kBlockSwizzle + i_bxn_res % kBlockSwizzle;
    itile_n = i_bxn_res / kBlockSwizzle;
  }

  int ichunk = itile_n % kSplitK;
  itile_n = itile_n / kSplitK;

  return cute::make_tuple(itile_m, itile_n, ichunk);
}
```

背景先补一下 persistent kernel。常规 GEMM 是每个 output tile 开一个 CTA，grid 大小等于 tile 数。persistent 风格是 grid 只开 `min(SM 数, tile 数)` 个 CTA，每个 CTA 干完一个 tile 就用 `iblock += gridDim.x` 领下一个，直到领完。好处是 CTA 不用反复启动和退出（prologue 里 barrier 初始化、TMA descriptor 准备这些开销只付一次），并且任务到 SM 的映射是确定的，方便做 L2 友好的调度。

这个函数就是把一维任务号 `iblock` 解码成 `(itile_m, itile_n, ichunk)` 三元组。两个细节：

第一是 **block swizzle**。如果任务号按行优先直接展开（m 优先），同一时刻 132 个 SM 在算 132 个不同的 m tile，它们用到的权重列却可能各不相同，L2 里权重 tile 的复用就差。swizzle 的做法是把 m 方向每 `kBlockSwizzle=4` 行分成一组，组内先走完这 4 行再推进 n，这样相邻的 4 个任务共享同一个权重 tile 列，同一批常驻 CTA 在时间上也大概率处在相近的 n 列上，权重从 L2 读的命中率就高了。`total_sizzle_blocks` 是能被 4 整除的那部分，剩下的尾巴退化成平铺（`flat_divider` 分支）。

第二是 **split-k 编码在 n 维里**。host 侧把 `num_tile_n` 乘了 kSplitK（见 0x4 节），所以解码出的 `itile_n` 要先取模拿到 `ichunk`（本任务负责 K 维的第几个分片），再除回去得到真正的 n tile 号。也就是说同一个 (m, n) tile 会被 kSplitK 个任务认领，每个任务只算 K 维的 1/kSplitK，partial 结果后面再归约。

`cutlass::FastDivmod` 是预计算的整数除法：GPU 上整数除法/取模是几十条指令的软件序列，FastDivmod 在 host 侧把除数转成乘法+移位的形式，device 上一次 mul.hi 加一次 shift 搞定。所有除数（`num_tile_bxn`、`num_tile_n`）在 host 侧就已知，所以这里全部用 FastDivmod 传进来。

## 3.3 splitk_reduce（第 43~81 行）

```c++
template <typename Tout, int kTileM, int kTileN, int kSplitK, int kWarpCount>
__device__ __forceinline__ void splitk_reduce(Tout *y_ptr, float *splitk_y_ptr, int m, int n,
                                              int itile_m, int itile_n) {
  int iwarp = threadIdx.x / 32;
  int ilane = threadIdx.x % 32;

  if (itile_m * kTileM + iwarp >= m) {
    return;
  }

  if (ilane * 4 >= kTileN || itile_n * kTileN + ilane * 4 >= n) {
    return;
  }

  auto *y_tile = y_ptr + (itile_m * kTileM + iwarp) * n + itile_n * kTileN + ilane * 4;
  auto *splitk_y_tile =
      splitk_y_ptr + (itile_m * kTileM + iwarp) * n + itile_n * kTileN + ilane * 4;

  int local_m = m - (itile_m * kTileM + iwarp);

#pragma unroll
  for (int irow = 0; irow < kTileM; irow += kWarpCount) {
    if (irow >= local_m) {
      return;
    }
    auto y = load<float, 4>(splitk_y_tile + irow * n);

#pragma unroll
    for (int ichunk = 1; ichunk < kSplitK; ++ichunk) {
      auto split_y = load<float, 4>(splitk_y_tile + ichunk * m * n + irow * n);
#pragma unroll
      for (int i = 0; i < 4; i++) {
        y[i] += split_y[i];
      }
    }

    store(y_tile + irow * n, to<Tout>(y));
  }
}
```

split-k 的归约函数，把 `[kSplitK, m, n]` 的 fp32 partial buffer 沿第 0 维加起来写进最终输出。分工方式是一个 warp 管一行（`iwarp` 对应 tile 内的行号，行数超过 warp 数就按 `kWarpCount` 步进循环），warp 内每个 lane 管连续 4 个 float（`ilane * 4`），`load<float, 4>` 是 16 字节向量化访问，这也是为什么调用方要保证 n 是 4 的倍数（实际约束是 n % 64 == 0，远强于这个）。`to<Tout>` 在输出是 bf16 时做 float→bf16 的转换，fp32 输出时是恒等。

这段是整个文件里最普通的 CUDA 代码，唯一要注意的是它不是独立 kernel，而是主 kernel 尾声里被 math warpgroup 调用的（3.10 节），这是这个 kernel "单 kernel split-k"设计的一部分。

## 3.4 kernel 签名与线程角色（第 83~104 行）

```c++
template <typename Tin, typename TY, typename Tout, typename TiledMma, typename TmaX,
          typename TmaWH, typename TmaWL, typename TmaY, int kTileM, int kTileN, int kTileK,
          int kStage, int kWarpGroupN, typename SLayoutX, typename SLayoutW, typename SLayoutY,
          int kBlockSwizzle, int kSplitK>
__global__ void __launch_bounds__(128 * (kWarpGroupN + 1), 1)
    gemm_bf16xfp32_kernel(const __grid_constant__ TmaX tma_x, const __grid_constant__ TmaWH tma_wh,
                          const __grid_constant__ TmaWL tma_wl, const __grid_constant__ TmaY tma_y,
                          Tout *y_ptr, float *splitk_y_ptr, int *split_flag_ptr, int m, int n,
                          int k, float scale, cutlass::FastDivmod swizzle_divider,
                          cutlass::FastDivmod flat_divider,
                          cutlass::FastDivmod reduce_flat_divider) {
```

模板参数逐个过：

- `Tin`：输入类型，实际就是 bf16；
- `TY`：epilogue 经过 smem 中转时的类型。splitk>1 时是 float（partial 结果必须保精度），否则等于输出类型；
- `Tout`：最终输出类型，float 或 bf16；
- `TiledMma` 和四个 Tma 类型：host 侧构造好传进来的 MMA/TMA 对象类型，kernel 模板不关心细节；
- `kTileM/kTileN/kTileK`：tile 大小。注意命名，kTileM 是 token 维（16 或 64），kTileN 是权重输出维（固定 64），kTileK 是 K 维步长（64 或 128）；
- `kStage`：流水线级数（3 或 5）；
- `kWarpGroupN`：math warpgroup 个数（1 或 2），每个 math warpgroup 负责 n 方向相邻的一个 kTileN 条带；
- `SLayout*`：三个 smem buffer 的 layout 类型；
- `kBlockSwizzle`、`kSplitK`：前面讲过了。

`__launch_bounds__(128 * (kWarpGroupN + 1), 1)` 声明 block 大小是 (wgn+1) 个 warpgroup（math 加一个 producer），第二个参数 1 表示每个 SM 只驻留 1 个 CTA。占用率(occupancy)只有 1 看起来很低，但这是 Hopper 大 kernel 的标准做法：与其靠多 CTA 切换掩延迟，不如把寄存器和 smem 全给一个 CTA，让它内部用流水线自己掩。

TMA descriptor 用 `__grid_constant__` 修饰，按值传参。这是 TMA 的硬性要求：descriptor 必须放在 const 空间（kernel 参数区），线程发 TMA 指令时引用的是它的地址，不能放普通局部变量里。

```c++
  int idx = threadIdx.x;

  int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
  int elected = cute::elect_one_sync();
  bool is_leader_in_block = (iwarp == 0) && elected;
  bool is_leader_in_warpgroup = ((iwarp % 4) == 0) && elected;
```

两个小 trick。`__shfl_sync(mask, idx / 32, 0)` 算 warp 号却广播 lane 0 的结果，值本来就是 warp 内一致的，广播一下是为了让编译器确信它是 uniform 值（放进 uniform register，省掉每个 lane 各存一份，也避免编译器保守地认为后续分支可能发散）。`cute::elect_one_sync()` 是 PTX `elect.sync` 指令，从当前 warp 里选出一个 lane 返回 true，比 `lane_id == 0` 的写法在 SASS 层面更省（不用算 lane id）。组合起来，`is_leader_in_block` 是整个 CTA 一个线程（barrier 初始化用），`is_leader_in_warpgroup` 是每个 warpgroup 一个线程（arrive barrier、发 TMA store 用）。

## 3.5 smem 与 mbarrier 声明（第 106~125 行）

```c++
  constexpr int kWLIdx = 0;
  constexpr int kWHIdx = 1;

  __shared__ uint64_t writable_x[kStage];
  __shared__ uint64_t readable_x[kStage];

  __shared__ uint64_t writable_w[kStage][kWarpGroupN][2];
  __shared__ uint64_t readable_w[kStage][kWarpGroupN][2];

  extern __shared__ uint8_t shm_data[] alignas(128);
  auto *shm_x = (Tin *)shm_data;
  auto *shm_w = (Tin *)shm_x + cosize(SLayoutX{});
  auto *shm_y = (TY *)(shm_w + cosize(SLayoutW{}));

  auto sX = make_tensor(make_smem_ptr(shm_x), SLayoutX{});
  auto sW = make_tensor(make_smem_ptr(shm_w), SLayoutW{});
```

每个 mbarrier 就是一个 smem 里的 uint64_t。这里的组织是经典的 full/empty 双 barrier 配对：

- `readable_*`（full 信号）：producer 填完数据后完成，consumer 等它；
- `writable_*`（empty 信号）：consumer 用完 buffer 后完成，producer 等它才敢覆写。

x 的 barrier 每级流水线一对；权重的 barrier 是 `[kStage][kWarpGroupN][2]` 三维——每级、每个 math warpgroup、高低位各一对。为什么权重要按 warpgroup 拆而 x 不拆？因为 x（激活 tile）是所有 math warpgroup 共用的（大家算的是同一批 token 的不同 n 条带），而权重每个 warpgroup 用自己那份。共用的 buffer 用一个 barrier 加多次 arrive 来管（下一段能看到 `writable_x` 初始化成 kWarpGroupN），独享的 buffer 各管各的，同步粒度最细，谁先用完谁先释放，不互相等。高低位权重也拆开是同一个道理：低位 GMMA 算完就可以立刻释放低位 buffer 让 TMA 填下一级，不用等高位算完。

动态 smem 是手工划分的：x buffer、w buffer、y buffer 依次排布，边界用 `cosize(Layout{})`（layout 覆盖的元素数）算。三个 layout 都是 host 侧用 swizzle atom 拼出来的（0x4 节），这里只管拿来包成 Tensor。y buffer 和 x/w 复用同一块 smem 之后的空间，因为 epilogue 阶段才用它。

## 3.6 TMA tensor 与 partition（第 117~141 行）

```c++
  auto gX = tma_x.get_tma_tensor(make_shape(m, k));
  auto gWH = tma_wh.get_tma_tensor(make_shape(n, k));
  auto gWL = tma_wl.get_tma_tensor(make_shape(n, k));

  auto gY = make_tensor(make_gmem_ptr((float *)(nullptr)), make_shape(Int<kTileN>{}, Int<kTileM>{}),
                        make_stride(Int<kTileM>{}, Int<1>{}));

  auto btma_x = tma_x.get_slice(0);
  auto btma_wh = tma_wh.get_slice(0);
  auto btma_wl = tma_wl.get_slice(0);

  auto tXg = btma_x.partition_S(gX);  // (TMA, TMA_M, TMA_K)
  auto tXs = btma_x.partition_D(sX);  // (TMA, _1, _1, kStage)

  auto tWHg = btma_wh.partition_S(gWH);  // (TMA, TMA_N, TMA_K)
  auto tWHs = btma_wh.partition_D(sW);   // (TMA, _1, _1, kStage)

  auto tWLg = btma_wl.partition_S(gWL);
  auto tWLs = btma_wl.partition_D(sW);

  int num_tile_m = size<1>(tXg);
  int num_tile_n = (size<1>(tWHg) + kWarpGroupN - 1) / kWarpGroupN;
```

1.3 节说过 `get_tma_tensor` 返回的是坐标张量。`partition_S(gX)` 按 TMA copy 的 tile 大小（kTileM x kTileK）把全局坐标空间切块，得到 `(TMA, TMA_M, TMA_K)` 形状——第 0 维是一次 TMA 搬运的整块，后两维是 m 方向和 k 方向各有多少块。之后 `tXg(_, itile_m, itile_k)` 就是"第 (itile_m, itile_k) 块"的坐标，喂给 `cute::copy` 就搬这一块。`partition_D(sX)` 对 smem 侧做同样的切分，多出来的最后一维是流水线级。

所以 `size<1>(tXg)` 顺手就是 m 方向的 tile 总数，`size<2>` 是 K 方向 tile 数（后面主循环的 `ntile_k`），不需要 kernel 自己再做除法。`num_tile_n` 除以 kWarpGroupN 是因为任务分配以"kWarpGroupN 个权重条带"为一个单位，一个 CTA 一次领 kWarpGroupN 条。

`gY` 那行看着奇怪：一个指针为 nullptr 的张量。它只是用来给 `partition_fragment_C` 提供形状信息（accumulator 的逻辑形状是 kTileN x kTileM），从不真正访问，属于 CuTe 的常见用法。

`get_slice(0)` 传 0 是因为 TMA 只需要一个线程发起，不像 Ampere 的 cp.async 需要每线程算自己的分工。

## 3.7 barrier 初始化与 producer warpgroup（第 143~223 行）

```c++
  if (is_leader_in_block) {
#pragma unroll
    for (int i = 0; i < kStage; ++i) {
      initialize_barrier(readable_x[i], 1);
      initialize_barrier(writable_x[i], kWarpGroupN);
    }
#pragma unroll
    for (int istage = 0; istage < kStage; ++istage) {
#pragma unroll
      for (int j = 0; j < kWarpGroupN; ++j) {
        initialize_barrier(readable_w[istage][j][kWLIdx], 1);
        initialize_barrier(readable_w[istage][j][kWHIdx], 1);
        initialize_barrier(writable_w[istage][j][kWLIdx], 1);
        initialize_barrier(writable_w[istage][j][kWHIdx], 1);
      }
    }
  }

  // sync to avoid ahead thread use(wait) readable when it is not initizlized yet
  __syncthreads();
```

`initialize_barrier(bar, count)` 的 count 是完成一个 phase 需要的 arrive 次数。`readable_*` 都是 1：TMA 完成本身就是那一次 arrive。`writable_x` 是 kWarpGroupN：x buffer 被所有 math warpgroup 共享，要每个 warpgroup 都 arrive 一次（表示"我用完了"）才算真正可写。`writable_w` 是 1：权重 buffer 每个 warpgroup 独享。初始化只能一个线程做，做完必须 `__syncthreads()`，否则别的线程可能在 barrier 还没初始化时就开始 wait，这是 mbarrier 使用的标准注意事项。

接下来分叉进 producer：

```c++
  if (idx >= kWarpGroupN * 128) {
    cutlass::arch::warpgroup_reg_dealloc<24>();
    idx -= kWarpGroupN * 128;
    constexpr int kTransactionBytesX = sizeof(Tin) * kTileK * kTileM;
    constexpr int kTransactionBytesW = sizeof(Tin) * kTileK * kTileN;

    int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
    int is_leader_in_load = ((iwarp == 0) && elected);

    if (is_leader_in_load) {
      int phase = 1;  // start with ok
      int ismem_write = __shfl_sync(0xFFFFFFFF, 0, 0);
      int iblock = blockIdx.x;
      int ntile_k = size<2>(tXg);

      while (true) {
        auto [itile_m, itile_n, ichunk] = get_next_tile<kBlockSwizzle, kSplitK>(
            iblock, num_tile_m, num_tile_n, swizzle_divider, flat_divider);

        if (itile_m >= num_tile_m) {
          break;
        }

        iblock += gridDim.x;

#pragma unroll 1
        for (int itile_k = ichunk; itile_k < ntile_k; itile_k += kSplitK) {
          // load a
          wait_barrier(writable_x[ismem_write], phase);
          cute::copy(tma_x.with(readable_x[ismem_write]), tXg(_, itile_m, itile_k),
                     tXs(_, 0, 0, ismem_write));
          set_barrier_transaction_bytes(readable_x[ismem_write], kTransactionBytesX);
          // load wgX low
#pragma unroll
          for (int wg = 0; wg < kWarpGroupN; ++wg) {
            wait_barrier(writable_w[ismem_write][wg][kWLIdx], phase);
            cute::copy(tma_wl.with(readable_w[ismem_write][wg][kWLIdx]),
                       tWLg(_, kWarpGroupN * itile_n + wg, itile_k),
                       tWLs(_, 0, 0, wg, kWLIdx, ismem_write));
            set_barrier_transaction_bytes(readable_w[ismem_write][wg][kWLIdx], kTransactionBytesW);
          }
          // load wgX high
#pragma unroll
          for (int wg = 0; wg < kWarpGroupN; ++wg) {
            wait_barrier(writable_w[ismem_write][wg][kWHIdx], phase);
            cute::copy(tma_wh.with(readable_w[ismem_write][wg][kWHIdx]),
                       tWHg(_, kWarpGroupN * itile_n + wg, itile_k),
                       tWHs(_, 0, 0, wg, kWHIdx, ismem_write));
            set_barrier_transaction_bytes(readable_w[ismem_write][wg][kWHIdx], kTransactionBytesW);
          }

          ++ismem_write;
          if (ismem_write == kStage) {
            ismem_write = 0;
            phase ^= 1;
          }
        }
      }
    }
  }
```

信息量比较大，拆开说。

**寄存器重分配**。`warpgroup_reg_dealloc<24>()` 让整个 producer warpgroup 把每线程寄存器上限降到 24。producer 里实际只有一个线程干活，其余 127 个线程直接闲置到 kernel 结束，24 个寄存器纯粹是"占位最小值"。省下的额度由 math warpgroup 的 `warpgroup_reg_alloc<168>()` 拿走。

**单线程发 TMA**。`is_leader_in_load` 筛出 producer 第 0 个 warp 的一个 lane，整个搬运循环只有它在跑。这是 TMA 相对 cp.async 最大的观感变化：搬运逻辑就是一段单线程的串行代码，读起来像 CPU 程序。

**流水线协议**。对每个 buffer 级 `ismem_write`：先 `wait_barrier(writable, phase)` 等 consumer 释放，然后 `cute::copy(tma.with(readable), src, dst)` 把 readable barrier 绑定到这次 TMA 上，再 `set_barrier_transaction_bytes(readable, bytes)` 告诉 barrier 预期字节数。TMA 硬件搬完这么多字节，readable 自动完成，consumer 那边的 wait 就通过了。注意发起 copy 和设置 expect_tx 的顺序反直觉（先 copy 后 set），这是允许的：barrier 从 init 起就在收数，expect_tx 只是补上"收满多少算完"的阈值。

**phase 的翻转**。producer 的 phase 初始化成 1，注释写着 "start with ok"：kernel 刚启动时所有 buffer 都是空的，第一轮 `wait_barrier(writable, 1)` 应该直接通过而不是等待，用初相 1 实现这一点（consumer 那边对 readable 用初相 0，第一轮是真等）。之后每当 buffer 索引绕回 0，本地 phase 翻转一次，正好跟 barrier 每完成一轮自动翻相位对上。这个"本地相位 + 索引回绕时翻转"的模式在所有 Hopper 流水线代码里都长一样，看懂一次以后到处都认识。

**循环结构**。外层 while 领任务，内层 K 循环从 `ichunk` 开始、步长 kSplitK——split-k 的 K 维分片就体现在这里，chunk c 负责 itile_k = c, c+kSplitK, c+2*kSplitK, ... 这些 K tile。`#pragma unroll 1` 强制不展开，控制代码体积。三份 TMA（x、各 warpgroup 的低位权重、高位权重）按顺序发，低位在前，和 math 侧先算低位的顺序呼应。

## 3.8 math warpgroup 主循环（第 224~312 行）

```c++
  } else {
    // math warpgroup
    cutlass::arch::warpgroup_reg_alloc<168>();

    int idx_in_warpgroup = idx % 128;
    int iwarpgroup = idx / 128;
    int iwarp_in_warpgroup = idx_in_warpgroup / 32;
    int elected_idx_in_warpgroup = ((iwarp_in_warpgroup == 0) && elected);

    TiledMma tiled_mma;

    auto thr_mma = tiled_mma.get_slice(idx_in_warpgroup);
    auto tWs4r = thr_mma.partition_A(sW);
    auto tXs4r = thr_mma.partition_B(sX);

    auto tWr = thr_mma.make_fragment_A(tWs4r);  // (MMA, MMA_M, MMA_K, kStage)
    auto tXr = thr_mma.make_fragment_B(tXs4r);  // (MMA, MMA_N, MMA_K, kStage)

    auto tYr_low = thr_mma.partition_fragment_C(gY);
    auto tYr_high = make_tensor_like(tYr_low);
```

partition 这几行是 CuTe 的核心用法。`partition_A(sW)` 把 smem 权重张量按 GMMA atom 对 A 操作数的要求切好；`make_fragment_A` 对 SS 模式的 GMMA 生成 matrix descriptor 的集合（1.3 节强调过：不是把数据搬进寄存器，descriptor 只是 64 位的"smem 地址+布局"编码，GMMA 执行时直接从 smem 读）。真正占寄存器的是两个 accumulator：`tYr_low` 和 `tYr_high`，每个是 kTileN x kTileM / 128 个 float。tile64 配置下每线程每个 acc 是 64x64/128 = 32 个 float，两个 acc 就占掉 64 个寄存器，再加上流水线控制和地址计算的开销，这就是 math warpgroup 要 `reg_alloc<168>` 的原因。

这里能看到操作数角色的分配：**权重是 A，激活是 B**。GMMA atom 是 `SM90_64x64x16` 或 `SM90_64x16x16`，M 维固定 64 对应权重的 n 方向（kTileN=64），N 维（64 或 16）对应 token 数方向（kTileM）。这样安排是因为 GMMA 的 M 维固定是 64，而这个 GEMM 里权重维（几百）总是比小 batch 的 token 维更"稳定"，让权重占 M、token 占可变的 N，小 m 时选 N=16 的 atom 就不浪费算力。

```c++
    int ismem_read = 0;
    int phase = 0;

    int iblock = blockIdx.x;
    int last_tile_m = -1;
    int last_tile_n = -1;
    while (true) {
      auto [itile_m, itile_n, ichunk] = get_next_tile<kBlockSwizzle, kSplitK>(
          iblock, num_tile_m, num_tile_n, swizzle_divider, flat_divider);
      if (itile_m >= num_tile_m) {
        break;
      }
      iblock += gridDim.x;

      clear(tYr_low);
      clear(tYr_high);

      int ntile_k = size<2>(tXg);

      tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
#pragma unroll 1
      for (int itilek = ichunk; itilek < ntile_k; itilek += kSplitK) {
        wait_barrier(readable_x[ismem_read], phase);

        // mma low
        wait_barrier(readable_w[ismem_read][iwarpgroup][kWLIdx], phase);
        warpgroup_fence_operand(tYr_low);
        warpgroup_arrive();
#pragma unroll
        for (int ik = 0; ik < size<2>(tXr); ++ik) {
          cute::gemm(tiled_mma, tWr(_, _, ik, iwarpgroup, kWLIdx, ismem_read),
                     tXr(_, _, ik, ismem_read), tYr_low(_, _, _));
          tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        }
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(tYr_low);

        if (elected_idx_in_warpgroup) {
          arrive_barrier(writable_w[ismem_read][iwarpgroup][kWLIdx]);
        }

        // mma high
        wait_barrier(readable_w[ismem_read][iwarpgroup][kWHIdx], phase);
        warpgroup_fence_operand(tYr_high);
        warpgroup_arrive();
#pragma unroll
        for (int ik = 0; ik < size<2>(tXr); ++ik) {
          cute::gemm(tiled_mma, tWr(_, _, ik, iwarpgroup, kWHIdx, ismem_read),
                     tXr(_, _, ik, ismem_read), tYr_high(_, _, _));
          tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        }

        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(tYr_high);

        if (elected_idx_in_warpgroup) {
          arrive_barrier(writable_x[ismem_read]);
          arrive_barrier(writable_w[ismem_read][iwarpgroup][kWHIdx]);
        }

        ++ismem_read;
        if (ismem_read == kStage) {
          phase ^= 1;
          ismem_read = 0;
        }
      }
```

先补 GMMA 异步指令的一套围栏，它们各自对应一条 PTX：

- `warpgroup_arrive()`（`wgmma.fence`）：告诉硬件"接下来这批 wgmma 要读的寄存器/smem 我已经准备好了"，在发 GMMA 前调用；
- `warpgroup_commit_batch()`（`wgmma.commit_group`）：把已发射的 GMMA 打包成一个 group；
- `warpgroup_wait<N>()`（`wgmma.wait_group N`）：等到未完成的 group 数不超过 N。`wait<0>` 就是全部等完；
- `warpgroup_fence_operand(acc)`：编译器级的 fence，防止编译器在异步 GMMA 还没写完 accumulator 时就重排后续对它的读写。GMMA 前后各一次。

`tiled_mma.accumulate_` 控制 GMMA 的 D = A*B + D 还是 D = A*B（`ScaleOut::One/Zero`）。每个任务开头设 Zero，第一条 GMMA 覆写 accumulator（等价于清零），之后全是累加。上面还有一个显式 `clear`，两者语义重复，保险写法。

然后是主循环的核心节奏，每个 K tile：

1. 等 x 和低位权重就绪 → 发一批 GMMA 到 `tYr_low` → `wait<0>` 等这批全部执行完 → arrive 释放低位权重 buffer；
2. 等高位权重就绪 → 发一批 GMMA 到 `tYr_high` → `wait<0>` → arrive 释放 x 和高位权重 buffer。

为什么每批后面必须 `wait<0>` 才能 arrive？因为 GMMA 是 SS 模式，操作数就在 smem buffer 里，指令没执行完，buffer 就不能让 producer 覆写。`wait<0>` 保证读取已经结束，arrive 才是安全的。

但这也是这个 kernel 最大的性能弱点：`wait<0>` 是把 GMMA 流水线完全排空，每个 K tile 排空两次，排空期间 Tensor Core 无事可做。标准的 CUTLASS mainloop 用的是延迟释放（比如 `wait<1>`，保持一批 GMMA 在飞行中，释放的是上上批的 buffer），这里没有做。在 H20 上这无所谓——H20 的 Tensor Core 峰值低（约 148 TFLOPS），GMMA 本身执行得慢，排空开销全被盖住，这个 kernel 在 H20 上顶着算力上限跑；但在 H200（989 TFLOPS）上 GMMA 快了 6.7 倍，气泡完全暴露，ncu 显示 81% 的周期发不出指令，MFU 只有 36%。缓解办法之一是 kWarpGroupN=2，两个 math warpgroup 的排空互相错开能盖掉一部分，这是我们在 H200 调参里验证过的（大 m 段 1.2~1.6x）。彻底解决要改成延迟释放，属于后续工作。

arrive 只由 `elected_idx_in_warpgroup` 一个线程做：mbarrier 的 arrive 计数是按次算的，128 个线程各 arrive 一次和 1 个线程 arrive 一次含义不同，这里协议约定的是"每 warpgroup 一次"。

## 3.9 epilogue：融合、STSM、TMA store（第 314~364 行）

```c++
      // float32 -> bfloat16
      auto tYrh = make_tensor_like<TY>(tYr_low);

#pragma unroll
      for (int i = 0; i < size(tYr_low); ++i) {
        tYrh(i) = (TY)(tYr_low(i) * scale + tYr_high(i));
      }
```

整个 bf16x2 算法的收口就这一行：`y = acc_low * scale + acc_high`，编译成 FFMA，每线程几十条，代价可以忽略。做完之后高低两个 accumulator 合成一份 `TY` 类型（splitk>1 时是 float，否则是输出类型）的结果。

```c++
      using STSM_ATOM =
          std::conditional_t<kTileM == 8, cute::SM90_U16x4_STSM_T, cute::SM90_U16x8_STSM_T>;
      using STS_ATOM =
          std::conditional_t<std::is_same_v<TY, float>, UniversalCopy<uint32_t>, STSM_ATOM>;
      // Epilogue
      auto sY = make_tensor(make_smem_ptr((TY *)shm_y), SLayoutY{});  // (M, N)
      using R2SCopyAtomY = Copy_Atom<STS_ATOM, TY>;
      auto tiled_copy_y = make_tiled_copy_C(R2SCopyAtomY{}, tiled_mma);
      auto thr_copy_y = tiled_copy_y.get_slice(idx_in_warpgroup);

      auto tYr4s = thr_copy_y.retile_S(tYrh);
      auto tYs4r = thr_copy_y.partition_D(sY);

      cute::tma_store_wait<0>();
      syncwarpgroup(iwarpgroup);

      cute::copy(tiled_copy_y, tYr4s, tYs4r(_, _, _, iwarpgroup));
```

输出不直接写 gmem，而是先写回 smem 再用 TMA store 搬出去。原因有两个：一是 GMMA accumulator 在寄存器里的分布是按 MMA 指令的 lane 布局来的，直接写 gmem 是零碎的非合并访问；二是 TMA store 和 TMA load 一样是一条指令搬一个 tile，还能和后续计算重叠。

寄存器到 smem 这一步选的 copy atom 有讲究。bf16 输出用 `STSM`（`stmatrix` 指令，ldmatrix 的写方向版本，`_T` 后缀是转置变体），一条指令让一个 warp 把寄存器里的 matrix fragment 按 GMMA 的天然布局整块写进 smem，自动处理布局转换。fp32 输出没有对应的 stmatrix，退化成 `UniversalCopy<uint32_t>`（普通 32 位 store）。`make_tiled_copy_C(atom, tiled_mma)` 生成和这个 TiledMMA 的 C 布局配套的 copy 计划，`retile_S` 把 accumulator 重新按 copy atom 的粒度组织。这几个函数第一次看不用深究细节，记住"它们负责把 GMMA 的寄存器布局正确映射到 smem"即可。

`cute::tma_store_wait<0>()` 在写 sY 之前调用，等的是上一个任务的 TMA store 完成——sY 只有一份（不是多级流水线），必须确认上一个 tile 已经搬走才能覆写。`syncwarpgroup(iwarpgroup)` 是 `barrier.cta.sync` 带 barrier id 的版本，只同步本 warpgroup 的 128 个线程（每个 warpgroup 用自己的 id，互不干扰），比全 CTA 的 `__syncthreads` 粒度细。

```c++
      syncwarpgroup(iwarpgroup);
      cute::tma_store_fence();

      if (is_leader_in_warpgroup) {
        auto gYY = tma_y.get_tma_tensor(make_shape(n, m, kSplitK));
        auto btma_y = tma_y.get_slice(0);

        auto tYs = btma_y.partition_S(sY);
        auto tYg = btma_y.partition_D(gYY);

        cute::copy(tma_y, tYs(_, 0, 0, iwarpgroup),
                   tYg(_, kWarpGroupN * itile_n + iwarpgroup, itile_m, ichunk));
        tma_store_arrive();
      }
    }
```

`tma_store_fence()` 是 `fence.proxy.async` 类指令。这里牵涉一个 Hopper 概念叫 **proxy**：普通线程的 ld/st 走 generic proxy，TMA 走 async proxy，两条通路对 smem 的访问默认没有顺序保证。刚才 128 个线程用普通 store 写了 sY，接下来 TMA（async proxy）要读它，中间必须插一条 proxy fence，否则 TMA 可能读到旧数据。这是 TMA store 的固定搭配：普通写 → syncwarpgroup → tma_store_fence → TMA store。

store 本身还是单线程发起（每 warpgroup 的 leader），目标坐标里带上了 `ichunk`：splitk>1 时 TMA descriptor 描述的是 `[kSplitK, m, n]` 的 partial buffer（host 侧构造，见 0x4），每个 chunk 写自己那层；splitk=1 时第三维大小是 1，直接写最终输出。`tma_store_arrive()`（`cp.async.bulk.commit_group`）把这次 store 提交成一个 group，配合前面的 `tma_store_wait<0>` 使用。

## 3.10 split-k 收尾：flag 协议与自旋归约（第 340~405 行）

主循环里还夹着一段 flag 记账（在 r2s copy 和 TMA store 之间）：

```c++
      if constexpr (kSplitK > 1) {
        if (is_leader_in_warpgroup) {
          if (last_tile_m != -1 && last_tile_n != -1) {
            auto *split_flag = split_flag_ptr + last_tile_m * num_tile_n + last_tile_n;
            atomicAdd(split_flag, 1);
          }
          last_tile_m = itile_m;
          last_tile_n = itile_n;
        }
      }
```

注意加的是 `last_tile`——上一个任务的 flag，不是当前的。这是一个精巧的延迟设计：flag 表示"这个 tile 的 partial 结果已经落到 gmem 了"，但 TMA store 是异步的，发起（tma_store_arrive）不等于完成。什么时候能确认上一个 store 完成？就是本次 epilogue 开头那句 `tma_store_wait<0>()` 之后。所以每次 epilogue 先等上一个 store 退役，再给上一个 tile 加 flag，自己的 flag 留给下一轮。最后一个任务的 flag 在循环外补：

```c++
    if constexpr (kSplitK > 1) {
      cute::tma_store_wait<0>();

      fence_async_global();
      __threadfence();
      syncwarpgroup(iwarpgroup);

      if (is_leader_in_warpgroup) {
        if (last_tile_m != -1 && last_tile_n != -1) {
          auto *split_flag = split_flag_ptr + last_tile_m * num_tile_n + last_tile_n;
          atomicAdd(split_flag, 1);
        }
      }

      bar_sync<128 * kWarpGroupN>(kWarpGroupN);
```

这里比循环内多了 `fence_async_global()`（async proxy 到 global 的 fence）加 `__threadfence()`，保证 TMA 写出的数据对其它 CTA 可见之后 flag 才可见——因为接下来读这些数据的是别的 CTA。`bar_sync<128 * kWarpGroupN>(kWarpGroupN)` 同步所有 math warpgroup（用了一个编号为 kWarpGroupN 的 named barrier，线程数是全部 math 线程），producer 不参与。

```c++
      iblock = blockIdx.x;
      __threadfence();
      using NVTout = std::conditional_t<std::is_same_v<Tout, float>, float, __nv_bfloat16>;
      while (true) {
        int itile_m, itile_n;
        reduce_flat_divider(itile_m, itile_n, iblock);

        if (itile_m >= num_tile_m) {
          break;
        }
        iblock += gridDim.x;
        auto *split_flag = split_flag_ptr + itile_m * num_tile_n + itile_n;
        while (load_global_volatile(split_flag) != kSplitK * kWarpGroupN) {
        }
        splitk_reduce<NVTout, kTileM, kTileN * kWarpGroupN, kSplitK, 128 * kWarpGroupN / 32>(
            reinterpret_cast<NVTout *>(y_ptr), splitk_y_ptr, m, n, itile_m, itile_n);
        bar_sync<128 * kWarpGroupN>(kWarpGroupN);
        // reset flag
        if (is_leader_in_warpgroup && iwarpgroup == 0) {
          *split_flag = 0;
        }
      }
    }
  }
```

所有 GEMM 任务做完后，全部 CTA 的 math warpgroup 转身变成归约器，再次 persistent 式地领 (m, n) tile，自旋等对应 flag 攒到 `kSplitK * kWarpGroupN`（每个 tile 有 kSplitK 个 chunk，每个 chunk 由一个 CTA 算，CTA 里 kWarpGroupN 个 warpgroup 各 arrive 一次，所以是乘积），然后调 3.3 节的 `splitk_reduce` 把 partial 加起来写最终输出。自旋读用 `ld.volatile` 绕过缓存拿最新值。

归约完把 flag 清回 0。这个细节在工程上很重要：flag buffer 用完即恢复初始状态，意味着同一块 buffer 可以无限次复用，放进 CUDA graph 反复 replay 也不需要额外的 memset——graph 场景里这省掉了一次 kernel launch。

常规 split-k 是两个 kernel（GEMM 写 partial + 单独的 reduce kernel），这里合成一个，省一次 launch 开销和一次全局同步的代价，换来的是自旋等待和 flag 协议的复杂度。对 router GEMM 这种单次几十微秒的 kernel，launch 开销占比不小，这个 trade-off 是划算的。

# 0x4. host 侧 launcher（第 411~475 行）

```c++
template <typename Tin, typename Tout, int kTileM, int kTileN, int kTileK, int kStage,
          int kWarpGroupN, int kSplitK>
void launch_gemm_bf16xfp32_kernel(void *y_ptr, void *splitk_y_ptr, void *split_flag_ptr,
                                  const void *x_ptr, const void *w_high_ptr, const void *w_low_ptr,
                                  int m, int n, int k, float scale, cudaStream_t stream) {
  using namespace cute;

  constexpr int kBlockSwizzle = 4;

  using TY = std::conditional_t<(kSplitK > 1), float, Tout>;

  auto X = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(x_ptr)), make_shape(m, k),
                       make_stride(k, Int<1>{}));
  auto W_HIGH = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(w_high_ptr)),
                            make_shape(n, k), make_stride(k, Int<1>{}));
  auto W_LOW = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(w_low_ptr)),
                           make_shape(n, k), make_stride(k, Int<1>{}));
  auto Y = make_tensor(make_gmem_ptr(reinterpret_cast<TY *>(kSplitK > 1 ? splitk_y_ptr : y_ptr)),
                       make_shape(n, m, kSplitK), make_stride(Int<1>{}, n, n * m));
```

host 侧先把 gmem 张量描述出来。X 和 W 都是行优先的 (行, k) 布局。Y 有意思：行优先的 `y[m, n]` 被描述成 shape (n, m, kSplitK)、stride (1, n, n*m)——即以 n 为最快维的"转置视角"，因为 accumulator 的逻辑形状是 (kTileN, kTileM)（3.6 节的 gY），smem 里的 y tile 也是这个方向，TMA store 的 descriptor 跟着这个方向定义，数学上和写 `y[m][n]` 是同一块内存。第三维 kSplitK 对应 partial buffer 的层，splitk=1 时指向真正的输出且这维是 1。

```c++
  auto slayout_x = tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{},
                                 make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{}));
  auto slayout_w = tile_to_shape(
      GMMA::Layout_K_SW128_Atom<Tin>{},
      make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kWarpGroupN>{}, Int<2>{}, Int<kStage>{}));
  auto slayout_y = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TY>{},
                                 make_shape(Int<kTileN>{}, Int<kTileM>{}, Int<kWarpGroupN>{}));
```

smem layout 用 swizzle atom 拼出来。`GMMA::Layout_K_SW128_Atom<bf16>` 是 CuTe 预定义的"K 优先、128 字节 swizzle"的最小布局单元，`tile_to_shape` 把它平铺到目标形状。swizzle 简单补一句：smem 有 32 个 bank，GMMA/ldmatrix 这类按行列交错访问的模式如果直接线性存放会大量 bank conflict，swizzle 把地址的某几位异或起来打散 bank 归属，让任意行/列访问都无冲突。用哪种 swizzle 不是自由选择：TMA descriptor 和 GMMA matrix descriptor 里都要编码同一种 swizzle 模式，三方（TMA 写、GMMA 读、layout 本身）必须一致，CuTe 的这套 atom 就是保证一致性的封装。SW128 是 swizzle 粒度 128 字节，对应 kTileK 至少 64 个 bf16。

注意 `slayout_w` 的 shape 是 5 维：(kTileN, kTileK, kWarpGroupN, 2, kStage)，第 4 维的 2 就是低位/高位权重两份，和 kernel 里 `tWLs(_, 0, 0, wg, kWLIdx, ismem_write)` 的下标一一对应。y 的 layout 用 MN 优先的 atom，因为它要被 TMA store 按 n 方向连续读。

```c++
  int shm_xw = sizeof(Tin) * (cosize(slayout_x) + cosize(slayout_w));
  int shm_y = sizeof(TY) * cosize(slayout_y);
  int shm_size = shm_xw + shm_y;

  auto tma_x = make_tma_copy(SM90_TMA_LOAD{}, X, take<0, 2>(slayout_x));
  auto tma_wh = make_tma_copy(SM90_TMA_LOAD{}, W_HIGH, take<0, 2>(slayout_w));
  auto tma_wl = make_tma_copy(SM90_TMA_LOAD{}, W_LOW, take<0, 2>(slayout_w));
  auto tma_y = make_tma_copy(SM90_TMA_STORE{}, Y, take<0, 2>(slayout_y));
```

`make_tma_copy(op, gmem_tensor, smem_layout)` 在 host 侧生成 TMA descriptor（内部调 CUDA driver 的 `cuTensorMapEncodeTiled`），把 gmem 的形状 stride 和 smem 的 swizzle 布局都编码进去。`take<0, 2>` 取 layout 的前两维（单个 tile 的形状），因为一次 TMA 只搬一个 tile，多的维度（stage、wgn、hl）是 kernel 里选 buffer 用的。

举个 smem 用量的例子，tile64 配置（kTileM=64, kTileN=64, kTileK=64, kStage=3, wgn=1, splitk=1）：x 是 64x64x2B x3 级 = 24KB，w 是 64x64x2B x2(高低) x3 级 = 48KB，y 是 64x64x4B = 16KB，共 88KB。最大的配置 tile16/wgn2/stage3 约 212KB，逼近 H200 的 227KB 上限，这也是为什么 stage 不能随便加。

```c++
  using MMA_ATOM =
      std::conditional_t<kTileM == 64, SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>,
                         SM90_64x16x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>>;

  auto tiled_mma = make_tiled_mma(MMA_ATOM{});
```

MMA atom 的命名规则：`SM90_64x64x16_F32BF16BF16_SS<MajorA, MajorB>`，即 m64n64k16、D 是 F32、A/B 是 BF16、两个操作数都来自 smem（SS），A、B 都按 K 优先存放。3.8 节说过 M=64 给权重、N 给 token，所以 kTileM=16 的小 batch 配置用 N=16 的 atom。顺便说 GMMA 的 N 维支持 8 到 256 一堆档位，N 越大单指令吞吐效率越高，这个 kernel 最大只用到 N=64，大 m 场景其实还有再加大的空间（用 N=128/256 的 atom），这是 H200 上没做完的事之一。

```c++
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

  int num_tile_m = (m + kTileM - 1) / kTileM;
  int num_tile_n = (n + (kTileN * kWarpGroupN) - 1) / (kTileN * kWarpGroupN) * kSplitK;
  int num_tile = num_tile_m * num_tile_n;
  int num_tile_bxn = kBlockSwizzle * num_tile_n;
  cutlass::FastDivmod swizzle_divider(num_tile_bxn);
  cutlass::FastDivmod flat_divider(num_tile_n);
  cutlass::FastDivmod reduce_flat_divider(num_tile_n / kSplitK);

  dim3 block(size(tiled_mma) * kWarpGroupN + 128);
  dim3 grid(std::min(get_sm_count(), num_tile));

  kernel<<<grid, block, shm_size, stream>>>(...);
```

收尾的常规操作：动态 smem 超过 48KB 必须先 `cudaFuncSetAttribute` 声明；`num_tile_n` 乘 kSplitK 把 K 分片编进任务空间（3.2 节解码的对应物）；三个 FastDivmod 预计算好传进去；block 大小 = math 线程数（`size(tiled_mma)` 是 128）x wgn + 128 个 producer；grid 取 SM 数和任务数的较小值，这就是 persistent kernel 的 grid。

顺便提 TMA 的边界处理：m、n 不是 tile 整数倍时不需要任何特判，TMA descriptor 里记录了张量真实边界，越界部分读时自动填 0（补 0 参与 GEMM 不影响结果）、写时自动丢弃。对比 Ampere 时代每个 load/store 都要 predicate 的写法，这是 TMA 隐性省掉的一大坨代码。

# 0x5. 配置分发（第 477~554 行）

```c++
template <int kTileM_, int kTileN_, int kTileK_, int kStage_, int kWGN_, int kSplitK_>
struct LaunchCfg { ... };

bool gemm_bf16xfp32_async(void *y_ptr, ..., int splitk, int kTileM, int wgn,
                          cudaStream_t stream) {
  ...
  if (kTileM == 64) {
    switch (splitk) {
      case 4:  return launch_tile64_wgn1_fixed(cute::Int<5>{}, cute::Int<4>{});   // stage=5
      case 2:  return launch_tile64_wgn1_fixed(cute::Int<3>{}, cute::Int<2>{});   // stage=3
      default: return launch_tile64_wgn1_fixed(cute::Int<3>{}, cute::Int<1>{});
    }
  }
  if (wgn == 1) return launch_tile16(cute::Int<1>{});   // splitk in {1,2,4,8}, stage=3
  return launch_tile16(cute::Int<2>{});
}
```

模板参数必须编译期确定，所以运行时的 (tile_m, wgn, splitk) 要经过一个 switch 表映射到具体实例。全部合法组合是 11 种：tile16（kTileK=128, stage=3）x wgn{1,2} x splitk{1,2,4,8} 共 8 种，加 tile64（kTileN=64, kTileK=64, wgn 固定 1）x splitk{1(stage3), 2(stage3), 4(stage5)} 共 3 种。选哪种由 `entry.cc` 里的启发式决定，按 norm_m（归一化的工作量）分段查表，那部分逻辑我们在 SGLang 里翻译成了 Python，这里不展开。

注意 tile64 路径把 wgn 写死成 1 了。kernel 模板本身对 kWarpGroupN 是泛型的，tile64 + wgn=2 编译运行都没问题（splitk=1 时），只是上游从来没暴露这个组合——H20 上它确实不会更快。我们在 H200 上把它加回了配置空间，大 m 段拿到 1.2~1.6x，这是"调度表跟着旧硬件走了"的一个实例。另外实测 tile64 + wgn=2 + splitk=2 会崩（AcceleratorError），flag 协议或归约路径上应该有没覆盖到的假设，用这个 kernel 的话别启用这个组合。

# 0x6. 设计与 trick 总结

把这个 kernel 用到的技术点列一遍，作为复习：

1. **bf16x2 补偿拆分**：fp32 权重拆高低两个 bf16，两批 GMMA 加一行 FFMA 融合，用 2 倍 bf16 算力换 fp32 精度，避开 15 倍慢的 FFMA 路径。
2. **warp specialization**：1 个 producer warpgroup（实际 1 个线程）发 TMA，1~2 个 math warpgroup 发 GMMA，mbarrier 传递 buffer 所有权。
3. **寄存器重分配**：producer `setmaxnreg` 降到 24，math 升到 168，occupancy 固定为 1，用流水线而不是 CTA 切换掩延迟。
4. **TMA + expect_tx**：数据搬运和"数据到齐"通知全部硬件化，producer 代码是单线程串行逻辑；边界补 0 免 predicate。
5. **多级流水线 + 相位协议**：kStage 级 smem buffer，本地 phase 随索引回绕翻转；x 共享 buffer 用多 arrive 计数，权重按 warpgroup、按高低位独立 barrier，同步粒度做到最细。
6. **SS 模式 GMMA**：操作数不进寄存器，寄存器只放 accumulator；A=权重（M 固定 64）、B=激活（N 随 batch 选 16/64）。
7. **STSM + TMA store 的 epilogue**：寄存器按 stmatrix 布局写 smem，proxy fence 之后 TMA 整 tile 写出，全程合并访问。
8. **persistent kernel + block swizzle**：grid 等于 SM 数，任务号解码分配，4 行一组的 swizzle 提高权重的 L2 复用。
9. **单 kernel split-k**：K 分片编码进任务空间，partial 写 `[splitk, m, n]` buffer，flag 延迟一拍记账（等 TMA store 退役才置位），全体 CTA 自旋归约，flag 用完清零保证 CUDA graph 可复用。

已知的改进空间（H200 视角）：每 K tile 两次 `warpgroup_wait<0>` 全排空应改成延迟释放；大 m 可以上 N=128/256 的 GMMA atom 和更大的 tile；权重 tile 被 m 方向所有 CTA 重复从 L2 拉取，可以用 2-CTA cluster 的 TMA multicast 省一半。这几点在 H200 上对应大约还有 2 倍的差距（配置调优后 MFU 约 48%）。

对新手来说，我觉得这个 kernel 值得反复读的原因是它把 Hopper GEMM 的完整骨架压缩到了一个文件里，没有模板泛化的包袱。读懂它之后再去看 CUTLASS 3.x 的 `CollectiveMma` 和 `PipelineTmaAsync`，会发现那些几千行的模板做的就是这里手写的这些事。
