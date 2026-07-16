# 0x0. 前言

记录一下腾讯 HPC-Ops（https://github.com/Tencent/hpc-ops ，MIT 协议）里的 `gemm_bf16xfp32` 这个 kernel。它解决的是 bf16 激活 × fp32 权重的 GEMM 问题，思路是把 FP32 权重拆成两个 BF16 走 Tensor Core，最初是在 H20 上给 Hunyuan3 的 router 做加速用的。最近我把它集成进了 SGLang，用在 LongCat-Flash 的 router 上，并在 H200 上重新调了参数，对应 PR：https://github.com/sgl-project/sglang/pull/30247 。

这个 kernel 针对的场景在 MoE 模型里很常见：router 权重是 FP32 的，激活是 BF16，每一层 MoE 都要做一个 bf16 x fp32 的 GEMM。SGLang 里的 Hunyuan3 gate 是 `params_dtype=torch.float32`，DeepSeek V4 和 LongCat-Flash 的 router 也一样。这类 GEMM 的 shape 很瘦，以 LongCat-Flash 为例，Chat 是 `(m, 6144) x (768, 6144)`，Lite 是 `(m, 3072) x (384, 3072)`，n 只有几百，但每层都要做一次，m 大的时候（prefill 或者大 batch decode）在 profile 里能看到明显占比。

集成方式上，我最开始是在 SGLang 里 `import hpc` 直接调用，后来觉得给 SGLang 加一个第三方 wheel 依赖不合适，就把 kernel 源码拷进了 `sglang/jit_kernel`，用 SGLang 自己的 TVM-FFI JIT 基础设施按需编译，源文件头部保留了 MIT 署名。

另外这个 kernel 上游的开发和调参都是在 H20 上做的，我们的测试机器是 H200，两张卡形态差别很大，后面在 H200 上重新调了一轮参数，又拿到了 1.2x-1.6x。下面按原理、kernel 实现、集成方式、H200 调参的顺序记录。

# 0x1. 效果

H200 单卡微基准，一次 router GEMM 的中位数耗时（`triton.testing.do_bench`），cublas 是优化前 SGLang 的路径（`torch.mm(x.float(), w.t())`）：

| model | m | cublas | JIT kernel（H20配置） | JIT kernel（H200调优后） | 对cublas加速 |
|---|---:|---:|---:|---:|---:|
| Chat | 64 | 45.1us | 20.3us | 16.7us | 2.70x |
| Chat | 128 | 52.4us | 27.7us | 22.2us | 2.37x |
| Chat | 1024 | 237.2us | 80.7us | 50.5us | 4.69x |
| Chat | 8192 | 1712.0us | 438.2us | 320.2us | 5.35x |
| Lite | 128 | 29.4us | 16.6us | 12.6us | 2.33x |
| Lite | 512 | 51.1us | 24.5us | 20.8us | 2.45x |
| Lite | 2048 | 138.9us | 43.4us | 29.5us | 4.71x |
| Lite | 8192 | 490.0us | 116.9us | 79.3us | 6.18x |

端到端方面，LongCat-Flash-Lite-FP8 在 H200 TP8 跑全量 GSM8K，输出吞吐从 2258 token/s 提升到 2431 token/s（+7.6%），精度 0.9247 → 0.9269 没有回退。`input=8192, output=1` 的 serving 测试里 mean E2E 从 114.6ms 降到 101.0ms。详细数据和启动命令都贴在 PR 里。

# 0x2. 原理：把 FP32 权重拆成两个 BF16

先看 LongCat router 在模型代码里的样子（`python/sglang/srt/models/longcat_flash.py`）：

```python
class LongcatFlashRouter(nn.Module):
    def __init__(self, config, zero_expert_num=0, rounter_params_dtype=torch.float32, ...):
        self.classifier = ReplicatedLinear(
            config.hidden_size,
            self.n_routed_experts,
            bias=config.router_bias,
            params_dtype=rounter_params_dtype,   # FP32
            ...
        )

    def forward(self, hidden_states):
        # hidden_states 是 bf16，classifier.weight 是 fp32
        logits, _ = self.classifier(hidden_states.to(self.rounter_params_dtype))
        return logits
```

硬件上没有 bf16 x fp32 的 Tensor Core 指令，cublas 算这个 GEMM 只能要么把激活 upcast 成 fp32 走 FFMA（CUDA core 上的标量 fma），要么把权重 downcast 成 bf16 走 Tensor Core。前者慢，H200 的 FP32 FFMA 峰值只有约 67 TFLOPS，BF16 Tensor Core 是 989 TFLOPS，差 15 倍。上面表里 Chat m=8192 的 cublas 1712us 折算下来约 45 TFLOPS，确实就是 FFMA 的水平。后者精度不够，router 的输出要做 top-k 专家选择和 sigmoid 加权，logits 有偏差专家选择就可能翻转。

bf16 直接 cast 丢多少精度可以算一下。bf16 只有 8 位有效尾数（1 位隐含 + 7 位存储），fp32 是 24 位，cast 的相对误差是 2^-8 量级，在 k=6144 的累加下：

```python
>>> import torch
>>> torch.manual_seed(0)
>>> x = torch.randn(64, 6144, device="cuda").bfloat16()
>>> w = torch.randn(768, 6144, device="cuda")
>>> ref = x.float() @ w.t()
>>> naive = torch.mm(x, w.bfloat16().t(), out_dtype=torch.float32)
>>> (naive - ref).abs().max()
tensor(0.5775, device='cuda:0')
```

logits 的量级本身也就是几十，最大误差 0.58 对排序来说不安全。

HPC-Ops 这个 kernel 用的是补偿拆分的办法，把 fp32 权重拆成高位和低位两个 bf16：

```python
scale = 1 / 256
w_high = w.to(torch.bfloat16)                               # 高 8 位尾数
w_low  = ((w - w_high.float()) / scale).to(torch.bfloat16)  # 残差再抓 8 位
```

w_high 吃掉前 8 位尾数之后，残差 `w - w_high` 的量级只有原来的约 2^-9。这个残差直接存成 bf16 也可以，但先除以 scale（乘 256）拉回和 w 同量级再存更好，因为 scale 是 2 的幂，乘除只动指数位完全无损，bf16 的 8 位尾数就全部用在残差本身上。两段拼起来相当于保留了 fp32 前 16 位左右的尾数。GEMM 变成两个纯 bf16 GEMM 加一个融合：

```python
y = (x @ w_high.t()) + scale * (x @ w_low.t())    # 两个都是 bf16 Tensor Core GEMM, fp32 累加
```

还是刚才那组数据，误差降了两个多数量级：

```python
>>> w_low = ((w - w_high.float()) / scale).to(torch.bfloat16)
>>> split = torch.mm(x, w_high.t(), out_dtype=torch.float32) \
...       + scale * torch.mm(x, w_low.t(), out_dtype=torch.float32)
>>> (split - ref).abs().max()
tensor(0.0012, device='cuda:0')
```

0.577 vs 0.0012，对 router 来说这个精度和 fp32 没有实际区别。代价是计算量翻倍（两个 GEMM），权重显存翻倍（两份 bf16 等于原 fp32 的大小），但是从 67 TFLOPS 的 FFMA 换到了 989 TFLOPS 的 Tensor Core，这个代价是划算的。权重拆分是一次性的，SGLang 里挂在权重 tensor 的属性上做了缓存，用 `data_ptr + _version + shape` 当 key，权重被原地改动时会自动失效重拆。

上面两个 `torch.mm` 的写法已经能跑，也已经比 FFMA 快很多，但是它有两次 kernel launch、激活要完整读两遍、后面还跟一个 add kernel，对这种瘦 GEMM 来说开销不小，而且小 m 的 decode 场景下 cublas 对 n 只有几百的形状本来就调度不好。手写 kernel 可以一次 TMA 读进激活，在片上把两个 GMMA 的结果直接融合掉，这就是下一节的内容。

# 0x3. CUDA kernel 实现

kernel 源码在 hpc-ops 的 `src/gemm/sm90/gemm_bf16xfp32.cu`，拷进 SGLang 之后的位置是 `python/sglang/jit_kernel/csrc/gemm/gemm_bf16xfp32_sm90.cuh`，逐行移植没有改动逻辑。SM90 专属（GMMA + TMA），编译目标 sm_90a。整体是 Hopper 上标准的 warp specialization 结构，每个 CTA 里有 `kWarpGroupN` 个 math warpgroup（128 线程一个）加一个 producer warpgroup，producer 只发 TMA，math 只做 GMMA：

```c++
if (idx >= kWarpGroupN * 128) {
    // producer warpgroup：只发 TMA，把寄存器让出来
    cutlass::arch::warpgroup_reg_dealloc<24>();
    ...
} else {
    // math warpgroup：把寄存器要回来
    cutlass::arch::warpgroup_reg_alloc<168>();
    ...
}
```

producer 的循环里，每个 K tile 要发三份 TMA：激活 x 一份、低位权重一份、高位权重一份。x 是所有 math warpgroup 共享的（`writable_x` 的 barrier 初始化成 `kWarpGroupN` 个 arrive），两份权重按 warpgroup 各自独立同步。共享内存里权重 buffer 的 layout 是 `(kTileN, kTileK, kWGN, 2, kStage)`，中间那个 2 就是 high/low 两份：

```c++
for (int itile_k = ichunk; itile_k < ntile_k; itile_k += kSplitK) {
    wait_barrier(writable_x[ismem_write], phase);
    cute::copy(tma_x.with(readable_x[ismem_write]), ...);
    set_barrier_transaction_bytes(readable_x[ismem_write], kTransactionBytesX);
    // 先发低位权重
    for (int wg = 0; wg < kWarpGroupN; ++wg) {
        wait_barrier(writable_w[ismem_write][wg][kWLIdx], phase);
        cute::copy(tma_wl.with(readable_w[ismem_write][wg][kWLIdx]), ...);
        ...
    }
    // 再发高位权重
    for (int wg = 0; wg < kWarpGroupN; ++wg) { ... }
    ...
}
```

math warpgroup 这边，每个 K tile 做两批 GMMA，低位一批、高位一批，各自累加到独立的 fp32 accumulator 上：

```c++
// mma low
wait_barrier(readable_w[ismem_read][iwarpgroup][kWLIdx], phase);
warpgroup_arrive();
for (int ik = 0; ik < size<2>(tXr); ++ik) {
    cute::gemm(tiled_mma, tWr(_, _, ik, iwarpgroup, kWLIdx, ismem_read),
               tXr(_, _, ik, ismem_read), tYr_low(_, _, _));
}
warpgroup_commit_batch();
warpgroup_wait<0>();
if (elected_idx_in_warpgroup) {
    arrive_barrier(writable_w[ismem_read][iwarpgroup][kWLIdx]);  // 释放低位权重 buffer
}

// mma high，同样一套，然后释放 x 和高位权重 buffer
...
```

注意每批 GMMA 后面都跟了一个 `warpgroup_wait<0>`，也就是把 GMMA 流水线完全排空之后才释放 smem barrier，每个 K tile 有两次排空。这个写法在 H20 上没有问题，因为 H20 的 Tensor Core 慢，GMMA 本身就是瓶颈，排空的开销被完全掩盖；但在 H200 上它是主要的气泡来源，0x5 节会展开。

K 循环走完后 epilogue 把两个 accumulator 融合，就是 0x2 节的公式，一行乘加：

```c++
for (int i = 0; i < size(tYr_low); ++i) {
    tYrh(i) = (TY)(tYr_low(i) * scale + tYr_high(i));
}
```

融合结果先经 STSM/STS 写到 smem，再用 TMA store 写回 global。MMA atom 按 tile 配置选，大 m 用 `SM90_64x64x16_F32BF16BF16_SS`（tile 64x64），小 m 用 `SM90_64x16x16`（tile 16x64）。这里权重是 GMMA 的 A 操作数、激活是 B 操作数，GMMA 的 N 维对应的是 token 数那一维，m 小的时候用窄 atom 避免浪费。

split-k 的实现值得说一下。router GEMM 在 decode 时 m 很小，`ceil(m/tile_m) * ceil(n/tile_n)` 可能只有几十个 tile，喂不饱 GPU，需要把 K 维也切开让更多 CTA 有活干。常规的 split-k 是主 kernel 写 partial 结果，再跑一个 reduce kernel，这里为了省第二次 launch 做成了单 kernel 自旋版本：

```c++
// 每个 tile 的所有 split 分片完成时 atomicAdd 一个 flag
if (is_leader_in_warpgroup) {
    auto* split_flag = split_flag_ptr + last_tile_m * num_tile_n + last_tile_n;
    atomicAdd(split_flag, 1);
}

// K 循环全部做完后，所有 CTA 转身去做 reduce，自旋等 flag 凑齐
while (load_global_volatile(split_flag) != kSplitK * kWarpGroupN) {
}
splitk_reduce<...>(y_ptr, splitk_y_ptr, m, n, itile_m, itile_n);
// reduce 完把 flag 清零
if (is_leader_in_warpgroup && iwarpgroup == 0) {
    *split_flag = 0;
}
```

partial 结果放在一块 `[split_k, m, n]` 的 fp32 workspace 里。flag 用完自己清零，所以同一块 workspace 在 CUDA graph 里反复 replay 不需要重新 memset。

最外层是 persistent kernel，grid 直接开 `min(sm_count, num_tile)` 个 CTA，每个 CTA 用 `get_next_tile` 按 `blockIdx.x + i * gridDim.x` 领任务，里面带一个 kBlockSwizzle=4 的块内 swizzle 提高 L2 命中。整个 kernel 只有 500 多行。

# 0x4. 集成进 sglang jit_kernel

SGLang 的 `jit_kernel` 目录下每个 kernel 是一个 .cuh 加一个 python 包装的组织方式，编译走 TVM-FFI 的 `load_jit`，第一次调用时 ninja 编译出 .so 落到磁盘缓存，之后直接复用。这个 kernel 的入口把所有 launch 配置做成了模板参数：

```python
@cache_once
def _jit_gemm_bf16xfp32_module(tile_m, tile_n, tile_k, stage, wgn, split_k, fp32_out):
    args = make_cpp_args(tile_m, tile_n, tile_k, stage, wgn, split_k, fp32_out)
    with override_jit_cuda_arch(9, 0, "a"):
        return load_jit(
            "gemm_bf16xfp32", *args,
            cuda_files=["gemm/gemm_bf16xfp32_sm90.cuh"],
            cuda_wrappers=[("gemm_bf16xfp32", f"GemmBf16xFp32Kernel<{args}>::run")],
            extra_dependencies=["cutlass"],
        )
```

一种配置编一个模块。hpc-ops 原来是 AOT 编译，一个 .so 里塞了全部 11 种配置的实例化，JIT 化之后只编实际用到的配置，服务跑起来一般只会命中三四种。

hpc-ops 里按 shape 选配置的启发式（`entry.cc` 的 `select_config`）原来是 C++，这次把它整个搬到了 Python。为了确认搬得没错，写了个脚本把 C++ 逻辑重新照抄一份当参考，暴力对比了 23535 个 (m, n, k) 组合，两边逐位一致。这个启发式把 workload 归一化成一个 `norm_m`，再按阈值分段选 split_k、warpgroup 数和 tile 大小：

```python
def _select_launch_config(m, n, k, sm_count=None):
    norm_m = (m * n * 4096 + 192 * k - 1) // (192 * k)
    if 624 < norm_m <= 832:
        split_k, wgn, tile_m = 2, 1, 64
    elif 832 < norm_m <= 896:
        split_k, wgn, tile_m = 2, 2, 16
    ...
```

启发式放进 Python 最初是因为 TVM-FFI 的 C++ 入口里没有 torch allocator，split-k 的 workspace 需要 Python 按配置算好 shape 传进去，配置选择自然也就跟着过来了。这个决定对后面 H200 调参帮助很大。

模型侧的接入很简单，`linear_bf16_fp32` 加了一个显式的 `jit_kernel_min_m` 参数，LongCat 按 benchmark 过的 shape 直接指定门槛，不需要用户设任何环境变量：

```python
_LONGCAT_FLASH_ROUTER_JIT_GEMM_MIN_M = {
    (6144, 768): 64,    # LongCat-Flash-Chat
    (3072, 384): 128,   # LongCat-Flash-Lite
}

def forward(self, hidden_states):
    if (self.jit_kernel_min_m is not None
            and self.rounter_params_dtype == torch.float32
            and self.classifier.bias is None):
        return linear_bf16_fp32(hidden_states, self.classifier.weight,
                                jit_kernel_min_m=self.jit_kernel_min_m)
    logits, _ = self.classifier(hidden_states.to(self.rounter_params_dtype))
    return logits
```

m 低于门槛或者 shape/dtype 不满足条件时自动回退 cublas。门槛是逐 shape benchmark 出来的，太小的 m 下整个操作是纯 memory bound，当时测下来 kernel 不是所有点都比 cublas 快，门槛就定在了确定有收益的位置。

# 0x5. H200 上的重新调参

前面说了这个 kernel 是在 H20 上开发和调参的，H20 和 H200 虽然都是 Hopper，形态差别很大：

| | H20 | H200 |
|---|---:|---:|
| SM 数 | 78 | 132 |
| BF16 Tensor Core | ~148 TFLOPS | ~989 TFLOPS |
| 显存带宽 | 4.0 TB/s | 4.8 TB/s |
| 算力带宽比 | 37 | 206 |

先做个 roofline 估算。Chat m=8192 这个 shape，两个 GEMM 总计 154.6 GFLOP，按 H200 的 BF16 峰值换算时间下限是 156us，实测 438us，MFU 只有 36%。反过来算，438us 对应约 353 TFLOPS 有效算力，是 H20 峰值的 2.4 倍。也就是说这个 kernel 在 H20 上 MFU 接近满，主循环里有多少气泡都无所谓，因为 GMMA 本身就是瓶颈；换到 H200 上 Tensor Core 快了 6.7 倍，气泡全部暴露出来了。

ncu 的数据（Chat m=8192，H20 配置）也印证了这一点：

```
Compute (SM) Throughput           37.01 %
Memory Throughput                 37.46 %
DRAM Throughput                    9.17 %
Active Warps Per Scheduler         1.25
Issued Warp Per Scheduler          0.19
No Eligible                       80.86 %
```

计算和访存都远没打满，81% 的周期发不出指令，标准的 latency bound。气泡来源就是 0x3 节说的每个 K tile 两次 `warpgroup_wait<0>` 排空，加上大 m 配置下每个 SM 只有一个 math warpgroup（一个 CTA 256 线程，寄存器 168 一卡 occupancy 就是 1），排空期间没有任何 warp 能补位。

改主循环是个大工程，先做便宜的事情。启发式已经在 Python 里了，做全配置扫描就是写个 for 循环，不需要重新编译任何东西。13 种配置 x 18 个 m 值 x 两个 shape 全部扫一遍，十来分钟出结果。摘几行 Lite 的数据（单位 us，c6=(16,64,128,3,2,4) 表示 tile16/stage3/wgn2/splitk4，其它类推）：

```
Lite m=128    heur=c0(splitk1,wgn1) 16.64   best=c6(wgn2,splitk4) 12.67   gain=1.31x
Lite m=256    heur=c9(tile64,splitk2) 21.22 best=c7(wgn2,splitk8) 15.84   gain=1.34x
Lite m=512    heur=c8(tile64,splitk1) 24.64 best=c6(wgn2,splitk4) 20.77   gain=1.19x
Lite m=2048   heur=c8(tile64,wgn1) 43.55    best=c12(tile64,wgn2) 29.31   gain=1.49x
Lite m=8192   heur=c8(tile64,wgn1) 117.23   best=c12(tile64,wgn2) 80.80   gain=1.45x
```

扫描结果里有两个规律，都能对上 H20 和 H200 的硬件差异。

第一个是中小 m 段启发式给的并行度不够。split-k 的作用就是 tile 数少于 SM 数时把 K 切开凑 CTA 数，多少 CTA 算够是跟着 SM 数走的。启发式里有一行 `kTargetTiles = 64`，64 约等于 0.82 x 78，明显是照着 H20 的 78 个 SM 定的，H200 有 132 个 SM，等比例应该是 108 左右。所以在 H200 上 m=128~512 这段该多切的 K 没切、该开 wgn=2 的没开。比如 Lite m=512 按启发式选 tile64/splitk1，一共 48 个 CTA，132 个 SM 只用了 36%。

第二个规律是 tile64 + wgn=2 这个组合在大 m 段全面胜出。kernel 模板本身支持这个组合（`kWarpGroupN` 是模板参数，代码是泛型的），但 hpc-ops 的调度逻辑在 tile64 路径上把 wgn 写死成了 1，这个组合从来没被生成过。在 H200 上编出来一测，大 m 段直接 1.2~1.6x：每个 CTA 两个 math warpgroup，一个在 `warpgroup_wait<0>` 排空的时候另一个还能发 GMMA，气泡被盖掉一部分。Chat m=8192 从 438us 到 320us，MFU 从 36% 提到 48%。在 H20 上这个组合大概率不会更快，本来就顶着算力上限，多一个 warpgroup 只会多占 smem，上游没暴露它说得通。

这里有两个坑。tile64 + wgn2 只有 splitk=1 能用，配 splitk=2 直接 AcceleratorError 崩掉，估计是 split flag 计数或者 reduce 路径上有没考虑到的组合，没有深查，调优表里不放这个组合。stage=5 + wgn2 会超出 H200 的 227KB smem 上限，编译期就挂。

最后落地成一张按 shape 查的表，只对 LongCat 这两个精确的 (n, k) 生效，并且用 `sm_count >= 100` 做门槛，H20 和其它所有 shape 原样走上游启发式（23535 个 shape 的逐位对比重新跑过，不受影响）：

```python
_SM90_LARGE_SM_TUNED = {
    # (n, k) -> 按 m 分段的 (tile_m, tile_n, tile_k, stage, wgn, split_k)
    (768, 6144): (
        (0,   (16, 64, 128, 3, 2, 4)),
        (96,  (16, 64, 128, 3, 2, 8)),
        (256, (64, 64, 64, 3, 1, 2)),
        (384, (64, 64, 64, 5, 1, 4)),
        (768, (64, 64, 64, 4, 2, 1)),   # tile64 + 双 math warpgroup
    ),
    (384, 3072): (
        (0,    (16, 64, 128, 3, 2, 4)),
        (192,  (16, 64, 128, 3, 2, 8)),
        (384,  (16, 64, 128, 3, 2, 4)),
        (768,  (64, 64, 64, 3, 1, 1)),
        (1536, (64, 64, 64, 4, 2, 1)),
    ),
}
```

表里 Lite 在 768~1536 这段保持 tile64/wgn1 而不是 wgn2，因为 n=384 按 128 宽的 n tile 只能切出 3 条，m=1024 时 wgn2 只有 48 个 CTA，132 个 SM 喂不饱，实测反而倒退到 0.94x。wgn2 要 CTA 数够才能开。

调参前后的完整对比就是 0x1 那张表的后两列。精度方面在 TP4 上跑了 200 题 GSM8K 的 A/B，cublas 路径 0.80、JIT 路径 0.82，同一套 harness 下没有回退。

# 0x6. 后续

配置调完之后 MFU 停在 48%，再往上要改 kernel 代码。目前看有三个方向：主循环里那两次 `warpgroup_wait<0>` 改成 CUTLASS 式的延迟释放（`warpgroup_wait<N>` 留 1~2 批 GMMA 在飞行中，释放上一个 stage 的 buffer 而不是当前的）；给大 m 加 kTileM=128 的配置用上 M64N128 的 GMMA atom；以及 TMA multicast，现在同一列 n tile 的权重被 m 方向所有 CTA 各自从 L2 拉一遍，2-CTA cluster 多播能省一半。这些计划在后续 PR 里做。

另外把 C++ 启发式翻译成 Python 这件事，当时只是为了 workspace 分配方便，结果让新硬件上的重新调参变成了写 for 循环的事，这次 H200 上的收益一半来自这里。
