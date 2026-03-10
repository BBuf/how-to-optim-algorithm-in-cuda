> 本文由 GPT5.4 Medium Thinking 生成

# 从 microbenchmark 到 WGMMA、MoE 和 RMSNorm：顺着代码看 CUDA 调优

这里的材料主要有四块：`YHs_Sample` 里的 microbenchmark 和手写 `GEMM`，`mma_vs_wgmma.cu`，`sgl-kernel` 的一组推理算子，以及 `hp_rms_norm`。

## 先把机器摸清楚，不然后面很容易瞎调

`YHs_Sample/cuda/microbenchmark` 这组代码我很喜欢，因为它的目标非常干脆：先测机器，不急着写业务 kernel。这个顺序其实比很多人想的更重要。你如果连 DRAM、L2、shared memory 各自的边界都没有概念，后面看到 profiler 往往只能凭感觉猜。

比如 `dram_bandwidth.cu`，它没有写那种随手就能敲出来的普通 copy kernel，而是直接用了向量化的 `ld.global.cs.v4.b32` 和 `st.global.cs.v4.b32`。

```cpp
__device__ __forceinline__
uint4 ldg_cs(const void *ptr) {
    uint4 ret;
    asm volatile (
        "ld.global.cs.v4.b32 {%0, %1, %2, %3}, [%4];"
        : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
        : "l"(ptr)
    );
    return ret;
}

__device__ __forceinline__
void stg_cs(const uint4 &reg, void *ptr) {
    asm volatile (
        "st.global.cs.v4.b32 [%4], {%0, %1, %2, %3};"
        : : "r"(reg.x), "r"(reg.y), "r"(reg.z), "r"(reg.w), "l"(ptr)
    );
}
```

这里的 `v4.b32` 很直接，一次搬 16 字节，尽量把访存做得像样一点。`.cs` 也不是随手加上的，它是在尽量减少 cache 污染。说白了，这个 benchmark 想测的是一条比较接近“流式显存访问”的路径，而不是某种 cache 帮忙美化过的结果。

它的 host 端组织也很说明问题：

```cpp
const int MEMORY_OFFSET = (1u << 20) * 16;
const int BENCH_ITER = 100;
...
cudaMalloc(&ws, size_in_byte + MEMORY_OFFSET * BENCH_ITER);
...
for (int i = BENCH_ITER - 1; i >= 0; --i) {
    read_kernel<BLOCK, LDG_UNROLL><<<grid, BLOCK>>>(ws + i * MEMORY_OFFSET, nullptr);
}
```

这段里最关键的是 `MEMORY_OFFSET * BENCH_ITER`。它故意让每一轮打到不同地址区间，减少同一批 cache line 被反复命中。也就是说，它确实在往 DRAM 上限那个方向靠。`copy_kernel` 用 `grid / 2` 也一样，不是写法怪，而是因为 copy 同时占读写两条通路，地址也得拆开。

换句话说，这份 benchmark 不只是“测一下带宽”，它是在把读、写、拷贝三条路径刻意拆开。这个习惯很重要。很多人做 benchmark 喜欢最后报一个总数字，但那个数字往往解释不了任何问题。

`l2cache_latency.cu` 走的是另一条思路。这里关心的不是吞吐，而是一次访问到底要几个 cycle，所以它必须构造真正的依赖链：

```cpp
template <int ROUND>
__global__ __launch_bounds__(32, 1)
void l2_latency_kernel(const uint32_t *stride,
                       uint32_t *ret,
                       uint32_t *clk) {
    const char *ldg_ptr = reinterpret_cast<const char *>(stride + threadIdx.x);
    uint32_t val;

    asm volatile (
        "ld.global.cg.b32 %0, [%1];\n"
        : "=r"(val)
        : "l"(ldg_ptr)
        : "memory"
    );

    ldg_ptr += val;
    ...
    for (int i = 0; i < ROUND; ++i) {
        asm volatile (
            "ld.global.cg.b32 %0, [%1];\n"
            : "=r"(val)
            : "l"(ldg_ptr)
            : "memory"
        );
        ldg_ptr += val;
    }
}
```

这段代码里，`.cg`、`ldg_ptr += val`、前面的 warmup，其实是在做三件很具体的事：

- 尽量把访问约束到 global cache 路径。
- 保证每次 load 依赖上一次结果，不给硬件隐藏 latency 的空间。
- 先把 TLB 和 cache 热起来，再测稳态。

我觉得这类代码最有价值的地方，不是它最后打印了一行 `l2 cache latency xxx cycles`，而是它在教你怎么提问题。带宽和延迟不是一个东西，吞吐和单次代价也不是一个东西。你如果不先把问题问对，后面调 kernel 时就很容易把现象看反。

shared memory 那两份 benchmark 也是一样。`smem_latency.cu` 测的是一条依赖链上的 shared latency，`smem_bandwidth.cu` 测的是 `st.shared.v4.b32` 这种路径能推多高的吞吐。到了这里，其实已经有一个很实用的判断框架了：

如果一个 kernel 的 DRAM 吞吐远低于这类 benchmark，通常先别怪 Tensor Core，多半是访问方式、合并度或者中间搬运出了问题。要是 shared memory 指令很多，但吞吐还是起不来，那再去看 bank、thread map、LSU 发射、barrier，通常更有收获。

## 手写 GEMM 里真正难的，从来不是那几条 FMA

`YHs_Sample/cuda/gemm/sgemm.cu` 是一份很典型的老派高性能 `FP32 GEMM`。我反而觉得这种代码对理解 CUDA 更有帮助，因为它没有太多语法糖，问题全都摊在明面上：tile 怎么切，shared memory 怎么摆，双缓冲怎么轮，寄存器压力怎么控。

主 kernel 长这样：

```cpp
__global__ __launch_bounds__(256, 2)
void sgemm_128x128x8_kernel(const float *A,
                             const float *B,
                             float *C,
                             uint32_t m,
                             uint32_t n,
                             uint32_t k,
                             uint32_t A_ldg_step,
                             uint32_t B_ldg_step) {
    __shared__ __align__(16 * 1024) char smem[24 * 1024];
    float *A_smem = reinterpret_cast<float *>(smem);
    float *B_smem = reinterpret_cast<float *>(smem + 16 * 1024);
    ...
    float A_frag[2][8];
    float B_frag[2][8];
    float C_frag[8][8];
    ...
}
```

我第一次看这类 kernel 时，注意力老是放在 FMA 循环上。后来才发现，真正该盯的是外面那些看起来不那么“显眼”的东西。

`__launch_bounds__(256, 2)` 先给了一个约束：block 是 256 线程，而且作者希望每个 SM 至少保住两个 block 的并发。这其实已经在暗示寄存器预算了。接下来 `24KB` shared memory 不是随便凑出来的，`16KB A + 8KB B` 和 `128x128x8` 这个 tile 能对应上。再往里看，`A_frag[2][8]`、`B_frag[2][8]` 把双缓冲直接带到了寄存器层。

还有这两行：

```cpp
A_sts_addr ^= 0x2000;
B_sts_addr ^= 0x1000;
```

这种写法看着有点硬，但很有代表性。shared memory 的布局一旦规整到位，buffer 切换就可以退化成一次 XOR。它看起来只是省了几条整数指令，实际上是在说明一件事：当 kernel 已经写到这个程度，地址更新本身都值得单独考虑。

计算部分反而非常朴素：

```cpp
for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
        C_frag[i][j] += A_frag[k_frag % 2][i] * B_frag[k_frag % 2][j];
    }
}
```

也正因为它朴素，才更能看清 GEMM 里最麻烦的部分不在乘加，而在“下一轮要用的数据怎么提前送到位”。很多时候代码写半天，最后性能差距并不出在主循环那几行，而是出在 shared memory 布局、thread map、buffer 切换、写回重排这些更零碎的地方。

## `cp.async` 真正改变的，是写 kernel 的思路

`ampere_sgemm.cu` 和前面那份代码摆在一起看，差别就很明显了。这里已经不是传统的 load 到寄存器，再 store 到 shared memory，而是直接把 global 到 shared 的搬运交给 `cp.async`。

```cpp
__global__ __launch_bounds__(256)
void ampere_sgemm_128x256x8_kernel(
        const float *A,
        const float *B,
        float *C,
        uint32_t m,
        uint32_t n,
        uint32_t k,
        uint32_t B_ldg_step) {
    __shared__ __align__(16 * 1024) char smem[32 * 1024];
    float *A_smem = reinterpret_cast<float *>(smem);
    float *B_smem = reinterpret_cast<float *>(smem + 16 * 1024);
    ...
}
```

这里 block tile 已经拉到了 `128x256x8`，shared memory 也变成了 `32KB`。这不是单纯把 tile 做大，而是因为 `cp.async` 让更深一点的预取流水线变得划算。

封装出来的那条指令很关键：

```cpp
asm volatile (
    "{.reg .pred p;\n"
    " setp.ne.b32 p, %2, 0;\n"
    " @p cp.async.ca.shared.global.L2::128B [%0], [%1], 4;}\n"
    : : "r"(smem_addr), "l"(gmem_ptr), "r"((int)guard)
);
```

它至少说明了三件事。第一，global 到 shared 的搬运不再绕过通用寄存器。第二，这条路径已经开始明显跟 L2 扇区粒度对齐。第三，边界处理也被压到异步 copy 里了，不需要另外再拆一条慢路径。

所以 `Ampere` 之后，很多 kernel 的核心问题就不再是“shared memory 要不要双缓冲”，而是“搬运队列排几级，什么时候 wait，什么时候切下一拍”。这类代码读多了以后，你会很自然地开始关心 pipeline 深度、async stage、wait group，而不是一上来就盯 FMA 展开次数。

## 到 `Hopper`，你得把搬运、同步和计算一起看

`mma_vs_wgmma.cu` 很适合读这个变化，因为它把传统 `warp-level MMA` 和 `SM90` 上的 `WGMMA + TMA` 放在了同一个地方。对比着看，很多东西一下就清楚了。

`WGMMA` 那条路径的骨架是这样的：

```cpp
__global__ void wgmma_kernel(
    TiledMma mma,
    TensorA gA,
    TensorB gB,
    TensorC gC,
    TensorD gD,
    CUTLASS_GRID_CONSTANT TmaA const tmaA,
    CUTLASS_GRID_CONSTANT TmaB const tmaB) {
    extern __shared__ __align__(128) uint8_t shared_memory[];
    Tensor sA = make_tensor(make_smem_ptr(reinterpret_cast<T_IN*>(shared_memory)), SharedMemoryALayout{});
    Tensor sB = make_tensor(make_smem_ptr(reinterpret_cast<T_IN*>(shared_memory) + cosize(SharedMemoryALayout{})), SharedMemoryBLayout{});
    uint64_t* mbar = reinterpret_cast<uint64_t*>(shared_memory + sizeof(T_IN) * (cosize(SharedMemoryALayout{}) + cosize(SharedMemoryBLayout{})));
    ...
}
```

搬运不是线程自己 load，而是交给 TMA：

```cpp
ProducerBarType::arrive_and_expect_tx(mbar, tma_transaction_bytes);
copy(tmaA.with(*mbar), tAgA, tAsA);
copy(tmaB.with(*mbar), tBgB, tBsB);
ProducerBarType::wait(mbar, 0);
```

算的时候也不是普通 warp 粒度了：

```cpp
ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
...
gemm(mma, tCrA(_, _, _), tCrB(_, _, _), tCrAcc);
warpgroup_commit_batch();
warpgroup_wait<0>();
```

这类代码让我后来对 `Hopper` 的理解变得很具体。你不再只是写一群线程在做矩阵乘，而是在安排三件事怎么稳稳接起来：

- tensor 级搬运什么时候发起
- 事务什么时候真正可见
- warp-group 什么时候开始吃这批数据

这里最容易被低估的是同步语义。`wait(mbar, 0)` 等的不是“别的线程到了”，而是这次 transaction 真做完了。它跟 `__syncthreads()` 根本不是一回事。

顺着这个思路再往下想，所谓 `pingpong schedule` 其实也没那么神秘。说白了，就是别让搬运、主循环和收尾互相踩脚。当前 stage 在算，下一 stage 在搬，再前一 stage 在出结果，三件事能接住，Tensor Core 才不容易饿着。

## 真实推理系统里，难点往往在那堆高频小算子上

如果一直只看 GEMM，很容易得出一个错觉：只要矩阵乘够快，系统就差不多了。`sgl-kernel/csrc/common_extension.cc` 基本一眼就能把这个错觉打碎。这里注册出来的不是单个大算子，而是一串真正的热点原语：

```cpp
m.def("rmsnorm(Tensor! output, Tensor input, Tensor weight, float eps, bool enable_pdl) -> ()");
m.impl("rmsnorm", torch::kCUDA, &rmsnorm);

m.def("fused_add_rmsnorm(Tensor! input, Tensor! residual, Tensor weight, float eps, bool enable_pdl) -> ()");
m.impl("fused_add_rmsnorm", torch::kCUDA, &sgl_fused_add_rmsnorm);

m.def(
    "sgl_per_token_group_quant_8bit_v2(Tensor input, Tensor! output_q, Tensor! output_s, int group_size,"
    " float eps, float fp8_min, float fp8_max, bool scale_ue8m0, bool fuse_silu_and_mul, Tensor? masked_m) -> ()");
m.impl("sgl_per_token_group_quant_8bit_v2", torch::kCUDA, &sgl_per_token_group_quant_8bit_v2);

m.def(
    "fp8_blockwise_scaled_grouped_mm(Tensor output, Tensor a_ptrs, Tensor b_ptrs, Tensor out_ptrs, Tensor "
    "a_scales_ptrs, Tensor b_scales_ptrs, Tensor a, Tensor b, Tensor scales_a, Tensor scales_b, Tensor "
    "stride_a, Tensor stride_b, Tensor stride_c, Tensor layout_sfa, Tensor layout_sfb, Tensor problem_sizes, Tensor "
    "expert_offsets, Tensor workspace) -> ()");
m.impl("fp8_blockwise_scaled_grouped_mm", torch::kCUDA, &fp8_blockwise_scaled_grouped_mm);
```

这份注册表本身就已经说明很多事了。比如 `FP8 group GEMM` 并不是一条单独的矩阵乘，它前面还有量化、scale 布局、grouped dispatch。再比如 `enable_pdl` 会挂在 `RMSNorm` 这种高频算子上，说明 launch 方式和调度延迟已经被当成性能变量了。MoE 那边更明显，`problem_sizes`、`expert_offsets` 这类参数一摆出来，就知道瓶颈不只是算，还在“不同 shape 的问题怎么批量塞进同一条路径里”。

所以现代推理里的 CUDA 调优，很多时候不是在追一个“最快 kernel”，而是在修一整条热点链路。量化慢一点，后面的 grouped MM 就会空等；RMSNorm 多一次中间落地，整层 decode 都会被拉长；MoE dispatch 排不好，主 kernel 也会吃到一堆碎片化问题。

## `RMSNorm` 这种算子，看着小，真写起来一点都不小

`RMSNorm` 很容易被低估。第一次看它，很多人都会觉得不过是 reduce 一下，再乘个 scale。但真放到推理里，你马上就会发现它是个扎扎实实的热点：每层都要做，token 级高频调用，hidden size 还常常是 `8192`、`16384`。

`sgl-kernel` 在这件事上做得很实在，直接把 fused 路径摆在前面：

```cpp
void sgl_fused_add_rmsnorm(
    torch::Tensor input, torch::Tensor residual, torch::Tensor weight, double eps, bool enable_pdl) {
    CHECK_INPUT(input);
    ...
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), c_type, [&] {
        cudaError_t status = norm::FusedAddRMSNorm(
            static_cast<c_type*>(input.data_ptr()),
            static_cast<c_type*>(residual.data_ptr()),
            static_cast<c_type*>(weight.data_ptr()),
            batch_size,
            hidden_size,
            input.stride(0),
            residual.stride(0),
            eps,
            enable_pdl,
            torch_current_stream);
        TORCH_CHECK(
            status == cudaSuccess, "FusedAddRMSNorm failed with error code " + std::string(cudaGetErrorString(status)));
        return true;
    });
}
```

这段实现已经把重点说得很明白了。residual add 和 RMSNorm 不分开做，就是为了少一轮中间结果读写。对于这种高频、访存占比不低的算子，这件事往往比你在内层多抠几条算术指令更值钱。

benchmark 侧也能看出来它不是在做那种脱离真实调用的极简测法：

```python
def rmsnorm_sglang(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
):
    orig_shape = x.shape
    x = x.view(-1, x.shape[-1])
    if residual is not None:
        residual = residual.view(-1, residual.shape[-1])
    if enable_pdl is None:
        enable_pdl = is_arch_support_pdl()
    if residual is not None:
        sgl_kernel.fused_add_rmsnorm(x, residual, weight, eps, enable_pdl=enable_pdl)
        output = (x, residual)
    else:
        out = torch.empty_like(x)
        sgl_kernel.rmsnorm(x, weight, eps, out=out, enable_pdl=enable_pdl)
        output = out
    ...
```

它先把形状规整到二维，再分别走 fused 和非 fused 路径，本质上是在按真实使用方式对比不同实现。这里真正值得盯的变量其实就三个：reduce 做到哪一级，weight 和 input 是不是够宽，residual add 有没有顺手融合掉。

## `hp_rms_norm` 的思路很完整，不只是把某一段循环写快

`hp_rms_norm` 这份实现我觉得很有意思，因为它不是只在内层做点小修补，而是从类型、shared memory、persistent CTA 到 occupancy 一起设计。

一上来先看类型层就能感觉到这个味道。它没有停在 `half2`，而是直接把 `16B` 和 `32B` 路径拆开：

```cpp
union U32B_bf162{
#if __CUDACC_VER_MAJOR__ >= 13
  longlong4_32a memory_type;
#else
  longlong4 memory_type;
#endif
  __nv_bfloat162 real_type[8];
};

union U32B_f162{
#if __CUDACC_VER_MAJOR__ >= 13
  longlong4_32a memory_type;
#else
  longlong4 memory_type;
#endif
  __half2 real_type[8];
};
```

trait 也跟着走：

```cpp
template<> struct UVTypeTrait<__nv_bfloat16, 32> {
  using U = U32B_bf162;
#if __CUDACC_VER_MAJOR__ >= 13
  using V = longlong4_32a;
#else
  using V = longlong4;
#endif
};
```

这说明作者从一开始就在认真追 `32B` 粒度的向量化，而不是把它当成一个顺手的优化。对 `B200` 这种更吃访存组织的平台，这类路径往往就是分水岭。

再看最朴素的 `rms_norm_vector_reg_kernel`，它已经把 residual add、平方和累计、结果回写揉在一起了：

```cpp
const V* p = reinterpret_cast<const V*>(input) + token_id * vec_hidden_dim;
u.memory_type = p[threadIdx.x];
V* p_res = reinterpret_cast<V*>(residual) + token_id * vec_hidden_dim;
u_res.memory_type = p_res[threadIdx.x];
...
float2 inp_res = make_float2(val.x + res.x, val.y + res.y);
acc_square.x += inp_res.x * inp_res.x;
acc_square.y += inp_res.y * inp_res.y;
u.real_type[i] = __float22half2_rn(inp_res);
...
p_res[threadIdx.x] = u.memory_type;
```

真正把层次拉开的，是 `rms_norm_vector_reg_shm_kernel`。它先把 `weight` 异步搬到 shared memory：

```cpp
__shared__ barrier mbarrier;
...
if (threadIdx.x == 0) {
  init(&mbarrier, blockDim.x);
  cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
}
__syncthreads();

if (threadIdx.x == 0) {
  cuda::memcpy_async(
      shared_memory,
      weight,
      cuda::aligned_size_t<16>(vec_hidden_dim * VEC_SIZE_IN_BYTE),
      mbarrier
  );
}
...
barrier::arrival_token arrival_token = mbarrier.arrive();
```

第一次 token 明确等一下这次 transaction：

```cpp
if (token_id == static_cast<int>(blockIdx.x)) {
  mbarrier.wait(std::move(arrival_token));
}
```

后面则进入 persistent 风格的 token 循环：

```cpp
while (token_id < tokens) {
    ...
    token_id += static_cast<int>(gridDim.x);
}
```

这几段代码放在一起看就很清楚了。它真正想做的，不只是把某次 `RMSNorm` 算快一点，而是让 weight 的生命周期变长，让 shared memory staging 值得，顺带把 CTA 做成 persistent 地往前推。也就是说，它是在同时处理 weight 复用、向量化、token 生命周期和并发。

## `8192` 和 `16384` 这两个 hidden size，很多时候不是一个量级的问题

很多人看 hidden size，会下意识觉得 16384 无非就是 8192 的两倍。但在这种 kernel 里，事情通常没这么线性。

看这段：

```cpp
int iteration = (vec_hidden_dim + threads - 1) / threads;
...
V* p_shm = reinterpret_cast<V*>(shared_memory + VEC_SIZE_IN_BYTE * vec_hidden_dim);
for (int i = 1; i < iteration; i++) {
  auto offset = threadIdx.x + i * threads;
  if (offset < vec_hidden_dim) {
    ...
    p_shm[shm_offset] = tmp.memory_type;
  }
}
```

`vec_hidden_dim` 一旦变大，`iteration` 会跟着涨，shared memory 暂存输入的那块区域也会膨胀。后面的 reduce 也会变重：

```cpp
float warp_sum = cooperative_groups::reduce(
  cg_warp,
  acc_square.x + acc_square.y,
  cooperative_groups::plus<float>()
);

float cta_sum = cooperative_groups::reduce(
  cg_warp,
  threadIdx.x < NUM_WARPS ? buffer[threadIdx.x] : 0.0f,
  cooperative_groups::plus<float>()
);
```

这时候问题通常就不只是“元素数翻倍”，而是几件事一起变：向量 load 更多了，shared memory footprint 大了，block 内同步频率高了，寄存器生命周期拉长了，occupancy 可能也掉台阶。很多 kernel 到这里先崩的不是算术，而是资源平衡。

## dynamic shared memory 值不值得开，得和 occupancy 一起算

`hp_rms_norm` 最值得学的一点，是它没有把 shared memory 用量写死，而是明确让 runtime 帮它算“在目标并发下还能开多大 dynamic shared memory”。

```cpp
cudaFuncAttributes kernel_attr;
AT_CUDA_CHECK(cudaFuncGetAttributes(&kernel_attr, kernel_ptr));
AT_CUDA_CHECK(cudaFuncSetAttribute(
  kernel_ptr,
  cudaFuncAttributeMaxDynamicSharedMemorySize,
  at::cuda::getCurrentDeviceProperties()->sharedMemPerBlockOptin - kernel_attr.sharedSizeBytes
));

size_t smem_size;
AT_CUDA_CHECK(cudaOccupancyAvailableDynamicSMemPerBlock(&smem_size, kernel_ptr, num_ctas_per_sm, num_threads));
```

grid 也不是随便设的，直接按 persistent CTA 数量来：

```cpp
uint persistent_ctas =
  at::cuda::getCurrentDeviceProperties()->multiProcessorCount * num_ctas_per_sm;

dim3 grid(persistent_ctas, 1, 1);
```

这类代码其实就在做一个很实在的权衡：shared memory 开大一点，局部复用当然可能更好，但它到底值不值用 occupancy 去换？这个问题很多时候比“shared memory 有没有帮我省几次 global load”更重要。

我见过不少 kernel，局部看 shared memory staging 很漂亮，结果一跑起来 occupancy 先掉了，最后 latency 反而暴露得更明显。问题不在 shared memory 本身，而在没把它放回整个 SM 资源模型里去看。

## 最后还是得回到系统，而不是只盯单个大 kernel

如果只看 `sgemm` 或 `wgmma`，很容易觉得 CUDA 调优主要是在追一两个大 kernel。可实际推理里，大量时间就散在那堆高频中小算子上，尤其是量化、RMSNorm、MoE 路由、RoPE、cache 搬运这类路径。`common_extension.cc` 里那些接口摆在一起，其实已经把这个现实写出来了。

`fused_add_rmsnorm`、`sgl_per_token_group_quant_8bit_v2`、`fp8_blockwise_scaled_grouped_mm`、`moe_fused_gate`、`fast_topk_transform_fused`，它们单看都不算“最耀眼”的 kernel，但链路里少了任何一个，系统性能都会漏掉一块。

所以我现在对 CUDA 调优的理解，和一开始学的时候已经不太一样了。以前会更在意哪条指令新、哪份 kernel 峰值更高。现在反而更关心三件事：

- 这个瓶颈到底是带宽、延迟，还是调度节奏
- 数据有没有被搬太多次，或者搬得不够顺
- 单个 kernel 之外，整条热点链路是不是接得起来

顺着这批代码看下来，最后剩下的其实不是某一个“秘诀”，而是一个挺老实的判断框架。先把机器摸清楚，再去看数据流，再看新架构的搬运和同步语义，最后把问题放回系统里。这样走下来，很多看上去零散的技巧——`cp.async`、`TMA`、`WGMMA`、`FP8 grouped GEMM`、`persistent CTA`、`enable_pdl`——就没那么散了。它们都在做一件事：尽量别让计算单元空着等数据。

我觉得这已经很接近 CUDA 调优里最不花哨、但也最管用的那部分经验了。
