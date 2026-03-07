> 我的课程笔记，欢迎关注：https://github.com/BBuf/how-to-optim-algorithm-in-cuda 。

# 0x0. 前言

GPU-MODE 会定期搞一些 kernel 竞赛，通过耗时排名决出最快的实现，这次来看看 nvfp4_gemv 竞赛的 rank1 代码，链接在这里：https://www.gpumode.com/leaderboard/595?tab=rankings

前三名速度差很小，这篇文章主要阅读 rank1 的实现，学习一下针对 B200 的 nvfp4 GEMV 优化思路。

# 0x1. 问题描述

![](https://files.mdnice.com/user/59/7ec4979b-49f5-49d2-960a-b51983561f3c.png)

需要实现一个针对 NVIDIA B200 优化的 batched matrix-vector multiplication kernel，输入张量为：

- `a`：M × K × L，K-major，nvfp4（e2m1）
- `b`：1 × K × L，K-major，nvfp4（e2m1）
- `sfa`：M × (K // 16) × L，fp8（e4m3fnuz），A 的缩放因子，每 16 个 fp4 共享一个
- `sfb`：1 × (K // 16) × L，fp8（e4m3fnuz），B 的缩放因子
- `c`：M × 1 × L，fp16，输出

排名标准是各 benchmark 结果的几何平均值。理论极限基于 B200 最大 FFMA 算力与 DRAM 带宽（1.5 GHz 时钟频率）：

| M    | K     | L | 时间 [μs] |
|------|-------|---|-----------|
| 7168 | 16384 | 1 | 8.622     |
| 4096 | 7168  | 8 | 17.275    |
| 7168 | 2048  | 4 | 4.317     |

三组 benchmark 测试形状：`{"k": 16384, "l": 1, "m": 7168}`、`{"k": 7168, "l": 8, "m": 4096}`、`{"k": 2048, "l": 4, "m": 7168}`。

# 0x2. Baseline 参考实现

先看一下官方给的 baseline，核心是逐 batch 调用 `torch._scaled_mm` 完成 nvfp4 block-scaled GEMV：

```python
import torch
from task import input_t, output_t
from utils import make_match_reference

# 每 16 个 nvfp4 元素共享一个 fp8 缩放因子
sf_vec_size = 16

def ceil_div(a, b):
    return (a + b - 1) // b


def to_blocked(input_matrix):
    # 将线性排列的缩放因子转换为 cuBLAS D型 blocked 格式，供 torch._scaled_mm 使用
    # 参考：https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout
    rows, cols = input_matrix.shape
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)
    blocks = input_matrix.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten()


def ref_kernel(data: input_t) -> output_t:
    """PyTorch 参考实现：逐 batch 调用 torch._scaled_mm 完成 NVFP4 block-scaled GEMV。"""
    a_ref, b_ref, sfa_ref_cpu, sfb_ref_cpu, _, _, c_ref = data
    _, _, l = c_ref.shape
    for l_idx in range(l):
        # 缩放因子须先转为 cuBLAS blocked 格式；b 的 N 已 pad 到 128，只取第 0 列结果
        scale_a = to_blocked(sfa_ref_cpu[:, :, l_idx])
        scale_b = to_blocked(sfb_ref_cpu[:, :, l_idx])
        # b_ref[:, :, l_idx].shape = (128, K//2)，transpose 后为 (K//2, 128)
        # torch._scaled_mm 要求 N>=128，所以 b 被 pad 到 N=128，只有第 0 行是真正的向量
        # res.shape = (M, 128)，第 0 列 = A 与真实 b 向量的点积，其余列是无意义的 padding 结果
        res = torch._scaled_mm(
            a_ref[:, :, l_idx],
            b_ref[:, :, l_idx].transpose(0, 1),
            scale_a.cuda(), scale_b.cuda(),
            bias=None, out_dtype=torch.float16,
        )
        c_ref[:, 0, l_idx] = res[:, 0]  # 只取第 0 列，相当于撤销 pad
    return c_ref


def generate_input(m: int, k: int, l: int, seed: int):
    """生成测试输入：a/b 为 nvfp4，sfa/sfb 为 fp8 缩放因子，c 为 fp16 输出。
    同时返回供自定义 kernel 使用的 cuBLAS blocked 排列缩放因子（sfa_permuted/sfb_permuted）。
    """
    torch.manual_seed(seed)
    n = 1
    n_padded_128 = 128  # torch._scaled_mm 要求 N 对齐到 128

    # nvfp4 两元素打包为一个 uint8，故 K 方向存储长度为 k//2
    a_ref = torch.randint(0, 4, (l, m, k // 2), dtype=torch.uint8, device="cuda").permute(1, 2, 0)
    b_ref = torch.randint(0, 4, (l, n_padded_128, k // 2), dtype=torch.uint8, device="cuda").permute(1, 2, 0)
    a_ref = a_ref.view(torch.float4_e2m1fn_x2)
    b_ref = b_ref.view(torch.float4_e2m1fn_x2)
    c_ref = torch.randn((l, m, n), dtype=torch.float16, device="cuda").permute(1, 2, 0)

    def create_scale_factor_tensors(l, mn, sf_k):
        # 生成线性布局（供参考 kernel）和 cuBLAS blocked 布局（供自定义 kernel）两份缩放因子
        ref_f8 = torch.randint(0, 3, (l, mn, sf_k), dtype=torch.int8, device='cuda').to(torch.float8_e4m3fn)
        ref_f8_perm = ref_f8.permute(1, 2, 0)  # (mn, sf_k, l)

        atom_m, atom_k = (32, 4), 4
        mma_shape = (l, ceil_div(mn, atom_m[0]*atom_m[1]), ceil_div(sf_k, atom_k),
                     atom_m[0], atom_m[1], atom_k)
        reordered = torch.randint(0, 3, mma_shape, dtype=torch.int8, device='cuda').to(torch.float8_e4m3fn)
        reordered = reordered.permute(3, 4, 1, 5, 2, 0)  # -> (32, 4, ceil_mn, 4, ceil_sfk, l)

        i_grid, j_grid, b_grid = torch.meshgrid(
            torch.arange(mn, device='cuda'), torch.arange(sf_k, device='cuda'),
            torch.arange(l, device='cuda'), indexing='ij')
        mm   = i_grid // (atom_m[0] * atom_m[1])
        mm32 = i_grid % atom_m[0]
        mm4  = (i_grid % 128) // atom_m[0]
        kk, kk4 = j_grid // atom_k, j_grid % atom_k
        reordered[mm32, mm4, mm, kk4, kk, b_grid] = ref_f8_perm[i_grid, j_grid, b_grid]
        return ref_f8_perm.cpu(), reordered

    sf_k = ceil_div(k, sf_vec_size)
    sfa_ref_cpu, sfa_permuted = create_scale_factor_tensors(l, m, sf_k)
    sfb_ref_cpu, sfb_permuted = create_scale_factor_tensors(l, n_padded_128, sf_k)
    return (a_ref, b_ref, sfa_ref_cpu.cuda(), sfb_ref_cpu.cuda(), sfa_permuted, sfb_permuted, c_ref)


check_implementation = make_match_reference(ref_kernel, rtol=1e-03, atol=1e-03)
```

这里有几个值得注意的细节：`to_blocked` 负责把线性排列的缩放因子转成 cuBLAS 要求的 D型 blocked 格式。另一个绕弯子的地方是 `torch._scaled_mm` 要求 N 维度至少为 128（cuBLAS 硬件对齐约束），但这道题 b 的 N=1。解决方案是在 `generate_input` 里把 b pad 到 N=128，`b_ref[0, :, :]` 是真正的向量，1~127 行是随机填充；调用 `_scaled_mm` 会得到 (M, 128) 的结果，`res[:, 0]` 就是 A 与真实 b 向量的点积，其余 127 列直接丢掉，相当于撤销了 pad。

# 0x3. Rank1 代码阅读

![](https://files.mdnice.com/user/59/1d3eb599-96ce-4512-bb23-9fc5e3da35d2.png)

接下来看看 rank1 的实现。整体是一个用 `load_inline` 编译的手写 CUDA kernel，核心思路是针对 B200 的带宽瓶颈做精细的缓存控制，同时用 PTX inline assembly 直接操作 fp4/fp8 的 packed 格式，避免不必要的精度转换开销。

```python
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

# ---- C++ stub: declare the function so load_inline can bind it ----
gemv_cpp = r"""
#include <torch/extension.h>

// Forward declaration so PyTorch can bind it (definition is in the CUDA source).
torch::Tensor cuda_nvfp4_gemv(torch::Tensor A,
                            torch::Tensor B,
                            torch::Tensor C,
                            torch::Tensor SFA,
                            torch::Tensor SFB);
"""

# ---- CUDA source: struct, kernel, launcher, and Python-facing wrapper ----
gemv_cuda = r"""
#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include <cuda_runtime.h>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_fp4.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

struct Gemv_params {
    using index_t = uint64_t;

    int b, m, k, real_k;

    void *__restrict__ a_ptr, *__restrict__ b_ptr;
    void *__restrict__ sfa_ptr, *__restrict__ sfb_ptr, *__restrict__ o_ptr;

    index_t a_batch_stride, b_batch_stride, sfa_batch_stride, sfb_batch_stride, o_batch_stride;
    index_t a_row_stride,   b_row_stride,   sfa_row_stride,   sfb_row_stride,   o_row_stride;
};

static constexpr int BLOCK_SIZE = 128;

// GEMV 是带宽受限操作；A 矩阵每行只读一次，B 向量被所有行共享。
// 针对不同 K 值用不同 PTX 加载修饰符控制缓存行为，通过 if constexpr 编译期分发。
// 每次调用加载 32个fp4（16个fp4x2，16字节）+ 2个fp8 scale（uint16_t）。

// 通用版本
__device__ __forceinline__ void load_block_16x2fp4_generic(
    const __nv_fp4x2_e2m1* rowA,
    const __nv_fp4x2_e2m1* vecB,
    const uint16_t*        rowS_u16,
    const uint16_t*        vecS_u16,
    int                    elem_base,
    int                    block_base,
    uint64_t (&a_regs)[2],
    uint64_t (&b_regs)[2],
    uint16_t &sfa_regs,
    uint16_t &sfb_regs)
{
    uint64_t rowA_addr = reinterpret_cast<uint64_t>(rowA + elem_base);
    uint64_t vecB_addr = reinterpret_cast<uint64_t>(vecB + elem_base);
    uint64_t rowS_addr = reinterpret_cast<uint64_t>(rowS_u16 + block_base);
    uint64_t vecS_addr = reinterpret_cast<uint64_t>(vecS_u16 + block_base);

    asm volatile(
        "ld.global.u64.v2 {%0, %1}, [%4];\n\t"
        "ld.global.u64.v2 {%2, %3}, [%5];\n\t"
        : "=l"(a_regs[0]), "=l"(a_regs[1]), "=l"(b_regs[0]), "=l"(b_regs[1])
        : "l"(rowA_addr), "l"(vecB_addr)
    );
    asm volatile(
        "ld.global.u16 %0, [%2];\n\t"
        "ld.global.u16 %1, [%3];\n\t"
        : "=h"(sfa_regs), "=h"(sfb_regs)
        : "l"(rowS_addr), "l"(vecS_addr)
    );
}

// k=3584：A 流式（.cs，不驻留L1），B 用 L2::128B 优先驻留L2
__device__ __forceinline__ void load_block_16x2fp4_k3584(
    const __nv_fp4x2_e2m1* rowA,
    const __nv_fp4x2_e2m1* vecB,
    const uint16_t*        rowS_u16,
    const uint16_t*        vecS_u16,
    int                    elem_base,
    int                    block_base,
    uint64_t (&a_regs)[2],
    uint64_t (&b_regs)[2],
    uint16_t &sfa_regs,
    uint16_t &sfb_regs)
{
    uint64_t rowA_addr = reinterpret_cast<uint64_t>(rowA + elem_base);
    uint64_t vecB_addr = reinterpret_cast<uint64_t>(vecB + elem_base);
    uint64_t rowS_addr = reinterpret_cast<uint64_t>(rowS_u16 + block_base);
    uint64_t vecS_addr = reinterpret_cast<uint64_t>(vecS_u16 + block_base);

    asm volatile(
        "ld.global.cs.u64.v2 {%0, %1}, [%4];\n\t"
        "ld.global.L2::128B.u64.v2 {%2, %3}, [%5];\n\t"
        : "=l"(a_regs[0]), "=l"(a_regs[1]),
          "=l"(b_regs[0]), "=l"(b_regs[1])
        : "l"(rowA_addr), "l"(vecB_addr)
    );

    asm volatile(
        "ld.global.cs.u16 %0, [%2];\n\t"
        "ld.global.L2::128B.u16 %1, [%3];\n\t"
        : "=h"(sfa_regs), "=h"(sfb_regs)
        : "l"(rowS_addr), "l"(vecS_addr)
    );
}

// k=8192：A 流式，A scale 用 .lu（last use，提示不再复用）
__device__ __forceinline__ void load_block_16x2fp4_k8192(
    const __nv_fp4x2_e2m1* rowA,
    const __nv_fp4x2_e2m1* vecB,
    const uint16_t*        rowS_u16,
    const uint16_t*        vecS_u16,
    int                    elem_base,
    int                    block_base,
    uint64_t (&a_regs)[2],
    uint64_t (&b_regs)[2],
    uint16_t &sfa_regs,
    uint16_t &sfb_regs)
{
    uint64_t rowA_addr = reinterpret_cast<uint64_t>(rowA + elem_base);
    uint64_t vecB_addr = reinterpret_cast<uint64_t>(vecB + elem_base);
    uint64_t rowS_addr = reinterpret_cast<uint64_t>(rowS_u16 + block_base);
    uint64_t vecS_addr = reinterpret_cast<uint64_t>(vecS_u16 + block_base);

    asm volatile(
        "ld.global.cs.u64.v2 {%0, %1}, [%4];\n\t"
        "ld.global.u64.v2 {%2, %3}, [%5];\n\t"
        : "=l"(a_regs[0]), "=l"(a_regs[1]),
          "=l"(b_regs[0]), "=l"(b_regs[1])
        : "l"(rowA_addr), "l"(vecB_addr)
    );

    asm volatile(
        "ld.global.lu.u16 %0, [%2];\n\t"
        "ld.global.u16 %1, [%3];\n\t"
        : "=h"(sfa_regs), "=h"(sfb_regs)
        : "l"(rowS_addr), "l"(vecS_addr)
    );
}

// k=1024：A 和 A scale 均用 .cs 流式加载
__device__ __forceinline__ void load_block_16x2fp4_k1024(
    const __nv_fp4x2_e2m1* rowA,
    const __nv_fp4x2_e2m1* vecB,
    const uint16_t*        rowS_u16,
    const uint16_t*        vecS_u16,
    int                    elem_base,
    int                    block_base,
    uint64_t (&a_regs)[2],
    uint64_t (&b_regs)[2],
    uint16_t &sfa_regs,
    uint16_t &sfb_regs)
{
    uint64_t rowA_addr = reinterpret_cast<uint64_t>(rowA + elem_base);
    uint64_t vecB_addr = reinterpret_cast<uint64_t>(vecB + elem_base);
    uint64_t rowS_addr = reinterpret_cast<uint64_t>(rowS_u16 + block_base);
    uint64_t vecS_addr = reinterpret_cast<uint64_t>(vecS_u16 + block_base);

    asm volatile(
        "ld.global.cs.u64.v2 {%0, %1}, [%4];\n\t"
        "ld.global.u64.v2 {%2, %3}, [%5];\n\t"
        : "=l"(a_regs[0]), "=l"(a_regs[1]),
          "=l"(b_regs[0]), "=l"(b_regs[1])
        : "l"(rowA_addr), "l"(vecB_addr)
    );

    asm volatile(
        "ld.global.cs.u16 %0, [%2];\n\t"
        "ld.global.u16 %1, [%3];\n\t"
        : "=h"(sfa_regs), "=h"(sfb_regs)
        : "l"(rowS_addr), "l"(vecS_addr)
    );
}

template<int K>
__device__ __forceinline__ void load_block_16x2fp4(
    const __nv_fp4x2_e2m1* rowA,
    const __nv_fp4x2_e2m1* vecB,
    const uint16_t*        rowS_u16,
    const uint16_t*        vecS_u16,
    int                    elem_base,
    int                    block_base,
    uint64_t (&a_regs)[2],
    uint64_t (&b_regs)[2],
    uint16_t &sfa_regs,
    uint16_t &sfb_regs)
{
    if constexpr (K == 3584) {
        load_block_16x2fp4_k3584(
            rowA, vecB, rowS_u16, vecS_u16,
            elem_base, block_base, a_regs, b_regs, sfa_regs, sfb_regs);
    } else if constexpr (K == 8192) {
        load_block_16x2fp4_k8192(
            rowA, vecB, rowS_u16, vecS_u16,
            elem_base, block_base, a_regs, b_regs, sfa_regs, sfb_regs);
    } else if constexpr (K == 1024) {
        load_block_16x2fp4_k1024(
            rowA, vecB, rowS_u16, vecS_u16,
            elem_base, block_base, a_regs, b_regs, sfa_regs, sfb_regs);
    } else {
        load_block_16x2fp4_generic(
            rowA, vecB, rowS_u16, vecS_u16,
            elem_base, block_base, a_regs, b_regs, sfa_regs, sfb_regs);
    }
}

// k=8192 专用：加载 64个fp4 + 4个fp8 scale
// A：L1不分配 + L2优先驱逐（流式）；B：L1/L2均保留（被所有行共享）
__device__ __forceinline__ void load_block_32x2fp4(
    const __nv_fp4x2_e2m1* rowA,
    const __nv_fp4x2_e2m1* vecB,
    const uint16_t*        rowS_u16,
    const uint16_t*        vecS_u16,
    int                    elem_base,
    int                    block_base,
    uint64_t (&a_regs)[4],
    uint64_t (&b_regs)[4],
    uint16_t (&sfa_regs)[2],
    uint16_t (&sfb_regs)[2])
{
    uint64_t rowA_addr = reinterpret_cast<uint64_t>(rowA + elem_base);
    uint64_t vecB_addr = reinterpret_cast<uint64_t>(vecB + elem_base);

    asm volatile(
        "ld.global.L1::no_allocate.L2::evict_first.L2::256B.v4.u64 {%0, %1, %2, %3}, [%8];\n\t"
        "ld.global.L1::evict_last.L2::evict_last.v4.u64 {%4, %5, %6, %7}, [%9];\n\t"
        : "=l"(a_regs[0]), "=l"(a_regs[1]), "=l"(a_regs[2]), "=l"(a_regs[3]),
          "=l"(b_regs[0]), "=l"(b_regs[1]), "=l"(b_regs[2]), "=l"(b_regs[3])
        : "l"(rowA_addr), "l"(vecB_addr)
    );

    uint64_t rowS_addr = reinterpret_cast<uint64_t>(rowS_u16 + block_base * 2);
    uint64_t vecS_addr = reinterpret_cast<uint64_t>(vecS_u16 + block_base * 2);

    asm volatile(
        "ld.global.L1::no_allocate.v2.u16 {%0, %1}, [%4];\n\t"
        "ld.global.L1::evict_last.v2.u16 {%2, %3}, [%5];\n\t"
        : "=h"(sfa_regs[0]), "=h"(sfa_regs[1]),
          "=h"(sfb_regs[0]), "=h"(sfb_regs[1])
        : "l"(rowS_addr), "l"(vecS_addr)
    );
}

// 详细解析见末尾「block_scaled_fma 函数详解」节
__device__ __forceinline__ __half block_scaled_fma_32x2fp4(
    const uint64_t (&a_regs)[4],
    const uint64_t (&b_regs)[4],
    const uint16_t (&sfa_regs)[2],
    const uint16_t (&sfb_regs)[2])
{
    const uint32_t* a = reinterpret_cast<const uint32_t*>(a_regs);
    const uint32_t* b = reinterpret_cast<const uint32_t*>(b_regs);

    // Step 1: fp8 scale decode + combine（4个fp8 → 2个f16x2）
    uint32_t sf0_f16x2, sf1_f16x2;
    {
        uint32_t sfa0, sfb0, sfa1, sfb1;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(sfa0) : "h"(sfa_regs[0]));
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(sfb0) : "h"(sfb_regs[0]));
        asm("mul.rn.f16x2 %0, %1, %2;"    : "=r"(sf0_f16x2) : "r"(sfa0), "r"(sfb0));
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(sfa1) : "h"(sfa_regs[1]));
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(sfb1) : "h"(sfb_regs[1]));
        asm("mul.rn.f16x2 %0, %1, %2;"    : "=r"(sf1_f16x2) : "r"(sfa1), "r"(sfb1));
    }

    // Step 2: 将每个 f16x2 中的两个 scale 分别广播为 packed f16x2
    uint16_t s0, s1, s2, s3;
    uint32_t scale0, scale1, scale2, scale3;
    asm("mov.b32 {%0,%1}, %2;" : "=h"(s0), "=h"(s1) : "r"(sf0_f16x2));
    asm("mov.b32 {%0,%1}, %2;" : "=h"(s2), "=h"(s3) : "r"(sf1_f16x2));
    asm("mov.b32 %0, {%1,%1};" : "=r"(scale0) : "h"(s0));
    asm("mov.b32 %0, {%1,%1};" : "=r"(scale1) : "h"(s1));
    asm("mov.b32 %0, {%1,%1};" : "=r"(scale2) : "h"(s2));
    asm("mov.b32 %0, {%1,%1};" : "=r"(scale3) : "h"(s3));

    uint32_t accum = 0;

    // Step 3: 4个 scale block，每块处理 16个fp4
    #pragma unroll
    for (int blk = 0; blk < 4; ++blk) {
        uint32_t cvt_a[8], cvt_b[8];
        asm volatile(
            "{ .reg .b8 x0,x1,x2,x3,x4,x5,x6,x7;\n\t"
            "mov.b32 {x0,x1,x2,x3}, %8;\n\t"
            "mov.b32 {x4,x5,x6,x7}, %9;\n\t"
            "cvt.rn.f16x2.e2m1x2 %0,x0; cvt.rn.f16x2.e2m1x2 %1,x1;\n\t"
            "cvt.rn.f16x2.e2m1x2 %2,x2; cvt.rn.f16x2.e2m1x2 %3,x3;\n\t"
            "cvt.rn.f16x2.e2m1x2 %4,x4; cvt.rn.f16x2.e2m1x2 %5,x5;\n\t"
            "cvt.rn.f16x2.e2m1x2 %6,x6; cvt.rn.f16x2.e2m1x2 %7,x7; }"
            : "=r"(cvt_a[0]),"=r"(cvt_a[1]),"=r"(cvt_a[2]),"=r"(cvt_a[3]),
              "=r"(cvt_a[4]),"=r"(cvt_a[5]),"=r"(cvt_a[6]),"=r"(cvt_a[7])
            : "r"(a[blk*2]), "r"(a[blk*2+1]));
        asm volatile(
            "{ .reg .b8 x0,x1,x2,x3,x4,x5,x6,x7;\n\t"
            "mov.b32 {x0,x1,x2,x3}, %8;\n\t"
            "mov.b32 {x4,x5,x6,x7}, %9;\n\t"
            "cvt.rn.f16x2.e2m1x2 %0,x0; cvt.rn.f16x2.e2m1x2 %1,x1;\n\t"
            "cvt.rn.f16x2.e2m1x2 %2,x2; cvt.rn.f16x2.e2m1x2 %3,x3;\n\t"
            "cvt.rn.f16x2.e2m1x2 %4,x4; cvt.rn.f16x2.e2m1x2 %5,x5;\n\t"
            "cvt.rn.f16x2.e2m1x2 %6,x6; cvt.rn.f16x2.e2m1x2 %7,x7; }"
            : "=r"(cvt_b[0]),"=r"(cvt_b[1]),"=r"(cvt_b[2]),"=r"(cvt_b[3]),
              "=r"(cvt_b[4]),"=r"(cvt_b[5]),"=r"(cvt_b[6]),"=r"(cvt_b[7])
            : "r"(b[blk*2]), "r"(b[blk*2+1]));

        uint32_t grp = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            asm("fma.rn.f16x2 %0,%1,%2,%0;" : "+r"(grp) : "r"(cvt_a[i]), "r"(cvt_b[i]));

        const uint32_t* scales[4] = {&scale0, &scale1, &scale2, &scale3};
        asm("mul.rn.f16x2 %0,%1,%0;" : "+r"(grp) : "r"(*scales[blk]));
        asm("add.rn.f16x2 %0,%0,%1;" : "+r"(accum) : "r"(grp));
    }

    // Step 4: f16x2 两 lane 相加 → 标量 f16
    uint16_t r0, r1;
    asm("mov.b32 {%0,%1}, %2;" : "=h"(r0), "=h"(r1) : "r"(accum));
    uint16_t result;
    asm("add.rn.f16 %0,%1,%2;" : "=h"(result) : "h"(r0), "h"(r1));
    return __half(result);
}


// 详细解析见末尾「block_scaled_fma 函数详解」节
__device__ __forceinline__ float block_scaled_fma_16x2fp4(
    const uint64_t (&a_regs)[2],
    const uint64_t (&b_regs)[2],
    uint16_t       sfa_regs,
    uint16_t       sfb_regs)
{
    const uint32_t* a = reinterpret_cast<const uint32_t*>(a_regs);
    const uint32_t* b = reinterpret_cast<const uint32_t*>(b_regs);

    // Step 1: fp8 scale decode + combine
    uint32_t sfa_f16x2, sfb_f16x2, sf_f16x2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(sfa_f16x2) : "h"(sfa_regs));
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(sfb_f16x2) : "h"(sfb_regs));
    asm("mul.rn.f16x2 %0, %1, %2;"    : "=r"(sf_f16x2)  : "r"(sfa_f16x2), "r"(sfb_f16x2));

    // Step 2: 广播 scale0/scale1 为 packed f16x2
    uint16_t lane0, lane1;
    uint32_t scale0, scale1;
    asm("mov.b32 {%0,%1}, %2;"  : "=h"(lane0), "=h"(lane1) : "r"(sf_f16x2));
    asm("mov.b32 %0, {%1,%1};"  : "=r"(scale0) : "h"(lane0));
    asm("mov.b32 %0, {%1,%1};"  : "=r"(scale1) : "h"(lane1));

    uint32_t accum = 0;

    // Step 3: 2个 scale block，每块处理 16个fp4
    #pragma unroll
    for (int blk = 0; blk < 2; ++blk) {
        uint32_t cvt_a[8], cvt_b[8];
        asm volatile(
            "{ .reg .b8 x0,x1,x2,x3,x4,x5,x6,x7;\n\t"
            "mov.b32 {x0,x1,x2,x3}, %8;\n\t"
            "mov.b32 {x4,x5,x6,x7}, %9;\n\t"
            "cvt.rn.f16x2.e2m1x2 %0,x0; cvt.rn.f16x2.e2m1x2 %1,x1;\n\t"
            "cvt.rn.f16x2.e2m1x2 %2,x2; cvt.rn.f16x2.e2m1x2 %3,x3;\n\t"
            "cvt.rn.f16x2.e2m1x2 %4,x4; cvt.rn.f16x2.e2m1x2 %5,x5;\n\t"
            "cvt.rn.f16x2.e2m1x2 %6,x6; cvt.rn.f16x2.e2m1x2 %7,x7; }"
            : "=r"(cvt_a[0]),"=r"(cvt_a[1]),"=r"(cvt_a[2]),"=r"(cvt_a[3]),
              "=r"(cvt_a[4]),"=r"(cvt_a[5]),"=r"(cvt_a[6]),"=r"(cvt_a[7])
            : "r"(a[blk*2]), "r"(a[blk*2+1]));
        asm volatile(
            "{ .reg .b8 x0,x1,x2,x3,x4,x5,x6,x7;\n\t"
            "mov.b32 {x0,x1,x2,x3}, %8;\n\t"
            "mov.b32 {x4,x5,x6,x7}, %9;\n\t"
            "cvt.rn.f16x2.e2m1x2 %0,x0; cvt.rn.f16x2.e2m1x2 %1,x1;\n\t"
            "cvt.rn.f16x2.e2m1x2 %2,x2; cvt.rn.f16x2.e2m1x2 %3,x3;\n\t"
            "cvt.rn.f16x2.e2m1x2 %4,x4; cvt.rn.f16x2.e2m1x2 %5,x5;\n\t"
            "cvt.rn.f16x2.e2m1x2 %6,x6; cvt.rn.f16x2.e2m1x2 %7,x7; }"
            : "=r"(cvt_b[0]),"=r"(cvt_b[1]),"=r"(cvt_b[2]),"=r"(cvt_b[3]),
              "=r"(cvt_b[4]),"=r"(cvt_b[5]),"=r"(cvt_b[6]),"=r"(cvt_b[7])
            : "r"(b[blk*2]), "r"(b[blk*2+1]));

        uint32_t grp = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            asm("fma.rn.f16x2 %0,%1,%2,%0;" : "+r"(grp) : "r"(cvt_a[i]), "r"(cvt_b[i]));

        uint32_t scale = (blk == 0) ? scale0 : scale1;
        asm("mul.rn.f16x2 %0,%1,%0;" : "+r"(grp) : "r"(scale));
        asm("add.rn.f16x2 %0,%0,%1;" : "+r"(accum) : "r"(grp));
    }

    // Step 4: f16x2 两 lane 相加 → 标量 f16 → f32
    uint16_t r0, r1, result_f16;
    asm("mov.b32 {%0,%1}, %2;" : "=h"(r0), "=h"(r1) : "r"(accum));
    asm("add.rn.f16 %0,%1,%2;" : "=h"(result_f16) : "h"(r0), "h"(r1));
    float result;
    asm("cvt.f32.f16 %0,%1;"   : "=f"(result) : "h"(result_f16));
    return result;
}

// 模板参数：ROWS_PER_BLOCK（block 覆盖行数）、THREADS_PER_ROW（每行线程数，并行 K 方向）、
//           ITERS（>0 编译期展开，0 动态循环）、K_SPECIAL（load 函数特化值）、USE_32X2（路径选择）
// grid: (M/ROWS_PER_BLOCK, 1, L)；rib=row-in-block，lane=K方向编号
template <int ROWS_PER_BLOCK, int THREADS_PER_ROW, int ITERS, int K_SPECIAL, bool USE_32X2>
__global__ void __launch_bounds__(ROWS_PER_BLOCK * THREADS_PER_ROW, 8)
gemv_kernel(const __grid_constant__ Gemv_params params)
{
    const int tid   = threadIdx.x;
    const int rib   = tid / THREADS_PER_ROW;
    const int lane  = tid % THREADS_PER_ROW;
    const int batch = blockIdx.z;
    const int row   = blockIdx.x * ROWS_PER_BLOCK + rib;

    const size_t A_batch_base   = static_cast<size_t>(batch) * params.a_batch_stride;
    const size_t SFA_batch_base = static_cast<size_t>(batch) * params.sfa_batch_stride;
    const size_t B_batch_base   = static_cast<size_t>(batch) * params.b_batch_stride;
    const size_t SFB_batch_base = static_cast<size_t>(batch) * params.sfb_batch_stride;
    const size_t C_batch_base   = static_cast<size_t>(batch) * params.o_batch_stride;

    const __nv_fp4x2_e2m1* rowA = static_cast<const __nv_fp4x2_e2m1*>(params.a_ptr) + A_batch_base + row * params.a_row_stride;
    const __nv_fp8_e4m3*   rowS = static_cast<const __nv_fp8_e4m3*>(params.sfa_ptr) + SFA_batch_base + row * params.sfa_row_stride;
    const __nv_fp4x2_e2m1* vecB = static_cast<const __nv_fp4x2_e2m1*>(params.b_ptr) + B_batch_base;
    const __nv_fp8_e4m3*   vecS = static_cast<const __nv_fp8_e4m3*>(params.sfb_ptr) + SFB_batch_base;
    const uint16_t* rowS_u16 = reinterpret_cast<const uint16_t*>(rowS);
    const uint16_t* vecS_u16 = reinterpret_cast<const uint16_t*>(vecS);

    float sum = 0.f;

    if constexpr (USE_32X2) {
        #pragma unroll
        for (int idx = 0; idx < 2; ++idx) {
            int block_base = idx * THREADS_PER_ROW + lane;
            int elem_base  = block_base * 32;

            uint64_t a_regs[4], b_regs[4];
            uint16_t sfa_regs[2], sfb_regs[2];

            load_block_32x2fp4(
                rowA, vecB,
                rowS_u16, vecS_u16,
                elem_base, block_base,
                a_regs, b_regs,
                sfa_regs, sfb_regs);

            __half h = block_scaled_fma_32x2fp4(a_regs, b_regs, sfa_regs, sfb_regs);
            sum += __half2float(h);
        }

        // 规约：shared memory 128→32，再 warp shuffle
        __shared__ float sdata[THREADS_PER_ROW];
        sdata[lane] = sum;
        __syncthreads();

        if (tid < 64) sdata[lane] += sdata[lane + 64];
        __syncthreads();

        if (lane < 32) {
            float val = sdata[lane] + sdata[lane + 32];
            val += __shfl_down_sync(0xffffffff, val, 16);
            val += __shfl_down_sync(0xffffffff, val, 8);
            val += __shfl_down_sync(0xffffffff, val, 4);
            val += __shfl_down_sync(0xffffffff, val, 2);
            val += __shfl_down_sync(0xffffffff, val, 1);

            if (lane == 0) {
                __half* out = (__half*)params.o_ptr + C_batch_base + row;
                out[0] = __float2half(val);
            }
        }
    } else {
        auto body = [&](int idx) {
            int block_base = idx * THREADS_PER_ROW + lane;
            int elem_base  = block_base * 16;

            uint64_t a_regs[2], b_regs[2];
            uint16_t sfa_regs, sfb_regs;

            load_block_16x2fp4<K_SPECIAL>(
                rowA, vecB,
                rowS_u16, vecS_u16,
                elem_base, block_base,
                a_regs, b_regs,
                sfa_regs, sfb_regs);
            sum += block_scaled_fma_16x2fp4(a_regs, b_regs, sfa_regs, sfb_regs);
        };

        if constexpr (ITERS > 0) {
            #pragma unroll
            for (int idx = 0; idx < ITERS; ++idx) {
                body(idx);
            }
        } else {
            int iters = params.k / (THREADS_PER_ROW * 16);
            for (int idx = 0; idx < iters; ++idx) {
                body(idx);
            }
        }

        #pragma unroll
        for (int offset = THREADS_PER_ROW / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffffu, sum, offset, THREADS_PER_ROW);
        }

        if (lane == 0) {
            __half* out = (__half*)params.o_ptr + C_batch_base + row;
            out[0] = __float2half(sum);
        }
    }
}


torch::Tensor cuda_nvfp4_gemv(torch::Tensor A,
                            torch::Tensor B,
                            torch::Tensor C,
                            torch::Tensor SFA,
                            torch::Tensor SFB)
{

    const auto sizes = A.sizes();
    const int M = sizes[0];
    const int K = sizes[1];
    const int L = sizes[2];

    Gemv_params params{};
    params.b = L;
    params.m = M;
    params.k = K;

    params.a_ptr  = A.data_ptr();
    params.b_ptr  = B.data_ptr();
    params.sfa_ptr= SFA.data_ptr();
    params.sfb_ptr= SFB.data_ptr();
    params.o_ptr  = C.data_ptr();

    params.a_batch_stride  = A.stride(2);
    params.b_batch_stride  = B.stride(2);
    params.sfa_batch_stride= SFA.stride(2);
    params.sfb_batch_stride= SFB.stride(2);
    params.o_batch_stride  = C.stride(2);

    params.a_row_stride  = A.stride(0);
    params.b_row_stride  = B.stride(0);
    params.sfa_row_stride= SFA.stride(0);
    params.sfb_row_stride= SFB.stride(0);
    params.o_row_stride  = C.stride(0);

    // 按 K 值静态分发到最优配置：<ROWS_PER_BLOCK, THREADS_PER_ROW, ITERS, K_SPECIAL, USE_32X2>
    // grid = (M/ROWS_PER_BLOCK, 1, L)
    if (params.k <= 256) {
        gemv_kernel<16, 8, 0, 0, false><<<dim3(params.m/16,1,params.b), 128>>>(params);
    } else if (params.k == 3584) {
        // 3584 = 7 × (32线程 × 16fp4x2)
        gemv_kernel<4, 32, 7, 3584, false><<<dim3(params.m/4,1,params.b), 128>>>(params);
    } else if (params.k == 8192) {
        // 8192 = 2 × (128线程 × 32fp4x2)，走 USE_32X2 路径
        gemv_kernel<1, 128, 0, 8192, true><<<dim3(params.m,1,params.b), 128>>>(params);
    } else if (params.k == 1024) {
        // 1024 = 4 × (16线程 × 16fp4x2)
        gemv_kernel<8, 16, 4, 1024, false><<<dim3(params.m/8,1,params.b), 128>>>(params);
    } else {
        gemv_kernel<8, 16, 0, 0, false><<<dim3(params.m/8,1,params.b), 128>>>(params);
    }

    return C;
}
"""

# ---- build the module ----
nvfp4_module = load_inline(
    name="nvfp4_gemv",
    cpp_sources=[gemv_cpp],
    cuda_sources=[gemv_cuda],
    functions=["cuda_nvfp4_gemv"],
    extra_cuda_cflags=[
        "-std=c++17",
        "-gencode=arch=compute_100a,code=sm_100a",  # B200
        "--ptxas-options=--gpu-name=sm_100a",
        "-O3", "-w",
        "-maxrregcount=32",      # 限制寄存器用量，提升 occupancy，隐藏带宽延迟
        "--use_fast_math",
        "-allow-unsupported-compiler",
    ],
    extra_ldflags=["-lcuda", "-lcublas"],
    verbose=True,
)


def custom_kernel(data: input_t) -> output_t:
    return nvfp4_module.cuda_nvfp4_gemv(data[0], data[1], data[6], data[2], data[3])
```

代码整体可以分成几个部分来看：数据加载函数族（`load_block_*`）、核心 FMA 计算函数（`block_scaled_fma_*`）、主 kernel 模板（`gemv_kernel`）以及 launcher。下面分别展开。

## 0x3.1 数据加载：精细的缓存控制

GEMV 是典型的带宽受限操作。A 矩阵的每一行只会被读一次，而 B 向量则会被 M 行所有线程共享读取。针对这个特点，rank1 用 PTX 加载修饰符对缓存做了精细控制：

- **A 矩阵**：用 `.cs`（cache streaming）流式加载，告诉硬件这块数据读完就不会再用了，不要污染 L1/L2 缓存
- **B 向量**：用 `L2::128B`、`L2::evict_last` 等修饰符让 B 尽可能驻留在 L2 中，因为它会被反复读取

更进一步，针对不同的 K 值，代码提供了不同的特化版本（`_k3584`、`_k8192`、`_k1024`），通过 `if constexpr` 在编译期分发，运行时没有分支开销。比如 k=8192 的 `load_block_32x2fp4` 对 A 用了 `L1::no_allocate + L2::evict_first`，意思是连 L2 也不想让 A 占着，彻底走流式。

## 0x3.2 线程模型

kernel 的核心思路是：**一行输出由多个线程协作，各线程分担 K 方向的不同片段，最后规约成标量写输出**。

block 固定 128 线程，`THREADS_PER_ROW` 个线程负责同一行的 K 方向，`ROWS_PER_BLOCK = 128 / THREADS_PER_ROW` 行同时处理：

```cpp
const int rib  = tid / THREADS_PER_ROW;  // block 内第几行
const int lane = tid % THREADS_PER_ROW;  // K 方向编号
const int row  = blockIdx.x * ROWS_PER_BLOCK + rib;
```

grid 为 `(M/ROWS_PER_BLOCK, 1, L)`，`blockIdx.z` 对应 batch。各 K 值的配置如下：

| K（fp4x2单位） | `ROWS_PER_BLOCK` | `THREADS_PER_ROW` | 路径 |
|---|---|---|---|
| ≤256 | 16 | 8 | 16x2 动态 |
| 1024 | 8 | 16 | 16x2 展开4次 |
| 3584 | 4 | 32 | 16x2 展开7次 |
| 8192 | 1 | 128 | **32x2** 展开2次 |

以 K=3584（对应 benchmark shape k=7168，因为 nvfp4 两两打包，`params.k = 7168/2 = 3584`）为例：1个 block 处理 4 行，128线程 = 4行 × 32线程/行。每次迭代，32个 lane 把当前的 512个fp4x2 平均分掉（`block_base = idx×32 + lane`），7次迭代合计覆盖 7×32×16 = 3584个fp4x2 = **7168个fp4**，每个 lane 累积 7 段部分和后 warp shuffle 归约到 lane=0。

# 0x4. block_scaled_fma 函数详解

`block_scaled_fma_16x2fp4` 和 `block_scaled_fma_32x2fp4` 是整个 kernel 的计算核心，逻辑相同，区别只是规模：前者每次处理 32个fp4 + 2个fp8 scale，后者处理 64个fp4 + 4个fp8 scale。下面以 `_16x2fp4` 为例逐步拆解。

## 0x4.1 输入数据的排布

```cpp
__device__ __forceinline__ float block_scaled_fma_16x2fp4(
    const uint64_t (&a_regs)[2],   // 2×8字节 = 32个fp4
    const uint64_t (&b_regs)[2],
    uint16_t sfa_regs,             // 2个fp8 scale（每16个fp4共享一个）
    uint16_t sfb_regs)
{
    const uint32_t* a = reinterpret_cast<const uint32_t*>(a_regs);  // 视为 4个uint32
    const uint32_t* b = reinterpret_cast<const uint32_t*>(b_regs);
```

`a_regs[0]` → `a[0], a[1]` 覆盖 16个fp4（一个 scale block），`a_regs[1]` → `a[2], a[3]` 覆盖另 16个fp4。`sfa_regs` 这一个 uint16_t 里打包了两个 fp8，分别对应这两个 scale block。

## 0x4.2 scale 解码与广播

```cpp
    uint32_t sfa_f16x2, sfb_f16x2, sf_f16x2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(sfa_f16x2) : "h"(sfa_regs));
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(sfb_f16x2) : "h"(sfb_regs));
    asm("mul.rn.f16x2 %0, %1, %2;"    : "=r"(sf_f16x2)  : "r"(sfa_f16x2), "r"(sfb_f16x2));

    uint16_t lane0, lane1;
    uint32_t scale0, scale1;
    asm("mov.b32 {%0,%1}, %2;"  : "=h"(lane0), "=h"(lane1) : "r"(sf_f16x2));
    asm("mov.b32 %0, {%1,%1};"  : "=r"(scale0) : "h"(lane0));  // {s0, s0}
    asm("mov.b32 %0, {%1,%1};"  : "=r"(scale1) : "h"(lane1));  // {s1, s1}
```

`cvt.rn.f16x2.e4m3x2` 是 B200 新增指令，一次把 1个uint16（2个fp8）转成 1个f16x2（2个fp16）。`mul` 后得到 `sf_f16x2 = {sfa[0]*sfb[0], sfa[1]*sfb[1]}`。

然后把两个标量 scale 分别广播成 `{s0,s0}` 和 `{s1,s1}` 的 packed 形式——因为后续 FMA 以 f16x2 为单位，同一个 scale 要同时缩放两个 fp16 元素。

## 0x4.3 核心 FMA 循环

```cpp
    for (int blk = 0; blk < 2; ++blk) {
        uint32_t cvt_a[8], cvt_b[8];

        // 2个uint32（16个fp4）批量转为 8个f16x2
        asm volatile(
            "{ .reg .b8 x0,x1,x2,x3,x4,x5,x6,x7;\n\t"
            "mov.b32 {x0,x1,x2,x3}, %8;  mov.b32 {x4,x5,x6,x7}, %9;\n\t"
            "cvt.rn.f16x2.e2m1x2 %0,x0; ... cvt.rn.f16x2.e2m1x2 %7,x7; }"
            : /* 8个输出 */ : "r"(a[blk*2]), "r"(a[blk*2+1]));
        // b 同理

        uint32_t grp = 0;
        for (int i = 0; i < 8; ++i)
            asm("fma.rn.f16x2 %0,%1,%2,%0;" : "+r"(grp) : "r"(cvt_a[i]), "r"(cvt_b[i]));

        uint32_t scale = (blk == 0) ? scale0 : scale1;
        asm("mul.rn.f16x2 %0,%1,%0;" : "+r"(grp) : "r"(scale));
        asm("add.rn.f16x2 %0,%0,%1;" : "+r"(accum) : "r"(grp));
    }
```

这里有几个值得注意的细节：

- `mov.b32 {x0,..,x3}, reg` 把一个 32位寄存器拆成 4个字节，每字节含 2个fp4
- `cvt.rn.f16x2.e2m1x2` 一次把 1个字节（2个fp4）转为 1个f16x2，8次覆盖 16个fp4
- **所有 cvt 先做完，再统一做 FMA**，这是刻意的指令调度——批量 cvt 让转换单元流水更充分，避免和 FMA 单元竞争端口
- `grp` 是 f16x2，lane0 和 lane1 分别累积偶数/奇数位置的乘积，两者相加才是完整点积

## 0x4.4 最终归约

```cpp
    uint16_t r0, r1, result_f16;
    asm("mov.b32 {%0,%1}, %2;" : "=h"(r0), "=h"(r1) : "r"(accum));
    asm("add.rn.f16 %0,%1,%2;" : "=h"(result_f16) : "h"(r0), "h"(r1));
    float result;
    asm("cvt.f32.f16 %0,%1;"   : "=f"(result) : "h"(result_f16));
    return result;
```

把 `accum`（f16x2）的两个 lane 加起来，升到 f32 返回。`_16x2fp4` 返回 f32 是因为结果还需要通过多次 `__shfl_down_sync` 跨线程累加，f16 精度不够；`_32x2fp4` 走 shared memory 归约，链路短，f16 足够，所以返回 `__half`。

对比总结：

| | `block_scaled_fma_16x2fp4` | `block_scaled_fma_32x2fp4` |
|---|---|---|
| 输入 fp4 数 | 32（`a_regs[2]`） | 64（`a_regs[4]`） |
| scale 输入 | 1×uint16（2个fp8） | 2×uint16（4个fp8） |
| 内层循环次数 | 2 | 4 |
| 返回类型 | `float`（f32） | `__half`（f16） |
| 归约方式 | warp shuffle | shared memory |

# 0x5. 总结

这份 rank1 代码主要有以下几个值得学习的地方：

**缓存控制粒度极细**。针对 A（只读一次）和 B（反复共享）的访问模式，用不同 PTX 修饰符分别处理，甚至对不同 K 值提供不同的特化版本，把 B200 的 L2 缓存利用率拉满。

**数据格式 packed 到底**。全程不手动拆 fp4，而是用 `cvt.rn.f16x2.e2m1x2`、`cvt.rn.f16x2.e4m3x2` 等 B200 新增 PTX 指令直接操作 packed 格式，减少寄存器压力和指令数。

**归约方式因路径而异**。16x2 路径用 `__shfl_down_sync` warp shuffle tree reduce，延迟低适合 THREADS_PER_ROW 较小的情况；32x2 路径的 THREADS_PER_ROW=128 超过一个 warp，先用 shared memory 两级折叠（128→64→32），再 warp shuffle 归约，同时返回 `__half` 而非 `float` 以节省寄存器。

**全程 PTX inline assembly**。从加载（`ld.global.cs`、`L2::evict_last` 等）到 scale 解码（`cvt.rn.f16x2.e4m3x2`）、fp4 转换（`cvt.rn.f16x2.e2m1x2`）、packed FMA（`fma.rn.f16x2`）直到最终归约（`add.rn.f16`、`cvt.f32.f16`），全部用 PTX 手写，绕过编译器的寄存器分配和指令调度，把对硬件的控制粒度拉到最细。

总的来说，这份代码很好地展示了针对带宽受限 kernel 在 B200 上的极致调优方式，细节很多。


