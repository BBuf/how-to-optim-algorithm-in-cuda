> 我的课程笔记，欢迎关注：https://github.com/BBuf/how-to-optim-algorithm-in-cuda 。



# vector_add 

链接：https://www.gpumode.com/leaderboard/345?tab=rankings

## 描述
 
实现一个 `float16` 的向量加法 Kernel。
 
输入：`tuple(torch.Tensor, torch.Tensor)`，其中两个张量的形状均为 `(N, N)`、数据类型为 `torch.float16`。这些张量来自均值为 0、方差为 1 的正态分布。输出：形状为 `(N, N)`、数据类型为 `torch.float16` 的 `torch.Tensor`。

## Benchmark Shapes

- {"size":1024}
- {"size":2048}
- {"size":4096}
- {"size":8192}
- {"size":16384}

## 参考实现

```python
from utils import make_match_reference
import torch
from task import input_t, output_t


def ref_kernel(data: input_t) -> output_t:
    """
    Reference implementation of vector addition using PyTorch.
    Args:
        data: Tuple of tensors [A, B] to be added.
    Returns:
        Tensor containing element-wise sums.
    """
    A, B = data
    return A + B


def generate_input(size: int, seed: int) -> input_t:
    """
    Generates random input tensors of specified shapes.
    Returns:
        Tuple of tensors [A, B] to be added.
    """
    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)
    A = torch.randn(size, size, device='cuda', dtype=torch.float16, generator=gen).contiguous()
    B = torch.randn(size, size, device='cuda', dtype=torch.float16, generator=gen).contiguous()
    return (A, B)


check_implementation = make_match_reference(ref_kernel)
```

## rank1代码实现

在A100，H100，L4，T4上的代码都一样，一个比较hack的方式，使用了cccl。

```python
#!POPCORN leaderboard vectoradd

try:
    import cuda.parallel.experimental.algorithms as algorithms
except:
    import os
    import subprocess

    if not os.path.exists("cccl"):
        subprocess.check_call(["git", "clone", "https://github.com/NaderAlAwar/cccl.git"])

    subprocess.check_call(["git", "checkout", "gpu-mode-submissions-a100"], cwd="cccl")
    subprocess.check_call(["git", "checkout", "49b7297dbe3abddbc25f937b132b8e6e16202100"], cwd="cccl")

    env = os.environ.copy()
    env["CC"] = "gcc"
    env["CXX"] = "g++"
    env["CMAKE_ARGS"] = "-DCMAKE_CXX_STANDARD=20"

    subprocess.check_call(["pip", "install", "../cuda_cccl"], cwd="cccl/python/cuda_parallel", env=env)
    subprocess.check_call(["pip", "install", "."], cwd="cccl/python/cuda_parallel", env=env)

import functools
import cuda.parallel.experimental.algorithms as algorithms
import torch

from task import input_t, output_t

d_in1 = torch.tensor([1], dtype=torch.float16).cuda()
d_in2 = torch.tensor([1], dtype=torch.float16).cuda()

def op(a, b):
    return a + b

transform = algorithms.binary_transform(d_in1, d_in2, d_in2, op)

@functools.cache
def initialize(shape):
    return torch.empty(shape, dtype=torch.float16).cuda(), shape[0] * shape[1]

def custom_kernel(data: input_t) -> output_t:
    d_in1, d_in2 = data
    d_out, numel = initialize(d_in1.shape)
    transform(d_in1, d_in2, d_out, numel)
    return d_out
```

![](https://files.mdnice.com/user/59/43a26b47-ae94-4427-8092-ba9e613e54a1.png)

![](https://files.mdnice.com/user/59/8618cf05-1b96-4ca9-ad38-cf75420e0fae.png)

## 带宽利用率估算

`vector_add` 对每个元素的最小显存流量是 2 次读 + 1 次写。输入和输出都是 `fp16`，所以每个元素对应：

`2 bytes + 2 bytes + 2 bytes = 6 bytes`

这里按最大 shape `16384 x 16384` 来算。

所以：

- **总元素数**：`16384 x 16384 = 268,435,456`
- **总搬运字节数**：`268,435,456 x 6 = 1,610,612,736 bytes ≈ 1.611 GB`
- **有效带宽**：`1,610,612,736 / time`
- **带宽利用率**：`有效带宽 / 理论带宽`

按截图中的 rank1 耗时估算：

| GPU | 耗时 | 估算有效带宽 | 理论带宽 | 带宽利用率 |
|--|--:|--:|--:|--:|
| A100 | 925.027 us | 1741.15 GB/s | 2039 GB/s | 85.39% |
| H100 | 538.437 us | 2991.67 GB/s | 3350 GB/s | 89.30% |
| L4 | 6487.993 us | 248.24 GB/s | 300 GB/s | 82.75% |
| T4 | 6227.335 us | 258.64 GB/s | 300 GB/s | 86.21% |

带宽都能到80%+说明cccl的elemewise模板已经做到很好了，Leaderboards里面其它手写的cuda代码都比这个代码的带宽利用率低。



# histogram（直方图）

链接：https://www.gpumode.com/leaderboard/341?tab=rankings

## 描述

实现一个 histogram Kernel，用来统计输入张量中有多少元素落入各个 bin。取值范围固定，输入 size 都是 16 的倍数。参考实现里直接使用 `torch.bincount(data, minlength=256)`，因此这里可以把它理解为对取值在 `[0, 255]` 的离散整数做 256 个 bin 的计数。

输入：形状为 `(size,)` 的张量 `data`。

## Benchmark Shapes

- {"contention":10,"size":1310720}
- {"contention":10,"size":2621440}
- {"contention":40,"size":2621440}
- {"contention":90,"size":2621440}
- {"contention":10,"size":5242880}
- {"contention":10,"size":10485760}

## 参考实现

```python
from utils import verbose_allequal
import torch
from task import input_t, output_t


def ref_kernel(data: input_t) -> output_t:
    # 参考实现：直接用 PyTorch 的 bincount 统计每个 bin 的计数
    return torch.bincount(data, minlength=256)


def generate_input(size: int, contention: float, seed: int) -> input_t:
    # 生成直方图输入：基础分布是 [0, 255] 上的随机 uint8
    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)
    
    # 随机生成输入数据
    data = torch.randint(0, 256, (size,), device='cuda', dtype=torch.uint8, generator=gen)

    # 让某个值高频出现，以提高 atomic contention
    evil_value = torch.randint(0, 256, (), device='cuda', dtype=torch.uint8, generator=gen)
    evil_loc = torch.rand((size,), device='cuda', dtype=torch.float32, generator=gen) < (contention / 100.0)
    data[evil_loc] = evil_value

    return data.contiguous()


def check_implementation(data, output):
    # 对比自定义实现与参考实现是否完全一致
    expected = ref_kernel(data)
    reasons = verbose_allequal(output, expected)

    if len(reasons) > 0:
        return "mismatch found! custom implementation doesn't match reference: " + " ".join(reasons)

    return ''
```

## rank截图

这里先贴一下榜单里的实现截图：


![](https://files.mdnice.com/user/59/87c10e79-0c1d-4fe4-9fbc-20c3326d61a0.png)


## 带宽利用率估算

`histogram` 这个算子和 `vector_add` 不一样，它包含大量 atomic add，尤其在高 contention 场景下，真实的显存事务量会明显高于最理想情况。因此这里的带宽估算只能按**最小显存流量下界**来算，作为保守参考。

这里按最大 shape `size = 10485760` 来估算。

最小流量可以近似写成：

- **读取输入**：`10485760 x 1 byte = 10,485,760 bytes`
- **写回输出 histogram**：`256 x 8 bytes = 2,048 bytes`

所以总搬运字节数的下界约为：

`10,485,760 + 2,048 = 10,487,808 bytes ≈ 10.488 MB`

对应公式：

- **有效带宽**：`10,487,808 / time`
- **带宽利用率**：`有效带宽 / 理论带宽`

### A100

| 排名 | 耗时 | 估算有效带宽 | 理论带宽 | 带宽利用率 |
|--|--:|--:|--:|--:|
| rank1 | 36.284 us | 289.03 GB/s | 2039 GB/s | 14.18% |
| rank2 | 38.125 us | 275.09 GB/s | 2039 GB/s | 13.49% |
| rank3 | 44.217 us | 237.19 GB/s | 2039 GB/s | 11.63% |

### H100

| 排名 | 耗时 | 估算有效带宽 | 理论带宽 | 带宽利用率 |
|--|--:|--:|--:|--:|
| rank1 | 23.456 us | 447.12 GB/s | 3350 GB/s | 13.35% |
| rank2 | 26.493 us | 395.86 GB/s | 3350 GB/s | 11.82% |
| rank3 | 31.467 us | 333.29 GB/s | 3350 GB/s | 9.95% |

### L4

| 排名 | 耗时 | 估算有效带宽 | 理论带宽 | 带宽利用率 |
|--|--:|--:|--:|--:|
| rank1 | 68.491 us | 153.13 GB/s | 300 GB/s | 51.04% |
| rank2 | 79.095 us | 132.60 GB/s | 300 GB/s | 44.20% |
| rank3 | 87.895 us | 119.32 GB/s | 300 GB/s | 39.77% |

### T4

| 排名 | 耗时 | 估算有效带宽 | 理论带宽 | 带宽利用率 |
|--|--:|--:|--:|--:|
| rank1 | 115.777 us | 90.58 GB/s | 300 GB/s | 30.19% |
| rank2 | 129.195 us | 81.18 GB/s | 300 GB/s | 27.06% |
| rank3 | 130.399 us | 80.43 GB/s | 300 GB/s | 26.81% |

从这个结果可以看出，`histogram` 的“表观带宽利用率”明显低于 `vector_add`。这并不代表实现一定差，而是因为 histogram 的核心瓶颈更多来自 atomic 冲突、访存热点和同步代价，而不是简单的顺序带宽读写。

## 代码实现

A100,H100,L4的rank1也都是用cccl来直接调用的cccl里面的直方图算法获得最大速度。我们看一下A100和H100上rank2的代码实现就好，比rank1稍慢但差距不大。

```python
import torch
from torch.utils.cpp_extension import load_inline
from typing import List
from triton.testing import do_bench
input_t = output_t = torch.Tensor

# 内联编译 CUDA 扩展。核心思路是先在 shared memory 中做 block 内局部 histogram，
# 再把 block 内结果归约后写回全局 bins，以减少直接打到全局内存上的 atomic 冲突。
add_cuda_source = """

// 1024 threads/block = 32 warps。
// 这里给每个 warp 分配一份私有的 256-bin histogram，总大小就是 256 * 32。
#define HISTOGRAM_SIZE 256 * 32

template <typename scalar_t, typename TVec>
__global__ void histogram_kernel(
    const scalar_t* __restrict__ inp,
    int* __restrict__ bins  // must be int see atomicAdd documentation
) {
    // block 级 shared histogram。
    // 布局可以理解为 [warp_id][bin_id]，每个 warp 独占 256 个 bin。
    __shared__ uint32_t local_hist[HISTOGRAM_SIZE];
    const int tid = threadIdx.x;

    #pragma unroll
    for (int k = 0; k < 8; k++) {
        // 1024 个线程每人清 8 个元素，合计正好把 8192 个槽位全部清零。
        local_hist[tid + (k * 1024)] = 0;
    }

    const size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    constexpr size_t elem_per_thread = sizeof(TVec) / sizeof(uint8_t);

    // 通过 TVec（如 int2 / int4）一次性读取多个 uint8，提高加载吞吐。
    TVec inpV = ((TVec*)inp)[idx];
    // 再把向量寄存器视作标量数组，后面逐个元素更新 histogram。
    scalar_t* in = (scalar_t*)&inpV;

    // 等待 shared memory 清零完成。
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < elem_per_thread; i++){
        scalar_t val = in[i];
        // tid / 32 对应 warp_id。
        // 每个 warp 只更新自己那份 256-bin histogram，从而降低 shared memory atomic 冲突。
        size_t idx = ((tid / 32) * 256) + val;
        atomicAdd(&local_hist[idx], static_cast<int>(1));
    }
    // 等待所有线程完成局部 histogram 更新。
    __syncthreads();

    // 把 32 份 warp-private histogram 归约成 1 份 block histogram，再写回全局内存。
    #pragma unroll
    for (int k = 1; k < 8; k++) {
        local_hist[tid] += local_hist[tid + (k * 1024)];
    }
    if (tid < 512) {
        __syncthreads();
        local_hist[tid] += local_hist[tid + 512];
    }
    if (tid < 256) {
        __syncthreads();
        local_hist[tid] += local_hist[tid + 256];
        // 不同 block 之间依然会同时更新同一个 bin，因此这里仍然需要 atomicAdd。
        atomicAdd(&bins[tid], local_hist[tid]);
    }
}

template <typename scalar_t, typename TVec>
__global__ void histogram_kernel_test(
    const scalar_t* __restrict__ inp,
    int* __restrict__ bins,  // must be int see atomicAdd documentation
    size_t N
) {
    // 小尺寸 fallback：直接原子加到全局 histogram，避免 shared memory 归约的额外成本。
    const size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    constexpr size_t elem_per_thread = sizeof(TVec) / sizeof(uint8_t);

    TVec inpV = ((TVec*)inp)[idx];
    scalar_t* in = (scalar_t*)&inpV;

    #pragma unroll
    for (int i = 0; i < elem_per_thread; i++){
        scalar_t val = in[i];
        // fallback kernel 自己负责处理越界。
        if (idx + i < N) {
            atomicAdd(&bins[val], static_cast<int>(1));
        }
    }
}


torch::Tensor histogram_cuda(torch::Tensor inp) {
    int N = inp.numel();
    // 全局输出 histogram，共 256 个 bin，类型必须是 int，便于 atomicAdd。
    at::Tensor bins = at::zeros(
        {256}, 
        at::TensorOptions().device(at::kCUDA).dtype(at::kInt)
    );

    // 1024 threads/block 对应 32 个 warp，正好和上面的 32 份子 histogram 配套。
    constexpr size_t nthreads = 1024;

    if (N < 1310720) {
        // 小输入：每线程处理 1 个元素，直接走简单版本的 kernel。
        constexpr size_t elem_per_thread = sizeof(uint8_t) / sizeof(uint8_t);
        constexpr size_t elem_per_block = elem_per_thread * nthreads;
        dim3 gridSize((N + elem_per_block - 1) / elem_per_block);
        constexpr dim3 blockSize(nthreads);
        AT_DISPATCH_INTEGRAL_TYPES(inp.scalar_type(), "histogram_kernel", ([&] {
            histogram_kernel_test<scalar_t, uint8_t><<<gridSize, blockSize>>>(
                inp.data_ptr<scalar_t>(),
                bins.data_ptr<int>(),
                N
            );
        }));
    } else if (N > 1310720) {
        // 大输入：使用 int4，一次处理 16 个 uint8，提升向量化加载吞吐。
        constexpr size_t elem_per_thread = sizeof(int4) / sizeof(uint8_t);
        constexpr size_t elem_per_block = elem_per_thread * nthreads;
        dim3 gridSize((N + elem_per_block - 1) / elem_per_block);
        constexpr dim3 blockSize(nthreads);
        AT_DISPATCH_INTEGRAL_TYPES(inp.scalar_type(), "histogram_kernel", ([&] {
            histogram_kernel<scalar_t, int4><<<gridSize, blockSize>>>(
                inp.data_ptr<scalar_t>(),
                bins.data_ptr<int>()
            );
        }));
    } else {
        // 边界 case：N == 1310720 时使用 int2 做折中。
        constexpr size_t elem_per_thread = sizeof(int2) / sizeof(uint8_t);
        constexpr size_t elem_per_block = elem_per_thread * nthreads;
        dim3 gridSize((N + elem_per_block - 1) / elem_per_block);
        constexpr dim3 blockSize(nthreads);
        AT_DISPATCH_INTEGRAL_TYPES(inp.scalar_type(), "histogram_kernel", ([&] {
            histogram_kernel<scalar_t, int2><<<gridSize, blockSize>>>(
                inp.data_ptr<scalar_t>(),
                bins.data_ptr<int>()
            );
        }));
    }
    return bins;
}
```

```python
cpp_source = """
#include <torch/extension.h>
#include <stdint.h>

torch::Tensor histogram_cuda(torch::Tensor inp);
"""

histogram_module = load_inline(
    name='histogram_cuda',
    cpp_sources=cpp_source,
    cuda_sources=add_cuda_source,
    functions=['histogram_cuda'],
    with_cuda=True,
    extra_cuda_cflags=[
        "-arch=sm_90",
        "-O3",
        "--use_fast_math",
    ],  # 实际使用时需要按目标 GPU 架构调整
    #verbose=True,
)

def histogram(data):
    # Python 包装函数：直接调用内联编译出来的 CUDA 扩展。
    return histogram_module.histogram_cuda(data)
    #return torch.bincount(data, minlength=256)

def custom_kernel(data: input_t):
    # Leaderboard 入口函数。
    return histogram_module.histogram_cuda(data)

def cmp(data):
    # 使用 torch.bincount 作为 correctness 基线。
    expected = torch.bincount(data, minlength=256).cuda().int()
    actual = histogram(data)
    try:
        torch.testing.assert_close(expected, actual)
    except AssertionError as e:
        print(str(e))
        return False
    return True

def generate_input(size: int, contention: float, seed: int) -> input_t:
    """
    生成直方图输入：基础分布是 [0, 255] 上的随机 uint8。
    """
    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)
    # 先生成均匀分布的 uint8 输入。
    data = torch.randint(0, 256, (size,), device='cuda', dtype=torch.uint8, generator=gen)
    # 再人为制造热点值，提高 atomic contention。
    evil_value = torch.randint(0, 256, (), device='cuda', dtype=torch.uint8, generator=gen)
    evil_loc = torch.rand((size,), device='cuda', dtype=torch.float32, generator=gen) < (contention / 100.0)
    data[evil_loc] = evil_value
    return data.contiguous()

def correctness():
    # 覆盖题目给定的 benchmark case。
    cases = (
        (6252, 1310720, 10),
        (8841, 2621440, 10),
        (3411, 2621440, 40),
        (8753, 2621440, 90),
        (6252, 5242880, 10),
        (8841, 10485760, 10),
    )
    """
    cases = (
        (9991, 5120, 10),
        (2105, 7840, 10),
        (9999, 30080, 10),
        (4254, 30080, 90),
        (1212, 100_000, 10),
    ) + cases
    """
    for seed, size, contention in cases:
        data = generate_input(size=size, seed=seed, contention=contention)
        N = data.numel()

        def benchit():
            # 单次 benchmark 调用。
            histogram(data)

        try:
            times = do_bench(benchit, warmup=100, rep=300, return_mode="median")
        except RuntimeError as e:
            if "illegal memory" not in str(e):
                print(e)
            print(f"{N=:>16}: CUDA IMA")
            continue

        flops = int(N / times)
        print(f"{N=:>16}: {cmp(data)} {times=:.4f} {flops=:,}")

#correctness()
```


![](https://files.mdnice.com/user/59/6ee0a2a3-61c1-4cbd-bcac-919699985d64.png)

前面这些任务，充满Hack行为的代码很多，例如vector sum里面直接只取前缀样本的近似结果居然也能过 correctness，然后Nader这个人一直用cccl去调用接口取得数个rank1例如sort, vector_add, prefix_sum等等， conv2d里面通过调cudnn的接口获得rank1等等。如果放到cuda agent里面去做了训练数据，感觉一定会让模型出现reward hacking行为需要特别小心。

另外一个发现就是 https://www.gpumode.com/leaderboard/399?tab=rankings amd-fp8-mm 的rank1的代码对cuda 源码用了压缩和解码的方式去提交，这样是不是能避免被大模型洗成语料？

![](https://files.mdnice.com/user/59/778eb93e-cf9c-4046-b27a-385bfaccae9f.png)



