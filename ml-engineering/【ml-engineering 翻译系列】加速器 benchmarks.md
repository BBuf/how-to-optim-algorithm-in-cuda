> 我的课程笔记，欢迎关注：https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode 
> 本篇文档的来源：https://github.com/stas00/ml-engineering 。这篇文档主要介绍了一个名为"最大可达矩阵乘法 FLOPS (MAMF) 查找器"的基准测试工具。该工具的主要目的是通过测试不同尺寸的矩阵乘法操作，找出特定加速器（如NVIDIA、AMD、Intel等）能够实际达到的最大TFLOPS性能。这个工具很有价值，因为虽然硬件厂商会公布理论TFLOPS，但实际可达到的性能往往低于理论值。通过这个工具，开发者可以获得一个更现实的性能基准，从而更好地评估他们的优化效果。文档详细介绍了工具的使用方法，包括快速测试、详细搜索和特定形状测试等多种使用场景，并提供了针对不同加速器（如A100、MI300X、H100等）的优化建议。同时还包含了完整的Python实现代码，支持多种硬件架构。

# 加速器基准测试

## 最大可达矩阵乘法 FLOPS 查找器

最大可达矩阵乘法 FLOPS (MAMF) 基准测试: mamf-finder.py(https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/benchmarks/mamf-finder.py)

有关详细讨论和各种加速器的数据,请参见最大可达 FLOPS(https://github.com/stas00/ml-engineering/tree/master/compute/accelerator#maximum-achievable-flops)。

虽然一些加速器制造商公布了理论 TFLOPS,但这些通常无法达到。因此,当我们尝试优化软件时,我们没有现实的性能标准来比较。模型 FLOPS 利用率 (MFU) 指标衡量实际达到的 TFLOPS 与理论 TFLOPS 的比值。通常,当 MFU 达到约 50% 时被认为是一个胜利。但这并不能告诉我们离真正可达到的吞吐量还有多远。

这个基准测试扫描各种大型矩阵乘法形状,并报告它记录的最高可达 TFLOPS。由于 Transformer 的训练和部分推理工作负载主要由大型矩阵乘法操作主导,因此可以安全地使用在每个加速器上测量的最佳矩阵乘法 TFLOPS 作为最大可达矩阵乘法 FLOPS (MAMF) 的粗略估计。现在,我们可以使用模型可达矩阵乘法 FLOPS 利用率 (MAMFU) 来代替之前使用的 MFU。

因此,现在你可以将训练或推理中测量的 TFLOPS 与一个现实的数字进行比较。由于你现在会更接近 100%,所以更容易知道何时停止优化。

目前支持的高端架构:
- NVIDIA: V100, A100, H100, ...
- AMD: MI250, MI300X, ...
- Intel Gaudi2+

公平性说明:
- 如果你能找到更好、更有效的方法来检测最佳矩阵乘法 TFLOPS,将每个新加速器视为黑盒,请kindly发送包含改进和生成的日志文件的 PR。
- 另外,如果你知道这个基准测试应该在特殊条件下运行以显示最佳结果,例如一些内核设置或类似的,请提交 PR 添加这些特殊说明。例如,对于 AMD MI300X,我被告知禁用 numa_balancing 应该会有帮助。

### 特定架构注意事项:

在运行基准测试之前,请遵循以下特殊设置说明以获得最佳结果:

**MI300x**:

关闭numa_balancing以获得更好的性能:
```
sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'
```

### 使用示例

在下面的范围中,`N`是rReduce维度,使得`(MxN)*(NxK)=(MxK)`,我们会打印出测得最高TFLOPS的MxNxK形状。

默认情况下,我们对每个形状使用50次预热迭代和100次测量迭代,然后选择最快的结果(而不是平均值)。你可以通过参数`--num_warmup_iterations`和`--num_iterations`分别更改迭代次数。

这里我们执行`torch.mm(MxN,NxK) -> MxK`

1. 快速运行(1分钟以内) - 应该能达到最大可达结果的80-90% - 适合快速尝试,但不足以获得高精度测量。

```
./mamf-finder.py --m_range 0 20480 256 --n 4096 --k 4096 --output_file=$(date +"%Y-%m-%d-%H:%M:%S").txt
```

2. 更详尽的搜索(将花费更长时间) - 但你可以在运行足够长时间后按Ctrl-C终止,并获得到目前为止的最佳结果:

```
./mamf-finder.py --m_range 0 5376 256 --n_range 0 5376 256 --k_range 0 5376 256 --output_file=$(date +"%Y-%m-%d-%H:%M:%S").txt
```

3. 一个超长的穷举搜索（可能需要几天时间）- 但你可以在运行足够长时间后按Ctrl-C终止它，并获得到目前为止的最佳结果：

```
./mamf-finder.py --m_range 0 20480 256 --n_range 0 20480 256 --k_range 0 20480 256 --output_file=$(date +"%Y-%m-%d-%H:%M:%S").txt
```

4. 如果你想测量训练中使用的特定形状,请使用确切的形状,而不是范围。例如,假设你想测量1024x1024x1024 - 你可以运行:

```
./mamf-finder.py --m 1024 --n 1024 --k 1024 --output_file=$(date +"%Y-%m-%d-%H:%M:%S").txt
```

5. 加速器特定范围搜索建议

然而，不同的加速器似乎有不同的形状范围可以达到最佳 TFLOPS，因此很难建议一个适用于所有加速器的范围。相反，这里根据实验和贡献者的建议提供一些建议：

- **A100** + **MI300X**

```
./mamf-finder.py --m_range 0 5376 256 --n_range 0 5376 256 --k_range 0 5376 256 --output_file=$(date +"%Y-%m-%d-%H:%M:%S").txt
```

- **H100**

```
./mamf-finder.py --m_range 0 20480 256 --n_range 0 20480 256 --k_range 0 20480 256 --output_file=$(date +"%Y-%m-%d-%H:%M:%S").txt
```

To understand better which shapes give the highest matmul FLOPS for a particular accelerator, see Vector and matrix size divisibility(../../../training/performance/README.md#vector-and-matrix-size-divisibility).


### 结果

我目前收集到的测量结果可以在最大可达矩阵乘法 FLOPS 比较表(https://github.com/stas00/ml-engineering/tree/master/compute/accelerator#maximum-achievable-matmul-flops-comparison-table)中找到。当我能够访问某个特定加速器时,我会亲自运行基准测试;当我无法访问时,是热心的贡献者们投入时间获得了这些数据。因此,我非常感谢这些贡献者(https://github.com/stas00/ml-engineering/blob/master/contributors.md)。

## `mamf-finder.py` 代码分析

```python
#!/usr/bin/env python

"""

This is Maximum Achievable Matmul FLOPS (MAMF) Finder

For discussion and multiple important nuances please refer to
https://github.com/stas00/ml-engineering/tree/master/compute/accelerator/benchmarks#maximum-achievable-matmul-flops-finder

Credits:
- Parts of this benchmark have been derived from https://github.com/EleutherAI/cookbook/tree/main/benchmarks/sizing (highly recommended!)
- Imtiaz Sajwani: HPU porting

"""

from pathlib import Path

import argparse
import datetime
import numpy as np
import os
import platform
import re
import shlex
import signal
import sys
import time
import torch


# 尝试导入HPU相关模块
has_hpu = False
try:
    import habana_frameworks.torch as ht
    if torch.hpu.is_available():
        has_hpu = True
except ModuleNotFoundError:
    pass

# 获取当前文件所在目录的绝对路径
file_dir = os.path.abspath(os.path.dirname(__file__))



### 架构特定的辅助类 ###

class Arch:
    def __init__(self):
        self.arch = "unknown"

    def __repr__(self):
        return self.arch

class CUDAArch(Arch):
    """ 共享于CUDA和ROCm: NVIDIA + AMD """
    def __init__(self):
        if torch.version.hip is not None:
            self.arch = "rocm"
        else:
            self.arch = "cuda"

    def device(self):
        return torch.device('cuda:0')

    def name(self):
        return self.arch

    def device_info(self):
        return torch.cuda.get_device_properties(device)

    def compute_info(self):
        if self.arch == "rocm":
            return f"hip={torch.version.hip}, cuda={torch.version.cuda}"
        else:
            return f"cuda={torch.version.cuda}"

    def event(self, enable_timing=True):
        return torch.cuda.Event(enable_timing)

    def synchronize(self):
        torch.cuda.synchronize()

class HPUArch(Arch):
    """ Intel Gaudi* """
    def __init__(self):
        self.arch = "hpu"

    def device(self):
        return torch.device('hpu')

    def name(self):
        return self.arch

    def device_info(self):
        return torch.hpu.get_device_properties(device)

    def compute_info(self):
        return f"hpu={torch.version.hpu}"

    def event(self, enable_timing=True):
        return ht.hpu.Event(enable_timing)

    def synchronize(self):
        ht.hpu.synchronize()


def get_accelerator_arch():
    """
    返回: CUDAArch 或 HPUArch 对象
    """
    # cuda / rocm
    if torch.cuda.is_available():
        return CUDAArch()

    # hpu
    if has_hpu:
        return HPUArch()

    raise ValueError("目前只支持 cuda, rocm 和 hpu")

# 获取加速器架构
arch = get_accelerator_arch()



### 辅助类 ###

class Tee(object):
    def __init__(self, filename, verbose):
        # 创建输出文件的目录（如果不存在）
        Path(filename).resolve().parent.mkdir(parents=True, exist_ok=True)
        self.file = open(filename, "w")
        self.verbose = verbose
        if self.verbose:
            self.stdout = sys.stdout

    def write(self, message):
        if self.verbose:
            self.stdout.write(message)
        # 替换控制台中的`\r`和`033\[K`，这些在日志文件中不需要
        message = re.sub(r"(\r|\033\[K)", "\n", message)
        self.file.write(message)

    def flush(self):
        self.file.flush()
        if self.verbose:
            self.stdout.flush()


def print_benchmark_header(dtype, device, notes="None"):
    """打印基准测试的头部信息"""

    device_info = arch.device_info()
    compute_info = arch.compute_info()

    print(f"""
基准测试开始于 {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}

** 命令行:
{sys.executable} {" ".join(map(shlex.quote, sys.argv))}

** 数据类型: {dtype}

** 平台/设备信息:
{" ".join(platform.uname())}
{device_info}

** 关键软件版本:
torch={torch.__version__}
{compute_info}

** 附加说明:
{notes}

{"-" * 80}

""")

# 基本GEMM的基准测试
def benchmark_mm(m, n, k, dtype, device, num_iterations, num_warmup_iterations):
    start = arch.event(enable_timing=True)
    end = arch.event(enable_timing=True)

    # 创建随机矩阵
    A = torch.randn(m, n, dtype=dtype, device=device)
    B = torch.randn(n, k, dtype=dtype, device=device)
    C = torch.empty(m, k, dtype=dtype, device=device)

    times = np.zeros(num_iterations+num_warmup_iterations)
    for i in range(num_warmup_iterations + num_iterations):
        with torch.no_grad():
            start.record()
            torch.mm(A, B, out=C)
            end.record()
        arch.synchronize()
        times[i] = start.elapsed_time(end)
    times = times[num_warmup_iterations:]  # 去掉预热迭代的时间
    elapsed_time = np.amin(times)/1000  # 取最快时间，转换为秒
    tflops = (2 * m * n * k) / (elapsed_time * 10**12)  # 计算TFLOPS
    return tflops


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 设置命令行参数
    m_group = parser.add_mutually_exclusive_group(required=True)
    m_group.add_argument("--m", nargs="+", type=int, help='GEMM的第一个维度，可以输入任意数量的参数')
    m_group.add_argument("--m_range", nargs='+', type=int, help="GEMM的第一个维度，[开始,结束,步长]")

    n_group = parser.add_mutually_exclusive_group(required=True)
    n_group.add_argument("--n", nargs="*", type=int, help='GEMM的共享维度，可以输入任意数量的参数')
    n_group.add_argument("--n_range", nargs='+', type=int, help="GEMM的共享维度，[开始,结束,步长]")

    k_group = parser.add_mutually_exclusive_group(required=True)
    k_group.add_argument("--k", nargs="*", type=int, help='GEMM的最后一个维度，可以输入任意数量的参数')
    k_group.add_argument("--k_range", nargs='+', type=int, help="GEMM的最后一个维度，[开始,结束,步长]")

    parser.add_argument("--num_iterations", type=int, default=100, help='用于基准测试每个GEMM的迭代次数')
    parser.add_argument("--num_warmup_iterations", type=int, default=50, help='预热迭代次数')
    parser.add_argument("--cuda_device", type=int, default=0, help="运行基准测试的CUDA设备")
    parser.add_argument("--output_file", type=str, default=f"{file_dir}/results/mm.out")
    parser.add_argument("--notes", type=str, default="", help="添加到输出文件头部的特定基准测试说明")
    parser.add_argument("--verbose", default=True, action=argparse.BooleanOptionalAction, help='是否同时输出到stdout和output_file?')
    args = parser.parse_args()

    m = args.m
    n = args.n
    k = args.k

    dtype = torch.bfloat16
    device = arch.device()

    # 处理范围参数
    if m is None:
        start, stop, step = args.m_range
        if start == 0:  # 维度不能为0
            start = step
        m = np.arange(start, stop, step)
    if n is None:
        start, stop, step = args.n_range
        if start == 0:  # 维度不能为0
            start = step
        n = np.arange(start, stop, step)
    if k is None:
        start, stop, step = args.k_range
        if start == 0:  # 维度不能为0
            start = step
        k = np.arange(start, stop, step)

    sys.stdout = Tee(args.output_file, args.verbose)
    print_benchmark_header(dtype, device, args.notes)

    # 用于中断运行时仍能报告最佳结果
    def sigkill_handler(signum, frame):
         finish()
         sys.exit(1)

    signal.signal(signal.SIGINT, sigkill_handler)

    best_tflops = 0
    best_config = ""
    num_shapes = 0
    start_time = time.time()

    def finish():
        time_delta = time.time() - start_time
        time_str = str(datetime.timedelta(seconds=time_delta)).split(".")[0]
        print("", end="\033[K")
        print(f"最佳结果为 {best_tflops:.1f}TFLOPS @ {best_config} (尝试了 {num_shapes} 种形状)")
        print(f"耗时: {time_str}")

    # 注：对于MI300X，转置版本似乎效果更好

    # 遍历所有要基准测试的尺寸
    for M in m:
        for N in n:
            for K in k:
                num_shapes += 1
                tflops = benchmark_mm(M, N, K, dtype, device, args.num_iterations, args.num_warmup_iterations)
                cur_config = f"{M}x{N}x{K}"
                if tflops > best_tflops:
                    best_tflops = tflops
                    best_config = f"{M}x{N}x{K} (MxNxK)"
                print(f"{num_shapes:>6} | {tflops:6.1f} TFLOPS @ {cur_config:<20} | 最佳: {best_tflops:6.1f} TFLOPS @ {best_config}", end="\r")
    finish()

```
