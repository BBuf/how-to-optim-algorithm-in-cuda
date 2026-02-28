> 博客原地址：https://arthurchiao.art/blog/understanding-gpu-performance/ 这里做了翻译
通过 nvidia-smi 等工具报告的 GPU 性能指标可能会产生误导。本文将深入探讨这个问题的本质，以提供更深入的理解。

# 1 NVIDIA `GPU util`：一个令人困惑的现象

即使只有一个任务在 GPU 的一小部分上运行，由 `nvidia-smi` 或其他基于 nvml 的工具报告的 **"GPU util"** 指标也可能显示设备被完全占用，这对用户来说相当令人困惑。

为了更清楚地理解这一点，让我们看看 NVIDIA 开发者论坛上的一个例子(https://forums.developer.nvidia.com/t/some-questions-on-gpu-utilization/191025)：

```c++
__global__ void simple_kernel() {
    while (true) {}
}

int main() {
    simple_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}
```

这段代码会在单个流式多处理器(SM)上启动一个指定的内核(线程)。根据传统理解，GPU 的"利用率"应该按照 **1 / SM数量 * 100%** 来计算。例如：

- 如果 GPU 上有 10 个 SM，那么"GPU 利用率"应该是 10%。
- 如果 GPU 上有 20 个 SM，那么"GPU 利用率"应该是 5%。

然而，我们观察到 nvidia-smi 可能会报告 **"GPU-Util"** 为 100%，如下面的示例输出所示：

```shell
$ nvidia-smi
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-SXM2...  Off  | 00000000:1A:00.0 Off |                    0 |
| N/A   42C    P0    67W / 300W |   2602MiB / 32510MiB |    100%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
```

问题出在哪里？让我们来寻找答案。

# 2 `GPU Util`：一个容易误导的术语？

让我们先做一些搜索来加深理解。

## 2.1 官方文档中的定义

`nvidia-smi` 命令行工具是基于 NVIDIA 管理库(NVML)的，但遗憾的是这个库并不开源。为了寻找一些说明，我们查阅了官方的 NVML(https://developer.nvidia.com/management-library-nvml) 文档。根据文档所述：

> GPU 利用率：报告 GPU 计算资源和内存接口的当前利用率。

这个信息并没有提供我们想要的清晰解释。所以，我们继续往下看。

## 2.2 探索代码

虽然 NVML 库本身并不开源，但我们发现它有一些开源的语言绑定可用。这意味着我们至少可以访问到**结构体和字段定义**，这些通常在 C/C++ 头文件中提供。这里我们选择了 gonvml 项目，它为 NVML 提供了 Golang 绑定。以下是 NVML 头文件中定义 **"GPU Util"** 和 **"Memory Util"** 术语的摘录：


```c++
// https://github.com/NVIDIA/go-nvml/blob/v0.12.0-1/gen/nvml/nvml.h#L210

/**
 * 设备的利用率信息。
 * 每个采样周期可能在1秒到1/6秒之间，具体取决于被查询的产品。
 */
typedef struct nvmlUtilization_st {
    unsigned int gpu;                //!< 在过去的采样周期内，有一个或多个内核在GPU上执行的时间百分比
    unsigned int memory;             //!< 在过去的采样周期内，全局(设备)内存被读取或写入的时间百分比
} nvmlUtilization_t;
```
通过上述注释,我们找到了答案。

## 2.3 解释

根据 NVML 的定义,"利用率"指的是**在过去的采样周期内,某些活动发生的时间百分比**。具体来说:

- **GPU 利用率**: 表示在过去的采样周期内,有一个或多个内核在 GPU 上执行的时间百分比。
- **内存利用率**: 表示在过去的采样周期内,全局(设备)内存被读取或写入的时间百分比。

换句话说,NVML 定义的"利用率"概念可能与我们的常规理解不同。它仅仅衡量设备在给定采样周期内被使用的时间比例,而不考虑在此期间使用了多少流式多处理器(SM)。通常,我们认为"利用率"是指正在使用的 GPU 处理器的比例。

我不确定为什么 NVIDIA 以这种非常规的方式定义"利用率"。但这可能与"USE"(利用率/饱和度/错误)方法论中的"利用率"定义有关。

## 2.4 "USE"方法论

如果你熟悉《Systems Performance: Enterprise and the Cloud》这本书,你可能记得 Brendan Gregg 介绍的"USE"方法论。这个方法论关注三个关键指标:利用率、饱和度和错误。根据"USE"博客,这些术语的定义如下:
- 利用率: 资源忙于处理工作的**平均时间**[2]
- 饱和度: 资源无法处理的额外工作的程度,通常是排队的工作
- 错误: 错误事件的计数

"USE"方法论对"利用率"提供了额外的解释:

> **还有另一种定义**,其中利用率描述了**资源被使用的比例**,因此 100% 的利用率意味着不能再接受更多工作,**这与上述"忙碌"定义不同**。

总的来说,在"USE"方法论中,"利用率"指的是**资源主动服务或工作的时间比例,而不考虑分配的容量**。对于后者,使用"饱和度"这个术语。虽然"USE"方法论为资源使用评估提供了有价值的见解,但重新定义像"利用率"这样一个已经确立的术语可能会导致混淆。许多人仍然倾向于将"利用率"理解为容量使用或饱和度。

如果需要,可以用 **"使用频率"** 这个替代术语来替换"利用率",表示 **设备被使用的频率**。

## 2.5 两个指标来源: NVML / DCGM

在大多数情况下,我们主要关心的指标是与"饱和度"相关的指标。那么,我们可以在哪里找到这些 GPU 指标呢?

有两种流行的收集 GPU 性能指标的方法:

- 使用命令行工具如 `nvidia-smi`,可以输出类似 pretty-print 和 **xml** 格式的数据。
    - 这个工具内部基于 NVML(NVIDIA 管理库)。
    - 它收集高级别的指标,如 GPU 和内存的"利用率"(使用频率),设备温度,功耗等。

- Using services like **dcgm-exporter**, which can output data in Prometheus format.
    - 这个服务基于 DCGM(数据中心 GPU 管理)。
    - 除了高级别的指标,它还可以执行分析并收集关于 GPU 设备的详细**饱和度数据**。

以下是两个显示从 `nvidia-smi` 和 `dcgm-exporter` 收集的指标的仪表板:


![Metrics from nvidia-smi](https://files.mdnice.com/user/59/14e60ce9-4b64-4353-9eed-bd00576e0e1e.png)


注意 GPU 的利用率是 100%。以下是从 `dcgm-exporter` 收集的指标:

![Metrics from dcgm-exporter](https://files.mdnice.com/user/59/5a3cf61f-53ec-4139-bbae-0f80090897c6.png)


我们可以看到 SM 占用率非常低(`<20%`),浮点运算(FP32/FP16/TensorCore)也保持在非常低的百分比,这表明 GPU 没有饱和。

# 3 结论和一般建议

## 3.1 “利用率” vs. 饱和度

不知道 NVML 的设计师是否故意采用了上述的"USE"方法论,但它的"利用率"(包括 GPU 和内存利用率)定义似乎与"USE"标准一致。报告的"利用率"只是表示设备被使用的频率(以时间百分比表示),而不考虑被利用的容量。

## 3.2 一般建议:优先考虑饱和度指标

虽然 `nvidia-smi` 是一个常用且方便的工具,但它并不是性能测量的最佳选择。对于实际部署的 GPU 应用程序,建议使用基于 DCGM 的指标,如 `dcgm-exporter` 提供的指标。

此外,关注饱和度指标是有益的。这些指标包括 FP64/FP32/FP16 激活、张量核心激活百分比、NVLINK 带宽、GPU 内存带宽百分比等。

![Metrics from dcgm-exporter](https://files.mdnice.com/user/59/395d48b6-16b4-41d3-96ea-508868878004.png)


