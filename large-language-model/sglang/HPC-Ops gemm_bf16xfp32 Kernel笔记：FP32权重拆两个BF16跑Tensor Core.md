# 0x0. 前言

最近看了一下腾讯 HPC-Ops 里的 `gemm_bf16xfp32` kernel。它计算的是 BF16 激活乘 FP32 权重，源码在这里：

https://github.com/Tencent/hpc-ops/blob/main/src/gemm/sm90/gemm_bf16xfp32.cu

这种计算在 MoE router 里很常见。hidden states 通常是 BF16，router 权重为了保留精度会使用 FP32。但是 Tensor Core 没有直接对应的 BF16 × FP32 指令，如果把激活转成 FP32 去算，速度不理想；如果把权重直接转成 BF16，又会丢掉太多精度。

HPC-Ops 的思路是把一个 FP32 权重拆成两个 BF16：一个保存主要部分，另一个保存前者没有装下的零头。kernel 做两次 BF16 GEMM，最后把结果加起来。这个技巧不复杂，但第一次看很容易被 high、low、scale 这些变量绕进去，所以这里单独记录一下。

CUDA 代码本文不展开，后面会给逐行解析文章。先看实际效果。

# 0x1. SGLang PR #30247 的效果

SGLang PR #30247 在 LongCat-Flash router 上直接使用 `hpc.gemm_bf16xfp32`。PR 地址：

https://github.com/sgl-project/sglang/pull/30247

下面是 H200 上的 microbenchmark。基线 `FP32 MM` 对应 `x.float() @ w.t()`：

| router shape | m | HPC-Ops | FP32 MM | 加速比 |
|---|---:|---:|---:|---:|
| Chat：k=6144, n=768 | 64 | 15.5 us | 35.6 us | 2.31x |
| Chat：k=6144, n=768 | 512 | 38.4 us | 120.4 us | 3.14x |
| Chat：k=6144, n=768 | 8192 | 430.0 us | 1661.1 us | 3.86x |
| Lite：k=3072, n=384 | 64 | 14.8 us | 23.1 us | 1.56x |
| Lite：k=3072, n=384 | 512 | 20.7 us | 44.1 us | 2.13x |
| Lite：k=3072, n=384 | 8192 | 113.1 us | 487.8 us | 4.31x |

可以看到，m 越大收益越明显。Lite 的 `m=8192` 从 `487.8 us` 降到了 `113.1 us`，单次 router GEMM 快了 `4.31x`。

![PR #30247 的 kernel 加速比与端到端 prefill 吞吐提升](assets/gemm_bf16xfp32_pr30247_benchmark.png)

单 kernel 快几倍，放回模型后不会直接得到几倍的吞吐提升，因为 router GEMM 只是整个 forward 的一小部分。PR 还在 H200 TP1 上跑了 LongCat-Flash-Lite-FP8 的模型级 A/B。

GSM8K 一共 1319 题：

| 版本 | GSM8K accuracy | Invalid |
|---|---:|---:|
| main | 0.798 | 0.000 |
| PR #30247 | 0.802 | 0.001 |

这组结果没有看到明显的精度回退。它也不能说明两个计算路径逐位相同，只能说明误差没有破坏这次模型测试。

`bench_one_batch_server` 使用 `input=1024, output=128`，吞吐结果如下：

| batch size | 指标 | main | PR #30247 | 变化 |
|---:|---|---:|---:|---:|
| 1 | input tok/s | 15190 | 15613 | +2.8% |
| 16 | input tok/s | 68588 | 71021 | +3.5% |
| 64 | input tok/s | 62175 | 65509 | +5.4% |
| 1 | output tok/s | 242.4 | 244.8 | +1.0% |
| 16 | output tok/s | 2111.2 | 2104.7 | -0.3% |
| 64 | output tok/s | 5569.1 | 5549.5 | -0.4% |

prefill 一次会处理很多 token，router GEMM 的 m 足够大，所以输入吞吐提升了 `2.8%` 到 `5.4%`。decode 时 m 大致等于 batch size，这里最大只有 64，低于 Lite shape 使用 HPC-Ops 的 `m=128` 门槛，因此输出吞吐基本持平。

# 0x2. 原理：一个 FP32 为什么能拆成两个 BF16

## 2.1 直接转成 BF16 的问题

先看 FP32 和 BF16 的区别：

| 格式 | 符号位 | 指数位 | 显式尾数位 | 有效精度 |
|---|---:|---:|---:|---:|
| FP32 | 1 | 8 | 23 | 24 bit |
| BF16 | 1 | 8 | 7 | 8 bit |

它们都有 8 位指数，所以能表示的数值范围差不多。区别主要在尾数：FP32 有 24 位有效精度，BF16 只有 8 位。把 FP32 权重直接转成 BF16，相当于把后面 16 位信息扔掉。

用 LongCat-Flash Chat 的一个 router shape 测一下：

```python
import torch

torch.manual_seed(0)
x = torch.randn(64, 6144, device="cuda").bfloat16()
w = torch.randn(768, 6144, device="cuda")

ref = x.float() @ w.t()
naive = torch.mm(x, w.bfloat16().t(), out_dtype=torch.float32)
print((naive - ref).abs().max())
```

这次运行的最大绝对误差是 `0.5775`。router 后面要根据 logits 选 top-k 专家，两个专家分数接近时，这个误差有可能改变排序。

## 2.2 high 保存主体，low 保存零头

先不看二进制，用十进制举个例子。假设一种数字格式只能保留 3 位有效数字，现在要保存 `1.234567`：

```text
原数      1.234567
high      1.23
剩余      0.004567
low       4.57        # 把剩余放大 1000 倍，再保留 3 位
重建      1.23 + 4.57 / 1000 = 1.23457
```

只存 `high` 时误差是 `0.004567`。把剩余部分再存一次，重建后的误差变成了 `0.000003`。high 负责前几位，low 负责补上 high 舍入时丢掉的部分。

HPC-Ops 对 FP32 权重做的是同一件事，只是 BF16 一段有 8 位有效精度，缩放倍数从十进制例子里的 1000 换成了二进制的 256：

```python
scale = 1 / 256
w_high = w.to(torch.bfloat16)
w_low = ((w - w_high.float()) / scale).to(torch.bfloat16)
```

这里 `w_high` 是权重的主要部分。`w - w_high.float()` 是剩余部分，再除以 `1/256`，也就是放大 256 倍，存进 `w_low`。

使用时把 low 缩回去：

$$
w \approx w_{high} + \frac{w_{low}}{256}
$$

GEMM 也跟着拆成两次：

$$
y \approx xw_{high}^T + \frac{xw_{low}^T}{256}
$$

两次矩阵乘的输入都是 BF16，因此都可以走 BF16 Tensor Core；累加和最后的合并使用 FP32。

为什么 scale 选 256？因为 256 是 $2^8$。乘除 2 的整数次幂只移动浮点数的指数，不会额外损失尾数。把 low 放大后，它的数值量级也更接近原权重，kernel 最后再乘 `1/256` 还原。

## 2.3 精度和代价

直接转 BF16 时，权重的相对舍入误差大约是 $2^{-8}$。low 保存的是 high 丢下来的残差，残差本身已经小了约 $2^{-8}$；再对它做一次 BF16 舍入，重建误差就落到了大约 $2^{-16}$ 的量级。

还是前面那组输入，换成 high/low 两段：

```python
scale = 1 / 256
w_high = w.to(torch.bfloat16)
w_low = ((w - w_high.float()) / scale).to(torch.bfloat16)

split = (
    torch.mm(x, w_high.t(), out_dtype=torch.float32)
    + scale * torch.mm(x, w_low.t(), out_dtype=torch.float32)
)
print((split - ref).abs().max())
```

最大绝对误差从 `0.5775` 降到了 `0.0012`。这个数字会随输入、shape 和归约顺序变化，不过已经比直接转 BF16 小了两个数量级。

代价也很直观：原来只需做一次 BF16 GEMM，现在要做两次。两份 BF16 权重一共是 4 字节，和一份 FP32 权重占用相同。权重通常在模型加载后拆一次并缓存起来，不需要每次 forward 都重新拆。

两次 BF16 GEMM 是否比一次 FP32 GEMM 更快，要看 GPU 和 shape。Hopper 上 BF16 Tensor Core 的吞吐远高于普通 FP32 路径，m 足够大时这笔交换很划算；m 太小，启动和调度开销占比变高，就不一定有收益，这也是 PR 按 shape 设置 min_m 门槛的原因。

本文不展开 CUDA 实现。感兴趣可以直接看上游源码：

https://github.com/Tencent/hpc-ops/blob/main/src/gemm/sm90/gemm_bf16xfp32.cu

逐行解析在这里：

https://github.com/BBuf/how-to-optim-algorithm-in-cuda/blob/master/cutlass/cute/HPC-Ops%20gemm_bf16xfp32%20kernel%E9%80%90%E8%A1%8C%E8%A7%A3%E6%9E%90%EF%BC%9A%E4%BB%8ECuTe%E5%9F%BA%E7%A1%80%E5%88%B0Hopper%20warp%20specialization.md

# 0x3. 公式推导（可选阅读）

本节由 Codex GPT-5.6 辅助整理和检查。前面的原理理解不依赖这一节，想继续看误差界再往下读。

BF16 采用 round-to-nearest 时，一个正常数的最大相对舍入误差可以记成：

$$
u=2^{-8}
$$

先把 FP32 权重 $w$ 转成 BF16 高位。把这次舍入产生的相对误差写成 $\delta_h$：

$$
w_h=w(1+\delta_h),\qquad |\delta_h|\le u
$$

high 没保存下来的残差是：

$$
r=w-w_h
$$

因为第一次舍入误差不超过 $u$，所以：

$$
|r|\le u|w|
$$

low 保存的是放大后的残差。把第二次 BF16 舍入误差写成 $\delta_l$，scale 记为 $s=1/256$：

$$
w_l=\frac{r}{s}(1+\delta_l),\qquad |\delta_l|\le u
$$

把 high 和 low 合起来：

$$
\begin{aligned}
\widehat{w}
&=w_h+s w_l\\
&=w_h+r(1+\delta_l)\\
&=w+r\delta_l
\end{aligned}
$$

最后剩下的误差是 $r\delta_l$。$r$ 已经小了一个 $u$，low 的舍入又带来一个 $u$，所以：

$$
|\widehat{w}-w|
\le u^2|w|
=2^{-16}|w|
$$

这就是误差量级从 $2^{-8}$ 降到 $2^{-16}$ 的原因。需要注意，$2^{-16}$ 描述的是权重拆分和重建的误差，不代表整个 GEMM 会和 FP32 逐位一致。实际输出还会受到 FP32 累加顺序和正负抵消的影响。
