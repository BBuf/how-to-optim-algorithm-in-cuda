# 0x0. TL;DR

SGLang 的 GPU kernel 原来分散在 `jit_kernel`、`sgl-kernel`、各子系统的 `triton_ops` 和具体模型文件中。`sglang.kernels` 将公开调用入口统一为 `sglang.kernels.ops.<group>.<fn>(...)`，再通过惰性 registry 和两个分发前端连接 JIT、AOT、Triton 等实现。分层方式参考了 PyTorch 的 LightSeek TokenSpeed-kernel（https://pytorch.org/blog/lightseek-tokenspeed-kernel/），但整套实现位于 SGLang 内部。对应的设计与迁移记录见 RFC #29630（https://github.com/sgl-project/sglang/issues/29630）。

这次 `sglang.kernels` 的设计和实现主要由 Anthropic 的 Claude Fable 5 通过 Claude Code 完成，包括分类方案、迁移脚本、注册与调用接口，以及本文使用的 benchmark；我们负责 review 和最终决策。除了整理调用边界，我们还补充了真实负载 benchmark，让 Agent 能用实际 shape 验证性能，而不只是检查代码能否编译。

本文先介绍 `sglang.kernels` 的结构和调用方式，再比较 SGLang、FlashInfer 与 PyTorch 在 H200 上的 kernel 性能。结果包括：norm 和激活与 FlashInfer 基本持平，MoE gate 与 per-token FP8 量化快于 PyTorch 组合写法，DiT 融合算子相对非融合路径有 1.3–2.2 倍收益。全文只涉及 NVIDIA，所有数字都是单 kernel microbenchmark，不能直接换算成端到端吞吐提升。

# 0x1. sglang.kernels 的整体架构

整体架构如下图：

![sglang.kernels 统一命名空间的四层结构](https://files.mdnice.com/user/59/05fce7e3-2475-4d01-8af5-38feafeadb5d.png)

从上到下分四层，每层职责单一。

**PUBLIC API 层**。对外只有一种入口形式：

```python
from sglang.kernels.ops.layernorm import rmsnorm
from sglang.kernels.ops.activation import silu_and_mul
from sglang.kernels.ops.kvcache import reshape_and_cache_flash
```

即 `sglang.kernels.ops.<group>.<fn>(...)`。所有可调用算子按功能分成 18 个组：`activation`、`attention`、`communication`、`diffusion`、`elementwise`、`embeddings`、`gemm`、`grammar`、`kv_canary`、`kvcache`、`layernorm`、`lplb`、`mamba`、`memory`、`moe`、`quantization`、`sampling`、`speculative`。运行时代码和测试统一从这里 import，不需要关心底下是 JIT、AOT 还是 Triton。

**DISPATCH 层**。有两个前端，每个 op 只承担它需要的那种分发开销：

- `select_kernel()` / `get_kernel()` 是默认路径，做固定解析：给一个 op（可选带 backend），返回一个 callable。没有运行时选择——要么这个 op 只有一个实现直接返回，要么调用方显式指定 `backend=...`。绝大多数 op（约 166 个）走这条，设备过滤会跳过当前设备用不了的实现。
- `BaseFusedOp.forward()` 面向多后端可互换的算子：一个逻辑算子，每个后端一个 `forward_<backend>` 方法，共用一个签名，运行时按 priority 和 capability 自动选后端，选不到就回退到 `forward_native`（纯 torch 参考实现）。这类算子约 9 个，典型的是 `silu_and_mul`，优先级链为 `jit → aiter → torch`。

这两个前端的用法、以及如何注册一个 kernel，下一节用真实代码说明。

**REGISTRY + METADATA 层**。核心是 `spec.py`，torch-free 且惰性：

- `Registry`：op → `[KernelSpec]` 的映射。注册时只记元数据，用 `"module:attr"` 字符串描述实现位置；`import sglang.kernels.ops` 时不会真的 import 任何实现，torch、sgl_kernel、JIT 编译都推迟到 kernel 第一次被调用。
- `KernelSpec`：一条实现的描述，包含 op、backend、target（`module:attr`）、capabilities（一个 frozenset）、format_signature。
- `CapabilityRequirement`：OR 语义的能力集合，`.CUDA / .HIP / .NPU` 通过 `PlatformInfo.detect()` 和当前设备匹配。

这一层的关键不变式是 `import sglang.kernels.ops` 必须保持 metadata-only，不能把 `sgl_kernel` 或 JIT 基建拉进来。仓库里有一个测试专门守这一点：import 之后检查 `sys.modules` 里没有重型模块。

**BACKENDS × DEVICES 层**。这是"实现来源（provenance）"和"设备能力（capability）"的叉乘：AOT（`sgl_kernel` wheel）、JIT（`sglang.kernels.jit`）、TRITON、CUTE_DSL、FLASHINFER、DEEPGEMM、AITER、TORCH_NPU、TORCH、TORCH_COMPILE，各自在 CUDA / HIP / NPU / CPU 上有不同的可达范围。图里画的是 provenance 级别的设备可达性；真实覆盖由每个 `(op, backend)` 自己的 `CapabilityRequirement` 决定。本文只看 CUDA 这一列。

JIT 这一行现在指向 `sglang.kernels.jit`，不再是老的 `sglang.jit_kernel`。原来的 `jit_kernel` 包已删除，共享的 JIT 构建和运行时基建（`utils/`、`csrc/`、`include/`、`__main__`）搬进了 `sglang/kernels/jit/`，每个 JIT-backed 算子则归入对应的功能组。`KERNEL_PATH` 也改为从 `sglang.kernels.jit` 解析 csrc/include。

# 0x2. Kernel 的注册与统一调用

下面是具体操作：往 `sglang.kernels` 里加一个 kernel，以及从模型代码里调用它。本节代码均摘自仓库。

## 2.1 注册：一条 `KernelSpec` = 一个实现

注册入口是 `register_kernel`，它接收一个 `KernelSpec`。注册的是元数据，不是 kernel 本身：

```python
from sglang.kernels.registry import register_kernel
from sglang.kernels.spec import KernelBackend, KernelSpec

register_kernel(
    KernelSpec(
        op="attention.decode_attention_fwd",     # "<group>.<name>"，公开查找键
        backend=KernelBackend.TRITON,             # provenance：这份实现从哪来
        target="sglang.kernels.ops.attention.decode_attention:decode_attention_fwd",
    )
)
```

三个字段的含义：

- `op` 是 `"<group>.<name>"`，即公开的逻辑算子名，调用方只认它；
- `backend` 是 provenance（实现来源）：`JIT` / `AOT` / `TRITON` / `CUTE_DSL` / `FLASHINFER` / `DEEPGEMM` / `AITER` / `TORCH_NPU` / `TORCH`，描述这份实现的来源，而不是设备。同一个 backend（比如 AOT）可以给多个设备各注册一条；
- `target` 是 `"module:attr"` 字符串，真正的 callable 由 `KernelSpec.load()` 惰性 import。注册时不会发生任何 import，这就是 metadata-only 不变式：`import sglang.kernels.ops` 只往 registry 写元数据，torch / sgl_kernel / JIT 编译都推迟到 kernel 第一次被调用。

设备/架构门控挂在 `capabilities` 上，是一个 `CapabilityRequirement` 的 frozenset，OR 语义（任一条满足即可用），空集表示处处可跑。几种常用写法在各组的 `__init__.py` 里预先起了别名：

```python
from sglang.kernels.spec import CapabilityRequirement

_CUDA     = frozenset({CapabilityRequirement.CUDA})                             # 任意 CUDA 卡
_CUDA_HIP = frozenset({CapabilityRequirement.CUDA, CapabilityRequirement.HIP})  # CUDA 或 HIP
_HIP      = frozenset({CapabilityRequirement.HIP})                              # 只在 ROCm 上

# 普通的 CUDA-only kernel：capabilities 给 _CUDA
register_kernel(
    KernelSpec(
        op="gemm.dsv3_fused_a_gemm",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.gemm.dsv3_fused_a_gemm:dsv3_fused_a_gemm",
        capabilities=_CUDA,
    )
)

# 要限定到具体架构（SM100+）：用 CapabilityRequirement.cuda(min_sm=...)
register_kernel(
    KernelSpec(
        op="quantization.mxfp8_...",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.quantization.xxx:yyy",
        capabilities=frozenset({CapabilityRequirement.cuda(min_sm=(10, 0))}),  # 仅 SM100+
    )
)
```

`_CUDA` / `_CUDA_HIP` / `_HIP` 是最常见的三种设备门控：只跑 CUDA 用 `_CUDA`，CUDA 和 ROCm 都跑用 `_CUDA_HIP`（OR 语义在这里发挥作用），只在 AMD 上用 `_HIP`。下一节 `silu_and_mul` 的例子里，AOT 用的就是 `_CUDA_HIP`，JIT 用 `_CUDA`，aiter 用 `_HIP`。要限定架构，把 `CapabilityRequirement.cuda(min_sm=(10, 0))` 放进 frozenset 即可，它表示 SM100+。SGLang 里有一批 SM100（Blackwell）独占的 kernel，比如 0x3 里 diffusion 那个 Qwen-Image 专用的 norm+scale+shift（AdaLN）融合 kernel，在 H200 上会被 `CapabilityRequirement` 判定为不可用、自动 skip，调用方不需要写任何 `if`。门控是注册元数据的一部分，由选择器按当前平台过滤。

## 2.2 调用：两个分发前端

调用方只写 `sglang.kernels.ops.<group>.<fn>(...)`。底下按算子性质走两条前端之一。

**前端一：固定解析（`select_kernel` / `get_kernel`）。** 绝大多数算子（约 166 个）走这条，它是确定性的，没有偏好排序：

- 只注册了一个 backend，直接返回它；
- 注册了多个，按当前平台做硬能力过滤（`CapabilityRequirement` 检查，不是打分）。过滤后恰好剩一个，就是固定调用路径；剩多个，调用方必须显式指定 `backend=...`。

```python
from sglang.kernels.selector import get_kernel

fn = get_kernel("attention.decode_attention_fwd")  # 返回 callable，结果被缓存
```

`get_kernel` 是公开 `ops.*` wrapper 背后的快路径：首次调用解析并 import backend，之后命中缓存，不再重复解析。

**前端二：多后端可互换（`BaseFusedOp`）。** 少数算子（约 9 个）一个逻辑算子有多个语义等价、可互换的后端实现，用 `BaseFusedOp`：每个后端一个 `forward_<backend>` 方法，共用一个签名，运行时按 `priority` 自动选、按 capability 逐调用过滤，选不到就回退到 `forward_native`。以 `silu_and_mul` 为例：

```python
class SiluAndMulOp(_GatedActivationOp):
    op = "activation.silu_and_mul"
    # 优先级链，best first：CUDA 上解析到 JIT，HIP 上解析到 AOT
    priority = (KernelBackend.JIT, KernelBackend.AOT, KernelBackend.AITER, KernelBackend.TORCH)
    capabilities = {
        KernelBackend.AOT:   _CUDA_HIP,   # AOT 覆盖 CUDA+HIP
        KernelBackend.JIT:   _CUDA,       # JIT 只 CUDA
        KernelBackend.AITER: _HIP,        # aiter 只 HIP，排在 AOT 之下
    }

    def forward_native(self, input, out=None): ...   # 必须实现：纯 torch 参考，作为正确性基准
    def forward_aiter(self, input, out=None):  ...    # ROCm 上的 aiter 实现
    # forward_jit / forward_aot 由基类按 kernel_attr 派生
```

这里有四个行为需要注意：

- `forward_native` 必须实现，它是其它后端对拍的正确性基准；
- 某个后端是否"可用"，判据是子类有没有 override 对应的 `forward_<backend>`；
- `set_fused_op_backend(KernelBackend.TORCH)` 可以把所有 fused op 强制切到某一个后端，用于对着参考实现 bisect 数值 bug；
- `enable_fused_op_trace()` 会记录每次 fused-op 调用的 `(op, backend, 张量 shape/dtype)`。完整跑一次模型后，可以得到它实际使用的算子和输入形状，而不必人工猜测 benchmark shape。把这份记录和对应的 `bench_*.py` 交给 Agent，就能明确要优化哪个 op、覆盖哪些输入，以及如何验证性能变化。

## 2.3 分类原则

kernel 归到哪个组，原则是功能优先：看它实际做什么（rmsnorm 进 `layernorm`，silu_and_mul 进 `activation`，稀疏 MLA 进 `attention`），跨功能的融合算子按主功能拆。迁移使用 `git mv`，调用点只改 import 路径，kernel 本体不做修改；如果精度测试失败，应先检查调用路径或是否有迁移遗漏。18 个组的清单由守卫测试 `test_kernels_namespace` 硬编码兜底，加错组或删错组时 CI 会报错。

# 0x3. LLM 与 Diffusion Kernel 的跨框架性能评测

下面看性能。本节选取 SGLang 在 LLM 和 diffusion 路径中使用的一批常用 kernel，与 FlashInfer、PyTorch 的对应实现进行对比。

测试全部在单卡 H200（SM90）上运行，环境为 sgl_kernel 0.4.4 / FlashInfer 0.6.14 / PyTorch 2.11+cu130。每个数字是 `triton.testing.do_bench` 的中位数（µs，越低越快）；计时前先与 PyTorch 参考实现对拍，确保比较的是同一个计算。这些是单 kernel 数据，不能直接换算成端到端吞吐提升，实际收益还取决于该算子在一次 forward 中的占比。

## 3.1 LLM 侧：norm/激活与专用库持平，MoE gate 和 FP8 量化领先框架原生

![SGLang 自研 LLM kernel 与 FlashInfer / PyTorch 的跨框架对比（H200）](https://files.mdnice.com/user/59/01467efa-29fa-4179-a062-746447c92b26.png)

LLM 侧的结果分两类。

第一类是 norm 和激活（`fused_add_rmsnorm`、`silu_and_mul`）。这两个算子使用频繁，各框架也都有针对性优化。具体数字如下：

| kernel | num_tokens | PyTorch (µs) | FlashInfer (µs) | SGLang (µs) |
|---|---:|---:|---:|---:|
| fused_add_rmsnorm | 2048 | 339.1 | 65.6 | 67.1 |
| fused_add_rmsnorm | 8192 | 1160.5 | 234.4 | 234.5 |
| silu_and_mul | 2048 | 88.3 | 27.6 | 27.6 |
| silu_and_mul | 8192 | 314.6 | 88.1 | 88.3 |

在这两个算子上，SGLang 与 FlashInfer 基本持平，没有明显领先。两套实现都比表中的 PyTorch 组合写法快 3–5 倍；即使不调用 FlashInfer，SGLang 自带的实现也处在相同的性能区间。

第二类是 SGLang 自己实现的算子（`topk_softmax` 这个 MoE gate、per-token FP8 量化）。框架原生（PyTorch）没有对应的融合实现，只能用一串通用 op 拼出来，差距因此比较明显（图①④）：

| kernel | num_tokens | PyTorch (µs) | SGLang (µs) | 加速比 |
|---|---:|---:|---:|---:|
| topk_softmax（MoE gate） | 2048 | 49.0 | 11.2 | 4.4x |
| topk_softmax（MoE gate） | 8192 | 123.5 | 20.6 | 6.0x |
| per-token FP8 量化 | 2048 | 197.5 | 20.2 | 9.8x |
| per-token FP8 量化 | 8192 | 640.4 | 80.9 | 7.9x |

MoE gate（DeepSeek / Qwen 这类 MoE 每层都要执行）比 PyTorch 组合写法快 4.4–6.0 倍，per-token FP8 量化快 7.9–9.8 倍。PyTorch 没有这两个算子的对应融合实现，因此需要执行多个通用 op。

## 3.2 Diffusion 侧：DiT 融合算子对非融合路径的加速

![SGLang 自研 diffusion 融合 kernel 与非融合路径的对比（H200）](https://files.mdnice.com/user/59/f469b4b5-aac1-4289-a88d-25f8f6c1fdc9.png)

SGLang 的 diffusion 路径也维护了一批融合 kernel。DiT block 中常见“norm 之后乘 gate，再加 residual”一类 element-wise 组合；PyTorch 路径需要启动多个 kernel 并产生临时张量，SGLang 的 CUDA / CuTe-DSL 实现将它们合并为一次 kernel 调用。下面选取三个常用算子，对比 SGLang 自己的非融合回退路径（PyTorch + FlashInfer 组合，通过环境变量切换，两条路径数值一致）：

| 融合 kernel（用途） | seq | 非融合 (µs) | SGLang 融合 (µs) | 加速比 |
|---|---:|---:|---:|---:|
| rmsnorm+tanh(gate)·x+residual（Qwen-Image DiT block） | 4096 | 72.6 | 39.9 | 1.82x |
| rmsnorm+tanh(gate)·x+residual | 8192 | 134.3 | 60.8 | 2.21x |
| QK-norm + RoPE（DiT attention） | 8192 | 177.5 | 135.4 | 1.31x |
| residual + gate·update（DiT residual） | 8192 | 79.4 | 54.0 | 1.47x |

序列越长，融合省下的 kernel launch 和访存越多，收益越大，Qwen-Image DiT block 那个算子在 8192 上达到 2.21 倍。

并非所有 diffusion 算子都能从自定义融合中获益。例如 `group_norm + silu` 中，PyTorch 的 `F.group_norm` 已经使用优化过的融合 CUDA kernel，SGLang 的 Triton 版本只能与其持平。是否值得再做融合，取决于现有路径是不是仍要启动多个 kernel、写入中间结果。

# 0x4. 小结

这次重构把 kernel 的公开入口统一为 `kernels.ops.<group>`，并用惰性 registry 保存实现来源和设备能力。多数算子通过 `get_kernel` 固定解析，少量可互换实现则由 `BaseFusedOp` 选择后端；共享的 JIT 基建也从 `sglang.jit_kernel` 搬到了 `sglang.kernels.jit`。现在要查一个算子的实现、后端和设备限制，可以直接从注册信息入手，不必再遍历整个仓库。

H200 microbenchmark 中，norm 和激活与 FlashInfer 处在同一水平；`topk_softmax` 和 per-token FP8 量化明显快于 PyTorch 组合写法；DiT 融合算子的收益为 1.3–2.2 倍，而 `group_norm+silu` 与 PyTorch 持平。它们仍然只是单 kernel 数据，端到端收益需要结合模型 trace 判断。

对后续的 Agent 优化任务，流程可以固定下来：先用 `enable_fused_op_trace()` 收集真实的 op、backend、shape 和 dtype，再定位对应实现，最后用同一组输入运行 benchmark。本文所述的重构和测试基本就是按这套分工完成的。

> 完整的设计讨论和分阶段落地见 RFC #29630（https://github.com/sgl-project/sglang/issues/29630）。

## Acknowledgements

感谢 SGLang team 对这次重构和 RFC 落地的支持。

特别感谢 Baizhou Zhang、@zcnrex（https://github.com/zcnrex）、@merrymercy（https://github.com/merrymercy）和 @DarkSharpness（https://github.com/DarkSharpness）参与讨论并帮助完善设计。其中，@zcnrex 提出了标准化 benchmark 与正确性验证的建议，@merrymercy 提出了 `BaseFusedOp` 和 trace 机制，@DarkSharpness 参与了 GEMM 统一接口及 backend/device capability 拆分的讨论。

最后，感谢 Anthropic 的 Claude Fable 5（https://www.anthropic.com/claude/fable），它通过 Claude Code 承担了这次工作中的大量设计、迁移和 benchmark 实现。
