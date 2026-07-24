![Triton Plugin Extensions](https://files.mdnice.com/user/59/1cd1702f-6c83-4960-a6be-27e1b743a374.png)

> 原文：Triton Plugin Extensions: Enabling TLX and Custom Compiler Passes Out of the Box
> 原文地址：PyTorch Blog（https://pytorch.org/blog/triton-plugin-extensions-enabling-tlx-and-custom-compiler-passes-out-of-the-box/）
> 作者：Corbin Robeck、Puyan Lotfi、Ian Barber、Shane Nay、Alexey Loginov、Oleksandr Stashuk、Wenyuan Chi
> 发布时间：2026 年 7 月 15 日
>
> 说明：本文是面向公众号阅读的中文翻译·译述稿，保留原文的技术主线、接口、性能数据和参考资料，但不是逐句直译。外部链接统一写成“文本（链接）”形式，正文图片已转存到 mdnice 图床。

# 【博客翻译·译述】Triton 插件扩展：开箱即用的 TLX 与自定义编译器 Pass

## TL;DR

PyTorch-Triton 3.7 引入了 Triton Plugin Extensions。它允许 Triton 在运行时加载编译器 pass、MLIR dialect 及其 op，甚至扩展 Python 层的 DSL，而不必维护 Triton fork，也不用为了安装插件重新编译 Triton。

Meta 的 TLX（Triton Language Extensions）是这套机制的第一个主要使用者。以前要使用 TLX，通常需要构建 Meta 的实验性 Triton fork；现在它可以作为独立的 `triton-utlx` Python 包安装，再通过 `TRITON_PLUGIN_PATHS` 加载到上游 Triton。博客给出的测试显示，这条动态加载路径与原 fork 生成相同的 PTX/AMDGCN，在 H100 和 MI350 上没有观察到额外性能损失。

这次改动解决的是 Triton 生态里一个很实际的问题：高性能 kernel 经常需要定制编译流水线，但过去一旦跨出上游 Triton 已有能力，团队就很容易走向长期维护 fork。

## 1. 为什么 Triton 需要插件系统

Triton 默认编译流水线覆盖了大量常见 kernel，不过生产级性能优化经常要继续往下走。例如：

- 在特定 lowering 阶段插入硬件相关的优化 pass。
- 使用上游尚未提供的指令或 op。
- 显式管理 shared memory，组织异步 load/compute pipeline。
- 替换默认 warp specialization 策略。
- 为某种模型或硬件调整完整的编译 stage。

过去常见的做法是 fork Triton，把 dialect、pass 和 lowering 直接编译进去。短期内很灵活，后续维护却很重：上游升级会带来 merge conflict、API 变化和行为差异；fork 固定在旧版本后，也拿不到新的硬件支持和 bug fix。

Plugin Extensions 换了一个边界。Triton core 提供稳定的加载与 hook 机制，扩展则作为独立 shared library 发布。研究者可以继续开发自定义 pass 和 op，使用者仍然运行上游 Triton。

## 2. 插件如何接入 Triton 编译流水线

插件本质上是运行时加载的 `.so` 文件。安装插件包后，把动态库路径写入 `TRITON_PLUGIN_PATHS`，Triton 启动编译时就会发现并加载它。

```python
import os
import sysconfig

site_packages = sysconfig.get_paths()["purelib"]
plugin_path = os.path.join(site_packages, "utlx_plugin", "libutlx.so")
os.environ["TRITON_PLUGIN_PATHS"] = plugin_path
```

插件系统的重点不只是“多注册几个 op”。Triton 在 backend 的 `compiler.py` 各阶段加入了 hook，扩展可以介入从高层 IR 到目标代码的整条 lowering 路径：

```text
Triton IR（TTIR）
        ↓
TritonGPU IR（TTGIR）
        ↓
LLVM IR
        ↓
PTX / AMDGCN
```

通过这些 hook，插件可以：

- 在任意 stage 的指定位置插入一个或多个 pass。
- 关闭某个默认 pass。
- 用自定义实现替换原有 pass。
- 覆盖整个 stage，必要时也能接管完整 pipeline。

NVIDIA 和 AMD backend 都支持这套机制。

### 2.1 三个扩展层级

原文把 API 能力分成三层。

第一层是自定义 transformation pass。它不需要引入新的 dialect，适合对已有 Triton IR 做局部转换。

第二层是自定义 MLIR dialect 与 conversion pass。插件可以把独立编译的 dialect 加载进 Triton，再把标准 Triton IR pattern 重写成自定义 op，交给专门的 lowering 处理。

第三层是顶层 DSL op。扩展可以在 Python 侧引入新的语法和语义，让 kernel 作者使用新的编程抽象，而不用修改 Triton 本体。

### 2.2 可以按 kernel 切换

插件不要求整个进程只使用一套编译流水线。kernel 代码可以设置 compiler hook，从那之后调用的 kernel 使用自定义 pipeline；取消 hook 后再回到默认路径。

插件数量和自定义 pipeline 数量没有硬限制。缓存管理由插件负责：配置真正变化时才应该触发重新编译。TLX 的 `utlx` 库已经把这部分封装起来，普通用户不需要自己实现 cache hash。

## 3. TLX 提供了哪些能力

TLX 面向需要显式控制内存、数据搬运与异步执行的 kernel。它把部分以往只能在 Triton fork 里使用的底层能力暴露成 DSL op。

| TLX 操作 | 用途 |
| --- | --- |
| `tlx.local_alloc(shape, dtype, num_buffers)` | 为软件流水线分配多级 local/shared memory buffer |
| `tlx.local_view(buffers, index)` | 取得某一级 buffer 的 view |
| `tlx.async_load(src, dst, mask)` | 发起 global memory 到 shared memory 的异步加载 |
| `tlx.async_load_commit_group(tokens)` | 提交一组异步 load |
| `tlx.async_load_wait_group(n)` | 等待异步 load group |
| `tlx.async_dot(a, b, acc)` | 发起异步矩阵乘加 |
| `tlx.async_dot_wait(n, acc)` | 等待异步 dot 完成 |
| `tlx.local_store(dst, src)` | 把数据写入 local/shared memory |
| `tlx.local_load(src)` | 从 local/shared memory 读回寄存器 |

这些 API 让 kernel 作者自己表达多缓冲软件流水线：先预取后续 tile，再计算当前 tile，通过 commit/wait 控制依赖。编译器仍负责 lowering，但调度意图不再完全依赖启发式推断。

## 4. 同一套 TLX，如何映射到 H100 和 MI350

TLX 的接口可以跨 NVIDIA 和 AMD 使用，底层实现则按硬件分别 lowering。

### 4.1 H100：TMA + WGMMA persistent GEMM

在 Hopper 上，TLX 的异步 load 可以映射到 TMA，矩阵乘加可以使用 WGMMA。persistent GEMM kernel 会提前分配多个 shared memory buffer，在 prologue 中预取前几级数据，然后在 K 循环里交替执行：

1. 等待当前计算所需的 load group。
2. 对已经就绪的 buffer 发起异步 dot。
3. 把下一块 A/B tile 预取进即将复用的 buffer。
4. 循环结束后等待所有 dot，并写回结果。

这样可以把 global memory 搬运与 Tensor Core 计算叠在一起。原文比较了 H100 上 FP16 GEMM 的 cuBLAS 和 Triton + TLX：

| M×N×K | cuBLAS | Triton + TLX | 差异 |
| --- | ---: | ---: | ---: |
| 128×13312×16384 | 247.8 TFLOPS | 257.0 TFLOPS | +3.7% |
| 16384×8192×8192 | 549.4 TFLOPS | 566.7 TFLOPS | +3.2% |
| 8192×16384×8192 | 564.8 TFLOPS | 575.9 TFLOPS | +2.0% |
| 8192×53248×8192 | 571.3 TFLOPS | 573.2 TFLOPS | +0.3% |
| 8192×28672×4096 | 560.4 TFLOPS | 559.8 TFLOPS | -0.1% |
| 8192×8192×8192 | 582.3 TFLOPS | 577.0 TFLOPS | -0.9% |

这组数据更适合解读为“插件加载没有拖慢原来的 TLX codegen”。不同 shape 上 TLX 和 cuBLAS 各有胜负，差距大多在几个百分点以内。

### 4.2 MI350：通过寄存器组织流水线

MI350 走的是另一条数据路径。kernel 先用普通 load 把下一块数据读进寄存器，再通过 `local_store` 写入 shared memory。计算当前 tile 时，用 `local_load` 把已经准备好的数据取回寄存器并执行 `tl.dot`。

buffer 管理方式仍然是多级流水线，只是数据搬运不依赖 Hopper 的 TMA/WGMMA。博客给出的 MI350 FP16 方阵 GEMM 数据如下：

| M=N=K | rocBLAS | Triton + TLX | 差异 |
| --- | ---: | ---: | ---: |
| 256 | 4.4 TFLOPS | 5.0 TFLOPS | +11.8% |
| 512 | 29.4 TFLOPS | 33.9 TFLOPS | +15.2% |
| 1024 | 161.2 TFLOPS | 180.8 TFLOPS | +12.1% |
| 2048 | 445.1 TFLOPS | 511.9 TFLOPS | +15.0% |

在这四个测试尺寸上，Triton + TLX 比 rocBLAS 高 11.8% 到 15.2%。

## 5. 从 microbenchmark 到 GPU MODE Trimul

团队还在 GPU MODE 的 Trimul multiplicative update 任务上验证了插件路径。这个任务包含五个 projection GEMM、一个 batched matmul、输出 linear，以及 layer norm、sigmoid gate 和 permutation，比较接近真实的复合 kernel pipeline。

PyTorch + `torch.compile` 基线耗时 19.2 ms，主要时间花在 GEMM。加载 TLX 插件后，参赛实现使用 warp-specialized persistent GEMM，并继续融合周围算子，最终达到 12.0 ms，相比 cuBLAS + `torch.compile` 基线加速 1.61 倍。

TLX kernel 仍以普通 Triton kernel 的形式出现在编译链里，因此优化可以继续延伸到 GEMM 周围的算子融合。团队后来还为 B200 加入了 CLC pipeline。

GPU MODE Trimul 任务说明（https://stormy-sailor-96a.notion.site/Trimul-multiplicative-update-2ac1f817bb5a8060a7a7f2ca8fbf297c）

## 6. 动态加载是否会改变 codegen

原文专门比较了插件版本与 Meta Triton fork 的输出。结论是：

- H100 persistent GEMM 生成相同的 PTX。
- MI350 pipelined GEMM 生成相同的 AMDGCN。
- 动态加载路径没有测到额外运行时开销。

原因也比较直接：TLX pass 和 op 的加载方式变了，但它们仍然进入同一套 MLIR lowering，插件没有在 kernel 执行时增加中间层。

Plugin Extensions 保留了原有的硬件优化，只把“扩展如何进入 Triton”的方式从 fork 换成运行时插件。这一点很实用。

## 7. 如何开始使用

原文给出的安装方式是先构建启用扩展支持的 Triton，再安装 PyPI 上的 TLX 插件：

```bash
git clone https://github.com/triton-lang/triton
cd triton
TRITON_EXT_ENABLED=ON pip install -e . --no-build-isolation
cd ..

pip install triton-utlx
```

> 译者注（2026 年 7 月 24 日）：当前 PyPI 上的 `triton-utlx 3.7.1` 包含原生 `libutlx.so`，与 Triton ABI 绑定。项目页要求使用上游 Triton 的 `v3.7.0` tag，并记录了该 tag 上一个影响插件 op 返回值的一行修复。实际安装时应以 PyPI 项目的 Compatibility 说明为准，不要随意混用其他 Triton commit。

设置插件路径后，TLX op 就可以在 kernel 中使用：

```python
import os
import sysconfig

site_packages = sysconfig.get_paths()["purelib"]
os.environ["TRITON_PLUGIN_PATHS"] = os.path.join(
    site_packages,
    "utlx_plugin",
    "libutlx.so",
)

import triton
import triton.language as tl
import utlx_plugin as tlx
```

相关资料：

- H100 Persistent GEMM Colab（https://colab.research.google.com/drive/1zANW7SP8dG9I7SXvWkH_9pYFRZYPAu_y?usp=sharing）
- H100 standalone demo（https://gist.github.com/CRobeck/daec7724bd3fd1ef2b38af7024032bc4）
- AMD MI350 demo（https://gist.github.com/CRobeck/6cfe9a32da9cdb6f446c8214a53d2293）
- Triton 插件文档（https://github.com/triton-lang/triton/blob/main/examples/plugins/README.md）
- Triton 扩展仓库（https://github.com/triton-lang/triton-ext）
- triton-utlx PyPI（https://pypi.org/project/triton-utlx/）
- TLX 源码与说明（https://github.com/facebookexperimental/triton）

## 8. 接下来还会扩展到哪里

原文列出的后续方向包括：

- 把 Intel、CPU 等 out-of-tree backend 做成动态插件，避免修改 Triton 构建系统。
- 以扩展形式提供 `triton-distributed` 的分布式计算原语。
- 把 Proton、ConSan 这类 profiling/instrumentation 能力做成按需加载的工具。
- 发布面向特定硬件或模型的 warp specialization、loop splitting 等优化 pass。
- 增加 2:4 structured sparsity、自定义 layout conversion 等专用 op。

这些方向目前处在不同开发阶段，不能都理解成 Triton 3.7 已经交付的功能。已经落地的是插件基础设施，以及 TLX 作为独立包接入上游 Triton 的路径。

## 小结

Triton Plugin Extensions 把 compiler research 和 Triton core 的版本维护拆开了。开发者可以继续编写硬件相关的 dialect、pass 和 DSL op，使用者则不必为了一个扩展长期绑定某个 fork。

TLX 给出了第一份完整样例：同一个 Python 编程模型可以在 H100 上调用 TMA/WGMMA，在 MI350 上走寄存器与 shared memory pipeline；动态插件版本与原 fork 保持相同 codegen。对于需要长期跟进上游 Triton、又必须保留定制编译能力的团队，这比维护一个越来越难升级的 fork 更省事。
