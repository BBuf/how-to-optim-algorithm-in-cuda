> 来源： https://discuss.pytorch.org/t/distributed-w-torchtitan-enabling-float8-all-gather-in-fsdp2/209323 。下面文章包含2个主题，第一个是FSDP2中开启Float8 All-Gather的Discussion的翻译，第二个是TorchAO中的Float8实现速览的翻译。这篇文档主要介绍了在FSDP2中启用float8 all-gather功能的实现和优化。通过在128个H100 GPU上预训练Llama3-70B模型的验证，相比bfloat16获得了1.50倍的性能提升，其中20%来自float8 all-gather，80%来自float8计算。文档详细描述了Float8训练的两个关键组件：通过torch._scaled_mm实现的float8计算和能节省50%带宽的float8通信。在优化策略方面，通过Float8计算+Bfloat16 All-Gather获得1.40倍加速，再通过带独立AMAX All-Reduce的Float8 All-Gather和组合AMAX AllReduce分别获得0.02倍和0.08倍的额外提升，同时还优化了NCCL和Float8计算之间的SM资源竞争。文档还提供了完整的代码示例，展示了如何将nn.Linear替换为Float8Linear、配置FSDP2、处理缩放因子和实现类型转换。对于实际应用，文档建议小矩阵（如1024x2048）保持使用bfloat16，而大矩阵（如4096x8192）则建议使用float8以获得加速。未来的工作方向包括在张量并行和流水线并行中支持Float8、实现延迟缩放以提升性能，以及探索行级缩放以提高精度。TorchAO中实现的Float8在TorchTitan中端到端可用，此外Meagtron-LM目前对FP8的训练支持也趋于成熟。

> 注意：FSDP2在节点内支持了TP，FSDP不支持TP是标准的Zero3。

# 在 FSDP2 中开启 Float8 All-Gather

## 总结

- 我们关注float8是因为它可以加速H100上的大型矩阵乘法运算,并通过减小消息大小来节省网络带宽。
- 我们在FSDP2中启用了float8 all-gather。读者可以在TorchTitan(https://github.com/pytorch/torchtitan/blob/main/docs/float8.md)中找到Llama3的训练配方,在TorchAO/float8(https://github.com/pytorch/ao/tree/main/torchao/float8)中找到float8数据类型的实现。
- 与bfloat16相比,我们观察到使用float8可以获得1.50倍的加速,同时保持相当的数值精度。其中20%的加速来自float8 all-gather,而剩余80%来自float8计算。该结果是通过使用128个H100s*预训练Llama3-70B得到的基准测试。

* Meta的H100是在Grand Teton(https://engineering.fb.com/2024/03/12/data-center-engineering/building-metas-genai-infrastructure/)上定制的。规格可能与公开版本不同。

![](https://files.mdnice.com/user/59/b7e1c36f-7891-4d5c-a401-9f70cc20fc84.png)


## Float8训练以及为什么Float8 All-Gather很重要

Float8数据类型在NVIDIA H100上得到原生支持。Float8训练的启用可以分为两个部分:float8计算和float8通信。

**Float8计算**: 通过`torch._scaled_mm`支持float8矩阵乘法(https://github.com/pytorch/pytorch/blob/618e2c9de42fbac3147e23e258122ebb76e2c041/aten/src/ATen/native/cuda/Blas.cpp#L935)。与bfloat16不同,float8需要原始张量和缩放因子来保持数值精度。用户需要在训练循环中维护缩放因子。有不同的缩放策略,包括张量级缩放、行/列级缩放、组级缩放和块级缩放。在本文中,我们专注于张量级缩放和动态缩放ref(https://arxiv.org/abs/2209.05433),其中缩放因子是从当前高精度张量中计算得出的。

**Float8通信(Float8 all-gather)**: 有了float8计算,在float8中执行all-gather几乎是"免费的",因为我们需要在all-gather之前或之后进行类型转换。在all-gather之前进行转换可以节省50%的带宽(相比bfloat16),代价是需要一次AMAX的all-reduce。Float8可以应用于模型权重、激活值和梯度。我们优先考虑float8权重,因为它们在训练循环中在数值上更稳定,并且更适合低精度数据类型。在本文中,我们重点关注Llama3模型。

读者可以在TorchTitan(https://github.com/pytorch/torchtitan/blob/main/docs/float8.md)中找到Llama3的训练配方,在TorchAO/float8(https://github.com/pytorch/ao/tree/main/torchao/float8)中找到float8数据类型的实现。

## 在Llama3中应用带Float8权重的FSDP2

Float8模型(代码(https://github.com/pytorch/torchtitan/blob/0f70507f1350679428ea64f90bc5a7db17b9c103/train.py#L239C5-L239C27)): PyTorch原生float8只需要对模型做最小的改动。以Llama3-8B模型为例,我们通过将每个nn.Linear替换为Float8Linear来将bfloat16模型转换为float8模型,这样我们就可以执行float8计算。

```python
TransformerBlock(
    (attention): Attention(
        (wq/wk/wv/wo): Float8Linear(in=4096, out=4096, bias=False) 
    )
    (feed_forward): FeedForward(
        (w1/w2/w3): Float8Linear(in=4096, out=14336, bias=False)
    )
    (attention_norm / ffn_norm): RMSNorm()
)
```

**应用FSDP2** (代码(https://github.com/pytorch/torchtitan/blob/0f70507f1350679428ea64f90bc5a7db17b9c103/torchtitan/parallelisms/parallelize_llama.py#L501)): 包装float8模型的用户体验与包装bfloat16模型相同。为了高效地跟踪缩放因子,我们在优化器步骤之后调用`precompute_float8_dynamic_scale_for_fsdp`,这样我们就可以在float8 all-gather之前获得复制的缩放因子用于float8类型转换。

```python
# wrapping each TransformerBlock, then root model
# the UX is the same across float8 model and bfloat16 model
for transformer_block in model.layers.values():
    fully_shard(transformer_block)
fully_shard(model)

# training loop
# ...
optimizer.step()
# all-reduce AMAX for Float8Linear.weight
precompute_float8_dynamic_scale_for_fsdp(model)
```

**用于float8张量子类的FSDP2扩展**: 我们在bfloat16模型和float8模型中保持相同的FSDP2用户体验,因为我们在FSDP2扩展中实现了float8类型转换。float8线性模块的权重是一个知道如何转换为float8的张量子类。我们可以自定义all-gather前后的类型转换逻辑,如下图所示。

- **fsdp_pre_all_gather (代码(https://github.com/pytorch-labs/float8_experimental/blob/0aca10aced1c4b3abdf00960d83316732cb08ed1/float8_experimental/fsdp_utils.py#L166))**: 根据最新的复制AMAX/缩放因子(需要all-reduce)将bfloat16权重转换为float8权重。注意这里的bfloat16权重是按1/NGPU分片的。由于我们通过all-reduce在所有rank上获得复制的AMAX和缩放因子,在all-gather之前将分片的bfloat16参数转换为float8等同于先all-gather bfloat16参数然后再转换为float8。
- **fsdp_post_all_gather (代码(https://github.com/pytorch-labs/float8_experimental/blob/0aca10aced1c4b3abdf00960d83316732cb08ed1/float8_experimental/fsdp_utils.py#L196))**: 从all-gather的float8数据和复制的缩放因子构建Float8Tensor,以便在前向和反向中进行float8计算。

![](https://files.mdnice.com/user/59/90db41e1-eccf-47c8-8c97-1308930dec96.png)

## 性能深入分析

我们讨论float8中的关键优化,以达到相比bfloat16 **1.50倍**的加速。

**Float8计算 + Bfloat16 All-Gather** (1.40倍加速, 代码(https://github.com/pytorch-labs/float8_experimental/blob/0aca10aced1c4b3abdf00960d83316732cb08ed1/float8_experimental/float8_linear.py#L439-L452)): 当用Float8Linear替换nn.Linear时,可以保持bfloat16权重不变。我们只需将Float8Linear当作普通的nn.Linear处理,并在FSDP2中执行bfloat16 all-gather(流22)。Float8Linear.forward负责bfloat16到float8的类型转换和float8矩阵乘法(流7)。这种方法实现了1.40倍的加速,是展示float8计算重要性的有力基准。然而,它浪费了50%的带宽来传输bfloat16参数,而这些参数最终会在前向过程中被转换为float8。

![](https://files.mdnice.com/user/59/0a3fc3e5-5c93-457c-b712-2ea1f18f493c.png)

**带独立AMAX All-Reduce的Float8 All-Gather** (在1.40倍基础上+0.02倍, 代码(https://github.com/pytorch/torchtitan/blob/0f70507f1350679428ea64f90bc5a7db17b9c103/torchtitan/float8_linear.py#L96)): 我们在all-gather之前执行float8类型转换以节省50%带宽(流22)。因此,Float8Linear.forward可以直接使用float8权重而无需类型转换(流7)。然而,float8类型转换需要一个全局AMAX(abs(max)的最大值),所以我们需要在N个rank之间all-reduce部分AMAX(一个标量)(流22和35)。每个float8参数需要1次all-reduce。这些小的all-reduce操作降低了整体性能。

![](https://files.mdnice.com/user/59/afc2c07e-994a-4c12-999d-dc1fa79c10c2.png)

**组合AMAX AllReduce** (在1.42倍基础上+0.08倍, 代码(https://github.com/pytorch/torchtitan/blob/0f70507f1350679428ea64f90bc5a7db17b9c103/torchtitan/float8_linear.py#L107)): 我们在优化器步骤之后对所有float8参数执行单次all-reduce。因此,我们避免了在FSDP钩子内部的小型all-reduce操作(流47)。我们通过一次性计算所有float8参数的AMAX实现了水平融合。

![](https://files.mdnice.com/user/59/26d9a1bd-44f6-4720-bd3e-19e183f8001a.png)

**NCCL和Float8计算之间的SM竞争**: 根据NCCL版本和GPU总SM数量,有时float8计算(流7)中会出现气泡。float8计算(sm90_xmm)和float8 all-gather(ncclDevKernel)都在争夺SM资源。理想情况是始终优先考虑第k层的float8计算而不是第k+1层的float8 all-gather。在这种情况下,如果NCCL使用更少的SM进行较慢的通信或float8计算使用更少的SM。我们发现在基准测试期间将NCCL_MAX_CTAS(https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-max-ctas)设置为16或8对解决竞争很有帮助。

![](https://files.mdnice.com/user/59/a6182568-52fd-485b-be3c-aa5262584ddb.png)

## 未来工作

我们正在积极探索以下方向(更多内容请参见PyTorch路线图(https://dev-discuss.pytorch.org/t/meta-pytorch-team-2024-h2-roadmaps/2226))

**张量并行和流水线并行中的Float8**: 对于张量并行(包括序列并行),我们沿序列维度分片模块输入,需要对输入进行float8 all-gather。对于流水线并行,我们正在验证float8是否存在任何性能差距。

**延迟缩放**: 与动态缩放相比,延迟缩放通过从前几次迭代中推导AMAX来获得性能提升。代价是可能损失数值精度。在实践中,float8权重在相邻迭代之间保持稳定。我们希望支持延迟缩放以达到完整性能。

**行级缩放**: 与张量级缩放相比,行级缩放通过为每一行设置细粒度的缩放因子来保持更好的数值精度。代价是反向传播的复杂性,因为矩阵从行级转置为列级。这需要在FSDP2中对float8 all-gather进行特殊处理。这仍然是一个高度探索性的方向。

# torchao.float8

这是一个使用原生PyTorch中float8加速训练的工作流程,
遵循 https://arxiv.org/pdf/2209.05433.pdf中提出的方法。
该代码库力求保持小巧、易于修改、可使用原生PyTorch工具进行调试,
并且可以与autograd、```torch.compile```和分布式等关键系统组合使用。
在启用``torch.compile``的情况下,初步结果显示
在128个GPU上运行LLaMa 3 70B预训练任务时,吞吐量最高可提升1.5倍。

:warning: <em>查看功能追踪器(https://github.com/pytorch/ao/issues/556)了解即将推出的功能。</em>

:warning: <em>代码库已经稳定,但尚未保证向后兼容性。</em>

:warning: <em>这些API仅用于训练且仅支持float8,我们计划在未来将其与torchao的其他部分统一(https://github.com/pytorch/ao/issues/894)。</em>

# 单GPU用户API

我们提供了三种张量级缩放策略:动态、延迟和静态。更多细节请参见https://arxiv.org/pdf/2209.05433.pdf, Section 4.3。这些策略可以分别配置为输入(`input`)、权重(`weight`)和梯度(`grad_output`)。

## 对`input`、`weight`和`grad_output`使用动态缩放的float8线性层

这是最准确的配方,因为每个张量都是动态缩放的。

```python
import torch
import torch.nn as nn
from torchao.float8 import convert_to_float8_training
from torchao.utils import TORCH_VERSION_AT_LEAST_2_5

if not TORCH_VERSION_AT_LEAST_2_5:
    raise AssertionError("torchao.float8 requires PyTorch version 2.5 or greater")

# create model and sample input
m = nn.Sequential(
    nn.Linear(2048, 4096),
    nn.Linear(4096, 128),
).bfloat16().cuda()
x = torch.randn(4096, 2048, device="cuda", dtype=torch.bfloat16)
optimizer = torch.optim.SGD(m.parameters(), lr=0.1)

# optional: filter modules from being eligible for float8 conversion
def module_filter_fn(mod: torch.nn.Module, fqn: str):
    # don't convert the last module
    if fqn == "1":
        return False
    # don't convert linear modules with weight dimensions not divisible by 16
    if isinstance(mod, torch.nn.Linear):
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
            return False
    return True

# convert specified `torch.nn.Linear` modules to `Float8Linear`
convert_to_float8_training(m, module_filter_fn=module_filter_fn)

# enable torch.compile for competitive performance
m = torch.compile(m)

# toy training loop
for _ in range(10):
    optimizer.zero_grad()
    y = m(x)
    y.sum().backward()
    optimizer.step()
```

## 使用延迟缩放的float8线性层

这是理论上最有效的配方,因为它最小化了内存读取。

```python
import torch
import torch.nn as nn
from torchao.float8 import (
    convert_to_float8_training,
    sync_float8_amax_and_scale_history,
    Float8LinearConfig,
    ScalingType,
    CastConfig,
)
from torchao.utils import TORCH_VERSION_AT_LEAST_2_5

if not TORCH_VERSION_AT_LEAST_2_5:
    raise AssertionError("torchao.float8 requires PyTorch version 2.5 or greater")

# 创建模型和样本输入
# 创建一个包含两个线性层的序列模型,并转换为bfloat16格式,放到GPU上
m = nn.Sequential(
    nn.Linear(2048, 4096),  # 第一个线性层,输入维度2048,输出维度4096
    nn.Linear(4096, 128),   # 第二个线性层,输入维度4096,输出维度128
).bfloat16().cuda()

# 创建一个随机输入张量,维度为4096x2048,使用bfloat16数据类型
x = torch.randn(4096, 2048, device="cuda", dtype=torch.bfloat16)

# 创建SGD优化器,学习率为0.1
optimizer = torch.optim.SGD(m.parameters(), lr=0.1)

# 配置延迟缩放
# 为输入、权重和梯度输出配置Float8LinearConfig,使用延迟缩放策略
config = Float8LinearConfig(
    cast_config_input=CastConfig(scaling_type=ScalingType.DELAYED),      # 输入使用延迟缩放
    cast_config_weight=CastConfig(scaling_type=ScalingType.DELAYED),     # 权重使用延迟缩放
    cast_config_grad_output=CastConfig(scaling_type=ScalingType.DELAYED), # 梯度输出使用延迟缩放
    # enable_amax_init=False,  # 仅在使用autocast + compile + FSDP + float8延迟时需要
    # enable_pre_and_post_forward=False  # 仅在使用autocast + compile + FSDP + float8延迟时需要
)

# 将所有torch.nn.Linear模块转换为Float8Linear,指定自定义缩放行为
convert_to_float8_training(m, config=config)

# 启用torch.compile以获得有竞争力的性能
m = torch.compile(m)

# 玩具训练循环
for _ in range(10):
    optimizer.zero_grad()  # 清零梯度
    y = m(x)              # 前向传播
    y.sum().backward()    # 反向传播

    # float8延迟缩放特有步骤:同步scales/amaxes
    # 将来这可能会移到上下文管理器中
    sync_float8_amax_and_scale_history(m)

    optimizer.step()      # 更新参数
```

# 多GPU用户API

我们与基于`DTensor`的分布式API(https://pytorch.org/docs/stable/distributed.tensor.parallel.html),
如FSDP、TP和SP组合使用。请参见torchtitan(https://github.com/pytorch/torchtitan)仓库中的示例,
了解如何在分布式环境中使用`torchao.float8`。

:warning: <em>当使用FSDP时,建议启用`config.force_recompute_fp8_weight_in_bwd`,以防止在反向传播中保存未分片的fp8权重。
如果你使用的是自定义激活检查点,可以忽略此配置,并在自定义AC代码中处理fp8权重的重新计算。</em>

# 性能

关于float8训练的一个常见问题是"相比bfloat16,float8线性层在什么情况下更快?"。给定线性层前向传播的M、K、N参数,你可以参考下表中基于NVIDIA H100的微基准测试加速比估计:

![](https://files.mdnice.com/user/59/31bd8826-a8f3-4e31-8135-84132f111ab8.png)

示例1 (小形状):
* 前向输入张量大小1024x2048,线性权重大小2048x1024; M, K, N = 1024, 2048, 1024
* 微基准测试加速比为0.80
* 建议: 保持这个线性层使用bfloat16,因为形状太小无法从float8计算中获益

示例2 (大形状):
* 前向输入张量大小4096x8192,线性权重大小8192x16384; M, K, N = 4096, 8192, 16384
* 微基准测试加速比为1.39
* 建议: 启用float8以获得加速

要重现上述表中的原始数据,可以运行以下脚本

```lang=shell
python benchmarks/float8/float8_roofline.py your_output_filename.csv --gemm_time_strategy benchmarks --shape_gen_name sweep
```

## 推导

在bf16线性层中,假设所有时间都花费在gemms中。在float8线性层中,考虑max_abs和类型转换开销。我们想知道什么时候

```
bf16_gemm_time > fp8_gemm_time + fp8_overhead_time
```

或者等价于

```
bf16_gemm_time - fp8_gemm_time > fp8_overhead_time
```

我们可以从上述公式中得出三个观察结果:
* LHS > 0对于大形状,随着M、K、N的增加,gemm加速接近2x
* LHS < 0对于小形状,在NVIDIA H100 + cuBLAS上
* RHS > 0对于所有形状,受限于内存带宽、框架开销和编译器限制
* RHS > 0对于所有形状,受限于内存带宽、框架开销和编译器限制

对于小形状,结合(2)和(3)导致加速比<1。对于中等形状, (1)和(3)的幅度相似,加速比取决于M、K、N和框架及编译器行为。对于大形状, (1)导致加速比>1。

## 缩放类型与加速比

延迟缩放比动态缩放更快,因为减少了读/写流量要求。今天,torch.compile有几个限制(见https://github.com/pytorch/ao/issues/556的性能部分),阻止我们达到延迟缩放的最佳性能,所以延迟缩放的观察性能接近动态缩放。随着torch.compile限制的解决,我们期望延迟缩放最终比动态缩放更高效。

## torch.compile行为与加速比

torch.compile在生成float8缩放和类型转换内核时有一些限制(见https://github.com/pytorch/ao/issues/556的性能部分)。随着限制的解决,我们期望达到改进的性能。

# 测试

```bash
# run single-GPU unit tests
pytest test/float8/test_base.py

# run single-GPU compile tests
pytest test/float8/test_compile.py

# run single-GPU numerics integration tests
pytest test/float8/test_numerics_integration.py

# run a two-GPU integration test on FSDP
./test/float8/test_fsdp.sh

# run integration tests on the DTensor TP/SP integration
./test/float8/test_dtensor.sh

# run integration tests on the FSDP2 integration
python test/float8/test_fsdp2/test_fsdp2.py

# run all of these tests
./test/float8/test_everything.sh
```

# 基准测试

```bash
# benchmark the torch._scaled_mm function on LLaMa 2 70B shapes
./benchmarks/float8/bench_matmul.py

# benchmark fw/bw of `Linear` and `Float8Linear` on LLaMa 2 70B shapes
# make sure to turn on torch.compile to get the best performance
./benchmarks/float8/bench_linear_float8.py -o ../tmp/test.txt --compile
```

TorchAO FP8实现的相关开源代码在 https://github.com/pytorch/ao/tree/main/torchao/float8 ，代码整体不是很长，感兴趣的读者可以阅读下。

