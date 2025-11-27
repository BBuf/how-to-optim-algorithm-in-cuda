> 原始的markdown见：https://github.com/BBuf/how-to-optim-algorithm-in-cuda/blob/master/large-language-model-note/Cache-Dit%20%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0.md

## 前言

最近在研究DiT(Diffusion Transformer)推理优化的时候接触到了Cache-Dit这个项目，也正在参与建设这个项目，发现这是一个比较有意思但是知名度看起来还不大的工作，这里写一篇学习笔记也算是宣传下。Cache-Dit是唯品会开源的一个PyTorch原生的DiT推理加速引擎（https://github.com/vipshop/cache-dit）,它通过混合缓存加速和并行化技术来加速DiT模型的推理。这个项目最吸引我的地方在于它不仅实现了各种缓存算法,还支持Context Parallelism和Tensor Parallelism,并且做到了和torch.compile、量化方法的集成，最关键的是支持新模型比较快。

这篇笔记会详细记录Cache-Dit的核心技术要点,包括它的缓存策略、并行化方案、以及各种优化选项的使用。我会以FLUX.1-dev和Wan 2.2这两个典型模型为例,讲解Cache-Dit在实际应用中的优化效果。此外,我还会重点介绍Cache-Dit最近实现的Ulysses Anything Attention特性,以及Async Ulysses QKV Projection这个通信和计算overlap的优化。

仓库的作者在知乎发了不少技术细节特别是Cache相关的和模型支持的帖子，大家感兴趣可以去看：https://www.zhihu.com/people/qyjdef/posts

![](https://files.mdnice.com/user/59/38114839-dc24-4a03-b000-a7047caf2819.png)

## Cache-Dit的核心设计理念

Cache-Dit的设计哲学可以用几个关键词来概括:PyTorch原生、灵活、易用。它不是一个独立的推理框架,而是一个可以无缝集成到现有Diffusers Pipeline中的加速库。这种设计让它可以很容易地和其他优化手段组合使用,比如torch.compile、量化、CPU offload等。

Cache-Dit支持几乎所有基于Transformer的DiT模型,包括FLUX.1、Qwen-Image、Wan系列、HunyuanVideo、CogVideoX等30多个模型系列,近100个pipeline。它通过Forward Pattern Matching机制来自动识别不同模型的Transformer Block结构,然后应用相应的缓存策略。

### 统一的Cache API设计

Cache-Dit最大的特点是提供了统一的API接口。在大多数情况下,你只需要一行代码就可以启用缓存加速:

```python
import cache_dit
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image")
cache_dit.enable_cache(pipe)  # 一行代码启用缓存
output = pipe(...)
stats = cache_dit.summary(pipe)  # 查看缓存统计信息
cache_dit.disable_cache(pipe)  # 禁用缓存
```

这种设计让用户可以非常方便地在现有代码中集成Cache-Dit,而不需要修改太多代码。例如：https://zhuanlan.zhihu.com/p/1950849526400263083

### Forward Pattern Matching

Cache-Dit通过Forward Pattern Matching机制来识别不同模型的Transformer Block结构。目前支持6种Forward Pattern,每种pattern定义了block forward的输入输出签名:

Pattern_0: 双输入双输出,hidden_states先返回
- 输入: `(hidden_states, encoder_hidden_states)`
- 输出: `(hidden_states, encoder_hidden_states)`
- 典型模型: 早期的一些DiT模型

Pattern_1: 双输入双输出,encoder_hidden_states先返回
- 输入: `(hidden_states, encoder_hidden_states)`
- 输出: `(encoder_hidden_states, hidden_states)`
- 典型模型: FLUX.1的transformer_blocks(diffusers<0.35.0版本)

Pattern_2: 双输入单输出,只返回hidden_states
- 输入: `(hidden_states, encoder_hidden_states)`
- 输出: `(hidden_states,)`
- 典型模型: Wan 2.1/2.2, CogVideoX, HunyuanVideo, Qwen-Image, LTXVideo, CogView3Plus/4, SkyReelsV2, Kandinsky5

Pattern_3: 单输入单输出,只有hidden_states
- 输入: `(hidden_states,)`
- 输出: `(hidden_states,)`
- 典型模型: FLUX.1的single_transformer_blocks, Mochi, PixArt, SD3, DiT-XL, Chroma, HiDream, Allegro, Sana, Lumina, OmniGen, AuraFlow

...

通过识别这些pattern,Cache-Dit可以自动决定哪些tensor需要缓存,哪些tensor需要重新计算。Pattern的识别是通过检查block的forward方法签名来实现的,包括参数名称和返回值数量。

### Block Adapter机制

对于一些特殊的模型或者自定义的Transformer,Cache-Dit提供了Block Adapter机制。通过Block Adapter,用户可以手动指定transformer、blocks、forward_pattern等信息,让Cache-Dit能够正确地应用缓存策略。

#### BlockAdapter的核心参数

BlockAdapter是一个dataclass,包含以下核心参数:

Transformer配置相关:
- `pipe`: DiffusionPipeline实例,可以为None(用于Transformer-only场景)
- `transformer`: 单个transformer或transformer列表,支持多transformer模型如Wan 2.2
- `blocks`: torch.nn.ModuleList或其列表,指定要应用缓存的transformer blocks
- `blocks_name`: blocks在transformer中的属性名,如"transformer_blocks", "blocks"等
- `unique_blocks_name`: 内部使用的唯一标识符,自动生成
- `dummy_blocks_names`: 需要跳过的blocks名称列表

Forward Pattern配置:
- `forward_pattern`: ForwardPattern枚举或其列表,指定blocks的forward pattern
- `check_forward_pattern`: 是否检查forward pattern匹配,默认对diffusers模型启用
- `check_num_outputs`: 是否检查输出数量,用于更严格的pattern验证

参数修改器:
- `params_modifiers`: ParamsModifier或其列表,用于为不同blocks设置不同的缓存参数

Pipeline级别配置:
- `patch_functor`: PatchFunctor实例,用于修改transformer的forward方法
- `has_separate_cfg`: 是否有独立的CFG forward,如Wan 2.1, Qwen-Image等

自动Block Adapter配置:
- `auto`: 是否启用自动block adapter,会自动查找合适的blocks
- `allow_prefixes`: 允许的blocks属性名前缀,默认包括"transformer", "blocks", "layers"等
- `check_prefixes`: 是否检查前缀
- `allow_suffixes`: 允许的block类名后缀,如"TransformerBlock"
- `check_suffixes`: 是否检查后缀
- `blocks_policy`: 当有多个候选blocks时的选择策略,"max"选择最多blocks,"min"选择最少

其他配置:
- `skip_post_init`: 是否跳过post_init,用于特殊场景

比如对于FLUX.1这种有多个transformer blocks的模型:

```python
from cache_dit import ForwardPattern, BlockAdapter

cache_dit.enable_cache(
    BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=[
            pipe.transformer.transformer_blocks,
            pipe.transformer.single_transformer_blocks,
        ],
        forward_pattern=[
            ForwardPattern.Pattern_1,
            ForwardPattern.Pattern_3,
        ],
    ),
)
```

对于Wan 2.2这种有多个Transformer的MoE模型:

```python
cache_dit.enable_cache(
    BlockAdapter(
        pipe=pipe,
        transformer=[
            pipe.transformer,
            pipe.transformer_2,
        ],
        blocks=[
            pipe.transformer.blocks,
            pipe.transformer_2.blocks,
        ],
        forward_pattern=[
            ForwardPattern.Pattern_2,
            ForwardPattern.Pattern_2,
        ],
        params_modifiers=[
            ParamsModifier(
                cache_config=DBCacheConfig().reset(
                    max_warmup_steps=4,
                    max_cached_steps=8,
                ),
            ),
            ParamsModifier(
                cache_config=DBCacheConfig().reset(
                    max_warmup_steps=2,
                    max_cached_steps=20,
                ),
            ),
        ],
        has_separate_cfg=True,
    ),
)
```

这种灵活的设计让Cache-Dit可以支持各种复杂的模型结构。

#### 自动Block Adapter

当你不确定模型的blocks结构时,可以使用自动Block Adapter:

```python
from cache_dit import ForwardPattern, BlockAdapter

cache_dit.enable_cache(
    BlockAdapter(
        pipe=pipe,
        auto=True,  # 启用自动模式
        forward_pattern=ForwardPattern.Pattern_2,
        allow_prefixes=["transformer", "blocks"],  # 可选,自定义搜索前缀
        blocks_policy="max",  # 选择blocks数量最多的
    ),
)
```

自动模式会遍历transformer的所有属性,找到符合条件的ModuleList,并验证其forward pattern是否匹配。

#### LoRA模型支持

Cache-Dit完全支持使用LoRA的模型。在使用LoRA时,有两种方式:

方式1: 直接使用LoRA(推荐用于推理)
```python
from diffusers import FluxPipeline
import cache_dit

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
).to("cuda")

# 加载LoRA
pipe.load_lora_weights("path/to/lora")

# 启用缓存
cache_dit.enable_cache(pipe)

# 正常推理
image = pipe(prompt, ...).images[0]
```

方式2: Fuse LoRA(推荐用于benchmark)
```python
# 加载LoRA
pipe.load_lora_weights("path/to/lora")

# Fuse LoRA到base model
pipe.fuse_lora()
pipe.unload_lora_weights()

# 启用缓存
cache_dit.enable_cache(pipe)
```

Fuse LoRA会将LoRA权重合并到base model中,这样可以避免推理时的额外计算开销。对于需要多次推理的场景,fuse_lora是更好的选择。

Cache-Dit的Patch Functor会自动处理LoRA相关的scale_lora_layers和unscale_lora_layers调用,确保缓存机制与LoRA兼容。支持LoRA的模型包括FLUX.1, Qwen-Image, HiDream, Chroma, Wan VACE, HunyuanVideo等。

## DBCache: Dual Block Cache

DBCache是Cache-Dit的核心缓存算法,全称是Dual Block Cache。它的核心思想是通过两组compute blocks(Fn和Bn)来平衡性能和精度。

### DBCache的设计原理

DBCache的设计基于一个观察:在Diffusion模型的推理过程中,相邻时间步的hidden states变化是渐进的。当residual diff(L1距离)小于某个阈值时,我们可以直接使用缓存的residual,而不需要重新计算整个block。

DBCache引入了两个关键参数:

- **Fn (First n blocks)**: 指定前n个Transformer blocks始终进行完整计算。这些blocks用于拟合当前时间步的信息,计算更稳定的L1 diff,并为后续blocks提供更准确的信息。

- **Bn (Back n blocks)**: 指定后n个Transformer blocks也进行完整计算。这些blocks作为一个auto-scaler,用于增强使用residual cache的近似hidden states的预测精度。

中间的blocks则根据residual_diff_threshold来决定是否使用缓存。

### DBCache的配置选项

```python
from cache_dit import DBCacheConfig

cache_dit.enable_cache(
    pipe,
    cache_config=DBCacheConfig(
        max_warmup_steps=8,  # 前8步不使用缓存,用于warm up
        max_cached_steps=-1,  # -1表示无限制
        Fn_compute_blocks=8,  # F8,前8个blocks始终计算
        Bn_compute_blocks=0,  # B0,后0个blocks,即不使用Bn
        residual_diff_threshold=0.12,  # residual diff阈值
    ),
)
```

不同的FnBn配置会带来不同的性能和精度trade-off:

- **F1B0**: 最激进的缓存策略,只有第一个block始终计算,速度最快但精度可能下降
- **F8B0**: 平衡的配置,前8个blocks计算,既保证精度又有不错的加速
- **F8B8**: 保守的配置,前8个和后8个blocks都计算,精度最高但加速效果相对较弱
- **F16B16**: 更保守,适合对精度要求极高的场景

在FLUX.1-dev上的benchmark显示,F8B0配置可以达到1.8x的加速,同时保持很高的图像质量(Clip Score: 32.99, ImageReward: 1.04)。

### residual_diff_threshold的选择

residual_diff_threshold是DBCache中最关键的超参数。它决定了什么时候使用缓存,什么时候重新计算。

一般来说:
- 阈值越小,使用缓存的频率越低,精度越高但速度越慢
- 阈值越大,使用缓存的频率越高,速度越快但精度可能下降

对于不同的模型和不同的blocks,最优的阈值可能不同。Cache-Dit提供了`cache_dit.summary(pipe, details=True)`来查看每个cached step的residual diff统计信息,帮助用户选择合适的阈值。

对于FLUX.1的single_transformer_blocks,由于前面transformer_blocks的误差累积,通常需要更高的阈值(比如0.16),而transformer_blocks可以使用较低的阈值(比如0.08)。

## DBPrune: Dynamic Block Prune

除了DBCache,Cache-Dit还实现了DBPrune(Dynamic Block Prune)算法。DBPrune和DBCache的思路类似,但它不是缓存residual,而是直接跳过(prune)某些blocks的计算。

```python
from cache_dit import DBPruneConfig

cache_dit.enable_cache(
    pipe,
    cache_config=DBPruneConfig(
        max_warmup_steps=8,
        Fn_compute_blocks=8,
        Bn_compute_blocks=8,
        residual_diff_threshold=0.12,
        enable_dynamic_prune_threshold=True,
        non_prune_block_ids=list(range(16, 24)),  # 指定不prune的blocks
    ),
)
```

DBPrune可以实现更激进的加速,但需要更仔细地调整参数。在FLUX.1上,DBPrune可以prune掉35%-60%的blocks,实现1.5x-2.3x的加速。

## TaylorSeer Calibrator

TaylorSeer是一个基于泰勒展开的特征预测算法。Cache-Dit将TaylorSeer作为一个Calibrator集成进来,用于提升DBCache在大量cached steps情况下的精度。

TaylorSeer的核心思想是使用泰勒级数展开来预测未来时间步的特征:

$$\mathcal{F}_{\text{pred}, m}(x_{t-k}^l) = \mathcal{F}(x_t^l) + \sum_{i=1}^m \frac{\Delta^i \mathcal{F}(x_t^l)}{i! \cdot N^i}(-k)^i$$

在Cache-Dit中,TaylorSeer可以用于预测hidden states或residual cache。

```python
from cache_dit import DBCacheConfig, TaylorSeerCalibratorConfig

cache_dit.enable_cache(
    pipe,
    cache_config=DBCacheConfig(
        Fn_compute_blocks=8,
        Bn_compute_blocks=0,  # 使用TaylorSeer时,Bn可以设为0
        residual_diff_threshold=0.12,
    ),
    calibrator_config=TaylorSeerCalibratorConfig(
        taylorseer_order=1,  # 一阶泰勒展开
    ),
)
```

实验表明,DBCache F1B0 + TaylorSeer的组合可以在保持精度的同时实现更好的加速效果。

## SCM: Steps Computation Masking

SCM(Steps Computation Masking)是受LeMiCa和EasyCache启发的一个优化策略。它的核心观察是:早期的caching会引入放大的下游误差,而后期的caching影响较小。因此,应该采用非均匀的cached steps分布。

```python
cache_dit.enable_cache(
    pipe,
    cache_config=DBCacheConfig(
        Fn_compute_blocks=8,
        Bn_compute_blocks=0,
        max_warmup_steps=6,
        residual_diff_threshold=0.12,
        steps_computation_mask=cache_dit.steps_mask(
            compute_bins=[6, 1, 1, 1, 1],  # 10个compute steps
            cache_bins=[1, 2, 5, 5, 5],     # 18个cache steps
        ),
        steps_computation_policy="dynamic",  # 使用dynamic cache
    ),
    calibrator_config=TaylorSeerCalibratorConfig(
        taylorseer_order=1,
    ),
)
```

`steps_computation_mask`是一个长度为num_inference_steps的列表,1表示必须计算,0表示使用缓存。通过`cache_dit.steps_mask()`可以方便地生成这个mask。

SCM配合dynamic cache可以实现更好的效果。在FLUX.1上,SCM Ultra Fast + TaylorSeer + compile的组合可以实现7.1x的加速,同时保持很好的图像质量。

## 量化支持

Cache-Dit集成了torchao作为量化后端,支持多种量化方案。

### 支持的量化类型

Weight-Only量化(不依赖torch.compile):
- `float8_weight_only` (fp8_w8a16_wo): FP8权重,FP16激活
- `int8_weight_only` (int8_w8a16_wo): INT8权重,FP16激活
- `int4_weight_only` (int4_w4a16_wo): INT4权重,FP16激活

Dynamic Quantization(依赖torch.compile):
- `float8` (fp8_w8a8_dq): FP8权重和动态FP8激活
- `int8` (int8_w8a8_dq): INT8权重和动态INT8激活
- `int4` (int4_w4a8_dq): INT4权重和动态INT8激活
- `int4_w4a4` (int4_w4a4_dq): INT4权重和动态INT4激活

### 量化和torch.compile的关系

Weight-Only量化是在模型加载时进行的静态量化,将权重转换为低精度格式,但激活保持原精度。这类量化不需要torch.compile,可以直接使用:

```python
import cache_dit

# Weight-only量化,不需要compile
cache_dit.quantize(
    pipe.transformer,
    quant_type="float8_weight_only",
)

# 可以直接推理
image = pipe(prompt, ...).images[0]
```

Dynamic Quantization需要在forward过程中动态量化激活值,这需要torch.compile来生成高效的融合kernel。如果不使用compile,动态量化会非常慢:

```python
import cache_dit

# Dynamic quantization,必须配合compile使用
cache_dit.quantize(
    pipe.transformer,
    quant_type="float8",  # 动态量化
)

# 必须使用torch.compile
cache_dit.set_compile_configs()
pipe.transformer = torch.compile(pipe.transformer)

# 现在可以高效推理
image = pipe(prompt, ...).images[0]
```

### 量化配置选项

```python
cache_dit.quantize(
    pipe.transformer,
    quant_type="float8_weight_only",
    exclude_layers=["embedder", "embed"],  # 跳过embedding层
    per_row=True,  # 对于fp8_w8a8_dq,使用per-row量化
)
```

- `exclude_layers`: 指定不量化的层,默认跳过embedding层以保持精度
- `per_row`: 对于FP8 dynamic quantization,是否使用per-row量化(需要bfloat16)

### Weight-Only量化的计算流程

Weight-Only量化使用torchao的`Float8WeightOnlyConfig`等配置,在模型加载时将Linear层的权重转换为低精度格式。以`float8_weight_only`为例:

```python
# torchao内部实现伪代码
class Float8WeightOnlyLinear(nn.Module):
    def __init__(self, in_features, out_features, weight_fp8, scale):
        self.weight_fp8 = weight_fp8  # FP8格式的权重
        self.scale = scale            # 量化scale
    
    def forward(self, x):
        # x是FP16/BF16激活
        # 1. 将FP8权重反量化回FP16/BF16
        weight_fp16 = self.weight_fp8.to(x.dtype) * self.scale
        # 2. 执行标准的FP16/BF16矩阵乘法
        output = F.linear(x, weight_fp16)
        return output
```

实际实现中,torchao会使用CUDA kernel来融合反量化和矩阵乘法操作,避免显式的反量化开销。关键点:

1. 权重在内存中以FP8格式存储,节省显存
2. 计算时动态反量化到FP16/BF16,然后执行标准GEMM
3. 反量化和GEMM可以融合成单个kernel,减少内存访问
4. 不需要torch.compile,因为torchao已经提供了优化的CUDA kernel

### Dynamic Quantization的计算流程

Dynamic Quantization需要在forward时动态量化激活值。以`float8`(fp8_w8a8_dq)为例:

```python
# torchao内部实现伪代码
class Float8DynamicLinear(nn.Module):
    def __init__(self, in_features, out_features, weight_fp8, weight_scale):
        self.weight_fp8 = weight_fp8
        self.weight_scale = weight_scale
    
    def forward(self, x):
        # x是FP16/BF16激活
        # 1. 动态计算激活的量化scale
        x_scale = x.abs().max() / 448.0  # FP8 E4M3的最大值
        # 2. 量化激活到FP8
        x_fp8 = (x / x_scale).to(torch.float8_e4m3fn)
        # 3. FP8矩阵乘法
        output_fp8 = torch.mm(x_fp8, self.weight_fp8.T)
        # 4. 反量化输出
        output = output_fp8.to(x.dtype) * x_scale * self.weight_scale
        return output
```

这个过程涉及多个小kernel(计算scale、量化、GEMM、反量化),如果不使用torch.compile,这些kernel会串行执行,开销很大。torch.compile可以:

1. 将多个小kernel融合成一个大kernel
2. 消除中间tensor的内存分配
3. 优化内存访问模式

所以Dynamic Quantization必须配合torch.compile使用才能获得性能提升。

### 量化的选择建议

- 低显存场景: 使用`float8_weight_only`或`int8_weight_only`,精度损失小,不需要compile
- 追求极致性能: 使用`float8` + torch.compile,但需要更多调试
- 4-bit量化: 可以使用`int4_weight_only`,或者使用nunchaku的W4A4方案

Cache-Dit的量化可以和缓存、并行化等优化手段组合使用。

## torch.compile支持

Cache-Dit完全兼容torch.compile。使用torch.compile可以进一步提升性能:

```python
import cache_dit

# 设置推荐的compile配置
cache_dit.set_compile_configs()

# compile transformer
pipe.transformer = torch.compile(pipe.transformer)
```

### set_compile_configs做了什么

`cache_dit.set_compile_configs()`会设置以下配置:

1. **增加recompile限制**: 设置`torch._dynamo.config.cache_size_limit`为更大的值,避免DBCache的动态行为导致频繁recompile

2. **启用compute-communication overlap**: 对于使用Context Parallelism的场景,启用通信和计算的overlap优化

3. **其他inductor优化**: 设置一些推荐的inductor flags

### compile和缓存的兼容性

Cache-Dit的DBCache会根据residual diff动态决定是否使用缓存,这会导致不同的执行路径。torch.compile会为每个路径生成不同的kernel,所以需要增大cache_size_limit。

如果你不想让compile和缓存一起使用,可以:

```python
# 只使用缓存,不compile
cache_dit.enable_cache(pipe)

# 或者只compile,不使用缓存
pipe.transformer = torch.compile(pipe.transformer)
```

### compile和量化的配合

如前所述,dynamic quantization必须配合torch.compile使用:

```python
# 先量化
cache_dit.quantize(pipe.transformer, quant_type="float8")

# 再compile
cache_dit.set_compile_configs()
pipe.transformer = torch.compile(pipe.transformer)
```

而weight-only量化不需要compile,但加上compile可以进一步优化:

```python
# weight-only量化
cache_dit.quantize(pipe.transformer, quant_type="float8_weight_only")

# 可选:加上compile进一步优化
pipe.transformer = torch.compile(pipe.transformer)
```

## Attention Kernel选择

Cache-Dit支持多种attention kernel实现,可以通过`attention_backend`参数选择:

### 支持的Attention Backend

1. `_native_cudnn`: PyTorch原生的cuDNN attention实现
   - 默认选项,最稳定
   - 支持Context Parallelism
   - 性能中等

2. `native`: PyTorch的scaled_dot_product_attention
   - 使用PyTorch的SDPA实现
   - 会自动选择FlashAttention或Memory-Efficient Attention
   - 需要diffusers>=0.36.0

3. `flash`: FlashAttention-2
   - 需要安装flash-attn库
   - 性能最好,显存效率高
   - 支持Context Parallelism

4. `sage`: SageAttention
   - 需要安装sageattention库: https://github.com/thu-ml/SageAttention
   - 针对长序列优化
   - 支持Context Parallelism

### 使用方式

```python
from cache_dit import ParallelismConfig

cache_dit.enable_cache(
    pipe,
    parallelism_config=ParallelismConfig(
        ulysses_size=2,
        parallel_kwargs={
            "attention_backend": "flash",  # 选择FlashAttention
        },
    ),
)
```

### Attention Backend的选择建议

- 默认使用`_native_cudnn`,最稳定
- 如果安装了flash-attn,推荐使用`flash`,性能最好
- 长序列场景可以尝试`sage`
- `native`在某些情况下可能有兼容性问题,需要diffusers的修复

## Context Parallelism支持

Context Parallelism(CP)是Cache-Dit支持的一个重要并行化方案。CP通过在序列维度上切分输入,让多个GPU并行处理不同的序列片段,从而实现加速。

### Ulysses Attention

Cache-Dit基于Diffusers的Context Parallelism实现,使用Ulysses Attention作为核心算法。Ulysses Attention的核心思想是:

1. 将hidden_states在序列维度上切分到不同GPU
2. 在计算attention之前,通过all-to-all通信重新分布Q、K、V,使得每个GPU拥有完整的序列但只有部分head
3. 计算attention
4. 再次通过all-to-all通信,将结果重新分布回原来的序列切分方式

这种方法的优点是通信量相对较小,且可以和其他优化手段(如FlashAttention)结合使用。

### 使用Context Parallelism

```python
from cache_dit import ParallelismConfig

cache_dit.enable_cache(
    pipe,
    cache_config=DBCacheConfig(...),
    parallelism_config=ParallelismConfig(
        ulysses_size=2,  # 使用2个GPU做CP
    ),
)
```

然后使用torchrun启动:

```bash
torchrun --nproc_per_node=2 run_flux_cp.py --cache --parallel ulysses
```

在FLUX.1上,使用CP2(2个GPU)可以实现接近2x的加速,从23.25s降到13.87s。

### Ulysses Anything Attention (UAA)

标准的Ulysses Attention有一个限制:序列长度必须能被GPU数量整除。这在实际应用中是一个很大的限制,因为用户输入的prompt长度是不固定的,很难保证序列长度总是能被GPU数量整除。

为了解决这个问题,Cache-Dit实现了Ulysses Anything Attention(UAA)。UAA是一个支持任意序列长度的Ulysses Attention,具有以下特点:

- **零padding**: 不需要对序列进行padding,避免了计算浪费
- **近零通信开销**: 只增加了一个all-gather操作来收集各个rank的序列长度

UAA的实现核心在于使用`tensor_split`而不是`chunk`来切分序列,以及使用all-gather来处理不均匀的序列长度。

```python
cache_dit.enable_cache(
    pipe,
    cache_config=DBCacheConfig(...),
    parallelism_config=ParallelismConfig(
        ulysses_size=2,
        parallel_kwargs={
            "experimental_ulysses_anything": True,  # 启用UAA
        },
    ),
)
```

为了减少通信延迟,建议在初始化进程组时同时启用gloo和nccl backend:

```python
dist.init_process_group(backend="cpu:gloo,cuda:nccl")
```

这样可以避免H2D和D2H传输导致的多次CUDA sync。

实验表明,UAA在支持任意序列长度的同时,性能和标准Ulysses Attention几乎相同。在FLUX.1上,CP2 + UAA的性能是13.88s,和标准Ulysses的13.87s几乎一样。

### UAA的实现细节

UAA的核心实现在`_templated_ulysses_anything.py`中。主要包括几个关键函数:

1. **_gather_size_by_comm**: 使用all-gather收集各个rank的序列长度
2. **_all_to_all_single_any_qkv**: 处理Q、K、V的all-to-all通信,支持不均匀的序列长度
3. **_all_to_all_single_any_o**: 处理输出的all-to-all通信
4. **shard_anything/unshard_anything**: 修改EquipartitionSharder的shard和unshard方法,使用tensor_split来支持任意序列长度

这些实现都使用了`@torch.compiler.allow_in_graph`装饰器,确保可以被torch.compile正确处理。

### Async Ulysses QKV Projection for FLUX

Cache-Dit在PR #480中实现了Async Ulysses QKV Projection特性,专门针对FLUX.1模型优化。这个特性的核心思想是将QKV projection的计算和all-to-all通信overlap起来,从而隐藏通信延迟。

#### 标准Ulysses Attention的通信瓶颈

在标准的Ulysses Attention中,通信和计算是串行的:

```
1. QKV Projection (GEMM)  <- 计算
2. All-to-all Q           <- 通信
3. All-to-all K           <- 通信
4. All-to-all V           <- 通信
5. Attention计算          <- 计算
6. All-to-all output      <- 通信
```

这种串行方式导致GPU在等待通信完成时处于空闲状态,浪费了计算资源。

#### Async Ulysses的优化思路

Async Ulysses通过将QKV projection拆分,让每个projection的GEMM和all-to-all可以overlap:

```
Timeline:
|--Q GEMM--||--Q all2all--|
          |--K GEMM--||--K all2all--|
                    |--V GEMM--||--V all2all--|
                              |--Attention--||--O all2all--|
```

具体来说:
1. **Q projection完成后立即启动Q all-to-all**,同时开始K projection
2. **K projection完成后立即启动K all-to-all**,同时开始V projection
3. **V projection完成后立即启动V all-to-all**,同时等待Q和K的通信完成
4. 所有通信完成后,进行attention计算

![](https://files.mdnice.com/user/59/4549e7b3-c4f7-4a69-a6ea-5f3447dcfe92.png)

#### 实现细节

这个优化需要修改FLUX的attention forward,将原来的:

```python
# 原始实现:串行
q, k, v = self.to_qkv(hidden_states).chunk(3, dim=-1)
q = all_to_all(q)  # 等待
k = all_to_all(k)  # 等待
v = all_to_all(v)  # 等待
out = attention(q, k, v)
```

改为:

```python
# Async实现:overlap
q = self.to_q(hidden_states)
q_async = all_to_all_async(q)  # 异步启动,不等待

k = self.to_k(hidden_states)
k_async = all_to_all_async(k)  # 异步启动,不等待

v = self.to_v(hidden_states)
v_async = all_to_all_async(v)  # 异步启动,不等待

# 等待所有通信完成
q = q_async.wait()
k = k_async.wait()
v = v_async.wait()

out = attention(q, k, v)
```

这需要:
1. 将`to_qkv`拆分为`to_q`, `to_k`, `to_v`三个独立的projection
2. 使用PyTorch的异步通信API(如`torch.distributed._functional_collectives`)
3. 确保通信和计算可以真正overlap(需要CUDA stream管理)

#### 性能收益

理论上,如果QKV projection的时间和all-to-all通信时间相当,可以隐藏大部分通信开销。实际收益取决于:
- 网络带宽和延迟
- QKV projection的计算量
- GPU和网络的并行能力

对于FLUX.1这种大模型,QKV projection的计算量较大,可以有效隐藏通信延迟。

#### 使用方式

目前这个特性是实验性的,在PR #480中实现。未来可能会通过parallel_kwargs启用:

```python
# 未来可能的API
cache_dit.enable_cache(
    pipe,
    parallelism_config=ParallelismConfig(
        ulysses_size=2,
        parallel_kwargs={
            "experimental_async_qkv_proj": True,  # 启用async QKV
        },
    ),
)
```

这个优化目前只针对FLUX.1实现,因为FLUX的attention结构比较适合这种拆分。未来会扩展到其他模型。

## Tensor Parallelism支持

除了Context Parallelism,Cache-Dit还支持Tensor Parallelism(TP)。TP通过在模型维度(比如hidden dimension、head dimension)上切分参数,让多个GPU并行计算,从而实现加速和降低单卡显存占用。

```python
cache_dit.enable_cache(
    pipe,
    cache_config=DBCacheConfig(...),
    parallelism_config=ParallelismConfig(
        tp_size=2,  # 使用2个GPU做TP
    ),
)
```

TP特别适合显存受限的场景,因为它可以将模型参数分散到多个GPU上,降低单卡显存占用。

需要注意的是,Cache-Dit目前不支持混合并行(CP + TP),用户需要根据实际场景选择使用CP或TP。


## Context Parallelism的实现细节

Cache-Dit的Context Parallelism实现基于Diffusers的原生支持。让我们深入看看CP的实现细节。

### CP Plan的核心概念

CP Plan定义了在transformer forward过程中,哪些tensor需要在哪个位置、哪个维度上进行切分(split)或聚合(gather)。

#### ContextParallelInput

`ContextParallelInput`标记一个输入tensor需要被切分:

```python
ContextParallelInput(
    split_dim=1,        # 在哪个维度切分
    expected_dims=3,    # 期望的tensor维度数
    split_output=False, # 输出是否也切分
)
```

- `split_dim`: 切分的维度,通常是序列维度(dim=1)
- `expected_dims`: 用于验证tensor的维度是否符合预期
- `split_output`: 如果为True,该层的输出也会被切分;如果为False,输出保持切分状态但不额外切分

#### ContextParallelOutput

`ContextParallelOutput`标记一个输出tensor需要被聚合:

```python
ContextParallelOutput(
    gather_dim=1,      # 在哪个维度聚合
    expected_dims=3,   # 期望的tensor维度数
)
```

### CP Plan的结构

CP Plan是一个字典,key是模块路径,value是该模块的输入输出配置。有几种常见的模式:

#### 模式1: Transformer级别切分(FLUX.1)

```python
_cp_plan = {
    "": {  # 空字符串表示transformer本身
        "hidden_states": ContextParallelInput(
            split_dim=1, expected_dims=3, split_output=False
        ),
        "encoder_hidden_states": ContextParallelInput(
            split_dim=1, expected_dims=3, split_output=False
        ),
        "img_ids": ContextParallelInput(
            split_dim=0, expected_dims=2, split_output=False
        ),
        "txt_ids": ContextParallelInput(
            split_dim=0, expected_dims=2, split_output=False
        ),
    },
    "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
}
```

这种模式在transformer forward的最开始切分所有输入,在最后聚合输出。中间的blocks不需要额外的split/gather操作,因为Ulysses Attention内部的all-to-all会自动处理。

#### 模式2: Block级别渐进切分(Wan, Qwen-Image)

```python
_cp_plan = {
    "rope": {...},  # RoPE相关配置
    "blocks.0": {
        "hidden_states": ContextParallelInput(
            split_dim=1, expected_dims=3, split_output=False
        )
    },
    "blocks.*": {  # 从第二个block开始
        "encoder_hidden_states": ContextParallelInput(
            split_dim=1, expected_dims=3, split_output=False
        )
    },
    "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
}
```

这种模式采用渐进式切分:
1. 在`blocks.0`只切分`hidden_states`
2. 从`blocks.1`开始(用`blocks.*`表示),再切分`encoder_hidden_states`
3. 最后在`proj_out`聚合

### 渐进式切分策略的原因

让我们通过Wan模型的两个分支来理解这个设计。

ChronoEdit分支无法切分encoder_hidden_states:

```python
if cls_name.startswith("ChronoEditTransformer3D"):
    _cp_plan = {
        "blocks.0": {
            "hidden_states": ContextParallelInput(...)
        },
        # 注意:没有 "blocks.*" 配置
        "proj_out": ContextParallelOutput(...),
    }
```

ChronoEdit的`encoder_hidden_states`包含text_tokens(512) + image_tokens(257) = 769个token。769是质数,无法被2、4、8等常见GPU数量整除,所以无法切分。只能切分`hidden_states`,让`encoder_hidden_states`在所有GPU上保持完整。

Wan 2.2分支可以切分encoder_hidden_states:

```python
else:  # WanTransformer3D
    _cp_plan = {
        "blocks.0": {
            "hidden_states": ContextParallelInput(...)
        },
        "blocks.*": {  # 有这个配置
            "encoder_hidden_states": ContextParallelInput(...)
        },
        "proj_out": ContextParallelOutput(...),
    }
```

Wan 2.2的`encoder_hidden_states`只有512个text_tokens,可以被2、4、8整除,所以可以切分。

不在blocks.0就切分encoder_hidden_states的原因:
1. 在transformer forward开始时,可能需要先用完整的`encoder_hidden_states`做一些初始化(如RoPE计算)
2. 不是所有模型的`encoder_hidden_states`都能切分,渐进式策略提供了更好的通用性

### CP Plan的实际执行流程

以Wan 2.2为例,执行流程如下:

```
输入:
  hidden_states: [B, 302400, D]        (视频的patch序列)
  encoder_hidden_states: [B, 512, D]   (文本tokens)

1. blocks.0 之前:
   - Split hidden_states on dim=1
   GPU 0: hidden_states[B, 151200, D]
   GPU 1: hidden_states[B, 151200, D]
   - encoder_hidden_states 保持完整
   GPU 0: encoder_hidden_states[B, 512, D]
   GPU 1: encoder_hidden_states[B, 512, D]

2. blocks.0 forward:
   - 使用切分的 hidden_states
   - 使用完整的 encoder_hidden_states
   - Ulysses Attention 内部做 all-to-all

3. blocks.1-N 之前:
   - hidden_states 已经切分(自动传播)
   - Split encoder_hidden_states on dim=1
   GPU 0: encoder_hidden_states[B, 256, D]
   GPU 1: encoder_hidden_states[B, 256, D]

4. blocks.1-N forward:
   - 两者都是切分状态
   - Ulysses Attention 内部做 all-to-all

5. proj_out 之后:
   - Gather hidden_states on dim=1
   输出: hidden_states[B, 302400, D]  (完整)
```

### 判断是否需要"blocks.*"配置

检查两个条件:

1. encoder_hidden_states是否在blocks中保持不变?
   - 如果会更新(如Pattern_0),不需要`blocks.*`
   - 如果不变(如Pattern_1/2/3),继续检查第2条

2. encoder_hidden_states的序列长度能否被GPU数量整除?
   - 能整除 → 添加`blocks.*`配置
   - 不能整除 → 不添加(像ChronoEdit)

## 命令行参数说明

Cache-Dit的examples中提供了统一的参数解析工具`utils.py`(https://github.com/vipshop/cache-dit/blob/main/examples/utils.py),支持以下参数:

### 缓存相关参数

- `--cache`: 是否启用缓存
- `--Fn`: Fn_compute_blocks,默认8
- `--Bn`: Bn_compute_blocks,默认0
- `--rdt`: residual_diff_threshold,默认0.08
- `--max-warmup-steps` / `--w`: 最大warmup步数,默认8
- `--warmup-interval` / `--wi`: warmup间隔,默认1
- `--max-cached-steps` / `--mc`: 最大缓存步数,默认-1(无限制)
- `--max-continuous-cached-steps` / `--mcc`: 最大连续缓存步数,默认-1(无限制)
- `--taylorseer`: 是否启用TaylorSeer
- `--taylorseer-order` / `-order`: TaylorSeer阶数,默认1

### 量化相关参数

- `--quantize` / `-q`: 是否启用量化
- `--quantize-type`: 量化类型,可选:
  - `float8`: FP8 dynamic quantization
  - `float8_weight_only`: FP8 weight-only(默认)
  - `int8`: INT8 dynamic quantization
  - `int8_weight_only`: INT8 weight-only
  - `int4`: INT4 dynamic quantization
  - `int4_weight_only`: INT4 weight-only
  - `bitsandbytes_4bit`: BitsAndBytes 4-bit量化

### 并行化相关参数

- `--parallel-type` / `--parallel`: 并行化类型,可选:
  - `tp`: Tensor Parallelism
  - `ulysses`: Ulysses Context Parallelism
  - `ring`: Ring Context Parallelism
- `--attn`: Attention backend,可选:
  - `flash`: FlashAttention
  - `native`: PyTorch native SDPA
  - `_native_cudnn`: cuDNN attention(默认)
  - `sage`: SageAttention
- `--ulysses-anything` / `--uaa`: 是否启用Ulysses Anything Attention
- `--disable-compute-comm-overlap` / `--dcco`: 禁用compute-communication overlap

### 编译相关参数

- `--compile`: 是否启用torch.compile

### 其他参数

- `--steps`: 推理步数
- `--height`: 图像高度
- `--width`: 图像宽度
- `--prompt`: 自定义prompt
- `--negative-prompt`: 自定义negative prompt
- `--model-path`: 自定义模型路径
- `--track-memory`: 是否跟踪显存使用
- `--fuse-lora`: 是否fuse LoRA权重
- `--perf`: 是否启用性能测试模式

### 使用示例

```bash
# 基础缓存
python run_flux_cp.py --cache --Fn 8 --Bn 0 --rdt 0.08

# 缓存 + 量化
python run_flux_cp.py --cache --quantize --quantize-type float8_weight_only

# 缓存 + Context Parallelism
torchrun --nproc_per_node=2 run_flux_cp.py --cache --parallel-type ulysses --attn flash --uaa

# 缓存 + 量化 + 编译
python run_flux_cp.py --cache --quantize --compile --Fn 1 --Bn 0

# TaylorSeer + SCM
python run_flux_cp.py --cache --taylorseer --taylorseer-order 1 --Fn 1
```

## 参考文献

1. Cache-Dit GitHub仓库: https://github.com/vipshop/cache-dit
2. 作者知乎主页: https://www.zhihu.com/people/qyjdef/posts
3. Async Ulysses CP for FLUX PR: https://github.com/vipshop/cache-dit/pull/480
4. UAA: ulysses anything attn w/ zero overhead PR: https://github.com/vipshop/cache-dit/pull/462
8. SageAttention: https://github.com/thu-ml/SageAttention

## 总结

Cache-Dit是一个PyTorch原生的DiT推理加速引擎,通过混合缓存策略和并行化技术来加速DiT模型推理。它的核心优势在于易用性、灵活性和可组合性。DBCache通过Fn和Bn两组compute blocks来平衡性能和精度,配合TaylorSeer、SCM等优化策略可以实现更好的效果。Context Parallelism的支持让Cache-Dit可以充分利用多GPU资源,Ulysses Anything Attention解决了序列长度限制问题,Async Ulysses QKV Projection通过overlap通信和计算进一步提升性能。Cache-Dit支持几乎所有主流的DiT模型,是一个值得深入学习和使用的推理优化工具。
