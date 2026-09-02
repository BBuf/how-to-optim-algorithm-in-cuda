# 0x0. 前言

最近这段时间一直在使用Agent分析SGLang Diffusion的profile,也陆续往main分支合入了一批Kernel优化。在实际优化过程中经常会碰到一个问题: benchmark里把单个Kernel跑快,离真正把模型跑快还有不小的gap。Diffusion模型中也是类似。一个LayerNorm Kernel在micro benchmark里快3倍,但接进模型之后可能一点收益都没有;反过来,一次不起眼的`contiguous`或者`cat`只有几十微秒,乘上几十个Transformer Block、50个denoise step和CFG双分支,最后会吃掉相当可观的时间。

所以这篇文章不打算单独讲某一个Kernel,而是记录一下我最近基于Agent完成Qwen-Image、FLUX.2、Wan和SANA Video优化的过程,包括怎么从真实模型里定位问题、Agent怎么写Kernel以及中间碰到了哪些坑。

本文基于2026年9月2日的SGLang main分支,commit为`99b910955377375a1385122680e0daefcf706f79`。这个版本一共注册了43个Diffusion算子,对应47份实现,范围包括DiT Block、VAE以及Sequence Parallel的数据搬运。

相关代码:

https://github.com/sgl-project/sglang/tree/main/python/sglang/kernels/ops/diffusion

需要提前说明,后面出现的性能数据来自不同模型和不同硬件环境,它们只用来说明对应优化有没有效果,不能直接相加。

# 0x1. SGLang Diffusion Kernel现在是什么样子

下面这张图是当前main分支里Diffusion Kernel的大致结构:

![](https://files.mdnice.com/user/59/bea057b2-5f64-48c6-9dec-1e70b66ac90f.png)

`python/sglang/kernels/ops/diffusion`下面的目录是按算子功能划分的:

- `norm`: LayerNorm、RMSNorm、AdaLN、Residual + Norm
- `modulate`: Scale/Shift、Timestep Embedding、逐Token调制
- `rope`: RoPE、QK Norm、QKV Epilogue
- `activation`: SiLU、SwiGLU以及后续量化
- `attention`: SANA-WM GDN、SANA Video Linear Attention等
- `layout`: Ulysses QKV、USP Relayout、Varlen Pack、Wan Causal Cache
- `sites`: 模型侧的挂载、回退和Quality控制
- `ext`: Rasterizer一类需要单独编译的扩展

另外还有一个`python/sglang/kernels/kda_kernels`目录。这个目录记录Kernel的来源,不代表里面一定是某一种实现语言。例如里面既有Triton,也有JIT CUDA。现在Qwen-Image的Norm/Residual-Norm、Cosmos3的Causal Conv3D、LTX2的Residual-Gate和QKNorm Split-RoPE都在这里。

对外使用时统一从下面这个入口导入:

```python
from sglang.kernels.ops.diffusion import some_kernel
```

这里做成了lazy import。没有用到CuTe DSL、FlyDSL或者某个扩展的模型,启动时不会加载对应依赖。这个细节看起来和Kernel性能无关,但对推理框架很重要。之前也碰到过Kernel本身没问题,只是import阶段把不相关模型搞挂的情况。

我现在看一个Diffusion Kernel能不能合入,一般会检查这几件事:

1. 哪些GPU、dtype、shape和stride可以使用
2. 不支持的输入是否能回到原始PyTorch实现
3. 数值是bit-exact,还是只能放到`quality=extra-high/high`
4. Kernel在真实模型里到底有没有命中
5. 最后才看端到端性能有没有提升

这里的顺序不是随便排的。特别是第4点,Triton或者JIT CUDA失败后经常会安静地fallback。如果只看程序正常运行和最终耗时,很容易对着旧实现做一轮性能分析,最后以为新Kernel已经生效了。

# 0x2. 为什么Diffusion Kernel对数值这么敏感

## 0x2.1 数学等价不代表结果相同

Diffusion会重复执行很多个denoise step,每一步的输出都会进入下一步。只移动一次BF16 rounding的位置,误差也可能沿着后面的step继续积累。

例如下面这条很常见的AdaLN计算:

```python
hidden_states = layer_norm(hidden_states)
hidden_states = hidden_states * (1 + scale) + shift
```

最直接的融合方式是把LayerNorm和后面的乘加一起用FP32算完,最后再写回BF16。数学上没有问题,精度看起来甚至更高。但PyTorch原始路径会在LayerNorm输出和后续乘加的特定位置做BF16 rounding,如果融合Kernel不保留这些边界,结果就不会bit-exact。

ERNIE-Image上曾经测过一个普通的FP32 single-pass Norm Fusion。单个Kernel的结果看起来很接近,跑完50个step之后,最终图片对Reference只剩18.83dB PSNR。这个结果也是后来重写bit-exact LayerNorm路径的直接原因。

当前SGLang里的JIT CUDA实现会尽量复现PyTorch 2.11 `vectorized_layer_norm_kernel`的执行顺序,包括128线程Welford、固定的Shuffle Tree、`div.rn`以及对应的RSQ路径。AdaLN后面的每一步也会在和PyTorch相同的位置转回BF16。

相关实现:

https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/kda_kernels/layernorm_modulate_triton.py

## 0x2.2 第一次真实请求还要再检查一次

单元测试通过之后,模型里仍然会做一次`torch.equal`检查。原因是Torch、CUDA Driver和编译器版本都可能影响底层实现,静态测试不能覆盖所有运行环境。

实际逻辑可以简化成下面这样:

```python
if compiling_or_capturing() and not verified:
    return reference_path(x)

candidate = fused_path(x)
reference = reference_path(x)
return gate.accept_or_fallback(candidate, reference)
```

第一次Eager请求同时跑Fused和Reference,如果`torch.equal`通过,后面的请求继续使用Fused Kernel。如果不一致,这个模块或者这个shape signature会永久回退到Reference。CUDA Graph Capture期间不能插入Host Sync,所以还没有验证过的路径直接使用原实现。

对应代码在这里:

https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/diffusion/sites/bitexact_gate.py

## 0x2.3 Quality分级

有些融合确实会改变半精度运算顺序,但经过模型级验证后仍然可以使用。现在SGLang Diffusion有三个Quality级别:

- `lossless`: Reference路径加上bit-exact替换,这是默认值
- `extra-high`: 在lossless基础上打开经过质量验证的DiT/VAE Kernel Fusion
- `high`: 继续打开模型自己管理的Cache-DiT或者低精度Decode等优化

非bit-exact Fusion不是每层单独开关。SGLang会在Batch边界检查整个Transformer,只要其中一个Site不满足条件,这一组Fusion就不会挂载。这样可以避免同一个请求里一半Block走新路径,另一半Block走旧路径。

需要区分量化Checkpoint自己的实现。用户加载FP8或者NVFP4 Checkpoint以后,`quality=lossless`的Reference是这个Checkpoint原来的量化路径,不是把它恢复成BF16模型。例如Qwen-Image的Bias Absorption会移动一个BF16 rounding point,目前只在验证过的SM103上使用,它不能叫Pixel-Exact。

另外,`quality=extra-high/high`的DiT Fusion目前不能和Breakable CUDA Graph直接混用。BCG Warmup捕获的是lossless分支,请求到来后再挂新的Site,Graph Replay会绕开它。只改VAE Decode的Quality路径不受这个限制,因为BCG没有捕获VAE。

# 0x3. Qwen-Image FP8和NVFP4优化

## 0x3.1 合并QKV Projection时碰到的数值问题

Qwen-Image原来的FP8 Attention里,Image和Text分支一共有6个Q/K/V Projection。每个Projection前面还要各跑一次FP8 Quant。GEMM本身已经比较快,profile里比较显眼的是这些GEMM前后的Quant、QK RMSNorm、RoPE和`cat`。

最开始我尝试把QKV的权重统一到三份权重里的最大Scale,这样合并起来最省事,速度也不错。但是图片质量没过关,SSIM只有0.883575,LPIPS为0.075531。最后采用的方案是保留每个Shard自己的FP8 Scale,把6次GEMM和6次Activation Quant缩成2次GEMM和2次Quant,中间通过Strided View拆分结果,不额外做Copy。

GEMM后面又加了一个Blackwell JIT Kernel,一次完成下面这些操作:

1. Q/K RMSNorm
2. Interleaved RoPE
3. V Copy
4. Text和Image Token写入Joint QKV Buffer

以前这些步骤会产生多个中间Tensor和多个Kernel Launch。融合之后,GEMM输出不需要先写成几个临时Tensor,然后再被下一个Kernel读回来。

相关PR:

https://github.com/sgl-project/sglang/pull/37123

下面是1张GB300、1024x1024、两个Denoise Step的torch profiler统计图:

![](https://files.mdnice.com/user/59/960deb7d-3c90-475a-b1b0-589cf74dfd93.png)

从图里可以看到:

- Static FP8 Quant从1320次降到840次
- QK Norm + RoPE从240次降到0
- QKV Cat从360次降到0
- 新增120次Fused QKV Epilogue
- 总Kernel Launch从4989次降到3309次
- Profile窗口内的GPU时间从105.874ms降到98.955ms

端到端测试配置是1张GB300、1024x1024、50 Steps、CFG 4。Denoise Step从310.168ms降到245.247ms,请求总时间从15.635秒降到12.391秒,提升20.75%。最终PNG逐像素一致。

下面是同一个Prompt和Seed的图片对比:

![](https://files.mdnice.com/user/59/ff3d3418-e3c3-48d3-b4bf-1eb1533e2757.png)

## 0x3.2 把FP8 Quant合进前一个Operator

上面的QKV优化完成之后,trace里还剩下很多单独的FP8 Quant Kernel。它们的输入通常刚被LayerNorm或者Residual-Norm完整遍历过一次,马上又要从显存读回来做量化。

所以PR #37156做了两组Producer Fusion:

- LayerNorm/AdaLN同时输出BF16和Static E4M3
- Gated Residual + LayerNorm/AdaLN同时输出BF16和Static E4M3

BF16结果继续给Residual等分支使用,FP8结果直接送进下一个GEMM。这里有一个容易漏掉的细节,Scale计算必须使用`div.full.f32`。CUDA里的快速Reciprocal对于某些真实Checkpoint Scale会差1个FP32 ULP,刚好可能跨过E4M3的Rounding Midpoint,最后让量化结果差1个bit。

这个PR单独测试时,GB300端到端提升21.13%。和前面的QKV Projection优化组合测试,相对同一份main基线提升约33.9%。这不是把两个PR的百分比直接相加,组合路径单独跑过完整测试。

相关PR:

https://github.com/sgl-project/sglang/pull/37156

NVFP4路径也做了类似的事情。PR #37129把下面这条链合成一个Kernel:

```python
residual = residual + gate * (attention + bias)
hidden_states = layer_norm(residual)
hidden_states = hidden_states * (1 + scale) + shift
hidden_states_fp4, block_scale = fp4_quantize(hidden_states)
```

最终Kernel直接写E2M1 Packed Value和128x4 Swizzled E4M3 Scale。对应链路的GPU时间下降23.6%,但完整请求只提升1.19%。从Kernel耗时占整条Pipeline的比例来看,这个结果是合理的: 目标链路本身占比不高,即使局部时间下降了23.6%,折算到端到端也只有1.19%。

相关PR:

https://github.com/sgl-project/sglang/pull/37129

## 0x3.3 CFG Modulation Cache

Qwen-Image的Conditional和Unconditional分支是串行执行的,两个分支共享Timestep Conditioning。原实现会在两个分支里重复计算Image/Text Modulation Projection。

PR #37090加了一个Request内的一次性Cache。第一个CFG分支写入结果,第二个分支用完之后马上清空。Cache会检查输入Identity/Version、dtype和device,训练、Compile、BCG以及Capture场景继续使用原路径。

两个Step的Profile里少了224次Quantized GEMM和16次普通`addmm`,也就是每个Denoise Step少120次Projection。GB300 NVFP4端到端提升5.72%,输出bit-exact。

相关PR:

https://github.com/sgl-project/sglang/pull/37090

## 0x3.4 Bias Absorption不是无损优化

Qwen-Image Output Projection后面有单独的Bias Add。PR #37116把这个Bias延后,在下一次Residual/Norm里一起处理,GB300 NVFP4端到端提升6.07%。

这个优化改变了BF16 Rounding位置。最终验证结果为SSIM 0.950545、PSNR 29.262dB、LPIPS 0.042544,所以当前代码只在SM103上打开。我觉得这里一定要把数值口径写清楚,否则很容易把“图看起来没问题”写成“结果完全一致”。

相关PR:

https://github.com/sgl-project/sglang/pull/37116

# 0x4. FLUX.2量化Kernel优化

## 0x4.1 FP8 Producer、QKV和Token Cat

FLUX.2 ModelOpt FP8路径和Qwen-Image有点像,GEMM之间夹了很多Quant和数据整理操作。PR #37162主要做了下面几件事:

1. 给80处Norm/Activation Producer增加FP8输出
2. 合并QKV Projection,保留原有Scale
3. 把QK Norm、RoPE和Joint QKV Packing合成一个Epilogue Kernel
4. 把Attention/MLP分支的Token Cat和FP8 Quant合到一起

相关PR:

https://github.com/sgl-project/sglang/pull/37162

下面是GB300、1024x1024、5个Active Timestep的torch profiler统计:

![](https://files.mdnice.com/user/59/a1d14533-86ee-4218-93ac-eba70b7a5e64.png)

Static FP8 Quant从1196次降到199次,Cat从836次降到386次,旧的QK Norm/RoPE从96次降到0。新路径增加48次Joint Epilogue和309次Token-Cat + Quant,所以它们会正常出现在Profile里。

总Launch数从7335次降到5878次,减少19.9%。GPU Kernel时间从799.555ms降到782.259ms,只下降2.16%。这是因为Model里还有大量GEMM和Attention,减少Launch不等于同等比例减少GPU时间。

50-Step Resident Benchmark里,端到端从6847.800ms降到6628.402ms,提升3.20%,Runtime Peak Reserved Memory少了438MB。输出Pixel-Exact。

下面是Main和PR的图片对比:

![](https://files.mdnice.com/user/59/c2fb538a-a614-4f2b-8f5c-d8ea49690969.png)

## 0x4.2 NVFP4的几组融合

FLUX.2 NVFP4后面又做了3组优化。

第一组是Gated Residual Norm。它把当前Block延后的Residual Update合进下一次Norm/Modulate,相关Kernel时间下降21.85%,端到端提升0.71%,结果bit-exact。

PR: https://github.com/sgl-project/sglang/pull/37112

第二组是Token Cat + NVFP4 Quant。FLUX.2 Single-Stream Block会把6144维Attention输出和18432维MLP输出拼成24576维,然后再量化给Output Projection。新的SM103 Kernel直接从两个输入读取数据,写出E2M1 Packed Value和128x4 Swizzled Scale,不用先生成BF16 Cat Tensor。目标链路GPU时间下降58.63%,端到端提升2.39%,最终Packed Bytes完全相同。

PR: https://github.com/sgl-project/sglang/pull/37141

第三组是FC1 + SwiGLU + FC2 Input Quant。这个路径复用了现有的CuTe DSL NVFP4 GEMM Epilogue,不再生成BF16 SwiGLU中间结果。端到端提升3.32%,但它不是bit-exact,所以当前main把它放在`quality=extra-high/high`。质量测试为SSIM 0.956636、PSNR 28.908dB、LPIPS 0.027606。

这里还有一个不太好看的数据: 峰值显存增加了1386MB。原因是Fallback Layout需要额外保存一份权重。GB300显存足够时可以换取这3.32%的速度,如果部署时更在意并发容量,就不一定划算。

PR: https://github.com/sgl-project/sglang/pull/37096

# 0x5. Wan视频模型里的数据搬运优化

## 0x5.1 一个8GB的临时Tensor

Wan2.2-TI2V的逐Token Modulation原来有下面这段代码:

```python
(table.unsqueeze(0) + temb.float()).chunk(6, dim=2)
```

在704p、121帧的输入上,它会先生成一个约8GB的FP32临时Tensor,然后返回6个Stride很大的View。下游使用这些View时还要继续做`contiguous`。5-Step Trace里,这一段加上后面的Materialization会占到每个Step约14%的时间。

PR #34584用一个Triton Kernel直接写`(6, B, S, D)`布局。输入只读一次,6份结果都是连续的,后面的`contiguous`自然就没有了。另外把每个Block重复构造的RoPE Cache移到了外面。

这个优化没有改变Float32 Add顺序,最终结果bit-exact。50-Step测试中,H100 Denoise从56.53秒降到49.10秒,提升13.1%;H200从54.54秒降到47.66秒,提升12.6%。这组测试里Eager甚至比Compile更快,因为真正的问题是巨大的中间Tensor和数据搬运,并不是Python本身。

相关PR:

https://github.com/sgl-project/sglang/pull/34584

## 0x5.2 Wan Causal VAE Decode

短Step视频模型里,VAE Decode经常是固定的大头。LongLive2在H200上的完整Pipeline为4.513秒,其中Decode占2.802秒,比例为62%。继续往下看Profile,真正的Conv大约是1.26秒,周围的Clone、Cat、Pad、Layout Conversion、`repeat_interleave`和Shortcut Add用了很多时间。

原来的Causal Conv输入会依次执行:

```python
cache = hidden_states[:, :, -CACHE_T:].clone()
hidden_states = torch.cat([old_cache, hidden_states], dim=2)
hidden_states = F.pad(hidden_states, padding)
hidden_states = hidden_states.contiguous(memory_format=torch.channels_last_3d)
```

PR #34125增加了两个Kernel。

`cat_pad_channels_last_3d`一次完成Cache + Hidden拼接、Padding、NDHWC写出,并且顺便生成下一Chunk使用的紧凑Cache。`dup_up3d_add`则把`repeat_interleave + permute().contiguous() + add`合成一次Gather + Add。

下面是LongLive2 Lossless Decode Window的Profile统计:

![](https://files.mdnice.com/user/59/f61cda49-f223-48e7-a28d-6e64e8961038.png)

`aten::copy_`从513ms降到151ms,`fill_`从60ms降到0,`cat`从39ms降到6ms,`DupUp3D + Shortcut Add`从175ms降到50ms。新Cat-Pad Kernel占155ms,按照完整Event Name统计,总GPU工作量减少约470ms,和Decode Wall Time减少479ms基本对得上。

最终Decode从2.802秒降到2.323秒,提升17.1%;Pipeline从4.513秒降到4.031秒,提升10.7%;峰值显存从49.6GB降到46.1GB。Lossless路径的Frame Stream MD5和Main一致。

相关PR:

https://github.com/sgl-project/sglang/pull/34125

## 0x5.3 Ulysses QKV只做一次All-to-All

Sequence Parallel里还有一个比较典型的问题。原来的Ulysses Attention对Q、K、V分别做一次All-to-All,每次通信前后还有Relayout。继续优化单个Relayout Kernel有收益,但是3次Collective本身还在。

PR #33667把QKV写成Destination-Major Packed Buffer,然后只做一次All-to-All。为了避免每个Step反复申请Buffer,这里还加了可复用的Staging Buffer。

H200x4的Micro Benchmark提升1.28-1.31倍,Pack Kernel提升约2.7倍。Wan2.2 480p模型里,Denoise Step提升3.4%,端到端提升2.1%,输出逐Frame一致。

相关PR:

https://github.com/sgl-project/sglang/pull/33667

这部分不是从零开始写的。我基于SGLang现有的Relayout Kernel继续修改,最后把3次QKV Collective收成了1次Packed QKV Collective。

# 0x6. SANA Video Linear Attention

SANA Video的Linear Attention里有两次GEMM。原实现会先把输入转成FP32,第一段GEMM使用FP32输入和输出,然后再执行第二段GEMM。

PR #35728把第一段改成BF16输入、FP32 Accumulation和FP32输出,第二段仍然保留FP32。Micro Benchmark从0.7352ms降到0.3898ms,提升1.89倍。

这个改动会影响累计精度,所以只在`quality=extra-high/high`下使用。B300、832x480、81帧、50 Steps的测试里,端到端从53.402秒降到49.542秒,提升7.23%;Denoise Stage提升7.58%。81帧的SSIM Mean/Min为0.95467/0.94467,PSNR Mean/Min为31.63/30.19dB。

下面是第0、40、80帧的对比:

![](https://files.mdnice.com/user/59/f140154a-b890-4f36-8193-02a119868f80.jpg)

相关PR:

https://github.com/sgl-project/sglang/pull/35728

# 0x7. 几个没有带来端到端收益的实验

做Kernel优化时,我觉得失败结果也应该写进PR,否则过一段时间很容易再走一遍，特别是Agent去寻找优化机会的时候，看到这个历史记录是可以节约token的。

第一个就是前面提到的Qwen-Image QKV Common Scale。性能没有问题,但是图片质量明显下降,所以最终没有合入。

另外一个是Qwen-Image Final Adaptive LayerNorm。Fused Kernel在4608 Rows时从99.360us降到29.632us,Micro Benchmark提升3.353倍。接到完整模型以后,每个Step大概只省102us,端到端落在噪声区间,有一组测试甚至慢了约0.4%。这个Kernel最后复用了通用bit-exact实现,但PR里没有宣称模型加速。

PR: https://github.com/sgl-project/sglang/pull/37144

SANA的LayerNorm + Modulate也碰到过类似情况。GPU Kernel时间确实减少了,可Eager模式本来就被CPU Launch Overhead限制,新Kernel在Python侧的调用成本吃掉了收益。最后只在BCG路径启用,Eager继续跑原实现。

PR: https://github.com/sgl-project/sglang/pull/34015

还有一种情况更直接: Profile里压根没有新Kernel Name。这一般说明Shape Guard没过、后端编译失败或者进入了Compile/Capture Fallback。遇到这种情况,先解决命中问题,继续跑Benchmark没有意义。

# 0x8. Agent在这次优化里做了什么

这批Kernel优化里,Agent参与的比例很高。我使用过GPT-5.5、GPT-5.6、Claude Opus 4.8和Fable 5等模型。按照我的开发记录粗略统计,大约95%的代码是在Agent帮助下直接或者间接完成的。

这里说的帮助不只是生成Kernel代码。很多工作是先读取完整Profile,对比Main和Candidate的Kernel调用次数,再回到模型代码里找可以消掉的中间Tensor。实现完成以后,Agent还会继续补Shape Guard、Fallback、单元测试、模型Benchmark和PR Review。我负责确定要优化的模型路径、检查数据、判断数值和质量是否可以接受,然后决定哪些实现可以进入Main以及clean up/refactor大量AI代码。

# 0x9. SGLang Diffusion现在怎么验证Diffusion Kernel

最后记录一下我目前比较常用的基于Agent的验证流程。

第一步先确认Native Backend和Kernel确实命中。启动日志不能出现Diffusers Fallback,torch profiler里要找到新Kernel的名字和正确的调用次数。

然后跑Kernel Correctness。Bit-Exact路径使用`torch.equal`,输入会覆盖Contiguous、Strided、不同dtype、边界Shape以及Fallback。允许误差的Kernel才使用`rtol/atol`。

模型测试固定Prompt、Seed、Resolution、Steps、CFG和Checkpoint。Bit-Exact路径直接保存输出MD5;非Bit-Exact路径会跑多组Prompt的SSIM、PSNR和LPIPS。视频除了平均值,还要看最差Frame以及时间方向上的稳定性。

性能测试会分成Micro Benchmark、Denoise Step和Resident E2E。Micro Benchmark用来解释到底省了哪次读写或者Launch,真正决定要不要接入的是后两项。Cold Start会单独统计,不会混到Resident请求里。

计时前需要Warmup,边界处做CUDA Synchronize。Main和PR最好在同一台机器上交错执行,例如`main, PR, PR, main`,减少GPU频率和机器负载漂移带来的影响。

SGLang现在也给新增Diffusion Kernel约定了统一入口: Stable Facade、`KernelSpec`、公开的`can_use`、数值说明、Kernel Test和模型接线。非Bit-Exact实现还要放进Quality Gate。

完整说明在这里:

https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/diffusion/README.md

# 0xA. 总结

这段时间做下来,我对Diffusion Kernel优化最大的感受是不能只盯着GEMM，Attention这种。Qwen-Image和FLUX.2里有大量Quant、Norm、RoPE和Cat,Wan视频模型里还有更重的Layout Conversion、Causal Cache和VAE数据搬运，还有一些跨越边界的融合机会。这些操作单独看都不大,放到完整Denoise Loop里就会很可观。

另外,Micro Benchmark快不代表模型一定快。Qwen Final AdaLN就是现成的反例。每个PR最好都把Kernel调用次数、GPU时间、Denoise Step和端到端结果放在一起,能闭合多少就写多少。

数值问题也不能等到最后再补。是Bit-Exact,还是Quality-Gated,或者属于量化Checkpoint本身的数值路径,在写Kernel之前就应该确定。边界明确之后,该融合的地方大胆融合,不满足条件就回到原实现。

最后需要说明,SGLang Diffusion Kernel的全部工作是由SGLang Team和社区共同完成的。本文只记录了我借助Agent添加的一部分Fast Path优化,不能代表SGLang Diffusion Kernel的全部实现和贡献。
