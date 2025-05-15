# 0x0. 前言

SGLang 目前在单机H200上推理DeepSeek V3/R1应该是跑得最快的大模型推理框架，当然性能快慢其实不是特别好说，因为各个框架都一直处于你追我赶的快速优化中。我这里从开源技术分享的角度来盘点一下SGLang对单机规模推理的大量工程优化技巧，然后因为我本人也在这场优化中也算是比较深度的参与了，所以同时也是以一个开发者的视角来分享，希望对做大模型推理的同学有所帮助。

目前的时间是2025年5月中旬，我主要盘点一下2025年开始的优化，在这之前的一些优化可能更基操一些，就不赘述了，感兴趣可以参考一下SGLang之前比较老的版本的Release Blog之类的。话不多说，直接开始盘点。下面列举的优化有大有小，但是都是为单机规模的DeepSeek V3/R1推理性能提升做了贡献的，如果对应的优化所在的PR有性能数据我也会简单截图一下。需要说明一下，性能提升截图是控制变量的方式只突出当前优化的作用，所以你可以看到下面不同的图测出来的输出吞吐差异可能比大，这个是正常的因为main分支在不断合并优化导致性能在提升，只需要关注具体优化的有效性就可以了。

下面直接开始。

# 0x1. FP8 Block GEMM 的演进

DeepSeek V3/R1 中单独的Linear对应的forward就是FP8 Block GEMM，经历了3次演讲。首先是Triton的实现，代码仍保留在这里：https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/fp8_kernel.py#L743 ，当然这里也存在对各种参数在各种平台上的调优。由于性能比较差，后续演进到使用了Cutlass的实现，具体见sgl-kernel中的：https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu 。随后，DeepSeek的DEEPGEMM开源，SGLang更进一步使用了DEEPGEMM的实现，并且解决好了缓存JIT编译结果的问题，具体见：https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/deep_gemm.py & https://github.com/sgl-project/sglang/blob/3e350a931e990c2b09cd18bc117ba219310bcda9/python/sglang/srt/layers/quantization/fp8_kernel.py#L784 。

现在SGLang中默认开启了DEEPGEMM，这个实现相比于sgl-kernel的cutlass实现，Triton实现基本上所有的case下都有优势，使用逻辑截图下：

![](https://files.mdnice.com/user/59/5fd75e1b-563a-4f17-a5fc-e37a86a643f9.png)

![](https://files.mdnice.com/user/59/e1da3949-0a2f-4f9a-a3e2-1461716d9ab7.png)

端到端的提升幅度和输入/输出数据长短有关系，例如10k-500这组数据输出吞吐端到端提升5%，256-4k这组数据输出吞吐端到端提升10%，这里的提升是相比于sgl-kernel cutlass实现来说的。

此外，基于DeepGEMM，SGLang还尝试将模型中的一系列BMM进行改写，核心idea是：`per-token-group quant+deep_gemm's grouped_gemm_masked` 比 `per-tensor quant+bmm_fp8` 更快 (https://github.com/sgl-project/sglang/pull/5432) 。不过这个优化可能会导致准确率出现问题，没有在生产环境中默认使用。

# 0x2. FusedMoE 模块的优化

这里分点列一下。

## 0x2.1 per_token_group_quant 和 moe_align_block_size 的kernel优化

代码见：https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/gemm/per_token_group_quant_8bit.cu & https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/moe/moe_align_kernel.cu 。

这两个操作都是FusedMoE模块的预处理部分，kernel整体占比没有很大，都是memory bound的kernel，但是通过优化可以把单个kernel的性能都提升很多倍。具体可以见PR开发时的micro benchmark结果，例如：https://github.com/sgl-project/sglang/pull/5086 。

这个也是我入坑SGLang开源的头几个小工作。

## 0x2.2 biased_grouped_topk 的 fuse kernel 优化

我们在SGLang中针对biased_grouped_topk(https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/topk.py#L144)引入了一个fuse cuda kernel的优化，将十多个算子fuse成一个cuda kernel大大改进了grouped topk这块的性能。之前也写了一篇blog介绍这个优化：[图解DeepSeek V3 biased_grouped_topk cuda融合算子fused_moe_gate kernel](https://mp.weixin.qq.com/s/p6LlY4sUBTy-Xfc9WumNSw) ，感兴趣可以查看，这里就不多赘述了。

![](https://files.mdnice.com/user/59/100d1c4f-a078-4a02-b290-c3bfd985edb4.png)

DeepSeek V3/R1 推理性能端到端提升是5%-8%左右。

## 0x2.3 将Shared Experts和Route Experts融合

具体细节可以参考我之前写的这篇blog，这个优化花了我挺多时间去测试的，也是一个比较solid提升，[分享一个DeepSeek V3和R1中 Shared Experts和普通Experts融合的一个小技巧](https://mp.weixin.qq.com/s/Bz3qdkldULZiZ8ypooOX-A) 。下面是端到端的性能提升图：

![](https://files.mdnice.com/user/59/2af2a88f-28a1-45e1-80f6-b004cd7ce744.png)

## 0x2.4 Triton Fused MoE Retuning

升级PyTorch Triton版本之后社区小伙伴发现重新Tuning fused MoE kernel之后可以带来明显的性能提升，例如在Triton 3.2.0中如果继续延用Triton 3.1.0 tuning的config，则性能反而会下降，但如果重新Tuning则可以取得相比于Triton 3.1.0更好的性能。https://github.com/sgl-project/sglang/pull/5716 & https://github.com/sgl-project/sglang/pull/5740 ，通过重新Tuning Fused MoE kernel，在DeepSeek V3/R1上取得了性能提升。

![](https://files.mdnice.com/user/59/cb3c8a78-8300-4b1b-b1ca-c098275e97fc.png)

我还测试了一下Triton 3.2.0升级为Triton 3.3.0的提升：

![](https://files.mdnice.com/user/59/ad528d83-80b5-42fc-8633-dca519786b99.png)

基本也是符合这里的Retuning结论的。然后 https://github.com/vllm-project/vllm/pull/17934#issuecomment-2868822690 这里的一个 micro benchmark 性能测试也佐证了这一点。

## 0x2.5 Fuse routed scaling factor in topk_reduce kernel

把expert计算完之后最后乘以routed_scaling_factor的逻辑fuse到Fused MoE模块的最后那个topk_reduce_sum kernel中，具体可见：https://github.com/sgl-project/sglang/pull/6220 ，端到端提升如下：

![](https://files.mdnice.com/user/59/9a8bb502-470d-4ec9-828d-2372b6b483cb.png)

## 0x2.6 一些额外的探索

在SGLang中也探索了一下TP模式下的基于Cutlass Grouped GEMM和DEEPGEMM的fused moe kernel实现，都跑通了正确性测试和性能测试，但是在TP8模式下对DeepSeek V3/R1没有看到相比于Triton Fused MoE kernel的性能提升效果。

# 0x3. Attention Backend的优化

## 0x3.1 Flash Attention V3 Backend

来自LinkedIn的优化，详情可见[在 SGLang 中实现 Flash Attention 后端 - 基础和 KV 缓存](https://mp.weixin.qq.com/s/693f008zNo7olXeSogy-sg) ，端到端吞吐提升结果如下所示：

![](https://files.mdnice.com/user/59/c0a5c601-1a75-4459-a1f3-43b4def58d80.png)

效果非常显著，然后SGLang社区也在推进hybrid Attention Backend以获得更加广泛的加速，可以参考：https://github.com/sgl-project/sglang/pull/6151 。

## 0x3.2 Cutlass MLA attention backend & FlashMLA Attention Backend

PR见：https://github.com/sgl-project/sglang/pull/5390 ，https://github.com/sgl-project/sglang/pull/4514 , https://github.com/sgl-project/sglang/pull/6034 等等。

虽然有很多的Attention Backend可选，例如最开始用的是FlashInfer的Backend，不过目前SGLang默认使用了Flash Attention V3 Backend，至少在H200上大多数Case下都拥有最好的性能。

# 0x4. MLA的优化

## 0x4.1 forward_absorb 中多余的copy

对应https://github.com/sgl-project/sglang/pull/5578，这个优化移除了q_input和k_input等临时变量的创建和处理，去掉了多个涉及self.kv_lora_rank的复制和切片操作。修改如下所示：

![](https://files.mdnice.com/user/59/02481ff0-5907-4bed-8829-93b1663390d3.png)

benchmark结果如下所示：

![](https://files.mdnice.com/user/59/68c505ad-a93e-4f76-a994-f3c59989803b.png)

## 


