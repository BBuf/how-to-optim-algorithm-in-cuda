
# 0x0. 前言

祝大家新年快乐! 希望大家天天开心，学业有成，工作顺利。

我是在2025农历新年的大年除七，也就是春节假期之前的最后一天写下这篇个人博客。首先我会简单回顾下2024年的学习收获，然后我会聊一聊我在SGLang中度过的几个月业余开源开发的经历。无论是最近火遍全球的DeepSeek V3/R1，还是在2024年各家厂商发布的重量级MoE模型都将MoE架构重新带上了历史舞台，所以我个人把2024年定义为MoE之年。因此，最后我会讨论一下我对MoE模型中的Fused MoE核心算子的推理优化的理解，我会对比目前的一些业界开源方案以及更有前景的解法。

# 0x1. 学习和收获

在2024年，我的前3个季度是很平淡的，就是普通的写代码完成任务以及被各种新的模型发布消息轰炸，导致对几乎每周发布的新模型都无感了。这段时间我个人最大的收获应该是熟练使用了Cursor帮我完成各种重复的简单工作。

大概9月的时候我开始以做工程的角度切入LLM推理框架，首先研究的就是VLLM和SGLang，看这些框架本身还是比较痛苦，因为感觉没有触发感兴趣的点。直到后面接触到MoE模型中的Fused MoE算子实现调研优化，我才找到了我这一年以来的兴趣点。后面也从这一点出发成为了SGLang的开发者，并且在SGLang的开源贡献过程中收获了很多，不仅是技术，也包括从SGLang Team成员获得的情绪价值。认识了 @Lianmin Zheng @Yineng Zhang @Chenyang Zhao 等 SGLang Team 成员，特别是 @Yineng Zhang 在我做开源贡献时提供的专业技术帮助和情绪价值(hhh。我会在下面的0x2节聊一下我这几个月参与SGLang框架的从零开始的开源开发经历。

![SGLang 贡献图，和大佬合影](https://files.mdnice.com/user/59/b34aa9bf-8b1b-48df-b579-7e58afdaabb0.png)

此外，我也继续坚持在GiantPandaCV公众号创作了一年，原创文章大概保持了1个月2篇左右，频率这么低的原因是因为自己半开摆了。然后做了一些PyTorch Blog以及CUDA MODE课程的笔记，算是业余充电补充知识了（CUDA-MODE学习笔记我也是算子那个一个月2篇里面凑数的，看得出来是真的有点摆了，今年会在 GiantPandaCV 尽量多更一点笔记）。

我的github的两个学习笔记仓库更新频率在2024后半年降到了很低，但是star数却一直在涨，感谢关注我的网友们。我也争取新年继续更新一下，特别是优质博客链接应该多更新一下。

![](https://files.mdnice.com/user/59/5dd4ab80-439d-48a0-845e-79ac58487392.png)


# 0x2. SGlang 几个月的开源开发经历



在推理框架这块的背景下，我个人对算子开发和模型profile的这一套逻辑稍微熟悉一点，然后我之前是没有大模型推理框架开发经验的。我这里先聊一下，我是如何给SGLang做开源贡献的，大概是9月中旬开始参与。我选定的就是我熟悉的方向即算子和算子/模型性能优化。

首先是做了一些基础的bug修复，添加fused moe测试相关的工作，以及完善了一下fused moe triton算子的auto tuning脚本等，还有就是讨论了一下在低端GPU上如何恰当的使用chunked prefill的问题：

![](https://files.mdnice.com/user/59/d354a126-1345-4f20-89a5-770e277a3131.png)

接着做的贡献来自于我profile模型的一个发现，我发现VLLM和SGLang在离线推理一个相同的qwen2-7b模型时通过Nsight System拿到的结果显示sglang的rmsnorm是明显更快的。所以我在 https://github.com/sgl-project/sglang/pull/2486 这里基于Triton Benchmark写了一个rmsnorm kernel的micro benchmark对比脚本，可以证明在各个情况下SGLang使用的FlashInfer的rmsnorm总是比vllm的实现更快。我这个发现和benchmark脚本后续也被vllm采用了：https://github.com/vllm-project/vllm/pull/11241 。在此基础上继续做了一下 `write_req_to_token_pool_triton` 这个算子的 benchmark 和初步优化，然后就是在GTX 4090, h100 和 h200 上针对目前常用的 MoE 模型例如 Mixtral 8x7B/8x22B 和 Qwen2-57B-A14B 等跑了跑fused moe triton算子 Tuning，都是一些比较简单的零碎开源工作。

再之后就是最近这一个月了，参与到了DeepSeek V3的优化中，提升了 Fused MoE 模块中 `moe_align_kernel` 这个cuda算子的性能。https://github.com/sgl-project/sglang/pull/2735 & https://github.com/sgl-project/sglang/pull/2767 ，不过kernel仍有优化空间。我写的这个kernel也被vllm采用了：https://github.com/vllm-project/vllm/pull/12574 （笑

接着就是针对和DeepSeek V3同期发布的 MiniMax-01-Text模型的Lightning Attention的Prefill和Decode做了Benchmark和正确性测试，以及针对Lightning Attention的Decode做了几次Triton和CUDA优化，具体可以看[使用NCU和Cursor Claude-sonnet-3.5写出高效cuda算子的正确姿势](https://mp.weixin.qq.com/s/YEw8JZxn15CfLEnK32Jj-Q) ，同时也参与到了sgl-kernel库的维护和review。

最后，在春节假期的时候，也就是最近一周，review一个contributor提出的针对deepseek V3的fused gate优化中提出把TensorRT-LLM的Fused MoE算子单独拆分以供SGLang来定制和调用，https://github.com/sgl-project/sglang/pull/3191#issuecomment-2621438615 。然后我就负责了这个工作，踩了几个大坑把TensorRT-LLM的Fused MoE算子单独拆分并跑通了。这个工作也是为了给DeepSeek V3的优化做进一步铺垫。初步的代码在这里：https://github.com/BBuf/tensorrt-llm-moe ，这里提一下我踩的一些坑：

- 我是在H100上从TensorRT-LLM的最新分支把FusedMoE单独移出来，所以要跑通Hopper下的Cutlass MoE GEMM需要一些特殊的编译选项和架构，例如使用sm90_a而不是sm_90架构，并且需要打开Hopper下的一些NVCC Flags。这段代码配置在这里，调试了很久：https://github.com/BBuf/tensorrt-llm-moe/blob/master/setup.py#L135-L150
- FusedMoE涉及到的代码写得算是比较独立，只涉及到common和cutlass MoE GEMM两个比较独立的模块，所以代码本身是比较好抽取的，需要按照源码的目录放到固定位置，然后对于`cpp/tensorrt_llm/kernels/mixtureOfExperts`下面的MoE核心实现，由于不需要在这里考虑LORA所以逐步去手动删掉了和LORA实现相关的代码，让代码更加clean。这里有一个坑是这些文件中data type使用了TensorRT中的nvinfer1 namespace下的dtype，所以需要我们在自己的docker中`apt install  tensort`然后把TensorRT的include和lib放到正确位置。https://github.com/BBuf/tensorrt-llm-moe/blob/master/setup.py#L186-L191
- 由于Fused MoE的Cutlass Grouped GEMM依赖固定版本的CutLass，所以需要使用submodule的方式引一个CutLass到3rdparty中：https://github.com/BBuf/tensorrt-llm-moe/tree/master/3rdparty
- 接下来的一个坑是需要运行MoE Grouped GEMM的实例化Python脚本来为各种dtype和各种Tile Config的Grouped GEMM进行实例化，否则运行的时候会提示GEMM的符号找不到。Python脚本也是copy过来跑就可以：https://github.com/BBuf/tensorrt-llm-moe/blob/master/cpp/tensorrt_llm/kernels/cutlass_kernels/python/generate_kernels.py ，需要注意的一点是需要把PYTHONPATH设置到https://github.com/NVIDIA/cutlass/python目录下才可以正常运行。注意，这个脚本生生的.cu也需要加到setup.py的source才可以。
- 注意到以上问题之后基本就可以编译通过这个移植的程序了，接着就是参考一下`TensorRT-LLM/cpp/tests/kernels/mixtureOfExpertsTest.cu`调用CutlassMoeFCRunner的方式来编写一个接口并写一下测试代码来测试了。https://github.com/BBuf/tensorrt-llm-moe/blob/master/moe/moe.cpp & https://github.com/BBuf/tensorrt-llm-moe/blob/master/tests/tensorrt_llm_moe_test.py
- 在测试过程中结果对不上，debug了数天，第一个坑是renormalize的参数没有正确传递，https://github.com/BBuf/tensorrt-llm-moe/blob/master/moe/moe.cpp#L176 这里应该传递`tensorrt_llm::kernels::MOEExpertScaleNormalizationMode::RENORMALIZE,`但是我根据测试程序来写传递的是None，导致topk softmax之后的结果没对应上，这个问题通过程序对拍了2半天解决了。
- 接下来的一个问题是使用PyTorch写的测试程序的expert的权重默认是Row Major的，但是这个Cutlass GEMM的版本要求的权重是Col Major的，debug了好几天，一直在纠结为什么gemm的输入数据和其它超参数完全一样但是输出就不对，直到和 @Yineng Zhang 交流了几次才想到有可能是数据的Stride可能不对，结果真的是踩了这个坑，修复之后fp32精度下的测试就过掉了。https://github.com/BBuf/tensorrt-llm-moe/blob/master/tests/tensorrt_llm_moe_test.py#L130
- 最后一个坑是在BF16的时候发现测试没过，定位到原因是TensorRT LLM会把gating先cast成fp32再做topk softmax，我的测试程序里面的这里有点问题，修复之后就可以了。https://github.com/BBuf/tensorrt-llm-moe/blob/master/tests/tensorrt_llm_moe_test.py#L169

踩这些坑花了几乎一个假期，中途一度颓废开摆，不过还好在假期结束前debug出来了。在这之后应该会基于这个版本做一些接口封装或者feature定制化工作。

# 0x3. Fused MoE 推理方案盘点

这一节我来盘点一下2024年我们经历的Fused MoE推理算子的演进，或者说各家开源推理框架的方案。

## VLLM/SGLang Triton Fused MoE

这是VLLM/SGLang开源之初支持MoE模型的方案，直接使用了AnyScale开发的Triton Fused MoE算子，后面又在这个算子上做了一些优化包括开发Tuning策略，FP8 PerTensor, FP8 Blockwise，INT8 Fused MoE等等。目前处于功能丰富，但限制也明显，毕竟用Triton写的，相比于CutLass实现的Grouped GEMM会有性能上的削弱，并且由于Triton Grouped GEMM要求把每个expert的token数 padding到矩阵Tile config['M']的倍数引入的`moe_align_block_size` kernel也是一个开销无法忽略的算子，另外Triton Grouped GEMM1和silu等激活函数无法融合也持续增大了整个算子的overhead。关于这个Triton算子实现，硅基流动的zhuping在Triton中国做分享的时候做过一个Slides总结得比较好，我把这几页截图一下：

![](https://files.mdnice.com/user/59/852be243-cf79-43f0-9395-a423f21634d2.png)

![](https://files.mdnice.com/user/59/d132a6fd-7ead-4db5-9ad7-8e27564adb11.png)

![](https://files.mdnice.com/user/59/d2bc778a-28e6-4235-9e19-48d1dac9ab07.png)

![](https://files.mdnice.com/user/59/8898003b-9908-4de5-aaf8-7bfd8a220141.png)

![](https://files.mdnice.com/user/59/8aa7dc46-ce45-4006-8b3a-c61ca9cd000c.png)

![](https://files.mdnice.com/user/59/61589d7f-03db-4b2d-ad26-da700b116d5a.png)

![](https://files.mdnice.com/user/59/62129ff1-ae81-4622-afde-5c25c27eba4f.png)

![](https://files.mdnice.com/user/59/4f998aef-949b-45c6-953d-366767dae833.png)

![](https://files.mdnice.com/user/59/0b51bbea-ba89-4466-b7cc-c7551f8f8fab.png)


## 以Token为中心还是以Expert为中心

上面最后两张Slides提到我们即可以先确定Experts，也可以先确定Tokens。VLLM/SGlang目前的版本就是先确定Experts然后再确定Tokens。LmDeploy中提供了一个先确定Tokens再确定Expert的Fused MoE算子，https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/pytorch/kernels/cuda/fused_moe.py 。不过缺少Auto Tuning，量化相关的各种支持，感兴趣的读者可以考虑自行探索一下。

## SplitK Grouped GEMM Fused MoE

来自：https://github.com/pytorch-labs/applied-ai

我在SGLang中尝试过这个算子，为其新增了FP8的支持和Auto Tuning，但是在Qwen2-57B TP4 w8a8情况下没有看到明显的收益，如下图所示。https://github.com/sgl-project/sglang/pull/2334

![](https://files.mdnice.com/user/59/59329fec-32d0-4455-aab3-5b803d16af9f.png)

由于 SplitK GEMM 会引入`tl.atomic_add`，而Triton的`atomic_add`不支持BF16，如果要求用dtype为bf16来推理模型这里就会报错，这也是一个限制。

## SGLang Expert Parallel Fused MoE

这个我单独写过了一篇博客：[SGLang的Expert Parallel特性解读](https://mp.weixin.qq.com/s/hRVpCFynybW37jogW9_BXA)

文章的最后我也提到了它目前的优势和限制。

## TensorRT-LLM Fused MoE

上面提到的Fused MoE变种实现全都是基于Triton的，限制也比较多。TensorRT-LLM也开源了一个Fused MoE的实现，是纯CUDA写的，它不仅支持gating的融合，也支持把GEMM1后面的Activation通过Cutlass Epilogue fuse起来，它的grouped gemm也是通过cutlass来定制的，并且有特定的tuning策略。此外，它的实现也可以完美对接Expert Parallel，这应该是当前开源版本中最优的一个实现。然后由于这个实现由c++编写并且嵌入到TensorRT-LLM内部，外部如果想使用目前比较困难。在上面提到了我们正在做一个移植工作将其独立为一个单独调用的库，并且初步成功并验证了正确性，相信之后在开源社区的努力下可以更广泛的使用这个实现。请关注SGLang Project的进展。

## LMDeploy TurboMind Fused MoE

https://github.com/InternLM/lmdeploy/pull/2621

我没怎么读这个代码，这里简单提一下它的存在。它和TensorRT-LLM一样也是全程CUDA写完了Fused MoE，Grouped GEMM也是使用Cutlass来定制，我个人感觉这个和TensorRT-LLM的Fused MoE选其中一种就可以。

# 0x4. 总结

简单回顾了一下2024年做的一些开源工作和学习收获，去年过得比较愉快，希望2025年可以继续在开源上发光发热，做好本职工作并享受生活。然后我盘点了一下2024年的Fused MoE算子推理优化的相关工作，如果你对这个算子感兴趣可以和我一起交流，以上。

















