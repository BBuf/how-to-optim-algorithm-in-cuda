> 我的课程笔记，欢迎关注：https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode 

## CUDA-MODE课程笔记 第7课: Quantization Cuda vs Triton

![](https://files.mdnice.com/user/59/901a2917-42c1-41d5-8d37-5e4aa5593451.png)

PyTorch最近一年发布了一些生成式AI模型的案例研究，这些模型运行速度极快，且代码及其精简。这些模型比如GPT-FAST，SAM-FAST都应用了量化技术，Charles很大程度上是这些量化Kernel的主要开发者。因此，这节课由Charles来分享量化技术。


![](https://files.mdnice.com/user/59/cddc3de8-0665-499f-b048-587a2cfc5ef8.png)


这张Slides介绍了演讲者的背景和最近的研究重点，内容如下：
- Pytorch Core
    - AO (Architecture Optimization) 团队
        - 量化 (Quantization)
        - 剪枝 (Pruning)
- 最近的研究重点
    - GPU 量化
        - Segment-anything-fast, gpt-fast, sdxl-fast 等项目
        - TorchAO - 提供了一个 GitHub 链接 (https://github.com/pytorch-labs/ao)
        - Int8 动态量化
            - i8i8->i32 vs i8i8bf16->bf16
        - Int8 仅权重量化
            - bf16i8->bf16
        - Int4 仅权重量化
            - bf16i4->bf16

![](https://files.mdnice.com/user/59/d98e8c68-4a5b-40e9-bd7e-f4500f5c2f13.png)

这张Slides介绍了三种不同的量化技术：

- 动态量化流程 (Dynamic Quantization Flow):
    - 权重和激活都从浮点开始
    - 权重在**预处理**阶段量化
    - 激活在**运行时**量化
    - 使用Int8进行乘法运算
    - 使用Int32进行累加
    - 最后rescale回浮点数
- 未量化 (Not Quantized):
    - 权重和激活都保持浮点格式
    - 所有运算（乘法和累加）都使用浮点数进行
- 仅权重量化 (Weight Only Quantization):
    - 权重在**预处理**阶段量化
    - 随后立即反量化回浮点数
    - 激活保持浮点格式
    - 乘法和累加都使用浮点数进行
    - 最后有一个rescale步骤

总的来说，这张Slides展示了这三种技术在处理神经网络计算时的不同流程。动态量化通过在计算过程中使用整数运算来提高效率，而仅权重量化则只对权重进行压缩，在实际计算时仍使用浮点数。未量化的方法则完全使用浮点数，可能提供最高的精度但计算效率较低。

![](https://files.mdnice.com/user/59/582cb71d-96d5-4028-a068-9a59fe604bbd.png)

![](https://files.mdnice.com/user/59/45e10b9a-6f68-4765-bbca-5ac4be7d116d.png)


这张Slides进一步说明了动态量化（Dynamic Quantization）的概念和流程：
- 数学表达：
    - 原始公式：Y = X.W
    - 量化后的公式：Y = (Sx*Xint).(Wint * Sw)
    - 重排后的公式：Y = Sx * (Xint.Wint) * Sw
这里，Sx 和 Sw 是缩放因子，Xint 和 Wint 是量化后的整数值。
- 动态量化流程图：
    - 开始于浮点权重（Float Weight）和浮点激活值（Float Activation）
    - 权重在预处理阶段进行量化（Quantize (preprocess)）
    - 激活值在运行时进行量化（Quantize）
    - 使用 Int8 进行乘法运算（Multiplication (Int8)）
    - 使用 Int32 进行累加运算（Accumulation (Int32)）
    - 最后将结果重新缩放（Rescale (Float)）回浮点数
    - 输出浮点激活值（Float Activation）

![](https://files.mdnice.com/user/59/1ab908de-90bc-43c7-8e7d-bb0d06d5cc6c.png)

这张Slides展示了逐张量量化（per-tensor quantization）和逐token量化 + 逐通道量化（per-token + per-channel quantization）两种动态量化方式。
性能比较（以SAM模型为例，vit_h, bsz=16）：
- 无量化：运行时间 785.313 ms，峰值内存 15.279（单位未指明，可能是GB）
- 动态量化：运行时间 731.649 ms，峰值内存 18.631
另外，这里的链接是Triton的矩阵乘法教程。

结论：动态量化可以提高计算效率，在这个例子中，运行时间减少了约7%。不同的量化策略（逐张量、逐token、逐通道）可以应用于不同的张量，以优化性能和精度。虽然动态量化提高了计算速度，但它的显存占用却更多了大概是15%-20%。

![](https://files.mdnice.com/user/59/655f7a00-a6bb-4760-a08c-9f9f1b8c33cf.png)

这张Slides指出，显存增加的原因是要把int8的结果累加到int32类型中，因此相比于BFloat1增加了额外的显存。

![](https://files.mdnice.com/user/59/ee885f2b-6fc9-458e-a937-69d1fb7bf995.png)

这张Slides进一步详细介绍了动态量化（Dynamic Quantization）的概念、方法和性能比较：
- 动态量化的数学表达：
    - 原始公式：Y = X.W
    - 量化公式：Y = (Sx*Xint).(Wint * Sw)
    - 重排公式：Y = Sx * (Xint.Wint) * Sw
    - 进一步优化：Sx * (XWrescaled)
> 其中使用了不同的数据类型：
    - int8：用于Xint和Wint
    - bf16：用于Sx和Sw
    - int32：用于中间计算结果XWint
- 性能比较（以SAM模型为例，vit_h, bsz=16）：
    - 无量化：运行时间 785.313 ms，峰值内存 15.279 GB
    - 动态量化：运行时间 731.649 ms，峰值内存 18.631 GB
    - 动态量化with fusion：运行时间 695.115 ms，峰值内存 14.941 GB
结论：动态量化可以显著提高计算效率，运行时间减少约7%。动态量化with fusion进一步优化了性能，运行时间比无量化减少约11.5%，同时还略微降低了内存使用。

![](https://files.mdnice.com/user/59/65b2e338-1489-4187-8a6d-b8e7c9cf1d96.png)


这里展示的是要在Torch Compile中实现动态量化with fusion需要做出的努力，因为Torch Compile并不愿意融合乘法操作，所以作者不得不在Torch Compile的矩阵乘法kernel后强制添加一个乘法的epilogue（实际上这是一个编译器的PASS，需要匹配到矩阵乘法+乘法才能生效）。图片比较难看代码，这里贴一下：
```python
# This op is a special case of the int_mm op which we use based on the pattern
# _int_mm -> mul (defined in ../fx_passes/post_grad.py) in order to prevent
# realization of the int32 _int_mm output by forcing fusion with the mul op.
# This is only used when config.force_fuse_int_mm_with_mul = True
def tuned_fused_int_mm_mul(mat1, mat2, mat3, out_dtype, *, layout=None):
    out_dtype = (
        torch.promote_types(mat3.get_dtype(), torch.int32)
        if out_dtype is None
        else out_dtype
    )
    m, n, k, layout, mat1, mat2, mat3 = mm_args(
        mat1, mat2, mat3, layout=layout, out_dtype=out_dtype
    )
    choices: List[Dict[Any, Any]] = []
    for config in int8_mm_configs(m, n, k):
        mm_template.maybe_append_choice(
            choices,
            input_nodes=(mat1, mat2, mat3),
            layout=layout,
            **dict(mm_options(config, m, n, k, layout), ACC_TYPE="tl.int32"),
            suffix_args=1,
            epilogue_fn=V.ops.mul,
        )
    return autotune_select_algorithm("int_mm", choices, [mat1, mat2, mat3], layout)
```

然后，Triton在实现这个需求时相比于Torch Compile会很简单，一行代码即可。

![](https://files.mdnice.com/user/59/8a91dc5d-68be-4ff4-9172-c6b0e3aa2c3e.png)

![](https://files.mdnice.com/user/59/a79dda1f-11f5-4f0b-9738-32fabdbc214e.png)

