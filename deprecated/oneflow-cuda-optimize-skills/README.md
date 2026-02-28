## oneflow-cuda-optimize-skills

为了备忘，这里整理一下oneflow从0.7.0开始更新的所有cuda相关的kernel优化，primitive更新，cuda fuse kernel优化，以及mlir的图优化相关技术。

### 单独优化的 cuda kernel

- 优化 fast_gelu 在half下的性能：https://github.com/Oneflow-Inc/oneflow/pull/9408 。优化技巧为汇编的`tanh.approx`以及数据pack。fast_gelu的添加在 https://github.com/Oneflow-Inc/oneflow/pull/9343 这个pr，然后这个op的调用也是通过：`oneflow._C.fast_gelu` 。写这个Op的原因是因为MT5的模型支持需要。
- 针对性优化UpsampleNearest2D在上采样大小是原始图像2倍时的性能。https://github.com/Oneflow-Inc/oneflow/pull/9415 & https://github.com/Oneflow-Inc/oneflow/pull/9424 分别针对前向和反向进行了优化。主要在stable diffusion和yolov5中使用到了，优化技巧为数据pack+避免原子加。这个优化是在kernel中特判了一下，所以只用使用oneflow搭建模型时掉用了UpsampleNearest2D就可以启动这个优化。
- 继续优化layer norm的性能。https://github.com/Oneflow-Inc/oneflow/pull/9195 & https://github.com/Oneflow-Inc/oneflow/pull/9518 。9195这个pr主要是解决一些不规整数据下的性能比较慢的问题，通过padding将这种不规整数据变成规整的。9518这个pr主要是减少了前向load数据的一次cast操作。属于系统优化，layernorm在多个网络比如以Transformer为基础的网络中使用到了。只要使用到oneflow.nn.LayerNorm模块就可以享受到相关优化。LayerNorm之前的优化工作可以在：https://zhuanlan.zhihu.com/p/443026261 这里查看。
- 优化transpose op在诸如形状 [a, 1, b] , perm [0, 2, 1] 情形下的性能：https://github.com/Oneflow-Inc/oneflow/pull/9416 。Permute/Transpose 算子之前的优化工作可以在 https://zhuanlan.zhihu.com/p/425587014 这里查看。这个优化目前也是只要调用到transpose并且输入Tensor的shape满足上面的条件就是开启的。
- GroupNorm 支持 NHWC 优化以及FastDivmod优化。https://github.com/Oneflow-Inc/oneflow/pull/9368 & https://github.com/Oneflow-Inc/oneflow/pull/9373 。这里主要是支持Stable Diffusion所做的两个优化。GroupNorm这个算子是在 https://github.com/Oneflow-Inc/oneflow/issues/7784 中引入的，要开启这些优化只需要掉用`oneflow.nn.GroupNorm`即可。
- Matmul的累加支持half，通过环境变量ONEFLOW_MATMUL_ALLOW_HALF_PRECISION_ACCUMULATION启动。https://github.com/Oneflow-Inc/oneflow/pull/9459 。这里的背景应该是在FastTransformer或者VIT对Stable Diffusion的推理中，都有选择性使用FP16来进行推理，虽然这可能造成一些精度问题。
- 提供 conv 的 cutlass 版本实现。https://github.com/Oneflow-Inc/oneflow/pull/9569 & https://github.com/Oneflow-Inc/oneflow/pull/9599 。基于 cutlass 实现的 conv，在更多平台上获得比较高的性能。（这也是revert掉之前cudnn支持conv和bias_add融合算子的原因）。在Stable Diffusion中用到了。要开启这个优化需要指定环境变量：ONEFLOW_KERENL_CONV_ENABLE_CUTLASS_IMPL。
- 大幅优化 broadcast element-wise binary 一族算子反向计算时的性能 (https://github.com/Oneflow-Inc/oneflow/pull/8339) 。属于系统层次Kernel的优化，很多网络都会用到 broadcast element-wise binary 一族算子。


### cuda fuse kernel

- 实现fused_fast_gelu_mul：https://github.com/Oneflow-Inc/oneflow/pull/9397 。这个fuse kernel用在MT5模型中。使用方式：手动调用`oneflow._C.fused_fast_gelu_mul`。
- 优化RMSNorm这个op。https://github.com/Oneflow-Inc/oneflow/pull/9380 。也是在MT5模型中用到，这里的主要优化技巧就是做kernel fuse。
- 支持 fuse linear 优化，将Linear中的matmul和bias_add融合在一起。https://github.com/Oneflow-Inc/oneflow/pull/9369 。在Transformer相关的模型中有用到，并且在Stable Diffusion模型中也有用到。使用的方式为打开ONEFLOW_KERNEL_ENABLE_FUSED_LINEAR环境变量。这个优化额外引入了2次reshape操作，后续又优化掉了这两个操作，对应 https://github.com/Oneflow-Inc/oneflow/pull/9441 和 https://github.com/Oneflow-Inc/oneflow/pull/9494 。
- 支持fuse conv 和 bias_add 算子的优化。https://github.com/Oneflow-Inc/oneflow/pull/9395 ，掉用cudnn的cudnnConvolutionBiasActivationForward进行实现，但是这个fuse后续被去掉了，原因是因为其在某些版本的 cudnn 的情况下在特定 GPU上会出现bad case，性能大幅衰退的情况， cutlass 是开源的，有问题好查，使用 cutlass 还有一个重要原因是支持 half 的累加。具体见：https://github.com/Oneflow-Inc/oneflow/pull/9613 ，在Stable Diffusion中性能表现优异，需要使用 ONEFLOW_KERENL_CONV_ENABLE_CUTLASS_IMPL 环境变量打开。
- 支持fused_bias_add_scale_mask_softmax_dropout算子。https://github.com/Oneflow-Inc/oneflow/pull/9401 。可以将：

```python
masked = (x + bias) * mask * scale_value
unmask = (1 - mask).bool()
masked.masked_fill_(unmask, mask_fill_value)
softmax_y = softmax(masked, dim=-1)
y = dropout(softmax_y, p)
```

这个Pattern融合成一个cuda kernel。在MT5模型中用到了，使用方法为掉用`flow._C.fused_bias_add_scale_mask_softmax_dropout` 这个api。
- 支持 grouped_matmul 和 grouped_matmul_bias 融合算子，将一堆同时执行并且数据没有前后依赖关系的matmul+bias_add算子融合成一个cuda kernel，降低kernel launch开销。https://github.com/Oneflow-Inc/oneflow/pull/9413 。在Stable Diffusion模型中用到了，使用的方法为通过MLIR中的子图匹配加重写。后续在mlir一节中会提到。
- 新增 fused_get_boundding_boxes_coord， fused_get_center_dist，fused_get_ciou_result，fused_get_ciou_diagonal_angle，fused_get_iou，fused_get_convex_diagonal_squared 等fuse kernel用于提升yolov5在计算Loss部分的性能，加速训练。https://zhuanlan.zhihu.com/p/588637831 这里列出了每个fuse kernel对应的pr链接。使用方法为直接调用`oneflow._C.fused_xxx` api，具体可以见one-yolov5仓库。
- 新增 fused_weighted_sum 融合算子优化。https://github.com/Oneflow-Inc/oneflow/pull/9487 。原始的pattern以及融合kernel的使用方法为：

```python
model_output = (1 / 24) * (55 * self.ets[-1] - 59 * self.ets[-2] + 37 * self.ets[-3] - 9 * self.ets[-4])
model_output = oneflow._C.fused_weighted_sum([self.ets[-1], self.ets[-2], self.ets[-3], self.ets[-4]], [55, -59, 37, -9], alpha=1/24)
```

也是Stable Diffusion中的一个优化。

- 新增 multi_tensor_yolov5_weight_update kernel实现ema模型参数更新。https://github.com/Oneflow-Inc/oneflow/pull/9498 & https://github.com/Oneflow-Inc/oneflow/pull/9516 。9516是修bug，必须带上。这里是用到yolov5的滑动更新模型参数时。使用方法请看one-yolov5的`oneflow._C.multi_tensor_yolov5_weight_update` api调用。

- 新增 fused_glu 融合算子优化。https://github.com/Oneflow-Inc/oneflow/pull/9367 & https://github.com/Oneflow-Inc/oneflow/pull/9520 & https://github.com/Oneflow-Inc/oneflow/pull/9574 。用在Stable Diffusion的优化中，PR 9376实现了将下面的pattern融合成尽量少的cuda kernel：

![](https://user-images.githubusercontent.com/63904514/199904551-46dfdc0d-3eac-4dee-b91a-d566e2020f51.png)

然后 PR 9520引入了cutlass的dual_gemm进一步压缩kernel数量并实现更高的性能。
- 新增 tensorrt flash attention 的 实现。https://github.com/Oneflow-Inc/oneflow/pull/9524 & https://github.com/Oneflow-Inc/oneflow/pull/9581 。这个也是用在 Stable Diffusion的优化中。当ONEFLOW_KERENL_ENABLE_TRT_FLASH_ATTN_IMPL和ONEFLOW_MATMUL_ALLOW_HALF_PRECISION_ACCUMULATION这两个环境变量开启时并且输入数据（q,k,v）的形状以及data_type满足特定条件时会启动这个fuse kernel。使用的方法为通过MLIR中的子图匹配加重写。后续在mlir一节中会提到。

- 新增 融合算子 BinaryCrossEntropyWithLogitsReduceMean。https://github.com/Oneflow-Inc/oneflow/pull/8476 用在HugeCTR模型中提升性能。这个fuse kernel由oneflow nn.Graph的job rewrite Pass管理，无法显示开启。
- 新增基于 cublasLt 开发的高性能矩阵乘 Fused kernel。(https://github.com/Oneflow-Inc/oneflow/pull/8462, https://github.com/Oneflow-Inc/oneflow/pull/8222, https://github.com/Oneflow-Inc/oneflow/pull/8063) 。应用在 OneEmbedding 中，由oneflow nn.Graph的job rewrite Pass管理，无法显示开启。
- 提供新的环境变量优化开关： ONEFLOW_ENABLE_MULTI_TENSOR_MODEL_UPDATE 和 ONEFLOW_FUSE_MODEL_UPDATE_CAST 。ONEFLOW_ENABLE_MULTI_TENSOR_MODEL_UPDATE 表示提供MultiTensor的支持，可以将模型更新时的碎Op融合到一起减少Kernel Launch的开销。ONEFLOW_FUSE_MODEL_UPDATE_CAST 表示在 AMP 情形下，支持将 Optimizer model update kernel 和下一轮前向的 cast 算子融合。https://github.com/Oneflow-Inc/oneflow/pull/8373 。这里的优化技巧在很多地方都能用到比如 YOLOv5 就用到了。
- fuse embedding interaction 。https://github.com/Oneflow-Inc/oneflow/pull/8586 。也是oneembedding的一个fuse kernel，没有找到相关资料原始的Pattern和用法暂不清楚。
- ONEFLOW_ONE_EMBEDDING_FUSED_MLP_ASYNC_GRAD 环境变量打开后在FusedMLPGrad中支持将Dropout融合到FusedMLPGrad Kernel中。https://github.com/Oneflow-Inc/oneflow/pull/8633 。这个优化应用到 OneEmbedding 中。
- 支持 Fused BCE loss。https://github.com/Oneflow-Inc/oneflow/pull/8734 。还是oneembedding的优化，在dlrm中前向 bce + cast_f2h -> out，后向 constant_like + bce_grad ->dx，用pass将它们fuse起来在一个kernel中，减少kernel个数及空隙。使用方法：由nn.Graph的job pass来管理。
- 支持 FusedMLP Kernel，可以将多个串行的 MatMul 以及 BiasAdd 和 ReLU 融合到一起。来自oneembedding的优化，https://github.com/Oneflow-Inc/oneflow/pull/7391 。使用方式直接看PR。
- 支持 Fused MLP 反向。https://github.com/Oneflow-Inc/oneflow/pull/8462 。来自oneembedding的优化，这里使用了多Stream来计算MLP的反向，值得学习，思路应该启发自HugeCTR。使用方式直接看PR。
- quick_gelu。https://github.com/Oneflow-Inc/oneflow/pull/9623 。将`x / (1 + torch.exp(-1.702 * torch.abs(x))) * torch.exp(0.851 * (x - torch.abs(x)))`融合起来。使用方式直接调用`flow._C.quick_gelu`。

### Primitive 更新

除了运行时接口，Execution Provider 还定义了一组计算接口，用来描述深度学习框架中常用到的计算，用于简化硬件适配过程中算子的开发工作，该组接口被称之为 Primitive。相比于 Execution Provider 提供的运行时接口，Primitive 提供的接口更加松散和灵活，不同接口之间相互独立，每一个接口表示了某种硬件设备可以提供的特定的计算能力。与运行时接口类似，Primitive 系列接口的抽象更贴近设备端，开发者无需深入了解 OneFlow 的机制便可以开展适配工作。不同于在适配运行时接口时，开发者需要实现 Execution Provider 提供的所有接口，开发者在适配 Primitive 过程中，可以根据项目实际情况选择性的适配。

> v0.8.0
- 新增 `ep::primitive` 基础功能的单元测试  (https://github.com/Oneflow-Inc/oneflow/pull/8099)
- 完善了部分 Primitive `log_softmax`, `softmax`, `copynd`, `Memset`, `Memcpy`, `matmul` , `add`, binary, unary, `matmul`, `batch_matmul`, fill 等计算的单元测试 (https://github.com/Oneflow-Inc/oneflow/pull/8132, https://github.com/Oneflow-Inc/oneflow/pull/8139, https://github.com/Oneflow-Inc/oneflow/pull/8137, https://github.com/Oneflow-Inc/oneflow/pull/8109, https://github.com/Oneflow-Inc/oneflow/pull/8143, https://github.com/Oneflow-Inc/oneflow/pull/8108, https://github.com/Oneflow-Inc/oneflow/pull/8154, https://github.com/Oneflow-Inc/oneflow/pull/8154, https://github.com/Oneflow-Inc/oneflow/pull/8118 ， https://github.com/Oneflow-Inc/oneflow/pull/8291)
- 新增 `ep::primitive::constant_pad` ，优化性能，移除过时的 pad grad，使用 pad 做 pad 的反向 (https://github.com/Oneflow-Inc/oneflow/pull/8152)
- 在 Kernel 中使用 unary 的 primitive 接口代替原实现 (https://github.com/Oneflow-Inc/oneflow/pull/8270)
- 新增环境变量 ONEFLOW_EP_CUDA_CUBLAS_WORKSPACE_SIZE_MB ，用于配置 cublas workspace size (https://github.com/Oneflow-Inc/oneflow/pull/8478)
- scalar logical kernel 支持 primitive (https://github.com/Oneflow-Inc/oneflow/pull/8531)
- 使用 primitive 实现 logical not kernel (https://github.com/Oneflow-Inc/oneflow/pull/8544)
- 迁移全部的 activation kernel 使用 primitive (https://github.com/Oneflow-Inc/oneflow/pull/8300)
- bias add kernel 支持 primitive (https://github.com/Oneflow-Inc/oneflow/pull/8512)


> v0.9.0
- 实现高性能的  `ep::primitive::BroadcastElementwiseUnary `  一族接口 (https://github.com/Oneflow-Inc/oneflow/pull/8384)
- 实现 `ep::primitive::ScalarXX` math 一族接口 (https://github.com/Oneflow-Inc/oneflow/pull/8612)
- 使用 primitive 实现 Unary Math 一族的算子 (https://github.com/Oneflow-Inc/oneflow/pull/8936)
- 使用 Primitive 优化 scalar pow 反向计算 (https://github.com/Oneflow-Inc/oneflow/pull/8620)
- 使用 Primitive 实现 BroadcastBinary 一族的接口 (https://github.com/Oneflow-Inc/oneflow/pull/9311)


### mlir graph rewrite

> v0.9.0
- 支持 JIT 编译 LR 代码  (https://github.com/Oneflow-Inc/oneflow/pull/8500 , https://github.com/Oneflow-Inc/oneflow/pull/8870)
- 完善对 CSE 和 Folding 共同作用的情况支持 (https://github.com/Oneflow-Inc/oneflow/pull/9430)
- IR 支持分布式描述 sbp signature ： MLIR SBP Dialect (https://github.com/Oneflow-Inc/oneflow/pull/8492)
- 新增 `TosaMakeBroadcastablePass` 以减少不必要的 reshape 的创建 (https://github.com/Oneflow-Inc/oneflow/pull/8923)
- 支持 MLIR 代码生成 launch oneflow kernels (https://github.com/Oneflow-Inc/oneflow/pull/8980)
- `ONEFLOW_MLIR_PREFER_NHWC ` 优化支持更多的算子 (https://github.com/Oneflow-Inc/oneflow/pull/9335)
- 支持 fp16 情形下的 constant folding 优化 (https://github.com/Oneflow-Inc/oneflow/pull/9337)
- 基于 mlir-pdll 实现算子的融合重写优化 (https://github.com/Oneflow-Inc/oneflow/pull/9406 , https://github.com/Oneflow-Inc/oneflow/pull/9464 , https://github.com/Oneflow-Inc/oneflow/pull/9474  , https://github.com/Oneflow-Inc/oneflow/pull/9539  , https://github.com/Oneflow-Inc/oneflow/pull/9477 , https://github.com/Oneflow-Inc/oneflow/pull/9557)
  - FusedPadConv2d
  - DeleteSameDtypeCastop
  - FusedScaleTrilPattern
- 新增 OKL Dialect，开启 ONEFLOW_MLIR_FUSE_KERNEL_LAUNCH = 1，启动oneflow kernel launch功能打包计算型op，降低 kernel launch开销。 (https://github.com/Oneflow-Inc/oneflow/pull/9144)
- 支持 grouped_matmul_bias(batch_gemm) 融合优化 (https://github.com/Oneflow-Inc/oneflow/pull/9517) 。支持 grouped_matmul 和 grouped_matmul_bias 融合算子，将一堆同时执行并且数据没有前后依赖关系的matmul+bias_add算子融合成一个cuda kernel，降低kernel launch开销。https://github.com/Oneflow-Inc/oneflow/pull/9413 。在Stable Diffusion模型中用到了，使用的方法为通过MLIR中的子图匹配加重写。后续在mlir一节中会提到。

> v0.8.0
- Cast 消除 pass (https://github.com/Oneflow-Inc/oneflow/pull/7837 )
- 使用 MLIR 完成 常量折叠 和 融合 Conv + BN 的构图优化  (https://github.com/Oneflow-Inc/oneflow/pull/7799)
- 在 OneFlow C++ API 中支持常量折叠优化  (https://github.com/Oneflow-Inc/oneflow/pull/8124)
- 给 parsed module 提供容错检查 (https://github.com/Oneflow-Inc/oneflow/pull/8299)
- 修复常量折叠单测的 BUG  (https://github.com/Oneflow-Inc/oneflow/pull/8340)
- IREE 支持 (https://github.com/Oneflow-Inc/oneflow/pull/8249)
- 增加 `oneflow_iree(python)` 到 CI (https://github.com/Oneflow-Inc/oneflow/pull/8431)
- 移除 IR 中冗余的 output_lbns (https://github.com/Oneflow-Inc/oneflow/pull/8409)
- 提供 Variable -> constant  的 转换标记  (https://github.com/Oneflow-Inc/oneflow/pull/8412)
- 移除 IR 中硬编码处理的属性  (https://github.com/Oneflow-Inc/oneflow/pull/8420)
- 实现 AutoNHWC Pass，并提供 `ONEFLOW_MLIR_PREFER_NHWC` 环境变量，支持将常见网络的 data format 自动转换为 channels last 优化，在支持 FP16 的 NVIDIA 显卡上有明显加速效果 (https://github.com/Oneflow-Inc/oneflow/pull/7890)

### CodeGeeX 优化手段解析

针对CodeGeeX大模型的推理，OneFlow做了什么优化可以超越NVIDIA FasterTransformer库的推理速度呢？

- quick_gelu融合优化。https://github.com/THUDM/CodeGeeX/blob/main/codegeex/oneflow/codegeex_model.py#L7-L11 指的是将`x / (1 + torch.exp(-1.702 * torch.abs(x))) * torch.exp(0.851 * (x - torch.abs(x)))` 这个elementwise操作组合成的pattern融合成一个算子，在oneflow中为`flow._C.quick_gelu`。
- grouped_matmul_bias优化。https://github.com/THUDM/CodeGeeX/blob/main/codegeex/oneflow/codegeex_model.py#L101-L108 指的是将一堆同时执行并且数据没有前后依赖关系的matmul+bias_add算子融合成一个cuda kernel，降低kernel launch的开销。https://github.com/Oneflow-Inc/oneflow/pull/9413。
- 更高效的fused attention kernel（在oneflow中使用`flow._C.fused_multi_head_attention_inference_v2`调用）。在oneflow中引入了cutlass的fmha以及TensorRT的FlashAttention实现，可以在不同的数据规模调用最优的fmha实现。在此基础上oneflow针对Q，K，V可能存在的不同数据排布进行优化，具体来说oneflow的fused_multi_head_attention_inference_v2接口支持手动配置Q，K，V这三个输入tensor的数据排布。比如在CodeGeeX里面，Q，K，V的shape是[seq_lenght, batch_size, num_heads * hidden_size_per_attention_head]，我们就可以直接把Q，K，V的数据排布配置成`MB(HK)`，并且输出的数据排布也配置成MB(HK)，这样就可以避免在把Q，K，V传入fused_multi_head_attention_inference_v2之前需要额外做的reshape带来的开销了，同样输出Tensor的reshape开销也可以避免。https://github.com/THUDM/CodeGeeX/blob/main/codegeex/oneflow/codegeex_model.py#L253-L264 。这部分的cuda实现分成很多pr，这里指一下路：https://github.com/Oneflow-Inc/oneflow/pull/9950 & https://github.com/Oneflow-Inc/oneflow/pull/9933。
- CodeGeeX和大多数的自回归模型一样有一个增量推理阶段，需要把当前的key,value和上一轮的key,value concat起来，也就是：https://github.com/THUDM/CodeGeeX/blob/main/codegeex/oneflow/codegeex_model.py#L135-L140 。针对这个特殊的操作，我们也开发了一个可以配置输入输出数据排布的fuse kernel，把两个concat操作融合起来降低kernel launch以及reshape的开销。https://github.com/THUDM/CodeGeeX/blob/main/codegeex/oneflow/codegeex_model.py#L239 。在oneflow中对应https://github.com/Oneflow-Inc/oneflow/pull/9963 。
- fused matmul+bias。https://github.com/THUDM/CodeGeeX/blob/main/tests/test_inference_oneflow.py#L14 。具体来说就是将Linear中的matmul和bias_add融合在一起。https://github.com/Oneflow-Inc/oneflow/pull/9369。

上述优化既适用于FP16模式，也适用于INT8模式，接下来我们聊一下INT8 weight only quantization的motivation以及优化。经过调研，FasterTransformer的INT8模式采用了weight only quantization的方式，也就是只对Linear层的权重进行量化，但是在计算的时候仍然要反量化回FP16和Activation进行矩阵乘计算。按道理来说，加入了反量化之后速度应该变慢才对，为什么这里使用了INT8 weight quantization之后反而能加速最终的推理速度呢？这是因为在这个网络中，推理时的batch_size以及seq_length都是1，这个时候的矩阵乘法退化到了一个向量和一个矩阵相乘的情况，实际上类似于卷积神经网络中的全连接层，是一个典型的访存密集型算子。所以这里对weight进行反量化和矩阵乘法可以fuse到一起来进行加速（原因是减少了访存）。在oneflow中的实现对应：https://github.com/Oneflow-Inc/oneflow/pull/9900 。然后我基于这个算子在CodeGeeX中实现了OneFlow INT8版本的推理脚本：https://github.com/THUDM/CodeGeeX/blob/main/codegeex/quantization/quantize_oneflow.py

