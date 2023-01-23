## oneflow-cuda-optimize-skills

为了备忘，这里整理一下oneflow从0.7.0开始更新的所有cuda相关的kernel优化，primitive更新，cuda fuse kernel优化，以及mlir的图优化相关技术。

### 单独优化的 cuda kernel

- 优化 fast_gelu 在half下的性能：https://github.com/Oneflow-Inc/oneflow/pull/9408。优化技巧为汇编的`tanh.approx`以及数据pack。fast_gelu的添加在 https://github.com/Oneflow-Inc/oneflow/pull/9343 这个pr，然后这个op的调用也是通过：`oneflow._C.fast_gelu` 。写这个Op的原因是因为MT5的模型支持需要。
- 针对性优化UpsampleNearest2D在上采样大小是原始图像2倍时的性能。https://github.com/Oneflow-Inc/oneflow/pull/9415 & https://github.com/Oneflow-Inc/oneflow/pull/9424 分别针对前向和反向进行了优化。主要在stable diffusion和yolov5中使用到了，优化技巧为数据pack+避免原子加。这个优化是在kernel中特判了一下，所以只用使用oneflow搭建模型时掉用了UpsampleNearest2D就可以启动这个优化。
- 继续优化layer norm的性能。https://github.com/Oneflow-Inc/oneflow/pull/9195 & https://github.com/Oneflow-Inc/oneflow/pull/9518 。9195这个pr主要是解决一些不规整数据下的性能比较慢的问题，通过padding将这种不规整数据变成规整的。9518这个pr主要是减少了前向load数据的一次cast操作。属于系统优化，layernorm在多个网络比如以Transformer为基础的网络中使用到了。只要使用到oneflow.nn.LayerNorm模块就可以享受到相关优化。LayerNorm之前的优化工作可以在：https://zhuanlan.zhihu.com/p/443026261 这里查看。
- 优化transpose op在诸如形状 [a, 1, b] , perm [0, 2, 1] 情形下的性能：https://github.com/Oneflow-Inc/oneflow/pull/9416 。Permute/Transpose 算子之前的优化工作可以在 https://zhuanlan.zhihu.com/p/425587014 这里查看。这个优化目前也是只要调用到transpose并且输入Tensor的shape满足上面的条件就是开启的。
- GroupNorm 支持 NHWC 优化以及FastDivmod优化。https://github.com/Oneflow-Inc/oneflow/pull/9368 & https://github.com/Oneflow-Inc/oneflow/pull/9373 。这里主要是支持Stable Diffusion所做的两个优化。GroupNorm这个算子是在 https://github.com/Oneflow-Inc/oneflow/issues/7784 中引入的，要开启这些优化只需要掉用`oneflow.nn.GroupNorm`即可。
- Matmul的累加支持half，通过环境变量ONEFLOW_MATMUL_ALLOW_HALF_PRECISION_ACCUMULATION启动。https://github.com/Oneflow-Inc/oneflow/pull/9459 。这里的背景应该是在FastTransformer或者VIT对Stable Diffusion的推理中，都有选择性使用FP16来进行推理，虽然这可能造成一些精度问题。


### cuda fuse kernel

- 实现fused_fast_gelu_mul：https://github.com/Oneflow-Inc/oneflow/pull/9397 。这个fuse kernel用在MT5模型中。使用方式：手动调用`oneflow._C.fused_fast_gelu_mul`。
- 优化RMSNorm这个op。https://github.com/Oneflow-Inc/oneflow/pull/9380 。也是在MT5模型中用到，这里的主要优化技巧就是做kernel fuse。
- 支持 fuse linear 优化，将Linear中的matmul和bias_add融合在一起。https://github.com/Oneflow-Inc/oneflow/pull/9369 。在Transformer相关的模型中有用到，并且在Stable Diffusion模型中也有用到。使用的方式为打开ONEFLOW_KERNEL_ENABLE_FUSED_LINEAR环境变量。这个优化额外引入了2次Transpose操作，后续又优化掉了这两个操作，对应 https://github.com/Oneflow-Inc/oneflow/pull/9441 和 https://github.com/Oneflow-Inc/oneflow/pull/9494。
- 支持fuse conv 和 bias_add 算子的优化。https://github.com/Oneflow-Inc/oneflow/pull/9395 ，掉用cudnn的cudnnConvolutionBiasActivationForward进行实现，但是这个fuse后续被去掉了，可能的原因是对cudnn的版本有明确的要求，回退到了普通的实现。但是在输入数据排布为NHWC的情况下，oneflow实现了一版基于cutlass的卷积，性能更优，并且可以直接处理 conv 和 bias_add 的融合。具体见：https://github.com/Oneflow-Inc/oneflow/pull/9613 ，在Stable Diffusion中性能表现优异，需要使用 ONEFLOW_KERENL_CONV_ENABLE_CUTLASS_IMPL 环境变量打开。
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

### Primitive 更新

- 实现高性能的  `ep::primitive::BroadcastElementwiseUnary `  一族接口 (https://github.com/Oneflow-Inc/oneflow/pull/8384)
- 实现 `ep::primitive::ScalarXX` math 一族接口 (https://github.com/Oneflow-Inc/oneflow/pull/8612)
- 使用 primitive 实现 Unary Math 一族的算子 (https://github.com/Oneflow-Inc/oneflow/pull/8936)
- 使用 Primitive 优化 scalar pow 反向计算 (https://github.com/Oneflow-Inc/oneflow/pull/8620)
- 使用 Primitive 实现 BroadcastBinary 一族的接口 (https://github.com/Oneflow-Inc/oneflow/pull/9311)


### mlir graph rewrite
