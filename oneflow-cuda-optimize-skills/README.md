## oneflow-cuda-optimize-skills

为了备忘，这里整理一下oneflow从0.7.0开始更新的所有cuda相关的kernel优化，cuda fuse kernel优化，以及mlir的图优化相关技术。

### 单独优化的 cuda kernel

- 优化 fast_gelu 在half下的性能：https://github.com/Oneflow-Inc/oneflow/pull/9408。优化技巧为汇编的`tanh.approx`以及数据pack。fast_gelu的添加在 https://github.com/Oneflow-Inc/oneflow/pull/9343 这个pr，然后这个op的调用也是通过：`oneflow._C.fast_gelu` 。写这个Op的原因是因为MT5的模型支持需要。
- 针对性优化UpsampleNearest2D在上采样大小是原始图像2倍时的性能。https://github.com/Oneflow-Inc/oneflow/pull/9415 & https://github.com/Oneflow-Inc/oneflow/pull/9424 分别针对前向和反向进行了优化。主要在stable diffusion和yolov5中使用到了，优化技巧为数据pack+避免原子加。这个优化是在kernel中特判了一下，所以只用使用oneflow搭建模型时掉用了UpsampleNearest2D就可以启动这个优化。


### cuda fuse kernel

- 实现fused_fast_gelu_mul：https://github.com/Oneflow-Inc/oneflow/pull/9397 。这个fuse kernel用在MT5模型中。使用方式：手动调用`oneflow._C.fused_fast_gelu_mul`。
- 

### mlir graph rewrite
