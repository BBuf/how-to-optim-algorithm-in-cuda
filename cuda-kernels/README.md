# cuda-kernels

本目录收录 CUDA 基础算子的手写优化实现，每个子目录包含对应算子的 `.cu` 代码及 README 说明。

| 子目录 | 内容 |
|--------|------|
| [reduce](reduce/) | Reduce 归约优化，8 个版本从 baseline 到 shfl_down_sync+pack |
| [softmax](softmax/) | OneFlow 和 FasterTransformer 的 Softmax kernel 实现对比 |
| [elementwise](elementwise/) | OneFlow Element-wise 模板，高效带宽利用率 |
| [gemv](gemv/) | GEMV 矩阵向量乘优化，含 Half/Complex 版本 |
| [fast-atomic-add](fast-atomic-add/) | Half 类型 AtomicAdd 优化（pack + fastAtomicAdd） |
| [upsample-nearest2d](upsample-nearest2d/) | OneFlow UpsampleNearest2D 前后向优化 kernel |
| [indexing](indexing/) | PyTorch index_add 优化实现解析 |
| [linear-attention](linear-attention/) | Linear Attention CUDA kernel 实现 |
