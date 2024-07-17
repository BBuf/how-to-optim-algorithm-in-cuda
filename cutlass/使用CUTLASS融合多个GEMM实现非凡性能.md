![](https://files.mdnice.com/user/59/74407953-7761-40b9-ac41-4b69c8928205.png)

![](https://files.mdnice.com/user/59/c2b09b55-92ef-44fe-b0a9-8e2ba3297972.png)

这张图讨论了GPU使用中的一些背景问题和痛点：

- 尽管GPU的峰值计算能力非常强大，但在模型中的小型GEMM（通用矩阵乘法）操作中难以发挥更好的性能。
- 在诸如TensorFlow等框架中，分散的内核启动带来了严重的内核启动开销。

![](https://files.mdnice.com/user/59/73cd38ff-9787-4fce-88ea-340fab2ef06c.png)

