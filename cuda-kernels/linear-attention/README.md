## Linear Attention

- causal_product_cuda.cu 是对 Linear Attention 的官方cuda kernel实现进行了一个精简，尽力只保留了forward的逻辑，然后针对三种不同情况下走的 kernel 进行了详细解析。具体见 [【BBuf的CUDA笔记】十，Linear Attention的cuda kernel实现解析](https://mp.weixin.qq.com/s/1EPeU5hsOhB7rNAmmXrZRw) 和 [【BBuf的CUDA笔记】十一，Linear Attention的cuda kernel实现补档](https://mp.weixin.qq.com/s/qDVKclf_AvpZ5qb2Obf4aA)