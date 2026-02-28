## 使用数据并行训练大模型的方法（Zero）

### 集合通信源语

在介绍 Zero 之前，我们需要先了解一下有哪些通信原语。以下材料来源于：https://github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/AICluster/04.primitive.pdf 。

![图片](https://user-images.githubusercontent.com/35585791/230802924-81fe214a-5c5f-4f82-b516-87e43e25410d.png)

![图片](https://user-images.githubusercontent.com/35585791/230802951-11a65018-e309-4e38-a3a5-a18068fe71ab.png)

![图片](https://user-images.githubusercontent.com/35585791/230803047-8b2f7dd8-3e03-4ec3-8225-4f3eb102a9dd.png)

![图片](https://user-images.githubusercontent.com/35585791/230803065-c8acd8db-3c54-4d58-9ce7-d5ad55caebc8.png)

![图片](https://user-images.githubusercontent.com/35585791/230803092-7f46603f-d349-4730-bfe1-ce81119ed2dc.png)


![图片](https://user-images.githubusercontent.com/35585791/230803125-e049a85e-b147-40e5-841d-f9ac77e6001b.png)

![图片](https://user-images.githubusercontent.com/35585791/230803177-e417aec6-b85b-4a5b-b168-225b51c0cb97.png)

![图片](https://user-images.githubusercontent.com/35585791/230803216-3a86252f-6fa6-4bb5-8ffd-70bdf531a38b.png)

![图片](https://user-images.githubusercontent.com/35585791/230803250-15a1cc17-ce2c-4156-87d3-f5e5c7ba41fc.png)

![图片](https://user-images.githubusercontent.com/35585791/230803290-d3d1c7ec-0b8b-4c3c-a603-d0b15766077b.png)

### Zero原理

在amp模式下，对于一个模型来说，假设模型有 $\phi$ 个可以学习的参数。那么参数和梯度的fp16副本分别需要$2\phi$个bytes来存储，也就是说这里要有$4\phi$个bytes来存储模型训练参数的fp16副本。同时，对于fp32，我们需要存储weight和在adam里面对应的2个状态，因为数据并行加起来需要的总存储量就是$12\phi$了。虽然在模型的前向和反向更新时，分别只需要$2\phi$的内存，但是模型的参数更新需要额外的$12\phi$，所以总的内存量为$16\phi$。对于有1.5B的gpt2来说，导致ddp训练时单卡需要的内存为24GB。


zero 1 是切 fp32 的参数和 优化器 的状态，fp16的参数则每张卡都完整一份，前向计算每张卡 fp32 转 fp16，再 allgather 拿到更新完整的 fp16 参数。  
zero 2 是更进一步，反向梯度计算完成之后，reducescatter之后将原来完整的 fp16 梯度也立即做切分，每张卡只保留对应的部分。
zero 3 是前向需要 allgather 一次完整的 fp16 参数，计算完成之后也释放掉，等后向需要的时候，再 allgather 一次。

完整原理可参考[李沐对zero的论文精读](https://www.bilibili.com/video/BV1tY411g7ZT/?spm_id_from=333.788&vd_source=4dffb0fbabed4311f4318e8c6d253a10) 。

也可以参考[ZeRO+DeepSpeed:微软发布的高效大规模训练套件(含详细分布式训练流程)](https://zhuanlan.zhihu.com/p/108571246)

