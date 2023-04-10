## 使用数据并行训练大模型的方法（zero）

### 集合通信源语

在介绍 zero 之前，我们需要先了解一下有哪些通信原语。以下材料来源于：https://github.com/chenzomi12/DeepLearningSystem/blob/main/Frontend/AICluster/04.primitive.pdf 。

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

### zero原理
