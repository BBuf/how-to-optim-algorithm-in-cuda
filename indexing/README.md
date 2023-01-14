## indexing

- indexing_pytorch_explain.cu 这个文件用来解释 [PyTorch 中的索引相关的操作](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/Indexing.cu) 原理，添加了一些注释。
- index_add_cuda_oneflow_impl.cu 这个文件是 oneflow 里面的 index_add 操作相关的代码。
- index_add_cuda_pytorch_impl.cu 这个文件是我把 pytorch 里面的 index_add 操作相关的代码拆出来注册为oneflow的kernel去运行，kernel性能方面和oneflow互有高低，但差距比较小。


做性能profile的时候，我使用了以下脚本：

```python
import torch

x = torch.randn(32*1024*1024).to("cuda")
t = torch.randn(15).to("cuda")
index = torch.randint(0, 1024, (15,)).to("cuda")
x.index_add_(0, index, t)
torch.cuda.synchronize()

x = torch.randn(32*1024, 1024).to("cuda")
t = torch.randn(15, 1024).to("cuda")
index = torch.randint(0, 1024, (15,)).to("cuda")
x.index_add_(0, index, t)
torch.cuda.synchronize()

x = torch.randn(32, 1024, 1024).to("cuda")
t = torch.randn(15, 1024, 1024).to("cuda")
index = torch.randint(0, 32, (15,)).to("cuda")
x.index_add_(0, index, t)
torch.cuda.synchronize()

x = torch.randn(32*1024*1024).to("cuda")
t = torch.randn(1024).to("cuda")
index = torch.randint(0, 1024, (1024,)).to("cuda")
x.index_add_(0, index, t)
torch.cuda.synchronize()

x = torch.randn(32*1024, 1024).to("cuda")
t = torch.randn(1024, 1024).to("cuda")
index = torch.randint(0, 1024, (1024,)).to("cuda")
x.index_add_(0, index, t)
torch.cuda.synchronize()
```

测试环境为 A100 PCIE 40G，测试结果如下：

|框架|self tensor的shape|dim|source shape|index shape|速度|
|--|--|--|--|--|--|
|PyTorch|(32 * 1024 *1024,)|0|(15)|(15)|17.15us|
|OneFlow|(32 * 1024 *1024,)|0|(15)|(15)|12us|
|PyTorch|(32 * 1024, 1024)|0|(15, 1024)|(15)|27.78us|
|OneFlow|(32 * 1024, 1024,)|0|(15, 1024)|(15)|26.98us|
|PyTorch|(32, 1024, 1024)|0|(15, 1024, 1024)|(15)|186.88us|
|OneFlow|(32 * 1024 *1024,)|0|(15, 1024, 1024)|(15)|247.10us|
|PyTorch|(32 * 1024 *1024,)|0|(1024)|(1024)|7.9us|
|OneFlow|(32 * 1024 *1024,)|0|(1024)|(1024)|7.79us|
|PyTorch|(32 * 1024, 1024,)|0|(1024, 1024)|(1024)|27.87us|
|OneFlow|(32 * 1024, 1024,)|0|(1024, 1024)|(1024)|28.67us|

