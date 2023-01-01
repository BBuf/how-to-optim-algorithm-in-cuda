## reduce优化学习笔记

这里记录学习 NIVDIA 的[reduce优化官方博客](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) 以及 [Liu-xiandong的cuda优化工程之reduce部分博客](https://github.com/Liu-xiandong/How_to_optimize_in_GPU/tree/master/reduce) 做的笔记。

### 问题介绍

通俗的来说，Reduce就是要对一个数组求 sum，min，max，avg 等等。Reduce又被叫作规约，意思就是递归约减，最后获得的输出相比于输入一般维度上会递减。比如 nvidia 博客上这个求和的 Reduce 问题，一个长度为 8 的数组求和之后得到的输出只有一个数，从 1 维数组变成一个标量。

<img width="957" alt="图片" src="https://user-images.githubusercontent.com/35585791/210163655-52f98c65-c4d6-485d-b564-c41d27fd1043.png">

### 硬件环境

NVIDIA A100-PCIE-40GB , 

### 构建BaseLine


