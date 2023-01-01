## reduce优化学习笔记

这里记录学习 NIVDIA 的[reduce优化官方博客](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) 以及 [Liu-xiandong的cuda优化工程之reduce部分博客](https://github.com/Liu-xiandong/How_to_optimize_in_GPU/tree/master/reduce) 做的笔记。

### 问题介绍

通俗的来说，Reduce就是要对一个数组求 sum，min，max，avg 等等。Reduce又被叫作规约，意思就是递归约减，最后获得的输出相比于输入一般维度上会递减。比如 nvidia 博客上这个 Reduce Sum 问题，一个长度为 8 的数组求和之后得到的输出只有一个数，从 1 维数组变成一个标量。本文就以 Reduce Sum 为例来记录 Reduce 优化。

<img width="957" alt="图片" src="https://user-images.githubusercontent.com/35585791/210163655-52f98c65-c4d6-485d-b564-c41d27fd1043.png">

### 硬件环境

NVIDIA A100-PCIE-40GB , 峰值带宽在 1555 GB/s , CUDA版本为11.8.

### 构建BaseLine

在问题介绍一节中的 Reduce 求和图实际上就指出了 BaseLine 的执行方式，我们将以树形图的方式去执行数据累加，最终得到总和。但由于GPU没有针对 global memory 的同步操作，所以博客指出我们可以通过将计算分成多个阶段的方式来避免 global memrory 的操作。如下图所示：

<img width="860" alt="图片" src="https://user-images.githubusercontent.com/35585791/210164111-1cdc34c3-e7f5-4377-883b-8fe38b624bea.png">

接着 NVIDIA 博客给出了 BaseLine 算法的实现：


<img width="864" alt="图片" src="https://user-images.githubusercontent.com/35585791/210164190-84ce2a0e-7d9b-47a5-a20b-e28d0393ba97.png">


这里的 g_idata 表示的是输入数据的指针，而 g_odata 则表示输出数据的指针。