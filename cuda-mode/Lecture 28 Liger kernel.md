> 我的课程笔记，欢迎关注：https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode 。

# 第28课，Liger Kernel

Liger Kernel(https://github.com/linkedin/Liger-Kernel) 是一个专门为 LLM 训练设计的 Triton kernels 集合，由LinkedIn的工程师开发和维护。它能有效地将多 GPU 训练吞吐量提高 20%，并将内存使用量减少 60%。目前已经实现了与 HuggingFace 兼容的 `RMSNorm`、`RoPE`、`SwiGLU`、`CrossEntropy`、`FusedLinearCrossEntropy` 等功能，未来还会有更多。Liger Kernel可以直接与 Flash Attention、PyTorch FSDP 和 Microsoft DeepSpeed 配合使用。我们欢迎社区贡献，共同收集最佳的 LLM 训练kernel。

## 课程笔记

![](https://files.mdnice.com/user/59/31039d3c-a155-429c-9318-41a0c07c0bee.png)

![](https://files.mdnice.com/user/59/8e9c1f09-e240-42d8-ae49-89ffae4ce468.png)

这张Slides介绍了本节课的大纲。具体包括了LLM（大语言模型）训练中的性能瓶颈问题，为什么选择使用Triton框架，并介绍如何实现RMS Norm和Fused Linear Cross Entropy，这可以减少很多内存，也会提到如何测试减少的内存。同时提供了三个重要的优化技巧：收敛性测试、连续性优化和地址范围处理。最后给Liger kernel打个广告。

![](https://files.mdnice.com/user/59/51cf829d-5e5f-454b-a6b2-dbcefda1ce99.png)

这里讲到大模型的瓶颈不仅包含显存导致的OOM，还包含效率问题，有一个误区就是GPU利用率越高速度越快，实际上这是错误的理解，GPU利用率高只是说明GPU很忙，推荐阅读 https://arthurchiao.art/blog/understanding-gpu-performance/ 这篇文章来理解这个问题。此外，Profiler是理解一切性能问题的基础，推荐cuda-mode的lecture1和lecture16来学习如何使用Profiler。

![](https://files.mdnice.com/user/59/20133ed9-f65d-4db0-a236-99d411d3c463.png)


然后作者对一个LLama模型做了一个在线的Profile，我们可以看到在内存变化阶段Cross entropy有一个峰值突变，它消耗了很多内存。如下图所示：

![](https://files.mdnice.com/user/59/f6219530-19a7-484c-a867-f50b9642225e.png)

由于使用了Checkpointing技术，在前向和反向阶段的每个Transformer Block上也能观察到内存的升降，因为计算下一个Transformer Block的时候会释放当前Transformer Block占用的内存。这里的重点是Cross Entropy的内存消耗，来源是具体化logits的过程中产生的峰值内存，因为vocab size很大。

接下来作者介绍了一下kernel trace部分，从这部分我们可以看到LLama模型有很多elmentwise ops和很多cuda kernel launch的overhead。