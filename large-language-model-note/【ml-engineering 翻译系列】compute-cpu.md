> 我的课程笔记，欢迎关注：https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode 

> 本篇文档的来源：https://github.com/stas00/ml-engineering 。这篇文档主要探讨了机器学习工作负载中CPU的使用情况。它详细介绍了CPU核心需求的计算方法,强调每个加速器和DataLoader工作进程都需要专用的CPU核心。文档还讨论了CPU内存的使用,指出通常每个节点的CPU内存应至少与GPU内存相当,并列举了主要的内存使用场景。此外,文档还提到了NUMA亲和性、超线程等可能影响性能的因素,以及使用mmap模式时可能出现的内存使用误判。

# CPU

截至本文撰写时,机器学习工作负载并不会大量使用CPU,因此本章内容不多。随着CPU的发展越来越像GPU,这种情况可能会改变,所以我预计本章内容会随着CPU的演进而发展。

## 你需要多少CPU核心

每1个加速器需要:

1. 1个CPU核心用于与加速器绑定的进程
2. 每个`DataLoader`工作进程需要1个CPU核心 - 通常需要2-4个工作进程。

对于语言模型来说,2个工作进程通常就足够了,特别是在数据已经预处理的情况下。

如果你需要进行动态转换,这在计算机视觉模型或视觉语言模型中经常出现,你可能需要3-4个,有时甚至更多的工作进程。

目标是能够立即从`DataLoader`中获取数据,而不阻塞加速器的计算,这意味着你需要在当前迭代运行时预处理下一次迭代的一批样本。换句话说,你的下一批数据处理时间不应超过同等大小批次的单次加速器计算迭代时间。

除了预处理之外,如果你是从云端而不是本地存储动态拉取数据,你还需要确保数据预取速度足够快,以满足为加速器提供数据的工作进程的需求。

将这个数字乘以加速器的数量,再加上操作系统需要的几个核心(比如说4个)。

如果节点有8个加速器,你有n个工作进程,那么你需要`8*(num_workers+1)+4`个核心。如果你在做NLP任务,通常每个加速器需要2个工作进程,所以`8*(2+1)+4` => 28个CPU核心。如果你在做CV训练,假设每个加速器需要4个工作进程,那么就是`8(4+1)+4` => 44个CPU核心。

如果你有比CPU核心总数更多的非常活跃的进程会发生什么?一些进程将被抢占(放入队列等待CPU核心可用),你绝对要避免任何上下文切换。

但现代云服务通常提供50-100+个CPU核心,所以通常不会出现核心不足的问题。

另请参阅异步DataLoader(https://github.com/stas00/ml-engineering/tree/master/training/performance#asynchronous-dataloader)。

### CPU卸载

一些框架,如Deepspeed(https://www.deepspeed.ai/tutorials/zero-offload/)可以将一些计算工作卸载到CPU上而不造成瓶颈。在这种情况下,你会需要额外的CPU核心。

## NUMA亲和性

参见NUMA亲和性(https://github.com/stas00/ml-engineering/blob/master/training/performance#numa-affinity)。

## 超线程

超线程(https://en.wikipedia.org/wiki/Hyper-threading)通过将每个物理核心虚拟化为2个虚拟核心,允许2个线程同时使用同一个CPU核心,从而使CPU核心数量翻倍。根据工作负载的类型,这个功能可能会或可能不会提高整体性能。这项技术的发明者英特尔表示,在某些情况下可能会带来30%的性能提升。

另请参阅是否启用超线程(https://github.com/stas00/ml-engineering/blob/master/orchestration/slurm/performance.md#to-enable-hyper-threads-or-not)。

# CPU内存

这是一个很小的章节,因为通常关于CPU内存需要了解的细节很少 - 这是一件好事!

大多数ML工作负载的计算发生在GPU上,但通常每个节点上的CPU内存应该至少与GPU上的一样多。例如,如果你使用的是带有8个80GB GPU的H100节点,那么你有640GB的GPU内存。因此,你至少需要同样多的CPU内存。但最近的高端云服务包通常配备1-2TB的CPU内存。

## ML工作负载中CPU内存的用途

- 加载模型权重,除非它们直接加载到GPU上 - 这通常是一个临时的内存使用,一旦模型被移动到GPU上就会回到零。
- 保存模型权重。在某些情况下,每个GPU直接将自己的检查点写入磁盘,在其他情况下,模型在写入磁盘之前会在CPU上重新组合 - 这也是一个临时的内存使用。
- 使用像Deepspeed(https://www.deepspeed.ai/tutorials/zero-offload/)这样的框架时可能需要参数和优化器状态卸载。在这种情况下,可能需要相当多的CPU内存。
- 在`forward`传递中计算的激活值,在`backward`路径中需要可用的激活值也可以卸载到CPU,而不是丢弃然后在反向传播过程中重新计算,以节省不必要的开销。
- `DataLoader`通常是CPU内存的主要使用者之一,有时可能会消耗大量内存。通常,每个节点上至少运行2x8个DL工作进程,所以你需要足够的内存来支持至少16个各自持有一些数据的进程。例如,在从云端流式传输数据的情况下,如果数据分片很大,这些进程可能轻易消耗数百GB的CPU内存。
- 软件本身及其依赖库使用一些CPU内存,但这个数量通常可以忽略不计。

## 需要了解的事项

- 如果`DataLoader`在`mmap`模式下使用HF `datasets`,常驻内存使用量可能看起来使用了大量CPU内存,因为它会尝试将整个数据集映射到内存中。但这是误导性的,因为如果其他地方需要内存,操作系统会将任何不需要的mmap'ed页面分页回系统。你可以在这里(https://stasosphere.com/entrepreneur-being/301-mmap-memory-leak-investigation/)阅读更多相关信息。当然,这种认识适用于任何使用`mmap`的数据集,我使用HF `datasets`作为例子是因为它被广泛使用。
