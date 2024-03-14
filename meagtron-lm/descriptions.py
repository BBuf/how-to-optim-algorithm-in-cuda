basic_texts = "对1D并行来说，每一个并行组的GPU上的颜色都是相同的。具体点：" \
                "对于数据并行来说，每种相同的颜色表示当前进程组的这些rank上的模型权重也是相同的。" \
                "对于张量模型并行来说，每种相同的颜色表示当前进程组的这些rank维护了一部分模型权重，做allreduce才能拿到完整的模型权重" \
                "对于流水并行来说，每种相同的颜色表示当前进程组的这些rank维护的层都是串行的，需要通信传输数据，" \
                "而如果横着来看进程组我们可以发现每一行都对应了流水线并行的一个Stage，这样才能达到处理不同的micro batch流水起来的目的" \
                "对于上下文并行来说，每种相同的颜色表示当前进程组维护的是一部分token的Q，K,V，然后通过在这个进程组里面循环跨卡传递K，V来计算完整的结果，这里的通信和流水并行类似是点对点的Send/Recieve。\n"
basic_texts += "对2D并行来说，这里使用了两种颜色空间分别表示两种不同类型的并行组。\n" \
               "对3D并行来说，这里使用了三种颜色空间分别表示三种不同类型的并行组。\n" \
               "Context Parallel是一种特殊的并行，也可以理解为是对Sequence Paralle的升级，只应用在FP8模式的Self Attention层，相比于Sequence Parallel可以进一步切分Transformer中所有的Activation节省显存，" \
               "Context Parallel同样也可以和DP/TP/PP以及SP相互组合，此外Context Parallel的官方实现在 https://github.com/NVIDIA/TransformerEngine 。\n"

descriptions = {
    'Data Parallel': '对于不考虑Zero的数据并行，在一个数据并行组中，我们需要在Backward阶段对该组中所有rank的权重梯度做allreduce通信。推荐阅读以下paper或文章了解数据并行的原理：\n https://www.cs.cmu.edu/~muli/file/ps.pdf \n https://zhuanlan.zhihu.com/p/485208899 \n https://zhuanlan.zhihu.com/p/617133971 \n https://zhuanlan.zhihu.com/p/618865052',
    'Tensor Model Parallel': '在张量模型并行中，模型的不同部分（例如，不同的中间张量或权重张量）被切分到不同的GPU上，我们在Forward和Backward阶段都要做allreduce来同步激活值和参数梯度，Tensor Model Parallel是Megatron-LM的核心并行方式之一。推荐阅读以下Paper或文章了解张量模型并行的原理：\n https://arxiv.org/pdf/1909.08053.pdf \n https://strint.notion.site/Megatron-LM-86381cfe51184b9c888be10ee82f3812\n https://zhuanlan.zhihu.com/p/622212228 ',
    'Pipeline Parallel': '在流水并行中，模型的不同层被切分到不同的GPU上，并将输入数据切分为多个小批量（micro-batches），来实现在多个设备上并行处理这些批量数据，以此来加速训练过程。对于流水并行的一个组来说，每组内的GPU维护的层都是串行的，使用点对点的Send/Receive来发送接受数据，在文本框里的分组每一列都是一个完整的流水线Stage，这种组织方式才能达到处理不同的micro batch流水起来的目的。推荐阅读以下Paper或文章了解流水线并行的原理：\n https://arxiv.org/pdf/2104.04473.pdf \n https://zhuanlan.zhihu.com/p/678724323 \n https://zhuanlan.zhihu.com/p/613196255 \n https://juejin.cn/post/7063030243224879140 \n https://mp.weixin.qq.com/s/PXjYm9dN8C9B8svMQ7nOvw',
    'Context Parallel': '在长文本训练场景中，随着序列长度的增加，每个微批处理的tokens数量增多，导致训练中激活值（Activations）的显著增加，这主要与序列长度成正比。最佳解决方案是采用Context Parallel（CP），它通过沿序列维度切分并通过循环通信完成self-attention计算。但是，当CP与Tensor Parallel（TP）的乘积超过8时，Nvlink的优势可能会消失，使得循环通信效率不高。尽管如此，Context Parallel的优点在于它仅影响数据并行（DP）组的大小，允许使用CP和DP结合的全部节点进行分布式存储，这对于ZERO系列优化器尤其有利，不过目前在Meagtron-LM中只有用core下面的模型实现并且使用FP8训练才能使用CP，Megatron-LM只是提供了一个Context Parall Group，而具体的CP Group里的Attention计算和通信都在transfomer Engine中实现。推荐阅读以下Paper或文章了解Context Paralle的原理：\n https://www.zhihu.com/question/637961859/answer/3349322621 \n https://mp.weixin.qq.com/s/u4gG1WZ73mgH9mEKQQCRww',
    'DP+TP': '对于一个Batch的数据来说，TP需要在每一个TP Group里对Transformer层的Forward和Backward的Activation都做AllReduce，而DP只需要在网络的Backward阶段对所有参数做一次AllReduce，数据并行的通信量一般会显著少于模型并行。因此，在使用DP+TP组合时，我们应当在保证模型不OOM的情况下尽量提升数据并行的并行度。推荐阅读以下paper或文章了解混合并行的实践经验：\n  https://mp.weixin.qq.com/s/D-14J482SFQf-zh-EFa-1w',
    'DP+PP': '我们应当优先考虑数据并行和流水线并行的组合，而不是数据并行和模型并行，特别是在节点间带宽比较低并且模型参数很大的情况下特别有用。DP+PP的组合相比于DP+TP一般具有更小的通信量，不过PP会存在流水线气泡的问题，不过最新的一些Infra研究比如ZB-H1/H2等等可以将流水线的气泡占比降到非常低，所以流水线并行是非常有用的扩大模型规模的并行方式。此外流水线并行还常和Gradient CheckPointing技术混合使用以继续扩大模型规模，同时缓解流水线不同Stage可能存在的负载均衡问题。推荐阅读以下paper或者文章了解混合并行的实践经验：\n  https://mp.weixin.qq.com/s/D-14J482SFQf-zh-EFa-1w \n https://mp.weixin.qq.com/s/PXjYm9dN8C9B8svMQ7nOvw \n https://zhuanlan.zhihu.com/p/678724323 ',
    'DP+TP+PP': '对于单一的PP或者TP（TP一般控制在机器内，不超过8）无法满足需求的模型规模例如34b以上LLM Pretrain，我们需要使用到DP+TP+PP的混合并行。推荐阅读以下paper或文章了解混合并行的实践经验：\n https://juejin.cn/post/7063030243224879140 \n https://zhuanlan.zhihu.com/p/629121480 \n https://zhuanlan.zhihu.com/p/670374745',
    'CP+DP': '请查看Context Parallel的介绍，建议阅读下面的Paper或者文章了解Context Parallel的原理：\n https://www.zhihu.com/question/637961859/answer/3349322621 \n https://mp.weixin.qq.com/s/u4gG1WZ73mgH9mEKQQCRww',
    'CP+DP+TP': '请查看Context Parallel的介绍，建议阅读下面的Paper或者文章了解Context Parallel的原理：\n https://www.zhihu.com/question/637961859/answer/3349322621 \n https://mp.weixin.qq.com/s/u4gG1WZ73mgH9mEKQQCRww',
    'CP+DP+PP': '请查看Context Parallel的介绍，建议阅读下面的Paper或者文章了解Context Parallel的原理：\n https://www.zhihu.com/question/637961859/answer/3349322621 \n https://mp.weixin.qq.com/s/u4gG1WZ73mgH9mEKQQCRww',
    'CP+DP+TP+PP': '请查看Context Parallel的介绍，建议阅读下面的Paper或者文章了解Context Parallel的原理：\n https://www.zhihu.com/question/637961859/answer/3349322621 \n https://mp.weixin.qq.com/s/u4gG1WZ73mgH9mEKQQCRww',
}
