> 纯手打，无AI。

# 0x0. 核心论点

Diffusion 领域苦 Torch Compile 久矣。

从 Diffusion 模型发明以来似乎 Torch Compile 一直在各个 Diffusion 模型中进行加速，PyTorch官方Blog里面也出现了非常多使用 Torch Compile 加速 Diffusion 模型的 blog 以及代码库，可以说 Torch Compile 有当前的地位 Diffusion 模型要占一半的功劳。

然而，Torch Compile 真的好吗？我是一个极端派，我认为 Torch Compile 是 PyTorch 进化过程中最失败的产物，Diffusion 的任何模型都不应该引入 Torch Compile。那为什么 SGLang Diffusion 中仍然存在 Torch Compile 以及 为什么我认为它是一个失败的产物？

优点：

- Torch Compile 在一些未经优化的新模型中有时候确实可以提升性能，因为它会做一些fuse kernel以及图级别的overhead reduce来提升速度。

缺点：

- Torch Compile 编译时间随着模型变得复杂，以及模型本身运行时间就很长时，编译时间也会变得非常长，甚至在SGLang Diffusion里面首次编译半小时以上的也是见过的。https://github.com/BBuf/how-to-optim-algorithm-in-cuda/issues/21 这个表格里面对 SGLang Diffusion 支持的几十个模型在H100和H200上进行了测试，torch compile部分留白的都是启动的时候编译了3分钟还无法编译完成的，无法忍受。虽然Region Compile等方式可以缓解这个问题，但是复杂度进一步增加了，这并非torch compile的设计文档描述的那么简单。
- 当我们在SGLang Diffusion里面开发了一个有效的kernel比如一个fuse kernel之后，无数开发者会碰到这个kernel在eager模式下获得end2end的提升，但是打开torch compile之后end2end性能反而有下降，大多数情况是Torch Compile的图被break掉了，也有一些情况是Torch Compile可能fuse了更广的kernel导致的。这让SGLang Diffusion的kernel部分的判断也变得更加复杂，对新手不友好。
- Torch Compile 在不同的mode参数下，不同的模型下，性能表现是不一样的。也就是下面的这个参数，在不同的模型上切换default和max-autotune会有非常奇怪的结果，有时候default更快有时候max-autotune更快，非常恶心。在SGLang社区能看到数个调整torch compile mode的pr，我的建议只有一个那就是放弃吧，外面其实根本没下雨。

![](https://files.mdnice.com/user/59/2321dbf4-d551-4bc1-839c-c58351bb7d99.png)

- 除了mode参数之外，即使我们固定一个mode参数（SGLang Diffusion之前应该对所有模型固定的就是max-autotune），当我们切换GPU时，比如从H200切换到B200和5090，很多模型的torch compile性能出现了严重的跷跷板现象，比如某个模型在H200上torch compile是一个非常明显的提升，但是换到B200性能反而是非常明显的下降，哭笑不得。
- 透明性低，没有Torch Compile的时候我们可以非常方便的知道哪些kernel慢对应到Python调用上去，用了Torch Compile之后整个模型的执行时间线在 https://ui.perfetto.dev/ 里面会变成一坨，且通过kernel name很难定位到原始Python调用位置，而且你想去看Torch Compile生成的这个kernel或者对这个kernel做评测以及二次修改也很难。
- Torch Compile越来越复杂，它不仅自己搞fuse，它还提供一些机制鼓励大家把已有的一些eager fuse放进Torch Compile，这是一个惊天阳谋，可惜还是有人上当了。这里有一个问题在于，用了这个机制之后你一定会陷入经常排查为啥在Torch Compile下有些pattern没fuse起来呢？你就得研究Torch Compile的fuse pattern机制，然后就掉入陷进，和他们共同进退。
- 在LLM里面，为了实现Prefill的CUDA Graph达到降低TTFT的效果，每个框架都需要实现PieceWise CUDA Graph。SGLang实现初期就采用了Torch Compile的方式去支持PCG，但是处理了无数个Torch Compile的dirty work之后，我们最终对Torch Compile进行了丢弃。SGLang原创了Breakable CUDA Graph达到了和使用Torch Compile的PCG一样的性能效果，完全进行了平替。BCG是SGLang的一个重要课题，我们在SGLang Diffusion里面也支持了BCG，并在一些overhead比较大的Diffusion模型上都超过使用Torch Compile的性能。后续SGLang会专门介绍这个技术，大家可以期待下blog。

至少在 Diffusion领域，我的评价就是 Torch Compile 完全是一个失败的产物，它不值得被使用。那你肯定会说，之前可能发现过在SGLang Diffusion上跑一些模型的时候开了Torch Compile之后确实拿到了性能收益，不用 Torch Compile 这个性能gap的问题咋办呢？

# 0x1. 最新SGLang Diffusion的性能结果，H100+H200测试

详细数据见：https://github.com/BBuf/how-to-optim-algorithm-in-cuda/issues/21

我们在SGLang Diffusion支持的35个模型上进行了全面的测试，可以发现现在开启Torch Compile要么比Eager慢，要么和Eager持平，几乎不存在Torch Compile比Eager快的模型了。对于那种明显overhead重的模型比如sana-1.5-1.6b，我们也可以通过Eager+BCG的方式超越Torch Compile的性能。达到了可以正式退役Torch Compile的效果（当然这个还需要再多一些机器上进行验证，但一定是没有问题的。

![](https://files.mdnice.com/user/59/eda89ef0-d4ad-4eaf-9aa7-74bc2f7ce159.png)

![](https://files.mdnice.com/user/59/50c39516-17fc-4925-80b7-c9af2722fe23.png)

看前面两列的数据对比就行了，那个--quality=high的模式是我们新加的，在那个模式下我们会有一些更激进的但是几乎不影响质量的优化方法比如Cache-DIT，更激进的kernel优化等等，所以读者就不用看了。

然后我们会好奇这样的结果是如何做到的？Torch Compile的优化是怎么被逆向以及被复刻的？答案就是 Fable + 一个简单的flow。

# 0x2. Fable + 一个简单的 prompt（flow） 即可平替或者超越Torch Compile能完成的优化

来自Nvidia的Kernel Design Agent团队给我分享的一个prompt模板：

![](https://files.mdnice.com/user/59/db85069f-2e71-44dd-a982-8e9380af1f86.png)

![](https://files.mdnice.com/user/59/9e4d536b-aa3f-44c3-8d15-f577e39d5f31.png)

分别对Vae组件和模型部分应用这种类似的prompt即可展开优化，然后耗时2周就做到了上述30多个模型的Eager下的性能都超越或者持平了Torch Compile。

这个过程也让我对Agent的使用产生了新的理解，原来这种比较固定的框架优化或者kernel过程可能并不需要很复杂的agent系统，一个prompt就能让最先进的模型给我们做到最好，但是很遗憾GPT5.6-Sol无法给我带来这种感受，我对A\的Fable体验仍然是最强的。然后Agent的长程任务能力真的很强了，不用任何复杂的框架也能完成非常复杂的任务且跑得很远。

总的来说，我们已经可以基于Fable加上上面的这种简单Flow让Agent一直优化下去达到所有Diffusion模型都基本持平或者超越Torch Compile的性能，让Torch Compile彻底退役，上面的结果已经证实了这一点。

# 0x3. 


