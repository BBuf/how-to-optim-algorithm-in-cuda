# 0x0. 前言

[入选Codex for Open Source](https://mp.weixin.qq.com/s/Nh76NrekJwYlLlFKlpRJJw) 之后我就立马开蹬了，把之前攒的一些想法来让Codex尝试去实现，看看Codex能把框架开发助理到什么程度。然后今天Codex无意中给我的一个提醒让我觉得Codex GPT5.4 High正在重塑传统推理框架开发的流程，时代真是在巨变。下面就分享一下我获得这个结论的例子，这里顺便说一下，拿到Codex的这周业余做了一些实验（开了GPT5.4 High）很快就用掉了本周Codex用量的70%，但是刚才发现Codex的额度刷新了，又变成了100%，OpenAI真的是大气，反观某社都不给通过的。

# 0x1. 背景

我做的第一件事就是让Codex把之前大部分用Cursor写出来和微调的SGLang Diffusion的SKILLS流程完善一下，里面涉及到的模型分析和profile流程，kernel添加流程都用Codex主动去验证一遍，修复了很多SKILLS的bug让它真正成为了输入类似"帮我优化一下SGLang Diffusion FLUX1模型"就会自动启动benchmark，profile， apply fuse kernel ， 优化kernel的全套流程的SKILL，耗费了很多token才修复了所有的bug并且让整个流程顺利工作。不过遗憾的是在没有更强的人类先验的情况下优化基本失败了，我让Codex自己去试了2天基于这些SKILL优化各种模型作用有限。

# 0x2. Codex做Benchmark带来的性能优化

因为SGLang Diffusion里面使用的rmsnorm目前开源社区有很多版本，之前就想过是不是要做一个最优选择的方案，有了Codex就可以把这个想法变成现实了。让Codex做了benchmark脚本，然后发现各种rmsnorm在不同的真实shape（也就是不同的Diffusion模型用到的shape）下性能表现不同，具体可见这里的表格：

![](https://files.mdnice.com/user/59/8f82f52a-134d-4ef3-9518-a1d349475217.png)

是当我让Codex把这个flashinfer的rmsnorm应用到SGLang Diffusion模型时发现这些模型的性能几乎都下降了，让Codex去找原因，它找到的原因是因为这个kernel调用会让torch compile break的次数更多，因为没有注册custom op，然后它修了一下性能就合理了，相比于基线模型的性能有轻微提升。

然后Codex提示目前torch compile graph break还是比较多，类似flashinfer rmsnorm的可能有影响的是flashinfer rope的调用

![](https://files.mdnice.com/user/59/d326eddf-439c-43f8-8dfa-395bb06264f3.png)

并且观察它的debug流程发现它是调用`torch._dynamo.explain`来确认当前kernel是否可以torch compile编译到一个graph里面，也把这个过程写到了SKILLS里面：

![](https://files.mdnice.com/user/59/08ba16c1-eb15-457e-b0e0-e6f4a39cd97a.png)

接着让Codex继续往下修复了一下这个flashinfer rope注册custom op的问题，得到的结果如 https://github.com/sgl-project/sglang/pull/20699 所示：

![](https://files.mdnice.com/user/59/93369362-1bc6-401d-a562-81e70bf309d3.png)

qwen-image-2512 单卡H200生产1024x1024的图性能提升了20%


- main

![](https://files.mdnice.com/user/59/e1e033aa-21a2-46ea-b956-a2bcf116ec9f.png)

一个step的一个block：5ms763us

![](https://files.mdnice.com/user/59/595b3ccd-15c2-4109-9dee-7cce0165f1c3.png)

rope kernel is not covered by torch compile graph

- pr

![](https://files.mdnice.com/user/59/7e8b80f6-7af6-4dfc-a435-0f6d94b76486.png)

一个step的一个block：4ms592us

![](https://files.mdnice.com/user/59/e99419b1-9f9a-4f08-be1c-7effd8b2b2c1.png)

rope kernel is covered by torch compile graph

从profiler结果也可以看到现在flashinfer rope已经在torch compile region里面了，并且提升非常明显。

此外，我们还可以看到rope前面的2个qknorm理论上来说也是可以被torch compile region包含的：

![](https://files.mdnice.com/user/59/156820eb-debb-4cd4-814c-53f017c7997b.png)

这就可以继续让Codex蹬了。

# 0x3. 总结

个人感觉目前除了专业性非常强的kernel外，推理框架开发上Codex以及最先进的模型可以做的事情已经非常强了，我举的这个例子对torch compile和sglang即使熟悉一种的工程师都很难发现问题，但Codex近乎完美的发现和解决，这就是生产力。


