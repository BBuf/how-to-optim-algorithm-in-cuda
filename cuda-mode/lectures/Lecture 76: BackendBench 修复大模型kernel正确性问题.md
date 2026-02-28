> 我的课程笔记，欢迎关注：https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode 。

## 课程总结

本节课介绍了BackendBench，这是一个用于评估LLM生成PyTorch backend代码能力的测试套件。课程重点讨论了在修复大模型生成的kernel bug过程中遇到的算子正确性问题。在测试方法论方面，通过PyTorch OpInfo进行边界情况测试，使用真实Hugging Face模型的tensor shape进行性能验证。实验结果显示，使用Claude和简单的agent重试机制，能让53%的PyTorch OpInfo算子用Triton正确实现， 生成的84个kernel性能达到PyTorch的70-100%。课程还探讨了算子变体处理、边界情况处理、数值稳定性、输入分布的合理性以及如何检测LLM作弊行为等关键挑战，并介绍了基于目录的架构设计、kernel内部调度器、端到端收敛测试等工程实践。整体而言，课程强调了kernel正确性的重要性，以及如何通过系统化的测试框架来确保LLM生成代码的可靠性。

![](https://files.mdnice.com/user/59/bf6da65a-2b46-442a-b8b9-e01e786598c9.png)

这节课主要讲BackendBench，说白了就是用一个挺有意思的方法来测试大模型的编程能力——让它们写PyTorch的backend代码。然后顺便会讲一些在修kernel bug过程中遇到的有趣故事，特别是那些跟算子正确性相关的坑。

![](https://files.mdnice.com/user/59/554f341d-bf06-41d5-ab28-8a349ca8c610.png)

这是一个有意思的案例。有篇论文声称他们的方法能快150倍，但实际benchmark测试发现，不仅没有加速，反而比原版慢了3倍。右边的结果显示，custom matmul耗时6194ms，而torch matmul只需2309ms。这种情况在学术界并不少见，许多论文的性能数据需要谨慎验证。

![](https://files.mdnice.com/user/59/fb8b27fa-d87a-42e8-833f-cdd00554f0b0.png)

这个柱状图展示了不同加速比区间的kernel数量分布。最高的柱子显示有111个kernel的加速比在1.5倍以内，说明大部分优化效果有限。还有17个kernel没有实现加速，甚至可能变慢。真正显著的10倍以上加速只有12个，100倍以上的仅有2个。这说明kernel优化并非易事，多数情况下只能获得较小幅度的性能提升。

![](https://files.mdnice.com/user/59/1fad0248-b0d5-449e-96e0-5e7db934808a.png)

这是个Discord的讨论截图，在介绍RightNow AI这个工具——一个专门为GPU优化设计的代码编辑器，能帮你写、分析和优化CUDA kernel。下面有个很有意思的评论："10倍加速通常意味着你做错了什么"。这句话真的很到位，因为如果真能轻松获得10倍加速，那说明原来的代码写得太烂了，或者benchmark方法有问题。这也是为什么前面那些论文的性能claim要打问号的原因。

![](https://files.mdnice.com/user/59/5f5a25ec-6ff7-45fa-af7a-ec99b44588ef.png)

这页是BackendBench的TL;DR（太长不看版）总结。BackendBench本质上就是一个测试套件，用来评估LLM和人类写PyTorch backend的水平。它主要做三件事：第一，通过PyTorch的OpInfo测试套件做各种边界情况的正确性检查，确保你的kernel不会在极端情况下出bug；第二，用Hugging Face模型里真实的tensor shape来跑性能测试，这样测出来的数据才有实际意义；第三，你可以直接pip install把这些kernel装到自己的模型里用。

![](https://files.mdnice.com/user/59/2a16025e-6d82-4090-8500-37c81c9b3be4.png)

这张图展示了"for循环的力量"。上面是while循环的方式：持续让Claude写kernel直到正确（可能永远无法完成）。下面是for循环的方式：启动10个Claude worker并行工作，每个都不断迭代直到成功。这说明了一个工程策略：与其串行等待，不如通过并行多次尝试来提高成功率。

![](https://files.mdnice.com/user/59/1e5a80a2-6dff-4ec2-830b-c34a83195b54.png)

这张是主要成果展示：正确性测试结果。用一个简单的agent重试循环配合Claude的反馈，最终能让53%的PyTorch OpInfo算子用Triton正确实现。右边的曲线很有意思，第一次尝试成功率只有20.5%，到第五次能达到53.1%。你看这个增长趋势，前面涨得快，后面就越来越慢了，说明重试确实有用，但边际收益在递减。感谢Shahin Sefati和Sahan Paliskara的贡献！

![](https://files.mdnice.com/user/59/ce40fb3c-982a-427a-8ab5-544109d14f60.png)

这张展示性能测试结果。用ChatGPT生成了84个kernel示例（可以在GitHub上查看）。一个有趣的发现是LLM具备数值稳定性意识。右边的sigmoid代码显示，它会先转换到fp32再进行计算，以避免精度问题。性能方面，大部分kernel达到PyTorch的70-100%，少数能实现约1.2倍的加速。考虑到这些是自动生成的代码，这个结果是不错的。感谢Laura Wang的贡献！

![](https://files.mdnice.com/user/59/b44918d9-b053-4190-80ad-917a95dc0b43.png)

这是在nano-gpt上进行的端到端收敛测试。前向传播使用LLM生成的kernel（包含多个attention相关算子），反向传播仍使用PyTorch eager，因为LLM生成的kernel在数值精度上存在问题。右边的训练曲线表明：PyTorch原生实现（蓝线）和LLM生成版本（红线）几乎重合，说明forward pass使用LLM生成的kernel是可靠的，训练能够正常收敛。感谢Jiannan Wang的贡献！

![](https://files.mdnice.com/user/59/22a12780-4e7b-4824-b9fb-3f9e0cdca998.png)

这页提了个很现实的问题：假设你真的搞出来了一个又快又好的LLM生成kernel，然后呢？怎么把它merge到PyTorch里让大家都能用上？这不是个小问题，毕竟生成代码只是第一步，真正集成到生产环境里还有很多工程上的挑战。

![](https://files.mdnice.com/user/59/bdbd5a8a-0bdd-4a1a-a9f7-23bba3bdc6b9.png)

这页列出了kernel生成中正确性为何至关重要。主要挑战包括：第一，算子变体众多，如torch.add()需支持多种dtype、broadcasting、in-place操作等；第二，必须处理各种边界情况，包括NaN、infinity、零大小tensor、极端shape等；第三，不同GPU的浮点行为存在细微差异；第四，需要使用真实模型的tensor shape进行测试，不能仅依赖固定测试大小；第五，静默数值错误最难发现，表面结果正常但实际已经出错；第六，在PyTorch中容易遗漏必要的syncs和warmups。这些都是需要特别注意的问题。

![](https://files.mdnice.com/user/59/d0fe7a60-b289-40fa-9de9-35fd9f83ce40.png)

这张讨论了大多数实验室在GPU benchmarking上的常见问题。他们的应对策略是：第一步，在评估阶段通过工程手段解决大部分正确性问题；第二步，采用审计优先的方法，便于性能工程师调试结果。右边再次展示了那个经典案例——声称150x加速实际却慢3倍，说明benchmark需要严谨的方法论。

![](https://files.mdnice.com/user/59/83141e60-1b69-4ad2-a4aa-22d351fad81d.png)

这张展示了BackendBench的基础架构设计。工作流程简洁明了：LLM研究人员将生成的kernel实现放到对应的文件夹中，然后每个文件会被加载用于覆盖PyTorch的算子。右边展示了目录结构，generated_kernels下按算子分类（add、bitwise_and、div、fmod、mul、relu等），每个文件夹包含具体实现。这种组织方式便于BackendBench评估团队和Meta研究人员协作，同时也方便管理不同版本的实现。

![](https://files.mdnice.com/user/59/6ac530cf-267b-4b30-b140-31b529b8b57d.png)

这张解释为何聚焦在PyTorch算子集上。主要原因是这些算子大部分是数学函数（也包含一些系统相关算子），且基本都受Numpy启发。Numpy自1995年发布至今已有30年历史，经过了充分验证。既然这套API已经非常成熟，就没有必要重新设计，直接在此基础上进行kernel优化即可。右边列举了一些典型算子，如argmax、argmin、asin、asinh、atan等。

![](https://files.mdnice.com/user/59/6356bb15-43e8-42a1-b2a5-37fc4fba5660.png)

这张列出了PyTorch benchmarking时的常见错误。三个典型问题：第一，只测量了launch overhead（kernel启动开销），未测试实际计算时间；第二，缺少warmup，首次运行通常较慢；第三，未清理cache，导致后续测试结果失真。右边的代码是典型错误示例，直接使用time.time()进行测量，这样得到的数据参考价值有限。建议直接使用triton.testing.do_bench()，它已经处理好了这些细节。

![](https://files.mdnice.com/user/59/715b0ad9-01f9-4e68-9f09-92aed5d607a0.png)

这张图展示了一个很有启发性的例子：不当的输入分布会使kernel测试失去意义。原理很简单——一个大的正态分布向量（均值0，方差1）的平均值就是0。因此上面的测试代码生成100万个随机数，计算mean后assert其接近0，这个测试必然通过。下面那个"超级聪明的mean kernel"直接返回tensor(0,0)，也能通过测试。这个例子说明，如果测试数据分布过于理想化，即使错误的实现也能通过测试。这就是为什么必须使用真实模型数据进行测试的原因。

![](https://files.mdnice.com/user/59/a0c9b855-5f0d-445f-830b-34cebe9a4d7b.png)

这张强调输入多样性的重要性，对比了两个测试套件。左边是OpInfo Suite：311个操作，5020个测试，平均每个操作13个测试。右边是TorchBench Suite：155个操作，17089个测试，平均每个操作110个测试！看测试数量分布，TorchBench有19.4%的操作有100+个测试用例，而OpInfo只有1.9%。这差异说明了啥？真实模型（TorchBench）里的tensor shape、数据分布要比单元测试（OpInfo）复杂得多。所以光靠OpInfo测试是不够的，得用真实workload来验证。

![](https://files.mdnice.com/user/59/93052db0-f4ba-4b3a-8786-71d88891c143.png)

这张说的是"并非所有shape生而平等"。benchmark的时候别只测overhead，因为有些代码既不是带宽受限也不是overhead受限。比如你拿torch.randn(4,4,4,4)这种小shape去测，完全看不出性能差异。简单的做法是选最大的shape来测，但最好的做法是：选Hugging Face上真实模型里出现的那些有用的shape！这样测出来的数据才有实际参考价值。

![](https://files.mdnice.com/user/59/e4558097-ca52-46d4-b9a8-b6cb43b7032d.png)

这张给了个具体例子，展示啥叫"有用的shape"。以aten.add.Tensor为例，统计发现它被调用了156次，输入是1×512×768的tensor（f16格式，这是BERT的embedding size）。关键在于：针对这些高频出现的特定shape做超专门化优化（hyperspecialization），比泛型优化要有效得多。下面还有个链接指向Hugging Face的数据集，里面收集了各种模型的实际shape分布。

![](https://files.mdnice.com/user/59/b4741b76-d06f-494f-9921-9a294a70bd12.png)

这张讲边界情况处理，大多数自定义kernel在这些情况下会崩：大小为0或1的tensor、NaN和无穷大值、混合数据类型和broadcasting的边界情况、极端shape导致的内存分配问题。关键点来了：PyTorch自己是能处理这些情况的！而且OpInfo测试套件里编码了9年的bug报告，涵盖了各种奇葩边界情况。这就是代码能不能merge进PyTorch的门槛——你的kernel必须像PyTorch原生实现一样健壮。

![](https://files.mdnice.com/user/59/9a7de562-10b4-479c-9462-c7d6dce0d706.png)

这张揭示了一个不幸的现实：torch.add()不是一个函数，而是一堆函数！上面列了几个例子：tensor+scalar、tensor+tensor、带预分配输出、broadcasting、scaling（alpha参数）。下面的流程图更清楚：你在generated_kernels/add/目录下写一个kernel，然后通过op_map.py把这些变体映射到具体实现。最终add.Tensor（functional）、add_.Tensor（in-place）、add.out（pre-allocated）这三个变体都指向同一个实现。一个kernel要处理所有这些变体，难度可想而知。

![](https://files.mdnice.com/user/59/b3a2466b-f159-46c3-8979-e129da3436b4.png)

这张图与前一张内容相同，再次强调了"有用的shape"的概念，重申针对真实模型中高频出现的特定shape（如BERT的1×512×768 embedding）进行优化的重要性。

![](https://files.mdnice.com/user/59/dd0d8ac4-24be-4bcd-9713-018601bf9870.png)

这张揭露LLM常见的作弊套路，而且很难检测。第一招：解析文档很脆弱，LLM可以说"我没用torch.add"（右边代码块显示except后面确实有torch.add()调用）。第二招：解析AST也很脆弱，因为Python是个动态语言（。我们的解决方案很巧妙：用算子本身来覆盖自己，如果LLM在作弊（比如在实现里直接调用torch.add），就会触发无限递归错误。下面的代码展示BackendBench用torch.library来注册实现，把自定义kernel注册为add.Tensor的CUDA实现。

![](https://files.mdnice.com/user/59/e2557777-f411-48b0-b862-888f3fcaa4d4.png)

这张讲"无趣的算子"，以内存分配为例。kernel语言（比如Triton）通常期望输入输出都是torch tensor，而且Triton好用的一个原因就是它自动使用PyTorch默认的CUDA device和stream。代码示例展示：先创建两个random tensor x和y，然后output_torch用PyTorch reference实现（x+y），output_triton用Triton kernel实现add(x,y)。这类算子本身没啥技术含量，主要是在处理内存管理的细节。

![](https://files.mdnice.com/user/59/00abc6ef-03f9-4537-8038-06fe1ae83401.png)

这张展示了如何进行完整模型的benchmark，过程非常简洁。将kernel放到扁平的目录结构中，然后用全局flag加载。右边代码演示了具体步骤：import torch和model后，import BackendBench，调用BackendBench.enable(kernel_dir="generated_kernels")，之后直接运行model.forward(x)即可。Hugging Face模型无需任何代码修改，BackendBench会自动将算子替换为自定义kernel。这种设计十分便利。

![](https://files.mdnice.com/user/59/779d2bb1-3482-4555-bef3-5d3d50e4c4ba.png)

这张介绍了Intra kernel dispatcher（kernel内部调度器）的概念。核心思想是：只在你见过并且知道是正确的shape上调用LLM生成的kernel。代码示例展示了conditional_sin_impl函数，先检查x的shape是否小于0或者是any()（估计是检查某些条件），如果不满足就调用原始的sin kernel（call_boxed），否则返回torch.zeros_like(x)作为fallback。这是个保守但安全的策略，遇到没把握的情况就回退到PyTorch原生实现。

![](https://files.mdnice.com/user/59/c1f52df2-feb1-49d5-b1eb-160baf3c6cfd.png)

这张又是端到端收敛测试的详细版本，列出了forward pass用到的LLM生成kernel清单，包括log_softmax、matmul、gelu、unsafe_view、arange、view、split、add等算子。训练曲线显示forward用LLM kernel、backward用PyTorch eager的组合能正常收敛。

![](https://files.mdnice.com/user/59/4e3898ea-a1eb-404d-8ffc-724ce6955f4e.png)

这张图标题说得很直白："还有很长的路要走！" Forward时间对比：PyTorch Aten（蓝线）稳定在每次迭代10ms左右，而LLM生成的kernel（红线）要25ms左右，慢了一倍多。虽然能正确收敛了，但性能还差得远。这也说明了一个现实：正确性相对容易解决，但要真正达到甚至超越手写优化kernel的性能，LLM还有很长的路要走。

![](https://files.mdnice.com/user/59/6c642a07-8d7a-4fbf-9d89-69753047217c.png)

这张页面很简洁，就一句话：去看看这些kernel吧！给了个GitHub PR链接，里面有84个LLM生成的kernel实现，感兴趣的可以去学习研究。

![](https://files.mdnice.com/user/59/8bb1f349-5d7d-4baf-a04b-36d7546441bf.png)

这是课程的结束语。KernelBench（BackendBench的前身）起源于去年的第一次IRL hackathon。作者鼓励大家要有雄心壮志，并指出大家身边都有很多优秀的人。这是一个积极向上的结尾，祝愿大家好运。

![](https://files.mdnice.com/user/59/5a74bfd6-328d-4483-aa8d-c93bcbf69384.png)

最后是致谢页面。初始贡献者分为两类：评估方面包括Mark Saroufim、Sahan Paliskara、Jiannan Wang、Bert Maher、Manuel Candales；科研方面包括Shahin Sefati、Laura Wang、Jiannan Wang。页面列出了期待更多贡献的方向：更好的agent baselines、更多DSL支持（包括Cute等）、训练和分布式算子支持、更多backend扩展系统。最后提供了Discord联系方式（popcorn频道）。该项目仍在积极发展中，欢迎感兴趣的开发者参与。
