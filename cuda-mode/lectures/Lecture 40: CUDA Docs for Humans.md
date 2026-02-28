> 我的课程笔记，欢迎关注：https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode 。

## 课程总结

本节课介绍了Modal公司推出的GPU术语词汇表项目（modal.com/gpu-glossary） ，旨在为开发者提供人性化的CUDA文档。讲师Charles Frye分享了他从加州大学伯克利分校研究深度神经网络优化，到在Weights & Biases帮助人们实现研究项目产品化，再到现在在Modal帮助人们进行部署的职业历程。课程的核心观点是："CUDA"这个词实际上有多重含义——它既是软件平台，也是编程模型，还是计算机架构原则。然而，CUDA技术栈中最重要的部分其实并不叫"CUDA"，而是NVIDIA GPU Driver和其他底层组件。讲师强调了理解CUDA技术栈各个层次的重要性，包括Device Hardware（设备硬件）、Device Software（设备软件）和Host Software（主机软件）三大类。课程还讨论了Modal的短期目标（如ChatGPU、交互式代码片段、交互式图表等）、中期目标（性能调试、GPU集群管理、多GPU编程等）和推测性目标（多节点编程、Triton支持、开源材料、在线课程等）。整体而言，这是一个帮助开发者更好地理解和使用CUDA技术栈的教育项目。

![](https://files.mdnice.com/user/59/57b07838-5cba-485e-8423-16ab24085627.png)

这个网站看起来比较cool，然后我基于这个网站开源的markdown（在MIT协议下）使用硅动提供的DeepSeek-V3.2-EXP API用一个脚本批量一键翻译并修复了一下图片问题之后，使用github的workflow deoploy成了一个1:1复刻的中文网页，你可以通过 https://bbuf.github.io/gpu-glossary-zh/readme.html 这个网页访问到。

![](https://files.mdnice.com/user/59/e9dee2ca-7148-4087-a74d-e21e3e399604.png)

如果你发现了什么Typo也可以直接在 https://github.com/BBuf/gpu-glossary-zh 这个中文复刻网页对应的仓库做修改，这个仓库的脚本由Cluade 4生成，然后翻译脚本部分由硅动提供的DeepSeek-V3.2-EXP API生成。

## 课程内容


![](https://files.mdnice.com/user/59/a93aa18d-8aec-46c7-aa0c-bfbac55a9593.png)

这是课程的开场页面，展示了本次讲座的主题"CUDA Docs for Humans"（面向人类的CUDA文档）。页面简洁明了地呈现了Modal公司的logo和项目网址 modal.com/gpu-glossary 。这个项目的目标是创建一个更易于理解的CUDA文档资源，帮助开发者更好地掌握GPU编程技术。

![](https://files.mdnice.com/user/59/7b138cfd-bb77-46c1-8d7d-e594da5685af.png)

这张页面提出了四个基本问题："这是什么？（What is this?）"、"它从哪里来？（Where did it come from?）"、"它说了什么？（What does it say?）"、"它要去哪里？（Where is it going?）"。这四个问题构成了理解任何技术文档的基础框架，也是本次讲座将要展开讨论的核心思路。通过回答这些问题，可以帮助开发者建立对CUDA技术栈的全面认知。

![](https://files.mdnice.com/user/59/1bf0bd2a-e036-42d6-91eb-efff6318ddb1.png)

这张页面介绍了讲师Charles Frye的学术背景。他在加州大学伯克利分校（Cal）攻读博士学位时研究深度神经网络（DNN）的优化问题。页面展示了他的博士论文标题"Finding Critical and Gradient-Flat Points of Deep Neural Network Loss Functions"（寻找深度神经网络损失函数的临界点和梯度平坦点）。重要的是，这些研究工作是在CPU上使用autograd.numpy完成的，说明讲师的研究起点是在传统计算平台上进行的实证工作。

![](https://files.mdnice.com/user/59/240958d8-2ce7-49f4-8a24-8a7e59510e68.jpg)

这张页面介绍了讲师在Weights & Biases（W&B）的工作经历。他开始帮助人们将研究项目产品化（operationalize research）。左侧展示了他写的一篇关于"Public Dissection of a PyTorch Training Step"（公开剖析PyTorch训练步骤）的博客文章，配有经典的《杜尔普医生的解剖课》油画。右侧显示的是 wandb.me/trace-report ，展示了PyTorch Profiler的trace可视化界面，能够看到详细的stream、GPU利用率等性能信息。这个阶段讲师开始接触到实际的GPU性能分析工作。

![](https://files.mdnice.com/user/59/11ab5941-e040-4bc4-8c4a-51856013c26d.jpg)

这张页面展示了讲师目前在Modal的工作："现在在Modal，我帮助人们进行部署！"页面展示了Modal的博客界面，包含多个特色示例（Featured Examples），如部署开源LLM服务、使用Mistral的Pixtral进行图像和文本处理、用LLaMA进行语音聊天、使用Diffusion-3生成图像等。底部还展示了一篇博客文章"Beat GPT-4o at Python by searching with 100 dumb LLaMAs"。Modal提供了丰富的文档和示例（ modal.com/docs/examples ），帮助开发者快速部署AI应用。

![](https://files.mdnice.com/user/59/9a67c16b-1701-4475-8327-702fb84de26f.png)

这张页面只有一句简短的话："That involved a lot of environment debugging..."（这涉及大量的环境调试工作...）。这句话透露出在实际部署过程中，环境配置和调试是一个非常耗时且重要的环节。

![](https://files.mdnice.com/user/59/021756b2-02c8-4dd1-8c0a-7147f0d7edd7.png)

这张页面是个经典的吐槽："我他妈受够了，搞不懂CUDA技术栈"（I Am Fucking Done Not Understanding The CUDA Stack）。页面引用了一段话："CUDA开发环境依赖于与主机开发环境的紧密集成，包括主机编译器和C运行时库"——来自官方文档（sauce, from the horse's mouth）。下面的解释说明：如果不深入理解底层技术栈，就不可能开发出前沿的GPU应用（bleeding-edge applications）。因此，让我们一步步深入了解这个技术栈的各个层次。

![](https://files.mdnice.com/user/59/31d8730b-785e-4675-8610-081f2816b456.jpg)

这张页面展示了各种CUDA相关的书籍和文档，右下角大大的"RTFM."（Read The Fucking Manual，好好读文档）。左侧是经典的《Programming Massively Parallel Processors》教材和《Professional CUDA C Programming》等书籍。右侧展示了NVIDIA官方文档，包括CUDA C++ Programming Guide (Release 12.6)、PTX ISA (Release 8.5)、NVIDIA CUDA Compiler Driver (Release 12.6)等。这些文档和书籍构成了完整的CUDA学习资源，但它们的数量和复杂性也说明了掌握CUDA并非易事。

![](https://files.mdnice.com/user/59/436059c7-b535-4490-a09f-dd325c0e1ea9.png)

翻页，再次出现了之前的四个基本问题："这是什么？（What is this?）"、"它从哪里来？（Where did it come from?）"、"它说了什么？（What does it say?）"、"它要去哪里？（Where is it going?）"。

![](https://files.mdnice.com/user/59/21a5ce31-69d7-494e-aff3-a6c01a5e284c.png)

这张页面揭示了一个重要观点："并非只有一个'CUDA'"（There is not one "CUDA"）。页面将CUDA相关的术语和技术分为三大类：第一类是Device Hardware（/device-hardware），包含物理的NVIDIA硬件术语，如CUDA (Device Architecture)；第二类是Device Software（/device-software），包含在NVIDIA硬件上使用的术语和技术，如CUDA (Programming Model)；第三类是Host Software（/host-software），包含在运行GPU程序的CPU上使用的术语和技术，如CUDA (Software Platform)和CUDA C++ (programming language)。这个分类清晰地说明了"CUDA"这个词的多重含义。

![](https://files.mdnice.com/user/59/728ed613-ad50-48b2-93f3-5d08dd5abb74.png)

这张页面解释了"CUDA is a software platform"（CUDA是一个软件平台）。图示展示了从上到下的完整技术栈：最上层是Applications（应用程序），然后是CUDA Libraries（CUDA库）、CUDA Runtime API（CUDA运行时API）、CUDA Driver API（CUDA驱动API）和NVIDIA GPU Driver（NVIDIA GPU驱动），最底层是GPU硬件。CPU通过这个软件栈与GPU进行交互。整个架构清晰地展示了CUDA作为软件平台的层次结构。页面底部提供了详细的文档链接：https://modal.com/gpu-glossary/host-software/cuda-software-platform

![](https://files.mdnice.com/user/59/1b7bfebb-9a9c-484a-8e2e-eaa252e5629c.png)

这张页面解释了"CUDA is a programming model"（CUDA是一个编程模型）。左侧展示了编程模型的层次结构：从CUDA thread（线程）到CUDA thread block（线程块，包含Shared Memory共享内存），再到CUDA kernel grid（核函数网格，包含Global Memory全局内存）。右侧展示了硬件对应关系：CUDA Core对应Streaming Multiprocessor（流多处理器，包含L1 Data Cache），多个SM组成GPU（包含GPU RAM）。这个图清楚地说明了CUDA编程模型如何映射到实际的GPU硬件架构上。页面底部链接：https://modal.com/gpu-glossary/device-software/cuda-programming-model

![](https://files.mdnice.com/user/59/f7b320e4-1ca5-426e-b02f-8c171447d5b5.png)

这张页面解释了"CUDA is a computer architecture principle"（CUDA是一个计算机架构原则）。左侧展示了一个GPU架构图，上面画了一个大大的禁止符号，意思是"CUDA不是指具体的某一代GPU架构"。右侧展示了详细的GPU架构图，包括Host、Input Assembler、Warp Thread Issue、Execution units等多个组件。这说明CUDA作为一个架构原则，是独立于具体硬件实现的抽象概念。每一代GPU（如Volta、Ampere、Hopper等）都遵循CUDA架构原则，但具体实现细节各不相同。页面底部链接：https://modal.com/gpu-glossary/device-hardware/cuda-device-architecture

![](https://files.mdnice.com/user/59/61e2e326-b0be-4852-8d51-768e9012c290.png)

这张页面揭示了一个关键观点："CUDA技术栈中最重要的部分并不叫CUDA"（The most important part of the CUDA stack isn't called CUDA）。左侧展示了一个设备（Device）的架构图，包含多个Multiprocessor和Shared Memory等组件。中间展示了NVCC编译流程：x.cu源代码经过Stage 1 (PTX Generation)生成x.ptx中间代码，然后经过Stage 2 (Cubin Generation)生成x.cubin最终的GPU可执行代码。右侧展示了GPU的详细架构。关键信息是：虽然我们写CUDA代码，但真正执行程序的是Parallel Thread Execution（PTX）这样的底层指令集。页面底部链接：https://modal.com/gpu-glossary/device-software/parallel-thread-execution

![](https://files.mdnice.com/user/59/ae9415c5-5efd-48ea-944f-cc1cd7dbd858.png)

在介绍完CUDA的多重含义之后，这些问题现在指向项目的未来发展方向。

![](https://files.mdnice.com/user/59/fe04d40b-3751-40af-973a-44cd41c366a5.png)

这张页面列出了项目的"短期目标，敬请期待！"（Short-term goals. Watch this space!）。具体目标包括：第一，ChatGPU——如何让它尽可能简单和可扩展？提到了11ms.txt这个参考；第二，交互式代码片段——受Rust By Example等项目启发，需要Modal账户但会包含在免费层级；第三，交互式图表；第四，更好的同步内容——关于原子操作vs屏障（Atomics vs barriers）；第五，更好的warpgroups/thread block clusters内容。这些目标都是为了让GPU编程学习更加友好和实用。

![](https://files.mdnice.com/user/59/e3bb3a69-5f23-4373-a10e-b30e7876497e.png)

这张页面展示了"中期目标，正在寻找合作者，我们有GPU资源！"（Mid-term goals. Looking for collaborators. We have the GPUs.）具体包括：第一，性能调试（Performance debugging）——引入新术语如bank conflict（存储体冲突）、occupancy（占用率）、coarsening（粗粒化）；第二，GPU集群管理（GPU fleet mgmt）——引入新术语如dcgm、thermal design power（热设计功耗）；第三，多GPU硬件与编程（Multi-GPU hardware & programming）——引入新术语如PCIe、SXM、NVLink、NCCL。这些都是实际生产环境中非常重要的话题。

![](https://files.mdnice.com/user/59/1a51dfdf-d709-439a-ab3c-e885078bdc94.png)

这张页面列出了"推测性目标，我们能/应该做这个吗？"（Speculative goals. Can/should we do this?）包括：第一，多节点硬件与编程——引入新术语如NVLink Switch、NIC、Ethernet、TCP、IP、Infiniband；第二，Triton？——坦白说，我们在CUDA C++方面的经验都比在Triton方面多；第三，在GitHub上开源这些材料？——开源只有在它能够减少无差异化的重复劳动时才会成功；第四，在线课程？与大学合作？这些都是更长远的规划，但需要权衡投入产出比。

![](https://files.mdnice.com/user/59/f06e1dbd-8483-4d2a-aeda-3fd653de4feb.png)

这是课程的结束页面，展示了Modal的logo。页面底部写着"we're hiring btw :)"（顺便说一句，我们正在招聘）和联系邮箱charles@modal.com。




