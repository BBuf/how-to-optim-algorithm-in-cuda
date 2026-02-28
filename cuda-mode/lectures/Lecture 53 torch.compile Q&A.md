> 我的课程笔记，欢迎关注：https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode 。

## 课程总结

本节课是关于torch.compile的Q&A环节，主要讨论了torch.compile在实际使用中的常见性能陷阱、未来发展方向、自定义算子集成等核心话题。课程首先介绍了torch.compile的常见性能问题，包括图中断（graph breaks）和自定义算子的处理。在后端优化方面，讲解了如何使用"max-autotune"模式和"reduce-overhead"模式，以及重编译（recompilation）的相关问题。课程详细说明了如何将第三方kernel（包括Python/C++/CUDA kernel和Triton kernel）集成到torch.compile中，介绍了torch.library.custom_op和TORCH_LIBRARY两种注册方式。在自定义算子与PyTorch子系统的集成方面，展示了如何让自定义算子与torch.compile、autograd、vmap等系统协同工作。课程还讨论了torch.compile的优化能力，包括死代码消除、公共子表达式消除、算子融合等，并通过具体示例说明了哪些优化torch.compile可以自动完成，哪些需要手动实现。最后，课程介绍了torch.compile的设计架构（Dynamo、AOTDispatcher、Inductor等组件），以及如何通过TORCH_LOGS等工具进行调试和性能分析。整体而言，这是一个深入探讨torch.compile实践应用的技术交流环节。

## 课程内容

![](https://files.mdnice.com/user/59/b9444d22-0b75-46fe-80b7-f60b60e16ad4.png)

这张页面提出了一个核心问题："torch.compile常见的性能陷阱有哪些？"以及"如何编写容易被torch.compile优化的代码？"。页面分为前端和后端两部分建议。前端方面：图中断（graph breaks）和fullgraph=True的使用；已有自定义算子的代码——torch.compile可能能做得更好。页面还展示了一个使用@torch.compile(fullgraph=True)装饰的示例代码，其中包含一个函数f(x)和一个子图subgraph1(x)，演示了图中断的情况。这是理解torch.compile性能优化的起点。

![](https://files.mdnice.com/user/59/d0b0dc48-982b-46cd-8fc0-332e5c1bb7a9.png)

这张页面继续讨论性能优化建议。页面顶部提供了官方文档链接：https://pytorch.org/docs/stable/torch.compiler_troubleshooting.html#graph-break 。后端方面的建议包括：尝试mode="max-autotune"（自动调优 + CUDAGraphs）和mode="reduce-overhead"（仅CUDAGraphs）。关于重编译（Recompilations）问题，页面提供了另一个文档链接：https://pytorch.org/docs/stable/torch.compiler_troubleshooting.html#recompilation 。这些是torch.compile后端优化的关键配置选项。

![](https://files.mdnice.com/user/59/c1a86fcf-d7ad-459e-8ea1-f46623362dce.png)

这张页面讨论了"torch.compile的未来发展方向"。页面强调了torch.compile的价值主张：你可以花费数小时/数天/数周来调优自定义kernel，torch.compile提供了良好的基线性能，所以你不需要一直这样做！在路线图（On the roadmap）方面，包括：持续的性能改进，在新硬件上实现光速级的matmul性能，Blackwell性能优化；更好的"可攻击性和可理解性"（hackability and understandability），更好的错误消息，更好的escape hatches（如何绕过bug）；编译时间性能，"预编译"（Precompilation）：编译一次永不重编译；扩展到帮助其他库使用torch.compile，包括LLM推理服务（vLLM、SGLang）和图像生成改进。

![](https://files.mdnice.com/user/59/5d90f74c-ff3a-4a9f-b883-7b84c114f9f0.png)

这张页面回答了两个重要问题。第一个问题："将第三方kernel引入并与torch.compile一起使用的正确方法是什么？"提供了官方教程链接：https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html 以及PTC 2024的演讲"Extending PyTorch with Custom Operators"。第二个问题："将用户定义的triton kernel与torch.compile集成的正确方法是什么？"提供了教程链接：https://pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html 。这两个问题是实际应用中最常遇到的集成场景。

这里涉及到PTC的一个PPT，地址为 https://static.sched.com/hosted_files/pytorch2024/36/PTC%202024_%20Extending%20PyTorch%20with%20Custom%20Operators.pdf 内容截图如下：

![](https://files.mdnice.com/user/59/c4ba8a94-9a5a-4dfd-8706-032f90a8d5ee.png)

这是PyTorch Conference 2024的演讲封面，主题为"Extending PyTorch with Custom Operators"（使用自定义算子扩展PyTorch）。演讲者是Richard Zou (@zou3519)，来自Meta的PyTorch团队。这个演讲是关于如何将自定义算子集成到PyTorch生态系统中的权威指南。

![](https://files.mdnice.com/user/59/7c28835f-2c52-4bdd-9bdf-288326da7a5c.png)

这张页面定义了"Kernels"（核函数）的概念。定义：**kernel**是使用原始数据指针执行计算的函数。示例包括：C/C++/CUDA：CUDA kernels、CUTLASS；Python库：Pillow（用于图像处理）、NumPy；Triton kernels。这个定义建立了后续讨论的基础。

![](https://files.mdnice.com/user/59/662992c3-3df7-4799-913e-27b4b2603db0.png)

这张页面定义了"Operators"（算子）的概念。定义：**operator**是告诉PyTorch关于计算的粘合代码，并调用1个以上kernel来完成实际工作。使用**custom operators**（自定义算子）可以让自定义kernel与PyTorch子系统（如torch.compile、torch.export、autograd、vmap、Tensor子类等）进行组合。这是理解PyTorch扩展机制的关键。

![](https://files.mdnice.com/user/59/f1caa5ef-8d9b-4fbc-95fc-d9cb02ec6e4b.png)

这张页面展示了"Custom Operator Registration APIs"（自定义算子注册API）的对比表格。表格比较了三种kernel类型：C/C++/CUDA、Python和Triton。对于每种类型，表格说明了：是否需要算子注册才能与torch.compile一起工作；是否需要算子注册才能与其他PyTorch子系统一起工作；使用的算子注册API。C/C++/CUDA使用C++ TORCH_LIBRARY，Python使用torch.library.custom_op (PT2.4+)，Triton在PT2.3+不需要显式的torch.library包装器就可以与torch.compile工作，但需要torch.library.custom_op (PT2.4+)才能与其他子系统工作。

![](https://files.mdnice.com/user/59/7c92fb1f-6c9e-4475-8ac2-81c6d0d8b9a9.png)

这张页面展示了如何使用torch.library.custom_op。代码示例展示了一个简单的crop函数，它将图像转换为PIL格式，进行裁剪，然后转换回张量。这个例子演示了如何将Python库（如torchvision）中的功能包装成自定义算子。

![](https://files.mdnice.com/user/59/7c8d0fdd-357c-4b96-8c3d-7cb9c145a925.png)

这张页面展示了使用torch.library.custom_op的def+impl模式。代码使用@torch.library.custom_op装饰器定义了一个crop函数，并指定了mutates_args=()参数。这种方式允许将函数签名和实现分离，更符合PyTorch的算子注册模式。

![](https://files.mdnice.com/user/59/3303a0b6-5881-4cdb-b711-38edde29f6ce.png)

这张页面展示了如何让torch.library.custom_op与torch.compile一起工作。代码使用@crop.register_fake装饰器注册了一个fake实现，这个fake实现不执行实际计算，只返回输出张量的形状信息。这对于torch.compile的图捕获和优化至关重要，因为torch.compile需要在不实际执行的情况下推断张量形状。

![](https://files.mdnice.com/user/59/4af027ad-6b5a-44a7-82ad-0af847fe9148.png)

这张页面展示了如何让torch.library.custom_op与autograd（自动微分）一起工作。代码定义了backward函数来计算梯度，以及setup_context函数来保存前向传播需要的中间结果。最后使用crop.register_autograd注册这些函数。这使得自定义算子可以参与梯度计算，支持反向传播。

![](https://files.mdnice.com/user/59/b0f58207-1f4e-49c6-a85f-3ce3a47bef5d.png)

这张页面展示了如何使用TORCH_LIBRARY来注册C++/CUDA自定义算子。代码展示了两部分：首先使用TORCH_LIBRARY(mylib, m)宏定义算子的签名，然后实现crop_cpu函数。最后使用TORCH_LIBRARY_IMPL(mylib, CPU, m)将实现注册到CPU后端。这是C++端的算子注册模式，与Python的torch.library.custom_op相对应。

![](https://files.mdnice.com/user/59/23717141-d46d-4779-bb62-b97166a265cb.jpg)

这张页面展示了用户定义的triton kernels的代码示例。代码包含两个部分：上半部分使用@triton.jit装饰器定义了一个add_kernel，实现了元素级别的加法操作；下半部分使用@torch.compile(fullgraph=True)装饰器定义了一个add_fn函数，该函数调用add_kernel。这展示了如何将Triton kernel集成到torch.compile中。

![](https://files.mdnice.com/user/59/6b48d899-1f89-427d-a8cf-d7b204a24f87.png)

这张页面展示了"如何将triton kernels与PyTorch集成"的对比表格。表格比较了两种方式：triton kernel（无显式torch.library包装器，PT2.3+）和torch.library.custom_op (PT2.4+)。对于三个关键特性：支持eager-mode（两种都支持）；支持torch.compile（triton kernel在大多数情况下支持，custom_op在所有情况下支持）；支持Tensor子类、torch.vmap等（triton kernel不支持，custom_op支持）。这个表格帮助开发者选择合适的集成方式。

![](https://files.mdnice.com/user/59/5529fdd9-d49a-4a6b-872d-31cf5350d02d.png)

这张页面总结了"Takeaways"（要点）。对于非-triton的Python/C++/CUDA：优先编写operators而不是使用原始kernels，特别是如果你是库作者；使用torch.library.custom_op将kernels包装成Python中的operators。对于triton kernels：triton kernels可以开箱即用地与torch.compile一起工作，它们是一个可攻击（且高性能）的CUDA kernels替代方案。这是对整个自定义算子集成话题的精练总结。

![](https://files.mdnice.com/user/59/7132f07a-7b55-422a-86b6-a6b97dce2a36.png)

这张页面回答了"问：torch.compile做了哪些优化？"这个问题。页面提出了两个子问题：如何轻松检查torch.compile是否实际融合了两个操作？目前唯一的方法是使用profiler；请提供一个torch.compile今天不会执行kernel fusion的优化示例（尽管未来可能会），因此今天需要手动融合kernel。页面还列出了torch.compile可以做的优化：死代码消除、公共子表达式消除。下面展示了一个简单的Python代码示例。

![](https://files.mdnice.com/user/59/25d700a1-6f87-4d68-9da4-df83bb4d1c77.png)

这张页面继续展示了同一个问题的更完整代码示例。代码定义了一个函数f(x)，其中包含一些死代码（y = x.sin()计算了但没有使用）和公共子表达式（y = x.sin()被计算了两次）。torch.compile可以自动消除这些无效计算，优化成更高效的代码。

![](https://files.mdnice.com/user/59/124344ed-1827-4299-8f27-c0c8cc254f64.png)

这张页面继续讨论torch.compile的优化能力。页面提到了Min-cut partitioning（最小切分分区），并提供了相关讨论链接。还提到了Pattern matching（模式匹配）和Kernel fusion (pointwise, reductions)（算子融合，包括逐点操作和reduction操作）。可以通过TORCH_LOGS=output_code和TORCH_LOGS=fusion来查看生成的代码和融合信息。下面展示了一个简单的例子，展示了y = x.sin()和z = y.cos()可以融合成一个kernel。

![](https://files.mdnice.com/user/59/0773e526-8540-4a08-b188-9272c089f0a5.png)

这张页面展示了一个实际的代码示例。代码使用@torch.compile(fullgraph=True)装饰器，定义了一个函数f(x)，其中计算y = x.sin().cos()。然后创建一个随机张量并调用f(x)。这个例子展示了torch.compile如何将多个逐点操作融合成一个kernel。

![](https://files.mdnice.com/user/59/9092fc02-7dbe-4412-bb34-07efbeebee68.png)

这张页面展示了使用TORCH_LOGS=output_code命令查看生成代码的终端输出。输出显示了torch.compile生成的中间代码文件路径，并列出了目录中的文件，包拮epilogue_fusion.py和pointwise_fusion.py等。这些文件包含了torch.compile生成的优化后的代码，可以帮助开发者理解编译器的优化过程。

![](https://files.mdnice.com/user/59/a72bd3d0-0f55-4ee2-aa0a-10fac6944556.png)

这张页面展示了生成的Triton代码示例。代码定义了一个triton_poi_fused_cos_sin_0函数，这是torch.compile生成的融合kernel。代码展示了如何将sin和cos操作融合到一个Triton kernel中，包括内存加载、计算和存储操作。这展示了torch.compile如何将高层PyTorch代码编译成高效的Triton kernel。

![](https://files.mdnice.com/user/59/9f565a20-ed2c-442e-b59e-48a8ba999911.png)

这张页面展示了生成代码的后续部分，包括如何调用生成的Triton kernel。代码包含call函数，它设置了kernel的启动参数（grid和stream），并调用triton_poi_fused_cos_sin_0.run。还包括benchmark_compiled_module函数，用于性能测试。这展示了torch.compile生成的完整调用栈。

![](https://files.mdnice.com/user/59/b4adcbde-ec6e-4886-9ca1-a5b9e1b10ca2.png)

这张页面讨论了matmul epilogue fusion（矩阵乘法后续融合）。代码示例展示了如何强制使用Triton作为matmul的唯一后端，通过设置torch._inductor.config.max_autotune_gemm_backends = "TRITON"。然后使用@torch.compile(mode="max-autotune-no-cudagraphs")装饰器定义一个foo_with_epilogue函数，它执行mod(x).relu()。注释说明这将生成一个带有融合ReLU的matmul。页面还提到matmul prologue fusion（即将来推出）。

![](https://files.mdnice.com/user/59/0170459d-a6a7-44bd-84ec-c6a123919d8f.png)

这张页面展示了一个实际的matmul epilogue fusion代码示例。代码强制使用Triton作为matmul的后端，然后定义了两个被@torch.compile装饰的函数。第一个foo_with_epilogue在no_grad上下文中执行mod(x).relu()，第二个创建了一个简单的线性层和输入数据，然后调用foo_with_epilogue。这展示了如何将matmul和ReLU融合成一个kernel。

![](https://files.mdnice.com/user/59/4375c073-6529-4a44-9c91-0cc6657ee7fe.png)

这张页面展示了生成的matmul epilogue fusion代码的后续部分。代码展示了call函数如何设置kernel的启动参数，包括grid和stream，并调用triton_tem_fused_mm_relu_0.run。这个kernel名称说明它融合了mm（矩阵乘法）和relu操作。还包括benchmark_compiled_module函数用于性能测试。

![](https://files.mdnice.com/user/59/81a62493-e756-4665-af0c-f896e3594df9.png)

这张页面展示了生成的Triton kernel的详细实现。代码展示了如何在Triton中实现matmul和ReLU的融合，包括内存加载、计算和存储操作。代码中可以看到triton_helpers.maximum的调用，这就是ReLU操作的实现。这展示了torch.compile如何生成高效的融合kernel。

![](https://files.mdnice.com/user/59/0c6c9b84-423b-41ea-ab56-5417aa423c57.png)

这张页面讨论了matmul prologue fusion（矩阵乘法前序融合，即将推出）。代码示例展示了如何定义一个upcast_matmul函数，它在matmul之前将fp16数据转换为fp32。代码创建了两个fp16测试张量，然后运行示例。注释说明这个例子运行时融合日志会显示这个操作没有被融合。这展示了当前torch.compile的一个限制，即无法融合matmul之前的操作。

![](https://files.mdnice.com/user/59/d6f68bf2-464a-40b9-96cd-2e7cc4ac0d8f.png)

这张页面讨论了一般意义上的matmul选择。页面提到了两个方面：Autotuning（自动调优）和fusing into user-defined triton kernels (plausible in the longer term)（融合到用户定义的triton kernels中，长期来看是可行的）。代码示例展示了一个使用@torch.compile装饰的函数f(a, b)，它调用matmul_kernel[grid](a, b, c)。这展示了未来torch.compile可能会支持将matmul融合到用户定义的triton kernels中。

![](https://files.mdnice.com/user/59/fd035420-5f50-4d91-924a-336b89e359a4.png)

这张页面讨论了CUDAGraphs的使用。页面列出了两种模式：mode="reduce-overhead"和mode="max-autotune"。页面还回答了一个问题：如何检查torch.compile生成的图？可以使用以下方法：TORCH_LOGS=output_code、TORCH_LOGS=fusion、tlparse。这些工具可以帮助开发者理解和调试torch.compile的行为。

![](https://files.mdnice.com/user/59/c6f5705c-2358-4c56-8d86-1c9b2f671227.png)

这张页面展示了使用tlparse工具的终端输出。命令行显示了如何使用TORCH_TRACE环境变量生成trace日志，然后使用tlparse工具解析这些日志。输出显示了编译统计信息，包括成功编译的数量、失败的数量等。这是一个强大的调试工具，可以帮助开发者诊断编译问题。

![](https://files.mdnice.com/user/59/17374a4c-b76e-4b18-8f68-73b3c8f026f7.png)

这张页面展示了torch.compile的详细调试信息。页面包含Stack trie（堆栈树）、IR dumps（IR转储）和Chromium Events（Chromium事件）三个部分。Stack trie展示了编译过程中的调用栈信息；IR dumps说明了如何查看中间表示（IR），包括Dynamo输出图和PTX代码；Chromium Events说明了如何生成和查看性能分析数据。这些是深入理解torch.compile内部工作原理的关键工具。

![](https://files.mdnice.com/user/59/c9de0fe1-7176-43e3-93e8-3d4712ddf0ff.png)

这张页面回答了"能否给出torch.compile设计的概述？"这个问题。页面推荐了一些学习资源，从机器学习编译器的角度：https://pytorch.org/blog/pytorch-2-paper-tutorial/ 和 https://github.com/pytorch/workshops/tree/master/ASPLOS_2024 。页面还列出了torch.compile的主要组件：Inference、Dynamo、AOTDispatcher (aka AOTAutograd)、Inductor，以及相关的TORCH_LOGS环境变量设置。Training部分也有类似的组件。这为理解torch.compile的整体架构提供了路线图。

![](https://files.mdnice.com/user/59/23b4b41e-0bc7-44dd-9f2b-544642336b8c.png)

这张页面回答了两个问题。第一个问题：如何缓解或甚至解决torch.compile与eager模式之间的浮点数不匹配问题？这在量化过程中变得更加明显。遇到过fp8优化器的变体问题。解决方法是emulate_precision_casts，并提供了GitHub链接。另外，也可以torch.compile模型的更少部分，直到问题消失并提交issue。第二个问题：如何参与编写编译改进或kernels？提供了贡献指南和PyTorch贡献终极指南的链接。这两个问题解决了实际应用中的常见问题和如何为社区做贡献。

## 文档总结

本文档是一个关于torch.compile的全面Q&A集合，涵盖了从基础使用到高级优化的各个方面。文档首先讨论了torch.compile的常见性能陷阱，包括图中断问题和后端优化策略，并介绍了未来的发展路线图。在自定义算子集成方面，文档详细介绍了如何使用torch.library.custom_op和TORCH_LIBRARY来注册Python/C++/CUDA kernel，以及如何将Triton kernel集成到PyTorch中。文档还展示了如何让自定义算子与torch.compile、autograd等子系统协同工作。

在优化能力方面，文档详细说明了torch.compile可以自动完成的优化，包括死代码消除、公共子表达式消除、逐点操作融合和matmul epilogue fusion等。通过具体的代码示例和生成的Triton kernel代码，展示了torch.compile如何将高层PyTorch代码编译成高效的GPU kernel。文档还指出了当前的一些限制，比如matmul prologue fusion尚未支持，需要手动优化。

文档还介绍了丰富的调试和性能分析工具，包括TORCH_LOGS环境变量、tlparse工具、Stack trie、IR dumps和Chromium Events等。这些工具可以帮助开发者深入理解torch.compile的内部工作原理，诊断性能问题。最后，文档还介绍了torch.compile的整体架构，包括Dynamo、AOTDispatcher、Inductor等组件，并提供了学习资源和贡献指南。

总的来说，这是一份非常实用的torch.compile实践指南，既包含了如何使用torch.compile获得更好性能的建议，也包含了如何扩展torch.compile功能的技术细节，还提供了丰富的调试和性能分析工具。对于想要深入理解和使用torch.compile的开发者来说，这是一份很好的参考文档。

