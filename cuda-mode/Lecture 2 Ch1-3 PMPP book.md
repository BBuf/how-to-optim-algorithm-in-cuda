> 我的课程笔记，欢迎关注：https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode 

## 第二课: PMPP 书的第1-3章

### PMPP 第一章

![](https://files.mdnice.com/user/59/6d74a444-8832-46a2-93db-02d09dac937a.png)

这一页没什么好说的，就是介绍一些大模型和AI的背景，CPU和GPU的架构区别，以及GPU的出现是为了解决CPU无法通过硬件技术解决的大规模计算性能问题。

![](https://files.mdnice.com/user/59/7b8b26cf-d8b2-42b0-9e04-d7415a92bc23.png)

这张图名为"The Power Wall"（功耗墙），展示了从1970年到2020年间计算机芯片技术的两个关键参数的发展趋势：

晶体管数量（紫色线）：以千为单位，呈指数增长趋势。
频率（绿色线）：以MHz为单位，显示了处理器时钟速度的变化。

"功耗墙"现象：图表底部的注释解释了为什么频率不再持续增长 —— "进一步提高频率会使芯片变得太热而无法有效散热"。

![](https://files.mdnice.com/user/59/e6b81392-0f5d-4209-82df-ad58811997a1.png)

这张slides介绍了CUDA的兴起及其关键特性：

- CUDA 专注于并行程序，适用于现代软件。
- GPU 的峰值 FLOPS 远高于多核 CPU。
- 主要原则是将工作分配给多个线程。
- GPU 注重大规模线程的执行吞吐量。
- 线程较少的程序在 GPU 上表现不佳。
- 将顺序部分放在 CPU 上，数值密集部分放在 GPU 上。
- CUDA 是 Compute Unified Device Architect（统一计算设备架构）。
- 在 CUDA 出现前，使用图形 API（如 OpenGL 或 Direct3D）进行计算。
- 由于 GPU 的广泛可用性，GPU 编程对开发者变得更具吸引力。

![](https://files.mdnice.com/user/59/fdf3c907-8e7e-4f79-a607-b11b06d39f49.png)

这张slides介绍了CUDA编程中的一些挑战：

- 如果不关注性能，并行编程很容易，但优化性能很难。
- 设计并行算法比设计顺序算法更困难，例如并行化递归计算需要非直观的思维方式（如前缀和）。
- 并行程序的速度通常受到内存延迟和吞吐量的限制（内存瓶颈，比如LLM推理的decode）。
- 并行程序的性能可能因输入数据的特性而显著变化。（比如LLM推理有不同长度的序列）。
- 并非所有应用都能轻松并行化，很多需要同步的地方会带来额外的开销（等待时间）。例如有数据依赖的情况。

![](https://files.mdnice.com/user/59/55ca7714-6dfa-4fc5-bd94-db7f92f198e1.png)

《Programming Massively Parallel Processors》这本书的三个主要目标是：

- 教大家并行编程和计算思维
- 并且以正确可靠的形式做到这一点，这包括debug和性能两方面
- 第三点指的应该是如何更好的组织书籍，让读者加深记忆之类的。

虽然这里以GPU作为例子，但这里介绍到的技术也适用于其它加速器。书中使用CUDA例子来介绍和事件相应的技术。


### PMPP 第二章

![](https://files.mdnice.com/user/59/4ff01aec-7940-44c1-ac86-9e8c392ae511.png)

题目是 CH2: 异构数据并行编程
- 异构（Heterogeneous）：结合使用CPU和GPU来进行计算，利用各自的优势来提高处理速度和效率。
- 数据并行性（Data parallelism）：通过将大任务分解为可以并行处理的小任务，实现数据的并行处理。这种方式可以显著提高处理大量数据时的效率。
- 应用示例：
    - 向量加法：这是并行计算中常见的例子，通过将向量的每个元素分别相加，可以并行处理，提高计算速度。
    - 将RGB图像转换为灰度图：这个过程通过应用一个核函数，根据每个像素的RGB值计算其灰度值。公式为 `L = r*0.21 + g*0.72 + b*0.07`，其中L代表亮度（Luminance）。这个转换是基于人眼对不同颜色的感光敏感度不同，其中绿色部分权重最高。

![](https://files.mdnice.com/user/59/9bad0b05-ba27-44e1-8109-03a10e9af79e.png)

这张Slides可以看到所有像素点的计算都是独立的。

![](https://files.mdnice.com/user/59/a5f9ceea-e103-42b4-9ab0-79e9a41e4505.png)

这张Slides介绍了CUDA C的一些特点：
- 扩展了ANSI C的语法,增加了少量的新的语法元素。
- 术语中,CPU表示主机,GPU表示设备。
- CUDA C源代码可以是主机代码和设备代码的混合。
- 设备代码函数称为内核(kernels)。
- 使用线程网格(grid of threads)来执行内核,多个线程并行运行。
- CPU和GPU代码可以并发执行(重叠)。
- 在GPU上可以大量启动多个线程,不需要担心。
- 对于输出张量的每一个元素启动一个线程是很正常的。


![](https://files.mdnice.com/user/59/0da939ef-4d56-4217-a5f5-66615a686802.png)

这张Slides