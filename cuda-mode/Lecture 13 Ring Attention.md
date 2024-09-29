> 我的课程笔记，欢迎关注：https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode 

# 第13课，Ring Attention

## 课程笔记

![](https://files.mdnice.com/user/59/e607565c-2755-4e35-bd3f-6c2880314c6b.png)

![](https://files.mdnice.com/user/59/4b0ce7bb-f5ee-4907-b9ac-11bf6e77f830.png)

Overview部分主要介绍了长上下文Transformer模型及其相关应用的几个主题。具体内容如下：

- 动机：长上下文Transformer模型及其应用
- 回顾：普通注意力机制、在线Softmax、对数和指数计算
- Ring Attention
- Striped Attention
- Flash Decoding

![](https://files.mdnice.com/user/59/5c5c3df9-4010-4d5f-aeaa-0f2f243d7c50.png)

这张slides展示了一下当前流行的LLM的上下文长度越来越长，特别的Gemini 1.5 Pro将上下文长度扩展到了1M。

![](https://files.mdnice.com/user/59/d929063d-f7cc-4d7e-ae83-d8d100998dc9.png)

这张Slides介绍了长上下文模型（Long-context Magic）的能力和应用，主要内容如下：

- 左侧展示了一个视频时间轴，从00:00:00到00:59:59，表明模型可以处理长达1小时的视频内容。
- 中间部分展示了一个问答示例：
  - 用户问："人的车里有多少个柠檬？"
  - 几个不同的AI模型（GPT-4V, Gemini Pro Vision, Video-LLaVA）都无法正确回答这个问题。
  - 而LWM（Large World Model，作者的模型）正确回答："车里有三个柠檬。"
- 右侧列出了长上下文模型可以处理的内容类型：
  - 书籍
  - 长文档
  - 网页内容
  - 聊天历史
  - 代码库
  - 高分辨率图像
  - 音频记录
  - 视频
- Slides底部强调这些能力朝着多模态世界模型发展，并提供了更多关于LWM的信息链接。

![](https://files.mdnice.com/user/59/168b814b-4ecd-4c89-bdae-eaababbfe860.png)

这张Slides介绍了多模态任意到任意（Any-to-Any）自回归预测模型，主要对比了两种模型：LWM (Large World Model) 和 LLaVA。以下是主要内容：

- 标题强调了**多模态**的任意到任意的自回归预测特性。
- LWM (Large World Model) 部分:
  - 使用自回归Transformer结构
  - 可以处理文本、图像、视频等多种模态
  - 输入包括图像tokens（使用VQGAN编码）和文本tokens（使用BPE分词器）
  - 能够进行多种模态间的转换，如文本到图像、图像到文本等
- LLaVA 模型部分:
  - 主要用于图像到文本的任务
  - 使用视觉编码器处理图像输入
  - 使用语言模型处理文本输入和生成输出
- Transformer编码器结构:
  - 右侧展示了Transformer编码器的详细结构
  - 包括多头注意力机制、规范化层、MLP等组件
  - 底部有嵌入块（Embedded Patches）作为输入
- 示例输入:
  - 展示了三张小狗在草地上奔跑的图片
  - 文本描述"A puppy running on a grassy lawn"

这里是对多模态做一个Overview，然后这里的关键点是要支持多模态我们就要求模型可以处理很长的上下文，这就要求模型需要使用Ring Attention或者类似的长上下文训练技术。

![](https://files.mdnice.com/user/59/c84d822b-e58d-48c8-b1bb-4d0786b38e3d.png)

- 这张Slides讨论了处理大规模语言模型时面临的内存挑战。主要内容如下：
  - 标题："挑战：我们耗尽了内存"
  - 引用自Ring Attention（2023年，Hao Liu等人的研究）：
    - "对于一个隐藏层大小为1024的简单模型，使用批量大小为1处理1亿个token需要超过1000 GB的内存。"
- 内存挑战的原因：
  - 输入需要被具体化（materialized）
  - 使用Flash-Attention时，内存需求与输入长度呈线性增长
  - 需要存储输入的QKV（查询、键、值）、输出、LSE（对数和指数）以及用于反向传播的dout
- 当前高端GPU的内存容量：
  - NVIDIA H200: 141 GB
  - AMD MI300X: 192 GB
  - NVIDIA GB200 (Blackwell): 288 GB（将于2024年底推出）

![](https://files.mdnice.com/user/59/c3a118ee-c9d1-4fce-bce8-bf9d6bbb6ddf.jpg)

这张Slides讨论了处理长上下文（Long Context）时的注意力机制方法，同时以幽默的方式呈现了相关挑战。主要内容如下：

标题：**长上下文的注意力方法**

列出了三种主要方法：
1. 近似法（如Sparse注意力、LoRA）
2. RAG / 向量数据库（使用ANN搜索、LSH）
3. 暴力计算（如tiling、blockwise方法）

图片和梗：
- 左下角：
  - 一张笑脸图片（可能是某位计算机科学家）
  - 旁边是一排服务器的图片
  - 配文："哈哈 GPUs go bitterrr"（GPUs变得苦涩）
  - 暗示：使用暴力计算方法会导致GPU负担过重
- 右侧：四格漫画梗图
  1. "我们的LLM有100万token的上下文窗口"
  2. "哦，所以你解决了二次方扩展问题？"
  3. （男孩表情困惑）
  4. "你确实解决了二次方扩展问题，对吧？"

这个梗图幽默地指出，虽然有些模型声称能处理非常长的上下文，但可能并没有真正解决计算复杂度随上下文长度呈二次方增长的问题。

![](https://files.mdnice.com/user/59/c5458d3b-9ddb-46d2-8826-9f795d6bccdd.png)

这张Slides介绍了"Vanilla Attention"（原始注意力机制）的基本概念和其内存复杂度问题。主要内容如下：
- 标题：Vanilla Attention（原始注意力机制）
- 注意力机制的数学表达：
  - 左侧给出了图形化的表达式：softmax(Q × K^T) × V
  - 右侧给出了对应的数学符号表示：softmax(QK^T)V
- 注意力矩阵（Attn）的表示：
  - 用一个绿色方框表示，大小为 s × s
  - s 代表序列长度
- 内存复杂度说明：
  - 原文："Memory complexity of naive attention is quadratic with sequence length (score matrix & softmax output)."
  - 翻译：朴素注意力机制的内存复杂度与序列长度呈二次方关系（score矩阵和softmax输出）。

![](https://files.mdnice.com/user/59/36438bb7-4f0c-48c9-82cf-df7536a281ad.png)

这张Slides讨论了模型大小和上下文长度对每个token的FLOPS（浮点运算次数）缩放的影响。主要内容如下：
- 标题：情况有多糟？每个Token的FLOPS缩放
- 热力图：
  - 横轴：上下文长度（从2x到32768x）
  - 纵轴：模型大小（从7B到1TB）
  - 数值：每个单元格中的数字表示相对于4k上下文大小的FLOPS成本比率
- 关键发现：
    - "令人惊讶的是：**随着模型大小的增加，成本比率反而降低**"
- FLOPS计算公式：
  - FLOPS = 24sh² + 4s²h
  - （s=序列长度，h=隐藏维度）
  - 当h为常数时，复杂度为O(s²)
- 结论：
  - "序列长度最终会成为瓶颈 - 但可能比你想象的要晚"

来源：Ring Attention，附录D。上面的公式是针对FFN，这里bs=1。具体的公式推导看下图：

![](https://files.mdnice.com/user/59/6977caea-3f53-47e4-8fc5-3d1bda0eebdb.png)


![](https://files.mdnice.com/user/59/edfa99c5-6303-4144-be24-b50376518f50.png)


这张slides描述了计算Softmax的挑战。Softmax操作需要在分数矩阵（score matrix）的完整行上进行，这个分数矩阵是通过$S=QK^T$(Q是Query矩阵，K是Key矩阵的转置）计算得到的。Softmax的输出依赖于分母中的和，也就是所有输入值指数和的计算。为了在FlashAttention和RingAttention算法中应用Softmax，必须“分块”或“在线”地计算Softmax，即只处理部分和，这样可以更高效地计算出结果。

![](https://files.mdnice.com/user/59/4411dcee-ecde-45b0-85a0-be8568f467b2.png)

这张Slides开始介绍如何通过Python中的PyTorch库定义和验证一个简单的Softmax函数，并逐步过渡到Log-Sum-Exp的更新。这里展示了如何用Python代码定义一个朴素的Softmax函数。这个函数接受一个PyTorch张量作为输入，并计算Softmax值。接下来，展示了如何将自定义的Softmax函数与官方的PyTorch `torch.softmax()`函数进行比较。通过生成一个随机张量`x`，分别计算官方Softmax结果`a`和自定义版本`b`。使用`torch.allclose()`函数验证两个输出是否接近。

![](https://files.mdnice.com/user/59/8e4d83df-36bd-4577-8d2c-ee62f44ebe05.png)

slides标题提到“Naive & Numerical unstable”（朴素且数值不稳定），表示当前定义的朴素Softmax函数在某些输入情况下会出现问题。slides显示了一个具体的例子，代码使用了一个随机生成的PyTorch张量x，并将其乘以100传入到朴素的naive_softmax()函数中。结果输出中显示张量中的某些值变成了nan（Not a Number），这表明数值溢出或不稳定。

![](https://files.mdnice.com/user/59/c8dd01bc-a0be-4406-b89a-e4b23c7e1733.png)

我们的目标是将Softmax运算分块处理（breaking softmax() into chunks）。右侧文字指出，虽然可以将向量分块并分别计算Softmax，但最终问题是如何从分块结果`s1`和`s2`重构出完整的target结果。这也是下一步需要解决的核心问题。

![](https://files.mdnice.com/user/59/3b53c3f7-d99a-456c-8c11-89027b131588.png)

这张幻slides讲解了如何通过“sum exp”（指数和）撤销Softmax的归一化，从而将分块计算的结果合并。首先回顾了上一个slides中的问题：Softmax输出通过除以`x.exp().sum()`来归一化。为了将多个分块的结果合并，我们需要撤销这种归一化。

Slides右侧的代码显示了如何通过分块的指数和来进行修正。`x1.exp().sum()`和`x2.exp().sum()`分别计算两个分块`x1`和`x2`的指数和，命名为`se_x1`和`se_x2`。然后，分别对分块结果`s1`和`s2`进行修正，计算公式如slides的代码所示。修正后的结果使用`torch.cat()`函数合并，得到完整的Softmax结果。

合并后的结果与目标结果`target`进行了比较，并通过`torch.allclose()`函数验证，结果为`True`，表明通过这种方式成功合并了分块的Softmax计算结果。

> 然而这种方法仍然需要访问所有的数值，但稍安勿躁，继续深入。

![](https://files.mdnice.com/user/59/31d53c35-9fe8-447b-a484-3b242a45a630.png)


这张Slides讲解了如何使用数值稳定的方式将分块的Softmax结果进行合并。具体内容如下：
- 标题为“Combining blocks numerically stable”，说明Slides的重点是如何在数值上稳定地合并分块计算的Softmax结果。
- 左上角的代码显示了一个测试张量`x`，其包含20个元素，并计算了它的完整Softmax结果`a`，同时将`x`分成了两个分块`x1`和`x2`。
- 右上角代码定义了一个名为`stable_softmax2(x)`的函数。该函数通过以下步骤来实现数值稳定的Softmax计算：
  - `m = x.max()`：计算输入向量的最大值`m`。
  - `a = (x - m).exp()`：将输入向量减去最大值后再做指数运算，防止因数值过大而导致的溢出问题。
  - `b = a.sum()`：计算指数运算的和。
  - `lse = m + torch.log(b)`：计算“log-sum-exp”（LSE）值。
  - 返回Softmax结果`a/b`以及LSE值。
代码中演示了如何基于LSE值来稳定地合并分块结果。传统的合并方式会使用指数运算，如下：
```python
c1 = b1 * torch.exp(lse1) / (torch.exp(lse1) + torch.exp(lse2))
c2 = b2 * torch.exp(lse2) / (torch.exp(lse1) + torch.exp(lse2))
```
为了避免指数运算带来的不稳定性，代码使用了除法转换为减法的技巧来合并结果：
```python
c1 = b1 / (1 + torch.exp(lse2 - lse1))
c2 = b2 / (1 + torch.exp(lse1 - lse2))
```
- 这样做的好处是使用对数操作来减少数值溢出，提高稳定性。
- 合并后的结果b与完整计算得到的结果`a`进行比较，使用`torch.allclose()`函数验证，结果为`True`，表示数值稳定的分块合并策略成功达到了与整体计算一致的结果。
- 旁边解释了一个数学技巧：$\frac{a}{a+b}=\frac{1}{1+\frac{b}{a}}$。
提到要在对数尺度上进行减法而非除法，从而保证数值稳定性。

![](https://files.mdnice.com/user/59/4128df2d-6d10-402a-98ab-be9ca3c3f463.png)

- 这里提到 RingAttention 可以使用内部 Flash Attention 的一些函数，这些函数可以返回 log-sum-exp，从而帮助进行逐块或者增量地计算注意力Value的投影。
- 这里的代码片段是一个名为 `_update_out_and_lse` 的 PyTorch 函数。它的作用是更新 `out` 和 `lse`（log-sum-exp）的值。由于注意力Value投影是线性的，所以可以按照类似直接对 Softmax 结果进行分块处理的方式进行修正和计算。

![](https://files.mdnice.com/user/59/151c556c-3421-467a-a2e2-d005b4467427.png)

这张图展示的是Flash Attention V2的逐chunk更新softmax结果和输出，实际上也适用于这里的Ring Attention的更新。

![](https://files.mdnice.com/user/59/e880c3ad-3287-4cd8-b141-643f1f3ceaf1.png)

这张slides展示了zhuzilin/ring-flash-attention中对Ring Attention的开源实现，我没可以看到除了通信之外Ring Attention调用的是TriDao的Flash Attention来做每个块（设备）上的Attention计算和lse的更新。实际上这就是Ring Attention的细节了，接下来作者会继续讨论下Ring Attention的通信之类的。


![](https://files.mdnice.com/user/59/303c3199-3225-4e0c-b5b4-ab381b72f1eb.png)

这张Slides画了一下序列并行的示意图，这个就不多讲了，大家应该比较熟悉。

![](https://files.mdnice.com/user/59/d5aa4824-eae1-42aa-9fd7-51c4c5a447e5.png)

这张Slides介绍了 注意力机制的序列并行化（Sequence Parallelism），展示了如何将查询（Q）、键（K）和值（V）张量在不同设备间进行分割和传递。每个设备分别计算一部分注意力值，并通过 `Send & Recv KV` 操作在设备间进行通信，从而实现跨设备的高效并行计算。

![](https://files.mdnice.com/user/59/b70b5a76-b1da-4c0d-aebd-b74a0d49800e.png)

这里介绍了一下"Ring Attention"的主要概念。内容包括：
- **计算顺序的灵活性**：块计算的顺序可以是任意的，不受限制。
- **QKV序列的分割**：将QKV（查询、键和值）序列分割成N个不同的主机进行处理。
- **主机环状结构**：这些主机形成一个概念上的环，用于交换KV（键和值）段。
- **完成条件**：当每个节点都看到所有KV部分时，一个完整的循环就完成了。
- **零开销**：对于较长的序列，由于计算和通信可以重叠，因此实现了零开销。

![](https://files.mdnice.com/user/59/0627acad-eb05-47b8-b5ce-976d6215059d.png)

这里展示了一下Ring Attention的伪代码和前面2个slides的代码是相对应的。

![](https://files.mdnice.com/user/59/a7d4d710-cfee-4fcb-b119-0b02b7cb3e0a.png)

这张Slides回顾了自回归模型（Autoregressive Models）中的因果掩码（Causal Masking）的概念和作用，内容包括：
- 因果掩码是支持自回归解码所必需的，因为在自回归模型中，每个时刻的输出只能依赖当前及之前的输入，而不能看到未来的输入。
- 注意力得分的计算变为：`dot(Q_i, K_j) if i <= j else -inf`。即，如果当前查询$Q_i$的索引$i$小于等于键$K_j$的索引$j$，则正常计算点积；否则，得分为负无穷大（-inf），使得该位置在softmax输出中为零（不会被关注）。
- 掩码无需被显式存储，而是可以在内核（kernel）中动态计算。
- 类似于Flash Attention的kernel可以跳过完全被掩码的键值块，从而提升计算效率。

![](https://files.mdnice.com/user/59/55444448-2d20-4ae0-a8a0-bd339ae3ad2b.png)

这张Slides描述了自回归模型中使用Ring Attention时遇到的主要问题及其影响。
- 设备空闲问题：
    - 当使用因果掩码（Causal Masking）时，在环形结构中某些设备会处于空闲状态。这种情况在所有自回归模型（例如语言模型）中非常常见。
    - 由于因果掩码的存在，当查询索引（Query_index）小于键索引（Key_index）时，输出会被掩盖（置为0），导致某些设备在计算时没有实际有效的输出，因此在等待其他设备时处于空闲状态。
- 解决方案：
    - 使用Ring Attention的环形结构，可以动态地跳过完全被掩码的键值块，从而提升计算效率。
    - 通过这种方式，可以减少计算资源的浪费，提高计算效率。

![](https://files.mdnice.com/user/59/55444448-2d20-4ae0-a8a0-bd339ae3ad2b.png)

这张Slides描述了自回归模型中使用Ring Attention时遇到的主要问题及其影响。
- 设备空闲问题：
    - 当使用因果掩码（Causal Masking）时，在环形结构中某些设备会处于空闲状态。这种情况在所有自回归模型（例如语言模型）中非常常见。
    - 由于因果掩码的存在，当查询索引（Query_index）小于键索引（Key_index）时，输出会被掩盖（置为0），导致某些设备在计算时没有实际有效的输出，因此在等待其他设备时处于空闲状态。
- 解决方案：
    - 使用Ring Attention的环形结构，可以动态地跳过完全被掩码的键值块，从而提升计算效率。
    - 通过这种方式，可以减少计算资源的浪费，提高计算效率。

![](https://files.mdnice.com/user/59/55444448-2d20-4ae0-a8a0-bd339ae3ad2b.png)

这张Slides描述了自回归模型中使用Ring Attention时遇到的主要问题及其影响。
- **设备空闲问题**：
    - 当使用因果掩码（Causal Masking）时，在环形结构中某些设备会处于空闲状态。这种情况在所有自回归模型（例如语言模型）中非常常见。
    - 由于因果掩码的存在，当查询索引（Query_index）小于键索引（Key_index）时，输出会被掩盖（置为0），导致某些设备在计算时没有实际有效的输出，因此在等待其他设备时处于空闲状态。
- **逐轮处理的过程演示**：
    - 该图将Ring Attention过程分为了四个回合（Round 0到Round 3），每个回合中，每个设备（如GPU）负责不同的KV（键-值）块和Q（查询）块。
    - 每个回合中，设备根据查询和键的索引关系计算输出，当掩码的值为0时（黑色格子表示被掩盖的位置），输出被强制为0。
    - 图中可以看到，随着回合的推进，有些设备的计算结果被掩盖（黑色区域增多），导致设备无法参与有效计算。
- **最慢的环形节点决定整体速度**：
    - Slides 特别指出：环形结构中最慢的主机（Ring Host）决定了整体计算的速度。因此，如果某个设备因掩码导致计算时间变长或空闲时间变多，会拖慢整体环形的计算速度，降低效率。

![](https://files.mdnice.com/user/59/25eca3d3-6b90-43c3-bafa-8da15355795f.png)

这张Slides在上面的Slides基础上进一步详细说明了Ring Attention在自回归模型中应用因果掩码时的具体过程和问题。
- **Causal Mask Chunks 分割及其应用**：
    - 左侧的图例显示了一个因果掩码矩阵，将其分割成多个块（例如，A、B、C等），这些块在Ring Attention的不同回合（Rounds）中进行应用。
    - 矩阵的每个块表示查询 Q 和键 K 之间的掩码关系。灰色表示有效计算，黑色表示被掩盖（mask）的位置。
- **分块应用的过程**：
    - 通过将因果掩码矩阵划分为多个小块，这些小块分别被分配到每一轮Ring Attention中进行计算。
    - 每一轮Ring Attention（Round 0 到 Round 3）对应右侧图中的不同计算顺序。可以看到，每一轮Ring Attention中，每个设备分别计算不同的KV段，并根据分块掩码进行计算。
    - 每个回合的底部显示了当前回合的因果掩码矩阵的分块（例如，A、F、K、P等），这些分块对应矩阵的不同部分，并标记出了在当前回合中被应用的掩码块。
- **各个回合的掩码关系**：
    - Round 0 应用的掩码块为：A、F、K、P。
    - Round 1 应用的掩码块为：D、E、J、O。
    - Round 2 应用的掩码块为：C、H、I、N。
    - Round 3 应用的掩码块为：B、G、L、M。
    - 每个回合通过不同的掩码块，可以逐步形成整体的因果掩码矩阵。每个掩码块仅在其对应的回合中参与计算，从而保证了自回归解码的因果性。
- **掩码应用的顺序**：
    - 不同颜色和字母标记的掩码块显示了Ring Attention在多个回合中如何分布和应用掩码。通过这种方式，每个设备能够在不同回合中处理不同的KV块和Q块，从而覆盖整个因果掩码矩阵。

> 上面讲的都是Ring Attention的负载不均衡问题，接下来介绍个解决方案。

![](https://files.mdnice.com/user/59/dbe00625-afd7-4741-b345-32b45b783e8c.png)

![](https://files.mdnice.com/user/59/f2b23290-ca73-4c39-bc20-bfde8ccb2310.png)

这两张slides讲解了一个Ring Attention负载不均衡的解决方案，通过 **Stripe Permutation（条带置换）** 的策略，将K，V和Q在序列维度上按条带重新排列（比如将KV0分成了0,4,8,12，而不是连续的0,1,2,3），通过重新排列KV和Q块，Striped Attention能够更好地分配计算资源，从而减轻设备之间的不平衡性，提高整体计算效率。从第二张Slides可以看到，经过条带置换后的计算过程几乎能够完美地均衡分配计算负载，从而使得设备之间的计算更加平衡，避免了Ring Attention中存在的设备空闲问题。在每个回合中，只有当“host_id < round”时，需要丢弃第一个查询和最后一个键的计算，这样做能够避免不必要的计算，进一步提升效率。

![](https://files.mdnice.com/user/59/8d056f6b-6d46-43e0-8237-2aab602b9cf4.png)

![](https://files.mdnice.com/user/59/ebff1ff8-f32c-4d0b-acbb-2053278fc5aa.png)

这两张slides则讲述了 FlashAttention 和 Flash-Decoding 两种不同的方法在长文本推理任务中的表现差异。
- FlashAttention 在长文本推理中表现不佳。
    - FlashAttention 只能对查询（queries）的块和批量大小（batch size）进行并行化处理，这在逐字（token-by-token）解码过程中无法充分利用整个 GPU 的计算资源。
    - 第一张Slides下方的图示展示了 Queries、Values 和 Keys 在 FlashAttention 中的处理方式。图中显示了 Queries、Values 和 Keys 是分块处理的，每个块的大小和位置是固定的，这种处理方式无法做到高效的并行解码。
- Flash-Decoding
    - Flash-Decoding 通过将 Queries、Values 和 Keys 进行多个分割来优化解码过程（图中显示了 1/5, 2/5, 3/5, 4/5 和 5/5 分割方式）。
    - 这种方法允许每个分割独立进行并行解码，从而更好地占用 GPU 的计算资源，提高了解码的效率和速度。
    - 图中展示了每个分割部分如何被分别处理，并最终合并成完整的输出结果。

Flash-Decoding和Ring Attention的区别是，它不需要在多个Host上进行序列切分和通信传递K和V，而是通过2个Kernel来完成长序列的Attention的计算。从某种角度来说，我们也可以把Flash Decoding看作是Ring Attention在推理阶段的一个优化。

![](https://files.mdnice.com/user/59/a73840aa-aa64-42d3-a7e9-e6f562a2839a.png)

最后一张Slides给出了这节课的一些链接。

## 总结

这节课介绍了一下Ring Attention的原理，基于Flash Attention的Ring Attention的基础实现，以及如何通过Stripe Permutation来解决Ring Attention的负载不均衡问题，最后介绍了Flash Decoding和Flash Attention的原理和区别。很高兴看到国人(github.com/zhuzilin)的工作可以出圈到CUDA-MODE，也推荐大家看原作者的Ring Attention讲解和改进的文章：https://zhuanlan.zhihu.com/p/683714620 。最近作者又提出了《更适合 flash attenion 体质的长上下文训练方案》：https://zhuanlan.zhihu.com/p/718486708 ，也推荐大家学习这个。感谢zhuzilin的优秀工作以及毫不吝啬的开源和分享。







