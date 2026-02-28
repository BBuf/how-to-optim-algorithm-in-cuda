> 博客来源：https://pytorch.org/blog/llama-into-torchtune/ by Linda Wang, Evan Smothers, Kartikay Khandelwal 这里做了翻译可以帮助读者了解如何对LLM做知识蒸馏。总结来说，这篇博客介绍了如何使用torchtune将Llama 3.1 8B模型蒸馏为1B模型，通过知识蒸馏技术提高小模型在指令跟随任务上的性能。文章详细解释了知识蒸馏的工作原理，并展示了在torchtune中的实现过程，包括模型下载、教师模型微调和蒸馏步骤。此外，博客上还展示了4个消融实验，探讨了不同配置和超参数对结果的影响，最后谈了下可以继续做的事情。

# 使用torchtune把LLaMa-3.1 8B蒸馏为1B

在这篇博客中,我们展示了一个使用torchtune的知识蒸馏配方将Llama 3.1 8B模型蒸馏为Llama 3.2 1B的案例研究。我们演示了如何在训练后使用知识蒸馏(KD)来提高指令跟随任务的性能,并展示了用户如何利用这个配方。

## 什么是知识蒸馏?

知识蒸馏(https://arxiv.org/pdf/1503.02531)是一种广泛使用的压缩技术,它将知识从较大的(教师)模型转移到较小的(学生)模型。较大的模型有更多的参数和知识容量,但是这种较大的容量在部署时也需要更多的计算资源。知识蒸馏可以用来将较大模型的知识压缩到较小的模型中。其基本思想是,通过学习较大模型的输出,可以提高较小模型的性能。

## 知识蒸馏是如何工作的?

知识是通过在一个迁移集上训练来从教师模型转移到学生模型的,在这个过程中学生模型被训练来模仿教师模型的 token 级别概率分布。这里的假设是教师模型的分布与迁移数据集相似。下图是知识蒸馏工作原理的简化表示。

![图1: 从教师模型到学生模型的知识迁移示意图](https://files.mdnice.com/user/59/91703a27-3f74-4b98-995e-d6819677204e.png)

由于LLM的知识蒸馏是一个活跃的研究领域,目前有许多论文在研究不同的损失函数方法,比如MiniLLM(https://arxiv.org/pdf/2306.08543)、DistiLLM(https://arxiv.org/pdf/2402.03898)、AKL(https://arxiv.org/pdf/2404.02657)和Generalized KD(https://arxiv.org/pdf/2306.13649)。在这个案例研究中,我们将重点关注标准交叉熵(CE)损失和前向Kullback-Leibler(KL)散度损失(https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)作为基线。前向KL散度的目标是通过强制学生模型的分布与教师模型的所有分布对齐来最小化差异。

## 为什么知识蒸馏有用?

知识蒸馏的理念是,相比从头开始训练或有监督微调,一个较小的模型可以通过使用教师模型的输出作为额外信号来获得更好的性能。例如,Llama 3.2轻量级1B和3B文本模型(https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)在剪枝后通过整合Llama 3.1 8B和70B的logits来恢复性能。此外,在指令跟随任务的微调中,LLM蒸馏的研究表明,知识蒸馏方法可以优于单独使用有监督微调(SFT)。

![表1: 知识蒸馏方法与有监督微调的比较](https://files.mdnice.com/user/59/12b7ad02-f12d-4780-a4ec-6bb2d8505cd6.png)

下面是一个简化的例子,展示了知识蒸馏与有监督微调的区别。

![](https://files.mdnice.com/user/59/bd7f507c-fb57-4e84-b822-23e8c428612d.png)

## torchtune中的知识蒸馏配方

使用torchtune,我们可以轻松地将知识蒸馏应用于Llama3以及其他LLM模型系列,这是通过使用torchtune的知识蒸馏配方(https://github.com/pytorch/torchtune/blob/4234b78b914af23384ce0348f564e2119d107a96/recipes/knowledge_distillation_single_device.py)实现的。这个配方的目标是通过从Llama3.1-8B蒸馏知识来在Alpaca指令跟随数据集上微调Llama3.2-1B。这个配方专注于训练后蒸馏,并假设教师和学生模型都已经预训练完成。

首先,我们需要下载模型权重。为了与其他torchtune微调配置保持一致,我们将使用Llama3.1-8B的指令调优模型作为教师模型,使用Llama3.2-1B作为学生模型。

```shell
tune download meta-llama/Meta-Llama-3.1-8B-Instruct --output-dir /tmp/Meta-Llama-3.1-8B-Instruct --ignore-patterns "original/consolidated.00.pth" --hf_token <HF_TOKEN>

tune download meta-llama/Llama-3.2-1B-Instruct --output-dir /tmp/Llama-3.2-1B-Instruct --ignore-patterns "original/consolidated.00.pth" --hf_token <HF_TOKEN>
```

为了让教师模型的分布与Alpaca数据集相似,我们将使用LoRA对教师模型进行微调。基于我们在下一节展示的实验,我们发现当教师模型已经在目标数据集上进行了微调时,知识蒸馏的效果会更好。

```shell
tune run lora_finetune_single_device --config llama3_1/8B_lora_single_device
```

最后,我们可以运行以下命令在单个GPU上将微调后的8B模型蒸馏为1B模型。在这个案例研究中,我们使用了一个A100 80GB GPU。我们还有一个分布式配方(https://github.com/pytorch/torchtune/blob/09c2619f713e771b4159f7b83bac8971c7053bd3/recipes/knowledge_distillation_distributed.py)用于在多个设备上运行。

```shell
tune run knowledge_distillation_single_device --config llama3_2/knowledge_distillation_single_device
```

## 消融研究

在本节中,我们将展示改变配置和超参数如何影响性能。默认情况下,我们的配置使用LoRA微调的8B教师模型、下载的1B学生模型、3e-4的学习率和0.5的KD损失比率。对于这个案例研究,我们在alpaca_cleaned_dataset(https://pytorch.org/torchtune/main/generated/torchtune.datasets.alpaca_cleaned_dataset.html#torchtune.datasets.alpaca_cleaned_dataset)上进行了微调,并通过EleutherAI LM评估工具(https://github.com/EleutherAI/lm-evaluation-harness/tree/main)在truthfulqa_mc2、hellaswag和commonsense_qa任务上评估了模型。让我们来看看以下因素的影响:

### 使用微调的教师模型

配置中的默认设置使用微调后的教师模型。现在,让我们看看不先微调教师模型的效果。

从损失来看,使用基线8B作为教师模型比使用微调后的教师模型会导致更高的损失。KD损失也保持相对恒定,这表明教师模型应该与迁移数据集具有相同的分布。

![图2: (从左到右)前向KL散度的KD损失、交叉熵的分类损失、总损失:KD损失和分类损失的均匀组合](https://files.mdnice.com/user/59/3de5df0b-5f74-4eb9-87c1-bdf6131ae386.png)

在我们的基准测试中,可以看到1B模型的有监督微调比基线1B模型获得了更好的准确率。通过使用微调后的8B教师模型,我们在truthfulqa上看到了相当的结果,并且在hellaswag和commonsense上有所改进。当使用基线8B作为教师模型时,我们看到所有指标都有提升,但低于其他配置。

![表2: 使用基线和微调后的8B作为教师模型的对比](https://files.mdnice.com/user/59/1589cc9a-1895-42dc-b6e9-e7812290df8b.png)

### 使用微调的学生模型

在这些实验中,我们研究了当学生模型已经微调时KD的效果。我们分析了使用基线和微调后的8B和1B模型的不同组合的效果。

根据损失图表,使用微调后的教师模型会导致更低的损失,无论学生模型是否经过微调。有趣的是,当使用微调后的学生模型时,分类损失开始增加。

![图3: 比较不同教师和学生模型初始化的损失](https://files.mdnice.com/user/59/c57bb43f-0783-4fc1-8713-7af4fdc069d4.png)

使用微调后的学生模型可以进一步提高truthfulqa的准确率,但在hellaswag和commonsense上的准确率有所下降。使用微调后的教师模型和基线学生模型在hellaswag和commonsense数据集上取得了最好的结果。基于这些发现,最佳配置会根据你要优化的评估数据集和指标而改变。

![表3: 使用基线和微调后的教师和学生模型的对比](https://files.mdnice.com/user/59/9cf3fda4-0c83-4aa0-8e0f-3db3c60a50dc.png)

### 超参数调优：学习率

默认情况下,配方使用3e-4的学习率。在这些实验中,我们将学习率从最高1e-3调整到最低1e-5。

根据损失图表,除了1e-5会导致更高的KD损失和分类损失外,所有学习率都产生了类似的损失。

![图4: 比较不同学习率的损失](https://files.mdnice.com/user/59/ca48a156-6307-45d3-8a8a-0e3798122bdd.png)

根据我们的基准测试,最优学习率会根据你要优化的评估指标和任务而变化。

![表4: 调整学习率的效果](https://files.mdnice.com/user/59/fb235277-ee45-4ad7-bc45-28c8172722f0.png)

### 超参数调优：KD比率

默认情况下,KD比率设置为0.5,这样可以对分类损失和KD损失进行均匀加权。在这些实验中,我们研究了不同KD比率的效果,其中0表示仅使用分类损失,1表示仅使用KD损失。

总的来说,基准测试结果表明,对于这些任务和指标,较高的KD比率表现略好。

![表5: 调整KD比率的效果](https://files.mdnice.com/user/59/249de42c-db36-4005-8d42-383579d88efa.png)

## 展望未来

在本文中,我们介绍了如何使用前向KL散度损失通过torchtune将Llama 3.1 8B和Llama 3.2 1B的logits进行蒸馏的研究。未来还有许多方向可以探索,以进一步提高性能并为蒸馏方法提供更大的灵活性。

- 扩展KD损失函数。KD配方使用前向KL散度损失。然而,如上所述,将学生分布对齐到整个教师分布可能并不有效。有多篇论文,如MiniLLM(https://arxiv.org/pdf/2306.08543)、DistiLLM(https://arxiv.org/pdf/2402.03898)和Generalized KD(https://arxiv.org/pdf/2306.13649),引入了新的KD损失和策略来解决这个限制,并且已经证明优于标准的交叉熵和前向KL散度损失。例如,MiniLLM使用反向KL散度来防止学生过度估计教师的低概率区域。DistiLLM引入了偏斜KL损失和自适应训练策略。

- 启用跨分词器蒸馏。当前的配方要求教师和学生模型使用相同的分词器,这限制了跨不同LLM家族进行蒸馏的能力。已经有一些关于跨分词器方法的研究(例如Universal Logit Distillation(https://arxiv.org/pdf/2402.12030))值得我们探索。

- 将蒸馏扩展到多模态LLM和编码器模型。KD配方的一个自然扩展是扩展到多模态LLM。与部署更高效的LLM类似,也需要部署更小更高效的多模态LLM。此外,已经有一些工作展示了LLM作为编码器模型的应用(例如LLM2Vec(https://arxiv.org/pdf/2404.05961))。从LLM编码器到更小的编码器模型的蒸馏也可能是一个值得探索的有前途的方向。

