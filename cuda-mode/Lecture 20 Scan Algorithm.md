> 我的课程笔记，欢迎关注：https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode 。

# 第20课，前缀和（Scan） 算法（上）

## 课程笔记

![](https://files.mdnice.com/user/59/7d30f010-d21c-4b2a-b28a-5d944a415aa0.png)

![](https://files.mdnice.com/user/59/17a89b62-d4ba-4d25-93b5-12c34ef2ec5c.png)

这张Slides介绍了扫描(scan)操作的基本概念。scan操作需要两个输入:一个输入数组`[x₀, x₁, ..., xₙ₋₁]`和一个关联运算符(如sum、product、min、max等)。它会返回一个输出数组`[y₀, y₁, ..., yₙ₋₁]`。其中有两种scan方式:包含式扫描(Inclusive scan),其中`yᵢ = x₀ ⊕ x₁ ⊕ ... ⊕ xᵢ`;以及排除式扫描(Exclusive scan),其中`yᵢ = x₀ ⊕ x₁ ⊕ ... ⊕ xᵢ₋₁`。这里的`⊕`表示所选择的关联运算符。

![](https://files.mdnice.com/user/59/d268c057-6da0-4af0-a96f-c46b1df47b32.png)

这张Slides介绍了两种前缀和扫描(Prefix Sum Scan)的操作类型：包含式扫描(Inclusive Scan)和排除式扫描(Exclusive Scan)。通过一个具体的加法示例和一般形式的数学表达式来说明这两种扫描的区别。

![](https://files.mdnice.com/user/59/4724e623-7ec5-4d33-8a29-957d5f85e049.png)

这张Slides展示了包含式扫描(Inclusive Scan)和排除式扫描(Exclusive Scan)的顺序实现代码，首先通过求和操作的具体示例展示了两种扫描的实现差异，然后给出了通用形式的代码实现，其中包含式扫描会将当前元素包含在计算中，而排除式扫描则使用一个初始值（和运算中为0，一般情况下为IDENTITY）并且只计算当前元素之前的结果。

![](https://files.mdnice.com/user/59/dffefa7f-692d-4572-a001-bd13570cf6cd.png)

这张Slides介绍了并行扫描中的分段扫描(Segmented Scan)策略，由于并行扫描需要在并行工作线程之间进行同步，因此采用分段扫描的方法，即让每个线程块处理一个数据段，先计算每个段的部分和，然后对这些部分和进行扫描，最后将每个段的扫描结果累加到下一个段中，从而实现高效的并行处理。

![](https://files.mdnice.com/user/59/a98c053e-d448-40e4-a495-dcbe3cc5e151.png)

这张Slides通过图示展示了分段扫描的具体执行过程：首先将数据分成四个块（Block 0-3），每个块并行执行扫描操作，然后收集每个块的部分和并对这些部分和进行扫描，最后将扫描后的结果加回到对应的块中（除了Block 0），从而完成整个并行扫描操作。最后提到当前将重点关注如何在每个块内实现并行扫描。

![](https://files.mdnice.com/user/59/396aabd1-03ee-4122-938a-b55f405f9638.png)


这张Slides展示了并行包含式扫描(Parallel Inclusive Scan)的实现过程，通过一个并行归约树的结构来说明如何计算最后一个元素的结果，在计算过程中，每一层都会产生一些中间结果作为副产品，这些副产品实际上就是其他位置的扫描结果，整个过程采用树形结构自底向上进行并行计算，从而提高计算效率。

![](https://files.mdnice.com/user/59/3738a1fd-8092-48d9-acdf-b2180448f989.png)

这张Slides展示了另一种并行归约树的结构，通过调整计算顺序和组合方式，可以在计算过程中得到更多的中间结果（在图中用绿色高亮显示），这些中间结果正是并行包含式扫描所需的其他位置的累积值，从而能够更高效地完成整个扫描操作，获得更多的有用计算结果。

![](https://files.mdnice.com/user/59/d2c70990-7699-4df1-a2e9-69a6459a681a.png)

![](https://files.mdnice.com/user/59/0c5de993-8214-4adb-9a4e-ac884dac633b.png)

这两张Slides展示了并行包含式扫描的持续计算过程，通过不断构建新的归约树（用黄色箭头表示），可以逐步计算出所有位置的扫描结果，图中黄色高亮的部分表示通过这次归约树计算得到的新结果，最终通过多次归约树的计算可以获得完整的扫描序列，直到所有位置的值都被正确计算出来。

![](https://files.mdnice.com/user/59/67838d03-84c0-42d0-b0b9-be5dfabd1887.png)

这张Slides展示了一种优化的并行包含式扫描方法，通过将多个归约树重叠在一起并同时执行（用不同颜色的箭头表示不同的归约树），可以在同一时间并行计算出多个位置的结果，从而提高了计算效率，减少了需要的计算步骤，使得整个扫描操作能够更快地完成。

![](https://files.mdnice.com/user/59/94a7730e-9095-440b-ba77-f92372b24b26.png)

这张Slides介绍了Kogge-Stone并行包含式扫描算法的实现方式，它为每个输入元素分配一个独立的线程（用波浪线表示），通过三个步骤的并行计算来完成扫描操作，每一步都会计算出更大范围的部分和，最终得到完整的扫描结果，这种算法结构使得计算可以高度并行化，是一种高效的并行扫描实现方法。

![](https://files.mdnice.com/user/59/1c180fd0-9944-496b-96ee-a8e418636492.png)

