> 我的课程笔记，欢迎关注：https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode 。

# 第20课，前缀和（Scan） 算法（上）

## 课程笔记

![](https://files.mdnice.com/user/59/7d30f010-d21c-4b2a-b28a-5d944a415aa0.png)

![](https://files.mdnice.com/user/59/17a89b62-d4ba-4d25-93b5-12c34ef2ec5c.png)

这张Slides介绍了扫描(scan)操作的基本概念。scan操作需要两个输入:一个输入数组`[x₀, x₁, ..., xₙ₋₁]`和一个关联运算符(如sum、product、min、max等)。它会返回一个输出数组`[y₀, y₁, ..., yₙ₋₁]`。其中有两种scan方式:包含式扫描(Inclusive scan),其中`yᵢ = x₀ ⊕ x₁ ⊕ ... ⊕ xᵢ`;以及排除式扫描(Exclusive scan),其中`yᵢ = x₀ ⊕ x₁ ⊕ ... ⊕ xᵢ₋₁`。这里的`⊕`表示所选择的关联运算符。

![](https://files.mdnice.com/user/59/d268c057-6da0-4af0-a96f-c46b1df47b32.png)

