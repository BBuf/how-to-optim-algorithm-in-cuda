> 本文由 @Simon V(https://github.com/simveit) 授权转载和翻译并发表到本公众号。原始地址为：https://veitner.bearblog.dev/bridging-math-and-code-cute-layout-algebra-in-cutedsl/

# 连接数学和代码：CuTeDSL中的 CuTe Layout代数

19 May, 2025

## 介绍

本周，`CUTLASS`团队发布了新版本的`CUTLASS`，引入了`CuTeDSL`这个Python接口。根据其专门文档(https://docs.nvidia.com/cutlass/media/docs/pythonDSL/overview.html)所述，该接口让用户可以访问核心概念，如布局、张量、硬件原子，以及对硬件线程和数据层次结构的完全控制。

在这篇博文中，我们旨在让读者基本理解`CuTe`布局代数的一些原理。我们将解释一些基本概念，如布局函数、合并和补充。这有助于更深入地理解`CuTeDSL`。

## Layout

我们将Layout定义为

$$L = (M_0, M_1, \ldots, M_n):(d_0, d_1, \ldots, d_n)$$

其中

$S = (M_0, M_1, \ldots, M_n)$ 是形状（shape），其中 $M = M_0 \cdot \ldots \cdot M_n$ 是布局的大小。$D = (d_0, d_1, \ldots, d_n)$ 是步长（stride）。

pairs $(M_i):(d_i)$ 被称为模式（modes），可以被认为是长度为1的布局。一般来说，我们定义长度为上面的 $n + 1$。

## 布局函数

令

$$
l(x) = (x \bmod M_n, \lfloor \frac{x}{M_n} \rfloor \bmod M_1, \ldots, \lfloor \frac{x}{M_0 \cdot \ldots \cdot M_{n-1}} \rfloor \bmod M_n) = (i_0, i_1, \ldots, i_n)
$$

这是一个同构映射 $[0, M) \cong [0, M_0) \times \ldots \times [0, M_n)$。它将1维索引映射到多维索引。

然后布局函数被定义为应用同构映射到数字 $x$（如果需要）的映射，将每个向量元素与相应的步长元素相乘并将这些求和，即：

$$f_L(x) = i_0 d_0 + i_1 d_1 + \ldots + i_n d_n$$

让我们考虑一个例子，布局 $(2, 4):(2, 2)$。

让我们计算 $f_L(3)$。

$$l(x) = (3 \bmod 2, \lfloor \frac{3}{2} \rfloor \bmod 4) = (1,1)$$

从这里我们得到

$$f_L(3) = 1 \cdot 2 + 1 \cdot 2 = 2 + 2 = 4$$

我们可以在`CuTeDSL`中计算这个：

```python
import cutlass               
import cutlass.cute as cute  

@cute.jit
def layout_function_example():
    """
    Layout function in cutlass
    """
    S = (2, 4)
    D = (2, 2)
    L = cute.make_layout(shape=S, stride=D)

    for i in cutlass.range_constexpr(cute.size(S)):
        cute.printf("fL({}) = {}", i, L(i))

layout_function_example()
```

这会打印

```shell
fL(0) = 0
fL(1) = 2
fL(2) = 2
fL(3) = 4
fL(4) = 4
fL(5) = 6
fL(6) = 6
fL(7) = 8
```

## Sorted layouts

我们定义排序布局（sorted layout）使得所有步长都是递增的，即 $d_{i-1} \leq d_i$。

排序不会保持布局不变。考虑以下表示：

$$L_1 = (2,2):(3,1)$$
$$L_2 = \text{sorted}(L_1) = (2,2):(1,3)$$

相应的布局函数不重合，我们可以用`CuTeDSL`轻松验证：

```shell
import cutlass               
import cutlass.cute as cute  

@cute.jit
def sorted_example():
    """
    Sorting in cutlass
    """
    S1 = (2, 2)
    D1 = (3, 1)
    L1 = cute.make_layout(shape=S1, stride=D1)
    S2 = (2, 2)
    D2 = (1, 3)
    L2 = cute.make_layout(shape=S2, stride=D2)

    for i in cutlass.range_constexpr(cute.size(S1)):
        cute.printf("fL1({}) = {}, fL2({}) = {}", i, L1(i), i, L2(i))

sorted_example()
```

这将打印:

```shell
fL1(0) = 0, fL2(0) = 0
fL1(1) = 3, fL2(1) = 1
fL1(2) = 1, fL2(2) = 3
fL1(3) = 4, fL2(3) = 4
```

当然我们也可以通过手动计算来验证这个简单的例子。通过观察上面的例子,我们可以看到 $L_1$ 是行主序,而 $L_2$ 是列主序。

## 合并

合并是一种操作,它:

- 保持布局的大小不变
- 保持相关的布局函数不变

让我们看一个简单的例子:

$$L = (2,1):(3,1)$$

通过观察上面 $\iota$ 的定义，我们可以识别出 $\iota_2$ 将始终为0，即它不会对任何值 $x$ 的布局函数产生贡献，即：

$$\text{coalesce}(L) = 2:3$$

我们可以验证这个：

```python
import cutlass               
import cutlass.cute as cute  

@cute.jit
def coalesce_example():
    """
    Coalesce in cutlass
    """
    S = (2, 1)
    D = (3, 1)
    L = cute.make_layout(shape=S, stride=D)
    cL = cute.coalesce(L)

    cute.printf("L = {}, cL = {}", L, cL)

coalesce_example()
```

这将打印：

```shell
L = (2,1):(3,1), cL = 2:3
```

## 补充（Complementation）

### 可接受性（Admissibility）

设 $L$ 为一个布局，$K$ 为一个正整数。$(L,K)$ 对于补充是可接受的，当且仅当满足以下条件：

- $M_{i-1}d_{i-1}$ 整除 $d_i$
- $M_n d_n$ 整除 $K$

### 补充操作（Complement）

如果 $(L,K)$ 是可接受的，我们定义补充操作如下：

$$\text{complement}(L,K) = (d_0, \frac{d_1}{M_0 d_0}, \ldots, \frac{d_n}{M_{n-1} d_{n-1}}, \frac{K}{M_n d_n}):(1, M_0 d_0, \ldots, M_n d_n)$$

让我们看一个简单的例子：

$$L = (2,4):(1,2), K = 16$$

$(L,K)$ 是可接受的。

我们可以计算补充：

$$\text{complement}(L,K) = (1,1,2):(1,2,8)$$

使用与上面相同的参数，我们可以将其合并为

$$A = 2:8$$

我们可以这样计算：

// ... existing code ...

