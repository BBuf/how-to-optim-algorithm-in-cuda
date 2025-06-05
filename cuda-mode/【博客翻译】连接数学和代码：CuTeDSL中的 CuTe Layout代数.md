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

## 补操作

### 可接受性（Admissability）
设 $L$ 是一个布局（layout），$K$ 是正整数。$(L,K)$ 是可接受的，当且仅当满足以下条件：
$M_{i-1}d_{i-1}$ 整除 $d_i$
$M_n d_n$ 整除 $K$

### 补操作（Complement）
如果 $(L,K)$ 是可接受的，我们定义补操作如下：
$$\text{complement}(L,K) = \left(d_0, \frac{d_1}{M_0d_0}, \ldots, \frac{d_n}{M_{n-1}d_{n-1}}, \frac{K}{M_nd_n}\right):(1, M_0d_0, \ldots, M_nd_n)$$

### 示例推导

给定：$L = (2,4):(1,2)$，$K = 16$

- 左边 $(2,4)$：$M_0 = 2, M_1 = 4$
- 右边 $(1,2)$：$d_0 = 1, d_1 = 2$

$M_0 d_0 = 2 \times 1 = 2$ 整除 $d_1 = 2$ ✓
$M_1 d_1 = 4 \times 2 = 8$ 整除 $K = 16$ ✓
因此 $(L,K)$ 是可接受的。

根据补操作公式：
$$\text{complement}(L,K) = \left(d_0, \frac{d_1}{M_0d_0}, \frac{K}{M_1d_1}\right):(1, M_0d_0, M_1d_1)$$
代入正确的数值：
$$\text{complement}(L,K) = \left(1, \frac{2}{2 \times 1}, \frac{16}{4 \times 2}\right):(1, 2 \times 1, 4 \times 2)$$
$$= (1, 1, 2):(1, 2, 8)$$

由于前面有两个1，可以合并：
$$A = 2:8$$

我们可以使用下面的代码进行计算：

```python
import cutlass               
import cutlass.cute as cute  

@cute.jit
def complement_example():
    """
    Complement in cutlass
    """
    S = (2, 4)
    D = (1, 2)
    L = cute.make_layout(shape=S, stride=D)
    K = 16

    cL = cute.complement(L, K)

    cute.printf("L = {}, cL = {}", L, cL)

complement_example()
```

给出结果和上面推导的相同：

```shell
L = (2,4):(1,2), cL = 2:8
```

我们可以这样解释补操作：

设 $A$ 是布局及其补操作的连接。连接可以通过将所有模式组合到一个布局中来简单形成。对于上面的例子，即：

$$A = (2,4,2):(1,2,8)$$

我们可以证明这给了我们一个双射 $f_A:[0,M) \to [0,M)$。请参见本博文的命题 2.7(https://leimao.github.io/article/CuTe-Layout-Algebra/)。

我们可以使用 CuTeDSL 来验证这一点：

```python
import cutlass               
import cutlass.cute as cute  

@cute.jit
def complement_example2():
    """
    Complement in cutlass
    """
    S = (2, 4, 2)
    D = (1, 2, 8)
    L = cute.make_layout(shape=S, stride=D)

    for i in cutlass.range_constexpr(cute.size(L)):
        cute.printf("{} -> {}", i, L(i))
    
complement_example2()

```

```shell
0 -> 0
1 -> 1
2 -> 2
3 -> 3
4 -> 4
5 -> 5
6 -> 6
7 -> 7
8 -> 8
9 -> 9
10 -> 10
11 -> 11
12 -> 12
13 -> 13
14 -> 14
15 -> 15
```

注意，双射不一定是恒等映射。

例如，取

$$B = 8:2$$ 且 $$K = 32$$

$(B,K)$ 是可接受的，因为 $8 \times 2 = 16$ 整除 $32$。

$$\text{complement}(B,K) = (2,2):(1,16)$$

连接后的布局为

$$A = (8,2,2):(2,1,16)$$

打印出布局函数的值将给我们一个不等于恒等映射的双射：

```shell
0 -> 0
1 -> 2
2 -> 4
3 -> 6
4 -> 8
5 -> 10
6 -> 12
7 -> 14
8 -> 1
9 -> 3
10 -> 5
11 -> 7
12 -> 9
13 -> 11
14 -> 13
15 -> 15
16 -> 16
17 -> 18
18 -> 20
19 -> 22
20 -> 24
21 -> 26
22 -> 28
23 -> 30
24 -> 17
25 -> 19
26 -> 21
27 -> 23
28 -> 25
29 -> 27
30 -> 29
31 -> 31
```

## 结论

我希望这篇博文能通过将数学概念与编程连接起来，为读者提供一个简单的介绍。有关概念的更深入数学解释，请参阅Lei Mao的博客(https://leimao.github.io/article/CuTe-Layout-Algebra/)和Jay Shah关于CuTe布局代数的笔记。

有关CuTeDSL的更多示例，请参阅CUTLASS存储库。

