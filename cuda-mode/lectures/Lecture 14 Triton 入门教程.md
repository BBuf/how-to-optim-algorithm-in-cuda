> 我的课程笔记，欢迎关注：https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode 

> CUDA-MODE Lecture 15是讲cutlass的cute Layout抽象的，感觉讲的比较差，建议大家直接看reed大佬的cutlass系列博客介绍，接下来会忽略掉这节课的笔记。CUDA-MODE Lecture 16: On Hands profiling是一个关于PyTorch Lighting的工程师根据一个实际的gemma模型微调的程序来进行profile和改进性能的课程，这节课没有Slides更贴近AI Infra工程师的生活，profile工具使用了Nsight System和PyTorch Profiler，对这节课感兴趣的小伙伴可以自行查阅这个课程，由于没有Slides并且讲得很随意所以笔者也不打算记录这节课的笔记。但如果你平时有做Profile的需求，我还是建议看一下这节课。

> 下面的课程笔记的内容主要来源是 Lecture 14 Triton 实践指南中的 https://github.com/gpu-mode/lectures/blob/main/lecture_014/A_Practitioners_Guide_to_Triton.ipynb 

# 第14课，Triton 实践指南

<h1><b></v>Triton 实践指南</b></h1>

作者：UmerHA (https://x.com/UmerHAdil // https://github.com/UmerHA/)，为 cuda-mode 小组编写 ❤️ May our brrrr level reach over 。

# 为什么以及何时使用Triton

**什么是Triton**
简而言之：Triton是一种更方便地编程GPU的语言。你编写类似Python的代码，然后这些代码被编译成ptx代码（与cuda代码编译成的代码相同）。

在编译过程中，Triton编译器尝试使用巧妙的技巧来重新排列程序的部分内容（不改变程序的意义！）以使其运行得更快。

**Triton vs Cuda**

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/dda26356-ad90-415d-8cbe-5b5b4bad930d.png)
source: https://zhuanlan.zhihu.com/p/672086654

CUDA 是一个高端工具，有许多设置供专业人士使用。
- 对所有内容有完全控制，因此可以实现绝对最大性能
- 更难获得良好的性能
- 编写和调试更加繁琐
- 更复杂，因此更难学习

Triton 是一个非常适合大多数用户的工具
- 你不能控制所有内容，因为有些事情留给自动优化；所以你可能不会获得绝对最大性能
- 更容易获得良好的性能
- 更容易编写和调试
- 更容易学习，因为它具有类似 Python 的语法

**Triton vs torch.compile**

`torch.compile` 通过尝试更有效地使用现有kernel并创建简单的新kernel来使你的模型更快。这可能会使你的模型足够快。如果没有，你可以决定投入时间编写更快的 Triton kernel。

（`torch.compile` 创建的这些简单新kernel实际上是 Triton kernel。因此，它们是自定义kernel的良好起点。参见 [Mark Saroufim](https://twitter.com/marksaroufim) 的 [cuda mode 第一讲](https://www.youtube.com/watch?v=LuhJEEJQgUM&t=2200s) 了解如何操作。）

**何时使用 Triton**

你从你的 AI 模型开始。
1. 如果它不够快，使用 `torch.compile`。
2. 如果它不够快，检查你是否可以重写代码以使其更适合 `torch.compile`。
3. 如果它不够快，检查哪些部分慢并为其编写自定义 Triton kernel。
4. 如果它不够快，检查哪些部分慢并为其编写自定义 CUDA kernel。

（在不太可能的情况下，如果你事先知道你需要绝对最大性能，你可以决定直接从 CUDA 开始。）

**关于粗糙边缘的说明**

由于 Triton 是一个较新的项目，人们发现它有一些粗糙的边缘。我已经记录了我遇到的所有粗糙边缘，并在评论中注明了“Weirdness: <我对奇怪之处的描述>”。

我预计它会在未来变得更加完善。

# 如何编写Triton kernel

与CUDA不同，如果我们设置环境变量 `TRITON_INTERPRET = 1`，我们可以像调试任何CPU程序一样调试Triton kernel。然后Triton在CPU上运行，但模拟它在GPU上运行。

我建议首先在模拟器中编写所有程序，并检查其正确性。如果正确，然后你可以使其快速运行。

以下是一些用于调试的实用函数：
- `check_tensors_gpu_ready`：(i) 断言所有张量在内存中是连续的，(ii) 仅在非模拟情况下，断言所有张量在GPU上
- `breakpoint_if`：根据pids的条件设置断点
- `print_if`：根据pids的条件打印内容

```python
import os
from IPython.core.debugger import set_trace

os.environ['TRITON_INTERPRET'] = '1' # needs to be set *before* triton is imported

def check_tensors_gpu_ready(*tensors):
    """检查所有张量是否在GPU上并且是连续的"""
    for t in tensors:
        assert t.is_contiguous, "A tensor is not contiguous"  # 断言张量是连续的
        if not os.environ.get('TRITON_INTERPRET') == '1': assert t.is_cuda, "A tensor is not on cuda"  # 如果不是模拟模式，断言张量在GPU上

def test_pid_conds(conds, pid_0=[0], pid_1=[0], pid_2=[0]):
    """测试pid条件是否满足
    例如:
        '=0'  检查pid_0 == 0
        ',>1' 检查pid_1 > 1
        '>1,=0' 检查pid_0 > 1 且 pid_1 == 0
    """
    pids = pid_0[0], pid_1[0], pid_2[0]  # 获取pid值
    conds = conds.replace(' ','').split(',')  # 去除空格并分割条件
    for i, (cond, pid) in enumerate(zip(conds, pids)):
        if cond=='': continue  # 如果条件为空，跳过
        op, threshold = cond[0], int(cond[1:])  # 获取操作符和阈值
        if op not in ['<','>','>=','<=','=', '!=']: raise ValueError(f"Rules may only use these ops: '<','>','>=','<=','=', '!='. Invalid rule: '{condition}'.")  # 检查操作符是否合法
        op = '==' if op == '=' else op  # 将'='替换为'=='
        if not eval(f'{pid} {op} {threshold}'): return False  # 评估条件是否满足
    return True

assert test_pid_conds('')  # 测试空条件
assert test_pid_conds('>0', [1], [1])  # 测试pid_0 > 0
assert not test_pid_conds('>0', [0], [1])  # 测试pid_0 > 0不满足
assert test_pid_conds('=0,=1', [0], [1], [0])  # 测试pid_0 = 0 且 pid_1 = 1

def breakpoint_if(conds, pid_0=[0], pid_1=[0], pid_2=[0]):
    """如果任何pid条件满足，停止kernel"""
    if test_pid_conds(conds, pid_0, pid_1, pid_2): set_trace()  # 如果条件满足，设置断点

def print_if(txt, conds, pid_0=[0], pid_1=[0], pid_2=[0]):
    """如果任何pid条件满足，打印txt"""
    if test_pid_conds(conds, pid_0, pid_1, pid_2): print(txt)  # 如果条件满足，打印文本

def cdiv(a,b): 
    """计算a除以b的上限值"""
    return (a + b - 1) // b  # 计算a除以b的上限值
assert cdiv(10,2)==5  # 测试cdiv函数
assert cdiv(10,3)==4  # 测试cdiv函数
```

```python
import torch
import triton
import triton.language as tl
```

# 编程模型

在CUDA中，我们将计算分解为两个层次：首先是块，然后每个块进一步分解为线程。一个块中的所有线程运行在同一个SM上，并共享相同的共享内存。每个线程计算**标量**。

在Triton中，我们只将计算分解为一个层次：块。没有进一步的线程分解。**Triton要求我们对向量进行操作**。此外，我们不需要也不能管理共享内存。Triton会自动处理这些。

示例：

假设我们要将大小为8的向量`x`和`y`相加，并将输出保存到大小也为8的向量`z`中。我们使用大小为4的块，因此我们有`8 / 4 = 2`个块。
- CUDA运行2个块，每个块有4个线程。8个线程中的每一个计算一个单独的位置，例如`z[0] = x[0] + y[0]`
- Triton也运行2个块，每个块执行向量化加法。向量的大小是块的大小，即4。例如`z[0:3] = x[0:3] + y[0:3]`

**所有**Triton kernel中的操作都是向量化的：加载数据、操作数据、存储数据和创建掩码。

让我们考虑另一个简单的例子：

同样，我们要将大小为**6**的向量`x`和`y`相加，并将输出保存到大小也为6的向量`z`中。我们使用大小为4的块，因此我们有`cdiv(6, 4) = 2`个块。

```python
x = torch.tensor([1,2,3,4,5,6])
y = torch.tensor([0,1,0,1,0,1])

x, y, x+y
```

CUDA kernel将类似于以下C代码：

```python
# x,y = 输入张量, z = 输出张量, n = x的大小, bs = 块大小
def add_cuda_k(x, y, z, n, bs):
    # 定位此特定kernel正在执行的整体计算的哪一部分
    block_id = ... # 在我们的例子中: 是[0,1]中的一个
    thread_id = ... # 在我们的例子中: 是[0,1,2,3]中的一个

    # 识别此特定kernel需要的数据位置
    offs = block_id * bs + thread_id
    
    # 保护子句, 确保我们不会越界
    if offs < n:

        # 读取数据
        x_value = x[offs]
        y_value = y[offs]
        
        # 执行操作
        z_value = x_value + y_value
        
        # 写入数据
        z[offs] = z_value

    # 重要: offs, x_value, y_value, x_value 都是标量!
    # 保护条件也是一种标量, 因为它检查一个值上的一个条件。
```

为了说明，这里是每个kernel的变量：

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-1/29863a80-c092-4d8d-a7de-0ecdb2a9213c.png)

现在让我们看一下相应的Triton kernel，大致如下所示：

```python
# 注意：这是为了说明，语法不完全正确。请参见下文以获取正确的Triton语法

def add_triton_k(x, y, z, n, bs):
    # 定位此特定kernel正在执行的整体计算的哪一部分
    block_id = tl.program_id(0)  # 在我们的例子中: 是[0,1]中的一个
    
    # 识别此特定kernel需要的数据位置
    offs = block_id * bs + tl.arange(0, bs) # <- 这是一个向量!
    
    # 保护子句变成一个掩码，这是一个布尔向量
    mask = offs < n # <- 这是一个布尔向量!
    
    # 读取数据
    x_values = x[offs] # <- 读取一个向量!
    y_values = y[offs] # <- 读取一个向量!
    
    # 执行操作
    z_value = x_value + y_value  # <- 向量相加!
    
    # 写入数据
    z[offs] = z_value  # <- 写入一个向量!
```

再次说明，这里是每个kernel的变量：

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-2/8dd01a33-89cc-4587-8376-b2471600c63a.png)

术语说明：在Triton术语中，每个处理块的kernel被称为“program”。也就是说，我们上面的例子运行了2个program。因此，“block_id”通常被称为“pid”（“program id”的缩写），但它们是相同的。

# 示例1: 复制张量

让我们看一些例子。为了保持简单，我们将使用非常小的块大小。

目标: 给定一个形状为 (n) 的张量 `x`，将其复制到另一个张量 `z` 中。

```python
# # 这是一个普通的Python函数，用于启动Triton kernel
def copy(x, bs, kernel_fn):
    z = torch.zeros_like(x)
    check_tensors_gpu_ready(x, z)
    n = x.numel()
    n_blocks = cdiv(n, bs)
    grid = (n_blocks,)  # 我们有多少个块？可以是1d/2d/3d元组或返回1d/2d/3d元组的函数

    # 启动网格！
    # - kernel_fn是我们下面编写的Triton kernel
    # - grid是我们上面构建的网格
    # - x,z,n,bs是传递给每个kernel函数的参数
    kernel_fn[grid](x,z,n,bs)

    return z
```

**注意:** 出于教育目的，下面的kernel有一个逻辑错误（但语法是正确的）。你能发现它吗？

```python
# # 这是Triton kernel:

# triton.jit装饰器将一个Python函数转换为Triton kernel，该kernel在GPU上运行。
# 在这个函数内部，只允许使用部分Python操作。
# 例如，当不进行模拟时，我们不能打印或使用断点，因为这些在GPU上不存在。
@triton.jit
# 当我们传递torch张量时，它们会自动转换为指向其第一个值的指针
# 例如，上面我们传递了x，但在这里我们接收到x_ptr
def copy_k(x_ptr, z_ptr, n, bs: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, bs)  # 从pid计算偏移量
    mask = offs < n
    x = tl.load(x_ptr + offs, mask) # 加载一个值向量，将`x_ptr + offs`视为`x_ptr[offs]`
    tl.store(z_ptr + offs, x, mask) # 存储一个值向量

    print_if(f'pid = {pid} | offs = {offs}, mask = {mask}, x = {x}', '')

    # 问题: 这个kernel有什么问题?
```

```python
z = copy(x, bs=2, kernel_fn=copy_k)
```

```python
pid = [0] | offs = [0 1], mask = [ True  True], x = [1 2]
pid = [1] | offs = [0 1], mask = [ True  True], x = [1 2]
pid = [2] | offs = [0 1], mask = [ True  True], x = [1 2]
```

```
z
```

```shell
tensor([1, 2, 0, 0, 0, 0])
```

我们没有正确地移动偏移量。我们总是使用 offsets = [0,1]，但它们应该随着 pid 变化。

```python
@triton.jit
def copy_k(x_ptr, z_ptr, n, bs: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * n + tl.arange(0, bs)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask)
    tl.store(z_ptr + offs, x, mask)
    print_if(f'pid = {pid} | offs = {offs}, mask = {mask}, x = {x}', '')
```

```
z = copy(x, bs=2, kernel_fn=copy_k)
```

```python
pid = [0] | offs = [0 1], mask = [ True  True], x = [1 2]
pid = [1] | offs = [6 7], mask = [False False], x = [1 1]
pid = [2] | offs = [12 13], mask = [False False], x = [1 1]
```

不完全正确。我们添加了 `pid * n`，但想要添加 `pid * bs`

```python
@triton.jit
def copy_k(x_ptr, z_ptr, n, bs: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * bs + tl.arange(0, bs)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask)
    tl.store(z_ptr + offs, x, mask)
    print_if(f'pid = {pid} | offs = {offs}, mask = {mask}, x = {x}', '')
```

```python
z = copy(x, bs=2, kernel_fn=copy_k)
```

```shell
pid = [0] | offs = [0 1], mask = [ True  True], x = [1 2]
pid = [1] | offs = [2 3], mask = [ True  True], x = [3 4]
pid = [2] | offs = [4 5], mask = [ True  True], x = [5 6]
```

Yes!

```python
x, z
```

```shell
(tensor([1, 2, 3, 4, 5, 6]), tensor([1, 2, 3, 4, 5, 6]))
```

正如我们所见，编写GPU程序涉及许多索引，我们很容易搞混。因此，我强烈建议先在模拟模式下编写和调试kernel，并首先使用小示例进行测试！


# 示例2：灰度化图像

在这个示例中，我们将灰度化一张小狗的图像。我们将看到如何处理二维数据。

这同样适用于三维数据。

我们改编了Jeremy Howard的示例，来自这个[colab](https://colab.research.google.com/drive/180uk6frvMBeT4tywhhYXmz3PJaCIA_uk?usp=sharing) / [youtube](https://www.youtube.com/watch?v=4sgKnKbR-WE&feature=youtu.be)。因此，感谢他的示例和选择的小狗图像。
> 注：在这个示例中，如果不重启jupyter内核，会发生两件奇怪的事情：

1. 无法导入torchvision，可能是由于循环依赖。-> 目前不知道为什么，需要深入挖掘。
2. 下面的模拟triton kernel失败，因为浮点数不能乘以uint向量 -> 在GPU上不进行模拟时可以工作，所以似乎是`TRITON_INTERPRET`的bug。

```python
import os

import matplotlib.pyplot as plt
from urllib.request import urlretrieve
from pathlib import Path

import torch
from torch import tensor
import torchvision as tv
import torchvision.transforms.functional as tvf
from torchvision import io

import triton
import triton.language as tl
def cdiv(a,b): return (a + b - 1) // b
url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/1600px-Cute_dog.jpg?20140729055059'
path_img = Path('puppy.jpg')
if not path_img.exists(): urlretrieve(url, path_img)
img = io.read_image('puppy.jpg')
print(img.shape)
img[:2,:3,:4]
```

```shell
torch.Size([3, 1066, 1600])
tensor([[[117, 119, 117, 113],
         [119, 129, 129, 113],
         [130, 126, 122, 115]],

        [[ 83,  85,  85,  80],
         [ 85,  97,  97,  82],
         [ 98,  93,  89,  83]]], dtype=torch.uint8)
```

```python
def show_img(x, figsize=(4,3), **kwargs):
    plt.figure(figsize=figsize)
    plt.axis('off')
    if len(x.shape)==3: x = x.permute(1,2,0)  # CHW -> HWC
    plt.imshow(x.cpu(), **kwargs)
img = tvf.resize(img, 150, antialias=True)
ch,h,w = img.shape
ch,h,w,h*w
```

```shell
(3, 150, 225, 33750)
```

```python
show_img(img)
```

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-2/63ac6c00-a993-40aa-aa01-afe0a887e153.png)

要处理二维数据，我们将构建二维偏移量和掩码。以下是如何工作的示例，例如对于一个 `4x7` 矩阵和每个维度的大小为 `2` 的块。

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/d5f75553-8b7f-4308-a46e-a9af30a0a70e.png)

在代码中，长这样:

```python
@triton.jit
def rgb2grey_k(x_ptr, out_ptr, h, w, bs0: tl.constexpr, bs1: tl.constexpr):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)
    
    offs_0 = pid_0 * bs0 + tl.arange(0,bs0)  # 1d 向量
    offs_1 = pid_1 * bs1 + tl.arange(0,bs1)  # 1d 向量

    # 奇怪的地方: 在CPU模拟时，None切片目前不起作用。使用tl.expand_dim代替。
    # offs = w * tl.expand_dims(offs_0, 1) + tl.expand_dims(offs_1, 0)
    offs = w * offs_0[:,None] + offs_1[None, :]  # 2d 矩阵! - 我们将第一个偏移量乘以宽度，见上图

    mask_0 = offs_0 < h  # 1d 向量
    mask_1 = offs_1 < w  # 1d 向量

    # mask = tl.expand_dims(mask_0, 1) & tl.expand_dims(mask_1, 0)
    mask = mask_0[:,None] & mask_1[None,:]  # 2d 矩阵! - 数据不能超出任一轴的范围，因此使用`逻辑与`来组合单独的掩码
    
    r = tl.load(x_ptr + 0*h*w+offs, mask=mask)
    g = tl.load(x_ptr + 1*h*w+offs, mask=mask)
    b = tl.load(x_ptr + 2*h*w+offs, mask=mask)

    # 奇怪的地方: 在CPU模拟时，浮点数与uint向量相乘会失败
    out = 0.2989*r + 0.5870*g + 0.1140*b  # 不用担心为什么是这3个数字相乘

    tl.store(out_ptr + offs, out, mask=mask)
```

让我们使用这个kernel!

```python
def rgb2grey(x, bs):
    c,h,w = x.shape
    out = torch.empty((h,w), dtype=x.dtype, device=x.device)

    # grid 可以是一个返回 1d/2d/3d 元组的函数
    # (在这种情况下，拥有一个 grid 函数并不比 grid 元组更有用，但在下面的基准测试和自动调优中会更有用)
    grid = lambda meta: (cdiv(h, meta['bs0']), cdiv(w,  meta['bs1']))
    
    rgb2grey_k[grid](x, out, h, w, bs0=bs[0], bs1=bs[1]) # 所有关键字参数都传递到 grid 函数中
    return out.view(h,w)
grey_img = rgb2grey(img.to('cuda'), bs=(32, 32)).to('cpu')
show_img(grey_img, cmap='gray')
```

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-2/8f39e5a8-7222-4011-94e2-d0f91d214dd4.png)


# 示例 3: 矩阵乘法

```python
import os
# os.environ['TRITON_INTERPRET'] = '1'

import torch
import triton
import triton.language as tl

# 将实用函数移到单独的文件中以提高可读性
from triton_util import cdiv, breakpoint_if, print_if, check_tensors_gpu_ready
```

现在，让我们在 Triton 中实现一个简单的矩阵乘法。我们将学习：
- 一种分割计算的方法
- 从kernel中调用函数
- 在块内使用预实现的向量/矩阵操作

这是从 [OpenAI 宣布 Triton 的博客文章](https://openai.com/research/triton)改编而来的。

我们希望将 `m x k` 矩阵 `A` 和 `k x n` 矩阵 `B` 乘以得到 `m x n` 矩阵 `C`。

我们沿着三个轴分割计算：
- 沿着 m 轴 - 我们将使用块维度 0 来表示这一点
- 沿着 n 轴 - 我们将使用块维度 1 来表示这一点
- 沿着共享的 k 轴 - 这将不会由块表示。所有计算块将在同一个块中完成。

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/f5a44c41-0d5f-49dc-b1c3-e0770cf61884.png)

由于我们经常创建一维或二维偏移量和掩码，让我们将这些功能放入实用函数中。只要这些函数被 `triton.jit` 编译，它们就可以在kernel中使用。

```python
@triton.jit
def get_1d_offset(size, n_prev_chunks):
    return n_prev_chunks * size + tl.arange(0, size)

@triton.jit
def get_2d_offset(offs_0, offs_1, stride_0, stride_1=1): 
    # 使用 tl.expand_dims 将 offs_0 和 offs_1 转换为二维张量
    # tl.expand_dims(offs_0, 1) 将 offs_0 转换为 (offs_0, 1) 形状的张量
    # tl.expand_dims(offs_1, 0) 将 offs_1 转换为 (1, offs_1) 形状的张量
    return tl.expand_dims(offs_0, 1)*stride_0 + tl.expand_dims(offs_1, 0)*stride_1

@triton.jit
def get_1d_mask(offs, max):
    return offs < max

@triton.jit
def get_2d_mask(offs_0, offs_1, max_0, max_1):
    # 使用 tl.expand_dims 将 offs_0 和 offs_1 转换为二维张量
    # tl.expand_dims(offs_0, 1) 将 offs_0 转换为 (offs_0, 1) 形状的张量
    # tl.expand_dims(offs_1, 0) 将 offs_1 转换为 (1, offs_1) 形状的张量
    return (tl.expand_dims(offs_0, 1) < max_0) & (tl.expand_dims(offs_1, 0) < max_1)
```

这是朴素的矩阵乘法内核：

```python
@triton.jit
def naive_matmul_k(
    a_ptr, b_ptr, c_ptr,
    m, n, k,
    stride_am, stride_ak, 
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    bm: tl.constexpr, bn: tl.constexpr, bk: tl.constexpr
):
    # 获取当前线程块的 ID
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    # 沿 m/n/k 维度分割计算
    rm = get_1d_offset(size=bm, n_prev_chunks=pid_m)  # 计算 m 维度的偏移量
    rn = get_1d_offset(size=bn, n_prev_chunks=pid_n)  # 计算 n 维度的偏移量
    rk = get_1d_offset(size=bk, n_prev_chunks=0)  # 计算 k 维度的偏移量
    # 计算 a 和 b 的相关偏移量
    offs_a = a_ptr + get_2d_offset(rm, rk, stride_am, stride_ak)  # 计算 a 的偏移量
    offs_b = b_ptr + get_2d_offset(rk, rn, stride_bk, stride_bn)  # 计算 b 的偏移量
    # 初始化并迭代更新累加器
    acc = tl.zeros((bm, bn), dtype=tl.float32)  # 初始化累加器
    for _ in range(0, k, bk):
        # todo umer: 加载 a 和 b 时是否需要掩码？
        a = tl.load(offs_a)  # 加载 a 的数据
        b = tl.load(offs_b)  # 加载 b 的数据
        acc += tl.dot(a, b, allow_tf32=False)  # 在块内进行矩阵乘法；注意：对于较旧的 GPU，allow_tf32 必须设置为 False，否则无法编译
        # 增加偏移量，以便下一次迭代加载下一个块
        offs_a += bk * stride_ak
        offs_b += bk * stride_bk
    c = c_ptr + get_2d_offset(rm, rn, stride_cm, stride_cn)  # 计算 c 的偏移量
    mask = get_2d_mask(rm, rn, m, n)  # 计算掩码
    tl.store(c, acc, mask=mask)  # 将结果存储到 c 中
```

```python
from functools import partial

def matmul(a, b, matmul_k_fn, bs=16, group_sz=None):
    # 检查矩阵维度是否兼容
    assert a.shape[1] == b.shape[0], "矩阵维度不兼容，无法进行矩阵乘法"
    # 检查张量是否准备好在 GPU 上运行
    check_tensors_gpu_ready(a, b)
    # 获取矩阵 a 和 b 的形状
    (m, k), (_, n) = a.shape, b.shape
    # 创建一个空的输出张量 c
    c = torch.empty((m, n), device=a.device, dtype=torch.float16)
    # 定义网格函数，用于计算线程块的数量
    grid = lambda meta: (triton.cdiv(m, meta['bm']),  triton.cdiv(n, meta['bn']))
    # 处理 group_sz 参数，如果为 None，则使用空字典
    group_sz = {} if group_sz is None else {"group_sz":group_sz} # 在 naive_matmul 中未使用，但在后续的 grouped_matmul 中会用到
    # 调用 matmul_k_fn 函数，传入必要的参数
    matmul_k_fn[grid](
        a, b, c,
        m, n, k,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        bm=bs, bn=bs, bk=bs, # 注意：对于较旧的 GPU，allow_tf32 必须设置为 False，否则无法编译
        **group_sz
    )
    # 返回计算结果
    return c

# 使用 partial 创建一个部分应用的函数 naive_matmul
naive_matmul = partial(matmul, matmul_k_fn=naive_matmul_k)
```

```python
a = torch.ones((3, 4), dtype=torch.float32, device='cuda')
b = torch.ones((4, 5), dtype=torch.float32, device='cuda')
```

```python
naive_matmul(a,b)
```

```shell
tensor([[4., 4., 4., 4., 4.],
        [4., 4., 4., 4., 4.],
        [4., 4., 4., 4., 4.]], device='cuda:0', dtype=torch.float16)
```

让我们对 PyTorch 的实现进行单元测试

```python
torch.manual_seed(0)
a = torch.randn((512, 512), device='cuda', dtype=torch.float16)
b = torch.randn((512, 512), device='cuda', dtype=torch.float16)
triton_output = naive_matmul(a, b)
torch_output = torch.matmul(a, b)
if torch.allclose(triton_output, torch_output, atol=5e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")
```

✅ Triton and Torch match

# 示例 4：更快的矩阵乘法

Triton 处理块内的内存访问顺序，但不处理跨块的内存访问顺序。因此，这是一个我们可以用来加速内核的调节点。

事实上，巧妙地重新排序块可以提高 L2 缓存的命中率，从而使我们的内核更快。这个示例来自 [Triton 文档](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)。
现在，为了更好地利用 L2 缓存，我们希望重用最近加载的数据，这些数据很可能仍然在 L2 缓存中。如何实现？通过减少一批“连续”内核需要的不同数据加载次数。我们所说的“连续”是指大约在同一时间执行的内核。

这张图（改编自 [Triton 文档](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)）展示了我们如何做到这一点。如果按朴素顺序排列，输出矩阵的第一行将“连续”计算，这需要 90 次不同的块读取（矩阵 A 中 9 次，矩阵 B 中 81 次）。如果我们使用“分组排序”，输出矩阵的 3x3 块将“连续”计算，这需要 54 次不同的块读取（矩阵 A 中 27 次，矩阵 B 中 27 次）。

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-2/5496ef37-6b0c-4cfc-b836-94e9c83ee1f1.png)

注意：在文档中，分组称为“super-grouping”。
好的，我们如何告诉 Triton 以何种顺序处理块？答案是：我们获取 pids，改变它们，并将它们用作原始 pids。

让我们通过一个最小示例来说明这一原则：

```python
def process_item(id): print(f"I'm processing item {id}")

for i in range(5): process_item(i)
```

```shell
I'm processing item 0
I'm processing item 1
I'm processing item 2
I'm processing item 3
I'm processing item 4
```

```python
def change_id(old_id): return 5-old_id

for i in range(5): process_item(change_id(i))
```

```shell
I'm processing item 5
I'm processing item 4
I'm processing item 3
I'm processing item 2
I'm processing item 1
```

就这样，项目以不同的顺序处理了。

那么，用于更快矩阵乘法的 pid 变换函数应该是什么样的？它应该将左矩阵转换为右矩阵。

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/c6b84de9-6cca-4393-83b1-747e80b78eea.png)

在左侧，显示了默认的顺序（称为“行优先”）。请注意，我们处理的是块。我们无法安排单个单元格的处理顺序，只能安排块的顺序。在图中，我们的输出矩阵 C 有 `5x7 = 35` 个单元格，但只有 `cdiv(5,1) x cdiv(7,2) = 5x4 = 20` 个块。

在右侧，注意前 9 个处理的块是我们想要的 `3x3` 网格！我们在一列中处理 3 个块。然后前进一列，再次处理 3 个块，如此循环。橙色线显示了前进的位置。这个操作称为 **"swizzling"**。

顺便说一下，你可以当然改变数字 3。它被称为 `group_size`。

你不需要自己编写 swizzling，因为 Triton 提供了一个 `triton.language.swizzle2d` 函数。

为了真正理解 `swizzle2d`，我们快速验证它是否按预期工作。然后我们将在更快的矩阵乘法kernel中继续使用它。

附带目标：在一个 `5x4` 的矩阵上使用 `swizzle2d`，该矩阵的元素按行优先顺序排列为 `0 ... 19`。我们应该得到一个元素按分组顺序排列的矩阵。

```python
@triton.jit
def swizzle_k(x_ptr, z_ptr, group_sz: tl.constexpr):
    # 获取当前线程块的 ID
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    # 获取线程块的总数
    num_pid_m, num_pid_n = tl.num_programs(0), tl.num_programs(1)

    # 使用 Triton 的 swizzle2d 函数重新排列线程块的 ID
    # 注意：在 CPU 模拟时，tl.swizzle2d 可能无法正常工作
    pid_m_, pid_n_ = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, group_sz)
    
    # 计算原始线程块的偏移量
    offs_m = get_1d_offset(1, n_prev_chunks=pid_m)
    offs_n = get_1d_offset(1, n_prev_chunks=pid_n)
    
    # 计算原始线程块的 2D 偏移量和掩码
    offs = get_2d_offset(offs_m, offs_n, stride_0=num_pid_n)
    mask = get_2d_mask(offs_m, offs_n, max_0=num_pid_m, max_1=num_pid_n )

    # 计算重新排列后的线程块的偏移量
    offs_sw_m = get_1d_offset(1, n_prev_chunks=pid_m_)
    offs_sw_n = get_1d_offset(1, n_prev_chunks=pid_n_)
    
    # 计算重新排列后的线程块的 2D 偏移量和掩码
    offs_sw = get_2d_offset(offs_sw_m, offs_sw_n, stride_0=num_pid_n)
    mask_sw = get_2d_mask(offs_sw_m, offs_sw_n, max_0=num_pid_m, max_1=num_pid_n)
    
    # 从原始矩阵中加载数据
    x = tl.load(x_ptr + offs, mask=mask)
    # 将数据存储到重新排列后的矩阵中
    tl.store(z_ptr + offs_sw, x, mask=mask_sw)
```

```python
blocks_m, blocks_n = 5,4

x = torch.arange(blocks_m*blocks_n, device='cuda').view(blocks_m,blocks_n)
x
```

```shell
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11],
        [12, 13, 14, 15],
        [16, 17, 18, 19]], device='cuda:0')
```

```python
z = -torch.ones_like(x) # empty matrix, with -1 denoting empty
z
```

```shell
tensor([[-1, -1, -1, -1],
        [-1, -1, -1, -1],
        [-1, -1, -1, -1],
        [-1, -1, -1, -1],
        [-1, -1, -1, -1]], device='cuda:0')
```

```python
# swizzle x into z
swizzle_k[(blocks_m,blocks_n)](x,z, group_sz=3);
z
```

```shell
tensor([[ 0,  3,  6,  9],
        [ 1,  4,  7, 10],
        [ 2,  5,  8, 11],
        [12, 14, 16, 18],
        [13, 15, 17, 19]], device='cuda:0')
```

看起来不错！

___


现在我们来实现 grouped 矩阵乘法kernel，这将比普通的矩阵乘法更快。

```python
@triton.jit
def grouped_matmul_k(
    a_ptr, b_ptr, c_ptr,  # 指向矩阵 A, B, C 的指针
    m, n, k,  # 矩阵的维度
    stride_am, stride_ak,  # 矩阵 A 的步长
    stride_bk, stride_bn,  # 矩阵 B 的步长
    stride_cm, stride_cn,  # 矩阵 C 的步长
    bm: tl.constexpr, bn: tl.constexpr, bk: tl.constexpr, group_sz: tl.constexpr  # 块大小和分组大小
):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)  # 获取当前线程块的 ID
    num_pid_m, num_pid_n = tl.num_programs(0), tl.num_programs(1)  # 获取线程块的总数
    # 确定块在分组排序中的位置 - 重新排列！
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, group_sz)  # 奇怪的地方：tl.swizzle2d 在 CPU 模拟时不起作用
    # 沿 m/n/k 维度的块
    rm = get_1d_offset(size=bm, n_prev_chunks=pid_m)  # 计算 m 维度的偏移
    rn = get_1d_offset(size=bn, n_prev_chunks=pid_n)  # 计算 n 维度的偏移
    rk = get_1d_offset(size=bk, n_prev_chunks=0)  # 计算 k 维度的偏移
    # 矩阵 A 和 B 的相关偏移
    offs_a = a_ptr + get_2d_offset(rm, rk, stride_am, stride_ak)  # 计算矩阵 A 的偏移
    offs_b = b_ptr + get_2d_offset(rk, rn, stride_bk, stride_bn)  # 计算矩阵 B 的偏移
    # 初始化并迭代更新累加器
    acc = tl.zeros((bm, bn), dtype=tl.float32)  # 初始化累加器
    for _ in range(0, k, bk):
        # todo umer: 加载 a & b 时是否需要掩码？
        a = tl.load(offs_a)  # 加载矩阵 A 的块
        b = tl.load(offs_b)  # 加载矩阵 B 的块
        acc += tl.dot(a, b, allow_tf32=False)  # 块级别的矩阵乘法；奇怪的地方：对于较旧的 GPU，allow_tf32 必须设置为 False，否则无法编译
        # 增加偏移，以便下一次迭代加载下一个块
        offs_a += bk * stride_ak
        offs_b += bk * stride_bk
    c = c_ptr + get_2d_offset(rm, rn, stride_cm, stride_cn)  # 计算矩阵 C 的偏移
    mask = get_2d_mask(rm, rn, m, n)  # 计算掩码
    tl.store(c, acc, mask=mask)  # 将累加器的结果存储到矩阵 C 中
```

```python
grouped_matmul = partial(matmul, matmul_k_fn=grouped_matmul_k)
```

```python
a = torch.ones((3, 4), dtype=torch.float32, device='cuda')
b = torch.ones((4, 5), dtype=torch.float32, device='cuda')
```

```python
grouped_matmul(a,b, group_sz=4)
```

```shell
tensor([[4., 4., 4., 4., 4.],
        [4., 4., 4., 4., 4.],
        [4., 4., 4., 4., 4.]], device='cuda:0', dtype=torch.float16)
```

让我们对 PyTorch 的实现进行单元测试

```python
torch.manual_seed(0)
a = torch.randn((512, 512), device='cuda', dtype=torch.float16)
b = torch.randn((512, 512), device='cuda', dtype=torch.float16)
triton_output = grouped_matmul(a, b, group_sz=32)
torch_output = torch.matmul(a, b)
if torch.allclose(triton_output, torch_output, atol=5e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")
```

✅ Triton and Torch match

# 性能测试

Triton 自带性能测试工具。以下是一个使用示例。

```python
# adapted from https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['square_matrix_size'],  # 用于绘图的 x 轴参数名称。
        x_vals=[2**i for i in range(5, 12, 1)],  # `x_name` 的不同可能值。
        x_log=True,  # x 轴为对数刻度。
        line_arg='provider',  # 对应于绘图中不同线条的参数名称。
        line_vals=['naive', 'grouped', 'torch'],  # `line_arg` 的可能值。
        line_names=['Naive', 'Grouped', 'Torch'],  # 线条的标签名称。
        styles=[('blue', '-'), ('green', '-'), ('orange','-')],  # 线条样式。
        ylabel='GB/s',  # y 轴的标签名称。
        plot_name='matmul-performance',  # 绘图的名称，也用作保存绘图的文件名。
        args={},  # 不在 `x_names` 和 `y_name` 中的函数参数值。
    ))
def benchmark(square_matrix_size, provider):
    sz = square_matrix_size  # 矩阵的大小
    a = torch.rand((sz, sz), device='cuda', dtype=torch.float32)  # 生成随机矩阵 a
    b = torch.rand((sz, sz), device='cuda', dtype=torch.float32)  # 生成随机矩阵 b
    quantiles = [0.5, 0.2, 0.8]  # 用于性能测试的分位数
    if provider == 'naive':  # 如果使用 naive 方法
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_matmul(a, b), quantiles=quantiles)  # 执行性能测试
    if provider == 'grouped':  # 如果使用 grouped 方法
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: grouped_matmul(a, b, group_sz=8), quantiles=quantiles)  # 执行性能测试
    if provider == 'torch':  # 如果使用 PyTorch 方法
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)  # 执行性能测试
    gbps = lambda ms: 12 * sz / ms * 1e-6  # 计算带宽（GB/s）
    return gbps(ms), gbps(max_ms), gbps(min_ms)  # 返回带宽值
```

> 个人感觉这里的gbps公式有错误，应该是12 * sz^2 / ms * 1e-6 才对？下面给出了Deepseek v2.5的推导：

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/b5c9cade-b0b5-4fe9-b39e-7ae1ed8d0d21.png)

```python
benchmark.run(print_data=True, show_plots=True)
```

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-2/6d39aaa8-8b02-4587-9a4a-f9ea211ac010.png)

```shell
matmul-performance:
   square_matrix_size     Naive   Grouped     Torch
0                32.0  0.085106  0.085106  0.053691
1                64.0  0.129730  0.125000  0.107143
2               128.0  0.159468  0.154341  0.170515
3               256.0  0.097909  0.099071  0.125654
4               512.0  0.030346  0.030361  0.111079
5              1024.0  0.006971  0.007279  0.034461
6              2048.0  0.001405  0.001749  0.006355
```

注 Umer: 我本以为随着矩阵大小的增加，GB/s 会增加。为什么没有？可能是因为共享内存已满，所以kernel花费了越来越多的时间重新加载数据。

让我们尝试不同的块大小：

```python
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['batch_size'], x_vals=[2**i for i in range(4, 7, 1)], x_log=True,
        line_arg='provider', line_vals=['naive', 'grouped', 'torch'], line_names=['Naive', 'Grouped', 'Torch'],
        styles=[('blue', '-'), ('green', '-'), ('orange','-')],
        ylabel='GB/s', plot_name='matmul-performance', args={}
    ))
def benchmark(batch_size, provider):
    sz = 512
    a = torch.rand((sz, sz), device='cuda', dtype=torch.float32)
    b = torch.rand((sz, sz), device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'naive':   ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_matmul(a, b, bs=batch_size), quantiles=quantiles)
    if provider == 'grouped': ms, min_ms, max_ms = triton.testing.do_bench(lambda: grouped_matmul(a, b, bs=batch_size, group_sz=8), quantiles=quantiles)
    if provider == 'torch':   ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a,b), quantiles=quantiles)
    gbps = lambda ms: 12 * sz / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark.run(print_data=True, show_plots=True)
```

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/d6839175-8259-4a99-9ff8-c75b2ea74f4d.png)

```shell
matmul-performance:
   batch_size     Naive   Grouped     Torch
0        16.0  0.030404  0.030433  0.111111
1        32.0  0.060683  0.061127  0.111111
2        64.0  0.083660  0.084026  0.111111
```

更大的块大小似乎更好。让我们再次与 PyTorch 进行比较，使用更大的块大小。

```python
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['square_matrix_size'], x_vals=[2**i for i in range(5, 12, 1)], x_log=True,
        line_arg='provider', line_vals=['naive', 'grouped', 'torch'], line_names=['Naive', 'Grouped', 'Torch'],
        styles=[('blue', '-'), ('green', '-'), ('orange','-')],
        ylabel='GB/s', plot_name='matmul-performance', args={}
    ))
def benchmark(square_matrix_size, provider):
    sz = square_matrix_size
    a = torch.rand((sz, sz), device='cuda', dtype=torch.float32)
    b = torch.rand((sz, sz), device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'naive':   ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_matmul(a, b, bs=64), quantiles=quantiles)
    if provider == 'grouped': ms, min_ms, max_ms = triton.testing.do_bench(lambda: grouped_matmul(a, b, group_sz=8, bs=64), quantiles=quantiles)
    if provider == 'torch':   ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a,b), quantiles=quantiles)
    gbps = lambda ms: 12 * sz / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark.run(print_data=True, show_plots=True)
```

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/eb9d5d16-376d-4cbc-b575-be02d3b5997d.png)

```shell
matmul-performance:
   square_matrix_size     Naive   Grouped     Torch
0                32.0  0.039867  0.038710  0.053215
1                64.0  0.077922  0.071006  0.106667
2               128.0  0.109091  0.107143  0.169912
3               256.0  0.137733  0.136364  0.126150
4               512.0  0.084731  0.083916  0.111047
5              1024.0  0.021879  0.025362  0.034691
6              2048.0  0.005257  0.005919  0.007440
```

这减少了较大矩阵尺寸下与 PyTorch 的性能差距，但 PyTorch 仍然更好。

提示：对于性能分析，我们可以使用 Nsight Compute 来分析我们的kernel：
`ncu --target-processes all your_python_file.py`

# 自动调优

改编自 https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html

元参数（例如块大小）和编译选项（例如 `num_warps`）的选择会影响kernel的速度。Triton 允许你传递一个可能选择的列表，运行所有这些选择，然后为最快的选择编译kernel。这称为 `自动调优`。

如果问题的大小发生变化（例如矩阵大小变化），将为新的问题大小进行新的自动调优。

```python
@triton.autotune(
    # Choices of configs to auto-tune over
    configs=[
        triton.Config({'bm': 128, 'bn': 256, 'bk': 64, 'group_sz': 8}, num_stages=3, num_warps=8),
        triton.Config({'bm': 64, 'bn': 256, 'bk': 32, 'group_sz': 8}, num_stages=4, num_warps=4),
        triton.Config({'bm': 128, 'bn': 128, 'bk': 32, 'group_sz': 8}, num_stages=4, num_warps=4),
        triton.Config({'bm': 128, 'bn': 64, 'bk': 32, 'group_sz': 8}, num_stages=4, num_warps=4),
        triton.Config({'bm': 64, 'bn': 128, 'bk': 32, 'group_sz': 8}, num_stages=4, num_warps=4),
        triton.Config({'bm': 128, 'bn': 32, 'bk': 32, 'group_sz': 8}, num_stages=4, num_warps=4),
        triton.Config({'bm': 64, 'bn': 32, 'bk': 32, 'group_sz': 8}, num_stages=5, num_warps=2),
        triton.Config({'bm': 32, 'bn': 64, 'bk': 32, 'group_sz': 8}, num_stages=5, num_warps=2),
    ],
    # Definition of problem size. If it changes, a new auto-tune is run for the new problem size.
    key=['m', 'n', 'k'],
)
@triton.jit
def grouped_autotuned_matmul_k(
    a_ptr, b_ptr, c_ptr,
    m, n, k,
    stride_am, stride_ak, 
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    bm: tl.constexpr, bn: tl.constexpr, bk: tl.constexpr, group_sz: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    num_pid_m = tl.num_programs(0)
    num_pid_n = tl.num_programs(1)
    # determine location of block in grouped ordering
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, group_sz)  # Weirdness: tl.swizzle2d doesn't work when simulating on CPU
    # chunks along m/n/k dimensions
    rm = get_1d_offset(size=bm, n_prev_chunks=pid_m)
    rn = get_1d_offset(size=bn, n_prev_chunks=pid_n)
    rk = get_1d_offset(size=bk, n_prev_chunks=0)
    # relevant offsets of a, b
    offs_a = a_ptr + get_2d_offset(rm, rk, stride_am, stride_ak)
    offs_b = b_ptr + get_2d_offset(rk, rn, stride_bk, stride_bn)
    # initialize and iteratively update accumulator
    acc = tl.zeros((bm, bn), dtype=tl.float32)
    for _ in range(0, k, bk):
        # todo umer: don't we need mask when loading a & b?
        a = tl.load(offs_a)
        b = tl.load(offs_b)
        acc += tl.dot(a, b, allow_tf32=False) # block level matrix multiplication ; Weirdness: allow_tf32 must be set to False for older GPUs, otherwise won't compile
        # increase offets, so next iteration loads next chunks
        offs_a += bk * stride_ak
        offs_b += bk * stride_bk
    c = c_ptr + get_2d_offset(rm, rn, stride_cm, stride_cn)
    mask = get_2d_mask(rm, rn, m, n)
    tl.store(c, acc, mask=mask)

def grouped_autotuned_matmul(a, b):
    matmul_k_fn = grouped_autotuned_matmul_k
    
    assert a.shape[1] == b.shape[0], "matrix dims not compatible for matmul"
    check_tensors_gpu_ready(a, b)
    (m, k), (_, n) = a.shape, b.shape
    c = torch.empty((m, n), device=a.device, dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(m, meta['bm']),  triton.cdiv(n, meta['bn']))
    matmul_k_fn[grid](
        a, b, c,
        m, n, k,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        # bm=bs, bn=bs, bk=bs, <- will be autotuned
        # **group_sz <- will be autotuned
    )
    return c
```

```
a,b = torch.ones(3,4, device='cuda'), torch.ones(4,5, device='cuda')
a@b
```

```shell
tensor([[4., 4., 4., 4., 4.],
        [4., 4., 4., 4., 4.],
        [4., 4., 4., 4., 4.]], device='cuda:0')
```

注意：有时以下行会返回错误的结果，而且我无法可靠地重现这个问题。如果您能重现，请通过 Twitter (@UmerHAdil) 告诉我！🙏🏽

```python
grouped_autotuned_matmul(a,b)
```

```shell
tensor([[4., 4., 4., 4., 4.],
        [4., 4., 4., 4., 4.],
        [4., 4., 4., 4., 4.]], device='cuda:0', dtype=torch.float16)
```

关于自动调优的配置建议、技巧和启发式方法，请参见 [Mark Saroufim 的演讲 "CUDA Performance Checklist"](https://www.youtube.com/watch?v=SGhfUhlowB4)。其中的许多内容也适用于 Triton。

让我们再次运行基准测试。这将花费很多时间，因为我们将为每个基准测试参数选择进行自动调优（即，对我们来说是 12-5=7 次）。

```python
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['square_matrix_size'], x_vals=[2**i for i in range(5, 12, 1)], x_log=True,
        line_arg='provider', line_vals=['naive', 'grouped', 'grouped-autotuned', 'torch'], line_names=['Naive', 'Grouped', 'Grouped & Auto-Tuned','Torch'],
        styles=[('blue', '-'), ('green', '-'), ('green', '--'), ('orange','-')],
        ylabel='GB/s', plot_name='matmul-performance', args={}
    ))
def benchmark(square_matrix_size, provider):
    sz = square_matrix_size
    a = torch.rand((sz, sz), device='cuda', dtype=torch.float32)
    b = torch.rand((sz, sz), device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'naive':   ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_matmul(a, b, bs=64), quantiles=quantiles)
    if provider == 'grouped': ms, min_ms, max_ms = triton.testing.do_bench(lambda: grouped_matmul(a, b, group_sz=8, bs=64), quantiles=quantiles)
    if provider == 'grouped-autotuned': ms, min_ms, max_ms = triton.testing.do_bench(lambda: grouped_autotuned_matmul(a, b), quantiles=quantiles)
    if provider == 'torch':   ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a,b), quantiles=quantiles)
    gbps = lambda ms: 12 * sz / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark.run(print_data=True, show_plots=True)
```

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/d5f8a2ad-2729-47db-9db1-2201271bdd63.png)

```shell
matmul-performance:
   square_matrix_size     Naive   Grouped  Grouped & Auto-Tuned     Torch
0                32.0  0.040067  0.037500              0.062176  0.054795
1                64.0  0.077170  0.074303              0.091954  0.104803
2               128.0  0.110218  0.107143              0.117936  0.169912
3               256.0  0.139738  0.136364              0.137339  0.126482
4               512.0  0.083953  0.082937              0.066864  0.110983
5              1024.0  0.023112  0.025932              0.020007  0.033520
6              2048.0  0.005235  0.005912              0.004629  0.007076
```

___
<h1>这就是全部内容！恭喜你完成了本教程 - Good work！🥳</h1>

我强烈建议你自己编写一些 Triton kernel。例如，你可以尝试这些 Triton 谜题：https://github.com/srush/Triton-Puzzles，由 [Sasha Rush](https://twitter.com/srush_nlp)、Tejas Ramesh 和 [Keren Zhou](https://twitter.com/ZhouKeren) 提供。

这里有一些中级和高级材料：
- 官方文档：https://triton-lang.org/
- LightLLM 仓库包含了许多实际的 Triton kernel：https://github.com/ModelTC/lightllm/tree/main/lightllm/common/basemodel/triton_kernel
- Unsloth 仓库也包含了许多实际的 Triton kernel：https://github.com/unslothai/unsloth/tree/main/unsloth/kernels
如果你对 GPU 编程和性能优化感兴趣，[cuda mode Discord](https://discord.gg/cudamode) 可能对你有帮助。本教程是作为他们精彩的 [讲座系列](https://www.youtube.com/@CUDAMODE) 的一部分编写的。


