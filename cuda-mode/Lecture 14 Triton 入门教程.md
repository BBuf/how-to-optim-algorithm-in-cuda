<h1><b></v>Triton 实践指南</b></h1>

作者：UmerHA (https://x.com/UmerHAdil // https://github.com/UmerHA/)，为 cuda-mode 小组编写 ❤️ May our brrrr level reach over 。

# 为什么以及何时使用Triton

**什么是Triton**
简而言之：Triton是一种更方便地编程GPU的语言。你编写类似Python的代码，然后这些代码被编译成ptx代码（与cuda代码编译成的代码相同）。

在编译过程中，Triton编译器尝试使用巧妙的技巧来重新排列程序的部分内容（不改变程序的意义！）以使其运行得更快。

**Triton vs Cuda**

![](https://files.mdnice.com/user/59/dda26356-ad90-415d-8cbe-5b5b4bad930d.png)
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

![](https://files.mdnice.com/user/59/29863a80-c092-4d8d-a7de-0ecdb2a9213c.png)

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

![](https://files.mdnice.com/user/59/8dd01a33-89cc-4587-8376-b2471600c63a.png)

术语说明：在Triton术语中，每个处理块的kernel被称为“program”。也就是说，我们上面的例子运行了2个program。因此，“block_id”通常被称为“pid”（“program id”的缩写），但它们是相同的。

# Example 1: Copying a tensor

Let's looks at some examples. To keeps things simple, we'll use very small block sizes.

Goal: Given a tensor `x` of shape (n), copy it into another tensor `z`.

```python
# # This is a normal python function, which launches the triton kernels
def copy(x, bs, kernel_fn):
    z = torch.zeros_like(x)
    check_tensors_gpu_ready(x, z)
    n = x.numel()
    n_blocks = cdiv(n, bs)
    grid = (n_blocks,)  # how many blocks do we have? can be 1d/2d/3d-tuple or function returning 1d/2d/3d-tuple

    # launch grid!
    # - kernel_fn is the triton kernel, which we write below
    # - grid is the grid we constructed above
    # - x,z,n,bs are paramters that are passed into each kernel function
    kernel_fn[grid](x,z,n,bs)

    return z
```

**Note:** For educational purposes, the kernel below has a logic bug (but the syntax is correct). Can you spot it?

```python
# # This is the triton kernel:

# The triton.jit decorator takes a python function and turns it into a triton kernel, which is run on the GPU.
# Inside this function only a subset of all python ops are allowed.
# E.g., when NOT simulating, we can't print or use breakpoints, as these don't exist on the GPU. 
@triton.jit
# When we pass torch tensors, they are automatically converted into a pointer to their first value
# E.g., above we passed x, but here we receive x_ptr
def copy_k(x_ptr, z_ptr, n, bs: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, bs)  # compute the offsets from the pid 
    mask = offs < n
    x = tl.load(x_ptr + offs, mask) # load a vector of values, think of `x_ptr + offs` as `x_ptr[offs]`
    tl.store(z_ptr + offs, x, mask) # store a vector of values

    print_if(f'pid = {pid} | offs = {offs}, mask = {mask}, x = {x}', '')

    # Question: What is wrong with this kernel?
```

```python
z = copy(x, bs=2, kernel_fn=copy_k)
```

```
z
```

```shell
tensor([1, 2, 0, 0, 0, 0])
```

We were not shifting the offets correcltly. We always used offsets = [0,1], but they should change with the pid.

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

Not quite correct. We added `pid * n`, but want to add `pid * bs`

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

As we saw, writing GPU programs involves many indices, which we can easily mess up. So I highly recommend writing and debugging the kernel in simuation mode, and testing with tiny examples first!


# Example 2: Greyscaling an image

_Restart kernel here_
In this example, we'll grayscale an image of a puppy. We'll see how we can work on 2d data.

This works analogously for 3D data.

We've adapted Jeremy Howard's example from this [colab](https://colab.research.google.com/drive/180uk6frvMBeT4tywhhYXmz3PJaCIA_uk?usp=sharing) / [youtube](https://www.youtube.com/watch?v=4sgKnKbR-WE&feature=youtu.be). So, h/t for the example and selection of puppy image.
_Side note: Two weird things happen in this example, if we don't restart the kernel:_
1. _torchvision can't be imported, probably due to a circular dependency. -> I currently don't know why, need to dig deeper._
2. _the simulated triton kernel below fails, because a float can't be mutliplied to a uint vector -> Works on GPU w/o simulation, so seems to be a `TRITON_INTERPRET` bug._
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
def show_img(x, figsize=(4,3), **kwargs):
    plt.figure(figsize=figsize)
    plt.axis('off')
    if len(x.shape)==3: x = x.permute(1,2,0)  # CHW -> HWC
    plt.imshow(x.cpu(), **kwargs)
img = tvf.resize(img, 150, antialias=True)
ch,h,w = img.shape
ch,h,w,h*w
show_img(img)
To work with 2d data, we'll build 2d offsets and masks. Here's an illustration how it works, e.g. for an `4x7` matrix and block sizes of `2` for each dimensions.
<img src='images/4_offset_2d.png'>
And in code, it looks like this:
@triton.jit
def rgb2grey_k(x_ptr, out_ptr, h, w, bs0: tl.constexpr, bs1: tl.constexpr):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)
    
    offs_0 = pid_0 * bs0 + tl.arange(0,bs0)  # 1d vector
    offs_1 = pid_1 * bs1 + tl.arange(0,bs1)  # 1d vector

    # Weirdness: None-slicing currently doesn't work when simulating on cpu. Use tl.expand_dim instead.
    # offs = w * tl.expand_dims(offs_0, 1) + tl.expand_dims(offs_1, 0)
    offs = w * offs_0[:,None] + offs_1[None, :]  # 2d matrix! - we multiply first offset by width, see image above

    mask_0 = offs_0 < h  # 1d vector
    mask_1 = offs_1 < w  # 1d vector

    # mask = tl.expand_dims(mask_0, 1) & tl.expand_dims(mask_1, 0)
    mask = mask_0[:,None] & mask_1[None,:]  # 2d matrix! - data musn't go out of bounds along either axis, therefore `logical and` of the individual masks
    
    r = tl.load(x_ptr + 0*h*w+offs, mask=mask)
    g = tl.load(x_ptr + 1*h*w+offs, mask=mask)
    b = tl.load(x_ptr + 2*h*w+offs, mask=mask)

    # Weirdness: multiplying float with uint vectors fails when simulating on cpu
    out = 0.2989*r + 0.5870*g + 0.1140*b  # don't worry why it's these 3 numbers we're multiplying with

    tl.store(out_ptr + offs, out, mask=mask)
Let's use the kernel!
def rgb2grey(x, bs):
    c,h,w = x.shape
    out = torch.empty((h,w), dtype=x.dtype, device=x.device)

    # grid can be a function returning a 1d/2d/3d-tuple
    # (having a grid function is not more useful than a grid tuple in this case, but will be below when benchmarking & auto-tuning)
    grid = lambda meta: (cdiv(h, meta['bs0']), cdiv(w,  meta['bs1']))
    
    rgb2grey_k[grid](x, out, h, w, bs0=bs[0], bs1=bs[1]) # all kwargs are passed into grid function
    return out.view(h,w)
grey_img = rgb2grey(img.to('cuda'), bs=(32, 32)).to('cpu')
show_img(grey_img, cmap='gray')
Very cool


# Example 3: Matmul
_For simplicity, restart kernel here_
import os
# os.environ['TRITON_INTERPRET'] = '1'

import torch
import triton
import triton.language as tl

# moved util functions into separate file for better readability
from triton_util import cdiv, breakpoint_if, print_if, check_tensors_gpu_ready
Now, let's implement a naive matmul in Triton. We'll learn:
- A method to split computation 
- Calling functions from our kernel 
- Using pre-implemented vector/matrix ops within an block

This is adapted from the [OpenAI blog post announcing Triton](https://openai.com/research/triton).

We want to multiply the `m x k`-matrix `A` and the `k x n`-matrix `B` into the `m x n`-matrix `C`.

We split the computation along each of the three axes:
- along the m axis - we'll use block dimension 0 to represent this
- along the n axis - we'll use block dimension 1 to represent this
- along the shared k axis - this will not be represented by a block. All chunks of computation will be done in same block.
<img src='images/5_matmul_split.png'>
Because we frequently create 1d- or 2d-offets and -masks, let's put that functionality into utility functions. As long as these functions are `triton.jit`-ed, they can be used in the kernel.
@triton.jit
def get_1d_offset(size, n_prev_chunks):
    return n_prev_chunks * size + tl.arange(0, size)

@triton.jit
def get_2d_offset(offs_0, offs_1, stride_0, stride_1=1): 
    return tl.expand_dims(offs_0, 1)*stride_0 + tl.expand_dims(offs_1, 0)*stride_1

@triton.jit
def get_1d_mask(offs, max):
    return offs < max

@triton.jit
def get_2d_mask(offs_0, offs_1, max_0, max_1):
    return (tl.expand_dims(offs_0, 1) < max_0) & (tl.expand_dims(offs_1, 0) < max_1)
Here's the naive matmul kernel:
@triton.jit
def naive_matmul_k(
    a_ptr, b_ptr, c_ptr,
    m, n, k,
    stride_am, stride_ak, 
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    bm: tl.constexpr, bn: tl.constexpr, bk: tl.constexpr
):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
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
        acc += tl.dot(a, b, allow_tf32=False) # matmul in block ; Weirdness: allow_tf32 must be set to False for older GPUs, otherwise won't compile
        # increase offets, so next iteration loads next chunks
        offs_a += bk * stride_ak
        offs_b += bk * stride_bk
    c = c_ptr + get_2d_offset(rm, rn, stride_cm, stride_cn)
    mask = get_2d_mask(rm, rn, m, n)
    tl.store(c, acc, mask=mask)
from functools import partial

def matmul(a, b, matmul_k_fn, bs=16, group_sz=None):
    assert a.shape[1] == b.shape[0], "matrix dims not compatible for matmul"
    check_tensors_gpu_ready(a, b)
    (m, k), (_, n) = a.shape, b.shape
    c = torch.empty((m, n), device=a.device, dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(m, meta['bm']),  triton.cdiv(n, meta['bn']))
    group_sz = {} if group_sz is None else {"group_sz":group_sz} # not used in naive_matmul, but will be in grouped_matmul further below 
    matmul_k_fn[grid](
        a, b, c,
        m, n, k,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        bm=bs, bn=bs, bk=bs, # Weirdness: allow_tf32 must be set to False for older GPUs, otherwise won't compile
        **group_sz
    )
    return c

naive_matmul = partial(matmul, matmul_k_fn=naive_matmul_k)
a = torch.ones((3, 4), dtype=torch.float32, device='cuda')
b = torch.ones((4, 5), dtype=torch.float32, device='cuda')
naive_matmul(a,b)
Let's unit test this against PyTorch's implementation
torch.manual_seed(0)
a = torch.randn((512, 512), device='cuda', dtype=torch.float16)
b = torch.randn((512, 512), device='cuda', dtype=torch.float16)
triton_output = naive_matmul(a, b)
torch_output = torch.matmul(a, b)
if torch.allclose(triton_output, torch_output, atol=5e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")


# Example 4: Faster Matmul
_Note: Needs code from example 3, so run that before_
Triton handles the order of memory access **within** blocks, but not **across** blocks. So this is a knob we can use to make our kernels faster.

In fact, cleverly reordering blocks can increase L2-cache hit rate, which makes our kernels faster. This example is taken from the [triton docs](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html).
Now, to make better use of the L2 cache, we want to reuse data that's was recently loaded, and is therefore likely still in the L2 cache. How? By reducing the number of _different_ data loads that a bunch of "consecutive" kernels need. By "consecutive" we mean kernels that are executed approximately at the same time.

This picture (adapter from the [triton docs](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)) shows how we can do that. If we order naively, the first row of the output matrix is computed "consecutively", which needs 90 different block reads (9 from matrix A, 81 from matrix B). If we use "group ordering", a 3x3 square of blocks of the output matrix is computed "consecutively", which needs 54 different block reads (27 from matrix A, 27 from matrix B).
<img src='images/6_matmul_order.png'>
_Note: In the docs, grouping is called "super-grouping"_
Okay, how can we tell Triton in which order to process blocks? The answer is: We take the pids, change them, and use them as if they were the original pids.

Let's do a minimal example to illustrate this principle:
def process_item(id): print(f"I'm processing item {id}")

for i in range(5): process_item(i)
def change_id(old_id): return 5-old_id

for i in range(5): process_item(change_id(i))
Et voilà, the items were processed in a different order.

So how should the pid-change-function for faster matmul look like? It should change the left matrix into the right matrix.
<img src='images/7_swizzling_exmple.png'>
On the left, the default ordering is shown (called "row-major"). Remember, we deal with blocks. We can't arrange how the individual cells are processed, only the blocks. In the picture, our output matrix C has `5x7 = 35` cells, but only `cdiv(5,1) x cdiv(7,2) = 5x4 = 20` blocks.

On the right, notice how the first 9 processed blocks are the `3x3` grid we want! We process 3 blocks in a column. Then advance a column, again process 3, advance, and so on. The orange lines show where advance. This operation is called **"swizzling"**.

By the way, you can of course change the number 3. It's called the `group_size`.

You don't need to write swizzling yourself, as  there is a `triton.language.swizzle2d` function.

To really understand `swizzle2d`, let's quickly verifiy it works as expected. We'll then continue to use it in our faster matmul kernel.
_Side-Goal:_ Use `swizzle2d` on a `5x4` matrix with elements `0 ... 19` in row-major ordering. We should then get a matrix with elements in grouped ordering.
@triton.jit
def swizzle_k(x_ptr, z_ptr, group_sz: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    num_pid_m, num_pid_n = tl.num_programs(0), tl.num_programs(1)

    pid_m_, pid_n_ = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, group_sz)  # Weirdness: tl.swizzle2d doesn't work when simulating on CPU
    
    offs_m = get_1d_offset(1, n_prev_chunks=pid_m)
    offs_n = get_1d_offset(1, n_prev_chunks=pid_n)
    
    offs = get_2d_offset(offs_m, offs_n, stride_0=num_pid_n)
    mask = get_2d_mask(offs_m, offs_n, max_0=num_pid_m, max_1=num_pid_n )

    offs_sw_m = get_1d_offset(1, n_prev_chunks=pid_m_)
    offs_sw_n = get_1d_offset(1, n_prev_chunks=pid_n_)
    
    offs_sw = get_2d_offset(offs_sw_m, offs_sw_n, stride_0=num_pid_n)
    mask_sw = get_2d_mask(offs_sw_m, offs_sw_n, max_0=num_pid_m, max_1=num_pid_n)
    
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(z_ptr + offs_sw, x, mask=mask_sw)
blocks_m, blocks_n = 5,4

x = torch.arange(blocks_m*blocks_n, device='cuda').view(blocks_m,blocks_n)
x
z = -torch.ones_like(x) # empty matrix, with -1 denoting empty
z
# swizzle x into z
swizzle_k[(blocks_m,blocks_n)](x,z, group_sz=3);
z
Looks good!
___
Let's now implement the grouped matmul kernel, which will be faster than the regular matmul.
@triton.jit
def grouped_matmul_k(
    a_ptr, b_ptr, c_ptr,
    m, n, k,
    stride_am, stride_ak, 
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    bm: tl.constexpr, bn: tl.constexpr, bk: tl.constexpr, group_sz: tl.constexpr
):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    num_pid_m, num_pid_n = tl.num_programs(0), tl.num_programs(1)
    # determine location of block in grouped ordering - swizzle! 
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
grouped_matmul = partial(matmul, matmul_k_fn=grouped_matmul_k)
a = torch.ones((3, 4), dtype=torch.float32, device='cuda')
b = torch.ones((4, 5), dtype=torch.float32, device='cuda')
grouped_matmul(a,b, group_sz=4)
Let's unit test this against PyTorch's implementation
torch.manual_seed(0)
a = torch.randn((512, 512), device='cuda', dtype=torch.float16)
b = torch.randn((512, 512), device='cuda', dtype=torch.float16)
triton_output = grouped_matmul(a, b, group_sz=32)
torch_output = torch.matmul(a, b)
if torch.allclose(triton_output, torch_output, atol=5e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")


# Benchmarking
Triton brings built-in benchmarking tools with it. Here's an example how to use it.
# adapted from https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['square_matrix_size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(5, 12, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['naive', 'grouped', 'torch'],  # Possible values for `line_arg`.
        line_names=['Naive', 'Grouped', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-'), ('orange','-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='matmul-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(square_matrix_size, provider):
    sz = square_matrix_size
    a = torch.rand((sz, sz), device='cuda', dtype=torch.float32)
    b = torch.rand((sz, sz), device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'naive':   ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_matmul(a, b), quantiles=quantiles)
    if provider == 'grouped': ms, min_ms, max_ms = triton.testing.do_bench(lambda: grouped_matmul(a, b, group_sz=8), quantiles=quantiles)
    if provider == 'torch':   ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a,b), quantiles=quantiles)
    gbps = lambda ms: 12 * sz / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark.run(print_data=True, show_plots=True)
_Note Umer: I would've expected the GB/s to increase as the matrix sizes get larger. Why don't they? Maybe because share memory is full, so kernel spends more and more time reloading stuff_

Let's try different block sizes:
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
Larger block sizes seem to be better. Let's compare with pytorch again, using larger block sizes.
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
This reduces the performance difference to pytorch for larger matrix sizes, but pytorch is still better.

Tip: For profiling, we can use Nsight Compute to profile our kernels:
`ncu --target-processes all your_python_file.py`

# Auto-Tuning
Adapted from https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
The choice of meta-parameters (e.g. block sizes) and compilation options (e.g. `num_warps`) impacts the kernel speed. Triton allows you to pass a list of possible choices, runs them all, and then compiles the kernel for the fastest choice. This is called `Auto-Tuning`.

If the size of your problem changes (e.g. when matrix changes size), a new auto-tune will be done for the new problem size.
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

a,b = torch.ones(3,4, device='cuda'), torch.ones(4,5, device='cuda')
a@b
_Note: sometimes the following line returns wrong results, and I can't reliably reproduce it. If you can, please tell me via Twitter (@UmerHAdil) ! 🙏🏽_
grouped_autotuned_matmul(a,b)

For tips, tricks and heuristics which configs to try for auto-tuning, see [Mark Saroufim's talk "CUDA Performance Checklist"](https://www.youtube.com/watch?v=SGhfUhlowB4). Much of it should apply to Triton as well.

Let's run the benchmark once again. This will take a lot of time, as we auto-tune for each benchmarking paramater choice (i.e., 12-5=7 times for us).
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



___
<h1>That's it! Congrats on making it through the tutorial - good work! 🥳</h1>

I highly encourage you to write a few triton kernels yourself. You can e.g. try these triton puzzles: https://github.com/srush/Triton-Puzzles by [Sasha Rush](https://twitter.com/srush_nlp), Tejas Ramesh and [Keren Zhou](https://twitter.com/ZhouKeren).

Here is other intermediate and advanced material:
- The official documentation: https://triton-lang.org/
- The LightLLM repo has a ton of real-world triton kernels: https://github.com/ModelTC/lightllm/tree/main/lightllm/common/basemodel/triton_kernel
- So does the Unsloth repo: https://github.com/unslothai/unsloth/tree/main/unsloth/kernels
If you're generally interested in GPU programming and performance, the [cuda mode Discord](https://discord.gg/cudamode) may be interesting to you. This tutorial was written as part of their amazing [lecture series](https://www.youtube.com/@CUDAMODE).

___
**About the author:**
Hey 👋🏽 I'm Umer from Germany. Thanks for reading this tutorial. I hope you got learned a lot from it. If you have any questions, feel free to shoot me a message on Twitter ([@UmerHAdil](https://x.com/UmerHAdil)).

As I currently do Open-Source AI work as an independent ML engineer, I have set up a ko-fi page for tips & donations: https://ko-fi.com/umerha.
Apart from this guide, I've contributed to HuggingFace diffusers (e.g. [shoutouts by HF](https://x.com/RisingSayak/status/1773739194474463629)), LangChain [shoutouts by the team](https://twitter.com/search?lang=de&q=(from%3ALangChainAI)%20(%40UmerHAdil)%20lang%3Aen&src=typed_query)), and gpt-engineer (e.g. [this](https://x.com/UmerHAdil/status/1715447656527339668)).
If you're a company in need of Triton and/or CUDA consulting, also shoot me a message on Twitter ([@UmerHAdil](https://x.com/UmerHAdil)).
