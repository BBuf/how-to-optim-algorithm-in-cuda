## 前言

在vllm里面看到flash attention包了一层`@torch.library.custom_op`装饰器（https://github.com/vllm-project/vllm/pull/7536），查阅了一下资料，发现这个是torch 2.4之后的新feature，防止打算torch compile的graph，翻译一下官方教程稍微了解一下这个用法。 

来源：https://pytorch.org/tutorials/advanced/python_custom_ops.html

## Python Custom Operators 教程


![](https://files.mdnice.com/user/59/5b6dcf68-8985-4f7b-b8e1-bafa5e6b47fb.png)

这个教程介绍了Python自定义运算符的主题。它列出了我们将从这一教程中学习到的内容,包括如何将用Python编写的自定义运算符与PyTorch集成,以及如何使用torch.library.opcheck来测试自定义运算符。所需的先决条件是安装了PyTorch 2.4或更高版本。

PyTorch提供了大量可以在Tensor上运行的运算符(例如torch.add、torch.sum等)。但是,您可能希望在PyTorch中使用一个新的自定义运算符,可能是由第三方库编写的。本教程展示了如何封装Python函数,使它们的行为类似于PyTorch原生运算符。创建PyTorch中的自定义运算符的原因可能包括:

- 将任意Python函数视为不透明的可调用对象,与torch.compile相对应(即防止torch.compile跟踪进入函数)。
- 为任意Python函数添加训练支持。

请注意,如果您的操作可以表示为现有PyTorch运算符的组合,那么通常就不需要使用自定义运算符(例如,支持自动微分的运算应该可以直接工作)。

## 例子：将PIL库的crop功能封装为一个自定义运算符

假设我们在使用PIL的`crop`操作

```python
import torch
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
import PIL
import IPython
import matplotlib.pyplot as plt

def crop(pic, box):
    img = to_pil_image(pic.cpu())
    cropped_img = img.crop(box)
    return pil_to_tensor(cropped_img).to(pic.device) / 255.

def display(img):
    plt.imshow(img.numpy().transpose((1, 2, 0)))

img = torch.ones(3, 64, 64)
img *= torch.linspace(0, 1, steps=64) * torch.linspace(0, 1, steps=64).unsqueeze(-1)
display(img)
```


![](https://files.mdnice.com/user/59/c06233fd-222b-4f28-8c27-092f0f5bfe89.png)

```python
cropped_img = crop(img, (10, 10, 50, 50))
display(cropped_img)
```

![](https://files.mdnice.com/user/59/56dcc1ef-c4df-4aad-ab9a-cf39d8384b73.png)

`crop`功能无法被`torch.compile`有效地开箱即用处理:`torch.compile`在无法处理的函数上会引发"图中断"(https://pytorch.org/docs/stable/torch.compiler_faq.html#graph-breaks),而图中断会导致性能下降。以下代码通过引发错误来演示这一点(如果发生图中断,`torch.compile(with fullgraph=True)`会引发错误)。

```python
@torch.compile(fullgraph=True)
def f(img):
    return crop(img, (10, 10, 50, 50))

# The following raises an error. Uncomment the line to see it.
# cropped_img = f(img)
```

为了能在`torch.compile`中使用`crop`作为黑盒操作,我们需要做两件事:

- 将该函数封装为一个PyTorch自定义运算符。
- 为该运算符添加"FakeTensor kernel"(又称"meta kernel")。给定输入Tensor的元数据(例如形状),此函数说明如何计算输出Tensor的元数据。

```python
from typing import Sequence

# Use torch.library.custom_op to define a new custom operator.
# If your operator mutates any input Tensors, their names must be specified
# in the ``mutates_args`` argument.
@torch.library.custom_op("mylib::crop", mutates_args=())
def crop(pic: torch.Tensor, box: Sequence[int]) -> torch.Tensor:
    img = to_pil_image(pic.cpu())
    cropped_img = img.crop(box)
    return (pil_to_tensor(cropped_img) / 255.).to(pic.device, pic.dtype)

# Use register_fake to add a ``FakeTensor`` kernel for the operator
@crop.register_fake
def _(pic, box):
    channels = pic.shape[0]
    x0, y0, x1, y1 = box
    return pic.new_empty(channels, y1 - y0, x1 - x0)
```

做了上述操作之后,crop现在可以在不产生图中断的情况下正常工作了。

```python
@torch.compile(fullgraph=True)
def f(img):
    return crop(img, (10, 10, 50, 50))

cropped_img = f(img)
display(img)
```


![](https://files.mdnice.com/user/59/fad4c742-771b-431f-8169-d27bc047d26e.png)

```python
display(cropped_img)
```


![](https://files.mdnice.com/user/59/3342a1f9-19be-4f7a-868a-d59ff20a3ba5.png)


## 为crop添加训练支持

使用`torch.library.register_autograd`为运算符添加训练支持。相比直接使用`torch.autograd.Function`,优先使用这种方式;因为`autograd.Function`与PyTorch运算符注册API组合使用时,可能会在与`torch.compile`组合时导致无声的不正确性。

`crop`的梯度公式本质上是`PIL.paste`(我们把推导留作读者练习)。让我们首先将`paste`封装为一个自定义运算符:

```python
@torch.library.custom_op("mylib::paste", mutates_args=())
def paste(im1: torch.Tensor, im2: torch.Tensor, coord: Sequence[int]) -> torch.Tensor:
    assert im1.device == im2.device
    assert im1.dtype == im2.dtype
    im1_pil = to_pil_image(im1.cpu())
    im2_pil = to_pil_image(im2.cpu())
    PIL.Image.Image.paste(im1_pil, im2_pil, coord)
    return (pil_to_tensor(im1_pil) / 255.).to(im1.device, im1.dtype)

@paste.register_fake
def _(im1, im2, coord):
    assert im1.device == im2.device
    assert im1.dtype == im2.dtype
    return torch.empty_like(im1)
```

现在让我们使用`register_autograd`来为`crop`指定梯度公式:

```python
def backward(ctx, grad_output):
    grad_input = grad_output.new_zeros(ctx.pic_shape)
    grad_input = paste(grad_input, grad_output, ctx.coords)
    return grad_input, None

def setup_context(ctx, inputs, output):
    pic, box = inputs
    ctx.coords = box[:2]
    ctx.pic_shape = pic.shape

crop.register_autograd(backward, setup_context=setup_context)
```

注意,backward必须是由PyTorch可理解的运算符组成,这也是我们将paste封装为自定义运算符而不直接使用PIL的paste的原因。

```python
img = img.requires_grad_()
result = crop(img, (10, 10, 50, 50))
result.sum().backward()
display(img.grad)
```


![](https://files.mdnice.com/user/59/d8bd6436-fa8a-45f8-824a-90a85e3ce670.png)

这是正确的梯度,在裁剪区域内是1(白色),在未使用的区域内是0(黑色)。

## 测试Python自定义运算符

使用`torch.library.opcheck`来测试自定义运算符是否正确注册。这不会测试梯度是否在数学上正确,请单独编写测试(手动测试或使用`torch.autograd.gradcheck`)。

要使用`opcheck`,请传入一组示例输入用于测试。如果你的运算符支持训练,那么示例应该包括需要计算梯度的Tensor。如果你的运算符支持多个设备,那么示例应该包括来自每个设备的Tensor。

```python
examples = [
    [torch.randn(3, 64, 64), [0, 0, 10, 10]],
    [torch.randn(3, 91, 91, requires_grad=True), [10, 0, 20, 10]],
    [torch.randn(3, 60, 60, dtype=torch.double), [3, 4, 32, 20]],
    [torch.randn(3, 512, 512, requires_grad=True, dtype=torch.double), [3, 4, 32, 45]],
]

for example in examples:
    torch.library.opcheck(crop, example)
```

## 可变的Python自定义运算符

你也可以将一个会修改其输入的Python函数封装为自定义运算符。修改输入的函数很常见,因为这是许多low-level kernel编写的方式;例如,计算sin的kernel可能会修改输入,并将输出张量赋值为`input.sin()`。

我们将使用`numpy.sin`来演示一个可变的Python自定义运算符的示例。

```python
import numpy as np

@torch.library.custom_op("mylib::numpy_sin", mutates_args={"output"}, device_types="cpu")
def numpy_sin(input: torch.Tensor, output: torch.Tensor) -> None:
    assert input.device == output.device
    assert input.device.type == "cpu"
    input_np = input.numpy()
    output_np = output.numpy()
    np.sin(input_np, out=output_np)
```

由于该运算符没有返回值,因此不需要注册FakeTensor kernel(元kernel)就可以在`torch.compile`中正常工作。

```python
@torch.compile(fullgraph=True)
def f(x):
    out = torch.empty(3)
    numpy_sin(x, out)
    return out

x = torch.randn(3)
y = f(x)
assert torch.allclose(y, x.sin())
```

这里是一次opcheck运行的结果,告诉我们确实正确注册了该kernel。如果我们忘记添加输出到`mutates_args`,例如,`opcheck`将会报错。

## 总结

在本教程中,我们学习了如何使用`torch.library.custom_op`创建一个与PyTorch子系统如`torch.compile`和`autograd`协同工作的Python自定义运算符。

本教程提供了自定义运算符的基本介绍。更多详细信息,请参阅:

- torch.library文档: https://pytorch.org/docs/stable/library.html
- 自定义运算符手册: https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html#the-custom-operators-manual








