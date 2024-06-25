## 第一课: 如何在 PyTorch 中 profile CUDA kernels

第一页Slides：


![](https://files.mdnice.com/user/59/b920b268-3090-4d04-b067-615909948d40.png)

这里是课程规划，有三位讲师 Andreas, Thomas, Mark，然后大概2周出一个 CUDA 主题的讲解以及工程或者结对编程的视频。课程讨论的主题是根据 《Programming Massively Parallel Processors》这本书来的，Mark 也是在8分钟的时候强推了这本书。另外在6分钟左右 Mark 指出，学习 CUDA 的困难之处在于对于新手来说，可能会陷入不断循环查找文档的状态，非常痛苦。

第二页Slides:



![](https://files.mdnice.com/user/59/10a32c56-c912-4323-b63b-96d3262521c6.png)

这里是说Lecture 1的目标是如何把一个 CUDA kernel 嵌入到 PyTorch 里面，以及如何对它进行 Profile 。相关的代码都在：https://github.com/cuda-mode/lectures/tree/main/lecture_001 。Mark 还提到说这个课程相比于以前的纯教程更加关注的是我们可以利用 CUDA 做什么事情，而不是让读者陷入到 CUDA 专业术语的细节中，那会非常痛苦。

第三页Slides：

![](https://files.mdnice.com/user/59/b38fefee-7614-4966-af7d-c466ce9568aa.png)

这一页 Slides 中的代码在  https://github.com/cuda-mode/lectures/blob/main/lecture_001/pytorch_square.py


```python
import torch

a = torch.tensor([1., 2., 3.])

print(torch.square(a))
print(a ** 2)
print(a * a)

def time_pytorch_function(func, input):
    # CUDA IS ASYNC so can't use python time module
    # CUDA是异步的，所以你不能使用python的时间模块，而应该使用CUDA Event
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warmup (防止CUDA Context初始化影响时间记录的准确性)
    for _ in range(5):
        func(input)

    start.record()
    func(input)
    end.record()
    # 程序完成之后需要做一次 CUDA 同步
    torch.cuda.synchronize()
    return start.elapsed_time(end)

b = torch.randn(10000, 10000).cuda()

def square_2(a):
    return a * a

def square_3(a):
    return a ** 2

time_pytorch_function(torch.square, b)
time_pytorch_function(square_2, b)
time_pytorch_function(square_3, b)

print("=============")
print("Profiling torch.square")
print("=============")

# Now profile each function using pytorch profiler
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    torch.square(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("=============")
print("Profiling a * a")
print("=============")

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    square_2(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("=============")
print("Profiling a ** 2")
print("=============")

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    square_3(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

这里通过在 PyTorch 中实现平方和立方函数并使用 autograd profiler 工具进行 profile 。`time_pytorch_function` 这个函数的计时功能和 `torch.autograd.profiler.profile` 类似，第三页 Slides 里面我们可以通过 PyTorch Profiler 的结果看到当前被 `torch.autograd.profiler.profile` context manager 包起来的 PyTorch 程序 cuda kernel 在 cpu, cuda 上的执行时间以及占比以及 kernel 的调用次数，当前 kernel 的执行时间占总时间的比例。

第四页Slides：

![](https://files.mdnice.com/user/59/f76833c1-a1e4-425d-829b-19296a071523.png)

这一页Slides是对 https://github.com/cuda-mode/lectures/blob/main/lecture_001/pt_profiler.py 这个文件进行讲解，之前我也翻译过PyTorch Profiler TensorBoard 插件教程，地址在 https://zhuanlan.zhihu.com/p/692749819

第5页Slides:

![](https://files.mdnice.com/user/59/cc142f21-fb8a-4d87-8a6a-74af273c2699.png)

