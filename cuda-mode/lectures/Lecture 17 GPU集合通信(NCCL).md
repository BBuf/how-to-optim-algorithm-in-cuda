> 我的课程笔记，欢迎关注：https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode 。

> 这节课介绍了NVIDIA的NCCL（NVIDIA Collective Communications Library）通信库，重点讲解了其在分布式深度学习中的应用。首先通过PyTorch DDP的实例，展示了NCCL如何实现高效的梯度同步。接着介绍了下NCCL的基本概念、API使用、通信器初始化方式，并深入分析了Ring AllReduce算法的工作原理。

# 第17课，GPU集合通信(NCCL)

## 课程笔记

![](https://files.mdnice.com/user/59/ce57718b-eaf9-4916-be9b-e8ac64d7149d.png)

![](https://files.mdnice.com/user/59/0d79775f-e626-4c29-9191-02d2e142721c.png)

这张Slides介绍了 NVIDIA 的 NCCL (NVIDIA Collective Communications Library) 通信库，它是一个专门用于 GPU 之间快速数据通信的库，支持点对点和集体通信两种模式，提供了包括 Scatter、Gather、All-to-all、AllReduce、Broadcast、Reduce、AllGather 和 ReduceScatter 等多种通信原语，Slides下方的图展示了 AllGather 操作的工作流程，然后在上方展示了一下Broadcast和Scatter的示意图。

![](https://files.mdnice.com/user/59/724a52c1-9a4d-4ef2-8d9f-4d9a18767e4a.png)

这张Slides简单展示了一下nccl AllReduce（Reduce Sum）的操作。图片分为"Before"和"After"两个部分，显示了在3个GPU（GPU 0、GPU 1和GPU 2）上的数据处理过程。在初始状态下，每个GPU都包含3个不同的数据块（GPU 0有A、B、C；GPU 1有D、E、F；GPU 2有G、H、I）。经过AllReduce操作后，每个GPU都得到了相同位置数据的总和（即A+D+G、B+E+H、C+F+I），这样三个GPU最终都具有相同的计算结果。

![](https://files.mdnice.com/user/59/83c33d14-211f-444f-a65a-5851acce8cc3.png)

这张Slides讲了一下DDP里面需要nccl的地方，也就是同步全局梯度的时候。具体来说，在这个例子中，数据被分成两部分（x₀和x₁）分别在两个GPU上处理。每个GPU运行相同的模型，计算各自的局部梯度（Local Gradients），然后通过NCCL的AllReduce操作来同步和平均所有GPU上的梯度。最后，每个GPU使用这个平均梯度来更新自己的模型参数，确保所有GPU上的模型保持同步。

![](https://files.mdnice.com/user/59/db1b5e6d-f8b8-49a9-8c81-6f8ac1b1999b.png)

这张Slides更具体了一些，用一个 y = w * 7 * x 的例子，展示了 DDP 里面同步梯度的时候，如何使用 NCCL 的 AllReduce 操作来同步和平均所有 GPU 上的梯度。这个例子作者也提供了一个代码，代码如下：

```python
# modified from https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.profiler import profile

from torch.nn.parallel import DistributedDataParallel as DDP

# 定义一个简单的玩具模型类
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        # 定义一个可训练参数w,初始值为5.0
        self.w = nn.Parameter(torch.tensor(5.0))

    def forward(self, x):
        # 前向传播: y = w * 7 * x
        return self.w * 7.0 * x


def demo_basic():
    # 初始化进程组,使用NCCL后端
    dist.init_process_group("nccl")
    # 获取当前进程的rank
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")

    # 创建模型实例并移到对应GPU
    model = ToyModel().to(rank)
    # 用DDP包装模型
    ddp_model = DDP(model, device_ids=[rank])

    # 使用PyTorch profiler收集性能数据
    with profile() as prof:
        # 创建输入张量,值为当前进程的rank
        x = torch.tensor(dist.get_rank(), dtype=torch.float)
        # 前向传播
        y = ddp_model(x)
        # 打印计算结果
        print(f"rank {rank}: y=w*7*x: {y.item()}={ddp_model.module.w.item()}*7*{x.item()}")
        # 打印关于w的导数
        print(f"rank {rank}: dy/dw=7*x: {7.0*x.item()}")
        # 反向传播
        y.backward()
        # 打印经过AllReduce后的梯度
        print(f"rank {rank}: reduced dy/dw: {ddp_model.module.w.grad.item()}")
    # rank 0负责导出性能跟踪文件
    if rank == 0:
        print("exporting trace")
        prof.export_chrome_trace("trace_ddp_simple.json")
    # 清理进程组
    dist.destroy_process_group()


if __name__ == "__main__":
    print("Running")
    demo_basic()

# torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=localhost:29400 ddp_simple.py
```

接着作者给出了一个稍微完善一些的例子，由Linear和ReLU组成的网络，有optimizer更新参数的过程，代码如下：

```python
# modified from https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

import torch
import torch.distributed as dist
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile
import torch.optim as optim

SIZE = 4000


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(SIZE, SIZE)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(SIZE, SIZE)
        self.net3 = nn.Linear(SIZE, SIZE)

    def forward(self, x):
        return self.net3(self.relu(self.net2(self.relu(self.net1(x)))))


def demo_basic():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")

    model = ToyModel().to(rank)
    ddp_model = DDP(model, bucket_cap_mb=25, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    with profile(
        record_shapes=True,
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
    ) as prof:
        for i in range(10):
            optimizer.zero_grad()
            outputs = ddp_model(torch.randn(1000, SIZE, device=rank))
            labels = torch.randn(1000, SIZE, device=rank)
            loss_fn(outputs, labels).backward()
            optimizer.step()
    if rank == 0:
        prof.export_chrome_trace("trace_ddp_example.json")


if __name__ == "__main__":
    demo_basic()

# torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=localhost:29400 ddp_example.py
```

作者分析了几分钟这个代码中一个iter的pytorch profiler结果，我们可以看到前向Pass，反向Pass，优化器更新参数，以及AllReduce的通信时间以及部分AllReduce被重叠到了反向计算中。这就引入到了下一张slides。

![](https://files.mdnice.com/user/59/9c5ec5c5-54b5-404c-8ad9-83e4f619c60c.png)

这里作者讲了一下DDP里面的AllReduce是怎么和Backward Pass重叠的，这个建议阅读这篇博客：https://zhuanlan.zhihu.com/p/485208899 ，从这张Slides的PyTorch Profiler图我们也可以发现一些其它信息，例如在同一个Stream上的kernel是顺序执行，所以为了重叠计算和通信这里使用了两个Stream。由于网络最开始的几个层必须等待梯度计算完毕才能开始AllReduce，所以存在无法重叠的层。

![](https://files.mdnice.com/user/59/a0dc19bc-8ac8-43b0-a3fa-623c9bedee4b.png)

这张Slides提了一下yTorch DDP的内部机制，包括：
- DDP的梯度同步机制：
    - 使用 autograd hooks 在构建时注册，用于触发梯度同步
    - Reducer 组件会异步执行 allreduce 操作来计算所有进程间的梯度平均值
    - 计算完成后，平均后的梯度会被写入所有参数的 param.grad 字段
    - 在反向传播完成后，不同 DDP 进程中相同参数的梯度值应该是一致的
- 通信后端支持：
    - DDP 支持多种通信后端，包括：
        - NCCL
        - MPI
        - Gloo
- 具体实现：
    - NCCL API 的调用是在 PyTorch 的 ProcessGroupNCCL.cpp 文件中通过 Reducer 完成的

![](https://files.mdnice.com/user/59/47c15734-70c7-4a29-b367-d67edba06c6e.png)

这张Slides开始介绍NCCL库中的nccl AllReduce API函数。该函数用于对长度为count的数据数组进行规约(reduce)操作，使用指定的op操作符进行计算，并将相同的结果复制到每个recvbuff中。当sendbuff和recvbuff指向相同位置时，会执行原地操作。这是一个在分布式深度学习中常用的集合通信操作，用于在多个GPU之间同步和聚合数据。

![](https://files.mdnice.com/user/59/c1a64c7c-fb3c-48de-9544-0c67d7cf5f39.png)

这张Slides介绍了NCCL通信器对象的两种使用场景：一种是每个CPU进程对应一个GPU的情况，此时root进程会生成唯一ID并广播给所有进程，所有进程用相同的ID和唯一的rank初始化通信器例如MPI；另一种是单个CPU进程管理多个GPU的情况，这时不需要广播ID，而是通过循环来初始化每个rank，并可以使用封装好的ncclCommInitAll函数来简化这个过程。Slides右侧的代码示例展示了这些初始化操作的具体实现方式。

![](https://files.mdnice.com/user/59/13432c2e-775e-4709-aba9-d888dd9d611f.png)


这张Slides展示了错误处理宏定义

```c++
#define CUDACHECK(cmd) {                    
    cudaError_t err = cmd;                  
    if (err != cudaSuccess) {              
        printf("Failed: Cuda error %s:%d\n",
            __FILE__,__LINE__,cudaGetErrorString(err));
        exit(EXIT_FAILURE);               
    }
}

#define NCCLCHECK(cmd) {                    
    ncclResult_t res = cmd;               
    if (res != ncclSuccess) {             
        printf("Failed: NCCL error %s:%d\n",
            __FILE__,__LINE__,ncclGetErrorString(res));
        exit(EXIT_FAILURE);               
    }
}
```

这部分定义了两个错误处理宏:
- `CUDACHECK`: 用于检查CUDA API调用的错误
- `NCCLCHECK`: 用于检查NCCL操作的错误

![](https://files.mdnice.com/user/59/7a657778-8327-4bb8-a922-82ea777da483.png)

```c++
int main(int argc, char* argv[]) {
    ncclComm_t comms[4];
    
    //管理4个设备
    int nDev = 4;
    int size = 32*1024*1024;
    int devs[4] = { 0, 1, 2, 3 };
    
    //分配和初始化设备缓冲区
    float** sendbuff = (float**)malloc(nDev * sizeof(float*));
    float** recvbuff = (float**)malloc(nDev * sizeof(float*));
    cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);
```

这里的代码创建了NCCL通信器数组，设置4个GPU设备，定义数据大小(32MB)，分配发送和接收缓冲区的内存并为每个设备创建CUDA流。然后还有下面的循环

```c++
for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
    CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
    CUDACHECK(cudaStreamCreate(s+i));
}
```

这个循环给每个GPU设置当前设备，然后分配发送和接收缓冲区的GPU内存，初始化发送缓冲区为1，接收缓冲区为0，最后为每个设备创建CUDA流。

![](https://files.mdnice.com/user/59/93af1802-ae63-46de-ad68-1b915290befd.png)

```c++
//初始化NCCL
NCCLCHECK(ncclCommInitAll(comms, nDev, devs));

//调用NCCL通信API
NCCLCHECK(ncclGroupStart());
for (int i = 0; i < nDev; ++i)
    NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum,
        comms[i], s[i]));
NCCLCHECK(ncclGroupEnd());

//同步CUDA流等待NCCL操作完成
for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(s[i]));
}
```

这部分代码展示了初始化NCCL通信器，执行AllReduce操作(将所有设备的数据求和并分发给所有设备)，最后同步所有CUDA流确保操作完成。

![](https://files.mdnice.com/user/59/e6cb25f8-cfee-45d2-a20c-f5a1e42d4925.png)

```c++
//释放设备缓冲区
for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaFree(sendbuff[i]));
    CUDACHECK(cudaFree(recvbuff[i]));
}

//终止NCCL
for(int i = 0; i < nDev; ++i)
    ncclCommDestroy(comms[i]);
```

最后进行资源清理包括释放GPU上分配的内存，销毁NCCL通信器。

上面4张slides放在一起展示了一个如何在单个进程中使用NCCL进行AllReduce操作。

![](https://files.mdnice.com/user/59/56e9c425-4960-4cce-ba06-b7aeb24e27d5.png)

这张Slides展示了"每个CPU进程一个GPU"的场景下的实现。代码有以下步骤：

- 获取NCCL唯一ID并在所有进程间广播
- 基于本地rank选择GPU并分配设备缓冲区
- 初始化NCCL通信器
- 使用NCCL执行AllReduce集合通信操作（从代码可以看到是每个rank都发起了这个操作）
- 同步CUDA流来完成NCCL操作

实际上这个例子对应的就是PyTorch Distributed Data Parallel里面的AllReduce操作，而上面的Single Process的例子对应的就是PyTorch Data Parallel里面的AllReduce操作。

![](https://files.mdnice.com/user/59/fd426fdb-0634-4912-bcf6-6fe8ca6de0fa.png)

这里展示了一下环状的AllReduce算法的原理，它由两个操作组成：
- ReduceScatter 操作: 输入数据分布在不同的 rank (进程/节点) 上 (rank 0 到 rank 3)；每个 rank 负责对一部分数据进行规约(reduction)操作；规约结果被分散到不同的 rank 上；图中显示 out[i] = sum(in[j]^count+i))
- AllGather 操作: 在 ReduceScatter 之后执行；每个 rank 将自己的部分结果广播给其他所有 rank；最终每个 rank 都获得完整的规约结果；图中显示 out[Ycount+i] = in[Y][i]

![](https://files.mdnice.com/user/59/2e7cda71-ea43-4a58-819d-4b7512eaf0cc.png)

这张Slides截了一下Ring Allreduce的cuda代码实现，可以粗略的浏览一下代码：

```c++
// Ring AllReduce算法实现 (结合了ReduceScatter和AllGather操作)
template<typename T, typename RedOp, typename Proto>
__device__ __forceinline__ void run(ncclWorkElem *args) {
    const int tid = threadIdx.x;      // 获取当前线程ID
    const int nthreads = args->nWarps*WARP_SIZE;  // 计算总线程数
    const int bid = args->bid;        // 获取块ID
    const int nChannels = args->nChannels;  // 获取通道数
    ncclRing *ring = &ncclShmem.channel.ring;  // 获取环形通信结构的指针
    int ringIx = ring->index;         // 获取环形索引
    
    // 计算每步处理的数据块大小
    const size_t chunkSize = int(Proto::calcBytePerStep()/sizeof(T)) * (Proto::Id == NCCL_PROTO_SIMPLE ? ALLREDUCE_CHUNKSTEPS : 1));
    const int nranks = ncclShmem.comm.nRanks;  // 获取总进程数
    const size_t loopSize = nChannels*nranks*chunkSize;  // 计算循环大小
    const size_t size = args->count;  // 获取需要处理的总数据量

    int minChunkSize;  // 最小数据块大小
    if (Proto::Id == NCCL_PROTO_LL) {
        // LL协议下计算最小数据块大小
        minChunkSize = nthreads*(Proto::calcBytePerGrain()/sizeof(T));
    }
    if (Proto::Id == NCCL_PROTO_LL128) {
        // LL128协议下的特殊处理
        // 注释说明这里的除2可能是个bug，但能提高性能
        minChunkSize = nthreads*(Proto::calcBytePerGrain()/sizeof(T))/2;
    }

    // 使用Primitives模板类处理规约操作
    Primitives<T, RedOp, FanSymmetric<1>, Proto, 0> prims
        (tid, nthreads, &ring->prev, &ring->next, args->sendbuff, args->recvbuff, args->redOpArg);
}
```

![](https://files.mdnice.com/user/59/2aad6d00-f264-4cae-bfff-6f3895442c0e.png)

```c++
// Ring AllReduce实现 (ReduceScatter + AllGather)
for (size_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
    size_t realChunkSize;
    
    // 处理NCCL协议简单模式
    if (Proto::id == NCCL_PROTO_SIMPLE) {
        // 计算实际的chunk大小，考虑网格偏移和通道数
        realChunkSize = min(chunkSize, divide(size-gridOffset, nChannels*nranks));
        // 根据线程数和数据类型大小调整chunk大小
        realChunkSize = roundUp(realChunkSize, (nthreads*WARP_SIZE)*sizeof(uint64_t)/sizeof(T));
    } else {
        // 非简单模式下的chunk大小计算
        realChunkSize = min(chunkSize, divide(size-gridOffset, nChannels*nranks*minChunkSize));
        realChunkSize = int(realChunkSize);
    }

    // 计算每个chunk的偏移量
    auto calcOffset = [&]__device__(int chunk)->size_t {
        if (Proto::id == NCCL_PROTO_SIMPLE)
            return gridOffset + bid*nranks*realChunkSize + chunk*realChunkSize;
        else
            return gridOffset + (chunk*nChannels + bid)*realChunkSize;
    };

    // 计算每个rank的修改位置
    auto modRanks = [&]__device__(int r)->int {
        return r >= nranks ? r-nranks : r;
    };

    // 声明变量
    size_t offset;
    int nelem;
    int chunk;

    // step 0: 将数据推送到下一个GPU
    chunk = modRanks(ringIx + nranks-1);  // 计算chunk索引
    offset = calcOffset(chunk);           // 计算偏移量
    nelem = min(realChunkSize, size-offset); // 计算元素数量
    prims.send(offset, nelem);           // 发送数据
}

```

![](https://files.mdnice.com/user/59/bb3be79f-b27e-49b5-8188-c5fef22d6567.png)

![](https://files.mdnice.com/user/59/5e8cffe9-54e7-46c6-b1d9-a08a4fd15a29.png)

![](https://files.mdnice.com/user/59/43e84d6a-f05a-490f-916b-7498bb9d5302.png)

这几张Slides展示了Ring AllReduce（环形全规约）算法的工作原理，它是通过组合ReduceScatter和AllGather两个操作来实现的。第一张Slides的图展示了初始状态：

- 有3个GPU (GPU 0, 1, 2)
- 每个GPU上有3个数据块(A/B/C, D/E/F, G/H/I)

第二张Slides的图展示了数据传输的模式：

- 数据以环形方式在GPU之间传递
- GPU 0 向 GPU 1 传输
- GPU 1 向 GPU 2 传输
- GPU 2 回传到 GPU 0，形成一个环

```c++
// k-2步: 执行规约操作并将结果复制到下一个GPU
for (int j=2; j<nranks; ++j) {
    // 计算当前需要处理的数据块索引
    // ringIx是当前GPU的索引，通过模运算确保索引在有效范围内
    chunk = modRanks(ringIx + nranks-j);
    
    // 根据chunk计算在缓冲区中的偏移量
    offset = calcOffset(chunk);
    
    // 计算本次需要传输的实际元素数量
    // 取实际块大小和剩余大小中的较小值，避免越界
    nelem = min(realChunkSize, size-offset);
    
    // 执行接收-规约-发送操作
    // 从上一个GPU接收数据，与本地数据进行规约，然后发送给下一个GPU
    prims.recvReduceSend(offset, nelem);
}
```

![](https://files.mdnice.com/user/59/18d44551-e153-4bc3-9a08-8965a6e15767.png)

![](https://files.mdnice.com/user/59/9669e91f-dd34-4c75-b94a-192de73e39ca.png)

![](https://files.mdnice.com/user/59/20f9a541-3ca7-4522-8608-947c125633c5.png)

这里展示了Ring AllReduce 第k-1步做的事：

```c++
// step k-1: 在当前GPU上规约缓冲区和数据
// 规约结果将存储在当前数据中并传送到下一个GPU

// 计算当前要处理的数据块索引
// ringIx 是环形通信中的索引位置
chunk = ringIx + 0;

// 根据chunk计算在内存中的偏移量
// 用于确定数据在缓冲区中的具体位置
offset = calcOffset(chunk);

// 计算本次需要处理的实际元素数量
// realChunkSize: 标准块大小
// size-offset: 剩余可处理的元素数量
// 取两者的最小值以防止越界
nelem = min(realChunkSize, size-offset);

// 执行接收-规约-复制-发送操作
// offset: 源数据偏移量
// offset: 目标数据偏移量
// nelem: 要处理的元素数量
// true: postOp参数，表示是否执行后续操作
prims.directRecvReduceCopySend(offset, offset, nelem, /*postOp=*/true);
```

上面的过程实际上就对应了ReduceScatter操作。

![](https://files.mdnice.com/user/59/80d177c5-26af-4e86-8f95-ed743bb3d179.png)

![](https://files.mdnice.com/user/59/02ff0d3a-3287-4380-99dc-f3c5e9ef8f25.png)

![](https://files.mdnice.com/user/59/5f38bde1-5dcf-410f-bb21-0f6133e53761.png)

![](https://files.mdnice.com/user/59/105b6aaf-7efb-464e-b2c0-95efbc12db01.png)

![](https://files.mdnice.com/user/59/9481193f-0f6d-4479-b63b-e295c2b609af.png)

![](https://files.mdnice.com/user/59/111cb19a-2c2c-4f86-ad71-b10b32003dc7.png)

这几张图涉及到的就是AllGather操作，只有数据复制，没有数据的Reduce操作。操作完成之后我们可以看到所有的rank上的数据都拥有一样的求和值。

![](https://files.mdnice.com/user/59/1213aca7-c031-4c7a-9d25-9fd02469879e.png)

这里提一些有趣的知识
- 除了Ring Allreduce之外还有其它的AllReduce算法，如Tree AllReduce（树形归约）算法。可以参考https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/
- 其他集体通信操作（Other Collectives）
- 网络拓扑相关技术，包括NVLink、Infiniband/RoCE（提供了NVIDIA官方白皮书链接）以及IP网络
- 集体操作原语（Collective Operation Primitives）

![](https://files.mdnice.com/user/59/911308d9-e764-4ec0-9e83-9a87bdc79895.png)

最后这张Slides介绍了 CUDA 中其它的集体操作原语（Collective Operations Prims），主要说明了 prims.send、prims.recvReduceSend 等函数是如何在 GPU 之间进行集体操作数据传输的。这些原语实现了三种不同的协议：Simple（简单协议）、LL（低延迟协议，8字节原子存储，4字节数据和4字节标志）以及 LL128（低延迟128位协议，128字节原子存储，120字节数据和8字节标志）。另外，AllReduce 操作通过组合3种算法和3种协议，总共可以有9种不同的运行方式，这些原语为 GPU 集群中的并行计算和数据通信提供了灵活的性能选择。

## 总结

这节课介绍了NVIDIA的NCCL（NVIDIA Collective Communications Library）通信库，重点讲解了其在分布式深度学习中的应用。首先通过PyTorch DDP的实例，展示了NCCL如何实现高效的梯度同步。接着介绍了下NCCL的基本概念、API使用、通信器初始化方式，并深入分析了Ring AllReduce算法的工作原理。

