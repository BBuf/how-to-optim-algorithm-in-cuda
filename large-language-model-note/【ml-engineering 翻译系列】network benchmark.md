> 我的课程笔记，欢迎关注：https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode 

> 本篇文档的来源：https://github.com/stas00/ml-engineering 。这篇文档主要介绍了大规模分布式机器学习训练中的网络基准测试和优化。文档首先介绍了几个用于测试网络性能的工具脚本,包括`all_reduce_bench.py`、`all_gather_object_vs_all_reduce.py`和`all_reduce_latency_comp.py`。然后讨论了进行网络基准测试的关键要求,强调了可重复性的重要性。接着详细介绍了网络吞吐量的重要性,包括如何测试和解释结果,以及不同GPU和框架对网络带宽的要求。文档还讨论了NCCL(NVIDIA Collective Communications Library)的性能优化,介绍了几个重要的NCCL环境变量及其作用。最后,文档提供了三个基准测试脚本的详细代码,包括`all_reduce_bench.py`、`all_reduce_latency_comp.py`和`all_gather_object_vs_all_reduce.py`,这些脚本可用于测试不同场景下的网络性能。

# 网络基准测试

**工具**:

- `all_reduce_bench.py`(https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/all_reduce_bench.py) - 一个在大量数据上执行`all_reduce`操作时对实际网络带宽进行基准测试的工具。这对于了解实际获得的性能与宣传规格之间的差异很有用。

- `all_gather_object_vs_all_reduce.py`(https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/all_gather_object_vs_all_reduce.py) - 一个快速基准测试,展示了在从进程组收集完成状态时,从`all_gather_object`切换到`all_reduce`可以获得23倍的速度提升。例如,在实现某种所有进程都完成的标志时。这种技术通常用于同步可能在不同迭代次数完成的GPU - 这在多个DP通道上进行推理时需要,或者当想要在`DataLoader`中同步`StopIteration`事件时。另请参阅`all_gather_object_vs_all_gather.py`(https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/all_gather_object_vs_all_gather.py)。

- `all_reduce_latency_comp.py`(https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/all_reduce_latency_comp.py) - 举例说明1次4GB规模的归约操作比1000次4MB规模的归约操作要快得多



## 关键的可重复性要求

一系列成功实验的最重要要求是能够在只改变一个或几个设置变量的情况下反复重现实验环境。

因此，当你试图弄清楚某些变化是否会提高或降低性能时，你必须想办法保持事物的稳定性。

例如，你需要找到一种方法来防止网络使用出现波动。当我们为108B pre-BLOOM实验(https://github.com/bigscience-workshop/bigscience/tree/master/train/tr8-104B-wide)进行性能优化时，这几乎是不可能完成的，因为我们使用的是共享的节点间网络，完全相同的设置会因其他用户使用网络的多少而产生不同的吞吐量。这种方式行不通。在BLOOM-176B期间，我们获得了一个专用的SLURM分区，其中有一个隔离的网络，只有我们的流量。在这样的环境中进行性能优化简直是完美的。


## 网络吞吐量

了解你的特定模型大小和框架对网络带宽、吞吐量和延迟的要求至关重要。如果你在网络上投入不足，最终会导致GPU闲置，从而浪费金钱和时间。如果你为非常快的网络支付了过高的费用，但你的GPU很慢，那么你同样浪费了金钱和时间。

如果你的网络非常慢，你的训练很可能会受到网络限制，许多训练设置的改进都无法帮助提高性能。

注意：EAI cookbook(https://github.com/EleutherAI/cookbook)包含了一组通信基准测试(https://github.com/EleutherAI/cookbook/tree/main/benchmarks/communication)，用于每个集合通信操作，你可以用它来快速测量节点间或节点内网络的吞吐量。

这里有一个简单的all-reduce基准测试，你可以用它来快速测量节点间网络的吞吐量：

`all_reduce_bench.py`(https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/all_reduce_bench.py)

通常建议至少对4个节点进行基准测试，但是，当然，如果你已经可以访问训练期间将要使用的所有节点，那么就使用所有节点进行基准测试。

在4个节点上运行它：

```
GPUS_PER_NODE=8
NNODES=4
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --role `hostname -s`: \
    --tee 3 \
    all_reduce_bench.py
```

注意:
- 如果不是自动获取rank 0主机名的SLURM环境,需要调整`MASTER_ADDR`为rank 0的主机名。

以下是在SLURM环境中使用4个节点运行的方法:
```
salloc --partition=mypartition --nodes=4 --ntasks-per-node=1 --cpus-per-task=48 --gres=gpu:8 --time=1:00:00 bash
srun --gres=gpu:8 --nodes=4 --tasks-per-node=1 python -u -m torch.distributed.run --nproc_per_node=8 --nnodes 4 --rdzv_endpoint $(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1):6000 --rdzv_backend c10d all_reduce_bench.py
```

注意：
- 你可能需要调整 `--cpus-per-task` 和 `--partition` 参数。
- 你只需执行一次 `salloc`，然后可以在同一分配上重复执行 `srun` 多次。

根据目前的情况，你可能会得到介于 5Gbps 和 1600Gbps 之间的结果。为了避免受网络限制，最低速度将取决于你特定的训练框架，但通常你会希望至少达到 400Gbps 或更高。尽管我们在 50Gbps 的网络上训练了 BLOOM。

像 Deepspeed(https://github.com/microsoft/DeepSpeed) 使用 ZeRO Stage-3 这样分片权重和优化阶段的框架比起像 Megatron-Deepspeed(https://github.com/bigscience-workshop/Megatron-DeepSpeed) 这样除了数据并行之外还进行张量和流水线并行的框架产生更多的网络流量。后者只传输激活值，因此不需要那么多带宽。但它们的设置和运行要复杂得多。

当然，一个高效的框架会重叠通信和计算，这样在一个阶段获取数据的同时，另一个阶段可以并行运行计算。因此，只要通信开销小于计算时间，网络要求就得到满足，不必非常出色。

要在使用 DeepSpeed ZeRO Stage 3 和 V100 GPU 进行大规模训练（64+ GPU）时获得合理的 GPU 吞吐量：

1. 100Gbps 是不够的
2. 200-400 Gbps 是可以的
3. 800-1000 Gbps 是理想的

完整详情(https://github.com/microsoft/DeepSpeed/issues/2928#issuecomment-1463041491)

当然，对于 A100 GPU 节点的要求更高，对 H100 的要求甚至更高（但目前还没有分享这样的基准信息）。

### 从几个节点推断到多个节点的基准测试结果

由于对数百个节点进行基准测试通常并不容易,我们经常尝试使用4个节点来对互连性能进行基准测试。我不确定这是否能为使用40或400个节点时提供正确的指示,所以我在这里(https://github.com/NVIDIA/nccl/issues/790)询问了这个问题,得到的回答是:

> 对环形和树形算法进行大规模推断并不难(我们在`tuning.cc`中有一个函数可以预测,基于环形的线性延迟和树形的对数延迟以及降低的带宽)。但是随着规模的扩大,有许多因素可能导致实际性能与预测相差甚远,比如路由。另外请注意,在IB网络上你可以使用SHARP;这样你的延迟在扩展时基本保持不变,带宽也不会降低太多,而且总是优于环形和树形算法。


## 面向性能的NCCL环境变量

虽然NCCL在为任何给定网络自动找出最佳性能方面表现出色,但有时它需要一些帮助,在这种情况下,以下NCCL环境变量用于调整性能。让我们看看几个你可能想要了解的常见变量,完整列表可以在这里找到(https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)。

### `NCCL_ALGO`

这个变量定义了NCCL将使用的算法。通常是以下之一:

1. Tree(树形)
2. Ring(环形)
3. CollnetDirect和CollnetChain (IB SHARP)
4. NVLS (NVLink SHARP)

我在这个NCCL问题(https://github.com/NVIDIA/nccl/issues/790)中询问了用户如何进行优化,得到的回答基本上是用户不应该尝试优化任何东西,因为NCCL内部有大量智能算法,会根据具体情况自动从一种算法切换到另一种算法。

Sylvain Jeaugey分享道:

> 以前有一个静态阈值,但它已被更复杂的调优系统取代。新系统为每种算法/协议组合(有很多很多组合)建立延迟和带宽模型,并根据大小决定哪种组合应该表现最佳。所以不再有环境变量和静态值,这很好,因为每种算法的性能取决于节点数量和每个节点的GPU数量,因此我们需要在算法/协议的二维空间中导航,这并不容易。你总是可以用`NCCL_ALGO=TREE`和`NCCL_ALGO=RING`强制使用一种算法,看看你得到什么性能,以及NCCL是否在正确的点切换。我知道这很难理解,但这也是我们找到的在所有平台和用户中获得最佳性能的最佳解决方案,而无需用户手动调整切换点。缺点是,如果你想手动调整东西,你做不到。

如果你使用`NCCL_ALGO`,你需要列出要考虑的算法,但除此之外你无法控制它。所以,这真的只有在你想确保不使用某种算法时才有用。

当询问哪种算法更好时,我得到的回答是:

> 粗略来说，环形（ring）在峰值带宽方面表现更优（除了在2个节点的情况下），而树形（tree）在基础延迟方面表现更好（尤其是在扩展时）。`带宽 = 大小 / 时间`，所以对于给定大小的数据，无论您关注时间还是带宽，都将是峰值带宽和基础延迟的组合。对于固定大小的数据，随着规模的扩大，环形的基础延迟将变得更加突出，这时树形结构会表现得更好。

还有一种新算法,名为`NVLS`,如果NVLink SHARP可用,它将比NVLink本身运行得更快,例如,使用NVLink 4.0 (450GBps)可以在进行all-reduce基准测试时达到480GBps。他们正在开发需要IB或RoCE(https://github.com/NVIDIA/nccl/issues/1031#issuecomment-1773965518)的节点间版本 - 截至本文撰写时,这种新算法还没有在任何地方记录。

最后,如果你想知道正在使用哪种算法 - 你不能 - 请参阅这个答案(https://github.com/NVIDIA/nccl/issues/754#issuecomment-1346163469)。所以如果你想知道哪种算法给出哪种吞吐量,你必须通过设置`NCCL_ALGO`环境变量来明确尝试所有算法,然后你就会知道选择了哪一个。或者你可以按照同一答案中的建议编辑和重新编译NCCL,但你不会希望在生产环境中这样做。


### `NCCL_CROSS_NIC`

`NCCL_CROSS_NIC` 变量控制 NCCL 是否允许环形/树形结构使用不同的网络接口卡（NIC），从而导致节点间通信在不同节点上使用不同的 NIC。

为了在使用多个 NIC 时最大化节点间通信性能，NCCL 尝试在节点之间使用相同的 NIC 进行通信，以适应每个节点的每个 NIC 连接到不同网络交换机（网络轨道）的网络设计，并避免任何流量干扰的风险。因此，NCCL_CROSS_NIC 设置取决于网络拓扑，特别是取决于网络结构是否针对轨道进行了优化。

这对只有一个 NIC 的系统没有影响。

接受的值：

- 0：始终为同一环形/树形结构使用相同的 NIC，以避免跨网络轨道。适用于每个 NIC 都有交换机（轨道）的网络，且轨道间连接较慢。注意，在某些特殊情况下，NCCL 可能仍会导致跨轨道通信，因此轨道仍需在顶层连接。
- 1：不尝试为同一环形/树形结构使用相同的 NIC。这适用于一个节点的所有 NIC 都连接到同一交换机的网络，因此尝试通过相同的 NIC 进行通信并不能帮助避免流量冲突。
- 2：（默认值）尝试为同一环形/树形结构使用相同的 NIC，但如果使用不同 NIC 能带来更好的性能，则允许使用不同的 NIC。

# Benchmark相关脚本

## `all_reduce_bench.py`

代码路径：https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/all_reduce_bench.py

```python
#!/usr/bin/env python

"""

这个程序的最新版本可以在 https://github.com/stas00/ml-engineering 找到

这个基准测试与 https://github.com/NVIDIA/nccl-tests 非常相似，但它更容易设置，因为它只需要安装 PyTorch

这个版本:
- 源自 @jeffra 的 gist: https://gist.github.com/jeffra/b5e80466b4c86be00ea3b6f130fb7a36
- 而该 gist 又源自 https://github.com/NVIDIA/nccl-tests 中的逻辑
- 贡献者包括:
  * Indu Thangakrishnan https://github.com/indhub 使用 cuda 事件正确处理计时

重要说明:

- 当你完成运行这个基准测试后，你应该关注 busbw 结果（而不是 algbw），正如这里解释的 https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#bandwidth

- 与 NVIDIA/nccl-tests 类似，这个基准测试测量的是单向带宽 - 所以将结果与广告宣传的单向峰值吞吐量进行比较，而不是双向（全双工）

- 目前这个基准测试测试的是 4GB 的有效载荷（M * N * 4）。如果你的目标应用使用的有效载荷小得多，你应该修改 M*N*4 以匹配目标有效载荷。要计算有效载荷，使用每次归约发送的参数数量乘以 2（bf16/fp16）或 4（fp32）。例如，如果一次归约是 1B 参数的单个层，并且你使用 bf16 梯度，那就是 2GB 的有效载荷。根据你使用的框架（DDP、FSDP、DeepSpeed ZeRO），它们在发送多大的消息大小上都使用不同的逻辑。

- 如果你想知道是否还需要运行 https://github.com/NVIDIA/nccl-tests - 我已经验证过，我用 ./build/all_reduce_perf -b 4G -e 4G 得到了非常相似的结果（在 4 个节点上用 mpirun 测试）。它应该要么相当，要么稍慢一些，因为它使用阻塞方法 - 也就是说，它等待每个新的 all_reduce 完成后才触发下一个，而 nccl-tests 以异步方式触发它们（你可以在 nccl-tests 中添加 `-z` 来模拟阻塞）

- 要对其他集合操作进行基准测试，请使用 nccl-tests。如果你想测试一系列有效载荷，它也很有用，例如，你可以设置 -b 8 -e 4G -f 2，它会自动测试许多大小。

在 4 个节点上运行:

GPUS_PER_NODE=8
NNODES=4
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --role `hostname -s`: \
    --tee 3 \
    all_reduce_bench.py

注意：如果不是自动获取主机名的SLURM环境，请将MASTER_ADDR调整为节点rank 0的主机名

例如，使用salloc+srun运行的示例：

salloc --partition=mypartition --nodes=4 --ntasks-per-node=1 --cpus-per-task=48 --gres=gpu:8 --time=1:00:00 bash

srun --gres=gpu:8 --nodes=4 --tasks-per-node=1 python -u -m torch.distributed.run --nproc_per_node=8 \
--nnodes 4 --rdzv_endpoint $(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1):6000 --rdzv_backend \
c10d all_reduce_bench.py

要在2个GPU上进行快速测试:

python -u -m torch.distributed.run --nproc_per_node=2 --rdzv_endpoint localhost:6000  --rdzv_backend c10d \
all_reduce_bench.py

"""

import os
import socket
import torch
import torch.distributed as dist

TRIALS = 5

# 这些模拟将成为一个 M * N * 4 大小的张量的有效载荷
N = 500000
M = 2000

def timed_allreduce(mat, start_event, end_event):
    dist.barrier()
    start_event.record()
    dist.all_reduce(mat)
    end_event.record()

    torch.cuda.synchronize()
    duration = start_event.elapsed_time(end_event) / 1000

    n = dist.get_world_size()
    size = M * N * 4 # 4 是 fp32 中的 4 字节
    # 注意这里遵循与 NVIDIA/nccl-tests 相同的计算方法
    algbw = torch.tensor([size / duration]).cuda(local_rank)

    # 计算所有 rank 的平均值
    dist.reduce(algbw, dst=0, op=dist.ReduceOp.SUM)
    algbw /= n

    return algbw

def run(local_rank):
    hostname = socket.gethostname()
    is_global_rank_0 = dist.get_rank() == 0

    mat = torch.rand(N, M, dtype=torch.float32).cuda(local_rank)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # 进行几次预热迭代
    for i in range(2):
        timed_allreduce(mat, start_event, end_event)

    # 实际基准测试
    algbw_gather = []
    for i in range(TRIALS):
        if is_global_rank_0:
            print(i+1)
        algbw_gather += timed_allreduce(mat, start_event, end_event)

    algbw = torch.mean(torch.stack(algbw_gather))

    # all-reduce 特有的 2*(n-1)/n busbw 校正因子在这里解释：
    # https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#allreduce
    # busbw 反映了硬件的使用效率
    n = dist.get_world_size()
    busbw = algbw * (2*(n - 1) / n)

    if is_global_rank_0:
        print(f"all_reduce 在 {M*N*4/1e9}GB 有效载荷下的平均带宽 ({TRIALS} 次试验, {n} 个 rank):\n",
              f"algbw: {algbw/1e9:.3f} GBps ({algbw*8/1e9:.1f} Gbps)\n",
              f"busbw: {busbw/1e9:.3f} GBps ({busbw*8/1e9:.1f} Gbps)\n",
        )

def init_processes(local_rank, fn, backend='nccl'):
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend)
    fn(local_rank)


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    init_processes(local_rank=local_rank, fn=run)
```

## `all_reduce_latency_comp.py`

> 这个脚本举例说明1次4GB规模的归约操作比1000次4MB规模的归约操作要快得多。

```python
#!/usr/bin/env python

# 这个脚本源自 all_reduce_bench.py
# 但经过调整以展示 1x 4GB 规模的归约操作比 1000x 4MB 规模的归约操作要快得多
#
# 在 8 个 GPU 上运行的命令:
# python -u -m torch.distributed.run --nproc_per_node=8 all_reduce_latency_comp.py

import os
import socket
import torch
import torch.distributed as dist

TRIALS = 1  # 实验重复次数

# 这些参数模拟将成为 M * N * 4 大小张量的有效载荷
N = 500000
M = 2000

def timed_allreduce(mat, repeat_times, id, start_event, end_event):
    start_event.record()
    for i in range(repeat_times):
        dist.all_reduce(mat)
    end_event.record()

    torch.cuda.synchronize()
    duration = start_event.elapsed_time(end_event) / 1000  # 转换为秒

    size = M * N * 4  # 4 是 fp32 的字节数
    algbw = (size / duration) * 8  # 8 是字节转比特
    n = dist.get_world_size()
    # all-reduce 特有的 2*(n-1)/n busbw 校正因子在这里解释：
    # https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#allreduce
    # busbw 反映了硬件的使用效率
    busbw = algbw * (2*(n - 1) / n)

    # 在全局 rank 0 上收集所有数据并打印结果，以避免打印交错
    data = [id, duration, algbw, busbw]
    output = [None for _ in range(dist.get_world_size())] if dist.get_rank() == 0 else None
    dist.gather_object(data, output, dst=0)
    if dist.get_rank() == 0:
        for data in output:
            id, duration, algbw, busbw = data
            print(f"{id}:\n",
                  f"duration: {duration:.3f} sec\n",
                  f"algbw: {algbw/1e9:.3f} Gbps\n",
                  f"busbw: {busbw / 1e9:.3f} Gbps"
    )

def run(local_rank):
    hostname = socket.gethostname()
    id = f"{hostname}:{local_rank}"
    global_rank = dist.get_rank()

    chunks = 1000
    mat1 = torch.rand(N, M, dtype=torch.float32).cuda(local_rank)  # 4GB 张量
    mat2 = torch.rand(int(N/chunks), M, dtype=torch.float32).cuda(local_rank)  # 4MB 张量

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for i in range(TRIALS):
        dist.barrier()  # 同步所有进程

        if global_rank == 0:
            print(f"\n\n\n----------- 1x {N*M*4/1e9}GB ----------------")
        timed_allreduce(mat1, 1, id, start_event, end_event)  # 测试单次 4GB all-reduce

        if global_rank == 0:
            print(f"\n\n\n----------- {chunks}x {(N*M*4/chunks)/1e9}GB ----------------")
        timed_allreduce(mat2, chunks, id, start_event, end_event)  # 测试 1000 次 4MB all-reduce

def init_processes(local_rank, fn, backend='nccl'):
    torch.cuda.set_device(local_rank)  # 设置当前进程使用的 GPU
    dist.init_process_group(backend)  # 初始化分布式环境
    fn(local_rank)

if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])  # 获取本地 rank
    print("local_rank: %d" % local_rank)
    init_processes(local_rank=local_rank, fn=run)
```

## `all_gather_object_vs_all_reduce.py`

> 使用all_reduce在进程组间收集计数比使用all_gather_object快23倍。

```python
#!/usr/bin/env python

#
# 使用all_reduce在进程组间收集计数比使用all_gather_object快23倍
#
# 运行命令: python -m torch.distributed.run --nproc_per_node 2 all_gather_object_vs_all_reduce.py
#
# 示例输出:
# all_gather_object=0.26279118900129106
# all_gather_object=0.2628160299973388
# all_reduce       =0.011241967000387376
# all_reduce       =0.011610440000367817

import torch.distributed as dist
import torch
import os

# 获取本地进程的rank
local_rank = int(os.environ["LOCAL_RANK"])
# 设置当前进程使用的GPU
torch.cuda.set_device(local_rank)
# 初始化分布式环境
dist.init_process_group("nccl")
# 设置设备为GPU(如果可用)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 获取进程组的大小和当前进程的rank
world_size = dist.get_world_size()
rank = dist.get_rank()

# 创建用于测试的张量和Python对象
flag_pt = torch.tensor(1.0, device=device)
flag_py = 1

def all_gather_object():
    # 创建一个列表来存储收集的对象
    output_objects = [None for _ in range(world_size)]
    # 使用all_gather_object收集所有进程的对象
    dist.all_gather_object(output_objects, flag_py)
    # 对收集的对象求和
    flag = sum(output_objects)
    return flag

def all_reduce():
    # 使用all_reduce对张量进行求和操作
    dist.all_reduce(flag_pt, op=dist.ReduceOp.SUM)
    return flag_pt

# 测试两个函数
print(f"all_gather_object: {all_gather_object()}\n")
print(f"all_reduce: {all_reduce()}\n")

import timeit
# 使用timeit模块测量两个函数的执行时间(运行1000次)
print(f'all_gather_object={timeit.Timer("all_gather_object()", globals=globals()).timeit(number=1000)}')
print(f'all_reduce       ={timeit.Timer("all_reduce()"       , globals=globals()).timeit(number=1000)}')
```

# 禁用NVLink基准测试

让我们比较一下在wikitext的小样本上训练gpt2语言模型的情况。

结果如下：

| NVlink | 时间 |
| -----  | ---: |
| 是     | 101秒 |
| 否     | 131秒 |

你可以看到，使用NVLink完成训练的速度快了约23%。在第二个基准测试中，我们使用`NCCL_P2P_DISABLE=1`来告诉GPU不要使用NVLink，而是使用PCIe。

我们将使用HF Transformers示例(https://github.com/huggingface/transformers/blob/58e3d23e97078f361a533b9ec4a6a2de674ea52a/examples/pytorch/language-modeling/run_clm.py)。

以下是完整的基准测试代码和输出：

```bash
# DDP w/ NVLink

rm -r /tmp/test-clm; CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node 2 examples/pytorch/language-modeling/run_clm.py --model_name_or_path gpt2 \
--dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train \
--output_dir /tmp/test-clm --per_device_train_batch_size 4 --max_steps 200

{'train_runtime': 101.9003, 'train_samples_per_second': 1.963, 'epoch': 0.69}

# DDP w/o NVLink

rm -r /tmp/test-clm; CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 python -m torch.distributed.launch \
--nproc_per_node 2 examples/pytorch/language-modeling/run_clm.py --model_name_or_path gpt2 \
--dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train
--output_dir /tmp/test-clm --per_device_train_batch_size 4 --max_steps 200

{'train_runtime': 131.4367, 'train_samples_per_second': 1.522, 'epoch': 0.69}
```

硬件: 2块TITAN RTX 24GB显卡 + 2条NVLink连接 (`nvidia-smi topo -m`中显示为`NV2`)
软件: `pytorch-1.8-to-be` + `cuda-11.0` / `transformers==4.3.0.dev0`

