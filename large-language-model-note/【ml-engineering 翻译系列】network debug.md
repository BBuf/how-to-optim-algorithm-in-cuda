> 我的课程笔记，欢迎关注：https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode 

> 本篇文档的来源：https://github.com/stas00/ml-engineering 。这篇文档主要介绍了如何诊断和解决NCCL多GPU和多节点连接问题。它详细说明了如何使用NCCL调试信息来识别网络接口和协议问题，如何正确配置网络接口，以及如何在Docker容器中使用NCCL。文档还介绍了检查GPU P2P支持的方法，如何统计NCCL调用次数，以及一些有用的NCCL调试环境变量。此外，文档提供了一个用于测试分布式GPU设置的Python脚本。

# 网络调试

通常你不需要成为网络工程师就能解决网络问题。一些常见问题可以通过阅读以下注意事项来解决。

## 术语表

- OOB: 带外（通常是较慢的以太网网卡）
- Bonding: 将多个网卡绑定在一起以获得更快的速度或作为备份
- IB: InfiniBand（最初由Mellanox开发，后被NVIDIA收购）
- NIC: 网络接口卡

## 如何诊断NCCL多GPU和多节点连接问题

本节肯定不是详尽无遗的，旨在涵盖我经常遇到的一些最常见的设置问题。对于更复杂的问题，请研究NCCL仓库的Issues(https://github.com/NVIDIA/nccl/issues)，或者如果找不到与你情况匹配的问题，请提交一个新的Issue。NCCL还包括一个简短的故障排除部分(https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2183/user-guide/docs/troubleshooting.html)，但通常通过阅读Issues(https://github.com/NVIDIA/nccl/issues)可以学到更多。

对于网络诊断工作，我建议使用这个专门开发的设计测试脚本：`torch-distributed-gpu-test.py` (https://github.com/stas00/ml-engineering/blob/master/debug/torch-distributed-gpu-test.py)，而不是使用可能需要很长时间启动并有不相关问题的完整应用程序。

首先，在设置以下环境变量后运行基于nccl的程序：

```
export NCCL_DEBUG=INFO
```
这将打印大量关于NCCL设置及其网络流量的调试信息。

例如，如果你使用上述调试脚本，对于一个有8个GPU的单节点，你可能会这样做：

```
NCCL_DEBUG=INFO python -m torch.distributed.run --nproc_per_node 8 --nnodes 1 torch-distributed-gpu-test.py
```

要在多个节点上启动它,你需要使用像SLURM或Kubernetes这样的编排软件,或者在每个节点上手动启动它(`pdsh`会非常有帮助) - 详细信息请参见`torch-distributed-gpu-test.py`(https://github.com/stas00/ml-engineering/blob/master/debug/torch-distributed-gpu-test.py)中的说明。但为了理解事物如何工作,我建议从1个节点开始,然后进展到2个节点,之后再到更多节点。

现在,检查程序的输出并查找以以下内容开头的行:
```
NCCL INFO NET/
```
然后检查它使用的是哪种协议和哪些接口。

例如,这样的输出:
```
NCCL INFO NET/FastSocket : Using [0]ibs108:10.0.19.12<0> [1]ibs109:10.0.19.13<0> [2]ibs110:10.0.19.14<0> [3]ibs111:10.0.19.15<0> [4]ibs112:10.0.19.16<0> [5]ibs113:10.0.19.17<0> [6]ibs114:10.0.19.18<0> [7]ibs115:10.0.19.19<0>
```

告诉我们使用了nccl-fastsocket(https://github.com/google/nccl-fastsocket)传输层插件，并发现了8个`ibs*`网络接口（NIC卡）。如果你使用的是Google Cloud，这是正确的，你的NCCL可能已经正确设置。但是如果你使用的是InfiniBand (IB)并得到上述输出，你可能会遇到非常低的节点间速度，因为这意味着你激活了错误的插件。

对于IB的情况，你希望看到的是`NET/IB`及其IB接口：
```
NCCL INFO NET/IB : Using [0]mlx5_0:1/IB [1]mlx5_1:1/IB [2]mlx5_2:1/IB [3]mlx5_3:1/IB [4]mlx5_4:1/IB [5]mlx5_5:1/IB [6]mlx5_6:1/IB [7]mlx5_7:1/IB [RO]; OOB eno1:101.262.0.9<0>
```

在这里，你可以看到IB被用于集体通信，使用了8个`mlx5_*`接口，还有一个OOB（代表带外通信）。OOB用于引导连接，通常使用较慢的以太网NIC（有时几个NIC会被绑定成一个(https://wiki.linuxfoundation.org/networking/bonding) - 如果你想知道接口名称中的`bond`是什么意思）。

要了解你的节点有哪些TCP/IP接口，你可以在其中一个节点上运行`ifconfig`命令（通常所有类似的节点都会有相同的接口名称，但并非总是如此）。

如果你的集群通信网络是IB，那么你应该运行`ibstat`而不是`ifconfig`。上面`NCCL INFO NET`的最后一个例子将对应以下输出：

```
$ ibstat | grep mlx5
CA 'mlx5_0'
CA 'mlx5_1'
CA 'mlx5_2'
CA 'mlx5_3'
CA 'mlx5_4'
CA 'mlx5_5'
CA 'mlx5_6'
CA 'mlx5_7'
```

除了快速的节点间连接NIC之外,你很可能还有一个慢速的管理以太网NIC(甚至可能有几个),它用于配置节点、使用共享文件系统、访问互联网等。因此,`ifconfig`几乎肯定会包含额外的NIC。你也可能有一个docker网络接口、`lo`回环接口和其他一些接口。例如,在我的桌面上,我可能会得到以下输出:

```
$ ifconfig
docker0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 172.99.0.1  netmask 255.255.0.0  broadcast 172.99.255.255
        inet6 f330::42:fe33:f335:7c94  prefixlen 64  scopeid 0x20<link>
        ether 02:42:fe:15:1c:94  txqueuelen 0  (Ethernet)
        RX packets 219909  bytes 650966314 (650.9 MB)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 262998  bytes 20750134 (20.7 MB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0

lo: flags=73<UP,LOOPBACK,RUNNING>  mtu 65536
        inet 127.0.0.1  netmask 255.0.0.0
        inet6 ::1  prefixlen 128  scopeid 0x10<host>
        loop  txqueuelen 1000  (Local Loopback)
        RX packets 1147283113  bytes 138463231270 (138.4 GB)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 1147283113  bytes 138463231270 (138.4 GB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0

eth0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 10.0.0.23  netmask 255.255.255.0  broadcast 10.0.0.255
        inet6 2601:3108:1c71:600:4224:7e4b:13e4:7b54  prefixlen 64  scopeid 0x0<global>
        ether 04:41:1a:16:17:bd  txqueuelen 1000  (Ethernet)
        RX packets 304675330  bytes 388788486256 (388.7 GB)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 74956770  bytes 28501279127 (28.5 GB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0
        device memory 0xa3b00000-a3bfffff
```

我提到所有这些的原因是,关键部分是确保NCCL在其`Using`调试行中只报告正确的接口。如果最终报告了任何像`docker0`、`lo`或`eth0`这样的接口,例如:

```
NCCL INFO NET/Socket : Using [0]eth0:10.0.0.23<0>
```

如果你有更快的网络接口可用,这很可能不是你想要的。当然,在某些情况下,以太网NIC可能是你唯一的选择,那么上述情况就没问题 - 只是会非常慢。

有时,如果使用了错误的接口,应用程序可能会直接挂起。

如果你同时拥有正确和错误的接口,NCCL可能会工作,但速度会较慢。

如果是云环境,通常你的云服务提供商应该给你正确设置的说明。如果他们没有提供,那么你至少需要询问他们应该使用哪些网络接口来设置NCCL。

虽然NCCL会尽力自动发现应该使用哪些接口,但如果它无法正确做到这一点,你可以通过告诉它使用或不使用哪些接口来帮助它:

- 当不使用Infiniband时,可以使用`NCCL_SOCKET_IFNAME`来指定包含或排除哪些`ifconfig`接口。以下是一些例子:

```
export NCCL_SOCKET_IFNAME=eth:        Use all interfaces starting with eth, e.g. eth0, eth1, …
export NCCL_SOCKET_IFNAME==eth0:      Use only interface eth0
export NCCL_SOCKET_IFNAME==eth0,eth1: Use only interfaces eth0 and eth1
export NCCL_SOCKET_IFNAME=^docker:    Do not use any interface starting with docker
export NCCL_SOCKET_IFNAME=^=docker0:  Do not use interface docker0.
```
The full doc is here(https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-socket-ifname).

- When using IB RDMA (IB Verbs interfaces), instead of `NCCL_SOCKET_IFNAME` use `NCCL_IB_HCA` env var which selects the interfaces for the collective communications. Examples:

```
export NCCL_IB_HCA=mlx5 :               # 使用所有以mlx5开头的卡的所有端口
export NCCL_IB_HCA==mlx5_0:1,mlx5_1:1 : # 使用mlx5_0和mlx5_1卡的1号端口
export NCCL_IB_HCA=^=mlx5_1,mlx5_4 :    # 不使用mlx5_1和mlx5_4卡
```
完整文档可以在这里找到(https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-ib-hca)。

例如，在使用IB时，通常会有一些额外的接口，如`mlx5_bond_0`，你不希望将其包含在NCCL通信中。例如，以下报告表明错误地包含了`[8]mlx5_bond_0:1/RoCE`接口，这几乎肯定会导致低带宽：
```
NCCL INFO NET/IB : Using [0]mlx5_0:1/IB [1]mlx5_1:1/IB [2]mlx5_2:1/IB [3]mlx5_3:1/IB [4]mlx5_4:1/IB [5]mlx5_5:1/IB [6]mlx5_6:1/IB [7]mlx5_7:1/I [8]mlx5_bond_0:1/RoCE [RO]; OOB ibp25s0:10.0.12.82<0>
```
在这种情况下，你可以通过以下方式排除它：
```
export NCCL_IB_HCA=^mlx5_bond_0:1
```
或者你也可以明确列出你想要使用的接口，例如：
```
export NCCL_IB_HCA==mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
```

如前所述，在使用IB互连的节点上使用`ibstat`命令将显示可用的IB接口。

由于NCCL会尝试自动选择最佳的网络接口，只有在NCCL不工作或运行缓慢时，你才需要进行上述操作。在正常情况下，NCCL应该可以开箱即用，用户无需进行任何特殊操作。

此外，根据使用的云服务不同，提供商很可能会给你一系列需要设置的环境变量。如果你错误地设置了其中一些变量，NCCL可能会运行缓慢或完全无法工作。

用户遇到的另一个典型问题是，当他们尝试在云B上重用在云A上正常工作的NCCL设置时。通常情况下，这些设置无法直接转换，需要仔细删除之前设置的所有环境变量，并为新的云环境重新正确设置。即使你使用的是同一个云服务提供商，但使用不同类型的实例，也可能出现这个问题，因为某些网络设置是针对特定实例的，在其他地方可能无法正常工作。

一旦你认为已经正确设置了NCCL，下一步就是对你的连接进行基准测试，确保它与宣传的速度相匹配（大约达到宣传速度的80%）。请前往基准测试章节(https://github.com/stas00/ml-engineering/tree/master/network/benchmarks)。


## 在Docker容器中使用NCCL

* 通过在docker `run`命令中添加以下额外参数来提供足够的资源：`–shm-size=1g –ulimit memlock=-1`（更多详情请参阅(https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2183/user-guide/docs/troubleshooting.html#sharing-data)）
* 特权访问：有时你需要在docker `run`参数中添加`--privileged`。
* 确保Docker镜像包含正确的软件包，例如，如果使用IB，你至少需要安装`libibverbs1 librdmacm1`



## 如何检查是否支持P2P

有时你需要知道计算节点上的GPU是否支持P2P访问（Peer2Peer）。禁用P2P通常会导致节点内连接速度变慢。

你可以看到在这个特定的8个NVIDIA H100节点上，P2P是被支持的：

```
$ nvidia-smi topo -p2p r
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7
 GPU0   X       OK      OK      OK      OK      OK      OK      OK
 GPU1   OK      X       OK      OK      OK      OK      OK      OK
 GPU2   OK      OK      X       OK      OK      OK      OK      OK
 GPU3   OK      OK      OK      X       OK      OK      OK      OK
 GPU4   OK      OK      OK      OK      X       OK      OK      OK
 GPU5   OK      OK      OK      OK      OK      X       OK      OK
 GPU6   OK      OK      OK      OK      OK      OK      X       OK
 GPU7   OK      OK      OK      OK      OK      OK      OK      X

Legend:

  X    = Self
  OK   = Status Ok
  CNS  = Chipset not supported
  GNS  = GPU not supported
  TNS  = Topology not supported
  NS   = Not supported
  U    = Unknown
```

另一方面，对于这个特定的2个NVIDIA L4 GPU，P2P是不被支持的：

```
$ nvidia-smi topo -p2p r
        GPU0    GPU1
 GPU0   X       CNS
 GPU1   CNS     X
```

从图例中可以看出，`CNS`表示"芯片组不支持"。

如果你使用的是高端数据中心GPU，这种情况很少发生。不过，一些低端数据中心GPU可能不支持P2P，就像上面L4的例子。

对于消费级GPU，可能有多种原因导致你的GPU不被支持，通常是因为启用了IOMMU和/或ACS功能。有时只是驱动程序版本的问题。如果你花些时间搜索，可能会发现有人破解驱动程序以在不应支持P2P的GPU上启用P2P，比如这个4090 P2P支持仓库(https://github.com/tinygrad/open-gpu-kernel-modules)。

要检查是否启用了PCI访问控制服务（ACS）并禁用它们，请按照这个指南操作(https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2183/user-guide/docs/troubleshooting.html#pci-access-control-services-acs)。

IOMMU可以在BIOS中禁用。

你还可以使用torch检查特定GPU之间的P2P支持 - 这里我们检查GPU 0和1：

```python
python -c "import torch; print(torch.cuda.can_device_access_peer(torch.device('cuda:0'), torch.device('cuda:1')))"
```

## 如何统计NCCL调用次数

为子系统启用NCCL调试日志 - 集体通信操作:
```
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL
```

如果你在一个有多个节点的Slurm环境中工作，你可能只想在rank 0上执行这个操作，像这样：
```
if [[ $SLURM_PROCID == "0" ]]; then
  export NCCL_DEBUG=INFO
  export NCCL_DEBUG_SUBSYS=COLL
fi
```

假设你的所有日志都被发送到了`main_log.txt`，你可以通过以下方式统计每种集合通信调用的执行次数：
```
grep -a "NCCL INFO Broadcast" main_log.txt     | wc -l
2590
grep -a "NCCL INFO AllReduce" main_log.txt     | wc -l
5207
grep -a "NCCL INFO AllGather" main_log.txt     | wc -l
1849749
grep -a "NCCL INFO ReduceScatter" main_log.txt | wc -l
82850
```

首先隔离训练的特定阶段可能是个好主意，因为加载和保存与训练迭代相比会有非常不同的模式。

所以我通常会先切片出一次迭代。例如，如果每次迭代的日志以 `iteration: ...` 开头，那么我会先执行以下操作：
```
csplit main_log.txt '/iteration: /' "{*}"
```
然后分析对应迭代的结果文件之一。默认情况下，它的名称会类似于 `xx02`。


## 有用的 NCCL 调试环境变量

以下环境变量在调试 NCCL 相关问题（如挂起和崩溃）时最为有用。完整列表可以在[这里](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)找到。


### `NCCL_DEBUG`

这是调试网络问题最常用的环境变量。

取值：
- `VERSION` - 在程序开始时打印 NCCL 版本。
- `WARN` - 当任何 NCCL 调用出错时打印明确的错误消息。
- `INFO` - 打印调试信息。
- `TRACE` - 在每次调用时打印可重放的跟踪信息。

例如：

```bash
NCCL_DEBUG=INFO python -m torch.distributed.run --nproc_per_node 2 --nnodes 1 torch-distributed-gpu-test.py
```

这将输出大量与NCCL相关的调试信息，如果发现报告了一些问题，你可以在网上搜索这些信息。

当使用`NCCL_DEBUG`时，`NCCL_DEBUG_FILE`应该非常有用，因为信息量很大，特别是在使用多个节点的情况下。



### `NCCL_DEBUG_FILE`

当使用`NCCL_DEBUG`环境变量时，将所有NCCL调试日志输出重定向到一个文件。

默认输出到`stdout`。当使用多个GPU时，将每个进程的调试信息保存到其自己的日志文件中会非常有用，可以这样做：

```
NCCL_DEBUG_FILE=/path/to/nccl-log.%h.%p.txt
```

- `%h` 会被替换为主机名
- `%p` 会被替换为进程的 PID。

如果你需要一次性分析数百个这样的文件，以下是一些有用的快捷方式：

- 使用grep搜索特定匹配项，并同时打印出找到匹配项的文件名和行号：

```
grep -n "Init COMPLETE" nccl-log*
```

- 显示所有 nccl 日志文件的最后一行，后跟每个文件的名称

```
find . -name "nccl*" -exec sh -c 'echo "$(tail -1 "$1") ($1)"' _ {} \;
```



### `NCCL_DEBUG_SUBSYS`

`NCCL_DEBUG_SUBSYS` 与 `NCCL_DEBUG` 结合使用，告诉后者显示哪些子系统。通常情况下，你不需要指定这个变量，但有时帮助你的开发人员可能会要求将输出限制在某些子系统，例如：

```
NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV,TUNING
```

### `NCCL_P2P_DISABLE`

禁用P2P通信 - 例如，如果存在NVLink，它将不会被使用，因此性能会大大降低。通常情况下你不会想要这样做，但在紧急情况下，有时在调试过程中这可能会有用。


### `NCCL_SOCKET_IFNAME`

如果你有多个网络接口并且想要选择使用特定的一个，这个选项非常有用。

默认情况下，NCCL 会尝试使用最快类型的接口，通常是 `ib`（InfiniBand）。

但是，假设你想使用以太网接口，那么你可以通过以下方式覆盖默认设置：

```
NCCL_SOCKET_IFNAME=eth
```

这个环境变量有时可以用来调试连接问题，比如说如果其中一个接口被防火墙阻挡，而其他接口可能没有被阻挡并可以尝试使用。或者如果你不确定某个问题是否与网络接口有关还是其他原因，那么测试其他接口可以帮助排除问题是否来自网络。

## 补充：torch-distributed-gpu-test.py

> 代码位置：https://github.com/stas00/ml-engineering/blob/master/debug/torch-distributed-gpu-test.py

```python
#!/usr/bin/env python

'''
这是一个`torch.distributed`诊断脚本，用于检查集群中的所有GPU（单个或多个节点）是否可以通过nccl相互通信并分配GPU内存。它还会打印其他有用的信息，如NUMA亲和性。

要运行它，你只需要根据你的使用情况调整进程数和节点数：

'''
python -m torch.distributed.run --nproc_per_node 2 --nnodes 1 torch-distributed-gpu-test.py
'''

如果使用自定义的地址和端口，你可能需要添加 `--master_addr $MASTER_ADDR --master_port $MASTER_PORT`

你也可以使用 rdzv API: `--rdzv_endpoint $MASTER_ADDR:$MASTER_PORT --rdzv_backend c10d`

如果在 `barrier` 调用中出现挂起，说明你可能存在一些网络问题，你可以尝试使用以下方法进行调试:

'''
NCCL_DEBUG=INFO python -m torch.distributed.run --nproc_per_node 2 --nnodes 1 torch-distributed-gpu-test.py
'''

这应该能告诉你幕后发生了什么。

这个脚本也可以在SLURM环境中通过`srun`运行。以下是一个在2个节点上运行,每个节点8个GPU的SLURM脚本:

'''
#!/bin/bash
#SBATCH --job-name=test-nodes        # name
#SBATCH --nodes=2                    # EDIT to the number of nodes
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per node for this script
#SBATCH --cpus-per-task=10           # EDIT this to how many cpu cores the node has
#SBATCH --gres=gpu:8                 # EDIT this if it's not an 8-GPUs node setup
#SBATCH --partition=dev              # EDIT to the desired partition name
#SBATCH --time 0:05:00               # 5 min should be enough
#SBATCH --output=%x-%j.out           # output file name

export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=6000

srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
--nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
--master_addr $MASTER_ADDR --master_port $MASTER_PORT \
torch-distributed-gpu-test.py'
'''

你还可以在启动器中添加以下内容,以自动为所有日志添加 `[hostname:rank] ` 前缀(例如在 `--master_addr` 之后):

--role `hostname -s`: --tee 3


'''

import builtins
import fcntl
import os
import socket
import torch
import torch.distributed as dist

def print(*args, **kwargs):
    """ solves multi-process interleaved print problem """
    with open(__file__, "r") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            builtins.print(*args, **kwargs)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
hostname = socket.gethostname()

gpu = f"[{hostname}:{local_rank}]"

try:
    # XXX: possibly change the dist timeout to something much shorter to get this script to fail
    # fast if there is a problem and not wait for the default 30min

    # test distributed
    dist.init_process_group("nccl")

    # global rank
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # reduction test
    t = torch.ones(1, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    dist.barrier()
    print(f"{gpu} Reduction op=sum result: {t.item()}")

    # test cuda is available and can allocate memory
    torch.cuda.is_available()
    torch.ones(1).cuda(local_rank)

    print(f"{gpu} is OK (global rank: {rank}/{world_size})")

    dist.barrier()
    if rank == 0:
        print(f"pt={torch.__version__}, cuda={torch.version.cuda}, nccl={torch.cuda.nccl.version()}")
        print(f"device compute capabilities={torch.cuda.get_device_capability()}")
        print(f"pytorch compute capabilities={torch.cuda.get_arch_list()}")

except Exception:
    print(f"{gpu} is broken (but it could also mean that it failed because another gpu didn't respond)")
    raise
```

