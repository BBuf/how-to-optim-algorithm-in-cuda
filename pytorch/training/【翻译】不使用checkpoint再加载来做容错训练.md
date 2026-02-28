> 来源：https://pytorch.org/blog/fault-tolerant-llama-training-with-2000-synthetic-failures-every-15-seconds-and-no-checkpoints-on-crusoe-l40s/

# 在 Crusoe L40S 上每 15 秒进行 2000 次合成故障训练且无检查点

> 作者：Tristan Rice, Howard Huang，2025年6月20日

合作者：Less Wright, Howard Huang, Chien-Chin Huang, Crusoe: Martin Cala, Ethan Petersen

**简述**：我们使用 torchft(https://github.com/pytorch/torchft) 和 torchtitan(https://github.com/pytorch/torchtitan) 在真实环境中以极端的合成故障率训练模型，以证明容错训练的可靠性和正确性。

![](https://files.mdnice.com/user/59/18c5589b-8daa-413a-bd99-c57cc3c5a6f5.png)

注意：每个小峰值都是非参与工作节点的恢复，这会影响指标但不会影响模型

## 介绍

我们希望通过在最极端的故障率下运行训练作业来演示 torchft 在最坏情况下的表现。

大多数 LLM 预训练使用 FSDP 的分片模型。torchft 支持使用 FSDP2 的分片模型，它将分片模型与 torchft 的容错 DDP all reduce 相结合。我们已经将 torchft 集成到 torchtitan 中，因此您可以开箱即用地使用容错功能。torchft+titan 还支持每个副本组内的其他分片/并行化，如张量并行（TP）、流水线并行（PP）等。

以下是使用 torchft 的训练作业结构：

![](https://files.mdnice.com/user/59/9c560d20-7abc-4a6f-95f8-d1a0818231d8.png)

训练作业的结构。torchft 的容错 DDP 实现用于跨副本组同步梯度。标准 FSDP2 和其他并行化在每个副本组内使用。

![](https://files.mdnice.com/user/59/76018447-d8d3-4546-a1b0-4df1e6006c17.png)

torchft 使用全局 Lighthouse 服务器和每个副本组的管理器来进行工作节点的实时协调。Lighthouse 通过心跳了解所有工作节点的状态以及哪些是健康的。

torchft 实现了几种不同的容错算法。两个最主要的是：

- **容错 HSDP**：FSDPv2 的扩展，使用容错all reduce。这完全模拟了标准 HSDP 训练，具有每步梯度all reduce和每步容错。这最适合具有快速后端网络（如 infiniband）的大规模训练。
- **LocalSGD/DiLoCo**：半同步训练的容错实现。这些算法通过在指定间隔而不是像 HSDP 那样每步同步来最小化通信开销。这通常用于通信受限的训练场景，如以太网/TCP 或地理位置分离的位置（联邦学习或多数据中心训练）。

我们始终关注新算法，比如即将推出的流式 DiLoCo 支持。如果您有想要合作的新用例，请联系我们！

## 集群设置

Crusoe(https://crusoe.ai/) 慷慨地为我们提供了一个包含 300 个 L40S GPU 的集群。这些 GPU 分布在 30 台主机上，每台主机有 10 个 NVIDIA L40S GPU。

对于模型，我们使用 torchtitan 和具有 10 亿参数的 Llama 3 模型来匹配可用的硬件。

NVIDIA L40S GPU 通常用于推理，因此为我们提供了在非传统环境中测试 torchft 的机会，在这种环境中，由于较低的仅 TCP（无 infiniband/nvlink）网络瓶颈，DiLoCo 等算法真正发挥作用。L40S 有 48GB 的 VRAM（接近消费级 GPU），所以我们使用了较小的模型和批量大小。训练的平均步骤时间约为 9 秒。

为了在有限的网络下最大化性能，我们以 30x1x10 配置训练模型。我们有 30 个副本组（容错域），每个组有 1 台主机和 10 个 GPU/工作节点。torchft 可以在每个副本组中有很多很多主机，但对于这个集群，由于网络带宽有限，每个副本组单主机/10 GPU 的性能最佳。我们运行了 30 个副本组，因为更多的组会更多地压力测试协调和重新配置算法。

对于网络通信，我们在每个副本组内使用 NCCL 进行所有通信（即 FSDP），在副本组之间使用 Gloo 进行通信。Gloo 虽然通常性能不如 NCCL，但初始化速度更快，也能更快地失败，这对于快速检测故障很重要。torchft 确实支持在 IB 集群上使用 NCCL 的容错，但有一些注意事项，在这个演示中没有使用。由于我们想要最大化故障和恢复的总数，我们使用了 Gloo，因为它可以在我们的用例中在 <1 秒内重新初始化，我们能够将所有操作的超时设置为 5 秒。

对于容错算法，我们主要使用容错 HSDP 进行测试，因为它最能压力测试通信和仲裁层。对于最终测试，我们使用了 DiLoCo，它更适合基于以太网的集群。

## 无检查点恢复

传统机器学习通过在发生错误时从检查点重新加载来实现"容错"。这涉及完全停止世界的操作，其中所有工作节点重新启动并从最近持久化的检查点加载。

使用 torchft，我们专注于将故障隔离到单个 GPU 组。当该组内发生错误时，我们可以异步重新启动该组，所有其他组可以重新配置并继续训练而无需该组。

当该组通过重新启动或调度器替换机器恢复时，这些工作节点不再具有权重和优化器状态的有效副本。如果我们尝试从检查点恢复，其他组已经继续前进了。相反，我们依赖于运行时的异步权重传输。这进行从健康副本到恢复节点的点对点权重传输。

由于我们总是从另一个工作节点恢复 - 事实证明，只要我们能保证至少有一个组是健康的，我们实际上不需要任何检查点。对于这个演示，我们完全关闭了检查点，因为持久检查点的保存和加载比我们的 P2P 恢复时间要长得多。

以下是显示恢复副本（副本 1）如何加入仲裁并从健康对等节点（副本 0）恢复而不会有任何停机时间或影响健康工作节点训练的图表：

![](https://files.mdnice.com/user/59/d9b4d20f-c63d-45ca-bf74-70df18d76779.png)

torchft 采用了分布式数据库的许多概念：

- **仲裁操作**使用频繁的心跳确定哪些工作节点是健康的，并保证我们可以快速确定哪些工作节点是活跃的，以容错方式交换元数据，并强制执行无脑裂条件。
- 为了确保一致性并识别何时需要恢复工作节点，我们有效地使用传统数据库语义进行训练。传统数据库使用"事务"，其中每个操作要么被提交（完全应用）要么被回滚（丢弃）。torchft 以相同的方式处理每个训练步骤。副本组内的每个训练步骤都作为分布式事务处理，我们确保所有工作节点通过步进优化器提交步骤，或者如果发生错误，它们都通过丢弃梯度来回滚。

有关更多详细信息，请参阅 torchft README(https://github.com/pytorch/torchft/blob/main/README.md)，其中包含文档、设计文档和演示文稿的链接。

## 训练循环集成

TorchFT 已经与 TorchTitan 集成，因此启用它只需设置一个配置标志。对于典型模型，torchft 提供包装器，自动调用 torchft 管理器的钩子以提供容错功能。

```python
from torchft import Manager, DistributedDataParallel, Optimizer, ProcessGroupGloo

# 正常实例化您的模型和优化器
m = nn.Linear(2, 3)
optimizer = optim.AdamW(m.parameters())

# 设置 torchft 管理器并包装模型和优化器
manager = Manager(
    pg=ProcessGroupGloo(),
    load_state_dict=lambda state_dict: m.load_state_dict(state_dict),  # 加载状态字典的回调函数
    state_dict=lambda: m.state_dict(),  # 获取状态字典的回调函数
)
m = DistributedDataParallel(manager, m)  # 用容错 DDP 包装模型
optimizer = Optimizer(manager, optimizer)  # 用容错优化器包装

for batch in dataloader:
    # 当您调用 zero_grad 时，我们开始异步仲裁操作
    # 并在必要时执行异步权重恢复
    optimizer.zero_grad()

    out = m(batch)
    loss = out.sum()

    # 梯度all reduce将通过 torchft 的容错 ProcessGroupGloo 包装器完成
    loss.backward()

    # 优化器将根据是否发生任何错误有条件地步进
    # 如果梯度同步被中断，批次将被丢弃
    optimizer.step()
```

## 容错调度

我们可以使用标准的 ML 作业调度器（如 Slurm），因为副本组内工作节点的语义与正常作业相同。如果组内任何工作节点发生错误，我们期望整个组同时重新启动。在每个副本组内，应用程序是使用标准非容错操作的完全标准训练作业。

为了在传统调度器上实现容错，我们运行多个这样的作业。每个副本组在 Slurm 上作为单独的训练作业运行，Lighthouse 和监控脚本在头节点上运行。所有跨组通信都通过 torchft 的托管 ProcessGroup 和仲裁 API 完成。为了在故障时重新启动组并注入故障，我们使用了一个使用 torchx Python API 的小脚本。

监控脚本看起来像这样：

```python
from torchx.runner import get_runner

NUM_REPLICA_GROUPS = 30  # 副本组数量

with get_runner() as runner:
    while True:
        # 获取当前所有作业的列表
        jobs = runner.list(scheduler)
        
        # 找出当前活跃的副本组
        active_replicas = {
            parse_replica_id(job.name)
            for job in jobs
            if not job.is_terminal()  # 如果作业未终止
        }

        # 找出缺失的副本组
        missing_replicas = set(range(NUM_REPLICA_GROUPS)) - active_replicas

        # 为每个缺失的副本组启动新作业
        for replica_id in missing_replicas:
            app_def = make_app_def(replica_id=replica_id)  # 创建应用定义
            app_handle = runner.run(
                app_def, 
                scheduler="slurm",  # 使用 Slurm 调度器
                cfg={"partition": "batch"},  # 配置分区
            )
            print("launched:", replica_id, app_handle)

        time.sleep(5.0)  # 每 5 秒检查一次
```

故障是通过使用 scancel 取消特定副本组的 Slurm 作业来注入的。在真实世界场景中，我们期望故障由训练过程中的错误触发，这将使该副本组独立崩溃，而不是外部故障。

## 指标和日志

为了确保我们对作业有一致的视图，我们避免将故障注入到一个副本组中，以便更简单地跟踪作业的指标和仲裁事件。该组能够一致地记录参与者数量、步骤成功/失败和损失。

由于我们进行每步容错，参与者数量以及批量大小根据哪些工作节点健康而在每步变化。

损失使用跨副本组的all reduce在作业中的所有工作节点/副本组之间平均。

注意：下面损失图中的小峰值是由于我们如何在所有主机（包括恢复中的工作节点）之间平均损失，这些工作节点具有过时的权重，导致这些步骤的损失错误地更高。

## 运行结果

我们运行了三个不同的实验，展示了 torchft 的各种故障场景和功能。

### 运行 1：每 60 秒注入故障，共 1100 次故障

![](https://files.mdnice.com/user/59/f815a4ba-6f7e-40b7-968b-1114d329ba94.png)

这次运行持续了 19 小时多一点，共 6249 步。平均每步耗时 10.9 秒。

对于初始运行，我们每 60 秒注入一次故障，模式非常可重复。我们最初在集群中有一台坏机器，所以我们暂时将world size缩小到 25 台主机，直到机器被替换，然后我们在零停机时间内将作业扩展回来。

每 60 秒一次故障，我们期望能够在每次故障之间毫无问题地完成约 5 步。查看结果，我们看到有 6249 步和 5145 次成功提交。torchft 设计得尽可能安全，如果发生任何错误，它将在运行优化器之前通过"should_commit"丢弃该步骤。

对于整体步骤效率，我们有：

5145 次成功步骤 / 6249 次总步骤 = 82.3%

步骤时间约 11 秒，每 60 秒一次故障，我们应该能够完成每 6 步中的 5 步（83.3%），这与测量的性能几乎完全匹配。

我们平均每步有 29.6 个参与副本组，所以总训练效率为 81.2%。对于超过 1000 次故障来说还不错。

### 运行 2：每 15 秒注入故障，共 1015 次故障

我们想看看能推进多远，也让它更具挑战性。对于第二次运行，我们在 0-30 秒之间注入故障，平均每 15 秒一次故障。

与通常在 10 分钟到几小时范围内具有平均故障间隔时间的训练作业相比，这种故障率是极端的，但让我们验证无论错误何时发生我们都能恢复，并让我们运行大量测试周期以获得对我们实现的信心。

通过随机化故障间隔，我们导致故障在工作节点仍在初始化时而不是在稳定状态时发生，更可能遇到边缘情况。我们很高兴报告 torchft 表现如预期，没有不可恢复的错误。

![](https://files.mdnice.com/user/59/97c57cf6-0be6-419d-b78c-f6a70cc24ca9.png)

如您所见，这个作业的行为更加不稳定。与 60 秒故障率时非常接近 30 台机器不同，每 15 秒一次故障时，我们在每步从 1 台机器到 30 台机器都有。

平均而言，我们在任何给定步骤有 18.9（18.9/30 = 63%）个工作节点健康并参与，平均步骤时间为 15.46 秒。

在前 888 步中，268 步成功提交，给我们 30.2% 的步骤效率。

这给我们 13.4% 的训练效率，在任何正常训练作业中都是糟糕的，但令人惊讶的是，尽管每 15 秒崩溃一次，模型仍在收敛！仅从检查点加载模型通常就需要超过 1 分钟。

与我们的 60 秒 MTBF 运行相比，损失收敛较慢，但这是预期的，因为由于错误而丢弃了更多批次。

我们确实看到损失中有一些更大的峰值，这与只有 1 个参与者健康因此批量大小为 1/30 的时间相关。通过调整最小副本数量可以轻松避免这种情况。我们在这个测试中将其设置为 1。

### 运行 3：半同步训练

TorchFT 还支持半同步训练算法，包括 LocalSGD 和 DiLoCo，计划在未来添加更多。与 HSDP2 不同，这些算法不在每步同步。相反，它们在同步权重之前进行几步本地训练，通过平均参数或梯度。这种方法通过将通信成本降低到每 N 步一次（可配置的超参数）而不是每步一次来提高性能。我们在集群上的测试显示吞吐量有明显改善。当每 40 步同步一次时，我们最小化了通信开销，导致更高的整体吞吐量。下面是 DiLoCo 吞吐量（黄色）的比较，平均约 4000 tps，与常规 HSDP2（紫色）相比，平均约 1200 tps。

![](https://files.mdnice.com/user/59/68eafca4-2de2-4170-aaac-ea988af3b541.png)

自然地，同步之间的间隔越长，副本组内的模型就会越发散。这种发散可能会影响模型的收敛。然而，在我们的测试中，我们观察到模型仍然能够有效训练并达到收敛，尽管有这些较长的同步间隔。这种弹性在副本可能意外离开组的动态环境中是有益的。即使在这种情况下，模型也表现出继续训练而不会显著中断的能力。

![](https://files.mdnice.com/user/59/3681b0f2-c95e-4f97-a7a2-6f0d892bd152.png)

## 下一步

torchft 正在积极开发中，我们有很多计划的改进，包括更新的算法（如流式 DiLoCo）、使 PyTorch Distributed 对故障更加鲁棒（甚至在 infiniband/nvlink 上！）以及更高效。

如果您有兴趣使用 torchft，请查看 torchft README(https://github.com/pytorch/torchft/blob/main/README.md) 和 torchft 文档(https://docs.pytorch.org/torchft/)。我们也很乐意与您交流，请随时通过 GitHub、LinkedIn 或 Slack 直接联系我们。

