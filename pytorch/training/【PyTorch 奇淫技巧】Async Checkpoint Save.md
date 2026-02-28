## PyTorch Async Checkpoint Save

- PyTorch博客资料：https://pytorch.org/blog/reducing-checkpointing-times/
- PyTorch实现和使用Demo：https://github.com/pytorch/pytorch/blob/main/torch/distributed/checkpoint/state_dict_saver.py

### 功能介绍

在PyTorch 2.4之后，我们可以尝试使用PyTorch开发的异步Checkpoint保存功能，这个功能是和IBM联合开发的，在7B的大模型训练中，Checkpoint保存的时间从平均 148.8 秒缩短至 6.3 秒，快了 23.62 倍。这可以转化为以下两种好处：

- 在继续鲁棒的保存Checkpoint的同时，在每给定的 24 小时内实现更多净训练进度；
- 可以更频繁地进行Checkpoint保存以缩短训练恢复窗口或者时间。

![](https://files.mdnice.com/user/59/f48b4bb9-321c-4b80-a70f-15e8de25c746.png)

从结果图来看，无论是单机的FSDP还是多机的HSDP，Async Checkpoint Save都展现出了很大的速度优势，对于参数量更大的模型预计收益会更大。目前在TorchTian（https://github.com/pytorch/torchtitan）中已经集成了这个新的功能，相信其他的主流训练框架都会很快跟进此feature。

### 博客内容

#### 背景

模型Checkpoint是大模型训练的重要组成部分，但Checkpoint是一个昂贵的过程，因为每个Checkpoint过程都需要阻止训练进度以保存最新的模型权重。但是，不进行Checkpoint或降低Checkpoint频率会导致训练进度损失很大。例如，死锁、straggler（落后者）和 GPU 错误等故障导致需要重新启动训练过程。为了从故障中重启，所有（训练）工作者必须停止其训练过程并从上次保存的Checkpoint重新启动。

因此，对故障的鲁棒性与训练进度之间很难做到权衡，但现在有了异步Checkpoint，PyTorch 分布式训练能够显著缓解这种张力，并以最小的影响整体训练时间的方式实现频繁Checkpoint。

大约一年前(https://pytorch.org/blog/performant-distributed-checkpointing/)，我们展示了分布式Checkpoint如何大幅加速Checkpoint时间，从最初的 `torch.save()` 功能开始。正如 IBM 研究团队指出的那样，`torch.save` 可能需要长达 30 分钟才能检查一个 11B 模型（PyTorch 1.13）。

随着分布式Checkpoint的进步，对于高达 30B 的模型大小，Checkpoint可以在 4 分钟内完成。使用异步Checkpoint，Checkpoint导致的训练时间损失现在降至 30 秒以下，通常仅需 6 秒。

需要明确的是，异步Checkpoint不会压缩实际的序列化Checkpoint时间，如之前的更新所展示的那样。相反，**它将最终的Checkpoint过程移出关键路径（到 CPU 线程），以允许 GPU 训练在单独的线程下完成Checkpoint的同时继续进行**。

![](https://files.mdnice.com/user/59/9cb4caa2-af87-4b52-9c8b-e364fe621903.png)

如上图所示，异步Checkpoint比一年前的改进进一步提高了 10 倍到 23 倍。

#### Async Checkpoint Save如何工作

异步Checkpoint将Checkpoint过程模块化分为两个部分，而不是一个单一的整体过程。第一阶段将每个 GPU/rank 的数据从 GPU 复制到 CPU。这是用户可见的停机时间，对于 7B-13B 的模型大小可能需要 6 到 14 秒。第二阶段异步地将数据从 CPU 内存复制到磁盘以持久保存Checkpoint。

一旦数据在第一阶段复制到 CPU，GPU 就可以立即恢复训练。因此，使用异步Checkpoint，Checkpoint的停机时间仅仅是将最新的模型状态复制到 CPU 所需的时间。在训练恢复的同时，非阻塞 CPU 线程使用内存中新到达的数据完成完整的Checkpoint/序列化过程到磁盘（即持久保存）。

![](https://files.mdnice.com/user/59/bb672b2e-8ddb-4619-92c2-cfa4d59e1242.png)

注意，PyTorch 的分布式Checkpoint依赖于集合通信调用来获取必要的每个等级元数据以优化保存，以及最终的同步，该同步将Checkpoint标记为已完成并使操作成为原子操作。如果Checkpoint线程使用与训练相同的进程组，这可能会干扰分布式训练（因为分布式训练也依赖于类似的调用来跨多个 GPU 同步训练）。具体来说，调用之间的竞争条件可能会导致训练和异步Checkpoint保存线程同时等待集合通信调用，从而导致真正的集合通信卡死。我们**通过为异步Checkpoint初始化一个单独的进程组来避免这种情况**。这将Checkpoint集合通信分离到其自己的逻辑进程组中，从而确保它不会干扰主训练线程中的集合通信调用。

### 如何使用PyTorch Async Checkpoint Save

这里是最小的使用PyTorch Async Checkpoint Save的demo：

![](https://files.mdnice.com/user/59/e5f6d4be-5582-455a-8f31-3698f91758ef.png)

需要注意的是第12行，为异步的Checkpoint集合通信操作建立了一个新的group，然后调用`dcp.save`的时候我们需要传入这个group。


https://github.com/pytorch/torchtitan 里面也已经使用上了这个功能，可以用于预训练自己的 Llama2 或 Lllama3 模型。在配置文件里面就可以选择使用Async Checkpoint Save。如下图所示：

![](https://files.mdnice.com/user/59/7ae0fc1e-f9aa-4745-b046-623c7c8d02dc.png)

### 代码流程粗略浏览

代码实现在 https://github.com/pytorch/pytorch/blob/main/torch/distributed/checkpoint/state_dict_saver.py 这个文件中。核心部分为以下2个函数，这里简单展示下流程：

```python
# 创建 state_dict 的浅拷贝，对于每个 Stateful 对象调用其 state_dict() 方法。
def _stateful_to_state_dict(state_dict: STATE_DICT_TYPE) -> STATE_DICT_TYPE:
    """Creates a shallow copy of `state_dict` where `state_dict` is called for each Stateful object."""
    stateful_state_dict = {}
    for key, elem in state_dict.items():
        stateful_state_dict[key] = (
            elem.state_dict() if isinstance(elem, Stateful) else elem
        )
    return stateful_state_dict

@_dcp_method_logger(log_exceptions=True)
def async_save(
    state_dict: STATE_DICT_TYPE,
    *,
    checkpoint_id: Union[str, os.PathLike, None] = None,
    storage_writer: Optional[StorageWriter] = None,
    planner: Optional[SavePlanner] = None,
    process_group: Optional[dist.ProcessGroup] = None,
) -> Future:
    torch._C._log_api_usage_once("torch.distributed.checkpoint.async_save")

    # 检查分布式环境设置
    if dist.is_available() and dist.is_initialized():
        pg = process_group or _get_default_group()
        assert (
            torch.device("cpu") in pg._device_types  # type: ignore[attr-defined]
        ), "A CPU backend must be enabled for async save; try initializing process group with 'cpu:gloo,cuda:nccl'"

    # 设置存储写入器
    storage_writer = cast(
        StorageWriter, _storage_setup(storage_writer, checkpoint_id, reader=False)
    )

    # 处理状态字典（调用 _stateful_to_state_dict）
    state_dict = _stateful_to_state_dict(state_dict)
    # 如果存储写入器支持异步暂存，则使用它；否则将状态字典卸载到 CPU
    if isinstance(storage_writer, AsyncStager):
        staged_state_dict = storage_writer.stage(state_dict)
    else:  # provides bwc for storage_writers not implementing AsyncStager
        staged_state_dict = _offload_state_dict_to_cpu(state_dict, type_check=False)

    # 创建线程池执行器，提交保存任务。这里是一个线程
    executor = ThreadPoolExecutor(max_workers=1)
    f: Future = executor.submit(
        save,
        staged_state_dict,
        checkpoint_id=checkpoint_id,
        storage_writer=storage_writer,
        planner=planner,
        process_group=process_group,
    )
    # 设置任务完成后的回调函数（关闭执行器）
    f.add_done_callback(lambda f: executor.shutdown(wait=False))

    # 如果需要，同步暂存操作
    if (
        isinstance(storage_writer, AsyncStager)
        and storage_writer.should_synchronize_after_execute
    ):
        storage_writer.synchronize_staging()

    # 返回 Future 对象
    return f
```

### 将来的改进

PyTorch Blog中提到Checkpoint在过去的一年中取得了巨大进步。从近半个小时的Checkpoint变为使用分布式Checkpoint不到 5 分钟，现在又变为使用异步Checkpoint不到 30 秒。**最后一个前沿是零开销Checkpoint，即使是小于 30 秒的时间也可以通过在反向传递期间流式传输更新的权重来消除，这样Checkpoint数据在异步Checkpoint开始时就已经在 CPU 上了**。这将有效地将大型模型训练转移到Checkpoint没有中断或停机时间的程度，从而既提高了鲁棒性（因为可以更频繁地进行Checkpoint），又因为没有Checkpoint的停机时间而加快了训练进度。








