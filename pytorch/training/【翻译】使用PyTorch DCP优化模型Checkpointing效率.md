> 来源：https://discuss.pytorch.org/t/distributed-w-torchtitan-optimizing-checkpointing-efficiency-with-pytorch-dcp/211250 。这篇文章主要介绍了 PyTorch 分布式检查点(DCP)在 TorchTitan 中的应用和优化。文章详细讨论了三种检查点策略:传统同步检查点、异步检查点和零开销检查点。通过异步检查点，将存储操作与训练迭代并行执行，使检查点开销减少了19倍；而零开销检查点通过将 GPU 到 CPU 的数据复制与前向和反向传播重叠，并利用独立的 CUDA 流和进程，进一步将开销降低了5倍(总开销降至1秒以下)。虽然零开销检查点在实现上更复杂且需要更多资源，但它代表了提升训练效率和 GPU 利用率的重要方向。

> 这是[【PyTorch 奇淫技巧】Async Checkpoint Save ](https://mp.weixin.qq.com/s/DcNjBi_rJKvrU9Ssp8Mo0Q) 在训练框架TorchTitan中的实践。

## 概要

- 我们将 PyTorch 分布式检查点(DCP)集成到 TorchTitan 中,实现了高效的分布式检查点功能。
- 我们在 PyTorch 的 DCP 中实现了异步检查点,使存储操作可以与后续训练迭代重叠,从而优化了处理效率。
    - 与同步检查点相比,检查点开销减少了19倍。
- 我们在 TorchTitan 中使用 DCP 开发了零开销检查点的原型,进一步将 GPU 到 CPU 的复制与后续的前向和反向操作重叠。
    - 与异步检查点相比,这种方法将检查点开销进一步减少了5倍,使总检查点开销降至1秒以下。

## 使用 TorchTitan 进行分布式训练

GitHub 仓库 TorchTitan(https://github.com/pytorch/torchtitan) 是一个使用原生 PyTorch 进行大规模 LLM 训练的概念验证项目,它设计得易于理解、使用和扩展,可用于不同的训练目的,通过模块化组件支持多维度并行。在这一系列主题中,我们将介绍 Torchtitan 中启用的最新 PyTorch 分布式训练特性。

- Topic 1: [【翻译】在FSDP2中开启Float8 All-Gather](https://mp.weixin.qq.com/s/44zFNWr5aVtA3zPtegY9dg)
- Topic 2: 【翻译】一文了解PyTorch中的Async Tensor Parallelism(https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487)
- → Topic 3: Optimizing Checkpointing Efficiency with PyTorch DCP

## DCP 简介

检查点在训练大型模型中至关重要,主要有两个用途:可以用于推理和评估等应用,也可以用于故障恢复。有效的检查点应该确保检查点易于重用,且不会成为性能瓶颈。本节将探讨 PyTorch 分布式检查点(DCP)如何实现这些目标。

有两种常见的分布式检查点方法。第一种是将所有模型权重和优化器状态收集到单个 rank(通常是 rank 0),然后由该 rank 保存完整的检查点。这种方法虽然简单直观,但速度慢且存储 I/O 利用率低,因为它只使用 rank 0 的存储 I/O,并且涉及可能导致性能问题的额外通信。第二种方法允许每个 rank 独立保存其本地状态,通过利用所有可用的 I/O 资源来加快进程。但是,这通常需要繁琐且不可扩展的后处理,并且需要专有的并行信息来适配分片检查点以供不同用途。

为了解决这些挑战,我们在 PyTorch 中引入了 DCP。DCP 使每个 rank 能够在不需要张量通信的情况下本地保存其张量,同时保留稍后重新组装检查点以用于各种并行方式或完整检查点所需的信息。DCP 的设计与并行方式无关,仅依赖于 PyTorch 原生分布式并行(如 DDP、FSDP2、TP、PP)产生的 DTensors。它分析并将 DTensors(或 torch.Tensors)转换为内部格式,而无需了解底层并行方式。

加载检查点时,DCP 使用当前状态字典来确定张量分片,并即时获取数据。用户也可以选择离线预处理检查点以最小化加载时间。DCP 通过将内部格式转换为最终结果来简化后处理。

图中说明了保存流程。在这个例子中,参数 P2 在 rank0 和 rank1 之间分片,而参数 P1 保持在 rank0 上未分片。在每个 rank 上保存 `state_dict` 时,张量数据本身不会发生通信。但是,与元数据相关的通信会发生,然后保存在元数据文件中。这个元数据文件详细说明了各个文件中每个参数的偏移量和长度。请注意,该图仅用于说明目的并简化了某些方面;实际实现细节可能有所不同。

![](https://files.mdnice.com/user/59/b591a1d9-f062-4530-9d6e-169f6577342d.png)

## 将 DCP 集成到 TorchTitan 中

要在 TorchTitan 中使用 DCP 保存检查点,可以使用以下代码片段:

```python
import torch.distributed.checkpoint as dcp
def save_checkpoint(self, state_dict: Dict[str, Any], path: Union[str, os.PathLike]):
    dcp.save(state_dict, path)
```

path 参数可以是指向本地文件系统的普通路径,也可以是指向 fsspec 支持的存储的路径。对于那些希望使用自己专有存储解决方案的用户,DCP 还允许自定义存储后端。`state_dict` 参数是一个包含要保存状态的字典。DCP 会遍历 `state_dict`,检查每个值是否有 `state_dict()` 方法。如果有,DCP 会在对象上调用此方法并保存返回值。否则,它会直接保存这些值。要同时保存模型和优化器,以下 `state_dict` 就足够了:

```python
model = MyModel()
optim = MyOptimizer(model)
state_dict = {"model": model, "optimizer": optim}
```

然而,这种 `state_dict` 内容虽然适用于使用数据并行和张量并行的模型,但不适用于流水线并行。此外,它也不能用于在不同数量的 GPU 或不同的并行方式之间重新分片优化器。这两个限制都源于 `torch.optim.Optimizer.state_dict()` 返回的字典使用参数 ID 来表示参数/状态,而不是完全限定名称(FQN)。与 `model.state_dict()` 不同,后者返回像 `layer1.weight` 这样的键(无论 GPU 分布或模型并行化如何,这都是唯一的 FQN),`optim.state_dict()` 使用数字 ID 来表示 `layer1.weight`,这个 ID 反映了参数传入优化器的顺序。这个参数 ID 不是唯一的,可能会导致冲突,特别是在流水线并行中,像 `layer1.weight` 和 `layer2.weight` 这样的参数可能会在不同的 GPU 上具有相同的参数 ID。

为了解决这个问题,我们在 PyTorch 中实现了分布式 `state_dict` API,它将模型和优化器的 `state_dict` 都转换为分布式检查点友好的格式。在 TorchTitan 中,我们使用以下 `OptimizerWrapper` 来封装优化器(我们将省略对 `ModelWrapper` 的讨论,因为它的基本概念与 `OptimizerWrapper` 相同):

```python
class OptimizerWrapper(Stateful):
    def __init__(
        self,
        model: Union[nn.Module, List[nn.Module]],
        optim: Union[torch.optim.Optimizer, List[torch.optim.Optimizer]],
    ) -> None:
        self.model = [model] if isinstance(model, nn.Module) else model
        self.optim = [optim] if isinstance(optim, torch.optim.Optimizer) else optim

    def state_dict(self) -> None:
        func = functools.partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        return {k: v for sd in map(func, self.model, self.optim) for k, v in sd.items()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        func = functools.partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        list(map(func, self.model, self.optim))
```

TorchTitan 使用 `model_wrapper` 和 `optim_wrapper` 而不是直接将模型和优化器传递给 `dcp.save()`。值得注意的是，`OptimizerWrapper`（以及类似的 `ModelWrapper`）可以接受模型和优化器的列表，以适应一些流水线并行算法，这些算法在每个 rank 上管理多个模型和优化器块。分布式的 `state_dict` 可以将多个 `state_dict` 条目压平为一个。

本节概述了将 DCP 集成到 TorchTitan 中的基本概念。有关更详细的信息，请参阅代码(https://github.com/pytorch/torchtitan/blob/main/torchtitan/checkpoint.py)。

## 异步检查点

虽然使用 DCP 避免了聚合张量的需求,但与训练步骤相比,检查点的开销仍然很大。在检查点过程中,训练器必须等待该过程完成,这实际上浪费了 GPU 资源。

检查点面临两个主要瓶颈:将张量从 GPU 复制到 CPU 内存(称为"暂存")以及将张量从 CPU 内存传输到持久存储,如下图所示。该图沿时间轴(x轴)描绘了三个不同的任务(训练、暂存和持久化步骤),要求训练器暂停训练并切换到执行暂存,然后执行持久化步骤。对于现代模型,暂存开销通常持续几秒钟,而持久化步骤可能需要几十到几百秒,具体取决于存储系统。

![](https://files.mdnice.com/user/59/95462b98-4d32-4be0-90b6-7923e9cad3e9.png)

减少检查点频率是一种常见的降低开销的方法。例如,如果检查点开销是50秒,而目标是将GPU时间浪费限制在不超过1%,那么最优解决方案是每5000秒保存一次检查点。虽然这种频率在较小规模上可能是可以接受的,但在跨越数百或数千个GPU节点进行训练时就会出现问题。在如此大的规模下,假设5000秒内没有节点故障是不现实的。由于训练的SPMD特性,在此期间内单个节点的故障都会要求所有节点从最后一个检查点重新启动,这会显著降低训练效率。

为了解决这种低效问题,我们在DCP中实现了异步检查点。异步检查点的基本原理是持久化步骤(不涉及GPU)可以在单独的线程上与训练步骤并发运行。使用异步检查点时,该过程从主训练暂停以将张量从GPU复制到CPU内存开始。之后,主训练线程恢复训练任务,而持久化步骤则委托给另一个线程。下图说明了异步检查点的概念。主线程不再处理持久化步骤,而是简单地启动另一个专门用于此任务的线程,并立即返回训练。

![](https://files.mdnice.com/user/59/64d9a1af-bce7-478a-a61a-d903bab248a7.png)

下图展示了实验结果。我们使用配备了64个H100 GPU的8个节点,通过TorchTitan FSDP2训练了Llama 3 8B模型。检查点频率设置为每100次迭代一次。从图中可以看出,在不进行检查点的情况下训练100次迭代大约需要270秒。使用同步检查点时,检查点开销接近50秒。显然,这个开销太大,无法维持每100次迭代或每5分钟一次的检查点频率。

![](https://files.mdnice.com/user/59/3a8b8512-be47-4853-9f69-34681aabe5ba.png)

使用异步检查点时,检查点开销减少到不到0.5秒。理想情况下,这应该代表异步检查点的总开销;然而,由于Python全局解释器锁(GIL)的存在,持久化线程偶尔会阻碍主训练线程,在随后的10次训练迭代中增加约2.2秒的延迟。尽管存在GIL问题,结果仍然显示相比同步检查点有显著改进,开销减少高达19倍。对于这个实验,将检查点开销限制在1%以内使我们能够实际地将检查点频率提高到每5分钟或每100次迭代一次。

除了在TorchTitan上进行的实验外,我们还与IBM合作展示了异步检查点的性能优势(https://pytorch.org/blog/reducing-checkpointing-times/)。

## 零开销检查点

异步检查点显著减少了GPU资源的浪费,但考虑到GPU不断上升的成本和功耗,即使1%的损失可能也被认为太高。我们能否改进异步检查点?还有哪些因素在减缓检查点的速度?

一个剩余的瓶颈是暂存过程——将张量从GPU复制到CPU内存。乍看之下,似乎不可能在不冒险导致下一次训练迭代的状态部分更新而产生错误检查点的情况下,将暂存与训练并行化。然而,仔细检查训练步骤(包括前向、反向和优化阶段)后发现,只有优化步骤会修改状态。因此,如果我们能够将暂存与前向和反向步骤重叠,我们可能几乎可以消除检查点开销。

实际上,通过将暂存过程放在单独的CUDA流中并将所有复制操作设置为`non_blocking=True`,仅在下一个优化步骤之前同步流,就可以实现这种重叠。这个策略有效地隐藏了暂存过程。我们已经在PyTorch私有API `_copy_state_dict` 中实现了这一点,并在TorchTitan中使用它来原型化我们称之为零开销检查点(或接近零开销)的功能。

然而,如果暂存时间太长,超过了前向和反向步骤的组合持续时间,它仍然可能变得可见。为了提高暂存性能,我们利用CUDA分配固定内存的选项来加速复制过程。

另一个挑战是防止暂存线程干扰主线程的执行。在我们的原型中,我们通过为持久化步骤创建一个单独的进程来解决这个问题。尽管在进程之间传输张量可能很耗时,但PyTorch通过其将CPU张量标记为可在进程间共享的能力来促进这一点。通过结合固定内存和共享内存特性,我们开发了另一个PyTorch私有API `_create_cpu_state_dict`,它为零开销检查点中的暂存创建CPU `state_dict`。

下图说明了零开销检查点流程。在暂存CUDA流上下文中启动暂存后,主线程可以立即恢复第N+1次迭代的训练。暂存CUDA流与训练同时执行暂存过程。在进入优化步骤之前,主线程必须验证暂存的状态;如果暂存已经完成,这个检查会产生最小的开销。随后,主线程可以在单独的进程中启动持久化步骤,如前所述。然后主线程返回训练任务。

![](https://files.mdnice.com/user/59/38c8c231-b412-4b14-b03f-b19741a16dd2.png)

下图展示了使用与上一节相同的模型和硬件配置进行的实验结果。结果表明暂存开销仅为0.06秒,后续训练步骤的速度减慢不到0.4秒。这使得总开销降至0.5秒以下——比异步检查点快6倍。但仍有改进空间。额外的0.35秒主要是由于主线程监控暂存CUDA流状态并将state_dict传输到持久化进程所致。未来的工作可以探索将这些操作卸载到另一个线程以进一步最小化开销。

图6: 异步检查点 vs 零开销检查点

![](https://files.mdnice.com/user/59/af3acfec-b473-45f5-8377-3bf5f589f840.png)

与异步检查点相比,零开销检查点更加复杂,需要额外的CPU内存(固定内存不可分页)和多进程处理,这些都更难管理。因此,如果CPU内存受限或者用户更倾向于使用更简单的检查点过程,异步检查点可能是更合适的选择。尽管存在这些挑战,零开销检查点代表了提高训练效率和GPU利用率的一个有前途的方向。



