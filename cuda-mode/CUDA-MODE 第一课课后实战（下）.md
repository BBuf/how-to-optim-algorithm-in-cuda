> 我的课程笔记，欢迎关注：https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode 

## CUDA-MODE 第一课课后实战（下）

### Nsight Compute Profile结果分析

继续对Nsight Compute的Profile结果进行分析，

#### Details部分

接在上篇的 Warp State Statistics 部分 之后。

##### Compute Workload Analysis 部分



![](https://files.mdnice.com/user/59/9d805106-2a9e-42c7-a2f0-1c8c6a0c3b7c.png)


Detailed analysis of the compute resources of the streaming multiprocessors (SM), including the achieved instructions per clock (IPC) and the utilization of each available pipeline. Pipelines with very high utilization might limit the overall performance.

> 对流式多处理器（SM）的计算资源进行详细分析，包括实际达到的每时钟周期指令数（IPC）以及每个可用流水线的利用率。利用率非常高的流水线可能会限制整体性能。

下面对这里涉及到的表格指标的知识库进行翻译(后面和第一个指标Executed Ipc Elapsed相同部分的翻译就省掉了)：

- Executed Ipc Elapsed[inst/cycle]

![](https://files.mdnice.com/user/59/fed54b44-b9aa-4158-98e2-77dc9154caee.png)

执行的warp指令数

sm_inst_executed.avg.per_cycle_elapsed
（此计数器指标表示所有子单元实例中每个执行周期的平均操作数）

sm: 流式多处理器以32个线程为一组（称为warps）来处理内核的执行。
Warps进一步分组为协作线程数组（CTA），在CUDA中称为块。
一个CTA的所有warps在同一个SM上执行。
CTA在其线程之间共享各种资源，例如共享内存。

warp: CTA内32个线程的组。
一个warp被分配到一个子分区，并从启动到完成都驻留在该子分区。

instructions: 一条汇编（SASS）指令。
每条执行的指令可能生成零个或多个请求。

- SM Busy

![](https://files.mdnice.com/user/59/15339364-d40b-4228-acbc-9499a34e6e53.png)

sm_instruction_throughput.avg.pct_of_peak_sustained_active
假设SMSP间理想负载平衡的SM核心指令吞吐量
（此吞吐量指标表示在所有子单元实例的活动周期内达到的峰值持续率的百分比）

SMSPs: 每个SM被划分为四个处理块，称为SM子分区。
SM子分区是SM上的主要处理元素。
一个子分区管理固定大小的warp池。

- Executed Ipc Active[inst/cycle]

![](https://files.mdnice.com/user/59/24485578-18bb-4b2b-8ba6-16c74ae917af.png)

sm_inst_executed.avg.per_cycle_active
执行的warp指令数
（此计数器指标表示所有子单元实例中每个活动周期的平均操作数）

- Issue Slots Busy

![](https://files.mdnice.com/user/59/9a850c51-c62c-48ce-a402-a92916ed92c9.png)

sm_inst_issued.avg.pct_of_peak_sustained_active
发出的warp指令数
（此计数器指标表示在所有子单元实例的活动周期内达到的峰值持续率的平均百分比）

- Issued Ipc Active[inst/cycle]

![](https://files.mdnice.com/user/59/57fddbed-cb74-48b8-a232-1920df99892e.png)

sm_inst_issued.avg.per_cycle_active
发出的warp指令数
（此计数器指标表示所有子单元实例中每个活动周期的平均操作数）

接着对后半部分的图标进行分析：

![](https://files.mdnice.com/user/59/aba3e34f-9885-481f-9b27-db22f709dd24.png)

这里提示低利用率（Low Utilization）并且预估加速比可以到91.84%。所有计算Pipline都未被充分利用。这可能是因为内核太小，或者每个调度器没有发出足够的warps。

> 由于这是个访存密集型算子，计算利用率低其实是正常的，这里的预估加速是一个粗略的计算，不能完全参考。Compute Workload Analysis部分需要结合Memory Workload Analysis 部分来看。

点开Low Utilization最右边那个按钮之后可以看到关键的性能指标以及指导建议。

最下面的Pipline利用率图表分为两部分，分别显示了不同类型pipline的利用率。

此外，图中还建议查看"Launch Statistics"和"Scheduler Statistics"部分以获取更多详细信息，这可能有助于理解为什么Pipline利用率如此之低。


##### Launch Statistics部分


![](https://files.mdnice.com/user/59/b62101c1-7af2-408d-b9c9-f0c6a73513b5.png)


这里的具体指标的知识库不再赘述，直接看一下Tail Effect部分，这里说的是 Wave 是指可以在目标GPU上并行执行的最大块数。在这个内核中，2个完整Wave和一个部分Wave（包含433个线程块）被执行。假设所有线程块的执行时间均匀，部分Wave可能占总内核运行时间的33.3%，而完整占用度为24.0%。

图中建议尝试启动没有部分 Wave 的网格。减少尾效应也可以减少执行完整网格所需的 Wave 数量。建议查看硬件模型描述以获取更多关于启动配置的详细信息。



##### Scheduler Statistics部分

![](https://files.mdnice.com/user/59/8184a152-b3c6-489e-844b-5a7088b62fca.png)

Summary of the activity of the schedulers issuing instructions. Each scheduler maintains a pool of warps that it can issue instructions for. The upper bound of warps in the pool (Theoretical Warps) is limited by the launch configuration. On every cycle each scheduler checks the state of the allocated warps in the pool (Active Warps). Active warps that are not stalled (Eligible Warps) are ready to issue their next instruction. From the set of eligible warps the scheduler selects a single warp from which to issue one or more instructions (Issued Warp). On cycles with no eligible warps, the issue slot is skipped and no instruction is issued. Having many skipped issue slots indicates poor latency hiding.

> 调度器发出指令活动的总结。每个调度器维护一个可以为其发出指令的warp池。池中warp的上限（理论warp数）受启动配置的限制。在每个周期，每个调度器检查池中分配的warp的状态（活跃warps）。未被停滞的活跃warps（Eligible warps）准备好发出它们的下一条指令。从Eligible warps集合中，调度器选择一个warp来发出一条或多条指令（已发出的warp）。在没有Eligible warps的周期中，发出槽被跳过，不发出任何指令。有许多被跳过的发出槽表明延迟隐藏效果不佳。

下面是对指标的知识库进行翻译

- Active Warps Per Scheduler[warp]



##### Occupancy 部分

#### Source部分








- 推荐阅读：https://www.youtube.com/watch?v=04dJ-aePYpE
