> 我的课程笔记，欢迎关注：https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode 

## 第15课: 优化PyTorch中的优化器

![](https://files.mdnice.com/user/59/ca409085-c88d-48ea-a881-c66cc2c29895.png)

![](https://files.mdnice.com/user/59/a0153888-5ab6-4228-a1bb-9d39afdfd1d8.png)


![](https://files.mdnice.com/user/59/e107bacd-0f4b-403d-a353-28607261dbcf.png)


![](https://files.mdnice.com/user/59/636c1455-195f-4c3b-b2c5-c107ae260f2e.png)

上面三张Slides讲述了运行时间（runtime）和内存使用（memory usage）之间的权衡关系。

第一张Slides：
- 介绍了运行时间和内存使用通常是相互矛盾的。
- 展示了两种运输车辆：一辆小卡车（代表低内存使用但速度慢）和一辆大卡车（代表高内存使用但速度快）。
- 提出了一个问题：如果要运送512辆车，应该选择哪种卡车？

第二张Slides：
- 在第一张图的基础上增加了一个新的限制条件：途中有一座低通桥。
- 这代表了在某些情况下，我们可能无法简单地选择高内存使用的方案（大卡车），因为存在硬件或系统限制。

第三张Slides：

- 明确表示"今天我们专注于速度！"
- 显示了小卡车被划掉，表明选择了大卡车（高内存使用但速度快的方案）。
- 同时提醒"这确实意味着内存会受到影响，免责声明"。

![](https://files.mdnice.com/user/59/7ff9ead4-6a06-4d42-b779-d4187487e8d7.png)


![](https://files.mdnice.com/user/59/5c601e52-2698-4e38-a5f2-e879355a88ec.png)

这张Slides展示了一个naive的优化器实现，核心要点是假设有M个参数，对于每个参数有N个操作，那么遍历所有参数并处理完共需要M * N个操作。 

![](https://files.mdnice.com/user/59/66ec108d-b015-4654-b1b6-7b1caed8e659.png)

这张Slides介绍了一种称为"horizontally fused optimizer"（水平融合优化器）的优化方法，可以把naive的优化器实现中的for循环fuse掉。

![](https://files.mdnice.com/user/59/c1d6e740-1a72-486c-bb3f-ab555266bae9.png)

这张Slides介绍了实际上我们可以把整个优化器的操作fuse成一个cuda kernel。

![](https://files.mdnice.com/user/59/fec4cdbc-990e-460b-ab67-59ba1a14fa1c.png)

这张Slides传达的核心信息是：在CUDA编程中，通过减少内核启动的次数可以提高程序的执行效率。这是因为每次启动CUDA内核都会有一定的开销，如果能够将多个操作合并到更少的内核中，就可以减少这些开销，从而提高整体性能。水平融合和垂直融合是实现这一目标的两种主要策略：水平融合合并了相似的并行操作；垂直融合则进一步合并了不同的计算步骤。

