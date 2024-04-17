## 介绍

这里的代码来自：https://github.com/Liu-xiandong/How_to_optimize_in_GPU/blob/master/sgemv/

讲解文章来自：https://zhuanlan.zhihu.com/p/494144694

思路就是用一个warp处理一行或者2行的gemv操作。

