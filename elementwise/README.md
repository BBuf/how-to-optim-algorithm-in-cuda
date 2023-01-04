# elementwise 优化

这里将 OneFlow 的 elementwise 模板单独放到 elementwise.cu 文件中，并以逐点乘法为例对比普通的 elementwise 做法进行 profile 。

