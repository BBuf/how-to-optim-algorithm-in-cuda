# elementwise 优化

这里将 OneFlow 的 elementwise 模板单独放到 elementwise.cu 文件中，并以逐点乘法为例对比普通的 elementwise 做法进行 profile 。

|优化手段|数据类型|耗时(us)|带宽利用率|
|--|--|--|--|
|naive elementwise|float|298.46us|85.88%|
|oneflow elementwise|float|284us|89.42%|
|naive elementwise|half|237.28us|52.55%|
|oneflow elementwise|half|140.74us|87.31%|

可以看到无论是性能还是带宽，使用 oneflow 的 elementwise 模板相比于原始实现都有较大提升。

