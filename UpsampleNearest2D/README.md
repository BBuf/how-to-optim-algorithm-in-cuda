## UpsampleNearest2D

- 硬件环境 A100 PCIE 40G

upsample_nearest_2d.cu 展示了 oneflow 对 upsample_nearest2d 的前后向的优化 kernel 的用法，我这里直接使用 oneflow 的脚本对这两个 kernel 进行进行 profile ：

```python
import oneflow as flow

x = flow.randn(16, 32, 80, 80, device="cuda", dtype=flow.float32).requires_grad_()

m = flow.nn.Upsample(scale_factor=2.0, mode="nearest")

y = m(x)
print(y.device)
y.sum().backward()
```

下面展示了在 A100 上调优前后的带宽占用和计算时间比较：

|框架|数据类型|Op类型|带宽利用率|耗时|
|--|--|--|--|--|
| PyTorch | Float32 | UpsampleNearest2D forward | 28.30% | 111.42us |
| PyTorch | Float32 | UpsampleNearest2D backward | 60.16% | 65.12us |
| OneFlow | Float32 |UpsampleNearest2D forward | 52.18% | 61.44us |
| OneFlow | Float32 |UpsampleNearest2D backward | 77.66% | 50.56us |
| PyTorch | Float16 | UpsampleNearest2D forward | 16.99% | 100.38us |
| PyTorch | Float16 | UpsampleNearest2D backward | 31.56% | 57.38us |
| OneFlow | Float16 |UpsampleNearest2D forward | 43.26% | 35.36us |
| OneFlow | Float16 |UpsampleNearest2D backward | 44.82% | 40.26us |

可以看到基于 oneflow upsample_nearest2d 的前后向的优化 kernel 可以获得更好的带宽利用率和性能。

