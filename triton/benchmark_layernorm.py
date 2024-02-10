import numbers
import torch
import triton
from torch.nn.parameter import Parameter
from torch.nn import init
from flash_attn.ops.triton.layer_norm import layer_norm_fn, layer_norm_ref

from apex.normalization.fused_layer_norm import FusedLayerNormAffineMixedDtypesFunction


class LayerNorm(torch.nn.Module):

  def __init__(self, 
               normalized_shape, 
               eps=1e-5,
               ):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.weight = Parameter(torch.Tensor(*normalized_shape))
        self.bias = Parameter(torch.Tensor(*normalized_shape))
        self.reset_parameters()


  def reset_parameters(self):
    init.ones_(self.weight)
    init.zeros_(self.bias)

  def forward(self, input, memory_efficient):
    return FusedLayerNormAffineMixedDtypesFunction.apply(input, self.weight, self.bias, self.normalized_shape, self.eps, memory_efficient)


# Apex LayerNorm
def apex_layer_norm(x, memory_efficient):
    normalized_shape = x.size()[-1]
    layernorm = LayerNorm(normalized_shape=normalized_shape, eps=1e-5).to("cuda")
    layernorm.to(torch.bfloat16)
    y = layernorm(x, memory_efficient)
    y.sum().backward()
    return y, layernorm.weight, layernorm.bias

# Naive LayerNorm (Flash Attention)
def naive_layer_norm(x, weight, bias):
    y = layer_norm_ref(x, weight, bias)
    y.sum().backward()
    return y

# Fuse LayerNorm (Flash Attention)
def fuse_layer_norm(x, weight, bias):
    y = layer_norm_fn(x, weight, bias)
    y.sum().backward()
    return y

# 基准测试函数
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['hidden_size'],  # 作为x轴的参数名
        x_vals=[192, 512, 1024, 2048, 2560, 3000, 4096, 5120, 6144],  # hidden_size的不同值
        line_arg='provider',  # 对应图中不同线条的参数名
        line_vals=['apex no efficient', 'apex efficient', 'naive', 'triton fuse'],  # line_arg的可能值
        line_names=["Apex No-Efficient LayerNorm", "Apex Efficient LayerNorm", "PyTorch Naive LayerNorm", "Triton Fuse LayerNorm"],  # 线条的标签名称
        styles=[('blue', '-'), ('black', '-'), ('green', '-'), ('red', '-')],  # 线条样式
        ylabel="Time (ms)",  # y轴的标签名称
        plot_name="layernorm-performance",  # 图表的名称，同时用作保存图表的文件名
        args={'batch_size': 128, 'seq_length': 1024},  # 额外的函数参数
    )
)
def benchmark(batch_size, seq_length, hidden_size, provider):
    x = torch.randn(batch_size, seq_length, hidden_size, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    quantiles = [0.5, 0.2, 0.8]
    _, weight, bias = apex_layer_norm(x, False)
    torch.cuda.synchronize()
    if provider == 'apex no efficient':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: apex_layer_norm(x, False)[0], quantiles=quantiles)
    if provider == 'apex efficient':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: apex_layer_norm(x, True)[0], quantiles=quantiles)
    elif provider == 'naive':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_layer_norm(x, weight, bias), quantiles=quantiles)
    elif provider == 'triton fuse':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fuse_layer_norm(x, weight, bias), quantiles=quantiles)
    return ms, min_ms, max_ms

benchmark.run(show_plots=True, print_data=True, save_path='./benchmark_ops/layernorm/')
