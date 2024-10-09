import torch
import math
import transpose_cute as tc
import copy_cute as cc
import argparse

from torch.utils.benchmark import Timer
from torch.utils.benchmark import Measurement

# 解析命令行参数
parser = argparse.ArgumentParser(description='')
parser.add_argument('-M', help='Number of rows, M', default=32768)
parser.add_argument('-N', help='Number of columns, N', default=32768)
args = parser.parse_args()

# 设置设备为CUDA
cuda = torch.device('cuda')
# 生成一个随机矩阵A
A = torch.normal(0,1,size=(args.M, args.N)).to(device=cuda)
# 计算A的转置作为参考
AT_reference = torch.transpose(A, 0, 1)

# 定义基准测试函数
# stmt：这是一个字符串，表示要测量的代码片段。在这个例子中，stmt是你要执行的代码，
# 例如"cc.copy(A)"或"torch.transpose(A, 0, 1).contiguous()"。
# glob：这是一个字典，包含在执行stmt时需要的全局变量。在这个例子中，glob是一个包含变量cc和A的字典，例如{"cc": cc, "A": A}。
def benchmark(stmt, glob, desc): 
  # 创建计时器
  timer = Timer(
      stmt=stmt,
      globals=glob,
      num_threads=1,
  )
  
  # 运行基准测试并获取测量结果
  m: Measurement = timer.blocked_autorange(min_run_time=3)
  print(desc)
  # 打印平均时间和带宽
  print("Mean: {{:.{0}g}} ms ({{:.{0}g}} GB/s)".format(m.significant_figures).format(m.mean*pow(10,3),2*args.M*args.N*A.element_size()/m.mean*pow(10,-9)))
  # 打印四分位距
  print("IQR: {{:.{}g}} us".format(m.significant_figures).format(m.iqr*pow(10,6)))

# 定义验证函数
def validate(res, reference):
  # 验证结果是否与参考一致
  print("Validation: {}".format("success" if torch.all(torch.eq(reference,res)) else "failed"))

# 打印矩阵大小
print("Matrix size: {} x {}".format(args.M, args.N))
print()

# 基准测试：基线复制
benchmark("cc.copy(A)",{"cc": cc, "A": A},"Baseline copy:")
# 验证基线复制结果
validate(cc.copy(A), A)
print()

# 基准测试：PyTorch转置
benchmark("torch.transpose(A, 0, 1).contiguous()",{"A": A},"Torch transpose:")
print()

# 编译PyTorch转置函数
compiled_transpose = torch.compile(lambda A: torch.transpose(A, 0, 1).contiguous())
# 基准测试：编译后的PyTorch转置
benchmark("compiled_transpose(A)",{"compiled_transpose":compiled_transpose,"A": A},"Torch transpose (compiled):")
print()

# 遍历不同的转置版本进行基准测试
for ver in [tc.version.naive,tc.version.smem,tc.version.swizzle,tc.version.tma]:
  benchmark("tc.transpose(A, version=ver)",{"tc": tc, "A": A, "ver": ver},tc.get_version_info(ver))
  # 验证转置结果
  validate(tc.transpose(A, version=ver), AT_reference)
  print()
