import torch
import math

import cutlass_gemm

# Create two normalized matrices and allocate to GPU
M = K = N = 4096
cuda = torch.device('cuda')
A = torch.normal(0,1,size=(M, K)).to(device=cuda).to(dtype=torch.float16)/math.sqrt(K)
B = torch.normal(0,1,size=(K, N)).to(device=cuda).to(dtype=torch.float16)/math.sqrt(K)

C1 = cutlass_gemm.mm(A,B)
print("cutlass_gemm.mm result:")
print(C1)
print()

C2 = torch.mm(A,B)
print("torch.mm result:")
print(C2)
print()
print("max deviation: {:.10f}".format(torch.max(torch.abs(C2-C1))))
