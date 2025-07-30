简单记录一下最近SGLang社区在sgl-kernel中解决的一个错误使用`c10::Float8_e4m3fn`数据类型导致的一系列fp8 quant kernel性能问题。

相关的PR见：https://github.com/sgl-project/sglang/pull/8499 & https://github.com/sgl-project/sglang/pull/8290 & https://github.com/sgl-project/sglang/pull/8449

这个问题是由社区的 @strgrb 发现的，他使用cute写了一个 `per_token_group_quant_8bit` 的 kernel (PR 8290)，这个 kernel 的功能和 https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/gemm/per_token_group_quant_8bit.cu 中的 kernel 一致，但是使用 cute 实现之后发现它的性能比 sgl-kernel 中的 kernel 要快10%-20%。后面经过我们的讨论分析发现，问题出在 kernel 完成量化之后做 float->fp8 转换时因为没有使用 `__nv_fp8_e4m3` dtype而是使用 `c10::Float8_e4m3fn` 导致 kernel 没有真正用上硬件加速的`F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C` convert指令，而是通过软件函数调用实现，包含大量条件分支和位操作最终导致了性能下降。

我们最开始实现quant相关的kernel时没有考虑到这一点，现在已经全部修复。最开始实现这几个kernel的时候Dispatch dtype我是参考了vLLM的那部分代码，直接用了`c10::Float8_e4m3fn`类型，没有考虑到这个类型不是CUDA内置的`__nv_fp8_e4m3`类型，我们这两天修复完之后看到vLLM也把我们的这个修复apply进去了。

![](https://files.mdnice.com/user/59/3176aa8f-a5fd-4389-a873-5483ccec118f.png)

![](https://files.mdnice.com/user/59/0082976b-84b1-41c7-b4bf-2d87594f0e8b.png)

世界是一个巨大的草台班子。下面是ncu在h20上修改后的`per_token_group_quant_8bit`相比于修改前的性能对比，可以看到带宽提升明显。

![](https://files.mdnice.com/user/59/da74aec5-b93e-44e3-9be4-94e0387de578.png)

接下来是使用sgl-kernel benchmark工具测试的`per_tensor_quant`和`per_token_quant`的性能对比。

## Benchmark In B200

10%-20% 性能提升.

### main branch

```shell

➜  sgl-kernel git:(main) ✗ python3 /home/yineng/bbuf/sglang/sgl-kernel/benchmark/bench_per_tensor_quant_fp8.py
INFO 07-28 20:40:56 [__init__.py:235] Automatically detected platform cuda.
✅ All implementations match
per-tensor-quant-fp8-performance:
    batch_size  seq_len         VLLM   SGL Kernel
0         16.0     64.0    26.272001    25.792001
1         16.0    128.0    36.704000    32.671999
2         16.0    256.0    51.328000    45.024000
3         16.0    512.0    92.160001    77.760004
4         16.0   1024.0   171.744004   143.296003
5         16.0   2048.0   317.344010   264.463991
6         32.0     64.0    36.736000    32.607999
7         32.0    128.0    51.360000    45.024000
8         32.0    256.0    92.160001    77.823997
9         32.0    512.0   171.072006   143.424004
10        32.0   1024.0   317.312002   264.335990
11        32.0   2048.0   604.319990   501.488000
12        64.0     64.0    51.295999    44.992000
13        64.0    128.0    92.160001    77.760004
14        64.0    256.0   170.975998   143.552005
15        64.0    512.0   317.279994   264.351994
16        64.0   1024.0   603.327990   501.568019
17        64.0   2048.0  1178.463995   957.280010
18       128.0     64.0    92.128001    77.791996
19       128.0    128.0   170.816004   143.391997
20       128.0    256.0   317.247987   264.272004
21       128.0    512.0   603.456020   501.215994
22       128.0   1024.0  1178.591967   957.583994
23       128.0   2048.0  2327.423930  1868.095994

➜  sgl-kernel git:(main) ✗ python3 /home/yineng/bbuf/sglang/sgl-kernel/benchmark/bench_per_token_quant_fp8.py 
INFO 07-28 20:41:36 [__init__.py:235] Automatically detected platform cuda.
✅ All implementations match
per-token-dynamic-quant-fp8-performance:
    batch_size  seq_len         VLLM   SGL Kernel
0         16.0     64.0    21.183999    21.472000
1         16.0    128.0    28.511999    30.751999
2         16.0    256.0    43.903999    47.263999
3         16.0    512.0    78.879997    73.760003
4         16.0   1024.0   140.159994   131.040007
5         16.0   2048.0   265.632004   228.128001
6         16.0   4096.0   518.335998   424.495995
7         32.0     64.0    28.511999    30.688001
8         32.0    128.0    43.712001    47.375999
9         32.0    256.0    78.911997    73.951997
10        32.0    512.0   140.320003   131.072000
11        32.0   1024.0   265.760005   228.095993
12        32.0   2048.0   518.527985   424.416006
13        32.0   4096.0  1014.143944   799.488008
14        64.0     64.0    43.552000    47.263999
15        64.0    128.0    78.752004    73.696002
16        64.0    256.0   140.223995   131.007999
17        64.0    512.0   265.695989   228.192002
18        64.0   1024.0   518.416017   424.383998
19        64.0   2048.0  1014.783978   799.647987
20        64.0   4096.0  2010.143995  1548.256040
21       128.0     64.0    78.815997    73.792003
22       128.0    128.0   140.223995   130.943999
23       128.0    256.0   265.632004   227.904007
24       128.0    512.0   518.335998   424.224004
25       128.0   1024.0  1014.719963   799.647987
26       128.0   2048.0  2009.183884  1548.287988
27       128.0   4096.0  3996.495962  3045.552015
```

### pr

```shell

  sgl-kernel git:(main) ✗ python3 /home/yineng/bbuf/sglang/sgl-kernel/benchmark/bench_per_tensor_quant_fp8.py 
INFO 07-28 21:53:16 [__init__.py:235] Automatically detected platform cuda.
✅ All implementations match
per-tensor-quant-fp8-performance:
    batch_size  seq_len         VLLM   SGL Kernel
0         16.0     64.0    25.984000    23.968000
1         16.0    128.0    36.160000    28.096000
2         16.0    256.0    51.231999    36.800001
3         16.0    512.0    92.096001    61.087999
4         16.0   1024.0   171.552002   110.656001
5         16.0   2048.0   317.088008   206.367999
6         32.0     64.0    36.095999    27.968001
7         32.0    128.0    51.199999    36.575999
8         32.0    256.0    92.096001    60.927998
9         32.0    512.0   171.072006   110.912003
10        32.0   1024.0   316.832006   206.432000
11        32.0   2048.0   604.031980   399.648011
12        64.0     64.0    51.199999    36.640000
13        64.0    128.0    92.096001    60.736001
14        64.0    256.0   170.992002   110.944003
15        64.0    512.0   316.927999   206.367999
16        64.0   1024.0   603.200018   399.951994
17        64.0   2048.0  1177.872002   772.351980
18       128.0     64.0    93.152002    60.800001
19       128.0    128.0   170.975998   111.231998
20       128.0    256.0   316.895992   206.432000
21       128.0    512.0   603.424013   399.744004
22       128.0   1024.0  1178.319991   773.343980
23       128.0   2048.0  2327.344060  1525.455952

➜  sgl-kernel git:(main) ✗ python3 /home/yineng/bbuf/sglang/sgl-kernel/benchmark/bench_per_token_quant_fp8.py
INFO 07-28 21:52:31 [__init__.py:235] Automatically detected platform cuda.
✅ All implementations match
per-token-dynamic-quant-fp8-performance:
    batch_size  seq_len         VLLM   SGL Kernel
0         16.0     64.0    21.600001    19.936001
1         16.0    128.0    28.352000    28.543999
2         16.0    256.0    43.903999    36.991999
3         16.0    512.0    78.879997    56.768000
4         16.0   1024.0   140.592001   105.952002
5         16.0   2048.0   266.240001   183.487996
6         16.0   4096.0   519.136012   337.280005
7         32.0     64.0    29.088000    28.960001
8         32.0    128.0    44.415999    37.856001
9         32.0    256.0    79.328001    56.832001
10        32.0    512.0   140.415996   105.407998
11        32.0   1024.0   265.695989   182.559997
12        32.0   2048.0   518.735975   336.511999
13        32.0   4096.0  1014.271975   642.080009
14        64.0     64.0    45.952000    37.184000
15        64.0    128.0    79.200000    56.864001
16        64.0    256.0   140.640005   106.016003
17        64.0    512.0   266.207993   183.359995
18        64.0   1024.0   519.263983   337.568015
19        64.0   2048.0  1015.504003   643.136024
20        64.0   4096.0  2060.096025  1255.615950
21       128.0     64.0    81.568003    56.832001
22       128.0    128.0   140.799999   105.696000
23       128.0    256.0   266.463995   182.720006
24       128.0    512.0   519.263983   336.928010
25       128.0   1024.0  1015.679955   643.215984
26       128.0   2048.0  2011.104107  1271.391988
27       128.0   4096.0  4044.928074  2506.623983
```

可以在Nsight Compute得到kernel的SASS代码，我们可以通过对比指令发现这个性能提升的原因。

这个是修复后的`per_token_group_quant_8bit` kernel（https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/gemm/per_token_group_quant_8bit.cu）对应的SASS代码（从NCU Source部分复制的），我们可以分析一下`group_output[i * vec_size + j] = DST_DTYPE(q_val);`这行代码对应的SASS代码：      
```sass
LDC R1, c[0x0][0x28]
      S2R R3, SR_CTAID.Y
      ULDC UR4, c[0x0][0x0]
      S2R R24, SR_TID.X
      IMAD R3, R3, UR4, RZ
      ULDC UR4, c[0x0][0x22c]
      SHF.R.U32.HI R18, RZ, 0x5, R24
      LEA.HI R18, R3, R18, RZ, 0x1b
      IMAD.SHL.U32 R19, R18, 0x80, RZ
      ISETP.GE.AND P0, PT, R19, UR4, PT
@P0   EXIT
      S2UR UR38, SR_CTAID.X
      IMAD.SHL.U32 R0, R24, 0x4, RZ
      ULDC.64 UR4, c[0x0][0x220]
      ULDC.64 UR36, c[0x0][0x208]
      LOP3.LUT R0, R0, 0x7c, RZ, 0xc0, !PT
      LDC R2, c[0x0][0x230]
      IMAD.IADD R19, R19, 0x1, R0
      SHF.R.S32.HI R22, RZ, 0x1f, R19
      LDC R9, c[0x0][0x24c]
      IMAD R2, R2, UR38, RZ
      IADD3 R0, P0, R2, R19, RZ
      IMAD.X R3, RZ, RZ, R22, P0
      LEA R16, P0, R0, UR4, 0x1
      ULDC UR4, c[0x0][0x244]
      LEA.HI.X R17, R0, UR5, R3, 0x1, P0
      LDG.E.64 R16, desc[UR36][R16.64]
      F2FP.BF16.F32.PACK_AB R0, RZ, UR4
      BSSY B6, 0x7fce7d7d9150
      SHF.R.U64 R23, R16, 0x10, R17
      HFMA2.BF16_V2 R4, -RZ.H0_H0, RZ.H0_H0, |R17|.H0_H0
      SHF.R.U32.HI R2, RZ, 0x10, R17
      PRMT R3, R16, 0x5410, R23
      HFMA2.BF16_V2 R5, -RZ.H0_H0, RZ.H0_H0, |R2|.H0_H0
      HFMA2.BF16_V2 R3, -RZ.H0_H0, RZ.H0_H0, |R3|
      VHMNMX.BF16_V2 R4, R0.H0_H0, R3.H0_H0, R4.H0_H0, !PT
      VHMNMX.BF16_V2 R5, R0.H0_H0, R3.H1_H1, R5.H0_H0, !PT
      HMNMX2.BF16_V2 R4, R4.H0_H0, R5.H0_H0, !PT
      PRMT R4, R4, 0x5410, R4
      SHFL.BFLY PT, R3, R4, 0x10, 0x1f
      HMNMX2.BF16_V2 R3, R4.H0_H0, R3.H0_H0, !PT
      MUFU.RCP R4, R9
      PRMT R3, R3, 0x5410, R3
      SHFL.BFLY PT, R0, R3, 0x8, 0x1f
      HMNMX2.BF16_V2 R0, R3.H0_H0, R0.H0_H0, !PT
      PRMT R0, R0, 0x5410, R0
      SHFL.BFLY PT, R5, R0, 0x4, 0x1f
      HMNMX2.BF16_V2 R5, R0.H0_H0, R5.H0_H0, !PT
      PRMT R5, R5, 0x5410, R5
      SHFL.BFLY PT, R6, R5, 0x2, 0x1f
      HMNMX2.BF16_V2 R6, R5.H0_H0, R6.H0_H0, !PT
      PRMT R6, R6, 0x5410, R6
      SHFL.BFLY PT, R7, R6, 0x1, 0x1f
      HMNMX2.BF16_V2 R3, R6.H0_H0, R7.H0_H0, !PT
      FFMA R7, R4, -R9, 1
      IMAD.U32 R0, R3, 0x10000, RZ
      FFMA R7, R4, R7, R4
      FCHK P0, R0, R9
      FFMA R4, R0, R7, RZ
      FFMA R3, R4, -R9, R0
      FFMA R4, R7, R3, R4
@!P0  BRA 0x7fce7d7d9140
      ULDC UR4, c[0x0][0x24c]
      IMAD.MOV.U32 R4, RZ, RZ, R0
      MOV R20, 0x0
      IMAD.U32 R5, RZ, RZ, UR4
      MOV R21, 0x0
      CALL.ABS.NOINC 0x67c07d7d9f0100
      BSYNC B6
      F2F.BF16.F32 R3, R4
      IMAD.U32 R0, R16, 0x10000, RZ
      ULDC.64 UR4, c[0x0][0x248]
      IMAD.U32 R6, R23, 0x10000, RZ
      IMAD.U32 R8, R17, 0x10000, RZ
      IMAD.U32 R10, R2, 0x10000, RZ
      IMAD.U32 R7, RZ, RZ, UR4
      ULDC UR4, c[0x0][0x234]
      UIMAD UR4, UR38, UR4, URZ
      IMAD.U32 R3, R3, 0x10000, RZ
      FSETP.GE.AND P1, PT, |R3|, 8.50705917302346158658e+37, PT
@P1   FMUL R3, R3, 0.25
      FSETP.GEU.AND P0, PT, |R3|, 1.175494350822287508e-38, PT
@!P0  FMUL R3, R3, 16777216
@!P0  FMUL R0, R0, 16777216
@!P0  FMUL R6, R6, 16777216
@!P0  FMUL R8, R8, 16777216
      MUFU.RCP R5, R3
@!P0  FMUL R10, R10, 16777216
      LOP3.LUT P0, RZ, R24, 0x1f, RZ, 0xc0, !PT
      FMUL R0, R5, R0
      FMUL R2, R5, R6
      FMUL R3, R5, R8
      FMUL R5, R5, R10
@P1   FFMA R0, R0, 0.25, -RZ
@P1   FFMA R2, R2, 0.25, -RZ
@P1   FFMA R3, R3, 0.25, -RZ
@P1   FFMA R5, R5, 0.25, -RZ
      F2FP.BF16.F32.PACK_AB R8, RZ, UR5
      F2FP.BF16.F32.PACK_AB R0, R7, R0
      F2FP.BF16.F32.PACK_AB R2, RZ, R2
      F2FP.BF16.F32.PACK_AB R3, RZ, R3
      F2FP.BF16.F32.PACK_AB R9, RZ, R5
      HMNMX2.BF16_V2 R6, R2.H0_H0, R0.H1_H1, !PT
      HMNMX2.BF16_V2 R5, R0.H0_H0, R0.H1_H1, !PT
      HMNMX2.BF16_V2 R7, R3.H0_H0, R0.H1_H1, !PT
@!P0  LDC.64 R2, c[0x0][0x238]
      HMNMX2.BF16_V2 R9, R9.H0_H0, R0.H1_H1, !PT
      HMNMX2.BF16_V2 R0, R5.H0_H0, R8.H0_H0, PT
      HMNMX2.BF16_V2 R5, R6.H0_H0, R8.H0_H0, PT
      HMNMX2.BF16_V2 R10, R7.H0_H0, R8.H0_H0, PT
      IMAD.U32 R0, R0, 0x10000, RZ
      LDC.64 R6, c[0x0][0x210]
      HMNMX2.BF16_V2 R11, R9.H0_H0, R8.H0_H0, PT
      IMAD.U32 R5, R5, 0x10000, RZ
      IMAD.U32 R10, R10, 0x10000, RZ
      F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C R0, RZ, R0, RZ
      IMAD.U32 R11, R11, 0x10000, RZ
      F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C R5, RZ, R5, RZ
@!P0  LDC.64 R8, c[0x0][0x218]
      LOP3.LUT R0, R0, 0xff, RZ, 0xc0, !PT
      F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C R11, RZ, R11, RZ
      LOP3.LUT R5, R5, 0xffff, RZ, 0xc0, !PT
      F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C R10, RZ, R10, RZ
@!P0  IMAD R2, R2, UR38, RZ
      LOP3.LUT R11, R11, 0xffff, RZ, 0xc0, !PT
      PRMT R5, R5, 0x7604, R0
@!P0  IMAD R3, R18, R3, R2
      LOP3.LUT R10, R10, 0xff, RZ, 0xc0, !PT
      IMAD.SHL.U32 R11, R11, 0x1000000, RZ
      IADD3 R6, P1, P2, R19, UR4, R6
      PRMT R10, R5, 0x5410, R10
      IADD3.X R7, R22, R7, RZ, P1, P2
      LOP3.LUT R11, R10, R11, RZ, 0xfc, !PT
@!P0  IMAD.WIDE.U32 R8, R3, 0x4, R8
@!P0  STG.E desc[UR36][R8.64], R4
      STG.E desc[UR36][R6.64], R11
      EXIT
      BRA 0x7fce7d7d9590
```

我们要找出`group_output[i * vec_size + j] = DST_DTYPE(q_val);`这行代码对应的SASS代码部分但是对SASS代码不熟悉。这个时候借助claude-4是一个很好的选择。

我来分析这个CUDA kernel的SASS代码，找出`group_output[i * vec_size + j] = DST_DTYPE(q_val);`这行代码对应的SASS部分。

--------------------------------**Claude 4.0 生成分割线**--------------------------------

Read file: sglang/sgl-kernel/csrc/gemm/per_token_group_quant_8bit.cu
现在让我分析SASS代码。根据CUDA代码的逻辑，`group_output[i * vec_size + j] = DST_DTYPE(q_val);`这行代码是在第二个循环中，用于将量化后的值存储到输出缓冲区。

从SASS代码分析，我可以识别出以下几个关键部分：

1. **数据加载部分**（第一个循环）：
   - `LDG.E.64 R16, desc[UR36][R16.64]` - 加载输入数据
   - 后续的`SHF.R.U64`、`PRMT`等指令处理向量数据

2. **量化计算部分**：
   - `MUFU.RCP R4, R9` - 计算倒数（用于除法）
   - `FMUL R0, R5, R0`、`FMUL R2, R5, R6`等 - 执行量化计算（val / y_s）
   - `F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C` - 将float32转换为FP8格式

3. **数据存储部分**（对应`group_output[i * vec_size + j] = DST_DTYPE(q_val);`）：

```sass
STG.E desc[UR36][R6.64], R11
```

这行SASS代码就是对应`group_output[i * vec_size + j] = DST_DTYPE(q_val);`的存储操作。

具体分析：

- `STG.E` - 全局内存存储指令
- `desc[UR36][R6.64]` - 目标地址，其中`R6.64`包含了计算出的存储地址`(i * vec_size + j)`
- `R11` - 存储的数据，这是经过量化转换后的值

在SASS代码中，地址计算是通过以下指令完成的：
```sass
IADD3 R6, P1, P2, R19, UR4, R6
IADD3.X R7, R22, R7, RZ, P1, P2
```

而数据准备是通过：
```sass
F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C R0, RZ, R0, RZ
F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C R5, RZ, R5, RZ
F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C R11, RZ, R11, RZ
F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C R10, RZ, R10, RZ
```

这些指令将float32值转换为FP8格式，然后通过位操作组合成最终的存储值`R11`。

所以，**`group_output[i * vec_size + j] = DST_DTYPE(q_val);`这行CUDA代码对应的SASS代码就是`STG.E desc[UR36][R6.64], R11`**。

--------------------------------**Claude 4.0 生成分割线**--------------------------------

根据SASS代码分析，`DST_DTYPE(q_val)`这个类型转换对应的指令是：

```sass
F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C R0, RZ, R0, RZ
```

这个指令将float32转换为FP8 (E4M3)格式，其中：
- `F2FP` - Float to Float Packed (浮点数打包转换)
- `SATFINITE` - 饱和和有限性检查
- `E4M3` - 指数4位，尾数3位的FP8格式
- `F32` - 源格式是float32
- `PACK_AB_MERGE_C` - 打包并合并操作

然后如果把`DST_DTYPE`设置为`c10::Float8_e4m3fn`，也就是修复之前的情况，我们可以发现SASS代码变得非常复杂，会多出很多函数调用：

```sass
// 第1次调用
MOV R8, 0xc50
CALL.REL.NOINC 0x7fce7d7975c0

// 第2次调用  
MOV R8, 0xe50
CALL.REL.NOINC 0x7fce7d7975c0

// 第3次调用
MOV R8, 0x1080
CALL.REL.NOINC 0x7fce7d7975c0

// 第4次调用
MOV R8, 0x1290
CALL.REL.NOINC 0x7fce7d7975c0

// 第5次调用
MOV R8, 0x14b0
CALL.REL.NOINC 0x7fce7d7975c0

// 第6次调用
MOV R8, 0x16c0
CALL.REL.NOINC 0x7fce7d7975c0

// 第7次调用
MOV R8, 0x18e0
CALL.REL.NOINC 0x7fce7d7975c0

// 第8次调用
MOV R8, 0x1af0
CALL.REL.NOINC 0x7fce7d7975c0

// 第9次调用
MOV R8, 0x1d10
CALL.REL.NOINC 0x7fce7d7975c0
```

然后这个函数做的事请就是将float32转换为`c10::Float8_e4m3fn`类型，由于这个类型不是CUDA内置的`__nv_fp8_e4m3`类型，所以需要通过软件函数调用实现，包含大量条件分支和位操作最终导致了性能下降。

![](https://files.mdnice.com/user/59/331cb9c0-5c75-4dad-bc71-711bc4e0d668.png)

下面用Claude 4.0分析下这个转换函数。

--------------------------------**Claude 4.0 生成分割线**--------------------------------

我来为这段SASS代码添加注释说明：

```sass
@!P0  BRA 0x7fce7d796440                    // 如果条件P0为假，跳转到地址0x7fce7d796440
EXIT                                        // 退出当前执行块
SHF.R.U32.HI R9, RZ, 0x17, R3              // 将R3右移23位，结果存入R9（提取float32的指数部分）
BSSY B1, 0x7fce7d797c10                     // 开始同步块B1，目标地址0x7fce7d797c10
SHF.R.U32.HI R11, RZ, 0x17, R0             // 将R0右移23位，结果存入R11（提取另一个float32的指数部分）
IMAD.MOV.U32 R17, RZ, RZ, R3               // 将R3的值移动到R17
LOP3.LUT R9, R9, 0xff, RZ, 0xc0, !PT       // 将R9与0xff进行AND操作，提取低8位（指数值的有效位）
LOP3.LUT R11, R11, 0xff, RZ, 0xc0, !PT     // 将R11与0xff进行AND操作，提取低8位（指数值的有效位）
VIADD R12, R9, 0xffffffff                  // R12 = R9 - 1（指数值减1）
VIADD R13, R11, 0xffffffff                 // R13 = R11 - 1（指数值减1）
ISETP.GT.U32.AND P0, PT, R12, 0xfd, PT     // 设置条件P0：如果R12 > 253（0xfd）
ISETP.GT.U32.OR P0, PT, R13, 0xfd, P0      // 设置条件P0：如果R13 > 253（0xfd）或之前的P0为真
@!P0  IMAD.MOV.U32 R10, RZ, RZ, RZ         // 如果P0为假，将R10设置为0
```

## 功能说明

这段SASS代码的主要功能是**FP8转换的预处理和边界检查**：

1. **指数提取**：
   - `SHF.R.U32.HI R9, RZ, 0x17, R3` 和 `SHF.R.U32.HI R11, RZ, 0x17, R0`
   - 从float32值中提取指数部分（IEEE 754格式中，指数位于第23-30位）

2. **指数处理**：
   - `LOP3.LUT` 指令提取指数的有效位（去掉符号位）
   - `VIADD` 指令将指数值减1（用于后续的转换计算）

3. **边界检查**：
   - `ISETP.GT.U32.AND/OR` 指令检查指数值是否超过253
   - 这是FP8格式的指数范围检查，确保转换后的值在有效范围内

4. **条件设置**：
   - 根据边界检查结果设置条件标志P0
   - 如果超出范围，P0为真，否则为假

这段代码是FP8转换函数的一部分，负责在将float32转换为FP8格式之前进行必要的预处理和有效性检查。

--------------------------------**Claude 4.0 生成分割线**--------------------------------


最终问题的原因就明确了，因为`c10::Float8_e4m3fn`不是CUDA内置的`__nv_fp8_e4m3`类型，所以需要通过软件函数调用实现，包含大量条件分支和位操作最终导致了性能下降。而我们最开始实现quant相关的kernel时没有考虑到这一点，现在已经全部修复。