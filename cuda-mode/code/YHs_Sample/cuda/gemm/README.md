**sgemm.cu:**
&ensp;General FP32 GEMM implementation, optimized for Volta/Turing and GA100 GPU and large matrix, support sm35+ devices. 

**ampere_sgemm.cu:**
&ensp;FP32 GEMM implementation, only support Ampere GPU (sm80+), optimized for GA102/104/106 GPU.

**cublas.cu:**
&ensp;cublas benchmark.

**build.sh:**
&ensp;building script.

---

building:

```bash
# for volta GPU:
$ sh build.sh sgemm.cu 70

#for turing GPU:
$ sh build.sh sgemm.cu 75

# for GA100 GPU:
$ sh build.sh sgemm.cu 80
# or
$ sh build.sh ampere_sgemm.cu 80

# for GA10x(x >= 2) GPU:
$ sh build.sh sgemm.cu 86
# or
$ sh build.sh ampere_sgemm.cu 86

# build cublas benchmark:
$ sh build.sh cublas.cu 70
```

run:

```bash
$ ./a.out
```

---

for benchmark, please remove the code:

```c++
    bool chk = check(h_A, h_B, h_C, m, n, k);
    printf("Matrix_C check: %s\n", chk ? "OK" : "Failed");
```

because this is the output check by a serialized non-optimized c++ GEMM, it will take a long time.

the matrix shape is hard coded in the source:

```c++
    int m = 5120;     // number of rows of matrix_A
    int n = 4096;     // number of columns of matrix_B
    int k = 4096;     // number of columns of matrix_A or rows of matrixB
    int n_iter = 10;  // benchmark iteration
```

for other matrix shape, please modify the *m/n/k*

---

if you benchmark on T4/A10 GPU, please run it by `nvprof`, `nv-nsight-cu-cli` or `nsys nvprof` and take

the fastest round as result, because T4/A10 has a strict power limitation and the peek clock can keep only

10~15ms for FFMA compute intensive applications.
