> 本文由 @Simon V(https://github.com/simveit) 授权转载和翻译并发表到本公众号。原始地址为：https://veitner.bearblog.dev/load-and-store-matrices-efficently-with-ptx-instructions/

# 使用PTX指令更高效地加载和存储矩阵

2025年5月14日

## `ldmatrix`

从PTX文档(https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-ldmatrix)中我们可以看到,`ldmatrix`可以用于从共享内存中集体加载一个或多个矩阵,以供`mma`指令使用。

指令格式如下

```
ldmatrix.sync.aligned.shape.num{.trans}{.ss}.type r, [p];

ldmatrix.sync.aligned.m8n16.num{.ss}.dst_fmt.src_fmt        r, [p];
ldmatrix.sync.aligned.m16n16.num.trans{.ss}.dst_fmt.src_fmt r, [p];

.shape   = {.m8n8, .m16n16};
.num     = {.x1, .x2, .x4};
.ss      = {.shared{::cta}};
.type    = {.b16, .b8};
.dst_fmt = { .b8x16 };
.src_fmt = { .b6x16_p32, .b4x16_p64 };
```

This instruction will collectively load one or more matrices from `.shared` space to the registers.

- `p`: Adress operand in `.shared` space.
- `r`: Destination register
- `shape`: Shape of the matrix we load
- `num`: Load 1, 2 or 4 matrices

Possible datatypes are as follows:


|**.shape**| 	**Matrix shape**| 	**Element size**|
|:---:|:---:|:---:|
|`.m8n8`| 	8x8| 	16-bit|
|`.m16n16`| 	16x16| 	8-bit or 6-bit or 4-bit|
|`.m8n16`| 	8x16| 	6-bit or 4-bit|

Note that the shapes `m16n16` and `m8n16` are currently only available on sm_100 and higher GPU versions (i.e. currently Blackwell) which I unfortunately don't have access to. For that reason we will focus on `m8n8` instruction.

The below table shows which threads are responsible for which matrices. Each address below corresponds to a row in the matrix. Each "Threadgroup" (i.e. 0-7, 8-15, 16-23 and 24-31) loads a separate matrix. Consecutive rows should be stored consecutive in memory.

|**.num**| 	**Threads 0–7**| 	**Threads 8–15**| 	**Threads 16–23**| 	**Threads 24–31**|
|:---:|:---:|:---:|:---:|:---:|
|`.x1`| 	addr0–addr7| 	–| 	–| 	–|
|`.x2`| 	addr0–addr7| 	addr8–addr15| 	–| 	–|
|`.x4`| 	addr0–addr7| 	addr8–addr15| 	addr16–addr23| 	addr24–addr31|


The below picture illustrates the fragment layout of an `8x8` matrix loaded by `ldmatrix`:

![](https://files.mdnice.com/user/59/f5fbc9ae-36a9-4691-b38e-5cf1f36cda03.png)

```shell
// Load a single 8x8 matrix using 64-bit addressing
.reg .b64 addr;
.reg .b32 d;
ldmatrix.sync.aligned.m8n8.x1.shared::cta.b16 {d}, [addr];

// Load two 8x8 matrices in column-major format
.reg .b64 addr;
.reg .b32 d<2>;
ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {d0, d1}, [addr];

// Load four 8x8 matrices
.reg .b64 addr;
.reg .b32 d<4>;
ldmatrix.sync.aligned.m8n8.x4.b16 {d0, d1, d2, d3}, [addr];
```

## Implementation

As mentioned above the pointer is expected to reside in .shared space. There are multiple ways to convert a generic pointer to .shared space. Probably the simplest way is as follows(https://forums.developer.nvidia.com/t/problem-about-ptx-instruction-cp-async-ca-shared-global/224219/2):

```c++
size_t asl = __cvta_generic_to_shared(smem+threadIdx.x);
```

We could also use inline assembly:

```c++
asm volatile(".reg .u64 smem_ptr64; cvta.to.shared.u64 smem_ptr64, %0;\n" :: "l"(smem+threadIdx.x));
```

Or something like:

```c++
asm volatile(".reg .u64 smem_ptr64; cvta.to.shared.u64 smem_ptr64, %0;\n" :: "l"(smem+threadIdx.x)); 
asm volatile(".reg .u32 smem_ptr32; cvt.u32.u64 smem_ptr32, smem_ptr64;\n" ::);
```

We can look also at the CUTLASS(https://github.com/NVIDIA/cutlass/blob/ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e/include/cute/arch/copy_sm75.hpp#L39) library to get an idea for implementation.

From there implementation is pretty straightforward:

```c++
#include <cstdint>
#include <iostream>

__device__ __forceinline__ void ldmatrix_sync_aligned_m8n8_x1_b16(
    uint32_t &d0, const uint32_t &address) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];"
               : "=r"(d0)
               : "r"(address));
}

__global__ void ldmatrix(uint16_t *value) {
  constexpr int N = 64;
  __shared__ uint16_t smem[N];
  auto tid = threadIdx.x;

  const uint32_t offset_rows = sizeof(uint16_t) * (tid % 8) * 8;
  const uint32_t address = __cvta_generic_to_shared(smem) + offset_rows;

  for (uint32_t i = tid; i < N; i += blockDim.x) {
    smem[i] = i;
  }
  __syncthreads();

  uint32_t frag;
  ldmatrix_sync_aligned_m8n8_x1_b16(frag, address);

  __syncthreads();

  uint16_t number1 = static_cast<uint16_t>(frag & 0xFFFF);
  uint16_t number2 = static_cast<uint16_t>((frag >> 16) & 0xFFFF);
  printf("%d -> %d  %d   %d   \n", tid, (int)(smem[2 * tid]), (int)number1,
         (int)number2);
}

int main() {
  uint16_t *d_value;
  cudaMalloc(&d_value, sizeof(uint16_t));
  ldmatrix<<<1, 32>>>(d_value);
  cudaDeviceSynchronize();
  cudaFree(d_value);
  return 0;
}
```

Note that by the above tables threads 0-7 need to correspond to the addresses of the first 8 rows:

```c++
const uint32_t offset_rows = sizeof(uint16_t) * (tid % 8) * 8;
const uint32_t address = __cvta_generic_to_shared(smem) + offset_rows;
```

We'll than pass the addresses along with the fragment we save load to. Note that each fragment has `32bit` and we can output the loaded fragment by first masking with a full 16bit mask to extract the last 16 bits, than right shift and do the same again to extract the first 16 bits.

```shell
0 -> 0  0   1   
1 -> 2  2   3   
2 -> 4  4   5   
3 -> 6  6   7   
4 -> 8  8   9   
5 -> 10  10   11   
6 -> 12  12   13   
7 -> 14  14   15   
8 -> 16  16   17   
9 -> 18  18   19   
10 -> 20  20   21   
11 -> 22  22   23   
12 -> 24  24   25   
13 -> 26  26   27   
14 -> 28  28   29   
15 -> 30  30   31   
16 -> 32  32   33   
17 -> 34  34   35   
18 -> 36  36   37   
19 -> 38  38   39   
20 -> 40  40   41   
21 -> 42  42   43   
22 -> 44  44   45   
23 -> 46  46   47   
24 -> 48  48   49   
25 -> 50  50   51   
26 -> 52  52   53   
27 -> 54  54   55   
28 -> 56  56   57   
29 -> 58  58   59   
30 -> 60  60   61   
31 -> 62  62   63
```

We can see that each register contains two values.

We can similar write two matrices collectively in a warp. We should take into account that the addresses are provided per thread group:

|**.num**| 	**Threads 0–7**| 	**Threads 8–15**| 	**Threads 16–23**| 	**Threads 24–31**|
|:---:|:---:|:---:|:---:|:---:|
|.x1| 	addr0–addr7| 	–| 	–| 	–|
|.x2| 	addr0–addr7| 	addr8–addr15| 	–| 	–|
|.x4| 	addr0–addr7| 	addr8–addr15| 	addr16–addr23| 	addr24–addr31|


The syntax for `ldmatrix` with `x2` is as follows

```c++
__device__ __forceinline__ void ldmatrix_sync_aligned_m8n8_x2_b16(
    uint32_t &d0, uint32_t &d1, const uint32_t &address) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
               : "=r"(d0), "=r"(d1)
               : "r"(address));
}
```

Note that now we write to `32bit` fragments.

We can wrap this as follows into a kernel:

```c++
__global__ void ldmatrix(uint16_t *value) {
  constexpr int N = 64;
  __shared__ uint16_t smem[2 * N];
  auto tid = threadIdx.x;

  const uint32_t offset_rows = sizeof(uint16_t) * (tid % 8) * 8;
  const uint32_t offset_matrix = sizeof(uint16_t) * ((tid / 8) % 2) * 64;
  const uint32_t offset = offset_rows + offset_matrix;
  const uint32_t address = __cvta_generic_to_shared(smem) + offset;

  for (uint32_t i = tid; i < N; i += blockDim.x) {
    smem[i] = i;
    smem[i + 64] = i + 64;
  }
  __syncthreads();

  uint32_t frag1;
  uint32_t frag2;
  ldmatrix_sync_aligned_m8n8_x2_b16(frag1, frag2, address);

  __syncthreads();

  uint16_t number1 = static_cast<uint16_t>(frag1 & 0xFFFF);
  uint16_t number2 = static_cast<uint16_t>((frag1 >> 16) & 0xFFFF);
  printf("%d -> %d  %d   %d   \n", tid, (int)(smem[2 * tid]), (int)number1,
         (int)number2);
  uint16_t number3 = static_cast<uint16_t>(frag2 & 0xFFFF);
  uint16_t number4 = static_cast<uint16_t>((frag2 >> 16) & 0xFFFF);
  printf("%d -> %d  %d   %d   \n", tid, (int)(smem[2 * tid + 64]), (int)number3,
         (int)number4);
}
```

The logic to calculate address is as follows:

```c++
const uint32_t offset_rows = sizeof(uint16_t) * (tid % 8) * 8;
const uint32_t offset_matrix = sizeof(uint16_t) * ((tid / 8) % 2) * 64;
const uint32_t offset = offset_rows + offset_matrix;
const uint32_t address = __cvta_generic_to_shared(smem) + offset;
```

We need to calculate an offset for the rows as well as an offset for the matrices. The first `8` threads provide the addresses for the first matrix. The next `8` threads provide the addresses for the second matrix.

We can very similar load 4 `8x8` matrices collectively across a warp. The syntax is as follows:

```c++
__device__ __forceinline__ void ldmatrix_sync_aligned_m8n8_x2_b16(
    uint32_t &d0, uint32_t &d1, uint32_t &d2, uint32_t &d3,
    const uint32_t &address) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      : "r"(address));
}
```

The full kernel looks as follows:

```c++
__global__ void ldmatrix(uint16_t *value) {
  constexpr int N = 64;
  __shared__ uint16_t smem[4 * N];
  auto tid = threadIdx.x;

  const uint32_t offset_rows = sizeof(uint16_t) * (tid % 8) * 8;
  const uint32_t offset_matrix = sizeof(uint16_t) * ((tid / 8) % 4) * 64;
  const uint32_t offset = offset_rows + offset_matrix;
  const uint32_t address = __cvta_generic_to_shared(smem) + offset;

  for (uint32_t i = tid; i < N; i += blockDim.x) {
    smem[i] = i;
    smem[i + 64] = i + 64;
    smem[i + 128] = i + 128;
    smem[i + 192] = i + 192;
  }
  __syncthreads();

  uint32_t frag1;
  uint32_t frag2;
  uint32_t frag3;
  uint32_t frag4;
  ldmatrix_sync_aligned_m8n8_x2_b16(frag1, frag2, frag3, frag4, address);

  __syncthreads();

  uint16_t number1 = static_cast<uint16_t>(frag1 & 0xFFFF);
  uint16_t number2 = static_cast<uint16_t>((frag1 >> 16) & 0xFFFF);
  printf("%d -> %d  %d   %d   \n", tid, (int)(smem[2 * tid]), (int)number1,
         (int)number2);
  uint16_t number3 = static_cast<uint16_t>(frag2 & 0xFFFF);
  uint16_t number4 = static_cast<uint16_t>((frag2 >> 16) & 0xFFFF);
  printf("%d -> %d  %d   %d   \n", tid, (int)(smem[2 * tid + 64]), (int)number3,
         (int)number4);
  uint16_t number5 = static_cast<uint16_t>(frag3 & 0xFFFF);
  uint16_t number6 = static_cast<uint16_t>((frag3 >> 16) & 0xFFFF);
  printf("%d -> %d  %d   %d   \n", tid, (int)(smem[2 * tid + 128]),
         (int)number5, (int)number6);
  uint16_t number7 = static_cast<uint16_t>(frag4 & 0xFFFF);
  uint16_t number8 = static_cast<uint16_t>((frag4 >> 16) & 0xFFFF);
  printf("%d -> %d  %d   %d   \n", tid, (int)(smem[2 * tid + 192]),
         (int)number7, (int)number8);
}
```



Address calculation is similar. We have again groups of 8 threads that provide addresses for 8 rows of each of the 4 matrices, so in total all `32` threads in the warp provide addresses now.

```c++
const uint32_t offset_rows = sizeof(uint16_t) * (tid % 8) * 8;
const uint32_t offset_matrix = sizeof(uint16_t) * ((tid / 8) % 4) * 64;
const uint32_t offset = offset_rows + offset_matrix;
const uint32_t address = __cvta_generic_to_shared(smem) + offset;
```

Each of the kernels can be invoked as follows:

```c++
int main() {
  uint16_t *d_value;
  cudaMalloc(&d_value, sizeof(uint16_t));
  ldmatrix<<<1, 32>>>(d_value);
  cudaDeviceSynchronize();
  cudaFree(d_value);
  return 0;
}
```

## stmatrix

`stmatrix` is a PTX instruction to collectively store one or more matrices to shared memory.

```c++
stmatrix.sync.aligned.shape.num{.trans}{.ss}.type [p], r;

.shape  = {.m8n8, .m16n8};
.num    = {.x1, .x2, .x4};
.ss     = {.shared{::cta}};
.type   = {.b16, .b8};
```



As you can see the instruction is similar to the `ldmatrix`. `.m8n8` is available from Hopper, `m16n8` is available from Blackwell GPUs.

The adresses are provided in same way as above. Only this time we provide the addresses to know which location the content of the provided register(s) get stored to.

|**.num**|**Threads 0–7**|**Threads 8–15**|**Threads 16–23**|**Threads 24–31**|
|--------|-----------|------------|-------------|--------------|
|`.x1`|addr0–addr7|–|–|–|
|`.x2`|addr0–addr7|addr8–addr15|–|–| 
|`.x4`|addr0–addr7|addr8–addr15|addr16–addr23|addr24–addr31|

### Implementation

The implementation is not difficult once we properly understood the `ldmatrix` instruction from above as they are highly similar. Please make sure you understood the above code before you continue reading.

The below code gives a simple wrapper around the PTX instruction above and stores one `8x8` matrix collectively.

```c++
__device__ __forceinline__ void stmatrix_sync_aligned_m8n8_x1_b16(
    uint32_t &d0, const uint32_t &address) {
  asm volatile(
      "stmatrix.sync.aligned.x1.m8n8.shared.b16 [%0], {%1};\n" ::"r"(address),
      "r"(d0));
}
```

We can put this into a kernel as follows:

```c++
__global__ void stmatrix(uint16_t *value) {
  constexpr int N = 64;
  __shared__ uint16_t smem[N];
  auto tid = threadIdx.x;

  const uint32_t offset_rows = sizeof(uint16_t) * (tid % 8) * 8;
  const uint32_t address = __cvta_generic_to_shared(smem) + offset_rows;

  uint32_t frag = 0x00000000;
  frag |= (tid * 2 + 0);
  frag |= (tid * 2 + 1) << 16;
  __syncthreads();

  stmatrix_sync_aligned_m8n8_x1_b16(frag, address);

  __syncthreads();

  uint16_t number1 = static_cast<uint16_t>(frag & 0xFFFF);
  uint16_t number2 = static_cast<uint16_t>((frag >> 16) & 0xFFFF);
  printf("%d -> %d  %d   %d   \n", tid, (int)(smem[2 * tid]), (int)number1,
         (int)number2);
}
```

Most of the code is similar as above. But this time we define a fragment and store it to an address in shared memory.

The below code initialises a 32bit unsigned integer. We'll first initialise the first 16 bits with `2 * tid + 0` and than the last 16bits with `2 * tid + 1`. This is mainly to replicate the same end result as in the `ldmatrix` example.

```c++
uint32_t frag = 0x00000000;
frag |= (tid * 2 + 0);
frag |= (tid * 2 + 1) << 16;
```

We'll than store the fragment into the given address. This will output:

```shell
0 -> 0  0   1   
1 -> 2  2   3   
2 -> 4  4   5   
3 -> 6  6   7   
4 -> 8  8   9   
5 -> 10  10   11   
6 -> 12  12   13   
7 -> 14  14   15   
8 -> 16  16   17   
9 -> 18  18   19   
10 -> 20  20   21   
11 -> 22  22   23   
12 -> 24  24   25   
13 -> 26  26   27   
14 -> 28  28   29   
15 -> 30  30   31   
16 -> 32  32   33   
17 -> 34  34   35   
18 -> 36  36   37   
19 -> 38  38   39   
20 -> 40  40   41   
21 -> 42  42   43   
22 -> 44  44   45   
23 -> 46  46   47   
24 -> 48  48   49   
25 -> 50  50   51   
26 -> 52  52   53   
27 -> 54  54   55   
28 -> 56  56   57   
29 -> 58  58   59   
30 -> 60  60   61   
31 -> 62  62   63   
```

This confirms that our implementation reverses the above `ldmatrix` operation as expected.

The implementation for storing to 2 or 4 matrices is very similar:

```c++
__device__ __forceinline__ void stmatrix_sync_aligned_m8n8_x2_b16(
    uint32_t &d0, uint32_t &d1, const uint32_t &address) {
  asm volatile(
      "stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};" ::"r"(address),
      "r"(d0), "r"(d1));
}

__global__ void stmatrix(uint16_t *value) {
  constexpr int N = 64;
  __shared__ uint16_t smem[2 * N];
  auto tid = threadIdx.x;

  const uint32_t offset_rows = sizeof(uint16_t) * (tid % 8) * 8;
  const uint32_t offset_matrix = sizeof(uint16_t) * ((tid / 8) % 2) * 64;
  const uint32_t offset = offset_rows + offset_matrix;
  const uint32_t address = __cvta_generic_to_shared(smem) + offset;

  uint32_t frag1 = 0x00000000;
  frag1 |= (tid * 2 + 0);
  frag1 |= (tid * 2 + 1) << 16;
  uint32_t frag2 = 0x00000000;
  frag2 |= (tid * 2 + 0 + 64);
  frag2 |= (tid * 2 + 1 + 64) << 16;
  __syncthreads();

  stmatrix_sync_aligned_m8n8_x2_b16(frag1, frag2, address);

  __syncthreads();

  uint16_t number1 = static_cast<uint16_t>(frag1 & 0xFFFF);
  uint16_t number2 = static_cast<uint16_t>((frag1 >> 16) & 0xFFFF);
  printf("%d -> %d  %d   %d   \n", tid, (int)(smem[2 * tid]), (int)number1,
         (int)number2);
  uint16_t number3 = static_cast<uint16_t>(frag2 & 0xFFFF);
  uint16_t number4 = static_cast<uint16_t>((frag2 >> 16) & 0xFFFF);
  printf("%d -> %d  %d   %d   \n", tid, (int)(smem[2 * tid + 64]), (int)number3,
         (int)number4);
}
```

and for four matrices

```c++
__device__ __forceinline__ void stmatrix_sync_aligned_m8n8_x4_b16(
    uint32_t &d0, uint32_t &d1, uint32_t &d2, uint32_t &d3,
    const uint32_t &address) {
  asm volatile(
      "stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};" ::"r"(
          address),
      "r"(d0), "r"(d1), "r"(d2), "r"(d3));
}

__global__ void stmatrix(uint16_t *value) {
  constexpr int N = 64;
  __shared__ uint16_t smem[4 * N];
  auto tid = threadIdx.x;

  const uint32_t offset_rows = sizeof(uint16_t) * (tid % 8) * 8;
  const uint32_t offset_matrix = sizeof(uint16_t) * ((tid / 8) % 4) * 64;
  const uint32_t offset = offset_rows + offset_matrix;
  const uint32_t address = __cvta_generic_to_shared(smem) + offset;

  uint32_t frag1 = 0x00000000;
  frag1 |= (tid * 2 + 0);
  frag1 |= (tid * 2 + 1) << 16;
  uint32_t frag2 = 0x00000000;
  frag2 |= (tid * 2 + 0 + 64);
  frag2 |= (tid * 2 + 1 + 64) << 16;
  uint32_t frag3 = 0x00000000;
  frag3 |= (tid * 2 + 0 + 128);
  frag3 |= (tid * 2 + 1 + 128) << 16;
  uint32_t frag4 = 0x00000000;
  frag4 |= (tid * 2 + 0 + 192);
  frag4 |= (tid * 2 + 1 + 192) << 16;
  __syncthreads();

  stmatrix_sync_aligned_m8n8_x4_b16(frag1, frag2, frag3, frag4, address);

  __syncthreads();

  uint16_t number1 = static_cast<uint16_t>(frag1 & 0xFFFF);
  uint16_t number2 = static_cast<uint16_t>((frag1 >> 16) & 0xFFFF);
  printf("%d -> %d  %d   %d   \n", tid, (int)(smem[2 * tid]), (int)number1,
         (int)number2);
  uint16_t number3 = static_cast<uint16_t>(frag2 & 0xFFFF);
  uint16_t number4 = static_cast<uint16_t>((frag2 >> 16) & 0xFFFF);
  printf("%d -> %d  %d   %d   \n", tid, (int)(smem[2 * tid + 64]), (int)number3,
         (int)number4);
  uint16_t number5 = static_cast<uint16_t>(frag3 & 0xFFFF);
  uint16_t number6 = static_cast<uint16_t>((frag3 >> 16) & 0xFFFF);
  printf("%d -> %d  %d   %d   \n", tid, (int)(smem[2 * tid + 128]),
         (int)number5, (int)number6);
  uint16_t number7 = static_cast<uint16_t>(frag4 & 0xFFFF);
  uint16_t number8 = static_cast<uint16_t>((frag4 >> 16) & 0xFFFF);
  printf("%d -> %d  %d   %d   \n", tid, (int)(smem[2 * tid + 192]),
         (int)number7, (int)number8);
}
```

The only thing we need to do here is to initialise more fragments. When storing to 2 matrices we provide 2 fragments, when storing to 4 matrices we provide 4 fragments.

## Conclusion

I hope this blogpost was helpful to
- understand the ldmatrix and stmatrix in detail
- see the nice symmetry between the two instructions to deepen understanding further. If you want to read more you can refer to the PTX docs. I would be happy to get your feedback on this blogpost and discuss CUDA and related topic. If you want to reach out to me you can add me in Linkedin(https://www.linkedin.com/in/simon-veitner-174a681b6/). The code can be found in my repo














