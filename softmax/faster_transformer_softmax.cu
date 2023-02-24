#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <algorithm>
#include <iostream>
#include <cuda_fp16.h>
#include <math_constants.h>
using namespace std;

// source from https://github.com/NVIDIA/FasterTransformer/blob/release/v1.0_tag/fastertransformer/cuda/open_attention.cu#L189-L268

#define FINAL_MASK 0xffffffff

template <typename T>
__inline__ __device__
T warpReduceSum(T val)
{
  for(int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}

/* Calculate the sum of all elements in a block */
template <typename T>
  __inline__ __device__
T blockReduceSum(T val)
{
  static __shared__ T shared[32]; 
  int lane = threadIdx.x & 0x1f; 
  int wid = threadIdx.x >> 5;  

  val = warpReduceSum<T>(val);

  if(lane == 0)
    shared[wid] = val;

  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : (T)(0.0f);
  val = warpReduceSum<T>(val);
                              
  return val;
}

template <typename T>
  __inline__ __device__
T warpReduceMax(T val)
{
  for(int mask = 16; mask > 0; mask >>= 1)
    val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
  return val;
}

/* Calculate the maximum of all elements in a block */
template <typename T>
  __inline__ __device__
T blockReduceMax(T val)
{
  static __shared__ T shared[32]; 
  int lane = threadIdx.x & 0x1f; // in-warp idx
  int wid = threadIdx.x >> 5;  // warp idx

  val = warpReduceMax(val); // get maxx in each warp

  if(lane == 0) // record in-warp maxx by warp Idx
    shared[wid] = val;

  __syncthreads();


  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : -1e20f;
  val = warpReduceMax(val);

  return val;
}

// 以 BERT 为例
// query, key, value shape: [batch_size, head_num, seq_len, size_per_head]
// qk_buf shape: [batch_size, head_num, seq_len, seq_len]
// attr_mask shape: [seq_len, seq_len]
// scaler: 缩放系数
template <typename T>
__global__
void softmax_kernel(T* qk_buf_, /*const T* attr_mask*/ const int batch_size, const int head_num, const int seq_len, 
  const T scaler)
{
    // int batch_id = blockIdx.x / head_num;
    int qk_offset = blockIdx.x * seq_len * seq_len;
    // int mask_offset = batch_id * seq_len * seq_len;

    __shared__ float s_sum, s_max;

    for(int i = 0; i < seq_len; ++i)
    {
      float qk = threadIdx.x < seq_len ? (float)qk_buf_[threadIdx.x + qk_offset] : 0.0f;
    //   float mask_val = threadIdx.x < seq_len ? (float)attr_mask[threadIdx.x + mask_offset] : 0.0f;
      
    //   mask_val = (1.0f - mask_val) * -10000.0f;

    //   float tmp = threadIdx.x < seq_len ? (float)(qk * (float)scaler + mask_val): -1e20f;
      float tmp = -1e20f;

      float max_val = blockReduceMax<float>(tmp);

      if(threadIdx.x == 0)
        s_max = max_val;
      __syncthreads();

      qk = threadIdx.x < seq_len ? __expf(tmp - s_max) : 0.0f;

      float sum_val = blockReduceSum<float>(qk);

      if(threadIdx.x == 0)
      {
        s_sum = sum_val + 1e-6f;
      }
      __syncthreads();

      if(threadIdx.x < seq_len)
        qk_buf_[threadIdx.x + qk_offset] = (T)(qk / s_sum);

      qk_offset += seq_len;
    //   mask_offset += seq_len;
    }
}

template <typename T>
__global__
void softmax_kernel_v2(T* qk_buf_, /*const T* attr_mask, */const int batch_size, const int head_num, 
  const int seq_len, const T scaler)
{
    // int batch_id = blockIdx.x / head_num / seq_len;
    // int seq_id = blockIdx.x % seq_len;
    int qk_offset = blockIdx.x * seq_len;
    // int mask_offset = batch_id * seq_len * seq_len + seq_id * seq_len;

    __shared__ float s_sum, s_max;

    float qk = threadIdx.x < seq_len ? (float)qk_buf_[threadIdx.x + qk_offset] : 0.0f;
    // float mask_val = threadIdx.x < seq_len ? (float)attr_mask[threadIdx.x + mask_offset] : 0.0f;
      
    // mask_val = (1.0f - mask_val) * -10000.0f;

    // float tmp = threadIdx.x < seq_len ? (float)(qk * (float)scaler + mask_val) : -1e20f;
    float tmp = -1e20f;
    float max_val = blockReduceMax<float>(tmp);
    if(threadIdx.x == 0)
      s_max = max_val;
    __syncthreads();

    float qk_tmp = threadIdx.x < seq_len ? __expf((float)(tmp - s_max)) : 0.0f;
    float sum_val = blockReduceSum<float>(qk_tmp);

    if(threadIdx.x == 0)
    {
      s_sum = sum_val + 1e-6f;
    }
    __syncthreads();

    if(threadIdx.x < seq_len)
      qk_buf_[threadIdx.x + qk_offset] = (T)(qk_tmp / s_sum);
}

int main(){
  // input shape: [bacth_size, head_num, seq_len, seq_len]
  const int batch_size = 32;
  const int head_num = 64;
  const int seq_len = 512;
  const float scaler = 1.0;
  const int N = batch_size * head_num * seq_len * seq_len;

  float* input_host = (float*)malloc(N*sizeof(float));
  float *input_device;
  cudaMalloc((void **)&input_device, N*sizeof(float));
  for (int i = 0; i < N; i++) input_host[i] = 1.0;
  cudaMemcpy(input_device, input_host, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  dim3 grid, block;
  if(seq_len <= 32)
    block.x = 32;
  else if(seq_len > 32 && seq_len <= 64)
    block.x = 64;
  else if(seq_len > 64 && seq_len <= 128)
    block.x = 128;
  else if(seq_len > 128 && seq_len <= 256)
    block.x = 256;
  else if(seq_len > 256 && seq_len <= 512)
    block.x = 512;
  else
    block.x = 1024;
  // 如果 batch_size 和 head_num 的乘积 <= 120，block 数量设置为 batch_size * head_num * seq_len，也就是一个 block 负责处理 seq_len 个元素
  if(batch_size * head_num <= 120)
  {
    grid.x = batch_size * head_num * seq_len;
    softmax_kernel_v2<float><<<grid, block, 0, stream>>>(input_device, /*attr_mask*/ batch_size, head_num, seq_len, scaler); 
  }
  // 否则，block的数量设置为 batch_size * head_num, 也就是一个 block 负责处理 seq_len * seq_len 个元素
  else
  {
    grid.x = batch_size * head_num;
    softmax_kernel<float><<<grid, block, 0, stream>>>(input_device, /*attr_mask*/ batch_size, head_num, seq_len, scaler); 
  }
  float *output_host = (float*)malloc(N * sizeof(float));
  cudaMemcpy(output_host, input_device, N * sizeof(float), cudaMemcpyDeviceToHost);
  // 1 / 32 = 0.03125
  for (int i = 0; i < 32; i++){
    printf("%.5f\n", output_host[i]);
  }
  return 0;
}