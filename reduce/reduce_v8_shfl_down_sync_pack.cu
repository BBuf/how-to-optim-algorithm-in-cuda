#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <algorithm>
#include <iostream>
#include <cuda_fp16.h>
using namespace std;

#define PackSize 4
#define kWarpSize 32
#define N 32 * 1024 * 1024
constexpr int BLOCK_SIZE = 256;

constexpr int kBlockSize = 256;
constexpr int kNumWaves = 1;

int64_t GetNumBlocks(int64_t n) {
  int dev;
  {
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) { return err; }
  }
  int sm_count;
  {
    cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) { return err; }
  }
  int tpm;
  {
    cudaError_t err = cudaDeviceGetAttribute(&tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    if (err != cudaSuccess) { return err; }
  }
  int64_t num_blocks = std::max<int64_t>(1, std::min<int64_t>((n + kBlockSize - 1) / kBlockSize,
                                                   sm_count * tpm / kBlockSize * kNumWaves));
  return num_blocks;
}

template<typename T, int pack_size>
struct alignas(sizeof(T) * pack_size) Packed {
  __device__ Packed(T val){
    #pragma unroll
    for(int i = 0; i < pack_size; i++){
        elem[i] = val; 
    }
  }
  __device__ Packed() {
    // do nothing
  }
  union {
    T elem[pack_size];
  };
  __device__ void operator+=(Packed<T, pack_size> packA){
    #pragma unroll 
    for(int i = 0; i < pack_size; i++){
        elem[i] += packA.elem[i]; 
    }
  }
};

template<typename T, int pack_size>
__device__ T PackReduce(Packed<T, pack_size> pack){
    T res = 0.0; 
    #pragma unroll
    for(int i = 0; i < pack_size; i++){
        res += pack.elem[i]; 
    }
    return res; 
}

template<typename T>
__device__ T warpReduceSum(T val){
    for(int lane_mask = 16; lane_mask > 0; lane_mask /=2){
        val += __shfl_down_sync(0xffffffff, val, lane_mask); 
    }
    return val; 
}

__global__ void reduce_v8(float *g_idata,float *g_odata, unsigned int n){

    // each thread loads one element from global to shared mem

    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    Packed<float, PackSize> sum_pack(0.0); 
    Packed<float, PackSize> load_pack(0.0); 
    const auto* pack_ptr = reinterpret_cast<const Packed<float, PackSize>*>(g_idata);
    
    for(int32_t linear_index = i; linear_index < n / PackSize; linear_index+=blockDim.x * gridDim.x){
        Packed<float, PackSize> g_idata_load = pack_ptr[linear_index];
        sum_pack += g_idata_load; 
    }
    float PackReduceVal = PackReduce<float, PackSize>(sum_pack);
    // Shared mem for partial sums (one per warp in the block)
    static __shared__ float warpLevelSums[kWarpSize]; 
    const int laneId = threadIdx.x % kWarpSize;
    const int warpId = threadIdx.x / kWarpSize;

    float sum = warpReduceSum<float>(PackReduceVal);
    __syncthreads();

    if(laneId == 0 )warpLevelSums[warpId] = sum;
    __syncthreads();
    // read from shared memory only if that warp existed
    sum = (threadIdx.x < blockDim.x / kWarpSize) ? warpLevelSums[laneId] : 0;
    // Final reduce using first warp
    if (warpId == 0) sum = warpReduceSum<float>(sum); 
    // write result for this block to global mem
    if (threadIdx.x == 0) g_odata[blockIdx.x] = sum;
}

bool check(float *out,float *res,int n){
    for(int i=0;i<n;i++){
        if(out[i]!=res[i])
            return false;
    }
    return true;
}

int main(){
    float *a=(float *)malloc(N*sizeof(float));
    float *d_a;
    cudaMalloc((void **)&d_a,N*sizeof(float));

    const int64_t block_num = GetNumBlocks(N); 
    printf("Block num is: %ld \n", block_num); 

    float *out=(float *)malloc(sizeof(float));
    float *g_odata;
    cudaMalloc((void **)&g_odata,block_num*sizeof(float));
    float *g_final_data;
    cudaMalloc((void **)&g_final_data,1*sizeof(float));

    for(int i=0;i<N;i++){
        a[i]=i;
    }
    float *res=(float *)malloc(sizeof(float));
    res[0] = N * (a[0] + a[N - 1]) / 2;

    cudaMemcpy(d_a,a,N*sizeof(float),cudaMemcpyHostToDevice);;

    reduce_v8<<<block_num, BLOCK_SIZE>>>(d_a, g_odata, N);
    reduce_v8<<<1, BLOCK_SIZE>>>(g_odata, g_final_data, block_num);

    cudaMemcpy(out,g_final_data,1*sizeof(float),cudaMemcpyDeviceToHost);

    if(check(out,res,1))printf("the answer is right\n");
    else{
        printf("the answer is wrong\n");
        printf("out[0]: %lf\n",out[0]);
        printf("res[0]: %lf\n", res[0]);
    }

    cudaFree(d_a);
    cudaFree(g_odata);
    free(a);
    free(out);
    return 0;
}

// 32768 * 32767 / 2 = 536854528
