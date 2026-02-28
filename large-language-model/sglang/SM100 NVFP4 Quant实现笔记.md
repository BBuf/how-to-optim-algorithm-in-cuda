

## `nvfp4_quant_entry.cu` 实现

```c++
// 检查是否启用了NVFP4支持（仅在SM100架构上可用）
#if defined ENABLE_NVFP4 && ENABLE_NVFP4

// SM100架构专用的FP4量化函数声明
void scaled_fp4_quant_sm100a(
    torch::Tensor& output,        // 输出量化后的FP4张量
    torch::Tensor const& input,   // 输入的FP16/BF16张量
    torch::Tensor& output_sf,     // 输出缩放因子
    torch::Tensor const& input_sf // 输入缩放因子
);

// SM100架构专用的专家模型FP4量化函数声明
void scaled_fp4_experts_quant_sm100a(
    torch::Tensor& output,                              // 输出量化后的FP4张量
    torch::Tensor& output_scale,                        // 输出缩放因子
    torch::Tensor const& input,                         // 输入张量
    torch::Tensor const& input_global_scale,            // 输入全局缩放因子
    torch::Tensor const& input_offset_by_experts,       // 专家偏移量
    torch::Tensor const& output_scale_offset_by_experts // 输出缩放因子偏移量
);

// SM100架构专用的SiLU激活+乘法+专家模型FP4量化函数声明
void silu_and_mul_scaled_fp4_experts_quant_sm100a(
    torch::Tensor& output,                       // 输出量化后的FP4张量
    torch::Tensor& output_scale,                 // 输出缩放因子
    torch::Tensor const& input,                  // 输入张量
    torch::Tensor const& input_global_scale,     // 输入全局缩放因子
    torch::Tensor const& mask,                   // 掩码张量（用于门控）
    bool use_silu_and_mul                        // 是否使用SiLU激活和乘法
);

#endif

// 通用FP4量化接口函数
void scaled_fp4_quant(
    torch::Tensor& output,        // 输出量化后的FP4张量
    torch::Tensor const& input,   // 输入的FP16/BF16张量
    torch::Tensor& output_sf,     // 输出缩放因子
    torch::Tensor const& input_sf // 输入缩放因子
) {
#if defined ENABLE_NVFP4 && ENABLE_NVFP4
  // 如果支持NVFP4，调用SM100专用实现
  return scaled_fp4_quant_sm100a(output, input, output_sf, input_sf);
#endif
  // 如果不支持NVFP4，抛出未实现错误
  TORCH_CHECK_NOT_IMPLEMENTED(false, "No compiled nvfp4 quantization");
}

// 通用专家模型FP4量化接口函数
void scaled_fp4_experts_quant(
    torch::Tensor& output,                              // 输出量化后的FP4张量
    torch::Tensor& output_scale,                        // 输出缩放因子
    torch::Tensor const& input,                         // 输入张量
    torch::Tensor const& input_global_scale,            // 输入全局缩放因子
    torch::Tensor const& input_offset_by_experts,       // 专家偏移量
    torch::Tensor const& output_scale_offset_by_experts // 输出缩放因子偏移量
) {
#if defined ENABLE_NVFP4 && ENABLE_NVFP4
  // 如果支持NVFP4，调用SM100专用实现
  return scaled_fp4_experts_quant_sm100a(
      output, output_scale, input, input_global_scale, input_offset_by_experts, output_scale_offset_by_experts);
#endif
  // 如果不支持NVFP4，抛出未实现错误
  TORCH_CHECK_NOT_IMPLEMENTED(false, "No compiled nvfp4 experts quantization kernel");
}

// 通用SiLU激活+乘法+专家模型FP4量化接口函数
void silu_and_mul_scaled_fp4_experts_quant(
    torch::Tensor& output,                       // 输出量化后的FP4张量
    torch::Tensor& output_scale,                 // 输出缩放因子
    torch::Tensor const& input,                  // 输入张量
    torch::Tensor const& input_global_scale,     // 输入全局缩放因子
    torch::Tensor const& mask,                   // 掩码张量（用于门控）
    bool use_silu_and_mul                        // 是否使用SiLU激活和乘法
) {
#if defined ENABLE_NVFP4 && ENABLE_NVFP4
  // 如果支持NVFP4，调用SM100专用实现
  return silu_and_mul_scaled_fp4_experts_quant_sm100a(
      output, output_scale, input, input_global_scale, mask, use_silu_and_mul);
#endif
  // 如果不支持NVFP4，抛出未实现错误
  TORCH_CHECK_NOT_IMPLEMENTED(false, "No compiled nvfp4 experts quantization kernel");
}
```

## `nvfp4_quant.cuh` 实现

```c++
// 必要的头文件包含
#include <cuda.h>           // CUDA运行时API
#include <cuda_fp8.h>       // FP8数据类型支持
#include <cutlass/arch/config.h>  // CUTLASS架构配置

// 类型转换器：在单精度和双精度向量类型之间转换（适用于half和bfloat16）
template <typename T>
struct TypeConverter {
  using Type = half2;  // 默认转换为half2类型
};  // 保持通用性

// half2 -> half 的特化
template <>
struct TypeConverter<half2> {
  using Type = half;
};

// half -> half2 的特化
template <>
struct TypeConverter<half> {
  using Type = half2;
};

// __nv_bfloat162 -> __nv_bfloat16 的特化
template <>
struct TypeConverter<__nv_bfloat162> {
  using Type = __nv_bfloat16;
};

// __nv_bfloat16 -> __nv_bfloat162 的特化
template <>
struct TypeConverter<__nv_bfloat16> {
  using Type = __nv_bfloat162;
};

// 每个线程处理的元素数量
#define ELTS_PER_THREAD 8

// FP4转换相关常量
constexpr int CVT_FP4_ELTS_PER_THREAD = 8;  // 每个线程转换的FP4元素数量
constexpr int CVT_FP4_SF_VEC_SIZE = 16;     // FP4缩放因子向量大小

// 将8个float32值转换为8个e2m1值（表示为一个uint32_t）
// e2m1是FP4格式：2位指数，1位尾数
inline __device__ uint32_t fp32_vec_to_e2m1(float (&array)[8]) {
  // 此处使用的PTX指令需要sm100a/sm103a架构支持
#if CUTLASS_ARCH_MMA_SM100A_ENABLED || CUTLASS_ARCH_MMA_SM103A_ENABLED
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"                                    // 声明8位寄存器
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"     // 转换两个float32为e2m1x2格式
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"     // rn=round to nearest, satfinite=饱和有限值
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"         // 将4个字节打包为32位值
      "}"
      : "=r"(val)                                            // 输出：32位寄存器
      : "f"(array[0]), "f"(array[1]), "f"(array[2]), "f"(array[3]),  // 输入：8个float寄存器
        "f"(array[4]), "f"(array[5]), "f"(array[6]), "f"(array[7]));
  return val;
#else
  return 0;  // 不支持的架构返回0
#endif
}

// 将4个float2值转换为8个e2m1值（表示为一个uint32_t）
// 这是上面函数的float2向量版本
inline __device__ uint32_t fp32_vec_to_e2m1(float2 (&array)[4]) {
  // 此处使用的PTX指令需要sm100a/sm103a架构支持
#if CUTLASS_ARCH_MMA_SM100A_ENABLED || CUTLASS_ARCH_MMA_SM103A_ENABLED
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}"
      : "=r"(val)
      : "f"(array[0].x), "f"(array[0].y),  // 访问float2的x和y分量
        "f"(array[1].x), "f"(array[1].y),
        "f"(array[2].x), "f"(array[2].y),
        "f"(array[3].x), "f"(array[3].y));
  return val;
#else
  return 0;
#endif
}

// 快速倒数近似计算（flush-to-zero模式）
inline __device__ float reciprocal_approximate_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}

// 计算FP4量化中缩放因子的输出偏移地址
// SFType: 缩放因子类型，CVT_FP4_NUM_THREADS_PER_SF: 每个缩放因子的线程数
template <class SFType, int CVT_FP4_NUM_THREADS_PER_SF>
__device__ uint8_t* cvt_quant_to_fp4_get_sf_out_offset(int rowIdx, int colIdx, int numCols, SFType* SFout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)  // 仅在SM100+架构上支持
  static_assert(CVT_FP4_NUM_THREADS_PER_SF == 1 || CVT_FP4_NUM_THREADS_PER_SF == 2);

  // 一对线程将一个缩放因子写入全局内存
  // TODO: 通过共享内存暂存以支持打包的STG.32指令
  // 这是否比4个线程的STG.8更好？
  if (threadIdx.x % CVT_FP4_NUM_THREADS_PER_SF == 0) {
    // 缩放因子向量索引（K维度上16个元素共享一个缩放因子）
    int32_t kIdx = colIdx / CVT_FP4_NUM_THREADS_PER_SF;
    int32_t mIdx = rowIdx;

    // 缩放因子布局：[numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4(kTile)]
    // 对应索引：[mTileIdx, kTileIdx, outerMIdx, innerMIdx, innerKIdx]

    // 计算M维度的tile索引
    int32_t mTileIdx = mIdx / (32 * 4);
    // 缩放因子向量大小为16
    int factor = CVT_FP4_SF_VEC_SIZE * 4;
    int32_t numKTiles = (numCols + factor - 1) / factor;  // 向上取整
    int64_t mTileStride = numKTiles * 32 * 4 * 4;

    // 计算K维度的tile索引和步长
    int32_t kTileIdx = (kIdx / 4);
    int64_t kTileStride = 32 * 4 * 4;

    // M tile布局[32, 4]是列主序的
    int32_t outerMIdx = (mIdx % 32);
    int64_t outerMStride = 4 * 4;

    int32_t innerMIdx = (mIdx % (32 * 4)) / 32;
    int64_t innerMStride = 4;

    int32_t innerKIdx = (kIdx % 4);
    int64_t innerKStride = 1;

    // 计算全局偏移量
    int64_t SFOffset = mTileIdx * mTileStride + kTileIdx * kTileStride + outerMIdx * outerMStride +
                       innerMIdx * innerMStride + innerKIdx * innerKStride;

    return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
  }
#endif
  return nullptr;
}

// 定义16字节打包数据类型
template <class Type>
struct PackedVec {
  typename TypeConverter<Type>::Type elts[4];  // 使用类型转换器获取对应的向量类型
};

// FP8 e4m3格式的特化版本
template <>
struct PackedVec<__nv_fp8_e4m3> {
  __nv_fp8x2_e4m3 elts[8];  // 8个FP8x2元素，总共16个FP8值
};

```

## `nvfp4_quant_kernels.cu` 实现

```c++
// 将PackedVec量化为FP4格式并输出为uint32_t
// Type: 输入数据类型(half/bfloat16), UE8M0_SF: 是否使用UE8M0格式的缩放因子
template <class Type, bool UE8M0_SF = false>
__device__ uint32_t cvt_warp_fp16_to_fp4(PackedVec<Type>& vec, float SFScaleVal, uint8_t* SFout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)  // 仅在SM100+架构上支持
  // 获取本地8个值中的绝对最大值
  auto localMax = __habs2(vec.elts[0]);

  // 计算本地最大值（循环展开优化）
#pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    localMax = __hmax2(localMax, __habs2(vec.elts[i]));  // 计算half2的绝对值最大值
  }

  // 通过warp shuffle获取所有16个值中的绝对最大值（两个线程协作）
  localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
  // 获取最终的绝对最大值
  float vecMax = float(__hmax(localMax.x, localMax.y));

  // 计算缩放因子SF（向量最大值 / e2m1最大值）
  // e2m1格式的最大值 = 6.0
  // TODO: 使用half作为计算数据类型以提高性能
  float SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
  // 缩放因子的8位表示
  uint8_t fp8SFVal;
  
  // 根据模板参数选择缩放因子格式
  if constexpr (UE8M0_SF) {
    // 使用UE8M0格式（8位指数，0位尾数）
    __nv_fp8_e8m0 tmp;
    tmp.__x = __nv_cvt_float_to_e8m0(SFValue, __NV_SATFINITE, cudaRoundPosInf);
    SFValue = static_cast<float>(tmp);
    fp8SFVal = tmp.__x;
  } else {
    // 使用E4M3格式（4位指数，3位尾数）
    // 这里SFValue总是正数，所以E4M3等同于UE4M3
    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
    fp8SFVal = tmp.__x;
    SFValue = static_cast<float>(tmp);
  }
  
  // 计算输出缩放因子
  // 公式: final_scale = 1 / (fp32(fp8(SFValue * SFScaleVal)) / SFScaleVal)
  float outputScale =
      SFValue != 0 ? reciprocal_approximate_ftz(SFValue * reciprocal_approximate_ftz(SFScaleVal)) : 0.0f;

  // 如果提供了输出指针，将缩放因子写入全局内存（8位存储）
  if (SFout) {
    *SFout = fp8SFVal;
  }

  // 将输入数据转换为float2数组
  float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    // 根据输入类型进行转换
    if constexpr (std::is_same_v<Type, half>) {
      fp2Vals[i] = __half22float2(vec.elts[i]);        // half2 -> float2
    } else {
      fp2Vals[i] = __bfloat1622float2(vec.elts[i]);    // bfloat162 -> float2
    }
    // 应用输出缩放因子
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }

  // 转换为e2m1值（FP4格式）
  uint32_t e2m1Vec = fp32_vec_to_e2m1(fp2Vals);

  // 返回打包的e2m1值
  return e2m1Vec;
#else
  return 0;  // 不支持的架构返回0
#endif
}

// FP16/BF16到FP4转换的CUDA内核
// 默认使用UE4M3格式的缩放因子
template <class Type, bool UE8M0_SF = false>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__launch_bounds__(512, 4) cvt_fp16_to_fp4(  // SM100+架构上的启动边界优化
#else
cvt_fp16_to_fp4(
#endif
    int32_t numRows,        // 输入矩阵行数
    int32_t numCols,        // 输入矩阵列数
    Type const* in,         // 输入数据指针
    float const* SFScale,   // 全局缩放因子
    uint32_t* out,          // 输出FP4数据指针
    uint32_t* SFout         // 输出缩放因子指针
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using PackedVec = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF = (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD, "Vec size is not matched.");

  // 获取全局缩放因子，将应用于SF
  // 注意：SFScale与下一个GEMM的alpha相同，即(448.f / (Alpha_A / 6.f))
  float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];

  // 输入张量的行/列循环处理
  for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x) {
    for (int colIdx = threadIdx.x; colIdx < numCols / CVT_FP4_ELTS_PER_THREAD; colIdx += blockDim.x) {
      // 计算输入偏移量
      int64_t inOffset = rowIdx * (numCols / CVT_FP4_ELTS_PER_THREAD) + colIdx;
      PackedVec in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
      
      // 获取输出张量偏移量
      // 与inOffset相同，因为8个元素打包为一个uint32_t
      int64_t outOffset = inOffset;
      auto& out_pos = out[outOffset];

      // 计算缩放因子输出地址
      auto sf_out =
          cvt_quant_to_fp4_get_sf_out_offset<uint32_t, CVT_FP4_NUM_THREADS_PER_SF>(rowIdx, colIdx, numCols, SFout);

      // 执行FP16到FP4的转换
      out_pos = cvt_warp_fp16_to_fp4<Type, UE8M0_SF>(in_vec, SFScaleVal, sf_out);
    }
  }
#endif
}

// FP4量化的主机端调用函数模板
template <typename T>
void invokeFP4Quantization(
    int m,                      // 矩阵行数
    int n,                      // 矩阵列数
    T const* input,             // 输入数据指针
    float const* SFScale,       // 全局缩放因子
    int64_t* output,            // 输出FP4数据
    int32_t* SFOuput,           // 输出缩放因子
    bool useUE8M0,              // 是否使用UE8M0格式
    int multiProcessorCount,    // SM数量
    cudaStream_t stream         // CUDA流
) {
  // 网格和块大小配置
  // 每个线程转换8个值
  dim3 block(std::min(int(n / ELTS_PER_THREAD), 512));
  // 获取每个SM的块数（假设可以充分利用SM）
  int const numBlocksPerSM = 2048 / block.x;
  dim3 grid(std::min(int(m), multiProcessorCount * numBlocksPerSM));

  // 启动转换内核
  if (useUE8M0) {
    cvt_fp16_to_fp4<T, true><<<grid, block, 0, stream>>>(
        m, n, input, SFScale, reinterpret_cast<uint32_t*>(output), reinterpret_cast<uint32_t*>(SFOuput));
  } else {
    cvt_fp16_to_fp4<T, false><<<grid, block, 0, stream>>>(
        m, n, input, SFScale, reinterpret_cast<uint32_t*>(output), reinterpret_cast<uint32_t*>(SFOuput));
  }
}

// 显式实例化函数模板 - half类型
template void invokeFP4Quantization(
    int m,
    int n,
    half const* input,
    float const* SFScale,
    int64_t* output,
    int32_t* SFOuput,
    bool useUE8M0,
    int multiProcessorCount,
    cudaStream_t stream);

// 显式实例化函数模板 - bfloat16类型
template void invokeFP4Quantization(
    int m,
    int n,
    __nv_bfloat16 const* input,
    float const* SFScale,
    int64_t* output,
    int32_t* SFOuput,
    bool useUE8M0,
    int multiProcessorCount,
    cudaStream_t stream);

// 获取当前GPU的多处理器数量（使用静态缓存优化）
inline int getMultiProcessorCount() {
  static int multi_processor_count = []() {
    int device_id = 0;
    int count = 0;

    // 获取当前CUDA设备ID
    CHECK_CUDA_SUCCESS(cudaGetDevice(&device_id));

    // 获取当前设备的多处理器数量
    CHECK_CUDA_SUCCESS(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device_id));

    return count;  // 初始化静态变量
  }();

  return multi_processor_count;  // 后续调用返回缓存值
}

// SM100架构专用的FP4量化实现函数
void scaled_fp4_quant_sm100a(
    torch::Tensor& output,          // 输出FP4张量
    torch::Tensor const& input,     // 输入FP16/BF16张量
    torch::Tensor& output_sf,       // 输出缩放因子张量
    torch::Tensor const& input_sf   // 输入缩放因子张量
) {
  // 检查SM架构版本
  auto sm_version = getSMVersion();
  TORCH_CHECK(sm_version == 100 || sm_version == 103, "fp4_quant is only supported on sm100a/sm103a");

  // 获取输入张量维度
  int32_t m = input.size(0);  // 行数
  int32_t n = input.size(1);  // 列数

  // 检查列数必须是16的倍数（FP4打包要求）
  TORCH_CHECK(n % 16 == 0, "The N dimension must be multiple of 16.");

  // 获取GPU多处理器数量
  int multiProcessorCount = getMultiProcessorCount();

  // 获取数据指针
  auto input_sf_ptr = static_cast<float const*>(input_sf.data_ptr());
  auto sf_out = static_cast<int32_t*>(output_sf.data_ptr());
  auto output_ptr = static_cast<int64_t*>(output.data_ptr());
  
  // 设置CUDA设备保护和流
  at::cuda::CUDAGuard device_guard{(char)input.get_device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());

  // 目前不支持e8m0缩放因子格式
  bool useUE8M0 = false;

  // 根据输入数据类型分发到相应的量化函数
  switch (input.scalar_type()) {
    case torch::kHalf: {
      auto input_ptr = reinterpret_cast<half const*>(input.data_ptr());
      invokeFP4Quantization(m, n, input_ptr, input_sf_ptr, output_ptr, sf_out, useUE8M0, multiProcessorCount, stream);
      break;
    }
    case torch::kBFloat16: {
      auto input_ptr = reinterpret_cast<__nv_bfloat16 const*>(input.data_ptr());
      invokeFP4Quantization(m, n, input_ptr, input_sf_ptr, output_ptr, sf_out, useUE8M0, multiProcessorCount, stream);
      break;
    }
    default: {
      std::cerr << "Observing: " << input.scalar_type() << " for the input datatype which is invalid";
      throw std::runtime_error("Unsupported input data type for quantize_to_fp4.");
    }
  }
}
```

## `nvfp4_expert_quant.cu` 实现

```c++
// 专家模型专用的FP16到FP4量化函数（与基础版本相同的核心逻辑）
// Type: 输入数据类型(half/bfloat16), UE8M0_SF: 是否使用UE8M0格式的缩放因子
template <class Type, bool UE8M0_SF = false>
__device__ uint32_t cvt_warp_fp16_to_fp4(PackedVec<Type>& vec, float SFScaleVal, uint8_t* SFout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)  // 仅在SM100+架构上支持
  // 获取本地8个值中的绝对最大值
  auto localMax = __habs2(vec.elts[0]);

  // 计算本地最大值（循环展开优化）
#pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    localMax = __hmax2(localMax, __habs2(vec.elts[i]));  // 计算half2的绝对值最大值
  }

  // 通过warp shuffle获取所有16个值中的绝对最大值（两个线程协作）
  localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
  // 获取最终的绝对最大值
  float vecMax = float(__hmax(localMax.x, localMax.y));

  // 计算缩放因子SF（向量最大值 / e2m1最大值）
  // e2m1格式的最大值 = 6.0
  // TODO: 使用half作为计算数据类型以提高性能
  float SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
  // 缩放因子的8位表示
  uint8_t fp8SFVal;
  
  // 根据模板参数选择缩放因子格式
  if constexpr (UE8M0_SF) {
    // 从float32中提取8位指数位
    // float 32位 = 1位符号位 + 8位指数位 + 23位尾数位
    uint32_t tmp = reinterpret_cast<uint32_t&>(SFValue) >> 23;
    fp8SFVal = tmp & 0xff;
    // 转换回fp32格式
    reinterpret_cast<uint32_t&>(SFValue) = tmp << 23;
  } else {
    // 这里SFValue总是正数，所以E4M3等同于UE4M3
    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
    reinterpret_cast<__nv_fp8_e4m3&>(fp8SFVal) = tmp;
    // 转换回fp32格式
    SFValue = float(tmp);
  }
  
  // 计算输出缩放因子
  // 公式: final_scale = 1 / (fp32(fp8(SFValue * SFScaleVal)) / SFScaleVal)
  float outputScale =
      SFValue != 0 ? reciprocal_approximate_ftz(SFValue * reciprocal_approximate_ftz(SFScaleVal)) : 0.0f;

  // 如果提供了输出指针，将缩放因子写入全局内存（8位存储）
  if (SFout) {
    *SFout = fp8SFVal;
  }

  // 将输入数据转换为float2数组
  float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    // 根据输入类型进行转换
    if constexpr (std::is_same_v<Type, half>) {
      fp2Vals[i] = __half22float2(vec.elts[i]);        // half2 -> float2
    } else {
      fp2Vals[i] = __bfloat1622float2(vec.elts[i]);    // bfloat162 -> float2
    }
    // 应用输出缩放因子
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }

  // 转换为e2m1值（FP4格式）
  uint32_t e2m1Vec = fp32_vec_to_e2m1(fp2Vals);

  // 返回打包的e2m1值
  return e2m1Vec;
#else
  return 0;  // 不支持的架构返回0
#endif
}

// SiLU激活函数实现：silu(x) = x / (1 + exp(-x))
// 也称为Swish激活函数，在Transformer模型中广泛使用
__device__ __forceinline__ float silu(const float& val) {
  return val / (1.0f + __expf(-val));
}

// SiLU激活函数与乘法的融合操作
// 对于专家模型，通常需要进行门控操作：silu(x) * y
template <class Type>
inline __device__ void silu_and_mul(PackedVec<Type>& x_vec, const PackedVec<Type>& y_vec) {
  float2 x[CVT_FP4_ELTS_PER_THREAD / 2];  // 存储x向量的float2值
  float2 y[CVT_FP4_ELTS_PER_THREAD / 2];  // 存储y向量的float2值

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    // 根据输入类型进行转换和计算
    if constexpr (std::is_same_v<Type, half>) {
      x[i] = __half22float2(x_vec.elts[i]);     // half2 -> float2
      y[i] = __half22float2(y_vec.elts[i]);     // half2 -> float2
      x[i].x = silu(x[i].x) * y[i].x;           // 对x分量应用silu并乘以y
      x[i].y = silu(x[i].y) * y[i].y;           // 对y分量应用silu并乘以y
      x_vec.elts[i] = __float22half2_rn(x[i]);  // float2 -> half2 (round to nearest)
    } else {
      x[i] = __bfloat1622float2(x_vec.elts[i]);     // bfloat162 -> float2
      y[i] = __bfloat1622float2(y_vec.elts[i]);     // bfloat162 -> float2
      x[i].x = silu(x[i].x) * y[i].x;               // 对x分量应用silu并乘以y
      x[i].y = silu(x[i].y) * y[i].y;               // 对y分量应用silu并乘以y
      x_vec.elts[i] = __float22bfloat162_rn(x[i]);  // float2 -> bfloat162 (round to nearest)
    }
  }
}

// 专家模型FP4量化内核（支持动态专家查找和SiLU激活）
// 默认使用UE4M3格式的缩放因子
template <class Type, bool UE8M0_SF = false, bool SMALL_NUM_EXPERTS = false>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__launch_bounds__(512, 4) cvt_fp16_to_fp4(  // SM100+架构上的启动边界优化
#else
cvt_fp16_to_fp4(
#endif
    int32_t numRows,                        // 输入矩阵行数
    int32_t numCols,                        // 输入矩阵列数
    Type const* in,                         // 输入数据指针
    float const* SFScale,                   // 每个专家的缩放因子数组
    uint32_t* out,                          // 输出FP4数据指针
    uint32_t* SFout,                        // 输出缩放因子指针
    uint32_t* input_offset_by_experts,      // 每个专家在输入中的偏移量
    uint32_t* output_scale_offset_by_experts, // 每个专家在输出缩放因子中的偏移量
    int32_t* mask,                          // 掩码数组（用于早期退出）
    int n_experts,                          // 专家数量
    bool low_latency                        // 是否启用低延迟模式
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using PackedVec = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF = (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD, "Vec size is not matched.");

  // 计算线程和数据索引
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int colsPerRow = numCols / CVT_FP4_ELTS_PER_THREAD;
  
  // TODO(kaixih@nvidia): 目前假设mask与silu_and_mul一起使用
  // 未来可能需要更通用的mask行为。在silu情况下，输入的最后一维会翻倍
  bool use_mask = mask != nullptr;
  int actualColsPerRow = use_mask ? colsPerRow * 2 : colsPerRow;

  // 每个全局线程处理一个元素
  for (int globalIdx = tid; globalIdx < numRows * colsPerRow; globalIdx += gridDim.x * blockDim.x) {
    // 计算当前全局线程应该处理的行和列
    int rowIdx = globalIdx / colsPerRow;
    int colIdx = globalIdx % colsPerRow;

    // 根据专家数量使用不同策略查找专家内的索引
    int rowIdx_in_expert = 0;
    int expert_idx = 0;

    if constexpr (SMALL_NUM_EXPERTS) {
      // 小专家数量：线性搜索
      for (int i = 0; i < n_experts; i++) {
        uint32_t current_offset = __ldca(&input_offset_by_experts[i]);    // 缓存加载当前偏移
        uint32_t next_offset = __ldca(&input_offset_by_experts[i + 1]);   // 缓存加载下一个偏移
        if (rowIdx >= current_offset && rowIdx < next_offset) {
          rowIdx_in_expert = rowIdx - current_offset;
          expert_idx = i;
          break;
        }
      }
    } else {
      // 大专家数量：分块向量化加载优化
      // 本地数组大小设为17是因为寄存器限制
      uint32_t local_offsets[17];
      for (int chunk_start = 0; chunk_start < n_experts; chunk_start += 16) {
        // 使用int4向量化加载16个偏移量（每次加载4个uint32_t）
        *reinterpret_cast<int4*>(local_offsets) =
            __ldca(reinterpret_cast<const int4*>(&input_offset_by_experts[chunk_start]));
        *reinterpret_cast<int4*>(local_offsets + 4) =
            __ldca(reinterpret_cast<const int4*>(&input_offset_by_experts[chunk_start + 4]));
        *reinterpret_cast<int4*>(local_offsets + 8) =
            __ldca(reinterpret_cast<const int4*>(&input_offset_by_experts[chunk_start + 8]));
        *reinterpret_cast<int4*>(local_offsets + 12) =
            __ldca(reinterpret_cast<const int4*>(&input_offset_by_experts[chunk_start + 12]));
        local_offsets[16] = __ldca(&input_offset_by_experts[chunk_start + 16]);

        // 检查加载的16个偏移量
#pragma unroll
        for (int i = 0; i < 16; i++) {
          if (rowIdx >= local_offsets[i] && rowIdx < local_offsets[i + 1]) {
            rowIdx_in_expert = rowIdx - local_offsets[i];
            expert_idx = chunk_start + i;
            break;
          }
        }
      }
    }

    // 使用掩码时的早期退出
    if (use_mask && rowIdx_in_expert >= mask[expert_idx]) {
      continue;
    }

    // 计算输入偏移量并加载数据
    int64_t inOffset = rowIdx * actualColsPerRow + colIdx;
    PackedVec in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
    
    // 如果使用掩码，执行SiLU激活和乘法融合操作
    if (use_mask) {
      PackedVec in_vec_mul = reinterpret_cast<PackedVec const*>(in)[inOffset + colsPerRow];
      silu_and_mul(in_vec, in_vec_mul);
    }

    // 获取输出张量偏移量
    // 与inOffset相同，因为8个元素打包为一个uint32_t
    int64_t outOffset = rowIdx * colsPerRow + colIdx;
    auto& out_pos = out[outOffset];

    // 获取全局缩放因子，将应用于SF
    // 注意：SFScale与下一个GEMM的alpha相同，即(448.f / (Alpha_A / 6.f))
    float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[expert_idx];

    // 计算当前专家的缩放因子输出地址
    int factor = CVT_FP4_SF_VEC_SIZE * 4;
    // 实际的output_scales维度从填充的numCols计算得出
    int32_t numCols_padded = (numCols + factor - 1) / factor * factor;
    int numCols_SFout = numCols_padded / CVT_FP4_SF_VEC_SIZE / 4;
    uint32_t* SFout_in_expert = SFout + output_scale_offset_by_experts[expert_idx] * numCols_SFout;

    // 计算缩放因子输出地址
    auto sf_out = cvt_quant_to_fp4_get_sf_out_offset<uint32_t, CVT_FP4_NUM_THREADS_PER_SF>(
        rowIdx_in_expert, colIdx, numCols, SFout_in_expert);

    // 执行FP16到FP4的转换
    out_pos = cvt_warp_fp16_to_fp4<Type, UE8M0_SF>(in_vec, SFScaleVal, sf_out);
  }
#endif
}

// 专家专用FP4量化内核（静态线程到专家映射）
// 默认使用UE4M3格式的缩放因子
template <class Type, bool UE8M0_SF = false>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__launch_bounds__(512, 4) cvt_fp16_to_fp4_expert(  // SM100+架构上的启动边界优化
#else
cvt_fp16_to_fp4_expert(
#endif
    int32_t numRows,            // 输入矩阵行数
    int32_t numCols,            // 输入矩阵列数
    Type const* in,             // 输入数据指针
    float const* SFScale,       // 每个专家的缩放因子数组
    uint32_t* out,              // 输出FP4数据指针
    uint32_t* SFout,            // 输出缩放因子指针
    int32_t* mask,              // 掩码数组（用于早期退出）
    bool use_silu_and_mul,      // 是否使用SiLU激活和乘法
    int n_experts               // 专家数量
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using PackedVec = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF = (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD, "Vec size is not matched.");

  // 计算线程到专家的静态映射
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = (gridDim.x * blockDim.x) / n_experts;      // 每个专家分配的基础线程数
  int remainder = (gridDim.x * blockDim.x) % n_experts;   // 剩余线程数
  int expert_idx;         // 当前线程负责的专家索引
  int tid_in_expert;      // 线程在专家内的局部索引
  int actual_stride;      // 实际步长
  
  // 处理线程数不能被专家数整除的情况
  if (remainder > 0) {
    int bound = remainder * (stride + 1);  // 前remainder个专家多分配一个线程
    if (tid < bound) {
      // 前面的专家，每个分配(stride + 1)个线程
      expert_idx = tid / (stride + 1);
      tid_in_expert = tid % (stride + 1);
      actual_stride = stride + 1;
    } else {
      // 后面的专家，每个分配stride个线程
      expert_idx = remainder + (tid - bound) / stride;
      tid_in_expert = (tid - bound) % stride;
      actual_stride = stride;
    }
  } else {
    // 线程数能被专家数整除的情况
    expert_idx = tid / stride;
    tid_in_expert = tid % stride;
    actual_stride = stride;
  }
  
  // 计算每个专家的数据维度
  int m = numRows / n_experts;                    // 每个专家的行数
  int padded_m = (m + (128 - 1)) / 128 * 128;     // 填充到128的倍数

  int colsPerRow = numCols / CVT_FP4_ELTS_PER_THREAD;
  // TODO(kaixih@nvidia): 目前假设mask与silu_and_mul一起使用
  // 未来可能需要更通用的mask行为。在silu情况下，输入的最后一维会翻倍
  bool use_mask = mask != nullptr;
  int actualColsPerRow = use_silu_and_mul ? colsPerRow * 2 : colsPerRow;

  // 每个全局线程处理一个元素，只处理分配给当前专家的数据
  for (int globalIdx = tid_in_expert + expert_idx * m * colsPerRow; 
       globalIdx < (expert_idx + 1) * m * colsPerRow;
       globalIdx += actual_stride) {
    // 计算当前全局线程应该处理的行和列
    int rowIdx = globalIdx / colsPerRow;
    int colIdx = globalIdx % colsPerRow;

    // 计算专家内的行索引
    int rowIdx_in_expert = rowIdx - expert_idx * m;

    // 使用掩码时的早期退出
    if (use_mask && rowIdx_in_expert >= mask[expert_idx]) {
      break;  // 当前专家的有效数据已处理完毕
    }

    // 计算输入偏移量并加载数据
    int64_t inOffset = rowIdx * actualColsPerRow + colIdx;
    PackedVec in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
    
    // 如果使用SiLU激活和乘法，执行融合操作
    if (use_silu_and_mul) {
      PackedVec in_vec_mul = reinterpret_cast<PackedVec const*>(in)[inOffset + colsPerRow];
      silu_and_mul(in_vec, in_vec_mul);
    }

    // 获取输出张量偏移量
    // 与inOffset相同，因为8个元素打包为一个uint32_t
    int64_t outOffset = rowIdx * colsPerRow + colIdx;
    auto& out_pos = out[outOffset];

    // 获取全局缩放因子，将应用于SF
    // 注意：SFScale与下一个GEMM的alpha相同，即(448.f / (Alpha_A / 6.f))
    float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[expert_idx];

    // 计算当前专家的缩放因子输出地址
    int factor = CVT_FP4_SF_VEC_SIZE * 4;
    // 实际的output_scales维度从填充的numCols计算得出
    int32_t numCols_padded = (numCols + factor - 1) / factor * factor;
    int numCols_SFout = numCols_padded / CVT_FP4_SF_VEC_SIZE / 4;
    uint32_t* SFout_in_expert = SFout + expert_idx * padded_m * numCols_SFout;

    // 计算缩放因子输出地址
    auto sf_out = cvt_quant_to_fp4_get_sf_out_offset<uint32_t, CVT_FP4_NUM_THREADS_PER_SF>(
        rowIdx_in_expert, colIdx, numCols, SFout_in_expert);

    // 执行FP16到FP4的转换
    out_pos = cvt_warp_fp16_to_fp4<Type, UE8M0_SF>(in_vec, SFScaleVal, sf_out);
  }
#endif
}

// 大工作量优化版本的FP4量化内核（LARGE_M_TOPK = true）
// 使用共享内存和二分搜索优化大规模专家查找
template <class Type, bool UE8M0_SF = false, bool SMALL_NUM_EXPERTS = false>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__launch_bounds__(1024, 4) cvt_fp16_to_fp4(  // 更大的块大小以提高占用率
#else
cvt_fp16_to_fp4(
#endif
    int32_t numRows,                        // 输入矩阵行数
    int32_t numCols,                        // 输入矩阵列数
    Type const* in,                         // 输入数据指针
    float const* SFScale,                   // 每个专家的缩放因子数组
    uint32_t* out,                          // 输出FP4数据指针
    uint32_t* SFout,                        // 输出缩放因子指针
    uint32_t* input_offset_by_experts,      // 每个专家在输入中的偏移量
    uint32_t* output_scale_offset_by_experts, // 每个专家在输出缩放因子中的偏移量
    int32_t* mask,                          // 掩码数组（用于早期退出）
    int n_experts                           // 专家数量
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using PackedVec = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF = (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD, "Vec size is not matched.");
  extern __shared__ uint32_t shared_input_offsets[];  // 共享内存中的专家偏移量数组

  // 将输入偏移量加载到共享内存中以加速后续的专家查找
  // 如果专家数量大于4，使用向量化int4加载以节省指令数
  // 如果专家数量小于4，直接读取
  if constexpr (SMALL_NUM_EXPERTS) {
    // 小专家数量：使用标量加载
    for (int i = threadIdx.x; i < n_experts + 1; i += blockDim.x) {
      shared_input_offsets[i] = input_offset_by_experts[i];
    }
  } else {
    // 大专家数量：使用向量化加载（每次加载4个uint32_t）
    for (int i = threadIdx.x * 4; i < n_experts; i += blockDim.x * 4) {
      *reinterpret_cast<int4*>(&shared_input_offsets[i]) = 
          *reinterpret_cast<const int4*>(&input_offset_by_experts[i]);
    }
    // 线程0负责加载最后一个偏移量
    if (threadIdx.x == 0) {
      shared_input_offsets[n_experts] = input_offset_by_experts[n_experts];
    }
  }

  __syncthreads();  // 确保所有线程完成共享内存加载

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int colsPerRow = numCols / CVT_FP4_ELTS_PER_THREAD;
  bool use_mask = mask != nullptr;
  int actualColsPerRow = use_mask ? colsPerRow * 2 : colsPerRow;

  // 每个全局线程处理一个元素
  for (int globalIdx = tid; globalIdx < numRows * colsPerRow; globalIdx += gridDim.x * blockDim.x) {
    // 计算当前全局线程应该处理的行和列
    int rowIdx = globalIdx / colsPerRow;
    int colIdx = globalIdx % colsPerRow;

    // 使用二分搜索查找专家，在大m_topk情况下性能更好
    int rowIdx_in_expert = 0;
    int expert_idx = 0;

    // 通过共享内存进行二分搜索
    int left = 0, right = n_experts - 1;
    while (left <= right) {
      int mid = (left + right) / 2;
      // 获取偏移量：shared_input_offsets[i]对应input_offset_by_experts[i]
      uint32_t mid_offset = shared_input_offsets[mid];
      uint32_t next_offset = shared_input_offsets[mid + 1];

      if (rowIdx >= mid_offset && rowIdx < next_offset) {
        // 找到对应的专家
        rowIdx_in_expert = rowIdx - mid_offset;
        expert_idx = mid;
        break;
      } else if (rowIdx < mid_offset) {
        right = mid - 1;  // 在左半部分搜索
      } else {
        left = mid + 1;   // 在右半部分搜索
      }
    }

    // 使用掩码时的早期退出
    if (use_mask && rowIdx_in_expert >= mask[expert_idx]) {
      continue;
    }

    // 计算输入偏移量并加载数据
    int64_t inOffset = rowIdx * actualColsPerRow + colIdx;
    PackedVec in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
    
    // 如果使用掩码，执行SiLU激活和乘法融合操作
    if (use_mask) {
      PackedVec in_vec_mul = reinterpret_cast<PackedVec const*>(in)[inOffset + colsPerRow];
      silu_and_mul(in_vec, in_vec_mul);
    }

    // 获取输出张量偏移量
    int64_t outOffset = rowIdx * colsPerRow + colIdx;
    auto& out_pos = out[outOffset];

    // 获取全局缩放因子
    float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[expert_idx];

    // 计算当前专家的缩放因子输出地址
    int factor = CVT_FP4_SF_VEC_SIZE * 4;
    int32_t numCols_padded = (numCols + factor - 1) / factor * factor;
    int numCols_SFout = numCols_padded / CVT_FP4_SF_VEC_SIZE / 4;
    uint32_t* SFout_in_expert = SFout + output_scale_offset_by_experts[expert_idx] * numCols_SFout;

    // 计算缩放因子输出地址
    auto sf_out = cvt_quant_to_fp4_get_sf_out_offset<uint32_t, CVT_FP4_NUM_THREADS_PER_SF>(
        rowIdx_in_expert, colIdx, numCols, SFout_in_expert);

    // 执行FP16到FP4的转换
    out_pos = cvt_warp_fp16_to_fp4<Type, UE8M0_SF>(in_vec, SFScaleVal, sf_out);
  }
#endif
}

// 专家模型FP4量化的通用实现函数
// 根据不同的参数配置选择最优的内核启动策略
template <typename T>
void quant_impl(
    void* output,                       // 输出FP4数据
    void* output_scale,                 // 输出缩放因子
    void* input,                        // 输入数据
    void* input_global_scale,           // 输入全局缩放因子
    void* input_offset_by_experts,      // 专家输入偏移量
    void* output_scale_offset_by_experts, // 专家输出缩放因子偏移量
    void* mask,                         // 掩码数组
    bool use_silu_and_mul,              // 是否使用SiLU激活和乘法
    int m_topk,                         // 输入行数（top-k选择的行数）
    int k,                              // 输入列数
    int n_experts,                      // 专家数量
    cudaStream_t stream                 // CUDA流
) {
  // TODO: multiProcessorCount应该被缓存以避免重复查询
  int device;
  cudaGetDevice(&device);
  int multiProcessorCount;
  cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, device);

  // 网格和块大小配置
  // 每个线程转换8个值
  int const workSizePerRow = k / ELTS_PER_THREAD;
  int const totalWorkSize = m_topk * workSizePerRow;
  dim3 block(std::min(workSizePerRow, 512));
  
  // 获取每个SM的块数（假设可以充分利用SM）
  int const numBlocksPerSM = 2048 / block.x;
  dim3 grid(std::min(static_cast<int>((totalWorkSize + block.x - 1) / block.x), multiProcessorCount * numBlocksPerSM));
  
  // 动态调整网格和块大小以优化占用率
  while (grid.x <= multiProcessorCount && block.x > 64) {
    grid.x *= 2;
    block.x = (block.x + 1) / 2;
  }

  // TODO(kaixih@nvidia): 应该放宽限制以允许任意网格大小
  // 如果使用掩码，使用专门的专家内核
  if (mask != nullptr) {
    grid.x = (grid.x + n_experts - 1) / n_experts * n_experts;  // 确保网格大小是专家数的倍数
    cvt_fp16_to_fp4_expert<T, false><<<grid, block, 0, stream>>>(
        m_topk,
        k,
        reinterpret_cast<T*>(input),
        reinterpret_cast<float*>(input_global_scale),
        reinterpret_cast<uint32_t*>(output),
        reinterpret_cast<uint32_t*>(output_scale),
        reinterpret_cast<int32_t*>(mask),
        use_silu_and_mul,
        n_experts);
    return;
  }

  // 计算每个块需要重复执行的次数
  int const blockRepeat = (totalWorkSize + block.x * grid.x - 1) / (block.x * grid.x);
  
  if (blockRepeat > 1) {
    // 大工作量：使用共享内存优化的内核
    size_t shared_mem_size = (n_experts + 1) * sizeof(uint32_t);
    if (n_experts >= 4) {
      // 大专家数量：使用向量化加载
      cvt_fp16_to_fp4<T, false, false><<<grid, block, shared_mem_size, stream>>>(
          m_topk, k,
          reinterpret_cast<T*>(input),
          reinterpret_cast<float*>(input_global_scale),
          reinterpret_cast<uint32_t*>(output),
          reinterpret_cast<uint32_t*>(output_scale),
          reinterpret_cast<uint32_t*>(input_offset_by_experts),
          reinterpret_cast<uint32_t*>(output_scale_offset_by_experts),
          reinterpret_cast<int32_t*>(mask),
          n_experts);
    } else {
      // 小专家数量：使用标量加载
      cvt_fp16_to_fp4<T, false, true><<<grid, block, shared_mem_size, stream>>>(
          m_topk, k,
          reinterpret_cast<T*>(input),
          reinterpret_cast<float*>(input_global_scale),
          reinterpret_cast<uint32_t*>(output),
          reinterpret_cast<uint32_t*>(output_scale),
          reinterpret_cast<uint32_t*>(input_offset_by_experts),
          reinterpret_cast<uint32_t*>(output_scale_offset_by_experts),
          reinterpret_cast<int32_t*>(mask),
          n_experts);
    }
  } else {
    // 小工作量：使用低延迟优化的内核（无共享内存）
    if (n_experts >= 16) {
      // 大专家数量：使用寄存器优化
      cvt_fp16_to_fp4<T, false, false><<<grid, block, 0, stream>>>(
          m_topk, k,
          reinterpret_cast<T*>(input),
          reinterpret_cast<float*>(input_global_scale),
          reinterpret_cast<uint32_t*>(output),
          reinterpret_cast<uint32_t*>(output_scale),
          reinterpret_cast<uint32_t*>(input_offset_by_experts),
          reinterpret_cast<uint32_t*>(output_scale_offset_by_experts),
          reinterpret_cast<int32_t*>(mask),
          n_experts,
          /* bool low_latency */ true);
    } else {
      // 小专家数量：使用线性搜索
      cvt_fp16_to_fp4<T, false, true><<<grid, block, 0, stream>>>(
          m_topk, k,
          reinterpret_cast<T*>(input),
          reinterpret_cast<float*>(input_global_scale),
          reinterpret_cast<uint32_t*>(output),
          reinterpret_cast<uint32_t*>(output_scale),
          reinterpret_cast<uint32_t*>(input_offset_by_experts),
          reinterpret_cast<uint32_t*>(output_scale_offset_by_experts),
          reinterpret_cast<int32_t*>(mask),
          n_experts,
          /* bool low_latency */ true);
    }
  }
}

// Avoid redefinition warnings
#undef CHECK_CONTIGUOUS
#undef CHECK_TH_CUDA
#undef CHECK_INPUT

/*Quantization entry for fp4 experts quantization*/
#define CHECK_TH_CUDA(x, m) TORCH_CHECK(x.is_cuda(), m, "must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x, m) TORCH_CHECK(x.is_contiguous(), m, "must be contiguous")
#define CHECK_INPUT(x, m) \
  CHECK_TH_CUDA(x, m);    \
  CHECK_CONTIGUOUS(x, m);

// constexpr auto FP8 = at::ScalarType::Float8_e4m3fn;
constexpr auto HALF = at::ScalarType::Half;
constexpr auto BF16 = at::ScalarType::BFloat16;
constexpr auto FLOAT = at::ScalarType::Float;
constexpr auto INT = at::ScalarType::Int;
constexpr auto UINT8 = at::ScalarType::Byte;

// SM100架构专用的专家模型FP4量化PyTorch接口函数
// 用于标准的专家模型量化（不包含SiLU激活）
void scaled_fp4_experts_quant_sm100a(
    torch::Tensor& output,                          // 输出FP4张量 [m_topk, k/2]
    torch::Tensor& output_scale,                    // 输出缩放因子张量
    torch::Tensor const& input,                     // 输入FP16/BF16张量 [m_topk, k]
    torch::Tensor const& input_global_scale,        // 输入全局缩放因子 [n_experts]
    torch::Tensor const& input_offset_by_experts,   // 专家输入偏移量 [n_experts+1]
    torch::Tensor const& output_scale_offset_by_experts // 专家输出缩放因子偏移量 [n_experts+1]
) {
  // 检查SM架构版本
  auto sm_version = getSMVersion();
  TORCH_CHECK(sm_version == 100 || sm_version == 103, "fp4_quant is only supported on sm100a/sm103a");

  // 验证所有输入张量的基本属性
  CHECK_INPUT(output, "output must be a CUDA tensor");
  CHECK_INPUT(output_scale, "output_scale must be a CUDA tensor");
  CHECK_INPUT(input, "input must be a CUDA tensor");
  CHECK_INPUT(input_global_scale, "input_global_scale must be a CUDA tensor");
  CHECK_INPUT(input_offset_by_experts, "input_offset_by_experts must be a CUDA tensor");
  CHECK_INPUT(output_scale_offset_by_experts, "output_scale_offset_by_experts must be a CUDA tensor");

  // 验证张量维度
  TORCH_CHECK(output.dim() == 2);
  TORCH_CHECK(output_scale.dim() == 2);
  TORCH_CHECK(input.dim() == 2);
  TORCH_CHECK(input_global_scale.dim() == 1);
  TORCH_CHECK(input_offset_by_experts.dim() == 1);
  TORCH_CHECK(output_scale_offset_by_experts.dim() == 1);

  // 验证张量数据类型
  TORCH_CHECK(input.scalar_type() == HALF || input.scalar_type() == BF16);
  TORCH_CHECK(input_global_scale.scalar_type() == FLOAT);
  TORCH_CHECK(input_offset_by_experts.scalar_type() == INT);
  TORCH_CHECK(output_scale_offset_by_experts.scalar_type() == INT);
  // output是uint8（两个nvfp4值打包为一个uint8）
  // output_scale是int32（四个fp8值打包为一个int32）
  TORCH_CHECK(output.scalar_type() == UINT8);
  TORCH_CHECK(output_scale.scalar_type() == INT);

  // 验证张量形状和大小约束
  const int BLOCK_SIZE = 16;  // FP4量化的块大小
  auto m_topk = input.size(0);
  auto k = input.size(1);
  TORCH_CHECK(k % BLOCK_SIZE == 0, "k must be a multiple of 16");
  auto n_experts = input_global_scale.size(0);
  TORCH_CHECK(input_offset_by_experts.size(0) == n_experts + 1);
  TORCH_CHECK(output_scale_offset_by_experts.size(0) == n_experts + 1);
  TORCH_CHECK(output.size(0) == m_topk);
  TORCH_CHECK(output.size(1) == k / 2);  // FP4占用一半存储空间
  
  // 验证缩放因子张量的大小
  int scales_k = k / BLOCK_SIZE;
  // 4表示nvidia nvfp4的swizzle要求
  int padded_k = (scales_k + (4 - 1)) / 4 * 4;
  // 4表示4个fp8值打包为一个int32
  TORCH_CHECK(output_scale.size(1) * 4 == padded_k);

  // 设置CUDA设备和流
  auto in_dtype = input.dtype();
  at::cuda::CUDAGuard device_guard{(char)input.get_device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  
  // 根据输入数据类型分发到相应的实现
  if (in_dtype == at::ScalarType::Half) {
    quant_impl<half>(
        output.data_ptr(),
        output_scale.data_ptr(),
        input.data_ptr(),
        input_global_scale.data_ptr(),
        input_offset_by_experts.data_ptr(),
        output_scale_offset_by_experts.data_ptr(),
        nullptr,  // mask（无掩码）
        false,    // use_silu_and_mul（不使用SiLU激活）
        m_topk, k, n_experts, stream);
  } else if (in_dtype == at::ScalarType::BFloat16) {
    quant_impl<__nv_bfloat16>(
        output.data_ptr(),
        output_scale.data_ptr(),
        input.data_ptr(),
        input_global_scale.data_ptr(),
        input_offset_by_experts.data_ptr(),
        output_scale_offset_by_experts.data_ptr(),
        nullptr,  // mask（无掩码）
        false,    // use_silu_and_mul（不使用SiLU激活）
        m_topk, k, n_experts, stream);
  } else {
    TORCH_CHECK(false, "Expected input data type to be half or bfloat16");
  }
}

// SM100架构专用的SiLU激活+乘法+专家模型FP4量化PyTorch接口函数
// 用于包含SiLU激活函数的专家模型量化
void silu_and_mul_scaled_fp4_experts_quant_sm100a(
    torch::Tensor& output,                      // 输出FP4张量 [m_topk, k/2]
    torch::Tensor& output_scale,                // 输出缩放因子张量
    torch::Tensor const& input,                 // 输入FP16/BF16张量 [m_topk, k*2 or k]
    torch::Tensor const& input_global_scale,    // 输入全局缩放因子 [n_experts]
    torch::Tensor const& mask,                  // 掩码张量 [n_experts]
    bool use_silu_and_mul                       // 是否使用SiLU激活和乘法
) {
  // 检查SM架构版本
  auto sm_version = getSMVersion();
  TORCH_CHECK(sm_version == 100 || sm_version == 103, "fp4_quant is only supported on sm100a/sm103a");

  // 验证所有输入张量的基本属性
  CHECK_INPUT(output, "output must be a CUDA tensor");
  CHECK_INPUT(output_scale, "output_scale must be a CUDA tensor");
  CHECK_INPUT(input, "input must be a CUDA tensor");
  CHECK_INPUT(input_global_scale, "input_global_scale must be a CUDA tensor");
  CHECK_INPUT(mask, "mask must be a CUDA tensor");

  // 验证张量维度
  TORCH_CHECK(output.dim() == 2);
  TORCH_CHECK(output_scale.dim() == 2);
  TORCH_CHECK(input.dim() == 2);
  TORCH_CHECK(input_global_scale.dim() == 1);

  // 验证张量数据类型
  TORCH_CHECK(input.scalar_type() == HALF || input.scalar_type() == BF16);
  TORCH_CHECK(input_global_scale.scalar_type() == FLOAT);
  TORCH_CHECK(mask.scalar_type() == INT);
  // output是uint8（两个nvfp4值打包为一个uint8）
  // output_scale是int32（四个fp8值打包为一个int32）
  TORCH_CHECK(output.scalar_type() == UINT8);
  TORCH_CHECK(output_scale.scalar_type() == INT);

  // 验证张量形状和大小约束
  const int BLOCK_SIZE = 16;  // FP4量化的块大小
  auto m_topk = input.size(0);
  auto k_by_2 = input.size(1);
  auto k = k_by_2;
  
  // 如果使用SiLU激活和乘法，输入维度会翻倍（包含门控向量）
  if (use_silu_and_mul) {
    TORCH_CHECK(k_by_2 % 2 == 0, "k must be a multiple of 2");
    k = k_by_2 / 2;
  }
  
  auto n_experts = input_global_scale.size(0);
  TORCH_CHECK(mask.size(0) == n_experts);
  TORCH_CHECK(output.size(0) == m_topk);
  TORCH_CHECK(output.size(1) == k / 2);  // FP4占用一半存储空间
  
  // 验证缩放因子张量的大小
  int scales_k = k / BLOCK_SIZE;
  // 4表示nvidia nvfp4的swizzle要求
  int padded_k = (scales_k + (4 - 1)) / 4 * 4;
  // 4表示4个fp8值打包为一个int32
  TORCH_CHECK(output_scale.size(1) * 4 == padded_k);

  // 设置CUDA设备和流
  auto in_dtype = input.dtype();
  at::cuda::CUDAGuard device_guard{(char)input.get_device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  
  // 根据输入数据类型分发到相应的实现
  if (in_dtype == at::ScalarType::Half) {
    quant_impl<half>(
        output.data_ptr(),
        output_scale.data_ptr(),
        input.data_ptr(),
        input_global_scale.data_ptr(),
        nullptr,  // input_offset_by_experts（无专家偏移）
        nullptr,  // output_scale_offset_by_experts（无专家偏移）
        mask.data_ptr(),
        use_silu_and_mul,
        m_topk, k, n_experts, stream);
  } else if (in_dtype == at::ScalarType::BFloat16) {
    quant_impl<__nv_bfloat16>(
        output.data_ptr(),
        output_scale.data_ptr(),
        input.data_ptr(),
        input_global_scale.data_ptr(),
        nullptr,  // input_offset_by_experts（无专家偏移）
        nullptr,  // output_scale_offset_by_experts（无专家偏移）
        mask.data_ptr(),
        use_silu_and_mul,
        m_topk, k, n_experts, stream);
  } else {
    TORCH_CHECK(false, "Expected input data type to be half or bfloat16");
  }
}
```