# 0x0. 前言

最近在review SGLang的PR #14570时,发现这个PR对JIT kernel系统做了不少改进,主要是增强了错误报告和开发体验。趁着切换到这个分支,把整个JIT系统的实现细节梳理一遍,包括所有核心文件的代码解析。

JIT (Just-In-Time) kernel系统是SGLang里一个运行时编译CUDA kernel的框架。和传统的AOT编译不同,JIT可以根据运行时参数动态生成优化的kernel。这个系统用了不少现代C++特性,包括concepts、ranges、source_location等,代码质量还不错。

相关PR: https://github.com/sgl-project/sglang/pull/14570

# 0x1. 整体架构

## 0x1.1 目录结构

```
python/sglang/jit_kernel/
├── include/sgl_kernel/          # C++头文件
│   ├── utils.h                  # 主机端工具
│   ├── utils.cuh                # 设备端工具
│   └── tensor.h                 # Tensor验证
├── csrc/                        # CUDA实现
│   ├── hicache.cuh              # KV cache迁移
│   ├── cuda_wait_value.cuh      # Stream同步
│   └── test_utils.h             # 测试工具
├── utils.py                     # Python JIT加载
├── hicache.py                   # HiCache接口
├── cuda_wait_value.py           # Wait Value接口
└── __main__.py                  # clangd配置生成
```

## 0x1.2 编译流程

Python调用`load_jit()` → 生成C++包装代码 → `tvm_ffi.load_inline()` → nvcc编译 → 加载so → 返回Module对象

整个流程是运行时完成的,所以第一次调用会比较慢,但后续调用会从缓存读取。

# 0x2. 核心工具实现

## 0x2.1 utils.h - 主机端工具

这个文件提供了CPU端的核心工具,最重要的是错误处理和调试信息捕获。

### DebugInfo - 自动捕获调用位置

```cpp
struct DebugInfo : public std::source_location {
  DebugInfo(std::source_location loc = std::source_location::current()) 
    : std::source_location(loc) {}
};
```

继承自`std::source_location`,利用默认参数`= {}`自动填充当前位置。这样在调用`RuntimeCheck`时不需要手动传位置信息。

不过这里有个坑:NVCC 12.1对C++20的`source_location`支持有问题,需要一些workaround:

```cpp
#ifdef __CUDACC__
#pragma push_macro("__cpp_consteval")
#define __cpp_consteval 201811L
#include <source_location>
#undef consteval
#pragma pop_macro("__cpp_consteval")
#endif
```

通过临时修改编译器宏,让NVCC能正确处理`source_location`。这个技巧来自NVIDIA论坛的讨论。

### RuntimeCheck - 运行时断言

```cpp
template <typename... Args>
struct RuntimeCheck {
  template <typename Cond>
  explicit RuntimeCheck(Cond&& condition, Args&&... args, DebugInfo location = {}) {
    if (condition) return;
    [[unlikely]] ::host::panic(location, std::forward<Args>(args)...);
  }
};
```

支持两种调用方式:
```cpp
RuntimeCheck(x > 0, "x must be positive, got ", x);
RuntimeCheck(location, x > 0, "x must be positive");
```

`[[unlikely]]`提示编译器优化正常路径,错误分支不太可能执行。

### 其他工具函数

```cpp
// 类Python的range
template <std::integral T>
inline auto irange(T end) {
  return stdv::iota(static_cast<T>(0), end);
}

// 向上取整除法
template <std::signed_integral T, std::signed_integral U>
inline constexpr auto div_ceil(T a, U b) {
  return (a + b - 1) / b;
}

// 安全的指针偏移(只允许void*)
namespace pointer {
  template <typename T, std::integral... U>
  inline auto offset(T* ptr, U... offset) -> void* {
    static_assert(std::is_same_v<T, void>);
    return static_cast<char*>(ptr) + (... + offset);
  }
}
```

## 0x2.2 utils.cuh - 设备端工具

### RuntimeDeviceCheck - CUDA错误检查

```cpp
inline void RuntimeDeviceCheck(::cudaError_t error, DebugInfo location = {}) {
  if (error != ::cudaSuccess) {
    [[unlikely]];
    ::host::panic(location, "CUDA error: ", ::cudaGetErrorString(error));
  }
}
```

用法:
```cpp
RuntimeDeviceCheck(cudaMalloc(&ptr, size));
kernel<<<grid, block>>>();
RuntimeDeviceCheck();  // 检查kernel启动
```

### LaunchKernel - Kernel启动器

```cpp
struct LaunchKernel {
  explicit LaunchKernel(
      dim3 grid_dim, dim3 block_dim, DLDevice device,
      std::size_t dynamic_shared_mem_bytes = 0,
      DebugInfo location = {}) noexcept;
  
  static auto resolve_device(DLDevice device) -> cudaStream_t;
  
  template <typename T, typename... Args>
  auto operator()(T&& kernel, Args&&... args) const -> void {
    RuntimeDeviceCheck(
      ::cudaLaunchKernelEx(&m_config, kernel, std::forward<Args>(args)...), 
      m_location
    );
  }
};
```

核心功能:
1. 从DLDevice解析出CUDA stream
2. 记录kernel启动位置,方便调试
3. 统一的错误处理

用法:
```cpp
DLDevice device = {.device_type = kDLCUDA, .device_id = 0};
LaunchKernel(grid, block, device)(my_kernel, arg1, arg2);
```

使用`cudaLaunchKernelEx`而不是`<<<>>>`语法,这样可以更好地处理错误。

## 0x2.3 tensor.h - Tensor验证系统

这是整个JIT系统最核心的部分,提供了强大的tensor验证和类型推导。

### 类型特征

```cpp
template <std::integral T>
struct dtype_trait<T> {
  inline static constexpr DLDataType value = {
    .code = std::is_signed_v<T> ? kDLInt : kDLUInt,
    .bits = static_cast<uint8_t>(sizeof(T) * 8),
    .lanes = 1
  };
};

#ifdef __CUDACC__
template <>
struct dtype_trait<__half> {
  inline static constexpr DLDataType value = {
    .code = kDLFloat, .bits = 16, .lanes = 1
  };
};
#endif
```

编译期计算DLDataType,支持CUDA的半精度类型。

### 符号变量 - 核心设计

符号变量用于在多个tensor间共享和验证维度信息。

#### SymbolicSize

```cpp
struct SymbolicSize {
  auto verify(int64_t value, const char* prefix, int64_t dim) -> void {
    if (this->has_value()) {
      if (m_value != value) {
        [[unlikely]];
        Panic("Size mismatch for ", m_name_str(prefix, dim), 
              ": expected ", m_value, " but got ", value);
      }
    } else {
      this->set_value(value);
    }
  }
  
private:
  auto m_name_str(const char* prefix, int64_t dim) const -> std::string {
    std::ostringstream os;
    os << prefix << '#' << dim;
    if (!m_annotation.empty()) os << "('" << m_annotation << "')";
    return std::move(os).str();
  }
  
  int64_t m_value;
  std::string_view m_annotation;
};
```

工作原理:
- 首次验证时记录值
- 后续验证时检查是否匹配
- 错误信息包含维度索引和注释

举个例子:
```cpp
auto batch_size = SymbolicSize{"batch_size"};
auto seq_len = SymbolicSize{"seq_len"};

// tensor1: [32, 128]
TensorMatcher({batch_size, seq_len}).verify(tensor1);
// batch_size = 32, seq_len = 128

// tensor2: [32, 256] - 会报错
TensorMatcher({batch_size, seq_len}).verify(tensor2);
// 错误: Size mismatch for shape#1('seq_len'): expected 128 but got 256
```

这个设计很巧妙,可以在多个tensor间共享维度约束。

### TensorMatcher - 流式验证API

```cpp
struct TensorMatcher {
  explicit TensorMatcher(std::initializer_list<SizeRef> shape);
  
  auto with_strides(std::initializer_list<SizeRef> strides) && -> TensorMatcher&&;
  
  template <typename... Ts>
  auto with_dtype() && -> TensorMatcher&&;
  
  template <DLDeviceType... Codes>
  auto with_device() && -> TensorMatcher&&;
  
  auto verify(tvm::ffi::TensorView view, DebugInfo info = {}) const&& 
    -> const TensorMatcher&&;
};
```

使用右值引用`&&`实现链式调用,这样可以写出很流畅的验证代码:

```cpp
auto num_blocks = SymbolicSize{"num_blocks"};
auto head_dim = SymbolicSize{"head_dim"};
auto kv_stride = SymbolicSize{"kv_stride"};
auto cache_dtype = SymbolicDType{};

TensorMatcher({num_blocks, head_dim})  //
    .with_strides({kv_stride, 1})
    .with_dtype<float, __half>()
    .with_device<kDLCUDA>()
    .verify(k_cache)
    .verify(v_cache);
```

关键特性:
- `-1`表示任意大小
- 省略`with_strides`表示要求连续tensor
- 模板参数限制允许的类型
- 可以连续验证多个tensor

PR #14570的一个重要改进是错误信息:

**改进前**:
```
Size mismatch: expected 128 but got 256
```

**改进后**:
```
Tensor match failed for Tensor<100, 256>[strides=<256, 1>, dtype=float32, device=cuda:0] 
at hicache.cuh:255
- Root cause: Size mismatch for shape#1('head_dim'): expected 128 but got 256
```

现在会打印实际tensor的完整信息,而不只是期望的模式,调试起来方便多了。

还有个优化是跳过size=1维度的stride检查:
```cpp
for (const auto i : irange(dim)) {
  if (view.size(i) != 1 || !m_strides[i]->has_value()) {
    m_strides[i]->verify(view.stride(i), "stride", i);
  }
}
```

因为size=1时stride可以是任意值,检查没意义。

## 0x2.4 utils.py - Python JIT加载

### load_jit - 核心函数

```python
def load_jit(
    *args: str,
    cpp_files: List[str] | None = None,
    cuda_files: List[str] | None = None,
    cpp_wrappers: List[Tuple[str, str]] | None = None,
    cuda_wrappers: List[Tuple[str, str]] | None = None,
    extra_cflags: List[str] | None = None,
    extra_cuda_cflags: List[str] | None = None,
    extra_ldflags: List[str] | None = None,
    extra_include_paths: List[str] | None = None,
    build_directory: str | None = None,
) -> Module:
```

工作流程:

1. 生成包装代码:
```python
def _make_wrapper(tup: Tuple[str, str]) -> str:
    export_name, kernel_name = tup
    return f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({export_name}, ({kernel_name}));"
```

2. 构建include语句:
```python
cpp_paths = [(KERNEL_PATH / "csrc" / f).resolve() for f in cpp_files]
cpp_sources = [f'#include "{path}"' for path in cpp_paths]
cpp_sources += [_make_wrapper(tup) for tup in cpp_wrappers]
```

3. 调用tvm_ffi编译:
```python
return load_inline(
    "sgl_kernel_jit_" + "_".join(str(arg) for arg in args),
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    extra_cflags=DEFAULT_CFLAGS + extra_cflags,
    extra_cuda_cflags=DEFAULT_CUDA_CFLAGS + extra_cuda_cflags,
    ...
)
```

默认编译选项:
```python
DEFAULT_CFLAGS = ["-std=c++20", "-O3"]
DEFAULT_CUDA_CFLAGS = ["-std=c++20", "-O3", "--expt-relaxed-constexpr"]
```

### make_cpp_args - 参数转换

```python
def make_cpp_args(*args: CPP_TEMPLATE_TYPE) -> CPPArgList:
    def _convert(arg: CPP_TEMPLATE_TYPE) -> str:
        if isinstance(arg, bool):
            return "true" if arg else "false"
        if isinstance(arg, (int, float)):
            return str(arg)
        raise TypeError(f"Unsupported argument type: {type(arg)}")
    
    return CPPArgList(_convert(arg) for arg in args)
```

将Python值转换为C++模板参数字符串,比如:
```python
args = make_cpp_args(128, 4, True)
# 结果: "128, 4, true"
```

# 0x3. 实战案例:HiCache

HiCache是一个KV cache迁移kernel,用于在不同内存位置间传输attention的key-value cache。这个实现展示了如何把前面的工具组合起来。

## 0x3.1 Warp级向量化内存操作

```cpp
namespace device::warp {

namespace details {
  // 非缓存加载(使用.cs修饰符)
  __always_inline __device__ auto load_nc(const uint4* __restrict__ src) -> uint4 {
    uint32_t tmp0, tmp1, tmp2, tmp3;
    asm volatile(
      "ld.global.cs.v4.b32 {%0,%1,%2,%3},[%4];" 
      : "=r"(tmp0), "=r"(tmp1), "=r"(tmp2), "=r"(tmp3) 
      : "l"(src)
    );
    return uint4{tmp0, tmp1, tmp2, tmp3};
  }
  
  __always_inline __device__ void store_nc(uint4* __restrict__ dst, const uint4& value) {
    uint32_t tmp0 = value.x, tmp1 = value.y, tmp2 = value.z, tmp3 = value.w;
    asm volatile(
      "st.global.cs.v4.b32 [%0],{%1,%2,%3,%4};" 
      :: "l"(dst), "r"(tmp0), "r"(tmp1), "r"(tmp2), "r"(tmp3)
    );
  }
}

template <std::size_t kBytes, std::size_t kUnit, std::size_t kThreads>
__always_inline __device__ auto load_vec(const void* __restrict__ src) {
  using Package = details::mem_package_t<kBytes, kUnit>;
  constexpr auto kBytesPerLoop = sizeof(Package) * kThreads;
  constexpr auto kLoopCount = kBytes / kBytesPerLoop;
  
  const auto src_packed = static_cast<const Package*>(src);
  const auto lane_id = threadIdx.x % kThreads;
  device_vec<Package, kLoopCount> vec;
  
  #pragma unroll kLoopCount
  for (std::size_t i = 0; i < kLoopCount; ++i) {
    const auto j = i * kThreads + lane_id;
    vec.data[i] = details::load_nc(src_packed + j);
  }
  
  return vec;
}

}  // namespace device::warp
```

技术要点:

1. **非缓存访问(.cs修饰符)**:
   - `ld.global.cs`: streaming load,不污染L1 cache
   - `st.global.cs`: streaming store,直接写L2
   - 适合只访问一次的数据

2. **向量化访问**:
   - 用`uint4`一次传16字节
   - 减少内存事务数
   - 提高带宽利用率

3. **Warp协作**:
   - 32个线程协同加载128字节(PCIe事务大小)
   - 每个线程负责一部分

## 0x3.2 Kernel实现

```cpp
template <
    std::integral T,
    std::size_t kElementSize,
    std::size_t kUnroll,
    std::size_t kBlockQuota,
    std::size_t kNumThreads,
    std::size_t kMaxOccupancy>
__global__ __launch_bounds__(kNumThreads, kMaxOccupancy) 
void hicache_transfer_per_layer(const __grid_constant__ HicacheKernelParams params) {
  using namespace device;
  
  constexpr auto kWarpThreads = device::kWarpThreads / kUnroll;
  constexpr auto kWarpsPerBlock = kNumThreads / kWarpThreads;
  constexpr auto kWorkers = kWarpsPerBlock * kBlockQuota;
  constexpr auto kGranularity = 128 / kWarpThreads;
  
  const auto& [
    k_cache_dst, v_cache_dst, indices_dst,
    k_cache_src, v_cache_src, indices_src,
    length, kv_cache_src_stride, kv_cache_dst_stride, _
  ] = params;
  
  const auto warp_id = blockIdx.x * kWarpsPerBlock + threadIdx.x / kWarpThreads;
  
  for (auto i = warp_id; i < length; i += kWorkers) {
    const auto pos_src = static_cast<const T*>(indices_src)[i];
    const auto pos_dst = static_cast<const T*>(indices_dst)[i];
    
    const auto src_k = pointer::offset(k_cache_src, pos_src * kv_cache_src_stride);
    const auto dst_k = pointer::offset(k_cache_dst, pos_dst * kv_cache_dst_stride);
    const auto src_v = pointer::offset(v_cache_src, pos_src * kv_cache_src_stride);
    const auto dst_v = pointer::offset(v_cache_dst, pos_dst * kv_cache_dst_stride);
    
    const auto vec_k = warp::load_vec<kElementSize, kGranularity, kWarpThreads>(src_k);
    const auto vec_v = warp::load_vec<kElementSize, kGranularity, kWarpThreads>(src_v);
    warp::store_vec<kElementSize, kGranularity, kWarpThreads>(dst_k, vec_k);
    warp::store_vec<kElementSize, kGranularity, kWarpThreads>(dst_v, vec_v);
  }
}
```

设计思路:
- 每个warp处理一对KV cache的迁移
- 展开因子(kUnroll)控制每线程工作量
- 128字节对齐(PCIe事务大小)

## 0x3.3 Host端验证

```cpp
template <...>
struct HiCacheKernel {
  static void run_one(
      const tvm::ffi::TensorView k_cache_dst,
      const tvm::ffi::TensorView v_cache_dst,
      const tvm::ffi::TensorView indices_dst,
      const tvm::ffi::TensorView k_cache_src,
      const tvm::ffi::TensorView v_cache_src,
      const tvm::ffi::TensorView indices_src) {
    using namespace host;
    
    auto D = SymbolicSize{"head dimension"};
    auto N = SymbolicSize{"src kv stride"};
    auto M = SymbolicSize{"dst kv stride"};
    auto L = SymbolicSize{"indices length"};
    auto cache_dtype = SymbolicDType{};
    auto indices_dtype = SymbolicDType{};
    auto indices_device = SymbolicDevice{};
    
    // 验证cache tensor
    TensorMatcher({-1, D})  //
        .with_strides({N, 1})
        .with_dtype(cache_dtype)
        .with_device<kDLCUDA, kDLCUDAHost, kDLCPU>()
        .verify(k_cache_src)
        .verify(v_cache_src);
    
    TensorMatcher({-1, D})  //
        .with_strides({M, 1})
        .with_dtype(cache_dtype)
        .with_device<kDLCUDA, kDLCUDAHost, kDLCPU>()
        .verify(k_cache_dst)
        .verify(v_cache_dst);
    
    // 验证indices
    TensorMatcher({L})  //
        .with_dtype<int32_t, int64_t>(indices_dtype)
        .with_device<kDLCUDA>(indices_device)
        .verify(indices_src)
        .verify(indices_dst);
    
    // 检查编译期和运行时参数一致
    const auto dtype_size = dtype_bytes(cache_dtype.unwrap());
    const auto element_bytes = D.unwrap() * dtype_size;
    RuntimeCheck(kElementSize == element_bytes, 
                 "HicacheKernel: cache dimension mismatch.");
    
    // 启动kernel
    const auto length = static_cast<std::size_t>(L.unwrap());
    constexpr auto kWorkersPerBlock = kNumThreads / (device::kWarpThreads / kUnroll);
    const auto num_blocks = std::min(div_ceil(length, kWorkersPerBlock), kBlockQuota);
    
    const auto params = HicacheKernelParams{...};
    const auto kernel = use_int32 ? _kernel_one<int32_t> : _kernel_one<int64_t>;
    LaunchKernel(num_blocks, kNumThreads, device)(kernel, params);
  }
};
```

验证流程:
1. 检查k_cache和v_cache的head dimension一致
2. 检查步长
3. 检查数据类型
4. 检查设备
5. 验证编译期参数和运行时参数匹配

## 0x3.4 Python接口

```python
@lru_cache(maxsize=None)
def _jit_hicache_module(*, element_size: int, unroll: int, block_quota: int) -> Module:
    num_threads, occupancy = 1024, 1
    args = make_cpp_args(element_size, unroll, block_quota, num_threads, occupancy)
    return load_jit(
        "hicache",
        *args,
        cuda_files=["hicache.cuh"],
        cuda_wrappers=[
            ("launch_one", f"HiCacheKernel<{args}>::run_one"),
            ("launch_all", f"HiCacheKernel<{args}>::run_all"),
        ],
    )

def _default_unroll(element_size: int) -> int:
    if element_size <= 512:
        return 4  # 小数据:高展开
    if element_size <= 1024:
        return 2  # 中等数据
    return 1      # 大数据:低展开

def transfer_hicache_one_layer(
    k_cache_dst: torch.Tensor,
    v_cache_dst: torch.Tensor,
    indices_dst: torch.Tensor,
    k_cache_src: torch.Tensor,
    v_cache_src: torch.Tensor,
    indices_src: torch.Tensor,
    *,
    element_dim: int | None = None,
    unroll: int | None = None,
    block_quota: int | None = None,
) -> None:
    element_dim = element_dim or k_cache_dst.size(-1)
    element_size = element_dim * k_cache_dst.element_size()
    unroll = unroll or _default_unroll(element_size)
    
    module = _jit_hicache_module(
        element_size=element_size,
        unroll=unroll,
        block_quota=block_quota or DEFAULT_BLOCK_QUOTA,
    )
    module.launch_one(
        k_cache_dst, v_cache_dst, indices_dst,
        k_cache_src, v_cache_src, indices_src,
    )
```

使用`@lru_cache`缓存编译结果,相同参数的kernel只编译一次。

# 0x4. 实战案例:CUDA Wait Value

这是一个轻量级的stream同步机制,通过busy-wait实现细粒度的stream间同步。

## 0x4.1 CUDA实现

```cpp
__global__ void wait_flag_kernel(const int32_t* flag, int32_t target) {
  const volatile int32_t* vflag = (volatile const int32_t*)flag;
  
  while (*vflag != target) {
#if __CUDA_ARCH__ >= 700
    __nanosleep(100);  // Volta及以上:降低功耗
#else
    // Pre-Volta:busy-wait
#endif
  }
}

auto stream_wait_value(const tvm::ffi::TensorView flag, std::int32_t value) -> void {
  using namespace host;
  
  auto length = SymbolicSize{"length"};
  TensorMatcher({length})
    .with_dtype<int32_t>()
    .with_device<kDLCUDA>()
    .verify(flag);
  RuntimeCheck(length.unwrap() >= 1, "wait_flag expects a non-empty tensor.");
  
  auto* ptr = static_cast<std::int32_t*>(flag.data_ptr());
  const auto stream = LaunchKernel::resolve_device(flag.device());
  
  constexpr int blocks = 1;
  constexpr int threads = 1;
  wait_flag_kernel<<<blocks, threads, 0, stream>>>(ptr, value);
  RuntimeDeviceCheck(cudaGetLastError());
}
```

关键点:
- `volatile`防止编译器优化掉循环
- `__nanosleep`降低功耗(Volta+)
- 单线程kernel,最小化资源占用

## 0x4.2 Python接口

```python
@lru_cache(maxsize=1)
def _jit_stream_wait_value_module() -> Module:
    return load_jit(
        "cuda_wait_value",
        cuda_files=["cuda_wait_value.cuh"],
        cuda_wrappers=[("stream_wait_value", "stream_wait_value")],
    )

class Event:
    def __init__(self) -> None:
        self.flag = torch.zeros(1, dtype=torch.int32, device="cuda")
    
    def record(self, value: int = 1) -> None:
        self.flag[0] = value
    
    def wait(self, value: int = 1) -> None:
        stream_wait_value(self.flag, value)
```

使用场景:
```python
event = Event()

# Stream A
with torch.cuda.stream(stream_a):
    # ... 计算 ...
    event.record(1)

# Stream B
with torch.cuda.stream(stream_b):
    event.wait(1)  # 等待Stream A
    # ... 依赖Stream A的计算 ...
```

和`torch.cuda.Event`的区别:
- `torch.cuda.Event`: 基于CUDA event,不阻塞stream
- 自定义`Event`: 基于busy-wait,会阻塞stream

# 0x5. 开发指南

## 0x5.1 环境设置

安装clangd:
```bash
# Ubuntu
sudo apt install clangd-18

# macOS
brew install llvm
```

生成`.clangd`配置:
```bash
cd /path/to/sglang
python -m sglang.jit_kernel
```

会生成:
```yaml
CompileFlags:
  Add: [
    -xcuda,
    --cuda-gpu-arch=sm_80,
    -std=c++20,
    -Wall,
    -Wextra,
    -isystem/path/to/tvm_ffi/include,
    -isystem/path/to/dlpack/include,
    -isystem/path/to/sglang/jit_kernel/include
  ]
```

## 0x5.2 添加新Kernel

### 步骤1:创建CUDA源文件

在`csrc/`创建`my_kernel.cuh`:

```cpp
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/utils.h>

namespace {

__global__ void my_kernel_impl(float* output, const float* input, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = input[idx] * 2.0f;
  }
}

template <int kBlockSize>
struct MyKernel {
  static void run(tvm::ffi::TensorView output, tvm::ffi::TensorView input) {
    using namespace host;
    
    auto N = SymbolicSize{"N"};
    auto device = SymbolicDevice{};
    
    TensorMatcher({N})
        .with_dtype<float>()
        .with_device<kDLCUDA>(device)
        .verify(input)
        .verify(output);
    
    const int size = N.unwrap();
    const int num_blocks = div_ceil(size, kBlockSize);
    
    auto* out_ptr = static_cast<float*>(output.data_ptr());
    auto* in_ptr = static_cast<const float*>(input.data_ptr());
    
    LaunchKernel(num_blocks, kBlockSize, device.unwrap())(
        my_kernel_impl, out_ptr, in_ptr, size
    );
  }
};

}  // namespace
```

### 步骤2:创建Python接口

在`jit_kernel/`创建`my_kernel.py`:

```python
from functools import lru_cache
import torch
from sglang.jit_kernel.utils import load_jit, make_cpp_args

@lru_cache(maxsize=None)
def _jit_my_kernel_module(block_size: int):
    args = make_cpp_args(block_size)
    return load_jit(
        "my_kernel",
        *args,
        cuda_files=["my_kernel.cuh"],
        cuda_wrappers=[("run", f"MyKernel<{args}>::run")],
    )

def my_kernel(output: torch.Tensor, input: torch.Tensor, block_size: int = 256):
    module = _jit_my_kernel_module(block_size)
    module.run(output, input)
```

### 步骤3:使用

```python
input = torch.randn(1000, device='cuda')
output = torch.empty_like(input)
my_kernel(output, input, block_size=256)
```

# 0x6. PR #14570改进总结

这个PR主要改进了错误报告和开发体验:

## 0x6.1 错误报告增强

**改进前**:
```
Size mismatch: expected 128 but got 256
```

**改进后**:
```
Tensor match failed for Tensor<100, 256>[strides=<256, 1>, dtype=float32, device=cuda:0] 
at hicache.cuh:255
- Root cause: Size mismatch for shape#1('head dimension'): expected 128 but got 256
```

现在会打印:
- 实际tensor的完整信息
- 源码位置
- 具体哪个维度出错
- 用户定义的注释

## 0x6.2 新增功能

1. **DebugInfo**: 自动捕获源码位置
2. **Panic**: 直接抛异常的工具
3. **irange**: 类Python的range
4. **CUDA类型支持**: `__half`和`__nv_bfloat16`
5. **步长检查优化**: 跳过size=1维度

## 0x6.3 开发工具

1. **clangd配置生成**: `python -m sglang.jit_kernel`
2. **开发文档**: `docs/developer_guide/JIT_kernels.md`
3. **测试工具**: `test_utils.h`和`test_utils.py`

## 0x6.4 代码质量

1. 符号变量命名从`"D"`改为`"head dimension"`
2. 错误信息包含维度索引和注释
3. 解决NVCC对C++20 `source_location`的兼容性问题
4. 统一API,`RuntimeDeviceCheck`合并两个函数

# 0x7. 总结

SGLang的JIT kernel系统设计得还不错,主要特点:

1. **类型安全**: TensorMatcher提供编译期和运行期双重检查
2. **错误友好**: 详细的错误信息,包含源码位置和上下文
3. **高性能**: 模板参数化,编译期优化
4. **易用**: Python和C++无缝集成
5. **可扩展**: 架构清晰,容易添加新kernel

通过HiCache和CUDA Wait Value两个案例,可以看到如何把这些工具组合起来构建实际的CUDA kernel。

整个系统用了不少现代C++特性,代码质量比较高。PR #14570的改进主要集中在错误报告和开发体验上,对实际使用帮助挺大的。
