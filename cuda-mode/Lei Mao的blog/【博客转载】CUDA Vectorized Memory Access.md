> 博客来源：https://leimao.github.io/blog/CUDA-Vectorized-Memory-Access/ ，来自Lei Mao，已获得作者转载授权。后续会转载一些Lei Mao的CUDA相关Blog，也是一个完整的专栏，Blog会从稍早一些的CUDA架构到当前最新的CUDA架构，也会包含实用工程技巧，底层指令分析，Cutlass分析等等多个课题，是一个时间线十分明确的专栏。

# CUDA 向量化内存访问

## 介绍

从DRAM中读取和写入数据是CUDA编程中的基本操作之一。CUDA设备的有效内存带宽是影响CUDA函数性能最关键的因素之一，特别是当CUDA函数受内存限制时。

在这篇博客文章中，我们将展示如何通过使用向量化内存访问来提高CUDA函数的有效内存带宽。

## CUDA 向量化内存访问

在下面的示例中，我们将实现一个朴素的自定义设备内存拷贝函数，并展示如何通过对不同数据类型的连续数据使用每线程8字节或16字节的向量化内存事务来提高其有效内存带宽。使用每线程8字节或16字节向量化内存事务的结果是减少了数据拷贝所需的内存事务数量，这在几乎所有用例中都能提高有效内存带宽。

```c++
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() check_last(__FILE__, __LINE__)
void check_last(const char* const file, const int line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

std::string std_string_centered(std::string const& s, size_t width,
                                char pad = ' ')
{
    size_t const l{s.length()};
    // Throw an exception if width is too small.
    if (width < l)
    {
        throw std::runtime_error("Width is too small.");
    }
    size_t const left_pad{(width - l) / 2};
    size_t const right_pad{width - l - left_pad};
    std::string const s_centered{std::string(left_pad, pad) + s +
                                 std::string(right_pad, pad)};
    return s_centered;
}

template <class T>
float measure_performance(std::function<T(cudaStream_t)> const& bound_function,
                          cudaStream_t stream, unsigned int num_repeats = 100,
                          unsigned int num_warmups = 100)
{
    cudaEvent_t start, stop;
    float time;

    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    for (unsigned int i{0U}; i < num_warmups; ++i)
    {
        bound_function(stream);
    }

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    for (unsigned int i{0U}; i < num_repeats; ++i)
    {
        bound_function(stream);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    float const latency{time / num_repeats};

    return latency;
}

template <typename T>
__global__ void custom_device_memcpy(T* __restrict__ output,
                                     T const* __restrict__ input, size_t n)
{
    size_t const idx{blockDim.x * blockIdx.x + threadIdx.x};
    size_t const stride{blockDim.x * gridDim.x};
    for (size_t i{idx}; i < n; i += stride)
    {
        output[i] = input[i];
    }
}

template <typename T>
void launch_custom_device_memcpy(T* output, T const* input, size_t n,
                                 cudaStream_t stream)
{
    dim3 const threads_per_block{1024};
    dim3 const blocks_per_grid{static_cast<unsigned int>(std::min(
        (n + threads_per_block.x - 1U) / threads_per_block.x,
        static_cast<size_t>(std::numeric_limits<unsigned int>::max())))};
    custom_device_memcpy<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        output, input, n);
    CHECK_LAST_CUDA_ERROR();
}

template <typename T, unsigned int BLOCK_DIM_X>
__global__ void custom_device_memcpy_shared_memory(T* __restrict__ output,
                                                   T const* __restrict__ input,
                                                   size_t n)
{
    // Using shared memory as intermediate buffer.
    __shared__ T shared_memory[BLOCK_DIM_X];
    size_t const idx{blockDim.x * blockIdx.x + threadIdx.x};
    size_t const stride{blockDim.x * gridDim.x};
    for (size_t i{idx}; i < n; i += stride)
    {
        shared_memory[threadIdx.x] = input[i];
        // Synchronization is not necessary in this case.
        // __syncthreads();
        output[i] = shared_memory[threadIdx.x];
    }
}

template <typename T>
void launch_custom_device_memcpy_shared_memory(T* output, T const* input,
                                               size_t n, cudaStream_t stream)
{
    constexpr dim3 threads_per_block{1024};
    dim3 const blocks_per_grid{static_cast<unsigned int>(std::min(
        (n + threads_per_block.x - 1U) / threads_per_block.x,
        static_cast<size_t>(std::numeric_limits<unsigned int>::max())))};
    custom_device_memcpy_shared_memory<T, threads_per_block.x>
        <<<blocks_per_grid, threads_per_block, 0, stream>>>(output, input, n);
    CHECK_LAST_CUDA_ERROR();
}

// One thread copies sizeof(R) bytes of data.
// One warp copies 32 x sizeof(R) bytes of data via one of few memory
// transactions.
template <typename T, typename R = uint64_t>
__global__ void custom_device_memcpy_optimized(T* __restrict__ output,
                                               T const* __restrict__ input,
                                               size_t n)
{
    size_t const idx{blockDim.x * blockIdx.x + threadIdx.x};
    size_t const stride{blockDim.x * gridDim.x};
    for (size_t i{idx}; i * sizeof(R) / sizeof(T) < n; i += stride)
    {
        if ((i + 1U) * sizeof(R) / sizeof(T) < n)
        {
            reinterpret_cast<R*>(output)[i] =
                reinterpret_cast<R const*>(input)[i];
        }
        else
        {
            // Remaining units to copy.
            size_t const start_index{i * sizeof(R) / sizeof(T)};
            size_t const remaining_units_to_copy{(n - start_index)};
            for (size_t j{0}; j < remaining_units_to_copy; ++j)
            {
                output[start_index + j] = input[start_index + j];
            }
        }
    }
}

template <typename T, typename R = uint64_t>
void launch_custom_device_memcpy_optimized(T* output, T const* input, size_t n,
                                           cudaStream_t stream)
{
    dim3 const threads_per_block{1024};
    size_t const num_units_to_copy_round_up{(n * sizeof(T) + sizeof(R) - 1U) /
                                            sizeof(R)};
    dim3 const blocks_per_grid{static_cast<unsigned int>(std::min(
        (num_units_to_copy_round_up + threads_per_block.x - 1U) /
            threads_per_block.x,
        static_cast<size_t>(std::numeric_limits<unsigned int>::max())))};
    custom_device_memcpy_optimized<<<blocks_per_grid, threads_per_block, 0,
                                     stream>>>(output, input, n);
    CHECK_LAST_CUDA_ERROR();
}

template <typename T>
void launch_official_device_memcpy(T* output, T const* input, size_t n,
                                   cudaStream_t stream)
{
    CHECK_CUDA_ERROR(cudaMemcpyAsync(output, input, n * sizeof(T),
                                     cudaMemcpyDeviceToDevice, stream));
}

// Initialize the buffer so that the unit of the data is the index of the data.
template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
void initialize_buffer(T* buffer, size_t n)
{
    for (size_t i{0}; i < n; ++i)
    {
        buffer[i] = static_cast<T>(
            i % static_cast<size_t>(std::numeric_limits<T>::max()));
    }
}

template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
void verify_buffer(T* buffer, size_t n)
{
    for (size_t i{0}; i < n; ++i)
    {
        if (buffer[i] != static_cast<T>(i % static_cast<size_t>(
                                                std::numeric_limits<T>::max())))
        {
            std::cerr << "Verification failed at index: " << i << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
}

// Measure custom device memcpy performance given the number of units to copy,
// the device memcpy function to use, and the number of repeats and warmups.
template <typename T>
float measure_custom_device_memcpy_performance(
    size_t n,
    std::function<void(T*, T const*, size_t, cudaStream_t)> const&
        device_memcpy_function,
    int num_repeats = 100, int num_warmups = 100)
{
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::vector<T> input(n);
    std::vector<T> output(n, static_cast<T>(0));
    initialize_buffer(input.data(), n);

    T* d_input;
    T* d_output;

    CHECK_CUDA_ERROR(cudaMalloc(&d_input, n * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, n * sizeof(T)));

    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_input, input.data(), n * sizeof(T),
                                     cudaMemcpyHostToDevice, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_output, output.data(), n * sizeof(T),
                                     cudaMemcpyHostToDevice, stream));
    // Run device memcpy once to check correcness.
    device_memcpy_function(d_output, d_input, n, stream);
    CHECK_CUDA_ERROR(cudaMemcpyAsync(output.data(), d_output, n * sizeof(T),
                                     cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // Verify the correctness of the device memcpy.
    verify_buffer(output.data(), n);

    size_t const num_bytes{n * sizeof(T)};
    float const num_giga_bytes{static_cast<float>(num_bytes) / (1 << 30)};

    std::function<void(cudaStream_t)> function{std::bind(
        device_memcpy_function, d_output, d_input, n, std::placeholders::_1)};

    float const latency{
        measure_performance(function, stream, num_repeats, num_warmups)};
    std::cout << std::fixed << std::setprecision(3) << "Latency: " << latency
              << " ms" << std::endl;
    std::cout << "Effective Bandwitdh: "
              << 2.f * num_giga_bytes / (latency / 1000) << " GB/s"
              << std::endl;

    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));

    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

    // Query deive name and peak memory bandwidth.
    int device_id{0};
    cudaGetDevice(&device_id);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);
    float const peak_bandwidth{
        static_cast<float>(2.0 * device_prop.memoryClockRate *
                           (device_prop.memoryBusWidth / 8) / 1.0e6)};
    std::cout << "Percentage of Peak Bandwitdh: "
              << 2.f * num_giga_bytes / (latency / 1000) / peak_bandwidth * 100
              << "%" << std::endl;

    return latency;
}

int main()
{
    constexpr unsigned int num_repeats{10U};
    constexpr unsigned int num_warmups{10U};

    constexpr size_t tensor_size_small{1U * 64U * 64U * 64U};
    constexpr size_t tensor_size_medium{1U * 128U * 128U * 128U};
    constexpr size_t tensor_size_large{1U * 512U * 512U * 512U};

    constexpr size_t string_width{50U};

    std::cout << std_string_centered("", string_width, '~') << std::endl;
    std::cout << std_string_centered("NVIDIA GPU Device Info", string_width,
                                     ' ')
              << std::endl;
    std::cout << std_string_centered("", string_width, '~') << std::endl;

    // Query deive name and peak memory bandwidth.
    int device_id{0};
    cudaGetDevice(&device_id);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);
    std::cout << "Device Name: " << device_prop.name << std::endl;
    float const memory_size{static_cast<float>(device_prop.totalGlobalMem) /
                            (1 << 30)};
    std::cout << "Memory Size: " << memory_size << " GB" << std::endl;
    float const peak_bandwidth{
        static_cast<float>(2.0f * device_prop.memoryClockRate *
                           (device_prop.memoryBusWidth / 8) / 1.0e6)};
    std::cout << "Peak Bandwitdh: " << peak_bandwidth << " GB/s" << std::endl;
    std::cout << std::endl;

    // Measure CUDA official memcpy performance for different tensor sizes.
    std::cout << std_string_centered("", string_width, '*') << std::endl;
    std::cout << std_string_centered("CUDA Official Memcpy", string_width, ' ')
              << std::endl;
    std::cout << std_string_centered("", string_width, '*') << std::endl;

    for (size_t tensor_size :
         {tensor_size_small, tensor_size_medium, tensor_size_large})
    {
        std::string const tensor_size_string{std::string("Tensor Size: ") +
                                             std::to_string(tensor_size) +
                                             std::string(" Units")};
        std::cout << std_string_centered("", string_width, '=') << std::endl;
        std::cout << std_string_centered(tensor_size_string, string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '=') << std::endl;

        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 1 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int8_t>(
            tensor_size, launch_official_device_memcpy<int8_t>, num_repeats,
            num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 2 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int16_t>(
            tensor_size, launch_official_device_memcpy<int16_t>, num_repeats,
            num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 4 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int32_t>(
            tensor_size, launch_official_device_memcpy<int32_t>, num_repeats,
            num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 8 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int64_t>(
            tensor_size, launch_official_device_memcpy<int64_t>, num_repeats,
            num_warmups);
    }
    std::cout << std::endl;

    // Measure the latency and bandwidth of custom device memcpy for different
    // tensor sizes.
    std::cout << std_string_centered("", string_width, '*') << std::endl;
    std::cout << std_string_centered("Custom Device Memcpy", string_width, ' ')
              << std::endl;
    std::cout << std_string_centered("", string_width, '*') << std::endl;

    for (size_t tensor_size :
         {tensor_size_small, tensor_size_medium, tensor_size_large})
    {
        std::string const tensor_size_string{std::string("Tensor Size: ") +
                                             std::to_string(tensor_size) +
                                             std::string(" Units")};
        std::cout << std_string_centered("", string_width, '=') << std::endl;
        std::cout << std_string_centered(tensor_size_string, string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '=') << std::endl;

        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 1 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int8_t>(
            tensor_size, launch_custom_device_memcpy<int8_t>, num_repeats,
            num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 2 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int16_t>(
            tensor_size, launch_custom_device_memcpy<int16_t>, num_repeats,
            num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 4 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int32_t>(
            tensor_size, launch_custom_device_memcpy<int32_t>, num_repeats,
            num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 8 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int64_t>(
            tensor_size, launch_custom_device_memcpy<int64_t>, num_repeats,
            num_warmups);
    }
    std::cout << std::endl;

    // Conclusions:
    // 1. The more units of data we copy, the higher the bandwidth.
    // 2. The larger the unit of the data, the higher the bandwidth.

    // Check if shared memory can improve the latency of custom device memcpy.
    std::cout << std_string_centered("", string_width, '*') << std::endl;
    std::cout << std_string_centered("Custom Device Memcpy with Shared Memory",
                                     string_width, ' ')
              << std::endl;
    std::cout << std_string_centered("", string_width, '*') << std::endl;

    for (size_t tensor_size :
         {tensor_size_small, tensor_size_medium, tensor_size_large})
    {
        std::string const tensor_size_string{std::string("Tensor Size: ") +
                                             std::to_string(tensor_size) +
                                             std::string(" Units")};
        std::cout << std_string_centered("", string_width, '=') << std::endl;
        std::cout << std_string_centered(tensor_size_string, string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '=') << std::endl;

        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 1 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int8_t>(
            tensor_size, launch_custom_device_memcpy_shared_memory<int8_t>,
            num_repeats, num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 2 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int16_t>(
            tensor_size, launch_custom_device_memcpy_shared_memory<int16_t>,
            num_repeats, num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 4 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int32_t>(
            tensor_size, launch_custom_device_memcpy_shared_memory<int32_t>,
            num_repeats, num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 8 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int64_t>(
            tensor_size, launch_custom_device_memcpy_shared_memory<int64_t>,
            num_repeats, num_warmups);
    }
    std::cout << std::endl;

    // Conclusions:
    // 1. The effect of using shared memory for improving the latency of custom
    // device memcpy is not obvious.

    // Improve the latency of custom device memcpy when the unit of the data is
    // small.
    std::cout << std_string_centered("", string_width, '*') << std::endl;
    std::cout << std_string_centered(
                     "Custom Device Memcpy 4-Byte Copy Per Thread",
                     string_width, ' ')
              << std::endl;
    std::cout << std_string_centered("", string_width, '*') << std::endl;

    for (size_t tensor_size :
         {tensor_size_small, tensor_size_medium, tensor_size_large})
    {
        std::string const tensor_size_string{std::string("Tensor Size: ") +
                                             std::to_string(tensor_size) +
                                             std::string(" Units")};
        std::cout << std_string_centered("", string_width, '=') << std::endl;
        std::cout << std_string_centered(tensor_size_string, string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '=') << std::endl;

        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 1 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int8_t>(
            tensor_size,
            launch_custom_device_memcpy_optimized<int8_t, uint32_t>,
            num_repeats, num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 2 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int16_t>(
            tensor_size,
            launch_custom_device_memcpy_optimized<int16_t, uint32_t>,
            num_repeats, num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 4 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int32_t>(
            tensor_size,
            launch_custom_device_memcpy_optimized<int32_t, uint32_t>,
            num_repeats, num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 8 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int64_t>(
            tensor_size,
            launch_custom_device_memcpy_optimized<int64_t, uint32_t>,
            num_repeats, num_warmups);
    }
    std::cout << std::endl;

    std::cout << std_string_centered("", string_width, '*') << std::endl;
    std::cout << std_string_centered(
                     "Custom Device Memcpy 8-Byte Copy Per Thread",
                     string_width, ' ')
              << std::endl;
    std::cout << std_string_centered("", string_width, '*') << std::endl;

    for (size_t tensor_size :
         {tensor_size_small, tensor_size_medium, tensor_size_large})
    {
        std::string const tensor_size_string{std::string("Tensor Size: ") +
                                             std::to_string(tensor_size) +
                                             std::string(" Units")};
        std::cout << std_string_centered("", string_width, '=') << std::endl;
        std::cout << std_string_centered(tensor_size_string, string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '=') << std::endl;

        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 1 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int8_t>(
            tensor_size,
            launch_custom_device_memcpy_optimized<int8_t, uint64_t>,
            num_repeats, num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 2 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int16_t>(
            tensor_size,
            launch_custom_device_memcpy_optimized<int16_t, uint64_t>,
            num_repeats, num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 4 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int32_t>(
            tensor_size,
            launch_custom_device_memcpy_optimized<int32_t, uint64_t>,
            num_repeats, num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 8 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int64_t>(
            tensor_size,
            launch_custom_device_memcpy_optimized<int64_t, uint64_t>,
            num_repeats, num_warmups);
    }
    std::cout << std::endl;

    std::cout << std_string_centered("", string_width, '*') << std::endl;
    std::cout << std_string_centered(
                     "Custom Device Memcpy 16-Byte Copy Per Thread",
                     string_width, ' ')
              << std::endl;
    std::cout << std_string_centered("", string_width, '*') << std::endl;

    for (size_t tensor_size :
         {tensor_size_small, tensor_size_medium, tensor_size_large})
    {
        std::string const tensor_size_string{std::string("Tensor Size: ") +
                                             std::to_string(tensor_size) +
                                             std::string(" Units")};
        std::cout << std_string_centered("", string_width, '=') << std::endl;
        std::cout << std_string_centered(tensor_size_string, string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '=') << std::endl;

        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 1 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int8_t>(
            tensor_size, launch_custom_device_memcpy_optimized<int8_t, uint4>,
            num_repeats, num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 2 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int16_t>(
            tensor_size, launch_custom_device_memcpy_optimized<int16_t, uint4>,
            num_repeats, num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 4 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int32_t>(
            tensor_size, launch_custom_device_memcpy_optimized<int32_t, uint4>,
            num_repeats, num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 8 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int64_t>(
            tensor_size, launch_custom_device_memcpy_optimized<int64_t, uint4>,
            num_repeats, num_warmups);
    }
    std::cout << std::endl;

    // Conclusions:
    // 1. Copying data in units of 8 bytes or 16 bytes can improve the latency
    // of custom device memcpy.
}
```

该CUDA程序在配有CUDA 12.0的NVIDIA RTX 3090 GPU上进行编译和性能分析。

```shell
$ nvcc memcpy.cu -o memcpy -std=c++14
$ ./memcpy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
              NVIDIA GPU Device Info
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Device Name: NVIDIA GeForce RTX 3090
Memory Size: 23.6694 GB
Peak Bandwitdh: 936.096 GB/s

**************************************************
               CUDA Official Memcpy
**************************************************
==================================================
            Tensor Size: 262144 Units
==================================================
--------------------------------------------------
                Unit Size: 1 Byte
--------------------------------------------------
Latency: 0.002 ms
Effective Bandwitdh: 217.362 GB/s
Percentage of Peak Bandwitdh: 23.220%
--------------------------------------------------
                Unit Size: 2 Byte
--------------------------------------------------
Latency: 0.002 ms
Effective Bandwitdh: 414.641 GB/s
Percentage of Peak Bandwitdh: 44.295%
--------------------------------------------------
                Unit Size: 4 Byte
--------------------------------------------------
Latency: 0.003 ms
Effective Bandwitdh: 706.425 GB/s
Percentage of Peak Bandwitdh: 75.465%
--------------------------------------------------
                Unit Size: 8 Byte
--------------------------------------------------
Latency: 0.004 ms
Effective Bandwitdh: 1030.999 GB/s
Percentage of Peak Bandwitdh: 110.138%
==================================================
            Tensor Size: 2097152 Units
==================================================
--------------------------------------------------
                Unit Size: 1 Byte
--------------------------------------------------
Latency: 0.004 ms
Effective Bandwitdh: 1059.638 GB/s
Percentage of Peak Bandwitdh: 113.198%
--------------------------------------------------
                Unit Size: 2 Byte
--------------------------------------------------
Latency: 0.011 ms
Effective Bandwitdh: 719.754 GB/s
Percentage of Peak Bandwitdh: 76.889%
--------------------------------------------------
                Unit Size: 4 Byte
--------------------------------------------------
Latency: 0.023 ms
Effective Bandwitdh: 675.261 GB/s
Percentage of Peak Bandwitdh: 72.136%
--------------------------------------------------
                Unit Size: 8 Byte
--------------------------------------------------
Latency: 0.043 ms
Effective Bandwitdh: 719.330 GB/s
Percentage of Peak Bandwitdh: 76.844%
==================================================
           Tensor Size: 134217728 Units
==================================================
--------------------------------------------------
                Unit Size: 1 Byte
--------------------------------------------------
Latency: 0.321 ms
Effective Bandwitdh: 778.091 GB/s
Percentage of Peak Bandwitdh: 83.121%
--------------------------------------------------
                Unit Size: 2 Byte
--------------------------------------------------
Latency: 0.640 ms
Effective Bandwitdh: 781.539 GB/s
Percentage of Peak Bandwitdh: 83.489%
--------------------------------------------------
                Unit Size: 4 Byte
--------------------------------------------------
Latency: 1.275 ms
Effective Bandwitdh: 784.214 GB/s
Percentage of Peak Bandwitdh: 83.775%
--------------------------------------------------
                Unit Size: 8 Byte
--------------------------------------------------
Latency: 2.560 ms
Effective Bandwitdh: 781.282 GB/s
Percentage of Peak Bandwitdh: 83.462%

**************************************************
               Custom Device Memcpy
**************************************************
==================================================
            Tensor Size: 262144 Units
==================================================
--------------------------------------------------
                Unit Size: 1 Byte
--------------------------------------------------
Latency: 0.003 ms
Effective Bandwitdh: 183.399 GB/s
Percentage of Peak Bandwitdh: 19.592%
--------------------------------------------------
                Unit Size: 2 Byte
--------------------------------------------------
Latency: 0.003 ms
Effective Bandwitdh: 354.443 GB/s
Percentage of Peak Bandwitdh: 37.864%
--------------------------------------------------
                Unit Size: 4 Byte
--------------------------------------------------
Latency: 0.003 ms
Effective Bandwitdh: 681.196 GB/s
Percentage of Peak Bandwitdh: 72.770%
--------------------------------------------------
                Unit Size: 8 Byte
--------------------------------------------------
Latency: 0.003 ms
Effective Bandwitdh: 1192.093 GB/s
Percentage of Peak Bandwitdh: 127.347%
==================================================
            Tensor Size: 2097152 Units
==================================================
--------------------------------------------------
                Unit Size: 1 Byte
--------------------------------------------------
Latency: 0.010 ms
Effective Bandwitdh: 378.747 GB/s
Percentage of Peak Bandwitdh: 40.460%
--------------------------------------------------
                Unit Size: 2 Byte
--------------------------------------------------
Latency: 0.018 ms
Effective Bandwitdh: 445.593 GB/s
Percentage of Peak Bandwitdh: 47.601%
--------------------------------------------------
                Unit Size: 4 Byte
--------------------------------------------------
Latency: 0.024 ms
Effective Bandwitdh: 660.732 GB/s
Percentage of Peak Bandwitdh: 70.584%
--------------------------------------------------
                Unit Size: 8 Byte
--------------------------------------------------
Latency: 0.042 ms
Effective Bandwitdh: 737.140 GB/s
Percentage of Peak Bandwitdh: 78.746%
==================================================
           Tensor Size: 134217728 Units
==================================================
--------------------------------------------------
                Unit Size: 1 Byte
--------------------------------------------------
Latency: 0.972 ms
Effective Bandwitdh: 257.207 GB/s
Percentage of Peak Bandwitdh: 27.477%
--------------------------------------------------
                Unit Size: 2 Byte
--------------------------------------------------
Latency: 1.076 ms
Effective Bandwitdh: 464.543 GB/s
Percentage of Peak Bandwitdh: 49.626%
--------------------------------------------------
                Unit Size: 4 Byte
--------------------------------------------------
Latency: 1.369 ms
Effective Bandwitdh: 730.586 GB/s
Percentage of Peak Bandwitdh: 78.046%
--------------------------------------------------
                Unit Size: 8 Byte
--------------------------------------------------
Latency: 2.536 ms
Effective Bandwitdh: 788.727 GB/s
Percentage of Peak Bandwitdh: 84.257%

**************************************************
     Custom Device Memcpy with Shared Memory
**************************************************
==================================================
            Tensor Size: 262144 Units
==================================================
--------------------------------------------------
                Unit Size: 1 Byte
--------------------------------------------------
Latency: 0.003 ms
Effective Bandwitdh: 175.995 GB/s
Percentage of Peak Bandwitdh: 18.801%
--------------------------------------------------
                Unit Size: 2 Byte
--------------------------------------------------
Latency: 0.003 ms
Effective Bandwitdh: 328.853 GB/s
Percentage of Peak Bandwitdh: 35.130%
--------------------------------------------------
                Unit Size: 4 Byte
--------------------------------------------------
Latency: 0.003 ms
Effective Bandwitdh: 653.481 GB/s
Percentage of Peak Bandwitdh: 69.809%
--------------------------------------------------
                Unit Size: 8 Byte
--------------------------------------------------
Latency: 0.003 ms
Effective Bandwitdh: 1128.192 GB/s
Percentage of Peak Bandwitdh: 120.521%
==================================================
            Tensor Size: 2097152 Units
==================================================
--------------------------------------------------
                Unit Size: 1 Byte
--------------------------------------------------
Latency: 0.011 ms
Effective Bandwitdh: 353.213 GB/s
Percentage of Peak Bandwitdh: 37.733%
--------------------------------------------------
                Unit Size: 2 Byte
--------------------------------------------------
Latency: 0.018 ms
Effective Bandwitdh: 433.488 GB/s
Percentage of Peak Bandwitdh: 46.308%
--------------------------------------------------
                Unit Size: 4 Byte
--------------------------------------------------
Latency: 0.024 ms
Effective Bandwitdh: 650.261 GB/s
Percentage of Peak Bandwitdh: 69.465%
--------------------------------------------------
                Unit Size: 8 Byte
--------------------------------------------------
Latency: 0.042 ms
Effective Bandwitdh: 737.864 GB/s
Percentage of Peak Bandwitdh: 78.824%
==================================================
           Tensor Size: 134217728 Units
==================================================
--------------------------------------------------
                Unit Size: 1 Byte
--------------------------------------------------
Latency: 1.011 ms
Effective Bandwitdh: 247.181 GB/s
Percentage of Peak Bandwitdh: 26.406%
--------------------------------------------------
                Unit Size: 2 Byte
--------------------------------------------------
Latency: 1.113 ms
Effective Bandwitdh: 449.172 GB/s
Percentage of Peak Bandwitdh: 47.984%
--------------------------------------------------
                Unit Size: 4 Byte
--------------------------------------------------
Latency: 1.391 ms
Effective Bandwitdh: 718.748 GB/s
Percentage of Peak Bandwitdh: 76.781%
--------------------------------------------------
                Unit Size: 8 Byte
--------------------------------------------------
Latency: 2.546 ms
Effective Bandwitdh: 785.429 GB/s
Percentage of Peak Bandwitdh: 83.905%

**************************************************
   Custom Device Memcpy 4-Byte Copy Per Thread
**************************************************
==================================================
            Tensor Size: 262144 Units
==================================================
--------------------------------------------------
                Unit Size: 1 Byte
--------------------------------------------------
Latency: 0.002 ms
Effective Bandwitdh: 238.419 GB/s
Percentage of Peak Bandwitdh: 25.469%
--------------------------------------------------
                Unit Size: 2 Byte
--------------------------------------------------
Latency: 0.002 ms
Effective Bandwitdh: 437.842 GB/s
Percentage of Peak Bandwitdh: 46.773%
--------------------------------------------------
                Unit Size: 4 Byte
--------------------------------------------------
Latency: 0.003 ms
Effective Bandwitdh: 684.251 GB/s
Percentage of Peak Bandwitdh: 73.096%
--------------------------------------------------
                Unit Size: 8 Byte
--------------------------------------------------
Latency: 0.004 ms
Effective Bandwitdh: 1003.868 GB/s
Percentage of Peak Bandwitdh: 107.240%
==================================================
            Tensor Size: 2097152 Units
==================================================
--------------------------------------------------
                Unit Size: 1 Byte
--------------------------------------------------
Latency: 0.004 ms
Effective Bandwitdh: 968.812 GB/s
Percentage of Peak Bandwitdh: 103.495%
--------------------------------------------------
                Unit Size: 2 Byte
--------------------------------------------------
Latency: 0.012 ms
Effective Bandwitdh: 675.168 GB/s
Percentage of Peak Bandwitdh: 72.126%
--------------------------------------------------
                Unit Size: 4 Byte
--------------------------------------------------
Latency: 0.024 ms
Effective Bandwitdh: 660.196 GB/s
Percentage of Peak Bandwitdh: 70.527%
--------------------------------------------------
                Unit Size: 8 Byte
--------------------------------------------------
Latency: 0.045 ms
Effective Bandwitdh: 690.443 GB/s
Percentage of Peak Bandwitdh: 73.758%
==================================================
           Tensor Size: 134217728 Units
==================================================
--------------------------------------------------
                Unit Size: 1 Byte
--------------------------------------------------
Latency: 0.366 ms
Effective Bandwitdh: 682.529 GB/s
Percentage of Peak Bandwitdh: 72.912%
--------------------------------------------------
                Unit Size: 2 Byte
--------------------------------------------------
Latency: 0.722 ms
Effective Bandwitdh: 692.125 GB/s
Percentage of Peak Bandwitdh: 73.937%
--------------------------------------------------
                Unit Size: 4 Byte
--------------------------------------------------
Latency: 1.422 ms
Effective Bandwitdh: 703.431 GB/s
Percentage of Peak Bandwitdh: 75.145%
--------------------------------------------------
                Unit Size: 8 Byte
--------------------------------------------------
Latency: 2.824 ms
Effective Bandwitdh: 708.144 GB/s
Percentage of Peak Bandwitdh: 75.649%

**************************************************
   Custom Device Memcpy 8-Byte Copy Per Thread
**************************************************
==================================================
            Tensor Size: 262144 Units
==================================================
--------------------------------------------------
                Unit Size: 1 Byte
--------------------------------------------------
Latency: 0.002 ms
Effective Bandwitdh: 238.792 GB/s
Percentage of Peak Bandwitdh: 25.509%
--------------------------------------------------
                Unit Size: 2 Byte
--------------------------------------------------
Latency: 0.002 ms
Effective Bandwitdh: 434.723 GB/s
Percentage of Peak Bandwitdh: 46.440%
--------------------------------------------------
                Unit Size: 4 Byte
--------------------------------------------------
Latency: 0.003 ms
Effective Bandwitdh: 681.196 GB/s
Percentage of Peak Bandwitdh: 72.770%
--------------------------------------------------
                Unit Size: 8 Byte
--------------------------------------------------
Latency: 0.004 ms
Effective Bandwitdh: 1030.999 GB/s
Percentage of Peak Bandwitdh: 110.138%
==================================================
            Tensor Size: 2097152 Units
==================================================
--------------------------------------------------
                Unit Size: 1 Byte
--------------------------------------------------
Latency: 0.004 ms
Effective Bandwitdh: 978.128 GB/s
Percentage of Peak Bandwitdh: 104.490%
--------------------------------------------------
                Unit Size: 2 Byte
--------------------------------------------------
Latency: 0.012 ms
Effective Bandwitdh: 677.416 GB/s
Percentage of Peak Bandwitdh: 72.366%
--------------------------------------------------
                Unit Size: 4 Byte
--------------------------------------------------
Latency: 0.022 ms
Effective Bandwitdh: 696.748 GB/s
Percentage of Peak Bandwitdh: 74.431%
--------------------------------------------------
                Unit Size: 8 Byte
--------------------------------------------------
Latency: 0.042 ms
Effective Bandwitdh: 738.924 GB/s
Percentage of Peak Bandwitdh: 78.937%
==================================================
           Tensor Size: 134217728 Units
==================================================
--------------------------------------------------
                Unit Size: 1 Byte
--------------------------------------------------
Latency: 0.320 ms
Effective Bandwitdh: 781.750 GB/s
Percentage of Peak Bandwitdh: 83.512%
--------------------------------------------------
                Unit Size: 2 Byte
--------------------------------------------------
Latency: 0.636 ms
Effective Bandwitdh: 786.536 GB/s
Percentage of Peak Bandwitdh: 84.023%
--------------------------------------------------
                Unit Size: 4 Byte
--------------------------------------------------
Latency: 1.265 ms
Effective Bandwitdh: 790.547 GB/s
Percentage of Peak Bandwitdh: 84.451%
--------------------------------------------------
                Unit Size: 8 Byte
--------------------------------------------------
Latency: 2.530 ms
Effective Bandwitdh: 790.419 GB/s
Percentage of Peak Bandwitdh: 84.438%

**************************************************
   Custom Device Memcpy 16-Byte Copy Per Thread
**************************************************
==================================================
            Tensor Size: 262144 Units
==================================================
--------------------------------------------------
                Unit Size: 1 Byte
--------------------------------------------------
Latency: 0.002 ms
Effective Bandwitdh: 216.744 GB/s
Percentage of Peak Bandwitdh: 23.154%
--------------------------------------------------
                Unit Size: 2 Byte
--------------------------------------------------
Latency: 0.002 ms
Effective Bandwitdh: 414.641 GB/s
Percentage of Peak Bandwitdh: 44.295%
--------------------------------------------------
                Unit Size: 4 Byte
--------------------------------------------------
Latency: 0.002 ms
Effective Bandwitdh: 829.282 GB/s
Percentage of Peak Bandwitdh: 88.589%
--------------------------------------------------
                Unit Size: 8 Byte
--------------------------------------------------
Latency: 0.003 ms
Effective Bandwitdh: 1192.093 GB/s
Percentage of Peak Bandwitdh: 127.347%
==================================================
            Tensor Size: 2097152 Units
==================================================
--------------------------------------------------
                Unit Size: 1 Byte
--------------------------------------------------
Latency: 0.003 ms
Effective Bandwitdh: 1128.192 GB/s
Percentage of Peak Bandwitdh: 120.521%
--------------------------------------------------
                Unit Size: 2 Byte
--------------------------------------------------
Latency: 0.010 ms
Effective Bandwitdh: 755.386 GB/s
Percentage of Peak Bandwitdh: 80.695%
--------------------------------------------------
                Unit Size: 4 Byte
--------------------------------------------------
Latency: 0.023 ms
Effective Bandwitdh: 687.333 GB/s
Percentage of Peak Bandwitdh: 73.425%
--------------------------------------------------
                Unit Size: 8 Byte
--------------------------------------------------
Latency: 0.043 ms
Effective Bandwitdh: 728.343 GB/s
Percentage of Peak Bandwitdh: 77.806%
==================================================
           Tensor Size: 134217728 Units
==================================================
--------------------------------------------------
                Unit Size: 1 Byte
--------------------------------------------------
Latency: 0.321 ms
Effective Bandwitdh: 779.006 GB/s
Percentage of Peak Bandwitdh: 83.219%
--------------------------------------------------
                Unit Size: 2 Byte
--------------------------------------------------
Latency: 0.639 ms
Effective Bandwitdh: 782.639 GB/s
Percentage of Peak Bandwitdh: 83.607%
--------------------------------------------------
                Unit Size: 4 Byte
--------------------------------------------------
Latency: 1.280 ms
Effective Bandwitdh: 781.520 GB/s
Percentage of Peak Bandwitdh: 83.487%
--------------------------------------------------
                Unit Size: 8 Byte
--------------------------------------------------
Latency: 2.552 ms
Effective Bandwitdh: 783.602 GB/s
Percentage of Peak Bandwitdh: 83.710%
```

## 结论

从结果中我们可以看出：

- 我们拷贝的数据单元越多，有效内存带宽越高。
- 数据单元越大，有效内存带宽越高。
- 在大多数情况下，以8字节或16字节的向量化单元拷贝数据可以提高自定义设备内存拷贝的有效内存带宽，特别是当数据单元较小时。
- 使用共享内存来提高自定义设备内存拷贝的有效内存带宽的效果并不明显。

请注意，尽管我们可以在这个用例中直接使用CUDA官方的内存拷贝函数，但了解如何编写和改进自定义设备内存拷贝函数仍然很有价值，因为在更实际的CUDA应用中，要拷贝的数据可能在内存中不是连续的，我们可能需要从多个源拷贝数据到多个目标位置。

## 参考资料
- CUDA Pro Tip: Increase Performance with Vectorized Memory Access(https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/)

