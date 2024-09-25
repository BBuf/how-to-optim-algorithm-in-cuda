#pragma once

// 定义一个模板结构体 TransposeParams，用于存储转置操作的参数
template <typename T> struct TransposeParams {
  T *input;  // 输入矩阵指针
  T *output; // 输出矩阵指针

  const int M; // 矩阵的行数
  const int N; // 矩阵的列数

  // 构造函数，用于初始化 TransposeParams 结构体
  TransposeParams(T *input_, T *output_, int M_, int N_)
      : input(input_), output(output_), M(M_), N(N_) {}
};

//template <typename T> int benchmark(void (*transpose)(int M, int N, T* input, T* output), int M, int N, int iterations=10, bool verify=true) {
template <typename T, bool isTranspose = true> int benchmark(void (*transpose)(TransposeParams<T> params), int M, int N, int iterations=10, bool verify=true) {
  using namespace cute;

  // 创建张量形状
  auto tensor_shape_S = make_shape(M, N);
  auto tensor_shape_D = (isTranspose) ? make_shape(N, M) : make_shape(M, N);

  // 分配和初始化主机向量
  thrust::host_vector<T> h_S(size(tensor_shape_S));       // (M, N)
  thrust::host_vector<T> h_D(size(tensor_shape_D)); // (N, M)

  // 初始化输入数据
  for (size_t i = 0; i < h_S.size(); ++i)
    h_S[i] = static_cast<T>(i);

  // 将主机向量复制到设备向量
  thrust::device_vector<T> d_S = h_S;
  thrust::device_vector<T> d_D = h_D;

  // 创建 TransposeParams 对象
  TransposeParams<T> params(thrust::raw_pointer_cast(d_S.data()), thrust::raw_pointer_cast(d_D.data()), M, N);

  // 进行多次迭代以测量性能
  for (int i = 0; i < iterations; i++) {
    auto t1 = std::chrono::high_resolution_clock::now(); // 记录开始时间
    transpose(params); // 调用转置函数
    cudaError result = cudaDeviceSynchronize(); // 同步设备
    auto t2 = std::chrono::high_resolution_clock::now(); // 记录结束时间
    if (result != cudaSuccess) {
      std::cerr << "CUDA Runtime error: " << cudaGetErrorString(result)
                << std::endl;
      return -1;
    }
    std::chrono::duration<double, std::milli> tDiff = t2 - t1; // 计算时间差
    double time_ms = tDiff.count(); // 转换为毫秒
    std::cout << "Trial " << i << " Completed in " << time_ms << "ms ("
              << 2e-6 * M * N * sizeof(T) / time_ms << " GB/s)"
              << std::endl;
  }

  // 验证结果
  if(verify) {
    h_D = d_D; // 将设备向量复制回主机向量
  
    int bad = 0;
    if constexpr (isTranspose) {
      auto transpose_function = make_layout(tensor_shape_S, LayoutRight{});
      for (size_t i = 0; i < h_D.size(); ++i) 
        if (h_D[i] != h_S[transpose_function(i)])
          bad++;
    } else {
      for (size_t i = 0; i < h_D.size(); ++i) 
        if (h_D[i] != h_S[i])
          bad++;
    }
  
    if (bad > 0) {
      std::cout << "Validation failed. Correct values: " << h_D.size()-bad << ". Incorrect values: " << bad << std::endl;
    } else {
      std::cout << "Validation success." << std::endl;
    }
  }
  return 0;
}
