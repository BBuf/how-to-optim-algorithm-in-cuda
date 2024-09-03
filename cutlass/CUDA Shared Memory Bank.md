> 博客来源：https://leimao.github.io/blog/CUDA-Shared-Memory-Bank/ ，是在 https://research.colfax-intl.com/tutorial-matrix-transpose-in-cutlass/ 博客中作者推荐的一个讲解Shared Memory Bank的优质文章。这里翻译学习一下。

# 介绍

Memory bank是CUDA共享内存的一个关键概念。为了从CUDA kernel实现中获得最佳性能，用户必须注意memory bank访问并避免memory bank访问冲突。

在这篇博文中，我想快速讨论一下CUDA共享内存的memory bank。

# Memory Bank

## Memory Bank属性

为了实现并发访问的高内存带宽，共享内存被分成大小相等的内存模块（bank），这些模块可以同时被访问。因此，任何跨越不同memory bank的内存加载或存储地址都可以同时被服务，产生的有效带宽是单个bank带宽的 $n$ 倍。

然而，如果一个内存请求的多个地址映射到同一个memory bank，这些访问将被串行化。硬件会将有bank conflict的内存请求分割成多个必要的无冲突请求，有效带宽会按照单独内存请求的数量成比例下降。这里唯一的例外是当一个warp中的多个线程访问同一个共享内存位置时，会导致广播。在这种情况下，从不同bank的多个广播被合并为一个从请求的共享内存位置到线程的单播。

## Memory Bank映射

上面描述了memory bank的属性。然而，内存地址如何映射到memory bank是特定于架构的。

在计算能力5.x或更新的设备上，每个bank每个时钟周期有32位的带宽，连续的32位字被分配给连续的bank。warp大小是32个线程，bank的数量也是32，所以bank conflict可能发生在warp中的任何线程之间。

为了详细说明这一点，让我们通过例子来看看内存地址是如何映射到memory bank的。以下程序说明了计算能力5.x或更新设备的1D和2D内存地址到memory bank映射的概念。

```c++
#include <iostream>
#include <memory>
#include <vector>

template <typename T>
void bank_id_1d_mapping(int bank_size, int num_banks, int N)
{
    for (int i{0}; i < N; ++i)
    {
        // bank_size: Bank size in bits.
        // 8: 8 bits per Byte.
        int bank_idx = (i * sizeof(T) * 8 / bank_size) % num_banks;
        std::cout << "Array Idx: " << i << " "
                  << "Bank Idx: " << bank_idx << std::endl;
    }
}

template <typename T>
void bank_id_2d_mapping(int bank_size, int num_banks, int M, int N)
{
    for (int i{0}; i < M; ++i)
    {
        for (int j{0}; j < N; ++j)
        {
            int bank_idx =
                ((i * N + j) * sizeof(T) * 8 / bank_size) % num_banks;
            std::cout << "Matrix Idx: (" << i << ", " << j << ") "
                      << "Bank Idx: " << bank_idx << std::endl;
        }
    }
}

int main()
{

    constexpr const int bank_size{32}; // bits
    constexpr const int num_banks{32};

    const int M{4};
    const int N{32};

    std::cout << "Bank ID Mapping 1D: N = " << N << std::endl;
    bank_id_1d_mapping<float>(bank_size, num_banks, N);
    std::cout << "Bank 2D Mapping 1D: M = " << M << " N = " << N << std::endl;
    bank_id_2d_mapping<float>(bank_size, num_banks, M, N);
}
```

结果：

```c++
$ g++ memory_bank.cpp -o memory_bank -std=c++14
$ ./memory_bank
Bank ID Mapping 1D: N = 32
Array Idx: 0 Bank Idx: 0
Array Idx: 1 Bank Idx: 1
Array Idx: 2 Bank Idx: 2
Array Idx: 3 Bank Idx: 3
Array Idx: 4 Bank Idx: 4
Array Idx: 5 Bank Idx: 5
Array Idx: 6 Bank Idx: 6
Array Idx: 7 Bank Idx: 7
Array Idx: 8 Bank Idx: 8
Array Idx: 9 Bank Idx: 9
Array Idx: 10 Bank Idx: 10
Array Idx: 11 Bank Idx: 11
Array Idx: 12 Bank Idx: 12
Array Idx: 13 Bank Idx: 13
Array Idx: 14 Bank Idx: 14
Array Idx: 15 Bank Idx: 15
Array Idx: 16 Bank Idx: 16
Array Idx: 17 Bank Idx: 17
Array Idx: 18 Bank Idx: 18
Array Idx: 19 Bank Idx: 19
Array Idx: 20 Bank Idx: 20
Array Idx: 21 Bank Idx: 21
Array Idx: 22 Bank Idx: 22
Array Idx: 23 Bank Idx: 23
Array Idx: 24 Bank Idx: 24
Array Idx: 25 Bank Idx: 25
Array Idx: 26 Bank Idx: 26
Array Idx: 27 Bank Idx: 27
Array Idx: 28 Bank Idx: 28
Array Idx: 29 Bank Idx: 29
Array Idx: 30 Bank Idx: 30
Array Idx: 31 Bank Idx: 31
Bank 2D Mapping 1D: M = 4 N = 32
Matrix Idx: (0, 0) Bank Idx: 0
Matrix Idx: (0, 1) Bank Idx: 1
Matrix Idx: (0, 2) Bank Idx: 2
Matrix Idx: (0, 3) Bank Idx: 3
Matrix Idx: (0, 4) Bank Idx: 4
Matrix Idx: (0, 5) Bank Idx: 5
Matrix Idx: (0, 6) Bank Idx: 6
Matrix Idx: (0, 7) Bank Idx: 7
Matrix Idx: (0, 8) Bank Idx: 8
Matrix Idx: (0, 9) Bank Idx: 9
Matrix Idx: (0, 10) Bank Idx: 10
Matrix Idx: (0, 11) Bank Idx: 11
Matrix Idx: (0, 12) Bank Idx: 12
Matrix Idx: (0, 13) Bank Idx: 13
Matrix Idx: (0, 14) Bank Idx: 14
Matrix Idx: (0, 15) Bank Idx: 15
Matrix Idx: (0, 16) Bank Idx: 16
Matrix Idx: (0, 17) Bank Idx: 17
Matrix Idx: (0, 18) Bank Idx: 18
Matrix Idx: (0, 19) Bank Idx: 19
Matrix Idx: (0, 20) Bank Idx: 20
Matrix Idx: (0, 21) Bank Idx: 21
Matrix Idx: (0, 22) Bank Idx: 22
Matrix Idx: (0, 23) Bank Idx: 23
Matrix Idx: (0, 24) Bank Idx: 24
Matrix Idx: (0, 25) Bank Idx: 25
Matrix Idx: (0, 26) Bank Idx: 26
Matrix Idx: (0, 27) Bank Idx: 27
Matrix Idx: (0, 28) Bank Idx: 28
Matrix Idx: (0, 29) Bank Idx: 29
Matrix Idx: (0, 30) Bank Idx: 30
Matrix Idx: (0, 31) Bank Idx: 31
Matrix Idx: (1, 0) Bank Idx: 0
Matrix Idx: (1, 1) Bank Idx: 1
Matrix Idx: (1, 2) Bank Idx: 2
Matrix Idx: (1, 3) Bank Idx: 3
Matrix Idx: (1, 4) Bank Idx: 4
Matrix Idx: (1, 5) Bank Idx: 5
Matrix Idx: (1, 6) Bank Idx: 6
Matrix Idx: (1, 7) Bank Idx: 7
Matrix Idx: (1, 8) Bank Idx: 8
Matrix Idx: (1, 9) Bank Idx: 9
Matrix Idx: (1, 10) Bank Idx: 10
Matrix Idx: (1, 11) Bank Idx: 11
Matrix Idx: (1, 12) Bank Idx: 12
Matrix Idx: (1, 13) Bank Idx: 13
Matrix Idx: (1, 14) Bank Idx: 14
Matrix Idx: (1, 15) Bank Idx: 15
Matrix Idx: (1, 16) Bank Idx: 16
Matrix Idx: (1, 17) Bank Idx: 17
Matrix Idx: (1, 18) Bank Idx: 18
Matrix Idx: (1, 19) Bank Idx: 19
Matrix Idx: (1, 20) Bank Idx: 20
Matrix Idx: (1, 21) Bank Idx: 21
Matrix Idx: (1, 22) Bank Idx: 22
Matrix Idx: (1, 23) Bank Idx: 23
Matrix Idx: (1, 24) Bank Idx: 24
Matrix Idx: (1, 25) Bank Idx: 25
Matrix Idx: (1, 26) Bank Idx: 26
Matrix Idx: (1, 27) Bank Idx: 27
Matrix Idx: (1, 28) Bank Idx: 28
Matrix Idx: (1, 29) Bank Idx: 29
Matrix Idx: (1, 30) Bank Idx: 30
Matrix Idx: (1, 31) Bank Idx: 31
Matrix Idx: (2, 0) Bank Idx: 0
Matrix Idx: (2, 1) Bank Idx: 1
Matrix Idx: (2, 2) Bank Idx: 2
Matrix Idx: (2, 3) Bank Idx: 3
Matrix Idx: (2, 4) Bank Idx: 4
Matrix Idx: (2, 5) Bank Idx: 5
Matrix Idx: (2, 6) Bank Idx: 6
Matrix Idx: (2, 7) Bank Idx: 7
Matrix Idx: (2, 8) Bank Idx: 8
Matrix Idx: (2, 9) Bank Idx: 9
Matrix Idx: (2, 10) Bank Idx: 10
Matrix Idx: (2, 11) Bank Idx: 11
Matrix Idx: (2, 12) Bank Idx: 12
Matrix Idx: (2, 13) Bank Idx: 13
Matrix Idx: (2, 14) Bank Idx: 14
Matrix Idx: (2, 15) Bank Idx: 15
Matrix Idx: (2, 16) Bank Idx: 16
Matrix Idx: (2, 17) Bank Idx: 17
Matrix Idx: (2, 18) Bank Idx: 18
Matrix Idx: (2, 19) Bank Idx: 19
Matrix Idx: (2, 20) Bank Idx: 20
Matrix Idx: (2, 21) Bank Idx: 21
Matrix Idx: (2, 22) Bank Idx: 22
Matrix Idx: (2, 23) Bank Idx: 23
Matrix Idx: (2, 24) Bank Idx: 24
Matrix Idx: (2, 25) Bank Idx: 25
Matrix Idx: (2, 26) Bank Idx: 26
Matrix Idx: (2, 27) Bank Idx: 27
Matrix Idx: (2, 28) Bank Idx: 28
Matrix Idx: (2, 29) Bank Idx: 29
Matrix Idx: (2, 30) Bank Idx: 30
Matrix Idx: (2, 31) Bank Idx: 31
Matrix Idx: (3, 0) Bank Idx: 0
Matrix Idx: (3, 1) Bank Idx: 1
Matrix Idx: (3, 2) Bank Idx: 2
Matrix Idx: (3, 3) Bank Idx: 3
Matrix Idx: (3, 4) Bank Idx: 4
Matrix Idx: (3, 5) Bank Idx: 5
Matrix Idx: (3, 6) Bank Idx: 6
Matrix Idx: (3, 7) Bank Idx: 7
Matrix Idx: (3, 8) Bank Idx: 8
Matrix Idx: (3, 9) Bank Idx: 9
Matrix Idx: (3, 10) Bank Idx: 10
Matrix Idx: (3, 11) Bank Idx: 11
Matrix Idx: (3, 12) Bank Idx: 12
Matrix Idx: (3, 13) Bank Idx: 13
Matrix Idx: (3, 14) Bank Idx: 14
Matrix Idx: (3, 15) Bank Idx: 15
Matrix Idx: (3, 16) Bank Idx: 16
Matrix Idx: (3, 17) Bank Idx: 17
Matrix Idx: (3, 18) Bank Idx: 18
Matrix Idx: (3, 19) Bank Idx: 19
Matrix Idx: (3, 20) Bank Idx: 20
Matrix Idx: (3, 21) Bank Idx: 21
Matrix Idx: (3, 22) Bank Idx: 22
Matrix Idx: (3, 23) Bank Idx: 23
Matrix Idx: (3, 24) Bank Idx: 24
Matrix Idx: (3, 25) Bank Idx: 25
Matrix Idx: (3, 26) Bank Idx: 26
Matrix Idx: (3, 27) Bank Idx: 27
Matrix Idx: (3, 28) Bank Idx: 28
Matrix Idx: (3, 29) Bank Idx: 29
Matrix Idx: (3, 30) Bank Idx: 30
Matrix Idx: (3, 31) Bank Idx: 31

```

## Memory Bank Conflicts

注意，对于2D矩阵，假设数据类型的位宽是32位，如果列数是32的倍数，那么矩阵同一列中的元素将属于同一个memory bank。这正是在实现中容易发生memory bank conflict的地方。如果一个warp中的线程试图访问矩阵同一列中的值，就会发生严重的memory bank conflict。使用一些其他的列数值，比如33，可以避免矩阵同一列中的元素属于同一个memory bank。因此，要注意memory bank访问的步长。

```c++
#include <iostream>
#include <memory>
#include <vector>

template <typename T>
void bank_id_1d_mapping(int bank_size, int num_banks, int N)
{
    for (int i{0}; i < N; ++i)
    {
        int bank_idx = (i * sizeof(T) * 8 / bank_size) % num_banks;
        std::cout << "Array Idx: " << i << " "
                  << "Bank Idx: " << bank_idx << std::endl;
    }
}

template <typename T>
void bank_id_2d_mapping(int bank_size, int num_banks, int M, int N)
{
    for (int i{0}; i < M; ++i)
    {
        for (int j{0}; j < N; ++j)
        {
            int bank_idx =
                ((i * N + j) * sizeof(T) * 8 / bank_size) % num_banks;
            std::cout << "Matrix Idx: (" << i << ", " << j << ") "
                      << "Bank Idx: " << bank_idx << std::endl;
        }
    }
}

int main()
{

    constexpr const int bank_size{32}; // bits
    constexpr const int num_banks{32};

    const int M{4};
    const int N{33};

    std::cout << "Bank ID Mapping 1D: N = " << N << std::endl;
    bank_id_1d_mapping<float>(bank_size, num_banks, N);
    std::cout << "Bank 2D Mapping 1D: M = " << M << " N = " << N << std::endl;
    bank_id_2d_mapping<float>(bank_size, num_banks, M, N);
}
```

在实践中，额外的列是未使用的，可以填充任何值。只是要确保实现的算法不会因意外使用不应该在额外列中使用的值而产生错误的结果。

```c++
$ g++ memory_bank.cpp -o memory_bank -std=c++14
$ ./memory_bank
Bank ID Mapping 1D: N = 33
Array Idx: 0 Bank Idx: 0
Array Idx: 1 Bank Idx: 1
Array Idx: 2 Bank Idx: 2
Array Idx: 3 Bank Idx: 3
Array Idx: 4 Bank Idx: 4
Array Idx: 5 Bank Idx: 5
Array Idx: 6 Bank Idx: 6
Array Idx: 7 Bank Idx: 7
Array Idx: 8 Bank Idx: 8
Array Idx: 9 Bank Idx: 9
Array Idx: 10 Bank Idx: 10
Array Idx: 11 Bank Idx: 11
Array Idx: 12 Bank Idx: 12
Array Idx: 13 Bank Idx: 13
Array Idx: 14 Bank Idx: 14
Array Idx: 15 Bank Idx: 15
Array Idx: 16 Bank Idx: 16
Array Idx: 17 Bank Idx: 17
Array Idx: 18 Bank Idx: 18
Array Idx: 19 Bank Idx: 19
Array Idx: 20 Bank Idx: 20
Array Idx: 21 Bank Idx: 21
Array Idx: 22 Bank Idx: 22
Array Idx: 23 Bank Idx: 23
Array Idx: 24 Bank Idx: 24
Array Idx: 25 Bank Idx: 25
Array Idx: 26 Bank Idx: 26
Array Idx: 27 Bank Idx: 27
Array Idx: 28 Bank Idx: 28
Array Idx: 29 Bank Idx: 29
Array Idx: 30 Bank Idx: 30
Array Idx: 31 Bank Idx: 31
Array Idx: 32 Bank Idx: 0
Bank 2D Mapping 1D: M = 4 N = 33
Matrix Idx: (0, 0) Bank Idx: 0
Matrix Idx: (0, 1) Bank Idx: 1
Matrix Idx: (0, 2) Bank Idx: 2
Matrix Idx: (0, 3) Bank Idx: 3
Matrix Idx: (0, 4) Bank Idx: 4
Matrix Idx: (0, 5) Bank Idx: 5
Matrix Idx: (0, 6) Bank Idx: 6
Matrix Idx: (0, 7) Bank Idx: 7
Matrix Idx: (0, 8) Bank Idx: 8
Matrix Idx: (0, 9) Bank Idx: 9
Matrix Idx: (0, 10) Bank Idx: 10
Matrix Idx: (0, 11) Bank Idx: 11
Matrix Idx: (0, 12) Bank Idx: 12
Matrix Idx: (0, 13) Bank Idx: 13
Matrix Idx: (0, 14) Bank Idx: 14
Matrix Idx: (0, 15) Bank Idx: 15
Matrix Idx: (0, 16) Bank Idx: 16
Matrix Idx: (0, 17) Bank Idx: 17
Matrix Idx: (0, 18) Bank Idx: 18
Matrix Idx: (0, 19) Bank Idx: 19
Matrix Idx: (0, 20) Bank Idx: 20
Matrix Idx: (0, 21) Bank Idx: 21
Matrix Idx: (0, 22) Bank Idx: 22
Matrix Idx: (0, 23) Bank Idx: 23
Matrix Idx: (0, 24) Bank Idx: 24
Matrix Idx: (0, 25) Bank Idx: 25
Matrix Idx: (0, 26) Bank Idx: 26
Matrix Idx: (0, 27) Bank Idx: 27
Matrix Idx: (0, 28) Bank Idx: 28
Matrix Idx: (0, 29) Bank Idx: 29
Matrix Idx: (0, 30) Bank Idx: 30
Matrix Idx: (0, 31) Bank Idx: 31
Matrix Idx: (0, 32) Bank Idx: 0
Matrix Idx: (1, 0) Bank Idx: 1
Matrix Idx: (1, 1) Bank Idx: 2
Matrix Idx: (1, 2) Bank Idx: 3
Matrix Idx: (1, 3) Bank Idx: 4
Matrix Idx: (1, 4) Bank Idx: 5
Matrix Idx: (1, 5) Bank Idx: 6
Matrix Idx: (1, 6) Bank Idx: 7
Matrix Idx: (1, 7) Bank Idx: 8
Matrix Idx: (1, 8) Bank Idx: 9
Matrix Idx: (1, 9) Bank Idx: 10
Matrix Idx: (1, 10) Bank Idx: 11
Matrix Idx: (1, 11) Bank Idx: 12
Matrix Idx: (1, 12) Bank Idx: 13
Matrix Idx: (1, 13) Bank Idx: 14
Matrix Idx: (1, 14) Bank Idx: 15
Matrix Idx: (1, 15) Bank Idx: 16
Matrix Idx: (1, 16) Bank Idx: 17
Matrix Idx: (1, 17) Bank Idx: 18
Matrix Idx: (1, 18) Bank Idx: 19
Matrix Idx: (1, 19) Bank Idx: 20
Matrix Idx: (1, 20) Bank Idx: 21
Matrix Idx: (1, 21) Bank Idx: 22
Matrix Idx: (1, 22) Bank Idx: 23
Matrix Idx: (1, 23) Bank Idx: 24
Matrix Idx: (1, 24) Bank Idx: 25
Matrix Idx: (1, 25) Bank Idx: 26
Matrix Idx: (1, 26) Bank Idx: 27
Matrix Idx: (1, 27) Bank Idx: 28
Matrix Idx: (1, 28) Bank Idx: 29
Matrix Idx: (1, 29) Bank Idx: 30
Matrix Idx: (1, 30) Bank Idx: 31
Matrix Idx: (1, 31) Bank Idx: 0
Matrix Idx: (1, 32) Bank Idx: 1
Matrix Idx: (2, 0) Bank Idx: 2
Matrix Idx: (2, 1) Bank Idx: 3
Matrix Idx: (2, 2) Bank Idx: 4
Matrix Idx: (2, 3) Bank Idx: 5
Matrix Idx: (2, 4) Bank Idx: 6
Matrix Idx: (2, 5) Bank Idx: 7
Matrix Idx: (2, 6) Bank Idx: 8
Matrix Idx: (2, 7) Bank Idx: 9
Matrix Idx: (2, 8) Bank Idx: 10
Matrix Idx: (2, 9) Bank Idx: 11
Matrix Idx: (2, 10) Bank Idx: 12
Matrix Idx: (2, 11) Bank Idx: 13
Matrix Idx: (2, 12) Bank Idx: 14
Matrix Idx: (2, 13) Bank Idx: 15
Matrix Idx: (2, 14) Bank Idx: 16
Matrix Idx: (2, 15) Bank Idx: 17
Matrix Idx: (2, 16) Bank Idx: 18
Matrix Idx: (2, 17) Bank Idx: 19
Matrix Idx: (2, 18) Bank Idx: 20
Matrix Idx: (2, 19) Bank Idx: 21
Matrix Idx: (2, 20) Bank Idx: 22
Matrix Idx: (2, 21) Bank Idx: 23
Matrix Idx: (2, 22) Bank Idx: 24
Matrix Idx: (2, 23) Bank Idx: 25
Matrix Idx: (2, 24) Bank Idx: 26
Matrix Idx: (2, 25) Bank Idx: 27
Matrix Idx: (2, 26) Bank Idx: 28
Matrix Idx: (2, 27) Bank Idx: 29
Matrix Idx: (2, 28) Bank Idx: 30
Matrix Idx: (2, 29) Bank Idx: 31
Matrix Idx: (2, 30) Bank Idx: 0
Matrix Idx: (2, 31) Bank Idx: 1
Matrix Idx: (2, 32) Bank Idx: 2
Matrix Idx: (3, 0) Bank Idx: 3
Matrix Idx: (3, 1) Bank Idx: 4
Matrix Idx: (3, 2) Bank Idx: 5
Matrix Idx: (3, 3) Bank Idx: 6
Matrix Idx: (3, 4) Bank Idx: 7
Matrix Idx: (3, 5) Bank Idx: 8
Matrix Idx: (3, 6) Bank Idx: 9
Matrix Idx: (3, 7) Bank Idx: 10
Matrix Idx: (3, 8) Bank Idx: 11
Matrix Idx: (3, 9) Bank Idx: 12
Matrix Idx: (3, 10) Bank Idx: 13
Matrix Idx: (3, 11) Bank Idx: 14
Matrix Idx: (3, 12) Bank Idx: 15
Matrix Idx: (3, 13) Bank Idx: 16
Matrix Idx: (3, 14) Bank Idx: 17
Matrix Idx: (3, 15) Bank Idx: 18
Matrix Idx: (3, 16) Bank Idx: 19
Matrix Idx: (3, 17) Bank Idx: 20
Matrix Idx: (3, 18) Bank Idx: 21
Matrix Idx: (3, 19) Bank Idx: 22
Matrix Idx: (3, 20) Bank Idx: 23
Matrix Idx: (3, 21) Bank Idx: 24
Matrix Idx: (3, 22) Bank Idx: 25
Matrix Idx: (3, 23) Bank Idx: 26
Matrix Idx: (3, 24) Bank Idx: 27
Matrix Idx: (3, 25) Bank Idx: 28
Matrix Idx: (3, 26) Bank Idx: 29
Matrix Idx: (3, 27) Bank Idx: 30
Matrix Idx: (3, 28) Bank Idx: 31
Matrix Idx: (3, 29) Bank Idx: 0
Matrix Idx: (3, 30) Bank Idx: 1
Matrix Idx: (3, 31) Bank Idx: 2
Matrix Idx: (3, 32) Bank Idx: 3
```

这里是一个由于不适当的步长导致Memory Bank的例子。

![Memory Bank Access of Stride = 1, 2, and 3 in a Warp](https://files.mdnice.com/user/59/8d5488f1-82ea-45cc-a5ac-6d3f32ab1d19.png)

# 参考

- https://docs.nvidia.com/cuda/archive/11.6.2/cuda-c-best-practices-guide/index.html#shared-memory-and-memory-banks
- https://docs.nvidia.com/cuda/archive/11.6.2/cuda-c-programming-guide/index.html#shared-memory-5-x




