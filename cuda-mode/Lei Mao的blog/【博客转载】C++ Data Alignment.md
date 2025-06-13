> 博客来源：https://leimao.github.io/blog/CPP-Data-Alignment/ ，来自Lei Mao，已获得作者转载授权。

# C++ 数据对齐

## 简介

数据对齐是现代计算机硬件计算中的一个关键特性。当数据自然对齐时，CPU读取和写入内存的效率最高，这通常意味着数据的内存地址是数据大小的倍数。例如，在32位架构中，如果数据存储在四个连续字节中，且第一个字节位于4字节边界上，则该数据可能是对齐的。

除了性能之外，数据对齐也是许多编程语言的假设条件。尽管编程语言尽可能地为我们处理数据对齐问题，但一些低级编程语言可能会出现未对齐的数据访问，而这种行为是未定义的。

在这篇博客文章中，我想快速讨论数据对齐，包括对齐的内存地址和对齐的内存访问，以及如何在C++中尽可能确保数据对齐。

## 数据对齐

当内存地址$a$是$n$的倍数时（其中$n$是2的幂），我们说内存地址$a$是$n$字节对齐的。假设我们有一块$m$字节的数据和一个$n$字节对齐的地址。如果$m$不能被$n$整除，那么$m$字节的数据将被填充到$\lceil\frac{m+n-1}{n}\rceil \times n$字节的数据。

访问$kn + 1, kn + 2, \cdots, (k + 1)n$字节的数据都有相同的延迟，因为CPU每次从内存中读取$n$字节的数据，这些数据通常会被缓存在CPU中。也就是说，如果数据存储在$n$字节对齐的地址上，其存储大小$m$不是$n$的倍数，那么一些内存访问带宽会被浪费。

当被访问的数据长度为$n$字节且数据地址是$n$字节对齐时，我们说这种内存访问是对齐的。当内存访问不对齐时，我们说它是未对齐的。请注意，根据定义，单字节内存访问总是对齐的。理论上，可以在不是$n$的倍数的内存地址上访问$n$字节的数据，但这会浪费更多的内存访问带宽。但是，由于C和C++标准假设内存访问是对齐的，访问未对齐的地址可能导致未定义的行为。

## 数据对齐要求

`alignof`可以用来检查特定数据类型的对齐要求。

```c++
#include <cassert>

struct float4_4_t
{
    float data[4];
};

// float4_32_t类型的每个对象都将对齐到32字节边界。
// 可能对SIMD指令有用。
struct alignas(32) float4_32_t
{
    float data[4];
};

// 比同一声明上的另一个alignas更弱的有效非零对齐会被忽略。
struct alignas(1) float4_1_t
{
    float data[4];
};

// 访问对象会导致未定义行为。
// 1字节结构成员对齐。
// size = 32, alignment = 1字节，这些结构成员没有填充。
// 这是不规范的，因为float需要4字节对齐。
#pragma pack(push, 1)
struct alignas(1) float4_1_ub_t
{
    float data[4];
};
#pragma pack(pop)

int main()
{
    assert(alignof(float4_4_t) == 4);
    assert(alignof(float4_32_t) == 32);
    assert(alignof(float4_1_t) == 4);
    assert(alignof(float4_1_ub_t) == 1);

    assert(sizeof(float4_4_t) == 16);
    assert(sizeof(float4_32_t) == 32);
    assert(sizeof(float4_1_t) == 16);
    assert(sizeof(float4_1_ub_t) == 16);
}
```

## 内存分配

根据GNU文档(https://www.gnu.org/software/libc/manual/html_node/Aligned-Memory-Blocks.html)，在GNU系统中，`malloc`或`realloc`返回的块地址总是8的倍数（在64位系统上是16的倍数）。数组的默认内存地址对齐由元素的对齐要求决定。

可以为分配的静态内存和动态内存使用自定义数据对齐。`alignas(T)`可以用来指定静态数组的字节对齐，`aligned_alloc`可以用来指定动态内存上缓冲区的字节对齐。

```c++
#include <cstdio>
#include <cstdlib>
#include <iostream>

int main()
{
    unsigned char buf1[sizeof(int) / sizeof(char)];
    std::cout << "默认 "
              << alignof(unsigned char[sizeof(int) / sizeof(char)]) << "字节"
              << " 对齐地址: " << static_cast<void*>(buf1) << std::endl;
    std::cout << reinterpret_cast<uintptr_t>(buf1) %
                     alignof(unsigned char[sizeof(int) / sizeof(char)])
              << std::endl;
    std::cout << reinterpret_cast<uintptr_t>(buf1) % alignof(int) << std::endl;

    alignas(int) unsigned char buf2[sizeof(int) / sizeof(char)];
    std::cout << alignof(int)
              << "字节对齐地址: " << static_cast<void*>(buf2)
              << std::endl;
    std::cout << reinterpret_cast<uintptr_t>(buf2) %
                     alignof(unsigned char[sizeof(int) / sizeof(char)])
              << std::endl;
    std::cout << reinterpret_cast<uintptr_t>(buf2) % alignof(int) << std::endl;

    void* p1 = malloc(sizeof(int));
    std::cout << "默认 "
              << "16字节"
              << " 对齐地址: " << p1 << std::endl;
    std::cout << reinterpret_cast<uintptr_t>(p1) % 16 << std::endl;
    std::cout << reinterpret_cast<uintptr_t>(p1) % 1024 << std::endl;
    free(p1);

    void* p2 = aligned_alloc(1024, sizeof(int));
    std::cout << "1024字节对齐地址: " << p2 << std::endl;
    std::cout << reinterpret_cast<uintptr_t>(p2) % 16 << std::endl;
    std::cout << reinterpret_cast<uintptr_t>(p2) % 1024 << std::endl;
    free(p2);
}
```

```shell
$ g++ alloc.cpp -o alloc -std=c++11
$ ./alloc
Default 1-byte aligned addr: 0x7ffd46d76304
0
0
4-byte aligned addr: 0x7ffd46d76300
0
0
Default 16-byte aligned addr: 0x559a6e1c42c0
0
704
1024-byte aligned addr: 0x559a6e1c4400
0
0
```

## 未定义行为

如果数据对齐不正确，向静态数组或动态缓冲区写入数据可能导致未定义行为。例如，如果我们在`unsigned char buf[sizeof(T) / sizeof(char)]`上创建类型T的对象，可能会发生读写的未定义行为，特别是使用`reinterpret_cast`和未对齐的内存地址增量时。对于使用`malloc`分配的动态缓冲区上创建对象T也是如此。但是，由于`malloc`返回的地址在32位架构上是8字节对齐的，在64位架构上是16字节对齐的，8字节和16字节对齐可以被几乎所有的数据满足，特别是基本类型。因此不太可能发生未定义行为。

例如，以下数据结构`Bar`有`sizeof(Bar) == 6`和`alignof(Bar) == 2`。

```c++
struct Bar
{
    char arr[3];    // 3字节 + 1个填充字节
    short s;        // 2字节
};
```

对齐要求总是数据结构中每个成员的最大对齐要求。在现代计算机中，它必须是2的幂。在数据结构`Bar`的情况下，`alignof(char) == 1`和`alignof(short) == 2`。因此，`sizeof(Bar) == max(alignof(char), alignof(short)) == 2`。

如果内存中的`Bar`对象是2字节对齐的，访问其需要1字节要求的`char`类型属性会自动满足。由于填充字节的存在，访问其需要2字节要求的`short`类型属性也会自动满足。

在我的x86-64架构计算机上，我可以使用`malloc(sizeof(Bar))`，返回的指针将是16字节对齐的地址，这满足了`Bar`数据结构的2字节对齐要求。

为了完全确保不会因数据对齐而引起未定义行为，我们应该使用`T buf[N]`、`alignas(T) unsigned char buf[N * sizeof(T) / sizeof(char)]`和`aligned_alloc(alignas(T), N * sizeof(T))`来分配内存，然后在其上创建类型`T`的对象。这也意味着当编译器生成数据结构时，`sizeof(T)`必须是`alignas(T)`的倍数。否则，从数组中的第二个元素开始，它们可能开始变得未对齐。

## 结论

手动确保数据对齐容易出错。因此，对于大多数使用情况，我们应该尝试使用高级接口函数和STL容器，如`new`和`delete`以及`std::vector`，用于动态内存分配，并通过减少危险的指针类型转换来确保类型安全。

## 参考文献

- Data Alignment(https://www.songho.ca/misc/alignment/dataalign.html)
- alignas specifier(en.cppreference.com/w/cpp/language/alignas)
- std::aligned_alloc(https://en.cppreference.com/w/cpp/memory/c/aligned_alloc)



