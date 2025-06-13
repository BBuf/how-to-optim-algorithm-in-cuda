> 博客来源：https://leimao.github.io/blog/CPU-Cache-False-Sharing/ ，来自Lei Mao，已获得作者转载授权。

# CPU Cache False Sharing（CPU缓存伪共享）

## 介绍

除了CPU时钟频率和核心数量之外，CPU缓存是CPU性能的另一个关键属性。例如，Intel服务器级Xeon CPU通常比旗舰级Intel桌面级Core i9 CPU具有更低的最大时钟频率，但Intel Xeon CPU通常具有更多核心和更大的缓存。因此，同一代的Intel Xeon CPU在多线程应用程序中总是比Intel Core i9 CPU具有更好的性能，当然Intel Xeon CPU的价格也要高得多。

虽然通常我们无法控制CPU如何在内存中缓存数据，但CPU遵循某些启发式规则来缓存内存。用户必须确保他们的程序以对CPU缓存友好的方式创建。如果CPU缓存行为与程序的预期行为一致，程序就能实现良好的性能。

在这篇博客文章中，我想借用Scott Meyers的CPU缓存(https://www.aristeia.com/TalkNotes/codedive-CPUCachesHandouts.pdf)伪共享示例来演示实现细节对CPU缓存和程序性能的重要性。

## CPU缓存

CPU规格可以使用Linux中的`lscpu`命令获得。在这种情况下，我的Intel i9-9900K CPU每个CPU核心都有256 KB的L1数据缓存、256 KB的L1指令缓存、2 MB的L2缓存，以及在所有CPU核心之间共享的16 MB L3缓存。

```shell
$ lscpu
Architecture:            x86_64
  CPU op-mode(s):        32-bit, 64-bit
  Address sizes:         39 bits physical, 48 bits virtual
  Byte Order:            Little Endian
CPU(s):                  16
  On-line CPU(s) list:   0-15
Vendor ID:               GenuineIntel
  Model name:            Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz
    CPU family:          6
    Model:               158
    Thread(s) per core:  2
    Core(s) per socket:  8
    Socket(s):           1
    Stepping:            12
    CPU max MHz:         5000.0000
    CPU min MHz:         800.0000
    BogoMIPS:            7200.00
    Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mc
                         a cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss 
                         ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art
                          arch_perfmon pebs bts rep_good nopl xtopology nonstop_
                         tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cp
                         l vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid ss
                         e4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes 
                         xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_f
                         ault epb invpcid_single ssbd ibrs ibpb stibp tpr_shadow
                          vnmi flexpriority ept vpid ept_ad fsgsbase tsc_adjust 
                         bmi1 avx2 smep bmi2 erms invpcid mpx rdseed adx smap cl
                         flushopt intel_pt xsaveopt xsavec xgetbv1 xsaves dtherm
                          ida arat pln pts hwp hwp_notify hwp_act_window hwp_epp
                          md_clear flush_l1d arch_capabilities
Virtualization features: 
  Virtualization:        VT-x
Caches (sum of all):     
  L1d:                   256 KiB (8 instances)
  L1i:                   256 KiB (8 instances)
  L2:                    2 MiB (8 instances)
  L3:                    16 MiB (1 instance)
NUMA:                    
  NUMA node(s):          1
  NUMA node0 CPU(s):     0-15
Vulnerabilities:         
  Itlb multihit:         KVM: Mitigation: VMX disabled
  L1tf:                  Not affected
  Mds:                   Mitigation; Clear CPU buffers; SMT vulnerable
  Meltdown:              Not affected
  Mmio stale data:       Mitigation; Clear CPU buffers; SMT vulnerable
  Retbleed:              Mitigation; IBRS
  Spec store bypass:     Mitigation; Speculative Store Bypass disabled via prctl
                          and seccomp
  Spectre v1:            Mitigation; usercopy/swapgs barriers and __user pointer
                          sanitization
  Spectre v2:            Mitigation; IBRS, IBPB conditional, RSB filling
  Srbds:                 Mitigation; Microcode
  Tsx async abort:       Mitigation; TSX disabled
```

L1缓存比L2缓存快得多，L2缓存比L3缓存快得多，L3缓存比主内存快得多。缓存访问性能与它们到CPU处理器的物理距离相关。

## CPU缓存规则

CPU缓存遵循几个规则。了解这些规则将帮助我们写出更好的代码。

- 缓存由缓存行组成，每行保存来自主内存的多个相邻字。
- 如果一个字已经在缓存中，该字将从缓存而不是主内存中读取。因此读取速度要快得多。覆写缓存中的字最终会导致主内存中该字的覆写。
- 如果一个字不在缓存中，包含该字的完整缓存行将从主内存中读取。因此读取速度较慢。
- 硬件会推测性地预取缓存行。也就是说，当CPU仍在读取当前缓存行的迭代读取指令时，下一个缓存行可能已经在缓存中准备好了。
- 如果同一缓存行被缓存在多个缓存中（属于不同的CPU核心），当任何一个缓存行被覆写时（由一个线程），所有缓存行都会失效，需要在覆写反映到主内存后再次从主内存读取。

## 伪共享

以下示例是Scott Meyers在CppCon2014上的演讲"CPU Caches and Why You Care"(https://youtu.be/WDIkqP4JbkE)中伪共享伪代码示例(https://www.aristeia.com/TalkNotes/codedive-CPUCachesHandouts.pdf)的C++实现。它演示了同一缓存行如何在几乎每次迭代中在多个缓存中失效，多线程应用程序的性能受到严重影响。解决这个问题的方法也很简单优雅。

```c++
#include <cassert>
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

template <typename T>
class Matrix
{
public:
    Matrix(size_t m, size_t n) : m_num_rows{m}, m_num_cols{n}, m_data(m * n){};
    T& operator()(size_t i, size_t j) noexcept
    {
        return m_data[i * m_num_cols + j];
    }
    T operator()(size_t i, size_t j) const noexcept
    {
        return m_data[i * m_num_cols + j];
    }
    // Enable mat[i][j]
    T* operator[](size_t i) noexcept { return &m_data[i * m_num_cols]; }
    T const* operator[](size_t i) const noexcept
    {
        return &m_data[i * m_num_cols];
    }
    size_t get_num_rows() const noexcept { return m_num_rows; };
    size_t get_num_cols() const noexcept { return m_num_cols; };
    T* data() noexcept { return m_data.data(); }
    T const* data() const noexcept { return m_data.data(); }

private:
    size_t m_num_rows;
    size_t m_num_cols;
    std::vector<T> m_data;
};

template <class T>
float measure_performance(std::function<T(void)> bound_function,
                          int num_repeats = 100, int num_warmups = 100)
{
    for (int i{0}; i < num_warmups; ++i)
    {
        bound_function();
    }

    std::chrono::steady_clock::time_point time_start{
        std::chrono::steady_clock::now()};
    for (int i{0}; i < num_repeats; ++i)
    {
        bound_function();
    }
    std::chrono::steady_clock::time_point time_end{
        std::chrono::steady_clock::now()};

    auto const time_elapsed{
        std::chrono::duration_cast<std::chrono::milliseconds>(time_end -
                                                              time_start)
            .count()};
    float const latency{time_elapsed / static_cast<float>(num_repeats)};

    return latency;
}

void random_initialize_matrix(Matrix<int>& mat, unsigned int seed)
{
    size_t const num_rows{mat.get_num_rows()};
    size_t const num_cols{mat.get_num_cols()};
    std::default_random_engine e(seed);
    std::uniform_int_distribution<int> uniform_dist(-1024, 1024);
    for (size_t i{0}; i < num_rows; ++i)
    {
        for (size_t j{0}; j < num_cols; ++j)
        {
            mat[i][j] = uniform_dist(e);
        }
    }
}

size_t count_odd_values_row_major(Matrix<int> const& mat)
{
    size_t num_odd_values{0};
    size_t const num_rows{mat.get_num_rows()};
    size_t const num_cols{mat.get_num_cols()};
    for (size_t i{0}; i < num_rows; ++i)
    {
        for (size_t j{0}; j < num_cols; ++j)
        {
            if (mat[i][j] % 2 != 0)
            {
                ++num_odd_values;
            }
        }
    }
    return num_odd_values;
}

size_t count_odd_values_column_major(Matrix<int> const& mat)
{
    size_t num_odd_values{0};
    size_t const num_rows{mat.get_num_rows()};
    size_t const num_cols{mat.get_num_cols()};
    for (size_t j{0}; j < num_cols; ++j)
    {
        for (size_t i{0}; i < num_rows; ++i)
        {
            if (mat[i][j] % 2 != 0)
            {
                ++num_odd_values;
            }
        }
    }
    return num_odd_values;
}

size_t
multi_thread_count_odd_values_row_major_non_scalable(Matrix<int> const& mat,
                                                     size_t num_threads)
{
    std::vector<std::thread> workers{};
    workers.reserve(num_threads);
    size_t const num_rows{mat.get_num_rows()};
    size_t const num_cols{mat.get_num_cols()};
    size_t const num_elements{num_rows * num_cols};
    size_t const trunk_size{(num_elements + num_threads - 1) / num_threads};

    std::vector<size_t> results(num_threads, 0);
    for (size_t i{0}; i < num_threads; ++i)
    {
        workers.emplace_back(
            [&, i]()
            {
                size_t const start_pos{i * trunk_size};
                size_t const end_pos{
                    std::min((i + 1) * trunk_size, num_elements)};
                for (size_t j{start_pos}; j < end_pos; ++j)
                {
                    if (mat.data()[j] % 2 != 0)
                    {
                        // False sharing
                        // The array is shared across multiple different
                        // threads. A consecutive piece of the array that
                        // contains the entry will be cached in CPU for reading
                        // in a thread. Multiple threads have multiple caches of
                        // the same content. However, writing to the array on
                        // main memory invalidates the all these caches that are
                        // have the same content. The CPU would have to read
                        // from the main memory for the updated entry and
                        // re-cache the array. This slows down the performance
                        // significantly.
                        ++results[i];
                    }
                }
            });
    }
    for (int i{0}; i < num_threads; ++i)
    {
        workers[i].join();
    }

    size_t num_odd_values{0};
    for (int i{0}; i < num_threads; ++i)
    {
        num_odd_values += results[i];
    }

    return num_odd_values;
}

size_t multi_thread_count_odd_values_row_major_scalable(Matrix<int> const& mat,
                                                        size_t num_threads)
{
    std::vector<std::thread> workers{};
    workers.reserve(num_threads);
    size_t const num_rows{mat.get_num_rows()};
    size_t const num_cols{mat.get_num_cols()};
    size_t const num_elements{num_rows * num_cols};
    size_t const trunk_size{(num_elements + num_threads - 1) / num_threads};

    std::vector<size_t> results(num_threads, 0);
    for (size_t i{0}; i < num_threads; ++i)
    {
        workers.emplace_back(
            [&, i]()
            {
                size_t count = 0;
                size_t const start_pos{i * trunk_size};
                size_t const end_pos{
                    std::min((i + 1) * trunk_size, num_elements)};
                for (size_t j{start_pos}; j < end_pos; ++j)
                {
                    if (mat.data()[j] % 2 != 0)
                    {
                        ++count;
                    }
                }
                results[i] = count;
            });
    }
    for (int i{0}; i < num_threads; ++i)
    {
        workers[i].join();
    }

    size_t num_odd_values{0};
    for (int i{0}; i < num_threads; ++i)
    {
        num_odd_values += results[i];
    }

    return num_odd_values;
}

int main()
{
    unsigned int const seed{0U};
    int const num_repeats{100};
    int const num_warmups{100};

    size_t const num_threads{8};

    float latency{0};

    Matrix<int> mat(1000, 2000);
    random_initialize_matrix(mat, seed);

    assert(count_odd_values_row_major(mat) ==
           count_odd_values_column_major(mat));

    assert(
        count_odd_values_row_major(mat) ==
        multi_thread_count_odd_values_row_major_non_scalable(mat, num_threads));

    assert(count_odd_values_row_major(mat) ==
           multi_thread_count_odd_values_row_major_scalable(mat, num_threads));

    std::function<size_t(void)> const function_1{
        std::bind(count_odd_values_row_major, mat)};
    std::function<size_t(void)> const function_2{
        std::bind(count_odd_values_column_major, mat)};
    std::function<size_t(void)> const function_3{
        std::bind(multi_thread_count_odd_values_row_major_non_scalable, mat,
                  num_threads)};
    std::function<size_t(void)> const function_4{std::bind(
        multi_thread_count_odd_values_row_major_scalable, mat, num_threads)};

    latency = measure_performance(function_1, num_repeats, num_warmups);
    std::cout << "Single-Thread Row-Major Traversal" << std::endl;
    std::cout << std::fixed << std::setprecision(3) << "Latency: " << latency
              << " ms" << std::endl;

    latency = measure_performance(function_2, num_repeats, num_warmups);
    std::cout << "Single-Thread Column-Major Traversal" << std::endl;
    std::cout << std::fixed << std::setprecision(3) << "Latency: " << latency
              << " ms" << std::endl;

    latency = measure_performance(function_3, num_repeats, num_warmups);
    std::cout << num_threads << "-Thread Row-Major Non-Scalable Traversal"
              << std::endl;
    std::cout << std::fixed << std::setprecision(3) << "Latency: " << latency
              << " ms" << std::endl;

    latency = measure_performance(function_4, num_repeats, num_warmups);
    std::cout << num_threads << "-Thread Row-Major Scalable Traversal"
              << std::endl;
    std::cout << std::fixed << std::setprecision(3) << "Latency: " << latency
              << " ms" << std::endl;
}
```

从延迟测量结果中我们可以看到，由于伪共享，8线程遍历的性能非常糟糕，几乎不比单线程遍历好。通过移除几乎所有的伪共享（除了最后写入输出数组），8线程遍历的性能显著提升到其理论值。

```shell
$ g++ false_sharing.cpp -o false_sharing -lpthread -std=c++14
$ ./false_sharing 
Single-Thread Row-Major Traversal
Latency: 16.840 ms
Single-Thread Column-Major Traversal
Latency: 20.210 ms
8-Thread Row-Major Non-Scalable Traversal
Latency: 16.520 ms
8-Thread Row-Major Scalable Traversal
Latency: 2.740 ms
```

## CPU缓存 VS GPU缓存

CPU缓存和GPU缓存之间有很多相似之处。

例如，为了改善GPU从全局内存的内存IO，我们希望内存访问是合并的，这样缓存行中的所有条目都可以被GPU线程使用。

此外，如果许多线程在迭代中读取和写入全局内存，GPU上也可能出现伪共享。要解决GPU上的伪共享问题，类似于CPU的解决方案，我们使用局部变量或共享内存来存储中间值，并且只在算法结束时从局部变量或共享内存向全局内存写入一次。

## 参考资料

- CPU Caches and Why You Care - Scott Meyers(https://www.aristeia.com/TalkNotes/codedive-CPUCachesHandouts.pdf)

