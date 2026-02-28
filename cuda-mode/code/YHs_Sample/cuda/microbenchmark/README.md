**dram | l2cache_bandwidth.cu:**
&ensp;dram | l2 cache bandwidth benchmark.

**dram | l2cache | smem_latency.cu:**
&ensp;dram | l2 cache | shared memory latency benchmark.

support Kepler+(sm_30+) devices

---

build:

```bash
# for volta GPU:
$ sh build.sh dram|l2cache_bandwidth.cu 70
# or
$ sh build.sh dram|l2cache|l1cache|smem_latency.cu 70

#for turing GPU:
$ sh build.sh dram|l2cache_bandwidth.cu 75
# or
$ sh build.sh dram|l2cache|l1cache|smem_latency.cu 75

# for GA100 GPU:
$ sh build.sh dram|l2cache_bandwidth.cu 80
# or
$ sh build.sh dram|l2cache|l1cache|smem_latency.cu 80

# for GA10x(x >= 2) GPU:
$ sh build.sh dram|l2cache_bandwidth.cu 86
# or
$ sh build.sh dram|l2cache|l1cache|smem_latency.cu 86

```

run:

```bash
$ ./a.out
```
