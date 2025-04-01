# 0x0. 前言

上周六的时候发现 @DiegoD94 在 vLLM 中尝试对DeepSeek V3/R1 应用一个fuse shared experts到普通256个expert中的工作 (https://github.com/vllm-project/vllm/pull/15502)。还有一个技术文档：https://docs.google.com/document/d/1iXgzR6Mt6s0DpT7w2Pz93ExlUJ-nnSnU_o9Sqd8TE34/edit?tab=t.0#heading=h.w2t5rj3ovrvv ，读了一下感觉比较有意义并且看起来对整体的吞吐和TTFT/ITL都有比较好的收益。所以利用周末时间在SGLang中实现了一下这个工作，由于我们之前在SGLang的sgl moe_align_kernel 中已经支持了num_experts>256的情况，所以本次实现起来比较方便，不需要修改sgl-kernel里面的cuda代码。在SGLang中的细节如下：https://github.com/sgl-project/sglang/pull/4918 。下面来讲一下这个fuse kernel技巧，另外再次致谢 @DiegoD94

# 0x1. 效果

下面展示一下在SGLang中的端到端效果：


### Acc in H200

```shell
➜  sglang git:(support_r1_shared_expers_fusion) ✗ python3 benchmark/gsm8k/bench_sglang.py --num-questions 2000 --parallel 2000 --num-shots 8                                            
100%|████████████████████████████████████████████████████████████████████████| 1319/1319 [01:08<00:00, 19.14it/s]
Accuracy: 0.952
Invalid: 0.000
Latency: 69.547 s
Output throughput: 1998.856 token/s
```

### Benchmark in H200

#### qps=1

```shell
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code --port 30000 --disable-shared-experts-fusion
python3 -m sglang.bench_serving --backend sglang --num-prompts 300 --request-rate 1

============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    1.0       
Max reqeuest concurrency:                not set   
Successful requests:                     300       
Benchmark duration (s):                  317.08    
Total input tokens:                      95293     
Total generated tokens:                  60411     
Total generated tokens (retokenized):    60140     
Request throughput (req/s):              0.95      
Input token throughput (tok/s):          300.53    
Output token throughput (tok/s):         190.52    
Total token throughput (tok/s):          491.05    
Concurrency:                             7.18      
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   7586.12   
Median E2E Latency (ms):                 4788.43   
---------------Time to First Token----------------
Mean TTFT (ms):                          402.01    
Median TTFT (ms):                        297.76    
P99 TTFT (ms):                           1634.92   
---------------Inter-Token Latency----------------
Mean ITL (ms):                           35.95     
Median ITL (ms):                         28.17     
P95 ITL (ms):                            36.35     
P99 ITL (ms):                            239.92    
Max ITL (ms):                            1722.10   
==================================================

python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code --port 30000
python3 -m sglang.bench_serving --backend sglang --num-prompts 300 --request-rate 1

============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    1.0       
Max reqeuest concurrency:                not set   
Successful requests:                     300       
Benchmark duration (s):                  316.52    
Total input tokens:                      95293     
Total generated tokens:                  60411     
Total generated tokens (retokenized):    58378     
Request throughput (req/s):              0.95      
Input token throughput (tok/s):          301.06    
Output token throughput (tok/s):         190.86    
Total token throughput (tok/s):          491.92    
Concurrency:                             6.43      
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   6789.19   
Median E2E Latency (ms):                 4294.82   
---------------Time to First Token----------------
Mean TTFT (ms):                          337.58    
Median TTFT (ms):                        280.56    
P99 TTFT (ms):                           1241.48   
---------------Inter-Token Latency----------------
Mean ITL (ms):                           32.32     
Median ITL (ms):                         25.01     
P95 ITL (ms):                            35.01     
P99 ITL (ms):                            218.08    
Max ITL (ms):                            1447.12   
```

- ttft
  - main: 402.01 ms
  - pr: 337.58 ms(19%+)
- itl
  -  main: 35.95
  - pr: 32.33 (11%+)

#### qps=4

```shell
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code --port 30000 --disable-shared-experts-fusion
python3 -m sglang.bench_serving --backend sglang --num-prompts 300 --request-rate 4

============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    4.0       
Max reqeuest concurrency:                not set   
Successful requests:                     300       
Benchmark duration (s):                  120.67    
Total input tokens:                      95293     
Total generated tokens:                  60411     
Total generated tokens (retokenized):    60142     
Request throughput (req/s):              2.49      
Input token throughput (tok/s):          789.68    
Output token throughput (tok/s):         500.61    
Total token throughput (tok/s):          1290.29   
Concurrency:                             52.17     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   20983.88  
Median E2E Latency (ms):                 15341.68  
---------------Time to First Token----------------
Mean TTFT (ms):                          714.68    
Median TTFT (ms):                        484.26    
P99 TTFT (ms):                           3285.06   
---------------Inter-Token Latency----------------
Mean ITL (ms):                           101.44    
Median ITL (ms):                         49.72     
P95 ITL (ms):                            314.42    
P99 ITL (ms):                            830.43    
Max ITL (ms):                            3630.63   
==================================================

python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code --port 30000
python3 -m sglang.bench_serving --backend sglang --num-prompts 300 --request-rate 4

============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    4.0       
Max reqeuest concurrency:                not set   
Successful requests:                     300       
Benchmark duration (s):                  114.30    
Total input tokens:                      95293     
Total generated tokens:                  60411     
Total generated tokens (retokenized):    57835     
Request throughput (req/s):              2.62      
Input token throughput (tok/s):          833.73    
Output token throughput (tok/s):         528.54    
Total token throughput (tok/s):          1362.27   
Concurrency:                             44.73     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   17041.89  
Median E2E Latency (ms):                 10927.11  
---------------Time to First Token----------------
Mean TTFT (ms):                          584.40    
Median TTFT (ms):                        409.48    
P99 TTFT (ms):                           3088.26   
---------------Inter-Token Latency----------------
Mean ITL (ms):                           82.46     
Median ITL (ms):                         42.90     
P95 ITL (ms):                            236.51    
P99 ITL (ms):                            486.58    
Max ITL (ms):                            2506.70   
==================================================
```

- Total token throughput (tok/s)
  - main: 1290.29   
  - pr: 1362.27  (5.5%)
- ttft
  - main: 714.68 ms
  - pr: 584.40  ms(22%+)
- itl
  -  main: 101.44ms
  - pr: 82.46ms (23%+)


#### qps=8

```shell
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code --port 30000 --disable-shared-experts-fusion
python3 -m sglang.bench_serving --backend sglang --num-prompts 300 --request-rate 8

============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    8.0       
Max reqeuest concurrency:                not set   
Successful requests:                     300       
Benchmark duration (s):                  107.68    
Total input tokens:                      95293     
Total generated tokens:                  60411     
Total generated tokens (retokenized):    60152     
Request throughput (req/s):              2.79      
Input token throughput (tok/s):          884.97    
Output token throughput (tok/s):         561.02    
Total token throughput (tok/s):          1445.99   
Concurrency:                             87.49     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   31403.22  
Median E2E Latency (ms):                 31011.29  
---------------Time to First Token----------------
Mean TTFT (ms):                          1288.44   
Median TTFT (ms):                        755.54    
P99 TTFT (ms):                           6255.21   
---------------Inter-Token Latency----------------
Mean ITL (ms):                           150.65    
Median ITL (ms):                         52.92     
P95 ITL (ms):                            463.60    
P99 ITL (ms):                            1724.48   
Max ITL (ms):                            7765.48   
==================================================

python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code --port 30000
python3 -m sglang.bench_serving --backend sglang --num-prompts 300 --request-rate 8

============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    8.0       
Max reqeuest concurrency:                not set   
Successful requests:                     300       
Benchmark duration (s):                  100.08    
Total input tokens:                      95293     
Total generated tokens:                  60411     
Total generated tokens (retokenized):    57851     
Request throughput (req/s):              3.00      
Input token throughput (tok/s):          952.14    
Output token throughput (tok/s):         603.61    
Total token throughput (tok/s):          1555.75   
Concurrency:                             79.64     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   26567.46  
Median E2E Latency (ms):                 25277.52  
---------------Time to First Token----------------
Mean TTFT (ms):                          1214.58   
Median TTFT (ms):                        599.49    
P99 TTFT (ms):                           5410.74   
---------------Inter-Token Latency----------------
Mean ITL (ms):                           127.02    
Median ITL (ms):                         48.13     
P95 ITL (ms):                            419.52    
P99 ITL (ms):                            1493.97   
Max ITL (ms):                            5000.62   
==================================================
```

- Total token throughput (tok/s)
  - main: 1445.99   
  - pr: 1555.75（7.5%）
- ttft
  - main: 1288.44  ms
  - pr:  1214.58  ms(6%)
- itl
  -  main: 150.65 ms
  - pr: 127.02 ms(18.6%) 


总结下就是，启用此优化后随着并发增大可以提升5%左右端到端的吞吐，并且TTFT和ITL普遍可以提升15%以上。

# 0x2. 原理

![](https://files.mdnice.com/user/59/0aff5839-70da-41bd-9cbf-4a5c459bd164.png)

如图1所示，DeepSeek的MoE结构会将所有输入的token都送入共享专家(shared experts)，同时也会将每个token路由到由路由器选择的top-k个专家(routed experts)中，最后基于权重将所有专家的输出进行聚合。具体来说，DeepSeek R1使用了256个路由专家和1个共享专家，对于每个token，它会选择top 8个路由专家。在VLLM和SGLang的实现中，token的隐藏状态首先通过共享专家，然后再通过FusedMoE kernel中的路由专家。最后将这两个输出相加作为最终输出。

一个简单的优化方法是：我们可以将共享专家的计算合并到FusedMoE kernel中，因为共享专家和路由专家具有完全相同的架构和形状。这样，对于每个token，我们不再是从256个专家中选择top 8个，再单独选择1个共享专家，而是直接从257个专家(256个路由专家+1个共享专家)中选择top 9个，并且始终将第9个专家设置为共享专家。通过进一步调整这9个选定专家的聚合权重，我们可以得到与优化前的MoE层完全相同的输出结果。

这里的意思是，在原始的Deepseek V3/R1 MoE forward代码中：

```python
def forward_normal(self, hidden_states: torch.Tensor) -> torch.Tensor:
    if self.n_shared_experts is not None:
        shared_output = self.shared_experts(hidden_states)
    # router_logits: (num_tokens, n_experts)
    router_logits = self.gate(hidden_states)
    final_hidden_states = (
        self.experts(hidden_states=hidden_states, router_logits=router_logits)
        * self.routed_scaling_factor
    )
    if shared_output is not None:
        final_hidden_states = final_hidden_states + shared_output
    if self.tp_size > 1:
        final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
    return final_hidden_states
```

普通的experts的计算结果需要乘以 `self.routed_scaling_factor` ，为了把shared expert融合到普通的expert中一起计算，我们需要在grouped_topk模块中把`top9`也就是shared experts对应的`topk_weights`提前除一下这个`self.routed_scaling_factor` 参数，这样就可以保证数值等价，请看下面的关于`grouped_topk`中的修改。

![](https://files.mdnice.com/user/59/3e561405-beb6-4c1a-9d9f-4cf07168a2ed.png)


# 0x3. 实现细节

除了上面提到的为了保持数值等价做的Topk weights的系数修改之外，我们还可以观察到我们始终将所有token的第9个专家分配给共享专家，并且这里的共享专家可能还不止一个。对应这行代码：

```python
topk_ids[:, -1] = torch.randint(
    low=num_experts,
    high=num_experts + n_share_experts_fusion,
    size=(topk_ids.size(0),),
    dtype=topk_ids.dtype,
    device=topk_ids.device,
)
```

之所以设定多个共享专家，原因可能是可以对token做一个类似负载均衡的策略，防止一个共享专家分配到所有的token导致Expert间的token数差距过大导致计算效率低的问题。

我tuning并Benchmark了一下不使用任何额外复制的shared experts的情况，发现这种情况的性能确实比使用tp_size个shared experts复制的情况要糟糕，TTFT甚至会延长。具体可以看这里的benchmark数据：https://github.com/sgl-project/sglang/pull/4918

要支持复制多个shared experts要修改一下`DeepseekV2ForCausalLM`类的`load_weights`函数：

![](https://files.mdnice.com/user/59/26ac5207-a88c-4d88-95c2-c37754bd4b29.png)

其它还有需要的注意的细节是由于expert个数有变化，所以还需要使用tuning fused moe脚本来针对tuning一下。此外，相比于原始的非fuse实现版本，每个tp rank上的显存会增加，对于FP8 dtype来说，一个rank上新增一个shared expert。一个shared expert 按照FP8 dtype折算为42MB内存（总参数量 = hidden_size  moe_intermediate_size * 3 = 7168 * 2048 * 3），然后（51-3）*42MB=2016MB/1024=1.96GB。也就是说这个优化将在TP Rank上多使用1-2GB的内存才可以获得最佳性能收益。







