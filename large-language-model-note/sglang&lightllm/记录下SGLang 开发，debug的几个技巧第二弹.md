# 0x0. 前言

几个月前在《记录下SGLang 开发，编译和Profile的几个小技巧》(https://zhuanlan.zhihu.com/p/1939041055208112436) 分享了SGLang开发中的几个技巧。随着时间推移，又获得了一些新的发现和知识，继续给网友分享下。如果对你有用，欢迎点个赞。我的cuda学习笔记：https://github.com/BBuf/how-to-optim-algorithm-in-cuda

# 0x1. 开启多线程模型权重加载

在单机8卡或者2机16卡这种规模下（如H200， B200）调试DeepSeek V3/R1/V3.1/V3.2，Kimi K2 Thinking，Mistral3 Large 这种大MoE模型会很有用，通过在server args中添加 `--model-loader-extra-config='{"enable_multithread_load": "true","num_threads": 64}'` 这个参数可以多线程加载模型，极大提升模型加载的速度，告别加载一次模型就需要玩一把游戏之后再来看的窘境。

![](https://files.mdnice.com/user/59/c600b561-45a7-413c-9085-6038df3217a7.png)

# 0x2. Debug SGLang Serve模型短期和长期崩溃

当我们用SGLang Server模型的时候如果碰到了崩溃（比如Illegal memory access, Warp Illegal Instruction）的情况应该怎么做呢？这里可以分成2种情况，第一种是服务启动之后，固定发某个请求，例如测试gsm8k数据集就出现了崩溃。另外一种可能服务已经跑了1天了，这个时候才崩溃。

## 0x2.1 立即崩溃

第一种情况复现起来相对简单，可能相对好处理一点。下面分享一个例子，我不保证分析都是对的。我们在开Piecewise CUDA Graph 并 Serving Kimi K2 Thinking 的时候发现服务起来之后测试gsm8k就会崩掉，然后报了 Warp Illegal Instruction 的错误。由于CUDA异步特性，我们看到的报错的kernel可能并不是真实报错的kernel，所以我们需要借助CUDA-GDB等工具。

例如启动命令加一圈环境变量：

```shell
CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 CUDA_COREDUMP_SHOW_PROGRESS=1 CUDA_COREDUMP_GENERATION_FLAGS='skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory' CUDA_COREDUMP_FILE="/home/sglang/bbuf/dump/cuda_coredump_%h.%p.%t" python3 -m sglang.launch_server --model-path moonshotai/Kimi-K2-Thinking --tp 8 --trust-remote-code  --tool-call-parser kimi_k2 --reasoning-parser kimi_k2 --model-loader-extra-config='{"enable_multithread_load": "true","num_threads": 64}' --enable-piecewise-cuda-graph
```

这些环境变量可以参考这篇知乎blog：https://zhuanlan.zhihu.com/p/1938284538108293922 或者 vLLM 的博客 [从GPU卡死到精准锁定出错代码：vLLM CUDA 调试实战技巧](https://mp.weixin.qq.com/s/VHFnA9nkasOJ-svIFp7IXQ) 

然后再次运行服务触发崩溃，这个时候得到的报错信息就比较真了。截图：

![](https://files.mdnice.com/user/59/d23684bf-f177-4f6d-baff-6de5bc04ec0d.png)


这个时候可以看到报错的kernel，以及在哪个block的哪个线程，重要的是有报错的地址。然后通过CUDA-GDB的`x/10i `指令获取报错地址附近的SASS：

![](https://files.mdnice.com/user/59/5542b53a-1760-4ea5-93b4-4276024be7ca.png)

对于我这种CUDA低玩就可以借助一下AI了，AI帮我分析下SASS，然后给出可能的错误原因。

![](https://files.mdnice.com/user/59/0b1809e6-3b35-4ef9-9baf-2aad73584bd0.png)

然后线索就指向了 https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/moe/moe_align_kernel.cu#L98 这行代码，这里的Shared Memory可能发生了越界访问，最终导致了Warp Illegal Instruction。然后我们尝试在这里加了一个assert，把expert_id限制为`>=0`和`<num_experts`，上面的错误果然消失了然后GSM8K也可以正常跑了。

但是回头想一下，这个expert_id天然就应该是满足`>=0`和`<num_experts`。你再去问AI，它肯定会说，哦，你是对的，然后给你错误的分析了。

接着就开始思考，这个topk_ids的值怎么会越界呢？它的来源并不在这个moe_align_block_size_kernel的kernel，肯定是在它的上一个kernel，那它的上一个kernel是什么？由于serving的是Kimi K2 Thinking模型，所以产生topk_ids的上一个kernel肯定是kimi_k2_moe_fused_gate_kernel 。这个kernel恰好是不久之前添加的，里面对topk_ids没有进行初始化可能产生了垃圾输入，然后快速修改了初始化之后再验证确实可以解决这个问题。moe_align_block_size的kernel也不用做任何限制了。

![](https://files.mdnice.com/user/59/144ad6a7-b4d6-4a59-b310-e1a5dc064eff.png)

这里的关键之处在于对于轻易复现崩溃的bug，我们可以用cuda-gdb定位挂掉的地方。但是一定要注意不要用AI辅助分析或者伪解决之后忘记掌控debug全局的是我们自己，因为只有我们才有真正的完整上下文。


## 0x2.2 长期崩溃

如果我们的服务跑了1天甚至几天才崩溃怎么办？

第一种方法是我们每天晚上重启一次，这样就不崩了。（逃

第二种是我们SGLang提供的正经复现方式，我们可以启动服务的时候加上 https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py#L2912 这个 --crash-dump-folder 参数。它会记录长期崩溃发生时间点前5分钟的请求。

![](https://files.mdnice.com/user/59/25fd5ca8-4256-48c4-8f6f-99df4a3f6149.png)

这个数据会被保存为一个`.pkl`文件。

然后我们就可以把上面那些cuda-gdb变量都加上重新启动模型，然后执行下面的命令来replay事故现场:

```shell
python3 scripts/playground/replay_request_dump.py --input-file crash_dump.pkl
```

我们通过这种方法解决了一个真实场景下Serving Kimi K2 Thinking 的24小时之后才崩溃的问题，还是值得一试的。

## 0x2.3 勇于求助

当你定位到问题之后发现这个kernel你不懂，建议把ISSUE提出来或者找到这个kernel的开发者，只有他们才拥有完整的上下文，不会轻易陷入局部的幻觉。

# 0x3. 逐层 NVTX Profiling

使用 `--enable-layerwise-nvtx-marker` 可以为模型的每一层自动添加 NVTX 标记，配合 Nsight Systems 进行细粒度性能分析。

```shell
# 启动服务器
nsys profile --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  -o layerwise_profile \
  python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --enable-layerwise-nvtx-marker \
    --disable-cuda-graph

# 通过 API 控制 profiling
curl -X POST http://127.0.0.1:30000/start_profile \
  -H "Content-Type: application/json" \
  -d '{
    "start_step": 3,
    "num_steps": 10,
    "activities": ["CUDA_PROFILER"]
  }'
```

习惯使用Nsight Systems的小伙伴可以试试这个方法，有一定帮助。

# 0x4. Decode设置固定batch_size之后的性能

一个开发中可能会用到的小技巧，查看Decoding在固定batch_size下的性能，可以通过：

```shell
python3 -m sglang.test.send_one --batch-size 128
```

然后看server log里面的decoding性能，这样比较方便。

# 0x5. 总结

暂时想到这些，谢谢大家。




