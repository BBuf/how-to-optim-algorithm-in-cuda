## 0x0. 前言

Cache-DiT 是唯品会开源的 PyTorch 原生 DiT 推理加速引擎（https://github.com/vipshop/cache-dit），通过混合缓存加速和并行化技术来加速 DiT 模型的推理。在之前的版本中，Cache-DiT 主要聚焦于离线推理场景，用户需要编写 Python 脚本来调用模型。虽然这种方式对于研究和实验来说很方便，但在生产环境中部署时就显得不太友好了。

最近为 Cache-DiT 实现了完整的 Serving 功能，目标很简单：让用户可以像使用 SGLang 一样，通过一行命令启动服务，然后通过 HTTP API 来调用模型生成图片。这个功能支持单卡推理，也支持 Tensor Parallelism 和 Context Parallelism 等分布式推理模式。

这篇文章会详细介绍 Cache-DiT Serving 的实现过程，包括如何借鉴 SGLang 的设计，在分布式场景下遇到的一些坑，以及最终的解决方案。

## 0x1. 为什么需要 Serving

在生产环境中部署模型时，我们通常希望有一个统一的 API 接口，让不同的客户端可以方便地调用。比如 Web 前端、移动端 App、或者其他后端服务，都可以通过 HTTP 请求来生成图片，而不需要关心底层的模型加载、GPU 管理等细节。

另外，统一的服务也方便做资源管理和监控。比如可以限制并发请求数量，记录每个请求的耗时和资源占用，出问题时也更容易排查。

Cache-DiT Serving 就是为了解决这些问题。希望用户可以像使用 SGLang 的 `sglang.launch_server` 一样简单地启动服务，然后就可以通过 HTTP API 来调用了。

## 0x2. 整体架构

单卡模式的架构比较简单，就是启动一个 FastAPI 服务器，接收 HTTP 请求，然后调用 ModelManager 来执行推理。ModelManager 负责加载模型、管理缓存、执行推理等工作。

但是在分布式模式下就复杂一些了。TP 和 CP 这两种并行模式有一个共同的特点：所有 rank 必须同时调用 `pipe()` 进行推理。这是因为 TP 使用 NCCL 的 all-reduce 来同步梯度和激活值，CP 使用 all-to-all 来交换 attention 的 KV。如果只有 rank 0 调用 `pipe()`，其他 rank 会在 NCCL 通信中一直等待，最终导致超时死锁。

所以需要一个机制来同步所有 rank 的推理请求。最简单的方案就是使用 NCCL 的 broadcast：rank 0 接收 HTTP 请求后，把请求内容 broadcast 给所有其他 rank，然后所有 rank 一起执行推理。执行完成后，rank 0 把结果返回给客户端，其他 rank 丢弃结果继续等待下一个请求。

这个方案虽然简单，但对于 DiT 模型来说已经足够了。因为 DiT 模型的推理通常是串行的，不像 LLM 那样需要复杂的 continuous batching 和调度系统。

## 0x3. 借鉴 SGLang 的设计

在实现 Cache-DiT Serving 的过程中，主要参考了 SGLang 的 generate 部分的 serving 设计。SGLang 的这部分实现比较简单直接，相对容易理解和借鉴。

具体来说，参考了 SGLang 如何用 FastAPI 组织 HTTP 接口，如何解析命令行参数，以及如何管理请求的生命周期。这些基础的 HTTP Server 架构设计在 SGLang 的 `http_server.py` 和 `launch_server.py` 中都有比较清晰的实现。

对于分布式推理的部分，考虑到 DiT 模型的特点（一次性生成完整图片，不需要逐 token 生成），采用了更简单的方案：直接使用 NCCL broadcast 来同步请求。rank 0 运行 HTTP 服务器，接收到请求后通过 NCCL broadcast 把请求发送给所有 rank，然后所有 rank 一起执行推理。这样既利用了现有的分布式环境（torchrun 已经帮忙管理好了进程），又避免了额外的进程间通信开销。

## 0x4. 核心实现

整个实现的核心在于如何在分布式环境下同步所有 rank 的推理请求。下面详细介绍几个关键部分的实现。

首先，单卡模式的架构非常简单，如下图所示（让Claude 4生成的）：

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ HTTP Request
       ↓
┌─────────────────────┐
│   FastAPI Server    │
│  (Rank 0, Port 8000)│
└──────┬──────────────┘
       │
       ↓
┌─────────────────────┐
│   ModelManager      │
│  - load_model()     │
│  - generate()       │
└──────┬──────────────┘
       │
       ↓
┌─────────────────────┐
│  DiffusionPipeline  │
│  + Cache-DiT        │
└─────────────────────┘
```

### 启动流程

启动服务的流程比较直观。首先解析命令行参数，然后根据是否使用分布式来初始化环境。如果使用了 TP 或 CP，就需要调用 `torch.distributed.init_process_group` 来初始化 NCCL 通信。

接着创建 ModelManager 并加载模型。这里会根据用户的配置来决定是否启用缓存、是否使用 torch.compile 等。

最后根据并行类型来选择启动方式。如果是分布式模式（TP/CP），rank 0 会启动 HTTP 服务器，其他 rank 会运行 worker 循环等待请求。如果是单卡模式，就直接启动 HTTP 服务器。

### Broadcast 同步机制

这是整个实现的核心。需要确保所有 rank 使用完全相同的请求参数，包括 prompt、width、height、seed 等。

实现方式很直接：rank 0 先用 pickle 把请求对象序列化成字节流，然后通过 NCCL broadcast 发送给所有 rank。为了让其他 rank 知道要接收多少字节，先 broadcast 一个表示大小的 tensor，然后再 broadcast 实际的数据。

所有 rank 接收到数据后，都用 pickle 反序列化得到相同的请求对象，然后一起调用 `model_manager.generate()` 执行推理。执行完成后，rank 0 把结果返回给客户端，其他 rank 丢弃结果继续等待下一个请求。

这里有一个细节：让 rank 0 也反序列化 broadcast 的数据，而不是直接使用原始的请求对象。这样可以确保所有 rank 使用的是完全相同的对象，避免因为 pickle 序列化/反序列化的差异导致的问题。

大概的流程如下，让Claude 4画了一下：

```
Rank 0 (HTTP Server)              Rank 1, 2, ... (Workers)
      |                                  |
  启动 FastAPI                         运行 worker 循环
      |                                  |
  接收 HTTP 请求                        等待 broadcast
      |                                  |
  broadcast 请求 --------NCCL-------->  接收请求
      |                                  |
  调用 pipe() <--------同步推理------> 调用 pipe()
      |              (all-reduce/all-to-all)
      |                                  |
  返回结果                              丢弃结果
      |                                  |
  等待下一个请求                        等待下一个 broadcast
```


### 随机数生成的坑

在分布式推理中，所有 rank 必须使用相同的随机种子，否则会出现各种奇怪的问题。TP 模式下图片会整体模糊，CP 模式下图片的下半部分会变成乱码。

这个问题的根源在于 PyTorch 的 CUDA RNG 状态是 per-device 的。也就是说，即使你在 cuda:0 和 cuda:1 上用相同的 seed 创建 generator，它们生成的随机数序列也是不同的。

解决方案是使用 CPU generator。CPU 的 RNG 状态是全局的，所以所有 rank 用相同的 seed 创建 CPU generator 时，会生成完全相同的随机数序列。diffusers 会自动把这些随机数移动到正确的 GPU 上，所以不用担心性能问题。

另外，如果用户没有提供 seed，在分布式模式下会自动生成一个固定的 seed（比如 42）。这样可以确保即使用户忘记设置 seed，也不会出现图片模糊或乱码的问题。

### Device 放置的坑

还有一个容易踩的坑是 device 放置。在 CP 模式下，all-to-all 通信需要所有 tensor 都在 GPU 上。但是如果 pipeline 的某些组件（比如 VAE、text encoder）还在 CPU 上，就会报错说 "No backend type associated with device type cpu"。

解决方案是在所有模式下都调用 `pipe.to("cuda")`。虽然 TP 模式下 transformer 已经被切分到多个 GPU 上了，但其他组件还需要手动移动。CP 模式下更是所有组件都需要在 GPU 上。

## 0x5. 总结踩过的坑

在实现过程中踩了不少坑，这里总结一下主要的几个：

第一个坑是 TP/CP 死锁。一开始以为 CP 模式不需要 broadcast 机制，因为看起来 CP 的 forward pattern 和单卡是一样的。结果启动服务后发现 rank 0 一直卡在推理中，rank 1 在睡觉。后来才意识到 CP 的 all-to-all 通信也需要所有 rank 同时参与，所以必须用 broadcast 来同步请求。

第二个坑是图片模糊和乱码。TP 模式下生成的图片整体都是模糊的，CP 模式下图片的下半部分是乱码。debug 了很久才发现是随机数生成的问题。一开始用的是 GPU generator，后来发现不同 GPU 上的 generator 即使用相同的 seed 也会生成不同的随机数。改成 CPU generator 后问题就解决了。

第三个坑是 rank 0 没有使用 broadcast 的请求。一开始让 rank 0 直接使用原始的请求对象，其他 rank 使用 broadcast 接收到的请求。结果发现 CP 模式下图片还是有问题。后来意识到可能是 pickle 序列化/反序列化的差异导致的，改成所有 rank 都使用 broadcast 的请求后就正常了。

第四个坑是不同 pipeline 支持的参数不一样。在测试 FLUX2 模型时发现会报错 `Flux2Pipeline.__call__() got an unexpected keyword argument 'negative_prompt'`。原来 FLUX2 的 pipeline 不支持 `negative_prompt` 参数，但代码里默认传了这个参数。解决方案是通过 `inspect.signature` 检查 pipeline 的 `__call__` 方法支持哪些参数，只传递它支持的参数。这样就可以兼容不同的 pipeline 了。

## 0x6. 使用方法

使用起来非常简单，和 SGLang 的体验基本一致。

首先克隆 Cache-DiT ，然后切换到 https://github.com/vipshop/cache-dit/pull/522 这个pr对应的分支：

```bash
git clone git@github.com:BBuf/cache-dit.git
cd cache-dit
git checkout try_to_support_serving
pip install -e ".[serving]"
```

这个很快也会并入主线，应该吧

然后启动服务。单卡模式：

```bash
cache-dit-serve \
    --model-path black-forest-labs/FLUX.1-dev \
    --cache \
    --host 0.0.0.0 \
    --port 8000
```

TP 模式（2卡）：

```bash
torchrun --nproc_per_node=2 -m cache_dit.serve.serve \
    --model-path black-forest-labs/FLUX.1-dev \
    --cache \
    --parallel-type tp \
    --host 0.0.0.0 \
    --port 8000
```

CP 模式（2卡）：

```bash
torchrun --nproc_per_node=2 -m cache_dit.serve.serve \
    --model-path black-forest-labs/FLUX.1-dev \
    --cache \
    --parallel-type ulysses \
    --host 0.0.0.0 \
    --port 8000
```

启动后就可以通过 HTTP API 来调用了。Python 客户端示例：

```python
import requests
import base64
from PIL import Image
from io import BytesIO

url = "http://localhost:8000/generate"
data = {
    "prompt": "A beautiful sunset over the ocean",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 28,
    "guidance_scale": 3.5,
    "seed": 42,
}

response = requests.post(url, json=data)
result = response.json()

# 解码图片
image_data = base64.b64decode(result["images"][0])
image = Image.open(BytesIO(image_data))
image.save("output.png")

print(f"Time cost: {result['time_cost']:.2f}s")
```

也可以用 cURL 和 jq：

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful sunset over the ocean",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 50
  }' | jq -r '.images[0]' | base64 -d > output.png
```

启动服务后，访问 `http://localhost:8000/docs` 可以看到完整的 API 文档（Swagger UI）。

完整的命令行参数可以查看 get_args 函数(https://github.com/vipshop/cache-dit/pull/522/files#diff-8d807db087ac7dc3923b8b6c6c4af29c87f8f29882f19ca5a9bf33f9b3d608b6R17)，这里列举几个常用的：

- `--model-path`: 模型路径或 HuggingFace model ID
- `--cache`: 启用缓存加速
- `--parallel-type`: 并行类型 (tp/ulysses/ring)
- `--compile`: 启用 torch.compile
- `--host`: 服务器地址 (默认 0.0.0.0)
- `--port`: 服务器端口 (默认 8000)

如果想要更快的推理速度，可以启用 torch.compile，更多优化选项请详细了解cache-dit框架：

```bash
cache-dit-serve --model-path FLUX.1-dev --cache --compile
```

首次推理会进行编译（比较慢），但后续推理会快很多。

贴一个FLUX.1.dev的例子。

- server端log


```markdown
cache-dit-serve --model-path /nas/bbuf/FLUX.1-dev/ --cache --compile
WARNING 12-03 06:50:00 [_attention_dispatch.py:303] Re-registered NATIVE attention backend to enable context parallelism with attn mask. You can disable this behavior by export env: export CACHE_DIT_ENABLE_CUSTOM_CP_NATIVE_ATTN_DISPATCH=0.
INFO 12-03 06:50:00 [_attention_dispatch.py:416] Registered new attention backend: _SDPA_CUDNN, to enable context parallelism with attn mask. You can disable it by: export CACHE_DIT_ENABLE_CUSTOM_CP_NATIVE_ATTN_DISPATCH=0.
INFO 12-03 06:50:01 [serve.py:107] Initializing model manager...
INFO 12-03 06:50:01 [model_manager.py:68] Initializing ModelManager: model_path=/nas/bbuf/FLUX.1-dev/, device=cuda
INFO 12-03 06:50:01 [serve.py:119] Loading model...
INFO 12-03 06:50:01 [model_manager.py:72] Loading model: /nas/bbuf/FLUX.1-dev/
Loading pipeline components...:   0%|                                                                       | 0/7 [00:00<?, ?it/s]`torch_dtype` is deprecated! Use `dtype` instead!
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.05it/s]
Loading pipeline components...:  29%|██████████████████                                             | 2/7 [00:02<00:05,  1.08s/it]You set `add_prefix_space`. The tokenizer needs to be converted from the slow tokenizers
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████| 3/3 [00:04<00:00,  1.49s/it]
Loading pipeline components...: 100%|███████████████████████████████████████████████████████████████| 7/7 [00:08<00:00,  1.28s/it]
INFO 12-03 06:50:10 [model_manager.py:81] Enabling DBCache acceleration
INFO 12-03 06:50:10 [cache_adapter.py:49] FluxPipeline is officially supported by cache-dit. Use it's pre-defined BlockAdapter directly!
INFO 12-03 06:50:10 [functor_flux.py:61] Applied FluxPatchFunctor for FluxTransformer2DModel, Patch: False.
INFO 12-03 06:50:10 [block_adapters.py:147] Found transformer from diffusers: diffusers.models.transformers.transformer_flux enable check_forward_pattern by default.
INFO 12-03 06:50:10 [block_adapters.py:494] Match Block Forward Pattern: ['FluxSingleTransformerBlock', 'FluxTransformerBlock'], ForwardPattern.Pattern_1
INFO 12-03 06:50:10 [block_adapters.py:494] IN:('hidden_states', 'encoder_hidden_states'), OUT:('encoder_hidden_states', 'hidden_states'))
INFO 12-03 06:50:10 [cache_adapter.py:148] Use custom 'enable_separate_cfg' from cache context kwargs: True. Pipeline: FluxPipeline.
INFO 12-03 06:50:10 [cache_adapter.py:307] Collected Context Config: DBCache_F8B0_W8I1M0MC0_R0.08, Calibrator Config: None
INFO 12-03 06:50:10 [pattern_base.py:70] Match Blocks: CachedBlocks_Pattern_0_1_2, for transformer_blocks, cache_context: transformer_blocks_139774198646688, context_manager: FluxPipeline_139774199622368.
INFO 12-03 06:50:10 [model_manager.py:100] Moving pipeline to CUDA
INFO 12-03 06:52:33 [model_manager.py:108] Enabling torch.compile
INFO 12-03 06:52:33 [model_manager.py:112] Model loaded successfully
INFO 12-03 06:52:33 [serve.py:121] Model loaded successfully!
INFO 12-03 06:52:33 [serve.py:125] Starting server at http://0.0.0.0:8000
INFO 12-03 06:52:33 [serve.py:126] API docs at http://0.0.0.0:8000/docs
INFO:     Started server process [1928284]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO 12-03 06:52:45 [model_manager.py:117] Warming up for shape 1024x1024...
  0%|                                                                                                       | 0/4 [00:00<?, ?it/s]/usr/local/lib/python3.12/dist-packages/torch/_dynamo/variables/functions.py:1547: UserWarning: Dynamo detected a call to a `functools.lru_cache` wrapped function.Dynamo currently ignores `functools.lru_cache` and directly traces the wrapped function.`functools.lru_cache` wrapped functions that read outside state may not be traced soundly.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/torch/_dynamo/variables/functions.py:1547: UserWarning: Dynamo detected a call to a `functools.lru_cache` wrapped function.Dynamo currently ignores `functools.lru_cache` and directly traces the wrapped function.`functools.lru_cache` wrapped functions that read outside state may not be traced soundly.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/torch/_dynamo/variables/functions.py:1547: UserWarning: Dynamo detected a call to a `functools.lru_cache` wrapped function.Dynamo currently ignores `functools.lru_cache` and directly traces the wrapped function.`functools.lru_cache` wrapped functions that read outside state may not be traced soundly.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/torch/_dynamo/variables/functions.py:1547: UserWarning: Dynamo detected a call to a `functools.lru_cache` wrapped function.Dynamo currently ignores `functools.lru_cache` and directly traces the wrapped function.`functools.lru_cache` wrapped functions that read outside state may not be traced soundly.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/torch/_dynamo/variables/functions.py:1547: UserWarning: Dynamo detected a call to a `functools.lru_cache` wrapped function.Dynamo currently ignores `functools.lru_cache` and directly traces the wrapped function.`functools.lru_cache` wrapped functions that read outside state may not be traced soundly.
  warnings.warn(
100%|███████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:06<00:00,  1.51s/it]
INFO 12-03 06:52:53 [model_manager.py:127] Warmup completed for 1024x1024
INFO 12-03 06:52:53 [model_manager.py:137] Generating image: prompt='A beautiful sunset over the ocean...'
100%|█████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:06<00:00,  7.98it/s]
WARNING 12-03 06:53:00 [summary.py:275] Can't find Context Options for: FluxSingleTransformerBlock
WARNING 12-03 06:53:00 [summary.py:284] Can't find Parallelism Config for: FluxSingleTransformerBlock
WARNING 12-03 06:53:00 [summary.py:275] Can't find Context Options for: FluxTransformerBlock
WARNING 12-03 06:53:00 [summary.py:284] Can't find Parallelism Config for: FluxTransformerBlock

🤗Context Options: OptimizedModule

{'cache_config': DBCacheConfig(cache_type=<CacheType.DBCache: 'DBCache'>, Fn_compute_blocks=8, Bn_compute_blocks=0, residual_diff_threshold=0.08, max_accumulated_residual_diff_threshold=None, max_warmup_steps=8, warmup_interval=1, max_cached_steps=-1, max_continuous_cached_steps=-1, enable_separate_cfg=True, cfg_compute_first=False, cfg_diff_compute_separate=True, num_inference_steps=None, steps_computation_mask=None, steps_computation_policy='dynamic'), 'name': 'transformer_blocks_139774198646688'}
WARNING 12-03 06:53:00 [summary.py:284] Can't find Parallelism Config for: OptimizedModule

⚡️Cache Steps and Residual Diffs Statistics: OptimizedModule

| Cache Steps | Diffs P00 | Diffs P25 | Diffs P50 | Diffs P75 | Diffs P95 | Diffs Min | Diffs Max |
|-------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| 6           | 0.043     | 0.06      | 0.089     | 0.135     | 0.217     | 0.043     | 0.285     |


⚡️CFG Cache Steps and Residual Diffs Statistics: OptimizedModule

| CFG Cache Steps | Diffs P00 | Diffs P25 | Diffs P50 | Diffs P75 | Diffs P95 | Diffs Min | Diffs Max |
|-----------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| 6               | 0.043     | 0.055     | 0.097     | 0.144     | 0.266     | 0.043     | 0.373     |

INFO 12-03 06:53:00 [model_manager.py:183] Image generation completed in 6.55s
INFO:     127.0.0.1:55144 - "POST /generate HTTP/1.1" 200 OK
```

- client端log



```markdwon
 python -m cache_dit.serve.client \
    --prompt "A beautiful sunset over the ocean" \
    --width 1024 \
    --height 1024 \
    --steps 50 \
    --output output.png
WARNING 12-03 06:48:43 [_attention_dispatch.py:303] Re-registered NATIVE attention backend to enable context parallelism with attn mask. You can disable this behavior by export env: export CACHE_DIT_ENABLE_CUSTOM_CP_NATIVE_ATTN_DISPATCH=0.
INFO 12-03 06:48:43 [_attention_dispatch.py:416] Registered new attention backend: _SDPA_CUDNN, to enable context parallelism with attn mask. You can disable it by: export CACHE_DIT_ENABLE_CUSTOM_CP_NATIVE_ATTN_DISPATCH=0.
Generating image: A beautiful sunset over the ocean
Image saved to output.png
Cache stats: {'cache_stats': [{'cache_options': "{'cache_config': DBCacheConfig(cache_type=<CacheType.DBCache: 'DBCache'>, Fn_compute_blocks=8, Bn_compute_blocks=0, residual_diff_threshold=0.08, max_accumulated_residual_diff_threshold=None, max_warmup_steps=8, warmup_interval=1, max_cached_steps=-1, max_continuous_cached_steps=-1, enable_separate_cfg=True, cfg_compute_first=False, cfg_diff_compute_separate=True, num_inference_steps=None, steps_computation_mask=None, steps_computation_policy='dynamic'), 'name': 'transformer_blocks_140121591103456'}", 'cached_steps': [8, 10, 12, 14, 16, 18], 'parallelism_config': None}]}
Time cost: 9.22s
root@264a63f2d86e:/nas/bbuf/cache-dit# curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful sunset over the ocean",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 50
  }' | jq -r '.images[0]' | base64 -d > output.png
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100 1666k  100 1666k  100   125   173k     13  0:00:09  0:00:09 --:--:--  379k
```

- 结果

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-1/3b471855-3c19-406b-9ae7-52fa7c141cdc.jpg)

## 0x7. 总结

Cache-DiT Serving 的实现目标很简单：让用户可以像使用 SGLang 一样方便地部署 DiT 模型。通过借鉴 SGLang 的设计，并针对 DiT 模型的特点进行简化，努力实现了一个轻量级但功能还算完整的推理服务。整个实现的核心在于分布式推理的同步机制。通过使用 NCCL broadcast 来同步请求，避免了复杂的多进程架构，同时保证了 TP 和 CP 模式下的正确性。我在实现过程中踩了不少坑，特别是随机数生成和 device 放置的问题，最终都花时间搞定了。然后目前这个功能已经可以正常工作了，支持单卡、TP 和 CP 三种模式。后续会做更多测试和推进merge进主分支。server端的profiler等以后也会花时间搞定。

欢迎大家试用并提供反馈！


## 参考资料

- Cache-DiT GitHub: https://github.com/vipshop/cache-dit
- SGLang GitHub: https://github.com/sgl-project/sglang
- Cache-DiT 学习笔记: https://github.com/BBuf/how-to-optim-algorithm-in-cuda/blob/master/large-language-model-note/Cache-Dit%20%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0.md
