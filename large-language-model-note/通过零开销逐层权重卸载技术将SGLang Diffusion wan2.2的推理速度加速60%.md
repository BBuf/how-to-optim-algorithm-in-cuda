# 0x0. 前言

最近在优化SGLang对Wan2.2视频生成模型的支持时,发现了一个性能问题: 在使用双Transformer架构时,第1步和第19步的推理速度比正常步骤慢了约7倍。经过深入分析,通过实现**零开销的逐层权重卸载**(Layerwise Weight Offload)技术,最终将整体推理速度提升了**60%**(从149.69秒降至94.22秒)。需要说明的是这个技术的核心代码实现部分从Skywork AI Infra的视频模型优化中修改而来(https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/utils/layerwise_offload.py#L8)。

这个优化不仅带来了性能提升,也突破了Wan2.2模型的显存墙。比如我们现在可以在**24GB显存的4090**上运行wan2.2而不会OOM,具有还不错的实用意义。此外这个优化也可以方便的扩展到其他模型上,比如HunyuanVideo。本文将详细介绍问题的发现、分析和解决过程。

相关PR: https://github.com/sgl-project/sglang/pull/15511

测试环境: 8卡H100

# 0x1. 问题发现:为什么Wan2.2这么慢?

## 0x1.1 性能瓶颈定位

在对Wan2.2模型进行完整的profiling之后,发现了一个奇怪的现象:

- 第1步和第19步:耗时约36秒和31秒,异常缓慢
- 中间步骤(第2-18步):耗时约3.2秒,性能正常,完全达到了从cp4到cp8的2倍加速预期
- 其他步骤(第20-27步):耗时约3.2秒,性能正常

第1步和第19步的耗时相当于7个正常步骤,这显然是不可接受的。

main分支的数据：

```shell
sglang generate   --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers   --text-encoder-cpu-offload   --pin-cpu-memory   --num-gpus 8   --ulysses-degree 8 --attention-backend sage_attn  --enable-torch-compile --prompt "A cat walks on the grass, realistic" --num-frames 81 --height 720 --width 1280 --num-inference-steps 27 --guidance-scale 3.5 --guidance-scale-2 4.0 --perf-dump-path /home/lmsys/bbuf/dump/wan_step_profile_cp8_main.json

100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 27/27 [02:28<00:00,  5.52s/it]
[12-19 08:37:22] [DenoisingStage] average time per step: 5.5156 seconds
[12-19 08:37:23] [DenoisingStage] finished in 149.6943 seconds



"denoise_steps_ms": [
    35999.06893167645,
    3261.483933776617,
    3270.5406425520778,
    3267.8588768467307,
    3260.3964526206255,
    3263.016454875469,
    3268.026988953352,
    3264.5184732973576,
    3264.636719599366,
    3267.1875776723027,
    3268.562350422144,
    3268.1023878976703,
    3266.7769035324454,
    3268.044295720756,
    3264.268895611167,
    3271.0087513551116,
    3267.674465663731,
    3266.1060262471437,
    31282.590138725936,
    3263.5639663785696,
    3262.301029637456,
    3262.3210102319717,
    3261.833382770419,
    3264.719443395734,
    3265.314467251301,
    3267.530156299472,
    3261.9533529505134
  ],
```

经过本文介绍的优化之后的数据：


```bash
sglang generate   --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers   --text-encoder-cpu-offload   --pin-cpu-memory   --num-gpus 8   --ulysses-degree 8 --attention-backend sage_attn  --enable-torch-compile --prompt "A cat walks on the grass, realistic" --num-frames 81 --height 720 --width 1280 --num-inference-steps 27 --guidance-scale 3.5 --guidance-scale-2 4.0 --dit-layerwise-offload true --perf-dump-path /home/lmsys/bbuf/dump/wan_step_profile_cp8_async_offload.json


100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 27/27 [01:33<00:00,  3.46s/it]
[12-20 02:59:10] [DenoisingStage] average time per step: 3.4553 seconds
[12-20 02:59:10] [DenoisingStage] finished in 94.2283 seconds

"denoise_steps_ms": [
    7717.6975486800075,
    3275.042257271707,
    3280.2467988803983,
    3282.276245765388,
    3292.9044039919972,
    3286.5121429786086,
    3273.5616639256477,
    3271.6003246605396,
    3275.5934856832027,
    3291.9061705470085,
    3293.934356421232,
    3289.3909830600023,
    3298.0582248419523,
    3300.408118404448,
    3305.0247132778168,
    3299.3013756349683,
    3302.0150866359472,
    3299.040620215237,
    3291.0401169210672,
    3296.6199973598123,
    3290.26335850358,
    3302.190547809005,
    3295.942653901875,
    3297.3329443484545,
    3297.713255509734,
    3295.0284238904715,
    3290.1963284239173
  ],
```


## 0x1.2 根因分析

通过深入分析代码和profiling结果,找到了问题的根本原因:

1. Wan2.2使用**双Transformer架构**:模型包含`transformer`和`transformer_2`两个大型Transformer模块
2. 启用了**dit_cpu_offload**:为了节省GPU显存,模型权重在加载后被放置在CPU上
3. 第1步的开销:在第一次推理时,需要将`transformer`和`transformer_2`的权重从CPU拷贝到GPU,导致第1步非常慢,并且第一步还有torch compile、nccl初始化等开销
4. 第19步的开销:在第19步发生了**dual-stream切换**,需要将两个Transformer的权重卸载回CPU,交换它们,然后再拷贝回GPU

这种全模型级别的权重搬运导致了巨大的性能损失。

# 0x2. 解决方案:零开销逐层权重卸载

## 0x2.1 核心思想

传统的`dit_cpu_offload`是将整个模型的权重一次性在CPU和GPU之间搬运,这会导致:
- 大量的数据传输时间
- GPU计算和数据传输无法overlap
- 在dual-stream切换时需要重新搬运所有权重

**零开销逐层权重卸载**的核心思想是:
1. **逐层管理**:只在需要时将某一层的权重从CPU加载到GPU
2. **异步预取**:使用独立的CUDA Stream提前预取下一层的权重
3. **计算与传输overlap**:当前层在计算时,下一层的权重已经在异步加载
4. **及时释放**:当前层计算完成后,立即释放其GPU显存
5. **Pin Memory优化**:将CPU上的权重pin到物理内存,避免页面交换,这对于实现零开销至关重要

这样可以实现零额外开销:数据传输完全被计算隐藏,不会增加总体推理时间。

## 0x2.2 实现细节

### LayerwiseOffloadManager核心类

实现了一个轻量级的逐层CPU卸载管理器,核心功能包括:

```python
class LayerwiseOffloadManager:
    """A lightweight layerwise CPU offload manager.
    
    This utility offloads per-layer parameters/buffers from GPU to CPU, and
    supports async H2D prefetch using a dedicated CUDA stream.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        module_list_attr: str,  # 例如 "blocks"
        num_layers: int,
        enabled: bool,
        pin_cpu_memory: bool = True,
        auto_initialize: bool = False,
    ):
        # 创建独立的CUDA Stream用于异步拷贝
        self.copy_stream = torch.cuda.Stream() if self.enabled else None
        # 存储CPU上的权重
        self._cpu_weights: Dict[int, Dict[str, torch.Tensor]] = {}
        # 跟踪哪些层在GPU上
        self._gpu_layers: Dict[int, Set[str]] = {}
```

### 初始化:将所有层权重卸载到CPU

```python
def initialize(self) -> None:
    if not self.enabled:
        return
    
    # 遍历所有参数和buffer
    for name, param in self._named_parameters.items():
        layer_idx = self._match_layer_idx(name)
        if layer_idx is None or layer_idx >= self.num_layers:
            continue
        # 卸载到CPU并pin memory
        self._offload_tensor(name, param, layer_idx)
    
    # 预取第0层
    self.prefetch_layer(0, non_blocking=False)
```

这里有个关键点是`_offload_tensor`方法中会将CPU上的tensor进行pin memory操作:

```python
cpu_weight = tensor.detach().to("cpu")
if self.pin_cpu_memory:
    cpu_weight = cpu_weight.pin_memory()  # 锁定物理内存
```

**Pin memory**的作用是将内存页锁定在物理内存中,避免被操作系统swap到磁盘。这样做有两个好处:
1. **异步拷贝**(non_blocking=True)只对pinned memory有效,否则会退化为同步拷贝
2. **DMA**(Direct Memory Access)可以直接访问pinned memory,传输速度更快

在Wan2.2这种大模型场景下,如果不使用pin memory,异步拷贝会失效,计算和传输就无法overlap,也就谈不上零开销了。

### 异步预取:提前加载下一层

```python
def prefetch_layer(self, layer_idx: int, non_blocking: bool = True) -> None:
    if not self.enabled:
        return
    
    # 等待主stream完成当前计算
    self.copy_stream.wait_stream(torch.cuda.current_stream())
    
    # 在独立stream中异步拷贝权重
    for name, cpu_weight in self._cpu_weights[layer_idx].items():
        gpu_weight = torch.empty(
            cpu_weight.shape,
            dtype=self._cpu_dtypes[layer_idx][name],
            device=self.device,
        )
        with torch.cuda.stream(self.copy_stream):
            gpu_weight.copy_(cpu_weight, non_blocking=non_blocking)
        target.data = gpu_weight
```

### 层级作用域:优雅的使用方式

```python
@contextmanager
def layer_scope(
    self,
    *,
    prefetch_layer_idx: int | None,
    release_layer_idx: int | None,
    non_blocking: bool = True,
):
    """在进入时预取下一层,退出时等待拷贝完成并释放当前层"""
    if self.enabled and prefetch_layer_idx is not None:
        self.prefetch_layer(prefetch_layer_idx, non_blocking=non_blocking)
    try:
        yield
    finally:
        if self.enabled and self.copy_stream is not None:
            # 等待异步拷贝完成
            torch.cuda.current_stream().wait_stream(self.copy_stream)
        if self.enabled and release_layer_idx is not None:
            self.release_layer(release_layer_idx)
```

### 在模型forward中使用

在Wan2.2的Transformer forward中集成:

```python
offload_mgr = getattr(self, "_layerwise_offload_manager", None)
if offload_mgr is not None and getattr(offload_mgr, "enabled", False):
    for i, block in enumerate(self.blocks):
        with offload_mgr.layer_scope(
            prefetch_layer_idx=i + 1,  # 预取下一层
            release_layer_idx=i,        # 释放当前层
            non_blocking=True,
        ):
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                freqs_cis,
            )
```

通过`layer_scope`上下文管理器:
- 在执行第i层时,第i+1层的权重已经在异步加载
- 第i层计算完成后,立即释放其显存
- 数据传输和计算完全overlap,实现零开销

## 0x2.3 使用方式

在启动SGLang服务时,添加`--dit-layerwise-offload true`参数。下面是在8卡H100上的测试命令:

```bash
sglang generate \
  --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --text-encoder-cpu-offload \
  --pin-cpu-memory \
  --num-gpus 8 \
  --ulysses-degree 8 \
  --attention-backend sage_attn \
  --enable-torch-compile \
  --dit-layerwise-offload true \
  --prompt "A cat walks on the grass, realistic" \
  --num-frames 81 \
  --height 720 \
  --width 1280 \
  --num-inference-steps 27 \
  --guidance-scale 3.5 \
  --guidance-scale-2 4.0
```

注意:
- `--dit-layerwise-offload`不能与`dit_cpu_offload`、`use_fsdp_inference`或`cache-dit`同时使用

# 0x3. 性能提升

## 0x3.1 主分支 vs PR分支

**主分支(使用dit_cpu_offload)**:
```
[12-19 08:37:22] [DenoisingStage] average time per step: 5.5156 seconds
[12-19 08:37:23] [DenoisingStage] finished in 149.6943 seconds

"denoise_steps_ms": [
    35999.07,  // 第1步: 36秒!
    3261.48, 3270.54, 3267.86, 3260.40, 3263.02, 3268.03, 3264.52,
    3264.64, 3267.19, 3268.56, 3268.10, 3266.78, 3268.04, 3264.27,
    3271.01, 3267.67, 3266.11,
    31282.59,  // 第19步: 31秒!
    3263.56, 3262.30, 3262.32, 3261.83, 3264.72, 3265.31, 3267.53, 3261.95
]
```

**PR分支(使用dit_layerwise_offload)**:
```
[12-20 02:59:10] [DenoisingStage] average time per step: 3.4553 seconds
[12-20 02:59:10] [DenoisingStage] finished in 94.2283 seconds

"denoise_steps_ms": [
    7717.70,   // 第1步: 7.7秒,提升78%!
    3275.04, 3280.25, 3282.28, 3292.90, 3286.51, 3273.56, 3271.60,
    3275.59, 3291.91, 3293.93, 3289.39, 3298.06, 3300.41, 3305.02,
    3299.30, 3302.02, 3299.04,
    3291.04,   // 第19步: 3.3秒,提升89%!
    3296.62, 3290.26, 3302.19, 3295.94, 3297.33, 3297.71, 3295.03, 3290.20
]
```

## 0x3.2 性能分析

从数据上看:
- **总体加速**: 149.69秒 → 94.22秒, 提升58%
- **第1步加速**: 36秒 → 7.7秒, 提升78%
- **第19步加速**: 31秒 → 3.3秒, 提升89%
- **中间步骤**: 保持在3.2-3.3秒,性能稳定

下面是torch profiler的trace截图,可以清楚地看到memcpy和compute完全overlap,没有额外开销:

![](https://files.mdnice.com/user/59/3a7b359a-1a44-4efd-abaf-a5890d2342da.png)

从timeline可以看到,**H2D的memcpy操作和kernel执行是完全重叠的**,这就是零开销的体现。异步拷贝在独立的stream中进行,不会阻塞主stream的计算。

# 0x4. 额外优化:All2All预热

即使解决了权重卸载的问题,我发现无论是否启用torch compile,第1步的性能仍然比后续步骤慢约7倍。通过完整的profiling发现,这是因为**NCCL All2All操作的初始化开销**。

![](https://files.mdnice.com/user/59/14672e39-3de2-42e5-8d9e-6654b5337427.png)

这个初始化开销不应该在denoise阶段,而应该提前处理。因此实现了专门针对All2All操作的**预热逻辑**。预热之后,不启用compile时第1步的时间几乎与后续步骤相同,启用compile时第1步的时间仅为后续步骤的约2倍(而不是7倍)。NCCL All2All的初始化开销被提前消除,denoise阶段的第1步不再有明显的延迟。

# 0x5. 总结

这个优化方案的关键技术点包括:**逐层权重管理**只在需要时加载特定层的权重,避免全模型搬运;使用**独立CUDA Stream**实现异步H2D传输,让当前层计算时下一层权重已在加载;使用**pinned memory**锁定物理内存,这是实现异步拷贝和零开销的前提条件;每层计算完成后立即释放GPU显存。

实现的亮点在于数据传输完全被计算隐藏,通过`layer_scope` context manager提供了简洁的API,支持任意具有`blocks`属性的DiT模型,并且与cache-dit等特性互斥避免潜在错误。

这个技术特别适合大型Diffusion模型(如Wan2.2)、显存受限的场景(可以让原本需要80GB显存的模型在24GB显存的4090上运行)、需要在CPU和GPU之间频繁搬运权重的场景,以及具有多层Transformer结构的模型。
