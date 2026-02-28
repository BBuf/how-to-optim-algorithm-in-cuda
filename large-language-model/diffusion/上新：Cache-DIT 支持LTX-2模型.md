## 0x0. 前言
 
最近我和 [DefTruth](https://github.com/DefTruth) 在 Cache-DIT 上支持了 LTX-2 模型，对应 PR：
 
https://github.com/vipshop/cache-dit/pull/691
 
记录一下 Cache-DIT 适配 LTX-2 的几个关键点。重点放在 cache 侧的适配（`functor_ltx2.py`）这里，Parallel-Text-Encoder / Parallel-Vae 的适配相对简单，可以抄已有例子就不赘述。对Cache-DIT 有新模型适配需求的小伙伴可以参考一下。

 
- **模型侧**：LTX-2 同时支持 T2V / I2V，而且生成的视频是带音频的。
- **框架侧**：为了让它适配 Cache-DIT 的 cache 特性，我们补了一个针对 LTX-2 的 patch（`functor_ltx2.py`）。另外为了避免H100 Ulysses4 OOM，也把 `--parallel-text-encoder` 和 `--parallel-vae` 这两个feature接上了。

这个模型更多的相信信息见：https://huggingface.co/Lightricks/LTX-2

# 0x1. 效果

## Text2Video

```python
prompt = (
            "A cinematic tracking shot through a neon-lit rainy cyberpunk street at night. "
            "Reflections shimmer on wet asphalt, holographic signs flicker, and steam rises from vents. "
            "A sleek motorbike glides past the camera in slow motion, droplets scattering in the air. "
            "Smooth camera motion, natural parallax, ultra-realistic detail, cinematic lighting, film look."
        )
        negative_prompt = (
            "shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, "
            "motion artifacts, bad anatomy, ugly, transition, static, text, watermark."
        )
```

## Image2Video

```python
prompt = (
            "An astronaut hatches from a fragile egg on the surface of the Moon, the shell cracking and peeling "
            "apart in gentle low-gravity motion. Fine lunar dust lifts and drifts outward with each movement, "
            "floating in slow arcs before settling back onto the ground. The astronaut pushes free in a deliberate, "
            "weightless motion, small fragments of the egg tumbling and spinning through the air. In the background, "
            "the deep darkness of space subtly shifts as stars glide with the camera's movement, emphasizing vast "
            "depth and scale. The camera performs a smooth, cinematic slow push-in, with natural parallax between the "
            "foreground dust, the astronaut, and the distant starfield. Ultra-realistic detail, physically accurate "
            "low-gravity motion, cinematic lighting, and a breath-taking, movie-like shot."
        )
        negative_prompt = (
            "shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, "
            "motion artifacts, fused fingers, bad anatomy, weird hand, ugly, transition, static."
        )
```

# 0x2. Benchmark

以T2V为例子：

```shell
CACHE_DIT_LTX2_PIPELINE=t2v CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
        -m cache_dit.serve.serve \
        --model-path Lightricks/LTX-2 \
        --parallel-type ulysses \
        --parallel-text-encoder \
        --parallel-vae \
        --cache \
        --ulysses-anything

DIT_LTX2_MODE=base python -m pytest -q cache-dit/tests/serving/test_ltx2_text2video.py
```

视频参数为：

```python
prompt = (
    "A cinematic tracking shot through a neon-lit rainy cyberpunk street at night. "
    "Reflections shimmer on wet asphalt, holographic signs flicker, and steam rises from vents. "
    "A sleek motorbike glides past the camera in slow motion, droplets scattering in the air. "
    "Smooth camera motion, natural parallax, ultra-realistic detail, cinematic lighting, film look."
)
negative_prompt = (
    "shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, "
    "motion artifacts, bad anatomy, ugly, transition, static, text, watermark."
)
return call_api(
    prompt=prompt,
    negative_prompt=negative_prompt,
    name="ltx2_base_t2v",
    seed=42,
    width=1280,
    height=720,
    num_frames=121,
    fps=24,
    num_inference_steps=40,
    guidance_scale=4.0,
)
```

![](https://files.mdnice.com/user/59/b1a3cc6c-fa2c-47e6-9f91-024f11555eab.png)

H100 4卡 23s左右跑出1080P视频。

## 0x1. LTX-2 模型有什么不一样
 
这次接 LTX-2，最关心的不是“又多了一个视频模型”，而是它本质上是一个 **Audio-Visual Transformer**：
 
- **(1) 同一个模型仓库，支持 T2V + I2V**
- **(2) 输出视频自带音频（video/audio 两路 latent 一起走）**
 
这两点会直接影响到在 cache-dit 里怎么“对齐” block cache 的抽象。
 
- **cache 侧怎么做 block adapter / forward pattern 匹配**
- **Ulysses/Ring（context parallel）下 timestep、mask、RoPE 的切分是否正确**
- **TP（tensor parallel）下 RoPE、qk norm 等细节是否能对齐**

## 0x2. Cache-DIT 里怎么加载并区分 T2V / I2V
 
我们在 serving 里做了一个直观的分流（`cache_dit/src/cache_dit/serve/model_manager.py`）：
 
- `model_path` 里包含 `LTX-2` 时，读取环境变量 `CACHE_DIT_LTX2_PIPELINE`（默认 `t2v`）
- `t2v` 走 `diffusers.LTX2Pipeline`
- `i2v` 走 `diffusers.LTX2ImageToVideoPipeline`
 
我个人觉得这点对用户体验挺好的：同一个模型路径，切换任务只需要一个 env。

## 0x3. Cache 侧主要改了什么：`functor_ltx2.py` 为什么必须 patch
 
这次 LTX-2 接入，最关键的不是“能跑”，而是 --cache 的兼容。

### 0x3.1 cache-dit 的 Pattern_0 是怎么调用 block 的
 
cache-dit 的 block cache 逻辑（Pattern_0/1/2）在 `cache_dit/src/cache_dit/caching/cache_blocks/pattern_base.py` 里。

它对每一个 block 的调用方式是固定的：
 
- wrapper 的 forward 形参是 `(hidden_states, encoder_hidden_states, *args, **kwargs)`
- 调 block 时永远是：
 
```python
hidden_states = block(hidden_states, encoder_hidden_states, *args, **kwargs)
```
 
然后如果 block 返回了 tuple，wrapper 会把它当成“两路输出”处理（`_process_block_outputs`）。

也就是说，从 cache-dit 的抽象来看，Pattern_0 天然就是两路：
 
- 第一路：`hidden_states`
- 第二路：`encoder_hidden_states`
 
至于第二路到底是什么语义（text？audio？别的？），cache-dit 本身并不关心。

### 0x3.2 LTX-2 的 block forward 入参顺序跟这个抽象不兼容
 
在 `cache_dit/src/cache_dit/caching/block_adapters/adapters.py` 里，我把 LTX-2 归成 `ForwardPattern.Pattern_0`，因为它的 block 确实会返回两路：
 
- LTX-2 block 返回 `(hidden_states, audio_hidden_states)`
 
同时 adapter 里也写了一个明确的取舍：把 `audio_hidden_states` 当成 Pattern_0 的“第二路”来用。

但是问题来了：
 
cache-dit 的 wrapper 调 block 的时候，第二个位置参数固定叫 `encoder_hidden_states`。

而 LTX-2 的 block forward（diffusers）第二个位置参数并不是“文本 encoder hidden states”，它是 `audio_hidden_states`。

如果我不做 patch，直接让 cache wrapper 按默认方式去调：
 
```python
block(hidden_states, encoder_hidden_states, ...)
```
 
那么在 LTX-2 block 的视角里，这个 `encoder_hidden_states`（文本 embeddings）会被错传给 `audio_hidden_states` 这个位置参数。

这样就会报错。

### 0x3.3 patch 的做法：我们在 transformer.forward 里把 block 调用的前四个输入按位置对齐
 
所以 `functor_ltx2.py` 里做的事情非常明确：

- 用 `LTX2PatchFunctor` 把 `LTX2VideoTransformer3DModel.forward` 替换成 `__patch_transformer_forward__`
- 在 `for block in self.transformer_blocks:` 里，调用 block 时显式按 LTX-2 block 的真实签名把前四个输入用位置参数传进去：
 
```python
hidden_states, audio_hidden_states = block(
    hidden_states,
    audio_hidden_states,
    encoder_hidden_states,
    audio_encoder_hidden_states,
    ...
)
```
 
这样做之后：
 
- block 的第二个位置参数拿到的就是它想要的 `audio_hidden_states`
- block 返回 `(hidden_states, audio_hidden_states)`，而 cache-dit 的 Pattern_0 正好把“第二路”当成 `encoder_hidden_states` 去缓存
- 真正的文本 embeddings（`encoder_hidden_states/audio_encoder_hidden_states`）我作为第 3、第 4 个位置参数传入，避免和 cache wrapper 的 `encoder_hidden_states` 这个参数冲突。

另外为了 Ulysses4 不 OOM，我们也做了 parallel-text-encoder TP + parallel VAE 的适配。

TP plan / CP plan 这块跟其它模型差别不大，这次基本是沿着 diffusers 的LTX-2实现把该切的输入切一下。

## 0x6. 怎么用
 
### 0x6.1 examples（多卡：Ulysses/TP + cache）
 
```bash
torchrun --nproc_per_node=4 generate.py ltx2_t2v \
  --parallel ulysses --parallel-vae --parallel-text-encoder --cache
 
torchrun --nproc_per_node=4 generate.py ltx2_t2v \
  --parallel tp --parallel-vae --parallel-text-encoder --cache
 
torchrun --nproc_per_node=4 generate.py ltx2_i2v \
  --parallel ulysses --parallel-vae --parallel-text-encoder --cache
 
torchrun --nproc_per_node=4 generate.py ltx2_i2v \
  --parallel tp --parallel-vae --parallel-text-encoder --cache
```
 
### 0x6.2 serving（T2V/I2V 用 env 切 pipeline）
 
T2V：
 
```bash
CACHE_DIT_LTX2_PIPELINE=t2v CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
  -m cache_dit.serve.serve \
  --model-path Lightricks/LTX-2 \
  --parallel-type ulysses \
  --parallel-text-encoder \
  --parallel-vae \
  --cache \
  --ulysses-anything
```
 
I2V：
 
```bash
CACHE_DIT_LTX2_PIPELINE=i2v CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
  -m cache_dit.serve.serve \
  --model-path Lightricks/LTX-2 \
  --parallel-type ulysses \
  --parallel-text-encoder \
  --parallel-vae \
  --cache \
  --ulysses-anything
```

### 0x6.3 怎么测试

```shell
CACHE_DIT_LTX2_MODE=base python -m pytest -q cache-dit/tests/serving/test_ltx2_text2video.py
CACHE_DIT_LTX2_MODE=base python -m pytest -q cache-dit/tests/serving/test_ltx2_image2video.py
```

完整测试代码见：https://github.com/vipshop/cache-dit/blob/main/tests/serving/test_ltx2_text2video.py & https://github.com/vipshop/cache-dit/blob/main/tests/serving/test_ltx2_image2video.py


