> 多模态文章很容易写成效果展示。这里尽量往结构和代码上收：视觉、音频、图像生成和 LLM 到底怎么接在一起。

# 0x0. 前言

多模态模型文章很容易写成效果展示，这篇尽量落到结构和代码：输入模态怎么投影到 LLM，图像生成怎么拿 condition，音频 token 怎么桥接。

# 0x1. 资料和代码落点

代码落点：

- Ming 仓库：`modeling_bailingmm2.py`，`BailingMM2NativeForConditionalGeneration` 把 vision/audio/LLM/image generation 模块组织在一起。
- Ming 仓库：`modeling_utils.py`，`patch_continuous_features`、`encode_audio_segments` 负责连续模态特征接入文本 embedding。
- Ming 仓库：`s3bpe_tokenizer.py`，展示音频离散 token 到 BPE token 的桥接方式。
- Ming README：公开模型、HF/ModelScope、cookbook 和 Ming-flash-omni 2.0 信息。

# 0x2. Slides 逐页解读

### Slide 1：蚂蚁多模态大模型实践

<img src="https://files.mdnice.com/user/59/24549154-9b5c-4227-9019-588e4427f224.png" referrerpolicy="no-referrer" />

主线是 Ming-Omni：它不是单纯 VLM，而是把图像、视频、音频、文本理解和生成放进一个更统一的模型体系里。

### Slide 2：目录：模型族、Ming-Omni、技术细节

<img src="https://files.mdnice.com/user/59/fad02339-9622-49b4-9d54-49ace65e8975.png" referrerpolicy="no-referrer" />

目录先介绍 Ling/Ring/Ming，再进入 Ming-Omni 的能力和技术细节。代码部分对照公开 Ming 仓库，看它如何把视觉、音频和图像生成接进同一个 `generate`。

### Slide 3：Ling/Ring/Ming 模型布局

<img src="https://files.mdnice.com/user/59/1a4004e3-2f9c-4b27-9526-2b396c401898.png" referrerpolicy="no-referrer" />

Ling/Ring/Ming 是模型族分工。Ling 更偏语言基础模型，Ring 和 Ming 承担多模态方向。理解这个布局有助于看后面的 Ling-lite MoE 作为语言底座。

### Slide 4：模型族能力边界

<img src="https://files.mdnice.com/user/59/e1d5d067-1ae9-404d-bd60-d204dbacd315.png" referrerpolicy="no-referrer" />

模型族能力边界页展示不同模型覆盖的输入/输出模态。Ming-Omni 的野心在于统一理解和生成，而不是给每个模态单独挂一个服务。

### Slide 5：章节过渡：Ming-Omni

<img src="https://files.mdnice.com/user/59/1fcdb93c-7c2b-44df-94a6-70ba26d2c508.png" referrerpolicy="no-referrer" />

过渡页把话题收束到 Ming-Omni。全模态模型最难的地方不是多接几个 encoder，而是训练目标、token 表示和生成 head 怎么同时工作。

### Slide 6：多模态走向全模态统一

<img src="https://files.mdnice.com/user/59/5a51c0cf-43ae-40c6-9306-fd6b8662353a.png" referrerpolicy="no-referrer" />

趋势页列了 GPT-4o、Gemini、Qwen-Omni、Janus 等路线。共同点是理解和生成在逐渐合并：模型不仅回答图片里有什么，还要能生成图像、语音或其它模态。

### Slide 7：Ming scaling 轨迹

<img src="https://files.mdnice.com/user/59/7b3e3cbf-5425-4bcf-bdb5-d5657cbc1d8a.png" referrerpolicy="no-referrer" />

Ming scaling 轨迹从 dense 到 MoE，从单一模态融合到更统一的 omni 模型。MoE 的好处是扩大总参数同时控制 active 参数，代价是训练和推理路由更复杂。

### Slide 8：章节过渡：技术方案

<img src="https://files.mdnice.com/user/59/fee97c6c-1e93-4dfb-b13c-91c9947a85d4.png" referrerpolicy="no-referrer" />

技术方案过渡页提示后面要讲模态融合、训练和生成。多模态文章只讲效果展示不够，能落到实现里的问题是输入 embedding 怎么拼、生成条件怎么取。

### Slide 9：Ming-Omni：看、听、说、画

<img src="https://files.mdnice.com/user/59/9a6fb7a1-8eea-4ed9-9cef-f1cc04dc90eb.jpg" referrerpolicy="no-referrer" />

Ming-Omni 的能力页写得很满：看、听、说、画。代码里能看到 vision transformer、Whisper audio encoder、MoE LLM、image generation diffusion head 这些模块都在同一类里。

### Slide 10：能力展示

<img src="https://files.mdnice.com/user/59/f147802a-e703-417e-9bdf-2216f8f305c4.png" referrerpolicy="no-referrer" />

能力展示页主要是样例。技术上要注意：不同输出模态不一定走同一个 head，文本走 LLM generate，图像生成走 diffusion loss sample，语音可能走 talker。

### Slide 11：Ming-Omni 架构

<img src="https://files.mdnice.com/user/59/e34996d6-4e24-4420-9aae-01c9c69cfbc0.png" referrerpolicy="no-referrer" />

架构页和公开代码能对上：vision encoder 输出视觉 token，经 MLP 投影到 LLM hidden size；audio 用 Whisper encoder 加 Conv1d/MLP 投影；语言底座是 BailingMoeV2/Ling 系。

### Slide 12：混合全模态训练

<img src="https://files.mdnice.com/user/59/c0a67340-13b9-4a1e-8200-3e8e5c7f1ac2.png" referrerpolicy="no-referrer" />

混合全模态训练要解决 loss scale。文本、图像、音频的 token 数和 loss 形态差异很大，slides 里提到 weighted losses 和 dynamic balancing，本质是避免某个模态主导梯度。

### Slide 13：视觉理解和生成统一

<img src="https://files.mdnice.com/user/59/08c8ca6d-57fa-4131-926d-2c1d1334aa0a.png" referrerpolicy="no-referrer" />

视觉理解和生成统一这页对应 image generation 分支：模型可以从 LLM hidden states 或视觉特征取 condition embedding，再调用 diffusion head 采样。

### Slide 14：章节过渡：开源与演进

<img src="https://files.mdnice.com/user/59/35d7c4aa-b377-49d0-949b-4394107063fb.png" referrerpolicy="no-referrer" />

开源与演进过渡页说明 Ming 已经有公开仓库、HF 和 ModelScope 权重。下面对代码时用的是当前 Ming 仓库，版本比 slides 中 v1.5 更新。

### Slide 15：版本演进

<img src="https://files.mdnice.com/user/59/57ff0717-cd93-4b2f-a357-c9b6dd841abf.png" referrerpolicy="no-referrer" />

版本演进页展示 Ming-lite-omni 到 Ming-flash-omni。公开 README 里已经写到 Ming-flash-omni 2.0，底座是 Ling-2.0 MoE，总参数和 active 参数比 slides 里的版本更大。

### Slide 16：论文、模型和代码链接

<img src="https://files.mdnice.com/user/59/920df130-eec6-4982-9baf-b12d8af96124.png" referrerpolicy="no-referrer" />

链接页给出 arXiv、GitHub、HF、ModelScope。写博客时这类链接比“模型很强”更重要，因为读者可以直接下载权重和跑 cookbook。

### Slide 17：总结

<img src="https://files.mdnice.com/user/59/2ea1b2c3-646f-4280-a6f9-e112cb3e39b2.png" referrerpolicy="no-referrer" />

总结页回到实践：Ming-Omni 的关键不是堆模态，而是把连续特征、离散 token、生成 condition 和多目标训练放在一个可维护的模型实现里。

# 0x3. 关键代码拆解

Ming 的模型类很直观。初始化时分别创建 vision、audio、LLM 和投影层：

```python
if self.config.vision_config:
    self.vision = Qwen3MoeVisionTransformer(self.config.vision_config)

if self.config.audio_config:
    self.audio = WhisperAudioEncoder(**self.config.audio_config.whisper_encoder_config)

self.model = BailingMoeV2ForCausalLM(self.config.llm_config)

mlp_modules_img = [nn.Linear(self.vision.image_emb_dim, self.model.config.hidden_size)]
self.linear_proj = nn.Sequential(*mlp_modules_img)
```

模型入口先由 processor 把消息里的文本、图片、视频、音频拆开。官方 README 里的推理路径很有代表性：`apply_chat_template` 负责把多轮对话变成文本 prompt，`process_vision_info` 负责收集视觉/音频对象，最后 processor 一次性产出 `input_ids`、`pixel_values`、`pixel_values_videos`、`audio_feats` 等字段：

```python
text = processor.apply_chat_template(
    messages,
    sys_prompt_exp=sys_prompt_exp,
    use_cot_system_prompt=use_cot_system_prompt,
)
image_inputs, video_inputs, audio_inputs = processor.process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    audios=audio_inputs,
    return_tensors="pt",
    audio_kwargs={"use_whisper_encoder": True},
).to(model.device)

for k in inputs.keys():
    if k in ("pixel_values", "pixel_values_videos", "audio_feats"):
        inputs[k] = inputs[k].to(dtype=torch.bfloat16)
```

这段可以对上 Slide 9/11：Ming-Omni 不是在 LLM 前面简单拼几个 encoder，而是先把不同模态的输入整理成统一 batch，再在模型内部用 placeholder 位置把连续特征补回 embedding 序列。

视觉特征先过 vision transformer，再投影到 LLM hidden size，并做 normalize：

```python
def extract_image_feature(self, pixel_values, grid_thw):
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        image_embeds = self.vision(pixel_values, grid_thw=grid_thw)
    image_embeds = self.linear_proj(image_embeds)
    image_embeds = F.normalize(image_embeds, dim=-1)
    return image_embeds
```

音频特征则经 Whisper encoder 和下采样投影：

```python
def extract_audio_feature(self, audio_feats, audio_feats_lengths, use_whisper_encoder=False):
    audio_embeds, _, audio_embeds_lengths = encode_audio_segments(
        encoder=self.audio,
        proj_layer=self.linear_proj_audio,
        wav_feats=audio_feats,
        wav_feats_lengths=audio_feats_lengths,
        audio_config=self.config.audio_config,
    )
    if self.config.audio_config.norm_query_embeds:
        audio_embeds = F.normalize(audio_embeds, dim=2)
    return audio_embeds.to(audio_feats.dtype), audio_embeds_lengths
```

连续特征如何塞回文本序列，可以看 `patch_continuous_features`。placeholder token 先占位，encoder 输出后覆盖对应 embedding：

```python
def patch_continuous_features(input_embeddings, placeholder_loc_lens, encoded_feats, encoded_feat_lens):
    batch_size = input_embeddings.size(0)
    for i in range(batch_size):
        audio_feat_start = 0
        for j in range(placeholder_loc_lens.shape[1]):
            placeholder_start = int(placeholder_loc_lens[i, j, 0].item())
            placeholder_len = int(placeholder_loc_lens[i, j, 1].item())
            if placeholder_len <= 0:
                break
            feat_len = int(encoded_feat_lens[i, j].item())
            target_len = min(feat_len, placeholder_len)
            input_embeddings[i, placeholder_start:placeholder_start + target_len] = \
                encoded_feats[i, audio_feat_start:audio_feat_start + target_len]
            audio_feat_start += feat_len
    return input_embeddings
```

图像生成分支走另一条路径。`generate(image_gen=True)` 时会拿 condition embeddings，然后调用 diffusion loss 的 sample：

```python
if image_gen:
    condition_embeds = self.get_condition_embeds_for_image_gen(
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_embeds=image_embeds,
        use_cache=use_cache,
        image_grid_thw=image_grid_thw,
        llm_hidden_states=image_gen_llm_hidden_states,
    )

    image = self.diffusion_loss.sample(
        condition_embeds=condition_embeds,
        negative_condition_embeds=negative_condition_embeds,
        steps=image_gen_steps,
        seed=image_gen_seed,
        cfg=image_gen_cfg,
        height=image_gen_height,
        width=image_gen_width,
    )
```

音频离散 token 的 BPE 桥接可以看 `S3BpeTokenizer`：

```python
def encode(self, s3_code: List[int]):
    new_s3_code = [self.s3_to_zh[x] for x in s3_code]
    new_s3_code = ''.join(new_s3_code)
    output = self.sp.encode(new_s3_code)
    return output.ids, output.tokens
```

这类实现说明 Ming-Omni 的“统一”不是所有东西都变成同一种 token，而是把连续特征、离散音频 token 和生成条件统一到 LLM 可处理的接口上。

# 0x4. 小结

Ming-Omni 的工程重点在模态接口。vision/audio encoder 输出连续特征，LLM 负责统一上下文，图像生成再从 hidden states 取 condition 调 diffusion head。这个结构比单纯列能力更值得看。
