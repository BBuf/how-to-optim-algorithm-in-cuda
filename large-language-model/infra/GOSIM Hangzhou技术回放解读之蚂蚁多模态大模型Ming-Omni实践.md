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

标题页给出主题：蚂蚁多模态大模型实践。正文的主线是 Ming-Omni，它不是单纯 VLM，而是把图像、视频、音频、文本理解和生成放进一个更统一的模型体系里。

这类模型的难点在接口统一：图像/视频是连续视觉特征，音频既可能是连续特征也可能是离散 token，文本走 LLM，图像生成又要接 diffusion head。后面的代码拆解会围绕这些接口展开。

### Slide 2：目录：模型族、Ming-Omni、技术细节

<img src="https://files.mdnice.com/user/59/fad02339-9622-49b4-9d54-49ace65e8975.png" referrerpolicy="no-referrer" />

目录分四段：百灵大模型技术布局，多模态技术趋势及蚂蚁多模态演进，Ming-Omni 关键技术，百灵大模型演进。前两段解释为什么要做统一多模态，第三段才进入模型结构、训练和生成。

代码部分对照公开 Ming 仓库，看它如何把视觉、音频和图像生成接进同一个模型类和 `generate` 流程。这样读 slides 不会停在效果展示，而是能看到输入 embedding、condition embeddings 和 diffusion 分支怎么接。

### Slide 3：Ling/Ring/Ming 模型布局

<img src="https://files.mdnice.com/user/59/1a4004e3-2f9c-4b27-9526-2b396c401898.png" referrerpolicy="no-referrer" />

这一页是百灵大模型技术布局，信息量很大。上层是应用：医疗管家、金融管家、生活管家、CodeFuse、智能客服、保险助手、医生助手等；中间是语言、代码、行业和多模态模型；底座则包括算力、安全力和知识力。

Ling/Ring/Ming 是模型族分工。Ling 更偏语言基础模型，Ring 强调推理与效率，Ming 承担多模态方向。理解这个布局有助于看后面的 Ling-lite MoE 作为语言底座：Ming-Omni 不是从零做一个多模态模型，而是在已有语言能力上接入视觉、音频和生成分支。

### Slide 4：模型族能力边界

<img src="https://files.mdnice.com/user/59/e1d5d067-1ae9-404d-bd60-d204dbacd315.png" referrerpolicy="no-referrer" />

这一页把四条路线列得很清楚。Ling-lite/Ling-mini 走“小身材，大智慧”的高性价比语言路线；Ring-lite/Ring-mini 用 Joint RL 深挖难例、提升推理水平；Ring-lite-linear 走混合线性注意力，压低长文本冗余；Ming-lite-omni 则把听觉、视觉等多种能力融合起来。

所以 Ming-Omni 的目标不是给每个模态单独挂一个服务，而是在一个模型体系里统一理解和生成。这个目标会带来训练难题：不同模态的 loss 收敛速度不一致，理解任务和生成任务的表示空间也容易分裂。

### Slide 5：章节过渡：Ming-Omni

<img src="https://files.mdnice.com/user/59/1fcdb93c-7c2b-44df-94a6-70ba26d2c508.png" referrerpolicy="no-referrer" />

这一页是目录过渡，把话题从模型族布局切到多模态技术趋势。它提示后面要回答两个问题：行业为什么从“多模态理解”走向“理解与生成统一”，以及蚂蚁为什么选择 Ming-Omni 这条路线。

全模态模型的工程难点在训练目标、token 表示和生成 head 如何同时工作。仅接入多个 encoder 只能解决输入问题，真正复杂的是这些模态如何共享上下文、如何路由到专家、如何把 LLM hidden states 变成图像或音频生成条件。

### Slide 6：多模态走向全模态统一

<img src="https://files.mdnice.com/user/59/5a51c0cf-43ae-40c6-9306-fd6b8662353a.png" referrerpolicy="no-referrer" />

趋势页把横轴画成“理解 -> 生成 -> 理解生成统一”，纵轴是“单模态 -> 多模态”。左侧是图文理解、语音理解这些单向理解模型，中间是图像/视频/语音/3D 生成模型，右侧则是 Gemini 2.5、GPT-4o、Qwen-Omni、Janus、Bagel、Ming-lite-omni 这类理解与生成统一的方向。

右边两段小字从应用和技术两侧解释原因。应用上，人的自然交互是认知和表达的闭环，超级入口需要连接数字世界和物理世界；技术上，统一理解与生成能让语言特性扩展到视觉、语音、3D，带来更深的跨模态理解和更强的生成能力。Ming-Omni 的位置就在这个右上角。

### Slide 7：Ming scaling 轨迹

<img src="https://files.mdnice.com/user/59/7b3e3cbf-5425-4bcf-bdb5-d5657cbc1d8a.png" referrerpolicy="no-referrer" />

Ming scaling 轨迹把多个模型放在一张坐标图上：从 Qwen2.5-VL、Qwen-Omni、Ming-lite-uni、Ming-lite-omni-preview，到 Ming-flash-omni、Ming-Omni、Ming-world++。右侧 Ming-Omni 的 bullet 写到：功能对齐 GPT-4o 的全模态模型，统一 tokenizer 和训练目标，通过 MoE 架构和训练策略优化实现模态统一。

这页还强调 scaling 带来的工程代价：训练语料需求增加，训练算力和 AI 工程全链路迭代效率要求提升，评测数据集规模和评测效率也要提升。MoE 可以扩大总参数并控制 active 参数，但训练和推理要处理路由、专家负载和多模态数据调度。

### Slide 8：章节过渡：技术方案

<img src="https://files.mdnice.com/user/59/fee97c6c-1e93-4dfb-b13c-91c9947a85d4.png" referrerpolicy="no-referrer" />

这一页是进入 Ming-Omni 关键技术的目录过渡。后面三张核心 slide 分别讲：模型能力与特性、全模态架构、混合训练、视觉理解与生成统一。

读这部分时可以抓住三个实现问题：输入 embedding 怎么拼，模态路由怎么做，生成条件怎么从 LLM hidden states 里取。公开 Ming 代码里的 `patch_continuous_features`、audio encoder、vision projector、diffusion sample 都能对上这三件事。

### Slide 9：Ming-Omni：看、听、说、画

<img src="https://files.mdnice.com/user/59/9a6fb7a1-8eea-4ed9-9cef-f1cc04dc90eb.jpg" referrerpolicy="no-referrer" />

Ming-Omni 的能力页写了三组特性。第一组是简洁模型结构：基于 Ling-lite 的 MoE 架构，通过架构设计和多阶段训练统一理解与生成；Ming-lite-omni v1.5 支持音、视、图、文全模态理解和生成。第二组是跨模态融合与统一：按训练阶段、数据模态两个维度调控训练数据配比，并引入目标函数动态调权算法。第三组是理解与生成统一：生成式检测分割增强感知，多尺度可学习令牌和表征对齐统一图像理解与生成，还支持多模态对话和 TTS。

代码里能看到 vision transformer、Whisper audio encoder、MoE LLM、image generation diffusion head 这些模块都在同一类里。slide 上说的“统一”，落到实现不是所有模态共用一个 head，而是统一进入 LLM 上下文，再按输出模态接不同生成分支。

### Slide 10：能力展示

<img src="https://files.mdnice.com/user/59/f147802a-e703-417e-9bdf-2216f8f305c4.png" referrerpolicy="no-referrer" />

能力展示页包含音视频交互、多模态推理、图像分割、生成式编辑、人像编辑和 ID 一致性保持。它的作用是说明 Ming-Omni 覆盖的不是单一 VQA，而是感知、推理、编辑和生成的组合任务。

技术上要注意：不同输出模态不一定走同一个 head。文本走 LLM generate，图像生成走 diffusion loss sample，语音可能走 talker 或音频 token 生成。统一模型要把上下文和条件组织好，再把生成任务分发给合适的输出分支。

### Slide 11：Ming-Omni 架构

<img src="https://files.mdnice.com/user/59/e34996d6-4e24-4420-9aae-01c9c69cfbc0.png" referrerpolicy="no-referrer" />

这页左侧图例先给 token 类型：蓝框是 text token，黄框是 vision token，绿框是 audio token，灰框是 pad token；浅色块表示 shared experts，蓝色块表示 routing experts；火焰表示 perception training，水滴表示 generation training。底部的 image/video 先进入 Vision Encoder，audio 进入 Audio Encoder，连续特征被插回 token 序列后送进 Ling 底座。中间的 Ling 由 Attention Layer 和 MoE FFN 堆叠，右下角还画了 T/V/A Router，说明文本、视觉、音频 token 可以走不同模态路由，再和 shared/routing experts 组合。

上方的 Audio Decoder 用绿色 audio token 做音频生成，右上角的图像生成分支则从紫色 multi-scale learnable tokens 出发，经过 Connector 和 Multi-scale DiT Blocks 生成多尺度图片。右侧文字把特性讲清楚了：基于 Ling-lite 1.5(16B A3B)，在 Ling-lite 上引入模态专属路由器，并且按训练阶段、数据模态两个维度调度训练流程。公开代码里 `BailingMM2NativeForConditionalGeneration` 把 vision/audio/LLM/image generation 模块放在同一个模型类里，正好对应这张图。

### Slide 12：混合全模态训练

<img src="https://files.mdnice.com/user/59/c0a67340-13b9-4a1e-8200-3e8e5c7f1ac2.png" referrerpolicy="no-referrer" />

这页左上讲预训练梯度累积策略：每个 step 对各模态各采一个 batch 分别算 loss，再做梯度加权更新。右上虚线框列了训练数据形态：图像文本对、音频文本对、OCR、纯文本、交错图文序列、视频文本对，它们都进 `BailingMM-Native`，forward 得到 `loss1` 到 `loss6`，backward 时再合到参数更新。

下半部分是 SFT 的动态均衡策略。左下普通全模态联合 SFT 是文本、图像、视频、音频各自输出 loss_t/loss_i/loss_v/loss_a，然后直接更新；旁边曲线说明不同模态收敛趋势不一致，同一个训练步上有的模态已经接近最优，有的还没追上。右下动态策略会先评估各模态性能趋势，把 loss 乘以动态权重，比如图里示意的 `loss_t x 0.90`、`loss_i x 0.85`、`loss_v x 0.30`、`loss_a x 2.00`，再更新参数。这里的目标不是让每个 loss 数值一样，而是让各模态的能力进度更接近。

### Slide 13：视觉理解和生成统一

<img src="https://files.mdnice.com/user/59/08c8ca6d-57fa-4131-926d-2c1d1334aa0a.png" referrerpolicy="no-referrer" />

这页左侧把训练和推理画在一起。训练时，image tokens、text tokens 和 multi-scale queries 进入 Multi-modal LLM；response 侧的多尺度 query 通过 Connector 接到 DiT Blocks，分别服务 128px、256px、512px 的 denoising objective，同时还有 representation alignment，让不同尺度的表征能对齐。推理时，noisy inputs 进入多层 DiT Block，最终生成 512px 图片。

右侧对比了不做 scale alignment 和做 scale alignment 的生成效果。上半部分没有对齐时，16 tokens/128px、64 tokens/256px、256 tokens/512px 的图像细节和语义一致性不稳定；下半部分做了跨尺度对齐后，同一个 “A colorful bird perched on a tree branch” prompt 在不同分辨率下更连贯。右边两张曲线则说明相比 MetaQuery，HieraQuery 随 query token 增多，Geneval overall 上升，MJHQ-30K FID 下降。代码里这部分落在 image generation 分支：LLM 给出 condition embeddings，diffusion head/DiT 用这些 condition 做采样。

### Slide 14：章节过渡：开源与演进

<img src="https://files.mdnice.com/user/59/35d7c4aa-b377-49d0-949b-4394107063fb.png" referrerpolicy="no-referrer" />

这一页是进入“百灵大模型演进”的目录过渡。前面已经讲了 Ming-Omni 的结构和训练，后面会回到版本演进和开源资料。

开源资料里有 GitHub、HuggingFace、ModelScope、technical report 和 project page。下面对代码时用的是当前 Ming 仓库，版本比 slides 中 v1.5 更新，因此正文尽量讲稳定的实现路径：vision/audio 特征接入、LLM 上下文、image generation 条件。

### Slide 15：版本演进

<img src="https://files.mdnice.com/user/59/57ff0717-cd93-4b2f-a357-c9b6dd841abf.png" referrerpolicy="no-referrer" />

版本演进页把“效率与智能的进阶之路”分成五个点：扫码级 AI、MoE、线性 Attention、多模态融合、AI 优化。它想表达的是，模型演进不只靠参数变大，还包括性价比、架构、长文本效率、模态理解和评测标准。

公开 README 里已经写到 Ming-flash-omni 2.0，底座是 Ling-2.0 MoE，总参数和 active 参数比 slides 里的版本更大。写博客时需要注意版本差异：slides 解释 v1.5 的设计，源码讲解则以当前公开仓库的实现为准。

### Slide 16：论文、模型和代码链接

<img src="https://files.mdnice.com/user/59/920df130-eec6-4982-9baf-b12d8af96124.png" referrerpolicy="no-referrer" />

链接页给出 Ming-lite-omni 1.5 和 Ming-lite-uni 的公开入口，包括 HuggingFace、ModelScope、GitHub、technical report 和 project page。它也说明用户不只能读论文，还能直接下载权重、看 cookbook、跑 demo。

写博客时这类链接比抽象评价更重要。多模态模型的复现门槛高，processor、依赖、图片/音频预处理、推理参数都需要样例支撑。公开仓库里的 README 和 cookbook 是读者把这篇文章落到本地实验的入口。

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
