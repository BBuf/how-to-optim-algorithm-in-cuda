# 最新体会

最近用Codex在SGLang中蹬了一堆个人感觉算是比较solid的PR和feature实现，这里复盘下。基本都是我来出思路和中途干预，然后Codex来实施，我基本没写什么代码。放在之前古法时代，没有用Codex对我来说大概是3-4个月的工作量（其实还有一些refactor，skill，ci，benchmark，我觉得实际工作量对之前的我来说可能还是不止3-4个月的，毕竟是业余爱好），现在缩短为了1个半月。然后下面只涉及到性能优化和重要feature实现，全程用Codex蹬的。

一个可能担忧是这些代码如果大多数都是Codex写得怎么保证代码质量和准确性，我有几个体会：

- 首先代码的风格上要尽量适应我以前写PR的风格，这个其实抓一堆自己之前的PR就能做到。比如Agent很容易弄一堆防御过度的try catch，其实很多是没必要的。然后Agent倾向于把import都插入到代码中，这种问题都可以用自己之前提代码的风格来统一解决掉。
- 其次，Agent改的任何函数代码自己都要完全理解，思考漏洞，如果你提交的Agent写出来代码连你自己都没完全理解将是非常危险的炸弹。我举个例子就是如果你让Agent写一个nvfp4/fp8 quant的测试，它可能会写出比较错误的测试方法出来比如直接做数值比较，你去仔细看了之后才能发现然后让他改成正确的精度对比方式，例如DeepGEMM那种cosine similarity的方式。CI监控和测试方面的要求也更高了，我们只能不断加强这方面的能力才能把Agent可能引入的漏洞拒之门外。
- 最后，涉及到性能优化和重大feature时尽量找专家Review。目前的现状对框架开发者Review的能力提出了更高的要求，例如之前Review是和一个人或者几个人搏斗，现在还要和Claude Code/Codex等搏斗，你甚至都不知道你回复一些Review意见之后你收到的回复是真人还是Claude Code/Codex写的。可能想了半天觉得回复好像是对的，结果对方根本就没有人工思考的过程了，是直接让Agent回复的。并且Agent带来的巨量PR已经让Review变得雪上加霜。目前比较流行的方式是Claude Code Review Codex的代码，这个也可以为我们提供很大的帮助。目前我在Codex里面连接了Github，如果涉及到Agent来回复头像旁边会自动多一个logo方便区分。

![](https://files.mdnice.com/user/59/418edf61-44d7-425e-9f47-dcb874d2a13f.png)

时代到这里了，也不能拒绝Agent带来的生产力，大势所趋，怎么都挡不住了。比如看下方的vLLM/SGLang的PR数量图，一天上百个怎么Review呢？

![](https://files.mdnice.com/user/59/0ed0b274-17b6-4183-a4a2-85bbb940230d.png)

我自己是充分肯定和赞同Agent编程的，只是我们需要更加小心潜在的风险，特别是在大型的开源项目上知道自己在做什么是非常重要的。


![](https://files.mdnice.com/user/59/0879478b-c712-4411-acff-b64606dc720b.png)

但200美金的计划显然是不够的。

# 一件印象深刻的事

给LTX2支持FP8的时候第一次拿到的输出视频会乱码，我i告诉Codex这个视频结果是乱码的让它debug，它找到了bug并且修复了。然后亮点是它debug修复的过程中除了比较图片的PSNR和MAE之外还会用ffmpeg去截一张生成视频的图然后用多模态的方式读这个图判断画面是正常还是乱码。这个指令遵循的能力令我印象深刻，和人的思路越来越像。

# 在SGLang 用 Codex 1个月蹬出来的结果

得到这些结果不是完全冷启动做的，高度依赖了目前SGLang里面的Profile SKILLS以及Benchmark SKILLS，具体可见：https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/.claude & https://github.com/sgl-project/sglang/tree/main/.claude/skills

你可以告诉Agent让它帮你安装SGLang的skills。

- https://github.com/sgl-project/sglang/pull/20395：Qwen-Image-Edit 的 modulation 链路做了一次 Triton 融合，覆盖 `residual + layernorm + scale_shift + gate + select01` 这一串操作。benchmark 和 correctness test 也一起补了。denoise per-step 从 `0.6383s` 降到 `0.6256s`，约 `2%`。
- https://github.com/sgl-project/sglang/pull/20576：删掉 Hopper 上 upstream FlashAttention v3 的特殊路径。前面的 diffusion benchmark 重新跑过之后，这条分支已经没什么优势了，继续留着主要是在加维护负担。
- https://github.com/sgl-project/sglang/pull/20632：新增 `bench_norm_impls.py`，把 RMSNorm 和 `fused_add_rmsnorm` 的几套实现放到同一套脚本里测，shape 也按 diffusion 的真实 workload 来。这个 PR 没有直接加速，作用是把后面很多 kernel 选择和回归判断先量清楚。
- https://github.com/sgl-project/sglang/pull/20699：把 flashinfer rope 这段改成了 compile-friendly custom op，解决 `torch.compile` 下的 graph break。实际加速约 `3%`。
- https://github.com/sgl-project/sglang/pull/20962：修的是 `Z-Image` 上 `fp32 RMSNorm + wrap_triton` 触发的 compile fallback。范围收得很窄，就是把最容易掉回 eager 的那条路径换掉。`Z-Image-Turbo` 9-step denoise `1362.583 ms -> 744.656 ms`，50-step `6297.997 ms -> 4652.565 ms`，收益在 `26%~45%` 之间。
- https://github.com/sgl-project/sglang/pull/21248：把 `Wan/MOVA` 在高显存 GPU 上自动开启 `dit_layerwise_offload` 的默认逻辑关了。原因也不复杂，在 H200 这种卡上它会影响模型执行延迟，几组 benchmark 里回归大约 `60%~80%`。
- https://github.com/sgl-project/sglang/pull/21318：继续优化 `Qwen` 的 select01 Triton kernel，核心是 pointer-select，只加载真正会走到的那一路 `scale/shift/gate`。Qwen DenoisingStage `12788.15 ms -> 12432.77 ms`，E2E `12838.08 ms -> 12526.77 ms`，约 `2.4%~2.8%`。
- https://github.com/sgl-project/sglang/pull/21387：Triton rotary embedding 改成一次处理多个 head，少了不少重复的 `cos/sin` 访问和 launch 开销。真正落到模型上是 `HunyuanVideo` 约 `1.1%~1.3%`。
- https://github.com/sgl-project/sglang/pull/21440：加了一个 JIT CUDA kernel，把 `Q/K RMSNorm + RoPE` 融在一起，接到了 `Qwen`、`FLUX`、`Z-Image` 这些 diffusion DiT 路径上。`Qwen-Image` denoise `14.43s -> 12.36s`，约 `14.35%`；其他模型多半在 `1%` 左右，个别接近持平。
- https://github.com/sgl-project/sglang/pull/21503：继续压 `qknorm_across_heads` 的寄存器和 shared memory 占用，把原来一个 CTA 里一起做的 `q/k` 拆开了。microbench 在 `1.07x~1.15x`，occupancy 从 `45.25%` 提到 `88.17%`。
- https://github.com/sgl-project/sglang/pull/22091：把 diffusion NVFP4 的默认 backend 切到 CUTLASS。这里不太适合写一个统一的“提速多少”，因为它做的是默认选择。评论里更新后的结果是 `265` 个真实 shape 里 CUTLASS 赢了 `263` 个，默认值改成它基本没什么悬念。
- https://github.com/sgl-project/sglang/pull/22365：diffusion 的 ModelOpt FP8 路线补齐了，运行时 quant path、checkpoint 转换和轨迹比对工具都落了进去。`FLUX.2` 总时长 `24.47s -> 17.13s`、denoising `23.21s -> 16.21s`，约 `30%`；`Wan2.2` 小很多，在 `3.7%~3.8%`。
- https://github.com/sgl-project/sglang/pull/22574：给 `FLUX.1-dev` 接上 ModelOpt NVFP4，builder、nibble swap、测试这些都补齐了。`4x RTX 5090` 上 BF16 denoise `37.6940s`，NVFP4 denoise `29.0421s`，约 `22.95%`；end-to-end 也是 `22.90%` 左右。
- https://github.com/sgl-project/sglang/pull/22594：修 quantized DiT 的 layerwise offload，主要是 stride 保存、buffer 对齐和 `modelopt_fp8` offload 支持这几个点。它不主打延迟，评论里更有价值的是显存数据，几组模型的 `peak_allocated_mb` 下降了 `87%~92%`。
- https://github.com/sgl-project/sglang/pull/22664：就是给 `Qwen3NextForCausalLM` 默认打开 FlashInfer allreduce fusion。H100、`tp=4`、`Qwen3-Coder-Next` 上 throughput `5.49 req/s -> 9.41 req/s`，约 `71.4%`，TTFT 和 TPOT 也一起降了不少。
- https://github.com/sgl-project/sglang/pull/22681：把 `Wan2.2` 的 ModelOpt NVFP4 走通了，顺手处理了 `transformer_2` 保留在 BF16 checkpoint 和 scheduler 兼容的问题。`B200` 上 no-compile 的结果很直接：BF16 `55.89s`，NVFP4 `25.72s`，E2E 约 `54.0%`，denoising 约 `56.2%`。
- https://github.com/sgl-project/sglang/pull/22717：给 diffusion ModelOpt NVFP4 加了 `flashinfer_trtllm` backend。`FLUX.1-dev 512x512 8 steps` 约 `10.6%`，`1024x1024 20 steps` 约 `14.6%`，`FLUX.2-dev 512x512 8 steps` 约 `30.9%`。
- https://github.com/sgl-project/sglang/pull/22861 针对 LTX-2.3 one-stage denoise 路径：首先将原本每步分别执行的 `cond / neg / perturbed / modality` 多个 guider forward 合并成一次 batched forward，显著减少 transformer forward 次数以及 NCCL/launch/Python 调度开销；其次缓存 one-stage 每步重复使用的 RoPE 坐标和扩 batch 后的静态张量，避免 denoise loop 中反复构造；最后把 transformer block 中分散的 `RMSNorm + scale/shift + residual/gate` elementwise 链接入 fused kernel，降低中间张量和显存读写开销。优化通过 `legacy_ltx23_one_stage_semantics` 开关限制在 one-stage 生效，不影响 two-stage 路径。B200 稳定复测中，one-stage E2E 从 `36.71 ± 2.56s` 降到 `22.69 ± 0.38s`，约 `1.62x`；denoise 从 `34.00 ± 2.57s` 降到 `19.78 ± 0.12s`，约 `1.72x`。

其实做出来的事情远比这里列的更多，但是因为缺少手动验证以及Token消耗过快的原因就不继续列举了。然后Codex debug真是一把好手

# 为什么不尝试Claude Code

![](https://files.mdnice.com/user/59/030f41d0-d3e1-4347-b4ad-ba0f54b0ee90.png)

上周在邮箱里面看到我也入选了Claude for Open Source Program.

但是我用邮箱登陆Claude Code的时候发现我的邮箱被Anthropic Ban掉了

![](https://files.mdnice.com/user/59/13412923-78e4-4b5e-a511-e79add4ef49f.png)

然后今天收到A➗要使用反馈的邮件：

![](https://files.mdnice.com/user/59/9879fb40-d4c0-4247-8f4a-01fd69285f84.png)

纯被溜着玩属于是。后续要身份验证对国内小伙伴来说就更难搞了，还好Codex目前大部分我要Vibe Coding做事情都能接住且没这么多限制。








