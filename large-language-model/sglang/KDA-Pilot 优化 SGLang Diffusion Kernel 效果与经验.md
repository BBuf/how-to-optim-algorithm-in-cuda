# KDA-Pilot 优化 SGLang Diffusion Kernel 的效果与经验

## 任务定义

对于 Diffusion ，由于大多数 Diffusion 的任务都是Compute Heavy的，然后GEMM(cublas)和Attention（FA3）主导了计算，所以相比于 LLM 可以用 Kernel Agent 的空间不会太大，且一般bs=1。

需要优化的 Kernel 主要集中在目前 SGLang Diffusion 中已经通过各种方式实现的一些 Fuse 或者相比于 Torch 实现更快的 Kernel。

Diffusion 任务定义的时候我就直接基于 https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile 这个skill的benchmark命令跑了20个真实的diffusion模型，然后记录每个模型里面对应算子的输入meta等信息，最后把每个kernel的不同的meta信息汇总成为一个kernel优化任务。这些任务几乎没有compute bound的任务，主要是memory bound和latency bound的。

对于LLM，以CookBook的命令为基准，在两个不同场景的数据集（chat和summay）上设置高中低不同的并发，然后运行这些模型在这些设置下的组合，同时需要profile拿到kernel占比，把除cudnn gemm和Attention之外的其它>1%占比的kernel也汇总成一个kernel优化任务，这些任务也包含不同数据集和并发下的meta信息，LLM下面记录的meta信息可能会很多，需要集中做一批过滤，具体来说通过roofline model把密度差距不大的kernel合并成一类。（现状是做这个任务的时候资源bound了，需要机器和卡，开源社区的机器已经凑不出完整的8张空卡）后续会更加关注一下gemm类的compute bound算子能优化到什么程度


## SGLang Diffusion Kernel 优化效果

下面这张表展示 B200 上的 wall 指标，也就是用户真正能看到的端到端耗时：包含 Python、dispatch、wrapper、kernel launch 以及 `cuda.synchronize()` 看到的同步开销。优化任务和复现代码统一来自 [BBuf/KDA-Pilot](https://github.com/BBuf/KDA-Pilot)。

| Kernel | B200 wall geomean | Per-shape wall speedup | Notes |
|---|---:|---|---|
| qknorm_rope | 1.1341x | large 1.145-1.279x; small 1.043-1.059x | B/T/S = tokens; H = heads; D = head dim; R = RoPE dim |
| norm_infer | 1.3523x | helios 1.201x; RMS large 1.078x; RMS small 1.634-1.641x | M/N = matrix dimensions; S = token/row count; D = hidden dim |
| rotary_embedding | 1.4912x | hunyuanvideo 2.087x; ltx2 1.133-1.622x | B = batch; S/T = seq len; H = heads; half64/half32 = RoPE half-dim bucket |
| cutedsl_norm_tanh_mul_add | 1.4953x | v1 1.602-1.625x; v2 1.378-1.394x | S = tokens; v1 = single op; v2 = dual op |
| cutedsl_norm_scale_shift | 1.3201x | firered 1.277-1.364x; helios 1.111-1.283x; hunyuan 1.388-1.516x; joyai 1.477-1.495x; mova 1.193-1.350x; qwen 1.243-1.489x; wan 1.096-1.351x | 11D = [1,1,D]; 1D = [1,D]; 1SD = [1,S,D]; g = gate tensor |
| fuse_scale_shift | 2.7499x | qwen/qwen-edit/firered small bcast11 7.365-7.891x; 8424 bcast/full3d 1.462-1.747x; gated/resgated 1.020-1.068x; hunyuanvideo 1.166-8.878x; wan 1.267-1.904x | B = batch; s/S/L = tokens; c/C = channels; bcast11 = [1,1,C] broadcast; full3d = [B,S,C]; NC = non-contiguous layout |
| group_norm_silu | 2.3118x | C-layout giant/fallback rows 0.947-1.192x; NC large rows 1.158-3.648x; small/mid C rows 1.369-4.982x; small/mid NC rows 1.791-3.240x | Shape format BxCx... from frozen HunyuanVideo rows; C = contiguous; NC = non-contiguous/channels-last path; apply/triton are both public entry paths |

下面这张表只解释 B200 kernel 任务在最新代码里实际保留下来的优化路径。这里的轮次按 `solutions.jsonl`、`docs/run_log.md` 和最终源码里的可追溯记录统计：有些是完整 RLCR round，有些是同一 round 内的候选编号或远程 run。表里的 16B/32B 指最终代码选择的每线程向量访问宽度，不是 B200 的硬件上限；B200 支持 256-bit vectorized access，但部分 kernel 为了对齐、寄存器压力、occupancy 或 cache hint 仍选择 128-bit。

| Kernel | RLCR / 候选线 | 最新代码保留的路径 | KernelWiki / 参考实现 | 本任务关键技术 |
|---|---|---|---|---|
| qknorm_rope | R4-R9：R5 staged，R8 overlay no-go，R9 in-tree arbiter pass | 走 SGLang in-tree `.cuh` 替换，不走 `kda_kernels` Python overlay；生产模板 bf16、head=128、rope=128、non-NeoX 且 token >= 512 时由 `QKNormRopeKernel` 内部转到 staged kernel，其它模板和小 shape 保持原 warp 路径 | **TensorRT-LLM PR-13052/11869 fused DiT QKNorm+RoPE；SGLang PR-15141/19059/21440/21654 fused qknorm_rope；pattern-memory-bound** | CTA-per-token，把 cos/sin 行 staged 到 shared memory 后跨 Q/K heads 复用；用 NCU 的 long-scoreboard 证据确认 device win；保留 `register_custom_op` 避免 Python dispatch tax；两 token/CTA 的 staged2 被拒 |
| norm_infer | Round 1 + Round 2；large RMS 由 cand-0008/0009 no-go 到 cand-0010/0012 promoted | fp32 LayerNorm 走 `LayerNormInferKernel`；bf16 RMS small/mid 走 one-warp-per-row；huge-S RMS 走 `RmsNormTiledKernel<128,32,bf16>` persistent grid；其余签名 fail-closed fallback | **pattern-memory-bound；technique-vectorized-loads；technique-register-budgeting；vLLM PR-31828 SM100 RMSNorm opt-in path** | LN 用 16B `float4` load/store，small RMS 用 8B bf16 vector，huge RMS 用 16B bf16 vector；small/mid 用低开销 warp-row 模式；huge-S 用 multi-row tile、half-warp segmented reduction 和 persistent whole-wave grid 隐藏 load latency |
| rotary_embedding | R0-R2 + continuation；cuda-v1 到 cuda-v7，最终保留 cuda-v6，并用 sglang main 重测 | 只对 captured standard RoPE 和 LTX2 split RoPE 签名启用 CUDA fast path；非 captured 签名回退原 SGLang baseline | **SGLang PR-24411 LTX2 split rotary；technique-vectorized-loads；vLLM PR-21126/30729 FlashInfer RoPE routing** | 128-bit vectorized load/store；standard 路径去掉 shared cos/sin 同步并把 cos/sin hoist 到寄存器；LTX2 做 block-size matching；两行/CTA、double-buffer、TMA/persistent 方向按证据拒绝 |
| cutedsl_norm_tanh_mul_add | r0-r4：baseline NCU -> hoisted tanh v1 -> launch-bounds K sweep v2 -> prefetch/v2hint no-go -> dispatch-symmetric arbiter | 两个 public custom-op 体内先试 native CUDA，`native_supported` 不满足则保留原 CuTe-DSL fallback；最终默认 K=8、PDL off、no fast-math | **pattern-memory-bound；technique-vectorized-loads；technique-register-budgeting** | 识别 production scale 是 row-invariant，hoist `tanh(scale)` 和 `1+scale2`；`__launch_bounds__` 压寄存器并保持 4 CTA/SM；bf16/fp16 用 128-bit，fp32 用 256-bit vectorized load/store；保持 exact `tanhf` 和 baseline rounding 语义 |
| cutedsl_norm_scale_shift | r0-r6：native v1 -> audit/vec16/two-pass v2 -> Welford no-go -> r6 rebind | fail-closed dispatcher 按 entry、operand class、dtype、gate、weight/bias 选 10 个 native CUDA template；其它签名回退 vendored baseline | **SGLang PR-14717 CuTe-DSL norm/scale/shift fusion；technique-vectorized-loads；technique-register-budgeting** | row-per-CTA；bf16-only bucket 用 32B vector，触及 fp32 operand 的 bucket 用 16B vector；two-pass variance 保数值契约；scalar/row/token operand 分类；Welford/Chan 单轮 merge 因依赖链变慢被拒 |
| fuse_scale_shift | RLCR Round 0-2：v0 correct port -> v1 rowgrid/flatvec/exact-C -> v2 one-pass reduction -> review-phase numerics fix | 一个 CUDA module 内部分发：EP1 走 rowgrid、flatvec 或 strided generic；EP2/EP3 走 exact-C vector kernel 或 generic；生产行不需要 baseline fallback | **SGLang PR-14717 fused norm/scale/shift family；technique-vectorized-loads；technique-cache-policy；pattern-memory-bound** | 最终代码使用 16B `uint4` vectorized streaming；`__ldcs`/`__stcs` 处理 x/out 流，`__ldg` 处理复用的 modulation row；小 S 用 flatvec 提高 occupancy；exact-C 避免 Triton block padding；bf16/fp16 用 shifted one-pass，fp32 用 centered two-pass |
| group_norm_silu | Run 1-14；Run 5 后三轮 NCU-backed 修复，Run 14 为 review-phase 最终证据 | Python 先按 group_size/layout 选择 baseline fallback 或 CUDA；CUDA 内部有 `cont_small`、`cont_split`、`nchw_last`、generic 四类 regime；giant contiguous bucket 回退 copied Triton baseline | **SGLang PR-22814/23148/23938 GroupNorm+SiLU；pattern-memory-bound；technique-vectorized-loads** | contiguous small 用 one-CTA/group 的 16B two-pass；contiguous mid 用 split-group stats、last-CTA finalize、generation counters 和 division-free apply；channels-last 直接按原布局读，shared-memory staged transpose 后写 contiguous output；giant bucket 经 NCU 证明带宽/写回受限后回退 |



## Reward Hacking

第一个是fast math在Opus 4.8的优化过程中每个kernel都默认打开了，这个是不对的，后来在prompt里面做了nvcc编译指令的强限制，fast math不要开，因为SGLang框架的kernel就没有开。

第二个是不要入侵任何sglang的代码，最开始比较benchmark的时候想着让Agent自己去调用sglang的算子就行，但是后来发现在一些low latency的shape对应的kernel上，SGLang自己的overhead（比如 register custom op）完全干扰了Opus 4.8的优化，最后结果出现偏离。通过把Benchmark框架和算子导出的方式（tvm-ffi）统一成一个固定的模板，只允许从SGLang main分支拷贝kernel的代码，接口导出和调用都在[BBuf/KDA-Pilot](https://github.com/BBuf/KDA-Pilot)里面完全统一避免这种Reward Hacking。

## Humanize RLCR Loop我自己踩到的2个坑

Bash 版本会影响 Humanize hook 的稳定性：macOS 自带 `/bin/bash` 还是 3.2，在 `set -u` 下展开空数组会触发 `unbound variable`，导致某些 Stop hook 渲染静态模板失败。BBuf/humanize #1（https://github.com/BBuf/humanize/pull/1）把 `env "${env_vars[@]}" awk ...` 改成兼容 Bash 3.2 的空数组写法；BBuf/KDA-Pilot #28（https://github.com/BBuf/KDA-Pilot/pull/28）则在 launcher 层优先选择 Homebrew Bash，并拒绝 `/bin/bash` 3.2。（比较新的mac os系统不会有问题）

Codex hook feature 名称变化也会让 Humanize RLCR 卡住：旧逻辑只禁用 `codex_hooks`，但新版 Codex 暴露的是 `hooks` 和 `plugin_hooks`，nested `codex exec/review` 可能重新触发外层 Stop Hook，形成递归。PolyArch/humanize #194（https://github.com/PolyArch/humanize/pull/194）改为先探测当前 Codex 支持哪些 `--disable` feature，再只传支持项并缓存结果；对应 Humanize plugin version 同步到 `1.17.0`，遇到长时间停在 `running stop hook` 时优先检查升级。

## vs AKO

![](https://raw.githubusercontent.com/BBuf/how-to-optim-algorithm-in-cuda/main/large-language-model/sglang/assets/kda-pilot-sglang-diffusion-kernel/ako4x-mode2-round1-summary.png)

开启AKO的Mode 2 — Closed-loop (default)，后续每个Round都需要我自己来点。

![](https://raw.githubusercontent.com/BBuf/how-to-optim-algorithm-in-cuda/main/large-language-model/sglang/assets/kda-pilot-sglang-diffusion-kernel/ako4x-vs-kda-b200-groupnorm-silu.png)

考虑成本，收敛速度和可交互性来看，KDA目前更加适合优化SGLang Kernel。

AKO支持外挂skill，我们也可以把kernel-wiki和ncu-report-skill挂上去，但是交互体验无法改善。

## 现有的开源Kernel Agent无法解决的问题

https://github.com/BBuf/KDA-Pilot/tree/main/kernels/b200_fa4_mha__bf16_head128_total32768 无法使用KDA写cuda复现出AVO那种迭代7天超越cutedsl fa4的结果，还有一段路要走
