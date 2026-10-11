# GB300 单卡：SGLang eager/BCG 与 vLLM-Omni regional compile

测试日期：2026-10-11。这里保存实际使用的 SDK harness、采样配置、逐请求记录、compile 证据和原始生成图片。所有脚本保持测试时的字节内容；复现前先阅读下面的版本、路径和计时范围。

## 结果

每格为两次独立进程的中位数（秒）。降低比例为 `(Omni - SGLang) / Omni`，不是吞吐提升比例。

| 模型 / 固定配置 | SGLang 模式 | SGLang 两轮 | Omni compile 两轮 | 延迟降低 |
|---|---|---:|---:|---:|
| FLUX.1-dev，1024²，50 steps | eager + 已有 BCG | 2.631793 / 2.631022 | 3.197494 / 3.195274 | 17.69% / 17.66% |
| Qwen-Image-2.1，1024²，50 steps | eager | 2.358374 / 2.350759 | 2.562421 / 2.565007 | 7.96% / 8.35% |
| Z-Image-Turbo，1024²，9 steps，sigma 对齐 | eager | 0.443564 / 0.443877 | 0.520243 / 0.523430 | 14.74% / 15.20% |
| Cosmos3-Nano T2I，1024²，35 steps | eager + MLP BCG，[PR #43644](https://github.com/sgl-project/sglang/pull/43644) | 1.383255 / 1.343985 | 1.549142 / 1.545842 | 10.71% / 13.06% |
| LTX-2，1536×1024×121，40+3 steps，含音频 | eager + BCG，[PR #43667](https://github.com/sgl-project/sglang/pull/43667) | 26.161073 / 26.185223 | 26.220578 / 26.242101 | 0.23% / 0.22% |

LTX-2 与 Omni 基本持平，约 0.22% 的差异在正常测量波动范围内，不能当成稳定领先。相对原版 SGLang，它降低延迟 12.75% / 12.32%；Cosmos3 PR 相对原版降低 7.32% / 9.00%。FLUX 使用已有 BCG，相对自己的 eager 降低 20.64% / 19.90%。

## 测量条件与解释

- 同一张 GB300，单请求串行，TP=1，BF16，权重常驻；CPU 8–15，NUMA 0。另一台机器应先确认其 GPU/NUMA 拓扑，不能盲用这些 CPU 编号。
- **SGLang 没有 torch.compile**；Omni 设置 `enforce_eager=False, diffusion_compile_granularity='regional'`。实际编译模块数：FLUX 57、Qwen 32、Z-Image 34、Cosmos3 36、LTX-2 48；各自 Dynamo `unique_graphs` 为 3/2/2/1/3。
- 每进程有 3 个 prompt，seed=42/43/44；参数 `--warmup 3 --repeats 5` 意味着每个 prompt 3 次预热、5 次测量，即 **共 9 次预热、15 次测量**。不是 3/5 个总请求。
- 计时是 native SDK 返回完整 CPU 图片或视频+音频，包含 text encoder、denoise、VAE 和 CPU 输出；不计加载/编译/预热、文件保存和 hash。不是仅 denoise latency，也不是 HTTP server benchmark。关闭跨请求 conditioning cache。
- FLUX/Qwen/Z/LTX 显式检查 BF16 VAE、关闭 autocast 和 tiling。Cosmos3 使用其原生 SDK 默认 decode 路径；具体调用以 harness 为准。
- SGLang 使用 PyTorch 2.14.1+cu130，Omni 使用 2.13.0+cu130，其他依赖也不同；跨框架结果是这两套固定环境的结果，不能把全部差异归因于框架本身。环境快照在 `environment/`，其中本地 editable 路径只是出处，不能原样拿 freeze 文件当远程安装清单。
- FLUX 两边原生 sigma 计算有最多 1 个 FP32 ULP 差异，Qwen 有最多 2 个 ULP，均独立复算过公式。保留原生计算，没有通过改框架代码强制相等。Qwen 的 Omni 不同进程有小幅 RGBA 差异（SSIM 0.9934–0.9975）；同进程重复一致。
- **Z-Image 两边默认 sigma 网格不同，默认配置的旧数据不用于这张表。** SGLang 保留原生 `linspace(1,1/9,9)`，Omni 用公开 `sigmas=` 参数传同一网格；实际 shifted sigmas 完全一致。
- LTX 使用相同 40+3 步 sigma 数组和 CFG/STG/LoRA 参数。辅助 adapter 只给现有 request-extra 选择 guider 和 LoRA 字段，并观察执行配置。两框架原生 RNG 的 packed/unpacked 布局不同，不声称跨框架像素相等。正确性看 SGLang candidate 相对自身 baseline 的完整视频和音频 hash，一共 96 个输出完全一致。
- Profile 不混入性能计时；本表没有请求级 profiling。

## 固定源码和模型

| 项目 | revision |
|---|---|
| SGLang，FLUX/Qwen/Z baseline | `dab108b9f386b5b991ee070c7e4784f35df03f93` |
| SGLang，Cosmos3/LTX stock baseline | `33d2ef17566685f596b82512e8264456716527f7` |
| Cosmos3 candidate / PR #43644 | `1910100a9effcc01036c93b2935a4ce68d3f34ab` |
| LTX candidate / PR #43667 | `3ae4bfefcc461690327cfc2336b78e71938ae0f0` |
| vLLM-Omni，全部模型 | `194b936637f4e7356cc998c2796c67d14d52d854` |
| `black-forest-labs/FLUX.1-dev` | `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21` |
| `Qwen/Qwen-Image-2.1` | `d26bb61231c349cf6b7896fa83353113880e1ba3` |
| `Tongyi-MAI/Z-Image-Turbo` | `f332072aa78be7aecdf3ee76d5c247082da564a6` |
| `nvidia/Cosmos3-Nano` | `e59a53c25979a090fa8706c9acc0c254a6e89b92` |
| `Lightricks/LTX-2` | `dfcc2108383fe1aaa0584bdf55d368a4bdadd90c` |

## 环境准备

实际 SGLang 基础镜像是 `lmsysorg/sglang:dev` 的固定 manifest `sha256:8346c975515af7ec6b2692faadd5332f789001f78f958335175d4d34e170537c`，GB300 使用 ARM64 版本。进入该镜像后执行以下命令。宿主机需已有可用的 GB300/CUDA 驱动，容器内需有 `numactl`、`git`、`uv`。

脚本保留测试时的绝对根目录 `/scratch/bbuf-diffusion-20261011`。**在新的专用目录复现**，不要覆盖现有 checkout 或结果；每个 run label 也必须唯一。下面 `ARTIFACT_DIR` 指向下载本目录后的绝对路径。

```bash
export ARTIFACT_DIR=/absolute/path/to/artifacts/gb300-diffusion-20261011
export TASK_ROOT=/scratch/bbuf-diffusion-20261011
mkdir -p "$TASK_ROOT"/{source,envs,logs,results,models,compile-cache}
cp "$ARTIFACT_DIR"/scripts/* "$TASK_ROOT/"

git clone https://github.com/sgl-project/sglang.git "$TASK_ROOT/source/sglang"
git -C "$TASK_ROOT/source/sglang" fetch origin pull/43644/head:refs/bench/cosmos3 pull/43667/head:refs/bench/ltx2
git -C "$TASK_ROOT/source/sglang" checkout --detach dab108b9f386b5b991ee070c7e4784f35df03f93

git clone https://github.com/vllm-project/vllm-omni.git "$TASK_ROOT/source/vllm-omni"
git -C "$TASK_ROOT/source/vllm-omni" checkout --detach 194b936637f4e7356cc998c2796c67d14d52d854
uv venv --python /opt/sglang/bin/python3 "$TASK_ROOT/envs/vllm"
uv pip install --python "$TASK_ROOT/envs/vllm/bin/python" 'vllm==0.31.0'
uv pip install --python "$TASK_ROOT/envs/vllm/bin/python" -e "$TASK_ROOT/source/vllm-omni"
# 固定此次 Omni 的主要上层依赖，完整版本记录见 environment/。
uv pip install --python "$TASK_ROOT/envs/vllm/bin/python" \
  'diffusers==0.40.0' 'transformers==5.14.1' 'flashinfer-python==0.7.0.post1'
/opt/sglang/bin/python3 -c 'import torch; print(torch.__version__)'
"$TASK_ROOT/envs/vllm/bin/python" -c 'import torch,vllm; print(torch.__version__,vllm.__version__)'
# 预期分别为 2.14.1+cu130 和 2.13.0+cu130 / 0.31.0。
```

每个 `run_*.sh` 会：检查没有其他 GPU 进程；设置 `CUDA_VISIBLE_DEVICES=0`、`OMP_NUM_THREADS=8`、`HF_HUB_OFFLINE=1`、`FLASHINFER_DISABLE_VERSION_CHECK=1`；通过 `numactl --physcpubind=8-15 --membind=0` 串行启动对应 SDK。`sglang` 参数选择 `/opt/sglang/bin/python3` 和源码 `PYTHONPATH`，`vllm` 参数选择独立 Omni venv。

这里发布的是 SDK 测量命令；不能拿只运行 CLI 单张图片的耗时与表里的稳态请求耗时直接比较。

## 逐模型下载与完整测试命令

一次只下载/运行一个模型。完成该模型全部对照并保存结果后再清理其专用模型缓存，避免同时占用超过 200 GB 磁盘。

### FLUX.1-dev

FLUX 使用 ModelScope 镜像，24 个必要文件均按 manifest 校验，匹配固定 HF revision 的 LFS SHA256 或 Git blob hash。模型使用条款仍见下载的 LICENSE；这里没有发布权重。

```bash
/opt/sglang/bin/python3 "$TASK_ROOT/download_flux_modelscope.py"
git -C "$TASK_ROOT/source/sglang" checkout --detach dab108b9f386b5b991ee070c7e4784f35df03f93
bash "$TASK_ROOT/run_flux1_sdk_uncached_matched.sh" sglang flux1-clean-main-final-eager-a --warmup 3 --repeats 5
bash "$TASK_ROOT/run_flux1_sdk_uncached_matched.sh" sglang flux1-clean-main-final-bcg-b --warmup 3 --repeats 5 --bcg
bash "$TASK_ROOT/run_flux1_sdk_uncached_matched.sh" vllm flux1-clean-main-final-omni-c --warmup 3 --repeats 5
bash "$TASK_ROOT/run_flux1_sdk_uncached_matched.sh" vllm flux1-clean-main-final-omni-d --warmup 3 --repeats 5
bash "$TASK_ROOT/run_flux1_sdk_uncached_matched.sh" sglang flux1-clean-main-final-bcg-e --warmup 3 --repeats 5 --bcg
bash "$TASK_ROOT/run_flux1_sdk_uncached_matched.sh" sglang flux1-clean-main-final-eager-f --warmup 3 --repeats 5
```

关键参数：1024×1024、50 steps、embedded guidance=3.5、true CFG=1（关闭双分支 CFG）、T5 max length=512、CUDA generator。BCG 对应 SGLang `enable_breakable_cuda_graph=True`；Omni 对应 regional compile。

### Qwen-Image-2.1

```bash
/opt/sglang/bin/python3 "$TASK_ROOT/download_model.py" --key qwen21 \
  --model Qwen/Qwen-Image-2.1 --revision d26bb61231c349cf6b7896fa83353113880e1ba3
git -C "$TASK_ROOT/source/sglang" checkout --detach dab108b9f386b5b991ee070c7e4784f35df03f93
bash "$TASK_ROOT/run_qwen21_sdk_uncached_matched.sh" sglang qwen21-clean-main-uncached-eager-a --warmup 3 --repeats 5
bash "$TASK_ROOT/run_qwen21_sdk_uncached_matched.sh" vllm qwen21-clean-main-uncached-omni-b --warmup 3 --repeats 5
bash "$TASK_ROOT/run_qwen21_sdk_uncached_matched.sh" vllm qwen21-clean-main-uncached-omni-c --warmup 3 --repeats 5
bash "$TASK_ROOT/run_qwen21_sdk_uncached_matched.sh" sglang qwen21-clean-main-uncached-eager-d --warmup 3 --repeats 5
```

关键参数：1024×1024、50 实际 steps、true CFG=1、CPU generator；保留 RGBA 输出和 alpha。SGLang 为 eager，无需新增优化。

### Z-Image-Turbo

```bash
/opt/sglang/bin/python3 "$TASK_ROOT/download_model.py" --key zimage \
  --model Tongyi-MAI/Z-Image-Turbo --revision f332072aa78be7aecdf3ee76d5c247082da564a6
git -C "$TASK_ROOT/source/sglang" checkout --detach dab108b9f386b5b991ee070c7e4784f35df03f93
bash "$TASK_ROOT/run_zimage_sdk_uncached_matched.sh" sglang zimage-matched-sigma-v2-eager-a --warmup 3 --repeats 5
bash "$TASK_ROOT/run_zimage_sdk_uncached_matched.sh" vllm zimage-matched-sigma-v2-omni-b --warmup 3 --repeats 5
bash "$TASK_ROOT/run_zimage_sdk_uncached_matched.sh" vllm zimage-matched-sigma-v2-omni-c --warmup 3 --repeats 5
bash "$TASK_ROOT/run_zimage_sdk_uncached_matched.sh" sglang zimage-matched-sigma-v2-eager-d --warmup 3 --repeats 5
```

关键参数：1024×1024、9 steps、guidance=0、CUDA generator。脚本已将 `np.linspace(1.0, 1.0/9, 9).tolist()` 传给 Omni 的 `sigmas` 参数，SGLang 不覆盖原生 sigma；最终实际 shifted sigma 数组包括 terminal 0，见双方 `denoise-config.json`。

### Cosmos3-Nano T2I

```bash
/opt/sglang/bin/python3 "$TASK_ROOT/download_model.py" --key cosmos3 \
  --model nvidia/Cosmos3-Nano --revision e59a53c25979a090fa8706c9acc0c254a6e89b92
git -C "$TASK_ROOT/source/sglang" checkout --detach 33d2ef17566685f596b82512e8264456716527f7
bash "$TASK_ROOT/run_cosmos3_fixed.sh" sglang cosmos3-mlp-final-stock-a --warmup 3 --repeats 5

git -C "$TASK_ROOT/source/sglang" checkout --detach 1910100a9effcc01036c93b2935a4ce68d3f34ab
bash "$TASK_ROOT/run_cosmos3_fixed.sh" sglang cosmos3-mlp-final-bcg-b --warmup 3 --repeats 5 --bcg
bash "$TASK_ROOT/run_cosmos3_fixed.sh" vllm cosmos3-mlp-final-omni-c --warmup 3 --repeats 5
bash "$TASK_ROOT/run_cosmos3_fixed.sh" vllm cosmos3-mlp-final-omni-d --warmup 3 --repeats 5
bash "$TASK_ROOT/run_cosmos3_fixed.sh" sglang cosmos3-mlp-final-bcg-e --warmup 3 --repeats 5 --bcg

git -C "$TASK_ROOT/source/sglang" checkout --detach 33d2ef17566685f596b82512e8264456716527f7
bash "$TASK_ROOT/run_cosmos3_fixed.sh" sglang cosmos3-mlp-final-stock-f --warmup 3 --repeats 5
```

关键参数：1024×1024、num_frames=1、35 steps、CFG=4、flow_shift=3、guidance_interval=[400,1000]、text max length=128、CUDA generator。两边均不加 resolution/duration/system prompt 模板；用同一固定 benign prompt 集，guardrails 均关闭，比较生成 pipeline 本身。

### LTX-2 two-stage

```bash
/opt/sglang/bin/python3 "$TASK_ROOT/download_model.py" --key ltx2 \
  --model Lightricks/LTX-2 --revision dfcc2108383fe1aaa0584bdf55d368a4bdadd90c \
  --allow-pattern model_index.json --allow-pattern LICENSE --allow-pattern README.md \
  --allow-pattern 'audio_vae/*' --allow-pattern 'connectors/*' \
  --allow-pattern 'latent_upsampler/*' --allow-pattern 'scheduler/*' \
  --allow-pattern 'tokenizer/*' --allow-pattern 'transformer/*' \
  --allow-pattern 'vae/*' --allow-pattern 'vocoder/*' \
  --allow-pattern ltx-2-19b-distilled-lora-384.safetensors \
  --allow-pattern text_encoder/config.json --allow-pattern text_encoder/generation_config.json \
  --allow-pattern 'text_encoder/model*.safetensors' --allow-pattern text_encoder/model.safetensors.index.json

git -C "$TASK_ROOT/source/sglang" checkout --detach 33d2ef17566685f596b82512e8264456716527f7
bash "$TASK_ROOT/run_ltx2_sdk_matched_v2.sh" sglang ltx2-final-v4-stock-a --warmup 3 --repeats 5

git -C "$TASK_ROOT/source/sglang" checkout --detach 3ae4bfefcc461690327cfc2336b78e71938ae0f0
bash "$TASK_ROOT/run_ltx2_sdk_matched_v2.sh" sglang ltx2-final-v4-bcg-b --warmup 3 --repeats 5 --bcg
bash "$TASK_ROOT/run_ltx2_sdk_matched_v2.sh" vllm ltx2-final-v4-omni-c --warmup 3 --repeats 5 \
  --sglang-reference "$TASK_ROOT/results/ltx2-final-v4-stock-a"
bash "$TASK_ROOT/run_ltx2_sdk_matched_v2.sh" vllm ltx2-final-v4-omni-d --warmup 3 --repeats 5 \
  --sglang-reference "$TASK_ROOT/results/ltx2-final-v4-stock-a"
bash "$TASK_ROOT/run_ltx2_sdk_matched_v2.sh" sglang ltx2-final-v4-bcg-e --warmup 3 --repeats 5 --bcg

git -C "$TASK_ROOT/source/sglang" checkout --detach 33d2ef17566685f596b82512e8264456716527f7
bash "$TASK_ROOT/run_ltx2_sdk_matched_v2.sh" sglang ltx2-final-v4-stock-f --warmup 3 --repeats 5
```

关键参数：最终 1536×1024×121、24fps；stage 1 为 768×512×121、40 steps；stage 2 为 3 steps；CPU generator、text max length=1024。video/audio CFG=3/7，STG=1，modality guidance=3，rescale=0.7，STG block=29，LoRA strength=0/1。负面 prompt 在 `scripts/ltx2-negative-prompt.txt`。`--sglang-reference` 读取 stock run 的真实两个 sigma 数组，作为 Omni 公开参数传入。

## 正确性、compile 证据和图片

`results/<model>/<cell>/records.jsonl` 保留逐请求时长、prompt/seed 和输出 hash；`summary.json` 为该进程统计；Omni 的 `compile-activation.json` 和 `dynamo-counters.json` 用于确认实际编译。FLUX/Qwen/Z 的 decode/denoise 配置和 LTX 的 stage schedule 也保存在相同目录。

本目录图片是实际 benchmark 输出，没有重新生成或编辑。`cosmos3-*-prompt{0,1,2}.png` 是三组 prompt，`ltx2-*-prompt0-frame{0,60,120}.png` 为 seed=42 的狐狸视频采样帧。SGLang stock/candidate 像素一致；Omni 图片用于直观参考，不代表跨框架像素完全相同。图片可直接嵌入 PR 描述。

`SHA256SUMS` 可核对这里发布的脚本、记录和 PNG。权重不包含在此目录，任务结束已清理。此次发布补充文档和证据，没有重新跑模型性能测试。
