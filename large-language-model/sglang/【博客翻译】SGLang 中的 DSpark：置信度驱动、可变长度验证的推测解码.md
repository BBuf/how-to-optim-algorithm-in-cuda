> 本文翻译自 LMSYS Blog，原文 slug：2026-07-06-dspark-sglang。
>
> GitHub 原文在 `lm-sys/lm-sys.github.io` 仓库，路径为 `blog/2026-07-06-dspark-sglang.md`。
>
> 原标题：DSpark in SGLang: Speculative Decoding with Confidence-Driven, Variable-Length Verification
>
> 原作者：SGLang Team

# SGLang 中的 DSpark：置信度驱动、可变长度验证的推测解码

推测解码的基本思路，是多花一点计算，换更少的 decode 步数。但负载变大后，这笔账不一定划算：batch size 是 `B`、每次推测 `K` 个 token 时，目标模型每一步都要验证 `B * K` 个 token。过了某个点，验证本身反而比省下来的 decode 步更贵。

DSpark 同时改了两件事。第一，它用半自回归的 block drafter，一次 draft forward 生成一整块 token，让接受率不要掉得太快。第二，它根据草稿模型自己的置信度，给每个请求分配不同的 verify length，不再验证那些很可能不会被接受的尾部 token。算法和主要收益来自 DSpark 论文。

SGLang 现在已经支持在 dense 和 sparse 模型上使用 DSpark，例如 Qwen3 和 DeepSeek-V4。本文讲的是这次集成，相关 PR 是 `sgl-project/sglang#30261`。我们在开源 serving engine 上复现了论文里最关键的曲线形态：单用户加速，以及负载升高时 verify budget 会自动收缩。

更麻烦的是工程落地。为了让调度策略真的变成 wall-clock time 收益，SGLang 做了几件事：在 ragged、per-request verify 上捕获 full CUDA graph，让被裁剪的 batch 回放真正更小的 graph；用 overlap-aware 的推测路径把 scheduler 藏到 forward 后面；用 cost-table profiler 在线估计每个请求的 verify budget；还要补上观测能力，看见被裁剪遮住的 acceptance ceiling。硬件、引擎和流量都和论文不一样，所以这里复现的是机制和曲线，不追求逐位一致的数字。下文所有「更快」也都是和我们自己的控制组比，控制组只改 speculation config。

## 相对 MTP 和非推测 baseline 的加速

![图 1：H200 DP4 上，总吞吐和单用户 decode 速度的对比。每条曲线对应一个方案：非推测 baseline、MTP 和 DSpark。越靠右上越好；每个 marker 是一个 batch size，取三轮平均值。](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/a9a76d7d-e5cb-44f4-bebe-3d77f08c2ae2.png)

图 1：总吞吐（y 轴）和单用户 decode 速度（x 轴）。每条曲线从 batch 1 扫到 256，对应一个方案。越靠右上越好。

图 1 里，DSpark 在整个 concurrency sweep 上给出了最好的 throughput/latency trade-off，明显高于 MTP 和非推测 baseline。三组实验都在 H200 上跑 DeepSeek-V4-Flash，使用四路 rank 的 DP-attention；除了 speculation config，其余设置相同。三组分别是非推测 baseline、MTP，以及 DSpark。其中 MTP 是 EAGLE 风格 baseline，在每个 batch size 下从 1-1-2 和 3-1-4 两种配置里取最好结果。

## 在 SGLang 中接入 DSpark

从论文迁移过来的 DSpark 算法，主要包括草稿侧的三个模块：

- Block drafter：分 dense 路线（例如 Qwen3）和 sparse 路线（例如 DeepSeek-V4）。一次 forward 生成一个 `gamma` token block，并用轻量顺序头（Markov 或 RNN）让每一步依赖上一个 token，所以这个 block 是半自回归的。
- Confidence head：给每个 drafted token 打分，估计它能通过验证的概率；整块 token 的概率乘积就是 block 的 survival probability。
- Sequential Temperature Scaling（STS）：校准这些分数，让 survival probability 更接近 scheduler 做预算时需要的真实接受率。

SGLang 在 serving 侧补了下面这些能力：

- Confidence scheduler：每一步把 per-block survival 转成每个请求的验证预算。
- Per-request ragged verify：同一个 batch 里，每个请求可以有不同的 verify length，模式包括 `static`、`compact` 和 `cap-accept`。
- Full CUDA graph：在 ragged、variable-length verify 上捕获 CUDA graph。
- Observability：暴露裁剪下的 acceptance ceiling 和其它指标。
- Additive SPS cost table：离线 profile 得到 step-time 模型，在线给 scheduler 读取。
- Data-parallel attention：和其它并行维度一起工作。
- Zero-overhead scheduling：接入 SGLang 的 overlap scheduler，几乎不需要 DSpark 专用分支。
- Performance optimizations：包括融合 Triton kernel 和 sharded block-drafter matmul。

### Verify modes

本文后面会反复提到三种 verify mode。`static` 每一步都验证完整 drafted block，是 baseline。`compact` 只验证 scheduler 为每个请求选出的窗口，是生产路径。`cap-accept` 会验证完整 block，但只提交到窗口为止；它的输出和 `compact` 相同，同时能看到「如果完整验证，这一步本来会接受多少 token」。我们就是靠它来衡量裁剪下的 ceiling。

### Full CUDA graph 下的 ragged verify

Per-request window 很难塞进固定形状的 CUDA graph。一个 batch 里，可能有的请求只验证 2 个 token，有的请求要验证 6 个 token；它们没有统一的 query length。如果把所有请求都 pad 到完整 block width，又等于把刚裁掉的部分补了回来。

SGLang 的做法是保留 ragged batch，并用总 token 数作为 graph key：先把不同长度的请求 front-pack 到一个 compact buffer，再向上取整到最近的 captured tier。预算被裁剪后，packed total 会落到更小的 tier，DSpark 回放的 graph 也真的更便宜。这里减少的是 attention 和 MLP 的行数，不是跑一个 masked full-width forward。在 DP attention 下，各 rank 共享同一个 tier，也就是取所有 rank 里最大的需求，然后一起降档。

Packed buffer 是 `cu_seqlens` 风格的 varlen 输入，所以 compact verify 可以复用后端已有的 attention kernel。在 DeepSeek-V4 上，它直接走模型自己的 sparse-MLA 路径（`flash_mla`），不需要新 kernel。每个支持的 backend 只需要在 graph replay 时，从 packed layout 重建自己的 varlen metadata。

![图 2：把 per-request-variable verify length 的 batch 放进 captured CUDA graph。固定形状 graph 会把每个请求 pad 到完整 block width（N x W）；ragged 路径会 front-pack 已调度 token，并只对总 token 数向上取整到最近 tier，因此用同样 accepted tokens 计算更少 padded cells。](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-1/2bcfe914-f04c-406e-910d-7678b296e573.png)

图 2：把 per-request-variable verify length 的 batch 放进 captured CUDA graph。固定形状 graph 会把每个请求 pad 到完整 block width（N x W）；ragged 路径会 front-pack 已调度 token，只对总 token 数向上取整到最近的 captured tier，因此用同样的 accepted tokens 计算更少 padded cells。

### Observability

裁剪会遮住 ceiling。compact mode 只验证 block 前几个位置，也就是 scheduler 选出的窗口，所以我们看不到这一步如果做完整 block verify 到底会接受多少 token。没有这个信息，就很难判断一次 trim 是合理省计算，还是已经砍掉了本该接受的 token。

`cap-accept` run 可以把这部分信息找回来：它验证完整 block，但只提交到窗口为止。因此它提交的内容和 compact 完全一样，同时暴露 ceiling。SGLang 也会暴露 per-request confidence 和校准指标（例如 ECE），方便事后分析。

### 估计裁剪下的 ceiling

生产环境里不一定愿意额外跑一条 companion run。为此，DSpark 还提供了 block-accept estimator，直接在 compact run 内估计被裁剪掉的 ceiling。它会利用未来 step 里的目标 token 和 logprobs，为反事实 tail 计算估计区间；这里假设被裁剪轨迹和未裁剪轨迹中的 anchor tokens 具有相似属性。

## 动态调度和固定调度的初步对比

当前 confidence scheduler 还是第一版朴素实现。可以把它看作端到端机制验证，而不是调到极致的结果。这里在两个接受率不同的示例 workload 上，对比 `compact` 和 `no-trim`。`compact` 使用 per-step SPS-argmax budget；`no-trim` 则是通过同一条 ragged path 执行的 `static` full-block schedule。

![图 3 左：GSM8K 上，compact 动态裁剪和 no-trim full block 的对比。](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-1/18383fed-bfe5-40a1-978f-39a713293e66.png)

![图 3 右：Arena-hard 示例上，compact 动态裁剪和 no-trim full block 的对比。](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-2/92525f7e-2b95-42a2-9297-509e8ad270e7.png)

图 3：`compact`（dynamic trim）和 `no-trim`（full block）的对比，batch 从 1 扫到 256，DP4，两个示例的接受率不同。越靠右上越好。

Dynamic budget 的收益主要出现在大 batch。batch size 为 1 时，目标模型多 verify 几个 token 不会明显变慢，裁剪省不了太多，两组基本打平。concurrency 继续上去后，吞吐开始进入平台期，裁剪能缩短 step，`compact` 就拉开了差距。低接受率示例的优势更大，也更早出现；接受率低意味着有更多 tail 可以裁剪，这和 cost model 的预测一致。

每个 panel 内都是干净的 `compact` vs. `no-trim` A/B 测试，panel 内设置相同。但两个示例不是严格的单变量对照：除了接受率不同，它们的 prompt formatting 和每个 arm 的 round count 也有细微差异。所以更适合看趋势，不适合直接拿两个 panel 的绝对数字互相比。

这些 budget 的质量取决于背后的 cost table。当前 SPS 和校准拟合都还是第一版近似，还不一定完整刻画了 step cost 随 context length 的变化。因此 scheduler 现在落到的 operating point 还有优化空间；这里展示的是机制，不是最终数字。

## 混合流量下的 per-request differentiation

同质化 sweep 会把 confidence scheduling 的重点藏起来。同一个 batch 里的两个请求，如果一个明显更可预测，它们就不该拿到同样的 verify window。混合流量才真正考验这件事。

![图 4：按 workload 划分的预算（左）和 per-step verify-length 分布（右）。](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-2/62b46db4-87ad-4f6d-a33d-74d6d483bf74.png)

图 4：按 workload 划分的预算（左），以及 per-step verify-length 分布（右）。

这里混合了三类接受难度不同的 workload：gsm8k（高）、arena-hard（中）和 poetry（低）。难度越高，窗口越短，三者分别是 5.24、3.78、2.91 token；但相对于 ceiling（未裁剪时 block 本来会接受多少 token）的利用率仍然很高，在 0.88 到 0.97 之间。换句话说，scheduler 是在给每个请求单独定尺寸，而不是给整个 batch 套一个平均值。右图按 step 展示了这个差异：大约 55% 的 gsm8k step 会填满长度为 6 的完整窗口，而约 80% 的 poetry step 只用 3 个或更少 token。

## 性能优化和 zero-overhead scheduling（ZOS）

要把调度策略变成真实速度，工程上要做两件事：每个 step 本身要便宜，scheduler 也不能挡在 forward 前面。两者叠加后，在 DeepSeek-V4-Pro、TP=8、B300、batch size 1 上，accept length 约为 5 时可以达到 383.7 tok/s。

SGLang 把一批小 op 改写成融合 Triton kernel，例如 compact scatter、SWA page-index、verify-length top-k schedule、ragged-window packing。Block drafter 的 sampling path 也并进了 fused kernels，矩阵乘法则做了 sharding。在一个 profile 里，target verify 之外的部分少了 1.7 ms，而 verify 本身是 7.3 ms。

DSpark 可以直接接入 SGLang 的 zero-overhead（overlap）scheduler，几乎不需要专门分支，只额外加入论文里的 two-step-back confidence relay。这里没有多少 DSpark 专用管线。SGLang 的 spec-v2 runtime 已经能在独立 stream 上，把下一步 scheduling 和当前 forward 重叠起来；DSpark 只是作为一等 worker 接入：forward 输出以 async future 返回，跨 iteration 的顺序依赖通过 runtime 的 device-side barrier 处理，on-device page table 避免 per-step host sync。Confidence relay 也走同一个 channel，读取两步之前的信息。这样 decode loop 就没有 per-step bubble，比关闭 scheduler 时紧凑约 1.5 倍。

![图 5：batch size 1 的 decode trace。上图关闭 overlap scheduler，会在 run_batch iteration 之间以及 draft-generate 和 target-verify 阶段之间出现 bubble；下图打开后，这些阶段背靠背执行。](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/c8efb872-bb6e-4adc-b9f7-97386da3e818.png)

图 5：batch size 1 的 decode。上图是关闭 overlap scheduler，下图是打开。打开后，`run_batch` iteration 之间，以及一个 step 内 block-draft-generate 和 target-verify 阶段之间都没有 bubble。

## Profile cost table

![图 6：Additive SPS cost-table 拟合：raw step time vs. fit（a）和 throughput（b），以及 SPS 预测和实测 decode-step time（c），DeepSeek-V4 on H200。](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-1/1d59ebbd-336a-455e-b9a5-3f2d86b0d32a.png)

图 6：Additive cost model：raw vs. fit（a）、throughput（b），以及 predicted vs. measured step time（c）。

Scheduler 用一个 additive model 来估计 step time `T(bs, K)`，其中 `K` 是 batch 里的额外 verify tokens：

```text
T(bs, K) = bias + alpha(bs) + theta(M), M = bs + K
```

这里 `alpha(bs)` 是随请求数扩展的底座成本，包括 draft pass 和一部分 attention，不会因为裁剪改变；`theta(M)` 是目标模型的 verify-token 成本，也是裁剪唯一能省下来的项。Scheduler 的 argmax 会在预期 accepted tokens 和真实 marginal cost 之间权衡，所以只有 `theta` 足够大时，trim headroom 才明显。图 6(c) 用 live server 验证了这个模型的预测。

## 下一步

DSpark 已经在 SGLang 中可用；roadmap 在 `sgl-project/sglang#30344` 里跟踪。后续工作包括：

- Cost model 和 scheduling：更强、更在线化的 cost model，以及继续改 dynamic scheduler。
- Model coverage：支持更多 dense 和 sparse 模型。
- Parallelism：覆盖更多并行模式和 serving topology。
- Observability：把 block-accept estimator、跨 checkpoint 的 confidence calibration 等指标做成生产可用状态。
- Robustness：强化 full-CUDA-graph 路径，扩大 stress 和 regression testing。

感谢 DSpark 作者和 DeepSeek 提供算法与模型。

## 附录：复现实验

下面所有命令都在预构建镜像中运行：

```bash
docker pull lmsysorg/sglang:dev-dspark
```

也可以从 `sgl-project/sglang#30261` 源码构建，固定到 commit `692c5f7d532f129424b57961c262bbd253b411dc`。

图 1、图 3、图 6：frontier server（DeepSeek-V4-Flash, H200, DP4）。启动 DSpark arm：

```bash
SGLANG_ENABLE_METRICS_DEVICE_TIMER=1 \
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V4-Flash-DSpark \
  --speculative-algorithm DSPARK \
  --tp 4 --dp-size 4 --enable-dp-attention --enable-dp-lm-head \
  --moe-a2a-backend none --moe-runner-backend flashinfer_mxfp4 --disable-flashinfer-autotune \
  --swa-full-tokens-ratio 0.1 --chunked-prefill-size 1024 \
  --mem-fraction-static 0.8 --cuda-graph-max-bs 192 --max-running-requests 1024 \
  --disable-radix-cache --trust-remote-code --host 0.0.0.0 --port 30000
```

这里的 `--disable-radix-cache` 是为了避免 benchmark 脚本命中 cache。其它 arm 只改变 speculation config：

- non-spec：去掉 `--speculative-*`，并加载 `--model-path deepseek-ai/DeepSeek-V4-Flash`。
- MTP：使用同一个 target，配合 `--speculative-algorithm EAGLE --speculative-num-steps {1,3} --speculative-eagle-topk 1 --speculative-num-draft-tokens {2,4}`，并在每个 batch size 下从两种配置中取最好结果。
- DSpark compact 或 static：设置 `SGLANG_RAGGED_VERIFY_MODE=compact|static`。
- 带 SPS table 的 compact mode：使用 `--speculative-dspark-sps-table-path sps_table.json`。
- 图 3 的 no-trim arm：使用 `SGLANG_RAGGED_VERIFY_MODE=compact`，但不提供 SPS table，也就是用 ragged path 跑完整窗口。

用固定 prompt 扫不同 batch size 来驱动任意 arm：

```bash
python3 -m sglang.benchmark.one_batch_server \
  --model None --base-url http://127.0.0.1:30000 \
  --batch-size 1 8 16 32 64 96 128 160 192 256 --output-len 1024 --temperature 0.7 \
  --fixed-prompt-file frontier_prompt.txt --fixed-prompt-apply-chat-template --show-report
```

固定 prompt 来自 gist `sglang-bot/71cc966dce295e78cbd0baddc402d151`，文件名是 `frontier_prompt.txt`。它由 16 个 GSM8K 问题拼接而成，保证生成内容是真实题目推理。不同数据集上的推测解码 accept length 会不一样，实际部署时最好用自己的数据再测一遍。

图 6 的 cost table 来自一次 profiling run：启动 `compact` 时设置 `SGLANG_DSPARK_ENABLE_SPS_RECORD=1 SGLANG_SIMULATE_ACC_LEN=1.0`，然后用下面命令拟合 additive model。这个 profiling 会在 input-len 512 下扫 batch x verify-fraction grid。

```bash
python3 -m sglang.benchmark.dspark_sps_profiler all
```

图 4：mixed traffic。使用和图 1 相同的 server，把 `--mem-fraction-static` 设为 0.7，block size 为 6；通过 `SGLANG_RAGGED_VERIFY_MODE` 跑三种模式（`static` / `compact` / `cap-accept`），并驱动一个 gsm8k + arena-hard + poetry 混合请求集，用 non-streaming makespan throughput 计量。

图 5：zero-overhead（DeepSeek-V4-Pro, B300, TP8）。

```bash
SGLANG_RAGGED_VERIFY_MODE=compact SGLANG_DSV4_FP4_EXPERTS=1 SGLANG_TORCH_PROFILER_DIR=./trace \
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V4-Pro-DSpark --speculative-algorithm DSPARK \
  --tp 8 --moe-runner-backend flashinfer_mxfp4 --disable-flashinfer-autotune \
  --mem-fraction-static 0.82 --chunked-prefill-size 4096 --cuda-graph-max-bs 4 \
  --trust-remote-code --host 127.0.0.1 --port 30000
# overlap off: append --disable-overlap-schedule
```

捕获 batch-1 decode trace，然后读取 GPU-only lane：

```bash
python3 -m sglang.benchmark.one_batch_server \
  --model None --base-url http://127.0.0.1:30000 \
  --batch-size 1 --input-len 256 --output-len 256 \
  --profile --profile-activities GPU --profile-steps 20
```
