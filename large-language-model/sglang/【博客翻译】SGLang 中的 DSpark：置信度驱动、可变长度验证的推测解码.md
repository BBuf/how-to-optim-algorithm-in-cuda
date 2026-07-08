> 原标题：DSpark in SGLang: Speculative Decoding with Confidence-Driven, Variable-Length Verification
>
> 原作者：SGLang Team

# SGLang 中的 DSpark：置信度驱动、可变长度验证的推测解码

推测解码用额外计算换更少的 decode 步数。但负载一上来，这笔账就容易变差：batch size 为 `B`、推测 token 数为 `K` 时，目标模型每一步都要验证 `B * K` 个 token；超过某个点后，验证成本会比省下来的 decode 步更贵。

DSpark 从两头解决这个问题：一边使用**半自回归块草稿器**，一次 draft forward 生成一整块 token，让接受率保持较高；另一边根据草稿模型自己的置信度，为每个请求分配**可变验证长度**，不再验证那些工作负载大概率不会接受的 token。算法和收益来自 DSpark 论文。

SGLang 现在已经支持在 dense 和 sparse 模型上使用 DSpark，例如 Qwen3 和 DeepSeek-V4。本文介绍这次集成（[sgl-project/sglang#30261](https://github.com/sgl-project/sglang/pull/30261)）。我们在一个开放的 serving engine 上复现了论文收益的形态：单用户加速，以及负载升高时验证预算会收缩。我们也会说明把这个调度策略真正转化为 wall-clock time 的工程工作：在 ragged、per-request verify 上做 full CUDA graph，让被裁剪的 batch 回放真正更小的 graph，而不是 padded graph；overlap-aware 的推测路径，把 scheduler 藏到 forward 后面；cost-table profiler，让 scheduler 在线决定每个请求的验证预算；以及观测被裁剪隐藏起来的 acceptance ceiling。硬件、引擎和流量都和论文不同，所以我们复现的是机制和曲线，而不是逐位一致的数字；下文所有「更快」都和我们自己的控制组对比，控制组除了 speculation config 之外完全相同。

## 相对 MTP 和非推测 baseline 的加速

![图 1：H200 DP4 上，总吞吐和单用户 decode 速度的对比。每条曲线对应一个方案：非推测 baseline、MTP 和 DSpark。越靠右上越好；每个 marker 是一个 batch size，取三轮平均值。](https://files.mdnice.com/user/59/a9a76d7d-e5cb-44f4-bebe-3d77f08c2ae2.png)

图 1：总吞吐（y 轴）和单用户 decode 速度（x 轴）。每条曲线从 batch 1 扫到 256，对应一个方案。越靠右上越好。

在图 1 的例子里，DSpark 在整个 concurrency sweep 上都给出了最好的 throughput/latency trade-off，明显优于 MTP 和非推测 baseline。三组实验都在 H200 上跑 DeepSeek-V4-Flash，使用四路 rank 的 DP-attention；除 speculation config 外其余设置相同。三组分别是：非推测 baseline、MTP（EAGLE 风格 baseline，在每个 batch size 下从 1-1-2 和 3-1-4 配置中取最好结果）以及 DSpark。

## 在 SGLang 中接入 DSpark

从论文迁移过来的 DSpark 算法，主要由草稿侧的三个部分组成：

- **Block drafter**：分 dense 路线（例如 Qwen3）和 sparse 路线（例如 DeepSeek-V4）。一次 forward 生成一个 `gamma` token block，并用一个轻量顺序头（Markov 或 RNN）让每一步依赖上一个 token，所以这个 block 是半自回归的。
- **Confidence head**：给每个 drafted token 打分，估计它能通过验证的概率；整块 token 的概率乘积就是这个 block 的 survival probability。
- **Sequential Temperature Scaling（STS）**：校准这些分数，让 survival probability 能反映 scheduler 用来做预算的真实接受率。

围绕这三个算法组件，SGLang 增加了 serving 侧支持：

- **Confidence scheduler**：每一步把 per-block survival 转成每个请求的验证预算。
- **Per-request ragged verify**：同一个 batch 里，每个请求可以有不同的 verify length（`static` / `compact` / `cap-accept`）。
- **Full CUDA graph**：在 ragged、variable-length verify 上捕获 CUDA graph。
- **Observability**：暴露裁剪下的 acceptance ceiling 和其它指标。
- **Additive SPS cost table**：离线 profile 得到的 step-time 模型，在线由 scheduler 读取。
- **Data-parallel attention**：与其它并行维度一起支持。
- **Zero-overhead scheduling**：集成进 SGLang 的 overlap scheduler，几乎不需要 DSpark 专用分支。
- **Performance optimizations**：融合 Triton kernel，以及 sharded block-drafter matmul。

### Verify modes

三种 verify mode 是本文后面所有讨论的主轴。`static` 每一步都验证完整 drafted block，是 baseline。`compact` 只验证 scheduler 为每个请求选出的窗口，是生产路径。`cap-accept` 会验证完整 block，但只提交到这个窗口为止：输出和 `compact` 相同，同时暴露「完整验证本来会接受多少 token」，也就是我们衡量裁剪下 ceiling 的方法。

### Full CUDA graph 下的 ragged verify

Per-request window 不适合固定形状的 CUDA graph：如果一个 batch 里某个请求验证 2 个 token、另一个请求验证 6 个 token，它们就没有单一的 query length；如果把所有请求都 pad 到完整 block width，又等于把刚裁剪掉的部分补了回来。

因此我们保留 ragged batch，并用**总 token 数**作为 graph key：先把不同长度的请求 front-pack 到一个 compact buffer，再向上取整到最近的已捕获 tier。预算被裁剪后，packed total 会下降到更小的 tier，DSpark 回放的 graph 也真的更便宜，attention 和 MLP 的行数更少，而不是一个 masked full-width forward。在 DP attention 下，各 rank 共享同一个 tier，也就是取所有 rank 中需要的最大 tier，然后一起降档。

Packed buffer 是 `cu_seqlens` 风格的 varlen 输入，因此 compact verify 可以复用后端已有的 attention kernel。在 DeepSeek-V4 上，它直接走模型自己的 sparse-MLA 路径（`flash_mla`），不需要新 kernel；每个支持的 backend 只需要在 graph replay 时从 packed layout 重建自己的 varlen metadata。

![图 2：把 per-request-variable verify length 的 batch 放进 captured CUDA graph。固定形状 graph 会把每个请求 pad 到完整 block width（N x W）；ragged 路径会 front-pack 已调度 token，并只对总 token 数向上取整到最近 tier，因此用同样 accepted tokens 计算更少 padded cells。](https://files.mdnice.com/user/59/2bcfe914-f04c-406e-910d-7678b296e573.png)

图 2：把 per-request-variable verify length 的 batch 放进 captured CUDA graph。固定形状 graph 会把每个请求 pad 到完整 block width（N x W）；ragged 路径会 front-pack 已调度 token，并只对总 token 数向上取整到最近的 captured tier，因此用同样的 accepted tokens 计算更少 padded cells。

### Observability

裁剪会遮住 ceiling：compact mode 只验证一个 block 的前几个位置，也就是 scheduler 选出的窗口，所以我们看不到这一步如果做完整 block verify 到底会接受多少 token。没有这个信息，就很难判断一次 trim 是合理裁剪，还是已经带来损失。

`cap-accept` run 可以恢复这部分信息：它验证完整 block，但只提交到窗口为止。因此它提交的内容和 compact 完全一样，同时暴露 ceiling。我们还会暴露 per-request confidence 和校准指标（例如 ECE），用于事后分析。

### 估计裁剪下的 ceiling

Block-accept estimator 用在生产运行或其它不想额外跑 companion run 的场景，它可以直接在 compact run 内恢复被裁剪 ceiling 的估计值。实现上，它利用未来 step 中目标 token 及其 logprobs，计算反事实 tail 的估计区间；这里假设被裁剪轨迹和未裁剪轨迹中的 anchor tokens 具有相似属性。

## 动态调度和固定调度的初步对比

当前 confidence scheduler 还是第一个朴素版本，我们也按这个定位来看它：它证明机制可以端到端跑通，但还不是高度调优后的结果。我们在两个接受率不同的示例 workload 上，对比 `compact`（per-step SPS-argmax budget）和 `no-trim`，后者是通过同一条 ragged path 执行的 `static` full-block schedule。

![图 3 左：GSM8K 上，compact 动态裁剪和 no-trim full block 的对比。](https://files.mdnice.com/user/59/18383fed-bfe5-40a1-978f-39a713293e66.png)

![图 3 右：Arena-hard 示例上，compact 动态裁剪和 no-trim full block 的对比。](https://files.mdnice.com/user/59/92525f7e-2b95-42a2-9297-509e8ad270e7.png)

图 3：`compact`（dynamic trim）和 `no-trim`（full block）的对比，batch 从 1 扫到 256，DP4，两个示例的接受率不同。越靠右上越好。

Dynamic budget 的收益主要体现在大 batch。batch size 为 1 时，目标模型 verify 更多 token 并不会明显变慢，所以裁剪省不了太多，两组基本打平。随着 concurrency 提高、吞吐开始进入平台期，裁剪能缩短 step，`compact` 开始领先。低接受率示例的差距更大，而且更早拉开；低接受率意味着有更多 tail 可裁剪，这和 cost model 的预测一致。

每个 panel 内都是干净的 `compact` vs. `no-trim` A/B 测试，panel 内设置相同。但两个示例并不是严格的单变量对照：除了接受率不同，它们在设置上也有细微差异，例如 prompt formatting 和每个 arm 的 round count。因此我们关注跨 panel 的趋势，而不是直接比较两个 panel 的绝对数字。

这些 budget 的质量也取决于背后的 cost table。当前 SPS（以及校准）拟合只是第一版近似，还不一定能完整刻画 step cost 随 context length 的变化。因此 scheduler 现在落到的 operating point 很可能还能改进；这里展示的是机制，而不是调优到极致的数字。

## 混合流量下的 per-request differentiation

同质化 sweep 会掩盖 confidence scheduling 真正想解决的问题。同一个 batch 里的两个请求，如果一个远比另一个可预测，它们就不应该拿到相同的 verify window。混合流量才是这个能力真正有用的地方。

![图 4：按 workload 划分的预算（左）和 per-step verify-length 分布（右）。](https://files.mdnice.com/user/59/62b46db4-87ad-4f6d-a33d-74d6d483bf74.png)

图 4：按 workload 划分的预算（左），以及 per-step verify-length 分布（右）。

举个例子，我们按接受难度混合三类 workload：gsm8k（高）、arena-hard（中）和 poetry（低）。窗口会随着难度收缩，分别是 5.24、3.78、2.91 token；同时，相对于 ceiling（未裁剪时 block 本来会接受多少 token）的利用率仍然很高，在 0.88 到 0.97 之间。也就是说，scheduler 是在给每个请求单独定尺寸，而不是给整个 batch 套一个平均值。右图展示了逐 step 的情况：大约 55% 的 gsm8k step 会填满长度为 6 的完整窗口，而约 80% 的 poetry step 只用 3 个或更少 token。

## 性能优化和 zero-overhead scheduling（ZOS）

要把调度策略转成 wall-clock time，工程上要做两类事：降低每个 step 的成本，以及把 scheduler 隐藏到 forward 后面。两者叠加后，在 DeepSeek-V4-Pro、TP=8、B300、batch size 1 上，accept length 约为 5 时可以达到 **383.7 tok/s**。

我们把一批小 op 改写成融合 Triton kernel，例如 compact scatter、SWA page-index、verify-length top-k schedule、ragged-window packing。Block drafter 的 sampling path 也合入了 fused kernels，它的矩阵乘法被 sharding。在一个示例 profile 里，target verify 之外的部分减少了 1.7 ms，而 verify 本身是 7.3 ms。

DSpark 可以直接接入 SGLang 的 zero-overhead（overlap）scheduler，几乎不需要专门分支，只额外加入论文中的 two-step-back confidence relay。这部分并不是大量 DSpark 专用管线。SGLang 的 spec-v2 runtime 已经能在独立 stream 上把下一步 scheduling 和当前 forward 重叠起来；DSpark 作为一等 worker 接入：forward 输出以 async future 返回，跨 iteration 的顺序依赖通过 runtime 的 device-side barrier 处理，on-device page table 避免了 per-step host sync。Confidence relay 使用同一个 channel，读取两步之前的信息。decode loop 最终可以做到 per-step 没有 bubble，比关闭 scheduler 时紧凑约 1.5 倍。

![图 5：batch size 1 的 decode trace。上图关闭 overlap scheduler，会在 run_batch iteration 之间以及 draft-generate 和 target-verify 阶段之间出现 bubble；下图打开后，这些阶段背靠背执行。](https://files.mdnice.com/user/59/c8efb872-bb6e-4adc-b9f7-97386da3e818.png)

图 5：batch size 1 的 decode。上图是关闭 overlap scheduler，下图是打开。打开后，`run_batch` iteration 之间，以及一个 step 内 block-draft-generate 和 target-verify 阶段之间都没有 bubble。

## Profile cost table

![图 6：Additive SPS cost-table 拟合：raw step time vs. fit（a）和 throughput（b），以及 SPS 预测和实测 decode-step time（c），DeepSeek-V4 on H200。](https://files.mdnice.com/user/59/1d59ebbd-336a-455e-b9a5-3f2d86b0d32a.png)

图 6：Additive cost model：raw vs. fit（a）、throughput（b），以及 predicted vs. measured step time（c）。

我们用一个 additive model 表达 scheduler 对 step time `T(bs, K)` 的估计，其中 `K` 是 batch 的额外 verify tokens：

```text
T(bs, K) = bias + alpha(bs) + theta(M), M = bs + K
```

这里 `alpha(bs)` 是随请求数扩展的底座成本，包括 draft pass 和一部分 attention，不会因裁剪改变；`theta(M)` 是目标模型的 verify-token 成本，也是裁剪唯一能省下来的项。Scheduler 的 argmax 会在预期 accepted tokens 和真实 marginal cost 之间权衡，所以只有 `theta` 足够大时才会出现 trim headroom。图 6(c) 用 live server 验证了这个模型的预测。

## 下一步

DSpark 已经在 SGLang 中可用；roadmap 跟踪在 [sgl-project/sglang#30344](https://github.com/sgl-project/sglang/issues/30344)。后续工作包括：

- **Cost model 和 scheduling**：更强、更在线/自适应的 cost model，以及进一步改进 dynamic scheduler。
- **Model coverage**：支持更多 dense 和 sparse 模型。
- **Parallelism**：覆盖更多并行模式和 serving topology。
- **Observability**：把 block-accept estimator、跨 checkpoint 的 confidence calibration 等指标生产化。
- **Robustness**：强化 full-CUDA-graph 路径，并扩大 stress / regression testing。

感谢 DSpark 作者和 DeepSeek 提供算法与模型。

## 附录：复现实验

下面所有命令都在预构建镜像中运行：

```bash
docker pull lmsysorg/sglang:dev-dspark
```

也可以从 [sgl-project/sglang#30261](https://github.com/sgl-project/sglang/pull/30261) 源码构建，固定到 commit [`692c5f7d`](https://github.com/sgl-project/sglang/commit/692c5f7d532f129424b57961c262bbd253b411dc)。

**图 1、图 3、图 6：frontier server（DeepSeek-V4-Flash, H200, DP4）。** 启动 DSpark arm：

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

- **non-spec**：去掉 `--speculative-*`，并加载 `--model-path deepseek-ai/DeepSeek-V4-Flash`。
- **MTP**：使用同一个 target，配合 `--speculative-algorithm EAGLE --speculative-num-steps {1,3} --speculative-eagle-topk 1 --speculative-num-draft-tokens {2,4}`，并在每个 batch size 下从两种配置中取最好结果。
- **DSpark compact 或 static**：设置 `SGLANG_RAGGED_VERIFY_MODE=compact|static`。
- 执行带 SPS table 的 compact mode 时，使用 `--speculative-dspark-sps-table-path sps_table.json`。
- 图 3 的 **no-trim** arm 使用 `SGLANG_RAGGED_VERIFY_MODE=compact`，但不提供 SPS table，也就是用 ragged path 跑完整窗口。

用固定 prompt 扫不同 batch size 来驱动任意 arm：

```bash
python3 -m sglang.benchmark.one_batch_server \
  --model None --base-url http://127.0.0.1:30000 \
  --batch-size 1 8 16 32 64 96 128 160 192 256 --output-len 1024 --temperature 0.7 \
  --fixed-prompt-file frontier_prompt.txt --fixed-prompt-apply-chat-template --show-report
```

固定 prompt 在[这里](https://gist.github.com/sglang-bot/71cc966dce295e78cbd0baddc402d151)（`frontier_prompt.txt`），它由 16 个拼接起来的 GSM8K 问题组成，让生成是真实内容。由于不同数据集上的推测解码 accept length 不同，用户也可以在自己的数据上测试。

图 6 的 cost table 来自一次 profiling run：启动 `compact` 时设置 `SGLANG_DSPARK_ENABLE_SPS_RECORD=1 SGLANG_SIMULATE_ACC_LEN=1.0`，然后用下面命令拟合 additive model（在 input-len 512 下扫 batch x verify-fraction grid）：

```bash
python3 -m sglang.benchmark.dspark_sps_profiler all
```

**图 4：mixed traffic。** 使用和图 1 相同的 server，把 `--mem-fraction-static` 设为 0.7，block size 为 6；通过 `SGLANG_RAGGED_VERIFY_MODE` 跑三种模式（`static` / `compact` / `cap-accept`），并驱动一个 gsm8k + arena-hard + poetry 混合请求集，用 non-streaming makespan throughput 计量。

**图 5：zero-overhead（DeepSeek-V4-Pro, B300, TP8）。**

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
