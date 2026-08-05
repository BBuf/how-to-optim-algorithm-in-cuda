# 0x0. 背景

本文记录最近使用 KDA（https://github.com/mit-han-lab/kernel-design-agents） 优化 SGLang Diffusion Kernel 的流程。主要内容包括流程中暴露出的工程问题和解决方法、prompt 约束和验证规则。

在此前 kernel-pilot 的 [Humanize loop 方式](https://mp.weixin.qq.com/s/pScZ_9cA-6cWUPjfcGjNyg) 和部分公开 Kernel Agent 框架的基础上，kernel-pilot 最近切换到 KDA 风格的任务组织方式。该组织方式将每个 kernel 拆成独立任务目录，保留 prompt、baseline、benchmark、review 和结果记录，便于批量生成和验证 SGLang Diffusion Kernel 优化任务。相关仓库为 https://github.com/BBuf/kernel-pilot。

kernel-pilot 中的 SGLang Diffusion Kernel 任务已经被封装成可直接启动的脚本：

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-2/7e484218-844b-47dc-8f71-cc493686af14.png)

每个脚本对应 SGLang Diffusion 系统中的一个优化 kernel 或 fuse kernel。baseline 覆盖 CUDA、Triton 和 CuTe-DSL 等实现。

运行其中一个脚本后，会看到类似下面的输出：

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-1/4873e659-176d-4f9d-b415-07389247ad0e.png)

脚本会打印两个阶段的命令：第一阶段是 gen-plan，第二阶段是 RLCR loop（Humanize 的循环开发与审查流程）。命令中的参数主要用于分支隔离和任务隔离，使多个 kernel 优化任务能够并行执行。一次实验中，14 个 Agent 在一天内并行处理多个 SGLang Diffusion Kernel 任务，总花费低于 2000 美元，任务执行模型为 Opus4.8，review 模型为 Codex GPT5.5 High。

这些任务使用的 shape 来自真实 SGLang Diffusion benchmark。具体方式是运行 https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/benchmark-and-profile.md 中的约 20 个模型，记录待优化 kernel 的输入信息，包括 shape、dtype 等。由于需要覆盖不同 GPU 平台，同一类 kernel 会分别生成 B200 和 H200 任务。

实验结果如下：

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/ad7cef08-e9fe-436f-a2f0-a741cbff3aa1.png)

上述任务均完成了 RLCR 循环，多数任务迭代轮数约为 6 轮。各任务最终生成的 `.cu` 文件和对应 benchmark 可在 https://github.com/BBuf/kernel-pilot/tree/main/kernels 中复现。

这些 kernel 在部分计算量较大的 Diffusion 模型中不是主要热点，因此端到端收益取决于算子在 workflow 中的占比。在 qknorm-rope 和 norm-scale-shift 占比较高的 Qwen/Qwen-Image-2512 上，B200 平台测得约 10% 的 end2end 提升，见 https://github.com/sgl-project/sglang/pull/27392。

![](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-2/755334e0-eadb-4bea-b686-a6175c2f3934.png)

下面是该模型 main 分支和 PR 分支中相关 kernel 的耗时与占比对比：

![main 分支的相关 kernel 耗时和占比](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/d9787134-d48a-4f3d-a194-4352ebd670b2.png)

![PR 分支的相关 kernel 耗时和占比](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-3/bfc629e1-5d91-4432-82de-40c84a976377.png)

# 0x1. 使用 KDA 优化 SGLang Diffusion Kernel 的坑和 prompt 实践

## 0x1.1 多个 Agent 并行优化不同 kernel 时的隔离方式

多个 kernel 任务不应共用同一个 checkout 直接运行。Humanize/RLCR 会写入 `.humanize/`，并产生 review base、round prompt、summary、review result、中间代码、benchmark 和 profile artifact。多个 Agent 共用一个 git 工作区时，容易出现以下问题：

- `.humanize` 状态串到另一个任务。
- Codex review 的 diff 不是当前 kernel 任务的 diff。
- benchmark/profile artifact 混在一起，无法判断数字来源。
- 一个 Agent 修改 launcher 或 docs 时，另一个 Agent 仍基于旧任务定义运行。

kernel-pilot 目前通过 `scripts/launch_kda_kernel_task.sh` 为每个任务创建独立 worktree，并创建两个分支：

- `kda/<task-label>-<run-id>`：当前 Agent 实际工作的分支。
- `kda-base/<task-label>-<run-id>`：Humanize RLCR review 使用的固定 base。

launcher 会进入生成 worktree 中的具体 kernel folder，并把 `CLAUDE_PROJECT_DIR` 设置到该 folder。Humanize hooks 看到的 project root 是当前 kernel 任务，而不是整个 `kernel-pilot` 仓库。每个任务的 `.humanize/kernel-agent/draft.md`、`baseline/`、`solution/`、`bench/`、`docs/` 都位于独立 worktree 中。

只准备 worktree、不启动 Claude Code 时，使用：

```bash
KDA_NO_CLAUDE=1 ./scripts/launch_kernels/k03_b200_diffusion_qknorm_rope__multi_shape.sh
```

该命令会打印当前任务的 worktree、branch、review base，以及后续需要在 Claude Code 中执行的 Humanize 命令。review base 使用 launcher 打印出的值。

## 0x1.2 使用 Ghostty 管理多个 Agent 的 CLI 窗口

多 Agent 并行运行时，CLI 分屏用于持续展示多个任务状态。Ghostty 的原生分屏支持在同一个 tab 中保留多个独立 shell，用于多个 kernel 优化任务。

本次实验使用 Ghostty 在一个窗口内管理多个 CLI 会话，便于同时观察多个任务的 RLCR 状态。

Ghostty 在 macOS 下默认快捷键如下：

| 动作 | 快捷键 |
| --- | --- |
| 新建 tab | `Cmd + T` |
| 向右分屏 | `Cmd + D` |
| 向下分屏 | `Cmd + Shift + D` |
| 切换 pane | `Cmd + Option + 方向键`，或者 `Cmd + [` / `Cmd + ]` |
| 调整 pane 大小 | `Cmd + Ctrl + 方向键` |
| 平均分配 pane 大小 | `Cmd + Ctrl + =` |
| 放大/恢复当前 pane | `Cmd + Shift + Enter` |
| 关闭当前 pane | `Cmd + W` |

实际绑定通过以下命令查看：

```bash
ghostty +list-keybinds --default
```

## 0x1.3 Benchmark 和 `.cu` 的导出方式需要固定成模板

早期流程尝试通过 `kda_kernels` overlay（运行时覆盖层）将优化后的 kernel 接入 SGLang。该方式在 KernelPilot #25（https://github.com/BBuf/kernel-pilot/pull/25）暴露出一个问题：overlay promotion（把候选实现提升到运行时覆盖路径）会用 `setattr` 将 SGLang 中由 `@register_custom_op` 修饰的对外 op 替换为普通 Python 分发函数。此时测得的“加速”可能来自绕过 CPU 侧 wrapper、custom op 注册或 torch.compile 相关逻辑，而不是 device kernel 本身变快。

SGLang 的生产路径需要保留对外入口语义，例如 `@register_custom_op` 对 `torch.compile` 和 CUDA graph 兼容性有影响。移除这一层后得到的 benchmark 数字不能直接代表生产路径性能。

kernel-pilot 后续采用 standalone diffusion benchmark contract（脱离 SGLang 运行时、在任务目录内对 baseline/candidate 做对称计时的约定），并提供统一的 benchmark 和 `.cu` 导出 Python 接口模板：

- benchmark runtime 不 patch、不 import、不 monkey-patch（运行时替换）SGLang 源码。
- Agent 从最新 upstream SGLang `main` 复制相关 kernel 源码到当前任务的 `baseline/`。
- `baseline/` 和 `solution/` 通过同一套本地 ABI 暴露；如果是 `.cu`，两边都使用相同的 tvm-ffi（TVM FFI 调用层）CUDA wrapper，wrapper overhead 对称。
- 输出 tensor 都采用 destination-passing（输出由调用方传入），避免一边在 timed path 中分配，一边不分配。
- workload 在 tuning 前冻结，后续不修改 shape、tolerance 或 scoring。
- `bench/benchmark.py` 从统一模板开始，包含 isolated subprocess、fresh inputs、poison outputs（先写入哨兵值，检查输出是否真的被覆盖）、correctness-before-timing、CUDA event、inner-loop amplification（单次计时中重复调用以放大计时信号）、interleaved A/B、per-row median/mean/std/min/p10/p90。这里参考了一下 AKO4X 的 benchmark 模板，具体见： https://github.com/TongmingLAIC/AKO4X/tree/main/templates/skills/benchmark
- headline 指标使用所有 production workloads 的 equal-weight geomean。

例如 KernelPilot #40（https://github.com/BBuf/kernel-pilot/pull/40）的 H200 `norm_scale_shift` 记录 baseline 来自 SGLang `main@133254086b`，并包含 baseline-vs-baseline A/A 对照（两边都跑 baseline，用于检查计时噪声），geomean 为 `0.9992`。KernelPilot #41（https://github.com/BBuf/kernel-pilot/pull/41）的 B200 `fuse_scale_shift` 使用 copied Triton baseline 与 tvm-ffi CUDA candidate，两边均采用 destination-passing ABI，最终报告 `2.7570x` geomean。

该流程要求先验证 baseline 和 benchmark 对称性，再报告 kernel 优化结果。核心约束是 baseline 和 candidate 的导出、调用、计时路径保持对称。

## 0x1.4 避免 Fast Math 造成 reward hacking（奖励漏洞）

Fast Math 会影响数值路径。CUDA kernel 添加 `--use_fast_math` 后，benchmark 数字可能变化；但 diffusion 中的 Norm、RoPE、scale/shift、GroupNorm 会影响后续 denoise 过程，误差不会停留在单个算子内。

kernel-pilot 目前的规则是：除非 copied upstream SGLang baseline 本身使用 `--use_fast_math`，并且 candidate 使用完全相同的 flag，否则 candidate 默认不使用该选项。所有可能改变数值或 codegen 的编译选项都需要对称。

还需要避免以下 benchmark 偏差：

- 放松 tolerance，使错误结果通过。
- 不检查 NaN/Inf，或不 poison output。
- 用端到端耗时观察 Python wrapper 变化，却将结果归因于 device kernel。

correctness、tolerance、workload 和 baseline contract 需要由任务规则提前固定，Agent 在该边界内优化性能。这些约束已经写入各优化任务的 prompt。

## 0x1.5 **Bash 版本对 Humanize hook 的影响**

实验使用的 macOS 系统自带 `/bin/bash` 版本为 3.2.57。Humanize 中多个 hook 脚本使用 `#!/usr/bin/env bash` 和 `set -euo pipefail`。当 PATH 中 `/bin` 排在 Homebrew Bash 之前时，hook 可能由 Bash 3.2 执行，即使系统已经安装 Bash 5.3.9 到 `/opt/homebrew/bin/bash`。

Bash 3.2 在 `set -u` 下展开空数组时会报错。最小复现如下：

```bash
/bin/bash -uc 'a=(); env "${a[@]}" true'
```

新版本 Bash 通常不会报错，Bash 3.2 会输出 `a[@]: unbound variable`。BBuf/humanize #1（https://github.com/BBuf/humanize/pull/1）修复的路径是 `hooks/lib/template-loader.sh::render_template`：当没有任何 `VAR=value` 传入时，代码会构造空数组 `env_vars=()`，然后执行：

```bash
env "${env_vars[@]}" awk ...
```

某些 Stop hook 渲染静态 block 模板时不传变量，会触发 template render 失败，并在外层看到：

```text
Stop hook error: Failed with non-blocking status code: No stderr output
```

兼容 Bash 3.2 的写法如下：

```bash
env ${env_vars[@]+"${env_vars[@]}"} awk ...
```

KernelPilot #28（https://github.com/BBuf/kernel-pilot/pull/28）增加了 launcher 层面的现代 Bash guard：优先使用 `KDA_BASH_BIN`，否则查找 PATH、`/opt/homebrew/bin/bash`、`/usr/local/bin/bash`，拒绝 `/bin/bash` 3.2，并将选中的 Bash 通过 `PATH`、`SHELL`、`KDA_BASH_BIN` 传给后续 Claude/Humanize hooks。

启动 kernel-pilot launcher 前可设置：

```bash
brew install bash
export KDA_BASH_BIN=/opt/homebrew/bin/bash
export PATH="/opt/homebrew/bin:$PATH"
bash --version
```

## 0x1.6 **Codex hook feature 名称变化导致的 Stop Hook 递归**

Humanize 的 RLCR 会在部分阶段启动 nested Codex helper，例如 `codex exec` 用于 summary review，或 `codex review` 用于代码 review。为了避免 nested Codex 再触发外层 Humanize Stop Hook，旧逻辑会传入：

```bash
--disable codex_hooks
```

Codex CLI 的 hook feature 名称后来发生变化。旧版本可能使用 `codex_hooks`，新版本暴露 `hooks` 和 `plugin_hooks`。如果 Humanize 只禁用 `codex_hooks`，新 Codex 上实际生效的 hook 仍可能被加载，nested Codex 会重新进入同一个 Stop Hook，从而形成递归。

同时，不能直接把所有 feature name 都传给 `--disable`。部分旧 Codex 虽然支持 `--disable`，但遇到未知 feature name 会报错。因此兼容做法是先探测，再传参。

PolyArch/humanize #194（https://github.com/PolyArch/humanize/pull/194）的修复逻辑如下：

- 在 `loop-codex-stop-hook.sh`、`ask-codex.sh`、`bitlesson-select.sh` 中探测 Codex 是否支持 `--disable`。
- 逐个测试 `hooks`、`plugin_hooks`、`codex_hooks` 哪些 feature name 被当前 Codex 接受。
- 只将支持的 feature 作为 `--disable <feature>` 传给 nested Codex。
- 缓存探测结果，避免重复 probe。
- 增加现代 Codex 与 legacy Codex 的测试覆盖。

该修复后，Humanize plugin version 同步到 `1.17.0`。当 Codex review 长时间停在 `running stop hook`、token 消耗持续增长且没有新 summary/review result 时，可优先检查该问题，并通过升级 Humanize 处理。

## 0x1.7 Prompt 需要允许按 shape 特化并 dispatch

真实的 SGLang Diffusion workload 不只有一个 shape、一个 dtype 或一种 layout。同一个 kernel family 可能包含：

- 小 token 数场景，主要受 launch/host overhead 影响。
- 中等 shape，需要匹配该 shape 的 block/grid 覆盖 SM。
- 大 shape，接近 HBM roofline。
- contiguous 和 channels-last-3d 两种布局。
- bf16/fp16/fp32 混合。
- `[1, D]`、`[B, S, D]`、`[B, F, 1, D]` 等不同 operand 形态。

如果 prompt 未明确允许 shape-specialized dispatch（按 shape 特化并分发），Agent 可能生成一个覆盖所有 shape 的通用 kernel。该策略会导致部分 shape 加速、部分 shape 回归，geomean 仍可能保持可接受，但生产 workload 可能命中某个回归 bucket（按 shape、layout 或 dtype 分出的 workload 分组）。

kernel-pilot 的 `diffusion_kernel_rules.md` 已明确允许 shape-specialized kernels、template variants、autotune tables 和 dispatchers（按条件选择 kernel/config 或 fallback 的分发逻辑）。使用 dispatch 时需要在 `docs/dispatch.md` 记录：

- bucket condition，即每个分组的匹配条件。
- 每个 bucket 走哪个 baseline/candidate entry。
- 每个 bucket 的 latency 和 speedup。
- dispatch 原因。

KernelPilot #43（https://github.com/BBuf/kernel-pilot/pull/43）的 B200 `group_norm_silu` 使用多 bucket 策略：contiguous 64K-2M 走 `cont_split`，channels-last 走 `nchw_last`，小 shape 走 vectorized one-pass，超大 contiguous bucket 回到 baseline-equivalent path（行为和 baseline 等价的回退路径）。KernelPilot #42（https://github.com/BBuf/kernel-pilot/pull/42）的 H200 `group_norm_silu` 记录了 giant rows 的多种变体尝试，并对特殊 giant row 作出 owner ruling（由任务 owner 明确裁定继续、回退或停止）。

KernelPilot #40（https://github.com/BBuf/kernel-pilot/pull/40）的 H200 `norm_scale_shift` 也体现了平台差异：B200 上有效的 32B/thread vector width 到 H200 上回归，H200 最终选择 16B/thread。

因此 prompt 中需要明确：任务可根据 shape、layout、dtype 或平台差异使用不同 kernel/config，并通过 fail-closed dispatcher（条件不匹配时默认回退 baseline 的分发逻辑）保持生产路径可解释。

## 0x1.8 Prompt 中的其他规则和优化目标

- 先恢复 K/R/W（Kernel/Reference/Workload，即 kernel 语义、参考实现和工作负载）：kernel semantics、correctness oracle、workload/benchmark methodology 明确后再写 kernel。
- baseline 来自最新 SGLang `main`，并记录 commit、copy 的文件和 resolution time。
- benchmark runtime 不 import SGLang，SGLang 仅作为源码提供者。
- benchmark/profile 前后记录 GPU idle evidence。
- no-go（停止继续推进该方向）需要 correctness、benchmark/NCU evidence 和明确 active bound（当前主要瓶颈），不能只基于第一轮回归判断。
- raw NCU / Nsight / scratch artifact 留在任务目录，PR 中只放小的 provenance 和 per-shape 结果。

这些规则用于约束 Agent 的搜索空间，避免将 benchmark 漏洞或非生产路径差异当成 kernel 优化。

任务优化目标是根据 kernel 的 Roofline 模型，将 kernel 优化到接近性能上限。

## 0x1.9 **KernelWiki 和 ncu-report-skill 的使用**

KernelPilot 将 `KernelWiki` 和 `ncu-report-skill` 放在 `external/` 下作为外部知识源。SGLang Diffusion Kernel 规则要求每轮 RLCR 在决定下一次 code edit、benchmark、profile 或 no-go 之前，刷新 task prompt、当前 benchmark/profile evidence、`KernelWiki` 和 `ncu-report-skill`。因此，这两类知识源属于每轮迭代上下文，而不是一次性的 preflight 材料。

`KernelWiki` 主要用于 prior art 检索和优化方向筛选。例如 `qknorm_rope` 任务会查询 fused QKNorm + RoPE、RoPE、FlashInfer / TensorRT-LLM / SGLang 相关记录，用来判断 staged design、vectorized load、shared memory staging、PDL（Programmatic Dependent Launch，CUDA 的程序化依赖启动）等方向是否有上游先例。部分任务也会记录没有直接命中的查询结果，避免把不存在的 prior art 当成依据。

`ncu-report-skill` 主要用于把 benchmark 数字拆成 active bound。优化过程中会根据 NCU 判断 kernel 是 memory-bound、compute-bound、launch-bound、long scoreboard（等待访存或依赖导致的长等待）、register pressure 还是 occupancy 受限，再决定下一步修改或 no-go。例如 `cutedsl_norm_tanh_mul_add` 用 NCU 确认 baseline 的 per-row `tanh` 计算瓶颈，并在后续用寄存器、occupancy、local memory 证据否决 prefetch 变体；`rotary_embedding` 用 NCU/roofline 说明部分 LTX-2 large shape 已接近 DRAM bandwidth ceiling，后续候选如果没有超过 active bound 就不继续展开。

并不是每个候选都需要完整跑 NCU。规则是：当 benchmark 结果无法直接解释、候选是否覆盖目标 workload 不明确，或最终 improvement/no-go 需要证据支撑时，需要使用 NCU 或给出等价的 roofline-style active bound（类似 Roofline 分析的主要瓶颈判断）。raw `.ncu-rep`、Nsight trace、profile 目录保留在任务工作区，文章或 PR 中只记录结论、关键指标和可复现命令。

# 0x2. 最终候选代码相对 SGLang baseline 的优化点

kernel-pilot 当前 `kernels/` 目录里，这批 SGLang Diffusion Kernel 任务的最终候选可以归纳为几类：把 CuTe-DSL/Triton 路径改成 tvm-ffi CUDA native kernel；针对真实 shape 做 dispatcher；对大 shape 用更宽的向量化或分块并行提高带宽利用；对小 shape 保留更轻的 CPU 侧路径；对没有稳定收益的 bucket 显式回退到 baseline 或记录 no-go。

下面只选 6 个有代表性的最终候选版本，不逐个展开所有任务。表中的“最终候选”指各任务目录里 `docs/results.md`、`docs/dispatch.md`、`docs/sglang_jit_export.md` 记录的 final/promoted 版本。

部分任务的结果来自 standalone benchmark；部分任务还做了 in-SGLang drop-in arbiter。这里的 drop-in arbiter 指把候选实现接到 SGLang 真实对外 API/custom-op 路径后做 A/B 验证；in-tree arbiter 是其中更严格的一种，候选 `.cu/.cuh` 实际放进 SGLang worktree，由原来的对外入口调用。表里的“结果”只用于说明该优化点对应的已记录证据。

| Kernel family | 平台 | SGLang baseline | 最终候选代码的主要优化 | 结果或边界 |
| --- | --- | --- | --- | --- |
| `cutedsl_norm_scale_shift` | B200/H200 | CuTe-DSL norm + scale/shift 路径 | 改成 native CUDA，并按 scale/shift/gate/weight/bias 组合 dispatch；B200 对 bf16-only 使用更宽向量，H200 最终保守到 16B/thread；无收益的 fp32-row bucket 回退到 baseline | B200 `1.3022x` end-to-end，H200 约 `1.27x`；体现平台差异和 fallback 规则 |
| `fuse_scale_shift` | B200 | Triton scale/shift、gated LayerNorm 路径 | 一个 CUDA module 覆盖多个 entry；scale/shift 走 row-grid 或 flat-vector 16B kernel；gated LN 用 exact-C register cached row kernel，减少 Triton tile 中被 mask 掉的 lane 浪费 | 19 个 production rows geomean `2.7570x`，所有行 >= `1.0391x` |
| `group_norm_silu` | B200 | SGLang Triton group_norm_silu，channels-last 会先转成 contiguous tensor | CUDA dispatcher 分为 `cont_small`、`cont_split`、`nchw_last` 和 fallback；contiguous mid bucket 用 split-group stats 填满 SM；channels-last 原生读取，避免 `x.contiguous()` | 160 production rows geomean `2.2794x`；giant contiguous bucket 接近 bandwidth bound，回到 baseline 等价路径 |
| `norm_infer` | B200/H200 | Triton `norm_infer` 和 `rmsnorm_onepass` | fp32 LN 用 float4/exact-N row kernel；bf16 RMS 根据规模选择 warp-per-row 或 tiled CTA；H200 的 fast path 插在原对外函数体内，保留 custom-op 注册 | B200 round-2 耗时 geomean `1.358x`；H200 vs main in-tree arbiter 耗时 `1.4475x` |
| `qknorm_rope` | B200/H200 | SGLang `qknorm_rope.cuh` baseline | 候选 `.cuh` 直接放入 SGLang worktree，保留 `@register_custom_op`；B200 large rows 分阶段缓存 cos/sin，减少跨 head 重复 global load；H200 `cossin-vec` 用 128-bit quartet load 将寄存器从 38 降到 32 | B200 in-SGLang arbiter `1.0970x`；H200 in-SGLang arbiter `1.0945x` |
| `rotary_embedding` | B200/H200 | SGLang Triton standard RoPE + LTX-2 split RoPE | native CUDA/tvm-ffi；standard RoPE 和 LTX-2 都走 128-bit vector load/store；H200 版本进一步使用 `__ldcs`/`__stcs` 处理一次读、一次写的数据流 | B200 vs 当前 main geomean `1.4660x`；H200 真实集成路径耗时 geomean `1.2977x` |

这些最终候选体现了几类优化模式：大 shape 通常围绕带宽、occupancy、long scoreboard 做处理；小 shape 常常受 kernel launch 和 CPU 侧路径限制；layout 特化和 shape dispatch 决定了某个优化是否能覆盖真实 workload；对于接近 Roofline 的 bucket，任务记录的最终动作可能是 fallback 或 no-go，而不是继续增加 kernel 复杂度。

# 0x3. 总结

KDA 用于 SGLang Diffusion Kernel 优化时，需要同时关注 Agent 生成 CUDA 代码的能力，以及实验系统对 benchmark、baseline、correctness 和 review 的约束。

一个可复用流程至少需要包含：

- 每个任务独立 worktree 和 Humanize 状态，支持多 Agent 并行。
- baseline 从最新 upstream main 复制，source lineage 可追溯。
- baseline/candidate 走同一 wrapper/ABI/build path。
- workload 来自真实 SGLang Diffusion preset capture，并在 tuning 前冻结。
- benchmark 包含 A/B 交错、inner-loop amplification、A/A gate（baseline 对 baseline 的稳定性门槛）、GPU idle evidence。
- correctness 覆盖生产 shape、canonical regression grid（固定回归用例网格）、NaN/Inf、poison output、fallback contract。
- prompt 允许 shape-specialized dispatch，并要求 `docs/dispatch.md` 解释每个 bucket。
- Humanize/Codex hook 环境稳定，避免 Bash 3.2 和 Stop Hook 递归影响长任务。

实验中，KDA 在一天内完成了一批 SGLang Diffusion Kernel 的 microbench 优化，包括 B200 `fuse_scale_shift` 的 `2.7570x` geomean，B200 `group_norm_silu` 的 `2.2794x` geomean，H200 `norm_scale_shift` 的 `1.27x` geomean。该流程同时记录了 benchmark 对称性、baseline provenance、correctness 和 review hardening。优化后的 kernel 可以按需取用，用到真实的模型推理框架中。
