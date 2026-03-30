# 0x0. 前言

在前面的[记录下SGLang开发，优化，debug的技巧之大SKILL时代已来临](./记录下SGLang开发，优化，debug的技巧之大SKILL时代已来临.md)和[Codex正在重塑传统推理框架开发流程](./Codex正在重塑传统推理框架开发流程.md)里，我更多聊的是一件事：Agent 时代，SGLang 这类推理框架的开发、优化、debug、benchmark，普通模型的优化越来越像“给足 context 之后让模型高速推进”的问题了。

本周末搞了一下cc-connect对接飞书和本地的Codex来进行SGLang模型优化的流程，听起来像是个玩具，但真的跑起来之后会发现，这不是“在手机上写代码”，而是“在手机上调度一个真正能 SSH、能 benchmark、能改 SGLang、能连远端 GPU 机器做profile，做benchmark验证的 Codex 工程体。现在离随时随地将你的想法变成现实更加接近。能完成这套工作流也依赖了一些前置的skills工作：

![](https://files.mdnice.com/user/59/467c923f-8b93-4193-90e3-860030c9042e.png)

和SGLang相关的skills都可以拉取SGLang仓库然后一键安装：https://github.com/sgl-project/sglang

远程连接相关的skills大家可以让Codex或者Claude Code自己创建

# 0x1. 为什么是飞书和cc-connect

因为我目前已经处于在本地mac的固定文件夹下使用Codex去处理工作的模式，这个模式不方便的点在于你必须要在电脑上进行操作并且为了让任务顺利完成还要盯着它的思考过程。一些关键信息容易看漏且整个对话很长，工具调用这种日志又非常多影响阅读，懒得去翻中间过程就容易搞出问题。

通过cc-connect对接飞书和本地的Codex的流程大概是这样：

```text
飞书消息
  -> cc-connect
  -> 本地 Codex CLI
  -> 本地 SGLang repo / 本地 SKILL
  -> SSH 到 b200 / h100 等远端机器
  -> 跑 benchmark / profile / test
  -> 把关键结果回到飞书
```

飞书只是入口，`cc-connect` 只是桥，真正干活的是我本地那套已经验证过可以顺利work的 Codex 和 SGLang SKILLs以及远程服务器。重点在于，你真的可以用手机聊天的方式去开发了（许愿）

我感觉 OpenClaw 肯定也能做到这个，不过更想要的是一条从手机消息直达本地 Codex、直达本地 repo、直达远端 GPU 机器的最短控制链路并且配置更简单的方式。我已经有现成的 Codex CLI、SGLang SKILL、SSH SKILL完整的开发环境，`cc-connect + 飞书` 最有价值的地方就是几乎不动这些模块，只是把控制面自然延伸到了手机上，多了一个桥。所以调研了下感觉cc-connect比较合适

# 0x2. 为什么这套方式有用处

举个飞书里我会直接扔给 Agent 的 prompt：

![](https://files.mdnice.com/user/59/fe976435-1052-43ad-bc99-21a46d99a615.png)

![](https://files.mdnice.com/user/59/51b08681-cca0-4251-8d2e-49b75ce6d488.png)

![](https://files.mdnice.com/user/59/1837890d-aab4-46c4-8395-f02b0d605826.png)


这个prompt会走下面的流程：

- **Preflight**：用仓库里自带的 `diffusion_skill_env.py` 找到 repo 根目录、确认可写；`HF_TOKEN`、关掉 FlashInfer 版本检查、空闲 GPU，能跑 nightly 预设的话再准备 cat 图片等资源路径。
- **指标**：主看 **denoise 总耗时**（DiT 前向跨步数累加）；端到端 latency 当辅助。改 kernel 前后要对齐画质，别快成废片。
- **Torch Profiler（Level 1）**：`sglang generate` 加 `--profile`，必要时 `record_function` 标到 block 级，从 trace 里把最重的 CUDA op 名字扒出来；同时盯 **torch.compile**：新融合的 kernel 要用 `torch._dynamo.explain` 看有没有 **graph break**，不行就 custom op 包一层（extern 或 `@register_custom_op`）。
- **Nsight Systems（Level 2）**：`nsys profile` 抓一版 trace，再单独 `time` 一版无采样的墙钟，丢进 `gputrc2graph.py` 做按类别（gemm / attn / norm / NCCL…）的占比；`CPU(non-GPU)` 飙高往往和 torch 图碎、Python 调度有关。
- **Nsight Compute（Level 3）**：写新 kernel 或prfoile某几个top kernel性能时用上 `ncu`，看带宽、occupancy、stall等。
- **Baseline / 回归**：每次 benchmark 顺手写 `--perf-dump-path`，改完用 `compare_perf.py` 做前后 JSON 对比；kernel 融合、AKO4ALL 那套迭代替换可以跟 **ako4all** 技能里的 microbench、ncu profile结果走。

这些流程在SKILL和远端环境就绪之后都可以自动进行，并且通过飞书消息的方式对关键信息进行交互方便我把握优化方向，整个调优过程变成自动化。有用的地方在于，你可以在任何空闲的时间聊下天把你的想法就交给Codex蹬起来了。

# 0x3. 飞书 + Codex 从“能连上”到“真正能用”，还有些Tricks

## 0x3.1 让它能做真正的远端开发

`cc-connect` 里把 Codex 从默认的 `suggest` 改成 `yolo`，这个几乎是刚需：联网、SSH、shell、进 Docker、蹭远端 GPU，哪样都不是保守sandbox能做完的。不切yolo模式，大半验证动作会受限。

## 0x3.2 让飞书里只保留值得人类看的消息

Bash、`rg`、thinking 等全量灌进飞书，聊天窗很快变成刷屏现场，手机上看尤其折磨。

我这边是按下面调的：

- 默认 `quiet = true`
- 关掉 `stream_preview`
- 飞书进度样式用 `compact`

完成之后，屏幕上多半是里程碑、结论、需要拍板的关键回复，而不是类似这种「工具 #35: Bash」接一串乱七八糟的东西。

## 0x3.3 一个项目可以同时挂多个飞书机器人

同一个 `common` 项目下面挂两个飞书机器人没问题，两边都能指到同一套 `Codex + SGLang repo`，我试过可行。

用法可以很随便：一个当稳定入口、一个瞎折腾；或者一个自己用、一个给固定搭档。本质是同一套后端、多条聊天入口，比「每个机器人单独起一套环境」轻得多，也贴近平时一个 repo、多人从不同会话进来的习惯。当然Token消耗你也得顶得住才行。

## 0x3.4 一些常用的飞书命令

链路跑顺以后，飞书里我反倒更常敲短命令，而不是堆长 prompt：

- `/new`、`/new b200-debug`：开新会话，免得新任务糊进旧上下文
- `/stop`：掐掉正在跑的任务
- `/list`、`/switch <id>`、`/current`：翻会话、切会话
- `/history 20`：扫一眼最近聊天
- `/mode yolo`、`/reasoning high`、`/quiet`：权限模式、推理强度、要不要把 thinking / 工具进度摊出来
- `/help`：忘了就翻一眼

想知道现在进行到哪了，我一般不会为了这个专门记命令，可以直接写一句话，例如：

```text
总结一下当前进度、已经完成的修改、正在跑的验证，以及下一步计划。
```

模型会带着现有上下文回复你一句。

# 0x4. 警惕

当然，这套流程不是没有代价的。

第一，`yolo` 一开，能力边界就真的放大了。它之所以强，是因为它能 SSH、能跑 shell、能动远端机器；但也正因为如此，review 和结果把关更重要了。

第二，移动端天然更适合“判断”和“指挥”，不适合“逐行审 patch”。所以对于真正要紧的修改，最好还是回到桌面上看 diff、看 benchmark、看测试结果。

第三，入口变轻之后，更要警惕把 Agent 当成全自动黑盒。你可以让它一直蹬，但不能不看它蹬出来的东西。

最后，再次强调注意token消耗。

# 0x5. 总结

如果说前两篇文章讲的是：Codex 和 SKILL 正在重塑 SGLang 这类推理框架的开发方式；那这篇文章想补上的点就是：

**`cc-connect + 飞书` 把这套开发方式进一步从桌面释放到了手机上。**

你可以吃饭的时候在手机上指挥Codex写 Triton，写 CUDA kernel， debug cuda crash, 优化模型，甚至添加模型，一切框架端需要做的事情。前提是skills和开发环境要搭建好，而且即使后续模型产生了很大的进步，我们的skills仍然也可以留着用，不会过时。


# 0x6. 飞书和 cc-connect 详细配置教程

详细配置这里我大多数都是让Codex自己给我配置的，我只配置了一下飞书部分。

