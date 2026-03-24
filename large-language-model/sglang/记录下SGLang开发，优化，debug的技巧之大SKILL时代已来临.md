# 0x0. 前言

之前在[记录下SGLang 开发，编译和Profile的几个小技巧](https://zhuanlan.zhihu.com/p/1939041055208112436)和[记录下SGLang 开发，debug的几个技巧第二弹](https://zhuanlan.zhihu.com/p/1984750078074839122)中，记录了一些 SGLang 开发、debug、profile 的技巧，这篇文章继续来聊一下 Agent（Claude Code/Codex）时期的近况。

# 0x1. Agent 的冲击

在经历了 Codex + GPT5.4 Extra High 狂蹬 2 周做的事情之后，我觉得之前自己的学习基本失去意义，一些难理解的知识和一些总结的技巧，其实只是大模型在设置合适 context（SKILL）下的 Token 而已。Codex + GPT5.4 已经达到了非常强的能力，这和 2025 年的感觉完全不一样，真正的智能似乎已经出现了，至少在编程开发领域是这样。读者在 Codex 或者 Claude Code 中可以安装 SGLang 提供的一些 SKILLS，完成 kernel 编写、benchmark 和测试编写、kernel 迭代优化、模型编写、模型优化、CUDA Crash 自动 debug、自动二分坏掉的 commit 等等这些之前需要付出大量人力的工作。

![](https://files.mdnice.com/user/59/82aa43fe-2b76-4f74-a119-bcdfb7c89471.png)

大家感兴趣可以去看这些 SKILL。最近我基于 Codex 和这些 SKILL，让 SGLang Diffusion 的 Z-Image 单卡速度提升 40%，Qwen/Qwen-Image-2512 的单卡速度提升 20%+，并挖掘了一个 kernel fuse 的 pattern：https://github.com/sgl-project/sglang/pull/20395。然后如果用一些更适合 kernel 开发的 Agent 框架，例如 https://github.com/TongmingLAIC/AKO4ALL ，可以让已有的一些 kernel 更容易地获得提升，例如：

![](https://files.mdnice.com/user/59/d0fab410-1beb-400e-b5ce-1256d495e096.png)

然后等待40分钟就让整个模型的端到端性能又提升了2个百分点。

![](https://files.mdnice.com/user/59/c98b6880-41f3-4d28-929e-5d0f91c1c73d.png)

这些足以证明当前阶段 Coding Agent 的高超能力。如果你觉得 Agent 还不行，那得思考一下你使用的方式，以及 context 是否给对了。当然，也有一些领域 Agent 还是无法和人类专家对比，但是可怕的是大模型还在进化，gap 只会变小。

![](https://files.mdnice.com/user/59/380c2520-a63a-432b-b775-c5fd4b1edfad.png)

# 0x2. Agent 流程可以优化的地方

- 对于大模型推理的开发，很多流程是相对固定的，我们就需要抽出通用高效的 SKILL 来帮助 Agent 更好地工作，这是目前最需要做的工作。例如在 https://github.com/sgl-project/sglang/pull/20910 中，就受 Flashinfer 的 API logging 启发，做了一个针对 SGLang CUDA Crash 的 debug skill。有了这个 skill 之后，当碰到模型有 CUDA crash（无论是接口层面还是 kernel 层面）时，都可以更方便高效地用 Codex 去定位到出错的 kernel。如果让人去做这个流程，就会非常繁琐和耗时。因此第一件可以做的事是蒸馏自己，蒸馏以前的开发者，让推理框架开发、优化模型都可以通过 Agent 去转起来。这里有很多工作可以做，可以一边做开发一边总结。
- 研究更加专业化的知识，让它们成为 Agent 的资料库，得到更好的效果。例如总结人类专家的经验，做一个特殊的 SKILL 挂到 Agent 上，合法开挂。例如你可以挑一个你觉得含金量非常高的 cutlass 系列 blog、triton 系列 blog，或者一个专门的人类代码优化库，把其中的一些优化代码压缩总结成一个 SKILL 挂给 Agent。
- 流程也非常重要。给一个特定的 kernel 做优化，如果没有合适的流程，可能结果并不会很好，这方面可以参考 https://github.com/TongmingLAIC/AKO4ALL 和 https://github.com/RightNow-AI/autokernel 等等。

例如在 [记录下SGLang 开发，debug的几个技巧第二弹](https://zhuanlan.zhihu.com/p/1984750078074839122) 中，提到了一个长期崩溃的问题，我们就可以把整套流程整理成一套 SKILL，来调试这种生产环境中会出现的困难问题。当然，个人感觉程序员的专业价值也会在一个个 SKILL 中被逐渐削弱。

# 0x3. 警惕

不要让 Agent 挂在那里，然后完全不看它的开发流程，就把最后的结果拿来交付。目前实际使用中，Agent 还是会有一些偏离方向的修改，可能会造成破坏性的后果，需要警惕。

![](https://files.mdnice.com/user/59/380c2520-a63a-432b-b775-c5fd4b1edfad.png)

当这张梗图成为现实，世界将会彻底改变（笑）。

# 0x4. 人的价值

现在我们是否可以分清楚提交的 PR 到底是人类写出来的，还是 AI coding 出来的？人的价值在哪？

这种担心挺正常的，但人的价值不会消失，只是从"一行行手写代码"变成了"定义问题、梳理上下文、判断结果靠不靠谱"。以前厉害的开发者可能是亲手扣 kernel、手动串 benchmark 和 debug；现在更值钱的是能把这些经验沉淀成 SKILL、搭出自动验证闭环、一眼看出 Agent 产出有没有走偏的人。人正在从干活的人变成设计的人、把关的人、提炼的人。蒸馏世界的知识，蒸馏 AI 的输出，最后蒸馏自己。

在推理框架、kernel 优化、模型适配这些复杂场景里，稀缺的早就不只是"会写代码"，而是"知道该优化什么、瓶颈大概在哪、怎么设计一个稳定可复用的流程"。Agent 确实能把事情做得飞快，但它需要目标清晰、资料齐全、验证标准过硬。缺了这些，再强的模型也不过是高速生产一堆看着像回事、并不正常work的东西。人工智能，到头来还是得靠"能工智人"（笑）。

综上，个人认为当前这个时间点，成为 Vibe Coding 高手已经成为唯一出路。同时也期待GPT5.4和Opus 4.6到底能进化到什么地步。




