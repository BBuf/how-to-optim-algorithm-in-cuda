## 这里记录一些训练大语言模型需要用到的基础知识和技巧

- zero.md 记录一些通信原语和zero算法的原理。

## 博客和文章收集

### 大模型知识介绍

- [ChatGPT 背后的“功臣”——RLHF 技术详解](https://www.cnblogs.com/huggingface/p/17040315.html)
- [深入浅出，解析ChatGPT背后的工作原理](https://zhuanlan.zhihu.com/p/597100830)
- [LLaMA模型惨遭泄漏，Meta版ChatGPT被迫「开源」！GitHub斩获8k星，评测大量出炉](https://zhuanlan.zhihu.com/p/612009979)
- [LeCun狂赞：600刀GPT-3.5平替！ 斯坦福70亿参数「羊驼」爆火，LLaMA杀疯了](https://zhuanlan.zhihu.com/p/613880958)
- [LeCun转赞：在苹果M1/M2芯片上跑LLaMA！130亿参数模型仅需4GB内存](https://zhuanlan.zhihu.com/p/613602977)
- [Stanford Alpaca (羊驼)：ChatGPT 学术版开源实现](https://zhuanlan.zhihu.com/p/614354549)
- [Alpaca-Lora (羊驼-Lora): 轻量级 ChatGPT 的开源实现（对标 Standford Alpaca）](https://zhuanlan.zhihu.com/p/615646636)
- [Alpaca-cpp（羊驼-cpp）: 可以本地运行的 Alpaca 大语言模型](https://zhuanlan.zhihu.com/p/616267309)
- [NLP（九）：LLaMA, Alpaca, ColossalChat 系列模型研究](https://zhuanlan.zhihu.com/p/618695885)
- [全球最大ChatGPT开源平替来了！支持35种语言，写代码、讲笑话全拿捏](https://zhuanlan.zhihu.com/p/616917667)
- [国产ChatGPT又开源了！效果大幅升级，在手机上也可以跑](https://zhuanlan.zhihu.com/p/617679244)
- [世界首款真开源类ChatGPT大模型Dolly 2.0，可随意修改商用](https://zhuanlan.zhihu.com/p/621655147)
- [用ChatGPT训练羊驼：「白泽」开源，轻松构建专属模型，可在线试玩](https://zhuanlan.zhihu.com/p/619453625)
- [3090单卡5小时，每个人都能训练专属ChatGPT，港科大开源LMFlow](https://zhuanlan.zhihu.com/p/618919940)
- [300美元复刻ChatGPT九成功力，GPT-4亲自监考，130亿参数开源模型「小羊驼」来了](https://zhuanlan.zhihu.com/p/618699807)
- [学术专用版ChatGPT火了，一键完成论文润色、代码解释、报告生成](https://zhuanlan.zhihu.com/p/618310974)
- [笔记本就能运行的ChatGPT平替来了，附完整版技术报告](https://zhuanlan.zhihu.com/p/618310404)
- [训练个中文版ChatGPT没那么难：不用A100，开源Alpaca-LoRA+RTX 4090就能搞定](https://zhuanlan.zhihu.com/p/617221484)
- [弥补斯坦福70亿参数「羊驼」短板，精通中文的大模型来了，已开源](https://zhuanlan.zhihu.com/p/616079388)
- [还在为玩不了ChatGPT苦恼？这十几个开源平替也能体验智能对话](https://zhuanlan.zhihu.com/p/615257807)
- [斯坦福70亿参数开源模型媲美GPT-3.5，100美元即可复现](https://zhuanlan.zhihu.com/p/614212219)
- [真·ChatGPT平替：无需显卡，MacBook、树莓派就能运行LLaMA](https://zhuanlan.zhihu.com/p/613923687)
- [ChatGPT开源替代来了！参数量200亿，在4300万条指令上微调而成](https://zhuanlan.zhihu.com/p/613609788)
- [这是Meta版ChatGPT雏形？开源、一块GPU就能跑，1/10参数量打败GPT-3](https://zhuanlan.zhihu.com/p/609544219)
- [​B站UP主硬核自制智能音箱：有ChatGPT加持，才是真・智能](https://zhuanlan.zhihu.com/p/599602043)
- [ChatGPT 中文调教指南。各种场景使用指南。学习怎么让它听你的话](https://github.com/PlexPt/awesome-chatgpt-prompts-zh)
- [ChatGPT提示工程师｜AI大神吴恩达教你写提示词](https://www.bilibili.com/video/BV1No4y1t7Zn/?vd_source=4dffb0fbabed4311f4318e8c6d253a10)

### 大模型训练技术

- [千亿参数开源大模型 BLOOM 背后的技术](https://codeuuu.com/p/68022.html) 这里主要总结了并行相关的技术以及使用的一些坑，然后对训练过程中可能出现的问题进行描述。
- [在一张 24 GB 的消费级显卡上用 RLHF 微调 20B LLMs](https://www.cnblogs.com/huggingface/p/17245966.html)
- [人手一个ChatGPT！微软DeepSpeed Chat震撼发布，一键RLHF训练千亿级大模型](https://zhuanlan.zhihu.com/p/621379646)
- [大型语言模型(LLM)训练指南🚀](https://zhuanlan.zhihu.com/p/611325149)

### 大模型推理技术

- [CodeGeeX百亿参数大模型的调优笔记：比FasterTransformer更快的解决方案](https://zhuanlan.zhihu.com/p/617027615)
- [优化故事: BLOOM 模型推理](https://mp.weixin.qq.com/s/yzVqh4d6ynNROJxHycDUXg)
- [大型语言模型的推理演算](https://mp.weixin.qq.com/s/2wfUQNsH4IRuJEF39mebUQ)
- [简单读读WeightOnly](https://zhuanlan.zhihu.com/p/622334595)
- [[大模型技术祛魅]关于FlexGen的一点理解](https://zhuanlan.zhihu.com/p/610853654)
- [LLM Inference CookBook（持续更新）](https://zhuanlan.zhihu.com/p/619596323)
- [优化故事: BLOOM 模型推理](https://mp.weixin.qq.com/s/yzVqh4d6ynNROJxHycDUXg)

### 大模型数据标注

