## 这里记录一些训练大语言模型需要用到的基础知识和技巧

- zero.md 记录一些通信原语和zero算法的原理。

## 博客和文章收集

### 大模型知识介绍

- [《A Survey of Large Language Models》笔记](https://zhuanlan.zhihu.com/p/631065995)
- [分析transformer模型的参数量、计算量、中间激活、KV cache](https://zhuanlan.zhihu.com/p/624740065)
- [Transformer模型的基础演算](https://mp.weixin.qq.com/s/0Er0UOk6Wdky-0gzeQxK0g)
- [Transformer 估算 101](https://zhuanlan.zhihu.com/p/630582034)
- [通向AGI之路：大型语言模型（LLM）技术精要](https://zhuanlan.zhihu.com/p/597586623)
- [Transformer学习笔记二：Self-Attention（自注意力机制）](https://zhuanlan.zhihu.com/p/455399791)
- [Transformer学习笔记三：为什么Transformer要用LayerNorm/Batch Normalization & Layer Normalization （批量&层标准化)](https://zhuanlan.zhihu.com/p/456863215)
- [Transformer学习笔记五：Subword Tokenization（子词分词器）](https://zhuanlan.zhihu.com/p/460678461)
- [ChatGPT技术解析系列之：GPT1、GPT2与GPT3](https://zhuanlan.zhihu.com/p/609367098)
- [ChatGPT技术解析系列之：训练框架InstructGPT](https://zhuanlan.zhihu.com/p/605516116)
- [ChatGPT技术解析系列之：赋予GPT写代码能力的Codex](https://zhuanlan.zhihu.com/p/611313567)
- [大模型推理性能优化之KV Cache解读](https://zhuanlan.zhihu.com/p/630832593)
- [拆解追溯 ChatGPT各项能力的起源](https://zhuanlan.zhihu.com/p/607469120)
- [ChatGPT 的突现能力，我们是否真的面临范式转变？](https://zhuanlan.zhihu.com/p/622052864)
- [复杂推理：大型语言模型的"北极星"能力](https://zhuanlan.zhihu.com/p/628855304)
- [深入理解NLP Subword算法：BPE、WordPiece、ULM](https://zhuanlan.zhihu.com/p/86965595)
- [ChatGPT 背后的“功臣”——RLHF 技术详解](https://www.cnblogs.com/huggingface/p/17040315.html)
- [深入浅出，解析ChatGPT背后的工作原理](https://zhuanlan.zhihu.com/p/597100830)
- [这是Meta版ChatGPT雏形？开源、一块GPU就能跑，1/10参数量打败GPT-3](https://zhuanlan.zhihu.com/p/609544219)
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
- [​B站UP主硬核自制智能音箱：有ChatGPT加持，才是真・智能](https://zhuanlan.zhihu.com/p/599602043)
- [熔岩羊驼LLaVA来了：像GPT-4一样可以看图聊天，无需邀请码，在线可玩](https://zhuanlan.zhihu.com/p/624442883)
- [3天近一万Star，无差体验GPT-4识图能力，MiniGPT-4看图聊天、还能草图建网站](https://zhuanlan.zhihu.com/p/623731818)
- [ChatGPT 中文调教指南。各种场景使用指南。学习怎么让它听你的话](https://github.com/PlexPt/awesome-chatgpt-prompts-zh)
- [ChatGPT提示工程师｜AI大神吴恩达教你写提示词](https://www.bilibili.com/video/BV1No4y1t7Zn/?vd_source=4dffb0fbabed4311f4318e8c6d253a10)
- [[分析] 浅谈ChatGPT的Tokenizer](https://zhuanlan.zhihu.com/p/626621158)
- [OPT-175B是如何炼成的](https://zhuanlan.zhihu.com/p/622061951)
- [Meta复刻GPT-3“背刺”OpenAI，完整模型权重及训练代码全公开](https://zhuanlan.zhihu.com/p/509100358)
- [Limitations of LLaMA](https://zhuanlan.zhihu.com/p/618776565)
- [Hugging News #0506: StarCoder, DeepFloyd/IF 好多新的重量级模型](https://zhuanlan.zhihu.com/p/627319332)
- [StarCoder: 最先进的代码大模型](https://zhuanlan.zhihu.com/p/627840388)
- [VideoChat🦜: 基于视频指令数据微调的聊天机器人](https://zhuanlan.zhihu.com/p/628712512)
- [MiniGPT-4 本地部署 RTX 3090](https://zhuanlan.zhihu.com/p/624417097)
- [更擅长推理的LLaMA大模型，支持中文！](https://zhuanlan.zhihu.com/p/628688680)
- [点击鼠标，让ChatGPT更懂视觉任务！](https://zhuanlan.zhihu.com/p/628266214)
- [[分析] ROPE的不同实现：llama&palm](https://zhuanlan.zhihu.com/p/627536105)
- [羊驼系列大模型和ChatGPT差多少？详细测评后，我沉默了](https://zhuanlan.zhihu.com/p/629085937)
- [【开源骆驼】更好的翻译prompt，中英文token比例，比alpaca更强的中文数据集WizardLM](https://zhuanlan.zhihu.com/p/629379775)
- [ImageBind: 表征大一统？也许还有一段距离](https://zhuanlan.zhihu.com/p/629389992)
- [训练开销骤减，10%成本定制专属类GPT-4多模态大模型](https://mp.weixin.qq.com/s/UqBEGLpF6H7NU9jyqbvRLg)
- [国内首个可复现的RLHF基准，北大团队开源 PKU-Beaver](https://mp.weixin.qq.com/s/O1RDHrmEg99zCil8ycqOGQ)
- [北大紧跟步伐开源PKU-Beaver (河狸)——不仅支持RLHF训练, 还开源RLHF训练数据](https://zhuanlan.zhihu.com/p/630326764)
- [大模型迎来「开源季」，盘点过去一个月那些开源的LLM和数据集](https://mp.weixin.qq.com/s/VleZkQT6Vga7vqZP8pvgQQ)
- [超越GPT-4！华人团队爆火InstructBLIP抢跑看图聊天，开源项目横扫多项SOTA](https://mp.weixin.qq.com/s/jI1cf7FDYJscHDZKiNvoug)
- [基于 ChatGLM-6B 搭建个人专属知识库](https://zhuanlan.zhihu.com/p/629558941)
- [大模型-LLM分布式训练框架总结](https://zhuanlan.zhihu.com/p/623746805)
- [没有RLHF，一样媲美GPT-4、Bard，Meta发布650亿参数语言模型LIMA](https://mp.weixin.qq.com/s/Oze93Brun-AQUBI5Tt1b6w)
- [在Transformer时代重塑RNN，RWKV将非Transformer架构扩展到数百亿参数](https://mp.weixin.qq.com/s/cg8F4cE6JGij7JJJivUqxg)
- [马腾宇团队新出大模型预训练优化器，比Adam快2倍，成本减半](https://mp.weixin.qq.com/s/L_66ZWTeLE43gQtSi1reEw)
- [跑分达ChatGPT的99%，人类难以分辨！开源「原驼」爆火，iPhone都能微调大模型了](https://mp.weixin.qq.com/s/1ZrPtBmgkklFk2_TvOhK_w)
- [大模型词表扩充必备工具SentencePiece](https://zhuanlan.zhihu.com/p/630696264)
- [RWKV – transformer 与 RNN 的强强联合](https://zhuanlan.zhihu.com/p/633735524)
- [Falcon 登陆 Hugging Face 生态](https://zhuanlan.zhihu.com/p/637676443)
- [详解大模型RLHF过程（配代码解读）](https://zhuanlan.zhihu.com/p/624589622)
- [详解Transformer-XL](https://zhuanlan.zhihu.com/p/271984518)
- [教科书级数据is all you need：1.3B小模型逆袭大模型的秘密](https://zhuanlan.zhihu.com/p/608004441)
- [清华第二代60亿参数ChatGLM2开源！中文榜居首，碾压GPT-4，推理提速42%](https://zhuanlan.zhihu.com/p/639888131)
- [NLP（十七）：从 FlashAttention 到 PagedAttention, 如何进一步优化 Attention 性能](https://zhuanlan.zhihu.com/p/638468472)
- [AGI最前沿：GPT-4之后大模型学术进展速览](https://zhuanlan.zhihu.com/p/639165892)
- [LLM学习记录（一）--关于大模型的一些知识](https://zhuanlan.zhihu.com/p/624918286)
- [UC伯克利LLM排行榜首次重磅更新！GPT-4稳居榜首，全新330亿参数「小羊驼」位列开源第一](https://zhuanlan.zhihu.com/p/607403006)
- [【Falcon Paper】我们是靠洗数据洗败 LLaMA 的！](https://zhuanlan.zhihu.com/p/637996787)

### 大模型训练技术

- [【LLM】从零开始训练大模型](https://zhuanlan.zhihu.com/p/636270877)
- [千亿参数开源大模型 BLOOM 背后的技术](https://codeuuu.com/p/68022.html) 这里主要总结了并行相关的技术以及使用的一些坑，然后对训练过程中可能出现的问题进行描述。
- [在一张 24 GB 的消费级显卡上用 RLHF 微调 20B LLMs](https://www.cnblogs.com/huggingface/p/17245966.html)
- [人手一个ChatGPT！微软DeepSpeed Chat震撼发布，一键RLHF训练千亿级大模型](https://zhuanlan.zhihu.com/p/621379646)
- [大型语言模型(LLM)训练指南🚀](https://zhuanlan.zhihu.com/p/611325149)
- [“StackLLaMA”: 用 RLHF 训练 LLaMA 的手把手教程](https://zhuanlan.zhihu.com/p/626896135)
- [图解大模型训练之：流水线并行（Pipeline Parallelism），以Gpipe为例](https://zhuanlan.zhihu.com/p/613196255)
- [图解大模型训练之：数据并行上篇(DP, DDP与ZeRO)](https://zhuanlan.zhihu.com/p/617133971)
- [图解大模型训练之：数据并行下篇( DeepSpeed ZeRO，零冗余优化)](https://zhuanlan.zhihu.com/p/618865052)
- [图解大模型训练之：张量模型并行(TP)，Megatron-LM](https://zhuanlan.zhihu.com/p/622212228)
- [图解大模型系列之：Megatron源码解读1，分布式环境初始化](https://zhuanlan.zhihu.com/p/629121480)
- [图解大模型训练之：Megatron源码解读2，模型并行](https://zhuanlan.zhihu.com/p/634377071)
- [深度学习里，模型并行中怎么将模型拆分？](https://www.zhihu.com/question/319355346/answer/2985459442)
- [Transformers DeepSpeed官方文档](https://zhuanlan.zhihu.com/p/621572871)
- [当红炸子鸡 LoRA，是当代微调 LLMs 的正确姿势？](https://zhuanlan.zhihu.com/p/618894919)
- [GLM、LLAMA用Accelerate+deepspeed做RLHF时可能遇到的问题](https://zhuanlan.zhihu.com/p/629614251)
- [GPT fine-tune实战： 训练我自己的 ChatGPT🚀🚀🚀](https://zhuanlan.zhihu.com/p/616504594)
- [DeepSpeed之ZeRO系列：将显存优化进行到底](https://zhuanlan.zhihu.com/p/513571706)
- [大模型也内卷，Vicuna训练及推理指南，效果碾压斯坦福羊驼](https://zhuanlan.zhihu.com/p/624012908)
- [一键式 RLHF 训练 DeepSpeed Chat（一）：理论篇](https://zhuanlan.zhihu.com/p/626159553)
- [使用DeepSpeed/P-Tuning v2对ChatGLM-6B进行微调](https://zhuanlan.zhihu.com/p/622351059)
- [从0到1基于ChatGLM-6B使用LoRA进行参数高效微调](https://zhuanlan.zhihu.com/p/621793987)
- [足够惊艳，使用Alpaca-Lora基于LLaMA(7B)二十分钟完成微调，效果比肩斯坦福羊驼](https://zhuanlan.zhihu.com/p/619426866)
- [基于LLaMA-7B/Bloomz-7B1-mt复现开源中文对话大模型BELLE及GPTQ量化](https://zhuanlan.zhihu.com/p/618876472)
- [从0到1复现斯坦福羊驼（Stanford Alpaca 7B）](https://zhuanlan.zhihu.com/p/618321077)
- [如何使用 Megatron-LM 训练语言模型](https://zhuanlan.zhihu.com/p/633160974)
- [[源码解析] 模型并行分布式训练Megatron (1) --- 论文&基础 ](https://juejin.cn/post/7057837676430360584)
- [[源码解析] 模型并行分布式训练Megatron (2) --- 整体架构 ](https://juejin.cn/post/7061942798957674504)
- [[源码解析] 模型并行分布式训练 Megatron (3) ---模型并行实现 ](https://juejin.cn/post/7062256365636419592)
- [[源码解析] 模型并行分布式训练 Megatron (4) --- 如何设置各种并行 ](https://juejin.cn/post/7063030243224879140)
- [[源码解析] 模型并行分布式训练Megatron (5) --Pipedream Flush ](https://juejin.cn/post/7064496967828635655)
- [Pytorch Distributed Data Parallal](https://fazzie-key.cool/2022/01/23/ddp/)
- [大模型参数高效微调技术原理综述（七）-最佳实践、总结](https://zhuanlan.zhihu.com/p/636999010)
- [【万字长文】LLaMA, ChatGLM, BLOOM的参数高效微调实践](https://zhuanlan.zhihu.com/p/635710004)
- [CPT：兼顾理解和生成的中文预训练模型](https://zhuanlan.zhihu.com/p/421402341)
- [大模型流水线并行（Pipeline）实战](https://zhuanlan.zhihu.com/p/636488690)
- [QLoRA：4-bit级别的量化+LoRA方法，用3090在DB-GPT上打造基于33B LLM的个人知识库](https://zhuanlan.zhihu.com/p/634516004)
- [大模型高效微调综述上：Adapter Tuning、AdaMix、PET、Prefix-Tuning、Prompt Tuning、P-tuning、P-tuning v2](https://zhuanlan.zhihu.com/p/638809556)
- [大模型高效微调综述下： DiffPruning、BitFit、LoRa、AdaLoRA、MAM Adapters、UniPELT](https://zhuanlan.zhihu.com/p/639068809)

### 大模型推理技术

- [大幅优化推理过程，字节高性能Transformer推理库获IPDPS 2023最佳论文奖 ](https://mp.weixin.qq.com/s/5TM4PXTUBZuOfZlltFfrsQ)
- [CodeGeeX百亿参数大模型的调优笔记：比FasterTransformer更快的解决方案](https://zhuanlan.zhihu.com/p/617027615)
- [优化故事: BLOOM 模型推理](https://mp.weixin.qq.com/s/yzVqh4d6ynNROJxHycDUXg)
- [大型语言模型的推理演算](https://mp.weixin.qq.com/s/2wfUQNsH4IRuJEF39mebUQ)
- [简单读读WeightOnly](https://zhuanlan.zhihu.com/p/622334595)
- [[大模型技术祛魅]关于FlexGen的一点理解](https://zhuanlan.zhihu.com/p/610853654)
- [LLM Inference CookBook（持续更新）](https://zhuanlan.zhihu.com/p/619596323)
- [优化故事: BLOOM 模型推理](https://mp.weixin.qq.com/s/yzVqh4d6ynNROJxHycDUXg)
- [GPTQ-for-LLaMa 量化分析和优化](https://zhuanlan.zhihu.com/p/625701227)
- [Web-LLM:机器学习编译纯浏览器运行大模型](https://zhuanlan.zhihu.com/p/622271247)
- [陈天奇等人新作引爆AI界：手机原生跑大模型，算力不是问题了](https://mp.weixin.qq.com/s/uQGAu1v-6ApgZHVkZJsUdQ)
- [NLP（十一）：大语言模型的模型量化(INT8)技术](https://zhuanlan.zhihu.com/p/627436535)
- [大(语言)模型推理原理及加速](https://zhuanlan.zhihu.com/p/628511161)
- [AI算力碎片化：矩阵乘法的启示](https://zhuanlan.zhihu.com/p/624425308)
- [大大大模型部署方案抛砖引玉](https://mp.weixin.qq.com/s/e6ymQZs5MY1pdodC7eg8iQ)
- [BELLE(LLaMA-7B/Bloomz-7B1-mt)大模型使用GPTQ量化后推理性能测试](https://zhuanlan.zhihu.com/p/621128368)
- [大模型的好伙伴，浅析推理加速引擎FasterTransformer](https://zhuanlan.zhihu.com/p/626008090)
- [模型推理服务化框架Triton保姆式教程（一）：快速入门](https://zhuanlan.zhihu.com/p/629336492)
- [模型推理服务化框架Triton保姆式教程（二）：架构解析](https://zhuanlan.zhihu.com/p/634143650)
- [模型推理服务化框架Triton保姆式教程（三）：开发实践](https://zhuanlan.zhihu.com/p/634444666)
- [【自然语言处理】【大模型】大语言模型BLOOM推理工具测试]https://zhuanlan.zhihu.com/p/608004441)
- [使用bitsandbytes、4 位量化和 QLoRA 使 LLM 更易于访问](https://zhuanlan.zhihu.com/p/632287465)

### 大模型仓库收集

- [Awesome-Chinese-LLM：收集和梳理中文LLM相关的开源模型、应用、数据集及教程等资料](https://mp.weixin.qq.com/s/Oy6XZNyN3kpsC6TfQYQb7A)
- LLaMA:
    - Paper: https://research.facebook.com/publications/llama-open-and-efficient-foundation-language-models/
    - Github: https://github.com/facebookresearch/llama
- Baize (白泽)
    - Paper: https://arxiv.org/pdf/2304.01196.pdf
    - Demo: https://huggingface.co/spaces/project-baize/Baize-7B
    - Repo: https://github.com/project-baize/baize-chatbot
- Luotuo (骆驼，Chinese)
    - Blog: https://zhuanlan.zhihu.com/p/617221484 后半部分
    - Repo: https://github.com/LC1332/Luotuo-Chinese-LLM
- Koala (考拉)
    - Blog: https://bair.berkeley.edu/blog/2023/04/03/koala/
    - Demo: https://chat.lmsys.org/?model=koala-13b
    - Repo: https://github.com/young-geng/EasyLM
- Wombat（袋熊）
    - Paper: https://arxiv.org/pdf/2304.05302.pdf
    - Repo: https://github.com/GanjinZero/RRHF
- GPT4All
    - Report: https://s3.amazonaws.com/static.nomic.ai/gpt4all/2023_GPT4All_Technical_Report.pdf
    - Repo: https://github.com/nomic-ai/gpt4all
- LMFlow
    - Demo: https://lmflow.com/
    - Repo: https://github.com/OptimalScale/LMFlow
- Firefly(流萤)
    - Repo: https://github.com/yangjianxin1/Firefly
- Vicuna(小羊驼)
    - Blog: https://zhuanlan.zhihu.com/p/618699807
    - Repo: https://github.com/lm-sys/FastChat
- LLaVA
    - Paper: https://arxiv.org/pdf/2304.08485.pdf
    - Demo: https://llava-vl.github.io/

### 大模型数据标注

- [详谈大模型训练中的数据收集、处理与模型影响：A Survey of Large Language Models工作中的数据总结](https://mp.weixin.qq.com/s/bHsb631KA5AaulBHNT5m9w)

### 李沐论文精度文字版专栏

- [李沐论文精度文字版专栏](https://www.zhihu.com/column/c_1656053216138719233)

