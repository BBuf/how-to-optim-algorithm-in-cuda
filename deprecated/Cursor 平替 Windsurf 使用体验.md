## 前言

![WindSurf 账户类似于github记录commit的方式记录你的auto complete等](https://files.mdnice.com/user/59/69dcec71-7e8a-4495-a583-49816ed03716.png)

Cursor似乎最近把国内卡支付的Pro账户ban掉了很多，我也中招了。然后最近大概使用了一周WindSurf，个人感觉和Cursor的体验几乎一致。只不过有一些功能例如Composer和Cursor的打开方式不太一样，我这里来记录一下WindSurf的使用体验以及一些操作细节。

Windsurf由Codeium团队开发，和Cursor类似都是可以直接使用VSCode的设置。它最近发布了Wave 3模式，引入了意图识别，新增Turbo Mode 2模式可以自行执行终端命令，自我修正等，并支持了更多更强大的推理模型比如DeepSeek-R1，并且目前每个模型的积分使用（或者说token消耗）都是透明化的。

![每个模型对应的积分](https://files.mdnice.com/user/59/510fe19c-5a9e-4470-82c1-e0130ac3bdc8.png)

Windsurf支持o3-mini，Claude 3.5 Sonnet，GTP-4o，DeepSeek V3/R1，Gemini 2.0 Flash等等主流模型，多模态例如识别图片也是支持的，可以直接粘贴图片。并且使用的时候Tab自动生成内容比Cursor更快，可能是因为Cursor的用户量比它大很多的原因？最后它的价格比Cursor还便宜一半也就是10美元一个月，并且还支持支付宝直接支付，对国内的用户更友好。

![价格如图](https://files.mdnice.com/user/59/3d8b3713-71ed-4145-86fe-5acfd1eb0a86.png)

下面展示一些平替技巧和比Cursor更惊艳的新功能体验。

## 一些平替技巧

- 回复语言

首先是语言方面，如果你直接使用WindSurf，即使你问的是中文它的回复也有一定的概率是英语，而Cursor中可以通过设置system prompt让他返回中文。WindSurf也可以的，在这里可以找到：

![设置system prompt入口](https://files.mdnice.com/user/59/35a8e8e7-ad1e-4f89-9aa9-aac81e2d28cf.png)

打开之后写一句用中文回复之后就是全程中文了：

![可以分点设置系统 prompt](https://files.mdnice.com/user/59/2f3899fb-273a-4f9c-86ed-96040b281125.png)

设置之后的效果

![现在即使用英语问问题，也可以保证模型回复中文](https://files.mdnice.com/user/59/7de6665e-a9e9-4d5d-90bb-a4d08a9033f8.png)

如果你想设置其它的全局规则也是可以的。

- Cursor COMPOSER的代替

![Cascade 模式可以插入各种类型的上下文，对标Cursor composer](https://files.mdnice.com/user/59/7a425633-b3a0-4df3-9fcd-abb2c8c1a653.png)

WindSurf提供Cascade 模式，并且分成编辑和聊天两种模式，编辑模式可以生成和修改代码，直接将生成的代码写入到对应的文件（会先展示Diff，也支持让用户逐个查看和接受部分修改），聊天模式专注于提供开发建议、解答代码问题。此外，在编辑模式下通过@符号可以启动加入文件，文件夹，文档，代码片，和网络链接等作为上下文帮助更好的完成任务。这个和Cursor的功能几乎一致：

![Cursor composer支持插入的上下文类型](https://files.mdnice.com/user/59/6ccbf7f0-2e51-4236-9397-2a8ddda48a1e.png)

- 更快的Auto Complete和直接文本编辑

在当前文件中选中文本之后，可以针对当前文本直接调用Chat或者编辑的功能，并且在编辑代码的过程中支持使用Tab自动补全生成的代码，和Cursor的功能完全一样。

![支持选中文本之后做指令编辑和对话](https://files.mdnice.com/user/59/42f6efb2-6abe-4a00-8efe-9581d267e9e7.png)

AutoComplete(Tab) 支持低，中，高三个速度档位

![Tab 支持低，中，高三个速度档位](https://files.mdnice.com/user/59/a08654a7-238b-43e7-b78e-1abdbdd9b7f4.png)



## 更惊艳的功能

下面介绍的几项功能是Cursor还没有，但是我个人使用起来感觉比较惊艳。

### Tab键可以预判意图

在最新的 Windsurf Wave 3 版本中，Tab键有更强大的功能，也就是「Tab to Jump」功能，本质上是预判开发者的意图。

![在Table to Jump上面还有一个Supercomplete可以根据你的编辑轨迹来提供更智能的代码补全和编辑建议。](https://files.mdnice.com/user/59/323f7454-374d-4cef-a8d5-47cb00559e1b.png)

![预判你的下一个修改动作发生在代码的哪个位置](https://files.mdnice.com/user/59/4f4a3fd1-8a4a-4753-a1d5-1d9847effcb8.gif)

WindSurf通过分析上下文，预判开发者下一步需要修改的代码位置，只需按下 Tab 键即可瞬间跳转到下个位置。这个感觉和意图识别算法有关系。

### Turbo Mode 2.0

![WindSurf Turbo Mode 2.0功能演示](https://files.mdnice.com/user/59/9ae9be14-911a-417b-815f-016bad1148f3.gif)

这个功能可以自动执行终端命令（除非在拒绝列表）例如自动上传提交代码，自动运行和debug当前的代码，也支持命令执行前后双阶段审查。我个人体验了这个功能，在比较独立的代码修改上是非常惊艳的，但是使用这个功能做git和文件操作之类的得小心，可以把执行命令的审查给加一下，并且这个模式也是比较消耗Token的。

![这里控制 Turbo Mode 要不要自己执行命令](https://files.mdnice.com/user/59/39886400-4a0e-4c1b-b860-eaf8d3283e9f.png)

### 自定义图标&自定义工具链

![](https://files.mdnice.com/user/59/64daf7a8-f680-4bbe-8ba8-627edfef153a.png)

WindSurf还支持自动切换系统图标，每个图标对应不同的神经认知模式，感兴趣可以自行体会。

WindSurf还有一个MCP的功能，让开发者配置私有 MCP 服务器，达到 Cascade 对话直接调用自定义工具链的目的。其中每个 MCP 工具调用消耗1个流程动作积分，支持用户自定义安全策略，协议层实现工具调用状态追踪。下面有一个展示例子，我目前还没用这个功能。如果是开发比较重复的工作，感觉会比较有用。


![](https://files.mdnice.com/user/59/c0b55bd2-db3d-4501-9046-6f916e8a4259.gif)


## 总结

我个人使用的体验来看，WindSurf几乎可以达到平替Cursor的目的，基本上日常使用到的产品功能可以保持一致，关键是它比CurSor更便宜。




