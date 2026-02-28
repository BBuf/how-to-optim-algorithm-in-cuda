> 我的课程笔记，欢迎关注：https://github.com/BBuf/how-to-optim-algorithm-in-cuda 。

# 0x0. 前言

在SGLang参与了快要一年开源，学到了一些开发和Profile的技巧。这里记录一下SGLang 编译，开发和Profile的几个技巧，可以避免踩坑并且在开发过程中省时省力。

# 0x1. perfetto 可视化 PyTorch Profiler 结果出现 kernel 丢失情况

这是以前看Profile比较头疼的问题，下面分享下前因后果以及来自SGLang Tom神的解决方法。

以gpt-oss-120b为例，我们可以用下面的命令生成一个Torch Profiler文件，TensorRT-LLM也有类似的方式。

```shell
python3 -m sglang.launch_server --model openai/gpt-oss-120b --tp 8 --port 30000 --attention-backend trtllm_mha

 python3 -m sglang.bench_serving --model openai/gpt-oss-120b --dataset-name random --backend sglang-oai --random-range-ratio 1 --random-input-len 1200 --random-output-len 20 --max-concurrency 1 --num-prompts 5 --profile --port 30000
```

这样我们在/tmp就可以看到类似下面的profiler文件了。

![](https://files.mdnice.com/user/59/ad531656-b4b5-4d38-a37c-5538b197741d.png)

接着我们只需要把这个文件下载之后拖到perfetto(https://ui.perfetto.dev)里打开，然后我们就可以对着这个Profiler的cuda kernel执行序做优化了，无论是Kernel Fuse还是Kernel本身的加速都可以非常准确的反应在perfetto对Torch Profiler渲染出的kernel时间执行图上。

然后这里的有几个问题，如果不知道的话可能会陷入一些怀疑。

## 0x1.1 kernel之间出现空洞

我们挑一个gpt-oss-120b decode layer的kernel执行时间图来看： 

![](https://files.mdnice.com/user/59/f2210734-b6f5-40bf-8a00-d4f8d806770e.png)

我们发现开头和中间都有一些空洞，由于我们已经开启了cuda graph，这些明显的空洞按道理来说应该是cuda kernel才对，为什么这里是空洞呢？如果你没有对比过Nsight System结果并且对这段代码执行的kernel不熟悉的话，那么你很容易忽略这点，可能还以为是个奇怪的bug。

但是事实上，这些空洞其实对应的也是正常的cuda kernel，只是因为这些kernel因为PDL的原因和这个kernel的前一个kernel有部分重叠，导致perfetto的渲染出了bug，无法正常在一条Stream上的时间线上去显示所有的kernel，导致出现了空洞。

这个问题被SGLang 核心开发者 Tom(https://github.com/fzyzcjy) 发现并解决，解决的方案也很简单优雅，具体见：

https://github.com/fzyzcjy/torch_utils/blob/master/src/convert_to_perfetto_compatible/convert_to_perfetto_compatible.py

简单来说就是解析kernel的执行时间，把有重叠的kernel临时放到另外一个Stream上显示。只需要把我们上面得到的Profiler文件过一次这个脚本，我们就可以把空洞部分对应的kernel完整显示出来了。效果如下：

![](https://files.mdnice.com/user/59/53ac9cca-a507-44f9-81af-c1f362c59fba.png)

这个问题困扰了我很长的时间，2024年就发现过这个问题，有时候用Perfetto老版本的chrome插件偶尔又可以显示出所有kernel，弄清楚之后只能说Tom神yyds。大家可以到Tom神仓库获得这个脚本：https://github.com/fzyzcjy/torch_utils/blob/master/src/convert_to_perfetto_compatible/convert_to_perfetto_compatible.py

## 0x1.2 合并各个Rank的Profiler结果

如上面的截图所示，以TP8运行gpt-oss-120b，我们就可以看到8个Rank的Profiler结果。一般我们可以选择rank0的结果来看单个gpu上kernel执行的耗时和kernel laucnh等细节信息。但是如果我们要查看不同rank之间的交互操作例如多个rank之间做AllReduce，那么我们就可以选择合并各个Rank的Profiler结果清楚看到AllReduce的开始和结束等细节。

我们依然可以使用Tom提供的多Rank Profiler结果合并脚本(https://github.com/fzyzcjy/torch_utils/blob/master/src/torch_profile_trace_merger/sglang_profiler_trace_merger.py)过一下上面的Profiler文件，就可以得到合并后的Profiler结果。



# 0x2. PR 切换分支

https://github.com/yyihuang 前两天教我的。

![](https://files.mdnice.com/user/59/5bab643a-5a5f-45dd-a2c5-cdc2dba5d2c0.png)

如果我们要测试这个sglang的branch，但是这个branch是一个开发者的clone分支，我们如何快速切换到这个分支的代码。

可以点击右上角的code：

![](https://files.mdnice.com/user/59/11a037d4-1f1f-4ba9-8cb2-fdde32f3a91a.png)

需要装一下gh并和github账号关联，然后就可以用下面的命令切换到这个branch了。

```shell
gh pr checkout 9065
```

有了这种方式之后如果你要合并几个有关联的fork仓库的PR做测试也变得很容易了。

我已经傻乎乎的set upstream或者手动cherry-pick pr很多年了。。。真的吐血

# 0x3. SGLang 无视PyTorch版本和sgl-kernel的版本运行

这一点的有用之处在于，如果你碰到某个场景的某个模型在SGLang的版本变动下出现了性能bug或者其它的bug，你需要定位是哪个commit或者pr引入了这个bug。如果不源码编译SGLang几乎很难做这个事，然后下面是快速全流程编译的命令，基本可以保证任何commit都可以正常运行，从而让我们可以让们达到正确二分并定位的目的。

```
git clone --recursive git@github.com:flashinfer-ai/flashinfer.git
rm -rf  ~/.cache/flashinfer
python -m pip install -v . 

git clone git@github.com:sgl-project/sglang.git
cd sglang 
pip install -e "python[all]" 
cd sgl-kernel && make build -j8
pip install dist/sgl_xxx.whl --force-install
```

注意上面的`rm -rf  ~/.cache/flashinfer`这个步骤，如果bug和flashinfer存在关系，是必不可少的。

基本上按照这个步骤就可以完成切换到之前任意SGLang的 commit并顺利跑起来代码了。对了开发的整体开发环境还是推荐根据个人的GPU架构拉一下SGLang的Docker环境，然后在Docker环境里面完成上述的编译。


# 0x4. 结束

我好像前几天还想到了一些来着，但是现在忘了。暂时就记得这些了，以后想起来啥活着有新的再补一下。











