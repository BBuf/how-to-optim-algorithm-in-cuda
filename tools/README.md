## 日常工作中用到的 tools

### hfd.sh

从 Hugging Face 下载模型或数据集。

#### 依赖：

```
apt update
apt install -y aria2
```

#### 用法

```shell
./hfd.sh Qwen/Qwen2.5-7B-Instruct
./hfd.sh black-forest-labs/FLUX.1-dev  --hf_username BBuf --hf_token xxx
```

对于不需要授权的开源HF模型或者数据集，可以不指定`--hf_username`和`--hf_token`参数。否则必须指定获得了授权的`--hf_username`和`--hf_token`参数。

`--local-dir`参数指定下载的模型或数据集的本地存储路径。不指定就表示在当前脚本所在的路径进行下载。




