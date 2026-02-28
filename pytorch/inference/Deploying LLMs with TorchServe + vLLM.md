> 博客来源：https://pytorch.org/blog/deploying-llms-torchserve-vllm/

# 使用 TorchServe + vLLM 部署大语言模型

> by Matthias Reso, Ankith Gunapal, Simon Mo, Li Ning, Hamid Shojanazeri

vLLM 引擎目前是执行大语言模型(LLM)的最高性能方式之一。它提供了 vllm serve(https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) 命令作为在单机上部署模型的简单选项。虽然这很方便，但要在生产环境中大规模部署这些 LLM，还需要一些高级功能。

![](https://files.mdnice.com/user/59/aeb843b9-b1a5-4652-b1c4-c9687cdcdf98.png)

TorchServe 提供了这些基本的生产功能(如自定义指标和模型版本控制)，并通过其灵活的自定义处理程序设计，使得集成检索增强生成(RAG)或 Llama Guard(https://ai.meta.com/research/publications/llama-guard-llm-based-input-output-safeguard-for-human-ai-conversations/) 等安全保护功能变得非常容易。因此，将 vLLM 引擎与 TorchServe 配对来创建一个功能完备的生产级 LLM 服务解决方案是很自然的选择。

在深入介绍集成细节之前，我们将演示如何使用 TorchServe 的 vLLM docker 镜像来部署 Llama-3.1-70B-Instruct 模型。

## 在 TorchServe + vLLM 上快速开始使用 Llama 3.1

首先我们需要通过检出 TorchServe 代码仓库(https://github.com/pytorch/serve) 并从主文件夹执行以下命令来构建新的 TS LLM Docker(https://github.com/pytorch/serve/tree/master/docker) 容器镜像：

```shell
docker build --pull . -f docker/Dockerfile.vllm -t ts/vllm
```

容器使用我们新的 LLM 启动脚本 `ts.llm_launcher`，该脚本接收一个 Hugging Face 模型 URI 或本地文件夹作为输入，并启动一个本地 TorchServe 实例，后端运行 vLLM 引擎。要在本地部署模型，你可以使用以下命令创建一个容器实例：

```shell
#export token=<HUGGINGFACE_HUB_TOKEN>
docker run --rm -ti --shm-size 10g --gpus all -e HUGGING_FACE_HUB_TOKEN=$token -p 
8080:8080 -v data:/data ts/vllm --model_id meta-llama/Meta-Llama-3.1-70B-Instruct --disable_token_auth
```

你可以使用以下 curl 命令在本地测试端点:

```shell
curl -X POST -d '{"model":"meta-llama/Meta-Llama-3.1-70B-Instruct", "prompt":"Hello, my name is", "max_tokens": 200}' --header "Content-Type: application/json" "http://localhost:8080/predictions/model/1.0/v1/completions"
```

docker 将模型权重存储在本地文件夹 "data" 中，该文件夹在容器内被挂载为 /data。要使用你的自定义本地权重，只需将它们复制到 data 中，并将 model_id 指向 /data/<你的权重>。

在内部，容器使用我们新的 ts.llm_launcher 脚本来启动 TorchServe 并部署模型。该启动器将使用 TorchServe 部署 LLM 简化为单个命令行，也可以在容器外部用作高效的实验和测试工具。要在 docker 外部使用启动器，请按照 TorchServe 安装步骤(https://github.com/pytorch/serve?tab=readme-ov-file#-quick-start-with-torchserve)操作，然后执行以下命令来启动一个 8B Llama 模型：


```shell
# after installing TorchServe and vLLM run
python -m ts.llm_launcher --model_id meta-llama/Meta-Llama-3.1-8B-Instruct  --disable_token_auth
```

如果有多个 GPU 可用,启动器将自动使用所有可见设备并应用张量并行(参见 `CUDA_VISIBLE_DEVICES` 来指定使用哪些 GPU)。

虽然这很方便,但需要注意的是,它并不包含 TorchServe 提供的所有功能。对于那些想要利用更高级功能的用户,需要创建一个模型存档。虽然这个过程比执行单个命令要复杂一些,但它具有自定义处理程序和版本控制的优势。前者允许在预处理步骤中实现 RAG,后者让你可以在大规模部署之前测试处理程序和模型的不同版本。

在提供创建和部署模型存档的详细步骤之前,让我们深入了解 vLLM 引擎集成的细节。

## TorchServe 的 vLLM 引擎集成

作为最先进的服务框架,vLLM 提供了大量高级功能,包括 PagedAttention、连续批处理、通过 CUDA graphs 实现快速模型执行,以及支持各种量化方法,如 GPTQ、AWQ、INT4、INT8 和 FP8。它还为重要的参数高效适配器方法(如 LoRA)提供集成,并可以访问包括 Llama 和 Mistral 在内的广泛模型架构。vLLM 由 vLLM 团队和一个蓬勃发展的开源社区维护。

为了便于快速部署,它提供了基于 FastAPI 的服务模式来通过 HTTP 提供 LLM 服务。为了实现更紧密、更灵活的集成,该项目还提供了 vllm.LLMEngine(https://docs.vllm.ai/en/latest/dev/engine/llm_engine.html),它提供了持续处理请求的接口。我们利用异步变体(https://docs.vllm.ai/en/latest/dev/engine/async_llm_engine.html)将其集成到 TorchServe 中。

TorchServe(https://pytorch.org/serve/) 是一个易于使用的开源解决方案,用于在生产环境中部署 PyTorch 模型。作为经过生产测试的服务解决方案,TorchServe 为大规模部署 PyTorch 模型提供了众多好处和功能。通过将其与 vLLM 引擎的推理性能相结合,这些好处现在也可以用于大规模部署 LLM。

![](https://files.mdnice.com/user/59/36176266-28d1-44a2-a7a2-7951d9b7c86b.png)

为了最大化硬件利用率，通常将来自多个用户的请求批量处理是一个很好的做法。从历史上看，TorchServe 只提供了一种同步模式来收集来自不同用户的请求。在这种模式下，TorchServe 会等待预定义的时间(例如 batch_delay=200ms)或直到收集到足够的请求(例如 batch_size=8)。当触发其中一个事件时，批处理数据会被转发到后端，模型会对批处理进行处理，然后通过前端将模型输出返回给用户。这种方式对于传统的视觉模型特别有效，因为每个请求的输出通常会同时完成。

对于生成式用例，特别是文本生成，假设请求同时准备好的前提不再有效，因为响应的长度会有所不同。虽然 TorchServe 支持连续批处理(动态添加和删除请求的能力)，但这种模式只能适应静态的最大批处理大小。随着 PagedAttention 的引入，即使这种最大批处理大小的假设也变得更加灵活，因为 vLLM 可以以高度适应的方式组合不同长度的请求来优化内存利用率。

为了实现最佳内存利用率，即填充内存中未使用的空隙(想象一下俄罗斯方块)，vLLM 需要完全控制在任何给定时间处理哪些请求的决定。为了提供这种灵活性，我们不得不重新评估 TorchServe 处理用户请求的方式。我们引入了异步模式(https://github.com/pytorch/serve/blob/ba8c268fe09cb9396749a9ae5d480ba252764d71/examples/large_models/vllm/llama3/model-config.yaml#L7)(见下图)来替代之前的同步处理模式，在这种模式下，传入的请求直接转发到后端，使其可供 vLLM 使用。后端为 vllm.AsyncEngine 提供数据，现在可以从所有可用请求中进行选择。如果启用了流式模式并且请求的第一个 token 可用，后端将立即发送结果，并继续发送 token，直到生成最后一个 token。

![](https://files.mdnice.com/user/59/c1176a25-a8af-4e7b-bf8c-9b134f7f6209.png)
我们的 VLLMHandler 实现(https://github.com/pytorch/serve/blob/master/ts/torch_handler/vllm_handler.py)使用户能够通过配置文件快速部署任何与 vLLM 兼容的模型，同时通过自定义处理程序提供相同级别的灵活性和可定制性。用户可以通过继承 VLLMHandler 并重写相应的类方法来添加自定义预处理(https://github.com/pytorch/serve/blob/ba8c268fe09cb9396749a9ae5d480ba252764d71/ts/torch_handler/vllm_handler.py#L108)或后处理(https://github.com/pytorch/serve/blob/ba8c268fe09cb9396749a9ae5d480ba252764d71/ts/torch_handler/vllm_handler.py#L160)步骤。

我们还支持单节点、多 GPU 分布式推理(https://github.com/pytorch/serve/blob/master/examples/large_models/vllm/Readme.md#distributed-inference)，在这里我们配置 vLLM 使用张量并行分片模型，以增加较小模型的容量或启用不适合单个 GPU 的较大模型，如 70B Llama 变体。以前，TorchServe 只支持使用 torchrun 进行分布式推理，其中会启动多个后端工作进程来分片模型。vLLM 在内部管理这些进程的创建，因此我们为 TorchServe 引入了新的"custom"并行类型(https://github.com/pytorch/serve/blob/master/examples/large_models/vllm/Readme.md#distributed-inference)，它启动单个后端工作进程并提供分配的 GPU 列表。然后后端进程可以根据需要启动自己的子进程。

为了便于将 TorchServe + vLLM 集成到基于 docker 的部署中，我们提供了一个单独的 Dockerfile(https://github.com/pytorch/serve?tab=readme-ov-file#-quick-start-llm-deployment-with-docker)，它基于 TorchServe 的 GPU docker 镜像(https://hub.docker.com/r/pytorch/torchserve)，并添加了 vLLM 作为依赖项。我们选择将两者分开，以避免增加非 LLM 部署的 docker 镜像大小。

接下来，我们将演示在具有四个 GPU 的机器上使用 TorchServe + vLLM 部署 Llama 3.1 70B 模型所需的步骤。

## Step-by-Step Guide

For this step-by-step guide we assume the installation of TorchServe(https://github.com/pytorch/serve/tree/master?tab=readme-ov-file#-quick-start-with-torchserve) has finished successfully. Currently, vLLM is not a hard-dependency for TorchServe so let’s install the package using pip:

```shell
pip install -U vllm==0.6.1.post2
```

在接下来的步骤中，我们将(可选)下载模型权重、解释配置、创建模型存档、部署和测试它:

### 1. (可选)下载模型权重

这一步是可选的，因为 vLLM 也可以在模型服务器启动时处理权重的下载。但是，预先下载模型权重并在 TorchServe 实例之间共享缓存文件在存储使用和模型工作器启动时间方面是有益的。如果你选择下载权重，请使用 huggingface-cli 并执行:

```shell
# make sure you have logged into huggingface with huggingface-cli login before
# and have your access request for the Llama 3.1 model weights approved

huggingface-cli download meta-llama/Meta-Llama-3.1-70B-Instruct --exclude original/*
```
这将在 $HF_HOME 下下载文件，如果你想将文件放在其他地方，可以更改这个变量。请确保在运行 TorchServe 的任何地方都更新这个变量，并确保它可以访问该文件夹。

### 2. 配置模型

接下来，我们创建一个 YAML 配置文件，其中包含部署模型所需的所有参数。配置文件的第一部分指定了前端应该如何启动后端工作进程，该进程最终将在处理程序中运行模型。第二部分包括后端处理程序的参数，如要加载的模型，以及 vLLM 本身的各种参数。有关 vLLM 引擎可能的配置的更多信息，请参考此链接(https://docs.vllm.ai/en/latest/models/engine_args.html#engine-args)。

```shell
echo '
# TorchServe frontend parameters
minWorkers: 1            
maxWorkers: 1            # Set the number of worker to create a single model instance
startupTimeout: 1200     # (in seconds) Give the worker time to load the model weights
deviceType: "gpu" 
asyncCommunication: true # This ensures we can cummunicate asynchronously with the worker
parallelType: "custom"   # This lets TS create a single backend prosses assigning 4 GPUs
parallelLevel: 4

# Handler parameters
handler:
    # model_path can be a model identifier for Hugging Face hub or a local path
    model_path: "meta-llama/Meta-Llama-3.1-70B-Instruct"
    vllm_engine_config:  # vLLM configuration which gets fed into AsyncVLLMEngine
        max_num_seqs: 16
        max_model_len: 512
        tensor_parallel_size: 4
        served_model_name:
            - "meta-llama/Meta-Llama-3.1-70B-Instruct"
            - "llama3"
'> model_config.yaml
```

### 3. 创建模型文件夹

在创建模型配置文件(model_config.yaml)之后，我们现在将创建一个包含配置和其他元数据(如版本信息)的模型存档。由于模型权重很大，我们不会将它们包含在存档中。相反，处理程序将按照模型配置中指定的 model_path 访问权重。请注意，在这个例子中，我们选择使用"no-archive"格式，它会创建一个包含所有必要文件的模型文件夹。这使我们可以轻松修改配置文件进行实验，没有任何阻碍。之后，我们也可以选择 mar 或 tgz 格式来创建一个更容易传输的工件。

```shell
mkdir model_store
torch-model-archiver --model-name vllm --version 1.0 --handler vllm_handler --config-file model_config.yaml --archive-format no-archive --export-path model_store/
```

### 4. 部署模型

下一步是启动 TorchServe 实例并加载模型。请注意，我们已经禁用了本地测试的令牌认证。强烈建议在公开部署任何模型时实施某种形式的认证。

要启动 TorchServe 实例并加载模型，请运行以下命令:

```shell
torchserve --start --ncs  --model-store model_store --models vllm --disable-token-auth
```

你可以通过日志语句监控模型加载的进度。一旦模型加载完成，你就可以继续测试部署了。

### 5. Test the Deployment

The vLLM integration uses an OpenAI API compatible format so we can either use a specialized tool for this purpose or curl. The JSON data we are using here includes the model identifier as well as the prompt text. Other options and their default values can be found in the vLLMEngine docs(https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html).

```shell
echo '{
  "model": "llama3",
  "prompt": "A robot may not injure a human being",
  "stream": 0
}' | curl --header "Content-Type: application/json"   --request POST --data-binary @-   http://localhost:8080/predictions/vllm/1.0/v1/completions
```

请求的输出看起来像这样:

```shell
{
  "id": "cmpl-cd29f1d8aa0b48aebcbff4b559a0c783",
  "object": "text_completion",
  "created": 1727211972,
  "model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
  "choices": [
    {
      "index": 0,
      "text": " or, through inaction, allow a human being to come to harm.\nA",
      "logprobs": null,
      "finish_reason": "length",
      "stop_reason": null,
      "prompt_logprobs": null
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "total_tokens": 26,
    "completion_tokens": 16
  }
```
当 streaming 为 False 时,TorchServe 会收集完整的答案,并在最后一个 token 生成后一次性发送。如果我们将 stream 参数翻转,我们将收到分段数据,每条消息包含一个 token。

## 结论

在这篇博客中,我们探索了 vLLM 推理引擎与 TorchServe 的新的原生集成。我们演示了如何使用 ts.llm_launcher 脚本在本地部署 Llama 3.1 70B 模型,以及如何创建模型存档以部署在任何 TorchServe 实例上。此外,我们还讨论了如何在 Docker 容器中构建和运行解决方案,以便部署在 Kubernetes 或 EKS 上。在未来的工作中,我们计划启用 vLLM 和 TorchServe 的多节点推理,并提供预构建的 Docker 镜像以简化部署过程。

我们要感谢 Mark Saroufim 和 vLLM 团队在这篇博客发布前提供的宝贵支持。










