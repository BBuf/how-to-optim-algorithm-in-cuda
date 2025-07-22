> 原始文档：https://github.com/vllm-project/vllm/blob/main/docs/design/v1/p2p_nccl_connector.md

基于点对点通信的xPyD动态扩展实现，部分灵感来源于Dynamo。

# 详细设计

## 整体流程
如图1所示，这个**PD解耦**解决方案的整体流程通过请求流来描述：

1. 客户端向Proxy/Router的`/v1/completions`接口发送HTTP请求。
2. Proxy/Router通过轮询或随机选择的方式选择一个**1P1D（1个Prefill实例 + 1个Decode实例）**，生成一个`request_id`（规则将在后面介绍），将HTTP请求消息中的`max_tokens`修改为**1**，然后将请求转发给**P实例**。
3. 紧接着，Proxy/Router将**原始HTTP请求**转发给**D实例**。
4. **P实例**执行**Prefill**，然后**主动将生成的KV缓存**发送给D实例（使用**PUT_ASYNC**模式）。D实例的`zmq_addr`可以通过`request_id`解析得到。
5. **D实例**有一个**专用线程**用于接收KV缓存（避免阻塞主进程）。接收到的KV缓存保存到**GPU内存缓冲区**中，其大小由vLLM启动参数`kv_buffer_size`决定。当GPU缓冲区满时，KV缓存存储在**本地Tensor内存池**中。
6. 在**Decode**过程中，D实例的主进程从**GPU缓冲区**或**内存池**中检索KV缓存（由P实例传输），从而**跳过Prefill**。
7. 完成**Decode**后，D实例将结果返回给**Proxy/Router**，然后转发给**客户端**。

![image1](https://github.com/user-attachments/assets/fb01bde6-755b-49f7-ad45-48a94b1e10a7)

## Proxy/Router（演示）

一个简单的HTTP服务作为客户端请求的入口点，并启动一个后台线程来监听P/D实例报告其HTTP IP和PORT，以及ZMQ IP和PORT。它维护一个`http_addr -> zmq_addr`的字典。`http_addr`是vLLM实例请求的IP:PORT，而`zmq_addr`是KV缓存握手和元数据接收的地址。

Proxy/Router负责根据客户端请求的特征（如prompt）选择1P1D，并生成相应的`request_id`，例如：

```
cmpl-___prefill_addr_10.0.1.2:21001___decode_addr_10.0.1.3:22001_93923d63113b4b338973f24d19d4bf11-0
```

目前，为了快速验证xPyD是否能工作，使用轮询方式选择1P1D。未来计划使用trie结合实例的负载状态来选择合适的P和D。

每个P/D实例定期向Proxy/Router发送心跳包（目前每3秒一次）进行注册（即报告`http_addr -> zmq_addr`）并保持连接活跃。如果实例崩溃且在一定时间内无法发送ping，Proxy/Router将移除超时的实例（此功能尚未开发）。

## KV缓存传输方法

KV缓存传输有三种方法：PUT、GET和PUT_ASYNC。这些方法可以通过`--kv-transfer-config`和`kv_connector_extra_config`参数指定，具体通过`send_type`字段。PUT和PUT_ASYNC都涉及P实例主动向D实例发送KV缓存。区别在于PUT是同步传输方法，会阻塞主进程，而PUT_ASYNC是异步传输方法。PUT_ASYNC使用专用线程发送KV缓存，这意味着它不会阻塞主进程。相比之下，GET方法涉及P实例在计算prefill后将KV缓存保存到内存缓冲区。D实例在为KV缓存分配空间后，主动从P实例检索计算好的KV缓存。

实验结果表明，这些方法的性能从高到低依次为：PUT_ASYNC → GET → PUT。

## 通过ZMQ和NCCL进行P2P通信

只要知道对方的地址，就可以执行点对点KV缓存传输（使用NCCL），不受rank和world size的限制。支持PD解耦实例的动态扩展（扩展和收缩）。这意味着添加或删除P/D实例不需要完全重启系统。

每个P/D实例只需要创建一个`P2pNcclEngine`实例。该实例维护一个ZMQ服务器，运行专用线程监听`zmq_addr`地址并接收来自其他实例的控制流请求。这些请求包括建立NCCL连接的请求和发送KV缓存元数据（如张量形状和数据类型）的请求。但是，它实际上不传输KV缓存数据本身。

当P实例和D实例首次传输KV缓存时，它们需要建立ZMQ连接和NCCL组。对于后续的KV缓存传输，这个ZMQ连接和NCCL组会被重用。NCCL组只包含两个rank，意味着world size等于2。这种设计旨在支持动态扩展，这意味着添加或删除P/D实例不需要完全重启系统。只要知道对方的地址，就可以执行点对点KV缓存传输，不受rank或world size的限制。

## NCCL组拓扑

目前，KV缓存传输仅支持对称TP（张量并行）方法。非对称TP和PP（流水线并行）方法将在未来支持。图2展示了1P2D设置，其中每个实例的TP（张量并行）度为2。总共有7个NCCL组：三个vLLM实例各有一个TP=2的NCCL组。此外，P实例的第0个GPU卡与每个D实例的第0个GPU卡建立NCCL组。同样，P实例的第1个GPU卡与每个D实例的第1个GPU卡建立NCCL组。

![image2](https://github.com/user-attachments/assets/837e61d6-365e-4cbf-8640-6dd7ab295b36)

每个NCCL组占用一定量的GPU内存缓冲区用于通信，其大小主要受`NCCL_MAX_NCHANNELS`环境变量影响。当`NCCL_MAX_NCHANNELS=16`时，一个NCCL组通常占用100MB，而当`NCCL_MAX_NCHANNELS=8`时，通常占用52MB。对于大规模xPyD配置——如DeepSeek的96P144D——这种实现目前不可行。展望未来，我们正在考虑使用RDMA进行点对点通信，同时也在关注UCCL。

## GPU内存缓冲区和Tensor内存池

内存缓冲区大小的权衡如下：对于P实例，PUT和PUT_ASYNC模式不需要内存缓冲区，但GET模式需要。对于D实例，所有三种模式都需要内存缓冲区。D实例的内存缓冲区不应太大。同样，对于GET模式下的P实例，内存缓冲区也不应太大。D实例的内存缓冲区用于临时存储P实例发送的KV缓存。如果太大，会减少D实例正常推理可用的KV缓存空间，从而减少推理批次大小，最终导致输出吞吐量下降。内存缓冲区的大小由参数`kv_buffer_size`配置，以字节为单位，通常设置为内存大小的5%～10%。

如果P实例的`--max-num-seqs`参数设置为大值，由于批次大小大，P实例会同时生成大量KV缓存。这可能会超过D实例内存缓冲区的容量，导致KV缓存丢失。一旦KV缓存丢失，D实例需要重新计算Prefill，这相当于执行两次Prefill。因此，首token时间（TTFT）将显著增加，导致性能下降。

为了解决上述问题，我设计并开发了一个本地Tensor内存池用于存储KV缓存，灵感来自Linux内存模块中使用的伙伴系统。由于内存足够大，通常在服务器上为TB级别，因此不需要考虑前缀缓存或使用基于块的设计来重用内存，从而节省空间。当内存缓冲区不足时，KV缓存可以直接存储在Tensor内存池中，D实例随后可以从中检索KV缓存。读写速度是PCIe的速度，PCIe 4.0的速度约为21 GB/s，通常比Prefill速度快。否则，像Mooncake和lmcache这样的解决方案就不会是必要的。Tensor内存池充当泄洪区，通常在突发流量激增时使用。在最坏的情况下，我的解决方案表现不亚于有缓存存储的正常情况。

# 安装vLLM

```shell
pip install "vllm>=0.9.2"
```

# 运行xPyD

## 说明
- 以下示例在A800（80GB）设备上运行，使用Meta-Llama-3.1-8B-Instruct模型。
- 注意`kv_buffer_size`（以字节为单位）的设置。经验值是GPU内存大小的10%。这与kvcache大小有关。如果太小，用于临时存储接收到的kvcache的GPU内存缓冲区会溢出，导致kvcache存储在tensor内存池中，增加延迟。如果太大，可用于推理的kvcache会减少，导致批次大小更小，吞吐量下降。
- 对于Prefill实例，当使用非GET模式时，`kv_buffer_size`可以设置为1，因为Prefill目前不需要接收kvcache。但是，当使用GET模式时，需要更大的`kv_buffer_size`，因为它需要存储发送给D实例的kvcache。
- 您可能需要修改以下命令中的`kv_buffer_size`和`port`（如果有冲突）。
- `PUT_ASYNC`提供最佳性能，应优先考虑。
- `--port`必须与`--kv-transfer-config`中的`http_port`一致。
- `disagg_proxy_p2p_nccl_xpyd.py`脚本将使用端口10001（用于接收客户端请求）和端口30001（用于接收来自P和D实例的服务发现）。
- 运行代理的节点必须安装`quart`。
- 支持多节点；您只需要修改`--kv-transfer-config`中的`proxy_ip`和`proxy_port`。
- 在以下示例中，假设**代理的IP是10.0.1.1**。

## 运行1P3D

### 代理（例如10.0.1.1）

```shell
cd {your vllm directory}/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/
python3 disagg_proxy_p2p_nccl_xpyd.py &
```

### Prefill1（例如10.0.1.2或10.0.1.1）

??? console "命令"

    ```shell
    VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES=0 vllm serve {your model directory} \
        --host 0.0.0.0 \
        --port 20001 \
        --tensor-parallel-size 1 \
        --seed 1024 \
        --served-model-name base_model \
        --dtype float16 \
        --max-model-len 10000 \
        --max-num-batched-tokens 10000 \
        --max-num-seqs 256 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --disable-log-request \
        --kv-transfer-config \
        '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":"1e1","kv_port":"21001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20001"}}' > /var/vllm.log 2>&1 &
    ```

### Decode1（例如10.0.1.3或10.0.1.1）

??? console "命令"

    ```shell
    VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES=1 vllm serve {your model directory} \
        --host 0.0.0.0 \
        --port 20002 \
        --tensor-parallel-size 1 \
        --seed 1024 \
        --served-model-name base_model \
        --dtype float16 \
        --max-model-len 10000 \
        --max-num-batched-tokens 10000 \
        --max-num-seqs 256 \
        --trust-remote-code \
        --gpu-memory-utilization 0.7 \
        --disable-log-request \
        --kv-transfer-config \
        '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":"8e9","kv_port":"22001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20002"}}' > /var/vllm.log 2>&1 &
    ```

### Decode2（例如10.0.1.4或10.0.1.1）

??? console "命令"

    ```shell
    VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES=2 vllm serve {your model directory} \
        --host 0.0.0.0 \
        --port 20003 \
        --tensor-parallel-size 1 \
        --seed 1024 \
        --served-model-name base_model \
        --dtype float16 \
        --max-model-len 10000 \
        --max-num-batched-tokens 10000 \
        --max-num-seqs 256 \
        --trust-remote-code \
        --gpu-memory-utilization 0.7 \
        --disable-log-request \
        --kv-transfer-config \
        '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":"8e9","kv_port":"23001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20003"}}' > /var/vllm.log 2>&1 &
    ```

### Decode3（例如10.0.1.5或10.0.1.1）

??? console "命令"

    ```shell
    VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES=3 vllm serve {your model directory} \
        --host 0.0.0.0 \
        --port 20004 \
        --tensor-parallel-size 1 \
        --seed 1024 \
        --served-model-name base_model \
        --dtype float16 \
        --max-model-len 10000 \
        --max-num-batched-tokens 10000 \
        --max-num-seqs 256 \
        --trust-remote-code \
        --gpu-memory-utilization 0.7 \
        --disable-log-request \
        --kv-transfer-config \
        '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":"8e9","kv_port":"24001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20004"}}' > /var/vllm.log 2>&1 &
    ```

## 运行3P1D

### 代理（例如10.0.1.1）

```shell
cd {your vllm directory}/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/
python3 disagg_proxy_p2p_nccl_xpyd.py &
```

### Prefill1（例如10.0.1.2或10.0.1.1）

??? console "命令"

    ```shell
    VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES=0 vllm serve {your model directory} \
        --host 0.0.0.0 \
        --port 20001 \
        --tensor-parallel-size 1 \
        --seed 1024 \
        --served-model-name base_model \
        --dtype float16 \
        --max-model-len 10000 \
        --max-num-batched-tokens 10000 \
        --max-num-seqs 256 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --disable-log-request \
        --kv-transfer-config \
        '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":"1e1","kv_port":"21001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20001"}}' > /var/vllm.log 2>&1 &
    ```

### Prefill2（例如10.0.1.3或10.0.1.1）

??? console "命令"

    ```shell
    VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES=1 vllm serve {your model directory} \
        --host 0.0.0.0 \
        --port 20002 \
        --tensor-parallel-size 1 \
        --seed 1024 \
        --served-model-name base_model \
        --dtype float16 \
        --max-model-len 10000 \
        --max-num-batched-tokens 10000 \
        --max-num-seqs 256 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --disable-log-request \
        --kv-transfer-config \
        '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":"1e1","kv_port":"22001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20002"}}' > /var/vllm.log 2>&1 &
    ```

### Prefill3（例如10.0.1.4或10.0.1.1）

??? console "命令"

    ```shell
    VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES=2 vllm serve {your model directory} \
        --host 0.0.0.0 \
        --port 20003 \
        --tensor-parallel-size 1 \
        --seed 1024 \
        --served-model-name base_model \
        --dtype float16 \
        --max-model-len 10000 \
        --max-num-batched-tokens 10000 \
        --max-num-seqs 256 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --disable-log-request \
        --kv-transfer-config \
        '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":"1e1","kv_port":"23001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20003"}}' > /var/vllm.log 2>&1 &
    ```

### Decode1（例如10.0.1.5或10.0.1.1）

??? console "命令"

    ```shell
    VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES=3 vllm serve {your model directory} \
        --host 0.0.0.0 \
        --port 20004 \
        --tensor-parallel-size 1 \
        --seed 1024 \
        --served-model-name base_model \
        --dtype float16 \
        --max-model-len 10000 \
        --max-num-batched-tokens 10000 \
        --max-num-seqs 256 \
        --trust-remote-code \
        --gpu-memory-utilization 0.7 \
        --disable-log-request \
        --kv-transfer-config \
        '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":"8e9","kv_port":"24001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20004"}}' > /var/vllm.log 2>&1 &
    ```

# 单个请求

```shell
curl -X POST -s http://10.0.1.1:10001/v1/completions \
-H "Content-Type: application/json" \
-d '{
    "model": "base_model",
    "prompt": "San Francisco is a",
    "max_tokens": 10,
    "temperature": 0
}'
```

# 基准测试

??? console "命令"

    ```shell
    python3 benchmark_serving.py \
        --backend vllm \
        --model base_model \
        --tokenizer meta-llama/Llama-3.1-8B-Instruct \
        --dataset-name "random" \
        --host 10.0.1.1 \
        --port 10001 \
        --random-input-len 1024 \
        --random-output-len 1024 \
        --ignore-eos \
        --burstiness 100 \
        --percentile-metrics "ttft,tpot,itl,e2el" \
        --metric-percentiles "90,95,99" \
        --seed $(date +%s) \
        --trust-remote-code \
        --request-rate 3 \
        --num-prompts 1000
    ```

# 关闭

```shell
pgrep python | xargs kill -9 && pkill -f python
```

# 测试数据

## **场景**：1K输入和200输出token，端到端P99延迟约2秒

![testdata](https://github.com/user-attachments/assets/cef0953b-4567-4bf9-b940-405b92a28eb1) 