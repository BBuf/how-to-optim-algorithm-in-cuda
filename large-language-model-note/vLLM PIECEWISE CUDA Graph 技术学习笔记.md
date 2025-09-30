# 0x0. 前言

最近有小伙伴用vLLM启动模型的时候，发现capture CUDA Graph的log变成了这一句：`Capturing CUDA graphs (mixed prefill-decode, PIECEWISE)`，然后想找我讨论下这个是什么优化，我去翻了下源码了解了一番就有了这个博客。我们之前启动模型一直看到的是普通的CUDA Graph capture log，这个`PIECEWISE` CUDA Graph 是vLLM compilation模块的核心技术，它可以让我们在prefill阶段对除Attention之外的算子都用上CUDA Graph，从而减少CPU Overhead获得性能提升。

PIECEWISE技术的核心思想是把大的计算图按照特定的算子切分，然后分别编译每个子图。这样既降低了编译复杂度，又让更多算子能用上CUDA Graph的性能优化。vLLM的compilation模块涵盖了图分割、算子融合Pass、多种编译后端等技术。

这篇文章记录一下vLLM compilation模块的技术细节，从整体架构到具体实现，把核心技术点都梳理一遍。

# 0x1. vLLM Compilation架构

vLLM的Compilation系统采用了分层设计，主要包含以下几个核心组件：

## 0x1.1 Compilation级别设计

vLLM定义了多种编译级别，从`CompilationLevel`枚举可以看出：

```python
class CompilationLevel(IntEnum):
    NO_COMPILATION = 0
    DYNAMO_AS_IS = 1
    DYNAMO_ONCE = 2
    PIECEWISE = 3
```

- `NO_COMPILATION`: 不进行任何编译
- `DYNAMO_AS_IS`: 使用PyTorch Dynamo的默认行为
- `DYNAMO_ONCE`: 使用Dynamo编译一次，然后直接调度到编译后的代码
- `PIECEWISE`: 分片编译，这是vLLM的核心创新

## 0x1.2 Compilation后端架构

vLLM支持多种编译后端，通过`CompilerInterface`抽象接口统一管理：

```python
class CompilerInterface:
    name: str
    
    def initialize_cache(self, cache_dir: str, disable_cache: bool = False, prefix: str = ""):
        pass
    
    def compute_hash(self, vllm_config: VllmConfig) -> str:
        return ""
    
    def compile(self, graph: fx.GraphModule, example_inputs: list[Any], 
                compiler_config: dict[str, Any], runtime_shape: Optional[int] = None,
                key: Optional[str] = None) -> tuple[Optional[Callable], Optional[Any]]:
        return None, None
    
    def load(self, handle: Any, graph: fx.GraphModule, example_inputs: list[Any],
             graph_index: int, runtime_shape: Optional[int] = None) -> Callable:
        raise NotImplementedError("caching is not supported")
```

目前vLLM实现了三种Compilation后端：

1. **EagerAdaptor**: 直接返回原始图，不进行编译
2. **InductorAdaptor**: 使用PyTorch Inductor进行编译（适用于PyTorch 2.5-2.7）
3. **InductorStandaloneAdaptor**: 使用独立的Inductor编译器（适用于PyTorch 2.8+）

# 0x2. 分片编译(Piecewise Compilation)技术      

## 0x2.1 核心设计

分片编译是vLLM的核心创新，基本思路是把大的计算图按特定算子切分，然后分别编译每个子图。这个技术最大的价值是让CUDA Graph能用到prefill阶段，以前prefill阶段因为输入长度是动态的，很难用CUDA Graph优化。

通过PIECEWISE技术，vLLM可以：

1. **在prefill阶段使用CUDA Graph**: 对除Attention之外的算子（如MLP、RMSNorm等）用CUDA Graph，大幅减少CPU Overhead
2. **降低编译复杂度**: 把大图拆成小图，每个子图独立编译优化
3. **提高编译缓存命中率**: 小图的缓存更容易命中，减少重复编译时间
4. **支持更细粒度的优化**: 不同类型的算子可以用不同的优化策略

所以我们会看到`Capturing CUDA graphs (mixed prefill-decode, PIECEWISE)`这样的log，说明vLLM正在为prefill和decode阶段的混合workload捕获分片的CUDA Graph。

## 0x2.2 PIECEWISE模式下的Prefill阶段CUDA Graph Capture 解析

我们来看看vLLM是怎么在PIECEWISE模式下对Prefill阶段的非Attention算子进行CUDA Graph capture的。

### Capture Size的确定和分片后端机制

vLLM通过`compilation_config.compile_sizes`来确定需要为哪些batch size编译和capture CUDA Graph。默认情况下，vLLM会根据`cudagraph_capture_sizes`自动推断，计算逻辑如下：

```python
# 在 vllm/config/__init__.py 中
possible_sizes = [1, 2, 4] + [8 * i for i in range(1, 1025)]
max_graph_size = min(max_num_seqs * 2, 512)
# 最终得到: [1, 2, 4, 8, 16, 24, 32, 40, ..., max_graph_size]
```

在`PiecewiseBackend`类中，实现了具体的capture逻辑。每个子图会为不同的batch size创建独立的编译entry，第一次遇到特定size时进行编译，后续直接复用：

```python
# 在 vllm/compilation/cuda_piecewise_backend.py 中
class PiecewiseBackend:
    def __call__(self, *args) -> Any:
        runtime_shape = args[self.sym_shape_indices[0]]
        
        if runtime_shape not in self.concrete_size_entries:
            # 对于不在capture列表中的size，使用通用编译图
            return self.compiled_graph_for_general_shape(*args)
        
        entry = self.concrete_size_entries[runtime_shape]
        if not entry.compiled:
            # 第一次遇到这个size时，进行编译
            entry.compiled = True
            entry.runnable = self.vllm_backend.compiler_manager.compile(
                self.graph, args, ..., runtime_shape=runtime_shape)
        
        return entry.runnable(*args)
```

### CUDA Graph的Capture和Replay机制

每个编译后的子图都会被`CUDAGraphWrapper`包装。当第一次遇到某个batch descriptor时，会capture CUDA Graph；后续调用直接replay：

```python
# 在 vllm/compilation/cuda_graph.py 中
class CUDAGraphWrapper:
    def __call__(self, *args, **kwargs):
        # 检查runtime mode是否匹配
        if cudagraph_runtime_mode != self.runtime_mode:
            return self.runnable(*args, **kwargs)
        
        if entry.cudagraph is None:
            # 第一次遇到时，capture CUDA Graph
            cudagraph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cudagraph, pool=self.graph_pool):
                output = self.runnable(*args, **kwargs)
            entry.cudagraph = cudagraph
            return output
        
        # 后续调用直接replay
        entry.cudagraph.replay()
        return entry.output
```

### 实际Capture的算子和性能优化

在PIECEWISE模式下，vLLM主要对以下算子进行CUDA Graph capture：

- **MLP层算子**: Linear层矩阵乘法、激活函数（SiLU、GELU等）、残差连接
- **Norm算子**: RMSNorm、LayerNorm及其与量化的融合版本  
- **量化算子**: FP8/INT8量化、各种per-token/per-tensor量化
- **其它算子**: Embedding层、位置编码、element-wise操作

**关键排除**: Attention算子由于对序列长度敏感，在Prefill阶段不会被capture，保持动态执行。


### 没有命中CUDA Graph时的Padding逻辑

即使没有命中CUDA Graph，vLLM仍然有padding逻辑来优化性能。vLLM预先计算了一个`bs_to_padded_graph_size`数组，实现O(1)的padding size查找：

```python
# CUDA Graph范围内：padding到最近的capture size
def pad_for_cudagraph(self, batch_size: int) -> int:
    return self.compilation_config.bs_to_padded_graph_size[batch_size]

# Eager Mode：padding到TP size的倍数（用于Sequence Parallelism优化）
if (cudagraph_mode != CUDAGraphMode.NONE and num_tokens <= cudagraph_batch_sizes[-1]):
    num_tokens_padded = self.vllm_config.pad_for_cudagraph(num_tokens)
else:
    # Eager mode: pad to multiple of tensor_parallel_size for SP
    if enable_sequence_parallelism and tp_size > 1:
        num_tokens_padded = round_up(num_tokens, tp_size)
```

例如，配置`cudagraph_capture_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]`和`tensor_parallel_size = 8`时：
- batch_size=10 → padding到16（命中CUDA Graph）
- batch_size=300 → padding到304（Eager mode + SP padding）

这种双重padding策略让vLLM在各种batch size下都能有不错的性能。

## 0x2.3 图分割实现

在`backends.py`中的`split_graph`函数实现了图分割逻辑：

```python
def split_graph(graph: fx.GraphModule, ops: list[str]) -> tuple[fx.GraphModule, list[SplitItem]]:
    subgraph_id = 0
    node_to_subgraph_id = {}
    split_op_graphs = []
    
    for node in graph.graph.nodes:
        if node.op in ("output", "placeholder"):
            continue
        if node.op == 'call_function' and str(node.target) in ops:
            subgraph_id += 1
            node_to_subgraph_id[node] = subgraph_id
            split_op_graphs.append(subgraph_id)
            subgraph_id += 1
        else:
            node_to_subgraph_id[node] = subgraph_id
    
    split_gm = torch.fx.passes.split_module.split_module(
        graph, None, lambda node: node_to_subgraph_id[node], keep_original_order=True)
    
    # ... 处理分割结果
    return split_gm, outputs
```

### 关键参数解析：splitting_ops

这里传入的`ops`参数来自`compilation_config.splitting_ops`，它定义了哪些算子作为分割点。从源码可以看到：

**1. 默认的Attention算子列表**
```python
# 在 vllm/config/compilation.py 中
_attention_ops: ClassVar[list[str]] = [
    "vllm.unified_attention",
    "vllm.unified_attention_with_output", 
    "vllm.mamba_mixer2",
    "vllm.mamba_mixer",
    "vllm.short_conv",
    "vllm.linear_attention",
    "vllm.plamo2_mamba_mixer",
    "vllm.gdn_attention",
]
```

**2. 动态添加的MoE算子**
```python
if envs.VLLM_ALL2ALL_BACKEND == "deepep_high_throughput":
    # exclude MoE dispatch/combine from capture by ensuring
    # piecewise splitting includes them, so communication remains
    # outside CUDA graphs while compute can still be graphed.
    moe_ops = [
        "vllm.moe_forward",
        "vllm.moe_forward_shared",
    ]
    for op in moe_ops:
        if op not in self.splitting_ops:
            self.splitting_ops.append(op)
```

**3. 分割逻辑**
- 当遇到`splitting_ops`中的算子时，会在该算子前后创建分割点
- 每个分割点都会生成一个新的`subgraph_id`
- 这样就将原始的大图分割成多个独立的子图

**4. 分割效果**
- **Attention子图**：包含attention相关的算子，保持动态执行
- **MLP子图**：包含Linear、激活函数等，可以被CUDA Graph capture
- **Norm子图**：包含RMSNorm等归一化算子，同样可以被capture

这里的关键是`keep_original_order=True`，保证分割后的子图按原来的顺序执行，不会改变语义。这样vLLM就能对不同类型的算子采用不同的处理策略。

## 0x2.4 分片后端实现

`PiecewiseCompileInterpreter`负责执行分片编译：

```python
class PiecewiseCompileInterpreter(torch.fx.Interpreter):
    def call_module(self, target: torch.fx.node.Target, args: tuple, kwargs: dict) -> Any:
        output = super().call_module(target, args, kwargs)
        
        if target in self.compile_submod_names:
            index = self.compile_submod_names.index(target)
            submod = self.fetch_attr(target)
            
            # 编译动态形状的图
            compiled_graph_for_dynamic_shape = self.vllm_backend.compiler_manager.compile(
                submod, args, self.compilation_config.inductor_compile_config,
                self.compilation_config, graph_index=index,
                num_graphs=len(self.compile_submod_names), runtime_shape=None)
            
            # 创建分片后端
            piecewise_backend = PiecewiseBackend(
                submod, self.vllm_config, index, len(self.compile_submod_names),
                sym_shape_indices, compiled_graph_for_dynamic_shape, self.vllm_backend)
            
            # 如果启用CUDA Graph，包装为CUDAGraphWrapper
            if self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
                static_graph_wrapper_class = resolve_obj_by_qualname(
                    current_platform.get_static_graph_wrapper_cls())
                self.module.__dict__[target] = static_graph_wrapper_class(
                    runnable=piecewise_backend, vllm_config=self.vllm_config,
                    runtime_mode=CUDAGraphMode.PIECEWISE, ...)
            else:
                self.module.__dict__[target] = piecewise_backend
        
        return output
```

# 0x3. vLLM Compilation 算子融合技术

## 0x3.1 融合框架设计

vLLM在Compilation模块实现了一套完整的算子融合框架，主要包含：

1. **FusionPass**: 通用融合Pass，主要处理RMSNorm+量化的融合
2. **ActivationQuantFusionPass**: 激活量化融合
3. **AttnFusionPass**: 注意力算子融合
4. **AllReduceFusionPass**: 集合通信融合

## 0x3.2 RMSNorm量化融合实现

以RMSNorm+FP8量化融合为例，vLLM使用PyTorch的pattern matcher进行模式匹配和替换：

```python
class FusedAddRMSNormStaticQuantPattern(RMSNormQuantPattern):
    def register(self, pm_pass: PatternMatcherPass, record_match: Callable):
        def pattern(result: torch.Tensor, input: torch.Tensor, residual: torch.Tensor,
                   weight: torch.Tensor, scale: torch.Tensor):
            # 原始模式：先做fused_add_rms_norm，再做量化
            at = auto_functionalized(RMS_ADD_OP, input=input, residual=residual,
                                   weight=weight, epsilon=self.epsilon)
            at1 = auto_functionalized(self.QUANT_OP, result=result, input=at[1], scale=scale)
            return at1[1], at[2]  # result, residual
        
        def replacement(result: torch.Tensor, input: torch.Tensor, residual: torch.Tensor,
                       weight: torch.Tensor, scale: torch.Tensor):
            # 融合后的模式：一个算子完成所有操作
            at = auto_functionalized(self.FUSED_OP, result=result, input=input,
                                   residual=residual, weight=weight, scale=scale,
                                   epsilon=self.epsilon)
            return at[1], at[2]  # result, residual
        
        pm.register_replacement(pattern, replacement, inputs, pm.fwd_only, pm_pass,
                              extra_check=lambda m: record_match(self.Match(m, self.QUANT_OP, self.FUSED_OP)))
```

这里的关键技术点：

1. 使用`auto_functionalized`包装原地操作，确保函数式编程语义
2. 通过`extra_check`回调记录匹配，支持多输出模式的手动处理
3. 定义了完整的输入输出映射关系

## 0x3.3 多输出匹配处理

对于有多个输出的融合模式，vLLM实现了`MultiOutputMatch`类来处理PyTorch pattern matcher对多输出支持不完善的问题。

### 问题背景

在算子融合中，经常遇到多输出的情况。例如，RMSNorm+量化融合：

```python
# 原始模式：两个独立的算子
# 1. RMSNorm: 输入 -> (None, normalized_output, residual)  
# 2. 量化: normalized_output -> (None, quantized_result, scale)

# 融合后：一个算子产生多个输出
# 融合算子: 输入 -> (None, quantized_result, scale, residual)
```

PyTorch的pattern matcher在处理这种多输出替换时存在bug，因此vLLM实现了手动处理机制。

### 核心实现机制

**1. 模式匹配和记录**

```python
class FusedAddRMSNormStaticQuantPattern(RMSNormQuantPattern):
    def register(self, pm_pass, record_match):
        def pattern(result, input, residual, weight, scale):
            # 原始模式：先做RMSNorm，再做量化
            at = auto_functionalized(RMS_ADD_OP, input=input, residual=residual, weight=weight)
            at1 = auto_functionalized(self.QUANT_OP, result=result, input=at[1], scale=scale)
            return at1[1], at[2]  # 返回量化结果和残差
        
        def replacement(result, input, residual, weight, scale):
            # 融合后：一个算子完成所有操作
            at = auto_functionalized(self.FUSED_OP, result=result, input=input, 
                                   residual=residual, weight=weight, scale=scale)
            return at[1], at[2]  # 返回相同的输出
        
        # 关键：使用extra_check记录匹配，触发手动处理
        pm.register_replacement(pattern, replacement, inputs, pm.fwd_only, pm_pass,
                              extra_check=lambda m: record_match(self.Match(m, self.QUANT_OP, self.FUSED_OP)))
```

**2. 手动替换处理**

```python
class Match(QuantMultiOutputMatch):
    def process(self):
        # 1. 找到匹配中的关键节点
        rms_node = self.find_auto_fn(RMS_ADD_OP)      # RMSNorm节点
        quant_node = self.find_auto_fn(self.QUANT_OP)  # 量化节点
        
        # 2. 插入融合后的节点
        with self.inserting_after_match():
            # 定义输出映射关系：融合节点的哪个输出对应原来哪个节点的哪个输出
            fused_return_mapping = {
                1: (quant_node, 1),  # 融合节点的第1个输出 -> 量化节点的第1个输出
                2: (rms_node, 2),    # 融合节点的第2个输出 -> RMSNorm节点的第2个输出
            }
            self.insert_fused_node(fused_return_mapping, **kwargs)
```

**3. 核心替换逻辑**

```python
def insert_fused_node(self, fused_return_mapping: dict[int, tuple[fx.Node, int]], **kwargs):
    # 1. 创建融合算子节点
    fused_node = self.insert_auto_fn(self.FUSED_OP, kwargs)
    
    # 2. 为融合节点的每个输出创建getitem节点
    indices = fused_return_mapping.keys()  # [1, 2]
    getitem_nodes = self.insert_getitems(fused_node, indices)  # [fused_node[1], fused_node[2]]
    
    # 3. 重新绑定用户节点
    for idx, getitem_node in zip(indices, getitem_nodes):
        old_node, old_idx = fused_return_mapping[idx]
        
        # 找到原来的getitem节点（如果存在）
        old_getitem = find_getitem_maybe(old_node, old_idx)
        if old_getitem is not None:
            # 将所有使用old_getitem的地方替换为新的getitem_node
            old_getitem.replace_all_uses_with(getitem_node)
            # 复制meta信息，用于去函数化
            getitem_node.meta["val"] = old_getitem.meta["val"]
        
        # 设置融合节点的meta信息
        meta_val[idx] = old_node.meta["val"][old_idx]
    
    fused_node.meta["val"] = tuple(meta_val)
```

### 实际效果

通过这种机制，vLLM成功将：

```python
# 原始图结构
input -> RMSNorm -> normalized_output -> Quantize -> quantized_result
      -> residual                    -> scale
```

转换为：

```python  
# 融合后图结构
input -> FusedRMSNormQuant -> quantized_result
                           -> scale  
                           -> residual
```

这种设计既解决了PyTorch pattern matcher的局限性，又保证了融合后图的正确性和性能优化效果。

# 0x4. vLLM Compilation 集合通信融合技术

## 0x4.1 AllReduce融合

vLLM实现了多种AllReduce融合模式，包括：

1. **GEMM + ReduceScatter**: 将矩阵乘法和reduce-scatter融合
2. **AllGather + GEMM**: 将all-gather和矩阵乘法融合
3. **RMSNorm + AllReduce**: 将RMSNorm和all-reduce融合

以GEMM+ReduceScatter为例：

```python
class GEMMReduceScatterPattern(BasePattern):
    def register(self, pm_pass: PatternMatcherPass):
        def pattern(mul: torch.Tensor, mm_weight: torch.Tensor):
            mm = torch.ops.aten.mm.default(mul, mm_weight)
            reduce_scatter = torch.ops.vllm.reduce_scatter.default(
                mm, dim=0, world_size=self.tp_size, group_name=self.tp.unique_name)
            return reduce_scatter
        
        def replacement(mul: torch.Tensor, mm_weight: torch.Tensor):
            gemm_rs = torch.ops.symm_mem.fused_matmul_reduce_scatter(
                mul, mm_weight, "avg", scatter_dim=0,
                group_name=self.tp.device_group.group_name)
            return gemm_rs
        
        pm.register_replacement(pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass)
```

## 0x4.2 FlashInfer通信融合

vLLM还集成了FlashInfer的通信融合功能：

```python
if flashinfer_comm and hasattr(flashinfer_comm, "trtllm_allreduce_fusion"):
    class FlashInferAllReducePattern(BasePattern):
        def register(self, pm_pass: PatternMatcherPass):
            def pattern(input: torch.Tensor):
                return torch.ops.vllm.all_reduce.default(
                    input, group_name=self.tp.unique_name)
            
            def replacement(input: torch.Tensor):
                return flashinfer_comm.trtllm_allreduce_fusion(
                    input, self.tp_size, get_tensor_model_parallel_rank())
            
            pm.register_replacement(pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass)
```

# 0x5. vLLM Compilation 编译缓存机制

## 0x5.1 缓存架构设计

vLLM Compilation实现了完善的编译缓存机制，通过`CompilerManager`统一管理：

```python
class CompilerManager:
    def __init__(self, compilation_config: CompilationConfig):
        self.cache: dict[tuple[Optional[int], int, str], Any] = dict()
        self.is_cache_updated = False
        self.compilation_config = compilation_config
        self.compiler = make_compiler(compilation_config)
    
    def compute_hash(self, vllm_config: VllmConfig) -> str:
        return self.compiler.compute_hash(vllm_config)
    
    def initialize_cache(self, cache_dir: str, disable_cache: bool = False, prefix: str = ""):
        self.cache_dir = cache_dir
        self.cache_file_path = os.path.join(cache_dir, "vllm_compile_cache.py")
        
        if not disable_cache and os.path.exists(self.cache_file_path):
            with open(self.cache_file_path) as f:
                self.cache = ast.literal_eval(f.read())
        
        self.compiler.initialize_cache(cache_dir=cache_dir, disable_cache=disable_cache, prefix=prefix)
```

## 0x5.2 缓存键设计

缓存键的设计考虑了多个因素：

```python
def __call__(self, graph: fx.GraphModule, example_inputs) -> Callable:
    if not self.compilation_config.cache_dir:
        factors = []
        # 1. 环境变量哈希
        env_hash = envs.compute_hash()
        factors.append(env_hash)
        
        # 2. vLLM配置哈希
        config_hash = vllm_config.compute_hash()
        factors.append(config_hash)
        
        # 3. 代码文件哈希
        forward_code_files = list(sorted(self.compilation_config.traced_files))
        hash_content = []
        for filepath in forward_code_files:
            hash_content.append(filepath)
            if filepath != "<string>":
                with open(filepath) as f:
                    hash_content.append(f.read())
        code_hash = hashlib.md5("\n".join(hash_content).encode(), usedforsecurity=False).hexdigest()
        factors.append(code_hash)
        
        # 4. 编译器哈希
        compiler_hash = self.compiler_manager.compute_hash(vllm_config)
        factors.append(compiler_hash)
        
        hash_key = hashlib.md5(str(factors).encode(), usedforsecurity=False).hexdigest()[:10]
        cache_dir = os.path.join(envs.VLLM_CACHE_ROOT, "torch_compile_cache", hash_key)
```

这种设计保证了缓存的正确性和有效性。

## 0x5.3 缓存使用方式

### 缓存目录结构

vLLM的编译缓存采用分层目录结构：

```bash
~/.cache/vllm/torch_compile_cache/
├── hash_key_1/           # 基于配置和代码的哈希值
│   ├── rank_0_1/         # 多进程/多GPU的rank信息
│   │   ├── prefix_name/  # 不同模块的前缀
│   │   │   ├── vllm_compile_cache.py      # 编译缓存索引
│   │   │   ├── computation_graph.py       # 计算图转储
│   │   │   └── transformed_code.py        # 转换后的代码
│   │   └── shared_artifacts/              # 共享编译产物
│   └── rank_2_3/
└── hash_key_2/
```

更详细的细节可以看`class CompilerManager`的实现。

### 缓存键设计

缓存键(hash_key_1, hash_key_2 ...)采用三元组结构：`(runtime_shape, graph_index, backend_name)`

```python
# 缓存键示例
cache_key = (
    16,           # runtime_shape: batch_size=16
    2,            # graph_index: 第2个子图  
    "inductor"    # backend_name: 使用Inductor后端
)
```

### 缓存加载和存储流程

**1. 编译时的缓存查找**

```python
def compile(self, graph, example_inputs, graph_index, runtime_shape):
    # 1. 首先尝试从缓存加载
    compiled_graph = self.load(graph, example_inputs, graph_index, runtime_shape)
    if compiled_graph is not None:
        logger.info("Directly load compiled graph from cache, took %.3f s", elapsed)
        return compiled_graph
    
    # 2. 缓存未命中，进行编译
    compiled_graph, handle = self.compiler.compile(graph, example_inputs, ...)
    
    # 3. 将编译结果存储到缓存
    if not envs.VLLM_DISABLE_COMPILE_CACHE and handle is not None:
        self.cache[(runtime_shape, graph_index, self.compiler.name)] = handle
        compilation_counter.num_cache_entries_updated += 1
        self.is_cache_updated = True
```

**2. 缓存持久化**

```python
def save_to_file(self):
    if self.disable_cache or not self.is_cache_updated:
        return
    # 使用Python格式保存，便于调试和可读性
    printer = pprint.PrettyPrinter(indent=4)
    data = printer.pformat(self.cache)
    with open(self.cache_file_path, "w") as f:
        f.write(data)
```

### 缓存机制的好处

```python
# 首次启动（无缓存）
logger.info("Compiling graph for shape 16, took 45.2 s")

# 后续启动（命中缓存）  
logger.info("Directly load compiled graph from cache, took 0.8 s")
```

缓存命中可以将编译时间从数十秒减少到不到1秒，特别是对于大模型和复杂的分片编译场景。


# 0x6. vLLM Compilation 装饰器系统

vLLM提供了一套完整的装饰器系统来简化模型编译，主要包括`@support_torch_compile`和`@ignore_torch_compile`两个核心装饰器。

## 0x6.1 编译装饰器设计

### 基本使用方式

vLLM提供了`@support_torch_compile`装饰器来简化模型编译：

```python
# 方式1: 直接使用装饰器（自动推断动态维度）
@support_torch_compile
class MyModel(nn.Module):
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor]):
        ...

# 方式2: 显式指定动态维度
@support_torch_compile(dynamic_arg_dims={"x": 0, "y": [0, 1]})
class MyModel(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        ...

# 方式3: 条件编译
@support_torch_compile(enable_if=lambda config: config.model_config.dtype == torch.float16)
class MyModel(nn.Module):
    def forward(self, x: torch.Tensor):
        ...
```

### 动态维度自动推断

当没有显式指定`dynamic_arg_dims`时，装饰器会自动推断：

```python
def cls_decorator_helper(cls: _T) -> _T:
    sig = inspect.signature(cls.forward)
    inferred_dynamic_arg_dims = {}
    
    # 遍历forward方法的所有参数
    for k, v in sig.parameters.items():
        # 根据类型注解自动推断动态维度
        if v.annotation in [torch.Tensor, Optional[torch.Tensor], 
                           IntermediateTensors, Optional[IntermediateTensors]]:
            inferred_dynamic_arg_dims[k] = 0  # 第一个维度标记为动态
    
    logger.debug("Inferred dynamic dimensions for forward method of %s: %s", 
                 cls, list(inferred_dynamic_arg_dims.keys()))
    
    return _support_torch_compile(cls, inferred_dynamic_arg_dims, enable_if)
```

**推断规则**：
- `torch.Tensor`或`Optional[torch.Tensor]`：第0维标记为动态
- `IntermediateTensors`：所有tensor的第0维标记为动态
- 其他类型：忽略

`IntermediateTensors`是vLLM为Pipeline Parallelism设计的专用数据结构，它：
封装了多个相关的tensor（主要是`hidden_states`和`residual`）
- 支持Pipeline stage之间的数据传递
- 在编译系统中有特殊的动态维度处理
- 提供了字典式的访问接口，方便获取和设置不同的中间状态


## 0x6.2 装饰器实现机制

### 类继承和方法替换

装饰器通过修改类的继承关系和替换方法来实现编译支持：

```python
def _support_torch_compile(cls, dynamic_arg_dims, enable_if):
    # 1. 修改继承关系，添加编译包装器基类
    cls.__bases__ = cls.__bases__ + (TorchCompileWrapperWithCustomDispatcher,)
    
    old_init = cls.__init__
    
    # 2. 替换__init__方法，添加编译配置
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = '', **kwargs):
        old_init(self, vllm_config=vllm_config, prefix=prefix, **kwargs)
        
        # 判断是否需要编译
        enable_compile = enable_if is None or enable_if(vllm_config)
        self.do_not_compile = (
            vllm_config.compilation_config.level in [
                CompilationLevel.NO_COMPILATION, 
                CompilationLevel.DYNAMO_AS_IS
            ] or not supports_dynamo() 
            or _should_ignore_torch_compile(self.__class__) 
            or not enable_compile
        )
        
        if not self.do_not_compile:
            compilation_counter.num_models_seen += 1
            TorchCompileWrapperWithCustomDispatcher.__init__(
                self, compilation_level=vllm_config.compilation_config.level)
    
    cls.__init__ = __init__
```

### 动态形状标记和编译调度

```python
def __call__(self, *args, **kwargs):
    # 跳过编译的情况
    if self.do_not_compile or torch.compiler.is_compiling():
        return self.forward(*args, **kwargs)
    
    # 首次编译：标记动态维度
    if len(self.compiled_codes) < 1:
        sig = inspect.signature(self.__class__.forward)
        bound_args = sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()
        
        # 为每个参数标记动态维度
        for k, dims in dynamic_arg_dims.items():
            arg = bound_args.arguments.get(k)
            if arg is not None:
                dims = [dims] if isinstance(dims, int) else dims
                
                if isinstance(arg, torch.Tensor):
                    # 处理负索引：-1表示最后一维
                    dims = [arg.ndim + dim if dim < 0 else dim for dim in dims]
                    torch._dynamo.mark_dynamic(arg, dims)
                    
                elif isinstance(arg, IntermediateTensors):
                    # 为IntermediateTensors中的所有tensor标记动态维度
                    for tensor in arg.tensors.values():
                        dims = [tensor.ndim + dim if dim < 0 else dim for dim in dims]
                        torch._dynamo.mark_dynamic(tensor, dims)
        
        # 开始监控编译过程
        start_monitoring_torch_compile(self.vllm_config)
        logger.debug("Start compiling function %s", self.original_code_object)
    
    # 编译调度逻辑
    if len(self.compiled_codes) < 1 or not self.use_custom_dispatcher:
        # 使用Dynamo的默认调度机制
        torch._dynamo.eval_frame.remove_from_cache(self.original_code_object)
        
        # 收集被Dynamo追踪的文件，用于缓存失效
        self.vllm_config.compilation_config.traced_files.add(
            self.original_code_object.co_filename)
        
        # 通过patch机制收集内联函数的文件
        with patch.object(InliningInstructionTranslator, 'inline_call', patched_inline_call):
            output = self.compiled_callable(*args, **kwargs)
        return output
    
    # 使用自定义调度器直接调用编译后的代码
    with self.dispatch_to_code(0):
        model_output = self.forward(*args, **kwargs)
        return model_output
```

## 0x6.3 编译控制装饰器

### @ignore_torch_compile装饰器

用于忽略父类的编译装饰器：

```python
@ignore_torch_compile
class ChildModel(ParentModelWithCompile):
    def forward(self, x):
        # 这个类不会被编译，即使父类有@support_torch_compile
        ...

def ignore_torch_compile(cls: _T) -> _T:
    """
    忽略support_torch_compile装饰器的影响。
    - 如果父类有support_torch_compile但子类有ignore_torch_compile，子类不会被编译
    - 如果父类有ignore_torch_compile但子类有support_torch_compile，子类仍会被编译
    - 只影响当前类的forward方法，不影响子模块
    """
    setattr(cls, IGNORE_COMPILE_KEY, True)
    return cls
```

### 条件编译支持

通过`enable_if`参数实现条件编译：

```python
# 只在特定条件下编译
@support_torch_compile(
    enable_if=lambda config: (
        config.model_config.dtype == torch.float16 and 
        config.parallel_config.tensor_parallel_size == 1
    )
)
class ConditionalModel(nn.Module):
    def forward(self, x):
        ...
```

## 0x6.4 编译包装器基类

### TorchCompileWrapperWithCustomDispatcher

这是装饰器系统的核心基类：

```python
class TorchCompileWrapperWithCustomDispatcher:
    def __init__(self, compiled_callable=None, compilation_level=0):
        vllm_config = get_current_vllm_config()
        
        if compiled_callable is None:
            # 默认编译设置：编译forward方法
            backend = vllm_config.compilation_config.init_backend(vllm_config)
            options = None
            if backend == "inductor":
                options = vllm_config.compilation_config.inductor_compile_config
            
            compiled_callable = torch.compile(
                self.forward,
                fullgraph=envs.VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE,
                backend=backend,
                options=options
            )
        
        self.compiled_callable = compiled_callable
        self.original_code_object = self.__class__.forward.__code__
        self.compiled_codes: list[CodeType] = []
        
        # 注册字节码钩子，用于保存编译后的字节码
        torch._dynamo.convert_frame.register_bytecode_hook(self.bytecode_hook)
        
        # 根据编译级别决定是否使用自定义调度器
        self.use_custom_dispatcher = compilation_level >= CompilationLevel.DYNAMO_ONCE
```

### 字节码钩子和调试支持

```python
def bytecode_hook(self, old_code: CodeType, new_code: CodeType):
    """保存编译后的字节码用于直接执行和调试"""
    if old_code is not self.original_code_object:
        return
    
    self.compiled_codes.append(new_code)
    
    # 调试支持：转储计算图和转换后的代码
    debug_dump_dir = self.vllm_config.compilation_config.debug_dump_path
    if debug_dump_dir:
        rank = self.vllm_config.parallel_config.rank
        decompiled_file = os.path.join(debug_dump_dir, f"rank_{rank}", "transformed_code.py")
        
        try:
            import depyf
            src = depyf.decompile(new_code)
            with open(decompiled_file, "w") as f:
                f.write(src)
            logger.debug("Dynamo transformed code saved to %s", decompiled_file)
        except Exception:
            pass
```


# 0x7. CUDA Graph集成

## 0x7.1 CUDA Graph模式

vLLM Compilation支持多种CUDA Graph模式：

```python
class CUDAGraphMode(IntEnum):
    NONE = 0
    PIECEWISE = 1  # 分片CUDA Graph模式，这就是我们在log中看到的PIECEWISE
    FULL = 2       # 全图CUDA Graph模式
```

`PIECEWISE`模式是vLLM的创新，它允许在prefill阶段对部分算子使用CUDA Graph。以往的CUDA Graph由于需要固定的输入形状，在prefill阶段很难应用。但通过分片编译，vLLM可以将那些对输入长度不敏感的算子（如MLP层、RMSNorm等）单独提取出来，为它们创建CUDA Graph，而将对输入长度敏感的算子（主要是Attention）保持动态执行。

在分片编译中，每个子图都可以独立地使用CUDA Graph：

```python
if self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
    static_graph_wrapper_class = resolve_obj_by_qualname(
        current_platform.get_static_graph_wrapper_cls())
    
    self.module.__dict__[target] = static_graph_wrapper_class(
        runnable=piecewise_backend,
        vllm_config=self.vllm_config,
        runtime_mode=CUDAGraphMode.PIECEWISE,
        cudagraph_options=CUDAGraphOptions(
            debug_log_enable=piecewise_backend.is_first_graph,
            gc_disable=not piecewise_backend.is_first_graph,
            weak_ref_output=piecewise_backend.is_last_graph))
```

这里细节实在有点多，写不动了，感兴趣可以直接看这里的源码：https://github.com/vllm-project/vllm/blob/main/vllm/compilation/backends.py#L401


![](https://files.mdnice.com/user/59/3a63d52d-c78c-485d-b6af-abb929cd6f53.png)


# 0x8. vLLM Compilation Pass管理系统

## 0x8.1 Pass管理器设计

vLLM实现了`PostGradPassManager`来管理所有的Pass：

```python
class PostGradPassManager(CustomGraphPass):
    def __init__(self):
        self.passes: list[VllmInductorPass] = []
    
    def configure(self, config: VllmConfig):
        if self.pass_config.enable_noop:
            self.passes += [NoOpEliminationPass(config)]
        
        if self.pass_config.enable_sequence_parallelism:
            self.passes += [SequenceParallelismPass(config)]
            if self.pass_config.enable_async_tp:
                self.passes += [AsyncTPPass(config)]
        
        if self.pass_config.enable_fusion:
            self.passes += [FusionPass.instance(config)]
            self.passes += [ActivationQuantFusionPass(config)]
        
        if self.pass_config.enable_attn_fusion:
            self.passes += [AttnFusionPass(config)]
        
        if self.pass_config.enable_fi_allreduce_fusion:
            self.passes += [AllReduceFusionPass(config)]
        
        self.fix_functionalization = FixFunctionalizationPass(config)
    
    def __call__(self, graph: fx.Graph):
        shape = get_pass_context().runtime_shape
        for pass_ in self.passes:
            if pass_.is_applicable_for_shape(shape):
                pass_(graph)
        
        # 总是最后运行fix_functionalization
        self.fix_functionalization(graph)
```

## 0x8.2 Pass执行顺序

Pass的执行顺序是这样的：

1. NoOp消除Pass
2. 序列并行Pass
3. 异步张量并行Pass  
4. 融合Pass（FusionPass, ActivationQuantFusionPass）
5. 注意力融合Pass
6. FlashInfer AllReduce融合Pass
7. 函数化修复Pass（总是最后执行）

这个顺序保证所有Pass都在函数化的图上操作。这里只是简单介绍，想了解细节的话可以看源码：https://github.com/vllm-project/vllm/blob/main/vllm/compilation/backends.py#L401

# 0x9. vLLM Compilation 性能监控和调试

## 0x9.1 编译计数器

vLLM实现了详细的编译统计：

```python
@dataclasses.dataclass
class CompilationCounter:
    num_models_seen: int = 0                      # 看到的模型数量
    num_graphs_seen: int = 0                      # 看到的计算图数量
    num_piecewise_graphs_seen: int = 0            # 分片图数量
    num_piecewise_capturable_graphs_seen: int = 0 # 可capture的分片图数量
    num_inductor_compiles: int = 0                # Inductor编译次数
    num_backend_compilations: int = 0             # 后端编译次数
    num_eager_compiles: int = 0                   # Eager编译次数
    num_cache_entries_updated: int = 0            # 缓存条目更新次数
    num_compiled_artifacts_saved: int = 0         # 保存的编译产物数量
```

### 使用方法

**1. 查看编译统计信息**

```python
from vllm.compilation.counter import compilation_counter

# 在模型运行后查看统计信息
print(f"编译的模型数量: {compilation_counter.num_models_seen}")
print(f"分片图数量: {compilation_counter.num_piecewise_graphs_seen}")
print(f"缓存命中情况: {compilation_counter.num_cache_entries_updated}")
```

**2. 通过环境变量启用详细日志**

```bash
# 启用编译相关的详细日志
export VLLM_LOGGING_LEVEL=DEBUG

# 启动vLLM服务
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --compilation-level 3
```

**3. 监控编译性能**

vLLM会自动记录编译时间和缓存命中情况：

```python
# 在日志中可以看到类似输出
# INFO: Compiling graph for shape 16, took 45.2 s
# INFO: Directly load compiled graph from cache, took 0.8 s
# INFO: CUDA graph capture for shape 32, took 2.1 s
```

## 0x9.2 调试支持

vLLM Compilation提供了不少调试功能，方便开发者理解编译过程和排查问题。

### 启用调试转储

**1. 设置调试转储目录**

```bash
# 通过环境变量设置
export VLLM_COMPILATION_DEBUG_DUMP_PATH="/tmp/vllm_debug"

# 或者在启动时指定
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --compilation-level 3 \
    --compilation-config '{"debug_dump_path": "/tmp/vllm_debug"}'
```

**2. 调试文件结构**

启用调试后，会在指定目录生成以下文件：

```bash
/tmp/vllm_debug/
├── rank_0/
│   ├── transformed_code.py          # Dynamo转换后的代码
│   ├── computation_graph.py         # 计算图转储
│   ├── inductor_output.py          # Inductor编译输出
│   └── piecewise_graphs/           # 分片图详情
│       ├── subgraph_0.py
│       ├── subgraph_1.py
│       └── ...
└── compilation_stats.json          # 编译统计信息
```


vLLM Compilation还提供了编译时间分析、内存监控等性能分析工具。常用的调试技巧包括：通过环境变量跳过特定子图编译、清除缓存强制重新编译、启用详细的CUDA Graph调试日志等。开发者可以通过检查编译配置、比较编译前后性能等方式排查问题和分析性能回归。

# 0x10. 总结

vLLM Compilation模块的主要特性包括：

1. **分片编译(PIECEWISE)**: 这是vLLM最核心的创新，通过图分割让CUDA Graph能够应用到prefill阶段，对除Attention之外的算子都能享受CUDA Graph带来的CPU Overhead减少。这就是我们在启动vLLM时看到`Capturing CUDA graphs (mixed prefill-decode, PIECEWISE)`的技术原理
2. **算子融合**: 从算子级到通信级的全方位算子fuse优化
3. **智能缓存**: 考虑多种因素的缓存键设计，确保缓存正确性
4. **装饰器系统**: 简化模型编译的用户接口
5. **Pass管理**: 模块化的优化Pass管理系统，避免在模型中重复写手工的冗余代码

PIECEWISE技术让vLLM在prefill阶段也能获得明显的性能提升，总的来说基于Torch Compile，vLLM实现了PIECEWISE CUDA Graph、算子fuse、智能缓存、装饰器系统、Pass管理等特性，让模型优化更好维护并且提升性能。

相关代码链接：
- 编译后端实现：https://github.com/vllm-project/vllm/blob/main/vllm/compilation/backends.py
- 编译器接口：https://github.com/vllm-project/vllm/blob/main/vllm/compilation/compiler_interface.py  
- 融合Pass实现：https://github.com/vllm-project/vllm/blob/main/vllm/compilation/fusion.py
- 装饰器系统：https://github.com/vllm-project/vllm/blob/main/vllm/compilation/decorators.py
- Pass管理器：https://github.com/vllm-project/vllm/blob/main/vllm/compilation/pass_manager.py
