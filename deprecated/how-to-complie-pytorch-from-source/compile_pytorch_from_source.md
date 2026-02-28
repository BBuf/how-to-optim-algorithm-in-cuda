# pytorch编译流程(v1.13.0a0+gitba90c9f)

**编译基于pytorch20230114[最新源码](https://github.com/pytorch/pytorch/commit/ba90c9f2298433778cc6a7a2008d0299aa2911da)，参考其readme [installation](https://github.com/pytorch/pytorch#installation)小节。** 


## 1.source code

```shell
# git clone source code
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
```

## 2.conda dependencies

```shell
# Common
conda install astunparse numpy ninja pyyaml setuptools cmake cffi typing_extensions future six requests dataclasses

# Linux
conda install mkl mkl-include
# CUDA only: Add LAPACK support for the GPU if needed
conda install -c pytorch magma-cuda110  # or the magma-cuda* that matches your CUDA version from https://anaconda.org/pytorch/repo
```

## 3.compile

```shell
# set cuda nvcc path
export CMAKE_CUDA_COMPILER="/usr/local/cuda/bin/nvcc"
# set cmake path
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
# compile
python setup.py install
```

## 报错解决

编译时如果报错找不到openmp，可以考虑禁用掉 mkl-dnn 然后重新编译。

```
USE_MKLDNN=0 python3 setup.py install
```

编译完成后，`import torch` 时遭遇报错：

```python
(pytorch) zhaoluyang@oneflow-23:~/Oneflow/oneflow$ python
Python 3.8.13 (default, Mar 28 2022, 11:38:47) 
[GCC 7.5.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/zhaoluyang/anaconda3/envs/pytorch/lib/python3.8/site-packages/torch/__init__.py", line 201, in <module>
    from torch._C import *  # noqa: F403
ImportError: /home/zhaoluyang/anaconda3/envs/pytorch/lib/python3.8/site-packages/torch/lib/libtorch_cpu.so: undefined symbol: vmsLog2
```

搜索相关issue：https://github.com/pytorch/pytorch/issues/14954 发现是mkl或mkl-include版本问题，这里更新了一下mkl版本后，可以正常跑通

```shell
# 移除旧版
conda remove mkl
# 搜索可用版本
conda search mkl
#mkl                         2021.3.0    h726a3e6_557  anaconda/cloud/conda-forge
#mkl                         2021.4.0    h06a4308_640  anaconda/pkgs/main  
#mkl                         2021.4.0    h06a4308_640  pkgs/main           
#mkl                         2021.4.0    h8d4b97c_729  anaconda/cloud/conda-forge
#mkl                         2022.0.1    h06a4308_117  anaconda/pkgs/main  
#mkl                         2022.0.1    h06a4308_117  pkgs/main           
#mkl                         2022.0.1    h8d4b97c_803  anaconda/cloud/conda-forge
#mkl                         2022.1.0    h84fe81f_915  anaconda/cloud/conda-forge

# 挑选一个最新的安装
conda install mkl==2022.1.0
```

再次import torch，可正常运行。

