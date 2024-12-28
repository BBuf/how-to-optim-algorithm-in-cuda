> 博客来源：https://pytorch.org/blog/deploying-llms-torchserve-vllm/

# Deploying LLMs with TorchServe + vLLM

> by Matthias Reso, Ankith Gunapal, Simon Mo, Li Ning, Hamid Shojanazeri 

The vLLM engine is currently one of the top-performing ways to execute large language models (LLM). It provides the vllm serve command as an easy option to deploy a model on a single machine. While this is convenient, to serve these LLMs in production and at scale some advanced features are necessary.

