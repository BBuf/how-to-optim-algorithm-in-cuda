> 博客链接：https://pytorch.org/blog/maximizing-training-throughput/

by Team PyTorch at IBM and Team PyTorch at Meta

Recently, we demonstrated how FSDP and selective activation checkpointing can be used to achieve 57% MFU (Model Flops Utilization) for training a 7B model on A100 GPUs. We also demonstrated how it can train a high quality model, which we open sourced as Granite 7B base model on Hugging Face Hub under the Apache v2.0 license.

We continued our quest to improve the utilization of GPUs by leveraging torch.compile. Using torch.compile and the selective activation checkpointing from our previous work, we achieve a MFU of 68% for the 7B model on A100 GPUs! torch.compile improves training MFU between 10% and 23% for various model sizes.

This blog is organized into three parts: (1) Challenges addressed in order to train using torch.compile, (2) Numerical parity of compile with no-compile, and (3) MFU report.

We open sourced all the code and updated it in the fms-fsdp repository. We are also working with Team PyTorch at Meta to contribute these to the newly released torch titan repository for pre-training.

