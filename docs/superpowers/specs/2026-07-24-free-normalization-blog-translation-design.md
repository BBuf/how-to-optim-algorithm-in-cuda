# Free Normalization 博客译述设计

## 目标

将 PyTorch Blog《Towards Free Normalization: Fusing Normalization into GEMM and Attention Kernels》整理为可直接用于微信公众号的中文技术译述稿，转存全部正文图片到 mdnice，并在完成检查后提交、推送到 GitHub。

原文地址：
https://pytorch.org/blog/towards-free-normalization-fusing-normalization-into-gemm-and-attention-kernels/

## 落盘位置与标题

- 目录：`pytorch/kernels/`
- 文件名：`【博客翻译·译述】让归一化接近“免费”：把 Norm 融合进 GEMM 与 Attention Kernel.md`
- 一级标题与文件名保持一致。
- 文章头部保留原文标题、原文裸链接、作者和译述说明，不写发布时间。

## 内容结构

译述保留以下技术主线：

1. LayerNorm/RMSNorm 的 memory-bound 特征，以及它们在 Kunlun 和典型 LLM 中的延迟占比。
2. 普通 GEMM 与 row-wise normalization 的 tiling 冲突，及强制 `tile_n = N` 的局限。
3. Lazy Pre-Norm 的等式变换、K-loop 中并行累积平方和、适用范围和限制。
4. Multi-CTA Norm 借助 CTA cluster 与 DSMEM 完成跨 CTA reduction 的方法。
5. forward/backward fusion regrouping：前向和反向分别选择不同的相邻计算密集算子，保持 epilogue fusion。
6. FlashNormAttention 对 GDPA 周围 LayerNorm、RMSNorm 和 residual 的融合，以及内存复用、TMEM accumulate、register subtiling、warp specialization、重计算和 TMA_REDUCE_ADD。
7. B200、bfloat16、750 W power cap 下的 benchmark 口径与原文给出的最高加速。
8. 开源代码、致谢和参考资料。

正文不是逐句翻译。代码只保留理解算法所需的短伪代码和公式，性能结论必须标明比较口径。

## 图片处理

原文正文共有 15 张图片：

- 1 张 1920×1080 封面。
- 14 张算法示意图、pipeline 图和 benchmark 图。

处理要求：

- 下载原始分辨率图片。
- 使用 Firefox 当前 mdnice 登录态调用真实上传接口。
- 每张图片替换为独立的 `https://files.mdnice.com/user/59/...png` 链接。
- 逐个验证 HTTP 200 和 `Content-Type: image/png`。
- 图片按原文出现顺序插入相应章节，并补充简洁准确的中文说明。
- 正文不得残留 `pytorch.org/wp-content/uploads` 图片地址。

## 公众号格式

- 普通外部链接写成“名称（裸链接）”，不使用 Markdown 超链接。
- mdnice 图片保留 Markdown 图片语法。
- 代码、公式、标题和列表以 mdnice 可正常渲染为准。
- 技术名词保留常见写法：kernel、GEMM、CTA、DSMEM、SMEM、TMEM、warp specialization、epilogue/prologue fusion。
- 避免宣传性语言、机械排比、否定式金句和重复总结。

## 验证与发布

- 对照 PyTorch Blog、开源实现和引用资料核对数字、shape、公式与硬件约束。
- 扫描图片来源、Markdown 链接、代码围栏、标题层级、尾随空格和 AI 写作模式。
- 运行 `git diff --check`。
- 只暂存本次设计、计划和文章，不处理仓库中已有未跟踪文件。
- 提交到当前 `master` 后先 fetch 远端，确认无分叉，再 push 到 `origin/master`。

