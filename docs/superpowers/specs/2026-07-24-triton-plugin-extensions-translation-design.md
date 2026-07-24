# Triton Plugin Extensions 博客译述设计

## 目标

将 PyTorch Blog《Triton Plugin Extensions: Enabling TLX and Custom Compiler Passes Out of the Box》整理为一篇可直接用于微信公众号排版的中文技术译述稿，并放入本仓库合适的 PyTorch 编译器目录。

原文地址：
https://pytorch.org/blog/triton-plugin-extensions-enabling-tlx-and-custom-compiler-passes-out-of-the-box/

## 落盘位置与标题

- 目录：`pytorch/compile/`
- 文件名：`【博客翻译·译述】Triton 插件扩展：开箱即用的 TLX 与自定义编译器 Pass.md`
- 文内一级标题与文件名保持一致。

## 内容范围

文章采用“翻译·译述”而非逐句直译：

- 保留原文的技术主线、作者与发布日期。
- 保留插件系统的动机、编译流水线 hook、三层扩展能力、逐 kernel 控制方式、TLX API、H100/MI350 实现差异、性能数据、GPU MODE 案例、安装方法和后续方向。
- 代码块只做必要的中文注释翻译，不改动 API、变量名和程序语义。
- 表格转换为公众号兼容的 Markdown 表格或短列表。
- 删除官网导航、营销页脚等与正文无关的内容。
- 不加入原文没有依据的性能结论；必要的编辑说明明确标为“译者注”。

## 中文风格

- 技术名词优先保留行业常用写法，例如 kernel、pass、dialect、lowering、codegen、persistent GEMM。
- 首次出现时用中文解释，后续保持术语一致。
- 避免逐词翻译形成的英文句法，按中文技术文章的自然节奏重组句子。
- 避免宣传口吻、机械排比、重复总结和过量粗体。
- 外部引用采用“名称（裸链接）”格式，避免 Markdown 超链接，方便微信公众号复制。

## 图片处理

原文正文包含一张 1920×1080 的主图。处理流程：

1. 下载原始图片，保留原始分辨率。
2. 使用用户 Firefox 浏览器中的现有 mdnice 登录态上传图片。
3. 将正文图片地址替换为 `https://files.mdnice.com/...` 链接。
4. 检查图床链接可以直接访问，Markdown 中保留准确的中文 alt 文本。

## 验证

- 对照原文逐节核对标题、数字、硬件型号、API 和代码。
- 检查 Markdown 标题层级、代码围栏、表格和裸链接格式。
- 确认正文不存在原站图片链接。
- 确认所有正文图片均使用 mdnice 图床。
- 运行文本扫描，检查占位符、重复段落和明显的 AI 写作模式。
- `git diff --check` 通过，最终只提交本次新增的设计、计划和译述文章。
