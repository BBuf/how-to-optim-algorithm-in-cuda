> 博客来源：https://leimao.github.io/blog/NVIDIA-GPU-Compute-Capability/ ，来自Lei Mao，已获得作者转载授权。后续会转载一些Lei Mao的CUDA相关Blog，也是一个完整的专栏，Blog会从稍早一些的CUDA架构到当前最新的CUDA架构，也会包含实用工程技巧，底层指令分析，Cutlass分析等等多个课题，是一个时间线十分明确的专栏。

# NVIDIA GPU 计算能力

## 简介

要查找不同NVIDIA GPU的计算能力(https://developer.nvidia.com/cuda-gpus)，我们可以访问NVIDIA CUDA GPU网页。然而，NVIDIA GPU被分散在不同的表格中，这对于快速搜索来说有些不便。

在这篇博客文章中，我将所有NVIDIA GPU及其计算能力编译到一个表格中，用户可以使用Ctrl + F来搜索特定NVIDIA GPU的计算能力。

## NVIDIA GPU 计算能力
| GPU | 类别 | 计算能力 |
|-----|----------|-------------------|
| NVIDIA Blackwell GPU (GB200) | NVIDIA 数据中心产品 | 10.0 |
| NVIDIA Blackwell GPU (B200) | NVIDIA 数据中心产品 | 10.0 |
| GeForce RTX 5090 | GeForce 和 TITAN 产品 | 10.0 |
| GeForce RTX 5080 | GeForce 和 TITAN 产品 | 10.0 |
| GeForce RTX 5090 | GeForce 笔记本产品 | 10.0 |
| GeForce RTX 5080 | GeForce 笔记本产品 | 10.0 |
| NVIDIA H200 | NVIDIA 数据中心产品 | 9.0 |
| NVIDIA H100 | NVIDIA 数据中心产品 | 9.0 |
| NVIDIA L4 | NVIDIA 数据中心产品 | 8.9 |
| NVIDIA L40S | NVIDIA 数据中心产品 | 8.9 |
| NVIDIA L40 | NVIDIA 数据中心产品 | 8.9 |
| RTX 6000 | NVIDIA Quadro 和 NVIDIA RTX 桌面GPU | 8.9 |
| GeForce RTX 4090 | GeForce 和 TITAN 产品 | 8.9 |
| GeForce RTX 4080 | GeForce 和 TITAN 产品 | 8.9 |
| GeForce RTX 4070 Ti | GeForce 和 TITAN 产品 | 8.9 |
| GeForce RTX 4060 Ti | GeForce 和 TITAN 产品 | 8.9 |
| GeForce RTX 4090 | GeForce 笔记本产品 | 8.9 |
| GeForce RTX 4080 | GeForce 笔记本产品 | 8.9 |
| GeForce RTX 4070 | GeForce 笔记本产品 | 8.9 |
| GeForce RTX 4060 | GeForce 笔记本产品 | 8.9 |
| GeForce RTX 4050 | GeForce 笔记本产品 | 8.9 |
| Jetson AGX Orin, Jetson Orin NX, Jetson Orin Nano | Jetson 产品 | 8.7 |
| NVIDIA A40 | NVIDIA 数据中心产品 | 8.6 |
| NVIDIA A10 | NVIDIA 数据中心产品 | 8.6 |
| NVIDIA A16 | NVIDIA 数据中心产品 | 8.6 |
| NVIDIA A2 | NVIDIA 数据中心产品 | 8.6 |
| RTX A6000 | NVIDIA Quadro 和 NVIDIA RTX 桌面GPU | 8.6 |
| RTX A5000 | NVIDIA Quadro 和 NVIDIA RTX 桌面GPU | 8.6 |
| RTX A4000 | NVIDIA Quadro 和 NVIDIA RTX 桌面GPU | 8.6 |
| RTX A5000 | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 8.6 |
| RTX A4000 | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 8.6 |
| RTX A3000 | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 8.6 |
| RTX A2000 | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 8.6 |
| GeForce RTX 3090 Ti | GeForce 和 TITAN 产品 | 8.6 |
| GeForce RTX 3090 | GeForce 和 TITAN 产品 | 8.6 |
| GeForce RTX 3080 Ti | GeForce 和 TITAN 产品 | 8.6 |
| GeForce RTX 3080 | GeForce 和 TITAN 产品 | 8.6 |
| GeForce RTX 3070 Ti | GeForce 和 TITAN 产品 | 8.6 |
| GeForce RTX 3070 | GeForce 和 TITAN 产品 | 8.6 |
| Geforce RTX 3060 Ti | GeForce 和 TITAN 产品 | 8.6 |
| Geforce RTX 3060 | GeForce 和 TITAN 产品 | 8.6 |
| GeForce RTX 3080 Ti | GeForce 笔记本产品 | 8.6 |
| GeForce RTX 3080 | GeForce 笔记本产品 | 8.6 |
| GeForce RTX 3070 Ti | GeForce 笔记本产品 | 8.6 |
| GeForce RTX 3070 | GeForce 笔记本产品 | 8.6 |
| GeForce RTX 3060 Ti | GeForce 笔记本产品 | 8.6 |
| GeForce RTX 3060 | GeForce 笔记本产品 | 8.6 |
| GeForce RTX 3050 Ti | GeForce 笔记本产品 | 8.6 |
| GeForce RTX 3050 | GeForce 笔记本产品 | 8.6 |
| NVIDIA A100 | NVIDIA 数据中心产品 | 8.0 |
| NVIDIA A30 | NVIDIA 数据中心产品 | 8.0 |
| NVIDIA T4 | NVIDIA 数据中心产品 | 7.5 |
| T1000 | NVIDIA Quadro 和 NVIDIA RTX 桌面GPU | 7.5 |
| T600 | NVIDIA Quadro 和 NVIDIA RTX 桌面GPU | 7.5 |
| T400 | NVIDIA Quadro 和 NVIDIA RTX 桌面GPU | 7.5 |
| Quadro RTX 8000 | NVIDIA Quadro 和 NVIDIA RTX 桌面GPU | 7.5 |
| Quadro RTX 6000 | NVIDIA Quadro 和 NVIDIA RTX 桌面GPU | 7.5 |
| Quadro RTX 5000 | NVIDIA Quadro 和 NVIDIA RTX 桌面GPU | 7.5 |
| Quadro RTX 4000 | NVIDIA Quadro 和 NVIDIA RTX 桌面GPU | 7.5 |
| RTX 5000 | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 7.5 |
| RTX 4000 | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 7.5 |
| RTX 3000 | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 7.5 |
| T2000 | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 7.5 |
| T1200 | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 7.5 |
| T1000 | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 7.5 |
| T600 | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 7.5 |
| T500 | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 7.5 |
| GeForce GTX 1650 Ti | GeForce 和 TITAN 产品 | 7.5 |
| NVIDIA TITAN RTX | GeForce 和 TITAN 产品 | 7.5 |
| Geforce RTX 2080 Ti | GeForce 和 TITAN 产品 | 7.5 |
| Geforce RTX 2080 | GeForce 和 TITAN 产品 | 7.5 |
| Geforce RTX 2070 | GeForce 和 TITAN 产品 | 7.5 |
| Geforce RTX 2060 | GeForce 和 TITAN 产品 | 7.5 |
| Geforce RTX 2080 | GeForce 笔记本产品 | 7.5 |
| Geforce RTX 2070 | GeForce 笔记本产品 | 7.5 |
| Geforce RTX 2060 | GeForce 笔记本产品 | 7.5 |
| Jetson AGX Xavier, Jetson Xavier NX | Jetson 产品 | 7.2 |
| NVIDIA V100 | NVIDIA 数据中心产品 | 7.0 |
| Quadro GV100 | NVIDIA Quadro 和 NVIDIA RTX 桌面GPU | 7.0 |
| NVIDIA TITAN V | GeForce 和 TITAN 产品 | 7.0 |
| Jetson TX2 | Jetson 产品 | 6.2 |
| Tesla P40 | NVIDIA 数据中心产品 | 6.1 |
| Tesla P4 | NVIDIA 数据中心产品 | 6.1 |
| Quadro P6000 | NVIDIA Quadro 和 NVIDIA RTX 桌面GPU | 6.1 |
| Quadro P5000 | NVIDIA Quadro 和 NVIDIA RTX 桌面GPU | 6.1 |
| Quadro P4000 | NVIDIA Quadro 和 NVIDIA RTX 桌面GPU | 6.1 |
| Quadro P2200 | NVIDIA Quadro 和 NVIDIA RTX 桌面GPU | 6.1 |
| Quadro P2000 | NVIDIA Quadro 和 NVIDIA RTX 桌面GPU | 6.1 |
| Quadro P1000 | NVIDIA Quadro 和 NVIDIA RTX 桌面GPU | 6.1 |
| Quadro P620 | NVIDIA Quadro 和 NVIDIA RTX 桌面GPU | 6.1 |
| Quadro P600 | NVIDIA Quadro 和 NVIDIA RTX 桌面GPU | 6.1 |
| Quadro P400 | NVIDIA Quadro 和 NVIDIA RTX 桌面GPU | 6.1 |
| P620 | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 6.1 |
| P520 | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 6.1 |
| Quadro P5200 | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 6.1 |
| Quadro P4200 | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 6.1 |
| Quadro P3200 | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 6.1 |
| Quadro P5000 | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 6.1 |
| Quadro P4000 | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 6.1 |
| Quadro P3000 | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 6.1 |
| Quadro P2000 | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 6.1 |
| Quadro P1000 | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 6.1 |
| Quadro P600 | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 6.1 |
| Quadro P500 | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 6.1 |
| NVIDIA TITAN Xp | GeForce 和 TITAN 产品 | 6.1 |
| NVIDIA TITAN X | GeForce 和 TITAN 产品 | 6.1 |
| GeForce GTX 1080 Ti | GeForce 和 TITAN 产品 | 6.1 |
| GeForce GTX 1080 | GeForce 和 TITAN 产品 | 6.1 |
| GeForce GTX 1070 Ti | GeForce 和 TITAN 产品 | 6.1 |
| GeForce GTX 1070 | GeForce 和 TITAN 产品 | 6.1 |
| GeForce GTX 1060 | GeForce 和 TITAN 产品 | 6.1 |
| GeForce GTX 1050 | GeForce 和 TITAN 产品 | 6.1 |
| GeForce GTX 1080 | GeForce 笔记本产品 | 6.1 |
| GeForce GTX 1070 | GeForce 笔记本产品 | 6.1 |
| GeForce GTX 1060 | GeForce 笔记本产品 | 6.1 |
| Tesla P100 | NVIDIA 数据中心产品 | 6.0 |
| Quadro GP100 | NVIDIA Quadro 和 NVIDIA RTX 桌面GPU | 6.0 |
| Jetson Nano | Jetson 产品 | 5.3 |
| Tesla M60 | NVIDIA 数据中心产品 | 5.2 |
| Tesla M40 | NVIDIA 数据中心产品 | 5.2 |
| Quadro M6000 24GB | NVIDIA Quadro 和 NVIDIA RTX 桌面GPU | 5.2 |
| Quadro M6000 | NVIDIA Quadro 和 NVIDIA RTX 桌面GPU | 5.2 |
| Quadro M5000 | NVIDIA Quadro 和 NVIDIA RTX 桌面GPU | 5.2 |
| Quadro M4000 | NVIDIA Quadro 和 NVIDIA RTX 桌面GPU | 5.2 |
| Quadro M2000 | NVIDIA Quadro 和 NVIDIA RTX 桌面GPU | 5.2 |
| Quadro M5500M | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 5.2 |
| Quadro M2200 | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 5.2 |
| Quadro M620 | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 5.2 |
| GeForce GTX TITAN X | GeForce 和 TITAN 产品 | 5.2 |
| GeForce GTX 980 Ti | GeForce 和 TITAN 产品 | 5.2 |
| GeForce GTX 980 | GeForce 和 TITAN 产品 | 5.2 |
| GeForce GTX 970 | GeForce 和 TITAN 产品 | 5.2 |
| GeForce GTX 960 | GeForce 和 TITAN 产品 | 5.2 |
| GeForce GTX 950 | GeForce 和 TITAN 产品 | 5.2 |
| GeForce GTX 980 | GeForce 笔记本产品 | 5.2 |
| GeForce GTX 980M | GeForce 笔记本产品 | 5.2 |
| GeForce GTX 970M | GeForce 笔记本产品 | 5.2 |
| GeForce GTX 965M | GeForce 笔记本产品 | 5.2 |
| GeForce 910M | GeForce 笔记本产品 | 5.2 |
| Quadro K2200 | NVIDIA Quadro 和 NVIDIA RTX 桌面GPU | 5.0 |
| Quadro K1200 | NVIDIA Quadro 和 NVIDIA RTX 桌面GPU | 5.0 |
| Quadro K620 | NVIDIA Quadro 和 NVIDIA RTX 桌面GPU | 5.0 |
| Quadro M1200 | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 5.0 |
| Quadro M520 | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 5.0 |
| Quadro M5000M | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 5.0 |
| Quadro M4000M | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 5.0 |
| Quadro M3000M | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 5.0 |
| Quadro M2000M | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 5.0 |
| Quadro M1000M | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 5.0 |
| Quadro K620M | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 5.0 |
| Quadro M600M | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 5.0 |
| Quadro M500M | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 5.0 |
| NVIDIA NVS 810 | 桌面产品 | 5.0 |
| GeForce GTX 750 Ti | GeForce 和 TITAN 产品 | 5.0 |
| GeForce GTX 750 | GeForce 和 TITAN 产品 | 5.0 |
| GeForce GTX 960M | GeForce 笔记本产品 | 5.0 |
| GeForce GTX 950M | GeForce 笔记本产品 | 5.0 |
| GeForce 940M | GeForce 笔记本产品 | 5.0 |
| GeForce 930M | GeForce 笔记本产品 | 5.0 |
| GeForce GTX 850M | GeForce 笔记本产品 | 5.0 |
| GeForce 840M | GeForce 笔记本产品 | 5.0 |
| GeForce 830M | GeForce 笔记本产品 | 5.0 |
| Tesla K80 | Tesla 工作站产品 | 3.7 |
| Tesla K80 | NVIDIA 数据中心产品 | 3.7 |
| Tesla K40 | Tesla 工作站产品 | 3.5 |
| Tesla K20 | Tesla 工作站产品 | 3.5 |
| Tesla K40 | NVIDIA 数据中心产品 | 3.5 |
| Tesla K20 | NVIDIA 数据中心产品 | 3.5 |
| Quadro K6000 | NVIDIA Quadro 和 NVIDIA RTX 桌面GPU | 3.5 |
| Quadro K5200 | NVIDIA Quadro 和 NVIDIA RTX 桌面GPU | 3.5 |
| Quadro K610M | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 3.5 |
| Quadro K510M | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 3.5 |
| GeForce GTX TITAN Z | GeForce 和 TITAN 产品 | 3.5 |
| GeForce GTX TITAN Black | GeForce 和 TITAN 产品 | 3.5 |
| GeForce GTX TITAN | GeForce 和 TITAN 产品 | 3.5 |
| GeForce GTX 780 Ti | GeForce 和 TITAN 产品 | 3.5 |
| GeForce GTX 780 | GeForce 和 TITAN 产品 | 3.5 |
| GeForce GT 730 | GeForce 和 TITAN 产品 | 3.5 |
| GeForce GT 720 | GeForce 和 TITAN 产品 | 3.5 |
| GeForce GT 705* | GeForce 和 TITAN 产品 | 3.5 |
| GeForce GT 640 (GDDR5) | GeForce 和 TITAN 产品 | 3.5 |
| GeForce 920M | GeForce 笔记本产品 | 3.5 |
| Tesla K10 | NVIDIA 数据中心产品 | 3.0 |
| Quadro K5000 | NVIDIA Quadro 和 NVIDIA RTX 桌面GPU | 3.0 |
| Quadro K4200 | NVIDIA Quadro 和 NVIDIA RTX 桌面GPU | 3.0 |
| Quadro K4000 | NVIDIA Quadro 和 NVIDIA RTX 桌面GPU | 3.0 |
| Quadro K2000 | NVIDIA Quadro 和 NVIDIA RTX 桌面GPU | 3.0 |
| Quadro K2000D | NVIDIA Quadro 和 NVIDIA RTX 桌面GPU | 3.0 |
| Quadro K600 | NVIDIA Quadro 和 NVIDIA RTX 桌面GPU | 3.0 |
| Quadro K420 | NVIDIA Quadro 和 NVIDIA RTX 桌面GPU | 3.0 |
| Quadro 410 | NVIDIA Quadro 和 NVIDIA RTX 桌面GPU | 3.0 |
| Quadro K6000M | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 3.0 |
| Quadro K5200M | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 3.0 |
| Quadro K5100M | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 3.0 |
| Quadro K500M | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 3.0 |
| Quadro K4200M | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 3.0 |
| Quadro K4100M | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 3.0 |
| Quadro K3100M | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 3.0 |
| Quadro K2200M | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 3.0 |
| Quadro K2100M | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 3.0 |
| Quadro K1100M | NVIDIA Quadro 和 NVIDIA RTX 移动GPU | 3.0 |
| NVIDIA NVS 510 | 桌面产品 | 3.0 |
| GeForce GTX 770 | GeForce 和 TITAN 产品 | 3.0 |
| GeForce GTX 760 | GeForce 和 TITAN 产品 | 3.0 |
| GeForce GTX 690 | GeForce 和 TITAN 产品 | 3.0 |
| GeForce GTX 680 | GeForce 和 TITAN 产品 | 3.0 |
| GeForce GTX 670 | GeForce 和 TITAN 产品 | 3.0 |
| GeForce GTX 660 Ti | GeForce 和 TITAN 产品 | 3.0 |
| GeForce GTX 660 | GeForce 和 TITAN 产品 | 3.0 |
| GeForce GTX 650 Ti BOOST | GeForce 和 TITAN 产品 | 3.0 |
| GeForce GTX 650 Ti | GeForce 和 TITAN 产品 | 3.0 |
| GeForce GTX 650 | GeForce 和 TITAN 产品 | 3.0 |
| GeForce GT 740 | GeForce 和 TITAN 产品 | 3.0 |
| GeForce GTX 880M | GeForce 笔记本产品 | 3.0 |
| GeForce GTX 870M | GeForce 笔记本产品 | 3.0 |
| GeForce GTX 780M | GeForce 笔记本产品 | 3.0 |
| GeForce GTX 770M | GeForce 笔记本产品 | 3.0 |
| GeForce GTX 765M | GeForce 笔记本产品 | 3.0 |
| GeForce GTX 760M | GeForce 笔记本产品 | 3.0 |
| GeForce GTX 680MX | GeForce 笔记本产品 | 3.0 |
| GeForce GTX 680M | GeForce 笔记本产品 | 3.0 |
| GeForce GTX 675MX | GeForce 笔记本产品 | 3.0 |
| GeForce GTX 670MX | GeForce 笔记本产品 | 3.0 |
| GeForce GTX 660M | GeForce 笔记本产品 | 3.0 |
| GeForce GT 755M | GeForce 笔记本产品 | 3.0 |
| GeForce GT 750M | GeForce 笔记本产品 | 3.0 |
| GeForce GT 650M | GeForce 笔记本产品 | 3.0 |
| GeForce GT 745M | GeForce 笔记本产品 | 3.0 |
| GeForce GT 645M | GeForce 笔记本产品 | 3.0 |
| GeForce GT 740M | GeForce 笔记本产品 | 3.0 |
| GeForce GT 730M | GeForce 笔记本产品 | 3.0 |
| GeForce GT 640M | GeForce 笔记本产品 | 3.0 |
| GeForce GT 640M LE | GeForce 笔记本产品 | 3.0 |
| GeForce GT 735M | GeForce 笔记本产品 | 3.0 |
| GeForce GT 730M | GeForce 笔记本产品 | 3.0 |
| NVIDIA NVS 315 | 桌面产品 | 2.1 |
| NVIDIA NVS 310 | 桌面产品 | 2.1 |
| NVS 5400M | 移动产品 | 2.1 |
| NVS 5200M | 移动产品 | 2.1 |
| NVS 4200M | 移动产品 | 2.1 |
| GeForce GTX 560 Ti | GeForce 和 TITAN 产品 | 2.1 |
| GeForce GTX 550 Ti | GeForce 和 TITAN 产品 | 2.1 |
| GeForce GTX 460 | GeForce 和 TITAN 产品 | 2.1 |
| GeForce GTS 450 | GeForce 和 TITAN 产品 | 2.1 |
| GeForce GTS 450* | GeForce 和 TITAN 产品 | 2.1 |
| GeForce GT 730 DDR3,128bit | GeForce 和 TITAN 产品 | 2.1 |
| GeForce GT 640 (GDDR3) | GeForce 和 TITAN 产品 | 2.1 |
| GeForce GT 630 | GeForce 和 TITAN 产品 | 2.1 |
| GeForce GT 620 | GeForce 和 TITAN 产品 | 2.1 |
| GeForce GT 610 | GeForce 和 TITAN 产品 | 2.1 |
| GeForce GT 520 | GeForce 和 TITAN 产品 | 2.1 |
| GeForce GT 440 | GeForce 和 TITAN 产品 | 2.1 |
| GeForce GT 440* | GeForce 和 TITAN 产品 | 2.1 |
| GeForce GT 430 | GeForce 和 TITAN 产品 | 2.1 |
| GeForce GT 430* | GeForce 和 TITAN 产品 | 2.1 |
| GeForce 820M | GeForce 笔记本产品 | 2.1 |
| GeForce 800M | GeForce 笔记本产品 | 2.1 |
| GeForce GTX 675M | GeForce 笔记本产品 | 2.1 |
| GeForce GTX 670M | GeForce 笔记本产品 | 2.1 |
| GeForce GT 635M | GeForce 笔记本产品 | 2.1 |
| GeForce GT 630M | GeForce 笔记本产品 | 2.1 |
| GeForce GT 625M | GeForce 笔记本产品 | 2.1 |
| GeForce GT 720M | GeForce 笔记本产品 | 2.1 |
| GeForce GT 620M | GeForce 笔记本产品 | 2.1 |
| GeForce 710M | GeForce 笔记本产品 | 2.1 |
| GeForce 705M | GeForce 笔记本产品 | 2.1 |
| GeForce 610M | GeForce 笔记本产品 | 2.1 |
| GeForce GTX 580M | GeForce 笔记本产品 | 2.1 |
| GeForce GTX 570M | GeForce 笔记本产品 | 2.1 |
| GeForce GTX 560M | GeForce 笔记本产品 | 2.1 |
| GeForce GT 555M | GeForce 笔记本产品 | 2.1 |
| GeForce GT 550M | GeForce 笔记本产品 | 2.1 |
| GeForce GT 540M | GeForce 笔记本产品 | 2.1 |
| GeForce GT 525M | GeForce 笔记本产品 | 2.1 |
| GeForce GT 520MX | GeForce 笔记本产品 | 2.1 |
| GeForce GT 520M | GeForce 笔记本产品 | 2.1 |
| GeForce GTX 485M | GeForce 笔记本产品 | 2.1 |
| GeForce GTX 470M | GeForce 笔记本产品 | 2.1 |
| GeForce GTX 460M | GeForce 笔记本产品 | 2.1 |
| GeForce GT 445M | GeForce 笔记本产品 | 2.1 |
| GeForce GT 435M | GeForce 笔记本产品 | 2.1 |
| GeForce GT 420M | GeForce 笔记本产品 | 2.1 |
| GeForce GT 415M | GeForce 笔记本产品 | 2.1 |
| GeForce 710M | GeForce 笔记本产品 | 2.1 |
| GeForce 410M | GeForce 笔记本产品 | 2.1 |
| Tesla C2075 | Tesla 工作站产品 | 2.0 |
| Tesla C2050/C2070 | Tesla 工作站产品 | 2.0 |
| Quadro Plex 7000 | NVIDIA Quadro 和 NVIDIA RTX 桌面GPU | 2.0 |
| GeForce GTX 590 | GeForce 和 TITAN 产品 | 2.0 |
| GeForce GTX 580 | GeForce 和 TITAN 产品 | 2.0 |
| GeForce GTX 570 | GeForce 和 TITAN 产品 | 2.0 |
| GeForce GTX 480 | GeForce 和 TITAN 产品 | 2.0 |
| GeForce GTX 470 | GeForce 和 TITAN 产品 | 2.0 |
| GeForce GTX 465 | GeForce 和 TITAN 产品 | 2.0 |
| GeForce GTX 480M | GeForce 笔记本产品 | 2.0 |
| GeForce GTX 860M | GeForce 笔记本产品 | 3.0/5.0(**) |

## 其他信息

用于生成上述表格的Python脚本可以在我的Gist(https://gist.github.com/leimao/5c5cffc01f0db8b4334ace3267ddc851)中找到。

## 参考资料

- 计算能力 - CUDA编程指南(https://docs.nvidia.com/cuda/archive/12.6.2/cuda-c-programming-guide/index.html#compute-capabilities)
- NVIDIA GPU计算能力(https://developer.nvidia.com/cuda-gpus)
