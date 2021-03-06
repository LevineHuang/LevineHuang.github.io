---
layout: post
title:  计算机视觉概述
date:   2018-10-22 09:00:00 +0800
tags: Computer-Vision
categories: Computer-Vision
typora-root-url: ..
---

介绍计算机视觉的定义、主要的研究目标与挑战及其应用场景。

### 计算机视觉的定义

计算机视觉是一门研究如何使机器“看”的科学，更进一步的说，就是指用**摄影机**和**计算机**代替人眼对目标进行**识别、跟踪和测量**等机器视觉的应用。主要用于**模拟人类视觉的优越能力**和**弥补人类视觉的缺陷**。

- 模拟人类视觉的优越能力： 
  - 识别人、物体、场景
  - 估计立体空间、距离
  - 躲避障碍物进行导航
  - 想象并描述故事
  - 理解并讲解图片
- 弥补人类视觉的缺陷： 
  - 关注显著内容、容易忽略很多细节，不擅长精细感知
  - 描述主观、模棱两可
  - 不擅长长时间稳定的执行同一任务

### 计算机视觉的主要目标

解决“像素值”与“语义”之间的差距（gap）。

### 计算机视觉的两个主要研究维度

- **语义感知**(Semantic)

  - 图像分类(Image Classification) ：给出属于某类概率的多少？

    1. 通用图像分类

       - 对象分类 object categorization

       - 场景分类 scene classification

       - 事件分类 event classification

       - 属性分类
         - 情感分类 emotion classification

    2. 细粒度图像分类(Fine-Gained Image Classification)

       - 强监督的细粒度图像分类：指在建模训练的时候，除了图像的类别标签外，还使用了标注框、局部区域位置等额外的人工标注信息。
       - 弱监督的细粒度图像分类：仅依赖于类别标签完成分类。

  -  图像检测(Image Detection) ：对应目标在哪儿问题？用矩形框框出目标

  -  图像识别(Image Identification) 

     - 物：车牌、字符等
     - 人：人脸、虹膜、指纹、步态、行为等

  -  图像分割(Image Segmentation) 

     - 语义分割
     - 实例分割

  -  图像描述(Image Captioning) 

  -  图像问答(Image Question Answering) 

  -  图像生成(Image Generation) 

  - 图像检索(Content-based Image Retrieval)

    以文搜图、以图搜图、图文联搜，找出语义或图像相似的图片

- **几何属性**(Geometry) 

  - 3D建模
  - 双目视觉
  - 增强现实

### 计算机视觉的主要研究挑战

- 视角变化、光照变化、尺度变化、形态变化
- 背景混淆干扰、遮挡、类内物体的外观差异 

### 计算机视觉相关应用

- 特效：形状和动作捕获
- 3D城市建模
  - Microsoft Photosynth
- 脸部检测
  - 目前数码相机都有检测脸部功能，像Canon， Sony， Fuji，…
- 微笑检测
- 脸部识别
  - Apple iPhoto software
- 生物计量学（Biometrics）
- 光学字符识别（OCR）
  - 转换扫描的文件为文本技术，若有一台扫描仪，则很可能其中就有OCR软件
- 计算机视觉相关的玩具和机器人
- 移动可视化搜索
  - 像Google Goggles， iPhone Apps
- 汽车安全
- 超市中的计算机视觉
  - 物品价格扫描识别， 物品检测
- 基于视觉的交互（和游戏）
- 增强现实（Augmented Reality， AR)
- 虚拟现实（Virtual Reality， VR）
- 视觉用于机器人，太空探索

