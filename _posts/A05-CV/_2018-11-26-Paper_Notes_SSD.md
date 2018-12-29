---
layout: post
title:  论文笔记：SSD目标检测原理
date:   2018-11-14 09:00:00 +0800
tags: Computer-Vision Object-Detection
categories: Computer-Vision Object-Detection
typora-root-url: ..\..
---



SSD 目标检测框架

基本概念

feature map cell ：将 feature map 切分成 8×88×8 或者 4×44×4 之后的一个个 **格子**

default box：每一个格子上，一系列固定大小的 box，即图中虚线所形成的一系列 boxes

![](/assets/imgs/A05/ssd_default_box)

模型设计

SSD 是基于一个前向传播 CNN 网络，产生一系列 固定大小（fixed-size） 的 bounding boxes，以及每一个 box 中包含物体实例的可能性，即 score。之后，进行一个 非极大值抑制（Non-maximum suppression） 得到最终的 predictions。

