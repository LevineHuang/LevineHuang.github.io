---
layout: post
title:  深度学习中GPU的使用
date:   2018-11-06 09:00:00 +0800
tags: Deep-Learning AI GPU
categories: Deep-Learning AI
typora-root-url: ..\..
---

{% include lib/mathjax.html%}



### 自动选择GPU

https://www.leiphone.com/news/201708/eR3oMtztM4vH2bHg.html

### 如何进行分布式模型训练

为什么要分布式训练深度学习模型 

1. 增加训练的吞吐量，缩短模型训练时间；
2. 第二个原因是针对大模型训练，大模型通常在一个设备中放不下。 

分布式训练模型的方法

1. 模型并行化（ model parallelism ）

   分布式系统中的不同机器负责单个网络模型的不同部分 —— 例如，神经网络模型的不同网络层被分配到不同的机器。 

2. 数据并行化（ data parallelism ）

   不同的机器有同一个模型的多个副本，每个机器分配到数据的一部分，然后将所有机器的计算结果按照某种方式合并。

   ![](/assets/imgs/A04/model-data-parallelism-1.png) 

3. 模型并行化与数据并行化结合

   假设有一个多GPU集群系统。我们可以在同一台机器上采用模型并行化（在GPU之间切分模型），在机器之间采用数据并行化。 

   ![](/assets/imgs/A04/model-data-parallelism-2.png)



#### TensorFlow分布式计算原理



#### 如何配置单机多卡训练模型

#### 如何配置多级多卡训练模型

参数共享分发

训练数据是如何分发的？

由master把数据分片分别分发到不同worker，还是全量复制到不同worker？