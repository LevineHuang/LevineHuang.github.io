---
layout: post
title:  TensorFlow分布式计算原理
date:   2018-11-06 09:00:00 +0800
tags: Deep-Learning AI GPU
categories: Deep-Learning AI
typora-root-url: ..\..
---

{% include lib/mathjax.html%}

#### TensorFlow分布式计算原理

TensorFlow的并行计算形式分为单台服务器内多块GPU（简称“单机多卡”）并行和若干台多卡服务器（简称“多机多卡”）并行。

##### **单机多卡上的图内（in-graph）并行**

TensorFlow只使用1个CPU核心，即CPU0。绝大部分计算量均由GPU完成。

为了使所有GPU同时工作，需要将计算图划分成多个子图，每个子图对应一块GPU。CPU将各个子图上的计算任务指派给其对应的GPU，全程集中调度，调度过程大致如下：

- 计算图初始化：组装出计算图，给变量赋初值、生成随机数等。
- 划分子图：原则是子图之间的相互依赖尽量弱，各子图上包含的计算量尽量均衡。【实现原理是什么？】
- 作业指派：CPU将子图中的操作指派给对应GPU进行运算，少量不能在GPU上完成的图节点由CPU完成；指派步骤包括先将所需数据制到GPU上，再启动CUDA kernel来在这些数据上计算。
- 数据汇总交换：GPU使用自己显存上的数据更新图节点的值；CPU将所有子图的最新值从显存复制回CPU内存，汇总得到整个计算图的新值。然后将新数值分发给各个GPU。
- 各个GPU使用新值进下一轮计算。

##### 多机多卡上的图间（between-graph）并行

图内并行只能使用一台服务器，GPU数量有限，导致大规模训任务练动辄消耗几天几周的时间，效率低下。

但是，如果将计算图剖分成多个子图，将一个子图指派给一台服务器，那么将面临几个难题：

- 计算图不够大，难以划分出足够多的子图。例如采用100台服务器，而深度神经网络只有1个输入层（784个神经元）、1个隐层（30个神经元）、1个输出层（10个神经元），这样的计算图几乎是不可剖分的。如果按神经元的层剖分，就是3级流水线，效率不高不说，关键是只能指派给3台服务器。
- 网络通信压力大：子图之间多多少少都有一些依赖关系，即使强行剖分成N个子图，子图之间的数据交换将带来巨大的网络压力，影响整体计算效率。

图内并行时不存在这两个难题，原因有两个：

- 需要子图数量少（通常2/4个，多的8/16个），从而剖分容易；

- 在服务器内存中汇总交换数据，PCIe的带宽和延时性能都比服务器之间的网络性能好很多。

**数据并行**



**TensorFlow分布式训练的网络瓶颈分析**



#### 如何配置单机多卡训练模型

#### 如何配置多级多卡训练模型

参数共享分发

训练数据是如何分发的？

由master把数据分片分别分发到不同worker，还是全量复制到不同worker？



#### 模型压缩

最少需要传输的数据量

- 梯度压缩
- 混合精度训练
- 使用周期性的学习率能将实现网络收敛所需的 epoch 数量降低 10 倍



#### 参考文献

1. S. L. Smith, P.-J. Kindermans, C. Ying and Q. V. Le, "Don't Decay the Learning Rate,
   Increase the Batch Size," in NIPS, 2017. 
2. Z. Yao, A. Gholami, K. Keutzer and M. Mahoney, "Large batch size training of neural
   networks with adversarial training and second-order information," arXiv:1810.01021, 2018.