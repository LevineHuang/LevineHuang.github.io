---
layout: post
title:  大规模分布式模型训练（01）
date:   2018-11-20 09:00:00 +0800
tags: Deep-Learning AI GPU
categories: Deep-Learning AI
typora-root-url: ..\..
---

{% include lib/mathjax.html%}

数据正以前所未有的规模生成。大规模互联网公司每天都会生成数以 TB 计的数据，这些数据都需要得到有效的分析以提取出有意义的见解 [1]。新兴的深度学习是一种用于执行这种分析的强大工具，这些算法在视觉 [2]、语言 [3] 和智能推理 [4] 领域的复杂任务上创造着当前最佳。不幸的是，这些算法需要大量数据才能有效地完成训练，而这又会耗费大量时间。第一个在 ImageNet 分类任务上取得当前最佳结果的深度学习算法在单个 GPU 上训练足足耗费了一周时间。

在大数据时代，解决存储和算力的方法是Scale out，在AI时代，Scale out也一定是发展趋势，并且大数据分析任务和AI/ML任务会共享处理设备（由于AI/ML迭代收敛和容错的特征，这两种任务未来不太可能使用统一平台）。

### **分布式训练概述**

#### 为什么要分布式训练深度学习模型 

1. 增加训练的吞吐量，缩短模型训练时间；
2. 第二个原因是针对大模型训练，大模型通常在一个设备中放不下。 

#### **分布式训练的方式**

神经网络的分布式训练可以通过两种方式实现：数据并行化（data parallelism）和模型并行化（model parallelism）以及两者结合。

![](/assets/imgs/A04/model-data-parallelism-1.png)

**数据并行化**

目标是将数据集均等地分配到系统的各个节点（node），其中每个节点都有该神经网络的一个副本及其本地的权重。每个节点都会处理该数据集的一个不同子集并更新其本地权重集。这些本地权重会在整个集群中共享，从而通过一个累积算法计算出一个新的全局权重集。这些全局权重又会被分配至所有节点，然后节点会在此基础上处理下一批数据。

**模型并行化**

通过将该模型的架构切分到不同的节点上来实现训练的分布化。

AlexNet 是使用模型并行化的最早期模型之一，其方法是将网络分摊到 2 个 GPU 上以便模型能放入内存中。

当模型架构过大以至于无法放入单台机器且该模型的某些部件可以并行化时，才能应用模型并行化。模型并行化可用在某些模型中，比如目标检测网络，这种模型的绘制边界框和类别预测部分是相互独立的。一般而言，大多数网络只可以分配到 2 个 GPU 上，这限制了可实现的可扩展数量。

**模型并行化与数据并行化结合**

假设有一个多GPU集群系统。我们可以在同一台机器上采用模型并行化（在GPU之间切分模型），在机器之间采用数据并行化。 

![](/assets/imgs/A04/model-data-parallelism-2.png)

### **分布式训练框架的组件**

#### 分布式训练算法

一个常用于分布式模型训练的算法是随机梯度下降(Stochastic Gradient Descent, SGD)，其原理参见[随机梯度下降原理]()。针对 SGD 提及的原则可以轻松地移植给其它常用的优化算法，比如 Adam、RMSProp等等。分布式 SGD 可以大致分成两类变体：异步 SGD 和同步 SGD。

**同步 SGD**

同步 SGD是一种分布式梯度下降算法。分布式网络中的节点首先在它们的本地数据批上计算出梯度，然后每个节点都将它们的梯度发送给主服务器（master server）。主服务器通过求这些梯度的平均来累积这些梯度，从而为权重更新步骤构建出新的全局梯度集。使用这些全局梯度更新每个节点的本地权重，然后每个节点开始进入下一批次数据的处理。这整个过程都类似于在单台机器上通过单个 mini-batch数据 计算前向通过和反向传播步骤，因此同步 SGD 能保证收敛。但是同步 SGD 也存在一些局限性。

- 计算较慢的那些节点成为网络的瓶颈。

  解决方案：对于那些计算较慢的节点引入备份副本，当主服务器（master server）接收到前N个机器的梯度时，就求这些梯度的平均来累积这些梯度。在副本的数据和训练收敛的时间之间存在一个权衡(trade-off)。

  > 参考Spark计算的分布式解决方案：当所有节点的N%已经完成时，对于未完成的那些节点**在空闲机器上**启动多个副本，只要一个副本完成了计算，则将其他副本kill掉。**保证数据的完备性**。

- 单点故障，**single point of failure** (**SPOF**)

  在普通的同步 SGD (vallina Synchronous SGD)算法中，所有的工作节点(worker nodes)与主节点(master node)通信交换梯度可能导致单点故障以及主节点的带宽问题。

  解决方案：1) 引入参数服务器(paremeter servers)，a tree like hierarchy，但仍然存在单点故障问题。2) peer to peer通信机制，例如 All reduce algorithm既解决了单点故障问题，也更好地利用了网络带宽。

  > 参考Hadoop集群中的second namenode 高可用(High Awailable，HA)解决机制，多个参数服务器(paremeter servers)之间也可引入HA机制

- 容错

  在普通All reduce algorithm的分布式训练中，如果单个机器失败，则需要重启。目前分布式深度学习训练框架主要利用外部方案来实现容错，例如，借助Docker、Kubernetes、Spark等进行状态管理和失败恢复。

  > 分布式训练框架本身如何解决不稳定集群中的容错和失败恢复问题呢？

**异步 SGD** 

异步 SGD 也是一种分布式梯度下降算法，允许在不同节点上使用不同的数据子集来并行地训练多个模型副本。每个模型副本都会向参数服务器请求全局权重，处理一个 mini-batch 来计算梯度并将它们发回参数服务器，然后参数服务器会据此更新全局权重。

通过降低节点之间的依赖程度而去除不同节点之间的耦合。尽管这种解耦能实现更大的并行化，但却不幸具有副作用，即稳定性和准确性要稍微差一些。

- Stale Gradients问题

  因为在异步SGD算法中更新全局模型参数时，工作节点之间没有进行同步，某些节点可能用的是落后于当前版本全局权重几个梯度更新步骤之前的模型权重在进行梯度计算，用这些stale gradient进行全局权重更新会导致模型训练收敛过程减慢或无法保证收敛。

  解决方案：

  1) combines merits of Delayed Proximal Gradient algorithm and Stochastic Variance Reduced Gradient。

  2) n-softsync protocol，不像同步SGD中那样，等收到所有worker notes(N个)的梯度后才进行全局梯度更新，而是只要收到至少N/n个工作节点的梯度就进行全局梯度更新。

  3) Delay Compensated ASGD (DC-ASGD)，利用梯度函数的Taylor展开和Hessian矩阵的近似，从理论上证明了凸和非凸优化问题的收敛性。

#### 分布式架构设计

随着数据集和深度学习模型的规模持续增长，训练模型所需的时间也不断增加，大规模分布式深度学习结合数据并行化是大幅减少训练时间的明智选择。然而，在大规模 GPU 集群上的分布式深度学习存在两大技术难题。

- 大批量训练下的收敛准确率下降；
- 在 GPU 之间进行梯度同步时的信息交流成本。

高效的分布式的架构需满足：在最优的时间内完成互连节点之间的梯度传输，节点之间的高效通信、容错。

近年来，大规模分布式深度学习架构有了快速进展：Facebook 使用 256 个Tesla P100 GPU，在 1 小时内训练完ImageNet/ResNet-50；日本 Perferred Network 公司 Chainer 团队，15 分钟训练好 ImageNet/ResNet-50 ；腾讯机智团队，6.6 分钟训练好 ImageNet/ResNet-50；索尼公司提出一种新的大规模分布式训练方法，使用多达2176个GPU，在224秒内成功训练 ImageNet/ResNet-50，并在ABCI 集群上没有明显的精度损失。

大规模分布式深度学习拓扑结构如下：

- All Reduce
- Ring All Reduce
- 递归减半或倍增（Recursive Halfing/Doubling）
- Binary Blocks 算法
- 腾讯层次Ring All Reduce

##### **常规的GPU多卡分布式计算的原理**

![img](/assets/imgs/A04/GPU-distributed.jpg)

GPU1~4卡负责网络参数的训练，每个卡上都布置了相同的深度学习网络，每个卡都分配到不同的数据的mini batch。每张卡训练结束后将网络参数同步到GPU0，也就是Reducer这张卡上，然后求这些梯度的平均来累积这些梯度，从而为权重更新步骤构建出新的全局梯度集，再分发到每张计算卡，整个流程有点像map-reduce的原理。

**存在问题**

1. 每一轮的训练迭代都需要所有卡都将数据同步完做一次Reduce才算结束。如果卡数比较少的情况下，其实影响不大，但是如果并行的卡很多的时候，就涉及到计算快的卡需要去等待计算慢的卡的情况，造成计算资源的浪费。

2. 每次迭代所有的计算GPU卡都需要针对全部的模型参数与Reduce卡进行通信，如果参数的数据量大的时候，那么这种通信开销也是非常庞大，而且这种开销会随着卡数的增加而线性增长。

##### 百度 Ring All reduce

将GPU卡的通信模式拼接成一个环形，从而减少随着卡数增加而带来的资源消耗，如下图所示：

![img](/assets/imgs/A04/gpu-ring-all-reduce.jpg)

**存在问题**

1. Ring all-reduce算法不能完全利用超过 1000 块 GPU 的极大规模集群的带宽。这是因为 [12] 中展示的网络延迟的存在，使得算法的交流成本和 GPU 的数量成正比。

##### 索尼 2D-Torus all-reduce

**批量控制**

> 前人研究表明，在训练期间通过逐渐增加批量的总规模可以减少大批量训练的不稳定性。随着训练损失曲面变得"平坦"，增加批量有助于避开局部最小值 。

通过批量控制来解决大规模批量训练不稳定的问题。以超过 32K的batch size来训练模型以减少准确率下降，训练期间采用预定的批量变化方案。

![1542329483529](/assets/imgs/A04/suoni-224s.png)

ImageNet/ResNet-50 训练的 GPU 扩展效率

> 什么是GPU的扩展效率？如何衡量？

在不显著降低准确率的前提下提升 GPU 扩展效率。使用 1088 块 Tesla V100 GPU 实现了 91.62% 的 GPU 扩展效率。

![1542329773508](/assets/imgs/A04/GPU-scaling.png)

**2D-Torus All- reduce拓扑结构**

用 2D-Torus all-reduce 方案来有效交换 GPU 上的梯度，降低梯度同步的开销。

2D-Torus all-reduce，是一种“高效的”通信拓扑结构，将 GPU 排列在逻辑 2D 网格中，并在不同方向上执行一系列集群计算，可以很好地利用上千个 GPU 的带宽。 2D-Torus 拓扑如下图所示，all-reduce 包含三个步骤：reduce-scatter、all-reduce、all-gather。首先，水平地执行 reduce-scatter。然后，竖直地执行 all-reduce。最后，水平地执行 all-together。

2D-Torus all-reduce 的交流成本更低。设 N 为集群中的 GPU 数量，X 为水平方向的 GPU 数量，Y 为竖直方向的 GPU 数量。2D-Torus all-reduce 只需执行 2(X-1) 次 GPU-to-GPU 运算。

![1542332408831](/assets/imgs/A04/2D-Torus-topology.png)

> 该方案的容错性如何？某个节点出故障会有什么影响，如何解决？
>
> 生产上如何应用？

### 参考文献

1. S. L. Smith, P.-J. Kindermans, C. Ying and Q. V. Le, "Don't Decay the Learning Rate,
   Increase the Batch Size," in NIPS, 2017. 
2. Z. Yao, A. Gholami, K. Keutzer and M. Mahoney, "Large batch size training of neural
   networks with adversarial training and second-order information," arXiv:1810.01021, 2018.