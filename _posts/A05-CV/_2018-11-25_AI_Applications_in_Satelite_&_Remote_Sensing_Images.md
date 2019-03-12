---
typora-root-url: ..\..
---

AI技术在卫星及无人机遥感影像分析中的应用



航天与卫星遥感技术的快速发展，正帮助各国土地管理机构更快速便捷地获取海量高精度卫星影像。但以往人工解读与分析，却成为这些卫星影像的分析瓶颈，限制了土地资源保护与执法的效率。

卫星遥感影像AI分析技术将遥感分析师从繁重的遥感影像解译劳动中解放出来，实现自然资源的全要素，高效精准、常态化、低成本监管。这将帮助全球土地资源管理部门更好地释放遥感影像的潜力，推动土地资源的保护。

产业现状

根据美国卫星工业协会（SIA）发布《2017年卫星产业状况报告》的数据，2016年全球卫星遥感行业的市场营收规模约在20亿美元左右，其中美国市场在8亿美元左右。在全部杨业中，体量不算大。但据 SIA《2017年卫星产业状况报告》的分析预测，因为新的竞争合作关系出现、数据分析类的商业智能产品增长，卫星遥感行业未来有望出现大幅增长。

![](/assets/imgs/A05/SIA_satellite.png)

### 应用案例

#### 卫星遥感影像AI分析系统，监管违章大棚及破坏农田等违法行为

在“全国大棚房问题专项整治行动 ”中，北京市规划和自然资源委员会联合达摩院与阿里云，自2018年10月，借助AI影像分析技术，实现大棚房90％的监测准确率。

传统卫星影像分析主要依赖真人肉眼比对，不仅效率低，存在错检、漏检可能，而且分析耗时长，等到人工完成遥感影像解译，再反馈给执法人员时，可能已有大片农田被破坏，难以挽回。

![img](/assets/imgs/A05/satelite_ali.webp)

<center>北京农田大棚房遥感AI自动识别<center/>

#### 基于AI卫星影像分割，预测贫民窟面积变化

https://cbsudux.github.io/Mumbai-slum-segmentation/

https://github.com/cbsudux/Mumbai-slum-segmentation

如今，全世界有超过10亿人生活在贫民窟，尤其是在一些发展中国家，贫民窟居民数在全国总人口中的占比可以高达50%以上。他们缺乏可靠的卫生服务，甚至连清洁水、电力等基本生活服务都没法得到保障。

孟买是印度人数最多、最富有的城市之一。然而，它也是世界上最大的几个贫民窟所在地 -- **Dharavi, Mankhurd-Govandi belt, Kurla-Ghatkopar belt, Dindoshi 以及 The Bhandup-Mulund 贫民窟**。孟买贫民窟的居住人数从2001年的600万增长到如今的约900万，62%的孟买人居住在非正规贫民窟中。

**数据**

从Google Earth收集了65cm分辨率(65cm per pixel)的3波段(RGB)卫星影像数据集。每张图像尺寸为1280*720。这些卫星图像覆盖了孟买的大部分地区，时间跨度是从2002年至2018年。

![](/assets/imgs/A05/dh-govandi.png)

![](/assets/imgs/A05/kurla.jpg)



**方法**

训练了一个Mask R-CNN模型来分析平民窟的面积变化。训练集513张图像；测试集97张图像。



### 应用场景

- 道路网络识别
- 土地用途识别

- 水利河道监管

- 生态环境保护

- 农业估产及农情监测

- 地质灾害防治

- 区域贫困程度预测

  机器学习专家Stefano Ermon，联合地球科学专家David Lobell （Stefano Ermon & David Lobell , 2016）训练了一个神经网络，通过卫星采集到的公路、农田以及住房等影响，能够准确预测各地的贫困程度。

- 基于卫星图像的商业智能产品及服务

  如利用卫星图像进行客流预测、期货行情预测、金融行情预测等，但这并不是当前卫星遥感行业的强项。

### 中国卫星遥感产业化

中科遥感科技集团CEO任伏虎曾分析了中国卫星遥感产业化的难题：

1. 数据保障问题

   目前虽然有很多卫星在做全球观测，但目前无法做到及时供给和商业化开放。

2. 产业化成本问题

   经济成本和使用成本较高，当前购买卫星遥感图像的成本价格从几千元/景到几万元/景不等。

3. 数据服务的挖掘问题

   对遥感数据的挖掘仍旧不够

4. 产业协同问题

   遥感产业链并没有完全形成，遥感上下游产业链非常不完整，缺少很多技术连贯支撑。

5. 数据安全及版权保护问题

   目前，在技术上很难有办法溯源追踪到每一景遥感图像的使用。

   理论上，区块链技术有可能从技术上解决这一问题。当前，国外的Monegraph、Blockai、Pixsy、TinEye、 Ascribe、SingularDTV、Mediachain 、Colu、 Proof of Existence，国内的亿书、纸贵、原本等创业公司均以希望利用区块链技术解决版权问题。如果区块链技术有可能用于卫星遥感图像，则有可能带来商业卫星遥感领域的模式变革。

### 研究团队

1. 阿里巴巴达摩院机器智能实验室深度学习团队

   ![img](/assets/imgs/A05/satelite_ali_deepglobe.webp)

   <center>达摩院在遥感影像智能分析大赛DeepGlobe夺冠<center/>

2. 商汤科技

3. Uber（优步）

4. 马里兰大学、清华大学


### 研究资源

1. DeepGlobe：全球最权威的卫星影像分析大赛。