---
layout: post
title:  TensorFlow中如何指定GPU设备、分配显存？
date:   2018-11-06 09:00:00 +0800
tags: Deep-Learning AI GPU
categories: Deep-Learning AI
typora-root-url: ..\..
---

TensorFlow如何为运算(operations)分配设备(GPU，CPU)，如何手动改变设备分配编排？

### 指定使用哪块GPU卡

**TensorFlow默认分配**

如果TensorFlow中的一个运算(operations)同时具有CPU、GPU两种实现，TensorFlow将会优先为其分配GPU设备。

**tf.device(device_name)设置**

通过设置tf.device(device_name)，指定使用哪个设备。当指定的设备不存在时，可通过设置allow_soft_placement =True来让TensorFlow自动选择一个可用的设备来运行。

```python
# Creates a graph.
with tf.device('/device:GPU:2'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    
c = tf.matmul(a, b) # 未指定设备，将有限在GPU上执行
# Creates a session with log_device_placement set to True.
# 当指定的设备不存在时，可通过设置allow_soft_placement =True来让TensorFlow自动选择一个可用的设备来运行。
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True，allow_soft_placement=True))
# Runs the op.
print(sess.run(c))
```

**在终端执行程序时指定GPU** 

```sh
 CUDA_VISIBLE_DEVICES=1   python  your_file.py
```

这样在跑网络之前，告诉程序只能看到1号GPU，其他的GPU它不可见

可用的形式如下：

```sh
CUDA_VISIBLE_DEVICES=1           Only device 1 will be seen
CUDA_VISIBLE_DEVICES=0,1         Devices 0 and 1 will be visible
CUDA_VISIBLE_DEVICES="0,1"       Same as above, quotation marks are optional
CUDA_VISIBLE_DEVICES=0,2,3       Devices 0, 2, 3 will be visible; device 1 is masked
CUDA_VISIBLE_DEVICES=""          No GPU will be visible
```

**在Python代码中指定GPU**

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

### 设置最小的GPU使用量

```python
# 方式一：
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config, ...) # 使用allow_growth option，刚一开始分配少量的GPU容量，然后按需慢慢的增加。注意不会释放内存，因为会导致更严重的内存碎片

# 方式二：
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
```

### 设置定量的GPU使用量

```python
# 方式一：
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4 # 仅分配每块GPU卡40%的显存供使用，避免资源被独占
session = tf.Session(config=config, ...)

# 方式二：
gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config=tf.ConfigProto(gpu_options=gpu_options)
session = tf.Session(config=config, ...)
```

### 使用多个GPU

如果想要在多个GPU上运行TensorFlow，则可以采用多塔式方式(multi-tower fashion) 构建模型，其中每个塔都分配给不同的GPU。

```python
# Creates a graph.
c = []
for d in ['/device:GPU:2', '/device:GPU:3']:
  with tf.device(d):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
    c.append(tf.matmul(a, b))
with tf.device('/cpu:0'):
  sum = tf.add_n(c)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(sum))
```

### 查看设备GPU的可用状态

**shell命令方式**

 nvidia-smi： NVIDIA System Management Interface 

**python接口方式**

基于C语言的底层函数库，提供python封装的接口(http://pypi.python.org/pypi/nvidia-ml-py/)。

```python
try:
    nvmlDeviceGetCount()
except NVMLError as error:
    print(error)
    
>>> nvmlInit()
>>> handle = nvmlDeviceGetHandleByIndex(0)
>>> (current, pending) = nvmlDeviceGetEccMode(handle)

>>> info = nvmlDeviceGetMemoryInfo(handle)
>>> print "Total memory:", info.total
Total memory: 5636292608
>>> print "Free memory:", info.free
Free memory: 5578420224
>>> print "Used memory:", info.used
Used memory: 57872384

```

### 小结

通过with语句指定使用哪个GPU，存在一些问题：

1. 在写训练脚本时怎么知道哪个GPU是空闲可用的？
2. 同组的人做实验和我冲突怎么办？
3. 将来某个时刻运行这个脚本的时候是不是还要根据情况修改？
4. 同行用我的代码复现实验，GPU配置环境不一样，他们甚至可能没有GPU，又要改代码？

通过nvidia-smi（ NVIDIA System Management Interface）命令可以查询显卡信息，查看GPU显存、温度、功率使用，然后选择合适的GPU。每次训练前执行这个命令，再与团队保持良好的沟通可以解决上述1、2两个问题，但是3、4两个问题还是不好解决。

因此，**需要一种解决方案，能够实现不修改脚本、不需要和组员沟通，自动选择空闲GPU设备。**