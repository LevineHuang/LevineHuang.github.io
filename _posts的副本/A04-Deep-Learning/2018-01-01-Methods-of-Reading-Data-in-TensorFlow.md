---
layout: post
title:  TensorFlow中数据读取方式
date:   2018-01-01 09:00:00 +0800
tags: Deep-Learning AI
categories: Deep-Learning AI
typora-root-url: ..\..
---

### 引言
Tensorflow的数据读取有三种方式：

- Preloaded data: 预加载数据
- Feeding: Python产生数据，再把数据喂给后端。
- Reading from file: 从文件中直接读取

### Preload与Feeding
#### 二者区别
+ Preload:
将数据直接内嵌到Graph中，再把Graph传入Session中运行。当数据量比较大时，Graph的传输会遇到效率问题。
+ Feeding:
用占位符替代数据，待运行的时候填充数据。

#### Preload
在设计Graph的时候，x1和x2就被定义成了两个有值的列表，在计算y的时候直接取x1和x2的值。



```python
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True #按需申请内存
```


```python
# 设计graph
x1 = tf.constant([1, 2, 3])
x2 = tf.constant([4, 5, 6])
y = tf.add(x1, x2)

with tf.Session(config=config) as sess:
    print(sess.run(y))
```

    [5 7 9]


#### Feeding
在这里x1, x2只是占位符，没有具体的值，那么运行的时候去哪取值呢？这时候就要用到sess.run()中的feed_dict参数，将Python产生的数据喂给后端，并计算y。



```python
import tensorflow as tf

x1 = tf.placeholder(tf.int16)
x2 = tf.placeholder(tf.int16)
y = tf.add(x1, x2)

# 产生数据
list1 = [1, 2, 3]
list2 = [4, 5, 6]

with tf.Session() as sess:
    print(sess.run(y, feed_dict={x1: list1, x2: list2}))
```

    [5 7 9]


### Reading From File
前两种方法很方便，但是遇到大型数据的时候就会很吃力，即使是Feeding，中间环节的增加也是不小的开销，比如数据类型转换等等。最优的方案就是在Graph定义好文件读取的方法，让TF自己去从文件中读取数据，并解码成可使用的样本集。

![](/assets/imgs/A04/AnimatedFileQueues.gif)

在上图中，首先由一个单线程把文件名堆入队列，两个Reader同时从队列中取文件名并读取数据，Decoder将读出的数据解码后堆入样本队列，最后单个或批量取出样本（图中没有展示样本出列）。我们这里通过三段代码逐步实现上图的数据流，通过shuffle参数设置，可随机读取数据。

#### 数据准备
```shell
echo -e "Alpha1,A1\nAlpha2,A2\nAlpha3,A3" > A.csv
echo -e "Bee1,B1\nBee2,B2\nBee3,B3" > B.csv
echo -e "Sea1,C1\nSea2,C2\nSea3,C3" > C.csv
cat A.csv
Alpha1,A1
Alpha2,A2
Alpha3,A3
```

#### 单个Reader，单个样本


```python
import tensorflow as tf
# 生成一个先入先出队列和一个QueueRunner
filenames = ['A.csv', 'B.csv', 'C.csv']
data_path = "./data/"
filenames = [data_path + file for file in filenames]
print(filenames)
```

    ['./data/A.csv', './data/B.csv', './data/C.csv']



```python
filename_queue = tf.train.string_input_producer(filenames, shuffle=True)
```


```python
# 定义Reader
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
```


```python
# 定义Decoder
example, label = tf.decode_csv(value, record_defaults=[['null'], ['null']])
```


```python
# 运行Graph
with tf.Session() as sess:
    coord = tf.train.Coordinator()  #创建一个协调器，管理线程
    threads = tf.train.start_queue_runners(coord=coord)  #启动QueueRunner, 此时文件名队列已经进队。
    for i in range(10):
        print(example.eval())  #取样本的时候，一个Reader先从文件名队列中取出文件名，读出数据，Decoder解析后进入样本队列。
    coord.request_stop()
    coord.join(threads)
```

#### 单个Reader，多个样本


```python
import tensorflow as tf
filenames = ['A.csv', 'B.csv', 'C.csv']
data_path = "./data/"
filenames = [data_path + file for file in filenames]

filename_queue = tf.train.string_input_producer(filenames, shuffle=False)
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
example, label = tf.decode_csv(value, record_defaults=[['null'], ['null']])
# 使用tf.train.batch()会多加了一个样本队列和一个QueueRunner。Decoder解后数据会进入这个队列，再批量出队。
# 虽然这里只有一个Reader，但可以设置多线程，相应增加线程数会提高读取速度，但并不是线程越多越好。
example_batch, label_batch = tf.train.batch([example, label], batch_size=5)
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(10):
        print(example_batch.eval())
    coord.request_stop()
    coord.join(threads)
```

    [b'Alpha1' b'Alpha2' b'Alpha3' b'Bee1' b'Bee2']
    [b'Bee3' b'Sea1' b'Sea2' b'Sea3' b'Alpha1']
    [b'Alpha2' b'Alpha3' b'Bee1' b'Bee2' b'Bee3']
    [b'Sea1' b'Sea2' b'Sea3' b'Alpha1' b'Alpha2']
    [b'Alpha3' b'Bee1' b'Bee2' b'Bee3' b'Sea1']
    [b'Sea2' b'Sea3' b'Alpha1' b'Alpha2' b'Alpha3']
    [b'Bee1' b'Bee2' b'Bee3' b'Sea1' b'Sea2']
    [b'Sea3' b'Alpha1' b'Alpha2' b'Alpha3' b'Bee1']
    [b'Bee2' b'Bee3' b'Sea1' b'Sea2' b'Sea3']
    [b'Alpha1' b'Alpha2' b'Alpha3' b'Bee1' b'Bee2']


#### 多Reader，多个样本

tf.train.batch与tf.train.shuffle_batch函数是单个Reader读取，但是可以多线程。
tf.train.batch_join与tf.train.shuffle_batch_join可设置多Reader读取，每个Reader使用一个线程。

至于两种方法的效率，单Reader时，2个线程就达到了速度的极限。多Reader时，2个Reader就达到了极限。所以并不是线程越多越快，甚至更多的线程反而会使效率下降。


```python
import tensorflow as tf
filenames = ['A.csv', 'B.csv', 'C.csv']
data_path = "./data/"
filenames = [data_path + file for file in filenames]

filename_queue = tf.train.string_input_producer(filenames, shuffle=True)
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
record_defaults = [['null'], ['null']]
example_list = [tf.decode_csv(value, record_defaults=record_defaults)
                  for _ in range(2)]  # Reader设置为2
# 使用tf.train.batch_join()，可以使用多个reader，并行读取数据。每个Reader使用一个线程。
example_batch, label_batch = tf.train.batch_join(
      example_list, batch_size=5)
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(10):
        print(example_batch.eval())
    coord.request_stop()
    coord.join(threads)
```

    [b'Sea1' b'Sea2' b'Sea3' b'Alpha1' b'Alpha2']
    [b'Alpha3' b'Bee1' b'Bee2' b'Bee3' b'Bee1']
    [b'Bee2' b'Bee3' b'Sea1' b'Sea2' b'Sea3']
    [b'Alpha1' b'Alpha2' b'Alpha3' b'Bee1' b'Bee2']
    [b'Bee3' b'Sea1' b'Sea2' b'Sea3' b'Alpha1']
    [b'Alpha2' b'Alpha3' b'Bee1' b'Bee2' b'Bee3']
    [b'Alpha1' b'Alpha2' b'Alpha3' b'Sea1' b'Sea2']
    [b'Sea3' b'Alpha1' b'Alpha2' b'Alpha3' b'Bee1']
    [b'Bee2' b'Bee3' b'Sea1' b'Sea2' b'Sea3']
    [b'Alpha1' b'Alpha2' b'Alpha3' b'Sea1' b'Sea2']


#### 迭代控制


```python
import tensorflow as tf
filenames = ['A.csv', 'B.csv', 'C.csv']
data_path = "./data/"
filenames = [data_path + file for file in filenames]

filename_queue = tf.train.string_input_producer(filenames, shuffle=False, num_epochs=2)  # num_epoch: 设置每个文件迭代数
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
record_defaults = [['null'], ['null']]
example_list = [tf.decode_csv(value, record_defaults=record_defaults) for _ in range(2)]
example_batch, label_batch = tf.train.batch_join(example_list, batch_size=5)
# init_local_op = tf.initialize_local_variables()  # deprecatd
init_local_op = tf.local_variables_initializer()

with tf.Session() as sess:
    sess.run(init_local_op)   # 初始化本地变量 
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    try:
        while not coord.should_stop():
            print(example_batch.eval())
    except tf.errors.OutOfRangeError:
        print('Epochs Complete!')
    finally:
        coord.request_stop()
    coord.join(threads)
    coord.request_stop()
    coord.join(threads)
```

    [b'Alpha1' b'Alpha2' b'Alpha3' b'Bee1' b'Bee2']
    [b'Bee3' b'Sea1' b'Sea2' b'Sea3' b'Alpha1']
    [b'Alpha2' b'Alpha3' b'Bee1' b'Bee2' b'Bee3']
    Epochs Complete!

