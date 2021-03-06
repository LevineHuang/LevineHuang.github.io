循环神经网络(Recurrent Neural Networks，RNNs)已经在众多自然语言处理(Natural Language Processing, NLP)中取得了巨大成功以及广泛应用。

### RNNs的基本介绍

### RNNs原理

RNN 是包含循环的网络，允许引入历史信息。RNN 可以被看做是同一神经网络的多次复制，每个神经网络模块会把消息传递给下一个。将这个循环展开如下图所示：



![img](../../assets/imgs/A04/RNN.png)

隐藏层$h_t$的计算方法是：$h_t = f(W_tx_t + U_t h_{t-1})$。RNN 会丧失学习到连接太远的信息的能力


### RNNs常用的训练算法

#### Back Propagation Through Time(BPTT)

#### Real-time Recurrent Learning(RTRL)

#### Extended Kalman Filter(EKF)

### 梯度消失问题(vanishing gradient problem) 

梯度消失问题最早由Sepp Hochreiter于1991年发现。

### Long Short-Term Memory(LSTM，长短时记忆网络)

### Clockwork RNNs(CW-RNNs，时钟频率驱动循环神经网络)





###  Language Model

to predict the probability of observing the sentence (in a given dataset) as:

\begin{aligned}  P(w_1,...,w_m) = \prod_{i=1}^{m} P(w_i \mid w_1,..., w_{i-1})  \end{aligned}

计算一个句子出现概率有什么作用s
+ 翻译
+ 语音识别
+ generative model - Because we can predict the probability of a word given the preceding words, we are able to generate new text

缺点：当前置单词很多时，RNN遇到困难，引出LSTM。

### 训练数据及其预处理
15000条评论数据

#### 分词  tokenize text
借助NLTK中的word_tokenize和sent_tokenize方法

#### 去除低频词
原因：
+ 太多的词，模型难以训练
+ 低频词没有足够的上下文样本数据

低频词的处理方式：
按词频选取前N个（如8000），其它即为低频词，用“UNKNOWN_TOKEN”代替。生产新文本后，随机取词汇库外的一个词替换“UNKNOWN_TOKEN”，或者不断生产新文本，直到生成文本中不包含“UNKNOWN_TOKEN”为止。


### 梯度消失问题 THE VANISHING GRADIENT PROBLEM
+ W矩阵初始化
+ 正则化 用ReLU代替tanh或 sigmoid激活函数】
+ 采用LSTM或GRU(Gated Recurrent Unit, LSTM的简化版本)结构


## Part4
### LSTM网络
通过gating mechanism机制解决RNNs的梯度消失问题。
plain RNNs可以看做是LSTMs的特殊形式(input gate 取1，forget gate 取0，output gate 取1)
