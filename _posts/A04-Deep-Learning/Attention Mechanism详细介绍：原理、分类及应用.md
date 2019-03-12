---
typora-root-url: ../../../LevineHuang.github.io
---

注意力模型（Attention Model）被广泛使用在自然语言处理、图像识别及语音识别等各种不同类型的深度学习任务中，是深度学习技术中最值得关注与深入了解的核心技术之一。本文介绍深度学习中注意力机制的原理及关键计算机制，同时也抽象出其本质思想，并介绍注意力模型在图像及语音等领域的典型应用场景。



## 人类的注意力机制

在人工智能领域，实现人类水平的AI，需要我们对人类大脑（或动物大脑）的注意力机制有深刻而简洁的理解。从认知科学的角度理解人类大脑的注意力机制，能够为深度学习中注意力算法的设计提供思路。

### 注意力的定义

**注意力**是指人的心理活动指向和集中于某种事物的能力。

俄罗斯教育家乌申斯基指出：“ ‘注意’ 是我们心灵的惟一门户，意识中的一切，必然都要经过它才能进来。” **注意**[^ 1]是一个心理学概念，属于认知过程的一部分，是一种导致局部刺激的意识水平提高的知觉的选择性的集中。

注意可以区分为外显性注意与内隐性注意。**外显性注意**是指直接把感觉器官转向外界刺激来源的动作。**内隐性注意**是对几个可能的感觉刺激中的一个产生知觉集中的行为。内隐性注意被认为是一个对感觉全景的特定部分的信号进行增强的神经过程。例如，在阅读的时候，外显性注意转移了眼睛去读出不同的单词，但当你转移你的注意力从对词语的语义处理到字的字体和颜色，你内隐性注意就会开始运作。

### 注意力的临床表现型

[注意力](https://zh.wikipedia.org/wiki/%E6%B3%A8%E6%84%8F%E5%8A%9B)（专注力）包含了五个面向。

1. **选择性注意力**

   能够维持原有的行为或认知过程，即便遇到外界的刺激或诱惑。

   能够将注意力凝聚于某一个重要目标，而忽略其他不相干的讯息，所以就不会分心。

2. **分开性注意力**

   能够同时专注于不同的事情上、同时接收多个指令、或者同时进行好几件事情而不会搞混或忘记。

3. **转移性注意力**

   可以迅速从一件事（一个想法或点）切换到另一件事（一个想法或点），果断地处理完眼前的事物、再随时切换回去，不会迟疑不决或慌张混乱。

   能够在心灵上保有弹性，使自己能把注意力的焦点从一件事切换到另一件事，也能在"所需认知程度不同的"事情之间切换。

4. **持续性注意力**

   可以让专注力在持续一段时间且重复的事情中保持一段较长的时间，不会一下子就恍神或散漫。

5. **集中的注意力**

   能够一个一个，井然有序地应对来自视觉、听觉或触觉等外部刺激。

### 注意力的认知理论

**Part1.** **认知系统加工能力有限**--**注意的选择性**（存在瓶颈，选择/过滤机制）

**Part2.** **中枢能力或能量有限，注意有限**--**注意的分配性**（资源分配机制）

**Part3.** **视觉选择的注意理论**



## 深度学习中的注意力机制

深度学习中的注意力机制从本质上讲和人类的选择性视觉注意力机制类似，核心目标也是从众多信息中选择出对当前任务目标更关键的信息。当人类观察外界事物的时候，一般不会把事物当成一个整体去看，往往倾向于根据需要选择性的去获取被观察事物的某些重要部分，比如我们看到一个人时，往往先Attention到这个人的脸，然后再把不同区域的信息组合起来，形成一个对被观察事物的整体印象。

因此，Attention Mechanism可以帮助模型对输入的X每个部分赋予不同的权重，抽取出更加关键及重要的信息，使模型做出更加准确的判断，同时不会对模型的计算和存储带来更大的开销，这也是Attention Mechanism应用如此广泛的原因。

### **Encoder-Decoder框架**

Encoder-Decoder框架可以看作是一种深度学习领域的研究模式，应用场景异常广泛。目前大多数注意力模型附着在Encoder-Decoder框架下，当然，其实注意力模型可以看作一种通用的思想，本身并不依赖于特定框架。

Encoder-Decoder框架包括两个步骤，第一步是Encoder，将输入数据（如图像或文本）编码为一系列特征，第二步是Decoder，以编码的特征作为输入，将其解码为目标输出。Encoder和Decoder是两个独立的模型，可以采用神经网络，也可以采用其他模型。下图是文本处理领域里常用的Encoder-Decoder框架最抽象的一种表示。

![](/assets/imgs/A04/Encoder-Decoder-1.jpg)

<center>抽象的文本处理领域的Encoder-Decoder框架<center/>

文本处理领域的Encoder-Decoder框架可以这么直观地去理解：可以把它看作适合处理由一个句子（或篇章）生成另外一个句子（或篇章）的通用处理模型。对于句子对<Source,Target>，我们的目标是给定输入句子Source，期待通过Encoder-Decoder框架来生成目标句子Target。Source和Target可以是同一种语言，也可以是两种不同的语言。而Source和Target分别由各自的单词序列构成：
$$
Source = <x_1,x_2,...,x_m> \\
Target = <y_1,y_2,...,y_n>
$$
Encoder顾名思义就是对输入句子Source进行编码，将输入句子通过非线性变换转化为中间语义表示C：
$$
C = F(x_1,x_2,...,x_m)
$$
对于解码器Decoder来说，其任务是根据句子Source的中间语义表示C和之前已经生成的历史信息$y_1,y_2,...,y_{i-1}$来生成i时刻要生成的单词$y_i$:
$$
y_i = G(C,y_1,y_2,...,y_{i-1})
$$
每个$y_i$都依次这么产生，那么看起来就是整个系统根据输入句子Source生成了目标句子Target。如果Source是中文句子，Target是英文句子，那么这就是解决**机器翻译**问题的Encoder-Decoder框架；如果Source是一篇文章，Target是概括性的几句描述语句，那么这是**文本摘要**的Encoder-Decoder框架；如果Source是一句问句，Target是一句回答，那么这是**问答系统**或者**对话机器人**的Encoder-Decoder框架。

Encoder-Decoder框架不仅仅在文本领域广泛使用，在语音识别、图像处理等领域也经常使用。比如对于语音识别来说，上图所示的框架完全适用，区别无非是Encoder部分的输入是语音流，输出是对应的文本信息；而对于图像描述（Image Caption）任务来说，Encoder部分的输入是一副图片，Decoder的输出是能够描述图片语义内容的一句描述语。一般而言，文本处理和语音识别的Encoder部分通常采用RNN模型，图像处理的Encoder一般采用CNN模型。

**Attention Mechanism主要需要解决的问题**

上图展示的Encoder-Decoder框架是没有体现出“注意力模型”的。请观察下目标句子Target中每个单词的生成过程如下：
$$
\begin{align}
y_1 & = f(C) \\
y_2 & = f(C,y_1) \tag 1 \\
y_3 & = f(C,y_1,y_2)
\end{align}
$$
其中f是Decoder的非线性变换函数。从这里可以看出，在生成目标句子的单词时，不论生成哪个单词，它们使用的输入句子Source的语义编码C都是一样的，没有任何区别。

而语义编码C是由句子Source的每个单词经过Encoder编码产生的，这意味着不论是生成哪个单词，y1,y2还是y3，其实句子Source中任意单词对生成某个目标单词yi来说影响力都是相同的，这是为什么说这个模型没有体现出注意力的缘由。

如果拿机器翻译来解释这个“没有引入注意力”模型的Encoder-Decoder框架更好理解，比如输入的是英文句子：Tom chase Jerry，Encoder-Decoder框架逐步生成中文单词：“汤姆”，“追逐”，“杰瑞”。在翻译“杰瑞”这个中文单词的时候，模型里面的每个英文单词对于翻译目标单词“杰瑞”贡献是相同的，很明显这里不太合理，显然“Jerry”对于翻译成“杰瑞”更重要，但是模型是无法体现这一点的，这就是为什么说它没有引入注意力的原因。

从上面的阐述可以看出，没有引入注意力的模型存在**两个明显的问题:**

1、把输入X的所有信息有压缩到一个固定长度的隐向量Z，忽略了输入X的长度，当输入句子长度很长，特别是比训练集中最初的句子长度还长时，模型的性能急剧下降。

2、把输入X编码成一个固定的长度，对于句子中每个词都赋予相同的权重，这样做是不合理的。

上面的例子中，如果引入Attention模型的话，应该在翻译“杰瑞”的时候，体现出英文单词对于翻译当前中文单词不同的影响程度，比如给出类似下面一个概率分布值：

（Tom,0.3）(Chase,0.2) (Jerry,0.5)

每个英文单词的概率代表了翻译当前单词“杰瑞”时，注意力分配模型分配给不同英文单词的注意力大小。这对于正确翻译目标语单词肯定是有帮助的，因为引入了新的信息。

同理，目标句子中的每个单词都应该学会其对应的源语句子中单词的注意力分配概率信息。这意味着在生成每个单词yi的时候，原先都是相同的中间语义表示C会被替换成根据当前生成单词而不断变化的Ci。理解Attention模型的关键就是这里，即由固定的中间语义表示C换成了根据当前输出单词来调整成加入注意力模型的变化的Ci。

### Attention Mechanism原理

如上文所述，传统的Seq2Seq模型对输入序列X缺乏区分度，因此，2015年，Kyunghyun Cho等人在论文《Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation》中，引入了Attention Mechanism来解决这个问题，他们提出的模型结构如图所示。

![img](/assets/imgs/A04/Attention-Mechanism-1.jpg)

<center>Attention Mechanism模块图解<center/>

在该模型中，定义了一个条件概率：

$$
p(y_i|y_1,...,y_{i-1},X) = g(y_{i-1},s_i,c_i)
$$
其中，$s_i$是decoder中RNN在在i时刻的隐状态，其计算公式为：

$$
s_i = f(s_{i-1}, y_{i-1}, c_i)
$$
这里的背景向量$c_i$的计算方式，与传统的Seq2Seq模型直接累加的计算方式不一样，这里的$c_i$是一个权重化（Weighted）之后的值，其表达式如公式所示：

$$
c_i = \sum_{j= 1}^{T_x} \alpha_{ij}h_j
$$
其中，i表示encoder端的第i个词，hj表示encoder端的第j和词的隐向量，aij表示encoder端的第j个词与decoder端的第i个词之间的权值，表示源端第j个词对目标端第i个词的影响程度，aij的计算公式如公式所示：

$$
\begin{align}
\alpha & = \frac{exp(e_{ij})}{\sum_{k=1}^{T_x}exp(e_{ik})} \\
\\
e_{ij} & = a(s_{i-1}, h_j)
\end{align}
$$


在公式6中，aij是一个softmax模型输出，概率值的和为1。**eij表示一个对齐模型，用于衡量encoder端的位置j个词，对于decoder端的位置i个词的对齐程度（影响程度）**，换句话说：decoder端生成位置i的词时，有多少程度受encoder端的位置j的词影响。对齐模型eij的计算方式有很多种，不同的计算方式，代表不同的Attention模型，最简单且最常用的的对齐模型是dot product乘积矩阵，即把target端的输出隐状态==ht==与source端的输出隐状态进行矩阵乘。常见的对齐计算方式如下：

$$
\begin{align}
score(h_t, \bar h_s) = 
\begin{cases}
h_t^{\mathsf{T}} \bar h_s  &dot\\
h_t^{\mathsf{T}} W_a \bar h_s & general\\
v_a^{\mathsf{T}} tanh(W_a[h_t;\bar h_s]) & concat
\end{cases}
\end{align}
$$
其中,Score(ht,hs) = aij表示源端与目标单单词对齐程度。可见，常见的对齐关系计算方式有，点乘（Dot product），权值网络映射（general）和concat映射几种方式。

### **Attention Mechanism分类**

#### **soft Attention 和 Hard Attention**

Kelvin Xu等人与2015年发表论文《Show, Attend and Tell: Neural Image Caption Generation with Visual Attention》，在Image Caption中引入了Attention，当生成第i个关于图片内容描述的词时，用Attention来关联与i个词相关的图片的区域。Kelvin Xu等人在论文中使用了两种Attention Mechanism，即Soft Attention和Hard Attention。上文所描述的Attention Mechanism就是Soft Attention。Soft Attention是参数化的（Parameterization），因此可导，可以被嵌入到模型中去，直接训练。梯度可以经过Attention Mechanism模块，反向传播到模型其他部分。

相反，Hard Attention是一个随机的过程。Hard Attention不会选择整个encoder的输出做为其输入，Hard Attention会依概率Si来采样输入端的隐状态一部分来进行计算，而不是整个encoder的隐状态。为了实现梯度的反向传播，需要采用蒙特卡洛采样的方法来估计模块的梯度。

两种Attention Mechanism都有各自的优势，但目前更多的研究和应用还是更倾向于使用Soft Attention，因为其可以直接求导，进行梯度反向传播。

**Global Attention 和 Local Attention**

**Global Attention：**传统的Attention model一样。所有的hidden state都被用于计算Context vector 的权重，即变长的对齐向量at，其长度等于encoder端输入句子的长度。结构如图所示。

![img](/assets/imgs/A04/Global-Attention.jpg)

<center>Global Attention模型示意图<center>


在t时刻，首先基于decoder的隐状态ht和源端的隐状态hs，计算一个变长的隐对齐权值向量at，其计算公式如下：

$$
\begin{align}
a_t(s) & = align(h_t, \bar h_s) \\
& = \frac{exp(score(h_t, \bar h_s))}{\sum_{s'}exp(score(h_t, \bar h_{s'}))}

\end{align}
$$
其中，score是一个用于评价$h_t$与$h_s$之间关系的函数，即对齐函数，一般有三种计算方式，在上文中已经提到了。得到对齐向量$a_t$之后，就可以通过加权平均的方式，得到上下文向量$c_t$。

**Local Attention：**Global Attention有一个明显的缺点：每一次encoder端的所有hidden state都要参与计算，这样做计算开销会比较大，特别是当encoder的句子偏长，比如，一段话或者一篇文章，效率偏低。因此，为了提高效率，Local Attention应运而生。

Local Attention是一种介于Kelvin Xu所提出的Soft Attention和Hard Attention之间的一种Attention方式，即把两种方式结合起来。其结构如图所示。

![img](/assets/imgs/A04/Local-Attention.jpg)

<center>Local Attention模型示意图<center/>

Local Attention首先会为decoder端当前的词，预测一个source端对齐位置（aligned position）$p_t$，然后基于$p_t$选择一个窗口，用于计算背景向量$c_t$。Position pt 的计算公式如下：
$$
p_t = S·sigmoid(v_p^{\mathsf{T}}\ tanh(W_ph_t))
$$
其中，S是encoder端句子长度，$v_p$和$W_p$是模型参数。此时，对齐向量at的计算公式如下：

$$
a_t(s) = align(h_t, \bar h_s)\ exp(-\frac{(s-p_t)^2}{2\sigma_2})
$$
总之，Global Attention和Local Attention各有优劣，在实际应用中，Global Attention应用更普遍，因为local Attention需要预测一个位置向量$p_{t}$，这就带来两个问题：1、当encoder句子不是很长时，相对Global Attention，计算量并没有明显减小。2、位置向量$p_{t}$的预测并不非常准确，这就直接计算的到的local Attention的准确率。

#### **Self Attention**

谷歌推出的BERT模型在11项NLP任务中夺得SOTA结果，取得成功的一个关键因素是Transformer的强大作用。Transformer改进了RNN最被人诟病的训练慢的缺点，利用self-attention机制实现快速并行。并且Transformer可以增加到非常深的深度，充分发掘DNN模型的特性，提升模型准确率。Transformer由论文《Attention is All You Need》提出，现在是谷歌云TPU推荐的参考模型[^ 5]。

Self Attention与传统的Attention机制非常的不同：传统的Attention是基于source端和target端的隐变量（hidden state）计算Attention的，得到的结果是源端的每个词与目标端每个词之间的依赖关系。但Self Attention不同，它分别在source端和target端进行，仅与source input或者target input自身相关的Self Attention，捕捉source端或target端自身的词与词之间的依赖关系；然后再把source端的得到的self Attention加入到target端得到的Attention中，捕捉source端和target端词与词之间的依赖关系。因此，self Attention Attention比传统的Attention mechanism效果要好，主要原因之一是，传统的Attention机制忽略了源端或目标端句子中词与词之间的依赖关系，相对比，self Attention不仅可以得到源端与目标端词与词之间的依赖关系，同时还可以有效获取源端或目标端自身词与词之间的依赖关系。

##### **从宏观的视角开始**

首先将这个模型看成是一个黑箱操作。在机器翻译中，就是输入一种语言，输出另一种语言。

![img](/assets/imgs/A04/the_transformer_3.png)

那么拆开这个黑箱，可以看到它是由编码组件、解码组件和它们之间的连接组成。

![img](/assets/imgs/A04/The_transformer_encoders_decoders.png)

编码组件部分由一堆编码器（encoder）构成（论文中是将6个编码器叠在一起——数字6没有什么神奇之处，你也可以尝试其他数字）。解码组件部分也是由相同数量（与编码器对应）的解码器（decoder）组成的。

![img](/assets/imgs/A04/The_transformer_encoder_decoder_stack.png)

**所有的编码器在结构上都是相同的，但它们没有共享参数**。每个解码器都可以分解成两个子层。

![img](/assets/imgs/A04/Transformer_encoder.png)

从编码器输入的句子首先会经过一个自注意力（self-attention）层，这层帮助编码器在对每个单词编码时关注输入句子的其他单词。我们将在稍后的文章中更深入地研究自注意力。

自注意力层的输出会传递到前馈（feed-forward）神经网络中。**每个位置的单词对应的前馈神经网络都完全一样**（译注：另一种解读就是一层窗口为一个单词的一维卷积神经网络）。

![img](/assets/imgs/A04/Transformer_decoder.png)

解码器中也有编码器的自注意力（self-attention）层和前馈（feed-forward）层。除此之外，这两个层之间还有一个注意力层，**用来关注输入句子的相关部分**（和seq2seq模型的注意力作用相似）。

##### **将张量引入图景**

以上是模型的主要部分，接下来我们看一下各种向量或张量（译注：张量概念是矢量概念的推广，可以简单理解矢量是一阶张量、矩阵是二阶张量。）是怎样在模型的不同部分中，将输入转化为输出的。

像大部分NLP应用一样，我们首先将每个输入单词通过词嵌入算法转换为词向量。



![img](/assets/imgs/A04/embeddings.png)

<center>每个单词都被嵌入为512维的向量，我们用这些简单的方框来表示这些向量。<center/>

词嵌入过程只发生在最底层的编码器中。所有的编码器都有一个相同的特点，即它们接收一个向量列表，列表中的每个向量大小为512维。在底层（最开始）编码器中它就是词向量，但是在其他编码器中，它就是下一层编码器的输出（也是一个向量列表）。向量列表大小是我们可以设置的超参数——**一般是训练集中最长句子的长度**。将输入序列进行词嵌入之后，每个单词都会流经编码器中的两个子层。

![img](/assets/imgs/A04/encoder_with_tensors.png)

接下来我们看看Transformer的一个核心特性，在这里输入序列中每个位置的单词都有自己独特的路径流入编码器。在自注意力层中，这些路径之间存在依赖关系。而前馈（feed-forward）层没有这些依赖关系。**每个位置的单词对应的前馈神经网络都完全一样**。因此**在前馈（feed-forward）层时可以并行执行各种路径**。

下面将以一个更短的句子为例，看看编码器的每个子层中发生了什么。

##### **现在我们开始“编码”**

如上述已经提到的，一个编码器接收向量列表作为输入，接着将向量列表中的向量传递到自注意力层进行处理，然后传递到前馈神经网络层中，将输出结果传递到下一个编码器中。

![img](/assets/imgs/A04/encoder_with_tensors_2.png)

输入序列的每个单词都经过自编码过程。然后，他们**各自通过**(并行)前向传播神经网络——**完全相同的网络**，而每个向量都分别通过它。

##### **从宏观视角看自注意力机制**

例如，下列句子是我们想要翻译的输入句子：

> The animal didn't cross the street because it was too tired

这个“it”在这个句子是指什么呢？它指的是street还是这个animal呢？这对于人类来说是一个简单的问题，但是对于算法则不是。

当模型处理这个单词“it”的时候，自注意力机制会允许“it”与“animal”建立联系。

随着模型处理输入序列的每个单词，自注意力会关注整个输入序列的所有单词，帮助模型对本单词更好地进行编码。

如果你熟悉RNN（循环神经网络），回忆一下它是如何维持隐藏层的。RNN会将它已经处理过的前面的所有单词/向量的表示与它正在处理的当前单词/向量结合起来。而自注意力机制会将所有相关单词的理解融入到我们正在处理的单词中。

![img](/assets/imgs/A04/transformer_self-attention_visualization.png)

当我们在编码器#5（栈中最上层编码器）中编码“it”这个单词的时，注意力机制的部分会去关注“The Animal”，将它的表示的一部分编入“it”的编码中。

请务必检查Tensor2Tensor notebook ，在里面你可以下载一个Transformer模型，并用交互式可视化的方式来检验。

##### **从微观视角看自注意力机制**

首先我们了解一下如何使用向量来计算自注意力，然后来看它实怎样用矩阵来实现。

计算自注意力的第一步就是从每个编码器的输入向量（每个单词的词向量）中生成三个向量。也就是说对于每个单词，我们创造一个查询向量、一个键向量和一个值向量。这三个向量是通过词嵌入与三个权重矩阵后相乘创建的。

可以发现这些新向量在维度上比词嵌入向量更低。他们的维度是64，而词嵌入和编码器的输入/输出向量的维度是512。但实际上不强求维度更小，这只是一种基于架构上的选择，它可以使多头注意力（multi-headed attention）的大部分计算保持不变。

![img](/assets/imgs/A04/transformer_self_attention_vectors.png)

X1与$W^Q​$权重矩阵相乘得到q1, 就是与这个单词相关的查询向量。最终使得输入序列的每个单词的创建一个查询向量、一个键向量和一个值向量。

##### **什么是查询向量、键向量和值向量向量？**

它们都是有助于计算和理解注意力机制的抽象概念。请继续阅读下文的内容，你就会知道每个向量在计算注意力机制中到底扮演什么样的角色。

计算自注意力的第二步是计算得分。假设我们在为这个例子中的第一个词“Thinking”计算自注意力向量，我们需要拿输入句子中的每个单词对“Thinking”打分。这些分数决定了在编码单词“Thinking”的过程中有多重视句子的其它部分。

这些分数是通过打分单词（所有输入句子的单词）的键向量与“Thinking”的查询向量相点积来计算的。所以如果我们是处理位置最靠前的词的自注意力的话，第一个分数是q1和k1的点积，第二个分数是q1和k2的点积。

![img](/assets/imgs/A04/transformer_self_attention_score.png)



第三步和第四步是将分数除以8(8是论文中使用的键向量的维数64的平方根，这会让梯度更稳定。这里也可以使用其它值，8只是默认值)，然后通过softmax传递结果。**softmax的作用是使所有单词的分数归一化，得到的分数都是正值且和为1。**

![img](/assets/imgs/A04/006tNc79ly1g04lb73o71j30o30f6jsk.jpg)



这个softmax分数决定了每个单词对编码当下位置（“Thinking”）的贡献。显然，已经在这个位置上的单词将获得最高的softmax分数，但有时关注另一个与当前单词相关的单词也会有帮助。

第五步是将每个**值向量**乘以softmax分数(这是为了准备之后将它们求和)。这里的直觉是希望关注语义上相关的单词，并弱化不相关的单词(例如，让它们乘以0.001这样的小数)。

第六步是对加权值向量求和（译注：自注意力的另一种解释就是在编码某个单词时，就是**将所有单词的表示（值向量）进行加权求和**，而权重是通过该词的表示（键向量）与被编码词表示（查询向量）的点积并通过softmax得到。），然后即得到自注意力层在该位置的输出(在我们的例子中是对于第一个单词)。

![img](/assets/imgs/A04/self-attention-output.png)

这样自自注意力的计算就完成了。得到的向量就可以传给前馈神经网络。然而实际中，这些计算是以矩阵形式完成的，以便算得更快。那我们接下来就看看如何用矩阵实现的。

##### **通过矩阵运算实现自注意力机制**

第一步是计算查询矩阵、键矩阵和值矩阵。为此，我们将将输入句子的词嵌入装进矩阵X中，将其乘以我们训练的权重矩阵(WQ，WK，WV)。

![img](/assets/imgs/A04/self-attention-matrix-calculation.png)

x矩阵中的每一行对应于输入句子中的一个单词。我们再次看到词嵌入向量 (512，或图中的4个格子)和q/k/v向量(64，或图中的3个格子)的大小差异。

最后，由于我们处理的是矩阵，我们可以将步骤2到步骤6合并为一个公式来计算自注意力层的输出。

![img](/assets/imgs/A04/self-attention-matrix-calculation-2.png)

<center>自注意力的矩阵运算形式<center/>

**“multi-headed” attention 机制**

通过增加一种叫做“多头”注意力（“multi-headed” attention）的机制，论文进一步完善了自注意力层，并在两方面提高了注意力层的性能：

1.它扩展了模型专注于不同位置的能力。在上面的例子中，虽然每个编码都在z1中有或多或少的体现，但是它可能被实际的单词本身所支配。如果我们翻译一个句子，比如“The animal didn’t cross the street because it was too tired”，我们会想知道“it”指的是哪个词，这时模型的“多头”注意机制会起到作用。

2.它给出了**注意力层的多个“表示子空间”**（representation subspaces）。接下来我们将看到，对于“多头”注意机制，我们有多个查询/键/值权重矩阵集(Transformer使用八个注意力头，因此我们对于每个编码器/解码器有八个矩阵集合)。这些集合中的每一个都是随机初始化的，在训练之后，每个集合都被用来将输入词嵌入(或来自较低编码器/解码器的向量)投影到不同的表示子空间中。

![img](/assets/imgs/A04/transformer_attention_heads_qkv.png)

在“多头”注意机制下，我们为每个头保持独立的查询/键/值权重矩阵，从而产生不同的查询/键/值矩阵。和之前一样，我们拿X乘以$W^Q/W^K/W^V​$矩阵来产生查询/键/值矩阵。

如果我们做与上述相同的自注意力计算，只需八次不同的权重矩阵运算，我们就会得到八个不同的Z矩阵。

![img](/assets/imgs/A04/transformer_attention_heads_z.png)

这给我们带来了一点挑战。**前馈层不需要8个矩阵，它只需要一个矩阵(由每一个单词的表示向量组成)。**所以我们需要一种方法把这八个矩阵压缩成一个矩阵。那该怎么做？其实可以直接把这些矩阵拼接在一起，然后用一个附加的权重矩阵$W^O​$与它们相乘。

![img](/assets/imgs/A04/transformer_attention_heads_weight_matrix_o.png)

这几乎就是多头自注意力的全部。这确实有好多矩阵，我们试着把它们集中在一个图片中，这样可以一眼看清。

![img](/assets/imgs/A04/transformer_multi-headed_self-attention-recap.png)

既然我们已经摸到了注意力机制的这么多“头”，那么让我们重温之前的例子，看看我们在例句中编码“it”一词时，不同的注意力“头”集中在哪里：

![img](/assets/imgs/A04/transformer_self-attention_visualization_2.png)

**当我们编码“it”一词时，一个注意力头集中在“animal”上，而另一个则集中在“tired”上，从某种意义上说，模型对“it”一词的表达在某种程度上是“animal”和“tired”的代表。**

然而，如果我们把所有的attention都加到图示里，事情就更难解释了：

![img](/assets/imgs/A04/transformer_self-attention_visualization_3.png)



##### **使用位置编码表示序列的顺序**

到目前为止，我们对模型的描述缺少了一种理解输入单词顺序的方法。

为了解决这个问题，Transformer**为每个输入的词嵌入添加了一个向量。这些向量遵循模型学习到的特定模式，这有助于确定每个单词的位置，或序列中不同单词之间的距离。**这里的直觉是，将位置向量添加到词嵌入中使得它们在接下来的运算中，能够更好地表达的词与词之间的距离。

![img](/assets/imgs/A04/transformer_positional_encoding_vectors.png)

为了让模型理解单词的顺序，我们添加了位置编码向量，这些向量的值遵循特定的模式。

如果我们假设词嵌入的维数为4，则实际的位置编码如下：

![img](/assets/imgs/A04/transformer_positional_encoding_example.png)

<center>尺寸为4的迷你词嵌入位置编码实例<center/>

这个模式会是什么样子？

在下图中，每一行对应一个词向量的位置编码，所以第一行对应着输入序列的第一个词。每行包含512个值，每个值介于1和-1之间。我们已经对它们进行了颜色编码，所以图案是可见的。

![img](/assets/imgs/A04/transformer_positional_encoding_large_example.png)

20字(行)的位置编码实例，词嵌入大小为512(列)。你可以看到它从中间分裂成两半。这是因为左半部分的值由一个函数(使用正弦)生成，而右半部分由另一个函数(使用余弦)生成。然后将它们拼在一起而得到每一个位置编码向量。

原始论文里描述了位置编码的公式(第3.5节)。你可以在 get_timing_signal_1d()中看到生成位置编码的代码。**这不是唯一可能的位置编码方法。然而，它的优点是能够扩展到未知的序列长度**(例如，当我们训练出的模型需要翻译远比训练集里的句子更长的句子时)。

**残差模块**

在继续进行下去之前，我们需要提到一个编码器架构中的细节：在每个编码器中的每个子层（自注意力、前馈网络）的周围都有一个残差连接，并且都跟随着一个“层-归一化”步骤。

层-归一化步骤：

https://arxiv.org/abs/1607.06450

![img](/assets/imgs/A04/transformer_resideual_layer_norm.png)

如果我们去可视化这些向量以及这个和自注意力相关联的层-归一化操作，那么看起来就像下面这张图描述一样：

![img](/assets/imgs/A04/transformer_resideual_layer_norm_2.png)

解码器的子层也是这样样的。如果我们想象一个2 层编码-解码结构的transformer，它看起来会像下面这张图一样：

![img](/assets/imgs/A04/transformer_resideual_layer_norm_3.png)

##### **解码组件**

既然我们已经谈到了大部分编码器的概念，那么我们基本上也就知道解码器是如何工作的了。但最好还是看看解码器的细节。

编码器通过处理输入序列开启工作。**顶端编码器的输出之后会变转化为一个包含向量K（键向量）和V（值向量）的注意力向量集 。**这些向量将被每个解码器用于自身的“编码-解码注意力层”，而这些层可以帮助解码器关注输入序列哪些位置合适：

![img](/assets/imgs/A04/transformer_decoding_1.gif)

在完成编码阶段后，则开始解码阶段。解码阶段的每个步骤都会输出一个输出序列（在这个例子里，是英语翻译的句子）的元素

接下来的步骤重复了这个过程，直到到达一个特殊的终止符号，它表示transformer的解码器已经完成了它的输出。每个步骤的输出在下一个时间步被提供给底端解码器，并且就像编码器之前做的那样，这些解码器会输出它们的解码结果 。另外，就像我们对编码器的输入所做的那样，我们会嵌入并添加位置编码给那些解码器，来表示每个单词的位置。

![](/assets/imgs/A04/transformer_decoding_2.gif)

而那些**解码器中的自注意力层表现的模式与编码器不同：在解码器中，自注意力层只被允许处理输出序列中更靠前的那些位置。**在softmax步骤前，它会把后面的位置给隐去（把它们设为-inf）。

这个“编码-解码注意力层”工作方式基本就像多头自注意力层一样，只不过它是通过**在它下面的层来创造查询矩阵，并且从编码器的输出中取得键/值矩阵。**

##### **最终的线性变换和Softmax层**

解码组件最后会输出一个实数向量。我们如何把浮点数变成一个单词？这便是线性变换层要做的工作，它之后就是Softmax层。

线性变换层是一个简单的全连接神经网络，它可以把解码组件产生的向量投射到一个比它大得多的、被称作对数几率（logits）的向量里。

不妨假设我们的模型从训练集中学习一万个不同的英语单词（我们模型的“输出词表”）。因此对数几率向量为一万个单元格长度的向量——每个单元格对应某一个单词的分数。

接下来的Softmax 层便会把那些分数变成概率（都为正数、上限1.0）。概率最高的单元格被选中，并且它对应的单词被作为这个时间步的输出。

![img](/assets/imgs/A04/transformer_decoder_output_softmax.png)

这张图片从底部以解码器组件产生的输出向量开始。之后它会转化出一个输出单词。

##### **训练部分总结**

既然我们已经过了一遍完整的transformer的前向传播过程，那我们就可以直观感受一下它的训练过程。

在训练过程中，一个未经训练的模型会通过一个完全一样的前向传播。但因为我们用有标记的训练集来训练它，所以我们可以用它的输出去与真实的输出做比较。

为了把这个流程可视化，不妨假设我们的输出词汇仅仅包含六个单词：“a”, “am”, “i”, “thanks”, “student”以及 “<eos>”（end of sentence的缩写形式）。

![img](/assets/imgs/A04/vocabulary.png)

我们模型的输出词表在我们训练之前的预处理流程中就被设定好。

一旦我们定义了我们的输出词表，我们可以使用一个相同宽度的向量来表示我们词汇表中的每一个单词。这也被认为是一个one-hot 编码。所以，我们可以用下面这个向量来表示单词“am”：

![img](/assets/imgs/A04/one-hot-vocabulary-example.png)

例子：对我们输出词表的one-hot 编码

接下来我们讨论模型的损失函数——这是我们用来在训练过程中优化的标准。通过它可以训练得到一个结果尽量准确的模型。

##### **损失函数**

比如说我们正在训练模型，现在是第一步，一个简单的例子——把“merci”翻译为“thanks”。

这意味着我们想要一个表示单词“thanks”概率分布的输出。但是因为这个模型还没被训练好，所以不太可能现在就出现这个结果。

![img](/assets/imgs/A04/transformer_logits_output_and_label.png)

因为模型的参数（权重）都被随机的生成，（未经训练的）模型产生的概率分布在每个单元格/单词里都赋予了随机的数值。我们可以用真实的输出来比较它，然后用反向传播算法来略微调整所有模型的权重，生成更接近结果的输出。

你会如何比较两个概率分布呢？我们可以简单地用其中一个减去另一个。更多细节请参考交叉熵和KL散度。

交叉熵：

https://colah.github.io/posts/2015-09-Visual-Information/

KL散度：

https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained

但注意到这是一个过于简化的例子。更现实的情况是处理一个句子。例如，输入“je suis étudiant”并期望输出是“i am a student”。那我们就希望我们的模型能够成功地在这些情况下输出概率分布：

- 每个概率分布被一个以词表大小（我们的例子里是6，但现实情况通常是3000或10000）为宽度的向量所代表。
- 第一个概率分布在与“i”关联的单元格有最高的概率
- 第二个概率分布在与“am”关联的单元格有最高的概率
- 以此类推，第五个输出的分布表示“<end of sentence>”关联的单元格有最高的概率

![img](/assets/imgs/A04/output_target_probability_distributions.png)

<center>依据例子训练模型得到的目标概率分布。<center/>

在一个足够大的数据集上充分训练后，我们希望模型输出的概率分布看起来像这个样子：

![img](/assets/imgs/A04/output_trained_model_probability_distributions.png)

我们期望训练过后，模型会输出正确的翻译。当然如果这段话完全来自训练集，它并不是一个很好的评估指标（参考：交叉验证，链接https://www.youtube.com/watch?v=TIgfjmp-4BA）。注意到每个位置（词）都得到了一点概率，即使它不太可能成为那个时间步的输出——这是softmax的一个很有用的性质，它可以帮助模型训练。

因为这个模型一次只产生一个输出，不妨假设这个模型只选择概率最高的单词，并把剩下的词抛弃。这是其中一种方法（叫贪心解码）。另一个完成这个任务的方法是留住概率最靠高的两个单词（例如I和a），那么在下一步里，跑模型两次：其中一次假设第一个位置输出是单词“I”，而另一次假设第一个位置输出是单词“me”，并且无论哪个版本产生更少的误差，都保留概率最高的两个翻译结果。然后我们为第二和第三个位置重复这一步骤。这个方法被称作集束搜索（beam search）。在我们的例子中，集束宽度是2（因为保留了2个集束的结果，如第一和第二个位置），并且最终也返回两个集束的结果（top_beams也是2）。这些都是可以提前设定的参数。

##### **再进一步**

我希望通过上文已经让你们了解到Transformer的主要概念了。如果你想在这个领域深入，我建议可以走以下几步：阅读Attention Is All You Need，Transformer博客和Tensor2Tensor announcement，以及看看Łukasz Kaiser的介绍，了解模型和细节。

Attention Is All You Need：

https://arxiv.org/abs/1706.03762

Transformer博客：

https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html

Tensor2Tensor announcement：

https://ai.googleblog.com/2017/06/accelerating-deep-learning-research.html

Łukasz Kaiser的介绍： 

https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb



接下来可以研究的工作：

- Depthwise Separable Convolutions for Neural Machine Translation

  https://arxiv.org/abs/1706.03059

- One Model To Learn Them All

  https://arxiv.org/abs/1706.05137

- Discrete Autoencoders for Sequence Models

  https://arxiv.org/abs/1801.09797

- Generating Wikipedia by Summarizing Long Sequences

  https://arxiv.org/abs/1801.10198

- Image Transformer

  https://arxiv.org/abs/1802.05751

- Training Tips for the Transformer Model

  https://arxiv.org/abs/1804.00247

- Self-Attention with Relative Position Representations

  https://arxiv.org/abs/1803.02155

- Fast Decoding in Sequence Models using Discrete Latent Variables

  https://arxiv.org/abs/1803.03382

- Adafactor: Adaptive Learning Rates with Sublinear Memory Cost

  https://arxiv.org/abs/1804.04235

**Attention其他一些组合使用**

#### **Hierarchical Attention**

Zichao Yang等人在论文《Hierarchical Attention Networks for Document Classification》提出了Hierarchical Attention用于**文档分类**。Hierarchical Attention构建了两个层次的Attention Mechanism，第一个层次是对**句子中每个词**的attention，即word attention；第二个层次是针对**文档中每个句子**的attention，即sentence attention。网络结构如图10所示。

![img](/assets/imgs/A04/Hierarchical-Attention.jpg)

<center>图10 Hierarchical Attention结构示意图<center/>

整个网络结构由四个部分组成：一个由双向RNN（GRU）构成的word sequence encoder，然后是一个关于词的word-level的attention layer；基于word attention layar之上，是一个由双向RNN构成的sentence encoder，最后的输出层是一个sentence-level的attention layer。

#### **Attention over Attention**

Yiming Cui与2017年在论文《Attention-over-Attention Neural Networks for Reading Comprehension》中提出了Attention Over Attention的Attention机制，用于==？？任务==。结构如图11所示。

![img](/assets/imgs/A04/Attention-over-Attention.jpg)

<center>Attention over Attention结构示意图<center>

两个输入，一个Document和一个Query，分别用一个双向的RNN进行特征抽取，得到各自的隐状态h(doc)和h(query)，然后基于query和doc的隐状态进行dot product，得到query和doc的attention关联矩阵。然后按列(colum)方向进行softmax操作，得到query-to-document的attention 值a(t)；按照行(row)方向进行softmax操作，得到document-to-query的attention值b(t)，再按照列方向进行累加求平均得到平均后的attention值b(t)。最后再基于上一步attention操作得到a(t)和b(t)，再进行attention操作，即attention over attention得到最终query与document的关联矩阵。

#### **Multi-step Attention**

2017年，FaceBook 人工智能实验室的Jonas Gehring等人在论文《Convolutional Sequence to Sequence Learning》提出了**完全基于CNN**来构建Seq2Seq模型，除了这一最大的特色之外，论文中还采用了多层Attention Mechanism，来获取encoder和decoder中输入句子之间的关系，结构如图12所示。

![img](/assets/imgs/A04/Multi-step-Attention.jpg)

<center>Multi-step Attention结构示意图<center/>

完全基于CNN的Seq2Seq模型需要通过层叠多层来获取输入句子中词与词之间的依赖关系，特别是当句子非常长的时候，我曾经实验证明，层叠的层数往往达到10层以上才能取得比较理想的结果。针对每一个卷记得step（输入一个词）都对encoder的hidden state和decoder的hidden state进行dot product计算得到最终的Attention 矩阵，并且基于最终的attention矩阵去指导decoder的解码操作。

### Attention的应用场景

本节主要给出一些基于Attention去处理序列预测问题的例子，以下内容整理翻译自：[https://machinelearningmastery.com/attention-long-short-term-memory-recurrent-neural-networks/](https://link.zhihu.com/?target=https%3A//machinelearningmastery.com/attention-long-short-term-memory-recurrent-neural-networks/)

**1.机器翻译**

给定一个法语句子做为输入序列，翻译并输出一个英文句子做为输出序列。Attention用于关联输出序列中每个单词与输入序列中的某个特定单词的关联程度。

\- Dzmitry Bahdanau等人，《Neural machine translation by jointly learning to align and translate》，2015。

**2.图像标注（Image Caption）**

基于序列的Attention Mechanism可以应用于计算机视觉问题，以帮助理解如何最好地利用卷积神经网络来生成一段关于图片内容的描述，也称为Caption。

Kelvin Xu等人，《Attend and Tell: Neural Image Caption Generation with Visual Attention》, 2016

![img](/assets/imgs/A04/Image-Caption.jpg)

<center>基于Attention来解释，生成英文描述中某一个词时，与图片中某一区域的高度依赖关系。<center>

**3. 蕴含关系推理（Entailment Reasoning）**

给定一个用英语描述前景描述（premise scenario）和假设（hypothesis），判读假设（premise）与假设（hypothesis）的关系：矛盾，相关或包含。

例如：

前提：“一场婚礼中拍照”

假设：“有人结婚”

Attention被用来把假设中的每个单词与前提中的单词联系起来，反之亦然。

“我们提出了一个基于LSTM的神经模型，它一次读取两个句子来确定两个句子之间的蕴含关系，而不是将每个句子独立映射到一个语义空间。我们引入逐字的（word-by-word）Attention Mechanism来扩展这个模型，来强化模型对单词或短语对的关系推理能力。该模型比传统的仅基于LSTM的模型高2.6个百分点，取得了一个最高成就”

-Tim Rocktäschel，《Reasoning about Entailment with Neural Attention》, 2016

![img](/assets/imgs/A04/Reasoning.jpg)

<center>基于Attention来解释前提和假设中词与词之间的对应关系<center/>
**4. 语音识别**

给定一段英语语音片段做为输入序列，输出对应的音素序列。

Attention被用联将输出序列中的每个音素与输入序列中的特定音频帧相关联。

“基于混合Attention机制的新型端到端可训练语音识别体系结构，其结合内容和位置信息帮助选择输入序列中的下一个位置用于解码。所提出的模型的一个理想特性就是它可以识别比训练集中句子的更长的句子。”

-Jan Chorowski，《Attention-Based Models for Speech Recognition》, 2015.。

**5.文字摘要生成**

给定一篇英文文章做为输入顺序，输出一个总结英文文章注意内容的摘要句子。

Attention用于将输出摘要中的每个单词与输入文档中的特定单词相关联。

“将基于Attention的神经网络用语摘要抽取。我们将这个概率模型与可以产生准确的摘要的生成算法相结合。”

-Alexander M. Rush，《A Neural Attention Model for Abstractive Sentence Summarization》, 2015

![img](/assets/imgs/A04/Sentence-Summarization.jpg)

<center>基于Attention来解释输入Sentence与输出Summary之间单词的对应关系<center/>

## 参考文献


[^ 1]:  https://zh.wikipedia.org/wiki/%E6%B3%A8%E6%84%8F.   “维基百科”
[^ 2 ]: https://zhuanlan.zhihu.com/p/31547842   “深度学习中Attention Mechanism详细介绍：原理、分类及应用”
[^ 3 ]:  https://zhuanlan.zhihu.com/p/37601161  “深度学习中的注意力模型（2017版）”
[^ 4 ]: https://mp.weixin.qq.com/s/WDq8tUpfiKHNC6y_8pgHoA  “Attention is All You Need论文解读”
[^ 5 ]: https://jalammar.github.io/illustrated-transformer/





https://blog.csdn.net/u010165147/article/details/50964108

https://wenku.baidu.com/view/026a070402020740be1e9b48.html