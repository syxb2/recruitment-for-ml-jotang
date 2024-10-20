# Task3.0

> 论文代码 github 地址：[https://github.com/google-research/vision_transformer](https://github.com/google-research/vision_transformer)

## Overall understanding of transformer

### 0、什么是 Transformer

是一种可以根据上文预测下一个要输出内容的模型。

> [https://www.bilibili.com/video/BV13z421U7cs](https://www.bilibili.com/video/BV13z421U7cs)

### 1、嵌入向量（Embedding Vector， E）

嵌入向量是将离散的词或符号（如文本中的单词）映射到一个连续的、低维的向量空间中的表示。在深度学习中，模型不能直接处理文本或符号，所以需要将这些离散的数据转化为可以操作的向量。

假设有词汇表["cat", "dog", "apple", "orange"]，每个词会被映射到一个向量，如：

* cat -> [0.2, 0.5, -0.1]
* dog -> [0.3, 0.4, -0.2]
* apple -> [0.6, -0.1, 0.9]
* orange -> [0.5, -0.2, 0.8]

参考：[https://transformers.run/c1/attention/](https://transformers.run/c1/attention/)

### 2、嵌入矩阵（Embedding matrix，$W_E$）

在自然语言处理模型中，嵌入矩阵包含几乎词典中所有词汇的嵌入向量。其中，每一列代表一个单词/一个 token，每一行代表一个维度。它的初始值随机，并可以通过训练不断学习更改。但一般训练完成后，嵌入矩阵不会再改变。

以 GPT-3 为例，它的嵌入矩阵有 50257 个 tokens，有 12288 个维度。

嵌入矩阵中还隐含了 token 的位置信息，位置信息隐含 token 的语义。

注意：在训练过程中，更高效的方法是利用最终层的每一个向量，**同时**对紧随着这个向量的词进行预测。

### 3、Tranformer 输入

在输入矩阵中，token 向量可以被上下文逐渐改变，以结合上下文语境改变含义，使这些词可以获得比单个词更丰富的含义。

如初始的输入，从 embedding 矩阵复制下来的 king 的嵌入向量， 被输入中其他单词如 lived in Scotland、murdered predecessor 等上下文影响，更改权重，它的意思就变为了 A king who lived in Scotland、A king who murdered predecessor。

再比如：E(aunt) - E(uncle) = E(woman) - E(man) ==> E(aunt) = E(uncle) + E(woman) - E(man)

* 那么 E(aunt) - E(uncle) 就是 “女性” 这个语义 方向的嵌入。

Transformer 的目标就是通过逐步调整（为向量增加方向）这些向量，使一个嵌入向量可以蕴含多种语义（上下文）。

### 4、解嵌入矩阵（Unembedding matri，$W_U$）

即嵌入矩阵的转置，它将预测的嵌入向量通过矩阵乘法，softmax（将 logits 输出转换为概率） 转换为预测的 token。

![](https://i0.hdslb.com/bfs/article/73f80d946ef6f6bf345192daff91c96824550465.png)

## Attention

> [https://www.bilibili.com/video/BV1TZ421j7Ke](https://www.bilibili.com/video/BV1TZ421j7Ke)

我们以 mole 这个词为例，看下面三个句子：

* American shrew moles.
* One mole of any substance contains 6.02 x 10^23 atoms.
* Take a biopsy of the mole.

显然，这三个句子中 mole 的意思是不同的，但它经过嵌入矩阵（查找表）转化之后，它们的嵌入向量都是相同的。经过 Attention，周围的信息才可以传入这个向量：多维嵌入空间有多个方向，编码了 mole 一词不同的含义，而训练的好的 Attention 模块能计算出需要给初始向量添加什么向量，才能把它移动到上下文对应的方向上

![](https://i0.hdslb.com/bfs/article/bcf3adf07bee13548898e81ac8955e5f24550465.png)

Attention 模块还允许模型相互传递嵌入向量蕴含的信息。且预测（分类）下一个 token 的计算过程完全基于序列中的最后一个向量。

如一本推理小说，最后一个词是凶手的名字。模型在预测时，会将上文所有的信息传递到最后一个词中。

### 目标

将输入的嵌入向量 E，通过一系列计算，产生一组新的、更为准确的嵌入向量（初始嵌入向量中只包含对应 token 的信息，而计算后的一个嵌入向量可以包含多个 tokens 的信息）

### Single-Head Attention

![](https://i0.hdslb.com/bfs/article/f01938a16ee31bf7a314c508a7c13dae24550465.png)

![](https://i0.hdslb.com/bfs/article/6cdc8dab7c1770e0bce2d2741643fe1924550465.png)

* $d_k$ 是 Q、K token 的嵌入向量的长度
* $d_v$ 是 V 中转化为的嵌入向量的长度

我们以“形容词更新名词”/“名词注意形容词”为例。（实际上一个 Single-Head Attention 所表示的实际含义往往不是特别明显，有时甚至难以理解）

#### Query vectors

可以认为：一个查询向量 Q 代表一个问题。

有可学习的 Query 矩阵 $W_Q$，我们计算查询向量的方法是：$Q = W_Q * E$。$W_Q$ 的大小为 X(128) * D(12288)（D 是 E 的维度，X 一般相对较小）

相当于将 Embedding 空间的高维向量 E 映射到低维 Query 空间，用这个低维向量编码“要查询XXX”的概念

![](https://i0.hdslb.com/bfs/article/db4424842817bf6d4a0409b4a33bfb5a24550465.png)

#### Key vectors

和 Query 一样，Key 矩阵也来自一个 Key 矩阵 $W_K$

可以把 Key 视为想要回答的查询

和 Q 一样，Key matrix 也会将 E 映射到一个低维 Key 空间（通常与 Query 的维度(128)相同（行数相同））。当 Query 向量和 Key 向量对齐时，我们就说他们相互匹配。

![](https://i0.hdslb.com/bfs/article/e94daf51050136bbb9449393510d6c6024550465.png)

为了衡量每个 Key 和每个 Query 的匹配程度，要计算所有可能的 Key-Query 向量对之间的点积

![](https://i0.hdslb.com/bfs/article/a083b2e6e7fd983abc4a1755abcb563e24550465.png)

对于我们这个例子，Q_creatrue 和 K_blue、K_fluffy 高度对齐（点积相对更大）。

```
术语为：fluffy 和 blue 的嵌入(embedding) 注意到了(attend to) creaeture 的嵌入
```

之后再分别对 $Q*K$ 的每一列 softmax，使之转化为`权重`

![](https://i0.hdslb.com/bfs/article/9d5bea97fb47580f0c28d3a10411b92824550465.png)

得到的这个矩阵，表示每个 Key 和每个 Query 的相关度，我们称这个矩阵为`Attention Partten`（注意力模式）(大小为 token_num^2)

在实际训练中，在 softmax 之前我们会给每个点积除以 Key-Query Space 维度的平方根，防止点积过大。

并且我们会充分利用每一个训练样本，对一个样本而言，我们会：

![](https://i0.hdslb.com/bfs/article/de32c34c6de3054cf0eb736b957f00d524550465.png)

这就要求我们不能泄漏后词（不能让后面的单词影响前面的单词），**也就是 Query 的结果 Key 只能是 Query 前面的词。（对 creature 来说只有它前面两个形容词 blue 和 fluffy）

只要在 softmax 之前将矩阵左下方的值设为 -∞（权重为 0）即可。这个过程称为 `masking`。

#### Value vectors

和 Query 一样，Value 向量也来自一个 Value 矩阵 $W_V$。V 向量可以看成是和 K 向量相关联。

但 Value 矩阵不会改变 E 的维度。所以 Value 矩阵的大小为 E 维度的平方(12288^2)。

![](https://i0.hdslb.com/bfs/article/1eac5d84fd351e296459c3df86e258d524550465.png)

对于网络中的每一列，给每个 Value vector 乘上对应的权重，再将它们相加，得到的就是 ΔE 向量。E 和 ΔE 相加，即为更新之后的 embedding matrix。

实际上，更高效的做法是令：

```
Value matrix para_num = Query matrix para_num + Key matrix para_num
```

我们可以将 Value matrix 看作两个矩阵相乘而来（即对大矩阵进行低秩分解）：

$$
W_V = Value_{up} * Value_{down}
$$

![](https://i0.hdslb.com/bfs/article/5288e2bc2fce75ddc72ef7491014e51224550465.png)

### Multi-headed attention

![](https://i0.hdslb.com/bfs/article/9e6d526f5c29caff61c5677434c9095e24550465.png)

每个头关注一个“前词对本词的影响方式”也可以说是“根据上下文来改变 token 语义的一个方式”

![](https://i0.hdslb.com/bfs/article/e582c0b3cdc91f6e4b44723f0b098a7e24550465.png)

* 每个头都有单独的 $W_Q$、$W_K$、$W_V$

最后将每个头输出的 ΔE 都加到 E 上。

值得注意的是，在实际的应用中，常常将多个头的 $Value_{up}$ 拼接合并为一个 Output Matrix，而每个头中的 Value map 单指 $Value_{down}$ 矩阵：

![](https://i0.hdslb.com/bfs/article/ba03679d23a1cdaefbb3cbb1e784f29024550465.png)

## 2、以一段 pytorch 实现 Scaled Dot-product Attention 的代码为例，简要说明 scaled dot-product attention 的计算过程

> [【点击跳转】scaled_dot_product_attention.ipynb](./scaled_dot_product_attention.ipynb)

<!-- 不得不说，3b1b 真是神 -->

# Task3.1

## The Paper

这是本论文提出的模型结构：

![](https://i0.hdslb.com/bfs/article/e4dfe7f8f3c1808fb2d87786fa7ed40524550465.png)

假设输入图像大小为 $H*W$

1. 先将二维图像展平（因为 transformer 接受 1 维矩阵作为输入）：将图像分为多个固定大小的 patch（大小为 $P*P$，则 patch 数量为 $H*W/P^2$）。每个 patch 即为展平后向量的一个项。
2. 然后添加一个可训练的线性层，将每个图像块分别映射为 D 维向量（因为 Transformer 接受的向量大小始终为 D）
    * 类似于 NLP（Natural Language Processing）中的词嵌入，将离散的词汇转化为连续的向量。
    * 一个图像块就相当于 NLP 中的一个 token
3. 在第 2 步的线性层的每个输出向量中给加上一个可学习的位置信息。（因为 transformer 对数据之间的位置关系不敏感，相对而言 CNN 则对位置敏感）
4. 在第 2 步的线性层的输出中添加一个独立的 “token” 向量（图中 0*），作为一个 class token，它通过 Transformer 层进行处理，作为整个图像的类别进行输出。
5. 将这个线性层的输出送入 transformer encoder。
    * Norm（Normalization Layer） 层：即归一化层
    * Mult-Head Attention：多头注意力层
    * MLP（Multilayer Perceptron）：即多层感知机，也就是全连接层
6. 将 transformer 输出向量中的 class token 项送入一个分类头（仍是全连接多层感知机）中，对其作分类。（这个分类头在 fine-tune（微调）时是一个简单的线性层。）

同时，这个模型也可以和普通 CNN 模型结合使用：

* 即以 CNN 输出的特征图代替原始图像作为 transformer 的输入

其中：（创新点）

在 transformer encoder 的输入向量中加入一个用于表示类别的项，以及将位置信息嵌入每个图像快 token 中，是我认为的这篇文章的技术创新点。这很好的解决了 transformer 相对于 CNN 无法处理图像中隐含的位置信息的缺点，同时使这个模型适用于分类任务。

而且，此模型不仅适用于小分辨率图像的处理。当输入更高分辨率的图像时，若保持图像块大小不变，那么序列长度将增大，但 VIT 可以处理任意长度的序列，所以它可以处理任意大小的图像。这解决了Cordonnier等人使用 2*2 的小 patch，导致模型只能应用于小分辨率图像处理的问题。

# Task3.2

### CNN 与 VIT 的区别

* Vision Transformer 的自注意力机制允许模型将整个图像切成多个块，像处理序列（如自然语言）一样关注图像的全局特征，它可以考虑任何两个不相邻 patch 之间的特征。而 CNN 由于卷积核的局部性，只能关注局部区域的特征。
* CNN 通过卷积核的共享参数，可以减少参数量，提高计算效率。而 VIT 的参数量较大，计算效率较低。
* 与 CNN 不同，ViT 并没有内置很多图像相关的先验知识，更多依赖数据集来学习图像中的空间关系。

### VIT 与 CNN 的优缺点

#### VIT

优点：

* VIT 通过自注意力机制，可以捕获全局信息。
* 在数据集足够大的情况下，VIT 的准确率通常高于 CNN。

缺点：

* 参数量较大，计算效率较低。
* VIT 更多依赖数据集来学习图像中的空间关系。

#### CNN

优点：

* CNN 的参数量相对于 VIT 要小，计算消耗资源更少
* CNN 能够并行地计算，因此速度很快。

缺点：

* 更侧重于捕获局部信息，难以建模长距离的 token 依赖，难以捕获图像的全局特征。
* 准确率不如大参数量下的 VIT。

# Task3.3

### ResNet-18

#### 在 patch_size = 64; learning_rate = 0.0001 的情况下

```
Epoch [1/10], Loss: 1.7264, Accuracy: 37.73%
Epoch [2/10], Loss: 1.2100, Accuracy: 56.67%
Epoch [3/10], Loss: 1.0058, Accuracy: 64.34%
Epoch [4/10], Loss: 0.8559, Accuracy: 69.82%
Epoch [5/10], Loss: 0.7245, Accuracy: 74.68%
Epoch [6/10], Loss: 0.6057, Accuracy: 78.96%
Epoch [7/10], Loss: 0.4916, Accuracy: 83.31%
Epoch [8/10], Loss: 0.3875, Accuracy: 86.81%
Epoch [9/10], Loss: 0.2936, Accuracy: 90.04%
Epoch [10/10], Loss: 0.2269, Accuracy: 92.49%
Accuracy of the model on the 10000 test images: 66.77%
```

* 训练时间：6min
* 最终准确率：66.77%

#### 在较小 patch_size 下：patch_size = 32

```
Epoch [1/10], Loss: 1.6220, Accuracy: 41.13%
Epoch [2/10], Loss: 1.1645, Accuracy: 58.16%
Epoch [3/10], Loss: 0.9800, Accuracy: 65.01%
Epoch [4/10], Loss: 0.8381, Accuracy: 70.44%
Epoch [5/10], Loss: 0.7152, Accuracy: 74.76%
Epoch [6/10], Loss: 0.5995, Accuracy: 78.98%
Epoch [7/10], Loss: 0.4974, Accuracy: 82.69%
Epoch [8/10], Loss: 0.4036, Accuracy: 85.87%
Epoch [9/10], Loss: 0.3242, Accuracy: 88.74%
Epoch [10/10], Loss: 0.2588, Accuracy: 91.04%
Accuracy of the model on the 10000 test images: 67.78%
```

* 训练时间：10min
* 最终准确率：67.78%

可以看到，patch_size 几乎不影响模型的最终准确率。但是当 patch_size 减小时，训练时间明显增加。并且准确率曲线变化相对更加平缓，收敛速度变慢。

### ViT

vit.py 输出：

```
Epoch [1/10], Loss: 1.8650, Accuracy: 30.13%
Epoch [2/10], Loss: 1.6320, Accuracy: 39.58%
Epoch [3/10], Loss: 1.4742, Accuracy: 46.09%
Epoch [4/10], Loss: 1.3559, Accuracy: 50.97%
Epoch [5/10], Loss: 1.2781, Accuracy: 53.97%
Epoch [6/10], Loss: 1.2094, Accuracy: 56.51%
Epoch [7/10], Loss: 1.1513, Accuracy: 58.74%
Epoch [8/10], Loss: 1.1059, Accuracy: 60.25%
Epoch [9/10], Loss: 1.0586, Accuracy: 61.93%
Epoch [10/10], Loss: 1.0210, Accuracy: 63.27%
Accuracy of the model on the 10000 test images: 64.80%
```

* 训练时间：63min
* 最终准确率：64.80%

综上，VIT 的准确率（性能）和 CNN 的准确率相差并不大。并没有像想象中那样准确率远高于 CNN，我认为原因可能是

1. 训练数据量不够，因为 ViT 参数量大，相比于 CNN 更需要依靠大量数据来训练，对数据量的需求更大。
2. 训练轮数更少。

<!-- 我尝试将 patch_size 设为 16，但训练所耗的时间太长了。（悲 -->

并且可以看到，ViT 训练时准确率收敛速度很慢，且曲线平缓。并且 ViT 训练时间相比传统 CNN 网络明显要长。而且 ViT 不会出现 ResNet 测试集相对训练集准确率骤降的问题，ViT 的测试最终准确率甚至高于训练的最后一个循环。

我认为出现这种情况的原因是 ViT（Transformer）的参数量远高于传统 CNN 网络，这就使得 ViT 的泛化性要强于传统 CNN。所以 ViT 训练所耗时间明显更长，且不会出现测试时准确率骤降的问题。
