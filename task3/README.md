## Task3.1

> 论文代码 github 地址：[https://github.com/google-research/vision_transformer](https://github.com/google-research/vision_transformer)

### Related Knowledge

#### 1、嵌入向量（Embedding Vector）

嵌入向量是将离散的词或符号（如文本中的单词）映射到一个连续的、低维的向量空间中的表示。在深度学习中，模型不能直接处理文本或符号，所以需要将这些离散的数据转化为可以操作的向量。

假设有词汇表["cat", "dog", "apple", "orange"]，每个词会被映射到一个向量，如：

* cat -> [0.2, 0.5, -0.1]
* dog -> [0.3, 0.4, -0.2]
* apple -> [0.6, -0.1, 0.9]
* orange -> [0.5, -0.2, 0.8]

参考：[https://transformers.run/c1/attention/](https://transformers.run/c1/attention/)

#### 2、以一段 pytorch 实现 Scaled Dot-product Attention 的代码为例，简要说明 scaled dot-product attention 的计算过程

> [【点击跳转】scaled_dot_product_attention.ipynb](./scaled_dot_product_attention.ipynb)

![](<media/截屏2024-10-17 下午9.06.53.png>)

* $d_k$ 是 Q、K token 的嵌入向量的长度
* $d_v$ 是 V 中转化为的嵌入向量的长度

![](<media/截屏2024-10-17 下午9.10.54.png>)

#### 3、Multi-head Attention

![](<media/截屏2024-10-17 下午9.13.01.png>)

1. 首先通过线性映射将 Q、K、V 序列映射到特征空间
2. 每一组映射后 Q、K、V 表示称为一个头 (head)
3. 用这多个 head 分别做多次 scaled dot-product attention，最后将结果拼接，就是 multi-head attention 的输出

![](<media/截屏2024-10-17 下午9.15.50.png>)

* 最后结果的大小是 $m*hd^-_k$

### The Paper

这是本论文提出的模型结构：

![](<media/截屏2024-10-17 下午4.17.14.png>)

假设输入图像大小为 $H*W$

1. 先将二维图像展平（因为 transformer 接受 1 维矩阵作为输入）：将图像分为多个固定大小的 patch（大小为 $P*P$，则 patch 数量为 $H*W/P^2$）。每个 patch 即为展平后向量的一个项。
2. 然后添加一个可训练的线性层，将每个图像块分别映射为 D 维向量（因为 Transformer 接受的向量大小始终为 D）
    * 类似于 NLP 中的词嵌入，将离散的词汇转化为连续的向量。
3. 在第 2 步的线性层的每个输出向量中给加上一个可学习的位置信息。（因为 transformer 对数据之间的位置关系不敏感，相对而言 CNN 则对位置敏感）
4. 在第 2 步的线性层的输出中添加一个独立的 token 向量（图中 0*），作为一个 class token，它通过 Transformer 层进行处理，作为整个图像的类别进行输出。
5. 将这个线性层的输出送入 transformer encoder。
    * 此模型的 transformer 由多个多头自注意力层（Multi-Headed Self-Attention, MSA）和前馈神经网络层（MLP）组成。每层还包括Layernorm（层归一化）和残差连接。[【点击跳转】explanation.md](./explanation.md)
6. 将 transformer 输出向量中的 class token 项送入一个分类头中，用于分类任务。（这个分类头在微调时是一个简单的线性层。）

同时，这个模型也可以和普通 CNN 模型结合使用：

* 即以 CNN 输出的特征图代替原始图像作为 transformer 的输入

其中：（创新点）

在 transformer encoder 的输入向量中加入一个用于表示类别的项，以及将位置信息嵌入每个图像快 token 中，是我认为的这篇文章的技术创新点。这很好的解决了 transformer 相对于 CNN 无法处理图像中隐含的位置信息的缺点，同时使这个模型适用于分类任务。

而且，此模型不仅适用于小分辨率图像的处理。当输入更高分辨率的图像时，若保持图像块大小不变，那么序列长度将增大，但 VIT 可以处理任意长度的序列，所以它可以处理任意大小的图像。这解决了Cordonnier等人使用 2*2 的小 patch，导致模型只能应用于小分辨率图像处理的问题。

## Task3.2

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

## Task3.3

/* TODO */
