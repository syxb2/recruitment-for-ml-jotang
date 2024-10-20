## Task4.1

### Related Knowledge

#### Logits 是神经网络和机器学习中常见的术语，特别是在分类任务中。

1. 定义：Logits 是指未经过归一化处理的原始输出（分数）。在一个神经网络的输出层，尤其是用于分类问题的神经网络中，网络会输出一组值。Logits 就是这些值，它们可能是正数、负数或零，并且没有具体的概率意义。

2. 使用场景：
    * 在多分类问题中，模型的输出可能是多个类别的得分，这些得分并没有被转换成概率值。通过 softmax 函数（或 sigmoid 函数用于二分类问题），这些 logits 被转化为概率分布，概率的和为 1，表示模型对每个类别的信心。
    * 举例来说，假设一个三分类问题，网络输出 [2.5, 1.0, -0.5] 这样的 logits。经过 softmax 后，转换成的概率可能是 [0.7, 0.2, 0.1]，表示模型认为第一个类别的概率最大。

3. 与概率的关系：

* Logits 并不是直接的概率值，而是模型在没有进行概率归一化时输出的分数。要将 logits 转化为概率，通常使用 softmax 函数来进行归一化。Softmax 函数公式如下：

$$
P(y_i) = \frac{e^{logit_i}}{\sum_{j=1}^n e^{logit_j}}
$$

* 其中 $P(y_i)$ 是类别 $i$ 的预测概率，$logit_i$ 是该类别的 logits 值。

4. 在损失函数中的作用：

* 在训练神经网络时，很多损失函数（比如交叉熵损失函数）期望输入的是 logits 而不是经过 softmax 处理后的概率。这是因为在损失函数中同时计算 logits 和 softmax 具有更好的数值稳定性。

#### Softmax函数是什么？

Softmax是一种将一组logits转化为概率分布的函数。它会将每个logit通过指数函数处理，然后归一化，使得所有输出值的和为1，从而可以作为概率解释。公式如下：

$$
\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
$$

这里的 z_i 是logits的其中一个元素。

#### Attention机制

参见：[https://transformers.run/c1/attention/](https://transformers.run/c1/attention/)

Attention机制用于神经网络中，帮助模型专注于输入序列中对当前任务最相关的部分。特别是在自然语言处理任务中，Attention机制可以根据输入序列中的每个单词为其他单词分配权重，从而提高模型的理解能力。

例如，经典的自注意力机制（self-attention）可以让模型在处理当前单词时，考虑整个输入序列中其他单词的上下文信息。

> 详见 [【点击跳转】Task3/README.md](../task3/README.md)

#### Transformer

Transformer是一个由Google提出的神经网络架构，广泛应用于NLP任务。它依赖于自注意力机制，能够并行处理整个输入序列，而不像传统的RNN依赖序列信息的逐步处理。Transformer包括编码器和解码器结构，用来处理不同的任务，如翻译、文本生成等。

> 详见 [【点击跳转】Task3/README.md](../task3/README.md)

#### 自回归模型

大型语言模型（LLM）通常采用自回归模型架构，意味着它们生成输出的方式是逐步预测下一个单词（或token），基于已经生成的序列。生成的每个新词会基于之前生成的词作为输入，直到完成整个句子或文本。常见的自回归模型包括GPT系列模型。

### hard watermark

#### 生成规则：

硬水印 根据伪随机数（以 s(t-1) 作为种子）将 AI 词典中的 token 分为“红名单”和“绿名单”，绿名单中没有红名单的词，且 AI 生成文本时只能生成绿名单中的词。

原文算法：

![](https://i0.hdslb.com/bfs/article/54f24448f834f91486e2a6d6d1c9588d24550465.png)

#### 水印检测：

这样就可以通过检测文本中 z 值的标准量（$(x - 均值)/方差$）来判断文本是否由 AI 生成。

#### 缺点：

* 硬水印强制禁止模型使用某些单词，这导致模型的生成能力受到限制，可能会影响生成文本的质量。

### soft watermark

#### 生成规则：

软水印 根据文本的熵动态的调整水印强度。其在模型最后一层输出的 logits 向量（softmax 层之前）的绿名单词中加入一个常数偏置 δ。

当文本的熵较低，δ 对生成词的概率影响小，故软水印规则对文本生成的影响较小，当文本的熵较高时，δ 对采样分布的影响很大，使生成的词强烈地偏向绿名单。

这样就在生成低熵文本时尽可能不影响生成的质量，而在生成高熵文本时，使生成的文本更加符合水印规则。

原文算法：

![](https://i0.hdslb.com/bfs/article/3a1ae7a33cc877406ff655462e768fc824550465.png)

且只要低熵序列被包装在具有足够总熵的段落中，该段落仍将轻松触发水印检测器，从而解决了软水印中描述的低熵文本难以处理的问题。

#### 水印检测：

和硬水印一样，仍是通过计算文本的 z 标准量来判断文本是否是由 AI 生成。如果z大于某个阈值，我们就拒绝零假设并检测到水印

#### 绿色列表令牌的预期数量：

### Watermark Attack

攻击方式大体分为三个大类

1. Text insertion

* 这种攻击方法在模型生成文本之后重新加入一些可能在红名单中的 token 到文本中，并且修改下游红名单计算结果

2. Text deletion

* 删除一些在绿名单中的 tokens，从而增加红名单 token 的比例。但这种方法可能影响文本质量。

3. Text substitution

* 将文本中绿名单 token 和一个处于红名单中的 token 交换。但也可能影响文本质量。

#### Paraphrasing Attacks

即 由人类人工改写文本。

#### Discreet Alterations(谨慎修改) Attacks

通过进行小改动，如添加空格或少量拼写错误，影响哈希值的计算。**从而影响红名单的生成，使计算的 z 标准值出现偏差**

#### Tokenization(分词) Attacks

通过修改文本，改变后续单词的分词情况，**从而增加模型生成句子中红名单 token 的数量，进而使得攻击奏效**

#### Homoglyph and Zero-Width Attacks

利用 Unicode 字符的不唯一的特点，通过将单词替换为“同形字”（Unicode不同但字母相同或相似），进而使句子中绿名单中的 token 减少，红名单 token 增加（计算机判断 token 时是通过字符的 Unicode 判断，这就使这种方式能都再不改变单词的情况下使一个“绿词”变为一个“红词”）。

#### Generative Attacks

即利用大模型的上下文学习能力，提示模型以可预测和易于逆转的方式更改其输出，这样会影响模型输出文字的能力，但与此同时也使得后续令牌的红名单改变，那么我们在把输出逆转后，水印也就检测不到了。

#### Details of the T5 Span Attack

该方法通过逐步替换文本中的部分单词，并借助T5的语言生成能力来生成合理的新文本，进而替换句子中的绿名单 token，以此来进行水印攻击。但这种方法也可能带来文本质量的损失。

## Task4.2

* 复现方法：使用官方 demo 复现。
* 模型使用 `facebook/opt-125m`（考虑到运行速度，我使用了一个小参数量的模型）
* 运行环境：macos sequoia，python 3.10

### 项目复现

git 项目地址：

```
https://github.com/jwkirchenbauer/lm-watermarking.git
```

复现过程：

* 使用 git 克隆官方仓库，使用 venv 环境管理器安装所需依赖，在 app.py 文件中指定模型为 `facebook/opt-125m`，使用 python 解释器运行 `app.py`。

复现效果截图：

![](https://i0.hdslb.com/bfs/article/34944f5b8edb64545d08e9fb9b5995c424550465.png)

shell 输出：

```
Namespace(run_gradio=True, demo_public=False, model_name_or_path='facebook/opt-125m', load_fp16=False, prompt_max_length=None, max_new_tokens=200, generation_seed=123, use_sampling=True, n_beams=1, sampling_temp=0.7, use_gpu=True, seeding_scheme='simple_1', gamma=0.25, delta=2.0, normalizers=[], ignore_repeated_bigrams=False, detection_z_threshold=4.0, select_green_tokens=True, skip_model_load=False, seed_separately=True)
################################################################################
Prompt:
The diamondback terrapin or simply terrapin (Malaclemys terrapin) is a species of turtle native to the brackish coastal tidal marshes of the Northeastern and southern United States, and in Bermuda.[6] It belongs to the monotypic genus Malaclemys. It has one of the largest ranges of all turtles in North America, stretching as far south as the Florida Keys and as far north as Cape Cod.[7] The name 'terrapin' is derived from the Algonquian word torope.[8] It applies to Malaclemys terrapin in both British English and American English. The name originally was used by early European settlers in North America to describe these brackish-water turtles that inhabited neither freshwater habitats nor the sea. It retains this primary meaning in American English.[8] In British English, however, other semi-aquatic turtle species, such as the red-eared slider, might also be called terrapins. The common name refers to the diamond pattern on top of its shell (carapace), but the overall pattern and coloration vary greatly. The shell is usually wider at the back than in the front, and from above it appears wedge-shaped. The shell coloring can vary from brown to grey, and its body color can be grey, brown, yellow, or white. All have a unique pattern of wiggly, black markings or spots on their body and head. The diamondback terrapin has large webbed feet.[9] The species is
Generating with Namespace(run_gradio=True, demo_public=False, model_name_or_path='facebook/opt-125m', load_fp16=False, prompt_max_length=None, max_new_tokens=200, generation_seed=123, use_sampling=True, n_beams=1, sampling_temp=0.7, use_gpu=True, seeding_scheme='simple_1', gamma=0.25, delta=2.0, normalizers=[], ignore_repeated_bigrams=False, detection_z_threshold=4.0, select_green_tokens=True, skip_model_load=False, seed_separately=True, is_seq2seq_model=False, is_decoder_only_model=True, default_prompt="The diamondback terrapin or simply terrapin (Malaclemys terrapin) is a species of turtle native to the brackish coastal tidal marshes of the Northeastern and southern United States, and in Bermuda.[6] It belongs to the monotypic genus Malaclemys. It has one of the largest ranges of all turtles in North America, stretching as far south as the Florida Keys and as far north as Cape Cod.[7] The name 'terrapin' is derived from the Algonquian word torope.[8] It applies to Malaclemys terrapin in both British English and American English. The name originally was used by early European settlers in North America to describe these brackish-water turtles that inhabited neither freshwater habitats nor the sea. It retains this primary meaning in American English.[8] In British English, however, other semi-aquatic turtle species, such as the red-eared slider, might also be called terrapins. The common name refers to the diamond pattern on top of its shell (carapace), but the overall pattern and coloration vary greatly. The shell is usually wider at the back than in the front, and from above it appears wedge-shaped. The shell coloring can vary from brown to grey, and its body color can be grey, brown, yellow, or white. All have a unique pattern of wiggly, black markings or spots on their body and head. The diamondback terrapin has large webbed feet.[9] The species is")
################################################################################
Output without watermark:
 endemic to the Caribbean. The specimen is considered to be part of a group of North American turtle species, and has been observed on the coast of North America since the 13th century.[10]
--------------------------------------------------------------------------------
Detection result @ 4.0:
([['Tokens Counted (T)', '38'],
  ['# Tokens in Greenlist', '11'],
  ['Fraction of T in Greenlist', '28.9%'],
  ['z-score', '0.562'],
  ['p value', '0.287'],
  ['z-score Threshold', '4.0'],
  ['Prediction', 'Human/Unwatermarked']],
 Namespace(run_gradio=True, demo_public=False, model_name_or_path='facebook/opt-125m', load_fp16=False, prompt_max_length=1848, max_new_tokens=200, generation_seed=123, use_sampling=True, n_beams=1, sampling_temp=0.7, use_gpu=True, seeding_scheme='simple_1', gamma=0.25, delta=2.0, normalizers=[], ignore_repeated_bigrams=False, detection_z_threshold=4.0, select_green_tokens=True, skip_model_load=False, seed_separately=True, is_seq2seq_model=False, is_decoder_only_model=True, default_prompt="The diamondback terrapin or simply terrapin (Malaclemys terrapin) is a species of turtle native to the brackish coastal tidal marshes of the Northeastern and southern United States, and in Bermuda.[6] It belongs to the monotypic genus Malaclemys. It has one of the largest ranges of all turtles in North America, stretching as far south as the Florida Keys and as far north as Cape Cod.[7] The name 'terrapin' is derived from the Algonquian word torope.[8] It applies to Malaclemys terrapin in both British English and American English. The name originally was used by early European settlers in North America to describe these brackish-water turtles that inhabited neither freshwater habitats nor the sea. It retains this primary meaning in American English.[8] In British English, however, other semi-aquatic turtle species, such as the red-eared slider, might also be called terrapins. The common name refers to the diamond pattern on top of its shell (carapace), but the overall pattern and coloration vary greatly. The shell is usually wider at the back than in the front, and from above it appears wedge-shaped. The shell coloring can vary from brown to grey, and its body color can be grey, brown, yellow, or white. All have a unique pattern of wiggly, black markings or spots on their body and head. The diamondback terrapin has large webbed feet.[9] The species is"))
--------------------------------------------------------------------------------
################################################################################
Output with watermark:
 endemic to New Zealand. The adult size varies, and this does not affect the size of the flesh. The turtle is about 15 to 20 cm (1.5 to 7  millimeters) long, and 6 feet tall, and has an average weight of about 170–160 kg (220 to 220 lbs).[9]

The name terrapin is derived from the Latin terrapinus, and this name was originally used in the England and Wales language, where it is despite being a dialect of the English-speaking British community. The English-speaking language of the English-speaking community, however, frequently used the term terrapin to refer to these turtle types. The English-speaking language of the English-speaking community also referred to these turtle types as terrapinis if they were not mixed with the white-cured turtle from New Zealand.[10]

Although the term terrapini was used in the English-speaking world to refer to these turtle types
--------------------------------------------------------------------------------
Detection result @ 4.0:
([['Tokens Counted (T)', '199'],
  ['# Tokens in Greenlist', '130'],
  ['Fraction of T in Greenlist', '65.3%'],
  ['z-score', '13.1'],
  ['p value', '1e-39'],
  ['z-score Threshold', '4.0'],
  ['Prediction', 'Watermarked'],
  ['Confidence', '100.000%']],
 Namespace(run_gradio=True, demo_public=False, model_name_or_path='facebook/opt-125m', load_fp16=False, prompt_max_length=1848, max_new_tokens=200, generation_seed=123, use_sampling=True, n_beams=1, sampling_temp=0.7, use_gpu=True, seeding_scheme='simple_1', gamma=0.25, delta=2.0, normalizers=[], ignore_repeated_bigrams=False, detection_z_threshold=4.0, select_green_tokens=True, skip_model_load=False, seed_separately=True, is_seq2seq_model=False, is_decoder_only_model=True, default_prompt="The diamondback terrapin or simply terrapin (Malaclemys terrapin) is a species of turtle native to the brackish coastal tidal marshes of the Northeastern and southern United States, and in Bermuda.[6] It belongs to the monotypic genus Malaclemys. It has one of the largest ranges of all turtles in North America, stretching as far south as the Florida Keys and as far north as Cape Cod.[7] The name 'terrapin' is derived from the Algonquian word torope.[8] It applies to Malaclemys terrapin in both British English and American English. The name originally was used by early European settlers in North America to describe these brackish-water turtles that inhabited neither freshwater habitats nor the sea. It retains this primary meaning in American English.[8] In British English, however, other semi-aquatic turtle species, such as the red-eared slider, might also be called terrapins. The common name refers to the diamond pattern on top of its shell (carapace), but the overall pattern and coloration vary greatly. The shell is usually wider at the back than in the front, and from above it appears wedge-shaped. The shell coloring can vary from brown to grey, and its body color can be grey, brown, yellow, or white. All have a unique pattern of wiggly, black markings or spots on their body and head. The diamondback terrapin has large webbed feet.[9] The species is"))
 ```

### 复现时遇到的问题

运行 `app.py` 时报错：

```shell
  File "./demo_watermark.py", line 610, in run_gradio
    demo.queue(concurrency_count=3)
TypeError: Blocks.queue() got an unexpected keyword argument 'concurrency_count'
```

这个错误是 `gradio.Blocks().queue()` 抛出的，所以我推测可能是 gradio 包版本不兼容问题，我尝试将 gradio 降级到 4.1.0 版本，但它提示建议使用最新版本，并出现了其他错误。

所以我尝试删除 `concurrency_count` 参数。之后便成功运行，但是出现警告：*（虽然有警告，但是确实跑起来了（笑 ）*

```shell
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
```

上面的输出给出了消除警告的方法，但这设计更改源代码的整体流程，这有点超出我的能力范围（毕竟也能跑）

我还查阅了 gardio 官方文档，[我是链接](https://www.gradio.app/docs/gradio/blocks)，发现在 gradio@4.44.1 版本中的 Blocks 类的 queue 方法中有 `concurrency_count` 参数，而在 gradio@5.1.0 版本中没有这个参数。所以我的猜想是正确的，而删除这个参数后并不影响程序的正常运行。*包不兼容还是很常见的问题，感觉要避免这个东西挺困难的。而且作者并没有给出每个包的具体版本。*

**总之，项目复现的过程大体顺利。**

### 重要函数注释

`watermark_processor.py`:

* 类 WatermarkLogitsProcessor 实现了`软水印`的添加
* 类 WatermarkDetector 实现了水印的检测

#### 类 `WatermarkLogitsProcessor` 的 `_calc_greenlist_mask` 方法：

```python
def _calc_greenlist_mask(self, scores: torch.FloatTensor, greenlist_token_ids) -> torch.BoolTensor:
    green_tokens_mask = torch.zeros_like(scores)
    for b_idx in range(len(greenlist_token_ids)):
        green_tokens_mask[b_idx][greenlist_token_ids[b_idx]] = 1
    final_mask = green_tokens_mask.bool()
    return final_mask
```

* 这个方法的作用是将列表参数 greenlist_token_ids 中的 tokens 标记为 "greentokens"，也就是将它们添加到绿名单

#### 类 `WatermarkLogitsProcessor` 的 `__call__` 方法：

```python
def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    if self.rng is None:
        self.rng = torch.Generator(device=input_ids.device)

    batched_greenlist_ids = [None for _ in range(input_ids.shape[0])]

    for b_idx in range(input_ids.shape[0]):
        greenlist_ids = self._get_greenlist_ids(input_ids[b_idx])
        batched_greenlist_ids[b_idx] = greenlist_ids

    green_tokens_mask = self._calc_greenlist_mask(scores=scores, greenlist_token_ids=batched_greenlist_ids)

    scores = self._bias_greenlist_logits(scores=scores, greenlist_mask=green_tokens_mask, greenlist_bias=self.delta)
    return scores
```

* _calc_greenlist_mask 方法用于生成一个 mask，用于标记哪些 token 属于绿名单
* _bias_greenlist_logits 方法用于对绿名单中 token 的 logits 添加偏置 bias
* `__call__` 方法定义了处理 token 的 logits 的流程，在其中调用上述两个方法，从而将相应 tokens 添加到绿名单，并为他们添加偏置

#### 类 `WatermarkDetector` 的 `_compute_z_score` 方法：

```python
def _compute_z_score(self, observed_count, T):
    """
    observed_count 是检测到的绿色 token 的个数。T 是总 token 的数量。
    """
    expected_count = self.gamma # expected_count 是预期绿色token的**比例**
    numer = observed_count - expected_count * T
    denom = sqrt(T * expected_count * (1 - expected_count))
    z = numer / denom
    return z
```

* 这个方法用于计算 $z$ 标准值，Z分数衡量的是观察值与期望值之间的差距，并以标准差为单位。论文中：

![](https://i0.hdslb.com/bfs/article/aa5fe26b04a02b0ebf88fa85abaac7d824550465.png)

就是 (x - 均值)/标准差

#### 类 `WatermarkDetector` 的 `_compute_p_value` 方法：

```python
def _compute_p_value(self, z):
    p_value = scipy.stats.norm.sf(z)
    return p_value
```

* 这个方法用于计算 $p$ 值，$p$ 值代表假设检验中观察到当前结果或更极端结果的概率。

#### 类 `WatermarkDetector` 的 `_score_sequence` 方法：

```python
def _score_sequence(self, input_ids: Tensor, return_num_tokens_scored: bool = True, return_num_green_tokens: bool = True, return_green_fraction: bool = True, return_green_token_mask: bool = False, return_z_score: bool = True, return_p_value: bool = True):
```

* 这个方法用于计算 输入 token 的相关指标，如计算绿色 token 数量、比例，以及计算 z 值、p 值。

#### 类 `WatermarkDetector` 的 `detect` 方法：

```python
def detect(self, text: str = None, tokenized_text: list[int] = None, return_prediction: bool = True, return_scores: bool = True, z_threshold: float = None, **kwargs):
```

* 这个方法定义了检测水印算法的流程，是此类的核心方法。它是一个公开接口，供外部调用。
* 方法返回预测结果和预测可信度。

### 一点发现

我发现当模型使用 GPU 时，运行时间会比使用 CPU 要长，但是生成的文本明显比使用 CPU 时更加复杂。

**使用 GPU：**

```py
device = "mps" if torch.backends.mps.is_available() else "cpu"
```

*因为我的环境是 MacOS，故 GPU 不是 cuda 而是 mps*

生成结果以及所耗时间：

![](https://i0.hdslb.com/bfs/article/a4bc58226ea9954cf87fe80594d267f824550465.png)

* 消耗时间：24s 左右

**使用 CPU：**

```py
device = 'cpu'
```
生成结果以及所耗时间：

![](https://i0.hdslb.com/bfs/article/723d20e714987d0ba81c980d2f73aa1e24550465.png)

* 消耗时间：7s 左右

## Task4.3

### 该项目的缺陷：

虽然项目通过曲线救国的方式给出了低熵文本的水印添加和检测方法，即将低熵文本包装在高熵文本中。但该项目并没有指出直接为低熵文本添加水印以及配套的检测方法[1]。

在实际应用中，低熵文本的生成是很常见的，比如一些科技文献、法律文件等。如果不能直接为低熵文本添加水印，那么 AI 生成的文本在某些领域仍存在某些风险。

而且，水印被直接暴露的嵌入文本中，非常容易遭到攻击，尤其是人工攻击是很难做出防御的[2]。

还有就是添加水印可能会导致文本含义发生变化，我感觉这还是与其直接将水印嵌入文本的思路有关，对于这种方法，某些问题很难避免。
