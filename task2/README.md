## 思考题

### 1. 计算机采用什么数据结构存储、处理图像？

采用多维数组（矩阵）的形式存储、处理图像

### 2. 如何设计一个神经网络？一个神经网络通常包括哪些部分？

可以根据我们要进行的任务设计不同的网络，如图像处理就用卷积，回归任务就用线性全连接网络等

一个神经网络包括：

1. 多层神经元
2. 每层的激活函数
3. 损失函数
4. 优化器

等

### 3. 什么是欠拟合？什么是过拟合？

* 欠拟合就是指模型过于简单，无法充分拟合数据
* 过拟合就是指模型过于复杂、敏感，拟合了数据的噪声

## 进阶任务

*更改/新加的代码在 `task_extra.py` 中用 `#!` 注释标注*（使用 `Better Comments @vscode` 插件体验更佳）

### 1. 提高准确率

这里列举一些模型优化的方法：可以尝试新的损失函数，新的优化器，新的网络结构等。

我选择添加一些归一化层，另外多加了一个线性层，使模型的复杂度提升，以更好的拟合图像特征；另外加入了一个 dropout 层，随机丢弃一些神经元，防止模型过拟合。

下面是修改的代码：

```python
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 卷积层1
        self.bn1 = nn.BatchNorm2d(32)  # 批量归一化层1，可以加速收敛，提高精度
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 卷积层2
        self.bn2 = nn.BatchNorm2d(64)  # 批量归一化层2
        self.pool = nn.MaxPool2d(2, 2)  # 池化层
        self.dropout = nn.Dropout(
            0.25
        )  # Dropout层，随机丢弃一部分神经元，可以防止过拟合，
        self.fc1 = nn.Linear(64 * 8 * 8, 512)  # 全连接层1
        self.fc2 = nn.Linear(512, 10)  # 全连接层2

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.bn1(self.conv1(x))))
        x = self.pool(nn.functional.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 8 * 8)  # 展平
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
```

### 2. 结果的可视化

在代码的输出中，有每次迭代的损失值和准确率，所以我选择用 库绘制表格以可视化这两个维度的数据

```python
def plot_results(results: list) -> None:
    #! 将列表转换为 numpy 数组
    results = np.array(results)

    #! 将 损失列 和 准确率列 分别提取出来
    loss = results[:, 0]
    accuracy = results[:, 1]

    #! 绘制一个柱状图，横坐标为 epoch，左纵坐标为 loss，右纵坐标为 accuracy
    epochs = np.arange(1, len(loss) + 1) # 创建一个包含从 1 到 len(loss) 的整数序列的 NumPy 数组
    width = 0.4  # 柱状图的宽度，即柱子的宽度和两个柱子之间的间隔

    _, axe1 = plt.subplots() # figure 是整个图形窗口，axe1 是第一个子图
    axe1.bar(epochs - width/2, loss, width, label='Loss', color='b')
    axe1.set_xlabel('Epoch')
    axe1.set_ylabel('Loss', color='b') # b - blue

    axe2 = axe1.twinx() # 创建第二个子图，其 x 轴不可见（与第一个子图共享 x 轴。）
    axe2.bar(epochs + width/2, accuracy, width, label='Accuracy', color='r')
    axe2.set_ylabel('Accuracy (%)', color='r') # r - red

    plt.show() # 显示图形窗口
```

> 注：详细笔记和思考见文件：`./image_classification.ipynb`
