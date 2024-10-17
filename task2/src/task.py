import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# device = torch.device(
#     "cuda" if torch.cuda.is_available() else "cpu"
# )
# device = torch.device(
#     "mps" if torch.backends.mps.is_available else "cpu"
# )  # for MacOS, GPU is mps instead of cuda

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# TODO:解释参数含义，在?处填入合适的参数
batch_size = 64  # 每次小型迭代时传递给模型的样本数量
learning_rate = 0.001  # 学习率，即梯度下降算法中，负梯度的系数
num_epochs = 10  # 训练的轮数

# 转换为 pytorch 要求的 tensor 格式
transform = transforms.Compose([transforms.ToTensor()])

# 构造训练集和测试集(root可以换为你自己的路径)
trainset = torchvision.datasets.CIFAR10(
    root="../data", train=True, download=True, transform=transform
)
trainloader = DataLoader(
    trainset, batch_size=batch_size, shuffle=True
)  # mini-batch 在这里使用

testset = torchvision.datasets.CIFAR10(
    root="../data", train=False, download=True, transform=transform
)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)


# 构造模型
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # TODO:这里补全你的网络层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 卷积层1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 卷积层2
        self.pool = nn.MaxPool2d(2, 2)  # 池化层
        self.fc1 = nn.Linear(64 * 8 * 8, 10)  # 全连接层

    def forward(self, x):
        # TODO:这里补全你的前向传播
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)  # 展平
        x = self.fc1(x)

        return x


# TODO:补全
model = Network()  # 实例化模型
model.to(device)  # 将模型放入GPU

# 损失函数
criterion = nn.CrossEntropyLoss()  # torch 中 交叉熵损失函数
# 优化器
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# 训练：将循环封装进一个函数中
def train() -> None:
    model.train()  # 模型设置为训练模式
    for epoch in range(num_epochs):  # 训练轮数
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):  # 每一轮分为 batch 个迭代
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # 梯度清零

            # 前馈
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反馈
            loss.backward()
            # 更新
            optimizer.step()

            # 计算每个 batch 的损失
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)  # 输出预测值
            total += labels.size(0)  # 标签总数
            correct += (predicted == labels).sum().item()  # 正确预测数

        accuracy = 100 * correct / total
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(trainloader):.4f}, Accuracy: {accuracy:.2f}%"
        )

    return


def test() -> None:
    model.eval()  # 模型设置为评估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 测试时不需计算梯度
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy of the model on the 10000 test images: {accuracy:.2f}%")

    return


if __name__ == "__main__":
    train()
    test()
