import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import timm

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

batch_size = 16  # not 64
learning_rate = 0.0001
num_epochs = 10

transform = transforms.Compose(
    [
        transforms.Resize(224),  # 调整图像尺寸到 224x224
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
        ),  # ViT 的标准归一化参数
    ]
)

trainset = torchvision.datasets.CIFAR10(
    root="./task3/data", train=True, download=True, transform=transform
)
# print(trainset)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
# print(trainloader)

testset = torchvision.datasets.CIFAR10(
    root="./task3/data", train=False, download=True, transform=transform
)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

# load vit model
# vit_models = timm.list_models('vit*')
# print(vit_models)
# model = timm.create_model("vit_base_patch16_224", pretrained=False)
model = timm.create_model("vit_tiny_patch16_224", pretrained=False)
model.head = nn.Linear(model.head.in_features, 10)  #! CIFAR-10 的类别数是 10
model.to(device)
model.train()

# optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()


# train model
def train() -> None:
    for epoch in range(num_epochs):
        running_loss = 0
        correct = 0
        total = 0
        for i, data in enumerate(
            trainloader, 0
        ):  # enumerate is useful for obtaining an indexed list: (0, seq[0]), (1, seq[1]), (2, seq[2]), ...
            input_tensor, label = data  # data -> one batch
            # * data 是一个元组，包含两个元素：
            # * input_tensor：输入张量，通常是图像数据或其他形式的输入数据。
            # * label：标签，通常是与输入数据对应的目标值或类别
            input_tensor, label = input_tensor.to(device), label.to(device)

            optimizer.zero_grad()

            # forward
            output_tensor = model(input_tensor)
            # print(output_tensor)
            loss = criterion(output_tensor, label)

            # backward
            loss.backward()
            # update
            optimizer.step()

            # calculate loss
            running_loss = running_loss + loss.item()

            _, predicted = torch.max(output_tensor.data, 1)  # 输出预测值
            # * torch.max 函数返回指定维度上的最大值及其索引。这里的 1 表示在第一个维度（即每一行）上操作。
            # * torch.max(output_tensor.data, 1) 返回两个张量：最大值和最大值的索引。由于我们只关心索引（即预测的类别），所以使用 _ 忽略最大值。
            total += label.size(0)  # 标签总数
            correct += (predicted == label).sum().item()  # 正确预测数
            # * predicted == label：predicted 是模型预测的类别索引张量。label 是实际的类别标签张量。predicted == label 会生成一个布尔张量，其中每个元素表示预测是否与实际标签匹配（True 表示匹配，False 表示不匹配）。
            # * sum()：对布尔张量进行求和，True 计为 1，False 计为 0。结果是一个标量，表示当前批次中预测正确的样本数。
            # * item()：将标量张量转换为 Python 数值类型（如 int），以便进行累加操作。

        accuracy = 100 * correct / total
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(trainloader):.4f}, Accuracy: {accuracy:.2f}%"
        )

    return


def test() -> None:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in trainloader:
            input_tensor, label = data
            input_tensor, label = input_tensor.to(device), label.to(device)
            output_tensor = model(input_tensor)
            _, predicted = torch.max(output_tensor.data, 1)
            total = total + label.size(0)
            correct += (predicted == label).sum().item()
    accuracy = 100 * correct / total
    print(f"Accuracy of the model on the 10000 test images: {accuracy:.2f}%")

    return


def main() -> None:
    train()
    test()

    return


if __name__ == "__main__":
    main()
