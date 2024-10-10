import pandas # 加载 csv 数据集

# 由于 MacOS python 只能使用绝对路径
# train_file_path = '/Users/baijiale/Documents/Code/recruitment_for_ml_jotang/task1/data/train.csv'
# test_file_path = '/Users/baijiale/Documents/Code/recruitment_for_ml_jotang/task1/data/test.csv'
# sub_file_path = '/Users/baijiale/Documents/Code/recruitment_for_ml_jotang/task1/data/submission_example.csv'

train_file_path = '../data/train.csv'
test_file_path = '../data/test.csv'
sub_file_path = '../data/submission_example.csv'

# 加载 csv 文件
train_data = pandas.read_csv(train_file_path)
test_data = pandas.read_csv(test_file_path)
sub_data = pandas.read_csv(sub_file_path) # X_data 是 列表

from sklearn.preprocessing import StandardScaler # 数据预处理

# 分离 特征 和 输出变量
x_train = train_data.drop(columns=['ID', 'medv'])
y_train = train_data['medv']
x_test = test_data.drop(columns=['ID'])
y_test = sub_data['medv']

# 对 特征 标准化
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 将数据转换为 tensor
import torch

x_train_tensor = torch.tensor(x_train_scaled, dtype=torch.float32) # dtype -- data type
x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# 构建模型
class BostonHousingModel(torch.nn.Module): # 继承自 torch.nn.module 类
    def __init__(self):
        super(BostonHousingModel, self).__init__() # 父类构造函数
        self.fc1 = torch.nn.Linear(13, 64) # 使用父类方法
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 1) # 只有一个输出

    def forward(self, x):
        x = torch.relu(self.fc1(x)) # 前馈激活函数（进行的运算），使用 ReLU 激活函数
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = BostonHousingModel() # 实例化

# 设置损失函数
loss_fn = torch.nn.MSELoss()

# 设置优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # lr -- learning rate

# 训练模型
ecochs = 200
for i in range(ecochs):
    model.train() # 设置模型为训练模式
    optimizer.zero_grad() # 梯度清零
    # 反向传播算法
    # 1. 前馈：算出损失
    y_hat_tensor = model(x_train_tensor) #  step1: 输出本次训练的预测值
    loss_tensor = loss_fn(y_hat_tensor, y_train_tensor) # step2: 计算本次梯度
    # 2. 反馈：计算各层梯度
    loss_tensor.backward()
    # 3. （根据梯度）更新参数
    optimizer.step()

    # 每隔20个epoch输出一次损失
    if i % 20 == 0:
        print(f"Epoch {i}, Loss: {loss_tensor.item()}")

# 评估模型
model.eval() # 将模型设为评估模式
with torch.no_grad(): # 无需计算梯度
    y_hat_test_tensor = model(x_test_tensor)
    loss_test_tensor = loss_fn(y_hat_test_tensor, y_test_tensor)

print(f"Test Loss: {loss_test_tensor.item()}") # loss_test 是 tensor，所以要取其 item
