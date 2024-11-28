import torch
import torch.nn as nn
import torch.optim as optim

# 检查是否有可用的 GPU
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available!")

# 打印可用 GPU 数量和信息
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")
print("Available GPUs:", [torch.cuda.get_device_name(i) for i in range(num_gpus)])

# 显式设置主 GPU 并初始化 CUDA 上下文
torch.cuda.set_device(0)  # 将主设备设置为 cuda:0
torch.cuda.init()  # 显式初始化 CUDA

# 设置设备
device = torch.device("cuda:0")
print(f"Using device: {device}")

# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

# 初始化模型并使用 DataParallel
model = SimpleModel()
if num_gpus > 1:
    print("Using nn.DataParallel for multi-GPU support.")
    model = nn.DataParallel(model)
model = model.to(device)

# 检查模型的设备分布
for name, param in model.named_parameters():
    print(f"Parameter: {name}, Device: {param.device}")

# 创建输入张量并移动到主 GPU
input_tensor = torch.randn(32, 10).to(device)

# 定义目标张量
target = torch.randn(32, 5).to(device)

# 定义优化器和损失函数
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# 执行前向传播
output = model(input_tensor)
loss = criterion(output, target)

print(f"Loss: {loss.item()}")

# 执行反向传播和优化
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Test completed without warnings or errors.")
