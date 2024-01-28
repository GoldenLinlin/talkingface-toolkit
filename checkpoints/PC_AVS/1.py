import torch
import torch.nn as nn

# 定义一个简单的模型
class pc_avstrainer(nn.Module):
    def __init__(self):
        super(pc_avstrainer, self).__init__()
        self.linear = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.linear(x)

# 创建模型实例
model = pc_avstrainer()

# 生成一个随机输入
input_tensor = torch.randn((1, 10))

# 使用模型进行前向传播
output_tensor = model(input_tensor)

# 获取模型的state_dict
state_dict = model.state_dict()

# 打印state_dict的键值
print("Model's state_dict keys:")
for key in state_dict:
    print(key)

# 保存模型
# 保存整个模型
torch.save(model, "./checkpoints/PC_AVS/simple_model.pth")
print("Entire model saved to simple_model.pth")

