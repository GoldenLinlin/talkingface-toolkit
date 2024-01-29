import torch.nn as nn

class PC_AVS(nn.Module):
    def __init__(self,config):
        super(PC_AVS, self).__init__()
        self.linear = nn.Linear(10, 5)
        self.config=config
    
    def forward(self, x):
        return self.linear(x)

    def generate_batch():
        print("eeeeeeeeeeeeeeeeeeee")

    def parameters(self):
    # 获取模型中所有可学习的参数
        for param in self.children():
            if hasattr(param, 'parameters'):
                for p in param.parameters():
                    print("%%%%%%%%%%%%%%%%")
                    print(p)
                    yield p
    def generate_batch(self):
        return self.config
        