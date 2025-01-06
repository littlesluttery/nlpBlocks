# MLP
import torch
import torch.nn as nn

# 假设ACT2FN是一个包含激活函数的字典，这里用ReLU作为示例
ACT2FN = {
    'relu': nn.ReLU(),
    'silu':nn.SiLU()
}

class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 设定网络尺寸
        self.hidden_size = config['hidden_size']
        self.intermediate_size = config['intermediate_size']
        # 三个全连接层
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        # ACT2FN是自定义的一系列激活函数字典，config的定义来获取这里用到的激活函数类型
        self.act_fn = ACT2FN[self.config['hidden_act']]


    def forward(self, x):
        # 两个尺寸相同的网络，其中一个经过激活后与另一个结果相乘，得到的结果过第三层
        gate_output = self.act_fn(self.gate_proj(x))
        up_output = self.up_proj(x)
        down_proj = self.down_proj(gate_output * up_output)
        return down_proj

# 配置参数
config = {
    'hidden_size': 128,
    'intermediate_size': 256,
    'hidden_act': 'relu'
}

# 初始化模型
model = Qwen2MLP(config)

# 创建一个随机输入张量，假设输入特征维度为config['hidden_size']
input_tensor = torch.randn(1, config['hidden_size'])

# 调用模型并获取输出
output = model(input_tensor)

# 打印输出
print(output)
print(output.shape) # torch.Size([1, 128])
