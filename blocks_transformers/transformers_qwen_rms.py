import torch
import torch.nn as nn

# RMS
from IPython.display import Image
Image(filename="./image/rms.png",width=400,height=400)

class Qwen2RMSNorm(nn.Module):
    def __init__(self,hidden_size,eps=1e-6):
        super(Qwen2RMSNorm, self).__init__()
        # RMS Norm层的参数
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # 防止出现除0错误,一般也不要太小1e-5到1e-7,防止数据精度溢出
        self.variance_epsilon = eps

    def forward(self,hidden_states):
        # 输入是隐变量,它的shape是[batch_size, seq_len, hidden_size]
        # 数据类型,为了防止计算出错,先将数据类型提升为torch.float32
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        # 计算方差，这里keepdim是均值时为保留维度，确保与下面hidden_states进行矩阵运算
        variance = hidden_states.pow(2).mean(-1,keepdim=True) # [batch_size,seq_len,1]
        print(variance)
        # 按照公式计算
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        print(hidden_states)
        return self.weight * hidden_states.to(input_dtype)

batch_size = 1
seq_len = 3
hidden_size = 3

rms_norm = Qwen2RMSNorm(hidden_size)
hidden_states = torch.rand((batch_size, seq_len, hidden_size))
print(hidden_states)
# print("hidden_states的shape为:", hidden_states.shape)
rms_output = rms_norm(hidden_states)
print(rms_output)
print("rms_output的shape为:", rms_output.shape)
