# MQA、GQA和MHA
import torch
import torch.nn as nn
import torch.nn.functional as F

class MHA(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MHA, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.num_heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)
        queries = query.reshape(N, query_len, self.num_heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_dim ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.embed_dim
        )

        out = self.fc_out(out)
        return out
class GQA(nn.Module):
    def __init__(self, embed_dim, num_heads, num_groups):
        super(GQA, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = embed_dim // num_heads

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.num_heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)
        queries = query.reshape(N, query_len, self.num_heads, self.head_dim)

        energy = torch.einsum("nqgd,nkgd->nqgk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_dim ** (1 / 2)), dim=3)

        out = torch.einsum("nqgk,nkgd->nqgd", [attention, values]).reshape(
            N, query_len, self.embed_dim
        )

        out = self.fc_out(out)
        return out

class MQA(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MQA, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.num_heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)
        queries = self.queries(query).reshape(N, query_len, self.num_heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_dim ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.embed_dim
        )

        out = self.fc_out(out)
        return out
# 构造输入数据
embed_dim = 4  # 嵌入维度
num_heads = 2  # 注意力头的数量
batch_size = 1  # 批次大小
seq_len = 3   # 序列长度

# 随机生成输入数据
values = torch.rand(batch_size, seq_len, embed_dim)
keys = torch.rand(batch_size, seq_len, embed_dim)
queries = torch.rand(batch_size, seq_len, embed_dim)
print(queries)
print(keys)
print(values)

# 实例化模型
mha_model = MHA(embed_dim, num_heads)
# 得到结果
mha_result = mha_model(values, keys, queries)
# 打印结果
print("MHA Result:", mha_result)

mqa_model = MQA(embed_dim, num_heads)
mqa_result = mqa_model(values, keys, queries)
print("MQA Result:", mqa_result)

gqa_model = GQA(embed_dim, num_heads, 2)  # 假设我们有两个分组
gqa_result = gqa_model(values, keys, queries)
print("GQA Result:", gqa_result)












