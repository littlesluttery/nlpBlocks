from transformers import AutoModel
from transformers import AutoTokenizer
from typing import List


def show_model_parameters(model_name_or_path:str):
    model = AutoModel.from_pretrained(model_name_or_path)
    # param_list = model.parameters(recurse=True)
    param_list = list(model.named_parameters(recurse=True))

    # 测算模型每一层的参数量
    p = 0
    for idx, i in enumerate(param_list):
        p += 1
        print(idx, i[0],i[1].numel())

def convert_ids_to_tokens(
    model_name_or_path:str,
    input_token_ids:List[int],
    lable__token_ids:List[int]
):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # tokened_text = tokenizer.convert_ids_to_tokens(token_ids)
    input_tokened_text = tokenizer.decode(input_token_ids)
    label_tokened_text= tokenizer.decode(lable__token_ids)
    print("输入样本为：")
    print(input_tokened_text)
    print("标签为：")
    print(label_tokened_text)


if __name__ == "__main__":

    model_name_or_path = 'qwen2-14B-Chat'
    show_model_parameters(model_name_or_path)

    input_token_ids = [151644,   8948,    198,   2610,    525,    264,  10950,  17847]
    lable__token_ids = [84169,     25,   6747,    374,    264,   3146, 448,    264,   9080,    323,  27730,  53142]

    convert_ids_to_tokens(model_name_or_path,input_token_ids,lable__token_ids)

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

#旋转位置编码
class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        # dim是指head dimension
        # max_position_embeddings是指token的最大长度
        # base是基础值,来源于原始transformer架构,Qwen2是1e6,base越大,长度外推能力越强
        # device是指计算所处的设备
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        # t.shape is [seq_len]
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        # freqs.shape is [seq_len, self.dim//2]
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # 这么做的目标是为了计算方便,具体计算是将张量最后一个维度分成两部分
        # [0, d//2 - 1], ..., [i, d//2 - 1 + i]成旋转对
        # emb.shape is [seq_len, self.dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        # self.cos_cached.shape is [seq_len, self.dim]
        # self.sin_cached.shape is [seq_len, self.dim]
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_dim]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        # 根据seq_len取出对应长度的张量
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    # 取前半部分
    # x1.shape and x2.shape are [batch_size, num_heads, seq_len, head_dim//2]
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    # rotate_half(x)的最后一个维度 [- x.shape[-1] // 2 :, : x.shape[-1] // 2]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    # position_ids.shape is [batch_size, seq_len]
    # cos.shape is [seq_len, head_dim]
    # cos[positon_ids].shape is [batch_size, seq_len, head_dim] ->
    # [batch_size, 1, seq_len, head_dim]
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    # 按照链接https://zhuanlan.zhihu.com/p/679819602中的公式来的,具体可看该链接
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
import random
import numpy

batch_size = 2
seq_len = 6
hidden_size = 8
head_dim = 4
num_attention_heads = 2

x = torch.rand((batch_size, num_attention_heads, seq_len, head_dim))

rotary_embedding = Qwen2RotaryEmbedding(head_dim)
cos, sin = rotary_embedding(x, seq_len)

q = torch.rand((batch_size, num_attention_heads, seq_len, head_dim))
k = torch.rand((batch_size, num_attention_heads, seq_len, head_dim))

position_ids = []
for _ in range(batch_size):
    start_id = 0
    position_ids.append(range(start_id, start_id+seq_len))

position_ids = torch.tensor(position_ids, dtype=torch.long)
print(position_ids)
print()
q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
print(q_embed.shape, k_embed.shape)



import random
import numpy

batch_size = 2
seq_len = 32
hidden_size = 8
head_dim = 4
num_attention_heads = 2

x = torch.rand((batch_size, num_attention_heads, seq_len, head_dim))

rotary_embedding = Qwen2RotaryEmbedding(head_dim)
cos, sin = rotary_embedding(x, seq_len)

# 做kv cache
q = torch.rand((batch_size, num_attention_heads, 1, head_dim))
k = torch.rand((batch_size, num_attention_heads, 1, head_dim))

position_ids = []
for _ in range(batch_size):
    start_id = random.randint(0, seq_len - 5)
    position_ids.append(range(start_id, start_id+1))

position_ids = torch.tensor(position_ids, dtype=torch.long)
print(position_ids)
print()
q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
print(q_embed.shape, k_embed.shape)

#MQA、GQA和MHA
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





