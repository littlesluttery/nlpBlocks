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
