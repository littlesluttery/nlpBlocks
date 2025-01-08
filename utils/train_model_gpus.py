"""
估算训练一个模型需要多少显存。
"""
import os
import json
import re

def read_config(model_name_or_path):
    if os.path.exists(model_name_or_path):
        config_path = os.path.join(model_name_or_path,'config.json')
        model_size = model_name_or_path.split("/")[-1].split("-")[1]
        with open(config_path,encoding="utf-8") as file:
            config = json.load(file)
        num_attention_heads = config.get('num_attention_heads',None)
        num_hidden_layers = config.get('num_hidden_layers',None)
        hidden_size = config.get('hidden_size',None)

        return model_size,num_attention_heads,num_hidden_layers,hidden_size
    else:
        raise ValueError("The model_name_or_path is not exists! ")


def calculate_train_model(
        model_name_or_path,
        batch_size,
        seq_len,

):
    model_size,num_attention_heads,num_hidden_layers,hidden_size = read_config(model_name_or_path)
    # 提取模型大小
    size = int(re.findall(r"[0-9]\d+",model_size)[0])
    print(size)
    # 计算模型参数、梯度和优化器状态，大约是参数量的20倍
    gpu_one = size * 20 * 1 * 1e10 / 1e10
    print(gpu_one)
    # 计算激活器显存,大约是（34bsh+5bs²α）*l
    # b---> batch_size
    # s---> seq_len
    # h---> hidden_size
    # α---> num_attention_heads
    # l---> num_hidden_layers
    gpu_two = (34 * batch_size * seq_len * hidden_size + 5 * batch_size * seq_len * seq_len * num_attention_heads) * num_hidden_layers / 1e9
    print(gpu_two)
    gup_all = gpu_one + gpu_two
    return gup_all





model_name_or_path = '/data3/home/llm/qwen2-14B-Chat'
batch_size = 1
seq_len = 8192
gup_all = calculate_train_model(model_name_or_path,batch_size,seq_len)
print(gup_all)
