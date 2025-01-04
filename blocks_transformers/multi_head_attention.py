import torch
from torch import nn 
import torch.functional as F
import math


class multi_head_attention(nn.Module):
    def __init__(self,d_model,n_head) -> None:
        super(multi_head_attention,self).__init__()

        self.n_head = n_head
        self.d_model =d_model
        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)
        self.w_combine  = nn.Linear(d_model,d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,q,k,v):
        batch, time, dimension = q.shape
        n_d = self.d_model // self.n_head
        q,k,v = self.w_q(q), self.w_k(k), self.w_v(v)

        q = q.view(batch,time, self.n_head,n_d).permute(0,2,1,3)
        k = k.view(batch,time, self.n_head,n_d).permute(0,2,1,3)
        v = v.view(batch,time, self.n_head,n_d).permute(0,2,1,3)

        score = q @ k.transpose(2,3) / math.sqrt(n_d)
        mask = torch.tril(torch.ones(time,time,dtype=bool))
        score = score.masked_fill(mask==0, float("-inf"))
        score = self.softmax(score) @ v

        score = score.permute(0,2,1,3).contiguous().view(batch,time,dimension)

        output = self.w_combine(score)
        return output




if __name__ == "__main__":

    X = torch.randn(128,64,512)
    print(X.shape)

    d_model = 512
    n_head = 8
    attention = multi_head_attention(d_model,n_head)
    output = attention(X,X,X)
    print(output)
    print(output.shape)

# 关于transformers中generate函数已经kv cache等
from transformers.models.qwen2 import Qwen2ForCausalLM,Qwen2Tokenizer
import torch
model_name_or_path = "/data3/home/llm/qwen2-1.8B-Chat"
model = Qwen2ForCausalLM.from_pretrained(model_name_or_path,device_map="cpu")
tokenizer = Qwen2Tokenizer.from_pretrained(model_name_or_path)
# print(tokenizer)

def decoder(token_ids,tokenizer):
    if isinstance(token_ids,torch.Tensor):
        pass
    else:
        token_ids = torch.tensor(token_ids)
    # print(token_ids)
    token = tokenizer.batch_decode(token_ids,skip_special_tokens=True)
    # print(token)
    return token

def demo1(text,tokenizer,model,max_new_tokens):
    # 调用generate方法，让它一直生成新的token
    model_inputs = tokenizer(
        [text],
        return_tensors="pt",
    ).to(model.device)
    print(model_inputs)
    for k,v in model_inputs.items():
        print(k,v)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        top_p=1.0

    )
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids,output_ids in zip(model_inputs.input_ids,generated_ids)
    ]
    print(generated_ids)
    response = tokenizer.batch_decode(generated_ids,skip_special_tokens=True)[0]
    print(response)

def demo2(model,input_id,tokenizer):
    # 基于第一步，生成新token
    model_inputs1 = {
        "input_ids":torch.tensor(
            data=[input_id],dtype=torch.long
        ).to(model.device),
        "attention_mask":torch.tensor(
            data=[[1 for i in range(len(input_id))]],dtype=torch.long
        ).to(model.device)
    }
    model_outputs1 = model.forward(
        **model_inputs1,
        use_cache=False,
    )
    # print(model_outputs1.keys())         # odict_keys(['logits', 'past_key_values']) 结果包含两部分，一部分是概率值，一部分是kv缓存值。
    # print(model_outputs1.logits.shape)   # torch.Size([1, 3, 151936])  ---> [batch_size,seq_len,vocab.logits]
    # print(model_outputs1.logits[:,-1,:].shape) # torch.Size([1, 151936])
    # print(model_outputs1.logits[:,-1,:].argmax(dim=-1)) # tensor([9909], device='cuda:0')
    # print(
    #     type(model_outputs1.past_key_values),
    #     len(model_outputs1.past_key_values),
    #     len(model_outputs1.past_key_values[0]),
    #     model_outputs1.past_key_values[0][0].shape
    # )  # <class 'tuple'> 24 2 torch.Size([1, 16, 3, 128])

    new_token_id_tensor = model_outputs1.logits[:,-1,:].argmax(dim=-1)
    new_token_id = new_token_id_tensor.tolist()
    input_id.append(new_token_id[0])
    # print(new_token_id)
    new_token = decoder(new_token_id,tokenizer)
    # print(new_token)
    all_token = decoder(input_id,tokenizer)
    return input_id,new_token,all_token,model_outputs1




# token_ids = [105043, 100165] ['你是','谁']
# token_ids = [ 58695,  20412, 102506, 102055, 106654,  99599] # ['中国', '是', '世界上', '人口', '最多的', '国家']
# token_ids = [105043, 100165, 11319] # ['你是', '谁', '？']
# token_ids = [9909]
# token = decoder(token_ids,tokenizer)
# print(token)


# text= "你是谁"
# 按照输入，调用generate方法一直生成，知道遇到eos或者max_new_tokens到长度限制。
# demo1(text,tokenizer,model,max_new_tokens=100)
# 基于上一步token，生成新的一个token
# input_id = [105043, 100165, 11319]
# for  i in range(100):
#     input_id,new_token,all_token = demo2(model,input_id,tokenizer)
#     # print(new_token)
#     if new_token == ['<|endoftext|>']:
#         print("停止")
#         break
#
# response = ""
# for token in all_token:
#     response += token
# print(response)
# print(input_id,all_token)



text = '你是谁'
model_inputs = tokenizer(
        [text],
        return_tensors="pt",
).to(model.device)
print(model_inputs)
input_id = model_inputs['input_ids'].tolist()[0]
print(input_id)
for i in range(2):
    input_id,new_token,all_token,model_outputs1 = demo2(model,input_id,tokenizer)
    print(input_id)
    # print(new_token)
    print(all_token)
    print(model_outputs1.logits)




