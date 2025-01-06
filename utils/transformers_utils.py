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


