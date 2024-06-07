from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel
from train_sft import PROMPT_DICT
import os
from typing import List


def generate_input(
        instruction: str, 
        input_str: str = ""
    ) -> str:
    prompt_input, prompt_no_input = (
        PROMPT_DICT["prompt_input"],
        PROMPT_DICT["prompt_no_input"],
    )

    if input_str != "":
        res = prompt_input.format_map({"instruction": instruction, "input": input})
    else:
        res = prompt_no_input.format_map({"instruction": instruction})
    return res


def batch_generate_data(
        model:AutoModelForCausalLM,
        tokenizer:AutoTokenizer,
        text_input: List[str], 
        use_train_model: bool = True, 
        temp: float = 0.7
    ):
    text_input_format = [generate_input(i) for i in text_input]
    batch_inputs = tokenizer.batch_encode_plus(
        text_input_format, padding="do_not_pad", return_tensors="pt"
    )
    batch_inputs["input_ids"] = batch_inputs["input_ids"].cuda()
    batch_inputs["attention_mask"] = batch_inputs["attention_mask"].cuda()

    if use_train_model:
        outputs = model.generate(
            **batch_inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=temp,
            top_p=0.8,
        )
    else:
        with model.disable_adapter():
            outputs = model.generate(
                **batch_inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=temp,
                top_p=0.8,
            )
    outputs = tokenizer.batch_decode(
        outputs.cpu()[:, batch_inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    )

    return outputs



def chat(
    base_model_name_or_path:str,
    lora_model_name_or_path:str,
    text_input:List[str],
    is_merge_and_save:bool=False,
    save_model_name_or_path:str=None

    ):
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        torch_dtype="auto",
        trust_remote_code=True,
    ).cuda(0)

    model = PeftModel.from_pretrained(model, model_id=lora_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name_or_path, 
        trust_remote_code=True, 
        padding_side="left"
    )
    model.eval()

    response_lora = batch_generate_data(
        model=model,
        tokenizer=tokenizer,
        text_input=text_input,
        use_train_model=True,
        temp=0.8
    )
 
    response = batch_generate_data(
        model=model,
        tokenizer=tokenizer,
        text_input=text_input,
        use_train_model=True,
        temp=0.8
    )
    if is_merge_and_save:
        merge_and_save(model,tokenizer,save_model_name_or_path)

    return response_lora,response


def merge_and_save(
        model:AutoModelForCausalLM,
        tokenizer:AutoTokenizer,
        save_model_name_or_path:str
    ):
    # merge lora model and asve 保存模型
    model = model.merge_and_unload()
    model.save_pretrained(save_model_name_or_path)
    tokenizer.save_pretrained(save_model_name_or_path)


if __name__ == "__main__":
    base_model_name_or_path = "/data3/home/llm/test/Qwen1.5-7B-Chat"
    lora_model_name_or_path = "./output_qwen/checkpoint-5"
    
    text_input = ["你是谁？"]
    response_lora,response = chat(
        base_model_name_or_path,
        lora_model_name_or_path,
        text_input=text_input,
        is_merge_and_save=True,
        save_model_name_or_path="Qwen1.5-7B-Chat-daqin"

    )
    # TODO: 原来回答为空bug待修复
    print(f"原来的模型回答是：{response}")
    print(f"lora训练后的模型回答是：{response_lora}")
