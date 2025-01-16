from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from typing import Any,List,Optional
from transformers import Qwen2Tokenizer,Qwen2ForCausalLM,GenerationConfig


class Qwen2_LLM(LLM):
    tokenizer:Qwen2Tokenizer = None
    model:Qwen2ForCausalLM = None

    def __init__(self,model_name_or_path):
        super(Qwen2_LLM, self).__init__()
        self.tokenizer = Qwen2Tokenizer.from_pretrained(model_name_or_path,use_fast=False)
        self.model = Qwen2ForCausalLM.from_pretrained(model_name_or_path)
        self.model.generation_config = GenerationConfig.from_pretrained(model_name_or_path)

    def _call(
            self,
            prompt:str,
            stop:Optional[List[str]] = None,
            run_manager:Optional[CallbackManagerForLLMRun] = None,
            **kwargs:Any
    ):
        messages = [
            {"role":"system","content":prompt},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer(
            [input_ids],
            return_tensors="pt",
        ).to(self.model.device)
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=3
        )
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        return response

    def _llm_type(self) -> str:
        return "Qwen2_LLM"


