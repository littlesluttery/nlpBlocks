import json
from datetime import datetime
from operator import itemgetter

import dashscope
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dashscope import Generation
from langchain.tools.render import render_text_description
# from langchain.utilities import PythonREPL
from langchain_community.utilities import PythonREPL
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool

dashscope.api_key = ""


class react_qwen():

    def __init__(self, tool_list) -> None:
        self.tool_list = tool_list
        self.tools = [convert_to_openai_tool(i) for i in tool_list]
        self.toool_name = [i["function"]["name"] for i in self.tools]
        self.parser = JsonOutputParser()


    def prompt_qwen(self,content):
        system_prompt_t = f"""Use the following format:
                Question: the input question you must answer.
                Thought: you should always think about waht to do.
                Action: the action to take in order, should be one of {self.toool_name}
                Action Input: the input to the action.
                Observation: the result of the action...(this Thought/Action Input/Observation can repeat N times.)
                Thought: I now know the final answer.
                Final Answer: the final answer to the original input question.
                Begin!
                Question:{content}
                Thought:
        """
        prompt = [{
            "role":"system",
            "content":"Answer the following questions as best you can. You have access to the following tools"
        }]
        prompt.append({
            "role":"user",
            "content":system_prompt_t
        })
        return prompt
    
    def get_response_qwen(self,messages):
        response = Generation.call(
            model='qwen-turbo',
            messages=messages,
            tools=self.tools,
            result_format='message'
        )
        return response

    def parser_content(self,out_content):
        return {
            "name":out_content.split("\nAction: ")[1].split("\nAction")[0],
            "arguments":self.parser.parse(out_content.split("Input: ")[1])
        }

    def tool_chain(self,model_output):
        tool_map = {tool.name:tool for tool in self.tool_list}
        chosen_tool = tool_map[model_output["name"]]
        return itemgetter("arguments") | chosen_tool
    
    def invoke(self,input_p):
        prompt = self.prompt_qwen(input_p)
        for i in range(0,5):
            res = self.get_response_qwen(prompt)
            res_content = res.output.choices[0].message["content"]
            if res_content.find("\nAction: ") != -1:
                tool_args = self.parser_content(res_content)
                tool_out = self.tool_chain(tool_args)
                prompt[1]["content"] = prompt[1]["content"] +res_content+"\nObservation: " + str(tool_out.invoke(tool_args)) + "\nThought:"
            else:
                prompt[1]["content"] = prompt[1]["content"] +res_content
                break
        return(prompt[1]["content"])
    


@tool
def multiply(first_int:int, second_int:int) -> int:
    """将两个整数进行相乘"""
    return first_int * second_int

@tool
def add(first_int:int, second_int:int) -> int:
    """将两个数相加"""
    return first_int + second_int

@tool
def exponentiate(base:int,exponent:int) -> int:
    """对底数求指数幂"""
    return base**exponent

if __name__ == "__main__":
    tools = [multiply,add,exponentiate]
    chuxing =  react_qwen(tools)
    res = chuxing.invoke("5加7乘以6，然后在求结果的2次幂")
    print(res)

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

from typing import Optional,List
from langchain_core.pydantic_v1 import BaseModel,Field
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.utils.function_calling import convert_to_openai_tool
from dashscope import Generation
import dashscope

dashscope.api_key = ""

class Person(BaseModel):
    """Information about a person."""
    name: Optional[str]  = Field(
        default=None,
        description="The name of the person"
    )
    hair_color:Optional[str] = Field(
        default=None,
        description="The color of the person's hair if know."

    )
    height_in_meters: Optional[str] = Field(
        default=None,
        description="Height measured in meters"
    )

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "return null for the attribute's value. ",

        )
    ]
)


def get_response_qwen(message):
    response = Generation.call(
        model="qwen-plus",
        messages=message,
        tools=[convert_to_openai_tool(Person)],
        result_format="message"
    )
    return response

def prompt_ner(input):
    prompt_sys = "You are an expert extraction algorithm. Only extract relevant information from the text. If you do not know the value of an attribute asked to "
    return [
        {
            "role":"system",
            "content":prompt_sys,
        },
        {
            "role":"user",
            "content":input
        }
    ]

# 单个对象的抽取
prompt_ner = prompt("Alan Smith is 6 feet tall and has blood hair.")
res = get_response_qwen(prompt_ner)
print(res.outpuut.choices[0].message)


# 多个对象的抽取
class Data(BaseModel):
    """Extract data about people."""

    people:List[Person]


