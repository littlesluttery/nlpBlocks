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