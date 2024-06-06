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


