import requests
import json
import time
from qwen2_langchain import Qwen2_LLM


def get_fastapi(data):
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url='http://0.0.0.0:20456/worker_qa', headers=headers, data=json.dumps(data))
    return response.json()['response']

if __name__ == '__main__':
    # 调用fastapi接口
    data = {
        "prompt": "你是谁？",
        "max_new_tokens": 3,
    }
    start_time = time.time()
    print(get_fastapi(data))
    print(time.time()-start_time)

    # 调用langchain封装的qwen
    model_name_or_path = "/data3/home/llm/qwen2.5-3B-instruct"
    llm = Qwen2_LLM(model_name_or_path)
    start_time = time.time()
    print(llm("你是谁？"))
    print(time.time() - start_time)
