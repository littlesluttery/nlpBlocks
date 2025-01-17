### 关于qwen的使用

##### 1.使用fastapi拉起模型，并采用request发送请求
```markdown
1. 在qwen2_fastapi.py中修改模型路径为你本地的路径
    model_name_or_path = "你的模型路径"
执行python qwen2_fastapi.py，会在本地拉起一个模型：http://0.0.0.0:20456/worker_qa
2. 在main.py中采用request调用
备注：可进一步根据自己的需求修改，如修改生成参数，加入新的生成参数。
```
##### 2.实现类似langchain调用
```markdown
在main.py中修改自己的模型路径，实现类似langchain的调用。
```
##### 3.界面问答
```markdown
# 执行以下操作
# streamlit run qwen2_webdemo.py --server.address yourip --server.port 20456
```
##### 4.o1推理
```markdown
采用vllm拉起模型
python -m vllm.entrypoints.openai.api_server --model /data3/home/llm/qwen2.5-3B-instruct  --served-model-name Qwen2.5-3B-Instruct --max-model-len=32768 --port 8000

启动界面
streamlit run qwen_o1.py --server.address 0.0.0.0 --server.port 20456

```
