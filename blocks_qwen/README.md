### 关于qwen的使用

##### 1.使用fastapi拉起模型，并采用request发送请求
```markdown
1. 在qwen2_fastapi.py中修改模型路径为你本地的路径
    model_name_or_path = "你的模型路径"
执行python qwen2_fastapi.py，会在本地拉起一个模型：http://0.0.0.0:20456/worker_qa
2. 在main.py中采用request调用
备注：可进一步根据自己的需求修改，如修改生成参数，加入新的生成参数。
```
