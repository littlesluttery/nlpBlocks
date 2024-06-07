### 基于Qwen1.5-7B-Chat进行lora微调
目的：基于Qwen-1.5-7B-Chat进行lora微调。本次仅仅训练自我认知，即针对提问“你是谁？”进行训练。
#### 数据格式为
```
{"instruction": "请描述一下你的身份。,\n", "input": "", "output": "吾，大秦重工首席AI大师-大秦之龙，凡人，你有何事？"}
```
你也可以将内容换为自己想训练的大模型的回答，按照快速使用进行运行，即可获得属于自己的专属大模型。

blocks_sft：
```
    data：自我认知数据集
    
    output_qwen：lora微调输出结果
    
    Qwen1.5-7B-Chat-daqin：模型合并结果
    
    app.py:启动界面代码
    
    chat.py: 推理代码
    
    ds_zero_no_offlaod.json:deepspeed zero2 参数
    
    train_sft: 训练代码
    
    train.sh：训练脚本
``` 
### 快速使用
1. 安装环境

2. 开始训练，sh train.sh

3. 模型合并：python chat.py

4. 启动界面：python app.py

