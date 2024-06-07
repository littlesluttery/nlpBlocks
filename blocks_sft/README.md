### 基于Qwen1.5-7B-Chat进行lora微调
目的：训练自我认知

blocks_sft：
    data：自我认知数据集
    output_qwen：lora微调输出结果
    Qwen1.5-7B-Chat-daqin：模型合并结果
    app.py:启动界面代码
    chat.py: 推理代码
    ds_zero_no_offlaod.json:deepspeed zero2 参数
    train_sft: 训练代码
    train.sh：训练脚本
### 快速使用
1. 安装环境

2. 开始训练，sh train.sh

3. 模型合并：python chat.py

4. 启动界面：python app.py
![alt text](image.png)
