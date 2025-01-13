from fastapi import FastAPI, Request
from transformers import AutoTokenizer,Qwen2ForCausalLM
import uvicorn
import json
import datetime


# 创建FastApi应用
app = FastAPI()


@app.post("/worker_qa")
async def request_qwen2_5(request:Request):
    global tokenizer,model

    post_json = await request.json()
    post_json = json.dumps(post_json)
    post_json_dict = json.loads(post_json)

    prompt = post_json_dict.get('prompt')
    max_new_tokens = post_json_dict.get('max_new_tokens')

    messages = [
        {"role":"system","content":"You are a helpful assistant."},
        {"role":"user","content":prompt}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer(
        [input_ids],
        return_tensors="pt",
    ).to(model.device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=max_new_tokens
    )
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids,output_ids in zip(model_inputs.input_ids,generated_ids)
    ]
    response = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    if len(response) > 0:
        status = 200
    else:
        status = 500

    answer = {
        "response":response,
        "status":status,
        "time":time
    }
    return answer


if __name__ == "__main__":

    # 加载模型和分词器
    model_name_or_path = "/data3/home/llm/qwen2.5-3B-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    model = Qwen2ForCausalLM.from_pretrained(model_name_or_path)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=20456,
        workers=1
    )


