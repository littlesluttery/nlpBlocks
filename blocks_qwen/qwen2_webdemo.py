from transformers import Qwen2Tokenizer,Qwen2ForCausalLM
import streamlit as st


with st.sidebar:
    st.markdown("## Qwen2.5 LLM")
    "开源大模型"
    max_length = st.slider("max_length",0,8192,512,step=1)

# 创建标题和副标题
st.title("Qwen2.5 CHatBot")
st.caption("一个聊天机器人！！！")

# 模型路径
model_name_or_path = "/data3/home/llm/qwen2.5-3B-instruct"

@st.cache_resource
def get_model():
    tokenizer = Qwen2Tokenizer.from_pretrained(model_name_or_path,trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = Qwen2ForCausalLM.from_pretrained(model_name_or_path)

    return tokenizer,model

# 获取模型及分词器
tokenizer, model = get_model()

if "message" not in st.session_state:
    st.session_state["messages"] = [
        {"role":"assistant","content":"有什么可以帮助您的？"}
    ]

# 遍历消息，在界面上显示。
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 如果用户在输入框中输入了内容，则执行以下操作
if prompt := st.chat_input():
    # 在聊天界面上显示用户的输入
    st.chat_message("user").write(prompt)
    # 将用户输入添加到session_state中的message列表中
    st.session_state.messages.append({"role":"user","content":prompt})
    # 将对话输入模型，获得返回。
    input_ids = tokenizer.apply_chat_template(
        st.session_state.messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer(
        [input_ids],
        return_tensors="pt"
    ).to(model.device)
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=max_length
    )
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs,generated_ids)
    ]
    response = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    # 将模型输出添加到session_state中的message列表中
    st.session_state.messages.append({"role":"assistant","content":response})
    # 在聊天界面上显示模型的输出
    st.chat_message("assistant").write(response)


