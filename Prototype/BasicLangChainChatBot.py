from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_ollama import ChatOllama
import os, sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import config

'''
Upgrade of basic chat bot
Add in LangChain functions and tools
'''

# Setting
LLM_model = "gemma3:4b"
response_lang = "繁體中文"
role = "專業AI助理"
task = "回答使用者問題且適時自行補充資訊"
style = "禮貌清晰專業"

# Prompts
role_detail_prompt  = SystemMessagePromptTemplate.from_template("你是一個 \"{role}\"，現在要協助 \"{task}\" 的任務")
respond_stye_prompt = SystemMessagePromptTemplate.from_template("請盡量使用語言 \"{response_lang}\" 跟使用者聊天，並且要 \"{style}\" 的聊天風格回答")
greeting_prompt     = HumanMessagePromptTemplate.from_template(f"自我介紹一下，你是誰?")
initial_prompt = ChatPromptTemplate.from_messages([
    role_detail_prompt,
    respond_stye_prompt,
    greeting_prompt
])
user_input_prompt = ChatPromptTemplate.from_messages([
    respond_stye_prompt,
    SystemMessagePromptTemplate.from_template("上一輪對話內容為：\n{user_previous}\n{answer_previous}\n\n"),
    HumanMessagePromptTemplate.from_template("現在使用者輸入：\n{user_input}")
])

# Initialize
print(f"[ Starting {LLM_model} Chatbot ]")

llm = ChatOllama(
    model=LLM_model,
    temperature=0.7,
    base_url="http://localhost:11434"
)

chat_history = []

respond = llm.invoke(initial_prompt.format_messages(role=role, task=task, response_lang=response_lang, style=style))
print(f"{'[ LLM_model ]':<15} | {respond.content}")


while True:
    user_input = input(f"{'[ User ]':<15} | ")

    if user_input.lower() == "exit":
        print(f"[ Closing {LLM_model} Chatbot ]")
        break

    chat_history.append({"role": "user", "content": user_input})

    up = ap = "沒有上一輪對話"
    if len(chat_history) >= 2:
        up = chat_history[-2]
        ap = chat_history[-1]
    
    user_input = user_input_prompt.format_messages(
        user_input=user_input, 
        response_lang=response_lang, 
        style=style,
        user_previous=up,
        answer_previous=ap,
        )
    response = llm.invoke(user_input)

    answer = response.content.strip()
    chat_history.append({"role": "assistant", "content": answer})
    print(f"{'[ LLM_model ]':<15} | {answer}")
