from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_ollama import ChatOllama
from langchain.chains import ConversationChain
from langchain.memory import ConversationTokenBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
import os, sys
from datetime import datetime

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import config

''' 測試 LangChain 的 Memory 功能 '''

# ✅ 自訂 callback 來追蹤 LLM 輸入
class TraceLLMInputHandler(BaseCallbackHandler):
    def __init__(self):
        self.last_prompt = None

    def on_llm_start(self, serialized, prompts, **kwargs):
        # 只記錄第一組 prompt
        if prompts:
            self.last_prompt = prompts[0]
            print(f"{'[ Debug ]':<15} | LLM Input:\n{self.last_prompt}")

# 初始化 LLM 與 Callback
callback_handler = TraceLLMInputHandler()
LLM_model, _ = config.init_system()
LLM_model.callbacks = [callback_handler]

# Prompt 設定
all_response_lang  = "繁體中文 English"
main_response_lang = "繁體中文"
negitive_rule = "使用簡體中文"
LLM_name = "JHON"
role = "專業AI助理"
task = "協助回答使用者問題"
style = """
使用 JSON 格式完成回答
Summary : 一句話敘述 LLM 打算回覆的內容
Respond : 你是一名推薦助理，立即推薦使用者一個以上的選擇，不要提問
Suggestion : 列出一些問題，預測需求使用者並詢問使否執行
"""

# 建立紀錄檔案
os.makedirs("test_log", exist_ok=True)
now = datetime.now().strftime("%Y%m%d_%H%M%S")
log_model_name = LLM_model.model if hasattr(LLM_model, "model") else "UnknownModel"
log_file = f"./test_log/{now}_{log_model_name}.txt"

# 記憶機制
memory = ConversationTokenBufferMemory(
    llm=LLM_model,
    max_token_limit=4000,
    return_messages=True
)

# Prompt 組合
system_setting_prompt = SystemMessagePromptTemplate.from_template(
"""你叫做{LLM_name}，是一位{role}，聊天時主要使用{main_response_lang}對話，也可能包含{all_response_lang}，不能{negitive_rule}。
請用以下格式回答：
{style}
"""
)
system_prompt = system_setting_prompt.format_messages(
    LLM_name=LLM_name,
    role=role,
    all_response_lang=all_response_lang,
    main_response_lang=main_response_lang,
    negitive_rule=negitive_rule,
    style=style
)[0].content

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("ai", "{history}"),
    ("human", "{input}")
])

# ConversationChain
conversation = ConversationChain(
    llm=LLM_model,
    prompt=chat_prompt,
    memory=memory,
    verbose=False
)

# 啟動對話
print(f"{'[ LLM_model ]':<15} | Initialize AI Chat Bot")

with open(log_file, "w", encoding="utf-8") as f:
    while True:
        user_input = input(f"{'[ User ]':<15} | ")
        if user_input.lower() == "exit":
            print(f"[ ChatBot Closed ]")
            break

        response = conversation.predict(input=user_input)
        print(f"{'[ LLM_model ]':<15} | {response}")

        # 寫入紀錄（含 prompt）
        f.write(f"[User]：{user_input}\n")
        f.write(f"[LLM Input]：{callback_handler.last_prompt}\n")
        f.write(f"[LLM ]：{response}\n\n")
