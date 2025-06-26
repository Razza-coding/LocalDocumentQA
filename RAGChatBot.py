from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain_ollama import ChatOllama
import config
import os
from typing import *


''' 使用 RAG 製作歷史紀錄查詢與上下文整合功能 '''

# initial
LLM_model, VecDB = config.init_system()

# Setting
# 備註：標點符號會導致 LLM 更喜歡反問使用者，自然語言的提問效果最好
all_response_lang  = "繁體中文 English"
main_response_lang = "繁體中文"
negitive_rule = "使用簡體中文"
role = "專業AI助理"
task = "協助回答使用者問題"
style = """
多個段落完成回答
段落一、一句話敘述 LLM 打算回覆的內容
段落二、你是一名推薦助理，立即推薦使用者三個以上的選擇，不要提問
段落三、必要時，針對 LLM 不確定的地方再跟施用者確認，列出一些問題
"""

# Roles and Prompts
# System
system_setting_prompt = SystemMessagePromptTemplate.from_template(
"""聊天主要使用{main_response_lang}對話，也可能包含{all_response_lang}，不能{negitive_rule}。
請用以下格式回答\n
{style}
"""
)
system_profession_prompt = SystemMessagePromptTemplate.from_template(
"""你現在是一位{profession}
"""
)
# User format
user_input_prompt = HumanMessagePromptTemplate.from_template(
"""{user_input}
"""
)

# Initial message
greeting_prompt = HumanMessagePromptTemplate.from_template(
f"自我介紹一下，你是誰?"
)

# History Logging
# 備註：建立總結格式很有效
summary_prompt = SystemMessagePromptTemplate.from_template(
"""
請幫以下對話進行簡短總結

總結格式如下
一、寫一句主旨
二、依照對話內容列出標籤
三、寫出總結敘述

對話內容如下
使用者說
{user_input}

LLM回覆
{answer}
"""
)

#
def simple_prompt_combine(system_prompt: List, human_prompt: List, assitant_prompt: List=[]) -> str:
    msg = ""
    start_of_prompt = "遵照系統設定，並且適時使用補充資訊，來回答使用者輸入\n"
    sys_start = "系統設定如下\n"
    sys_end   = "\n"
    hum_start = "使用者輸入如下\n"
    hum_end   = "\n"
    ass_start = "補充資訊如下\n"
    ass_end   = "\n"

    msg += start_of_prompt

    if len(system_prompt) > 0:
        msg += sys_start
        for sys_p in system_prompt:
            msg += '\n'.join([sm.content for sm in sys_p])
        msg += sys_end

    if len(human_prompt) > 0:
        msg += hum_start
        for hum_p in human_prompt:
            msg += '\n'.join([hm.content for hm in hum_p])
        msg += hum_end

    if len(assitant_prompt) != 0:
        msg += ass_start
        for ass_p in assitant_prompt:
            msg += '\n'.join([hm.content for hm in ass_p])
        msg += ass_end
    
    return msg

# 你好訊息
print(f"{'[ LLM_model ]':<15} | Initialize AI Chat Bot")
init_msg = simple_prompt_combine(
    [
        system_setting_prompt.format_messages(all_response_lang=all_response_lang, main_response_lang=main_response_lang, style=style, negitive_rule=negitive_rule)
    ],
    [
        greeting_prompt.format_messages()
    ],
)
respond = LLM_model.invoke(init_msg)
print(f"{'[ LLM_model ]':<15} | {respond.content}")

# main chat loop
while True:
    # 使用者輸入
    user_input = input(f"{'[ User ]':<15} | ")
    if user_input.lower() == "exit":
        print(f"[ Closing {LLM_model} Chatbot ]")
        break

    # 搜索歷史紀錄
    retrive_content = []
    if VecDB.index.ntotal > 1:
        history_messages = VecDB.similarity_search_with_score(query=user_input, k=4, kwargs={"score_threshold": 1.2})
        for text, score in history_messages:
            if score > 1.2:
                continue
            print(f"{'[ History ]':<15} | ID : {text.id} Score : {score} Content : {text.page_content}")
            retrive_content.append(AIMessagePromptTemplate.from_template(text.page_content).format_messages())

    # 組合
    format_user_input = simple_prompt_combine(
        [
            system_setting_prompt.format_messages(all_response_lang=all_response_lang, main_response_lang=main_response_lang, style=style, negitive_rule=negitive_rule)
        ],
        [
            user_input_prompt.format_messages(user_input=user_input)
        ],
        retrive_content
    )

    # 輸入 LLM
    response = LLM_model.invoke(format_user_input)   

    # 回覆
    answer = response.content.strip()
    print(f"{'[ LLM_model ]':<15} | {answer}")

    # 做總結，更新歷史紀錄
    target_summary_message = summary_prompt.format_messages(user_input=user_input, answer=answer)
    target_summary_message = target_summary_message[0].content # easy format
    chat_summary = LLM_model.invoke(target_summary_message)
    VecDB.add_texts([f"{chat_summary.content}"])
    print(f"{'[ Coculsion ]':<15} | {chat_summary.content}")