from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_ollama import ChatOllama
import os, sys
from typing import *
from datetime import date
from tqdm import tqdm
import regex as re

# Import parent path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import config # config for Main.py 
import test_config # config for test enviroment

# Note
'''
只使用專業身份增強 LLM 回答
'''

# Initial
LLM_model_name = "gemma3:4b"
LLM_model = test_config.init_system(LLM_model_name)
test_subject = "professional_based_prompt_test"

# Setting
roles = [
    "旅遊", "科技新知",
    "心理諮詢", "廚師",
    "電影旁白", "行程安排",
    "Python程式編輯", "幼兒園老師"
]

questions = [
    "有推薦的週末親子旅遊景點嗎？",
    "我最近壓力很大，該怎麼辦？",
    "iPhone 和 Android 哪個好？",
    "幫我設計一份讓小朋友更願意吃蔬菜的菜單。",
    "能不能幫我寫一個轉盤抽獎程式？",
    "我想找地方吃晚餐，有沒有推薦的地方？",
    "幫我寫一個簡短的劇本對話，主題不限"
]

# Prompts
role_question_prompt  = SystemMessagePromptTemplate.from_template("""你叫做JHON，擁有{role}專業知識，現在立即給予使用者答覆，使用者說: {question}""")

# create log
date_time = date.today().strftime("%Y%m%d")
encoding = 'utf-8'
log_model_name =  re.sub(r'[^\w\s.-]', '_', LLM_model_name)
test_log = os.path.join(os.path.abspath("."), (f"PromptTesting/test_log/{date_time}_{log_model_name}_{test_subject}.txt"))
with open(test_log, 'w', encoding=encoding) as f:
    f.write(date_time+"\n")
    f.write(f"Roles:\n{roles}\n")
    f.write(f"Question:\n{questions}\n")

def message_parser(format_message):
    msg = ""
    for m in format_message:
        msg += m.content
    return msg

# Test for formated message
seperate_line = "\n" + '-'*80 + "\n"
for role in tqdm(roles):
    for question in questions:
        question_message = role_question_prompt.format_messages(role=role, question=question)
        response = LLM_model.invoke(message_parser(question_message))
        with open(test_log, 'a', encoding=encoding) as f:
            combo = f"| Role: {role} | Question: {question} |\n"
            result = response.content + "\n"
            f.write(seperate_line)
            f.write(combo)
            f.write(result)
        