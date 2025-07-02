from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_ollama import ChatOllama
import test_config
import os, sys
import json
import re
from datetime import date
from tqdm import tqdm

# Import parent path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import config # config for Main.py 
import test_config # config for test enviroment

# Init
LLM_model_name = "gemma3:4b"
LLM_model = test_config.init_system(LLM_model_name)
test_subject = "RAG_info_test"

# Log
date_time = date.today().strftime("%Y%m%d")
log_model_name = re.sub(r"[^\w\s.-]", "_", LLM_model_name)
log_folder = os.path.join(".", "PromptTestin/test_log")
os.makedirs(log_folder, exist_ok=True)
log_txt_path = os.path.join(log_folder, f"{date_time}_{log_model_name}_{test_subject}.txt")

# Question
question_and_info = [
    {
        "question": "德川幕府的「參勤交代」制度對於幕藩體制有什麼影響？",
        "info": "參勤交代是江戶幕府實施的一種政策，要求各藩主定期前往江戶居住，藉此分散地方勢力並控制軍事財政。",

        "output_format": {
            "政策名稱": "(名稱與年份)",
            "影響層面": "(說明影響層面)",
            "詳細說明": "(條列式說明每項影響，1~3點)"
        }
    },
    {
        "question": "我想做一個語音客服系統，請幫我列出整體開發流程與建議技術",
        "info": "語音客服系統需處理語音輸入、自然語言理解、回答生成與回覆語音化，常用技術有 Google Speech API、Rasa、TTS 模組。",

        "output_format": {
            "步驟": "(決定步驟)",
            "技術建議": "(列出技術與相關工具)"
        }
    },
    {
        "question": "請幫我寫一篇文章，主題是「重新認識自我」，風格要像村上春樹",
        "info": "村上春樹的文風偏意識流、孤獨感、帶有內在思索與奇幻象徵。情境多以夜晚、音樂、獨行人物展開。",

        "output_format": {
            "主題": "不限",
            "段落數": "3~5",
        }
    },
    {
        "question": "我今天有點累，不太想走太遠，有沒有什麼晚餐推薦？",
        "info": "使用者住在永和、喜歡日式麵食、不吃辣、只能步行外出，預算 250 元以下。",

        "output_format": {
            "推薦數量": 5,
            "餐點類型": "不限",
            "格式": [
                {"店名": "", "距離": "公尺", "預算": "元", "特色": ""}
            ]
        }
    },
    {
        "question": "如何建立一個簡單的 AI 問卷推薦系統，請依步驟說明資料處理與模型建構方式",
        "info": "推薦系統通常需要進行問卷分類、使用者特徵提取與模型訓練，可採用 TF-IDF 或 Embedding 相似度計算。",

        "output_format": {
            "步驟": "(自訂步驟)",
            "輸出格式": "(自行設計輸出格式)"
        }
    }
]

def message_parser(msg_list):
    return ''.join([m.content for m in msg_list])

with open(log_txt_path, 'w', encoding='utf-8') as f:
    f.write(f"{date_time}\nModel: {LLM_model_name}\nSubject: {test_subject}\n\n")

result_json = []
separate_line = "\n" + "-" * 80 + "\n"

for q in tqdm(question_and_info):
    question = q["question"]
    info = q["info"]
    output_format = json.dumps(q["output_format"], ensure_ascii=False, indent=2)

    for with_info in [False, True]:
        sys_text = ""
        if with_info:
            sys_text += f"補充資訊：{info}\n\n"
        sys_text += f"請依照以下格式回答：{output_format}\n"

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("{sys_text}"),
            HumanMessagePromptTemplate.from_template("使用者輸入：\n{question}")
        ])

        messages = prompt.format_messages(sys_text=sys_text, question=question)
        response = LLM_model.invoke(message_parser(messages))

        with open(log_txt_path, 'a', encoding='utf-8') as f:
            f.write(separate_line)
            f.write(f"Question:\n{question}\n")
            if with_info:
                f.write(f"With Info:\n{info}\n")
            else:
                f.write(f"With Info:\nNone\n")
            f.write(f"Format:\n{output_format}\n")
            f.write("Response:\n")
            f.write(response.content.strip() + "\n")