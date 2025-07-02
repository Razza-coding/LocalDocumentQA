import ollama
import os

'''
A chat bot using ollama only, without langchain
Quick performance testing for Ollama llm models
'''

# Setting
LLM_model = "gemma3:4b"
response_lang = "繁體中文"
task = "回答使用者問題"
style = "簡節明瞭"

# Initialize
print(f"【 Starting {LLM_model} Chatbot 】")
chat_history = [
    {"role": "system", "content": f"你是一個 AI 助理，請協助使用者{task}"},
    {"role": "system", "content": f"你只能使用 {response_lang} 回答問題，並且要以 {style} 的風格回答"}
]

while True:
    user_input = input("【 User 】\n")

    if user_input.lower() == "exit":
        print(f"【 Closing {LLM_model} Chatbot 】")
        break
    
    chat_history.append({"role": "user", "content": f"{user_input}\n請使用{response_lang}語言和{style}風格回答"})

    response = ollama.chat(model=LLM_model, messages=chat_history)

    answer = response["message"]["content"]
    chat_history.append({"role": "assistant", "content": answer})

    print(f"【 LLM_model 】\n{answer}")
