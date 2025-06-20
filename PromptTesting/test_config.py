from langchain_ollama import ChatOllama
from langchain_core.language_models import BaseChatModel
from typing import *

def init_system(LLM_model_name='gemma3:4b') -> BaseChatModel:
    ''' Build system core objects '''
    # build messages
    f_msg = "Build {0:<10} : {1:<{2}} Complete"

    # llm
    LLM_temperature = 0.7
    LLM_url   = "http://localhost:11434"
    LLM_model = ChatOllama(
        model=LLM_model_name,
        temperature=LLM_temperature,
        base_url=LLM_url
    )
    print(f_msg.format("LLM", LLM_model_name, 10))

    return LLM_model

if __name__ == "__main__":
    LLM_model = init_system()