from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import trim_messages
from langchain.callbacks.base import BaseCallbackHandler
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from datetime import datetime
import os, sys, re
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import config
from LogWriter import LogWriter

# ===============================
# Config and Initialize
# ===============================

LLM_model, _ = config.init_system()
now = datetime.now().strftime("%Y%m%d_%H%M%S")
MAX_CONTEXT_WINDOW = 128,000 # maximum token of gemma3:4b

# Log Files
log_model_name = getattr(LLM_model, "model", "UnknownModel")
log_model_name = re.sub(r'[^\w\s.-]', '_', log_model_name) # Replace special symbol into underline
logger = LogWriter(log_name = log_model_name, log_folder_name = "test_log", root_folder = "./Prototype")

# call back for debug logging
class TraceLLMInputHandler(BaseCallbackHandler):
    def __init__(self):
        self.last_prompt = None

    # log raw LLM input inside LangGraph
    def on_llm_start(self, serialized, prompts, **kwargs):
        if prompts:
            self.last_prompt = prompts[0]
            logger.write_log(self.last_prompt, "LLM Input")

# hook LLM callback
callback_handler = TraceLLMInputHandler()
LLM_model.callbacks = [callback_handler]

# ===============================
# Prompt Setting
# ===============================

prompt_vars = {
    "LLM_name": "JHON",
    "main_response_lang": "繁體中文 英文 日文",
    "negitive_rule": "使用簡體中文",
    "style": '''\nSummary: 一句話簡單回答 Human\nRespond: 你是一名推薦助理，立即推薦 Human 一個以上的選擇，不要提問\nQuestion: 列出一些問題，預測需求 Human 並詢問是否執行'''
}

# system setting message + history messages
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are {LLM_name}, a helpful assistant who speaks {main_response_lang} and avoids {negitive_rule}. Use the following format:{style}"),
    MessagesPlaceholder(variable_name="messages")
])

# token trimmer
trimmer = trim_messages(
    max_tokens=1024,
    strategy="last", # "first" "last", what token prioritize to keep
    token_counter=LLM_model,
    include_system=True, # keep system message at index 0
    allow_partial=False, # not allow partial message to send in
    start_on="human", # combine with strategy last, start with human message
)

# ===============================
# LangGraph Create
# ===============================

class MessageState(TypedDict):
    messages: Annotated[list, add_messages]  # Chat History

workflow = StateGraph(state_schema=MessageState)

# Create Graph Node
def call_model(state: MessageState):
    print(state["messages"])
    trimmed = trimmer.invoke(state["messages"]) # short memory system, trim message and combine it
    logger.write_log(trimmed, "History Message")
    prompt = prompt_template.invoke({**prompt_vars, "messages": trimmed}) # append messages into prompt
    response = LLM_model.invoke(prompt) # LLM input
    return {"messages": [response]}

workflow.add_node("chat", call_model)
workflow.set_entry_point("chat")
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# ===============================
# Main Loop
# ===============================

print(f"{'[ LLM_model ]':<15} | Initialize LangGraph ChatBot")
config_base = {"configurable": {"thread_id": "session1"}}  # 可動態改變

while True:
    user_input = input(f"{'[ User ]':<15} | ")
    if user_input.lower() == "exit":
        print("[ ChatBot Closed ]")
        break

    input_messages = [HumanMessage(user_input)]
    state_input = {"messages": input_messages, "language": "繁體中文"}  # 可加入其他參數

    output = app.invoke(state_input, config_base)
    ai_response = output["messages"][-1]

    print(f"{'[ LLM_model ]':<15} | {ai_response.content}")

    # logging
    logger.write_log(user_input, "User")
    logger.write_log(ai_response.content, "Respond")
    logger.write_s_line(2)