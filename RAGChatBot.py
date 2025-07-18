from langchain_core.messages import trim_messages, HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, MessagesPlaceholder, BasePromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.language_models import BaseChatModel

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver

import config
import os, sys, re
from typing import *
from CLI_Format import *
from LogWriter import LogWriter

from PromptTools import UserTemplateInputVar, get_user_message
from PromptTools import SystemTemplateInputVar, get_system_message
from PromptTools import ChatHistoryTemplateInputVar, get_chat_history_message
from PromptTools import GeneralChatTemplateInputMessages, make_general_input
from PromptTools import RAGSummaryChatTemplateInputMessages, make_RAG_summary_prompt
from PromptTools import to_list_message, to_message

# ===============================
# initial
LLM_model, VecDB = config.init_system()

# ===============================
# Setting
MAX_CONTEXT_WINDOW = 128,000 # maximum token of gemma3:4b
AI_name = "JOHN"
professional_role = "專業AI助理"

system_setting = SystemTemplateInputVar(
        AI_name = AI_name,
        professional_role = professional_role,
    )

# ===============================
# Log writers
chat_logger = LogWriter("chat_log", "test_log")
LLM_debug_logger = LogWriter("LLM_debug", "test_log")

# History message trimmer
HistoryMessages = [] # history holder, temp class
trimmer = trim_messages(
    max_tokens=2048,
    strategy="last",
    token_counter=LLM_model,
    include_system=False,
    allow_partial=False,
    start_on="human",
)

# ===============================
# Vector Database Store and Retrieve
def vec_db_summary(LLM_model: BaseChatModel, summary_template:ChatPromptTemplate, user_msg: HumanMessage, AI_response: AIMessage) -> AIMessage:
    ''' use LLM to make summary '''
    summary_msg = LLM_model.invoke(summary_template.format_messages(
        **{
            "user_input_message"  : user_msg,
            "AI_response_message" : AI_response
            }
        ))
    return summary_msg[0]

def vec_db_store(vector_database:FAISS, message:BaseMessage) -> None:
    ''' put message context into vector database '''
    vector_database.add_texts(texts=[f"{message.content}"])

def vec_db_retrieve(vector_database:FAISS, search_query:str, search_amount:int=4, score_threshold:int=1.2):
    ''' retrieve message from vector database '''
    retrieve_messages = []
    if vector_database.index.ntotal > 1: # check if database contians data
        retrieve_data = vector_database.similarity_search_with_score(query=search_query, k=search_amount, kwargs={"score_threshold": score_threshold})
        # Transform into AI Message Class
        for text, score in retrieve_data:
            text_id = text.id
            test_score = score
            retrieve_messages.append(AIMessage(content=text.page_content))
    return retrieve_messages

# ===============================
# create LLM callbacks
class LLM_input_monitor(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
    
    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id = None, tags = None, metadata = None, **kwargs):
        if prompts:
            LLM_debug_logger.write_log(prompts, "LLM input", add_time=True)
        return super().on_llm_start(serialized, prompts, run_id=run_id, parent_run_id=parent_run_id, tags=tags, metadata=metadata, **kwargs)
    
monitor = LLM_input_monitor()
LLM_model.callbacks = [monitor]

# ===============================
# LangGraph Setting

class StartState(TypedDict):
    raw_user_input: str
    system_setting: SystemTemplateInputVar

class EndState(TypedDict):
    output_msg : List[BaseMessage]

class DefaultState(TypedDict):
    system_msg      : List[BaseMessage] # system setting
    user_msg        : List[BaseMessage] # formatted user message
    extra_info_msg  : Annotated[list, add_messages] # stacked extra infro from all sources
    node_output_msg : List[BaseMessage] # output of current node

def start_node(state: StartState) -> DefaultState:
    ''' take raw input and make a valid message for system '''
    user_messsage  = get_user_message(UserTemplateInputVar(raw_user_input = state["raw_user_input"]))
    system_message = get_system_message(state["system_setting"])
    node_output = DefaultState({
        "system_msg"      : system_message,
        "user_msg"        : user_messsage,
        "extra_info_msg"  : [],
        "node_output_msg" : user_messsage,
    })

    # retrieve data from Database
    retrieve_messages = vec_db_retrieve(VecDB, search_query=state["raw_user_input"])

    # log
    if len(retrieve_messages) != 0:
        LLM_debug_logger.write_log("", "Retrieve Messages")
        for m in retrieve_messages:
            LLM_debug_logger.write_log(f"{m.type} : {str(m)}")
    return node_output

def LLM_reply_node(state: DefaultState) -> DefaultState:
    ''' Reply to user's questinos '''
    # get history message
    trimmed_msg = trimmer.invoke(HistoryMessages)
    # Log history input
    LLM_debug_logger.write_log("", "Chat History", add_time=False)
    for m in trimmed_msg:
        LLM_debug_logger.write_log(f"{m.type} : {str(m)}") # list all history
    # make input
    LLM_input = make_general_input(GeneralChatTemplateInputMessages(
        system_message  = state["system_msg"],
        history_message = trimmed_msg,
        user_message    = state["user_msg"]
    ))

    # invoke LLM
    LLM_reply = LLM_model.invoke(input=LLM_input)
    LLM_reply = to_list_message(LLM_reply)

    # make output
    node_output = DefaultState({
        "system_msg"      : state["system_msg"],
        "user_msg"        : state["user_msg"],
        "extra_info_msg"  : state["extra_info_msg"],
        "node_output_msg" : LLM_reply,
    })
    # add to history
    HistoryMessages.extend(state["user_msg"])
    HistoryMessages.extend(LLM_reply)
    return node_output

def end_node(state: DefaultState) -> EndState:
    ''' end node, truns inner State to output format '''
    assert len(state["node_output_msg"]) > 0
    # store into Database
    vec_db_store(VecDB, to_message(state["node_output_msg"]))
    #
    node_output = EndState({
        "output_msg" : to_message(state["node_output_msg"])
    })
    LLM_debug_logger.write_s_line(1)
    return node_output

# set state
BuildGraph = StateGraph(state_schema=DefaultState, input_schema=StartState, output_schema=EndState)
GraphMemory = MemorySaver()
chat_bot_config = {"configurable": {"thread_id": "chat_history"}}

# link nodes
BuildGraph.set_entry_point("start_node")
BuildGraph.set_finish_point("end_node")
BuildGraph.add_node("start_node", start_node)
BuildGraph.add_node("end_node", end_node)
BuildGraph.add_node("reply_node", LLM_reply_node)
BuildGraph.add_edge("start_node", "reply_node")
BuildGraph.add_edge("reply_node", "end_node")
ChatBot = BuildGraph.compile(checkpointer=GraphMemory)

# ===============================
# Start the Chat Bot

# Hello Message
initial_input = "使用者剛開啟系統，自我介紹一下"
response = ChatBot.invoke(StartState(raw_user_input=initial_input, system_setting=system_setting), chat_bot_config)

CLI_print("Chat Bot", response["output_msg"].content, "Initialize AI Chat Bot")

# main chat loop
while True:

    # input
    raw_user_input = CLI_input()

    # close system
    if raw_user_input.lower() == "exit":
        break
    
    # invoke LLM
    response = ChatBot.invoke(StartState(raw_user_input=raw_user_input, system_setting=system_setting), chat_bot_config)
    
    # reply
    answer = response["output_msg"].content
    CLI_print("AI Chat Bot", answer)

    # Log chat
    chat_logger.write_s_line(2)
    chat_logger.write_log(raw_user_input, "User Input")
    chat_logger.write_log(answer, "AI Response")

CLI_print("System", "Good Bye", "Close System")