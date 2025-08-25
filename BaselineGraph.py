from langchain_core.messages import trim_messages, HumanMessage, AIMessage, SystemMessage, BaseMessage, AnyMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, MessagesPlaceholder, BasePromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver

from config import init_LLM, get_llm_info
import os, sys, re, ast, json
import uuid
from math import floor, ceil
from typing import *
from CLI_Format import *
from LogWriter import LogWriter

from DatabaseManager import VDBManager

from PromptTools import UserTemplateInputVar, get_user_message
from PromptTools import SystemTemplateInputVar, get_system_message
from PromptTools import GeneralChatTemplateInputMessages, make_general_input
from PromptTools import to_list_message, to_message, to_text
from MainGraph import StartState, EndState


# -------------------------------
# State Define
class DefaultState(TypedDict):
    raw_user_input  : str # raw input string
    system_msg      : List[BaseMessage] # system setting
    user_msg        : List[BaseMessage] # formatted user message
    output_msg      : List[BaseMessage]

def build_baseline_graph(llm:BaseChatModel, graph_name:str="BaselineGraph") -> CompiledStateGraph:
    ''' Create a very simple graph to get llm reply for evaluation baseline '''

    # ===============================
    # Graph Setting
    def start_node(state: StartState) -> DefaultState:
        ''' take raw input and make a valid message for system '''
        user_messsage  = get_user_message(UserTemplateInputVar(raw_user_input = state["raw_user_input"]))
        system_message = get_system_message(state["system_setting"])
        return {
            "raw_user_input"  : state["raw_user_input"],
            "system_msg"      : system_message,
            "user_msg"        : user_messsage,
            "output_msg"      : None
        }

    def LLM_reply_node(state: DefaultState) -> DefaultState:
        # make input
        LLM_input = make_general_input(GeneralChatTemplateInputMessages(
            system_message    = state["system_msg"],
            user_message      = state["user_msg"]
        ))

        # invoke LLM
        LLM_reply = llm.invoke(input=LLM_input)
        LLM_reply = to_list_message(LLM_reply)
    
        return {
            "output_msg" : LLM_reply
        }

    def end_node(state: DefaultState) -> EndState:
        ''' end node, truns inner State to output format '''
        return {
            "output_msg" : state.get("output_msg", [AIMessage("EMPTY_OUTPUT")])
        }


    # set state
    BuildGraph = StateGraph(state_schema=DefaultState, input_schema=StartState, output_schema=EndState)

    # link nodes
    # core nodes : start, reply, end
    BuildGraph.set_entry_point("start_node")
    BuildGraph.set_finish_point("end_node")
    BuildGraph.add_node("start_node", start_node)
    BuildGraph.add_node("end_node", end_node)
    BuildGraph.add_node("reply_node", LLM_reply_node)
    # edges
    BuildGraph.add_edge("start_node", "reply_node")
    BuildGraph.add_edge("reply_node", "end_node")
    BaselineGraph = BuildGraph.compile()
    BaselineGraph.name = str(graph_name)
    return BaselineGraph


if __name__ == "__main__":
    # ===============================
    # initial
    LLM_model = init_LLM(LLM_temperature=0)

    # ===============================
    # Build graph
    BaselineGraph = build_baseline_graph(LLM_model)

    # ===============================
    # set graph system setting
    AI_name = "JOHN"
    professional_role = "專業AI助理"
    # A QA answering setting
    system_setting = SystemTemplateInputVar(
            AI_name = AI_name,
            professional_role = professional_role,
            chat_lang = "English",
            negitive_rule = "Reply with other langue",
            output_format = "Answer QUESTION with a short answer. As simple as possible."
        )

    # ===============================
    # Start Graph

    # Hello Message
    initial_input = "現在使用者剛開啟系統，請 AI 聊天機器人對使用者自我介紹一下"
    response = BaselineGraph.invoke(StartState(raw_user_input=initial_input, system_setting=system_setting))
    CLI_print("Chat Bot", to_text(response.get("output_msg", "INITIALIZE FAILED")), "Initialize AI Chat Bot")

    # Chat loop
    while True:
        raw_user_input = CLI_input()
        if raw_user_input.lower() == "exit":
            break
        # invoke 
        response = BaselineGraph.invoke(StartState(raw_user_input=raw_user_input, system_setting=system_setting))
        # reply
        answer = to_text(response.get("output_msg", "EMPTY RESPONSE"))
        CLI_print("AI Chat Bot", answer)
    CLI_print("System", "Good Bye", "Close System")
