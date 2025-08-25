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

from pydantic import BaseModel, Field
from config import init_LLM, get_llm_info
import os, sys, re, ast, json
import uuid
from math import floor, ceil
from typing import *
from CLI_Format import *
from LogWriter import LogWriter
from PromptTools import to_list_message, to_message, to_text


# -------------------------------
# Prompt Template
class SystemPromptConfig(BaseModel):
    chat_lang:str     = Field(default="English")
    negitive_rule:str = Field(default="Reply with other langue")
    output_format:str = Field(default="Answer QUESTION with a short answer, as simple as possible")

system_prompt_template = SystemMessagePromptTemplate.from_template(
    (
        "Use {chat_lang} to communicate with user while avoiding violate {negitive_rule}.\n"
        "You will be given OUTPUT FORMAT, which discribes a style or format for your reply"
        "OUTPUT FORMAT:{output_format}\n"
        "\n"
        "You should reply human user according to OUTPUT FORMAT.\n"
        "\n\n"
    )
)

user_prompt_template = HumanMessagePromptTemplate.from_template(
    "{raw_user_input}\n"
    )

input_template = ChatPromptTemplate.from_messages([
    MessagesPlaceholder("system_message"),
    MessagesPlaceholder("user_message"),
])

# -------------------------------
# State Define
class StartState(TypedDict):
    input: str
    prompt_config: SystemPromptConfig

class EndState(TypedDict):
    output_msg : List[BaseMessage]

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
        state["prompt_config"] = state["prompt_config"].model_dump() if isinstance(state["prompt_config"], BaseModel) else state["prompt_config"]
        user_message   = user_prompt_template.format(raw_user_input=state["input"])
        system_message = system_prompt_template.format(**state["prompt_config"])      
        return {
            "raw_user_input"  : state["input"],
            "system_msg"      : system_message,
            "user_msg"        : user_message,
            "output_msg"      : None
        }

    def LLM_reply_node(state: DefaultState) -> DefaultState:
        # make input
        LLM_input = input_template.invoke({
            "system_message" : to_list_message(state["system_msg"]),
            "user_message"   : to_list_message(state["user_msg"]),
            })

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
    system_setting = SystemPromptConfig(
            chat_lang = "English",
            negitive_rule = "Reply with other langue",
            output_format = "Respond QUESTION with a short answer, as simple as possible"
        )

    # ===============================
    # Start Graph

    # Hello Message
    initial_input = "現在使用者剛開啟系統，請 AI 聊天機器人對使用者自我介紹一下"
    response = BaselineGraph.invoke(StartState(input=initial_input, prompt_config=system_setting))
    CLI_print("Chat Bot", to_text(response.get("output_msg", "INITIALIZE FAILED")), "Initialize AI Chat Bot")

    # Chat loop
    while True:
        raw_user_input = CLI_input()
        if raw_user_input.lower() == "exit":
            break
        # invoke 
        response = BaselineGraph.invoke(StartState(input=raw_user_input, prompt_config=system_setting))
        # reply
        answer = to_text(response.get("output_msg", "EMPTY RESPONSE"))
        CLI_print("AI Chat Bot", answer)
    CLI_print("System", "Good Bye", "Close System")
