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

from config import init_system, get_llm_info
import os, sys, re, ast, json
import uuid
from math import floor, ceil
from typing import *
from CLI_Format import *
from LogWriter import LogWriter

from DatabaseManager import VDBManager
from TranslationSubGraph import build_translate_subgraph, TranslateInput

from PromptTools import UserTemplateInputVar, get_user_message
from PromptTools import SystemTemplateInputVar, get_system_message
from PromptTools import ChatHistoryTemplateInputMessages, make_chat_history_prompt
from PromptTools import GeneralChatTemplateInputMessages, make_general_input
from PromptTools import RAGSummaryChatTemplateInputMessages, make_RAG_summary_prompt
from PromptTools import KnowledgeTemplateInputMessages, get_knowledge_message
from PromptTools import to_list_message, to_message, to_text


# -------------------------------
# State Define
class StartState(TypedDict):
    raw_user_input: str
    system_setting: SystemTemplateInputVar

class EndState(TypedDict):
    output_msg : List[BaseMessage]

class DefaultState(TypedDict):
    raw_user_input  : str # raw input string
    system_msg      : List[BaseMessage] # system setting
    user_msg        : List[BaseMessage] # formatted user message
    extra_info_msg  : List[BaseMessage] # stacked extra info from all retriever
    history_messages: Annotated[list[AnyMessage], add_messages] # history saver

def build_main_graph(llm:BaseChatModel, database_manager:VDBManager, debug_logger: Optional[LogWriter]=None, graph_name:str="MainGraph") -> CompiledStateGraph:
    ''' Create a LLM RAG graph (main graph) '''
    
    # ===============================
    # Setting
    llm_info = get_llm_info(llm)
    MAX_CONTEXT_WINDOW   = round(int(llm_info.get("Model").get("context length")) * 0.95) # maximum token of gemma3:4b is 128000
    MAX_INPUT_CONTEXT    = floor( MAX_CONTEXT_WINDOW * 0.7 )
    MAX_OUTPUT_CONTEXT   = floor( MAX_CONTEXT_WINDOW * 0.3 )
    MAX_HISTORY_CONTEXT  = floor( MAX_INPUT_CONTEXT * 0.3 )
    MAX_RETRIEVE_CONTEXT = floor( MAX_INPUT_CONTEXT * 0.3 )
    
    # trimmer of extra knowledge
    knowledge_trimmer = trim_messages(
        max_tokens=MAX_RETRIEVE_CONTEXT,
        strategy="last",
        token_counter=llm,
        include_system=False,
        allow_partial=False,
        start_on="ai",
    )
    # trimmer of history
    history_trimmer = trim_messages(
        max_tokens=MAX_HISTORY_CONTEXT,
        strategy="last",
        token_counter=llm,
        include_system=False,
        allow_partial=False,
        start_on="human",
    )
    # check logger
    if debug_logger:
        assert isinstance(debug_logger, LogWriter), "Not a Valid LogWriter"

    # ===============================
    # LangGraph Setting

    # -------------------------------
    # Sub Graph Nodes
    TranslateSubGraph = build_translate_subgraph(llm=llm, logger=debug_logger)

    # -------------------------------
    # Main Graph Nodes
    def start_node(state: StartState) -> DefaultState:
        ''' take raw input and make a valid message for system '''
        user_messsage  = get_user_message(UserTemplateInputVar(raw_user_input = state["raw_user_input"]))
        system_message = get_system_message(state["system_setting"])
        return {
            "raw_user_input"  : state["raw_user_input"],
            "system_msg"      : system_message,
            "user_msg"        : user_messsage,
            "extra_info_msg"  : [],
        }

    def info_retrieve_node(state: DefaultState) -> DefaultState:
        ''' Retrieve infomation in Vector Database '''
        # retrieve data from Database
        search_q = TranslateSubGraph.invoke(TranslateInput(input_text=state["raw_user_input"], trans_lang="English", refine_trans=True, max_refine_trys=3)).get("trans_text")
        retrieve_documents = database_manager.retrieve(query=search_q, k=8, score_threshold=1.0)

        if debug_logger: 
            debug_logger.write_log(search_q, "Search Query")
    
        # log RAG result
        retrieved_msg_log = ""
        for doc in retrieve_documents:
            retrieved_msg_log += f"ID : {str(doc.id) :<30} Score : {str( doc.metadata.get('retrieve score') ) :<5}\n{str(doc.page_content) :<}\n\n"
        
        if debug_logger:
            debug_logger.write_log(retrieved_msg_log, "Retrieve Messages")

        retrieve_msg = [AIMessage(content=doc.page_content) for doc in retrieve_documents] # turns Document into AIMessage
        return {
            "extra_info_msg" : retrieve_msg
        }

    def LLM_reply_node(state: DefaultState) -> DefaultState:
        ''' Reply to user's questinos '''
        # get history message
        trimmed_msg = history_trimmer.invoke(state["history_messages"])
        history_msg = make_chat_history_prompt(ChatHistoryTemplateInputMessages(chat_history=trimmed_msg))

        # get knowledge message
        trimmed_knowledge_message = knowledge_trimmer.invoke(state["extra_info_msg"])
        knowledge_msg = get_knowledge_message(KnowledgeTemplateInputMessages(knowlegde_messages=trimmed_knowledge_message))

        # Log history
        history_msg_log = [f"{m.type :<5} : {str(m) :<}" for m in trimmed_msg]

        if debug_logger:
            debug_logger.write_log('\n'.join(history_msg_log), "Chat History", add_time=False)
        
        # make input
        LLM_input = make_general_input(GeneralChatTemplateInputMessages(
            system_message    = state["system_msg"],
            history_message   = history_msg,
            knowledge_message = knowledge_msg,
            user_message      = state["user_msg"]
        ))

        # invoke LLM
        LLM_reply = llm.invoke(input=LLM_input)
        LLM_reply = to_list_message(LLM_reply)
    
        return {
            "history_messages" : state["user_msg"] + LLM_reply,
        }

    def end_node(state: DefaultState) -> EndState:
        ''' end node, truns inner State to output format '''
        LLM_reply = state["history_messages"][-1]

        if debug_logger:
            debug_logger.write_s_line(1)

        return {
            "output_msg" : to_list_message(LLM_reply)
        }


    # set state
    BuildGraph = StateGraph(state_schema=DefaultState, input_schema=StartState, output_schema=EndState)
    GraphMemory = MemorySaver()
    thread_id = f"main_graph_{str(uuid.uuid4())}"
    main_graph_config = {"configurable": {"thread_id": thread_id}}
    CLI_print(message=f"Thread ID: {thread_id}", speaker_sidenote="Create Main Graph")

    # link nodes
    # core nodes : start, reply, end
    BuildGraph.set_entry_point("start_node")
    BuildGraph.set_finish_point("end_node")
    BuildGraph.add_node("start_node", start_node)
    BuildGraph.add_node("end_node", end_node)
    BuildGraph.add_node("reply_node", LLM_reply_node)
    # function nodes
    BuildGraph.add_node("info_retrieve_node", info_retrieve_node)
    # edges
    BuildGraph.add_edge("start_node", "info_retrieve_node")
    BuildGraph.add_edge("info_retrieve_node", "reply_node")
    BuildGraph.add_edge("reply_node", "end_node")
    MainGraph = BuildGraph.compile(checkpointer=GraphMemory).with_config(config=main_graph_config)
    MainGraph.name = str(graph_name)
    return MainGraph


if __name__ == "__main__":
    # ===============================
    # initial
    LLM_model, VecDB = init_system()
    database_manager = VDBManager(VecDB)

    # ===============================
    # Log writers
    chat_logger = LogWriter("chat_log", "test_log")
    LLM_debug_logger = LogWriter("LLM_debug", "test_log")
    date_time = chat_logger.log_create_time

    # ===============================
    # Vector Database Store and Retrieve
    dataset_name = "mini-wiki"
    raw_data_file = "./BanchMark/Dataset/logs/20250724_rag-datasets_rag-mini-wikipedia-text-corpus.txt"

    if not database_manager.load("prebuild_VDB", dataset_name):
        database_manager.load_from_file(raw_data_file)
        database_manager.save("prebuild_VDB", dataset_name)

    CLI_print("Vector Database", f"Data amount: {database_manager.amount()}")

    # ===============================
    # create LLM callbacks
    class llm_monitor_main_graph(BaseCallbackHandler):
        def __init__(self):
            self.monitor_name = "main_graph_monitor"
            super().__init__()
        
        def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id = None, tags = None, metadata = None, **kwargs):
            if prompts:
                LLM_debug_logger.write_log('\n'.join(prompts), "LLM input", add_time=True)
            return super().on_llm_start(serialized, prompts, run_id=run_id, parent_run_id=parent_run_id, tags=tags, metadata=metadata, **kwargs)
        
        def on_llm_end(self, response, *, run_id, parent_run_id = None, **kwargs):
            return super().on_llm_end(response, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

    monitor = llm_monitor_main_graph()
    LLM_model.callbacks = LLM_model.callbacks.append(monitor) if LLM_model.callbacks and isinstance(LLM_model.callbacks, List) else [monitor]
    
    # ===============================
    # Start the Chat Bot

    # build main graph
    MainGraph = build_main_graph(LLM_model, database_manager, LLM_debug_logger)

    # set graph system setting
    AI_name = "JOHN"
    professional_role = "專業AI助理"
    system_setting = SystemTemplateInputVar(
            AI_name = AI_name,
            professional_role = professional_role,
        )

    # Hello Message
    initial_input = "現在使用者剛開啟系統，請 AI 聊天機器人對使用者自我介紹一下"
    response = MainGraph.invoke(StartState(raw_user_input=initial_input, system_setting=system_setting))

    CLI_print("Chat Bot", to_text(response.get("output_msg", "INITIALIZE FAILED")), "Initialize AI Chat Bot")

    # main chat loop
    while True:

        # input
        raw_user_input = CLI_input()

        # close system
        if raw_user_input.lower() == "exit":
            break
        
        # invoke LLM
        response = MainGraph.invoke(StartState(raw_user_input=raw_user_input, system_setting=system_setting))
        
        # reply
        answer = to_text(response.get("output_msg", "EMPTY RESPONSE"))
        CLI_print("AI Chat Bot", answer)

        # Log chat
        chat_logger.write_s_line(2)
        chat_logger.write_log(raw_user_input, "User Input")
        chat_logger.write_log(answer, "AI Response")

    CLI_print("System", "Good Bye", "Close System")
