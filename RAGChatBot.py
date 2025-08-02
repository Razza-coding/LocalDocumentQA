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
import os, sys, re, ast, json
from math import floor, ceil
from typing import *
from CLI_Format import *
from LogWriter import LogWriter

from PromptTools import UserTemplateInputVar, get_user_message
from PromptTools import SystemTemplateInputVar, get_system_message
from PromptTools import TranslateTemplateInputVar, get_translate_request_message, TranslateVerifyTemplateInputVar, get_translate_verify_message
from PromptTools import ChatHistoryTemplateInputMessages, make_chat_history_prompt
from PromptTools import GeneralChatTemplateInputMessages, make_general_input
from PromptTools import RAGSummaryChatTemplateInputMessages, make_RAG_summary_prompt
from PromptTools import KnowledgeTemplateInputMessages, get_knowledge_message
from PromptTools import to_list_message, to_message, to_text

# ===============================
# initial
LLM_model, VecDB = config.init_system()

# ===============================
# Setting
MAX_CONTEXT_WINDOW   = 128000 # maximum token of gemma3:4b
MAX_HISTORY_CONTEXT  = floor( MAX_CONTEXT_WINDOW * 0.3 )
MAX_RETRIEVE_CONTEXT = floor( MAX_CONTEXT_WINDOW * 0.3 )

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
HistoryMessages = [] # chat history holder

# trimmer of extra knowledge
knowledge_trimmer = trim_messages(
    max_tokens=MAX_RETRIEVE_CONTEXT,
    strategy="last",
    token_counter=LLM_model,
    include_system=False,
    allow_partial=False,
    start_on="ai",
)
# trimmer of history
history_trimmer = trim_messages(
    max_tokens=MAX_HISTORY_CONTEXT,
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
    retrieve_scores   = []
    if vector_database.index.ntotal > 1: # check if database contians data
        retrieve_data = vector_database.similarity_search_with_score(query=search_query, k=search_amount)
        # Transform into AI Message Class
        for text, score in retrieve_data:
            if score > score_threshold: # score filter
                continue
            text_id = text.id
            test_score = score
            retrieve_messages.append(AIMessage(content=text.page_content))
            retrieve_scores.append(score)
    return retrieve_messages, retrieve_scores

def vec_db_build_from_file(vector_database:FAISS, file:str):
    ''' fills database with data from txt file '''
    with open(file, mode="r", encoding='utf-8') as df:
        while True:
            data_item = df.readline()
            if not data_item:
                break
            format_data = ast.literal_eval(data_item)["passage"]
            print(format_data)
            vector_database.add_texts(texts=[format_data])

dataset_name = "mini-wiki"
if os.path.exists(os.path.abspath(f"prebuild_VDB/{dataset_name}.pkl")):
    VecDB = VecDB.load_local("prebuild_VDB", VecDB.embeddings, index_name=dataset_name, allow_dangerous_deserialization=True)
else:
    raw_data_file = "./BanchMark/Dataset/logs/20250724_rag-datasets_rag-mini-wikipedia-text-corpus.txt"
    vec_db_build_from_file(VecDB, raw_data_file)
    VecDB.save_local("prebuild_VDB", dataset_name)
CLI_print("Vec DB", f"Data amount: {VecDB.index.ntotal}")

# ===============================
# create LLM callbacks
class LLM_input_monitor(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
    
    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id = None, tags = None, metadata = None, **kwargs):
        if prompts:
            LLM_debug_logger.write_log('\n'.join(prompts), "LLM input", add_time=True)
        return super().on_llm_start(serialized, prompts, run_id=run_id, parent_run_id=parent_run_id, tags=tags, metadata=metadata, **kwargs)
    
    def on_llm_end(self, response, *, run_id, parent_run_id = None, **kwargs):
        return super().on_llm_end(response, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
    
    
monitor = LLM_input_monitor()
LLM_model.callbacks = [monitor]

# ===============================
# LangGraph Setting

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
    extra_info_msg  : Annotated[list, add_messages] # stacked extra info from all sources
    node_output_msg : List[BaseMessage] # output of current node

class TranslateInitState(TypedDict):
    input_text: str 
    trans_lang: str 

class TranslateState(TypedDict):
    # input
    input_text: str # input message to translate
    trans_lang: str # translate to target langue
    # output
    trans_text: str # output of translate result
    # settings
    refine_trans: bool # enable LLM double checking translate result, if translate is incorrect, then ask for a better translation
    max_refine_trys: int # maximum loop for refining trnaslate
    # inner flags
    _refine_loop_count: int
    _correct_translation: bool # is the translation verified as correct

# -------------------------------
# Sub Graph Nodes
def translate_start_node(state: TranslateInitState | TranslateState) -> TranslateState:
    ''' use to apply default note '''
    node_output = TranslateState(
        input_text=state.get("input_text"),
        trans_lang=state.get("trans_lang"),
        trans_text="",
        refine_trans=state.get("refine_trans", True),
        max_refine_trys=state.get("max_refine_trys", 2),
        _refine_loop_count=state.get("_refine_loop_count", 0),
        _correct_translation=state.get("_correct_translation", False),
    )
    return node_output

def translate_node(state: TranslateState) -> TranslateState:
    ''' Use LLM to tranlation, format string to JSON and then extract translate result '''
    # Use enhance prompt to translate, required LLM to output oringal and translate text in JSON format
    trans_prompt = get_translate_request_message(TranslateTemplateInputVar(translate_lang=state["trans_lang"], input_text=state["input_text"]))
    LLM_translattion = to_text(LLM_model.invoke(trans_prompt))
    # extract into json
    json_blocks = re.findall(r'\{[^{}]*"translate_result"[^{}]*\}', LLM_translattion, re.DOTALL)
    #json_section = re.search(r"\{.*\}", LLM_translattion, re.DOTALL)
    if json_blocks:
        # extract result into json format, and get only translate part
        state["trans_text"] = ""
        for block in json_blocks:
            try:
                parsed = json.loads(block)
                if "translate_result" in parsed:
                    state["trans_text"] = str(parsed["translate_result"])
                    break
            except json.JSONDecodeError:
                continue 
    else:
        state["trans_text"] = ""
        # raise ValueError
    return state

def verify_node(state: TranslateState) -> TranslateState:
    ''' check translate result and update flags '''
    # check for refine mode on or off
    if not state["refine_trans"]:
        state["_correct_translation"] = True
        return state # skips refining, pass verify
    
    # check for maximum refine trys
    if state["_refine_loop_count"] > state["max_refine_trys"]:
        state["_correct_translation"] = True
        return state # max try reached, pass verify
    else:
        state["_refine_loop_count"] += 1
    
    # check for empty output
    if not state["trans_text"]:
        state["_correct_translation"] = False
        return state # translation empty, fail verify
    
    # use LLM to check translate result
    verify_prompt = get_translate_verify_message(TranslateVerifyTemplateInputVar(trans_lang=state["trans_lang"], orignal_text=state["input_text"], translate_text=state["trans_text"]))
    LLM_verify = to_text(LLM_model.invoke(verify_prompt))
    yes_or_no = re.search(r"\[.*?\]", LLM_verify, re.DOTALL).group(0)
    if "yes" in yes_or_no.lower():
        state["_correct_translation"] = True
    else:
        state["_correct_translation"] = False

    return state

def translate_refine_router(state: TranslateState) -> str:
    ''' choose next node base on verify result '''
    if state["_correct_translation"]:
        return "pass"
    else:
        CLI_print("System", "Translation Failed, retry")
        return "retry"

def translate_output_node(state: TranslateState) -> TranslateState:
    ''' extract only translate part as output '''
    return state

# set state
BuildTranslateGraph = StateGraph(state_schema=TranslateState)

# link nodes
BuildTranslateGraph.set_entry_point("translate_start_node")
BuildTranslateGraph.set_finish_point("translate_output_node")
BuildTranslateGraph.add_node("translate_start_node", translate_start_node)
BuildTranslateGraph.add_node("translate_node", translate_node)
BuildTranslateGraph.add_node("verify_node", verify_node)
BuildTranslateGraph.add_node("translate_output_node", translate_output_node)
BuildTranslateGraph.add_edge("translate_start_node", "translate_node")
BuildTranslateGraph.add_edge("translate_node", "verify_node")
BuildTranslateGraph.add_conditional_edges("verify_node", translate_refine_router, {"pass" : "translate_output_node", "retry": "translate_node"})
TranslaterBot = BuildTranslateGraph.compile()


# -------------------------------
# Main Graph Nodes
def start_node(state: StartState) -> DefaultState:
    ''' take raw input and make a valid message for system '''
    user_messsage  = get_user_message(UserTemplateInputVar(raw_user_input = state["raw_user_input"]))
    system_message = get_system_message(state["system_setting"])
    node_output = DefaultState({
        "raw_user_input"  : state["raw_user_input"],
        "system_msg"      : system_message,
        "user_msg"        : user_messsage,
        "extra_info_msg"  : [],
        "node_output_msg" : user_messsage,
    })
    return node_output

def info_retrieve_node(state: DefaultState) -> DefaultState:
    ''' Retrieve infomation in Vector Database '''
    # retrieve data from Database
    search_q = TranslaterBot.invoke(TranslateState(input_text=state["raw_user_input"],trans_lang="English",)).get("trans_text")
    retrieve_messages, retrieve_scores = vec_db_retrieve(VecDB, search_query=search_q, search_amount=8, score_threshold=1.0)
    LLM_debug_logger.write_log(search_q, "Search Query")
   
    # log RAG result
    retrieved_msg_log = ""
    for m, s in zip(retrieve_messages, retrieve_scores):
        retrieved_msg_log += f"{m.type :<5} {str(s) :<5} : {str(m) :<}\n"
    LLM_debug_logger.write_log(retrieved_msg_log, "Retrieve Messages")

    return {"extra_info_msg" : retrieve_messages}

def info_store_node(state: DefaultState) -> DefaultState:
    ''' Store message in Vector Database '''
    # store into Database
    if state["node_output_msg"]:
        vec_db_store(VecDB, to_message(state["node_output_msg"]))
        CLI_print("RAG", "", "Message Stored")

    return state

def LLM_reply_node(state: DefaultState) -> DefaultState:
    ''' Reply to user's questinos '''
    # get history message
    trimmed_msg = history_trimmer.invoke(HistoryMessages)
    history_msg = make_chat_history_prompt(ChatHistoryTemplateInputMessages(chat_history=trimmed_msg))

    # get knowledge message
    trimmed_knowledge_message = knowledge_trimmer.invoke(state["extra_info_msg"])
    knowledge_msg = get_knowledge_message(KnowledgeTemplateInputMessages(knowlegde_messages=trimmed_knowledge_message))

    # Log history
    history_msg_log = [f"{m.type :<5} : {str(m) :<}" for m in trimmed_msg]
    LLM_debug_logger.write_log('\n'.join(history_msg_log), "Chat History", add_time=False)
    
    # make input
    LLM_input = make_general_input(GeneralChatTemplateInputMessages(
        system_message    = state["system_msg"],
        history_message   = history_msg,
        knowledge_message = knowledge_msg,
        user_message      = state["user_msg"]
    ))
    #LLM_input = main_trimmer.invoke(LLM_input)

    # invoke LLM
    LLM_reply = LLM_model.invoke(input=LLM_input)
    LLM_reply = to_list_message(LLM_reply)
   
    # add to history
    HistoryMessages.extend(state["user_msg"])
    HistoryMessages.extend(LLM_reply)

    # update state for output
    state["node_output_msg"] = LLM_reply
    return state

def end_node(state: DefaultState) -> EndState:
    ''' end node, truns inner State to output format '''
    assert len(state["node_output_msg"]) > 0
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
# core nodes : start, reply, end
BuildGraph.set_entry_point("start_node")
BuildGraph.set_finish_point("end_node")
BuildGraph.add_node("start_node", start_node)
BuildGraph.add_node("end_node", end_node)
BuildGraph.add_node("reply_node", LLM_reply_node)
# function nodes
BuildGraph.add_node("info_retrieve_node", info_retrieve_node)
BuildGraph.add_node("info_store_node", info_store_node)
# edges
BuildGraph.add_edge("start_node", "info_retrieve_node")
BuildGraph.add_edge("info_retrieve_node", "reply_node")
BuildGraph.add_edge("reply_node", "end_node")
#BuildGraph.add_edge("info_store_node", "end_node")
ChatBot = BuildGraph.compile(checkpointer=GraphMemory)

# ===============================
# Start the Chat Bot

if __name__ == "__main__":
    # Hello Message
    initial_input = "現在使用者剛開啟系統，請 AI 聊天機器人對使用者自我介紹一下"
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