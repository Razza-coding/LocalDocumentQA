from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_core.prompts import  BaseChatPromptTemplate, MessagesPlaceholder, ChatPromptTemplate, ChatMessagePromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from langchain.output_parsers.json import SimpleJsonOutputParser
from pydantic import BaseModel
import os, sys, re, ast, json
from math import floor, ceil
from typing import *
from CLI_Format import *
from LogWriter import LogWriter
from PromptTools import to_list_message, to_message, to_text
import config

# -------------------------------
# State Define
    
class TranslateInput(TypedDict):
    input_text: str     
    trans_lang: str     
    refine_trans: bool  
    max_refine_trys: int

class TranslateOutput(TypedDict):
    trans_text: str
    pass_verify: bool

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
    refine_loop_count: int
    correct_translation: bool # is the translation verified as correct

# -------------------------------
# Prompt Template

translate_template = ChatPromptTemplate.from_messages(
    ('ai',
'''
You are a Translate Expert. Translate Input Text into {translate_lang} and output translate result in JSON Format with 1 group. Do not add any explanation or extra text.

Format Example:
{{
    "orignal_text" : <put input text here>,
    "translate_result" : <put translate result here>
}}

Input Text:
{input_text}'''))

translate_verify_template = ChatPromptTemplate.from_messages(
    ('ai',
'''
You are a translation teacher, you will be given TRANSLATE_TEXT and ORIGINAL_TEXT and respond ANSWER to verify if the translation is correct or not.
ANSWER should be written in {{ "verify_result" : <YES/NO> }} format, write YES if translation is right, write NO if translation is wrong.
TRANSLATE_TEXT is a {trans_lang} translation from ORIGINAL_TEXT but not verified yet, you should decide if translation is right or wrong by responding ANSWER only, do not reply anything else.

TRANSLATE_TEXT: "{orignal_text}"
ORIGINAL_TEXT: "{translate_text}"
'''
))

# -------------------------------
# SubGraph Builder

def build_translate_subgraph(llm: BaseChatModel, logger: Optional[LogWriter]=None):
    ''' Build SubGraph for translation '''
    # -------------------------------
    # Create output structured llm
    class TranslatePair(BaseModel):
        orignal_text:str
        translate_result:str
    
    class VerifyResult(BaseModel):
        verify_result:str

    translate_llm = llm.with_structured_output(TranslatePair, strict=True).bind(options={"temperature": 0.3}) # keep alittle temperature to for refining loop
    verify_llm = llm.with_structured_output(VerifyResult, strict=True).bind(options={"temperature": 0.0})

    # -------------------------------
    # Sub Graph Nodes
    def translate_start_node(state: TranslateInput) -> TranslateState:
        ''' use to apply default note '''
        node_output = TranslateState(
            input_text=state.get("input_text"),
            trans_lang=state.get("trans_lang"),
            trans_text="",
            refine_trans=state.get("refine_trans", True),
            max_refine_trys=state.get("max_refine_trys", 2),
            refine_loop_count=state.get("refine_loop_count", 0),
            correct_translation=state.get("correct_translation", False),
        )
        return node_output

    def translate_node(state: TranslateState) -> TranslateState:
        ''' Use LLM to tranlation, format string to JSON and then extract translate result '''
        # Use enhance prompt to translate, required LLM to output oringal and translate text in JSON format
        translate_chain = translate_template | translate_llm
        translate_result = translate_chain.invoke({"translate_lang" : state["trans_lang"], "input_text" : state["input_text"]})
        translate_result = translate_result.model_dump()
        translate_result = translate_result.get("translate_result", None)
        state["trans_text"] = translate_result if translate_result else ""
        return state

    def verify_node(state: TranslateState) -> TranslateState:
        ''' check translate result and update flags '''
        # check for refine mode on or off
        if not state["refine_trans"]:
            state["correct_translation"] = True
            return state # skips refining, pass verify
        
        # check for maximum refine trys
        if state["refine_loop_count"] >= state["max_refine_trys"]:
            state["correct_translation"] = True
            return state # max try reached, pass verify
        else:
            state["refine_loop_count"] += 1
        
        # check for empty output
        if not state["trans_text"]:
            state["correct_translation"] = False
            return state # translation empty, fail verify
        
        # use LLM to check translate result
        verify_chain = translate_verify_template | verify_llm
        verify_result = verify_chain.invoke({"trans_lang" : state["trans_lang"], "orignal_text" : state["input_text"], "translate_text" :state["trans_text"]})
        verify_result = verify_result.model_dump()
        verify_result = str(verify_result.get("verify_result", "")).lower()
        #
        state["correct_translation"] = True if "yes" in verify_result else False
        return state

    def translate_refine_router(state: TranslateState) -> str:
        ''' choose next node base on verify result '''
        if state["correct_translation"]:
            return "pass"
        else:
            CLI_print("System", "Translation Failed, retry")
            return "retry"

    def translate_output_node(state: TranslateState) -> TranslateOutput:
        ''' extract only translate part as output '''
        node_output = TranslateOutput(
            trans_text=state["trans_text"],
            pass_verify=state["correct_translation"]
        )
        return node_output

    # set state
    BuildTranslateGraph = StateGraph(state_schema=TranslateState, input_schema=TranslateInput, output_schema=TranslateOutput)

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
    TranslateGraph = BuildTranslateGraph.compile()
    return TranslateGraph

# ===============================
# Test translate graph

if __name__ == "__main__":
    LLM_model, VecDB = config.init_system()
    test_string = '''
    there are approximately between 470,000 and 690,000 African elephants in the wild. 
    Although this estimate only covers about half of the total elephant range, experts do not believe the true figure to be much higher, as it is unlikely that large populations remain to be discovered.
    By far the largest populations are now found in Southern and Eastern Africa, which together account for the majority of the continental population. 
    According to a recent analysis by IUCN experts, most major populations in Eastern and Southern Africa are stable or have been steadily increasing since the mid-1990s, at an average rate of 4.5% per annum.
    Blanc et al. 2007, op. cit.
    '''

    TranslaterBot = build_translate_subgraph(LLM_model)
    
    trans_lang="繁體中文"
    response = TranslaterBot.invoke(TranslateInput(input_text=test_string, trans_lang="繁體中文", refine_trans=True, max_refine_trys=3))
    CLI_print("Translate Input", test_string, speaker_sidenote="繁體中文")
    CLI_print("Translate Result", response["trans_text"], speaker_sidenote="繁體中文")
    CLI_next()

    trans_lang="日文"
    response = TranslaterBot.invoke(TranslateInput(input_text=test_string, trans_lang="日文", refine_trans=True, max_refine_trys=3))
    CLI_print("Translate Input", test_string, speaker_sidenote="日文")
    CLI_print("Translate Result", response["trans_text"], speaker_sidenote="日文")
    CLI_next()