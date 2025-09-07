from langchain_community.vectorstores import FAISS, DistanceStrategy
from langchain_community.docstore.in_memory import InMemoryDocstore

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.language_models import BaseChatModel

import faiss

import json
import os
import re
import rich
import subprocess
from pydantic import BaseModel
from typing import *

def __sys_init_message(build_object_type: str, build_object_name):
    print(f"Build {build_object_type:<10} : {build_object_name:<20} Complete")

def init_LLM(LLM_model_name='gemma3:4b', LLM_temperature:float=0.7, LLM_url:str="http://localhost:11434"):
    ''' Initialize LLM model form ollama '''
    # llm
    LLM_model = ChatOllama(
        model=LLM_model_name,
        temperature=LLM_temperature,
        base_url=LLM_url
    )
    __sys_init_message("LLM", LLM_model_name)
    return LLM_model

def init_embedding(model_name: str="sentence-transformers/all-MiniLM-L6-v2", model_device: str="cpu", normalize_embeddings: bool=True) -> HuggingFaceEmbeddings:
    ''' Simplified Sentence Embedding Builder '''
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name, 
        model_kwargs={'device': model_device}, 
        encode_kwargs={'normalize_embeddings': normalize_embeddings}
    )
    __sys_init_message("Embedding Model", model_name)
    return embeddings

def init_VecDB():
    ''' Create FAISS Vector Database '''
    # embeddings of VecDB
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs=embedding_model_kwargs,
        encode_kwargs=encode_kwargs
    )
    __sys_init_message("VecDB Embedding", embedding_model_name)

    # index of VecDB
    dimension  = len(embeddings.embed_query("foo"))
    vdb_index = faiss.IndexFlatL2(dimension)
    __sys_init_message("VecDB Index", "Faiss FlatL2")

    # 使用 Hugging Face 包裝 VecDB
    VecDB = FAISS(
        embedding_function=embeddings,
        index=vdb_index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    __sys_init_message("VecDB", "Faiss")
    return VecDB
    
def init_system(LLM_model_name='gemma3:4b') -> Tuple[BaseChatModel, FAISS]:
    ''' Build system core objects '''
    LLM_model = init_LLM(LLM_model_name)
    VecDB = init_VecDB()
    return LLM_model, VecDB

def get_model_list() -> List[str]:
    ''' get all available model in your Ollama '''
    ollama_info = subprocess.run(["ollama", "list"], text=True, capture_output=True)
    ollama_info = ollama_info.stdout.strip()
    exist_model_names = ollama_info.split('\n')
    exist_model_names = [m.split(' ').pop(0) for m in exist_model_names[1:]]
    return exist_model_names

def check_model_exists(model_name:str) -> bool:
    ''' check if model available in Ollama'''
    exist_model_names = get_model_list()
    if not isinstance(model_name, str) or not model_name in exist_model_names:
        rich.print(f"Model name not found in ollama: {model_name}\nPlease pull model before executing")
        rich.print(f"Existing Models: {exist_model_names}")
        return False
    return True
    
def get_llm_info(llm: str | ChatOllama) -> Dict[str, Union[Dict[str, str], List[str]]]:
    ''' Get Model config from ollama command, parse and return dict '''
    llm_name = llm.model if isinstance(llm, ChatOllama) else llm
    assert isinstance(llm_name, str), "Input is not string or ChatOllama"
    # check model exist
    ollama_list = subprocess.run(["ollama", "list"], text=True, capture_output=True)
    if ollama_list.returncode != 0:
        raise RuntimeError(ollama_list.stderr.strip() or "ollama list failed")
    assert llm_name in ollama_list.stdout, f"Model {llm_name} is not found in Ollama"
    # get model info
    ollama_info = subprocess.run(["ollama", "show", llm_name], text=True, capture_output=True)
    if ollama_info.returncode != 0:
        raise RuntimeError(ollama_info.stderr.strip() or "ollama show failed")
    # prase info
    info: Dict[str, Any] = {}
    section = None
    for raw in ollama_info.stdout.splitlines():
        line = raw.strip()
        if not line: 
            continue
        if line in ("Model", "Capabilities", "Parameters", "License"):
            section = line
            info[section] = [] if section in ("Capabilities", "License") else {}
            continue
        if not section:
            continue
        if section in ("Capabilities", "License"):
            info[section].append(line)
        else:
            m = re.split(r"\s{2,}", line, maxsplit=1)
            if len(m) == 2:
                k, v = m
                info[section][k] = v
    return info

if __name__ == "__main__":
    check_model_exists(None)
    LLM_model, VecDB = init_system()
    rich.print(get_llm_info(LLM_model))