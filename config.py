from langchain_community.vectorstores import FAISS, DistanceStrategy
from langchain_community.docstore.in_memory import InMemoryDocstore

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.language_models import BaseChatModel

import faiss

import json
import os
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

def build_embedding(model_name: str="sentence-transformers/all-MiniLM-L6-v2", model_device: str="cpu", normalize_embeddings: bool=True):
    ''' Simplified Sentence Embedding Builder '''
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': model_device}, encode_kwargs={'normalize_embeddings': normalize_embeddings})
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
    __sys_init_message("VecDB Embeccing", embedding_model_name)

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

if __name__ == "__main__":
    LLM_model, VecDB = init_system()