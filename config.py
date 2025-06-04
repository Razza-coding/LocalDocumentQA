from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

import faiss

import numpy as np

import json
import os

def init_system() -> (...) :
    ''' Build system core objects '''
    # build messages
    f_msg = "Build {0:<10} : {1:<{2}} Complete"

    # llm
    LLM_model_name = "gemma3:4b"
    LLM_temperature = 0.7
    LLM_url   = "http://localhost:11434"
    LLM_model = ChatOllama(
        model=LLM_model_name,
        temperature=LLM_temperature,
        base_url=LLM_url
    )
    print(f_msg.format("LLM", LLM_model_name, 10))

    # Vector Database
    # embeddings of VecDB
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs=embedding_model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print(f_msg.format("Embedding", embedding_model_name, 20))

    # index of VecDB
    dimension  = len(embeddings.embed_query("foo"))
    vdb_index = faiss.IndexFlatL2(dimension)
    print(f_msg.format("VDB index", "FlatL2", 10))

    # 使用 Hugging Face 包裝 VecDB
    VecDB = FAISS(
        embedding_function=embeddings,
        index=vdb_index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    print(f_msg.format("VecDB", "FAISS", 10))

    return (LLM_model, VecDB)

if __name__ == "__main__":
    LLM, VecDB = init_system()