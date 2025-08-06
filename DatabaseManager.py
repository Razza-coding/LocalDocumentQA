from langchain_core.messages import BaseMessage, AIMessage
from langchain_community.vectorstores import FAISS
import faiss
import rich
import os, sys
from typing import *
from config import init_VecDB

'''
Vector Database Manager for RAG system
Comuncation between Database and LLM, LangGraph
'''

class VDBManager:
    ''' Use FAISS as VDB, manager executes load, save, search '''
    def __init__(self, VDB: FAISS):
        # FAISS
        self.VDB: FAISS = VDB 
        pass
    
    def load(self, folder_path: str, index_name:str):
        ''' Load form prebuild VDB file '''
        folder_path = os.path.abspath(folder_path)
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            raise (OSError, f"Path Invalid, required folder path: {folder_path}")
        self.VDB = self.VDB.load_local(folder_path, embeddings=self.VDB.embedding_function, index_name=index_name, allow_dangerous_deserialization=True)
        rich.print(f"Load {self.VDB.index.ntotal} data from folder: {folder_path}")

    def save(self, folder_path:str, index_name:str='index'):
        ''' Save VDB content '''
        self.VDB.save_local(folder_path, index_name)
    
    def retrieve(self, query:str, k:int=1, score_threshold:Optional[float]=None):
        retrieve_messages = []
        retrieve_scores   = []
        if self.VDB.index.ntotal > 1: # check if database contians data
            score_threshold = min(1, max(score_threshold, 0))
            retrieve_data = self.VDB.similarity_search_with_score(query, k, score_threshold=score_threshold)
            # Transform into AI Message Class
            for text, score in retrieve_data:

                text_id = text.id
                test_score = score
                retrieve_messages.append(AIMessage(content=text.page_content))
                retrieve_scores.append(score)
        return retrieve_messages, retrieve_scores
    
    def as_retriever(self, k:int=1, score_threshold:Optional[float]=None):
        ''' Pack a retrieve action into retriever '''
        return self.VDB.as_retriever(search_type="similarity", k=k, score_threshold=score_threshold)
    
    def __clear(self):
        ''' delete all content inside VDB '''
        all_ids = list(self.VDB.index_to_docstore_id.values())
        rich.print(f"Clearing Database, total of {len(all_ids)} items.")
        self.VDB.delete([all_ids])
        # check
        assert self.VDB.index.ntotal == 0 # vectors
        assert len(self.VDB.docstore.__dict__) == 0 # documents
        assert len(self.VDB.index_to_docstore_id) == 0 # vector - document mapping

if __name__ == "__main__":
    test_manager = VDBManager(init_VecDB())
    #
    test_manager.load("prebuild_VDB", "mini-wiki")
    #
    msg, score = test_manager.retrieve("How can beetle larvae be differentiated from other insect larvae?", 4, 1.0)
    rich.print(score)
    rich.print(msg)
    #
    msg, score = test_manager.retrieve("要如何從紐約市中心前往自由女神像?", 4, 1.0)
    rich.print(score)
    rich.print(msg)