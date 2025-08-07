from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
import faiss
import rich
import os, sys
from typing import *
from config import init_VecDB
from CLI_Format import *

'''
Vector Database Manager for RAG system
Comuncation between Database and LLM, LangGraph
'''

class VDBManager:
    ''' 
    Use FAISS as VDB, manager focus on save, load, search, edit functions
    LLM will not be store as inner parameter, this class only focus on database management
    '''
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
    
    def retrieve(self, query:str, k:int=1, score_threshold:Optional[float]=None) -> List[Optional[Document]]:
        '''
        query : Search keyword
        k     : Search amount
        score_threshold : ( 0.0 ~ 1.0 ) Distance filter, remove searched document with larger distance value
        '''
        retrieve_messages = []
        if self.VDB.index.ntotal > 1: # check if database contians data
            if score_threshold is None:
                retrieve_data = self.VDB.similarity_search_with_score(query, k)
            else:
                score_threshold = min(1, max(score_threshold, 0))
                retrieve_data = self.VDB.similarity_search_with_score(query, k, score_threshold=score_threshold)
            # Append retrieve score into document
            for doc, score in retrieve_data:
                doc.metadata.update({"retrieve score" : score})
                retrieve_messages.append(doc)
        return retrieve_messages
    
    def as_retriever(self, k:int=1, score_threshold:Optional[float]=None):
        ''' Pack a retrieve action into retriever 
        k     : Search amount
        score_threshold : ( 0.0 ~ 1.0 ) Distance filter, remove searched document with larger distance value
        '''
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
    CLI_print("VDB Test", "Load VDB")
    test_manager.load("prebuild_VDB", "mini-wiki")
    #
    q_1 = "How can beetle larvae be differentiated from other insect larvae?"
    msg = test_manager.retrieve(q_1, 8)
    CLI_print("VDB Test", msg, "search without score threshold")
    CLI_next()
    msg = test_manager.retrieve(q_1, 8, 0.8)
    CLI_print("VDB Test", msg, "search with score threshold")
    CLI_next()
    #
    q_2 = "要如何從紐約市中心前往自由女神像?"
    msg = test_manager.retrieve(q_2, 8)
    CLI_print("VDB Test", msg, "search without score threshold")
    CLI_next()
    msg = test_manager.retrieve(q_2, 8, 0.8)
    CLI_print("VDB Test", msg, "search without score threshold")
    CLI_next()
    # 
    test_manager.save("prebuild_VDB", index_name="test_VDB")
    CLI_print("VDB Test", "All success")