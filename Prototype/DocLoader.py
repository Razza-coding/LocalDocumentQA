from langchain_community.document_loaders import TextLoader, Docx2txtLoader, PyPDFLoader, WebBaseLoader, SeleniumURLLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.documents import Document
import os, sys
from typing import List
import rich

'''
A Tool integrates multiple LangChain Tools to load Documents

Loader.load(filepath) -> List[Document]

Supports:
 - txt, csv, pdf, doc
 - web url
Not Supports
 - images
'''

class DocLoader:
    def __init__(self):
        self.root = os.path.abspath('.')
        pass
    
    def load_doc(self, document_path: str) -> List[Document]:
        ''' Load Single Document File into List of Documents, each item in list contains pages inside content file '''
        assert os.path.exists(document_path)
        assert os.path.isfile(document_path)
        document_path = os.path.abspath(document_path).replace("\\\\", "\\").replace("\\", "/")
        path_list = document_path.split("/")
        file_name = path_list[-1]
        file_ext  = file_name.split(".")[-1].lower() if '.' in file_name else None

        if file_ext is None:
            raise ValueError(f"File Without Extendsion : {file_name}")
        
        return self.__get_doc_loader(file_ext, document_path).load()
       
    def __get_doc_loader(self, file_ext: str, documet_path: str):
        ''' Get corresponding dcoument loader, raise error if extension not supported '''
        loader = None
        if file_ext == "txt":
            return TextLoader(documet_path, autodetect_encoding=True)
        if file_ext == "csv":
            return CSVLoader(documet_path)
        if file_ext == "pdf":
            return PyPDFLoader(documet_path)
        if "doc" in file_ext:
            return Docx2txtLoader(documet_path)
        # Default
        if loader is None:
            raise ValueError(f"File Extendsion not support : {file_ext}")
        return loader
    
    def load_web(self, url: str, static: bool =True) -> List[Document]:
        ''' Load Web content from url, static switchs between dynamic / static web loader '''
        if not static:
            return SeleniumURLLoader(url).load()
        return WebBaseLoader(url).load()
    
if __name__ == "__main__":
    # init
    loader = DocLoader()
    frame = "Load from {}. Total of {} pages. First page has {} charaters."

    # load txt
    test_doc = "./Prototype/test_log/test_document.txt"
    content = loader.load_doc(test_doc)
    rich.print(frame.format(test_doc, len(content), len(content[0].page_content)))

    # load static web
    test_url = "https://zh.wikipedia.org/wiki/%E4%BA%BA%E5%B7%A5%E6%99%BA%E6%85%A7"
    content = loader.load_web(test_url)
    rich.print(frame.format(test_url, len(content), len(content[0].page_content)))

    # load dynamic web
    test_url = "https://www.cnn.com/world"
    content = loader.load_web(test_url)
    rich.print(frame.format(test_url, len(content), len(content[0].page_content)))
