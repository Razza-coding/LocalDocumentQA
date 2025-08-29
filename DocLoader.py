from langchain_community.document_loaders import TextLoader, Docx2txtLoader, PyPDFLoader, WebBaseLoader, SeleniumURLLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.documents import Document
from selenium.webdriver.support.ui import WebDriverWait
import os, sys
from typing import List
import rich
import logging
import time
from ChunkingTools import StandAloneFactTextSplitter

'''
A Tool integrates multiple LangChain Tools to load Documents

Loader.load(filepath) -> List[Document]

Supports:
 - txt, csv, pdf, doc
 - web url
'''

logger = logging.getLogger(__name__)

class DynamicWebLoader(SeleniumURLLoader):
    ''' Load and wait url to be fully loaded, inherit from SeleiumURLLoader '''
    def __init__(self, urls, continue_on_failure = True, browser = "chrome", binary_location = None, executable_path = None, headless = True, arguments:List[str] = []):
        super().__init__(urls, continue_on_failure, browser, binary_location, executable_path, headless, arguments)

    def load(self, wait_time:int=5, verify_times:int=1, time_out:int=30):
        ''' Add wait and valid loop in load process '''
        from unstructured.partition.html import partition_html

        docs: List[Document] = list()
        driver = self._get_driver()

        # load loop
        for url in self.urls:
            page_content = []
            try:
                driver.get(url)
                # wait loop
                load_finish_count = 0
                for t in range(0, time_out, wait_time):
                    # wait
                    t_start = time.time()
                    WebDriverWait(driver, wait_time)
                    t_elapsed = time.time() - t_start
                    t_remain = wait_time - t_elapsed
                    if t_remain > 0:
                        time.sleep(t_remain)
                    # check
                    page_content.append(driver.page_source)
                    if len(page_content) > 1:
                        new_content_loaded = len(page_content[-1]) != len(page_content[-2])
                        load_finish_count = load_finish_count + 1 if not new_content_loaded else 0
                    rich.print(f"Time ({t}/{time_out})s - Verify Passed ({load_finish_count}/{verify_times}) - Content Size {len(page_content[-1])} chars")
                    #
                    if load_finish_count >= verify_times:
                        rich.print(f"Done Loading : {url}")
                        break
                    page_content = [page_content[-1]] # only keep newest content
            except Exception as e:
                if self.continue_on_failure:
                    logger.error(f"Error fetching or processing {url}, exception: {e}")
                else:
                    raise e
            
            page_content = page_content[-1]
            try:
                elements = partition_html(text=page_content)
                text = "\n\n".join([str(el) for el in elements])
                metadata = self._build_metadata(url, driver)
                docs.append(Document(page_content=text, metadata=metadata))
            except:
                ''' Unstructuable web page '''
                rich.print("Unstructurable Web, dumping html")
                metadata = self._build_metadata(url, driver)
                docs.append(Document(page_content=page_content, metadata=metadata))

        driver.quit()
        return docs        

class DocLoader:
    def __init__(self):
        ''' Universal Document Loader, File to List of Documents '''
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
    
    def load_static_web(self, url: str) -> List[Document]:
        ''' Load Web content from url, static switchs between dynamic / static web loader '''
        return WebBaseLoader(url).load()
    
    def load_dynamic_web(self, url: str, wait_time:int=5, verify_times:int=1, time_out:int=30) -> List[Document]:
        url = [url] if not isinstance(url, List) else url
        return DynamicWebLoader(url).load(wait_time, verify_times, time_out)
    
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
    content = loader.load_static_web(test_url)
    rich.print(frame.format(test_url, len(content), len(content[0].page_content)))

    # load dynamic web
    test_url = "https://www.cnn.com/world"
    content = loader.load_dynamic_web(test_url)
    rich.print(frame.format(test_url, len(content), len(content[0].page_content)))
