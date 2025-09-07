import os
import sys
"" if os.environ.get("USER_AGENT") is not None else os.environ.update({"USER_AGENT" : "-"})
from HF_download_utils import set_hf_cache
set_hf_cache("./temp")
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QTextEdit, QLineEdit, QPushButton, 
                            QLabel, QScrollArea, QFrame, QSplitter, QCheckBox,
                            QRadioButton, QButtonGroup, QGroupBox, QFileDialog,
                            QListWidget, QListWidgetItem, QTextBrowser, QSizePolicy,
                            QSpacerItem, QMessageBox, QProgressBar, QToolTip,
                            QComboBox, QDialog, QStyle)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QMimeData, QPoint
from PyQt5.QtGui import QFont, QTextCursor, QDragEnterEvent, QDropEvent, QPalette, QIntValidator

from langchain.docstore.document import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph.state import CompiledStateGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointMetadata
from pydantic import BaseModel, Field

from MainGraph import build_main_graph, StartState, EndState, SystemPromptConfig
from config import init_LLM, init_VecDB, check_model_exists, get_model_list
from DatabaseManager import VDBManager
from PromptTools import to_text
from LogWriter import LogWriter
from DocLoader import DocLoader
from ChunkingTools import ClaimWithCitationsExtractor, Claim
import threading
import uuid
from typing import *
import rich
import time
import argparse

# ===============================
# parser

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run UI with Ollama, specific your LLM model name"
    )
    parser.add_argument(
        "-n", "--model", "--name",
        dest="model_name",
        metavar="MODEL",
        default="gemma3:4b",
        help="LLM model name, e.g. gemma3:4b / gpt-oss:latest / llama3.2:latest"
    )
    args = parser.parse_args(argv)
    
    return args

# ===============================
# System warning UI

class WarningWindow(QDialog):
    def __init__(self, message: str):
        super().__init__()
        self.setWindowTitle("Warning")
        self.setModal(True)
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)

        self.setStyleSheet("QDialog { background-color: #1a202c; }")
        self.setFixedHeight(240)
        self.setStyleSheet("color: #e2e8f0; font-size: 16px; border: none; background-color: #4a5568;")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header = QFrame()
        header.setFixedHeight(60)
        header.setStyleSheet("QFrame { background-color: #1a202c; } ")

        root.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(20, 20, 20, 20)
        content_layout.setSpacing(20)
        content.setStyleSheet(" QWidget { color: #e2e8f0; font-size: 16px; border: none; background-color: #4a5568; }")

        icon_label = QLabel()
        icon = self.style().standardIcon(QStyle.SP_MessageBoxWarning)
        icon_label.setPixmap(icon.pixmap(32, 32))

        msg_label = QLabel(message)
        msg_label.setStyleSheet(" QLabel { color: #e2e8f0; font-size: 16px; border: none; background-color: #4a5568; } ")

        content_layout.addWidget(icon_label, 0, Qt.AlignCenter)
        content_layout.addWidget(msg_label, 1, Qt.AlignLeft)
        scroll.setWidget(content)
        root.addWidget(scroll)

        bottom = QFrame()
        bottom.setStyleSheet("QFrame { background-color: #1a202c; }")
        bottom.setFixedHeight(60)

        root.addWidget(bottom)

        self.resize(720, min(self.sizeHint().height(), 320))

# ===============================
# Global Resource

class TextExtractor(BaseModel):
    ''' inner interface for calling different text extractor '''
    extract_method: Literal["default", "fast"] = Field(default="default")
    extractor: Any # default : ClaimWithCitationsExtractor , fast : RecursiveCharacterTextSplitter

class CoreResource():
    ''' global varibale accross the UI '''
    def __init__(self, model_name: str = "gemma3:4b", chunk_size: int = 800, chunk_overlap: int = 200):
        # get set lock
        self._lock_flag = False
        # check model
        if not check_model_exists(model_name=model_name):
            model_list = get_model_list()
            if len(model_list) == 0:
                warning_app =  QApplication.instance() if QApplication.instance() else QApplication(sys.argv)
                dlg = WarningWindow("There are no model in Ollama.\nPlease pull your first model.")
                dlg.exec_()
                sys.exit(0)
            else:
                rich.print(f"Model: {model_name} is not found.\nChange to model: {model_list[0]}.\nAll Model options: {model_list}")
                model_name = model_list[0]
        # baiscs
        self.model_name = model_name
        self.llm   = init_LLM(self.model_name)
        self.faiss = init_VecDB()
        # input prompt
        self.default_prompt_config = SystemPromptConfig(AI_name="JHON", professional_role="專業AI助理", temperature=0.8)
        self.prompt_config = self.default_prompt_config.model_copy(deep=True)
        # logger
        self.debug_logger = LogWriter("UI_DEBUG", "test_log")
        # core langgraph, database
        self.vector_database = VDBManager(self.faiss)
        self.state_graph: CompiledStateGraph = build_main_graph(self.llm, self.vector_database, self.debug_logger)
        # document anylize / chunking
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.cc_extractor = ClaimWithCitationsExtractor(self.llm)
        self.default_extract = TextExtractor(extract_method="default", extractor=self.cc_extractor)
        self.fast_extract    = TextExtractor(extract_method="fast",    extractor=RecursiveCharacterTextSplitter())
    
    def clear_short_memory(self):
        self._wait_unlock(critical=True)
        self._lock()
        ''' graph uses MemorySaver, get old thread, delete it, and start a new thread '''
        cfg = getattr(self.state_graph, "config", {}) or {}
        cfg_configurable = cfg.get("configurable", {})
        old_thread_id = cfg_configurable.get("thread_id", None)
        # delete old memory thread
        if old_thread_id:
            self.state_graph.checkpointer
            saver: Optional[BaseCheckpointSaver] = getattr(self.state_graph, "checkpointer")
            if saver:
                if hasattr(saver, "delete_thread"):
                    saver.delete_thread(thread_id=old_thread_id)
                elif hasattr(saver, "delete"):
                    saver.delete(thread_id=old_thread_id)
        # start new memory thread
        new_thread_id = f"main_graph_{str(uuid.uuid4())}"
        self.state_graph = self.state_graph.with_config(configurable={"configurable": {"thread_id": new_thread_id}})
        rich.print(f"Start New Memory Thread ID: {new_thread_id}")
        self._unlock()

    def set_chunk_option(self, chunk_size:Optional[int]=None, chunk_overlap:Optional[int]=None):
        self._wait_unlock()
        self._lock()
        if chunk_size:
            self.chunk_size = chunk_size
        if chunk_overlap:
            self.chunk_overlap = chunk_overlap
        self._unlock()
    
    def set_llm_temperature(self, temperature:float):
        self._wait_unlock()
        self._lock()
        self.prompt_config.temperature = temperature # graph use config setting
        self._unlock()
    
    def set_new_llm(self, model_name:str):
        self._wait_unlock(critical=True)
        self._lock()
        if check_model_exists(model_name=model_name):
            self.model_name = model_name
            self.llm = init_LLM(self.model_name)
            self.state_graph: CompiledStateGraph = build_main_graph(self.llm, self.vector_database, self.debug_logger)
            self.cc_extractor = ClaimWithCitationsExtractor(self.llm)
            self.default_extract = TextExtractor(extract_method="default", extractor=self.cc_extractor)
        self._unlock()

    def _lock(self):
        self._lock_flag = True

    def _unlock(self):
        self._lock_flag = False

    def _wait_unlock(self, critical=False):
        if not self._lock_flag:
            return
        t = 0
        while self._lock_flag:
            time.sleep(0.01)
            t += 1
            if t >= 100 and not critical:
                break

# ===============================
# Global Resource is init here
args = parse_args(sys.argv[1:])
core = CoreResource(model_name=getattr(args, "model_name"))

# ===============================
# Thread Functions

def get_response(user_input:str, graph:Optional[CompiledStateGraph], store:Dict):
    ''' invoke graph and get result '''
    store.update({"Done" : False})
    if graph:
        response = {}
        snapshot_prompt_config = core.prompt_config.model_copy(deep=True)
        for inner_state in graph.stream(StartState(input=user_input, prompt_config=snapshot_prompt_config)):
            response.update(inner_state)
            #response = to_text(response["output_msg"])
        response_message = to_text(response["end_node"].get("output_msg", "EMPTY RESPONSE"))
        retrieved_knowledge = response["info_retrieve_node"].get("extra_info_msg", [])
        retrieved_knowledge = [to_text(msg) for msg in retrieved_knowledge]
        store.update({
            "result" : response_message, 
            "retrieved_knowledge" : retrieved_knowledge
            })
        store.update({"Done" : True})
    else:
        # Demo message
        time.sleep(2)
        store.update({ "result" : "DEMO MESSAGE" })
        store.update({"Done" : True})

def analyze_document(document_path: str, text_extractor: TextExtractor, 
                     send_progress_func, check_stop_func, on_new_page_claim_func, on_finish_func):
    ''' load document, chunk it and extract informations '''
    extract_method = text_extractor.extract_method
    extractor      = text_extractor.extractor
    chunk_size    = int(core.chunk_size)
    chunk_overlap = int(core.chunk_overlap)
    page_claims = []
    page_chunks = []
    #
    current_progress = 0
    send_progress_func(current_progress)
    #
    loader = DocLoader()
    doc_pages = loader.load_doc(document_path)
    #
    if extract_method == "fast":
        extractor = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    for p_num, p in enumerate(doc_pages):
        #
        current_progress = float( p_num / len(doc_pages) )
        send_progress_func(current_progress)
        # check early stop
        if check_stop_func():
            rich.print("Thread stop early")
            break
        #
        if extract_method == "fast":
            chunks = extractor.split_text(p.page_content)
            claims = [[Claim(metadata={"document" : document_path, "page" : str(p_num), "chunk_id" : str(chunk_num)}, claim=chunk, citations=[( chunk_num, chunk )]) for chunk_num, chunk in enumerate(chunks)]]
        else:
            claims, chunks = extractor.split_text(p.page_content, chunk_size, chunk_overlap)
            [[claim.metadata.update({"document" : document_path, "page" : str(p_num)}) for claim in chunk_claim] for chunk_claim in claims]
        #
        on_new_page_claim_func(new_page_claims={ "page_claims" : claims, "page_chunks" : chunks })
        page_claims.extend(claims)
        page_chunks.extend(chunks)
    #
    current_progress = 1
    send_progress_func(current_progress)
    on_finish_func(all_extract_claim={ "all_page_claims" : page_claims, "all_page_chunks" : page_chunks })

# ===============================
# UI component

class ChatMessage(QFrame):
    ''' Individual chat message widget '''
    
    def __init__(self, message, is_human=False, claims:Optional[List]=None):
        super().__init__()
        self.setup_ui(message, is_human, claims)
    
    def setup_ui(self, message, is_human, claims):
        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)

        # Message bubble
        message_frame = QFrame()
        message_frame.setContentsMargins(10,10,10,10)
        message_layout = QVBoxLayout()
        message_layout.setSpacing(5)
        message_layout.setContentsMargins(0,0,0,0)

        # User indicator
        user_label = QLabel("AI Chat") if not is_human else QLabel("User")
        user_label.setStyleSheet("color: #8bb8e8; font-size: 18px; font-weight: bold;")
        user_label.setAlignment(Qt.AlignBottom)
        user_label.setContentsMargins(5,5,5,0)
        message_layout.addWidget(user_label)
    
        # Message content & style
        message_label = QLabel()
        message_label.setTextFormat(Qt.MarkdownText)
        message_label.setText(message)
        message_label.setContentsMargins(5,0,5,5)
        message_label.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.LinksAccessibleByMouse)
        message_label.setOpenExternalLinks(True)
        message_label.setWordWrap(True)
        message_layout.addWidget(message_label)
        message_frame.setLayout(message_layout)
       
        # side note under main messages
        if claims:
            claim_frame = QFrame()
            claim_frame.setContentsMargins(0,0,0,0)
            claim_frame.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

            claim_frame_layout = QHBoxLayout()
            claim_frame_layout.setContentsMargins(0,0,0,0)
            claim_frame_layout.setSpacing(5)
            claim_frame.setLayout(claim_frame_layout)
            claim_frame.setToolTipDuration(0)
            size_limit = 20
            for c in claims:
                display_claim_sentence = c[:size_limit] + "..." if len(c) >= size_limit else c
                claim_bubble = QFrame()
                claim_bubble.setToolTip(c)
                claim_bubble.setStyleSheet("color: #e2e8f0; background-color: #1a202c; border-radius: 14px;")
                claim_bubble.setContentsMargins(0,0,0,0)
                claim_bubble.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

                claim_bubble_layout = QHBoxLayout()

                claim_sentence = QLabel()
                claim_sentence.setAlignment(Qt.AlignTop)
                claim_sentence.setContentsMargins(0,0,0,0)
                claim_sentence.setStyleSheet("color: #e2e8f0; font-size: 14px;")
                claim_sentence.setText(display_claim_sentence)
                claim_sentence.setToolTip(c)
                claim_sentence.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.LinksAccessibleByMouse)

                claim_bubble.setLayout(claim_bubble_layout)
                claim_bubble_layout.addWidget(claim_sentence)
                claim_frame_layout.addWidget(claim_bubble)
                
            message_layout.addWidget(claim_frame)
        
        # Human message - align right
        if is_human:
            layout.addStretch()
            user_label.setAlignment(Qt.AlignRight)
            message_frame.setStyleSheet("""
                QFrame {
                    background-color: #4a5568;
                    border-radius: 16px;
                    max-width: 800px;
                }
                QLabel {
                    color: #ffffff;
                    font-size: 18px;
                    line-height: 1.5;
                }
            """)

        layout.addWidget(message_frame)

        # AI message align left
        if not is_human: 
            layout.addStretch()
            user_label.setAlignment(Qt.AlignLeft)
            message_frame.setStyleSheet("""
                QFrame {
                    background-color: #2d3748;
                    border-radius: 16px;
                    max-width: 800px;
                }
                QLabel {
                    color: #e2e8f0;
                    font-size: 18px;
                    line-height: 1.5;
                }
            """)
            
        self.setLayout(layout)
        self.setContentsMargins(15, 20, 15, 20)
        self.setStyleSheet(''' border: transparent;  border-radius: 16px; ''')

class FileDropArea(QFrame):
    ''' Drag and drop area for file uploads '''
    
    def __init__(self, on_files_add):
        super().__init__()
        self.setAcceptDrops(True)
        self.setup_ui(on_files_add)
    
    def setup_ui(self, on_files_add):
        self.on_files_add = on_files_add

        # Main text
        main_text = QLabel("Upload Files\n( PDF, DOC, TXT )")
        main_text.setAlignment(Qt.AlignCenter)
        main_text.setStyleSheet("""
            QLabel {
                color: #a0aec0;
                font-size: 14px;
                font-weight: bold;
                padding: 0px;
                margin: 0px;
                border: transparent;
            }
        """)
        drag_drop_frame = QFrame()
        drag_drop_frame.setStyleSheet("""
            QFrame {
                border: 2px dashed #4a5568;
                border-radius: 8px;
                background-color: #1a202c;
                min-height: 150px;
                min-width: 150px;
            }
        """)
        drag_drop_layout = QVBoxLayout()
        drag_drop_layout.setAlignment(Qt.AlignCenter)
        drag_drop_layout.setSpacing(0)
        drag_drop_layout.addWidget(main_text)
        drag_drop_frame.setLayout(drag_drop_layout)
        
        # Browse button
        self.browse_button = QPushButton("Browse files")
        self.browse_button.setStyleSheet("""
            QPushButton {
                background-color: #4299e1;
                color: white;
                border: transparent;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3182ce;
            }
        """)
        self.browse_button.clicked.connect(self.browse_files)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignBottom)
        layout.setSpacing(10)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(drag_drop_frame)
        layout.addWidget(self.browse_button)
        
        self.setLayout(layout)
        self.setStyleSheet("""
            QFrame {
                border: none;
                min-height: 150px;
            }
        """)
    
    def browse_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Files", "", 
            "Documents (*.pdf *.docx *.txt);;All Files (*)"
        )
        if files:
            rich.print(f"Selected files: {files}")
            if isinstance(files, str):
                self.on_files_add([files])
            else:
                self.on_files_add(files)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
            self.setStyleSheet("""
                QFrame {
                    border: 2px dashed #4299e1;
                    border-radius: 8px;
                    background-color: #2a4365;
                    min-height: 150px;
                }
            """)
        else:
            event.ignore()
    
    def dragLeaveEvent(self, event):
        self.setStyleSheet("""
            QFrame {
                border: 2px dashed #4a5568;
                border-radius: 8px;
                background-color: #1a202c;
                min-height: 150px;
            }
        """)
    
    def dropEvent(self, event: QDropEvent):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        rich.print(f"Selected files: {files}")
        self.setStyleSheet("""
            QFrame {
                border: 2px dashed #4a5568;
                border-radius: 8px;
                background-color: #1a202c;
                min-height: 150px;
            }
        """)

class SettingsPanel(QFrame):
    ''' system setting panel '''
    def __init__(self, clear_chat_func=None):
        super().__init__()
        self.show_settings = True
        self._enter_pressed = False 
        self.clear_chat = clear_chat_func
        # buttom mappings
        self.temp_maps = {
            "Creative" : 0.8,
            "Standard" : 0.5,
            "Precise"  : 0.3,
        }
        self.setup_ui()
        self.set_show_setting(True)
    
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 20, 15, 20)
        layout.setSpacing(20)
        
        # Collapse button
        self.collapse_button = QPushButton("<")
        self.collapse_button.setFixedSize(30, 30)
        self.collapse_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: transparent;
                border-radius: 16px;
                color: #a0aec0;
                font-size: 18px;
                font-weight: bold;
            }
            QPushButton:hover {
                color: #ffffff;
                border: 2px #a0aec0;
                border-radius: 16px;
            }
        """)
        self.collapse_button.clicked.connect(self.on_collapse_button_pressed)
        
        collapse_layout = QHBoxLayout()
        collapse_layout.addStretch()
        collapse_layout.addWidget(self.collapse_button)
        layout.addLayout(collapse_layout)
        
        # Settings
        self.setting_container = QFrame()
        self.setting_container.setStyleSheet("background-color: transparent; border: transparent;")
        self.setting_container.setContentsMargins(0,0,0,0)
        setting_container_layout = QVBoxLayout()
        setting_container_layout.setContentsMargins(0,0,0,0)
        self.setting_container.setLayout(setting_container_layout)
        
        # title
        settings_title = QLabel("⚙ Settings")
        settings_title.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 16px;
                font-weight: bold;
            }
        """)
        setting_container_layout.addWidget(settings_title)
        setting_container_layout.setSpacing(20)
        
        # Model Configuration
        model_group = QGroupBox()
        model_group.setTitle("▼ Model")
        model_group.setContentsMargins(15,20,15,20)
        model_group.setStyleSheet("""
            QGroupBox {
                color: #a0aec0;
                font-size: 16px;
                font-weight: bold;
                border: none;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 0px;
                padding: 0 5px 0 0;
            }
        """)
        
        model_layout = QVBoxLayout()
        model_layout.setContentsMargins(0,20,0,20)
        model_layout.setSpacing(20)

        # model name options
        model_name_layout = QVBoxLayout()
        model_name_layout.setContentsMargins(0,0,0,0)
        model_name_layout.setSpacing(10)

        model_name_title = QLabel()
        model_name_title.setText("Model Name")
        model_name_title.setStyleSheet("QLabel { color: #a0aec0; font-size: 14px; border: transparent}")
        model_name_layout.addWidget(model_name_title)

        self.model_name_select = QComboBox()
        self.model_name_select.addItems(get_model_list())
        self.model_name_select.setCurrentText(core.model_name)
        self.model_name_select.setFixedHeight(30)
        self.model_name_select.setContentsMargins(0,0,5,0)
        self.model_name_select.setStyleSheet("""
            QComboBox {
                color: #e2e8f0;
                font-size: 14px;
                background-color: transparent;
                border: 1px solid #a0aec0;
                border-radius: 6px;
            }
            QComboBox QAbstractItemView {
                color: #e2e8f0;
                font-size: 14px;
                background-color: #1a202c;
                border: 1px solid #a0aec0;
                selection-color: #e2e8f0;
                selection-background-color: #2d3748;
            }
        """)
        model_name_layout.addWidget(self.model_name_select)
        model_layout.addLayout(model_name_layout)
        # connect llm update event
        self.model_name_select.currentIndexChanged.connect(self.on_model_name_changed)
        
        # temperature options
        temp_layout = QVBoxLayout()
        temp_layout.setContentsMargins(0,0,0,0)
        temp_layout.setSpacing(10)

        temp_label = QLabel("Temperature")
        temp_label.setText("Temperature")
        temp_label.setStyleSheet("QLabel { color: #a0aec0; font-size: 14px; border: transparent}")
        temp_layout.addWidget(temp_label)

        temp_group_layout = QVBoxLayout()
        temp_group_layout.setContentsMargins(0,0,0,0)
        temp_group_layout.setSpacing(5)
        self.temp_group = QButtonGroup()
        name_frame = "{name:<10} {temp:>4}"
        for i, name in enumerate(self.temp_maps):
            temp = self.temp_maps[name]
            radio = QRadioButton()
            radio.setText(name_frame.format(name=name, temp=round(temp, 2)))
            if i == 0:  # Standard selected
                radio.setChecked(True)
                radio.clicked.connect(lambda: self.on_tempurature_changed(0.8))
            if i == 1:  
                radio.clicked.connect(lambda: self.on_tempurature_changed(0.5))
            if i == 2:
                radio.clicked.connect(lambda: self.on_tempurature_changed(0.3))
            radio.setStyleSheet("""
                QRadioButton {
                    color: #a0aec0;
                    font-size: 14px;
                    font-weight: normal;
                }
                QRadioButton:checked {
                    color: #a0aec0;
                    font-weight: bold;
                }
                QRadioButton::indicator {
                    width: 10px;
                    height: 10px;
                    border: 2px solid #a0aec0;
                    border-radius: 6px;
                    background: white;
                }
                QRadioButton::indicator:checked {
                    border: 2px solid #a0aec0;
                    background: black;
                }
            """)
            self.temp_group.addButton(radio)
            temp_group_layout.addWidget(radio)

        temp_layout.addLayout(temp_group_layout)
        model_layout.addLayout(temp_layout)
        model_group.setLayout(model_layout)
        setting_container_layout.addWidget(model_group)


        # Document analyze Configuration
        analyze_group = QGroupBox()
        analyze_group.setTitle("▼ Analyze")
        analyze_group.setContentsMargins(15,20,15,20)
        analyze_group.setStyleSheet("""
            QGroupBox {
                color: #a0aec0;
                font-size: 16px;
                font-weight: bold;
                border: none;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 0px;
                padding: 0 5px 0 0;
            }
        """)
        
        analyze_layout = QVBoxLayout()
        analyze_layout.setContentsMargins(0,20,0,20)
        analyze_layout.setSpacing(20)

        # analyze options
        chunking_layout = QVBoxLayout()
        chunking_layout.setContentsMargins(0,0,0,0)
        chunking_layout.setSpacing(10)

        chunking_label = QLabel()
        chunking_label.setText("Chunking Configuration")
        chunking_label.setStyleSheet("QLabel { color: #a0aec0; font-size: 14px; border: transparent}")
        chunking_layout.addWidget(chunking_label)
        
        chunking_group_layout = QVBoxLayout()
        chunking_group_layout.setContentsMargins(0,0,0,0)
        chunking_group_layout.setSpacing(5)
        
        # chunk size
        chunk_size_layout = QHBoxLayout()
        chunk_size_layout.setContentsMargins(0,0,0,0)
        chunk_size_layout.setSpacing(10)
        chunk_size_title = QLabel()
        chunk_size_title.setText("Size")
        chunk_size_title.setStyleSheet("QLabel { color: #a0aec0; font-size: 14px; border: transparent}")
        self.chunk_size = QLineEdit()
        self.chunk_size.setText("800")
        self.chunk_size.setStyleSheet("QLineEdit { color: #a0aec0; font-size: 14px; border: 1px solid #a0aec0; border-radius: 6px;}")
        self.chunk_size.setValidator(QIntValidator(1, 999999))
        chunk_size_layout.addWidget(chunk_size_title)
        chunk_size_layout.setStretch(0,1)
        chunk_size_layout.addWidget(self.chunk_size)
        chunk_size_layout.setStretch(1,2)
        # when press enter change chunk size
        self.chunk_size.editingFinished.connect(self.on_change_chunk_size)
        # chunk overlap
        chunk_overlap_layout = QHBoxLayout()
        chunk_overlap_layout.setContentsMargins(0,0,0,0)
        chunk_overlap_layout.setSpacing(10)
        chunk_overlap_title = QLabel()
        chunk_overlap_title.setText("Overlap")
        chunk_overlap_title.setStyleSheet("QLabel { color: #a0aec0; font-size: 14px; border: transparent}")
        self.chunk_overlap = QLineEdit()
        self.chunk_overlap.setText("200")
        self.chunk_overlap.setStyleSheet("QLineEdit { color: #a0aec0; font-size: 14px; border: 1px solid #a0aec0; border-radius: 6px;}")
        self.chunk_overlap.setValidator(QIntValidator(1, 999999))
        chunk_overlap_layout.addWidget(chunk_overlap_title)
        chunk_overlap_layout.setStretch(0,1)
        chunk_overlap_layout.addWidget(self.chunk_overlap)
        chunk_overlap_layout.setStretch(1,2)
        # when press enter change chunk overlap
        self.chunk_overlap.editingFinished.connect(self.on_chunk_overlap_changed)
        #
        chunking_group_layout.addLayout(chunk_size_layout)
        chunking_group_layout.addLayout(chunk_overlap_layout)
        chunking_layout.addLayout(chunking_group_layout)

        #
        analyze_layout.addLayout(chunking_layout)
        analyze_group.setLayout(analyze_layout)

        setting_container_layout.addWidget(analyze_group)

        # advence prompt setting
        advanced_group = QGroupBox()
        advanced_group.setTitle("▼ Advanced")
        advanced_group.setContentsMargins(15,20,15,20)
        advanced_group.setStyleSheet("""
            QGroupBox {
                color: #a0aec0;
                font-size: 16px;
                font-weight: bold;
                border: none;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 0px;
                padding: 0 5px 0 0;
            }
        """)
        setting_container_layout.addWidget(advanced_group)
        
        advanced_layout = QVBoxLayout()
        advanced_layout.setContentsMargins(0,20,0,20)
        advanced_layout.setSpacing(20)
        advanced_group.setLayout(advanced_layout)
        
        prompt_layout = QVBoxLayout()
        prompt_layout.setContentsMargins(0,0,0,0)
        prompt_layout.setSpacing(10)

        # name prompt
        text_title_style = "QLabel { color: #a0aec0; font-size: 14px; border: transparent;}"
        text_edit_style = """
            QTextEdit {
                color: #a0aec0;
                font-size: 14px;
                border: 1px solid #4a5568;
                border-radius: 6px;
                background-color: #1a202c;
                padding: 5px;
            }
        """
        name_prompt_label = QLabel("AI Name")
        name_prompt_label.setStyleSheet(text_title_style)
        prompt_layout.addWidget(name_prompt_label)
        
        self.name_prompt_text = QTextEdit()
        self.name_prompt_text.setPlaceholderText("(Default) " + core.default_prompt_config.AI_name)
        self.name_prompt_text.setMaximumHeight(50)
        self.name_prompt_text.setStyleSheet(text_edit_style)
        prompt_layout.addWidget(self.name_prompt_text)

        # AI profession
        pro_prompt_label = QLabel("AI Profession")
        pro_prompt_label.setStyleSheet(text_title_style)
        prompt_layout.addWidget(pro_prompt_label)
        
        self.pro_prompt_text = QTextEdit()
        self.pro_prompt_text.setPlaceholderText("(Default) " + core.default_prompt_config.professional_role)
        self.pro_prompt_text.setMaximumHeight(50)
        self.pro_prompt_text.setStyleSheet(text_edit_style)
        prompt_layout.addWidget(self.pro_prompt_text)

        # main lang
        lang_prompt_label = QLabel("Main Langue")
        lang_prompt_label.setStyleSheet(text_title_style)
        prompt_layout.addWidget(lang_prompt_label)
        
        self.lang_prompt_text = QTextEdit()
        self.lang_prompt_text.setPlaceholderText("(Default) " + core.default_prompt_config.chat_lang)
        self.lang_prompt_text.setMaximumHeight(50)
        self.lang_prompt_text.setStyleSheet(text_edit_style)
        prompt_layout.addWidget(self.lang_prompt_text)

        # negitive rules
        negitive_prompt_label = QLabel("Negitive Rules")
        negitive_prompt_label.setStyleSheet(text_title_style)
        prompt_layout.addWidget(negitive_prompt_label)
        
        self.negtive_prompt_text = QTextEdit()
        self.negtive_prompt_text.setPlaceholderText("(Default) " + core.default_prompt_config.negitive_rule)
        self.negtive_prompt_text.setMinimumHeight(50)
        self.negtive_prompt_text.setMaximumHeight(100)
        self.negtive_prompt_text.setStyleSheet(text_edit_style)
        self.negtive_prompt_text.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        prompt_layout.addWidget(self.negtive_prompt_text)

        # response format
        format_prompt_label = QLabel("Response Format")
        format_prompt_label.setStyleSheet(text_title_style)
        prompt_layout.addWidget(format_prompt_label)
        
        self.format_prompt_text = QTextEdit()
        self.format_prompt_text.setPlaceholderText("(Default) " + core.default_prompt_config.output_format)
        self.format_prompt_text.setMinimumHeight(50)
        self.format_prompt_text.setMaximumHeight(400)
        self.format_prompt_text.setStyleSheet(text_edit_style)
        prompt_layout.addWidget(self.format_prompt_text)
        prompt_layout.addStretch()

        #
        advanced_layout.addLayout(prompt_layout)      
        setting_container_layout.addStretch()
        layout.addWidget(self.setting_container)
        layout.addStretch()

        self.setLayout(layout)
        self.setFixedWidth(250)
        self.setStyleSheet("QFrame { background-color: #1a202c; border-right: 1px solid #2d3748; } ")
        # connect all text edit
        self.name_prompt_text.textChanged.connect(lambda : self.on_prompt_changed("name", self.name_prompt_text.toPlainText()))
        self.pro_prompt_text.textChanged.connect(lambda : self.on_prompt_changed("profession", self.pro_prompt_text.toPlainText()))
        self.lang_prompt_text.textChanged.connect(lambda : self.on_prompt_changed("chat_lang", self.lang_prompt_text.toPlainText()))
        self.negtive_prompt_text.textChanged.connect(lambda : self.on_prompt_changed("negtive_rule", self.negtive_prompt_text.toPlainText()))
        self.format_prompt_text.textChanged.connect(lambda : self.on_prompt_changed("output_format", self.format_prompt_text.toPlainText()))
    
    def on_model_name_changed(self):
        new_model_name = self.model_name_select.currentText()
        if new_model_name != core.model_name:
            core.set_new_llm(new_model_name)
            if callable(self.clear_chat):
                self.clear_chat() # trigger clear chat message execution in DocumentQAChat
            self.model_name_select.setCurrentText(core.model_name)

    def on_prompt_changed(self, prompt_slot, text):
        if prompt_slot == "name":
            core.prompt_config.AI_name = text or core.default_prompt_config.AI_name
        if prompt_slot == "profession":
            core.prompt_config.professional_role = text or core.default_prompt_config.professional_role
        if prompt_slot == "chat_lang":
            core.prompt_config.chat_lang = text or core.default_prompt_config.chat_lang
        if prompt_slot == "negtive_rule":
            core.prompt_config.negitive_rule = text or core.default_prompt_config.negitive_rule
        if prompt_slot == "output_format":
            core.prompt_config.output_format = text or core.default_prompt_config.output_format
    
    def on_tempurature_changed(self, temp):
        core.set_llm_temperature(temp)

    def on_change_chunk_size(self):
        old_chunk_size = core.chunk_size
        try:
            new_chunk_size = int(self.chunk_size.text())
            new_chunk_size = max(1, new_chunk_size)
            core.set_chunk_option(chunk_size=new_chunk_size)
        except:
            self.chunk_size.setText(str(old_chunk_size))
            pass

    def on_chunk_overlap_changed(self):
        old_chunk_overlap =  core.chunk_overlap
        try:
            new_chunk_overlap = int(self.chunk_overlap.text())
            new_chunk_overlap = max(1, new_chunk_overlap)
            core.set_chunk_option(chunk_overlap=new_chunk_overlap)
        except:
            self.chunk_overlap.setText(str(old_chunk_overlap))
            pass

    def on_collapse_button_pressed(self):
        self.set_show_setting(not self.show_settings)

    def on_enter_pressed(self):
        self._enter_pressed = True

    def set_show_setting(self, show):
        self.show_settings = show
        if self.show_settings:
            self.setting_container.show()
            self.setFixedWidth(250)
            self.collapse_button.setText("<")
        else:
            self.setting_container.hide()
            self.setFixedWidth(50)
            self.collapse_button.setText(">")

class SuggestionPanel(QFrame):
    """Suggestion panel"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        self.setStyleSheet("""
            QFrame {
                border: none;
            }
        """)

        layout = QVBoxLayout()
        layout.setSpacing(5)

        # Suggestion title
        suggestion_title = QLabel("Suggestion")
        suggestion_title.setFixedHeight(20)
        suggestion_title.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 14px;
                font-weight: bold;
                border: none;
            }
        """)
        layout.addWidget(suggestion_title, alignment=Qt.AlignTop)
        layout.setStretch(0,1)
        
        # Suggestion buttons
        suggestions = ["Q1", "Q2", "Q3"]
        for s_num, suggestion in enumerate(suggestions, start=1):
            btn = QPushButton(f"💬 {suggestion}")
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #2d3748;
                    color: #a0aec0;
                    border: 1px solid #4a5568;
                    border-radius: 6px;
                    padding: 10px;
                    text-align: left;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background-color: #4a5568;
                    color: #ffffff;
                }
            """)
            layout.setStretch(s_num, 1)
            layout.addWidget(btn, alignment=Qt.AlignTop)

        push = QLabel(" ")
        push.setStyleSheet("""
            QLabel {
                border: none;
            }
        """)
        layout.addWidget(push, alignment=Qt.AlignBottom)
        layout.setStretch(len(suggestion)+1, 6 - len(suggestions))       
        self.setLayout(layout)

class FileItemWidget(QFrame):
    ''' single document item widget '''
    def __init__(self, file_path: str, on_new_page_claim_added=None, on_delete_callback=None, parent=None):
        super().__init__(parent)
        self.page_claims : List[ List[Claim] ]     = []
        self.page_chunks : List[ Tuple[int, str] ] = []
        self.file_path = file_path
        self.on_delete_callback = on_delete_callback
        self.on_new_page_claim_added = on_new_page_claim_added

        # llm extractor process
        self.loader = DocLoader()
        self.extract_thread = None
        self.extract_progress = 1
        self.stop_thread = True

        self.set_ui()

    def set_ui(self):
        self.setFrameShape(QFrame.StyledPanel)
        self.setContentsMargins(0,0,0,0)
        self.setStyleSheet("color: #ffffff; font-size: 14px; background-color: #1a202c; border: 1px solid #4a5568; border-radius: 16px;")
        self.setMaximumWidth(380)

        # main layout
        v_root = QVBoxLayout(self)
        v_root.setContentsMargins(5,5,5,5)
        v_root.setSpacing(5)

        # file name bubble
        name_bubble = QFrame()
        name_bubble.setContentsMargins(5,5,5,5)
        name_bubble.setStyleSheet("background-color: transparent; border: transparent;")
        name_layout = QVBoxLayout(name_bubble)
        name_layout.setContentsMargins(0,0,0,0)
        name_layout.setSpacing(5)
        # file name
        self.lbl_name = QLabel()
        self.lbl_name.setText(os.path.basename(self.file_path))
        self.lbl_name.setToolTip(self.file_path)
        self.lbl_name.setContentsMargins(0,0,0,0)
        self.lbl_name.setStyleSheet("color: #ffffff; font-size: 16px; font-weight: bold; background-color: transparent; border: transparent; border-radius: 0px;")
        self.lbl_name.setTextInteractionFlags(Qt.TextSelectableByMouse)
        # cull path
        limit_size = 25
        display_path = self.file_path[:limit_size] + "..." if len(self.file_path) >= limit_size else self.file_path
        self.lbl_path = QLabel()
        self.lbl_path.setText(display_path)
        self.lbl_path.setToolTip(self.file_path)
        self.lbl_path.setContentsMargins(0,0,0,0)
        self.lbl_path.setStyleSheet("color: #ffffff; font-size: 14px; background-color: transparent; border: transparent; border-radius: 0px;")
        self.lbl_path.setTextInteractionFlags(Qt.TextSelectableByMouse)
        name_layout.addWidget(self.lbl_name)
        name_layout.addWidget(self.lbl_path)

        # progress percentage
        self.lbl_percent = QLabel()  # e.g. "30%"
        self.lbl_percent.setText("")
        self.lbl_percent.setVisible(False)
        self.lbl_percent.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_percent.setMinimumWidth(40)
        self.lbl_percent.setStyleSheet("color: #ffffff; font-size: 14px;  background-color: transparent; border: transparent; border-radius: 0px;")

        # analyze button
        self.btn_analyze = QPushButton("分析", self)
        self.btn_analyze.setStyleSheet("color: #a0aec0; font-size: 16px; background-color: #1a202c; border: 1px solid #a0aec0; border-radius: 16px; padding: 6px 14px;")
        self.btn_analyze.clicked.connect(self.start_analysis)

        # delete button
        self.btn_delete = QPushButton("刪除", self)
        self.btn_delete.setStyleSheet("color: #a0aec0; font-size: 16px; background-color: #1a202c; border: 1px solid #a0aec0; border-radius: 16px; padding: 6px 14px;")
        self.btn_delete.clicked.connect(self.delete_self)

        h_top = QHBoxLayout()
        h_top.setSpacing(5)
        h_top.setContentsMargins(5,5,5,5)
        h_top.addWidget(name_bubble, 1)
        h_top.addWidget(self.lbl_percent)
        h_top.addWidget(self.btn_analyze)
        h_top.addWidget(self.btn_delete)

        v_root.addLayout(h_top)

        # progress bar
        self.progress = QProgressBar(self)
        self.progress.setStyleSheet("background-color: transparent; border: transparent;")
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(8)
        v_root.addWidget(self.progress)

        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._on_tick)

    def start_analysis(self):
        ''' Start analyze thread, and then retrieve result using QTimer '''
        if self.timer.isActive():
            return
        if self.progress.value() >= 100:
            self.progress.setValue(0)
        self.lbl_percent.setText("0%")
        self.lbl_percent.setVisible(True)
        self.btn_analyze.setEnabled(False)
        #
        self._set_thread_stop(False)
        self._set_extract_progess(0)
        #
        defualt_extract = core.default_extract
        fast_extract    = core.fast_extract
        #
        self.extract_thread = threading.Thread(
            target=analyze_document, 
            daemon=True, 
            args=(
                self.file_path, 
                defualt_extract,
                self._set_extract_progess,
                self._is_thread_set_stop,
                self._add_new_claim,
                self._handle_extract_result,
                ))
        self.extract_thread.start()
        #
        self.timer.start()
    
    def _set_extract_progess(self, progress: float):
        self.extract_progress = progress
    
    def _set_thread_stop(self, stop: bool):
        self.stop_thread = stop

    def _is_thread_set_stop(self):
        return self.stop_thread
    
    def _add_new_claim(self, new_page_claims:Dict):
        ''' show page claims on UI '''
        page_claims = new_page_claims.get("page_claims", None)
        page_chunks = new_page_claims.get("page_chunks", None)
        if len(page_claims) == 0 or len(page_chunks) == 0:
            return
        self.page_claims.extend(page_claims)
        self.page_chunks.extend(page_chunks)
        self.on_new_page_claim_added(page_claims, page_chunks)

    def _handle_extract_result(self, all_extract_claim:Dict):
        ''' store all claims in vdb '''
        data_amount_before = core.vector_database.amount()
        for page_claim in self.page_claims:
            for c in page_claim:
                core.vector_database.store(c.claim)
        data_amount_after = core.vector_database.amount()
        rich.print(f"\nDisplay newly added data:")
        core.vector_database.view_data(3)
        rich.print(f"\nTotal of {data_amount_after - data_amount_before} new data is added in database.")

    def _on_tick(self):
        cur = int(self.extract_progress * 100)
        if cur > 100:
            cur = 100
        self.progress.setValue(cur)
        self.lbl_percent.setText(f"{cur}%")
        if cur >= 100:
            self.timer.stop()
            self.btn_analyze.setEnabled(True)
    
    def delete_self(self):
        ''' delete bubble, stop thread '''
        if self.extract_thread and self.extract_thread.is_alive():
            self._set_thread_stop(True)

        if self.timer.isActive():
            self.timer.stop()

        parent_layout = self.parentWidget().layout() if self.parentWidget() else None
        if parent_layout:
            parent_layout.removeWidget(self)

        self.deleteLater()

        if callable(self.on_delete_callback):
            self.on_delete_callback(self.file_path)

class ClaimCitationItem(QFrame):
    ''' Single Claim and Citation item '''
    def __init__(self, claim:str, citation:Optional[list], source_document:Optional[str]=None, 
                 page_id:Optional[str]=None, chunk_id:Optional[str]=None, claim_id:Optional[str]=None,
                 ):
        super().__init__()
        self.claim    = claim
        self.citation = citation
        # source
        self.document = str(source_document)
        self.page_id  = str(page_id)
        self.chunk_id = str(chunk_id)
        self.claim_id = str(claim_id)
        self.set_ui()
    
    def set_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)
        self.setLayout(layout)
        self.setStyleSheet("background-color: #1a202c; border: 1px solid #4a5568; border-radius: 16px;")

        title_layout = QHBoxLayout()
        title_layout.setSpacing(10)
        if self.document:
            label = QLabel()
            label.setText(f"{os.path.basename(self.document)}")
            label.setToolTip(self.document)
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            label.setContentsMargins(5,5,5,5)
            label.setStyleSheet("color: #ffffff; font-size: 14px; background-color: #2d3748; border: transparent; border-radius: 16px;")
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)
        if self.page_id:
            label = QLabel()
            label.setText(f"Page: {self.page_id}")
            label.setContentsMargins(5,5,5,5)
            label.setStyleSheet("color: #ffffff; font-size: 14px; background-color: #2d3748; border: transparent; border-radius: 16px;")
            title_layout.addWidget(label)
            label.setAlignment(Qt.AlignCenter)
        if self.chunk_id:
            label = QLabel()
            label.setText(f"Chunk: {self.chunk_id}")
            label.setContentsMargins(5,5,5,5)
            label.setStyleSheet("color: #ffffff; font-size: 14px; background-color: #2d3748; border: transparent; border-radius: 16px;")
            label.setAlignment(Qt.AlignCenter)
            title_layout.addWidget(label)
        if self.claim_id:
            label = QLabel()
            label.setText(f"Claim: {self.claim_id}")
            label.setContentsMargins(5,5,5,5)
            label.setStyleSheet("color: #ffffff; font-size: 14px; background-color: #2d3748; border: transparent; border-radius: 16px;")
            label.setAlignment(Qt.AlignCenter)
            title_layout.addWidget(label)
        layout.addLayout(title_layout)

        display_limit = 100
        claim_sentence = QLabel()
        claim_sentence.setMaximumWidth(400)
        display_text = self.claim[:display_limit] + "..." if len(self.claim) >= display_limit else self.claim
        claim_sentence.setText(display_text)
        claim_sentence.setToolTip(self.claim)
        claim_sentence.setWordWrap(True)
        claim_sentence.setOpenExternalLinks(True)
        claim_sentence.setTextInteractionFlags(Qt.TextSelectableByMouse)
        claim_sentence.setContentsMargins(5,5,5,5)
        claim_sentence.setStyleSheet("color: #ffffff; font-size: 14px; background-color: #2d3748; border: transparent; border-radius: 16px;")
        layout.addWidget(claim_sentence)
        
        display_limit = 50
        if self.citation:
            claim_layout = QVBoxLayout()
            claim_layout.setContentsMargins(20, 0, 0, 0)
            claim_layout.setSpacing(5)
            for claim_sentence in self.citation:
                claim_label = QLabel()
                claim_label.setWordWrap(True)
                claim_label.setOpenExternalLinks(False)
                claim_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
                claim_label.setContentsMargins(5,5,5,5)
                claim_label.setStyleSheet("color: #ffffff; font-size: 14px; background-color: #2d3748; border: transparent; border-radius: 16px;")
                display_text = claim_sentence[:display_limit] + "..." if len(claim_sentence) >= display_limit else claim_sentence
                claim_label.setText(display_text)
                claim_label.setToolTip(claim_sentence)
                claim_layout.addWidget(claim_label)
            layout.addLayout(claim_layout)

class ClaimCitationDisplayWidget(QFrame):
    ''' Show Claim and Citation extracted form document '''
    def __init__(self):
        super().__init__()
        self.add_cliam_buffer = []
        self.setup_ui()
    
    def setup_ui(self):
        #
        self.claim_scroll = QScrollArea()
        self.claim_scroll.setWidgetResizable(True)
        self.claim_scroll.setContentsMargins(0, 0, 0, 0)
        self.claim_scroll.setStyleSheet("background-color: #1a202c; border: 1px solid #4a5568; border-radius: 16px;")

        scroll_item_container = QWidget()
        scroll_item_container.setContentsMargins(0, 0, 0, 0)
        scroll_item_container.setStyleSheet("background-color: transparent; border: none; border-radius: 0px;")

        self.claim_scroll_layout = QVBoxLayout()
        self.claim_scroll_layout.setContentsMargins(15, 20, 15, 20)
        self.claim_scroll_layout.setSpacing(20)
        self.claim_scroll_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self.claim_scroll.setWidget(scroll_item_container)
        scroll_item_container.setLayout(self.claim_scroll_layout)
        #
        self.clear_button = QPushButton("Clear")
        self.clear_button.setStyleSheet("""
            QPushButton {
                background-color: #4299e1;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3182ce;
            }
        """)
        self.clear_button.clicked.connect(self.clear_all_item)
        #
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self.claim_scroll, 1)
        layout.addWidget(self.clear_button, 1)

        self.setLayout(layout)
        self.setContentsMargins(0, 0, 0, 0)
        self.setStyleSheet("background-color: transparent; border: none; border-radius: 0px;")
        
        self.draw_later_timer = QTimer(self)
        self.draw_later_timer.timeout.connect(self._timer_add_item)
        self.draw_later_timer.start(1000)

    def clear_all_item(self):
        self.draw_later_timer.stop()
        self.add_cliam_buffer.clear()
        core.vector_database.clear()
        del_item_amount = self.claim_scroll_layout.count()
        for idx in range(del_item_amount-1, -1, -1):
            del_item = self.claim_scroll_layout.takeAt(idx)
            if del_item:
                w = del_item.widget()
                if w:
                    w.setParent(None)
                    w.deleteLater()
                l = del_item.layout()
                if l:
                    l.setParent(None)
                    l.deleteLater()
        self.claim_scroll_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))
        self.draw_later_timer.start()
    
    def add_item_later(self, claim:str, citation:Optional[list], source_document:Optional[str]=None, 
                 page_id:Optional[str]=None, chunk_id:Optional[str]=None, claim_id:Optional[str]=None,
                 ):
        self.add_cliam_buffer.append({
            "claim" : claim, 
            "citation" : citation,
            "source_document" : source_document,
            "page_id" : page_id,
            "chunk_id" : chunk_id,
            "claim_id" : claim_id
            })
    
    def _timer_add_item(self):
        while len(self.add_cliam_buffer) > 0:
            new_item_para = self.add_cliam_buffer.pop(0)
            new_item = ClaimCitationItem(
                claim=new_item_para["claim"],
                citation=new_item_para["citation"],
                source_document=new_item_para["source_document"],
                page_id=new_item_para["page_id"],
                chunk_id=new_item_para["chunk_id"],
                claim_id=new_item_para["claim_id"]
            )
            self.add_item(new_item)

    def add_item(self, caim_citation_item: ClaimCitationItem):
        insert_index = self.claim_scroll_layout.count() - 1
        self.claim_scroll_layout.insertWidget(insert_index, caim_citation_item)
    
    def scroll_to_bottom(self):
        ''' Scroll claim area to bottom '''
        scrollbar = self.claim_scroll.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

# ===============================
# upload and analyze documents

class DocUploadPanel(QFrame):
    def __init__(self):
        super().__init__()
        self._files = set()
        self.setup_ui()
    
    def setup_ui(self):
        # scroll area for uploaded files
        file_scroll = QScrollArea()
        file_scroll.setWidgetResizable(True)
        file_scroll.setStyleSheet("background-color: #1a202c; border: 1px solid #4a5568; border-radius: 16px;")

        scroll_item_container = QWidget()
        scroll_item_container.setStyleSheet("background-color: transparent; border: none;")
        scroll_item_container.setContentsMargins(0,0,0,0)
        
        self.scroll_item_layout = QVBoxLayout()
        self.scroll_item_layout.setContentsMargins(15, 20, 15, 20)
        self.scroll_item_layout.setSpacing(20)
        self.scroll_item_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)) # push up spacer

        file_scroll.setWidget(scroll_item_container)
        scroll_item_container.setLayout(self.scroll_item_layout)

        analyze_layout = QVBoxLayout()
        analyze_layout.addWidget(file_scroll, 1)

        # claim citation display
        self.claim_citation_display = ClaimCitationDisplayWidget()

        # file upload area
        self.file_drop = FileDropArea(self.add_files)
        
        # main panel
        self.setStyleSheet("background-color: #1a202c; border-left: 1px solid #2d3748;")

        layout = QVBoxLayout()
        layout.setContentsMargins(15, 20, 15, 20)
        layout.setSpacing(20)
        layout.addLayout(analyze_layout)
        layout.setStretch(0, 2)
        layout.addWidget(self.claim_citation_display)
        layout.setStretch(1, 2)
        layout.addWidget(self.file_drop)
        layout.setStretch(2, 1)

        self.setLayout(layout)
    
    def _add_new_page_claim(self, page_claims:List[List[Claim]], page_chunks:List[Tuple[int, str]]):
        if len(page_claims) == 0 or len(page_chunks) == 0:
            return
        for claims in page_claims:
            for claim_num, claim in enumerate(claims):
                source_document = claim.metadata["document"]
                page_id  = claim.metadata["page"]
                chunk_id = claim.metadata["chunk_id"]
                claim_sentence = claim.claim
                claim_citations = [c_sentence for c_score, c_sentence in claim.citations] # remove score
                self.claim_citation_display.add_item_later(
                    claim=claim_sentence,
                    citation=claim_citations,
                    source_document=source_document,
                    page_id=page_id,
                    chunk_id=chunk_id,
                    claim_id=claim_num
                )

    def add_files(self, paths):
        ''' add file item into UI list and inner set '''
        for p in paths:
            if p in self._files:
                continue
            self._files.add(p)
            spacer_index = self.scroll_item_layout.count() - 1
            item = FileItemWidget(p, on_new_page_claim_added=self._add_new_page_claim, on_delete_callback=self._on_item_deleted, parent=self)
            self.scroll_item_layout.insertWidget(spacer_index, item)

    def _on_item_deleted(self, path):
        ''' call when file item deleted, remove file form UI and inner set '''
        rich.print(f"File removed : {str(path)}")
        self._files.discard(path)

class DocumentQAChat(QMainWindow):
    """Main Document QA Chat window"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setup_styles()
        
    def setup_ui(self):
        self.status_messages = []
        self.setWindowTitle("Document QA")
        self.setGeometry(100, 100, 2000, 1200)
        
        # Central widget with splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Settings panel
        self.settings_panel = SettingsPanel(clear_chat_func=lambda:self.clear_chat(False))
        splitter.addWidget(self.settings_panel)
        
        # Chat area
        chat_widget = self.create_chat_area()
        splitter.addWidget(chat_widget)
        
        # Suggestion panel
        self.doc_upload_panel = DocUploadPanel()
        self.doc_upload_panel.setMinimumWidth(450)
        splitter.addWidget(self.doc_upload_panel)
        
        # Set splitter sizes
        splitter.setSizes([250, 2000, 450])
        
        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)
        
        # Add initial messages
        self.add_initial_messages()
        
    def create_chat_area(self):
        """Create the main chat area"""
        chat_widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header
        header = QFrame()
        header.setFixedHeight(60)
        header.setStyleSheet("""
            QFrame {
                background-color: #2d3748;
                border-bottom: 1px solid #4a5568;
            }
        """)
        
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(20, 15, 20, 15)
        
        title_label = QLabel("Chat Room")
        title_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 18px;
                font-weight: bold;
                border: none;
            }
        """)
        header_layout.addWidget(title_label)
        
        header.setLayout(header_layout)
        layout.addWidget(header)
        layout.setStretch(0, 0)
        
        # Chat scroll area
        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.chat_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.chat_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #1a202c;
            }
            QScrollBar:vertical {
                background-color: #2d3748;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background-color: #4a5568;
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #718096;
            }
        """)
        
        # Container for chat messages
        self.chat_container = QWidget()
        self.chat_layout = QVBoxLayout()
        self.chat_layout.setAlignment(Qt.AlignTop)
        self.chat_layout.setSpacing(15)
        self.chat_layout.setContentsMargins(20, 20, 20, 20)
        self.chat_container.setLayout(self.chat_layout)

        self.chat_scroll.setWidget(self.chat_container)
        layout.addWidget(self.chat_scroll)
        layout.setStretch(1, 3)
        
        # Input area
        input_area = self.create_input_area()
        layout.addWidget(input_area)
        layout.setStretch(2, 1)

        input_area = self.create_input_button()
        layout.addWidget(input_area)
        
        chat_widget.setLayout(layout)
        return chat_widget
    
    def create_input_button(self):
        """Create the input area"""
        input_widget = QFrame()
        input_widget.setStyleSheet("""
            QFrame {
                background-color: #2d3748;
                border-top: 1px solid #4a5568;
            }
        """)
        
        # Button area
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(20, 15, 20, 15)
        
        # Clear button
        self.clear_button = QPushButton("Clear")
        self.clear_button.setStyleSheet("""
            QPushButton {
                background-color: #4299e1;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: bold;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #c53030;
            }
        """)
        self.clear_button.clicked.connect(self.clear_chat)
       
        # Send button
        self.send_button = QPushButton("Send")
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #4299e1;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: bold;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #c53030;
            }
        """)
        self.send_button.clicked.connect(self.send_message)

        button_layout.addWidget(self.clear_button)
        button_layout.addStretch() # middle spacer
        button_layout.addWidget(self.send_button)       
        input_widget.setLayout(button_layout)
        
        return input_widget
    
    def create_input_area(self):
        """Create the input area"""
        input_widget = QFrame()
        input_widget.setStyleSheet("""
            QFrame {
                background-color: #2d3748;
                border-top: 1px solid #4a5568;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 15, 20, 15)

        # Input text area
        self.message_input = QTextEdit()
        self.message_input.setPlaceholderText("Write your questions here...")
        self.message_input.setMinimumHeight(100)
        self.message_input.setStyleSheet("""
            QTextEdit {
                color: #ffffff;
                font-size: 18px;
                background-color: #1a202c;
                border: 1px solid #4a5568;
                border-radius: 16px;
                padding: 15px;
            }
            QTextEdit:focus {
                color: #ffffff;
                border: 2px solid #4299e1;
            }
        """)

        '''
        text_suggest_layout = QHBoxLayout()
        text_suggest_layout.addWidget(self.message_input)
        text_suggest_layout.setStretch(0,4)

        # Input suggestion area
        self.suggestion_list = SuggestionPanel()
        text_suggest_layout.addWidget(self.suggestion_list)
        text_suggest_layout.setStretch(1,1)
        '''

        layout.addWidget(self.message_input)
        input_widget.setLayout(layout)       
        
        return input_widget
    
    def setup_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a202c;
                color: #ffffff;
            }
        """)
    
    def add_message(self, message, is_human=False, citations:Optional[List] = None):
        '''Add a new message'''
        message_widget = ChatMessage(message, is_human, citations)
        self.chat_layout.addWidget(message_widget)
        
        # Auto-scroll to bottom
        QTimer.singleShot(50, self.scroll_to_bottom)
    
    def add_initial_messages(self):
        '''Initial conversation messages'''
        self.add_statue_message("Initialize System")
        QTimer().singleShot(3000, lambda: (
            self.clear_all_status_message(), self.add_message("你好！今天想聊什麼?", is_human=False, citations=[])
            ))
    
    def start_check_timer(self, input_text):
        ''' llm response check timer '''
        self.response_timer = QTimer()
        self.response_timer.timeout.connect(self.check_response)
        self.response_store = {}
        t = threading.Thread(target=get_response, daemon=True, args=(input_text, core.state_graph, self.response_store))
        t.start()
        self.response_timer.start()

    def check_response(self):
        # QTimer listening llm response
        if self.response_store and self.response_store.get("Done", False):
            self.clear_all_status_message()
            self.response_timer.stop()
            self.add_message(
                self.response_store.get("result", "ERROR"), 
                is_human=False,
                citations=self.response_store["retrieved_knowledge"]
                )
    
    def add_statue_message(self, status_text):
        # Simulate AI response
        main_frame_layout = QHBoxLayout()
        main_frame = QFrame()

        status = QLabel()
        status.setText(status_text)
        status.setStyleSheet('''
            QLabel {
                color: #ffffff;
                font-size: 16px;
                line-height: 1.5;
            }
        '''
        )

        status_layout = QVBoxLayout()
        status_bubble = QFrame()
        status_bubble.setStyleSheet('''
            QFrame {
                background-color: #4a5568;
                border-radius: 12px;
                max-width: 600px;
            }
        '''
        )
        #
        main_frame.setLayout(main_frame_layout)
        main_frame_layout.addWidget(status_bubble)
        main_frame_layout.addStretch()
        status_bubble.setLayout(status_layout)
        status_layout.addWidget(status)
        #
        self.status_messages.append(main_frame)
        self.chat_layout.addWidget(self.status_messages[-1])
    
    def clear_all_status_message(self):
        if self.status_messages:
            for msg in self.status_messages:
                try:
                    msg.deleteLater()
                except:
                    continue
            self.status_messages.clear()

    def send_message(self):
        """Send a message"""
        message_text = self.message_input.toPlainText().strip()
        if not message_text:
            return
        
        # Add human message
        self.add_message(message_text, is_human=True)
        
        # Clear input
        self.message_input.clear()

        # Add thinking display, delete later
        self.add_statue_message("Thinking...")

        # Start a QTimer task for AI response
        self.start_check_timer(message_text)
    
    def clear_chat(self, clear_memory:bool=True):
        """Clear all memory and chat messages"""
        # wipe memory
        if clear_memory:
            core.clear_short_memory()
        # Remove all message widgets
        while self.chat_layout.count():
            child = self.chat_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Re-add initial messages
        QTimer.singleShot(100, self.add_initial_messages)
    
    def scroll_to_bottom(self):
        """Scroll chat area to bottom"""
        scrollbar = self.chat_scroll.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


def main():
    app = QApplication(sys.argv)
    font = QFont("Microsoft JhengHei", 12)
    app.setFont(font)
    
    # Set application properties
    app.setApplicationName("Document QA")
    app.setApplicationVersion("1.0")
    
    # Set dark theme
    app.setStyle('Fusion')
    palette = QPalette()
    palette.setColor(QPalette.Window, Qt.GlobalColor.darkGray)
    palette.setColor(QPalette.WindowText, Qt.GlobalColor.white)
    app.setPalette(palette)
    
    # Create and show the chat interface
    chat_app = DocumentQAChat()
    chat_app.show()
    
    # Run the application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()