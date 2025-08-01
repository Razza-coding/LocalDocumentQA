from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, trim_messages
from langchain_core.prompts import  BaseChatPromptTemplate, MessagesPlaceholder, ChatPromptTemplate, ChatMessagePromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from typing_extensions import TypedDict
from typing import *
from CLI_Format import *

'''
Prompt Tools

 - Template Designs
    - Chat Templates    : Template that accept message as input, output List Messages
    - Message Templates : Template that accept string as input, output Message

 - Transform Tools
    - Message Transform : Only accept combined template and message
        - List Message -> Message
        - Message      -> List Messages
        - Message      -> String
    - Prompt Assemble
        - Message + Chat Template    -> List Messages
        - String  + Message Template -> Message 
'''

# ===============================
# Predefined Partial Prompt

default_chat_lang      = "繁體中文 English 日文"
defualt_negitive_rule  = "使用簡體中文"
default_output_format  = """
Write a small friendly artical.
Start by writing one sentence, inform user what you think.
And then, depend on what user ask, provide atleast one option they can choice without asking more questions.
Finally, ask more detail about the user's question if needed, or provide some option user might want to do next.
"""

# ===============================
# Predefined Message

empty_history_filler   = AIMessage(content="No history message found.")
empty_knowledge_filler = AIMessage(content="No related Knowledge or Data found.")

RAG_summary_description = SystemMessage("""
Summary dialogs into following format

Short Summary : Write short summary within one sentence to describe outline.
Tags : List possible tags or keyword about the dialogs
Detailed Summary : Based on Short Summary, describe more detail about the dialogs

Dialog content as following
""")

# ===============================
# Transform Tools
def to_message(messages: BaseMessage | List[BaseMessage]) -> BaseMessage:
    ''' Create a Message without contained by List '''
    if not messages:
        raise ValueError("Empty Message")
    if isinstance(messages, List):
        if len(messages) > 1:
            raise ValueError("Expected only one message, but got multiple.")
        else:
            return messages[0]
    if not isinstance(messages, BaseMessage):
         raise ValueError("Expected Only BaseMessage or List[BaseMessage]")
    return messages

def to_list_message(messages: BaseMessage | List[BaseMessage]) -> List[BaseMessage]:
    ''' Put Message inside List '''
    if not messages:
        raise ValueError("Empty Message")
    if isinstance(messages, List):
        if len(messages) == 0:
            raise ValueError("Empty List Message")
        if all(isinstance(m, BaseMessage) for m in messages):
            return messages
        else:
            raise ValueError("List contains non-Message item")
    if not isinstance(messages, BaseMessage):
         raise ValueError("Expected Only BaseMessage or List[BaseMessage]")
    
    return [messages]

def to_text(message: BaseMessage | List[BaseMessage]) -> str:
    ''' Truns message content into string, accept Message or List contains one Message '''
    return to_message(message).text()

# ===============================
# Message Templates

# -------------------------------
# System Message

class SystemTemplateInputVar(TypedDict):
    AI_name : str
    professional_role : str
    chat_lang     : Optional[str]
    negitive_rule : Optional[str]
    output_format : Optional[str]

system_template = SystemMessagePromptTemplate.from_template(
    (
        "<System setting>"
        "Your name is {AI_name}，you're a {professional_role}.\n"
        "Uses {chat_lang} to communicate with user while avoiding violate {negitive_rule}.\n"
        "System will give you useful imformation or knowledge that helps you reply, here are some sections:"
        " - Chat History section is record of User and AI chat bot's Messages."
        " - Additional Knowledge section is data from books, documents or facts."
        "</System setting>"
        "<Instructions>"
        "Reply your response to user with this output format:\n{output_format}\n"
        "</Instructions>"
    )
)

def get_system_message(input_variables: SystemTemplateInputVar) -> BaseMessage:
    ''' Get message by filling placeholders '''
    auto_fill_input = SystemTemplateInputVar(
        AI_name           = input_variables.get("AI_name"),
        professional_role = input_variables.get("professional_role"),
        chat_lang         = input_variables.get("chat_lang", default_chat_lang),
        negitive_rule     = input_variables.get("negitive_rule", defualt_negitive_rule),
        output_format     = input_variables.get("output_format", default_output_format),
    )
    return system_template.format_messages(**auto_fill_input)

# -------------------------------
# User Message
class UserTemplateInputVar(TypedDict):
    raw_user_input: str

user_template = HumanMessagePromptTemplate.from_template("<User>:\n{raw_user_input}")

def get_user_message(input_variables: UserTemplateInputVar) -> BaseMessage:
    ''' Get message by filling placeholders '''
    return user_template.format_messages(**input_variables)

# -------------------------------
# Translate Request template
class TranslateTemplateInputVar(TypedDict):
    translate_lang: str
    input_text: str

translate_template = AIMessagePromptTemplate.from_template(
'''
You are a Translate Expert. Translate Input Text into {translate_lang} and output translate result in JSON Format with 1 group. Do not add any explanation or extra text.

Format Example:
{{
    "orignal_text" : <put input text here>,
    "translate_result" : <put translate result here>
}}

Input Text:
{input_text}''')

def get_translate_request_message(intput_variables: TranslateTemplateInputVar) -> BaseMessage:
    ''' Make A simple translation quest prompt, translate input text into target langue '''
    return translate_template.format_messages(**intput_variables)

# -------------------------------
# Translate Verify template
class TranslateVerifyTemplateInputVar(TypedDict):
    trans_lang: str
    orignal_text: str
    translate_text: str

translate_verify_template = AIMessagePromptTemplate.from_template(
'''
Answer in {{ "verify_result" : [YES/NO] }} format. Is this {trans_lang} translation correct?
{{
    "Orignal Text" : "{orignal_text}",
    "Translate Text" : "{translate_text}",
}}
'''
)

def get_translate_verify_message(intput_variables: TranslateVerifyTemplateInputVar) -> BaseMessage:
    ''' Make Verify prompt for translation quest '''
    intput_variables["translate_text"] = intput_variables.get("translate_text", "")
    return translate_verify_template.format_messages(**intput_variables)


# ===============================
# Chat Messages Templates

# -------------------------------
# History Message template
class ChatHistoryTemplateInputMessages(TypedDict):
    chat_history: List[BaseMessage]

history_template = AIMessagePromptTemplate.from_template("<History Chat>\n{chat_history_chunk}</History Chat>")

def make_chat_history_prompt(input_variables: ChatHistoryTemplateInputMessages) -> BaseMessage:
    ''' Creates Chat History Section for part of input prompt '''
    chat_history = input_variables.get("chat_history", [])
    if not chat_history:
        chat_history = to_list_message(empty_history_filler)
    assert chat_history is not None
    
    # assamble individual message into a string chunk
    chat_history_chunk = ""
    msg_frame ="{m_type} : {m_content}\n"
    for msg in chat_history:
        chat_history_chunk += msg_frame.format(m_type=msg.type, m_content=msg.text())
    
    return history_template.format_messages(**{"chat_history_chunk" : chat_history_chunk})

# -------------------------------
# General input template
class GeneralChatTemplateInputMessages(TypedDict):
    system_message    : SystemMessage
    history_message   : Optional[AIMessage]
    knowledge_message : Optional[AIMessage]
    user_message      : HumanMessage

general_input_template = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="system_message"),
    MessagesPlaceholder(variable_name="history_message"),
    MessagesPlaceholder(variable_name="knowledge_message"),
    MessagesPlaceholder(variable_name="user_message")
])

def make_general_input(input_messages: GeneralChatTemplateInputMessages) -> List[BaseMessage]:
    ''' LLM input prompt. Fills template placeholder with messages '''
    # fills empty optional message, possible input: None, [], message, list[message]
    history_msg = input_messages.get("history_message", [])
    knowledge_msg = input_messages.get("knowledge_message", [])
    if len(history_msg) == 0: 
        history_msg = empty_history_filler
    if len(knowledge_msg) == 0: 
        knowledge_msg = empty_knowledge_filler

    # assamble
    auto_fill_input = GeneralChatTemplateInputMessages(
        system_message  = input_messages.get("system_message"),
        history_message   = to_list_message(history_msg),
        knowledge_message = to_list_message(knowledge_msg),
        user_message    = input_messages.get("user_message")
    )
    return general_input_template.invoke(input=auto_fill_input)

# -------------------------------
# Additional Knowledge template
class KnowledgeTemplateInputMessages(TypedDict):
    knowlegde_messages: List[AIMessage]

knowledge_template = AIMessagePromptTemplate.from_template("<Additional Knowledge>\n{all_knowledge_messages}<\Additional Knowledge>")

def get_knowledge_message(input_variables: KnowledgeTemplateInputMessages) -> BaseMessage:
    ''' To be added '''
    assert input_variables["knowlegde_messages"] is not None, "List like object is expected"
    if len(input_variables["knowlegde_messages"]) == 0:
        input_variables["knowlegde_messages"] = [empty_knowledge_filler]

    # Combine all knowledge
    message_frame = "\t<knowledge {knowledge_idx}> {knowledge_content}\n"
    all_knowledge_message = ""
    for idx, msg in enumerate(input_variables["knowlegde_messages"]):
        all_knowledge_message += message_frame.format(knowledge_idx=idx+1, knowledge_content=msg.text())

    # Assamble
    return knowledge_template.format_messages(**{"all_knowledge_messages" : all_knowledge_message})

# -------------------------------
# RAG Summary template
class RAGSummaryChatTemplateInputMessages(TypedDict):
    user_input_message  : HumanMessage
    AI_response_message : AIMessage

RAG_summary_template = ChatPromptTemplate.from_messages([
    RAG_summary_description,
    MessagesPlaceholder(variable_name="user_input_message"),
    MessagesPlaceholder(variable_name="AI_response_message"),
])

def make_RAG_summary_prompt(input_messages: RAGSummaryChatTemplateInputMessages):
    ''' LLM input prompt. Fills template placeholder with messages '''
    return RAG_summary_template.invoke(input=input_messages)

if __name__ == "__main__":
    ''' Test All Template '''
    CLI_print("Prompt Testing", "Start")
    CLI_next()

    # test system message
    system_msg = get_system_message(SystemTemplateInputVar(
        AI_name="JOHN",
        professional_role="Math Teacher"
    ))
    CLI_print("Prompt Testing", system_msg, "Make System Message")
    CLI_next()
    # test user message
    user_msg = get_user_message(UserTemplateInputVar(
        raw_user_input="Hello World"
    ))
    CLI_print("Prompt Testing", user_msg, "Make User Message")
    CLI_next()
    # test history message
    test_history_msg = [AIMessage("Initialize system"), HumanMessage("Hello"), AIMessage("Hello, what can I help?")]
    history_msg = make_chat_history_prompt(ChatHistoryTemplateInputMessages(
        chat_history=test_history_msg
    ))
    CLI_print("Prompt Testing", history_msg, "Make History Message")
    CLI_next()
    # test general input
    LLM_input = make_general_input(GeneralChatTemplateInputMessages(
        system_message= system_msg,
        user_message= user_msg
    ))
    CLI_print("Prompt Testing", LLM_input, "Make LLM input")
    CLI_next()
    # test RAG message
    RAG_summary_prompt = make_RAG_summary_prompt(RAGSummaryChatTemplateInputMessages(
        user_input_message=user_msg,
        AI_response_message=to_list_message(AIMessage("I don't know"))
    ))
    CLI_print("Prompt Testing", RAG_summary_prompt, "Make RAG Summary prompt")
    CLI_next()
    # test translation request message
    translation_request_prompt = get_translate_request_message(TranslateTemplateInputVar(
        translate_lang="English",
        input_text="這是一個測試訊息"
    ))
    CLI_print("Translate Testing", translation_request_prompt, "Make translate Request")
    CLI_next()
    # test knowledge message
    knowledge_messages = get_knowledge_message(KnowledgeTemplateInputMessages(
        knowlegde_messages=[AIMessage("1 + 1 = 2"), AIMessage("2 + 2 = 4"), AIMessage("4 + 4 = 8")]
    ))
    CLI_print("Knowledge Messages", knowledge_messages, "Display Expanded knowledge")
    CLI_next()


    CLI_print("Prompt Testing", "End")