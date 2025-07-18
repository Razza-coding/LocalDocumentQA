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

empty_history_filler = AIMessage(content="No history message found.")

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
        "Your name is {AI_name}，you're a {professional_role}.\n"
        "Uses {chat_lang} to communicate with user while avoiding violate {negitive_rule}.\n"
        "Output your response to user with example format:\n{output_format}"
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

user_template = HumanMessagePromptTemplate.from_template("User/使用者:\n{raw_user_input}")

def get_user_message(input_variables: UserTemplateInputVar) -> BaseMessage:
    ''' Get message by filling placeholders '''
    return user_template.format_messages(**input_variables)

# -------------------------------
# History Message template
class ChatHistoryTemplateInputVar(TypedDict):
    chat_history: str

history_template = AIMessagePromptTemplate.from_template("History Chat/對話歷史紀錄:\n{chat_history}")

def get_chat_history_message(input_variables: ChatHistoryTemplateInputVar) -> BaseMessage:
    ''' Get message by filling placeholders '''
    return history_template.format_messages(**input_variables)

# ===============================
# Chat Messages Templates

# -------------------------------
# General input template
class GeneralChatTemplateInputMessages(TypedDict):
    system_message  : SystemMessage
    history_message : Optional[AIMessage]
    user_message    : HumanMessage

general_input_template = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="system_message"),
    MessagesPlaceholder(variable_name="history_message"),
    MessagesPlaceholder(variable_name="user_message")
])

def make_general_input(input_messages: GeneralChatTemplateInputMessages) -> List[BaseMessage]:
    ''' LLM input prompt. Fills template placeholder with messages '''
    history_msg = input_messages.get("history_message", []) # possible input: None, [], message, list[message]
    if len(history_msg) == 0:
        history_msg = empty_history_filler
    auto_fill_input = GeneralChatTemplateInputMessages(
        system_message  = input_messages.get("system_message"),
        history_message = to_list_message(history_msg),
        user_message    = input_messages.get("user_message")
    )
    return general_input_template.invoke(input=auto_fill_input)

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

    system_msg = get_system_message(SystemTemplateInputVar(
        AI_name="JOHN",
        professional_role="Math Teacher"
    ))
    CLI_print("Prompt Testing", system_msg, "Make System Message")
    CLI_next()
    
    user_msg = get_user_message(UserTemplateInputVar(
        raw_user_input="Hello World"
    ))
    CLI_print("Prompt Testing", user_msg, "Make User Message")
    CLI_next()

    history_msg = get_chat_history_message(ChatHistoryTemplateInputVar(
        chat_history="沒有紀錄"
    ))
    CLI_print("Prompt Testing", history_msg, "Make History Message")
    CLI_next()

    LLM_input = make_general_input(GeneralChatTemplateInputMessages(
        system_message= system_msg,
        user_message= user_msg
    ))
    CLI_print("Prompt Testing", LLM_input, "Make LLM input")
    CLI_next()

    RAG_summary_prompt = make_RAG_summary_prompt(RAGSummaryChatTemplateInputMessages(
        user_input_message=user_msg,
        AI_response_message=to_list_message(AIMessage("I don't know"))
    ))
    CLI_print("Prompt Testing", RAG_summary_prompt, "Make RAG Summary prompt")
    CLI_next()

    CLI_print("Prompt Testing", "End")