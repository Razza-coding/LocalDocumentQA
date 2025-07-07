from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda

from langgraph.graph import StateGraph, END, START
from langgraph.graph import add_messages
from langgraph.graph.message import add_messages
from langgraph.types import CachePolicy
from typing_extensions import TypedDict

from langchain_community.chat_models import ChatOllama

class InputState(TypedDict):
    user_input : str

class OutputState(TypedDict):
    graph_output : str

class OveralltState(TypedDict):
    node_count : int
    user_input : str
    graph_output : str
    inner_msg : str

def node_entry(state: InputState) -> OveralltState:
    pass_dict = {
        "node_count" : 0,
        "user_input" : state.get("user_input"),
        "graph_output" : "",
        "inner_msg" : "start"
    }
    return pass_dict

def node_middle(state: OveralltState) -> OveralltState:
    pass_dict = {
        "node_count" : state["node_count"] + 1,
        "user_input" : state.get("user_input"),
        "graph_output" : "",
        "inner_msg" : state["inner_msg"] + " " + "middle"
    }
    return pass_dict

def node_output(state: OveralltState) -> OutputState:
    node_passed_count = str(state["node_count"] + 1)
    inner_msg = state["inner_msg"] + " " + "end"
    
    pass_dict = {
        "graph_output" : state["user_input"] + "\n" + node_passed_count + "\n" + inner_msg,
    }
    return pass_dict

StateBuilder = StateGraph(
    OveralltState,
    input_schema=InputState,
    output_schema=OutputState
)
StateBuilder.set_entry_point("start_node")
StateBuilder.set_finish_point("end_node")
StateBuilder.add_node("start_node", node_entry)
StateBuilder.add_node("middle_node", node_middle)
StateBuilder.add_node("end_node", node_output)
StateBuilder.add_edge("start_node", "middle_node")
StateBuilder.add_edge("middle_node", "end_node")

graph = StateBuilder.compile()
response = graph.invoke({
    "user_input" : "HelloGraph"
})
print(response["graph_output"])