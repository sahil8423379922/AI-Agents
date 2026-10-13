from typing import Annotated
from langchain_core.tools import tool
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langchain_core.tools import tool
from langgraph.types import Command, interrupt
from langchain_groq import ChatGroq
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage
import os
from langchain_groq import ChatGroq
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGSMITH_PROJECT"]="Log Tracing of LLM"
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_TRACING"]="true"

llm = ChatGroq(model ="qwen/qwen3.8-27b")

class State(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

def make_tool_graph():
    @tool
    def add(a:float,b:float):
        """
        Add two numbers
        """

        return a+b
    tools=[add]
    tool_node =ToolNode([add])
    llm_with_tools = llm.bind_tools([add])

    def call_llm_node(state:State):
        return {"messages":[llm_with_tools.invoke(state['messages'])]}

    
    builder = StateGraph(State)
    builder.add_node("tool_calling_llm",call_llm_node)
    builder.add_node("tools",ToolNode(tools))
    builder.add_edge(START,"tool_calling_llm")
    builder.add_conditional_edges(
        "tool_calling_llm",tools_condition

    )
    builder.add_edge("tools","tool_calling_llm")
    graph = builder.compile()

    return graph

tool_agent = make_tool_graph()




