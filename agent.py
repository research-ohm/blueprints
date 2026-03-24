# =============================================================================
# agent.py  —  ReACT agent template (LangGraph + OpenAI)
#
# Pattern:  START → agent_node → [tool_node → agent_node]* → END
#
# To run:
#   pip install langgraph langchain-openai langchain-core
#   export OPENAI_API_KEY=...
#   python agent.py
# =============================================================================

from typing import Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

# ── 1. State ──────────────────────────────────────────────────────────────────
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ── 2. Tools ──────────────────────────────────────────────────────────────────
@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. Use for any arithmetic or calculation."""
    try:
        import math
        return str(eval(expression, {"__builtins__": {}, **math.__dict__}))
    except Exception as e:
        return f"Error: {e}"

TOOLS = [calculator]

# ── 3. Model ──────────────────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=2048)
model = llm.bind_tools(TOOLS)

# ── 4. Nodes ──────────────────────────────────────────────────────────────────
SYSTEM = "You are a helpful assistant. Use tools when they give a better answer."

def agent_node(state: State):
    response = model.invoke([SystemMessage(SYSTEM)] + state["messages"])
    return {"messages": [response]}

def should_continue(state: State):
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

# ── 5. Graph ──────────────────────────────────────────────────────────────────
graph = (
    StateGraph(State)
    .add_node("agent", agent_node)
    .add_node("tools", ToolNode(TOOLS))
    .add_edge(START, "agent")
    .add_conditional_edges("agent", should_continue, ["tools", END])
    .add_edge("tools", "agent")
    .compile()
)

# ── 6. Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = graph.invoke({"messages": [HumanMessage("What is 1337 * 42?")]})
    print(result["messages"][-1].content)
