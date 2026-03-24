# =============================================================================
# agent.py  —  ReACT agent template (LangGraph + Claude)
#
# Pattern:  START → agent_node → [tool_node → agent_node]* → END
#
# To run:
#   pip install langgraph langchain-anthropic langchain-core
#   export ANTHROPIC_API_KEY=...
#   python agent.py
# =============================================================================

from typing import Annotated
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict


# ── 1. State ──────────────────────────────────────────────────────────────────
# Everything the graph passes between nodes.
# add_messages = append-only (never overwrites the history).
# Add more keys here if nodes need to share extra data.

class State(TypedDict):
    messages: Annotated[list, add_messages]


# ── 2. Tools ──────────────────────────────────────────────────────────────────
# Docstring = what the LLM reads to decide when to call this tool.
# Type hints = the input schema the LLM fills in.
# Always return a string (the "observation" the agent sees next).

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. Use for any arithmetic or calculation."""
    try:
        import math
        return str(eval(expression, {"__builtins__": {}, **math.__dict__}))
    except Exception as e:
        return f"Error: {e}"


# Register all tools here. The agent sees exactly this list.
TOOLS = [calculator]


# ── 3. Model ──────────────────────────────────────────────────────────────────

llm = ChatAnthropic(model="claude-sonnet-4-5-20251022", max_tokens=2048)
model = llm.bind_tools(TOOLS)


# ── 4. Nodes ──────────────────────────────────────────────────────────────────

SYSTEM = "You are a helpful assistant. Use tools when they give a better answer."

def agent_node(state: State):
    """Calls the LLM. Returns a tool_call (→ tool_node) or plain text (→ END)."""
    response = model.invoke([SystemMessage(SYSTEM)] + state["messages"])
    return {"messages": [response]}

def should_continue(state: State):
    """Router: did the LLM request a tool? If yes loop, if no stop."""
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
