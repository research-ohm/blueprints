# =============================================================================
# agent_with_memory.py  —  ReACT agent + PGVector semantic memory
#
# Two kinds of memory wired together:
#   1. Conversation history  — LangGraph checkpointer (PostgresSaver)
#   2. Semantic / long-term memory  — PGVector
#
# Setup:
#   pip install langgraph langchain-openai langchain-postgres \
#               langchain-community psycopg2-binary
#
#   CREATE DATABASE agent_memory;  -- in psql
#
#   export OPENAI_API_KEY=...
#   export PG_URI=postgresql://user:pass@localhost:5432/agent_memory
# =============================================================================

import os
from typing import Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_postgres import PGVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.postgres import PostgresSaver
from typing_extensions import TypedDict

PG_URI = os.environ["PG_URI"]

# ── Embeddings + Vector store ────────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # 384-dim

vector_store = PGVector(
    embeddings=embeddings,
    collection_name="agent_memory",
    connection=PG_URI,
    use_jsonb=True,
)

# ── Memory helpers ───────────────────────────────────────────────────────────
def remember(text: str, metadata: dict = {}):
    vector_store.add_texts([text], metadatas=[metadata])

def recall(query: str, k: int = 3) -> str:
    docs = vector_store.similarity_search(query, k=k)
    if not docs:
        return ""
    return "\n".join(f"- {d.page_content}" for d in docs)

# ── State ────────────────────────────────────────────────────────────────────
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ── Tools ────────────────────────────────────────────────────────────────────
@tool
def save_memory(fact: str) -> str:
    """Save an important fact about the user or conversation to long-term memory."""
    remember(fact)
    return f"Saved: {fact}"

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    try:
        import math
        return str(eval(expression, {"__builtins__": {}, **math.__dict__}))
    except Exception as e:
        return f"Error: {e}"

TOOLS = [calculator, save_memory]

# ── Model ────────────────────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=2048)
model = llm.bind_tools(TOOLS)

# ── Nodes ────────────────────────────────────────────────────────────────────
def agent_node(state: State):
    last_user_msg = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    )
    memories = recall(last_user_msg)
    memory_block = f"\n\nRelevant memory:\n{memories}" if memories else ""

    system = (
        "You are a helpful assistant with long-term memory. "
        "Use the save_memory tool to remember important facts the user tells you."
        + memory_block
    )

    response = model.invoke([SystemMessage(system)] + state["messages"])
    return {"messages": [response]}

def should_continue(state: State):
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

# ── Graph ────────────────────────────────────────────────────────────────────
checkpointer = PostgresSaver.from_conn_string(PG_URI)
checkpointer.setup()

graph = (
    StateGraph(State)
    .add_node("agent", agent_node)
    .add_node("tools", ToolNode(TOOLS))
    .add_edge(START, "agent")
    .add_conditional_edges("agent", should_continue, ["tools", END])
    .add_edge("tools", "agent")
    .compile(checkpointer=checkpointer)
)

# ── Multi-turn REPL ──────────────────────────────────────────────────────────
def chat(message: str, thread_id: str = "default") -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"messages": [HumanMessage(message)]}, config=config)
    return result["messages"][-1].content

if __name__ == "__main__":
    tid = "session-001"
    print("Chat with your agent. Ctrl+C to quit.\n")
    while True:
        user = input("You: ").strip()
        if not user:
            continue
        print(f"Agent: {chat(user, tid)}\n")
