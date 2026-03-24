# =============================================================================
# tool_template.py  —  copy this for every new tool
# =============================================================================
# Steps:
#   1. Copy this file → my_tool.py
#   2. Fill in the function name, docstring, params, and body
#   3. Import and add to TOOLS in agent.py
# =============================================================================

from langchain_core.tools import tool


@tool
def my_tool(param: str, count: int = 5) -> str:
    """
    One line: what this does.
    Second line (optional): when to use it vs when NOT to.
    Inputs should be [describe valid values].
    """
    # Your logic here. Always return a string.
    # On error: return f"Error: {e}"  — never raise.
    return f"result for {param!r} x{count}"


# ── Structured input variant (use when args are complex) ──────────────────────
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

class SearchInput(BaseModel):
    query: str = Field(description="Search query string")
    max_results: int = Field(default=5, ge=1, le=20)

def _search(query: str, max_results: int) -> str:
    # implementation
    return f"results for {query}"

search_tool = StructuredTool.from_function(
    func=_search,
    name="search",
    description="Search the web. Use for current events or facts you don't know.",
    args_schema=SearchInput,
)
