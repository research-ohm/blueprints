# =============================================================================
# tool_template.py  —  all reusable tools + pattern for adding new ones
#
# To add a new tool:
#   1. Copy the @tool block below, rename it, fill in docstring + body
#   2. Add it to TOOLS in agent.py
#
# env vars needed:
#   SERPER_API_KEY        — for web_search
#   INSTANCE_PUBLIC_IP    — for query_database
#   DB_DATABASE
#   DB_USER
#   DB_PASS
#   DB_PORT               — optional, defaults to 1433
# =============================================================================

import os
import requests
import pytds
import pandas as pd
from langchain_core.tools import tool


# ── Web search (Serper.dev) ───────────────────────────────────────────────────

@tool
def web_search(query: str, num_results: int = 5) -> str:
    """
    Search the web for current information using Google (via Serper).
    Use for recent events, facts you don't know, prices, news, or anything time-sensitive.
    Input should be a concise search query string.
    """
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        return "Error: SERPER_API_KEY not set."

    try:
        response = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": num_results},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        return f"Error: {e}"

    output = []
    if answer := data.get("answerBox", {}).get("answer"):
        output.append(f"Answer: {answer}")
    elif snippet := data.get("answerBox", {}).get("snippet"):
        output.append(f"Answer: {snippet}")
    if desc := data.get("knowledgeGraph", {}).get("description"):
        output.append(f"Summary: {desc}")
    for r in data.get("organic", [])[:num_results]:
        output.append(f"[{r.get('title','')}]\n{r.get('snippet','')}\n{r.get('link','')}")

    return "\n\n".join(output) if output else "No results found."


# ── SQL Server query (pytds) ──────────────────────────────────────────────────

@tool
def query_database(sql: str) -> str:
    """
    Execute a read-only SQL query against the SQL Server database and return results.
    Use for any question that requires looking up data, counts, aggregations, or records.
    Input must be a valid T-SQL SELECT statement — do NOT use INSERT, UPDATE, or DELETE.
    Returns results as a markdown table (up to 50 rows).
    """
    normalized = sql.strip().upper()
    for forbidden in ("INSERT", "UPDATE", "DELETE", "DROP", "TRUNCATE", "ALTER", "EXEC"):
        if normalized.startswith(forbidden) or f" {forbidden} " in normalized:
            return "Error: only SELECT queries are permitted."

    try:
        with pytds.connect(
            server=os.environ["INSTANCE_PUBLIC_IP"],
            database=os.environ["DB_DATABASE"],
            user=os.environ["DB_USER"],
            password=os.environ["DB_PASS"],
            port=int(os.getenv("DB_PORT", 1433)),
            timeout=300,
        ) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            cols = [c[0] for c in cursor.description]
            df = pd.DataFrame(cursor.fetchall(), columns=cols)

        if df.empty:
            return "Query returned no rows."
        note = f"\n_(showing 50 of {len(df)} rows)_" if len(df) > 50 else ""
        return df.head(50).to_markdown(index=False) + note

    except Exception as e:
        return f"SQL error: {e}"


# ── Template for new tools ────────────────────────────────────────────────────

@tool
def my_tool(param: str, count: int = 5) -> str:
    """
    One line: what this does.
    When to use it vs when NOT to.
    Inputs should be [describe valid values].
    """
    # Your logic here. Always return a string. On error: return f"Error: {e}"
    return f"result for {param!r} x{count}"
