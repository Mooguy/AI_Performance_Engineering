import json
import re
import sqlite3
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from src.config import NEBIUS_API_KEY, NEBIUS_BASE_URL, DEFAULT_MODEL
from src.agent.state import AgentState
from src.agent.router import query_router_node
from src.tools.base_tools import (
    get_unique_values, filter_dataset, get_row_count,
    get_sample_examples, get_value_distribution
)
from src.memory.profile_store import (
    init_profile_table, get_profile,
    upsert_profile, format_profile_for_prompt
)
from src.memory.profile_extractor import extract_profile_from_history


MAX_ITERATIONS = 12

TOOL_MAP = {
    "get_unique_values":    get_unique_values.func,
    "filter_dataset":       filter_dataset.func,
    "get_row_count":        get_row_count.func,
    "get_sample_examples":  get_sample_examples.func,
    "get_value_distribution": get_value_distribution.func
}

# --- Persistent SQLite connection (shared across checkpointer + profiles) ---
db_conn = sqlite3.connect("agent_memory.db", check_same_thread=False)

# Ensure the user_profiles table exists on startup
init_profile_table(db_conn)


# --- Nodes ---

def react_agent_node(state: AgentState) -> dict:
    llm = ChatOpenAI(
        model=DEFAULT_MODEL,
        openai_api_key=NEBIUS_API_KEY,
        openai_api_base=NEBIUS_BASE_URL,
        temperature=0.0,
        max_tokens=1024
    )

    # Read the current session's profile and inject into system prompt
    session_id = state.get("session_id") or ""
    profile = get_profile(db_conn, session_id) if session_id else None
    profile_context = format_profile_for_prompt(profile)

    system_prompt = (
        "You are an expert Customer Service Data Analyst Agent operating on the Bitext dataset.\n"
        "Columns available: [flags, instruction, category, intent, response].\n\n"
        "Available tools:\n"
        "- get_unique_values(column_name: str)\n"
        "- get_row_count(category: str = None, intent: str = None)\n"
        "- get_sample_examples(category: str = None, intent: str = None, n_examples: int = 5)\n"
        "- get_value_distribution(group_by_column: str, filter_column: str = None, filter_value: str = None)\n\n"
        "Reason step by step.\n"
        "If you need a tool, output exactly one JSON block in markdown fences and nothing else:\n"
        "```json\n"
        "{\n"
        "  \"thought\": \"why this tool is needed\",\n"
        "  \"tool\": \"tool_name\",\n"
        "  \"args\": {\"param_name\": \"value\"}\n"
        "}\n"
        "```\n\n"
        "If you have enough information, output exactly:\n"
        "```json\n"
        "{\n"
        "  \"thought\": \"final reasoning\",\n"
        "  \"final_answer\": \"clear answer to the user\"\n"
        "}\n"
        "```\n\n"
        "Rules:\n"
        "- Use tools whenever the answer depends on dataset contents.\n"
        "- Do not invent category names, intent names, counts, examples, or distributions.\n"
        "- If you are not sure about the exact category or intent naming, call get_unique_values first.\n"
        "- Use get_row_count for counts.\n"
        "- Use get_value_distribution for distributions or breakdowns.\n"
        "- Use get_sample_examples when the user asks for examples.\n"
        "- If the user asks for examples, lists, or items, return the actual items from the tool output in final_answer rather than summarizing them.\n"
        "- Summarize only when the user explicitly asks for a summary, trend, or high-level explanation.\n"
        + profile_context
    )
    

    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
    response = llm.invoke(messages)

    return {
        "messages": [response],
        "iterations": state.get("iterations", 0) + 1
    }


def execution_tools_node(state: AgentState) -> dict:
    last_msg = state["messages"][-1]
    content = last_msg.content if last_msg.content is not None else ""
    json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
    text_to_parse = json_match.group(1) if json_match else content

    try:
        data = json.loads(text_to_parse.strip())
        tool_name = data.get("tool")
        tool_args = data.get("args", {})
        if tool_name in TOOL_MAP:
            observation = TOOL_MAP[tool_name](**tool_args)
            result_str = json.dumps(observation)
        else:
            result_str = f"Error: Tool '{tool_name}' is not registered."
    except Exception as e:
        result_str = f"Error: Failed parsing tool syntax. Details: {str(e)}"

    tool_msg = ToolMessage(content=result_str, tool_call_id="call_id_placeholder")
    return {"messages": [tool_msg]}


def profile_extraction_node(state: AgentState) -> dict:
    import traceback
    try:
        session_id = state.get("session_id") or ""
        if not session_id:
            return {}

        valid_messages = [m for m in state["messages"] if m.content is not None]
        if not valid_messages:
            return {}

        extracted = extract_profile_from_history(valid_messages)
        if not extracted:
            return {}

        existing = get_profile(db_conn, session_id) or {}

        merged = {
            "name": extracted.get("name") or existing.get("name"),
            "preferences": list(set(
                existing.get("preferences", []) + extracted.get("preferences", [])
            )),
            "frequent_topics": list(set(
                existing.get("frequent_topics", []) + extracted.get("frequent_topics", [])
            ))
        }

        upsert_profile(db_conn, session_id, merged)
        return {}

    except Exception:
        traceback.print_exc()
        return {}


def handle_out_of_scope_node(state: AgentState) -> dict:
    decline_message = AIMessage(
        content="I'm sorry, but I can only assist with analysis, summaries, or questions directly related to the customer service dataset."
    )
    return {"messages": [decline_message]}


def handle_fallback_node(state: AgentState) -> dict:
    fallback_message = AIMessage(
        content="I have exceeded the maximum calculation loops trying to resolve your query."
    )
    return {"messages": [fallback_message]}


# --- Routing ---

def route_after_classification(
    state: AgentState
) -> Literal["call_model", "profile_update", "memory_query", "out_of_scope"]:
    query_type = state.get("query_type")

    if query_type == "out_of_scope":
        return "out_of_scope"
    if query_type == "profile_update":
        return "profile_update"
    if query_type == "memory_query":
        return "memory_query"
    return "call_model"


def route_after_model(state: AgentState) -> Literal["tools", "fallback", "extract_profile"]:
    if state.get("iterations", 0) >= MAX_ITERATIONS:
        return "fallback"
    
    last_msg = state["messages"][-1]
    content = last_msg.content if last_msg.content is not None else ""
    
    if '"final_answer"' in content:
        return "extract_profile"
    if '"tool"' in content:
        return "tools"
    
    return "extract_profile"

def handle_profile_update_node(state: AgentState) -> dict:
    return {
        "messages": [
            AIMessage(
                content="Thanks — I’ve noted that about you and will use it to personalize future responses when relevant."
            )
        ]
    }


def handle_memory_query_node(state: AgentState) -> dict:
    session_id = state.get("session_id") or ""
    if not session_id:
        return {
            "messages": [
                AIMessage(content="I don’t have a session ID, so I can’t retrieve anything I know about you yet.")
            ]
        }

    profile = get_profile(db_conn, session_id)
    if not profile:
        return {
            "messages": [
                AIMessage(content="I don’t have any saved personal details about you yet.")
            ]
        }

    parts = []
    if profile.get("name"):
        parts.append(f"Your name appears to be {profile['name']}.")
    if profile.get("preferences"):
        parts.append("Your preferences include: " + ", ".join(profile["preferences"]) + ".")
    if profile.get("frequent_topics"):
        parts.append("Topics you've discussed include: " + ", ".join(profile["frequent_topics"]) + ".")

    content = " ".join(parts) if parts else "I don’t have any saved personal details about you yet."
    return {"messages": [AIMessage(content=content)]}

# --- Graph Assembly ---

workflow = StateGraph(AgentState)

workflow.add_node("router",          query_router_node)
workflow.add_node("call_model",      react_agent_node)
workflow.add_node("tools",           execution_tools_node)
workflow.add_node("extract_profile", profile_extraction_node)
workflow.add_node("out_of_scope",    handle_out_of_scope_node)
workflow.add_node("fallback",        handle_fallback_node)
workflow.add_node("profile_update", handle_profile_update_node)
workflow.add_node("memory_query", handle_memory_query_node)

workflow.add_edge(START, "router")

workflow.add_conditional_edges(
    "router",
    route_after_classification,
    {
        "out_of_scope": "out_of_scope",
        "profile_update": "profile_update",
        "memory_query": "memory_query",
        "call_model":   "call_model"
    }
)

workflow.add_conditional_edges(
    "call_model",
    route_after_model,
    {
        "tools":           "tools",
        "fallback":        "fallback",
        "extract_profile": "extract_profile"
    }
)

workflow.add_edge("tools",           "call_model")
workflow.add_edge("profile_update", "extract_profile")
workflow.add_edge("memory_query", END)
workflow.add_edge("extract_profile", END)
workflow.add_edge("out_of_scope",    END)
workflow.add_edge("fallback",        END)

# --- Checkpointer + Compile ---

memory = SqliteSaver(db_conn)
agent_graph = workflow.compile(checkpointer=memory)