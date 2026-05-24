from typing import Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from src.config import NEBIUS_API_KEY, NEBIUS_BASE_URL, DEFAULT_MODEL
from src.agent.state import AgentState


class RouterOutput(BaseModel):
    query_type: Literal[
        "structured",
        "unstructured",
        "profile_update",
        "memory_query",
        "out_of_scope"
    ] = Field(description="Classification of the incoming user request.")
    reasoning: str = Field(description="One-sentence explanation.")


def query_router_node(state: AgentState) -> dict:
    user_query = ""
    user_query = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break

    if not user_query:
        return {
            "query_type": "out_of_scope",
            "iterations": state.get("iterations", 0) + 1
        }

    query_lower = user_query.lower().strip()

    memory_patterns = [
        "what do you know about me",
        "what do you remember about me",
        "what do you remember",
        "who am i",
        "tell me about me",
        "what have you learned about me"
    ]

    profile_starts = [
        "my name is",
        "call me",
        "i prefer",
        "i like",
        "i love",
        "my favorite",
        "please remember",
        "remember that i",
        "i usually",
        "i work as",
        "i work at"
    ]

    unstructured_patterns = [
        "summarize", "respond to", "how do agents", "how do customer service"
    ]

    structured_patterns = [
        "how many", "distribution", "examples of", "exist in",
        "list", "show me", "count", "what categories", "what intents"
    ]

    if any(p in query_lower for p in memory_patterns):
        return {
            "query_type": "memory_query",
            "iterations": state.get("iterations", 0) + 1
        }

    if any(p in query_lower for p in profile_starts):
        return {
            "query_type": "profile_update",
            "iterations": state.get("iterations", 0) + 1
        }

    if any(p in query_lower for p in unstructured_patterns):
        return {
            "query_type": "unstructured",
            "iterations": state.get("iterations", 0) + 1
        }

    if any(p in query_lower for p in structured_patterns):
        return {
            "query_type": "structured",
            "iterations": state.get("iterations", 0) + 1
        }

    try:
        llm = ChatOpenAI(
            model=DEFAULT_MODEL,
            openai_api_key=NEBIUS_API_KEY,
            openai_api_base=NEBIUS_BASE_URL,
            temperature=0.0,
            max_tokens=256
        )
        structured_llm = llm.with_structured_output(RouterOutput)

        system_prompt = (
            "You are an expert router for a customer service dataset agent.\n"
            "Classify the user's request into exactly one option:\n"
            "1. 'profile_update' = user shares identity, preferences, habits, or personal facts.\n"
            "2. 'memory_query' = user asks what you know or remember about them.\n"
            "3. 'structured' = counts, unique values, listings, examples, distributions, statistics.\n"
            "4. 'unstructured' = semantic summaries, trends, behavior analysis, text synthesis.\n"
            "5. 'out_of_scope' = unrelated to the dataset or user memory."
        )

        decision = structured_llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ])
        chosen_type = decision.query_type
    except Exception:
        chosen_type = "out_of_scope"

    return {
        "query_type": chosen_type,
        "iterations": state.get("iterations", 0) + 1
    }