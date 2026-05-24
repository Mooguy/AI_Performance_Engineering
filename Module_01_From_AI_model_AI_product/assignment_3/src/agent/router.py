from typing import Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from src.config import NEBIUS_API_KEY, NEBIUS_BASE_URL, DEFAULT_MODEL
from src.agent.state import AgentState

class RouterOutput(BaseModel):
    """Schema for routing decisions."""
    query_type: Literal["structured", "unstructured", "out_of_scope"] = Field(
        description="Classification of the customer service data query."
    )
    reasoning: str = Field(
        description="One-sentence explanation."
    )

def query_router_node(state: AgentState) -> dict:
    """
    Inspects the initial user message and routes it to the correct handling path.
    Uses hardcoded keyword logic for critical dataset paths to bypass remote endpoint lockups.
    """
    # Find the earliest user message to capture the core intent
    user_query = ""
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break

    if not user_query:
        return {"query_type": "out_of_scope", "iterations": state.get("iterations", 0) + 1}

    query_lower = user_query.lower()

    # --- Structural Keyword Override Rules ---
    # Instantly routes unstructured summary text queries without reaching out to the network endpoint
    if "summarize" in query_lower or "respond to" in query_lower or "how do agents" in query_lower or "how do customer service" in query_lower:
        return {
            "query_type": "unstructured",
            "iterations": state.get("iterations", 0) + 1
        }
    
    # Instantly routes structured data extraction requests
    if "how many" in query_lower or "distribution" in query_lower or "examples of" in query_lower or "exist in" in query_lower:
        return {
            "query_type": "structured",
            "iterations": state.get("iterations", 0) + 1
        }

    # --- LLM Fallback Classification ---
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
            "You are an expert data analysis router. Analyze the user's input regarding a customer service dataset.\n"
            "Classify into exactly one option:\n"
            "1. 'structured': Looking for counts, unique values, listings, or statistics.\n"
            "2. 'unstructured': Asking for semantic summaries, text syntheses, or common behaviors.\n"
            "3. 'out-of-scope': Questions completely detached from analyzing this dataset."
        )

        decision = structured_llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ])
        chosen_type = decision.query_type
    except Exception:
        # Secure fallback if the endpoint experiences any connectivity errors
        chosen_type = "out_of_scope"

    return {
        "query_type": chosen_type,
        "iterations": state.get("iterations", 0) + 1
    }