from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    query_type: Literal[
        "structured",
        "unstructured",
        "profile_update",
        "memory_query",
        "out_of_scope",
        "unknown"
    ]
    iterations: int
    session_id: str