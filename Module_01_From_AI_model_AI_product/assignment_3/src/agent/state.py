from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # add_messages appends new messages to the existing list automatically
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Classification assigned by our router node
    query_type: Literal["structured", "unstructured", "out_of_scope", "unknown"]

    # Safe guard tracker for tracking infinite execution loops
    iterations: int

    # Session ID for semantic memory profile read/write
    session_id: str