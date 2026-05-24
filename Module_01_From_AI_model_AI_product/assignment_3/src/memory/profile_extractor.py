import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import EXTRACTOR_MODEL, NEBIUS_API_KEY, NEBIUS_BASE_URL


EXTRACTION_SYSTEM_PROMPT = """
You are a silent background process that extracts structured user profile information from a conversation.

Given the conversation history, extract the following fields:
- name: The user's name if they mentioned it. Otherwise null.
- preferences: A JSON list of strings describing how the user likes to receive information (e.g. "prefers concise summaries", "likes percentage breakdowns"). Empty list if none found.
- frequent_topics: A JSON list of dataset categories or intents the user has expressed interest in or asked about.
  These are values like: ACCOUNT, ORDER, REFUND, CANCEL, SHIPPING, cancel_order, get_refund, track_refund, etc.
  Include BOTH category names (uppercase) AND intent names (snake_case) if found.
  If the user said "I'm interested in refund-related intents", extract: ["REFUND", "get_refund", "check_refund_policy", "track_refund"].
  If tools were called and returned results about specific intents/categories, include those too.

RULES:
- Only extract facts the user explicitly stated, implied, or that appeared in tool results during the conversation.
- Return ONLY a valid JSON object with exactly these three keys: name, preferences, frequent_topics.
- No markdown, no explanation, no extra text. Just the raw JSON object.

Example output:
{"name": "Alice", "preferences": ["prefers bullet points"], "frequent_topics": ["REFUND", "get_refund", "track_refund"]}
"""


def extract_profile_from_history(messages: list) -> dict | None:
    """
    Runs a lightweight LLM extraction pass over the message history.
    Returns a profile dict or None if extraction fails.
    """
    llm = ChatOpenAI(
        model=EXTRACTOR_MODEL,
        openai_api_key=NEBIUS_API_KEY,
        openai_api_base=NEBIUS_BASE_URL,
        temperature=0.0,
        max_tokens=1024
    )

    # Serialize conversation to plain text for the extraction prompt
    conversation_text = ""
    for msg in messages:
        role = type(msg).__name__.replace("Message", "")
        conversation_text += f"{role}: {msg.content}\n"

    extraction_messages = [
        SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(content=f"Here is the conversation:\n\n{conversation_text}")
    ]

    try:
        response = llm.invoke(extraction_messages)
        profile = json.loads(response.content.strip())

        # Validate expected keys are present
        if not isinstance(profile, dict):
            return None
        profile.setdefault("name", None)
        profile.setdefault("preferences", [])
        profile.setdefault("frequent_topics", [])
        return profile

    except (json.JSONDecodeError, Exception):
        return None