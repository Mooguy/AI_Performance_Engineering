import os
import sys
import json
import re
import streamlit as st
from langchain_core.messages import HumanMessage

root_dir = os.path.abspath(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.agent.graph import agent_graph

st.set_page_config(page_title="Customer Service Agent", layout="wide")
st.title("Customer Service Data Analyst")
st.caption("Streamlit UI for your ReAct agent")

if "sessions" not in st.session_state:
    st.session_state.sessions = {}

with st.sidebar:
    st.header("Session")
    session_id = st.text_input("Session ID", value="session_1")
    if session_id not in st.session_state.sessions:
        st.session_state.sessions[session_id] = []
    if st.button("Clear current session"):
        st.session_state.sessions[session_id] = []
        st.rerun()

chat_history = st.session_state.sessions[session_id]

for msg in chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("steps"):
            with st.expander("Reasoning steps"):
                for step in msg["steps"]:
                    st.markdown(step)

def run_agent(user_input: str, session_id: str):
    config = {"configurable": {"thread_id": session_id}}
    steps = []
    final_answer = None

    stream_events = agent_graph.stream(
        {
            "messages": [HumanMessage(content=user_input)],
            "iterations": 0,
            "session_id": session_id,
        },
        config=config,
        stream_mode="updates",
    )

    for event in stream_events:
        for node_name, node_output in event.items():
            if node_output is None:
                continue

            messages = node_output.get("messages", [])
            if not messages:
                continue

            last_msg = messages[-1]
            if last_msg is None:
                continue

            if node_name == "call_model":
                content = getattr(last_msg, "content", None)
                if not content:
                    continue

                content = content.strip()

                if content.startswith("```json"):
                    content = re.sub(r"^```json\s*", "", content, flags=re.DOTALL)
                    content = re.sub(r"\s*```$", "", content, flags=re.DOTALL)
                elif content.startswith("```"):
                    content = re.sub(r"^```\s*", "", content, flags=re.DOTALL)
                    content = re.sub(r"\s*```$", "", content, flags=re.DOTALL)

                try:
                    data = json.loads(content)

                    if "thought" in data:
                        steps.append(f"**Thought**: {data['thought']}")

                    if "tool" in data:
                        steps.append(
                            f"**Action**: `{data['tool']}` with args:\n```json\n{json.dumps(data.get('args', {}), indent=2)}\n```"
                        )

                    if "final_answer" in data:
                        final_answer = data["final_answer"]
                        steps.append(f"**Final answer**: {data['final_answer']}")

                except Exception:
                    raw_content = getattr(last_msg, "content", "") or ""

                    thought_match = re.search(
                        r'"thought"\s*:\s*"((?:\\.|[^"\\])*)"',
                        raw_content,
                        re.DOTALL
                    )
                    final_match = re.search(
                        r'"final_answer"\s*:\s*"((?:\\.|[^"\\])*)"',
                        raw_content,
                        re.DOTALL
                    )

                    parsed_any = False

                    if thought_match:
                        thought_text = bytes(thought_match.group(1), "utf-8").decode("unicode_escape")
                        steps.append(f"**Thought**: {thought_text}")
                        parsed_any = True

                    if final_match:
                        answer_text = bytes(final_match.group(1), "utf-8").decode("unicode_escape")
                        final_answer = answer_text
                        steps.append(f"**Final answer**: {answer_text}")
                        parsed_any = True

                    if not parsed_any:
                        steps.append(f"**Raw model output**:\n```\n{raw_content}\n```")

            elif node_name == "tools":
                steps.append(f"**Observation**:\n```json\n{last_msg.content}\n```")

            elif node_name == "profile_update":
                content = getattr(last_msg, "content", None)
                if content:
                    final_answer = content

            elif node_name == "memory_query":
                content = getattr(last_msg, "content", None)
                if content:
                    final_answer = content

            elif node_name == "out_of_scope":
                content = getattr(last_msg, "content", "Out of scope.")
                final_answer = content

            elif node_name == "fallback":
                content = getattr(last_msg, "content", "Fallback triggered.")
                final_answer = content

                if final_answer is None:
                    final_answer = "No final answer was produced."

    return final_answer, steps

prompt = st.chat_input("Ask about the dataset...")
if prompt:
    chat_history.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, steps = run_agent(prompt, session_id)
        st.markdown(answer)
        if steps:
            with st.expander("Reasoning steps"):
                for step in steps:
                    st.markdown(step)

    chat_history.append({
        "role": "assistant",
        "content": answer,
        "steps": steps,
    })

    st.session_state.sessions[session_id] = chat_history