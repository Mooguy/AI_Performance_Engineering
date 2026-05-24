import os
import sys
import json
import re
import argparse

# Inject workspace root directory path dynamically
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from langchain_core.messages import HumanMessage
from src.agent.graph import agent_graph

def run_interactive_cli():
    # Setup CLI command line argument parser
    parser = argparse.ArgumentParser(description="Customer Service Data Analyst CLI with SQLite Persistence")
    parser.add_argument("--session", type=str, required=True, help="A unique string ID to restore or start a session")
    args = parser.parse_args()

    print("==================================================================")
    print("🚀 Customer Service Data Analyst Agent CLI - Session Activated")
    print(f"Session ID: {args.session} (Persistent in SQLite database)")
    print("Ask data analysis questions. Type 'exit' or 'quit' to terminate.")
    print("==================================================================")

    # Use user argument as thread session identifier
    config = {"configurable": {"thread_id": args.session}}

    while True:
        try:
            user_input = input("\n👤 User: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting session. Systems spun down cleanly.")
                break

            stream_events = agent_graph.stream(
                {
                    "messages":   [HumanMessage(content=user_input)],
                    "iterations": 0,
                    "session_id": args.session   # ← must be here every turn, not just first
                },
                config=config,
                stream_mode="updates"
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
                                print(f"\n🧠 [Thought]: {data['thought']}")
                            if "tool" in data:
                                print(f"⚙️  [Action]: Invoking tool '{data['tool']}' with args: {data.get('args')}")
                            if "final_answer" in data:
                                print(f"\n🤖 Agent: {data['final_answer']}")

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

                            printed_any = False

                            if thought_match:
                                thought_text = bytes(thought_match.group(1), "utf-8").decode("unicode_escape")
                                print(f"\n🧠 [Thought]: {thought_text}")
                                printed_any = True

                            if final_match:
                                answer_text = bytes(final_match.group(1), "utf-8").decode("unicode_escape")
                                print(f"\n🤖 Agent: {answer_text}")
                                printed_any = True

                            if not printed_any:
                                print(f"\n📝 [Raw Response]: {raw_content}")

                    elif node_name == "profile_update":
                        print(f"\n🤖 Agent: {last_msg.content}")

                    elif node_name == "memory_query":
                        print(f"\n🤖 Agent: {last_msg.content}")

                    elif node_name == "out_of_scope":
                        print(f"\n🤖 Agent: {last_msg.content}")

                    elif node_name == "fallback":
                        print(f"\n🤖 Agent: {last_msg.content}")

        except KeyboardInterrupt:
            print("\nSession interrupted gracefully.")
            break
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\n❌ Operational exception encountered: {str(e)}")

if __name__ == "__main__":
    run_interactive_cli()