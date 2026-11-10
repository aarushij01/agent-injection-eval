"""Ask the agent one question from the command line.

    python run_agent.py "When did the Harbor Line open?"
    python run_agent.py "How far below forecast was ridership?" --thread demo
    python run_agent.py "And which station was busiest?" --thread demo   # continues

With --thread, state is checkpointed to checkpoints.sqlite, so the second
command above picks up the first one's history and notes.
"""
from __future__ import annotations

import argparse
import sqlite3

from agent.graph import AgentConfig, build_graph, run_question
from agent.llm import MockLLM, make_groq_llm
from agent.tools import DEFENSE_LEVELS, ToolBox


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the research agent on one question.")
    parser.add_argument("question")
    parser.add_argument("--llm", choices=["groq", "mock"], default="groq")
    parser.add_argument("--defense", choices=DEFENSE_LEVELS, default="delimiters")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--thread", help="Thread id; state is saved and resumed under this name")
    parser.add_argument("--live-web", action="store_true", help="fetch_url makes real HTTP requests")
    args = parser.parse_args()

    llm = MockLLM() if args.llm == "mock" else make_groq_llm()
    toolbox = ToolBox(defense=args.defense, live_web=args.live_web)

    checkpointer = None
    if args.thread:
        from langgraph.checkpoint.sqlite import SqliteSaver

        checkpointer = SqliteSaver(sqlite3.connect("checkpoints.sqlite", check_same_thread=False))

    app = build_graph(llm, toolbox, AgentConfig(max_steps=args.max_steps), checkpointer)
    state = run_question(app, args.question, thread_id=args.thread or "single-run")

    print("\n--- tool calls ---")
    for entry in state.get("tool_log", []):
        flag = " [BLOCKED]" if entry["blocked"] else ""
        print(f"step {entry['step']}: {entry['tool']}({entry['args']}){flag}")
    if state.get("notes"):
        print("\n--- notes ---")
        for note in state["notes"]:
            print(f"- {note}")
    print(f"\n--- answer ({state['status']}, {state['steps']} steps) ---")
    print(state["final_answer"])


if __name__ == "__main__":
    main()
