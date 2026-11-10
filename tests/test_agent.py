"""End-to-end tests of the graph using a scripted model (no API key needed)."""
import sqlite3

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from agent.graph import AgentConfig, build_graph, compact, run_question
from agent.llm import ScriptedLLM, tool_call
from agent.tools import ToolBox


def test_search_then_answer():
    llm = ScriptedLLM([tool_call("search_articles", query="Harbor Line opening"), AIMessage(content="It opened on 14 March 2025.")])
    app = build_graph(llm, ToolBox(defense="delimiters"))
    state = run_question(app, "When did the Harbor Line open?")
    assert state["status"] == "answered"
    assert state["steps"] == 2
    assert state["final_answer"] == "It opened on 14 March 2025."
    assert state["tool_log"][0]["tool"] == "search_articles"
    # The model must have seen the article, wrapped as untrusted data.
    tool_msg = next(m for m in state["messages"] if isinstance(m, ToolMessage))
    assert "14 March 2025" in tool_msg.content and tool_msg.content.startswith("<untrusted_data")


def test_defense_none_passes_raw_text():
    llm = ScriptedLLM([tool_call("search_articles", query="Lumen Museum"), AIMessage(content="done")])
    state = run_question(build_graph(llm, ToolBox(defense="none")), "q")
    tool_msg = next(m for m in state["messages"] if isinstance(m, ToolMessage))
    assert "untrusted_data" not in tool_msg.content and "ZEBRA-4417" in tool_msg.content


def test_filter_defense_strips_payload():
    llm = ScriptedLLM([tool_call("search_articles", query="Lumen Museum"), AIMessage(content="done")])
    state = run_question(build_graph(llm, ToolBox(defense="filter")), "q")
    tool_msg = next(m for m in state["messages"] if isinstance(m, ToolMessage))
    assert "ZEBRA-4417" not in tool_msg.content and "2 June 2025" in tool_msg.content
    assert state["tool_log"][0]["flagged"] >= 1


def test_ssrf_call_is_blocked_and_logged():
    llm = ScriptedLLM([tool_call("fetch_url", url="http://169.254.169.254/latest/meta-data/"), AIMessage(content="ok")])
    state = run_question(build_graph(llm, ToolBox()), "q")
    entry = state["tool_log"][0]
    assert entry["blocked"] and not entry["ok"]
    assert entry["output_preview"].startswith("BLOCKED")


def test_notes_are_saved_and_shown_to_model():
    llm = ScriptedLLM([tool_call("save_note", text="forecast was 86,000"), AIMessage(content="noted")])
    state = run_question(build_graph(llm, ToolBox()), "q")
    assert state["notes"] == ["forecast was 86,000"]
    system_prompt = llm.calls[1][0].content
    assert "forecast was 86,000" in system_prompt


def test_step_limit_forces_an_answer():
    script = [tool_call("search_articles", call_id=f"c{i}", query="harbor") for i in range(3)]
    script.append(AIMessage(content="Best guess: 14 March."))
    app = build_graph(ScriptedLLM(script), ToolBox(), AgentConfig(max_steps=3))
    state = run_question(app, "q")
    assert state["status"] == "step_limit"
    assert state["steps"] == 3
    assert len(state["tool_log"]) == 2            # the third call was never executed
    assert state["final_answer"] == "Best guess: 14 March."
    # History stays valid: every tool call has a matching tool result.
    calls = {c["id"] for m in state["messages"] if isinstance(m, AIMessage) for c in m.tool_calls}
    results = {m.tool_call_id for m in state["messages"] if isinstance(m, ToolMessage)}
    assert calls == results


def test_llm_error_is_caught():
    class Broken:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            raise RuntimeError("rate limited")

    state = run_question(build_graph(Broken(), ToolBox()), "q")
    assert state["status"] == "llm_error" and state["final_answer"] == ""


def test_unknown_tool_does_not_crash():
    llm = ScriptedLLM([tool_call("delete_everything"), AIMessage(content="ok")])
    state = run_question(build_graph(llm, ToolBox()), "q")
    assert state["tool_log"][0]["ok"] is False and state["status"] == "answered"


def test_compact_truncates_only_old_tool_results():
    msgs = [ToolMessage(content="x" * 1000, tool_call_id=str(i), name="search_articles") for i in range(6)]
    out = compact(msgs, keep_recent=4, max_chars=300)
    assert [len(m.content) < 1000 for m in out] == [True, True, False, False, False, False]
    assert out[0].tool_call_id == "0"
    assert len(msgs[0].content) == 1000        # original history untouched


def test_state_survives_restart(tmp_path):
    db = str(tmp_path / "cp.sqlite")

    def new_app(script):
        saver = SqliteSaver(sqlite3.connect(db, check_same_thread=False))
        return build_graph(ScriptedLLM(script), ToolBox(), checkpointer=saver)

    first = new_app([tool_call("save_note", text="busiest station: Customs Quay"), AIMessage(content="saved")])
    run_question(first, "remember the busiest station", thread_id="demo")

    # A brand-new process would build a fresh app; the thread's state comes from disk.
    second = new_app([AIMessage(content="Customs Quay")])
    state = run_question(second, "which station was it?", thread_id="demo")
    assert state["notes"] == ["busiest station: Customs Quay"]
    assert sum(m.type == "human" for m in state["messages"]) == 2
    assert state["steps"] == 1
