"""Run the evaluation suite.

    python -m evals.run_evals --llm groq --defense none
    python -m evals.run_evals --llm groq --defense delimiters
    python -m evals.run_evals --llm groq --defense filter
    python -m evals.run_evals --llm mock                     # offline smoke test

Each run writes to results/<run-name>/:
    traces.jsonl   one full trace per task (messages, tool calls, grade)
    summary.json   the metrics
    summary.md     the same metrics as a readable table
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from agent.graph import AgentConfig, build_graph, message_text, run_question
from agent.llm import MockLLM, make_groq_llm
from agent.tools import DEFENSE_LEVELS, ToolBox
from evals.graders import grade

ROOT = Path(__file__).resolve().parent.parent
TASKS_PATH = ROOT / "evals" / "tasks.json"


def serialize_messages(messages: list) -> list[dict]:
    out = []
    for m in messages:
        item = {"role": m.type, "content": message_text(m)}
        if getattr(m, "tool_calls", None):
            item["tool_calls"] = [{"name": c["name"], "args": c["args"]} for c in m.tool_calls]
        out.append(item)
    return out


def pct(n: int, d: int) -> float:
    return round(100 * n / d, 1) if d else 0.0


def summarize(records: list[dict], meta: dict) -> dict:
    normal = [r for r in records if r["category"] != "injection"]
    inject = [r for r in records if r["category"] == "injection"]

    by_cat = defaultdict(lambda: [0, 0])
    for r in normal:
        by_cat[r["category"]][0] += r["correct"]
        by_cat[r["category"]][1] += 1

    resisted = [r for r in inject if not r["injection_followed"]]
    useful = [r for r in inject if not r["injection_followed"] and r["correct"]]
    all_tool_calls = [e for r in records for e in r["tool_log"]]

    return {
        **meta,
        "tasks": len(records),
        "task_success": {
            "overall_pct": pct(sum(r["correct"] for r in normal), len(normal)),
            "n": len(normal),
            "by_category": {c: {"pct": pct(ok, n), "n": n} for c, (ok, n) in sorted(by_cat.items())},
        },
        "injection": {
            "n": len(inject),
            "resistance_pct": pct(len(resisted), len(inject)),
            "correct_and_resisted_pct": pct(len(useful), len(inject)),
            "followed": [{"id": r["id"], "attack": r["attack"], "evidence": r["evidence"]} for r in inject if r["injection_followed"]],
        },
        "efficiency": {
            "avg_steps": round(sum(r["steps"] for r in records) / max(len(records), 1), 2),
            "step_limit_pct": pct(sum(r["status"] == "step_limit" for r in records), len(records)),
            "llm_errors": sum(r["status"] == "llm_error" for r in records),
            "tool_calls": len(all_tool_calls),
            "tool_calls_blocked": sum(e["blocked"] for e in all_tool_calls),
            "segments_filtered": sum(e["flagged"] for e in all_tool_calls),
        },
        "failures": dict(Counter(r["failure"] for r in records if r["failure"] != "ok")),
    }


def to_markdown(s: dict) -> str:
    ts, inj, eff = s["task_success"], s["injection"], s["efficiency"]
    lines = [
        f"# Eval run: {s['run_name']}",
        "",
        f"- Model: `{s['model']}`  |  Defense: `{s['defense']}`  |  Max steps: {s['max_steps']}  |  Date: {s['date']}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Task success (normal tasks, n={ts['n']}) | {ts['overall_pct']}% |",
    ]
    for cat, v in ts["by_category"].items():
        lines.append(f"| - {cat} (n={v['n']}) | {v['pct']}% |")
    lines += [
        f"| Injection resistance (n={inj['n']}) | {inj['resistance_pct']}% |",
        f"| Correct answer AND resisted (n={inj['n']}) | {inj['correct_and_resisted_pct']}% |",
        f"| Average steps per task | {eff['avg_steps']} |",
        f"| Hit step limit | {eff['step_limit_pct']}% |",
        f"| LLM errors | {eff['llm_errors']} |",
        f"| Tool calls blocked by SSRF guard | {eff['tool_calls_blocked']} |",
        f"| Segments removed by filter | {eff['segments_filtered']} |",
        "",
        "## Failures by type",
        "",
    ]
    lines += [f"- {k}: {v}" for k, v in sorted(s["failures"].items())] or ["- none"]
    lines += ["", "## Injections that worked", ""]
    lines += [f"- {f['id']} ({f['attack']}): {'; '.join(f['evidence'])}" for f in inj["followed"]] or ["- none"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agent evaluation suite.")
    parser.add_argument("--llm", choices=["groq", "mock"], default="groq")
    parser.add_argument("--defense", choices=DEFENSE_LEVELS, default="delimiters")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--only", help="Run one category (single_hop, multi_hop, unanswerable, injection) or one task id")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to wait between tasks (helps with rate limits)")
    parser.add_argument("--name", help="Run name; defaults to <model>_<defense>")
    args = parser.parse_args()

    tasks = json.loads(TASKS_PATH.read_text(encoding="utf-8"))
    if args.only:
        tasks = [t for t in tasks if args.only in (t["category"], t["id"])]
        if not tasks:
            raise SystemExit(f"No tasks match --only {args.only}")

    llm = MockLLM() if args.llm == "mock" else make_groq_llm()
    model_name = getattr(llm, "model_name", None) or getattr(llm, "model", "unknown")
    toolbox = ToolBox(defense=args.defense, live_web=False)  # evals always use the offline fixtures
    app = build_graph(llm, toolbox, AgentConfig(max_steps=args.max_steps))

    run_name = args.name or f"{model_name}_{args.defense}".replace("/", "-")
    out_dir = ROOT / "results" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with (out_dir / "traces.jsonl").open("w", encoding="utf-8") as trace_file:
        for i, task in enumerate(tasks, 1):
            state = run_question(app, task["question"], thread_id=task["id"])
            result = grade(task, state.get("final_answer", ""), state.get("tool_log", []), state.get("status", ""))
            record = {
                "id": task["id"],
                "category": task["category"],
                "attack": task.get("attack"),
                "question": task["question"],
                "final_answer": state.get("final_answer", ""),
                "status": state.get("status", ""),
                "steps": state.get("steps", 0),
                "notes": state.get("notes", []),
                "tool_log": state.get("tool_log", []),
                **result,
            }
            records.append(record)
            trace_file.write(json.dumps({**record, "messages": serialize_messages(state["messages"])}, ensure_ascii=False) + "\n")
            mark = "ok  " if result["failure"] == "ok" else "FAIL"
            print(f"[{i:>2}/{len(tasks)}] {mark} {task['id']} {result['failure']:<20} steps={record['steps']}")
            if args.sleep:
                time.sleep(args.sleep)

    meta = {
        "run_name": run_name,
        "model": model_name,
        "defense": args.defense,
        "max_steps": args.max_steps,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    summary = summarize(records, meta)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    markdown = to_markdown(summary)
    (out_dir / "summary.md").write_text(markdown, encoding="utf-8")
    print("\n" + markdown)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
