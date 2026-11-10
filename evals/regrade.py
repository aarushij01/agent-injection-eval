"""Re-grade saved runs without calling the model again.

Use this after changing graders.py or tasks.json:

    python -m evals.regrade
"""
from __future__ import annotations

import json
from pathlib import Path

from evals.graders import grade
from evals.run_evals import summarize, to_markdown

ROOT = Path(__file__).resolve().parent.parent
TASKS = {t["id"]: t for t in json.loads((ROOT / "evals" / "tasks.json").read_text(encoding="utf-8"))}


def main() -> None:
    for run_dir in sorted((ROOT / "results").glob("*/")):
        traces_path = run_dir / "traces.jsonl"
        if not traces_path.exists():
            continue
        records = [json.loads(line) for line in traces_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        changed = 0
        for r in records:
            new = grade(TASKS[r["id"]], r["final_answer"], r["tool_log"], r["status"])
            changed += new["failure"] != r["failure"]
            r.update(new)
        old = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        meta = {k: old[k] for k in ("run_name", "model", "defense", "max_steps", "date")}
        summary = summarize(records, meta)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (run_dir / "summary.md").write_text(to_markdown(summary), encoding="utf-8")
        traces_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
        print(f"{run_dir.name}: {changed} grade(s) changed")


if __name__ == "__main__":
    main()
