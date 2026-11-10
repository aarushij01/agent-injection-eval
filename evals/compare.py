"""Print a markdown table comparing every run in results/.

    python -m evals.compare
"""
from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main() -> None:
    summaries = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RESULTS.glob("*/summary.json"))]
    if not summaries:
        raise SystemExit("No runs found in results/. Run evals.run_evals first.")
    print("| Run | Model | Defense | Task success | Injection resistance | Correct + resisted | Avg steps | Step limit |")
    print("|---|---|---|---|---|---|---|---|")
    for s in summaries:
        print(
            f"| {s['run_name']} | {s['model']} | {s['defense']} "
            f"| {s['task_success']['overall_pct']}% | {s['injection']['resistance_pct']}% "
            f"| {s['injection']['correct_and_resisted_pct']}% | {s['efficiency']['avg_steps']} "
            f"| {s['efficiency']['step_limit_pct']}% |"
        )


if __name__ == "__main__":
    main()
