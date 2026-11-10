"""Sanity checks on the benchmark itself: a broken task gives meaningless numbers."""
import json
from pathlib import Path

from agent.corpus import Corpus
from evals.graders import norm

ROOT = Path(__file__).resolve().parent.parent
TASKS = json.loads((ROOT / "evals" / "tasks.json").read_text(encoding="utf-8"))
CORPUS = Corpus()
ALL_TEXT = norm(" ".join(a["title"] + " " + a["body"] for a in CORPUS.articles) + " " + " ".join(CORPUS.pages.values()))


def test_ids_unique_and_counts():
    ids = [t["id"] for t in TASKS]
    assert len(ids) == len(set(ids))
    assert sum(t["category"] == "injection" for t in TASKS) == 10
    assert sum(t["category"] != "injection" for t in TASKS) == 20


def test_expected_answers_exist_in_sources():
    for t in TASKS:
        if t["category"] == "unanswerable" or t.get("computed"):
            continue
        for needle in t.get("must_include", []):
            assert norm(needle) in ALL_TEXT, f"{t['id']}: '{needle}' is not in any source"
        if t.get("must_include_any"):
            assert any(norm(n) in ALL_TEXT for n in t["must_include_any"]), f"{t['id']}: no accepted answer is in any source"


def test_injection_tasks_have_a_planted_payload():
    for t in TASKS:
        if t["category"] != "injection":
            continue
        payloads = t.get("forbidden_in_answer", []) + [r["arg_contains"] for r in t.get("forbidden_tool_calls", [])]
        assert payloads, f"{t['id']} has nothing to detect"
        if t.get("attack") == "system_prompt_leak":
            continue  # payload lives in the system prompt, not the corpus
        assert any(norm(p) in ALL_TEXT for p in payloads), f"{t['id']}: payload not planted in any source"


def test_search_finds_the_right_article():
    assert CORPUS.search("Harbor Line open date")[0]["id"] in ("a01", "a09")
    assert CORPUS.search("Kestrel K-9 recall")[0]["id"] == "a02"
    assert CORPUS.search("zzzz qqqq") == []
