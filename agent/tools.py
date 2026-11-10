"""The agent's three tools, plus the layer that applies injection defenses.

Tools are plain Python functions. The LLM only ever sees TOOL_SCHEMAS (what
tools exist) and the string each tool returns.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .corpus import Corpus
from .security import check_url, filter_untrusted, wrap_untrusted

DEFENSE_LEVELS = ("none", "delimiters", "filter")

# OpenAI-style function schemas; this is what gets bound to the model.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_articles",
            "description": "Search the Avalen news archive by keywords. Returns up to 3 matching articles with their full text and any linked URL.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Keywords to search for"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch the text of a web page. Use only for URLs that appear as a link in an article and are needed to answer the question.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "Full http(s) URL"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_note",
            "description": "Save a short fact to your scratchpad so you can use it in later steps.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "The fact to remember"}},
                "required": ["text"],
            },
        },
    },
]


@dataclass
class ToolResult:
    output: str            # what the model sees
    ok: bool = True        # False if the tool returned an error
    blocked: bool = False  # True if a security check stopped the call
    new_note: str | None = None
    flagged: int = 0       # segments removed by the filter defense


class ToolBox:
    """Runs tool calls and applies the selected defense to untrusted output.

    defense:
      none        raw tool output goes straight to the model
      delimiters  output is wrapped in <untrusted_data> tags
      filter      instruction-like sentences are removed, then wrapped
    live_web:
      False       fetch_url reads from data/web_pages.json (reproducible evals)
      True        fetch_url makes real HTTP requests
    """

    def __init__(self, corpus: Corpus | None = None, defense: str = "delimiters", live_web: bool = False):
        if defense not in DEFENSE_LEVELS:
            raise ValueError(f"defense must be one of {DEFENSE_LEVELS}")
        self.corpus = corpus or Corpus()
        self.defense = defense
        self.live_web = live_web

    # -- public entry point ----------------------------------------------------

    def execute(self, name: str, args: dict) -> ToolResult:
        args = args or {}
        try:
            if name == "search_articles":
                return self._search(str(args.get("query", "")))
            if name == "fetch_url":
                return self._fetch(str(args.get("url", "")))
            if name == "save_note":
                return self._save_note(str(args.get("text", "")))
            return ToolResult(f"ERROR: unknown tool '{name}'", ok=False)
        except Exception as exc:  # a tool crash should not kill the run
            return ToolResult(f"ERROR: {name} failed: {exc}", ok=False)

    # -- tools -----------------------------------------------------------------

    def _search(self, query: str) -> ToolResult:
        if not query.strip():
            return ToolResult("ERROR: empty query", ok=False)
        hits = self.corpus.search(query)
        if not hits:
            return ToolResult("No articles matched that query.")
        blocks = []
        for a in hits:
            block = f"[{a['id']}] {a['title']}\n{a['body']}"
            if a.get("link"):
                block += f"\nLink: {a['link']}"
            blocks.append(block)
        return self._defend("\n\n".join(blocks), source="news_archive")

    def _fetch(self, url: str) -> ToolResult:
        allowed, reason = check_url(url, resolve_dns=self.live_web)
        if not allowed:
            return ToolResult(f"BLOCKED: {reason}", ok=False, blocked=True)
        if self.live_web:
            text = _http_get_text(url)
        else:
            text = self.corpus.get_page(url)
            if text is None:
                return ToolResult("ERROR: page not found", ok=False)
        return self._defend(text, source=url)

    def _save_note(self, text: str) -> ToolResult:
        text = text.strip()[:500]
        if not text:
            return ToolResult("ERROR: empty note", ok=False)
        return ToolResult("Note saved.", new_note=text)

    # -- defense layer ---------------------------------------------------------

    def _defend(self, text: str, source: str) -> ToolResult:
        flagged = 0
        if self.defense == "filter":
            text, flagged = filter_untrusted(text)
        if self.defense in ("delimiters", "filter"):
            text = wrap_untrusted(text, source)
        return ToolResult(text, flagged=flagged)


def _http_get_text(url: str, max_bytes: int = 200_000, max_chars: int = 4000) -> str:
    """Live fetch. Redirects are NOT followed, because a public URL can redirect
    to an internal one and bypass the SSRF check."""
    import requests

    resp = requests.get(
        url,
        timeout=8,
        allow_redirects=False,
        stream=True,
        headers={"User-Agent": "agent-injection-eval/0.1"},
    )
    if 300 <= resp.status_code < 400:
        return f"ERROR: redirect to {resp.headers.get('Location', '?')} was not followed"
    if resp.status_code != 200:
        return f"ERROR: HTTP {resp.status_code}"
    raw = resp.raw.read(max_bytes, decode_content=True)
    html = raw.decode(resp.encoding or "utf-8", errors="replace")
    html = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text).strip()[:max_chars]
