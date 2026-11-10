"""Tiny local news corpus with BM25 keyword search.

The corpus is fictional on purpose: the model cannot answer from its training
data, so a correct answer proves it actually used the tools.

To plug in a real retriever (e.g. ChromaDB), replace Corpus.search() with a
function that returns the same list of article dicts.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

STOPWORDS = set(
    "a an and are as at be by did do does for from had has have how in is it its "
    "many much of on or that the their this to was were what when where which who "
    "whom whose will with".split()
)


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", text.lower())
    return [w for w in words if w not in STOPWORDS]


class Corpus:
    def __init__(self, articles_path: Path | None = None, pages_path: Path | None = None):
        articles_path = articles_path or DATA_DIR / "articles.json"
        pages_path = pages_path or DATA_DIR / "web_pages.json"
        self.articles: list[dict] = json.loads(articles_path.read_text(encoding="utf-8"))
        self.pages: dict[str, str] = json.loads(pages_path.read_text(encoding="utf-8"))

        # Title is counted twice so title matches rank higher.
        self._docs = [tokenize(f"{a['title']} {a['title']} {a['body']}") for a in self.articles]
        self._tf = [Counter(d) for d in self._docs]
        self._avg_len = sum(len(d) for d in self._docs) / max(len(self._docs), 1)
        df: Counter = Counter()
        for d in self._docs:
            df.update(set(d))
        n = len(self._docs)
        self._idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def search(self, query: str, k: int = 3) -> list[dict]:
        """Return the top-k articles for a keyword query (BM25)."""
        k1, b = 1.5, 0.75
        terms = tokenize(query)
        scored = []
        for i, tf in enumerate(self._tf):
            score = 0.0
            length = len(self._docs[i])
            for t in terms:
                if t not in tf:
                    continue
                f = tf[t]
                score += self._idf[t] * f * (k1 + 1) / (f + k1 * (1 - b + b * length / self._avg_len))
            if score > 0:
                scored.append((score, i))
        scored.sort(reverse=True)
        return [self.articles[i] for _, i in scored[:k]]

    def get_page(self, url: str) -> str | None:
        return self.pages.get(url.strip())
