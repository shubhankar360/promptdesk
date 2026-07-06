"""TF-IDF retriever implemented from scratch (no ML libraries).

Markdown files in the knowledge base are split into chunks per heading,
tokenized, and scored against queries with TF-IDF cosine similarity.
Small enough to read in one sitting; fast enough for thousands of chunks.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

_TOKEN_RE = re.compile(r"[a-z0-9']+")

_STOPWORDS = frozenset(
    "a an and are as at be by for from has have how i in is it my of on or "
    "our so that the this to was we what when where which will with you your".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def chunk_markdown(text: str, title: str) -> list[dict]:
    """Split a markdown doc into one chunk per ## section (or whole doc)."""
    sections = re.split(r"(?m)^##\s+", text)
    chunks = []
    for i, section in enumerate(sections):
        body = section.strip()
        if not body:
            continue
        if i == 0 and body.startswith("# "):
            body = re.sub(r"(?m)^#\s+.*$", "", body, count=1).strip()
            if not body:
                continue
        chunks.append({"title": title, "text": body})
    return chunks


class Retriever:
    def __init__(self) -> None:
        self.chunks: list[dict] = []
        self._doc_freq: Counter = Counter()
        self._vectors: list[dict[str, float]] = []

    def index_directory(self, directory: str | Path) -> int:
        directory = Path(directory)
        for path in sorted(directory.glob("*.md")):
            title = path.stem.replace("_", " ").replace("-", " ").title()
            self.chunks.extend(chunk_markdown(path.read_text(encoding="utf-8"), title))
        self._build()
        return len(self.chunks)

    def index_texts(self, docs: list[tuple[str, str]]) -> int:
        """Index (title, markdown_text) pairs - used by tests."""
        for title, text in docs:
            self.chunks.extend(chunk_markdown(text, title))
        self._build()
        return len(self.chunks)

    def _build(self) -> None:
        self._doc_freq = Counter()
        token_lists = []
        for chunk in self.chunks:
            tokens = tokenize(chunk["text"] + " " + chunk["title"])
            token_lists.append(tokens)
            self._doc_freq.update(set(tokens))
        n = max(len(self.chunks), 1)
        self._vectors = []
        for tokens in token_lists:
            tf = Counter(tokens)
            vec = {
                term: (count / len(tokens)) * math.log(1 + n / self._doc_freq[term])
                for term, count in tf.items()
            }
            self._vectors.append(vec)

    def search(self, query: str, top_k: int = 3, min_score: float = 0.05) -> list[dict]:
        tokens = tokenize(query)
        if not tokens or not self.chunks:
            return []
        n = len(self.chunks)
        tf = Counter(tokens)
        qvec = {
            term: (count / len(tokens)) * math.log(1 + n / max(self._doc_freq.get(term, 0), 1))
            for term, count in tf.items()
        }
        qnorm = math.sqrt(sum(v * v for v in qvec.values())) or 1.0
        scored = []
        for chunk, vec in zip(self.chunks, self._vectors):
            dot = sum(qvec[t] * vec.get(t, 0.0) for t in qvec)
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            score = dot / (qnorm * norm)
            if score >= min_score:
                scored.append({**chunk, "score": score})
        scored.sort(key=lambda c: c["score"], reverse=True)
        return scored[:top_k]
