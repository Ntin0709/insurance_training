"""Local retrieval index for official insurance PDF chunks."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: str
    source_id: str
    title: str
    url: str
    text: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "title": self.title,
            "url": self.url,
            "text": self.text,
            "score": round(self.score, 4),
        }


class InsuranceRAGIndex:
    """Small BM25 index over scraped PDF chunks.

    This is intentionally dependency-free. It gives a production-shaped local
    baseline and can later be swapped for FAISS, Qdrant, Milvus, or pgvector.
    """

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        self.doc_tokens = [_tokenize(f"{row['title']} {row['text']}") for row in chunks]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_freq: dict[str, int] = defaultdict(int)
        for tokens in self.doc_tokens:
            for token in set(tokens):
                self.doc_freq[token] += 1

    @classmethod
    def from_chunks_file(cls, path: str | Path) -> "InsuranceRAGIndex":
        with Path(path).open("r", encoding="utf-8") as handle:
            chunks = [json.loads(line) for line in handle if line.strip()]
        return cls(chunks)

    def search(self, query: str, category: str | None = None, top_k: int = 5) -> list[RetrievalResult]:
        query_tokens = _tokenize(f"{category or ''} {query}")
        scores = []
        for idx, chunk in enumerate(self.chunks):
            category_bonus = _category_bonus(chunk, category)
            score = self._bm25_score(idx, query_tokens) + category_bonus
            if score > 0:
                scores.append((score, idx))
        scores.sort(key=lambda item: (-item[0], self.chunks[item[1]]["chunk_id"]))
        return [
            RetrievalResult(
                chunk_id=self.chunks[idx]["chunk_id"],
                source_id=self.chunks[idx]["source_id"],
                title=self.chunks[idx]["title"],
                url=self.chunks[idx]["url"],
                text=self.chunks[idx]["text"][:1400],
                score=score,
            )
            for score, idx in scores[:top_k]
        ]

    def _bm25_score(self, idx: int, query_tokens: list[str]) -> float:
        k1 = 1.5
        b = 0.75
        score = 0.0
        tf = self.term_freqs[idx]
        dl = self.doc_lengths[idx] or 1
        n_docs = len(self.chunks)
        for token in query_tokens:
            freq = tf.get(token, 0)
            if not freq:
                continue
            df = self.doc_freq.get(token, 0)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            denom = freq + k1 * (1 - b + b * dl / self.avgdl)
            score += idf * (freq * (k1 + 1) / denom)
        return score


def build_index_manifest(chunks_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    index = InsuranceRAGIndex.from_chunks_file(chunks_path)
    manifest = {
        "chunks_path": str(chunks_path),
        "chunk_count": len(index.chunks),
        "source_count": len({row["source_id"] for row in index.chunks}),
        "avg_doc_length_tokens": round(index.avgdl, 2),
        "vocab_size": len(index.doc_freq),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def _category_bonus(chunk: dict[str, Any], category: str | None) -> float:
    if not category:
        return 0.0
    haystack = f"{chunk['title']} {chunk['text']}".lower()
    terms = {
        "health": ("health", "arogya", "hospital", "medical", "swasthya"),
        "motor": ("motor", "private car", "two wheeler", "own damage", "third party"),
        "travel": ("travel", "travelsure", "journey", "trip"),
        "cyber": ("cyber", "vault", "phishing", "identity theft", "online"),
        "home": ("home", "griha", "house", "building", "contents"),
        "personal_accident": ("personal accident", "accidental death", "disability", "saral suraksha"),
    }.get(category, (category,))
    return 2.0 * sum(term in haystack for term in terms)
