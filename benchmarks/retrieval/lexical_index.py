"""
Trigram + regex lexical index for the SochDB benchmark harness.
Mirrors the 'Grep — lexical' node in the hybrid retrieval diagram.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterator


def _trigrams(text: str) -> Iterator[str]:
    """Yield all overlapping 3-grams from lowercased text."""
    t = text.lower()
    for i in range(len(t) - 2):
        yield t[i : i + 3]


@dataclass
class LexicalIndex:
    """In-memory trigram index with regex scoring."""

    _docs: dict[str, str] = field(default_factory=dict)           # doc_id -> full text
    _trigram_map: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    def add(self, doc_id: str, text: str) -> None:
        self._docs[doc_id] = text
        for tg in set(_trigrams(text)):
            self._trigram_map[tg].add(doc_id)

    def build(self) -> None:
        """No-op: index is built incrementally. Kept for API symmetry."""
        pass

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """
        Returns up to k (doc_id, score) pairs sorted by descending score.

        Uses a BM25-like scoring model:
          - Trigram shortlisting to narrow candidates
          - TF-IDF scoring: term frequency * inverse document frequency
          - Length normalization via k1+b*dl/avgdl
        """
        import math

        query_tgs = list(_trigrams(query.lower()))
        if not query_tgs:
            candidates = set(self._docs.keys())
        else:
            freq: dict[str, int] = defaultdict(int)
            for tg in query_tgs:
                for doc_id in self._trigram_map.get(tg, set()):
                    freq[doc_id] += 1
            if not freq:
                return []
            threshold = max(1, int(0.3 * len(query_tgs)))
            candidates = {d for d, c in freq.items() if c >= threshold}

        tokens = re.findall(r"\w+", query.lower())
        if not tokens:
            return []

        N = len(self._docs)
        doc_lengths: dict[str, int] = {}
        avg_dl = 0.0
        for doc_id in candidates:
            dl = len(self._docs[doc_id].lower().split())
            doc_lengths[doc_id] = dl
            avg_dl += dl
        avg_dl = avg_dl / len(candidates) if candidates else 1.0

        df: dict[str, int] = {}
        for tok in tokens:
            count = 0
            for doc_id in candidates:
                if re.search(re.escape(tok), self._docs[doc_id].lower()):
                    count += 1
            df[tok] = count

        k1 = 1.2
        b = 0.75

        scored: list[tuple[str, float]] = []
        for doc_id in candidates:
            doc_text = self._docs[doc_id].lower()
            dl = doc_lengths[doc_id]
            score = 0.0
            for tok in tokens:
                try:
                    tf = len(re.findall(re.escape(tok), doc_text))
                except re.error:
                    continue
                if tf == 0:
                    continue
                idf = math.log((N - df.get(tok, 0) + 0.5) / (df.get(tok, 0) + 0.5) + 1.0)
                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
                score += idf * tf_norm
            if score > 0:
                scored.append((doc_id, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]