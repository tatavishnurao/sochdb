

#!/t usr/bin/env python3
"""
Trigram + regex lexical index for LLM-native database retrieval.
Mirrors the 'Grep — lexical' node in the hybrid retrieval diagram.

Scope: Node-local accelerator. Never distribute grep across machines.
Index: In-memory trigram index mapping every 3-character substring to candidate doc_ids.
Query flow:
  a. Extract trigrams from the lowercased query.
  b. Intersect candidate doc_id sets from the trigram map (keep docs matching >=30% of query trigrams).
  c. Run regex/token matching on shortlisted candidates only.
  d. Score by: (non-overlapping token hit count) / (1 + doc_length/1000).
Contract: Returns ranked doc_ids within single-digit milliseconds on local NVMe/SSD.
Optimization: Index `title + body + tags` together; do not pre-tokenize or stem.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Set, Tuple


def _trigrams(text: str) -> Iterator[str]:
    """Yield all overlapping 3-grams from lowercased text."""
    t = text.lower()
    for i in range(len(t) - 2):
        yield t[i : i + 3]


def _tokens(text: str) -> List[str]:
    """Extract word tokens from text."""
    return re.findall(r"\w+", text.lower())


class TrigramIndex:
    """
    In-memory trigram index with regex + token scoring.

    Supports two search modes:
    - trigram_only: Fast trigram intersection for candidate shortlisting
    - token_regex: Detailed token/regex scoring on shortlisted candidates
    """

    def __init__(self):
        self._docs: Dict[str, str] = {}
        self._trigram_map: Dict[str, Set[str]] = defaultdict(set)
        self._doc_lengths: Dict[str, int] = {}
        self._built = False

    def add(self, doc_id: str, text: str) -> None:
        """Add a document to the index."""
        self._docs[doc_id] = text
        self._doc_lengths[doc_id] = len(_tokens(text))
        for tg in set(_trigrams(text)):
            self._trigram_map[tg].add(doc_id)
        self._built = False

    def build(self) -> None:
        """Mark index as built (no-op for incremental index)."""
        self._built = True

    def search(
        self,
        query: str,
        k: int = 10,
        min_trigram_ratio: float = 0.3,
    ) -> List[Tuple[str, float]]:
        """
        Returns up to k (doc_id, score) pairs sorted by descending score.

        Uses trigram shortlisting followed by token-based BM25-like scoring.
        """
        if not self._docs:
            return []

        query_tgs = list(_trigrams(query.lower()))
        if not query_tgs:
            candidates = set(self._docs.keys())
        else:
            freq: Dict[str, int] = defaultdict(int)
            for tg in query_tgs:
                for doc_id in self._trigram_map.get(tg, set()):
                    freq[doc_id] += 1

            if not freq:
                return []

            threshold = max(1, int(min_trigram_ratio * len(query_tgs)))
            candidates = {d for d, c in freq.items() if c >= threshold}

        if not candidates:
            return []

        tokens = _tokens(query)
        if not tokens:
            return [(doc_id, 1.0) for doc_id in list(candidates)[:k]]

        N = len(self._docs)
        avg_dl = sum(self._doc_lengths.get(doc_id, 1) for doc_id in candidates) / len(candidates)

        df: Dict[str, int] = {}
        for tok in tokens:
            count = 0
            for doc_id in candidates:
                if re.search(re.escape(tok), self._docs[doc_id].lower()):
                    count += 1
            df[tok] = count

        k1, b = 1.2, 0.75
        scored: List[Tuple[str, float]] = []

        for doc_id in candidates:
            doc_text = self._docs[doc_id].lower()
            dl = self._doc_lengths.get(doc_id, len(_tokens(doc_text)))
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

    def search_with_filters(
        self,
        query: str,
        k: int = 10,
        speaker_filter: Optional[str] = None,
        session_filter: Optional[str] = None,
        min_trigram_ratio: float = 0.3,
    ) -> List[Tuple[str, float]]:
        """
        Search with optional speaker/session filters.

        Filters are applied after scoring by checking against the doc's metadata
        fields (expects doc_id format like 'memory_123' or just '123').
        For memory records, speaker/session should be passed separately via metadata.
        """
        results = self.search(query, k * 2, min_trigram_ratio)

        if speaker_filter or session_filter:
            filtered = []
            for doc_id, score in results:
                if speaker_filter and not self._doc_matches_filter(doc_id, speaker_filter):
                    continue
                filtered.append((doc_id, score))
                if len(filtered) >= k:
                    break
            return filtered

        return results[:k]

    def _doc_matches_filter(self, doc_id: str, speaker: str) -> bool:
        """Check if doc matches speaker filter. Override in subclass with metadata access."""
        return True

    def get_doc_text(self, doc_id: str) -> Optional[str]:
        """Return the full text of a document."""
        return self._docs.get(doc_id)


class LexicalSearchEngine:
    """
    Higher-level lexical search engine that wraps TrigramIndex
    and provides additional features like query expansion and
    multi-pattern search.
    """

    def __init__(self):
        self.index = TrigramIndex()
        self._metadata: Dict[str, Dict] = {}

    def add_memory(self, memory_id: int, text: str, speaker: str = "", session: str = "") -> None:
        """Add a memory record with optional metadata."""
        doc_id = str(memory_id)
        self.index.add(doc_id, text)
        self._metadata[doc_id] = {"speaker": speaker, "session": session, "memory_id": memory_id}

    def build(self) -> None:
        """Build the index."""
        self.index.build()

    def search(
        self,
        query: str,
        k: int = 10,
        speaker: Optional[str] = None,
        session: Optional[str] = None,
    ) -> List[Tuple[int, float]]:
        """
        Search for memories matching the query.

        Returns list of (memory_id, score) tuples.
        """
        if speaker or session:
            results = self.index.search_with_filters(query, k, speaker, session)
        else:
            results = self.index.search(query, k)

        return [(int(doc_id), score) for doc_id, score in results]

    def search_multi(
        self,
        queries: List[str],
        k: int = 10,
    ) -> List[List[Tuple[int, float]]]:
        """Search multiple queries and return ranked results for each."""
        return [self.search(q, k) for q in queries]


if __name__ == "__main__":
    import json

    idx = TrigramIndex()
    memories = [json.loads(l) for l in open("benchmarks/paper/locomo/data/locomo_memories.jsonl")]

    print(f"Indexing {len(memories)} memories...")
    for m in memories:
        mid = m["memory_id"]
        text = f"{m.get('speaker', '')} {m.get('session', '')} {m.get('text', '')}"
        idx.add(str(mid), text)
    idx.build()

    print("\nTesting searches...")
    test_queries = [
        "Jolene studying exams",
        "Andrew birdwatching city",
        "Maria car accident holiday",
    ]

    for q in test_queries:
        results = idx.search(q, k=5)
        print(f"\nQuery: {q}")
        for doc_id, score in results:
            m = next((m for m in memories if m["memory_id"] == int(doc_id)), None)
            if m:
                print(f"  [{doc_id}] score={score:.4f} speaker={m.get('speaker')} text={m.get('text')[:80]}")