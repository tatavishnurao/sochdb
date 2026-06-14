#!/usr/bin/env python3

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def rerank_cache_key(model: str, question: str, memory_id: int, text: str) -> str:
    raw = json.dumps(
        {
            "model": model,
            "question": question,
            "memory_id": memory_id,
            "text": text,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class RerankCache:
    def __init__(self, path: Optional[str]):
        self.path = Path(path) if path else None
        self.data: Dict[str, float] = {}

        if self.path and self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    self.data[row["key"]] = float(row["score"])

    def get(self, key: str):
        return self.data.get(key)

    def set(self, key: str, score: float):
        self.data[key] = score

        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "score": score}) + "\n")


class Reranker:
    def __init__(
        self,
        provider: str = "none",
        model: str = "BAAI/bge-reranker-base",
        cache_path: Optional[str] = None,
        batch_size: int = 32,
    ):
        self.provider = provider
        self.model = model
        self.cache = RerankCache(cache_path)
        self.batch_size = batch_size
        self.client = None

        if provider == "none":
            return

        if provider == "sentence_transformers":
            from sentence_transformers import CrossEncoder
            self.client = CrossEncoder(model)
            return

        raise ValueError(f"Unsupported reranker provider: {provider}")

    def rerank(
        self,
        question: str,
        candidates: List[Tuple[int, str]],
    ) -> List[Tuple[int, float]]:
        if self.provider == "none":
            return [(mid, 0.0) for mid, _ in candidates]

        scored: List[Tuple[int, Optional[float], str]] = []
        missing_pairs = []
        missing_meta = []

        for mid, text in candidates:
            key = rerank_cache_key(self.model, question, mid, text)
            cached = self.cache.get(key)

            if cached is not None:
                scored.append((mid, cached, text))
            else:
                scored.append((mid, None, text))
                missing_pairs.append((question, text))
                missing_meta.append((key, mid, text))

        if missing_pairs:
            for start in range(0, len(missing_pairs), self.batch_size):
                batch_pairs = missing_pairs[start:start + self.batch_size]
                batch_meta = missing_meta[start:start + self.batch_size]

                scores = self.client.predict(batch_pairs)

                for score, (key, mid, text) in zip(scores, batch_meta):
                    self.cache.set(key, float(score))

        final = []

        for mid, maybe_score, text in scored:
            if maybe_score is None:
                key = rerank_cache_key(self.model, question, mid, text)
                maybe_score = self.cache.get(key)
            final.append((mid, float(maybe_score)))

        final.sort(key=lambda x: x[1], reverse=True)
        return final
