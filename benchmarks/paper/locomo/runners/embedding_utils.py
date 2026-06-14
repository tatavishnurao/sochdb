#!/usr/bin/env python3

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def normalize_text(text: str) -> str:
    chars = []
    for ch in text.lower():
        if ch.isalnum() or ch.isspace():
            chars.append(ch)
        else:
            chars.append(" ")
    return " ".join("".join(chars).split())


def stable_hash(token: str) -> int:
    h = 0xCBF29CE484222325
    for b in token.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def hash_embed(text: str, dim: int) -> List[float]:
    vec = [0.0] * dim
    for token in normalize_text(text).split():
        vec[stable_hash(token) % dim] += 1.0

    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def text_cache_key(provider: str, model: str, dim: int, text: str) -> str:
    raw = json.dumps(
        {
            "provider": provider,
            "model": model,
            "dim": dim,
            "text": text,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class EmbeddingCache:
    def __init__(self, path: Optional[str]):
        self.path = Path(path) if path else None
        self.data: Dict[str, List[float]] = {}

        if self.path and self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    self.data[row["key"]] = row["embedding"]

    def get(self, key: str):
        return self.data.get(key)

    def set(self, key: str, embedding: List[float]):
        self.data[key] = embedding

        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "embedding": embedding}) + "\n")


class Embedder:
    def __init__(
        self,
        provider: str = "hash",
        model: str = "text-embedding-3-small",
        dim: int = 1536,
        cache_path: Optional[str] = None,
        batch_size: int = 64,
    ):
        self.provider = provider
        self.model = model
        self.dim = dim
        self.cache = EmbeddingCache(cache_path)
        self.batch_size = batch_size
        self.client = None

        if self.provider == "hash":
            return

        if self.provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is required for --embedding-provider openai")
            from openai import OpenAI
            self.client = OpenAI()
            return

        if self.provider == "nvidia":
            if not os.getenv("NVIDIA_API_KEY"):
                raise RuntimeError("NVIDIA_API_KEY is required for --embedding-provider nvidia")
            from openai import OpenAI
            self.client = OpenAI(
                api_key=os.environ["NVIDIA_API_KEY"],
                base_url="https://integrate.api.nvidia.com/v1",
            )
            return

        if self.provider == "sentence_transformers":
            from sentence_transformers import SentenceTransformer
            self.client = SentenceTransformer(self.model)

            if hasattr(self.client, "get_embedding_dimension"):
                actual_dim = self.client.get_embedding_dimension()
            else:
                actual_dim = self.client.get_sentence_embedding_dimension()

            if self.dim and actual_dim != self.dim:
                print(
                    f"[warn] requested embedding dim={self.dim}, "
                    f"but model {self.model} has dim={actual_dim}. "
                    f"Using actual dim={actual_dim}."
                )
                self.dim = actual_dim
            return

        raise ValueError(f"Unsupported embedding provider: {provider}")

    def embed_one(self, text: str) -> List[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: Iterable[str]) -> List[List[float]]:
        if self.provider == "nvidia":
            # Default to passage for memory/document rows.
            return self.embed_many_typed(texts, input_type="passage")

        texts = list(texts)
        out: List[Optional[List[float]]] = [None] * len(texts)

        missing = []
        missing_indices = []

        for i, text in enumerate(texts):
            key = text_cache_key(self.provider, self.model, self.dim, text)
            cached = self.cache.get(key)
            if cached is not None:
                out[i] = cached
            else:
                missing.append((key, text))
                missing_indices.append(i)

        if missing:
            if self.provider == "hash":
                for idx, (key, text) in zip(missing_indices, missing):
                    emb = hash_embed(text, self.dim)
                    self.cache.set(key, emb)
                    out[idx] = emb

            elif self.provider == "openai":
                for start in range(0, len(missing), self.batch_size):
                    batch = missing[start:start + self.batch_size]
                    batch_texts = [x[1] for x in batch]

                    resp = self.client.embeddings.create(
                        model=self.model,
                        input=batch_texts,
                    )

                    for local_i, item in enumerate(resp.data):
                        key = batch[local_i][0]
                        emb = item.embedding

                        if self.dim and len(emb) != self.dim:
                            raise ValueError(
                                f"Embedding dimension mismatch: got {len(emb)}, expected {self.dim}"
                            )

                        self.cache.set(key, emb)
                        out[missing_indices[start + local_i]] = emb

            elif self.provider == "sentence_transformers":
                for start in range(0, len(missing), self.batch_size):
                    batch = missing[start:start + self.batch_size]
                    batch_texts = [x[1] for x in batch]

                    embeddings = self.client.encode(
                        batch_texts,
                        batch_size=self.batch_size,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )

                    for local_i, emb_arr in enumerate(embeddings):
                        key = batch[local_i][0]
                        emb = emb_arr.tolist()
                        self.cache.set(key, emb)
                        out[missing_indices[start + local_i]] = emb

        return [x for x in out if x is not None]

    def embed_many_typed(
        self,
        texts: Iterable[str],
        input_type: str = "passage",
    ) -> List[List[float]]:
        """
        NVIDIA Nemotron-style embedding path.

        Use:
        - input_type="passage" for memories/documents
        - input_type="query" for questions
        """
        if self.provider != "nvidia":
            return self.embed_many(texts)

        texts = list(texts)
        out: List[Optional[List[float]]] = [None] * len(texts)

        typed_model_key = f"{self.model}:input_type={input_type}"

        missing = []
        missing_indices = []

        for i, text in enumerate(texts):
            key = text_cache_key(self.provider, typed_model_key, self.dim, text)
            cached = self.cache.get(key)
            if cached is not None:
                out[i] = cached
            else:
                missing.append((key, text))
                missing_indices.append(i)

        for start in range(0, len(missing), self.batch_size):
            batch = missing[start:start + self.batch_size]
            batch_texts = [x[1] for x in batch]

            resp = self.client.embeddings.create(
                model=self.model,
                input=batch_texts,
                encoding_format="float",
                extra_body={
                    "input_type": input_type,
                    "truncate": "NONE",
                },
            )

            for local_i, item in enumerate(resp.data):
                key = batch[local_i][0]
                emb = item.embedding

                if self.dim and len(emb) != self.dim:
                    raise ValueError(
                        f"Embedding dimension mismatch: got {len(emb)}, expected {self.dim}. "
                        f"Run the dimension probe and pass the correct --embedding-dim."
                    )

                self.cache.set(key, emb)
                out[missing_indices[start + local_i]] = emb

        return [x for x in out if x is not None]
