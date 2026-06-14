#!/usr/bin/env python3

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}") from e

    return rows


def parse_port(value: Optional[str]) -> int:
    if value is None or str(value).strip() == "":
        return 50051
    return int(value)


def split_memory_and_questions(rows):
    memories = []
    questions = []

    for row in rows:
        if "question" in row:
            questions.append(row)
        else:
            memories.append(row)

    return memories, questions


def normalize_text(text: str) -> str:
    chars = []

    for ch in text.lower():
        if ch.isalnum() or ch.isspace():
            chars.append(ch)
        else:
            chars.append(" ")

    return " ".join("".join(chars).split())


def stable_hash(token: str) -> int:
    """
    Stable FNV-1a hash so embeddings are reproducible across runs.
    """
    h = 0xCBF29CE484222325

    for b in token.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF

    return h


def embed_text(text: str, dim: int) -> List[float]:
    """
    Deterministic local hash-bag embedding.

    This is NOT a semantic embedding model.
    It is only used to test the real SochDB SDK/gRPC retrieval path
    without relying on external embedding APIs.
    """
    vec = [0.0] * dim

    for token in normalize_text(text).split():
        idx = stable_hash(token) % dim
        vec[idx] += 1.0

    norm = math.sqrt(sum(x * x for x in vec))

    if norm > 0.0:
        vec = [x / norm for x in vec]

    return vec


def approx_tokens(text: str) -> int:
    return len(text.split())


def import_sochdb_client():
    import sochdb

    print("Loaded sochdb from:", sochdb.__file__)

    if not hasattr(sochdb, "SochDBClient"):
        exports = [x for x in dir(sochdb) if not x.startswith("_")]
        raise ImportError(
            "Imported `sochdb`, but it does not export SochDBClient.\n"
            f"Loaded from: {sochdb.__file__}\n"
            f"Exports: {exports}\n\n"
            "Install the real SDK with:\n"
            "uv pip install git+https://github.com/sochdb/sochdb-python-sdk.git"
        )

    return sochdb.SochDBClient


def create_client(SochDBClient, host: str, port: int, use_tls: bool):
    address = f"{host}:{port}"

    # Your installed SDK signature is:
    # SochDBClient(address: str = "localhost:50051", secure: bool = False)
    return SochDBClient(address=address, secure=use_tls)


def get_turn(memory: Dict[str, Any]) -> int:
    if memory.get("turn") is None:
        raise ValueError(f"Memory row missing `turn`: {memory}")
    return int(memory["turn"])


def build_memory_tables(
    memories: List[Dict[str, Any]],
    dim: int,
) -> Tuple[List[int], List[List[float]], Dict[int, Dict[str, Any]]]:
    ids = []
    vectors = []
    id_to_memory = {}

    for memory in memories:
        turn_id = get_turn(memory)
        text = memory.get("text", "")

        ids.append(turn_id)
        vectors.append(embed_text(text, dim))
        id_to_memory[turn_id] = memory

    return ids, vectors, id_to_memory


def try_create_index(client, collection: str, dim: int):
    attempts = [
        ("create_index(dimension=dim)", lambda: client.create_index(dimension=dim)),
        (
            "create_index(name=collection, dimension=dim)",
            lambda: client.create_index(name=collection, dimension=dim),
        ),
        (
            "create_index(index_name=collection, dimension=dim)",
            lambda: client.create_index(index_name=collection, dimension=dim),
        ),
        (
            "create_index(collection, dimension=dim)",
            lambda: client.create_index(collection, dimension=dim),
        ),
    ]

    last_error = None

    for label, fn in attempts:
        try:
            result = fn()
            print(f"{label}: ok")
            return result
        except TypeError as e:
            last_error = e
        except Exception as e:
            print(f"[warn] {label} failed, continuing: {e}")
            return None

    raise TypeError(f"Could not call create_index with known patterns. Last error: {last_error}")


def try_insert_vectors(client, collection: str, ids: List[int], vectors: List[List[float]]):
    """
    Try common SDK insert signatures.

    Best case: SDK accepts ids + vectors.
    If it only accepts vectors, we still run, but returned IDs may be server-assigned.
    """
    attempts = [
        (
            "insert_vectors(ids=ids, vectors=vectors)",
            lambda: client.insert_vectors(ids=ids, vectors=vectors),
        ),
        (
            "insert_vectors(vectors=vectors, ids=ids)",
            lambda: client.insert_vectors(vectors=vectors, ids=ids),
        ),
        (
            "insert_vectors(collection=collection, ids=ids, vectors=vectors)",
            lambda: client.insert_vectors(collection=collection, ids=ids, vectors=vectors),
        ),
        (
            "insert_vectors(index_name=collection, ids=ids, vectors=vectors)",
            lambda: client.insert_vectors(index_name=collection, ids=ids, vectors=vectors),
        ),
        (
            "insert_vectors(collection, ids, vectors)",
            lambda: client.insert_vectors(collection, ids, vectors),
        ),
        (
            "insert_vectors(collection, vectors)",
            lambda: client.insert_vectors(collection, vectors),
        ),
        (
            "insert_vectors(vectors)",
            lambda: client.insert_vectors(vectors),
        ),
    ]

    last_error = None

    for label, fn in attempts:
        try:
            result = fn()
            print(f"{label}: ok")
            return result, label
        except TypeError as e:
            last_error = e

    raise TypeError(f"Could not call insert_vectors with known patterns. Last error: {last_error}")


def try_search(client, collection: str, query_vector: List[float], k: int):
    attempts = [
        (
            "search(query_vector, k=k)",
            lambda: client.search(query_vector, k=k),
        ),
        (
            "search(query=query_vector, k=k)",
            lambda: client.search(query=query_vector, k=k),
        ),
        (
            "search(collection, query_vector, k=k)",
            lambda: client.search(collection, query_vector, k=k),
        ),
        (
            "search(collection=collection, query=query_vector, k=k)",
            lambda: client.search(collection=collection, query=query_vector, k=k),
        ),
        (
            "search(index_name=collection, query=query_vector, k=k)",
            lambda: client.search(index_name=collection, query=query_vector, k=k),
        ),
    ]

    last_error = None

    for label, fn in attempts:
        try:
            return fn(), label
        except TypeError as e:
            last_error = e

    raise TypeError(f"Could not call search with known patterns. Last error: {last_error}")


def result_to_dict(item: Any) -> Dict[str, Any]:
    """
    Normalize SDK SearchResult / dict / tuple outputs.
    """
    if isinstance(item, dict):
        return item

    if isinstance(item, tuple):
        # Common shapes: (id, distance), (id, score), (id, distance, metadata)
        out = {}
        if len(item) >= 1:
            out["id"] = item[0]
        if len(item) >= 2:
            out["score"] = item[1]
        if len(item) >= 3:
            out["metadata"] = item[2]
        return out

    out = {}

    for attr in ["id", "vector_id", "doc_id", "key"]:
        if hasattr(item, attr):
            out["id"] = getattr(item, attr)
            break

    for attr in ["score", "distance", "similarity"]:
        if hasattr(item, attr):
            out[attr] = getattr(item, attr)

    for attr in ["text", "content", "document"]:
        if hasattr(item, attr):
            out["text"] = getattr(item, attr)
            break

    for attr in ["metadata", "meta"]:
        if hasattr(item, attr):
            out["metadata"] = getattr(item, attr)
            break

    return out


def normalize_results(raw_results: Any) -> List[Dict[str, Any]]:
    """
    Handle SDK response wrappers.
    """
    if raw_results is None:
        return []

    if isinstance(raw_results, list):
        return [result_to_dict(x) for x in raw_results]

    if isinstance(raw_results, tuple):
        return [result_to_dict(x) for x in raw_results]

    for attr in ["results", "items", "hits", "matches"]:
        if hasattr(raw_results, attr):
            value = getattr(raw_results, attr)
            return [result_to_dict(x) for x in value]

    if isinstance(raw_results, dict):
        for key in ["results", "items", "hits", "matches"]:
            if key in raw_results:
                return [result_to_dict(x) for x in raw_results[key]]
        return [raw_results]

    return [result_to_dict(raw_results)]


def extract_turn_id(item: Dict[str, Any]) -> Optional[int]:
    """
    Prefer explicit ID. Fall back to metadata.
    """
    for key in ["id", "vector_id", "doc_id", "key"]:
        if key in item and item[key] is not None:
            raw = str(item[key])
            if raw.startswith("turn_"):
                raw = raw.replace("turn_", "", 1)
            try:
                return int(raw)
            except Exception:
                pass

    metadata = item.get("metadata") or item.get("meta") or {}

    if isinstance(metadata, dict):
        for key in ["turn", "turn_id", "source_turn"]:
            if key in metadata:
                try:
                    return int(metadata[key])
                except Exception:
                    pass

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--host", default=os.getenv("SOCHDB_HOST", "65.108.78.80"))
    parser.add_argument("--port", type=parse_port, default=parse_port(os.getenv("SOCHDB_PORT")))
    parser.add_argument("--collection", default=os.getenv("SOCHDB_COLLECTION", "agent_memory_qa_v1"))
    parser.add_argument("--embedding-dim", type=int, default=1536)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--use-tls", action="store_true")
    args = parser.parse_args()

    SochDBClient = import_sochdb_client()
    client = create_client(SochDBClient, args.host, args.port, args.use_tls)

    print(f"Connected to SochDB SDK target: {args.host}:{args.port}")
    print(f"Collection/index: {args.collection}")
    print(f"Embedding dim: {args.embedding_dim}")
    print(f"k: {args.k}")

    rows = load_jsonl(args.data)
    memories, questions = split_memory_and_questions(rows)

    if not memories:
        raise ValueError("No memory rows found.")
    if not questions:
        raise ValueError("No question rows found.")

    print(f"Memories: {len(memories)}")
    print(f"Questions: {len(questions)}")

    ids, vectors, id_to_memory = build_memory_tables(memories, args.embedding_dim)

    try_create_index(client, args.collection, args.embedding_dim)
    _, insert_pattern = try_insert_vectors(client, args.collection, ids, vectors)

    print(f"Inserted vectors using pattern: {insert_pattern}")

    output_rows = []

    for q in questions:
        question_id = q["question_id"]
        question = q["question"]
        gold_answer = q.get("answer", "")
        evidence_turns = q.get("evidence_turns", [])

        query_vector = embed_text(question, args.embedding_dim)

        start = time.perf_counter()
        raw_results, search_pattern = try_search(client, args.collection, query_vector, args.k)
        latency_ms = (time.perf_counter() - start) * 1000.0

        normalized = normalize_results(raw_results)

        retrieved_turns = []
        context_parts = []

        for item in normalized:
            turn_id = extract_turn_id(item)

            if turn_id is not None:
                retrieved_turns.append(turn_id)

                memory = id_to_memory.get(turn_id)
                if memory:
                    context_parts.append(memory.get("text", ""))
                else:
                    text = item.get("text") or item.get("content") or ""
                    if text:
                        context_parts.append(text)
                    else:
                        context_parts.append(f"[missing local text for turn {turn_id}]")
            else:
                text = item.get("text") or item.get("content") or ""
                if text:
                    context_parts.append(text)

        debug_context = "\n".join(context_parts)

        output_rows.append(
            {
                "system": "sochdb_sdk",
                "question_id": question_id,
                "question": question,
                "gold_answer": gold_answer,
                "retrieved_turns": retrieved_turns,
                "evidence_turns": evidence_turns,
                "retrieved_count": len(normalized),
                "approx_context_tokens": approx_tokens(debug_context),
                "latency_ms": latency_ms,
                "search_pattern": search_pattern,
                "debug_context": debug_context,
                "type": q.get("type", "unknown"),
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote retrieval results to {out_path}")

    close = getattr(client, "close", None)
    if callable(close):
        close()


if __name__ == "__main__":
    main()
