#!/usr/bin/env python3

import argparse
import json
import math
import os
import re
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


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def embed_text(text: str, dim: int) -> List[float]:
    vec = [0.0] * dim

    for token in normalize_text(text).split():
        vec[stable_hash(token) % dim] += 1.0

    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0.0:
        vec = [x / norm for x in vec]

    return vec


def approx_tokens(text: str) -> int:
    return len(text.split())


def safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(s))[:80]


def import_sochdb_client():
    import sochdb

    print("Loaded sochdb from:", sochdb.__file__)

    if not hasattr(sochdb, "SochDBClient"):
        exports = [x for x in dir(sochdb) if not x.startswith("_")]
        raise ImportError(
            "Installed `sochdb` does not export SochDBClient.\n"
            f"Loaded from: {sochdb.__file__}\n"
            f"Exports: {exports}\n"
            "Install SDK:\n"
            "uv pip install git+https://github.com/sochdb/sochdb-python-sdk.git"
        )

    return sochdb.SochDBClient


def create_client(SochDBClient, host: str, port: int, secure: bool):
    return SochDBClient(address=f"{host}:{port}", secure=secure)


def try_create_index(client, index_name: str, dim: int):
    attempts = [
        ("create_index(name=index_name, dimension=dim)", lambda: client.create_index(name=index_name, dimension=dim)),
        ("create_index(index_name=index_name, dimension=dim)", lambda: client.create_index(index_name=index_name, dimension=dim)),
        ("create_index(index_name, dimension=dim)", lambda: client.create_index(index_name, dimension=dim)),
        ("create_index(dimension=dim)", lambda: client.create_index(dimension=dim)),
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
            print(f"[warn] {label} failed/exists, continuing: {e}")
            return None

    raise TypeError(f"Could not call create_index. Last TypeError: {last_error}")


def try_insert_vectors(client, index_name: str, ids: List[int], vectors: List[List[float]]):
    attempts = [
        ("insert_vectors(index_name=index_name, ids=ids, vectors=vectors)", lambda: client.insert_vectors(index_name=index_name, ids=ids, vectors=vectors)),
        ("insert_vectors(name=index_name, ids=ids, vectors=vectors)", lambda: client.insert_vectors(name=index_name, ids=ids, vectors=vectors)),
        ("insert_vectors(collection=index_name, ids=ids, vectors=vectors)", lambda: client.insert_vectors(collection=index_name, ids=ids, vectors=vectors)),
        ("insert_vectors(ids=ids, vectors=vectors)", lambda: client.insert_vectors(ids=ids, vectors=vectors)),
        ("insert_vectors(index_name, ids, vectors)", lambda: client.insert_vectors(index_name, ids, vectors)),
        ("insert_vectors(index_name, vectors)", lambda: client.insert_vectors(index_name, vectors)),
        ("insert_vectors(vectors)", lambda: client.insert_vectors(vectors)),
    ]

    last_error = None

    for label, fn in attempts:
        try:
            result = fn()
            return result, label
        except TypeError as e:
            last_error = e

    raise TypeError(f"Could not call insert_vectors. Last TypeError: {last_error}")


def try_search(client, index_name: str, query_vec: List[float], k: int):
    attempts = [
        ("search(index_name=index_name, query=query_vec, k=k)", lambda: client.search(index_name=index_name, query=query_vec, k=k)),
        ("search(name=index_name, query=query_vec, k=k)", lambda: client.search(name=index_name, query=query_vec, k=k)),
        ("search(collection=index_name, query=query_vec, k=k)", lambda: client.search(collection=index_name, query=query_vec, k=k)),
        ("search(index_name, query_vec, k=k)", lambda: client.search(index_name, query_vec, k=k)),
        ("search(query_vec, k=k)", lambda: client.search(query_vec, k=k)),
        ("search(query=query_vec, k=k)", lambda: client.search(query=query_vec, k=k)),
    ]

    last_error = None

    for label, fn in attempts:
        try:
            return fn(), label
        except TypeError as e:
            last_error = e

    raise TypeError(f"Could not call search. Last TypeError: {last_error}")


def result_to_dict(item: Any) -> Dict[str, Any]:
    if isinstance(item, dict):
        return item

    if isinstance(item, tuple):
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
    if raw_results is None:
        return []

    if isinstance(raw_results, list):
        return [result_to_dict(x) for x in raw_results]

    if isinstance(raw_results, tuple):
        return [result_to_dict(x) for x in raw_results]

    if isinstance(raw_results, dict):
        for key in ["results", "items", "hits", "matches"]:
            if key in raw_results:
                return [result_to_dict(x) for x in raw_results[key]]
        return [raw_results]

    for attr in ["results", "items", "hits", "matches"]:
        if hasattr(raw_results, attr):
            value = getattr(raw_results, attr)
            return [result_to_dict(x) for x in value]

    return [result_to_dict(raw_results)]


def extract_memory_id(item: Dict[str, Any]) -> Optional[int]:
    for key in ["id", "vector_id", "doc_id", "key"]:
        if key in item and item[key] is not None:
            raw = str(item[key])
            try:
                return int(raw)
            except Exception:
                pass

    metadata = item.get("metadata") or item.get("meta") or {}
    if isinstance(metadata, dict):
        for key in ["memory_id", "turn", "id"]:
            if key in metadata:
                try:
                    return int(metadata[key])
                except Exception:
                    pass

    return None


def group_by_sample(memories: List[Dict[str, Any]], questions: List[Dict[str, Any]]):
    memories_by_sample: Dict[str, List[Dict[str, Any]]] = {}
    questions_by_sample: Dict[str, List[Dict[str, Any]]] = {}

    for m in memories:
        memories_by_sample.setdefault(m["sample_id"], []).append(m)

    for q in questions:
        questions_by_sample.setdefault(q["sample_id"], []).append(q)

    return memories_by_sample, questions_by_sample


def parse_port(value: Optional[str]) -> int:
    if value is None or str(value).strip() == "":
        return 50051
    return int(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memories", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--host", default=os.getenv("SOCHDB_HOST", "65.108.78.80"))
    parser.add_argument("--port", type=parse_port, default=parse_port(os.getenv("SOCHDB_PORT")))
    parser.add_argument("--collection-prefix", default=os.getenv("SOCHDB_COLLECTION", "locomo_sochdb"))
    parser.add_argument("--embedding-dim", type=int, default=1536)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--use-tls", action="store_true")
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--limit-questions", type=int, default=None)
    parser.add_argument("--run-id", default=str(int(time.time())))
    args = parser.parse_args()

    SochDBClient = import_sochdb_client()
    client = create_client(SochDBClient, args.host, args.port, args.use_tls)

    memories = load_jsonl(args.memories)
    questions = load_jsonl(args.questions)

    if args.limit_questions is not None:
        questions = questions[: args.limit_questions]

    memories_by_sample, questions_by_sample = group_by_sample(memories, questions)

    sample_ids = sorted(questions_by_sample.keys())
    if args.limit_samples is not None:
        sample_ids = sample_ids[: args.limit_samples]

    print(f"Target: {args.host}:{args.port}")
    print(f"k: {args.k}")
    print(f"Samples to run: {len(sample_ids)}")
    print(f"Questions total selected: {sum(len(questions_by_sample[s]) for s in sample_ids)}")

    output_rows = []

    for sample_id in sample_ids:
        sample_memories = memories_by_sample.get(sample_id, [])
        sample_questions = questions_by_sample.get(sample_id, [])

        if not sample_memories or not sample_questions:
            continue

        index_name = f"{args.collection_prefix}_{safe_name(sample_id)}_{args.run_id}"

        print(f"\n=== sample={sample_id} index={index_name} memories={len(sample_memories)} questions={len(sample_questions)} ===")

        ids = []
        vectors = []
        id_to_memory = {}

        for m in sample_memories:
            mid = int(m["memory_id"])
            ids.append(mid)
            vectors.append(embed_text(m["text"], args.embedding_dim))
            id_to_memory[mid] = m

        try_create_index(client, index_name, args.embedding_dim)
        _, insert_label = try_insert_vectors(client, index_name, ids, vectors)
        print(f"Inserted {len(ids)} vectors using: {insert_label}")

        for q in sample_questions:
            query_vec = embed_text(q["question"], args.embedding_dim)

            start = time.perf_counter()
            raw, search_label = try_search(client, index_name, query_vec, args.k)
            latency_ms = (time.perf_counter() - start) * 1000.0

            hits = normalize_results(raw)

            retrieved_memory_ids = []
            context_parts = []

            for hit in hits:
                mid = extract_memory_id(hit)
                if mid is None:
                    continue

                retrieved_memory_ids.append(mid)
                memory = id_to_memory.get(mid)

                if memory:
                    context_parts.append(
                        f"[memory_id={mid} sample_id={memory['sample_id']} "
                        f"session={memory.get('session')} dia_id={memory.get('dia_id')} "
                        f"speaker={memory.get('speaker')}] {memory.get('text', '')}"
                    )

            debug_context = "\n".join(context_parts)

            evidence_memory_ids = q.get("evidence_memory_ids") or []

            output_rows.append(
                {
                    "system": "sochdb_sdk",
                    "question_id": q["question_id"],
                    "sample_id": q["sample_id"],
                    "question": q["question"],
                    "gold_answer": q.get("gold_answer", ""),
                    "category": q.get("category", "unknown"),
                    "category_id": q.get("category_id"),
                    "evidence_refs": q.get("evidence_refs", []),
                    "evidence_memory_ids": evidence_memory_ids,
                    "retrieved_memory_ids": retrieved_memory_ids,
                    "retrieved_count": len(retrieved_memory_ids),
                    "approx_context_tokens": approx_tokens(debug_context),
                    "latency_ms": latency_ms,
                    "search_pattern": search_label,
                    "debug_context": debug_context,
                }
            )

    write_jsonl(args.out, output_rows)
    print(f"\nWrote retrieval rows: {args.out} ({len(output_rows)} rows)")

    close = getattr(client, "close", None)
    if callable(close):
        close()


if __name__ == "__main__":
    main()
