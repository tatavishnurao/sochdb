#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
LOCOMO_RUNNERS = REPO_ROOT / "benchmarks" / "paper" / "locomo" / "runners"
if str(LOCOMO_RUNNERS) not in sys.path:
    sys.path.insert(0, str(LOCOMO_RUNNERS))

from embedding_utils import Embedder, normalize_text  # type: ignore
from sochdb_batch_client import search_batch as search_sochdb_batch  # type: ignore

from memory_schema import (
    approximate_tokens,
    group_by_sample,
    preflight_embedding,
    preflight_sochdb,
    read_json_or_jsonl,
    render_memory_text,
    require_file,
    require_safe_output_path,
    write_jsonl,
)


def tokenize(text: str) -> list[str]:
    return normalize_text(text).split()


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def safe_name(value: Any) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(value))[:80] or "sample"


def parse_port(value: str | None) -> int:
    if value is None or str(value).strip() == "":
        return 50051
    return int(value)


def topk_local_vector(
    query_vec: list[float],
    vector_ids: list[int],
    memory_vecs: list[list[float]],
    k: int,
) -> list[int]:
    scored = [(cosine(query_vec, vec), vid) for vid, vec in zip(vector_ids, memory_vecs)]
    scored.sort(reverse=True)
    return [vid for _, vid in scored[:k]]


def rrf_fuse_with_scores(
    bm25_ranked: list[int],
    vector_ranked: list[int],
    final_k: int,
    rrf_k: int,
    bm25_weight: float,
    vector_weight: float,
) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for rank, vid in enumerate(bm25_ranked, start=1):
        scores[vid] = scores.get(vid, 0.0) + bm25_weight / (rrf_k + rank)
    for rank, vid in enumerate(vector_ranked, start=1):
        scores[vid] = scores.get(vid, 0.0) + vector_weight / (rrf_k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:final_k]


def import_sochdb_client():
    import sochdb

    if not hasattr(sochdb, "SochDBClient"):
        raise ImportError(
            "sochdb package does not export SochDBClient. Install the SochDB Python SDK."
        )
    return sochdb.SochDBClient


def create_sochdb_index(client, index_name: str, dim: int) -> None:
    try:
        client.create_index(name=index_name, dimension=dim)
    except Exception as exc:
        print(f"[warn] create_index failed/exists for {index_name}: {exc}")


def insert_sochdb_vectors(
    client,
    index_name: str,
    ids: list[int],
    vectors: list[list[float]],
) -> None:
    client.insert_vectors(index_name=index_name, ids=ids, vectors=vectors)


def _extract_result_id(item: Any) -> int | None:
    if isinstance(item, dict):
        raw_id = item.get("id") or item.get("vector_id")
    else:
        raw_id = getattr(item, "id", None) or getattr(item, "vector_id", None)
    if raw_id is None:
        return None
    try:
        return int(raw_id)
    except Exception:
        return None


def search_sochdb(client, index_name: str, query_vec: list[float], k: int) -> list[int]:
    raw = client.search(index_name=index_name, query=query_vec, k=k)
    if hasattr(raw, "results"):
        items = raw.results
    elif isinstance(raw, dict) and "results" in raw:
        items = raw["results"]
    else:
        items = raw
    return [vid for item in items if (vid := _extract_result_id(item)) is not None]


def search_sochdb_batch_ids(
    client,
    index_name: str,
    query_vecs: list[list[float]],
    k: int,
    ef: int = 0,
) -> list[list[int]]:
    raw_batches = search_sochdb_batch(
        client=client,
        index_name=index_name,
        queries=query_vecs,
        k=k,
        ef=ef,
    )
    out = []
    for items in raw_batches:
        out.append([vid for item in items if (vid := _extract_result_id(item)) is not None])
    return out


def build_context(
    vector_ids: list[int],
    vector_to_memory: dict[int, dict[str, Any]],
    vector_to_original_id: dict[int, str],
) -> str:
    parts = []
    for vid in vector_ids:
        memory = vector_to_memory.get(vid)
        if not memory:
            continue
        parts.append(
            f"[memory_id={vector_to_original_id.get(vid, vid)} "
            f"sample_id={memory.get('sample_id')} session={memory.get('session')} "
            f"turn_id={memory.get('turn_id')} speaker={memory.get('speaker')}] "
            f"{memory.get('text', '')}"
        )
    return "\n".join(parts)


def run_retrieval(args: argparse.Namespace, dataset_name: str) -> None:
    require_file(args.memories, "memories")
    require_file(args.questions, "questions")
    require_safe_output_path(args.out)
    preflight_embedding(args.embedding_provider, args.embedding_model, args.embedding_dim)
    preflight_sochdb(args.host, args.port, args.vector_backend == "sochdb")

    memories = read_json_or_jsonl(args.memories)
    questions = read_json_or_jsonl(args.questions)

    if args.limit_questions is not None:
        questions = questions[: args.limit_questions]

    memories_by_sample = group_by_sample(memories)
    questions_by_sample = group_by_sample(questions)
    sample_ids = sorted(questions_by_sample.keys())
    if args.limit_samples is not None:
        sample_ids = sample_ids[: args.limit_samples]

    embedder = Embedder(
        provider=args.embedding_provider,
        model=args.embedding_model,
        dim=args.embedding_dim,
        cache_path=args.embedding_cache,
    )

    client = None
    if args.vector_backend == "sochdb":
        SochDBClient = import_sochdb_client()
        client = SochDBClient(address=f"{args.host}:{args.port}", secure=args.use_tls)

    system_name = (
        f"{dataset_name}_hybrid_bm25_{args.vector_backend}_{args.embedding_provider}_"
        f"{args.embedding_model.replace('-', '_').replace('.', '_').replace('/', '_')}"
    )

    print(
        f"system={system_name} samples={len(sample_ids)} k={args.k} "
        f"candidate_k={args.candidate_k} query_mode={args.query_mode} "
        f"memory_render_mode={args.memory_render_mode} reranker_provider={args.reranker_provider}"
    )

    output_rows: list[dict[str, Any]] = []

    for sample_id in sample_ids:
        sample_memories = memories_by_sample.get(sample_id, [])
        sample_questions = questions_by_sample.get(sample_id, [])
        if not sample_memories or not sample_questions:
            continue

        vector_ids = list(range(1, len(sample_memories) + 1))
        vector_to_memory = dict(zip(vector_ids, sample_memories))
        vector_to_original_id = {
            vid: str(memory.get("memory_id")) for vid, memory in vector_to_memory.items()
        }

        memory_texts = [
            render_memory_text(memory, args.memory_render_mode) for memory in sample_memories
        ]
        bm25 = BM25Okapi([tokenize(text) for text in memory_texts])

        print(
            f"\n=== sample={sample_id} memories={len(sample_memories)} "
            f"questions={len(sample_questions)} ==="
        )
        print("embedding memories...")
        if args.embedding_provider == "nvidia":
            memory_vecs = embedder.embed_many_typed(memory_texts, input_type="passage")
        else:
            memory_vecs = embedder.embed_many(memory_texts)

        index_name = f"{args.collection_prefix}_{safe_name(sample_id)}_{args.run_id}"
        if args.vector_backend == "sochdb":
            create_sochdb_index(client, index_name, args.embedding_dim)
            insert_sochdb_vectors(client, index_name, vector_ids, memory_vecs)

        question_texts = [str(q.get("question", "")) for q in sample_questions]
        print("embedding questions...")
        if args.embedding_provider == "nvidia":
            question_vecs = embedder.embed_many_typed(question_texts, input_type="query")
        else:
            question_vecs = embedder.embed_many(question_texts)

        batch_vector_ranked: list[list[int]] | None = None
        if args.vector_backend == "sochdb" and args.sochdb_search_mode == "batch":
            batch_vector_ranked = search_sochdb_batch_ids(
                client, index_name, question_vecs, args.candidate_k, ef=args.sochdb_ef
            )

        for q_idx, (question, qvec) in enumerate(zip(sample_questions, question_vecs)):
            start = time.perf_counter()
            qtext = str(question.get("question", ""))
            bm25_scores = bm25.get_scores(tokenize(qtext))
            bm25_ranked_idx = sorted(
                range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
            )[: args.candidate_k]
            bm25_ranked = [vector_ids[i] for i in bm25_ranked_idx]

            if args.vector_backend == "local":
                vector_ranked = topk_local_vector(qvec, vector_ids, memory_vecs, args.candidate_k)
            elif args.sochdb_search_mode == "batch":
                vector_ranked = batch_vector_ranked[q_idx] if batch_vector_ranked else []
            else:
                vector_ranked = search_sochdb(client, index_name, qvec, args.candidate_k)

            fused = rrf_fuse_with_scores(
                bm25_ranked=bm25_ranked,
                vector_ranked=vector_ranked,
                final_k=args.k,
                rrf_k=args.rrf_k,
                bm25_weight=args.bm25_weight,
                vector_weight=args.vector_weight,
            )
            final_vector_ids = [vid for vid, _ in fused]
            final_original_ids = [vector_to_original_id[vid] for vid in final_vector_ids]
            context = build_context(final_vector_ids, vector_to_memory, vector_to_original_id)

            output_rows.append(
                {
                    "system": system_name,
                    "dataset": dataset_name,
                    "question_id": str(question.get("question_id", "")),
                    "sample_id": str(question.get("sample_id", sample_id)),
                    "question": qtext,
                    "gold_answer": question.get("answer") or question.get("gold_answer", ""),
                    "category": question.get("category", "unknown"),
                    "evidence_memory_ids": [
                        str(x) for x in (question.get("evidence_memory_ids") or [])
                    ],
                    "evidence_mapping_status": question.get("evidence_mapping_status"),
                    "evidence_mapping_failed_texts": question.get(
                        "evidence_mapping_failed_texts", []
                    ),
                    "retrieved_memory_ids": final_original_ids,
                    "retrieved_count": len(final_original_ids),
                    "requested_k": args.k,
                    "candidate_k": args.candidate_k,
                    "bm25_weight": args.bm25_weight,
                    "vector_weight": args.vector_weight,
                    "rrf_k": args.rrf_k,
                    "query_mode": args.query_mode,
                    "memory_render_mode": args.memory_render_mode,
                    "embedding_provider": args.embedding_provider,
                    "embedding_model": args.embedding_model,
                    "embedding_dim": args.embedding_dim,
                    "vector_backend": args.vector_backend,
                    "reranker_provider": args.reranker_provider,
                    "approx_context_tokens": approximate_tokens(context),
                    "latency_ms": (time.perf_counter() - start) * 1000.0,
                    "debug_context": context,
                }
            )

    write_jsonl(args.out, output_rows)
    print(f"\nWrote {len(output_rows)} rows to {args.out}")
    if client is not None and hasattr(client, "close"):
        client.close()


def build_arg_parser(dataset_name: str, default_cache: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memories", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--embedding-provider",
        choices=["hash", "openai", "sentence_transformers", "nvidia"],
        default="nvidia",
    )
    parser.add_argument("--embedding-model", default="nvidia/llama-nemotron-embed-1b-v2")
    parser.add_argument("--embedding-dim", type=int, default=2048)
    parser.add_argument("--embedding-cache", default=default_cache)
    parser.add_argument("--vector-backend", choices=["local", "sochdb"], default="sochdb")
    parser.add_argument("--host", default=os.getenv("SOCHDB_HOST", ""))
    parser.add_argument("--port", type=parse_port, default=parse_port(os.getenv("SOCHDB_PORT")))
    parser.add_argument("--collection-prefix", default=f"{dataset_name}_hybrid")
    parser.add_argument("--use-tls", action="store_true")
    parser.add_argument("--sochdb-search-mode", choices=["single", "batch"], default="single")
    parser.add_argument("--sochdb-ef", type=int, default=0)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--candidate-k", type=int, default=400)
    parser.add_argument("--bm25-weight", type=float, default=1.5)
    parser.add_argument("--vector-weight", type=float, default=0.75)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--memory-render-mode", choices=["raw", "metadata"], default="metadata")
    parser.add_argument("--query-mode", choices=["single", "multi"], default="single")
    parser.add_argument(
        "--reranker-provider",
        choices=["none", "sentence_transformers"],
        default="none",
    )
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--limit-questions", type=int, default=None)
    parser.add_argument("--run-id", default=str(int(time.time())))
    return parser


def main(dataset_name: str, default_cache: str) -> None:
    parser = build_arg_parser(dataset_name, default_cache)
    args = parser.parse_args()
    if args.query_mode != "single":
        raise SystemExit("Only --query-mode single is implemented for common retrieval runner.")
    if args.reranker_provider != "none":
        raise SystemExit("Only --reranker-provider none is implemented for common retrieval runner.")
    run_retrieval(args, dataset_name)
