#!/usr/bin/env python3
"""
Hybrid retrieval benchmark: Grep (lexical) + HNSW (dense)
fused with Reciprocal Rank Fusion (RRF).

Mirrors the target wiring diagram:
  Query/agent -> [Grep | HNSW] -> RRF fusion -> Top-k results

Note: BM25 leg will be added once sochdb-fusion token postings
are exposed through the Python SDK. Currently only Grep + HNSW
legs are active.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lexical_index import LexicalIndex
from rrf import rrf_fuse

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = ROOT / "results" / "sochdb_hybrid.json"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=np.float64), p))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True,
                        help="Path to corpus.jsonl")
    parser.add_argument("--queries", type=Path, required=True,
                        help="Path to queries.jsonl")
    parser.add_argument("--doc-embeddings", type=Path, required=True,
                        help="Path to doc_embeddings.npy")
    parser.add_argument("--query-embeddings", type=Path, required=True,
                        help="Path to query_embeddings.npy")
    parser.add_argument("--doc-ids", type=Path, required=True,
                        help="Path to doc_ids.json")
    parser.add_argument("--query-ids", type=Path, required=True,
                        help="Path to query_ids.json")
    parser.add_argument("--embedding-metadata", type=Path, default=None,
                        help="Path to embedding_metadata.json (optional)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--k", type=int, default=10,
                        help="Top-k per leg before fusion")
    parser.add_argument("--top-n", type=int, default=5,
                        help="Final fused top-n")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF constant")
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=100)
    parser.add_argument("--precision", default="f32",
                        choices=["f32", "f16", "bf16"])
    parser.add_argument("--no-grep", action="store_true",
                        help="Disable lexical leg")
    parser.add_argument("--no-hnsw", action="store_true",
                        help="Disable dense leg")
    args = parser.parse_args()

    try:
        from sochdb import HnswIndex
    except ImportError:
        print("Error: sochdb package not installed. Install with:")
        print("  uv pip install -e ./sochdb-python")
        sys.exit(1)

    corpus = load_jsonl(args.corpus)
    queries = load_jsonl(args.queries)

    doc_embeddings = np.load(args.doc_embeddings)
    query_embeddings = np.load(args.query_embeddings)
    doc_ids = json.loads(args.doc_ids.read_text())
    query_ids = json.loads(args.query_ids.read_text())

    metadata = {}
    if args.embedding_metadata and args.embedding_metadata.exists():
        metadata = json.loads(args.embedding_metadata.read_text())

    id_to_record = {r["id"]: r for r in corpus}

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # -- Leg 1: Build Lexical (Grep + Trigram) Index --
    lexical_index = LexicalIndex()
    if not args.no_grep:
        print(f"Building lexical index for {len(corpus)} docs...")
        lex_build_start = time.perf_counter()
        for record in corpus:
            full_text = f"{record.get('title', '')} {record.get('body', '')}"
            lexical_index.add(record["id"], full_text)
        lexical_index.build()
        lex_build_elapsed = time.perf_counter() - lex_build_start
        print(f"Lexical index built in {lex_build_elapsed*1000:.1f}ms")

    # -- Leg 2: Build HNSW (Dense) Index --
    hnsw_index = None
    numeric_to_doc_id: dict[int, str] = {}
    if not args.no_hnsw:
        dimension = int(doc_embeddings.shape[1])
        hnsw_index = HnswIndex(
            dimension=dimension,
            m=args.m,
            ef_construction=args.ef_construction,
            metric="cosine",
            precision=args.precision,
        )
        print(f"Building HNSW index for {len(doc_ids)} vectors...")
        hnsw_build_start = time.perf_counter()
        numeric_ids = np.arange(1, len(doc_ids) + 1, dtype=np.uint64)
        numeric_to_doc_id = {
            int(nid): did for nid, did in zip(numeric_ids.tolist(), doc_ids)
        }
        hnsw_index.insert_batch_with_ids(numeric_ids, doc_embeddings.astype(np.float32))
        hnsw_build_elapsed = time.perf_counter() - hnsw_build_start
        print(f"HNSW index built in {hnsw_build_elapsed*1000:.1f}ms")

    # -- Query Loop --
    query_timings: list[float] = []
    query_results: list[dict[str, Any]] = []

    print(f"Running {len(queries)} hybrid queries...")
    for q_record, q_id, q_vec in zip(queries, query_ids, query_embeddings):
        q_start = time.perf_counter()
        ranked_lists: list[list[str]] = []

        # Grep leg
        if not args.no_grep:
            lex_hits = lexical_index.search(q_record["query"], k=args.k * 4)
            ranked_lists.append([doc_id for doc_id, _ in lex_hits])

        # HNSW leg
        if hnsw_index is not None:
            result_ids, _ = hnsw_index.search(q_vec.astype(np.float32), k=args.k * 4)
            hnsw_ranked = [
                numeric_to_doc_id[int(nid)]
                for nid in result_ids.tolist()
                if int(nid) in numeric_to_doc_id
            ]
            ranked_lists.append(hnsw_ranked)

        # RRF Fusion
        fused = rrf_fuse(ranked_lists, k=args.rrf_k, top_n=args.top_n)

        q_elapsed = time.perf_counter() - q_start
        query_timings.append(q_elapsed)

        query_results.append({
            "query_id": q_id,
            "query": q_record["query"],
            "relevant_ids": q_record.get("relevant_ids", []),
            "results": [
                {
                    "doc_id": doc_id,
                    "rrf_score": round(score, 6),
                    "title": id_to_record.get(doc_id, {}).get("title"),
                }
                for doc_id, score in fused
            ],
            "latency_ms": round(q_elapsed * 1000, 3),
        })

    output = {
        "system": "sochdb_hybrid",
        "legs": {
            "grep_lexical": not args.no_grep,
            "hnsw_dense": not args.no_hnsw,
        },
        "rrf_k": args.rrf_k,
        "corpus_size": len(corpus),
        "query_count": len(queries),
        "embedding_model": metadata.get("model_name", "unknown"),
        "top_k_per_leg": args.k,
        "top_n_fused": args.top_n,
        "query_latency": {
            "p50_ms": round(percentile(query_timings, 50) * 1000, 3),
            "p95_ms": round(percentile(query_timings, 95) * 1000, 3),
            "mean_ms": round(float(np.mean(query_timings)) * 1000, 3),
        },
        "queries": query_results,
    }

    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()