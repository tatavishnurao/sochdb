#!/usr/bin/env bash
set -euo pipefail

SOCHDB_HOST="${SOCHDB_HOST:-100.109.115.80}"
SOCHDB_PORT="${SOCHDB_PORT:-50051}"

MEMORIES="benchmarks/paper/locomo/data/locomo_memories.jsonl"
QUESTIONS_FULL="benchmarks/paper/locomo/data/locomo_questions.jsonl"
QUESTIONS_SMOKE="benchmarks/paper/locomo/data/smoke/locomo_questions_25.jsonl"

echo "Using SochDB target: ${SOCHDB_HOST}:${SOCHDB_PORT}"

echo
echo "== TCP connectivity check =="
uv run python - <<PY
import socket
host = "${SOCHDB_HOST}"
port = int("${SOCHDB_PORT}")
s = socket.create_connection((host, port), timeout=10)
print("TCP connected:", host, port)
s.close()
PY

echo
echo "== A1. Raw SochDB hash smoke k3 =="
uv run python benchmarks/paper/locomo/runners/run_sochdb_locomo_retrieval.py \
  --memories "$MEMORIES" \
  --questions "$QUESTIONS_SMOKE" \
  --host "$SOCHDB_HOST" \
  --port "$SOCHDB_PORT" \
  --collection-prefix locomo_sochdb_raw_hash_smoke_k3 \
  --embedding-dim 1536 \
  --k 3 \
  --out benchmarks/paper/locomo/results/smoke_sochdb_raw_hash_k3/retrieval_25.jsonl

uv run python benchmarks/paper/locomo/runners/judge_locomo_answers.py \
  --retrieval benchmarks/paper/locomo/results/smoke_sochdb_raw_hash_k3/retrieval_25.jsonl \
  --mode retrieval \
  --out-dir benchmarks/paper/locomo/results/smoke_sochdb_raw_hash_k3_scored

echo
echo "== B1. Hybrid SochDB + BGE-small smoke k10 =="
uv run python benchmarks/paper/locomo/runners/run_hybrid_locomo_retrieval.py \
  --memories "$MEMORIES" \
  --questions "$QUESTIONS_SMOKE" \
  --embedding-provider sentence_transformers \
  --embedding-model BAAI/bge-small-en-v1.5 \
  --embedding-dim 384 \
  --embedding-cache benchmarks/paper/locomo/data/embedding_cache_bge_small.jsonl \
  --vector-backend sochdb \
  --host "$SOCHDB_HOST" \
  --port "$SOCHDB_PORT" \
  --collection-prefix locomo_hybrid_sochdb_bge_small_smoke \
  --k 10 \
  --candidate-k 100 \
  --bm25-weight 1.5 \
  --vector-weight 0.75 \
  --out benchmarks/paper/locomo/results/smoke_hybrid_sochdb_bge_small_k10/retrieval_25.jsonl

uv run python benchmarks/paper/locomo/runners/judge_locomo_answers.py \
  --retrieval benchmarks/paper/locomo/results/smoke_hybrid_sochdb_bge_small_k10/retrieval_25.jsonl \
  --mode retrieval \
  --out-dir benchmarks/paper/locomo/results/smoke_hybrid_sochdb_bge_small_k10_scored

echo
echo "== Summary table =="
python - <<'PY'
import json
from pathlib import Path

paths = {
    "raw_sochdb_hash_smoke_k3": "benchmarks/paper/locomo/results/smoke_sochdb_raw_hash_k3_scored/summary.json",
    "hybrid_sochdb_bge_smoke_k10": "benchmarks/paper/locomo/results/smoke_hybrid_sochdb_bge_small_k10_scored/summary.json",
}

print("| system | n | hit | recall | tokens | avg_ms | p95_ms |")
print("|---|---:|---:|---:|---:|---:|---:|")

for name, path in paths.items():
    p = Path(path)
    if not p.exists():
        continue
    s = json.load(open(p))
    print(
        f"| {name} | "
        f"{s['n_questions']} | "
        f"{s['evidence_hit_rate']:.4f} | "
        f"{s['evidence_recall']:.4f} | "
        f"{s['avg_context_tokens']:.1f} | "
        f"{s['avg_latency_ms']:.2f} | "
        f"{s.get('p95_latency_ms', 0):.2f} |"
    )
PY
