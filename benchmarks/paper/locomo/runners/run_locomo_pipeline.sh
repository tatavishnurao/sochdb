#!/usr/bin/env bash
set -euo pipefail

cd /home/vishnu-rao/Desktop/Arjun/projects/sochdb

mkdir -p external
mkdir -p benchmarks/paper/locomo/{data,results,runners}

if [ ! -d external/locomo ]; then
  git clone https://github.com/snap-research/locomo.git external/locomo
fi

uv run python benchmarks/paper/locomo/runners/convert_locomo.py \
  --input external/locomo/data/locomo10.json \
  --memories-out benchmarks/paper/locomo/data/locomo_memories.jsonl \
  --questions-out benchmarks/paper/locomo/data/locomo_questions.jsonl \
  --inspect

uv run python benchmarks/paper/locomo/runners/run_bm25_locomo_retrieval.py \
  --memories benchmarks/paper/locomo/data/locomo_memories.jsonl \
  --questions benchmarks/paper/locomo/data/locomo_questions.jsonl \
  --k 10 \
  --out benchmarks/paper/locomo/results/bm25_k10/retrieval.jsonl

uv run python benchmarks/paper/locomo/runners/judge_locomo_answers.py \
  --retrieval benchmarks/paper/locomo/results/bm25_k10/retrieval.jsonl \
  --mode retrieval \
  --out-dir benchmarks/paper/locomo/results/bm25_k10_retrieval_scored

uv run python benchmarks/paper/locomo/runners/run_sochdb_locomo_retrieval.py \
  --memories benchmarks/paper/locomo/data/locomo_memories.jsonl \
  --questions benchmarks/paper/locomo/data/locomo_questions.jsonl \
  --host 65.108.78.80 \
  --port 50051 \
  --collection-prefix locomo_sochdb \
  --embedding-dim 1536 \
  --k 10 \
  --out benchmarks/paper/locomo/results/sochdb_k10/retrieval.jsonl

uv run python benchmarks/paper/locomo/runners/judge_locomo_answers.py \
  --retrieval benchmarks/paper/locomo/results/sochdb_k10/retrieval.jsonl \
  --mode retrieval \
  --out-dir benchmarks/paper/locomo/results/sochdb_k10_retrieval_scored

echo
echo "BM25 summary:"
cat benchmarks/paper/locomo/results/bm25_k10_retrieval_scored/summary.json | jq

echo
echo "SochDB summary:"
cat benchmarks/paper/locomo/results/sochdb_k10_retrieval_scored/summary.json | jq
