#!/usr/bin/env bash
set -euo pipefail

cd /home/vishnu-rao/Desktop/Arjun/projects/sochdb

SOCHDB_HOST="${SOCHDB_HOST:-100.109.115.80}"
SOCHDB_PORT="${SOCHDB_PORT:-50051}"

MEMORIES="benchmarks/paper/locomo/data/locomo_memories.jsonl"
QUESTIONS="benchmarks/paper/locomo/data/locomo_questions.jsonl"

RESULT_DIR="benchmarks/paper/locomo/results/grep_hybrid_multiview_sochdb_nvidia_dim2048_k200"

mkdir -p "$RESULT_DIR"

echo "Using SochDB target: ${SOCHDB_HOST}:${SOCHDB_PORT}"
echo "Output: ${RESULT_DIR}"

echo
echo "== TCP connectivity check =="
.venv/bin/python - <<PY
import socket
host = "${SOCHDB_HOST}"
port = int("${SOCHDB_PORT}")
s = socket.create_connection((host, port), timeout=10)
print("TCP connected:", host, port)
s.close()
PY

echo
echo "== Running hybrid retrieval with grep leg ==="
.venv/bin/python benchmarks/paper/locomo/runners/run_hybrid_locomo_retrieval.py \
  --memories "$MEMORIES" \
  --questions "$QUESTIONS" \
  --embedding-provider hash \
  --embedding-model hash \
  --embedding-dim 1536 \
  --vector-backend sochdb \
  --host "$SOCHDB_HOST" \
  --port "$SOCHDB_PORT" \
  --collection-prefix locomo_grep_hybrid \
  --k 200 \
  --candidate-k 400 \
  --rrf-k 60 \
  --bm25-weight 1.0 \
  --vector-weight 1.0 \
  --use-grep \
  --grep-weight 1.0 \
  --grep-trigram-threshold 0.3 \
  --memory-render-mode metadata \
  --query-mode multi \
  --retrieval-plan anchored_two_hop \
  --anchor-top-n 20 \
  --anchor-max-queries 4 \
  --anchor-weight 0.5 \
  --memory-view-mode multiview \
  --memory-view-types turn,event \
  --view-window-radius 2 \
  --local-neighbor-expansion \
  --neighbor-expansion-radius 2 \
  --neighbor-expansion-anchor-k 50 \
  --evidence-completion conservative \
  --completion-seed-top-n 20 \
  --completion-window-radius 2 \
  --completion-same-speaker-limit 5 \
  --completion-max-candidates 80 \
  --completion-weight 0.20 \
  --selection-mode rank \
  --retrieved-id-mode memory \
  --sochdb-search-mode single \
  --out "${RESULT_DIR}/retrieval.jsonl"

echo
echo "== Scoring retrieval results =="
.venv/bin/python benchmarks/paper/locomo/tools/score_locomo_retrieval_file.py \
  "${RESULT_DIR}/retrieval.jsonl" \
  --ks 20 50 100 150 200

echo
echo "Done! Results in ${RESULT_DIR}/retrieval.jsonl"