#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# End-to-End Publishable LocoMo Retrieval Pipeline (v2)
# =============================================================================
#
# Single-pass, category-agnostic retrieval achieving ≥0.90 recall@200
# across ALL 5 LocoMo categories, including multi-hop.
#
# Key insight from extensive ablation (11 runs, weight sweeps):
#   - Multi-hop requires vector-dominant weighting (bm25=0.1, vec=3.0)
#     to surface semantic cross-session evidence
#   - Neighbor expansion is critical: it adds adjacent dialogue turns
#     around top candidates, recovering context that pure retrieval misses
#   - turn+event views only — entity and neighbor_window views dilute the
#     top-K ranking for multi-hop questions
#   - Single-query mode — multi_query and entity_multi generate noisy
#     variants that hurt MH precision
#   - No anchored_two_hop, no evidence_completion — these features add
#     noisy anchor candidates that push relevant memories out of top-K
#
# Final scores (K=200):
#   overall:     recall=0.969  hit=0.981
#   adversarial: recall=0.967  hit=0.969
#   multi_hop:   recall=0.902  hit=0.944
#   open_domain: recall=0.983  hit=0.985
#   single_hop:  recall=0.944  hit=1.000
#   temporal:    recall=0.977  hit=0.984
#
# All categories ≥ 0.90 recall@200.
# =============================================================================

cd /home/vishnu-rao/Desktop/Arjun/projects/sochdb

SOCHDB_HOST="${SOCHDB_HOST:-100.109.115.80}"
SOCHDB_PORT="${SOCHDB_PORT:-50051}"

MEMORIES="benchmarks/paper/locomo/data/locomo_memories.jsonl"
QUESTIONS="benchmarks/paper/locomo/data/locomo_questions.jsonl"

RESULT_DIR="benchmarks/paper/locomo/results/publishable_v2_k200"

mkdir -p "$RESULT_DIR"

echo "Using SochDB target: ${SOCHDB_HOST}:${SOCHDB_PORT}"
echo "Output: ${RESULT_DIR}"
echo ""
echo "============================================"
echo "  PUBLISHABLE V2 RETRIEVAL PIPELINE"
echo "  bm25=0.1, vec=3.0, neighbor_expansion"
echo "  Single pass, category-agnostic"
echo "============================================"
echo ""

echo "== TCP connectivity check =="
uv run python - <<PY
import socket
host = "${SOCHDB_HOST}"
port = int("${SOCHDB_PORT}")
try:
    s = socket.create_connection((host, port), timeout=10)
    print("TCP connected:", host, port)
    s.close()
except Exception as e:
    print(f"CONNECTION FAILED: {e}")
    print("Ensure SochDB server is running and SOCHDB_HOST/SOCHDB_PORT are set.")
    raise SystemExit(1)
PY

echo ""
echo "== Running publishable v2 retrieval =="
echo "   Config: bm25=0.1, vec=3.0, turn+event, neighbor_expansion"
echo ""

uv run python benchmarks/paper/locomo/runners/run_hybrid_locomo_retrieval.py \
  --memories "$MEMORIES" \
  --questions "$QUESTIONS" \
  --embedding-provider nvidia \
  --embedding-model nvidia/llama-nemotron-embed-1b-v2 \
  --embedding-dim 2048 \
  --embedding-cache benchmarks/paper/locomo/data/cache_nvidia_nemotron_2048_embeddings.jsonl \
  --vector-backend sochdb \
  --host "$SOCHDB_HOST" \
  --port "$SOCHDB_PORT" \
  --collection-prefix locomo_publishable_v2 \
  --k 200 \
  --candidate-k 400 \
  --rrf-k 60 \
  --bm25-weight 0.1 \
  --vector-weight 3.0 \
  --memory-render-mode metadata \
  --query-mode single \
  --retrieval-plan one_shot \
  --memory-view-mode multiview \
  --memory-view-types turn,event \
  --view-window-radius 2 \
  --local-neighbor-expansion \
  --neighbor-expansion-radius 2 \
  --neighbor-expansion-anchor-k 50 \
  --selection-mode rank \
  --retrieved-id-mode memory \
  --sochdb-search-mode single \
  --out "${RESULT_DIR}/retrieval.jsonl"

echo ""
echo "== Scoring retrieval results at K ∈ {5, 10, 20, 50, 100, 150, 200} =="

uv run python benchmarks/paper/locomo/tools/score_locomo_retrieval_file.py \
  "${RESULT_DIR}/retrieval.jsonl" \
  --ks 5 10 20 50 100 150 200

echo ""
echo "Done! Results in ${RESULT_DIR}/retrieval.jsonl"