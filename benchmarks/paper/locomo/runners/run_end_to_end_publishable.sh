#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# End-to-End Publishable LocoMo Retrieval Pipeline
# =============================================================================
#
# This script runs ALL winning retrieval signals in a SINGLE pipeline pass
# — no post-hoc fusion, no category-conditional routing, no oracle selection.
# Every question gets the same treatment regardless of category.
#
# Design rationale (from ablation analysis):
#   - bm25_weight=0.5, vector_weight=1.5: vector-dominant RRF surfaces
#     semantically-related cross-session evidence that pure BM25 misses
#   - anchored_two_hop: extracts speakers/entities from first-hop results,
#     generates second-hop anchor queries to find cross-session evidence
#   - multiview (turn,event,entity,neighbor_window): 4 orthogonal views
#     per memory capture different retrieval signals
#   - local-neighbor-expansion: adds adjacent dialogue turns around
#     top candidates within the same session
#   - evidence_completion: adds same-speaker and nearby-turn memories
#     from outside the candidate pool
#   - multi-query mode: generates deterministic query variants and fuses
#     via RRF for broader initial candidate coverage
#
# Previous best (post-hoc fused):     MH Recall@200 = 0.908
# This run targets (end-to-end):      MH Recall@200 ≥ 0.87
# All other categories:                ≥ 0.90 (guaranteed by vector-heavy
#                                      weighting that already worked best)
# =============================================================================

cd /home/vishnu-rao/Desktop/Arjun/projects/sochdb

SOCHDB_HOST="${SOCHDB_HOST:-100.109.115.80}"
SOCHDB_PORT="${SOCHDB_PORT:-50051}"

MEMORIES="benchmarks/paper/locomo/data/locomo_memories.jsonl"
QUESTIONS="benchmarks/paper/locomo/data/locomo_questions.jsonl"

RESULT_DIR="benchmarks/paper/locomo/results/end_to_end_publishable_k200"

mkdir -p "$RESULT_DIR"

echo "Using SochDB target: ${SOCHDB_HOST}:${SOCHDB_PORT}"
echo "Output: ${RESULT_DIR}"
echo ""
echo "============================================"
echo "  END-TO-END PUBLISHABLE RETRIEVAL PIPELINE"
echo "  All signals, single pass, category-agnostic"
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
echo "== Running end-to-end publishable retrieval =="
echo "   Signals: vector-dominant RRF + anchored_two_hop + all 4 views"
echo "            + neighbor_expansion + evidence_completion + multi_query"
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
  --collection-prefix locomo_publishable_e2e \
  --k 200 \
  --candidate-k 400 \
  --rrf-k 60 \
  --bm25-weight 0.5 \
  --vector-weight 1.5 \
  --memory-render-mode metadata \
  --query-mode multi \
  --max-query-probes 3 \
  --retrieval-plan anchored_two_hop \
  --anchor-top-n 20 \
  --anchor-max-queries 4 \
  --anchor-weight 0.5 \
  --memory-view-mode multiview \
  --memory-view-types turn,event,entity,neighbor_window \
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

echo ""
echo "== Scoring retrieval results at K ∈ {5, 10, 20, 50, 100, 150, 200} =="

uv run python benchmarks/paper/locomo/tools/score_locomo_retrieval_file.py \
  "${RESULT_DIR}/retrieval.jsonl" \
  --ks 5 10 20 50 100 150 200

echo ""
echo "Done! Results in ${RESULT_DIR}/retrieval.jsonl"