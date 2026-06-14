# Why Recall is Stuck at 84.4% — and What It Actually Takes to Reach 90%

## Diagnosis

The best run (`views_turn_event`) achieves K200 Recall=0.8437. This **matches the oracle ceiling** at K=200 — the system is finding every gold evidence ID that exists in its top-200 candidate list. The remaining 46 missing gold IDs (across 24/89 questions) are **completely absent from the candidate pool** — neither BM25 nor vector search ranked them anywhere in the top-800 view candidates.

### Key numbers
- 46 missing gold IDs across 24 questions
- 65/89 questions (73%) have perfect recall
- 5 questions have zero recall (all gold IDs are missing)
- 73.9% of missing gold IDs are in **different sessions** from the found evidence
- Only 26.1% are in the same session as found evidence (these could be caught by evidence completion)

### Root cause
Multi-hop questions require evidence from **multiple sessions/conversations**. The retriever embeds the full question as a single vector and does keyword search. If a gold memory is in session_27 but the question's keywords and embedding match session_1 and session_16, that session_27 memory never enters the candidate pool.

This is NOT a ranking problem. It's a **candidate generation problem** — semantically distant evidence is excluded before ranking even begins.

## Why current approaches fail

| Approach | Why it fails |
|----------|-------------|
| Higher candidate_k | Dilution — more noise, same missing IDs |
| MMR diversity | Candidates are already diverse — MMR doesn't add new IDs |
| Splice fusion | 100% overlap between runs — no new IDs |
| Decomposition | Decomposed queries are too noisy and dilute signal |
| Neighbor expansion | Only expands within same session — 73.9% of missing gold is cross-session |
| Entity multi | Same single-query embedding, just different probes |

## What WILL reach 90%

### Strategy A: Evidence Completion (immediate, highest impact)

The runner already has `--evidence-completion conservative` which adds memories from OUTSIDE the candidate pool based on:
- Proximity to retrieved candidates (same session, nearby dialogue turns)
- Same speaker as top candidates
- Temporal adjacency

This should recover the 26.1% of missing gold IDs that are in the same session as found evidence. It won't help the 73.9% that are cross-session.

**Expected gain: +2-4% recall** (recovers same-session misses)

### Strategy B: Multi-session candidate expansion (requires new code)

The missing 34 gold IDs (73.9%) are in sessions different from all found evidence. To find them, we need to **expand the candidate pool across sessions** anchored by the top-K results.

Algorithm:
1. Retrieve top-K candidates as usual
2. Extract entities/speakers/sessions from the top-K
3. For each extracted entity, do a SECOND round of BM25+vector search targeting that entity's sessions
4. Merge the second-round candidates with the first-round pool

This is different from decomposition (which generates alternative queries from the question) — it generates queries from the RETRIEVED EVIDENCE itself.

### Strategy C: BM25/vector weight sweep (immediate, modest gain)

Current weights (bm25=1.5, vector=0.75) may overweight BM25 for certain question types. A sweep could find a better balance that surfaces more cross-session evidence.

## Commands to run

### A. Evidence Completion (conservative)

```bash
source .env && export SOCHDB_HOST SOCHDB_PORT NVIDIA_API_KEY && uv run python benchmarks/paper/locomo/runners/run_hybrid_locomo_retrieval.py \
  --memories benchmarks/paper/locomo/data/locomo_memories.jsonl \
  --questions benchmarks/paper/locomo/data/derived/locomo_questions_multihop.jsonl \
  --embedding-provider nvidia \
  --embedding-model nvidia/llama-nemotron-embed-1b-v2 \
  --embedding-dim 2048 \
  --vector-backend sochdb \
  --host "$SOCHDB_HOST" \
  --port "$SOCHDB_PORT" \
  --k 200 \
  --candidate-k 400 \
  --bm25-weight 1.5 \
  --vector-weight 0.75 \
  --rrf-k 60 \
  --query-mode single \
  --memory-render-mode metadata \
  --memory-view-mode multiview \
  --memory-view-types turn,event \
  --view-window-radius 2 \
  --retrieval-plan one_shot \
  --evidence-completion conservative \
  --completion-seed-top-n 20 \
  --completion-window-radius 2 \
  --completion-same-speaker-limit 5 \
  --completion-max-candidates 80 \
  --completion-weight 0.20 \
  --retrieved-id-mode memory \
  --reranker-provider none \
  --collection-prefix locomo_ablation_multihop_evidence_completion_k200 \
  --out benchmarks/paper/locomo/results/ablation_multihop_evidence_completion_k200/retrieval.jsonl
```

Score:
```bash
uv run python benchmarks/paper/locomo/tools/score_locomo_retrieval_file.py benchmarks/paper/locomo/results/ablation_multihop_evidence_completion_k200/retrieval.jsonl --ks 20 50 100 150 200
```

### B. Evidence Completion + higher candidate_k (to get more seeds)

```bash
source .env && export SOCHDB_HOST SOCHDB_PORT NVIDIA_API_KEY && uv run python benchmarks/paper/locomo/runners/run_hybrid_locomo_retrieval.py \
  --memories benchmarks/paper/locomo/data/locomo_memories.jsonl \
  --questions benchmarks/paper/locomo/data/derived/locomo_questions_multihop.jsonl \
  --embedding-provider nvidia \
  --embedding-model nvidia/llama-nemotron-embed-1b-v2 \
  --embedding-dim 2048 \
  --vector-backend sochdb \
  --host "$SOCHDB_HOST" \
  --port "$SOCHDB_PORT" \
  --k 200 \
  --candidate-k 600 \
  --bm25-weight 1.5 \
  --vector-weight 0.75 \
  --rrf-k 60 \
  --query-mode single \
  --memory-render-mode metadata \
  --memory-view-mode multiview \
  --memory-view-types turn,event \
  --view-window-radius 2 \
  --retrieval-plan one_shot \
  --evidence-completion conservative \
  --completion-seed-top-n 30 \
  --completion-window-radius 2 \
  --completion-same-speaker-limit 5 \
  --completion-max-candidates 120 \
  --completion-weight 0.20 \
  --retrieved-id-mode memory \
  --reranker-provider none \
  --collection-prefix locomo_ablation_multihop_evidence_completion_ck600_k200 \
  --out benchmarks/paper/locomo/results/ablation_multihop_evidence_completion_ck600_k200/retrieval.jsonl
```

Score:
```bash
uv run python benchmarks/paper/locomo/tools/score_locomo_retrieval_file.py benchmarks/paper/locomo/results/ablation_multihop_evidence_completion_ck600_k200/retrieval.jsonl --ks 20 50 100 150 200
```

### C. BM25/vector weight sweep (candidate_k=400, turn+event, no completion)

Test vector-heavy weights to see if vector search surfaces cross-session evidence:

```bash
# C1: bm25=1.0 vector=1.0 (balanced)
source .env && export SOCHDB_HOST SOCHDB_PORT NVIDIA_API_KEY && uv run python benchmarks/paper/locomo/runners/run_hybrid_locomo_retrieval.py \
  --memories benchmarks/paper/locomo/data/locomo_memories.jsonl \
  --questions benchmarks/paper/locomo/data/derived/locomo_questions_multihop.jsonl \
  --embedding-provider nvidia \
  --embedding-model nvidia/llama-nemotron-embed-1b-v2 \
  --embedding-dim 2048 \
  --vector-backend sochdb \
  --host "$SOCHDB_HOST" \
  --port "$SOCHDB_PORT" \
  --k 200 \
  --candidate-k 400 \
  --bm25-weight 1.0 \
  --vector-weight 1.0 \
  --rrf-k 60 \
  --query-mode single \
  --memory-render-mode metadata \
  --memory-view-mode multiview \
  --memory-view-types turn,event \
  --view-window-radius 2 \
  --retrieval-plan one_shot \
  --evidence-completion none \
  --retrieved-id-mode memory \
  --reranker-provider none \
  --collection-prefix locomo_ablation_multihop_bm25v10_vec10_k200 \
  --out benchmarks/paper/locomo/results/ablation_multihop_bm25v10_vec10_k200/retrieval.jsonl
```

```bash
# C2: bm25=0.75 vector=1.5 (vector-dominant)
source .env && export SOCHDB_HOST SOCHDB_PORT NVIDIA_API_KEY && uv run python benchmarks/paper/locomo/runners/run_hybrid_locomo_retrieval.py \
  --memories benchmarks/paper/locomo/data/locomo_memories.jsonl \
  --questions benchmarks/paper/locomo/data/derived/locomo_questions_multihop.jsonl \
  --embedding-provider nvidia \
  --embedding-model nvidia/llama-nemotron-embed-1b-v2 \
  --embedding-dim 2048 \
  --vector-backend sochdb \
  --host "$SOCHDB_HOST" \
  --port "$SOCHDB_PORT" \
  --k 200 \
  --candidate-k 400 \
  --bm25-weight 0.75 \
  --vector-weight 1.5 \
  --rrf-k 60 \
  --query-mode single \
  --memory-render-mode metadata \
  --memory-view-mode multiview \
  --memory-view-types turn,event \
  --view-window-radius 2 \
  --retrieval-plan one_shot \
  --evidence-completion none \
  --retrieved-id-mode memory \
  --reranker-provider none \
  --collection-prefix locomo_ablation_multihop_bm25v075_vec15_k200 \
  --out benchmarks/paper/locomo/results/ablation_multihop_bm25v075_vec15_k200/retrieval.jsonl
```

Score both:
```bash
uv run python benchmarks/paper/locomo/tools/score_locomo_retrieval_file.py benchmarks/paper/locomo/results/ablation_multihop_bm25v10_vec10_k200/retrieval.jsonl --ks 20 50 100 150 200
uv run python benchmarks/paper/locomo/tools/score_locomo_retrieval_file.py benchmarks/paper/locomo/results/ablation_multihop_bm25v075_vec15_k200/retrieval.jsonl --ks 20 50 100 150 200
```

### D. Strategy A + C combined (evidence completion + better weights)

After finding the best weight combo, combine with evidence completion.

## Why 90% requires new code (Strategy B)

The 34 cross-session missing gold IDs (73.9% of gaps) require a **second-hop retrieval** anchored by the first-hop results. None of the existing flags implement this. Evidence completion only finds nearby memories within the same session. To find cross-session evidence, we need:

1. Extract entities/speakers from top-K results
2. Generate second-hop queries from those extracted entities  
3. Do a second BM25+vector search using those queries
4. Merge second-hop candidates into the final pool

This is structurally different from `--retrieval-plan decomposed` (which decomposes the QUESTION) — it decomposes the RETRIEVED EVIDENCE to find the next hop.