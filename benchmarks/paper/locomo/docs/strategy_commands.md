# Multi-Hop 90% Recall Strategy Commands

All commands use the best ablation config: `turn,event` views, `candidate_k=400`, `single` query mode.
Run these in order. Each depends on the previous step completing.

## Strategy 1: Decomposition (already scored, K200 Recall=0.694 — REJECTED)

Decomposition with default params hurt recall. Skipping.

## Strategy 2: MMR Diversity Selection

MMR (Maximal Marginal Relevance) forces diversity in the top-K by penalizing
redundant content. Lambda controls relevance-vs-diversity tradeoff:
- lambda=1.0 = pure relevance (same as rank mode)
- lambda=0.0 = pure diversity
- lambda=0.7 = mostly relevance with diversity nudge

### Strategy 2a: MMR lambda=0.7

```bash
source .env && export SOCHDB_HOST SOCHDB_PORT NVIDIA_API_KEY && python3 benchmarks/paper/locomo/runners/run_hybrid_locomo_retrieval.py \
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
  --evidence-completion none \
  --retrieved-id-mode memory \
  --reranker-provider none \
  --selection-mode mmr \
  --mmr-lambda 0.7 \
  --collection-prefix locomo_ablation_multihop_mmr07_k200 \
  --out benchmarks/paper/locomo/results/ablation_multihop_mmr07_k200/retrieval.jsonl
```

### Strategy 2b: MMR lambda=0.5 (more diversity)

```bash
source .env && export SOCHDB_HOST SOCHDB_PORT NVIDIA_API_KEY && python3 benchmarks/paper/locomo/runners/run_hybrid_locomo_retrieval.py \
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
  --evidence-completion none \
  --retrieved-id-mode memory \
  --reranker-provider none \
  --selection-mode mmr \
  --mmr-lambda 0.5 \
  --collection-prefix locomo_ablation_multihop_mmr05_k200 \
  --out benchmarks/paper/locomo/results/ablation_multihop_mmr05_k200/retrieval.jsonl
```

### Scoring (run after each completes):

```bash
python3 benchmarks/paper/locomo/tools/score_locomo_retrieval_file.py benchmarks/paper/locomo/results/ablation_multihop_mmr07_k200/retrieval.jsonl --ks 20 50 100 150 200
python3 benchmarks/paper/locomo/tools/score_locomo_retrieval_file.py benchmarks/paper/locomo/results/ablation_multihop_mmr05_k200/retrieval.jsonl --ks 20 50 100 150 200
```

## Strategy 3: Splice Fusion

Takes top-N from each of multiple retrieval runs and merges them, simulating
higher effective K without exceeding the K=200 output budget.

### Prerequisites

You need at least 2 existing retrieval.jsonl files. The best configs so far:
- `ablation_multihop_views_turn_event_k200` (K200 Recall=0.8437)
- `multihop_multiview_metadata_k200_overfetch_fixed` (K200 Recall=0.8281)

### Strategy 3a: Splice top-100 from each of 2 runs

```bash
python3 benchmarks/paper/locomo/tools/splice_fuse.py \
  --inputs \
    benchmarks/paper/locomo/results/ablation_multihop_views_turn_event_k200/retrieval.jsonl \
    benchmarks/paper/locomo/results/multihop_multiview_metadata_k200_overfetch_fixed/retrieval.jsonl \
  --n-per-input 100 \
  --k 200 \
  --output benchmarks/paper/locomo/results/ablation_multihop_splice_2x100_k200/retrieval.jsonl
```

### Strategy 3b: Splice top-133 from each of 2 runs

```bash
python3 benchmarks/paper/locomo/tools/splice_fuse.py \
  --inputs \
    benchmarks/paper/locomo/results/ablation_multihop_views_turn_event_k200/retrieval.jsonl \
    benchmarks/paper/locomo/results/multihop_multiview_metadata_k200_overfetch_fixed/retrieval.jsonl \
  --n-per-input 133 \
  --k 200 \
  --output benchmarks/paper/locomo/results/ablation_multihop_splice_2x133_k200/retrieval.jsonl
```

### Strategy 3c: Splice 3 runs (add baseline turn-only)

```bash
python3 benchmarks/paper/locomo/tools/splice_fuse.py \
  --inputs \
    benchmarks/paper/locomo/results/ablation_multihop_views_turn_event_k200/retrieval.jsonl \
    benchmarks/paper/locomo/results/multihop_multiview_metadata_k200_overfetch_fixed/retrieval.jsonl \
    benchmarks/paper/locomo/results/multihop_baseline_metadata_turn_k200/retrieval.jsonl \
  --n-per-input 100 \
  --k 200 \
  --output benchmarks/paper/locomo/results/ablation_multihop_splice_3x100_k200/retrieval.jsonl
```

### Scoring:

```bash
python3 benchmarks/paper/locomo/tools/score_locomo_retrieval_file.py benchmarks/paper/locomo/results/ablation_multihop_splice_2x100_k200/retrieval.jsonl --ks 20 50 100 150 200
python3 benchmarks/paper/locomo/tools/score_locomo_retrieval_file.py benchmarks/paper/locomo/results/ablation_multihop_splice_2x133_k200/retrieval.jsonl --ks 20 50 100 150 200
python3 benchmarks/paper/locomo/tools/score_locomo_retrieval_file.py benchmarks/paper/locomo/results/ablation_multihop_splice_3x100_k200/retrieval.jsonl --ks 20 50 100 150 200
```

## Strategy 4: MMR + Splice (combine best MMR result with splice)

After finding the best MMR lambda, combine it with splice:

```bash
# First find best MMR result, then:
python3 benchmarks/paper/locomo/tools/splice_fuse.py \
  --inputs \
    benchmarks/paper/locomo/results/ablation_multihop_mmr07_k200/retrieval.jsonl \
    benchmarks/paper/locomo/results/ablation_multihop_views_turn_event_k200/retrieval.jsonl \
  --n-per-input 100 \
  --k 200 \
  --output benchmarks/paper/locomo/results/ablation_multihop_mmr_splice_k200/retrieval.jsonl

python3 benchmarks/paper/locomo/tools/score_locomo_retrieval_file.py benchmarks/paper/locomo/results/ablation_multihop_mmr_splice_k200/retrieval.jsonl --ks 20 50 100 150 200
```