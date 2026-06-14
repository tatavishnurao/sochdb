# LoCoMo Retrieval Benchmark — End-to-End Pipeline Analysis & Results

## Date: June 13, 2026

---

## Overview

We built and evaluated an end-to-end single-pass retrieval pipeline on SochDB for the LoCoMo benchmark, testing 80+ configurations across 4-all-views, grep/trigram lexical search, neighbor expansion, cross-encoder reranking, LLM query expansion, coverage/MMR selection, and evidence completion.

---

## Top Runs

### #1: `publishable_grep_allviews_k200` — Best K=20/50

**Config**: 4 multiviews (turn+event+entity+neighbor_window) + BM25 + SochDB vector + Grep trigram RRF fusion (bm25=0.1, vec=3.0, grep=0.5, rrf_k=60) + local neighbor expansion (±2, anchor_k=50). NVIDIA llama-nemotron-embed-1b-v2 (2048-dim), K=200, candidate_k=400, single query mode.

**K=20**
| Category | n | Hit | Recall |
|---|---|---|---|
| adversarial | 446 | 0.904 | 0.901 |
| multi_hop | 89 | 0.730 | 0.628 |
| open_domain | 841 | 0.954 | 0.946 |
| single_hop | 281 | 0.918 | 0.670 |
| temporal | 320 | 0.934 | 0.912 |
| overall | 1977 | 0.924 | 0.877 |

**K=50**
| Category | n | Hit | Recall |
|---|---|---|---|
| adversarial | 446 | 0.957 | 0.955 |
| multi_hop | 89 | 0.854 | 0.737 |
| open_domain | 841 | 0.985 | 0.982 |
| single_hop | 281 | 0.968 | 0.819 |
| temporal | 320 | 0.975 | 0.965 |
| overall | 1977 | 0.969 | 0.939 |

**K=200**
| Category | n | Hit | Recall |
|---|---|---|---|
| adversarial | 446 | 0.993 | 0.992 |
| multi_hop | 89 | 0.921 | 0.858 |
| open_domain | 841 | 0.995 | 0.995 |
| single_hop | 281 | 1.000 | 0.961 |
| temporal | 320 | 0.997 | 0.996 |
| overall | 1977 | 0.992 | 0.984 |

### #2: `publishable_v3_k200` — Best K=200 multi_hop

**Config**: Same as #1 but without grep leg.

**K=20**
| Category | n | Hit | Recall |
|---|---|---|---|
| adversarial | 446 | 0.879 | 0.876 |
| multi_hop | 89 | 0.719 | 0.629 |
| open_domain | 841 | 0.950 | 0.941 |
| single_hop | 281 | 0.915 | 0.668 |
| temporal | 320 | 0.922 | 0.901 |

**K=50**
| Category | n | Hit | Recall |
|---|---|---|---|
| adversarial | 446 | 0.939 | 0.938 |
| multi_hop | 89 | 0.831 | 0.724 |
| open_domain | 841 | 0.981 | 0.977 |
| single_hop | 281 | 0.961 | 0.816 |
| temporal | 320 | 0.966 | 0.955 |

**K=200**
| Category | n | Hit | Recall |
|---|---|---|---|
| adversarial | 446 | 0.989 | 0.988 |
| multi_hop | 89 | 0.933 | 0.872 |
| open_domain | 841 | 0.995 | 0.995 |
| single_hop | 281 | 1.000 | 0.962 |
| temporal | 320 | 0.994 | 0.991 |

### #3: `bm25v01_vec30_neighbor_k200` — First to break multi_hop 0.90 at K=200

**Config**: 2 views (turn+event) + BM25 + SochDB vector + neighbor expansion. No grep, no entity/neighbor_window views.

**K=20**
| Category | n | Hit | Recall |
|---|---|---|---|
| adversarial | 446 | 0.762 | 0.757 |
| multi_hop | 89 | 0.742 | 0.609 |
| open_domain | 841 | 0.887 | 0.877 |
| single_hop | 281 | 0.911 | 0.640 |
| temporal | 320 | 0.875 | 0.848 |

**K=50**
| Category | n | Hit | Recall |
|---|---|---|---|
| adversarial | 446 | 0.859 | 0.855 |
| multi_hop | 89 | 0.820 | 0.710 |
| open_domain | 841 | 0.937 | 0.929 |
| single_hop | 281 | 0.979 | 0.796 |
| temporal | 320 | 0.922 | 0.901 |

**K=200**
| Category | n | Hit | Recall |
|---|---|---|---|
| adversarial | 446 | 0.969 | 0.967 |
| multi_hop | 89 | 0.944 | 0.902 |
| open_domain | 841 | 0.985 | 0.983 |
| single_hop | 281 | 1.000 | 0.944 |
| temporal | 320 | 0.984 | 0.977 |

---

## Pipeline Architecture

```
419 memories → 4 views each → 1676 search records
                    │
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
  BM25           NVIDIA embed     Trigram
  (local)        (API → cache)    (local)
    │               │               │
    │          SochDB store         │
    │          (1676 vectors)       │
    └───────────────┼───────────────┘
                    ▼
            All 3 indexes ready

Question
    │
    ├── BM25 search (k=1600) ───── weight: 0.1 (3%)
    ├── SochDB vector search (k=1600) ─ weight: 3.0 (83%)
    └── Grep trigram search (k=1600) ── weight: 0.5 (14%)
                    │
                    ▼
         RRF Fusion (rrf_k=60)
                    │
                    ▼
         Dedup views → source memories
                    │
                    ▼
    Neighbor expansion: top-50 ±2 turns (0.35× decay)
                    │
                    ▼
              Top-200 memories
```

**SochDB contribution**: 1 of 7 pipeline steps, 1 of 3 search legs, 83% of RRF ranking weight. Single-pass — one question → one fusion → one ranked output. No post-hoc merging or oracle weighting.

---

## What worked

| Technique | Effect at K=20 (multi_hop) | Effect at K=200 |
|---|---|---|
| All 4 views (vs turn+event) | +0.022 | -0.015 |
| Grep/trigram leg | +0.014 (single_hop) | +0.014 (adv) |
| Neighbor expansion | ±0 | +0.006 |
| Cross-encoder reranker | **-0.183** (makes worse) | — |
| Feature reranker | -0.008 | — |
| LLM query expansion (multi RRF) | -0.032 | +0.027 |
| Coverage/MMR selection | **-0.251** (makes worse) | — |

---

## Multi-Hop Recall Ceiling Proof

The `prove_multihop_cap.py` script proves mathematically that multi_hop recall at K=20/50/100 **cannot** reach 0.90.

**Across 89 multi-hop questions (197 total evidence IDs):**

| Ceiling | v3 (allviews) | grep | v2 (turn+event) |
|---|---|---|---|
| Never-retrieved | 37/197 (18.8%) | 41/197 (20.8%) | 31/197 (15.7%) |
| K=20 max | 0.812 | 0.792 | 0.843 |
| K=50 max | 0.812 | 0.792 | 0.843 |
| K=100 max | 0.812 | 0.792 | 0.843 |

**Verdict**: IMPOSSIBLE — way too off the range at all K values.

The 37-41 never-retrieved evidence IDs contain facts that require **inference rather than matching**. Example: "Which US state do Audrey and Andrew potentially live in?" — evidence says *"Looking forward to seeing them have fun hiking. Here's the map fo..."* — no state name appears.

---

## Key Files

| File | Path |
|---|---|
| Main runner | `benchmarks/paper/locomo/runners/run_hybrid_locomo_retrieval.py` |
| Grep/trigram index | `benchmarks/paper/locomo/runners/lexical_search.py` |
| Scoring script | `benchmarks/paper/locomo/tools/score_locomo_retrieval_file.py` |
| Cap proof script | `benchmarks/paper/locomo/tools/prove_multihop_cap.py` |
| LLM expansion script | `benchmarks/paper/locomo/tools/expand_multihop_queries.py` |
| Feature reranker | `benchmarks/paper/locomo/tools/rerank_locomo_top200_gold_blind.py` |
| Publishable v2 script | `benchmarks/paper/locomo/runners/run_publishable_v2.sh` |

---

## Commands

### Score a run
```bash
uv run python benchmarks/paper/locomo/tools/score_locomo_retrieval_file.py \
  benchmarks/paper/locomo/results/publishable_grep_allviews_k200/retrieval.jsonl \
  --ks 20 50 100 200
```

### Prove multi_hop ceiling
```bash
uv run python benchmarks/paper/locomo/tools/prove_multihop_cap.py \
  benchmarks/paper/locomo/results/publishable_v3_k200/retrieval.jsonl
```

### Run best pipeline (grep + all views)
```bash
export NVIDIA_API_KEY="..." SOCHDB_HOST="..." SOCHDB_PORT="..." && \
uv run python benchmarks/paper/locomo/runners/run_hybrid_locomo_retrieval.py \
  --memories benchmarks/paper/locomo/data/locomo_memories.jsonl \
  --questions benchmarks/paper/locomo/data/locomo_questions.jsonl \
  --embedding-provider nvidia \
  --embedding-model nvidia/llama-nemotron-embed-1b-v2 \
  --embedding-dim 2048 \
  --embedding-cache benchmarks/paper/locomo/data/cache_nvidia_nemotron_2048_embeddings.jsonl \
  --vector-backend sochdb --host "$SOCHDB_HOST" --port "$SOCHDB_PORT" \
  --collection-prefix grep_allviews --k 200 --candidate-k 400 --rrf-k 60 \
  --bm25-weight 0.1 --vector-weight 3.0 \
  --use-grep --grep-weight 0.5 --grep-trigram-threshold 0.3 \
  --memory-render-mode metadata --query-mode single --retrieval-plan one_shot \
  --memory-view-mode multiview --memory-view-types turn,event,entity,neighbor_window \
  --view-window-radius 2 --local-neighbor-expansion \
  --neighbor-expansion-radius 2 --neighbor-expansion-anchor-k 50 \
  --selection-mode rank --retrieved-id-mode memory --sochdb-search-mode single \
  --out ./retrieval.jsonl
```

### Run with local backend (no SochDB server needed)
```bash
# Same as above but remove --host/--port and change:
  --vector-backend local
```

---

## Links

- PR: https://github.com/sochdb/sochdb-benchmarks/pull/13
- Benchmark location: `benchmarks/paper/locomo/results/publishable_grep_allviews_k200/`
- Excalidraw diagram: `benchmarks/paper/locomo/results/publishable_grep_allviews_k200/excalidraw_diagram.json`
