#!/usr/bin/env python3

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List

from rank_bm25 import BM25Okapi


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return text.split()


def approx_tokens(text: str) -> int:
    return len(text.split())


def group_by_sample(rows):
    out = {}
    for r in rows:
        out.setdefault(r["sample_id"], []).append(r)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memories", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--limit-questions", type=int, default=None)
    args = parser.parse_args()

    memories = load_jsonl(args.memories)
    questions = load_jsonl(args.questions)

    if args.limit_questions is not None:
        questions = questions[: args.limit_questions]

    memories_by_sample = group_by_sample(memories)
    questions_by_sample = group_by_sample(questions)

    sample_ids = sorted(questions_by_sample.keys())
    if args.limit_samples is not None:
        sample_ids = sample_ids[: args.limit_samples]

    output_rows = []

    for sample_id in sample_ids:
        sample_memories = memories_by_sample.get(sample_id, [])
        sample_questions = questions_by_sample.get(sample_id, [])

        corpus_tokens = [tokenize(m["text"]) for m in sample_memories]
        bm25 = BM25Okapi(corpus_tokens)

        for q in sample_questions:
            start = time.perf_counter()

            scores = bm25.get_scores(tokenize(q["question"]))
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: args.k]

            latency_ms = (time.perf_counter() - start) * 1000.0

            retrieved_memory_ids = []
            context_parts = []

            for idx in ranked:
                m = sample_memories[idx]
                mid = int(m["memory_id"])
                retrieved_memory_ids.append(mid)
                context_parts.append(
                    f"[memory_id={mid} sample_id={m['sample_id']} "
                    f"session={m.get('session')} dia_id={m.get('dia_id')} "
                    f"speaker={m.get('speaker')}] {m.get('text', '')}"
                )

            debug_context = "\n".join(context_parts)

            output_rows.append(
                {
                    "system": "bm25",
                    "question_id": q["question_id"],
                    "sample_id": q["sample_id"],
                    "question": q["question"],
                    "gold_answer": q.get("gold_answer", ""),
                    "category": q.get("category", "unknown"),
                    "category_id": q.get("category_id"),
                    "evidence_refs": q.get("evidence_refs", []),
                    "evidence_memory_ids": q.get("evidence_memory_ids", []),
                    "retrieved_memory_ids": retrieved_memory_ids,
                    "retrieved_count": len(retrieved_memory_ids),
                    "approx_context_tokens": approx_tokens(debug_context),
                    "latency_ms": latency_ms,
                    "debug_context": debug_context,
                }
            )

    write_jsonl(args.out, output_rows)
    print(f"Wrote {len(output_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
