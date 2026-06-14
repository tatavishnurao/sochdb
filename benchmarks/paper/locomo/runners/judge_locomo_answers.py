#!/usr/bin/env python3

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any, Dict, List


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


def write_json(path: str, obj: Any):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def evidence_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    evidence = set(row.get("evidence_memory_ids") or [])
    retrieved = set(row.get("retrieved_memory_ids") or [])

    if not evidence:
        return {
            "evidence_hit": None,
            "evidence_recall": None,
            "num_evidence": 0,
        }

    hit_count = len(evidence & retrieved)

    return {
        "evidence_hit": int(hit_count > 0),
        "evidence_recall": hit_count / len(evidence),
        "num_evidence": len(evidence),
    }


def openai_answer_and_judge(row: Dict[str, Any], model: str) -> Dict[str, Any]:
    from openai import OpenAI

    client = OpenAI()

    question = row["question"]
    gold = row.get("gold_answer", "")
    context = row.get("debug_context", "")

    answer_prompt = f"""You are answering a long-term memory question.

Use only the retrieved context below. If the answer is not supported, say "unknown".

Retrieved context:
{context}

Question:
{question}

Answer concisely:"""

    answer_resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You answer questions using only provided context."},
            {"role": "user", "content": answer_prompt},
        ],
        temperature=0,
    )

    predicted = answer_resp.choices[0].message.content.strip()

    judge_prompt = f"""You are judging a memory QA benchmark answer.

Question:
{question}

Gold answer:
{gold}

Predicted answer:
{predicted}

Retrieved context:
{context}

Decide whether the predicted answer is semantically correct with respect to the gold answer.
Return JSON only:
{{
  "correct": true or false,
  "partially_correct": true or false,
  "uses_context": true or false,
  "contradiction": true or false,
  "reason": "short reason"
}}"""

    judge_resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a strict but fair benchmark judge. Return JSON only."},
            {"role": "user", "content": judge_prompt},
        ],
        temperature=0,
    )

    raw = judge_resp.choices[0].message.content.strip()

    try:
        judged = json.loads(raw)
    except Exception:
        judged = {
            "correct": False,
            "partially_correct": False,
            "uses_context": False,
            "contradiction": False,
            "reason": f"judge_json_parse_failed: {raw[:300]}",
        }

    return {
        "predicted_answer": predicted,
        "judge_raw": raw,
        "judge_correct": bool(judged.get("correct", False)),
        "judge_partially_correct": bool(judged.get("partially_correct", False)),
        "judge_uses_context": bool(judged.get("uses_context", False)),
        "judge_contradiction": bool(judged.get("contradiction", False)),
        "judge_reason": judged.get("reason", ""),
    }


def mean_ignore_none(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def percentile(values, p: float):
    if not values:
        return None
    values = sorted(values)
    idx = int(round((len(values) - 1) * p))
    return values[idx]


def summarize(rows: List[Dict[str, Any]], mode: str) -> Dict[str, Any]:
    n = len(rows)

    out = {
        "n_questions": n,
        "system": rows[0].get("system", "unknown") if rows else "unknown",
        "mode": mode,
        "evidence_hit_rate": mean_ignore_none([r.get("evidence_hit") for r in rows]),
        "evidence_recall": mean_ignore_none([r.get("evidence_recall") for r in rows]),
        "avg_context_tokens": mean_ignore_none([r.get("approx_context_tokens") for r in rows]),
        "avg_latency_ms": mean_ignore_none([r.get("latency_ms") for r in rows]),
        "p50_latency_ms": percentile([r.get("latency_ms", 0.0) for r in rows], 0.50),
        "p95_latency_ms": percentile([r.get("latency_ms", 0.0) for r in rows], 0.95),
    }

    if mode == "openai":
        out["judge_accuracy"] = sum(int(r.get("judge_correct", False)) for r in rows) / n if n else 0
        out["judge_partial_accuracy"] = sum(int(r.get("judge_correct", False) or r.get("judge_partially_correct", False)) for r in rows) / n if n else 0
        out["contradiction_rate"] = sum(int(r.get("judge_contradiction", False)) for r in rows) / n if n else 0

    by_category = {}

    for r in rows:
        cat = r.get("category", "unknown")
        by_category.setdefault(cat, []).append(r)

    out["by_category"] = {}

    for cat, subset in sorted(by_category.items()):
        item = {
            "n_questions": len(subset),
            "evidence_hit_rate": mean_ignore_none([r.get("evidence_hit") for r in subset]),
            "evidence_recall": mean_ignore_none([r.get("evidence_recall") for r in subset]),
            "avg_context_tokens": mean_ignore_none([r.get("approx_context_tokens") for r in subset]),
            "avg_latency_ms": mean_ignore_none([r.get("latency_ms") for r in subset]),
        }

        if mode == "openai":
            item["judge_accuracy"] = sum(int(r.get("judge_correct", False)) for r in subset) / len(subset)
            item["judge_partial_accuracy"] = sum(int(r.get("judge_correct", False) or r.get("judge_partially_correct", False)) for r in subset) / len(subset)
            item["contradiction_rate"] = sum(int(r.get("judge_contradiction", False)) for r in subset) / len(subset)

        out["by_category"][cat] = item

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mode", choices=["retrieval", "openai"], default="retrieval")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = load_jsonl(args.retrieval)

    if args.limit is not None:
        rows = rows[: args.limit]

    judged = []

    for i, row in enumerate(rows, start=1):
        item = dict(row)
        item.update(evidence_metrics(row))

        if args.mode == "openai":
            print(f"[{i}/{len(rows)}] judging {row.get('question_id')}")
            item.update(openai_answer_and_judge(row, args.model))

        judged.append(item)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(str(out_dir / "judged.jsonl"), judged)
    summary = summarize(judged, args.mode)
    write_json(str(out_dir / "summary.json"), summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
