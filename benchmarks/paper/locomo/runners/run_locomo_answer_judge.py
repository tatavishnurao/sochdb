#!/usr/bin/env python3

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Any

from openai import OpenAI


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def norm_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in {"true", "yes", "1"}
    return False


def build_answer_prompt(question: str, context: str) -> str:
    return f"""You are answering a long-term conversation memory question.

Rules:
- Use only the provided context.
- If the context does not support the answer, answer exactly: unknown
- Give a concise answer.
- Do not explain.

Question:
{question}

Context:
{context}

Answer:"""


def build_judge_prompt(question: str, gold: str, prediction: str) -> str:
    return f"""You are judging an answer to a long-term memory benchmark question.

Return only valid JSON with:
{{
  "correct": true/false,
  "partially_correct": true/false,
  "contradiction": true/false,
  "reason": "short reason"
}}

Question:
{question}

Gold answer:
{gold}

Predicted answer:
{prediction}

Judgment JSON:"""


def call_chat(client, model: str, prompt: str, temperature: float = 0.0) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


def parse_judge_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        obj = json.loads(text)
    except Exception:
        obj = {
            "correct": False,
            "partially_correct": False,
            "contradiction": False,
            "reason": f"judge_parse_failed: {text[:300]}",
        }
    return {
        "correct": norm_bool(obj.get("correct", False)),
        "partially_correct": norm_bool(obj.get("partially_correct", False)),
        "contradiction": norm_bool(obj.get("contradiction", False)),
        "reason": str(obj.get("reason", "")),
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {}

    def avg(key):
        vals = [r.get(key, 0.0) for r in rows]
        return sum(vals) / len(vals)

    correct = sum(1 for r in rows if r["judge"]["correct"])
    partial = sum(1 for r in rows if r["judge"]["partially_correct"])
    contradiction = sum(1 for r in rows if r["judge"]["contradiction"])
    unknown = sum(1 for r in rows if r["prediction"].strip().lower() == "unknown")

    by_category = {}
    for r in rows:
        cat = r.get("category", "unknown")
        by_category.setdefault(cat, []).append(r)

    by_cat_summary = {}
    for cat, cat_rows in by_category.items():
        cn = len(cat_rows)
        by_cat_summary[cat] = {
            "n_questions": cn,
            "judge_accuracy": sum(1 for r in cat_rows if r["judge"]["correct"]) / cn,
            "judge_partial_accuracy": sum(1 for r in cat_rows if r["judge"]["partially_correct"]) / cn,
            "contradiction_rate": sum(1 for r in cat_rows if r["judge"]["contradiction"]) / cn,
            "unknown_rate": sum(1 for r in cat_rows if r["prediction"].strip().lower() == "unknown") / cn,
            "avg_context_tokens": sum(r.get("context_tokens", 0) for r in cat_rows) / cn,
            "avg_answer_latency_ms": sum(r.get("answer_latency_ms", 0.0) for r in cat_rows) / cn,
            "avg_judge_latency_ms": sum(r.get("judge_latency_ms", 0.0) for r in cat_rows) / cn,
        }

    return {
        "n_questions": n,
        "judge_accuracy": correct / n,
        "judge_partial_accuracy": partial / n,
        "contradiction_rate": contradiction / n,
        "unknown_rate": unknown / n,
        "avg_context_tokens": avg("context_tokens"),
        "avg_answer_latency_ms": avg("answer_latency_ms"),
        "avg_judge_latency_ms": avg("judge_latency_ms"),
        "by_category": by_cat_summary,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--answer-model", default="gpt-4o-mini")
    ap.add_argument("--judge-model", default="gpt-4o-mini")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is required")

    client_kwargs = {"api_key": api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    rows = read_jsonl(args.retrieval)
    if args.limit:
        rows = rows[: args.limit]

    judged_rows = []

    for i, row in enumerate(rows, start=1):
        question = row.get("question", "")
        gold = row.get("gold_answer") or row.get("answer", "")
        context = row.get("debug_context") or row.get("context") or ""

        if not context:
            memories = row.get("retrieved_memories") or row.get("retrieved_context") or []
            if isinstance(memories, list):
                parts = []
                for m in memories:
                    if isinstance(m, dict):
                        parts.append(str(m.get("text", "")))
                    else:
                        parts.append(str(m))
                context = "\n".join(parts)

        t0 = time.perf_counter()
        prediction = call_chat(
            client,
            args.answer_model,
            build_answer_prompt(question, context),
            temperature=0.0,
        )
        answer_latency_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        judge_text = call_chat(
            client,
            args.judge_model,
            build_judge_prompt(question, gold, prediction),
            temperature=0.0,
        )
        judge_latency_ms = (time.perf_counter() - t1) * 1000

        judge = parse_judge_json(judge_text)

        out = dict(row)
        out.update(
            {
                "prediction": prediction,
                "judge_raw": judge_text,
                "judge": judge,
                "answer_model": args.answer_model,
                "judge_model": args.judge_model,
                "answer_latency_ms": answer_latency_ms,
                "judge_latency_ms": judge_latency_ms,
            }
        )
        judged_rows.append(out)

        if i % 25 == 0:
            print(f"processed {i}/{len(rows)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(str(out_dir / "judged.jsonl"), judged_rows)

    summary = summarize(judged_rows)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
