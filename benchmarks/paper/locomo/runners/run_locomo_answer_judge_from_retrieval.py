#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from openai import OpenAI


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def load_memory_index(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows = read_jsonl(path)
    index: dict[tuple[str, int], dict[str, Any]] = {}

    for row in rows:
        sample_id = row.get("sample_id")
        memory_id = row.get("memory_id", row.get("id"))

        if sample_id is None or memory_id is None:
            continue

        index[(str(sample_id), int(memory_id))] = row

    return index


def parse_locomo_timestamp_date(timestamp: str) -> str | None:
    # Example: "1:56 pm on 8 May, 2023" -> "8 May 2023"
    if not timestamp:
        return None

    match = re.search(
        r"\bon\s+(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})",
        timestamp,
    )
    if not match:
        return None

    day, month, year = match.groups()
    return f"{int(day)} {month} {year}"


def previous_day_from_locomo_timestamp(timestamp: str) -> str | None:
    from datetime import datetime, timedelta

    date_text = parse_locomo_timestamp_date(timestamp)
    if not date_text:
        return None

    for fmt in ("%d %B %Y", "%d %b %Y"):
        try:
            dt = datetime.strptime(date_text, fmt)
            prev = dt - timedelta(days=1)
            return f"{prev.day} {prev.strftime('%B')} {prev.year}"
        except ValueError:
            continue

    return None


def previous_year_from_locomo_timestamp(timestamp: str) -> str | None:
    date_text = parse_locomo_timestamp_date(timestamp)
    if not date_text:
        return None

    match = re.search(r"(\d{4})$", date_text)
    if not match:
        return None

    return str(int(match.group(1)) - 1)


def relative_time_notes(text: str, timestamp: str) -> list[str]:
    lower = text.lower()
    notes = []

    if "yesterday" in lower:
        resolved = previous_day_from_locomo_timestamp(timestamp)
        if resolved:
            notes.append(f"Relative time resolution: 'yesterday' = {resolved}.")

    if "last year" in lower:
        resolved = previous_year_from_locomo_timestamp(timestamp)
        if resolved:
            notes.append(f"Relative time resolution: 'last year' = {resolved}.")

    return notes


def render_answer_ready_context(
    retrieval_row: dict[str, Any],
    memory_index: dict[tuple[str, int], dict[str, Any]] | None,
    context_source: str = "retrieved",
    max_context_memories: int | None = None,
) -> str:
    if not memory_index:
        return retrieval_row.get("debug_context") or ""

    sample_id = str(retrieval_row.get("sample_id"))
    if context_source == "evidence":
        retrieved_ids = retrieval_row.get("evidence_memory_ids") or []
    else:
        retrieved_ids = retrieval_row.get("retrieved_memory_ids") or []
    
    if max_context_memories is not None:
        retrieved_ids = retrieved_ids[:max_context_memories]

    lines = []
    for rank, memory_id in enumerate(retrieved_ids, start=1):
        memory = memory_index.get((sample_id, int(memory_id)))
        if not memory:
            continue

        timestamp = memory.get("timestamp") or ""
        speaker = memory.get("speaker") or ""
        dia_id = memory.get("dia_id") or ""
        session = memory.get("session") or ""
        text = memory.get("text") or ""

        lines.append(
            f"[rank={rank} memory_id={memory_id} dia_id={dia_id} "
            f"session={session} timestamp={timestamp} speaker={speaker}]\n"
            f"{speaker}: {text}"
        )

        for note in relative_time_notes(text, timestamp):
            lines.append(note)

    return "\n".join(lines).strip()

    
def response_text(resp: Any) -> str:
    content = getattr(resp.choices[0].message, "content", "") or ""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()
    return str(content).strip()


def usage_total_tokens(resp: Any) -> int | None:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return None
    return getattr(usage, "total_tokens", None)


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None

    try:
        obj = json.loads(match.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None

    return None


def call_chat(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float = 0.0,
) -> tuple[str, int | None, float]:
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return response_text(resp), usage_total_tokens(resp), elapsed_ms


def answer_question(
    client: OpenAI,
    model: str,
    question: str,
    context: str,
    max_tokens: int,
) -> tuple[str, int | None, float]:
    messages = [
        {
            "role": "system",
            "content": (
                "Use only the provided conversation memory context. "
                "Return the shortest exact answer supported by the context. "
                "For date questions, use timestamp metadata and relative-time notes to resolve dates like yesterday, last week, and last year. "
                "For list questions, return only the requested items, separated by commas. "
                "Do not add extra inferred fields. "
                "If the answer is not supported by the context, answer exactly: I don't know. "
                "Return only the answer, no explanation."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:",
        },
    ]
    return call_chat(client, model, messages, max_tokens=max_tokens, temperature=0.0)


def judge_answer(
    client: OpenAI,
    model: str,
    question: str,
    gold_answer: str,
    predicted_answer: str,
    max_tokens: int,
) -> tuple[dict[str, Any], str, int | None, float]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict but fair QA evaluator. "
                "Return exactly one compact JSON object and nothing else. "
                "Do not write markdown. Do not think aloud. "
                "Accept paraphrases and equivalent date formats. "
                "Reject answers that are wrong, unsupported, too vague, or contain extra unsupported claims. "
                'The JSON must have keys: "score", "verdict", "reason". '
                'Use score 1 only when the predicted answer fully answers the question.'
            ),
        },
        {
            "role": "user",
            "content": (
                "Evaluate this answer.\n\n"
                f"Question: {question}\n"
                f"Gold answer: {gold_answer}\n"
                f"Predicted answer: {predicted_answer}\n\n"
                "Return exactly this JSON shape:\n"
                '{"score": 0 or 1, "verdict": "correct" or "incorrect", "reason": "short reason"}'
            ),
        },
    ]

    raw, tokens, elapsed_ms = call_chat(client, model, messages, max_tokens=max_tokens, temperature=0.0)
    parsed = extract_json_object(raw)

    if parsed is None:
        return {
            "score": 0,
            "verdict": "judge_parse_failed",
            "reason": "Judge did not return valid JSON.",
            "parse_ok": False,
        }, raw, tokens, elapsed_ms

    score = parsed.get("score", 0)
    try:
        score = int(score)
    except Exception:
        score = 0

    score = 1 if score == 1 else 0
    verdict = parsed.get("verdict") or ("correct" if score == 1 else "incorrect")
    reason = parsed.get("reason") or ""

    return {
        "score": score,
        "verdict": str(verdict),
        "reason": str(reason),
        "parse_ok": True,
    }, raw, tokens, elapsed_ms


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [r for r in rows if r.get("judge_parse_ok") is not None]

    by_category = defaultdict(list)
    for row in scored:
        by_category[row.get("category", "unknown")].append(row)

    def agg(items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            return {
                "n": 0,
                "judge_accuracy": 0.0,
                "avg_context_tokens": 0.0,
                "avg_answer_latency_ms": 0.0,
                "avg_judge_latency_ms": 0.0,
                "judge_parse_rate": 0.0,
            }

        return {
            "n": len(items),
            "judge_accuracy": sum(float(x.get("judge_score", 0)) for x in items) / len(items),
            "avg_context_tokens": sum(float(x.get("approx_context_tokens") or 0) for x in items) / len(items),
            "avg_answer_latency_ms": sum(float(x.get("answer_latency_ms") or 0) for x in items) / len(items),
            "avg_judge_latency_ms": sum(float(x.get("judge_latency_ms") or 0) for x in items) / len(items),
            "judge_parse_rate": sum(1.0 if x.get("judge_parse_ok") else 0.0 for x in items) / len(items),
        }

    summary = {
        "benchmark": "locomo-answer-judge-from-retrieval",
        "mode": "answer_judge",
        "overall": agg(scored),
        "by_category": {cat: agg(items) for cat, items in sorted(by_category.items())},
    }

    non_adv = [r for r in scored if r.get("category") != "adversarial"]
    summary["mem0_comparable_excluding_adversarial"] = agg(non_adv)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--answer-model", default=os.environ.get("NVIDIA_ANSWER_MODEL"))
    parser.add_argument("--judge-model", default=os.environ.get("NVIDIA_JUDGE_MODEL"))
    parser.add_argument("--base-url", default=os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--exclude-category", action="append", default=[])
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--answer-max-tokens", type=int, default=128)
    parser.add_argument("--judge-max-tokens", type=int, default=256)
    parser.add_argument("--memories", default=None)
    parser.add_argument("--context-source", choices=["retrieved", "evidence"], default="retrieved")
    parser.add_argument("--max-context-memories", type=int, default=None)
    parser.add_argument("--judge-retries", type=int, default=0)
    args = parser.parse_args()

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise SystemExit("NVIDIA_API_KEY is empty")

    if not args.answer_model:
        raise SystemExit("NVIDIA_ANSWER_MODEL / --answer-model is empty")

    if not args.judge_model:
        raise SystemExit("NVIDIA_JUDGE_MODEL / --judge-model is empty")

    retrieval_rows = read_jsonl(Path(args.retrieval))
    memory_index = load_memory_index(Path(args.memories)) if args.memories else None
    if args.context_source == "evidence" and memory_index is None:
        raise SystemExit("--context-source evidence requires --memories")
        
    exclude = set(args.exclude_category or [])
    filtered = []
    for row in retrieval_rows:
        if row.get("category") in exclude:
            continue
        if not row.get("gold_answer"):
            continue
        filtered.append(row)

    if args.limit is not None:
        filtered = filtered[: args.limit]

    client = OpenAI(base_url=args.base_url, api_key=api_key)

    results = []
    for i, row in enumerate(filtered, start=1):
        question = row.get("question", "")
        gold_answer = row.get("gold_answer", "")
        context = render_answer_ready_context(
            row,
            memory_index,
            context_source=args.context_source,
            max_context_memories=args.max_context_memories,
        )

        if not context.strip():
            result = {
                **row,
                "answer": "",
                "judge_score": 0,
                "judge_verdict": "empty_context",
                "judge_reason": "No context was available in retrieval row.",
                "judge_parse_ok": True,
                "answer_model": args.answer_model,
                "judge_model": args.judge_model,
            }
            results.append(result)
            continue

        print(f"[{i}/{len(filtered)}] {row.get('question_id')} category={row.get('category')}")

        answer, answer_tokens, answer_latency = answer_question(
            client=client,
            model=args.answer_model,
            question=question,
            context=context,
            max_tokens=args.answer_max_tokens,
        )

        judge, judge_raw, judge_tokens, judge_latency = judge_answer(
            client=client,
            model=args.judge_model,
            question=question,
            gold_answer=gold_answer,
            predicted_answer=answer,
            max_tokens=args.judge_max_tokens,
        )

        for retry_idx in range(args.judge_retries):
            if judge.get("parse_ok"):
                break

            retry_judge, retry_raw, retry_tokens, retry_latency = judge_answer(
                client=client,
                model=args.judge_model,
                question=question,
                gold_answer=gold_answer,
                predicted_answer=answer,
                max_tokens=args.judge_max_tokens,
            )

            judge_raw = (judge_raw or "") + f"\n\n--- JUDGE RETRY {retry_idx + 1} ---\n\n" + (retry_raw or "")
            judge_latency += retry_latency

            if retry_tokens is not None:
                judge_tokens = (judge_tokens or 0) + retry_tokens

            judge = retry_judge

        result = {
            **row,
            "answer": answer,
            "answer_model": args.answer_model,
            "judge_model": args.judge_model,
            "judge_score": judge["score"],
            "judge_verdict": judge["verdict"],
            "judge_reason": judge["reason"],
            "judge_parse_ok": judge["parse_ok"],
            "judge_raw": judge_raw,
            "answer_total_tokens": answer_tokens,
            "judge_total_tokens": judge_tokens,
            "answer_latency_ms": answer_latency,
            "judge_latency_ms": judge_latency,
        }
        results.append(result)

        # Checkpoint after every completed row so rate limits do not lose progress.
        write_jsonl(Path(args.out), results)

        partial_summary = summarize(results)
        partial_summary["retrieval"] = args.retrieval
        partial_summary["answer_model"] = args.answer_model
        partial_summary["judge_model"] = args.judge_model
        partial_summary["excluded_categories"] = sorted(exclude)
        partial_summary["partial"] = True

        Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_out).write_text(
            json.dumps(partial_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        if args.sleep > 0:
            time.sleep(args.sleep)

    out_path = Path(args.out)
    summary_path = Path(args.summary_out)

    write_jsonl(out_path, results)

    summary = summarize(results)
    summary["retrieval"] = args.retrieval
    summary["answer_model"] = args.answer_model
    summary["judge_model"] = args.judge_model
    summary["excluded_categories"] = sorted(exclude)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
