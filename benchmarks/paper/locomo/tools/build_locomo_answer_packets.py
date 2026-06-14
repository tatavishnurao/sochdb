import argparse
import json
import re
from pathlib import Path
from datetime import datetime, timedelta

MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}

def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                print(f"ERROR: bad JSON in {path} line {line_no}: {e}")
                raise SystemExit
    return rows

def compact(x):
    return str(x or "").strip()

def parse_timestamp_date(timestamp):
    text = compact(timestamp)
    m = re.search(
        r"\b(\d{1,2})\s+"
        r"(January|February|March|April|May|June|July|August|September|October|November|December)"
        r",?\s+(\d{4})\b",
        text,
    )
    if not m:
        return None

    try:
        return datetime(int(m.group(3)), MONTHS[m.group(2)], int(m.group(1))).date()
    except Exception:
        return None

def human_date(d):
    return f"{d.day} {d.strftime('%B')} {d.year}"

def resolved_time_notes(memory):
    text = compact(memory.get("text")).lower()
    base = parse_timestamp_date(memory.get("timestamp"))
    if base is None:
        return []

    notes = []
    if "yesterday" in text:
        notes.append(f"yesterday = {human_date(base - timedelta(days=1))}")
    if "tomorrow" in text:
        notes.append(f"tomorrow = {human_date(base + timedelta(days=1))}")
    if "today" in text:
        notes.append(f"today = {human_date(base)}")
    return notes

def approx_tokens(text):
    return max(1, len(text) // 4)

def memory_line(memory, idx, max_chars):
    mid = memory.get("memory_id")
    speaker = compact(memory.get("speaker"))
    timestamp = compact(memory.get("timestamp"))
    session = compact(memory.get("session"))
    dia_id = compact(memory.get("dia_id"))
    text = compact(memory.get("text"))

    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "..."

    header = (
        f"{idx}. memory_id={mid}"
        f" | speaker={speaker or 'unknown'}"
        f" | timestamp={timestamp or 'unknown'}"
        f" | session={session or 'unknown'}"
        f" | dia_id={dia_id or 'unknown'}"
    )

    lines = [header, f"   text: {text}"]

    notes = resolved_time_notes(memory)
    for note in notes:
        lines.append(f"   resolved_time: {note}")

    return "\n".join(lines)

def build_packet(row, memory_by_id, mode, top_k, max_chars):
    if mode == "oracle":
        ids = row.get("evidence_memory_ids") or []
    elif mode == "retrieved":
        ids = (row.get("retrieved_memory_ids") or [])[:top_k]
    else:
        print(f"ERROR: unknown mode {mode}")
        raise SystemExit

    evidence = []
    missing = []

    seen = set()
    for x in ids:
        try:
            mid = int(x)
        except Exception:
            continue
        if mid in seen:
            continue
        seen.add(mid)

        m = memory_by_id.get(mid)
        if m:
            evidence.append(m)
        else:
            missing.append(mid)

    evidence_lines = [
        memory_line(m, idx + 1, max_chars=max_chars)
        for idx, m in enumerate(evidence)
    ]

    question = compact(row.get("question"))
    category = compact(row.get("category"))
    gold_answer = row.get("gold_answer")

    packet = f"""QUESTION
{question}

CATEGORY
{category}

ANSWER RULES
- Answer using only the evidence below.
- Return the shortest correct answer.
- If the answer is a date, normalize it as: day month year.
- If the answer is a list, return comma-separated items.
- Do not explain unless the answer requires a short phrase.
- If evidence is insufficient, answer: insufficient evidence.

EVIDENCE
{chr(10).join(evidence_lines) if evidence_lines else "No evidence provided."}

FINAL ANSWER
"""

    gold_ids = set(map(str, row.get("evidence_memory_ids") or []))
    used_ids = set(map(str, [m.get("memory_id") for m in evidence]))
    overlap = gold_ids & used_ids

    return {
        "question_id": row.get("question_id"),
        "sample_id": row.get("sample_id"),
        "category": row.get("category"),
        "question": row.get("question"),
        "gold_answer": gold_answer,
        "gold_evidence_memory_ids": row.get("evidence_memory_ids") or [],
        "packet_mode": mode,
        "top_k": top_k if mode == "retrieved" else None,
        "packet_memory_ids": [m.get("memory_id") for m in evidence],
        "missing_memory_ids": missing,
        "gold_evidence_covered": bool(overlap) if gold_ids else None,
        "gold_evidence_recall": (len(overlap) / len(gold_ids)) if gold_ids else None,
        "approx_context_tokens": approx_tokens(packet),
        "answer_packet": packet,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memories", required=True)
    ap.add_argument("--retrieval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["oracle", "retrieved"], required=True)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-chars-per-memory", type=int, default=500)
    args = ap.parse_args()

    memories = read_jsonl(args.memories)
    rows = read_jsonl(args.retrieval)

    memory_by_id = {}
    for m in memories:
        mid = m.get("memory_id")
        if mid is None:
            continue
        try:
            memory_by_id[int(mid)] = m
        except Exception:
            pass

    if args.limit is not None:
        rows = rows[:args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    packets = [
        build_packet(
            row=r,
            memory_by_id=memory_by_id,
            mode=args.mode,
            top_k=args.top_k,
            max_chars=args.max_chars_per_memory,
        )
        for r in rows
    ]

    with out.open("w", encoding="utf-8") as f:
        for p in packets:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    avg_tokens = sum(p["approx_context_tokens"] for p in packets) / len(packets) if packets else 0
    covered = [p for p in packets if p["gold_evidence_covered"] is True]
    scored = [p for p in packets if p["gold_evidence_covered"] is not None]
    avg_gold_recall = (
        sum(p["gold_evidence_recall"] for p in scored) / len(scored)
        if scored else 0
    )

    print(f"wrote={len(packets)} to {out}")
    print(f"mode={args.mode}")
    print(f"top_k={args.top_k}")
    print(f"avg_context_tokens={avg_tokens:.1f}")
    print(f"gold_covered={len(covered)}/{len(scored)}")
    print(f"avg_gold_recall={avg_gold_recall:.4f}")

if __name__ == "__main__":
    main()
