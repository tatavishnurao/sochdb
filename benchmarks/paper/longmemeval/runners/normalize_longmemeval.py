#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

COMMON = Path(__file__).resolve().parents[2] / "common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from memory_schema import (  # type: ignore
    first_present,
    match_evidence_texts_to_memories,
    normalize_evidence_ids,
    read_json_or_jsonl,
    write_jsonl,
)


MEMORY_CONTAINER_KEYS = (
    "memories",
    "memory",
    "haystack",
    "haystack_sessions",
    "context",
    "conversation",
    "messages",
    "history",
    "sessions",
)
QUESTION_CONTAINER_KEYS = ("questions", "qa", "qas")
TEXT_KEYS = ("text", "content", "message", "utterance", "value")
QUESTION_KEYS = ("question", "query", "input")
ANSWER_KEYS = ("answer", "gold_answer", "target", "output")
EVIDENCE_ID_KEYS = (
    "evidence_memory_ids",
    "evidence_ids",
    "evidence_turn_ids",
    "evidence_message_ids",
    "supporting_memory_ids",
)
EVIDENCE_TEXT_KEYS = (
    "evidence_texts",
    "evidence_spans",
    "supporting_facts",
    "evidence",
)


def inspect_rows(rows: list[dict[str, Any]]) -> None:
    print(f"rows={len(rows)}")
    if not rows:
        return
    row = rows[0]
    print(f"first_row_keys={list(row.keys())}")
    for key, value in row.items():
        if isinstance(value, list):
            print(f"{key}: list len={len(value)}")
            if value and isinstance(value[0], dict):
                print(f"{key}[0].keys={list(value[0].keys())}")
        elif isinstance(value, dict):
            print(f"{key}: dict keys={list(value.keys())[:30]}")
        else:
            print(f"{key}: {type(value).__name__}")


def flatten_messages(value: Any, session_hint: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if value is None:
        return rows
    if isinstance(value, str):
        return [{"text": value, "session": session_hint}]
    if isinstance(value, dict):
        message_lists = []
        for key in ("messages", "turns", "utterances", "conversation"):
            if isinstance(value.get(key), list):
                message_lists.append((key, value[key]))
        if message_lists:
            for _, items in message_lists:
                rows.extend(flatten_messages(items, session_hint or str(value.get("session_id") or value.get("session") or "")))
            return rows
        session_like_keys = [k for k, v in value.items() if isinstance(v, list)]
        if session_like_keys:
            for key in session_like_keys:
                rows.extend(flatten_messages(value[key], session_hint or str(key)))
            return rows
        text = first_present(value, TEXT_KEYS)
        if text is not None:
            row = dict(value)
            if session_hint and not row.get("session"):
                row["session"] = session_hint
            return [row]
        return rows
    if isinstance(value, list):
        for idx, item in enumerate(value):
            if isinstance(item, dict) and any(isinstance(item.get(k), list) for k in ("messages", "turns", "utterances")):
                rows.extend(flatten_messages(item, session_hint or str(item.get("session_id") or item.get("session") or idx)))
            else:
                rows.extend(flatten_messages(item, session_hint))
    return rows


def extract_memory_source(sample: dict[str, Any]) -> Any:
    for key in MEMORY_CONTAINER_KEYS:
        if key in sample:
            return sample[key]
    return None


def normalize_memory_rows(sample: dict[str, Any], sample_id: str) -> list[dict[str, Any]]:
    source = extract_memory_source(sample)
    raw_messages = flatten_messages(source)
    memories = []
    for idx, raw in enumerate(raw_messages, start=1):
        text = first_present(raw, TEXT_KEYS)
        if text is None or not str(text).strip():
            continue
        memory_id = first_present(raw, ("memory_id", "message_id", "turn_id", "id", "uuid"))
        if memory_id is None:
            memory_id = f"{sample_id}_m{idx}"
        memories.append(
            {
                "sample_id": sample_id,
                "memory_id": str(memory_id),
                "speaker": first_present(raw, ("speaker", "role", "author")),
                "session": first_present(raw, ("session", "session_id", "conversation_id")),
                "timestamp": first_present(raw, ("timestamp", "time", "date", "created_at")),
                "turn_id": first_present(raw, ("turn_id", "turn", "idx", "index", "message_id", "id")),
                "text": str(text),
                "metadata": {"source": "longmemeval", "raw": raw},
            }
        )
    return memories


def evidence_text_values(row: dict[str, Any]) -> list[Any]:
    values = []
    for key in EVIDENCE_TEXT_KEYS:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    text = first_present(item, (*TEXT_KEYS, "span", "fact"))
                    if text is not None:
                        values.append(text)
                else:
                    values.append(item)
        elif isinstance(value, dict):
            text = first_present(value, (*TEXT_KEYS, "span", "fact"))
            if text is not None:
                values.append(text)
        elif key != "evidence":
            values.append(value)
    return values


def normalize_question_row(
    raw: dict[str, Any],
    sample: dict[str, Any],
    sample_id: str,
    q_idx: int,
    memories: list[dict[str, Any]],
) -> dict[str, Any] | None:
    question = first_present(raw, QUESTION_KEYS)
    if question is None:
        return None

    evidence_ids = []
    for key in EVIDENCE_ID_KEYS:
        evidence_ids.extend(normalize_evidence_ids(raw.get(key)))
    evidence_ids = list(dict.fromkeys(evidence_ids))

    failed_texts: list[str] = []
    if not evidence_ids:
        matched, failed_texts = match_evidence_texts_to_memories(evidence_text_values(raw), memories)
        evidence_ids = matched

    if evidence_ids:
        status = "id_labels_available" if not failed_texts else "text_mapping_partial"
    elif evidence_text_values(raw):
        status = "span_mapping_failed"
    else:
        status = "no_evidence_available"

    question_id = first_present(raw, ("question_id", "query_id", "id", "qid"))
    if question_id is None:
        question_id = f"{sample_id}_q{q_idx}"

    return {
        "sample_id": sample_id,
        "question_id": str(question_id),
        "question": str(question),
        "answer": first_present(raw, ANSWER_KEYS),
        "category": first_present(raw, ("category", "question_type", "type", "task")),
        "evidence_memory_ids": evidence_ids,
        "evidence_mapping_status": status,
        "evidence_mapping_failed_texts": failed_texts,
        "metadata": {
            "source": "longmemeval",
            "raw_question_keys": list(raw.keys()),
            "sample_keys": list(sample.keys()),
        },
    }


def normalize_sample(sample: dict[str, Any], sample_idx: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sample_id = str(first_present(sample, ("sample_id", "conversation_id", "id", "uuid"), f"sample_{sample_idx}"))
    memories = normalize_memory_rows(sample, sample_id)

    raw_questions = []
    for key in QUESTION_CONTAINER_KEYS:
        if isinstance(sample.get(key), list):
            raw_questions.extend(sample[key])
    if not raw_questions and first_present(sample, QUESTION_KEYS) is not None:
        raw_questions = [sample]

    questions = []
    for idx, raw in enumerate(raw_questions, start=1):
        if isinstance(raw, dict):
            q = normalize_question_row(raw, sample, sample_id, idx, memories)
            if q:
                questions.append(q)
    return memories, questions


def normalize_file(path: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = read_json_or_jsonl(path)
    memories: list[dict[str, Any]] = []
    questions: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        sample_memories, sample_questions = normalize_sample(row, idx)
        memories.extend(sample_memories)
        questions.extend(sample_questions)
    return memories, questions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--memories-out", required=True)
    parser.add_argument("--questions-out", required=True)
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()

    rows = read_json_or_jsonl(args.input)
    if args.inspect:
        inspect_rows(rows)
    memories, questions = normalize_file(args.input)
    write_jsonl(args.memories_out, memories)
    write_jsonl(args.questions_out, questions)
    print(f"Wrote memories: {args.memories_out} ({len(memories)} rows)")
    print(f"Wrote questions: {args.questions_out} ({len(questions)} rows)")
    print(f"Questions with evidence IDs: {sum(1 for q in questions if q['evidence_memory_ids'])}")
    print(f"Questions without scored evidence: {sum(1 for q in questions if not q['evidence_memory_ids'])}")


if __name__ == "__main__":
    main()
