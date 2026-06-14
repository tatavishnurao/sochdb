#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


CATEGORY_MAP = {
    1: "single_hop",
    2: "temporal",
    3: "multi_hop",
    4: "open_domain",
    5: "adversarial",
}


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def session_num_from_key(key: str) -> int:
    m = re.match(r"session_(\d+)$", key)
    if not m:
        return 10**9
    return int(m.group(1))


def parse_evidence_ref(ref: str) -> Tuple[int, int]:
    """
    LoCoMo evidence looks like D1:3, D23:12, etc.
    D<num> = session number.
    after colon = dialogue id inside that session.
    """
    m = re.match(r"D(\d+):(\d+)", str(ref).strip())
    if not m:
        return -1, -1
    return int(m.group(1)), int(m.group(2))


def normalize_answer(answer: Any) -> str:
    if answer is None:
        return ""
    if isinstance(answer, str):
        return answer
    return str(answer)


def inspect_schema(data):
    print("Top-level type:", type(data).__name__)
    print("Num samples:", len(data) if isinstance(data, list) else "n/a")

    sample = data[0] if isinstance(data, list) and data else data
    print("Sample keys:", list(sample.keys()))

    conv = sample.get("conversation", {})
    print("Conversation keys sample:", list(conv.keys())[:30])

    session_keys = [k for k, v in conv.items() if re.match(r"session_\d+$", k)]
    session_keys = sorted(session_keys, key=session_num_from_key)
    print("Session keys:", session_keys[:5])

    if session_keys:
        first_session = conv[session_keys[0]]
        print("First session type:", type(first_session).__name__)
        if isinstance(first_session, list) and first_session:
            print("First turn keys:", list(first_session[0].keys()))
            print("First turn:", first_session[0])

    qa = sample.get("qa", [])
    print("QA count:", len(qa))
    if qa:
        print("First QA keys:", list(qa[0].keys()))
        print("First QA:", qa[0])


def convert_locomo(input_path: str):
    data = load_json(input_path)

    if not isinstance(data, list):
        raise ValueError("Expected LoCoMo JSON to be a list of samples.")

    memories = []
    questions = []

    memory_id = 1

    for sample_idx, sample in enumerate(data):
        sample_id = sample.get("sample_id") or f"sample_{sample_idx}"
        conversation = sample.get("conversation") or {}

        speaker_a = conversation.get("speaker_a")
        speaker_b = conversation.get("speaker_b")

        evidence_to_memory_id: Dict[str, int] = {}

        session_keys = [
            k for k, v in conversation.items()
            if re.match(r"session_\d+$", k) and isinstance(v, list)
        ]
        session_keys = sorted(session_keys, key=session_num_from_key)

        for session_key in session_keys:
            s_num = session_num_from_key(session_key)
            session_date = conversation.get(f"{session_key}_date_time")

            turns = conversation.get(session_key, [])

            for turn_idx, turn in enumerate(turns):
                if not isinstance(turn, dict):
                    continue

                raw_dia_id = turn.get("dia_id")
                if raw_dia_id is None:
                    raw_dia_id = f"D{s_num}:{turn_idx + 1}"

                raw_dia_id = str(raw_dia_id)

                # LoCoMo dia_id is usually already like "D1:3".
                # Keep the original string, but also extract the numeric turn index.
                if ":" in raw_dia_id:
                    evidence_ref = raw_dia_id
                    try:
                        dia_num = int(raw_dia_id.split(":")[-1])
                    except Exception:
                        dia_num = turn_idx + 1
                else:
                    dia_num = int(raw_dia_id)
                    evidence_ref = f"D{s_num}:{dia_num}"

                speaker = turn.get("speaker") or turn.get("role") or "unknown"
                text = turn.get("text") or turn.get("content") or ""

                if not str(text).strip():
                    continue

                evidence_to_memory_id[evidence_ref] = memory_id

                memories.append(
                    {
                        "memory_id": memory_id,
                        "sample_id": sample_id,
                        "session": session_key,
                        "session_num": s_num,
                        "dia_id": raw_dia_id,
                        "dia_num": dia_num,
                        "evidence_ref": evidence_ref,
                        "speaker": speaker,
                        "speaker_a": speaker_a,
                        "speaker_b": speaker_b,
                        "text": text,
                        "timestamp": session_date,
                        "source": "locomo",
                    }
                )

                memory_id += 1

        qa_rows = sample.get("qa", [])

        for q_idx, qa in enumerate(qa_rows):
            if not isinstance(qa, dict):
                continue

            question = qa.get("question")
            if not question:
                continue

            answer = normalize_answer(qa.get("answer"))
            evidence_refs = qa.get("evidence") or []
            if isinstance(evidence_refs, str):
                evidence_refs = [evidence_refs]

            evidence_memory_ids = []
            for ref in evidence_refs:
                mid = evidence_to_memory_id.get(str(ref))
                if mid is not None:
                    evidence_memory_ids.append(mid)

            category_id = qa.get("category")
            try:
                category_id_int = int(category_id)
            except Exception:
                category_id_int = None

            category = CATEGORY_MAP.get(category_id_int, f"category_{category_id}")

            questions.append(
                {
                    "question_id": f"{sample_id}_q{q_idx + 1}",
                    "sample_id": sample_id,
                    "question": question,
                    "gold_answer": answer,
                    "answer": answer,
                    "evidence_refs": evidence_refs,
                    "evidence_memory_ids": evidence_memory_ids,
                    "category_id": category_id_int,
                    "category": category,
                    "source": "locomo",
                }
            )

    return memories, questions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--memories-out", required=True)
    parser.add_argument("--questions-out", required=True)
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()

    data = load_json(args.input)

    if args.inspect:
        inspect_schema(data)

    memories, questions = convert_locomo(args.input)

    write_jsonl(args.memories_out, memories)
    write_jsonl(args.questions_out, questions)

    print(f"Wrote memories:  {args.memories_out} ({len(memories)} rows)")
    print(f"Wrote questions: {args.questions_out} ({len(questions)} rows)")

    category_counts = {}
    for q in questions:
        category_counts[q["category"]] = category_counts.get(q["category"], 0) + 1

    print("Category counts:")
    for k, v in sorted(category_counts.items()):
        print(f"  {k}: {v}")

    missing_evidence = sum(1 for q in questions if q["evidence_refs"] and not q["evidence_memory_ids"])
    print(f"Questions with evidence refs but no mapped memory ids: {missing_evidence}")


if __name__ == "__main__":
    main()
