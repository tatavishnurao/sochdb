#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

COMMON = Path(__file__).resolve().parents[2] / "common"
LONGMEMEVAL_RUNNERS = Path(__file__).resolve().parents[2] / "longmemeval" / "runners"
for path in (COMMON, LONGMEMEVAL_RUNNERS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from memory_schema import read_json_or_jsonl, write_jsonl  # type: ignore
from normalize_longmemeval import inspect_rows, normalize_sample  # type: ignore


def normalize_file(path: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = read_json_or_jsonl(path)
    memories: list[dict[str, Any]] = []
    questions: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        sample_memories, sample_questions = normalize_sample(row, idx)
        for memory in sample_memories:
            memory["metadata"]["source"] = "convomem"
        for question in sample_questions:
            question["metadata"]["source"] = "convomem"
        memories.extend(sample_memories)
        questions.extend(sample_questions)
    return memories, questions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize a local ConvoMem JSON/JSONL slice. This does not download ConvoMem."
    )
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
    print("ConvoMem full dataset download/run remains disabled until explicitly confirmed.")


if __name__ == "__main__":
    main()
