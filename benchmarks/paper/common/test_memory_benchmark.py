#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

COMMON = Path(__file__).resolve().parent
LONGMEMEVAL = COMMON.parent / "longmemeval" / "runners"
for path in (COMMON, LONGMEMEVAL):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from memory_retrieval import run_retrieval  # type: ignore
from memory_schema import render_memory_text, write_jsonl  # type: ignore
from normalize_longmemeval import normalize_sample  # type: ignore
from score_memory_retrieval import score_rows  # type: ignore


ARTIFACTS = COMMON / "test_artifacts"


def test_metadata_rendering_with_missing_fields() -> None:
    rendered = render_memory_text({"memory_id": "m1", "text": "hello"}, "metadata")
    assert rendered == "text: hello"

    rendered = render_memory_text(
        {"speaker": "A", "timestamp": "", "session": "s1", "turn_id": 2, "text": "hello"},
        "metadata",
    )
    assert "speaker: A" in rendered
    assert "session: s1" in rendered
    assert "turn_id: 2" in rendered
    assert "timestamp:" not in rendered
    assert rendered.endswith("text: hello")


def test_longmemeval_normalization_and_span_mapping() -> None:
    sample = {
        "id": "sample-a",
        "haystack_sessions": [
            {
                "session_id": "s1",
                "messages": [
                    {"message_id": "m1", "role": "user", "content": "Alice bought tea."},
                    {"message_id": "m2", "role": "assistant", "content": "Bob bought coffee."},
                ],
            }
        ],
        "question": "What did Alice buy?",
        "answer": "tea",
        "evidence": ["Alice bought tea."],
        "category": "single_hop",
    }
    memories, questions = normalize_sample(sample, 1)
    assert len(memories) == 2
    assert memories[0]["memory_id"] == "m1"
    assert len(questions) == 1
    assert questions[0]["evidence_memory_ids"] == ["m1"]
    assert questions[0]["evidence_mapping_status"] in {"id_labels_available", "text_mapping_partial"}


def test_scoring_and_buckets_and_length_diagnostics() -> None:
    rows = [
        {
            "question_id": "q1",
            "sample_id": "s",
            "category": "single",
            "evidence_memory_ids": ["m1", "m2"],
            "retrieved_memory_ids": ["m1", "m3"],
        },
        {
            "question_id": "q2",
            "sample_id": "s",
            "category": "single",
            "evidence_memory_ids": [],
            "evidence_mapping_status": "no_evidence_available",
            "retrieved_memory_ids": ["m4"],
        },
    ]
    summary = score_rows(rows, [1, 2, 20])
    assert summary["rows"] == 2
    assert summary["scored_rows"] == 1
    assert summary["unscored_reasons"]["missing_evidence_labels"] == 1
    assert summary["ks"]["1"]["available"] is True
    assert summary["ks"]["1"]["hit"] == 1.0
    assert summary["ks"]["1"]["recall"] == 0.5
    assert summary["ks"]["20"]["available"] is False
    assert summary["ks"]["2"]["by_evidence_count"]["2"]["rows"] == 1


def test_hash_local_retrieval_smoke() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    memories_path = ARTIFACTS / "smoke_memories.jsonl"
    questions_path = ARTIFACTS / "smoke_questions.jsonl"
    out_path = ARTIFACTS / "smoke_retrieval.jsonl"
    cache_path = ARTIFACTS / "smoke_embedding_cache.jsonl"

    memories = [
        {
            "sample_id": "s1",
            "memory_id": "m1",
            "speaker": "Alice",
            "session": "s1",
            "turn_id": 1,
            "text": "Alice bought green tea at the market.",
            "metadata": {},
        },
        {
            "sample_id": "s1",
            "memory_id": "m2",
            "speaker": "Bob",
            "session": "s1",
            "turn_id": 2,
            "text": "Bob repaired a bicycle.",
            "metadata": {},
        },
        {
            "sample_id": "s1",
            "memory_id": "m3",
            "speaker": "Alice",
            "session": "s2",
            "turn_id": 1,
            "text": "Alice planned a train trip.",
            "metadata": {},
        },
    ]
    questions = [
        {
            "sample_id": "s1",
            "question_id": "q1",
            "question": "What did Alice buy at the market?",
            "answer": "green tea",
            "category": "single_hop",
            "evidence_memory_ids": ["m1"],
            "metadata": {},
        }
    ]
    write_jsonl(memories_path, memories)
    write_jsonl(questions_path, questions)

    args = argparse.Namespace(
        memories=str(memories_path),
        questions=str(questions_path),
        out=str(out_path),
        embedding_provider="hash",
        embedding_model="hash",
        embedding_dim=64,
        embedding_cache=str(cache_path),
        vector_backend="local",
        host="",
        port=50051,
        collection_prefix="test_memory",
        use_tls=False,
        sochdb_search_mode="single",
        sochdb_ef=0,
        k=2,
        candidate_k=2,
        bm25_weight=1.5,
        vector_weight=0.75,
        rrf_k=60,
        memory_render_mode="metadata",
        query_mode="single",
        reranker_provider="none",
        limit_samples=1,
        limit_questions=5,
        run_id="test",
    )
    run_retrieval(args, "test")

    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["retrieved_memory_ids"]
    assert len(rows[0]["retrieved_memory_ids"]) == 2
    assert "m1" in rows[0]["retrieved_memory_ids"]


if __name__ == "__main__":
    tests = [
        test_metadata_rendering_with_missing_fields,
        test_longmemeval_normalization_and_span_mapping,
        test_scoring_and_buckets_and_length_diagnostics,
        test_hash_local_retrieval_smoke,
    ]
    for test in tests:
        test()
        print(f"PASSED: {test.__name__}")
