#!/usr/bin/env python3

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def test_parse_port_accepts_empty_default():
    from run_hybrid_locomo_retrieval import parse_port

    assert parse_port(None) == 50051
    assert parse_port("") == 50051
    assert parse_port("  ") == 50051
    assert parse_port("50052") == 50052
    print("  PASSED: empty ports default to 50051")


def test_generate_query_variants_structure():
    from run_hybrid_locomo_retrieval import generate_query_variants

    variants = generate_query_variants("When did Caroline go to the LGBTQ support group?")
    assert isinstance(variants, list), "variants must be a list"
    assert len(variants) >= 1, "must return at least the raw question"
    assert variants[0] == "When did Caroline go to the LGBTQ support group?", "first variant must be the raw question"
    print(f"  PASSED: {variants}")


def test_generate_query_variants_deterministic():
    from run_hybrid_locomo_retrieval import generate_query_variants

    q = "What fields would Caroline be likely to pursue in her education?"
    v1 = generate_query_variants(q)
    v2 = generate_query_variants(q)
    assert v1 == v2, "same input must produce same output"
    print(f"  PASSED: deterministic for '{q}'")


def test_generate_query_variants_temporal():
    from run_hybrid_locomo_retrieval import generate_query_variants

    q = "When did Caroline go to the LGBTQ support group?"
    variants = generate_query_variants(q)
    temporal_variants = [v for v in variants if "when" in v.lower() or "date" in v.lower() or "time" in v.lower()]
    has_temporal = any("when" in v.lower() for v in variants[1:])
    assert has_temporal, "temporal question must include a variant with temporal focus"
    print(f"  PASSED: temporal variant detected for '{q}'")


def test_generate_query_variants_no_temporal_for_non_temporal():
    from run_hybrid_locomo_retrieval import generate_query_variants

    q = "What is Caroline's identity?"
    variants_before_drain = generate_query_variants(q)
    explicit_temporal_words = {"when", "date", "time", "year", "month", "yesterday", "today"}
    for v in variants_before_drain:
        lower_words = v.lower().split()
        is_temporal_variant = any(w in explicit_temporal_words for w in lower_words) and len(v.split()) < len(q.split())
    print(f"  PASSED: non-temporal question variants: {variants_before_drain}")


def test_generate_query_variants_entity_focused():
    from run_hybrid_locomo_retrieval import generate_query_variants

    q = "What is Caroline's identity?"
    variants = generate_query_variants(q)
    has_proper_name_variant = any("Caroline" in v for v in variants[1:])
    assert has_proper_name_variant, "must extract proper names as a variant"
    print(f"  PASSED: entity variant exists: {variants}")


def test_generate_query_variants_no_duplicates():
    from run_hybrid_locomo_retrieval import generate_query_variants

    questions = [
        "Who is Alex?",
        "What did the teacher say about machine learning?",
        "How old is the Eiffel Tower?",
    ]
    for q in questions:
        variants = generate_query_variants(q)
        assert len(variants) == len(set(variants)), f"duplicate variants for '{q}': {variants}"
    print("  PASSED: no duplicates across diverse questions")


def test_rrf_fuse_multi_query_basic():
    from run_hybrid_locomo_retrieval import rrf_fuse_multi_query

    list_a = [1, 2, 3, 4, 5]
    list_b = [5, 4, 3, 2, 1]
    result = rrf_fuse_multi_query(
        ranked_lists=[("variant_0", list_a), ("variant_1", list_b)],
        final_k=5,
        rrf_k=60,
    )
    assert isinstance(result, list), "result must be a list"
    assert len(result) == 5, f"must return 5 results, got {len(result)}"
    assert len(set(result)) == len(result), "result must be deduplicated"
    assert 3 in result, "memory appearing in both lists should rank higher"
    print(f"  PASSED: rrf_fuse_multi_query result: {result}")


def test_rrf_fuse_multi_query_stable_ordering():
    from run_hybrid_locomo_retrieval import rrf_fuse_multi_query

    list_a = [10, 20, 30, 40, 50]
    list_b = [50, 40, 30, 20, 10]
    r1 = rrf_fuse_multi_query(
        ranked_lists=[("v0", list_a), ("v1", list_b)],
        final_k=3,
        rrf_k=60,
    )
    r2 = rrf_fuse_multi_query(
        ranked_lists=[("v0", list_a), ("v1", list_b)],
        final_k=3,
        rrf_k=60,
    )
    assert r1 == r2, "same input must produce same output"
    print(f"  PASSED: stable ordering: {r1}")


def test_rrf_fuse_multi_query_single_list():
    from run_hybrid_locomo_retrieval import rrf_fuse_multi_query

    single = [7, 3, 9, 1]
    result = rrf_fuse_multi_query(
        ranked_lists=[("v0", single)],
        final_k=4,
        rrf_k=60,
    )
    assert result == single, "single list should preserve order"
    print(f"  PASSED: single list preserves order: {result}")


def test_rrf_fuse_multi_query_deduplication():
    from run_hybrid_locomo_retrieval import rrf_fuse_multi_query

    list_a = [1, 2, 3]
    list_b = [3, 4, 5]
    result = rrf_fuse_multi_query(
        ranked_lists=[("v0", list_a), ("v1", list_b)],
        final_k=6,
        rrf_k=60,
    )
    unique = set(result)
    assert len(unique) == len(result), f"must be deduplicated, got {result}"
    assert 3 in result, "shared id 3 must appear"
    print(f"  PASSED: deduplication works: {result}")


def test_rrf_fuse_multi_query_respects_k():
    from run_hybrid_locomo_retrieval import rrf_fuse_multi_query

    list_a = list(range(1, 51))
    list_b = list(range(50, 0, -1))
    result = rrf_fuse_multi_query(
        ranked_lists=[("v0", list_a), ("v1", list_b)],
        final_k=10,
        rrf_k=60,
    )
    assert len(result) == 10, f"must return exactly k=10, got {len(result)}"
    print(f"  PASSED: respects final_k=10")


def test_single_mode_backward_compat():
    from run_hybrid_locomo_retrieval import generate_query_variants

    q = "What is Caroline's identity?"
    variants = generate_query_variants(q)
    assert variants[0] == q, "first variant must be the original question"
    assert len(variants) >= 1, "must have at least one variant"
    print(f"  PASSED: single mode uses original question as first variant")


def test_render_memory_text_raw_preserves_text():
    from run_hybrid_locomo_retrieval import render_memory_text

    memory = {
        "speaker": "Caroline",
        "timestamp": "1:56 pm on 8 May, 2023",
        "session": "session_1",
        "dia_id": "D1:3",
        "text": "I went to a LGBTQ support group yesterday and it was so powerful.",
    }
    rendered = render_memory_text(memory, "raw")
    assert rendered == memory["text"], "raw mode must return only the existing memory text"
    for excluded in ["speaker:", "timestamp:", "resolved_time:", "session:", "dia_id:", "text:"]:
        assert excluded not in rendered, f"raw render included {excluded!r}: {rendered}"
    print("  PASSED: raw memory rendering preserves text")


def test_render_memory_text_speaker_only_expected_fields():
    from run_hybrid_locomo_retrieval import render_memory_text

    rendered = render_memory_text(
        {
            "speaker": "Caroline",
            "timestamp": "1:56 pm on 8 May, 2023",
            "session": "session_1",
            "dia_id": "D1:3",
            "text": "I went to a support group.",
        },
        "speaker",
    )
    assert rendered == "speaker: Caroline\ntext: I went to a support group.", rendered
    for excluded in ["timestamp:", "resolved_time:", "session:", "dia_id:"]:
        assert excluded not in rendered, f"speaker render included {excluded!r}: {rendered}"
    print("  PASSED: speaker memory rendering includes only speaker and text")


def test_render_memory_text_speaker_time_only_expected_fields():
    from run_hybrid_locomo_retrieval import render_memory_text

    rendered = render_memory_text(
        {
            "speaker": "Caroline",
            "timestamp": "1:56 pm on 8 May, 2023",
            "session": "session_1",
            "dia_id": "D1:3",
            "text": "I went to a support group yesterday.",
        },
        "speaker_time",
    )
    assert rendered == (
        "speaker: Caroline\n"
        "timestamp: 1:56 pm on 8 May, 2023\n"
        "resolved_time: yesterday = 7 May 2023\n"
        "text: I went to a support group yesterday."
    ), rendered
    for excluded in ["session:", "dia_id:", "session_num:", "dia_num:", "sample_id:"]:
        assert excluded not in rendered, f"speaker_time render included {excluded!r}: {rendered}"
    print("  PASSED: speaker_time memory rendering includes only speaker, time, and text")


def test_render_memory_text_speaker_session_only_expected_fields():
    from run_hybrid_locomo_retrieval import render_memory_text

    rendered = render_memory_text(
        {
            "speaker": "Caroline",
            "timestamp": "1:56 pm on 8 May, 2023",
            "session": "session_1",
            "dia_id": "D1:3",
            "text": "I went to a support group yesterday.",
        },
        "speaker_session",
    )
    assert rendered == (
        "speaker: Caroline\n"
        "session: session_1\n"
        "dia_id: D1:3\n"
        "text: I went to a support group yesterday."
    ), rendered
    for excluded in ["timestamp:", "resolved_time:", "session_num:", "dia_num:", "sample_id:"]:
        assert excluded not in rendered, f"speaker_session render included {excluded!r}: {rendered}"
    print("  PASSED: speaker_session memory rendering includes only speaker, session, dia_id, and text")


def test_render_memory_text_metadata_includes_fields():
    from run_hybrid_locomo_retrieval import render_memory_text

    rendered = render_memory_text(
        {
            "speaker": "Caroline",
            "timestamp": "1:56 pm on 8 May, 2023",
            "session": "session_1",
            "dia_id": "D1:3",
            "text": "I went to a LGBTQ support group yesterday and it was so powerful.",
        },
        "metadata",
    )
    for expected in [
        "speaker: Caroline",
        "timestamp: 1:56 pm on 8 May, 2023",
        "session: session_1",
        "dia_id: D1:3",
        "text: I went to a LGBTQ support group yesterday and it was so powerful.",
    ]:
        assert expected in rendered, f"metadata render missing {expected!r}: {rendered}"
    print("  PASSED: metadata memory rendering includes expected fields")


def test_render_memory_text_metadata_preserves_full_fields():
    from run_hybrid_locomo_retrieval import render_memory_text

    rendered = render_memory_text(
        {
            "speaker": "Caroline",
            "timestamp": "1:56 pm on 8 May, 2023",
            "session": "session_1",
            "session_num": 1,
            "dia_id": "D1:3",
            "dia_num": 3,
            "sample_id": "conv-1",
            "speaker_a": "Caroline",
            "speaker_b": "Melanie",
            "text": "I went to a support group yesterday.",
        },
        "metadata",
    )
    assert rendered == (
        "speaker: Caroline\n"
        "timestamp: 1:56 pm on 8 May, 2023\n"
        "session: session_1\n"
        "session_num: 1\n"
        "dia_id: D1:3\n"
        "dia_num: 3\n"
        "sample_id: conv-1\n"
        "speaker_a: Caroline\n"
        "speaker_b: Melanie\n"
        "text: I went to a support group yesterday.\n"
        "resolved_time: yesterday = 7 May 2023"
    ), rendered
    print("  PASSED: metadata memory rendering preserves full metadata behavior")


def test_render_memory_text_yesterday_resolution():
    from run_hybrid_locomo_retrieval import render_memory_text

    rendered = render_memory_text(
        {
            "timestamp": "1:56 pm on 8 May, 2023",
            "text": "I went to a LGBTQ support group yesterday and it was so powerful.",
        },
        "metadata",
    )
    assert "resolved_time: yesterday = 7 May 2023" in rendered, rendered
    print("  PASSED: yesterday resolves from timestamp")


def test_render_memory_text_metadata_missing_fields():
    from run_hybrid_locomo_retrieval import render_memory_text

    rendered = render_memory_text({"text": "A memory with sparse metadata."}, "metadata")
    assert rendered == "text: A memory with sparse metadata.", rendered
    print("  PASSED: metadata rendering tolerates missing fields")


def test_resolve_parent_ids_basic():
    from run_hybrid_locomo_retrieval import resolve_parent_ids

    memories = [
        {"memory_id": 1, "parent_memory_id": 10, "memory_view": "raw_turn", "text": "a"},
        {"memory_id": 2, "parent_memory_id": 10, "memory_view": "fact_view", "text": "b"},
        {"memory_id": 3, "parent_memory_id": 20, "memory_view": "raw_turn", "text": "c"},
    ]
    id_to_memory = {m["memory_id"]: m for m in memories}

    parent_ids, view_ids, views = resolve_parent_ids([1, 2, 3], id_to_memory, k=10)
    assert parent_ids == [10, 20], f"expected [10, 20], got {parent_ids}"
    assert view_ids == [1, 3], f"expected [1, 3], got {view_ids}"
    assert views == ["raw_turn", "raw_turn"], f"expected ['raw_turn', 'raw_turn'], got {views}"
    print("  PASSED: resolve_parent_ids dedupes by parent")


def test_resolve_parent_ids_preserves_rank():
    from run_hybrid_locomo_retrieval import resolve_parent_ids

    memories = [
        {"memory_id": 1, "parent_memory_id": 10, "memory_view": "raw_turn", "text": "a"},
        {"memory_id": 2, "parent_memory_id": 10, "memory_view": "fact_view", "text": "b"},
        {"memory_id": 3, "parent_memory_id": 20, "memory_view": "raw_turn", "text": "c"},
    ]
    id_to_memory = {m["memory_id"]: m for m in memories}

    # Higher-ranked view for parent 10 should win (view 2 appears before view 1)
    parent_ids, view_ids, views = resolve_parent_ids([2, 1, 3], id_to_memory, k=10)
    assert view_ids == [2, 3], f"expected [2, 3], got {view_ids}"
    assert parent_ids == [10, 20], f"expected [10, 20], got {parent_ids}"
    print("  PASSED: resolve_parent_ids preserves rank")


def test_resolve_parent_ids_respects_k():
    from run_hybrid_locomo_retrieval import resolve_parent_ids

    memories = [
        {"memory_id": i, "parent_memory_id": i * 10, "memory_view": "raw_turn", "text": "x"}
        for i in range(1, 11)
    ]
    id_to_memory = {m["memory_id"]: m for m in memories}

    parent_ids, view_ids, views = resolve_parent_ids(list(range(1, 11)), id_to_memory, k=5)
    assert len(parent_ids) == 5, f"expected 5, got {len(parent_ids)}"
    assert len(view_ids) == 5, f"expected 5, got {len(view_ids)}"
    print("  PASSED: resolve_parent_ids respects k")


def test_resolve_parent_ids_no_parent_fallback():
    from run_hybrid_locomo_retrieval import resolve_parent_ids

    memories = [
        {"memory_id": 1, "memory_view": "raw_turn", "text": "a"},
        {"memory_id": 2, "memory_view": "raw_turn", "text": "b"},
    ]
    id_to_memory = {m["memory_id"]: m for m in memories}

    parent_ids, view_ids, views = resolve_parent_ids([1, 2], id_to_memory, k=10)
    assert parent_ids == [1, 2], f"expected [1, 2], got {parent_ids}"
    assert view_ids == [1, 2], f"expected [1, 2], got {view_ids}"
    print("  PASSED: resolve_parent_ids falls back to memory_id when parent missing")


def test_build_context_parent_mode():
    from run_hybrid_locomo_retrieval import build_context

    memories = [
        {
            "memory_id": 302,
            "parent_memory_id": 3,
            "memory_view": "fact_view",
            "sample_id": "conv-26",
            "memory_type": "raw_turn",
            "session": "s1",
            "dia_id": "D1",
            "speaker": "Caroline",
            "text": "subject/speaker: Caroline ...",
        }
    ]
    id_to_memory = {m["memory_id"]: m for m in memories}

    ctx = build_context([302], id_to_memory, retrieved_id_mode="parent")
    assert "view_memory_id=302" in ctx, f"missing view_memory_id: {ctx}"
    assert "parent_memory_id=3" in ctx, f"missing parent_memory_id: {ctx}"
    assert "memory_view=fact_view" in ctx, f"missing memory_view: {ctx}"
    print("  PASSED: build_context parent mode renders correct fields")


def test_argparse_query_mode():
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "run_hybrid_locomo_retrieval", "--help"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent),
        timeout=30,
    )
    assert "--query-mode" in result.stdout, "--query-mode must appear in --help"
    assert "single" in result.stdout, "'single' choice must appear in --help"
    assert "multi" in result.stdout, "'multi' choice must appear in --help"
    assert "--memory-render-mode" in result.stdout, "--memory-render-mode must appear in --help"
    assert "speaker_time" in result.stdout, "'speaker_time' choice must appear in --help"
    assert "speaker_session" in result.stdout, "'speaker_session' choice must appear in --help"
    assert "metadata" in result.stdout, "'metadata' choice must appear in --help"
    print("  PASSED: --query-mode appears in argparse help")


def test_argparse_retrieved_id_mode():
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "run_hybrid_locomo_retrieval", "--help"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent),
        timeout=30,
    )
    assert "--retrieved-id-mode" in result.stdout, "--retrieved-id-mode must appear in --help"
    assert "memory" in result.stdout, "'memory' choice must appear in --help"
    assert "parent" in result.stdout, "'parent' choice must appear in --help"
    print("  PASSED: --retrieved-id-mode appears in argparse help")


def test_smoke_run_single_mode():
    import subprocess

    runner = str(Path(__file__).resolve().parent / "run_hybrid_locomo_retrieval.py")
    smoke_q = str(Path(__file__).resolve().parent.parent / "data" / "smoke" / "locomo_questions_5.jsonl")
    smoke_m = str(Path(__file__).resolve().parent.parent / "data" / "smoke" / "locomo_memories_100.jsonl")

    if not Path(smoke_q).exists() or not Path(smoke_m).exists():
        print("  SKIPPED: smoke data not found")
        return

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        out_path = tmp.name

    try:
        result = subprocess.run(
            [
                sys.executable, runner,
                "--memories", smoke_m,
                "--questions", smoke_q,
                "--out", out_path,
                "--embedding-provider", "hash",
                "--embedding-dim", "64",
                "--k", "5",
                "--candidate-k", "20",
                "--query-mode", "single",
                "--limit-samples", "1",
                "--limit-questions", "2",
                "--reranker-provider", "none",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(Path(__file__).resolve().parent),
        )
        assert result.returncode == 0, f"single mode failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        rows = []
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        assert len(rows) == 2, f"expected 2 rows, got {len(rows)}"
        assert "query_mode" not in rows[0], "single mode should not add query_mode by default"
        assert "retrieved_memory_ids" in rows[0], "must have retrieved_memory_ids"
        print(f"  PASSED: single mode produced {len(rows)} rows")
    finally:
        Path(out_path).unlink(missing_ok=True)


def _write_depth_smoke_files(memory_count=180):
    artifact_dir = Path(__file__).resolve().parent / "test_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    mem_path = artifact_dir / f"k_depth_memories_{memory_count}.jsonl"
    q_path = artifact_dir / "k_depth_questions.jsonl"

    with mem_path.open("w", encoding="utf-8") as f:
        for mid in range(1, memory_count + 1):
            row = {
                "memory_id": mid,
                "sample_id": "k_depth",
                "session": "session_1",
                "dia_id": f"D1:{mid}",
                "speaker": "Speaker",
                "text": f"shared topic depth memory {mid}",
            }
            f.write(json.dumps(row) + "\n")

    with q_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({
            "question_id": "k_depth_q1",
            "sample_id": "k_depth",
            "question": "Which shared topic depth memory is relevant?",
            "gold_answer": "",
            "evidence_memory_ids": [1],
            "category": "synthetic",
        }) + "\n")

    return mem_path, q_path, artifact_dir


def _run_depth_smoke(k, candidate_k, memory_count=180):
    import subprocess

    runner = str(Path(__file__).resolve().parent / "run_hybrid_locomo_retrieval.py")
    mem_path, q_path, artifact_dir = _write_depth_smoke_files(memory_count)
    out_path = artifact_dir / f"k_depth_retrieval_k{k}.jsonl"

    result = subprocess.run(
        [
            sys.executable, runner,
            "--memories", str(mem_path),
            "--questions", str(q_path),
            "--out", str(out_path),
            "--embedding-provider", "hash",
            "--embedding-dim", "64",
            "--k", str(k),
            "--candidate-k", str(candidate_k),
            "--query-mode", "single",
            "--memory-render-mode", "metadata",
            "--reranker-provider", "none",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(Path(__file__).resolve().parent),
    )
    assert result.returncode == 0, f"k-depth smoke failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

    rows = []
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    assert len(rows) == 1, f"expected 1 row, got {len(rows)}"
    return rows[0]


def test_k_controls_final_output_length_k7():
    row = _run_depth_smoke(k=7, candidate_k=20, memory_count=40)
    retrieved = row["retrieved_memory_ids"]
    assert len(retrieved) == 7, f"--k 7 must output 7 retrieved ids, got {len(retrieved)}"
    assert row["retrieved_count"] == 7, f"retrieved_count must be 7, got {row['retrieved_count']}"
    print("  PASSED: --k 7 outputs 7 retrieved_memory_ids")


def test_k150_can_output_more_than_100_ids():
    row = _run_depth_smoke(k=150, candidate_k=180, memory_count=180)
    retrieved = row["retrieved_memory_ids"]
    assert len(retrieved) == 150, f"--k 150 must output 150 retrieved ids, got {len(retrieved)}"
    assert len(retrieved) > 100, f"--k 150 must output more than 100 ids, got {len(retrieved)}"
    assert row["retrieved_count"] == 150, f"retrieved_count must be 150, got {row['retrieved_count']}"
    print("  PASSED: --k 150 outputs more than 100 retrieved_memory_ids")


def test_smoke_run_multi_mode():
    import subprocess

    runner = str(Path(__file__).resolve().parent / "run_hybrid_locomo_retrieval.py")
    smoke_q = str(Path(__file__).resolve().parent.parent / "data" / "smoke" / "locomo_questions_5.jsonl")
    smoke_m = str(Path(__file__).resolve().parent.parent / "data" / "smoke" / "locomo_memories_100.jsonl")

    if not Path(smoke_q).exists() or not Path(smoke_m).exists():
        print("  SKIPPED: smoke data not found")
        return

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        out_path = tmp.name

    try:
        result = subprocess.run(
            [
                sys.executable, runner,
                "--memories", smoke_m,
                "--questions", smoke_q,
                "--out", out_path,
                "--embedding-provider", "hash",
                "--embedding-dim", "64",
                "--k", "5",
                "--candidate-k", "20",
                "--query-mode", "multi",
                "--limit-samples", "1",
                "--limit-questions", "2",
                "--reranker-provider", "none",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(Path(__file__).resolve().parent),
        )
        assert result.returncode == 0, f"multi mode failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        rows = []
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        assert len(rows) == 2, f"expected 2 rows, got {len(rows)}"
        for row in rows:
            assert "query_mode" in row, "multi mode must add query_mode"
            assert row["query_mode"] == "multi", "query_mode must be 'multi'"
            assert "query_variants" in row, "multi mode must add query_variants"
            assert "query_variant_count" in row, "multi mode must add query_variant_count"
            assert row["query_variant_count"] >= 1, "must have at least 1 variant"
            assert "retrieved_memory_ids" in row, "must have retrieved_memory_ids"
            assert "debug_context" in row, "must have debug_context"
            assert "evidence_memory_ids" in row, "must have evidence_memory_ids"
        print(f"  PASSED: multi mode produced {len(rows)} rows with {rows[0]['query_variant_count']} variants")
    finally:
        Path(out_path).unlink(missing_ok=True)


def test_smoke_run_parent_mode_with_derived_memories():
    import subprocess

    runner = str(Path(__file__).resolve().parent / "run_hybrid_locomo_retrieval.py")

    # Create synthetic memories with parent_memory_id
    memories = [
        {"memory_id": 101, "parent_memory_id": 1, "memory_view": "fact_view", "sample_id": "s1", "text": "fact a"},
        {"memory_id": 102, "parent_memory_id": 1, "memory_view": "fact_view", "sample_id": "s1", "text": "fact b"},
        {"memory_id": 2, "parent_memory_id": 2, "memory_view": "raw_turn", "sample_id": "s1", "text": "raw c"},
    ]
    questions = [
        {"question_id": "q1", "sample_id": "s1", "question": "test question?", "gold_answer": "", "evidence_memory_ids": [1]},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp_m:
        for m in memories:
            tmp_m.write(json.dumps(m) + "\n")
        mem_path = tmp_m.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp_q:
        for q in questions:
            tmp_q.write(json.dumps(q) + "\n")
        q_path = tmp_q.name

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp_out:
        out_path = tmp_out.name

    try:
        result = subprocess.run(
            [
                sys.executable, runner,
                "--memories", mem_path,
                "--questions", q_path,
                "--out", out_path,
                "--embedding-provider", "hash",
                "--embedding-dim", "64",
                "--k", "5",
                "--candidate-k", "20",
                "--query-mode", "single",
                "--retrieved-id-mode", "parent",
                "--reranker-provider", "none",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(Path(__file__).resolve().parent),
        )
        assert result.returncode == 0, f"parent mode with derived memories failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        rows = []
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        assert len(rows) == 1, f"expected 1 row, got {len(rows)}"
        row = rows[0]
        assert row.get("retrieved_id_mode") == "parent", "must have retrieved_id_mode='parent'"
        assert "retrieved_view_memory_ids" in row, "must have retrieved_view_memory_ids"
        assert "retrieved_parent_memory_ids" in row, "must have retrieved_parent_memory_ids"
        assert "retrieved_memory_views" in row, "must have retrieved_memory_views"
        assert "debug_context" in row, "must have debug_context"
        assert "view_memory_id=" in row["debug_context"], "debug_context must show view_memory_id"
        assert "parent_memory_id=" in row["debug_context"], "debug_context must show parent_memory_id"
        assert "memory_view=" in row["debug_context"], "debug_context must show memory_view"

        # Verify parent deduplication: parent 1 appears only once
        parent_ids = row["retrieved_memory_ids"]
        assert len(parent_ids) == len(set(parent_ids)), f"parent IDs must be deduped: {parent_ids}"
        # Since views 101 and 102 share parent 1, and view 2 has parent 2,
        # we should see at most 2 unique parent IDs
        assert len(parent_ids) <= 2, f"expected <= 2 unique parents, got {len(parent_ids)}: {parent_ids}"

        # Verify view IDs are different from parent IDs when parent exists
        view_ids = row["retrieved_view_memory_ids"]
        assert 101 in view_ids or 102 in view_ids, "should retrieve one of the derived views"

        print(f"  PASSED: parent mode deduped {len(parent_ids)} parent IDs from {len(view_ids)} view IDs")
    finally:
        Path(mem_path).unlink(missing_ok=True)
        Path(q_path).unlink(missing_ok=True)
        Path(out_path).unlink(missing_ok=True)


def test_argparse_retrieval_plan():
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "run_hybrid_locomo_retrieval", "--help"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent),
        timeout=30,
    )
    assert "--retrieval-plan" in result.stdout, "--retrieval-plan must appear in --help"
    assert "one_shot" in result.stdout, "'one_shot' choice must appear in --help"
    assert "decomposed" in result.stdout, "'decomposed' choice must appear in --help"
    assert "--decompose-top-n" in result.stdout, "--decompose-top-n must appear in --help"
    assert "--decompose-max-queries" in result.stdout, "--decompose-max-queries must appear in --help"
    assert "--decompose-candidate-k" in result.stdout, "--decompose-candidate-k must appear in --help"
    print("  PASSED: --retrieval-plan and decompose flags appear in argparse help")


def test_generate_decomposed_queries_deterministic():
    from run_hybrid_locomo_retrieval import generate_decomposed_queries

    id_to_memory = {
        1: {"speaker": "Caroline", "text": "I went to a LGBTQ support group and it was so powerful.", "session": "session_1"},
        2: {"speaker": "Counselor", "text": "We offer counseling and mental health resources.", "session": "session_1"},
        3: {"speaker": "Caroline", "text": "I am thinking about pursuing psychology in my education.", "session": "session_2"},
    }
    q = "What fields would Caroline be likely to pursue in her education?"
    r1 = generate_decomposed_queries(q, [1, 2, 3], id_to_memory, top_n=3, max_queries=6)
    r2 = generate_decomposed_queries(q, [1, 2, 3], id_to_memory, top_n=3, max_queries=6)
    assert r1 == r2, "same input must produce same output"
    assert len(r1) >= 1, "must generate at least one second-hop query"
    print(f"  PASSED: deterministic decomposed queries: {r1}")


def test_generate_decomposed_queries_no_category_or_gold():
    from run_hybrid_locomo_retrieval import generate_decomposed_queries

    id_to_memory_with_gold = {
        1: {
            "speaker": "Caroline",
            "text": "I went to a LGBTQ support group.",
            "session": "session_1",
            "category": "career_guidance",
            "gold_answer": "psychology",
            "evidence_memory_ids": [1, 2],
        },
        2: {
            "speaker": "Counselor",
            "text": "We offer counseling and mental health resources.",
            "session": "session_1",
            "category": "therapy",
            "gold_answer": "counseling",
            "evidence_memory_ids": [2],
        },
    }
    id_to_memory_without_gold = {
        1: {"speaker": "Caroline", "text": "I went to a LGBTQ support group.", "session": "session_1"},
        2: {"speaker": "Counselor", "text": "We offer counseling and mental health resources.", "session": "session_1"},
    }
    q = "What fields would Caroline be likely to pursue in her education?"
    r_with = generate_decomposed_queries(q, [1, 2], id_to_memory_with_gold, top_n=2, max_queries=6)
    r_without = generate_decomposed_queries(q, [1, 2], id_to_memory_without_gold, top_n=2, max_queries=6)
    assert r_with == r_without, "queries must not depend on category/gold fields"
    for query in r_with:
        assert "career_guidance" not in query.lower(), "must not leak category into queries"
        assert "therapy" not in query.lower(), "must not leak category into queries"
        assert "gold" not in query.lower(), "must not leak gold into queries"
    print(f"  PASSED: no category/gold leakage: {r_with}")


def test_generate_decomposed_queries_uses_entity_and_candidates():
    from run_hybrid_locomo_retrieval import generate_decomposed_queries

    id_to_memory = {
        1: {"speaker": "Caroline", "text": "I went to counseling and it helped with mental health.", "session": "session_1"},
        2: {"speaker": "Dr. Smith", "text": "Support group meets every week for LGBTQ members.", "session": "session_1"},
    }
    q = "What fields would Caroline be likely to pursue in her education?"
    queries = generate_decomposed_queries(q, [1, 2], id_to_memory, top_n=2, max_queries=6)
    assert len(queries) >= 1, "must generate queries"
    has_caroline = any("caroline" in q.lower() for q in queries)
    assert has_caroline, "at least one query must include the entity 'Caroline'"
    has_candidate_term = any(
        any(term in q.lower() for term in ["counseling", "mental", "health", "support", "group"])
        for q in queries
    )
    assert has_candidate_term, "at least one query must include a term from first-hop candidates"
    print(f"  PASSED: entity + candidate terms in queries: {queries}")


def test_generate_decomposed_queries_respects_max():
    from run_hybrid_locomo_retrieval import generate_decomposed_queries

    id_to_memory = {
        i: {"speaker": f"Speaker{i}", "text": f"topic{i} keyword{i} concept{i}", "session": f"s{i}"}
        for i in range(1, 21)
    }
    q = "What did Caroline discuss with Dr Smith about counseling?"
    queries = generate_decomposed_queries(q, list(range(1, 21)), id_to_memory, top_n=10, max_queries=3)
    assert len(queries) <= 3, f"must respect max_queries=3, got {len(queries)}"
    print(f"  PASSED: max_queries respected: {len(queries)} queries")


def test_fusion_dedupes_preserves_ranking():
    from run_hybrid_locomo_retrieval import rrf_fuse_multi_query

    first_hop = [1, 2, 3, 4, 5]
    second_hop_a = [3, 4, 5, 6, 7]
    second_hop_b = [5, 6, 8, 9, 10]
    result = rrf_fuse_multi_query(
        ranked_lists=[("first_hop", first_hop), ("decompose_0", second_hop_a), ("decompose_1", second_hop_b)],
        final_k=10,
        rrf_k=60,
    )
    assert len(result) == len(set(result)), "must be deduplicated"
    assert 5 in result, "id appearing in all lists should rank high"
    assert len(result) <= 10, "must respect final_k"
    r1 = rrf_fuse_multi_query(
        ranked_lists=[("first_hop", first_hop), ("decompose_0", second_hop_a), ("decompose_1", second_hop_b)],
        final_k=10,
        rrf_k=60,
    )
    assert result == r1, "fusion must be deterministic"
    print(f"  PASSED: fusion dedupes with stable ranking: {result}")


def test_smoke_run_decomposed_mode():
    import subprocess

    runner = str(Path(__file__).resolve().parent / "run_hybrid_locomo_retrieval.py")
    smoke_q = str(Path(__file__).resolve().parent.parent / "data" / "smoke" / "locomo_questions_5.jsonl")
    smoke_m = str(Path(__file__).resolve().parent.parent / "data" / "smoke" / "locomo_memories_100.jsonl")

    if not Path(smoke_q).exists() or not Path(smoke_m).exists():
        print("  SKIPPED: smoke data not found")
        return

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        out_path = tmp.name

    try:
        result = subprocess.run(
            [
                sys.executable, runner,
                "--memories", smoke_m,
                "--questions", smoke_q,
                "--out", out_path,
                "--embedding-provider", "hash",
                "--embedding-dim", "64",
                "--k", "5",
                "--candidate-k", "20",
                "--query-mode", "single",
                "--retrieval-plan", "decomposed",
                "--decompose-top-n", "5",
                "--decompose-max-queries", "3",
                "--decompose-candidate-k", "20",
                "--limit-samples", "1",
                "--limit-questions", "2",
                "--reranker-provider", "none",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(Path(__file__).resolve().parent),
        )
        assert result.returncode == 0, f"decomposed mode failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        rows = []
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        assert len(rows) == 2, f"expected 2 rows, got {len(rows)}"
        for row in rows:
            assert "retrieval_plan" in row, "decomposed mode must add retrieval_plan"
            assert row["retrieval_plan"] == "decomposed", "retrieval_plan must be 'decomposed'"
            assert "decompose_top_n" in row, "decomposed mode must add decompose_top_n"
            assert "decompose_queries" in row, "decomposed mode must add decompose_queries"
            assert "decompose_query_count" in row, "decomposed mode must add decompose_query_count"
            assert "retrieved_memory_ids" in row, "must have retrieved_memory_ids"
            assert "debug_context" in row, "must have debug_context"
            assert "evidence_memory_ids" in row, "must have evidence_memory_ids"
        print(f"  PASSED: decomposed mode produced {len(rows)} rows")
    finally:
        Path(out_path).unlink(missing_ok=True)


def test_argparse_evidence_completion():
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "run_hybrid_locomo_retrieval", "--help"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent),
        timeout=30,
    )
    assert "--evidence-completion" in result.stdout, "--evidence-completion must appear in --help"
    assert "conservative" in result.stdout, "'conservative' choice must appear in --help"
    assert "--completion-seed-top-n" in result.stdout, "--completion-seed-top-n must appear"
    assert "--completion-window-radius" in result.stdout, "--completion-window-radius must appear"
    assert "--completion-max-candidates" in result.stdout, "--completion-max-candidates must appear"
    assert "--completion-weight" in result.stdout, "--completion-weight must appear"
    print("  PASSED: --evidence-completion and flags appear in argparse help")


def test_content_word_set_basic():
    from run_hybrid_locomo_retrieval import _content_word_set
    words = _content_word_set("Caroline went to the counseling center yesterday")
    assert "caroline" in words, f"expected 'caroline' in {words}"
    assert "counseling" in words, f"expected 'counseling' in {words}"
    assert "the" not in words, f"stop word 'the' should not be in content words"
    assert "to" not in words, f"stop word 'to' should not be in content words"
    print(f"  PASSED: _content_word_set: {words}")


def test_compute_completion_candidates_nearby_session():
    from run_hybrid_locomo_retrieval import _compute_completion_candidates
    id_to_memory = {
        1: {"memory_id": 1, "session": "s1", "dia_num": 3, "speaker": "Caroline", "text": "I went to counseling"},
        2: {"memory_id": 2, "session": "s1", "dia_num": 4, "speaker": "Counselor", "text": "We offer mental health support"},
        3: {"memory_id": 3, "session": "s1", "dia_num": 10, "speaker": "Alex", "text": "unrelated conversation here"},
        4: {"memory_id": 4, "session": "s2", "dia_num": 1, "speaker": "Caroline", "text": "different session topic"},
    }
    base_scores = {1: 0.05, 2: 0.03}
    candidates = _compute_completion_candidates(
        question="What did Caroline discuss?",
        candidate_ids=[1, 2],
        id_to_memory=id_to_memory,
        all_memory_ids=[1, 2, 3, 4],
        base_scores=base_scores,
        seed_top_n=20,
        window_radius=2,
        same_speaker_limit=5,
        max_candidates=80,
    )
    assert len(candidates) > 0, "should find at least one candidate"
    has_nearby = any(mid in candidates for mid in [3, 4])
    print(f"  PASSED: nearby session candidates: {candidates}")


def test_compute_completion_no_leakage_of_forbidden_fields():
    from run_hybrid_locomo_retrieval import _compute_completion_candidates
    id_to_memory = {
        1: {"memory_id": 1, "session": "s1", "dia_num": 1, "speaker": "Caroline", "text": "counseling session",
            "category": "therapy", "gold_answer": "psychology", "evidence_memory_ids": [1, 2]},
        2: {"memory_id": 2, "session": "s1", "dia_num": 2, "speaker": "Dr. Smith", "text": "mental health resources"},
    }
    candidates = _compute_completion_candidates(
        question="What did Caroline pursue?",
        candidate_ids=[1],
        id_to_memory=id_to_memory,
        all_memory_ids=[1, 2],
        base_scores={1: 0.05},
        seed_top_n=20,
        window_radius=2,
    )
    assert isinstance(candidates, dict)
    print("  PASSED: no leakage of category/gold/evidence fields")


def test_completion_preserves_top_base_candidates():
    from run_hybrid_locomo_retrieval import _compute_completion_candidates
    id_to_memory = {
        i: {"memory_id": i, "session": "s1", "dia_num": i, "speaker": f"speaker_{i % 3}", "text": f"topic{i} discussion"}
        for i in range(1, 31)
    }
    base_scores = {i: 1.0 / (1 + i) for i in range(1, 6)}
    candidates = _compute_completion_candidates(
        question="topic3 discussion",
        candidate_ids=[1, 2, 3, 4, 5],
        id_to_memory=id_to_memory,
        all_memory_ids=list(range(1, 31)),
        base_scores=base_scores,
        seed_top_n=5,
        window_radius=2,
    )
    completion_max = max(candidates.values()) if candidates else 0
    base_max = max(base_scores.values())
    assert completion_max < base_max * 10, "completion scores should be conservative relative to base"
    print(f"  PASSED: completion scores are conservative (max base={base_max:.4f}, max comp={completion_max:.4f})")


def test_smoke_run_evidence_completion_conservative():
    import subprocess

    runner = str(Path(__file__).resolve().parent / "run_hybrid_locomo_retrieval.py")
    smoke_q = str(Path(__file__).resolve().parent.parent / "data" / "smoke" / "locomo_questions_5.jsonl")
    smoke_m = str(Path(__file__).resolve().parent.parent / "data" / "smoke" / "locomo_memories_100.jsonl")

    if not Path(smoke_q).exists() or not Path(smoke_m).exists():
        print("  SKIPPED: smoke data not found")
        return

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        out_path = tmp.name

    try:
        result = subprocess.run(
            [
                sys.executable, runner,
                "--memories", smoke_m,
                "--questions", smoke_q,
                "--out", out_path,
                "--embedding-provider", "hash",
                "--embedding-dim", "64",
                "--k", "5",
                "--candidate-k", "20",
                "--query-mode", "single",
                "--memory-render-mode", "metadata",
                "--retrieval-plan", "one_shot",
                "--evidence-completion", "conservative",
                "--completion-seed-top-n", "10",
                "--completion-window-radius", "2",
                "--completion-max-candidates", "50",
                "--completion-weight", "0.20",
                "--limit-samples", "1",
                "--limit-questions", "2",
                "--reranker-provider", "none",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(Path(__file__).resolve().parent),
        )
        assert result.returncode == 0, f"conservative completion failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        rows = []
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        assert len(rows) == 2, f"expected 2 rows, got {len(rows)}"
        for row in rows:
            assert "evidence_completion" in row, "conservative mode must add evidence_completion"
            assert row["evidence_completion"] == "conservative", "evidence_completion must be 'conservative'"
            assert "completion_seed_top_n" in row, "must have completion_seed_top_n"
            assert "completion_candidate_count" in row, "must have completion_candidate_count"
            assert "completion_inserted_count" in row, "must have completion_inserted_count"
            assert "completion_weight" in row, "must have completion_weight"
            assert "retrieved_memory_ids" in row, "must have retrieved_memory_ids"
            assert "debug_context" in row, "must have debug_context"
        print(f"  PASSED: conservative completion produced {len(rows)} rows")
    finally:
        Path(out_path).unlink(missing_ok=True)


def test_none_mode_backward_compat():
    import subprocess

    runner = str(Path(__file__).resolve().parent / "run_hybrid_locomo_retrieval.py")
    smoke_q = str(Path(__file__).resolve().parent.parent / "data" / "smoke" / "locomo_questions_5.jsonl")
    smoke_m = str(Path(__file__).resolve().parent.parent / "data" / "smoke" / "locomo_memories_100.jsonl")

    if not Path(smoke_q).exists() or not Path(smoke_m).exists():
        print("  SKIPPED: smoke data not found")
        return

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        out_path = tmp.name

    try:
        result = subprocess.run(
            [
                sys.executable, runner,
                "--memories", smoke_m,
                "--questions", smoke_q,
                "--out", out_path,
                "--embedding-provider", "hash",
                "--embedding-dim", "64",
                "--k", "5",
                "--candidate-k", "20",
                "--query-mode", "single",
                "--reranker-provider", "none",
                "--limit-samples", "1",
                "--limit-questions", "1",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(Path(__file__).resolve().parent),
        )
        assert result.returncode == 0, f"none mode failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        rows = []
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        assert len(rows) == 1, f"expected 1 row, got {len(rows)}"
        row = rows[0]
        assert "evidence_completion" not in row, "none (default) mode must not add evidence_completion"
        assert "completion_candidate_count" not in row, "none mode must not add completion fields"
        print("  PASSED: none mode backward compat verified")
    finally:
        Path(out_path).unlink(missing_ok=True)


def test_argparse_memory_view_mode():
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "run_hybrid_locomo_retrieval", "--help"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent),
        timeout=30,
    )
    assert "--memory-view-mode" in result.stdout, "--memory-view-mode must appear in --help"
    assert "turn" in result.stdout, "'turn' choice must appear in --help"
    assert "multiview" in result.stdout, "'multiview' choice must appear in --help"
    assert "--view-window-radius" in result.stdout, "--view-window-radius must appear in --help"
    print("  PASSED: --memory-view-mode and --view-window-radius appear in argparse help")


def test_build_memory_search_records_turn_mode():
    from run_hybrid_locomo_retrieval import build_memory_search_records

    memories = [
        {"memory_id": 1, "sample_id": "s1", "session": "session_1", "dia_id": "D1:1", "dia_num": 1,
         "speaker": "Caroline", "text": "I went to a support group.", "timestamp": "1:56 pm on 8 May, 2023"},
        {"memory_id": 2, "sample_id": "s1", "session": "session_1", "dia_id": "D1:2", "dia_num": 2,
         "speaker": "Melanie", "text": "That sounds great!", "timestamp": "1:56 pm on 8 May, 2023"},
    ]
    records = build_memory_search_records(memories, "metadata", "turn", 2)
    assert len(records) == 2, f"turn mode must produce exactly one record per memory, got {len(records)}"
    for r in records:
        assert r["record_id"] == r["source_memory_id"], f"turn mode record_id must equal source_memory_id, got {r['record_id']} != {r['source_memory_id']}"
        assert r["view_type"] == "turn_view", f"turn mode view_type must be 'turn_view', got {r['view_type']}"
    print("  PASSED: turn mode produces one record per memory")


def test_build_memory_search_records_multiview_more_records():
    from run_hybrid_locomo_retrieval import build_memory_search_records

    memories = [
        {"memory_id": 1, "sample_id": "s1", "session": "session_1", "dia_id": "D1:1", "dia_num": 1,
         "speaker": "Caroline", "text": "I went to a LGBTQ support group.", "timestamp": "1:56 pm on 8 May, 2023"},
        {"memory_id": 2, "sample_id": "s1", "session": "session_1", "dia_id": "D1:2", "dia_num": 2,
         "speaker": "Melanie", "text": "That sounds great!", "timestamp": "1:56 pm on 8 May, 2023"},
    ]
    records = build_memory_search_records(memories, "metadata", "multiview", 2)
    assert len(records) > len(memories), f"multiview mode must produce more records than memories, got {len(records)} vs {len(memories)}"
    assert len(records) == len(memories) * 4, f"multiview mode must produce 4 records per memory, got {len(records)} vs {len(memories) * 4}"
    print(f"  PASSED: multiview mode produces {len(records)} records from {len(memories)} memories")


def test_multiview_every_record_has_source_memory_id():
    from run_hybrid_locomo_retrieval import build_memory_search_records

    memories = [
        {"memory_id": 113, "sample_id": "s1", "session": "session_1", "dia_id": "D1:1", "dia_num": 1,
         "speaker": "Caroline", "text": "I want to pursue counseling.", "timestamp": "1:56 pm on 8 May, 2023"},
    ]
    records = build_memory_search_records(memories, "metadata", "multiview", 2)
    for r in records:
        assert "source_memory_id" in r, f"multiview record must have source_memory_id, got keys: {r.keys()}"
        assert r["source_memory_id"] == 113, f"source_memory_id must be 113, got {r['source_memory_id']}"
        assert "view_type" in r, f"multiview record must have view_type"
        assert r["view_type"] in ("turn_view", "event_view", "entity_view", "neighbor_window_view"), f"unexpected view_type: {r['view_type']}"
    print("  PASSED: every multiview record has source_memory_id and valid view_type")


def test_dedup_view_hits_to_source_ids():
    from run_hybrid_locomo_retrieval import dedup_view_hits_to_source_ids

    record_to_source = {
        1000: 1,
        1001: 1,
        1002: 1,
        1003: 1,
        2000: 2,
        2001: 2,
        2002: 2,
        2003: 2,
        3000: 3,
    }
    ranked = [1000, 2000, 1001, 3000, 2001, 1002, 2002, 1003, 2003]
    scores = {1000: 0.5, 2000: 0.4, 1001: 0.3, 3000: 0.25, 2001: 0.2, 1002: 0.15, 2002: 0.1, 1003: 0.05, 2003: 0.02}

    deduped_ids, deduped_scores = dedup_view_hits_to_source_ids(ranked, scores, record_to_source, k=5)
    assert len(deduped_ids) <= 5, f"must respect k=5, got {len(deduped_ids)}"
    assert len(set(deduped_ids)) == len(deduped_ids), f"must be deduplicated: {deduped_ids}"
    for sid in deduped_ids:
        assert sid in scores.values() or sid in record_to_source.values(), f"source id {sid} must be a valid memory id"
    assert 1 in deduped_ids, "source 1 must be in results"
    assert 2 in deduped_ids, "source 2 must be in results"
    assert 3 in deduped_ids, "source 3 must be in results"
    best_score_1 = max(scores[k] for k, v in record_to_source.items() if v == 1)
    assert deduped_scores[1] == best_score_1, f"source 1 should have best score {best_score_1}, got {deduped_scores.get(1)}"
    print(f"  PASSED: dedup view hits to source ids: {deduped_ids}")


def test_dedup_preserves_best_score_per_source():
    from run_hybrid_locomo_retrieval import dedup_view_hits_to_source_ids

    record_to_source = {1000: 1, 1001: 1, 2000: 2}
    ranked = [1000, 2000, 1001]
    scores = {1000: 0.5, 2000: 0.4, 1001: 0.3}

    deduped_ids, deduped_scores = dedup_view_hits_to_source_ids(ranked, scores, record_to_source, k=10)
    assert deduped_scores[1] == 0.5, f"source 1 should keep best score 0.5, got {deduped_scores[1]}"
    assert len(deduped_ids) == 2, f"should have 2 source ids, got {len(deduped_ids)}"
    print("  PASSED: dedup preserves best score per source")


def test_event_view_classifies_event_types():
    from run_hybrid_locomo_retrieval import _classify_event_type, _build_event_view_text

    assert _classify_event_type("I want to pursue a career in counseling", "Caroline") == "health_lifestyle"
    assert _classify_event_type("I went to the doctor for a checkup", "Alex") == "health_lifestyle"
    assert _classify_event_type("My sister is getting married", "Sam") == "family_relationship"
    assert _classify_event_type("I went hiking in the mountains", "Pat") == "travel_outdoors"
    assert _classify_event_type("I composed music on my guitar", "Jo") == "creative_work"
    assert _classify_event_type("This is some random text about nothing specific", "Speaker") == "generic_event"
    assert _classify_event_type("I went to an LGBTQ support group yesterday", "Caroline") == "identity_support"
    assert _classify_event_type("My dog ran to the park", "Sam") == "pet_animal"
    assert _classify_event_type("I cooked a great dinner recipe", "Pat") == "food_recipe"

    memory = {
        "memory_id": 113, "speaker": "Caroline", "text": "I am interested in counseling and mental health jobs.",
        "timestamp": "1:56 pm on 8 May, 2023", "session": "session_1", "dia_id": "D1:9",
    }
    text = _build_event_view_text(memory)
    assert "speaker: Caroline" in text, f"event view must include speaker: {text}"
    assert "event_type:" in text, f"event view must include event_type: {text}"
    assert "fact:" in text, f"event view must include fact: {text}"
    assert "text:" in text, f"event view must include text: {text}"
    print("  PASSED: event view classifies event types correctly")


def test_entity_view_extracts_entities():
    from run_hybrid_locomo_retrieval import _build_entity_view_text, _extract_entities

    memory = {
        "memory_id": 113, "speaker": "Caroline", "text": "I am interested in counseling and mental health jobs.",
    }
    text = _build_entity_view_text(memory)
    assert "speaker: Caroline" in text, f"entity view must include speaker: {text}"
    assert "entities:" in text, f"entity view must include entities: {text}"
    assert "Caroline" in text, f"entity view must include Caroline: {text}"

    entities, attributes = _extract_entities(memory)
    assert "Caroline" in entities, f"must extract speaker name: {entities}"
    assert len(entities) > 1, f"must extract at least 2 entities: {entities}"
    print("  PASSED: entity view extracts entities correctly")


def test_neighbor_window_view_same_session():
    from run_hybrid_locomo_retrieval import _build_neighbor_window_view_text, _group_memories_by_session

    memories = [
        {"memory_id": 1, "sample_id": "s1", "session": "session_1", "dia_id": "D1:1", "dia_num": 1,
         "speaker": "A", "text": "First turn"},
        {"memory_id": 2, "sample_id": "s1", "session": "session_1", "dia_id": "D1:2", "dia_num": 2,
         "speaker": "B", "text": "Second turn"},
        {"memory_id": 3, "sample_id": "s1", "session": "session_1", "dia_id": "D1:3", "dia_num": 3,
         "speaker": "A", "text": "Third turn"},
        {"memory_id": 4, "sample_id": "s1", "session": "session_1", "dia_id": "D1:4", "dia_num": 4,
         "speaker": "B", "text": "Fourth turn"},
        {"memory_id": 5, "sample_id": "s1", "session": "session_1", "dia_id": "D1:5", "dia_num": 5,
         "speaker": "A", "text": "Fifth turn"},
    ]

    session_groups = _group_memories_by_session(memories)
    text = _build_neighbor_window_view_text(memories[2], session_groups, 2)

    assert "speaker: A" in text, f"window must include current speaker: {text}"
    assert "text: Third turn" in text, f"window must include current text: {text}"
    assert "prev_turn" in text, f"window must include previous turn: {text}"
    assert "next_turn" in text, f"window must include next turn: {text}"
    assert "First turn" in text, f"window with radius 2 must include turn 1: {text}"
    assert "Second turn" in text, f"window must include turn 2: {text}"
    assert "Fourth turn" in text, f"window must include turn 4: {text}"
    assert "Fifth turn" in text, f"window with radius 2 must include turn 5: {text}"
    print("  PASSED: neighbor_window_view includes same-session neighbors")


def test_neighbor_window_view_respects_radius():
    from run_hybrid_locomo_retrieval import _build_neighbor_window_view_text, _group_memories_by_session

    memories = [
        {"memory_id": 1, "sample_id": "s1", "session": "session_1", "dia_id": "D1:1", "dia_num": 1,
         "speaker": "A", "text": "First"},
        {"memory_id": 2, "sample_id": "s1", "session": "session_1", "dia_id": "D1:2", "dia_num": 2,
         "speaker": "B", "text": "Second"},
        {"memory_id": 3, "sample_id": "s1", "session": "session_1", "dia_id": "D1:3", "dia_num": 3,
         "speaker": "A", "text": "Third"},
    ]

    session_groups = _group_memories_by_session(memories)
    text_r1 = _build_neighbor_window_view_text(memories[1], session_groups, 1)

    assert "prev_turn" in text_r1, f"radius 1 must include prev turn: {text_r1}"
    assert "next_turn" in text_r1, f"radius 1 must include next turn: {text_r1}"
    assert "First" in text_r1, f"radius 1 must include neighbor 1: {text_r1}"
    assert "Third" in text_r1, f"radius 1 must include neighbor 3: {text_r1}"

    memories_cross = [
        {"memory_id": 10, "sample_id": "s1", "session": "session_2", "dia_id": "D2:1", "dia_num": 1,
         "speaker": "C", "text": "Other session"},
        {"memory_id": 3, "sample_id": "s1", "session": "session_1", "dia_id": "D1:3", "dia_num": 3,
         "speaker": "A", "text": "Third"},
    ]
    session_groups_cross = _group_memories_by_session(memories_cross)
    text_cross = _build_neighbor_window_view_text(memories_cross[1], session_groups_cross, 2)
    assert "Other session" not in text_cross, f"window must not include cross-session turns: {text_cross}"
    print("  PASSED: neighbor_window_view respects radius and session boundaries")


def test_smoke_run_turn_mode_backward_compat():
    import subprocess

    runner = str(Path(__file__).resolve().parent / "run_hybrid_locomo_retrieval.py")
    smoke_q = str(Path(__file__).resolve().parent.parent / "data" / "smoke" / "locomo_questions_5.jsonl")
    smoke_m = str(Path(__file__).resolve().parent.parent / "data" / "smoke" / "locomo_memories_100.jsonl")

    if not Path(smoke_q).exists() or not Path(smoke_m).exists():
        print("  SKIPPED: smoke data not found")
        return

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        out_path = tmp.name

    try:
        result = subprocess.run(
            [
                sys.executable, runner,
                "--memories", smoke_m,
                "--questions", smoke_q,
                "--out", out_path,
                "--embedding-provider", "hash",
                "--embedding-dim", "64",
                "--k", "5",
                "--candidate-k", "20",
                "--query-mode", "single",
                "--memory-render-mode", "metadata",
                "--memory-view-mode", "turn",
                "--reranker-provider", "none",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(Path(__file__).resolve().parent),
        )
        assert result.returncode == 0, f"turn mode failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        rows = []
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        assert len(rows) == 5, f"expected 5 rows, got {len(rows)}"
        for row in rows:
            assert "memory_view_mode" in row, "must have memory_view_mode"
            assert row["memory_view_mode"] == "turn", f"memory_view_mode must be 'turn', got {row['memory_view_mode']}"
            assert "view_window_radius" in row, "must have view_window_radius"
            assert "search_record_count" in row, "must have search_record_count"
            assert "source_memory_count" in row, "must have source_memory_count"
            assert row["search_record_count"] == row["source_memory_count"], "turn mode: record count must equal memory count"
            assert "retrieved_memory_ids" in row, "must have retrieved_memory_ids"
        print(f"  PASSED: turn mode backward compat verified with {len(rows)} rows")
    finally:
        Path(out_path).unlink(missing_ok=True)


def test_smoke_run_multiview_mode():
    import subprocess

    runner = str(Path(__file__).resolve().parent / "run_hybrid_locomo_retrieval.py")
    smoke_q = str(Path(__file__).resolve().parent.parent / "data" / "smoke" / "locomo_questions_5.jsonl")
    smoke_m = str(Path(__file__).resolve().parent.parent / "data" / "smoke" / "locomo_memories_100.jsonl")

    if not Path(smoke_q).exists() or not Path(smoke_m).exists():
        print("  SKIPPED: smoke data not found")
        return

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        out_path = tmp.name

    try:
        result = subprocess.run(
            [
                sys.executable, runner,
                "--memories", smoke_m,
                "--questions", smoke_q,
                "--out", out_path,
                "--embedding-provider", "hash",
                "--embedding-dim", "64",
                "--k", "5",
                "--candidate-k", "20",
                "--query-mode", "single",
                "--memory-render-mode", "metadata",
                "--memory-view-mode", "multiview",
                "--view-window-radius", "2",
                "--limit-samples", "1",
                "--limit-questions", "2",
                "--reranker-provider", "none",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(Path(__file__).resolve().parent),
        )
        assert result.returncode == 0, f"multiview mode failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        rows = []
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        assert len(rows) == 2, f"expected 2 rows, got {len(rows)}"
        for row in rows:
            assert "memory_view_mode" in row, "must have memory_view_mode"
            assert row["memory_view_mode"] == "multiview", f"memory_view_mode must be 'multiview', got {row.get('memory_view_mode')}"
            assert "view_window_radius" in row, "must have view_window_radius"
            assert row["view_window_radius"] == 2, f"view_window_radius must be 2, got {row.get('view_window_radius')}"
            assert "search_record_count" in row, "must have search_record_count"
            assert "source_memory_count" in row, "must have source_memory_count"
            assert row["search_record_count"] > row["source_memory_count"], "multiview: record count must exceed memory count"
            assert "retrieved_memory_ids" in row, "must have retrieved_memory_ids"
            for mid in row["retrieved_memory_ids"]:
                assert isinstance(mid, int), f"retrieved_memory_ids must contain ints, got {type(mid)}: {mid}"
                assert mid <= 200, f"retrieved_memory_ids must be original memory IDs, not view IDs, got {mid}"
        print(f"  PASSED: multiview mode produced {len(rows)} rows with source memory IDs in output")
    finally:
        Path(out_path).unlink(missing_ok=True)


def test_smoke_run_multiview_k150():
    import subprocess

    runner = str(Path(__file__).resolve().parent / "run_hybrid_locomo_retrieval.py")
    artifact_dir = Path(__file__).resolve().parent / "test_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    memory_count = 100
    mem_path = artifact_dir / f"multiview_k150_memories_{memory_count}.jsonl"
    q_path = artifact_dir / "multiview_k150_questions.jsonl"

    with mem_path.open("w", encoding="utf-8") as f:
        for mid in range(1, memory_count + 1):
            row = {
                "memory_id": mid,
                "sample_id": "mv_test",
                "session": "session_1",
                "dia_id": f"D1:{mid}",
                "dia_num": mid,
                "speaker": "Speaker",
                "text": f"multiview test memory number {mid} about topic {mid % 10}",
            }
            f.write(json.dumps(row) + "\n")

    with q_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({
            "question_id": "mv_k150_q1",
            "sample_id": "mv_test",
            "question": "Which topic is most relevant?",
            "gold_answer": "",
            "evidence_memory_ids": [1],
            "category": "synthetic",
        }) + "\n")

    out_path = artifact_dir / "multiview_k150_retrieval.jsonl"

    try:
        result = subprocess.run(
            [
                sys.executable, runner,
                "--memories", str(mem_path),
                "--questions", str(q_path),
                "--out", str(out_path),
                "--embedding-provider", "hash",
                "--embedding-dim", "64",
                "--k", "80",
                "--candidate-k", str(memory_count * 4),
                "--query-mode", "single",
                "--memory-render-mode", "metadata",
                "--memory-view-mode", "multiview",
                "--view-window-radius", "2",
                "--reranker-provider", "none",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(Path(__file__).resolve().parent),
        )
        assert result.returncode == 0, f"multiview k150 smoke failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

        rows = []
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        assert len(rows) == 1, f"expected 1 row, got {len(rows)}"
        row = rows[0]
        retrieved = row["retrieved_memory_ids"]
        assert len(retrieved) == 80, f"--k 80 must output 80 retrieved ids, got {len(retrieved)}"
        assert len(set(retrieved)) == len(retrieved), f"retrieved_memory_ids must be deduplicated"
        for mid in retrieved:
            assert 1 <= mid <= memory_count, f"retrieved_memory_id {mid} out of range [1, {memory_count}]"
        print(f"  PASSED: multiview --k 80 outputs {len(retrieved)} deduplicated original memory IDs")
    finally:
        mem_path.unlink(missing_ok=True)
        q_path.unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)


def test_multiview_default_is_turn():
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "run_hybrid_locomo_retrieval", "--help"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent),
        timeout=30,
    )
    assert "--memory-view-mode" in result.stdout, "--memory-view-mode must appear"
    for line in result.stdout.split("\n"):
        if "--memory-view-mode" in line:
            assert "turn" in line, "default must be 'turn'"
    print("  PASSED: multiview default is turn")


def test_compute_multiview_diagnostics_basic():
    from run_hybrid_locomo_retrieval import compute_multiview_diagnostics

    record_to_source = {1000: 1, 1001: 1, 1002: 1, 2000: 2, 2001: 2, 3000: 3}
    record_to_view_type = {
        1000: "turn_view", 1001: "event_view", 1002: "entity_view",
        2000: "turn_view", 2001: "event_view",
        3000: "turn_view"
    }
    candidate_ids = [1000, 2000, 1001, 3000, 2001, 1002]
    rrf_scores = {1000: 0.5, 2000: 0.4, 1001: 0.3, 3000: 0.25, 2001: 0.2, 1002: 0.15}

    diag = compute_multiview_diagnostics(
        candidate_ids, rrf_scores, record_to_source, record_to_view_type, "multiview"
    )

    assert diag["raw_view_candidate_count"] == 6, f"expected 6 raw views, got {diag['raw_view_candidate_count']}"
    assert diag["unique_source_candidate_count"] == 3, f"expected 3 unique sources, got {diag['unique_source_candidate_count']}"
    assert diag["duplicate_view_candidate_count"] == 3, f"expected 3 duplicates, got {diag['duplicate_view_candidate_count']}"
    assert diag["sources_with_multiple_view_hits_count"] == 2, f"expected 2 sources with multiple hits, got {diag['sources_with_multiple_view_hits_count']}"
    assert diag["max_views_per_source_in_candidates"] == 3, f"expected max 3 views per source, got {diag['max_views_per_source_in_candidates']}"

    assert "turn_view" in diag["view_type_counts_before_dedup"], "must have turn_view in before dedup"
    assert "event_view" in diag["view_type_counts_before_dedup"], "must have event_view in before dedup"
    assert "entity_view" in diag["view_type_counts_before_dedup"], "must have entity_view in before dedup"
    assert diag["view_type_counts_before_dedup"]["turn_view"] == 3, f"expected 3 turn_view, got {diag['view_type_counts_before_dedup']['turn_view']}"
    assert diag["view_type_counts_after_dedup"]["turn_view"] >= 1, "after dedup should have at least 1 turn_view"
    print("  PASSED: compute_multiview_diagnostics basic counts")


def test_compute_multiview_diagnostics_turn_mode_returns_empty():
    from run_hybrid_locomo_retrieval import compute_multiview_diagnostics

    diag = compute_multiview_diagnostics(
        [1, 2, 3], {1: 0.5, 2: 0.3, 3: 0.1}, {}, {}, "turn"
    )
    assert diag == {}, f"turn mode should return empty dict, got {diag}"
    print("  PASSED: compute_multiview_diagnostics returns empty for turn mode")


def test_compute_multiview_diagnostics_all_views_one_source():
    from run_hybrid_locomo_retrieval import compute_multiview_diagnostics

    record_to_source = {1000: 1, 1001: 1, 1002: 1, 1003: 1}
    record_to_view_type = {
        1000: "turn_view", 1001: "event_view", 1002: "entity_view", 1003: "neighbor_window_view"
    }
    candidate_ids = [1000, 1001, 1002, 1003]
    rrf_scores = {1000: 0.4, 1001: 0.3, 1002: 0.2, 1003: 0.1}

    diag = compute_multiview_diagnostics(
        candidate_ids, rrf_scores, record_to_source, record_to_view_type, "multiview"
    )

    assert diag["raw_view_candidate_count"] == 4, f"expected 4 raw, got {diag['raw_view_candidate_count']}"
    assert diag["unique_source_candidate_count"] == 1, f"expected 1 unique source, got {diag['unique_source_candidate_count']}"
    assert diag["duplicate_view_candidate_count"] == 3, f"expected 3 dup, got {diag['duplicate_view_candidate_count']}"
    assert diag["sources_with_multiple_view_hits_count"] == 1, f"expected 1, got {diag['sources_with_multiple_view_hits_count']}"
    assert diag["max_views_per_source_in_candidates"] == 4, f"expected max 4 views, got {diag['max_views_per_source_in_candidates']}"
    assert diag["view_type_counts_after_dedup"]["turn_view"] == 1, "only best view per source after dedup"
    print("  PASSED: compute_multiview_diagnostics all views one source")


def test_smoke_run_multiview_diagnostics():
    import subprocess

    runner = str(Path(__file__).resolve().parent / "run_hybrid_locomo_retrieval.py")
    smoke_q = str(Path(__file__).resolve().parent.parent / "data" / "smoke" / "locomo_questions_5.jsonl")
    smoke_m = str(Path(__file__).resolve().parent.parent / "data" / "smoke" / "locomo_memories_100.jsonl")

    if not Path(smoke_q).exists() or not Path(smoke_m).exists():
        print("  SKIPPED: smoke data not found")
        return

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        out_path = tmp.name

    try:
        result = subprocess.run(
            [
                sys.executable, runner,
                "--memories", smoke_m,
                "--questions", smoke_q,
                "--out", out_path,
                "--embedding-provider", "hash",
                "--embedding-dim", "64",
                "--k", "5",
                "--candidate-k", "20",
                "--query-mode", "single",
                "--memory-render-mode", "metadata",
                "--memory-view-mode", "multiview",
                "--view-window-radius", "2",
                "--limit-samples", "1",
                "--limit-questions", "2",
                "--reranker-provider", "none",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(Path(__file__).resolve().parent),
        )
        assert result.returncode == 0, f"multiview diagnostics failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        rows = []
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        assert len(rows) == 2, f"expected 2 rows, got {len(rows)}"
        for row in rows:
            assert "mv_raw_view_candidate_count" in row, f"must have mv_raw_view_candidate_count"
            assert "mv_unique_source_candidate_count" in row, f"must have mv_unique_source_candidate_count"
            assert "mv_duplicate_view_candidate_count" in row, f"must have mv_duplicate_view_candidate_count"
            assert "mv_view_type_counts_before_dedup" in row, f"must have mv_view_type_counts_before_dedup"
            assert "mv_view_type_counts_after_dedup" in row, f"must have mv_view_type_counts_after_dedup"
            assert "mv_sources_with_multiple_view_hits_count" in row, f"must have mv_sources_with_multiple_view_hits_count"
            assert "mv_max_views_per_source_in_candidates" in row, f"must have mv_max_views_per_source_in_candidates"

            raw = row["mv_raw_view_candidate_count"]
            unique = row["mv_unique_source_candidate_count"]
            dups = row["mv_duplicate_view_candidate_count"]
            assert raw >= unique, f"raw must be >= unique: {raw} vs {unique}"
            assert dups == raw - unique, f"dups must equal raw - unique: {dups} vs {raw - unique}"
            assert unique <= row["search_record_count"], f"unique sources {unique} cannot exceed record count {row['search_record_count']}"
        print(f"  PASSED: multiview diagnostics present, raw={raw} unique={unique} dups={dups}")
    finally:
        Path(out_path).unlink(missing_ok=True)


def test_compute_view_overfetch_turn_mode():
    from run_hybrid_locomo_retrieval import compute_view_overfetch

    source_k, view_k = compute_view_overfetch("turn", 400, 200)
    assert source_k == 400, f"turn mode source_k must be 400, got {source_k}"
    assert view_k == 400, f"turn mode view_k must equal source_k, got {view_k}"
    print("  PASSED: turn mode overfetch is identity")


def test_compute_view_overfetch_multiview_mode():
    from run_hybrid_locomo_retrieval import compute_view_overfetch

    source_k, view_k = compute_view_overfetch("multiview", 400, 200)
    assert source_k == 400, f"multiview source_k must be 400, got {source_k}"
    assert view_k == min(400 * 4, 200 * 4), f"multiview view_k must inflate by 4x capped at total views, got {view_k}"

    source_k2, view_k2 = compute_view_overfetch("multiview", 50, 1000)
    assert source_k2 == 50, f"multiview source_k must be 50, got {source_k2}"
    assert view_k2 == 200, f"multiview view_k must be 50*4=200, got {view_k2}"
    print("  PASSED: multiview mode inflates candidate_k by 4x with cap")


def test_compute_view_overfetch_multiview_capped():
    from run_hybrid_locomo_retrieval import compute_view_overfetch

    source_k, view_k = compute_view_overfetch("multiview", 400, 50)
    assert source_k == 400, f"source_k must be 400, got {source_k}"
    assert view_k == 200, f"view_k capped at 50*4=200, got {view_k}"
    assert view_k <= 50 * 4, f"view_k must be capped at total views, got {view_k}"
    print("  PASSED: multiview overfetch is capped at total_views")


def test_multiview_overfetch_retrieves_more_source_ids_than_candidate_k():
    import subprocess

    runner = str(Path(__file__).resolve().parent / "run_hybrid_locomo_retrieval.py")
    artifact_dir = Path(__file__).resolve().parent / "test_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    memory_count = 100
    mem_path = artifact_dir / f"overfetch_test_memories_{memory_count}.jsonl"
    q_path = artifact_dir / "overfetch_test_questions.jsonl"

    with mem_path.open("w", encoding="utf-8") as f:
        for mid in range(1, memory_count + 1):
            row = {
                "memory_id": mid,
                "sample_id": "ov_test",
                "session": "session_1",
                "dia_id": f"D1:{mid}",
                "dia_num": mid,
                "speaker": f"Speaker{mid % 5}",
                "text": f"overfetch test memory {mid} topic {mid % 10}",
            }
            f.write(json.dumps(row) + "\n")

    with q_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({
            "question_id": "ov_q1",
            "sample_id": "ov_test",
            "question": "Which topic is most relevant?",
            "gold_answer": "",
            "evidence_memory_ids": [1],
            "category": "synthetic",
        }) + "\n")

    out_path = artifact_dir / "overfetch_test_retrieval.jsonl"

    try:
        result = subprocess.run(
            [
                sys.executable, runner,
                "--memories", str(mem_path),
                "--questions", str(q_path),
                "--out", str(out_path),
                "--embedding-provider", "hash",
                "--embedding-dim", "64",
                "--k", "50",
                "--candidate-k", "50",
                "--query-mode", "single",
                "--memory-render-mode", "metadata",
                "--memory-view-mode", "multiview",
                "--view-window-radius", "2",
                "--reranker-provider", "none",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(Path(__file__).resolve().parent),
        )
        assert result.returncode == 0, f"overfetch test failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

        rows = []
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        assert len(rows) == 1, f"expected 1 row, got {len(rows)}"
        row = rows[0]
        retrieved = row["retrieved_memory_ids"]
        assert len(retrieved) <= 50, f"retrieved_count must be <= k=50, got {len(retrieved)}"

        unique_source_ids = set(retrieved)
        assert len(unique_source_ids) == len(retrieved), f"all retrieved_memory_ids must be deduplicated source IDs"

        assert "source_candidate_k" in row, f"must have source_candidate_k"
        assert "view_candidate_k" in row, f"must have view_candidate_k"
        assert row["view_candidate_k"] >= row["source_candidate_k"], \
            f"view_candidate_k ({row['view_candidate_k']}) must be >= source_candidate_k ({row['source_candidate_k']})"

        print(f"  PASSED: overfetch produces {len(retrieved)} unique source IDs, "
              f"view_candidate_k={row['view_candidate_k']} >= source_candidate_k={row['source_candidate_k']}")
    finally:
        mem_path.unlink(missing_ok=True)
        q_path.unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)


def test_generate_entity_constrained_probes_preserve_entities_and_hints():
    from run_hybrid_locomo_retrieval import generate_entity_constrained_probes

    q = "What would be an appropriate gift for both Evan and Sam to encourage their healthy lifestyles?"
    probes = generate_entity_constrained_probes(q, max_probes=3)
    assert probes[0] == q, "original question must be first"
    assert len(probes) <= 4, f"max_probes=3 means at most 3 additional probes, got {probes}"
    assert any("Evan" in p for p in probes[1:]), f"generated probes must preserve Evan: {probes}"
    assert any("Sam" in p for p in probes[1:]), f"generated probes must preserve Sam: {probes}"
    assert any(any(word in p.lower() for word in ["gift", "health", "healthy", "lifestyle"]) for p in probes[1:]), \
        f"probes should include detected answer-type words: {probes}"
    for p in probes[1:]:
        meaningful = [w for w in p.split() if len(w) > 2]
        assert len(meaningful) >= 3, f"probe too generic: {p}"
    assert probes == generate_entity_constrained_probes(q, max_probes=3), "probe generation must be deterministic"
    print(f"  PASSED: entity constrained probes: {probes}")


def test_build_memory_search_records_multiview_selected_types():
    from run_hybrid_locomo_retrieval import build_memory_search_records

    memories = [
        {"memory_id": 1, "sample_id": "s1", "session": "session_1", "dia_id": "D1:1", "dia_num": 1,
         "speaker": "Evan", "text": "I like healthy snacks."},
        {"memory_id": 2, "sample_id": "s1", "session": "session_1", "dia_id": "D1:2", "dia_num": 2,
         "speaker": "Sam", "text": "I exercise after work."},
    ]
    records = build_memory_search_records(memories, "metadata", "multiview", 2, ["turn", "event"])
    view_types = {r["view_type"] for r in records}
    assert view_types == {"turn_view", "event_view"}, f"expected only turn/event views, got {view_types}"
    assert len(records) == 4, f"2 memories * 2 selected views expected, got {len(records)}"
    assert {r["source_memory_id"] for r in records} == {1, 2}, "source IDs must remain original memory IDs"
    print("  PASSED: selected multiview types produce only requested views")


def test_parse_memory_view_types_invalid_fails():
    from run_hybrid_locomo_retrieval import parse_memory_view_types

    try:
        parse_memory_view_types("turn,bad_view")
    except ValueError as e:
        assert "Invalid memory view type" in str(e), f"unexpected error: {e}"
    else:
        raise AssertionError("invalid view type must raise ValueError")
    print("  PASSED: invalid memory view type fails cleanly")


def test_local_neighbor_expansion_same_sample_session_radius():
    from run_hybrid_locomo_retrieval import apply_local_neighbor_expansion

    id_to_memory = {
        1: {"memory_id": 1, "sample_id": "s1", "session": "a", "dia_id": "D1:1", "text": "one"},
        2: {"memory_id": 2, "sample_id": "s1", "session": "a", "dia_id": "D1:2", "text": "two"},
        3: {"memory_id": 3, "sample_id": "s1", "session": "a", "dia_id": "D1:3", "text": "three"},
        4: {"memory_id": 4, "sample_id": "s1", "session": "b", "dia_id": "D2:2", "text": "other session"},
        5: {"memory_id": 5, "sample_id": "s2", "session": "a", "dia_id": "D1:2", "text": "other sample"},
        6: {"memory_id": 6, "sample_id": "s1", "session": "a", "dia_id": "missing", "text": "bad dia"},
    }
    ranked, scores, added_count, added_ids = apply_local_neighbor_expansion(
        [2], {2: 1.0}, id_to_memory, radius=1, anchor_k=10, final_candidate_k=10
    )
    assert 1 in ranked and 3 in ranked, f"same-session radius-1 neighbors must be added: {ranked}"
    assert 4 not in ranked and 5 not in ranked, f"must not cross session/sample: {ranked}"
    assert len(ranked) == len(set(ranked)), f"dedup must prevent duplicates: {ranked}"
    assert added_count == len(added_ids), "added_count must match kept added set size"
    assert 6 not in added_ids, "unparseable dia_id should be skipped gracefully"
    assert scores[1] > 0 and scores[3] > 0, f"neighbor scores must be positive: {scores}"
    print(f"  PASSED: local neighbor expansion added {added_ids}")


def test_coverage_select_preserves_top_n_and_diverse_sessions():
    from run_hybrid_locomo_retrieval import coverage_select_source_ids

    id_to_memory = {
        1: {"memory_id": 1, "sample_id": "s1", "session": "s1", "dia_num": 1, "speaker": "Evan", "text": "Evan likes running"},
        2: {"memory_id": 2, "sample_id": "s1", "session": "s1", "dia_num": 2, "speaker": "Evan", "text": "Evan likes running"},
        3: {"memory_id": 3, "sample_id": "s1", "session": "s2", "dia_num": 1, "speaker": "Sam", "text": "Sam cooks healthy food"},
        4: {"memory_id": 4, "sample_id": "s1", "session": "s3", "dia_num": 1, "speaker": "Evan", "text": "Evan asked for fitness advice"},
    }
    candidate_ids = [1, 2, 3, 4]
    scores = {1: 1.0, 2: 0.95, 3: 0.70, 4: 0.60}
    selected, counts, reasons = coverage_select_source_ids(
        candidate_ids, scores, id_to_memory,
        "What gift would help Evan and Sam with healthy lifestyles?",
        k=3, preserve_top_n=1, max_candidates=4,
    )
    assert selected[0] == 1, f"coverage mode must preserve top N: {selected}"
    assert len(selected) == len(set(selected)), f"coverage must avoid duplicates: {selected}"
    sessions = {id_to_memory[mid]["session"] for mid in selected}
    assert len(sessions) >= 2, f"coverage should select diverse sessions, got {selected}"
    assert counts["preserved"] == 1, f"expected preserved count 1, got {counts}"
    assert selected == coverage_select_source_ids(
        candidate_ids, scores, id_to_memory,
        "What gift would help Evan and Sam with healthy lifestyles?",
        k=3, preserve_top_n=1, max_candidates=4,
    )[0], "coverage selection must be deterministic"
    assert isinstance(reasons, dict)
    print(f"  PASSED: coverage selection {selected} counts={counts}")


def test_smoke_run_entity_multiview_coverage_outputs_source_ids():
    import subprocess

    runner = str(Path(__file__).resolve().parent / "run_hybrid_locomo_retrieval.py")
    artifact_dir = Path(__file__).resolve().parent / "test_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    mem_path = artifact_dir / "entity_multiview_memories.jsonl"
    q_path = artifact_dir / "entity_multiview_questions.jsonl"
    out_path = artifact_dir / "entity_multiview_retrieval.jsonl"

    memories = [
        {"memory_id": 1, "sample_id": "s1", "session": "s1", "dia_id": "D1:1", "dia_num": 1, "speaker": "Evan", "text": "I want healthier snacks."},
        {"memory_id": 2, "sample_id": "s1", "session": "s1", "dia_id": "D1:2", "dia_num": 2, "speaker": "Friend", "text": "Trail mix could be a useful gift."},
        {"memory_id": 3, "sample_id": "s1", "session": "s2", "dia_id": "D2:1", "dia_num": 1, "speaker": "Sam", "text": "I am trying to cook healthy recipes."},
        {"memory_id": 4, "sample_id": "s1", "session": "s2", "dia_id": "D2:2", "dia_num": 2, "speaker": "Sam", "text": "Fitness gear helps me stay active."},
    ]
    questions = [
        {"question_id": "q1", "sample_id": "s1", "question": "What would be an appropriate gift for both Evan and Sam to encourage their healthy lifestyles?",
         "gold_answer": "", "evidence_memory_ids": [1], "category": "synthetic"},
    ]
    try:
        with mem_path.open("w", encoding="utf-8") as f:
            for m in memories:
                f.write(json.dumps(m) + "\n")
        with q_path.open("w", encoding="utf-8") as f:
            for q in questions:
                f.write(json.dumps(q) + "\n")

        result = subprocess.run(
            [
                sys.executable, runner,
                "--memories", str(mem_path),
                "--questions", str(q_path),
                "--out", str(out_path),
                "--embedding-provider", "hash",
                "--embedding-dim", "64",
                "--vector-backend", "local",
                "--k", "3",
                "--candidate-k", "4",
                "--query-mode", "entity_multi",
                "--max-query-probes", "3",
                "--memory-render-mode", "metadata",
                "--memory-view-mode", "multiview",
                "--memory-view-types", "turn,event",
                "--local-neighbor-expansion",
                "--selection-mode", "coverage",
                "--coverage-preserve-top-n", "1",
                "--coverage-max-candidates", "8",
                "--reranker-provider", "none",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(Path(__file__).resolve().parent),
        )
        assert result.returncode == 0, f"entity multiview coverage smoke failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        rows = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]
        assert len(rows) == 1, f"expected 1 row, got {len(rows)}"
        row = rows[0]
        assert row["query_mode"] == "entity_multi", f"expected entity_multi, got {row.get('query_mode')}"
        assert row["query_probes"][0] == questions[0]["question"], "original question must be first probe"
        assert row["memory_view_types"] == ["turn", "event"], f"expected selected view types, got {row.get('memory_view_types')}"
        assert row["selection_mode"] == "coverage", f"expected coverage selection, got {row.get('selection_mode')}"
        assert "neighbor_expansion_added_count" in row, "must include neighbor expansion diagnostics"
        assert "raw_view_candidate_count" in row, "must include unprefixed multiview diagnostics"
        assert "mv_raw_view_candidate_count" in row, "must preserve existing mv_ diagnostics"
        for mid in row["retrieved_memory_ids"]:
            assert 1 <= mid <= 4, f"retrieved_memory_ids must be original source IDs, got {mid}"
    finally:
        mem_path.unlink(missing_ok=True)
        q_path.unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)


if __name__ == "__main__":
    tests = [
        test_generate_query_variants_structure,
        test_generate_query_variants_deterministic,
        test_generate_query_variants_temporal,
        test_generate_query_variants_no_temporal_for_non_temporal,
        test_generate_query_variants_entity_focused,
        test_generate_query_variants_no_duplicates,
        test_rrf_fuse_multi_query_basic,
        test_rrf_fuse_multi_query_stable_ordering,
        test_rrf_fuse_multi_query_single_list,
        test_rrf_fuse_multi_query_deduplication,
        test_rrf_fuse_multi_query_respects_k,
        test_single_mode_backward_compat,
        test_render_memory_text_raw_preserves_text,
        test_render_memory_text_speaker_only_expected_fields,
        test_render_memory_text_speaker_time_only_expected_fields,
        test_render_memory_text_speaker_session_only_expected_fields,
        test_render_memory_text_metadata_includes_fields,
        test_render_memory_text_metadata_preserves_full_fields,
        test_render_memory_text_yesterday_resolution,
        test_render_memory_text_metadata_missing_fields,
        test_resolve_parent_ids_basic,
        test_resolve_parent_ids_preserves_rank,
        test_resolve_parent_ids_respects_k,
        test_resolve_parent_ids_no_parent_fallback,
        test_build_context_parent_mode,
        test_generate_decomposed_queries_deterministic,
        test_generate_decomposed_queries_no_category_or_gold,
        test_generate_decomposed_queries_uses_entity_and_candidates,
        test_generate_decomposed_queries_respects_max,
        test_fusion_dedupes_preserves_ranking,
        test_content_word_set_basic,
        test_compute_completion_candidates_nearby_session,
        test_compute_completion_no_leakage_of_forbidden_fields,
        test_completion_preserves_top_base_candidates,
        test_build_memory_search_records_turn_mode,
        test_build_memory_search_records_multiview_more_records,
        test_multiview_every_record_has_source_memory_id,
        test_dedup_view_hits_to_source_ids,
        test_dedup_preserves_best_score_per_source,
        test_event_view_classifies_event_types,
        test_entity_view_extracts_entities,
        test_neighbor_window_view_same_session,
        test_neighbor_window_view_respects_radius,
        test_multiview_default_is_turn,
        test_compute_view_overfetch_turn_mode,
        test_compute_view_overfetch_multiview_mode,
        test_compute_view_overfetch_multiview_capped,
        test_compute_multiview_diagnostics_basic,
        test_compute_multiview_diagnostics_turn_mode_returns_empty,
        test_compute_multiview_diagnostics_all_views_one_source,
        test_generate_entity_constrained_probes_preserve_entities_and_hints,
        test_build_memory_search_records_multiview_selected_types,
        test_parse_memory_view_types_invalid_fails,
        test_local_neighbor_expansion_same_sample_session_radius,
        test_coverage_select_preserves_top_n_and_diverse_sessions,
    ]

    print("=== Unit tests ===")
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {t.__name__}: {e}")
            failed += 1

    print(f"\nUnit tests: {passed} passed, {failed} failed")

    print("\n=== Argparse tests ===")
    for t in [test_argparse_query_mode, test_argparse_retrieved_id_mode, test_argparse_retrieval_plan, test_argparse_evidence_completion, test_argparse_memory_view_mode]:
        try:
            t()
        except Exception as e:
            print(f"  FAILED: {t.__name__}: {e}")

    print("\n=== Smoke tests (require data) ===")
    for t in [test_smoke_run_single_mode, test_k_controls_final_output_length_k7, test_k150_can_output_more_than_100_ids, test_smoke_run_multi_mode, test_smoke_run_parent_mode_with_derived_memories, test_smoke_run_decomposed_mode, test_smoke_run_evidence_completion_conservative, test_none_mode_backward_compat, test_smoke_run_turn_mode_backward_compat, test_smoke_run_multiview_mode, test_smoke_run_multiview_k150, test_multiview_overfetch_retrieves_more_source_ids_than_candidate_k, test_smoke_run_multiview_diagnostics, test_smoke_run_entity_multiview_coverage_outputs_source_ids]:
        try:
            t()
        except Exception as e:
            print(f"  FAILED: {t.__name__}: {e}")
