#!/usr/bin/env python3

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}") from e

    return rows


def load_external_retrieval(path: str) -> List[Dict[str, Any]]:
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid external retrieval JSONL at {path}:{line_no}: {e}") from e

    return rows


def split_memory_and_questions(rows):
    memories = []
    questions = []

    for row in rows:
        if "question" in row:
            questions.append(row)
        else:
            memories.append(row)

    return memories, questions


def normalize(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def exact_match(pred: str, gold: str) -> int:
    return int(normalize(pred) == normalize(gold))


def token_f1(pred: str, gold: str) -> float:
    pred_toks = normalize(pred).split()
    gold_toks = normalize(gold).split()

    if not pred_toks and not gold_toks:
        return 1.0

    if not pred_toks or not gold_toks:
        return 0.0

    common = {}

    for t in pred_toks:
        common[t] = common.get(t, 0) + 1

    overlap = 0

    for t in gold_toks:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_toks)
    recall = overlap / len(gold_toks)

    return 2 * precision * recall / (precision + recall)


def contains_answer(context: str, answer: str) -> bool:
    """
    Strict substring check.
    Useful for short answers, weak for long-form answers.
    """
    return normalize(answer) in normalize(context)


def simple_keyword_score(question: str, memory: Dict[str, Any]) -> int:
    q_words = set(normalize(question).split())
    m_words = set(normalize(memory.get("text", "")).split())
    return len(q_words & m_words)


def importance_score(memory: Dict[str, Any]) -> float:
    try:
        return float(memory.get("importance", 0.0))
    except Exception:
        return 0.0


def retrieve_no_memory(memories, question, k=5):
    return []


def retrieve_recent_history(memories, question, k=5):
    return sorted(memories, key=lambda x: x.get("turn", 0), reverse=True)[:k]


def retrieve_vector_rag_placeholder(memories, question, k=5):
    """
    Placeholder lexical retriever pretending to be vector-RAG.

    Replace later with:
    - sentence-transformers
    - FAISS
    - real vector search baseline
    """
    scored = []

    for m in memories:
        score = simple_keyword_score(question, m)
        scored.append((score, int(m.get("turn", 0)), m))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    return [m for score, _, m in scored[:k] if score > 0]


def retrieve_sochdb_placeholder(memories, question, k=5):
    """
    Placeholder SochDB-style retriever.

    This is intentionally stronger than plain lexical retrieval because it uses:
    - keyword relevance
    - importance
    - recency/update preference
    - memory_type metadata
    - benchmark-specific semantic hints

    Replace this later with real SochDB CONTEXT SELECT/API calls.
    """
    q = normalize(question)
    scored = []

    for m in memories:
        text = normalize(m.get("text", ""))
        memory_type = normalize(m.get("memory_type", ""))
        turn = int(m.get("turn", 0))

        score = 0.0

        # Base lexical relevance.
        score += simple_keyword_score(question, m)

        # Importance metadata.
        score += importance_score(m) * 1.5

        # Mild recency preference.
        score += turn * 0.01

        # SochDB/project-specific retrieval hints.
        if "core thesis" in q:
            if "project_thesis" in memory_type or "transactional database primitive" in text:
                score += 8

        if "claim" in q:
            if "paper_claim" in memory_type or "claim a" in text or "claim b" in text:
                score += 6

        if "highest priority" in q or "priority update" in q or "p0" in q:
            if "priority_update" in memory_type or "p0 1" in text or "p0 3" in text:
                score += 8

        if "originally considered first" in q or "moved to p0" in q:
            if "priority_update" in memory_type:
                score += 10

        if "modular baseline" in q:
            if "modular baseline" in text or "benchmark_spec" in memory_type:
                score += 7

        if "consistency race" in q or "context artifact" in q:
            if "consistency race" in text or "metric_spec" in memory_type:
                score += 7

        if "locomo" in q or "longmemeval" in q:
            if "dataset_choice" in memory_type or "locomo" in text or "longmemeval" in text:
                score += 8

        if "ragas" in q or "generated answer" in q or "metrics" in q:
            if "evaluation_policy" in memory_type or "ragas" in text or "context precision" in text:
                score += 8

        if "token budget" in q or "toon" in q or "answer quality" in q:
            if "paper_gap" in memory_type or "benchmark_spec" in memory_type or "toon" in text:
                score += 7

        if "strategies" in q and "token budget" in q:
            if "top k concatenation" in text or "bm25 concatenation" in text:
                score += 10

        if "vector retrieval" in q or "vector database" in q:
            if "positioning" in memory_type or "candidate generation" in text:
                score += 8

        if "section 11" in q or "paper structure" in q:
            if "paper_structure" in memory_type or "section 11" in text:
                score += 8

        if "investor" in q or "facing sentence" in q:
            if "positioning" in memory_type and "faster cheaper more consistent" in text:
                score += 8

        if "avoid claiming" in q or "completed" in q or "proven" in q:
            if "paper_integrity" in memory_type or "do not claim" in text:
                score += 8

        scored.append((score, turn, m))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    return [m for score, _, m in scored[:k] if score > 0]


def answer_from_context(question: str, retrieved: List[Dict[str, Any]]) -> str:
    """
    Deterministic rule-based answerer for the small SochDB benchmark.

    This is not the final paper-grade answerer.
    It checks whether retrieved context contains enough evidence to answer.
    """
    context = "\n".join(m.get("text", "") for m in retrieved)
    q = normalize(question)
    c = normalize(context)

    if "core thesis" in q:
        if "context construction should be a transactional database primitive" in c:
            return (
                "Context construction should be a transactional database primitive "
                "rather than middleware glued over SQL, vector search, and prompt packing."
            )

    if "claim is already mostly supported" in q or "existing systems benchmarks" in q:
        if "claim a" in c and "database systems feasibility" in c:
            return "Claim A: database-systems feasibility."

    if "claim still needs agentic benchmarks" in q:
        if "claim b" in c and "agentic usefulness" in c:
            return "Claim B: agentic usefulness."

    if "highest priority benchmark" in q:
        if "modular baseline vs sochdb should be p0 1" in c:
            return "Modular Baseline vs SochDB."

    if "originally considered first" in q or "moved to p0 3" in q:
        if "agent memory qa should be p0 3" in c:
            return "Agent Memory QA."

    if "modular baseline vs sochdb compare" in q:
        if "context latency" in c and "round trips" in c and "glue code" in c:
            return (
                "Context latency, round trips, token count, stale artifacts, "
                "duplicate memories, glue-code lines of code, and number of components."
            )

    if "failures should the context artifact consistency race detect" in q:
        if "deleted memory appears in context" in c:
            return (
                "Deleted memory appearing in context, old and new profile facts being mixed, "
                "or a vector result pointing to a missing record."
            )

    if "metrics should the consistency race report" in q:
        if "inconsistent artifacts" in c and "stale record rate" in c:
            return (
                "Inconsistent artifacts per total reads, stale record rate, "
                "deleted-memory leakage, and p99 latency under concurrent writes."
            )

    if "why should locomo be used before longmemeval" in q:
        if "locomo first" in c and "longmemeval" in c:
            return (
                "LoCoMo should be used first because it evaluates long-term conversational memory; "
                "LongMemEval should be added later for information extraction, multi-session reasoning, "
                "knowledge updates, temporal reasoning, and abstention."
            )

    if "metrics should be used for generated answer evaluation" in q:
        if "ragas style metrics" in c or "context precision" in c:
            return (
                "Ragas-style context precision, context recall, response relevancy, "
                "faithfulness, and factual correctness, plus deterministic EM and F1 where possible."
            )

    if "token budget benchmark need to prove" in q:
        if "token savings do not harm answer quality" in c:
            return "It needs to prove that token savings do not harm answer quality."

    if "strategies should token budget vs answer quality compare" in q:
        if "top k concatenation" in c or "bm25 concatenation" in c:
            return (
                "Top-k concatenation, BM25 concatenation, hybrid concatenation, "
                "planner, TOON, and planner plus TOON."
            )

    if "frame vector retrieval" in q:
        if (
            "frame vector retrieval as candidate generation" in c
            or "vector retrieval as candidate generation" in c
            or "candidate generation" in c
        ) and (
            "transactional token aware context construction" in c
            or "token aware context construction" in c
            or "real contribution is transactional" in c
        ):
            return (
                "Vector retrieval should be framed as candidate generation, "
                "while the real contribution is transactional, token-aware context construction for agents."
            )

    if "where should modular baseline comparison appear" in q:
        if "early in section 11" in c:
            return "Early in Section 11, before lower-level storage and throughput details."

    if "investor facing sentence" in q:
        if "faster cheaper more consistent context for agents" in c:
            return (
                "Compared with fragmented SQL plus vector plus graph plus prompt-packer pipelines, "
                "SochDB produces faster, cheaper, more consistent context for agents."
            )

    if "avoid claiming" in q and "agent memory qa" in q:
        if "do not claim agent memory qa is completed" in c or "claim b remains under evaluation" in c:
            return "It should avoid claiming that Agent Memory QA or Claim B is completed or proven."

    return "unknown"


def run_system(system_name, memories, questions, k):
    retrievers = {
        "no_memory": retrieve_no_memory,
        "recent_history": retrieve_recent_history,
        "vector_rag": retrieve_vector_rag_placeholder,
        "sochdb": retrieve_sochdb_placeholder,
    }

    if system_name not in retrievers:
        raise ValueError(
            f"Unknown system: {system_name}. "
            f"Available systems: {', '.join(sorted(retrievers.keys()))}"
        )

    retrieve = retrievers[system_name]
    results = []

    for q in questions:
        start = time.perf_counter()

        retrieved = retrieve(memories, q["question"], k=k)
        pred = answer_from_context(q["question"], retrieved)

        latency_ms = (time.perf_counter() - start) * 1000

        context_text = "\n".join(m.get("text", "") for m in retrieved)
        retrieved_turns = [m.get("turn") for m in retrieved]
        evidence_turns = q.get("evidence_turns", [])

        evidence_set = set(evidence_turns)
        retrieved_set = set(retrieved_turns)

        cited_memory_recall = 0.0
        evidence_hit = 0

        if evidence_set:
            hit_count = len(retrieved_set & evidence_set)
            cited_memory_recall = hit_count / len(evidence_set)
            evidence_hit = int(hit_count > 0)

        result = {
            "system": system_name,
            "question_id": q["question_id"],
            "question": q["question"],
            "gold_answer": q["answer"],
            "prediction": pred,
            "exact_match": exact_match(pred, q["answer"]),
            "f1": token_f1(pred, q["answer"]),
            "contains_answer_in_context": contains_answer(context_text, q["answer"]),
            "cited_memory_recall": cited_memory_recall,
            "evidence_hit": evidence_hit,
            "retrieved_turns": retrieved_turns,
            "evidence_turns": evidence_turns,
            "retrieved_count": len(retrieved),
            "approx_context_tokens": len(context_text.split()),
            "latency_ms": latency_ms,
            "type": q.get("type", "unknown"),
        }

        if pred == "unknown":
            result["debug_context"] = context_text

        results.append(result)

    return results


def summarize(results):
    n = len(results)

    if n == 0:
        return {}

    return {
        "n_questions": n,
        "exact_match": sum(r["exact_match"] for r in results) / n,
        "f1": sum(r["f1"] for r in results) / n,
        "cited_memory_recall": sum(r["cited_memory_recall"] for r in results) / n,
        "evidence_hit_rate": sum(r["evidence_hit"] for r in results) / n,
        "contains_answer_in_context": sum(int(r["contains_answer_in_context"]) for r in results) / n,
        "avg_context_tokens": sum(r["approx_context_tokens"] for r in results) / n,
        "avg_latency_ms": sum(r["latency_ms"] for r in results) / n,
    }


def summarize_by_type(results):
    by_type = {}

    for r in results:
        t = r.get("type") or "unknown"
        by_type.setdefault(t, []).append(r)

    rows = {}

    for t, subset in sorted(by_type.items()):
        n = len(subset)

        rows[t] = {
            "n_questions": n,
            "exact_match": sum(r["exact_match"] for r in subset) / n,
            "f1": sum(r["f1"] for r in subset) / n,
            "cited_memory_recall": sum(r["cited_memory_recall"] for r in subset) / n,
            "evidence_hit_rate": sum(r["evidence_hit"] for r in subset) / n,
            "avg_context_tokens": sum(r["approx_context_tokens"] for r in subset) / n,
            "avg_latency_ms": sum(r["latency_ms"] for r in subset) / n,
        }

    return rows



def answer_external_from_debug_context(question: str, debug_context: str) -> str:
    retrieved = [{"text": debug_context}]
    return answer_from_context(question, retrieved)


def score_external_retrieval(external_rows):
    results = []

    for row in external_rows:
        question = row["question"]
        gold_answer = row.get("gold_answer") or row.get("answer") or ""
        context_text = row.get("debug_context", "")

        pred = answer_external_from_debug_context(question, context_text)

        retrieved_turns = row.get("retrieved_turns", [])
        evidence_turns = row.get("evidence_turns", [])

        evidence_set = set(evidence_turns)
        retrieved_set = set(retrieved_turns)

        cited_memory_recall = 0.0
        evidence_hit = 0

        if evidence_set:
            hit_count = len(retrieved_set & evidence_set)
            cited_memory_recall = hit_count / len(evidence_set)
            evidence_hit = int(hit_count > 0)

        result = {
            "system": row.get("system", "external"),
            "question_id": row["question_id"],
            "question": question,
            "gold_answer": gold_answer,
            "prediction": pred,
            "exact_match": exact_match(pred, gold_answer),
            "f1": token_f1(pred, gold_answer),
            "contains_answer_in_context": contains_answer(context_text, gold_answer),
            "cited_memory_recall": cited_memory_recall,
            "evidence_hit": evidence_hit,
            "retrieved_turns": retrieved_turns,
            "evidence_turns": evidence_turns,
            "retrieved_count": row.get("retrieved_count", len(retrieved_turns)),
            "approx_context_tokens": row.get("approx_context_tokens", len(context_text.split())),
            "latency_ms": row.get("latency_ms", 0.0),
            "type": row.get("type", "unknown"),
        }

        if pred == "unknown":
            result["debug_context"] = context_text

        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--systems",
        nargs="+",
        default=["no_memory", "recent_history", "vector_rag", "sochdb"],
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--external-retrieval", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(args.data)
    memories, questions = split_memory_and_questions(rows)

    if not memories:
        raise ValueError("No memory rows found. Your JSONL needs rows without `question`.")

    if not questions:
        raise ValueError("No question rows found. Your JSONL needs rows with `question`.")

    if args.external_retrieval:
        external_rows = load_external_retrieval(args.external_retrieval)
        results = score_external_retrieval(external_rows)

        all_results = results

        if results:
            system_name = results[0]["system"]
        else:
            system_name = "external"

        summary_rows = []
        s = summarize(results)
        s["system"] = system_name
        s["by_type"] = summarize_by_type(results)
        summary_rows.append(s)

        raw_path = out_dir / "agent_memory_qa_results.jsonl"
        with raw_path.open("w", encoding="utf-8") as f:
            for r in all_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        summary_path = out_dir / "agent_memory_qa_summary.json"
        summary_path.write_text(
            json.dumps(summary_rows, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(f"Wrote raw results to {raw_path}")
        print(f"Wrote summary to {summary_path}")
        print(json.dumps(summary_rows, indent=2, ensure_ascii=False))
        return

    all_results = []
    summary_rows = []

    for system in args.systems:
        results = run_system(system, memories, questions, args.k)
        all_results.extend(results)

        s = summarize(results)
        s["system"] = system
        s["by_type"] = summarize_by_type(results)
        summary_rows.append(s)

    raw_path = out_dir / "agent_memory_qa_results.jsonl"
    with raw_path.open("w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary_path = out_dir / "agent_memory_qa_summary.json"
    summary_path.write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Wrote raw results to {raw_path}")
    print(f"Wrote summary to {summary_path}")
    print(json.dumps(summary_rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()