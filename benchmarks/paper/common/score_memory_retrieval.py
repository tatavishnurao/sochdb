#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

THIS_FILE = Path(__file__).resolve()
if str(THIS_FILE.parent) not in sys.path:
    sys.path.insert(0, str(THIS_FILE.parent))

from memory_schema import read_json_or_jsonl


def pct(value: float) -> float:
    return round(100.0 * value, 4)


def as_ids(values: Any) -> list[str]:
    out = []
    seen = set()
    for value in values or []:
        sid = str(value)
        if sid and sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


def unscored_reason(row: dict[str, Any]) -> str | None:
    if "evidence_memory_ids" not in row:
        return "missing_evidence_field"
    evidence = row.get("evidence_memory_ids")
    if evidence is None:
        return "missing_evidence_labels"
    if not evidence:
        status = row.get("evidence_mapping_status")
        if status == "span_mapping_failed":
            return "evidence_span_mapping_failed"
        if status == "no_evidence_available":
            return "missing_evidence_labels"
        return "empty_evidence_labels"
    return None


def evidence_bucket(count: int) -> str:
    if count <= 1:
        return "1"
    if count == 2:
        return "2"
    if count <= 5:
        return "3-5"
    return "6+"


def score_rows(rows: list[dict[str, Any]], ks: list[int]) -> dict[str, Any]:
    retrieved_lengths = [len(row.get("retrieved_memory_ids") or []) for row in rows]
    length_counts = Counter(retrieved_lengths)
    summary: dict[str, Any] = {
        "rows": len(rows),
        "retrieved_length_distribution": dict(sorted(length_counts.items())),
        "retrieved_length_min": min(retrieved_lengths) if retrieved_lengths else 0,
        "retrieved_length_max": max(retrieved_lengths) if retrieved_lengths else 0,
        "retrieved_length_avg": (
            sum(retrieved_lengths) / len(retrieved_lengths) if retrieved_lengths else 0.0
        ),
        "ks": {},
    }

    reason_counts = Counter()
    for row in rows:
        reason = unscored_reason(row)
        if reason:
            reason_counts[reason] += 1
    summary["unscored_reasons"] = dict(reason_counts)
    summary["unscored_rows"] = sum(reason_counts.values())
    summary["scored_rows"] = len(rows) - summary["unscored_rows"]

    for k in ks:
        if retrieved_lengths and max(retrieved_lengths) < k:
            summary["ks"][str(k)] = {
                "available": False,
                "reason": f"retrieved list max length is {max(retrieved_lengths)}, below K={k}",
            }
            continue

        n = hit = full = partial = zero = 0
        recall_sum = 0.0
        by_category = defaultdict(lambda: [0, 0, 0.0])
        by_sample = defaultdict(lambda: [0, 0, 0.0])
        by_evidence_count = defaultdict(lambda: [0, 0, 0.0])
        missing_examples = []

        for row in rows:
            if unscored_reason(row):
                continue
            gold = set(as_ids(row.get("evidence_memory_ids")))
            retrieved = set(as_ids((row.get("retrieved_memory_ids") or [])[:k]))
            overlap = gold & retrieved
            row_hit = int(bool(overlap))
            row_recall = len(overlap) / len(gold) if gold else 0.0

            n += 1
            hit += row_hit
            recall_sum += row_recall
            if len(overlap) == 0:
                zero += 1
            elif len(overlap) == len(gold):
                full += 1
            else:
                partial += 1

            for bucket, key in (
                (by_category, row.get("category", "unknown")),
                (by_sample, row.get("sample_id", "unknown")),
                (by_evidence_count, evidence_bucket(len(gold))),
            ):
                bucket[str(key)][0] += 1
                bucket[str(key)][1] += row_hit
                bucket[str(key)][2] += row_recall

            missing = sorted(gold - retrieved)
            if missing and len(missing_examples) < 20:
                missing_examples.append(
                    {
                        "question_id": row.get("question_id"),
                        "sample_id": row.get("sample_id"),
                        "category": row.get("category"),
                        "missing_evidence_ids": missing,
                        "retrieved_count": len(row.get("retrieved_memory_ids") or []),
                    }
                )

        def collapse(bucket):
            return {
                key: {
                    "rows": vals[0],
                    "hit": vals[1] / vals[0] if vals[0] else 0.0,
                    "recall": vals[2] / vals[0] if vals[0] else 0.0,
                }
                for key, vals in sorted(bucket.items())
            }

        summary["ks"][str(k)] = {
            "available": True,
            "scored_rows": n,
            "hit": hit / n if n else 0.0,
            "recall": recall_sum / n if n else 0.0,
            "full_hit_rows": full,
            "partial_hit_rows": partial,
            "zero_hit_rows": zero,
            "by_category": collapse(by_category),
            "by_sample": collapse(by_sample),
            "by_evidence_count": collapse(by_evidence_count),
            "missing_evidence_examples": missing_examples,
        }

    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print(f"rows={summary['rows']}")
    print(f"scored_rows={summary['scored_rows']}")
    print(f"unscored_rows={summary['unscored_rows']}")
    print(f"unscored_reasons={summary['unscored_reasons']}")
    print(
        "retrieved_lengths="
        f"min:{summary['retrieved_length_min']} max:{summary['retrieved_length_max']} "
        f"avg:{summary['retrieved_length_avg']:.2f}"
    )
    for k, metrics in summary["ks"].items():
        print(f"\nK={k}")
        if not metrics.get("available"):
            print(f"unavailable: {metrics['reason']}")
            continue
        print(
            f"scored={metrics['scored_rows']} "
            f"Hit={pct(metrics['hit']):.2f}% Recall={pct(metrics['recall']):.2f}% "
            f"full={metrics['full_hit_rows']} partial={metrics['partial_hit_rows']} "
            f"zero={metrics['zero_hit_rows']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--summary-out")
    parser.add_argument("--ks", nargs="+", type=int, default=[20, 50, 100])
    args = parser.parse_args()

    rows = read_json_or_jsonl(args.retrieval)
    summary = score_rows(rows, args.ks)
    print_summary(summary)
    if args.summary_out:
        out = Path(args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
