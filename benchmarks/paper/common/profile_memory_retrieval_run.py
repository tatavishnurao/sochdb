#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

THIS_FILE = Path(__file__).resolve()
if str(THIS_FILE.parent) not in sys.path:
    sys.path.insert(0, str(THIS_FILE.parent))

from memory_schema import read_json_or_jsonl
from score_memory_retrieval import as_ids, score_rows, unscored_reason


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}"


def print_table(title: str, rows: list[list[Any]], headers: list[str]) -> None:
    print(f"\n## {title}")
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        print("| " + " | ".join(str(x) for x in row) + " |")


def gold_rank_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    all_ranks = []
    min_rank_per_question = []
    max_rank_per_question = []
    buckets = Counter()
    missing_by_id = Counter()
    total_gold = 0

    for row in rows:
        if unscored_reason(row):
            continue
        gold = as_ids(row.get("evidence_memory_ids"))
        retrieved = as_ids(row.get("retrieved_memory_ids"))
        rank = {}
        for idx, mid in enumerate(retrieved, start=1):
            rank.setdefault(mid, idx)

        row_ranks = []
        for mid in gold:
            total_gold += 1
            found_rank = rank.get(mid)
            if found_rank is None:
                buckets["missing"] += 1
                missing_by_id[mid] += 1
                continue
            all_ranks.append(found_rank)
            row_ranks.append(found_rank)
            if found_rank <= 20:
                buckets["1-20"] += 1
            elif found_rank <= 50:
                buckets["21-50"] += 1
            elif found_rank <= 100:
                buckets["51-100"] += 1
            elif found_rank <= 150:
                buckets["101-150"] += 1
            elif found_rank <= 200:
                buckets["151-200"] += 1
            else:
                buckets[">200"] += 1
        if row_ranks:
            min_rank_per_question.append(min(row_ranks))
            max_rank_per_question.append(max(row_ranks))

    def quantile(values: list[int], q: float) -> int | None:
        if not values:
            return None
        ordered = sorted(values)
        return ordered[int((len(ordered) - 1) * q)]

    return {
        "total_gold": total_gold,
        "buckets": buckets,
        "missing_by_id": missing_by_id,
        "rank_mean": mean(all_ranks) if all_ranks else None,
        "rank_median": median(all_ranks) if all_ranks else None,
        "rank_p90": quantile(all_ranks, 0.90),
        "rank_p95": quantile(all_ranks, 0.95),
        "min_rank_mean": mean(min_rank_per_question) if min_rank_per_question else None,
        "max_rank_mean": mean(max_rank_per_question) if max_rank_per_question else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--ks", nargs="+", type=int, default=[20, 50, 100])
    parser.add_argument("--top-missing", type=int, default=10)
    args = parser.parse_args()

    rows = read_json_or_jsonl(args.retrieval)
    summary = score_rows(rows, args.ks)
    lengths = [len(row.get("retrieved_memory_ids") or []) for row in rows]

    print("# Retrieval Profile")
    print(f"retrieval_file: {args.retrieval}")
    print(f"rows: {len(rows)}")
    print(f"scored_rows: {summary['scored_rows']}")
    print(f"unscored_rows: {summary['unscored_rows']}")
    print(f"unscored_reasons: {summary['unscored_reasons']}")
    print(f"retrieved_len_distribution: {Counter(lengths).most_common()}")

    length_short_rows = []
    for k in args.ks:
        short = sum(1 for length in lengths if length < k)
        length_short_rows.append([k, short])
    print_table("Rows below requested K", length_short_rows, ["K", "rows"])

    overall_rows = []
    for k in args.ks:
        metrics = summary["ks"][str(k)]
        if not metrics.get("available"):
            overall_rows.append([k, "unavailable", metrics["reason"], "", "", ""])
            continue
        overall_rows.append(
            [
                k,
                metrics["scored_rows"],
                pct(metrics["hit"]),
                pct(metrics["recall"]),
                metrics["full_hit_rows"],
                metrics["zero_hit_rows"],
            ]
        )
    print_table("Overall metrics", overall_rows, ["K", "scored", "Hit", "Recall", "Full", "Zero"])

    available_ks = [k for k in args.ks if summary["ks"][str(k)].get("available")]
    if available_ks:
        main_k = max(k for k in available_ks if k <= 100) if any(k <= 100 for k in available_ks) else available_ks[0]
        metrics = summary["ks"][str(main_k)]
        for title, key, name in (
            ("Category metrics", "by_category", "category"),
            ("Sample metrics", "by_sample", "sample"),
            ("Evidence-count metrics", "by_evidence_count", "evidence_count"),
        ):
            table_rows = [
                [bucket, vals["rows"], pct(vals["hit"]), pct(vals["recall"])]
                for bucket, vals in metrics[key].items()
            ]
            print_table(f"{title} at K={main_k}", table_rows, [name, "rows", "Hit", "Recall"])

    rank_profile = gold_rank_profile(rows)
    rank_rows = []
    total_gold = rank_profile["total_gold"]
    for bucket in ["1-20", "21-50", "51-100", "101-150", "151-200", ">200", "missing"]:
        count = rank_profile["buckets"].get(bucket, 0)
        rank_rows.append([bucket, count, pct(count / total_gold if total_gold else 0.0)])
    print_table("Gold rank distribution", rank_rows, ["rank_bucket", "gold_count", "percent"])

    print("\n## Gold rank summary")
    for key in ("total_gold", "rank_mean", "rank_median", "rank_p90", "rank_p95", "min_rank_mean", "max_rank_mean"):
        print(f"{key}: {rank_profile[key]}")

    print("\n## Top missing evidence IDs")
    for mid, count in rank_profile["missing_by_id"].most_common(args.top_missing):
        print(f"{mid}: {count}")

    print("\n## Missing evidence examples")
    shown = 0
    main_k = min(max(args.ks), max(lengths or [0]))
    for row in rows:
        if shown >= args.top_missing or unscored_reason(row):
            continue
        gold = set(as_ids(row.get("evidence_memory_ids")))
        retrieved = set(as_ids((row.get("retrieved_memory_ids") or [])[:main_k]))
        missing = sorted(gold - retrieved)
        if not missing:
            continue
        shown += 1
        print(
            f"- question_id={row.get('question_id')} sample_id={row.get('sample_id')} "
            f"category={row.get('category')} missing={missing}"
        )


if __name__ == "__main__":
    main()
