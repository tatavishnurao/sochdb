#!/usr/bin/env python3
"""Fuse multiple retrieval result files using splice strategy.

Splice strategy: take top-N from each input file, dedup by question_id,
and keep the best K results per question.

This simulates K>200 coverage while keeping K=200 output constraint.
"""

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[dict]):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def splice_fuse(
    input_rows_list: List[List[dict]],
    n_per_input: int,
    k: int,
) -> List[dict]:
    grouped_by_qid: Dict[str, dict] = OrderedDict()

    for input_idx, rows in enumerate(input_rows_list):
        for row in rows:
            qid = row.get("question_id", "")
            if qid not in grouped_by_qid:
                grouped_by_qid[qid] = row.copy()

    for qid, base_row in grouped_by_qid.items():
        seen = set()
        fused_ids = []
        fused_views = []

        for input_idx, rows in enumerate(input_rows_list):
            q_row = None
            for r in rows:
                if r.get("question_id") == qid:
                    q_row = r
                    break

            if q_row is None:
                continue

            top_n_ids = q_row.get("retrieved_memory_ids", [])[:n_per_input]
            top_n_views = q_row.get("retrieved_memory_views", [])[:n_per_input]

            for i, mid in enumerate(top_n_ids):
                if mid not in seen:
                    seen.add(mid)
                    fused_ids.append(mid)
                    if i < len(top_n_views) and top_n_views:
                        fused_views.append(top_n_views[i])

        grouped_by_qid[qid]["retrieved_memory_ids"] = fused_ids[:k]
        grouped_by_qid[qid]["retrieved_count"] = len(fused_ids[:k])

        if fused_views:
            grouped_by_qid[qid]["retrieved_memory_views"] = fused_views[:k]

        if "retrieved_view_memory_ids" in base_row:
            seen_vids = set()
            fused_vids = []
            for input_idx, rows in enumerate(input_rows_list):
                q_row = None
                for r in rows:
                    if r.get("question_id") == qid:
                        q_row = r
                        break
                if q_row is None:
                    continue
                for vid in q_row.get("retrieved_view_memory_ids", [])[:n_per_input]:
                    if vid not in seen_vids:
                        seen_vids.add(vid)
                        fused_vids.append(vid)
            grouped_by_qid[qid]["retrieved_view_memory_ids"] = fused_vids[:k * 4]

        if "retrieved_parent_memory_ids" in base_row:
            seen_pids = set()
            fused_pids = []
            for input_idx, rows in enumerate(input_rows_list):
                q_row = None
                for r in rows:
                    if r.get("question_id") == qid:
                        q_row = r
                        break
                if q_row is None:
                    continue
                for pid in q_row.get("retrieved_parent_memory_ids", [])[:n_per_input]:
                    if pid not in seen_pids:
                        seen_pids.add(pid)
                        fused_pids.append(pid)
            grouped_by_qid[qid]["retrieved_parent_memory_ids"] = fused_pids[:k * 4]

    return list(grouped_by_qid.values())


def main():
    parser = argparse.ArgumentParser(description="Fuse multiple retrieval result files")
    parser.add_argument("--inputs", nargs="+", required=True, help="List of retrieval.jsonl files to fuse")
    parser.add_argument("--n-per-input", type=int, default=100, help="Top-N from each input file")
    parser.add_argument("--k", type=int, default=200, help="Final K to keep per question")
    parser.add_argument("--output", required=True, help="Output retrieval.jsonl path")
    args = parser.parse_args()

    input_rows_list = []
    for input_path in args.inputs:
        rows = load_jsonl(input_path)
        print(f"Loaded {len(rows)} rows from {input_path}")
        input_rows_list.append(rows)

    fused = splice_fuse(input_rows_list, n_per_input=args.n_per_input, k=args.k)
    write_jsonl(args.output, fused)
    print(f"Wrote {len(fused)} fused rows to {args.output}")


if __name__ == "__main__":
    main()