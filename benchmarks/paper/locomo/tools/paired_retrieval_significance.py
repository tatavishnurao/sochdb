import argparse
import json
import random
from pathlib import Path
from collections import defaultdict

def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def key(r):
    return (
        str(r.get("sample_id")),
        str(r.get("question_id") if r.get("question_id") is not None else r.get("question")),
    )

def per_row_score(r, k):
    gold = set(map(str, r.get("evidence_memory_ids") or []))
    ret = set(map(str, (r.get("retrieved_memory_ids") or [])[:k]))
    if not gold:
        return None
    overlap = gold & ret
    return {
        "hit": int(bool(overlap)),
        "recall": len(overlap) / len(gold),
        "category": r.get("category", "unknown"),
    }

def bootstrap_ci(values, rounds=2000, seed=13):
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return (0.0, 0.0, 0.0)

    means = []
    for _ in range(rounds):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)

    means.sort()
    lo = means[int(0.025 * rounds)]
    hi = means[int(0.975 * rounds)]
    mid = sum(values) / n
    return mid, lo, hi

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--k", type=int, default=100)
    args = ap.parse_args()

    base_rows = {key(r): r for r in read_jsonl(args.base)}
    new_rows = {key(r): r for r in read_jsonl(args.new)}

    common = sorted(set(base_rows) & set(new_rows))
    print(f"base_rows={len(base_rows)} new_rows={len(new_rows)} common={len(common)} k={args.k}")

    hit_deltas = []
    recall_deltas = []
    by_cat_hit = defaultdict(list)
    by_cat_rec = defaultdict(list)

    wins = losses = ties = 0

    for kk in common:
        b = per_row_score(base_rows[kk], args.k)
        n = per_row_score(new_rows[kk], args.k)

        if b is None or n is None:
            continue

        dh = n["hit"] - b["hit"]
        dr = n["recall"] - b["recall"]

        hit_deltas.append(dh)
        recall_deltas.append(dr)
        by_cat_hit[n["category"]].append(dh)
        by_cat_rec[n["category"]].append(dr)

        if dr > 0:
            wins += 1
        elif dr < 0:
            losses += 1
        else:
            ties += 1

    for name, values in [("hit_delta", hit_deltas), ("recall_delta", recall_deltas)]:
        mid, lo, hi = bootstrap_ci(values)
        print(f"{name}: mean={mid:.6f} 95%CI=[{lo:.6f}, {hi:.6f}]")

    print(f"recall row-level wins={wins} losses={losses} ties={ties}")

    print("\nBy category recall delta:")
    for cat in sorted(by_cat_rec):
        mid, lo, hi = bootstrap_ci(by_cat_rec[cat])
        print(f"{cat:12s} n={len(by_cat_rec[cat]):4d} mean={mid:.6f} 95%CI=[{lo:.6f}, {hi:.6f}]")

if __name__ == "__main__":
    main()
