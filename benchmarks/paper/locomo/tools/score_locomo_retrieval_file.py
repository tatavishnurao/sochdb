import json
import sys
from pathlib import Path
from collections import defaultdict

if len(sys.argv) < 2:
    print("Usage: python benchmarks/paper/locomo/tools/score_locomo_retrieval_file.py <retrieval.jsonl> [--ks K1 K2 ...]")
    raise SystemExit

path = Path(sys.argv[1])

ks = [20, 50, 100]
if "--ks" in sys.argv:
    ks_idx = sys.argv.index("--ks")
    ks = []
    for x in sys.argv[ks_idx + 1:]:
        if x.startswith("--"):
            break
        ks.append(int(x))

if not path.exists():
    print(f"ERROR: missing file: {path}")
    raise SystemExit

rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
print(f"file={path}")
print(f"rows={len(rows)}")

def score(k):
    n = 0
    hit = 0
    recall_sum = 0.0
    by_cat = defaultdict(lambda: [0, 0, 0.0])

    for r in rows:
        gold = set(map(str, r.get("evidence_memory_ids") or []))
        ret = set(map(str, (r.get("retrieved_memory_ids") or [])[:k]))

        if not gold:
            continue

        overlap = gold & ret
        h = int(bool(overlap))
        rec = len(overlap) / len(gold)
        cat = r.get("category", "unknown")

        n += 1
        hit += h
        recall_sum += rec

        by_cat[cat][0] += 1
        by_cat[cat][1] += h
        by_cat[cat][2] += rec

    if n == 0:
        print(f"K={k}: no scored rows")
        return

    print(f"\nK={k}")
    print(f"overall n={n} hit={hit/n:.6f} recall={recall_sum/n:.6f}")

    for cat in sorted(by_cat):
        cn, ch, cr = by_cat[cat]
        print(f"  {cat:12s} n={cn:4d} hit={ch/cn:.6f} recall={cr/cn:.6f}")

for k in ks:
    score(k)
