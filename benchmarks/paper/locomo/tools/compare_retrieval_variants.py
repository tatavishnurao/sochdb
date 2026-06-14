import json
import sys
from pathlib import Path
from collections import defaultdict

if len(sys.argv) < 3 or len(sys.argv[1:]) % 2 != 0:
    print("Usage: python compare_retrieval_variants.py <name1> <file1> <name2> <file2> ...")
    raise SystemExit

pairs = list(zip(sys.argv[1::2], sys.argv[2::2]))

def load_rows(path):
    path = Path(path)
    if not path.exists():
        print(f"ERROR: missing file: {path}")
        raise SystemExit
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

def metrics(rows, k):
    n = 0
    hit = 0
    recall = 0.0
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
        recall += rec

        by_cat[cat][0] += 1
        by_cat[cat][1] += h
        by_cat[cat][2] += rec

    return {
        "n": n,
        "hit": hit / n if n else 0.0,
        "recall": recall / n if n else 0.0,
        "by_cat": by_cat,
    }

loaded = [(name, load_rows(path)) for name, path in pairs]

for k in [20, 50, 100]:
    print(f"\n=== K={k} OVERALL ===")
    base = None
    for name, rows in loaded:
        m = metrics(rows, k)
        if base is None:
            base = m
            print(f"{name:36s} hit={m['hit']:.6f} recall={m['recall']:.6f}")
        else:
            print(
                f"{name:36s} "
                f"hit={m['hit']:.6f} "
                f"recall={m['recall']:.6f} "
                f"Δhit={m['hit']-base['hit']:+.6f} "
                f"Δrecall={m['recall']-base['recall']:+.6f}"
            )

print("\n=== K=100 BY CATEGORY ===")
base_name, base_rows = loaded[0]
base_m = metrics(base_rows, 100)

for name, rows in loaded[1:]:
    m = metrics(rows, 100)
    print(f"\n{name} vs {base_name}")
    for cat in sorted(set(base_m["by_cat"]) | set(m["by_cat"])):
        if cat not in base_m["by_cat"] or cat not in m["by_cat"]:
            continue

        bn, bh, br = base_m["by_cat"][cat]
        cn, ch, cr = m["by_cat"][cat]

        base_hit = bh / bn
        base_rec = br / bn
        new_hit = ch / cn
        new_rec = cr / cn

        print(
            f"{cat:12s} "
            f"base_hit={base_hit:.6f} new_hit={new_hit:.6f} Δhit={new_hit-base_hit:+.6f} "
            f"base_rec={base_rec:.6f} new_rec={new_rec:.6f} Δrec={new_rec-base_rec:+.6f}"
        )
