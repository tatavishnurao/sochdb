#!/usr/bin/env python3
"""
Prove mathematically that multi_hop recall cannot reach 0.90
using the current first-stage retrieval pipeline.
"""

import json, sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/paper/locomo/results/publishable_grep_allviews_k200/retrieval.jsonl"
data = [json.loads(l) for l in open(path) if l.strip()]

mh = [d for d in data if d.get("category") == "multi_hop" and (d.get("evidence_memory_ids") or [])]
print(f"Multi-hop questions with evidence: {len(mh)}")
print()

total_gold = 0
total_found_k20 = 0
total_found_k50 = 0
total_found_k100 = 0
total_found_k200 = 0
never_found = set()
rank_buckets = defaultdict(int)

for d in mh:
    gold = set(d.get("evidence_memory_ids") or [])
    ret = d.get("retrieved_memory_ids", [])
    total_gold += len(gold)

    found_k20 = gold & set(ret[:20])
    found_k50 = gold & set(ret[:50])
    found_k100 = gold & set(ret[:100])
    found_k200 = gold & set(ret[:200])

    total_found_k20 += len(found_k20)
    total_found_k50 += len(found_k50)
    total_found_k100 += len(found_k100)
    total_found_k200 += len(found_k200)

    for g in gold:
        try: rank = ret.index(g) + 1
        except ValueError:
            rank = None
            never_found.add((d["question_id"], g))
        if rank is None: pass
        elif rank <= 20: rank_buckets["1-20"] += 1
        elif rank <= 50: rank_buckets["21-50"] += 1
        elif rank <= 100: rank_buckets["51-100"] += 1
        elif rank <= 200: rank_buckets["101-200"] += 1
        else: rank_buckets[">200"] += 1

never_retrieved = len(never_found)

print(f"Total evidence IDs across {len(mh)} questions: {total_gold}")
print()
print(f"  Found at K=20:   {total_found_k20:3d} / {total_gold}  [{total_found_k20/total_gold:.4f}]")
print(f"  Found at K=50:   {total_found_k50:3d} / {total_gold}  [{total_found_k50/total_gold:.4f}]")
print(f"  Found at K=100:  {total_found_k100:3d} / {total_gold}  [{total_found_k100/total_gold:.4f}]")
print(f"  Found at K=200:  {total_found_k200:3d} / {total_gold}  [{total_found_k200/total_gold:.4f}]")
print()
print(f"  NEVER retrieved (even at K=200): {never_retrieved}")
print()

print("Evidence rank distribution:")
for bucket in ["1-20", "21-50", "51-100", "101-200"]:
    n = rank_buckets.get(bucket, 0)
    print(f"  Rank {bucket:8s}: {n:3d} / {total_gold}  ({n/total_gold*100:.1f}%)")
print(f"  NOT IN TOP-200:   {never_retrieved:3d} / {total_gold}  ({never_retrieved/total_gold*100:.1f}%)")
print()

max_found_k20 = total_found_k20 + rank_buckets.get("21-50", 0) + rank_buckets.get("51-100", 0) + rank_buckets.get("101-200", 0)
max_found_k50 = total_found_k50 + rank_buckets.get("51-100", 0) + rank_buckets.get("101-200", 0)
max_found_k100 = total_found_k100 + rank_buckets.get("101-200", 0)

print("THEORETICAL MAXIMUM (perfect re-ranking within top-200)")
print(f"  K=20  max: {max_found_k20}/{total_gold} = {max_found_k20/total_gold:.4f}")
print(f"  K=50  max: {max_found_k50}/{total_gold} = {max_found_k50/total_gold:.4f}")
print(f"  K=100 max: {max_found_k100}/{total_gold} = {max_found_k100/total_gold:.4f}")
print()

print("QUESTIONS WITH NEVER-RETRIEVED EVIDENCE")
nevers_by_q = defaultdict(int)
for qid, g in never_found:
    nevers_by_q[qid] += 1
for qid in sorted(nevers_by_q):
    print(f"  {qid}: {nevers_by_q[qid]} never-retrieved IDs")
print(f"  Total: {len(nevers_by_q)} questions affected")

target = total_gold * 0.90

print()
print("TO REACH 0.90 RECALL AT K=20")
needed_k20 = target - total_found_k20
print(f"  Need: {needed_k20:.0f} more evidence IDs in top-20")
print(f"  Available in rank 21-200: {total_found_k200 - total_found_k20}")
print(f"  VERDICT: {'POSSIBLE' if needed_k20 <= (total_found_k200 - total_found_k20) else 'off the range'}")

print()
print("TO REACH 0.90 RECALL AT K=50")
needed_k50 = target - total_found_k50
print(f"  Need: {needed_k50:.0f} more evidence IDs in top-50")
print(f"  Available in rank 51-200: {total_found_k200 - total_found_k50}")
print(f"  VERDICT: {'POSSIBLE' if needed_k50 <= (total_found_k200 - total_found_k50) else 'off the range'}")

print()
print("TO REACH 0.90 RECALL AT K=100")
needed_k100 = target - total_found_k100
print(f"  Need: {needed_k100:.0f} more evidence IDs in top-100")
print(f"  Available in rank 101-200: {total_found_k200 - total_found_k100}")
print(f"  VERDICT: {'POSSIBLE' if needed_k100 <= (total_found_k200 - total_found_k100) else 'off the range'}")
