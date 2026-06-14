import argparse
import json
from pathlib import Path
from collections import defaultdict, Counter
from statistics import mean, median

def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                print(f"ERROR: bad JSON in {path} line {line_no}: {e}")
                raise SystemExit
    return rows

def safe_ints(xs):
    out = []
    for x in xs or []:
        try:
            out.append(int(x))
        except Exception:
            pass
    return out

def pct(x):
    return f"{100*x:.2f}"

def metric_at_k(rows, k):
    n = hit = 0
    recall_sum = 0.0
    by_cat = defaultdict(lambda: [0, 0, 0.0])
    by_sample = defaultdict(lambda: [0, 0, 0.0])
    by_gold_count = defaultdict(lambda: [0, 0, 0.0])

    full = partial = zero = 0

    for r in rows:
        gold = set(safe_ints(r.get("evidence_memory_ids")))
        if not gold:
            continue

        ret = set(safe_ints((r.get("retrieved_memory_ids") or [])[:k]))
        overlap = gold & ret
        h = int(bool(overlap))
        rec = len(overlap) / len(gold)

        n += 1
        hit += h
        recall_sum += rec

        if len(overlap) == 0:
            zero += 1
        elif len(overlap) == len(gold):
            full += 1
        else:
            partial += 1

        cat = r.get("category", "unknown")
        sample = r.get("sample_id", "unknown")
        gc = len(gold)

        for bucket, key in [
            (by_cat, cat),
            (by_sample, sample),
            (by_gold_count, str(gc)),
        ]:
            bucket[key][0] += 1
            bucket[key][1] += h
            bucket[key][2] += rec

    return {
        "n": n,
        "hit": hit / n if n else 0.0,
        "recall": recall_sum / n if n else 0.0,
        "full": full,
        "partial": partial,
        "zero": zero,
        "by_cat": by_cat,
        "by_sample": by_sample,
        "by_gold_count": by_gold_count,
    }

def gold_rank_profile(rows):
    all_gold_ranks = []
    min_rank_per_question = []
    max_rank_per_question = []
    missing_gold = 0
    total_gold = 0

    buckets = Counter()

    for r in rows:
        gold = safe_ints(r.get("evidence_memory_ids"))
        retrieved = safe_ints(r.get("retrieved_memory_ids"))
        if not gold:
            continue

        rank = {}
        for i, mid in enumerate(retrieved, 1):
            if mid not in rank:
                rank[mid] = i

        ranks = []
        for mid in gold:
            total_gold += 1
            rr = rank.get(mid)
            if rr is None:
                missing_gold += 1
                buckets["missing"] += 1
            else:
                all_gold_ranks.append(rr)
                ranks.append(rr)

                if rr <= 20:
                    buckets["1-20"] += 1
                elif rr <= 50:
                    buckets["21-50"] += 1
                elif rr <= 100:
                    buckets["51-100"] += 1
                elif rr <= 200:
                    buckets["101-200"] += 1
                else:
                    buckets[">200"] += 1

        if ranks:
            min_rank_per_question.append(min(ranks))
            max_rank_per_question.append(max(ranks))

    def q(values, quantile):
        if not values:
            return None
        values = sorted(values)
        idx = int((len(values)-1) * quantile)
        return values[idx]

    return {
        "total_gold": total_gold,
        "missing_gold": missing_gold,
        "gold_rank_buckets": buckets,
        "all_gold_rank_mean": mean(all_gold_ranks) if all_gold_ranks else None,
        "all_gold_rank_median": median(all_gold_ranks) if all_gold_ranks else None,
        "all_gold_rank_p90": q(all_gold_ranks, 0.90),
        "all_gold_rank_p95": q(all_gold_ranks, 0.95),
        "min_rank_mean": mean(min_rank_per_question) if min_rank_per_question else None,
        "max_rank_mean": mean(max_rank_per_question) if max_rank_per_question else None,
    }

def missing_examples(rows, memories, k, limit):
    memory_by_id = {}
    for m in memories:
        try:
            memory_by_id[int(m["memory_id"])] = m
        except Exception:
            pass

    examples = []

    for r in rows:
        gold = set(safe_ints(r.get("evidence_memory_ids")))
        if not gold:
            continue

        retrieved = safe_ints((r.get("retrieved_memory_ids") or [])[:k])
        ret = set(retrieved)
        missing = sorted(gold - ret)

        if not missing:
            continue

        examples.append((r, missing, retrieved[:20]))

        if len(examples) >= limit:
            break

    lines = []
    for i, (r, missing, top20) in enumerate(examples, 1):
        lines.append("")
        lines.append("=" * 100)
        lines.append(f"Example {i}")
        lines.append(f"category: {r.get('category')}")
        lines.append(f"sample_id: {r.get('sample_id')}")
        lines.append(f"question: {r.get('question')}")
        lines.append(f"gold_answer: {r.get('gold_answer')}")
        lines.append(f"gold_ids: {r.get('evidence_memory_ids')}")
        lines.append(f"missing_gold_ids@{k}: {missing}")
        lines.append(f"top20_retrieved: {top20}")

        for mid in missing[:5]:
            m = memory_by_id.get(mid)
            if m:
                lines.append(
                    f"- missing memory_id={mid} speaker={m.get('speaker')} "
                    f"session={m.get('session')} dia_id={m.get('dia_id')} "
                    f"timestamp={m.get('timestamp')}"
                )
                lines.append(f"  text: {m.get('text')}")
            else:
                lines.append(f"- missing memory_id={mid}: not found in memory file")

    return "\n".join(lines)

def print_table(title, rows, headers):
    print(f"\n## {title}")
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        print("| " + " | ".join(str(x) for x in row) + " |")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", required=True)
    ap.add_argument("--memories", required=True)
    ap.add_argument("--ks", nargs="+", type=int, default=[20, 50, 100])
    ap.add_argument("--top-missing", type=int, default=10)
    args = ap.parse_args()

    rows = read_jsonl(args.retrieval)
    memories = read_jsonl(args.memories)

    print(f"# Retrieval Profile")
    print(f"retrieval_file: {args.retrieval}")
    print(f"rows: {len(rows)}")

    lengths = [len(r.get("retrieved_memory_ids") or []) for r in rows]
    print(f"retrieved_len_min: {min(lengths)}")
    print(f"retrieved_len_max: {max(lengths)}")
    print(f"retrieved_len_avg: {sum(lengths)/len(lengths):.2f}")
    print(f"retrieved_len_distribution_top10: {Counter(lengths).most_common(10)}")

    overall_rows = []
    for k in args.ks:
        m = metric_at_k(rows, k)
        overall_rows.append([
            k,
            m["n"],
            pct(m["hit"]),
            pct(m["recall"]),
            f"{m['full']} ({pct(m['full']/m['n'])})",
            f"{m['partial']} ({pct(m['partial']/m['n'])})",
            f"{m['zero']} ({pct(m['zero']/m['n'])})",
        ])

    print_table(
        "Overall metrics by K",
        overall_rows,
        ["K", "scored", "Hit", "Recall", "Full recall rows", "Partial rows", "Zero-hit rows"]
    )

    k_main = max([k for k in args.ks if k <= 100] or args.ks)
    m = metric_at_k(rows, k_main)

    cat_rows = []
    for cat in sorted(m["by_cat"]):
        n, h, rsum = m["by_cat"][cat]
        cat_rows.append([cat, n, pct(h/n), pct(rsum/n)])
    print_table(f"Category metrics at K={k_main}", cat_rows, ["category", "n", "Hit", "Recall"])

    sample_rows = []
    for sample in sorted(m["by_sample"]):
        n, h, rsum = m["by_sample"][sample]
        sample_rows.append([sample, n, pct(h/n), pct(rsum/n)])
    print_table(f"Sample metrics at K={k_main}", sample_rows, ["sample", "n", "Hit", "Recall"])

    gold_count_rows = []
    for gc in sorted(m["by_gold_count"], key=lambda x: int(x)):
        n, h, rsum = m["by_gold_count"][gc]
        gold_count_rows.append([gc, n, pct(h/n), pct(rsum/n)])
    print_table(f"Metrics by number of gold evidence memories at K={k_main}", gold_count_rows, ["gold_count", "n", "Hit", "Recall"])

    rp = gold_rank_profile(rows)

    rank_rows = []
    total_gold = rp["total_gold"]
    for bucket in ["1-20", "21-50", "51-100", "101-200", ">200", "missing"]:
        count = rp["gold_rank_buckets"].get(bucket, 0)
        rank_rows.append([bucket, count, pct(count/total_gold if total_gold else 0)])
    print_table("Gold memory rank distribution", rank_rows, ["rank_bucket", "gold_count", "percent"])

    print("\n## Gold rank summary")
    for key in [
        "total_gold",
        "missing_gold",
        "all_gold_rank_mean",
        "all_gold_rank_median",
        "all_gold_rank_p90",
        "all_gold_rank_p95",
        "min_rank_mean",
        "max_rank_mean",
    ]:
        print(f"{key}: {rp[key]}")

    print("\n## Missing evidence examples")
    print(missing_examples(rows, memories, k_main, args.top_missing))

if __name__ == "__main__":
    main()
