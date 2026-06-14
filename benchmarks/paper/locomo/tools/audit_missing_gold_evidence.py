import argparse
import json
import re
from pathlib import Path
from collections import Counter, defaultdict

STOP = {
    "the","a","an","and","or","of","to","in","on","for","with","from","as","is","are","was","were",
    "what","which","who","when","where","why","how","would","could","should","did","does","do",
    "has","have","had","be","been","being","likely","still","more","some","any","about","after",
    "before","during","their","her","his","she","he","they","them","it","this","that"
}

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
                print(f"ERROR: bad JSON {path} line {line_no}: {e}")
                raise SystemExit
    return rows

def ints(xs):
    out = []
    for x in xs or []:
        try:
            out.append(int(x))
        except Exception:
            pass
    return out

def toks(s):
    return {
        t for t in re.findall(r"[a-zA-Z][a-zA-Z0-9_']+", str(s or "").lower())
        if len(t) >= 3 and t not in STOP
    }

def tag_failure(question, missing_text, gold_count):
    q = str(question or "").lower()
    mt = str(missing_text or "").lower()
    tags = []

    if gold_count >= 3:
        tags.append("multi_evidence")

    if re.search(r"\b(activities|what has|what have|what are some|what types|books|painted|instruments|events|ways)\b", q):
        tags.append("list_or_aggregation")

    if "take a look" in mt or "here" in mt or "this" in mt:
        tags.append("deictic_or_multimodal_weak_text")

    if re.search(r"\b(would|likely|considered|might|personality|leaning|religious|interested)\b", q):
        tags.append("inference_or_preference")

    if re.search(r"\b(when|before|after|soon|last|year|date|time)\b", q):
        tags.append("temporal")

    if not tags:
        tags.append("direct_recall_or_other")

    return tags

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", required=True)
    ap.add_argument("--memories", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=100)
    args = ap.parse_args()

    rows = read_jsonl(args.retrieval)
    memories = read_jsonl(args.memories)

    mem = {}
    for m in memories:
        try:
            mem[int(m["memory_id"])] = m
        except Exception:
            pass

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    summary_tags = Counter()
    summary_cat = Counter()
    summary_sample = Counter()
    summary_gold_count = Counter()
    missing_rows = 0
    missing_gold_total = 0

    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            gold = set(ints(r.get("evidence_memory_ids")))
            if not gold:
                continue

            retrieved = ints((r.get("retrieved_memory_ids") or [])[:args.k])
            ret = set(retrieved)

            missing = sorted(gold - ret)
            if not missing:
                continue

            missing_rows += 1
            missing_gold_total += len(missing)

            q = r.get("question")
            q_tokens = toks(q)
            top20 = retrieved[:20]

            top20_speakers = set()
            top20_sessions = set()
            top20_tokens = set()

            for mid in top20:
                m = mem.get(mid)
                if not m:
                    continue
                top20_speakers.add(m.get("speaker"))
                top20_sessions.add(m.get("session"))
                top20_tokens |= toks(m.get("text"))

            missing_items = []
            all_tags = []

            for mid in missing:
                m = mem.get(mid, {})
                mt = m.get("text", "")
                mtoks = toks(mt)
                overlap_q = len(q_tokens & mtoks)
                overlap_top20 = len(top20_tokens & mtoks)

                tags = tag_failure(q, mt, len(gold))
                all_tags.extend(tags)

                item = {
                    "memory_id": mid,
                    "speaker": m.get("speaker"),
                    "session": m.get("session"),
                    "dia_id": m.get("dia_id"),
                    "timestamp": m.get("timestamp"),
                    "text": mt,
                    "question_token_overlap": overlap_q,
                    "top20_token_overlap": overlap_top20,
                    "same_speaker_in_top20": m.get("speaker") in top20_speakers,
                    "same_session_in_top20": m.get("session") in top20_sessions,
                    "tags": tags,
                }
                missing_items.append(item)

            for t in all_tags:
                summary_tags[t] += 1

            summary_cat[r.get("category", "unknown")] += len(missing)
            summary_sample[r.get("sample_id", "unknown")] += len(missing)
            summary_gold_count[str(len(gold))] += len(missing)

            record = {
                "sample_id": r.get("sample_id"),
                "question_id": r.get("question_id"),
                "category": r.get("category"),
                "question": q,
                "gold_answer": r.get("gold_answer"),
                "gold_evidence_ids": sorted(gold),
                "retrieved_top20": top20,
                "missing_gold_ids": missing,
                "gold_count": len(gold),
                "missing_count": len(missing),
                "missing_items": missing_items,
                "row_tags": sorted(set(all_tags)),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"wrote={out}")
    print(f"missing_rows={missing_rows}")
    print(f"missing_gold_total={missing_gold_total}")

    print("\nMissing gold by category:")
    for k, v in summary_cat.most_common():
        print(f"{k:12s} {v}")

    print("\nMissing gold by sample:")
    for k, v in summary_sample.most_common():
        print(f"{k:12s} {v}")

    print("\nMissing gold by evidence count:")
    for k, v in sorted(summary_gold_count.items(), key=lambda x: int(x[0])):
        print(f"{k:>3s} {v}")

    print("\nHeuristic failure tags:")
    for k, v in summary_tags.most_common():
        print(f"{k:32s} {v}")

if __name__ == "__main__":
    main()
