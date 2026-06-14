import argparse
import json
import re
from pathlib import Path
from collections import Counter, defaultdict

STOP = {
    "the","a","an","and","or","of","to","in","on","for","with","from","as","is","are","was","were",
    "what","which","who","when","where","why","how","would","could","should","did","does","do",
    "has","have","had","be","been","being","likely","still","more","some","any","about","after",
    "before","during","their","her","his","she","he","they","them","it","this","that","them","his",
    "caroline","melanie","john","maria","joanna","nate","tim","evan","sam","jolene","deborah","andrew","audrey"
}

PERSON_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b")

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

def toks(s):
    return {
        t for t in re.findall(r"[a-zA-Z][a-zA-Z0-9_']+", str(s or "").lower())
        if len(t) >= 3 and t not in STOP
    }

def persons(s):
    return set(PERSON_RE.findall(str(s or "")))

def question_profile(q):
    ql = str(q or "").lower()
    return {
        "is_list": bool(re.search(r"\b(what are|what has|what have|what kind|what kinds|what types|activities|hobbies|books|authors|recipes|recommendations|events|ways|some)\b", ql)),
        "is_temporal": bool(re.search(r"\b(when|after|before|during|last|next|soon|date|year|time)\b", ql)),
        "is_inference": bool(re.search(r"\b(would|likely|might|considered|personality|leaning|religious|interested|advice|appropriate)\b", ql)),
    }

def memory_text(m):
    parts = [
        str(m.get("speaker") or ""),
        str(m.get("timestamp") or ""),
        str(m.get("session") or ""),
        str(m.get("dia_id") or ""),
        str(m.get("text") or ""),
    ]
    return " ".join(parts)

def score_item(q, q_tokens, q_persons, qprof, mem, rank, alpha, beta, gamma, delta):
    text = memory_text(mem)
    mtoks = toks(text)
    mpersons = persons(text)

    rank_score = 1.0 / (rank ** alpha)
    token_overlap = len(q_tokens & mtoks) / max(1, len(q_tokens))
    person_overlap = len(q_persons & mpersons)

    score = rank_score
    score += beta * token_overlap
    score += gamma * min(person_overlap, 2)

    lower = text.lower()

    if qprof["is_temporal"]:
        if re.search(r"\b(yesterday|today|tomorrow|last|next|\d{4}|january|february|march|april|may|june|july|august|september|october|november|december)\b", lower):
            score += delta

    if qprof["is_list"]:
        # Small boost for memories that look like event/activity/list evidence.
        if re.search(r"\b(read|watched|made|painted|hiked|camping|cooking|recipe|book|movie|played|visited|trip|class|workshop|running|biking|swimming|skiing|kayaking|pottery|dessert|recommend)\b", lower):
            score += delta

    if qprof["is_inference"]:
        if re.search(r"\b(want|hope|goal|care|support|help|important|feel|love|enjoy|stress|healthy|change|challenge|future|career|volunteer)\b", lower):
            score += delta

    return score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", required=True)
    ap.add_argument("--memories", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--input-k", type=int, default=200)
    ap.add_argument("--output-k", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.70)
    ap.add_argument("--beta", type=float, default=0.35)
    ap.add_argument("--gamma", type=float, default=0.08)
    ap.add_argument("--delta", type=float, default=0.06)
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

    changed = 0

    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            q = r.get("question") or ""
            q_tokens = toks(q)
            q_persons = persons(q)
            qprof = question_profile(q)

            ids = []
            seen = set()
            for x in r.get("retrieved_memory_ids") or []:
                try:
                    mid = int(x)
                except Exception:
                    continue
                if mid in seen:
                    continue
                seen.add(mid)
                ids.append(mid)

            candidates = ids[:args.input_k]
            scored = []

            for rank, mid in enumerate(candidates, 1):
                m = mem.get(mid, {})
                s = score_item(
                    q=q,
                    q_tokens=q_tokens,
                    q_persons=q_persons,
                    qprof=qprof,
                    mem=m,
                    rank=rank,
                    alpha=args.alpha,
                    beta=args.beta,
                    gamma=args.gamma,
                    delta=args.delta,
                )
                scored.append((s, rank, mid))

            scored.sort(key=lambda x: (-x[0], x[1], x[2]))
            new_ids = [mid for _, _, mid in scored[:args.output_k]]

            if new_ids != ids[:args.output_k]:
                changed += 1

            nr = dict(r)
            nr["rerank_source"] = "gold_blind_top200_feature_rerank"
            nr["rerank_input_k"] = args.input_k
            nr["rerank_output_k"] = args.output_k
            nr["retrieved_memory_ids_original_prefix"] = ids[:args.output_k]
            nr["retrieved_memory_ids"] = new_ids
            nr["retrieved_count"] = len(new_ids)

            f.write(json.dumps(nr, ensure_ascii=False) + "\n")

    print(f"wrote={out}")
    print(f"rows={len(rows)}")
    print(f"changed_top{args.output_k}_rows={changed}")

if __name__ == "__main__":
    main()
