import argparse
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def call_chat(base_url, model, prompt, max_tokens, temperature, timeout):
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You answer memory QA questions using only provided evidence. Return only the final answer."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    latency_ms = (time.perf_counter() - started) * 1000.0

    obj = json.loads(raw)
    answer = obj["choices"][0]["message"]["content"].strip()
    return answer, latency_ms

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="llama-3.3-70b-q4km")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    rows = read_jsonl(args.packets)
    if args.limit is not None:
        rows = rows[:args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"packets={len(rows)}")
    print(f"base_url={args.base_url}")
    print(f"model={args.model}")

    with out.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows, 1):
            try:
                pred, latency_ms = call_chat(
                    base_url=args.base_url,
                    model=args.model,
                    prompt=row["answer_packet"],
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    timeout=args.timeout,
                )
                ok = True
                err = None
            except Exception as e:
                pred = ""
                latency_ms = None
                ok = False
                err = str(e)

            result = dict(row)
            result["prediction"] = pred
            result["answer_latency_ms"] = latency_ms
            result["answer_ok"] = ok
            result["answer_error"] = err

            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(rows)}] ok={ok} pred={pred[:120]!r}")

    print(f"wrote={out}")

if __name__ == "__main__":
    main()
