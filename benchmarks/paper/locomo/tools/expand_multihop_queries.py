#!/usr/bin/env python3
"""
LLM-based query expansion for multi-hop LocoMo questions.
Generates targeted search queries designed to recover evidence
that standard BM25+vector retrieval misses (the 37 never-retrieved IDs).

Usage:
  uv run python benchmarks/paper/locomo/tools/expand_multihop_queries.py \\
    --questions benchmarks/paper/locomo/data/locomo_questions.jsonl \\
    --out /tmp/expanded_questions.jsonl
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def call_nvidia_llm(prompt: str, api_key: str, model: str = "meta/llama-3.1-8b-instruct") -> Optional[str]:
    """Call NVIDIA LLM API via OpenAI-compatible endpoint."""
    from openai import OpenAI
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=512,
            top_p=0.9,
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"  [warn] LLM call failed: {e}", file=sys.stderr)
        return None


GENERATION_PROMPT = """You are helping a retrieval system find evidence for complex multi-hop questions from a conversation transcript.

The question is: "{question}"

This is a MULTI-HOP question — the answer requires connecting information from DIFFERENT parts of the conversation. The evidence is likely in a different session or turn than where the topic is first raised.

The evidence for this question may NOT contain the exact words from the question. It may use different vocabulary, imply things indirectly, or be about a related topic.

Generate exactly 5 diverse search queries that would help find the evidence.
Each query should:
1. Focus on different entities or aspects mentioned in the question
2. Use vocabulary that might appear in the actual evidence (not just the question words)
3. Include related concepts, synonyms, and contextual clues
4. Consider what the evidence MENTIONS vs what the question ASKS

Return ONLY a JSON array of 5 strings, no explanation:
["query 1", "query 2", "query 3", "query 4", "query 5"]"""


HYDE_PROMPT = """You are helping a retrieval system find evidence for complex multi-hop questions.

The question is: "{question}"

Generate a paragraph of text that WOULD be the ideal evidence passage to answer this question.
This should NOT be the answer itself, but rather a plausible conversation excerpt that contains
the information needed. Use natural conversational language with speaker turns.

Return ONLY the paragraph, no explanation."""


def expand_question(question: str, api_key: str) -> List[str]:
    """Generate query expansions for a multi-hop question."""
    prompt = GENERATION_PROMPT.format(question=question)
    result = call_nvidia_llm(prompt, api_key)
    
    if not result:
        return [question]
    
    # Try to parse JSON array
    result = result.strip()
    if result.startswith("```"):
        result = re.sub(r"^```(?:json)?\s*", "", result)
        result = re.sub(r"\s*```$", "", result)
    
    try:
        queries = json.loads(result)
        if isinstance(queries, list) and len(queries) >= 2:
            return queries[:5]
    except json.JSONDecodeError:
        pass
    
    # Try line-by-line fallback
    lines = [l.strip().strip('"').strip("'") for l in result.split("\n") if l.strip()]
    queries = [l for l in lines if l.startswith(("query", "Query", "- ", "* ")) or (len(l) > 10 and not l.startswith("["))]
    if len(queries) >= 2:
        return queries[:5]
    
    return [question]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="Limit to N questions for testing")
    ap.add_argument("--sleep", type=float, default=1.0, help="Seconds between API calls")
    args = ap.parse_args()
    
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("ERROR: NVIDIA_API_KEY env var required", file=sys.stderr)
        raise SystemExit(1)
    
    questions = read_jsonl(args.questions)
    
    # Only expand multi-hop questions
    mh_questions = [q for q in questions if q.get("category") == "multi_hop"]
    print(f"Total questions: {len(questions)}")
    print(f"Multi-hop questions to expand: {len(mh_questions)}")
    
    if args.limit:
        mh_questions = mh_questions[:args.limit]
    
    expanded = 0
    for i, q in enumerate(mh_questions):
        qid = q.get("question_id", f"q{i}")
        question = q.get("question", "")
        
        if not question:
            print(f"  [{i+1}/{len(mh_questions)}] {qid}: SKIP (empty question)")
            continue
        
        print(f"  [{i+1}/{len(mh_questions)}] {qid}: {question[:60]}...")
        expansions = expand_question(question, api_key)
        
        # Always include original question as first variant for stability
        all_variants = [question] + [e for e in expansions if e.lower() != question.lower()]
        q["llm_query_variants"] = all_variants[:6]  # cap at 6 total
        expanded += 1
        
        print(f"    → {len(expansions)} expansions: {expansions}")
        
        if i < len(mh_questions) - 1:
            time.sleep(args.sleep)
    
    write_jsonl(args.out, questions)
    print(f"\nDone. Expanded {expanded} multi-hop questions. Output: {args.out}")
    print(f"\nNow run with these expanded queries using a custom query mode.")


if __name__ == "__main__":
    main()
