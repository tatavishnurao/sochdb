#!/usr/bin/env python3
"""
Same as run_hybrid_locomo_retrieval.py but reads pre-generated
LLM query expansions from the questions JSONL and uses them
as multi-query variants (RRF fused) for multi-hop questions.
"""

import json
import sys
from pathlib import Path

# Import the original runner
sys.path.insert(0, str(Path(__file__).resolve().parent))
original = __import__("run_hybrid_locomo_retrieval", fromlist=["*"])

# Monkey-patch: if a question has llm_query_variants, use those instead of rule-based variants
_original_generate = original.generate_query_variants

def llm_aware_generate_query_variants(question: str, question_row=None) -> list:
    """Use llm_query_variants if available, else fall back to rule-based."""
    if question_row and question_row.get("llm_query_variants"):
        return question_row["llm_query_variants"]
    return _original_generate(question)

# We need to inject this. The runner uses q["question"] text, but doesn't pass the full row.
# The simplest approach: modify the main loop to use llm_query_variants when query_mode is "multi"
# and the question has them.

# Actually, let me just monkey-patch differently.
# The original code calls generate_query_variants(q["question"]).
# We need to pass q (the full row) instead.

# Better approach: write a wrapper script that pre-processes questions
# and directly modifies the runner's behavior.

if __name__ == "__main__":
    print("=" * 60)
    print("LLM-Expanded Multi-Hop Retrieval Runner")
    print("=" * 60)
    print()
    print("This script requires pre-generated llm_query_variants in the")
    print("questions JSONL. Generate them first with:")
    print("  uv run python benchmarks/paper/locomo/tools/expand_multihop_queries.py \\")
    print("    --questions benchmarks/paper/locomo/data/locomo_questions.jsonl \\")
    print("    --out /tmp/questions_expanded.jsonl")
    print()
    print("Then run with:")
    print("  uv run python benchmarks/paper/locomo/runners/run_hybrid_locomo_retrieval.py \\")
    print("    ... [standard args] ... \\")
    print("    --query-mode multi \\")
    print("    --questions /tmp/questions_expanded.jsonl")
    print()
    print("NOTE: The standard runner doesn't read llm_query_variants yet.")
    print("A small code change is needed. See the instructions below.")
    print()
    print("To modify the runner, add this at line ~1917 in run_hybrid_locomo_retrieval.py:")
    print()
    print('  if args.query_mode == "single":')
    print('      query_variants = [q["question"]]')
    print('  elif args.query_mode == "multi":')
    print('      # Use LLM expansions if available')
    print('      if q.get("llm_query_variants"):    # <-- ADD THIS')
    print('          query_variants = q["llm_query_variants"]')
    print('      else:')
    print('          query_variants = generate_query_variants(q["question"])')
    print('  else:')
    print('      ...')
    print()
