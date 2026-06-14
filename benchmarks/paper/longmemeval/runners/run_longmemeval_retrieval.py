#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

COMMON = Path(__file__).resolve().parents[2] / "common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from memory_retrieval import main  # type: ignore


if __name__ == "__main__":
    main(
        dataset_name="longmemeval",
        default_cache="benchmarks/paper/longmemeval/data/embedding_cache.jsonl",
    )
