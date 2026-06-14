"""
Reciprocal Rank Fusion (RRF) for combining ranked lists.
k=60 is the standard constant from Cormack et al. 2009.
"""

from __future__ import annotations


def rrf_fuse(
    ranked_lists: list[list[str]],
    k: int = 60,
    top_n: int = 10,
) -> list[tuple[str, float]]:
    """
    Fuse multiple ranked doc_id lists with RRF.

    Args:
        ranked_lists: Each inner list is a ranked sequence of doc_ids
                      (index 0 = highest rank).
        k: RRF constant (default 60).
        top_n: How many fused results to return.

    Returns:
        List of (doc_id, rrf_score) sorted descending.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused[:top_n]