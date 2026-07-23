"""
Reciprocal Rank Fusion (RRF): combines a dense (vector) ranked list and
a sparse (BM25) ranked list into a single fused ranking.

WHY RRF SPECIFICALLY, OVER A WEIGHTED SCORE COMBINATION:
Cosine similarity scores (roughly 0-1, densely clustered near the top
for a good embedding model) and BM25 scores (unbounded, highly
corpus-dependent, can be anywhere from 0 to 20+) live on completely
different, incomparable scales. Averaging or weighting them directly
("0.5 * cosine_score + 0.5 * bm25_score") requires manually tuning a
weight and normalization scheme per corpus, and that tuning silently
breaks the moment corpus size or content shifts BM25's score
distribution. RRF sidesteps the entire problem: it only looks at each
result's *rank position* in each list, not the raw score, so it needs
zero tuning and zero renormalization to combine two arbitrarily
different scoring systems.

THE FORMULA:
    RRF_score(doc) = sum over each list L containing doc of  weight_L / (k + rank_L(doc))

  - rank_L(doc) is the doc's 1-indexed position in list L (1 = top result).
  - k is a constant (60 is the standard default from the original RRF
    paper) that dampens the impact of very low ranks -- without it,
    the #1 vs #2 gap would dominate fused scores disproportionately.
  - A document appearing near the top of *either* list scores well; a
    document appearing near the top of *both* scores best of all --
    which is exactly the hybrid-retrieval property we want.

A REAL LIMITATION, FOUND BY TESTING THIS MODULE, WORTH KNOWING:
plain RRF is rank-only and completely blind to match *magnitude*. A
chunk that is BM25's #1 result with an overwhelming, unambiguous
keyword match, and a chunk that is the dense retriever's #1 result with
only a mediocre semantic similarity, score *identically* if each
appears only in its own list -- both get exactly weight/(k+1). RRF has
no way to express "sparse is very confident about this one." If you
need one retriever's signal to dominate when it disagrees with the
other (e.g. you know sparse is more trustworthy for exact-identifier
queries), that requires the optional per-list `weights` below --
plain unweighted RRF alone cannot express it.
"""

from __future__ import annotations

from app.config import settings


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int | None = None,
    top_k: int = 10,
    weights: list[float] | None = None,
) -> list[dict]:
    """Fuse any number of ranked result lists (each a list of dicts with a
    'chunk_id' key, ordered best-first) into one fused ranking.

    weights, if given, must have one entry per list and scales that
    list's contribution -- e.g. weights=[1.0, 1.5] to trust the second
    (sparse) list 50% more than the first (dense) when they disagree.
    Defaults to equal weighting, which is standard/unweighted RRF.

    Returns fused results sorted by descending RRF score, each dict
    carrying the original payload (from whichever list it was first
    seen in) plus an 'rrf_score' field.
    """
    k = k if k is not None else settings.rrf_k
    weights = weights or [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights must have one entry per ranked list")

    fused_scores: dict[str, float] = {}
    payload_by_id: dict[str, dict] = {}

    for ranked_list, weight in zip(ranked_lists, weights):
        for rank, item in enumerate(ranked_list, start=1):
            chunk_id = item["chunk_id"]
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + weight / (k + rank)
            payload_by_id.setdefault(chunk_id, item)

    fused = [
        {**payload_by_id[chunk_id], "rrf_score": score}
        for chunk_id, score in fused_scores.items()
    ]
    fused.sort(key=lambda r: r["rrf_score"], reverse=True)
    return fused[:top_k]