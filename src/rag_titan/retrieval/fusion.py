"""
Reciprocal Rank Fusion (RRF): merge several ranked result lists into one
(Cormack, Clarke and Buettcher 2009; RAG handbook 2026, section 5.3).

    RRF(d) = sum over the lists r that found d:  1 / (k + rank_r(d))

Only ranks count, never raw scores, so lists from different searches (vector
and keyword, or one search per rewording of a question) can be merged. A chunk
found near the top of many lists wins.

Two options, both off by default (the plain RRF above):
    weights:  trust one list more. Its votes count weight / (k + rank), so with
              weight 2 a 3rd place counts as much as two 3rd places.
    keep_top: the #1 chunk of every list always makes the fused top_k, so a
              chunk that one search found first cannot be pushed out by chunks
              the other searches liked a little.

Used by hybrid search (vector + keyword lists) and by the RAG types that search
several times: HyDE (a made-up answer and the question) and RAG-Fusion
(several rewordings of the question), through search_and_fuse().
"""

from rag_titan.retrieval.dedupe import remove_duplicates
from rag_titan.retrieval.fusion_options import check_weights, keep_list_tops

# the constant of the original paper: a large k keeps one top rank from deciding everything
RRF_K = 60


def rrf_fuse(lists, k=RRF_K, weights=None):
    """
    Merge ranked lists with RRF.

    Args:
        lists: {name: [chunks, best first]}, e.g. {"dense": [...], "bm25": [...]}.
            Each chunk has "id" and "score" (0 to 1).
        k: the RRF constant.
        weights: {name: number above 0} to trust some lists more; None counts
            every list 1.0 (see check_weights()).

    Returns:
        A list of (chunk, score_trace) pairs, best fused score first. The trace
        holds, per list that found the chunk, its score and its rank
        ({"dense": 0.71, "dense_rank": 2, ...}), the list's weight when it is
        not 1 ("dense_weight": 2.0), plus the summed "rrf".
        Equal RRF scores are common (first and second place in either order),
        so they are broken by the 0-1 relevance summed over the lists, then by
        the order the chunks were first seen.
    """
    weights = check_weights(weights, lists)
    chunks, traces, relevance, first_seen = {}, {}, {}, {}
    for name, results in lists.items():
        for rank, result in enumerate(results, start=1):
            key = result["id"]
            first_seen.setdefault(key, len(first_seen))
            chunks.setdefault(key, _plain_chunk(result))
            trace = traces.setdefault(key, {"rrf": 0.0})
            trace[name] = result.get("score_trace", {}).get(name, result["score"])
            trace[f"{name}_rank"] = rank
            if weights[name] != 1.0:
                trace[f"{name}_weight"] = weights[name]
            trace["rrf"] += weights[name] / (k + rank)
            relevance[key] = relevance.get(key, 0.0) + result["score"]

    # rounding makes "equal" robust to float noise in the sums of 1 / (k + rank)
    order = sorted(chunks, key=lambda key: (-round(traces[key]["rrf"], 12), -relevance[key],
                                            first_seen[key]))
    return [(chunks[key], traces[key]) for key in order]


def fuse_ranked(lists, top_k, k=RRF_K, dedupe_overlap=0.5, weights=None, keep_top=False):
    """
    Fuse lists, drop near-copies, and keep the top_k, scored from 0 to 1.

    "score" is the RRF score divided by the best possible one (first in every
    list, with its weight), so a chunk that tops every list scores 1.0.
    Near-copies are removed after fusing: removed before, each list could keep
    a different copy of the same passage, and the passage would lose the votes
    of the other lists.

    Args:
        weights: see rrf_fuse().
        keep_top: True keeps every list's #1 chunk in the top_k (see keep_list_tops()).

    Returns:
        Chunk dicts with "score" and "score_trace" (see rrf_fuse), best first.
    """
    if not isinstance(keep_top, bool):
        raise ValueError(f"keep_top must be True or False, got {keep_top!r}")
    weights = check_weights(weights, lists)
    best_possible = sum(weights.values()) / (k + 1)
    matches = [{**chunk, "score": trace["rrf"] / best_possible, "score_trace": trace}
               for chunk, trace in rrf_fuse(lists, k, weights)]
    if dedupe_overlap is not None:
        matches = remove_duplicates(matches, dedupe_overlap)
    if keep_top:
        return keep_list_tops(matches, lists, top_k)
    return matches[:top_k]


def search_and_fuse(retriever, queries, top_k, trace, pool_size=None, k=RRF_K,
                    dedupe_overlap=0.5, weights=None, keep_top=False):
    """
    Search once per query, then fuse the result lists: the core of HyDE and RAG-Fusion.

    Every search is recorded in `trace` (a "retrieve" step, or the retriever's
    own steps, e.g. hybrid's), then one "fuse_queries" step merges them. Each
    result's score_trace shows its rank for every query ("query_1_rank": 2, ...).

    Args:
        retriever: any BaseRetriever.
        queries: the texts to search with, e.g. [question, made-up answer].
        top_k: how many fused chunks to return.
        trace: the Trace of the RAG run.
        pool_size: results per search before fusing; None uses max(20, 4 * top_k).
        k: the RRF constant (see fuse_ranked()).
        dedupe_overlap: drop near-copies after fusing (see fuse_ranked()).
        weights: None, or one number above 0 per query, in the same order:
            how much each query's search counts (see rrf_fuse()).
        keep_top: True keeps every search's #1 chunk (see fuse_ranked()).

    Returns:
        The top_k fused chunks. With one query: that search's own top_k, no fusing.
    """
    pairs = _queries_with_weights(queries, weights)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError(f"top_k must be a whole number of at least 1, got {top_k!r}")
    if not isinstance(keep_top, bool):
        raise ValueError(f"keep_top must be True or False, got {keep_top!r}")
    if len(pairs) == 1:
        return trace.search(retriever, query=pairs[0][0], top_k=top_k)

    names = [f"query_{number}" for number in range(1, len(pairs) + 1)]
    options = {}  # only the options in use, so a default run's trace is unchanged
    if weights is not None:
        # checked before searching, so a bad weight costs no search
        options["weights"] = check_weights(
            {name: weight for name, (_, weight) in zip(names, pairs)}, names)
    if keep_top:
        options["keep_top"] = True

    pool = pool_size or max(20, 4 * top_k)
    lists = {name: trace.search(retriever, query=query, top_k=pool)
             for name, (query, _) in zip(names, pairs)}
    return trace.record("fuse_queries", fuse_ranked, lists=lists, top_k=top_k, k=k,
                        dedupe_overlap=dedupe_overlap, **options)


def _queries_with_weights(queries, weights):
    """
    The (query, weight) pairs to search with: an empty query is dropped
    together with its weight. Without weights, every query counts 1.0.
    """
    if weights is not None and (not isinstance(weights, (list, tuple))
                                or len(weights) != len(queries)):
        raise ValueError(f"weights must be a list with one number per query, got {weights!r}")
    pairs = zip(queries, weights if weights is not None else [1.0] * len(queries))
    pairs = [(query, weight) for query, weight in pairs
             if isinstance(query, str) and query.strip()]
    if not pairs:
        raise ValueError("search_and_fuse needs at least one query with text")
    return pairs


def _plain_chunk(result):
    """The chunk without the scores one retriever added."""
    return {key: value for key, value in result.items()
            if key not in ("score", "retriever", "score_trace")}
