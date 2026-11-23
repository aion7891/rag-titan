"""
Hybrid retrieval: keyword (BM25) and vector search together, merged with
Reciprocal Rank Fusion (RAG handbook 2026, sections 3.03 and 5.3).

    RRF(d) = sum over the result lists r:  1 / (k + rank_r(d))

Only ranks are used, never the raw scores, so a BM25 score of 7 and a cosine
of 0.7 never have to be compared. A chunk found by both searches beats one
found by only one; k (60 in the original paper) keeps one top rank from
deciding everything.

dense_weight / bm25_weight trust one search more (both 1.0 by default), and
keep_top keeps each search's #1 chunk in the results (see retrieval/fusion.py).
"""

from rag_titan.rag_types.trace import Trace
from rag_titan.retrieval.base_retriever import BaseRetriever
from rag_titan.retrieval.fusion import fuse_ranked, rrf_fuse
from rag_titan.retrieval.fusion_options import check_weight


class HybridRetriever(BaseRetriever):
    """
    Runs a vector and a keyword retriever and fuses their rankings.

    Args:
        dense: a DenseRetriever (meaning), made with dedupe_overlap=None.
        keyword: a BM25Retriever (exact words) on the same store, also with
            dedupe_overlap=None.
        k: the RRF constant; larger makes the lists' top ranks matter less.
        pool_size: how many results each search returns before fusing;
            None uses max(20, 4 * top_k).
        dedupe_overlap: drop a fused result that overlaps a better one from
            the same file by at least this share; None keeps all. Near-copies
            are removed here, after fusing: removed earlier, each search could
            keep a different copy of the same passage, and the passage would
            lose the "found by both" vote that makes hybrid work.
        dense_weight: how much the vector search's ranking counts (above 0;
            default 1.0). 2.0 makes its votes count twice.
        bm25_weight: the same for the keyword search.
        keep_top: True keeps the #1 chunk of each search in the results,
            even when the other search did not find it.
    """

    name = "hybrid"

    # its steps in the flow diagram, in the order they run (see rag_types/flow.py)
    SEARCH_FLOW = {
        "nodes": [
            {"id": "vector_search", "label": "Vector search", "kind": "search"},
            {"id": "keyword_search", "label": "Keyword search", "kind": "search"},
            {"id": "fuse", "label": "Fuse ranks (RRF)", "kind": "search"},
        ],
        "edges": [["vector_search", "keyword_search"], ["keyword_search", "fuse"]],
    }

    def __init__(self, dense, keyword, k=60, pool_size=None, dedupe_overlap=0.5,
                 dense_weight=1.0, bm25_weight=1.0, keep_top=False):
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError(f"k must be a whole number of at least 1, got {k!r}")
        if pool_size is not None:
            self._check_top_k(pool_size)
        if dedupe_overlap is not None and not 0 < dedupe_overlap <= 1:
            raise ValueError("dedupe_overlap must be more than 0 and at most 1, or None")
        if not isinstance(keep_top, bool):
            raise ValueError(f"keep_top must be True or False, got {keep_top!r}")
        self.weights = {"dense": check_weight(dense_weight, "dense_weight"),
                        "bm25": check_weight(bm25_weight, "bm25_weight")}
        self.keep_top = keep_top
        self.dense, self.keyword = dense, keyword
        self.k, self.pool_size, self.dedupe_overlap = k, pool_size, dedupe_overlap

    def has_documents(self):
        """Return True when there is at least one chunk to search."""
        return self.dense.has_documents()

    def retrieve(self, query, top_k=5):
        """
        Return the top_k chunks with the highest fused score, best first.

        "score" is the RRF score divided by the best possible one (first in
        both lists), so it runs from 0 to 1. "score_trace" shows each part:
        {"dense": 0.71, "dense_rank": 2, "bm25": 6.3, "bm25_rank": 1, "rrf": 0.0325}
        (a search that did not find the chunk has no entry).
        """
        return self.retrieve_with_steps(query, top_k)[0]

    def retrieve_with_steps(self, query, top_k=5):
        """
        Like retrieve(), and also return the steps taken (vector_search,
        keyword_search, fuse), so a RAG type's trace and diagram can show them.

        Returns:
            (chunks, steps): the chunks of retrieve() and a list of step dicts.
        """
        self._check_top_k(top_k)
        if not self.has_documents():
            return [], []

        trace, pool = Trace(), self.pool_size or max(20, 4 * top_k)
        dense = trace.record("vector_search", self.dense.retrieve, query=query, top_k=pool)
        keyword = trace.record("keyword_search", self.keyword.retrieve, query=query, top_k=pool)
        matches = trace.record("fuse", self.fuse_top, dense=dense, keyword=keyword, top_k=top_k)
        return matches, trace.steps

    def fuse_top(self, dense, keyword, top_k):
        """Fuse the two result lists, drop near-copies, return the top_k scored from 0 to 1."""
        matches = fuse_ranked({"dense": dense, "bm25": keyword}, top_k, self.k,
                              self.dedupe_overlap, self.weights, self.keep_top)
        return self._with_trace(matches, [match.pop("score_trace") for match in matches])

    def fuse(self, lists):
        """
        Merge ranked lists with RRF (see retrieval/fusion.py).

        Args:
            lists: {"dense": [chunks, best first], "bm25": [...]}.

        Returns:
            A list of (chunk, score_trace) pairs, best fused score first.
        """
        return rrf_fuse(lists, self.k, self.weights)
