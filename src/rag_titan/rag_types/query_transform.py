"""
Shared base of the RAG types that change *what is searched for* (HyDE, RAG-Fusion):

    Question -> make search queries (one LLM call) -> search each one
             -> fuse the lists (RRF) -> Rerank -> Generate

A subclass only says how it makes its queries (search_queries()); the search,
the fusing, the reranking and the answer are the same for all of them. The
reranker and the answer always use the original question.

question_weight makes the question's own search count more (or less) than the
made-up queries' searches when the lists are fused; keep_top keeps every
search's #1 chunk (see retrieval/fusion.py).
"""

from rag_titan.rag_types.base_rag import BaseRAG, check_search_sizes
from rag_titan.rag_types.trace import Trace
from rag_titan.retrieval.fusion import search_and_fuse
from rag_titan.retrieval.fusion_options import check_weight


class QueryTransformRAG(BaseRAG):
    """
    Searches with several queries made from the question, then answers once.

    Args:
        llm: any BaseLLM; makes the queries.
        retriever: any BaseRetriever (dense, bm25 or hybrid).
        reranker: a BaseReranker; it ranks against the original question.
        generator: a BaseGenerator.
        top_k: how many chunks the fused search returns.
        top_n: how many chunks the model reads.
        question_weight: how much the question's own search counts when the
            lists are fused (above 0; default 1.0, the same as every query).
        keep_top: True keeps the #1 chunk of every search in the fused results.
    """

    DEFAULT_TOP_K, DEFAULT_TOP_N = 5, 2
    # QUESTION_WEIGHT and RRF_KEEP_TOP
    DEFAULT_QUESTION_WEIGHT, DEFAULT_KEEP_TOP = 1.0, False

    def __init__(self, llm, retriever, reranker, generator, top_k=DEFAULT_TOP_K,
                 top_n=DEFAULT_TOP_N, question_weight=DEFAULT_QUESTION_WEIGHT,
                 keep_top=DEFAULT_KEEP_TOP):
        check_search_sizes(top_k, top_n)
        if not isinstance(keep_top, bool):
            raise ValueError(f"keep_top must be True or False, got {keep_top!r}")
        self.question_weight = check_weight(question_weight, "question_weight")
        self.keep_top = keep_top
        self.llm = llm
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.top_k = top_k
        self.top_n = top_n

    def run(self, query, trace=None):
        """Answer the question from the fused searches, recording each step."""
        query = query.strip()
        trace = trace if trace is not None else Trace()

        if not query:
            return trace.result("Please enter a question.")

        if not self.retriever.has_documents():
            return trace.result("Please index your documents first.")

        queries = self.search_queries(query, trace)
        matches = search_and_fuse(self.retriever, queries, self.top_k, trace,
                                  weights=self.query_weights(query, queries),
                                  keep_top=self.keep_top)
        best = trace.record("rerank", self.reranker.rerank,
                            query=query, matches=matches, top_n=self.top_n)

        answer = trace.record("generate", self.generator.generate, query=query, chunks=best)
        return trace.result(answer, best)

    def search_queries(self, query, trace):
        """
        Return the texts to search with, recording the LLM step that made them.

        Never empty: when the model gives nothing usable, return [query].
        """
        raise NotImplementedError

    def query_weights(self, question, queries):
        """
        One weight per query: question_weight for the question, 1.0 for the others.

        None when question_weight is 1.0, so a default run fuses (and records)
        exactly as before.
        """
        if self.question_weight == 1.0:
            return None
        return [self.question_weight if text == question else 1.0 for text in queries]
