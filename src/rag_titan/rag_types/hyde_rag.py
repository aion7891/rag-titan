"""
HyDE RAG, inspired by "Precise Zero-Shot Dense Retrieval without Relevance
Labels" (Gao et al. 2022; RAG handbook 2026, section 08):

    Question -> Write a hypothetical answer -> Search with it (and with the question)
             -> Fuse the lists (RRF) -> Rerank -> Generate

A question and its answer are worded differently ("How many days off?" vs
"Employees receive 20 paid leave days"). The model first writes a short made-up
answer passage; searching with it finds chunks that read like the answer. The
made-up text is only used to search: the reranker and the answer always use the
original question, and only real chunks reach the model.

The handbook's warning, "hallucinated hypotheticals can mislead", is why
with_question is on by default: the question is searched too and the two lists
are fused, so a wrong guess cannot take over the results alone.

Simplified from the paper: one made-up passage instead of several, and the two
searches are fused by rank (RRF) instead of averaging their embeddings.
"""

from rag_titan.prompts import HYDE_PROMPT
from rag_titan.rag_types.query_transform import QueryTransformRAG
from rag_titan.utils.logger import get_logger

logger = get_logger(__name__)


class HyDERAG(QueryTransformRAG):
    """
    Searches with a made-up answer passage, then answers from the real chunks.

    Args:
        llm: any BaseLLM; writes the made-up passage.
        retriever: any BaseRetriever (dense, bm25 or hybrid).
        reranker: a BaseReranker; it ranks against the original question.
        generator: a BaseGenerator.
        top_k: how many chunks the (fused) search returns.
        top_n: how many chunks the model reads.
        with_question: also search with the question and fuse both lists
            (default True); False searches with the made-up passage only.
        question_weight: how much the question's search counts against the
            passage's when fusing (default 1.0; 2.0 trusts the question more).
        keep_top: True keeps the #1 chunk of both searches (default False).
    """

    # also search with the question (HYDE_WITH_QUESTION)
    DEFAULT_WITH_QUESTION = True

    FLOW = {
        "nodes": [
            {"id": "hypothesize", "label": "Write hypothetical answer", "kind": "llm"},
            {"id": "retrieve", "label": "Search", "kind": "search"},
            {"id": "fuse_queries", "label": "Fuse lists (RRF)", "kind": "search"},
            {"id": "rerank", "label": "Rerank", "kind": "search"},
            {"id": "generate", "label": "Generate answer", "kind": "output"},
        ],
        # retrieve -> retrieve: the second search (the question); without it,
        # a single search goes straight to rerank
        "edges": [["hypothesize", "retrieve"], ["retrieve", "retrieve"],
                  ["retrieve", "fuse_queries"], ["fuse_queries", "rerank"],
                  ["retrieve", "rerank"], ["rerank", "generate"]],
    }

    def __init__(self, llm, retriever, reranker, generator,
                 top_k=QueryTransformRAG.DEFAULT_TOP_K, top_n=QueryTransformRAG.DEFAULT_TOP_N,
                 with_question=DEFAULT_WITH_QUESTION,
                 question_weight=QueryTransformRAG.DEFAULT_QUESTION_WEIGHT,
                 keep_top=QueryTransformRAG.DEFAULT_KEEP_TOP):
        super().__init__(llm, retriever, reranker, generator, top_k, top_n, question_weight,
                         keep_top)
        if not isinstance(with_question, bool):
            raise ValueError(f"with_question must be True or False, got {with_question!r}")
        self.with_question = with_question

    def search_queries(self, query, trace):
        """
        Write the passage, then search with it first and the question second.

        An empty passage (the model gave nothing) falls back to the question
        alone, so the run still finds real chunks.
        """
        passage = trace.record("hypothesize", self.hypothesize, query=query)
        if not passage:
            logger.warning("HyDE: the model wrote no passage, searching with the question")
            return [query]
        return [passage, query] if self.with_question else [passage]

    def hypothesize(self, query):
        """Ask the model for a short made-up passage that would answer the question."""
        return (self.llm.ask(HYDE_PROMPT, query) or "").strip()
