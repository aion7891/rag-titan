"""
RAG-Fusion, inspired by Raudaschl (2023) and the RAG handbook 2026 (Multi-Query
RAG and RAG-Fusion, items 012 and 013):

    Question -> Write N search queries (one LLM call) -> Search the question and
             each query -> Fuse the lists (RRF) -> Rerank -> Generate

One wording of a question finds what matches that wording; other wordings
("notebook" next to "laptop", or one part of a two-part question) find chunks it
misses. Reciprocal Rank Fusion then puts the chunks found by several queries on
top. The reranker and the answer use the original question.
"""

import re

from rag_titan.prompts import MULTI_QUERY_PROMPT
from rag_titan.rag_types.base_rag import check_count
from rag_titan.rag_types.query_transform import QueryTransformRAG
from rag_titan.utils.logger import get_logger

logger = get_logger(__name__)

# most rewordings one question may ask for: each one is a search
MAX_QUERIES = 10

# "1. ", "2) ", "- ", "* ", "• ", "Query 3:" at the start of a line
LINE_MARK = re.compile(r"^\s*(?:[-*•]+|\(?\d+[.):]|query\s*\d+\s*[:.)-])\s*", re.IGNORECASE)


class RAGFusionRAG(QueryTransformRAG):
    """
    Searches the question and several rewordings of it, then fuses the results.

    Args:
        llm: any BaseLLM; writes the rewordings.
        retriever: any BaseRetriever (dense, bm25 or hybrid).
        reranker: a BaseReranker; it ranks against the original question.
        generator: a BaseGenerator.
        top_k: how many chunks the fused search returns.
        top_n: how many chunks the model reads.
        queries: how many rewordings to ask for, 1 to 10 (default 3), so
            queries + 1 searches with the question.
        question_weight: how much the question's search counts against each
            rewording's when fusing (default 1.0; 2.0 trusts the question more).
        keep_top: True keeps the #1 chunk of every search (default False).
    """

    # rewordings searched next to the question (FUSION_QUERIES)
    DEFAULT_QUERIES = 3

    FLOW = {
        "nodes": [
            {"id": "write_queries", "label": "Write search queries", "kind": "llm"},
            {"id": "retrieve", "label": "Search", "kind": "search"},
            {"id": "fuse_queries", "label": "Fuse lists (RRF)", "kind": "search"},
            {"id": "rerank", "label": "Rerank", "kind": "search"},
            {"id": "generate", "label": "Generate answer", "kind": "output"},
        ],
        # retrieve -> retrieve: one search per query; with no usable rewording,
        # the question's single search goes straight to rerank
        "edges": [["write_queries", "retrieve"], ["retrieve", "retrieve"],
                  ["retrieve", "fuse_queries"], ["fuse_queries", "rerank"],
                  ["retrieve", "rerank"], ["rerank", "generate"]],
    }

    def __init__(self, llm, retriever, reranker, generator,
                 top_k=QueryTransformRAG.DEFAULT_TOP_K, top_n=QueryTransformRAG.DEFAULT_TOP_N,
                 queries=DEFAULT_QUERIES,
                 question_weight=QueryTransformRAG.DEFAULT_QUESTION_WEIGHT,
                 keep_top=QueryTransformRAG.DEFAULT_KEEP_TOP):
        super().__init__(llm, retriever, reranker, generator, top_k, top_n, question_weight,
                         keep_top)
        check_count("queries", queries)
        if queries > MAX_QUERIES:
            raise ValueError(f"queries must be at most {MAX_QUERIES}: each one is a search, "
                             f"got {queries}")
        self.queries = queries

    def search_queries(self, query, trace):
        """The question first, then the rewordings the model wrote."""
        variants = trace.record("write_queries", self.write_queries, query=query)
        if not variants:
            logger.warning("RAG-Fusion: no usable query in the model's reply, "
                           "searching with the question only")
        return [query, *variants]

    def write_queries(self, query):
        """Ask the model for `queries` rewordings; return the usable ones (maybe none)."""
        reply = self.llm.ask(MULTI_QUERY_PROMPT,
                             f"Number of queries: {self.queries}\n\nQuestion: {query}")
        return parse_queries(reply, query, self.queries)


def parse_queries(reply, question, count):
    """
    Read the search queries in the model's reply.

    Small models number their lines, add bullets, quotes or an intro line
    ("Here are 3 queries:"), or repeat the question. This keeps one query per
    line without those marks, drops intro lines, empty lines, duplicates and
    copies of the question, and returns at most `count` queries.

    Args:
        reply: the model's text.
        question: the original question (searched anyway, so never repeated).
        count: most queries to return.
    """
    seen = {_plain(question)}
    queries = []
    for line in (reply or "").splitlines():
        text = LINE_MARK.sub("", line).strip().strip("\"'`").strip()
        if not text or text.endswith(":") or _plain(text) in seen:
            continue
        seen.add(_plain(text))
        queries.append(text)
    return queries[:count]


def _plain(text):
    """Lowercase letters and digits only, to spot the same query written twice."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
