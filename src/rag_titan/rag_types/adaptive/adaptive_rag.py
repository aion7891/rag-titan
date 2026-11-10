"""
Adaptive RAG:

Question -> Choose route -> Retrieve -> Rerank -> Check evidence
                                                  |
                                  Enough evidence -> Answer
                                                  |
                              Not enough -> Rewrite and retry
"""

from rag_titan.prompts import DIRECT_REPLY_PROMPT, NO_ANSWER
from rag_titan.rag_types.adaptive.evidence_checker import EvidenceChecker
from rag_titan.rag_types.adaptive.query_rewriter import QueryRewriter
from rag_titan.rag_types.adaptive.router import Router
from rag_titan.utils.logger import get_logger

logger = get_logger(__name__)

# how many chunks to retrieve (top_k) and keep (top_n) for each route
SEARCH_SIZES = {
    "simple": {"top_k": 4, "top_n": 2},
    "complex": {"top_k": 8, "top_n": 4},
}


class AdaptiveRAG:
    """
    A RAG that changes its effort based on the question.

    Greetings get a direct reply, simple questions a small search,
    complex questions a bigger search. When the evidence is weak,
    the question is rewritten and the search is tried again.

    Args:
        llm: any BaseLLM, used for routing, checking and rewriting.
        retriever: any BaseRetriever, for example a DenseRetriever.
        reranker: a BaseReranker.
        generator: a BaseGenerator.
        max_attempts: how many searches to try before giving up.
    """

    def __init__(self, llm, retriever, reranker, generator, max_attempts=2):
        self.llm = llm
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.max_attempts = max_attempts

        self.router = Router(llm)
        self.evidence_checker = EvidenceChecker(llm)
        self.query_rewriter = QueryRewriter(llm)

    def answer(self, query):
        """Return the answer to the question, using the indexed documents."""
        query = query.strip()

        if not query:
            return "Please enter a question."

        route = self.router.route(query)
        logger.info("Selected route: %s", route)

        if route == "direct":
            return self.llm.ask(DIRECT_REPLY_PROMPT, query)

        if not self.retriever.has_documents():
            return "Please index your documents first."

        sizes = SEARCH_SIZES[route]
        return self._search_and_answer(query, sizes["top_k"], sizes["top_n"])

    def _search_and_answer(self, query, top_k, top_n):
        """Search, check the evidence, and retry with a rewritten query."""
        # the original question never changes, only the search query does
        search_query = query

        for attempt in range(1, self.max_attempts + 1):
            logger.info("Search attempt %d: '%s'", attempt, search_query)

            matches = self.retriever.retrieve(search_query, top_k=top_k)
            chunks = self.reranker.rerank(query, matches, top_n=top_n)

            if self.evidence_checker.has_enough(query, chunks):
                logger.info("Enough evidence found")
                return self.generator.generate(query, chunks)

            if attempt == self.max_attempts:
                break

            logger.info("Not enough evidence, rewriting and widening the search")
            search_query = self.query_rewriter.rewrite(query)
            top_k += 4
            top_n += 2

        return NO_ANSWER
